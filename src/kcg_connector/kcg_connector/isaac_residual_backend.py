"""Reusable Isaac Sim backend for the connector residual-RL task.

This module deliberately has no top-level Isaac Sim or USD imports.  Isaac Sim
requires ``SimulationApp`` to exist before most of its Python modules are
imported, while the rest of :mod:`kcg_connector` must remain importable from a
normal ROS Python process.  Isaac bindings are therefore loaded lazily on the
first physical operation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import math
from types import SimpleNamespace
from typing import Any

import numpy as np

from kcg_connector.residual_rl import (
    ConnectorResidualState,
    calculate_residual_reward,
    decode_residual_action,
    evaluate_residual_state,
    loaded_torque_channels,
    residual_observation,
)
from kcg_connector.residual_randomization import (
    RANDOMIZATION_SCHEMA_VERSION,
    randomized_finger_torque_sample,
    randomized_residual_config,
    sample_connector_residual_randomization,
)


@lru_cache(maxsize=1)
def _isaac_bindings() -> SimpleNamespace:
    """Load Isaac/pxr symbols only after the caller started SimulationApp."""

    from isaacsim.core.utils.types import ArticulationAction
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

    return SimpleNamespace(
        ArticulationAction=ArticulationAction,
        Gf=Gf,
        PhysxSchema=PhysxSchema,
        Sdf=Sdf,
        Usd=Usd,
        UsdGeom=UsdGeom,
        UsdPhysics=UsdPhysics,
    )


@dataclass(frozen=True)
class RuntimeThreadSpec:
    """Immutable topology and geometry for the runtime thread proxy."""

    stage: Any
    body_path: str
    nut_path: str
    hinge_path: str
    runtime_root: str
    maximum_travel_m: float
    ratio_degrees_per_meter: float

    @property
    def prismatic_path(self) -> str:
        return f"{self.runtime_root}/InsertionPrismatic"

    @property
    def rack_path(self) -> str:
        return f"{self.runtime_root}/ThreadCoupling"


def create_runtime_thread(
    spec: RuntimeThreadSpec,
    body_position: Any,
    body_orientation: Any,
) -> tuple[Any, Any]:
    """Create the one physical prismatic+rack proxy used by all runners."""

    bindings = _isaac_bindings()
    bindings.UsdGeom.Scope.Define(spec.stage, spec.runtime_root)
    prismatic = bindings.UsdPhysics.PrismaticJoint.Define(
        spec.stage, spec.prismatic_path
    )
    prismatic.CreateAxisAttr("Z")
    prismatic.CreateBody1Rel().SetTargets(
        [bindings.Sdf.Path(spec.body_path)]
    )
    prismatic.CreateLocalPos0Attr(
        bindings.Gf.Vec3f(
            float(body_position[0]),
            float(body_position[1]),
            float(body_position[2]),
        )
    )
    orientation_imaginary = body_orientation.GetImaginary()
    prismatic.CreateLocalRot0Attr(
        bindings.Gf.Quatf(
            float(body_orientation.GetReal()),
            bindings.Gf.Vec3f(
                float(orientation_imaginary[0]),
                float(orientation_imaginary[1]),
                float(orientation_imaginary[2]),
            ),
        )
    )
    prismatic.CreateLocalPos1Attr(bindings.Gf.Vec3f(0.0))
    prismatic.CreateLocalRot1Attr(bindings.Gf.Quatf(1.0))
    prismatic.CreateLowerLimitAttr(-spec.maximum_travel_m)
    prismatic.CreateUpperLimitAttr(spec.maximum_travel_m)
    prismatic.CreateCollisionEnabledAttr(False)

    rack = bindings.PhysxSchema.PhysxPhysicsRackAndPinionJoint.Define(
        spec.stage, spec.rack_path
    )
    rack.CreateBody0Rel().SetTargets([bindings.Sdf.Path(spec.nut_path)])
    rack.CreateBody1Rel().SetTargets([bindings.Sdf.Path(spec.body_path)])
    rack.CreateHingeRel().SetTargets([bindings.Sdf.Path(spec.hinge_path)])
    rack.CreatePrismaticRel().SetTargets(
        [bindings.Sdf.Path(spec.prismatic_path)]
    )
    rack.CreateRatioAttr(spec.ratio_degrees_per_meter)
    return prismatic, rack


@dataclass(frozen=True)
class PreparedConnectorScene:
    """All handles and immutable values needed after physical engagement."""

    simulation_app: Any
    world: Any
    stage: Any
    robot: Any
    body: Any
    nut: Any
    grasp_tcp_prim: Any
    thread_spec: RuntimeThreadSpec
    controlled_indices: Any
    sensor_indices: Any
    q7_index: int
    q7_command_offset: int
    clamp_command_offsets: Any
    insertion_target: Any
    kps: Any
    kds: Any
    tare_efforts: Any
    dof_properties: Any
    checkpoint_positions: Any
    checkpoint_body_position: Any
    checkpoint_body_orientation: Any
    checkpoint_nut_position: Any
    checkpoint_nut_orientation: Any
    residual_config: Any
    resolved_curriculum_stage: Any
    settle_steps: int
    maximum_episode_steps: int
    render: bool
    physics_rate_hz: float = 240.0
    randomization_config: Any | None = None


@dataclass
class EpisodeSafetyStats:
    """Safety values accumulated through one backend reset/episode."""

    max_abs_velocity: float = 0.0
    max_abs_q7_velocity: float = 0.0
    max_limit_violation: float = 0.0
    max_finger_torque_delta: float = 0.0
    max_abs_nut_angular_velocity_policy_boundary: float = 0.0
    max_abs_q7_tracking_error_policy_boundary: float = 0.0
    max_grasp_translation_error_policy_boundary: float = 0.0
    max_grasp_rotation_error_policy_boundary: float = 0.0
    finite_throughout: bool = True
    physics_substep_samples: int = 0
    policy_boundary_samples: int = 0


_RAW_SAFETY_JOINT_LIMIT_TOLERANCE_RAD = 0.02

_RAW_SAFETY_PEAK_ATTRIBUTES = (
    (
        "physics_substep_max_abs_joint_velocity_rad_s",
        "max_abs_velocity",
    ),
    (
        "physics_substep_max_abs_q7_velocity_rad_s",
        "max_abs_q7_velocity",
    ),
    (
        "physics_substep_max_joint_limit_violation_rad",
        "max_limit_violation",
    ),
    (
        "physics_substep_max_abs_finger_base_torque_nm",
        "max_finger_torque_delta",
    ),
    (
        "policy_boundary_max_abs_nut_angular_velocity_rad_s",
        "max_abs_nut_angular_velocity_policy_boundary",
    ),
    (
        "policy_boundary_max_abs_q7_tracking_error_rad",
        "max_abs_q7_tracking_error_policy_boundary",
    ),
    (
        "policy_boundary_max_grasp_translation_error_m",
        "max_grasp_translation_error_policy_boundary",
    ),
    (
        "policy_boundary_max_grasp_rotation_error_rad",
        "max_grasp_rotation_error_policy_boundary",
    ),
)

_RAW_SAFETY_LIMIT_FAILURE_REASONS = {
    "physics_substep_max_abs_q7_velocity_rad_s": (
        "physics_substep_q7_speed_limit_exceeded"
    ),
    "physics_substep_max_joint_limit_violation_rad": (
        "physics_substep_joint_limit_tolerance_exceeded"
    ),
    "physics_substep_max_abs_finger_base_torque_nm": (
        "physics_substep_finger_base_torque_limit_exceeded"
    ),
    "policy_boundary_max_abs_nut_angular_velocity_rad_s": (
        "policy_boundary_nut_speed_limit_exceeded"
    ),
    "policy_boundary_max_abs_q7_tracking_error_rad": (
        "policy_boundary_q7_tracking_limit_exceeded"
    ),
    "policy_boundary_max_grasp_translation_error_m": (
        "policy_boundary_grasp_translation_limit_exceeded"
    ),
    "policy_boundary_max_grasp_rotation_error_rad": (
        "policy_boundary_grasp_rotation_limit_exceeded"
    ),
}


def _finite_nonnegative_float_or_none(value: Any) -> float | None:
    if isinstance(value, (bool, np.bool_)):
        return None
    if not isinstance(value, (int, float, np.integer, np.floating)):
        return None
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        return None
    return normalized


def _finite_positive_float_or_none(value: Any) -> float | None:
    normalized = _finite_nonnegative_float_or_none(value)
    if normalized is None or normalized <= 0.0:
        return None
    return normalized


def _accumulate_nonnegative_peak(previous: float, value: float) -> float:
    """Accumulate a peak while preserving non-finite evidence."""

    if not math.isfinite(previous) or not math.isfinite(value):
        return math.nan
    if previous < 0.0 or value < 0.0:
        return math.nan
    return max(previous, value)


def raw_episode_safety_report(
    safety: Any,
    config: Any,
    physics_rate_hz: Any,
) -> dict[str, Any]:
    """Build one cumulative, JSON-safe report from raw physics only.

    ``physics_substep`` fields are sampled once at episode start and after
    every 240 Hz physics step.  The
    ``policy_boundary`` fields are derived only from the un-noised 10 Hz raw
    state.  No termination label or policy observation enters this gate.
    """

    invalid_evidence = False
    peaks: dict[str, float | None] = {}
    for output_name, attribute_name in _RAW_SAFETY_PEAK_ATTRIBUTES:
        value = _finite_nonnegative_float_or_none(
            getattr(safety, attribute_name, None)
        )
        peaks[output_name] = value
        invalid_evidence = bool(invalid_evidence or value is None)

    normalized_counts: dict[str, int | None] = {}
    for count_name in (
        "physics_substep_samples",
        "policy_boundary_samples",
    ):
        count = getattr(safety, count_name, None)
        valid_count = bool(
            not isinstance(count, (bool, np.bool_))
            and isinstance(count, (int, np.integer))
            and int(count) > 0
        )
        normalized_counts[count_name] = (
            int(count) if valid_count else None
        )
        invalid_evidence = bool(invalid_evidence or not valid_count)

    finite_throughout = getattr(safety, "finite_throughout", None)
    valid_finite_marker = isinstance(finite_throughout, bool)
    invalid_evidence = bool(
        invalid_evidence or not valid_finite_marker
    )

    maximum_q7_speed = _finite_positive_float_or_none(
        getattr(config, "maximum_q7_speed_rad_s", None)
    )
    maximum_finger_torque = _finite_positive_float_or_none(
        getattr(config, "maximum_absolute_finger_torque_nm", None)
    )
    maximum_q7_tracking_error = _finite_positive_float_or_none(
        getattr(config, "maximum_q7_tracking_error_rad", None)
    )
    maximum_grasp_translation_error = _finite_positive_float_or_none(
        getattr(config, "maximum_grasp_translation_error_m", None)
    )
    maximum_grasp_rotation_error = _finite_positive_float_or_none(
        getattr(config, "maximum_grasp_rotation_error_rad", None)
    )
    normalized_physics_rate = _finite_positive_float_or_none(
        physics_rate_hz
    )
    normalized_policy_rate = _finite_positive_float_or_none(
        getattr(config, "policy_rate_hz", None)
    )
    configured_limits = (
        maximum_q7_speed,
        maximum_finger_torque,
        maximum_q7_tracking_error,
        maximum_grasp_translation_error,
        maximum_grasp_rotation_error,
        normalized_physics_rate,
        normalized_policy_rate,
    )
    invalid_evidence = bool(
        invalid_evidence
        or any(value is None for value in configured_limits)
    )

    limits = {
        "physics_substep_max_abs_q7_velocity_rad_s": (
            None
            if maximum_q7_speed is None
            else maximum_q7_speed * 1.10
        ),
        "physics_substep_max_joint_limit_violation_rad": (
            _RAW_SAFETY_JOINT_LIMIT_TOLERANCE_RAD
        ),
        "physics_substep_max_abs_finger_base_torque_nm": (
            maximum_finger_torque
        ),
        "policy_boundary_max_abs_nut_angular_velocity_rad_s": (
            None
            if maximum_q7_speed is None
            else maximum_q7_speed * 1.25
        ),
        "policy_boundary_max_abs_q7_tracking_error_rad": (
            maximum_q7_tracking_error
        ),
        "policy_boundary_max_grasp_translation_error_m": (
            maximum_grasp_translation_error
        ),
        "policy_boundary_max_grasp_rotation_error_rad": (
            maximum_grasp_rotation_error
        ),
    }
    if any(
        _finite_nonnegative_float_or_none(value) is None
        for value in limits.values()
    ):
        invalid_evidence = True

    failure_reasons = []
    if invalid_evidence:
        failure_reasons.append("raw_safety_evidence_invalid")
    if valid_finite_marker and not finite_throughout:
        failure_reasons.append("raw_physics_nonfinite")
    for name, reason in _RAW_SAFETY_LIMIT_FAILURE_REASONS.items():
        peak = peaks[name]
        limit = limits[name]
        if peak is not None and limit is not None and peak > limit:
            failure_reasons.append(reason)

    return {
        "passed": not failure_reasons,
        "failure_reasons": failure_reasons,
        "finite_throughout": (
            finite_throughout if valid_finite_marker else None
        ),
        "limits": {
            name: (
                None if value is None else float(value)
            )
            for name, value in limits.items()
        },
        "metrics": peaks,
        "sampling": {
            "physics_substep": {
                "includes_episode_initial_snapshot": True,
                "rate_hz": normalized_physics_rate,
                "samples": normalized_counts[
                    "physics_substep_samples"
                ],
            },
            "policy_boundary": {
                "includes_episode_initial_snapshot": True,
                "rate_hz": normalized_policy_rate,
                "samples": normalized_counts[
                    "policy_boundary_samples"
                ],
            },
        },
        "signal_source": "raw_physics",
    }


def _wrapped_relative_z_angle(
    bindings: SimpleNamespace,
    stage: Any,
    body_path: str,
    nut_path: str,
) -> float:
    body_matrix = bindings.UsdGeom.Xformable(
        stage.GetPrimAtPath(body_path)
    ).ComputeLocalToWorldTransform(bindings.Usd.TimeCode.Default())
    nut_matrix = bindings.UsdGeom.Xformable(
        stage.GetPrimAtPath(nut_path)
    ).ComputeLocalToWorldTransform(bindings.Usd.TimeCode.Default())
    body_quaternion = (
        bindings.Gf.Transform(body_matrix).GetRotation().GetQuat()
    )
    nut_quaternion = (
        bindings.Gf.Transform(nut_matrix).GetRotation().GetQuat()
    )
    relative = body_quaternion.GetInverse() * nut_quaternion
    imaginary = relative.GetImaginary()
    angle = 2.0 * math.atan2(
        float(imaginary[2]), float(relative.GetReal())
    )
    return math.atan2(math.sin(angle), math.cos(angle))


def _unwrap(previous: float, wrapped: float) -> float:
    previous_wrapped = math.atan2(math.sin(previous), math.cos(previous))
    delta = math.atan2(
        math.sin(wrapped - previous_wrapped),
        math.cos(wrapped - previous_wrapped),
    )
    return previous + delta


def _matrix_pose(bindings: SimpleNamespace, prim: Any) -> tuple[Any, Any]:
    matrix = bindings.UsdGeom.Xformable(
        prim
    ).ComputeLocalToWorldTransform(bindings.Usd.TimeCode.Default())
    transform = bindings.Gf.Transform(matrix)
    return transform.GetTranslation(), transform.GetRotation().GetQuat()


def _quaternion_rotation_vector(reference: Any, current: Any) -> np.ndarray:
    relative = reference.GetInverse() * current
    real = float(relative.GetReal())
    imaginary = np.asarray(relative.GetImaginary(), dtype=np.float64)
    if real < 0.0:
        real = -real
        imaginary = -imaginary
    imaginary_norm = float(np.linalg.norm(imaginary))
    if imaginary_norm <= 1.0e-12:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * math.atan2(
        imaginary_norm,
        max(0.0, min(1.0, real)),
    )
    return imaginary * (angle / imaginary_norm)


def _array_quaternion_error_radians(reference: Any, current: Any) -> float:
    reference_array = np.asarray(reference, dtype=np.float64).copy()
    current_array = np.asarray(current, dtype=np.float64).copy()
    reference_array /= np.linalg.norm(reference_array)
    current_array /= np.linalg.norm(current_array)
    cosine = float(
        np.clip(abs(np.dot(reference_array, current_array)), 0.0, 1.0)
    )
    return 2.0 * math.acos(cosine)


def _maximum_pose_difference_speed(
    positions_history: list[np.ndarray],
    orientations_history: list[np.ndarray],
    physics_rate_hz: float,
) -> tuple[float, float]:
    linear = max(
        float(np.linalg.norm(second - first))
        for first, second in zip(
            positions_history[:-1], positions_history[1:]
        )
    ) * physics_rate_hz
    angular = max(
        _array_quaternion_error_radians(first, second)
        for first, second in zip(
            orientations_history[:-1], orientations_history[1:]
        )
    ) * physics_rate_hz
    return linear, angular


def _maximum_finite_tail(values: list[float], count: int) -> float:
    """Return a tail peak, or infinity when the window is unusable."""

    if count <= 0 or not values:
        return math.inf
    tail = values[-min(count, len(values)):]
    if not all(math.isfinite(value) for value in tail):
        return math.inf
    return max(tail)


_RESET_DIAGNOSTIC_INTENDED_DISTANCE_KEYS = frozenset(
    {
        "body_intended_reset_distance_m",
        "nut_intended_reset_distance_m",
    }
)

_RESET_DIAGNOSTIC_COMPATIBILITY_ALIAS_KEYS = frozenset(
    {
        "solver_body_linear_speed_m_s",
        "solver_body_angular_speed_rad_s",
        "solver_nut_linear_speed_m_s",
        "solver_nut_angular_speed_rad_s",
        "solver_q7_speed_degrees_s",
    }
)

_RESET_DIAGNOSTIC_LIMITS = {
    "body_position_error_m": 0.00005,
    "nut_position_error_m": 0.00005,
    "body_orientation_error_degrees": 0.05,
    "nut_orientation_error_degrees": 0.05,
    "q7_checkpoint_error_degrees": 0.05,
    "preconstraint_recovery_body_displacement_m": 0.0001,
    "preconstraint_recovery_nut_displacement_m": 0.0001,
    "preconstraint_recovery_body_rotation_degrees": 0.1,
    "preconstraint_recovery_nut_rotation_degrees": 0.1,
    "first_step_body_jump_m": 0.0001,
    "first_step_nut_jump_m": 0.0001,
    "first_step_body_orientation_jump_degrees": 0.1,
    "first_step_nut_orientation_jump_degrees": 0.1,
    "first_step_q7_jump_degrees": 0.05,
    "settled_body_position_error_m": 0.0001,
    "settled_nut_position_error_m": 0.0001,
    "settled_body_orientation_error_degrees": 0.1,
    "settled_nut_orientation_error_degrees": 0.1,
    "settled_q7_error_degrees": 0.1,
    "settled_body_linear_speed_m_s": 0.001,
    "settled_nut_linear_speed_m_s": 0.001,
    "settled_body_angular_speed_rad_s": 0.02,
    "settled_nut_angular_speed_rad_s": 0.02,
    "settled_q7_speed_degrees_s": 0.5,
    "first_ten_peak_body_linear_speed_m_s": 0.025,
    "first_ten_peak_nut_linear_speed_m_s": 0.025,
    "first_ten_peak_body_angular_speed_rad_s": 0.25,
    "first_ten_peak_nut_angular_speed_rad_s": 0.25,
    "first_ten_peak_q7_speed_degrees_s": 5.0,
    "post_solver_tail_peak_body_linear_speed_m_s": 0.010,
    "post_solver_tail_peak_nut_linear_speed_m_s": 0.060,
    "post_solver_tail_peak_body_angular_speed_rad_s": 0.005,
    "post_solver_tail_peak_nut_angular_speed_rad_s": 0.25,
    "post_solver_tail_peak_q7_speed_degrees_s": 0.5,
}

_RESET_DIAGNOSTIC_REQUIRED_KEYS = frozenset(
    _RESET_DIAGNOSTIC_LIMITS
).union(
    _RESET_DIAGNOSTIC_INTENDED_DISTANCE_KEYS,
    _RESET_DIAGNOSTIC_COMPATIBILITY_ALIAS_KEYS,
)


def _is_finite_reset_diagnostic_scalar(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return False
    if not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        return False
    return math.isfinite(float(value))


def summarize_reset_diagnostics(
    diagnostics: list[dict[str, float]],
) -> tuple[dict[str, float], bool]:
    """Return reset maxima and a complete fail-closed reset gate."""

    if not diagnostics:
        return {}, False
    normalized_diagnostics = []
    for diagnostic in diagnostics:
        if not isinstance(diagnostic, dict):
            return {}, False
        if set(diagnostic) != _RESET_DIAGNOSTIC_REQUIRED_KEYS:
            return {}, False
        if not all(
            _is_finite_reset_diagnostic_scalar(value)
            for value in diagnostic.values()
        ):
            return {}, False
        normalized_diagnostics.append(
            {
                name: float(value)
                for name, value in diagnostic.items()
            }
        )
    maxima = {
        key: max(
            diagnostic[key] for diagnostic in normalized_diagnostics
        )
        for key in normalized_diagnostics[0]
        if key not in _RESET_DIAGNOSTIC_INTENDED_DISTANCE_KEYS
    }
    return maxima, all(
        maxima[name] <= limit
        for name, limit in _RESET_DIAGNOSTIC_LIMITS.items()
    )


class ConnectorResidualIsaacBackend:
    """One physical reset/step implementation shared by smoke and Gym."""

    def __init__(self, scene: PreparedConnectorScene):
        self.scene = scene
        self._bindings = _isaac_bindings()
        self._policy_dt = 1.0 / scene.residual_config.policy_rate_hz
        policy_substeps_exact = (
            scene.physics_rate_hz / scene.residual_config.policy_rate_hz
        )
        self._policy_substeps = int(round(policy_substeps_exact))
        if not math.isclose(
            policy_substeps_exact,
            self._policy_substeps,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("policy rate must divide the physics rate")

        self.current_command = np.asarray(
            scene.insertion_target, dtype=np.float32
        ).copy()
        self.unwrapped_nut_angle = 0.0
        self.start_nut_angle = 0.0
        self.start_body_z = 0.0
        self.start_q7 = 0.0
        self.reference_grasp_position = None
        self.reference_grasp_orientation = None
        self.previous_nut_progress = 0.0
        self.previous_axial_progress = 0.0
        self.previous_torques = np.zeros(3)
        self.previous_state: ConnectorResidualState | None = None
        self.stable_hold_seconds = 0.0
        self.policy_steps = 0
        self.holding = False
        self.initial_signature = None
        self.reset_count = 0
        self.reset_diagnostics: list[dict[str, float]] = []
        self.last_reset_diagnostic = None
        self.last_reset_seed: int | None = None
        self.last_reset_options: dict[str, Any] = {}
        self.physics_randomization_applied = False
        self.safety_signal_source = "raw_physics"
        self.randomization_enabled = bool(
            scene.randomization_config is not None
            and scene.randomization_config.enabled
        )
        self.randomization_schema_version = (
            RANDOMIZATION_SCHEMA_VERSION
            if scene.randomization_config is not None
            else None
        )
        if scene.randomization_config is not None:
            randomization = scene.randomization_config
            if (
                randomization.interface_version
                != scene.residual_config.interface_version
            ):
                raise ValueError(
                    "randomization and residual interface versions differ"
                )
            if tuple(randomization.clamp_joint_names) != tuple(
                scene.residual_config.clamp_joint_names
            ):
                raise ValueError(
                    "randomization and residual clamp joints differ"
                )
        seed_sequence = np.random.SeedSequence()
        domain_seed, observation_seed = seed_sequence.spawn(2)
        self._domain_rng = np.random.default_rng(domain_seed)
        self._observation_rng = np.random.default_rng(observation_seed)
        self.active_residual_config = scene.residual_config
        self.last_episode_randomization: dict[str, Any] | None = None
        self.episode_randomization_history: list[dict[str, Any]] = []
        self._episode_randomization = None
        self._previous_observed_torques: np.ndarray | None = None
        self.last_observed_state: ConnectorResidualState | None = None
        self._delayed_actions: list[np.ndarray] = []
        self.thread_proxy_rebuild_count = 0
        self.episode_safety = EpisodeSafetyStats()
        self._closed = False

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("connector residual backend is closed")

    @property
    def raw_safety_report(self) -> dict[str, Any]:
        """Return the current episode's cumulative raw-physics report."""

        return raw_episode_safety_report(
            self.episode_safety,
            self.active_residual_config,
            self.scene.physics_rate_hz,
        )

    def _raw_safety_info_fields(self) -> dict[str, Any]:
        report = self.raw_safety_report
        return {
            "raw_safety_passed": report["passed"],
            "raw_safety_failure_reasons": list(
                report["failure_reasons"]
            ),
            "raw_safety_peaks": dict(report["metrics"]),
        }

    def _observe(
        self, *, record_physics_safety: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        scene = self.scene
        positions = np.asarray(
            scene.robot.get_joint_positions(), dtype=np.float64
        )
        velocities = np.asarray(
            scene.robot.get_joint_velocities(), dtype=np.float64
        )
        efforts = np.asarray(
            scene.robot.get_measured_joint_efforts(
                joint_indices=scene.sensor_indices
            ),
            dtype=np.float64,
        )
        body_position, body_orientation = scene.body.get_world_pose()
        nut_position, nut_orientation = scene.nut.get_world_pose()
        if record_physics_safety:
            self._record_physics_safety_sample(
                positions,
                velocities,
                efforts,
                body_position,
                body_orientation,
                nut_position,
                nut_orientation,
            )
        return positions, velocities, efforts

    def _record_physics_safety_sample(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        efforts: np.ndarray,
        body_position: Any,
        body_orientation: Any,
        nut_position: Any,
        nut_orientation: Any,
    ) -> None:
        """Record one initial or post-step raw physics sample."""

        scene = self.scene
        finite_now = bool(
            np.all(np.isfinite(positions))
            and np.all(np.isfinite(velocities))
            and np.all(np.isfinite(efforts))
            and np.all(np.isfinite(body_position))
            and np.all(np.isfinite(body_orientation))
            and np.all(np.isfinite(nut_position))
            and np.all(np.isfinite(nut_orientation))
        )
        safety = self.episode_safety
        safety.physics_substep_samples += 1
        safety.finite_throughout = bool(
            safety.finite_throughout and finite_now
        )
        safety.max_abs_velocity = _accumulate_nonnegative_peak(
            safety.max_abs_velocity,
            float(np.max(np.abs(velocities))),
        )
        safety.max_abs_q7_velocity = _accumulate_nonnegative_peak(
            safety.max_abs_q7_velocity,
            abs(float(velocities[scene.q7_index])),
        )
        safety.max_finger_torque_delta = (
            _accumulate_nonnegative_peak(
                safety.max_finger_torque_delta,
                float(
                    np.max(
                        np.abs(
                            efforts
                            - np.asarray(
                                scene.tare_efforts, dtype=np.float64
                            )
                        )
                    )
                ),
            )
        )
        limit_violation = 0.0
        limit_inputs_finite = bool(np.all(np.isfinite(positions)))
        for index in range(scene.robot.num_dof):
            if bool(scene.dof_properties[index]["hasLimits"]):
                lower = float(scene.dof_properties[index]["lower"])
                upper = float(scene.dof_properties[index]["upper"])
                limit_inputs_finite = bool(
                    limit_inputs_finite
                    and math.isfinite(lower)
                    and math.isfinite(upper)
                )
                limit_violation = max(
                    limit_violation,
                    lower - float(positions[index]),
                    float(positions[index]) - upper,
                )
        safety.max_limit_violation = _accumulate_nonnegative_peak(
            safety.max_limit_violation,
            float(
                limit_violation if limit_inputs_finite else math.nan
            ),
        )

    def _record_policy_boundary_safety(
        self, state: ConnectorResidualState
    ) -> None:
        """Accumulate raw-state peaks available only at 10 Hz boundaries."""

        safety = self.episode_safety
        translation = np.asarray(
            state.grasp_translation_error_m, dtype=np.float64
        )
        rotation = np.asarray(
            state.grasp_rotation_error_rad, dtype=np.float64
        )
        values = (
            float(state.nut_angular_velocity_rad_s),
            float(state.q7_tracking_error_rad),
            *translation.tolist(),
            *rotation.tolist(),
        )
        finite_now = bool(
            translation.shape == (3,)
            and rotation.shape == (3,)
            and all(math.isfinite(value) for value in values)
        )
        safety.policy_boundary_samples += 1
        safety.finite_throughout = bool(
            safety.finite_throughout and finite_now
        )
        safety.max_abs_nut_angular_velocity_policy_boundary = (
            _accumulate_nonnegative_peak(
                safety.max_abs_nut_angular_velocity_policy_boundary,
                abs(float(state.nut_angular_velocity_rad_s)),
            )
        )
        safety.max_abs_q7_tracking_error_policy_boundary = (
            _accumulate_nonnegative_peak(
                safety.max_abs_q7_tracking_error_policy_boundary,
                abs(float(state.q7_tracking_error_rad)),
            )
        )
        safety.max_grasp_translation_error_policy_boundary = (
            _accumulate_nonnegative_peak(
                safety.max_grasp_translation_error_policy_boundary,
                float(np.linalg.norm(translation)),
            )
        )
        safety.max_grasp_rotation_error_policy_boundary = (
            _accumulate_nonnegative_peak(
                safety.max_grasp_rotation_error_policy_boundary,
                float(np.linalg.norm(rotation)),
            )
        )

    def _update_nut_angle(self) -> None:
        scene = self.scene
        wrapped = _wrapped_relative_z_angle(
            self._bindings,
            scene.stage,
            scene.thread_spec.body_path,
            scene.thread_spec.nut_path,
        )
        self.unwrapped_nut_angle = _unwrap(
            self.unwrapped_nut_angle, wrapped
        )

    def _tcp_nut_relative_pose(self) -> tuple[np.ndarray, Any]:
        scene = self.scene
        tcp_position, tcp_orientation = _matrix_pose(
            self._bindings, scene.grasp_tcp_prim
        )
        nut_position, nut_orientation = _matrix_pose(
            self._bindings,
            scene.stage.GetPrimAtPath(scene.thread_spec.nut_path),
        )
        world_offset = self._bindings.Gf.Vec3d(
            float(nut_position[0] - tcp_position[0]),
            float(nut_position[1] - tcp_position[1]),
            float(nut_position[2] - tcp_position[2]),
        )
        inverse_tcp = tcp_orientation.GetInverse()
        relative_position = inverse_tcp.Transform(world_offset)
        relative_orientation = inverse_tcp * nut_orientation
        return (
            np.asarray(relative_position, dtype=np.float64),
            relative_orientation,
        )

    def _state(
        self,
        commit_history: bool,
        *,
        record_initial_physics_safety: bool = False,
    ) -> ConnectorResidualState:
        scene = self.scene
        config = self.active_residual_config
        positions, velocities, efforts = self._observe(
            record_physics_safety=record_initial_physics_safety
        )
        body_position, _ = scene.body.get_world_pose()
        nut_progress = float(
            self.unwrapped_nut_angle - self.start_nut_angle
        )
        axial_progress = float(self.start_body_z - body_position[2])
        finger_torques = np.asarray(
            efforts - scene.tare_efforts, dtype=np.float64
        )
        torque_deltas = finger_torques - self.previous_torques
        relative_position, relative_orientation = (
            self._tcp_nut_relative_pose()
        )
        translation_error = (
            relative_position - self.reference_grasp_position
        )
        rotation_error = _quaternion_rotation_vector(
            self.reference_grasp_orientation,
            relative_orientation,
        )
        state = ConnectorResidualState(
            phase_progress=float(
                np.clip(
                    nut_progress / config.target_angle_rad,
                    0.0,
                    1.0,
                )
            ),
            q7_position_rad=float(positions[scene.q7_index]),
            q7_tracking_error_rad=float(
                positions[scene.q7_index]
                - self.current_command[scene.q7_command_offset]
            ),
            q7_velocity_rad_s=float(velocities[scene.q7_index]),
            nut_angle_rad=nut_progress,
            nut_angular_velocity_rad_s=float(
                (nut_progress - self.previous_nut_progress)
                / self._policy_dt
            ),
            axial_travel_m=axial_progress,
            axial_velocity_m_s=float(
                (axial_progress - self.previous_axial_progress)
                / self._policy_dt
            ),
            grasp_translation_error_m=tuple(
                float(value) for value in translation_error
            ),
            grasp_rotation_error_rad=tuple(
                float(value) for value in rotation_error
            ),
            finger_torques_nm=tuple(
                float(value) for value in finger_torques
            ),
            finger_torque_deltas_nm=tuple(
                float(value) for value in torque_deltas
            ),
            clamp_positions_rad=tuple(
                float(positions[index]) for index in scene.sensor_indices
            ),
            stable_hold_seconds=self.stable_hold_seconds,
        )
        self._record_policy_boundary_safety(state)
        if commit_history:
            self.previous_nut_progress = nut_progress
            self.previous_axial_progress = axial_progress
            self.previous_torques = finger_torques
        return state

    def _policy_observation(
        self,
        raw_state: ConnectorResidualState,
        *,
        episode_start: bool = False,
    ) -> np.ndarray:
        """Encode policy input while retaining untouched raw safety state."""

        episode = self._episode_randomization
        if episode is None or not self.randomization_enabled:
            self.last_observed_state = raw_state
            return residual_observation(
                raw_state, self.active_residual_config
            )
        observed_torques = randomized_finger_torque_sample(
            raw_state.finger_torques_nm,
            episode,
            self._observation_rng,
        )
        if episode_start or self._previous_observed_torques is None:
            observed_deltas = np.zeros(3, dtype=np.float64)
        else:
            observed_deltas = (
                observed_torques - self._previous_observed_torques
            )
        self._previous_observed_torques = observed_torques.copy()
        observed_state = replace(
            raw_state,
            finger_torques_nm=tuple(
                float(value) for value in observed_torques
            ),
            finger_torque_deltas_nm=tuple(
                float(value) for value in observed_deltas
            ),
        )
        self.last_observed_state = observed_state
        return residual_observation(
            observed_state, self.active_residual_config
        )

    def _select_episode_randomization(self) -> None:
        """Sample and install control/observation parameters for one reset."""

        scene = self.scene
        distribution = scene.randomization_config
        self._previous_observed_torques = None
        self.last_observed_state = None
        if distribution is None:
            self._episode_randomization = None
            self.active_residual_config = scene.residual_config
            self.last_episode_randomization = None
            self._delayed_actions = []
            return
        episode = sample_connector_residual_randomization(
            distribution, self._domain_rng
        )
        self._episode_randomization = episode
        self.active_residual_config = randomized_residual_config(
            scene.residual_config, episode
        )
        report = episode.as_dict()
        report.update(
            {
                "episode": self.reset_count + 1,
                "seed": self.last_reset_seed,
            }
        )
        self.last_episode_randomization = report
        self.episode_randomization_history.append(dict(report))
        self._delayed_actions = [
            np.zeros(4, dtype=np.float32)
            for _ in range(episode.action_delay_policy_steps)
        ]

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        self._require_open()
        if seed is not None:
            valid_seed_type = isinstance(seed, (int, np.integer))
            if isinstance(seed, bool) or not valid_seed_type:
                raise TypeError("reset seed must be a nonnegative integer")
            if int(seed) < 0:
                raise ValueError("reset seed must be a nonnegative integer")
            seed = int(seed)
            seed_sequence = np.random.SeedSequence(seed)
            domain_seed, observation_seed = seed_sequence.spawn(2)
            self._domain_rng = np.random.default_rng(domain_seed)
            self._observation_rng = np.random.default_rng(
                observation_seed
            )
        if options is None:
            reset_options: dict[str, Any] = {}
        elif not isinstance(options, dict):
            raise TypeError("reset options must be a dict or None")
        else:
            reset_options = dict(options)
        if reset_options:
            raise ValueError(
                "connector residual v0 does not support reset options"
            )
        self.last_reset_seed = seed
        self.last_reset_options = reset_options
        self.physics_randomization_applied = False
        scene = self.scene
        self._select_episode_randomization()
        config = self.active_residual_config
        spec = scene.thread_spec
        bindings = self._bindings
        self.reset_count += 1
        self.episode_safety = EpisodeSafetyStats()
        self.current_command = np.asarray(
            scene.insertion_target, dtype=np.float32
        ).copy()
        self.current_command[scene.clamp_command_offsets] = np.asarray(
            config.clamp_nominal_positions_rad, dtype=np.float32
        )
        before_body_position, _ = scene.body.get_world_pose()
        before_nut_position, _ = scene.nut.get_world_pose()
        scene.world.pause()
        if not scene.stage.RemovePrim(spec.rack_path):
            raise RuntimeError("failed to remove reset rack proxy")
        if not scene.stage.RemovePrim(spec.prismatic_path):
            raise RuntimeError("failed to remove reset prismatic proxy")
        scene.simulation_app.update()
        scene.world.reset(soft=False)
        immediate_body_position, immediate_body_orientation = (
            scene.body.get_world_pose()
        )
        immediate_nut_position, immediate_nut_orientation = (
            scene.nut.get_world_pose()
        )
        immediate_positions = np.asarray(
            scene.robot.get_joint_positions(), dtype=np.float64
        )
        reset_diagnostic = {
            "body_intended_reset_distance_m": float(
                np.linalg.norm(
                    np.asarray(before_body_position, dtype=np.float64)
                    - scene.checkpoint_body_position
                )
            ),
            "body_position_error_m": float(
                np.linalg.norm(
                    np.asarray(immediate_body_position, dtype=np.float64)
                    - scene.checkpoint_body_position
                )
            ),
            "body_orientation_error_degrees": math.degrees(
                _array_quaternion_error_radians(
                    scene.checkpoint_body_orientation,
                    immediate_body_orientation,
                )
            ),
            "nut_intended_reset_distance_m": float(
                np.linalg.norm(
                    np.asarray(before_nut_position, dtype=np.float64)
                    - scene.checkpoint_nut_position
                )
            ),
            "nut_position_error_m": float(
                np.linalg.norm(
                    np.asarray(immediate_nut_position, dtype=np.float64)
                    - scene.checkpoint_nut_position
                )
            ),
            "nut_orientation_error_degrees": math.degrees(
                _array_quaternion_error_radians(
                    scene.checkpoint_nut_orientation,
                    immediate_nut_orientation,
                )
            ),
            "q7_checkpoint_error_degrees": math.degrees(
                abs(
                    float(immediate_positions[scene.q7_index])
                    - float(
                        scene.checkpoint_positions[scene.q7_index]
                    )
                )
            ),
        }
        self.last_reset_diagnostic = reset_diagnostic
        self.reset_diagnostics.append(reset_diagnostic)
        reset_controller = scene.robot.get_articulation_controller()
        episode_kps = np.asarray(scene.kps, dtype=np.float32).copy()
        episode_kds = np.asarray(scene.kds, dtype=np.float32).copy()
        if self._episode_randomization is not None:
            episode_kps[scene.sensor_indices] *= (
                self._episode_randomization.hand_kp_scale
            )
            episode_kds[scene.sensor_indices] *= (
                self._episode_randomization.hand_kd_scale
            )
        reset_controller.set_gains(
            kps=episode_kps,
            kds=episode_kds,
            save_to_usd=False,
        )
        scene.world.get_physics_context().set_gravity(-9.81)
        for _ in range(scene.settle_steps):
            scene.robot.apply_action(
                bindings.ArticulationAction(
                    joint_positions=self.current_command,
                    joint_indices=scene.controlled_indices,
                )
            )
            scene.world.step(render=scene.render)
            self._observe()

        (
            constraint_reference_body_position,
            constraint_reference_body_orientation,
        ) = scene.body.get_world_pose()
        (
            constraint_reference_nut_position,
            constraint_reference_nut_orientation,
        ) = scene.nut.get_world_pose()
        constraint_reference_positions = np.asarray(
            scene.robot.get_joint_positions(), dtype=np.float64
        )
        scene.world.pause()
        reference_orientation_imaginary = bindings.Gf.Vec3d(
            float(constraint_reference_body_orientation[1]),
            float(constraint_reference_body_orientation[2]),
            float(constraint_reference_body_orientation[3]),
        )
        create_runtime_thread(
            spec,
            constraint_reference_body_position,
            bindings.Gf.Quatd(
                float(constraint_reference_body_orientation[0]),
                reference_orientation_imaginary,
            ),
        )
        self.thread_proxy_rebuild_count += 1
        scene.world.play()
        scene.simulation_app.update()
        reset_diagnostic.update(
            {
                "preconstraint_recovery_body_displacement_m": float(
                    np.linalg.norm(
                        np.asarray(
                            constraint_reference_body_position,
                            dtype=np.float64,
                        )
                        - np.asarray(
                            immediate_body_position, dtype=np.float64
                        )
                    )
                ),
                "preconstraint_recovery_body_rotation_degrees": (
                    math.degrees(
                        _array_quaternion_error_radians(
                            immediate_body_orientation,
                            constraint_reference_body_orientation,
                        )
                    )
                ),
                "preconstraint_recovery_nut_displacement_m": float(
                    np.linalg.norm(
                        np.asarray(
                            constraint_reference_nut_position,
                            dtype=np.float64,
                        )
                        - np.asarray(
                            immediate_nut_position, dtype=np.float64
                        )
                    )
                ),
                "preconstraint_recovery_nut_rotation_degrees": (
                    math.degrees(
                        _array_quaternion_error_radians(
                            immediate_nut_orientation,
                            constraint_reference_nut_orientation,
                        )
                    )
                ),
            }
        )

        first_step_body_position = None
        first_step_body_orientation = None
        first_step_nut_position = None
        first_step_nut_orientation = None
        first_step_positions = None
        settle_body_positions = [
            np.asarray(
                constraint_reference_body_position, dtype=np.float64
            )
        ]
        settle_body_orientations = [
            np.asarray(
                constraint_reference_body_orientation, dtype=np.float64
            )
        ]
        settle_nut_positions = [
            np.asarray(
                constraint_reference_nut_position, dtype=np.float64
            )
        ]
        settle_nut_orientations = [
            np.asarray(
                constraint_reference_nut_orientation, dtype=np.float64
            )
        ]
        settle_q7_positions = [
            float(constraint_reference_positions[scene.q7_index])
        ]
        post_solver_body_linear_speeds = []
        post_solver_body_angular_speeds = []
        post_solver_nut_linear_speeds = []
        post_solver_nut_angular_speeds = []
        post_solver_q7_speeds = []
        for settle_index in range(scene.settle_steps):
            scene.robot.apply_action(
                bindings.ArticulationAction(
                    joint_positions=self.current_command,
                    joint_indices=scene.controlled_indices,
                )
            )
            scene.world.step(render=scene.render)
            positions, velocities, _ = self._observe()
            step_body_position, step_body_orientation = (
                scene.body.get_world_pose()
            )
            step_nut_position, step_nut_orientation = (
                scene.nut.get_world_pose()
            )
            settle_body_positions.append(
                np.asarray(step_body_position, dtype=np.float64)
            )
            settle_body_orientations.append(
                np.asarray(step_body_orientation, dtype=np.float64)
            )
            settle_nut_positions.append(
                np.asarray(step_nut_position, dtype=np.float64)
            )
            settle_nut_orientations.append(
                np.asarray(step_nut_orientation, dtype=np.float64)
            )
            settle_q7_positions.append(float(positions[scene.q7_index]))
            post_solver_body_linear_speeds.append(
                float(np.linalg.norm(scene.body.get_linear_velocity()))
            )
            post_solver_body_angular_speeds.append(
                float(np.linalg.norm(scene.body.get_angular_velocity()))
            )
            post_solver_nut_linear_speeds.append(
                float(np.linalg.norm(scene.nut.get_linear_velocity()))
            )
            post_solver_nut_angular_speeds.append(
                float(np.linalg.norm(scene.nut.get_angular_velocity()))
            )
            post_solver_q7_speeds.append(
                abs(float(velocities[scene.q7_index]))
            )
            if settle_index == 0:
                first_step_body_position = step_body_position
                first_step_body_orientation = step_body_orientation
                first_step_nut_position = step_nut_position
                first_step_nut_orientation = step_nut_orientation
                first_step_positions = positions.copy()

        settled_body_position, settled_body_orientation = (
            scene.body.get_world_pose()
        )
        settled_nut_position, settled_nut_orientation = (
            scene.nut.get_world_pose()
        )
        settled_positions = np.asarray(
            scene.robot.get_joint_positions(), dtype=np.float64
        )
        settled_velocities = np.asarray(
            scene.robot.get_joint_velocities(), dtype=np.float64
        )
        first_sample_count = min(11, len(settle_body_positions))
        (
            first_ten_peak_body_linear_speed,
            first_ten_peak_body_angular_speed,
        ) = _maximum_pose_difference_speed(
            settle_body_positions[:first_sample_count],
            settle_body_orientations[:first_sample_count],
            scene.physics_rate_hz,
        )
        (
            first_ten_peak_nut_linear_speed,
            first_ten_peak_nut_angular_speed,
        ) = _maximum_pose_difference_speed(
            settle_nut_positions[:first_sample_count],
            settle_nut_orientations[:first_sample_count],
            scene.physics_rate_hz,
        )
        first_ten_peak_q7_speed = max(
            abs(second - first)
            for first, second in zip(
                settle_q7_positions[: first_sample_count - 1],
                settle_q7_positions[1:first_sample_count],
            )
        ) * scene.physics_rate_hz
        tail_sample_count = min(11, len(settle_body_positions))
        (
            settled_body_linear_speed,
            settled_body_angular_speed,
        ) = _maximum_pose_difference_speed(
            settle_body_positions[-tail_sample_count:],
            settle_body_orientations[-tail_sample_count:],
            scene.physics_rate_hz,
        )
        (
            settled_nut_linear_speed,
            settled_nut_angular_speed,
        ) = _maximum_pose_difference_speed(
            settle_nut_positions[-tail_sample_count:],
            settle_nut_orientations[-tail_sample_count:],
            scene.physics_rate_hz,
        )
        settled_q7_speed = max(
            abs(second - first)
            for first, second in zip(
                settle_q7_positions[-tail_sample_count:-1],
                settle_q7_positions[-tail_sample_count + 1:],
            )
        ) * scene.physics_rate_hz
        post_solver_tail_peak_body_linear_speed = _maximum_finite_tail(
            post_solver_body_linear_speeds, 10
        )
        post_solver_tail_peak_body_angular_speed = _maximum_finite_tail(
            post_solver_body_angular_speeds, 10
        )
        post_solver_tail_peak_nut_linear_speed = _maximum_finite_tail(
            post_solver_nut_linear_speeds, 10
        )
        post_solver_tail_peak_nut_angular_speed = _maximum_finite_tail(
            post_solver_nut_angular_speeds, 10
        )
        post_solver_tail_peak_q7_speed = _maximum_finite_tail(
            post_solver_q7_speeds, 10
        )
        reset_diagnostic.update(
            {
                "first_step_body_jump_m": float(
                    np.linalg.norm(
                        np.asarray(
                            first_step_body_position, dtype=np.float64
                        )
                        - np.asarray(
                            constraint_reference_body_position,
                            dtype=np.float64,
                        )
                    )
                ),
                "first_step_body_orientation_jump_degrees": math.degrees(
                    _array_quaternion_error_radians(
                        constraint_reference_body_orientation,
                        first_step_body_orientation,
                    )
                ),
                "first_step_nut_jump_m": float(
                    np.linalg.norm(
                        np.asarray(
                            first_step_nut_position, dtype=np.float64
                        )
                        - np.asarray(
                            constraint_reference_nut_position,
                            dtype=np.float64,
                        )
                    )
                ),
                "first_step_nut_orientation_jump_degrees": math.degrees(
                    _array_quaternion_error_radians(
                        constraint_reference_nut_orientation,
                        first_step_nut_orientation,
                    )
                ),
                "first_step_q7_jump_degrees": math.degrees(
                    abs(
                        float(first_step_positions[scene.q7_index])
                        - float(
                            constraint_reference_positions[
                                scene.q7_index
                            ]
                        )
                    )
                ),
                "first_ten_peak_body_linear_speed_m_s": (
                    first_ten_peak_body_linear_speed
                ),
                "first_ten_peak_body_angular_speed_rad_s": (
                    first_ten_peak_body_angular_speed
                ),
                "first_ten_peak_nut_linear_speed_m_s": (
                    first_ten_peak_nut_linear_speed
                ),
                "first_ten_peak_nut_angular_speed_rad_s": (
                    first_ten_peak_nut_angular_speed
                ),
                "first_ten_peak_q7_speed_degrees_s": math.degrees(
                    first_ten_peak_q7_speed
                ),
                "settled_body_position_error_m": float(
                    np.linalg.norm(
                        np.asarray(
                            settled_body_position, dtype=np.float64
                        )
                        - scene.checkpoint_body_position
                    )
                ),
                "settled_body_orientation_error_degrees": math.degrees(
                    _array_quaternion_error_radians(
                        scene.checkpoint_body_orientation,
                        settled_body_orientation,
                    )
                ),
                "settled_nut_position_error_m": float(
                    np.linalg.norm(
                        np.asarray(
                            settled_nut_position, dtype=np.float64
                        )
                        - scene.checkpoint_nut_position
                    )
                ),
                "settled_nut_orientation_error_degrees": math.degrees(
                    _array_quaternion_error_radians(
                        scene.checkpoint_nut_orientation,
                        settled_nut_orientation,
                    )
                ),
                "settled_q7_error_degrees": math.degrees(
                    abs(
                        float(settled_positions[scene.q7_index])
                        - float(
                            scene.checkpoint_positions[scene.q7_index]
                        )
                    )
                ),
                "settled_body_linear_speed_m_s": (
                    settled_body_linear_speed
                ),
                "settled_body_angular_speed_rad_s": (
                    settled_body_angular_speed
                ),
                "settled_nut_linear_speed_m_s": settled_nut_linear_speed,
                "settled_nut_angular_speed_rad_s": settled_nut_angular_speed,
                "settled_q7_speed_degrees_s": math.degrees(
                    settled_q7_speed
                ),
                "post_solver_tail_peak_body_linear_speed_m_s": (
                    post_solver_tail_peak_body_linear_speed
                ),
                "post_solver_tail_peak_body_angular_speed_rad_s": (
                    post_solver_tail_peak_body_angular_speed
                ),
                "post_solver_tail_peak_nut_linear_speed_m_s": (
                    post_solver_tail_peak_nut_linear_speed
                ),
                "post_solver_tail_peak_nut_angular_speed_rad_s": (
                    post_solver_tail_peak_nut_angular_speed
                ),
                "post_solver_tail_peak_q7_speed_degrees_s": (
                    math.degrees(post_solver_tail_peak_q7_speed)
                ),
                # Compatibility aliases preserve the previous single-sample
                # fields.  They are post-solver state, not pose-snap speed,
                # and the strict gate above uses the explicit tail peaks.
                "solver_body_linear_speed_m_s": float(
                    np.linalg.norm(scene.body.get_linear_velocity())
                ),
                "solver_body_angular_speed_rad_s": float(
                    np.linalg.norm(scene.body.get_angular_velocity())
                ),
                "solver_nut_linear_speed_m_s": float(
                    np.linalg.norm(scene.nut.get_linear_velocity())
                ),
                "solver_nut_angular_speed_rad_s": float(
                    np.linalg.norm(scene.nut.get_angular_velocity())
                ),
                "solver_q7_speed_degrees_s": math.degrees(
                    abs(float(settled_velocities[scene.q7_index]))
                ),
            }
        )

        wrapped = _wrapped_relative_z_angle(
            bindings,
            scene.stage,
            spec.body_path,
            spec.nut_path,
        )
        self.unwrapped_nut_angle = wrapped
        self.start_nut_angle = wrapped
        body_position, _ = scene.body.get_world_pose()
        nut_position, _ = scene.nut.get_world_pose()
        positions = np.asarray(
            scene.robot.get_joint_positions(), dtype=np.float64
        )
        self.start_body_z = float(body_position[2])
        self.start_q7 = float(positions[scene.q7_index])
        (
            self.reference_grasp_position,
            self.reference_grasp_orientation,
        ) = self._tcp_nut_relative_pose()
        efforts = np.asarray(
            scene.robot.get_measured_joint_efforts(
                joint_indices=scene.sensor_indices
            ),
            dtype=np.float64,
        )
        self.previous_torques = efforts - scene.tare_efforts
        self.previous_nut_progress = 0.0
        self.previous_axial_progress = 0.0
        self.stable_hold_seconds = 0.0
        self.policy_steps = 0
        self.holding = False
        self.initial_signature = {
            "body_position": np.asarray(
                body_position, dtype=np.float64
            ),
            "nut_position": np.asarray(nut_position, dtype=np.float64),
            "q7": self.start_q7,
        }
        state = self._state(
            commit_history=True,
            record_initial_physics_safety=True,
        )
        self.previous_state = state
        observation = self._policy_observation(
            state, episode_start=True
        )
        return observation, {
            "control_observation_randomization_applied": (
                self.randomization_enabled
            ),
            "episode_randomization": self.last_episode_randomization,
            "loaded_channels": loaded_torque_channels(
                state, self.active_residual_config
            ),
            "physics_randomization_applied": (
                self.physics_randomization_applied
            ),
            "randomization_enabled": self.randomization_enabled,
            "randomization_schema_version": (
                self.randomization_schema_version
            ),
            "reset": (
                "engaged_seeded_control_observation_v1_hard_reset"
                if self.randomization_enabled
                else "engaged_default_state_hard_reset"
            ),
            "reset_checkpoint": reset_diagnostic,
            "reset_options": self.last_reset_options,
            "safety_signal_source": "raw_physics",
            "seed": self.last_reset_seed,
            "thread_proxy_reset": (
                "remove_hard_reset_contact_recovery_recreate"
            ),
            **self._raw_safety_info_fields(),
        }

    def step(
        self, action: Any
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._require_open()
        scene = self.scene
        config = self.active_residual_config
        if self.previous_state is None:
            raise RuntimeError("reset must be called before step")
        requested = decode_residual_action(action, config)
        requested_action = np.asarray(
            requested.normalized, dtype=np.float32
        )
        if self._delayed_actions:
            self._delayed_actions.append(requested_action.copy())
            applied_action = self._delayed_actions.pop(0)
        else:
            applied_action = requested_action
        decoded = decode_residual_action(applied_action, config)
        start_command = self.current_command.copy()
        end_command = self.current_command.copy()
        end_command[scene.clamp_command_offsets] = np.asarray(
            decoded.clamp_position_targets_rad, dtype=np.float32
        )
        if not self.holding:
            measured_remaining = max(
                0.0,
                config.target_angle_rad
                - self.previous_state.nut_angle_rad,
            )
            tightening_increment = min(
                abs(decoded.q7_velocity_target_rad_s) * self._policy_dt,
                measured_remaining,
            )
            end_command[scene.q7_command_offset] += (
                config.tightening_direction * tightening_increment
            )
        for substep in range(self._policy_substeps):
            blend = float(substep + 1) / float(self._policy_substeps)
            target = start_command + blend * (end_command - start_command)
            scene.robot.apply_action(
                self._bindings.ArticulationAction(
                    joint_positions=target.astype(np.float32),
                    joint_indices=scene.controlled_indices,
                )
            )
            scene.world.step(render=scene.render)
            self._observe()
            self._update_nut_angle()
        self.current_command = end_command
        self.policy_steps += 1
        current = self._state(commit_history=False)
        preliminary = evaluate_residual_state(current, config)
        if preliminary.reason == "holding":
            self.holding = True
            stable_velocity = bool(
                abs(current.q7_velocity_rad_s)
                <= config.hold_q7_velocity_tolerance_rad_s
                and abs(current.nut_angular_velocity_rad_s)
                <= config.hold_nut_velocity_tolerance_rad_s
                and abs(current.axial_velocity_m_s)
                <= config.hold_axial_velocity_tolerance_m_s
            )
            if stable_velocity:
                self.stable_hold_seconds += self._policy_dt
            else:
                self.stable_hold_seconds = 0.0
        elif not preliminary.terminated:
            self.stable_hold_seconds = 0.0
        current = replace(
            current, stable_hold_seconds=self.stable_hold_seconds
        )
        termination = evaluate_residual_state(current, config)
        reward = calculate_residual_reward(
            self.previous_state,
            current,
            applied_action,
            config,
        )
        self.previous_nut_progress = current.nut_angle_rad
        self.previous_axial_progress = current.axial_travel_m
        self.previous_torques = np.asarray(
            current.finger_torques_nm, dtype=np.float64
        )
        self.previous_state = current
        truncated = bool(
            not termination.terminated
            and self.policy_steps >= scene.maximum_episode_steps
        )
        info = {
            "action_delay_policy_steps": (
                0
                if self._episode_randomization is None
                else self._episode_randomization.action_delay_policy_steps
            ),
            "applied_action": [
                float(value) for value in applied_action
            ],
            "control_observation_randomization_applied": (
                self.randomization_enabled
            ),
            "requested_action": [
                float(value) for value in requested_action
            ],
            "reward_terms": reward.terms,
            "safety_signal_source": "raw_physics",
            "termination_reason": termination.reason,
            **self._raw_safety_info_fields(),
        }
        return (
            self._policy_observation(current),
            reward.total,
            termination.terminated,
            truncated,
            info,
        )

    def close(self) -> None:
        """Close this borrower without closing its externally owned app."""

        self._closed = True
