"""Strict pure-Python runtime for an opt-in virtual wrist-wrench monitor.

The helper deliberately contains no Isaac imports.  A caller reads the
``hand2arm`` reaction row, supplies the sensor pose, and receives task-frame
observation evidence.  It cannot authorize E2E success while operational
limits and same-scene baseline provenance remain unset.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


SCHEMA_VERSION = "kcg_d38999_wrist_ft_monitor_v1"
DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "config/d38999_wrist_ft_monitor_v1.yaml"
)
WRENCH_ORDER = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")
HOME_TARE_PHASE = "HOME_FREE_SPACE_EMPTY_HAND"
PAYLOAD_CAPTURE_PHASE = "POST_GRASP_FREE_SPACE"
PROTECTED_PHASES = ("INSERT", "ENGAGE", "SCREW", "HOLD")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _exact(value: Mapping[str, Any], expected, label: str) -> None:
    actual = set(value)
    required = set(expected)
    if actual != required:
        raise ValueError(
            f"{label} keys are invalid: missing={sorted(required - actual)}, "
            f"extra={sorted(actual - required)}"
        )


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be non-empty text")
    return value


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
    ):
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _optional_number(value: Any, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be null or numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be null or finite and positive")
    return result


def _texts(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty list")
    result = tuple(_text(item, f"{label}[]") for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{label} must not contain duplicates")
    return result


def _vector(values: Sequence[float], size: int, label: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(
            f"{label} must be a finite vector with shape ({size},)"
        )
    return result


def _matrix(
    values: Sequence[Sequence[float]], size: int, label: str
) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (size, size) or not np.all(np.isfinite(result)):
        raise ValueError(
            f"{label} must be a finite matrix with shape ({size}, {size})"
        )
    return result


@dataclass(frozen=True)
class VirtualWristFtMonitorConfig:
    """Validated immutable monitor configuration."""

    schema_version: str
    enabled: bool
    status: str
    wrist_design_path: str
    wrist_design_sha256: str
    robot_asset_path: str
    robot_asset_sha256: str
    measurement_joint: str
    metadata_joint_index_offset: int
    raw_frame: str
    canonical_frame: str
    raw_semantics: str
    canonical_semantics: str
    canonical_from_raw: tuple[tuple[float, ...], ...]
    task_frame_id: str
    task_origin_source: str
    task_z_axis_source: str
    task_x_reference_world: tuple[float, float, float]
    physics_rate_hz: int
    home_tare_window_steps: int
    payload_baseline_window_steps: int
    minimum_capture_samples: int
    electronic_bias_tare_allowed: tuple[str, ...]
    payload_capture_allowed: tuple[str, ...]
    tare_forbidden: tuple[str, ...]
    monitored_contact_phases: tuple[str, ...]
    threshold_status: str
    threshold_source_artifact: str | None
    threshold_source_artifact_sha256: str | None
    threshold_repeat_count: int
    safety_limits: tuple[float | None, ...]
    monitor_only: bool


def _parse_input(document: Mapping[str, Any], label: str) -> tuple[str, str]:
    _exact(document, {"path", "sha256"}, label)
    path = _text(document["path"], f"{label}.path")
    digest = _text(document["sha256"], f"{label}.sha256")
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{label}.sha256 must be lowercase SHA-256")
    return path, digest


def load_virtual_wrist_ft_monitor_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> VirtualWristFtMonitorConfig:
    """Load the strict observation-only contract and reject overclaiming."""

    config_path = Path(path).expanduser().resolve()
    root = _mapping(
        yaml.safe_load(config_path.read_text(encoding="utf-8")), "root"
    )
    _exact(
        root,
        {
            "schema_version",
            "enabled",
            "status",
            "compatibility",
            "inputs",
            "source",
            "task_frame",
            "sampling",
            "phase_policy",
            "compensation",
            "threshold_calibration",
            "safety_limits",
            "boundaries",
        },
        "root",
    )
    if _text(root["schema_version"], "schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected wrist FT monitor schema_version")

    compatibility = _mapping(root["compatibility"], "compatibility")
    _exact(
        compatibility,
        {
            "active_interface_version",
            "modifies_active_interface",
            "modifies_robot_asset",
            "modifies_default_e2e",
            "modifies_e2e_pass_gate",
        },
        "compatibility",
    )
    if (
        compatibility["active_interface_version"]
        != "kcg_connector_twist_residual_v0"
    ):
        raise ValueError("monitor must remain bound to residual_v0")
    for key in compatibility:
        if key != "active_interface_version" and _boolean(
            compatibility[key], f"compatibility.{key}"
        ):
            raise ValueError(f"compatibility.{key} must remain false")

    inputs = _mapping(root["inputs"], "inputs")
    _exact(inputs, {"wrist_ft_design", "robot_asset"}, "inputs")
    wrist_path, wrist_sha = _parse_input(
        _mapping(inputs["wrist_ft_design"], "inputs.wrist_ft_design"),
        "inputs.wrist_ft_design",
    )
    robot_path, robot_sha = _parse_input(
        _mapping(inputs["robot_asset"], "inputs.robot_asset"),
        "inputs.robot_asset",
    )

    source = _mapping(root["source"], "source")
    _exact(
        source,
        {
            "measurement_joint",
            "metadata_joint_index_offset",
            "raw_frame",
            "canonical_frame",
            "raw_semantics",
            "canonical_semantics",
            "wrench_order",
            "canonical_from_raw",
        },
        "source",
    )
    mapping = _matrix(
        source["canonical_from_raw"], 6, "source.canonical_from_raw"
    )
    if not np.array_equal(mapping, -np.eye(6, dtype=np.float64)):
        raise ValueError("source.canonical_from_raw must be exactly -I6")
    if tuple(source["wrench_order"]) != WRENCH_ORDER:
        raise ValueError("source.wrench_order must be Fx,Fy,Fz,Tx,Ty,Tz")

    task = _mapping(root["task_frame"], "task_frame")
    _exact(
        task,
        {
            "frame_id",
            "origin_source",
            "z_axis_source",
            "x_reference_world",
            "tightening_resistance_sign",
        },
        "task_frame",
    )
    if task["tightening_resistance_sign"] != "positive_about_task_z":
        raise ValueError(
            "tightening resistance must be positive about task z"
        )

    sampling = _mapping(root["sampling"], "sampling")
    _exact(
        sampling,
        {
            "physics_rate_hz",
            "home_tare_window_steps",
            "payload_baseline_window_steps",
            "minimum_capture_samples",
        },
        "sampling",
    )
    rate = _integer(
        sampling["physics_rate_hz"],
        "sampling.physics_rate_hz",
        minimum=1,
    )
    home_window = _integer(
        sampling["home_tare_window_steps"],
        "sampling.home_tare_window_steps",
        minimum=1,
    )
    payload_window = _integer(
        sampling["payload_baseline_window_steps"],
        "sampling.payload_baseline_window_steps",
        minimum=1,
    )
    minimum_samples = _integer(
        sampling["minimum_capture_samples"],
        "sampling.minimum_capture_samples",
        minimum=1,
    )
    if rate != 240 or minimum_samples > min(home_window, payload_window):
        raise ValueError(
            "sampling must remain 240 Hz with feasible capture windows"
        )

    policy = _mapping(root["phase_policy"], "phase_policy")
    _exact(
        policy,
        {
            "electronic_bias_tare_allowed",
            "payload_capture_allowed",
            "tare_forbidden",
            "monitored_contact_phases",
        },
        "phase_policy",
    )
    home_allowed = _texts(
        policy["electronic_bias_tare_allowed"],
        "phase_policy.electronic_bias_tare_allowed",
    )
    payload_allowed = _texts(
        policy["payload_capture_allowed"],
        "phase_policy.payload_capture_allowed",
    )
    forbidden = _texts(
        policy["tare_forbidden"], "phase_policy.tare_forbidden"
    )
    monitored = _texts(
        policy["monitored_contact_phases"],
        "phase_policy.monitored_contact_phases",
    )
    if home_allowed != (HOME_TARE_PHASE,) or payload_allowed != (
        PAYLOAD_CAPTURE_PHASE,
    ):
        raise ValueError("tare capture phases changed")
    if forbidden != PROTECTED_PHASES or monitored != PROTECTED_PHASES:
        raise ValueError(
            "contact-phase policy must be INSERT,ENGAGE,SCREW,HOLD"
        )

    compensation = _mapping(root["compensation"], "compensation")
    _exact(
        compensation,
        {
            "mode",
            "subtract_home_empty_baseline_for_payload_estimate",
            "subtract_captured_payload_baseline_from_contact_samples",
            "uses_current_sensor_orientation",
            "shifts_moment_to_engagement_datum",
            "dynamic_inertia_compensation_complete",
            "orientation_dependent_gravity_compensation_complete",
        },
        "compensation",
    )
    if compensation["mode"] != "captured_payload_quasistatic_baseline":
        raise ValueError(
            "only captured quasistatic payload compensation is supported"
        )
    for key in (
        "subtract_home_empty_baseline_for_payload_estimate",
        "subtract_captured_payload_baseline_from_contact_samples",
        "uses_current_sensor_orientation",
        "shifts_moment_to_engagement_datum",
    ):
        if not _boolean(compensation[key], f"compensation.{key}"):
            raise ValueError(f"compensation.{key} must be true")
    for key in (
        "dynamic_inertia_compensation_complete",
        "orientation_dependent_gravity_compensation_complete",
    ):
        if _boolean(compensation[key], f"compensation.{key}"):
            raise ValueError(f"compensation.{key} must remain false")

    calibration = _mapping(
        root["threshold_calibration"], "threshold_calibration"
    )
    _exact(
        calibration,
        {
            "status",
            "source_artifact",
            "source_artifact_sha256",
            "repeat_count",
            "statistic",
            "margin_policy",
        },
        "threshold_calibration",
    )
    threshold_status = _text(
        calibration["status"], "threshold_calibration.status"
    )
    source_artifact = calibration["source_artifact"]
    source_digest = calibration["source_artifact_sha256"]
    repeat_count = _integer(
        calibration["repeat_count"], "threshold_calibration.repeat_count"
    )
    for key in (
        "source_artifact",
        "source_artifact_sha256",
        "statistic",
        "margin_policy",
    ):
        if calibration[key] is not None:
            raise ValueError(
                f"threshold_calibration.{key} must remain null while pending"
            )
    if (
        threshold_status != "pending_same_scene_physical_baseline_scan"
        or repeat_count != 0
    ):
        raise ValueError(
            "threshold calibration must remain pending with zero runs"
        )

    limits_doc = _mapping(root["safety_limits"], "safety_limits")
    limit_keys = (
        "maximum_lateral_force_n",
        "maximum_axial_force_n",
        "maximum_bending_torque_nm",
        "maximum_tightening_torque_nm",
        "stale_timeout_s",
    )
    _exact(limits_doc, set(limit_keys), "safety_limits")
    limits = tuple(
        _optional_number(limits_doc[key], f"safety_limits.{key}")
        for key in limit_keys
    )
    if any(value is not None for value in limits):
        raise ValueError(
            "safety limits require completed same-scene calibration"
        )

    boundaries = _mapping(root["boundaries"], "boundaries")
    _exact(
        boundaries,
        {
            "monitor_only",
            "safety_gate_claimed",
            "dynamic_compensation_claimed",
            "calibrated_thresholds_claimed",
            "residual_v1_enabled",
            "assembly_success_claimed_from_wrench",
        },
        "boundaries",
    )
    if not _boolean(boundaries["monitor_only"], "boundaries.monitor_only"):
        raise ValueError("monitor_only must remain true")
    for key in boundaries:
        if key != "monitor_only" and _boolean(
            boundaries[key], f"boundaries.{key}"
        ):
            raise ValueError(f"boundaries.{key} must remain false")

    return VirtualWristFtMonitorConfig(
        schema_version=SCHEMA_VERSION,
        enabled=_boolean(root["enabled"], "enabled"),
        status=_text(root["status"], "status"),
        wrist_design_path=wrist_path,
        wrist_design_sha256=wrist_sha,
        robot_asset_path=robot_path,
        robot_asset_sha256=robot_sha,
        measurement_joint=_text(
            source["measurement_joint"], "source.measurement_joint"
        ),
        metadata_joint_index_offset=_integer(
            source["metadata_joint_index_offset"],
            "source.metadata_joint_index_offset",
            minimum=1,
        ),
        raw_frame=_text(source["raw_frame"], "source.raw_frame"),
        canonical_frame=_text(
            source["canonical_frame"], "source.canonical_frame"
        ),
        raw_semantics=_text(
            source["raw_semantics"], "source.raw_semantics"
        ),
        canonical_semantics=_text(
            source["canonical_semantics"], "source.canonical_semantics"
        ),
        canonical_from_raw=tuple(
            tuple(float(value) for value in row) for row in mapping
        ),
        task_frame_id=_text(task["frame_id"], "task_frame.frame_id"),
        task_origin_source=_text(
            task["origin_source"], "task_frame.origin_source"
        ),
        task_z_axis_source=_text(
            task["z_axis_source"], "task_frame.z_axis_source"
        ),
        task_x_reference_world=tuple(
            float(value)
            for value in _vector(
                task["x_reference_world"],
                3,
                "task_frame.x_reference_world",
            )
        ),
        physics_rate_hz=rate,
        home_tare_window_steps=home_window,
        payload_baseline_window_steps=payload_window,
        minimum_capture_samples=minimum_samples,
        electronic_bias_tare_allowed=home_allowed,
        payload_capture_allowed=payload_allowed,
        tare_forbidden=forbidden,
        monitored_contact_phases=monitored,
        threshold_status=threshold_status,
        threshold_source_artifact=source_artifact,
        threshold_source_artifact_sha256=source_digest,
        threshold_repeat_count=repeat_count,
        safety_limits=limits,
        monitor_only=True,
    )


def verify_virtual_wrist_ft_monitor_inputs(
    config: VirtualWristFtMonitorConfig, repository: str | Path
) -> dict[str, Path]:
    """Verify both immutable inputs without importing Isaac or editing them."""

    root = Path(repository).expanduser().resolve()
    resolved = {}
    for name, relative, expected in (
        (
            "wrist_ft_design",
            config.wrist_design_path,
            config.wrist_design_sha256,
        ),
        ("robot_asset", config.robot_asset_path, config.robot_asset_sha256),
    ):
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(
                f"{name} must remain repository-relative"
            ) from error
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"{name} SHA-256 mismatch")
        resolved[name] = path
    return resolved


def reaction_row_index(
    joint_indices: Mapping[str, int], config: VirtualWristFtMonitorConfig
) -> int:
    """Resolve Isaac's fixed-joint reaction row (metadata index plus one)."""

    if config.measurement_joint not in joint_indices:
        raise ValueError(
            f"{config.measurement_joint} is absent from articulation metadata"
        )
    index = joint_indices[config.measurement_joint]
    if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
        raise ValueError("measurement joint metadata index must be integral")
    result = int(index) + config.metadata_joint_index_offset
    if result <= int(index):
        raise ValueError("reaction row offset must be positive")
    return result


def classify_e2e_wrist_ft_phase(runtime_phase: str) -> str:
    """Map detailed E2E phase labels onto the strict tare/contact policy."""

    phase = _text(runtime_phase, "runtime_phase")
    if phase == "initial_settle":
        return HOME_TARE_PHASE
    if phase == "unsupported_final_hold":
        return PAYLOAD_CAPTURE_PHASE
    if (
        phase.startswith("mixed_grip_physical_insert")
        or phase.startswith("mixed_grip_preinsert")
        or phase.startswith("contact_response_")
    ):
        return "INSERT"
    if phase == "engaged_keying_proxy_activation" or phase.startswith(
        "end_to_end_release_mixed_grip"
    ):
        return "ENGAGE"
    if phase.startswith("end_to_end_rotation_") and phase.endswith("_motion"):
        return "SCREW"
    if (
        phase.startswith("end_to_end_rotation_") and phase.endswith("_hold")
    ) or phase == "end_to_end_nut_only_hold":
        return "HOLD"
    return "OTHER"


def task_rotation_world_from_axis(
    z_axis_world: Sequence[float], x_reference_world: Sequence[float]
) -> np.ndarray:
    """Return columns [x,y,z] for a deterministic right-handed task frame."""

    z_axis = _vector(z_axis_world, 3, "z_axis_world")
    z_axis /= np.linalg.norm(z_axis)
    reference = _vector(x_reference_world, 3, "x_reference_world")
    projected = reference - np.dot(reference, z_axis) * z_axis
    if np.linalg.norm(projected) <= 1.0e-9:
        raise ValueError("x reference is parallel to task z axis")
    x_axis = projected / np.linalg.norm(projected)
    y_axis = np.cross(z_axis, x_axis)
    return np.column_stack((x_axis, y_axis, z_axis))


def column_rotation_from_gf_matrix3d(
    gf_rotation_rows: Sequence[Sequence[float]],
) -> np.ndarray:
    """Convert a ``Gf.Matrix3d(rotation)`` to column-vector convention.

    USD/Gf matrices transform row vectors (``local * matrix``), whereas the
    wrench math below uses NumPy column vectors (``matrix @ local``).  The
    transpose therefore preserves the same local-to-world rotation.  Keeping
    this adapter pure Python avoids importing pxr outside the opt-in Isaac
    boundary and makes the convention directly unit-testable.
    """

    rotation_rows = _matrix(gf_rotation_rows, 3, "gf_rotation_rows")
    rotation_columns = rotation_rows.T.copy()
    if not np.allclose(
        rotation_columns.T @ rotation_columns,
        np.eye(3, dtype=np.float64),
        rtol=0.0,
        atol=1.0e-9,
    ) or not math.isclose(
        float(np.linalg.det(rotation_columns)),
        1.0,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise ValueError("Gf rotation matrix must be right-handed orthonormal")
    return rotation_columns


def transform_wrench_between_frames(
    wrench_source: Sequence[float],
    rotation_target_from_source: Sequence[Sequence[float]],
    position_target_source: Sequence[float],
) -> np.ndarray:
    """Transform one wrench from source origin/frame to target origin/frame.

    ``position_target_source`` is :math:`p^A_{AS}`: the vector from the
    target/assembly origin to the source/sensor origin, expressed in the
    target frame.  With ``T_A_S`` denoting source-to-target coordinates, the
    implemented convention is::

        F_A = R_A_S F_S
        M_A = R_A_S M_S + p_A_S x F_A

    Keeping the direction of this lever arm in the public API prevents the
    easy-to-miss sign reversal that occurs when ``sensor-to-target`` is used
    in prose but ``target-to-sensor`` is used in code.
    """

    wrench = _vector(wrench_source, 6, "wrench_source")
    rotation = _matrix(
        rotation_target_from_source,
        3,
        "rotation_target_from_source",
    )
    position = _vector(
        position_target_source,
        3,
        "position_target_source",
    )
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-9):
        raise ValueError("rotation_target_from_source must be orthonormal")
    if not math.isclose(
        float(np.linalg.det(rotation)), 1.0, rel_tol=0.0, abs_tol=1.0e-9
    ):
        raise ValueError("rotation_target_from_source must be right-handed")
    force_target = rotation @ wrench[:3]
    moment_target = rotation @ wrench[3:] + np.cross(
        position, force_target
    )
    return np.concatenate((force_target, moment_target))


def inverse_wrench_transform(
    wrench_target: Sequence[float],
    rotation_target_from_source: Sequence[Sequence[float]],
    position_target_source: Sequence[float],
) -> np.ndarray:
    """Invert :func:`transform_wrench_between_frames` exactly."""

    rotation = _matrix(
        rotation_target_from_source,
        3,
        "rotation_target_from_source",
    )
    position = _vector(
        position_target_source,
        3,
        "position_target_source",
    )
    # T_S_A has R_S_A = R_A_S.T and p_S_A = -R_S_A p_A_S.
    return transform_wrench_between_frames(
        wrench_target,
        rotation.T,
        -(rotation.T @ position),
    )


def transform_wrench_to_task(
    wrench_sensor: Sequence[float],
    sensor_position_world: Sequence[float],
    sensor_rotation_world: Sequence[Sequence[float]],
    task_origin_world: Sequence[float],
    task_rotation_world: Sequence[Sequence[float]],
) -> np.ndarray:
    """Rotate a wrench and shift its moment from sensor to task origin."""

    wrench = _vector(wrench_sensor, 6, "wrench_sensor")
    sensor_position = _vector(
        sensor_position_world, 3, "sensor_position_world"
    )
    task_origin = _vector(task_origin_world, 3, "task_origin_world")
    sensor_rotation = _matrix(
        sensor_rotation_world, 3, "sensor_rotation_world"
    )
    task_rotation = _matrix(task_rotation_world, 3, "task_rotation_world")
    world_to_task = task_rotation.T
    rotation_task_from_sensor = world_to_task @ sensor_rotation
    position_task_sensor = world_to_task @ (
        sensor_position - task_origin
    )
    return transform_wrench_between_frames(
        wrench,
        rotation_task_from_sensor,
        position_task_sensor,
    )


class VirtualWristFtMonitor:
    """Online baseline capture and compact protected-phase peak evidence."""

    def __init__(
        self,
        config: VirtualWristFtMonitorConfig,
        *,
        reaction_row: int,
        task_origin_world: Sequence[float],
        task_z_axis_world: Sequence[float],
    ) -> None:
        if not config.enabled:
            raise ValueError("virtual wrist FT monitor config is disabled")
        self.config = config
        self.reaction_row = int(reaction_row)
        if self.reaction_row < 1:
            raise ValueError("reaction_row must be positive")
        self.task_origin_world = _vector(
            task_origin_world, 3, "task_origin_world"
        )
        self.task_rotation_world = task_rotation_world_from_axis(
            task_z_axis_world, config.task_x_reference_world
        )
        maximum_window = max(
            config.home_tare_window_steps,
            config.payload_baseline_window_steps,
        )
        self._capture_samples = defaultdict(
            lambda: deque(maxlen=maximum_window)
        )
        self._phase_counts = defaultdict(int)
        self._phase_peaks: dict[
            str, dict[str, dict[str, Any]]
        ] = defaultdict(dict)
        self._home_baseline = None
        self._payload_baseline = None
        self._last_global_step = -1
        self._last_sample = None

    def _record(self, values, *, global_step, runtime_phase, policy_phase):
        return {
            "global_step": int(global_step),
            "timestamp_s": float(global_step) / self.config.physics_rate_hz,
            "runtime_phase": runtime_phase,
            "policy_phase": policy_phase,
            "source_frame": self.config.raw_frame,
            "target_frame": self.config.task_frame_id,
            "wrench": [float(value) for value in values],
        }

    def observe(
        self,
        raw_wrench: Sequence[float],
        *,
        global_step: int,
        runtime_phase: str,
        sensor_position_world: Sequence[float],
        sensor_rotation_world: Sequence[Sequence[float]],
    ) -> dict[str, Any]:
        """Consume one reaction sample with timestamp and frame IDs."""

        if isinstance(global_step, bool) or int(global_step) != global_step:
            raise ValueError("global_step must be integral")
        global_step = int(global_step)
        if global_step <= self._last_global_step:
            raise ValueError("wrist FT timestamps must be strictly increasing")
        policy_phase = classify_e2e_wrist_ft_phase(runtime_phase)
        raw = _vector(raw_wrench, 6, "raw_wrench")
        canonical = np.asarray(self.config.canonical_from_raw) @ raw
        sample = self._record(
            canonical,
            global_step=global_step,
            runtime_phase=runtime_phase,
            policy_phase=policy_phase,
        )
        sample["raw_wrench"] = [float(value) for value in raw]
        sample["canonical_wrench_sensor"] = sample.pop("wrench")
        self._last_global_step = global_step
        self._last_sample = sample
        self._phase_counts[policy_phase] += 1

        if policy_phase in (HOME_TARE_PHASE, PAYLOAD_CAPTURE_PHASE):
            self._capture_samples[policy_phase].append(canonical.copy())

        if policy_phase in self.config.monitored_contact_phases:
            if self._payload_baseline is None:
                raise RuntimeError(
                    "protected contact wrench observed before payload baseline"
                )
            compensated_sensor = canonical - self._payload_baseline
            task_wrench = transform_wrench_to_task(
                compensated_sensor,
                sensor_position_world,
                sensor_rotation_world,
                self.task_origin_world,
                self.task_rotation_world,
            )
            sample["compensated_wrench_sensor"] = [
                float(value) for value in compensated_sensor
            ]
            sample["compensated_wrench_task"] = [
                float(value) for value in task_wrench
            ]
            scalar_values = {
                "lateral_force_n": float(np.linalg.norm(task_wrench[:2])),
                "axial_force_n": float(task_wrench[2]),
                "bending_torque_nm": float(np.linalg.norm(task_wrench[3:5])),
                "tightening_torque_nm": float(task_wrench[5]),
            }
            sample["task_scalars"] = scalar_values
            for name, signed_value in scalar_values.items():
                magnitude = abs(signed_value)
                current = self._phase_peaks[policy_phase].get(name)
                if current is None or magnitude > current["absolute_peak"]:
                    self._phase_peaks[policy_phase][name] = {
                        "absolute_peak": magnitude,
                        "signed_value_at_peak": signed_value,
                        "sample": dict(sample),
                    }
        return dict(sample)

    def _capture(self, phase: str, window: int) -> np.ndarray:
        if phase in self.config.tare_forbidden:
            raise RuntimeError(f"tare is forbidden during {phase}")
        samples = list(self._capture_samples[phase])[-window:]
        if len(samples) < self.config.minimum_capture_samples:
            raise RuntimeError(
                f"{phase} has {len(samples)} samples; "
                f"requires {self.config.minimum_capture_samples}"
            )
        return np.mean(np.stack(samples), axis=0)

    def capture_home_tare(self) -> list[float]:
        """Capture empty-hand home baseline; no contact phase is accepted."""

        self._home_baseline = self._capture(
            HOME_TARE_PHASE, self.config.home_tare_window_steps
        )
        return [float(value) for value in self._home_baseline]

    def capture_payload_baseline(self) -> list[float]:
        """Capture grasped payload in free space before insertion begins."""

        if self._home_baseline is None:
            raise RuntimeError(
                "home empty-hand tare must precede payload capture"
            )
        self._payload_baseline = self._capture(
            PAYLOAD_CAPTURE_PHASE,
            self.config.payload_baseline_window_steps,
        )
        return [float(value) for value in self._payload_baseline]

    def report(self) -> dict[str, Any]:
        """Return compact JSON-safe evidence without inventing safety gates."""

        payload_estimate = None
        if (
            self._home_baseline is not None
            and self._payload_baseline is not None
        ):
            payload_estimate = self._payload_baseline - self._home_baseline
        return {
            "schema_version": self.config.schema_version,
            "status": "MONITOR_ONLY",
            "measurement_joint": self.config.measurement_joint,
            "reaction_row_index": self.reaction_row,
            "metadata_joint_index_offset": (
                self.config.metadata_joint_index_offset
            ),
            "raw_frame": self.config.raw_frame,
            "task_frame": self.config.task_frame_id,
            "canonical_from_raw": [
                list(row) for row in self.config.canonical_from_raw
            ],
            "home_empty_baseline_canonical": (
                None
                if self._home_baseline is None
                else [float(value) for value in self._home_baseline]
            ),
            "payload_baseline_canonical": (
                None
                if self._payload_baseline is None
                else [float(value) for value in self._payload_baseline]
            ),
            "payload_increment_estimate_canonical": (
                None
                if payload_estimate is None
                else [float(value) for value in payload_estimate]
            ),
            "phase_sample_counts": dict(sorted(self._phase_counts.items())),
            "protected_phase_peaks": {
                phase: dict(sorted(values.items()))
                for phase, values in sorted(self._phase_peaks.items())
            },
            "last_sample": self._last_sample,
            "compensation_mode": "captured_payload_quasistatic_baseline",
            "dynamic_inertia_compensation_complete": False,
            "orientation_dependent_gravity_compensation_complete": False,
            "same_scene_threshold_calibration_status": (
                self.config.threshold_status
            ),
            "calibrated_safety_limits": None,
            "monitor_only": True,
            "modifies_e2e_pass_gate": False,
            "residual_v1_enabled": False,
            "safety_gate_claimed": False,
            "assembly_success_claimed_from_wrench": False,
        }


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "HOME_TARE_PHASE",
    "PAYLOAD_CAPTURE_PHASE",
    "PROTECTED_PHASES",
    "SCHEMA_VERSION",
    "VirtualWristFtMonitor",
    "VirtualWristFtMonitorConfig",
    "classify_e2e_wrist_ft_phase",
    "column_rotation_from_gf_matrix3d",
    "load_virtual_wrist_ft_monitor_config",
    "reaction_row_index",
    "task_rotation_world_from_axis",
    "transform_wrench_between_frames",
    "inverse_wrench_transform",
    "transform_wrench_to_task",
    "verify_virtual_wrist_ft_monitor_inputs",
]
