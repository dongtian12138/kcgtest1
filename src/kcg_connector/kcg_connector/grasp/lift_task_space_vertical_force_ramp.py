"""B-V2-H18 bounded world-up force ramp for physical table breakaway."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .pre_lift_arm_drive_compliance import minimum_jerk_blend


THRESHOLD_LABEL = "SIM_TUNING_ONLY_B_V2_H18"
NUMERIC_JACOBIAN_STEP_RAD = 1.0e-6
CONFIG_KEYS = (
    "enabled",
    "threshold_label",
    "source_h12_run_id",
    "source_h13_run_id",
    "source_h17_run_id",
    "force_target_source",
    "ramp_steps_source",
    "world_force_axis",
    "keep_position_target_fixed_during_ramp",
    "immediately_preceding_arm_state_only",
    "recompute_mapping_during_staged_lift",
)


@dataclass(frozen=True)
class LiftTaskSpaceVerticalForceRampConfig:
    """Strict semantic authorization; numeric values come from frozen inputs."""

    enabled: bool
    threshold_label: str
    source_h12_run_id: str
    source_h13_run_id: str
    source_h17_run_id: str
    force_target_source: str
    ramp_steps_source: str
    world_force_axis: tuple[float, float, float]
    keep_position_target_fixed_during_ramp: bool
    immediately_preceding_arm_state_only: bool
    recompute_mapping_during_staged_lift: bool

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("H18 enabled must be boolean")
        if self.threshold_label != THRESHOLD_LABEL:
            raise ValueError(f"H18 must stay {THRESHOLD_LABEL}")
        expected_text = {
            "source_h12_run_id": "B-V2-H12-ZERO-LIFT-01",
            "source_h13_run_id": "B-V2-GRASP-14",
            "source_h17_run_id": "B-V2-H17-ABSOLUTE-LOAD-HOLD-01",
            "force_target_source": "FROZEN_BODY_PLUS_NUT_WEIGHT",
            "ramp_steps_source": "H13_TRANSITION_STEPS",
        }
        for name, expected in expected_text.items():
            if getattr(self, name) != expected:
                raise ValueError(f"H18 {name} must remain {expected}")
        if tuple(self.world_force_axis) != (0.0, 0.0, 1.0):
            raise ValueError("H18 force axis must remain world +Z")
        required_true = (
            "keep_position_target_fixed_during_ramp",
            "immediately_preceding_arm_state_only",
            "recompute_mapping_during_staged_lift",
        )
        for name in required_true:
            if getattr(self, name) is not True:
                raise ValueError(f"H18 {name} must remain true")


def load_lift_task_space_vertical_force_ramp_config(
    value: Any,
) -> LiftTaskSpaceVerticalForceRampConfig:
    """Load H18 strictly while historical contracts default disabled."""

    defaults = {
        "enabled": False,
        "threshold_label": THRESHOLD_LABEL,
        "source_h12_run_id": "B-V2-H12-ZERO-LIFT-01",
        "source_h13_run_id": "B-V2-GRASP-14",
        "source_h17_run_id": "B-V2-H17-ABSOLUTE-LOAD-HOLD-01",
        "force_target_source": "FROZEN_BODY_PLUS_NUT_WEIGHT",
        "ramp_steps_source": "H13_TRANSITION_STEPS",
        "world_force_axis": (0.0, 0.0, 1.0),
        "keep_position_target_fixed_during_ramp": True,
        "immediately_preceding_arm_state_only": True,
        "recompute_mapping_during_staged_lift": True,
    }
    if value is None:
        return LiftTaskSpaceVerticalForceRampConfig(**defaults)
    if not isinstance(value, Mapping):
        raise ValueError("lift_task_space_vertical_force_ramp must be a mapping")
    unknown = sorted(set(value) - set(CONFIG_KEYS))
    missing = sorted(set(CONFIG_KEYS) - set(value))
    if unknown or missing:
        raise ValueError(
            "lift_task_space_vertical_force_ramp has unknown keys "
            f"{unknown} and/or missing keys {missing}"
        )
    axis = value["world_force_axis"]
    if not isinstance(axis, (list, tuple)) or len(axis) != 3:
        raise ValueError("H18 world_force_axis must contain three values")
    return LiftTaskSpaceVerticalForceRampConfig(
        enabled=value["enabled"],
        threshold_label=str(value["threshold_label"]),
        source_h12_run_id=str(value["source_h12_run_id"]),
        source_h13_run_id=str(value["source_h13_run_id"]),
        source_h17_run_id=str(value["source_h17_run_id"]),
        force_target_source=str(value["force_target_source"]),
        ramp_steps_source=str(value["ramp_steps_source"]),
        world_force_axis=tuple(float(entry) for entry in axis),
        keep_position_target_fixed_during_ramp=(
            value["keep_position_target_fixed_during_ramp"]
        ),
        immediately_preceding_arm_state_only=(
            value["immediately_preceding_arm_state_only"]
        ),
        recompute_mapping_during_staged_lift=(
            value["recompute_mapping_during_staged_lift"]
        ),
    )


def _seven(values: Sequence[Real], label: str) -> np.ndarray:
    result = np.asarray(tuple(values), dtype=np.float64)
    if result.shape != (7,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must contain seven finite values")
    return result


def _positive(value: Real, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be positive and finite")
    return result


def numeric_tcp_translation_jacobian(
    arm_rad: Sequence[Real],
    forward_kinematics: Callable[[tuple[float, ...]], Sequence[Sequence[Real]]],
) -> np.ndarray:
    """Return the 3x7 world translation Jacobian from the shared pure FK."""

    arm = _seven(arm_rad, "arm_rad")
    jacobian = np.zeros((3, 7), dtype=np.float64)
    for index in range(7):
        plus = arm.copy()
        minus = arm.copy()
        plus[index] += NUMERIC_JACOBIAN_STEP_RAD
        minus[index] -= NUMERIC_JACOBIAN_STEP_RAD
        plus_transform = np.asarray(
            forward_kinematics(tuple(float(value) for value in plus)),
            dtype=np.float64,
        )
        minus_transform = np.asarray(
            forward_kinematics(tuple(float(value) for value in minus)),
            dtype=np.float64,
        )
        if (
            plus_transform.shape != (4, 4)
            or minus_transform.shape != (4, 4)
            or not np.all(np.isfinite(plus_transform))
            or not np.all(np.isfinite(minus_transform))
        ):
            raise ValueError("H18 FK must return finite 4x4 transforms")
        jacobian[:, index] = (
            plus_transform[:3, 3] - minus_transform[:3, 3]
        ) / (2.0 * NUMERIC_JACOBIAN_STEP_RAD)
    if not np.all(np.isfinite(jacobian)):
        raise ValueError("H18 translation Jacobian is non-finite")
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    if singular_values.shape != (3,) or singular_values[-1] <= 1.0e-9:
        raise ValueError("H18 translation Jacobian is rank deficient")
    return jacobian


def frozen_payload_weight_force_n(
    body_mass_kg: Real,
    nut_mass_kg: Real,
    gravity_m_s2: Real,
    sensor_origin_force_gate_n: Real,
) -> float:
    """Derive the one H18 force target from frozen masses and gravity."""

    body_mass = _positive(body_mass_kg, "body_mass_kg")
    nut_mass = _positive(nut_mass_kg, "nut_mass_kg")
    gravity = float(gravity_m_s2)
    if not math.isfinite(gravity) or gravity >= 0.0:
        raise ValueError("gravity_m_s2 must be finite and world-down")
    force_gate = _positive(
        sensor_origin_force_gate_n, "sensor_origin_force_gate_n"
    )
    target = (body_mass + nut_mass) * abs(gravity)
    if target >= force_gate:
        raise ValueError("H18 frozen payload weight must remain below the force gate")
    return target


def derive_vertical_force_step(
    arm_rad: Sequence[Real],
    arm_velocity_rad_s: Sequence[Real],
    body_mass_kg: Real,
    nut_mass_kg: Real,
    gravity_m_s2: Real,
    sensor_origin_force_gate_n: Real,
    profile_fraction: Real,
    previous_world_up_force_n: Real,
    previous_joint_effort_nm: Sequence[Real],
    actuator_effort_limits_nm: Sequence[Real],
    forward_kinematics: Callable[[tuple[float, ...]], Sequence[Sequence[Real]]],
) -> dict[str, Any]:
    """Map one bounded minimum-jerk world-up force sample to joint effort."""

    arm = _seven(arm_rad, "arm_rad")
    velocity = _seven(arm_velocity_rad_s, "arm_velocity_rad_s")
    previous_effort = _seven(
        previous_joint_effort_nm, "previous_joint_effort_nm"
    )
    limits = _seven(actuator_effort_limits_nm, "actuator_effort_limits_nm")
    if np.any(limits <= 0.0):
        raise ValueError("H18 actuator effort limits must be positive")
    fraction = float(profile_fraction)
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("H18 profile_fraction must be in [0, 1]")
    previous_force = float(previous_world_up_force_n)
    if not math.isfinite(previous_force) or previous_force < 0.0:
        raise ValueError("H18 previous force must be finite and non-negative")
    target_force = frozen_payload_weight_force_n(
        body_mass_kg,
        nut_mass_kg,
        gravity_m_s2,
        sensor_origin_force_gate_n,
    )
    blend = minimum_jerk_blend(fraction)
    world_up_force = target_force * blend
    if world_up_force + 1.0e-12 < previous_force:
        raise ValueError("H18 force profile must be monotonic")
    force_world = np.asarray((0.0, 0.0, world_up_force), dtype=np.float64)
    jacobian = numeric_tcp_translation_jacobian(arm, forward_kinematics)
    joint_effort = jacobian.T @ force_world
    if np.any(np.abs(joint_effort) > limits + 1.0e-9):
        raise ValueError("H18 mapped joint effort exceeds actuator readback")
    effort_delta = joint_effort - previous_effort
    task_velocity = jacobian @ velocity
    joint_power = float(np.dot(joint_effort, velocity))
    task_power = float(np.dot(force_world, task_velocity))
    virtual_work_residual = abs(joint_power - task_power)
    if virtual_work_residual > 1.0e-12:
        raise ValueError("H18 virtual-work identity residual is too large")
    return {
        "profile_fraction": fraction,
        "minimum_jerk_blend": blend,
        "target_world_up_force_n": target_force,
        "world_up_force_n": world_up_force,
        "world_up_force_step_n": world_up_force - previous_force,
        "force_world_xyz_n": force_world.tolist(),
        "translation_jacobian_world_m_rad": jacobian.tolist(),
        "translation_jacobian_singular_values": np.linalg.svd(
            jacobian, compute_uv=False
        ).tolist(),
        "joint_effort_nm": joint_effort.tolist(),
        "joint_effort_step_nm": effort_delta.tolist(),
        "maximum_abs_joint_effort_nm": float(np.max(np.abs(joint_effort))),
        "maximum_abs_joint_effort_step_nm": float(
            np.max(np.abs(effort_delta))
        ),
        "actuator_effort_limits_nm": limits.tolist(),
        "joint_power_w": joint_power,
        "task_power_w": task_power,
        "virtual_work_residual_w": virtual_work_residual,
        "numeric_jacobian_step_rad": NUMERIC_JACOBIAN_STEP_RAD,
        "joint_state_only": True,
        "object_truth_used": False,
        "contact_truth_used": False,
        "event_truth_used": False,
        "object_pose_written": False,
    }


__all__ = [
    "LiftTaskSpaceVerticalForceRampConfig",
    "NUMERIC_JACOBIAN_STEP_RAD",
    "derive_vertical_force_step",
    "frozen_payload_weight_force_n",
    "load_lift_task_space_vertical_force_ramp_config",
    "numeric_tcp_translation_jacobian",
]
