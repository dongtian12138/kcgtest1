"""B-V2-H14 frame-consistent bounded XY admittance for staged lift.

RUN14 showed that the residual sensor-origin moment was primarily the lever-arm
image of two lateral force components.  H8 consumed task-frame Fx but applied
its correction directly on world X.  H14 instead rotates the complete task XY
force through robot FK into world coordinates, projects it onto world XY, and
returns one norm-bounded target correction.  It has no object, contact, event,
or simulation-truth input.  The unchanged raw sensor-origin guard retains
authority on every physics step.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any, Mapping, Sequence


THRESHOLD_LABEL = "SIM_TUNING_ONLY_B_V2_H14"
SOURCE_RUN_ID = "B-V2-GRASP-14"
SOURCE_H8_RUN_ID = "B-V2-GRASP-05"
SOURCE_TARGET_STIFFNESS_N_M = 41024.63885398004
TASK_XY_COMPLIANCE_M_N = 1.0 / SOURCE_TARGET_STIFFNESS_N_M
MAXIMUM_TOTAL_CORRECTION_NORM_M = 30.0e-6
MAXIMUM_STEP_CORRECTION_NORM_M = 1.0e-6

CONFIG_KEYS = (
    "enabled",
    "threshold_label",
    "source_run_id",
    "source_h8_run_id",
    "source_target_stiffness_n_m",
    "task_xy_compliance_m_n",
    "maximum_total_correction_norm_m",
    "maximum_step_correction_norm_m",
    "sensor_origin_force_gate_n",
    "sensor_origin_moment_gate_nm",
)


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _vector(value: Any, size: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a length-{size} sequence")
    if len(value) != size:
        raise ValueError(f"{label} must contain exactly {size} values")
    return tuple(_finite(item, f"{label}[{index}]") for index, item in enumerate(value))


def _rotation(value: Any) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("world_from_task_rotation must be a 3x3 sequence")
    if len(value) != 3:
        raise ValueError("world_from_task_rotation must have three rows")
    rows = tuple(
        _vector(row, 3, f"world_from_task_rotation[{index}]")
        for index, row in enumerate(value)
    )
    for first in range(3):
        for second in range(3):
            dot = sum(rows[index][first] * rows[index][second] for index in range(3))
            expected = 1.0 if first == second else 0.0
            if not math.isclose(dot, expected, rel_tol=0.0, abs_tol=1.0e-6):
                raise ValueError("world_from_task_rotation must be orthonormal")
    determinant = (
        rows[0][0] * (rows[1][1] * rows[2][2] - rows[1][2] * rows[2][1])
        - rows[0][1] * (rows[1][0] * rows[2][2] - rows[1][2] * rows[2][0])
        + rows[0][2] * (rows[1][0] * rows[2][1] - rows[1][1] * rows[2][0])
    )
    if not math.isclose(determinant, 1.0, rel_tol=0.0, abs_tol=1.0e-6):
        raise ValueError("world_from_task_rotation must have determinant +1")
    return rows


def _norm(value: Sequence[float]) -> float:
    return math.sqrt(sum(component * component for component in value))


def _clamp_norm(
    value: tuple[float, float], maximum_norm: float
) -> tuple[tuple[float, float], bool]:
    norm = _norm(value)
    if norm <= maximum_norm:
        return value, False
    scale = maximum_norm / norm
    return (value[0] * scale, value[1] * scale), True


@dataclass(frozen=True)
class LiftXYForceAdmittanceConfig:
    enabled: bool
    threshold_label: str
    source_run_id: str
    source_h8_run_id: str
    source_target_stiffness_n_m: float
    task_xy_compliance_m_n: float
    maximum_total_correction_norm_m: float
    maximum_step_correction_norm_m: float
    sensor_origin_force_gate_n: float
    sensor_origin_moment_gate_nm: float

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("lift XY force admittance enabled must be boolean")
        exact = {
            "threshold_label": THRESHOLD_LABEL,
            "source_run_id": SOURCE_RUN_ID,
            "source_h8_run_id": SOURCE_H8_RUN_ID,
            "source_target_stiffness_n_m": SOURCE_TARGET_STIFFNESS_N_M,
            "task_xy_compliance_m_n": TASK_XY_COMPLIANCE_M_N,
            "maximum_total_correction_norm_m": MAXIMUM_TOTAL_CORRECTION_NORM_M,
            "maximum_step_correction_norm_m": MAXIMUM_STEP_CORRECTION_NORM_M,
            "sensor_origin_force_gate_n": 8.0,
            "sensor_origin_moment_gate_nm": 0.30,
        }
        for name, expected in exact.items():
            actual = getattr(self, name)
            if isinstance(expected, str):
                matches = actual == expected
            else:
                matches = math.isclose(
                    _finite(actual, name), expected, rel_tol=0.0, abs_tol=1.0e-15
                )
            if not matches:
                raise ValueError(f"H14 {name} is frozen at {expected!r}")
        if (
            self.maximum_step_correction_norm_m
            > self.maximum_total_correction_norm_m
        ):
            raise ValueError("H14 per-step norm cannot exceed the total norm bound")


def _default_config(enabled: bool = False) -> LiftXYForceAdmittanceConfig:
    return LiftXYForceAdmittanceConfig(
        enabled=enabled,
        threshold_label=THRESHOLD_LABEL,
        source_run_id=SOURCE_RUN_ID,
        source_h8_run_id=SOURCE_H8_RUN_ID,
        source_target_stiffness_n_m=SOURCE_TARGET_STIFFNESS_N_M,
        task_xy_compliance_m_n=TASK_XY_COMPLIANCE_M_N,
        maximum_total_correction_norm_m=MAXIMUM_TOTAL_CORRECTION_NORM_M,
        maximum_step_correction_norm_m=MAXIMUM_STEP_CORRECTION_NORM_M,
        sensor_origin_force_gate_n=8.0,
        sensor_origin_moment_gate_nm=0.30,
    )


def load_lift_xy_force_admittance_config(
    value: Any,
) -> LiftXYForceAdmittanceConfig:
    """Load the exact H14 contract; historical files remain disabled."""

    if value is None:
        return _default_config(False)
    if not isinstance(value, Mapping):
        raise ValueError("lift_xy_force_admittance must be a mapping")
    unknown = sorted(set(value) - set(CONFIG_KEYS))
    missing = sorted(set(CONFIG_KEYS) - set(value))
    if unknown or missing:
        raise ValueError(
            "lift_xy_force_admittance has unknown keys "
            f"{unknown} and/or missing keys {missing}"
        )
    return LiftXYForceAdmittanceConfig(
        enabled=value["enabled"],
        threshold_label=str(value["threshold_label"]),
        source_run_id=str(value["source_run_id"]),
        source_h8_run_id=str(value["source_h8_run_id"]),
        source_target_stiffness_n_m=float(value["source_target_stiffness_n_m"]),
        task_xy_compliance_m_n=float(value["task_xy_compliance_m_n"]),
        maximum_total_correction_norm_m=float(
            value["maximum_total_correction_norm_m"]
        ),
        maximum_step_correction_norm_m=float(
            value["maximum_step_correction_norm_m"]
        ),
        sensor_origin_force_gate_n=float(value["sensor_origin_force_gate_n"]),
        sensor_origin_moment_gate_nm=float(value["sensor_origin_moment_gate_nm"]),
    )


def derive_lift_xy_force_admittance_step(
    task_payload_force_xy_n: Any,
    world_from_task_rotation: Any,
    previous_correction_xy_m: Any,
    config: LiftXYForceAdmittanceConfig,
) -> dict[str, Any]:
    """Return one FK-rotated, norm-bounded world-XY correction step."""

    if not config.enabled:
        raise ValueError("lift XY force admittance is disabled")
    force_task = _vector(task_payload_force_xy_n, 2, "task_payload_force_xy_n")
    previous = _vector(previous_correction_xy_m, 2, "previous_correction_xy_m")
    rotation = _rotation(world_from_task_rotation)
    if max(abs(component) for component in force_task) > config.sensor_origin_force_gate_n:
        raise ValueError("H14 input force already exceeds the frozen force gate")
    if _norm(previous) > config.maximum_total_correction_norm_m + 1.0e-15:
        raise ValueError("previous H14 correction exceeds the total norm bound")

    force_task_3 = (force_task[0], force_task[1], 0.0)
    force_world_3 = tuple(
        sum(rotation[row][column] * force_task_3[column] for column in range(3))
        for row in range(3)
    )
    force_world_xy = (force_world_3[0], force_world_3[1])
    desired_unbounded = (
        config.task_xy_compliance_m_n * force_world_xy[0],
        config.task_xy_compliance_m_n * force_world_xy[1],
    )
    desired_bounded, total_bound_active = _clamp_norm(
        desired_unbounded, config.maximum_total_correction_norm_m
    )
    requested_delta = (
        desired_bounded[0] - previous[0],
        desired_bounded[1] - previous[1],
    )
    applied_delta, rate_bound_active = _clamp_norm(
        requested_delta, config.maximum_step_correction_norm_m
    )
    applied = (
        previous[0] + applied_delta[0],
        previous[1] + applied_delta[1],
    )
    if _norm(applied) > config.maximum_total_correction_norm_m + 1.0e-15:
        raise ValueError("applied H14 correction exceeds the total norm bound")
    return {
        "task_payload_force_xy_n": list(force_task),
        "world_from_task_rotation": [list(row) for row in rotation],
        "world_projected_force_xy_n": list(force_world_xy),
        "discarded_world_force_z_n": force_world_3[2],
        "previous_correction_xy_m": list(previous),
        "desired_unbounded_correction_xy_m": list(desired_unbounded),
        "desired_bounded_correction_xy_m": list(desired_bounded),
        "requested_delta_xy_m": list(requested_delta),
        "applied_delta_xy_m": list(applied_delta),
        "applied_correction_xy_m": list(applied),
        "task_lateral_force_norm_n": _norm(force_task),
        "world_projected_force_norm_n": _norm(force_world_xy),
        "desired_unbounded_correction_norm_m": _norm(desired_unbounded),
        "desired_bounded_correction_norm_m": _norm(desired_bounded),
        "requested_delta_norm_m": _norm(requested_delta),
        "applied_delta_norm_m": _norm(applied_delta),
        "applied_correction_norm_m": _norm(applied),
        "total_bound_active": total_bound_active,
        "rate_bound_active": rate_bound_active,
    }


__all__ = [
    "CONFIG_KEYS",
    "LiftXYForceAdmittanceConfig",
    "MAXIMUM_STEP_CORRECTION_NORM_M",
    "MAXIMUM_TOTAL_CORRECTION_NORM_M",
    "SOURCE_H8_RUN_ID",
    "SOURCE_RUN_ID",
    "SOURCE_TARGET_STIFFNESS_N_M",
    "TASK_XY_COMPLIANCE_M_N",
    "derive_lift_xy_force_admittance_step",
    "load_lift_xy_force_admittance_config",
]
