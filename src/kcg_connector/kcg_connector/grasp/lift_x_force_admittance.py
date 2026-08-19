"""B-V2-H8 bounded world-X target / task-Fx admittance for staged lift.

RUN05 measured the local response from a world-X robot target displacement to
payload-reference Fx expressed in the robot-FK grasp-TCP frame.  This module
therefore consumes only that task-frame force and returns a bounded world-X
target correction.  It has no Isaac, object-pose, contact-name, contact-normal,
or assembly-event input.  The raw sensor-origin force/moment guard remains
outside this helper and keeps authority over every physics step.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any, Mapping


SOURCE_RUN_ID = "B-V2-GRASP-05"
SOURCE_TARGET_STIFFNESS_N_M = 41024.63885398004
TASK_X_COMPLIANCE_M_N = 1.0 / SOURCE_TARGET_STIFFNESS_N_M
MAXIMUM_TOTAL_CORRECTION_M = 15.0e-6
MAXIMUM_STEP_CORRECTION_M = 1.0e-6

CONFIG_KEYS = (
    "enabled",
    "threshold_label",
    "source_run_id",
    "source_target_stiffness_n_m",
    "task_x_compliance_m_n",
    "maximum_total_correction_m",
    "maximum_step_correction_m",
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


@dataclass(frozen=True)
class LiftXForceAdmittanceConfig:
    enabled: bool
    threshold_label: str
    source_run_id: str
    source_target_stiffness_n_m: float
    task_x_compliance_m_n: float
    maximum_total_correction_m: float
    maximum_step_correction_m: float
    sensor_origin_force_gate_n: float
    sensor_origin_moment_gate_nm: float

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("lift X force admittance enabled must be boolean")
        if self.threshold_label != "SIM_TUNING_ONLY_B_V2_H8":
            raise ValueError("lift X force admittance threshold label changed")
        if self.source_run_id != SOURCE_RUN_ID:
            raise ValueError("H8 source run must remain B-V2-GRASP-05")
        frozen = {
            "source_target_stiffness_n_m": SOURCE_TARGET_STIFFNESS_N_M,
            "task_x_compliance_m_n": TASK_X_COMPLIANCE_M_N,
            "maximum_total_correction_m": MAXIMUM_TOTAL_CORRECTION_M,
            "maximum_step_correction_m": MAXIMUM_STEP_CORRECTION_M,
            "sensor_origin_force_gate_n": 8.0,
            "sensor_origin_moment_gate_nm": 0.30,
        }
        for name, expected in frozen.items():
            value = _finite(getattr(self, name), name)
            if not math.isclose(value, expected, rel_tol=0.0, abs_tol=1.0e-15):
                raise ValueError(f"H8 {name} is frozen at {expected}")
        if self.maximum_step_correction_m > self.maximum_total_correction_m:
            raise ValueError("H8 per-step correction cannot exceed total bound")


def load_lift_x_force_admittance_config(
    value: Any,
) -> LiftXForceAdmittanceConfig:
    """Load the exact one-value H8 contract; absence keeps old files off."""

    defaults = dict(
        enabled=False,
        threshold_label="SIM_TUNING_ONLY_B_V2_H8",
        source_run_id=SOURCE_RUN_ID,
        source_target_stiffness_n_m=SOURCE_TARGET_STIFFNESS_N_M,
        task_x_compliance_m_n=TASK_X_COMPLIANCE_M_N,
        maximum_total_correction_m=MAXIMUM_TOTAL_CORRECTION_M,
        maximum_step_correction_m=MAXIMUM_STEP_CORRECTION_M,
        sensor_origin_force_gate_n=8.0,
        sensor_origin_moment_gate_nm=0.30,
    )
    if value is None:
        return LiftXForceAdmittanceConfig(**defaults)
    if not isinstance(value, Mapping):
        raise ValueError("lift_x_force_admittance must be a mapping")
    unknown = sorted(set(value) - set(CONFIG_KEYS))
    missing = sorted(set(CONFIG_KEYS) - set(value))
    if unknown or missing:
        raise ValueError(
            "lift_x_force_admittance has unknown keys "
            f"{unknown} and/or missing keys {missing}"
        )
    return LiftXForceAdmittanceConfig(
        enabled=value["enabled"],
        threshold_label=str(value["threshold_label"]),
        source_run_id=str(value["source_run_id"]),
        source_target_stiffness_n_m=float(
            value["source_target_stiffness_n_m"]
        ),
        task_x_compliance_m_n=float(value["task_x_compliance_m_n"]),
        maximum_total_correction_m=float(
            value["maximum_total_correction_m"]
        ),
        maximum_step_correction_m=float(
            value["maximum_step_correction_m"]
        ),
        sensor_origin_force_gate_n=float(value["sensor_origin_force_gate_n"]),
        sensor_origin_moment_gate_nm=float(
            value["sensor_origin_moment_gate_nm"]
        ),
    )


def derive_lift_x_force_admittance_step(
    task_payload_force_x_n: Real,
    previous_correction_m: Real,
    config: LiftXForceAdmittanceConfig,
) -> dict[str, float | bool]:
    """Return one force-opposing, bounded world-X target correction step."""

    if not config.enabled:
        raise ValueError("lift X force admittance is disabled")
    force = _finite(task_payload_force_x_n, "task_payload_force_x_n")
    previous = _finite(previous_correction_m, "previous_correction_m")
    if abs(previous) > config.maximum_total_correction_m + 1.0e-15:
        raise ValueError("previous H8 correction exceeds the total bound")
    if abs(force) > config.sensor_origin_force_gate_n:
        raise ValueError("H8 input force already exceeds the frozen force gate")
    desired_unbounded = -config.task_x_compliance_m_n * force
    desired_bounded = max(
        -config.maximum_total_correction_m,
        min(config.maximum_total_correction_m, desired_unbounded),
    )
    requested_delta = desired_bounded - previous
    applied_delta = max(
        -config.maximum_step_correction_m,
        min(config.maximum_step_correction_m, requested_delta),
    )
    applied = previous + applied_delta
    if abs(applied) > config.maximum_total_correction_m + 1.0e-15:
        raise ValueError("applied H8 correction exceeds the total bound")
    return {
        "task_payload_force_x_n": force,
        "previous_correction_m": previous,
        "desired_unbounded_correction_m": desired_unbounded,
        "desired_bounded_correction_m": desired_bounded,
        "requested_delta_m": requested_delta,
        "applied_delta_m": applied_delta,
        "applied_correction_m": applied,
        "total_bound_active": not math.isclose(
            desired_unbounded, desired_bounded, rel_tol=0.0, abs_tol=0.0
        ),
        "rate_bound_active": not math.isclose(
            requested_delta, applied_delta, rel_tol=0.0, abs_tol=0.0
        ),
    }


__all__ = [
    "CONFIG_KEYS",
    "LiftXForceAdmittanceConfig",
    "MAXIMUM_STEP_CORRECTION_M",
    "MAXIMUM_TOTAL_CORRECTION_M",
    "SOURCE_RUN_ID",
    "SOURCE_TARGET_STIFFNESS_N_M",
    "TASK_X_COMPLIANCE_M_N",
    "derive_lift_x_force_admittance_step",
    "load_lift_x_force_admittance_config",
]
