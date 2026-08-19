"""Strict B-V2-H13 arm damping derived from RUN13 versus H12."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any, Mapping

from .pre_lift_arm_drive_compliance import minimum_jerk_blend


THRESHOLD_LABEL = "SIM_TUNING_ONLY_B_V2_H13"
SOURCE_LIFT_RUN_ID = "B-V2-GRASP-13"
SOURCE_HOLD_RUN_ID = "B-V2-H12-ZERO-LIFT-01"
SOURCE_MEDIAN_VELOCITY_RATIO = 2.18824945114135
INITIAL_DAMPING_NM_S_RAD = 400.0
FINAL_DAMPING_NM_S_RAD = (
    INITIAL_DAMPING_NM_S_RAD * SOURCE_MEDIAN_VELOCITY_RATIO
)
TRANSITION_STEPS = 240
MAXIMUM_READBACK_ERROR = 1.0e-3

CONFIG_KEYS = (
    "enabled",
    "threshold_label",
    "source_lift_run_id",
    "source_hold_run_id",
    "source_median_velocity_ratio",
    "initial_damping_nm_s_rad",
    "final_damping_nm_s_rad",
    "transition_steps",
    "maximum_readback_error",
)


@dataclass(frozen=True)
class LiftPhaseArmDampingConfig:
    enabled: bool
    threshold_label: str
    source_lift_run_id: str
    source_hold_run_id: str
    source_median_velocity_ratio: float
    initial_damping_nm_s_rad: float
    final_damping_nm_s_rad: float
    transition_steps: int
    maximum_readback_error: float

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("lift-phase arm damping enabled must be boolean")
        exact = {
            "threshold_label": THRESHOLD_LABEL,
            "source_lift_run_id": SOURCE_LIFT_RUN_ID,
            "source_hold_run_id": SOURCE_HOLD_RUN_ID,
            "source_median_velocity_ratio": SOURCE_MEDIAN_VELOCITY_RATIO,
            "initial_damping_nm_s_rad": INITIAL_DAMPING_NM_S_RAD,
            "final_damping_nm_s_rad": FINAL_DAMPING_NM_S_RAD,
            "transition_steps": TRANSITION_STEPS,
            "maximum_readback_error": MAXIMUM_READBACK_ERROR,
        }
        for name, required in exact.items():
            actual = getattr(self, name)
            if isinstance(required, str):
                matches = actual == required
            elif isinstance(required, int):
                matches = type(actual) is int and actual == required
            else:
                matches = (
                    isinstance(actual, Real)
                    and not isinstance(actual, bool)
                    and math.isclose(
                        float(actual), required, rel_tol=0.0, abs_tol=1.0e-12
                    )
                )
            if not matches:
                raise ValueError(
                    "H13 lift-phase arm damping "
                    f"{name} is frozen at {required!r}"
                )


def _default_config(enabled: bool = False) -> LiftPhaseArmDampingConfig:
    return LiftPhaseArmDampingConfig(
        enabled=enabled,
        threshold_label=THRESHOLD_LABEL,
        source_lift_run_id=SOURCE_LIFT_RUN_ID,
        source_hold_run_id=SOURCE_HOLD_RUN_ID,
        source_median_velocity_ratio=SOURCE_MEDIAN_VELOCITY_RATIO,
        initial_damping_nm_s_rad=INITIAL_DAMPING_NM_S_RAD,
        final_damping_nm_s_rad=FINAL_DAMPING_NM_S_RAD,
        transition_steps=TRANSITION_STEPS,
        maximum_readback_error=MAXIMUM_READBACK_ERROR,
    )


def load_lift_phase_arm_damping_config(
    value: Any,
) -> LiftPhaseArmDampingConfig:
    """Load H13; historical contracts without this section stay disabled."""

    if value is None:
        return _default_config(False)
    if not isinstance(value, Mapping):
        raise ValueError("lift_phase_arm_damping must be a mapping")
    unknown = sorted(set(value) - set(CONFIG_KEYS))
    missing = sorted(set(CONFIG_KEYS) - set(value))
    if unknown or missing:
        raise ValueError(
            "lift_phase_arm_damping has unknown keys "
            f"{unknown} and/or missing keys {missing}"
        )
    return LiftPhaseArmDampingConfig(
        enabled=value["enabled"],
        threshold_label=str(value["threshold_label"]),
        source_lift_run_id=str(value["source_lift_run_id"]),
        source_hold_run_id=str(value["source_hold_run_id"]),
        source_median_velocity_ratio=float(
            value["source_median_velocity_ratio"]
        ),
        initial_damping_nm_s_rad=float(value["initial_damping_nm_s_rad"]),
        final_damping_nm_s_rad=float(value["final_damping_nm_s_rad"]),
        transition_steps=value["transition_steps"],
        maximum_readback_error=float(value["maximum_readback_error"]),
    )


def derive_lift_phase_arm_damping_step(
    transition_step: int,
    config: LiftPhaseArmDampingConfig,
) -> dict[str, float | int]:
    """Return the only allowed monotonic H13 damping transition step."""

    if type(transition_step) is not int or not (
        0 <= transition_step < config.transition_steps
    ):
        raise ValueError("transition_step is outside the configured H13 window")
    fraction = float(transition_step + 1) / float(config.transition_steps)
    blend = minimum_jerk_blend(fraction)
    damping = config.initial_damping_nm_s_rad + blend * (
        config.final_damping_nm_s_rad - config.initial_damping_nm_s_rad
    )
    return {
        "transition_step": transition_step,
        "fraction": fraction,
        "minimum_jerk_blend": blend,
        "applied_damping_nm_s_rad": damping,
    }


__all__ = [
    "CONFIG_KEYS",
    "FINAL_DAMPING_NM_S_RAD",
    "LiftPhaseArmDampingConfig",
    "SOURCE_MEDIAN_VELOCITY_RATIO",
    "derive_lift_phase_arm_damping_step",
    "load_lift_phase_arm_damping_config",
]
