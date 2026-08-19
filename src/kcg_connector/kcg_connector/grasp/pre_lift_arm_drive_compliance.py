"""B-V2-H6 bumpless arm-drive compliance for contact-phase lift."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any, Mapping, Sequence

import numpy as np


CONFIG_KEYS = (
    "enabled",
    "threshold_label",
    "stiffness_scale",
    "damping_scale",
    "transition_steps",
    "stability_window_steps",
    "maximum_bumpless_target_delta_rad",
    "maximum_position_effort_residual_nm",
    "maximum_entry_moment_score_nm",
    "maximum_entry_load_imbalance",
)


@dataclass(frozen=True)
class PreLiftArmDriveComplianceConfig:
    enabled: bool
    threshold_label: str
    stiffness_scale: float
    damping_scale: float
    transition_steps: int
    stability_window_steps: int
    maximum_bumpless_target_delta_rad: float
    maximum_position_effort_residual_nm: float
    maximum_entry_moment_score_nm: float
    maximum_entry_load_imbalance: float

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("pre-lift arm-drive compliance enabled must be boolean")
        if self.threshold_label != "SIM_TUNING_ONLY_B_V2_H6":
            raise ValueError(
                "pre-lift arm-drive compliance must stay SIM_TUNING_ONLY_B_V2_H6"
            )
        # H6 is one evidence-derived parameter set, not a search surface.
        if not math.isclose(self.stiffness_scale, 0.25, rel_tol=0.0, abs_tol=0.0):
            raise ValueError("H6 stiffness_scale is frozen at the derived value 0.25")
        if not math.isclose(self.damping_scale, 1.0, rel_tol=0.0, abs_tol=0.0):
            raise ValueError("H6 damping_scale is frozen at 1.0")
        if self.transition_steps != 240 or self.stability_window_steps != 240:
            raise ValueError("H6 transition and stability windows must each be 240 steps")
        for name in (
            "maximum_bumpless_target_delta_rad",
            "maximum_position_effort_residual_nm",
            "maximum_entry_moment_score_nm",
            "maximum_entry_load_imbalance",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{name} must be numeric")
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.maximum_bumpless_target_delta_rad >= 0.03:
            raise ValueError("H6 target delta must stay below the 0.03 rad tracking gate")
        if self.maximum_entry_moment_score_nm >= 0.30:
            raise ValueError("H6 entry moment score must stay below the 0.30 N*m hard gate")
        if self.maximum_entry_load_imbalance > 0.18:
            raise ValueError("H6 entry load imbalance must be no greater than 0.18")


def load_pre_lift_arm_drive_compliance_config(
    value: Any,
) -> PreLiftArmDriveComplianceConfig:
    """Load the strict H6 section; absence keeps historical contracts disabled."""

    if value is None:
        return PreLiftArmDriveComplianceConfig(
            enabled=False,
            threshold_label="SIM_TUNING_ONLY_B_V2_H6",
            stiffness_scale=0.25,
            damping_scale=1.0,
            transition_steps=240,
            stability_window_steps=240,
            maximum_bumpless_target_delta_rad=0.01,
            maximum_position_effort_residual_nm=1.0e-6,
            maximum_entry_moment_score_nm=0.24,
            maximum_entry_load_imbalance=0.18,
        )
    if not isinstance(value, Mapping):
        raise ValueError("pre_lift_arm_drive_compliance must be a mapping")
    unknown = sorted(set(value) - set(CONFIG_KEYS))
    missing = sorted(set(CONFIG_KEYS) - set(value))
    if unknown or missing:
        raise ValueError(
            "pre_lift_arm_drive_compliance has unknown keys "
            f"{unknown} and/or missing keys {missing}"
        )
    return PreLiftArmDriveComplianceConfig(
        enabled=value["enabled"],
        threshold_label=str(value["threshold_label"]),
        stiffness_scale=float(value["stiffness_scale"]),
        damping_scale=float(value["damping_scale"]),
        transition_steps=value["transition_steps"],
        stability_window_steps=value["stability_window_steps"],
        maximum_bumpless_target_delta_rad=float(
            value["maximum_bumpless_target_delta_rad"]
        ),
        maximum_position_effort_residual_nm=float(
            value["maximum_position_effort_residual_nm"]
        ),
        maximum_entry_moment_score_nm=float(
            value["maximum_entry_moment_score_nm"]
        ),
        maximum_entry_load_imbalance=float(
            value["maximum_entry_load_imbalance"]
        ),
    )


def _seven(values: Sequence[Real], label: str) -> np.ndarray:
    result = np.asarray(tuple(values), dtype=np.float64)
    if result.shape != (7,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must contain seven finite values")
    return result


def minimum_jerk_blend(fraction: Real) -> float:
    if isinstance(fraction, bool) or not isinstance(fraction, Real):
        raise ValueError("fraction must be numeric")
    value = float(fraction)
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError("fraction must be finite and within [0, 1]")
    return value**3 * (10.0 + value * (-15.0 + 6.0 * value))


def capture_position_preload_nm(
    commanded_arm_rad: Sequence[Real],
    realized_arm_rad: Sequence[Real],
    nominal_stiffness: Real,
) -> np.ndarray:
    commanded = _seven(commanded_arm_rad, "commanded_arm_rad")
    realized = _seven(realized_arm_rad, "realized_arm_rad")
    stiffness = float(nominal_stiffness)
    if not math.isfinite(stiffness) or stiffness <= 0.0:
        raise ValueError("nominal_stiffness must be positive and finite")
    return stiffness * (commanded - realized)


def derive_bumpless_drive_step(
    realized_arm_rad: Sequence[Real],
    position_preload_nm: Sequence[Real],
    nominal_stiffness: Real,
    nominal_damping: Real,
    transition_step: int,
    config: PreLiftArmDriveComplianceConfig,
) -> dict[str, Any]:
    """Return one gain/target step preserving the captured position effort."""

    realized = _seven(realized_arm_rad, "realized_arm_rad")
    preload = _seven(position_preload_nm, "position_preload_nm")
    stiffness = float(nominal_stiffness)
    damping = float(nominal_damping)
    if not math.isfinite(stiffness) or stiffness <= 0.0:
        raise ValueError("nominal_stiffness must be positive and finite")
    if not math.isfinite(damping) or damping <= 0.0:
        raise ValueError("nominal_damping must be positive and finite")
    if type(transition_step) is not int or not (
        0 <= transition_step < config.transition_steps
    ):
        raise ValueError("transition_step is outside the configured transition")
    fraction = float(transition_step + 1) / float(config.transition_steps)
    blend = minimum_jerk_blend(fraction)
    stiffness_scale = 1.0 - (1.0 - config.stiffness_scale) * blend
    applied_stiffness = stiffness * stiffness_scale
    applied_damping = damping * config.damping_scale
    bias = preload / applied_stiffness
    maximum_bias = float(np.max(np.abs(bias)))
    if maximum_bias > config.maximum_bumpless_target_delta_rad:
        raise ValueError("H6 bumpless target delta exceeds its internal bound")
    target = realized + bias
    residual = applied_stiffness * (target - realized) - preload
    maximum_residual = float(np.max(np.abs(residual)))
    if maximum_residual > config.maximum_position_effort_residual_nm:
        raise ValueError("H6 position-effort continuity residual exceeds its bound")
    return {
        "fraction": fraction,
        "minimum_jerk_blend": blend,
        "stiffness_scale": stiffness_scale,
        "applied_stiffness": applied_stiffness,
        "applied_damping": applied_damping,
        "target_arm_rad": target.tolist(),
        "target_bias_rad": bias.tolist(),
        "maximum_target_bias_rad": maximum_bias,
        "position_effort_residual_nm": residual.tolist(),
        "maximum_position_effort_residual_nm": maximum_residual,
    }


def compliant_path_drive_target(
    path_arm_rad: Sequence[Real],
    position_preload_nm: Sequence[Real],
    nominal_stiffness: Real,
    config: PreLiftArmDriveComplianceConfig,
) -> dict[str, Any]:
    """Add the frozen H6 load-holding bias to one robot-only lift path state."""

    path = _seven(path_arm_rad, "path_arm_rad")
    preload = _seven(position_preload_nm, "position_preload_nm")
    stiffness = float(nominal_stiffness) * config.stiffness_scale
    if not math.isfinite(stiffness) or stiffness <= 0.0:
        raise ValueError("compliant stiffness must be positive and finite")
    bias = preload / stiffness
    maximum_bias = float(np.max(np.abs(bias)))
    if maximum_bias > config.maximum_bumpless_target_delta_rad:
        raise ValueError("H6 compliant path bias exceeds its internal bound")
    target = path + bias
    return {
        "path_arm_rad": path.tolist(),
        "target_arm_rad": target.tolist(),
        "target_bias_rad": bias.tolist(),
        "maximum_target_bias_rad": maximum_bias,
        "applied_stiffness": stiffness,
        "applied_damping_scale": config.damping_scale,
    }


__all__ = [
    "CONFIG_KEYS",
    "PreLiftArmDriveComplianceConfig",
    "capture_position_preload_nm",
    "compliant_path_drive_target",
    "derive_bumpless_drive_step",
    "load_pre_lift_arm_drive_compliance_config",
    "minimum_jerk_blend",
]
