"""B-V2-H15 bumpless transfer from position preload to gravity effort."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any, Mapping, Sequence

import numpy as np

from .pre_lift_arm_drive_compliance import minimum_jerk_blend


CONFIG_KEYS = (
    "enabled",
    "threshold_label",
    "transition_steps",
    "stability_window_steps",
    "maximum_gravity_feedforward_nm",
    "maximum_feedforward_step_nm",
    "maximum_target_bias_rad",
    "maximum_effort_continuity_residual_nm",
    "maximum_effort_readback_error_nm",
    "maximum_entry_moment_score_nm",
    "maximum_entry_load_imbalance",
)


@dataclass(frozen=True)
class PreLiftGravityPreloadTransferConfig:
    """One evidence-bounded H15 diagnostic parameter set."""

    enabled: bool
    threshold_label: str
    transition_steps: int
    stability_window_steps: int
    maximum_gravity_feedforward_nm: float
    maximum_feedforward_step_nm: float
    maximum_target_bias_rad: float
    maximum_effort_continuity_residual_nm: float
    maximum_effort_readback_error_nm: float
    maximum_entry_moment_score_nm: float
    maximum_entry_load_imbalance: float

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("H15 enabled must be boolean")
        if self.threshold_label != "SIM_TUNING_ONLY_B_V2_H15":
            raise ValueError("H15 must stay SIM_TUNING_ONLY_B_V2_H15")
        if self.transition_steps != 240 or self.stability_window_steps != 240:
            raise ValueError("H15 transition and stability windows must each be 240 steps")
        for name in CONFIG_KEYS[4:]:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{name} must be numeric")
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        # 300 N*m is the existing authored iiwa per-joint maxEffort.  H15 may
        # consume it as a fail-closed bound but may neither rewrite nor exceed it.
        if not math.isclose(
            self.maximum_gravity_feedforward_nm, 300.0, rel_tol=0.0, abs_tol=0.0
        ):
            raise ValueError("H15 gravity feedforward bound is frozen at 300 N*m")
        if not math.isclose(
            self.maximum_feedforward_step_nm, 2.5, rel_tol=0.0, abs_tol=0.0
        ):
            raise ValueError("H15 feedforward step bound is frozen at 2.5 N*m")
        if self.maximum_target_bias_rad > 0.01:
            raise ValueError("H15 target bias must not exceed the H6 0.01 rad bound")
        if self.maximum_entry_moment_score_nm >= 0.30:
            raise ValueError("H15 entry moment score must stay below the 0.30 N*m hard gate")
        if self.maximum_entry_load_imbalance > 0.18:
            raise ValueError("H15 entry load imbalance must be no greater than 0.18")


def load_pre_lift_gravity_preload_transfer_config(
    value: Any,
) -> PreLiftGravityPreloadTransferConfig:
    """Load H15 strictly; historical contracts remain disabled by absence."""

    if value is None:
        return PreLiftGravityPreloadTransferConfig(
            enabled=False,
            threshold_label="SIM_TUNING_ONLY_B_V2_H15",
            transition_steps=240,
            stability_window_steps=240,
            maximum_gravity_feedforward_nm=300.0,
            maximum_feedforward_step_nm=2.5,
            maximum_target_bias_rad=0.01,
            maximum_effort_continuity_residual_nm=1.0e-6,
            maximum_effort_readback_error_nm=1.0e-5,
            maximum_entry_moment_score_nm=0.24,
            maximum_entry_load_imbalance=0.18,
        )
    if not isinstance(value, Mapping):
        raise ValueError("pre_lift_gravity_preload_transfer must be a mapping")
    unknown = sorted(set(value) - set(CONFIG_KEYS))
    missing = sorted(set(CONFIG_KEYS) - set(value))
    if unknown or missing:
        raise ValueError(
            "pre_lift_gravity_preload_transfer has unknown keys "
            f"{unknown} and/or missing keys {missing}"
        )
    return PreLiftGravityPreloadTransferConfig(
        enabled=value["enabled"],
        threshold_label=str(value["threshold_label"]),
        transition_steps=value["transition_steps"],
        stability_window_steps=value["stability_window_steps"],
        maximum_gravity_feedforward_nm=float(
            value["maximum_gravity_feedforward_nm"]
        ),
        maximum_feedforward_step_nm=float(value["maximum_feedforward_step_nm"]),
        maximum_target_bias_rad=float(value["maximum_target_bias_rad"]),
        maximum_effort_continuity_residual_nm=float(
            value["maximum_effort_continuity_residual_nm"]
        ),
        maximum_effort_readback_error_nm=float(
            value["maximum_effort_readback_error_nm"]
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


def _derive_target(
    path_arm_rad: Sequence[Real],
    position_preload_nm: Sequence[Real],
    gravity_feedforward_nm: Sequence[Real],
    compliant_stiffness_nm_rad: Real,
    config: PreLiftGravityPreloadTransferConfig,
) -> dict[str, Any]:
    path = _seven(path_arm_rad, "path_arm_rad")
    preload = _seven(position_preload_nm, "position_preload_nm")
    feedforward = _seven(gravity_feedforward_nm, "gravity_feedforward_nm")
    stiffness = float(compliant_stiffness_nm_rad)
    if not math.isfinite(stiffness) or stiffness <= 0.0:
        raise ValueError("compliant_stiffness_nm_rad must be positive and finite")
    if float(np.max(np.abs(feedforward))) > config.maximum_gravity_feedforward_nm:
        raise ValueError("H15 gravity feedforward exceeds the authored actuator bound")
    remaining_position_preload = preload - feedforward
    target_bias = remaining_position_preload / stiffness
    maximum_target_bias = float(np.max(np.abs(target_bias)))
    if maximum_target_bias > config.maximum_target_bias_rad:
        raise ValueError("H15 residual target bias exceeds its internal bound")
    target = path + target_bias
    position_effort = stiffness * (target - path)
    total_effort = position_effort + feedforward
    continuity_residual = total_effort - preload
    maximum_continuity_residual = float(np.max(np.abs(continuity_residual)))
    if (
        maximum_continuity_residual
        > config.maximum_effort_continuity_residual_nm
    ):
        raise ValueError("H15 total-effort continuity residual exceeds its bound")
    return {
        "path_arm_rad": path.tolist(),
        "target_arm_rad": target.tolist(),
        "target_bias_rad": target_bias.tolist(),
        "maximum_target_bias_rad": maximum_target_bias,
        "position_preload_remaining_nm": remaining_position_preload.tolist(),
        "position_effort_nm": position_effort.tolist(),
        "gravity_feedforward_nm": feedforward.tolist(),
        "total_effort_nm": total_effort.tolist(),
        "effort_continuity_residual_nm": continuity_residual.tolist(),
        "maximum_effort_continuity_residual_nm": maximum_continuity_residual,
    }


def derive_gravity_preload_transfer_step(
    realized_arm_rad: Sequence[Real],
    position_preload_nm: Sequence[Real],
    generalized_gravity_nm: Sequence[Real],
    compliant_stiffness_nm_rad: Real,
    transition_step: int,
    previous_feedforward_nm: Sequence[Real],
    config: PreLiftGravityPreloadTransferConfig,
) -> dict[str, Any]:
    """Move one minimum-jerk fraction of gravity from position to effort."""

    if type(transition_step) is not int or not (
        0 <= transition_step < config.transition_steps
    ):
        raise ValueError("transition_step is outside the configured H15 transition")
    gravity = _seven(generalized_gravity_nm, "generalized_gravity_nm")
    previous = _seven(previous_feedforward_nm, "previous_feedforward_nm")
    fraction = float(transition_step + 1) / float(config.transition_steps)
    blend = minimum_jerk_blend(fraction)
    feedforward = blend * gravity
    feedforward_delta = feedforward - previous
    maximum_feedforward_step = float(np.max(np.abs(feedforward_delta)))
    if maximum_feedforward_step > config.maximum_feedforward_step_nm:
        raise ValueError("H15 feedforward step exceeds its internal bound")
    result = _derive_target(
        realized_arm_rad,
        position_preload_nm,
        feedforward,
        compliant_stiffness_nm_rad,
        config,
    )
    result.update(
        {
            "fraction": fraction,
            "minimum_jerk_blend": blend,
            "generalized_gravity_nm": gravity.tolist(),
            "feedforward_delta_nm": feedforward_delta.tolist(),
            "maximum_feedforward_step_nm": maximum_feedforward_step,
        }
    )
    return result


def gravity_supported_path_drive_target(
    path_arm_rad: Sequence[Real],
    position_preload_nm: Sequence[Real],
    generalized_gravity_nm: Sequence[Real],
    compliant_stiffness_nm_rad: Real,
    previous_feedforward_nm: Sequence[Real],
    config: PreLiftGravityPreloadTransferConfig,
) -> dict[str, Any]:
    """Hold/move a robot path with full gravity effort and residual preload."""

    gravity = _seven(generalized_gravity_nm, "generalized_gravity_nm")
    previous = _seven(previous_feedforward_nm, "previous_feedforward_nm")
    feedforward_delta = gravity - previous
    maximum_feedforward_step = float(np.max(np.abs(feedforward_delta)))
    if maximum_feedforward_step > config.maximum_feedforward_step_nm:
        raise ValueError("H15 gravity update exceeds its internal step bound")
    result = _derive_target(
        path_arm_rad,
        position_preload_nm,
        gravity,
        compliant_stiffness_nm_rad,
        config,
    )
    result.update(
        {
            "generalized_gravity_nm": gravity.tolist(),
            "feedforward_delta_nm": feedforward_delta.tolist(),
            "maximum_feedforward_step_nm": maximum_feedforward_step,
        }
    )
    return result
