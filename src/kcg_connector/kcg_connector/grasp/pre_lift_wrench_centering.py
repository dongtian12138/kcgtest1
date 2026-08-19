"""Bounded, truth-free pre-lift wrench-centering mathematics.

The runtime owns motion and sensor sampling.  This module only converts five
predeclared wrench means into a central-difference XY Jacobian and one bounded
damped-least-squares correction.  It has no object pose, contact identity or
Isaac dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any, Mapping, Sequence

import numpy as np


CONFIG_KEYS = (
    "enabled",
    "threshold_label",
    "probe_offset_m",
    "probe_move_steps",
    "probe_settle_steps",
    "probe_sample_steps",
    "correction_move_steps",
    "reference_window_steps",
    "maximum_correction_m",
    "maximum_total_offset_m",
    "damping_ratio",
    "objective_force_scale_n",
    "objective_moment_scale_nm",
    "maximum_entry_moment_score_nm",
    "maximum_entry_load_imbalance",
    "maximum_wrench_roundtrip_error",
)


@dataclass(frozen=True)
class PreLiftWrenchCenteringConfig:
    enabled: bool
    threshold_label: str
    probe_offset_m: float
    probe_move_steps: int
    probe_settle_steps: int
    probe_sample_steps: int
    correction_move_steps: int
    reference_window_steps: int
    maximum_correction_m: float
    maximum_total_offset_m: float
    damping_ratio: float
    objective_force_scale_n: float
    objective_moment_scale_nm: float
    maximum_entry_moment_score_nm: float
    maximum_entry_load_imbalance: float
    maximum_wrench_roundtrip_error: float

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("pre-lift centering enabled must be boolean")
        if self.threshold_label != "SIM_TUNING_ONLY_B_V2_H4":
            raise ValueError("pre-lift centering must stay SIM_TUNING_ONLY_B_V2_H4")
        for name in (
            "probe_offset_m",
            "maximum_correction_m",
            "maximum_total_offset_m",
            "damping_ratio",
            "objective_force_scale_n",
            "objective_moment_scale_nm",
            "maximum_entry_moment_score_nm",
            "maximum_entry_load_imbalance",
            "maximum_wrench_roundtrip_error",
        ):
            value = getattr(self, name)
            if not isinstance(value, Real) or isinstance(value, bool):
                raise ValueError(f"{name} must be numeric")
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        for name in (
            "probe_move_steps",
            "probe_settle_steps",
            "probe_sample_steps",
            "correction_move_steps",
            "reference_window_steps",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.maximum_correction_m > self.maximum_total_offset_m:
            raise ValueError("one correction cannot exceed the total XY bound")
        if self.maximum_entry_moment_score_nm >= 0.30:
            raise ValueError("pre-lift entry score must stay below the 0.30 N*m hard gate")
        if self.objective_force_scale_n != 8.0:
            raise ValueError("centering force normalization must preserve the 8 N gate")
        if self.objective_moment_scale_nm != 0.30:
            raise ValueError("centering moment normalization must preserve the 0.30 N*m gate")
        if not 0.0 < self.maximum_entry_load_imbalance <= 0.18:
            raise ValueError("pre-lift load imbalance must be no greater than 0.18")


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be positive and finite")
    return result


def _integer(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def load_pre_lift_wrench_centering_config(
    value: Any,
) -> PreLiftWrenchCenteringConfig:
    """Strictly load the V2 H4 section; absence keeps historical configs off."""

    if value is None:
        return PreLiftWrenchCenteringConfig(
            enabled=False,
            threshold_label="SIM_TUNING_ONLY_B_V2_H4",
            probe_offset_m=0.00020,
            probe_move_steps=48,
            probe_settle_steps=24,
            probe_sample_steps=24,
            correction_move_steps=96,
            reference_window_steps=240,
            maximum_correction_m=0.00050,
            maximum_total_offset_m=0.00050,
            damping_ratio=0.10,
            objective_force_scale_n=8.0,
            objective_moment_scale_nm=0.30,
            maximum_entry_moment_score_nm=0.24,
            maximum_entry_load_imbalance=0.18,
            maximum_wrench_roundtrip_error=1.0e-9,
        )
    if not isinstance(value, Mapping):
        raise ValueError("pre_lift_centering must be a mapping")
    unknown = sorted(set(value) - set(CONFIG_KEYS))
    missing = sorted(set(CONFIG_KEYS) - set(value))
    if unknown or missing:
        raise ValueError(
            "pre_lift_centering has unknown keys "
            f"{unknown} and/or missing keys {missing}"
        )
    if type(value["enabled"]) is not bool:
        raise ValueError("pre_lift_centering.enabled must be boolean")
    return PreLiftWrenchCenteringConfig(
        enabled=value["enabled"],
        threshold_label=str(value["threshold_label"]),
        probe_offset_m=_number(value["probe_offset_m"], "probe_offset_m"),
        probe_move_steps=_integer(value["probe_move_steps"], "probe_move_steps"),
        probe_settle_steps=_integer(value["probe_settle_steps"], "probe_settle_steps"),
        probe_sample_steps=_integer(value["probe_sample_steps"], "probe_sample_steps"),
        correction_move_steps=_integer(value["correction_move_steps"], "correction_move_steps"),
        reference_window_steps=_integer(value["reference_window_steps"], "reference_window_steps"),
        maximum_correction_m=_number(value["maximum_correction_m"], "maximum_correction_m"),
        maximum_total_offset_m=_number(value["maximum_total_offset_m"], "maximum_total_offset_m"),
        damping_ratio=_number(value["damping_ratio"], "damping_ratio"),
        objective_force_scale_n=_number(value["objective_force_scale_n"], "objective_force_scale_n"),
        objective_moment_scale_nm=_number(value["objective_moment_scale_nm"], "objective_moment_scale_nm"),
        maximum_entry_moment_score_nm=_number(value["maximum_entry_moment_score_nm"], "maximum_entry_moment_score_nm"),
        maximum_entry_load_imbalance=_number(value["maximum_entry_load_imbalance"], "maximum_entry_load_imbalance"),
        maximum_wrench_roundtrip_error=_number(value["maximum_wrench_roundtrip_error"], "maximum_wrench_roundtrip_error"),
    )


def _wrench(value: Sequence[Real], label: str) -> np.ndarray:
    result = np.asarray(tuple(value), dtype=np.float64)
    if result.shape != (6,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be a finite 6-vector")
    return result


def normalized_centering_objective(
    wrench_task: Sequence[Real],
    *,
    force_scale_n: float,
    moment_scale_nm: float,
) -> np.ndarray:
    """Return dimensionless [Fx,Fy,Mx,My] using frozen gate scales."""

    wrench = _wrench(wrench_task, "wrench_task")
    force_scale = _number(force_scale_n, "force_scale_n")
    moment_scale = _number(moment_scale_nm, "moment_scale_nm")
    return np.asarray(
        (
            wrench[0] / force_scale,
            wrench[1] / force_scale,
            wrench[3] / moment_scale,
            wrench[4] / moment_scale,
        ),
        dtype=np.float64,
    )


def solve_bounded_xy_centering(
    center_wrench_task: Sequence[Real],
    plus_x_wrench_task: Sequence[Real],
    minus_x_wrench_task: Sequence[Real],
    plus_y_wrench_task: Sequence[Real],
    minus_y_wrench_task: Sequence[Real],
    config: PreLiftWrenchCenteringConfig,
) -> dict[str, Any]:
    """Compute one bounded correction from a central-difference Jacobian."""

    if not config.enabled:
        raise ValueError("pre-lift wrench centering is disabled")
    objective = lambda wrench: normalized_centering_objective(
        wrench,
        force_scale_n=config.objective_force_scale_n,
        moment_scale_nm=config.objective_moment_scale_nm,
    )
    center = objective(center_wrench_task)
    plus_x = objective(plus_x_wrench_task)
    minus_x = objective(minus_x_wrench_task)
    plus_y = objective(plus_y_wrench_task)
    minus_y = objective(minus_y_wrench_task)
    jacobian = np.column_stack(
        (
            (plus_x - minus_x) / (2.0 * config.probe_offset_m),
            (plus_y - minus_y) / (2.0 * config.probe_offset_m),
        )
    )
    if not np.all(np.isfinite(jacobian)):
        raise ValueError("centering Jacobian is non-finite")
    left, singular_values, right_t = np.linalg.svd(
        jacobian, full_matrices=False
    )
    if singular_values.shape != (2,) or singular_values[0] <= 0.0:
        raise ValueError("centering Jacobian lacks two independent XY directions")
    rank_threshold = singular_values[0] * 1.0e-6
    rank = int(np.count_nonzero(singular_values > rank_threshold))
    if rank < 2:
        raise ValueError("centering Jacobian lacks two independent XY directions")
    damping = config.damping_ratio * singular_values[0]
    gains = singular_values / (singular_values * singular_values + damping * damping)
    unconstrained = -(right_t.T @ (gains * (left.T @ center)))
    unconstrained_norm = float(np.linalg.norm(unconstrained))
    correction = unconstrained.copy()
    clipped = False
    if unconstrained_norm > config.maximum_correction_m:
        correction *= config.maximum_correction_m / unconstrained_norm
        clipped = True
    correction_norm = float(np.linalg.norm(correction))
    if correction_norm > config.maximum_total_offset_m + 1.0e-15:
        raise ValueError("centering correction exceeds total XY bound")
    predicted = center + jacobian @ correction
    return {
        "objective_components": ["Fx/8N", "Fy/8N", "Mx/0.30Nm", "My/0.30Nm"],
        "center_objective": center.tolist(),
        "jacobian_per_m": jacobian.tolist(),
        "singular_values_per_m": singular_values.tolist(),
        "condition_number": float(singular_values[0] / singular_values[-1]),
        "damping_per_m": float(damping),
        "unconstrained_correction_xy_m": unconstrained.tolist(),
        "unconstrained_correction_norm_m": unconstrained_norm,
        "correction_xy_m": correction.tolist(),
        "correction_norm_m": correction_norm,
        "correction_clipped": clipped,
        "predicted_objective": predicted.tolist(),
        "predicted_objective_norm": float(np.linalg.norm(predicted)),
        "center_objective_norm": float(np.linalg.norm(center)),
    }


__all__ = [
    "CONFIG_KEYS",
    "PreLiftWrenchCenteringConfig",
    "load_pre_lift_wrench_centering_config",
    "normalized_centering_objective",
    "solve_bounded_xy_centering",
]
