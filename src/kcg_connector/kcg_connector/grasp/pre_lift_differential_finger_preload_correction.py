"""Strict H23 fixed differential-finger preload correction.

The correction is the single bounded result of the completed H23 Isaac
diagnostic.  This module contains no estimator and no parameter search.  It
only validates the frozen evidence lineage and produces a minimum-jerk target
overlay in three-finger closure coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from .pre_lift_arm_drive_compliance import minimum_jerk_blend
from .three_finger_sequential_grasp import SequentialGraspConfig


THRESHOLD_LABEL = "SIM_TUNING_ONLY_B_V2_H23_FIXED_CORRECTION"
SOURCE_RUN_ID = (
    "B-V2-H23-DIFFERENTIAL-FINGER-PRELOAD-DIAGNOSTIC-IFIX01"
)
SOURCE_ANALYSIS_SHA256 = (
    "719b942a1739400300ca5a6a7de7a0f8dd13de259934c726452bbfdcd3530946"
)
FIXED_CORRECTION_CLOSURE_RAD = (
    0.00060507064062471,
    0.002476928142280831,
    -0.003081998782905541,
)
TRANSITION_STEPS_SOURCE = "H13_TRANSITION_STEPS"
STABILITY_STEPS_SOURCE = "REFERENCE_WINDOW_STEPS"
CONFIG_KEYS = (
    "enabled",
    "threshold_label",
    "source_run_id",
    "source_analysis_sha256",
    "fixed_correction_closure_rad",
    "correction_norm_limit_source",
    "transition_steps_source",
    "stability_steps_source",
    "zero_sum_required",
    "arm_target_fixed_during_correction",
    "vertical_feedforward_zero_required",
    "wrist_payload_reference_rebase_forbidden",
    "root_reference_refresh_from_trailing_stability_window",
    "persistent_via_fresh_h17_equilibrium",
    "raw_sensor_hard_gate_unchanged",
    "hard_gate_detection_delay_steps",
    "object_contact_event_truth_forbidden",
    "second_parameter_set_allowed",
    "formal_b_pass_claimed_from_fix_window",
)


def _three_finite(values: Sequence[float], label: str) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ValueError(f"{label} must contain exactly three values")
    result = tuple(float(value) for value in values)
    if any(not math.isfinite(value) for value in result):
        raise ValueError(f"{label} must contain only finite values")
    return result


@dataclass(frozen=True)
class DifferentialFingerPreloadCorrectionConfig:
    enabled: bool
    threshold_label: str
    source_run_id: str
    source_analysis_sha256: str
    fixed_correction_closure_rad: tuple[float, float, float]
    correction_norm_limit_source: str
    transition_steps_source: str
    stability_steps_source: str
    zero_sum_required: bool
    arm_target_fixed_during_correction: bool
    vertical_feedforward_zero_required: bool
    wrist_payload_reference_rebase_forbidden: bool
    root_reference_refresh_from_trailing_stability_window: bool
    persistent_via_fresh_h17_equilibrium: bool
    raw_sensor_hard_gate_unchanged: bool
    hard_gate_detection_delay_steps: int
    object_contact_event_truth_forbidden: bool
    second_parameter_set_allowed: bool
    formal_b_pass_claimed_from_fix_window: bool

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("H23 correction enabled must be boolean")
        if not self.enabled:
            return
        expected_text = {
            "threshold_label": THRESHOLD_LABEL,
            "source_run_id": SOURCE_RUN_ID,
            "source_analysis_sha256": SOURCE_ANALYSIS_SHA256,
            "correction_norm_limit_source": "SEQUENTIAL_PROBE_INCREMENT_RAD",
            "transition_steps_source": TRANSITION_STEPS_SOURCE,
            "stability_steps_source": STABILITY_STEPS_SOURCE,
        }
        for name, expected in expected_text.items():
            if getattr(self, name) != expected:
                raise ValueError(f"enabled H23 correction {name} must remain {expected}")
        if any(
            abs(actual - expected) > 1.0e-15
            for actual, expected in zip(
                self.fixed_correction_closure_rad,
                FIXED_CORRECTION_CLOSURE_RAD,
            )
        ):
            raise ValueError(
                "enabled H23 correction must use the single diagnostic result"
            )
        required_true = (
            "zero_sum_required",
            "arm_target_fixed_during_correction",
            "vertical_feedforward_zero_required",
            "wrist_payload_reference_rebase_forbidden",
            "root_reference_refresh_from_trailing_stability_window",
            "persistent_via_fresh_h17_equilibrium",
            "raw_sensor_hard_gate_unchanged",
            "object_contact_event_truth_forbidden",
        )
        for name in required_true:
            if getattr(self, name) is not True:
                raise ValueError(f"enabled H23 correction {name} must remain true")
        if self.hard_gate_detection_delay_steps != 0:
            raise ValueError("H23 correction hard-gate delay must remain zero")
        if self.second_parameter_set_allowed is not False:
            raise ValueError("H23 correction cannot authorize a second parameter set")
        if self.formal_b_pass_claimed_from_fix_window is not False:
            raise ValueError("H23 correction window cannot claim a formal B pass")
        if abs(sum(self.fixed_correction_closure_rad)) > 1.0e-15:
            raise ValueError("H23 correction must remain exactly zero sum")


def _default_config() -> DifferentialFingerPreloadCorrectionConfig:
    return DifferentialFingerPreloadCorrectionConfig(
        enabled=False,
        threshold_label=THRESHOLD_LABEL,
        source_run_id=SOURCE_RUN_ID,
        source_analysis_sha256=SOURCE_ANALYSIS_SHA256,
        fixed_correction_closure_rad=FIXED_CORRECTION_CLOSURE_RAD,
        correction_norm_limit_source="SEQUENTIAL_PROBE_INCREMENT_RAD",
        transition_steps_source=TRANSITION_STEPS_SOURCE,
        stability_steps_source=STABILITY_STEPS_SOURCE,
        zero_sum_required=True,
        arm_target_fixed_during_correction=True,
        vertical_feedforward_zero_required=True,
        wrist_payload_reference_rebase_forbidden=True,
        root_reference_refresh_from_trailing_stability_window=True,
        persistent_via_fresh_h17_equilibrium=True,
        raw_sensor_hard_gate_unchanged=True,
        hard_gate_detection_delay_steps=0,
        object_contact_event_truth_forbidden=True,
        second_parameter_set_allowed=False,
        formal_b_pass_claimed_from_fix_window=False,
    )


def load_differential_finger_preload_correction_config(
    value: Any,
) -> DifferentialFingerPreloadCorrectionConfig:
    """Load the one-shot H23 correction; historical contracts default off."""

    if value is None:
        return _default_config()
    if not isinstance(value, Mapping):
        raise ValueError("pre_lift_differential_finger_preload_correction must be a mapping")
    unknown = sorted(set(value) - set(CONFIG_KEYS))
    missing = sorted(set(CONFIG_KEYS) - set(value))
    if unknown or missing:
        raise ValueError(
            "pre_lift_differential_finger_preload_correction has unknown keys "
            f"{unknown} and/or missing keys {missing}"
        )
    return DifferentialFingerPreloadCorrectionConfig(
        enabled=value["enabled"],
        threshold_label=str(value["threshold_label"]),
        source_run_id=str(value["source_run_id"]),
        source_analysis_sha256=str(value["source_analysis_sha256"]),
        fixed_correction_closure_rad=_three_finite(
            value["fixed_correction_closure_rad"],
            "H23 fixed correction",
        ),
        correction_norm_limit_source=str(value["correction_norm_limit_source"]),
        transition_steps_source=str(value["transition_steps_source"]),
        stability_steps_source=str(value["stability_steps_source"]),
        zero_sum_required=value["zero_sum_required"],
        arm_target_fixed_during_correction=value[
            "arm_target_fixed_during_correction"
        ],
        vertical_feedforward_zero_required=value[
            "vertical_feedforward_zero_required"
        ],
        wrist_payload_reference_rebase_forbidden=value[
            "wrist_payload_reference_rebase_forbidden"
        ],
        root_reference_refresh_from_trailing_stability_window=value[
            "root_reference_refresh_from_trailing_stability_window"
        ],
        persistent_via_fresh_h17_equilibrium=value[
            "persistent_via_fresh_h17_equilibrium"
        ],
        raw_sensor_hard_gate_unchanged=value["raw_sensor_hard_gate_unchanged"],
        hard_gate_detection_delay_steps=value["hard_gate_detection_delay_steps"],
        object_contact_event_truth_forbidden=value[
            "object_contact_event_truth_forbidden"
        ],
        second_parameter_set_allowed=value["second_parameter_set_allowed"],
        formal_b_pass_claimed_from_fix_window=value[
            "formal_b_pass_claimed_from_fix_window"
        ],
    )


def derive_fixed_correction_contract(
    config: DifferentialFingerPreloadCorrectionConfig,
    sequential: SequentialGraspConfig,
    *,
    transition_steps: int,
    stability_steps: int,
) -> dict[str, Any]:
    """Bind the fixed correction to existing step and norm limits."""

    if not config.enabled:
        raise ValueError("H23 fixed correction requires enabled config")
    if type(transition_steps) is not int or transition_steps <= 0:
        raise ValueError("H23 correction transition steps must be positive")
    if type(stability_steps) is not int or stability_steps <= 0:
        raise ValueError("H23 correction stability steps must be positive")
    correction = config.fixed_correction_closure_rad
    norm = math.sqrt(sum(value * value for value in correction))
    limit = float(sequential.probe_increment_rad)
    if norm > limit + 1.0e-15:
        raise ValueError("H23 correction exceeds the inherited probe norm limit")
    return {
        "source_run_id": config.source_run_id,
        "source_analysis_sha256": config.source_analysis_sha256,
        "fixed_correction_closure_rad": list(correction),
        "correction_sum_rad": float(sum(correction)),
        "correction_norm_rad": norm,
        "correction_norm_limit_rad": limit,
        "correction_norm_limit_source": config.correction_norm_limit_source,
        "transition_steps": transition_steps,
        "transition_steps_source": config.transition_steps_source,
        "stability_steps": stability_steps,
        "stability_steps_source": config.stability_steps_source,
        "wrist_payload_reference_rebase_forbidden": True,
        "root_reference_refresh_from_trailing_stability_window": True,
        "persistent_via_fresh_h17_equilibrium": True,
        "raw_sensor_hard_gate_unchanged": True,
        "hard_gate_detection_delay_steps": 0,
        "second_parameter_set_allowed": False,
        "formal_b_pass_claimed_from_fix_window": False,
    }


def correction_step(
    step_index: int,
    transition_steps: int,
    fixed_correction_closure_rad: Sequence[float],
) -> dict[str, Any]:
    """Return one minimum-jerk point of the fixed zero-sum correction."""

    if type(step_index) is not int or type(transition_steps) is not int:
        raise ValueError("H23 correction indices must be integers")
    if transition_steps <= 0 or not 0 <= step_index < transition_steps:
        raise ValueError("H23 correction step is outside its transition")
    correction = _three_finite(
        fixed_correction_closure_rad,
        "H23 fixed correction",
    )
    fraction = float(step_index + 1) / float(transition_steps)
    blend = minimum_jerk_blend(fraction)
    applied = tuple(blend * value for value in correction)
    return {
        "step_index": step_index,
        "transition_steps": transition_steps,
        "minimum_jerk_blend": blend,
        "applied_correction_closure_rad": list(applied),
        "applied_correction_sum_rad": float(sum(applied)),
        "complete": step_index == transition_steps - 1,
    }


def apply_closure_correction_to_targets(
    base_targets_rad: Sequence[float],
    open_targets_rad: Sequence[float],
    closed_targets_rad: Sequence[float],
    correction_closure_rad: Sequence[float],
) -> dict[str, Any]:
    """Apply one closure-coordinate correction and fail outside joint bounds."""

    base = _three_finite(base_targets_rad, "H23 base targets")
    opened = _three_finite(open_targets_rad, "H23 open targets")
    closed = _three_finite(closed_targets_rad, "H23 closed targets")
    correction = _three_finite(
        correction_closure_rad,
        "H23 closure correction",
    )
    direction = tuple(
        1.0 if close > open_ else -1.0
        for open_, close in zip(opened, closed)
    )
    if any(open_ == close for open_, close in zip(opened, closed)):
        raise ValueError("H23 correction requires a closure direction")
    targets = tuple(
        value + sign * delta
        for value, sign, delta in zip(base, direction, correction)
    )
    inside = all(
        min(open_, close) <= target <= max(open_, close)
        for target, open_, close in zip(targets, opened, closed)
    )
    if not inside:
        raise ValueError("H23 fixed correction would exceed a frozen finger bound")
    return {
        "base_targets_rad": list(base),
        "closure_directions": list(direction),
        "correction_closure_rad": list(correction),
        "corrected_targets_rad": list(targets),
        "correction_sum_rad": float(sum(correction)),
        "inside_frozen_open_closed_bounds": True,
    }


__all__ = [
    "DifferentialFingerPreloadCorrectionConfig",
    "FIXED_CORRECTION_CLOSURE_RAD",
    "SOURCE_ANALYSIS_SHA256",
    "SOURCE_RUN_ID",
    "apply_closure_correction_to_targets",
    "correction_step",
    "derive_fixed_correction_contract",
    "load_differential_finger_preload_correction_config",
]
