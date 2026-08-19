"""Post-H23 fixed finger-target ownership for the H25 unload hypothesis."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence


THRESHOLD_LABEL = "SIM_TUNING_ONLY_B_V2_H25"
SOURCE_H24_RUN_ID = "B-V2-H24-TWO-SAMPLE-ROOT-LOAD-01"
SOURCE_DERIVATION_SHA256 = (
    "1ef9abedea61d0eb529cb3282a2f3ef6152aa461f2d81cfa27c7dd12c68c9051"
)
CONFIG_KEYS = (
    "enabled",
    "threshold_label",
    "source_h24_run_id",
    "source_derivation_sha256",
    "activate_after_h23_stability_window",
    "freeze_current_finger_targets",
    "disable_h17_updates_during_vertical_force_ramp",
    "disable_h17_updates_during_staged_lift",
    "raw_finger_root_samples_recorded",
    "raw_sensor_hard_gate_unchanged",
    "hard_gate_detection_delay_steps",
    "no_new_numeric_control_parameter",
    "object_contact_event_truth_forbidden",
    "formal_b_pass_claimed_from_hold",
)


@dataclass(frozen=True)
class LiftFingerFixedTargetHoldConfig:
    enabled: bool
    threshold_label: str
    source_h24_run_id: str
    source_derivation_sha256: str
    activate_after_h23_stability_window: bool
    freeze_current_finger_targets: bool
    disable_h17_updates_during_vertical_force_ramp: bool
    disable_h17_updates_during_staged_lift: bool
    raw_finger_root_samples_recorded: bool
    raw_sensor_hard_gate_unchanged: bool
    hard_gate_detection_delay_steps: int
    no_new_numeric_control_parameter: bool
    object_contact_event_truth_forbidden: bool
    formal_b_pass_claimed_from_hold: bool

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("H25 enabled must be boolean")
        if self.threshold_label != THRESHOLD_LABEL:
            raise ValueError(f"H25 threshold_label must remain {THRESHOLD_LABEL}")
        if self.source_h24_run_id != SOURCE_H24_RUN_ID:
            raise ValueError("H25 source_h24_run_id must remain frozen")
        if self.source_derivation_sha256 != SOURCE_DERIVATION_SHA256:
            raise ValueError("H25 source derivation SHA-256 must remain frozen")
        for name in (
            "activate_after_h23_stability_window",
            "freeze_current_finger_targets",
            "disable_h17_updates_during_vertical_force_ramp",
            "disable_h17_updates_during_staged_lift",
            "raw_finger_root_samples_recorded",
            "raw_sensor_hard_gate_unchanged",
            "no_new_numeric_control_parameter",
            "object_contact_event_truth_forbidden",
        ):
            if getattr(self, name) is not True:
                raise ValueError(f"H25 {name} must remain true")
        if type(self.hard_gate_detection_delay_steps) is not int:
            raise ValueError("H25 hard_gate_detection_delay_steps must be integer")
        if self.hard_gate_detection_delay_steps != 0:
            raise ValueError("H25 raw hard-gate detection delay must remain zero")
        if self.formal_b_pass_claimed_from_hold is not False:
            raise ValueError("H25 fixed hold cannot claim formal B pass by itself")


def _defaults() -> dict[str, Any]:
    return {
        "enabled": False,
        "threshold_label": THRESHOLD_LABEL,
        "source_h24_run_id": SOURCE_H24_RUN_ID,
        "source_derivation_sha256": SOURCE_DERIVATION_SHA256,
        "activate_after_h23_stability_window": True,
        "freeze_current_finger_targets": True,
        "disable_h17_updates_during_vertical_force_ramp": True,
        "disable_h17_updates_during_staged_lift": True,
        "raw_finger_root_samples_recorded": True,
        "raw_sensor_hard_gate_unchanged": True,
        "hard_gate_detection_delay_steps": 0,
        "no_new_numeric_control_parameter": True,
        "object_contact_event_truth_forbidden": True,
        "formal_b_pass_claimed_from_hold": False,
    }


def load_lift_finger_fixed_target_hold_config(
    value: Any,
) -> LiftFingerFixedTargetHoldConfig:
    """Load the exact H25 contract; historical configs default disabled."""

    defaults = _defaults()
    if value is None:
        return LiftFingerFixedTargetHoldConfig(**defaults)
    if not isinstance(value, Mapping):
        raise ValueError("lift_finger_fixed_target_hold must be a mapping")
    unknown = sorted(set(value) - set(CONFIG_KEYS))
    missing = sorted(set(CONFIG_KEYS) - set(value))
    if unknown or missing:
        raise ValueError(
            "lift_finger_fixed_target_hold has unknown keys "
            f"{unknown} and/or missing keys {missing}"
        )
    return LiftFingerFixedTargetHoldConfig(
        **{key: value[key] for key in CONFIG_KEYS}
    )


def _three_finite(
    values: Sequence[float], label: str
) -> tuple[float, float, float]:
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ValueError(f"H25 {label} must contain exactly three values")
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"H25 {label} must contain only finite values")
    return result  # type: ignore[return-value]


class LiftFingerFixedTargetHold:
    """Own unchanged post-H23 targets without closing a load-control loop."""

    def __init__(
        self,
        config: LiftFingerFixedTargetHoldConfig,
        frozen_targets_rad: Sequence[float],
    ) -> None:
        if not config.enabled:
            raise ValueError("H25 controller requires an enabled config")
        self.config = config
        self.frozen_targets_rad = _three_finite(
            frozen_targets_rad, "frozen targets"
        )
        self.update_count = 0
        self.maximum_abs_target_error_rad = 0.0

    def update(
        self,
        current_targets_rad: Sequence[float],
        raw_finger_root_torque_nm: Sequence[float],
        input_global_step: int,
        phase: str,
        phase_step: int,
    ) -> dict[str, Any]:
        current = _three_finite(current_targets_rad, "current targets")
        raw = _three_finite(
            raw_finger_root_torque_nm, "raw finger-root torque"
        )
        if type(input_global_step) is not int or input_global_step < 0:
            raise ValueError("H25 input_global_step must be non-negative integer")
        if type(phase_step) is not int or phase_step < 0:
            raise ValueError("H25 phase_step must be non-negative integer")
        if not isinstance(phase, str) or not phase:
            raise ValueError("H25 phase must be non-empty text")
        target_error = max(
            abs(current[index] - self.frozen_targets_rad[index])
            for index in range(3)
        )
        self.maximum_abs_target_error_rad = max(
            self.maximum_abs_target_error_rad, target_error
        )
        if target_error != 0.0:
            raise ValueError("H25 current finger targets drifted from frozen targets")
        self.update_count += 1
        return {
            "input_global_step": input_global_step,
            "phase": phase,
            "phase_step": phase_step,
            "frozen_targets_rad": list(self.frozen_targets_rad),
            "input_targets_rad": list(current),
            "output_targets_rad": list(self.frozen_targets_rad),
            "maximum_abs_target_error_rad": target_error,
            "raw_finger_root_torque_nm": list(raw),
            "h17_update_executed": False,
            "finger_target_modified": False,
            "new_numeric_control_parameter": False,
            "raw_sensor_hard_gate_unchanged": True,
            "hard_gate_detection_delay_steps": 0,
            "object_truth_used": False,
            "contact_truth_used": False,
            "event_truth_used": False,
            "object_pose_written": False,
            "formal_b_pass_claimed": False,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "update_count": self.update_count,
            "frozen_targets_rad": list(self.frozen_targets_rad),
            "maximum_abs_target_error_rad": (
                self.maximum_abs_target_error_rad
            ),
            "h17_updates_executed": 0,
            "finger_target_modified": False,
            "new_numeric_control_parameter": False,
        }
