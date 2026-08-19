"""Fixed two-sample H17 root-load control-input suppression for H24."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence


THRESHOLD_LABEL = "SIM_TUNING_ONLY_B_V2_H24"
SOURCE_H22_RUN_ID = "B-V2-H22-PREUNLOADING-NULLING-01"
SOURCE_H23_RUN_ID = "B-V2-H23-DIFFERENTIAL-FINGER-PRELOAD-CORRECTION-01"
SOURCE_DERIVATION_SHA256 = (
    "b2d12112af3241dfe3301f9de1aac6e827f84874a1f0dbcc5229c6f0675a9b55"
)
CONFIG_KEYS = (
    "enabled",
    "threshold_label",
    "source_h22_run_id",
    "source_h23_run_id",
    "source_derivation_sha256",
    "two_sample_arithmetic_mean_required",
    "h17_control_input_only",
    "initialize_by_current_sample_duplication",
    "reset_after_h17_reference_refresh",
    "raw_finger_root_samples_recorded",
    "raw_sensor_hard_gate_unchanged",
    "hard_gate_detection_delay_steps",
    "no_new_gain_force_or_geometry_parameter",
)


@dataclass(frozen=True)
class LiftFingerRootLoadTwoSampleSuppressionConfig:
    enabled: bool
    threshold_label: str
    source_h22_run_id: str
    source_h23_run_id: str
    source_derivation_sha256: str
    two_sample_arithmetic_mean_required: bool
    h17_control_input_only: bool
    initialize_by_current_sample_duplication: bool
    reset_after_h17_reference_refresh: bool
    raw_finger_root_samples_recorded: bool
    raw_sensor_hard_gate_unchanged: bool
    hard_gate_detection_delay_steps: int
    no_new_gain_force_or_geometry_parameter: bool

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("H24 enabled must be boolean")
        if self.threshold_label != THRESHOLD_LABEL:
            raise ValueError(f"H24 threshold_label must remain {THRESHOLD_LABEL}")
        if self.source_h22_run_id != SOURCE_H22_RUN_ID:
            raise ValueError("H24 source_h22_run_id must remain frozen")
        if self.source_h23_run_id != SOURCE_H23_RUN_ID:
            raise ValueError("H24 source_h23_run_id must remain frozen")
        if self.source_derivation_sha256 != SOURCE_DERIVATION_SHA256:
            raise ValueError("H24 source derivation SHA-256 must remain frozen")
        for name in (
            "two_sample_arithmetic_mean_required",
            "h17_control_input_only",
            "initialize_by_current_sample_duplication",
            "reset_after_h17_reference_refresh",
            "raw_finger_root_samples_recorded",
            "raw_sensor_hard_gate_unchanged",
            "no_new_gain_force_or_geometry_parameter",
        ):
            if getattr(self, name) is not True:
                raise ValueError(f"H24 {name} must remain true")
        if type(self.hard_gate_detection_delay_steps) is not int:
            raise ValueError("H24 hard_gate_detection_delay_steps must be integer")
        if self.hard_gate_detection_delay_steps != 0:
            raise ValueError("H24 raw hard-gate detection delay must remain zero")


def _defaults() -> dict[str, Any]:
    return {
        "enabled": False,
        "threshold_label": THRESHOLD_LABEL,
        "source_h22_run_id": SOURCE_H22_RUN_ID,
        "source_h23_run_id": SOURCE_H23_RUN_ID,
        "source_derivation_sha256": SOURCE_DERIVATION_SHA256,
        "two_sample_arithmetic_mean_required": True,
        "h17_control_input_only": True,
        "initialize_by_current_sample_duplication": True,
        "reset_after_h17_reference_refresh": True,
        "raw_finger_root_samples_recorded": True,
        "raw_sensor_hard_gate_unchanged": True,
        "hard_gate_detection_delay_steps": 0,
        "no_new_gain_force_or_geometry_parameter": True,
    }


def load_lift_finger_root_load_two_sample_suppression_config(
    value: Any,
) -> LiftFingerRootLoadTwoSampleSuppressionConfig:
    """Load the exact H24 contract; historical configs default disabled."""

    defaults = _defaults()
    if value is None:
        return LiftFingerRootLoadTwoSampleSuppressionConfig(**defaults)
    if not isinstance(value, Mapping):
        raise ValueError(
            "lift_finger_root_load_two_sample_suppression must be a mapping"
        )
    unknown = sorted(set(value) - set(CONFIG_KEYS))
    missing = sorted(set(CONFIG_KEYS) - set(value))
    if unknown or missing:
        raise ValueError(
            "lift_finger_root_load_two_sample_suppression has unknown keys "
            f"{unknown} and/or missing keys {missing}"
        )
    return LiftFingerRootLoadTwoSampleSuppressionConfig(
        **{key: value[key] for key in CONFIG_KEYS}
    )


def _three_finite(
    values: Sequence[float], label: str
) -> tuple[float, float, float]:
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ValueError(f"H24 {label} must contain exactly three values")
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"H24 {label} must contain only finite values")
    return result  # type: ignore[return-value]


def two_sample_root_load_control_input(
    current_raw_root_torque_nm: Sequence[float],
    previous_raw_root_torque_nm: Sequence[float] | None,
) -> dict[str, Any]:
    """Average two allowed signed root-load samples without touching safety."""

    current = _three_finite(
        current_raw_root_torque_nm, "current raw root torque"
    )
    initialized = previous_raw_root_torque_nm is None
    previous = (
        current
        if initialized
        else _three_finite(
            previous_raw_root_torque_nm, "previous raw root torque"
        )
    )
    filtered = tuple(
        0.5 * (current[index] + previous[index]) for index in range(3)
    )
    return {
        "raw_current_root_torque_nm": list(current),
        "raw_previous_root_torque_nm": list(previous),
        "filtered_h17_control_root_torque_nm": list(filtered),
        "initialized_by_current_sample_duplication": initialized,
        "control_filter_sample_count": 2,
        "newest_control_sample_age_steps": 1,
        "oldest_control_sample_age_steps": 1 if initialized else 2,
        "additional_linear_phase_group_delay_steps": (
            0.0 if initialized else 0.5
        ),
        "nyquist_alternating_component_gain": 0.0,
        "h17_control_input_only": True,
        "raw_finger_root_samples_recorded": True,
        "raw_sensor_hard_gate_unchanged": True,
        "hard_gate_detection_delay_steps": 0,
        "object_truth_used": False,
        "contact_truth_used": False,
        "event_truth_used": False,
    }


class LiftFingerRootLoadTwoSampleSuppression:
    """Stateful sample-order guard around the fixed H24 transform."""

    def __init__(
        self, config: LiftFingerRootLoadTwoSampleSuppressionConfig
    ) -> None:
        if not config.enabled:
            raise ValueError("H24 suppression requires enabled config")
        self.config = config
        self.previous_raw_root_torque_nm: tuple[float, float, float] | None = None
        self.previous_input_global_step: int | None = None
        self.records = 0
        self.resets = 0
        self.maximum_raw_to_filtered_delta_nm = 0.0

    def update(
        self,
        current_raw_root_torque_nm: Sequence[float],
        current_input_global_step: int,
    ) -> dict[str, Any]:
        if (
            type(current_input_global_step) is not int
            or current_input_global_step < 0
        ):
            raise ValueError("H24 input global step must be a nonnegative integer")
        if (
            self.previous_input_global_step is not None
            and current_input_global_step
            != self.previous_input_global_step + 1
        ):
            raise ValueError("H24 root-load input steps must be consecutive")
        result = two_sample_root_load_control_input(
            current_raw_root_torque_nm,
            self.previous_raw_root_torque_nm,
        )
        current = tuple(
            float(value) for value in result["raw_current_root_torque_nm"]
        )
        filtered = tuple(
            float(value)
            for value in result["filtered_h17_control_root_torque_nm"]
        )
        result.update(
            {
                "current_input_global_step": current_input_global_step,
                "previous_input_global_step": (
                    current_input_global_step
                    if self.previous_input_global_step is None
                    else self.previous_input_global_step
                ),
                "input_steps_consecutive": True,
            }
        )
        self.maximum_raw_to_filtered_delta_nm = max(
            self.maximum_raw_to_filtered_delta_nm,
            max(abs(raw - smooth) for raw, smooth in zip(current, filtered)),
        )
        self.previous_raw_root_torque_nm = current
        self.previous_input_global_step = current_input_global_step
        self.records += 1
        return result

    def reset_after_h17_reference_refresh(self) -> None:
        self.previous_raw_root_torque_nm = None
        self.previous_input_global_step = None
        self.resets += 1

    def summary(self) -> dict[str, Any]:
        return {
            "record_count": self.records,
            "reference_refresh_reset_count": self.resets,
            "maximum_raw_to_filtered_delta_nm": (
                self.maximum_raw_to_filtered_delta_nm
            ),
            "control_filter_sample_count": 2,
            "nyquist_alternating_component_gain": 0.0,
            "h17_control_input_only": True,
            "raw_finger_root_samples_recorded": True,
            "raw_sensor_hard_gate_unchanged": True,
            "hard_gate_detection_delay_steps": 0,
        }


__all__ = [
    "LiftFingerRootLoadTwoSampleSuppression",
    "LiftFingerRootLoadTwoSampleSuppressionConfig",
    "load_lift_finger_root_load_two_sample_suppression_config",
    "two_sample_root_load_control_input",
]

