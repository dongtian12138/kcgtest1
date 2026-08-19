"""H21 two-sample suppression for the pre-lift XY control input only."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence


THRESHOLD_LABEL = "SIM_TUNING_ONLY_B_V2_H21"
CONFIG_KEYS = (
    "enabled",
    "threshold_label",
    "source_h20_run_id",
    "two_sample_arithmetic_mean_required",
    "xy_control_input_only",
    "raw_sensor_hard_gate_unchanged",
    "raw_peaks_recorded",
    "hard_gate_detection_delay_steps",
    "no_new_gain_force_or_geometry_parameter",
)


@dataclass(frozen=True)
class PreLiftXYNyquistSuppressionConfig:
    enabled: bool
    threshold_label: str
    source_h20_run_id: str
    two_sample_arithmetic_mean_required: bool
    xy_control_input_only: bool
    raw_sensor_hard_gate_unchanged: bool
    raw_peaks_recorded: bool
    hard_gate_detection_delay_steps: int
    no_new_gain_force_or_geometry_parameter: bool

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("H21 enabled must be boolean")
        if self.threshold_label != THRESHOLD_LABEL:
            raise ValueError(f"H21 threshold_label must remain {THRESHOLD_LABEL}")
        if self.source_h20_run_id != "B-V2-H20-PRELIFT-XY-ADMITTANCE-01":
            raise ValueError("H21 source_h20_run_id must remain frozen")
        for name in (
            "two_sample_arithmetic_mean_required",
            "xy_control_input_only",
            "raw_sensor_hard_gate_unchanged",
            "raw_peaks_recorded",
            "no_new_gain_force_or_geometry_parameter",
        ):
            if getattr(self, name) is not True:
                raise ValueError(f"H21 {name} must remain true")
        if type(self.hard_gate_detection_delay_steps) is not int:
            raise ValueError("H21 hard_gate_detection_delay_steps must be integer")
        if self.hard_gate_detection_delay_steps != 0:
            raise ValueError("H21 raw hard-gate detection delay must remain zero")


def load_pre_lift_xy_nyquist_suppression_config(
    value: Any,
) -> PreLiftXYNyquistSuppressionConfig:
    defaults = {
        "enabled": False,
        "threshold_label": THRESHOLD_LABEL,
        "source_h20_run_id": "B-V2-H20-PRELIFT-XY-ADMITTANCE-01",
        "two_sample_arithmetic_mean_required": True,
        "xy_control_input_only": True,
        "raw_sensor_hard_gate_unchanged": True,
        "raw_peaks_recorded": True,
        "hard_gate_detection_delay_steps": 0,
        "no_new_gain_force_or_geometry_parameter": True,
    }
    if value is None:
        return PreLiftXYNyquistSuppressionConfig(**defaults)
    if not isinstance(value, Mapping):
        raise ValueError("pre_lift_xy_nyquist_suppression must be a mapping")
    unknown = sorted(set(value) - set(CONFIG_KEYS))
    missing = sorted(set(CONFIG_KEYS) - set(value))
    if unknown or missing:
        raise ValueError(
            "pre_lift_xy_nyquist_suppression has unknown keys "
            f"{unknown} and/or missing keys {missing}"
        )
    return PreLiftXYNyquistSuppressionConfig(
        enabled=value["enabled"],
        threshold_label=str(value["threshold_label"]),
        source_h20_run_id=str(value["source_h20_run_id"]),
        two_sample_arithmetic_mean_required=(
            value["two_sample_arithmetic_mean_required"]
        ),
        xy_control_input_only=value["xy_control_input_only"],
        raw_sensor_hard_gate_unchanged=(
            value["raw_sensor_hard_gate_unchanged"]
        ),
        raw_peaks_recorded=value["raw_peaks_recorded"],
        hard_gate_detection_delay_steps=(
            value["hard_gate_detection_delay_steps"]
        ),
        no_new_gain_force_or_geometry_parameter=(
            value["no_new_gain_force_or_geometry_parameter"]
        ),
    )


def two_sample_xy_control_input(
    current_raw_xy_n: Sequence[float],
    previous_raw_xy_n: Sequence[float] | None,
) -> dict[str, Any]:
    """Average the two latest allowed XY samples without touching hard gates."""

    if not isinstance(current_raw_xy_n, (list, tuple)) or len(current_raw_xy_n) != 2:
        raise ValueError("H21 current raw XY input must contain two values")
    current = tuple(float(value) for value in current_raw_xy_n)
    if not all(math.isfinite(value) for value in current):
        raise ValueError("H21 current raw XY input must be finite")
    initialized_by_duplication = previous_raw_xy_n is None
    if initialized_by_duplication:
        previous = current
    else:
        if (
            not isinstance(previous_raw_xy_n, (list, tuple))
            or len(previous_raw_xy_n) != 2
        ):
            raise ValueError("H21 previous raw XY input must contain two values")
        previous = tuple(float(value) for value in previous_raw_xy_n)
        if not all(math.isfinite(value) for value in previous):
            raise ValueError("H21 previous raw XY input must be finite")
    filtered = tuple(0.5 * (current[i] + previous[i]) for i in range(2))
    return {
        "raw_current_task_force_xy_n": list(current),
        "raw_previous_task_force_xy_n": list(previous),
        "filtered_control_task_force_xy_n": list(filtered),
        "initialized_by_current_sample_duplication": initialized_by_duplication,
        "control_filter_sample_count": 2,
        "oldest_control_sample_age_steps": (
            0 if initialized_by_duplication else 1
        ),
        "linear_phase_group_delay_steps": (
            0.0 if initialized_by_duplication else 0.5
        ),
        "nyquist_alternating_component_gain": 0.0,
        "xy_control_input_only": True,
        "raw_sensor_hard_gate_unchanged": True,
        "raw_peaks_recorded": True,
        "hard_gate_detection_delay_steps": 0,
    }


__all__ = [
    "PreLiftXYNyquistSuppressionConfig",
    "load_pre_lift_xy_nyquist_suppression_config",
    "two_sample_xy_control_input",
]
