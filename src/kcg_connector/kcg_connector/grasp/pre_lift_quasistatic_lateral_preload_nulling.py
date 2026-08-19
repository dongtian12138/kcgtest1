"""H22 authorization for pre-unloading quasi-static lateral preload nulling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


THRESHOLD_LABEL = "SIM_TUNING_ONLY_B_V2_H22"
CONFIG_KEYS = (
    "enabled",
    "threshold_label",
    "source_h21_run_id",
    "steps_source",
    "zero_vertical_feedforward_required",
    "reuse_h14_xy_parameters",
    "reuse_h21_two_sample_input",
    "retain_correction_into_vertical_ramp",
    "correction_total_bound_reset_forbidden",
    "payload_reference_rebase_forbidden",
    "raw_sensor_hard_gate_unchanged",
    "hard_gate_detection_delay_steps",
    "no_new_gain_force_or_geometry_parameter",
)


@dataclass(frozen=True)
class PreLiftQuasistaticLateralPreloadNullingConfig:
    enabled: bool
    threshold_label: str
    source_h21_run_id: str
    steps_source: str
    zero_vertical_feedforward_required: bool
    reuse_h14_xy_parameters: bool
    reuse_h21_two_sample_input: bool
    retain_correction_into_vertical_ramp: bool
    correction_total_bound_reset_forbidden: bool
    payload_reference_rebase_forbidden: bool
    raw_sensor_hard_gate_unchanged: bool
    hard_gate_detection_delay_steps: int
    no_new_gain_force_or_geometry_parameter: bool

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("H22 enabled must be boolean")
        expected_text = {
            "threshold_label": THRESHOLD_LABEL,
            "source_h21_run_id": "B-V2-H21-NYQUIST-SUPPRESSION-01",
            "steps_source": "H13_TRANSITION_STEPS",
        }
        for name, expected in expected_text.items():
            if getattr(self, name) != expected:
                raise ValueError(f"H22 {name} must remain {expected}")
        for name in (
            "zero_vertical_feedforward_required",
            "reuse_h14_xy_parameters",
            "reuse_h21_two_sample_input",
            "retain_correction_into_vertical_ramp",
            "correction_total_bound_reset_forbidden",
            "payload_reference_rebase_forbidden",
            "raw_sensor_hard_gate_unchanged",
            "no_new_gain_force_or_geometry_parameter",
        ):
            if getattr(self, name) is not True:
                raise ValueError(f"H22 {name} must remain true")
        if type(self.hard_gate_detection_delay_steps) is not int:
            raise ValueError("H22 hard_gate_detection_delay_steps must be integer")
        if self.hard_gate_detection_delay_steps != 0:
            raise ValueError("H22 raw hard-gate detection delay must remain zero")


def load_pre_lift_quasistatic_lateral_preload_nulling_config(
    value: Any,
) -> PreLiftQuasistaticLateralPreloadNullingConfig:
    defaults = {
        "enabled": False,
        "threshold_label": THRESHOLD_LABEL,
        "source_h21_run_id": "B-V2-H21-NYQUIST-SUPPRESSION-01",
        "steps_source": "H13_TRANSITION_STEPS",
        "zero_vertical_feedforward_required": True,
        "reuse_h14_xy_parameters": True,
        "reuse_h21_two_sample_input": True,
        "retain_correction_into_vertical_ramp": True,
        "correction_total_bound_reset_forbidden": True,
        "payload_reference_rebase_forbidden": True,
        "raw_sensor_hard_gate_unchanged": True,
        "hard_gate_detection_delay_steps": 0,
        "no_new_gain_force_or_geometry_parameter": True,
    }
    if value is None:
        return PreLiftQuasistaticLateralPreloadNullingConfig(**defaults)
    if not isinstance(value, Mapping):
        raise ValueError(
            "pre_lift_quasistatic_lateral_preload_nulling must be a mapping"
        )
    unknown = sorted(set(value) - set(CONFIG_KEYS))
    missing = sorted(set(CONFIG_KEYS) - set(value))
    if unknown or missing:
        raise ValueError(
            "pre_lift_quasistatic_lateral_preload_nulling has unknown keys "
            f"{unknown} and/or missing keys {missing}"
        )
    return PreLiftQuasistaticLateralPreloadNullingConfig(
        enabled=value["enabled"],
        threshold_label=str(value["threshold_label"]),
        source_h21_run_id=str(value["source_h21_run_id"]),
        steps_source=str(value["steps_source"]),
        zero_vertical_feedforward_required=(
            value["zero_vertical_feedforward_required"]
        ),
        reuse_h14_xy_parameters=value["reuse_h14_xy_parameters"],
        reuse_h21_two_sample_input=value["reuse_h21_two_sample_input"],
        retain_correction_into_vertical_ramp=(
            value["retain_correction_into_vertical_ramp"]
        ),
        correction_total_bound_reset_forbidden=(
            value["correction_total_bound_reset_forbidden"]
        ),
        payload_reference_rebase_forbidden=(
            value["payload_reference_rebase_forbidden"]
        ),
        raw_sensor_hard_gate_unchanged=(
            value["raw_sensor_hard_gate_unchanged"]
        ),
        hard_gate_detection_delay_steps=(
            value["hard_gate_detection_delay_steps"]
        ),
        no_new_gain_force_or_geometry_parameter=(
            value["no_new_gain_force_or_geometry_parameter"]
        ),
    )


__all__ = [
    "PreLiftQuasistaticLateralPreloadNullingConfig",
    "load_pre_lift_quasistatic_lateral_preload_nulling_config",
]
