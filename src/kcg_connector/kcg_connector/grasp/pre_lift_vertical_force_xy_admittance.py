"""B-V2-H20 authorization for H14 XY admittance during the H18 ramp."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


THRESHOLD_LABEL = "SIM_TUNING_ONLY_B_V2_H20"
CONFIG_KEYS = (
    "enabled",
    "threshold_label",
    "source_h14_run_id",
    "source_h18_run_id",
    "reuse_h14_xy_parameters",
    "reuse_h18_force_profile",
    "immediately_preceding_task_wrench_only",
    "vertical_position_target_fixed",
    "sensor_origin_hard_gate_unchanged",
    "no_new_numeric_control_parameter",
)


@dataclass(frozen=True)
class PreLiftVerticalForceXYAdmittanceConfig:
    enabled: bool
    threshold_label: str
    source_h14_run_id: str
    source_h18_run_id: str
    reuse_h14_xy_parameters: bool
    reuse_h18_force_profile: bool
    immediately_preceding_task_wrench_only: bool
    vertical_position_target_fixed: bool
    sensor_origin_hard_gate_unchanged: bool
    no_new_numeric_control_parameter: bool

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("H20 enabled must be boolean")
        expected_text = {
            "threshold_label": THRESHOLD_LABEL,
            "source_h14_run_id": "B-V2-GRASP-15-IFIX01",
            "source_h18_run_id": "B-V2-H18-VERTICAL-FORCE-RAMP-01",
        }
        for name, expected in expected_text.items():
            if getattr(self, name) != expected:
                raise ValueError(f"H20 {name} must remain {expected}")
        for name in (
            "reuse_h14_xy_parameters",
            "reuse_h18_force_profile",
            "immediately_preceding_task_wrench_only",
            "vertical_position_target_fixed",
            "sensor_origin_hard_gate_unchanged",
            "no_new_numeric_control_parameter",
        ):
            if getattr(self, name) is not True:
                raise ValueError(f"H20 {name} must remain true")


def load_pre_lift_vertical_force_xy_admittance_config(
    value: Any,
) -> PreLiftVerticalForceXYAdmittanceConfig:
    defaults = {
        "enabled": False,
        "threshold_label": THRESHOLD_LABEL,
        "source_h14_run_id": "B-V2-GRASP-15-IFIX01",
        "source_h18_run_id": "B-V2-H18-VERTICAL-FORCE-RAMP-01",
        "reuse_h14_xy_parameters": True,
        "reuse_h18_force_profile": True,
        "immediately_preceding_task_wrench_only": True,
        "vertical_position_target_fixed": True,
        "sensor_origin_hard_gate_unchanged": True,
        "no_new_numeric_control_parameter": True,
    }
    if value is None:
        return PreLiftVerticalForceXYAdmittanceConfig(**defaults)
    if not isinstance(value, Mapping):
        raise ValueError("pre_lift_vertical_force_xy_admittance must be a mapping")
    unknown = sorted(set(value) - set(CONFIG_KEYS))
    missing = sorted(set(CONFIG_KEYS) - set(value))
    if unknown or missing:
        raise ValueError(
            "pre_lift_vertical_force_xy_admittance has unknown keys "
            f"{unknown} and/or missing keys {missing}"
        )
    return PreLiftVerticalForceXYAdmittanceConfig(
        enabled=value["enabled"],
        threshold_label=str(value["threshold_label"]),
        source_h14_run_id=str(value["source_h14_run_id"]),
        source_h18_run_id=str(value["source_h18_run_id"]),
        reuse_h14_xy_parameters=value["reuse_h14_xy_parameters"],
        reuse_h18_force_profile=value["reuse_h18_force_profile"],
        immediately_preceding_task_wrench_only=(
            value["immediately_preceding_task_wrench_only"]
        ),
        vertical_position_target_fixed=value["vertical_position_target_fixed"],
        sensor_origin_hard_gate_unchanged=(
            value["sensor_origin_hard_gate_unchanged"]
        ),
        no_new_numeric_control_parameter=(
            value["no_new_numeric_control_parameter"]
        ),
    )


__all__ = [
    "PreLiftVerticalForceXYAdmittanceConfig",
    "load_pre_lift_vertical_force_xy_admittance_config",
]
