"""B-V2-H19 sensor-origin-collocated world-up force ramp."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any, Callable, Mapping, Sequence

import numpy as np


THRESHOLD_LABEL = "SIM_TUNING_ONLY_B_V2_H19"
APPLICATION_FRAME = "handbase_link_sensor_origin"
HARD_GATE_REFERENCE_FRAME = "handbase_link_sensor_origin"
CONFIG_KEYS = (
    "enabled",
    "threshold_label",
    "source_h12_run_id",
    "source_h13_run_id",
    "source_h17_run_id",
    "source_h18_run_id",
    "force_target_source",
    "ramp_steps_source",
    "world_force_axis",
    "force_application_frame",
    "hard_gate_reference_frame",
    "handbase_to_tcp_source",
    "keep_position_target_fixed_during_ramp",
    "immediately_preceding_arm_state_only",
    "recompute_mapping_during_staged_lift",
)


@dataclass(frozen=True)
class LiftSensorOriginVerticalForceRampConfig:
    """Strict H19 authorization; all numeric action values remain inherited."""

    enabled: bool
    threshold_label: str
    source_h12_run_id: str
    source_h13_run_id: str
    source_h17_run_id: str
    source_h18_run_id: str
    force_target_source: str
    ramp_steps_source: str
    world_force_axis: tuple[float, float, float]
    force_application_frame: str
    hard_gate_reference_frame: str
    handbase_to_tcp_source: str
    keep_position_target_fixed_during_ramp: bool
    immediately_preceding_arm_state_only: bool
    recompute_mapping_during_staged_lift: bool

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("H19 enabled must be boolean")
        expected_text = {
            "threshold_label": THRESHOLD_LABEL,
            "source_h12_run_id": "B-V2-H12-ZERO-LIFT-01",
            "source_h13_run_id": "B-V2-GRASP-14",
            "source_h17_run_id": "B-V2-H17-ABSOLUTE-LOAD-HOLD-01",
            "source_h18_run_id": "B-V2-H18-VERTICAL-FORCE-RAMP-01",
            "force_target_source": "FROZEN_BODY_PLUS_NUT_WEIGHT",
            "ramp_steps_source": "H13_TRANSITION_STEPS",
            "force_application_frame": APPLICATION_FRAME,
            "hard_gate_reference_frame": HARD_GATE_REFERENCE_FRAME,
            "handbase_to_tcp_source": "FROZEN_PICK_GEOMETRY_CANDIDATE",
        }
        for name, expected in expected_text.items():
            if getattr(self, name) != expected:
                raise ValueError(f"H19 {name} must remain {expected}")
        if tuple(self.world_force_axis) != (0.0, 0.0, 1.0):
            raise ValueError("H19 force axis must remain world +Z")
        for name in (
            "keep_position_target_fixed_during_ramp",
            "immediately_preceding_arm_state_only",
            "recompute_mapping_during_staged_lift",
        ):
            if getattr(self, name) is not True:
                raise ValueError(f"H19 {name} must remain true")


def load_lift_sensor_origin_vertical_force_ramp_config(
    value: Any,
) -> LiftSensorOriginVerticalForceRampConfig:
    """Load H19 strictly while older contracts default disabled."""

    defaults = {
        "enabled": False,
        "threshold_label": THRESHOLD_LABEL,
        "source_h12_run_id": "B-V2-H12-ZERO-LIFT-01",
        "source_h13_run_id": "B-V2-GRASP-14",
        "source_h17_run_id": "B-V2-H17-ABSOLUTE-LOAD-HOLD-01",
        "source_h18_run_id": "B-V2-H18-VERTICAL-FORCE-RAMP-01",
        "force_target_source": "FROZEN_BODY_PLUS_NUT_WEIGHT",
        "ramp_steps_source": "H13_TRANSITION_STEPS",
        "world_force_axis": (0.0, 0.0, 1.0),
        "force_application_frame": APPLICATION_FRAME,
        "hard_gate_reference_frame": HARD_GATE_REFERENCE_FRAME,
        "handbase_to_tcp_source": "FROZEN_PICK_GEOMETRY_CANDIDATE",
        "keep_position_target_fixed_during_ramp": True,
        "immediately_preceding_arm_state_only": True,
        "recompute_mapping_during_staged_lift": True,
    }
    if value is None:
        return LiftSensorOriginVerticalForceRampConfig(**defaults)
    if not isinstance(value, Mapping):
        raise ValueError(
            "lift_sensor_origin_vertical_force_ramp must be a mapping"
        )
    unknown = sorted(set(value) - set(CONFIG_KEYS))
    missing = sorted(set(CONFIG_KEYS) - set(value))
    if unknown or missing:
        raise ValueError(
            "lift_sensor_origin_vertical_force_ramp has unknown keys "
            f"{unknown} and/or missing keys {missing}"
        )
    axis = value["world_force_axis"]
    if not isinstance(axis, (list, tuple)) or len(axis) != 3:
        raise ValueError("H19 world_force_axis must contain three values")
    return LiftSensorOriginVerticalForceRampConfig(
        enabled=value["enabled"],
        threshold_label=str(value["threshold_label"]),
        source_h12_run_id=str(value["source_h12_run_id"]),
        source_h13_run_id=str(value["source_h13_run_id"]),
        source_h17_run_id=str(value["source_h17_run_id"]),
        source_h18_run_id=str(value["source_h18_run_id"]),
        force_target_source=str(value["force_target_source"]),
        ramp_steps_source=str(value["ramp_steps_source"]),
        world_force_axis=tuple(float(entry) for entry in axis),
        force_application_frame=str(value["force_application_frame"]),
        hard_gate_reference_frame=str(value["hard_gate_reference_frame"]),
        handbase_to_tcp_source=str(value["handbase_to_tcp_source"]),
        keep_position_target_fixed_during_ramp=(
            value["keep_position_target_fixed_during_ramp"]
        ),
        immediately_preceding_arm_state_only=(
            value["immediately_preceding_arm_state_only"]
        ),
        recompute_mapping_during_staged_lift=(
            value["recompute_mapping_during_staged_lift"]
        ),
    )


def sensor_origin_from_grasp_tcp_transform(
    arm_rad: Sequence[Real],
    handbase_to_tcp_m: Real,
    grasp_tcp_forward_kinematics: Callable[
        [tuple[float, ...]], Sequence[Sequence[Real]]
    ],
) -> tuple[tuple[float, ...], ...]:
    """Return world handbase/sensor transform from the frozen grasp-TCP FK."""

    offset = float(handbase_to_tcp_m)
    if not math.isfinite(offset) or offset <= 0.0:
        raise ValueError("H19 handbase_to_tcp_m must be positive and finite")
    arm = tuple(float(value) for value in arm_rad)
    if len(arm) != 7 or not all(math.isfinite(value) for value in arm):
        raise ValueError("H19 arm_rad must contain seven finite values")
    world_tcp = np.asarray(
        grasp_tcp_forward_kinematics(arm), dtype=np.float64
    )
    if world_tcp.shape != (4, 4) or not np.all(np.isfinite(world_tcp)):
        raise ValueError("H19 grasp TCP FK must return one finite 4x4 transform")
    tcp_from_sensor_origin = np.eye(4, dtype=np.float64)
    tcp_from_sensor_origin[2, 3] = -offset
    world_sensor_origin = world_tcp @ tcp_from_sensor_origin
    if not np.all(np.isfinite(world_sensor_origin)):
        raise ValueError("H19 sensor-origin transform is non-finite")
    return tuple(
        tuple(float(value) for value in row)
        for row in world_sensor_origin
    )


__all__ = [
    "APPLICATION_FRAME",
    "HARD_GATE_REFERENCE_FRAME",
    "LiftSensorOriginVerticalForceRampConfig",
    "load_lift_sensor_origin_vertical_force_ramp_config",
    "sensor_origin_from_grasp_tcp_transform",
]
