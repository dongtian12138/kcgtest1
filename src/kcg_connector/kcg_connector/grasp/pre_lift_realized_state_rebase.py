"""Truth-free validation for a pre-lift robot target-state rebase.

The helper consumes robot joint state and robot FK matrices only.  It never
accepts an object pose, contact identity, contact normal, or event truth.
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
    "reference_window_steps",
    "maximum_rebase_joint_delta_rad",
    "maximum_rebase_tcp_translation_m",
    "maximum_rebase_tcp_rotation_rad",
    "maximum_entry_moment_score_nm",
    "maximum_entry_load_imbalance",
)


@dataclass(frozen=True)
class PreLiftRealizedStateRebaseConfig:
    enabled: bool
    threshold_label: str
    reference_window_steps: int
    maximum_rebase_joint_delta_rad: float
    maximum_rebase_tcp_translation_m: float
    maximum_rebase_tcp_rotation_rad: float
    maximum_entry_moment_score_nm: float
    maximum_entry_load_imbalance: float

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("pre-lift realized-state rebase enabled must be boolean")
        if self.threshold_label != "SIM_TUNING_ONLY_B_V2_H5":
            raise ValueError(
                "pre-lift realized-state rebase must stay SIM_TUNING_ONLY_B_V2_H5"
            )
        if type(self.reference_window_steps) is not int or self.reference_window_steps <= 0:
            raise ValueError("reference_window_steps must be a positive integer")
        for name in (
            "maximum_rebase_joint_delta_rad",
            "maximum_rebase_tcp_translation_m",
            "maximum_rebase_tcp_rotation_rad",
            "maximum_entry_moment_score_nm",
            "maximum_entry_load_imbalance",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{name} must be numeric")
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.maximum_rebase_joint_delta_rad >= 0.03:
            raise ValueError("rebase joint delta must stay below the 0.03 rad hard gate")
        if self.maximum_entry_moment_score_nm >= 0.30:
            raise ValueError("rebase entry score must stay below the 0.30 N*m hard gate")
        if self.maximum_entry_load_imbalance > 0.18:
            raise ValueError("rebase entry load imbalance must be no greater than 0.18")


def _positive_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be positive and finite")
    return result


def load_pre_lift_realized_state_rebase_config(
    value: Any,
) -> PreLiftRealizedStateRebaseConfig:
    """Load the H5 section; absence keeps all historical contracts disabled."""

    if value is None:
        return PreLiftRealizedStateRebaseConfig(
            enabled=False,
            threshold_label="SIM_TUNING_ONLY_B_V2_H5",
            reference_window_steps=240,
            maximum_rebase_joint_delta_rad=0.005,
            maximum_rebase_tcp_translation_m=0.002,
            maximum_rebase_tcp_rotation_rad=0.01,
            maximum_entry_moment_score_nm=0.24,
            maximum_entry_load_imbalance=0.18,
        )
    if not isinstance(value, Mapping):
        raise ValueError("pre_lift_realized_state_rebase must be a mapping")
    unknown = sorted(set(value) - set(CONFIG_KEYS))
    missing = sorted(set(CONFIG_KEYS) - set(value))
    if unknown or missing:
        raise ValueError(
            "pre_lift_realized_state_rebase has unknown keys "
            f"{unknown} and/or missing keys {missing}"
        )
    if type(value["enabled"]) is not bool:
        raise ValueError("pre_lift_realized_state_rebase.enabled must be boolean")
    steps = value["reference_window_steps"]
    if type(steps) is not int or steps <= 0:
        raise ValueError("reference_window_steps must be a positive integer")
    return PreLiftRealizedStateRebaseConfig(
        enabled=value["enabled"],
        threshold_label=str(value["threshold_label"]),
        reference_window_steps=steps,
        maximum_rebase_joint_delta_rad=_positive_number(
            value["maximum_rebase_joint_delta_rad"],
            "maximum_rebase_joint_delta_rad",
        ),
        maximum_rebase_tcp_translation_m=_positive_number(
            value["maximum_rebase_tcp_translation_m"],
            "maximum_rebase_tcp_translation_m",
        ),
        maximum_rebase_tcp_rotation_rad=_positive_number(
            value["maximum_rebase_tcp_rotation_rad"],
            "maximum_rebase_tcp_rotation_rad",
        ),
        maximum_entry_moment_score_nm=_positive_number(
            value["maximum_entry_moment_score_nm"],
            "maximum_entry_moment_score_nm",
        ),
        maximum_entry_load_imbalance=_positive_number(
            value["maximum_entry_load_imbalance"],
            "maximum_entry_load_imbalance",
        ),
    )


def validate_realized_state_rebase(
    commanded_arm_rad: Sequence[Real],
    realized_arm_rad: Sequence[Real],
    commanded_tcp: Sequence[Sequence[Real]],
    realized_tcp: Sequence[Sequence[Real]],
    config: PreLiftRealizedStateRebaseConfig,
) -> dict[str, Any]:
    """Validate one robot-only target rebase and return auditable deltas."""

    commanded = np.asarray(tuple(commanded_arm_rad), dtype=np.float64)
    realized = np.asarray(tuple(realized_arm_rad), dtype=np.float64)
    target_transform = np.asarray(commanded_tcp, dtype=np.float64)
    realized_transform = np.asarray(realized_tcp, dtype=np.float64)
    if commanded.shape != (7,) or realized.shape != (7,):
        raise ValueError("rebase requires two seven-joint robot states")
    if target_transform.shape != (4, 4) or realized_transform.shape != (4, 4):
        raise ValueError("rebase requires two 4x4 robot FK transforms")
    if not all(
        np.all(np.isfinite(value))
        for value in (commanded, realized, target_transform, realized_transform)
    ):
        raise ValueError("rebase robot state must be finite")
    joint_delta = realized - commanded
    translation_delta = realized_transform[:3, 3] - target_transform[:3, 3]
    relative_rotation = target_transform[:3, :3].T @ realized_transform[:3, :3]
    cosine = float(np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0))
    rotation_delta = float(math.acos(cosine))
    maximum_joint_delta = float(np.max(np.abs(joint_delta)))
    translation_norm = float(np.linalg.norm(translation_delta))
    if maximum_joint_delta > config.maximum_rebase_joint_delta_rad:
        raise ValueError("realized-state rebase joint delta exceeds its internal bound")
    if translation_norm > config.maximum_rebase_tcp_translation_m:
        raise ValueError("realized-state rebase TCP translation exceeds its internal bound")
    if rotation_delta > config.maximum_rebase_tcp_rotation_rad:
        raise ValueError("realized-state rebase TCP rotation exceeds its internal bound")
    return {
        "joint_delta_rad": joint_delta.tolist(),
        "maximum_joint_delta_rad": maximum_joint_delta,
        "tcp_translation_delta_m": translation_delta.tolist(),
        "tcp_translation_delta_norm_m": translation_norm,
        "tcp_rotation_delta_rad": rotation_delta,
        "robot_joint_state_only": True,
        "object_truth_used": False,
        "contact_truth_used": False,
    }


__all__ = [
    "CONFIG_KEYS",
    "PreLiftRealizedStateRebaseConfig",
    "load_pre_lift_realized_state_rebase_config",
    "validate_realized_state_rebase",
]
