"""B-V2-H7 damping with the B-V2-H10 evidence-bound joint selection."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any, Mapping

from .pre_lift_arm_drive_compliance import minimum_jerk_blend


THRESHOLD_LABEL = "SIM_TUNING_ONLY_B_V2_H7"
SOURCE_RUN_ID = "B-V2-GRASP-08"
SOURCE_LOAD_INCREMENT_NM = 0.04907255567783536
SOURCE_SPEED_RAD_S = 0.05643138289451599
INITIAL_DAMPING_NM_S_RAD = 1.0
FINAL_DAMPING_NM_S_RAD = (
    INITIAL_DAMPING_NM_S_RAD
    + SOURCE_LOAD_INCREMENT_NM / SOURCE_SPEED_RAD_S
)
TRANSITION_STEPS = 240
MAXIMUM_READBACK_ERROR = 1.0e-5
PRESHAPE_JOINT_NAME = "f1j1"
PRESHAPE_EXTENSION_SOURCE_RUN_ID = "B-V2-GRASP-11-IFIX02"

CONFIG_KEYS = (
    "enabled",
    "threshold_label",
    "source_run_id",
    "include_preshape_joint",
    "preshape_joint_name",
    "preshape_extension_source_run_id",
    "source_load_increment_nm",
    "source_speed_rad_s",
    "initial_damping_nm_s_rad",
    "final_damping_nm_s_rad",
    "transition_steps",
    "maximum_readback_error",
)


@dataclass(frozen=True)
class PostContactFingerDampingConfig:
    enabled: bool
    threshold_label: str
    source_run_id: str
    include_preshape_joint: bool
    preshape_joint_name: str
    preshape_extension_source_run_id: str
    source_load_increment_nm: float
    source_speed_rad_s: float
    initial_damping_nm_s_rad: float
    final_damping_nm_s_rad: float
    transition_steps: int
    maximum_readback_error: float

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("post-contact finger damping enabled must be boolean")
        if type(self.include_preshape_joint) is not bool:
            raise ValueError(
                "post-contact preshape damping selection must be boolean"
            )
        exact = {
            "threshold_label": THRESHOLD_LABEL,
            "source_run_id": SOURCE_RUN_ID,
            "preshape_joint_name": PRESHAPE_JOINT_NAME,
            "preshape_extension_source_run_id": (
                PRESHAPE_EXTENSION_SOURCE_RUN_ID
            ),
            "source_load_increment_nm": SOURCE_LOAD_INCREMENT_NM,
            "source_speed_rad_s": SOURCE_SPEED_RAD_S,
            "initial_damping_nm_s_rad": INITIAL_DAMPING_NM_S_RAD,
            "final_damping_nm_s_rad": FINAL_DAMPING_NM_S_RAD,
            "transition_steps": TRANSITION_STEPS,
            "maximum_readback_error": MAXIMUM_READBACK_ERROR,
        }
        for name, required in exact.items():
            actual = getattr(self, name)
            if isinstance(required, str):
                matches = actual == required
            elif isinstance(required, int):
                matches = type(actual) is int and actual == required
            else:
                matches = (
                    isinstance(actual, Real)
                    and not isinstance(actual, bool)
                    and math.isclose(
                        float(actual), required, rel_tol=0.0, abs_tol=1.0e-15
                    )
                )
            if not matches:
                raise ValueError(
                    "post-contact damping "
                    f"{name} is frozen at the evidence-derived value {required!r}"
                )


def _default_config(enabled: bool = False) -> PostContactFingerDampingConfig:
    return PostContactFingerDampingConfig(
        enabled=enabled,
        threshold_label=THRESHOLD_LABEL,
        source_run_id=SOURCE_RUN_ID,
        include_preshape_joint=False,
        preshape_joint_name=PRESHAPE_JOINT_NAME,
        preshape_extension_source_run_id=PRESHAPE_EXTENSION_SOURCE_RUN_ID,
        source_load_increment_nm=SOURCE_LOAD_INCREMENT_NM,
        source_speed_rad_s=SOURCE_SPEED_RAD_S,
        initial_damping_nm_s_rad=INITIAL_DAMPING_NM_S_RAD,
        final_damping_nm_s_rad=FINAL_DAMPING_NM_S_RAD,
        transition_steps=TRANSITION_STEPS,
        maximum_readback_error=MAXIMUM_READBACK_ERROR,
    )


def load_post_contact_finger_damping_config(
    value: Any,
) -> PostContactFingerDampingConfig:
    """Load the strict H7 section; historical contracts default disabled."""

    if value is None:
        return _default_config(False)
    if not isinstance(value, Mapping):
        raise ValueError("post_contact_finger_damping must be a mapping")
    unknown = sorted(set(value) - set(CONFIG_KEYS))
    missing = sorted(set(CONFIG_KEYS) - set(value))
    if unknown or missing:
        raise ValueError(
            "post_contact_finger_damping has unknown keys "
            f"{unknown} and/or missing keys {missing}"
        )
    return PostContactFingerDampingConfig(
        enabled=value["enabled"],
        threshold_label=str(value["threshold_label"]),
        source_run_id=str(value["source_run_id"]),
        include_preshape_joint=value["include_preshape_joint"],
        preshape_joint_name=str(value["preshape_joint_name"]),
        preshape_extension_source_run_id=str(
            value["preshape_extension_source_run_id"]
        ),
        source_load_increment_nm=float(value["source_load_increment_nm"]),
        source_speed_rad_s=float(value["source_speed_rad_s"]),
        initial_damping_nm_s_rad=float(value["initial_damping_nm_s_rad"]),
        final_damping_nm_s_rad=float(value["final_damping_nm_s_rad"]),
        transition_steps=value["transition_steps"],
        maximum_readback_error=float(value["maximum_readback_error"]),
    )


def select_post_contact_damping_hand_indices(
    active_hand_joint_names: tuple[str, ...],
    formal_finger_hand_indices: tuple[int, ...],
    config: PostContactFingerDampingConfig,
) -> tuple[int, ...]:
    """Select damping DOFs without changing the formal three-channel monitor."""

    if len(set(formal_finger_hand_indices)) != len(formal_finger_hand_indices):
        raise ValueError("formal finger hand indices must be unique")
    if any(
        type(index) is not int
        or index < 0
        or index >= len(active_hand_joint_names)
        for index in formal_finger_hand_indices
    ):
        raise ValueError("formal finger hand index is outside the active hand")
    selected = list(formal_finger_hand_indices)
    if config.include_preshape_joint:
        matches = [
            index
            for index, name in enumerate(active_hand_joint_names)
            if name == config.preshape_joint_name
        ]
        if len(matches) != 1:
            raise ValueError(
                "configured preshape joint must appear exactly once in the active hand"
            )
        if matches[0] in selected:
            raise ValueError(
                "configured preshape joint must remain outside formal torque channels"
            )
        selected.insert(0, matches[0])
    return tuple(selected)


def derive_post_contact_finger_damping_step(
    transition_step: int,
    config: PostContactFingerDampingConfig,
) -> dict[str, float | int]:
    """Return one monotonic, bounded damping step without a search surface."""

    if type(transition_step) is not int or not (
        0 <= transition_step < config.transition_steps
    ):
        raise ValueError("transition_step is outside the configured H7 window")
    fraction = float(transition_step + 1) / float(config.transition_steps)
    blend = minimum_jerk_blend(fraction)
    damping = config.initial_damping_nm_s_rad + blend * (
        config.final_damping_nm_s_rad - config.initial_damping_nm_s_rad
    )
    return {
        "transition_step": transition_step,
        "fraction": fraction,
        "minimum_jerk_blend": blend,
        "applied_damping_nm_s_rad": damping,
    }


__all__ = [
    "CONFIG_KEYS",
    "FINAL_DAMPING_NM_S_RAD",
    "PRESHAPE_EXTENSION_SOURCE_RUN_ID",
    "PRESHAPE_JOINT_NAME",
    "PostContactFingerDampingConfig",
    "derive_post_contact_finger_damping_step",
    "load_post_contact_finger_damping_config",
    "select_post_contact_damping_hand_indices",
]
