"""Pure wrist-FT-only guarded insertion policy for the D38999 pipeline.

The policy accepts only compensated wrist wrench and robot proprioception.
Simulator contact reports and object ground truth are intentionally absent from
the observation schema.  This module contains no Isaac imports and does not
claim runtime, hardware-safety or production readiness.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml


SCHEMA_VERSION = "kcg_d38999_wrist_ft_guarded_insertion_v1"
DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "config/d38999_wrist_ft_guarded_insertion_v1.yaml"
)
OBSERVATION_KEYS = {
    "timestamp_s",
    "sample_age_s",
    "compensated_wrench_task",
    "measured_tcp_position_task_m",
    "commanded_tcp_position_task_m",
    "arm_tracking_error_rad",
    "maximum_joint_speed_rad_s",
    "gripper_position_drift_from_preinsert_rad",
    "robot_state_finite",
    "vision_preinsert_id",
}


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _exact(value: Mapping[str, Any], expected, label: str) -> None:
    actual = set(value)
    required = set(expected)
    if actual != required:
        raise ValueError(
            f"{label} keys are invalid: missing={sorted(required-actual)}, "
            f"extra={sorted(actual-required)}"
        )


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be non-empty text")
    return value


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ValueError(f"{label} must be finite" + (" and positive" if positive else ""))
    return result


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _vector(value: Any, size: int, label: str) -> tuple[float, ...]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be a finite {size}-vector")
    return tuple(float(item) for item in array)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class GuardedInsertionContract:
    enabled: bool
    status: str
    input_files: tuple[tuple[str, str, str], ...]
    task_frame_id: str
    control_rate_hz: int
    preinsert_gap_m: float
    entry_gap_m: float
    engage_gap_m: float
    guarded_approach_speed_m_s: float
    insertion_speed_m_s: float
    contact_retract_distance_m: float
    contact_retract_speed_m_s: float
    maximum_contact_retries: int
    maximum_xy_correction_step_m: float
    maximum_xy_search_radius_m: float
    lateral_correction_gain_m_per_n: float
    hold_steps: int
    maximum_control_steps: int
    maximum_axial_force_n: float
    maximum_lateral_force_n: float
    maximum_bending_torque_nm: float
    maximum_tightening_torque_nm: float
    maximum_arm_tracking_error_rad: float
    maximum_joint_speed_rad_s: float
    maximum_gripper_position_drift_from_preinsert_rad: float
    maximum_sample_age_s: float
    minimum_axial_contact_force_n: float
    minimum_lateral_correction_force_n: float
    minimum_bending_correction_torque_nm: float
    maximum_hold_axial_force_n: float
    maximum_hold_lateral_force_n: float


class GuardedInsertionPhase(str, Enum):
    PREINSERT = "PREINSERT"
    GUARDED_APPROACH = "GUARDED_APPROACH"
    CONTACT_RETRACT = "CONTACT_RETRACT"
    CENTER_CORRECTION = "CENTER_CORRECTION"
    INSERT = "INSERT"
    HOLD = "HOLD"
    COMPLETE = "COMPLETE"
    ABORT = "ABORT"


@dataclass(frozen=True)
class GuardedInsertionObservation:
    timestamp_s: float
    sample_age_s: float
    compensated_wrench_task: tuple[float, ...]
    measured_tcp_position_task_m: tuple[float, ...]
    commanded_tcp_position_task_m: tuple[float, ...]
    arm_tracking_error_rad: float
    maximum_joint_speed_rad_s: float
    gripper_position_drift_from_preinsert_rad: float
    robot_state_finite: bool
    vision_preinsert_id: str


@dataclass(frozen=True)
class GuardedInsertionState:
    phase: GuardedInsertionPhase
    preinsert_tcp_task_m: tuple[float, float, float]
    xy_offset_task_m: tuple[float, float]
    contact_retract_target_z_task_m: float | None = None
    pending_xy_correction_task_m: tuple[float, float] = (0.0, 0.0)
    contact_retry_count: int = 0
    hold_count: int = 0
    step_count: int = 0
    abort_reason: str | None = None


@dataclass(frozen=True)
class GuardedInsertionCommand:
    next_state: GuardedInsertionState
    delta_tcp_task_m: tuple[float, float, float]
    stop_motion: bool
    status: str


def _input_file(value: Any, label: str) -> tuple[str, str, str]:
    item = _mapping(value, label)
    _exact(item, {"path", "sha256"}, label)
    path = _text(item["path"], f"{label}.path")
    digest = _text(item["sha256"], f"{label}.sha256")
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError(f"{label}.sha256 must be lowercase SHA-256")
    return label.rsplit(".", 1)[-1], path, digest


def load_guarded_insertion_contract(
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> GuardedInsertionContract:
    root = _mapping(
        yaml.safe_load(Path(path).read_text(encoding="utf-8")), "root"
    )
    _exact(
        root,
        {
            "schema_version", "enabled", "status", "inputs",
            "controller_interface", "motion", "experimental_abort_envelope",
            "contact_response", "boundaries",
        },
        "root",
    )
    if root["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unexpected guarded insertion schema_version")
    inputs = _mapping(root["inputs"], "inputs")
    expected_inputs = {
        "visual_preinsert", "wrist_ft_monitor", "physical_insertion",
        "wrist_ft_runtime", "insertion_geometry",
    }
    _exact(inputs, expected_inputs, "inputs")
    input_files = tuple(
        _input_file(inputs[name], f"inputs.{name}")
        for name in sorted(expected_inputs)
    )

    interface = _mapping(root["controller_interface"], "controller_interface")
    _exact(
        interface,
        {"task_frame_id", "wrench_order", "allowed_observations", "forbidden_control_inputs"},
        "controller_interface",
    )
    if tuple(interface["wrench_order"]) != ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz"):
        raise ValueError("wrench order must be Fx,Fy,Fz,Tx,Ty,Tz")
    if set(interface["allowed_observations"]) != OBSERVATION_KEYS:
        raise ValueError("allowed observation schema changed")
    forbidden = set(interface["forbidden_control_inputs"])
    required_forbidden = {
        "simulator_object_truth_pose", "simulator_truth_gap",
        "physx_contact_report", "physx_contact_manifold", "collider_path",
        "contact_normal", "contact_separation", "contact_material",
    }
    if forbidden != required_forbidden:
        raise ValueError("forbidden control inputs changed")

    motion = _mapping(root["motion"], "motion")
    _exact(
        motion,
        {
            "control_rate_hz", "preinsert_gap_m", "entry_gap_m", "engage_gap_m",
            "guarded_approach_speed_m_s", "insertion_speed_m_s",
            "contact_retract_distance_m", "contact_retract_speed_m_s",
            "maximum_contact_retries",
            "maximum_xy_correction_step_m", "maximum_xy_search_radius_m",
            "lateral_correction_gain_m_per_n", "correction_direction", "hold_steps",
            "maximum_control_steps",
        },
        "motion",
    )
    if motion["correction_direction"] != "environment_force_on_tool":
        raise ValueError("lateral correction sign is not explicit")
    abort = _mapping(root["experimental_abort_envelope"], "experimental_abort_envelope")
    abort_keys = {
        "maximum_axial_force_n", "maximum_lateral_force_n",
        "maximum_bending_torque_nm", "maximum_tightening_torque_nm",
        "maximum_arm_tracking_error_rad", "maximum_joint_speed_rad_s",
        "maximum_gripper_position_drift_from_preinsert_rad",
        "maximum_sample_age_s",
    }
    _exact(abort, abort_keys, "experimental_abort_envelope")
    response = _mapping(root["contact_response"], "contact_response")
    response_keys = {
        "minimum_axial_contact_force_n", "minimum_lateral_correction_force_n",
        "minimum_bending_correction_torque_nm", "maximum_hold_axial_force_n",
        "maximum_hold_lateral_force_n",
    }
    _exact(response, response_keys, "contact_response")
    boundaries = _mapping(root["boundaries"], "boundaries")
    expected_boundaries = {
        "uses_fingertip_tactile_sensor", "uses_physx_contact_for_control",
        "uses_simulator_truth_for_control", "hardware_safety_certified",
        "production_control_authorized", "runtime_integrated",
        "insertion_success_claimed", "rl_training_ready_claimed",
    }
    _exact(boundaries, expected_boundaries, "boundaries")
    runtime_integrated = _boolean(
        boundaries["runtime_integrated"],
        "boundaries.runtime_integrated",
    )
    if not runtime_integrated:
        raise ValueError("guarded insertion runtime must remain integrated")
    for key in boundaries:
        if key != "runtime_integrated" and _boolean(
            boundaries[key], f"boundaries.{key}"
        ):
            raise ValueError(f"boundaries.{key} must remain false")

    preinsert = _number(motion["preinsert_gap_m"], "motion.preinsert_gap_m", positive=True)
    entry = _number(motion["entry_gap_m"], "motion.entry_gap_m", positive=True)
    engage = _number(motion["engage_gap_m"], "motion.engage_gap_m", positive=True)
    if not preinsert > entry > engage:
        raise ValueError("guarded insertion gaps must decrease")

    return GuardedInsertionContract(
        enabled=_boolean(root["enabled"], "enabled"),
        status=_text(root["status"], "status"),
        input_files=input_files,
        task_frame_id=_text(interface["task_frame_id"], "task_frame_id"),
        control_rate_hz=_integer(motion["control_rate_hz"], "control_rate_hz", minimum=1),
        preinsert_gap_m=preinsert,
        entry_gap_m=entry,
        engage_gap_m=engage,
        guarded_approach_speed_m_s=_number(motion["guarded_approach_speed_m_s"], "guarded_approach_speed_m_s", positive=True),
        insertion_speed_m_s=_number(motion["insertion_speed_m_s"], "insertion_speed_m_s", positive=True),
        contact_retract_distance_m=_number(
            motion["contact_retract_distance_m"],
            "contact_retract_distance_m",
            positive=True,
        ),
        contact_retract_speed_m_s=_number(
            motion["contact_retract_speed_m_s"],
            "contact_retract_speed_m_s",
            positive=True,
        ),
        maximum_contact_retries=_integer(
            motion["maximum_contact_retries"],
            "maximum_contact_retries",
            minimum=1,
        ),
        maximum_xy_correction_step_m=_number(motion["maximum_xy_correction_step_m"], "maximum_xy_correction_step_m", positive=True),
        maximum_xy_search_radius_m=_number(motion["maximum_xy_search_radius_m"], "maximum_xy_search_radius_m", positive=True),
        lateral_correction_gain_m_per_n=_number(motion["lateral_correction_gain_m_per_n"], "lateral_correction_gain_m_per_n", positive=True),
        hold_steps=_integer(motion["hold_steps"], "hold_steps", minimum=1),
        maximum_control_steps=_integer(
            motion["maximum_control_steps"],
            "maximum_control_steps",
            minimum=1,
        ),
        **{key: _number(abort[key], key, positive=True) for key in abort_keys},
        **{key: _number(response[key], key, positive=True) for key in response_keys},
    )


def verify_guarded_insertion_inputs(
    contract: GuardedInsertionContract, repository: str | Path
) -> None:
    root = Path(repository).resolve()
    for label, relative_path, expected_digest in contract.input_files:
        target = (root / relative_path).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise ValueError(f"{label} input path is invalid")
        if _sha256(target) != expected_digest:
            raise ValueError(f"{label} input SHA-256 changed")


def parse_guarded_insertion_observation(
    value: Mapping[str, Any],
) -> GuardedInsertionObservation:
    item = _mapping(value, "observation")
    _exact(item, OBSERVATION_KEYS, "observation")
    return GuardedInsertionObservation(
        timestamp_s=_number(item["timestamp_s"], "timestamp_s"),
        sample_age_s=_number(item["sample_age_s"], "sample_age_s"),
        compensated_wrench_task=_vector(item["compensated_wrench_task"], 6, "compensated_wrench_task"),
        measured_tcp_position_task_m=_vector(item["measured_tcp_position_task_m"], 3, "measured_tcp_position_task_m"),
        commanded_tcp_position_task_m=_vector(item["commanded_tcp_position_task_m"], 3, "commanded_tcp_position_task_m"),
        arm_tracking_error_rad=_number(item["arm_tracking_error_rad"], "arm_tracking_error_rad"),
        maximum_joint_speed_rad_s=_number(item["maximum_joint_speed_rad_s"], "maximum_joint_speed_rad_s"),
        gripper_position_drift_from_preinsert_rad=_number(
            item["gripper_position_drift_from_preinsert_rad"],
            "gripper_position_drift_from_preinsert_rad",
        ),
        robot_state_finite=_boolean(item["robot_state_finite"], "robot_state_finite"),
        vision_preinsert_id=_text(item["vision_preinsert_id"], "vision_preinsert_id"),
    )


def initial_guarded_insertion_state(
    observation: GuardedInsertionObservation,
) -> GuardedInsertionState:
    return GuardedInsertionState(
        phase=GuardedInsertionPhase.PREINSERT,
        preinsert_tcp_task_m=tuple(observation.measured_tcp_position_task_m),
        xy_offset_task_m=(0.0, 0.0),
    )


def _abort(state: GuardedInsertionState, reason: str) -> GuardedInsertionCommand:
    return GuardedInsertionCommand(
        next_state=replace(
            state,
            phase=GuardedInsertionPhase.ABORT,
            abort_reason=reason,
            step_count=state.step_count + 1,
        ),
        delta_tcp_task_m=(0.0, 0.0, 0.0),
        stop_motion=True,
        status=f"ABORT:{reason}",
    )


def step_guarded_insertion(
    contract: GuardedInsertionContract,
    state: GuardedInsertionState,
    observation: GuardedInsertionObservation,
) -> GuardedInsertionCommand:
    """Return one task-frame command without simulator contact truth."""

    if state.phase in (GuardedInsertionPhase.COMPLETE, GuardedInsertionPhase.ABORT):
        return GuardedInsertionCommand(state, (0.0, 0.0, 0.0), True, state.phase.value)
    if state.step_count >= contract.maximum_control_steps:
        return _abort(state, "maximum_control_steps")

    wrench = np.asarray(observation.compensated_wrench_task, dtype=np.float64)
    if not observation.robot_state_finite:
        return _abort(state, "robot_state_nonfinite")
    checks = (
        (observation.sample_age_s > contract.maximum_sample_age_s, "wrench_sample_stale"),
        (observation.arm_tracking_error_rad > contract.maximum_arm_tracking_error_rad, "arm_tracking_error"),
        (observation.maximum_joint_speed_rad_s > contract.maximum_joint_speed_rad_s, "joint_speed"),
        (
            observation.gripper_position_drift_from_preinsert_rad
            > contract.maximum_gripper_position_drift_from_preinsert_rad,
            "gripper_position_drift_from_preinsert",
        ),
        (abs(wrench[2]) > contract.maximum_axial_force_n, "axial_force"),
        (float(np.linalg.norm(wrench[:2])) > contract.maximum_lateral_force_n, "lateral_force"),
        (float(np.linalg.norm(wrench[3:5])) > contract.maximum_bending_torque_nm, "bending_torque"),
        (abs(wrench[5]) > contract.maximum_tightening_torque_nm, "tightening_torque"),
    )
    for failed, reason in checks:
        if failed:
            return _abort(state, reason)

    measured = np.asarray(observation.measured_tcp_position_task_m, dtype=np.float64)
    preinsert = np.asarray(state.preinsert_tcp_task_m, dtype=np.float64)
    axial_travel = float(preinsert[2] - measured[2])
    target_travel = contract.preinsert_gap_m - contract.engage_gap_m
    entry_travel = contract.preinsert_gap_m - contract.entry_gap_m
    lateral_force = float(np.linalg.norm(wrench[:2]))
    bending = float(np.linalg.norm(wrench[3:5]))
    axial_force = abs(float(wrench[2]))

    if state.phase is GuardedInsertionPhase.PREINSERT:
        return GuardedInsertionCommand(
            replace(state, phase=GuardedInsertionPhase.GUARDED_APPROACH, step_count=state.step_count + 1),
            (0.0, 0.0, 0.0),
            False,
            "START_GUARDED_APPROACH",
        )

    if state.phase is GuardedInsertionPhase.HOLD:
        if abs(wrench[2]) > contract.maximum_hold_axial_force_n or lateral_force > contract.maximum_hold_lateral_force_n:
            return _abort(state, "hold_force")
        hold_count = state.hold_count + 1
        phase = GuardedInsertionPhase.COMPLETE if hold_count >= contract.hold_steps else GuardedInsertionPhase.HOLD
        return GuardedInsertionCommand(
            replace(state, phase=phase, hold_count=hold_count, step_count=state.step_count + 1),
            (0.0, 0.0, 0.0),
            phase is GuardedInsertionPhase.COMPLETE,
            "COMPLETE" if phase is GuardedInsertionPhase.COMPLETE else "HOLD",
        )

    if state.phase is GuardedInsertionPhase.CONTACT_RETRACT:
        target_z = state.contact_retract_target_z_task_m
        if target_z is None:
            return _abort(state, "contact_retract_target_missing")
        remaining = float(target_z - measured[2])
        if remaining > 0.0:
            step = min(
                remaining,
                contract.contact_retract_speed_m_s
                / contract.control_rate_hz,
            )
            return GuardedInsertionCommand(
                replace(state, step_count=state.step_count + 1),
                (0.0, 0.0, step),
                False,
                "WRIST_FT_CONTACT_RETRACT",
            )
        correction = np.asarray(
            state.pending_xy_correction_task_m, dtype=np.float64
        )
        new_offset = np.asarray(state.xy_offset_task_m) + correction
        if float(np.linalg.norm(new_offset)) > (
            contract.maximum_xy_search_radius_m
        ):
            return _abort(state, "xy_search_radius")
        return GuardedInsertionCommand(
            replace(
                state,
                phase=GuardedInsertionPhase.CENTER_CORRECTION,
                xy_offset_task_m=tuple(float(v) for v in new_offset),
                contact_retract_target_z_task_m=None,
                pending_xy_correction_task_m=(0.0, 0.0),
                step_count=state.step_count + 1,
            ),
            (float(correction[0]), float(correction[1]), 0.0),
            False,
            "WRIST_FT_POST_RETRACT_XY_CORRECTION",
        )

    needs_correction = (
        lateral_force >= contract.minimum_lateral_correction_force_n
        or bending >= contract.minimum_bending_correction_torque_nm
    )

    # Any pre-entry correction signal is treated as a guarded-search event.
    # Stop both downward and lateral motion first, unload along +task-Z, then
    # apply a bounded correction derived only from compensated wrist Fx/Fy.
    # This prevents a lateral move while the connector is still loaded.
    if axial_travel < entry_travel and (
        axial_force >= contract.minimum_axial_contact_force_n
        or needs_correction
    ):
        if state.contact_retry_count >= contract.maximum_contact_retries:
            return _abort(state, "maximum_contact_retries")
        if lateral_force < contract.minimum_lateral_correction_force_n:
            return _abort(state, "early_axial_contact_no_lateral_direction")
        correction = contract.lateral_correction_gain_m_per_n * wrench[:2]
        norm = float(np.linalg.norm(correction))
        if norm > contract.maximum_xy_correction_step_m:
            correction *= contract.maximum_xy_correction_step_m / norm
        return GuardedInsertionCommand(
            replace(
                state,
                phase=GuardedInsertionPhase.CONTACT_RETRACT,
                contact_retract_target_z_task_m=(
                    float(measured[2])
                    + contract.contact_retract_distance_m
                ),
                pending_xy_correction_task_m=(
                    float(correction[0]), float(correction[1])
                ),
                contact_retry_count=state.contact_retry_count + 1,
                step_count=state.step_count + 1,
            ),
            (0.0, 0.0, 0.0),
            False,
            "WRIST_FT_EARLY_CONTACT_FREEZE",
        )

    if needs_correction:
        correction = contract.lateral_correction_gain_m_per_n * wrench[:2]
        norm = float(np.linalg.norm(correction))
        if norm > contract.maximum_xy_correction_step_m:
            correction *= contract.maximum_xy_correction_step_m / norm
        new_offset = np.asarray(state.xy_offset_task_m) + correction
        if float(np.linalg.norm(new_offset)) > contract.maximum_xy_search_radius_m:
            return _abort(state, "xy_search_radius")
        return GuardedInsertionCommand(
            replace(
                state,
                phase=GuardedInsertionPhase.CENTER_CORRECTION,
                xy_offset_task_m=tuple(float(v) for v in new_offset),
                step_count=state.step_count + 1,
            ),
            (float(correction[0]), float(correction[1]), 0.0),
            False,
            "WRIST_FT_XY_CORRECTION",
        )

    if axial_travel >= target_travel:
        return GuardedInsertionCommand(
            replace(state, phase=GuardedInsertionPhase.HOLD, step_count=state.step_count + 1),
            (0.0, 0.0, 0.0),
            False,
            "ENTER_HOLD",
        )

    speed = (
        contract.guarded_approach_speed_m_s
        if axial_travel < entry_travel
        else contract.insertion_speed_m_s
    )
    phase = (
        GuardedInsertionPhase.GUARDED_APPROACH
        if axial_travel < entry_travel
        else GuardedInsertionPhase.INSERT
    )
    return GuardedInsertionCommand(
        replace(state, phase=phase, step_count=state.step_count + 1),
        (0.0, 0.0, -speed / contract.control_rate_hz),
        False,
        phase.value,
    )


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "GuardedInsertionCommand",
    "GuardedInsertionContract",
    "GuardedInsertionObservation",
    "GuardedInsertionPhase",
    "GuardedInsertionState",
    "initial_guarded_insertion_state",
    "load_guarded_insertion_contract",
    "parse_guarded_insertion_observation",
    "step_guarded_insertion",
    "verify_guarded_insertion_inputs",
]
