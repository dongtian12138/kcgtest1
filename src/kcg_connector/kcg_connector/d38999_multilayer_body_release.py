"""Non-executing three-finger body-release gate for the multilayer chain."""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


SCHEMA_VERSION = "kcg_d38999_multilayer_body_release_v1"
FROZEN_SOURCES = {
    "artifacts/agent_control/tasks/EIGHT-HOUR-E1-POSTINSERT-HOLD/"
    "TASK_RESULT.json": "a0e4db391789282efc3cff41ee12dc037b9d6d9f7ecc467988456aa86744e414",
    "src/kcg_connector/config/d38999_nut_regrasp_physx_v1.yaml": (
        "50799ed6ab595885b992d311c60f01bf2b4948cf7fe86a45dfdad232c38bf77f"
    ),
    "src/kcg_connector/config/d38999_q7_rewind_probe_v1.yaml": (
        "53cb8f354e72be54b4f72f9c4750a5365e611ea840a253b6a1b62741ed2333bf"
    ),
    "src/kcg_connector/config/d38999_tabletop_physical_grasp_v1.yaml": (
        "68d18977d920cec3681e99ee8beabf934a19d7c69f6ab50f2aa5db5f6e1504dd"
    ),
}


@dataclass(frozen=True)
class BodyReleaseReadiness:
    e1_postinsert_hold_dynamic_pass: bool
    wrist_guard_safe: bool
    wrist_guard_fault_latched: bool
    hand_joint_state_calibrated: bool
    e1_evidence_id: str | None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _vector(value: Any, length: int) -> tuple[float, ...] | None:
    if isinstance(value, (str, bytes, bool)):
        return None
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if len(result) != length or not all(math.isfinite(item) for item in result):
        return None
    return result


def _base(code: str | None = None) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "REJECTED_SAFE_HOLD",
        "rejection_code": code,
        "release_plan_candidate": None,
        "robot_motion_command_emitted": False,
        "finger_command_emitted": False,
        "control_authorized": False,
        "dynamic_body_release_pass_claimed": False,
        "hardware_authorized": False,
    }


def evaluate_body_release_gate(
    readiness: BodyReleaseReadiness,
    *,
    current_hand_positions_rad: Sequence[float],
    current_hand_efforts_nm: Sequence[float],
    sample_timestamp_s: float,
    now_s: float,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    base = _base()
    if not isinstance(readiness, BodyReleaseReadiness) or any(
        type(getattr(readiness, item.name)) is not bool
        for item in fields(readiness)[:4]
    ):
        return {**base, "rejection_code": "INVALID_READINESS_SNAPSHOT"}
    if not readiness.e1_postinsert_hold_dynamic_pass:
        return {**base, "rejection_code": "E1_POSTINSERT_HOLD_NOT_DYNAMIC"}
    if (
        not isinstance(readiness.e1_evidence_id, str)
        or not readiness.e1_evidence_id.strip()
    ):
        return {**base, "rejection_code": "E1_EVIDENCE_ID_MISSING"}
    if not readiness.hand_joint_state_calibrated:
        return {**base, "rejection_code": "HAND_JOINT_STATE_NOT_CALIBRATED"}
    if not readiness.wrist_guard_safe or readiness.wrist_guard_fault_latched:
        return {**base, "rejection_code": "WRIST_MOMENT_GUARD_REJECTED"}
    positions = _vector(current_hand_positions_rad, 4)
    efforts = _vector(current_hand_efforts_nm, 4)
    if positions is None or efforts is None:
        return {**base, "rejection_code": "INVALID_HAND_SAMPLE"}
    try:
        timestamp = float(sample_timestamp_s)
        now = float(now_s)
    except (TypeError, ValueError):
        return {**base, "rejection_code": "INVALID_HAND_SAMPLE_TIME"}
    if (
        isinstance(sample_timestamp_s, bool)
        or isinstance(now_s, bool)
        or not math.isfinite(timestamp)
        or not math.isfinite(now)
        or timestamp > now
        or now - timestamp > float(contract["maximum_sample_gap_s"]) + 1e-12
    ):
        return {**base, "rejection_code": "INVALID_OR_STALE_HAND_SAMPLE"}
    hard_stop = float(contract["hard_stop_nm"])
    if any(abs(value) > hard_stop for value in efforts):
        return {**base, "rejection_code": "FINGER_EFFORT_HARD_STOP"}
    target = tuple(float(value) for value in contract["open_hand_rad"])
    duration = float(contract["release_duration_s"])
    peak_speeds = tuple(1.875 * abs(end - start) / duration for start, end in zip(positions, target))
    if any(value > float(contract["maximum_joint_speed_rad_s"]) for value in peak_speeds):
        return {**base, "rejection_code": "RELEASE_PROFILE_SPEED_LIMIT"}
    return {
        **base,
        "status": "OFFLINE_BODY_RELEASE_PLAN_CANDIDATE",
        "rejection_code": "DIAGNOSTIC_ONLY_NOT_MOTION_AUTHORITY",
        "release_plan_candidate": {
            "joint_order": list(contract["hand_joint_order"]),
            "start_rad": list(positions),
            "target_rad": list(target),
            "duration_s": duration,
            "profile": "minimum_jerk",
            "peak_speed_rad_s": list(peak_speeds),
        },
    }


def build_body_release_contract(repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    sources = []
    for relative, expected in FROZEN_SOURCES.items():
        path = root / relative
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"frozen E2 source hash mismatch: {relative}")
        sources.append({"path": relative, "sha256": expected})
    e1 = json.loads((root / next(p for p in FROZEN_SOURCES if "E1-" in p)).read_text())
    regrasp = yaml.safe_load((root / "src/kcg_connector/config/d38999_nut_regrasp_physx_v1.yaml").read_text())
    rewind = yaml.safe_load((root / "src/kcg_connector/config/d38999_q7_rewind_probe_v1.yaml").read_text())
    grasp = yaml.safe_load((root / "src/kcg_connector/config/d38999_tabletop_physical_grasp_v1.yaml").read_text())
    values = {
        "hand_joint_order": ["f1j1", "f1j2", "f2j1", "f3j1"],
        "open_hand_rad": regrasp["prepared_engage"]["open_hand_rad"],
        "release_duration_s": regrasp["control"]["release_s"],
        "maximum_joint_speed_rad_s": regrasp["acceptance"]["maximum_joint_speed_rad_s"],
        "hard_stop_nm": regrasp["sensing"]["hard_stop_nm"],
        "maximum_sample_gap_s": grasp["detector"]["maximum_sample_gap_s"],
    }
    if (
        e1.get("outcome") != "OFFLINE_PASS"
        or e1.get("dynamic_postinsert_hold_pass_claimed") is not False
        or values["open_hand_rad"] != [1.0, 0.0, 0.0, 0.0]
        or values["release_duration_s"] != 2.5
        or rewind["control"]["release_s"] != values["release_duration_s"]
        or values["maximum_joint_speed_rad_s"] != 1.0
        or values["hard_stop_nm"] != 2.0
        or values["maximum_sample_gap_s"] != 0.0125
    ):
        raise ValueError("authoritative E2 release contract changed")
    current = evaluate_body_release_gate(
        BodyReleaseReadiness(False, False, True, False, None),
        current_hand_positions_rad=[0.0] * 4,
        current_hand_efforts_nm=[0.0] * 4,
        sample_timestamp_s=0.0,
        now_s=0.0,
        contract=values,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        **values,
        "current_decision": current,
        "excluded_legacy_fields": ["body_root_world_m", "temporary_world_body_constraint", "arm_rad"],
        "simulation_started": False,
        "dynamic_body_release_pass_claimed": False,
        "control_authorized": False,
        "hardware_authorized": False,
        "sources": sources,
    }


__all__ = ["BodyReleaseReadiness", "FROZEN_SOURCES", "build_body_release_contract", "evaluate_body_release_gate"]
