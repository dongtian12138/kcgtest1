"""Fail-closed pre-entry key-search interface for the multilayer D38999 model.

The module emits diagnostic yaw-step candidates only.  It consumes an
image-derived yaw residual, registered robot-FK gaps, wrist moment, and bounded
controller state.  Object truth, contact identity, contact normal, and assembly
event truth are deliberately absent from the API.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import yaml

from kcg_connector.d38999_key_branch_selector import BRANCH_IDS


SCHEMA_VERSION = "kcg_d38999_multilayer_safe_key_search_v1"
VISION_YAW_SOURCE = "IMAGE_DERIVED_C2_RESOLVED"
CONTROL_RATE_HZ = 240.0
MAXIMUM_RZ_SEARCH_SPEED_RAD_S = 0.010
MAXIMUM_RZ_STEP_RAD = MAXIMUM_RZ_SEARCH_SPEED_RAD_S / CONTROL_RATE_HZ
MAXIMUM_SEARCH_ANGLE_RAD = 0.008
MAXIMUM_SEARCH_ATTEMPTS = 2
MINIMUM_PREENTRY_GAP_M = 0.010
KEY_MISMATCH_TORSION_NM = 0.025
EXPERIMENTAL_BENDING_ABORT_NM = 0.18
EXPERIMENTAL_TORSION_ABORT_NM = 0.05
FORMAL_MOMENT_COMPONENT_LIMIT_NM = 0.30

FROZEN_SOURCES = {
    "src/kcg_connector/config/d38999_master_model_contract_v1.yaml": (
        "57010b3c6f8d2214712427193ef7b9b57ede3089a882aa88033f89774215c68b"
    ),
    "src/kcg_connector/config/d38999_compliant_insertion_v2.yaml": (
        "0f16e9b2fc5d615a4e8035dfa21c4ec9a18a341b4320ecbc3e66b874be489703"
    ),
    "src/kcg_connector/config/d38999_tactile_engage_probe_v2.yaml": (
        "e55143c3101ad4e5008d73c5ddbaa3af9831b9076657026895b4975b84558a2f"
    ),
    "src/kcg_connector/kcg_connector/d38999_key_branch_selector.py": (
        "aeac14722045418a942223d08863b27a99539c1dc192a0440e21789f6b93d09c"
    ),
    "src/kcg_connector/kcg_connector/wrist_moment_safety_guard.py": (
        "779f7601a69f31c87ba44ad88584f540c2178c13b8c2bc09f5bde69385df0db8"
    ),
    "artifacts/agent_control/tasks/EIGHT-HOUR-C10-POSE-CONFIDENCE-REJECTION/"
    "TASK_RESULT.json": (
        "da8a73cbe9970dc535357068b1667cc78b929a85fb04435760e6a347b3c0f64d"
    ),
    "artifacts/agent_control/tasks/EIGHT-HOUR-D5-ATTITUDE-LEVELING/"
    "TASK_RESULT.json": (
        "cfe15363e9cd2d64cd32ce37057ebf40cec61015db1660f75ce7f55a9073b9c5"
    ),
    "artifacts/agent_control/tasks/EIGHT-HOUR-B5-WRIST-MOMENT-MONITOR/"
    "TASK_RESULT.json": (
        "2bebe773c145d4afec89cdf1865ae97eb13db8bf9019d2006bb95ba635c38e0f"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _moment3(value: Any) -> tuple[float, float, float] | None:
    if isinstance(value, (str, bytes, bool)):
        return None
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if len(result) != 3 or not all(math.isfinite(item) for item in result):
        return None
    return result  # type: ignore[return-value]


def _base_result() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "REJECTED_SAFE_STOP",
        "rejection_code": None,
        "delta_twist_task": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "next_search_angle_rad": None,
        "key_search_candidate": False,
        "retract_required": False,
        "motion_command_emitted": False,
        "control_authorized": False,
        "dynamic_key_search_pass_claimed": False,
    }


def evaluate_safe_key_search_step(
    visual_yaw_error_rad: float,
    current_search_angle_rad: float,
    search_attempt_count: int,
    registered_preentry_command_fk_gap_m: float,
    registered_preentry_measured_fk_gap_m: float,
    wrist_moment_task_nm: Sequence[float],
    *,
    upstream_attitude_ready: bool,
    vision_pose_control_authorized: bool,
    selected_c2_branch_id: str | None,
    visual_yaw_source: str,
) -> dict[str, Any]:
    """Return one bounded Rz candidate while retaining all runtime gates."""

    base = _base_result()
    yaw_error = _finite_number(visual_yaw_error_rad)
    current_angle = _finite_number(current_search_angle_rad)
    command_gap = _finite_number(registered_preentry_command_fk_gap_m)
    measured_gap = _finite_number(registered_preentry_measured_fk_gap_m)
    moment = _moment3(wrist_moment_task_nm)
    if (
        yaw_error is None
        or current_angle is None
        or command_gap is None
        or measured_gap is None
        or moment is None
        or type(search_attempt_count) is not int
        or search_attempt_count < 0
        or type(upstream_attitude_ready) is not bool
        or type(vision_pose_control_authorized) is not bool
        or not isinstance(visual_yaw_source, str)
    ):
        return {**base, "rejection_code": "INVALID_KEY_SEARCH_INPUT"}

    if max(abs(value) for value in moment) > FORMAL_MOMENT_COMPONENT_LIMIT_NM:
        return {**base, "rejection_code": "FORMAL_MOMENT_COMPONENT_LIMIT"}
    if math.hypot(moment[0], moment[1]) > EXPERIMENTAL_BENDING_ABORT_NM:
        return {
            **base,
            "rejection_code": "EXPERIMENTAL_BENDING_ABORT",
            "retract_required": True,
        }
    if abs(moment[2]) > EXPERIMENTAL_TORSION_ABORT_NM:
        return {
            **base,
            "rejection_code": "EXPERIMENTAL_TORSION_ABORT",
            "retract_required": True,
        }
    if not upstream_attitude_ready:
        return {**base, "rejection_code": "UPSTREAM_ATTITUDE_REJECTED"}
    if not vision_pose_control_authorized:
        return {**base, "rejection_code": "VISION_KEY_YAW_UNAUTHORIZED"}
    if selected_c2_branch_id not in BRANCH_IDS:
        return {**base, "rejection_code": "C2_BRANCH_UNRESOLVED"}
    if visual_yaw_source != VISION_YAW_SOURCE:
        return {**base, "rejection_code": "VISUAL_YAW_PROVENANCE_REJECTED"}
    if (
        command_gap < MINIMUM_PREENTRY_GAP_M
        or measured_gap < MINIMUM_PREENTRY_GAP_M
    ):
        return {
            **base,
            "rejection_code": "PIN_DEPTH_WINDOW_CLOSED",
            "retract_required": True,
        }
    if search_attempt_count >= MAXIMUM_SEARCH_ATTEMPTS:
        return {
            **base,
            "rejection_code": "SEARCH_ATTEMPT_BUDGET_EXHAUSTED",
            "retract_required": True,
        }
    if abs(current_angle) > MAXIMUM_SEARCH_ANGLE_RAD:
        return {
            **base,
            "rejection_code": "SEARCH_ANGLE_STATE_OUT_OF_BOUNDS",
            "retract_required": True,
        }
    if abs(moment[2]) >= KEY_MISMATCH_TORSION_NM:
        return {
            **base,
            "rejection_code": "KEY_MISMATCH_TORSION_DETECTED",
            "retract_required": True,
        }
    if yaw_error == 0.0:
        return {
            **base,
            "status": "VISUAL_YAW_ERROR_ZERO_DIAGNOSTIC_ONLY",
            "rejection_code": "NO_ROTATION_NEEDED_DIAGNOSTIC_ONLY",
            "next_search_angle_rad": current_angle,
        }

    desired_step = max(
        -MAXIMUM_RZ_STEP_RAD,
        min(MAXIMUM_RZ_STEP_RAD, yaw_error),
    )
    lower = -MAXIMUM_SEARCH_ANGLE_RAD - current_angle
    upper = MAXIMUM_SEARCH_ANGLE_RAD - current_angle
    bounded_step = max(lower, min(upper, desired_step))
    if bounded_step == 0.0:
        return {
            **base,
            "rejection_code": "SEARCH_ANGLE_BUDGET_EXHAUSTED",
            "retract_required": True,
        }
    next_angle = current_angle + bounded_step
    return {
        **base,
        "status": "OFFLINE_KEY_SEARCH_STEP_CANDIDATE",
        "rejection_code": "DIAGNOSTIC_ONLY_NOT_MOTION_AUTHORITY",
        "delta_twist_task": [0.0, 0.0, 0.0, 0.0, 0.0, bounded_step],
        "next_search_angle_rad": next_angle,
        "key_search_candidate": True,
    }


def build_safe_key_search_contract(repository_root: str | Path) -> dict[str, Any]:
    """Cross-check the existing model, pre-entry, visual, and safety sources."""

    root = Path(repository_root).resolve()
    sources: list[dict[str, str]] = []
    for relative, expected in FROZEN_SOURCES.items():
        path = root / relative
        if not path.is_file():
            raise ValueError(f"frozen D6 source missing: {relative}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"frozen D6 source hash mismatch: {relative}")
        sources.append({"path": relative, "sha256": actual})

    master = yaml.safe_load(
        (root / "src/kcg_connector/config/d38999_master_model_contract_v1.yaml")
        .read_text(encoding="utf-8")
    )
    compliant = yaml.safe_load(
        (root / "src/kcg_connector/config/d38999_compliant_insertion_v2.yaml")
        .read_text(encoding="utf-8")
    )
    engage = yaml.safe_load(
        (root / "src/kcg_connector/config/d38999_tactile_engage_probe_v2.yaml")
        .read_text(encoding="utf-8")
    )
    c10 = json.loads(
        (root / "artifacts/agent_control/tasks/"
         "EIGHT-HOUR-C10-POSE-CONFIDENCE-REJECTION/TASK_RESULT.json")
        .read_text(encoding="utf-8")
    )
    d5 = json.loads(
        (root / "artifacts/agent_control/tasks/"
         "EIGHT-HOUR-D5-ATTITUDE-LEVELING/TASK_RESULT.json")
        .read_text(encoding="utf-8")
    )
    b5 = json.loads(
        (root / "artifacts/agent_control/tasks/"
         "EIGHT-HOUR-B5-WRIST-MOMENT-MONITOR/TASK_RESULT.json")
        .read_text(encoding="utf-8")
    )
    motion = compliant["motion"]
    classifier = compliant["contact_classifier"]
    safety = compliant["safety"]
    preentry = engage["entry_confirmation_policy"]
    keying = master["keying"]
    events = master["assembly_events"]["ordered"]
    if (
        keying["key_count"] != 5
        or keying["keyway_count"] != 5
        or keying["canonical_angles_deg"] != [0.0, 80.0, 142.0, 196.0, 293.0]
        or keying["polarization_precedes_thread_and_contact"] is not True
        or events[3]["name"] != "first_pin_socket_spring_touch"
        or events[3]["nominal_separation_m"] != 0.01202
        or compliant["control_rate_hz"] != CONTROL_RATE_HZ
        or motion["maximum_rz_search_speed_rad_s"]
        != MAXIMUM_RZ_SEARCH_SPEED_RAD_S
        or motion["maximum_search_angle_rad"] != MAXIMUM_SEARCH_ANGLE_RAD
        or motion["maximum_retries"] != MAXIMUM_SEARCH_ATTEMPTS
        or classifier["key_mismatch_mz_nm"] != KEY_MISMATCH_TORSION_NM
        or safety["hard_bending_moment_nm"] != EXPERIMENTAL_BENDING_ABORT_NM
        or safety["hard_torsional_moment_nm"] != EXPERIMENTAL_TORSION_ABORT_NM
        or preentry["minimum_registered_preentry_gap_m"]
        != MINIMUM_PREENTRY_GAP_M
        or preentry["truth_control_allowed"] is not False
        or b5["moment_limit_nm"] != FORMAL_MOMENT_COMPONENT_LIMIT_NM
        or c10["control_authorized"] is not False
        or c10["c2_resolved"] is not False
        or d5["outcome"] != "OFFLINE_PASS"
        or d5["dynamic_attitude_leveling_pass_claimed"] is not False
    ):
        raise ValueError("authoritative D6 key-search contract changed")

    current = evaluate_safe_key_search_step(
        0.0,
        0.0,
        0,
        MINIMUM_PREENTRY_GAP_M,
        MINIMUM_PREENTRY_GAP_M,
        (0.0, 0.0, 0.0),
        upstream_attitude_ready=False,
        vision_pose_control_authorized=False,
        selected_c2_branch_id=None,
        visual_yaw_source=VISION_YAW_SOURCE,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "OFFLINE_SAFE_KEY_SEARCH_INTERFACE_READY",
        "classification": "PREENTRY_IMAGE_AND_WRIST_ONLY_BOUNDED_RZ_SEARCH",
        "key_count": 5,
        "key_angles_deg": list(keying["canonical_angles_deg"]),
        "branch_ids": list(BRANCH_IDS),
        "required_visual_yaw_source": VISION_YAW_SOURCE,
        "minimum_preentry_gap_m": MINIMUM_PREENTRY_GAP_M,
        "first_pin_nominal_separation_m_posthoc_only": 0.01202,
        "control_rate_hz": CONTROL_RATE_HZ,
        "maximum_rz_search_speed_rad_s": MAXIMUM_RZ_SEARCH_SPEED_RAD_S,
        "maximum_rz_step_rad": MAXIMUM_RZ_STEP_RAD,
        "maximum_search_angle_rad": MAXIMUM_SEARCH_ANGLE_RAD,
        "maximum_search_attempts": MAXIMUM_SEARCH_ATTEMPTS,
        "key_mismatch_torsion_nm": KEY_MISMATCH_TORSION_NM,
        "formal_moment_component_limit_nm": FORMAL_MOMENT_COMPONENT_LIMIT_NM,
        "forbidden_inputs": [
            "object_truth_pose",
            "contact_object_name",
            "contact_normal",
            "collision_event_truth",
            "first_pin_event_truth",
        ],
        "current_readiness": current,
        "simulation_started": False,
        "dynamic_key_search_pass_claimed": False,
        "control_authorized": False,
        "hardware_authorized": False,
        "sources": sources,
    }


__all__ = [
    "BRANCH_IDS",
    "CONTROL_RATE_HZ",
    "FORMAL_MOMENT_COMPONENT_LIMIT_NM",
    "FROZEN_SOURCES",
    "KEY_MISMATCH_TORSION_NM",
    "MAXIMUM_RZ_SEARCH_SPEED_RAD_S",
    "MAXIMUM_RZ_STEP_RAD",
    "MAXIMUM_SEARCH_ANGLE_RAD",
    "MAXIMUM_SEARCH_ATTEMPTS",
    "MINIMUM_PREENTRY_GAP_M",
    "SCHEMA_VERSION",
    "VISION_YAW_SOURCE",
    "build_safe_key_search_contract",
    "evaluate_safe_key_search_step",
]
