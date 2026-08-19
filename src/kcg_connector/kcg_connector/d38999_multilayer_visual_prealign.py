"""Static D2 visual-prealignment plan gate for the multilayer model."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any

from kcg_connector.d38999_key_branch_selector import BRANCH_IDS


SCHEMA_VERSION = "kcg_d38999_multilayer_visual_prealign_v1"
LEGACY_BRANCH_IDS = ("YAW_0", "YAW_PI")
FROZEN_SOURCES = {
    "src/kcg_connector/isaac/postgrasp_palm_keyed_visual_runtime.py": (
        "d44cc1134098c2e38640dff4bca88c7a44e5cce4aaf19f71c74ac79579fb95a0"
    ),
    "src/kcg_connector/test/test_postgrasp_palm_keyed_visual_runtime.py": (
        "fecd63f92e61187b552354e9951aa7c920c7468375beb89cd2669a9b213f1a98"
    ),
    "artifacts/agent_control/tasks/EIGHT-HOUR-C9-MULTIVIEW-FUSION/"
    "TASK_RESULT.json": (
        "17fb7b32e6cfafce65b821d1fbea4947454bf4e2e2ffd4f7b3334721e523734a"
    ),
    "artifacts/agent_control/tasks/EIGHT-HOUR-C10-POSE-CONFIDENCE-REJECTION/"
    "TASK_RESULT.json": (
        "da8a73cbe9970dc535357068b1667cc78b929a85fb04435760e6a347b3c0f64d"
    ),
    "artifacts/agent_control/tasks/EIGHT-HOUR-D1-RECEPTACLE-SAFE-STANDOFF/"
    "TASK_RESULT.json": (
        "64d373c590e7b1ddb88ff961451367e369aaf677450ec796cd4f115d955179cc"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verified_sources(root: Path) -> list[dict[str, str]]:
    rows = []
    for relative, expected in FROZEN_SOURCES.items():
        path = root / relative
        if not path.is_file():
            raise ValueError(f"frozen D2 source missing: {relative}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"frozen D2 source hash mismatch: {relative}")
        rows.append({"path": relative, "sha256": actual})
    return rows


def _json_mapping(path: Path, label: str) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def evaluate_visual_prealign_readiness(
    pose_gate_result: Mapping[str, Any],
    standoff_result: Mapping[str, Any],
) -> dict[str, Any]:
    base = {
        "schema_version": SCHEMA_VERSION,
        "status": "REJECTED_SAFE_STOP",
        "rejection_code": None,
        "upstream_rejection_code": None,
        "T_HP_selected": None,
        "T_WP_target": None,
        "T_WH_target": None,
        "target_plan": None,
        "branch_mapping_authorized": False,
        "path_planning_authorized": False,
        "actuator_command_issued": False,
        "control_authorized": False,
        "simulation_started": False,
        "hardware_control_authorized": False,
    }
    if not isinstance(pose_gate_result, Mapping) or not isinstance(
        standoff_result, Mapping
    ):
        return {**base, "rejection_code": "UPSTREAM_EVIDENCE_INVALID"}
    if (
        pose_gate_result.get("task_id")
        != "EIGHT-HOUR-C10-POSE-CONFIDENCE-REJECTION"
        or pose_gate_result.get("outcome") != "OFFLINE_PASS"
        or standoff_result.get("task_id")
        != "EIGHT-HOUR-D1-RECEPTACLE-SAFE-STANDOFF"
        or standoff_result.get("outcome") != "OFFLINE_PASS"
    ):
        return {**base, "rejection_code": "UPSTREAM_EVIDENCE_INVALID"}
    if any(
        result.get("control_authorized") is not False
        or result.get("hardware_authorized") is not False
        for result in (pose_gate_result, standoff_result)
    ):
        return {**base, "rejection_code": "UPSTREAM_AUTHORIZATION_INVALID"}
    upstream_code = pose_gate_result.get("current_rejection_code")
    if not isinstance(upstream_code, str) or not upstream_code:
        return {**base, "rejection_code": "UPSTREAM_POSE_GATE_INVALID"}
    if standoff_result.get("current_dynamic_readiness") != "UPSTREAM_POSE_REJECTED":
        return {**base, "rejection_code": "STANDOFF_READINESS_INVALID"}
    return {
        **base,
        "rejection_code": "UPSTREAM_POSE_REJECTED",
        "upstream_rejection_code": upstream_code,
    }


def build_multilayer_visual_prealign_contract(
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    sources = _verified_sources(root)
    legacy_source = (
        root / "src/kcg_connector/isaac/postgrasp_palm_keyed_visual_runtime.py"
    ).read_text(encoding="utf-8")
    required_formulae = (
        "T_WP_target = T_WR_configured @ T_RP_target_configured",
        "T_WH_target = T_WP_target @ inverse(T_HP_selected)",
        "requires_collision_checked_path_planner",
        '"control_authorized": False',
        '"actuator_command_issued": False',
    )
    if any(value not in legacy_source for value in required_formulae):
        raise ValueError("legacy prealignment formula or safety boundary changed")
    base = root / "artifacts/agent_control/tasks"
    c9 = _json_mapping(base / "EIGHT-HOUR-C9-MULTIVIEW-FUSION/TASK_RESULT.json", "C9")
    c10 = _json_mapping(
        base / "EIGHT-HOUR-C10-POSE-CONFIDENCE-REJECTION/TASK_RESULT.json", "C10"
    )
    d1 = _json_mapping(
        base / "EIGHT-HOUR-D1-RECEPTACLE-SAFE-STANDOFF/TASK_RESULT.json", "D1"
    )
    if (
        c9.get("candidate_count") != 2
        or c9.get("dynamic_independent_views_proven") != 0
        or c10.get("current_rejection_code") != "CONFIDENCE_UNCALIBRATED"
        or d1.get("preinsert_gap_m") != 0.012
    ):
        raise ValueError("current D2 upstream evidence boundary changed")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "OFFLINE_PLAN_GATE_READY",
        "classification": "LEGACY_FORMULA_REUSED_WITH_MULTILAYER_FAIL_CLOSED_GATE",
        "transform_convention": "T_AB_IS_FRAME_B_EXPRESSED_IN_FRAME_A",
        "target_formulae": {
            "plug_target": "T_WP_target=T_WR_configured@T_RP_target_configured",
            "hand_target": "T_WH_target=T_WP_target@inverse(T_HP_selected)",
        },
        "current_branch_ids": list(BRANCH_IDS),
        "legacy_reference_branch_ids": list(LEGACY_BRANCH_IDS),
        "automatic_branch_id_mapping": None,
        "branch_mapping_authorized": False,
        "legacy_reference_scene": "keyed_v2_only",
        "legacy_positive_test_evidence": "HYPOTHETICAL_CPU_CALIBRATION_ONLY",
        "legacy_dynamic_evidence_promoted": False,
        "requires_collision_checked_path_planner": True,
        "insertion_motion_included": False,
        "truth_pose_input_allowed": False,
        "contact_truth_input_allowed": False,
        "event_truth_input_allowed": False,
        "current_readiness": evaluate_visual_prealign_readiness(c10, d1),
        "simulation_started": False,
        "robot_motion_started": False,
        "dynamic_prealign_pass_claimed": False,
        "control_authorized": False,
        "hardware_authorized": False,
        "sources": sources,
    }


__all__ = [
    "FROZEN_SOURCES",
    "LEGACY_BRANCH_IDS",
    "SCHEMA_VERSION",
    "build_multilayer_visual_prealign_contract",
    "evaluate_visual_prealign_readiness",
]
