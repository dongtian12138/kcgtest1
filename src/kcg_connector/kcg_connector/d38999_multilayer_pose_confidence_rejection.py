"""Fail-closed confidence rejection for the uncalibrated C2 pose chain.

This module deliberately has no control-promotion path.  It converts evidence
facts into one deterministic rejection code and binds the current decision to
the frozen C9 artifacts.  Raw object truth, contacts and event truth are not
inputs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from kcg_connector.d38999_key_branch_selector import BRANCH_IDS


SCHEMA_VERSION = "kcg_d38999_multilayer_pose_confidence_rejection_v1"
C9_SCHEMA_VERSION = "kcg_d38999_multilayer_multiview_fusion_v1"

FROZEN_SOURCES = {
    "artifacts/agent_control/tasks/EIGHT-HOUR-C9-MULTIVIEW-FUSION/"
    "TASK_RESULT.json": (
        "17fb7b32e6cfafce65b821d1fbea4947454bf4e2e2ffd4f7b3334721e523734a"
    ),
    "artifacts/agent_control/tasks/EIGHT-HOUR-C9-MULTIVIEW-FUSION/"
    "FUSION_CONTRACT_MANIFEST.json": (
        "a3842c86c66b4a471fee02aed7a2a84905ade2722ae6424c70d42adc0efdb90c"
    ),
    "artifacts/agent_control/tasks/EIGHT-HOUR-C9-MULTIVIEW-FUSION/"
    "OFFLINE_VALIDATION.json": (
        "4d673e6abc9ead754ba4fd86fff67723322564dd88656ef669042a2deb00d503"
    ),
    "src/kcg_connector/kcg_connector/d38999_multilayer_multiview_fusion.py": (
        "ff2222023a8ce82e4eb7fd93a42d660c61dd8d75c97ab4095070b6e54a379bf0"
    ),
    "src/kcg_connector/test/test_d38999_multilayer_multiview_fusion.py": (
        "b3697c6d10b759809b7e92126c1a4ac429a8887d9f03f6218ad9b57fac40e642"
    ),
}

REJECTION_REASONS = {
    "INVALID_GATE_INPUT": "拒识快照字段类型或取值无效",
    "SOURCE_CONTRACT_INVALID": "冻结来源缺失或摘要漂移",
    "FUSION_SCHEMA_INVALID": "多视角融合证据模式不匹配",
    "FUSION_STATUS_INVALID": "多视角融合状态不是可评估状态",
    "TRUTH_FIREWALL_VIOLATION": "证据声明使用了禁用真值通道",
    "NONFINITE_DIAGNOSTIC": "位姿诊断含非有限数值",
    "CONFIDENCE_UNCALIBRATED": "位姿置信度尚未由权威数据标定",
    "NO_DYNAMIC_INDEPENDENT_VIEWS": "没有动态独立视角证据",
    "DYNAMIC_EVIDENCE_MISSING": "动态证据记录缺失",
    "C2_CONTRACT_INVALID": "两个C2分支或其π关系不完整",
    "C2_UNRESOLVED": "C2分支尚未由允许的图像证据消解",
    "C2_SELECTION_INVALID": "C2分支选择不属于冻结分支集合",
    "CONTROL_PROMOTION_NOT_AUTHORIZED_BY_C10": "C10只负责拒识，不授予控制权限",
}


@dataclass(frozen=True)
class PoseConfidenceSnapshot:
    """Evidence-only facts consumed by the rejection evaluator."""

    source_contract_valid: bool
    source_contract_detail: str | None
    fusion_schema_valid: bool
    fusion_status: str
    diagnostics_finite: bool
    confidence_calibrated: bool
    dynamic_independent_views_proven: int
    dynamic_evidence_present: bool
    candidate_branch_ids: tuple[str, ...]
    c2_relation_preserved: bool
    selected_branch_id: str | None
    ground_truth_object_pose_used: bool = False
    semantic_truth_used: bool = False
    contact_truth_used: bool = False
    event_truth_used: bool = False


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_mapping(path: Path, label: str) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _source_state(root: Path) -> tuple[bool, str | None, list[dict[str, str]]]:
    rows: list[dict[str, str]] = []
    for relative, expected in FROZEN_SOURCES.items():
        path = root / relative
        if not path.is_file():
            return False, f"missing:{relative}", rows
        actual = _sha256(path)
        if actual != expected:
            return False, f"sha256_mismatch:{relative}", rows
        rows.append({"path": relative, "sha256": actual})
    return True, None, rows


def _valid_snapshot_types(snapshot: PoseConfidenceSnapshot) -> bool:
    booleans = (
        snapshot.source_contract_valid,
        snapshot.fusion_schema_valid,
        snapshot.diagnostics_finite,
        snapshot.confidence_calibrated,
        snapshot.dynamic_evidence_present,
        snapshot.c2_relation_preserved,
        snapshot.ground_truth_object_pose_used,
        snapshot.semantic_truth_used,
        snapshot.contact_truth_used,
        snapshot.event_truth_used,
    )
    return bool(
        all(type(value) is bool for value in booleans)
        and isinstance(snapshot.source_contract_detail, (str, type(None)))
        and isinstance(snapshot.fusion_status, str)
        and type(snapshot.dynamic_independent_views_proven) is int
        and snapshot.dynamic_independent_views_proven >= 0
        and isinstance(snapshot.candidate_branch_ids, tuple)
        and all(isinstance(value, str) for value in snapshot.candidate_branch_ids)
        and isinstance(snapshot.selected_branch_id, (str, type(None)))
    )


def _rejected(code: str, snapshot: PoseConfidenceSnapshot) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "REJECTED",
        "classification": "POSE_CONFIDENCE_FAIL_CLOSED_REJECTION",
        "passed": False,
        "pose_valid": False,
        "pose_rejected": True,
        "rejection_code": code,
        "reason_cn": REJECTION_REASONS[code],
        "source_contract_valid": (
            snapshot.source_contract_valid
            if type(snapshot.source_contract_valid) is bool
            else False
        ),
        "source_contract_detail": snapshot.source_contract_detail,
        "confidence_calibrated": (
            snapshot.confidence_calibrated
            if type(snapshot.confidence_calibrated) is bool
            else False
        ),
        "dynamic_independent_views_proven": (
            snapshot.dynamic_independent_views_proven
            if type(snapshot.dynamic_independent_views_proven) is int
            else 0
        ),
        "candidate_branch_ids": list(snapshot.candidate_branch_ids),
        "selected_for_control": None,
        "control_authorized": False,
        "simulation_prealign_control_authorized": False,
        "simulation_insertion_control_authorized": False,
        "hardware_control_authorized": False,
        "ground_truth_object_pose_used": False,
        "contact_truth_used": False,
        "event_truth_used": False,
    }


def evaluate_pose_confidence_snapshot(
    snapshot: PoseConfidenceSnapshot,
) -> dict[str, Any]:
    """Return exactly one deterministic rejection reason.

    Even a future snapshot satisfying every prerequisite is rejected at the
    final C10 scope boundary.  A separate, explicitly authorized contract is
    required before any controller may consume a selected pose.
    """

    if not isinstance(snapshot, PoseConfidenceSnapshot) or not _valid_snapshot_types(snapshot):
        fallback = snapshot if isinstance(snapshot, PoseConfidenceSnapshot) else PoseConfidenceSnapshot(
            source_contract_valid=False,
            source_contract_detail="wrong_snapshot_type",
            fusion_schema_valid=False,
            fusion_status="",
            diagnostics_finite=False,
            confidence_calibrated=False,
            dynamic_independent_views_proven=0,
            dynamic_evidence_present=False,
            candidate_branch_ids=(),
            c2_relation_preserved=False,
            selected_branch_id=None,
        )
        return _rejected("INVALID_GATE_INPUT", fallback)
    if not snapshot.source_contract_valid:
        return _rejected("SOURCE_CONTRACT_INVALID", snapshot)
    if not snapshot.fusion_schema_valid:
        return _rejected("FUSION_SCHEMA_INVALID", snapshot)
    if snapshot.fusion_status not in {"OFFLINE_PASS", "DYNAMIC_PASS"}:
        return _rejected("FUSION_STATUS_INVALID", snapshot)
    if any(
        (
            snapshot.ground_truth_object_pose_used,
            snapshot.semantic_truth_used,
            snapshot.contact_truth_used,
            snapshot.event_truth_used,
        )
    ):
        return _rejected("TRUTH_FIREWALL_VIOLATION", snapshot)
    if not snapshot.diagnostics_finite:
        return _rejected("NONFINITE_DIAGNOSTIC", snapshot)
    if not snapshot.confidence_calibrated:
        return _rejected("CONFIDENCE_UNCALIBRATED", snapshot)
    if snapshot.dynamic_independent_views_proven == 0:
        return _rejected("NO_DYNAMIC_INDEPENDENT_VIEWS", snapshot)
    if not snapshot.dynamic_evidence_present:
        return _rejected("DYNAMIC_EVIDENCE_MISSING", snapshot)
    if (
        snapshot.candidate_branch_ids != tuple(BRANCH_IDS)
        or not snapshot.c2_relation_preserved
    ):
        return _rejected("C2_CONTRACT_INVALID", snapshot)
    if snapshot.selected_branch_id is None:
        return _rejected("C2_UNRESOLVED", snapshot)
    if snapshot.selected_branch_id not in BRANCH_IDS:
        return _rejected("C2_SELECTION_INVALID", snapshot)
    return _rejected("CONTROL_PROMOTION_NOT_AUTHORIZED_BY_C10", snapshot)


def _recorded_snapshot(root: Path) -> PoseConfidenceSnapshot:
    source_valid, source_detail, _ = _source_state(root)
    if not source_valid:
        return PoseConfidenceSnapshot(
            source_contract_valid=False,
            source_contract_detail=source_detail,
            fusion_schema_valid=False,
            fusion_status="SOURCE_UNREADABLE",
            diagnostics_finite=False,
            confidence_calibrated=False,
            dynamic_independent_views_proven=0,
            dynamic_evidence_present=False,
            candidate_branch_ids=(),
            c2_relation_preserved=False,
            selected_branch_id=None,
        )
    base = root / "artifacts/agent_control/tasks/EIGHT-HOUR-C9-MULTIVIEW-FUSION"
    task = _json_mapping(base / "TASK_RESULT.json", "C9 task result")
    manifest = _json_mapping(base / "FUSION_CONTRACT_MANIFEST.json", "C9 contract")
    validation = _json_mapping(base / "OFFLINE_VALIDATION.json", "C9 validation")
    fixture = validation.get("fixture", {})
    numeric_diagnostics = (
        fixture.get("maximum_pairwise_translation_spread_m"),
        fixture.get("maximum_pairwise_rotation_spread_rad"),
        fixture.get("translation_residual_rms_m"),
        fixture.get("maximum_rotation_residual_rms_rad"),
    )
    diagnostics_finite = bool(
        all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in numeric_diagnostics
        )
    )
    readiness = manifest.get("current_readiness", {})
    firewall = manifest.get("truth_firewall", {})
    truth_used = not bool(
        firewall
        and all(value is False for value in firewall.values())
    )
    branches = manifest.get("candidate_branch_ids", ())
    if not isinstance(branches, list):
        branches = ()
    return PoseConfidenceSnapshot(
        source_contract_valid=True,
        source_contract_detail=None,
        fusion_schema_valid=bool(
            manifest.get("schema_version") == C9_SCHEMA_VERSION
            and manifest.get("classification")
            == "TRUTH_FREE_C2_MULTIVIEW_FUSION_INTERFACE"
        ),
        fusion_status=str(task.get("outcome", "")),
        diagnostics_finite=diagnostics_finite,
        confidence_calibrated=bool(
            task.get("confidence_calibrated") is True
            and manifest.get("confidence_calibrated") is True
        ),
        dynamic_independent_views_proven=(
            task.get("dynamic_independent_views_proven")
            if type(task.get("dynamic_independent_views_proven")) is int
            else 0
        ),
        dynamic_evidence_present=bool(
            task.get("dynamic_multiview_fusion_pass_claimed") is True
        ),
        candidate_branch_ids=tuple(branches),
        c2_relation_preserved=tuple(branches) == tuple(BRANCH_IDS),
        selected_branch_id=(
            readiness.get("selected_for_control")
            if isinstance(readiness, Mapping)
            else None
        ),
        ground_truth_object_pose_used=truth_used,
        semantic_truth_used=truth_used,
        contact_truth_used=truth_used,
        event_truth_used=truth_used,
    )


def evaluate_recorded_c9_pose_confidence(
    repository_root: str | Path,
) -> dict[str, Any]:
    """Evaluate only the frozen C9 record currently present on disk."""

    return evaluate_pose_confidence_snapshot(
        _recorded_snapshot(Path(repository_root).resolve())
    )


def build_pose_confidence_rejection_contract(
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    source_valid, source_detail, sources = _source_state(root)
    if not source_valid:
        raise ValueError(f"SOURCE_CONTRACT_INVALID:{source_detail}")
    current = evaluate_recorded_c9_pose_confidence(root)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "OFFLINE_REJECTION_INTERFACE_READY",
        "classification": "FAIL_CLOSED_NO_CONTROL_PROMOTION_PATH",
        "rejection_precedence": list(REJECTION_REASONS),
        "current_rejection_code": current["rejection_code"],
        "confidence_threshold_defined": False,
        "confidence_calibration_fabricated": False,
        "c2_branch_selected": False,
        "dynamic_independent_views_proven": 0,
        "selected_for_control": None,
        "control_authorized": False,
        "simulation_started": False,
        "dynamic_pose_pass_claimed": False,
        "hardware_authorized": False,
        "sources": sources,
    }


__all__ = [
    "FROZEN_SOURCES",
    "PoseConfidenceSnapshot",
    "REJECTION_REASONS",
    "SCHEMA_VERSION",
    "build_pose_confidence_rejection_contract",
    "evaluate_pose_confidence_snapshot",
    "evaluate_recorded_c9_pose_confidence",
]
