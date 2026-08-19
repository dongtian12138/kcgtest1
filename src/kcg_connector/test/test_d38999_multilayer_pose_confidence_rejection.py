from __future__ import annotations

from dataclasses import replace
import inspect
import json
from pathlib import Path

import pytest

from kcg_connector.d38999_multilayer_pose_confidence_rejection import (
    FROZEN_SOURCES,
    PoseConfidenceSnapshot,
    build_pose_confidence_rejection_contract,
    evaluate_pose_confidence_snapshot,
    evaluate_recorded_c9_pose_confidence,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
BRANCH_IDS = ("C2_LINKED_BRANCH_0", "C2_LINKED_BRANCH_PI")


def _ready_snapshot() -> PoseConfidenceSnapshot:
    return PoseConfidenceSnapshot(
        source_contract_valid=True,
        source_contract_detail=None,
        fusion_schema_valid=True,
        fusion_status="DYNAMIC_PASS",
        diagnostics_finite=True,
        confidence_calibrated=True,
        dynamic_independent_views_proven=2,
        dynamic_evidence_present=True,
        candidate_branch_ids=BRANCH_IDS,
        c2_relation_preserved=True,
        selected_branch_id=BRANCH_IDS[0],
    )


def _copy_sources(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    for relative in FROZEN_SOURCES:
        source = REPOSITORY_ROOT / relative
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    return root


def test_current_contract_is_rejection_only_and_uses_no_invented_threshold():
    contract = build_pose_confidence_rejection_contract(REPOSITORY_ROOT)
    assert contract["status"] == "OFFLINE_REJECTION_INTERFACE_READY"
    assert contract["current_rejection_code"] == "CONFIDENCE_UNCALIBRATED"
    assert contract["confidence_threshold_defined"] is False
    assert contract["confidence_calibration_fabricated"] is False
    assert contract["selected_for_control"] is None
    assert contract["control_authorized"] is False


def test_recorded_c9_evidence_has_one_expected_primary_rejection():
    result = evaluate_recorded_c9_pose_confidence(REPOSITORY_ROOT)
    assert result["status"] == "REJECTED"
    assert result["rejection_code"] == "CONFIDENCE_UNCALIBRATED"
    assert result["pose_rejected"] is True
    assert result["confidence_calibrated"] is False
    assert result["dynamic_independent_views_proven"] == 0
    assert result["control_authorized"] is False
    assert "rejection_codes" not in result


def test_source_hash_drift_returns_a_structured_primary_rejection(tmp_path):
    root = _copy_sources(tmp_path)
    target = root / next(iter(FROZEN_SOURCES))
    target.write_bytes(target.read_bytes() + b"\n")
    result = evaluate_recorded_c9_pose_confidence(root)
    assert result["rejection_code"] == "SOURCE_CONTRACT_INVALID"
    assert result["source_contract_valid"] is False
    assert result["control_authorized"] is False


def test_contract_builder_refuses_source_hash_drift(tmp_path):
    root = _copy_sources(tmp_path)
    target = root / next(iter(FROZEN_SOURCES))
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SOURCE_CONTRACT_INVALID"):
        build_pose_confidence_rejection_contract(root)


@pytest.mark.parametrize(
    ("changes", "expected"),
    (
        ({"fusion_schema_valid": False}, "FUSION_SCHEMA_INVALID"),
        ({"fusion_status": "ERROR"}, "FUSION_STATUS_INVALID"),
        ({"ground_truth_object_pose_used": True}, "TRUTH_FIREWALL_VIOLATION"),
        ({"diagnostics_finite": False}, "NONFINITE_DIAGNOSTIC"),
        ({"confidence_calibrated": False}, "CONFIDENCE_UNCALIBRATED"),
        ({"dynamic_independent_views_proven": 0}, "NO_DYNAMIC_INDEPENDENT_VIEWS"),
        ({"dynamic_evidence_present": False}, "DYNAMIC_EVIDENCE_MISSING"),
        ({"candidate_branch_ids": ("ONLY_ONE",)}, "C2_CONTRACT_INVALID"),
        ({"c2_relation_preserved": False}, "C2_CONTRACT_INVALID"),
        ({"selected_branch_id": None}, "C2_UNRESOLVED"),
        ({"selected_branch_id": "UNKNOWN"}, "C2_SELECTION_INVALID"),
    ),
)
def test_each_required_failure_path_is_closed(changes, expected):
    result = evaluate_pose_confidence_snapshot(replace(_ready_snapshot(), **changes))
    assert result["rejection_code"] == expected
    assert result["pose_valid"] is False
    assert result["selected_for_control"] is None
    assert result["control_authorized"] is False


def test_rejection_precedence_is_deterministic_and_returns_only_one_code():
    snapshot = replace(
        _ready_snapshot(),
        fusion_status="ERROR",
        diagnostics_finite=False,
        confidence_calibrated=False,
        dynamic_independent_views_proven=0,
        selected_branch_id=None,
    )
    first = evaluate_pose_confidence_snapshot(snapshot)
    second = evaluate_pose_confidence_snapshot(snapshot)
    assert first == second
    assert first["rejection_code"] == "FUSION_STATUS_INVALID"
    assert "rejection_codes" not in first


def test_even_ready_facts_cannot_promote_control_in_c10():
    result = evaluate_pose_confidence_snapshot(_ready_snapshot())
    assert result["rejection_code"] == "CONTROL_PROMOTION_NOT_AUTHORIZED_BY_C10"
    assert result["pose_valid"] is False
    assert result["control_authorized"] is False
    assert result["simulation_prealign_control_authorized"] is False
    assert result["simulation_insertion_control_authorized"] is False
    assert result["hardware_control_authorized"] is False


def test_invalid_types_fail_closed_instead_of_coercing_booleans():
    snapshot = replace(_ready_snapshot(), dynamic_independent_views_proven=True)
    result = evaluate_pose_confidence_snapshot(snapshot)
    assert result["rejection_code"] == "INVALID_GATE_INPUT"
    assert result["dynamic_independent_views_proven"] == 0


def test_result_is_strict_json_without_nonfinite_values():
    result = evaluate_recorded_c9_pose_confidence(REPOSITORY_ROOT)
    encoded = json.dumps(result, sort_keys=True, allow_nan=False)
    assert "NaN" not in encoded
    assert "Infinity" not in encoded


def test_public_api_has_no_pose_contact_event_or_selection_truth_inputs():
    forbidden = {
        "ground_truth_pose",
        "object_pose",
        "contact_name",
        "contact_normal",
        "event_truth",
        "selected_branch",
    }
    for function in (
        evaluate_pose_confidence_snapshot,
        evaluate_recorded_c9_pose_confidence,
    ):
        assert set(inspect.signature(function).parameters).isdisjoint(forbidden)
