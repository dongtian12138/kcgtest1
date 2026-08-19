from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from kcg_connector.d38999_multilayer_visual_prealign import (
    FROZEN_SOURCES,
    build_multilayer_visual_prealign_contract,
    evaluate_visual_prealign_readiness,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
TASKS = REPOSITORY_ROOT / "artifacts/agent_control/tasks"


def _record(task):
    return json.loads((TASKS / task / "TASK_RESULT.json").read_text())


def _inputs():
    return (
        _record("EIGHT-HOUR-C10-POSE-CONFIDENCE-REJECTION"),
        _record("EIGHT-HOUR-D1-RECEPTACLE-SAFE-STANDOFF"),
    )


def _copy_sources(tmp_path):
    root = tmp_path / "repository"
    for relative in FROZEN_SOURCES:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((REPOSITORY_ROOT / relative).read_bytes())
    return root


def test_contract_reuses_only_formula_and_keeps_model_identity_boundary():
    contract = build_multilayer_visual_prealign_contract(REPOSITORY_ROOT)
    assert contract["status"] == "OFFLINE_PLAN_GATE_READY"
    assert contract["target_formulae"] == {
        "plug_target": "T_WP_target=T_WR_configured@T_RP_target_configured",
        "hand_target": "T_WH_target=T_WP_target@inverse(T_HP_selected)",
    }
    assert contract["current_branch_ids"] == [
        "C2_LINKED_BRANCH_0", "C2_LINKED_BRANCH_PI"
    ]
    assert contract["legacy_reference_branch_ids"] == ["YAW_0", "YAW_PI"]
    assert contract["automatic_branch_id_mapping"] is None
    assert contract["branch_mapping_authorized"] is False


def test_legacy_hypothetical_calibration_is_not_promoted():
    contract = build_multilayer_visual_prealign_contract(REPOSITORY_ROOT)
    assert contract["legacy_reference_scene"] == "keyed_v2_only"
    assert contract["legacy_positive_test_evidence"] == (
        "HYPOTHETICAL_CPU_CALIBRATION_ONLY"
    )
    assert contract["legacy_dynamic_evidence_promoted"] is False
    assert contract["dynamic_prealign_pass_claimed"] is False


def test_current_upstream_rejection_emits_no_pose_or_plan():
    result = evaluate_visual_prealign_readiness(*_inputs())
    assert result["rejection_code"] == "UPSTREAM_POSE_REJECTED"
    assert result["upstream_rejection_code"] == "CONFIDENCE_UNCALIBRATED"
    assert result["T_HP_selected"] is None
    assert result["T_WP_target"] is None
    assert result["T_WH_target"] is None
    assert result["target_plan"] is None
    assert result["path_planning_authorized"] is False
    assert result["control_authorized"] is False


@pytest.mark.parametrize(
    ("index", "key", "value", "code"),
    (
        (0, "task_id", "OTHER", "UPSTREAM_EVIDENCE_INVALID"),
        (1, "outcome", "BLOCKED", "UPSTREAM_EVIDENCE_INVALID"),
        (0, "control_authorized", True, "UPSTREAM_AUTHORIZATION_INVALID"),
        (1, "hardware_authorized", True, "UPSTREAM_AUTHORIZATION_INVALID"),
        (0, "current_rejection_code", None, "UPSTREAM_POSE_GATE_INVALID"),
        (1, "current_dynamic_readiness", "READY", "STANDOFF_READINESS_INVALID"),
    ),
)
def test_upstream_mutations_fail_closed(index, key, value, code):
    records = list(_inputs())
    records[index][key] = value
    result = evaluate_visual_prealign_readiness(*records)
    assert result["rejection_code"] == code
    assert result["T_WH_target"] is None
    assert result["control_authorized"] is False


def test_wrong_input_type_fails_closed():
    _, d1 = _inputs()
    result = evaluate_visual_prealign_readiness(None, d1)
    assert result["rejection_code"] == "UPSTREAM_EVIDENCE_INVALID"
    assert result["control_authorized"] is False


def test_frozen_source_drift_is_rejected(tmp_path):
    root = _copy_sources(tmp_path)
    target = root / next(iter(FROZEN_SOURCES))
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        build_multilayer_visual_prealign_contract(root)


def test_api_has_no_pose_contact_branch_or_truth_input():
    parameters = set(inspect.signature(evaluate_visual_prealign_readiness).parameters)
    assert parameters == {"pose_gate_result", "standoff_result"}
    assert parameters.isdisjoint(
        {"object_pose", "ground_truth_pose", "contact_name", "contact_normal", "event_truth", "selected_branch"}
    )


def test_contract_is_finite_json_and_never_authorizes_control():
    contract = build_multilayer_visual_prealign_contract(REPOSITORY_ROOT)
    json.dumps(contract, sort_keys=True, allow_nan=False)
    assert contract["requires_collision_checked_path_planner"] is True
    assert contract["insertion_motion_included"] is False
    assert contract["simulation_started"] is False
    assert contract["robot_motion_started"] is False
    assert contract["control_authorized"] is False
