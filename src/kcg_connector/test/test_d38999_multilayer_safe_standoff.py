from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from kcg_connector.d38999_multilayer_safe_standoff import (
    FROZEN_SOURCES,
    build_multilayer_safe_standoff_contract,
    evaluate_safe_standoff_readiness,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _current_c10():
    return json.loads(
        (REPOSITORY_ROOT / (
            "artifacts/agent_control/tasks/"
            "EIGHT-HOUR-C10-POSE-CONFIDENCE-REJECTION/TASK_RESULT.json"
        )).read_text(encoding="utf-8")
    )


def _copy_sources(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    for relative in FROZEN_SOURCES:
        source = REPOSITORY_ROOT / relative
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    return root


def test_contract_preserves_authoritative_datum_axis_and_two_distinct_targets():
    contract = build_multilayer_safe_standoff_contract(REPOSITORY_ROOT)
    assert contract["status"] == "OFFLINE_CONTRACT_READY"
    assert contract["coordinate_frame"] == "world"
    assert contract["fixed_datum"]["position_world_m"] == pytest.approx(
        (0.550, 0.185, 0.2615)
    )
    assert contract["assembly_axis_world"] == pytest.approx((0.0, 0.0, 1.0))
    assert contract["approach_direction_world"] == pytest.approx((0.0, 0.0, -1.0))
    assert contract["plug_face_target_world_m"] == pytest.approx(
        (0.550, 0.185, 0.2735)
    )
    assert contract["nominal_tcp_target_world_m"] == pytest.approx(
        (0.550, 0.185, 0.32198)
    )
    assert contract["nominal_tcp_from_plug_face_m"] == pytest.approx(
        (0.0, 0.0, 0.04848)
    )


def test_gap_and_margin_are_cross_checked_not_reinterpreted_as_clearance():
    contract = build_multilayer_safe_standoff_contract(REPOSITORY_ROOT)
    assert contract["gap_definition"] == "dot(P-F,Fz)"
    assert contract["preinsert_gap_m"] == pytest.approx(0.012)
    assert contract["entry_gap_m"] == pytest.approx(0.010)
    assert contract["registered_margin_before_entry_m"] == pytest.approx(0.002)
    assert contract["margin_semantics"] == (
        "REGISTERED_GEOMETRY_NOT_MEASURED_COLLISION_CLEARANCE"
    )


def test_current_c10_rejection_blocks_target_and_motion():
    result = evaluate_safe_standoff_readiness(_current_c10())
    assert result["rejection_code"] == "UPSTREAM_POSE_REJECTED"
    assert result["upstream_rejection_code"] == "CONFIDENCE_UNCALIBRATED"
    assert result["target_pose_emitted"] is False
    assert result["path_planning_authorized"] is False
    assert result["actuator_command_issued"] is False
    assert result["control_authorized"] is False


def test_contract_exposes_current_rejection_without_dynamic_claim():
    contract = build_multilayer_safe_standoff_contract(REPOSITORY_ROOT)
    assert contract["current_readiness"]["rejection_code"] == (
        "UPSTREAM_POSE_REJECTED"
    )
    assert contract["simulation_started"] is False
    assert contract["robot_motion_started"] is False
    assert contract["dynamic_standoff_pass_claimed"] is False
    assert contract["control_authorized"] is False


@pytest.mark.parametrize(
    ("mutation", "expected"),
    (
        ({"task_id": "OTHER"}, "UPSTREAM_POSE_GATE_INVALID"),
        ({"outcome": "DYNAMIC_PASS"}, "UPSTREAM_POSE_GATE_INVALID"),
        ({"control_authorized": True}, "UPSTREAM_AUTHORIZATION_INVALID"),
        ({"hardware_authorized": True}, "UPSTREAM_AUTHORIZATION_INVALID"),
        ({"current_rejection_code": None}, "UPSTREAM_POSE_GATE_INVALID"),
        ({"dynamic_pose_pass_claimed": True}, "UPSTREAM_POSE_GATE_INVALID"),
        ({"selected_for_control": "C2_LINKED_BRANCH_0"}, "UPSTREAM_POSE_GATE_INVALID"),
    ),
)
def test_upstream_contract_mutations_fail_closed(mutation, expected):
    record = _current_c10()
    record.update(mutation)
    result = evaluate_safe_standoff_readiness(record)
    assert result["rejection_code"] == expected
    assert result["target_pose_emitted"] is False
    assert result["control_authorized"] is False


def test_wrong_upstream_type_fails_closed():
    result = evaluate_safe_standoff_readiness(None)
    assert result["rejection_code"] == "UPSTREAM_POSE_GATE_INVALID"
    assert result["control_authorized"] is False


def test_frozen_source_tamper_is_rejected_before_contract_use(tmp_path):
    root = _copy_sources(tmp_path)
    target = root / next(iter(FROZEN_SOURCES))
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        build_multilayer_safe_standoff_contract(root)


def test_public_readiness_api_has_no_pose_contact_or_event_input():
    parameters = set(inspect.signature(evaluate_safe_standoff_readiness).parameters)
    assert parameters == {"pose_gate_result"}
    assert parameters.isdisjoint(
        {"object_pose", "ground_truth_pose", "contact_name", "contact_normal", "event_truth"}
    )


def test_contract_is_strict_finite_json():
    contract = build_multilayer_safe_standoff_contract(REPOSITORY_ROOT)
    encoded = json.dumps(contract, sort_keys=True, allow_nan=False)
    assert "NaN" not in encoded and "Infinity" not in encoded
