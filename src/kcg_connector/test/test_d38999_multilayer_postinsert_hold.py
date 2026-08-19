from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from kcg_connector.d38999_multilayer_postinsert_hold import (
    FROZEN_SOURCES,
    PostInsertHoldReadiness,
    build_postinsert_hold_contract,
    evaluate_postinsert_hold_gate,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
AUTHORITY = {
    "source_path": "offline-test-fixture.json",
    "source_sha256": "a" * 64,
    "hold_duration_s": 0.5,
}


def _ready(**overrides):
    values = {
        "formal_insertion_dynamic_pass": True,
        "body_grasp_dynamic_pass": True,
        "wrist_guard_safe": True,
        "wrist_guard_fault_latched": False,
        "insertion_evidence_id": "insertion-run-1",
        "grasp_evidence_id": "grasp-run-1",
    }
    values.update(overrides)
    return PostInsertHoldReadiness(**values)


def _evaluate(readiness=None, **overrides):
    values = {
        "timing_authority": AUTHORITY,
        "now_s": 1.25,
        "hold_started_s": 1.0,
        "wrench_timestamp_s": 1.24,
        "maximum_wrench_sample_age_s": 0.020,
    }
    values.update(overrides)
    return evaluate_postinsert_hold_gate(readiness or _ready(), **values)


def test_contract_binds_upstream_and_reports_current_zero_output():
    contract = build_postinsert_hold_contract(REPOSITORY_ROOT)
    assert contract["maximum_wrench_sample_age_s"] == pytest.approx(0.020)
    assert contract["formal_moment_component_limit_nm"] == pytest.approx(0.30)
    assert contract["authoritative_hold_duration_s"] is None
    assert contract["hold_duration_authority_required"] is True
    assert contract["current_decision"]["rejection_code"] == "D7_INSERTION_NOT_DYNAMIC"
    assert contract["current_decision"]["hold_request_candidate"] is None


def test_valid_logic_fixture_produces_zero_twist_candidate_only():
    result = _evaluate()
    assert result["status"] == "OFFLINE_HOLD_REQUEST_CANDIDATE"
    assert result["hold_request_candidate"]["cartesian_twist_task"] == [0.0] * 6
    assert result["hold_request_candidate"]["preserve_current_gripper_setpoint"] is True
    assert result["hold_request_candidate"]["remaining_s"] == pytest.approx(0.25)
    assert result["hold_complete_candidate"] is False


def test_elapsed_authorized_duration_marks_candidate_complete_only():
    result = _evaluate(now_s=1.5, wrench_timestamp_s=1.49)
    assert result["status"] == "OFFLINE_HOLD_COMPLETE_CANDIDATE"
    assert result["hold_complete_candidate"] is True
    assert result["control_authorized"] is False


@pytest.mark.parametrize(
    "overrides, code",
    [
        ({"formal_insertion_dynamic_pass": False}, "D7_INSERTION_NOT_DYNAMIC"),
        ({"body_grasp_dynamic_pass": False}, "B4_BODY_GRASP_NOT_DYNAMIC"),
        ({"insertion_evidence_id": None}, "DYNAMIC_EVIDENCE_ID_MISSING"),
        ({"grasp_evidence_id": ""}, "DYNAMIC_EVIDENCE_ID_MISSING"),
        ({"wrist_guard_safe": False}, "WRIST_MOMENT_GUARD_REJECTED"),
        ({"wrist_guard_fault_latched": True}, "WRIST_MOMENT_GUARD_REJECTED"),
    ],
)
def test_readiness_gates_fail_closed(overrides, code):
    assert _evaluate(_ready(**overrides))["rejection_code"] == code


@pytest.mark.parametrize(
    "authority",
    [
        None,
        {},
        {**AUTHORITY, "source_sha256": "bad"},
        {**AUTHORITY, "hold_duration_s": 0.0},
        {**AUTHORITY, "source_path": ""},
    ],
)
def test_hold_duration_requires_explicit_traceable_authority(authority):
    result = _evaluate(timing_authority=authority)
    assert result["rejection_code"] == "HOLD_DURATION_AUTHORITY_MISSING"
    assert result["hold_request_candidate"] is None


def test_wrench_sample_exactly_at_age_limit_is_accepted():
    result = _evaluate(now_s=1.25, wrench_timestamp_s=1.23)
    assert result["status"] == "OFFLINE_HOLD_REQUEST_CANDIDATE"


def test_stale_wrench_is_rejected():
    result = _evaluate(now_s=1.25, wrench_timestamp_s=1.229999)
    assert result["rejection_code"] == "WRIST_WRENCH_STALE"


@pytest.mark.parametrize(
    "overrides",
    [
        {"now_s": float("nan")},
        {"hold_started_s": 2.0},
        {"wrench_timestamp_s": 2.0},
        {"maximum_wrench_sample_age_s": 0.0},
    ],
)
def test_invalid_timing_fails_closed(overrides):
    assert _evaluate(**overrides)["rejection_code"] == "INVALID_HOLD_TIMING"


def test_boolean_readiness_is_strict():
    result = _evaluate(_ready(formal_insertion_dynamic_pass=1))
    assert result["rejection_code"] == "INVALID_READINESS_SNAPSHOT"


def test_all_paths_emit_no_commands_or_dynamic_claim():
    for result in (_evaluate(), _evaluate(_ready(formal_insertion_dynamic_pass=False))):
        assert result["motion_command_emitted"] is False
        assert result["gripper_command_emitted"] is False
        assert result["control_authorized"] is False
        assert result["dynamic_postinsert_hold_pass_claimed"] is False
        assert result["hardware_authorized"] is False


def test_truth_firewall_is_absent_from_public_signature():
    names = set(inspect.signature(evaluate_postinsert_hold_gate).parameters)
    assert names.isdisjoint(
        {"object_pose", "contact_name", "contact_normal", "event_truth", "collider_path"}
    )


def test_frozen_source_drift_is_rejected(tmp_path):
    root = tmp_path / "repository"
    for relative in FROZEN_SOURCES:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((REPOSITORY_ROOT / relative).read_bytes())
    target = root / next(iter(FROZEN_SOURCES))
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        build_postinsert_hold_contract(root)


def test_decision_is_strict_json():
    json.dumps(_evaluate(), sort_keys=True, allow_nan=False)
