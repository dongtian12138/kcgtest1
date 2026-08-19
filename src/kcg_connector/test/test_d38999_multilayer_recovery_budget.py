from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from kcg_connector.d38999_multilayer_recovery_budget import (
    EXPECTED_D9_STAGE_SEQUENCE,
    FROZEN_SOURCES,
    MAXIMUM_RECOVERY_ATTEMPTS,
    load_recovery_budget_contract,
    new_recovery_budget_state,
    request_recovery_slot,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _request(state=None, index=1, **overrides):
    values = {
        "request_id": f"recovery-{index}",
        "d8_exit_latched": True,
        "d8_failure_reason": "EXPERIMENTAL_AXIAL_FORCE_ABORT",
        "d8_exit_event_id": f"d8-exit-{index}",
        "d9_plan_completed": True,
        "d9_plan_stage_sequence": list(EXPECTED_D9_STAGE_SEQUENCE),
        "fresh_reobservation_authorized": True,
        "fresh_reobservation_evidence_id": f"reobserve-{index}",
    }
    values.update(overrides)
    return request_recovery_slot(state or new_recovery_budget_state(), **values)


def _two_attempt_state():
    first = _request(index=1)
    return _request(first["state"], index=2)["state"]


def test_frozen_contract_binds_d8_and_d9_results():
    contract = load_recovery_budget_contract(REPOSITORY_ROOT)
    assert contract["maximum_recovery_attempts"] == 2
    assert contract["backoff_distance_m"] == pytest.approx(0.0004)
    assert contract["backoff_speed_m_s"] == pytest.approx(0.0003)
    assert contract["stage_sequence"] == EXPECTED_D9_STAGE_SEQUENCE
    assert len(contract["sources"]) == 2
    assert contract["execution_authorized"] is False


def test_empty_budget_is_safe_and_json_serializable():
    state = new_recovery_budget_state()
    assert state["attempts_reserved"] == []
    assert state["maximum_recovery_attempts"] == MAXIMUM_RECOVERY_ATTEMPTS
    assert state["terminal_safe_abort"] is False
    json.dumps(state, sort_keys=True, allow_nan=False)


def test_first_and_second_slots_are_reserved_monotonically():
    first = _request(index=1)
    second = _request(first["state"], index=2)
    assert first["attempt_number"] == 1
    assert second["attempt_number"] == 2
    assert [row["attempt_number"] for row in second["state"]["attempts_reserved"]] == [1, 2]
    assert second["state"]["terminal_safe_abort"] is False


def test_third_unique_request_enters_terminal_safe_abort_without_reservation():
    result = _request(_two_attempt_state(), index=3)
    assert result["status"] == "SAFE_ABORT"
    assert result["rejection_code"] == "RECOVERY_LIMIT_REACHED"
    assert result["state"]["terminal_safe_abort"] is True
    assert len(result["state"]["attempts_reserved"]) == 2


def test_duplicate_request_is_idempotent():
    first = _request(index=1)
    replay = _request(first["state"], index=1)
    assert replay["status"] == "RECOVERY_SLOT_ALREADY_RESERVED"
    assert replay["attempt_number"] == 1
    assert replay["state_changed"] is False
    assert replay["state"] == first["state"]


def test_same_request_id_with_changed_evidence_safe_aborts():
    first = _request(index=1)
    collision = _request(first["state"], index=1, d8_exit_event_id="changed")
    assert collision["status"] == "SAFE_ABORT"
    assert collision["rejection_code"] == "REQUEST_ID_COLLISION"


@pytest.mark.parametrize(
    "overrides",
    [
        {"d8_exit_latched": False},
        {"d8_failure_reason": ""},
        {"d9_plan_completed": False},
        {"d9_plan_stage_sequence": ["RETRACT_REQUEST"]},
        {"fresh_reobservation_authorized": False},
        {"fresh_reobservation_evidence_id": ""},
    ],
)
def test_missing_prerequisite_never_reserves_a_slot(overrides):
    result = _request(**overrides)
    assert result["status"] == "RECOVERY_REJECTED"
    assert result["state"]["attempts_reserved"] == []
    assert result["control_authorized"] is False


def test_exit_event_must_be_fresh_between_attempts():
    first = _request(index=1)
    stale = _request(first["state"], index=2, d8_exit_event_id="d8-exit-1")
    assert stale["rejection_code"] == "D8_EXIT_EVENT_NOT_FRESH"
    assert len(stale["state"]["attempts_reserved"]) == 1


def test_reobservation_evidence_must_be_fresh_between_attempts():
    first = _request(index=1)
    stale = _request(
        first["state"], index=2,
        fresh_reobservation_evidence_id="reobserve-1",
    )
    assert stale["rejection_code"] == "REOBSERVATION_EVIDENCE_NOT_FRESH"
    assert len(stale["state"]["attempts_reserved"]) == 1


def test_terminal_state_cannot_be_reopened():
    terminal = _request(_two_attempt_state(), index=3)["state"]
    result = _request(terminal, index=4)
    assert result["status"] == "SAFE_ABORT"
    assert result["rejection_code"] == "RECOVERY_ALREADY_TERMINAL"
    assert result["state_changed"] is False


def test_invalid_state_fails_closed():
    invalid = new_recovery_budget_state()
    invalid["attempts_reserved"] = [{}, {}, {}]
    result = _request(invalid, index=1)
    assert result["status"] == "SAFE_ABORT"
    assert result["rejection_code"] == "INVALID_BUDGET_STATE"
    assert result["state"]["terminal_safe_abort"] is True


def test_boolean_inputs_are_strict():
    result = _request(d8_exit_latched=1)
    assert result["rejection_code"] == "INVALID_RECOVERY_REQUEST"


def test_all_decisions_are_nonexecuting_and_nonclaiming():
    results = [_request(index=1), _request(_two_attempt_state(), index=3)]
    for result in results:
        assert result["motion_command_emitted"] is False
        assert result["capture_command_emitted"] is False
        assert result["control_authorized"] is False
        assert result["dynamic_recovery_pass_claimed"] is False
        assert result["hardware_authorized"] is False


def test_truth_firewall_is_absent_from_runtime_signature():
    names = set(inspect.signature(request_recovery_slot).parameters)
    assert names.isdisjoint(
        {"object_pose", "contact_name", "contact_normal", "event_truth", "collider_path"}
    )


def test_source_hash_drift_is_rejected(tmp_path):
    root = tmp_path / "repository"
    for relative in FROZEN_SOURCES:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((REPOSITORY_ROOT / relative).read_bytes())
    changed = root / next(iter(FROZEN_SOURCES))
    changed.write_bytes(changed.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_recovery_budget_contract(root)


def test_decision_is_strict_json():
    json.dumps(_request(index=1), sort_keys=True, allow_nan=False)
