from __future__ import annotations

from dataclasses import fields, replace
import inspect
import json
from pathlib import Path

import pytest

from kcg_connector.compliant_insertion import (
    ControllerState,
    InsertionObservation,
    InsertionState,
    load_compliant_insertion_config,
)
from kcg_connector.d38999_multilayer_compliant_insertion_gate import (
    CompliantInsertionReadiness,
    FROZEN_SOURCES,
    GATE_ORDER,
    build_compliant_insertion_gate_contract,
    evaluate_compliant_insertion_gate,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _ready(value=True):
    return CompliantInsertionReadiness(**{item.name: value for item in fields(CompliantInsertionReadiness)})


def _observation(**overrides):
    values = {
        "timestamp_s": 0.0,
        "sample_age_s": 0.0,
        "wrench_assembly": (0.0,) * 6,
        "tcp_position_assembly_m": (0.0, 0.0, 0.0),
        "tcp_rotation_vector_assembly_rad": (0.0, 0.0, 0.0),
        "vision_control_authorized": True,
        "synchronized_capture": True,
        "ft_valid": True,
        "ft_tared": True,
        "payload_compensated": True,
    }
    values.update(overrides)
    return InsertionObservation(**values)


def _evaluate(*, readiness=None, state=None, observation=None):
    return evaluate_compliant_insertion_gate(
        _ready() if readiness is None else readiness,
        load_compliant_insertion_config(),
        ControllerState() if state is None else state,
        _observation() if observation is None else observation,
    )


def _copy_sources(tmp_path):
    root = tmp_path / "repository"
    for relative in FROZEN_SOURCES:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((REPOSITORY_ROOT / relative).read_bytes())
    return root


def test_contract_preserves_existing_limits_and_current_rejection():
    contract = build_compliant_insertion_gate_contract(REPOSITORY_ROOT)
    assert contract["control_rate_hz"] == 240
    assert contract["axial_speed_m_s"] == pytest.approx(0.00020)
    assert contract["maximum_total_travel_m"] == pytest.approx(0.024)
    assert contract["maximum_retries"] == 2
    assert contract["formal_force_component_limit_n_per_driven_body"] == 8.0
    assert contract["formal_moment_component_limit_nm"] == 0.30
    assert contract["current_readiness"]["rejection_code"] == (
        "A2_NOMINAL_INSERTION_NOT_DYNAMIC"
    )
    assert contract["current_readiness"]["twist_candidate_task"] == [0.0] * 6


@pytest.mark.parametrize(("field_name", "code"), GATE_ORDER)
def test_each_upstream_gate_independently_fails_closed(field_name, code):
    readiness = replace(_ready(), **{field_name: False})
    result = _evaluate(readiness=readiness)
    assert result["rejection_code"] == code
    assert result["twist_candidate_task"] == [0.0] * 6
    assert result["state_machine_evaluated"] is False


def test_invalid_readiness_types_fail_closed():
    invalid = replace(_ready(), d3_light_contact_dynamic_pass=1)
    result = _evaluate(readiness=invalid)
    assert result["rejection_code"] == "INVALID_READINESS_SNAPSHOT"
    assert result["twist_candidate_task"] == [0.0] * 6


def test_missing_state_machine_inputs_fail_closed_after_all_gates():
    result = evaluate_compliant_insertion_gate(_ready())
    assert result["rejection_code"] == "STATE_MACHINE_INPUT_MISSING"
    assert result["twist_candidate_task"] == [0.0] * 6


def test_allowed_fixture_reaches_existing_guarded_approach_candidate():
    result = _evaluate()
    assert result["status"] == "OFFLINE_STATE_MACHINE_STEP_CANDIDATE"
    assert result["next_state"] == InsertionState.GUARDED_APPROACH.value
    assert result["twist_candidate_task"][:2] == [0.0, 0.0]
    assert 0.0 < result["twist_candidate_task"][2] <= 0.00020
    assert result["twist_candidate_task"][3:] == [0.0, 0.0, 0.0]
    assert result["motion_command_emitted"] is False
    assert result["control_authorized"] is False


def test_existing_acceleration_slew_caps_first_axial_step():
    result = _evaluate()
    assert result["twist_candidate_task"][2] == pytest.approx(0.003 / 240.0)


def test_underlying_vision_gate_still_reobserves_with_zero_motion():
    result = _evaluate(observation=_observation(vision_control_authorized=False))
    assert result["next_state"] == InsertionState.REOBSERVE.value
    assert result["request_reobserve"] is True
    assert result["twist_candidate_task"] == [0.0] * 6


@pytest.mark.parametrize(
    "observation",
    (
        _observation(wrench_assembly=(0.0, 0.0, 50.0, 0.0, 0.0, 0.0)),
        _observation(wrench_assembly=(20.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
        _observation(wrench_assembly=(0.0, 0.0, 0.0, 2.0, 0.0, 0.0)),
        _observation(wrench_assembly=(0.0, 0.0, 0.0, 0.0, 0.0, 0.6)),
    ),
)
def test_existing_hard_safety_envelope_yields_safe_abort(observation):
    result = _evaluate(observation=observation)
    assert result["status"] == "OFFLINE_SAFE_ABORT_CANDIDATE"
    assert result["next_state"] == InsertionState.SAFE_ABORT.value
    assert result["rejection_code"] == "HARD_SAFETY_GATE"
    assert result["twist_candidate_task"] == [0.0] * 6


@pytest.mark.parametrize(
    "observation",
    (
        _observation(sample_age_s=0.020001),
        _observation(ft_valid=False),
        _observation(ft_tared=False),
        _observation(payload_compensated=False),
    ),
)
def test_stale_or_unqualified_wrench_yields_safe_abort(observation):
    result = _evaluate(observation=observation)
    assert result["status"] == "OFFLINE_SAFE_ABORT_CANDIDATE"
    assert result["next_state"] == InsertionState.SAFE_ABORT.value
    assert result["twist_candidate_task"] == [0.0] * 6


def test_truth_firewall_is_absent_from_both_adapter_schemas():
    readiness_fields = {item.name for item in fields(CompliantInsertionReadiness)}
    observation_fields = {item.name for item in fields(InsertionObservation)}
    forbidden = {
        "object_truth",
        "object_pose",
        "contact_name",
        "contact_normal",
        "collider_identity",
        "event_truth",
        "penetration_depth_truth",
    }
    assert readiness_fields.isdisjoint(forbidden)
    assert observation_fields.isdisjoint(forbidden)
    assert set(inspect.signature(evaluate_compliant_insertion_gate).parameters).isdisjoint(forbidden)


def test_frozen_source_drift_is_rejected(tmp_path):
    root = _copy_sources(tmp_path)
    target = root / next(iter(FROZEN_SOURCES))
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        build_compliant_insertion_gate_contract(root)


def test_contract_and_candidate_are_strict_json_without_dynamic_claim():
    contract = build_compliant_insertion_gate_contract(REPOSITORY_ROOT)
    result = _evaluate()
    json.dumps(contract, sort_keys=True, allow_nan=False)
    json.dumps(result, sort_keys=True, allow_nan=False)
    assert contract["simulation_started"] is False
    assert contract["dynamic_compliant_insertion_pass_claimed"] is False
    assert contract["control_authorized"] is False
    assert result["motion_command_emitted"] is False
    assert result["control_authorized"] is False
    assert result["dynamic_compliant_insertion_pass_claimed"] is False
