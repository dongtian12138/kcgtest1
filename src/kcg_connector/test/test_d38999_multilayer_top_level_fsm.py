from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from kcg_connector.d38999_multilayer_top_level_fsm import (
    EXPECTED_STATES,
    NORMAL_TRANSITION_GATES,
    StepEvidence,
    WorkflowState,
    current_workflow_snapshot,
    initial_workflow_state,
    load_top_level_workflow_contract,
    step_top_level_workflow,
)


ROOT = Path(__file__).resolve().parents[3]
MODULE = (
    ROOT
    / "src/kcg_connector/kcg_connector/"
    "d38999_multilayer_top_level_fsm.py"
)


def _contract():
    return load_top_level_workflow_contract(ROOT)


def _evidence(gate, index=0, **overrides):
    values = {
        "gate_name": gate,
        "evidence_id": f"offline-{index:02d}-{gate}",
        "evidence_sha256": f"{index + 1:064x}"[-64:],
        "passed": True,
        "fresh": True,
        "evidence_level": "offline_fixture",
        "safety_abort_requested": False,
        "safety_reason": None,
        "controller_truth_used": False,
    }
    values.update(overrides)
    return StepEvidence(**values)


def test_contract_binds_frozen_states_safety_and_current_e8_boundary():
    contract = _contract()
    assert len(contract.source_rows) == 3
    assert contract.states == EXPECTED_STATES
    assert len(contract.states) == 21
    assert len(contract.normal_states) == 20
    assert len(contract.transition_rows) == 19
    assert contract.normal_transition_gates == NORMAL_TRANSITION_GATES
    assert contract.safe_abort_state == "SAFE_ABORT"
    assert contract.safe_abort_latched is True
    assert contract.maximum_recovery_attempts == 2
    assert contract.moment_component_limit_nm == pytest.approx(0.30)
    assert contract.current_e8_dynamic_assembly_passed is False


def test_every_normal_edge_is_unique_sequential_and_has_one_unique_gate():
    contract = _contract()
    assert len(set(contract.normal_transition_gates)) == 19
    assert len(set((row[0], row[1]) for row in contract.transition_rows)) == 19
    for index, (source, target, gate) in enumerate(contract.transition_rows):
        assert source == contract.normal_states[index]
        assert target == contract.normal_states[index + 1]
        assert gate == contract.normal_transition_gates[index]


def test_current_disk_evidence_holds_at_home_with_zero_commands():
    snapshot = current_workflow_snapshot(_contract())
    assert snapshot["state"] == "HOME"
    assert snapshot["state_advanced"] is False
    assert snapshot["rejection_code"] == "MISSION_START_NOT_AUTHORIZED"
    assert snapshot["next_expected_gate"] == "mission_start_authorized"
    assert snapshot["current_e8_dynamic_assembly_passed"] is False
    assert snapshot["robot_commands_emitted"] == 0
    assert snapshot["dynamic_task_pass_claimed"] is False
    assert snapshot["control_authorized"] is False


def test_offline_fixture_can_exercise_exact_path_without_dynamic_claim():
    contract = _contract()
    state = initial_workflow_state()
    for index, gate in enumerate(contract.normal_transition_gates):
        state, result = step_top_level_workflow(
            contract, state, _evidence(gate, index)
        )
        assert result["state_advanced"] is True
        assert result["dynamic_transition_claimed"] is False
        assert result["robot_commands_emitted"] == 0
        assert result["control_authorized"] is False
    assert state.state == "DONE"
    assert state.visited_states == contract.normal_states
    assert state.step_count == 19
    assert state.offline_fixture_path_only is True


def test_caller_cannot_jump_state_by_submitting_a_later_gate():
    contract = _contract()
    state, result = step_top_level_workflow(
        contract,
        initial_workflow_state(),
        _evidence("assembly_arrival_dynamic_pass"),
    )
    assert state.state == "HOME"
    assert result["state_advanced"] is False
    assert result["rejection_code"] == "EXPECTED_GATE:mission_start_authorized"


@pytest.mark.parametrize(
    "overrides,code",
    [
        ({"evidence_id": ""}, "EVIDENCE_ID_MISSING"),
        ({"evidence_sha256": "bad"}, "EVIDENCE_SHA256_INVALID"),
        ({"fresh": False}, "EVIDENCE_NOT_FRESH"),
        ({"passed": False}, "GATE_NOT_PASSED:mission_start_authorized"),
        ({"evidence_level": "self_report"}, "EVIDENCE_LEVEL_INVALID"),
    ],
)
def test_each_missing_evidence_property_holds_current_state(overrides, code):
    contract = _contract()
    state, result = step_top_level_workflow(
        contract,
        initial_workflow_state(),
        _evidence("mission_start_authorized", **overrides),
    )
    assert state == initial_workflow_state()
    assert result["state_advanced"] is False
    assert result["rejection_code"] == code


def test_same_evidence_id_cannot_be_replayed_on_next_edge():
    contract = _contract()
    evidence = _evidence("mission_start_authorized")
    state, _ = step_top_level_workflow(
        contract, initial_workflow_state(), evidence
    )
    replay = _evidence(
        "plug_localized",
        evidence_id=evidence.evidence_id,
        evidence_sha256=evidence.evidence_sha256,
    )
    held, result = step_top_level_workflow(contract, state, replay)
    assert held == state
    assert result["rejection_code"] == "EVIDENCE_REPLAY_REJECTED"


@pytest.mark.parametrize("state_name", EXPECTED_STATES[:-2])
def test_safety_abort_is_available_from_every_nonterminal_normal_state(state_name):
    contract = _contract()
    index = contract.normal_states.index(state_name)
    state = WorkflowState(
        state=state_name,
        visited_states=contract.normal_states[: index + 1],
        consumed_evidence_ids=tuple(f"evidence-{i}" for i in range(index)),
        step_count=index,
        abort_reason=None,
        offline_fixture_path_only=True,
    )
    aborted, result = step_top_level_workflow(
        contract,
        state,
        _evidence(
            contract.normal_transition_gates[index],
            safety_abort_requested=True,
            safety_reason="WRIST_MOMENT_LIMIT",
        ),
    )
    assert aborted.state == "SAFE_ABORT"
    assert aborted.abort_reason == "WRIST_MOMENT_LIMIT"
    assert result["rejection_code"] == "SAFETY_ABORT_LATCHED"
    assert result["control_authorized"] is False


def test_truth_firewall_violation_forces_safe_abort():
    contract = _contract()
    aborted, result = step_top_level_workflow(
        contract,
        initial_workflow_state(),
        _evidence("mission_start_authorized", controller_truth_used=True),
    )
    assert aborted.state == "SAFE_ABORT"
    assert aborted.abort_reason == "CONTROLLER_TRUTH_FIREWALL_VIOLATION"
    assert result["dynamic_transition_claimed"] is False


def test_safe_abort_is_latched_and_cannot_auto_recover():
    contract = _contract()
    aborted, _ = step_top_level_workflow(
        contract,
        initial_workflow_state(),
        _evidence(
            "mission_start_authorized",
            safety_abort_requested=True,
            safety_reason="OVERFORCE",
        ),
    )
    held, result = step_top_level_workflow(
        contract, aborted, _evidence("mission_start_authorized", 1)
    )
    assert held == aborted
    assert result["rejection_code"] == "SAFE_ABORT_LATCHED_EXPLICIT_RECOVERY_REQUIRED"


def test_done_is_terminal_and_never_restarts():
    contract = _contract()
    done = WorkflowState(
        state="DONE",
        visited_states=contract.normal_states,
        consumed_evidence_ids=tuple(f"e-{i}" for i in range(19)),
        step_count=19,
        abort_reason=None,
        offline_fixture_path_only=True,
    )
    held, result = step_top_level_workflow(
        contract, done, _evidence("mission_start_authorized")
    )
    assert held == done
    assert result["rejection_code"] == "DONE_TERMINAL"


@pytest.mark.parametrize(
    "state",
    [
        WorkflowState("UNKNOWN", ("UNKNOWN",), (), 0, None, False),
        WorkflowState("HOME", (), (), 0, None, False),
        WorkflowState("HOME", ("HOME",), (), -1, None, False),
        WorkflowState("HOME", ("LOCATE_PLUG",), (), 0, None, False),
    ],
)
def test_malformed_workflow_state_is_rejected(state):
    with pytest.raises(ValueError):
        step_top_level_workflow(
            _contract(), state, _evidence("mission_start_authorized")
        )


def test_safety_abort_without_reason_is_rejected():
    with pytest.raises(ValueError):
        step_top_level_workflow(
            _contract(),
            initial_workflow_state(),
            _evidence("mission_start_authorized", safety_abort_requested=True),
        )


def test_public_step_inputs_exclude_target_state_and_privileged_truth():
    names = set(inspect.signature(step_top_level_workflow).parameters)
    assert names == {"contract", "state", "evidence"}
    fields = set(StepEvidence.__dataclass_fields__)
    assert "target_state" not in fields
    assert fields.isdisjoint(
        {"object_pose", "contact_name", "contact_normal", "event_truth"}
    )


def test_module_is_cpu_only_and_has_no_pose_or_command_api():
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    assert roots.isdisjoint({"isaacsim", "omni", "pxr", "rclpy", "torch"})
    source = MODULE.read_text(encoding="utf-8")
    assert "set_world_pose" not in source
    assert "set_local_pose" not in source
    assert "apply_force" not in source
    assert "apply_torque" not in source
