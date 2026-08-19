from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from kcg_connector.d38999_multilayer_recovery_fsm import (
    PHASES,
    RecoveryEvidence,
    RecoveryWorkflowState,
    current_recovery_snapshot,
    initial_recovery_workflow_state,
    load_recovery_workflow_contract,
    step_recovery_workflow,
)


ROOT = Path(__file__).resolve().parents[3]
MODULE = (
    ROOT
    / "src/kcg_connector/kcg_connector/"
    "d38999_multilayer_recovery_fsm.py"
)
STAGES = (
    "LATCH_EXIT",
    "ZERO_TWIST_CONFIRMED",
    "RETRACT_COMPLETE",
    "SETTLE_COMPLETE",
    "REOBSERVATION_COMPLETE",
    "AUTHORIZE_REENTRY",
)


def _contract():
    return load_recovery_workflow_contract(ROOT)


def _evidence(stage, index=0, **overrides):
    values = {
        "stage": stage,
        "evidence_id": f"offline-{index:02d}-{stage}",
        "evidence_sha256": f"{index + 1:064x}"[-64:],
        "passed": True,
        "fresh": True,
        "evidence_level": "offline_fixture",
        "f1_safe_abort_latched": stage == "LATCH_EXIT",
        "failure_reason": "OVERFORCE" if stage == "LATCH_EXIT" else None,
        "explicit_reentry_authorization": stage == "AUTHORIZE_REENTRY",
        "controller_truth_used": False,
    }
    values.update(overrides)
    return RecoveryEvidence(**values)


def _drive_cycle(contract, state, cycle):
    base = cycle * 10
    results = []
    for offset, stage in enumerate(STAGES):
        state, result = step_recovery_workflow(
            contract, state, _evidence(stage, base + offset)
        )
        results.append(result)
    return state, results


def test_contract_binds_all_frozen_recovery_boundaries_and_current_f1():
    contract = _contract()
    assert len(contract.source_rows) == 8
    assert contract.phases == PHASES
    assert contract.stage_sequence == (
        "STOP_ZERO_TWIST",
        "RETRACT_REQUEST",
        "HOLD_FOR_SETTLE_REQUEST",
        "PRECOMMITTED_REOBSERVE_REQUEST",
    )
    assert contract.maximum_recovery_attempts == 2
    assert contract.backoff_distance_m == pytest.approx(0.0004)
    assert contract.backoff_speed_m_s == pytest.approx(0.0003)
    assert contract.settle_duration_s == pytest.approx(0.5)
    assert contract.candidate_view_sequence == ("V0", "V1", "V2")
    assert contract.current_f1_state == "HOME"
    assert contract.current_f1_dynamic_task_passed is False


def test_current_state_has_no_abort_to_recover_and_zero_commands():
    snapshot = current_recovery_snapshot(_contract())
    assert snapshot["phase"] == "IDLE"
    assert snapshot["rejection_code"] == "NO_F1_SAFE_ABORT_TO_RECOVER"
    assert snapshot["current_f1_state"] == "HOME"
    assert snapshot["attempts_reserved"] == 0
    assert snapshot["motion_command_emitted"] is False
    assert snapshot["capture_command_emitted"] is False
    assert snapshot["f1_safe_abort_reset_performed"] is False
    assert snapshot["control_authorized"] is False


def test_two_complete_offline_cycles_only_create_explicit_reentry_requests():
    contract = _contract()
    state = initial_recovery_workflow_state()
    state, first = _drive_cycle(contract, state, 0)
    assert state.phase == "REENTRY_REQUEST_READY"
    assert first[-1]["reentry_request_ready"] is True
    assert first[-1]["attempts_reserved"] == 1
    assert first[-1]["f1_safe_abort_reset_performed"] is False
    state, second = _drive_cycle(contract, state, 1)
    assert state.phase == "REENTRY_REQUEST_READY"
    assert second[-1]["reentry_request_ready"] is True
    assert second[-1]["attempts_reserved"] == 2
    assert all(result["motion_command_emitted"] is False for result in first + second)
    assert all(result["dynamic_recovery_pass_claimed"] is False for result in first + second)


def test_third_failure_immediately_latches_terminal_safe_abort():
    contract = _contract()
    state = initial_recovery_workflow_state()
    state, _ = _drive_cycle(contract, state, 0)
    state, _ = _drive_cycle(contract, state, 1)
    state, result = step_recovery_workflow(
        contract, state, _evidence("LATCH_EXIT", 99)
    )
    assert state.phase == "TERMINAL_SAFE_ABORT"
    assert state.terminal_reason == "RECOVERY_LIMIT_REACHED"
    assert result["reentry_request_ready"] is False
    assert result["f1_safe_abort_reset_performed"] is False


def test_caller_cannot_skip_stop_retract_settle_or_reobservation():
    contract = _contract()
    state, result = step_recovery_workflow(
        contract,
        initial_recovery_workflow_state(),
        _evidence("REOBSERVATION_COMPLETE"),
    )
    assert state.phase == "IDLE"
    assert result["rejection_code"] == "EXPECTED_STAGE:LATCH_EXIT"


@pytest.mark.parametrize(
    "overrides,code",
    [
        ({"evidence_id": ""}, "EVIDENCE_ID_MISSING"),
        ({"evidence_sha256": "bad"}, "EVIDENCE_SHA256_INVALID"),
        ({"fresh": False}, "EVIDENCE_NOT_FRESH"),
        ({"passed": False}, "STAGE_NOT_PASSED:LATCH_EXIT"),
        ({"evidence_level": "self_report"}, "EVIDENCE_LEVEL_INVALID"),
        ({"f1_safe_abort_latched": False}, "F1_SAFE_ABORT_EVIDENCE_REQUIRED"),
        ({"failure_reason": ""}, "F1_SAFE_ABORT_EVIDENCE_REQUIRED"),
    ],
)
def test_each_missing_latch_property_holds_idle(overrides, code):
    state, result = step_recovery_workflow(
        _contract(),
        initial_recovery_workflow_state(),
        _evidence("LATCH_EXIT", **overrides),
    )
    assert state == initial_recovery_workflow_state()
    assert result["rejection_code"] == code


def test_replayed_evidence_is_rejected():
    contract = _contract()
    first = _evidence("LATCH_EXIT")
    state, _ = step_recovery_workflow(
        contract, initial_recovery_workflow_state(), first
    )
    replay = _evidence(
        "ZERO_TWIST_CONFIRMED",
        evidence_id=first.evidence_id,
        evidence_sha256=first.evidence_sha256,
    )
    held, result = step_recovery_workflow(contract, state, replay)
    assert held == state
    assert result["rejection_code"] == "EVIDENCE_REPLAY_REJECTED"


def test_explicit_reentry_authorization_cannot_be_omitted():
    contract = _contract()
    state = initial_recovery_workflow_state()
    for index, stage in enumerate(STAGES[:-1]):
        state, _ = step_recovery_workflow(
            contract, state, _evidence(stage, index)
        )
    held, result = step_recovery_workflow(
        contract,
        state,
        _evidence(
            "AUTHORIZE_REENTRY",
            9,
            explicit_reentry_authorization=False,
        ),
    )
    assert held == state
    assert result["rejection_code"] == "EXPLICIT_REENTRY_AUTHORIZATION_REQUIRED"
    assert result["f1_safe_abort_reset_performed"] is False


def test_truth_firewall_violation_latches_terminal_abort():
    state, result = step_recovery_workflow(
        _contract(),
        initial_recovery_workflow_state(),
        _evidence("LATCH_EXIT", controller_truth_used=True),
    )
    assert state.phase == "TERMINAL_SAFE_ABORT"
    assert state.terminal_reason == "CONTROLLER_TRUTH_FIREWALL_VIOLATION"
    assert result["control_authorized"] is False


def test_terminal_safe_abort_cannot_auto_recover():
    contract = _contract()
    terminal = RecoveryWorkflowState(
        phase="TERMINAL_SAFE_ABORT",
        budget_state=initial_recovery_workflow_state().budget_state,
        consumed_evidence_ids=(),
        pending_failure_reason=None,
        exit_evidence_id=None,
        reobservation_evidence_id=None,
        step_count=1,
        terminal_reason="TEST",
        offline_fixture_path_only=True,
    )
    held, result = step_recovery_workflow(
        contract, terminal, _evidence("LATCH_EXIT")
    )
    assert held == terminal
    assert result["rejection_code"] == "RECOVERY_TERMINAL_SAFE_ABORT_LATCHED"


@pytest.mark.parametrize(
    "state",
    [
        RecoveryWorkflowState("UNKNOWN", {}, (), None, None, None, 0, None, False),
        RecoveryWorkflowState("IDLE", {}, (), None, None, None, 0, None, False),
        RecoveryWorkflowState("IDLE", initial_recovery_workflow_state().budget_state, (), None, None, None, -1, None, False),
        RecoveryWorkflowState("TERMINAL_SAFE_ABORT", initial_recovery_workflow_state().budget_state, (), None, None, None, 0, None, False),
    ],
)
def test_malformed_state_is_rejected(state):
    with pytest.raises(ValueError):
        step_recovery_workflow(_contract(), state, _evidence("LATCH_EXIT"))


def test_public_step_inputs_exclude_target_phase_and_privileged_truth():
    names = set(inspect.signature(step_recovery_workflow).parameters)
    assert names == {"contract", "state", "evidence"}
    fields = set(RecoveryEvidence.__dataclass_fields__)
    assert "target_phase" not in fields
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
