"""Explicit, bounded recovery workflow layered over D8, D9, D10, and F1.

This state machine only sequences evidence.  It does not execute the backoff,
capture a view, reset the F1 SAFE_ABORT latch, or emit a robot command.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from .d38999_multilayer_recovery_budget import (
    EXPECTED_D9_STAGE_SEQUENCE,
    load_recovery_budget_contract,
    new_recovery_budget_state,
    request_recovery_slot,
)


TASK_ID = "EIGHT-HOUR-F2-RECOVERY-STATE-MACHINE"
F1_RESULT_PATH = (
    "artifacts/agent_control/tasks/"
    "EIGHT-HOUR-F1-TOP-LEVEL-STATE-MACHINE/TASK_RESULT.json"
)
D8_RESULT_PATH = (
    "artifacts/agent_control/tasks/"
    "EIGHT-HOUR-D8-OVERFORCE-EXIT/TASK_RESULT.json"
)
D9_RESULT_PATH = (
    "artifacts/agent_control/tasks/"
    "EIGHT-HOUR-D9-BACKOFF-REOBSERVE/TASK_RESULT.json"
)
D10_RESULT_PATH = (
    "artifacts/agent_control/tasks/"
    "EIGHT-HOUR-D10-TWO-RECOVERY-LIMIT/TASK_RESULT.json"
)

PHASES = (
    "IDLE",
    "ZERO_TWIST_REQUIRED",
    "RETRACT_EVIDENCE_REQUIRED",
    "SETTLE_EVIDENCE_REQUIRED",
    "REOBSERVATION_EVIDENCE_REQUIRED",
    "EXPLICIT_REENTRY_AUTHORIZATION_REQUIRED",
    "REENTRY_REQUEST_READY",
    "TERMINAL_SAFE_ABORT",
)

EXPECTED_EVIDENCE_STAGE = {
    "IDLE": "LATCH_EXIT",
    "ZERO_TWIST_REQUIRED": "ZERO_TWIST_CONFIRMED",
    "RETRACT_EVIDENCE_REQUIRED": "RETRACT_COMPLETE",
    "SETTLE_EVIDENCE_REQUIRED": "SETTLE_COMPLETE",
    "REOBSERVATION_EVIDENCE_REQUIRED": "REOBSERVATION_COMPLETE",
    "EXPLICIT_REENTRY_AUTHORIZATION_REQUIRED": "AUTHORIZE_REENTRY",
    "REENTRY_REQUEST_READY": "LATCH_EXIT",
}

FROZEN_SOURCES = {
    "src/kcg_connector/kcg_connector/d38999_multilayer_overforce_exit.py": (
        "05df5ce57ab3a72b66cf94688cc37d603829879fa32853f33b47c90010dc1946"
    ),
    "src/kcg_connector/kcg_connector/d38999_multilayer_backoff_reobserve.py": (
        "68b92f7f4e04ca24a3ab71a5a0f37d1d395b3fa89918ce2223b9ecd9d4c97b5f"
    ),
    "src/kcg_connector/kcg_connector/d38999_multilayer_recovery_budget.py": (
        "ac5841be89f3ef541ec52eed06bf1fea84e01b2d1cc6c1970ae2d78caa0adb45"
    ),
    "src/kcg_connector/kcg_connector/d38999_multilayer_top_level_fsm.py": (
        "473258bfb5a634636559ea74619b7558cfee939ad1654fa983a16ac06832c9c9"
    ),
    D8_RESULT_PATH: (
        "fa07a1ca8842b2799fb26ef91446ae64347d168c477bebffb76147e6524c81f3"
    ),
    D9_RESULT_PATH: (
        "97c8452ddcba39936377416d65d4f9688c538744bb55f0492aca929fb36b4df8"
    ),
    D10_RESULT_PATH: (
        "6ba295872ebd695dc3894ad210085c344061b65435d3e8f5070eaf707f38f32c"
    ),
    F1_RESULT_PATH: (
        "040917a1d1e30f092f7b3b5b33d5c1991f7d9d4c28a0902c355c1582c7984bb7"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


@dataclass(frozen=True)
class RecoveryWorkflowContract:
    source_rows: tuple[tuple[str, str], ...]
    phases: tuple[str, ...]
    stage_sequence: tuple[str, ...]
    maximum_recovery_attempts: int
    backoff_distance_m: float
    backoff_speed_m_s: float
    settle_duration_s: float
    candidate_view_sequence: tuple[str, ...]
    current_f1_state: str
    current_f1_dynamic_task_passed: bool


@dataclass(frozen=True)
class RecoveryWorkflowState:
    phase: str
    budget_state: Mapping[str, Any]
    consumed_evidence_ids: tuple[str, ...]
    pending_failure_reason: str | None
    exit_evidence_id: str | None
    reobservation_evidence_id: str | None
    step_count: int
    terminal_reason: str | None
    offline_fixture_path_only: bool


@dataclass(frozen=True)
class RecoveryEvidence:
    stage: str
    evidence_id: str
    evidence_sha256: str
    passed: bool
    fresh: bool
    evidence_level: str
    f1_safe_abort_latched: bool
    failure_reason: str | None
    explicit_reentry_authorization: bool
    controller_truth_used: bool


def load_recovery_workflow_contract(
    repository_root: str | Path,
) -> RecoveryWorkflowContract:
    root = Path(repository_root).resolve()
    rows: list[tuple[str, str]] = []
    for relative, expected in FROZEN_SOURCES.items():
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError(f"frozen F2 source missing: {relative}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"frozen F2 source hash mismatch: {relative}")
        rows.append((relative, actual))

    budget = load_recovery_budget_contract(root)
    d8 = _mapping(
        json.loads((root / D8_RESULT_PATH).read_text(encoding="utf-8")),
        "D8 task result",
    )
    d9 = _mapping(
        json.loads((root / D9_RESULT_PATH).read_text(encoding="utf-8")),
        "D9 task result",
    )
    d10 = _mapping(
        json.loads((root / D10_RESULT_PATH).read_text(encoding="utf-8")),
        "D10 task result",
    )
    f1 = _mapping(
        json.loads((root / F1_RESULT_PATH).read_text(encoding="utf-8")),
        "F1 task result",
    )
    if (
        d8.get("outcome") != "OFFLINE_PASS"
        or d9.get("outcome") != "OFFLINE_PASS"
        or d10.get("outcome") != "OFFLINE_PASS"
        or f1.get("outcome") != "OFFLINE_PASS"
        or d8.get("backoff_distance_m") != budget["backoff_distance_m"]
        or d8.get("backoff_speed_m_s") != budget["backoff_speed_m_s"]
        or d9.get("stage_sequence") != list(EXPECTED_D9_STAGE_SEQUENCE)
        or d10.get("maximum_recovery_attempts")
        != budget["maximum_recovery_attempts"]
        or f1.get("safe_abort_latched") is not True
        or f1.get("current_state") != "HOME"
        or type(f1.get("dynamic_task_pass_claimed")) is not bool
    ):
        raise ValueError("F2 upstream recovery boundary changed")
    return RecoveryWorkflowContract(
        source_rows=tuple(rows),
        phases=PHASES,
        stage_sequence=tuple(EXPECTED_D9_STAGE_SEQUENCE),
        maximum_recovery_attempts=int(budget["maximum_recovery_attempts"]),
        backoff_distance_m=float(budget["backoff_distance_m"]),
        backoff_speed_m_s=float(budget["backoff_speed_m_s"]),
        settle_duration_s=float(d9["settle_duration_s"]),
        candidate_view_sequence=tuple(d9["candidate_view_sequence"]),
        current_f1_state=str(f1.get("current_state")),
        current_f1_dynamic_task_passed=f1["dynamic_task_pass_claimed"],
    )


def initial_recovery_workflow_state() -> RecoveryWorkflowState:
    return RecoveryWorkflowState(
        phase="IDLE",
        budget_state=new_recovery_budget_state(),
        consumed_evidence_ids=(),
        pending_failure_reason=None,
        exit_evidence_id=None,
        reobservation_evidence_id=None,
        step_count=0,
        terminal_reason=None,
        offline_fixture_path_only=False,
    )


def _validate_state(
    contract: RecoveryWorkflowContract,
    state: RecoveryWorkflowState,
) -> None:
    if state.phase not in contract.phases:
        raise ValueError("unknown recovery workflow phase")
    if isinstance(state.step_count, bool) or not isinstance(state.step_count, int):
        raise ValueError("recovery step count must be an integer")
    if state.step_count < 0 or not isinstance(state.budget_state, Mapping):
        raise ValueError("invalid recovery workflow state")
    attempts = state.budget_state.get("attempts_reserved")
    if not isinstance(attempts, list) or len(attempts) > contract.maximum_recovery_attempts:
        raise ValueError("invalid recovery budget inside F2")
    if state.phase == "TERMINAL_SAFE_ABORT" and not state.terminal_reason:
        raise ValueError("terminal recovery state requires a reason")


def _result(
    state: RecoveryWorkflowState,
    *,
    state_changed: bool,
    rejection_code: str | None,
    reentry_request_ready: bool,
) -> dict[str, Any]:
    attempts = state.budget_state.get("attempts_reserved", [])
    return {
        "schema_version": 1,
        "task_id": TASK_ID,
        "phase": state.phase,
        "step_count": state.step_count,
        "attempts_reserved": len(attempts),
        "maximum_recovery_attempts": 2,
        "terminal_reason": state.terminal_reason,
        "state_changed": state_changed,
        "rejection_code": rejection_code,
        "reentry_request_ready": reentry_request_ready,
        "explicit_external_handoff_required": reentry_request_ready,
        "f1_safe_abort_reset_performed": False,
        "offline_fixture_path_only": state.offline_fixture_path_only,
        "object_or_contact_truth_used": False,
        "zero_twist_command_emitted": False,
        "motion_command_emitted": False,
        "capture_command_emitted": False,
        "dynamic_recovery_pass_claimed": False,
        "control_authorized": False,
    }


def step_recovery_workflow(
    contract: RecoveryWorkflowContract,
    state: RecoveryWorkflowState,
    evidence: RecoveryEvidence,
) -> tuple[RecoveryWorkflowState, dict[str, Any]]:
    """Consume one ordered evidence item without executing recovery motion."""

    _validate_state(contract, state)
    if state.phase == "TERMINAL_SAFE_ABORT":
        return state, _result(
            state,
            state_changed=False,
            rejection_code="RECOVERY_TERMINAL_SAFE_ABORT_LATCHED",
            reentry_request_ready=False,
        )
    if evidence.controller_truth_used is not False:
        terminal = replace(
            state,
            phase="TERMINAL_SAFE_ABORT",
            step_count=state.step_count + 1,
            terminal_reason="CONTROLLER_TRUTH_FIREWALL_VIOLATION",
        )
        return terminal, _result(
            terminal,
            state_changed=True,
            rejection_code="CONTROLLER_TRUTH_FIREWALL_VIOLATION",
            reentry_request_ready=False,
        )
    expected_stage = EXPECTED_EVIDENCE_STAGE[state.phase]
    if evidence.stage != expected_stage:
        return state, _result(
            state,
            state_changed=False,
            rejection_code=f"EXPECTED_STAGE:{expected_stage}",
            reentry_request_ready=False,
        )
    if not evidence.evidence_id:
        return state, _result(
            state,
            state_changed=False,
            rejection_code="EVIDENCE_ID_MISSING",
            reentry_request_ready=False,
        )
    if (
        not isinstance(evidence.evidence_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", evidence.evidence_sha256) is None
    ):
        return state, _result(
            state,
            state_changed=False,
            rejection_code="EVIDENCE_SHA256_INVALID",
            reentry_request_ready=False,
        )
    if evidence.evidence_id in state.consumed_evidence_ids:
        return state, _result(
            state,
            state_changed=False,
            rejection_code="EVIDENCE_REPLAY_REJECTED",
            reentry_request_ready=False,
        )
    if evidence.fresh is not True:
        return state, _result(
            state,
            state_changed=False,
            rejection_code="EVIDENCE_NOT_FRESH",
            reentry_request_ready=False,
        )
    if evidence.passed is not True:
        return state, _result(
            state,
            state_changed=False,
            rejection_code=f"STAGE_NOT_PASSED:{expected_stage}",
            reentry_request_ready=False,
        )
    if evidence.evidence_level not in {
        "offline_fixture",
        "dynamic_signed_artifact",
    }:
        return state, _result(
            state,
            state_changed=False,
            rejection_code="EVIDENCE_LEVEL_INVALID",
            reentry_request_ready=False,
        )
    offline = evidence.evidence_level == "offline_fixture"
    consumed = state.consumed_evidence_ids + (evidence.evidence_id,)

    if evidence.stage == "LATCH_EXIT":
        attempts = state.budget_state.get("attempts_reserved", [])
        if len(attempts) >= contract.maximum_recovery_attempts:
            terminal = replace(
                state,
                phase="TERMINAL_SAFE_ABORT",
                consumed_evidence_ids=consumed,
                step_count=state.step_count + 1,
                terminal_reason="RECOVERY_LIMIT_REACHED",
                offline_fixture_path_only=(
                    state.offline_fixture_path_only or offline
                ),
            )
            return terminal, _result(
                terminal,
                state_changed=True,
                rejection_code="RECOVERY_LIMIT_REACHED",
                reentry_request_ready=False,
            )
        if evidence.f1_safe_abort_latched is not True or not evidence.failure_reason:
            return state, _result(
                state,
                state_changed=False,
                rejection_code="F1_SAFE_ABORT_EVIDENCE_REQUIRED",
                reentry_request_ready=False,
            )
        advanced = replace(
            state,
            phase="ZERO_TWIST_REQUIRED",
            consumed_evidence_ids=consumed,
            pending_failure_reason=evidence.failure_reason,
            exit_evidence_id=evidence.evidence_id,
            reobservation_evidence_id=None,
            step_count=state.step_count + 1,
            offline_fixture_path_only=state.offline_fixture_path_only or offline,
        )
    elif evidence.stage == "ZERO_TWIST_CONFIRMED":
        advanced = replace(
            state,
            phase="RETRACT_EVIDENCE_REQUIRED",
            consumed_evidence_ids=consumed,
            step_count=state.step_count + 1,
            offline_fixture_path_only=state.offline_fixture_path_only or offline,
        )
    elif evidence.stage == "RETRACT_COMPLETE":
        advanced = replace(
            state,
            phase="SETTLE_EVIDENCE_REQUIRED",
            consumed_evidence_ids=consumed,
            step_count=state.step_count + 1,
            offline_fixture_path_only=state.offline_fixture_path_only or offline,
        )
    elif evidence.stage == "SETTLE_COMPLETE":
        advanced = replace(
            state,
            phase="REOBSERVATION_EVIDENCE_REQUIRED",
            consumed_evidence_ids=consumed,
            step_count=state.step_count + 1,
            offline_fixture_path_only=state.offline_fixture_path_only or offline,
        )
    elif evidence.stage == "REOBSERVATION_COMPLETE":
        advanced = replace(
            state,
            phase="EXPLICIT_REENTRY_AUTHORIZATION_REQUIRED",
            consumed_evidence_ids=consumed,
            reobservation_evidence_id=evidence.evidence_id,
            step_count=state.step_count + 1,
            offline_fixture_path_only=state.offline_fixture_path_only or offline,
        )
    else:
        if evidence.explicit_reentry_authorization is not True:
            return state, _result(
                state,
                state_changed=False,
                rejection_code="EXPLICIT_REENTRY_AUTHORIZATION_REQUIRED",
                reentry_request_ready=False,
            )
        if (
            not state.pending_failure_reason
            or not state.exit_evidence_id
            or not state.reobservation_evidence_id
        ):
            raise ValueError("recovery evidence chain is incomplete")
        budget_decision = request_recovery_slot(
            state.budget_state,
            request_id=evidence.evidence_id,
            d8_exit_latched=True,
            d8_failure_reason=state.pending_failure_reason,
            d8_exit_event_id=state.exit_evidence_id,
            d9_plan_completed=True,
            d9_plan_stage_sequence=list(contract.stage_sequence),
            fresh_reobservation_authorized=True,
            fresh_reobservation_evidence_id=state.reobservation_evidence_id,
        )
        if budget_decision.get("status") != "RECOVERY_SLOT_RESERVED_NOT_EXECUTED":
            terminal = replace(
                state,
                phase="TERMINAL_SAFE_ABORT",
                budget_state=copy.deepcopy(budget_decision["state"]),
                consumed_evidence_ids=consumed,
                step_count=state.step_count + 1,
                terminal_reason=str(
                    budget_decision.get("rejection_code")
                    or "RECOVERY_BUDGET_REJECTED"
                ),
                offline_fixture_path_only=state.offline_fixture_path_only or offline,
            )
            return terminal, _result(
                terminal,
                state_changed=True,
                rejection_code=terminal.terminal_reason,
                reentry_request_ready=False,
            )
        advanced = replace(
            state,
            phase="REENTRY_REQUEST_READY",
            budget_state=copy.deepcopy(budget_decision["state"]),
            consumed_evidence_ids=consumed,
            step_count=state.step_count + 1,
            offline_fixture_path_only=state.offline_fixture_path_only or offline,
        )
        return advanced, _result(
            advanced,
            state_changed=True,
            rejection_code=None,
            reentry_request_ready=True,
        )
    return advanced, _result(
        advanced,
        state_changed=True,
        rejection_code=None,
        reentry_request_ready=False,
    )


def current_recovery_snapshot(
    contract: RecoveryWorkflowContract,
) -> dict[str, Any]:
    state = initial_recovery_workflow_state()
    return {
        **_result(
            state,
            state_changed=False,
            rejection_code="NO_F1_SAFE_ABORT_TO_RECOVER",
            reentry_request_ready=False,
        ),
        "current_f1_state": contract.current_f1_state,
        "current_f1_dynamic_task_passed": contract.current_f1_dynamic_task_passed,
    }


__all__ = [
    "PHASES",
    "RecoveryEvidence",
    "RecoveryWorkflowContract",
    "RecoveryWorkflowState",
    "current_recovery_snapshot",
    "initial_recovery_workflow_state",
    "load_recovery_workflow_contract",
    "step_recovery_workflow",
]
