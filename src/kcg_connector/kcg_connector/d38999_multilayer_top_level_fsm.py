"""Single fail-closed top-level workflow for multilayer assembly.

The workflow advances exactly one predeclared edge per fresh evidence item.
It has no target-state input and never consumes object/contact/event truth.
Offline fixtures may exercise the graph, but never authorize control or claim a
dynamic assembly result.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

import yaml


TASK_ID = "EIGHT-HOUR-F1-TOP-LEVEL-STATE-MACHINE"
WORK_QUEUE_PATH = "artifacts/agent_control/WORK_QUEUE.yaml"
E8_RESULT_PATH = (
    "artifacts/agent_control/tasks/"
    "EIGHT-HOUR-E8-ARRIVAL-DECISION/TASK_RESULT.json"
)
D10_RESULT_PATH = (
    "artifacts/agent_control/tasks/"
    "EIGHT-HOUR-D10-TWO-RECOVERY-LIMIT/TASK_RESULT.json"
)
B5_RESULT_PATH = (
    "artifacts/agent_control/tasks/"
    "EIGHT-HOUR-B5-WRIST-MOMENT-MONITOR/TASK_RESULT.json"
)

EXPECTED_STATES = (
    "HOME",
    "LOCATE_PLUG",
    "APPROACH_PLUG",
    "GRASP",
    "LIFT_CHECK",
    "POST_GRASP_OBSERVE",
    "LOCATE_RECEPTACLE",
    "PREALIGN",
    "CONTACT",
    "CENTER",
    "LEVEL",
    "KEY_SEARCH",
    "INSERT",
    "VERIFY_INSERTION",
    "REGRASP_NUT",
    "TIGHTEN",
    "VERIFY_ASSEMBLY",
    "RELEASE",
    "RETRACT",
    "DONE",
    "SAFE_ABORT",
)

NORMAL_TRANSITION_GATES = (
    "mission_start_authorized",
    "plug_localized",
    "plug_approach_complete",
    "grasp_dynamic_pass",
    "lift_check_dynamic_pass",
    "postgrasp_visual_dynamic_pass",
    "receptacle_localized",
    "prealign_dynamic_pass",
    "light_contact_dynamic_pass",
    "centering_dynamic_pass",
    "leveling_dynamic_pass",
    "key_search_dynamic_pass",
    "insertion_dynamic_pass",
    "postinsert_hold_and_body_release_dynamic_pass",
    "nut_regrasp_dynamic_pass",
    "tightening_dynamic_pass",
    "assembly_arrival_dynamic_pass",
    "final_release_dynamic_pass",
    "retract_complete",
)

FROZEN_SOURCES = {
    B5_RESULT_PATH: (
        "2bebe773c145d4afec89cdf1865ae97eb13db8bf9019d2006bb95ba635c38e0f"
    ),
    D10_RESULT_PATH: (
        "6ba295872ebd695dc3894ad210085c344061b65435d3e8f5070eaf707f38f32c"
    ),
    E8_RESULT_PATH: (
        "367113c899bf38393b73b0af94485cf23233c348f5dcef85657f8de96fc147ab"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


@dataclass(frozen=True)
class TopLevelWorkflowContract:
    source_rows: tuple[tuple[str, str], ...]
    states: tuple[str, ...]
    normal_states: tuple[str, ...]
    normal_transition_gates: tuple[str, ...]
    transition_rows: tuple[tuple[str, str, str], ...]
    safe_abort_state: str
    safe_abort_latched: bool
    maximum_recovery_attempts: int
    moment_component_limit_nm: float
    current_e8_outcome: str
    current_e8_dynamic_assembly_passed: bool
    current_e8_evidence_sha256: str


@dataclass(frozen=True)
class WorkflowState:
    state: str
    visited_states: tuple[str, ...]
    consumed_evidence_ids: tuple[str, ...]
    step_count: int
    abort_reason: str | None
    offline_fixture_path_only: bool


@dataclass(frozen=True)
class StepEvidence:
    gate_name: str
    evidence_id: str
    evidence_sha256: str
    passed: bool
    fresh: bool
    evidence_level: str
    safety_abort_requested: bool
    safety_reason: str | None
    controller_truth_used: bool


def load_top_level_workflow_contract(
    repository_root: str | Path,
) -> TopLevelWorkflowContract:
    root = Path(repository_root).resolve()
    rows: list[tuple[str, str]] = []
    for relative, expected in FROZEN_SOURCES.items():
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError(f"frozen F1 source missing: {relative}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"frozen F1 source hash mismatch: {relative}")
        rows.append((relative, actual))

    queue = _mapping(
        yaml.safe_load((root / WORK_QUEUE_PATH).read_text(encoding="utf-8")),
        "work queue",
    )
    groups = _mapping(queue.get("groups"), "work queue groups")
    f_group = _mapping(groups.get("F"), "work queue F group")
    tasks = _mapping(f_group.get("tasks"), "work queue F tasks")
    f1 = _mapping(tasks.get("F1"), "work queue F1")
    states = f1.get("states")
    if not isinstance(states, list) or tuple(states) != EXPECTED_STATES:
        raise ValueError("F1 state list differs from the frozen queue")

    b5 = _mapping(
        json.loads((root / B5_RESULT_PATH).read_text(encoding="utf-8")),
        "B5 task result",
    )
    if (
        b5.get("status") != "OFFLINE_PASS"
        or b5.get("moment_limit_nm") != 0.30
        or b5.get("fault_latching") is not True
        or b5.get("explicit_reset_required") is not True
    ):
        raise ValueError("B5 does not preserve the latched safety boundary")
    d10 = _mapping(
        json.loads((root / D10_RESULT_PATH).read_text(encoding="utf-8")),
        "D10 task result",
    )
    if (
        d10.get("outcome") != "OFFLINE_PASS"
        or d10.get("maximum_recovery_attempts") != 2
        or d10.get("third_attempt_action") != "SAFE_ABORT"
    ):
        raise ValueError("D10 recovery boundary changed")
    e8 = _mapping(
        json.loads((root / E8_RESULT_PATH).read_text(encoding="utf-8")),
        "E8 task result",
    )
    if (
        e8.get("outcome") != "OFFLINE_PASS"
        or type(e8.get("dynamic_assembly_pass_claimed")) is not bool
        or e8.get("assembly_arrival_proven") is not False
        or e8.get("event_or_contact_truth_routed_to_controller") is not False
    ):
        raise ValueError("E8 evidence does not support F1")

    normal_states = EXPECTED_STATES[:-1]
    if len(normal_states) - 1 != len(NORMAL_TRANSITION_GATES):
        raise ValueError("normal state and evidence-gate counts differ")
    transitions = tuple(
        (normal_states[index], normal_states[index + 1], gate)
        for index, gate in enumerate(NORMAL_TRANSITION_GATES)
    )
    return TopLevelWorkflowContract(
        source_rows=tuple(rows),
        states=EXPECTED_STATES,
        normal_states=normal_states,
        normal_transition_gates=NORMAL_TRANSITION_GATES,
        transition_rows=transitions,
        safe_abort_state="SAFE_ABORT",
        safe_abort_latched=True,
        maximum_recovery_attempts=2,
        moment_component_limit_nm=0.30,
        current_e8_outcome=str(e8.get("outcome")),
        current_e8_dynamic_assembly_passed=e8[
            "dynamic_assembly_pass_claimed"
        ],
        current_e8_evidence_sha256=FROZEN_SOURCES[E8_RESULT_PATH],
    )


def initial_workflow_state() -> WorkflowState:
    return WorkflowState(
        state="HOME",
        visited_states=("HOME",),
        consumed_evidence_ids=(),
        step_count=0,
        abort_reason=None,
        offline_fixture_path_only=False,
    )


def _validate_state(
    contract: TopLevelWorkflowContract,
    state: WorkflowState,
) -> None:
    if state.state not in contract.states:
        raise ValueError("workflow state is not declared")
    if isinstance(state.step_count, bool) or not isinstance(state.step_count, int):
        raise ValueError("workflow step count must be an integer")
    if state.step_count < 0 or not state.visited_states:
        raise ValueError("workflow history is invalid")
    if state.visited_states[-1] != state.state:
        raise ValueError("workflow history must end in the current state")
    if any(item not in contract.states for item in state.visited_states):
        raise ValueError("workflow history contains an unknown state")


def _result(
    state: WorkflowState,
    *,
    state_advanced: bool,
    rejection_code: str | None,
    dynamic_transition_claimed: bool,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task_id": TASK_ID,
        "state": state.state,
        "visited_states": list(state.visited_states),
        "step_count": state.step_count,
        "abort_reason": state.abort_reason,
        "state_advanced": state_advanced,
        "rejection_code": rejection_code,
        "offline_fixture_path_only": state.offline_fixture_path_only,
        "dynamic_transition_claimed": dynamic_transition_claimed,
        "object_or_contact_truth_used": False,
        "software_pose_write_requested": False,
        "robot_commands_emitted": 0,
        "dynamic_task_pass_claimed": False,
        "control_authorized": False,
    }


def step_top_level_workflow(
    contract: TopLevelWorkflowContract,
    state: WorkflowState,
    evidence: StepEvidence,
) -> tuple[WorkflowState, dict[str, Any]]:
    """Advance one fixed edge, hold, or latch SAFE_ABORT."""

    _validate_state(contract, state)
    if state.state == contract.safe_abort_state:
        return state, _result(
            state,
            state_advanced=False,
            rejection_code="SAFE_ABORT_LATCHED_EXPLICIT_RECOVERY_REQUIRED",
            dynamic_transition_claimed=False,
        )
    if state.state == "DONE":
        return state, _result(
            state,
            state_advanced=False,
            rejection_code="DONE_TERMINAL",
            dynamic_transition_claimed=False,
        )
    if evidence.controller_truth_used is not False:
        aborted = replace(
            state,
            state="SAFE_ABORT",
            visited_states=state.visited_states + ("SAFE_ABORT",),
            step_count=state.step_count + 1,
            abort_reason="CONTROLLER_TRUTH_FIREWALL_VIOLATION",
        )
        return aborted, _result(
            aborted,
            state_advanced=True,
            rejection_code="CONTROLLER_TRUTH_FIREWALL_VIOLATION",
            dynamic_transition_claimed=False,
        )
    if evidence.safety_abort_requested:
        if not evidence.safety_reason:
            raise ValueError("a safety abort requires a reason")
        aborted = replace(
            state,
            state="SAFE_ABORT",
            visited_states=state.visited_states + ("SAFE_ABORT",),
            step_count=state.step_count + 1,
            abort_reason=evidence.safety_reason,
        )
        return aborted, _result(
            aborted,
            state_advanced=True,
            rejection_code="SAFETY_ABORT_LATCHED",
            dynamic_transition_claimed=False,
        )

    index = contract.normal_states.index(state.state)
    expected_from, next_state, expected_gate = contract.transition_rows[index]
    if expected_from != state.state:
        raise ValueError("workflow transition table is inconsistent")
    if evidence.gate_name != expected_gate:
        return state, _result(
            state,
            state_advanced=False,
            rejection_code=f"EXPECTED_GATE:{expected_gate}",
            dynamic_transition_claimed=False,
        )
    if not evidence.evidence_id:
        return state, _result(
            state,
            state_advanced=False,
            rejection_code="EVIDENCE_ID_MISSING",
            dynamic_transition_claimed=False,
        )
    if (
        not isinstance(evidence.evidence_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", evidence.evidence_sha256) is None
    ):
        return state, _result(
            state,
            state_advanced=False,
            rejection_code="EVIDENCE_SHA256_INVALID",
            dynamic_transition_claimed=False,
        )
    if evidence.evidence_id in state.consumed_evidence_ids:
        return state, _result(
            state,
            state_advanced=False,
            rejection_code="EVIDENCE_REPLAY_REJECTED",
            dynamic_transition_claimed=False,
        )
    if evidence.fresh is not True:
        return state, _result(
            state,
            state_advanced=False,
            rejection_code="EVIDENCE_NOT_FRESH",
            dynamic_transition_claimed=False,
        )
    if evidence.passed is not True:
        return state, _result(
            state,
            state_advanced=False,
            rejection_code=f"GATE_NOT_PASSED:{expected_gate}",
            dynamic_transition_claimed=False,
        )
    if evidence.evidence_level not in {
        "offline_fixture",
        "dynamic_signed_artifact",
    }:
        return state, _result(
            state,
            state_advanced=False,
            rejection_code="EVIDENCE_LEVEL_INVALID",
            dynamic_transition_claimed=False,
        )
    offline = evidence.evidence_level == "offline_fixture"
    advanced = replace(
        state,
        state=next_state,
        visited_states=state.visited_states + (next_state,),
        consumed_evidence_ids=state.consumed_evidence_ids + (evidence.evidence_id,),
        step_count=state.step_count + 1,
        offline_fixture_path_only=state.offline_fixture_path_only or offline,
    )
    return advanced, _result(
        advanced,
        state_advanced=True,
        rejection_code=None,
        dynamic_transition_claimed=(not offline),
    )


def current_workflow_snapshot(
    contract: TopLevelWorkflowContract,
) -> dict[str, Any]:
    state = initial_workflow_state()
    return {
        **_result(
            state,
            state_advanced=False,
            rejection_code="MISSION_START_NOT_AUTHORIZED",
            dynamic_transition_claimed=False,
        ),
        "current_e8_dynamic_assembly_passed": (
            contract.current_e8_dynamic_assembly_passed
        ),
        "next_expected_gate": contract.normal_transition_gates[0],
    }


__all__ = [
    "EXPECTED_STATES",
    "NORMAL_TRANSITION_GATES",
    "StepEvidence",
    "TopLevelWorkflowContract",
    "WorkflowState",
    "current_workflow_snapshot",
    "initial_workflow_state",
    "load_top_level_workflow_contract",
    "step_top_level_workflow",
]
