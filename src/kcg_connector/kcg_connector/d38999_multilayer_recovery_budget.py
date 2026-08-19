"""Fail-closed, non-executing recovery-attempt budget for D38999 insertion."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "kcg_d38999_multilayer_recovery_budget_v1"
MAXIMUM_RECOVERY_ATTEMPTS = 2
EXPECTED_D9_STAGE_SEQUENCE = [
    "STOP_ZERO_TWIST",
    "RETRACT_REQUEST",
    "HOLD_FOR_SETTLE_REQUEST",
    "PRECOMMITTED_REOBSERVE_REQUEST",
]
FROZEN_SOURCES = {
    "artifacts/agent_control/tasks/EIGHT-HOUR-D8-OVERFORCE-EXIT/"
    "TASK_RESULT.json": (
        "fa07a1ca8842b2799fb26ef91446ae64347d168c477bebffb76147e6524c81f3"
    ),
    "artifacts/agent_control/tasks/EIGHT-HOUR-D9-BACKOFF-REOBSERVE/"
    "TASK_RESULT.json": (
        "97c8452ddcba39936377416d65d4f9688c538744bb55f0492aca929fb36b4df8"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_recovery_budget_contract(repository_root: str | Path) -> dict[str, Any]:
    """Load and verify the frozen D8/D9 evidence used by the budget."""

    root = Path(repository_root).resolve()
    verified_sources: list[dict[str, str]] = []
    documents: dict[str, Mapping[str, Any]] = {}
    for relative, expected in FROZEN_SOURCES.items():
        path = root / relative
        if not path.is_file():
            raise ValueError(f"frozen D10 source missing: {relative}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"frozen D10 source hash mismatch: {relative}")
        documents[relative] = json.loads(path.read_text(encoding="utf-8"))
        verified_sources.append({"path": relative, "sha256": actual})

    d8 = documents[next(relative for relative in documents if "D8-" in relative)]
    d9 = documents[next(relative for relative in documents if "D9-" in relative)]
    if (
        d8.get("outcome") != "OFFLINE_PASS"
        or d8.get("backoff_distance_m") != 0.0004
        or d8.get("backoff_speed_m_s") != 0.0003
        or d8.get("dynamic_overforce_exit_pass_claimed") is not False
        or d9.get("outcome") != "OFFLINE_PASS"
        or d9.get("stage_sequence") != EXPECTED_D9_STAGE_SEQUENCE
        or d9.get("selected_view_for_execution") is not None
        or d9.get("dynamic_reobserve_pass_claimed") is not False
    ):
        raise ValueError("authoritative D10 recovery evidence changed")
    return {
        "schema_version": SCHEMA_VERSION,
        "maximum_recovery_attempts": MAXIMUM_RECOVERY_ATTEMPTS,
        "backoff_distance_m": d8["backoff_distance_m"],
        "backoff_speed_m_s": d8["backoff_speed_m_s"],
        "stage_sequence": list(d9["stage_sequence"]),
        "candidate_view_sequence": list(d9["candidate_view_sequence"]),
        "sources": verified_sources,
        "execution_authorized": False,
        "dynamic_recovery_pass_claimed": False,
    }


def new_recovery_budget_state() -> dict[str, Any]:
    """Return an empty JSON-serializable recovery budget."""

    return {
        "schema_version": SCHEMA_VERSION,
        "maximum_recovery_attempts": MAXIMUM_RECOVERY_ATTEMPTS,
        "attempts_reserved": [],
        "terminal_safe_abort": False,
        "terminal_reason": None,
        "motion_command_emitted": False,
        "capture_command_emitted": False,
        "control_authorized": False,
        "dynamic_recovery_pass_claimed": False,
        "hardware_authorized": False,
    }


def _validated_state(state: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(state, Mapping):
        raise ValueError("recovery budget state must be a mapping")
    copied = copy.deepcopy(dict(state))
    attempts = copied.get("attempts_reserved")
    if (
        copied.get("schema_version") != SCHEMA_VERSION
        or copied.get("maximum_recovery_attempts") != MAXIMUM_RECOVERY_ATTEMPTS
        or not isinstance(attempts, list)
        or len(attempts) > MAXIMUM_RECOVERY_ATTEMPTS
        or type(copied.get("terminal_safe_abort")) is not bool
        or any(copied.get(key) is not False for key in (
            "motion_command_emitted",
            "capture_command_emitted",
            "control_authorized",
            "dynamic_recovery_pass_claimed",
            "hardware_authorized",
        ))
    ):
        raise ValueError("invalid recovery budget state")
    request_ids: list[str] = []
    exit_event_ids: list[str] = []
    reobserve_ids: list[str] = []
    for index, attempt in enumerate(attempts, start=1):
        if (
            not isinstance(attempt, Mapping)
            or attempt.get("attempt_number") != index
            or not all(isinstance(attempt.get(key), str) and attempt[key] for key in (
                "request_id", "d8_exit_event_id", "fresh_reobservation_evidence_id"
            ))
            or attempt.get("plan_stage_sequence") != EXPECTED_D9_STAGE_SEQUENCE
        ):
            raise ValueError("invalid recovery attempt record")
        request_ids.append(attempt["request_id"])
        exit_event_ids.append(attempt["d8_exit_event_id"])
        reobserve_ids.append(attempt["fresh_reobservation_evidence_id"])
    if (
        len(request_ids) != len(set(request_ids))
        or len(exit_event_ids) != len(set(exit_event_ids))
        or len(reobserve_ids) != len(set(reobserve_ids))
    ):
        raise ValueError("recovery evidence identifiers must be unique")
    return copied


def _decision(
    state: dict[str, Any],
    *,
    status: str,
    rejection_code: str | None,
    attempt_number: int | None,
    state_changed: bool,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "rejection_code": rejection_code,
        "attempt_number": attempt_number,
        "state_changed": state_changed,
        "state": state,
        "motion_command_emitted": False,
        "capture_command_emitted": False,
        "control_authorized": False,
        "dynamic_recovery_pass_claimed": False,
        "hardware_authorized": False,
    }


def request_recovery_slot(
    state: Mapping[str, Any],
    *,
    request_id: str,
    d8_exit_latched: bool,
    d8_failure_reason: str,
    d8_exit_event_id: str,
    d9_plan_completed: bool,
    d9_plan_stage_sequence: list[str],
    fresh_reobservation_authorized: bool,
    fresh_reobservation_evidence_id: str,
) -> dict[str, Any]:
    """Reserve at most two logical recovery slots without executing either one."""

    try:
        current = _validated_state(state)
    except ValueError:
        safe = new_recovery_budget_state()
        safe["terminal_safe_abort"] = True
        safe["terminal_reason"] = "INVALID_BUDGET_STATE"
        return _decision(
            safe,
            status="SAFE_ABORT",
            rejection_code="INVALID_BUDGET_STATE",
            attempt_number=None,
            state_changed=True,
        )

    if current["terminal_safe_abort"]:
        return _decision(
            current,
            status="SAFE_ABORT",
            rejection_code="RECOVERY_ALREADY_TERMINAL",
            attempt_number=None,
            state_changed=False,
        )
    string_inputs = (
        request_id,
        d8_failure_reason,
        d8_exit_event_id,
        fresh_reobservation_evidence_id,
    )
    if (
        any(not isinstance(value, str) or not value.strip() for value in string_inputs)
        or type(d8_exit_latched) is not bool
        or type(d9_plan_completed) is not bool
        or type(fresh_reobservation_authorized) is not bool
        or not isinstance(d9_plan_stage_sequence, list)
    ):
        return _decision(
            current,
            status="RECOVERY_REJECTED",
            rejection_code="INVALID_RECOVERY_REQUEST",
            attempt_number=None,
            state_changed=False,
        )

    for attempt in current["attempts_reserved"]:
        if attempt["request_id"] == request_id:
            same_payload = (
                attempt["d8_exit_event_id"] == d8_exit_event_id
                and attempt["fresh_reobservation_evidence_id"]
                == fresh_reobservation_evidence_id
            )
            if same_payload:
                return _decision(
                    current,
                    status="RECOVERY_SLOT_ALREADY_RESERVED",
                    rejection_code=None,
                    attempt_number=attempt["attempt_number"],
                    state_changed=False,
                )
            current["terminal_safe_abort"] = True
            current["terminal_reason"] = "REQUEST_ID_COLLISION"
            return _decision(
                current,
                status="SAFE_ABORT",
                rejection_code="REQUEST_ID_COLLISION",
                attempt_number=None,
                state_changed=True,
            )

    if (
        not d8_exit_latched
        or not d9_plan_completed
        or d9_plan_stage_sequence != EXPECTED_D9_STAGE_SEQUENCE
        or not fresh_reobservation_authorized
    ):
        return _decision(
            current,
            status="RECOVERY_REJECTED",
            rejection_code="RECOVERY_PREREQUISITES_NOT_MET",
            attempt_number=None,
            state_changed=False,
        )
    if any(
        attempt["d8_exit_event_id"] == d8_exit_event_id
        for attempt in current["attempts_reserved"]
    ):
        return _decision(
            current,
            status="RECOVERY_REJECTED",
            rejection_code="D8_EXIT_EVENT_NOT_FRESH",
            attempt_number=None,
            state_changed=False,
        )
    if any(
        attempt["fresh_reobservation_evidence_id"]
        == fresh_reobservation_evidence_id
        for attempt in current["attempts_reserved"]
    ):
        return _decision(
            current,
            status="RECOVERY_REJECTED",
            rejection_code="REOBSERVATION_EVIDENCE_NOT_FRESH",
            attempt_number=None,
            state_changed=False,
        )
    if len(current["attempts_reserved"]) >= MAXIMUM_RECOVERY_ATTEMPTS:
        current["terminal_safe_abort"] = True
        current["terminal_reason"] = "RECOVERY_LIMIT_REACHED"
        return _decision(
            current,
            status="SAFE_ABORT",
            rejection_code="RECOVERY_LIMIT_REACHED",
            attempt_number=None,
            state_changed=True,
        )

    attempt_number = len(current["attempts_reserved"]) + 1
    current["attempts_reserved"].append({
        "attempt_number": attempt_number,
        "request_id": request_id,
        "d8_exit_event_id": d8_exit_event_id,
        "fresh_reobservation_evidence_id": fresh_reobservation_evidence_id,
        "plan_stage_sequence": list(EXPECTED_D9_STAGE_SEQUENCE),
        "logical_slot_only": True,
    })
    return _decision(
        current,
        status="RECOVERY_SLOT_RESERVED_NOT_EXECUTED",
        rejection_code=None,
        attempt_number=attempt_number,
        state_changed=True,
    )


__all__ = [
    "EXPECTED_D9_STAGE_SEQUENCE",
    "FROZEN_SOURCES",
    "MAXIMUM_RECOVERY_ATTEMPTS",
    "SCHEMA_VERSION",
    "load_recovery_budget_contract",
    "new_recovery_budget_state",
    "request_recovery_slot",
]
