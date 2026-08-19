"""Pure termination classification for the empty-hand diagnostic replay.

This module has no Isaac dependency: the runner feeds it the monitor
summary and fault flags and it returns the exact exit code, terminal
reason and fault category.  Normal exit-3 reasons are frozen to the
three values below; every sensor/robot/evidence fault is exit 1.
"""

from __future__ import annotations

from typing import Any, Sequence

DIAGNOSTIC_MODE = "EMPTY_HAND_FIRST_STAGE_REPLAY_DIAGNOSTIC_ONLY"

ALLOWED_EXIT3_REASONS = (
    "empty_hand_wrist_force_gate_observed",
    "empty_hand_wrist_moment_gate_observed",
    "stage1_completed_without_gate",
)

FAULT_REASONS = (
    "empty_hand_wrist_ft_sensor_error",
    "empty_hand_nonfinite_effort",
    "empty_hand_nonfinite_sensor_or_robot_state",
    "empty_hand_arm_tracking_gate_observed",
    "empty_hand_finger_speed_gate_observed",
    "empty_hand_open_target_invariant_broken",
)


def classify_empty_hand_diagnostic_termination(
    monitor_failure_reason: str | None,
    *,
    termination_reason: str | None = None,
    stage1_completed: bool = False,
    open_target_invariant_ok: bool = True,
    sensor_fault_reason: str | None = None,
    return_error: str | None = None,
    snapshot_errors: Sequence[str] = (),
) -> dict[str, Any]:
    """Classify the diagnostic terminal state into exit-1/exit-3.

    Exit 3 is reserved for the frozen gate observations and a clean
    stage-1 completion.  Everything else (sensor faults, nonfinite
    samples, arm tracking, finger speed, open-target invariant breaks,
    return errors, snapshot errors) is exit 1 with the original reason
    preserved and a fault category attached.
    """
    reason = termination_reason or monitor_failure_reason
    faults: list[tuple[str, str]] = []
    if sensor_fault_reason is not None:
        faults.append(
            ("wrist_sensor_error", reason or "empty_hand_wrist_ft_sensor_error")
        )
    if reason == "empty_hand_nonfinite_effort":
        faults.append(("nonfinite", reason))
    elif reason == "empty_hand_nonfinite_sensor_or_robot_state":
        faults.append(("nonfinite", reason))
    elif reason == "empty_hand_arm_tracking_gate_observed":
        faults.append(("arm_tracking", reason))
    elif reason == "empty_hand_finger_speed_gate_observed":
        faults.append(("finger_speed", reason))
    elif reason == "empty_hand_open_target_invariant_broken":
        faults.append(("open_target_invariant", reason))
    if not open_target_invariant_ok and not faults:
        faults.append(
            ("open_target_invariant", "empty_hand_open_target_invariant_broken")
        )
    if return_error is not None:
        faults.append(("return_error", return_error))
    if snapshot_errors:
        faults.append(("snapshot_error", str(snapshot_errors[0])))
    if faults:
        category, fault_reason = faults[0]
        return {
            "exit_code": 1,
            "reason": fault_reason,
            "fault_category": category,
        }
    if reason in ALLOWED_EXIT3_REASONS[:2]:
        return {"exit_code": 3, "reason": reason, "fault_category": None}
    if stage1_completed:
        return {
            "exit_code": 3,
            "reason": "stage1_completed_without_gate",
            "fault_category": None,
        }
    return {
        "exit_code": 1,
        "reason": "empty_hand_diagnostic_incomplete",
        "fault_category": "incomplete_evidence",
    }


def stage1_completed_flag(
    loop_exhausted: bool, termination: Mapping[str, Any] | None
) -> bool:
    """Stage-1 completed only when the loop ran to exhaustion without

    any termination.  Early exits (open-target invariant break, nonfinite
    effort, monitor gate or fault) all yield False -- never inferred from
    a negative."""
    return bool(loop_exhausted and termination is None)


def stage_record_failure_evidence(termination: Mapping[str, Any]) -> dict[str, Any]:
    """Pure mapping of a stage termination onto stage_record evidence."""
    return {
        "passed_sensor_gate": False,
        "failure_reason": termination.get("reason"),
        "failure_step": termination.get("stage_step"),
        "termination_kind": termination.get("kind"),
    }


def diagnostic_report_status(exit_code: int) -> dict[str, Any]:
    """Report/wrist/console status mapping for the diagnostic terminal.

    Exit 3 is the only completed outcome; exit 1 writes FAILED everywhere
    with the classification reason and category preserved for humans and
    scripts."""
    completed = bool(exit_code == 3)
    return {
        "empty_hand_diagnostic_completed": completed,
        "wrist_status": (
            "EMPTY_HAND_DIAGNOSTIC_COMPLETED"
            if completed
            else "EMPTY_HAND_DIAGNOSTIC_FAILED"
        ),
        "console_label": (
            "ISAAC EMPTY HAND FIRST STAGE REPLAY "
            "DIAGNOSTIC_ONLY COMPLETED"
            if completed
            else "ISAAC EMPTY HAND FIRST STAGE REPLAY "
            "DIAGNOSTIC_ONLY FAILED"
        ),
    }
