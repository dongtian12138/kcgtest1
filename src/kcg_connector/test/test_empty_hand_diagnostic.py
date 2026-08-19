'''Behavior tests for the empty-hand diagnostic termination classifier.'''

from __future__ import annotations

import pytest

from kcg_connector.grasp.empty_hand_diagnostic import (
    ALLOWED_EXIT3_REASONS,
    classify_empty_hand_diagnostic_termination,
)


def test_allowed_exit3_reason_set_is_frozen_to_three():
    assert ALLOWED_EXIT3_REASONS == (
        "empty_hand_wrist_force_gate_observed",
        "empty_hand_wrist_moment_gate_observed",
        "stage1_completed_without_gate",
    )


@pytest.mark.parametrize(
    "reason",
    ["empty_hand_wrist_force_gate_observed", "empty_hand_wrist_moment_gate_observed"],
)
def test_gate_observation_classifies_exit3(reason):
    classification = classify_empty_hand_diagnostic_termination(
        reason, termination_reason=reason, stage1_completed=False
    )
    assert classification == {
        "exit_code": 3,
        "reason": reason,
        "fault_category": None,
    }


def test_stage1_completion_without_gate_classifies_exit3():
    classification = classify_empty_hand_diagnostic_termination(
        None, termination_reason=None, stage1_completed=True
    )
    assert classification == {
        "exit_code": 3,
        "reason": "stage1_completed_without_gate",
        "fault_category": None,
    }


@pytest.mark.parametrize(
    "reason, category",
    [
        ("empty_hand_nonfinite_sensor_or_robot_state", "nonfinite"),
        ("empty_hand_nonfinite_effort", "nonfinite"),
        ("empty_hand_arm_tracking_gate_observed", "arm_tracking"),
        ("empty_hand_finger_speed_gate_observed", "finger_speed"),
        ("empty_hand_open_target_invariant_broken", "open_target_invariant"),
    ],
)
def test_robot_faults_classify_exit1_with_preserved_reason(reason, category):
    classification = classify_empty_hand_diagnostic_termination(
        reason, termination_reason=reason, stage1_completed=False
    )
    assert classification == {
        "exit_code": 1,
        "reason": reason,
        "fault_category": category,
    }


def test_sensor_fault_classifies_exit1():
    classification = classify_empty_hand_diagnostic_termination(
        "empty_hand_wrist_ft_sensor_error",
        termination_reason="empty_hand_wrist_ft_sensor_error",
        sensor_fault_reason="hand2arm reaction wrench unavailable",
    )
    assert classification == {
        "exit_code": 1,
        "reason": "empty_hand_wrist_ft_sensor_error",
        "fault_category": "wrist_sensor_error",
    }


def test_return_error_classifies_exit1():
    classification = classify_empty_hand_diagnostic_termination(
        None, return_error="RuntimeError: return interrupted",
    )
    assert classification["exit_code"] == 1
    assert classification["fault_category"] == "return_error"
    assert "return interrupted" in classification["reason"]


def test_snapshot_error_classifies_exit1():
    classification = classify_empty_hand_diagnostic_termination(
        "empty_hand_wrist_moment_gate_observed",
        termination_reason="empty_hand_wrist_moment_gate_observed",
        snapshot_errors=("stage_terminal:ValueError:boom",),
    )
    assert classification["exit_code"] == 1
    assert classification["fault_category"] == "snapshot_error"


def test_open_target_invariant_flag_classifies_exit1():
    classification = classify_empty_hand_diagnostic_termination(
        None, open_target_invariant_ok=False, stage1_completed=True
    )
    assert classification == {
        "exit_code": 1,
        "reason": "empty_hand_open_target_invariant_broken",
        "fault_category": "open_target_invariant",
    }


def test_incomplete_evidence_without_fault_is_exit1():
    classification = classify_empty_hand_diagnostic_termination(
        None, stage1_completed=False
    )
    assert classification == {
        "exit_code": 1,
        "reason": "empty_hand_diagnostic_incomplete",
        "fault_category": "incomplete_evidence",
    }



def test_stage1_completed_flag_requires_loop_exhaustion():
    from kcg_connector.grasp.empty_hand_diagnostic import (
        stage1_completed_flag,
    )

    assert stage1_completed_flag(True, None) is True
    assert stage1_completed_flag(False, None) is False
    termination = {
        "kind": "diagnostic_gate_observation",
        "reason": "empty_hand_wrist_moment_gate_observed",
        "stage_step": 3,
    }
    assert stage1_completed_flag(True, termination) is False


def test_stage_record_failure_evidence_maps_termination():
    from kcg_connector.grasp.empty_hand_diagnostic import (
        stage_record_failure_evidence,
    )

    termination = {
        "kind": "sensor_or_robot_fault",
        "reason": "empty_hand_open_target_invariant_broken",
        "stage_step": 7,
    }
    assert stage_record_failure_evidence(termination) == {
        "passed_sensor_gate": False,
        "failure_reason": "empty_hand_open_target_invariant_broken",
        "failure_step": 7,
        "termination_kind": "sensor_or_robot_fault",
    }


def test_diagnostic_report_status_maps_exit3():
    from kcg_connector.grasp.empty_hand_diagnostic import (
        diagnostic_report_status,
    )

    assert diagnostic_report_status(3) == {
        "empty_hand_diagnostic_completed": True,
        "wrist_status": "EMPTY_HAND_DIAGNOSTIC_COMPLETED",
        "console_label": (
            "ISAAC EMPTY HAND FIRST STAGE REPLAY "
            "DIAGNOSTIC_ONLY COMPLETED"
        ),
    }


def test_diagnostic_report_status_maps_exit1():
    from kcg_connector.grasp.empty_hand_diagnostic import (
        diagnostic_report_status,
    )

    assert diagnostic_report_status(1) == {
        "empty_hand_diagnostic_completed": False,
        "wrist_status": "EMPTY_HAND_DIAGNOSTIC_FAILED",
        "console_label": (
            "ISAAC EMPTY HAND FIRST STAGE REPLAY "
            "DIAGNOSTIC_ONLY FAILED"
        ),
    }


def test_classifier_and_status_mapping_for_gate_observation():
    from kcg_connector.grasp.empty_hand_diagnostic import (
        diagnostic_report_status,
    )

    classification = classify_empty_hand_diagnostic_termination(
        "empty_hand_wrist_moment_gate_observed",
        termination_reason="empty_hand_wrist_moment_gate_observed",
        stage1_completed=False,
    )
    status = diagnostic_report_status(classification["exit_code"])
    assert classification["exit_code"] == 3
    assert status["empty_hand_diagnostic_completed"] is True
    assert status["wrist_status"].endswith("COMPLETED")
    assert status["console_label"].endswith("COMPLETED")


def test_classifier_and_status_mapping_for_fault():
    from kcg_connector.grasp.empty_hand_diagnostic import (
        diagnostic_report_status,
    )

    classification = classify_empty_hand_diagnostic_termination(
        "empty_hand_arm_tracking_gate_observed",
        termination_reason="empty_hand_arm_tracking_gate_observed",
        stage1_completed=False,
    )
    status = diagnostic_report_status(classification["exit_code"])
    assert classification["exit_code"] == 1
    assert status["empty_hand_diagnostic_completed"] is False
    assert status["wrist_status"].endswith("FAILED")
    assert status["console_label"].endswith("FAILED")

