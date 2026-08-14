import math

import numpy as np
import pytest

from kcg_connector.workflow_contract import (
    ConnectorWorkflowStage,
    STAGE_EVIDENCE_REQUIREMENTS,
    WORKFLOW_STAGE_ORDER,
    check_stage_evidence,
    required_evidence_for,
)


EXPECTED_STAGE_NAMES = (
    "DETECT_LOOSE",
    "PICK",
    "IN_HAND_RELOCALIZE",
    "DETECT_FIXED",
    "PREALIGN",
    "INSERT",
    "ENGAGE",
    "SCREW",
    "VERIFY",
    "RETREAT",
    "HOME",
)


def test_end_to_end_stage_order_is_explicit_and_complete():
    assert tuple(stage.value for stage in WORKFLOW_STAGE_ORDER) == (
        EXPECTED_STAGE_NAMES
    )
    assert tuple(STAGE_EVIDENCE_REQUIREMENTS) == WORKFLOW_STAGE_ORDER


@pytest.mark.parametrize("stage", WORKFLOW_STAGE_ORDER)
def test_every_stage_has_unique_positive_evidence_requirements(stage):
    required = required_evidence_for(stage)
    assert required
    assert len(required) == len(set(required))
    assert all(isinstance(name, str) and name for name in required)


def test_key_workflow_evidence_is_not_reduced_to_commanded_motion():
    assert "loose_pose_6d_with_uncertainty" in required_evidence_for(
        ConnectorWorkflowStage.DETECT_LOOSE
    )
    assert "loose_assembly_frame_in_tool_frame" in required_evidence_for(
        ConnectorWorkflowStage.IN_HAND_RELOCALIZE
    )
    assert "loose_fixed_compatibility_confirmed" in required_evidence_for(
        ConnectorWorkflowStage.DETECT_FIXED
    )
    assert "measured_coupling_rotation" in required_evidence_for(
        ConnectorWorkflowStage.SCREW
    )
    assert "measured_axial_thread_progress" in required_evidence_for(
        ConnectorWorkflowStage.SCREW
    )
    assert "stable_hold_passed" in required_evidence_for(
        ConnectorWorkflowStage.VERIFY
    )


def test_stage_check_fails_closed_for_bad_values():
    stage = ConnectorWorkflowStage.ENGAGE
    required = required_evidence_for(stage)
    evidence = {name: True for name in required}
    evidence[required[0]] = None
    evidence[required[1]] = False
    evidence[required[2]] = ""
    evidence[required[3]] = math.nan
    result = check_stage_evidence(stage, evidence)
    assert result.stage is stage
    assert result.complete is False
    assert result.missing == required


def test_stage_check_rejects_nonfinite_numpy_scalar_evidence():
    stage = ConnectorWorkflowStage.INSERT
    required = required_evidence_for(stage)
    evidence = {name: True for name in required}
    evidence[required[0]] = np.float32(np.nan)
    result = check_stage_evidence(stage, evidence)
    assert result.complete is False
    assert result.missing == (required[0],)


@pytest.mark.parametrize("stage", WORKFLOW_STAGE_ORDER)
def test_stage_check_passes_only_when_all_declared_evidence_is_present(stage):
    evidence = {name: True for name in required_evidence_for(stage)}
    result = check_stage_evidence(stage, evidence)
    assert result.complete is True
    assert result.missing == ()


def test_numeric_zero_is_valid_evidence_but_boolean_false_is_not():
    stage = ConnectorWorkflowStage.HOME
    evidence = {name: True for name in required_evidence_for(stage)}
    evidence["robot_home_joint_tolerance_passed"] = 0.0
    assert check_stage_evidence(stage, evidence).complete
    evidence["robot_home_joint_tolerance_passed"] = False
    assert not check_stage_evidence(stage, evidence).complete


def test_unknown_stage_and_non_mapping_evidence_are_rejected():
    with pytest.raises(ValueError, match="unknown connector workflow stage"):
        required_evidence_for("NOT_A_STAGE")
    with pytest.raises(ValueError, match="must be a mapping"):
        check_stage_evidence(ConnectorWorkflowStage.PICK, [])
