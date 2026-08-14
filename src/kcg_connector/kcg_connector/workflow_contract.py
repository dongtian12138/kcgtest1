"""Pure end-to-end connector workflow stages and evidence gates."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from numbers import Real
from typing import Any, Mapping


class ConnectorWorkflowStage(str, Enum):
    """Ordered stages from first observation through return to home."""

    DETECT_LOOSE = "DETECT_LOOSE"
    PICK = "PICK"
    IN_HAND_RELOCALIZE = "IN_HAND_RELOCALIZE"
    DETECT_FIXED = "DETECT_FIXED"
    PREALIGN = "PREALIGN"
    INSERT = "INSERT"
    ENGAGE = "ENGAGE"
    SCREW = "SCREW"
    VERIFY = "VERIFY"
    RETREAT = "RETREAT"
    HOME = "HOME"


WORKFLOW_STAGE_ORDER = tuple(ConnectorWorkflowStage)


# Evidence names are positive assertions: a false or null value never satisfies
# a gate.  Actual detector/controller payloads stay outside this pure contract.
STAGE_EVIDENCE_REQUIREMENTS = {
    ConnectorWorkflowStage.DETECT_LOOSE: (
        "registry_profile_enabled",
        "loose_model_id_matches_profile",
        "loose_pose_6d_with_uncertainty",
        "loose_assembly_frame_resolved",
    ),
    ConnectorWorkflowStage.PICK: (
        "selected_grasp_region_is_registered",
        "collision_checked_pick_plan",
        "grasp_force_within_registered_limit",
        "lift_and_hold_confirms_retention",
    ),
    ConnectorWorkflowStage.IN_HAND_RELOCALIZE: (
        "loose_assembly_frame_in_tool_frame",
        "in_hand_pose_uncertainty_within_limit",
        "post_pick_slip_check_passed",
    ),
    ConnectorWorkflowStage.DETECT_FIXED: (
        "fixed_model_id_matches_profile",
        "fixed_pose_6d_with_uncertainty",
        "fixed_assembly_frame_resolved",
        "loose_fixed_compatibility_confirmed",
    ),
    ConnectorWorkflowStage.PREALIGN: (
        "collision_checked_prealign_plan",
        "lateral_error_within_registered_limit",
        "angular_error_within_registered_limit",
        "key_error_within_registered_limit",
    ),
    ConnectorWorkflowStage.INSERT: (
        "measured_insertion_depth",
        "wrist_force_within_registered_limits",
        "alignment_remains_within_registered_limits",
        "finger_base_torques_stable",
    ),
    ConnectorWorkflowStage.ENGAGE: (
        "minimum_engage_depth_reached",
        "engagement_detector_passed",
        "cross_thread_check_passed",
        "wrist_wrench_within_registered_limits",
    ),
    ConnectorWorkflowStage.SCREW: (
        "measured_coupling_rotation",
        "measured_axial_thread_progress",
        "rotation_lead_consistency_passed",
        "wrist_wrench_within_registered_limits",
        "finger_base_torques_stable_without_slip",
    ),
    ConnectorWorkflowStage.VERIFY: (
        "final_depth_or_gap_within_limit",
        "final_rotation_and_lead_consistent",
        "terminal_wrench_within_registered_band",
        "stable_hold_passed",
        "required_electrical_acceptance_passed_or_not_applicable",
    ),
    ConnectorWorkflowStage.RETREAT: (
        "connector_released_without_overload",
        "assembled_pair_remains_fixed",
        "collision_checked_retreat_completed",
    ),
    ConnectorWorkflowStage.HOME: (
        "robot_home_joint_tolerance_passed",
        "tool_clear_and_empty",
        "workflow_fault_free",
    ),
}


@dataclass(frozen=True)
class StageEvidenceCheck:
    """Result of checking only the declared evidence for one stage."""

    stage: ConnectorWorkflowStage
    complete: bool
    missing: tuple[str, ...]


def required_evidence_for(
    stage: ConnectorWorkflowStage | str,
) -> tuple[str, ...]:
    """Return the fixed evidence keys for a stage or reject unknown stages."""
    try:
        normalized = ConnectorWorkflowStage(stage)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"unknown connector workflow stage: {stage!r}"
        ) from error
    return STAGE_EVIDENCE_REQUIREMENTS[normalized]


def _present(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, Real) and not math.isfinite(float(value)):
        return False
    if isinstance(value, (str, bytes, tuple, list, dict, set)) and not value:
        return False
    return True


def check_stage_evidence(
    stage: ConnectorWorkflowStage | str,
    evidence: Mapping[str, Any],
) -> StageEvidenceCheck:
    """Fail closed when a required positive assertion is absent or invalid."""
    if not isinstance(evidence, Mapping):
        raise ValueError("stage evidence must be a mapping")
    normalized = ConnectorWorkflowStage(stage)
    missing = tuple(
        name
        for name in required_evidence_for(normalized)
        if name not in evidence or not _present(evidence[name])
    )
    return StageEvidenceCheck(
        stage=normalized,
        complete=not missing,
        missing=missing,
    )
