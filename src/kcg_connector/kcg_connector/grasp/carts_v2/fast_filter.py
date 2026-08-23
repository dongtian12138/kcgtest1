"""Cheap hard rejection for all V2 candidates; never a safety certificate."""

from __future__ import annotations

import math

import numpy as np

from kcg_connector.grasp.carts_v2.models import (
    ClosurePrediction,
    FastFilterResult,
    V2Inputs,
    rotation_distance,
)


def _hard_reasons(inputs: V2Inputs, prediction: ClosurePrediction) -> list[str]:
    if prediction.status != "CLOSURE_SURVIVE":
        return [prediction.reason or "CLOSURE_REJECT"]
    reasons: list[str] = []
    contacts = prediction.contacts
    expected_pads = {pad.name for pad in inputs.hand_contract.pads}
    if len(contacts) != 3 or {contact.pad_name for contact in contacts} != expected_pads:
        reasons.append("THREE_DISTINCT_REGISTERED_PADS_NOT_PRESENT")
    face_count = len(inputs.object_contract.model.mesh.faces)
    if any(not 0 <= contact.object_face_index < face_count for contact in contacts):
        reasons.append("CONTACT_FACE_INDEX_OUT_OF_RANGE")
    elif any(
        not inputs.face_roles.face_is_allowed[contact.object_face_index]
        for contact in contacts
    ):
        reasons.append("CONTACT_ON_FORBIDDEN_FACE")
    minimum_area = float(
        inputs.config.section("fast_filter")[
            "minimum_three_contact_triangle_area_m2"
        ]
    )
    areas = inputs.object_contract.model.mesh.face_areas_m2
    if any(areas[contact.object_face_index] < minimum_area for contact in contacts):
        reasons.append("CONTACT_TRIANGLE_TOO_SMALL_FOR_FAST_MODEL")
    try:
        inputs.hand_model.resolve_joint_positions(
            prediction.final_joint_positions_rad, enforce_limits=True
        )
    except ValueError:
        reasons.append("JOINT_LIMIT_VIOLATION")
    if not np.all(np.isfinite(prediction.seed.object_from_hand_matrix())):
        reasons.append("NONFINITE_PALM_POSE")
    return reasons


def _same_realized_grasp(
    left: ClosurePrediction,
    right: ClosurePrediction,
    thresholds,
) -> bool:
    left_pose = left.seed.object_from_hand_matrix()
    right_pose = right.seed.object_from_hand_matrix()
    if (
        np.linalg.norm(left_pose[:3, 3] - right_pose[:3, 3])
        > float(thresholds["palm_position_m"])
        or rotation_distance(left_pose[:3, :3], right_pose[:3, :3])
        > float(thresholds["palm_orientation_rad"])
    ):
        return False
    left_contacts = {row.pad_name: np.asarray(row.object_position_m) for row in left.contacts}
    right_contacts = {
        row.pad_name: np.asarray(row.object_position_m) for row in right.contacts
    }
    if set(left_contacts) != set(right_contacts):
        return False
    squared = [
        float(np.sum((left_contacts[name] - right_contacts[name]) ** 2))
        for name in sorted(left_contacts)
    ]
    return math.sqrt(float(np.mean(squared))) <= float(thresholds["contact_rms_m"])


def fast_filter_predictions(
    inputs: V2Inputs, predictions: tuple[ClosurePrediction, ...]
) -> tuple[FastFilterResult, ...]:
    """Return FAST_REJECT or FAST_SURVIVE without promoting unresolved checks."""

    settings = inputs.config.section("fast_filter")
    thresholds = inputs.config.section("candidate_generation")["deduplication"]
    unresolved = (
        str(settings["arm_ik_policy"]),
        str(settings["nonpad_collision_policy"]),
        "TABLE_AND_ARM_PATH_REQUIRE_CANDIDATE_WORLD_ROUTE",
    )
    accepted: list[ClosurePrediction] = []
    results: list[FastFilterResult] = []
    for prediction in predictions:
        reasons = _hard_reasons(inputs, prediction)
        if not reasons:
            duplicate = next(
                (
                    previous
                    for previous in accepted
                    if _same_realized_grasp(prediction, previous, thresholds)
                ),
                None,
            )
            if duplicate is not None:
                reasons.append(
                    f"NEAR_DUPLICATE_CONTACTS_OF_{duplicate.seed.candidate_id}"
                )
        status = "FAST_REJECT" if reasons else "FAST_SURVIVE"
        if status == "FAST_SURVIVE":
            accepted.append(prediction)
        results.append(
            FastFilterResult(
                candidate_id=prediction.seed.candidate_id,
                status=status,
                reasons=tuple(reasons),
                unresolved_checks=() if reasons else unresolved,
            )
        )
    return tuple(results)


__all__ = ["fast_filter_predictions"]
