"""Cheap hard rejection for all V2 candidates; never a safety certificate."""

from __future__ import annotations

import math

import numpy as np

from kcg_connector.grasp.carts_v2.models import (
    ClosurePrediction,
    FastFilterResult,
    V2Inputs,
    joint_positions_for_phases,
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


def _sampled_hand_states(
    inputs: V2Inputs, prediction: ClosurePrediction
) -> tuple[tuple[str, np.ndarray, np.ndarray], ...]:
    settings = inputs.config.section("fast_filter")
    dynamic = inputs.config.section("dynamic")
    base = inputs.frozen_world_from_object @ prediction.seed.object_from_hand_matrix()
    pregrasp = np.asarray(prediction.seed.pregrasp_joint_positions_rad)
    height = float(dynamic["approach_clearance_height_m"])
    sample_count = int(settings["approach_path_sample_count"])
    states: list[tuple[str, np.ndarray, np.ndarray]] = []
    for index, fraction in enumerate(np.linspace(1.0, 0.0, sample_count)):
        shifted = np.array(base, copy=True)
        shifted[2, 3] += height * float(fraction)
        stage = "PREGRASP" if index == sample_count - 1 else f"APPROACH_{index:02d}"
        states.append((stage, shifted, pregrasp))
    phases = list(prediction.seed.pregrasp_closure_phases)
    phase_by_pad = {
        pad.name: index for index, pad in enumerate(inputs.hand_contract.pads)
    }
    for stop_index, pad_name in enumerate(
        inputs.config.section("closure_prediction")["closing_order"], start=1
    ):
        phase_index = phase_by_pad[str(pad_name)]
        phases[phase_index] = prediction.final_closure_phases[phase_index]
        joints = joint_positions_for_phases(inputs, tuple(phases))
        states.append((f"CONTACT_STOP_{stop_index}", base, joints))
    return tuple(states)


def _state_table_clearance(
    inputs: V2Inputs, base: np.ndarray, joints: np.ndarray
) -> tuple[float | None, str]:
    transforms = inputs.hand_model.forward_kinematics(
        joints, base_transform=base
    )
    bounds = inputs.table_xy_bounds_m
    minimum: tuple[float, str] | None = None
    for link_name, triangles in inputs.hand_collision_triangles_by_link.items():
        transform = transforms[link_name]
        world = triangles @ transform[:3, :3].T + transform[:3, 3]
        triangle_min = np.min(world[:, :, :2], axis=1)
        triangle_max = np.max(world[:, :, :2], axis=1)
        overlaps = (
            (triangle_max[:, 0] >= bounds[0, 0])
            & (triangle_min[:, 0] <= bounds[0, 1])
            & (triangle_max[:, 1] >= bounds[1, 0])
            & (triangle_min[:, 1] <= bounds[1, 1])
        )
        if not np.any(overlaps):
            continue
        gap = float(np.min(world[overlaps, :, 2]) - inputs.table_top_z_m)
        if minimum is None or gap < minimum[0]:
            minimum = (gap, link_name)
    return (None, "") if minimum is None else minimum


def _sampled_hand_table_clearance(
    inputs: V2Inputs, prediction: ClosurePrediction
) -> tuple[float | None, str, str]:
    minimum: tuple[float, str, str] | None = None
    for stage, base, joints in _sampled_hand_states(inputs, prediction):
        gap, link_name = _state_table_clearance(inputs, base, joints)
        if gap is not None and (minimum is None or gap < minimum[0]):
            minimum = (gap, link_name, stage)
    return (None, "", "") if minimum is None else minimum


def fast_filter_predictions(
    inputs: V2Inputs, predictions: tuple[ClosurePrediction, ...]
) -> tuple[FastFilterResult, ...]:
    """Return FAST_REJECT or FAST_SURVIVE without promoting unresolved checks."""

    settings = inputs.config.section("fast_filter")
    thresholds = inputs.config.section("candidate_generation")["deduplication"]
    unresolved = (
        str(settings["arm_ik_policy"]),
        str(settings["nonpad_collision_policy"]),
        "HAND_TABLE_SAMPLED_NOT_CONTINUOUS",
        "ARM_LINK_AND_JOINT_INTERPOLATED_PATH_NOT_FAST_CHECKED",
    )
    accepted: list[ClosurePrediction] = []
    results: list[FastFilterResult] = []
    for prediction in predictions:
        reasons = _hard_reasons(inputs, prediction)
        clearance: float | None = None
        clearance_link = ""
        clearance_stage = ""
        if not reasons:
            clearance, clearance_link, clearance_stage = (
                _sampled_hand_table_clearance(inputs, prediction)
            )
            tolerance = float(settings["table_penetration_tolerance_m"])
            if clearance is not None and clearance < -tolerance:
                reasons.append("HAND_TABLE_PENETRATION")
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
                sampled_hand_table_clearance_m=clearance,
                sampled_hand_table_clearance_link=clearance_link,
                sampled_hand_table_clearance_stage=clearance_stage,
            )
        )
    return tuple(results)


__all__ = ["fast_filter_predictions"]
