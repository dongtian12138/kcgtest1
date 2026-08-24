"""Generate dispersed, geometry-conditioned V2 grasp seeds and nothing else."""

from __future__ import annotations

import math

import numpy as np

from kcg_connector.grasp.carts_v2.models import (
    CandidateSeed,
    ClosurePrediction,
    FastFilterResult,
    V2Inputs,
    farthest_point_indices,
    joint_positions_for_phases,
    rotation_distance,
)
from kcg_connector.grasp.robust.surface_sampling import (
    RegisteredTaskFrame,
    sample_mesh_faces_area_stratified,
)


def _rotation_about_z(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )


def _native_contact_reference(inputs: V2Inputs, phase: float) -> np.ndarray:
    joints = joint_positions_for_phases(inputs, (phase, phase, phase))
    transforms = inputs.hand_model.pad_transforms(joints)
    inward_points: list[np.ndarray] = []
    for pad in inputs.hand_contract.pads:
        transform = transforms[pad.name]
        points = pad.points_local_m @ transform[:3, :3].T + transform[:3, 3]
        inward_points.append(points[int(np.argmin(np.linalg.norm(points[:, :2], axis=1)))])
    return np.mean(np.asarray(inward_points), axis=0)


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


def _table_height_conditioned_angular_order(
    inputs: V2Inputs,
    task_positions: np.ndarray,
    sample_positions_object: np.ndarray,
    fallback_order: np.ndarray,
    bin_count: int,
) -> np.ndarray:
    world_positions = (
        sample_positions_object @ inputs.frozen_world_from_object[:3, :3].T
        + inputs.frozen_world_from_object[:3, 3]
    )
    physical_heights = world_positions[:, 2] - inputs.table_top_z_m
    angles = np.mod(np.arctan2(task_positions[:, 1], task_positions[:, 0]), 2.0 * np.pi)
    bins = np.minimum(
        (angles / (2.0 * np.pi) * bin_count).astype(np.int64), bin_count - 1
    )
    primary: list[int] = []
    for bin_index in range(bin_count):
        members = np.flatnonzero(bins == bin_index)
        if len(members):
            ranked = np.lexsort((members, -physical_heights[members]))
            primary.append(int(members[ranked[0]]))
    used = set(primary)
    primary.extend(int(index) for index in fallback_order if int(index) not in used)
    return np.asarray(primary, dtype=np.int64)


def generate_raw_candidates(inputs: V2Inputs) -> tuple[CandidateSeed, ...]:
    """Generate the fixed raw surface pool before closure or path filtering."""

    settings = inputs.config.section("candidate_generation")
    loaded = inputs.object_contract
    task_frame = RegisteredTaskFrame(
        origin_object_m=loaded.model.assembly_axis_origin_m,
        basis_object=loaded.task_frame_rotation_object,
        source=loaded.task_frame_source,
    )
    samples = sample_mesh_faces_area_stratified(
        loaded.model,
        task_frame=task_frame,
        face_indices=inputs.face_roles.allowed_face_indices,
        sample_count=int(settings["surface_pool_count"]),
        seed=int(settings["random_seed"]),
    )
    task_positions = (
        samples.positions_m - task_frame.origin_object_m
    ) @ task_frame.basis_object
    task_normals = samples.normals @ task_frame.basis_object
    scale = max(loaded.characteristic_radius_m, np.finfo(np.float64).eps)
    features = np.column_stack((task_positions / scale, task_normals))
    fps_order = farthest_point_indices(features)
    requested = int(settings["candidate_count"])
    order = _table_height_conditioned_angular_order(
        inputs,
        task_positions,
        samples.positions_m,
        fps_order,
        requested,
    )

    reference_phase = float(settings["reference_closure_phase"])
    reference = _native_contact_reference(inputs, reference_phase)
    hand_reference_angle = math.atan2(float(reference[1]), float(reference[0]))
    pregrasp_phase = float(settings["pregrasp_closure_phase"])
    pregrasp_phases = (pregrasp_phase, pregrasp_phase, pregrasp_phase)
    pregrasp_joints = joint_positions_for_phases(inputs, pregrasp_phases)
    candidates: list[CandidateSeed] = []
    for sample_index in order:
        task_point = task_positions[sample_index]
        anchor_angle = math.atan2(float(task_point[1]), float(task_point[0]))
        hand_rotation = task_frame.basis_object @ _rotation_about_z(
            anchor_angle - hand_reference_angle
        )
        target_axis_point = (
            task_frame.origin_object_m
            + task_frame.basis_object[:, 2] * float(task_point[2])
        )
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = hand_rotation
        transform[:3, 3] = target_axis_point - hand_rotation @ reference
        seed = CandidateSeed(
            candidate_id=f"raw_seed_{int(sample_index):03d}",
            object_id=loaded.object_id,
            anchor_face_index=int(samples.face_indices[sample_index]),
            anchor_position_object_m=tuple(
                float(value) for value in samples.positions_m[sample_index]
            ),
            object_from_hand=tuple(float(value) for value in transform.ravel()),
            pregrasp_joint_positions_rad=tuple(float(value) for value in pregrasp_joints),
            pregrasp_closure_phases=pregrasp_phases,
            source_sample_index=int(sample_index),
        )
        candidates.append(seed)
    if len(candidates) != int(settings["surface_pool_count"]):
        raise RuntimeError("raw surface seed pool is incomplete")
    return tuple(candidates)


def select_diverse_predictions(
    inputs: V2Inputs,
    predictions: tuple[ClosurePrediction, ...],
    filters: tuple[FastFilterResult, ...],
) -> tuple[tuple[ClosurePrediction, ...], dict[str, str]]:
    """Keep at most the formal budget after closure and full-sweep rejection."""

    filter_by_id = {row.candidate_id: row for row in filters}
    thresholds = inputs.config.section("candidate_generation")["deduplication"]
    limit = int(inputs.config.section("candidate_generation")["candidate_count"])
    accepted: list[ClosurePrediction] = []
    rejected: dict[str, str] = {}
    for prediction in predictions:
        result = filter_by_id[prediction.seed.candidate_id]
        if (
            prediction.status != "CLOSURE_SURVIVE"
            or result.status != "FAST_SURVIVE"
            or not result.sequential_closure_sweep_pass
        ):
            continue
        duplicate = next(
            (
                row
                for row in accepted
                if _same_realized_grasp(prediction, row, thresholds)
            ),
            None,
        )
        if duplicate is not None:
            rejected[prediction.seed.candidate_id] = (
                f"NEAR_DUPLICATE_OF_{duplicate.seed.candidate_id}"
            )
            continue
        if len(accepted) == limit:
            rejected[prediction.seed.candidate_id] = "FORMAL_CANDIDATE_BUDGET_EXCEEDED"
            continue
        accepted.append(prediction)
    return tuple(accepted), rejected


__all__ = ["generate_raw_candidates", "select_diverse_predictions"]
