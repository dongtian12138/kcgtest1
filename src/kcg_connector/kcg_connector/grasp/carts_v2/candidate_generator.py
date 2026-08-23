"""Generate dispersed, geometry-conditioned V2 grasp seeds and nothing else."""

from __future__ import annotations

import math

import numpy as np

from kcg_connector.grasp.carts_v2.models import (
    CandidateSeed,
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


def _is_duplicate(
    candidate: CandidateSeed,
    accepted: list[CandidateSeed],
    settings: dict[str, float],
) -> bool:
    matrix = candidate.object_from_hand_matrix()
    anchor = np.asarray(candidate.anchor_position_object_m)
    for previous in accepted:
        other = previous.object_from_hand_matrix()
        if (
            np.linalg.norm(matrix[:3, 3] - other[:3, 3])
            <= settings["palm_position_m"]
            and rotation_distance(matrix[:3, :3], other[:3, :3])
            <= settings["palm_orientation_rad"]
            and np.linalg.norm(anchor - np.asarray(previous.anchor_position_object_m))
            <= settings["anchor_position_m"]
        ):
            return True
    return False


def generate_candidates(inputs: V2Inputs) -> tuple[CandidateSeed, ...]:
    """Generate 32--64 seeds from the object's V2-allowed real mesh faces."""

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
    order = farthest_point_indices(features)

    reference_phase = float(settings["reference_closure_phase"])
    reference = _native_contact_reference(inputs, reference_phase)
    hand_reference_angle = math.atan2(float(reference[1]), float(reference[0]))
    pregrasp_phase = float(settings["pregrasp_closure_phase"])
    pregrasp_phases = (pregrasp_phase, pregrasp_phase, pregrasp_phase)
    pregrasp_joints = joint_positions_for_phases(inputs, pregrasp_phases)
    duplicate_settings = {
        key: float(value) for key, value in settings["deduplication"].items()
    }

    accepted: list[CandidateSeed] = []
    requested = int(settings["candidate_count"])
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
            candidate_id=f"candidate_{len(accepted):02d}",
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
        if not _is_duplicate(seed, accepted, duplicate_settings):
            accepted.append(seed)
        if len(accepted) == requested:
            break
    if len(accepted) < requested:
        raise RuntimeError(
            f"only {len(accepted)} distinct candidates remain after V2 deduplication"
        )
    return tuple(accepted)


__all__ = ["generate_candidates"]
