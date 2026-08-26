from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from kcg_connector.grasp.carts_v2.models import (
    V2Inputs, farthest_point_indices, joint_positions_for_phases,
)
from kcg_connector.grasp.carts_v2.task_grip_surface import task_noncontact_triangles


_QP_ENDPOINT_LABEL_RAD, _QP_ENDPOINT_ROUNDING_TOLERANCE_RAD = math.pi / 2.0, math.radians(0.1)

def _unit(value: np.ndarray, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if vector.shape != (3,) or not np.isfinite(length) or length <= 1.0e-12:
        raise ValueError(f"{label} has no finite direction")
    return vector / length


def _surface_reference(surface) -> tuple[np.ndarray, np.ndarray]:
    triangles = np.asarray(surface.triangles_local_m, dtype=np.float64)
    areas = 0.5 * np.linalg.norm(np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    total = float(np.sum(areas))
    if len(triangles) == 0 or not np.isfinite(total) or total <= 0.0:
        raise ValueError(f"TASK_SURFACE_AREA_INVALID:{surface.pad_name}")
    center = np.average(np.mean(triangles, axis=1), axis=0, weights=areas)
    normal = _unit(np.sum(surface.face_normals_local * areas[:, None], axis=0),
                   f"TASK surface normal {surface.pad_name}")
    return center, normal


def _surface_samples(surface, count=64):
    triangles = np.asarray(surface.triangles_local_m, dtype=np.float64)
    centers = np.mean(triangles, axis=1)
    span = np.ptp(centers, axis=0)
    scaled = (centers - np.min(centers, axis=0)) / np.where(span > 0.0, span, 1.0)
    features = np.column_stack((scaled, np.asarray(surface.face_normals_local)))
    selected = farthest_point_indices(features, min(int(count), len(features)))
    return centers[selected], np.asarray(surface.face_normals_local)[selected], selected


def resolve_palm_configuration_rad(inputs: V2Inputs,
                                   requested_rad: float) -> float:
    names = tuple(inputs.hand_model.independent_joint_names)
    if "f1j1" not in names:
        raise ValueError("PALM_CONFIGURATION_JOINT_MISSING")
    lower, upper = inputs.hand_model.joint_limit_vectors()
    index = names.index("f1j1")
    requested = float(requested_rad)
    if lower[index] <= requested <= upper[index]:
        return requested
    exact_upper_label = math.isclose(
        requested, _QP_ENDPOINT_LABEL_RAD, rel_tol=0.0, abs_tol=1.0e-12)
    rounded_upper = (upper[index] <= requested and requested - upper[index] <=
                     _QP_ENDPOINT_ROUNDING_TOLERANCE_RAD)
    if exact_upper_label and rounded_upper:
        return float(upper[index])
    raise ValueError("PALM_CONFIGURATION_OUTSIDE_LIMITS")


def _reference_joint_vector(inputs: V2Inputs, palm_configuration_rad: float) -> np.ndarray:
    names = tuple(inputs.hand_model.independent_joint_names)
    lower, _upper = inputs.hand_model.joint_limit_vectors()
    vector = np.array(lower, copy=True)
    palm = resolve_palm_configuration_rad(inputs, palm_configuration_rad)
    vector[names.index("f1j1")] = palm
    return vector


def contact_coordinates(points: np.ndarray, approach: np.ndarray):
    center = np.mean(points, axis=0)
    axis = _unit(approach, "hand approach direction")
    first = points[0] - center
    first -= axis * float(first @ axis)
    x_axis = _unit(first, "finger-1 angular reference")
    y_axis = _unit(np.cross(axis, x_axis), "hand angular transverse axis")
    relative = points - center
    alpha = tuple(float(math.atan2(row @ y_axis, row @ x_axis) %
                        (2.0 * math.pi)) for row in relative)
    offsets = tuple(float(row @ axis) for row in relative)
    return alpha, offsets


def _finger_reference(inputs, surface, finger_index, preshape, reference,
                      work_center, approach, target_radius, sample_count,
                      non_task_triangles):
    maximum = float(inputs.config.section("candidate_generation")["maximum_closure_phase"])
    start = float(preshape[finger_index])
    if not 0.0 <= start < maximum <= 1.0:
        raise ValueError(f"INVALID_PRESHAPE_RANGE:{surface.pad_name}")
    local_points, local_normals, face_indices = _surface_samples(surface)
    non_task = np.unique(np.asarray(non_task_triangles).reshape(-1, 3), axis=0)
    non_task = non_task[ConvexHull(non_task).vertices]
    phases_grid = np.linspace(start, maximum, int(sample_count))
    derivative = float(inputs.config.section("closure_prediction")[
        "motion_derivative_phase_step"])
    minimum_motion = float(inputs.config.section("closure_prediction")[
        "minimum_inward_motion_m_per_phase"])
    rows = []
    for phase in phases_grid:
        phases = list(preshape)
        phases[finger_index] = float(phase)
        joints = joint_positions_for_phases(
            inputs, tuple(phases), reference_joint_positions_rad=reference)
        transform = np.asarray(inputs.hand_model.forward_kinematics(joints)[surface.link_name])
        points = local_points @ transform[:3, :3].T + transform[:3, 3]
        normals = local_normals @ transform[:3, :3].T
        moved_phase = min(maximum, float(phase) + derivative)
        if moved_phase == phase:
            moved_phase = max(start, float(phase) - derivative)
        moved = list(phases)
        moved[finger_index] = moved_phase
        moved_joints = joint_positions_for_phases(
            inputs, tuple(moved), reference_joint_positions_rad=reference)
        moved_transform = np.asarray(inputs.hand_model.forward_kinematics(
            moved_joints)[surface.link_name])
        moved_points = local_points @ moved_transform[:3, :3].T + moved_transform[:3, 3]
        delta = moved_phase - float(phase)
        motion = ((moved_points - points) / delta if delta != 0.0
                  else np.zeros_like(points))
        inward = np.asarray(work_center) - points
        inward -= np.outer(inward @ approach, approach)
        tolerance = 64.0 * np.finfo(np.float64).eps
        compatible = ((np.einsum("ij,ij->i", normals, inward) > tolerance)
                      & (np.einsum("ij,ij->i", motion, inward) > tolerance)
                      & (np.einsum("ij,ij->i", normals, motion) >= minimum_motion))
        radius_error = np.abs(np.linalg.norm(inward, axis=1) - target_radius)
        contact_distance = float(inputs.config.section("closure_prediction")[
            "contact_distance_m"])
        non_task_hand = non_task @ transform[:3, :3].T + transform[:3, 3]
        overhang = np.max(non_task_hand @ normals.T, axis=0) - np.einsum(
            "ij,ij->i", points, normals)
        scale = max(1.0, float(np.max(np.abs(non_task_hand))),
                    float(np.max(np.abs(points))))
        support_tolerance = 64.0 * np.finfo(np.float64).eps * scale
        compatible &= overhang <= support_tolerance
        for index in range(len(points)):
            rows.append((not bool(compatible[index]),
                         max(0.0, float(radius_error[index]) - contact_distance),
                         float(overhang[index]), float(radius_error[index]), float(phase),
                         int(face_indices[index]), joints, points[index], normals[index],
                         motion[index]))
    selected = min(rows, key=lambda row: row[:6])
    if selected[0]:
        raise ValueError(f"NO_MOTION_COMPATIBLE_HAND_REFERENCE:{surface.pad_name}")
    return selected


def hand_contact_references(
    inputs: V2Inputs, palm_configuration_rad: float,
    preshape_closure_phases: Sequence[float], target_radius_m: float,
    *, coarse_sample_count: int = 10,
) -> Mapping[str, object]:
    """Choose one real TASK-surface reference per finger by bounded FK search."""
    if inputs.task_grip_surfaces is None:
        raise ValueError("TASK_GRIP_SURFACE_REQUIRED")
    preshape = tuple(float(value) for value in preshape_closure_phases)
    radius = float(target_radius_m)
    if (len(preshape) != 3 or any(not 0.0 <= value <= 1.0 for value in preshape)
            or not np.isfinite(radius) or radius <= 0.0
            or not 8 <= int(coarse_sample_count) <= 12):
        raise ValueError("HAND_REFERENCE_INPUT_INVALID")
    ordered = tuple(sorted(inputs.task_grip_surfaces.items()))
    non_task = task_noncontact_triangles(
        inputs.hand_collision_triangles_by_link, inputs.task_grip_surfaces)
    if tuple(name for name, _surface in ordered) != (
            "finger_1_pad", "finger_2_pad", "finger_3_pad"):
        raise ValueError("FINGER_IDENTITY_CHANGED")
    reference = _reference_joint_vector(inputs, palm_configuration_rad)
    effective_palm = float(reference[
        tuple(inputs.hand_model.independent_joint_names).index("f1j1")])
    pregrasp = joint_positions_for_phases(
        inputs, preshape, reference_joint_positions_rad=reference)
    transforms = inputs.hand_model.forward_kinematics(pregrasp)
    base_centers = []
    for _name, surface in ordered:
        local, _normal = _surface_reference(surface)
        transform = transforms[surface.link_name]
        base_centers.append(local @ transform[:3, :3].T + transform[:3, 3])
    work_center = np.mean(base_centers, axis=0)
    approach = _unit(work_center, "pregrasp workspace approach direction")
    selected = [_finger_reference(
        inputs, surface, index, preshape, reference, work_center, approach, radius,
        coarse_sample_count, non_task[surface.link_name])
        for index, (_name, surface) in enumerate(ordered)]
    points = np.asarray([row[7] for row in selected])
    normals = np.asarray([row[8] for row in selected])
    alpha, axial_offsets = contact_coordinates(points, approach)
    return {
        "points_handbase_m": points, "normals_handbase": normals,
        "reference_closure_phases": tuple(float(row[4]) for row in selected),
        "reference_joint_positions_rad": tuple(
            tuple(float(value) for value in row[6]) for row in selected),
        "pregrasp_joint_positions_rad": tuple(float(value) for value in pregrasp),
        "effective_palm_configuration_rad": effective_palm,
        "relative_azimuths_rad": alpha,
        "relative_axial_offsets_m": axial_offsets,
        "approach_direction_handbase": approach,
        "radius_errors_m": tuple(float(row[3]) for row in selected),
        "terminal_overhang_m": tuple(float(row[2]) for row in selected),
        "terminal_overhang_definition": "NON_TASK_CONVEX_SUPPORT_ALONG_TASK_FACE_NORMAL",
        "task_surface_face_indices": tuple(int(row[5]) for row in selected),
    }


def kabsch_rigid_alignment(source_points_m: np.ndarray,
                           target_points_m: np.ndarray) -> np.ndarray:
    """Return the proper rigid transform that least-squares aligns three points."""

    source = np.asarray(source_points_m, dtype=np.float64)
    target = np.asarray(target_points_m, dtype=np.float64)
    if source.shape != (3, 3) or target.shape != (3, 3) or not (
            np.all(np.isfinite(source)) and np.all(np.isfinite(target))):
        raise ValueError("THREE_POINT_ALIGNMENT_INPUT_INVALID")
    left, right = source - np.mean(source, axis=0), target - np.mean(target, axis=0)
    _u0, singular, _v0 = np.linalg.svd(left, full_matrices=False)
    if singular[1] <= 128.0 * np.finfo(np.float64).eps * max(1.0, singular[0]):
        raise ValueError("THREE_POINT_ALIGNMENT_DEGENERATE")
    u, _values, vt = np.linalg.svd(left.T @ right)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.mean(target, axis=0) - rotation @ np.mean(source, axis=0)
    return transform


def _alignment_metrics(transform, hand_points, hand_normals, object_points, object_normals):
    rotation, translation = transform[:3, :3], transform[:3, 3]
    gaps = np.linalg.norm(hand_points @ rotation.T + translation - object_points, axis=1)
    cosines = np.clip(np.einsum("ij,ij->i", hand_normals @ rotation.T,
                                -object_normals), -1.0, 1.0)
    return gaps, np.arccos(cosines)


def initialize_three_contact_pose(
    hand_points_m: np.ndarray, hand_normals: np.ndarray,
    object_points_m: np.ndarray, object_outward_normals: np.ndarray,
) -> Mapping[str, object]:
    """Kabsch point fit followed by unit-separated least-squares normal tie-break."""

    hp, hn, op, on = (np.asarray(value, dtype=np.float64) for value in (
        hand_points_m, hand_normals, object_points_m, object_outward_normals))
    if any(value.shape != (3, 3) or not np.all(np.isfinite(value))
           for value in (hp, hn, op, on)):
        raise ValueError("POINT_NORMAL_ALIGNMENT_INPUT_INVALID")
    hn = np.asarray([_unit(row, "hand contact normal") for row in hn])
    on = np.asarray([_unit(row, "object contact normal") for row in on])
    initial = kabsch_rigid_alignment(hp, op)

    def normal_residual(delta):
        rotation = Rotation.from_rotvec(delta).as_matrix() @ initial[:3, :3]
        return (hn @ rotation.T + on).ravel()

    refined = least_squares(normal_residual, np.zeros(3), max_nfev=12,
                            ftol=1.0e-10, xtol=1.0e-10, gtol=1.0e-10)
    candidate = np.array(initial, copy=True)
    candidate[:3, :3] = Rotation.from_rotvec(refined.x).as_matrix() @ initial[:3, :3]
    candidate[:3, 3] = np.mean(op, axis=0) - candidate[:3, :3] @ np.mean(hp, axis=0)
    initial_gap, initial_angle = _alignment_metrics(initial, hp, hn, op, on)
    gap, angle = _alignment_metrics(candidate, hp, hn, op, on)
    scale = max(1.0, float(np.max(np.abs(hp))), float(np.max(np.abs(op))))
    cap = float(np.max(initial_gap)) + 128.0 * np.finfo(np.float64).eps * scale
    use_refined = bool(np.max(gap) <= cap and np.max(angle) < np.max(initial_angle))
    transform = candidate if use_refined else initial
    gap, angle = _alignment_metrics(transform, hp, hn, op, on)
    if np.any(np.cos(angle) <= 0.0):
        raise ValueError("POINT_NORMAL_ALIGNMENT_OPPOSED_NORMAL_FAILED")
    return {
        "object_from_hand": transform,
        "maximum_point_residual_m": float(np.max(gap)),
        "rms_point_residual_m": float(np.sqrt(np.mean(gap * gap))),
        "maximum_normal_residual_rad": float(np.max(angle)),
        "normal_tiebreak_accepted": use_refined,
        "method": "KABSCH_POINTS_THEN_CONSTRAINED_NORMAL_LEAST_SQUARES",
    }


__all__ = [
    "contact_coordinates", "hand_contact_references", "initialize_three_contact_pose",
    "kabsch_rigid_alignment", "resolve_palm_configuration_rad",
]
