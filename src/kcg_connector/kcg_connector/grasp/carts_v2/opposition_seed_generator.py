"""Deterministic FK anchors for one-finger-opposed-to-two grasp families."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import math
from typing import Mapping, Sequence

import numpy as np

from kcg_connector.grasp.carts_v2.full_palm_search import (
    fixed_pregrasp_phase_combinations,
)
from kcg_connector.grasp.carts_v2.models import (
    CandidateSeed,
    V2Inputs,
    joint_positions_for_phases,
)


_PALM_JOINT = "f1j1"
_AXIAL_FRACTIONS = (0.25, 0.50, 0.75)


def _unit(vector: np.ndarray, label: str) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(value))
    if value.shape != (3,) or not np.isfinite(length) or length <= 1.0e-12:
        raise ValueError(f"{label} cannot define a finite direction")
    return value / length


def _surface_center_local(surface) -> np.ndarray:
    triangles = np.asarray(surface.triangles_local_m, dtype=np.float64)
    areas = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0],
                 triangles[:, 2] - triangles[:, 0]), axis=1,
    )
    total = float(np.sum(areas))
    if len(triangles) == 0 or not np.isfinite(total) or total <= 0.0:
        raise ValueError(f"{surface.pad_name} has no finite task-surface area")
    return np.average(np.mean(triangles, axis=1), axis=0, weights=areas)


def extract_object_grasp_band(inputs: V2Inputs) -> Mapping[str, object]:
    """Describe the registered allowed band without object-specific coordinates."""

    mesh = inputs.object_contract.model.mesh
    face_ids = np.asarray(inputs.face_roles.allowed_face_indices, dtype=np.int64)
    if len(face_ids) == 0:
        raise ValueError("the object has no registered allowed grip band")
    areas = np.asarray(mesh.face_areas_m2[face_ids], dtype=np.float64)
    centroids = np.asarray(mesh.face_centroids_m[face_ids], dtype=np.float64)
    area = float(np.sum(areas))
    if not np.isfinite(area) or area <= 0.0:
        raise ValueError("the allowed grip band has invalid area")
    center = np.average(centroids, axis=0, weights=areas)
    axis = _unit(inputs.object_contract.model.assembly_axis, "object task axis")
    origin = np.asarray(
        inputs.object_contract.model.assembly_axis_origin_m, dtype=np.float64
    )
    vertices = np.asarray(mesh.face_vertices_m[face_ids], dtype=np.float64)
    axial = (vertices.reshape(-1, 3) - origin) @ axis
    axial_range = (float(np.min(axial)), float(np.max(axial)))
    center_axial = float((center - origin) @ axis)
    radial_offset = center - (origin + center_axial * axis)
    radial = centroids - origin - np.outer((centroids - origin) @ axis, axis)
    radius = float(np.average(np.linalg.norm(radial, axis=1), weights=areas))
    task_frame = np.asarray(
        inputs.object_contract.task_frame_rotation_object, dtype=np.float64
    )
    x_axis = task_frame[:, 0] - axis * float(task_frame[:, 0] @ axis)
    if np.linalg.norm(x_axis) <= 1.0e-12:
        basis = np.eye(3)[int(np.argmin(np.abs(axis)))]
        x_axis = basis - axis * float(basis @ axis)
    x_axis = _unit(x_axis, "object band reference axis")
    y_axis = _unit(np.cross(axis, x_axis), "object band transverse axis")
    x_axis = _unit(np.cross(y_axis, axis), "object band orthogonal axis")
    return {
        "center_object_m": center,
        "axis_object": axis,
        "reference_x_object": x_axis,
        "reference_y_object": y_axis,
        "axial_range_m": axial_range,
        "radial_offset_object_m": radial_offset,
        "mean_radius_m": radius,
        "allowed_face_count": int(len(face_ids)),
        "allowed_area_m2": area,
        "allowed_face_ids": face_ids,
        "allowed_face_centroids_m": centroids,
    }


def task_surface_triangle_geometry(
    inputs: V2Inputs,
    palm_configuration_rad: float,
    pregrasp_closure_phases: Sequence[float],
) -> Mapping[str, object]:
    """Return the real-FK triangle formed by the three task-surface centers."""

    if inputs.task_grip_surfaces is None:
        raise ValueError("opposition anchors require TASK_GRIP_SURFACE")
    phases = tuple(float(value) for value in pregrasp_closure_phases)
    if len(phases) != 3 or any(not math.isfinite(value) for value in phases):
        raise ValueError("one finite three-finger pregrasp is required")
    names = tuple(inputs.hand_model.independent_joint_names)
    if _PALM_JOINT not in names:
        raise ValueError("the registered palm-configuration joint is unavailable")
    reference = np.asarray([
        inputs.hand_model.joints[name].limit.lower for name in names
    ], dtype=np.float64)
    palm = float(palm_configuration_rad)
    reference[names.index(_PALM_JOINT)] = palm
    joints = joint_positions_for_phases(
        inputs, phases, reference_joint_positions_rad=reference
    )
    transforms = inputs.hand_model.forward_kinematics(joints)
    ordered = tuple(sorted(inputs.task_grip_surfaces.items()))
    if tuple(name for name, _surface in ordered) != (
        "finger_1_pad", "finger_2_pad", "finger_3_pad"
    ):
        raise ValueError("opposition finger identity changed")
    centers = []
    for _name, surface in ordered:
        local = _surface_center_local(surface)
        transform = np.asarray(transforms[surface.link_name], dtype=np.float64)
        centers.append(local @ transform[:3, :3].T + transform[:3, 3])
    centers_array = np.asarray(centers, dtype=np.float64)
    sides = np.asarray((
        np.linalg.norm(centers_array[1] - centers_array[2]),
        np.linalg.norm(centers_array[0] - centers_array[2]),
        np.linalg.norm(centers_array[0] - centers_array[1]),
    ))
    if np.any(sides <= 1.0e-9) or not np.all(np.isfinite(sides)):
        raise ValueError("task-surface centers do not form an opposition triangle")
    work_center = np.average(centers_array, axis=0, weights=sides)
    plane_normal = _unit(
        np.cross(centers_array[2] - centers_array[0],
                 centers_array[1] - centers_array[0]),
        "task-surface triangle normal",
    )
    if float(plane_normal @ work_center) < 0.0:
        plane_normal = -plane_normal
    opposition = 0.5 * (centers_array[0] + centers_array[2]) - centers_array[1]
    opposition -= plane_normal * float(opposition @ plane_normal)
    opposition = _unit(opposition, "one-against-two opposition axis")
    transverse = _unit(np.cross(plane_normal, opposition), "hand transverse axis")
    opposition = _unit(np.cross(transverse, plane_normal), "hand opposition axis")
    frame = np.column_stack((opposition, transverse, plane_normal))
    radii = np.linalg.norm(centers_array - work_center, axis=1)
    area = 0.5 * float(np.linalg.norm(np.cross(
        centers_array[1] - centers_array[0],
        centers_array[2] - centers_array[0],
    )))
    return {
        "palm_configuration_rad": palm,
        "pregrasp_closure_phases": phases,
        "joint_positions_rad": np.asarray(joints, dtype=np.float64),
        "task_surface_centers_handbase_m": centers_array,
        "work_center_handbase_m": work_center,
        "hand_frame_from_opposition": frame,
        "center_radii_m": radii,
        "side_lengths_m": sides,
        "triangle_area_m2": area,
        "triangle_quality": float(np.min(sides) / np.max(sides)),
    }


def _angle_token(angle_rad: float) -> str:
    microradians = int(round(float(angle_rad) * 1.0e6))
    return f"m{abs(microradians):07d}" if microradians < 0 else f"p{microradians:07d}"


def _anchor_record(
    inputs: V2Inputs,
    band: Mapping[str, object],
    geometry: Mapping[str, object],
    *,
    angle_index: int,
    azimuth_index: int,
    axial_index: int,
    axial_fraction: float,
    source_index: int,
) -> tuple[CandidateSeed, dict[str, object], tuple[float, ...]]:
    axis = np.asarray(band["axis_object"], dtype=np.float64)
    reference_x = np.asarray(band["reference_x_object"], dtype=np.float64)
    reference_y = np.asarray(band["reference_y_object"], dtype=np.float64)
    azimuth_count = int(band["azimuth_count"])
    theta = 2.0 * math.pi * azimuth_index / azimuth_count
    object_x = math.cos(theta) * reference_x + math.sin(theta) * reference_y
    object_y = _unit(np.cross(axis, object_x), "azimuth transverse axis")
    object_x = _unit(np.cross(object_y, axis), "azimuth opposition axis")
    object_frame = np.column_stack((object_x, object_y, axis))
    hand_frame = np.asarray(geometry["hand_frame_from_opposition"], dtype=np.float64)
    rotation = object_frame @ hand_frame.T
    axial_low, axial_high = band["axial_range_m"]
    axial_value = float(axial_low + axial_fraction * (axial_high - axial_low))
    origin = np.asarray(
        inputs.object_contract.model.assembly_axis_origin_m, dtype=np.float64
    )
    target = (origin + axial_value * axis
              + np.asarray(band["radial_offset_object_m"], dtype=np.float64))
    work_center = np.asarray(geometry["work_center_handbase_m"], dtype=np.float64)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = target - rotation @ work_center
    face_ids = np.asarray(band["allowed_face_ids"], dtype=np.int64)
    centroids = np.asarray(band["allowed_face_centroids_m"], dtype=np.float64)
    anchor_row = int(np.argmin(np.sum((centroids - target) ** 2, axis=1)))
    anchor_face = int(face_ids[anchor_row])
    anchor_position = np.asarray(centroids[anchor_row], dtype=np.float64)
    phases = tuple(float(value) for value in geometry["pregrasp_closure_phases"])
    palm = float(geometry["palm_configuration_rad"])
    identifier = (
        f"opposition_{_angle_token(palm)}_"
        f"a{azimuth_index:02d}_z{axial_index}"
    )
    seed = CandidateSeed(
        candidate_id=identifier,
        object_id=inputs.object_contract.object_id,
        anchor_face_index=anchor_face,
        anchor_position_object_m=tuple(float(value) for value in anchor_position),
        object_from_hand=tuple(float(value) for value in transform.ravel()),
        pregrasp_joint_positions_rad=tuple(
            float(value) for value in geometry["joint_positions_rad"]
        ),
        pregrasp_closure_phases=phases,
        source_sample_index=source_index,
        generator_score=0.0,
        descriptor_id=f"opposition_angle_{angle_index:03d}",
        approach_direction_object=tuple(float(value) for value in axis),
        maximum_closure_phase=float(inputs.config.section(
            "candidate_generation")["maximum_closure_phase"]),
        palm_configuration_rad=palm,
    )
    radii = np.asarray(geometry["center_radii_m"], dtype=np.float64)
    radius_mismatch = float(np.max(np.abs(radii - float(band["mean_radius_m"]))))
    radius_imbalance = float(np.ptp(radii))
    key = (
        radius_mismatch,
        radius_imbalance,
        -float(geometry["triangle_quality"]),
        *phases,
        float(azimuth_index),
        float(axial_index),
    )
    evidence = {
        "candidate_id": identifier,
        "palm_configuration_rad": palm,
        "pregrasp_closure_phases": list(phases),
        "candidate_id_excludes_rebindable_pregrasp": True,
        "azimuth_index": azimuth_index,
        "azimuth_rad": theta,
        "axial_index": axial_index,
        "axial_fraction": axial_fraction,
        "target_band_center_object_m": target.tolist(),
        "handbase_object_m": transform[:3, 3].tolist(),
        "triangle_side_lengths_m": np.asarray(
            geometry["side_lengths_m"]).tolist(),
        "triangle_center_radii_m": radii.tolist(),
        "triangle_quality": float(geometry["triangle_quality"]),
        "radius_mismatch_m": radius_mismatch,
        "radius_imbalance_m": radius_imbalance,
        "work_center_alignment_error_m": float(np.linalg.norm(
            rotation @ work_center + transform[:3, 3] - target
        )),
        "selection_key": list(key),
    }
    return seed, evidence, key


def generate_opposition_anchors(
    inputs: V2Inputs,
    palm_angles_rad: Sequence[float],
    *,
    azimuth_count: int = 12,
    axial_fractions: Sequence[float] = _AXIAL_FRACTIONS,
    maximum_per_angle: int = 12,
) -> tuple[tuple[CandidateSeed, ...], dict[str, object]]:
    """Generate bounded opposition anchors and retain at most 12 per palm angle."""

    angles = tuple(float(value) for value in palm_angles_rad)
    fractions = tuple(float(value) for value in axial_fractions)
    if (not angles or any(not math.isfinite(value) for value in angles)
            or azimuth_count != 12 or fractions != _AXIAL_FRACTIONS
            or not 1 <= int(maximum_per_angle) <= 12):
        raise ValueError("opposition anchor design must use 12 azimuths and 3 axial levels")
    band = dict(extract_object_grasp_band(inputs))
    band["azimuth_count"] = azimuth_count
    preshapes = fixed_pregrasp_phase_combinations()
    selected_seeds, selected_evidence, per_angle = [], [], []
    source_index = 0
    for angle_index, angle in enumerate(angles):
        geometries = [
            task_surface_triangle_geometry(inputs, angle, phases)
            for phases in preshapes
        ]
        rows = []
        for geometry in geometries:
            for azimuth_index in range(azimuth_count):
                for axial_index, fraction in enumerate(fractions):
                    seed, evidence, key = _anchor_record(
                        inputs, band, geometry, angle_index=angle_index,
                        azimuth_index=azimuth_index, axial_index=axial_index,
                        axial_fraction=fraction, source_index=source_index,
                    )
                    rows.append((key, seed.candidate_id, seed, evidence))
                    source_index += 1
        best_by_azimuth = []
        for azimuth_index in range(azimuth_count):
            axial_index = azimuth_index % len(fractions)
            group = [row for row in rows
                     if row[3]["azimuth_index"] == azimuth_index
                     and row[3]["axial_index"] == axial_index]
            best_by_azimuth.append(min(group, key=lambda row: (row[0], row[1])))
        retained = sorted(best_by_azimuth, key=lambda row: row[3]["azimuth_index"])
        retained = retained[:maximum_per_angle]
        selected_seeds.extend(row[2] for row in retained)
        selected_evidence.extend(row[3] for row in retained)
        per_angle.append({
            "palm_configuration_rad": angle,
            "palm_configuration_deg": math.degrees(angle),
            "preshape_count": len(preshapes),
            "raw_anchor_count": len(rows),
            "retained_anchor_count": len(retained),
            "retained_candidate_ids": [row[2].candidate_id for row in retained],
            "retained_axial_counts": {
                str(index): sum(row[3]["axial_index"] == index for row in retained)
                for index in range(len(fractions))
            },
            "evaluated_pregrasp_closure_phases": [list(row) for row in preshapes],
        })
    audit = {
        "schema_version": "carts_opposition_anchor_v1",
        "claim_scope": "STRUCTURED_FK_ANCHORS_NOT_CONTACT_OR_DYNAMIC_SUCCESS",
        "object_id": inputs.object_contract.object_id,
        "hand_variant": inputs.hand_variant,
        "object_grasp_band": {
            "center_object_m": np.asarray(band["center_object_m"]).tolist(),
            "axis_object": np.asarray(band["axis_object"]).tolist(),
            "axial_range_m": list(band["axial_range_m"]),
            "mean_radius_m": band["mean_radius_m"],
            "allowed_face_count": band["allowed_face_count"],
            "allowed_area_m2": band["allowed_area_m2"],
        },
        "azimuth_count": azimuth_count,
        "axial_fractions": list(fractions),
        "preshape_count": len(preshapes),
        "raw_anchor_count": len(angles) * len(preshapes) * azimuth_count * len(fractions),
        "retained_anchor_count": len(selected_seeds),
        "per_angle": per_angle,
        "selected": selected_evidence,
    }
    return tuple(selected_seeds), audit


def generate_feature_opposition_grid(
    inputs: V2Inputs,
) -> tuple[tuple[CandidateSeed, ...], dict[str, object]]:
    """Enumerate the preregistered 7x72x3x27 cheap-search design."""

    angles = tuple(math.radians(value) for value in range(45, 80, 5))
    band = dict(extract_object_grasp_band(inputs))
    band["azimuth_count"] = 72
    phases_grid = fixed_pregrasp_phase_combinations()
    seeds: list[CandidateSeed] = []
    identity = hashlib.sha256()
    source_index = 0
    for angle_index, angle in enumerate(angles):
        geometries = tuple(
            task_surface_triangle_geometry(inputs, angle, phases)
            for phases in phases_grid
        )
        for geometry in geometries:
            phase_token = "".join(
                str(int(round(value * 10.0)))
                for value in geometry["pregrasp_closure_phases"]
            )
            for azimuth_index in range(72):
                for axial_index, fraction in enumerate(_AXIAL_FRACTIONS):
                    seed, _evidence, _key = _anchor_record(
                        inputs, band, geometry, angle_index=angle_index,
                        azimuth_index=azimuth_index, axial_index=axial_index,
                        axial_fraction=fraction, source_index=source_index,
                    )
                    identifier = f"{seed.candidate_id}__p{phase_token}"
                    seed = replace(seed, candidate_id=identifier)
                    seeds.append(seed)
                    identity.update(identifier.encode("utf-8") + b"\0")
                    source_index += 1
    expected = len(angles) * 72 * len(_AXIAL_FRACTIONS) * len(phases_grid)
    if len(seeds) != expected or len({seed.candidate_id for seed in seeds}) != expected:
        raise RuntimeError("feature opposition design is incomplete or non-unique")
    audit = {
        "schema_version": "carts_feature_opposition_grid_v1",
        "claim_scope": "FULL_CHEAP_DESIGN_NOT_EXACT_CONTACT_OR_DYNAMIC_SUCCESS",
        "object_id": inputs.object_contract.object_id,
        "palm_configuration_deg": list(range(45, 80, 5)),
        "azimuth_count": 72,
        "azimuth_step_deg": 5,
        "axial_fractions": list(_AXIAL_FRACTIONS),
        "pregrasp_phase_values": [0.0, 0.1, 0.2],
        "pregrasp_combination_count": len(phases_grid),
        "candidate_count": len(seeds),
        "candidate_id_order_sha256": identity.hexdigest(),
    }
    return tuple(seeds), audit


__all__ = [
    "extract_object_grasp_band",
    "generate_feature_opposition_grid",
    "generate_opposition_anchors",
    "task_surface_triangle_geometry",
]
