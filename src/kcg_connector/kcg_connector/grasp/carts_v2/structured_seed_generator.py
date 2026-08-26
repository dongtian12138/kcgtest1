from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Mapping, Sequence

import numpy as np

from kcg_connector.grasp.carts_v2.b0_surface_semantics import B0_SURFACE_METHOD
from kcg_connector.grasp.carts_v2.models import (
    CandidateSeed, HARD_FORBIDDEN, PRIMARY_GRIP, V2Inputs,
    farthest_point_indices,
)
from kcg_connector.grasp.carts_v2.opposition_seed_generator import extract_object_grasp_band
from kcg_connector.grasp.carts_v2.surface_contact import material_bound_object_face_normals
from kcg_connector.grasp.carts_v2.three_contact_pose_initializer import (
    hand_contact_references, initialize_three_contact_pose,
    resolve_palm_configuration_rad,
)


_GLOBAL_QP_DEG = (0.0, 15.0, 30.0, 40.0, 45.0, 50.0, 55.0,
                  60.0, 65.0, 70.0, 75.0, 82.5, 90.0)
_DENSE_QP_DEG = (45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0)
_GLOBAL_AXIAL = (0.10, 0.30, 0.50, 0.70, 0.90)
_DENSE_AXIAL = (0.20, 0.40, 0.60, 0.80)
_PRESHAPES = {"P0": (.1, .1, .1), "P1": (.2, .05, .2), "P2": (.05, .2, .05)}
_AZIMUTH_STEP_DEG = 2.0
_FEATURE_SCAN_STEP_DEG = 1.0
_TARGET_WINDOW_RAD = math.radians(12.5)
_TARGET_REPRESENTATIVE_COUNT = 8
RADIAL_SUPPORT_TARGETS = "RADIAL_SUPPORT_72X24_REPRESENTATIVES"
FULL_ALLOWED_TARGETS = "B0_FULL_ALLOWED_SURFACE"
CONTINUOUS_ALLOWED_TARGETS = "B0_CONTINUOUS_ALLOWED_PROJECTED"


@dataclass(frozen=True)
class StructuredSeedSpec:
    source_index: int
    family: str
    qp_index: int
    qp_deg: float
    direction_index: int
    axial_index: int
    axial_ratio: float
    preshape_index: int
    preshape_id: str
    preshape: tuple[float, float, float]

    @property
    def candidate_id(self) -> str:
        tag = "a" if self.family == "GLOBAL" else "k"
        prefix = "g" if self.family == "GLOBAL" else "d"
        return (f"contactopt_{prefix}_q{self.qp_index:02d}_{tag}{self.direction_index:02d}_"
                f"z{self.axial_index}_p{self.preshape_index}")


def structured_seed_specifications() -> tuple[StructuredSeedSpec, ...]:
    """Return the preregistered 1,040 global plus 448 dense specifications."""
    rows, source = [], 0
    for qi, qp in enumerate(_GLOBAL_QP_DEG):
        for direction in range(16):
            for zi, ratio in enumerate(_GLOBAL_AXIAL):
                rows.append(StructuredSeedSpec(
                    source, "GLOBAL", qi, qp, direction, zi, ratio, 0, "P0",
                    _PRESHAPES["P0"]))
                source += 1
    for qi, qp in enumerate(_DENSE_QP_DEG):
        for zi, ratio in enumerate(_DENSE_AXIAL):
            for pi, name in enumerate(("P1", "P2")):
                for peak in range(8):
                    rows.append(StructuredSeedSpec(
                        source, "DENSE", qi, qp, peak, zi, ratio, pi, name,
                        _PRESHAPES[name]))
                    source += 1
    if (len(rows) != 1488 or sum(row.family == "GLOBAL" for row in rows) != 1040
            or len({row.candidate_id for row in rows}) != 1488):
        raise RuntimeError("STRUCTURED_1488_SPECIFICATION_INCOMPLETE")
    return tuple(rows)


def _cyclic_delta(left, right):
    return np.abs((np.asarray(left) - np.asarray(right) + math.pi) % (2.0 * math.pi) - math.pi)


def _radial_support_target_mask(allowed, areas, radii, angles, axial,
                                axial_range, angular_count, axial_count):
    """Choose one deterministic outside support face per existing angular/axial bin."""

    low, high = axial_range
    span = max(float(high - low), np.finfo(np.float64).eps)
    angular_bin = np.minimum(
        (np.asarray(angles) / (2.0 * math.pi) * angular_count).astype(np.int64),
        angular_count - 1)
    axial_bin = np.clip(
        ((np.asarray(axial) - low) / span * axial_count).astype(np.int64),
        0, axial_count - 1)
    face_ids = np.arange(len(radii), dtype=np.int64)
    valid = (np.asarray(allowed, dtype=np.bool_) & np.isfinite(radii)
             & np.isfinite(areas) & (np.asarray(areas) > 0.0))
    rows = face_ids[valid]
    if len(rows) == 0:
        raise ValueError("B0_SEED_TARGET_SURFACE_EMPTY")
    bins = axial_bin[rows] * angular_count + angular_bin[rows]
    order = np.lexsort((rows, -np.asarray(radii)[rows], bins))
    sorted_bins = bins[order]
    first = np.r_[True, sorted_bins[1:] != sorted_bins[:-1]]
    mask = np.zeros(len(radii), dtype=np.bool_)
    mask[rows[order[first]]] = True
    return mask


def _object_index(inputs: V2Inputs, target_surface_mode: str) -> dict[str, object]:
    band = dict(extract_object_grasp_band(inputs))
    mesh = inputs.object_contract.model.mesh
    points = np.asarray(mesh.face_centroids_m, dtype=np.float64)
    axis = np.asarray(band["axis_object"])
    origin = np.asarray(inputs.object_contract.model.assembly_axis_origin_m)
    relative = points - origin
    axial = relative @ axis
    radial = relative - np.outer(axial, axis)
    x_axis, y_axis = np.asarray(band["reference_x_object"]), np.asarray(
        band["reference_y_object"])
    angle = np.mod(np.arctan2(radial @ y_axis, radial @ x_axis), 2.0 * math.pi)
    roles, areas = (np.asarray(inputs.face_roles.face_role),
                    np.asarray(mesh.face_areas_m2))
    angular_count = int(inputs.config.section("surface_roles")["angular_bin_count"])
    axial_count = int(inputs.config.section("surface_roles")["axial_bin_count"])
    radii = np.linalg.norm(radial, axis=1)
    if target_surface_mode == RADIAL_SUPPORT_TARGETS:
        seed_targets = _radial_support_target_mask(
            inputs.face_roles.face_is_allowed, areas, radii, angle, axial,
            band["axial_range_m"], angular_count, axial_count)
        target_method = "MAXIMUM_RADIAL_SUPPORT_PER_72X24_BIN_FACE_ID_TIE_BREAK"
    elif target_surface_mode in (FULL_ALLOWED_TARGETS, CONTINUOUS_ALLOWED_TARGETS):
        seed_targets = np.asarray(inputs.face_roles.face_is_allowed, dtype=np.bool_)
        target_method = target_surface_mode
    else:
        raise ValueError("UNKNOWN_STRUCTURED_SEED_TARGET_SURFACE_MODE")
    return {**band, "points": points, "normals": material_bound_object_face_normals(inputs),
            "areas": areas, "roles": roles,
            "allowed": inputs.face_roles.face_is_allowed, "axial": axial, "angle": angle,
            "radius": radii, "seed_target": seed_targets,
            "seed_target_method": target_method,
            "project_axial_center": target_surface_mode == CONTINUOUS_ALLOWED_TARGETS,
            "use_complete_target_window": target_surface_mode == CONTINUOUS_ALLOWED_TARGETS,
            "target_representative_count": (64 if target_surface_mode ==
                                             CONTINUOUS_ALLOWED_TARGETS else 8),
            "axial_bin_count": axial_count}


def _axial_rows(index: Mapping[str, object], ratio: float, *, allowed: bool,
                offset_m: float = 0.0) -> np.ndarray:
    low, high = index["axial_range_m"]
    target, span = low + ratio * (high - low) + float(offset_m), high - low
    if target < low or target > high:
        raise ValueError("OBJECT_TARGET_AXIAL_OFFSET_OUTSIDE_GRASP_BAND")
    axial = np.asarray(index["axial"])
    mask = np.ones(len(axial), dtype=np.bool_)
    if allowed:
        mask &= np.asarray(index["seed_target"])
    rows = np.flatnonzero(mask & (np.abs(axial - target) <=
                         span / max(2, 2 * int(index["axial_bin_count"]))))
    if len(rows) == 0:
        candidates = np.flatnonzero(mask)
        if len(candidates) == 0:
            raise ValueError("OBJECT_AXIAL_SLICE_EMPTY")
        distance = np.abs(axial[candidates] - target)
        rows = candidates[distance <= float(np.min(distance)) + 1.0e-12]
    return rows


def _local_radius(index: Mapping[str, object], ratio: float) -> float:
    rows = _axial_rows(index, ratio, allowed=True)
    areas, radii = np.asarray(index["areas"])[rows], np.asarray(index["radius"])[rows]
    return float(np.average(radii, weights=areas))


def _angular_profile(index: Mapping[str, object], ratio: float) -> dict[str, np.ndarray]:
    count = int(round(360.0 / _AZIMUTH_STEP_DEG))
    rows = _axial_rows(index, ratio, allowed=False)
    angles, radii = np.asarray(index["angle"])[rows], np.asarray(index["radius"])[rows]
    bins = np.floor(np.degrees(angles) / _AZIMUTH_STEP_DEG).astype(int) % count
    order = np.lexsort((rows, -radii, bins))
    sorted_bins = bins[order]
    first = np.r_[True, sorted_bins[1:] != sorted_bins[:-1]]
    outer_rows, outer_bins = rows[order[first]], sorted_bins[first]
    role = np.full(count, HARD_FORBIDDEN, dtype=np.int64)
    role[outer_bins] = np.asarray(index["roles"])[outer_rows]
    usable = role != HARD_FORBIDDEN
    hard_margin, width = np.zeros(count), np.zeros(count)
    for row in np.flatnonzero(usable):
        left = right = 0
        while left < count and usable[(row - left - 1) % count]:
            left += 1
        while right < count and usable[(row + right + 1) % count]:
            right += 1
        width[row] = (left + right + 1) * _AZIMUTH_STEP_DEG
        hard_margin[row] = (min(left, right) + 0.5) * _AZIMUTH_STEP_DEG
    return {"usable": usable, "primary": role == PRIMARY_GRIP,
            "hard_margin_deg": hard_margin, "usable_width_deg": width}


def _feature_directions(profile: Mapping[str, np.ndarray], alphas) -> tuple[float, ...]:
    count = len(profile["usable"])
    ranked = []
    for sample in range(int(round(360.0 / _FEATURE_SCAN_STEP_DEG))):
        angle = sample * _FEATURE_SCAN_STEP_DEG
        rows = np.floor((angle + np.degrees(alphas)) /
                        _AZIMUTH_STEP_DEG).astype(int) % count
        if not bool(np.all(profile["usable"][rows])):
            continue
        margins, widths = profile["hard_margin_deg"][rows], profile["usable_width_deg"][rows]
        key = (-float(np.min(margins)), -float(np.min(widths)),
               -int(np.count_nonzero(profile["primary"][rows])), float(np.ptp(margins)), sample)
        ranked.append((key, angle))
    chosen = []
    for _key, angle in sorted(ranked):
        if all(float(_cyclic_delta(math.radians(angle), math.radians(old)))
               >= math.radians(25.0) for old in chosen):
            chosen.append(angle)
        if len(chosen) == 8:
            break
    return tuple(chosen)


def _target_region_candidates(index: Mapping[str, object], ratio: float, angle: float,
                              axial_offset_m: float = 0.0):
    rows = _axial_rows(index, ratio, allowed=True, offset_m=axial_offset_m)
    delta = _cyclic_delta(np.asarray(index["angle"])[rows], angle)
    eligible = rows[delta <= _TARGET_WINDOW_RAD]
    if len(eligible) == 0:
        raise ValueError("NO_USABLE_OBJECT_REGION_IN_TARGET_WINDOW")
    local_delta = _cyclic_delta(np.asarray(index["angle"])[eligible], angle)
    near = eligible
    if not bool(index.get("use_complete_target_window", False)):
        near = eligible[local_delta <= float(np.min(local_delta)) + math.radians(2.0)]
    areas, points = np.asarray(index["areas"])[near], np.asarray(index["points"])[near]
    normals = np.asarray(index["normals"])[near]
    center = np.average(points, axis=0, weights=areas)
    center_face = int(near[np.argmin(np.sum((points - center) ** 2, axis=1))])
    count = int(index.get("target_representative_count", _TARGET_REPRESENTATIVE_COUNT))
    if bool(index.get("use_complete_target_window", False)):
        span = np.ptp(points, axis=0)
        features = np.column_stack(((points - np.min(points, axis=0)) /
                                    np.where(span > 0.0, span, 1.0), normals))
        order = []
    else:
        features, order = points, [center_face]
    for local in farthest_point_indices(features, min(len(points), count)):
        face = int(near[int(local)])
        if face not in order:
            order.append(face)
        if len(order) == count:
            break
    low, high = index["axial_range_m"]
    target_axial = low + ratio * (high - low) + float(axial_offset_m)
    return tuple({"face": face, "point": np.asarray(index["points"])[face],
                  "normal": np.asarray(index["normals"])[face],
                  "role": int(np.asarray(index["roles"])[face]),
                  "region_area_m2": float(np.sum(areas)),
                  "target_axial_m": float(target_axial),
                  "angle_error_rad": float(_cyclic_delta(index["angle"][face], angle)),
                  "axial_error_m": abs(float(index["axial"][face]) - target_axial)}
                 for face in order)


def _edge_lengths(points):
    values = np.asarray(points, dtype=np.float64)
    return np.asarray((np.linalg.norm(values[0] - values[1]),
                       np.linalg.norm(values[0] - values[2]),
                       np.linalg.norm(values[1] - values[2])))


def _edge_lengths_batch(point_sets):
    values = np.asarray(point_sets, dtype=np.float64)
    return np.column_stack((np.linalg.norm(values[:, 0] - values[:, 1], axis=1),
                            np.linalg.norm(values[:, 0] - values[:, 2], axis=1),
                            np.linalg.norm(values[:, 1] - values[:, 2], axis=1)))


def _compatible_target_triplet(hand_points, hand_normals, candidate_sets,
                               contact_distance, maximum_alignment_count=None):
    sizes = tuple(len(rows) for rows in candidate_sets)
    grid = np.stack(np.meshgrid(*(np.arange(size) for size in sizes), indexing="ij"),
                    axis=-1).reshape(-1, 3)
    target_points = np.asarray([[candidate_sets[finger][choice]["point"]
                                 for finger, choice in enumerate(row)] for row in grid])
    edge_error = np.abs(_edge_lengths_batch(target_points) - _edge_lengths(hand_points))
    compatible = np.flatnonzero(np.all(edge_error <= 2.0 * contact_distance, axis=1))
    if len(compatible) == 0:
        raise ValueError("NO_EDGE_COMPATIBLE_TARGET_TRIPLET")
    evaluated = compatible
    if maximum_alignment_count is not None and len(evaluated) > maximum_alignment_count:
        maxima = np.max(edge_error[evaluated], axis=1)
        totals = np.sum(edge_error[evaluated], axis=1)
        order = np.lexsort((evaluated, totals, maxima))
        evaluated = evaluated[order[:int(maximum_alignment_count)]]
    best = None
    for index in evaluated:
        targets = tuple(candidate_sets[finger][int(choice)]
                        for finger, choice in enumerate(grid[index]))
        points = target_points[index]
        try:
            alignment = initialize_three_contact_pose(
                hand_points, hand_normals, points,
                np.asarray([row["normal"] for row in targets]))
        except ValueError as error:
            if str(error) == "POINT_NORMAL_ALIGNMENT_OPPOSED_NORMAL_FAILED":
                continue
            raise
        rank = (alignment["maximum_normal_residual_rad"],
                alignment["maximum_point_residual_m"],
                max(row["angle_error_rad"] for row in targets),
                max(row["axial_error_m"] for row in targets),
                -sum(row["role"] == PRIMARY_GRIP for row in targets),
                tuple(row["face"] for row in targets))
        if best is None or rank < best[0]:
            best = (rank, targets, alignment, edge_error[index])
    if best is None:
        raise ValueError("POINT_NORMAL_ALIGNMENT_OPPOSED_NORMAL_FAILED")
    rank, targets, alignment, errors = best
    audit = {"target_candidate_counts": list(sizes),
             "edge_compatible_triplet_count": int(len(compatible)),
             "normal_alignment_triplet_count": int(len(evaluated)),
             "selected_maximum_edge_error_m": float(np.max(errors)),
             "selected_maximum_angle_error_rad": float(rank[2]),
             "selected_maximum_axial_error_m": float(rank[3]),
             "selected_legacy_primary_target_count": int(-rank[4]),
             "legacy_primary_role": "DIAGNOSTIC_TIE_BREAK_ONLY"}
    return targets, alignment, audit


def _base_record(spec: StructuredSeedSpec, azimuth_deg=None,
                 effective_qp_rad=None) -> dict[str, object]:
    return {"candidate_id": spec.candidate_id, "family": spec.family,
            "qp_index": spec.qp_index, "qp_deg": spec.qp_deg,
            "requested_qp_deg": spec.qp_deg,
            "effective_qp_rad": effective_qp_rad,
            "effective_qp_deg": (None if effective_qp_rad is None else
                                 math.degrees(effective_qp_rad)),
            "azimuth_or_peak_index": spec.direction_index, "azimuth_deg": azimuth_deg,
            "axial_index": spec.axial_index, "axial_ratio": spec.axial_ratio,
            "preshape_index": spec.preshape_index, "preshape_id": spec.preshape_id,
            "preshape": list(spec.preshape)}


def _generate_one(inputs, index, spec, effective_qp, references, profiles, directions):
    key = (effective_qp, spec.axial_ratio, spec.preshape)
    if key not in references:
        try:
            references[key] = hand_contact_references(
                inputs, effective_qp, spec.preshape,
                _local_radius(index, spec.axial_ratio))
        except ValueError as error:
            references[key] = str(error)
    hand = references[key]
    if isinstance(hand, str):
        raise ValueError(hand)
    if spec.family == "GLOBAL":
        azimuth_deg = 22.5 * spec.direction_index
    else:
        if spec.axial_ratio not in profiles:
            profiles[spec.axial_ratio] = _angular_profile(index, spec.axial_ratio)
        if key not in directions:
            directions[key] = _feature_directions(
                profiles[spec.axial_ratio], hand["relative_azimuths_rad"])
        if spec.direction_index >= len(directions[key]):
            raise ValueError("INSUFFICIENT_25_DEG_FEATURE_DIRECTIONS")
        azimuth_deg = directions[key][spec.direction_index]
    angles = math.radians(azimuth_deg) + np.asarray(hand["relative_azimuths_rad"])
    offsets = np.asarray(hand["relative_axial_offsets_m"], dtype=np.float64)
    axial_shift = 0.0
    if bool(index.get("project_axial_center", False)):
        low, high = index["axial_range_m"]
        center = low + spec.axial_ratio * (high - low)
        feasible_low, feasible_high = low - float(np.min(offsets)), high - float(np.max(offsets))
        if feasible_low > feasible_high:
            raise ValueError("HAND_AXIAL_SPAN_EXCEEDS_OBJECT_GRASP_BAND")
        axial_shift = float(np.clip(center, feasible_low, feasible_high) - center)
    target_sets = [_target_region_candidates(
        index, spec.axial_ratio, float(angle), float(offset + axial_shift))
        for angle, offset in zip(angles, offsets)]
    contact_distance = float(inputs.config.section("closure_prediction")[
        "contact_distance_m"])
    targets, alignment, triplet_audit = _compatible_target_triplet(
        hand["points_handbase_m"], hand["normals_handbase"], target_sets,
        contact_distance, maximum_alignment_count=(64 if bool(
            index.get("use_complete_target_window", False)) else None))
    pose = np.asarray(alignment["object_from_hand"])
    approach = pose[:3, :3] @ np.asarray(hand["approach_direction_handbase"])
    approach_distance = float(inputs.config.section("dynamic")[
        "approach_clearance_height_m"])
    pregrasp_pose = np.array(pose, copy=True)
    pregrasp_pose[:3, 3] -= approach * approach_distance
    seed = CandidateSeed(
        candidate_id=spec.candidate_id, object_id=inputs.object_contract.object_id,
        anchor_face_index=int(targets[0]["face"]), anchor_position_object_m=tuple(
            float(value) for value in targets[0]["point"]),
        object_from_hand=tuple(float(value) for value in pose.ravel()),
        pregrasp_joint_positions_rad=hand["pregrasp_joint_positions_rad"],
        pregrasp_closure_phases=spec.preshape, source_sample_index=spec.source_index,
        generator_score=-float(alignment["maximum_point_residual_m"]),
        descriptor_id=f"CONTACTOPT_{spec.family}",
        approach_direction_object=tuple(float(value) for value in approach),
        maximum_closure_phase=float(inputs.config.section(
            "candidate_generation")["maximum_closure_phase"]),
        palm_configuration_rad=float(hand["effective_palm_configuration_rad"]))
    row = {**_base_record(spec, azimuth_deg, effective_qp),
           "status": "POSE_GENERATED", "reason": "",
           "target_object_face_indices": [int(value["face"]) for value in targets],
           "target_region_areas_m2": [float(value["region_area_m2"]) for value in targets],
           "target_axial_centers_m": [float(value["target_axial_m"]) for value in targets],
           "hand_relative_axial_offsets_m": list(hand["relative_axial_offsets_m"]),
           "projected_common_axial_shift_m": axial_shift,
           "reference_closure_phases": list(hand["reference_closure_phases"]),
           "grasp_pose_object_from_hand": [float(value) for value in pose.ravel()],
           "pregrasp_pose_object_from_hand": [float(value) for value in pregrasp_pose.ravel()],
           "approach_distance_m": approach_distance,
           "triplet_selection_policy": (
               "EDGE_COMPATIBLE_TOP64_THEN_MIN_MAX_NORMAL_THEN_MIN_MAX_POINT_RESIDUAL"
               if bool(index.get("use_complete_target_window", False)) else
               "ALL_EDGE_COMPATIBLE_THEN_MIN_MAX_NORMAL_THEN_MIN_MAX_POINT_RESIDUAL"),
           "maximum_point_residual_m": alignment["maximum_point_residual_m"],
           "maximum_normal_residual_rad": alignment["maximum_normal_residual_rad"],
           "pose_method": alignment["method"], **triplet_audit}
    return seed, row


def _reject_record(spec: StructuredSeedSpec, reason: str,
                   effective_qp_rad=None) -> dict[str, object]:
    code = str(reason).strip() or "UNSPECIFIED_SEED_GEOMETRY_REJECT"
    return {**_base_record(spec, effective_qp_rad=effective_qp_rad),
            "status": "SEED_GEOMETRY_REJECT", "reason": code}


def generate_structured_contact_seeds(
    inputs: V2Inputs,
    specifications: Sequence[StructuredSeedSpec] | None = None,
    *,
    target_surface_mode: str = RADIAL_SUPPORT_TARGETS,
) -> tuple[tuple[CandidateSeed, ...], dict[str, object]]:
    """Generate B0 poses; the default preserves all 1,488 specification outcomes."""

    if inputs.task_grip_surfaces is None or inputs.face_roles.method != B0_SURFACE_METHOD:
        raise ValueError("CONTACTOPT_REQUIRES_NAILFREE_B0_EXTERNAL_SURFACE")
    complete = specifications is None
    specs = (structured_seed_specifications() if complete else tuple(specifications))
    if not specs or len({row.candidate_id for row in specs}) != len(specs):
        raise ValueError("STRUCTURED_SEED_SUBSET_INVALID")
    index = _object_index(inputs, target_surface_mode)
    references, profiles, directions, seeds, records = {}, {}, {}, [], []
    identity = hashlib.sha256()
    for spec in specs:
        effective_qp = None
        try:
            effective_qp = resolve_palm_configuration_rad(
                inputs, math.radians(spec.qp_deg))
            seed, row = _generate_one(
                inputs, index, spec, effective_qp, references, profiles, directions)
            seeds.append(seed)
        except ValueError as error:
            row = _reject_record(spec, str(error), effective_qp)
        records.append(row)
        identity.update((spec.candidate_id + "\0" + row["status"] + "\n").encode())
    counts = {status: sum(row["status"] == status for row in records)
              for status in ("POSE_GENERATED", "SEED_GEOMETRY_REJECT")}
    if len(records) != len(specs) or sum(counts.values()) != len(specs):
        raise RuntimeError("STRUCTURED_1488_RESULT_INCOMPLETE")
    if complete and len(records) != 1488:
        raise RuntimeError("STRUCTURED_1488_RESULT_INCOMPLETE")
    audit = {"schema_version": "carts_contactopt_structured_seeds_v1",
             "claim_scope": "STRUCTURED_CONTACT_ALIGNMENT_NOT_COLLISION_OR_DYNAMIC_SUCCESS",
             "object_id": inputs.object_contract.object_id,
             "registered_specification_count": len(specs),
             "complete_1488_specification": complete,
             "global_specification_count": sum(row.family == "GLOBAL" for row in specs),
             "dense_specification_count": sum(row.family == "DENSE" for row in specs),
             "object_surface_semantics": B0_SURFACE_METHOD,
             "seed_target_method": index["seed_target_method"],
             "seed_target_face_count": int(np.count_nonzero(index["seed_target"])),
             "target_representative_count_per_finger": int(
                 index["target_representative_count"]),
             "generated_candidate_count": len(seeds),
             "status_counts": counts, "specification_status_sha256": identity.hexdigest(),
             "hand_reference_cache_count": len(references),
             "feature_direction_group_count": len(directions), "specifications": records}
    return tuple(seeds), audit


__all__ = ["CONTINUOUS_ALLOWED_TARGETS", "FULL_ALLOWED_TARGETS",
           "RADIAL_SUPPORT_TARGETS",
           "StructuredSeedSpec", "generate_structured_contact_seeds",
           "structured_seed_specifications"]
