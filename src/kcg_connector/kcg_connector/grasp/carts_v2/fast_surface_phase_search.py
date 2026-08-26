"""Cached Surface-V2 ranking before exact contact-conditioned validation."""

from __future__ import annotations

from collections import Counter
import hashlib
from itertools import combinations
import math

import numpy as np
from scipy.spatial import cKDTree

from kcg_connector.grasp.carts_v2.models import (
    CandidateSeed, HARD_FORBIDDEN, PRIMARY_GRIP, SECONDARY_GRIP, V2Inputs,
    joint_positions_for_phases,
)
from kcg_connector.grasp.carts_v2.opposition_seed_generator import (
    generate_feature_opposition_grid,
)
from kcg_connector.grasp.carts_v2.task_grip_surface import (
    allowed_object_grasp_center_m, motion_compatible_with_object_witness,
)


_EXPECTED_GRID_COUNT = 7 * 72 * 3 * 27
_MAXIMUM_EXACT = 24
_PATCH_POOL = 96
_AXIAL_LEVEL_COUNT = 3
_AXIAL_SELECTION_POLICY = "FIXED_AXIAL_STRATA_EXISTING_RANK_WITH_DETERMINISTIC_FILL_V1"
# Method-fixed tie band: values this close need exact triangles, never FAST-safe.
_SURFACE_ROLE_TIE_M = 1.0e-8
_TIE_SOURCE = "METHOD_FIXED_10_NM_SURFACE_ROLE_DISTANCE_TIE_BAND"


def _unit(value):
    vector = np.asarray(value, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if vector.shape != (3,) or not np.isfinite(length) or length <= 1.0e-12:
        raise ValueError("task-surface representative has no finite direction")
    return vector / length


def _surface_representatives(surface):
    triangles = np.asarray(surface.triangles_local_m, dtype=np.float64)
    centroids = np.mean(triangles, axis=1)
    areas = 0.5 * np.linalg.norm(np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    rows = []
    for patch in np.unique(surface.patch_indices):
        selected = surface.patch_indices == patch
        normal = np.sum(surface.face_normals_local[selected] * areas[selected, None], axis=0)
        rows.append((np.average(centroids[selected], axis=0, weights=areas[selected]),
                     _unit(normal), float(np.sum(areas[selected]))))
    normal = _unit(np.sum(surface.face_normals_local * areas[:, None], axis=0))
    return {
        "center": np.average(centroids, axis=0, weights=areas), "normal": normal,
        "patch_points": np.asarray([row[0] for row in rows]),
        "patch_normals": np.asarray([row[1] for row in rows]),
        "patch_areas": np.asarray([row[2] for row in rows]),
    }


def _box_corners(triangles):
    points = np.asarray(triangles, dtype=np.float64).reshape(-1, 3)
    low, high = np.min(points, axis=0), np.max(points, axis=0)
    return np.asarray([(x, y, z) for x in (low[0], high[0])
                       for y in (low[1], high[1]) for z in (low[2], high[2])])


def _geometry_support(inputs):
    boxes = {name: _box_corners(value)
             for name, value in inputs.hand_collision_triangles_by_link.items()}
    adjacent = {tuple(sorted((joint.parent_link, joint.child_link)))
                for joint in inputs.hand_model.joints.values()
                if joint.parent_link in boxes and joint.child_link in boxes}
    pairs = tuple(pair for pair in combinations(sorted(boxes), 2) if pair not in adjacent)
    return boxes, pairs


def _state_support(transforms, boxes, pairs):
    by_link = {name: points @ transforms[name][:3, :3].T + transforms[name][:3, 3]
               for name, points in boxes.items()}
    overlap = sum(bool(np.all(
        np.minimum(np.max(by_link[a], axis=0), np.max(by_link[b], axis=0)) >=
        np.maximum(np.min(by_link[a], axis=0), np.min(by_link[b], axis=0))))
        for a, b in pairs)
    return np.concatenate(tuple(by_link.values())), overlap


def _kinematic_cache(inputs, seed, representatives, boxes, pairs):
    maximum = float(inputs.config.section("candidate_generation")["maximum_closure_phase"])
    samples = int(inputs.config.section("closure_prediction")["phase_sample_count"])
    preshape = tuple(float(value) for value in seed.pregrasp_closure_phases)
    paths = []
    for finger, (pad_name, surface) in enumerate(sorted(inputs.task_grip_surfaces.items())):
        phase_grid = np.linspace(preshape[finger], maximum, samples)
        centers, center_normals, patches, patch_normals = [], [], [], []
        supports, overlaps = [], []
        for phase in phase_grid:
            phases = list(preshape)
            phases[finger] = float(phase)
            joints = joint_positions_for_phases(
                inputs, tuple(phases),
                reference_joint_positions_rad=seed.pregrasp_joint_positions_rad)
            transforms = inputs.hand_model.forward_kinematics(joints)
            transform = np.asarray(transforms[surface.link_name], dtype=np.float64)
            rotation, translation = transform[:3, :3], transform[:3, 3]
            rep = representatives[pad_name]
            centers.append(rep["center"] @ rotation.T + translation)
            center_normals.append(rep["normal"] @ rotation.T)
            patches.append(rep["patch_points"] @ rotation.T + translation)
            patch_normals.append(rep["patch_normals"] @ rotation.T)
            support, overlap = _state_support(transforms, boxes, pairs)
            supports.append(support)
            overlaps.append(overlap)
        center_array, patch_array = np.asarray(centers), np.asarray(patches)
        paths.append({
            "phases": phase_grid, "centers": center_array,
            "center_normals": np.asarray(center_normals),
            "center_motion": np.gradient(center_array, phase_grid, axis=0),
            "patches": patch_array, "patch_normals": np.asarray(patch_normals),
            "patch_motion": np.gradient(patch_array, phase_grid, axis=0),
            "patch_areas": representatives[pad_name]["patch_areas"],
            "support": np.asarray(supports), "self_overlap": max(overlaps),
        })
    return {"preshape": preshape, "paths": tuple(paths)}


def _role_indexes(inputs):
    mesh, result = inputs.object_contract.model.mesh, {}
    for role in (PRIMARY_GRIP, SECONDARY_GRIP, HARD_FORBIDDEN):
        faces = inputs.face_roles.indices_for_role(role)
        points = np.asarray(mesh.face_centroids_m[faces], dtype=np.float64)
        result[role] = (cKDTree(points), faces, points,
                        np.asarray(mesh.face_normals[faces], dtype=np.float64))
    return result, allowed_object_grasp_center_m(inputs)


def _query(indexes, role, points):
    tree, face_ids, witnesses, normals = indexes[role]
    distance, rows = tree.query(np.asarray(points).reshape(-1, 3), k=1)
    return np.asarray(distance), witnesses[rows], normals[rows], face_ids[rows]


def _table_proxy(inputs, points_object):
    transform = inputs.frozen_world_from_object
    world = points_object @ transform[:3, :3].T + transform[:3, 3]
    bounds = np.asarray(inputs.table_xy_bounds_m)
    inside = ((world[:, 0] >= bounds[0, 0]) & (world[:, 0] <= bounds[0, 1])
              & (world[:, 1] >= bounds[1, 0]) & (world[:, 1] <= bounds[1, 1]))
    return math.inf if not np.any(inside) else float(
        np.min(world[inside, 2]) - inputs.table_top_z_m)


def _contact_summary(inputs, indexes, object_center, points, normals, motion,
                     phases, areas=None):
    shape = points.shape[:-1]
    flat_points, flat_normals, flat_motion = (value.reshape(-1, 3)
                                              for value in (points, normals, motion))
    role_rows = {}
    for role in (PRIMARY_GRIP, SECONDARY_GRIP):
        distance, witness, object_normal, face = _query(indexes, role, flat_points)
        compatible = motion_compatible_with_object_witness(
            flat_points, witness, flat_normals, object_normal, flat_motion,
            object_center, float(inputs.config.section("closure_prediction")[
                "minimum_inward_motion_m_per_phase"]))
        role_rows[role] = (distance.reshape(shape), compatible.reshape(shape),
                           face.reshape(shape))
    hard_distance = _query(indexes, HARD_FORBIDDEN, flat_points)[0].reshape(shape)
    primary, secondary = role_rows[PRIMARY_GRIP], role_rows[SECONDARY_GRIP]
    primary_gap = np.where(primary[1], primary[0], np.inf)
    secondary_gap = np.where(secondary[1], secondary[0], np.inf)
    gap, is_primary = np.minimum(primary_gap, secondary_gap), primary_gap <= secondary_gap
    threshold = float(inputs.config.section("closure_prediction")["contact_distance_m"])
    if gap.ndim == 1:
        eligible = gap <= threshold
        chosen = int(np.flatnonzero(eligible)[0]) if np.any(eligible) else int(np.argmin(gap))
        return {"found": bool(eligible[chosen]), "phase": float(phases[chosen]),
                "hard_margin": float(hard_distance[chosen] - gap[chosen]),
                "primary_fraction": float(is_primary[chosen]), "area": 0.0,
                "witness_count": int(eligible[chosen])}
    patch_areas, best = np.asarray(areas), None
    for phase_index in range(len(phases)):
        eligible = gap[phase_index] <= threshold
        count, area = int(np.count_nonzero(eligible)), float(np.sum(patch_areas[eligible]))
        primary_area = float(np.sum(patch_areas[eligible & is_primary[phase_index]]))
        fraction = 0.0 if area == 0.0 else primary_area / area
        margin = (-math.inf if count == 0 else float(np.min(
            hard_distance[phase_index, eligible] - gap[phase_index, eligible])))
        key = (-min(count, 3), -margin, -fraction, -area, phase_index)
        if best is None or key < best[0]:
            best = (key, count, area, fraction, margin, phase_index)
    _, count, area, fraction, margin, phase_index = best
    return {"found": count >= 3, "phase": float(phases[phase_index]),
            "hard_margin": margin, "primary_fraction": fraction, "area": area,
            "witness_count": count}


def _boundary_status(three, hard_margin):
    if not three:
        return "FAST_NO_THREE_CONTACT_REGIONS"
    if hard_margin < -_SURFACE_ROLE_TIE_M:
        return "FAST_HARD_FIRST_PROXY"
    return ("UNRESOLVED_BOUNDARY" if hard_margin <= _SURFACE_ROLE_TIE_M
            else "FAST_SHORTLIST_ELIGIBLE")


def _score(inputs, indexes, object_center, cache, seed, *, patches):
    transform = seed.object_from_hand_matrix()
    rotation, translation = transform[:3, :3], transform[:3, 3]
    contacts, supports, self_overlaps = [], [], []
    for path in cache["paths"]:
        prefix = "patch" if patches else "center"
        points = path[f"{prefix}es" if patches else "centers"] @ rotation.T + translation
        contacts.append(_contact_summary(
            inputs, indexes, object_center, points,
            path[f"{prefix}_normals"] @ rotation.T,
            path[f"{prefix}_motion"] @ rotation.T, path["phases"],
            path["patch_areas"] if patches else None))
        support = path["support"] @ rotation.T + translation
        supports.append(support.reshape(-1, 3))
        self_overlaps.append(path["self_overlap"])
    three = all(row["found"] for row in contacts)
    margin = min(row["hard_margin"] for row in contacts)
    travel = [row["phase"] - cache["preshape"][index]
              for index, row in enumerate(contacts)]
    area = sum(row["area"] for row in contacts)
    primary = (sum(row["primary_fraction"] * row["area"] for row in contacts) / area
               if area else sum(row["primary_fraction"] for row in contacts) / 3.0)
    return {"status": _boundary_status(three, margin),
            "three_contact_regions": three, "hard_margin_m": margin,
            "primary_fraction": primary, "effective_area_m2": area,
            "table_clearance_proxy_m": _table_proxy(inputs, np.concatenate(supports)),
            "closure_balance_phase": float(np.ptp(travel)),
            "stop_phases": [row["phase"] for row in contacts],
            "witness_counts": [row["witness_count"] for row in contacts],
            "maximum_self_aabb_overlap_count": max(self_overlaps)}


def _rank_key(row):
    margin = row["hard_margin_m"] if math.isfinite(row["hard_margin_m"]) else -1.0e99
    table = (row["table_clearance_proxy_m"]
             if math.isfinite(row["table_clearance_proxy_m"]) else 1.0e99)
    return (not row["three_contact_regions"], -margin, -row["primary_fraction"],
            -row["effective_area_m2"], -table, row["closure_balance_phase"],
            row["candidate_id"])


def _stratified_rank(rows, total):
    """Keep fixed axial coverage, then preserve the existing rank within it."""

    ranked = sorted(rows, key=_rank_key)
    base, remainder = divmod(int(total), _AXIAL_LEVEL_COUNT)
    chosen = []
    for axial_index in range(_AXIAL_LEVEL_COUNT):
        quota = base + int(axial_index < remainder)
        layer = [row for row in ranked if row.get("axial_index") == axial_index]
        chosen.extend(layer[:quota])
    selected_ids = {row["candidate_id"] for row in chosen}
    chosen.extend(row for row in ranked if row["candidate_id"] not in selected_ids
                  and len(chosen) < total)
    if len(chosen) != total or len({row["candidate_id"] for row in chosen}) != total:
        raise RuntimeError("axial-stratified ranking cannot fill its fixed budget")
    return sorted(chosen, key=_rank_key)


def _axial_counts(rows):
    return {str(index): sum(row["axial_index"] == index for row in rows)
            for index in range(_AXIAL_LEVEL_COUNT)}


def _audit_row(row):
    return {
        key: (None if isinstance(value, (float, np.floating))
              and not math.isfinite(float(value)) else value)
        for key, value in row.items() if not key.startswith("_")
    }


def search_feature_aware_opposition(
    inputs: V2Inputs, *, maximum_exact_candidates: int = _MAXIMUM_EXACT,
) -> tuple[tuple[CandidateSeed, ...], dict[str, object]]:
    """Rank all 40,824 grid rows cheaply; make no exact-safety claim."""

    if inputs.task_grip_surfaces is None or inputs.face_roles.method != (
            "TASK_AXIS_OUTER_ENVELOPE_THREE_ROLE_V2"):
        raise ValueError("feature-aware search requires nail-free Surface V2 inputs")
    limit = int(maximum_exact_candidates)
    if not 1 <= limit <= _MAXIMUM_EXACT:
        raise ValueError("exact shortlist budget must lie in [1, 24]")
    seeds, grid_audit = generate_feature_opposition_grid(inputs)
    if (len(seeds) != _EXPECTED_GRID_COUNT
            or len({seed.candidate_id for seed in seeds}) != _EXPECTED_GRID_COUNT
            or len(grid_audit.get("axial_fractions", ())) != _AXIAL_LEVEL_COUNT):
        raise RuntimeError("fixed feature-opposition grid is incomplete")
    indexes, object_center = _role_indexes(inputs)
    representatives = {name: _surface_representatives(surface)
                       for name, surface in inputs.task_grip_surfaces.items()}
    boxes, pairs = _geometry_support(inputs)
    caches, rows, classification = {}, [], hashlib.sha256()
    counts = Counter()
    for seed in seeds:
        axial_index = int(seed.source_sample_index) % _AXIAL_LEVEL_COUNT
        if f"_z{axial_index}__" not in seed.candidate_id:
            raise RuntimeError("feature-grid axial identity changed")
        key = (float(seed.palm_configuration_rad), seed.pregrasp_closure_phases)
        if key not in caches:
            caches[key] = _kinematic_cache(inputs, seed, representatives, boxes, pairs)
        score = _score(inputs, indexes, object_center, caches[key], seed, patches=False)
        row = {"candidate_id": seed.candidate_id, "_seed": seed, "_cache_key": key,
               "source_sample_index": seed.source_sample_index,
               "axial_index": axial_index,
               "palm_configuration_rad": seed.palm_configuration_rad,
               "pregrasp_closure_phases": list(seed.pregrasp_closure_phases), **score}
        rows.append(row)
        counts[score["status"]] += 1
        classification.update(f"{seed.candidate_id}\0{score['status']}\n".encode())
    patch_pool = _stratified_rank(rows, max(_PATCH_POOL, limit * 4))
    patch_rows = []
    for row in patch_pool:
        score = _score(inputs, indexes, object_center, caches[row["_cache_key"]],
                       row["_seed"], patches=True)
        patch_rows.append({**row, **score})
    exact_rows = _stratified_rank(patch_rows, limit)
    shortlist = tuple(row["_seed"] for row in exact_rows)
    audit = {
        "schema_version": "carts_surface_v2_feature_search_v1",
        "claim_scope": "CACHED_PROXIMITY_RANKING_NOT_EXACT_OR_DYNAMIC_SUCCESS",
        "object_id": inputs.object_contract.object_id, "grid_audit": grid_audit,
        "registered_candidate_count": len(rows), "expected_candidate_count": 40824,
        "fk_cache_count": len(caches), "center_status_counts": dict(sorted(counts.items())),
        "center_classification_sha256": classification.hexdigest(),
        "patch_ranked_count": len(patch_rows), "exact_shortlist_count": len(shortlist),
        "exact_shortlist_limit": limit, "surface_role_tie_m": _SURFACE_ROLE_TIE_M,
        "surface_role_tie_source": _TIE_SOURCE,
        "axial_selection_policy": _AXIAL_SELECTION_POLICY,
        "patch_axial_counts": _axial_counts(patch_rows),
        "exact_axial_counts": _axial_counts(exact_rows),
        "distance_backend": "SCIPY_CKDTREE_REGISTERED_FACE_CENTROID_PROXY",
        "patch_candidates": [_audit_row(row) for row in patch_rows],
        "exact_shortlist": [row["candidate_id"] for row in exact_rows],
    }
    return shortlist, audit


__all__ = ["search_feature_aware_opposition"]
