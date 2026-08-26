"""Cheap proxy screening and sequential contact intervals for 1488 seeds."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Callable, Mapping, Sequence

import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.transform import Rotation

try:
    import fcl
except ImportError:
    fcl = None

from kcg_connector.grasp.carts_v2.models import (
    CandidateSeed, FACE_ROLE_NAMES, HARD_FORBIDDEN, PRIMARY_GRIP,
    SECONDARY_GRIP, V2Inputs, farthest_point_indices,
    joint_positions_for_phases,
)
from kcg_connector.grasp.carts_v2.surface_contact import (
    material_bound_object_face_normals,
)
from kcg_connector.grasp.carts_v2.task_grip_surface import (
    allowed_object_grasp_center_m, motion_compatible_with_object_witness,
)
from kcg_connector.grasp.carts_v2.fast_filter import build_fcl_bvh_model


_REPRESENTATIVE_COUNT = 64
_COARSE_PHASE_COUNT = 10
_MAXIMUM_INCREMENT_RAD = 0.0015
_EXPECTED_SPEC_COUNT = 1488
_FAMILY_LIMIT = 60
_TOP_COUNT = 120
_PALM_LIMITS = {"GLOBAL": 6, "DENSE_OPPOSITION": 10}


def _finite(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _representatives(inputs: V2Inputs) -> dict[str, Mapping[str, object]]:
    if inputs.task_grip_surfaces is None:
        raise ValueError("contact proxies require TASK_GRIP_SURFACE")
    result = {}
    for name, surface in sorted(inputs.task_grip_surfaces.items()):
        triangles = np.asarray(surface.triangles_local_m, dtype=np.float64)
        centers = np.mean(triangles, axis=1)
        span = np.ptp(centers, axis=0)
        scaled = (centers - np.min(centers, axis=0)) / np.where(span > 0.0, span, 1.0)
        features = np.column_stack((scaled, np.asarray(surface.face_normals_local)))
        if len(features) < _REPRESENTATIVE_COUNT:
            raise ValueError(f"{name} has fewer than 64 representative faces")
        selected = farthest_point_indices(features, _REPRESENTATIVE_COUNT)
        result[name] = {"link": surface.link_name, "points": centers[selected],
                        "normals": np.asarray(surface.face_normals_local)[selected]}
    if tuple(result) != ("finger_1_pad", "finger_2_pad", "finger_3_pad"):
        raise ValueError("three-finger TASK_GRIP_SURFACE identity changed")
    return result


def _role_trees(inputs: V2Inputs) -> tuple[dict[int, tuple], np.ndarray]:
    mesh, result = inputs.object_contract.model.mesh, {}
    normals = material_bound_object_face_normals(inputs)
    for role in (PRIMARY_GRIP, SECONDARY_GRIP, HARD_FORBIDDEN):
        faces = inputs.face_roles.indices_for_role(role)
        if len(faces) == 0:
            raise ValueError(f"object role {FACE_ROLE_NAMES[role]} is empty")
        points = np.asarray(mesh.face_centroids_m[faces], dtype=np.float64)
        result[role] = (cKDTree(points), faces, points, normals[faces])
    return result, allowed_object_grasp_center_m(inputs)


def _box_support(inputs: V2Inputs):
    if fcl is None:
        raise RuntimeError("python-fcl is required for cached self-collision broad phase")
    boxes, table_points, collision_objects = {}, {}, {}
    for name, triangles in inputs.hand_collision_triangles_by_link.items():
        points = np.unique(np.asarray(triangles, dtype=np.float64).reshape(-1, 3), axis=0)
        low, high = np.min(points, axis=0), np.max(points, axis=0)
        boxes[name] = np.asarray([(x, y, z) for x in (low[0], high[0])
                                 for y in (low[1], high[1]) for z in (low[2], high[2])])
        table_points[name] = points[ConvexHull(points).vertices]
        collision_objects[name] = fcl.CollisionObject(build_fcl_bvh_model(triangles))
    adjacent = {tuple(sorted((joint.parent_link, joint.child_link)))
                for joint in inputs.hand_model.joints.values()
                if joint.parent_link in boxes and joint.child_link in boxes}
    names = sorted(boxes)
    pairs = tuple((a, b) for index, a in enumerate(names) for b in names[index + 1:]
                  if tuple(sorted((a, b))) not in adjacent)
    return boxes, table_points, collision_objects, pairs


def _context(inputs: V2Inputs) -> dict[str, object]:
    maximum = float(inputs.config.section("candidate_generation")["maximum_closure_phase"])
    contact = float(inputs.config.section("closure_prediction")["contact_distance_m"])
    motion = float(inputs.config.section("closure_prediction")["minimum_inward_motion_m_per_phase"])
    table_clearance = float(inputs.config.section("height_projection")[
        "table_operation_clearance_m"])
    active = []
    for row in np.asarray(inputs.closing_directions):
        indices = np.flatnonzero(row)
        if len(indices) != 1:
            raise ValueError("each finger must have exactly one active closing joint")
        active.append((int(indices[0]), float(np.sign(row[indices[0]]))))
    boxes, table_points, collision_objects, pairs = _box_support(inputs)
    trees, center = _role_trees(inputs)
    return {"inputs": inputs, "representatives": _representatives(inputs),
            "trees": trees, "object_center": center, "boxes": boxes,
            "table_points": table_points, "self_objects": collision_objects,
            "pairs": pairs, "self_cache": {}, "maximum_phase": maximum,
            "contact_distance": contact,
            "minimum_motion": motion,
            "table_operation_clearance_m": table_clearance,
            "active": tuple(active), "fk": {}}


def _joints_and_fk(context, seed, phases):
    inputs = context["inputs"]
    joints = joint_positions_for_phases(
        inputs, tuple(float(value) for value in phases),
        reference_joint_positions_rad=seed.pregrasp_joint_positions_rad)
    key = tuple(np.round(joints, 12))
    if key not in context["fk"]:
        context["fk"][key] = inputs.hand_model.forward_kinematics(joints)
    return joints, context["fk"][key]


def _state_margins(context, seed, transforms) -> tuple[float, float]:
    object_from_hand = seed.object_from_hand_matrix()
    world_from_hand = context["inputs"].frozen_world_from_object @ object_from_hand
    table = math.inf
    bounds = np.asarray(context["inputs"].table_xy_bounds_m)
    hand_bounds = {}
    for link, corners in context["boxes"].items():
        hand = corners @ transforms[link][:3, :3].T + transforms[link][:3, 3]
        hand_bounds[link] = (np.min(hand, axis=0), np.max(hand, axis=0))
        support = context["table_points"][link]
        support = support @ transforms[link][:3, :3].T + transforms[link][:3, 3]
        world = support @ world_from_hand[:3, :3].T + world_from_hand[:3, 3]
        low, high = np.min(world, axis=0), np.max(world, axis=0)
        if high[0] >= bounds[0, 0] and low[0] <= bounds[0, 1] \
                and high[1] >= bounds[1, 0] and low[1] <= bounds[1, 1]:
            table = min(table, float(low[2] - context["inputs"].table_top_z_m))
    cache_key = id(transforms)
    if cache_key in context["self_cache"]:
        return table, context["self_cache"][cache_key]
    for link, collision_object in context["self_objects"].items():
        transform = transforms[link]
        collision_object.setTransform(fcl.Transform(transform[:3, :3], transform[:3, 3]))
    self_margin = math.inf
    for left, right in context["pairs"]:
        ll, lh = hand_bounds[left]
        rl, rh = hand_bounds[right]
        broad = float(np.max(np.maximum(ll - rh, rl - lh)))
        if broad > 0.0:
            self_margin = min(self_margin, broad)
            continue
        first, second = context["self_objects"][left], context["self_objects"][right]
        if fcl.collide(first, second, fcl.CollisionRequest(num_max_contacts=1),
                       fcl.CollisionResult()):
            self_margin = -np.finfo(np.float64).eps
            break
        distance = float(fcl.distance(first, second, fcl.DistanceRequest(),
                                      fcl.DistanceResult()))
        self_margin = min(self_margin, distance)
    context["self_cache"][cache_key] = self_margin
    return table, self_margin


def _query_role(context, role, points):
    tree, faces, witnesses, normals = context["trees"][role]
    distance, index = tree.query(points, k=1)
    return np.asarray(distance), witnesses[index], normals[index], faces[index]


def _finger_metrics(context, seed, phases, finger):
    joints, transforms = _joints_and_fk(context, seed, phases)
    name = f"finger_{finger + 1}_pad"
    representative = context["representatives"][name]
    link_transform = np.asarray(transforms[representative["link"]])
    hand_points = representative["points"] @ link_transform[:3, :3].T + link_transform[:3, 3]
    hand_normals = representative["normals"] @ link_transform[:3, :3].T
    object_from_hand = seed.object_from_hand_matrix()
    points = hand_points @ object_from_hand[:3, :3].T + object_from_hand[:3, 3]
    normals = hand_normals @ object_from_hand[:3, :3].T
    moved_phases = list(phases)
    delta = min(1.0e-4, context["maximum_phase"] - float(phases[finger]))
    if delta <= 1.0e-12:
        delta = -min(1.0e-4, float(phases[finger]) - seed.pregrasp_closure_phases[finger])
    moved_phases[finger] += delta
    _, moved_fk = _joints_and_fk(context, seed, moved_phases)
    moved_link = np.asarray(moved_fk[representative["link"]])
    moved_hand = representative["points"] @ moved_link[:3, :3].T + moved_link[:3, 3]
    moved = moved_hand @ object_from_hand[:3, :3].T + object_from_hand[:3, 3]
    motion = np.zeros_like(points) if abs(delta) <= 1.0e-12 else (moved - points) / delta
    candidates = []
    for role in (PRIMARY_GRIP, SECONDARY_GRIP):
        distance, witness, object_normal, face = _query_role(context, role, points)
        compatible = motion_compatible_with_object_witness(
            points, witness, normals, object_normal, motion, context["object_center"],
            context["minimum_motion"])
        for index in np.flatnonzero(compatible):
            candidates.append((float(distance[index]), role, int(index), int(face[index])))
    best = min(candidates, default=(math.inf, HARD_FORBIDDEN, -1, -1))
    hard = _query_role(context, HARD_FORBIDDEN, points)[0]
    table, self_margin = _state_margins(context, seed, transforms)
    hard_margin = float(np.min(hard) - best[0]) if math.isfinite(best[0]) else -math.inf
    q_index = context["active"][finger][0]
    safe = bool(math.isfinite(best[0]) and hard_margin > 0.0
                and table >= context["table_operation_clearance_m"]
                and self_margin > 0.0)
    return {"phase": float(phases[finger]), "q_rad": float(joints[q_index]),
            "gap_m": best[0], "region": FACE_ROLE_NAMES[best[1]],
            "object_face_index": best[3], "hard_margin_m": hard_margin,
            "table_margin_m": table, "self_margin_m": self_margin,
            "safe": safe, "contact": safe and best[0] <= context["contact_distance"]}


def _coarse_rows(context, seed, base_phases, finger):
    phases = np.linspace(float(base_phases[finger]), context["maximum_phase"],
                         _COARSE_PHASE_COUNT)
    result = []
    for phase in phases:
        current = list(base_phases); current[finger] = float(phase)
        result.append(_finger_metrics(context, seed, current, finger))
    return result


def _cheap_seed(context, seed, spec):
    fingers, path_table, path_self = [], math.inf, math.inf
    for finger in range(3):
        rows = _coarse_rows(context, seed, seed.pregrasp_closure_phases, finger)
        best_index = min(range(len(rows)), key=lambda index: rows[index]["gap_m"])
        best, prefix = rows[best_index], rows[:best_index + 1]
        decreasing = (best_index > 0 and best["gap_m"] < rows[0]["gap_m"]
                      or rows[0]["contact"])
        path_table = min(path_table, *(row["table_margin_m"] for row in prefix))
        path_self = min(path_self, *(row["self_margin_m"] for row in prefix))
        fingers.append({**best, "closing_direction_reasonable": decreasing,
                        "remaining_closure_phase": context["maximum_phase"] - best["phase"]})
    gaps = [row["gap_m"] for row in fingers]
    hard = min(row["hard_margin_m"] for row in fingers)
    passed = bool(all(row["closing_direction_reasonable"] for row in fingers)
                  and hard > 0.0
                  and path_table >= context["table_operation_clearance_m"]
                  and path_self > 0.0)
    return {"candidate_id": seed.candidate_id, "status": "NEAR_CONTACT_SEED" if passed else "CHEAP_REJECT",
            "reason": "" if passed else "CHEAP_PROXY_DIRECTION_OR_MARGIN_REJECT",
            "family": _family(spec), "palm_key": _palm(spec, seed),
            "axial_key": _axial(spec), "azimuth_key": _azimuth_bin(spec),
            "maximum_positive_gap_m": max(gaps),
            "gap_imbalance_m": float(np.ptp(gaps)), "hard_margin_m": hard,
            "table_margin_m": path_table, "self_margin_m": path_self,
            "remaining_closure_phase": max(row["remaining_closure_phase"] for row in fingers),
            "finger_proxies": fingers, "_seed": seed}


def _family(row) -> str:
    value = str(row.get("family", row.get("seed_family", row.get("group", "")))).upper()
    if "DENSE" in value:
        return "DENSE_OPPOSITION"
    if "GLOBAL" in value:
        return "GLOBAL"
    raise ValueError("seed specification lacks GLOBAL/DENSE_OPPOSITION family")


def _palm(row, seed=None) -> float:
    if "palm_configuration_rad" in row:
        return float(row["palm_configuration_rad"])
    if "palm_configuration_deg" in row:
        return math.radians(float(row["palm_configuration_deg"]))
    if seed is not None and seed.palm_configuration_rad is not None:
        return float(seed.palm_configuration_rad)
    raise ValueError("seed specification lacks palm configuration")


def _axial(row) -> float:
    for key in ("axial_ratio", "axial_fraction", "axial_index"):
        if key in row:
            return float(row[key])
    raise ValueError("seed specification lacks axial stratum")


def _azimuth_bin(row) -> int:
    if "azimuth_deg" not in row:
        raise ValueError("seed specification lacks azimuth")
    return int(math.floor((float(row["azimuth_deg"]) % 360.0) / 22.5))


def _rank(row):
    return (row["maximum_positive_gap_m"], row["gap_imbalance_m"],
            -row["hard_margin_m"], -row["table_margin_m"],
            row["remaining_closure_phase"], row["candidate_id"])


def _feature(seed: CandidateSeed) -> np.ndarray:
    transform = seed.object_from_hand_matrix()
    euler = Rotation.from_matrix(transform[:3, :3]).as_euler("xyz")
    return np.asarray((*transform[:3, 3], *euler, float(seed.palm_configuration_rad),
                       *seed.pregrasp_closure_phases), dtype=np.float64)


def _family_pick(rows, limit, palm_limit):
    ranked = sorted(rows, key=_rank)
    if not ranked or limit <= 0:
        return []
    quota = defaultdict(list)
    for index, row in enumerate(ranked):
        quota[("palm", row["palm_key"])].append(index)
        quota[("axial", row["axial_key"])].append(index)
        quota[("azimuth", row["azimuth_key"])].append(index)
    selected = []
    palm_counts, azimuth_counts = defaultdict(int), defaultdict(int)

    def add(index):
        row = ranked[index]
        if index in selected or palm_counts[row["palm_key"]] >= palm_limit \
                or azimuth_counts[row["azimuth_key"]] >= palm_limit:
            return False
        selected.append(index)
        palm_counts[row["palm_key"]] += 1
        azimuth_counts[row["azimuth_key"]] += 1
        return True

    for values in quota.values():
        for index in values:
            if add(index):
                break
    if len(selected) > limit:
        raise ValueError("palm/axial diversity quota exceeds family cap")
    features = np.asarray([_feature(row["_seed"]) for row in ranked])
    span = np.ptp(features, axis=0)
    features = (features - np.min(features, axis=0)) / np.where(span > 0.0, span, 1.0)
    while len(selected) < min(limit, len(ranked)):
        remaining = np.asarray([
            index for index, row in enumerate(ranked)
            if index not in selected and palm_counts[row["palm_key"]] < palm_limit
            and azimuth_counts[row["azimuth_key"]] < palm_limit
        ])
        if len(remaining) == 0:
            break
        distance = np.min(np.sum((features[remaining, None] - features[selected]) ** 2, axis=2), axis=1)
        add(int(remaining[int(np.argmax(distance))]))
    return [ranked[index] for index in selected]


def _select_top120(rows):
    orders = {
        family: _family_pick(
            [row for row in rows if row["family"] == family],
            len(rows), _PALM_LIMITS[family])
        for family in ("GLOBAL", "DENSE_OPPOSITION")
    }
    result = [row for family in orders for row in orders[family][:_FAMILY_LIMIT]]
    remaining = {
        family: list(orders[family][_FAMILY_LIMIT:]) for family in orders
    }
    while len(result) < _TOP_COUNT and any(remaining.values()):
        available = [(rows[0], family) for family, rows in remaining.items() if rows]
        _row, family = min(available, key=lambda item: (_rank(item[0]), item[1]))
        result.append(remaining[family].pop(0))
    return tuple(sorted(result, key=_rank))


def _bisect_pair(low, high, evaluate: Callable[[float], Mapping], predicate: Callable[[Mapping], bool]):
    low_state, high_state = bool(predicate(low)), bool(predicate(high))
    if low_state == high_state:
        raise ValueError("proxy boundary is not bracketed")
    for _ in range(24):
        if abs(float(high["q_rad"]) - float(low["q_rad"])) <= _MAXIMUM_INCREMENT_RAD:
            break
        middle = evaluate(0.5 * (float(low["phase"]) + float(high["phase"])))
        if bool(predicate(middle)) == low_state:
            low = middle
        else:
            high = middle
    return low, high


def _finger_interval(context, seed, base_phases, finger):
    coarse = _coarse_rows(context, seed, base_phases, finger)
    contact_index = next((index for index, row in enumerate(coarse) if row["contact"]), None)
    if contact_index is None or any(not row["safe"] for row in coarse[:contact_index + 1]):
        return None
    def evaluate(phase):
        phases = list(base_phases); phases[finger] = float(phase)
        return _finger_metrics(context, seed, phases, finger)
    expected = coarse[contact_index]
    if contact_index:
        _, expected = _bisect_pair(coarse[contact_index - 1], expected, evaluate,
                                   lambda row: row["contact"])
    failure_index = next((index for index in range(contact_index + 1, len(coarse))
                          if not coarse[index]["safe"]), None)
    if failure_index is None:
        safe_max, first_failure = coarse[-1], None
    else:
        safe_max, first_failure = _bisect_pair(
            coarse[failure_index - 1], coarse[failure_index], evaluate,
            lambda row: row["safe"])
    _, closing_sign = context["active"][finger]
    if closing_sign * (safe_max["q_rad"] - expected["q_rad"]) < -1.0e-12:
        return None
    return {"finger_index": finger + 1, "proxy_q_expected_rad": expected["q_rad"],
            "proxy_q_safe_max_rad": safe_max["q_rad"],
            "proxy_expected_phase": expected["phase"],
            "proxy_safe_max_phase": safe_max["phase"], "closing_sign": closing_sign,
            "expected_region_proxy": expected["region"],
            "expected_object_face_index_proxy": expected["object_face_index"],
            "hard_margin_m": min(expected["hard_margin_m"], safe_max["hard_margin_m"]),
            "table_margin_m": min(expected["table_margin_m"], safe_max["table_margin_m"]),
            "self_margin_m": min(expected["self_margin_m"], safe_max["self_margin_m"]),
            "first_proxy_failure_q_rad": None if first_failure is None else first_failure["q_rad"],
            "safe_boundary_bracket_rad": 0.0 if first_failure is None else
                abs(first_failure["q_rad"] - safe_max["q_rad"])}


def _sequential_intervals(context, seed):
    phases, result = list(seed.pregrasp_closure_phases), []
    for finger in range(3):
        interval = _finger_interval(context, seed, phases, finger)
        if interval is None:
            return {"candidate_id": seed.candidate_id, "status": "PROXY_INTERVAL_REJECT",
                    "reason": f"NO_SAFE_PROXY_CONTACT_INTERVAL_FINGER_{finger + 1}",
                    "finger_intervals": result}
        result.append(interval)
        phases[finger] = interval["proxy_expected_phase"]
    return {"candidate_id": seed.candidate_id, "status": "PROXY_INTERVAL_SURVIVE",
            "reason": "", "finger_intervals": result,
            "final_expected_phases": phases,
            "claim_scope": "PROXY_ONLY_Q_EXPECTED_AND_Q_SAFE_MAX_NOT_FINAL_ANGLES"}


class ProxyContactIntervalEvaluator:
    """Reuse one geometry/FK cache for bounded candidate refinement."""

    def __init__(self, inputs: V2Inputs) -> None:
        self._context = _context(inputs)

    def evaluate_cheap(self, seed: CandidateSeed, specification: Mapping[str, object]):
        row = dict(_cheap_seed(self._context, seed, specification))
        row.pop("_seed", None)
        return row

    def evaluate_interval(self, seed: CandidateSeed):
        return _sequential_intervals(self._context, seed)

    def evaluate(self, seed: CandidateSeed, specification: Mapping[str, object]):
        cheap = self.evaluate_cheap(seed, specification)
        interval = ({"candidate_id": seed.candidate_id,
                     "status": "PROXY_INTERVAL_NOT_EVALUATED",
                     "reason": "CHEAP_HARD_CONSTRAINTS_NOT_SATISFIED",
                     "finger_intervals": []}
                    if cheap["status"] != "NEAR_CONTACT_SEED"
                    else self.evaluate_interval(seed))
        return cheap, interval


def solve_proxy_contact_intervals(
    inputs: V2Inputs,
    seeds: Sequence[CandidateSeed],
    specification_rows: Sequence[Mapping[str, object]],
) -> tuple[tuple[CandidateSeed, ...], dict[str, object]]:
    """Screen all fixed specifications and solve sequential proxies for Top120."""
    specs = tuple(dict(row) for row in specification_rows)
    identifiers = tuple(str(row.get("candidate_id", row.get("spec_id", "")))
                        for row in specs)
    if (len(specs) != _EXPECTED_SPEC_COUNT or any(not value for value in identifiers)
            or len(set(identifiers)) != _EXPECTED_SPEC_COUNT):
        raise ValueError("contactopt requires 1488 unique non-empty specification identifiers")
    seed_by_id = {seed.candidate_id: seed for seed in seeds}
    if len(seed_by_id) != len(seeds):
        raise ValueError("candidate seed identifiers are not unique")
    generated_ids = {identifier for identifier, row in zip(identifiers, specs)
                     if row.get("status", "POSE_GENERATED") == "POSE_GENERATED"}
    if set(seed_by_id) != generated_ids:
        raise ValueError("POSE_GENERATED specifications and candidate seeds differ")
    context, rows = _context(inputs), []
    for identifier, spec in zip(identifiers, specs):
        seed = seed_by_id.get(identifier)
        if seed is None or spec.get("status", "POSE_GENERATED") != "POSE_GENERATED":
            rows.append({"candidate_id": identifier, "status": "CHEAP_REJECT",
                         "reason": str(spec.get("reason", "SEED_GEOMETRY_REJECT"))})
        else:
            rows.append(_cheap_seed(context, seed, spec))
    top = _select_top120([row for row in rows if row["status"] == "NEAR_CONTACT_SEED"])
    intervals = [_sequential_intervals(context, row["_seed"]) for row in top]
    selected = tuple(row["_seed"] for row, interval in zip(top, intervals)
                     if interval["status"] == "PROXY_INTERVAL_SURVIVE")
    def clean(row):
        return {key: ([clean(item) for item in value] if isinstance(value, list) else
                      _finite(value) if isinstance(value, (float, np.floating)) else value)
                for key, value in row.items() if key != "_seed"}
    audit = {"schema_version": "carts_contactopt_proxy_intervals_v1",
             "claim_scope": "CKDTREE_CONTACT_CONVEX_HULL_TABLE_FCL_SELF_SAMPLED_NOT_EXACT_OBJECT_CONTACT",
             "registered_specification_count": len(specs),
             "generated_seed_count": len(seeds), "cheap_rows": [clean(row) for row in rows],
             "near_contact_seed_count": sum(row["status"] == "NEAR_CONTACT_SEED" for row in rows),
             "top120_count": len(top),
             "top120_family_counts": {family: sum(row["family"] == family for row in top)
                                      for family in ("GLOBAL", "DENSE_OPPOSITION")},
             "top120_transferred_slots": {
                 family: max(0, sum(row["family"] == family for row in top) - _FAMILY_LIMIT)
                 for family in ("GLOBAL", "DENSE_OPPOSITION")},
             "interval_rows": intervals,
             "proxy_interval_survivor_count": len(selected),
             "representative_point_count_per_finger": _REPRESENTATIVE_COUNT,
             "coarse_phase_sample_count": _COARSE_PHASE_COUNT,
             "maximum_joint_increment_rad": _MAXIMUM_INCREMENT_RAD,
             "table_operation_clearance_m": context["table_operation_clearance_m"],
             "hardware_authorized": False, "formal_dynamic_pass": False}
    return selected, audit


__all__ = ["ProxyContactIntervalEvaluator", "solve_proxy_contact_intervals"]
