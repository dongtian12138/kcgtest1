"""Deterministic external first-hit visibility for graspable mesh faces.

The predicate implemented here is deliberately geometric and object agnostic.
For every approach direction it launches rays from the exterior supporting
plane of the complete object mesh through a fixed, symmetric set of interior
triangle witnesses.  A face is retained only when at least one of its own
witnesses is tied for the first ray/triangle intersection.  Consequently the
test is conservative for very small visible slivers, but every retained face
has a concrete, auditable exterior first-hit witness.

No simulator collision/contact state, model name, hand-written distance, or
third-party spatial-index package is used.  The broad phase is a deterministic
median-split BVH.  Every geometric tolerance is an IEEE-754 forward-error bound
scaled by the mesh characteristic length and is reported in the audit record.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from kcg_connector.grasp.robust.object_model import ObjectGraspModel


EXTERNAL_FIRST_HIT_METHOD = (
    "CARTS_DIRECTIONAL_TRIANGLE_INTERIOR_WITNESS_FIRST_HIT_BVH_V1"
)
NUMERICAL_POLICY = "IEEE754_BINARY64_FORWARD_ERROR_GAMMA_N_MESH_SCALE_V1"
WITNESS_RULE = "SYMMETRIC_DEGREE2_TRIANGLE_INTERIOR_RULE"
BVH_SPLIT_RULE = "LONGEST_CENTROID_EXTENT_STABLE_MEDIAN"

# This is a computational batching constant, not a physical or acceptance
# threshold.  It changes only how many triangle tests NumPy evaluates together;
# it cannot change the mathematical first-hit predicate.
_BVH_LEAF_FACE_CAPACITY = 8

# Symmetric degree-two triangle rule.  All witnesses are strictly interior, so
# ray ownership never depends on an edge/vertex tie convention.
_BARYCENTRIC_WITNESSES = np.asarray(
    (
        (2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0),
        (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0),
        (1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0),
    ),
    dtype=np.float64,
)
_BARYCENTRIC_WITNESSES.setflags(write=False)


def _gamma(operation_count: int) -> float:
    """Higham-style binary64 accumulated relative-error bound ``gamma_n``."""

    if operation_count <= 0:
        raise ValueError("operation_count must be positive")
    epsilon = np.finfo(np.float64).eps
    product = float(operation_count) * epsilon
    if product >= 1.0:
        raise ValueError("operation_count is too large for a finite gamma_n bound")
    return product / (1.0 - product)


_DIRECTION_NORMALISATION_GAMMA = _gamma(16)
_AABB_GAMMA = _gamma(32)
_DETERMINANT_GAMMA = _gamma(24)
_BARYCENTRIC_GAMMA = _gamma(64)
_DISTANCE_GAMMA = _gamma(128)
_ORIGIN_GAMMA = _gamma(64)


def _readonly_array(
    value: object,
    *,
    dtype: object,
    shape_tail: tuple[int, ...],
    name: str,
) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    tail_start = array.ndim - len(shape_tail)
    if array.ndim < len(shape_tail) or tuple(array.shape[tail_start:]) != shape_tail:
        raise ValueError(f"{name} must end in shape {shape_tail}, got {array.shape}")
    if np.issubdtype(array.dtype, np.floating) and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    array.setflags(write=False)
    return array


def _canonical_directions(
    directions: Sequence[Sequence[float]],
) -> tuple[np.ndarray, int]:
    raw = np.asarray(directions, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1:] != (3,) or len(raw) == 0:
        raise ValueError("approach_directions must be a non-empty finite Nx3 array")
    if not np.all(np.isfinite(raw)):
        raise ValueError("approach_directions contains non-finite values")
    lengths = np.linalg.norm(raw, axis=1)
    if np.any(lengths == 0.0) or not np.all(np.isfinite(lengths)):
        raise ValueError("approach directions must be non-zero finite vectors")
    normalised = raw / lengths[:, None]
    # Canonicalise signed zero before exact duplicate removal and hashing.
    normalised[normalised == 0.0] = 0.0
    unique: dict[bytes, np.ndarray] = {}
    for row in normalised:
        little_endian = np.asarray(row, dtype="<f8")
        unique.setdefault(little_endian.tobytes(order="C"), row.copy())
    ordered = sorted(unique.values(), key=lambda row: tuple(float(value) for value in row))
    result = np.asarray(ordered, dtype=np.float64)
    result.setflags(write=False)
    return result, len(raw)


def directions_from_kinematic_domains(domains: Mapping[str, Any]) -> np.ndarray:
    """Extract closing velocities from hand-derived kinematic normal domains.

    The helper uses a structural interface so this geometry module does not
    depend on a particular hand class.  Domain names are sorted, and magnitude
    is intentionally discarded by the subsequent direction normalisation.
    """

    if not isinstance(domains, Mapping) or not domains:
        raise ValueError("kinematic normal domains must be a non-empty mapping")
    rows: list[Sequence[float]] = []
    for name in sorted(domains, key=str):
        domain = domains[name]
        try:
            velocity = domain.closing_velocity_base_m_s
        except AttributeError as error:
            raise ValueError(
                f"kinematic domain {name!r} has no closing_velocity_base_m_s"
            ) from error
        rows.append(velocity)
    directions, _input_count = _canonical_directions(rows)
    return directions


@dataclass(frozen=True)
class SurfaceVisibilityAudit:
    """JSON-compatible evidence for one complete first-hit evaluation."""

    method_id: str
    numerical_policy: str
    witness_rule: str
    bvh_split_rule: str
    face_count: int
    input_direction_count: int
    unique_direction_count: int
    witness_count_per_face: int
    rays_cast: int
    rays_with_no_hit: int
    rays_with_first_hit_ties: int
    bvh_node_count: int
    bvh_node_visits: int
    ray_triangle_tests: int
    visible_face_count: int
    visible_face_count_by_direction: tuple[int, ...]
    characteristic_length_m: float
    ray_origin_padding_m: float
    distance_error_bound_m: float
    maximum_determinant_error_bound_m2: float
    barycentric_error_bound: float
    bvh_leaf_face_capacity: int
    canonical_directions_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "method_id": self.method_id,
            "numerical_policy": self.numerical_policy,
            "witness_rule": self.witness_rule,
            "bvh_split_rule": self.bvh_split_rule,
            "face_count": self.face_count,
            "input_direction_count": self.input_direction_count,
            "unique_direction_count": self.unique_direction_count,
            "witness_count_per_face": self.witness_count_per_face,
            "rays_cast": self.rays_cast,
            "rays_with_no_hit": self.rays_with_no_hit,
            "rays_with_first_hit_ties": self.rays_with_first_hit_ties,
            "bvh_node_count": self.bvh_node_count,
            "bvh_node_visits": self.bvh_node_visits,
            "ray_triangle_tests": self.ray_triangle_tests,
            "visible_face_count": self.visible_face_count,
            "visible_face_count_by_direction": list(
                self.visible_face_count_by_direction
            ),
            "characteristic_length_m": self.characteristic_length_m,
            "ray_origin_padding_m": self.ray_origin_padding_m,
            "distance_error_bound_m": self.distance_error_bound_m,
            "maximum_determinant_error_bound_m2": (
                self.maximum_determinant_error_bound_m2
            ),
            "barycentric_error_bound": self.barycentric_error_bound,
            "bvh_leaf_face_capacity": self.bvh_leaf_face_capacity,
            "canonical_directions_sha256": self.canonical_directions_sha256,
        }


@dataclass(frozen=True)
class SurfaceVisibilityResult:
    """Boolean face mask paired with its complete numerical audit."""

    face_mask: np.ndarray
    audit: SurfaceVisibilityAudit

    def __post_init__(self) -> None:
        mask = np.array(self.face_mask, dtype=bool, copy=True)
        if mask.ndim != 1 or len(mask) != self.audit.face_count:
            raise ValueError("face_mask must contain one Boolean per audited face")
        if int(np.count_nonzero(mask)) != self.audit.visible_face_count:
            raise ValueError("face_mask count does not match visibility audit")
        mask.setflags(write=False)
        object.__setattr__(self, "face_mask", mask)


@dataclass(frozen=True)
class _BvhNode:
    lower_m: np.ndarray
    upper_m: np.ndarray
    left: int
    right: int
    face_indices: np.ndarray

    @property
    def leaf(self) -> bool:
        return self.left < 0


@dataclass(frozen=True)
class _RayStatistics:
    node_visits: int
    triangle_tests: int
    first_hit_tie_count: int


@dataclass(frozen=True)
class FirstHitResult:
    """Deterministic first intersection of one directed exterior ray.

    ``face_index`` is the smallest mesh-face index among intersections whose
    distances are indistinguishable under the reported binary64 forward-error
    bound.  That tie rule is numerical bookkeeping, not a geometric clearance
    or contact-quality threshold.
    """

    hit: bool
    distance_m: float | None
    face_index: int | None
    position_m: np.ndarray | None
    outward_normal: np.ndarray | None
    first_hit_tie_count: int
    bvh_node_visits: int
    ray_triangle_tests: int
    distance_error_bound_m: float
    numerical_policy: str = NUMERICAL_POLICY

    def __post_init__(self) -> None:
        if self.hit:
            if self.distance_m is None or self.face_index is None:
                raise ValueError("a successful first-hit result needs distance and face")
            if self.position_m is None or self.outward_normal is None:
                raise ValueError("a successful first-hit result needs point and normal")
            position = _readonly_array(
                self.position_m,
                dtype=np.float64,
                shape_tail=(3,),
                name="first-hit position",
            )
            normal = _readonly_array(
                self.outward_normal,
                dtype=np.float64,
                shape_tail=(3,),
                name="first-hit normal",
            )
            object.__setattr__(self, "position_m", position)
            object.__setattr__(self, "outward_normal", normal)
        elif any(
            value is not None
            for value in (
                self.distance_m,
                self.face_index,
                self.position_m,
                self.outward_normal,
            )
        ):
            raise ValueError("a missed ray cannot carry hit geometry")
        if self.first_hit_tie_count < 0:
            raise ValueError("first_hit_tie_count cannot be negative")
        if self.bvh_node_visits < 0 or self.ray_triangle_tests < 0:
            raise ValueError("BVH audit counters cannot be negative")
        if not math.isfinite(self.distance_error_bound_m) or self.distance_error_bound_m < 0.0:
            raise ValueError("distance error bound must be finite and non-negative")


class _TriangleBvh:
    """Small dependency-free BVH with deterministic geometry-derived splits."""

    def __init__(self, triangles_m: np.ndarray) -> None:
        triangles = np.asarray(triangles_m, dtype=np.float64)
        if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
            raise ValueError("triangles_m must have shape (F, 3, 3)")
        self.triangles_m = triangles
        self.edge_one_m = triangles[:, 1] - triangles[:, 0]
        self.edge_two_m = triangles[:, 2] - triangles[:, 0]
        self.centroids_m = np.mean(triangles, axis=1)
        self.face_lower_m = np.min(triangles, axis=1)
        self.face_upper_m = np.max(triangles, axis=1)
        nodes: list[_BvhNode | None] = []

        def build(indices: np.ndarray) -> int:
            node_index = len(nodes)
            nodes.append(None)
            lower = np.min(self.face_lower_m[indices], axis=0)
            upper = np.max(self.face_upper_m[indices], axis=0)
            if len(indices) <= _BVH_LEAF_FACE_CAPACITY:
                leaf_indices = np.array(indices, dtype=np.int64, copy=True)
                leaf_indices.setflags(write=False)
                nodes[node_index] = _BvhNode(lower, upper, -1, -1, leaf_indices)
                return node_index

            centroids = self.centroids_m[indices]
            extent = np.max(centroids, axis=0) - np.min(centroids, axis=0)
            axis = int(np.argmax(extent))
            secondary = (axis + 1) % 3
            tertiary = (axis + 2) % 3
            order = np.lexsort(
                (
                    indices,
                    centroids[:, tertiary],
                    centroids[:, secondary],
                    centroids[:, axis],
                )
            )
            ordered = indices[order]
            middle = len(ordered) // 2
            left = build(ordered[:middle])
            right = build(ordered[middle:])
            empty = np.empty(0, dtype=np.int64)
            empty.setflags(write=False)
            nodes[node_index] = _BvhNode(lower, upper, left, right, empty)
            return node_index

        self.root = build(np.arange(len(triangles), dtype=np.int64))
        self.nodes = tuple(node for node in nodes if node is not None)

    @staticmethod
    def _aabb_entry_distance(
        origin_m: np.ndarray,
        direction: np.ndarray,
        node: _BvhNode,
        *,
        aabb_error_bound_m: float,
        maximum_distance_m: float,
    ) -> float | None:
        entry = -math.inf
        exit_distance = math.inf
        for axis in range(3):
            component = float(direction[axis])
            lower = float(node.lower_m[axis]) - aabb_error_bound_m
            upper = float(node.upper_m[axis]) + aabb_error_bound_m
            origin_component = float(origin_m[axis])
            if component == 0.0:
                if origin_component < lower or origin_component > upper:
                    return None
                continue
            first = (lower - origin_component) / component
            second = (upper - origin_component) / component
            if first > second:
                first, second = second, first
            entry = max(entry, first)
            exit_distance = min(exit_distance, second)
            if exit_distance < entry:
                return None
        if exit_distance < -aabb_error_bound_m:
            return None
        if entry > maximum_distance_m + aabb_error_bound_m:
            return None
        return max(0.0, entry)

    def _leaf_intersections(
        self,
        origin_m: np.ndarray,
        direction: np.ndarray,
        face_indices: np.ndarray,
        *,
        distance_error_bound_m: float,
    ) -> np.ndarray:
        edge_one = self.edge_one_m[face_indices]
        edge_two = self.edge_two_m[face_indices]
        p_vector = np.cross(np.broadcast_to(direction, edge_two.shape), edge_two)
        determinant = np.einsum("ij,ij->i", edge_one, p_vector)
        determinant_magnitude_bound = _DETERMINANT_GAMMA * np.sum(
            np.abs(edge_one * p_vector), axis=1
        )
        usable = np.abs(determinant) > determinant_magnitude_bound
        result = np.full(len(face_indices), np.inf, dtype=np.float64)
        if not np.any(usable):
            return result

        local = np.flatnonzero(usable)
        selected_faces = face_indices[local]
        selected_determinant = determinant[local]
        inverse_determinant = 1.0 / selected_determinant
        offset = origin_m - self.triangles_m[selected_faces, 0]
        first_barycentric = (
            np.einsum("ij,ij->i", offset, p_vector[local]) * inverse_determinant
        )
        q_vector = np.cross(offset, edge_one[local])
        second_barycentric = (
            np.einsum(
                "ij,ij->i", np.broadcast_to(direction, q_vector.shape), q_vector
            )
            * inverse_determinant
        )
        distance = (
            np.einsum("ij,ij->i", edge_two[local], q_vector)
            * inverse_determinant
        )
        valid = (
            (first_barycentric >= -_BARYCENTRIC_GAMMA)
            & (second_barycentric >= -_BARYCENTRIC_GAMMA)
            & (
                first_barycentric + second_barycentric
                <= 1.0 + _BARYCENTRIC_GAMMA
            )
            & (distance >= -distance_error_bound_m)
        )
        result[local[valid]] = distance[valid]
        return result

    def nearest_intersection(
        self,
        origin_m: np.ndarray,
        direction: np.ndarray,
        *,
        aabb_error_bound_m: float,
        distance_error_bound_m: float,
    ) -> tuple[float, int | None, _RayStatistics]:
        best = math.inf
        tied_faces: set[int] = set()
        node_visits = 0
        triangle_tests = 0
        stack = [self.root]
        while stack:
            node_index = stack.pop()
            node = self.nodes[node_index]
            entry = self._aabb_entry_distance(
                origin_m,
                direction,
                node,
                aabb_error_bound_m=aabb_error_bound_m,
                maximum_distance_m=best,
            )
            if entry is None:
                continue
            node_visits += 1
            if node.leaf:
                triangle_tests += len(node.face_indices)
                distances = self._leaf_intersections(
                    origin_m,
                    direction,
                    node.face_indices,
                    distance_error_bound_m=distance_error_bound_m,
                )
                finite = distances[np.isfinite(distances)]
                if len(finite) == 0:
                    continue
                leaf_best = float(np.min(finite))
                if leaf_best < best - distance_error_bound_m:
                    best = leaf_best
                    tied_faces = {
                        int(face)
                        for face, distance in zip(node.face_indices, distances)
                        if math.isfinite(float(distance))
                        and abs(float(distance) - best) <= distance_error_bound_m
                    }
                elif abs(leaf_best - best) <= distance_error_bound_m:
                    tied_faces.update(
                        int(face)
                        for face, distance in zip(node.face_indices, distances)
                        if math.isfinite(float(distance))
                        and abs(float(distance) - best) <= distance_error_bound_m
                    )
                continue

            children: list[tuple[float, int]] = []
            for child_index in (node.left, node.right):
                child_entry = self._aabb_entry_distance(
                    origin_m,
                    direction,
                    self.nodes[child_index],
                    aabb_error_bound_m=aabb_error_bound_m,
                    maximum_distance_m=best,
                )
                if child_entry is not None:
                    children.append((child_entry, child_index))
            # Stack is LIFO: reverse sorting visits the nearest child first;
            # node index makes exact distance ties deterministic.
            children.sort(key=lambda value: (value[0], value[1]), reverse=True)
            stack.extend(index for _entry, index in children)
        face_index = min(tied_faces) if tied_faces else None
        return best, face_index, _RayStatistics(
            node_visits, triangle_tests, len(tied_faces)
        )


def _mesh_characteristic_length(vertices_m: np.ndarray) -> tuple[np.ndarray, float]:
    """Return a translation-free centre and rotation-invariant diameter bound."""

    centre = np.mean(vertices_m, axis=0)
    radius = float(np.max(np.linalg.norm(vertices_m - centre, axis=1)))
    characteristic_length = 2.0 * radius
    if not math.isfinite(characteristic_length) or characteristic_length <= 0.0:
        raise ValueError("mesh must have a positive finite characteristic length")
    return centre, characteristic_length


def _canonical_directions_digest(directions: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(b"CARTS_CANONICAL_APPROACH_DIRECTIONS_V1\0")
    digest.update(np.asarray(directions.shape, dtype="<i8").tobytes(order="C"))
    digest.update(np.asarray(directions, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


class TriangleFirstHitIntersector:
    """Reusable deterministic BVH for object-surface first-hit queries.

    The mesh is recentered before BVH construction so translations do not
    inflate floating-point error bounds.  Callers provide physical ray origins
    and approach directions; the class neither assumes a connector shape nor
    invents a maximum travel distance.
    """

    def __init__(self, model: ObjectGraspModel) -> None:
        if not isinstance(model, ObjectGraspModel):
            raise TypeError("model must be an ObjectGraspModel")
        centre, characteristic_length = _mesh_characteristic_length(
            model.mesh.vertices_m
        )
        self.model = model
        self.centre_m = np.asarray(centre, dtype=np.float64)
        self.characteristic_length_m = characteristic_length
        self.distance_error_bound_m = _DISTANCE_GAMMA * (
            2.0 * characteristic_length
            + _ORIGIN_GAMMA * characteristic_length
        )
        self.aabb_error_bound_m = _AABB_GAMMA * characteristic_length
        self._bvh = _TriangleBvh(model.mesh.face_vertices_m - self.centre_m)

    @property
    def bvh_node_count(self) -> int:
        return len(self._bvh.nodes)

    def first_hit(
        self,
        origin_m: Sequence[float],
        approach_direction: Sequence[float],
    ) -> FirstHitResult:
        origin = np.asarray(origin_m, dtype=np.float64)
        direction = np.asarray(approach_direction, dtype=np.float64)
        if origin.shape != (3,) or not np.all(np.isfinite(origin)):
            raise ValueError("ray origin must be one finite three-vector")
        if direction.shape != (3,) or not np.all(np.isfinite(direction)):
            raise ValueError("approach direction must be one finite three-vector")
        direction_norm = float(np.linalg.norm(direction))
        if not math.isfinite(direction_norm) or direction_norm == 0.0:
            raise ValueError("approach direction must be non-zero")
        unit_direction = direction / direction_norm
        distance, face_index, statistics = self._bvh.nearest_intersection(
            origin - self.centre_m,
            unit_direction,
            aabb_error_bound_m=self.aabb_error_bound_m,
            distance_error_bound_m=self.distance_error_bound_m,
        )
        if not math.isfinite(distance) or face_index is None:
            return FirstHitResult(
                hit=False,
                distance_m=None,
                face_index=None,
                position_m=None,
                outward_normal=None,
                first_hit_tie_count=statistics.first_hit_tie_count,
                bvh_node_visits=statistics.node_visits,
                ray_triangle_tests=statistics.triangle_tests,
                distance_error_bound_m=self.distance_error_bound_m,
            )
        position = origin + distance * unit_direction
        return FirstHitResult(
            hit=True,
            distance_m=float(distance),
            face_index=int(face_index),
            position_m=position,
            outward_normal=self.model.mesh.face_normals[int(face_index)],
            first_hit_tie_count=statistics.first_hit_tie_count,
            bvh_node_visits=statistics.node_visits,
            ray_triangle_tests=statistics.triangle_tests,
            distance_error_bound_m=self.distance_error_bound_m,
        )

    def first_hits(
        self,
        origins_m: Sequence[Sequence[float]],
        approach_directions: Sequence[Sequence[float]],
    ) -> tuple[FirstHitResult, ...]:
        origins = np.asarray(origins_m, dtype=np.float64)
        directions = np.asarray(approach_directions, dtype=np.float64)
        if origins.ndim != 2 or origins.shape[1:] != (3,):
            raise ValueError("origins_m must have shape (R, 3)")
        if directions.shape != origins.shape:
            raise ValueError("approach_directions must match origins_m")
        return tuple(
            self.first_hit(origin, direction)
            for origin, direction in zip(origins, directions)
        )


def external_first_hit_face_visibility(
    model: ObjectGraspModel,
    approach_directions: Sequence[Sequence[float]],
) -> SurfaceVisibilityResult:
    """Evaluate conservative exterior first-hit visibility for every mesh face.

    ``approach_directions`` point from free space toward the object, matching
    PAD closing velocity.  A face is visible if any fixed interior witness is
    a first intersection for at least one canonicalised direction.  All mesh
    faces act as occluders, including faces forbidden for force transmission.
    """

    if not isinstance(model, ObjectGraspModel):
        raise TypeError("model must be an ObjectGraspModel")
    directions, input_direction_count = _canonical_directions(approach_directions)
    triangles_world = model.mesh.face_vertices_m
    centre, characteristic_length = _mesh_characteristic_length(model.mesh.vertices_m)
    triangles = triangles_world - centre
    vertices = model.mesh.vertices_m - centre
    bvh = _TriangleBvh(triangles)

    ray_origin_padding = _ORIGIN_GAMMA * characteristic_length
    distance_error_bound = _DISTANCE_GAMMA * (
        2.0 * characteristic_length + ray_origin_padding
    )
    aabb_error_bound = _AABB_GAMMA * characteristic_length
    edge_one = triangles[:, 1] - triangles[:, 0]
    edge_two = triangles[:, 2] - triangles[:, 0]
    maximum_determinant_error_bound = float(
        _DETERMINANT_GAMMA
        * np.max(np.linalg.norm(edge_one, axis=1) * np.linalg.norm(edge_two, axis=1))
    )

    witness_positions = np.einsum(
        "wk,fkj->fwj", _BARYCENTRIC_WITNESSES, triangles
    )
    normals = model.mesh.face_normals
    visible = np.zeros(len(triangles), dtype=bool)
    visible_by_direction: list[int] = []
    rays_cast = 0
    rays_with_no_hit = 0
    rays_with_first_hit_ties = 0
    node_visits = 0
    triangle_tests = 0

    for direction in directions:
        direction_visible = np.zeros(len(triangles), dtype=bool)
        projected_vertices = vertices @ direction
        minimum_projection = float(np.min(projected_vertices))
        outside_projection = np.nextafter(
            minimum_projection - ray_origin_padding, -math.inf
        )
        # A face parallel to a ray has no proper entry intersection.  The bound
        # is solely dot-product normalisation roundoff, not an angular gate.
        proper_crossing = (
            np.abs(normals @ direction) > _DIRECTION_NORMALISATION_GAMMA
        )
        for face_index in np.flatnonzero(proper_crossing):
            for witness in witness_positions[face_index]:
                witness_projection = float(witness @ direction)
                target_distance = witness_projection - outside_projection
                origin = witness + (outside_projection - witness_projection) * direction
                nearest, _nearest_face, ray_stats = bvh.nearest_intersection(
                    origin,
                    direction,
                    aabb_error_bound_m=aabb_error_bound,
                    distance_error_bound_m=distance_error_bound,
                )
                rays_cast += 1
                node_visits += ray_stats.node_visits
                triangle_tests += ray_stats.triangle_tests
                if not math.isfinite(nearest):
                    rays_with_no_hit += 1
                    continue
                if ray_stats.first_hit_tie_count > 1:
                    rays_with_first_hit_ties += 1
                if abs(nearest - target_distance) <= distance_error_bound:
                    direction_visible[face_index] = True
                    break
        visible |= direction_visible
        visible_by_direction.append(int(np.count_nonzero(direction_visible)))

    visible.setflags(write=False)
    audit = SurfaceVisibilityAudit(
        method_id=EXTERNAL_FIRST_HIT_METHOD,
        numerical_policy=NUMERICAL_POLICY,
        witness_rule=WITNESS_RULE,
        bvh_split_rule=BVH_SPLIT_RULE,
        face_count=len(triangles),
        input_direction_count=input_direction_count,
        unique_direction_count=len(directions),
        witness_count_per_face=len(_BARYCENTRIC_WITNESSES),
        rays_cast=rays_cast,
        rays_with_no_hit=rays_with_no_hit,
        rays_with_first_hit_ties=rays_with_first_hit_ties,
        bvh_node_count=len(bvh.nodes),
        bvh_node_visits=node_visits,
        ray_triangle_tests=triangle_tests,
        visible_face_count=int(np.count_nonzero(visible)),
        visible_face_count_by_direction=tuple(visible_by_direction),
        characteristic_length_m=characteristic_length,
        ray_origin_padding_m=ray_origin_padding,
        distance_error_bound_m=distance_error_bound,
        maximum_determinant_error_bound_m2=maximum_determinant_error_bound,
        barycentric_error_bound=_BARYCENTRIC_GAMMA,
        bvh_leaf_face_capacity=_BVH_LEAF_FACE_CAPACITY,
        canonical_directions_sha256=_canonical_directions_digest(directions),
    )
    return SurfaceVisibilityResult(face_mask=visible, audit=audit)


@dataclass(frozen=True)
class DirectionalFirstHitVisibilityPredicate:
    """Callable adapter accepted directly by ``surface_sampling`` APIs."""

    approach_directions: np.ndarray
    input_direction_count: int = 0

    def __init__(self, approach_directions: Sequence[Sequence[float]]) -> None:
        canonical, input_count = _canonical_directions(approach_directions)
        object.__setattr__(self, "approach_directions", canonical)
        object.__setattr__(self, "input_direction_count", input_count)

    @classmethod
    def from_kinematic_domains(
        cls, domains: Mapping[str, Any]
    ) -> "DirectionalFirstHitVisibilityPredicate":
        return cls(directions_from_kinematic_domains(domains))

    def evaluate(self, model: ObjectGraspModel) -> SurfaceVisibilityResult:
        result = external_first_hit_face_visibility(model, self.approach_directions)
        if self.input_direction_count == result.audit.input_direction_count:
            return result
        # Preserve how many hand-domain rows were supplied even when exact
        # duplicate directions collapsed in the canonical representation.
        values = result.audit.as_dict()
        values["input_direction_count"] = self.input_direction_count
        values["visible_face_count_by_direction"] = tuple(
            result.audit.visible_face_count_by_direction
        )
        audit = SurfaceVisibilityAudit(**values)  # type: ignore[arg-type]
        return SurfaceVisibilityResult(result.face_mask, audit)

    def __call__(self, model: ObjectGraspModel) -> np.ndarray:
        return self.evaluate(model).face_mask

    @property
    def contract(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "method_id": EXTERNAL_FIRST_HIT_METHOD,
                "numerical_policy": NUMERICAL_POLICY,
                "witness_rule": WITNESS_RULE,
                "bvh_split_rule": BVH_SPLIT_RULE,
                "input_direction_count": self.input_direction_count,
                "unique_direction_count": len(self.approach_directions),
                "canonical_directions_sha256": _canonical_directions_digest(
                    self.approach_directions
                ),
            }
        )


__all__ = [
    "BVH_SPLIT_RULE",
    "DirectionalFirstHitVisibilityPredicate",
    "EXTERNAL_FIRST_HIT_METHOD",
    "FirstHitResult",
    "NUMERICAL_POLICY",
    "SurfaceVisibilityAudit",
    "SurfaceVisibilityResult",
    "TriangleFirstHitIntersector",
    "WITNESS_RULE",
    "directions_from_kinematic_domains",
    "external_first_hit_face_visibility",
]
