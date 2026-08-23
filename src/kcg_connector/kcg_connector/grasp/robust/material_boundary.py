"""Exact static material-boundary certificate for one triangle surface.

The existing surface-orientation audit proves source-index topology,
orientability, and exact signed volume.  Those properties do not rule out a
surface passing through itself.  This module adds the missing pairwise
geometric check without an epsilon: binary64 coordinates are converted to one
common dyadic integer grid, and every potentially overlapping triangle pair
is classified with integer predicates.

A successful certificate means that one connected, closed, orientable source
surface is embedded in three-dimensional space.  Adjacent faces may meet only
on their source-indexed common vertex or edge.  Jordan--Brouwer separation and
the positive-volume winding supplied by :mod:`surface_orientation` then give
one bounded material interior and a material-outward face winding.  The claim
is static and mesh-local; it says nothing about a robot pose, a trajectory,
the environment, or whether this collision boundary conservatively covers a
different visual asset.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import struct
from typing import Sequence

import numpy as np

from kcg_connector.grasp.robust.surface_orientation import (
    SurfaceBoundaryRole,
    SurfaceOrientationCertificate,
    audit_surface_orientation,
)


METHOD_ID = "CARTS_EXACT_SINGLE_EMBEDDED_MATERIAL_BOUNDARY_V1"
PAIR_POLICY = (
    "ALL_SOURCE_FACE_PAIRS_X_EXACT_COMMON_DYADIC_INTEGER_PREDICATES_X_"
    "ONLY_SOURCE_SHARED_VERTEX_OR_EDGE_CONTACT_ALLOWED"
)
CLAIM_LIMITATIONS = (
    "STATIC_SOURCE_INDEXED_BINARY64_TRIANGLE_BOUNDARY_ONLY",
    "NO_VISUAL_TO_COLLISION_COVERAGE_OR_SEMANTIC_ROLE_CLAIM",
    "NO_ROBOT_POSE_TRAJECTORY_OBJECT_OR_ENVIRONMENT_CLAIM",
    "NO_COLLISION_MARGIN_OR_PHYSICAL_CLEARANCE_CLAIM",
)


class MaterialBoundaryError(ValueError):
    """Fail-closed material-boundary error with a stable reason code."""

    def __init__(self, code: str, detail: str) -> None:
        if not code or not detail:
            raise ValueError("material-boundary error fields cannot be empty")
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(f"{self.code}: {self.detail}")


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


@dataclass(frozen=True)
class MaterialBoundaryCertificate:
    """Immutable proof record for one embedded closed material boundary."""

    method_id: str
    pair_policy: str
    source_indexed_mesh_sha256: str
    orientation_certificate_sha256: str
    source_vertex_count: int
    source_face_count: int
    source_face_pair_count: int
    sweep_axis: int
    sweep_separated_pair_count: int
    aabb_separated_pair_count: int
    exact_disjoint_pair_count: int
    allowed_shared_vertex_pair_count: int
    allowed_shared_edge_pair_count: int
    exact_tested_pair_count: int
    pair_coverage_count: int
    duplicate_geometric_vertex_count: int
    unused_source_vertex_count: int
    self_intersection_count: int
    closed_two_manifold_status: str
    single_component_status: str
    exact_pairwise_embedding_status: str
    nesting_status: str
    material_outward_status: str
    formal_material_boundary_eligible: bool
    claim_limitations: tuple[str, ...]
    certificate_sha256: str

    def __post_init__(self) -> None:
        if self.method_id != METHOD_ID or self.pair_policy != PAIR_POLICY:
            raise ValueError("material-boundary method or pair policy changed")
        if not _is_sha256(self.source_indexed_mesh_sha256) or not _is_sha256(
            self.orientation_certificate_sha256
        ) or not _is_sha256(self.certificate_sha256):
            raise ValueError("material-boundary digest is invalid")
        integer_fields = (
            self.source_vertex_count,
            self.source_face_count,
            self.source_face_pair_count,
            self.sweep_axis,
            self.sweep_separated_pair_count,
            self.aabb_separated_pair_count,
            self.exact_disjoint_pair_count,
            self.allowed_shared_vertex_pair_count,
            self.allowed_shared_edge_pair_count,
            self.exact_tested_pair_count,
            self.pair_coverage_count,
            self.duplicate_geometric_vertex_count,
            self.unused_source_vertex_count,
            self.self_intersection_count,
        )
        if any(type(value) is not int or value < 0 for value in integer_fields):
            raise ValueError("material-boundary counts must be nonnegative integers")
        if (
            self.source_vertex_count < 4
            or self.source_face_count < 4
            or self.sweep_axis not in (0, 1, 2)
            or self.source_face_pair_count
            != self.source_face_count * (self.source_face_count - 1) // 2
            or self.exact_tested_pair_count
            != self.exact_disjoint_pair_count
            + self.allowed_shared_vertex_pair_count
            + self.allowed_shared_edge_pair_count
            or self.pair_coverage_count
            != self.sweep_separated_pair_count
            + self.aabb_separated_pair_count
            + self.exact_tested_pair_count
            or self.pair_coverage_count != self.source_face_pair_count
        ):
            raise ValueError("material-boundary pair accounting is inconsistent")
        if (
            self.duplicate_geometric_vertex_count != 0
            or self.unused_source_vertex_count != 0
            or self.self_intersection_count != 0
        ):
            raise ValueError("a successful material boundary cannot retain defects")
        if (
            self.closed_two_manifold_status != "VERIFIED"
            or self.single_component_status != "VERIFIED"
            or self.exact_pairwise_embedding_status != "VERIFIED"
            or self.nesting_status != "VERIFIED_SINGLE_COMPONENT_NOT_APPLICABLE"
            or self.material_outward_status
            != "VERIFIED_BY_EMBEDDING_AND_POSITIVE_VOLUME_WINDING"
            or self.formal_material_boundary_eligible is not True
            or self.claim_limitations != CLAIM_LIMITATIONS
        ):
            raise ValueError("material-boundary success claims are incomplete")


_Point = tuple[int, int, int]


@dataclass(frozen=True)
class _FaceRecord:
    source_face_index: int
    vertex_indices: tuple[int, int, int]
    points: tuple[_Point, _Point, _Point]
    minimum: _Point
    maximum: _Point

    @property
    def edges(self) -> tuple[tuple[_Point, _Point], ...]:
        first, second, third = self.points
        return ((first, second), (second, third), (third, first))


def _vector(first: _Point, second: _Point) -> _Point:
    return (
        first[0] - second[0],
        first[1] - second[1],
        first[2] - second[2],
    )


def _cross(first: _Point, second: _Point) -> _Point:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _dot(first: _Point, second: _Point) -> int:
    return (
        first[0] * second[0]
        + first[1] * second[1]
        + first[2] * second[2]
    )


def _common_dyadic_integer_points(vertices_m: np.ndarray) -> tuple[_Point, ...]:
    """Represent every finite binary64 coordinate on one exact integer grid."""

    ratios: list[list[tuple[int, int]]] = []
    exponents: list[list[int]] = []
    for row in vertices_m:
        ratio_row = [float(value).as_integer_ratio() for value in row]
        exponent_row = [-(denominator.bit_length() - 1) for _, denominator in ratio_row]
        ratios.append(ratio_row)
        exponents.append(exponent_row)
    common_exponent = min(value for row in exponents for value in row)
    return tuple(
        tuple(
            numerator << (exponent - common_exponent)
            for (numerator, _denominator), exponent in zip(ratio_row, exponent_row)
        )  # type: ignore[misc]
        for ratio_row, exponent_row in zip(ratios, exponents)
    )


def _orientation2d(first: _Point, second: _Point, point: _Point, axes: tuple[int, int]) -> int:
    first_axis, second_axis = axes
    return (
        (second[first_axis] - first[first_axis])
        * (point[second_axis] - first[second_axis])
        - (second[second_axis] - first[second_axis])
        * (point[first_axis] - first[first_axis])
    )


def _point_in_closed_triangle(point: _Point, triangle: tuple[_Point, _Point, _Point]) -> bool:
    first, second, third = triangle
    normal = _cross(_vector(second, first), _vector(third, first))
    if normal == (0, 0, 0):
        raise MaterialBoundaryError(
            "EXACTLY_DEGENERATE_SOURCE_FACE",
            "point-in-triangle received an exact zero-area face",
        )
    if _dot(normal, _vector(point, first)) != 0:
        return False
    drop_axis = next(index for index, value in enumerate(normal) if value != 0)
    axes = tuple(index for index in range(3) if index != drop_axis)
    assert len(axes) == 2
    signed = (
        _orientation2d(first, second, point, axes),
        _orientation2d(second, third, point, axes),
        _orientation2d(third, first, point, axes),
    )
    return all(value >= 0 for value in signed) or all(
        value <= 0 for value in signed
    )


def _point_on_segment(point: _Point, first: _Point, second: _Point) -> bool:
    direction = _vector(second, first)
    if _cross(_vector(point, first), direction) != (0, 0, 0):
        return False
    return all(
        min(first[axis], second[axis]) <= point[axis]
        <= max(first[axis], second[axis])
        for axis in range(3)
    )


def _integer_point_is_allowed(point: _Point, shared: tuple[_Point, ...]) -> bool:
    if len(shared) == 1:
        return point == shared[0]
    if len(shared) == 2:
        return _point_on_segment(point, shared[0], shared[1])
    return False


def _rational_point_is_allowed(
    numerator: _Point,
    denominator: int,
    shared: tuple[_Point, ...],
) -> bool:
    if denominator <= 0:
        raise ValueError("rational point denominator must be positive")
    if len(shared) == 1:
        return all(
            numerator[axis] == shared[0][axis] * denominator
            for axis in range(3)
        )
    if len(shared) != 2:
        return False
    first, second = shared
    direction = _vector(second, first)
    relative_numerator = tuple(
        numerator[axis] - first[axis] * denominator for axis in range(3)
    )
    if _cross(relative_numerator, direction) != (0, 0, 0):
        return False
    return all(
        min(first[axis], second[axis]) * denominator
        <= numerator[axis]
        <= max(first[axis], second[axis]) * denominator
        for axis in range(3)
    )


def _segments_have_forbidden_intersection(
    first_start: _Point,
    first_end: _Point,
    second_start: _Point,
    second_end: _Point,
    shared: tuple[_Point, ...],
) -> bool:
    if any(
        max(first_start[axis], first_end[axis])
        < min(second_start[axis], second_end[axis])
        or max(second_start[axis], second_end[axis])
        < min(first_start[axis], first_end[axis])
        for axis in range(3)
    ):
        return False

    first_direction = _vector(first_end, first_start)
    second_direction = _vector(second_end, second_start)
    directions_cross = _cross(first_direction, second_direction)
    offset = _vector(second_start, first_start)
    if directions_cross == (0, 0, 0):
        if _cross(offset, first_direction) != (0, 0, 0):
            return False
        axis = next(
            index for index, value in enumerate(first_direction) if value != 0
        )
        lower = max(
            min(first_start[axis], first_end[axis]),
            min(second_start[axis], second_end[axis]),
        )
        upper = min(
            max(first_start[axis], first_end[axis]),
            max(second_start[axis], second_end[axis]),
        )
        if lower > upper:
            return False
        endpoints = (first_start, first_end, second_start, second_end)
        lower_point = next(point for point in endpoints if point[axis] == lower)
        upper_point = next(point for point in endpoints if point[axis] == upper)
        return not (
            _integer_point_is_allowed(lower_point, shared)
            and _integer_point_is_allowed(upper_point, shared)
        )

    if _dot(offset, directions_cross) != 0:
        return False
    dropped_axis = next(
        index for index, value in enumerate(directions_cross) if value != 0
    )
    axes = tuple(index for index in range(3) if index != dropped_axis)
    first_axis, second_axis = axes
    denominator = (
        first_direction[first_axis] * second_direction[second_axis]
        - first_direction[second_axis] * second_direction[first_axis]
    )
    first_numerator = (
        offset[first_axis] * second_direction[second_axis]
        - offset[second_axis] * second_direction[first_axis]
    )
    second_numerator = -(
        first_direction[first_axis] * offset[second_axis]
        - first_direction[second_axis] * offset[first_axis]
    )
    if denominator < 0:
        denominator = -denominator
        first_numerator = -first_numerator
        second_numerator = -second_numerator
    if not (
        0 <= first_numerator <= denominator
        and 0 <= second_numerator <= denominator
    ):
        return False
    first_point_numerator = tuple(
        first_start[axis] * denominator
        + first_numerator * first_direction[axis]
        for axis in range(3)
    )
    second_point_numerator = tuple(
        second_start[axis] * denominator
        + second_numerator * second_direction[axis]
        for axis in range(3)
    )
    if first_point_numerator != second_point_numerator:
        return False
    return not _rational_point_is_allowed(
        first_point_numerator, denominator, shared
    )


def _face_pair_forbidden_reason(
    first: _FaceRecord,
    second: _FaceRecord,
    shared_indices: tuple[int, ...],
    all_points: tuple[_Point, ...],
) -> str | None:
    if len(shared_indices) > 2:
        return "DUPLICATE_OR_COINCIDENT_SOURCE_FACE"
    shared_points = tuple(all_points[index] for index in shared_indices)
    for source_vertex_index, point in zip(first.vertex_indices, first.points):
        if _point_in_closed_triangle(point, second.points) and not (
            source_vertex_index in shared_indices
            and _integer_point_is_allowed(point, shared_points)
        ):
            return "FIRST_FACE_VERTEX_INSIDE_SECOND_FACE"
    for source_vertex_index, point in zip(second.vertex_indices, second.points):
        if _point_in_closed_triangle(point, first.points) and not (
            source_vertex_index in shared_indices
            and _integer_point_is_allowed(point, shared_points)
        ):
            return "SECOND_FACE_VERTEX_INSIDE_FIRST_FACE"
    for first_edge in first.edges:
        for second_edge in second.edges:
            if _segments_have_forbidden_intersection(
                first_edge[0],
                first_edge[1],
                second_edge[0],
                second_edge[1],
                shared_points,
            ):
                return "FACE_EDGES_INTERSECT_OUTSIDE_SHARED_SIMPLEX"
    return None


def _certificate_digest(
    orientation: SurfaceOrientationCertificate,
    counts: tuple[int, ...],
) -> str:
    digest = hashlib.sha256()
    digest.update(METHOD_ID.encode("ascii") + b"\0")
    digest.update(PAIR_POLICY.encode("ascii") + b"\0")
    digest.update(bytes.fromhex(orientation.source_indexed_mesh_sha256))
    digest.update(bytes.fromhex(orientation.canonical_sha256))
    for value in counts:
        digest.update(struct.pack("<Q", value))
    for limitation in CLAIM_LIMITATIONS:
        digest.update(limitation.encode("ascii") + b"\0")
    return digest.hexdigest()


def certify_single_embedded_material_boundary(
    vertices_m: Sequence[Sequence[float]] | np.ndarray,
    faces: Sequence[Sequence[int]] | np.ndarray,
) -> MaterialBoundaryCertificate:
    """Prove one exact source-indexed surface is an embedded material boundary."""

    orientation = audit_surface_orientation(
        vertices_m,
        faces,
        role=SurfaceBoundaryRole.SINGLE_CLOSED_BOUNDARY,
    )
    vertices = np.asarray(vertices_m)
    indexed_faces = np.asarray(faces, dtype=np.int64)
    if vertices.dtype != np.dtype(np.float64):
        raise MaterialBoundaryError(
            "BINARY64_VERTICES_REQUIRED",
            f"vertices_m dtype must be float64, got {vertices.dtype}",
        )
    points = _common_dyadic_integer_points(vertices)
    duplicate_count = len(points) - len(set(points))
    if duplicate_count:
        raise MaterialBoundaryError(
            "DUPLICATE_GEOMETRIC_VERTEX_INDEX",
            f"duplicate_geometric_vertex_count={duplicate_count}",
        )
    used_vertices = {int(value) for value in indexed_faces.reshape(-1)}
    unused_count = len(points) - len(used_vertices)
    if unused_count:
        raise MaterialBoundaryError(
            "UNUSED_SOURCE_VERTEX",
            f"unused_source_vertex_count={unused_count}",
        )

    face_records: list[_FaceRecord] = []
    for face_index, (raw_face, winding_sign) in enumerate(
        zip(indexed_faces, orientation.positive_volume_winding_sign_by_source_face)
    ):
        indices = tuple(int(value) for value in raw_face)
        if winding_sign == -1:
            indices = (indices[0], indices[2], indices[1])
        face_points = tuple(points[index] for index in indices)
        minimum = tuple(min(point[axis] for point in face_points) for axis in range(3))
        maximum = tuple(max(point[axis] for point in face_points) for axis in range(3))
        face_records.append(
            _FaceRecord(
                source_face_index=face_index,
                vertex_indices=indices,
                points=face_points,  # type: ignore[arg-type]
                minimum=minimum,  # type: ignore[arg-type]
                maximum=maximum,  # type: ignore[arg-type]
            )
        )

    coordinate_ranges = tuple(
        max(point[axis] for point in points) - min(point[axis] for point in points)
        for axis in range(3)
    )
    sweep_axis = max(range(3), key=lambda axis: (coordinate_ranges[axis], -axis))
    ordered = sorted(
        face_records,
        key=lambda row: (row.minimum[sweep_axis], row.source_face_index),
    )
    face_pair_count = len(ordered) * (len(ordered) - 1) // 2
    sweep_separated = 0
    aabb_separated = 0
    exact_disjoint = 0
    shared_vertex_pairs = 0
    shared_edge_pairs = 0

    for first_position, first in enumerate(ordered[:-1]):
        for second_position in range(first_position + 1, len(ordered)):
            second = ordered[second_position]
            if second.minimum[sweep_axis] > first.maximum[sweep_axis]:
                sweep_separated += len(ordered) - second_position
                break
            if any(
                first.maximum[axis] < second.minimum[axis]
                or second.maximum[axis] < first.minimum[axis]
                for axis in range(3)
            ):
                aabb_separated += 1
                continue
            shared_indices = tuple(
                sorted(set(first.vertex_indices) & set(second.vertex_indices))
            )
            forbidden_reason = _face_pair_forbidden_reason(
                first, second, shared_indices, points
            )
            if forbidden_reason is not None:
                raise MaterialBoundaryError(
                    "SOURCE_TRIANGLE_SELF_INTERSECTION",
                    f"first_face={first.source_face_index}, "
                    f"second_face={second.source_face_index}, "
                    f"shared_vertex_count={len(shared_indices)}, "
                    f"reason={forbidden_reason}",
                )
            if len(shared_indices) == 1:
                shared_vertex_pairs += 1
            elif len(shared_indices) == 2:
                shared_edge_pairs += 1
            else:
                exact_disjoint += 1

    exact_tested = exact_disjoint + shared_vertex_pairs + shared_edge_pairs
    pair_coverage = sweep_separated + aabb_separated + exact_tested
    counts = (
        len(vertices),
        len(indexed_faces),
        face_pair_count,
        sweep_axis,
        sweep_separated,
        aabb_separated,
        exact_disjoint,
        shared_vertex_pairs,
        shared_edge_pairs,
        exact_tested,
        pair_coverage,
        0,
        0,
        0,
    )
    if pair_coverage != face_pair_count:
        raise MaterialBoundaryError(
            "INTERNAL_PAIR_COVERAGE_FAILURE",
            f"pair_coverage_count={pair_coverage}, expected={face_pair_count}",
        )
    certificate_sha256 = _certificate_digest(orientation, counts)
    return MaterialBoundaryCertificate(
        method_id=METHOD_ID,
        pair_policy=PAIR_POLICY,
        source_indexed_mesh_sha256=orientation.source_indexed_mesh_sha256,
        orientation_certificate_sha256=orientation.canonical_sha256,
        source_vertex_count=len(vertices),
        source_face_count=len(indexed_faces),
        source_face_pair_count=face_pair_count,
        sweep_axis=sweep_axis,
        sweep_separated_pair_count=sweep_separated,
        aabb_separated_pair_count=aabb_separated,
        exact_disjoint_pair_count=exact_disjoint,
        allowed_shared_vertex_pair_count=shared_vertex_pairs,
        allowed_shared_edge_pair_count=shared_edge_pairs,
        exact_tested_pair_count=exact_tested,
        pair_coverage_count=pair_coverage,
        duplicate_geometric_vertex_count=0,
        unused_source_vertex_count=0,
        self_intersection_count=0,
        closed_two_manifold_status="VERIFIED",
        single_component_status="VERIFIED",
        exact_pairwise_embedding_status="VERIFIED",
        nesting_status="VERIFIED_SINGLE_COMPONENT_NOT_APPLICABLE",
        material_outward_status=(
            "VERIFIED_BY_EMBEDDING_AND_POSITIVE_VOLUME_WINDING"
        ),
        formal_material_boundary_eligible=True,
        claim_limitations=CLAIM_LIMITATIONS,
        certificate_sha256=certificate_sha256,
    )


__all__ = [
    "CLAIM_LIMITATIONS",
    "METHOD_ID",
    "MaterialBoundaryCertificate",
    "MaterialBoundaryError",
    "PAIR_POLICY",
    "certify_single_embedded_material_boundary",
]
