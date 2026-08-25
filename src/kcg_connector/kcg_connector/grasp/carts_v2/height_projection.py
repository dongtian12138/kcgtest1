"""Pure numerical height bounds for sampled hand paths and contact predicates."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


HeightInterval = tuple[float, float]


@dataclass(frozen=True)
class TableHeightRequirement:
    """Lowest hand-base Z supported by a sampled finite-table query."""

    minimum_handbase_z_m: float | None
    minimum_relative_z_m: float | None
    contributing_state_index: int | None
    contributing_primitive_index: int | None
    contributing_vertex_index: int | None
    overlapping_primitive_count: int
    geometry_kind: str
    evidence_scope: str = "SAMPLED_PATH_CONSERVATIVE_FINITE_TABLE_NOT_CONTINUOUS"


@dataclass(frozen=True)
class HeightProjection:
    """Nearest deterministic projection into a non-empty feasible set."""

    original_height_m: float
    projected_height_m: float
    translation_world_z_m: float
    selected_interval_m: HeightInterval


def _finite_scalar(value: float, label: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite")
    return parsed


def translate_transform_world_z(transform: np.ndarray, delta_z_m: float) -> np.ndarray:
    """Translate a homogeneous pose along world +Z by left multiplication."""

    matrix = np.asarray(transform, dtype=np.float64)
    delta = _finite_scalar(delta_z_m, "world Z translation")
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("transform must be one finite 4x4 matrix")
    if not np.allclose(
        matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-12, rtol=0.0
    ):
        raise ValueError("transform must have a homogeneous final row")
    translation = np.eye(4, dtype=np.float64)
    translation[2, 3] = delta
    return translation @ matrix


def _world_oriented_geometry(
    sampled_geometry_handbase_m: np.ndarray,
    world_from_handbase_rotation: np.ndarray,
    handbase_world_xy_m: Sequence[float],
) -> tuple[np.ndarray, str]:
    values = np.asarray(sampled_geometry_handbase_m, dtype=np.float64)
    rotation = np.asarray(world_from_handbase_rotation, dtype=np.float64)
    xy = np.asarray(handbase_world_xy_m, dtype=np.float64)
    if values.ndim not in (3, 4) or values.shape[-1] != 3:
        raise ValueError("sampled geometry must have shape (S,P,3) or (S,T,3,3)")
    if values.ndim == 4 and values.shape[-2] != 3:
        raise ValueError("sampled triangles must have three vertices")
    if len(values) == 0 or values.shape[1] == 0 or not np.all(np.isfinite(values)):
        raise ValueError("sampled geometry must be finite and non-empty")
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("world-from-handbase rotation must be finite 3x3")
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-9, rtol=0.0)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-9, rtol=0.0)):
        raise ValueError("world-from-handbase rotation must be orthonormal")
    if xy.shape != (2,) or not np.all(np.isfinite(xy)):
        raise ValueError("handbase world XY must contain two finite values")
    oriented = values @ rotation.T
    oriented[..., :2] += xy
    return oriented, "TRIANGLE_AABB_CONSERVATIVE" if values.ndim == 4 else "POINT_DISCRETE"


def minimum_handbase_z_for_finite_table(
    sampled_geometry_handbase_m: np.ndarray,
    world_from_handbase_rotation: np.ndarray,
    handbase_world_xy_m: Sequence[float],
    table_xy_bounds_m: np.ndarray,
    table_top_z_m: float,
    *,
    required_clearance_m: float = 0.0,
) -> TableHeightRequirement:
    """Return the lowest base Z under a conservative sampled table projection.

    Point samples contribute only when their XY lies inside the finite table.
    A triangle contributes when its XY AABB overlaps the table; all three of its
    vertices then conservatively bound Z. The caller must provide every sampled
    state of the complete hand path; this function does not generate a path.
    """

    world, kind = _world_oriented_geometry(
        sampled_geometry_handbase_m, world_from_handbase_rotation,
        handbase_world_xy_m,
    )
    bounds = np.asarray(table_xy_bounds_m, dtype=np.float64)
    top = _finite_scalar(table_top_z_m, "table top Z")
    clearance = _finite_scalar(required_clearance_m, "required clearance")
    if clearance < 0.0:
        raise ValueError("required clearance must be nonnegative")
    if (bounds.shape != (2, 2) or not np.all(np.isfinite(bounds))
            or np.any(bounds[:, 0] >= bounds[:, 1])):
        raise ValueError("table XY bounds must be finite increasing intervals")
    if world.ndim == 4:
        lower = np.min(world[..., :2], axis=2)
        upper = np.max(world[..., :2], axis=2)
        active = ((upper[..., 0] >= bounds[0, 0])
                  & (lower[..., 0] <= bounds[0, 1])
                  & (upper[..., 1] >= bounds[1, 0])
                  & (lower[..., 1] <= bounds[1, 1]))
        candidate_z = np.where(active[..., None], world[..., 2], np.inf)
    else:
        active = ((world[..., 0] >= bounds[0, 0])
                  & (world[..., 0] <= bounds[0, 1])
                  & (world[..., 1] >= bounds[1, 0])
                  & (world[..., 1] <= bounds[1, 1]))
        candidate_z = np.where(active, world[..., 2], np.inf)
    if not np.any(active):
        return TableHeightRequirement(None, None, None, None, None, 0, kind)
    flat_index = int(np.argmin(candidate_z))
    index = np.unravel_index(flat_index, candidate_z.shape)
    state_index, primitive_index = int(index[0]), int(index[1])
    vertex_index = int(index[2]) if world.ndim == 4 else None
    minimum_relative_z = float(candidate_z[index])
    return TableHeightRequirement(
        top + clearance - minimum_relative_z,
        minimum_relative_z,
        state_index,
        primitive_index,
        vertex_index,
        int(np.count_nonzero(active)),
        kind,
    )


def intersect_contact_with_table(
    contact_intervals_m: Sequence[HeightInterval],
    minimum_table_height_m: float | None,
) -> tuple[HeightInterval, ...]:
    """Intersect sorted contact components with the table lower half-line."""

    table = (-math.inf if minimum_table_height_m is None else
             _finite_scalar(minimum_table_height_m, "minimum table height"))
    result: list[HeightInterval] = []
    previous_upper = -math.inf
    for raw_lower, raw_upper in contact_intervals_m:
        lower = _finite_scalar(raw_lower, "contact interval lower")
        upper = _finite_scalar(raw_upper, "contact interval upper")
        if lower > upper or lower < previous_upper:
            raise ValueError("contact intervals must be sorted and non-overlapping")
        previous_upper = upper
        clipped = max(lower, table)
        if clipped <= upper:
            result.append((clipped, upper))
    return tuple(result)


def project_height_to_intervals(
    original_height_m: float,
    feasible_intervals_m: Sequence[HeightInterval],
) -> HeightProjection:
    """Project to the nearest component; equal distances prefer larger clearance."""

    original = _finite_scalar(original_height_m, "original height")
    candidates = []
    for index, (raw_lower, raw_upper) in enumerate(feasible_intervals_m):
        lower = _finite_scalar(raw_lower, "feasible interval lower")
        upper = _finite_scalar(raw_upper, "feasible interval upper")
        if lower > upper:
            raise ValueError("feasible interval lower exceeds upper")
        projected = min(max(original, lower), upper)
        candidates.append((abs(projected - original), -projected, index,
                           projected, (lower, upper)))
    if not candidates:
        raise ValueError("no height satisfies both contact and table constraints")
    _distance, _negative_height, _index, projected, interval = min(candidates)
    return HeightProjection(
        original, projected, projected - original, interval,
    )


__all__ = [
    "HeightProjection",
    "TableHeightRequirement",
    "intersect_contact_with_table",
    "minimum_handbase_z_for_finite_table",
    "project_height_to_intervals",
    "translate_transform_world_z",
]
