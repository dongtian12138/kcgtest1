"""Fixed-mode parent-surface/pad-feature tangency validator for TE Stage 2.

The previous full SOS1 master silently omitted pad edges and vertices and used
one tessellation-triangle normal for cylindrical STEP faces.  That feasible set
was neither the declared pad polyhedron nor the analytic parent surfaces.  This
module now answers a narrower, valid question: can three explicitly supplied
object-chart/pad-feature modes share one hand configuration and one ``T_HC``?

Object triangles are only closed UV charts on their exact plane/cylinder
parents.  Pad face, internal-edge and internal-vertex modes use the frozen STL
polyhedron, including generalized normal cones and one-ring support.  Open pad
edges at the pad/non-pad material interface are rejected.  An incumbent still
does not prove trim clearance, global nonpenetration, a closure prefix, force
balance, robustness, Isaac lift/hold, or hardware behaviour.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import operator
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from kcg_connector.grasp.robust.analytic_outer_master import (
    AnalyticEnvelopeContract,
    _affine_source,
    _ancestor_joint_names,
    _cross,
    _hand_contact_radius_bound,
    _matmul,
    _numeric_matrix,
    _quaternion_rotation,
    _symbolic_forward_kinematics,
    _triangle_normals,
    load_analytic_envelope_contract,
)
from kcg_connector.grasp.robust.hand_contract import (
    CARTSHandContract,
    VerifiedPad,
    load_carts_hand_contract,
)
from kcg_connector.grasp.robust.hand_model import ThreeFingerHandModel
from kcg_connector.grasp.robust.surface_atlas import (
    ParentSurfaceFrame,
    StepContactAtlas,
    load_step_contact_atlas,
)


CLAIM_SCOPE = "SIMULATION_ONLY_FIXED_MODE_PARENT_SURFACE_PAD_FEATURE_TANGENCY"
LOCAL_MODE_CLAIM_SCOPE = (
    "SIMULATION_ONLY_LOCAL_MODE_PARENT_SURFACE_PAD_FEATURE_TANGENCY"
)
CELL_OUTER_CLAIM_SCOPE = (
    "SIMULATION_ONLY_CELL_AABB_NORMAL_CAP_GEOMETRIC_OUTER_RELAXATION"
)
INNER_WITNESS_SEARCH_CLAIM_SCOPE = (
    "SIMULATION_ONLY_BOUNDED_HIERARCHICAL_STRICT_GEOMETRIC_INNER_SEARCH"
)
REQUIRED_OBJECT_FORBIDDEN_CLEARANCE_M = 61.0e-6
CLOSING_APPROACH_MARGIN_M_PER_RAD = 1.0e-6
LOCAL_FEATURE_RELATIVE_INTERIOR_MARGIN = 1.0e-5
EXPECTED_ANALYTIC_OBJECT_PARENT_COUNT = 18
EXPECTED_PAD_FACE_COUNT = 2442
EXPECTED_PAD_INTERNAL_EDGE_COUNT = 3651
EXPECTED_PAD_INTERNAL_VERTEX_COUNT = 1209
PAD_INITIAL_CELL_MAX_FACE_COUNT = 256
PAD_INITIAL_CELL_MAX_NORMAL_HALF_ANGLE_RAD = math.radians(80.0)
_CELL_BOUND_ROUNDOFF_FACTOR = 128.0
CELL_OUTER_NORMAL_RADIUS = 1.0 + 256.0 * np.finfo(np.float64).eps
STRICT_RESTORATION_POSITION_RESIDUAL_SCALE_M = 1.0e-3
STRICT_RESTORATION_CLOSING_RESIDUAL_SCALE_M_PER_RAD = 1.0e-3
STRICT_RESTORATION_SUPPORT_RESIDUAL_SCALE_M = 1.0e-3
STRICT_RESTORATION_POSITION_WITNESS_TOL_M = 1.0e-8
STRICT_RESTORATION_NORMAL_WITNESS_TOL = 1.0e-8
STRICT_RESTORATION_CLOSING_WITNESS_TOL_M_PER_RAD = 1.0e-10
STRICT_RESTORATION_PARAMETER_WITNESS_TOL = 1.0e-10
STRICT_RESTORATION_SUPPORT_WITNESS_TOL_M = 1.0e-10
STRICT_RESTORATION_SLSQP_MAXITER = 2000
STRICT_RESTORATION_SLSQP_FTOL = 1.0e-12


class GeometricTangencyError(ValueError):
    """Raised when the frozen geometric-master inputs are inconsistent."""


def _strict_index(value: Any, label: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise GeometricTangencyError(f"{label} must be an integer, not boolean")
    try:
        return int(operator.index(value))
    except TypeError as error:
        raise GeometricTangencyError(f"{label} must be an integer") from error


@dataclass(frozen=True)
class FixedContactMode:
    object_triangle_index: int
    pad_feature_kind: Literal["face", "edge", "vertex"]
    pad_feature_id: int | tuple[int, int]


@dataclass(frozen=True)
class PadFeatureChoice:
    kind: Literal["face", "edge", "vertex"]
    feature_id: int | tuple[int, int]

    def __post_init__(self) -> None:
        if self.kind not in ("face", "edge", "vertex"):
            raise GeometricTangencyError("pad feature choice has an invalid kind")
        if self.kind == "edge":
            if not isinstance(self.feature_id, tuple) or len(self.feature_id) != 2:
                raise GeometricTangencyError("pad edge choice needs two vertices")
            edge = tuple(
                sorted(_strict_index(value, "pad edge vertex") for value in self.feature_id)
            )
            if edge[0] == edge[1]:
                raise GeometricTangencyError("pad edge choice has a repeated vertex")
            object.__setattr__(self, "feature_id", edge)
        else:
            object.__setattr__(
                self,
                "feature_id",
                _strict_index(self.feature_id, "pad face/vertex feature"),
            )


@dataclass(frozen=True)
class FixedGeometricState:
    q_contact_rad: tuple[float, float, float, float]
    quaternion_hc_wxyz: tuple[float, float, float, float]
    translation_hc_m: tuple[float, float, float]


@dataclass(frozen=True)
class PadFeatureTopology:
    face_normals: np.ndarray
    edge_faces: Mapping[tuple[int, int], tuple[int, ...]]
    vertex_faces: tuple[tuple[int, ...], ...]
    vertex_neighbors: tuple[tuple[int, ...], ...]
    boundary_edges: frozenset[tuple[int, int]]
    boundary_vertices: frozenset[int]


def _validate_cell_geometry(
    indices: Sequence[int],
    point_lower_m: Sequence[float],
    point_upper_m: Sequence[float],
    normal_axis: Sequence[float],
    normal_cosine_lower_bound: float,
    *,
    label: str,
) -> None:
    identifiers = tuple(int(value) for value in indices)
    if not identifiers or identifiers != tuple(sorted(set(identifiers))):
        raise GeometricTangencyError(f"{label} indices must be non-empty, sorted and unique")
    lower = np.asarray(point_lower_m, dtype=np.float64)
    upper = np.asarray(point_upper_m, dtype=np.float64)
    axis = np.asarray(normal_axis, dtype=np.float64)
    cosine = float(normal_cosine_lower_bound)
    if (
        lower.shape != (3,)
        or upper.shape != (3,)
        or axis.shape != (3,)
        or not np.all(np.isfinite(lower))
        or not np.all(np.isfinite(upper))
        or not np.all(np.isfinite(axis))
        or not math.isfinite(cosine)
        or np.any(lower > upper)
        or not np.isclose(np.linalg.norm(axis), 1.0, rtol=0.0, atol=1.0e-12)
        or cosine < -1.0
        or cosine > 1.0
    ):
        raise GeometricTangencyError(f"{label} has invalid point or normal bounds")


@dataclass(frozen=True)
class ObjectChartCell:
    """One conservative cell over closed analytic object UV charts."""

    parent_face_index: int
    triangle_indices: tuple[int, ...]
    point_lower_m: tuple[float, float, float]
    point_upper_m: tuple[float, float, float]
    normal_axis: tuple[float, float, float]
    normal_cosine_lower_bound: float
    split_path: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.parent_face_index < 1 or any(value not in (0, 1) for value in self.split_path):
            raise GeometricTangencyError("object chart cell identity is invalid")
        _validate_cell_geometry(
            self.triangle_indices,
            self.point_lower_m,
            self.point_upper_m,
            self.normal_axis,
            self.normal_cosine_lower_bound,
            label="object chart cell",
        )

    @property
    def normal_half_angle_rad(self) -> float:
        return math.acos(float(np.clip(self.normal_cosine_lower_bound, -1.0, 1.0)))


@dataclass(frozen=True)
class PadFaceCell:
    """One conservative cell over a non-empty set of frozen pad faces."""

    pad_name: str
    face_indices: tuple[int, ...]
    point_lower_m: tuple[float, float, float]
    point_upper_m: tuple[float, float, float]
    normal_axis: tuple[float, float, float]
    normal_cosine_lower_bound: float
    split_path: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.pad_name or any(value not in (0, 1) for value in self.split_path):
            raise GeometricTangencyError("pad face cell identity is invalid")
        _validate_cell_geometry(
            self.face_indices,
            self.point_lower_m,
            self.point_upper_m,
            self.normal_axis,
            self.normal_cosine_lower_bound,
            label="pad face cell",
        )

    @property
    def normal_half_angle_rad(self) -> float:
        return math.acos(float(np.clip(self.normal_cosine_lower_bound, -1.0, 1.0)))


def _outward_aabb(
    lower: Sequence[float], upper: Sequence[float]
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    lower_value = np.asarray(lower, dtype=np.float64)
    upper_value = np.asarray(upper, dtype=np.float64)
    if (
        lower_value.shape != (3,)
        or upper_value.shape != (3,)
        or not np.all(np.isfinite(lower_value))
        or not np.all(np.isfinite(upper_value))
        or np.any(lower_value > upper_value)
    ):
        raise GeometricTangencyError("cell AABB input is invalid")
    scale = max(
        1.0,
        float(np.max(np.abs(lower_value))),
        float(np.max(np.abs(upper_value))),
    )
    padding = _CELL_BOUND_ROUNDOFF_FACTOR * np.finfo(np.float64).eps * scale
    outward_lower = np.nextafter(lower_value - padding, -np.inf)
    outward_upper = np.nextafter(upper_value + padding, np.inf)
    return (
        tuple(float(value) for value in outward_lower),
        tuple(float(value) for value in outward_upper),
    )


def _deterministic_normal_cap(
    center_normals: np.ndarray,
    angular_half_widths_rad: Sequence[float],
    primitive_indices: Sequence[int],
) -> tuple[tuple[float, float, float], float]:
    normals = np.asarray(center_normals, dtype=np.float64)
    widths = np.asarray(angular_half_widths_rad, dtype=np.float64)
    identities = tuple(int(value) for value in primitive_indices)
    if (
        normals.shape != (len(identities), 3)
        or widths.shape != (len(identities),)
        or not identities
        or not np.all(np.isfinite(normals))
        or not np.all(np.isfinite(widths))
        or np.any(widths < 0.0)
    ):
        raise GeometricTangencyError("normal-cap primitives are invalid")
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths <= 1.0e-14):
        raise GeometricTangencyError("normal-cap primitive has a zero normal")
    unit_normals = normals / lengths[:, None]
    reference = np.sum(unit_normals, axis=0)
    reference_length = float(np.linalg.norm(reference))
    if reference_length <= 128.0 * np.finfo(np.float64).eps * len(unit_normals):
        reference = unit_normals[int(np.argmin(np.asarray(identities)))]
        reference_length = float(np.linalg.norm(reference))
    axis = reference / reference_length
    maximum_angle = 0.0
    for normal, half_width in zip(unit_normals, widths):
        center_angle = math.acos(float(np.clip(axis @ normal, -1.0, 1.0)))
        maximum_angle = max(
            maximum_angle,
            min(math.pi, center_angle + min(math.pi, float(half_width))),
        )
    cosine = math.cos(maximum_angle)
    cosine_padding = _CELL_BOUND_ROUNDOFF_FACTOR * np.finfo(np.float64).eps
    cosine = max(-1.0, float(np.nextafter(cosine - cosine_padding, -np.inf)))
    return tuple(float(value) for value in axis), cosine


def _object_triangle_cell_geometry(
    atlas: StepContactAtlas,
    triangle_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    index = _strict_index(triangle_index, "object triangle")
    if index < 0 or index >= atlas.triangle_count:
        raise GeometricTangencyError("object chart triangle is outside the atlas")
    parent = atlas.parent_surface(int(atlas.parent_face_index[index]))
    uv = np.asarray(atlas.triangle_uv[index], dtype=np.float64)
    uv_center = np.mean(uv, axis=0)
    center_point = parent.point_from_uv(float(uv_center[0]), float(uv_center[1]))
    center_normal = parent.normal_from_uv(float(uv_center[0]), float(uv_center[1]))
    if parent.kind == "plane":
        exact_vertices = np.asarray(
            [parent.point_from_uv(float(row[0]), float(row[1])) for row in uv],
            dtype=np.float64,
        )
        lower, upper = _outward_aabb(
            np.min(exact_vertices, axis=0), np.max(exact_vertices, axis=0)
        )
        return (
            np.asarray(lower),
            np.asarray(upper),
            center_normal,
            0.0,
            center_point,
        )

    assert parent.radius_m is not None
    u_lower, v_lower = np.min(uv, axis=0)
    u_upper, v_upper = np.max(uv, axis=0)
    u_center = 0.5 * float(u_lower + u_upper)
    v_center = 0.5 * float(v_lower + v_upper)
    u_half_width = 0.5 * float(u_upper - u_lower)
    v_half_width = 0.5 * float(v_upper - v_lower)
    center_point = parent.point_from_uv(u_center, v_center)
    center_normal = parent.normal_from_uv(u_center, v_center)
    x_direction = np.asarray(parent.x_direction, dtype=np.float64)
    y_direction = np.asarray(parent.y_direction, dtype=np.float64)
    axis_direction = np.asarray(parent.axis_direction, dtype=np.float64)
    coordinate_radius = np.sqrt(x_direction * x_direction + y_direction * y_direction)
    coordinate_delta = (
        parent.radius_m * u_half_width * coordinate_radius
        + parent.uv_length_scale_m * v_half_width * np.abs(axis_direction)
    )
    lower, upper = _outward_aabb(
        center_point - coordinate_delta,
        center_point + coordinate_delta,
    )
    return (
        np.asarray(lower),
        np.asarray(upper),
        center_normal,
        min(math.pi, u_half_width),
        center_point,
    )


def _make_object_chart_cell(
    atlas: StepContactAtlas,
    parent_face_index: int,
    triangle_indices: Sequence[int],
    *,
    split_path: tuple[int, ...],
) -> ObjectChartCell:
    indices = tuple(sorted(int(value) for value in triangle_indices))
    if not indices or len(set(indices)) != len(indices):
        raise GeometricTangencyError("object chart cell needs unique triangles")
    if any(
        index < 0
        or index >= atlas.triangle_count
        or int(atlas.parent_face_index[index]) != int(parent_face_index)
        for index in indices
    ):
        raise GeometricTangencyError("object chart cell crosses its analytic parent")
    rows = [_object_triangle_cell_geometry(atlas, index) for index in indices]
    lower, upper = _outward_aabb(
        np.min(np.asarray([row[0] for row in rows]), axis=0),
        np.max(np.asarray([row[1] for row in rows]), axis=0),
    )
    normal_axis, normal_cosine = _deterministic_normal_cap(
        np.asarray([row[2] for row in rows]),
        [row[3] for row in rows],
        indices,
    )
    return ObjectChartCell(
        parent_face_index=int(parent_face_index),
        triangle_indices=indices,
        point_lower_m=lower,
        point_upper_m=upper,
        normal_axis=normal_axis,
        normal_cosine_lower_bound=normal_cosine,
        split_path=split_path,
    )


def build_object_parent_root_cells(
    atlas: StepContactAtlas,
) -> tuple[ObjectChartCell, ...]:
    """Build one conservative root cell for each proven analytic parent face."""

    parent_faces = tuple(sorted(parent.face_index for parent in atlas.parent_surfaces))
    if (
        len(parent_faces) != EXPECTED_ANALYTIC_OBJECT_PARENT_COUNT
        or len(set(parent_faces)) != len(parent_faces)
    ):
        raise GeometricTangencyError(
            "proven analytic object-parent coverage changed from 18 faces"
        )
    cells = tuple(
        _make_object_chart_cell(
            atlas,
            parent_face,
            np.flatnonzero(atlas.parent_face_index == parent_face),
            split_path=(),
        )
        for parent_face in parent_faces
    )
    covered = tuple(
        sorted(index for cell in cells for index in cell.triangle_indices)
    )
    expected = tuple(
        int(value)
        for value in np.flatnonzero(
            np.isin(atlas.parent_face_index, np.asarray(parent_faces, dtype=np.int64))
        )
    )
    if covered != expected or len(covered) != len(set(covered)):
        raise GeometricTangencyError("object parent root cells do not partition the charts")
    return cells


def _make_pad_face_cell(
    pad: VerifiedPad,
    face_indices: Sequence[int],
    *,
    split_path: tuple[int, ...],
) -> PadFaceCell:
    indices = tuple(sorted(int(value) for value in face_indices))
    if (
        not indices
        or len(set(indices)) != len(indices)
        or indices[0] < 0
        or indices[-1] >= pad.triangle_count
    ):
        raise GeometricTangencyError("pad face cell needs valid unique faces")
    triangles = pad.points_local_m[pad.faces[np.asarray(indices, dtype=np.int64)]]
    lower, upper = _outward_aabb(
        np.min(triangles, axis=(0, 1)), np.max(triangles, axis=(0, 1))
    )
    normals = _triangle_normals(triangles)
    normal_axis, normal_cosine = _deterministic_normal_cap(
        normals,
        np.zeros(len(indices), dtype=np.float64),
        indices,
    )
    return PadFaceCell(
        pad_name=pad.name,
        face_indices=indices,
        point_lower_m=lower,
        point_upper_m=upper,
        normal_axis=normal_axis,
        normal_cosine_lower_bound=normal_cosine,
        split_path=split_path,
    )


def _stable_surface_bisection(
    primitive_indices: Sequence[int],
    point_centers: np.ndarray,
    center_normals: np.ndarray,
    *,
    global_point_scale: float,
    prefer_normals: bool | None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    indices = tuple(int(value) for value in primitive_indices)
    points = np.asarray(point_centers, dtype=np.float64)
    normals = np.asarray(center_normals, dtype=np.float64)
    if (
        len(indices) < 2
        or points.shape != (len(indices), 3)
        or normals.shape != (len(indices), 3)
        or not np.all(np.isfinite(points))
        or not np.all(np.isfinite(normals))
        or not math.isfinite(global_point_scale)
        or global_point_scale <= 0.0
    ):
        raise GeometricTangencyError("surface cell cannot be bisected")
    point_ranges = np.ptp(points, axis=0)
    normal_ranges = np.ptp(normals, axis=0)
    if prefer_normals is None:
        point_score = float(np.max(point_ranges)) / global_point_scale
        normal_score = 0.5 * float(np.max(normal_ranges))
        use_normals = normal_score >= point_score
    else:
        use_normals = bool(prefer_normals)
    features = normals if use_normals else points
    axis = int(np.argmax(normal_ranges if use_normals else point_ranges))
    ordered_rows = sorted(
        range(len(indices)), key=lambda row: (float(features[row, axis]), indices[row])
    )
    midpoint = len(ordered_rows) // 2
    left = tuple(sorted(indices[row] for row in ordered_rows[:midpoint]))
    right = tuple(sorted(indices[row] for row in ordered_rows[midpoint:]))
    if not left or not right or set(left).intersection(right) or set(left).union(right) != set(indices):
        raise GeometricTangencyError("stable surface bisection lost coverage")
    return left, right


def split_object_chart_cell(
    atlas: StepContactAtlas,
    cell: ObjectChartCell,
    *,
    prefer_normals: bool | None = None,
) -> tuple[ObjectChartCell, ObjectChartCell]:
    """Deterministically bisect one object cell without losing a UV chart."""

    rows = [
        _object_triangle_cell_geometry(atlas, index)
        for index in cell.triangle_indices
    ]
    analytic_faces = np.asarray(
        [parent.face_index for parent in atlas.parent_surfaces], dtype=np.int64
    )
    analytic_points = atlas.triangles_m[
        np.isin(atlas.parent_face_index, analytic_faces)
    ]
    global_scale = max(
        float(np.linalg.norm(np.ptp(analytic_points.reshape(-1, 3), axis=0))),
        np.finfo(np.float64).tiny,
    )
    left_indices, right_indices = _stable_surface_bisection(
        cell.triangle_indices,
        np.asarray([row[4] for row in rows]),
        np.asarray([row[2] for row in rows]),
        global_point_scale=global_scale,
        prefer_normals=prefer_normals,
    )
    return (
        _make_object_chart_cell(
            atlas,
            cell.parent_face_index,
            left_indices,
            split_path=cell.split_path + (0,),
        ),
        _make_object_chart_cell(
            atlas,
            cell.parent_face_index,
            right_indices,
            split_path=cell.split_path + (1,),
        ),
    )


def split_pad_face_cell(
    pad: VerifiedPad,
    cell: PadFaceCell,
    *,
    prefer_normals: bool | None = None,
) -> tuple[PadFaceCell, PadFaceCell]:
    """Deterministically bisect one pad cell without losing a mesh face."""

    if cell.pad_name != pad.name:
        raise GeometricTangencyError("pad cell belongs to a different pad")
    indices = np.asarray(cell.face_indices, dtype=np.int64)
    triangles = pad.points_local_m[pad.faces[indices]]
    point_centers = np.mean(triangles, axis=1)
    normals = _triangle_normals(triangles)
    global_scale = max(
        float(np.linalg.norm(np.ptp(pad.points_local_m, axis=0))),
        np.finfo(np.float64).tiny,
    )
    left_indices, right_indices = _stable_surface_bisection(
        cell.face_indices,
        point_centers,
        normals,
        global_point_scale=global_scale,
        prefer_normals=prefer_normals,
    )
    return (
        _make_pad_face_cell(
            pad, left_indices, split_path=cell.split_path + (0,)
        ),
        _make_pad_face_cell(
            pad, right_indices, split_path=cell.split_path + (1,)
        ),
    )


def build_pad_initial_cells(
    pad: VerifiedPad,
    *,
    max_face_count: int = PAD_INITIAL_CELL_MAX_FACE_COUNT,
    max_normal_half_angle_rad: float = PAD_INITIAL_CELL_MAX_NORMAL_HALF_ANGLE_RAD,
) -> tuple[PadFaceCell, ...]:
    """Partition all pad faces into bounded deterministic initial cells."""

    if pad.triangle_count != EXPECTED_PAD_FACE_COUNT:
        raise GeometricTangencyError("complete pad face count changed from 2442")
    if (
        isinstance(max_face_count, bool)
        or int(max_face_count) != max_face_count
        or max_face_count < 1
        or not math.isfinite(max_normal_half_angle_rad)
        or max_normal_half_angle_rad <= 0.0
        or max_normal_half_angle_rad > math.pi
    ):
        raise GeometricTangencyError("pad initial-cell limits are invalid")
    root = _make_pad_face_cell(
        pad, range(pad.triangle_count), split_path=()
    )
    pending = [root]
    accepted: list[PadFaceCell] = []
    while pending:
        cell = pending.pop()
        count_violation = len(cell.face_indices) > int(max_face_count)
        cap_violation = cell.normal_half_angle_rad > max_normal_half_angle_rad
        if not count_violation and not cap_violation:
            accepted.append(cell)
            continue
        if len(cell.face_indices) < 2:
            raise GeometricTangencyError("singleton pad face violates its normal cap")
        left, right = split_pad_face_cell(
            pad,
            cell,
            prefer_normals=True if cap_violation else None,
        )
        pending.extend((right, left))
    result = tuple(sorted(accepted, key=lambda cell: cell.split_path))
    covered = tuple(sorted(index for cell in result for index in cell.face_indices))
    expected = tuple(range(pad.triangle_count))
    if (
        covered != expected
        or len(covered) != len(set(covered))
        or any(len(cell.face_indices) > max_face_count for cell in result)
        or any(
            cell.normal_half_angle_rad > max_normal_half_angle_rad
            for cell in result
        )
    ):
        raise GeometricTangencyError("pad initial cells do not meet their partition bounds")
    return result


def _root_geometric_cell_node(
    atlas: StepContactAtlas,
    hand_contract: CARTSHandContract,
) -> GeometricCellNode:
    object_roots = build_object_parent_root_cells(atlas)
    pad_roots = tuple(build_pad_initial_cells(pad) for pad in hand_contract.pads)
    return GeometricCellNode(
        object_cell_rows=(object_roots, object_roots, object_roots),
        pad_cell_rows=(pad_roots[0], pad_roots[1], pad_roots[2]),
        path=(),
    )


def build_root_geometric_cell_node(
    contract: AnalyticEnvelopeContract,
) -> GeometricCellNode:
    """Build the complete 18-parent/full-pad tagged root search node."""

    atlas = load_step_contact_atlas(
        contract.step_contact_atlas_path, repository_root=contract.repository_root
    )
    hand_contract = load_carts_hand_contract(
        contract.hand_contact_contract_path,
        repository_root=contract.repository_root,
    )
    return _root_geometric_cell_node(atlas, hand_contract)


def _surface_cell_stable_key(
    cell: ObjectChartCell | PadFaceCell,
) -> tuple[Any, ...]:
    if isinstance(cell, ObjectChartCell):
        return (
            0,
            cell.parent_face_index,
            cell.split_path,
            cell.triangle_indices,
        )
    return (1, cell.pad_name, cell.split_path, cell.face_indices)


def _surface_cell_primitive_count(cell: ObjectChartCell | PadFaceCell) -> int:
    return (
        len(cell.triangle_indices)
        if isinstance(cell, ObjectChartCell)
        else len(cell.face_indices)
    )


def _balanced_cell_groups(
    cells: Sequence[ObjectChartCell] | Sequence[PadFaceCell],
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    if len(cells) < 2:
        raise GeometricTangencyError("cell-row grouping needs at least two cells")
    ordered = sorted(
        cells,
        key=lambda cell: (
            -_surface_cell_primitive_count(cell),
            _surface_cell_stable_key(cell),
        ),
    )
    groups: tuple[list[Any], list[Any]] = ([], [])
    totals = [0, 0]
    for cell in ordered:
        group = 0 if totals[0] <= totals[1] else 1
        groups[group].append(cell)
        totals[group] += _surface_cell_primitive_count(cell)
    if not groups[0] or not groups[1]:
        raise GeometricTangencyError("balanced cell grouping produced an empty child")
    return (
        tuple(sorted(groups[0], key=_surface_cell_stable_key)),
        tuple(sorted(groups[1], key=_surface_cell_stable_key)),
    )


def _object_row_primitives(row: Sequence[ObjectChartCell]) -> frozenset[int]:
    return frozenset(index for cell in row for index in cell.triangle_indices)


def _pad_row_primitives(row: Sequence[PadFaceCell]) -> frozenset[int]:
    return frozenset(index for cell in row for index in cell.face_indices)


def split_geometric_cell_node(
    node: GeometricCellNode,
    atlas: StepContactAtlas,
    hand_contract: CARTSHandContract,
) -> tuple[GeometricCellNode, GeometricCellNode]:
    """Split exactly one tagged row; child products are disjoint and exhaustive."""

    counts = node.row_primitive_counts
    remaining_bits = tuple((count - 1).bit_length() for count in counts)
    maximum_bits = max(remaining_bits)
    if maximum_bits == 0:
        raise GeometricTangencyError("singleton geometric cell node cannot be split")
    # Fixed tie order: object fingers 1..3, then pad fingers 1..3.
    selected_row = remaining_bits.index(maximum_bits)
    object_rows_left = list(node.object_cell_rows)
    object_rows_right = list(node.object_cell_rows)
    pad_rows_left = list(node.pad_cell_rows)
    pad_rows_right = list(node.pad_cell_rows)

    if selected_row < 3:
        row = node.object_cell_rows[selected_row]
        if len(row) > 1:
            left_row, right_row = _balanced_cell_groups(row)
        else:
            left_cell, right_cell = split_object_chart_cell(atlas, row[0])
            left_row, right_row = (left_cell,), (right_cell,)
        object_rows_left[selected_row] = left_row  # type: ignore[assignment]
        object_rows_right[selected_row] = right_row  # type: ignore[assignment]
        parent_primitives = _object_row_primitives(row)
        left_primitives = _object_row_primitives(left_row)
        right_primitives = _object_row_primitives(right_row)
    else:
        finger_index = selected_row - 3
        row = node.pad_cell_rows[finger_index]
        if len(row) > 1:
            left_row, right_row = _balanced_cell_groups(row)
        else:
            left_cell, right_cell = split_pad_face_cell(
                hand_contract.pads[finger_index], row[0]
            )
            left_row, right_row = (left_cell,), (right_cell,)
        pad_rows_left[finger_index] = left_row  # type: ignore[assignment]
        pad_rows_right[finger_index] = right_row  # type: ignore[assignment]
        parent_primitives = _pad_row_primitives(row)
        left_primitives = _pad_row_primitives(left_row)
        right_primitives = _pad_row_primitives(right_row)

    if (
        left_primitives.intersection(right_primitives)
        or left_primitives.union(right_primitives) != parent_primitives
    ):
        raise GeometricTangencyError("geometric node split lost or duplicated a primitive")
    left = GeometricCellNode(
        object_cell_rows=(
            object_rows_left[0],
            object_rows_left[1],
            object_rows_left[2],
        ),
        pad_cell_rows=(pad_rows_left[0], pad_rows_left[1], pad_rows_left[2]),
        path=node.path + (0,),
    )
    right = GeometricCellNode(
        object_cell_rows=(
            object_rows_right[0],
            object_rows_right[1],
            object_rows_right[2],
        ),
        pad_cell_rows=(pad_rows_right[0], pad_rows_right[1], pad_rows_right[2]),
        path=node.path + (1,),
    )
    for row_index in range(3):
        if row_index != selected_row and (
            left.object_cell_rows[row_index] != node.object_cell_rows[row_index]
            or right.object_cell_rows[row_index] != node.object_cell_rows[row_index]
        ):
            raise GeometricTangencyError("object node split changed an unselected row")
        if row_index + 3 != selected_row and (
            left.pad_cell_rows[row_index] != node.pad_cell_rows[row_index]
            or right.pad_cell_rows[row_index] != node.pad_cell_rows[row_index]
        ):
            raise GeometricTangencyError("pad node split changed an unselected row")
    return left, right


@dataclass(frozen=True)
class FixedObjectSurfaceVariables:
    point: tuple[Any, Any, Any]
    normal: tuple[Any, Any, Any]
    triangle_index: int
    parent_surface: ParentSurfaceFrame
    barycentric: tuple[Any, Any, Any]
    uv: tuple[Any, Any]
    triangle_altitudes_m: tuple[float, float, float]


@dataclass(frozen=True)
class FixedPadFeatureVariables:
    point: tuple[Any, Any, Any]
    normal: tuple[Any, Any, Any]
    mode: FixedContactMode
    parameter: tuple[Any, ...]
    normal_coefficients: tuple[Any, ...]
    incident_faces: tuple[int, ...]
    support_vertex_indices: tuple[int, ...]


@dataclass
class GeometricTangencyBundle:
    model: Any
    contract: AnalyticEnvelopeContract
    atlas: StepContactAtlas
    hand_contract: CARTSHandContract
    hand_model: ThreeFingerHandModel
    quaternion: tuple[Any, Any, Any, Any]
    translation_hc: tuple[Any, Any, Any]
    q_contact: Mapping[str, Any]
    fixed_modes: tuple[FixedContactMode, FixedContactMode, FixedContactMode]
    object_surfaces: tuple[FixedObjectSurfaceVariables, ...]
    pad_surfaces: tuple[FixedPadFeatureVariables, ...]
    pad_topologies: tuple[PadFeatureTopology, PadFeatureTopology, PadFeatureTopology]
    translation_bound_m: float


@dataclass
class LocalModeGeometricTangencyBundle:
    """One exact-one local mode choice per finger.

    Every candidate remains continuous on its object chart and pad feature.
    The candidate list is only an inner-feasibility aid and never defines the
    global grasp domain or an optimality certificate.
    """

    model: Any
    contract: AnalyticEnvelopeContract
    atlas: StepContactAtlas
    hand_contract: CARTSHandContract
    hand_model: ThreeFingerHandModel
    quaternion: tuple[Any, Any, Any, Any]
    translation_hc: tuple[Any, Any, Any]
    q_contact: Mapping[str, Any]
    object_triangle_choices: tuple[
        tuple[int, ...], tuple[int, ...], tuple[int, ...]
    ]
    pad_feature_choices: tuple[
        tuple[PadFeatureChoice, ...],
        tuple[PadFeatureChoice, ...],
        tuple[PadFeatureChoice, ...],
    ]
    object_selectors: tuple[tuple[Any, ...], tuple[Any, ...], tuple[Any, ...]]
    pad_selectors: tuple[tuple[Any, ...], tuple[Any, ...], tuple[Any, ...]]
    object_surfaces: tuple[
        tuple[FixedObjectSurfaceVariables, ...],
        tuple[FixedObjectSurfaceVariables, ...],
        tuple[FixedObjectSurfaceVariables, ...],
    ]
    pad_surfaces: tuple[
        tuple[FixedPadFeatureVariables, ...],
        tuple[FixedPadFeatureVariables, ...],
        tuple[FixedPadFeatureVariables, ...],
    ]
    pad_topologies: tuple[PadFeatureTopology, PadFeatureTopology, PadFeatureTopology]
    translation_bound_m: float


@dataclass
class GeometricCellOuterBundle:
    """Cell-disjunctive outer relaxation used only for safe refinement."""

    model: Any
    contract: AnalyticEnvelopeContract
    atlas: StepContactAtlas
    hand_contract: CARTSHandContract
    hand_model: ThreeFingerHandModel
    quaternion: tuple[Any, Any, Any, Any]
    translation_hc: tuple[Any, Any, Any]
    q_contact: Mapping[str, Any]
    object_cell_rows: tuple[
        tuple[ObjectChartCell, ...],
        tuple[ObjectChartCell, ...],
        tuple[ObjectChartCell, ...],
    ]
    pad_cell_rows: tuple[
        tuple[PadFaceCell, ...],
        tuple[PadFaceCell, ...],
        tuple[PadFaceCell, ...],
    ]
    object_selectors: tuple[tuple[Any, ...], tuple[Any, ...], tuple[Any, ...]]
    pad_selectors: tuple[tuple[Any, ...], tuple[Any, ...], tuple[Any, ...]]
    object_contact_points: tuple[tuple[Any, Any, Any], ...]
    object_contact_normals: tuple[tuple[Any, Any, Any], ...]
    pad_contact_points: tuple[tuple[Any, Any, Any], ...]
    pad_contact_normals: tuple[tuple[Any, Any, Any], ...]
    hand_contact_points: tuple[tuple[Any, Any, Any], ...]
    translation_bound_m: float
    geometry_bounds_verified: bool


@dataclass(frozen=True)
class GeometricCellNode:
    """One tagged Cartesian product of three object and three pad cell rows."""

    object_cell_rows: tuple[
        tuple[ObjectChartCell, ...],
        tuple[ObjectChartCell, ...],
        tuple[ObjectChartCell, ...],
    ]
    pad_cell_rows: tuple[
        tuple[PadFaceCell, ...],
        tuple[PadFaceCell, ...],
        tuple[PadFaceCell, ...],
    ]
    path: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if (
            len(self.object_cell_rows) != 3
            or len(self.pad_cell_rows) != 3
            or any(not row for row in self.object_cell_rows)
            or any(not row for row in self.pad_cell_rows)
            or any(value not in (0, 1) for value in self.path)
        ):
            raise GeometricTangencyError("geometric cell node identity is invalid")
        for row in self.object_cell_rows:
            if any(not isinstance(cell, ObjectChartCell) for cell in row):
                raise GeometricTangencyError("node object row contains a wrong value")
            indices = [index for cell in row for index in cell.triangle_indices]
            if len(indices) != len(set(indices)):
                raise GeometricTangencyError("node object row overlaps tagged charts")
        for row in self.pad_cell_rows:
            if any(not isinstance(cell, PadFaceCell) for cell in row):
                raise GeometricTangencyError("node pad row contains a wrong value")
            indices = [index for cell in row for index in cell.face_indices]
            if len(indices) != len(set(indices)):
                raise GeometricTangencyError("node pad row overlaps tagged faces")

    @property
    def row_primitive_counts(self) -> tuple[int, int, int, int, int, int]:
        object_counts = tuple(
            sum(len(cell.triangle_indices) for cell in row)
            for row in self.object_cell_rows
        )
        pad_counts = tuple(
            sum(len(cell.face_indices) for cell in row)
            for row in self.pad_cell_rows
        )
        return object_counts + pad_counts  # type: ignore[return-value]

    @property
    def total_candidate_count(self) -> int:
        return sum(self.row_primitive_counts)

    @property
    def is_splittable(self) -> bool:
        return any(count > 1 for count in self.row_primitive_counts)


def _triangle_altitudes(triangles: np.ndarray) -> np.ndarray:
    values = np.asarray(triangles, dtype=np.float64)
    doubled_area = np.linalg.norm(
        np.cross(values[:, 1] - values[:, 0], values[:, 2] - values[:, 0]),
        axis=1,
    )
    result = np.empty((len(values), 3), dtype=np.float64)
    for vertex in range(3):
        edge = values[:, (vertex + 2) % 3] - values[:, (vertex + 1) % 3]
        edge_length = np.linalg.norm(edge, axis=1)
        result[:, vertex] = doubled_area / edge_length
    if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
        raise GeometricTangencyError("surface contains a degenerate triangle")
    result.setflags(write=False)
    return result


def _pad_feature_topology(points: np.ndarray, faces: np.ndarray) -> PadFeatureTopology:
    values = np.asarray(points, dtype=np.float64)
    indexed_faces = np.asarray(faces, dtype=np.int64)
    normals = _triangle_normals(values[indexed_faces])
    edge_rows: dict[tuple[int, int], list[int]] = {}
    vertex_faces: list[list[int]] = [[] for _ in range(len(values))]
    vertex_neighbors: list[set[int]] = [set() for _ in range(len(values))]
    for face_index, face in enumerate(indexed_faces):
        for vertex in face:
            vertex_faces[int(vertex)].append(face_index)
        for first, second in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            a, b = sorted((int(first), int(second)))
            edge_rows.setdefault((a, b), []).append(face_index)
            vertex_neighbors[a].add(b)
            vertex_neighbors[b].add(a)
    if any(len(owners) > 2 for owners in edge_rows.values()):
        raise GeometricTangencyError("pad feature mesh contains a non-manifold edge")
    boundary_edges = frozenset(
        edge for edge, owners in edge_rows.items() if len(owners) == 1
    )
    boundary_vertices = frozenset(vertex for edge in boundary_edges for vertex in edge)
    normals.setflags(write=False)
    return PadFeatureTopology(
        face_normals=normals,
        edge_faces={edge: tuple(owners) for edge, owners in edge_rows.items()},
        vertex_faces=tuple(tuple(rows) for rows in vertex_faces),
        vertex_neighbors=tuple(tuple(sorted(rows)) for rows in vertex_neighbors),
        boundary_edges=boundary_edges,
        boundary_vertices=boundary_vertices,
    )


def _add_fixed_object_surface(
    model: Any,
    quicksum: Any,
    sin: Any,
    cos: Any,
    atlas: StepContactAtlas,
    *,
    triangle_index: int,
    prefix: str,
    activation: Any | None = None,
    relative_interior_margin: float = 0.0,
) -> FixedObjectSurfaceVariables:
    index = int(triangle_index)
    if index < 0 or index >= atlas.triangle_count:
        raise GeometricTangencyError("fixed object triangle is outside the atlas")
    active = 1.0 if activation is None else activation
    barycentric = tuple(
        model.addVar(lb=0.0, ub=1.0, name=f"{prefix}_b{vertex}")
        for vertex in range(3)
    )
    model.addCons(
        quicksum(barycentric) == active,
        name=f"{prefix}_closed_simplex",
    )
    if relative_interior_margin < 0.0 or relative_interior_margin >= 1.0 / 3.0:
        raise GeometricTangencyError("object chart interior margin is invalid")
    for vertex, value in enumerate(barycentric):
        if relative_interior_margin > 0.0:
            model.addCons(
                value >= relative_interior_margin * active,
                name=f"{prefix}_relative_interior{vertex}",
            )
    triangle_uv = atlas.triangle_uv[index]
    uv = tuple(
        model.addVar(
            lb=(
                float(np.min(triangle_uv[:, coordinate]))
                if activation is None
                else min(0.0, float(np.min(triangle_uv[:, coordinate])))
            ),
            ub=(
                float(np.max(triangle_uv[:, coordinate]))
                if activation is None
                else max(0.0, float(np.max(triangle_uv[:, coordinate])))
            ),
            name=f"{prefix}_{'uv'[coordinate]}",
        )
        for coordinate in range(2)
    )
    for coordinate in range(2):
        model.addCons(
            uv[coordinate]
            == quicksum(
                barycentric[vertex] * float(triangle_uv[vertex, coordinate])
                for vertex in range(3)
            ),
            name=f"{prefix}_uv_coordinate{coordinate}",
        )
    parent = atlas.parent_surface(int(atlas.parent_face_index[index]))
    origin = parent.origin_m
    x_direction = parent.x_direction
    y_direction = parent.y_direction
    axis_direction = parent.axis_direction
    if parent.kind == "plane":
        point = tuple(
            active * float(origin[axis])
            + parent.uv_length_scale_m
            * (
                uv[0] * float(x_direction[axis])
                + uv[1] * float(y_direction[axis])
            )
            for axis in range(3)
        )
        normal = tuple(
            active * parent.outward_sign * float(axis_direction[axis])
            for axis in range(3)
        )
    else:
        assert parent.radius_m is not None
        if activation is None:
            cosine, sine = cos(uv[0]), sin(uv[0])
        else:
            cosine = model.addVar(lb=-1.0, ub=1.0, name=f"{prefix}_cos")
            sine = model.addVar(lb=-1.0, ub=1.0, name=f"{prefix}_sin")
            model.addCons(cosine <= active, name=f"{prefix}_cos_active_upper")
            model.addCons(cosine >= -active, name=f"{prefix}_cos_active_lower")
            model.addCons(sine <= active, name=f"{prefix}_sin_active_upper")
            model.addCons(sine >= -active, name=f"{prefix}_sin_active_lower")
            model.addCons(
                cosine - cos(uv[0]) <= 2.0 * (1.0 - active),
                name=f"{prefix}_cos_graph_upper",
            )
            model.addCons(
                cos(uv[0]) - cosine <= 2.0 * (1.0 - active),
                name=f"{prefix}_cos_graph_lower",
            )
            model.addCons(
                sine - sin(uv[0]) <= 2.0 * (1.0 - active),
                name=f"{prefix}_sin_graph_upper",
            )
            model.addCons(
                sin(uv[0]) - sine <= 2.0 * (1.0 - active),
                name=f"{prefix}_sin_graph_lower",
            )
        radial = tuple(
            cosine * float(x_direction[axis]) + sine * float(y_direction[axis])
            for axis in range(3)
        )
        point = tuple(
            active * float(origin[axis])
            + parent.radius_m * radial[axis]
            + parent.uv_length_scale_m * uv[1] * float(axis_direction[axis])
            for axis in range(3)
        )
        normal = tuple(parent.outward_sign * radial[axis] for axis in range(3))
    altitudes = _triangle_altitudes(atlas.triangles_m[index : index + 1])[0]
    return FixedObjectSurfaceVariables(
        point=point,
        normal=normal,
        triangle_index=index,
        parent_surface=parent,
        barycentric=barycentric,
        uv=uv,
        triangle_altitudes_m=tuple(float(value) for value in altitudes),
    )


def _add_generalized_pad_normal(
    model: Any,
    quicksum: Any,
    *,
    normals: np.ndarray,
    points: np.ndarray,
    support_origin: np.ndarray,
    support_vertex_indices: Sequence[int],
    prefix: str,
    activation: Any | None = None,
) -> tuple[tuple[Any, Any, Any], tuple[Any, ...]]:
    generators = np.asarray(normals, dtype=np.float64)
    reference = np.sum(generators, axis=0)
    norm = float(np.linalg.norm(reference))
    if not math.isfinite(norm) or norm <= 1.0e-12:
        raise GeometricTangencyError(f"{prefix} normal cone has no common reference")
    reference /= norm
    rho = float(np.min(generators @ reference))
    if rho <= 1.0e-8:
        raise GeometricTangencyError(f"{prefix} normal generators do not share a hemisphere")
    coefficient_cap = 1.0 / rho
    active = 1.0 if activation is None else activation
    coefficients = tuple(
        model.addVar(lb=0.0, ub=coefficient_cap, name=f"{prefix}_alpha{index}")
        for index in range(len(generators))
    )
    model.addCons(
        quicksum(coefficients) <= coefficient_cap * active,
        name=f"{prefix}_bounded_cone_scale",
    )
    normal = tuple(
        model.addVar(lb=-1.0, ub=1.0, name=f"{prefix}_n{axis}")
        for axis in range(3)
    )
    for axis in range(3):
        model.addCons(
            normal[axis]
            == quicksum(
                coefficients[index] * float(generators[index, axis])
                for index in range(len(generators))
            ),
            name=f"{prefix}_normal_axis{axis}",
        )
        if activation is not None:
            model.addCons(
                normal[axis] <= active,
                name=f"{prefix}_normal_active_upper{axis}",
            )
            model.addCons(
                normal[axis] >= -active,
                name=f"{prefix}_normal_active_lower{axis}",
            )
    for row, vertex_index in enumerate(support_vertex_indices):
        delta = np.asarray(points[int(vertex_index)], dtype=np.float64) - support_origin
        model.addCons(
            quicksum(normal[axis] * float(delta[axis]) for axis in range(3)) <= 0.0,
            name=f"{prefix}_one_ring_support{row}",
        )
    return normal, coefficients  # type: ignore[return-value]


def _add_fixed_pad_feature(
    model: Any,
    quicksum: Any,
    *,
    points: np.ndarray,
    faces: np.ndarray,
    topology: PadFeatureTopology,
    mode: FixedContactMode,
    prefix: str,
    activation: Any | None = None,
    relative_interior_margin: float = 0.0,
) -> FixedPadFeatureVariables:
    values = np.asarray(points, dtype=np.float64)
    indexed_faces = np.asarray(faces, dtype=np.int64)
    active = 1.0 if activation is None else activation
    if relative_interior_margin < 0.0 or relative_interior_margin >= 0.5:
        raise GeometricTangencyError("pad feature interior margin is invalid")
    kind = mode.pad_feature_kind
    if kind == "face":
        if not isinstance(mode.pad_feature_id, int):
            raise GeometricTangencyError("a pad face mode needs one triangle index")
        face_index = _strict_index(mode.pad_feature_id, "pad face")
        if face_index < 0 or face_index >= len(indexed_faces):
            raise GeometricTangencyError("fixed pad face is outside the mesh")
        barycentric = tuple(
            model.addVar(lb=0.0, ub=1.0, name=f"{prefix}_b{vertex}")
            for vertex in range(3)
        )
        model.addCons(
            quicksum(barycentric) == active,
            name=f"{prefix}_closed_simplex",
        )
        if relative_interior_margin >= 1.0 / 3.0:
            raise GeometricTangencyError("pad face interior margin is invalid")
        for vertex, value in enumerate(barycentric):
            if relative_interior_margin > 0.0:
                model.addCons(
                    value >= relative_interior_margin * active,
                    name=f"{prefix}_relative_interior{vertex}",
                )
        face = indexed_faces[face_index]
        point = tuple(
            quicksum(
                barycentric[vertex] * float(values[int(face[vertex]), axis])
                for vertex in range(3)
            )
            for axis in range(3)
        )
        normal = tuple(
            active * float(value) for value in topology.face_normals[face_index]
        )
        return FixedPadFeatureVariables(
            point=point,
            normal=normal,
            mode=mode,
            parameter=barycentric,
            normal_coefficients=(),
            incident_faces=(face_index,),
            support_vertex_indices=(),
        )
    if kind == "edge":
        if not isinstance(mode.pad_feature_id, tuple) or len(mode.pad_feature_id) != 2:
            raise GeometricTangencyError("a pad edge mode needs two vertex indices")
        edge = tuple(
            sorted(
                _strict_index(value, "pad edge vertex")
                for value in mode.pad_feature_id
            )
        )
        incident = topology.edge_faces.get(edge)
        if incident is None or len(incident) != 2:
            raise GeometricTangencyError("fixed pad edge is absent or is a material boundary")
        if (
            relative_interior_margin <= 0.0
            and any(vertex in topology.boundary_vertices for vertex in edge)
        ):
            raise GeometricTangencyError("pad edge touches the pad/non-pad material interface")
        if activation is None:
            parameter = model.addVar(lb=0.0, ub=1.0, name=f"{prefix}_s")
            edge_weights = (1.0 - parameter, parameter)
        else:
            edge_weights = tuple(
                model.addVar(lb=0.0, ub=1.0, name=f"{prefix}_w{slot}")
                for slot in range(2)
            )
            model.addCons(
                quicksum(edge_weights) == active,
                name=f"{prefix}_closed_edge",
            )
            parameter = edge_weights[1]
        for slot, value in enumerate(edge_weights):
            if relative_interior_margin > 0.0:
                model.addCons(
                    value >= relative_interior_margin * active,
                    name=f"{prefix}_relative_interior{slot}",
                )
        point = tuple(
            edge_weights[0] * float(values[edge[0], axis])
            + edge_weights[1] * float(values[edge[1], axis])
            for axis in range(3)
        )
        support = tuple(
            sorted(
                set(int(value) for face in incident for value in indexed_faces[face])
                - set(edge)
            )
        )
        normal, coefficients = _add_generalized_pad_normal(
            model,
            quicksum,
            normals=topology.face_normals[np.asarray(incident, dtype=np.int64)],
            points=values,
            support_origin=values[edge[0]],
            support_vertex_indices=support,
            prefix=prefix,
            activation=activation,
        )
        return FixedPadFeatureVariables(
            point=point,
            normal=normal,
            mode=FixedContactMode(mode.object_triangle_index, "edge", edge),
            parameter=(parameter,),
            normal_coefficients=coefficients,
            incident_faces=incident,
            support_vertex_indices=support,
        )
    if kind != "vertex" or not isinstance(mode.pad_feature_id, int):
        raise GeometricTangencyError("a pad vertex mode needs one vertex index")
    vertex = _strict_index(mode.pad_feature_id, "pad vertex")
    if vertex < 0 or vertex >= len(values):
        raise GeometricTangencyError("fixed pad vertex is outside the mesh")
    if vertex in topology.boundary_vertices:
        raise GeometricTangencyError("pad vertex belongs to the pad/non-pad material interface")
    incident = topology.vertex_faces[vertex]
    support = topology.vertex_neighbors[vertex]
    if len(incident) < 3 or not support:
        raise GeometricTangencyError("fixed pad vertex has an incomplete one-ring")
    normal, coefficients = _add_generalized_pad_normal(
        model,
        quicksum,
        normals=topology.face_normals[np.asarray(incident, dtype=np.int64)],
        points=values,
        support_origin=values[vertex],
        support_vertex_indices=support,
        prefix=prefix,
        activation=activation,
    )
    return FixedPadFeatureVariables(
        point=tuple(active * float(value) for value in values[vertex]),
        normal=normal,
        mode=mode,
        parameter=(),
        normal_coefficients=coefficients,
        incident_faces=incident,
        support_vertex_indices=support,
    )


def _closing_point_velocity(
    hand: ThreeFingerHandModel,
    transforms: Mapping[str, list[list[Any]]],
    hand_point: Sequence[Any],
    *,
    link_name: str,
    closing_joint_name: str,
) -> tuple[Any, Any, Any]:
    velocity: list[Any] = [0.0, 0.0, 0.0]
    affine_cache: dict[str, tuple[str, float]] = {}
    for joint_name in _ancestor_joint_names(hand, link_name):
        joint = hand.joints[joint_name]
        if not joint.movable:
            continue
        source, multiplier = _affine_source(hand, joint_name, affine_cache)
        if source != closing_joint_name:
            continue
        joint_frame = _matmul(
            transforms[joint.parent_link], _numeric_matrix(joint.origin_transform())
        )
        axis_hand = tuple(
            sum(
                joint_frame[row][column] * float(joint.axis[column])
                for column in range(3)
            )
            for row in range(3)
        )
        lever = tuple(
            hand_point[axis] - joint_frame[axis][3] for axis in range(3)
        )
        if joint.joint_type in ("revolute", "continuous"):
            linear = _cross(axis_hand, lever)
        elif joint.joint_type == "prismatic":
            linear = axis_hand
        else:
            raise GeometricTangencyError(
                f"unsupported movable joint type {joint.joint_type!r}"
            )
        for axis in range(3):
            velocity[axis] += multiplier * linear[axis]
    return tuple(velocity)  # type: ignore[return-value]


def _add_activated_cell_outer_variables(
    model: Any,
    quicksum: Any,
    cell: ObjectChartCell | PadFaceCell,
    selector: Any,
    *,
    prefix: str,
    normal_cosine_override: float | None = None,
    enforce_normal_cap: bool = True,
) -> tuple[tuple[Any, Any, Any], tuple[Any, Any, Any]]:
    """Add one zero-when-inactive AABB and convex normal-cap contribution."""

    lower = np.asarray(cell.point_lower_m, dtype=np.float64)
    upper = np.asarray(cell.point_upper_m, dtype=np.float64)
    axis = np.asarray(cell.normal_axis, dtype=np.float64)
    point = tuple(
        model.addVar(
            lb=min(0.0, float(lower[coordinate])),
            ub=max(0.0, float(upper[coordinate])),
            name=f"{prefix}_p{coordinate}",
        )
        for coordinate in range(3)
    )
    for coordinate in range(3):
        model.addCons(
            point[coordinate] >= float(lower[coordinate]) * selector,
            name=f"{prefix}_point_lower{coordinate}",
        )
        model.addCons(
            point[coordinate] <= float(upper[coordinate]) * selector,
            name=f"{prefix}_point_upper{coordinate}",
        )
    normal = tuple(
        model.addVar(
            lb=-CELL_OUTER_NORMAL_RADIUS,
            ub=CELL_OUTER_NORMAL_RADIUS,
            name=f"{prefix}_n{coordinate}",
        )
        for coordinate in range(3)
    )
    model.addCons(
        quicksum(value * value for value in normal)
        <= CELL_OUTER_NORMAL_RADIUS * CELL_OUTER_NORMAL_RADIUS * selector,
        name=f"{prefix}_normal_unit_ball",
    )
    cosine_lower_bound = (
        float(cell.normal_cosine_lower_bound)
        if normal_cosine_override is None
        else float(normal_cosine_override)
    )
    if not math.isfinite(cosine_lower_bound) or not -1.0 <= cosine_lower_bound <= 1.0:
        raise GeometricTangencyError("cell outer normal-cap override is invalid")
    if enforce_normal_cap:
        model.addCons(
            quicksum(
                float(axis[coordinate]) * normal[coordinate]
                for coordinate in range(3)
            )
            >= cosine_lower_bound * selector,
            name=f"{prefix}_normal_cap",
        )
    return point, normal


def _cell_geometry_matches(
    supplied: ObjectChartCell | PadFaceCell,
    rebuilt: ObjectChartCell | PadFaceCell,
) -> bool:
    return (
        type(supplied) is type(rebuilt)
        and supplied == rebuilt
        and np.array_equal(
            np.asarray(supplied.point_lower_m), np.asarray(rebuilt.point_lower_m)
        )
        and np.array_equal(
            np.asarray(supplied.point_upper_m), np.asarray(rebuilt.point_upper_m)
        )
        and np.array_equal(
            np.asarray(supplied.normal_axis), np.asarray(rebuilt.normal_axis)
        )
        and supplied.normal_cosine_lower_bound
        == rebuilt.normal_cosine_lower_bound
    )


def build_cell_outer_geometric_tangency_master(
    contract: AnalyticEnvelopeContract,
    object_cell_rows: Sequence[Sequence[ObjectChartCell]],
    pad_cell_rows: Sequence[Sequence[PadFaceCell]],
    *,
    fixed_state: FixedGeometricState | None = None,
) -> GeometricCellOuterBundle:
    """Build a cell-disjunctive outer relaxation of exact geometric tangency.

    A cell incumbent is never an exact surface witness: its point and normal
    may be chosen independently anywhere in an AABB and convex normal cap.
    Only trustworthy global infeasibility can eliminate the supplied cell
    product; a time limit, node limit, or absent incumbent cannot eliminate it.
    """

    try:
        from pyscipopt import Model, cos, quicksum, sin
    except ImportError as error:
        raise GeometricTangencyError("PySCIPOpt is unavailable") from error

    atlas = load_step_contact_atlas(
        contract.step_contact_atlas_path, repository_root=contract.repository_root
    )
    hand_contract = load_carts_hand_contract(
        contract.hand_contact_contract_path,
        repository_root=contract.repository_root,
    )
    hand = hand_contract.build_hand_model()
    if tuple(hand.independent_joint_names) != contract.independent_joint_names:
        raise GeometricTangencyError("hand independent-joint order changed")
    if any(pad.triangle_count != EXPECTED_PAD_FACE_COUNT for pad in hand_contract.pads):
        raise GeometricTangencyError("complete-pad triangle count changed")

    object_rows = tuple(tuple(row) for row in object_cell_rows)
    pad_rows = tuple(tuple(row) for row in pad_cell_rows)
    if (
        len(object_rows) != 3
        or len(pad_rows) != 3
        or any(not row for row in object_rows)
        or any(not row for row in pad_rows)
    ):
        raise GeometricTangencyError(
            "cell outer needs three non-empty object and pad rows"
        )
    for finger_index, (object_row, pad_row, pad) in enumerate(
        zip(object_rows, pad_rows, hand_contract.pads)
    ):
        if any(not isinstance(cell, ObjectChartCell) for cell in object_row):
            raise GeometricTangencyError("object cell row contains a wrong value")
        if any(not isinstance(cell, PadFaceCell) for cell in pad_row):
            raise GeometricTangencyError("pad cell row contains a wrong value")
        object_indices = [
            index for cell in object_row for index in cell.triangle_indices
        ]
        pad_indices = [index for cell in pad_row for index in cell.face_indices]
        if len(object_indices) != len(set(object_indices)):
            raise GeometricTangencyError("object cell row overlaps itself")
        if len(pad_indices) != len(set(pad_indices)):
            raise GeometricTangencyError("pad cell row overlaps itself")
        for cell in object_row:
            if any(
                index < 0
                or index >= atlas.triangle_count
                or int(atlas.parent_face_index[index]) != cell.parent_face_index
                for index in cell.triangle_indices
            ):
                raise GeometricTangencyError(
                    f"finger {finger_index + 1} object cell is outside its parent"
                )
            rebuilt = _make_object_chart_cell(
                atlas,
                cell.parent_face_index,
                cell.triangle_indices,
                split_path=cell.split_path,
            )
            if not _cell_geometry_matches(cell, rebuilt):
                raise GeometricTangencyError(
                    f"finger {finger_index + 1} object cell bounds were not rebuilt from CAD"
                )
        for cell in pad_row:
            if cell.pad_name != pad.name or any(
                index < 0 or index >= pad.triangle_count
                for index in cell.face_indices
            ):
                raise GeometricTangencyError(
                    f"finger {finger_index + 1} pad cell belongs to a wrong pad"
                )
            rebuilt = _make_pad_face_cell(
                pad,
                cell.face_indices,
                split_path=cell.split_path,
            )
            if not _cell_geometry_matches(cell, rebuilt):
                raise GeometricTangencyError(
                    f"finger {finger_index + 1} pad cell bounds were not rebuilt from mesh"
                )

    model = Model("te_cell_aabb_normal_cap_geometric_outer_v1")
    joint_lower, joint_upper = hand.joint_limit_vectors()
    fixed_q: np.ndarray | None = None
    fixed_quaternion: np.ndarray | None = None
    fixed_translation: np.ndarray | None = None
    if fixed_state is not None:
        fixed_q = np.asarray(fixed_state.q_contact_rad, dtype=np.float64)
        fixed_quaternion = np.asarray(
            fixed_state.quaternion_hc_wxyz, dtype=np.float64
        )
        fixed_translation = np.asarray(
            fixed_state.translation_hc_m, dtype=np.float64
        )
        if (
            fixed_q.shape != (4,)
            or fixed_quaternion.shape != (4,)
            or fixed_translation.shape != (3,)
            or not np.all(np.isfinite(fixed_q))
            or not np.all(np.isfinite(fixed_quaternion))
            or not np.all(np.isfinite(fixed_translation))
        ):
            raise GeometricTangencyError("fixed geometric state has invalid shape or values")
        if np.any(fixed_q < joint_lower) or np.any(fixed_q > joint_upper):
            raise GeometricTangencyError("fixed contact joints exceed the URDF limits")
        if fixed_quaternion[0] < 0.0 or not np.isclose(
            np.linalg.norm(fixed_quaternion), 1.0, rtol=0.0, atol=1.0e-12
        ):
            raise GeometricTangencyError(
                "fixed quaternion must be unit length with w >= 0"
            )

    q_contact = {
        name: model.addVar(
            lb=(float(joint_lower[index]) if fixed_q is None else float(fixed_q[index])),
            ub=(float(joint_upper[index]) if fixed_q is None else float(fixed_q[index])),
            name=f"qcontact_{name}",
        )
        for index, name in enumerate(hand.independent_joint_names)
    }
    quaternion_bounds = (
        (0.0, 1.0),
        (-1.0, 1.0),
        (-1.0, 1.0),
        (-1.0, 1.0),
    )
    quaternion = tuple(
        model.addVar(
            lb=(bounds[0] if fixed_quaternion is None else float(fixed_quaternion[index])),
            ub=(bounds[1] if fixed_quaternion is None else float(fixed_quaternion[index])),
            name=f"quat_{'wxyz'[index]}",
        )
        for index, bounds in enumerate(quaternion_bounds)
    )
    model.addCons(
        quicksum(value * value for value in quaternion) == 1.0,
        name="unit_quaternion",
    )
    rotation_expression = _quaternion_rotation(quaternion)
    rotation_hc = [
        [
            model.addVar(lb=-1.0, ub=1.0, name=f"Rhc_{row}{column}")
            for column in range(3)
        ]
        for row in range(3)
    ]
    for row in range(3):
        for column in range(3):
            model.addCons(
                rotation_hc[row][column] == rotation_expression[row][column],
                name=f"quaternion_rotation_{row}{column}",
            )

    hand_radius = _hand_contact_radius_bound(hand, hand_contract)
    object_radius = float(
        np.max(np.linalg.norm(atlas.triangles_m.reshape(-1, 3), axis=1))
    )
    translation_bound = hand_radius + object_radius
    if fixed_translation is not None and np.any(
        np.abs(fixed_translation) > translation_bound
    ):
        raise GeometricTangencyError("fixed translation exceeds the complete geometry bound")
    translation_hc = tuple(
        model.addVar(
            lb=(-translation_bound if fixed_translation is None else float(fixed_translation[axis])),
            ub=(translation_bound if fixed_translation is None else float(fixed_translation[axis])),
            name=f"thc_{axis}",
        )
        for axis in range(3)
    )
    transforms = _symbolic_forward_kinematics(hand, q_contact, sin, cos)
    closing_joint_by_finger = ("f1j2", "f2j1", "f3j2")
    object_selector_rows: list[tuple[Any, ...]] = []
    pad_selector_rows: list[tuple[Any, ...]] = []
    object_points: list[tuple[Any, Any, Any]] = []
    object_normals: list[tuple[Any, Any, Any]] = []
    pad_points: list[tuple[Any, Any, Any]] = []
    pad_normals: list[tuple[Any, Any, Any]] = []
    hand_points: list[tuple[Any, Any, Any]] = []

    for finger_index, (
        pad,
        closing_joint_name,
        object_row,
        pad_row,
    ) in enumerate(
        zip(
            hand_contract.pads,
            closing_joint_by_finger,
            object_rows,
            pad_rows,
        )
    ):
        prefix = f"finger{finger_index}"
        object_selectors = tuple(
            model.addVar(vtype="B", name=f"{prefix}_object_cell{index}")
            for index in range(len(object_row))
        )
        pad_selectors = tuple(
            model.addVar(vtype="B", name=f"{prefix}_pad_cell{index}")
            for index in range(len(pad_row))
        )
        model.addCons(
            quicksum(object_selectors) == 1.0,
            name=f"{prefix}_exact_one_object_cell",
        )
        model.addCons(
            quicksum(pad_selectors) == 1.0,
            name=f"{prefix}_exact_one_pad_cell",
        )
        object_contributions = tuple(
            _add_activated_cell_outer_variables(
                model,
                quicksum,
                cell,
                selector,
                prefix=f"{prefix}_object_cell{index}",
            )
            for index, (cell, selector) in enumerate(
                zip(object_row, object_selectors)
            )
        )
        pad_contributions = tuple(
            _add_activated_cell_outer_variables(
                model,
                quicksum,
                cell,
                selector,
                prefix=f"{prefix}_pad_cell{index}",
                normal_cosine_override=-1.0,
                enforce_normal_cap=False,
            )
            for index, (cell, selector) in enumerate(zip(pad_row, pad_selectors))
        )
        object_point = tuple(
            quicksum(contribution[0][axis] for contribution in object_contributions)
            for axis in range(3)
        )
        object_normal = tuple(
            quicksum(contribution[1][axis] for contribution in object_contributions)
            for axis in range(3)
        )
        pad_point = tuple(
            quicksum(contribution[0][axis] for contribution in pad_contributions)
            for axis in range(3)
        )
        pad_normal = tuple(
            quicksum(contribution[1][axis] for contribution in pad_contributions)
            for axis in range(3)
        )
        link_transform = transforms[pad.link_name]
        hand_point = tuple(
            model.addVar(
                lb=-hand_radius,
                ub=hand_radius,
                name=f"{prefix}_ph{axis}",
            )
            for axis in range(3)
        )
        for axis in range(3):
            model.addCons(
                hand_point[axis]
                == sum(
                    link_transform[axis][column] * pad_point[column]
                    for column in range(3)
                )
                + link_transform[axis][3],
                name=f"{prefix}_fk_contact_{axis}",
            )
            model.addCons(
                hand_point[axis]
                == sum(
                    rotation_hc[axis][column] * object_point[column]
                    for column in range(3)
                )
                + translation_hc[axis],
                name=f"{prefix}_shared_contact_{axis}",
            )
            model.addCons(
                sum(
                    rotation_hc[axis][column] * object_normal[column]
                    for column in range(3)
                )
                + sum(
                    link_transform[axis][column] * pad_normal[column]
                    for column in range(3)
                )
                == 0.0,
                name=f"{prefix}_opposed_normal_{axis}",
            )
        closing_velocity = _closing_point_velocity(
            hand,
            transforms,
            hand_point,
            link_name=pad.link_name,
            closing_joint_name=closing_joint_name,
        )
        normal_hand = tuple(
            sum(
                rotation_hc[axis][column] * object_normal[column]
                for column in range(3)
            )
            for axis in range(3)
        )
        model.addCons(
            sum(
                normal_hand[axis] * closing_velocity[axis]
                for axis in range(3)
            )
            <= -CLOSING_APPROACH_MARGIN_M_PER_RAD,
            name=f"{prefix}_positive_closing_approach",
        )
        object_selector_rows.append(object_selectors)
        pad_selector_rows.append(pad_selectors)
        object_points.append(object_point)  # type: ignore[arg-type]
        object_normals.append(object_normal)  # type: ignore[arg-type]
        pad_points.append(pad_point)  # type: ignore[arg-type]
        pad_normals.append(pad_normal)  # type: ignore[arg-type]
        hand_points.append(hand_point)

    model.setObjective(0.0, "minimize")
    return GeometricCellOuterBundle(
        model=model,
        contract=contract,
        atlas=atlas,
        hand_contract=hand_contract,
        hand_model=hand,
        quaternion=quaternion,
        translation_hc=translation_hc,
        q_contact=q_contact,
        object_cell_rows=(object_rows[0], object_rows[1], object_rows[2]),
        pad_cell_rows=(pad_rows[0], pad_rows[1], pad_rows[2]),
        object_selectors=(
            object_selector_rows[0],
            object_selector_rows[1],
            object_selector_rows[2],
        ),
        pad_selectors=(
            pad_selector_rows[0],
            pad_selector_rows[1],
            pad_selector_rows[2],
        ),
        object_contact_points=tuple(object_points),
        object_contact_normals=tuple(object_normals),
        pad_contact_points=tuple(pad_points),
        pad_contact_normals=tuple(pad_normals),
        hand_contact_points=tuple(hand_points),
        translation_bound_m=translation_bound,
        geometry_bounds_verified=True,
    )


def build_geometric_tangency_master(
    contract: AnalyticEnvelopeContract,
    fixed_modes: Sequence[FixedContactMode],
    *,
    fixed_state: FixedGeometricState | None = None,
) -> GeometricTangencyBundle:
    try:
        from pyscipopt import Model, cos, quicksum, sin
    except ImportError as error:
        raise GeometricTangencyError("PySCIPOpt is unavailable") from error

    atlas = load_step_contact_atlas(
        contract.step_contact_atlas_path, repository_root=contract.repository_root
    )
    hand_contract = load_carts_hand_contract(
        contract.hand_contact_contract_path,
        repository_root=contract.repository_root,
    )
    hand = hand_contract.build_hand_model()
    if tuple(hand.independent_joint_names) != contract.independent_joint_names:
        raise GeometricTangencyError("hand independent-joint order changed")
    if any(pad.triangle_count != 2442 for pad in hand_contract.pads):
        raise GeometricTangencyError("complete-pad triangle count changed")
    modes = tuple(fixed_modes)
    if len(modes) != 3 or any(not isinstance(mode, FixedContactMode) for mode in modes):
        raise GeometricTangencyError("exactly three explicit fixed contact modes are required")

    model = Model("te_fixed_mode_parent_surface_pad_feature_tangency_v2")
    joint_lower, joint_upper = hand.joint_limit_vectors()
    fixed_q: np.ndarray | None = None
    fixed_quaternion: np.ndarray | None = None
    fixed_translation: np.ndarray | None = None
    if fixed_state is not None:
        fixed_q = np.asarray(fixed_state.q_contact_rad, dtype=np.float64)
        fixed_quaternion = np.asarray(
            fixed_state.quaternion_hc_wxyz, dtype=np.float64
        )
        fixed_translation = np.asarray(fixed_state.translation_hc_m, dtype=np.float64)
        if (
            fixed_q.shape != (4,)
            or fixed_quaternion.shape != (4,)
            or fixed_translation.shape != (3,)
            or not np.all(np.isfinite(fixed_q))
            or not np.all(np.isfinite(fixed_quaternion))
            or not np.all(np.isfinite(fixed_translation))
        ):
            raise GeometricTangencyError("fixed geometric state has invalid shape or values")
        if np.any(fixed_q < joint_lower) or np.any(fixed_q > joint_upper):
            raise GeometricTangencyError("fixed contact joints exceed the URDF limits")
        if fixed_quaternion[0] < 0.0 or not np.isclose(
            np.linalg.norm(fixed_quaternion), 1.0, rtol=0.0, atol=1.0e-12
        ):
            raise GeometricTangencyError("fixed quaternion must be unit length with w >= 0")
    q_contact = {
        name: model.addVar(
            lb=(
                float(joint_lower[index])
                if fixed_q is None
                else float(fixed_q[index])
            ),
            ub=(
                float(joint_upper[index])
                if fixed_q is None
                else float(fixed_q[index])
            ),
            name=f"qcontact_{name}",
        )
        for index, name in enumerate(hand.independent_joint_names)
    }
    quaternion_bounds = (
        (0.0, 1.0),
        (-1.0, 1.0),
        (-1.0, 1.0),
        (-1.0, 1.0),
    )
    quaternion = tuple(
        model.addVar(
            lb=(bounds[0] if fixed_quaternion is None else float(fixed_quaternion[index])),
            ub=(bounds[1] if fixed_quaternion is None else float(fixed_quaternion[index])),
            name=f"quat_{'wxyz'[index]}",
        )
        for index, bounds in enumerate(quaternion_bounds)
    )
    model.addCons(
        quicksum(value * value for value in quaternion) == 1.0,
        name="unit_quaternion",
    )
    rotation_expression = _quaternion_rotation(quaternion)
    rotation_hc = [
        [
            model.addVar(lb=-1.0, ub=1.0, name=f"Rhc_{row}{column}")
            for column in range(3)
        ]
        for row in range(3)
    ]
    for row in range(3):
        for column in range(3):
            model.addCons(
                rotation_hc[row][column] == rotation_expression[row][column],
                name=f"quaternion_rotation_{row}{column}",
            )

    hand_radius = _hand_contact_radius_bound(hand, hand_contract)
    object_radius = float(
        np.max(np.linalg.norm(atlas.triangles_m.reshape(-1, 3), axis=1))
    )
    translation_bound = hand_radius + object_radius
    if fixed_translation is not None and np.any(
        np.abs(fixed_translation) > translation_bound
    ):
        raise GeometricTangencyError("fixed translation exceeds the complete geometry bound")
    translation_hc = tuple(
        model.addVar(
            lb=(
                -translation_bound
                if fixed_translation is None
                else float(fixed_translation[axis])
            ),
            ub=(
                translation_bound
                if fixed_translation is None
                else float(fixed_translation[axis])
            ),
            name=f"thc_{axis}",
        )
        for axis in range(3)
    )
    transforms = _symbolic_forward_kinematics(hand, q_contact, sin, cos)
    closing_joint_by_finger = ("f1j2", "f2j1", "f3j2")
    pad_topologies = tuple(
        _pad_feature_topology(pad.points_local_m, pad.faces)
        for pad in hand_contract.pads
    )
    object_surfaces: list[FixedObjectSurfaceVariables] = []
    pad_surfaces: list[FixedPadFeatureVariables] = []

    for finger_index, (pad, closing_joint_name, mode, topology) in enumerate(
        zip(hand_contract.pads, closing_joint_by_finger, modes, pad_topologies)
    ):
        prefix = f"finger{finger_index}"
        object_surface = _add_fixed_object_surface(
            model,
            quicksum,
            sin,
            cos,
            atlas,
            triangle_index=mode.object_triangle_index,
            prefix=f"{prefix}_object",
        )
        pad_surface = _add_fixed_pad_feature(
            model,
            quicksum,
            points=pad.points_local_m,
            faces=pad.faces,
            topology=topology,
            mode=mode,
            prefix=f"{prefix}_pad",
        )
        object_surfaces.append(object_surface)
        pad_surfaces.append(pad_surface)
        object_normal = object_surface.normal
        pad_normal = pad_surface.normal
        link_transform = transforms[pad.link_name]
        hand_point = tuple(
            model.addVar(
                lb=-hand_radius,
                ub=hand_radius,
                name=f"{prefix}_ph{axis}",
            )
            for axis in range(3)
        )
        for axis in range(3):
            model.addCons(
                hand_point[axis]
                == sum(
                    link_transform[axis][column] * pad_surface.point[column]
                    for column in range(3)
                )
                + link_transform[axis][3],
                name=f"{prefix}_fk_contact_{axis}",
            )
            model.addCons(
                hand_point[axis]
                == sum(
                    rotation_hc[axis][column] * object_surface.point[column]
                    for column in range(3)
                )
                + translation_hc[axis],
                name=f"{prefix}_shared_contact_{axis}",
            )
            model.addCons(
                sum(
                    rotation_hc[axis][column] * object_normal[column]
                    for column in range(3)
                )
                + sum(
                    link_transform[axis][column] * pad_normal[column]
                    for column in range(3)
                )
                == 0.0,
                name=f"{prefix}_opposed_normal_{axis}",
            )
        closing_velocity = _closing_point_velocity(
            hand,
            transforms,
            hand_point,
            link_name=pad.link_name,
            closing_joint_name=closing_joint_name,
        )
        normal_hand = tuple(
            sum(
                rotation_hc[axis][column] * object_normal[column]
                for column in range(3)
            )
            for axis in range(3)
        )
        model.addCons(
            sum(
                normal_hand[axis] * closing_velocity[axis]
                for axis in range(3)
            )
            <= -CLOSING_APPROACH_MARGIN_M_PER_RAD,
            name=f"{prefix}_positive_closing_approach",
        )

    model.setObjective(0.0, "minimize")
    return GeometricTangencyBundle(
        model=model,
        contract=contract,
        atlas=atlas,
        hand_contract=hand_contract,
        hand_model=hand,
        quaternion=quaternion,
        translation_hc=translation_hc,
        q_contact=q_contact,
        fixed_modes=(modes[0], modes[1], modes[2]),
        object_surfaces=tuple(object_surfaces),
        pad_surfaces=tuple(pad_surfaces),
        pad_topologies=(pad_topologies[0], pad_topologies[1], pad_topologies[2]),
        translation_bound_m=translation_bound,
    )


def build_local_mode_geometric_tangency_master(
    contract: AnalyticEnvelopeContract,
    mode_choices: Sequence[Sequence[FixedContactMode]] | None = None,
    *,
    object_triangle_choices: Sequence[Sequence[int]] | None = None,
    pad_feature_choices: Sequence[Sequence[PadFeatureChoice]] | None = None,
    fixed_state: FixedGeometricState | None = None,
) -> LocalModeGeometricTangencyBundle:
    """Build one continuous inner geometry problem over explicit local modes.

    Inputs are factorized object-triangle and pad-feature rows.  The legacy
    combined form is accepted only when it explicitly contains their complete
    Cartesian product.  These rows are an upper-bound search aid only: omitted
    global modes remain possible and a geometric incumbent is not yet a
    force/PD/collision-feasible grasp.
    """

    try:
        from pyscipopt import Model, cos, quicksum, sin
    except ImportError as error:
        raise GeometricTangencyError("PySCIPOpt is unavailable") from error

    atlas = load_step_contact_atlas(
        contract.step_contact_atlas_path, repository_root=contract.repository_root
    )
    hand_contract = load_carts_hand_contract(
        contract.hand_contact_contract_path,
        repository_root=contract.repository_root,
    )
    hand = hand_contract.build_hand_model()
    if tuple(hand.independent_joint_names) != contract.independent_joint_names:
        raise GeometricTangencyError("hand independent-joint order changed")
    if any(pad.triangle_count != 2442 for pad in hand_contract.pads):
        raise GeometricTangencyError("complete-pad triangle count changed")

    object_choice_rows: list[tuple[int, ...]] = []
    pad_choice_rows: list[tuple[PadFeatureChoice, ...]] = []
    if mode_choices is not None:
        if object_triangle_choices is not None or pad_feature_choices is not None:
            raise GeometricTangencyError(
                "combined and factorized local mode inputs cannot be mixed"
            )
        rows = tuple(tuple(row) for row in mode_choices)
        if len(rows) != 3 or any(not row for row in rows):
            raise GeometricTangencyError(
                "exactly three non-empty local mode rows are required"
            )
        for row in rows:
            if any(not isinstance(mode, FixedContactMode) for mode in row):
                raise GeometricTangencyError("local mode rows contain a wrong value")
            identities = [
                (
                    _strict_index(mode.object_triangle_index, "object triangle"),
                    mode.pad_feature_kind,
                    (
                        tuple(
                            sorted(
                                _strict_index(value, "pad edge vertex")
                                for value in mode.pad_feature_id
                            )
                        )
                        if mode.pad_feature_kind == "edge"
                        and isinstance(mode.pad_feature_id, tuple)
                        else mode.pad_feature_id
                    ),
                )
                for mode in row
            ]
            if len(set(identities)) != len(identities):
                raise GeometricTangencyError("local mode row contains duplicates")
            object_choices = tuple(
                dict.fromkeys(identity[0] for identity in identities)
            )
            feature_map: dict[
                tuple[str, int | tuple[int, int]], PadFeatureChoice
            ] = {}
            for _, kind, feature_id in identities:
                feature_map.setdefault(
                    (kind, feature_id),
                    PadFeatureChoice(kind, feature_id),
                )
            pad_choices = tuple(feature_map.values())
            expected = {
                (object_triangle, mode.kind, mode.feature_id)
                for object_triangle in object_choices
                for mode in pad_choices
            }
            if set(identities) != expected:
                raise GeometricTangencyError(
                    "local mode row must be a complete object/pad Cartesian product"
                )
            object_choice_rows.append(object_choices)
            pad_choice_rows.append(pad_choices)
    else:
        if object_triangle_choices is None or pad_feature_choices is None:
            raise GeometricTangencyError(
                "factorized mode input needs object and pad rows"
            )
        object_rows_input = tuple(tuple(row) for row in object_triangle_choices)
        pad_rows_input = tuple(tuple(row) for row in pad_feature_choices)
        if (
            len(object_rows_input) != 3
            or len(pad_rows_input) != 3
            or any(not row for row in object_rows_input)
            or any(not row for row in pad_rows_input)
        ):
            raise GeometricTangencyError(
                "factorized mode input needs three non-empty rows per side"
            )
        for object_row, pad_row in zip(object_rows_input, pad_rows_input):
            converted_objects = tuple(
                _strict_index(value, "object triangle") for value in object_row
            )
            if len(set(converted_objects)) != len(converted_objects):
                raise GeometricTangencyError("object mode row contains duplicates")
            if any(not isinstance(value, PadFeatureChoice) for value in pad_row):
                raise GeometricTangencyError("pad feature row contains a wrong value")
            feature_keys = [(value.kind, value.feature_id) for value in pad_row]
            if len(set(feature_keys)) != len(feature_keys):
                raise GeometricTangencyError("pad feature row contains duplicates")
            object_choice_rows.append(converted_objects)
            pad_choice_rows.append(tuple(pad_row))

    model = Model("te_local_mode_parent_surface_pad_feature_tangency_v1")
    joint_lower, joint_upper = hand.joint_limit_vectors()
    fixed_q: np.ndarray | None = None
    fixed_quaternion: np.ndarray | None = None
    fixed_translation: np.ndarray | None = None
    if fixed_state is not None:
        fixed_q = np.asarray(fixed_state.q_contact_rad, dtype=np.float64)
        fixed_quaternion = np.asarray(
            fixed_state.quaternion_hc_wxyz, dtype=np.float64
        )
        fixed_translation = np.asarray(
            fixed_state.translation_hc_m, dtype=np.float64
        )
        if (
            fixed_q.shape != (4,)
            or fixed_quaternion.shape != (4,)
            or fixed_translation.shape != (3,)
            or not np.all(np.isfinite(fixed_q))
            or not np.all(np.isfinite(fixed_quaternion))
            or not np.all(np.isfinite(fixed_translation))
        ):
            raise GeometricTangencyError("fixed geometric state has invalid shape or values")
        if np.any(fixed_q < joint_lower) or np.any(fixed_q > joint_upper):
            raise GeometricTangencyError("fixed contact joints exceed the URDF limits")
        if fixed_quaternion[0] < 0.0 or not np.isclose(
            np.linalg.norm(fixed_quaternion), 1.0, rtol=0.0, atol=1.0e-12
        ):
            raise GeometricTangencyError(
                "fixed quaternion must be unit length with w >= 0"
            )

    q_contact = {
        name: model.addVar(
            lb=(float(joint_lower[index]) if fixed_q is None else float(fixed_q[index])),
            ub=(float(joint_upper[index]) if fixed_q is None else float(fixed_q[index])),
            name=f"qcontact_{name}",
        )
        for index, name in enumerate(hand.independent_joint_names)
    }
    quaternion_bounds = (
        (0.0, 1.0),
        (-1.0, 1.0),
        (-1.0, 1.0),
        (-1.0, 1.0),
    )
    quaternion = tuple(
        model.addVar(
            lb=(bounds[0] if fixed_quaternion is None else float(fixed_quaternion[index])),
            ub=(bounds[1] if fixed_quaternion is None else float(fixed_quaternion[index])),
            name=f"quat_{'wxyz'[index]}",
        )
        for index, bounds in enumerate(quaternion_bounds)
    )
    model.addCons(
        quicksum(value * value for value in quaternion) == 1.0,
        name="unit_quaternion",
    )
    rotation_expression = _quaternion_rotation(quaternion)
    rotation_hc = [
        [
            model.addVar(lb=-1.0, ub=1.0, name=f"Rhc_{row}{column}")
            for column in range(3)
        ]
        for row in range(3)
    ]
    for row in range(3):
        for column in range(3):
            model.addCons(
                rotation_hc[row][column] == rotation_expression[row][column],
                name=f"quaternion_rotation_{row}{column}",
            )

    hand_radius = _hand_contact_radius_bound(hand, hand_contract)
    object_radius = float(
        np.max(np.linalg.norm(atlas.triangles_m.reshape(-1, 3), axis=1))
    )
    translation_bound = hand_radius + object_radius
    if fixed_translation is not None and np.any(
        np.abs(fixed_translation) > translation_bound
    ):
        raise GeometricTangencyError(
            "fixed translation exceeds the complete geometry bound"
        )
    translation_hc = tuple(
        model.addVar(
            lb=(-translation_bound if fixed_translation is None else float(fixed_translation[axis])),
            ub=(translation_bound if fixed_translation is None else float(fixed_translation[axis])),
            name=f"thc_{axis}",
        )
        for axis in range(3)
    )
    transforms = _symbolic_forward_kinematics(hand, q_contact, sin, cos)
    closing_joint_by_finger = ("f1j2", "f2j1", "f3j2")
    pad_topologies = tuple(
        _pad_feature_topology(pad.points_local_m, pad.faces)
        for pad in hand_contract.pads
    )
    object_selector_rows: list[tuple[Any, ...]] = []
    pad_selector_rows: list[tuple[Any, ...]] = []
    object_rows: list[tuple[FixedObjectSurfaceVariables, ...]] = []
    pad_rows: list[tuple[FixedPadFeatureVariables, ...]] = []

    for finger_index, (
        pad,
        closing_joint_name,
        object_choices,
        pad_choices,
        topology,
    ) in enumerate(
        zip(
            hand_contract.pads,
            closing_joint_by_finger,
            object_choice_rows,
            pad_choice_rows,
            pad_topologies,
        )
    ):
        prefix = f"finger{finger_index}"
        object_selectors = tuple(
            model.addVar(vtype="B", name=f"{prefix}_object_mode{mode_index}")
            for mode_index in range(len(object_choices))
        )
        model.addCons(
            quicksum(object_selectors) == 1.0,
            name=f"{prefix}_exact_one_object_mode",
        )
        pad_selectors = tuple(
            model.addVar(vtype="B", name=f"{prefix}_pad_mode{mode_index}")
            for mode_index in range(len(pad_choices))
        )
        model.addCons(
            quicksum(pad_selectors) == 1.0,
            name=f"{prefix}_exact_one_pad_mode",
        )
        object_candidates: list[FixedObjectSurfaceVariables] = []
        pad_candidates: list[FixedPadFeatureVariables] = []
        for mode_index, (triangle_index, selector) in enumerate(
            zip(object_choices, object_selectors)
        ):
            object_candidates.append(
                _add_fixed_object_surface(
                    model,
                    quicksum,
                    sin,
                    cos,
                    atlas,
                    triangle_index=triangle_index,
                    prefix=f"{prefix}_object_candidate{mode_index}",
                    activation=selector,
                )
            )
        for mode_index, (mode, selector) in enumerate(
            zip(pad_choices, pad_selectors)
        ):
            fixed_pad_mode = FixedContactMode(
                object_choices[0], mode.kind, mode.feature_id
            )
            pad_candidates.append(
                _add_fixed_pad_feature(
                    model,
                    quicksum,
                    points=pad.points_local_m,
                    faces=pad.faces,
                    topology=topology,
                    mode=fixed_pad_mode,
                    prefix=f"{prefix}_pad_candidate{mode_index}",
                    activation=selector,
                    relative_interior_margin=(
                        LOCAL_FEATURE_RELATIVE_INTERIOR_MARGIN
                    ),
                )
            )
        object_point = tuple(
            quicksum(candidate.point[axis] for candidate in object_candidates)
            for axis in range(3)
        )
        object_normal = tuple(
            quicksum(candidate.normal[axis] for candidate in object_candidates)
            for axis in range(3)
        )
        pad_point = tuple(
            quicksum(candidate.point[axis] for candidate in pad_candidates)
            for axis in range(3)
        )
        pad_normal = tuple(
            quicksum(candidate.normal[axis] for candidate in pad_candidates)
            for axis in range(3)
        )
        link_transform = transforms[pad.link_name]
        hand_point = tuple(
            model.addVar(
                lb=-hand_radius,
                ub=hand_radius,
                name=f"{prefix}_ph{axis}",
            )
            for axis in range(3)
        )
        for axis in range(3):
            model.addCons(
                hand_point[axis]
                == sum(
                    link_transform[axis][column] * pad_point[column]
                    for column in range(3)
                )
                + link_transform[axis][3],
                name=f"{prefix}_fk_contact_{axis}",
            )
            model.addCons(
                hand_point[axis]
                == sum(
                    rotation_hc[axis][column] * object_point[column]
                    for column in range(3)
                )
                + translation_hc[axis],
                name=f"{prefix}_shared_contact_{axis}",
            )
            model.addCons(
                sum(
                    rotation_hc[axis][column] * object_normal[column]
                    for column in range(3)
                )
                + sum(
                    link_transform[axis][column] * pad_normal[column]
                    for column in range(3)
                )
                == 0.0,
                name=f"{prefix}_opposed_normal_{axis}",
            )
        closing_velocity = _closing_point_velocity(
            hand,
            transforms,
            hand_point,
            link_name=pad.link_name,
            closing_joint_name=closing_joint_name,
        )
        normal_hand = tuple(
            sum(
                rotation_hc[axis][column] * object_normal[column]
                for column in range(3)
            )
            for axis in range(3)
        )
        model.addCons(
            sum(
                normal_hand[axis] * closing_velocity[axis]
                for axis in range(3)
            )
            <= -CLOSING_APPROACH_MARGIN_M_PER_RAD,
            name=f"{prefix}_positive_closing_approach",
        )
        object_selector_rows.append(object_selectors)
        pad_selector_rows.append(pad_selectors)
        object_rows.append(tuple(object_candidates))
        pad_rows.append(tuple(pad_candidates))

    model.setObjective(0.0, "minimize")
    return LocalModeGeometricTangencyBundle(
        model=model,
        contract=contract,
        atlas=atlas,
        hand_contract=hand_contract,
        hand_model=hand,
        quaternion=quaternion,
        translation_hc=translation_hc,
        q_contact=q_contact,
        object_triangle_choices=(
            object_choice_rows[0],
            object_choice_rows[1],
            object_choice_rows[2],
        ),
        pad_feature_choices=(
            pad_choice_rows[0],
            pad_choice_rows[1],
            pad_choice_rows[2],
        ),
        object_selectors=(
            object_selector_rows[0],
            object_selector_rows[1],
            object_selector_rows[2],
        ),
        pad_selectors=(
            pad_selector_rows[0],
            pad_selector_rows[1],
            pad_selector_rows[2],
        ),
        object_surfaces=(object_rows[0], object_rows[1], object_rows[2]),
        pad_surfaces=(pad_rows[0], pad_rows[1], pad_rows[2]),
        pad_topologies=(pad_topologies[0], pad_topologies[1], pad_topologies[2]),
        translation_bound_m=translation_bound,
    )


def _rotation_from_quaternion(values: Sequence[float]) -> np.ndarray:
    w, x, y, z = map(float, values)
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def _solution_value(model: Any, value: Any) -> float:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return float(model.getVal(value))


def _selected_cell_index(
    model: Any,
    selectors: Sequence[Any],
    *,
    label: str,
) -> tuple[int, list[float]]:
    values = [float(model.getVal(selector)) for selector in selectors]
    selected = [index for index, value in enumerate(values) if value > 0.5]
    if len(selected) != 1 or not math.isclose(
        sum(values), 1.0, rel_tol=0.0, abs_tol=1.0e-6
    ):
        raise GeometricTangencyError(f"outer incumbent has no unique {label} cell")
    index = selected[0]
    if values[index] < 1.0 - 1.0e-6 or any(
        value > 1.0e-6
        for other_index, value in enumerate(values)
        if other_index != index
    ):
        raise GeometricTangencyError(f"outer incumbent {label} selectors are not integral")
    return index, values


def _serialized_object_cell(cell: ObjectChartCell) -> dict[str, Any]:
    return {
        "parent_face_one_based": cell.parent_face_index,
        "triangle_indices_zero_based": list(cell.triangle_indices),
        "triangle_count": len(cell.triangle_indices),
        "split_path": list(cell.split_path),
        "point_lower_m": list(cell.point_lower_m),
        "point_upper_m": list(cell.point_upper_m),
        "normal_axis": list(cell.normal_axis),
        "normal_cosine_lower_bound": cell.normal_cosine_lower_bound,
        "normal_half_angle_rad": cell.normal_half_angle_rad,
    }


def _serialized_pad_cell(cell: PadFaceCell) -> dict[str, Any]:
    return {
        "pad_name": cell.pad_name,
        "face_indices_zero_based": list(cell.face_indices),
        "face_count": len(cell.face_indices),
        "split_path": list(cell.split_path),
        "point_lower_m": list(cell.point_lower_m),
        "point_upper_m": list(cell.point_upper_m),
        "normal_axis": list(cell.normal_axis),
        "normal_cosine_lower_bound": cell.normal_cosine_lower_bound,
        "normal_half_angle_rad": cell.normal_half_angle_rad,
    }


def _cell_outer_incumbent_report(
    bundle: GeometricCellOuterBundle,
) -> dict[str, Any]:
    model = bundle.model
    object_indices: list[int] = []
    pad_indices: list[int] = []
    object_selector_values: list[list[float]] = []
    pad_selector_values: list[list[float]] = []
    for finger_index in range(3):
        object_index, object_values = _selected_cell_index(
            model,
            bundle.object_selectors[finger_index],
            label=f"finger {finger_index + 1} object",
        )
        pad_index, pad_values = _selected_cell_index(
            model,
            bundle.pad_selectors[finger_index],
            label=f"finger {finger_index + 1} pad",
        )
        object_indices.append(object_index)
        pad_indices.append(pad_index)
        object_selector_values.append(object_values)
        pad_selector_values.append(pad_values)
    return {
        "q_contact_rad": [
            float(model.getVal(bundle.q_contact[name]))
            for name in bundle.hand_model.independent_joint_names
        ],
        "quaternion_hc_wxyz": [
            float(model.getVal(value)) for value in bundle.quaternion
        ],
        "translation_hc_m": [
            float(model.getVal(value)) for value in bundle.translation_hc
        ],
        "selected_object_cell_indices": object_indices,
        "selected_pad_cell_indices": pad_indices,
        "object_selector_values": object_selector_values,
        "pad_selector_values": pad_selector_values,
        "selected_object_cells": [
            _serialized_object_cell(
                bundle.object_cell_rows[finger][object_indices[finger]]
            )
            for finger in range(3)
        ],
        "selected_pad_cells": [
            _serialized_pad_cell(bundle.pad_cell_rows[finger][pad_indices[finger]])
            for finger in range(3)
        ],
        "relaxed_object_points_m": [
            [_solution_value(model, value) for value in row]
            for row in bundle.object_contact_points
        ],
        "relaxed_object_normals": [
            [_solution_value(model, value) for value in row]
            for row in bundle.object_contact_normals
        ],
        "relaxed_pad_points_link_local_m": [
            [_solution_value(model, value) for value in row]
            for row in bundle.pad_contact_points
        ],
        "relaxed_pad_normals_link_local": [
            [_solution_value(model, value) for value in row]
            for row in bundle.pad_contact_normals
        ],
        "relaxed_hand_contact_points_m": [
            [_solution_value(model, value) for value in row]
            for row in bundle.hand_contact_points
        ],
        "is_exact_geometric_witness": False,
    }


def solve_cell_outer(
    bundle: GeometricCellOuterBundle,
    *,
    time_limit_s: float,
    node_limit: int,
) -> dict[str, Any]:
    """Solve one cell outer problem without promoting an incumbent to a witness."""

    if not math.isfinite(time_limit_s) or time_limit_s <= 0.0 or node_limit < 1:
        raise GeometricTangencyError("solve limits must be positive and finite")
    model = bundle.model
    model.setRealParam("limits/time", float(time_limit_s))
    model.setLongintParam("limits/nodes", int(node_limit))
    model.optimize()
    status = str(model.getStatus())
    solution_count = int(model.getNSols())
    analytic_faces = np.asarray(
        [parent.face_index for parent in bundle.atlas.parent_surfaces],
        dtype=np.int64,
    )
    complete_object_indices = set(
        int(value)
        for value in np.flatnonzero(
            np.isin(bundle.atlas.parent_face_index, analytic_faces)
        )
    )
    object_row_coverage = [
        set(index for cell in row for index in cell.triangle_indices)
        == complete_object_indices
        for row in bundle.object_cell_rows
    ]
    pad_row_coverage = [
        set(index for cell in row for index in cell.face_indices)
        == set(range(bundle.hand_contract.pads[finger].triangle_count))
        for finger, row in enumerate(bundle.pad_cell_rows)
    ]
    result: dict[str, Any] = {
        "claim_scope": CELL_OUTER_CLAIM_SCOPE,
        "status": status,
        "solution_count": solution_count,
        "elapsed_s": float(model.getSolvingTime()),
        "node_count": int(model.getNNodes()),
        "variable_count": int(model.getNVars()),
        "constraint_count": int(model.getNConss()),
        "object_cell_counts_per_finger": [
            len(row) for row in bundle.object_cell_rows
        ],
        "pad_cell_counts_per_finger": [len(row) for row in bundle.pad_cell_rows],
        "object_rows_cover_all_18_analytic_parents": object_row_coverage,
        "pad_rows_cover_all_2442_faces": pad_row_coverage,
        "point_model": "ACTIVATED_OUTWARD_AABB",
        "normal_model": "ACTIVATED_OUTWARD_UNIT_BALL_OBJECT_CAP_PAD_FULL_BALL",
        "normal_outer_radius": CELL_OUTER_NORMAL_RADIUS,
        "geometry_bounds_recomputed_and_verified": bundle.geometry_bounds_verified,
        "pad_tight_normal_caps_used_for_constraints": False,
        "pad_normal_caps_used_for_split_order_only": True,
        "full_pad_face_edge_vertex_generalized_normals_outer_covered_per_finger": (
            pad_row_coverage
        ),
        "full_pad_face_edge_vertex_generalized_normals_outer_covered": all(
            pad_row_coverage
        ),
        "supplied_pad_owner_features_outer_covered": True,
        "pad_edge_vertex_cell_owner_rule": "MINIMUM_INCIDENT_FACE_INDEX",
        "exact_one_object_cell_per_finger": True,
        "exact_one_pad_cell_per_finger": True,
        "exact_surface_position_normal_coupling_encoded": False,
        "exact_q_t_hc_fk_contact_opposition_and_closing_encoded": True,
        "outer_relaxation_only": True,
        "outer_incumbent_is_exact_geometric_witness": False,
        "geometric_inner_witness_claimed": False,
        "force_or_pd_feasibility_claimed": False,
        "dynamic_success_claimed": False,
        "hardware_authorized": False,
        "pruning_rule": "ONLY_TRUSTWORTHY_GLOBAL_OUTER_INFEASIBILITY",
        "pruning_scope": "SUPPLIED_OBJECT_CHART_AND_FULL_PAD_FEATURE_CELL_PRODUCT",
        "pruning_does_not_cover_cells_omitted_from_supplied_rows": True,
        "pruning_disposition": (
            "REQUIRES_INDEPENDENT_INTERVAL_CHECK"
            if status == "infeasible"
            else "DO_NOT_PRUNE"
        ),
        "independent_interval_infeasibility_check_encoded": False,
        "ordinary_floating_scip_infeasibility_prunes": False,
        "prune_current_cell_product": False,
        "time_or_node_limit_prunes": False,
        "zero_incumbent_prunes": False,
        "incumbent": None,
    }
    if solution_count:
        result["incumbent"] = _cell_outer_incumbent_report(bundle)
    return result


def _incumbent_report(bundle: GeometricTangencyBundle) -> dict[str, Any]:
    model = bundle.model
    q = np.asarray(
        [model.getVal(bundle.q_contact[name]) for name in bundle.hand_model.independent_joint_names],
        dtype=np.float64,
    )
    quaternion = np.asarray([model.getVal(value) for value in bundle.quaternion])
    rotation = _rotation_from_quaternion(quaternion)
    translation = np.asarray([model.getVal(value) for value in bundle.translation_hc])
    transforms = bundle.hand_model.forward_kinematics(q)
    contacts: list[dict[str, Any]] = []
    for finger_index, pad in enumerate(bundle.hand_contract.pads):
        object_surface = bundle.object_surfaces[finger_index]
        pad_surface = bundle.pad_surfaces[finger_index]
        object_barycentric = np.asarray(
            [model.getVal(value) for value in object_surface.barycentric]
        )
        object_uv = np.asarray([model.getVal(value) for value in object_surface.uv])
        object_point = object_surface.parent_surface.point_from_uv(*object_uv)
        object_normal = object_surface.parent_surface.normal_from_uv(*object_uv)
        pad_point = np.asarray(
            [_solution_value(model, value) for value in pad_surface.point]
        )
        pad_normal = np.asarray(
            [_solution_value(model, value) for value in pad_surface.normal]
        )
        parameter = [_solution_value(model, value) for value in pad_surface.parameter]
        coefficients = [
            _solution_value(model, value)
            for value in pad_surface.normal_coefficients
        ]
        transform = transforms[pad.link_name]
        object_point_hand = rotation @ object_point + translation
        pad_point_hand = transform[:3, :3] @ pad_point + transform[:3, 3]
        object_normal_hand = rotation @ object_normal
        pad_normal_hand = transform[:3, :3] @ pad_normal
        opposition_angle = math.acos(
            float(
                np.clip(
                    object_normal_hand @ -pad_normal_hand
                    / (
                        np.linalg.norm(object_normal_hand)
                        * np.linalg.norm(pad_normal_hand)
                    ),
                    -1.0,
                    1.0,
                )
            )
        )
        closing_name = ("f1j2", "f2j1", "f3j2")[finger_index]
        jacobian = bundle.hand_model.geometric_jacobian(
            pad.link_name, q, point_local_m=pad_point
        )
        closing_column = bundle.hand_model.independent_joint_names.index(closing_name)
        closing_projection = float(object_normal_hand @ jacobian[:3, closing_column])
        support_residuals = [
            float(pad_normal @ (pad.points_local_m[index] - pad_point))
            for index in pad_surface.support_vertex_indices
        ]
        feature_id = pad_surface.mode.pad_feature_id
        contacts.append(
            {
                "finger": finger_index + 1,
                "object_triangle_zero_based": object_surface.triangle_index,
                "object_parent_face_one_based": object_surface.parent_surface.face_index,
                "object_parent_surface_kind": object_surface.parent_surface.kind,
                "object_barycentric": object_barycentric.tolist(),
                "object_uv": object_uv.tolist(),
                "object_triangle_chord_proxy_edge_distances_m": (
                    object_barycentric
                    * np.asarray(object_surface.triangle_altitudes_m)
                ).tolist(),
                "pad_feature_kind": pad_surface.mode.pad_feature_kind,
                "pad_feature_id": (
                    list(feature_id) if isinstance(feature_id, tuple) else feature_id
                ),
                "pad_feature_parameter": parameter,
                "pad_incident_faces_zero_based": list(pad_surface.incident_faces),
                "pad_normal_cone_coefficients": coefficients,
                "pad_normal_norm": float(np.linalg.norm(pad_normal)),
                "pad_one_ring_support_max_m": (
                    max(support_residuals) if support_residuals else None
                ),
                "contact_point_residual_m": float(
                    np.linalg.norm(object_point_hand - pad_point_hand)
                ),
                "normal_opposition_angle_rad": opposition_angle,
                "closing_normal_projection_m_per_rad": closing_projection,
            }
        )
    return {
        "q_contact_rad": q.tolist(),
        "quaternion_hc_wxyz": quaternion.tolist(),
        "translation_hc_m": translation.tolist(),
        "contacts": contacts,
    }


def solve_geometric_tangency_master(
    bundle: GeometricTangencyBundle,
    *,
    time_limit_s: float,
    node_limit: int,
) -> dict[str, Any]:
    if not math.isfinite(time_limit_s) or time_limit_s <= 0.0 or node_limit < 1:
        raise GeometricTangencyError("solve limits must be positive and finite")
    model = bundle.model
    model.setRealParam("limits/time", float(time_limit_s))
    model.setLongintParam("limits/nodes", int(node_limit))
    model.optimize()
    solution_count = int(model.getNSols())
    result: dict[str, Any] = {
        "claim_scope": CLAIM_SCOPE,
        "status": str(model.getStatus()),
        "solution_count": solution_count,
        "elapsed_s": float(model.getSolvingTime()),
        "node_count": int(model.getNNodes()),
        "required_object_forbidden_clearance_m": REQUIRED_OBJECT_FORBIDDEN_CLEARANCE_M,
        "object_forbidden_clearance_encoded": False,
        "closing_approach_margin_m_per_rad": CLOSING_APPROACH_MARGIN_M_PER_RAD,
        "fixed_contact_modes": [
            {
                "object_triangle_zero_based": mode.object_triangle_index,
                "pad_feature_kind": mode.pad_feature_kind,
                "pad_feature_id": (
                    list(mode.pad_feature_id)
                    if isinstance(mode.pad_feature_id, tuple)
                    else mode.pad_feature_id
                ),
            }
            for mode in bundle.fixed_modes
        ],
        "covers_all_object_triangles": False,
        "covers_all_pad_features_per_finger": False,
        "analytic_parent_surface_carrier_encoded": True,
        "exact_trimmed_brep_contact_claimed": False,
        "selected_pad_feature_normal_semantics_encoded": True,
        "edge_vertex_one_ring_support_encoded": True,
        "closed_feature_boundary_degeneracy_resolved": False,
        "pad_nonpad_open_interface_edge_vertex_modes_rejected": True,
        "pad_nonpad_positive_boundary_clearance_encoded": False,
        "nonpenetration_or_collision_claimed": False,
        "force_or_pd_feasibility_claimed": False,
        "dynamic_success_claimed": False,
        "hardware_authorized": False,
        "incumbent": None,
    }
    if solution_count:
        result["incumbent"] = _incumbent_report(bundle)
    return result


def restore_strict_fixed_mode_geometric_feasibility(
    contract: AnalyticEnvelopeContract,
    fixed_modes: Sequence[FixedContactMode],
    *,
    initial_q_contact_rad: Sequence[float] | None = None,
    initial_state: FixedGeometricState | None = None,
    initial_object_barycentric: Sequence[Sequence[float]] | None = None,
    solver_kind: Literal["least_squares", "slsqp_min_slack"] = "least_squares",
    max_nfev: int = 2000,
) -> dict[str, Any]:
    """Run one deterministic local numerical restoration for three fixed modes.

    This is an incumbent heuristic only.  Its strict chart/feature margins and
    final raw checks are frozen before optimization; failure never proves
    infeasibility and success is not an exact, interval, force, or dynamic
    certificate.
    """

    try:
        from scipy.optimize import Bounds, least_squares, minimize
        from scipy.spatial.transform import Rotation
    except ImportError as error:
        raise GeometricTangencyError(
            "scipy least-squares/rotation support is unavailable"
        ) from error
    max_nfev_value = _strict_index(max_nfev, "restoration max_nfev")
    if max_nfev_value < 1:
        raise GeometricTangencyError("restoration max_nfev must be positive")
    if solver_kind not in ("least_squares", "slsqp_min_slack"):
        raise GeometricTangencyError(
            "restoration solver_kind must be least_squares or slsqp_min_slack"
        )
    modes = tuple(fixed_modes)
    if len(modes) != 3 or any(not isinstance(mode, FixedContactMode) for mode in modes):
        raise GeometricTangencyError("restoration needs three fixed contact modes")
    if initial_state is not None and initial_q_contact_rad is not None:
        raise GeometricTangencyError(
            "restoration initial_state and initial_q cannot both be supplied"
        )
    if initial_state is not None and not isinstance(initial_state, FixedGeometricState):
        raise GeometricTangencyError(
            "restoration initial_state must be FixedGeometricState"
        )
    atlas = load_step_contact_atlas(
        contract.step_contact_atlas_path, repository_root=contract.repository_root
    )
    hand_contract = load_carts_hand_contract(
        contract.hand_contact_contract_path,
        repository_root=contract.repository_root,
    )
    hand = hand_contract.build_hand_model()
    joint_lower, joint_upper = hand.joint_limit_vectors()
    if len(joint_lower) != 4:
        raise GeometricTangencyError("restoration needs four independent joints")

    object_descriptors: list[dict[str, Any]] = []
    for mode in modes:
        triangle_index = _strict_index(
            mode.object_triangle_index, "restoration object triangle"
        )
        if triangle_index < 0 or triangle_index >= atlas.triangle_count:
            raise GeometricTangencyError("restoration object triangle is outside atlas")
        try:
            parent = atlas.parent_surface(
                int(atlas.parent_face_index[triangle_index])
            )
        except KeyError as error:
            raise GeometricTangencyError(
                "restoration object triangle lacks an analytic carrier"
            ) from error
        object_descriptors.append(
            {
                "triangle_index": triangle_index,
                "parent": parent,
                "triangle_uv": np.asarray(
                    atlas.triangle_uv[triangle_index], dtype=np.float64
                ),
            }
        )

    pad_topologies = tuple(
        _pad_feature_topology(pad.points_local_m, pad.faces)
        for pad in hand_contract.pads
    )
    pad_descriptors: list[dict[str, Any]] = []
    for finger_index, (mode, pad, topology) in enumerate(
        zip(modes, hand_contract.pads, pad_topologies)
    ):
        kind = mode.pad_feature_kind
        feature = mode.pad_feature_id
        descriptor: dict[str, Any] = {
            "kind": kind,
            "pad": pad,
            "topology": topology,
            "support_origin": None,
            "support_vertex_indices": (),
            "generators": np.empty((0, 3), dtype=np.float64),
            "coefficient_cap": None,
        }
        if kind == "face":
            face_index = _strict_index(feature, "restoration pad face")
            if face_index < 0 or face_index >= pad.triangle_count:
                raise GeometricTangencyError("restoration pad face is invalid")
            face = pad.faces[face_index]
            descriptor.update(
                {
                    "feature_id": face_index,
                    "vertices": np.asarray(pad.points_local_m[face]),
                    "fixed_normal": np.asarray(
                        topology.face_normals[face_index], dtype=np.float64
                    ),
                }
            )
        elif kind == "edge":
            if not isinstance(feature, tuple) or len(feature) != 2:
                raise GeometricTangencyError("restoration pad edge is invalid")
            edge = tuple(
                sorted(
                    _strict_index(value, "restoration pad edge vertex")
                    for value in feature
                )
            )
            incident = topology.edge_faces.get(edge)
            if incident is None or len(incident) != 2:
                raise GeometricTangencyError(
                    "restoration pad edge is absent or a material boundary"
                )
            support = tuple(
                sorted(
                    set(
                        int(value)
                        for face_index in incident
                        for value in pad.faces[face_index]
                    )
                    - set(edge)
                )
            )
            descriptor.update(
                {
                    "feature_id": edge,
                    "edge": edge,
                    "support_origin": np.asarray(pad.points_local_m[edge[0]]),
                    "support_vertex_indices": support,
                    "generators": np.asarray(
                        topology.face_normals[np.asarray(incident, dtype=np.int64)]
                    ),
                }
            )
        elif kind == "vertex":
            vertex = _strict_index(feature, "restoration pad vertex")
            if vertex < 0 or vertex >= pad.vertex_count:
                raise GeometricTangencyError("restoration pad vertex is invalid")
            if vertex in topology.boundary_vertices:
                raise GeometricTangencyError(
                    "restoration pad vertex is on the material boundary"
                )
            incident = topology.vertex_faces[vertex]
            support = topology.vertex_neighbors[vertex]
            if (
                len(incident) < 3
                or not support
                or not _pad_vertex_one_ring_is_connected(topology, vertex)
            ):
                raise GeometricTangencyError(
                    "restoration pad vertex has an incomplete one-ring"
                )
            descriptor.update(
                {
                    "feature_id": vertex,
                    "vertex": vertex,
                    "support_origin": np.asarray(pad.points_local_m[vertex]),
                    "support_vertex_indices": support,
                    "generators": np.asarray(
                        topology.face_normals[np.asarray(incident, dtype=np.int64)]
                    ),
                }
            )
        else:
            raise GeometricTangencyError(
                f"restoration does not support pad feature kind {kind!r}"
            )
        generators = np.asarray(descriptor["generators"], dtype=np.float64)
        if len(generators):
            reference = np.sum(generators, axis=0)
            reference_norm = float(np.linalg.norm(reference))
            if not math.isfinite(reference_norm) or reference_norm <= 1.0e-12:
                raise GeometricTangencyError(
                    f"finger {finger_index + 1} normal cone has no reference"
                )
            reference /= reference_norm
            rho = float(np.min(generators @ reference))
            if rho <= 1.0e-8:
                raise GeometricTangencyError(
                    f"finger {finger_index + 1} normal cone lacks a common hemisphere"
                )
            descriptor["coefficient_cap"] = 1.0 / rho
            descriptor["reference_normal"] = reference
            descriptor["reference_coefficients"] = np.full(
                len(generators), 1.0 / reference_norm, dtype=np.float64
            )
        else:
            descriptor["reference_normal"] = descriptor["fixed_normal"]
            descriptor["reference_coefficients"] = np.empty(0, dtype=np.float64)
        pad_descriptors.append(descriptor)

    hand_radius = _hand_contact_radius_bound(hand, hand_contract)
    object_radius = float(
        np.max(np.linalg.norm(atlas.triangles_m.reshape(-1, 3), axis=1))
    )
    translation_bound = hand_radius + object_radius
    searchable_mask = np.isin(
        atlas.parent_face_index,
        np.asarray(
            atlas.contract.proven_searchable_parent_faces, dtype=np.int64
        ),
    )
    searchable_triangles = atlas.triangles_m[searchable_mask]
    if not len(searchable_triangles):
        raise GeometricTangencyError("restoration searchable atlas is empty")
    slsqp_length_scale_m = float(
        np.linalg.norm(
            np.max(searchable_triangles.reshape(-1, 3), axis=0)
            - np.min(searchable_triangles.reshape(-1, 3), axis=0)
        )
    )
    if not math.isfinite(slsqp_length_scale_m) or slsqp_length_scale_m <= 0.0:
        raise GeometricTangencyError("restoration SLSQP length scale is invalid")

    warm_start_input_tolerance = 1.0e-12
    if initial_object_barycentric is None:
        initial_object_barycentric_source = "UNIFORM_ONE_THIRD_DEFAULT"
        object_barycentric_initial = np.full((3, 3), 1.0 / 3.0)
    else:
        try:
            object_barycentric_initial = np.asarray(
                initial_object_barycentric, dtype=np.float64
            )
        except (TypeError, ValueError) as error:
            raise GeometricTangencyError(
                "restoration initial object barycentric must be a finite 3x3 array"
            ) from error
        if (
            object_barycentric_initial.shape != (3, 3)
            or not np.all(np.isfinite(object_barycentric_initial))
            or np.any(object_barycentric_initial < 0.0)
            or np.any(object_barycentric_initial > 1.0)
            or not np.allclose(
                np.sum(object_barycentric_initial, axis=1),
                np.ones(3),
                rtol=0.0,
                atol=warm_start_input_tolerance,
            )
        ):
            raise GeometricTangencyError(
                "restoration initial object barycentric violates the closed simplex"
            )
        initial_object_barycentric_source = "EXPLICIT_VALIDATED"
    object_simplex_initial_parameters: list[tuple[float, float]] = []
    for barycentric in object_barycentric_initial:
        s = float(barycentric[0])
        denominator = 1.0 - s
        if denominator <= warm_start_input_tolerance:
            t = 0.5
        else:
            t = float(barycentric[1]) / denominator
            if (
                not math.isfinite(t)
                or t < -warm_start_input_tolerance
                or t > 1.0 + warm_start_input_tolerance
            ):
                raise GeometricTangencyError(
                    "restoration initial object barycentric cannot map to simplex parameters"
                )
            t = float(np.clip(t, 0.0, 1.0))
        object_simplex_initial_parameters.append((s, t))

    initial_q_source = "EXPLICIT"
    initial_pose_source = "KABSCH_FROM_FEATURE_REPRESENTATIVE_POINTS"
    if initial_state is not None:
        initial_q_source = "FIXED_GEOMETRIC_STATE"
        initial_pose_source = "FIXED_GEOMETRIC_STATE"
        q_initial = np.asarray(initial_state.q_contact_rad, dtype=np.float64)
        quaternion_wxyz = np.asarray(
            initial_state.quaternion_hc_wxyz, dtype=np.float64
        )
        translation_initial = np.asarray(
            initial_state.translation_hc_m, dtype=np.float64
        )
        if (
            q_initial.shape != (4,)
            or quaternion_wxyz.shape != (4,)
            or translation_initial.shape != (3,)
            or not np.all(np.isfinite(q_initial))
            or not np.all(np.isfinite(quaternion_wxyz))
            or not np.all(np.isfinite(translation_initial))
            or np.any(q_initial < joint_lower)
            or np.any(q_initial > joint_upper)
            or quaternion_wxyz[0] < 0.0
            or not np.isclose(
                np.linalg.norm(quaternion_wxyz),
                1.0,
                rtol=0.0,
                atol=warm_start_input_tolerance,
            )
            or np.any(np.abs(translation_initial) > translation_bound)
        ):
            raise GeometricTangencyError(
                "restoration initial_state has invalid q/quaternion/translation"
            )
        rotation_initial = Rotation.from_quat(
            (
                float(quaternion_wxyz[1]),
                float(quaternion_wxyz[2]),
                float(quaternion_wxyz[3]),
                float(quaternion_wxyz[0]),
            )
        ).as_matrix()
    if initial_state is None:
        if initial_q_contact_rad is None:
            initial_q_source = "ROOT_LOWER_MIDPOINT_UPPER_GRID_SELECTED"
            root = _root_geometric_cell_node(atlas, hand_contract)
            root_lower = np.min(
                np.asarray([cell.point_lower_m for cell in root.object_cell_rows[0]]),
                axis=0,
            )
            root_upper = np.max(
                np.asarray([cell.point_upper_m for cell in root.object_cell_rows[0]]),
                axis=0,
            )
            root_scale_m = float(np.linalg.norm(root_upper - root_lower))
            midpoint = 0.5 * (joint_lower + joint_upper)
            grid_values = tuple(
                (
                    float(joint_lower[index]),
                    float(midpoint[index]),
                    float(joint_upper[index]),
                )
                for index in range(4)
            )
            grid_rows: list[tuple[float, tuple[float, ...]]] = []
            for grid_index in np.ndindex(3, 3, 3, 3):
                q_row = tuple(
                    grid_values[joint][grid_index[joint]] for joint in range(4)
                )
                priority = _node_rigid_contact_compatibility_priority(
                    root,
                    atlas,
                    hand_contract,
                    hand,
                    root_scale_m=root_scale_m,
                    q_reference=q_row,
                )
                grid_rows.append((float(priority["feature_distance"]), q_row))
            q_initial = np.asarray(
                min(grid_rows, key=lambda row: (row[0], row[1]))[1]
            )
        else:
            try:
                q_initial = np.asarray(initial_q_contact_rad, dtype=np.float64)
            except (TypeError, ValueError) as error:
                raise GeometricTangencyError(
                    "restoration initial q must contain four finite values"
                ) from error
        if (
            q_initial.shape != (4,)
            or not np.all(np.isfinite(q_initial))
            or np.any(q_initial < joint_lower)
            or np.any(q_initial > joint_upper)
        ):
            raise GeometricTangencyError(
                "restoration initial q is outside the frozen joint limits"
            )
        object_representatives = []
        for descriptor, barycentric in zip(
            object_descriptors, object_barycentric_initial
        ):
            uv = barycentric @ descriptor["triangle_uv"]
            parent = descriptor["parent"]
            object_representatives.append(
                parent.point_from_uv(float(uv[0]), float(uv[1]))
            )
        initial_transforms = hand.forward_kinematics(q_initial)
        pad_representatives_hand = []
        for descriptor in pad_descriptors:
            pad = descriptor["pad"]
            if descriptor["kind"] == "face":
                point_local = np.mean(descriptor["vertices"], axis=0)
            elif descriptor["kind"] == "edge":
                edge = descriptor["edge"]
                point_local = 0.5 * (
                    pad.points_local_m[edge[0]] + pad.points_local_m[edge[1]]
                )
            else:
                point_local = pad.points_local_m[descriptor["vertex"]]
            transform = initial_transforms[pad.link_name]
            pad_representatives_hand.append(
                transform[:3, :3] @ point_local + transform[:3, 3]
            )
        source = np.asarray(object_representatives)
        target = np.asarray(pad_representatives_hand)
        source_center = np.mean(source, axis=0)
        target_center = np.mean(target, axis=0)
        left, _singular_values, right_t = np.linalg.svd(
            (source - source_center).T @ (target - target_center)
        )
        rotation_initial = right_t.T @ left.T
        if float(np.linalg.det(rotation_initial)) < 0.0:
            right_t[-1] *= -1.0
            rotation_initial = right_t.T @ left.T
        translation_initial = target_center - rotation_initial @ source_center
        if np.any(np.abs(translation_initial) > translation_bound):
            raise GeometricTangencyError(
                "Kabsch initialization exceeds the complete geometry translation bound"
            )

    x0: list[float] = []
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []

    def add_block(
        initial: Sequence[float], lower: Sequence[float], upper: Sequence[float]
    ) -> slice:
        start = len(x0)
        x0.extend(float(value) for value in initial)
        lower_bounds.extend(float(value) for value in lower)
        upper_bounds.extend(float(value) for value in upper)
        return slice(start, len(x0))

    q_slice = add_block(q_initial, joint_lower, joint_upper)
    rotation_slice = add_block(
        Rotation.from_matrix(rotation_initial).as_rotvec(),
        np.full(3, -math.pi),
        np.full(3, math.pi),
    )
    translation_slice = add_block(
        translation_initial,
        np.full(3, -translation_bound),
        np.full(3, translation_bound),
    )
    object_margin = 0.0
    pad_margin = LOCAL_FEATURE_RELATIVE_INTERIOR_MARGIN
    object_simplex_parameter_slices = tuple(
        add_block(parameters, (0.0, 0.0), (1.0, 1.0))
        for parameters in object_simplex_initial_parameters
    )
    for descriptor in pad_descriptors:
        descriptor["parameter_slice"] = None
        descriptor["coefficient_slice"] = None
        if descriptor["kind"] == "face":
            descriptor["parameter_slice"] = add_block(
                (1.0 / 3.0, 0.5), (0.0, 0.0), (1.0, 1.0)
            )
        elif descriptor["kind"] == "edge":
            descriptor["parameter_slice"] = add_block(
                (0.5,), (pad_margin,), (1.0 - pad_margin,)
            )
        if len(descriptor["generators"]):
            cap = float(descriptor["coefficient_cap"])
            descriptor["coefficient_slice"] = add_block(
                descriptor["reference_coefficients"],
                np.zeros(len(descriptor["generators"])),
                np.full(len(descriptor["generators"]), cap),
            )
    lower_array = np.asarray(lower_bounds, dtype=np.float64)
    upper_array = np.asarray(upper_bounds, dtype=np.float64)

    closing_joint_by_finger = ("f1j2", "f2j1", "f3j2")

    def simplex_from_two_parameters(
        parameters: Sequence[float], relative_margin: float
    ) -> np.ndarray:
        s, t = (float(value) for value in parameters)
        base = np.asarray(
            (s, (1.0 - s) * t, (1.0 - s) * (1.0 - t)),
            dtype=np.float64,
        )
        return relative_margin + (1.0 - 3.0 * relative_margin) * base

    def evaluate(values: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        q = np.asarray(values[q_slice])
        rotation = Rotation.from_rotvec(values[rotation_slice]).as_matrix()
        translation = np.asarray(values[translation_slice])
        transforms = hand.forward_kinematics(q)
        residuals: list[float] = []
        position_vectors: list[np.ndarray] = []
        normal_vectors: list[np.ndarray] = []
        closing_projections: list[float] = []
        closing_violations: list[float] = []
        simplex_violations: list[float] = []
        bound_violations: list[float] = []
        cone_violations: list[float] = []
        support_violations: list[float] = []
        cone_sum_minus_cap_raw: list[float] = []
        one_ring_support_raw_m: list[float] = []
        pad_normal_components: list[float] = []
        contact_rows: list[dict[str, Any]] = []
        for finger_index, (object_descriptor, pad_descriptor, object_slice) in enumerate(
            zip(object_descriptors, pad_descriptors, object_simplex_parameter_slices)
        ):
            object_barycentric = simplex_from_two_parameters(
                values[object_slice], object_margin
            )
            object_simplex = float(np.sum(object_barycentric) - 1.0)
            simplex_violations.append(abs(object_simplex))
            bound_violations.extend(
                np.maximum(0.0, object_margin - object_barycentric).tolist()
            )
            bound_violations.extend(
                np.maximum(0.0, object_barycentric - 1.0).tolist()
            )
            object_uv = object_barycentric @ object_descriptor["triangle_uv"]
            parent = object_descriptor["parent"]
            object_point = parent.point_from_uv(
                float(object_uv[0]), float(object_uv[1])
            )
            object_normal = parent.normal_from_uv(
                float(object_uv[0]), float(object_uv[1])
            )

            pad = pad_descriptor["pad"]
            pad_parameter: list[float] = []
            if pad_descriptor["kind"] == "face":
                pad_barycentric = simplex_from_two_parameters(
                    values[pad_descriptor["parameter_slice"]], pad_margin
                )
                pad_simplex = float(np.sum(pad_barycentric) - 1.0)
                simplex_violations.append(abs(pad_simplex))
                bound_violations.extend(
                    np.maximum(0.0, pad_margin - pad_barycentric).tolist()
                )
                bound_violations.extend(
                    np.maximum(0.0, pad_barycentric - 1.0).tolist()
                )
                pad_point = pad_barycentric @ pad_descriptor["vertices"]
                pad_normal = np.asarray(pad_descriptor["fixed_normal"])
                pad_parameter = pad_barycentric.tolist()
            elif pad_descriptor["kind"] == "edge":
                edge_parameter = float(
                    values[pad_descriptor["parameter_slice"]][0]
                )
                edge = pad_descriptor["edge"]
                pad_point = (
                    (1.0 - edge_parameter) * pad.points_local_m[edge[0]]
                    + edge_parameter * pad.points_local_m[edge[1]]
                )
                bound_violations.extend(
                    (
                        max(0.0, pad_margin - edge_parameter),
                        max(0.0, edge_parameter - (1.0 - pad_margin)),
                    )
                )
                pad_parameter = [edge_parameter]
                coefficients = np.asarray(
                    values[pad_descriptor["coefficient_slice"]]
                )
                pad_normal = coefficients @ pad_descriptor["generators"]
            else:
                pad_point = pad.points_local_m[pad_descriptor["vertex"]]
                coefficients = np.asarray(
                    values[pad_descriptor["coefficient_slice"]]
                )
                pad_normal = coefficients @ pad_descriptor["generators"]

            coefficients = (
                np.asarray(values[pad_descriptor["coefficient_slice"]])
                if pad_descriptor["coefficient_slice"] is not None
                else np.empty(0, dtype=np.float64)
            )
            finger_support_violations: list[float] = []
            if len(coefficients):
                cap = float(pad_descriptor["coefficient_cap"])
                cone_sum_raw = float(np.sum(coefficients) - cap)
                cone_sum_minus_cap_raw.append(cone_sum_raw)
                coefficient_violation = max(
                    0.0,
                    float(np.max(-coefficients)),
                    float(np.max(coefficients - cap)),
                    cone_sum_raw,
                )
                cone_violations.append(coefficient_violation)
                residuals.append(max(0.0, cone_sum_raw))
                support_origin = np.asarray(pad_descriptor["support_origin"])
                contact_support_raw: list[float] = []
                for vertex_index in pad_descriptor["support_vertex_indices"]:
                    support_raw = float(
                        pad_normal
                        @ (pad.points_local_m[int(vertex_index)] - support_origin)
                    )
                    contact_support_raw.append(support_raw)
                    one_ring_support_raw_m.append(support_raw)
                    support_positive = max(0.0, support_raw)
                    support_violations.append(support_positive)
                    finger_support_violations.append(support_positive)
                    residuals.append(
                        support_positive / STRICT_RESTORATION_SUPPORT_RESIDUAL_SCALE_M
                    )
            else:
                contact_support_raw = []
            pad_normal_components.extend(float(value) for value in pad_normal)
            transform = transforms[pad.link_name]
            pad_point_hand = (
                transform[:3, :3] @ pad_point + transform[:3, 3]
            )
            object_point_hand = rotation @ object_point + translation
            position_vector = pad_point_hand - object_point_hand
            object_normal_hand = rotation @ object_normal
            pad_normal_hand = transform[:3, :3] @ pad_normal
            normal_vector = object_normal_hand + pad_normal_hand
            position_vectors.append(position_vector)
            normal_vectors.append(normal_vector)
            residuals.extend(
                (
                    position_vector
                    / STRICT_RESTORATION_POSITION_RESIDUAL_SCALE_M
                ).tolist()
            )
            residuals.extend(normal_vector.tolist())
            closing_name = closing_joint_by_finger[finger_index]
            closing_column = hand.independent_joint_names.index(closing_name)
            jacobian = hand.geometric_jacobian(
                pad.link_name, q, point_local_m=pad_point
            )
            closing_projection = float(
                object_normal_hand @ jacobian[:3, closing_column]
            )
            closing_violation = max(
                0.0,
                closing_projection + CLOSING_APPROACH_MARGIN_M_PER_RAD,
            )
            closing_projections.append(closing_projection)
            closing_violations.append(closing_violation)
            residuals.append(
                closing_violation
                / STRICT_RESTORATION_CLOSING_RESIDUAL_SCALE_M_PER_RAD
            )
            contact_rows.append(
                {
                    "object_triangle_zero_based": object_descriptor["triangle_index"],
                    "object_barycentric": object_barycentric.tolist(),
                    "object_uv": object_uv.tolist(),
                    "pad_feature_kind": pad_descriptor["kind"],
                    "pad_feature_id": (
                        list(pad_descriptor["feature_id"])
                        if isinstance(pad_descriptor["feature_id"], tuple)
                        else int(pad_descriptor["feature_id"])
                    ),
                    "pad_parameter": pad_parameter,
                    "normal_cone_coefficients": coefficients.tolist(),
                    "normal_cone_sum_minus_cap_raw": (
                        cone_sum_minus_cap_raw[-1] if len(coefficients) else None
                    ),
                    "pad_normal_link_local": np.asarray(pad_normal).tolist(),
                    "one_ring_support_raw_m": contact_support_raw,
                    "one_ring_support_positive_violations_m": (
                        finger_support_violations
                    ),
                    "position_residual_vector_m": position_vector.tolist(),
                    "opposed_normal_residual_vector": normal_vector.tolist(),
                    "closing_projection_m_per_rad": closing_projection,
                }
            )
        parameter_bound_violation = max(
            [0.0]
            + np.maximum(0.0, lower_array - values).tolist()
            + np.maximum(0.0, values - upper_array).tolist()
            + bound_violations
        )
        details = {
            "q_contact_rad": q.tolist(),
            "rotation_hc_matrix": rotation.tolist(),
            "rotation_hc_quaternion_xyzw": Rotation.from_matrix(rotation).as_quat().tolist(),
            "translation_hc_m": translation.tolist(),
            "contacts": contact_rows,
            "position_residual_norms_m": [
                float(np.linalg.norm(value)) for value in position_vectors
            ],
            "opposed_normal_residual_norms": [
                float(np.linalg.norm(value)) for value in normal_vectors
            ],
            "closing_projections_m_per_rad": closing_projections,
            "closing_positive_violations_m_per_rad": closing_violations,
            "simplex_violation_max": max([0.0] + simplex_violations),
            "parameter_bound_violation_max": float(parameter_bound_violation),
            "cone_coefficient_violation_max": max([0.0] + cone_violations),
            "one_ring_support_positive_violation_max_m": max(
                [0.0] + support_violations
            ),
            "cone_sum_minus_cap_raw": cone_sum_minus_cap_raw,
            "pad_normal_components": pad_normal_components,
            "one_ring_support_raw_m": one_ring_support_raw_m,
        }
        return np.asarray(residuals, dtype=np.float64), details

    def normalized_slack_terms(details: Mapping[str, Any]) -> dict[str, float]:
        position_component = max(
            [0.0]
            + [
                abs(float(component)) / slsqp_length_scale_m
                for contact in details["contacts"]
                for component in contact["position_residual_vector_m"]
            ]
        )
        normal_component = max(
            [0.0]
            + [
                abs(float(component))
                for contact in details["contacts"]
                for component in contact["opposed_normal_residual_vector"]
            ]
        )
        closing = max(
            [0.0]
            + [
                max(
                    0.0,
                    float(projection) + CLOSING_APPROACH_MARGIN_M_PER_RAD,
                )
                / slsqp_length_scale_m
                for projection in details["closing_projections_m_per_rad"]
            ]
        )
        return {
            "position_component_over_length_scale": position_component,
            "opposed_normal_component": normal_component,
            "closing_positive_over_length_scale": closing,
        }

    def slsqp_inequalities(values_with_slack: np.ndarray) -> np.ndarray:
        values = np.asarray(values_with_slack[:-1], dtype=np.float64)
        slack = float(values_with_slack[-1])
        _residual, details = evaluate(values)
        inequalities: list[float] = []
        for contact in details["contacts"]:
            for component in contact["position_residual_vector_m"]:
                inequalities.extend(
                    (
                        slsqp_length_scale_m * slack - float(component),
                        slsqp_length_scale_m * slack + float(component),
                    )
                )
            for component in contact["opposed_normal_residual_vector"]:
                inequalities.extend(
                    (slack - float(component), slack + float(component))
                )
        for projection in details["closing_projections_m_per_rad"]:
            inequalities.append(
                slsqp_length_scale_m * slack
                - (float(projection) + CLOSING_APPROACH_MARGIN_M_PER_RAD)
            )
        inequalities.extend(
            -float(value) for value in details["cone_sum_minus_cap_raw"]
        )
        for component in details["pad_normal_components"]:
            inequalities.extend((1.0 - float(component), 1.0 + float(component)))
        inequalities.extend(
            -float(value) for value in details["one_ring_support_raw_m"]
        )
        return np.asarray(inequalities, dtype=np.float64)

    local_minimum_slack: float | None = None
    if solver_kind == "least_squares":
        result = least_squares(
            lambda values: evaluate(values)[0],
            np.asarray(x0, dtype=np.float64),
            bounds=(lower_array, upper_array),
            method="trf",
            ftol=1.0e-12,
            xtol=1.0e-12,
            gtol=1.0e-12,
            max_nfev=max_nfev_value,
            x_scale="jac",
        )
        final_values = np.asarray(result.x, dtype=np.float64)
        solver_report = {
            "status": int(result.status),
            "success": bool(result.success),
            "message": str(result.message),
            "nfev": int(result.nfev),
            "njev": None if result.njev is None else int(result.njev),
            "cost": float(result.cost),
            "optimality": float(result.optimality),
            "max_nfev": max_nfev_value,
        }
    else:
        slsqp_initial_values = np.asarray(x0, dtype=np.float64)
        for descriptor in pad_descriptors:
            coefficient_slice = descriptor["coefficient_slice"]
            if coefficient_slice is not None:
                slsqp_initial_values[coefficient_slice] = 0.0
        _initial_residual, initial_details = evaluate(slsqp_initial_values)
        initial_normalized = normalized_slack_terms(initial_details)
        initial_slack = float(
            np.nextafter(max(initial_normalized.values()), np.inf)
        )
        slsqp_x0 = np.concatenate((slsqp_initial_values, (initial_slack,)))
        slsqp_bounds = Bounds(
            np.concatenate((lower_array, (0.0,))),
            np.concatenate((upper_array, (np.inf,))),
        )
        result = minimize(
            lambda values: float(values[-1]),
            slsqp_x0,
            method="SLSQP",
            jac=lambda values: np.concatenate(
                (np.zeros(len(values) - 1, dtype=np.float64), (1.0,))
            ),
            bounds=slsqp_bounds,
            constraints=(
                {
                    "type": "ineq",
                    "fun": slsqp_inequalities,
                },
            ),
            options={
                "ftol": STRICT_RESTORATION_SLSQP_FTOL,
                "maxiter": STRICT_RESTORATION_SLSQP_MAXITER,
                "disp": False,
            },
        )
        final_values = np.asarray(result.x[:-1], dtype=np.float64)
        local_minimum_slack = float(result.x[-1])
        solver_report = {
            "status": int(result.status),
            "success": bool(result.success),
            "message": str(result.message),
            "nfev": int(result.nfev),
            "njev": None if result.njev is None else int(result.njev),
            "nit": int(result.nit),
            "fun": float(result.fun),
            "ftol": STRICT_RESTORATION_SLSQP_FTOL,
            "maxiter": STRICT_RESTORATION_SLSQP_MAXITER,
        }
    final_residual, final = evaluate(final_values)
    position_max = max(final["position_residual_norms_m"])
    normal_max = max(final["opposed_normal_residual_norms"])
    closing_max = max(final["closing_positive_violations_m_per_rad"])
    pad_normal_component_excess_max = max(
        [0.0]
        + [
            max(0.0, abs(float(value)) - 1.0)
            for value in final["pad_normal_components"]
        ]
    )
    parameter_max = max(
        final["simplex_violation_max"],
        final["parameter_bound_violation_max"],
        final["cone_coefficient_violation_max"],
        pad_normal_component_excess_max,
    )
    support_max = final["one_ring_support_positive_violation_max_m"]
    threshold_rows = (
        ("contact_position", position_max, STRICT_RESTORATION_POSITION_WITNESS_TOL_M),
        ("opposed_normal", normal_max, STRICT_RESTORATION_NORMAL_WITNESS_TOL),
        (
            "positive_closing",
            closing_max,
            STRICT_RESTORATION_CLOSING_WITNESS_TOL_M_PER_RAD,
        ),
        ("simplex_bounds_cone", parameter_max, STRICT_RESTORATION_PARAMETER_WITNESS_TOL),
        ("one_ring_support", support_max, STRICT_RESTORATION_SUPPORT_WITNESS_TOL_M),
    )
    dominant_name, dominant_raw, dominant_threshold = max(
        threshold_rows,
        key=lambda row: (
            math.inf if not math.isfinite(row[1]) else row[1] / row[2],
            row[0],
        ),
    )
    thresholds_pass = all(
        math.isfinite(raw) and raw <= threshold for _name, raw, threshold in threshold_rows
    )
    normalized_maxima = normalized_slack_terms(final)
    slack_dominant_name, slack_dominant_value = max(
        normalized_maxima.items(), key=lambda row: (row[1], row[0])
    )
    hard_constraint_raw = {
        "variable_bound_positive_violation_max": float(
            final["parameter_bound_violation_max"]
        ),
        "cone_sum_positive_violation_max": max(
            [0.0]
            + [max(0.0, float(value)) for value in final["cone_sum_minus_cap_raw"]]
        ),
        "pad_normal_component_positive_excess_max": max(
            0.0, pad_normal_component_excess_max
        ),
        "one_ring_support_positive_violation_max_m": float(support_max),
    }
    hard_constraint_raw_max = max(hard_constraint_raw.values())
    witness = final if thresholds_pass else None
    return {
        "claim_scope": "NUMERICAL_LOCAL_STRICT_FIXED_MODE_GEOMETRIC_RESTORATION_ONLY",
        "fixed_modes": [_serialized_mode(mode) for mode in modes],
        "initial_q_source": initial_q_source,
        "initial_q_contact_rad": q_initial.tolist(),
        "initial_pose_source": initial_pose_source,
        "initial_object_barycentric_source": (
            initial_object_barycentric_source
        ),
        "initial_object_barycentric": object_barycentric_initial.tolist(),
        "single_start": True,
        "random_or_multistart_used": False,
        "solver_kind": solver_kind,
        "object_chart_relative_interior_margin": object_margin,
        "pad_feature_relative_interior_margin": pad_margin,
        "residual_scales": {
            "position_m": STRICT_RESTORATION_POSITION_RESIDUAL_SCALE_M,
            "closing_m_per_rad": STRICT_RESTORATION_CLOSING_RESIDUAL_SCALE_M_PER_RAD,
            "support_m": STRICT_RESTORATION_SUPPORT_RESIDUAL_SCALE_M,
            "normal_and_parameter": 1.0,
        },
        "witness_thresholds": {
            "contact_position_max_m": STRICT_RESTORATION_POSITION_WITNESS_TOL_M,
            "opposed_normal_max": STRICT_RESTORATION_NORMAL_WITNESS_TOL,
            "closing_positive_violation_max_m_per_rad": (
                STRICT_RESTORATION_CLOSING_WITNESS_TOL_M_PER_RAD
            ),
            "simplex_bounds_cone_max": STRICT_RESTORATION_PARAMETER_WITNESS_TOL,
            "one_ring_support_positive_max_m": (
                STRICT_RESTORATION_SUPPORT_WITNESS_TOL_M
            ),
        },
        "raw_verification": {
            "contact_position_max_m": position_max,
            "opposed_normal_max": normal_max,
            "closing_positive_violation_max_m_per_rad": closing_max,
            "simplex_violation_max": final["simplex_violation_max"],
            "parameter_bound_violation_max": final[
                "parameter_bound_violation_max"
            ],
            "cone_coefficient_violation_max": final[
                "cone_coefficient_violation_max"
            ],
            "pad_normal_component_positive_excess_max": (
                pad_normal_component_excess_max
            ),
            "one_ring_support_positive_violation_max_m": support_max,
            "residual_vector_l2": float(np.linalg.norm(final_residual)),
        },
        "dominant_term": {
            "name": dominant_name,
            "raw": dominant_raw,
            "threshold": dominant_threshold,
            "threshold_ratio": (
                math.inf
                if not math.isfinite(dominant_raw)
                else dominant_raw / dominant_threshold
            ),
        },
        "slsqp_min_slack": {
            "local_minimum_slack": local_minimum_slack,
            "length_scale_m": slsqp_length_scale_m,
            "normalized_maxima": normalized_maxima,
            "normalized_dominant_term": {
                "name": slack_dominant_name,
                "value": slack_dominant_value,
            },
            "hard_constraint_raw": hard_constraint_raw,
            "hard_constraint_raw_max": hard_constraint_raw_max,
            "solver_success_or_slack_is_witness": False,
        },
        "solver": solver_report,
        "numerical_witness_thresholds_pass": thresholds_pass,
        "witness": witness,
        "final_candidate_diagnostics": final,
        "final_candidate_is_witness": thresholds_pass,
        "numerical_local_only": True,
        "exact_or_interval_proof": False,
        "failure_prunes_any_domain": False,
        "finite_ub_a_claimed": False,
        "force_pd_collision_dynamic_claimed": False,
    }


def _restore_strict_fixed_mode_slsqp_min_slack(
    contract: AnalyticEnvelopeContract,
    fixed_modes: Sequence[FixedContactMode],
    *,
    initial_q_contact_rad: Sequence[float] | None = None,
    initial_state: FixedGeometricState | None = None,
    initial_object_barycentric: Sequence[Sequence[float]] | None = None,
) -> dict[str, Any]:
    return restore_strict_fixed_mode_geometric_feasibility(
        contract,
        fixed_modes,
        initial_q_contact_rad=initial_q_contact_rad,
        initial_state=initial_state,
        initial_object_barycentric=initial_object_barycentric,
        solver_kind="slsqp_min_slack",
        max_nfev=2000,
    )


def _serialized_mode(mode: FixedContactMode) -> dict[str, Any]:
    return {
        "object_triangle_zero_based": int(mode.object_triangle_index),
        "pad_feature_kind": mode.pad_feature_kind,
        "pad_feature_id": (
            list(mode.pad_feature_id)
            if isinstance(mode.pad_feature_id, tuple)
            else int(mode.pad_feature_id)
        ),
    }


def _serialized_pad_feature(mode: PadFeatureChoice) -> dict[str, Any]:
    return {
        "pad_feature_kind": mode.kind,
        "pad_feature_id": (
            list(mode.feature_id)
            if isinstance(mode.feature_id, tuple)
            else int(mode.feature_id)
        ),
    }


def _local_incumbent_report(
    bundle: LocalModeGeometricTangencyBundle,
) -> dict[str, Any]:
    model = bundle.model
    def selected_index(
        selectors: Sequence[Any], *, finger_index: int, label: str
    ) -> tuple[int, list[float]]:
        values = [float(model.getVal(selector)) for selector in selectors]
        selected = [index for index, value in enumerate(values) if value > 0.5]
        if len(selected) != 1 or not math.isclose(
            sum(values), 1.0, rel_tol=0.0, abs_tol=1.0e-6
        ):
            raise GeometricTangencyError(
                f"finger {finger_index + 1} incumbent has no unique {label} mode"
            )
        index = selected[0]
        if values[index] < 1.0 - 1.0e-6 or any(
            value > 1.0e-6
            for other_index, value in enumerate(values)
            if other_index != index
        ):
            raise GeometricTangencyError(
                f"finger {finger_index + 1} {label} selectors are not integral"
            )
        return index, values

    object_indices: list[int] = []
    pad_indices: list[int] = []
    object_selector_values: list[list[float]] = []
    pad_selector_values: list[list[float]] = []
    for finger_index in range(3):
        object_index, object_values = selected_index(
            bundle.object_selectors[finger_index],
            finger_index=finger_index,
            label="object",
        )
        pad_index, pad_values = selected_index(
            bundle.pad_selectors[finger_index],
            finger_index=finger_index,
            label="pad",
        )
        object_indices.append(object_index)
        pad_indices.append(pad_index)
        object_selector_values.append(object_values)
        pad_selector_values.append(pad_values)

    selected_modes = tuple(
        FixedContactMode(
            bundle.object_triangle_choices[finger][object_indices[finger]],
            bundle.pad_feature_choices[finger][pad_indices[finger]].kind,
            bundle.pad_feature_choices[finger][pad_indices[finger]].feature_id,
        )
        for finger in range(3)
    )

    fixed = GeometricTangencyBundle(
        model=bundle.model,
        contract=bundle.contract,
        atlas=bundle.atlas,
        hand_contract=bundle.hand_contract,
        hand_model=bundle.hand_model,
        quaternion=bundle.quaternion,
        translation_hc=bundle.translation_hc,
        q_contact=bundle.q_contact,
        fixed_modes=selected_modes,  # type: ignore[arg-type]
        object_surfaces=tuple(
            bundle.object_surfaces[finger][object_indices[finger]]
            for finger in range(3)
        ),
        pad_surfaces=tuple(
            bundle.pad_surfaces[finger][pad_indices[finger]]
            for finger in range(3)
        ),
        pad_topologies=bundle.pad_topologies,
        translation_bound_m=bundle.translation_bound_m,
    )
    result = _incumbent_report(fixed)
    result["selected_object_mode_indices"] = object_indices
    result["selected_pad_mode_indices"] = pad_indices
    result["object_selector_values"] = object_selector_values
    result["pad_selector_values"] = pad_selector_values
    result["selected_contact_modes"] = [
        _serialized_mode(fixed.fixed_modes[finger]) for finger in range(3)
    ]
    return result


def solve_local_mode_geometric_tangency_master(
    bundle: LocalModeGeometricTangencyBundle,
    *,
    time_limit_s: float,
    node_limit: int,
) -> dict[str, Any]:
    if not math.isfinite(time_limit_s) or time_limit_s <= 0.0 or node_limit < 1:
        raise GeometricTangencyError("solve limits must be positive and finite")
    model = bundle.model
    model.setRealParam("limits/time", float(time_limit_s))
    model.setLongintParam("limits/nodes", int(node_limit))
    model.optimize()
    solution_count = int(model.getNSols())
    result: dict[str, Any] = {
        "claim_scope": LOCAL_MODE_CLAIM_SCOPE,
        "status": str(model.getStatus()),
        "solution_count": solution_count,
        "elapsed_s": float(model.getSolvingTime()),
        "node_count": int(model.getNNodes()),
        "candidate_counts_per_finger": [
            len(bundle.object_triangle_choices[finger])
            * len(bundle.pad_feature_choices[finger])
            for finger in range(3)
        ],
        "object_candidate_counts_per_finger": [
            len(row) for row in bundle.object_triangle_choices
        ],
        "pad_candidate_counts_per_finger": [
            len(row) for row in bundle.pad_feature_choices
        ],
        "object_and_pad_selectors_factorized": True,
        "selection_layout": "factorized_object_triangle_x_pad_feature",
        "cartesian_product_intended": True,
        "object_triangle_choices": [
            list(row) for row in bundle.object_triangle_choices
        ],
        "pad_feature_choices": [
            [_serialized_pad_feature(mode) for mode in row]
            for row in bundle.pad_feature_choices
        ],
        "local_mode_rows_define_global_domain": False,
        "continuous_within_each_selected_surface_feature": True,
        "required_object_forbidden_clearance_m": (
            REQUIRED_OBJECT_FORBIDDEN_CLEARANCE_M
        ),
        "object_forbidden_clearance_encoded": False,
        "closing_approach_margin_m_per_rad": CLOSING_APPROACH_MARGIN_M_PER_RAD,
        "covers_all_object_triangles": False,
        "covers_all_pad_features_per_finger": False,
        "analytic_parent_surface_carrier_encoded": True,
        "exact_trimmed_brep_contact_claimed": False,
        "selected_pad_feature_normal_semantics_encoded": True,
        "edge_vertex_one_ring_support_encoded": True,
        "local_feature_relative_interior_margin": (
            LOCAL_FEATURE_RELATIVE_INTERIOR_MARGIN
        ),
        "strict_inner_subset": True,
        "omitted_pad_relative_boundary_strip_fraction": (
            LOCAL_FEATURE_RELATIVE_INTERIOR_MARGIN
        ),
        "object_chart_internal_tessellation_boundaries_excluded": False,
        "pad_internal_edges_with_boundary_endpoints_allowed_in_relative_interior": True,
        "no_incumbent_does_not_prove_declared_mode_infeasible": True,
        "pad_nonpad_open_interface_edge_vertex_modes_rejected": True,
        "pad_nonpad_positive_boundary_clearance_encoded": False,
        "nonpenetration_or_collision_claimed": False,
        "force_or_pd_feasibility_claimed": False,
        "finite_ub_a_claimed": False,
        "dynamic_success_claimed": False,
        "hardware_authorized": False,
        "incumbent": None,
    }
    if solution_count:
        result["incumbent"] = _local_incumbent_report(bundle)
    return result


def _geometric_cell_node_summary(
    node: GeometricCellNode,
    *,
    unresolved_reason: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": list(node.path),
        "depth": len(node.path),
        "row_primitive_counts": list(node.row_primitive_counts),
        "total_candidate_count": node.total_candidate_count,
        "object_cell_counts": [len(row) for row in node.object_cell_rows],
        "pad_cell_counts": [len(row) for row in node.pad_cell_rows],
        "splittable": node.is_splittable,
    }
    if unresolved_reason is not None:
        result["unresolved_reason"] = unresolved_reason
    return result


def _pad_vertex_one_ring_is_connected(
    topology: PadFeatureTopology,
    vertex: int,
) -> bool:
    incident = frozenset(topology.vertex_faces[vertex])
    if not incident:
        return False
    adjacency = {face: set() for face in incident}
    for neighbor in topology.vertex_neighbors[vertex]:
        edge = tuple(sorted((vertex, int(neighbor))))
        owners = tuple(
            face for face in topology.edge_faces.get(edge, ()) if face in incident
        )
        for first in owners:
            adjacency[first].update(second for second in owners if second != first)
    pending = [min(incident)]
    reached: set[int] = set()
    while pending:
        face = pending.pop()
        if face in reached:
            continue
        reached.add(face)
        pending.extend(sorted(adjacency[face] - reached, reverse=True))
    return reached == set(incident)


def _expected_pad_feature_identities(
    pad: VerifiedPad,
    topology: PadFeatureTopology,
) -> frozenset[tuple[str, int | tuple[int, int]]]:
    identities: set[tuple[str, int | tuple[int, int]]] = {
        ("face", face) for face in range(pad.triangle_count)
    }
    identities.update(
        ("edge", edge)
        for edge, incident_faces in topology.edge_faces.items()
        if len(incident_faces) == 2
    )
    for vertex, incident_faces in enumerate(topology.vertex_faces):
        if (
            vertex in topology.boundary_vertices
            or len(incident_faces) < 3
            or not topology.vertex_neighbors[vertex]
        ):
            continue
        if not _pad_vertex_one_ring_is_connected(topology, vertex):
            raise GeometricTangencyError(
                f"pad {pad.name} vertex {vertex} has a disconnected one-ring"
            )
        identities.add(("vertex", vertex))
    return frozenset(identities)


def _owned_pad_feature_choices(
    pad: VerifiedPad,
    owner_face_indices: Sequence[int],
    *,
    topology: PadFeatureTopology | None = None,
) -> tuple[PadFeatureChoice, ...]:
    """Expand tagged owner faces to all legal face/edge/vertex pad features."""

    owners = tuple(sorted(int(value) for value in owner_face_indices))
    if (
        not owners
        or len(set(owners)) != len(owners)
        or owners[0] < 0
        or owners[-1] >= pad.triangle_count
    ):
        raise GeometricTangencyError("pad feature owners are invalid")
    topology = (
        _pad_feature_topology(pad.points_local_m, pad.faces)
        if topology is None
        else topology
    )
    if len(topology.face_normals) != pad.triangle_count:
        raise GeometricTangencyError("pad feature topology belongs to a wrong mesh")
    owner_set = frozenset(owners)
    edges_by_owner: dict[int, list[tuple[int, int]]] = {
        owner: [] for owner in owners
    }
    vertices_by_owner: dict[int, list[int]] = {owner: [] for owner in owners}
    for edge, incident_faces in sorted(topology.edge_faces.items()):
        if (
            len(incident_faces) == 2
            and min(incident_faces) in owner_set
        ):
            edges_by_owner[min(incident_faces)].append(edge)
    for vertex, incident_faces in enumerate(topology.vertex_faces):
        if (
            vertex not in topology.boundary_vertices
            and len(incident_faces) >= 3
            and topology.vertex_neighbors[vertex]
            and min(incident_faces) in owner_set
        ):
            if not _pad_vertex_one_ring_is_connected(topology, vertex):
                raise GeometricTangencyError(
                    f"pad {pad.name} vertex {vertex} has a disconnected one-ring"
                )
            vertices_by_owner[min(incident_faces)].append(vertex)
    result: list[PadFeatureChoice] = []
    for owner in owners:
        result.append(PadFeatureChoice("face", owner))
        result.extend(
            PadFeatureChoice("edge", edge)
            for edge in sorted(edges_by_owner[owner])
        )
        result.extend(
            PadFeatureChoice("vertex", vertex)
            for vertex in sorted(vertices_by_owner[owner])
        )
    identities = [(choice.kind, choice.feature_id) for choice in result]
    if len(identities) != len(set(identities)):
        raise GeometricTangencyError("owned pad feature expansion contains duplicates")
    return tuple(result)


def _node_exact_choices(
    node: GeometricCellNode,
    hand_contract: CARTSHandContract,
) -> tuple[
    tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
    tuple[
        tuple[PadFeatureChoice, ...],
        tuple[PadFeatureChoice, ...],
        tuple[PadFeatureChoice, ...],
    ],
]:
    object_rows = tuple(
        tuple(sorted(index for cell in row for index in cell.triangle_indices))
        for row in node.object_cell_rows
    )
    pad_rows = tuple(
        _owned_pad_feature_choices(
            hand_contract.pads[finger_index],
            sorted(face for cell in row for face in cell.face_indices),
        )
        for finger_index, row in enumerate(node.pad_cell_rows)
    )
    return (
        (object_rows[0], object_rows[1], object_rows[2]),
        (pad_rows[0], pad_rows[1], pad_rows[2]),
    )


def _object_diversity_priority(
    node: GeometricCellNode,
) -> tuple[
    float,
    float,
    tuple[tuple[int, tuple[int, ...], tuple[int, ...]], ...],
]:
    """Return a navigation-only three-object spatial/normal diversity key."""

    best_distance = -math.inf
    best_determinant = -math.inf
    best_ids: tuple[tuple[int, tuple[int, ...], tuple[int, ...]], ...] | None = None
    for first in node.object_cell_rows[0]:
        first_center = 0.5 * (
            np.asarray(first.point_lower_m) + np.asarray(first.point_upper_m)
        )
        for second in node.object_cell_rows[1]:
            second_center = 0.5 * (
                np.asarray(second.point_lower_m) + np.asarray(second.point_upper_m)
            )
            first_second_distance = float(np.linalg.norm(first_center - second_center))
            for third in node.object_cell_rows[2]:
                third_center = 0.5 * (
                    np.asarray(third.point_lower_m) + np.asarray(third.point_upper_m)
                )
                minimum_distance = min(
                    first_second_distance,
                    float(np.linalg.norm(first_center - third_center)),
                    float(np.linalg.norm(second_center - third_center)),
                )
                normal_matrix = np.asarray(
                    (first.normal_axis, second.normal_axis, third.normal_axis),
                    dtype=np.float64,
                )
                determinant = abs(float(np.linalg.det(normal_matrix)))
                stable_ids = tuple(
                    (
                        cell.parent_face_index,
                        cell.triangle_indices,
                        cell.split_path,
                    )
                    for cell in (first, second, third)
                )
                if (
                    minimum_distance > best_distance
                    or (
                        minimum_distance == best_distance
                        and determinant > best_determinant
                    )
                    or (
                        minimum_distance == best_distance
                        and determinant == best_determinant
                        and (best_ids is None or stable_ids < best_ids)
                    )
                ):
                    best_distance = minimum_distance
                    best_determinant = determinant
                    best_ids = stable_ids
    if best_ids is None or not math.isfinite(best_distance) or not math.isfinite(
        best_determinant
    ):
        raise GeometricTangencyError("object diversity priority is non-finite")
    return best_distance, best_determinant, best_ids


def _rigid_contact_feature(
    points: Sequence[np.ndarray],
    normals: Sequence[np.ndarray],
    *,
    root_scale_m: float,
) -> np.ndarray:
    if len(points) != 3 or len(normals) != 3 or root_scale_m <= 0.0:
        raise GeometricTangencyError("rigid-contact feature input is invalid")
    point_rows = [np.asarray(value, dtype=np.float64) for value in points]
    normal_rows = [np.asarray(value, dtype=np.float64) for value in normals]
    if any(value.shape != (3,) for value in point_rows + normal_rows):
        raise GeometricTangencyError("rigid-contact feature needs three-vectors")
    lengths = [float(np.linalg.norm(value)) for value in normal_rows]
    if any(not math.isfinite(value) or value <= 1.0e-14 for value in lengths):
        raise GeometricTangencyError("rigid-contact feature has a zero normal")
    normal_rows = [value / length for value, length in zip(normal_rows, lengths)]
    pairs = ((0, 1), (0, 2), (1, 2))
    distances = [
        float(np.linalg.norm(point_rows[first] - point_rows[second]))
        / root_scale_m
        for first, second in pairs
    ]
    normal_dots = [
        float(np.clip(normal_rows[first] @ normal_rows[second], -1.0, 1.0))
        for first, second in pairs
    ]
    handedness = float(np.linalg.det(np.asarray(normal_rows, dtype=np.float64)))
    feature = np.asarray(distances + normal_dots + [handedness], dtype=np.float64)
    if feature.shape != (7,) or not np.all(np.isfinite(feature)):
        raise GeometricTangencyError("rigid-contact feature is non-finite")
    return feature


def _object_cell_true_representative(
    atlas: StepContactAtlas,
    cell: ObjectChartCell,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    midpoint = 0.5 * (
        np.asarray(cell.point_lower_m, dtype=np.float64)
        + np.asarray(cell.point_upper_m, dtype=np.float64)
    )
    axis = np.asarray(cell.normal_axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    candidates: list[tuple[float, float, int, np.ndarray, np.ndarray]] = []
    for triangle_index in cell.triangle_indices:
        if int(atlas.parent_face_index[triangle_index]) != cell.parent_face_index:
            raise GeometricTangencyError(
                "object representative triangle crosses its cell parent"
            )
        parent = atlas.parent_surface(cell.parent_face_index)
        uv = np.mean(atlas.triangle_uv[triangle_index], axis=0)
        point = parent.point_from_uv(float(uv[0]), float(uv[1]))
        normal = parent.normal_from_uv(float(uv[0]), float(uv[1]))
        normal = np.asarray(normal, dtype=np.float64)
        normal = normal / np.linalg.norm(normal)
        distance = float(np.linalg.norm(point - midpoint))
        angle = math.acos(float(np.clip(normal @ axis, -1.0, 1.0)))
        candidates.append((distance, angle, int(triangle_index), point, normal))
    if not candidates:
        raise GeometricTangencyError("object cell has no representative primitive")
    distance, angle, triangle_index, point, normal = min(
        candidates, key=lambda row: (row[0], row[1], row[2])
    )
    return (
        np.asarray(point, dtype=np.float64),
        np.asarray(normal, dtype=np.float64),
        {
            "primitive_kind": "analytic_object_triangle_chart",
            "triangle_index_zero_based": triangle_index,
            "parent_face_one_based": cell.parent_face_index,
            "point_to_cell_aabb_midpoint_distance_m": distance,
            "normal_to_cell_axis_angle_rad": angle,
        },
    )


def _pad_cell_true_representative(
    pad: VerifiedPad,
    cell: PadFaceCell,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if cell.pad_name != pad.name:
        raise GeometricTangencyError("pad representative cell belongs to a wrong pad")
    midpoint = 0.5 * (
        np.asarray(cell.point_lower_m, dtype=np.float64)
        + np.asarray(cell.point_upper_m, dtype=np.float64)
    )
    axis = np.asarray(cell.normal_axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    face_indices = np.asarray(cell.face_indices, dtype=np.int64)
    triangles = pad.points_local_m[pad.faces[face_indices]]
    normals = _triangle_normals(triangles)
    candidates: list[tuple[float, float, int, np.ndarray, np.ndarray]] = []
    for row, face_index in enumerate(cell.face_indices):
        point = np.mean(triangles[row], axis=0)
        normal = np.asarray(normals[row], dtype=np.float64)
        normal = normal / np.linalg.norm(normal)
        distance = float(np.linalg.norm(point - midpoint))
        angle = math.acos(float(np.clip(normal @ axis, -1.0, 1.0)))
        candidates.append((distance, angle, int(face_index), point, normal))
    if not candidates:
        raise GeometricTangencyError("pad cell has no representative primitive")
    distance, angle, face_index, point, normal = min(
        candidates, key=lambda row: (row[0], row[1], row[2])
    )
    return (
        np.asarray(point, dtype=np.float64),
        np.asarray(normal, dtype=np.float64),
        {
            "primitive_kind": "pad_triangle_face",
            "face_index_zero_based": face_index,
            "pad_name": pad.name,
            "point_to_cell_aabb_midpoint_distance_m": distance,
            "normal_to_cell_axis_angle_rad": angle,
        },
    )


def _node_rigid_contact_compatibility_priority(
    node: GeometricCellNode,
    atlas: StepContactAtlas,
    hand_contract: CARTSHandContract,
    hand: ThreeFingerHandModel,
    *,
    root_scale_m: float,
    q_reference: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Find the closest object/pad rigid-invariant feature pair for navigation."""

    try:
        from scipy.spatial import cKDTree
    except ImportError as error:
        raise GeometricTangencyError("scipy cKDTree is unavailable") from error
    joint_lower, joint_upper = hand.joint_limit_vectors()
    if len(joint_lower) != 4 or len(joint_upper) != 4:
        raise GeometricTangencyError("rigid-contact navigation needs four joints")
    default_midpoint = q_reference is None
    try:
        q_value = (
            0.5 * (joint_lower + joint_upper)
            if q_reference is None
            else np.asarray(q_reference, dtype=np.float64)
        )
    except (TypeError, ValueError) as error:
        raise GeometricTangencyError(
            "rigid-contact navigation q must contain four finite numbers"
        ) from error
    if (
        q_value.shape != (4,)
        or not np.all(np.isfinite(q_value))
        or np.any(q_value < joint_lower)
        or np.any(q_value > joint_upper)
    ):
        raise GeometricTangencyError(
            "rigid-contact navigation q must be finite and within all joint limits"
        )
    transforms = hand.forward_kinematics(q_value)

    object_representative_rows = tuple(
        tuple((cell,) + _object_cell_true_representative(atlas, cell) for cell in row)
        for row in node.object_cell_rows
    )
    object_features: list[np.ndarray] = []
    object_ids: list[tuple[tuple[int, tuple[int, ...], tuple[int, ...]], ...]] = []
    object_representative_metadata: list[tuple[dict[str, Any], ...]] = []
    for first in object_representative_rows[0]:
        for second in object_representative_rows[1]:
            for third in object_representative_rows[2]:
                rows = (first, second, third)
                cells = tuple(row[0] for row in rows)
                object_features.append(
                    _rigid_contact_feature(
                        [row[1] for row in rows],
                        [row[2] for row in rows],
                        root_scale_m=root_scale_m,
                    )
                )
                object_ids.append(
                    tuple(
                        (
                            cell.parent_face_index,
                            cell.triangle_indices,
                            cell.split_path,
                        )
                        for cell in cells
                    )
                )
                object_representative_metadata.append(
                    tuple(row[3] for row in rows)
                )

    pad_representative_rows = tuple(
        tuple(
            (cell,)
            + _pad_cell_true_representative(hand_contract.pads[finger_index], cell)
            for cell in row
        )
        for finger_index, row in enumerate(node.pad_cell_rows)
    )
    pad_features: list[np.ndarray] = []
    pad_ids: list[tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...]] = []
    pad_representative_metadata: list[tuple[dict[str, Any], ...]] = []
    for first in pad_representative_rows[0]:
        for second in pad_representative_rows[1]:
            for third in pad_representative_rows[2]:
                rows = (first, second, third)
                cells = tuple(row[0] for row in rows)
                points_hand: list[np.ndarray] = []
                opposed_normals_hand: list[np.ndarray] = []
                for finger_index, row in enumerate(rows):
                    cell, point_local, normal_local, _metadata = row
                    pad = hand_contract.pads[finger_index]
                    transform = transforms[pad.link_name]
                    points_hand.append(
                        transform[:3, :3] @ point_local + transform[:3, 3]
                    )
                    opposed_normals_hand.append(
                        -(transform[:3, :3] @ normal_local)
                    )
                pad_features.append(
                    _rigid_contact_feature(
                        points_hand,
                        opposed_normals_hand,
                        root_scale_m=root_scale_m,
                    )
                )
                pad_ids.append(
                    tuple(
                        (cell.pad_name, cell.face_indices, cell.split_path)
                        for cell in cells
                    )
                )
                pad_representative_metadata.append(
                    tuple(row[3] for row in rows)
                )
    object_array = np.asarray(object_features, dtype=np.float64)
    pad_array = np.asarray(pad_features, dtype=np.float64)
    if not len(object_array) or not len(pad_array):
        raise GeometricTangencyError("rigid-contact compatibility domain is empty")
    tree = cKDTree(pad_array)
    nearest_distances, nearest_indices = tree.query(object_array, k=1)
    global_distance = float(np.min(nearest_distances))
    tolerance = 256.0 * np.finfo(np.float64).eps * max(1.0, global_distance)
    candidate_objects = np.flatnonzero(nearest_distances <= global_distance + tolerance)
    stable_candidates: list[tuple[float, Any, Any, int, int]] = []
    for object_index in candidate_objects:
        pad_candidates = tree.query_ball_point(
            object_array[int(object_index)], global_distance + tolerance
        )
        for pad_index in pad_candidates:
            distance = float(
                np.linalg.norm(
                    object_array[int(object_index)] - pad_array[int(pad_index)]
                )
            )
            if distance <= global_distance + tolerance:
                stable_candidates.append(
                    (
                        distance,
                        object_ids[int(object_index)],
                        pad_ids[int(pad_index)],
                        int(object_index),
                        int(pad_index),
                    )
                )
    if stable_candidates:
        distance, chosen_object_ids, chosen_pad_ids, object_index, pad_index = min(
            stable_candidates,
            key=lambda row: (row[0], row[1], row[2]),
        )
    else:
        object_index = int(np.argmin(nearest_distances))
        pad_index = int(nearest_indices[object_index])
        distance = float(nearest_distances[object_index])
        chosen_object_ids = object_ids[object_index]
        chosen_pad_ids = pad_ids[pad_index]
    return {
        "feature_distance": float(distance),
        "root_scale_m": float(root_scale_m),
        "q_reference_rad": q_value.tolist(),
        "q_reference_source": (
            "JOINT_LIMIT_MIDPOINT_DEFAULT"
            if default_midpoint
            else "EXPLICIT_VALIDATED_REFERENCE"
        ),
        "object_feature": object_array[object_index].tolist(),
        "opposed_pad_hand_feature": pad_array[pad_index].tolist(),
        "object_cell_ids": [
            {
                "parent_face_one_based": row[0],
                "triangle_indices_zero_based": list(row[1]),
                "split_path": list(row[2]),
            }
            for row in chosen_object_ids
        ],
        "pad_cell_ids": [
            {
                "pad_name": row[0],
                "face_indices_zero_based": list(row[1]),
                "split_path": list(row[2]),
            }
            for row in chosen_pad_ids
        ],
        "selected_object_representatives": [
            dict(row) for row in object_representative_metadata[object_index]
        ],
        "selected_pad_representatives": [
            dict(row) for row in pad_representative_metadata[pad_index]
        ],
        "representative_selection_rule": (
            "LEXICOGRAPHIC_POINT_DISTANCE_TO_CELL_AABB_MIDPOINT_THEN_"
            "NORMAL_ANGLE_TO_CELL_AXIS_THEN_PRIMITIVE_ID"
        ),
        "representative_point_and_normal_share_one_real_primitive": True,
        "feature_components": (
            "d12_over_root_scale",
            "d13_over_root_scale",
            "d23_over_root_scale",
            "normal_dot12",
            "normal_dot13",
            "normal_dot23",
            "signed_normal_determinant",
        ),
        "navigation_only": True,
        "geometric_bound_or_feasibility_test": False,
    }


def find_first_geometric_inner_witness(
    contract: AnalyticEnvelopeContract,
    *,
    max_nodes: int,
    exact_candidate_budget: int = 24,
    outer_time_s: float = 0.0,
    exact_time_s: float = 10.0,
    outer_node_limit: int = 100000,
    exact_node_limit: int = 100000,
) -> dict[str, Any]:
    """Boundedly search for one strict face-interior geometric witness.

    The queue uses depth-first child-zero preference, with every eighth pop
    taken FIFO from the oldest retained sibling.  Strict-local infeasibility
    does not remove the corresponding closed face boundary or generalized pad
    edge/vertex modes.  No no-good is generated.
    """

    if (
        isinstance(max_nodes, bool)
        or max_nodes < 1
        or isinstance(exact_candidate_budget, bool)
        or exact_candidate_budget < 1
        or not math.isfinite(outer_time_s)
        or outer_time_s < 0.0
        or not math.isfinite(exact_time_s)
        or exact_time_s <= 0.0
        or outer_node_limit < 1
        or exact_node_limit < 1
    ):
        raise GeometricTangencyError("hierarchical search limits are invalid")
    atlas = load_step_contact_atlas(
        contract.step_contact_atlas_path, repository_root=contract.repository_root
    )
    hand_contract = load_carts_hand_contract(
        contract.hand_contact_contract_path,
        repository_root=contract.repository_root,
    )
    hand = hand_contract.build_hand_model()
    root = _root_geometric_cell_node(atlas, hand_contract)
    complete_pad_feature_counts: list[dict[str, int]] = []
    complete_pad_feature_identity_coverage: list[bool] = []
    for pad in hand_contract.pads:
        topology = _pad_feature_topology(pad.points_local_m, pad.faces)
        complete_choices = _owned_pad_feature_choices(
            pad,
            range(pad.triangle_count),
            topology=topology,
        )
        actual_identities = [
            (choice.kind, choice.feature_id) for choice in complete_choices
        ]
        expected_identities = _expected_pad_feature_identities(pad, topology)
        identities_unique = len(actual_identities) == len(set(actual_identities))
        identity_coverage = (
            identities_unique and set(actual_identities) == set(expected_identities)
        )
        kind_counts = {
            kind: sum(choice.kind == kind for choice in complete_choices)
            for kind in ("face", "edge", "vertex")
        }
        expected_kind_counts = {
            kind: sum(identity[0] == kind for identity in expected_identities)
            for kind in ("face", "edge", "vertex")
        }
        frozen_counts = {
            "face": EXPECTED_PAD_FACE_COUNT,
            "edge": EXPECTED_PAD_INTERNAL_EDGE_COUNT,
            "vertex": EXPECTED_PAD_INTERNAL_VERTEX_COUNT,
        }
        if expected_kind_counts != frozen_counts:
            raise GeometricTangencyError(
                f"pad {pad.name} legal feature topology counts changed: "
                f"{expected_kind_counts} != {frozen_counts}"
            )
        if not identity_coverage or kind_counts != expected_kind_counts:
            raise GeometricTangencyError(
                f"pad {pad.name} owner leaves do not partition feature identities"
            )
        complete_pad_feature_counts.append(kind_counts)
        complete_pad_feature_identity_coverage.append(identity_coverage)
    root_object_lower = np.min(
        np.asarray(
            [cell.point_lower_m for cell in root.object_cell_rows[0]],
            dtype=np.float64,
        ),
        axis=0,
    )
    root_object_upper = np.max(
        np.asarray(
            [cell.point_upper_m for cell in root.object_cell_rows[0]],
            dtype=np.float64,
        ),
        axis=0,
    )
    root_scale_m = float(np.linalg.norm(root_object_upper - root_object_lower))
    if not math.isfinite(root_scale_m) or root_scale_m <= 0.0:
        raise GeometricTangencyError("root object scale is invalid")
    joint_lower, joint_upper = hand.joint_limit_vectors()
    joint_midpoint = 0.5 * (joint_lower + joint_upper)
    q_grid_values = tuple(
        (
            float(joint_lower[index]),
            float(joint_midpoint[index]),
            float(joint_upper[index]),
        )
        for index in range(4)
    )
    root_q_grid_results: list[tuple[tuple[float, ...], dict[str, Any]]] = []
    for grid_index in np.ndindex(3, 3, 3, 3):
        q_reference = tuple(
            q_grid_values[joint][grid_index[joint]] for joint in range(4)
        )
        root_q_grid_results.append(
            (
                q_reference,
                _node_rigid_contact_compatibility_priority(
                    root,
                    atlas,
                    hand_contract,
                    hand,
                    root_scale_m=root_scale_m,
                    q_reference=q_reference,
                ),
            )
        )
    q_navigation, root_navigation_priority = min(
        root_q_grid_results,
        key=lambda row: (float(row[1]["feature_distance"]), row[0]),
    )
    midpoint_tuple = tuple(float(value) for value in joint_midpoint)
    upper_tuple = tuple(float(value) for value in joint_upper)
    midpoint_priority = next(
        result for q_value, result in root_q_grid_results if q_value == midpoint_tuple
    )
    upper_priority = next(
        result for q_value, result in root_q_grid_results if q_value == upper_tuple
    )
    root_q_grid_navigation_report = {
        "state_count": len(root_q_grid_results),
        "joint_names": list(hand.independent_joint_names),
        "per_joint_values_rule": "LOWER_MIDPOINT_UPPER",
        "selected": {
            "q_reference_rad": list(q_navigation),
            "feature_distance": float(
                root_navigation_priority["feature_distance"]
            ),
        },
        "joint_limit_midpoint": {
            "q_reference_rad": list(midpoint_tuple),
            "feature_distance": float(midpoint_priority["feature_distance"]),
        },
        "all_joint_upper": {
            "q_reference_rad": list(upper_tuple),
            "feature_distance": float(upper_priority["feature_distance"]),
        },
        "navigation_only": True,
        "changes_constraints_or_pruning": False,
        "finite_q_grid_is_continuous_joint_domain_proof": False,
    }
    frontier: list[GeometricCellNode] = [root]
    unresolved_leaves: list[tuple[GeometricCellNode, str]] = []
    processed_node_count = 0
    split_count = 0
    outer_attempt_count = 0
    outer_infeasible_status_count = 0
    outer_infeasible_pruned_count = 0
    exact_attempt_count = 0
    exact_status_counts: dict[str, int] = {}
    exact_attempts: list[dict[str, Any]] = []
    compatibility_cache: dict[tuple[int, ...], dict[str, Any]] = {}
    compatibility_cache[root.path] = root_navigation_priority
    diversity_cache: dict[
        tuple[int, ...],
        tuple[
            float,
            float,
            tuple[tuple[int, tuple[int, ...], tuple[int, ...]], ...],
        ],
    ] = {}

    def compatibility(node: GeometricCellNode) -> dict[str, Any]:
        if node.path not in compatibility_cache:
            compatibility_cache[node.path] = (
                _node_rigid_contact_compatibility_priority(
                    node,
                    atlas,
                    hand_contract,
                    hand,
                    root_scale_m=root_scale_m,
                    q_reference=q_navigation,
                )
            )
        return compatibility_cache[node.path]

    def diversity(
        node: GeometricCellNode,
    ) -> tuple[
        float,
        float,
        tuple[tuple[int, tuple[int, ...], tuple[int, ...]], ...],
    ]:
        if node.path not in diversity_cache:
            diversity_cache[node.path] = _object_diversity_priority(node)
        return diversity_cache[node.path]

    while frontier and processed_node_count < max_nodes:
        use_fifo = (processed_node_count + 1) % 8 == 0
        chosen_compatibility: dict[str, Any] | None = None
        chosen_diversity: tuple[Any, ...] | None = None
        navigation_selection_policy = "FIFO_OLDEST" if use_fifo else "RIGID_COMPATIBILITY"
        if use_fifo:
            node = frontier.pop(0)
        else:
            maximum_depth = max(len(candidate.path) for candidate in frontier)
            deepest_indices = [
                index
                for index, candidate in enumerate(frontier)
                if len(candidate.path) == maximum_depth
            ]
            compatibility_priorities = {
                index: compatibility(frontier[index])
                for index in deepest_indices
            }
            diversity_priorities = {
                index: diversity(frontier[index]) for index in deepest_indices
            }
            selected_index = min(
                deepest_indices,
                key=lambda index: (
                    float(compatibility_priorities[index]["feature_distance"]),
                    -diversity_priorities[index][0],
                    -diversity_priorities[index][1],
                    diversity_priorities[index][2],
                    frontier[index].path,
                ),
            )
            chosen_compatibility = compatibility_priorities[selected_index]
            chosen_diversity = diversity_priorities[selected_index]
            node = frontier.pop(selected_index)
        processed_node_count += 1

        if outer_time_s > 0.0:
            outer_bundle = build_cell_outer_geometric_tangency_master(
                contract,
                node.object_cell_rows,
                node.pad_cell_rows,
            )
            outer_bundle.model.hideOutput()
            outer_result = solve_cell_outer(
                outer_bundle,
                time_limit_s=outer_time_s,
                node_limit=outer_node_limit,
            )
            outer_attempt_count += 1
            if outer_result["status"] == "infeasible":
                outer_infeasible_status_count += 1

        last_exact_status: str | None = None
        if node.total_candidate_count <= exact_candidate_budget:
            if chosen_compatibility is None:
                chosen_compatibility = compatibility(node)
            if chosen_diversity is None:
                chosen_diversity = diversity(node)
            object_choices, pad_choices = _node_exact_choices(node, hand_contract)
            pad_owner_face_rows = tuple(
                tuple(
                    sorted(face for cell in row for face in cell.face_indices)
                )
                for row in node.pad_cell_rows
            )
            exact_bundle = build_local_mode_geometric_tangency_master(
                contract,
                object_triangle_choices=object_choices,
                pad_feature_choices=pad_choices,
            )
            exact_bundle.model.hideOutput()
            exact_result = solve_local_mode_geometric_tangency_master(
                exact_bundle,
                time_limit_s=exact_time_s,
                node_limit=exact_node_limit,
            )
            exact_attempt_count += 1
            last_exact_status = str(exact_result["status"])
            exact_status_counts[last_exact_status] = (
                exact_status_counts.get(last_exact_status, 0) + 1
            )
            singleton = all(count == 1 for count in node.row_primitive_counts)
            exact_attempts.append(
                {
                    "path": list(node.path),
                    "depth": len(node.path),
                    "row_primitive_counts": list(node.row_primitive_counts),
                    "total_candidate_count": node.total_candidate_count,
                    "status": last_exact_status,
                    "solution_count": int(exact_result["solution_count"]),
                    "elapsed_s": float(exact_result["elapsed_s"]),
                    "node_count": int(exact_result["node_count"]),
                    "object_triangle_choices": [list(row) for row in object_choices],
                    "pad_feature_choices": [
                        [_serialized_pad_feature(choice) for choice in row]
                        for row in pad_choices
                    ],
                    "pad_feature_candidate_counts": [
                        len(row) for row in pad_choices
                    ],
                    "pad_feature_kind_counts": [
                        {
                            kind: sum(choice.kind == kind for choice in row)
                            for kind in ("face", "edge", "vertex")
                        }
                        for row in pad_choices
                    ],
                    "pad_feature_scope": (
                        "MARGIN_CONTRACTED_FACE_AND_LEGAL_INTERNAL_EDGE_"
                        "RELATIVE_INTERIORS_PLUS_LEGAL_INTERNAL_VERTICES"
                    ),
                    "pad_feature_owner_rule": "MINIMUM_INCIDENT_FACE_INDEX",
                    "local_feature_relative_interior_margin": (
                        LOCAL_FEATURE_RELATIVE_INTERIOR_MARGIN
                    ),
                    "continuous_relative_interiors_fully_covered": False,
                    "omitted_relative_boundary_strip_fraction": (
                        LOCAL_FEATURE_RELATIVE_INTERIOR_MARGIN
                    ),
                    "material_interface_features_included": False,
                    "singleton": singleton,
                    "singleton_object_triangle_triple": (
                        [int(row[0]) for row in object_choices]
                        if singleton
                        else None
                    ),
                    "singleton_pad_face_triple": (
                        [int(row[0]) for row in pad_owner_face_rows]
                        if singleton
                        else None
                    ),
                    "incumbent_found": exact_result["incumbent"] is not None,
                    "navigation_selection_policy": navigation_selection_policy,
                    "rigid_contact_compatibility_priority": chosen_compatibility,
                    "object_diversity_tie_break": {
                        "minimum_pairwise_center_distance_m": chosen_diversity[0],
                        "absolute_normal_axis_determinant": chosen_diversity[1],
                    },
                }
            )
            if exact_result["incumbent"] is not None:
                retained = [
                    _geometric_cell_node_summary(frontier_node)
                    for frontier_node in frontier
                ] + [
                    _geometric_cell_node_summary(
                        leaf, unresolved_reason=reason
                    )
                    for leaf, reason in unresolved_leaves
                ]
                return {
                    "claim_scope": INNER_WITNESS_SEARCH_CLAIM_SCOPE,
                    "status": "WITNESS",
                    "processed_node_count": processed_node_count,
                    "split_count": split_count,
                    "outer_attempt_count": outer_attempt_count,
                    "outer_infeasible_status_count": outer_infeasible_status_count,
                    "outer_infeasible_pruned_count": outer_infeasible_pruned_count,
                    "exact_attempt_count": exact_attempt_count,
                    "exact_status_counts": exact_status_counts,
                    "exact_attempts": exact_attempts,
                    "witness_node": _geometric_cell_node_summary(node),
                    "witness": exact_result,
                    "unknown_frontier_count": len(retained),
                    "unknown_frontier": retained,
                    "frontier_siblings_retained": bool(retained),
                    "navigation_heuristic": (
                        "MAX_DEPTH_THEN_MIN_RIGID_CONTACT_FEATURE_DISTANCE_"
                        "THEN_OBJECT_DIVERSITY"
                    ),
                    "root_feature_distance_scale_m": root_scale_m,
                    "root_q_grid_navigation": root_q_grid_navigation_report,
                    "q_navigation_reused_for_all_child_nodes": True,
                    "root_q_grid_navigation_only": True,
                    "root_q_grid_changes_constraints_or_pruning": False,
                    "finite_q_grid_is_continuous_joint_domain_proof": False,
                    "navigation_heuristic_is_constraint_or_pruning": False,
                    "fifo_oldest_period": 8,
                    "search_pad_feature_scope": (
                        "MARGIN_CONTRACTED_FACE_AND_LEGAL_INTERNAL_EDGE_"
                        "RELATIVE_INTERIORS_PLUS_LEGAL_INTERNAL_VERTICES"
                    ),
                    "pad_feature_owner_rule": "MINIMUM_INCIDENT_FACE_INDEX",
                    "complete_pad_feature_counts": complete_pad_feature_counts,
                    "owner_feature_identity_coverage_per_finger": (
                        complete_pad_feature_identity_coverage
                    ),
                    "all_owner_leaves_cover_declared_feature_identities": all(
                        complete_pad_feature_identity_coverage
                    ),
                    "continuous_relative_interiors_fully_covered": False,
                    "local_feature_relative_interior_margin": (
                        LOCAL_FEATURE_RELATIVE_INTERIOR_MARGIN
                    ),
                    "omitted_relative_boundary_strip_fraction": (
                        LOCAL_FEATURE_RELATIVE_INTERIOR_MARGIN
                    ),
                    "material_interface_features_included": False,
                    "complete_pad_feature_search_exhausted": False,
                    "finite_ub_a_claimed": False,
                    "dynamic_success_claimed": False,
                }

        if node.is_splittable:
            left, right = split_geometric_cell_node(node, atlas, hand_contract)
            # Append right first so ordinary LIFO processing follows child zero.
            frontier.append(right)
            frontier.append(left)
            split_count += 1
        else:
            reason = (
                "STRICT_LOCAL_" + last_exact_status.upper()
                if last_exact_status is not None
                else "EXACT_BUDGET_DID_NOT_RUN"
            )
            unresolved_leaves.append((node, reason))

    retained_frontier = [
        _geometric_cell_node_summary(node) for node in frontier
    ] + [
        _geometric_cell_node_summary(node, unresolved_reason=reason)
        for node, reason in unresolved_leaves
    ]
    return {
        "claim_scope": INNER_WITNESS_SEARCH_CLAIM_SCOPE,
        "status": "UNKNOWN_FRONTIER",
        "processed_node_count": processed_node_count,
        "max_nodes": max_nodes,
        "split_count": split_count,
        "outer_time_s": outer_time_s,
        "outer_attempt_count": outer_attempt_count,
        "outer_infeasible_status_count": outer_infeasible_status_count,
        "outer_infeasible_pruned_count": outer_infeasible_pruned_count,
        "ordinary_floating_outer_infeasibility_prunes": False,
        "independent_interval_infeasibility_check_encoded": False,
        "outer_timeout_or_unknown_pruned_count": 0,
        "exact_candidate_budget": exact_candidate_budget,
        "exact_trigger_count_role": "OBJECT_CHART_AND_PAD_OWNER_FACE_PRIMITIVES_ONLY",
        "exact_time_s": exact_time_s,
        "exact_attempt_count": exact_attempt_count,
        "exact_status_counts": exact_status_counts,
        "exact_attempts": exact_attempts,
        "unknown_frontier_count": len(retained_frontier),
        "unknown_frontier": retained_frontier,
        "frontier_siblings_retained": bool(retained_frontier),
        "navigation_heuristic": (
            "MAX_DEPTH_THEN_MIN_RIGID_CONTACT_FEATURE_DISTANCE_THEN_"
            "OBJECT_DIVERSITY"
        ),
        "root_feature_distance_scale_m": root_scale_m,
        "root_q_grid_navigation": root_q_grid_navigation_report,
        "q_navigation_reused_for_all_child_nodes": True,
        "root_q_grid_navigation_only": True,
        "root_q_grid_changes_constraints_or_pruning": False,
        "finite_q_grid_is_continuous_joint_domain_proof": False,
        "navigation_heuristic_is_constraint_or_pruning": False,
        "fifo_oldest_period": 8,
        "witness": None,
        "strict_local_infeasibility_prunes_closed_or_generalized_domain": False,
        "search_pad_feature_scope": (
            "MARGIN_CONTRACTED_FACE_AND_LEGAL_INTERNAL_EDGE_RELATIVE_"
            "INTERIORS_PLUS_LEGAL_INTERNAL_VERTICES"
        ),
        "pad_feature_owner_rule": "MINIMUM_INCIDENT_FACE_INDEX",
        "complete_pad_feature_counts": complete_pad_feature_counts,
        "owner_feature_identity_coverage_per_finger": (
            complete_pad_feature_identity_coverage
        ),
        "all_owner_leaves_cover_declared_feature_identities": all(
            complete_pad_feature_identity_coverage
        ),
        "continuous_relative_interiors_fully_covered": False,
        "local_feature_relative_interior_margin": (
            LOCAL_FEATURE_RELATIVE_INTERIOR_MARGIN
        ),
        "omitted_relative_boundary_strip_fraction": (
            LOCAL_FEATURE_RELATIVE_INTERIOR_MARGIN
        ),
        "material_interface_features_included": False,
        "complete_pad_feature_search_exhausted": False,
        "no_good_generated": False,
        "finite_ub_a_claimed": False,
        "dynamic_success_claimed": False,
    }


def _fixed_mode_argument(value: str) -> FixedContactMode:
    parts = value.split(":")
    if len(parts) != 3 or parts[1] not in ("face", "edge", "vertex"):
        raise argparse.ArgumentTypeError(
            "fixed mode must be OBJECT_TRIANGLE:face|edge|vertex:FEATURE"
        )
    try:
        object_triangle = int(parts[0])
        if parts[1] == "edge":
            vertices = tuple(int(item) for item in parts[2].split(","))
            if len(vertices) != 2:
                raise ValueError
            feature: int | tuple[int, int] = (vertices[0], vertices[1])
        else:
            feature = int(parts[2])
    except ValueError as error:
        raise argparse.ArgumentTypeError("fixed-mode indices must be integers") from error
    return FixedContactMode(object_triangle, parts[1], feature)  # type: ignore[arg-type]


def _arguments() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[5]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", default=str(repository))
    parser.add_argument(
        "--contract",
        default="src/kcg_connector/config/te_continuous_grasp_analytic_envelope_v1.yaml",
    )
    parser.add_argument(
        "--fixed-mode",
        action="append",
        required=True,
        type=_fixed_mode_argument,
        help="repeat three times: OBJECT_TRIANGLE:face|edge|vertex:FEATURE",
    )
    parser.add_argument("--fixed-q-contact", nargs=4, type=float)
    parser.add_argument("--fixed-quaternion", nargs=4, type=float)
    parser.add_argument("--fixed-translation", nargs=3, type=float)
    parser.add_argument("--time-limit-s", type=float, default=30.0)
    parser.add_argument("--node-limit", type=int, default=1)
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    root = Path(arguments.repository_root).resolve(strict=True)
    contract = load_analytic_envelope_contract(
        arguments.contract, repository_root=root
    )
    supplied_state = (
        arguments.fixed_q_contact,
        arguments.fixed_quaternion,
        arguments.fixed_translation,
    )
    if any(value is not None for value in supplied_state) and not all(
        value is not None for value in supplied_state
    ):
        raise GeometricTangencyError(
            "fixed q, quaternion and translation must be supplied together"
        )
    fixed_state = None
    if all(value is not None for value in supplied_state):
        fixed_state = FixedGeometricState(
            q_contact_rad=tuple(arguments.fixed_q_contact),
            quaternion_hc_wxyz=tuple(arguments.fixed_quaternion),
            translation_hc_m=tuple(arguments.fixed_translation),
        )
    result = solve_geometric_tangency_master(
        build_geometric_tangency_master(
            contract, arguments.fixed_mode, fixed_state=fixed_state
        ),
        time_limit_s=arguments.time_limit_s,
        node_limit=arguments.node_limit,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if arguments.output:
        output = Path(arguments.output)
        if not output.is_absolute():
            output = root / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CELL_OUTER_CLAIM_SCOPE",
    "CELL_OUTER_NORMAL_RADIUS",
    "CLAIM_SCOPE",
    "EXPECTED_ANALYTIC_OBJECT_PARENT_COUNT",
    "LOCAL_MODE_CLAIM_SCOPE",
    "ObjectChartCell",
    "PAD_INITIAL_CELL_MAX_FACE_COUNT",
    "PAD_INITIAL_CELL_MAX_NORMAL_HALF_ANGLE_RAD",
    "PadFaceCell",
    "FixedContactMode",
    "FixedGeometricState",
    "GeometricCellNode",
    "GeometricCellOuterBundle",
    "GeometricTangencyBundle",
    "GeometricTangencyError",
    "INNER_WITNESS_SEARCH_CLAIM_SCOPE",
    "LocalModeGeometricTangencyBundle",
    "PadFeatureChoice",
    "build_cell_outer_geometric_tangency_master",
    "build_geometric_tangency_master",
    "build_local_mode_geometric_tangency_master",
    "build_object_parent_root_cells",
    "build_pad_initial_cells",
    "build_root_geometric_cell_node",
    "find_first_geometric_inner_witness",
    "restore_strict_fixed_mode_geometric_feasibility",
    "solve_cell_outer",
    "solve_local_mode_geometric_tangency_master",
    "solve_geometric_tangency_master",
    "split_object_chart_cell",
    "split_pad_face_cell",
    "split_geometric_cell_node",
]
