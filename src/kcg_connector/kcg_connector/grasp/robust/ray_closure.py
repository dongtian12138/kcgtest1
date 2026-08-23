"""Object-agnostic sequential finite-PAD witness closure certification.

This module maps a unit-hypercube design point to a three-finger grasp without
stored object poses or contact coordinates.  Each finger follows an independent
one-dimensional joint path obtained from a pre-registered actuation direction.
Exact URDF kinematics are enclosed along that path with directed MP interval
arithmetic.  A complete witness--face AABB broad phase is followed by strict
IVT, monotonicity, triangle-interior, direction, and earliest-event
certificates.  Rays, endpoint chords, and residual root thresholds are not
used by the acceptance path.

The method is intentionally named a *finite PAD witness closure predictor*.
Three symmetric interior witnesses are evaluated on every verified PAD
triangle, but this is not continuous triangle-triangle CCD and not a complete
hand-trajectory collision gate.  Tangential and triangle-boundary events fail
closed as unresolved.  Mesh/witness convergence and a separate complete
collision sweep remain mandatory before dynamic use.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from enum import Enum
from fractions import Fraction
import hashlib
import heapq
import json
import math
import re
from threading import local
from types import MappingProxyType
from typing import ClassVar, Iterator, Mapping, Sequence

import numpy as np
from scipy.optimize import brentq
from scipy.spatial import cKDTree

from kcg_connector.grasp.robust.grasp_optimizer import (
    GraspCandidate,
    PlannedPadContact,
)
from kcg_connector.grasp.robust.hand_contract import (
    OBJECT_CONTACT_NORMAL_POLICY,
    PAD_SURFACE_NORMAL_POLICY,
    VerifiedPad,
)
from kcg_connector.grasp.robust.hand_model import (
    HandModelError,
    ThreeFingerHandModel,
)
from kcg_connector.grasp.robust.interval_kinematics import (
    COMPILED_POINT_PLANE_BINDING_CACHE_CAPACITY,
    DISPLAY_APPROXIMATION_ROLE,
    DirectedIntervalKinematics,
    IntervalArithmeticOptions,
    IntervalBounds,
    IntervalKinematicsError,
    IntervalLinkTransformCache,
    IntervalPlaneRootClassification,
    IntervalPlaneRootState,
    IntervalPointMotion,
    IntervalPointMotionBatch,
    IntervalRootClassification,
    IntervalRootState,
    IntervalTransverseRootCertificate,
    METHOD_ID as INTERVAL_KINEMATICS_METHOD_ID,
    MULTIPHASE_PAD_AREA_CACHE_CAPACITY,
    MULTIPHASE_POINT_CACHE_CAPACITY,
    MULTIPHASE_TRANSFORM_CACHE_CAPACITY,
    NOMINAL_ROOT_SEED_MAXIMUM_ITERATIONS,
    NOMINAL_ROOT_SEED_POLICY,
)
from kcg_connector.grasp.robust.object_model import ObjectGraspModel
from kcg_connector.grasp.robust.surface_visibility import (
    NUMERICAL_POLICY,
    FirstHitResult,
    TriangleFirstHitIntersector,
)
from kcg_connector.grasp.robust.triangle_canonicalization import (
    RegisteredTaskFrame,
    TriangleCanonicalizationError,
    canonicalize_unoriented_triangles,
)


METHOD_ID = (
    "CARTS_SEQUENTIAL_MP_INTERVAL_FINITE_PAD_WITNESS_CLOSURE_CERTIFIER_V9"
)
INTERNAL_FORCE_ROLE = "PLANNING_PLACEHOLDER_ZERO_NOT_FORCE_COMMAND"
TRAJECTORY_CLEARANCE_ROLE = (
    "CERTIFIED_FINITE_PAD_WITNESS_PATH_LOWER_BOUND_NOT_FULL_HAND_CLEARANCE"
)
CLOSURE_FOCUS_METHOD = (
    "FINGER_BALANCED_FULL_TRIANGLE_AREA_CENTROIDS_AT_REGISTERED_"
    "FULL_CLOSED_ENDPOINT_V1"
)
RAY_EVALUATION_POLICY = (
    "NO_RAY_CALL_MAY_CREATE_OR_ACCEPT_A_CONTACT_EVENT"
)
FEATURE_ROOT_POLICY = (
    "COMPLETE_INTERVAL_AABB_WITNESS_FACE_PAIRS_THEN_STRICT_IVT_"
    "MONOTONE_INTERIOR_DIRECTIONAL_ROOT_WITH_POSSIBLE_EARLIEST_SET_V2"
)
POSSIBLE_FIRST_CONTACT_SET_METHOD_ID = (
    "CARTS_COMPLETE_POSSIBLE_EARLIEST_IMPLICIT_ROOT_SET_V1"
)
POSSIBLE_EARLIEST_ORDERING_POLICY = (
    "EXCLUDE_IFF_ROOT_LOWER_IS_STRICTLY_GREATER_THAN_MINIMUM_ROOT_UPPER_"
    "THEN_CANONICAL_BINARY64_INTERVAL_AND_FEATURE_ID_ORDER_V1"
)
CANDIDATE_REPRESENTATIVE_ROLE = (
    "REPRESENTATIVE_PROPOSAL_FROM_IMPLICIT_ROOT_SET_NOT_EXACT_CONTACT_ENDPOINT"
)
NO_CANDIDATE_ROLE = "NO_CANDIDATE"
REPRESENTATIVE_PROPOSAL_FAILURE_REASON = (
    "REPRESENTATIVE_PROPOSAL_ONLY_PENDING_ROOT_INTERVAL_PROPAGATION"
)
SEQUENTIAL_CLOSURE_POLICY_METHOD_ID = (
    "CARTS_CERTIFIED_SEQUENTIAL_CONTACT_STOP_CLOSURE_POLICY_V1"
)
SEQUENTIAL_CLOSURE_EXECUTION_SEMANTICS = (
    "START_FROM_REGISTERED_Q_AND_CLOSE_EACH_DISJOINT_SUPPORT_UNTIL_"
    "ITS_CONTACT_EVENT_WITHIN_THE_CERTIFIED_FIRST_CONTACT_SET"
)
PARAMETER_LAYOUT_PREFIX = (
    "assembly_axis_yaw_unit",
    "axial_target_unit",
    "lateral_task_x_unit",
    "lateral_task_y_unit",
)
CLOSURE_PARAMETER_DOMAIN_ID = (
    "SINGLE_EXCLUSIVE_CLOSURE_JOINT_FULL_OPEN_ENDPOINT_"
    "PLUS_NONCLOSURE_PRESHAPE_FULL_URDF_LIMITS_PLUS_PER_PAD_"
    "FULL_PATH_LIPSCHITZ_SWEPT_TASK_AABB_OVERLAP_INTERSECTION_"
    "PLUS_HALF_OPEN_YAW_AND_CANONICAL_DEGENERATE_AXES_V3"
)
CLOSURE_SUFFIX_DOMINANCE_ARGUMENT = (
    "EVERY_PARTIALLY_CLOSED_START_IS_A_SUFFIX_OF_THE_SAME_FULL_OPEN_PATH_"
    "AND_CANNOT_ADD_A_NEW_VALID_FIRST_CONTACT"
)
WITNESS_RULE = "SYMMETRIC_DEGREE2_INTERIOR_THREE_PER_VERIFIED_PAD_TRIANGLE"
INTERVAL_RULE = (
    "PER_WITNESS_DOWNSTREAM_REACH_LIPSCHITZ_FREE_PRUNING_PLUS_"
    "MP_DIRECTED_INTERVAL_COMPLETE_FACE_ROOT_ISOLATION"
)
DISTANCE_BVH_RULE = "LONGEST_TRIANGLE_CENTROID_EXTENT_STABLE_MEDIAN"
WHOLE_PATH_SPHERE_SCREEN_RULE = (
    "FULL_PAD_SINGLE_TRIANGLE_LEAF_AABB_HIERARCHY_X_8_DYADIC_PATH_"
    "SEGMENTS_X_PERSISTENT_DUAL_BVH_PAD_OBJECT_NODE_PAIRS_X_"
    "VECTORIZED_15_AXIS_OBB_AABB_SAT_X_CHEAP_CANDIDATE_PASS_"
    "BEFORE_SURVIVOR_ONLY_NARROWPHASE_X_LEAF_OBB_TRIANGLE_SAT_"
    "BEFORE_BATCHED_17_AXIS_MOVING_TRIANGLE_TRIANGLE_SAT_X_"
    "AMBIGUOUS_LEAF_ONLY_TWO_LEVEL_TIME_REFINEMENT_X_BOUNDED_"
    "FAIL_CLOSED_NARROWPHASE_WORK_X_LAZY_AFFINE_MOTION_BOUND_X_"
    "ROOT_ORDERED_STAGED_PAD_EARLY_CANDIDATE_REJECTION_X_ALLOWED_"
    "OBJECT_SEMANTIC_AND_INTERVAL_PAD_APPROACH_NECESSARY_CONTACT_"
    "CULL_X_OBJECT_WINDING_INVARIANT_X_DISTINCT_NO_VALID_CONTACT_TRUTH_V9"
)
CLAIM_LIMITATIONS = (
    "NOT_FULL_PAD_TRIANGLE_TRIANGLE_CONTINUOUS_COLLISION_DETECTION",
    "NOT_COMPLETE_HAND_TRAJECTORY_CLEARANCE_GATE",
    "FINITE_INTERIOR_WITNESSES_NOT_CONTINUOUS_PAD_SURFACE",
    "TANGENTIAL_CONTACTS_FAIL_CLOSED_UNRESOLVED",
    "TRIANGLE_EDGE_VERTEX_CONTACTS_FAIL_CLOSED_UNRESOLVED",
    "OVERLAPPING_ROOT_INTERVALS_ARE_POSSIBLE_EARLIEST_ALTERNATIVES_"
    "NOT_EQUAL_OR_COFIRST_ROOT_PROOFS",
    "CANDIDATE_USES_GUI_ONLY_NON_EVIDENTIARY_ROOT_REPRESENTATIVES",
    "MOTION_ORIENTED_TWO_SIDED_NORMAL_REQUIRES_EXTERNAL_START_AND_"
    "COMPLETE_COLLISION_GATE_FOR_SOLID_ENTRY_CLAIMS",
    "WITNESS_AND_MESH_CONVERGENCE_REQUIRED",
    "COMPLETE_COLLISION_STATIC_GATE_REQUIRED_BEFORE_DYNAMIC_USE",
)
MODEL_CONTRACT_DIGEST_METHOD_ID = (
    "CARTS_RAY_CLOSURE_CANONICAL_JSON_MODEL_CONTRACT_SHA256_V1"
)
MODEL_BINDING_COMPLETE_STATUS = "COMPLETE_CANONICAL_MODEL_CONTRACT_SHA256"
MODEL_BINDING_UNBOUND_STATUS = (
    "UNBOUND_SYNTHETIC_DIRECT_AUDIT_NOT_FORMAL_EVIDENCE"
)
_UNBOUND_MODEL_EVIDENCE = "UNBOUND_SYNTHETIC"
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")

_DISTANCE_BVH_LEAF_CAPACITY = 8
_WHOLE_PATH_SPHERE_SEGMENT_COUNT = 8
_PAD_SURFACE_SPHERE_HIERARCHY_MAXIMUM_DEPTH = 3
_PAD_AABB_FRONTIER_BATCH_PER_COVERAGE = 8
_PAD_AABB_MAXIMUM_TEMPORAL_REFINEMENT_DEPTH = 2
_PAD_AABB_MAXIMUM_MOVING_TRIANGLE_PAIR_TESTS_PER_COVERAGE = 4096
_EXACT_FACE_PAIR_PACKET_SIZE = 16384
_EXACT_FACE_PAIR_WORKER_COUNT = 4
_EXACT_FACE_PAIR_MAXIMUM_IN_FLIGHT = 8
_EXACT_PLANE_ROOT_WORKER_COUNT = 4
_EXACT_PLANE_ROOT_MAXIMUM_IN_FLIGHT = 8
_EXACT_PLANE_ROOT_TEMPORAL_DEFER_THRESHOLD = 256
_SWEPT_FACE_INITIAL_WITNESS_STAGE_SIZE = 1
_SWEPT_FACE_MAXIMUM_WITNESS_STAGE_SIZE = max(
    1, _EXACT_PLANE_ROOT_TEMPORAL_DEFER_THRESHOLD // 4
)
_BARYCENTRIC_WITNESSES = np.asarray(
    (
        (2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0),
        (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0),
        (1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0),
    ),
    dtype=np.float64,
)
_BARYCENTRIC_WITNESSES.setflags(write=False)


class RayClosureError(ValueError):
    """Raised when the mechanical or numerical closure contract is invalid."""


class _SubdivisionBudgetExhausted(RuntimeError):
    pass


def _float64_hex(value: float) -> str:
    """Return one exact, platform-independent binary64 text token."""

    return float(value).hex()


def _float64_array_hex(value: object) -> list[object]:
    """Return a nested JSON-compatible exact binary64 representation."""

    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        return [_float64_hex(float(array))]
    if array.ndim == 1:
        return [_float64_hex(float(item)) for item in array]
    return [_float64_array_hex(row) for row in array]


def _canonical_json(document: Mapping[str, object]) -> str:
    return json.dumps(
        document,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _pad_runtime_geometry_sha256(pad: VerifiedPad) -> str:
    """Bind the in-memory arrays actually consumed by the certifier."""

    digest = hashlib.sha256()
    digest.update(b"CARTS_VERIFIED_PAD_RUNTIME_TRIANGLE_MESH_SI_V1\0")
    for value, dtype in (
        (pad.points_local_m, np.dtype("<f8")),
        (pad.faces, np.dtype("<i8")),
    ):
        array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
        shape = np.asarray(array.shape, dtype="<i8")
        digest.update(np.asarray((array.ndim,), dtype="<i8").tobytes())
        digest.update(shape.tobytes(order="C"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _gamma(operation_count: int) -> float:
    epsilon = np.finfo(np.float64).eps
    product = float(operation_count) * epsilon
    if operation_count <= 0 or product >= 1.0:
        raise ValueError("operation_count cannot produce a finite gamma_n")
    return product / (1.0 - product)


_DOT_ERROR = _gamma(128)
_FK_ERROR = _gamma(1024)
_AABB_ERROR = _gamma(64)
_BARYCENTRIC_ERROR = _gamma(128)
_TIME_ERROR = _gamma(256)
_BRENT_ROOT_RELATIVE_TOLERANCE = 4.0 * np.finfo(np.float64).eps
_BRENT_ROOT_ABSOLUTE_TOLERANCE = np.finfo(np.float64).tiny
_BRENT_ROOT_MAXIMUM_ITERATIONS = 2 * (np.finfo(np.float64).nmant + 1)


def _immutable_array(value: object, *, shape_tail: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    tail_start = array.ndim - len(shape_tail)
    if array.ndim < len(shape_tail) or tuple(array.shape[tail_start:]) != shape_tail:
        raise RayClosureError(f"{name} must end in shape {shape_tail}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise RayClosureError(f"{name} must contain finite values")
    result = np.frombuffer(
        np.ascontiguousarray(array).tobytes(order="C"), dtype=np.float64
    ).reshape(array.shape)
    result.setflags(write=False)
    return result


def _readonly_int64(value: object) -> np.ndarray:
    """Return an immutable contiguous integer array."""

    array = np.ascontiguousarray(np.asarray(value, dtype=np.int64))
    array.setflags(write=False)
    return array


def _unit_vector(value: Sequence[float], label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise RayClosureError(f"{label} must be one finite three-vector")
    length = float(np.linalg.norm(vector))
    if length == 0.0 or not math.isfinite(length):
        raise RayClosureError(f"{label} must be non-zero")
    return vector / length


def _recover_closed_unit_coordinate(
    value: float,
    lower: float,
    upper: float,
    *,
    absolute_error: float,
    label: str,
) -> float:
    span = upper - lower
    if span == 0.0:
        if abs(value - lower) > absolute_error:
            raise RayClosureError(f"{label} is outside its zero-width geometric range")
        return 0.0
    unit = (value - lower) / span
    unit_error = absolute_error / abs(span) + _TIME_ERROR * max(1.0, abs(unit))
    if unit < -unit_error or unit > 1.0 + unit_error:
        raise RayClosureError(f"{label} is outside its geometry-derived range")
    return min(1.0, max(0.0, unit))


def _exact_dyadic_plane_key_fraction_reference(
    triangle_m: Sequence[Sequence[float]],
) -> tuple[int, int, int, int]:
    """Slow Fraction reference for the exact integer implementation."""

    triangle = np.asarray(triangle_m, dtype=np.float64)
    if triangle.shape != (3, 3) or not np.all(np.isfinite(triangle)):
        raise RayClosureError("plane key requires one finite triangle")
    points = tuple(
        tuple(Fraction.from_float(float(value)) for value in row)
        for row in triangle
    )
    first_edge = tuple(
        points[1][index] - points[0][index] for index in range(3)
    )
    second_edge = tuple(
        points[2][index] - points[0][index] for index in range(3)
    )
    normal = (
        first_edge[1] * second_edge[2]
        - first_edge[2] * second_edge[1],
        first_edge[2] * second_edge[0]
        - first_edge[0] * second_edge[2],
        first_edge[0] * second_edge[1]
        - first_edge[1] * second_edge[0],
    )
    if all(value == 0 for value in normal):
        raise RayClosureError("plane key triangle is exactly degenerate")
    offset = sum(
        normal[index] * points[0][index] for index in range(3)
    )
    rows = (*normal, -offset)
    common_denominator = 1
    for value in rows:
        common_denominator = math.lcm(
            common_denominator, value.denominator
        )
    integers = [
        value.numerator * (common_denominator // value.denominator)
        for value in rows
    ]
    common_divisor = 0
    for value in integers:
        common_divisor = math.gcd(common_divisor, abs(value))
    integers = [value // common_divisor for value in integers]
    leading = next(value for value in integers if value != 0)
    if leading < 0:
        integers = [-value for value in integers]
    return tuple(integers)  # type: ignore[return-value]


def _exact_dyadic_plane_key(
    triangle_m: Sequence[Sequence[float]],
) -> tuple[int, int, int, int]:
    """Return an exact unoriented plane key using dyadic integer arithmetic.

    Every finite binary64 coordinate has a power-of-two denominator.  Scaling
    all nine coordinates to their common largest denominator makes the edge
    cross product and plane offset exact integer operations.  A final gcd and
    sign normalization produces the same primitive key as the Fraction
    reference without constructing temporary rational objects.
    """

    triangle = np.asarray(triangle_m, dtype=np.float64)
    if triangle.shape != (3, 3) or not np.all(np.isfinite(triangle)):
        raise RayClosureError("plane key requires one finite triangle")
    ratios = tuple(
        tuple(float(value).as_integer_ratio() for value in row)
        for row in triangle
    )
    common_denominator = max(
        denominator
        for row in ratios
        for _numerator, denominator in row
    )
    integer_points = tuple(
        tuple(
            numerator * (common_denominator // denominator)
            for numerator, denominator in row
        )
        for row in ratios
    )
    first_edge = tuple(
        integer_points[1][index] - integer_points[0][index]
        for index in range(3)
    )
    second_edge = tuple(
        integer_points[2][index] - integer_points[0][index]
        for index in range(3)
    )
    normal = (
        first_edge[1] * second_edge[2]
        - first_edge[2] * second_edge[1],
        first_edge[2] * second_edge[0]
        - first_edge[0] * second_edge[2],
        first_edge[0] * second_edge[1]
        - first_edge[1] * second_edge[0],
    )
    if all(value == 0 for value in normal):
        raise RayClosureError("plane key triangle is exactly degenerate")
    coefficients = [
        normal[index] * common_denominator for index in range(3)
    ]
    coefficients.append(
        -sum(
            normal[index] * integer_points[0][index]
            for index in range(3)
        )
    )
    common_divisor = 0
    for value in coefficients:
        common_divisor = math.gcd(common_divisor, abs(value))
    coefficients = [value // common_divisor for value in coefficients]
    leading = next(value for value in coefficients if value != 0)
    if leading < 0:
        coefficients = [-value for value in coefficients]
    return tuple(coefficients)  # type: ignore[return-value]


def _exact_dyadic_oriented_plane_coefficient_key(
    triangle_m: Sequence[Sequence[float]],
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Return unnormalised exact oriented plane coefficients.

    Unlike :func:`_exact_dyadic_plane_key`, this key deliberately preserves
    winding and scale.  Two faces share it only when their binary64 vertices
    produce exactly the same rational plane predicate coefficients.
    """

    triangle = np.asarray(triangle_m, dtype=np.float64)
    if triangle.shape != (3, 3) or not np.all(np.isfinite(triangle)):
        raise RayClosureError(
            "oriented plane key requires one finite triangle"
        )
    points = tuple(
        tuple(Fraction.from_float(float(value)) for value in row)
        for row in triangle
    )
    first_edge = tuple(
        points[1][index] - points[0][index] for index in range(3)
    )
    second_edge = tuple(
        points[2][index] - points[0][index] for index in range(3)
    )
    normal = (
        first_edge[1] * second_edge[2]
        - first_edge[2] * second_edge[1],
        first_edge[2] * second_edge[0]
        - first_edge[0] * second_edge[2],
        first_edge[0] * second_edge[1]
        - first_edge[1] * second_edge[0],
    )
    if all(value == 0 for value in normal):
        raise RayClosureError("oriented plane key triangle is degenerate")
    offset = -sum(
        normal[index] * points[0][index] for index in range(3)
    )
    return tuple(
        (value.numerator, value.denominator)
        for value in (*normal, offset)
    )  # type: ignore[return-value]


@dataclass(frozen=True)
class PreRegisteredTaskFrame:
    """Explicit transverse direction; no PCA or object-symmetry inference."""

    transverse_axis_object: tuple[float, float, float]
    source: str

    def __post_init__(self) -> None:
        vector = _unit_vector(self.transverse_axis_object, "task transverse axis")
        if not str(self.source):
            raise RayClosureError("task frame source must be non-empty")
        object.__setattr__(
            self, "transverse_axis_object", tuple(float(item) for item in vector)
        )

    def basis(self, object_model: ObjectGraspModel) -> np.ndarray:
        axis = np.asarray(object_model.assembly_axis, dtype=np.float64)
        supplied = np.asarray(self.transverse_axis_object, dtype=np.float64)
        projected = supplied - float(supplied @ axis) * axis
        projection_norm = float(np.linalg.norm(projected))
        error_bound = _DOT_ERROR * (
            float(np.linalg.norm(supplied)) + abs(float(supplied @ axis))
        )
        if projection_norm <= error_bound:
            raise RayClosureError(
                "pre-registered task transverse axis is numerically parallel to assembly axis"
            )
        task_x = projected / projection_norm
        task_y = np.cross(axis, task_x)
        task_y /= np.linalg.norm(task_y)
        task_x = np.cross(task_y, axis)
        basis = np.column_stack((task_x, task_y, axis))
        basis.setflags(write=False)
        return basis


def _joint_model_manifest(hand_model: ThreeFingerHandModel) -> list[object]:
    rows: list[object] = []
    for mapping_key in sorted(hand_model.joints):
        joint = hand_model.joints[mapping_key]
        limit = (
            None
            if joint.limit is None
            else {
                "lower": _float64_hex(joint.limit.lower),
                "upper": _float64_hex(joint.limit.upper),
                "effort": (
                    None
                    if joint.limit.effort is None
                    else _float64_hex(joint.limit.effort)
                ),
                "velocity": (
                    None
                    if joint.limit.velocity is None
                    else _float64_hex(joint.limit.velocity)
                ),
            }
        )
        mimic = (
            None
            if joint.mimic is None
            else {
                "source_joint": joint.mimic.source_joint,
                "multiplier": _float64_hex(joint.mimic.multiplier),
                "offset": _float64_hex(joint.mimic.offset),
            }
        )
        rows.append(
            {
                "mapping_key": mapping_key,
                "name": joint.name,
                "joint_type": joint.joint_type,
                "parent_link": joint.parent_link,
                "child_link": joint.child_link,
                "origin_xyz_m": _float64_array_hex(joint.origin_xyz_m),
                "origin_rpy_rad": _float64_array_hex(joint.origin_rpy_rad),
                "axis": _float64_array_hex(joint.axis),
                "limit": limit,
                "mimic": mimic,
            }
        )
    return rows


def _hand_pad_model_manifest(hand_model: ThreeFingerHandModel) -> list[object]:
    rows: list[object] = []
    for name in sorted(hand_model.pads):
        pad = hand_model.pads[name]
        rows.append(
            {
                "mapping_key": name,
                "name": pad.name,
                "finger_name": pad.finger_name,
                "link_name": pad.link_name,
                "origin_xyz_m": _float64_array_hex(pad.origin_xyz_m),
                "origin_rpy_rad": _float64_array_hex(pad.origin_rpy_rad),
                "geometry": {
                    "kind": pad.geometry.kind,
                    "dimensions_m": _float64_array_hex(
                        pad.geometry.dimensions_m
                    ),
                    "mesh_uri": pad.geometry.mesh_uri,
                    "mesh_scale": _float64_array_hex(
                        pad.geometry.mesh_scale
                    ),
                },
                "contact_normal_pad": (
                    None
                    if pad.contact_normal_pad is None
                    else _float64_array_hex(pad.contact_normal_pad)
                ),
                "normal_force_capacity_n": (
                    None
                    if pad.normal_force_capacity_n is None
                    else _float64_hex(pad.normal_force_capacity_n)
                ),
            }
        )
    return rows


def _verified_pad_model_manifest(
    pads: Sequence[VerifiedPad],
) -> list[object]:
    rows: list[object] = []
    for pad in pads:
        rows.append(
            {
                "name": pad.name,
                "finger_name": pad.finger_name,
                "link_name": pad.link_name,
                "origin_xyz_m": _float64_array_hex(pad.origin_xyz_m),
                "origin_rpy_rad": _float64_array_hex(pad.origin_rpy_rad),
                "coordinate_frame": pad.coordinate_frame,
                "unit": pad.unit,
                "normal_force_capacity_n": _float64_hex(
                    pad.normal_force_capacity_n
                ),
                "source_mesh_repository_relative_path": (
                    pad.mesh.repository_relative_path
                ),
                "source_mesh_sha256": pad.mesh.sha256,
                "source_mesh_byte_count": int(pad.mesh.byte_count),
                "runtime_geometry_sha256": (
                    _pad_runtime_geometry_sha256(pad)
                ),
                "vertex_count": pad.vertex_count,
                "triangle_count": pad.triangle_count,
            }
        )
    return rows


def _hand_model_manifest(hand_model: ThreeFingerHandModel) -> dict[str, object]:
    independent_limits = hand_model.independent_joint_limits
    return {
        "base_link": hand_model.base_link,
        "joint_order": list(hand_model.joint_order),
        "independent_joint_names": list(
            hand_model.independent_joint_names
        ),
        "joints_by_mapping_key": _joint_model_manifest(hand_model),
        "independent_affine_limits": [
            {
                "joint_name": name,
                "lower": _float64_hex(independent_limits[name].lower),
                "upper": _float64_hex(independent_limits[name].upper),
                "effort": (
                    None
                    if independent_limits[name].effort is None
                    else _float64_hex(independent_limits[name].effort)
                ),
                "velocity": (
                    None
                    if independent_limits[name].velocity is None
                    else _float64_hex(independent_limits[name].velocity)
                ),
            }
            for name in hand_model.independent_joint_names
        ],
        "finger_chains": [
            {
                "finger_name": name,
                "joint_names": list(hand_model.fingers[name].joint_names),
                "terminal_link": hand_model.fingers[name].terminal_link,
                "pad_name": hand_model.fingers[name].pad_name,
            }
            for name in sorted(hand_model.fingers)
        ],
        "pad_mapping": _hand_pad_model_manifest(hand_model),
    }


def _model_contract_document(
    *,
    object_model: ObjectGraspModel,
    hand_model: ThreeFingerHandModel,
    pads: Sequence[VerifiedPad],
    task_frame: PreRegisteredTaskFrame,
    task_basis_object: np.ndarray,
    closing_directions_unit: np.ndarray,
    closing_directions_physical: np.ndarray,
    independent_actuation_supports: Sequence[Sequence[str]],
    parameter_layout: Sequence[str],
    object_contact_normal_policy: str,
    pad_surface_normal_policy: str,
    maximum_subdivision_intervals: int,
    interval_options: IntervalArithmeticOptions,
) -> dict[str, object]:
    return {
        "schema": MODEL_CONTRACT_DIGEST_METHOD_ID,
        "object": {
            "geometry_sha256": object_model.geometry_sha256,
            "assembly_axis": _float64_array_hex(
                object_model.assembly_axis
            ),
            "assembly_axis_origin_m": _float64_array_hex(
                object_model.assembly_axis_origin_m
            ),
        },
        "task_frame": {
            "source": task_frame.source,
            "pre_registered_transverse_axis_object": (
                _float64_array_hex(task_frame.transverse_axis_object)
            ),
            "basis_object": _float64_array_hex(task_basis_object),
        },
        "hand": _hand_model_manifest(hand_model),
        "verified_pads": _verified_pad_model_manifest(pads),
        "closure": {
            "closing_directions_unit": _float64_array_hex(
                closing_directions_unit
            ),
            "closing_directions_physical": _float64_array_hex(
                closing_directions_physical
            ),
            "independent_actuation_supports": [
                list(row) for row in independent_actuation_supports
            ],
            "parameter_layout": list(parameter_layout),
        },
        "ray_closure": {
            "method_id": METHOD_ID,
            "closure_parameter_domain_id": CLOSURE_PARAMETER_DOMAIN_ID,
            "closure_focus_method": CLOSURE_FOCUS_METHOD,
            "feature_root_policy": FEATURE_ROOT_POLICY,
            "possible_first_contact_set_method_id": (
                POSSIBLE_FIRST_CONTACT_SET_METHOD_ID
            ),
            "possible_earliest_ordering_policy": (
                POSSIBLE_EARLIEST_ORDERING_POLICY
            ),
            "representative_proposal_role": CANDIDATE_REPRESENTATIVE_ROLE,
            "display_approximation_role": DISPLAY_APPROXIMATION_ROLE,
            "ray_evaluation_policy": RAY_EVALUATION_POLICY,
            "witness_rule": WITNESS_RULE,
            "interval_rule": INTERVAL_RULE,
            "whole_path_sphere_screen_rule": (
                WHOLE_PATH_SPHERE_SCREEN_RULE
            ),
            "whole_path_sphere_segment_count": (
                _WHOLE_PATH_SPHERE_SEGMENT_COUNT
            ),
            "exact_initial_ordered_time_segment_count": (
                _WHOLE_PATH_SPHERE_SEGMENT_COUNT
            ),
            "exact_initial_ordered_time_segment_policy": (
                "EARLIEST_TO_LATEST_WITH_SHARED_BOUNDARY_POSITIONS_AND_"
                "ADJACENT_SEGMENT_COFIRST_BOUNDARY_MERGE"
            ),
            "pad_surface_sphere_hierarchy_maximum_depth": (
                _PAD_SURFACE_SPHERE_HIERARCHY_MAXIMUM_DEPTH
            ),
            "pad_surface_aabb_hierarchy_leaf_triangle_count": 1,
            "pad_surface_aabb_frontier_batch_per_coverage": (
                _PAD_AABB_FRONTIER_BATCH_PER_COVERAGE
            ),
            "pad_surface_aabb_maximum_temporal_refinement_depth": (
                _PAD_AABB_MAXIMUM_TEMPORAL_REFINEMENT_DEPTH
            ),
            "pad_surface_aabb_maximum_moving_triangle_pair_tests_per_coverage": (
                _PAD_AABB_MAXIMUM_MOVING_TRIANGLE_PAIR_TESTS_PER_COVERAGE
            ),
            "exact_face_pair_packet_size": _EXACT_FACE_PAIR_PACKET_SIZE,
            "exact_face_pair_worker_count": _EXACT_FACE_PAIR_WORKER_COUNT,
            "exact_face_pair_maximum_in_flight": (
                _EXACT_FACE_PAIR_MAXIMUM_IN_FLIGHT
            ),
            "exact_plane_root_worker_count": (
                _EXACT_PLANE_ROOT_WORKER_COUNT
            ),
            "exact_plane_root_maximum_in_flight": (
                _EXACT_PLANE_ROOT_MAXIMUM_IN_FLIGHT
            ),
            "exact_plane_root_temporal_defer_threshold": (
                _EXACT_PLANE_ROOT_TEMPORAL_DEFER_THRESHOLD
            ),
            "exact_plane_root_temporal_defer_threshold_role": (
                "COMPUTE_SCHEDULING_ONLY_NOT_PHYSICAL_OR_ACCEPTANCE_"
                "THRESHOLD"
            ),
            "swept_face_witness_stage_policy": (
                "ORIGINAL_WITNESS_ORDER_X_GEOMETRICALLY_GROWING_STAGES_"
                "FROM_ONE_TO_EXACT_ROOT_THRESHOLD_DIVIDED_BY_FOUR_X_"
                "FULL_CONSUMPTION_PRESERVES_ORIGINAL_FACE_ORDER"
            ),
            "staged_potential_root_defer_policy": (
                "POTENTIAL_UNCACHED_EXACT_PLANE_GROUP_COUNT_CROSSES_"
                "EXISTING_COMPUTE_ONLY_THRESHOLD_X_STOP_PARENT_"
                "MATERIALIZATION_X_TEMPORAL_CHILDREN_RECOMPUTE_COMPLETE_"
                "GEOMETRY_WITHOUT_PARTIAL_PARENT_FRONTIER"
            ),
            "exact_plane_root_execution_policy": (
                "BATCH_CERTIFIED_ENDPOINT_AND_DERIVATIVE_GATE_THEN_"
                "BOUNDED_THREAD_LOCAL_BACKENDS_WITH_ORIGINAL_ORDER_COMMIT"
            ),
            "pre_root_spatial_enclosure_policy": (
                "OUTWARD_MONOTONE_ROOT_PHASE_BOUND_THEN_ENDPOINT_PLUS_"
                "WHOLE_PATH_VELOCITY_POSITION_BOX_INTERSECTED_WITH_DIRECT_"
                "ENDPOINT_EDGE_FORM_VELOCITY_INTEGRATION_AND_SECOND_ORDER_"
                "ROOT_LOCAL_EDGE_CHORD_CURVATURE_BOUND_AND_STRICT_"
                "TRIANGLE_EDGE_REJECTION_BEFORE_EXACT_ROOT"
            ),
            "parent_temporal_classification_inheritance_policy": (
                "PARENT_CERTIFIED_FREE_PAIR_PRUNES_BOTH_CHILDREN_X_"
                "UNIQUE_ROOT_DISJOINT_CHILD_PRUNES_X_ROOT_INTERVAL_FULLY_"
                "CONTAINED_IN_CHILD_REUSES_CERTIFICATE_X_BOUNDARY_OVERLAP_"
                "OR_UNRESOLVED_RECOMPUTES_X_ORIGINAL_PAIR_ORDER_PRESERVED"
            ),
            "descendant_geometry_policy": (
                "ROOT_SEARCH_SEGMENT_USES_FULL_WITNESS_GEOMETRY_ONCE_X_"
                "DESCENDANTS_USE_CONSERVATIVE_PARENT_WITNESS_FACE_FRONTIER_"
                "THEN_EXISTING_TIGHT_INTERVAL_AND_CHORD_TUBE_FILTER"
            ),
            "pre_nearest_witness_aabb_policy": (
                "PER_SEGMENT_CLOSED_WITNESS_ENVELOPE_AABB_X_EMPTY_REAL_FACE_"
                "AABB_OVERLAP_CERTIFIES_NOT_POSSIBLE_X_EXACT_NEAREST_ONLY_"
                "FOR_SURVIVORS_X_ORIGINAL_HIERARCHY_FALLBACK_FOR_EXACT_FREE_"
                "MARGIN"
            ),
            "compiled_point_plane_binding_cache_capacity_per_worker": (
                COMPILED_POINT_PLANE_BINDING_CACHE_CAPACITY
            ),
            "compiled_point_plane_binding_reuse_policy": (
                "THREAD_LOCAL_FIXED_CAPACITY_PATH_IDENTITY_LRU_WITH_"
                "IN_PLACE_EXACT_TRIANGLE_PLANE_REBIND"
            ),
            "compiled_point_plane_link_plan_reuse_policy": (
                "ONE_IMMUTABLE_URDF_CHAIN_PAYLOAD_PER_LINK_PER_WORKER"
            ),
            "exact_query_aabb_bvh_leaf_capacity": (
                _DISTANCE_BVH_LEAF_CAPACITY
            ),
            "exact_query_object_traversal": (
                "DUAL_AABB_HIERARCHY_STREAMING_LEAF_PACKETS"
            ),
            "swept_face_candidate_count_role": (
                "CERTIFIED_INTERVAL_AND_CHORD_TUBE_INTERSECTION_AABB_"
                "FACE_COUNT"
            ),
            "pad_surface_aabb_motion_bound": (
                "NONNEGATIVE_RADIUS_SLOPE_TIMES_NODE_MAXIMUM_VERTEX_"
                "RADIUS_PLUS_NONNEGATIVE_TRANSLATION_INTERCEPT"
            ),
            "candidate_pad_screen_order": (
                "ASCENDING_GLOBAL_OBJECT_AABB_ROOT_OVERLAP_SEGMENT_"
                "COUNT_THEN_PAD_INDEX"
            ),
            "candidate_pad_root_overlap_role": (
                "ORDER_ONLY_NEVER_CERTIFIED_FREE_OR_REJECTION_EVIDENCE"
            ),
            "candidate_pad_skip_policy": (
                "SKIP_REMAINING_PADS_AFTER_ONE_PAD_CERTIFIED_FREE"
            ),
            "candidate_screen_cascade": (
                "ONE_CHEAP_PERSISTENT_DUAL_BVH_PASS_FOR_ALL_CANDIDATES_"
                "NO_DIRECTIONAL_OR_MOVING_TRIANGLE_SECOND_PASS"
            ),
            "pad_object_dual_bvh_split_policy": (
                "SPLIT_LARGER_WORLD_AABB_VOLUME_NONLEAF_OBJECT_ON_TIE"
            ),
            "pad_object_dual_bvh_leaf_policy": (
                "ONE_MOVING_PAD_TRIANGLE_X_AT_MOST_EIGHT_STATIC_OBJECT_"
                "TRIANGLES_BATCHED_OBB_STRICT_SAT_THEN_17_AXIS_MOVING_"
                "TRIANGLE_STRICT_SAT_THEN_BOUNDED_AMBIGUOUS_ONLY_"
                "DYADIC_TIME_REFINEMENT"
            ),
            "directional_contact_feasibility_policy": (
                "ALLOWED_OBJECT_SEMANTIC_X_INTERVAL_WITNESS_VELOCITY_"
                "X_INTERVAL_ROTATED_PAD_NORMAL_STRICT_POSITIVE_APPROACH_"
                "NECESSARY_CONDITION_X_OBJECT_FACE_WINDING_IGNORED_"
                "TANGENT_AND_INTERVAL_BOUNDARY_RETAINED"
            ),
            "object_contact_normal_policy": object_contact_normal_policy,
            "pad_surface_normal_policy": pad_surface_normal_policy,
            "maximum_subdivision_intervals": (
                maximum_subdivision_intervals
            ),
        },
        "interval_backend": {
            "method_id": INTERVAL_KINEMATICS_METHOD_ID,
            "decimal_precision": interval_options.decimal_precision,
            "maximum_root_bisection_iterations": (
                interval_options.maximum_root_bisection_iterations
            ),
            "multiphase_transform_cache_capacity": (
                MULTIPHASE_TRANSFORM_CACHE_CAPACITY
            ),
            "multiphase_point_cache_capacity": (
                MULTIPHASE_POINT_CACHE_CAPACITY
            ),
            "multiphase_pad_area_cache_capacity": (
                MULTIPHASE_PAD_AREA_CACHE_CAPACITY
            ),
            "nominal_root_seed_policy": NOMINAL_ROOT_SEED_POLICY,
            "nominal_root_seed_maximum_iterations": (
                NOMINAL_ROOT_SEED_MAXIMUM_ITERATIONS
            ),
            "nominal_root_seed_may_certify_or_reject": False,
            "nominal_root_seed_requires_exact_endpoint_reverification": True,
            "static_joint_origin_interval_transform_policy": (
                "PRECOMPILED_ONCE_PER_BACKEND_AND_JOINT_IDENTITY"
            ),
            "link_chain_and_mimic_affine_policy": (
                "PRECOMPILED_PER_EXACT_LINK_AND_INDEPENDENT_JOINT_INDEX"
            ),
            "per_root_object_plane_interval_policy": (
                "PRECOMPILED_ONCE_AND_BOUND_TO_EXACT_TRIANGLE_BINARY64_BYTES"
            ),
            "interval_rigid_transform_composition_policy": (
                "HOMOGENEOUS_RIGID_3X3_ROTATION_PLUS_TRANSLATION_ONLY"
            ),
            "interval_joint_step_policy": (
                "ORIGIN_COMPOSE_ONCE_THEN_DIRECT_REVOLUTE_OR_PRISMATIC_UPDATE"
            ),
            "cardinal_axis_rotation_policy": (
                "EXACT_POSITIVE_OR_NEGATIVE_X_Y_Z_ONLY_OTHERWISE_RODRIGUES"
            ),
        },
    }


@dataclass(frozen=True)
class PadClosureAudit:
    pad_name: str
    finger_name: str
    verified_triangle_count: int
    witness_count: int
    exact_fk_interval_evaluations: int
    leading_witness_evaluations: int
    first_hit_rays_cast: int
    finite_chord_feature_candidates: int
    nonlinear_feature_roots_solved: int
    nonlinear_root_fk_evaluations: int
    distance_bvh_node_visits: int
    distance_triangle_tests: int
    certified_free_interval_count: int
    certified_witness_path_clearance_lower_bound_m: float
    interval_point_motion_evaluations: int
    swept_face_candidate_count: int
    interval_pair_evaluation_count: int
    certified_contact_root_count: int
    unresolved_witness_face_pair_count: int
    cofirst_root_count: int
    competing_root_order_block_count: int
    acceptance_ray_call_count: int
    selected_triangle_index: int | None
    selected_witness_index: int | None
    selected_object_face_index: int | None
    selected_normalized_closure: float | None
    selected_closure_interval_width: float | None
    selected_spatial_error_bound_m: float | None
    selected_root_phase_lower: float | None
    selected_root_phase_upper: float | None
    selected_pad_approach_lower: float | None
    selected_path_local_free_side_approach_lower: float | None
    selected_object_source_winding_free_side_sign: int | None
    first_contact_classification: str
    possible_earliest_root_count: int = 0
    possible_first_contact_set_sha256: str | None = None
    selected_normalized_closure_role: str | None = None
    actual_plane_root_evaluation_count: int = 0
    batch_root_triangle_free_pair_count: int = 0
    batch_root_triangle_uncertain_pair_count: int = 0
    batch_plane_monotone_same_side_free_count: int = 0
    shared_plane_gate_root_count: int = 0
    parallel_plane_root_task_count: int = 0
    pre_root_spatial_enclosure_group_count: int = 0
    pre_root_spatial_free_pair_count: int = 0
    pre_root_spatial_fully_free_group_count: int = 0
    parent_certified_free_pair_prune_count: int = 0
    parent_certified_root_outside_pair_prune_count: int = 0
    parent_certified_root_pair_reuse_count: int = 0
    large_exact_batch_temporal_deferral_count: int = 0
    large_exact_batch_deferred_root_group_count: int = 0
    swept_face_witness_stage_count: int = 0
    swept_face_witness_materialized_count: int = 0
    staged_potential_root_temporal_deferral_count: int = 0
    staged_potential_root_group_count: int = 0
    staged_unmaterialized_witness_count: int = 0
    parent_frontier_geometry_bypass_count: int = 0
    pre_nearest_aabb_witness_test_count: int = 0
    pre_nearest_aabb_certified_free_witness_count: int = 0
    pre_nearest_aabb_exact_survivor_count: int = 0
    pre_nearest_aabb_fast_path_count: int = 0
    pre_nearest_aabb_fallback_count: int = 0
    root_interpolation_iteration_count: int = 0
    interval_newton_iteration_count: int = 0
    root_bisection_iteration_count: int = 0
    whole_path_sphere_screen_segment_count: int = 0
    whole_path_sphere_screen_query_count: int = 0
    whole_path_sphere_screen_bvh_node_visits: int = 0
    whole_path_sphere_screen_triangle_tests: int = 0
    whole_path_sphere_screen_obb_sat_certified_free_node_count: int = 0
    whole_path_sphere_screen_obb_sat_triangle_test_count: int = 0
    whole_path_sphere_screen_moving_triangle_sat_certified_free_pair_count: int = 0
    whole_path_sphere_screen_moving_triangle_sat_pair_test_count: int = 0
    whole_path_sphere_screen_temporal_refined_leaf_pair_count: int = 0
    whole_path_sphere_screen_temporal_refinement_transform_count: int = 0
    whole_path_sphere_screen_maximum_temporal_refinement_depth_reached: int = 0
    whole_path_sphere_screen_narrowphase_refinement_used: bool = False
    whole_path_sphere_screen_narrowphase_work_budget_exhausted: bool = False
    whole_path_sphere_screen_directional_contact_feasibility_used: bool = False
    whole_path_sphere_screen_directional_bvh_node_pair_test_count: int = 0
    whole_path_sphere_screen_directional_bvh_node_pair_rejected_count: int = 0
    whole_path_sphere_screen_directional_leaf_face_pair_test_count: int = 0
    whole_path_sphere_screen_directional_leaf_face_pair_rejected_count: int = 0
    whole_path_sphere_screen_directional_interval_witness_motion_evaluation_count: int = 0
    whole_path_sphere_screen_certified_no_valid_contact: bool = False
    whole_path_sphere_screen_certified_free: bool = False
    whole_path_sphere_screen_clearance_lower_bound_m: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "pad_name": self.pad_name,
            "finger_name": self.finger_name,
            "verified_triangle_count": self.verified_triangle_count,
            "witness_count": self.witness_count,
            "exact_fk_interval_evaluations": self.exact_fk_interval_evaluations,
            "leading_witness_evaluations": self.leading_witness_evaluations,
            "first_hit_rays_cast": self.first_hit_rays_cast,
            "finite_chord_feature_candidates": (
                self.finite_chord_feature_candidates
            ),
            "nonlinear_feature_roots_solved": (
                self.nonlinear_feature_roots_solved
            ),
            "nonlinear_root_fk_evaluations": (
                self.nonlinear_root_fk_evaluations
            ),
            "distance_bvh_node_visits": self.distance_bvh_node_visits,
            "distance_triangle_tests": self.distance_triangle_tests,
            "certified_free_interval_count": self.certified_free_interval_count,
            "certified_witness_path_clearance_lower_bound_m": (
                self.certified_witness_path_clearance_lower_bound_m
            ),
            "interval_point_motion_evaluations": (
                self.interval_point_motion_evaluations
            ),
            "swept_face_candidate_count": self.swept_face_candidate_count,
            "interval_pair_evaluation_count": (
                self.interval_pair_evaluation_count
            ),
            "actual_plane_root_evaluation_count": (
                self.actual_plane_root_evaluation_count
            ),
            "batch_root_triangle_free_pair_count": (
                self.batch_root_triangle_free_pair_count
            ),
            "batch_root_triangle_uncertain_pair_count": (
                self.batch_root_triangle_uncertain_pair_count
            ),
            "batch_plane_monotone_same_side_free_count": (
                self.batch_plane_monotone_same_side_free_count
            ),
            "shared_plane_gate_root_count": (
                self.shared_plane_gate_root_count
            ),
            "parallel_plane_root_task_count": (
                self.parallel_plane_root_task_count
            ),
            "pre_root_spatial_enclosure_group_count": (
                self.pre_root_spatial_enclosure_group_count
            ),
            "pre_root_spatial_free_pair_count": (
                self.pre_root_spatial_free_pair_count
            ),
            "pre_root_spatial_fully_free_group_count": (
                self.pre_root_spatial_fully_free_group_count
            ),
            "parent_certified_free_pair_prune_count": (
                self.parent_certified_free_pair_prune_count
            ),
            "parent_certified_root_outside_pair_prune_count": (
                self.parent_certified_root_outside_pair_prune_count
            ),
            "parent_certified_root_pair_reuse_count": (
                self.parent_certified_root_pair_reuse_count
            ),
            "large_exact_batch_temporal_deferral_count": (
                self.large_exact_batch_temporal_deferral_count
            ),
            "large_exact_batch_deferred_root_group_count": (
                self.large_exact_batch_deferred_root_group_count
            ),
            "swept_face_witness_stage_count": (
                self.swept_face_witness_stage_count
            ),
            "swept_face_witness_materialized_count": (
                self.swept_face_witness_materialized_count
            ),
            "staged_potential_root_temporal_deferral_count": (
                self.staged_potential_root_temporal_deferral_count
            ),
            "staged_potential_root_group_count": (
                self.staged_potential_root_group_count
            ),
            "staged_unmaterialized_witness_count": (
                self.staged_unmaterialized_witness_count
            ),
            "parent_frontier_geometry_bypass_count": (
                self.parent_frontier_geometry_bypass_count
            ),
            "pre_nearest_aabb_witness_test_count": (
                self.pre_nearest_aabb_witness_test_count
            ),
            "pre_nearest_aabb_certified_free_witness_count": (
                self.pre_nearest_aabb_certified_free_witness_count
            ),
            "pre_nearest_aabb_exact_survivor_count": (
                self.pre_nearest_aabb_exact_survivor_count
            ),
            "pre_nearest_aabb_fast_path_count": (
                self.pre_nearest_aabb_fast_path_count
            ),
            "pre_nearest_aabb_fallback_count": (
                self.pre_nearest_aabb_fallback_count
            ),
            "root_interpolation_iteration_count": (
                self.root_interpolation_iteration_count
            ),
            "interval_newton_iteration_count": (
                self.interval_newton_iteration_count
            ),
            "root_bisection_iteration_count": (
                self.root_bisection_iteration_count
            ),
            "whole_path_sphere_screen_segment_count": (
                self.whole_path_sphere_screen_segment_count
            ),
            "whole_path_sphere_screen_query_count": (
                self.whole_path_sphere_screen_query_count
            ),
            "whole_path_sphere_screen_bvh_node_visits": (
                self.whole_path_sphere_screen_bvh_node_visits
            ),
            "whole_path_sphere_screen_triangle_tests": (
                self.whole_path_sphere_screen_triangle_tests
            ),
            "whole_path_sphere_screen_obb_sat_certified_free_node_count": (
                self.whole_path_sphere_screen_obb_sat_certified_free_node_count
            ),
            "whole_path_sphere_screen_obb_sat_triangle_test_count": (
                self.whole_path_sphere_screen_obb_sat_triangle_test_count
            ),
            "whole_path_sphere_screen_moving_triangle_sat_certified_free_pair_count": (
                self.whole_path_sphere_screen_moving_triangle_sat_certified_free_pair_count
            ),
            "whole_path_sphere_screen_moving_triangle_sat_pair_test_count": (
                self.whole_path_sphere_screen_moving_triangle_sat_pair_test_count
            ),
            "whole_path_sphere_screen_temporal_refined_leaf_pair_count": (
                self.whole_path_sphere_screen_temporal_refined_leaf_pair_count
            ),
            "whole_path_sphere_screen_temporal_refinement_transform_count": (
                self.whole_path_sphere_screen_temporal_refinement_transform_count
            ),
            "whole_path_sphere_screen_maximum_temporal_refinement_depth_reached": (
                self.whole_path_sphere_screen_maximum_temporal_refinement_depth_reached
            ),
            "whole_path_sphere_screen_narrowphase_refinement_used": (
                self.whole_path_sphere_screen_narrowphase_refinement_used
            ),
            "whole_path_sphere_screen_narrowphase_work_budget_exhausted": (
                self.whole_path_sphere_screen_narrowphase_work_budget_exhausted
            ),
            "whole_path_sphere_screen_directional_contact_feasibility_used": (
                self.whole_path_sphere_screen_directional_contact_feasibility_used
            ),
            "whole_path_sphere_screen_directional_bvh_node_pair_test_count": (
                self.whole_path_sphere_screen_directional_bvh_node_pair_test_count
            ),
            "whole_path_sphere_screen_directional_bvh_node_pair_rejected_count": (
                self.whole_path_sphere_screen_directional_bvh_node_pair_rejected_count
            ),
            "whole_path_sphere_screen_directional_leaf_face_pair_test_count": (
                self.whole_path_sphere_screen_directional_leaf_face_pair_test_count
            ),
            "whole_path_sphere_screen_directional_leaf_face_pair_rejected_count": (
                self.whole_path_sphere_screen_directional_leaf_face_pair_rejected_count
            ),
            "whole_path_sphere_screen_directional_interval_witness_motion_evaluation_count": (
                self.whole_path_sphere_screen_directional_interval_witness_motion_evaluation_count
            ),
            "whole_path_sphere_screen_certified_no_valid_contact": (
                self.whole_path_sphere_screen_certified_no_valid_contact
            ),
            "whole_path_sphere_screen_certified_free": (
                self.whole_path_sphere_screen_certified_free
            ),
            "whole_path_sphere_screen_clearance_lower_bound_m": (
                self.whole_path_sphere_screen_clearance_lower_bound_m
            ),
            "certified_contact_root_count": (
                self.certified_contact_root_count
            ),
            "unresolved_witness_face_pair_count": (
                self.unresolved_witness_face_pair_count
            ),
            "cofirst_root_count": self.cofirst_root_count,
            "possible_earliest_root_count": (
                self.possible_earliest_root_count
            ),
            "possible_first_contact_set_sha256": (
                self.possible_first_contact_set_sha256
            ),
            "competing_root_order_block_count": (
                self.competing_root_order_block_count
            ),
            "acceptance_ray_call_count": self.acceptance_ray_call_count,
            "selected_triangle_index": self.selected_triangle_index,
            "selected_witness_index": self.selected_witness_index,
            "selected_object_face_index": self.selected_object_face_index,
            "selected_normalized_closure": self.selected_normalized_closure,
            "selected_normalized_closure_role": (
                self.selected_normalized_closure_role
            ),
            "selected_closure_interval_width": self.selected_closure_interval_width,
            "selected_spatial_error_bound_m": self.selected_spatial_error_bound_m,
            "selected_root_phase_lower": self.selected_root_phase_lower,
            "selected_root_phase_upper": self.selected_root_phase_upper,
            "selected_pad_approach_lower": self.selected_pad_approach_lower,
            "selected_path_local_free_side_approach_lower": (
                self.selected_path_local_free_side_approach_lower
            ),
            "selected_object_source_winding_free_side_sign": (
                self.selected_object_source_winding_free_side_sign
            ),
            "first_contact_classification": self.first_contact_classification,
        }


@dataclass(frozen=True)
class RayClosureAudit:
    method_id: str
    numerical_policy: str
    witness_rule: str
    interval_rule: str
    distance_bvh_rule: str
    ray_evaluation_policy: str
    feature_root_policy: str
    object_contact_normal_policy: str
    pad_surface_normal_policy: str
    parameter_layout: tuple[str, ...]
    pad_order: tuple[str, ...]
    full_verified_pad_mesh_used: bool
    pad_face_subset_input_allowed: bool
    independent_actuation_supports: tuple[tuple[str, ...], ...]
    closure_parameter_domain_id: str
    closure_suffix_dominance_argument: str
    preshape_joint_names: tuple[str, ...]
    closure_open_joint_positions_rad: tuple[float, ...]
    maximum_subdivision_intervals: int
    interval_arithmetic_method_id: str
    interval_decimal_precision: int
    maximum_root_bisection_iterations: int
    subdivision_intervals_used: int
    subdivision_budget_exhausted: bool
    internal_force_role: str
    trajectory_clearance_m: float
    trajectory_clearance_role: str
    task_frame_source: str
    closure_focus_method: str
    distance_bvh_node_count: int
    pad_audits: tuple[PadClosureAudit, ...]
    claim_limitations: tuple[str, ...]
    failure_reason: str | None
    model_binding_complete: bool = False
    model_binding_status: str = MODEL_BINDING_UNBOUND_STATUS
    object_geometry_sha256: str = _UNBOUND_MODEL_EVIDENCE
    model_contract_sha256: str = _UNBOUND_MODEL_EVIDENCE
    pad_geometry_sha256: tuple[str, ...] = ()
    pad_runtime_geometry_sha256: tuple[str, ...] = ()
    pad_link_names: tuple[str, ...] = ()
    closing_directions_physical: tuple[tuple[float, ...], ...] = ()
    model_contract_canonical_json: str = ""
    candidate_role: str = NO_CANDIDATE_ROLE
    candidate_exact_contact_endpoint_certified: bool = False
    display_approximation_role: str = DISPLAY_APPROXIMATION_ROLE
    possible_first_contact_set_sha256: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.candidate_role not in (
            NO_CANDIDATE_ROLE,
            CANDIDATE_REPRESENTATIVE_ROLE,
        ):
            raise RayClosureError("candidate role is not a recognised claim")
        if type(self.candidate_exact_contact_endpoint_certified) is not bool:
            raise RayClosureError(
                "candidate exact-contact certification flag must be a bool"
            )
        if self.candidate_exact_contact_endpoint_certified:
            raise RayClosureError(
                "V9 cannot certify a binary64 candidate as the exact implicit root"
            )
        if self.display_approximation_role != DISPLAY_APPROXIMATION_ROLE:
            raise RayClosureError(
                "root display approximation role must remain non-evidentiary"
            )
        for digest in self.possible_first_contact_set_sha256:
            if not isinstance(digest, str) or _HEX_SHA256.fullmatch(digest) is None:
                raise RayClosureError(
                    "possible-first-contact set digest must be lowercase SHA-256"
                )
        if type(self.model_binding_complete) is not bool:
            raise RayClosureError("model_binding_complete must be a bool")
        if not self.model_binding_complete:
            if (
                self.model_binding_status != MODEL_BINDING_UNBOUND_STATUS
                or self.object_geometry_sha256 != _UNBOUND_MODEL_EVIDENCE
                or self.model_contract_sha256 != _UNBOUND_MODEL_EVIDENCE
                or self.pad_geometry_sha256
                or self.pad_runtime_geometry_sha256
                or self.pad_link_names
                or self.closing_directions_physical
                or self.model_contract_canonical_json
            ):
                raise RayClosureError(
                    "unbound synthetic audit cannot carry partial model evidence"
                )
            return

        if self.model_binding_status != MODEL_BINDING_COMPLETE_STATUS:
            raise RayClosureError(
                "complete model binding requires the canonical status"
            )
        digest_rows = (
            ("object_geometry_sha256", self.object_geometry_sha256),
            ("model_contract_sha256", self.model_contract_sha256),
            *(
                (f"pad_geometry_sha256[{index}]", value)
                for index, value in enumerate(self.pad_geometry_sha256)
            ),
            *(
                (f"pad_runtime_geometry_sha256[{index}]", value)
                for index, value in enumerate(
                    self.pad_runtime_geometry_sha256
                )
            ),
        )
        for label, value in digest_rows:
            if not isinstance(value, str) or _HEX_SHA256.fullmatch(value) is None:
                raise RayClosureError(
                    f"{label} must be 64 lowercase hexadecimal digits"
                )
        evidence_count = len(self.pad_order)
        if (
            evidence_count != 3
            or len(set(self.pad_order)) != evidence_count
            or len(self.pad_geometry_sha256) != evidence_count
            or len(self.pad_runtime_geometry_sha256) != evidence_count
            or len(self.pad_link_names) != evidence_count
            or len(self.closing_directions_physical) != evidence_count
        ):
            raise RayClosureError(
                "bound PAD evidence must match the unique three-PAD order"
            )
        if any(not isinstance(name, str) or not name for name in self.pad_link_names):
            raise RayClosureError("bound PAD link names must be non-empty")
        direction_widths = {
            len(row) for row in self.closing_directions_physical
        }
        if len(direction_widths) != 1 or next(iter(direction_widths)) == 0:
            raise RayClosureError(
                "physical closing directions must have one non-empty width"
            )
        for row in self.closing_directions_physical:
            if (
                any(not math.isfinite(float(value)) for value in row)
                or not any(float(value) != 0.0 for value in row)
            ):
                raise RayClosureError(
                    "each physical closing direction must be finite and non-zero"
                )
        if not isinstance(self.model_contract_canonical_json, str):
            raise RayClosureError("model contract canonical JSON must be text")
        try:
            document = json.loads(self.model_contract_canonical_json)
        except (TypeError, ValueError) as error:
            raise RayClosureError(
                "model contract canonical JSON cannot be decoded"
            ) from error
        if not isinstance(document, Mapping):
            raise RayClosureError("model contract document must be a mapping")
        if _canonical_json(document) != self.model_contract_canonical_json:
            raise RayClosureError("model contract JSON is not canonical")
        expected_digest = hashlib.sha256(
            self.model_contract_canonical_json.encode("utf-8")
        ).hexdigest()
        if self.model_contract_sha256 != expected_digest:
            raise RayClosureError("model contract SHA-256 contradicts its manifest")
        try:
            pad_rows = document["verified_pads"]
            if not isinstance(pad_rows, list):
                raise TypeError("verified_pads is not a list")
            document_pad_order = tuple(row["name"] for row in pad_rows)
            document_pad_links = tuple(row["link_name"] for row in pad_rows)
            document_pad_hashes = tuple(
                row["source_mesh_sha256"] for row in pad_rows
            )
            document_runtime_hashes = tuple(
                row["runtime_geometry_sha256"] for row in pad_rows
            )
            document_directions = document["closure"][
                "closing_directions_physical"
            ]
            hand_joint_names = document["hand"][
                "independent_joint_names"
            ]
            if (
                document["schema"] != MODEL_CONTRACT_DIGEST_METHOD_ID
                or document["object"]["geometry_sha256"]
                != self.object_geometry_sha256
                or document_pad_order != self.pad_order
                or document_pad_links != self.pad_link_names
                or document_pad_hashes != self.pad_geometry_sha256
                or document_runtime_hashes
                != self.pad_runtime_geometry_sha256
                or document_directions
                != _float64_array_hex(self.closing_directions_physical)
                or len(hand_joint_names) != next(iter(direction_widths))
                or document["ray_closure"]["method_id"] != self.method_id
                or document["ray_closure"][
                    "object_contact_normal_policy"
                ]
                != self.object_contact_normal_policy
                or document["ray_closure"]["pad_surface_normal_policy"]
                != self.pad_surface_normal_policy
                or document["ray_closure"][
                    "maximum_subdivision_intervals"
                ]
                != self.maximum_subdivision_intervals
                or document["ray_closure"][
                    "possible_first_contact_set_method_id"
                ]
                != POSSIBLE_FIRST_CONTACT_SET_METHOD_ID
                or document["ray_closure"][
                    "possible_earliest_ordering_policy"
                ]
                != POSSIBLE_EARLIEST_ORDERING_POLICY
                or document["ray_closure"]["representative_proposal_role"]
                != CANDIDATE_REPRESENTATIVE_ROLE
                or document["ray_closure"]["display_approximation_role"]
                != DISPLAY_APPROXIMATION_ROLE
                or document["ray_closure"][
                    "whole_path_sphere_screen_rule"
                ]
                != WHOLE_PATH_SPHERE_SCREEN_RULE
                or document["ray_closure"][
                    "whole_path_sphere_segment_count"
                ]
                != _WHOLE_PATH_SPHERE_SEGMENT_COUNT
                or document["ray_closure"][
                    "pad_surface_sphere_hierarchy_maximum_depth"
                ]
                != _PAD_SURFACE_SPHERE_HIERARCHY_MAXIMUM_DEPTH
                or document["ray_closure"][
                    "pad_surface_aabb_maximum_temporal_refinement_depth"
                ]
                != _PAD_AABB_MAXIMUM_TEMPORAL_REFINEMENT_DEPTH
                or document["ray_closure"][
                    "pad_surface_aabb_maximum_moving_triangle_pair_tests_per_coverage"
                ]
                != _PAD_AABB_MAXIMUM_MOVING_TRIANGLE_PAIR_TESTS_PER_COVERAGE
                or document["interval_backend"]["method_id"]
                != self.interval_arithmetic_method_id
                or document["interval_backend"]["decimal_precision"]
                != self.interval_decimal_precision
                or document["interval_backend"][
                    "maximum_root_bisection_iterations"
                ]
                != self.maximum_root_bisection_iterations
                or document["task_frame"]["source"]
                != self.task_frame_source
            ):
                raise RayClosureError(
                    "model contract manifest contradicts audit evidence"
                )
        except (KeyError, TypeError) as error:
            raise RayClosureError(
                "model contract manifest is structurally incomplete"
            ) from error

    def as_dict(self) -> dict[str, object]:
        return {
            "method_id": self.method_id,
            "model_binding_complete": self.model_binding_complete,
            "model_binding_status": self.model_binding_status,
            "model_contract_digest_method_id": (
                MODEL_CONTRACT_DIGEST_METHOD_ID
            ),
            "object_geometry_sha256": self.object_geometry_sha256,
            "model_contract_sha256": self.model_contract_sha256,
            "pad_geometry_sha256": list(self.pad_geometry_sha256),
            "pad_runtime_geometry_sha256": list(
                self.pad_runtime_geometry_sha256
            ),
            "pad_link_names": list(self.pad_link_names),
            "closing_directions_physical": [
                list(row) for row in self.closing_directions_physical
            ],
            "model_contract_manifest": (
                json.loads(self.model_contract_canonical_json)
                if self.model_binding_complete
                else None
            ),
            "numerical_policy": self.numerical_policy,
            "witness_rule": self.witness_rule,
            "interval_rule": self.interval_rule,
            "distance_bvh_rule": self.distance_bvh_rule,
            "ray_evaluation_policy": self.ray_evaluation_policy,
            "feature_root_policy": self.feature_root_policy,
            "object_contact_normal_policy": (
                self.object_contact_normal_policy
            ),
            "pad_surface_normal_policy": self.pad_surface_normal_policy,
            "parameter_layout": list(self.parameter_layout),
            "pad_order": list(self.pad_order),
            "full_verified_pad_mesh_used": self.full_verified_pad_mesh_used,
            "pad_face_subset_input_allowed": self.pad_face_subset_input_allowed,
            "independent_actuation_supports": [
                list(row) for row in self.independent_actuation_supports
            ],
            "closure_parameter_domain_id": self.closure_parameter_domain_id,
            "closure_suffix_dominance_argument": (
                self.closure_suffix_dominance_argument
            ),
            "preshape_joint_names": list(self.preshape_joint_names),
            "closure_open_joint_positions_rad": list(
                self.closure_open_joint_positions_rad
            ),
            "maximum_subdivision_intervals": self.maximum_subdivision_intervals,
            "interval_arithmetic_method_id": (
                self.interval_arithmetic_method_id
            ),
            "interval_decimal_precision": self.interval_decimal_precision,
            "maximum_root_bisection_iterations": (
                self.maximum_root_bisection_iterations
            ),
            "subdivision_intervals_used": self.subdivision_intervals_used,
            "subdivision_budget_exhausted": self.subdivision_budget_exhausted,
            "internal_force_role": self.internal_force_role,
            "trajectory_clearance_m": self.trajectory_clearance_m,
            "trajectory_clearance_role": self.trajectory_clearance_role,
            "task_frame_source": self.task_frame_source,
            "closure_focus_method": self.closure_focus_method,
            "distance_bvh_node_count": self.distance_bvh_node_count,
            "pad_audits": [row.as_dict() for row in self.pad_audits],
            "claim_limitations": list(self.claim_limitations),
            "failure_reason": self.failure_reason,
            "candidate_role": self.candidate_role,
            "candidate_exact_contact_endpoint_certified": (
                self.candidate_exact_contact_endpoint_certified
            ),
            "display_approximation_role": self.display_approximation_role,
            "possible_first_contact_set_sha256": list(
                self.possible_first_contact_set_sha256
            ),
        }


@dataclass(frozen=True)
class DisplayOnlyGraspProposal:
    """A GUI/seed view that is deliberately not a formal GraspCandidate."""

    grasp_candidate: GraspCandidate
    role: str = CANDIDATE_REPRESENTATIVE_ROLE
    joint_and_contact_value_role: str = DISPLAY_APPROXIMATION_ROLE

    def __post_init__(self) -> None:
        if (
            not isinstance(self.grasp_candidate, GraspCandidate)
            or self.role != CANDIDATE_REPRESENTATIVE_ROLE
            or self.joint_and_contact_value_role
            != DISPLAY_APPROXIMATION_ROLE
        ):
            raise RayClosureError(
                "display-only grasp proposal provenance is invalid"
            )


@dataclass(frozen=True)
class RayClosureEvaluation:
    candidate: GraspCandidate | None
    audit: RayClosureAudit
    possible_first_contact_sets: tuple["PossibleFirstContactSet", ...] = ()
    display_only_proposal: DisplayOnlyGraspProposal | None = None
    sequential_closure_policy: "CertifiedSequentialClosurePolicy | None" = None

    def __post_init__(self) -> None:
        policy = self.sequential_closure_policy
        if policy is not None:
            if (
                not isinstance(policy, CertifiedSequentialClosurePolicy)
                or self.candidate is not None
                or self.display_only_proposal is None
                or self.audit.candidate_role != CANDIDATE_REPRESENTATIVE_ROLE
                or self.audit.failure_reason
                != REPRESENTATIVE_PROPOSAL_FAILURE_REASON
                or policy.possible_first_contact_sets
                != self.possible_first_contact_sets
                or policy.pad_order != self.audit.pad_order
                or policy.independent_actuation_supports
                != self.audit.independent_actuation_supports
                or policy.closing_directions_physical
                != self.audit.closing_directions_physical
                or policy.object_geometry_sha256
                != self.audit.object_geometry_sha256
                or policy.model_contract_sha256
                != self.audit.model_contract_sha256
                or tuple(
                    row.set_sha256
                    for row in policy.possible_first_contact_sets
                )
                != self.audit.possible_first_contact_set_sha256
            ):
                raise RayClosureError(
                    "sequential closure policy is not bound to its V9 evidence"
                )
        if self.display_only_proposal is None:
            return
        if (
            self.candidate is not None
            or self.audit.candidate_role != CANDIDATE_REPRESENTATIVE_ROLE
            or self.audit.failure_reason
            != REPRESENTATIVE_PROPOSAL_FAILURE_REASON
            or len(self.possible_first_contact_sets) != 3
        ):
            raise RayClosureError(
                "display-only proposal cannot enter the formal candidate channel"
            )

    @property
    def feasible(self) -> bool:
        return (
            self.candidate is not None
            and self.audit.failure_reason is None
            and self.audit.candidate_role != CANDIDATE_REPRESENTATIVE_ROLE
        )

    @property
    def candidate_is_representative_proposal(self) -> bool:
        return self.display_only_proposal is not None

    @property
    def static_policy_available(self) -> bool:
        """Whether a contact-stop path exists before collision/wrench gates."""

        return self.sequential_closure_policy is not None

    @property
    def exact_contact_endpoint_certified(self) -> bool:
        return False


@dataclass(frozen=True)
class _DistanceNode:
    lower_m: np.ndarray
    upper_m: np.ndarray
    left: int
    right: int
    face_indices: np.ndarray

    @property
    def leaf(self) -> bool:
        return self.left < 0


@dataclass(frozen=True)
class _QueryAabbNode:
    lower_m: np.ndarray
    upper_m: np.ndarray
    left: int
    right: int
    query_indices: np.ndarray

    @property
    def leaf(self) -> bool:
        return self.left < 0


@dataclass(frozen=True)
class _NearestPoint:
    distance_m: float
    position_m: np.ndarray
    face_index: int
    outward_normal: np.ndarray
    node_visits: int
    triangle_tests: int


@dataclass(frozen=True)
class _NearestMany:
    distances_m: np.ndarray
    positions_m: np.ndarray
    face_indices: np.ndarray
    outward_normals: np.ndarray
    node_visits: np.ndarray
    triangle_tests: np.ndarray

    def __len__(self) -> int:
        return len(self.distances_m)

    def point(self, index: int) -> _NearestPoint:
        return _NearestPoint(
            distance_m=float(self.distances_m[index]),
            position_m=self.positions_m[index],
            face_index=int(self.face_indices[index]),
            outward_normal=self.outward_normals[index],
            node_visits=int(self.node_visits[index]),
            triangle_tests=int(self.triangle_tests[index]),
        )


def _closest_points_on_segment(
    points: np.ndarray, start: np.ndarray, end: np.ndarray
) -> np.ndarray:
    edge = end - start
    denominator = float(edge @ edge)
    if denominator == 0.0:
        return np.broadcast_to(start, points.shape).copy()
    fractions = np.sum((points - start) * edge, axis=1) / denominator
    fractions = np.clip(fractions, 0.0, 1.0)
    return start + fractions[:, None] * edge


def _closest_points_on_triangle(
    points: np.ndarray, triangle: np.ndarray
) -> np.ndarray:
    """Vectorised closest points with the scalar candidate ordering preserved."""

    first, second, third = triangle
    edge_one = second - first
    edge_two = third - first
    normal = np.cross(edge_one, edge_two)
    normal_squared = float(normal @ normal)
    candidates = np.empty((len(points), 4, 3), dtype=np.float64)
    candidates[:, 0] = _closest_points_on_segment(points, first, second)
    candidates[:, 1] = _closest_points_on_segment(points, second, third)
    candidates[:, 2] = _closest_points_on_segment(points, third, first)
    projected_valid = np.zeros(len(points), dtype=bool)
    if normal_squared > 0.0:
        signed_projection = np.sum((points - first) * normal, axis=1)
        projected = points - (signed_projection / normal_squared)[:, None] * normal
        candidates[:, 3] = projected
        dot00 = float(edge_one @ edge_one)
        dot01 = float(edge_one @ edge_two)
        dot11 = float(edge_two @ edge_two)
        offset = projected - first
        dot20 = np.sum(offset * edge_one, axis=1)
        dot21 = np.sum(offset * edge_two, axis=1)
        denominator = dot00 * dot11 - dot01 * dot01
        if denominator > 0.0:
            second_coordinate = (dot11 * dot20 - dot01 * dot21) / denominator
            third_coordinate = (dot00 * dot21 - dot01 * dot20) / denominator
            first_coordinate = 1.0 - second_coordinate - third_coordinate
            projected_valid = (
                (first_coordinate >= -_BARYCENTRIC_ERROR)
                & (second_coordinate >= -_BARYCENTRIC_ERROR)
                & (third_coordinate >= -_BARYCENTRIC_ERROR)
            )
    else:
        candidates[:, 3] = first
    deltas = candidates - points[:, None, :]
    distances_squared = np.sum(deltas * deltas, axis=2)
    distances_squared[~projected_valid, 3] = math.inf
    selected = np.argmin(distances_squared, axis=1)
    return candidates[np.arange(len(points)), selected]


def _closest_points_on_triangle_pairs(
    points: np.ndarray, triangles: np.ndarray
) -> np.ndarray:
    """Closest points for aligned point/triangle rows in one array call."""

    point_rows = np.asarray(points, dtype=np.float64)
    triangle_rows = np.asarray(triangles, dtype=np.float64)
    if (
        point_rows.ndim != 2
        or point_rows.shape[1:] != (3,)
        or triangle_rows.shape != (len(point_rows), 3, 3)
        or not np.all(np.isfinite(point_rows))
        or not np.all(np.isfinite(triangle_rows))
    ):
        raise RayClosureError(
            "paired closest-point inputs must be aligned finite rows"
        )

    def segments(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
        edges = ends - starts
        denominators = np.sum(edges * edges, axis=1)
        numerators = np.sum((point_rows - starts) * edges, axis=1)
        fractions = np.zeros(len(point_rows), dtype=np.float64)
        np.divide(
            numerators,
            denominators,
            out=fractions,
            where=denominators != 0.0,
        )
        fractions = np.clip(fractions, 0.0, 1.0)
        return starts + fractions[:, None] * edges

    first = triangle_rows[:, 0]
    second = triangle_rows[:, 1]
    third = triangle_rows[:, 2]
    edge_one = second - first
    edge_two = third - first
    normals = np.cross(edge_one, edge_two)
    normal_squared = np.sum(normals * normals, axis=1)
    candidates = np.empty((len(point_rows), 4, 3), dtype=np.float64)
    candidates[:, 0] = segments(first, second)
    candidates[:, 1] = segments(second, third)
    candidates[:, 2] = segments(third, first)
    candidates[:, 3] = first
    projected_valid = np.zeros(len(point_rows), dtype=bool)
    nondegenerate = normal_squared > 0.0
    if np.any(nondegenerate):
        rows = np.flatnonzero(nondegenerate)
        signed_projection = np.sum(
            (point_rows[rows] - first[rows]) * normals[rows], axis=1
        )
        projected = point_rows[rows] - (
            signed_projection / normal_squared[rows]
        )[:, None] * normals[rows]
        candidates[rows, 3] = projected
        dot00 = np.sum(edge_one[rows] * edge_one[rows], axis=1)
        dot01 = np.sum(edge_one[rows] * edge_two[rows], axis=1)
        dot11 = np.sum(edge_two[rows] * edge_two[rows], axis=1)
        offset = projected - first[rows]
        dot20 = np.sum(offset * edge_one[rows], axis=1)
        dot21 = np.sum(offset * edge_two[rows], axis=1)
        denominators = dot00 * dot11 - dot01 * dot01
        valid_denominator = denominators > 0.0
        if np.any(valid_denominator):
            valid_rows = rows[valid_denominator]
            denominator = denominators[valid_denominator]
            second_coordinate = (
                dot11[valid_denominator] * dot20[valid_denominator]
                - dot01[valid_denominator] * dot21[valid_denominator]
            ) / denominator
            third_coordinate = (
                dot00[valid_denominator] * dot21[valid_denominator]
                - dot01[valid_denominator] * dot20[valid_denominator]
            ) / denominator
            first_coordinate = 1.0 - second_coordinate - third_coordinate
            projected_valid[valid_rows] = (
                (first_coordinate >= -_BARYCENTRIC_ERROR)
                & (second_coordinate >= -_BARYCENTRIC_ERROR)
                & (third_coordinate >= -_BARYCENTRIC_ERROR)
            )
    deltas = candidates - point_rows[:, None, :]
    distances_squared = np.sum(deltas * deltas, axis=2)
    distances_squared[~projected_valid, 3] = math.inf
    selected = np.argmin(distances_squared, axis=1)
    return candidates[np.arange(len(point_rows)), selected]


def _closest_point_on_triangle(
    point: np.ndarray, triangle: np.ndarray
) -> np.ndarray:
    return _closest_points_on_triangle(point[None, :], triangle)[0]


class _PointTriangleDistanceBvh:
    """Deterministic point/triangle nearest query with no all-face scan."""

    def __init__(self, object_model: ObjectGraspModel) -> None:
        vertices = np.asarray(object_model.mesh.vertices_m, dtype=np.float64)
        self.centre_m = np.mean(vertices, axis=0)
        radius = float(np.max(np.linalg.norm(vertices - self.centre_m, axis=1)))
        self.characteristic_length_m = 2.0 * radius
        if (
            not math.isfinite(self.characteristic_length_m)
            or self.characteristic_length_m <= 0.0
        ):
            raise RayClosureError(
                "object mesh must have a positive finite characteristic length"
            )
        self.triangles = np.asarray(
            object_model.mesh.face_vertices_m - self.centre_m, dtype=np.float64
        )
        self.normals = np.asarray(object_model.mesh.face_normals, dtype=np.float64)
        self.vertices = np.asarray(
            vertices - self.centre_m, dtype=np.float64
        )
        self._triangle_first_m = self.triangles[:, 0]
        self._triangle_second_m = self.triangles[:, 1]
        self._triangle_third_m = self.triangles[:, 2]
        self._triangle_edge_one_m = (
            self._triangle_second_m - self._triangle_first_m
        )
        self._triangle_edge_two_m = (
            self._triangle_third_m - self._triangle_first_m
        )
        self._triangle_second_segment_m = (
            self._triangle_third_m - self._triangle_second_m
        )
        self._triangle_third_segment_m = (
            self._triangle_first_m - self._triangle_third_m
        )
        self._triangle_normal = np.cross(
            self._triangle_edge_one_m, self._triangle_edge_two_m
        )
        self._triangle_normal_squared = np.sum(
            self._triangle_normal * self._triangle_normal, axis=1
        )
        self._triangle_dot00 = np.sum(
            self._triangle_edge_one_m * self._triangle_edge_one_m, axis=1
        )
        self._triangle_dot01 = np.sum(
            self._triangle_edge_one_m * self._triangle_edge_two_m, axis=1
        )
        self._triangle_dot11 = np.sum(
            self._triangle_edge_two_m * self._triangle_edge_two_m, axis=1
        )
        self._triangle_second_segment_squared = np.sum(
            self._triangle_second_segment_m
            * self._triangle_second_segment_m,
            axis=1,
        )
        self._triangle_third_segment_squared = np.sum(
            self._triangle_third_segment_m
            * self._triangle_third_segment_m,
            axis=1,
        )
        self._triangle_barycentric_denominator = (
            self._triangle_dot00 * self._triangle_dot11
            - self._triangle_dot01 * self._triangle_dot01
        )
        for array in (
            self._triangle_first_m,
            self._triangle_second_m,
            self._triangle_third_m,
            self._triangle_edge_one_m,
            self._triangle_edge_two_m,
            self._triangle_second_segment_m,
            self._triangle_third_segment_m,
            self._triangle_normal,
            self._triangle_normal_squared,
            self._triangle_dot00,
            self._triangle_dot01,
            self._triangle_dot11,
            self._triangle_second_segment_squared,
            self._triangle_third_segment_squared,
            self._triangle_barycentric_denominator,
        ):
            array.setflags(write=False)
        self.vertices.setflags(write=False)
        self.vertex_upper_bound_tree = cKDTree(
            self.vertices,
            compact_nodes=True,
            balanced_tree=True,
            copy_data=True,
        )
        face_lower = np.min(self.triangles, axis=1)
        face_upper = np.max(self.triangles, axis=1)
        self.face_lower_m = face_lower
        self.face_upper_m = face_upper
        centroids = np.mean(self.triangles, axis=1)
        nodes: list[_DistanceNode | None] = []

        def build(indices: np.ndarray) -> int:
            node_index = len(nodes)
            nodes.append(None)
            lower = np.min(face_lower[indices], axis=0)
            upper = np.max(face_upper[indices], axis=0)
            if len(indices) <= _DISTANCE_BVH_LEAF_CAPACITY:
                leaf = np.array(indices, dtype=np.int64, copy=True)
                leaf.setflags(write=False)
                nodes[node_index] = _DistanceNode(lower, upper, -1, -1, leaf)
                return node_index
            values = centroids[indices]
            extent = np.max(values, axis=0) - np.min(values, axis=0)
            axis = int(np.argmax(extent))
            second_axis = (axis + 1) % 3
            third_axis = (axis + 2) % 3
            order = np.lexsort(
                (
                    indices,
                    values[:, third_axis],
                    values[:, second_axis],
                    values[:, axis],
                )
            )
            ordered = indices[order]
            middle = len(ordered) // 2
            left = build(ordered[:middle])
            right = build(ordered[middle:])
            empty = np.empty(0, dtype=np.int64)
            empty.setflags(write=False)
            nodes[node_index] = _DistanceNode(lower, upper, left, right, empty)
            return node_index

        self.root = build(np.arange(len(self.triangles), dtype=np.int64))
        self.nodes = tuple(node for node in nodes if node is not None)
        allowed_face_mask = np.asarray(
            object_model.contact_face_mask, dtype=bool
        )
        allowed_counts = np.zeros(len(self.nodes), dtype=np.int64)
        subtree_face_counts = np.zeros(len(self.nodes), dtype=np.int64)
        for node_index in reversed(range(len(self.nodes))):
            node = self.nodes[node_index]
            if node.leaf:
                subtree_face_counts[node_index] = len(node.face_indices)
                allowed_counts[node_index] = int(
                    np.count_nonzero(
                        allowed_face_mask[node.face_indices]
                    )
                )
                continue
            children = (node.left, node.right)
            subtree_face_counts[node_index] = sum(
                int(subtree_face_counts[child]) for child in children
            )
            allowed_counts[node_index] = sum(
                int(allowed_counts[child]) for child in children
            )
        allowed_counts.setflags(write=False)
        subtree_face_counts.setflags(write=False)
        self.allowed_contact_face_count = allowed_counts
        self.subtree_face_count = subtree_face_counts
        self.aabb_error_bound_m = _AABB_ERROR * self.characteristic_length_m

    @staticmethod
    def _point_aabb_distances(
        points: np.ndarray, node: _DistanceNode
    ) -> np.ndarray:
        delta = np.maximum(
            np.maximum(node.lower_m - points, points - node.upper_m), 0.0
        )
        return np.sqrt(np.sum(delta * delta, axis=1))

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    def _vertex_surface_distance_upper_bounds(
        self, points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return conservative distances to actual object surface vertices.

        The spatial index chooses only a vertex identity.  Its reported
        distance is deliberately ignored: the distance to that stored vertex
        is recomputed here and rounded upward.  Therefore even a numerically
        imperfect index result remains a valid upper bound on point-to-surface
        distance and can affect pruning only, never the returned nearest face.
        """

        rows = np.asarray(points, dtype=np.float64)
        if (
            rows.ndim != 2
            or rows.shape[1:] != (3,)
            or len(rows) == 0
            or not np.all(np.isfinite(rows))
        ):
            raise RayClosureError(
                "vertex upper-bound points need finite non-empty shape (N, 3)"
            )
        _unused_distances, vertex_indices = (
            self.vertex_upper_bound_tree.query(
                rows,
                k=1,
                eps=0.0,
                workers=1,
            )
        )
        vertex_indices = np.asarray(vertex_indices, dtype=np.int64).reshape(-1)
        if (
            vertex_indices.shape != (len(rows),)
            or np.any(vertex_indices < 0)
            or np.any(vertex_indices >= len(self.vertices))
        ):
            raise RayClosureError(
                "vertex upper-bound index returned an invalid vertex"
            )
        selected_vertices = self.vertices[vertex_indices]
        deltas = rows - selected_vertices
        distances = np.sqrt(np.sum(deltas * deltas, axis=1))
        forward_error = _DOT_ERROR * (
            self.characteristic_length_m
            + np.max(np.abs(rows), axis=1)
            + np.max(np.abs(selected_vertices), axis=1)
            + distances
        )
        upper = np.nextafter(distances + forward_error, math.inf)
        if not np.all(np.isfinite(upper)) or np.any(upper < 0.0):
            raise RayClosureError(
                "vertex upper-bound distance is not finite"
            )
        upper.setflags(write=False)
        vertex_indices.setflags(write=False)
        return upper, vertex_indices

    def _closest_points_on_face_product(
        self,
        points: np.ndarray,
        face_indices: np.ndarray,
    ) -> np.ndarray:
        """Return every point-by-face closest point without row expansion."""

        point_rows = np.asarray(points, dtype=np.float64)
        faces = np.asarray(face_indices, dtype=np.int64)
        if (
            point_rows.ndim != 2
            or point_rows.shape[1:] != (3,)
            or len(point_rows) == 0
            or faces.ndim != 1
            or len(faces) == 0
            or np.any(faces < 0)
            or np.any(faces >= len(self.triangles))
            or not np.all(np.isfinite(point_rows))
        ):
            raise RayClosureError(
                "point-by-face closest-point inputs are malformed"
            )
        first = self._triangle_first_m[faces]
        second = self._triangle_second_m[faces]
        third = self._triangle_third_m[faces]
        edge_one = self._triangle_edge_one_m[faces]
        edge_two = self._triangle_edge_two_m[faces]
        normals = self._triangle_normal[faces]
        normal_squared = self._triangle_normal_squared[faces]

        def segments(
            starts: np.ndarray,
            edges: np.ndarray,
            denominators: np.ndarray,
        ) -> np.ndarray:
            numerators = np.sum(
                (point_rows[:, None, :] - starts[None, :, :])
                * edges[None, :, :],
                axis=2,
            )
            fractions = np.zeros(
                (len(point_rows), len(faces)), dtype=np.float64
            )
            np.divide(
                numerators,
                denominators[None, :],
                out=fractions,
                where=denominators[None, :] != 0.0,
            )
            np.clip(fractions, 0.0, 1.0, out=fractions)
            return (
                starts[None, :, :]
                + fractions[:, :, None] * edges[None, :, :]
            )

        candidates = np.empty(
            (len(point_rows), len(faces), 4, 3), dtype=np.float64
        )
        candidates[:, :, 0] = segments(
            first, edge_one, self._triangle_dot00[faces]
        )
        candidates[:, :, 1] = segments(
            second,
            self._triangle_second_segment_m[faces],
            self._triangle_second_segment_squared[faces],
        )
        candidates[:, :, 2] = segments(
            third,
            self._triangle_third_segment_m[faces],
            self._triangle_third_segment_squared[faces],
        )
        candidates[:, :, 3] = first[None, :, :]
        projected_valid = np.zeros(
            (len(point_rows), len(faces)), dtype=bool
        )
        nondegenerate = normal_squared > 0.0
        if np.any(nondegenerate):
            columns = np.flatnonzero(nondegenerate)
            signed_projection = np.sum(
                (
                    point_rows[:, None, :]
                    - first[None, columns, :]
                )
                * normals[None, columns, :],
                axis=2,
            )
            projected = point_rows[:, None, :] - (
                signed_projection / normal_squared[None, columns]
            )[:, :, None] * normals[None, columns, :]
            candidates[:, columns, 3] = projected
            offset = projected - first[None, columns, :]
            dot20 = np.sum(
                offset * edge_one[None, columns, :], axis=2
            )
            dot21 = np.sum(
                offset * edge_two[None, columns, :], axis=2
            )
            denominator = self._triangle_barycentric_denominator[
                faces[columns]
            ]
            valid_denominator = denominator > 0.0
            if np.any(valid_denominator):
                valid_columns = columns[valid_denominator]
                denominator_rows = denominator[valid_denominator][None, :]
                dot00 = self._triangle_dot00[
                    faces[valid_columns]
                ][None, :]
                dot01 = self._triangle_dot01[
                    faces[valid_columns]
                ][None, :]
                dot11 = self._triangle_dot11[
                    faces[valid_columns]
                ][None, :]
                dot20 = dot20[:, valid_denominator]
                dot21 = dot21[:, valid_denominator]
                second_coordinate = (
                    dot11 * dot20 - dot01 * dot21
                ) / denominator_rows
                third_coordinate = (
                    dot00 * dot21 - dot01 * dot20
                ) / denominator_rows
                first_coordinate = (
                    1.0 - second_coordinate - third_coordinate
                )
                projected_valid[:, valid_columns] = (
                    (first_coordinate >= -_BARYCENTRIC_ERROR)
                    & (second_coordinate >= -_BARYCENTRIC_ERROR)
                    & (third_coordinate >= -_BARYCENTRIC_ERROR)
                )
        deltas = candidates - point_rows[:, None, None, :]
        distances_squared = np.sum(deltas * deltas, axis=3)
        distances_squared[:, :, 3][~projected_valid] = math.inf
        selected = np.argmin(distances_squared, axis=2)
        return candidates[
            np.arange(len(point_rows))[:, None],
            np.arange(len(faces))[None, :],
            selected,
        ]

    def nearest_many(
        self,
        points_m: Sequence[Sequence[float]],
        *,
        _use_vertex_upper_bound_seed: bool = True,
    ) -> _NearestMany:
        points_world = np.asarray(points_m, dtype=np.float64)
        if (
            points_world.ndim != 2
            or points_world.shape[1:] != (3,)
            or len(points_world) == 0
            or not np.all(np.isfinite(points_world))
        ):
            raise RayClosureError(
                "distance query points must have finite non-empty shape (P, 3)"
            )
        if type(_use_vertex_upper_bound_seed) is not bool:
            raise RayClosureError(
                "vertex upper-bound seed switch must be boolean"
            )
        points = points_world - self.centre_m
        count = len(points)
        best_distances = (
            self._vertex_surface_distance_upper_bounds(points)[0].copy()
            if _use_vertex_upper_bound_seed
            else np.full(count, math.inf, dtype=np.float64)
        )
        best_positions = np.full((count, 3), math.nan, dtype=np.float64)
        best_faces = np.full(count, -1, dtype=np.int64)
        node_visits = np.zeros(count, dtype=np.int64)
        triangle_tests = np.zeros(count, dtype=np.int64)

        def visit(node_index: int, query_indices: np.ndarray) -> None:
            if len(query_indices) == 0:
                return
            node = self.nodes[node_index]
            lower_bounds = self._point_aabb_distances(
                points[query_indices], node
            )
            active = query_indices[
                lower_bounds
                <= best_distances[query_indices] + self.aabb_error_bound_m
            ]
            if len(active) == 0:
                return
            node_visits[active] += 1
            if node.leaf:
                triangle_tests[active] += len(node.face_indices)
                face_indices = np.asarray(
                    node.face_indices, dtype=np.int64
                )
                closest_rows = self._closest_points_on_face_product(
                    points[active], face_indices
                )
                for face_offset, face_index_value in enumerate(face_indices):
                    face_index = int(face_index_value)
                    closest = closest_rows[:, face_offset]
                    delta = closest - points[active]
                    distances = np.sqrt(np.sum(delta * delta, axis=1))
                    current_distances = best_distances[active]
                    current_faces = best_faces[active]
                    better = (
                        distances
                        < current_distances - self.aabb_error_bound_m
                    ) | (
                        (
                            np.abs(distances - current_distances)
                            <= self.aabb_error_bound_m
                        )
                        & ((current_faces < 0) | (face_index < current_faces))
                    )
                    selected = active[better]
                    best_distances[selected] = distances[better]
                    best_positions[selected] = closest[better]
                    best_faces[selected] = face_index
                return

            left_distances = self._point_aabb_distances(
                points[active], self.nodes[node.left]
            )
            right_distances = self._point_aabb_distances(
                points[active], self.nodes[node.right]
            )
            left_first = (left_distances < right_distances) | (
                (left_distances == right_distances) & (node.left < node.right)
            )
            left_first_indices = active[left_first]
            right_first_indices = active[~left_first]
            visit(node.left, left_first_indices)
            visit(node.right, left_first_indices)
            visit(node.right, right_first_indices)
            visit(node.left, right_first_indices)

        visit(self.root, np.arange(count, dtype=np.int64))
        if (
            np.any(best_faces < 0)
            or not np.all(np.isfinite(best_distances))
            or not np.all(np.isfinite(best_positions))
        ):
            raise RayClosureError("object distance BVH produced no finite nearest point")
        positions_world = best_positions + self.centre_m
        outward_normals = self.normals[best_faces]
        for array in (
            best_distances,
            positions_world,
            best_faces,
            outward_normals,
            node_visits,
            triangle_tests,
        ):
            array.setflags(write=False)
        return _NearestMany(
            distances_m=best_distances,
            positions_m=positions_world,
            face_indices=best_faces,
            outward_normals=outward_normals,
            node_visits=node_visits,
            triangle_tests=triangle_tests,
        )

    def nearest(self, point_m: Sequence[float]) -> _NearestPoint:
        point = np.asarray(point_m, dtype=np.float64)
        if point.shape != (3,):
            raise RayClosureError("distance query point must be one finite three-vector")
        return self.nearest_many(point[None, :]).point(0)

    def face_indices_intersecting_aabb(
        self,
        lower_world_m: Sequence[float],
        upper_world_m: Sequence[float],
    ) -> np.ndarray:
        """Return a deterministic superset of faces touching a closed AABB."""

        lower_world = np.asarray(lower_world_m, dtype=np.float64)
        upper_world = np.asarray(upper_world_m, dtype=np.float64)
        if (
            lower_world.shape != (3,)
            or upper_world.shape != (3,)
            or not np.all(np.isfinite(lower_world))
            or not np.all(np.isfinite(upper_world))
            or np.any(lower_world > upper_world)
        ):
            raise RayClosureError(
                "AABB query requires finite ordered three-vectors"
            )
        query_lower = np.nextafter(
            lower_world - self.centre_m, -math.inf
        )
        query_upper = np.nextafter(
            upper_world - self.centre_m, math.inf
        )
        rows: list[int] = []
        stack = [self.root]
        while stack:
            node_index = stack.pop()
            node = self.nodes[node_index]
            if np.any(node.upper_m < query_lower) or np.any(
                node.lower_m > query_upper
            ):
                continue
            if node.leaf:
                for face_index_value in node.face_indices:
                    face_index = int(face_index_value)
                    if np.any(
                        self.face_upper_m[face_index] < query_lower
                    ) or np.any(
                        self.face_lower_m[face_index] > query_upper
                    ):
                        continue
                    rows.append(face_index)
                continue
            stack.append(node.right)
            stack.append(node.left)
        result = np.asarray(sorted(set(rows)), dtype=np.int64)
        result.setflags(write=False)
        return result

    def _iter_face_pairs_intersecting_aabbs(
        self,
        lower_world_m: Sequence[Sequence[float]],
        upper_world_m: Sequence[Sequence[float]],
        *,
        maximum_pair_count: int,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield exact closed-AABB face overlaps by dual-tree traversal."""

        lower_world = np.asarray(lower_world_m, dtype=np.float64)
        upper_world = np.asarray(upper_world_m, dtype=np.float64)
        if (
            lower_world.ndim != 2
            or lower_world.shape[1:] != (3,)
            or len(lower_world) == 0
            or upper_world.shape != lower_world.shape
            or not np.all(np.isfinite(lower_world))
            or not np.all(np.isfinite(upper_world))
            or np.any(lower_world > upper_world)
            or isinstance(maximum_pair_count, bool)
            or not isinstance(maximum_pair_count, int)
            or maximum_pair_count <= 0
        ):
            raise RayClosureError(
                "dual batch AABB query requires finite non-empty ordered "
                "shape (N, 3) and a positive packet size"
            )
        query_lower = np.nextafter(
            lower_world - self.centre_m[None, :], -math.inf
        )
        query_upper = np.nextafter(
            upper_world - self.centre_m[None, :], math.inf
        )
        query_centres = query_lower + 0.5 * (query_upper - query_lower)
        query_nodes: list[_QueryAabbNode | None] = []

        def build_query_tree(indices: np.ndarray) -> int:
            node_index = len(query_nodes)
            query_nodes.append(None)
            lower = np.min(query_lower[indices], axis=0)
            upper = np.max(query_upper[indices], axis=0)
            frozen_indices = np.asarray(indices, dtype=np.int64).copy()
            frozen_indices.setflags(write=False)
            if len(indices) <= _DISTANCE_BVH_LEAF_CAPACITY:
                query_nodes[node_index] = _QueryAabbNode(
                    lower, upper, -1, -1, frozen_indices
                )
                return node_index
            values = query_centres[indices]
            extent = np.max(values, axis=0) - np.min(values, axis=0)
            axis = int(np.argmax(extent))
            second_axis = (axis + 1) % 3
            third_axis = (axis + 2) % 3
            order = np.lexsort(
                (
                    indices,
                    values[:, third_axis],
                    values[:, second_axis],
                    values[:, axis],
                )
            )
            ordered = indices[order]
            middle = len(ordered) // 2
            left = build_query_tree(ordered[:middle])
            right = build_query_tree(ordered[middle:])
            query_nodes[node_index] = _QueryAabbNode(
                lower, upper, left, right, frozen_indices
            )
            return node_index

        query_root = build_query_tree(
            np.arange(len(lower_world), dtype=np.int64)
        )
        frozen_query_nodes = tuple(
            node for node in query_nodes if node is not None
        )
        pending_query_chunks: list[np.ndarray] = []
        pending_face_chunks: list[np.ndarray] = []
        pending_count = 0
        stack: list[tuple[int, int]] = [(query_root, self.root)]
        while stack:
            query_node_index, object_node_index = stack.pop()
            query_node = frozen_query_nodes[query_node_index]
            object_node = self.nodes[object_node_index]
            if np.any(object_node.upper_m < query_node.lower_m) or np.any(
                object_node.lower_m > query_node.upper_m
            ):
                continue
            if query_node.leaf and object_node.leaf:
                query_indices = query_node.query_indices
                face_indices = object_node.face_indices
                face_overlap = np.all(
                    self.face_upper_m[face_indices][None, :, :]
                    >= query_lower[query_indices][:, None, :],
                    axis=2,
                ) & np.all(
                    self.face_lower_m[face_indices][None, :, :]
                    <= query_upper[query_indices][:, None, :],
                    axis=2,
                )
                query_offsets, face_offsets = np.nonzero(face_overlap)
                if len(query_offsets) > 0:
                    pending_query_chunks.append(
                        query_indices[query_offsets]
                    )
                    pending_face_chunks.append(face_indices[face_offsets])
                    pending_count += len(query_offsets)
                if pending_count >= maximum_pair_count:
                    flat_queries = np.concatenate(pending_query_chunks)
                    flat_faces = np.concatenate(pending_face_chunks)
                    complete_count = (
                        len(flat_queries) // maximum_pair_count
                    ) * maximum_pair_count
                    for packet_lower in range(
                        0, complete_count, maximum_pair_count
                    ):
                        packet_upper = packet_lower + maximum_pair_count
                        yield (
                            flat_queries[packet_lower:packet_upper],
                            flat_faces[packet_lower:packet_upper],
                        )
                    if complete_count < len(flat_queries):
                        pending_query_chunks = [
                            flat_queries[complete_count:]
                        ]
                        pending_face_chunks = [flat_faces[complete_count:]]
                        pending_count = len(flat_queries) - complete_count
                    else:
                        pending_query_chunks = []
                        pending_face_chunks = []
                        pending_count = 0
                continue
            if object_node.leaf or (
                not query_node.leaf
                and len(query_node.query_indices)
                >= int(self.subtree_face_count[object_node_index])
            ):
                stack.append((query_node.right, object_node_index))
                stack.append((query_node.left, object_node_index))
            else:
                stack.append((query_node_index, object_node.right))
                stack.append((query_node_index, object_node.left))

        if pending_count > 0:
            yield (
                np.concatenate(pending_query_chunks),
                np.concatenate(pending_face_chunks),
            )

    def face_indices_intersecting_aabbs(
        self,
        lower_world_m: Sequence[Sequence[float]],
        upper_world_m: Sequence[Sequence[float]],
    ) -> tuple[np.ndarray, ...]:
        """Query many closed AABBs with deterministic dual-tree packets."""

        lower_world = np.asarray(lower_world_m, dtype=np.float64)
        upper_world = np.asarray(upper_world_m, dtype=np.float64)
        face_count = len(self.face_lower_m)
        if (
            lower_world.ndim == 2
            and len(lower_world) > 0
            and len(lower_world) > np.iinfo(np.int64).max // face_count
        ):
            raise RayClosureError("batch AABB packed key would overflow")
        pair_key_chunks = [
            query_indices * face_count + face_indices
            for query_indices, face_indices in (
                self._iter_face_pairs_intersecting_aabbs(
                    lower_world,
                    upper_world,
                    maximum_pair_count=_EXACT_FACE_PAIR_PACKET_SIZE,
                )
            )
        ]

        if not pair_key_chunks:
            result = []
            for _index in range(len(lower_world)):
                row = np.empty(0, dtype=np.int64)
                row.setflags(write=False)
                result.append(row)
            return tuple(result)
        # Every face belongs to exactly one recursively partitioned BVH leaf,
        # so each query-face pair is unique by construction.  One packed
        # integer sort yields the same query-major, ascending-face order as a
        # two-column lexsort without its second full index array or dedup pass.
        packed_keys = np.concatenate(pair_key_chunks)
        packed_keys.sort(kind="quicksort")
        flat_queries = packed_keys // face_count
        flat_faces = packed_keys % face_count
        counts = np.bincount(
            flat_queries, minlength=len(lower_world)
        )
        offsets = np.concatenate(
            (np.asarray((0,), dtype=np.int64), np.cumsum(counts))
        )
        flat_faces.setflags(write=False)
        result: list[np.ndarray] = []
        for query_index in range(len(lower_world)):
            row = flat_faces[offsets[query_index] : offsets[query_index + 1]]
            row.setflags(write=False)
            result.append(row)
        return tuple(result)

    def aabbs_have_face_overlap(
        self,
        lower_world_m: Sequence[Sequence[float]],
        upper_world_m: Sequence[Sequence[float]],
    ) -> np.ndarray:
        """Return only whether each closed AABB overlaps any face AABB.

        This follows the same outward-expanded closed-box predicate as
        ``face_indices_intersecting_aabbs`` but stops each query after its
        first real face overlap.  It is therefore an exact boolean projection
        of that complete query, not a new geometric approximation.
        """

        lower_world = np.asarray(lower_world_m, dtype=np.float64)
        upper_world = np.asarray(upper_world_m, dtype=np.float64)
        if (
            lower_world.ndim != 2
            or lower_world.shape[1:] != (3,)
            or len(lower_world) == 0
            or upper_world.shape != lower_world.shape
            or not np.all(np.isfinite(lower_world))
            or not np.all(np.isfinite(upper_world))
            or np.any(lower_world > upper_world)
        ):
            raise RayClosureError(
                "batch AABB overlap query requires finite non-empty ordered shape (N, 3)"
            )
        query_lower = np.nextafter(
            lower_world - self.centre_m[None, :], -math.inf
        )
        query_upper = np.nextafter(
            upper_world - self.centre_m[None, :], math.inf
        )
        result = np.zeros(len(lower_world), dtype=bool)
        stack: list[tuple[int, np.ndarray, bool]] = [
            (
                self.root,
                np.arange(len(lower_world), dtype=np.int64),
                False,
            )
        ]
        while stack:
            node_index, query_indices, already_filtered = stack.pop()
            unresolved = query_indices[~result[query_indices]]
            if len(unresolved) == 0:
                continue
            node = self.nodes[node_index]
            if already_filtered:
                active = unresolved
            else:
                overlap = np.all(
                    node.upper_m[None, :] >= query_lower[unresolved],
                    axis=1,
                ) & np.all(
                    node.lower_m[None, :] <= query_upper[unresolved],
                    axis=1,
                )
                active = unresolved[overlap]
            if len(active) == 0:
                continue
            if node.leaf:
                face_indices = np.asarray(
                    node.face_indices, dtype=np.int64
                )
                face_overlap = np.all(
                    self.face_upper_m[face_indices][None, :, :]
                    >= query_lower[active][:, None, :],
                    axis=2,
                ) & np.all(
                    self.face_lower_m[face_indices][None, :, :]
                    <= query_upper[active][:, None, :],
                    axis=2,
                )
                result[active[np.any(face_overlap, axis=1)]] = True
                continue
            for child_index in (node.right, node.left):
                child = self.nodes[child_index]
                child_overlap = np.all(
                    child.upper_m[None, :] >= query_lower[active],
                    axis=1,
                ) & np.all(
                    child.lower_m[None, :] <= query_upper[active],
                    axis=1,
                )
                child_active = active[child_overlap]
                if len(child_active) > 0:
                    stack.append((child_index, child_active, True))
        result.setflags(write=False)
        return result


@dataclass(frozen=True)
class _PadSurfaceSphereNode:
    """Sphere enclosing every source triangle assigned to one tree node."""

    center_link_m: np.ndarray
    radius_upper_m: float
    box_half_extents_upper_m: np.ndarray
    maximum_vertex_radius_link_m: float
    depth: int
    left: int
    right: int
    triangle_indices: np.ndarray

    @property
    def leaf(self) -> bool:
        return self.left < 0


@dataclass(frozen=True)
class _PadSurfaceAabbNode:
    """Tight local box over a complete, disjoint PAD triangle subset."""

    center_link_m: np.ndarray
    box_half_extents_upper_m: np.ndarray
    maximum_vertex_radius_link_m: float
    depth: int
    left: int
    right: int
    triangle_indices: np.ndarray

    @property
    def leaf(self) -> bool:
        return self.left < 0


@dataclass(frozen=True)
class _PreparedPad:
    verified: VerifiedPad
    witness_points_link_m: np.ndarray
    witness_normals_link: np.ndarray
    triangle_indices: np.ndarray
    witness_indices: np.ndarray
    barycentric_coordinates: np.ndarray
    surface_centroid_link_m: np.ndarray
    maximum_point_radius_link_m: float
    full_pad_sphere_center_link_m: np.ndarray
    full_pad_sphere_radius_upper_m: float
    surface_sphere_nodes: tuple[_PadSurfaceSphereNode, ...]
    surface_sphere_root: int
    surface_aabb_nodes: tuple[_PadSurfaceAabbNode, ...]
    surface_aabb_root: int
    surface_aabb_leaf_node_indices: np.ndarray
    surface_aabb_leaf_triangle_indices: np.ndarray
    surface_aabb_left_child_indices: np.ndarray
    surface_aabb_right_child_indices: np.ndarray
    surface_aabb_internal_nodes_by_reverse_depth: tuple[np.ndarray, ...]
    surface_triangles_link_m: np.ndarray
    relevant_joint_indices: tuple[int, ...]
    witness_hierarchy_nodes: tuple["_WitnessHierarchyNode", ...]
    witness_hierarchy_root: int


@dataclass(frozen=True)
class _WitnessHierarchyNode:
    """A deterministic sphere enclosing every indexed finite-PAD witness."""

    center_link_m: np.ndarray
    radius_upper_m: float
    coordinate_scale_link_m: float
    left: int
    right: int
    witness_indices: np.ndarray

    @property
    def leaf(self) -> bool:
        return self.left < 0


@dataclass(frozen=True)
class WholePathPadSphereScreen:
    """One conservative complete-path PAD/surface separation result."""

    pad_name: str
    finger_name: str
    segment_count: int
    nearest_surface_query_count: int
    distance_bvh_node_visits: int
    distance_triangle_tests: int
    minimum_clearance_lower_bound_m: float
    certified_free: bool
    spatial_node_query_count: int = 0
    aabb_certified_free_node_count: int = 0
    exact_distance_query_count: int = 0
    maximum_spatial_depth_reached: int = 0
    obb_sat_certified_free_node_count: int = 0
    obb_sat_triangle_test_count: int = 0
    moving_triangle_sat_certified_free_pair_count: int = 0
    moving_triangle_sat_pair_test_count: int = 0
    temporal_refined_leaf_pair_count: int = 0
    temporal_refinement_transform_count: int = 0
    maximum_temporal_refinement_depth_reached: int = 0
    narrowphase_refinement_used: bool = False
    narrowphase_work_budget_exhausted: bool = False
    directional_contact_feasibility_used: bool = False
    directional_bvh_node_pair_test_count: int = 0
    directional_bvh_node_pair_rejected_count: int = 0
    directional_leaf_face_pair_test_count: int = 0
    directional_leaf_face_pair_rejected_count: int = 0
    directional_interval_witness_motion_evaluation_count: int = 0
    certified_no_valid_contact: bool = False
    skipped_due_to_other_pad_free: bool = False
    root_overlap_segment_count: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "pad_name": self.pad_name,
            "finger_name": self.finger_name,
            "segment_count": self.segment_count,
            "nearest_surface_query_count": self.nearest_surface_query_count,
            "distance_bvh_node_visits": self.distance_bvh_node_visits,
            "distance_triangle_tests": self.distance_triangle_tests,
            "minimum_clearance_lower_bound_m": (
                self.minimum_clearance_lower_bound_m
            ),
            "certified_free": self.certified_free,
            "spatial_node_query_count": self.spatial_node_query_count,
            "aabb_certified_free_node_count": (
                self.aabb_certified_free_node_count
            ),
            "exact_distance_query_count": self.exact_distance_query_count,
            "maximum_spatial_depth_reached": (
                self.maximum_spatial_depth_reached
            ),
            "obb_sat_certified_free_node_count": (
                self.obb_sat_certified_free_node_count
            ),
            "obb_sat_triangle_test_count": (
                self.obb_sat_triangle_test_count
            ),
            "moving_triangle_sat_certified_free_pair_count": (
                self.moving_triangle_sat_certified_free_pair_count
            ),
            "moving_triangle_sat_pair_test_count": (
                self.moving_triangle_sat_pair_test_count
            ),
            "temporal_refined_leaf_pair_count": (
                self.temporal_refined_leaf_pair_count
            ),
            "temporal_refinement_transform_count": (
                self.temporal_refinement_transform_count
            ),
            "maximum_temporal_refinement_depth_reached": (
                self.maximum_temporal_refinement_depth_reached
            ),
            "narrowphase_refinement_used": (
                self.narrowphase_refinement_used
            ),
            "narrowphase_work_budget_exhausted": (
                self.narrowphase_work_budget_exhausted
            ),
            "directional_contact_feasibility_used": (
                self.directional_contact_feasibility_used
            ),
            "directional_bvh_node_pair_test_count": (
                self.directional_bvh_node_pair_test_count
            ),
            "directional_bvh_node_pair_rejected_count": (
                self.directional_bvh_node_pair_rejected_count
            ),
            "directional_leaf_face_pair_test_count": (
                self.directional_leaf_face_pair_test_count
            ),
            "directional_leaf_face_pair_rejected_count": (
                self.directional_leaf_face_pair_rejected_count
            ),
            "directional_interval_witness_motion_evaluation_count": (
                self.directional_interval_witness_motion_evaluation_count
            ),
            "root_overlap_segment_count": self.root_overlap_segment_count,
            "certified_no_valid_contact": (
                self.certified_no_valid_contact
            ),
            "skipped_due_to_other_pad_free": (
                self.skipped_due_to_other_pad_free
            ),
        }


@dataclass(frozen=True)
class _WholePathPadSphereCoverage:
    prepared_pad_index: int
    centers_object_m: np.ndarray
    radius_upper_m: np.ndarray
    spatial_error_bound_m: float


@dataclass(frozen=True)
class _WholePathPadSphereHierarchyCoverage:
    prepared_pad_index: int
    rotations_object_from_link: np.ndarray
    translations_object_m: np.ndarray
    node_radius_upper_m: np.ndarray
    node_box_half_extents_upper_m: np.ndarray
    spatial_error_bound_m: float


@dataclass(frozen=True)
class _WholePathPadAabbHierarchyCoverage:
    prepared_pad_index: int
    rotations_object_from_link: np.ndarray
    translations_object_m: np.ndarray
    motion_speed_radius_slope_upper_per_unit: float
    motion_speed_intercept_upper_m_per_unit: float
    segment_half_width_unit: float
    spatial_error_bound_m: float
    q_start: np.ndarray
    direction: np.ndarray
    maximum_parameter: float
    object_from_hand: np.ndarray


@dataclass(frozen=True)
class _DirectionalWitnessSegmentBounds:
    pad_approach_possible: np.ndarray
    node_pad_approach_possible: np.ndarray
    interval_witness_motion_evaluation_count: int


@dataclass(frozen=True)
class _WitnessStates:
    positions_object_m: np.ndarray
    velocities_object_per_unit: np.ndarray
    pad_source_winding_normals_object: np.ndarray
    leading: np.ndarray
    leading_error_bounds: np.ndarray
    link_rotation_object: np.ndarray
    link_translation_object_m: np.ndarray

    def __len__(self) -> int:
        return len(self.positions_object_m)


@dataclass(frozen=True)
class _ContactEvent:
    normalized_closure: float
    interval_width: float
    spatial_error_bound_m: float
    triangle_index: int
    witness_index: int
    witness_flat_index: int
    object_face_index: int
    classification: str
    first_hit: FirstHitResult | None


class _PadSearchState(str, Enum):
    CERTIFIED_FREE = "CERTIFIED_FREE"
    CERTIFIED_ROOT = "CERTIFIED_ROOT"
    UNRESOLVED = "UNRESOLVED"


class _ParentPairInheritance(str, Enum):
    PRUNE_PARENT_CERTIFIED_FREE = "PRUNE_PARENT_CERTIFIED_FREE"
    PRUNE_PARENT_ROOT_DISJOINT = "PRUNE_PARENT_ROOT_DISJOINT"
    REUSE_PARENT_ROOT = "REUSE_PARENT_ROOT"
    RECOMPUTE = "RECOMPUTE"


@dataclass(frozen=True)
class CertifiedContactFeatureRoot:
    """A certified implicit root bound to one PAD witness/object face."""

    pad_name: str
    witness_flat_index: int
    pad_triangle_index: int
    witness_index: int
    object_face_index: int
    semantic_classification: str
    certificate: IntervalTransverseRootCertificate

    def __post_init__(self) -> None:
        if not isinstance(self.pad_name, str) or not self.pad_name:
            raise RayClosureError("certified contact root requires a PAD name")
        for label, value in (
            ("witness flat", self.witness_flat_index),
            ("PAD triangle", self.pad_triangle_index),
            ("witness", self.witness_index),
            ("object face", self.object_face_index),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                raise RayClosureError(
                    f"certified contact root {label} index is invalid"
                )
        if (
            not isinstance(self.semantic_classification, str)
            or not self.semantic_classification
            or not isinstance(
                self.certificate, IntervalTransverseRootCertificate
            )
        ):
            raise RayClosureError(
                "certified contact root semantic/certificate binding is invalid"
            )

    def _evidence_dict(self) -> dict[str, object]:
        certificate = self.certificate
        return {
            "pad_name": self.pad_name,
            "witness_flat_index": self.witness_flat_index,
            "pad_triangle_index": self.pad_triangle_index,
            "witness_index": self.witness_index,
            "object_face_index": self.object_face_index,
            "semantic_classification": self.semantic_classification,
            "implicit_root": certificate.implicit_root.as_dict(),
            "triangle_edge_halfspaces": [
                row.as_dict() for row in certificate.triangle_edge_halfspaces
            ],
            "pad_approach": certificate.pad_approach.as_dict(),
            "path_local_free_side_approach": (
                certificate.path_local_free_side_approach.as_dict()
            ),
            "object_source_winding_free_side_sign": (
                certificate.object_source_winding_free_side_sign
            ),
            "position_object_m": [
                row.as_dict() for row in certificate.position_object_m
            ],
            "bisection_iterations": certificate.bisection_iterations,
            "interval_method_id": certificate.method_id,
            "decimal_precision": certificate.decimal_precision,
        }

    @property
    def binding_sha256(self) -> str:
        return hashlib.sha256(
            _canonical_json(self._evidence_dict()).encode("utf-8")
        ).hexdigest()

    def as_dict(self) -> dict[str, object]:
        result = self._evidence_dict()
        result["binding_sha256"] = self.binding_sha256
        return result


def _possible_root_order_key(
    root: CertifiedContactFeatureRoot,
) -> tuple[float, float, int, int, str]:
    return (
        root.certificate.phase.lower,
        root.certificate.phase.upper,
        root.object_face_index,
        root.witness_flat_index,
        root.binding_sha256,
    )


@dataclass(frozen=True)
class PossibleFirstContactSet:
    """All roots that interval ordering cannot exclude from being earliest."""

    method_id: str
    pad_name: str
    all_certified_roots: tuple[CertifiedContactFeatureRoot, ...]
    possible_earliest_ordering: tuple[str, ...]
    guaranteed_earliest_phase_upper: float
    excluded_strictly_later_root_count: int
    ordering_policy: str
    display_proposal_binding_sha256: str
    display_proposal_role: str

    @classmethod
    def from_certified_roots(
        cls,
        roots: Sequence[CertifiedContactFeatureRoot],
    ) -> "PossibleFirstContactSet":
        canonical_roots = tuple(sorted(roots, key=_possible_root_order_key))
        if not canonical_roots:
            raise RayClosureError(
                "possible-first-contact set requires certified roots"
            )
        pad_names = {root.pad_name for root in canonical_roots}
        if len(pad_names) != 1:
            raise RayClosureError(
                "possible-first-contact roots must belong to one PAD"
            )
        earliest_upper = min(
            root.certificate.phase.upper for root in canonical_roots
        )
        possible = tuple(
            root
            for root in canonical_roots
            if root.certificate.phase.lower <= earliest_upper
        )
        proposal = min(possible, key=_possible_root_order_key)
        return cls(
            method_id=POSSIBLE_FIRST_CONTACT_SET_METHOD_ID,
            pad_name=next(iter(pad_names)),
            all_certified_roots=canonical_roots,
            possible_earliest_ordering=tuple(
                root.binding_sha256 for root in possible
            ),
            guaranteed_earliest_phase_upper=earliest_upper,
            excluded_strictly_later_root_count=(
                len(canonical_roots) - len(possible)
            ),
            ordering_policy=POSSIBLE_EARLIEST_ORDERING_POLICY,
            display_proposal_binding_sha256=proposal.binding_sha256,
            display_proposal_role=CANDIDATE_REPRESENTATIVE_ROLE,
        )

    def __post_init__(self) -> None:
        if (
            self.method_id != POSSIBLE_FIRST_CONTACT_SET_METHOD_ID
            or self.ordering_policy != POSSIBLE_EARLIEST_ORDERING_POLICY
            or self.display_proposal_role != CANDIDATE_REPRESENTATIVE_ROLE
            or not self.pad_name
            or not self.all_certified_roots
        ):
            raise RayClosureError(
                "possible-first-contact set provenance is incomplete"
            )
        canonical_roots = tuple(
            sorted(self.all_certified_roots, key=_possible_root_order_key)
        )
        if canonical_roots != self.all_certified_roots:
            raise RayClosureError(
                "possible-first-contact roots are not in canonical order"
            )
        if any(root.pad_name != self.pad_name for root in canonical_roots):
            raise RayClosureError(
                "possible-first-contact root PAD identity drifted"
            )
        binding_ids = tuple(root.binding_sha256 for root in canonical_roots)
        if len(set(binding_ids)) != len(binding_ids):
            raise RayClosureError(
                "possible-first-contact set contains duplicate root bindings"
            )
        expected_upper = min(
            root.certificate.phase.upper for root in canonical_roots
        )
        expected_possible = tuple(
            root
            for root in canonical_roots
            if root.certificate.phase.lower <= expected_upper
        )
        expected_ordering = tuple(
            root.binding_sha256 for root in expected_possible
        )
        if (
            self.guaranteed_earliest_phase_upper != expected_upper
            or self.possible_earliest_ordering != expected_ordering
            or self.excluded_strictly_later_root_count
            != len(canonical_roots) - len(expected_possible)
            or self.display_proposal_binding_sha256
            != min(expected_possible, key=_possible_root_order_key).binding_sha256
        ):
            raise RayClosureError(
                "possible-first-contact ordering contradicts root intervals"
            )

    @property
    def possible_earliest_roots(
        self,
    ) -> tuple[CertifiedContactFeatureRoot, ...]:
        by_digest = {
            root.binding_sha256: root for root in self.all_certified_roots
        }
        return tuple(
            by_digest[digest] for digest in self.possible_earliest_ordering
        )

    @property
    def display_proposal_root(self) -> CertifiedContactFeatureRoot:
        for root in self.all_certified_roots:
            if root.binding_sha256 == self.display_proposal_binding_sha256:
                return root
        raise RayClosureError("display proposal root binding is absent")

    @property
    def set_sha256(self) -> str:
        return hashlib.sha256(
            _canonical_json(self._evidence_dict()).encode("utf-8")
        ).hexdigest()

    def _evidence_dict(self) -> dict[str, object]:
        return {
            "method_id": self.method_id,
            "pad_name": self.pad_name,
            "all_certified_roots": [
                root.as_dict() for root in self.all_certified_roots
            ],
            "possible_earliest_ordering": list(
                self.possible_earliest_ordering
            ),
            "guaranteed_earliest_phase_upper_binary64_hex": (
                float(self.guaranteed_earliest_phase_upper).hex()
            ),
            "excluded_strictly_later_root_count": (
                self.excluded_strictly_later_root_count
            ),
            "ordering_policy": self.ordering_policy,
            "display_proposal_binding_sha256": (
                self.display_proposal_binding_sha256
            ),
            "display_proposal_role": self.display_proposal_role,
        }

    def as_dict(self) -> dict[str, object]:
        result = self._evidence_dict()
        result["set_sha256"] = self.set_sha256
        return result


@dataclass(frozen=True)
class CertifiedSequentialClosurePolicy:
    """Static contact-stop plan with no invented exact contact endpoint.

    The object/hand transform and initial joint state are exact binary64
    planning inputs.  Every finger endpoint remains the complete certified
    possible-first-contact set.  A runtime controller may later bind a real
    sensor event to the stop action, but this value does not claim that such an
    event has occurred and carries no force, collision, or dynamic acceptance.
    """

    object_from_hand: tuple[float, ...]
    initial_independent_joint_positions_rad: tuple[float, ...]
    independent_joint_names: tuple[str, ...]
    pad_order: tuple[str, str, str]
    independent_actuation_supports: tuple[tuple[str, ...], ...]
    closing_directions_physical: tuple[tuple[float, ...], ...]
    possible_first_contact_sets: tuple[PossibleFirstContactSet, ...]
    object_geometry_sha256: str
    model_contract_sha256: str
    method_id: str = SEQUENTIAL_CLOSURE_POLICY_METHOD_ID
    execution_semantics: str = SEQUENTIAL_CLOSURE_EXECUTION_SEMANTICS

    def __post_init__(self) -> None:
        transform_values = tuple(float(value) for value in self.object_from_hand)
        initial_q = tuple(
            float(value)
            for value in self.initial_independent_joint_positions_rad
        )
        joint_names = tuple(str(name) for name in self.independent_joint_names)
        pad_order = tuple(str(name) for name in self.pad_order)
        supports = tuple(
            tuple(str(name) for name in row)
            for row in self.independent_actuation_supports
        )
        directions = tuple(
            tuple(float(value) for value in row)
            for row in self.closing_directions_physical
        )
        contact_sets = tuple(self.possible_first_contact_sets)
        if (
            self.method_id != SEQUENTIAL_CLOSURE_POLICY_METHOD_ID
            or self.execution_semantics
            != SEQUENTIAL_CLOSURE_EXECUTION_SEMANTICS
        ):
            raise RayClosureError(
                "sequential closure policy method/semantics changed"
            )
        if len(transform_values) != 16 or not all(
            math.isfinite(value) for value in transform_values
        ):
            raise RayClosureError(
                "sequential closure policy needs one finite 4x4 hand pose"
            )
        transform = np.asarray(transform_values, dtype=np.float64).reshape(4, 4)
        tolerance = 64.0 * np.finfo(np.float64).eps
        if (
            float(
                np.linalg.norm(
                    transform[3] - np.asarray((0.0, 0.0, 0.0, 1.0))
                )
            )
            > tolerance
            or float(
                np.linalg.norm(
                    transform[:3, :3].T @ transform[:3, :3] - np.eye(3)
                )
            )
            > tolerance
            or abs(float(np.linalg.det(transform[:3, :3])) - 1.0)
            > tolerance
        ):
            raise RayClosureError(
                "sequential closure policy hand pose is not a proper transform"
            )
        if (
            not initial_q
            or len(initial_q) != len(joint_names)
            or len(set(joint_names)) != len(joint_names)
            or any(not name for name in joint_names)
            or not all(math.isfinite(value) for value in initial_q)
        ):
            raise RayClosureError(
                "sequential closure policy initial joint state is malformed"
            )
        if (
            len(pad_order) != 3
            or len(set(pad_order)) != 3
            or any(not name for name in pad_order)
            or len(supports) != 3
            or len(directions) != 3
            or len(contact_sets) != 3
        ):
            raise RayClosureError(
                "sequential closure policy must bind exactly three fingers"
            )
        used_supports: set[str] = set()
        for index, (support, direction) in enumerate(zip(supports, directions)):
            if (
                not support
                or len(set(support)) != len(support)
                or used_supports.intersection(support)
                or any(name not in joint_names for name in support)
                or len(direction) != len(joint_names)
                or not all(math.isfinite(value) for value in direction)
            ):
                raise RayClosureError(
                    f"sequential closure policy support {index} is malformed"
                )
            observed = tuple(
                name
                for name, value in zip(joint_names, direction)
                if value != 0.0
            )
            if observed != support:
                raise RayClosureError(
                    f"sequential closure direction {index} differs from its support"
                )
            used_supports.update(support)
        if any(
            not isinstance(row, PossibleFirstContactSet)
            or row.pad_name != pad_name
            for row, pad_name in zip(contact_sets, pad_order)
        ):
            raise RayClosureError(
                "sequential closure contact sets differ from the PAD order"
            )
        for label, digest in (
            ("object geometry", self.object_geometry_sha256),
            ("model contract", self.model_contract_sha256),
        ):
            if not isinstance(digest, str) or _HEX_SHA256.fullmatch(digest) is None:
                raise RayClosureError(
                    f"sequential closure policy {label} hash is invalid"
                )
        object.__setattr__(self, "object_from_hand", transform_values)
        object.__setattr__(
            self, "initial_independent_joint_positions_rad", initial_q
        )
        object.__setattr__(self, "independent_joint_names", joint_names)
        object.__setattr__(self, "pad_order", pad_order)
        object.__setattr__(self, "independent_actuation_supports", supports)
        object.__setattr__(self, "closing_directions_physical", directions)
        object.__setattr__(self, "possible_first_contact_sets", contact_sets)

    @staticmethod
    def _formal_root_document(
        root: CertifiedContactFeatureRoot,
    ) -> dict[str, object]:
        certificate = root.certificate
        implicit = certificate.implicit_root
        return {
            "pad_name": root.pad_name,
            "witness_flat_index": root.witness_flat_index,
            "pad_triangle_index": root.pad_triangle_index,
            "witness_index": root.witness_index,
            "object_face_index": root.object_face_index,
            "semantic_classification": root.semantic_classification,
            "implicit_root": {
                "method_id": implicit.method_id,
                "equation_sha256": implicit.equation_sha256,
                "feature_identity_sha256": implicit.feature_identity_sha256,
                "feature_type": implicit.feature_type,
                "isolating_interval": implicit.isolating_interval.as_dict(),
                "value_at_lower": implicit.value_at_lower.as_dict(),
                "value_at_upper": implicit.value_at_upper.as_dict(),
                "derivative": implicit.derivative.as_dict(),
                "uniqueness_proven": implicit.uniqueness_proven,
            },
            "triangle_edge_halfspaces": [
                row.as_dict() for row in certificate.triangle_edge_halfspaces
            ],
            "pad_approach": certificate.pad_approach.as_dict(),
            "path_local_free_side_approach": (
                certificate.path_local_free_side_approach.as_dict()
            ),
            "object_source_winding_free_side_sign": (
                certificate.object_source_winding_free_side_sign
            ),
            "position_object_m": [
                row.as_dict() for row in certificate.position_object_m
            ],
            "bisection_iterations": certificate.bisection_iterations,
            "interval_method_id": certificate.method_id,
            "decimal_precision": certificate.decimal_precision,
        }

    @classmethod
    def _formal_root_sha256(
        cls, root: CertifiedContactFeatureRoot
    ) -> str:
        return hashlib.sha256(
            _canonical_json(cls._formal_root_document(root)).encode("utf-8")
        ).hexdigest()

    @classmethod
    def _formal_contact_set_document(
        cls, contact_set: PossibleFirstContactSet
    ) -> dict[str, object]:
        roots = contact_set.all_certified_roots
        return {
            "method_id": contact_set.method_id,
            "pad_name": contact_set.pad_name,
            "all_certified_roots": [
                cls._formal_root_document(root) for root in roots
            ],
            "possible_earliest_formal_root_sha256": [
                cls._formal_root_sha256(root)
                for root in contact_set.possible_earliest_roots
            ],
            "guaranteed_earliest_phase_upper_binary64_hex": float(
                contact_set.guaranteed_earliest_phase_upper
            ).hex(),
            "excluded_strictly_later_root_count": (
                contact_set.excluded_strictly_later_root_count
            ),
            "ordering_policy": contact_set.ordering_policy,
        }

    def _formal_document(self) -> dict[str, object]:
        return {
            "method_id": self.method_id,
            "execution_semantics": self.execution_semantics,
            "object_from_hand_binary64_hex": _float64_array_hex(
                np.asarray(self.object_from_hand, dtype=np.float64).reshape(4, 4)
            ),
            "initial_independent_joint_positions_rad_binary64_hex": (
                _float64_array_hex(
                    self.initial_independent_joint_positions_rad
                )
            ),
            "independent_joint_names": list(self.independent_joint_names),
            "pad_order": list(self.pad_order),
            "independent_actuation_supports": [
                list(row) for row in self.independent_actuation_supports
            ],
            "closing_directions_physical_binary64_hex": _float64_array_hex(
                self.closing_directions_physical
            ),
            "possible_first_contact_sets": [
                self._formal_contact_set_document(row)
                for row in self.possible_first_contact_sets
            ],
            "object_geometry_sha256": self.object_geometry_sha256,
            "model_contract_sha256": self.model_contract_sha256,
            "exact_final_joint_vector_present": False,
            "exact_contact_points_present": False,
            "display_approximation_used_as_formal_evidence": False,
        }

    @property
    def policy_sha256(self) -> str:
        return hashlib.sha256(
            _canonical_json(self._formal_document()).encode("utf-8")
        ).hexdigest()

    def as_dict(self) -> dict[str, object]:
        result = self._formal_document()
        result["policy_sha256"] = self.policy_sha256
        return result


_CertifiedContactRootBinding = CertifiedContactFeatureRoot


@dataclass(frozen=True)
class _PairIntervalClassification:
    state: _PadSearchState
    witness_flat_index: int
    object_face_index: int
    possible_phase_lower: float
    root: _CertifiedContactRootBinding | None
    reason: str


@dataclass(frozen=True)
class _PlaneRootGroupWork:
    """One exact plane shared by one witness and one or more real faces."""

    witness_flat_index: int
    representative_face_index: int
    face_indices: np.ndarray


@dataclass(frozen=True)
class _PrecertifiedPlaneGate:
    """Shared rigorous monotonicity and endpoint bounds for one plane."""

    plane_derivative: IntervalBounds
    lower_value: IntervalBounds
    upper_value: IntervalBounds


@dataclass(frozen=True)
class _PreRootSpatialEnclosureBatch:
    """Rigorous root phase and point boxes derived without exact root calls."""

    valid: np.ndarray
    phase_lower: np.ndarray
    phase_upper: np.ndarray
    position_lower_object_m: np.ndarray
    position_upper_object_m: np.ndarray


@dataclass(frozen=True)
class _PadSearchOutcome:
    state: _PadSearchState
    interval_lower: float
    interval_upper: float
    possible_first_contact_set: PossibleFirstContactSet | None
    unresolved_reason: str | None

    @property
    def roots(self) -> tuple[CertifiedContactFeatureRoot, ...]:
        if self.possible_first_contact_set is None:
            return ()
        return self.possible_first_contact_set.possible_earliest_roots

    def __post_init__(self) -> None:
        if self.interval_lower > self.interval_upper:
            raise RayClosureError("PAD search interval is reversed")
        if self.state is _PadSearchState.CERTIFIED_ROOT:
            if (
                self.possible_first_contact_set is None
                or self.unresolved_reason is not None
            ):
                raise RayClosureError(
                    "certified PAD root outcome requires roots only"
                )
        elif self.possible_first_contact_set is not None:
            raise RayClosureError(
                "only certified PAD root outcomes may carry roots"
            )
        if (
            self.state is _PadSearchState.UNRESOLVED
            and not self.unresolved_reason
        ):
            raise RayClosureError(
                "unresolved PAD search outcome requires a reason"
            )


@dataclass
class _PadCounters:
    interval_evaluations: int = 0
    leading_evaluations: int = 0
    rays: int = 0
    finite_chord_feature_candidates: int = 0
    nonlinear_feature_roots_solved: int = 0
    nonlinear_root_fk_evaluations: int = 0
    distance_node_visits: int = 0
    distance_triangle_tests: int = 0
    certified_free_intervals: int = 0
    clearance_lower_bound_m: float = math.inf
    swept_face_candidates: int = 0
    certified_batch_linear_free_pairs: int = 0
    batch_linear_uncertain_pairs: int = 0
    interval_point_motion_evaluations: int = 0
    interval_pair_evaluations: int = 0
    actual_plane_root_evaluations: int = 0
    batch_root_triangle_free_pairs: int = 0
    batch_root_triangle_uncertain_pairs: int = 0
    batch_plane_monotone_same_side_free: int = 0
    shared_plane_gate_roots: int = 0
    parallel_plane_root_tasks: int = 0
    pre_root_spatial_enclosure_groups: int = 0
    pre_root_spatial_free_pairs: int = 0
    pre_root_spatial_fully_free_groups: int = 0
    parent_certified_free_pair_prunes: int = 0
    parent_certified_root_outside_pair_prunes: int = 0
    parent_certified_root_pair_reuses: int = 0
    large_exact_batch_temporal_deferrals: int = 0
    large_exact_batch_deferred_root_groups: int = 0
    swept_face_witness_stages: int = 0
    swept_face_witnesses_materialized: int = 0
    staged_potential_root_temporal_deferrals: int = 0
    staged_potential_root_groups: int = 0
    staged_unmaterialized_witnesses: int = 0
    parent_frontier_geometry_bypasses: int = 0
    pre_nearest_aabb_witness_tests: int = 0
    pre_nearest_aabb_certified_free_witnesses: int = 0
    pre_nearest_aabb_exact_survivors: int = 0
    pre_nearest_aabb_fast_paths: int = 0
    pre_nearest_aabb_fallbacks: int = 0
    root_interpolation_iterations: int = 0
    interval_newton_iterations: int = 0
    root_bisection_iterations: int = 0
    certified_contact_roots: int = 0
    unresolved_witness_face_pairs: int = 0
    cofirst_root_count: int = 0
    possible_earliest_root_count: int = 0
    competing_root_order_blocks: int = 0
    whole_path_sphere_screen_segments: int = 0
    whole_path_sphere_screen_queries: int = 0
    whole_path_sphere_screen_bvh_node_visits: int = 0
    whole_path_sphere_screen_triangle_tests: int = 0
    whole_path_sphere_screen_obb_sat_certified_free_nodes: int = 0
    whole_path_sphere_screen_obb_sat_triangle_tests: int = 0
    whole_path_sphere_screen_moving_triangle_sat_certified_free_pairs: int = 0
    whole_path_sphere_screen_moving_triangle_sat_pair_tests: int = 0
    whole_path_sphere_screen_temporal_refined_leaf_pairs: int = 0
    whole_path_sphere_screen_temporal_refinement_transforms: int = 0
    whole_path_sphere_screen_maximum_temporal_refinement_depth: int = 0
    whole_path_sphere_screen_narrowphase_refinement_used: bool = False
    whole_path_sphere_screen_narrowphase_work_budget_exhausted: bool = False
    whole_path_sphere_screen_directional_contact_feasibility_used: bool = False
    whole_path_sphere_screen_directional_bvh_node_pair_tests: int = 0
    whole_path_sphere_screen_directional_bvh_node_pair_rejections: int = 0
    whole_path_sphere_screen_directional_leaf_face_pair_tests: int = 0
    whole_path_sphere_screen_directional_leaf_face_pair_rejections: int = 0
    whole_path_sphere_screen_directional_interval_witness_motion_evaluations: int = 0
    whole_path_sphere_screen_certified_no_valid_contact: bool = False
    whole_path_sphere_screen_certified_free: bool = False
    whole_path_sphere_screen_clearance_lower_bound_m: float | None = None


@dataclass
class _GeometryExecutionStats:
    """Non-audit implementation counters local to exactly one evaluation."""

    witness_state_cache_hits: int = 0
    witness_state_cache_misses: int = 0
    nearest_batch_cache_hits: int = 0
    nearest_batch_cache_misses: int = 0
    witness_hierarchy_nodes_tested: int = 0
    witness_hierarchy_witnesses_pruned: int = 0
    exact_nearest_witness_queries: int = 0
    reference_shadow_witness_queries: int = 0
    fail_closed_fingers_skipped: int = 0
    interval_transform_cache_hits: int = 0
    interval_transform_cache_misses: int = 0
    interval_transform_cache_peak_entries: int = 0
    interval_point_cache_hits: int = 0
    interval_point_cache_misses: int = 0
    interval_point_cache_nonprimary_bypasses: int = 0
    interval_point_cache_peak_entries: int = 0
    interval_pad_area_cache_hits: int = 0
    interval_pad_area_cache_misses: int = 0
    interval_pad_area_cache_nonprimary_bypasses: int = 0
    interval_pad_area_cache_peak_entries: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "witness_state_cache_hits": self.witness_state_cache_hits,
            "witness_state_cache_misses": self.witness_state_cache_misses,
            "nearest_batch_cache_hits": self.nearest_batch_cache_hits,
            "nearest_batch_cache_misses": self.nearest_batch_cache_misses,
            "witness_hierarchy_nodes_tested": self.witness_hierarchy_nodes_tested,
            "witness_hierarchy_witnesses_pruned": (
                self.witness_hierarchy_witnesses_pruned
            ),
            "exact_nearest_witness_queries": self.exact_nearest_witness_queries,
            "reference_shadow_witness_queries": (
                self.reference_shadow_witness_queries
            ),
            "fail_closed_fingers_skipped": self.fail_closed_fingers_skipped,
            "interval_transform_cache_hits": (
                self.interval_transform_cache_hits
            ),
            "interval_transform_cache_misses": (
                self.interval_transform_cache_misses
            ),
            "interval_transform_cache_peak_entries": (
                self.interval_transform_cache_peak_entries
            ),
            "interval_point_cache_hits": self.interval_point_cache_hits,
            "interval_point_cache_misses": self.interval_point_cache_misses,
            "interval_point_cache_nonprimary_bypasses": (
                self.interval_point_cache_nonprimary_bypasses
            ),
            "interval_point_cache_peak_entries": (
                self.interval_point_cache_peak_entries
            ),
            "interval_pad_area_cache_hits": (
                self.interval_pad_area_cache_hits
            ),
            "interval_pad_area_cache_misses": (
                self.interval_pad_area_cache_misses
            ),
            "interval_pad_area_cache_nonprimary_bypasses": (
                self.interval_pad_area_cache_nonprimary_bypasses
            ),
            "interval_pad_area_cache_peak_entries": (
                self.interval_pad_area_cache_peak_entries
            ),
        }


@dataclass
class _GeometryExecutionContext:
    """Per-call deterministic caches; never retained by the surface model.

    ``verify_full_nearest`` runs the H2 all-witness nearest query as a shadow
    oracle.  Its work is deliberately absent from the scientific audit and
    cannot influence candidate construction.  It exists to prove that the
    hierarchy has preserved the old possible-witness set and free-interval
    minimum margin.
    """

    cache_enabled: bool = True
    verify_full_nearest: bool = False
    stats: _GeometryExecutionStats = field(default_factory=_GeometryExecutionStats)
    witness_state_cache: dict[tuple[object, ...], _WitnessStates] = field(
        default_factory=dict, repr=False
    )
    nearest_batch_cache: dict[tuple[object, ...], _NearestMany] = field(
        default_factory=dict, repr=False
    )


@dataclass(frozen=True)
class _IntervalGeometry:
    possible: np.ndarray
    nearest_face_indices: np.ndarray
    minimum_free_margin_m: float | None


@dataclass
class _Budget:
    maximum: int
    used: int = 0

    def consume(self) -> None:
        if self.used >= self.maximum:
            raise _SubdivisionBudgetExhausted
        self.used += 1


def _prepare_pad(
    pad: VerifiedPad,
    *,
    relevant_joint_indices: tuple[int, ...],
) -> _PreparedPad:
    points = np.asarray(pad.points_local_m, dtype=np.float64)
    faces = np.asarray(pad.faces, dtype=np.int64)
    triangles = np.asarray(points[faces], dtype=np.float64).copy()
    triangles.setflags(write=False)
    area_vectors = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    lengths = np.linalg.norm(area_vectors, axis=1)
    if np.any(lengths == 0.0):
        raise RayClosureError(f"verified PAD {pad.name} contains a degenerate triangle")
    normals = area_vectors / lengths[:, None]
    triangle_centroids = np.mean(triangles, axis=1)
    surface_centroid = np.sum(
        lengths[:, None] * triangle_centroids, axis=0
    ) / math.fsum(float(length) for length in lengths)
    witness_points = np.einsum(
        "wk,fkj->fwj", _BARYCENTRIC_WITNESSES, triangles
    ).reshape((-1, 3))
    witness_normals = np.repeat(normals, len(_BARYCENTRIC_WITNESSES), axis=0)
    triangle_indices = np.repeat(
        np.arange(len(faces), dtype=np.int64), len(_BARYCENTRIC_WITNESSES)
    )
    witness_indices = np.tile(
        np.arange(len(_BARYCENTRIC_WITNESSES), dtype=np.int64), len(faces)
    )
    barycentric = np.tile(_BARYCENTRIC_WITNESSES, (len(faces), 1))
    maximum_radius = float(np.max(np.linalg.norm(points, axis=1)))
    sphere_lower = np.min(points, axis=0)
    sphere_upper = np.max(points, axis=0)
    sphere_center = sphere_lower + 0.5 * (sphere_upper - sphere_lower)
    raw_sphere_radius = float(
        np.max(np.linalg.norm(points - sphere_center, axis=1))
    )
    sphere_scale = max(
        1.0,
        float(np.max(np.abs(points))),
        float(np.max(np.abs(sphere_center))),
        raw_sphere_radius,
    )
    sphere_radius_upper = float(
        np.nextafter(
            raw_sphere_radius + _DOT_ERROR * sphere_scale,
            math.inf,
        )
    )
    sphere_center = np.asarray(sphere_center, dtype=np.float64)
    sphere_center.setflags(write=False)
    surface_sphere_nodes: list[_PadSurfaceSphereNode | None] = []

    def build_surface_sphere_hierarchy(
        indices: np.ndarray,
        depth: int,
    ) -> int:
        node_index = len(surface_sphere_nodes)
        surface_sphere_nodes.append(None)
        node_triangles = triangles[indices]
        node_vertices = node_triangles.reshape((-1, 3))
        lower = np.min(node_vertices, axis=0)
        upper = np.max(node_vertices, axis=0)
        center = lower + 0.5 * (upper - lower)
        raw_box_half_extents = 0.5 * (upper - lower)
        raw_radius = float(
            np.max(np.linalg.norm(node_vertices - center, axis=1))
        )
        scale = max(
            1.0,
            float(np.max(np.abs(node_vertices))),
            float(np.max(np.abs(center))),
            raw_radius,
        )
        radius_upper = float(
            np.nextafter(raw_radius + _DOT_ERROR * scale, math.inf)
        )
        box_half_extents_upper = np.nextafter(
            raw_box_half_extents + _DOT_ERROR * scale,
            math.inf,
        )
        maximum_vertex_radius = float(
            np.nextafter(
                np.max(np.linalg.norm(node_vertices, axis=1)),
                math.inf,
            )
        )
        frozen_center = np.asarray(center, dtype=np.float64)
        frozen_center.setflags(write=False)
        box_half_extents_upper.setflags(write=False)
        frozen_indices = np.asarray(indices, dtype=np.int64).copy()
        frozen_indices.setflags(write=False)
        leaf = (
            depth >= _PAD_SURFACE_SPHERE_HIERARCHY_MAXIMUM_DEPTH
            or len(indices) <= 1
        )
        left = -1
        right = -1
        if not leaf:
            values = triangle_centroids[indices]
            extent = np.max(values, axis=0) - np.min(values, axis=0)
            axis = int(np.argmax(extent))
            second_axis = (axis + 1) % 3
            third_axis = (axis + 2) % 3
            order = np.lexsort(
                (
                    indices,
                    values[:, third_axis],
                    values[:, second_axis],
                    values[:, axis],
                )
            )
            ordered = indices[order]
            middle = len(ordered) // 2
            left = build_surface_sphere_hierarchy(
                ordered[:middle], depth + 1
            )
            right = build_surface_sphere_hierarchy(
                ordered[middle:], depth + 1
            )
        surface_sphere_nodes[node_index] = _PadSurfaceSphereNode(
            center_link_m=frozen_center,
            radius_upper_m=radius_upper,
            box_half_extents_upper_m=box_half_extents_upper,
            maximum_vertex_radius_link_m=maximum_vertex_radius,
            depth=depth,
            left=left,
            right=right,
            triangle_indices=frozen_indices,
        )
        return node_index

    surface_sphere_root = build_surface_sphere_hierarchy(
        np.arange(len(triangles), dtype=np.int64), 0
    )
    surface_root_node = surface_sphere_nodes[surface_sphere_root]
    if surface_root_node is None:  # pragma: no cover
        raise RayClosureError("PAD surface sphere root was not constructed")
    sphere_center = surface_root_node.center_link_m
    sphere_radius_upper = surface_root_node.radius_upper_m
    surface_aabb_nodes: list[_PadSurfaceAabbNode | None] = []

    def build_surface_aabb_hierarchy(
        indices: np.ndarray,
        depth: int,
    ) -> int:
        """Build a full balanced tree whose leaves are source triangles."""

        node_index = len(surface_aabb_nodes)
        surface_aabb_nodes.append(None)
        node_triangles = triangles[indices]
        node_vertices = node_triangles.reshape((-1, 3))
        lower = np.min(node_vertices, axis=0)
        upper = np.max(node_vertices, axis=0)
        center = lower + 0.5 * (upper - lower)
        scale = max(
            1.0,
            float(np.max(np.abs(node_vertices))),
            float(np.max(np.abs(center))),
        )
        half_extents = np.nextafter(
            0.5 * (upper - lower) + _DOT_ERROR * scale,
            math.inf,
        )
        maximum_vertex_radius = float(
            np.nextafter(
                np.max(np.linalg.norm(node_vertices, axis=1)),
                math.inf,
            )
        )
        frozen_center = np.asarray(center, dtype=np.float64)
        frozen_center.setflags(write=False)
        half_extents.setflags(write=False)
        frozen_indices = np.asarray(indices, dtype=np.int64).copy()
        frozen_indices.setflags(write=False)
        left = -1
        right = -1
        if len(indices) > 1:
            values = triangle_centroids[indices]
            extent = np.max(values, axis=0) - np.min(values, axis=0)
            axis = int(np.argmax(extent))
            second_axis = (axis + 1) % 3
            third_axis = (axis + 2) % 3
            order = np.lexsort(
                (
                    indices,
                    values[:, third_axis],
                    values[:, second_axis],
                    values[:, axis],
                )
            )
            ordered = indices[order]
            middle = len(ordered) // 2
            left = build_surface_aabb_hierarchy(
                ordered[:middle], depth + 1
            )
            right = build_surface_aabb_hierarchy(
                ordered[middle:], depth + 1
            )
        surface_aabb_nodes[node_index] = _PadSurfaceAabbNode(
            center_link_m=frozen_center,
            box_half_extents_upper_m=half_extents,
            maximum_vertex_radius_link_m=maximum_vertex_radius,
            depth=depth,
            left=left,
            right=right,
            triangle_indices=frozen_indices,
        )
        return node_index

    surface_aabb_root = build_surface_aabb_hierarchy(
        np.arange(len(triangles), dtype=np.int64), 0
    )
    hierarchy_nodes: list[_WitnessHierarchyNode | None] = []

    def build_witness_hierarchy(indices: np.ndarray) -> int:
        node_index = len(hierarchy_nodes)
        hierarchy_nodes.append(None)
        values = witness_points[indices]
        center = np.mean(values, axis=0)
        raw_radius = float(np.max(np.linalg.norm(values - center, axis=1)))
        radius_upper = float(
            np.nextafter(raw_radius * (1.0 + _FK_ERROR), math.inf)
        )
        coordinate_scale = float(
            max(
                np.max(np.abs(values)),
                np.max(np.abs(center)),
                radius_upper,
            )
        )
        center = np.array(center, dtype=np.float64, copy=True)
        center.setflags(write=False)
        frozen_indices = np.array(indices, dtype=np.int64, copy=True)
        frozen_indices.setflags(write=False)
        if len(indices) <= _DISTANCE_BVH_LEAF_CAPACITY:
            hierarchy_nodes[node_index] = _WitnessHierarchyNode(
                center_link_m=center,
                radius_upper_m=radius_upper,
                coordinate_scale_link_m=coordinate_scale,
                left=-1,
                right=-1,
                witness_indices=frozen_indices,
            )
            return node_index
        extent = np.max(values, axis=0) - np.min(values, axis=0)
        axis = int(np.argmax(extent))
        second_axis = (axis + 1) % 3
        third_axis = (axis + 2) % 3
        order = np.lexsort(
            (
                indices,
                values[:, third_axis],
                values[:, second_axis],
                values[:, axis],
            )
        )
        ordered = indices[order]
        middle = len(ordered) // 2
        left = build_witness_hierarchy(ordered[:middle])
        right = build_witness_hierarchy(ordered[middle:])
        hierarchy_nodes[node_index] = _WitnessHierarchyNode(
            center_link_m=center,
            radius_upper_m=radius_upper,
            coordinate_scale_link_m=coordinate_scale,
            left=left,
            right=right,
            witness_indices=frozen_indices,
        )
        return node_index

    hierarchy_root = build_witness_hierarchy(
        np.arange(len(witness_points), dtype=np.int64)
    )
    frozen_surface_aabb_nodes = tuple(
        node for node in surface_aabb_nodes if node is not None
    )
    surface_aabb_left_child_indices = np.asarray(
        [node.left for node in frozen_surface_aabb_nodes],
        dtype=np.int64,
    )
    surface_aabb_right_child_indices = np.asarray(
        [node.right for node in frozen_surface_aabb_nodes],
        dtype=np.int64,
    )
    surface_aabb_leaf_node_indices = np.flatnonzero(
        surface_aabb_left_child_indices < 0
    ).astype(np.int64, copy=False)
    surface_aabb_leaf_triangle_indices = np.asarray(
        [
            frozen_surface_aabb_nodes[int(node_index)].triangle_indices[0]
            for node_index in surface_aabb_leaf_node_indices
        ],
        dtype=np.int64,
    )
    internal_depths = sorted(
        {
            node.depth
            for node in frozen_surface_aabb_nodes
            if not node.leaf
        },
        reverse=True,
    )
    surface_aabb_internal_nodes_by_reverse_depth = tuple(
        np.asarray(
            [
                node_index
                for node_index, node in enumerate(
                    frozen_surface_aabb_nodes
                )
                if not node.leaf and node.depth == depth
            ],
            dtype=np.int64,
        )
        for depth in internal_depths
    )
    for array in (
        witness_points,
        witness_normals,
        triangle_indices,
        witness_indices,
        barycentric,
        surface_centroid,
        surface_aabb_leaf_node_indices,
        surface_aabb_leaf_triangle_indices,
        surface_aabb_left_child_indices,
        surface_aabb_right_child_indices,
        *surface_aabb_internal_nodes_by_reverse_depth,
    ):
        array.setflags(write=False)
    return _PreparedPad(
        verified=pad,
        witness_points_link_m=witness_points,
        witness_normals_link=witness_normals,
        triangle_indices=triangle_indices,
        witness_indices=witness_indices,
        barycentric_coordinates=barycentric,
        surface_centroid_link_m=surface_centroid,
        maximum_point_radius_link_m=maximum_radius,
        full_pad_sphere_center_link_m=sphere_center,
        full_pad_sphere_radius_upper_m=sphere_radius_upper,
        surface_sphere_nodes=tuple(
            node for node in surface_sphere_nodes if node is not None
        ),
        surface_sphere_root=surface_sphere_root,
        surface_aabb_nodes=frozen_surface_aabb_nodes,
        surface_aabb_root=surface_aabb_root,
        surface_aabb_leaf_node_indices=surface_aabb_leaf_node_indices,
        surface_aabb_leaf_triangle_indices=(
            surface_aabb_leaf_triangle_indices
        ),
        surface_aabb_left_child_indices=(
            surface_aabb_left_child_indices
        ),
        surface_aabb_right_child_indices=(
            surface_aabb_right_child_indices
        ),
        surface_aabb_internal_nodes_by_reverse_depth=(
            surface_aabb_internal_nodes_by_reverse_depth
        ),
        surface_triangles_link_m=triangles,
        relevant_joint_indices=relevant_joint_indices,
        witness_hierarchy_nodes=tuple(
            node for node in hierarchy_nodes if node is not None
        ),
        witness_hierarchy_root=hierarchy_root,
    )


def _joint_source(hand_model: ThreeFingerHandModel, joint_name: str) -> str:
    active: set[str] = set()
    cursor = joint_name
    while True:
        if cursor in active:
            raise RayClosureError("cyclic mimic relation in hand model")
        active.add(cursor)
        joint = hand_model.joints[cursor]
        if joint.mimic is None:
            return cursor
        cursor = joint.mimic.source_joint


def _finger_independent_sources(
    hand_model: ThreeFingerHandModel, finger_name: str
) -> frozenset[str]:
    sources = {
        _joint_source(hand_model, joint_name)
        for joint_name in hand_model.fingers[finger_name].joint_names
        if hand_model.joints[joint_name].movable
    }
    return frozenset(sources)


def _link_independent_source_indices(
    hand_model: ThreeFingerHandModel,
    link_name: str,
) -> tuple[int, ...]:
    """Return exactly the independent coordinates affecting ``link_name``."""

    by_child = {
        joint.child_link: name for name, joint in hand_model.joints.items()
    }
    sources: set[str] = set()
    cursor = link_name
    active_links: set[str] = set()
    while cursor != hand_model.base_link:
        if cursor in active_links:
            raise RayClosureError("cyclic link ancestry in hand model")
        active_links.add(cursor)
        joint_name = by_child.get(cursor)
        if joint_name is None:
            raise RayClosureError(f"PAD link {link_name} is disconnected from hand base")
        joint = hand_model.joints[joint_name]
        if joint.movable:
            sources.add(_joint_source(hand_model, joint_name))
        cursor = joint.parent_link
    independent_names = tuple(hand_model.independent_joint_names)
    return tuple(
        index for index, name in enumerate(independent_names) if name in sources
    )


class RayClosureSurfaceModel:
    """ObjectSurfaceModel-compatible sequential finite-PAD closure predictor."""

    method_id: ClassVar[str] = METHOD_ID
    closure_parameter_domain_id: ClassVar[str] = CLOSURE_PARAMETER_DOMAIN_ID

    @staticmethod
    def _object_triangle_affine_form_bounds_v9(
        triangles_m: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vectorise outward plane/edge coefficients for every object face."""

        triangles = np.asarray(triangles_m, dtype=np.float64)
        if (
            triangles.ndim != 3
            or triangles.shape[1:] != (3, 3)
            or len(triangles) == 0
            or not np.all(np.isfinite(triangles))
        ):
            raise RayClosureError(
                "object triangle coefficient input must have shape (F, 3, 3)"
            )

        def subtract(
            first_lower: np.ndarray,
            first_upper: np.ndarray,
            second_lower: np.ndarray,
            second_upper: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
            return (
                np.nextafter(first_lower - second_upper, -math.inf),
                np.nextafter(first_upper - second_lower, math.inf),
            )

        def multiply(
            first_lower: np.ndarray,
            first_upper: np.ndarray,
            second_lower: np.ndarray,
            second_upper: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
            products = np.stack(
                (
                    first_lower * second_lower,
                    first_lower * second_upper,
                    first_upper * second_lower,
                    first_upper * second_upper,
                ),
                axis=0,
            )
            return (
                np.nextafter(np.min(products, axis=0), -math.inf),
                np.nextafter(np.max(products, axis=0), math.inf),
            )

        def cross(
            first_lower: np.ndarray,
            first_upper: np.ndarray,
            second_lower: np.ndarray,
            second_upper: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
            lower_rows: list[np.ndarray] = []
            upper_rows: list[np.ndarray] = []
            for first_index, second_index in ((1, 2), (2, 0), (0, 1)):
                positive = multiply(
                    first_lower[:, first_index],
                    first_upper[:, first_index],
                    second_lower[:, second_index],
                    second_upper[:, second_index],
                )
                negative = multiply(
                    first_lower[:, second_index],
                    first_upper[:, second_index],
                    second_lower[:, first_index],
                    second_upper[:, first_index],
                )
                component = subtract(
                    positive[0], positive[1], negative[0], negative[1]
                )
                lower_rows.append(component[0])
                upper_rows.append(component[1])
            return np.stack(lower_rows, axis=1), np.stack(upper_rows, axis=1)

        def dot(
            first_lower: np.ndarray,
            first_upper: np.ndarray,
            second_lower: np.ndarray,
            second_upper: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
            result_lower, result_upper = multiply(
                first_lower[:, 0],
                first_upper[:, 0],
                second_lower[:, 0],
                second_upper[:, 0],
            )
            for column in (1, 2):
                product_lower, product_upper = multiply(
                    first_lower[:, column],
                    first_upper[:, column],
                    second_lower[:, column],
                    second_upper[:, column],
                )
                result_lower = np.nextafter(
                    result_lower + product_lower, -math.inf
                )
                result_upper = np.nextafter(
                    result_upper + product_upper, math.inf
                )
            return result_lower, result_upper

        edge_one = subtract(
            triangles[:, 1], triangles[:, 1],
            triangles[:, 0], triangles[:, 0],
        )
        edge_two = subtract(
            triangles[:, 2], triangles[:, 2],
            triangles[:, 0], triangles[:, 0],
        )
        area = cross(edge_one[0], edge_one[1], edge_two[0], edge_two[1])
        lower = np.empty((len(triangles), 4, 4), dtype=np.float64)
        upper = np.empty_like(lower)

        def store_row(
            row_index: int,
            coefficients: tuple[np.ndarray, np.ndarray],
            origin: np.ndarray,
        ) -> None:
            lower[:, row_index, :3] = coefficients[0]
            upper[:, row_index, :3] = coefficients[1]
            offset_lower, offset_upper = dot(
                coefficients[0], coefficients[1], origin, origin
            )
            lower[:, row_index, 3] = -offset_upper
            upper[:, row_index, 3] = -offset_lower

        store_row(0, area, triangles[:, 0])
        for edge_index in range(3):
            following = (edge_index + 1) % 3
            edge = subtract(
                triangles[:, following], triangles[:, following],
                triangles[:, edge_index], triangles[:, edge_index],
            )
            coefficients = cross(
                area[0], area[1], edge[0], edge[1]
            )
            store_row(
                edge_index + 1,
                coefficients,
                triangles[:, edge_index],
            )
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise RayClosureError(
                "object triangle affine coefficients are non-finite"
            )
        lower.setflags(write=False)
        upper.setflags(write=False)
        return lower, upper

    def __init__(
        self,
        *,
        object_model: ObjectGraspModel,
        hand_model: ThreeFingerHandModel,
        verified_pads: Sequence[VerifiedPad],
        task_frame: PreRegisteredTaskFrame,
        closing_actuation_directions_unit: Sequence[Sequence[float]],
        object_contact_normal_policy: str,
        pad_surface_normal_policy: str,
        maximum_subdivision_intervals: int,
        interval_decimal_precision: int,
        maximum_root_bisection_iterations: int,
    ) -> None:
        if not isinstance(object_model, ObjectGraspModel):
            raise RayClosureError("object_model must be an ObjectGraspModel")
        if not isinstance(hand_model, ThreeFingerHandModel):
            raise RayClosureError("hand_model must be a ThreeFingerHandModel")
        if object_contact_normal_policy != OBJECT_CONTACT_NORMAL_POLICY:
            raise RayClosureError(
                "object contact-normal policy differs from the hand contract"
            )
        if pad_surface_normal_policy != PAD_SURFACE_NORMAL_POLICY:
            raise RayClosureError(
                "PAD surface-normal policy differs from the hand contract"
            )
        if (
            not isinstance(maximum_subdivision_intervals, int)
            or isinstance(maximum_subdivision_intervals, bool)
            or maximum_subdivision_intervals <= 0
        ):
            raise RayClosureError(
                "maximum_subdivision_intervals must be an explicit positive integer budget"
            )
        pads = tuple(sorted(verified_pads, key=lambda pad: (pad.finger_name, pad.name)))
        if len(pads) != 3:
            raise RayClosureError("sequential closure requires exactly three verified PADs")
        if len({pad.name for pad in pads}) != 3 or len({pad.finger_name for pad in pads}) != 3:
            raise RayClosureError("verified PAD names and finger assignments must be unique")
        if {pad.name for pad in pads} != set(hand_model.pads):
            raise RayClosureError("verified PADs must match the three hand-model PADs")
        for pad in pads:
            hand_pad = hand_model.pads[pad.name]
            if (
                pad.link_name != hand_pad.link_name
                or pad.finger_name != hand_pad.finger_name
                or pad.coordinate_frame != pad.link_name
            ):
                raise RayClosureError(
                    f"verified PAD {pad.name} is not expressed in its hand terminal link"
                )

        lower, upper = hand_model.joint_limit_vectors()
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise RayClosureError("normalized joint parameters require finite URDF limits")
        spans = upper - lower
        if np.any(spans < 0.0):
            raise RayClosureError("hand joint limit spans cannot be negative")
        directions = np.asarray(closing_actuation_directions_unit, dtype=np.float64)
        expected_shape = (3, len(hand_model.independent_joint_names))
        if directions.shape != expected_shape or not np.all(np.isfinite(directions)):
            raise RayClosureError(
                f"closing_actuation_directions_unit must have shape {expected_shape}"
            )

        source_sets = {
            pad.finger_name: _finger_independent_sources(hand_model, pad.finger_name)
            for pad in pads
        }
        all_fingers = tuple(source_sets)
        exclusive: dict[str, frozenset[str]] = {}
        for finger in all_fingers:
            others = set().union(
                *(source_sets[other] for other in all_fingers if other != finger)
            )
            exclusive[finger] = frozenset(source_sets[finger] - others)

        canonical_rows: list[np.ndarray] = []
        support_rows: list[tuple[str, ...]] = []
        support_indices: list[int] = []
        used_support: set[int] = set()
        independent_names = tuple(hand_model.independent_joint_names)
        for row_index, (pad, row) in enumerate(zip(pads, directions)):
            support = set(int(index) for index in np.flatnonzero(row != 0.0))
            if not support:
                raise RayClosureError(
                    f"closing actuation row {row_index} for {pad.name} is empty"
                )
            if len(support) != 1:
                raise RayClosureError(
                    "V1 suffix-dominance requires exactly one independent "
                    f"closure joint for {pad.name}"
                )
            allowed_indices = {
                independent_names.index(name) for name in exclusive[pad.finger_name]
            }
            if not support <= allowed_indices:
                raise RayClosureError(
                    f"closing actuation for {pad.name} uses a shared or foreign joint"
                )
            if used_support & support:
                raise RayClosureError("finger closing actuation supports overlap")
            used_support |= support
            maximum = float(np.max(np.abs(row)))
            canonical = np.asarray(row / maximum, dtype=np.float64)
            physical = canonical * spans
            if not np.any(physical != 0.0):
                raise RayClosureError(
                    f"closing actuation for {pad.name} has zero physical joint span"
                )
            canonical.setflags(write=False)
            canonical_rows.append(canonical)
            support_rows.append(tuple(independent_names[index] for index in sorted(support)))
            support_indices.append(next(iter(support)))

        preshape_indices = tuple(
            index
            for index in range(len(independent_names))
            if index not in used_support and spans[index] > 0.0
        )
        open_joint_template = np.array(lower, copy=True)
        for row, support_index in zip(canonical_rows, support_indices):
            open_joint_template[support_index] = (
                lower[support_index]
                if row[support_index] > 0.0
                else upper[support_index]
            )

        self.object_model = object_model
        self._contact_face_mask = np.array(
            object_model.contact_face_mask, dtype=bool, copy=True
        )
        self.hand_model = hand_model
        self.object_contact_normal_policy = object_contact_normal_policy
        self.pad_surface_normal_policy = pad_surface_normal_policy
        self.task_frame = task_frame
        self.task_basis_object = task_frame.basis(object_model)
        try:
            canonical_task_frame = RegisteredTaskFrame(
                origin_object_m=object_model.assembly_axis_origin_m,
                basis_object=self.task_basis_object,
                source=task_frame.source,
            )
            self.canonical_object_face_vertices_m = (
                canonicalize_unoriented_triangles(
                object_model.mesh.face_vertices_m,
                    task_frame=canonical_task_frame,
                )
            )
        except TriangleCanonicalizationError as error:
            raise RayClosureError(
                "registered object triangles cannot be canonicalised"
            ) from error
        relative_vertices = (
            object_model.mesh.vertices_m
            - object_model.assembly_axis_origin_m
        )
        object_coordinates = relative_vertices @ self.task_basis_object
        self.object_coordinate_lower_m = np.min(object_coordinates, axis=0)
        self.object_coordinate_upper_m = np.max(object_coordinates, axis=0)
        self.verified_pads = pads
        self.prepared_pads = tuple(
            _prepare_pad(
                pad,
                relevant_joint_indices=_link_independent_source_indices(
                    hand_model, pad.link_name
                ),
            )
            for pad in pads
        )
        self.lower_joint_limits = np.array(lower, copy=True)
        self.upper_joint_limits = np.array(upper, copy=True)
        self.joint_spans = np.array(spans, copy=True)
        self.closing_directions_unit = np.vstack(canonical_rows)
        self.closing_directions_physical = self.closing_directions_unit * self.joint_spans
        self.independent_actuation_supports = tuple(support_rows)
        self.closure_support_indices = tuple(support_indices)
        self.preshape_joint_indices = preshape_indices
        self.preshape_joint_names = tuple(
            independent_names[index] for index in preshape_indices
        )
        self.open_joint_template = open_joint_template
        self.closure_open_joint_positions_rad = tuple(
            float(open_joint_template[index]) for index in support_indices
        )
        self.maximum_subdivision_intervals = maximum_subdivision_intervals
        self.interval_arithmetic_options = IntervalArithmeticOptions(
            decimal_precision=interval_decimal_precision,
            maximum_root_bisection_iterations=(
                maximum_root_bisection_iterations
            ),
        )
        self.interval_kinematics = DirectedIntervalKinematics(
            hand_model,
            self.interval_arithmetic_options,
        )
        (
            self._object_contact_affine_lower,
            self._object_contact_affine_upper,
        ) = self._object_triangle_affine_form_bounds_v9(
            self.canonical_object_face_vertices_m
        )
        self._exact_plane_key_cache: dict[
            int, tuple[int, int, int, int]
        ] = {}
        self._fast_plane_bucket_key_cache: dict[int, bytes] = {}
        self.intersector = TriangleFirstHitIntersector(object_model)
        self.distance_bvh = _PointTriangleDistanceBvh(object_model)
        self.parameter_layout = PARAMETER_LAYOUT_PREFIX + tuple(
            f"preshape_joint_unit:{name}" for name in self.preshape_joint_names
        )
        for array in (
            self.lower_joint_limits,
            self.upper_joint_limits,
            self.joint_spans,
            self.closing_directions_unit,
            self.closing_directions_physical,
            self.open_joint_template,
            self.object_coordinate_lower_m,
            self.object_coordinate_upper_m,
            self._contact_face_mask,
            self._object_contact_affine_lower,
            self._object_contact_affine_upper,
        ):
            array.setflags(write=False)
        pad_geometry_sha256 = tuple(pad.mesh.sha256 for pad in pads)
        pad_runtime_geometry_sha256 = tuple(
            _pad_runtime_geometry_sha256(pad) for pad in pads
        )
        for label, values in (
            ("object geometry", (object_model.geometry_sha256,)),
            ("PAD source geometry", pad_geometry_sha256),
            ("PAD runtime geometry", pad_runtime_geometry_sha256),
        ):
            if any(
                not isinstance(value, str)
                or _HEX_SHA256.fullmatch(value) is None
                for value in values
            ):
                raise RayClosureError(
                    f"{label} SHA-256 evidence is not canonical"
                )
        document = _model_contract_document(
            object_model=object_model,
            hand_model=hand_model,
            pads=pads,
            task_frame=task_frame,
            task_basis_object=self.task_basis_object,
            closing_directions_unit=self.closing_directions_unit,
            closing_directions_physical=self.closing_directions_physical,
            independent_actuation_supports=(
                self.independent_actuation_supports
            ),
            parameter_layout=self.parameter_layout,
            object_contact_normal_policy=object_contact_normal_policy,
            pad_surface_normal_policy=pad_surface_normal_policy,
            maximum_subdivision_intervals=maximum_subdivision_intervals,
            interval_options=self.interval_arithmetic_options,
        )
        canonical_json = _canonical_json(document)
        self.model_binding_complete = True
        self.model_binding_status = MODEL_BINDING_COMPLETE_STATUS
        self.object_geometry_sha256 = object_model.geometry_sha256
        self.pad_geometry_sha256 = pad_geometry_sha256
        self.pad_runtime_geometry_sha256 = pad_runtime_geometry_sha256
        self.pad_link_names = tuple(pad.link_name for pad in pads)
        self.closing_directions_physical_tuple = tuple(
            tuple(float(value) for value in row)
            for row in self.closing_directions_physical
        )
        self.model_contract_canonical_json = canonical_json
        self.model_contract_sha256 = hashlib.sha256(
            canonical_json.encode("utf-8")
        ).hexdigest()
        self._hand_model_contract_canonical_json = _canonical_json(
            _hand_model_manifest(hand_model)
        )

    @property
    def parameter_dimension(self) -> int:
        return len(self.parameter_layout)

    def _validate_hand(self, hand_model: ThreeFingerHandModel) -> None:
        if tuple(hand_model.independent_joint_names) != tuple(
            self.hand_model.independent_joint_names
        ):
            raise RayClosureError("optimizer supplied a different independent-joint order")
        if set(hand_model.pads) != set(self.hand_model.pads):
            raise RayClosureError("optimizer supplied a different PAD mapping")
        lower, upper = hand_model.joint_limit_vectors()
        if not np.array_equal(lower, self.lower_joint_limits) or not np.array_equal(
            upper, self.upper_joint_limits
        ):
            raise RayClosureError("optimizer supplied different hand joint limits")
        if _canonical_json(_hand_model_manifest(hand_model)) != (
            self._hand_model_contract_canonical_json
        ):
            raise RayClosureError(
                "optimizer supplied a different complete hand model contract"
            )

    def _placement_coordinate_bounds(
        self,
        q_start: np.ndarray,
        rotation_object_from_hand: np.ndarray,
        *,
        focus_result: tuple[np.ndarray, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return translations where every swept PAD AABB can overlap CAD.

        For object interval ``[o-, o+]`` and a certified outer interval
        ``[p-, p+]`` of one PAD over its complete registered closure path,
        overlap is possible only when the focus coordinate is in
        ``[o- - p+, o+ - p-]``.  Intersecting those intervals over all three
        PADs therefore gives a necessary geometry-derived chart domain.  It
        is not a clearance or collision certificate.
        """

        resolved_focus = (
            self._closure_focus_hand(q_start)
            if focus_result is None
            else focus_result
        )
        if resolved_focus is None:
            raise RayClosureError(
                "placement domain has no finite full-closed PAD focus"
            )
        focus_hand, _hand_extent = resolved_focus
        task_from_hand = (
            self.task_basis_object.T @ rotation_object_from_hand
        )
        lower_rows: list[np.ndarray] = []
        upper_rows: list[np.ndarray] = []
        for row_index, prepared in enumerate(self.prepared_pads):
            direction = self.closing_directions_physical[row_index]
            maximum_parameter = self._maximum_path_parameter(
                q_start, direction
            )
            if maximum_parameter <= 0.0:
                raise RayClosureError(
                    f"closure path for {prepared.verified.name} is empty"
                )
            midpoint_parameter = 0.5 * maximum_parameter
            q_midpoint = q_start + midpoint_parameter * direction
            links = self.hand_model.forward_kinematics(q_midpoint)
            transform = links[prepared.verified.link_name]
            points_hand = (
                prepared.verified.points_local_m @ transform[:3, :3].T
                + transform[:3, 3]
            )
            offsets_task = (
                points_hand - focus_hand
            ) @ task_from_hand.T
            speed_bounds = self._local_point_speed_bounds(
                prepared,
                prepared.verified.points_local_m,
                q_start,
                direction,
                maximum_parameter,
            )
            coordinate_scale = (
                np.max(np.abs(points_hand), axis=1)
                + float(np.linalg.norm(focus_hand, ord=np.inf))
                + self.intersector.characteristic_length_m
            )
            forward_error = _FK_ERROR * coordinate_scale
            enclosure_radii = np.nextafter(
                speed_bounds * midpoint_parameter + forward_error,
                math.inf,
            )
            pad_lower = np.min(
                offsets_task - enclosure_radii[:, None], axis=0
            )
            pad_upper = np.max(
                offsets_task + enclosure_radii[:, None], axis=0
            )
            lower_rows.append(self.object_coordinate_lower_m - pad_upper)
            upper_rows.append(self.object_coordinate_upper_m - pad_lower)
        lower = np.max(np.vstack(lower_rows), axis=0)
        upper = np.min(np.vstack(upper_rows), axis=0)
        if np.any(lower > upper):
            raise RayClosureError(
                "full-closed PAD/object AABB overlap domains have empty intersection"
            )
        lower.setflags(write=False)
        upper.setflags(write=False)
        return lower, upper

    def _decode_with_object_from_hand(
        self, parameters_unit: Sequence[float]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        parameters = np.asarray(parameters_unit, dtype=np.float64)
        if parameters.shape != (self.parameter_dimension,) or not np.all(
            np.isfinite(parameters)
        ):
            raise RayClosureError(
                f"unit parameters must have shape ({self.parameter_dimension},)"
            )
        if np.any(parameters < 0.0) or np.any(parameters > 1.0):
            raise RayClosureError("unit parameters must lie within [0, 1]")
        if parameters[0] >= 1.0:
            raise RayClosureError(
                "assembly-axis yaw uses the canonical half-open unit interval [0, 1)"
            )
        yaw = 2.0 * math.pi * float(parameters[0])
        joints = np.array(self.open_joint_template, copy=True)
        for unit_value, joint_index in zip(
            parameters[4:], self.preshape_joint_indices
        ):
            joints[joint_index] = (
                self.lower_joint_limits[joint_index]
                + float(unit_value) * self.joint_spans[joint_index]
            )
        rotation_about_axis = np.asarray(
            (
                (math.cos(yaw), -math.sin(yaw), 0.0),
                (math.sin(yaw), math.cos(yaw), 0.0),
                (0.0, 0.0, 1.0),
            ),
            dtype=np.float64,
        )
        rotation = self.task_basis_object @ rotation_about_axis
        focus_result = self._closure_focus_hand(joints)
        if focus_result is None:
            raise RayClosureError(
                "placement domain has no finite full-closed PAD focus"
            )
        lower, upper = self._placement_coordinate_bounds(
            joints,
            rotation,
            focus_result=focus_result,
        )
        placement_units = parameters[[2, 3, 1]]
        zero_width = lower == upper
        if np.any(zero_width & (placement_units != 0.0)):
            raise RayClosureError(
                "zero-width placement axes require canonical unit coordinate 0"
            )
        target_coordinates = np.asarray(
            (
                lower[0] + float(parameters[2]) * (upper[0] - lower[0]),
                lower[1] + float(parameters[3]) * (upper[1] - lower[1]),
                lower[2] + float(parameters[1]) * (upper[2] - lower[2]),
            ),
            dtype=np.float64,
        )
        target = (
            self.object_model.assembly_axis_origin_m
            + self.task_basis_object @ target_coordinates
        )
        focus_hand, hand_extent = focus_result
        object_from_hand = np.eye(4, dtype=np.float64)
        object_from_hand[:3, :3] = rotation
        object_from_hand[:3, 3] = target - rotation @ focus_hand
        return joints, target, rotation, object_from_hand, hand_extent

    def _decode(
        self, parameters_unit: Sequence[float]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        joints, target, rotation, _object_from_hand, _hand_extent = (
            self._decode_with_object_from_hand(parameters_unit)
        )
        return joints, target, rotation

    def _closure_focus_hand(
        self, q_start: np.ndarray
    ) -> tuple[np.ndarray, float] | None:
        # The placement reference is a coordinate-chart choice, not a grasp
        # score.  Evaluate it at the mechanically registered full-closed
        # endpoint so it lies on the actual nonlinear joint path.  Each PAD
        # contributes its complete triangle-area centroid with equal finger
        # weight, making the reference invariant to non-uniform remeshing.
        q_closed = np.array(q_start, copy=True)
        for direction in self.closing_directions_physical:
            q_closed += direction
        self.hand_model.resolve_joint_positions(q_closed)
        links = self.hand_model.forward_kinematics(q_closed)
        centroids: list[np.ndarray] = []
        maximum_extent = 0.0
        for prepared in self.prepared_pads:
            pad = prepared.verified
            transform = links[pad.link_name]
            points = (
                pad.points_local_m @ transform[:3, :3].T
                + transform[:3, 3]
            )
            centroid = (
                transform[:3, :3] @ prepared.surface_centroid_link_m
                + transform[:3, 3]
            )
            centroids.append(centroid)
            maximum_extent = max(
                maximum_extent,
                float(np.max(np.linalg.norm(points, axis=1))),
            )
        focus = np.mean(np.vstack(centroids), axis=0)
        if not np.all(np.isfinite(focus)) or not math.isfinite(maximum_extent):
            return None
        return focus, maximum_extent

    def _object_from_hand(
        self, q_start: np.ndarray, target_object_m: np.ndarray, rotation: np.ndarray
    ) -> tuple[np.ndarray, float] | None:
        focus_result = self._closure_focus_hand(q_start)
        if focus_result is None:
            return None
        focus_hand, hand_extent = focus_result
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        transform[:3, 3] = target_object_m - rotation @ focus_hand
        return transform, hand_extent

    def _maximum_path_parameter(self, q_start: np.ndarray, direction: np.ndarray) -> float:
        values: list[float] = []
        for position, lower, upper, rate in zip(
            q_start,
            self.lower_joint_limits,
            self.upper_joint_limits,
            direction,
        ):
            if rate > 0.0:
                values.append(float((upper - position) / rate))
            elif rate < 0.0:
                values.append(float((lower - position) / rate))
        if not values:
            return 0.0
        result = min(values)
        error = _TIME_ERROR * max(abs(result), *(abs(value) for value in values))
        if result < -error:
            return 0.0
        return max(0.0, result)

    def _whole_path_pad_sphere_coverages(
        self,
        *,
        q_start: np.ndarray,
        object_from_hand: np.ndarray,
        spatial_error_bound_m: float,
    ) -> tuple[_WholePathPadSphereCoverage, ...]:
        """Cover every complete PAD path by eight conservative spheres.

        Each local sphere contains every source PAD vertex and therefore every
        point of every PAD triangle.  The midpoint sphere for one time segment
        is enlarged by a global vertex-speed upper bound times the segment
        half-width, so it also contains that complete PAD throughout the
        segment.  No result from this helper can certify contact; it can only
        certify strict separation from the object surface.
        """

        if (
            np.asarray(q_start, dtype=np.float64).shape
            != (len(self.hand_model.independent_joint_names),)
            or np.asarray(object_from_hand, dtype=np.float64).shape != (4, 4)
            or not math.isfinite(spatial_error_bound_m)
            or spatial_error_bound_m < 0.0
        ):
            raise RayClosureError(
                "whole-path sphere coverage inputs are malformed"
            )
        rows: list[_WholePathPadSphereCoverage] = []
        for pad_index, prepared in enumerate(self.prepared_pads):
            direction = self.closing_directions_physical[pad_index]
            maximum_parameter = self._maximum_path_parameter(
                q_start, direction
            )
            vertex_speed_bounds = self._local_point_speed_bounds(
                prepared,
                prepared.verified.points_local_m,
                q_start,
                direction,
                maximum_parameter,
            )
            maximum_vertex_speed = float(np.max(vertex_speed_bounds))
            segment_width = (
                maximum_parameter / _WHOLE_PATH_SPHERE_SEGMENT_COUNT
            )
            half_width = 0.5 * segment_width
            expanded_radius = float(
                np.nextafter(
                    prepared.full_pad_sphere_radius_upper_m
                    + maximum_vertex_speed * half_width,
                    math.inf,
                )
            )
            centers = np.empty(
                (_WHOLE_PATH_SPHERE_SEGMENT_COUNT, 3),
                dtype=np.float64,
            )
            for segment_index in range(_WHOLE_PATH_SPHERE_SEGMENT_COUNT):
                midpoint = (segment_index + 0.5) * segment_width
                links = self.hand_model.forward_kinematics(
                    q_start + midpoint * direction,
                    base_transform=object_from_hand,
                )
                transform = links[prepared.verified.link_name]
                centers[segment_index] = (
                    transform[:3, :3]
                    @ prepared.full_pad_sphere_center_link_m
                    + transform[:3, 3]
                )
            radii = np.full(
                _WHOLE_PATH_SPHERE_SEGMENT_COUNT,
                expanded_radius,
                dtype=np.float64,
            )
            centers.setflags(write=False)
            radii.setflags(write=False)
            rows.append(
                _WholePathPadSphereCoverage(
                    prepared_pad_index=pad_index,
                    centers_object_m=centers,
                    radius_upper_m=radii,
                    spatial_error_bound_m=spatial_error_bound_m,
                )
            )
        return tuple(rows)

    def _classify_whole_path_pad_sphere_coverages(
        self,
        coverages: Sequence[_WholePathPadSphereCoverage],
    ) -> tuple[WholePathPadSphereScreen, ...]:
        """Classify many PAD covers with one exact surface-distance call."""

        rows = tuple(coverages)
        if not rows:
            return ()
        centers = np.vstack([row.centers_object_m for row in rows])
        radii = np.concatenate([row.radius_upper_m for row in rows])
        spatial_errors = np.concatenate(
            [
                np.full(
                    len(row.radius_upper_m),
                    row.spatial_error_bound_m,
                    dtype=np.float64,
                )
                for row in rows
            ]
        )
        nearest = self.distance_bvh.nearest_many(centers)
        clearances = np.nextafter(
            nearest.distances_m - radii - spatial_errors,
            -math.inf,
        )
        result: list[WholePathPadSphereScreen] = []
        offset = 0
        for row in rows:
            count = len(row.radius_upper_m)
            selected = slice(offset, offset + count)
            minimum_clearance = float(np.min(clearances[selected]))
            prepared = self.prepared_pads[row.prepared_pad_index]
            result.append(
                WholePathPadSphereScreen(
                    pad_name=prepared.verified.name,
                    finger_name=prepared.verified.finger_name,
                    segment_count=count,
                    nearest_surface_query_count=count,
                    distance_bvh_node_visits=int(
                        np.sum(nearest.node_visits[selected])
                    ),
                    distance_triangle_tests=int(
                        np.sum(nearest.triangle_tests[selected])
                    ),
                    minimum_clearance_lower_bound_m=minimum_clearance,
                    certified_free=minimum_clearance > 0.0,
                )
            )
            offset += count
        return tuple(result)

    def _whole_path_pad_sphere_hierarchy_coverages(
        self,
        *,
        q_start: np.ndarray,
        object_from_hand: np.ndarray,
        spatial_error_bound_m: float,
    ) -> tuple[_WholePathPadSphereHierarchyCoverage, ...]:
        """Prepare eight transforms and all node radii without world centres."""

        if (
            np.asarray(q_start, dtype=np.float64).shape
            != (len(self.hand_model.independent_joint_names),)
            or np.asarray(object_from_hand, dtype=np.float64).shape != (4, 4)
            or not math.isfinite(spatial_error_bound_m)
            or spatial_error_bound_m < 0.0
        ):
            raise RayClosureError(
                "whole-path sphere hierarchy inputs are malformed"
            )
        rows: list[_WholePathPadSphereHierarchyCoverage] = []
        for pad_index, prepared in enumerate(self.prepared_pads):
            direction = self.closing_directions_physical[pad_index]
            maximum_parameter = self._maximum_path_parameter(
                q_start, direction
            )
            segment_width = (
                maximum_parameter / _WHOLE_PATH_SPHERE_SEGMENT_COUNT
            )
            half_width = 0.5 * segment_width
            rotations = np.empty(
                (_WHOLE_PATH_SPHERE_SEGMENT_COUNT, 3, 3),
                dtype=np.float64,
            )
            translations = np.empty(
                (_WHOLE_PATH_SPHERE_SEGMENT_COUNT, 3),
                dtype=np.float64,
            )
            for segment_index in range(_WHOLE_PATH_SPHERE_SEGMENT_COUNT):
                midpoint = (segment_index + 0.5) * segment_width
                links = self.hand_model.forward_kinematics(
                    q_start + midpoint * direction,
                    base_transform=object_from_hand,
                )
                transform = links[prepared.verified.link_name]
                rotations[segment_index] = transform[:3, :3]
                translations[segment_index] = transform[:3, 3]
            nodes = prepared.surface_sphere_nodes
            maximum_vertex_speeds = self._local_radius_speed_bounds(
                prepared=prepared,
                local_radii_m=tuple(
                    node.maximum_vertex_radius_link_m for node in nodes
                ),
                q_start=q_start,
                direction=direction,
                maximum_parameter=maximum_parameter,
            )
            node_radii = np.nextafter(
                np.asarray(
                    [node.radius_upper_m for node in nodes],
                    dtype=np.float64,
                )
                + maximum_vertex_speeds * half_width,
                math.inf,
            )
            node_box_half_extents = np.nextafter(
                np.vstack(
                    [node.box_half_extents_upper_m for node in nodes]
                )
                + maximum_vertex_speeds[:, None] * half_width,
                math.inf,
            )
            rotations.setflags(write=False)
            translations.setflags(write=False)
            node_radii.setflags(write=False)
            node_box_half_extents.setflags(write=False)
            rows.append(
                _WholePathPadSphereHierarchyCoverage(
                    prepared_pad_index=pad_index,
                    rotations_object_from_link=rotations,
                    translations_object_m=translations,
                    node_radius_upper_m=node_radii,
                    node_box_half_extents_upper_m=(
                        node_box_half_extents
                    ),
                    spatial_error_bound_m=spatial_error_bound_m,
                )
            )
        return tuple(rows)

    def _whole_path_pad_aabb_hierarchy_coverages(
        self,
        *,
        q_start: np.ndarray,
        object_from_hand: np.ndarray,
        spatial_error_bound_m: float,
    ) -> tuple[_WholePathPadAabbHierarchyCoverage, ...]:
        """Prepare full triangle-leaf PAD boxes for eight path segments."""

        if (
            np.asarray(q_start, dtype=np.float64).shape
            != (len(self.hand_model.independent_joint_names),)
            or np.asarray(object_from_hand, dtype=np.float64).shape != (4, 4)
            or not math.isfinite(spatial_error_bound_m)
            or spatial_error_bound_m < 0.0
        ):
            raise RayClosureError(
                "whole-path PAD AABB hierarchy inputs are malformed"
            )
        frozen_q_start = np.asarray(q_start, dtype=np.float64).copy()
        frozen_object_from_hand = np.asarray(
            object_from_hand, dtype=np.float64
        ).copy()
        frozen_q_start.setflags(write=False)
        frozen_object_from_hand.setflags(write=False)
        rows: list[_WholePathPadAabbHierarchyCoverage] = []
        for pad_index, prepared in enumerate(self.prepared_pads):
            direction = np.asarray(
                self.closing_directions_physical[pad_index],
                dtype=np.float64,
            ).copy()
            direction.setflags(write=False)
            maximum_parameter = self._maximum_path_parameter(
                q_start, direction
            )
            segment_width = (
                maximum_parameter / _WHOLE_PATH_SPHERE_SEGMENT_COUNT
            )
            half_width = 0.5 * segment_width
            rotations = np.empty(
                (_WHOLE_PATH_SPHERE_SEGMENT_COUNT, 3, 3),
                dtype=np.float64,
            )
            translations = np.empty(
                (_WHOLE_PATH_SPHERE_SEGMENT_COUNT, 3),
                dtype=np.float64,
            )
            for segment_index in range(_WHOLE_PATH_SPHERE_SEGMENT_COUNT):
                midpoint = (segment_index + 0.5) * segment_width
                links = self.hand_model.forward_kinematics(
                    q_start + midpoint * direction,
                    base_transform=object_from_hand,
                )
                transform = links[prepared.verified.link_name]
                rotations[segment_index] = transform[:3, :3]
                translations[segment_index] = transform[:3, 3]
            radius_slope, intercept = (
                self._local_radius_speed_affine_coefficients(
                    prepared=prepared,
                    q_start=q_start,
                    direction=direction,
                    maximum_parameter=maximum_parameter,
                )
            )
            rotations.setflags(write=False)
            translations.setflags(write=False)
            rows.append(
                _WholePathPadAabbHierarchyCoverage(
                    prepared_pad_index=pad_index,
                    rotations_object_from_link=rotations,
                    translations_object_m=translations,
                    motion_speed_radius_slope_upper_per_unit=(
                        radius_slope
                    ),
                    motion_speed_intercept_upper_m_per_unit=intercept,
                    segment_half_width_unit=half_width,
                    spatial_error_bound_m=spatial_error_bound_m,
                    q_start=frozen_q_start,
                    direction=direction,
                    maximum_parameter=maximum_parameter,
                    object_from_hand=frozen_object_from_hand,
                )
            )
        return tuple(rows)

    @staticmethod
    def _triangle_obb_strict_separation_mask(
        *,
        triangles_object_m: Sequence[Sequence[Sequence[float]]],
        box_center_object_m: Sequence[float],
        box_axes_object: Sequence[Sequence[float]],
        box_half_extents_m: Sequence[float],
    ) -> np.ndarray:
        """Prove triangle/OBB separation on the complete 13 SAT axes."""

        triangles = np.asarray(triangles_object_m, dtype=np.float64)
        center = np.asarray(box_center_object_m, dtype=np.float64)
        rotation = np.asarray(box_axes_object, dtype=np.float64)
        half_extents = np.asarray(box_half_extents_m, dtype=np.float64)
        if (
            triangles.ndim != 3
            or triangles.shape[1:] != (3, 3)
            or len(triangles) == 0
            or center.shape != (3,)
            or rotation.shape != (3, 3)
            or half_extents.shape != (3,)
            or not np.all(np.isfinite(triangles))
            or not np.all(np.isfinite(center))
            or not np.all(np.isfinite(rotation))
            or not np.all(np.isfinite(half_extents))
            or np.any(half_extents < 0.0)
        ):
            raise RayClosureError(
                "triangle/OBB SAT inputs must be finite ordered arrays"
            )
        local = (triangles - center[None, None, :]) @ rotation
        edges = np.stack(
            (
                local[:, 1] - local[:, 0],
                local[:, 2] - local[:, 1],
                local[:, 0] - local[:, 2],
            ),
            axis=1,
        )
        identity = np.eye(3, dtype=np.float64)
        box_axes = np.broadcast_to(
            identity[None, :, :], (len(triangles), 3, 3)
        )
        triangle_normals = np.cross(edges[:, 0], -edges[:, 2])
        cross_axes = np.cross(
            edges[:, :, None, :],
            identity[None, None, :, :],
        ).reshape((len(triangles), 9, 3))
        candidate_axes = np.concatenate(
            (
                box_axes,
                triangle_normals[:, None, :],
                cross_axes,
            ),
            axis=1,
        )
        projections = np.einsum(
            "fvc,fac->fva", local, candidate_axes
        )
        minimum = np.min(projections, axis=1)
        maximum = np.max(projections, axis=1)
        box_radius = np.einsum(
            "fac,c->fa", np.abs(candidate_axes), half_extents
        )
        axis_norm = np.linalg.norm(candidate_axes, axis=2)
        coordinate_scale = (
            np.max(np.abs(local), axis=(1, 2))
            + float(np.sum(half_extents))
            + 1.0
        )
        projection_error = _FK_ERROR * (
            np.abs(minimum)
            + np.abs(maximum)
            + box_radius
            + axis_norm * coordinate_scale[:, None]
        )
        positive_limit = np.nextafter(
            box_radius + projection_error, math.inf
        )
        negative_limit = np.nextafter(
            -box_radius - projection_error, -math.inf
        )
        valid_axis = axis_norm > np.finfo(np.float64).tiny
        separated_axes = valid_axis & (
            (minimum > positive_limit) | (maximum < negative_limit)
        )
        result = np.any(separated_axes, axis=1)
        result.setflags(write=False)
        return result

    @staticmethod
    def _triangle_obb_pair_strict_separation_mask(
        *,
        triangles_object_m: Sequence[Sequence[Sequence[float]]],
        box_centers_object_m: Sequence[Sequence[float]],
        box_axes_object: Sequence[Sequence[Sequence[float]]],
        box_half_extents_m: Sequence[Sequence[float]],
    ) -> np.ndarray:
        """Prove separation for N triangle/OBB pairs in one matrix call."""

        triangles = np.asarray(triangles_object_m, dtype=np.float64)
        centers = np.asarray(box_centers_object_m, dtype=np.float64)
        rotations = np.asarray(box_axes_object, dtype=np.float64)
        half_extents = np.asarray(box_half_extents_m, dtype=np.float64)
        count = len(triangles)
        if (
            triangles.ndim != 3
            or triangles.shape[1:] != (3, 3)
            or count == 0
            or centers.shape != (count, 3)
            or rotations.shape != (count, 3, 3)
            or half_extents.shape != (count, 3)
            or not np.all(np.isfinite(triangles))
            or not np.all(np.isfinite(centers))
            or not np.all(np.isfinite(rotations))
            or not np.all(np.isfinite(half_extents))
            or np.any(half_extents < 0.0)
        ):
            raise RayClosureError(
                "triangle/OBB pair SAT inputs must have aligned finite arrays"
            )
        local = np.einsum(
            "nvj,njk->nvk",
            triangles - centers[:, None, :],
            rotations,
        )
        edges = np.stack(
            (
                local[:, 1] - local[:, 0],
                local[:, 2] - local[:, 1],
                local[:, 0] - local[:, 2],
            ),
            axis=1,
        )
        identity = np.eye(3, dtype=np.float64)
        box_axes = np.broadcast_to(
            identity[None, :, :], (count, 3, 3)
        )
        triangle_normals = np.cross(edges[:, 0], -edges[:, 2])
        cross_axes = np.cross(
            edges[:, :, None, :],
            identity[None, None, :, :],
        ).reshape((count, 9, 3))
        candidate_axes = np.concatenate(
            (
                box_axes,
                triangle_normals[:, None, :],
                cross_axes,
            ),
            axis=1,
        )
        projections = np.einsum(
            "nvc,nac->nva", local, candidate_axes
        )
        minimum = np.min(projections, axis=1)
        maximum = np.max(projections, axis=1)
        box_radius = np.einsum(
            "nac,nc->na", np.abs(candidate_axes), half_extents
        )
        axis_norm = np.linalg.norm(candidate_axes, axis=2)
        coordinate_scale = (
            np.max(np.abs(local), axis=(1, 2))
            + np.sum(half_extents, axis=1)
            + 1.0
        )
        projection_error = _FK_ERROR * (
            np.abs(minimum)
            + np.abs(maximum)
            + box_radius
            + axis_norm * coordinate_scale[:, None]
        )
        positive_limit = np.nextafter(
            box_radius + projection_error, math.inf
        )
        negative_limit = np.nextafter(
            -box_radius - projection_error, -math.inf
        )
        valid_axis = axis_norm > np.finfo(np.float64).tiny
        separated_axes = valid_axis & (
            (minimum > positive_limit) | (maximum < negative_limit)
        )
        result = np.any(separated_axes, axis=1)
        result.setflags(write=False)
        return result

    def _classify_whole_path_pad_sphere_hierarchies(
        self,
        coverages: Sequence[_WholePathPadSphereHierarchyCoverage],
    ) -> tuple[WholePathPadSphereScreen, ...]:
        """Refine only uncertain swept PAD boxes to depth three."""

        rows = tuple(coverages)
        if not rows:
            return ()
        terminal_clearances: list[list[float]] = [
            [] for _row in rows
        ]
        uncertain_leaf = np.zeros(len(rows), dtype=bool)
        spatial_node_queries = np.zeros(len(rows), dtype=np.int64)
        aabb_free_counts = np.zeros(len(rows), dtype=np.int64)
        exact_query_counts = np.zeros(len(rows), dtype=np.int64)
        node_visits = np.zeros(len(rows), dtype=np.int64)
        triangle_tests = np.zeros(len(rows), dtype=np.int64)
        sat_free_counts = np.zeros(len(rows), dtype=np.int64)
        sat_triangle_tests = np.zeros(len(rows), dtype=np.int64)
        maximum_depth = np.zeros(len(rows), dtype=np.int64)
        active: list[tuple[int, int, int]] = []
        for coverage_index, row in enumerate(rows):
            root = self.prepared_pads[
                row.prepared_pad_index
            ].surface_sphere_root
            active.extend(
                (coverage_index, root, segment_index)
                for segment_index in range(
                    _WHOLE_PATH_SPHERE_SEGMENT_COUNT
                )
            )

        while active:
            centers = np.empty((len(active), 3), dtype=np.float64)
            node_radii = np.empty(len(active), dtype=np.float64)
            spatial_errors = np.empty(len(active), dtype=np.float64)
            box_axes = np.empty((len(active), 3, 3), dtype=np.float64)
            box_half_extents = np.empty((len(active), 3), dtype=np.float64)
            for active_offset, (
                coverage_index,
                node_index,
                segment_index,
            ) in enumerate(active):
                row = rows[coverage_index]
                prepared = self.prepared_pads[row.prepared_pad_index]
                node = prepared.surface_sphere_nodes[node_index]
                centers[active_offset] = (
                    row.rotations_object_from_link[segment_index]
                    @ node.center_link_m
                    + row.translations_object_m[segment_index]
                )
                box_axes[active_offset] = (
                    row.rotations_object_from_link[segment_index]
                )
                node_radii[active_offset] = row.node_radius_upper_m[
                    node_index
                ]
                spatial_errors[active_offset] = (
                    row.spatial_error_bound_m
                )
                box_scale = (
                    self.intersector.characteristic_length_m
                    + float(
                        np.linalg.norm(centers[active_offset], ord=np.inf)
                    )
                    + float(
                        np.sum(
                            row.node_box_half_extents_upper_m[node_index]
                        )
                    )
                )
                box_half_extents[active_offset] = np.nextafter(
                    row.node_box_half_extents_upper_m[node_index]
                    + row.spatial_error_bound_m
                    + _FK_ERROR * box_scale,
                    math.inf,
                )
            world_half_extents = np.nextafter(
                np.einsum(
                    "nij,nj->ni",
                    np.abs(box_axes),
                    box_half_extents,
                )
                * (1.0 + _FK_ERROR),
                math.inf,
            )
            face_rows = self.distance_bvh.face_indices_intersecting_aabbs(
                np.nextafter(
                    centers - world_half_extents, -math.inf
                ),
                np.nextafter(
                    centers + world_half_extents, math.inf
                ),
            )
            overlaps = np.asarray(
                [len(face_indices) > 0 for face_indices in face_rows],
                dtype=bool,
            )
            clear = ~overlaps
            clearances = np.zeros(len(active), dtype=np.float64)
            exact_offsets = np.flatnonzero(overlaps)
            exact_nearest: _NearestMany | None = None
            if len(exact_offsets) > 0:
                exact_nearest = self.distance_bvh.nearest_many(
                    centers[exact_offsets]
                )
                exact_clearances = np.nextafter(
                    exact_nearest.distances_m
                    - node_radii[exact_offsets]
                    - spatial_errors[exact_offsets],
                    -math.inf,
                )
                clearances[exact_offsets] = exact_clearances
                clear[exact_offsets] = exact_clearances > 0.0

            sat_offsets = np.flatnonzero(overlaps & ~clear)
            sat_free = np.zeros(len(active), dtype=bool)
            for sat_offset_value in sat_offsets:
                sat_offset = int(sat_offset_value)
                face_indices = face_rows[sat_offset]
                separated = self._triangle_obb_strict_separation_mask(
                    triangles_object_m=(
                        self.canonical_object_face_vertices_m[
                            face_indices
                        ]
                    ),
                    box_center_object_m=centers[sat_offset],
                    box_axes_object=box_axes[sat_offset],
                    box_half_extents_m=box_half_extents[sat_offset],
                )
                sat_triangle_tests[active[sat_offset][0]] += len(
                    face_indices
                )
                if bool(np.all(separated)):
                    clear[sat_offset] = True
                    sat_free[sat_offset] = True

            next_active: list[tuple[int, int, int]] = []
            exact_row_by_offset = {
                int(active_offset): exact_row
                for exact_row, active_offset in enumerate(exact_offsets)
            }
            for active_offset, (
                coverage_index,
                node_index,
                segment_index,
            ) in enumerate(active):
                prepared = self.prepared_pads[
                    rows[coverage_index].prepared_pad_index
                ]
                node = prepared.surface_sphere_nodes[node_index]
                spatial_node_queries[coverage_index] += 1
                maximum_depth[coverage_index] = max(
                    maximum_depth[coverage_index], node.depth
                )
                if not overlaps[active_offset]:
                    aabb_free_counts[coverage_index] += 1
                else:
                    exact_query_counts[coverage_index] += 1
                    if exact_nearest is None:  # pragma: no cover
                        raise RayClosureError(
                            "exact hierarchy row lost its nearest query"
                        )
                    exact_row = exact_row_by_offset[active_offset]
                    node_visits[coverage_index] += int(
                        exact_nearest.node_visits[exact_row]
                    )
                    triangle_tests[coverage_index] += int(
                        exact_nearest.triangle_tests[exact_row]
                    )
                if sat_free[active_offset]:
                    sat_free_counts[coverage_index] += 1
                if clear[active_offset]:
                    terminal_clearances[coverage_index].append(
                        max(0.0, float(clearances[active_offset]))
                    )
                    continue
                if node.leaf:
                    uncertain_leaf[coverage_index] = True
                    terminal_clearances[coverage_index].append(
                        min(0.0, float(clearances[active_offset]))
                    )
                    continue
                next_active.append(
                    (coverage_index, node.left, segment_index)
                )
                next_active.append(
                    (coverage_index, node.right, segment_index)
                )
            active = next_active

        result: list[WholePathPadSphereScreen] = []
        for coverage_index, row in enumerate(rows):
            prepared = self.prepared_pads[row.prepared_pad_index]
            if not terminal_clearances[coverage_index]:
                raise RayClosureError(
                    "PAD sphere hierarchy produced no terminal branch"
                )
            result.append(
                WholePathPadSphereScreen(
                    pad_name=prepared.verified.name,
                    finger_name=prepared.verified.finger_name,
                    segment_count=_WHOLE_PATH_SPHERE_SEGMENT_COUNT,
                    nearest_surface_query_count=int(
                        exact_query_counts[coverage_index]
                    ),
                    distance_bvh_node_visits=int(
                        node_visits[coverage_index]
                    ),
                    distance_triangle_tests=int(
                        triangle_tests[coverage_index]
                    ),
                    minimum_clearance_lower_bound_m=float(
                        min(terminal_clearances[coverage_index])
                    ),
                    certified_free=not bool(
                        uncertain_leaf[coverage_index]
                    ),
                    spatial_node_query_count=int(
                        spatial_node_queries[coverage_index]
                    ),
                    aabb_certified_free_node_count=int(
                        aabb_free_counts[coverage_index]
                    ),
                    exact_distance_query_count=int(
                        exact_query_counts[coverage_index]
                    ),
                    maximum_spatial_depth_reached=int(
                        maximum_depth[coverage_index]
                    ),
                    obb_sat_certified_free_node_count=int(
                        sat_free_counts[coverage_index]
                    ),
                    obb_sat_triangle_test_count=int(
                        sat_triangle_tests[coverage_index]
                    ),
                )
            )
        return tuple(result)

    def _pad_aabb_root_global_overlap_count(
        self,
        coverage: _WholePathPadAabbHierarchyCoverage,
    ) -> int:
        """Count root path boxes overlapping the complete object AABB."""

        prepared = self.prepared_pads[coverage.prepared_pad_index]
        node = prepared.surface_aabb_nodes[prepared.surface_aabb_root]
        motion_speed = float(
            np.nextafter(
                coverage.motion_speed_radius_slope_upper_per_unit
                * node.maximum_vertex_radius_link_m
                + coverage.motion_speed_intercept_upper_m_per_unit,
                math.inf,
            )
        )
        moving_half_extents = np.nextafter(
            node.box_half_extents_upper_m
            + motion_speed * coverage.segment_half_width_unit,
            math.inf,
        )
        centers = np.einsum(
            "nij,j->ni",
            coverage.rotations_object_from_link,
            node.center_link_m,
        ) + coverage.translations_object_m
        box_scale = (
            self.intersector.characteristic_length_m
            + np.linalg.norm(centers, ord=np.inf, axis=1)
            + float(np.sum(moving_half_extents))
        )
        half_extents = np.nextafter(
            moving_half_extents[None, :]
            + coverage.spatial_error_bound_m
            + _FK_ERROR * box_scale[:, None],
            math.inf,
        )
        world_half_extents = np.nextafter(
            np.einsum(
                "nij,nj->ni",
                np.abs(coverage.rotations_object_from_link),
                half_extents,
            )
            * (1.0 + _FK_ERROR),
            math.inf,
        )
        lower = np.nextafter(centers - world_half_extents, -math.inf)
        upper = np.nextafter(centers + world_half_extents, math.inf)
        overlaps = np.all(
            upper >= self.object_coordinate_lower_m[None, :], axis=1
        ) & np.all(
            lower <= self.object_coordinate_upper_m[None, :], axis=1
        )
        return int(np.count_nonzero(overlaps))

    def _classify_whole_path_pad_aabb_hierarchies_restarted_reference(
        self,
        coverages: Sequence[_WholePathPadAabbHierarchyCoverage],
    ) -> tuple[WholePathPadSphereScreen, ...]:
        """Use boolean broadphase internally and exact face rows only at leaves."""

        rows = tuple(coverages)
        if not rows:
            return ()
        possible_contact = np.zeros(len(rows), dtype=bool)
        spatial_node_queries = np.zeros(len(rows), dtype=np.int64)
        aabb_free_counts = np.zeros(len(rows), dtype=np.int64)
        sat_free_counts = np.zeros(len(rows), dtype=np.int64)
        sat_triangle_tests = np.zeros(len(rows), dtype=np.int64)
        maximum_depth = np.zeros(len(rows), dtype=np.int64)
        pending: list[list[tuple[int, int]]] = []
        for coverage_index, row in enumerate(rows):
            root = self.prepared_pads[
                row.prepared_pad_index
            ].surface_aabb_root
            pending.append(
                [
                    (root, segment_index)
                    for segment_index in reversed(
                        range(_WHOLE_PATH_SPHERE_SEGMENT_COUNT)
                    )
                ]
            )

        while True:
            active: list[tuple[int, int, int]] = []
            for coverage_index, stack in enumerate(pending):
                if possible_contact[coverage_index]:
                    stack.clear()
                    continue
                for _unused in range(
                    min(_PAD_AABB_FRONTIER_BATCH_PER_COVERAGE, len(stack))
                ):
                    node_index, segment_index = stack.pop()
                    active.append(
                        (coverage_index, node_index, segment_index)
                    )
            if not active:
                break
            centers = np.empty((len(active), 3), dtype=np.float64)
            box_axes = np.empty((len(active), 3, 3), dtype=np.float64)
            box_half_extents = np.empty((len(active), 3), dtype=np.float64)
            leaf_mask = np.zeros(len(active), dtype=bool)
            for active_offset, (
                coverage_index,
                node_index,
                segment_index,
            ) in enumerate(active):
                row = rows[coverage_index]
                prepared = self.prepared_pads[row.prepared_pad_index]
                node = prepared.surface_aabb_nodes[node_index]
                rotation = row.rotations_object_from_link[segment_index]
                centers[active_offset] = (
                    rotation @ node.center_link_m
                    + row.translations_object_m[segment_index]
                )
                box_axes[active_offset] = rotation
                motion_speed = float(
                    np.nextafter(
                        row.motion_speed_radius_slope_upper_per_unit
                        * node.maximum_vertex_radius_link_m
                        + row.motion_speed_intercept_upper_m_per_unit,
                        math.inf,
                    )
                )
                moving_half_extents = np.nextafter(
                    node.box_half_extents_upper_m
                    + motion_speed * row.segment_half_width_unit,
                    math.inf,
                )
                box_scale = (
                    self.intersector.characteristic_length_m
                    + float(
                        np.linalg.norm(centers[active_offset], ord=np.inf)
                    )
                    + float(
                        np.sum(
                            moving_half_extents
                        )
                    )
                )
                box_half_extents[active_offset] = np.nextafter(
                    moving_half_extents
                    + row.spatial_error_bound_m
                    + _FK_ERROR * box_scale,
                    math.inf,
                )
                leaf_mask[active_offset] = node.leaf
            world_half_extents = np.nextafter(
                np.einsum(
                    "nij,nj->ni", np.abs(box_axes), box_half_extents
                )
                * (1.0 + _FK_ERROR),
                math.inf,
            )
            lower = np.nextafter(
                centers - world_half_extents, -math.inf
            )
            upper = np.nextafter(
                centers + world_half_extents, math.inf
            )
            overlaps = np.zeros(len(active), dtype=bool)
            nonleaf_offsets = np.flatnonzero(~leaf_mask)
            if len(nonleaf_offsets) > 0:
                overlaps[nonleaf_offsets] = (
                    self.distance_bvh.aabbs_have_face_overlap(
                        lower[nonleaf_offsets], upper[nonleaf_offsets]
                    )
                )
            leaf_offsets = np.flatnonzero(leaf_mask)
            leaf_sat_free = np.zeros(len(active), dtype=bool)
            if len(leaf_offsets) > 0:
                leaf_face_rows = (
                    self.distance_bvh.face_indices_intersecting_aabbs(
                        lower[leaf_offsets], upper[leaf_offsets]
                    )
                )
                overlaps[leaf_offsets] = np.asarray(
                    [len(face_indices) > 0 for face_indices in leaf_face_rows],
                    dtype=bool,
                )
                nonempty_leaf_rows = tuple(
                    (leaf_row_index, face_indices)
                    for leaf_row_index, face_indices in enumerate(
                        leaf_face_rows
                    )
                    if len(face_indices) > 0
                )
                if nonempty_leaf_rows:
                    pair_face_indices = np.concatenate(
                        [row[1] for row in nonempty_leaf_rows]
                    )
                    pair_active_offsets = np.concatenate(
                        [
                            np.full(
                                len(face_indices),
                                int(leaf_offsets[leaf_row_index]),
                                dtype=np.int64,
                            )
                            for leaf_row_index, face_indices in (
                                nonempty_leaf_rows
                            )
                        ]
                    )
                    pair_separated = (
                        self._triangle_obb_pair_strict_separation_mask(
                            triangles_object_m=(
                                self.canonical_object_face_vertices_m[
                                    pair_face_indices
                                ]
                            ),
                            box_centers_object_m=centers[
                                pair_active_offsets
                            ],
                            box_axes_object=box_axes[pair_active_offsets],
                            box_half_extents_m=box_half_extents[
                                pair_active_offsets
                            ],
                        )
                    )
                    pair_cursor = 0
                    for leaf_row_index, face_indices in nonempty_leaf_rows:
                        active_offset = int(leaf_offsets[leaf_row_index])
                        next_cursor = pair_cursor + len(face_indices)
                        leaf_sat_free[active_offset] = bool(
                            np.all(
                                pair_separated[pair_cursor:next_cursor]
                            )
                        )
                        coverage_index = active[active_offset][0]
                        sat_triangle_tests[coverage_index] += len(
                            face_indices
                        )
                        pair_cursor = next_cursor

            for active_offset, (
                coverage_index,
                node_index,
                segment_index,
            ) in enumerate(active):
                prepared = self.prepared_pads[
                    rows[coverage_index].prepared_pad_index
                ]
                node = prepared.surface_aabb_nodes[node_index]
                spatial_node_queries[coverage_index] += 1
                maximum_depth[coverage_index] = max(
                    maximum_depth[coverage_index], node.depth
                )
                if not overlaps[active_offset]:
                    aabb_free_counts[coverage_index] += 1
                    continue
                if leaf_sat_free[active_offset]:
                    sat_free_counts[coverage_index] += 1
                    continue
                if node.leaf:
                    possible_contact[coverage_index] = True
                    pending[coverage_index].clear()
                    continue
                pending[coverage_index].append(
                    (node.right, segment_index)
                )
                pending[coverage_index].append(
                    (node.left, segment_index)
                )

        result: list[WholePathPadSphereScreen] = []
        for coverage_index, row in enumerate(rows):
            prepared = self.prepared_pads[row.prepared_pad_index]
            result.append(
                WholePathPadSphereScreen(
                    pad_name=prepared.verified.name,
                    finger_name=prepared.verified.finger_name,
                    segment_count=_WHOLE_PATH_SPHERE_SEGMENT_COUNT,
                    nearest_surface_query_count=0,
                    distance_bvh_node_visits=0,
                    distance_triangle_tests=int(
                        sat_triangle_tests[coverage_index]
                    ),
                    minimum_clearance_lower_bound_m=0.0,
                    certified_free=not bool(
                        possible_contact[coverage_index]
                    ),
                    spatial_node_query_count=int(
                        spatial_node_queries[coverage_index]
                    ),
                    aabb_certified_free_node_count=int(
                        aabb_free_counts[coverage_index]
                    ),
                    exact_distance_query_count=0,
                    maximum_spatial_depth_reached=int(
                        maximum_depth[coverage_index]
                    ),
                    obb_sat_certified_free_node_count=int(
                        sat_free_counts[coverage_index]
                    ),
                    obb_sat_triangle_test_count=int(
                        sat_triangle_tests[coverage_index]
                    ),
                )
            )
        return tuple(result)

    @staticmethod
    def _obb_aabb_strict_separation_mask(
        *,
        obb_centers_m: Sequence[Sequence[float]],
        obb_axes: Sequence[Sequence[Sequence[float]]],
        obb_half_extents_m: Sequence[Sequence[float]],
        aabb_lower_m: Sequence[Sequence[float]],
        aabb_upper_m: Sequence[Sequence[float]],
    ) -> np.ndarray:
        """Prove N OBB/AABB pairs separate on the complete 15 SAT axes."""

        centers = np.asarray(obb_centers_m, dtype=np.float64)
        rotations = np.asarray(obb_axes, dtype=np.float64)
        obb_half = np.asarray(obb_half_extents_m, dtype=np.float64)
        lower = np.asarray(aabb_lower_m, dtype=np.float64)
        upper = np.asarray(aabb_upper_m, dtype=np.float64)
        count = len(centers)
        if (
            centers.shape != (count, 3)
            or count == 0
            or rotations.shape != (count, 3, 3)
            or obb_half.shape != (count, 3)
            or lower.shape != (count, 3)
            or upper.shape != (count, 3)
            or not np.all(np.isfinite(centers))
            or not np.all(np.isfinite(rotations))
            or not np.all(np.isfinite(obb_half))
            or not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper))
            or np.any(obb_half < 0.0)
            or np.any(lower > upper)
        ):
            raise RayClosureError(
                "OBB/AABB SAT inputs must have aligned finite boxes"
            )
        aabb_centers = lower + 0.5 * (upper - lower)
        aabb_half = np.nextafter(
            0.5 * (upper - lower), math.inf
        )
        obb_axes_rows = np.swapaxes(rotations, 1, 2)
        identity = np.eye(3, dtype=np.float64)
        world_axes = np.broadcast_to(
            identity[None, :, :], (count, 3, 3)
        )
        cross_axes = np.cross(
            obb_axes_rows[:, :, None, :],
            identity[None, None, :, :],
        ).reshape((count, 9, 3))
        candidate_axes = np.concatenate(
            (obb_axes_rows, world_axes, cross_axes), axis=1
        )
        axis_norm = np.linalg.norm(candidate_axes, axis=2)
        center_delta = aabb_centers - centers
        center_projection = np.abs(
            np.einsum("nc,nac->na", center_delta, candidate_axes)
        )
        axes_in_obb = np.einsum(
            "nac,njc->naj", candidate_axes, obb_axes_rows
        )
        obb_radius = np.einsum(
            "naj,nj->na", np.abs(axes_in_obb), obb_half
        )
        aabb_radius = np.einsum(
            "nac,nc->na", np.abs(candidate_axes), aabb_half
        )
        coordinate_scale = (
            np.max(np.abs(centers), axis=1)
            + np.max(np.abs(aabb_centers), axis=1)
            + np.sum(obb_half, axis=1)
            + np.sum(aabb_half, axis=1)
            + 1.0
        )
        projection_error = _FK_ERROR * (
            center_projection
            + obb_radius
            + aabb_radius
            + axis_norm * coordinate_scale[:, None]
        )
        strict_limit = np.nextafter(
            obb_radius + aabb_radius + projection_error,
            math.inf,
        )
        valid_axis = axis_norm > np.finfo(np.float64).tiny
        result = np.any(
            valid_axis & (center_projection > strict_limit), axis=1
        )
        result.setflags(write=False)
        return result

    @staticmethod
    def _moving_triangle_triangle_strict_separation_mask(
        *,
        moving_triangles_midpoint_m: Sequence[
            Sequence[Sequence[float]]
        ],
        moving_vertex_motion_radius_upper_m: Sequence[Sequence[float]],
        fixed_triangles_m: Sequence[Sequence[Sequence[float]]],
    ) -> np.ndarray:
        """Prove swept-triangle separation on seventeen fixed SAT axes.

        Each moving vertex may lie anywhere in its certified midpoint ball.
        The projection interval of the complete moving triangle is therefore
        enclosed by the three independently expanded vertex projections.
        Separation on any nondegenerate fixed axis is sufficient; failure to
        separate remains uncertain and falls back to the older OBB test.
        """

        moving = np.asarray(
            moving_triangles_midpoint_m, dtype=np.float64
        )
        radii = np.asarray(
            moving_vertex_motion_radius_upper_m, dtype=np.float64
        )
        fixed = np.asarray(fixed_triangles_m, dtype=np.float64)
        count = len(moving)
        if (
            moving.shape != (count, 3, 3)
            or count == 0
            or radii.shape != (count, 3)
            or fixed.shape != (count, 3, 3)
            or not np.all(np.isfinite(moving))
            or not np.all(np.isfinite(radii))
            or not np.all(np.isfinite(fixed))
            or np.any(radii < 0.0)
        ):
            raise RayClosureError(
                "moving triangle SAT inputs need aligned finite triangles "
                "and nonnegative vertex motion radii"
            )
        moving_edges = np.stack(
            (
                moving[:, 1] - moving[:, 0],
                moving[:, 2] - moving[:, 1],
                moving[:, 0] - moving[:, 2],
            ),
            axis=1,
        )
        fixed_edges = np.stack(
            (
                fixed[:, 1] - fixed[:, 0],
                fixed[:, 2] - fixed[:, 1],
                fixed[:, 0] - fixed[:, 2],
            ),
            axis=1,
        )
        moving_normals = np.cross(
            moving_edges[:, 0], -moving_edges[:, 2]
        )
        fixed_normals = np.cross(
            fixed_edges[:, 0], -fixed_edges[:, 2]
        )
        edge_cross_axes = np.cross(
            moving_edges[:, :, None, :],
            fixed_edges[:, None, :, :],
        ).reshape((count, 9, 3))
        moving_in_plane_axes = np.cross(
            moving_edges, moving_normals[:, None, :]
        )
        fixed_in_plane_axes = np.cross(
            fixed_edges, fixed_normals[:, None, :]
        )
        candidate_axes = np.concatenate(
            (
                moving_normals[:, None, :],
                fixed_normals[:, None, :],
                edge_cross_axes,
                moving_in_plane_axes,
                fixed_in_plane_axes,
            ),
            axis=1,
        )
        axis_norm = np.linalg.norm(candidate_axes, axis=2)
        moving_projection = np.einsum(
            "nvc,nac->nva", moving, candidate_axes
        )
        moving_expansion = (
            radii[:, :, None] * axis_norm[:, None, :]
        )
        moving_minimum = np.min(
            moving_projection - moving_expansion, axis=1
        )
        moving_maximum = np.max(
            moving_projection + moving_expansion, axis=1
        )
        fixed_projection = np.einsum(
            "nvc,nac->nva", fixed, candidate_axes
        )
        fixed_minimum = np.min(fixed_projection, axis=1)
        fixed_maximum = np.max(fixed_projection, axis=1)
        coordinate_scale = (
            np.max(np.abs(moving), axis=(1, 2))
            + np.max(np.abs(fixed), axis=(1, 2))
            + np.max(radii, axis=1)
            + 1.0
        )
        projection_error = np.nextafter(
            _DOT_ERROR
            * (
                np.abs(moving_minimum)
                + np.abs(moving_maximum)
                + np.abs(fixed_minimum)
                + np.abs(fixed_maximum)
                + axis_norm * coordinate_scale[:, None]
            ),
            math.inf,
        )
        forward_gap = fixed_minimum - moving_maximum
        reverse_gap = moving_minimum - fixed_maximum
        valid_axis = axis_norm > np.finfo(np.float64).tiny
        result = np.any(
            valid_axis
            & (
                (forward_gap > projection_error)
                | (reverse_gap > projection_error)
            ),
            axis=1,
        )
        result.setflags(write=False)
        return result

    @staticmethod
    def _box_dot_product_upper(
        *,
        first_lower: np.ndarray,
        first_upper: np.ndarray,
        second_lower: np.ndarray,
        second_upper: np.ndarray,
    ) -> np.ndarray:
        """Return an outward upper bound for aligned interval dot products."""

        first_lower = np.asarray(first_lower, dtype=np.float64)
        first_upper = np.asarray(first_upper, dtype=np.float64)
        second_lower = np.asarray(second_lower, dtype=np.float64)
        second_upper = np.asarray(second_upper, dtype=np.float64)
        if (
            first_lower.shape != first_upper.shape
            or first_lower.shape != second_lower.shape
            or first_lower.shape != second_upper.shape
            or first_lower.ndim < 1
            or first_lower.shape[-1] != 3
            or not np.all(np.isfinite(first_lower))
            or not np.all(np.isfinite(first_upper))
            or not np.all(np.isfinite(second_lower))
            or not np.all(np.isfinite(second_upper))
            or np.any(first_lower > first_upper)
            or np.any(second_lower > second_upper)
        ):
            raise RayClosureError(
                "interval dot products need aligned finite three-vector boxes"
            )
        products = np.nextafter(
            np.stack(
                (
                    first_lower * second_lower,
                    first_lower * second_upper,
                    first_upper * second_lower,
                    first_upper * second_upper,
                ),
                axis=-1,
            ),
            math.inf,
        )
        component_upper = np.max(products, axis=-1)
        result = component_upper[..., 0]
        for component in (1, 2):
            result = np.nextafter(
                result + component_upper[..., component], math.inf
            )
        result = np.asarray(result, dtype=np.float64)
        result.setflags(write=False)
        return result

    def _directional_witness_segment_bounds(
        self,
        coverage: _WholePathPadAabbHierarchyCoverage,
    ) -> _DirectionalWitnessSegmentBounds:
        """Enclose witness velocities and positive PAD approach per segment."""

        prepared = self.prepared_pads[coverage.prepared_pad_index]
        witness_points = prepared.witness_points_link_m
        witness_count = len(witness_points)
        pad_approach_possible = np.empty(
            (_WHOLE_PATH_SPHERE_SEGMENT_COUNT, witness_count),
            dtype=bool,
        )
        transform_cache = self.interval_kinematics.new_link_transform_cache()
        segment_width = 2.0 * coverage.segment_half_width_unit
        for segment_index in range(_WHOLE_PATH_SPHERE_SEGMENT_COUNT):
            phase_lower = segment_index * segment_width
            phase_upper = min(
                coverage.maximum_parameter,
                (segment_index + 1) * segment_width,
            )
            try:
                motion = (
                    self.interval_kinematics.point_velocity_and_vector_many(
                    link_name=prepared.verified.link_name,
                    q_start=coverage.q_start,
                    direction=coverage.direction,
                    phase_lower=phase_lower,
                    phase_upper=phase_upper,
                    base_transform=coverage.object_from_hand,
                    points_local_m=witness_points,
                    vectors_local=prepared.witness_normals_link,
                    transform_cache=transform_cache,
                )
                )
            except IntervalKinematicsError as error:
                raise RayClosureError(
                    "directional contact interval motion rejected the "
                    f"mechanical path for {prepared.verified.name}: {error}"
                ) from error
            pad_approach_upper = self._box_dot_product_upper(
                first_lower=motion.vector_lower_object,
                first_upper=motion.vector_upper_object,
                second_lower=(
                    motion.point_velocity_lower_object_m_per_unit
                ),
                second_upper=(
                    motion.point_velocity_upper_object_m_per_unit
                ),
            )
            pad_approach_possible[segment_index] = (
                pad_approach_upper > 0.0
            )
        pad_approach_possible.setflags(write=False)
        node_count = len(prepared.surface_aabb_nodes)
        node_pad_approach_possible = np.zeros(
            (_WHOLE_PATH_SPHERE_SEGMENT_COUNT, node_count),
            dtype=bool,
        )
        witness_count_per_triangle = len(_BARYCENTRIC_WITNESSES)
        leaf_nodes = prepared.surface_aabb_leaf_node_indices
        leaf_witness_indices = (
            prepared.surface_aabb_leaf_triangle_indices[:, None]
            * witness_count_per_triangle
            + np.arange(
                witness_count_per_triangle, dtype=np.int64
            )[None, :]
        )
        leaf_possible = pad_approach_possible[:, leaf_witness_indices]
        node_pad_approach_possible[:, leaf_nodes] = np.any(
            leaf_possible, axis=2
        )
        for node_indices in (
            prepared.surface_aabb_internal_nodes_by_reverse_depth
        ):
            left_indices = prepared.surface_aabb_left_child_indices[
                node_indices
            ]
            right_indices = prepared.surface_aabb_right_child_indices[
                node_indices
            ]
            child_indices = np.stack(
                (left_indices, right_indices), axis=1
            )
            child_possible = node_pad_approach_possible[
                :, child_indices
            ]
            node_pad_approach_possible[:, node_indices] = np.any(
                child_possible, axis=2
            )
        node_pad_approach_possible.setflags(write=False)
        return _DirectionalWitnessSegmentBounds(
            pad_approach_possible=pad_approach_possible,
            node_pad_approach_possible=node_pad_approach_possible,
            interval_witness_motion_evaluation_count=(
                2
                * witness_count
                * _WHOLE_PATH_SPHERE_SEGMENT_COUNT
            ),
        )

    def _classify_whole_path_pad_aabb_hierarchies(
        self,
        coverages: Sequence[_WholePathPadAabbHierarchyCoverage],
        *,
        enable_moving_triangle_refinement: bool = True,
        maximum_moving_triangle_pair_tests_per_coverage: int | None = None,
        enable_directional_contact_feasibility: bool = False,
    ) -> tuple[WholePathPadSphereScreen, ...]:
        """Traverse persistent PAD/object BVH pairs in cheap or bounded fine mode."""

        rows = tuple(coverages)
        if not rows:
            return ()
        if (
            enable_moving_triangle_refinement
            and enable_directional_contact_feasibility
        ):
            raise RayClosureError(
                "moving-triangle refinement and directional contact culling "
                "are separate narrowphase modes"
            )
        if (
            maximum_moving_triangle_pair_tests_per_coverage is not None
            and maximum_moving_triangle_pair_tests_per_coverage <= 0
        ):
            raise RayClosureError(
                "moving triangle pair-test budget must be positive"
            )
        possible_contact = np.zeros(len(rows), dtype=bool)
        spatial_possible_contact = np.zeros(len(rows), dtype=bool)
        directionally_pruned_branch = np.zeros(len(rows), dtype=bool)
        pair_queries = np.zeros(len(rows), dtype=np.int64)
        separated_pair_counts = np.zeros(len(rows), dtype=np.int64)
        sat_free_counts = np.zeros(len(rows), dtype=np.int64)
        sat_triangle_tests = np.zeros(len(rows), dtype=np.int64)
        moving_triangle_free_counts = np.zeros(
            len(rows), dtype=np.int64
        )
        moving_triangle_tests = np.zeros(len(rows), dtype=np.int64)
        temporal_refined_pair_counts = np.zeros(
            len(rows), dtype=np.int64
        )
        temporal_transform_counts = np.zeros(len(rows), dtype=np.int64)
        maximum_temporal_depth = np.zeros(len(rows), dtype=np.int64)
        work_budget_exhausted = np.zeros(len(rows), dtype=bool)
        directional_node_pair_tests = np.zeros(len(rows), dtype=np.int64)
        directional_node_pair_rejections = np.zeros(
            len(rows), dtype=np.int64
        )
        directional_leaf_face_pair_tests = np.zeros(
            len(rows), dtype=np.int64
        )
        directional_leaf_face_pair_rejections = np.zeros(
            len(rows), dtype=np.int64
        )
        directional_interval_motion_evaluations = np.zeros(
            len(rows), dtype=np.int64
        )
        maximum_depth = np.zeros(len(rows), dtype=np.int64)
        pending: list[list[tuple[int, int, int, int, int]]] = []
        refined_transform_cache: dict[
            tuple[int, int, int, int], tuple[np.ndarray, np.ndarray]
        ] = {}
        directional_bounds: tuple[
            _DirectionalWitnessSegmentBounds, ...
        ] | None = None
        if enable_directional_contact_feasibility:
            directional_bounds = tuple(
                self._directional_witness_segment_bounds(row)
                for row in rows
            )
            for coverage_index, bounds in enumerate(directional_bounds):
                directional_interval_motion_evaluations[coverage_index] = (
                    bounds.interval_witness_motion_evaluation_count
                )
        for row in rows:
            prepared = self.prepared_pads[row.prepared_pad_index]
            pending.append(
                [
                    (
                        prepared.surface_aabb_root,
                        self.distance_bvh.root,
                        segment_index,
                        0,
                        0,
                    )
                    for segment_index in reversed(
                        range(_WHOLE_PATH_SPHERE_SEGMENT_COUNT)
                    )
                ]
            )

        while True:
            active: list[tuple[int, int, int, int, int, int]] = []
            for coverage_index, stack in enumerate(pending):
                if possible_contact[coverage_index]:
                    stack.clear()
                    continue
                if (
                    enable_moving_triangle_refinement
                    and maximum_moving_triangle_pair_tests_per_coverage
                    is not None
                    and moving_triangle_tests[coverage_index]
                    >= maximum_moving_triangle_pair_tests_per_coverage
                ):
                    work_budget_exhausted[coverage_index] = True
                    possible_contact[coverage_index] = True
                    spatial_possible_contact[coverage_index] = True
                    stack.clear()
                    continue
                for _unused in range(
                    min(_PAD_AABB_FRONTIER_BATCH_PER_COVERAGE, len(stack))
                ):
                    (
                        pad_node_index,
                        object_node_index,
                        segment_index,
                        temporal_depth,
                        temporal_ordinal,
                    ) = stack.pop()
                    active.append(
                        (
                            coverage_index,
                            pad_node_index,
                            object_node_index,
                            segment_index,
                            temporal_depth,
                            temporal_ordinal,
                        )
                    )
            if not active:
                break

            centers = np.empty((len(active), 3), dtype=np.float64)
            box_axes = np.empty((len(active), 3, 3), dtype=np.float64)
            box_half_extents = np.empty((len(active), 3), dtype=np.float64)
            world_half_extents = np.empty((len(active), 3), dtype=np.float64)
            object_lower = np.empty((len(active), 3), dtype=np.float64)
            object_upper = np.empty((len(active), 3), dtype=np.float64)
            temporal_half_widths = np.empty(
                len(active), dtype=np.float64
            )
            link_translations = np.empty(
                (len(active), 3), dtype=np.float64
            )
            pad_leaf = np.zeros(len(active), dtype=bool)
            object_leaf = np.zeros(len(active), dtype=bool)
            for active_offset, (
                coverage_index,
                pad_node_index,
                object_node_index,
                segment_index,
                temporal_depth,
                temporal_ordinal,
            ) in enumerate(active):
                row = rows[coverage_index]
                prepared = self.prepared_pads[row.prepared_pad_index]
                pad_node = prepared.surface_aabb_nodes[pad_node_index]
                object_node = self.distance_bvh.nodes[object_node_index]
                temporal_half_width = (
                    row.segment_half_width_unit / (1 << temporal_depth)
                )
                temporal_half_widths[active_offset] = temporal_half_width
                maximum_temporal_depth[coverage_index] = max(
                    maximum_temporal_depth[coverage_index],
                    temporal_depth,
                )
                if temporal_depth == 0:
                    rotation = row.rotations_object_from_link[segment_index]
                    translation = row.translations_object_m[segment_index]
                else:
                    transform_key = (
                        coverage_index,
                        segment_index,
                        temporal_depth,
                        temporal_ordinal,
                    )
                    cached_transform = refined_transform_cache.get(
                        transform_key
                    )
                    if cached_transform is None:
                        base_segment_width = (
                            2.0 * row.segment_half_width_unit
                        )
                        refined_segment_width = (
                            base_segment_width / (1 << temporal_depth)
                        )
                        midpoint = (
                            segment_index * base_segment_width
                            + (temporal_ordinal + 0.5)
                            * refined_segment_width
                        )
                        links = self.hand_model.forward_kinematics(
                            row.q_start + midpoint * row.direction,
                            base_transform=row.object_from_hand,
                        )
                        transform = links[prepared.verified.link_name]
                        rotation = np.asarray(
                            transform[:3, :3], dtype=np.float64
                        ).copy()
                        translation = np.asarray(
                            transform[:3, 3], dtype=np.float64
                        ).copy()
                        rotation.setflags(write=False)
                        translation.setflags(write=False)
                        cached_transform = (rotation, translation)
                        refined_transform_cache[transform_key] = (
                            cached_transform
                        )
                        temporal_transform_counts[coverage_index] += 1
                    rotation, translation = cached_transform
                centers[active_offset] = (
                    rotation @ pad_node.center_link_m
                    + translation
                )
                box_axes[active_offset] = rotation
                link_translations[active_offset] = translation
                motion_speed = float(
                    np.nextafter(
                        row.motion_speed_radius_slope_upper_per_unit
                        * pad_node.maximum_vertex_radius_link_m
                        + row.motion_speed_intercept_upper_m_per_unit,
                        math.inf,
                    )
                )
                moving_half_extents = np.nextafter(
                    pad_node.box_half_extents_upper_m
                    + motion_speed * temporal_half_width,
                    math.inf,
                )
                box_scale = (
                    self.intersector.characteristic_length_m
                    + float(
                        np.linalg.norm(centers[active_offset], ord=np.inf)
                    )
                    + float(np.sum(moving_half_extents))
                )
                box_half_extents[active_offset] = np.nextafter(
                    moving_half_extents
                    + row.spatial_error_bound_m
                    + _FK_ERROR * box_scale,
                    math.inf,
                )
                world_half_extents[active_offset] = np.nextafter(
                    np.abs(rotation) @ box_half_extents[active_offset]
                    * (1.0 + _FK_ERROR),
                    math.inf,
                )
                object_lower[active_offset] = np.nextafter(
                    object_node.lower_m + self.distance_bvh.centre_m,
                    -math.inf,
                )
                object_upper[active_offset] = np.nextafter(
                    object_node.upper_m + self.distance_bvh.centre_m,
                    math.inf,
                )
                pad_leaf[active_offset] = pad_node.leaf
                object_leaf[active_offset] = object_node.leaf

            separated = self._obb_aabb_strict_separation_mask(
                obb_centers_m=centers,
                obb_axes=box_axes,
                obb_half_extents_m=box_half_extents,
                aabb_lower_m=object_lower,
                aabb_upper_m=object_upper,
            )
            directionally_impossible = np.zeros(
                len(active), dtype=bool
            )
            if enable_directional_contact_feasibility:
                if directional_bounds is None:  # pragma: no cover
                    raise RayClosureError(
                        "directional contact bounds were not prepared"
                    )
                unresolved_offsets = np.flatnonzero(~separated)
                for active_offset_value in unresolved_offsets:
                    active_offset = int(active_offset_value)
                    (
                        coverage_index,
                        pad_node_index,
                        object_node_index,
                        segment_index,
                        _temporal_depth,
                        _temporal_ordinal,
                    ) = active[active_offset]
                    directional_node_pair_tests[coverage_index] += 1
                    object_node = self.distance_bvh.nodes[
                        object_node_index
                    ]
                    if (
                        self.distance_bvh.allowed_contact_face_count[
                            object_node_index
                        ]
                        == 0
                    ):
                        directionally_impossible[active_offset] = True
                        continue
                    bounds = directional_bounds[coverage_index]
                    if not bounds.node_pad_approach_possible[
                        segment_index, pad_node_index
                    ]:
                        directionally_impossible[active_offset] = True
                for active_offset_value in np.flatnonzero(
                    directionally_impossible & ~separated
                ):
                    active_offset = int(active_offset_value)
                    coverage_index = active[active_offset][0]
                    directional_node_pair_rejections[coverage_index] += 1
            terminal_offsets = np.flatnonzero(
                ~separated
                & ~directionally_impossible
                & pad_leaf
                & object_leaf
            )
            leaf_pair_free = np.zeros(len(active), dtype=bool)
            leaf_pair_valid_contact_possible = np.zeros(
                len(active), dtype=bool
            )
            if len(terminal_offsets) > 0:
                pair_face_indices = np.concatenate(
                    [
                        self.distance_bvh.nodes[
                            active[int(active_offset)][2]
                        ].face_indices
                        for active_offset in terminal_offsets
                    ]
                )
                pair_active_offsets = np.concatenate(
                    [
                        np.full(
                            len(
                                self.distance_bvh.nodes[
                                    active[int(active_offset)][2]
                                ].face_indices
                            ),
                            int(active_offset),
                            dtype=np.int64,
                        )
                        for active_offset in terminal_offsets
                    ]
                )
                fixed_triangles = (
                    self.canonical_object_face_vertices_m[
                        pair_face_indices
                    ]
                )
                pair_separated = np.array(
                    self._triangle_obb_pair_strict_separation_mask(
                        triangles_object_m=fixed_triangles,
                        box_centers_object_m=centers[
                            pair_active_offsets
                        ],
                        box_axes_object=box_axes[pair_active_offsets],
                        box_half_extents_m=box_half_extents[
                            pair_active_offsets
                        ],
                    ),
                    dtype=bool,
                    copy=True,
                )
                moving_pair_tested = np.zeros(
                    len(pair_face_indices), dtype=bool
                )
                moving_pair_separated = np.zeros(
                    len(pair_face_indices), dtype=bool
                )
                if enable_moving_triangle_refinement:
                    unresolved_pair_offsets = np.flatnonzero(
                        ~pair_separated
                    )
                    if len(unresolved_pair_offsets) > 0:
                        unresolved_active_offsets = pair_active_offsets[
                            unresolved_pair_offsets
                        ]
                        pair_pad_local_triangles = np.stack(
                            [
                                self.prepared_pads[
                                    rows[
                                        active[int(active_offset)][0]
                                    ].prepared_pad_index
                                ].surface_triangles_link_m[
                                    int(
                                        self.prepared_pads[
                                            rows[
                                                active[int(active_offset)][0]
                                            ].prepared_pad_index
                                        ].surface_aabb_nodes[
                                            active[int(active_offset)][1]
                                        ].triangle_indices[0]
                                    )
                                ]
                                for active_offset in unresolved_active_offsets
                            ]
                        )
                        pair_rotations = box_axes[
                            unresolved_active_offsets
                        ]
                        pair_moving_triangles = (
                            np.einsum(
                                "nij,nvj->nvi",
                                pair_rotations,
                                pair_pad_local_triangles,
                            )
                            + link_translations[
                                unresolved_active_offsets, None, :
                            ]
                        )
                        pair_local_vertex_radii = np.linalg.norm(
                            pair_pad_local_triangles, axis=2
                        )
                        pair_radius_slopes = np.asarray(
                            [
                                rows[
                                    active[int(active_offset)][0]
                                ].motion_speed_radius_slope_upper_per_unit
                                for active_offset in unresolved_active_offsets
                            ],
                            dtype=np.float64,
                        )
                        pair_speed_intercepts = np.asarray(
                            [
                                rows[
                                    active[int(active_offset)][0]
                                ].motion_speed_intercept_upper_m_per_unit
                                for active_offset in unresolved_active_offsets
                            ],
                            dtype=np.float64,
                        )
                        pair_vertex_speeds = np.nextafter(
                            pair_radius_slopes[:, None]
                            * pair_local_vertex_radii
                            + pair_speed_intercepts[:, None],
                            math.inf,
                        )
                        pair_spatial_errors = np.asarray(
                            [
                                rows[
                                    active[int(active_offset)][0]
                                ].spatial_error_bound_m
                                for active_offset in unresolved_active_offsets
                            ],
                            dtype=np.float64,
                        )
                        pair_coordinate_scale = (
                            self.intersector.characteristic_length_m
                            + np.max(
                                np.abs(pair_moving_triangles), axis=2
                            )
                            + pair_local_vertex_radii
                        )
                        pair_motion_radii = np.nextafter(
                            pair_vertex_speeds
                            * temporal_half_widths[
                                unresolved_active_offsets, None
                            ]
                            + pair_spatial_errors[:, None]
                            + _FK_ERROR * pair_coordinate_scale,
                            math.inf,
                        )
                        refined_separated = (
                            self._moving_triangle_triangle_strict_separation_mask(
                                moving_triangles_midpoint_m=(
                                    pair_moving_triangles
                                ),
                                moving_vertex_motion_radius_upper_m=(
                                    pair_motion_radii
                                ),
                                fixed_triangles_m=fixed_triangles[
                                    unresolved_pair_offsets
                                ],
                            )
                        )
                        moving_pair_tested[
                            unresolved_pair_offsets
                        ] = True
                        moving_pair_separated[
                            unresolved_pair_offsets
                        ] = refined_separated
                        pair_separated[
                            unresolved_pair_offsets
                        ] = refined_separated
                pair_direction_possible = np.ones(
                    len(pair_face_indices), dtype=bool
                )
                if enable_directional_contact_feasibility:
                    if directional_bounds is None:  # pragma: no cover
                        raise RayClosureError(
                            "directional contact bounds were not prepared"
                        )
                    pair_direction_possible[:] = False
                    spatially_unresolved_offsets = np.flatnonzero(
                        ~pair_separated
                    )
                    if len(spatially_unresolved_offsets) > 0:
                        pad_approach_possible = np.empty(
                            len(spatially_unresolved_offsets), dtype=bool
                        )
                        for row_offset, pair_offset_value in enumerate(
                            spatially_unresolved_offsets
                        ):
                            pair_offset = int(pair_offset_value)
                            active_offset = int(
                                pair_active_offsets[pair_offset]
                            )
                            coverage_index = active[active_offset][0]
                            pad_node_index = active[active_offset][1]
                            segment_index = active[active_offset][3]
                            bounds = directional_bounds[coverage_index]
                            pad_approach_possible[row_offset] = (
                                bounds.node_pad_approach_possible[
                                    segment_index, pad_node_index
                                ]
                            )
                        allowed_faces = self._contact_face_mask[
                            pair_face_indices[spatially_unresolved_offsets]
                        ]
                        possible_rows = (
                            allowed_faces & pad_approach_possible
                        )
                        pair_direction_possible[
                            spatially_unresolved_offsets
                        ] = possible_rows
                pair_cursor = 0
                for active_offset_value in terminal_offsets:
                    active_offset = int(active_offset_value)
                    object_node = self.distance_bvh.nodes[
                        active[active_offset][2]
                    ]
                    next_cursor = pair_cursor + len(
                        object_node.face_indices
                    )
                    leaf_pair_free[active_offset] = bool(
                        np.all(pair_separated[pair_cursor:next_cursor])
                    )
                    leaf_pair_valid_contact_possible[
                        active_offset
                    ] = bool(
                        np.any(
                            ~pair_separated[pair_cursor:next_cursor]
                            & pair_direction_possible[
                                pair_cursor:next_cursor
                            ]
                        )
                    )
                    coverage_index = active[active_offset][0]
                    moving_triangle_tests[coverage_index] += int(
                        np.count_nonzero(
                            moving_pair_tested[pair_cursor:next_cursor]
                        )
                    )
                    moving_triangle_free_counts[coverage_index] += int(
                        np.count_nonzero(
                            moving_pair_separated[
                                pair_cursor:next_cursor
                            ]
                        )
                    )
                    sat_triangle_tests[coverage_index] += len(
                        object_node.face_indices
                    )
                    if enable_directional_contact_feasibility:
                        spatially_unresolved = ~pair_separated[
                            pair_cursor:next_cursor
                        ]
                        directional_leaf_face_pair_tests[
                            coverage_index
                        ] += int(np.count_nonzero(spatially_unresolved))
                        directional_leaf_face_pair_rejections[
                            coverage_index
                        ] += int(
                            np.count_nonzero(
                                spatially_unresolved
                                & ~pair_direction_possible[
                                    pair_cursor:next_cursor
                                ]
                            )
                        )
                    pair_cursor = next_cursor

            for active_offset, (
                coverage_index,
                pad_node_index,
                object_node_index,
                segment_index,
                temporal_depth,
                temporal_ordinal,
            ) in enumerate(active):
                prepared = self.prepared_pads[
                    rows[coverage_index].prepared_pad_index
                ]
                pad_node = prepared.surface_aabb_nodes[pad_node_index]
                object_node = self.distance_bvh.nodes[object_node_index]
                pair_queries[coverage_index] += 1
                maximum_depth[coverage_index] = max(
                    maximum_depth[coverage_index], pad_node.depth
                )
                if separated[active_offset]:
                    separated_pair_counts[coverage_index] += 1
                    continue
                if directionally_impossible[active_offset]:
                    directionally_pruned_branch[coverage_index] = True
                    continue
                if pad_node.leaf and object_node.leaf:
                    if leaf_pair_free[active_offset]:
                        sat_free_counts[coverage_index] += 1
                    elif enable_directional_contact_feasibility:
                        spatial_possible_contact[coverage_index] = True
                        if leaf_pair_valid_contact_possible[active_offset]:
                            possible_contact[coverage_index] = True
                            pending[coverage_index].clear()
                        else:
                            directionally_pruned_branch[
                                coverage_index
                            ] = True
                    elif (
                        enable_moving_triangle_refinement
                        and temporal_depth
                        < _PAD_AABB_MAXIMUM_TEMPORAL_REFINEMENT_DEPTH
                    ):
                        child_depth = temporal_depth + 1
                        first_child_ordinal = 2 * temporal_ordinal
                        pending[coverage_index].append(
                            (
                                pad_node_index,
                                object_node_index,
                                segment_index,
                                child_depth,
                                first_child_ordinal + 1,
                            )
                        )
                        pending[coverage_index].append(
                            (
                                pad_node_index,
                                object_node_index,
                                segment_index,
                                child_depth,
                                first_child_ordinal,
                            )
                        )
                        temporal_refined_pair_counts[coverage_index] += 2
                    else:
                        possible_contact[coverage_index] = True
                        spatial_possible_contact[coverage_index] = True
                        pending[coverage_index].clear()
                    continue
                pad_volume = float(
                    np.prod(2.0 * world_half_extents[active_offset])
                )
                object_volume = float(
                    np.prod(
                        object_upper[active_offset]
                        - object_lower[active_offset]
                    )
                )
                split_object = (
                    not object_node.leaf
                    and (pad_node.leaf or object_volume >= pad_volume)
                )
                if split_object:
                    pending[coverage_index].append(
                        (
                            pad_node_index,
                            object_node.right,
                            segment_index,
                            temporal_depth,
                            temporal_ordinal,
                        )
                    )
                    pending[coverage_index].append(
                        (
                            pad_node_index,
                            object_node.left,
                            segment_index,
                            temporal_depth,
                            temporal_ordinal,
                        )
                    )
                else:
                    pending[coverage_index].append(
                        (
                            pad_node.right,
                            object_node_index,
                            segment_index,
                            temporal_depth,
                            temporal_ordinal,
                        )
                    )
                    pending[coverage_index].append(
                        (
                            pad_node.left,
                            object_node_index,
                            segment_index,
                            temporal_depth,
                            temporal_ordinal,
                        )
                    )

        result: list[WholePathPadSphereScreen] = []
        for coverage_index, row in enumerate(rows):
            prepared = self.prepared_pads[row.prepared_pad_index]
            result.append(
                WholePathPadSphereScreen(
                    pad_name=prepared.verified.name,
                    finger_name=prepared.verified.finger_name,
                    segment_count=_WHOLE_PATH_SPHERE_SEGMENT_COUNT,
                    nearest_surface_query_count=0,
                    distance_bvh_node_visits=int(
                        pair_queries[coverage_index]
                    ),
                    distance_triangle_tests=int(
                        sat_triangle_tests[coverage_index]
                    ),
                    minimum_clearance_lower_bound_m=0.0,
                    certified_free=(
                        not bool(spatial_possible_contact[coverage_index])
                        and not bool(
                            directionally_pruned_branch[coverage_index]
                        )
                    ),
                    spatial_node_query_count=int(
                        pair_queries[coverage_index]
                    ),
                    aabb_certified_free_node_count=int(
                        separated_pair_counts[coverage_index]
                    ),
                    exact_distance_query_count=0,
                    maximum_spatial_depth_reached=int(
                        maximum_depth[coverage_index]
                    ),
                    obb_sat_certified_free_node_count=int(
                        sat_free_counts[coverage_index]
                    ),
                    obb_sat_triangle_test_count=int(
                        sat_triangle_tests[coverage_index]
                    ),
                    moving_triangle_sat_certified_free_pair_count=int(
                        moving_triangle_free_counts[coverage_index]
                    ),
                    moving_triangle_sat_pair_test_count=int(
                        moving_triangle_tests[coverage_index]
                    ),
                    temporal_refined_leaf_pair_count=int(
                        temporal_refined_pair_counts[coverage_index]
                    ),
                    temporal_refinement_transform_count=int(
                        temporal_transform_counts[coverage_index]
                    ),
                    maximum_temporal_refinement_depth_reached=int(
                        maximum_temporal_depth[coverage_index]
                    ),
                    narrowphase_refinement_used=(
                        enable_moving_triangle_refinement
                    ),
                    narrowphase_work_budget_exhausted=bool(
                        work_budget_exhausted[coverage_index]
                    ),
                    directional_contact_feasibility_used=(
                        enable_directional_contact_feasibility
                    ),
                    directional_bvh_node_pair_test_count=int(
                        directional_node_pair_tests[coverage_index]
                    ),
                    directional_bvh_node_pair_rejected_count=int(
                        directional_node_pair_rejections[coverage_index]
                    ),
                    directional_leaf_face_pair_test_count=int(
                        directional_leaf_face_pair_tests[coverage_index]
                    ),
                    directional_leaf_face_pair_rejected_count=int(
                        directional_leaf_face_pair_rejections[
                            coverage_index
                        ]
                    ),
                    directional_interval_witness_motion_evaluation_count=int(
                        directional_interval_motion_evaluations[
                            coverage_index
                        ]
                    ),
                    certified_no_valid_contact=(
                        enable_directional_contact_feasibility
                        and not bool(possible_contact[coverage_index])
                        and (
                            bool(spatial_possible_contact[coverage_index])
                            or bool(
                                directionally_pruned_branch[
                                    coverage_index
                                ]
                            )
                        )
                    ),
                )
            )
        return tuple(result)

    def screen_unit_parameter_batch(
        self,
        parameters_unit: Sequence[Sequence[float]],
        hand_model: ThreeFingerHandModel | None = None,
    ) -> tuple[tuple[WholePathPadSphereScreen, ...], ...]:
        """Screen many candidates in one batched surface-distance query.

        This method never creates or accepts a grasp.  Invalid parameter rows
        fail closed with an exception; callers use it only after canonical
        proposal validation.  An uncertain row is deliberately retained for
        the exact V9 evaluator.
        """

        supplied_hand = self.hand_model if hand_model is None else hand_model
        self._validate_hand(supplied_hand)
        parameters = np.asarray(parameters_unit, dtype=np.float64)
        if (
            parameters.ndim != 2
            or parameters.shape[1:] != (self.parameter_dimension,)
            or len(parameters) == 0
            or not np.all(np.isfinite(parameters))
        ):
            raise RayClosureError(
                "batch screen parameters need finite non-empty shape (N, D)"
            )
        coverage_rows_by_candidate: list[
            tuple[_WholePathPadAabbHierarchyCoverage, ...]
        ] = []
        for row_index, parameter_row in enumerate(parameters):
            try:
                (
                    q_start,
                    _target,
                    _rotation,
                    object_from_hand,
                    hand_extent,
                ) = self._decode_with_object_from_hand(
                    parameter_row
                )
                spatial_error = (
                    self.intersector.distance_error_bound_m
                    + self.distance_bvh.aabb_error_bound_m
                    + _FK_ERROR
                    * (
                        self.intersector.characteristic_length_m
                        + hand_extent
                    )
                )
                candidate_rows = (
                    self._whole_path_pad_aabb_hierarchy_coverages(
                        q_start=q_start,
                        object_from_hand=object_from_hand,
                        spatial_error_bound_m=spatial_error,
                    )
                )
            except RayClosureError as error:
                raise RayClosureError(
                    f"batch screen row {row_index} rejected: {error}"
                ) from error
            if len(candidate_rows) != len(self.prepared_pads):
                raise RayClosureError(
                    "candidate PAD coverage count differs from hand contract"
                )
            coverage_rows_by_candidate.append(tuple(candidate_rows))

        root_scores = tuple(
            tuple(
                self._pad_aabb_root_global_overlap_count(coverage)
                for coverage in rows
            )
            for rows in coverage_rows_by_candidate
        )
        pad_orders = tuple(
            tuple(
                sorted(
                    range(len(scores)),
                    key=lambda pad_index: (scores[pad_index], pad_index),
                )
            )
            for scores in root_scores
        )
        def run_stage(
            candidate_indices: Sequence[int],
            *,
            enable_moving_triangle_refinement: bool,
            enable_directional_contact_feasibility: bool,
        ) -> tuple[
            list[list[WholePathPadSphereScreen | None]], list[int]
        ]:
            stage_slots: list[
                list[WholePathPadSphereScreen | None]
            ] = [
                [None] * len(self.prepared_pads)
                for _candidate_index in coverage_rows_by_candidate
            ]
            remaining = list(candidate_indices)
            for order_index in range(len(self.prepared_pads)):
                stage_candidate_indices: list[int] = []
                stage_coverages: list[
                    _WholePathPadAabbHierarchyCoverage
                ] = []
                for candidate_index in remaining:
                    pad_index = pad_orders[candidate_index][order_index]
                    coverage = coverage_rows_by_candidate[candidate_index][
                        pad_index
                    ]
                    stage_candidate_indices.append(candidate_index)
                    stage_coverages.append(coverage)
                if stage_coverages:
                    stage_results = (
                        self._classify_whole_path_pad_aabb_hierarchies(
                            stage_coverages,
                            enable_moving_triangle_refinement=(
                                enable_moving_triangle_refinement
                            ),
                            enable_directional_contact_feasibility=(
                                enable_directional_contact_feasibility
                            ),
                            maximum_moving_triangle_pair_tests_per_coverage=(
                                _PAD_AABB_MAXIMUM_MOVING_TRIANGLE_PAIR_TESTS_PER_COVERAGE
                                if enable_moving_triangle_refinement
                                else None
                            ),
                        )
                    )
                    for candidate_index, screen in zip(
                        stage_candidate_indices, stage_results
                    ):
                        pad_index = pad_orders[candidate_index][order_index]
                        stage_slots[candidate_index][pad_index] = screen
                remaining = [
                    candidate_index
                    for candidate_index in remaining
                    if not bool(
                        stage_slots[candidate_index][
                            pad_orders[candidate_index][order_index]
                        ].certified_free
                        or stage_slots[candidate_index][
                            pad_orders[candidate_index][order_index]
                        ].certified_no_valid_contact
                    )
                ]
                if not remaining:
                    break
            return stage_slots, remaining

        cheap_slots, _cheap_survivors = run_stage(
            range(len(coverage_rows_by_candidate)),
            enable_moving_triangle_refinement=False,
            enable_directional_contact_feasibility=False,
        )

        result: list[tuple[WholePathPadSphereScreen, ...]] = []
        for candidate_index in range(len(coverage_rows_by_candidate)):
            candidate_slots = cheap_slots[candidate_index]
            for pad_index, screen in enumerate(candidate_slots):
                if screen is not None:
                    candidate_slots[pad_index] = replace(
                        screen,
                        root_overlap_segment_count=int(
                            root_scores[candidate_index][pad_index]
                        ),
                    )
                    continue
                prepared = self.prepared_pads[
                    coverage_rows_by_candidate[candidate_index][
                        pad_index
                    ].prepared_pad_index
                ]
                candidate_slots[pad_index] = WholePathPadSphereScreen(
                    pad_name=prepared.verified.name,
                    finger_name=prepared.verified.finger_name,
                    segment_count=0,
                    nearest_surface_query_count=0,
                    distance_bvh_node_visits=0,
                    distance_triangle_tests=0,
                    minimum_clearance_lower_bound_m=0.0,
                    certified_free=False,
                    narrowphase_refinement_used=False,
                    skipped_due_to_other_pad_free=True,
                    root_overlap_segment_count=int(
                        root_scores[candidate_index][pad_index]
                    ),
                )
            result.append(
                tuple(
                    screen
                    for screen in candidate_slots
                    if screen is not None
                )
            )
        return tuple(result)

    def screen_unit_parameters(
        self,
        parameters_unit: Sequence[float],
        hand_model: ThreeFingerHandModel | None = None,
    ) -> tuple[WholePathPadSphereScreen, ...]:
        parameters = np.asarray(parameters_unit, dtype=np.float64)
        if parameters.shape != (self.parameter_dimension,):
            raise RayClosureError(
                f"unit parameters must have shape ({self.parameter_dimension},)"
            )
        return self.screen_unit_parameter_batch(
            parameters[None, :], hand_model
        )[0]

    @staticmethod
    def _bind_whole_path_sphere_screen_to_counters(
        counters: _PadCounters,
        screen: WholePathPadSphereScreen,
    ) -> None:
        counters.whole_path_sphere_screen_segments = screen.segment_count
        counters.whole_path_sphere_screen_queries = (
            screen.spatial_node_query_count
        )
        counters.whole_path_sphere_screen_bvh_node_visits = (
            screen.distance_bvh_node_visits
        )
        counters.whole_path_sphere_screen_triangle_tests = (
            screen.distance_triangle_tests
        )
        counters.whole_path_sphere_screen_obb_sat_certified_free_nodes = (
            screen.obb_sat_certified_free_node_count
        )
        counters.whole_path_sphere_screen_obb_sat_triangle_tests = (
            screen.obb_sat_triangle_test_count
        )
        counters.whole_path_sphere_screen_moving_triangle_sat_certified_free_pairs = (
            screen.moving_triangle_sat_certified_free_pair_count
        )
        counters.whole_path_sphere_screen_moving_triangle_sat_pair_tests = (
            screen.moving_triangle_sat_pair_test_count
        )
        counters.whole_path_sphere_screen_temporal_refined_leaf_pairs = (
            screen.temporal_refined_leaf_pair_count
        )
        counters.whole_path_sphere_screen_temporal_refinement_transforms = (
            screen.temporal_refinement_transform_count
        )
        counters.whole_path_sphere_screen_maximum_temporal_refinement_depth = (
            screen.maximum_temporal_refinement_depth_reached
        )
        counters.whole_path_sphere_screen_narrowphase_refinement_used = (
            screen.narrowphase_refinement_used
        )
        counters.whole_path_sphere_screen_narrowphase_work_budget_exhausted = (
            screen.narrowphase_work_budget_exhausted
        )
        counters.whole_path_sphere_screen_directional_contact_feasibility_used = (
            screen.directional_contact_feasibility_used
        )
        counters.whole_path_sphere_screen_directional_bvh_node_pair_tests = (
            screen.directional_bvh_node_pair_test_count
        )
        counters.whole_path_sphere_screen_directional_bvh_node_pair_rejections = (
            screen.directional_bvh_node_pair_rejected_count
        )
        counters.whole_path_sphere_screen_directional_leaf_face_pair_tests = (
            screen.directional_leaf_face_pair_test_count
        )
        counters.whole_path_sphere_screen_directional_leaf_face_pair_rejections = (
            screen.directional_leaf_face_pair_rejected_count
        )
        counters.whole_path_sphere_screen_directional_interval_witness_motion_evaluations = (
            screen.directional_interval_witness_motion_evaluation_count
        )
        counters.whole_path_sphere_screen_certified_no_valid_contact = (
            screen.certified_no_valid_contact
        )
        counters.whole_path_sphere_screen_certified_free = (
            screen.certified_free
        )
        counters.whole_path_sphere_screen_clearance_lower_bound_m = (
            screen.minimum_clearance_lower_bound_m
        )
        if screen.certified_free:
            counters.clearance_lower_bound_m = (
                screen.minimum_clearance_lower_bound_m
            )

    def _ancestor_joint_names(self, link_name: str) -> tuple[str, ...]:
        by_child = {joint.child_link: name for name, joint in self.hand_model.joints.items()}
        names: list[str] = []
        cursor = link_name
        while cursor != self.hand_model.base_link:
            name = by_child.get(cursor)
            if name is None:
                raise RayClosureError(f"PAD link {link_name} is disconnected from hand base")
            names.append(name)
            cursor = self.hand_model.joints[name].parent_link
        names.reverse()
        return tuple(names)

    def _local_point_speed_bounds(
        self,
        prepared: _PreparedPad,
        points_local_m: np.ndarray,
        q_start: np.ndarray,
        direction: np.ndarray,
        maximum_parameter: float,
    ) -> np.ndarray:
        points = np.asarray(points_local_m, dtype=np.float64)
        if (
            points.ndim != 2
            or points.shape[1:] != (3,)
            or len(points) == 0
            or not np.all(np.isfinite(points))
        ):
            raise RayClosureError(
                "kinematic speed bounds require finite non-empty local points"
            )
        return self._local_radius_speed_bounds(
            prepared=prepared,
            local_radii_m=np.linalg.norm(points, axis=1),
            q_start=q_start,
            direction=direction,
            maximum_parameter=maximum_parameter,
        )

    def _local_radius_speed_bounds(
        self,
        *,
        prepared: _PreparedPad,
        local_radii_m: Sequence[float],
        q_start: np.ndarray,
        direction: np.ndarray,
        maximum_parameter: float,
    ) -> np.ndarray:
        local_radii = np.asarray(local_radii_m, dtype=np.float64)
        if (
            local_radii.ndim != 1
            or len(local_radii) == 0
            or not np.all(np.isfinite(local_radii))
            or np.any(local_radii < 0.0)
        ):
            raise RayClosureError(
                "kinematic speed bounds require finite non-negative radii"
            )
        radius_slope, intercept = (
            self._local_radius_speed_affine_coefficients(
                prepared=prepared,
                q_start=q_start,
                direction=direction,
                maximum_parameter=maximum_parameter,
            )
        )
        velocity_bounds = np.nextafter(
            radius_slope * local_radii + intercept,
            math.inf,
        )
        if np.any(velocity_bounds <= 0.0) or not np.all(
            np.isfinite(velocity_bounds)
        ):
            raise RayClosureError(
                f"closing path for {prepared.verified.name} has no finite PAD motion bound"
            )
        velocity_bounds.setflags(write=False)
        return velocity_bounds

    def _local_radius_speed_affine_coefficients(
        self,
        *,
        prepared: _PreparedPad,
        q_start: np.ndarray,
        direction: np.ndarray,
        maximum_parameter: float,
    ) -> tuple[float, float]:
        """Return speed <= radius_slope * local_radius + intercept."""

        endpoint = q_start + maximum_parameter * direction
        resolved_start = self.hand_model.resolve_joint_positions(q_start)
        resolved_end = self.hand_model.resolve_joint_positions(endpoint)
        resolved_velocity = self.hand_model.resolve_joint_velocities(
            direction, enforce_limits=False
        )
        ancestor_names = self._ancestor_joint_names(prepared.verified.link_name)
        radius_slope = 0.0
        intercept = 0.0
        for ancestor_index, name in enumerate(ancestor_names):
            joint = self.hand_model.joints[name]
            rate = abs(float(resolved_velocity[name]))
            if joint.joint_type in ("revolute", "continuous"):
                radius_slope += rate
                downstream_reach = 0.0
                for downstream_name in ancestor_names[ancestor_index + 1 :]:
                    downstream = self.hand_model.joints[downstream_name]
                    downstream_reach += float(
                        np.linalg.norm(downstream.origin_xyz_m)
                    )
                    if downstream.joint_type == "prismatic":
                        downstream_reach += max(
                            abs(float(resolved_start[downstream_name])),
                            abs(float(resolved_end[downstream_name])),
                        )
                intercept += rate * downstream_reach
            elif joint.joint_type == "prismatic":
                intercept += rate
        if (
            radius_slope + intercept <= 0.0
            or not math.isfinite(radius_slope)
            or not math.isfinite(intercept)
        ):
            raise RayClosureError(
                f"closing path for {prepared.verified.name} has no finite PAD motion bound"
            )
        radius_slope_upper = float(
            np.nextafter(
                radius_slope * (1.0 + _FK_ERROR), math.inf
            )
        )
        intercept_upper = float(
            np.nextafter(intercept * (1.0 + _FK_ERROR), math.inf)
        )
        return radius_slope_upper, intercept_upper

    def _witness_speed_bounds(
        self,
        prepared: _PreparedPad,
        q_start: np.ndarray,
        direction: np.ndarray,
        maximum_parameter: float,
    ) -> np.ndarray:
        return self._local_point_speed_bounds(
            prepared,
            prepared.witness_points_link_m,
            q_start,
            direction,
            maximum_parameter,
        )

    def _witness_states(
        self,
        prepared: _PreparedPad,
        q: np.ndarray,
        direction: np.ndarray,
        object_from_hand: np.ndarray,
    ) -> _WitnessStates:
        links = self.hand_model.forward_kinematics(
            q, base_transform=object_from_hand
        )
        link_transform = links[prepared.verified.link_name]
        rotation = link_transform[:3, :3]
        translation = link_transform[:3, 3]
        offsets_object = prepared.witness_points_link_m @ rotation.T
        positions_object = offsets_object + translation
        normals_object = prepared.witness_normals_link @ rotation.T
        origin_jacobian = self.hand_model.geometric_jacobian(
            prepared.verified.link_name,
            q,
            point_local_m=(0.0, 0.0, 0.0),
            base_transform=object_from_hand,
        )
        origin_twist = origin_jacobian @ direction
        velocities_object = (
            origin_twist[:3]
            + np.cross(
                np.broadcast_to(origin_twist[3:], offsets_object.shape),
                offsets_object,
            )
        )
        speeds = np.linalg.norm(velocities_object, axis=1)
        error_bounds = _DOT_ERROR * speeds
        approach = np.sum(normals_object * velocities_object, axis=1)
        leading = approach > error_bounds
        for array in (
            positions_object,
            velocities_object,
            normals_object,
            error_bounds,
            leading,
            rotation,
            translation,
        ):
            array.setflags(write=False)
        return _WitnessStates(
            positions_object_m=positions_object,
            velocities_object_per_unit=velocities_object,
            pad_source_winding_normals_object=normals_object,
            leading=leading,
            leading_error_bounds=error_bounds,
            link_rotation_object=rotation,
            link_translation_object_m=translation,
        )

    def _witness_positions_object(
        self,
        prepared: _PreparedPad,
        q: np.ndarray,
        object_from_hand: np.ndarray,
    ) -> np.ndarray:
        """Compute all witness positions once for one temporal endpoint."""

        links = self.hand_model.forward_kinematics(
            q, base_transform=object_from_hand
        )
        transform = links[prepared.verified.link_name]
        positions = (
            prepared.witness_points_link_m @ transform[:3, :3].T
            + transform[:3, 3]
        )
        positions.setflags(write=False)
        return positions

    @staticmethod
    def _witness_state_cache_key(
        prepared: _PreparedPad,
        q: np.ndarray,
        direction: np.ndarray,
        object_from_hand: np.ndarray,
    ) -> tuple[object, ...]:
        relevant = np.asarray(
            q[np.asarray(prepared.relevant_joint_indices, dtype=np.int64)],
            dtype="<f8",
        )
        return (
            "EXACT_FK_WITNESS_STATE",
            prepared.verified.name,
            relevant.tobytes(order="C"),
            np.asarray(direction, dtype="<f8").tobytes(order="C"),
            np.asarray(object_from_hand, dtype="<f8").tobytes(order="C"),
        )

    def _cached_witness_states(
        self,
        prepared: _PreparedPad,
        q: np.ndarray,
        direction: np.ndarray,
        object_from_hand: np.ndarray,
        execution: _GeometryExecutionContext,
    ) -> tuple[_WitnessStates, tuple[object, ...]]:
        key = self._witness_state_cache_key(
            prepared, q, direction, object_from_hand
        )
        if execution.cache_enabled:
            cached = execution.witness_state_cache.get(key)
            if cached is not None:
                execution.stats.witness_state_cache_hits += 1
                return cached, key
        execution.stats.witness_state_cache_misses += 1
        states = self._witness_states(
            prepared, q, direction, object_from_hand
        )
        if execution.cache_enabled:
            # A binary-subdivision midpoint is unique until an identical
            # final-q or explicit repeated sample is requested.  Retaining all
            # historical midpoints would therefore scale memory with the
            # computational budget without adding hits.  One latest state per
            # PAD is sufficient, while clearing nearest rows on a state change
            # keeps those rows tied to exactly one current geometry sample.
            stale_keys = tuple(
                cached_key
                for cached_key in execution.witness_state_cache
                if cached_key[1] == prepared.verified.name
            )
            for stale_key in stale_keys:
                del execution.witness_state_cache[stale_key]
            execution.nearest_batch_cache.clear()
            execution.witness_state_cache[key] = states
        return states, key

    def _cached_nearest_many(
        self,
        *,
        states: _WitnessStates,
        state_key: tuple[object, ...],
        witness_indices: np.ndarray,
        execution: _GeometryExecutionContext,
    ) -> _NearestMany:
        canonical_indices = np.asarray(witness_indices, dtype="<i8")
        key = (
            "OBJECT_NEAREST_FOR_WITNESS_SUBSET",
            state_key,
            canonical_indices.tobytes(order="C"),
        )
        if execution.cache_enabled:
            cached = execution.nearest_batch_cache.get(key)
            if cached is not None:
                execution.stats.nearest_batch_cache_hits += 1
                return cached
        execution.stats.nearest_batch_cache_misses += 1
        execution.stats.exact_nearest_witness_queries += len(canonical_indices)
        nearest = self.distance_bvh.nearest_many(
            states.positions_object_m[canonical_indices]
        )
        if execution.cache_enabled:
            execution.nearest_batch_cache[key] = nearest
        return nearest

    def _witness_node_margin_lower_bound(
        self,
        *,
        prepared: _PreparedPad,
        node_index: int,
        states: _WitnessStates,
        thresholds_m: np.ndarray,
    ) -> float:
        """Lower-bound every witness distance-minus-threshold in one node.

        The sphere is built from all finite witnesses, then enlarged by a
        binary64 forward-error bound for the current rigid transform.  The
        object root AABB contains every object triangle, so center-to-AABB
        distance minus that enlarged radius is a lower bound on every exact
        point-to-object distance used by the H2 reference implementation.
        """

        node = prepared.witness_hierarchy_nodes[node_index]
        rotation = states.link_rotation_object
        translation = states.link_translation_object_m
        rotation_one_norm = float(np.linalg.norm(rotation, ord=1))
        rotation_infinity_norm = float(np.linalg.norm(rotation, ord=np.inf))
        rotation_norm_upper = float(
            np.nextafter(
                math.sqrt(rotation_one_norm * rotation_infinity_norm)
                * (1.0 + _FK_ERROR),
                math.inf,
            )
        )
        transform_roundoff = (
            2.0
            * math.sqrt(3.0)
            * _FK_ERROR
            * (
                rotation_infinity_norm * node.coordinate_scale_link_m
                + float(np.linalg.norm(translation, ord=np.inf))
            )
        )
        radius_object_upper = float(
            np.nextafter(
                rotation_norm_upper * node.radius_upper_m + transform_roundoff,
                math.inf,
            )
        )
        center_object = rotation @ node.center_link_m + translation
        center_relative = center_object - self.distance_bvh.centre_m
        root = self.distance_bvh.nodes[self.distance_bvh.root]
        center_aabb_distance = float(
            self.distance_bvh._point_aabb_distances(
                center_relative[None, :], root
            )[0]
        )
        aabb_forward_error = (
            self.distance_bvh.aabb_error_bound_m
            + _FK_ERROR
            * (
                float(np.linalg.norm(center_object, ord=np.inf))
                + float(np.linalg.norm(self.distance_bvh.centre_m, ord=np.inf))
                + self.distance_bvh.characteristic_length_m
                + radius_object_upper
            )
        )
        raw_distance_lower = (
            center_aabb_distance - radius_object_upper - aabb_forward_error
        )
        distance_lower = 0.0
        if raw_distance_lower > 0.0:
            distance_lower = float(
                np.nextafter(raw_distance_lower, -math.inf)
            )
        maximum_threshold = float(
            np.max(thresholds_m[node.witness_indices])
        )
        subtraction_error = _FK_ERROR * (
            abs(distance_lower)
            + abs(maximum_threshold)
            + self.distance_bvh.characteristic_length_m
        )
        return float(
            np.nextafter(
                distance_lower - maximum_threshold - subtraction_error,
                -math.inf,
            )
        )

    def _interval_geometry(
        self,
        *,
        prepared: _PreparedPad,
        states: _WitnessStates,
        state_key: tuple[object, ...],
        enclosure_radii_m: np.ndarray,
        spatial_error_bound_m: float,
        counters: _PadCounters,
        execution: _GeometryExecutionContext,
        _use_pre_nearest_aabb_prefilter: bool = True,
    ) -> _IntervalGeometry:
        """Return the H2 possible set with certified hierarchy pruning.

        Nodes are visited in increasing lower-bound order.  A node may be
        skipped only after its strict lower bound is greater than zero when a
        possible witness already exists, or greater than the best exact margin
        when the interval is free.  Thus every possible witness and the exact
        free-interval minimum are identical to an all-witness nearest query.
        """

        if not isinstance(_use_pre_nearest_aabb_prefilter, bool):
            raise RayClosureError(
                "pre-nearest AABB prefilter selector must be boolean"
            )
        thresholds = np.asarray(
            enclosure_radii_m + spatial_error_bound_m, dtype=np.float64
        )
        witness_count = len(states)
        possible = np.zeros(witness_count, dtype=bool)
        exact_margins = np.full(witness_count, math.inf, dtype=np.float64)
        nearest_face_indices = np.full(witness_count, -1, dtype=np.int64)
        exact = np.zeros(witness_count, dtype=bool)

        if _use_pre_nearest_aabb_prefilter:
            # A point can be within ``threshold`` of a triangle only if that
            # triangle's closed AABB overlaps the point-centred closed cube
            # with the same half-width.  The boolean BVH query uses outward
            # rounding, so a false row rigorously excludes contact without a
            # nearest-triangle calculation.  We return early only when the
            # exact survivors already contain a possible witness; otherwise
            # the unchanged hierarchy below still computes the exact global
            # free-interval minimum clearance.
            counters.pre_nearest_aabb_witness_tests += witness_count
            query_lower = np.nextafter(
                states.positions_object_m - thresholds[:, None],
                -math.inf,
            )
            query_upper = np.nextafter(
                states.positions_object_m + thresholds[:, None],
                math.inf,
            )
            overlap = self.distance_bvh.aabbs_have_face_overlap(
                query_lower, query_upper
            )
            survivor_indices = np.flatnonzero(overlap).astype(
                np.int64, copy=False
            )
            counters.pre_nearest_aabb_certified_free_witnesses += (
                witness_count - len(survivor_indices)
            )
            if 0 < len(survivor_indices) < witness_count:
                counters.pre_nearest_aabb_exact_survivors += len(
                    survivor_indices
                )
                survivor_nearest = self._cached_nearest_many(
                    states=states,
                    state_key=state_key,
                    witness_indices=survivor_indices,
                    execution=execution,
                )
                counters.distance_node_visits += int(
                    np.sum(survivor_nearest.node_visits)
                )
                counters.distance_triangle_tests += int(
                    np.sum(survivor_nearest.triangle_tests)
                )
                survivor_margins = (
                    survivor_nearest.distances_m
                    - thresholds[survivor_indices]
                )
                survivor_possible = survivor_margins <= 0.0
                if np.any(survivor_possible):
                    possible[survivor_indices] = survivor_possible
                    nearest_face_indices[survivor_indices] = (
                        survivor_nearest.face_indices
                    )
                    counters.pre_nearest_aabb_fast_paths += 1
                    if execution.verify_full_nearest:
                        full = self.distance_bvh.nearest_many(
                            states.positions_object_m
                        )
                        execution.stats.reference_shadow_witness_queries += (
                            witness_count
                        )
                        full_margins = full.distances_m - thresholds
                        full_possible = full_margins <= 0.0
                        if not np.array_equal(possible, full_possible):
                            raise RayClosureError(
                                "pre-nearest AABB possible set differs from "
                                "full-nearest reference"
                            )
                        if not np.array_equal(
                            survivor_margins,
                            full_margins[survivor_indices],
                        ):
                            raise RayClosureError(
                                "pre-nearest AABB survivor margins differ "
                                "from full-nearest reference"
                            )
                        if not np.array_equal(
                            survivor_nearest.face_indices,
                            full.face_indices[survivor_indices],
                        ):
                            raise RayClosureError(
                                "pre-nearest AABB survivor faces differ "
                                "from full-nearest reference"
                            )
                    possible.setflags(write=False)
                    nearest_face_indices.setflags(write=False)
                    return _IntervalGeometry(
                        possible=possible,
                        nearest_face_indices=nearest_face_indices,
                        minimum_free_margin_m=None,
                    )
            counters.pre_nearest_aabb_fallbacks += 1

        queue: list[tuple[float, int]] = []
        root_index = prepared.witness_hierarchy_root
        root_bound = self._witness_node_margin_lower_bound(
            prepared=prepared,
            node_index=root_index,
            states=states,
            thresholds_m=thresholds,
        )
        execution.stats.witness_hierarchy_nodes_tested += 1
        heapq.heappush(queue, (root_bound, root_index))
        best_exact_margin = math.inf
        any_possible = False

        def evaluate_exact_indices(indices: np.ndarray) -> None:
            nonlocal any_possible, best_exact_margin
            nearest = self._cached_nearest_many(
                states=states,
                state_key=state_key,
                witness_indices=indices,
                execution=execution,
            )
            counters.distance_node_visits += int(np.sum(nearest.node_visits))
            counters.distance_triangle_tests += int(
                np.sum(nearest.triangle_tests)
            )
            margins = nearest.distances_m - thresholds[indices]
            exact_margins[indices] = margins
            nearest_face_indices[indices] = nearest.face_indices
            exact[indices] = True
            possible_rows = margins <= 0.0
            possible[indices] = possible_rows
            any_possible = any_possible or bool(np.any(possible_rows))
            best_exact_margin = min(
                best_exact_margin,
                float(np.min(margins)),
            )

        while queue:
            if best_exact_margin < math.inf:
                # Freeze the prune limit after the first exact leaf, expand the
                # whole remaining certified frontier, and query it in one
                # stable-index batch.  In a possible interval the limit is zero.
                # In a currently free interval it is the best exact margin.
                # A later batch can only lower that margin; every node already
                # pruned above the old value remains above the new value, so the
                # possible set and exact global free minimum are unchanged.
                frozen_prune_limit = 0.0 if any_possible else best_exact_margin
                remaining_indices: list[int] = []
                while queue:
                    lower_bound, node_index = heapq.heappop(queue)
                    node = prepared.witness_hierarchy_nodes[node_index]
                    if lower_bound > frozen_prune_limit:
                        execution.stats.witness_hierarchy_witnesses_pruned += len(
                            node.witness_indices
                        )
                        continue
                    if node.leaf:
                        remaining_indices.extend(
                            int(index) for index in node.witness_indices
                        )
                        continue
                    for child_index in (node.left, node.right):
                        child_bound = self._witness_node_margin_lower_bound(
                            prepared=prepared,
                            node_index=child_index,
                            states=states,
                            thresholds_m=thresholds,
                        )
                        execution.stats.witness_hierarchy_nodes_tested += 1
                        heapq.heappush(queue, (child_bound, child_index))
                if remaining_indices:
                    canonical_indices = np.asarray(
                        sorted(remaining_indices), dtype=np.int64
                    )
                    evaluate_exact_indices(canonical_indices)
                break
            lower_bound, node_index = heapq.heappop(queue)
            prune_limit = 0.0 if any_possible else best_exact_margin
            node = prepared.witness_hierarchy_nodes[node_index]
            if lower_bound > prune_limit:
                execution.stats.witness_hierarchy_witnesses_pruned += len(
                    node.witness_indices
                )
                continue
            if node.leaf:
                indices = node.witness_indices
                evaluate_exact_indices(indices)
                continue
            for child_index in (node.left, node.right):
                child_bound = self._witness_node_margin_lower_bound(
                    prepared=prepared,
                    node_index=child_index,
                    states=states,
                    thresholds_m=thresholds,
                )
                execution.stats.witness_hierarchy_nodes_tested += 1
                heapq.heappush(queue, (child_bound, child_index))

        if not np.any(exact):
            raise RayClosureError("witness hierarchy produced no exact leaf query")
        minimum_free_margin = None
        if not any_possible:
            minimum_free_margin = best_exact_margin

        if execution.verify_full_nearest:
            full = self.distance_bvh.nearest_many(states.positions_object_m)
            execution.stats.reference_shadow_witness_queries += witness_count
            full_margins = full.distances_m - thresholds
            full_possible = full_margins <= 0.0
            if not np.array_equal(possible, full_possible):
                raise RayClosureError(
                    "witness hierarchy possible set differs from full-nearest reference"
                )
            if np.any(exact) and not np.array_equal(
                exact_margins[exact], full_margins[exact]
            ):
                raise RayClosureError(
                    "witness hierarchy exact leaves differ from full-nearest reference"
                )
            if np.any(exact) and not np.array_equal(
                nearest_face_indices[exact], full.face_indices[exact]
            ):
                raise RayClosureError(
                    "witness hierarchy nearest faces differ from full-nearest reference"
                )
            if not any_possible and minimum_free_margin != float(
                np.min(full_margins)
            ):
                raise RayClosureError(
                    "witness hierarchy free margin differs from full-nearest reference"
                )

        possible.setflags(write=False)
        nearest_face_indices.setflags(write=False)
        return _IntervalGeometry(
            possible=possible,
            nearest_face_indices=nearest_face_indices,
            minimum_free_margin_m=minimum_free_margin,
        )

    @staticmethod
    def _certified_chord_tube_position_bounds_v9(
        *,
        motion: IntervalPointMotion,
        start_position_object_m: np.ndarray,
        end_position_object_m: np.ndarray,
        phase_lower: float,
        phase_upper: float,
        endpoint_error_bound_m: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Enclose a one-dimensional path by its chord and curvature.

        For each Cartesian component, the linear-interpolation remainder is
        bounded by ``max(abs(second derivative)) * width**2 / 8``.  The
        endpoint error covers binary64 path-parameter construction and FK.
        """

        start = np.asarray(start_position_object_m, dtype=np.float64)
        end = np.asarray(end_position_object_m, dtype=np.float64)
        if (
            not isinstance(motion, IntervalPointMotion)
            or start.shape != (3,)
            or end.shape != (3,)
            or not np.all(np.isfinite(start))
            or not np.all(np.isfinite(end))
            or not math.isfinite(phase_lower)
            or not math.isfinite(phase_upper)
            or phase_lower > phase_upper
            or not math.isfinite(endpoint_error_bound_m)
            or endpoint_error_bound_m < 0.0
        ):
            raise RayClosureError("certified chord-tube inputs are malformed")
        width = float(
            np.nextafter(phase_upper - phase_lower, math.inf)
        )
        curvature_scale = float(
            np.nextafter(
                np.nextafter(width * width, math.inf) / 8.0,
                math.inf,
            )
        )
        acceleration = np.asarray(
            [
                max(abs(row.lower), abs(row.upper))
                for row in motion.acceleration_object_m_per_unit_squared
            ],
            dtype=np.float64,
        )
        curvature = np.nextafter(
            acceleration * curvature_scale * (1.0 + _TIME_ERROR),
            math.inf,
        )
        radius = np.nextafter(
            curvature + endpoint_error_bound_m, math.inf
        )
        lower = np.nextafter(np.minimum(start, end) - radius, -math.inf)
        upper = np.nextafter(np.maximum(start, end) + radius, math.inf)
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise RayClosureError("certified chord tube is non-finite")
        lower.setflags(write=False)
        upper.setflags(write=False)
        return lower, upper

    @staticmethod
    def _certified_chord_tube_position_bounds_many_v9(
        *,
        motion: IntervalPointMotionBatch,
        start_positions_object_m: np.ndarray,
        end_positions_object_m: np.ndarray,
        phase_lower: float,
        phase_upper: float,
        endpoint_error_bound_m: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vector form of the same strict chord-and-curvature enclosure."""

        start = np.asarray(start_positions_object_m, dtype=np.float64)
        end = np.asarray(end_positions_object_m, dtype=np.float64)
        if (
            not isinstance(motion, IntervalPointMotionBatch)
            or start.ndim != 2
            or start.shape[1:] != (3,)
            or len(start) == 0
            or end.shape != start.shape
            or motion.position_lower_object_m.shape != start.shape
            or not np.all(np.isfinite(start))
            or not np.all(np.isfinite(end))
            or not math.isfinite(phase_lower)
            or not math.isfinite(phase_upper)
            or phase_lower > phase_upper
            or not math.isfinite(endpoint_error_bound_m)
            or endpoint_error_bound_m < 0.0
        ):
            raise RayClosureError(
                "certified batch chord-tube inputs are malformed"
            )
        width = float(
            np.nextafter(phase_upper - phase_lower, math.inf)
        )
        curvature_scale = float(
            np.nextafter(
                np.nextafter(width * width, math.inf) / 8.0,
                math.inf,
            )
        )
        acceleration = np.maximum(
            np.abs(motion.acceleration_lower_object_m_per_unit_squared),
            np.abs(motion.acceleration_upper_object_m_per_unit_squared),
        )
        curvature = np.nextafter(
            acceleration * curvature_scale * (1.0 + _TIME_ERROR),
            math.inf,
        )
        radius = np.nextafter(
            curvature + endpoint_error_bound_m, math.inf
        )
        lower = np.nextafter(np.minimum(start, end) - radius, -math.inf)
        upper = np.nextafter(np.maximum(start, end) + radius, math.inf)
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise RayClosureError(
                "certified batch chord tube is non-finite"
            )
        lower.setflags(write=False)
        upper.setflags(write=False)
        return lower, upper

    @staticmethod
    def _outward_affine_form_bounds(
        *,
        position_lower: np.ndarray,
        position_upper: np.ndarray,
        coefficient_lower: np.ndarray,
        coefficient_upper: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate many affine interval forms with outward binary64 steps."""

        position_lower = np.asarray(position_lower, dtype=np.float64)
        position_upper = np.asarray(position_upper, dtype=np.float64)
        coefficient_lower = np.asarray(
            coefficient_lower, dtype=np.float64
        )
        coefficient_upper = np.asarray(
            coefficient_upper, dtype=np.float64
        )
        if (
            position_lower.shape != (3,)
            or position_upper.shape != (3,)
            or coefficient_lower.ndim != 3
            or coefficient_lower.shape[1:] != (4, 4)
            or coefficient_upper.shape != coefficient_lower.shape
            or not np.all(np.isfinite(position_lower))
            or not np.all(np.isfinite(position_upper))
            or np.any(position_lower > position_upper)
        ):
            raise RayClosureError(
                "batch affine culling inputs are malformed"
            )
        homogeneous_lower = np.concatenate(
            (position_lower, np.asarray((1.0,), dtype=np.float64))
        )
        homogeneous_upper = np.concatenate(
            (position_upper, np.asarray((1.0,), dtype=np.float64))
        )
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            products = np.stack(
                (
                    coefficient_lower * homogeneous_lower,
                    coefficient_lower * homogeneous_upper,
                    coefficient_upper * homogeneous_lower,
                    coefficient_upper * homogeneous_upper,
                ),
                axis=0,
            )
            product_lower = np.nextafter(
                np.min(products, axis=0), -math.inf
            )
            product_upper = np.nextafter(
                np.max(products, axis=0), math.inf
            )
            lower = np.zeros(coefficient_lower.shape[:2], dtype=np.float64)
            upper = np.zeros(coefficient_lower.shape[:2], dtype=np.float64)
            for column in range(4):
                lower = np.nextafter(
                    lower + product_lower[:, :, column], -math.inf
                )
                upper = np.nextafter(
                    upper + product_upper[:, :, column], math.inf
                )
        finite = np.all(np.isfinite(lower) & np.isfinite(upper), axis=1)
        if not np.all(finite):
            lower[~finite] = -math.inf
            upper[~finite] = math.inf
        return lower, upper

    @staticmethod
    def _outward_affine_form_bounds_pairwise(
        *,
        position_lower: np.ndarray,
        position_upper: np.ndarray,
        coefficient_lower: np.ndarray,
        coefficient_upper: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate one affine-form packet for one position box per row."""

        position_lower = np.asarray(position_lower, dtype=np.float64)
        position_upper = np.asarray(position_upper, dtype=np.float64)
        coefficient_lower = np.asarray(
            coefficient_lower, dtype=np.float64
        )
        coefficient_upper = np.asarray(
            coefficient_upper, dtype=np.float64
        )
        count = len(position_lower)
        coefficient_shape_valid = (
            coefficient_lower.ndim == 3
            and coefficient_lower.shape[0] == count
            and coefficient_lower.shape[1] > 0
            and coefficient_lower.shape[2] == 4
        )
        form_count = (
            coefficient_lower.shape[1]
            if coefficient_shape_valid
            else 0
        )
        if (
            position_lower.ndim != 2
            or position_lower.shape[1:] != (3,)
            or position_upper.shape != position_lower.shape
            or not coefficient_shape_valid
            or coefficient_upper.shape != coefficient_lower.shape
            or not np.all(np.isfinite(position_lower))
            or not np.all(np.isfinite(position_upper))
            or np.any(position_lower > position_upper)
        ):
            raise RayClosureError(
                "pairwise affine culling inputs are malformed"
            )
        if count == 0:
            return (
                np.empty((0, form_count), dtype=np.float64),
                np.empty((0, form_count), dtype=np.float64),
            )
        homogeneous_lower = np.empty((count, 4), dtype=np.float64)
        homogeneous_upper = np.empty((count, 4), dtype=np.float64)
        homogeneous_lower[:, :3] = position_lower
        homogeneous_upper[:, :3] = position_upper
        homogeneous_lower[:, 3] = 1.0
        homogeneous_upper[:, 3] = 1.0
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            lower = np.zeros((count, form_count), dtype=np.float64)
            upper = np.zeros((count, form_count), dtype=np.float64)
            for column in range(4):
                coefficient_column_lower = coefficient_lower[:, :, column]
                coefficient_column_upper = coefficient_upper[:, :, column]
                position_column_lower = homogeneous_lower[:, column, None]
                position_column_upper = homogeneous_upper[:, column, None]
                product_one = (
                    coefficient_column_lower * position_column_lower
                )
                product_two = (
                    coefficient_column_lower * position_column_upper
                )
                product_three = (
                    coefficient_column_upper * position_column_lower
                )
                product_four = (
                    coefficient_column_upper * position_column_upper
                )
                product_lower = np.minimum(product_one, product_two)
                np.minimum(product_lower, product_three, out=product_lower)
                np.minimum(product_lower, product_four, out=product_lower)
                product_lower = np.nextafter(product_lower, -math.inf)
                product_upper = np.maximum(product_one, product_two)
                np.maximum(product_upper, product_three, out=product_upper)
                np.maximum(product_upper, product_four, out=product_upper)
                product_upper = np.nextafter(product_upper, math.inf)
                lower = np.nextafter(
                    lower + product_lower, -math.inf
                )
                upper = np.nextafter(
                    upper + product_upper, math.inf
                )
        finite = np.all(np.isfinite(lower) & np.isfinite(upper), axis=1)
        if not np.all(finite):
            lower[~finite] = -math.inf
            upper[~finite] = math.inf
        return lower, upper

    @staticmethod
    def _outward_affine_form_bounds_at_points_pairwise(
        *,
        positions: np.ndarray,
        coefficient_lower: np.ndarray,
        coefficient_upper: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate affine intervals at one exact position per row.

        This is bitwise-equivalent to the general position-box evaluator when
        its lower and upper positions are identical, but it calculates only
        the two unique coefficient-endpoint products instead of four repeated
        products.
        """

        points = np.asarray(positions, dtype=np.float64)
        coefficient_lower = np.asarray(
            coefficient_lower, dtype=np.float64
        )
        coefficient_upper = np.asarray(
            coefficient_upper, dtype=np.float64
        )
        count = len(points)
        coefficient_shape_valid = (
            coefficient_lower.ndim == 3
            and coefficient_lower.shape[0] == count
            and coefficient_lower.shape[1] > 0
            and coefficient_lower.shape[2] == 4
        )
        form_count = (
            coefficient_lower.shape[1]
            if coefficient_shape_valid
            else 0
        )
        if (
            points.ndim != 2
            or points.shape[1:] != (3,)
            or not coefficient_shape_valid
            or coefficient_upper.shape != coefficient_lower.shape
            or not np.all(np.isfinite(points))
        ):
            raise RayClosureError(
                "pairwise exact-point affine inputs are malformed"
            )
        if count == 0:
            return (
                np.empty((0, form_count), dtype=np.float64),
                np.empty((0, form_count), dtype=np.float64),
            )
        homogeneous = np.empty((count, 4), dtype=np.float64)
        homogeneous[:, :3] = points
        homogeneous[:, 3] = 1.0
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            lower = np.zeros((count, form_count), dtype=np.float64)
            upper = np.zeros((count, form_count), dtype=np.float64)
            for column in range(4):
                position_column = homogeneous[:, column, None]
                product_lower_endpoint = (
                    coefficient_lower[:, :, column] * position_column
                )
                product_upper_endpoint = (
                    coefficient_upper[:, :, column] * position_column
                )
                product_lower = np.nextafter(
                    np.minimum(
                        product_lower_endpoint, product_upper_endpoint
                    ),
                    -math.inf,
                )
                product_upper = np.nextafter(
                    np.maximum(
                        product_lower_endpoint, product_upper_endpoint
                    ),
                    math.inf,
                )
                lower = np.nextafter(
                    lower + product_lower, -math.inf
                )
                upper = np.nextafter(
                    upper + product_upper, math.inf
                )
        finite = np.all(np.isfinite(lower) & np.isfinite(upper), axis=1)
        if not np.all(finite):
            lower[~finite] = -math.inf
            upper[~finite] = math.inf
        return lower, upper

    def _certified_chord_pairwise_affine_bounds_v9(
        self,
        *,
        start_positions_object_m: np.ndarray,
        end_positions_object_m: np.ndarray,
        tube_lower_object_m: np.ndarray,
        tube_upper_object_m: np.ndarray,
        coefficient_lower: np.ndarray,
        coefficient_upper: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Bound selected affine forms for one certified chord per row."""

        start = np.asarray(start_positions_object_m, dtype=np.float64)
        end = np.asarray(end_positions_object_m, dtype=np.float64)
        tube_lower = np.asarray(tube_lower_object_m, dtype=np.float64)
        tube_upper = np.asarray(tube_upper_object_m, dtype=np.float64)
        start_lower, start_upper = (
            self._outward_affine_form_bounds_at_points_pairwise(
                positions=start,
                coefficient_lower=coefficient_lower,
                coefficient_upper=coefficient_upper,
            )
        )
        end_lower, end_upper = (
            self._outward_affine_form_bounds_at_points_pairwise(
            positions=end,
            coefficient_lower=coefficient_lower,
            coefficient_upper=coefficient_upper,
        )
        )
        radius = np.maximum(
            np.nextafter(
                np.minimum(start, end) - tube_lower, math.inf
            ),
            np.nextafter(
                tube_upper - np.maximum(start, end), math.inf
            ),
        )
        radius = np.maximum(radius, 0.0)
        coefficient_magnitude = np.nextafter(
            np.maximum(
                np.abs(coefficient_lower[:, :, :3]),
                np.abs(coefficient_upper[:, :, :3]),
            ),
            math.inf,
        )
        deviation_terms = np.nextafter(
            coefficient_magnitude * radius[:, None, :], math.inf
        )
        deviation = deviation_terms[:, :, 0]
        for column in (1, 2):
            deviation = np.nextafter(
                deviation + deviation_terms[:, :, column], math.inf
            )
        lower = np.nextafter(
            np.minimum(start_lower, end_lower) - deviation,
            -math.inf,
        )
        upper = np.nextafter(
            np.maximum(start_upper, end_upper) + deviation,
            math.inf,
        )
        finite = np.all(np.isfinite(lower) & np.isfinite(upper), axis=1)
        if not np.all(finite):
            lower[~finite] = -math.inf
            upper[~finite] = math.inf
        return lower, upper

    def _certified_batch_free_face_mask_v9(
        self,
        *,
        position_lower: np.ndarray,
        position_upper: np.ndarray,
        face_indices: np.ndarray,
    ) -> np.ndarray:
        """Return only faces proven free by plane or triangle-edge bounds."""

        face_indices = np.asarray(face_indices, dtype=np.int64)
        if face_indices.ndim != 1:
            raise RayClosureError("batch face indices must be one-dimensional")
        if len(face_indices) == 0:
            return np.zeros(0, dtype=bool)
        if np.any(face_indices < 0) or np.any(
            face_indices >= len(self.canonical_object_face_vertices_m)
        ):
            raise RayClosureError("batch face index is outside object geometry")
        lower, upper = self._outward_affine_form_bounds(
            position_lower=position_lower,
            position_upper=position_upper,
            coefficient_lower=(
                self._object_contact_affine_lower[face_indices]
            ),
            coefficient_upper=(
                self._object_contact_affine_upper[face_indices]
            ),
        )
        plane_free = (lower[:, 0] > 0.0) | (upper[:, 0] < 0.0)
        triangle_free = np.any(upper[:, 1:] < 0.0, axis=1)
        result = plane_free | triangle_free
        result.setflags(write=False)
        return result

    @classmethod
    def _certified_chord_affine_form_bounds_v9(
        cls,
        *,
        start_position_object_m: np.ndarray,
        end_position_object_m: np.ndarray,
        tube_lower_object_m: np.ndarray,
        tube_upper_object_m: np.ndarray,
        coefficient_lower: np.ndarray,
        coefficient_upper: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Preserve chord coordinate correlation in affine face bounds."""

        start = np.asarray(start_position_object_m, dtype=np.float64)
        end = np.asarray(end_position_object_m, dtype=np.float64)
        tube_lower = np.asarray(tube_lower_object_m, dtype=np.float64)
        tube_upper = np.asarray(tube_upper_object_m, dtype=np.float64)
        if (
            start.shape != (3,)
            or end.shape != (3,)
            or tube_lower.shape != (3,)
            or tube_upper.shape != (3,)
            or not np.all(np.isfinite(start))
            or not np.all(np.isfinite(end))
            or not np.all(np.isfinite(tube_lower))
            or not np.all(np.isfinite(tube_upper))
            or np.any(tube_lower > np.minimum(start, end))
            or np.any(tube_upper < np.maximum(start, end))
        ):
            raise RayClosureError(
                "correlation-preserving chord bounds are malformed"
            )
        start_lower, start_upper = cls._outward_affine_form_bounds(
            position_lower=start,
            position_upper=start,
            coefficient_lower=coefficient_lower,
            coefficient_upper=coefficient_upper,
        )
        end_lower, end_upper = cls._outward_affine_form_bounds(
            position_lower=end,
            position_upper=end,
            coefficient_lower=coefficient_lower,
            coefficient_upper=coefficient_upper,
        )
        radius = np.maximum(
            np.nextafter(
                np.minimum(start, end) - tube_lower, math.inf
            ),
            np.nextafter(
                tube_upper - np.maximum(start, end), math.inf
            ),
        )
        radius = np.maximum(radius, 0.0)
        coefficient_magnitude = np.nextafter(
            np.maximum(
                np.abs(coefficient_lower[:, :, :3]),
                np.abs(coefficient_upper[:, :, :3]),
            ),
            math.inf,
        )
        deviation_terms = np.nextafter(
            coefficient_magnitude * radius, math.inf
        )
        deviation = deviation_terms[:, :, 0]
        for column in (1, 2):
            deviation = np.nextafter(
                deviation + deviation_terms[:, :, column], math.inf
            )
        lower = np.nextafter(
            np.minimum(start_lower, end_lower) - deviation,
            -math.inf,
        )
        upper = np.nextafter(
            np.maximum(start_upper, end_upper) + deviation,
            math.inf,
        )
        finite = np.all(np.isfinite(lower) & np.isfinite(upper), axis=1)
        if not np.all(finite):
            lower[~finite] = -math.inf
            upper[~finite] = math.inf
        return lower, upper

    def _certified_chord_batch_free_face_mask_v9(
        self,
        *,
        start_position_object_m: np.ndarray,
        end_position_object_m: np.ndarray,
        tube_lower_object_m: np.ndarray,
        tube_upper_object_m: np.ndarray,
        face_indices: np.ndarray,
    ) -> np.ndarray:
        """Cull faces using exact chord correlation plus curvature radius."""

        face_indices = np.asarray(face_indices, dtype=np.int64)
        if face_indices.ndim != 1:
            raise RayClosureError("batch face indices must be one-dimensional")
        if len(face_indices) == 0:
            return np.zeros(0, dtype=bool)
        if np.any(face_indices < 0) or np.any(
            face_indices >= len(self.canonical_object_face_vertices_m)
        ):
            raise RayClosureError("batch face index is outside object geometry")
        lower, upper = self._certified_chord_affine_form_bounds_v9(
            start_position_object_m=start_position_object_m,
            end_position_object_m=end_position_object_m,
            tube_lower_object_m=tube_lower_object_m,
            tube_upper_object_m=tube_upper_object_m,
            coefficient_lower=(
                self._object_contact_affine_lower[face_indices]
            ),
            coefficient_upper=(
                self._object_contact_affine_upper[face_indices]
            ),
        )
        plane_free = (lower[:, 0] > 0.0) | (upper[:, 0] < 0.0)
        triangle_free = np.any(upper[:, 1:] < 0.0, axis=1)
        result = plane_free | triangle_free
        result.setflags(write=False)
        return result

    def _certified_chord_pairwise_free_face_mask_v9(
        self,
        *,
        start_positions_object_m: np.ndarray,
        end_positions_object_m: np.ndarray,
        tube_lower_object_m: np.ndarray,
        tube_upper_object_m: np.ndarray,
        face_indices: np.ndarray,
    ) -> np.ndarray:
        """Cull a packet with one chord/tube and one face per row."""

        faces = np.asarray(face_indices, dtype=np.int64)
        start = np.asarray(start_positions_object_m, dtype=np.float64)
        end = np.asarray(end_positions_object_m, dtype=np.float64)
        tube_lower = np.asarray(tube_lower_object_m, dtype=np.float64)
        tube_upper = np.asarray(tube_upper_object_m, dtype=np.float64)
        count = len(faces)
        if (
            faces.ndim != 1
            or start.shape != (count, 3)
            or end.shape != start.shape
            or tube_lower.shape != start.shape
            or tube_upper.shape != start.shape
            or np.any(faces < 0)
            or np.any(faces >= len(self.canonical_object_face_vertices_m))
            or not np.all(np.isfinite(start))
            or not np.all(np.isfinite(end))
            or not np.all(np.isfinite(tube_lower))
            or not np.all(np.isfinite(tube_upper))
            or np.any(tube_lower > np.minimum(start, end))
            or np.any(tube_upper < np.maximum(start, end))
        ):
            raise RayClosureError(
                "pairwise chord culling inputs are malformed"
            )
        if count == 0:
            return np.zeros(0, dtype=bool)
        coefficient_lower = self._object_contact_affine_lower[faces]
        coefficient_upper = self._object_contact_affine_upper[faces]
        plane_lower, plane_upper = (
            self._certified_chord_pairwise_affine_bounds_v9(
                start_positions_object_m=start,
                end_positions_object_m=end,
                tube_lower_object_m=tube_lower,
                tube_upper_object_m=tube_upper,
                coefficient_lower=coefficient_lower[:, :1, :],
                coefficient_upper=coefficient_upper[:, :1, :],
            )
        )
        result = (plane_lower[:, 0] > 0.0) | (
            plane_upper[:, 0] < 0.0
        )
        unresolved = np.flatnonzero(~result)
        if len(unresolved) > 0:
            _triangle_lower, triangle_upper = (
                self._certified_chord_pairwise_affine_bounds_v9(
                    start_positions_object_m=start[unresolved],
                    end_positions_object_m=end[unresolved],
                    tube_lower_object_m=tube_lower[unresolved],
                    tube_upper_object_m=tube_upper[unresolved],
                    coefficient_lower=(
                        coefficient_lower[unresolved, 1:, :]
                    ),
                    coefficient_upper=(
                        coefficient_upper[unresolved, 1:, :]
                    ),
                )
            )
            result[unresolved] = np.any(triangle_upper < 0.0, axis=1)
        result.setflags(write=False)
        return result

    def _fast_plane_bucket_key_v9(
        self, object_face_index: int
    ) -> bytes:
        """Return a cheap orientation-free pre-bucket for exact splitting.

        This rounded key can create false-positive buckets only.  The exact
        dyadic key below is the sole authority for sharing a plane root.
        """

        if (
            not isinstance(object_face_index, int)
            or isinstance(object_face_index, bool)
            or object_face_index < 0
            or object_face_index
            >= len(self.canonical_object_face_vertices_m)
        ):
            raise RayClosureError("object face index is outside object geometry")
        cached = self._fast_plane_bucket_key_cache.get(object_face_index)
        if cached is not None:
            return cached
        triangle = np.asarray(
            self.canonical_object_face_vertices_m[object_face_index],
            dtype=np.float64,
        )
        normal = np.cross(
            triangle[1] - triangle[0], triangle[2] - triangle[0]
        )
        scale = float(np.max(np.abs(normal)))
        if scale == 0.0 or not math.isfinite(scale):
            raise RayClosureError("plane pre-bucket triangle is degenerate")
        coefficients = np.concatenate(
            (normal / scale, np.asarray((-np.dot(normal, triangle[0]) / scale,)))
        )
        leading = next(
            float(value) for value in coefficients[:3] if value != 0.0
        )
        if leading < 0.0:
            coefficients = -coefficients
        bucket = np.round(coefficients, decimals=10)
        bucket[bucket == 0.0] = 0.0
        key = np.ascontiguousarray(
            bucket, dtype=np.dtype(">f8")
        ).tobytes(order="C")
        self._fast_plane_bucket_key_cache[object_face_index] = key
        return key

    def _exact_plane_key_for_face_v9(
        self, object_face_index: int
    ) -> tuple[int, int, int, int]:
        """Lazily bind a face to its exact unoriented dyadic plane."""

        cached = self._exact_plane_key_cache.get(object_face_index)
        if cached is not None:
            return cached
        key = _exact_dyadic_plane_key(
            self.canonical_object_face_vertices_m[object_face_index]
        )
        self._exact_plane_key_cache[object_face_index] = key
        return key

    def _exact_plane_groups_v9(
        self, face_indices: np.ndarray
    ) -> tuple[np.ndarray, ...]:
        """Group faces only when their oriented plane predicate is identical.

        Binary64 coefficient bounds form a cheap first bucket.  Exact rational
        coefficients are constructed only for repeated buckets, so a large
        object mesh does not pay an eager Fraction-conversion cost.
        """

        faces = np.asarray(face_indices, dtype=np.int64)
        if faces.ndim != 1:
            raise RayClosureError("plane-group face indices must be one-dimensional")
        if np.any(faces < 0) or np.any(
            faces >= len(self.canonical_object_face_vertices_m)
        ):
            raise RayClosureError("plane-group face index is outside object geometry")
        if len(faces) == 0:
            return ()

        fast_keys = tuple(
            self._fast_plane_bucket_key_v9(int(face_index))
            for face_index in faces
        )
        fast_counts: dict[bytes, int] = {}
        for key in fast_keys:
            fast_counts[key] = fast_counts.get(key, 0) + 1

        group_rows: dict[
            tuple[
                bytes,
                tuple[int, int, int, int] | None,
            ],
            list[int],
        ] = {}
        group_order: list[
            tuple[
                bytes,
                tuple[int, int, int, int] | None,
            ]
        ] = []
        for face_index_value, fast_key in zip(faces, fast_keys):
            face_index = int(face_index_value)
            exact_key = (
                self._exact_plane_key_for_face_v9(face_index)
                if fast_counts[fast_key] > 1
                else None
            )
            group_key = (fast_key, exact_key)
            if group_key not in group_rows:
                group_rows[group_key] = []
                group_order.append(group_key)
            group_rows[group_key].append(face_index)

        result: list[np.ndarray] = []
        for group_key in group_order:
            row = np.asarray(group_rows[group_key], dtype=np.int64)
            row.setflags(write=False)
            result.append(row)
        return tuple(result)

    def _certified_root_batch_triangle_free_face_mask_v9(
        self,
        *,
        position_lower_object_m: np.ndarray,
        position_upper_object_m: np.ndarray,
        face_indices: np.ndarray,
    ) -> np.ndarray:
        """Prove root positions outside actual triangles in one array call."""

        faces = np.asarray(face_indices, dtype=np.int64)
        if faces.ndim != 1:
            raise RayClosureError("root-batch face indices must be one-dimensional")
        if len(faces) == 0:
            return np.zeros(0, dtype=bool)
        if np.any(faces < 0) or np.any(
            faces >= len(self.canonical_object_face_vertices_m)
        ):
            raise RayClosureError("root-batch face index is outside object geometry")
        _lower, upper = self._outward_affine_form_bounds(
            position_lower=np.asarray(
                position_lower_object_m, dtype=np.float64
            ),
            position_upper=np.asarray(
                position_upper_object_m, dtype=np.float64
            ),
            coefficient_lower=self._object_contact_affine_lower[faces],
            coefficient_upper=self._object_contact_affine_upper[faces],
        )
        result = np.any(upper[:, 1:] < 0.0, axis=1)
        result.setflags(write=False)
        return result

    def _filter_complete_swept_face_batches_v9(
        self,
        *,
        possible_witness_indices: np.ndarray,
        query_lower_object_m: np.ndarray,
        query_upper_object_m: np.ndarray,
        start_positions_object_m: np.ndarray,
        end_positions_object_m: np.ndarray,
        tube_lower_object_m: np.ndarray,
        tube_upper_object_m: np.ndarray,
        counters: _PadCounters,
        restricted_face_batches: tuple[np.ndarray, ...] | None = None,
        pair_cull_executor: ThreadPoolExecutor | None = None,
    ) -> tuple[np.ndarray, ...]:
        """Cull all witness-face pairs in fixed-size NumPy packets.

        The result preserves the witness-major, ascending-face order of the
        complete BVH query.  Packet sizing bounds temporary memory without
        changing the closed-AABB or outward-rounded certificate predicates.
        """

        possible_indices = np.asarray(
            possible_witness_indices, dtype=np.int64
        )
        start_positions = np.asarray(
            start_positions_object_m, dtype=np.float64
        )
        end_positions = np.asarray(
            end_positions_object_m, dtype=np.float64
        )
        query_lower = np.asarray(
            query_lower_object_m, dtype=np.float64
        )
        query_upper = np.asarray(
            query_upper_object_m, dtype=np.float64
        )
        tube_lower = np.asarray(tube_lower_object_m, dtype=np.float64)
        tube_upper = np.asarray(tube_upper_object_m, dtype=np.float64)
        row_count = len(possible_indices)
        if (
            possible_indices.ndim != 1
            or start_positions.ndim != 2
            or start_positions.shape[1:] != (3,)
            or end_positions.shape != start_positions.shape
            or query_lower.shape != (row_count, 3)
            or query_upper.shape != query_lower.shape
            or tube_lower.shape != (row_count, 3)
            or tube_upper.shape != tube_lower.shape
            or np.any(possible_indices < 0)
            or np.any(possible_indices >= len(start_positions))
            or not np.all(np.isfinite(start_positions))
            or not np.all(np.isfinite(end_positions))
            or not np.all(np.isfinite(query_lower))
            or not np.all(np.isfinite(query_upper))
            or not np.all(np.isfinite(tube_lower))
            or not np.all(np.isfinite(tube_upper))
            or np.any(query_lower > query_upper)
            or np.any(tube_lower > tube_upper)
        ):
            raise RayClosureError(
                "complete swept-face packet inputs are malformed"
            )
        if restricted_face_batches is not None:
            if len(restricted_face_batches) != row_count:
                raise RayClosureError(
                    "restricted swept-face rows do not match witnesses"
                )
            for row in restricted_face_batches:
                faces = np.asarray(row, dtype=np.int64)
                if (
                    faces.ndim != 1
                    or np.any(faces < 0)
                    or np.any(
                        faces
                        >= len(self.distance_bvh.face_lower_m)
                    )
                    or (
                        len(faces) > 1
                        and np.any(faces[1:] <= faces[:-1])
                    )
                ):
                    raise RayClosureError(
                        "restricted swept-face rows are not canonical"
                    )

        face_count = len(self.distance_bvh.face_lower_m)
        if row_count > np.iinfo(np.int64).max // face_count:
            raise RayClosureError("swept-face packed key would overflow")
        kept_key_chunks: list[np.ndarray] = []
        certified_free_count = 0
        uncertain_count = 0

        def restricted_pair_packets() -> Iterator[
            tuple[np.ndarray, np.ndarray]
        ]:
            assert restricted_face_batches is not None
            counts = np.fromiter(
                (len(row) for row in restricted_face_batches),
                dtype=np.int64,
                count=row_count,
            )
            pair_count = int(np.sum(counts, dtype=np.int64))
            if pair_count == 0:
                return
            flat_owners = np.repeat(
                np.arange(row_count, dtype=np.int64), counts
            )
            flat_faces = np.concatenate(restricted_face_batches).astype(
                np.int64, copy=False
            )
            centred_lower = np.nextafter(
                query_lower - self.distance_bvh.centre_m[None, :],
                -math.inf,
            )
            centred_upper = np.nextafter(
                query_upper - self.distance_bvh.centre_m[None, :],
                math.inf,
            )
            for packet_lower in range(
                0, pair_count, _EXACT_FACE_PAIR_PACKET_SIZE
            ):
                packet_upper = min(
                    packet_lower + _EXACT_FACE_PAIR_PACKET_SIZE,
                    pair_count,
                )
                owners = flat_owners[packet_lower:packet_upper]
                faces = flat_faces[packet_lower:packet_upper]
                overlap = np.all(
                    self.distance_bvh.face_upper_m[faces]
                    >= centred_lower[owners],
                    axis=1,
                ) & np.all(
                    self.distance_bvh.face_lower_m[faces]
                    <= centred_upper[owners],
                    axis=1,
                )
                if np.any(overlap):
                    yield owners[overlap], faces[overlap]

        pair_packets = (
            self.distance_bvh._iter_face_pairs_intersecting_aabbs(
                query_lower,
                query_upper,
                maximum_pair_count=_EXACT_FACE_PAIR_PACKET_SIZE,
            )
            if restricted_face_batches is None
            else restricted_pair_packets()
        )

        def evaluate_packet(
            packet: tuple[np.ndarray, np.ndarray],
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            owners, faces = packet
            witness_indices = possible_indices[owners]
            free_mask = self._certified_chord_pairwise_free_face_mask_v9(
                start_positions_object_m=start_positions[witness_indices],
                end_positions_object_m=end_positions[witness_indices],
                tube_lower_object_m=tube_lower[owners],
                tube_upper_object_m=tube_upper[owners],
                face_indices=faces,
            )
            return owners, faces, free_mask

        def commit_packet(
            packet_result: tuple[np.ndarray, np.ndarray, np.ndarray],
        ) -> None:
            nonlocal certified_free_count, uncertain_count
            owners, faces, free_mask = packet_result
            counters.swept_face_candidates += len(faces)
            packet_free_count = int(np.count_nonzero(free_mask))
            certified_free_count += packet_free_count
            uncertain_count += len(faces) - packet_free_count
            keep_mask = ~free_mask
            if np.any(keep_mask):
                kept_key_chunks.append(
                    owners[keep_mask] * face_count + faces[keep_mask]
                )

        if pair_cull_executor is None:
            for packet in pair_packets:
                commit_packet(evaluate_packet(packet))
        else:
            pending: deque[
                Future[tuple[np.ndarray, np.ndarray, np.ndarray]]
            ] = deque()
            for packet in pair_packets:
                pending.append(
                    pair_cull_executor.submit(evaluate_packet, packet)
                )
                if len(pending) >= _EXACT_FACE_PAIR_MAXIMUM_IN_FLIGHT:
                    commit_packet(pending.popleft().result())
            while pending:
                commit_packet(pending.popleft().result())

        counters.certified_batch_linear_free_pairs += certified_free_count
        counters.batch_linear_uncertain_pairs += uncertain_count
        if not kept_key_chunks:
            return tuple(
                _readonly_int64(np.empty(0, dtype=np.int64))
                for _index in range(row_count)
            )
        kept_keys = np.concatenate(kept_key_chunks)
        kept_keys.sort(kind="quicksort")
        kept_owners = kept_keys // face_count
        kept_faces = kept_keys % face_count
        kept_counts = np.bincount(kept_owners, minlength=row_count)
        offsets = np.concatenate(
            (
                np.asarray((0,), dtype=np.int64),
                np.cumsum(kept_counts, dtype=np.int64),
            )
        )
        kept_faces.setflags(write=False)
        result: list[np.ndarray] = []
        for row_index in range(row_count):
            row = kept_faces[offsets[row_index] : offsets[row_index + 1]]
            row.setflags(write=False)
            result.append(row)
        return tuple(result)

    def _iter_complete_swept_face_batches_v9(
        self,
        *,
        prepared: _PreparedPad,
        possible_witness_indices: np.ndarray,
        q_start: np.ndarray,
        direction: np.ndarray,
        lower: float,
        upper: float,
        object_from_hand: np.ndarray,
        counters: _PadCounters,
        transform_cache: IntervalLinkTransformCache | None,
        apply_certified_batch_cull: bool = False,
        witness_start_positions_object_m: np.ndarray | None = None,
        witness_end_positions_object_m: np.ndarray | None = None,
        endpoint_error_bound_m: float = 0.0,
        parent_face_frontier: Mapping[int, np.ndarray] | None = None,
        pair_cull_executor: ThreadPoolExecutor | None = None,
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Yield one deterministic complete face batch per possible witness.

        The caller may stop after an unresolved pair because that one pair is
        already sufficient to force conservative temporal subdivision.  An
        accepted or certified-free interval still exhausts this iterator and
        therefore evaluates the complete pair set.
        """

        possible_indices = np.asarray(
            possible_witness_indices, dtype=np.int64
        )
        if (
            possible_indices.ndim != 1
            or np.any(possible_indices < 0)
            or np.any(
                possible_indices >= len(prepared.witness_points_link_m)
            )
        ):
            raise RayClosureError(
                "possible witness indices must address prepared witnesses"
            )
        restricted_face_batches: tuple[np.ndarray, ...] | None = None
        if parent_face_frontier is not None:
            restricted_indices: list[int] = []
            restricted_rows: list[np.ndarray] = []
            for witness_index_value in possible_indices:
                witness_index = int(witness_index_value)
                row = parent_face_frontier.get(witness_index)
                if row is None or len(row) == 0:
                    continue
                restricted_indices.append(witness_index)
                restricted_rows.append(np.asarray(row, dtype=np.int64))
            possible_indices = np.asarray(
                restricted_indices, dtype=np.int64
            )
            restricted_face_batches = tuple(restricted_rows)

        if apply_certified_batch_cull:
            start_positions = np.asarray(
                witness_start_positions_object_m, dtype=np.float64
            )
            end_positions = np.asarray(
                witness_end_positions_object_m, dtype=np.float64
            )
            expected_shape = prepared.witness_points_link_m.shape
            if (
                start_positions.shape != expected_shape
                or end_positions.shape != expected_shape
                or not np.all(np.isfinite(start_positions))
                or not np.all(np.isfinite(end_positions))
            ):
                raise RayClosureError(
                    "certified batch cull requires all endpoint witnesses"
                )
            cursor = 0
            stage_size = _SWEPT_FACE_INITIAL_WITNESS_STAGE_SIZE
            while cursor < len(possible_indices):
                stage_upper = min(cursor + stage_size, len(possible_indices))
                stage_indices = possible_indices[cursor:stage_upper]
                stage_restricted = (
                    None
                    if restricted_face_batches is None
                    else restricted_face_batches[cursor:stage_upper]
                )
                try:
                    stage_motion = self.interval_kinematics.point_motion_many(
                        link_name=prepared.verified.link_name,
                        q_start=q_start,
                        direction=direction,
                        phase_lower=lower,
                        phase_upper=upper,
                        base_transform=object_from_hand,
                        points_local_m=(
                            prepared.witness_points_link_m[stage_indices]
                        ),
                        transform_cache=transform_cache,
                    )
                except IntervalKinematicsError as error:
                    raise RayClosureError(
                        "batch interval point-motion broadphase rejected the "
                        f"mechanical path: {error}"
                    ) from error
                counters.interval_point_motion_evaluations += len(stage_indices)
                stage_tube_lower, stage_tube_upper = (
                    self._certified_chord_tube_position_bounds_many_v9(
                        motion=stage_motion,
                        start_positions_object_m=start_positions[stage_indices],
                        end_positions_object_m=end_positions[stage_indices],
                        phase_lower=lower,
                        phase_upper=upper,
                        endpoint_error_bound_m=endpoint_error_bound_m,
                    )
                )
                tight_query_lower = np.maximum(
                    stage_motion.position_lower_object_m, stage_tube_lower
                )
                tight_query_upper = np.minimum(
                    stage_motion.position_upper_object_m, stage_tube_upper
                )
                if np.any(tight_query_lower > tight_query_upper):
                    raise RayClosureError(
                        "certified interval and chord tube bounds do not "
                        "intersect"
                    )
                stage_face_indices = (
                    self._filter_complete_swept_face_batches_v9(
                        possible_witness_indices=stage_indices,
                        query_lower_object_m=tight_query_lower,
                        query_upper_object_m=tight_query_upper,
                        start_positions_object_m=start_positions,
                        end_positions_object_m=end_positions,
                        tube_lower_object_m=stage_tube_lower,
                        tube_upper_object_m=stage_tube_upper,
                        counters=counters,
                        restricted_face_batches=stage_restricted,
                        pair_cull_executor=pair_cull_executor,
                    )
                )
                counters.swept_face_witness_stages += 1
                counters.swept_face_witnesses_materialized += len(
                    stage_indices
                )
                for witness_index_value, face_indices in zip(
                    stage_indices, stage_face_indices
                ):
                    if len(face_indices) == 0:
                        continue
                    yield int(witness_index_value), face_indices
                cursor = stage_upper
                stage_size = min(
                    _SWEPT_FACE_MAXIMUM_WITNESS_STAGE_SIZE,
                    stage_size * 2,
                )
            return

        for witness_index_value in possible_indices:
            witness_index = int(witness_index_value)
            try:
                motion = self.interval_kinematics.point_motion(
                    link_name=prepared.verified.link_name,
                    q_start=q_start,
                    direction=direction,
                    phase_lower=lower,
                    phase_upper=upper,
                    base_transform=object_from_hand,
                    point_local_m=(
                        prepared.witness_points_link_m[witness_index]
                    ),
                    transform_cache=transform_cache,
                )
            except IntervalKinematicsError as error:
                raise RayClosureError(
                    "interval point-motion broadphase rejected the "
                    f"mechanical path: {error}"
                ) from error
            counters.interval_point_motion_evaluations += 1
            position_lower = np.asarray(
                [row.lower for row in motion.position_object_m],
                dtype=np.float64,
            )
            position_upper = np.asarray(
                [row.upper for row in motion.position_object_m],
                dtype=np.float64,
            )
            face_indices = self.distance_bvh.face_indices_intersecting_aabb(
                position_lower, position_upper
            )
            counters.swept_face_candidates += len(face_indices)
            face_indices = np.asarray(face_indices, dtype=np.int64)
            if len(face_indices) == 0:
                continue
            face_indices.setflags(write=False)
            yield witness_index, face_indices

    def _iter_complete_swept_face_pairs_v9(
        self,
        *,
        prepared: _PreparedPad,
        possible_witness_indices: np.ndarray,
        q_start: np.ndarray,
        direction: np.ndarray,
        lower: float,
        upper: float,
        object_from_hand: np.ndarray,
        counters: _PadCounters,
        transform_cache: IntervalLinkTransformCache | None,
        apply_certified_batch_cull: bool = False,
        witness_start_positions_object_m: np.ndarray | None = None,
        witness_end_positions_object_m: np.ndarray | None = None,
        endpoint_error_bound_m: float = 0.0,
    ) -> Iterator[tuple[int, int]]:
        """Compatibility iterator that flattens complete witness batches."""

        for witness_index, face_indices in (
            self._iter_complete_swept_face_batches_v9(
                prepared=prepared,
                possible_witness_indices=possible_witness_indices,
                q_start=q_start,
                direction=direction,
                lower=lower,
                upper=upper,
                object_from_hand=object_from_hand,
                counters=counters,
                transform_cache=transform_cache,
                apply_certified_batch_cull=apply_certified_batch_cull,
                witness_start_positions_object_m=(
                    witness_start_positions_object_m
                ),
                witness_end_positions_object_m=(
                    witness_end_positions_object_m
                ),
                endpoint_error_bound_m=endpoint_error_bound_m,
            )
        ):
            for face_index_value in face_indices:
                yield witness_index, int(face_index_value)

    def _complete_swept_face_pairs_v9(
        self,
        *,
        prepared: _PreparedPad,
        possible_witness_indices: np.ndarray,
        q_start: np.ndarray,
        direction: np.ndarray,
        lower: float,
        upper: float,
        object_from_hand: np.ndarray,
        counters: _PadCounters,
        transform_cache: IntervalLinkTransformCache | None,
    ) -> tuple[tuple[int, int], ...]:
        """Materialise the complete pair set for diagnostics and tests."""

        return tuple(
            sorted(
                set(
                    self._iter_complete_swept_face_pairs_v9(
                        prepared=prepared,
                        possible_witness_indices=possible_witness_indices,
                        q_start=q_start,
                        direction=direction,
                        lower=lower,
                        upper=upper,
                        object_from_hand=object_from_hand,
                        counters=counters,
                        transform_cache=transform_cache,
                    )
                )
            )
        )

    def _classify_witness_face_pair_v9(
        self,
        *,
        prepared: _PreparedPad,
        witness_flat_index: int,
        object_face_index: int,
        q_start: np.ndarray,
        direction: np.ndarray,
        lower: float,
        upper: float,
        object_from_hand: np.ndarray,
        counters: _PadCounters,
        transform_cache: IntervalLinkTransformCache | None,
    ) -> _PairIntervalClassification:
        triangle_index = int(
            prepared.triangle_indices[witness_flat_index]
        )
        pad_face = np.asarray(
            prepared.verified.faces[triangle_index], dtype=np.int64
        )
        pad_triangle = np.asarray(
            prepared.verified.points_local_m[pad_face], dtype=np.float64
        )
        object_triangle = np.asarray(
            self.canonical_object_face_vertices_m[object_face_index],
            dtype=np.float64,
        )
        counters.interval_pair_evaluations += 1
        try:
            row = self.interval_kinematics.certify_transverse_contact_root(
                link_name=prepared.verified.link_name,
                q_start=q_start,
                direction=direction,
                phase_lower=lower,
                phase_upper=upper,
                base_transform=object_from_hand,
                witness_point_local_m=(
                    prepared.witness_points_link_m[witness_flat_index]
                ),
                pad_triangle_local_m=pad_triangle,
                object_triangle_m=object_triangle,
                transform_cache=transform_cache,
            )
        except IntervalKinematicsError as error:
            counters.unresolved_witness_face_pairs += 1
            return _PairIntervalClassification(
                state=_PadSearchState.UNRESOLVED,
                witness_flat_index=witness_flat_index,
                object_face_index=object_face_index,
                possible_phase_lower=lower,
                root=None,
                reason=f"INTERVAL_BACKEND_REJECTED:{error}",
            )
        return self._bind_interval_root_classification_v9(
            prepared=prepared,
            witness_flat_index=witness_flat_index,
            object_face_index=object_face_index,
            row=row,
            lower=lower,
            counters=counters,
        )

    def _new_exact_plane_root_worker_backend_v9(
        self,
    ) -> DirectedIntervalKinematics:
        """Create one interval backend owned by exactly one worker thread."""

        return DirectedIntervalKinematics(
            self.hand_model,
            self.interval_arithmetic_options,
        )

    @staticmethod
    def _certified_monotone_root_position_enclosures_v9(
        *,
        phase_lower: float,
        phase_upper: float,
        derivative_lower: np.ndarray,
        derivative_upper: np.ndarray,
        lower_value_lower: np.ndarray,
        lower_value_upper: np.ndarray,
        upper_value_lower: np.ndarray,
        upper_value_upper: np.ndarray,
        whole_motion: IntervalPointMotionBatch,
        lower_motion: IntervalPointMotionBatch,
        upper_motion: IntervalPointMotionBatch,
        eligible: np.ndarray,
    ) -> _PreRootSpatialEnclosureBatch:
        """Enclose monotone plane roots and their points without solving them.

        The phase bound follows directly from the mean-value theorem.  The
        point box integrates the whole-segment coordinate velocity enclosure
        once from each endpoint, then intersects both results with the whole
        position enclosure.  Every binary64 operation is expanded outward.
        """

        derivative_lower = np.asarray(derivative_lower, dtype=np.float64)
        derivative_upper = np.asarray(derivative_upper, dtype=np.float64)
        lower_value_lower = np.asarray(
            lower_value_lower, dtype=np.float64
        )
        lower_value_upper = np.asarray(
            lower_value_upper, dtype=np.float64
        )
        upper_value_lower = np.asarray(
            upper_value_lower, dtype=np.float64
        )
        upper_value_upper = np.asarray(
            upper_value_upper, dtype=np.float64
        )
        eligible = np.asarray(eligible, dtype=bool)
        count = len(eligible)
        scalar_rows = (
            derivative_lower,
            derivative_upper,
            lower_value_lower,
            lower_value_upper,
            upper_value_lower,
            upper_value_upper,
        )
        motion_rows = (
            whole_motion.position_lower_object_m,
            whole_motion.position_upper_object_m,
            whole_motion.velocity_lower_object_m_per_unit,
            whole_motion.velocity_upper_object_m_per_unit,
            lower_motion.position_lower_object_m,
            lower_motion.position_upper_object_m,
            upper_motion.position_lower_object_m,
            upper_motion.position_upper_object_m,
        )
        if (
            not math.isfinite(phase_lower)
            or not math.isfinite(phase_upper)
            or phase_lower >= phase_upper
            or any(row.shape != (count,) for row in scalar_rows)
            or any(row.shape != (count, 3) for row in motion_rows)
            or not all(np.all(np.isfinite(row)) for row in scalar_rows)
            or not all(np.all(np.isfinite(row)) for row in motion_rows)
        ):
            raise RayClosureError(
                "pre-root spatial enclosure inputs are malformed"
            )

        valid = np.array(eligible, copy=True)
        increasing = derivative_lower > 0.0
        decreasing = derivative_upper < 0.0
        valid &= increasing | decreasing
        sign = np.where(increasing, 1.0, -1.0)
        oriented_derivative_lower = np.where(
            increasing, derivative_lower, -derivative_upper
        )
        oriented_derivative_upper = np.where(
            increasing, derivative_upper, -derivative_lower
        )
        oriented_lower_value_lower = np.where(
            increasing, lower_value_lower, -lower_value_upper
        )
        oriented_lower_value_upper = np.where(
            increasing, lower_value_upper, -lower_value_lower
        )
        oriented_upper_value_lower = np.where(
            increasing, upper_value_lower, -upper_value_upper
        )
        oriented_upper_value_upper = np.where(
            increasing, upper_value_upper, -upper_value_lower
        )
        del sign
        valid &= (
            (oriented_derivative_lower > 0.0)
            & (oriented_lower_value_upper < 0.0)
            & (oriented_upper_value_lower > 0.0)
        )

        root_phase_lower = np.full(count, phase_lower, dtype=np.float64)
        root_phase_upper = np.full(count, phase_upper, dtype=np.float64)
        position_lower = np.array(
            whole_motion.position_lower_object_m, copy=True
        )
        position_upper = np.array(
            whole_motion.position_upper_object_m, copy=True
        )
        active = np.flatnonzero(valid)
        if len(active) > 0:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                start_delta_lower = np.nextafter(
                    -oriented_lower_value_upper[active]
                    / oriented_derivative_upper[active],
                    -math.inf,
                )
                start_delta_upper = np.nextafter(
                    -oriented_lower_value_lower[active]
                    / oriented_derivative_lower[active],
                    math.inf,
                )
                end_delta_lower = np.nextafter(
                    oriented_upper_value_lower[active]
                    / oriented_derivative_upper[active],
                    -math.inf,
                )
                end_delta_upper = np.nextafter(
                    oriented_upper_value_upper[active]
                    / oriented_derivative_lower[active],
                    math.inf,
                )
                from_start_lower = np.nextafter(
                    phase_lower + start_delta_lower, -math.inf
                )
                from_start_upper = np.nextafter(
                    phase_lower + start_delta_upper, math.inf
                )
                from_end_lower = np.nextafter(
                    phase_upper - end_delta_upper, -math.inf
                )
                from_end_upper = np.nextafter(
                    phase_upper - end_delta_lower, math.inf
                )
            root_phase_lower[active] = np.maximum(
                phase_lower,
                np.maximum(from_start_lower, from_end_lower),
            )
            root_phase_upper[active] = np.minimum(
                phase_upper,
                np.minimum(from_start_upper, from_end_upper),
            )
            ordered = (
                np.isfinite(root_phase_lower[active])
                & np.isfinite(root_phase_upper[active])
                & (root_phase_lower[active] <= root_phase_upper[active])
            )
            valid[active[~ordered]] = False
            active = active[ordered]

        def displacement_bounds(
            velocity_lower: np.ndarray,
            velocity_upper: np.ndarray,
            duration_lower: np.ndarray,
            duration_upper: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
            products = np.stack(
                (
                    velocity_lower * duration_lower[:, None],
                    velocity_lower * duration_upper[:, None],
                    velocity_upper * duration_lower[:, None],
                    velocity_upper * duration_upper[:, None],
                ),
                axis=0,
            )
            return (
                np.nextafter(np.min(products, axis=0), -math.inf),
                np.nextafter(np.max(products, axis=0), math.inf),
            )

        if len(active) > 0:
            width = phase_upper - phase_lower
            forward_duration_lower = np.maximum(
                0.0,
                np.nextafter(
                    root_phase_lower[active] - phase_lower, -math.inf
                ),
            )
            forward_duration_upper = np.minimum(
                width,
                np.nextafter(
                    root_phase_upper[active] - phase_lower, math.inf
                ),
            )
            backward_duration_lower = np.maximum(
                0.0,
                np.nextafter(
                    phase_upper - root_phase_upper[active], -math.inf
                ),
            )
            backward_duration_upper = np.minimum(
                width,
                np.nextafter(
                    phase_upper - root_phase_lower[active], math.inf
                ),
            )
            velocity_lower = (
                whole_motion.velocity_lower_object_m_per_unit[active]
            )
            velocity_upper = (
                whole_motion.velocity_upper_object_m_per_unit[active]
            )
            forward_displacement_lower, forward_displacement_upper = (
                displacement_bounds(
                    velocity_lower,
                    velocity_upper,
                    forward_duration_lower,
                    forward_duration_upper,
                )
            )
            backward_displacement_lower, backward_displacement_upper = (
                displacement_bounds(
                    velocity_lower,
                    velocity_upper,
                    backward_duration_lower,
                    backward_duration_upper,
                )
            )
            from_lower_position_lower = np.nextafter(
                lower_motion.position_lower_object_m[active]
                + forward_displacement_lower,
                -math.inf,
            )
            from_lower_position_upper = np.nextafter(
                lower_motion.position_upper_object_m[active]
                + forward_displacement_upper,
                math.inf,
            )
            from_upper_position_lower = np.nextafter(
                upper_motion.position_lower_object_m[active]
                - backward_displacement_upper,
                -math.inf,
            )
            from_upper_position_upper = np.nextafter(
                upper_motion.position_upper_object_m[active]
                - backward_displacement_lower,
                math.inf,
            )
            position_lower[active] = np.maximum.reduce(
                (
                    whole_motion.position_lower_object_m[active],
                    from_lower_position_lower,
                    from_upper_position_lower,
                )
            )
            position_upper[active] = np.minimum.reduce(
                (
                    whole_motion.position_upper_object_m[active],
                    from_lower_position_upper,
                    from_upper_position_upper,
                )
            )
            position_ordered = np.all(
                np.isfinite(position_lower[active])
                & np.isfinite(position_upper[active])
                & (position_lower[active] <= position_upper[active]),
                axis=1,
            )
            valid[active[~position_ordered]] = False

        for value in (
            valid,
            root_phase_lower,
            root_phase_upper,
            position_lower,
            position_upper,
        ):
            value.setflags(write=False)
        return _PreRootSpatialEnclosureBatch(
            valid=valid,
            phase_lower=root_phase_lower,
            phase_upper=root_phase_upper,
            position_lower_object_m=position_lower,
            position_upper_object_m=position_upper,
        )

    @staticmethod
    def _certified_affine_values_at_root_from_endpoints_v9(
        *,
        phase_lower: float,
        phase_upper: float,
        root_phase_lower: np.ndarray,
        root_phase_upper: np.ndarray,
        lower_value_lower: np.ndarray,
        lower_value_upper: np.ndarray,
        upper_value_lower: np.ndarray,
        upper_value_upper: np.ndarray,
        derivative_lower: np.ndarray,
        derivative_upper: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Bound affine values at root phases by integrating from both ends."""

        root_phase_lower = np.asarray(root_phase_lower, dtype=np.float64)
        root_phase_upper = np.asarray(root_phase_upper, dtype=np.float64)
        lower_value_lower = np.asarray(lower_value_lower, dtype=np.float64)
        lower_value_upper = np.asarray(lower_value_upper, dtype=np.float64)
        upper_value_lower = np.asarray(upper_value_lower, dtype=np.float64)
        upper_value_upper = np.asarray(upper_value_upper, dtype=np.float64)
        derivative_lower = np.asarray(derivative_lower, dtype=np.float64)
        derivative_upper = np.asarray(derivative_upper, dtype=np.float64)
        count = len(root_phase_lower)
        shape = lower_value_lower.shape
        if (
            not math.isfinite(phase_lower)
            or not math.isfinite(phase_upper)
            or phase_lower >= phase_upper
            or root_phase_lower.shape != (count,)
            or root_phase_upper.shape != (count,)
            or len(shape) != 2
            or shape[0] != count
            or shape[1] == 0
            or any(
                row.shape != shape
                for row in (
                    lower_value_upper,
                    upper_value_lower,
                    upper_value_upper,
                    derivative_lower,
                    derivative_upper,
                )
            )
            or not np.all(np.isfinite(root_phase_lower))
            or not np.all(np.isfinite(root_phase_upper))
            or root_phase_lower.size > 0
            and (
                np.any(root_phase_lower < phase_lower)
                or np.any(root_phase_upper > phase_upper)
                or np.any(root_phase_lower > root_phase_upper)
            )
            or not all(
                np.all(np.isfinite(row))
                for row in (
                    lower_value_lower,
                    lower_value_upper,
                    upper_value_lower,
                    upper_value_upper,
                    derivative_lower,
                    derivative_upper,
                )
            )
            or np.any(lower_value_lower > lower_value_upper)
            or np.any(upper_value_lower > upper_value_upper)
            or np.any(derivative_lower > derivative_upper)
        ):
            raise RayClosureError(
                "direct affine root enclosure inputs are malformed"
            )
        if count == 0:
            return np.empty(shape, dtype=np.float64), np.empty(
                shape, dtype=np.float64
            )
        width = phase_upper - phase_lower
        forward_duration_lower = np.maximum(
            0.0,
            np.nextafter(root_phase_lower - phase_lower, -math.inf),
        )
        forward_duration_upper = np.minimum(
            width,
            np.nextafter(root_phase_upper - phase_lower, math.inf),
        )
        backward_duration_lower = np.maximum(
            0.0,
            np.nextafter(phase_upper - root_phase_upper, -math.inf),
        )
        backward_duration_upper = np.minimum(
            width,
            np.nextafter(phase_upper - root_phase_lower, math.inf),
        )

        def displacement(
            duration_lower: np.ndarray,
            duration_upper: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
            products = np.stack(
                (
                    derivative_lower * duration_lower[:, None],
                    derivative_lower * duration_upper[:, None],
                    derivative_upper * duration_lower[:, None],
                    derivative_upper * duration_upper[:, None],
                ),
                axis=0,
            )
            return (
                np.nextafter(np.min(products, axis=0), -math.inf),
                np.nextafter(np.max(products, axis=0), math.inf),
            )

        forward_lower, forward_upper = displacement(
            forward_duration_lower, forward_duration_upper
        )
        backward_lower, backward_upper = displacement(
            backward_duration_lower, backward_duration_upper
        )
        from_lower_lower = np.nextafter(
            lower_value_lower + forward_lower, -math.inf
        )
        from_lower_upper = np.nextafter(
            lower_value_upper + forward_upper, math.inf
        )
        from_upper_lower = np.nextafter(
            upper_value_lower - backward_upper, -math.inf
        )
        from_upper_upper = np.nextafter(
            upper_value_upper - backward_lower, math.inf
        )
        result_lower = np.maximum(from_lower_lower, from_upper_lower)
        result_upper = np.minimum(from_lower_upper, from_upper_upper)
        ordered = np.all(
            np.isfinite(result_lower)
            & np.isfinite(result_upper)
            & (result_lower <= result_upper),
            axis=1,
        )
        result_lower[~ordered] = -math.inf
        result_upper[~ordered] = math.inf
        result_lower.setflags(write=False)
        result_upper.setflags(write=False)
        return result_lower, result_upper

    @staticmethod
    def _certified_second_order_affine_chord_root_bounds_v9(
        *,
        phase_lower: float,
        phase_upper: float,
        root_phase_lower: np.ndarray,
        root_phase_upper: np.ndarray,
        lower_value_lower: np.ndarray,
        lower_value_upper: np.ndarray,
        upper_value_lower: np.ndarray,
        upper_value_upper: np.ndarray,
        second_derivative_lower: np.ndarray,
        second_derivative_upper: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Enclose root-local chord values plus rigorous curvature error."""

        root_phase_lower = np.asarray(root_phase_lower, dtype=np.float64)
        root_phase_upper = np.asarray(root_phase_upper, dtype=np.float64)
        lower_value_lower = np.asarray(lower_value_lower, dtype=np.float64)
        lower_value_upper = np.asarray(lower_value_upper, dtype=np.float64)
        upper_value_lower = np.asarray(upper_value_lower, dtype=np.float64)
        upper_value_upper = np.asarray(upper_value_upper, dtype=np.float64)
        second_derivative_lower = np.asarray(
            second_derivative_lower, dtype=np.float64
        )
        second_derivative_upper = np.asarray(
            second_derivative_upper, dtype=np.float64
        )
        count = len(root_phase_lower)
        shape = lower_value_lower.shape
        rows = (
            lower_value_lower,
            lower_value_upper,
            upper_value_lower,
            upper_value_upper,
            second_derivative_lower,
            second_derivative_upper,
        )
        if (
            not math.isfinite(phase_lower)
            or not math.isfinite(phase_upper)
            or phase_lower >= phase_upper
            or root_phase_lower.shape != (count,)
            or root_phase_upper.shape != (count,)
            or len(shape) != 2
            or shape[0] != count
            or shape[1] == 0
            or any(row.shape != shape for row in rows[1:])
            or not np.all(np.isfinite(root_phase_lower))
            or not np.all(np.isfinite(root_phase_upper))
            or not all(np.all(np.isfinite(row)) for row in rows)
            or np.any(root_phase_lower < phase_lower)
            or np.any(root_phase_upper > phase_upper)
            or np.any(root_phase_lower > root_phase_upper)
            or np.any(lower_value_lower > lower_value_upper)
            or np.any(upper_value_lower > upper_value_upper)
            or np.any(second_derivative_lower > second_derivative_upper)
        ):
            raise RayClosureError(
                "second-order affine chord inputs are malformed"
            )
        if count == 0:
            return np.empty(shape, dtype=np.float64), np.empty(
                shape, dtype=np.float64
            )
        width = phase_upper - phase_lower
        alpha_lower = np.maximum(
            0.0,
            np.nextafter(
                (root_phase_lower - phase_lower) / width, -math.inf
            ),
        )
        alpha_upper = np.minimum(
            1.0,
            np.nextafter(
                (root_phase_upper - phase_lower) / width, math.inf
            ),
        )

        chord_lowers: list[np.ndarray] = []
        chord_uppers: list[np.ndarray] = []
        for alpha in (alpha_lower, alpha_upper):
            one_minus_alpha_lower = np.nextafter(
                1.0 - alpha, -math.inf
            )
            one_minus_alpha_upper = np.nextafter(
                1.0 - alpha, math.inf
            )
            for start_value in (lower_value_lower, lower_value_upper):
                for end_value in (upper_value_lower, upper_value_upper):
                    start_products = np.stack(
                        (
                            one_minus_alpha_lower[:, None] * start_value,
                            one_minus_alpha_upper[:, None] * start_value,
                        ),
                        axis=0,
                    )
                    end_product = alpha[:, None] * end_value
                    chord_lowers.append(
                        np.nextafter(
                            np.nextafter(
                                np.min(start_products, axis=0), -math.inf
                            )
                            + np.nextafter(end_product, -math.inf),
                            -math.inf,
                        )
                    )
                    chord_uppers.append(
                        np.nextafter(
                            np.nextafter(
                                np.max(start_products, axis=0), math.inf
                            )
                            + np.nextafter(end_product, math.inf),
                            math.inf,
                        )
                    )
        chord_lower = np.min(np.stack(chord_lowers, axis=0), axis=0)
        chord_upper = np.max(np.stack(chord_uppers, axis=0), axis=0)

        maximum_alpha_product_location = np.minimum(
            alpha_upper, np.maximum(alpha_lower, 0.5)
        )
        one_minus_location = np.nextafter(
            1.0 - maximum_alpha_product_location, math.inf
        )
        alpha_product_upper = np.nextafter(
            maximum_alpha_product_location * one_minus_location,
            math.inf,
        )
        second_derivative_magnitude = np.nextafter(
            np.maximum(
                np.abs(second_derivative_lower),
                np.abs(second_derivative_upper),
            ),
            math.inf,
        )
        width_squared = np.nextafter(width * width, math.inf)
        curvature_radius = np.nextafter(
            np.nextafter(
                second_derivative_magnitude * width_squared, math.inf
            )
            * alpha_product_upper[:, None],
            math.inf,
        )
        curvature_radius = np.nextafter(
            curvature_radius / 2.0, math.inf
        )
        result_lower = np.nextafter(
            chord_lower - curvature_radius, -math.inf
        )
        result_upper = np.nextafter(
            chord_upper + curvature_radius, math.inf
        )
        ordered = np.all(
            np.isfinite(result_lower)
            & np.isfinite(result_upper)
            & (result_lower <= result_upper),
            axis=1,
        )
        result_lower[~ordered] = -math.inf
        result_upper[~ordered] = math.inf
        result_lower.setflags(write=False)
        result_upper.setflags(write=False)
        return result_lower, result_upper

    def _batch_precertified_plane_gates_v9(
        self,
        *,
        prepared: _PreparedPad,
        work: Sequence[_PlaneRootGroupWork],
        q_start: np.ndarray,
        direction: np.ndarray,
        lower: float,
        upper: float,
        object_from_hand: np.ndarray,
        counters: _PadCounters,
        transform_cache: IntervalLinkTransformCache | None,
    ) -> tuple[
        np.ndarray,
        tuple[_PrecertifiedPlaneGate | None, ...],
        tuple[frozenset[int], ...],
    ]:
        """Share one rigorous endpoint/velocity calculation across all planes.

        A strictly monotone plane whose certified endpoints have the same
        sign is root-free.  Opposite, direction-consistent endpoint signs are
        passed to the compiled root transaction, which therefore need not
        rebuild the same whole-segment motion and endpoint predicates.
        Everything else falls back to the original exact per-plane path.
        """

        rows = tuple(work)
        count = len(rows)
        if count == 0:
            empty = np.empty(0, dtype=bool)
            empty.setflags(write=False)
            return empty, (), ()
        witness_indices = np.fromiter(
            (row.witness_flat_index for row in rows),
            dtype=np.int64,
            count=count,
        )
        representative_faces = np.fromiter(
            (row.representative_face_index for row in rows),
            dtype=np.int64,
            count=count,
        )
        points = prepared.witness_points_link_m[witness_indices]
        try:
            whole_motion = self.interval_kinematics.point_motion_many(
                link_name=prepared.verified.link_name,
                q_start=q_start,
                direction=direction,
                phase_lower=lower,
                phase_upper=upper,
                base_transform=object_from_hand,
                points_local_m=points,
                transform_cache=transform_cache,
            )
            lower_motion = self.interval_kinematics.point_motion_many(
                link_name=prepared.verified.link_name,
                q_start=q_start,
                direction=direction,
                phase_lower=lower,
                phase_upper=lower,
                base_transform=object_from_hand,
                points_local_m=points,
                transform_cache=transform_cache,
            )
            upper_motion = self.interval_kinematics.point_motion_many(
                link_name=prepared.verified.link_name,
                q_start=q_start,
                direction=direction,
                phase_lower=upper,
                phase_upper=upper,
                base_transform=object_from_hand,
                points_local_m=points,
                transform_cache=transform_cache,
            )
        except IntervalKinematicsError:
            no_free = np.zeros(count, dtype=bool)
            no_free.setflags(write=False)
            return no_free, (None,) * count, (frozenset(),) * count
        counters.interval_point_motion_evaluations += 3 * count

        coefficient_lower = np.array(
            self._object_contact_affine_lower[
                representative_faces, :1, :
            ],
            copy=True,
        )
        coefficient_upper = np.array(
            self._object_contact_affine_upper[
                representative_faces, :1, :
            ],
            copy=True,
        )
        derivative_coefficient_lower = coefficient_lower.copy()
        derivative_coefficient_upper = coefficient_upper.copy()
        derivative_coefficient_lower[:, :, 3] = 0.0
        derivative_coefficient_upper[:, :, 3] = 0.0
        derivative_lower, derivative_upper = (
            self._outward_affine_form_bounds_pairwise(
                position_lower=(
                    whole_motion.velocity_lower_object_m_per_unit
                ),
                position_upper=(
                    whole_motion.velocity_upper_object_m_per_unit
                ),
                coefficient_lower=derivative_coefficient_lower,
                coefficient_upper=derivative_coefficient_upper,
            )
        )
        lower_value_lower, lower_value_upper = (
            self._outward_affine_form_bounds_pairwise(
                position_lower=lower_motion.position_lower_object_m,
                position_upper=lower_motion.position_upper_object_m,
                coefficient_lower=coefficient_lower,
                coefficient_upper=coefficient_upper,
            )
        )
        upper_value_lower, upper_value_upper = (
            self._outward_affine_form_bounds_pairwise(
                position_lower=upper_motion.position_lower_object_m,
                position_upper=upper_motion.position_upper_object_m,
                coefficient_lower=coefficient_lower,
                coefficient_upper=coefficient_upper,
            )
        )
        derivative_sign = np.where(
            derivative_lower[:, 0] > 0.0,
            1,
            np.where(derivative_upper[:, 0] < 0.0, -1, 0),
        )
        lower_sign = np.where(
            lower_value_lower[:, 0] > 0.0,
            1,
            np.where(lower_value_upper[:, 0] < 0.0, -1, 0),
        )
        upper_sign = np.where(
            upper_value_lower[:, 0] > 0.0,
            1,
            np.where(upper_value_upper[:, 0] < 0.0, -1, 0),
        )
        same_side_free = (
            (derivative_sign != 0)
            & (lower_sign != 0)
            & (lower_sign == upper_sign)
        )
        reusable_root_gate = (
            (derivative_sign != 0)
            & (lower_sign == -derivative_sign)
            & (upper_sign == derivative_sign)
        )
        gates: list[_PrecertifiedPlaneGate | None] = []
        for index in range(count):
            if not bool(reusable_root_gate[index]):
                gates.append(None)
                continue
            gates.append(
                _PrecertifiedPlaneGate(
                    plane_derivative=IntervalBounds(
                        float(derivative_lower[index, 0]),
                        float(derivative_upper[index, 0]),
                    ),
                    lower_value=IntervalBounds(
                        float(lower_value_lower[index, 0]),
                        float(lower_value_upper[index, 0]),
                    ),
                    upper_value=IntervalBounds(
                        float(upper_value_lower[index, 0]),
                        float(upper_value_upper[index, 0]),
                    ),
                )
            )
        counters.batch_plane_monotone_same_side_free += int(
            np.count_nonzero(same_side_free)
        )
        counters.shared_plane_gate_roots += int(
            np.count_nonzero(reusable_root_gate)
        )
        enclosure = self._certified_monotone_root_position_enclosures_v9(
            phase_lower=lower,
            phase_upper=upper,
            derivative_lower=derivative_lower[:, 0],
            derivative_upper=derivative_upper[:, 0],
            lower_value_lower=lower_value_lower[:, 0],
            lower_value_upper=lower_value_upper[:, 0],
            upper_value_lower=upper_value_lower[:, 0],
            upper_value_upper=upper_value_upper[:, 0],
            whole_motion=whole_motion,
            lower_motion=lower_motion,
            upper_motion=upper_motion,
            eligible=reusable_root_gate,
        )
        enclosure_rows = np.flatnonzero(enclosure.valid)
        counters.pre_root_spatial_enclosure_groups += len(enclosure_rows)
        free_faces: list[frozenset[int]] = [frozenset() for _ in rows]
        if len(enclosure_rows) > 0:
            face_counts = np.fromiter(
                (len(rows[index].face_indices) for index in enclosure_rows),
                dtype=np.int64,
                count=len(enclosure_rows),
            )
            expanded_work_indices = np.repeat(enclosure_rows, face_counts)
            expanded_faces = np.concatenate(
                tuple(rows[index].face_indices for index in enclosure_rows)
            ).astype(np.int64, copy=False)
            _edge_lower, edge_upper = (
                self._outward_affine_form_bounds_pairwise(
                    position_lower=(
                        enclosure.position_lower_object_m[
                            expanded_work_indices
                        ]
                    ),
                    position_upper=(
                        enclosure.position_upper_object_m[
                            expanded_work_indices
                        ]
                    ),
                    coefficient_lower=(
                        self._object_contact_affine_lower[
                            expanded_faces, 1:, :
                        ]
                    ),
                    coefficient_upper=(
                        self._object_contact_affine_upper[
                            expanded_faces, 1:, :
                        ]
                    ),
                )
            )
            edge_coefficient_lower = self._object_contact_affine_lower[
                expanded_faces, 1:, :
            ]
            edge_coefficient_upper = self._object_contact_affine_upper[
                expanded_faces, 1:, :
            ]
            derivative_edge_coefficient_lower = np.array(
                edge_coefficient_lower, copy=True
            )
            derivative_edge_coefficient_upper = np.array(
                edge_coefficient_upper, copy=True
            )
            derivative_edge_coefficient_lower[:, :, 3] = 0.0
            derivative_edge_coefficient_upper[:, :, 3] = 0.0
            start_edge_lower, start_edge_upper = (
                self._outward_affine_form_bounds_pairwise(
                    position_lower=(
                        lower_motion.position_lower_object_m[
                            expanded_work_indices
                        ]
                    ),
                    position_upper=(
                        lower_motion.position_upper_object_m[
                            expanded_work_indices
                        ]
                    ),
                    coefficient_lower=edge_coefficient_lower,
                    coefficient_upper=edge_coefficient_upper,
                )
            )
            end_edge_lower, end_edge_upper = (
                self._outward_affine_form_bounds_pairwise(
                    position_lower=(
                        upper_motion.position_lower_object_m[
                            expanded_work_indices
                        ]
                    ),
                    position_upper=(
                        upper_motion.position_upper_object_m[
                            expanded_work_indices
                        ]
                    ),
                    coefficient_lower=edge_coefficient_lower,
                    coefficient_upper=edge_coefficient_upper,
                )
            )
            edge_derivative_lower, edge_derivative_upper = (
                self._outward_affine_form_bounds_pairwise(
                    position_lower=(
                        whole_motion.velocity_lower_object_m_per_unit[
                            expanded_work_indices
                        ]
                    ),
                    position_upper=(
                        whole_motion.velocity_upper_object_m_per_unit[
                            expanded_work_indices
                        ]
                    ),
                    coefficient_lower=derivative_edge_coefficient_lower,
                    coefficient_upper=derivative_edge_coefficient_upper,
                )
            )
            (
                edge_second_derivative_lower,
                edge_second_derivative_upper,
            ) = self._outward_affine_form_bounds_pairwise(
                position_lower=(
                    whole_motion.
                    acceleration_lower_object_m_per_unit_squared[
                        expanded_work_indices
                    ]
                ),
                position_upper=(
                    whole_motion.
                    acceleration_upper_object_m_per_unit_squared[
                        expanded_work_indices
                    ]
                ),
                coefficient_lower=derivative_edge_coefficient_lower,
                coefficient_upper=derivative_edge_coefficient_upper,
            )
            _direct_edge_lower, direct_edge_upper = (
                self._certified_affine_values_at_root_from_endpoints_v9(
                    phase_lower=lower,
                    phase_upper=upper,
                    root_phase_lower=(
                        enclosure.phase_lower[expanded_work_indices]
                    ),
                    root_phase_upper=(
                        enclosure.phase_upper[expanded_work_indices]
                    ),
                    lower_value_lower=start_edge_lower,
                    lower_value_upper=start_edge_upper,
                    upper_value_lower=end_edge_lower,
                    upper_value_upper=end_edge_upper,
                    derivative_lower=edge_derivative_lower,
                    derivative_upper=edge_derivative_upper,
                )
            )
            _chord_edge_lower, chord_edge_upper = (
                self._certified_second_order_affine_chord_root_bounds_v9(
                    phase_lower=lower,
                    phase_upper=upper,
                    root_phase_lower=(
                        enclosure.phase_lower[expanded_work_indices]
                    ),
                    root_phase_upper=(
                        enclosure.phase_upper[expanded_work_indices]
                    ),
                    lower_value_lower=start_edge_lower,
                    lower_value_upper=start_edge_upper,
                    upper_value_lower=end_edge_lower,
                    upper_value_upper=end_edge_upper,
                    second_derivative_lower=edge_second_derivative_lower,
                    second_derivative_upper=edge_second_derivative_upper,
                )
            )
            edge_upper = np.minimum.reduce(
                (edge_upper, direct_edge_upper, chord_edge_upper)
            )
            flat_free = np.any(edge_upper < 0.0, axis=1)
            offset = 0
            for work_index, face_count in zip(enclosure_rows, face_counts):
                group_faces = rows[int(work_index)].face_indices
                group_mask = flat_free[offset : offset + int(face_count)]
                free_faces[int(work_index)] = frozenset(
                    int(value) for value in group_faces[group_mask]
                )
                offset += int(face_count)
            counters.pre_root_spatial_free_pairs += int(
                np.count_nonzero(flat_free)
            )
            counters.pre_root_spatial_fully_free_groups += sum(
                len(free_faces[index]) == len(rows[index].face_indices)
                for index in enclosure_rows
            )
        same_side_free.setflags(write=False)
        return same_side_free, tuple(gates), tuple(free_faces)

    def _iter_classify_witness_face_batches_parallel_v9(
        self,
        *,
        prepared: _PreparedPad,
        swept_batches: Sequence[tuple[int, np.ndarray]],
        q_start: np.ndarray,
        direction: np.ndarray,
        lower: float,
        upper: float,
        object_from_hand: np.ndarray,
        counters: _PadCounters,
        transform_cache: IntervalLinkTransformCache | None,
        plane_root_executor: ThreadPoolExecutor,
        plane_root_worker_local: local,
        cache_enabled: bool,
        preclassified_pairs: Mapping[
            tuple[int, int], _PairIntervalClassification
        ] | None = None,
    ) -> Iterator[_PairIntervalClassification]:
        """Classify independent exact planes in a bounded parallel pipeline.

        Worker completion order is never observable.  Faces are yielded in the
        original witness-major and face order, and each worker owns a separate
        interval backend plus its own phase-local transform cache.
        """

        batches = tuple(
            (int(witness), np.asarray(faces, dtype=np.int64))
            for witness, faces in swept_batches
        )
        cached_pairs = dict(preclassified_pairs or {})
        batch_pair_keys = {
            (witness_index, int(face_index))
            for witness_index, faces in batches
            for face_index in faces
        }
        if set(cached_pairs) - batch_pair_keys:
            raise RayClosureError(
                "preclassified parent root is absent from child face batches"
            )
        for key, row in cached_pairs.items():
            if (
                key != (row.witness_flat_index, row.object_face_index)
                or row.state is not _PadSearchState.CERTIFIED_ROOT
                or row.root is None
            ):
                raise RayClosureError(
                    "preclassified child pair must be its exact parent root"
                )
        work: list[_PlaneRootGroupWork] = []
        face_work_rows: list[np.ndarray] = []
        for witness_index, faces in batches:
            uncached_faces = np.asarray(
                [
                    int(face_index)
                    for face_index in faces
                    if (witness_index, int(face_index)) not in cached_pairs
                ],
                dtype=np.int64,
            )
            groups = self._exact_plane_groups_v9(uncached_faces)
            face_to_work: dict[int, int] = {}
            for group_faces in groups:
                work_index = len(work)
                representative = int(group_faces[0])
                work.append(
                    _PlaneRootGroupWork(
                        witness_flat_index=witness_index,
                        representative_face_index=representative,
                        face_indices=group_faces,
                    )
                )
                for face_index_value in group_faces:
                    face_to_work[int(face_index_value)] = work_index
            work_indices = np.fromiter(
                (
                    -1
                    if (witness_index, int(face)) in cached_pairs
                    else face_to_work[int(face)]
                    for face in faces
                ),
                dtype=np.int64,
                count=len(faces),
            )
            work_indices.setflags(write=False)
            face_work_rows.append(work_indices)
        work_rows = tuple(work)
        (
            same_side_free,
            precertified_gates,
            pre_root_free_faces,
        ) = (
            self._batch_precertified_plane_gates_v9(
                prepared=prepared,
                work=work_rows,
                q_start=q_start,
                direction=direction,
                lower=lower,
                upper=upper,
                object_from_hand=object_from_hand,
                counters=counters,
                transform_cache=transform_cache,
            )
        )

        def evaluate_plane_root(
            work_index: int,
        ) -> tuple[IntervalPlaneRootClassification | None, str | None]:
            backend = getattr(
                plane_root_worker_local, "interval_backend", None
            )
            if backend is None:
                backend = self._new_exact_plane_root_worker_backend_v9()
                plane_root_worker_local.interval_backend = backend
                plane_root_worker_local.phase_key = None
                plane_root_worker_local.transform_cache = None
            phase_key = (float(lower).hex(), float(upper).hex())
            if getattr(plane_root_worker_local, "phase_key", None) != phase_key:
                plane_root_worker_local.phase_key = phase_key
                plane_root_worker_local.transform_cache = (
                    backend.new_link_transform_cache()
                    if cache_enabled
                    else None
                )
            row = work_rows[work_index]
            triangle_index = int(
                prepared.triangle_indices[row.witness_flat_index]
            )
            pad_face = np.asarray(
                prepared.verified.faces[triangle_index], dtype=np.int64
            )
            pad_triangle = np.asarray(
                prepared.verified.points_local_m[pad_face], dtype=np.float64
            )
            gate = precertified_gates[work_index]
            gate_arguments: dict[str, IntervalBounds] = {}
            if gate is not None:
                gate_arguments = {
                    "_precertified_plane_derivative": gate.plane_derivative,
                    "_precertified_lower_value": gate.lower_value,
                    "_precertified_upper_value": gate.upper_value,
                }
            try:
                classification = backend.certify_transverse_plane_root(
                    link_name=prepared.verified.link_name,
                    q_start=q_start,
                    direction=direction,
                    phase_lower=lower,
                    phase_upper=upper,
                    base_transform=object_from_hand,
                    witness_point_local_m=(
                        prepared.witness_points_link_m[
                            row.witness_flat_index
                        ]
                    ),
                    pad_triangle_local_m=pad_triangle,
                    object_triangle_m=(
                        self.canonical_object_face_vertices_m[
                            row.representative_face_index
                        ]
                    ),
                    transform_cache=(
                        plane_root_worker_local.transform_cache
                    ),
                    **gate_arguments,
                )
            except IntervalKinematicsError as error:
                return None, f"INTERVAL_BACKEND_REJECTED:{error}"
            return classification, None

        pending: dict[
            int,
            Future[
                tuple[IntervalPlaneRootClassification | None, str | None]
            ],
        ] = {}
        next_submit = 0
        plane_rows: dict[int, IntervalPlaneRootClassification] = {}
        plane_errors: dict[int, str] = {}
        root_free_faces: dict[int, frozenset[int]] = {}
        pre_root_fully_free_work = frozenset(
            index
            for index, row in enumerate(work_rows)
            if len(row.face_indices) > 0
            and len(pre_root_free_faces[index]) == len(row.face_indices)
        )
        exact_work_indices = tuple(
            index
            for index in range(len(work_rows))
            if not bool(same_side_free[index])
            and index not in pre_root_fully_free_work
        )
        if (
            len(exact_work_indices)
            > _EXACT_PLANE_ROOT_TEMPORAL_DEFER_THRESHOLD
            and np.nextafter(lower, upper) < upper
        ):
            deferred_index = exact_work_indices[0]
            deferred = work_rows[deferred_index]
            counters.large_exact_batch_temporal_deferrals += 1
            counters.large_exact_batch_deferred_root_groups += len(
                exact_work_indices
            )
            yield _PairIntervalClassification(
                state=_PadSearchState.UNRESOLVED,
                witness_flat_index=deferred.witness_flat_index,
                object_face_index=deferred.representative_face_index,
                possible_phase_lower=lower,
                root=None,
                reason=(
                    "LARGE_EXACT_ROOT_BATCH_DEFERRED_TO_TEMPORAL_CHILD:"
                    f"{len(exact_work_indices)}>"
                    f"{_EXACT_PLANE_ROOT_TEMPORAL_DEFER_THRESHOLD}"
                ),
            )
            return

        def fill_pipeline() -> None:
            nonlocal next_submit
            while (
                len(pending) < _EXACT_PLANE_ROOT_MAXIMUM_IN_FLIGHT
                and next_submit < len(work_rows)
            ):
                work_index = next_submit
                next_submit += 1
                if bool(same_side_free[work_index]):
                    plane_rows[work_index] = IntervalPlaneRootClassification(
                        IntervalPlaneRootState.CERTIFIED_FREE,
                        IntervalBounds(lower, upper),
                        None,
                        "STRICTLY_MONOTONE_PLANE_WITH_SAME_SIDE_ENDPOINTS",
                    )
                    root_free_faces[work_index] = frozenset()
                    continue
                if work_index in pre_root_fully_free_work:
                    root_free_faces[work_index] = pre_root_free_faces[
                        work_index
                    ]
                    continue
                pending[work_index] = plane_root_executor.submit(
                    evaluate_plane_root, work_index
                )
                counters.actual_plane_root_evaluations += 1
                counters.parallel_plane_root_tasks += 1

        def resolve_work(work_index: int) -> None:
            if (
                work_index in plane_rows
                or work_index in plane_errors
                or work_index in pre_root_fully_free_work
            ):
                return
            fill_pipeline()
            future = pending.pop(work_index)
            plane_row, error_reason = future.result()
            fill_pipeline()
            if error_reason is not None:
                plane_errors[work_index] = error_reason
                return
            if plane_row is None:  # pragma: no cover
                raise RayClosureError("plane-root worker returned no result")
            plane_rows[work_index] = plane_row
            free_faces: frozenset[int] = frozenset()
            if (
                plane_row.state
                is IntervalPlaneRootState.CERTIFIED_TRANSVERSE_PLANE_ROOT
            ):
                plane_root = plane_row.root
                if plane_root is None:  # pragma: no cover
                    raise RayClosureError(
                        "certified parallel plane root lacks its root"
                    )
                counters.interval_newton_iterations += (
                    plane_root.interval_newton_iterations
                )
                counters.root_interpolation_iterations += (
                    plane_root.interpolation_iterations
                )
                counters.root_bisection_iterations += (
                    plane_root.bisection_iterations
                )
                pre_root_free = pre_root_free_faces[work_index]
                group_faces = np.asarray(
                    tuple(
                        int(value)
                        for value in work_rows[work_index].face_indices
                        if int(value) not in pre_root_free
                    ),
                    dtype=np.int64,
                )
                position_lower = np.asarray(
                    [value.lower for value in plane_root.position_object_m],
                    dtype=np.float64,
                )
                position_upper = np.asarray(
                    [value.upper for value in plane_root.position_object_m],
                    dtype=np.float64,
                )
                free_mask = (
                    self._certified_root_batch_triangle_free_face_mask_v9(
                        position_lower_object_m=position_lower,
                        position_upper_object_m=position_upper,
                        face_indices=group_faces,
                    )
                    if len(group_faces) > 0
                    else np.zeros(0, dtype=bool)
                )
                post_root_free = frozenset(
                    int(value) for value in group_faces[free_mask]
                )
                free_faces = pre_root_free | post_root_free
                free_count = len(post_root_free)
                counters.batch_root_triangle_free_pairs += free_count
                counters.batch_root_triangle_uncertain_pairs += (
                    len(group_faces) - free_count
                )
            root_free_faces[work_index] = free_faces

        fill_pipeline()
        try:
            for (
                (witness_index, faces),
                work_indices,
            ) in zip(batches, face_work_rows):
                triangle_index = int(
                    prepared.triangle_indices[witness_index]
                )
                pad_face = np.asarray(
                    prepared.verified.faces[triangle_index], dtype=np.int64
                )
                pad_triangle = np.asarray(
                    prepared.verified.points_local_m[pad_face],
                    dtype=np.float64,
                )
                for face_index_value, work_index_value in zip(
                    faces, work_indices
                ):
                    face_index = int(face_index_value)
                    work_index = int(work_index_value)
                    cached_row = cached_pairs.get(
                        (witness_index, face_index)
                    )
                    if cached_row is not None:
                        if work_index != -1:  # pragma: no cover
                            raise RayClosureError(
                                "cached parent root received exact work"
                            )
                        counters.parent_certified_root_pair_reuses += 1
                        yield cached_row
                        continue
                    if work_index < 0:  # pragma: no cover
                        raise RayClosureError(
                            "uncached child pair lacks exact work"
                        )
                    resolve_work(work_index)
                    counters.interval_pair_evaluations += 1
                    if work_index in pre_root_fully_free_work:
                        row = IntervalRootClassification(
                            IntervalRootState.CERTIFIED_FREE,
                            IntervalBounds(lower, upper),
                            None,
                            "MONOTONE_PLANE_ROOT_SPATIAL_ENCLOSURE_"
                            "STRICTLY_OUTSIDE_TRIANGLE",
                        )
                        yield self._bind_interval_root_classification_v9(
                            prepared=prepared,
                            witness_flat_index=witness_index,
                            object_face_index=face_index,
                            row=row,
                            lower=lower,
                            counters=counters,
                        )
                        continue
                    error_reason = plane_errors.get(work_index)
                    if error_reason is not None:
                        counters.unresolved_witness_face_pairs += 1
                        yield _PairIntervalClassification(
                            state=_PadSearchState.UNRESOLVED,
                            witness_flat_index=witness_index,
                            object_face_index=face_index,
                            possible_phase_lower=lower,
                            root=None,
                            reason=error_reason,
                        )
                        continue
                    plane_row = plane_rows[work_index]
                    if face_index in root_free_faces[work_index]:
                        plane_root = plane_row.root
                        if plane_root is None:  # pragma: no cover
                            raise RayClosureError(
                                "parallel root-free face lacks its plane root"
                            )
                        row = IntervalRootClassification(
                            IntervalRootState.CERTIFIED_FREE,
                            plane_root.isolating_interval,
                            None,
                            "UNIQUE_PLANE_ROOT_STRICTLY_OUTSIDE_TRIANGLE",
                        )
                    else:
                        try:
                            row = (
                                self.interval_kinematics.
                                finalize_transverse_plane_root_for_triangle(
                                plane_classification=plane_row,
                                link_name=prepared.verified.link_name,
                                q_start=q_start,
                                direction=direction,
                                phase_lower=lower,
                                phase_upper=upper,
                                base_transform=object_from_hand,
                                witness_point_local_m=(
                                    prepared.witness_points_link_m[
                                        witness_index
                                    ]
                                ),
                                pad_triangle_local_m=pad_triangle,
                                object_triangle_m=(
                                    self.canonical_object_face_vertices_m[
                                        face_index
                                    ]
                                ),
                                representative_object_triangle_m=(
                                    self.canonical_object_face_vertices_m[
                                        work_rows[
                                            work_index
                                        ].representative_face_index
                                    ]
                                ),
                                    transform_cache=transform_cache,
                                )
                            )
                        except IntervalKinematicsError as error:
                            counters.unresolved_witness_face_pairs += 1
                            yield _PairIntervalClassification(
                                state=_PadSearchState.UNRESOLVED,
                                witness_flat_index=witness_index,
                                object_face_index=face_index,
                                possible_phase_lower=lower,
                                root=None,
                                reason=f"INTERVAL_BACKEND_REJECTED:{error}",
                            )
                            continue
                    yield self._bind_interval_root_classification_v9(
                        prepared=prepared,
                        witness_flat_index=witness_index,
                        object_face_index=face_index,
                        row=row,
                        lower=lower,
                        counters=counters,
                    )
        finally:
            for future in pending.values():
                if future.cancel():
                    counters.actual_plane_root_evaluations -= 1

    def _iter_classify_witness_face_batch_v9(
        self,
        *,
        prepared: _PreparedPad,
        witness_flat_index: int,
        object_face_indices: np.ndarray,
        q_start: np.ndarray,
        direction: np.ndarray,
        lower: float,
        upper: float,
        object_from_hand: np.ndarray,
        counters: _PadCounters,
        transform_cache: IntervalLinkTransformCache | None,
    ) -> Iterator[_PairIntervalClassification]:
        """Reuse one exact plane root across all matching actual triangles."""

        faces = np.asarray(object_face_indices, dtype=np.int64)
        groups = self._exact_plane_groups_v9(faces)
        face_group: dict[int, int] = {}
        for group_index, group_faces in enumerate(groups):
            for face_index_value in group_faces:
                face_group[int(face_index_value)] = group_index

        triangle_index = int(prepared.triangle_indices[witness_flat_index])
        pad_face = np.asarray(
            prepared.verified.faces[triangle_index], dtype=np.int64
        )
        pad_triangle = np.asarray(
            prepared.verified.points_local_m[pad_face], dtype=np.float64
        )
        plane_rows: dict[int, IntervalPlaneRootClassification] = {}
        plane_errors: dict[int, str] = {}
        root_free_faces: dict[int, frozenset[int]] = {}

        for face_index_value in faces:
            face_index = int(face_index_value)
            group_index = face_group[face_index]
            representative = int(groups[group_index][0])
            if group_index not in plane_rows and group_index not in plane_errors:
                group_faces = groups[group_index]
                counters.actual_plane_root_evaluations += 1
                try:
                    plane_row = (
                        self.interval_kinematics.certify_transverse_plane_root(
                            link_name=prepared.verified.link_name,
                            q_start=q_start,
                            direction=direction,
                            phase_lower=lower,
                            phase_upper=upper,
                            base_transform=object_from_hand,
                            witness_point_local_m=(
                                prepared.witness_points_link_m[
                                    witness_flat_index
                                ]
                            ),
                            pad_triangle_local_m=pad_triangle,
                            object_triangle_m=(
                                self.canonical_object_face_vertices_m[
                                    representative
                                ]
                            ),
                            transform_cache=transform_cache,
                        )
                    )
                except IntervalKinematicsError as error:
                    plane_errors[group_index] = (
                        f"INTERVAL_BACKEND_REJECTED:{error}"
                    )
                else:
                    plane_rows[group_index] = plane_row
                    free_faces: frozenset[int] = frozenset()
                    if (
                        plane_row.state
                        is IntervalPlaneRootState.CERTIFIED_TRANSVERSE_PLANE_ROOT
                    ):
                        plane_root = plane_row.root
                        if plane_root is None:  # pragma: no cover
                            raise RayClosureError(
                                "certified plane-root batch lacks its root"
                            )
                        counters.interval_newton_iterations += (
                            plane_root.interval_newton_iterations
                        )
                        counters.root_interpolation_iterations += (
                            plane_root.interpolation_iterations
                        )
                        counters.root_bisection_iterations += (
                            plane_root.bisection_iterations
                        )
                        position_lower = np.asarray(
                            [
                                row.lower
                                for row in plane_root.position_object_m
                            ],
                            dtype=np.float64,
                        )
                        position_upper = np.asarray(
                            [
                                row.upper
                                for row in plane_root.position_object_m
                            ],
                            dtype=np.float64,
                        )
                        free_mask = (
                            self._certified_root_batch_triangle_free_face_mask_v9(
                                position_lower_object_m=position_lower,
                                position_upper_object_m=position_upper,
                                face_indices=group_faces,
                            )
                        )
                        free_faces = frozenset(
                            int(value)
                            for value in group_faces[free_mask]
                        )
                        free_count = int(np.count_nonzero(free_mask))
                        counters.batch_root_triangle_free_pairs += free_count
                        counters.batch_root_triangle_uncertain_pairs += (
                            len(group_faces) - free_count
                        )
                    root_free_faces[group_index] = free_faces

            counters.interval_pair_evaluations += 1
            error_reason = plane_errors.get(group_index)
            if error_reason is not None:
                counters.unresolved_witness_face_pairs += 1
                yield _PairIntervalClassification(
                    state=_PadSearchState.UNRESOLVED,
                    witness_flat_index=witness_flat_index,
                    object_face_index=face_index,
                    possible_phase_lower=lower,
                    root=None,
                    reason=error_reason,
                )
                continue

            plane_row = plane_rows[group_index]
            if face_index in root_free_faces[group_index]:
                plane_root = plane_row.root
                if plane_root is None:  # pragma: no cover
                    raise RayClosureError(
                        "root-batch free face lacks its plane root"
                    )
                row = IntervalRootClassification(
                    IntervalRootState.CERTIFIED_FREE,
                    plane_root.isolating_interval,
                    None,
                    "UNIQUE_PLANE_ROOT_STRICTLY_OUTSIDE_TRIANGLE",
                )
            else:
                try:
                    row = (
                        self.interval_kinematics.
                        finalize_transverse_plane_root_for_triangle(
                            plane_classification=plane_row,
                            link_name=prepared.verified.link_name,
                            q_start=q_start,
                            direction=direction,
                            phase_lower=lower,
                            phase_upper=upper,
                            base_transform=object_from_hand,
                            witness_point_local_m=(
                                prepared.witness_points_link_m[
                                    witness_flat_index
                                ]
                            ),
                            pad_triangle_local_m=pad_triangle,
                            object_triangle_m=(
                                self.canonical_object_face_vertices_m[
                                    face_index
                                ]
                            ),
                            representative_object_triangle_m=(
                                self.canonical_object_face_vertices_m[
                                    representative
                                ]
                            ),
                            transform_cache=transform_cache,
                        )
                    )
                except IntervalKinematicsError as error:
                    counters.unresolved_witness_face_pairs += 1
                    yield _PairIntervalClassification(
                        state=_PadSearchState.UNRESOLVED,
                        witness_flat_index=witness_flat_index,
                        object_face_index=face_index,
                        possible_phase_lower=lower,
                        root=None,
                        reason=f"INTERVAL_BACKEND_REJECTED:{error}",
                    )
                    continue
            yield self._bind_interval_root_classification_v9(
                prepared=prepared,
                witness_flat_index=witness_flat_index,
                object_face_index=face_index,
                row=row,
                lower=lower,
                counters=counters,
            )

    def _bind_interval_root_classification_v9(
        self,
        *,
        prepared: _PreparedPad,
        witness_flat_index: int,
        object_face_index: int,
        row: IntervalRootClassification,
        lower: float,
        counters: _PadCounters,
    ) -> _PairIntervalClassification:
        if row.state is IntervalRootState.CERTIFIED_FREE:
            return _PairIntervalClassification(
                state=_PadSearchState.CERTIFIED_FREE,
                witness_flat_index=witness_flat_index,
                object_face_index=object_face_index,
                possible_phase_lower=lower,
                root=None,
                reason=row.reason,
            )
        if row.state is IntervalRootState.UNRESOLVED:
            counters.unresolved_witness_face_pairs += 1
            return _PairIntervalClassification(
                state=_PadSearchState.UNRESOLVED,
                witness_flat_index=witness_flat_index,
                object_face_index=object_face_index,
                possible_phase_lower=lower,
                root=None,
                reason=row.reason,
            )
        certificate = row.certificate
        if certificate is None:
            raise RayClosureError(
                "interval root state lacks its mandatory certificate"
            )
        if row.state is (
            IntervalRootState.CERTIFIED_PAD_DIRECTIONAL_TRANSVERSE_ROOT
        ):
            semantic_classification = (
                "ALLOWED_PATH_LOCAL_FREE_SIDE_TRANSVERSE_CONTACT"
                if self._contact_face_mask[object_face_index]
                else "FORBIDDEN_SEMANTIC_FIRST_CONTACT"
            )
        elif row.state is (
            IntervalRootState.CERTIFIED_PAD_DIRECTION_REJECTED_TRANSVERSE_ROOT
        ):
            if certificate.pad_approach.strictly_negative:
                semantic_classification = "PAD_NORMAL_DOMAIN_REJECTED"
            else:
                raise RayClosureError(
                    "direction-rejected root has no strict rejected direction"
                )
        else:
            raise RayClosureError(
                f"unsupported interval root state: {row.state}"
            )
        counters.certified_contact_roots += 1
        binding = _CertifiedContactRootBinding(
            pad_name=prepared.verified.name,
            witness_flat_index=witness_flat_index,
            pad_triangle_index=int(
                prepared.triangle_indices[witness_flat_index]
            ),
            witness_index=int(
                prepared.witness_indices[witness_flat_index]
            ),
            object_face_index=object_face_index,
            semantic_classification=semantic_classification,
            certificate=certificate,
        )
        return _PairIntervalClassification(
            state=_PadSearchState.CERTIFIED_ROOT,
            witness_flat_index=witness_flat_index,
            object_face_index=object_face_index,
            possible_phase_lower=certificate.phase.lower,
            root=binding,
            reason=row.reason,
        )

    @staticmethod
    def _combine_pair_classifications_v9(
        rows: Sequence[_PairIntervalClassification],
        *,
        lower: float,
        upper: float,
        counters: _PadCounters,
    ) -> _PadSearchOutcome:
        roots = [row.root for row in rows if row.root is not None]
        unresolved = [
            row for row in rows if row.state is _PadSearchState.UNRESOLVED
        ]
        if not roots:
            if unresolved:
                counters.competing_root_order_blocks += 1
                reasons = tuple(sorted({row.reason for row in unresolved}))
                return _PadSearchOutcome(
                    state=_PadSearchState.UNRESOLVED,
                    interval_lower=lower,
                    interval_upper=upper,
                    possible_first_contact_set=None,
                    unresolved_reason="|".join(reasons),
                )
            return _PadSearchOutcome(
                state=_PadSearchState.CERTIFIED_FREE,
                interval_lower=lower,
                interval_upper=upper,
                possible_first_contact_set=None,
                unresolved_reason=None,
            )

        if unresolved:
            counters.competing_root_order_blocks += 1
            reasons = tuple(sorted({row.reason for row in unresolved}))
            return _PadSearchOutcome(
                state=_PadSearchState.UNRESOLVED,
                interval_lower=lower,
                interval_upper=upper,
                possible_first_contact_set=None,
                unresolved_reason=(
                    "UNRESOLVED_PAIR_MAY_PRECEDE_CERTIFIED_ROOT:"
                    + "|".join(reasons)
                ),
            )
        possible_set = PossibleFirstContactSet.from_certified_roots(
            tuple(root for root in roots if root is not None)
        )
        possible_roots = possible_set.possible_earliest_roots
        counters.possible_earliest_root_count += len(possible_roots)
        return _PadSearchOutcome(
            state=_PadSearchState.CERTIFIED_ROOT,
            interval_lower=min(
                root.certificate.phase.lower for root in possible_roots
            ),
            interval_upper=max(
                root.certificate.phase.upper for root in possible_roots
            ),
            possible_first_contact_set=possible_set,
            unresolved_reason=None,
        )

    @staticmethod
    def _parent_pair_inheritance_for_child_v9(
        row: _PairIntervalClassification,
        *,
        child_lower: float,
        child_upper: float,
    ) -> _ParentPairInheritance:
        """Restrict one rigorous parent pair result to one closed child."""

        if (
            not math.isfinite(child_lower)
            or not math.isfinite(child_upper)
            or child_lower >= child_upper
        ):
            raise RayClosureError("child inheritance interval is malformed")
        if row.state is _PadSearchState.CERTIFIED_FREE:
            if row.root is not None:
                raise RayClosureError("certified-free parent pair carries a root")
            return _ParentPairInheritance.PRUNE_PARENT_CERTIFIED_FREE
        if row.state is _PadSearchState.UNRESOLVED:
            if row.root is not None:
                raise RayClosureError("unresolved parent pair carries a root")
            return _ParentPairInheritance.RECOMPUTE
        root = row.root
        if root is None:
            raise RayClosureError("certified parent root is missing")
        phase = root.certificate.phase
        if phase.upper < child_lower or phase.lower > child_upper:
            return _ParentPairInheritance.PRUNE_PARENT_ROOT_DISJOINT
        if child_lower <= phase.lower and phase.upper <= child_upper:
            return _ParentPairInheritance.REUSE_PARENT_ROOT
        return _ParentPairInheritance.RECOMPUTE

    def _search_pad_first_contact_v9(
        self,
        *,
        prepared: _PreparedPad,
        q_start: np.ndarray,
        direction: np.ndarray,
        maximum_parameter: float,
        object_from_hand: np.ndarray,
        spatial_error_bound_m: float,
        budget: _Budget,
        counters: _PadCounters,
        execution: _GeometryExecutionContext,
    ) -> _PadSearchOutcome:
        speed_bounds = self._witness_speed_bounds(
            prepared, q_start, direction, maximum_parameter
        )
        plane_root_worker_local = local()

        def search(
            lower: float,
            upper: float,
            lower_positions: np.ndarray | None = None,
            upper_positions: np.ndarray | None = None,
            parent_face_frontier: Mapping[int, np.ndarray] | None = None,
            parent_pair_classifications: Mapping[
                tuple[int, int], _PairIntervalClassification
            ] | None = None,
        ) -> _PadSearchOutcome:
            budget.consume()
            counters.interval_evaluations += 1
            midpoint = lower + 0.5 * (upper - lower)
            states, state_key = self._cached_witness_states(
                prepared,
                q_start + midpoint * direction,
                direction,
                object_from_hand,
                execution,
            )
            interval_half_width = 0.5 * (upper - lower)
            if parent_face_frontier is None:
                geometry = self._interval_geometry(
                    prepared=prepared,
                    states=states,
                    state_key=state_key,
                    enclosure_radii_m=(
                        speed_bounds * interval_half_width
                    ),
                    spatial_error_bound_m=spatial_error_bound_m,
                    counters=counters,
                    execution=execution,
                )
                possible_indices = np.flatnonzero(geometry.possible)
                if len(possible_indices) == 0:
                    if geometry.minimum_free_margin_m is None:
                        raise RayClosureError(
                            "free interval lacks an exact minimum witness margin"
                        )
                    counters.certified_free_intervals += 1
                    counters.clearance_lower_bound_m = min(
                        counters.clearance_lower_bound_m,
                        max(0.0, geometry.minimum_free_margin_m),
                    )
                    return _PadSearchOutcome(
                        state=_PadSearchState.CERTIFIED_FREE,
                        interval_lower=lower,
                        interval_upper=upper,
                        possible_first_contact_set=None,
                        unresolved_reason=None,
                    )
            else:
                counters.parent_frontier_geometry_bypasses += 1
                possible_indices = np.asarray(
                    sorted(
                        int(witness_index)
                        for witness_index, face_indices in (
                            parent_face_frontier.items()
                        )
                        if len(face_indices) > 0
                    ),
                    dtype=np.int64,
                )
                if len(possible_indices) == 0:
                    counters.certified_free_intervals += 1
                    return _PadSearchOutcome(
                        state=_PadSearchState.CERTIFIED_FREE,
                        interval_lower=lower,
                        interval_upper=upper,
                        possible_first_contact_set=None,
                        unresolved_reason=None,
                    )
            if lower_positions is None:
                lower_positions = self._witness_positions_object(
                    prepared,
                    q_start + lower * direction,
                    object_from_hand,
                )
            if upper_positions is None:
                upper_positions = self._witness_positions_object(
                    prepared,
                    q_start + upper * direction,
                    object_from_hand,
                )
            transform_cache = (
                self.interval_kinematics.new_link_transform_cache()
                if execution.cache_enabled
                else None
            )
            root_rows: list[_PairIntervalClassification] = []
            combined: _PadSearchOutcome | None = None
            staged_temporal_defer = False
            potential_exact_plane_groups = 0
            materialized_before = counters.swept_face_witnesses_materialized
            swept_batch_rows: list[tuple[int, np.ndarray]] = []
            swept_batch_iterator = self._iter_complete_swept_face_batches_v9(
                prepared=prepared,
                possible_witness_indices=possible_indices,
                q_start=q_start,
                direction=direction,
                lower=lower,
                upper=upper,
                object_from_hand=object_from_hand,
                counters=counters,
                transform_cache=transform_cache,
                apply_certified_batch_cull=True,
                witness_start_positions_object_m=lower_positions,
                witness_end_positions_object_m=upper_positions,
                endpoint_error_bound_m=spatial_error_bound_m,
                parent_face_frontier=parent_face_frontier,
                pair_cull_executor=pair_cull_executor,
            )
            known_parent_pairs = parent_pair_classifications or {}
            for witness_index, face_indices in swept_batch_iterator:
                swept_batch_rows.append((witness_index, face_indices))
                uncached_faces = np.asarray(
                    [
                        int(face_index)
                        for face_index in face_indices
                        if (witness_index, int(face_index))
                        not in known_parent_pairs
                    ],
                    dtype=np.int64,
                )
                if len(uncached_faces) > 0:
                    potential_exact_plane_groups += len(
                        self._exact_plane_groups_v9(uncached_faces)
                    )
                if (
                    potential_exact_plane_groups
                    > _EXACT_PLANE_ROOT_TEMPORAL_DEFER_THRESHOLD
                    and np.nextafter(lower, upper) < upper
                ):
                    staged_temporal_defer = True
                    swept_batch_iterator.close()
                    break
            swept_batches = tuple(swept_batch_rows)
            if staged_temporal_defer:
                materialized_count = (
                    counters.swept_face_witnesses_materialized
                    - materialized_before
                )
                counters.staged_potential_root_temporal_deferrals += 1
                counters.staged_potential_root_groups += (
                    potential_exact_plane_groups
                )
                counters.staged_unmaterialized_witnesses += max(
                    0, len(possible_indices) - materialized_count
                )
                counters.large_exact_batch_temporal_deferrals += 1
                counters.large_exact_batch_deferred_root_groups += (
                    potential_exact_plane_groups
                )
                counters.competing_root_order_blocks += 1
                combined = _PadSearchOutcome(
                    state=_PadSearchState.UNRESOLVED,
                    interval_lower=lower,
                    interval_upper=upper,
                    possible_first_contact_set=None,
                    unresolved_reason=(
                        "STAGED_POTENTIAL_EXACT_ROOT_BATCH_DEFERRED_TO_"
                        "TEMPORAL_CHILD:"
                        f"{potential_exact_plane_groups}>"
                        f"{_EXACT_PLANE_ROOT_TEMPORAL_DEFER_THRESHOLD}"
                    ),
                )
            child_face_frontier = (
                None
                if staged_temporal_defer
                else {
                    witness_index: face_indices
                    for witness_index, face_indices in swept_batches
                }
            )
            classified_pairs: dict[
                tuple[int, int], _PairIntervalClassification
            ] = {}
            classified_iterator = (
                ()
                if staged_temporal_defer
                else self._iter_classify_witness_face_batches_parallel_v9(
                    prepared=prepared,
                    swept_batches=swept_batches,
                    q_start=q_start,
                    direction=direction,
                    lower=lower,
                    upper=upper,
                    object_from_hand=object_from_hand,
                    counters=counters,
                    transform_cache=transform_cache,
                    plane_root_executor=pair_cull_executor,
                    plane_root_worker_local=plane_root_worker_local,
                    cache_enabled=execution.cache_enabled,
                    preclassified_pairs=parent_pair_classifications,
                )
            )
            for row in classified_iterator:
                classified_pairs[
                    (row.witness_flat_index, row.object_face_index)
                ] = row
                if row.state is _PadSearchState.UNRESOLVED:
                    counters.competing_root_order_blocks += 1
                    combined = _PadSearchOutcome(
                        state=_PadSearchState.UNRESOLVED,
                        interval_lower=lower,
                        interval_upper=upper,
                        possible_first_contact_set=None,
                        unresolved_reason=row.reason,
                    )
                    break
                if row.root is not None:
                    root_rows.append(row)
            if combined is None:
                combined = self._combine_pair_classifications_v9(
                    root_rows,
                    lower=lower,
                    upper=upper,
                    counters=counters,
                )
            if transform_cache is not None:
                execution.stats.interval_transform_cache_hits += (
                    transform_cache.hit_count
                )
                execution.stats.interval_transform_cache_misses += (
                    transform_cache.miss_count
                )
                execution.stats.interval_transform_cache_peak_entries = max(
                    execution.stats.interval_transform_cache_peak_entries,
                    transform_cache.entry_count,
                )
                execution.stats.interval_point_cache_hits += (
                    transform_cache.point_hit_count
                )
                execution.stats.interval_point_cache_misses += (
                    transform_cache.point_miss_count
                )
                execution.stats.interval_point_cache_nonprimary_bypasses += (
                    transform_cache.point_nonprimary_bypass_count
                )
                execution.stats.interval_point_cache_peak_entries = max(
                    execution.stats.interval_point_cache_peak_entries,
                    transform_cache.point_entry_count,
                )
                execution.stats.interval_pad_area_cache_hits += (
                    transform_cache.pad_area_hit_count
                )
                execution.stats.interval_pad_area_cache_misses += (
                    transform_cache.pad_area_miss_count
                )
                execution.stats.interval_pad_area_cache_nonprimary_bypasses += (
                    transform_cache.pad_area_nonprimary_bypass_count
                )
                execution.stats.interval_pad_area_cache_peak_entries = max(
                    execution.stats.interval_pad_area_cache_peak_entries,
                    transform_cache.pad_area_entry_count,
                )
            del transform_cache
            if combined.state is not _PadSearchState.UNRESOLVED:
                return combined
            if np.nextafter(lower, upper) >= upper:
                return combined
            if staged_temporal_defer:
                first = search(
                    lower,
                    midpoint,
                    lower_positions,
                    states.positions_object_m,
                    None,
                    None,
                )
                if first.state is not _PadSearchState.CERTIFIED_FREE:
                    return first
                return search(
                    midpoint,
                    upper,
                    states.positions_object_m,
                    upper_positions,
                    None,
                    None,
                )

            def child_inheritance(
                child_lower: float,
                child_upper: float,
            ) -> tuple[
                dict[int, np.ndarray],
                dict[tuple[int, int], _PairIntervalClassification],
            ]:
                child_frontier: dict[int, np.ndarray] = {}
                inherited_roots: dict[
                    tuple[int, int], _PairIntervalClassification
                ] = {}
                assert child_face_frontier is not None
                for witness_index, face_indices in child_face_frontier.items():
                    retained: list[int] = []
                    for face_index_value in face_indices:
                        face_index = int(face_index_value)
                        key = (witness_index, face_index)
                        parent_row = classified_pairs.get(key)
                        if parent_row is None:
                            retained.append(face_index)
                            continue
                        decision = (
                            self._parent_pair_inheritance_for_child_v9(
                                parent_row,
                                child_lower=child_lower,
                                child_upper=child_upper,
                            )
                        )
                        if decision is (
                            _ParentPairInheritance.
                            PRUNE_PARENT_CERTIFIED_FREE
                        ):
                            counters.parent_certified_free_pair_prunes += 1
                            continue
                        if decision is (
                            _ParentPairInheritance.
                            PRUNE_PARENT_ROOT_DISJOINT
                        ):
                            counters.parent_certified_root_outside_pair_prunes += (
                                1
                            )
                            continue
                        retained.append(face_index)
                        if decision is (
                            _ParentPairInheritance.REUSE_PARENT_ROOT
                        ):
                            inherited_roots[key] = parent_row
                    if retained:
                        retained_array = np.asarray(
                            retained, dtype=np.int64
                        )
                        retained_array.setflags(write=False)
                        child_frontier[witness_index] = retained_array
                return child_frontier, inherited_roots

            left_frontier, left_inherited_roots = child_inheritance(
                lower, midpoint
            )
            first = search(
                lower,
                midpoint,
                lower_positions,
                states.positions_object_m,
                left_frontier,
                left_inherited_roots,
            )
            if first.state is not _PadSearchState.CERTIFIED_FREE:
                return first
            right_frontier, right_inherited_roots = child_inheritance(
                midpoint, upper
            )
            return search(
                midpoint,
                upper,
                states.positions_object_m,
                upper_positions,
                right_frontier,
                right_inherited_roots,
            )

        with ThreadPoolExecutor(
            max_workers=max(
                _EXACT_FACE_PAIR_WORKER_COUNT,
                _EXACT_PLANE_ROOT_WORKER_COUNT,
            ),
            thread_name_prefix="carts-exact-geometry",
        ) as pair_cull_executor:
            boundaries = np.asarray(
                [
                    (
                        maximum_parameter
                        if index == _WHOLE_PATH_SPHERE_SEGMENT_COUNT
                        else maximum_parameter
                        * (
                            float(index)
                            / float(_WHOLE_PATH_SPHERE_SEGMENT_COUNT)
                        )
                    )
                    for index in range(
                        _WHOLE_PATH_SPHERE_SEGMENT_COUNT + 1
                    )
                ],
                dtype=np.float64,
            )
            boundary_positions = tuple(
                self._witness_positions_object(
                    prepared,
                    q_start + boundary * direction,
                    object_from_hand,
                )
                for boundary in boundaries
            )
            collected_roots: dict[
                tuple[int, int], CertifiedContactFeatureRoot
            ] = {}

            def merged_root_outcome() -> _PadSearchOutcome:
                possible_set = PossibleFirstContactSet.from_certified_roots(
                    tuple(collected_roots.values())
                )
                possible_roots = possible_set.possible_earliest_roots
                counters.possible_earliest_root_count += len(
                    possible_roots
                )
                return _PadSearchOutcome(
                    state=_PadSearchState.CERTIFIED_ROOT,
                    interval_lower=min(
                        root.certificate.phase.lower
                        for root in possible_roots
                    ),
                    interval_upper=max(
                        root.certificate.phase.upper
                        for root in possible_roots
                    ),
                    possible_first_contact_set=possible_set,
                    unresolved_reason=None,
                )

            for segment_index in range(
                _WHOLE_PATH_SPHERE_SEGMENT_COUNT
            ):
                segment_lower = float(boundaries[segment_index])
                segment_upper = float(boundaries[segment_index + 1])
                if collected_roots:
                    current_set = PossibleFirstContactSet.from_certified_roots(
                        tuple(collected_roots.values())
                    )
                    if (
                        segment_lower
                        > current_set.guaranteed_earliest_phase_upper
                    ):
                        return merged_root_outcome()
                outcome = search(
                    segment_lower,
                    segment_upper,
                    boundary_positions[segment_index],
                    boundary_positions[segment_index + 1],
                )
                if outcome.state is _PadSearchState.UNRESOLVED:
                    return outcome
                if outcome.state is _PadSearchState.CERTIFIED_FREE:
                    if collected_roots:
                        return merged_root_outcome()
                    continue
                possible_set = outcome.possible_first_contact_set
                if possible_set is None:  # pragma: no cover
                    raise RayClosureError(
                        "ordered root segment lacks possible-first set"
                    )
                counters.possible_earliest_root_count -= len(
                    possible_set.possible_earliest_roots
                )
                for root in possible_set.all_certified_roots:
                    feature_key = (
                        root.witness_flat_index,
                        root.object_face_index,
                    )
                    current = collected_roots.get(feature_key)
                    if current is None or _possible_root_order_key(
                        root
                    ) < _possible_root_order_key(current):
                        collected_roots[feature_key] = root
                merged_set = PossibleFirstContactSet.from_certified_roots(
                    tuple(collected_roots.values())
                )
                if (
                    merged_set.guaranteed_earliest_phase_upper
                    < segment_upper
                ):
                    return merged_root_outcome()
            if collected_roots:
                return merged_root_outcome()
            return _PadSearchOutcome(
                state=_PadSearchState.CERTIFIED_FREE,
                interval_lower=0.0,
                interval_upper=maximum_parameter,
                possible_first_contact_set=None,
                unresolved_reason=None,
            )

    @staticmethod
    def _finite_chord_triangle_mask(
        start_points: np.ndarray,
        end_points: np.ndarray,
        triangles: np.ndarray,
        signed_start_m: np.ndarray,
        signed_end_m: np.ndarray,
    ) -> np.ndarray:
        """Return finite triangles hit by endpoint chords after plane crossing.

        This is a feature-seeding predicate, not a continuous-collision gate.
        A curved witness path whose endpoint chord misses is left to temporal
        subdivision and may still fail closed at the computation budget.
        """

        fractions = signed_start_m / (signed_start_m - signed_end_m)
        chord_points = start_points + fractions[:, None] * (
            end_points - start_points
        )
        first_edges = triangles[:, 1] - triangles[:, 0]
        second_edges = triangles[:, 2] - triangles[:, 0]
        offsets = chord_points - triangles[:, 0]
        dot00 = np.sum(first_edges * first_edges, axis=1)
        dot01 = np.sum(first_edges * second_edges, axis=1)
        dot11 = np.sum(second_edges * second_edges, axis=1)
        dot20 = np.sum(offsets * first_edges, axis=1)
        dot21 = np.sum(offsets * second_edges, axis=1)
        denominators = dot00 * dot11 - dot01 * dot01
        valid = denominators > 0.0
        first_coordinate = np.full(len(triangles), math.nan, dtype=np.float64)
        second_coordinate = np.full(len(triangles), math.nan, dtype=np.float64)
        first_coordinate[valid] = (
            dot11[valid] * dot20[valid] - dot01[valid] * dot21[valid]
        ) / denominators[valid]
        second_coordinate[valid] = (
            dot00[valid] * dot21[valid] - dot01[valid] * dot20[valid]
        ) / denominators[valid]
        third_coordinate = 1.0 - first_coordinate - second_coordinate
        return (
            valid
            & (first_coordinate >= -_BARYCENTRIC_ERROR)
            & (second_coordinate >= -_BARYCENTRIC_ERROR)
            & (third_coordinate >= -_BARYCENTRIC_ERROR)
        )

    def _solve_finite_witness_face_root(
        self,
        *,
        prepared: _PreparedPad,
        witness_flat_index: int,
        object_face_index: int,
        lower: float,
        upper: float,
        q_start: np.ndarray,
        direction: np.ndarray,
        object_from_hand: np.ndarray,
        speed_bound_m_per_unit: float,
        spatial_error_bound_m: float,
        counters: _PadCounters,
    ) -> tuple[_ContactEvent, float] | None:
        triangle = self.object_model.mesh.face_vertices_m[object_face_index]
        object_normal = self.object_model.mesh.face_normals[object_face_index]
        point_local = prepared.witness_points_link_m[witness_flat_index]

        def signed_plane_distance(phase: float) -> float:
            counters.nonlinear_root_fk_evaluations += 1
            q = q_start + float(phase) * direction
            links = self.hand_model.forward_kinematics(
                q, base_transform=object_from_hand
            )
            transform = links[prepared.verified.link_name]
            point_object = transform[:3, :3] @ point_local + transform[:3, 3]
            return float(object_normal @ (point_object - triangle[0]))

        try:
            phase = float(
                brentq(
                    signed_plane_distance,
                    lower,
                    upper,
                    xtol=_BRENT_ROOT_ABSOLUTE_TOLERANCE,
                    rtol=_BRENT_ROOT_RELATIVE_TOLERANCE,
                    maxiter=_BRENT_ROOT_MAXIMUM_ITERATIONS,
                    disp=True,
                )
            )
        except (RuntimeError, ValueError):
            return None
        counters.nonlinear_feature_roots_solved += 1
        counters.nonlinear_root_fk_evaluations += 1
        q_root = q_start + phase * direction
        links = self.hand_model.forward_kinematics(
            q_root, base_transform=object_from_hand
        )
        transform = links[prepared.verified.link_name]
        point_object = transform[:3, :3] @ point_local + transform[:3, 3]
        closest = _closest_point_on_triangle(point_object, triangle)
        triangle_distance = float(np.linalg.norm(point_object - closest))
        phase_error = _TIME_ERROR * max(1.0, abs(phase))
        root_spatial_error = float(
            np.nextafter(
                spatial_error_bound_m
                + speed_bound_m_per_unit * phase_error,
                math.inf,
            )
        )
        if triangle_distance > root_spatial_error:
            return None
        jacobian = self.hand_model.geometric_jacobian(
            prepared.verified.link_name,
            q_root,
            point_local_m=point_local,
            base_transform=object_from_hand,
        )
        velocity = jacobian[:3] @ direction
        speed = float(np.linalg.norm(velocity))
        if speed == 0.0 or not math.isfinite(speed):
            return None
        pad_normal = transform[:3, :3] @ prepared.witness_normals_link[
            witness_flat_index
        ]
        pad_approach = float(pad_normal @ velocity)
        object_approach = -float(object_normal @ velocity)
        directional_error = _DOT_ERROR * speed
        if not self._contact_face_mask[object_face_index]:
            classification = "FORBIDDEN_SEMANTIC_FIRST_CONTACT"
        elif pad_approach <= directional_error:
            classification = "PAD_NORMAL_DOMAIN_REJECTED"
        elif object_approach <= directional_error:
            classification = "OBJECT_NORMAL_DOMAIN_REJECTED"
        else:
            classification = "ALLOWED_DIRECTIONAL_FIRST_CONTACT"
        event = _ContactEvent(
            normalized_closure=phase,
            interval_width=2.0 * phase_error,
            spatial_error_bound_m=root_spatial_error,
            triangle_index=int(
                prepared.triangle_indices[witness_flat_index]
            ),
            witness_index=int(prepared.witness_indices[witness_flat_index]),
            witness_flat_index=witness_flat_index,
            object_face_index=int(object_face_index),
            classification=classification,
            first_hit=None,
        )
        return event, phase_error

    def _finite_chord_feature_event(
        self,
        *,
        prepared: _PreparedPad,
        states_midpoint: _WitnessStates,
        geometry: _IntervalGeometry,
        speed_bounds_m_per_unit: np.ndarray,
        lower: float,
        upper: float,
        q_start: np.ndarray,
        direction: np.ndarray,
        object_from_hand: np.ndarray,
        spatial_error_bound_m: float,
        counters: _PadCounters,
    ) -> _ContactEvent | None:
        possible_witnesses = np.flatnonzero(geometry.possible)
        if len(possible_witnesses) == 0:
            return None
        face_indices = geometry.nearest_face_indices[possible_witnesses]
        if np.any(face_indices < 0):
            raise RayClosureError("possible witness lacks an exact nearest face")
        lower_states = self._witness_states(
            prepared,
            q_start + lower * direction,
            direction,
            object_from_hand,
        )
        upper_states = self._witness_states(
            prepared,
            q_start + upper * direction,
            direction,
            object_from_hand,
        )
        triangles = self.object_model.mesh.face_vertices_m[face_indices]
        object_normals = self.object_model.mesh.face_normals[face_indices]
        lower_points = lower_states.positions_object_m[possible_witnesses]
        upper_points = upper_states.positions_object_m[possible_witnesses]
        signed_lower = np.sum(
            object_normals * (lower_points - triangles[:, 0]), axis=1
        )
        signed_upper = np.sum(
            object_normals * (upper_points - triangles[:, 0]), axis=1
        )
        coordinate_scale = np.sum(
            np.abs(object_normals)
            * (
                np.abs(lower_points)
                + np.abs(upper_points)
                + 2.0 * np.abs(triangles[:, 0])
            ),
            axis=1,
        )
        plane_error = _FK_ERROR * (
            coordinate_scale + self.intersector.characteristic_length_m
        )
        crossing = (
            (signed_lower > plane_error)
            & (signed_upper < -plane_error)
        )
        if not np.any(crossing):
            return None
        possible_witnesses = possible_witnesses[crossing]
        face_indices = face_indices[crossing]
        triangles = triangles[crossing]
        signed_lower = signed_lower[crossing]
        signed_upper = signed_upper[crossing]
        plane_error = plane_error[crossing]
        chord_mask = self._finite_chord_triangle_mask(
            lower_points[crossing],
            upper_points[crossing],
            triangles,
            signed_lower,
            signed_upper,
        )
        if not np.any(chord_mask):
            return None
        possible_witnesses = possible_witnesses[chord_mask]
        face_indices = face_indices[chord_mask]
        signed_lower = signed_lower[chord_mask]
        plane_error = plane_error[chord_mask]
        counters.finite_chord_feature_candidates += len(possible_witnesses)
        speed_rows = speed_bounds_m_per_unit[possible_witnesses]
        lower_time_bounds = lower + np.maximum(
            0.0,
            (signed_lower - plane_error - spatial_error_bound_m) / speed_rows,
        )
        order = np.lexsort(
            (face_indices, possible_witnesses, lower_time_bounds)
        )
        candidates: list[tuple[float, float, int, _ContactEvent]] = []
        for row_index_value in order:
            row_index = int(row_index_value)
            if candidates:
                earliest_phase = min(row[0] for row in candidates)
                if lower_time_bounds[row_index] > earliest_phase:
                    break
            witness_index = int(possible_witnesses[row_index])
            solved = self._solve_finite_witness_face_root(
                prepared=prepared,
                witness_flat_index=witness_index,
                object_face_index=int(face_indices[row_index]),
                lower=lower,
                upper=upper,
                q_start=q_start,
                direction=direction,
                object_from_hand=object_from_hand,
                speed_bound_m_per_unit=float(speed_rows[row_index]),
                spatial_error_bound_m=spatial_error_bound_m,
                counters=counters,
            )
            if solved is None:
                continue
            event, phase_error = solved
            candidates.append(
                (
                    event.normalized_closure,
                    phase_error,
                    witness_index,
                    event,
                )
            )
        if not candidates:
            return None
        earliest = min(candidates, key=lambda row: (row[0], row[2]))
        tied = [
            row
            for row in candidates
            if abs(row[0] - earliest[0]) <= row[1] + earliest[1]
        ]
        tied.sort(key=lambda row: row[2])
        return tied[0][3]

    def _search_pad_first_contact(
        self,
        *,
        prepared: _PreparedPad,
        q_start: np.ndarray,
        direction: np.ndarray,
        maximum_parameter: float,
        object_from_hand: np.ndarray,
        spatial_error_bound_m: float,
        budget: _Budget,
        counters: _PadCounters,
        execution: _GeometryExecutionContext,
    ) -> _ContactEvent | None:
        speed_bounds = self._witness_speed_bounds(
            prepared, q_start, direction, maximum_parameter
        )

        def search(lower: float, upper: float) -> _ContactEvent | None:
            budget.consume()
            counters.interval_evaluations += 1
            midpoint = lower + 0.5 * (upper - lower)
            q_midpoint = q_start + midpoint * direction
            states, state_key = self._cached_witness_states(
                prepared,
                q_midpoint,
                direction,
                object_from_hand,
                execution,
            )
            interval_half_width = 0.5 * (upper - lower)
            enclosure_radii = speed_bounds * interval_half_width
            geometry = self._interval_geometry(
                prepared=prepared,
                states=states,
                state_key=state_key,
                enclosure_radii_m=enclosure_radii,
                spatial_error_bound_m=spatial_error_bound_m,
                counters=counters,
                execution=execution,
            )
            possible = geometry.possible
            if not np.any(possible):
                if geometry.minimum_free_margin_m is None:
                    raise RayClosureError(
                        "free interval lacks an exact minimum witness margin"
                    )
                counters.certified_free_intervals += 1
                counters.clearance_lower_bound_m = min(
                    counters.clearance_lower_bound_m,
                    max(0.0, geometry.minimum_free_margin_m),
                )
                return None

            possible_indices = np.flatnonzero(possible)
            feature_event = self._finite_chord_feature_event(
                prepared=prepared,
                states_midpoint=states,
                geometry=geometry,
                speed_bounds_m_per_unit=speed_bounds,
                lower=lower,
                upper=upper,
                q_start=q_start,
                direction=direction,
                object_from_hand=object_from_hand,
                spatial_error_bound_m=spatial_error_bound_m,
                counters=counters,
            )
            if feature_event is not None:
                root_lower = max(
                    lower,
                    feature_event.normalized_closure
                    - 0.5 * feature_event.interval_width,
                )
                if np.nextafter(lower, root_lower) < root_lower:
                    earlier = search(lower, root_lower)
                    if earlier is not None:
                        earlier_upper = float(
                            np.nextafter(
                                earlier.normalized_closure
                                + 0.5 * earlier.interval_width,
                                math.inf,
                            )
                        )
                        disallowed_root = feature_event.classification != (
                            "ALLOWED_DIRECTIONAL_FIRST_CONTACT"
                        )
                        if (
                            earlier.classification
                            == "NUMERICALLY_UNRESOLVED_FIRST_PROXIMITY"
                            and disallowed_root
                            and earlier_upper >= root_lower
                        ):
                            # The unresolved terminal interval and the
                            # independently solved feature-root certificate
                            # describe the same numerically inseparable event.
                            # A disallowed root may conservatively absorb that
                            # overlap.  An allowed root may not: in that case
                            # the unresolved prefix remains fail-closed.
                            earlier_lower = float(
                                np.nextafter(
                                    earlier.normalized_closure
                                    - 0.5 * earlier.interval_width,
                                    -math.inf,
                                )
                            )
                            feature_upper = float(
                                np.nextafter(
                                    feature_event.normalized_closure
                                    + 0.5 * feature_event.interval_width,
                                    math.inf,
                                )
                            )
                            half_width = max(
                                feature_event.normalized_closure
                                - earlier_lower,
                                feature_upper
                                - feature_event.normalized_closure,
                            )
                            return _ContactEvent(
                                normalized_closure=(
                                    feature_event.normalized_closure
                                ),
                                interval_width=float(
                                    np.nextafter(
                                        2.0 * half_width, math.inf
                                    )
                                ),
                                spatial_error_bound_m=max(
                                    earlier.spatial_error_bound_m,
                                    feature_event.spatial_error_bound_m,
                                ),
                                triangle_index=(
                                    feature_event.triangle_index
                                ),
                                witness_index=feature_event.witness_index,
                                witness_flat_index=(
                                    feature_event.witness_flat_index
                                ),
                                object_face_index=(
                                    feature_event.object_face_index
                                ),
                                classification=(
                                    feature_event.classification
                                ),
                                first_hit=feature_event.first_hit,
                            )
                        return earlier
                return feature_event
            maximum_possible_radius = float(np.max(enclosure_radii[possible_indices]))
            adjacent = np.nextafter(lower, upper) >= upper
            if maximum_possible_radius <= spatial_error_bound_m or adjacent:
                # A ray result is consumed only when this interval can no
                # longer be subdivided.  Casting it earlier is dead work: the
                # recursive branches recompute exact-FK states at their own
                # midpoints and cannot reuse a parent ray geometrically.
                ray_rows: dict[int, FirstHitResult] = {}
                counters.leading_evaluations += int(
                    np.count_nonzero(possible & states.leading)
                )
                counters.rays += len(possible_indices)
                for flat_index in possible_indices:
                    index = int(flat_index)
                    ray_rows[index] = self.intersector.first_hit(
                        states.positions_object_m[index],
                        states.velocities_object_per_unit[index],
                    )
                candidates: list[
                    tuple[float, float, int, _ContactEvent]
                ] = []
                unresolved = False
                for flat_index in possible_indices:
                    index = int(flat_index)
                    ray = ray_rows.get(index)
                    if ray is None or not ray.hit:
                        unresolved = True
                        continue
                    assert ray.distance_m is not None
                    velocity = states.velocities_object_per_unit[index]
                    speed = float(np.linalg.norm(velocity))
                    if speed == 0.0:
                        unresolved = True
                        continue
                    ray_reach = enclosure_radii[index] + spatial_error_bound_m
                    if ray.distance_m > ray_reach + ray.distance_error_bound_m:
                        unresolved = True
                        continue
                    predicted = midpoint + max(0.0, ray.distance_m) / speed
                    parameter_error = (
                        spatial_error_bound_m + ray.distance_error_bound_m
                    ) / speed + _TIME_ERROR * abs(predicted)
                    if predicted < lower - parameter_error or predicted > upper + parameter_error:
                        unresolved = True
                        continue
                    predicted = min(upper, max(lower, predicted))
                    assert ray.face_index is not None
                    pad_approach = float(
                        states.pad_source_winding_normals_object[index]
                        @ velocity
                    )
                    object_approach = -float(
                        np.asarray(ray.outward_normal) @ velocity
                    )
                    object_error = _DOT_ERROR * speed
                    if not self._contact_face_mask[ray.face_index]:
                        classification = "FORBIDDEN_SEMANTIC_FIRST_CONTACT"
                    elif pad_approach <= states.leading_error_bounds[index]:
                        classification = "PAD_NORMAL_DOMAIN_REJECTED"
                    elif object_approach <= object_error:
                        classification = "OBJECT_NORMAL_DOMAIN_REJECTED"
                    else:
                        classification = "ALLOWED_DIRECTIONAL_FIRST_CONTACT"
                    event = _ContactEvent(
                        normalized_closure=predicted,
                        interval_width=upper - lower,
                        spatial_error_bound_m=(
                            spatial_error_bound_m + enclosure_radii[index]
                        ),
                        triangle_index=int(prepared.triangle_indices[index]),
                        witness_index=int(prepared.witness_indices[index]),
                        witness_flat_index=index,
                        object_face_index=int(ray.face_index),
                        classification=classification,
                        first_hit=ray,
                    )
                    candidates.append((predicted, parameter_error, index, event))
                if candidates:
                    earliest = min(candidates, key=lambda row: (row[0], row[2]))
                    earliest_time, earliest_error, _earliest_index, _event = earliest
                    tied = [
                        row
                        for row in candidates
                        if abs(row[0] - earliest_time)
                        <= row[1] + earliest_error
                    ]
                    tied.sort(key=lambda row: row[2])
                    return tied[0][3]
                if unresolved:
                    return _ContactEvent(
                        normalized_closure=midpoint,
                        interval_width=upper - lower,
                        spatial_error_bound_m=(
                            spatial_error_bound_m + maximum_possible_radius
                        ),
                        triangle_index=int(
                            prepared.triangle_indices[int(possible_indices[0])]
                        ),
                        witness_index=int(
                            prepared.witness_indices[int(possible_indices[0])]
                        ),
                        witness_flat_index=int(possible_indices[0]),
                        object_face_index=int(
                            geometry.nearest_face_indices[
                                int(possible_indices[0])
                            ]
                        ),
                        classification="NUMERICALLY_UNRESOLVED_FIRST_PROXIMITY",
                        first_hit=None,
                    )
                return None

            middle = midpoint
            first = search(lower, middle)
            if first is not None:
                return first
            return search(middle, upper)

        return search(0.0, maximum_parameter)

    def _pad_audit(
        self,
        prepared: _PreparedPad,
        counters: _PadCounters,
        event: _ContactEvent | None,
        classification: str | None = None,
    ) -> PadClosureAudit:
        clearance = counters.clearance_lower_bound_m
        if not math.isfinite(clearance):
            clearance = 0.0
        resolved_classification = (
            classification
            if classification is not None
            else ("NO_CONTACT_WITHIN_JOINT_LIMITS" if event is None else event.classification)
        )
        return PadClosureAudit(
            pad_name=prepared.verified.name,
            finger_name=prepared.verified.finger_name,
            verified_triangle_count=prepared.verified.triangle_count,
            witness_count=len(prepared.witness_points_link_m),
            exact_fk_interval_evaluations=counters.interval_evaluations,
            leading_witness_evaluations=counters.leading_evaluations,
            first_hit_rays_cast=counters.rays,
            finite_chord_feature_candidates=(
                counters.finite_chord_feature_candidates
            ),
            nonlinear_feature_roots_solved=(
                counters.nonlinear_feature_roots_solved
            ),
            nonlinear_root_fk_evaluations=(
                counters.nonlinear_root_fk_evaluations
            ),
            distance_bvh_node_visits=counters.distance_node_visits,
            distance_triangle_tests=counters.distance_triangle_tests,
            certified_free_interval_count=counters.certified_free_intervals,
            certified_witness_path_clearance_lower_bound_m=clearance,
            interval_point_motion_evaluations=(
                counters.interval_point_motion_evaluations
            ),
            swept_face_candidate_count=counters.swept_face_candidates,
            interval_pair_evaluation_count=counters.interval_pair_evaluations,
            actual_plane_root_evaluation_count=(
                counters.actual_plane_root_evaluations
            ),
            batch_root_triangle_free_pair_count=(
                counters.batch_root_triangle_free_pairs
            ),
            batch_root_triangle_uncertain_pair_count=(
                counters.batch_root_triangle_uncertain_pairs
            ),
            batch_plane_monotone_same_side_free_count=(
                counters.batch_plane_monotone_same_side_free
            ),
            shared_plane_gate_root_count=(
                counters.shared_plane_gate_roots
            ),
            parallel_plane_root_task_count=(
                counters.parallel_plane_root_tasks
            ),
            pre_root_spatial_enclosure_group_count=(
                counters.pre_root_spatial_enclosure_groups
            ),
            pre_root_spatial_free_pair_count=(
                counters.pre_root_spatial_free_pairs
            ),
            pre_root_spatial_fully_free_group_count=(
                counters.pre_root_spatial_fully_free_groups
            ),
            parent_certified_free_pair_prune_count=(
                counters.parent_certified_free_pair_prunes
            ),
            parent_certified_root_outside_pair_prune_count=(
                counters.parent_certified_root_outside_pair_prunes
            ),
            parent_certified_root_pair_reuse_count=(
                counters.parent_certified_root_pair_reuses
            ),
            large_exact_batch_temporal_deferral_count=(
                counters.large_exact_batch_temporal_deferrals
            ),
            large_exact_batch_deferred_root_group_count=(
                counters.large_exact_batch_deferred_root_groups
            ),
            swept_face_witness_stage_count=(
                counters.swept_face_witness_stages
            ),
            swept_face_witness_materialized_count=(
                counters.swept_face_witnesses_materialized
            ),
            staged_potential_root_temporal_deferral_count=(
                counters.staged_potential_root_temporal_deferrals
            ),
            staged_potential_root_group_count=(
                counters.staged_potential_root_groups
            ),
            staged_unmaterialized_witness_count=(
                counters.staged_unmaterialized_witnesses
            ),
            parent_frontier_geometry_bypass_count=(
                counters.parent_frontier_geometry_bypasses
            ),
            pre_nearest_aabb_witness_test_count=(
                counters.pre_nearest_aabb_witness_tests
            ),
            pre_nearest_aabb_certified_free_witness_count=(
                counters.pre_nearest_aabb_certified_free_witnesses
            ),
            pre_nearest_aabb_exact_survivor_count=(
                counters.pre_nearest_aabb_exact_survivors
            ),
            pre_nearest_aabb_fast_path_count=(
                counters.pre_nearest_aabb_fast_paths
            ),
            pre_nearest_aabb_fallback_count=(
                counters.pre_nearest_aabb_fallbacks
            ),
            root_interpolation_iteration_count=(
                counters.root_interpolation_iterations
            ),
            interval_newton_iteration_count=(
                counters.interval_newton_iterations
            ),
            root_bisection_iteration_count=(
                counters.root_bisection_iterations
            ),
            whole_path_sphere_screen_segment_count=(
                counters.whole_path_sphere_screen_segments
            ),
            whole_path_sphere_screen_query_count=(
                counters.whole_path_sphere_screen_queries
            ),
            whole_path_sphere_screen_bvh_node_visits=(
                counters.whole_path_sphere_screen_bvh_node_visits
            ),
            whole_path_sphere_screen_triangle_tests=(
                counters.whole_path_sphere_screen_triangle_tests
            ),
            whole_path_sphere_screen_obb_sat_certified_free_node_count=(
                counters.whole_path_sphere_screen_obb_sat_certified_free_nodes
            ),
            whole_path_sphere_screen_obb_sat_triangle_test_count=(
                counters.whole_path_sphere_screen_obb_sat_triangle_tests
            ),
            whole_path_sphere_screen_moving_triangle_sat_certified_free_pair_count=(
                counters.whole_path_sphere_screen_moving_triangle_sat_certified_free_pairs
            ),
            whole_path_sphere_screen_moving_triangle_sat_pair_test_count=(
                counters.whole_path_sphere_screen_moving_triangle_sat_pair_tests
            ),
            whole_path_sphere_screen_temporal_refined_leaf_pair_count=(
                counters.whole_path_sphere_screen_temporal_refined_leaf_pairs
            ),
            whole_path_sphere_screen_temporal_refinement_transform_count=(
                counters.whole_path_sphere_screen_temporal_refinement_transforms
            ),
            whole_path_sphere_screen_maximum_temporal_refinement_depth_reached=(
                counters.whole_path_sphere_screen_maximum_temporal_refinement_depth
            ),
            whole_path_sphere_screen_narrowphase_refinement_used=(
                counters.whole_path_sphere_screen_narrowphase_refinement_used
            ),
            whole_path_sphere_screen_narrowphase_work_budget_exhausted=(
                counters.whole_path_sphere_screen_narrowphase_work_budget_exhausted
            ),
            whole_path_sphere_screen_directional_contact_feasibility_used=(
                counters.whole_path_sphere_screen_directional_contact_feasibility_used
            ),
            whole_path_sphere_screen_directional_bvh_node_pair_test_count=(
                counters.whole_path_sphere_screen_directional_bvh_node_pair_tests
            ),
            whole_path_sphere_screen_directional_bvh_node_pair_rejected_count=(
                counters.whole_path_sphere_screen_directional_bvh_node_pair_rejections
            ),
            whole_path_sphere_screen_directional_leaf_face_pair_test_count=(
                counters.whole_path_sphere_screen_directional_leaf_face_pair_tests
            ),
            whole_path_sphere_screen_directional_leaf_face_pair_rejected_count=(
                counters.whole_path_sphere_screen_directional_leaf_face_pair_rejections
            ),
            whole_path_sphere_screen_directional_interval_witness_motion_evaluation_count=(
                counters.whole_path_sphere_screen_directional_interval_witness_motion_evaluations
            ),
            whole_path_sphere_screen_certified_no_valid_contact=(
                counters.whole_path_sphere_screen_certified_no_valid_contact
            ),
            whole_path_sphere_screen_certified_free=(
                counters.whole_path_sphere_screen_certified_free
            ),
            whole_path_sphere_screen_clearance_lower_bound_m=(
                counters.whole_path_sphere_screen_clearance_lower_bound_m
            ),
            certified_contact_root_count=counters.certified_contact_roots,
            unresolved_witness_face_pair_count=(
                counters.unresolved_witness_face_pairs
            ),
            cofirst_root_count=counters.cofirst_root_count,
            competing_root_order_block_count=(
                counters.competing_root_order_blocks
            ),
            acceptance_ray_call_count=counters.rays,
            selected_triangle_index=None if event is None else event.triangle_index,
            selected_witness_index=None if event is None else event.witness_index,
            selected_object_face_index=(
                None if event is None else event.object_face_index
            ),
            selected_normalized_closure=(
                None if event is None else event.normalized_closure
            ),
            selected_closure_interval_width=(
                None if event is None else event.interval_width
            ),
            selected_spatial_error_bound_m=(
                None if event is None else event.spatial_error_bound_m
            ),
            selected_root_phase_lower=None,
            selected_root_phase_upper=None,
            selected_pad_approach_lower=None,
            selected_path_local_free_side_approach_lower=None,
            selected_object_source_winding_free_side_sign=None,
            first_contact_classification=resolved_classification,
        )

    def _pad_audit_v9(
        self,
        prepared: _PreparedPad,
        counters: _PadCounters,
        outcome: _PadSearchOutcome,
        classification: str | None = None,
    ) -> PadClosureAudit:
        if counters.rays != 0:
            raise RayClosureError(
                "V9 acceptance audit cannot contain a ray call"
            )
        clearance = counters.clearance_lower_bound_m
        if not math.isfinite(clearance):
            clearance = 0.0
        possible_set = outcome.possible_first_contact_set
        root = (
            None
            if possible_set is None
            else possible_set.display_proposal_root
        )
        certificate = None if root is None else root.certificate
        if classification is None:
            if outcome.state is _PadSearchState.CERTIFIED_FREE:
                resolved_classification = "NO_CONTACT_WITHIN_JOINT_LIMITS"
            elif outcome.state is _PadSearchState.UNRESOLVED:
                resolved_classification = "UNRESOLVED_FIRST_CONTACT"
            else:
                root_classes = {
                    row.semantic_classification for row in outcome.roots
                }
                resolved_classification = (
                    next(iter(root_classes))
                    if len(root_classes) == 1
                    else "POSSIBLE_EARLIEST_CONTACT_CLASSIFICATION_CONFLICT"
                )
        else:
            resolved_classification = classification
        if certificate is None:
            spatial_bound = None
        else:
            widths = np.asarray(
                [
                    row.upper - row.lower
                    for row in certificate.position_object_m
                ],
                dtype=np.float64,
            )
            spatial_bound = 0.5 * float(np.linalg.norm(widths))
        return PadClosureAudit(
            pad_name=prepared.verified.name,
            finger_name=prepared.verified.finger_name,
            verified_triangle_count=prepared.verified.triangle_count,
            witness_count=len(prepared.witness_points_link_m),
            exact_fk_interval_evaluations=counters.interval_evaluations,
            leading_witness_evaluations=0,
            first_hit_rays_cast=0,
            finite_chord_feature_candidates=0,
            nonlinear_feature_roots_solved=0,
            nonlinear_root_fk_evaluations=0,
            distance_bvh_node_visits=counters.distance_node_visits,
            distance_triangle_tests=counters.distance_triangle_tests,
            certified_free_interval_count=counters.certified_free_intervals,
            certified_witness_path_clearance_lower_bound_m=clearance,
            interval_point_motion_evaluations=(
                counters.interval_point_motion_evaluations
            ),
            swept_face_candidate_count=counters.swept_face_candidates,
            interval_pair_evaluation_count=counters.interval_pair_evaluations,
            actual_plane_root_evaluation_count=(
                counters.actual_plane_root_evaluations
            ),
            batch_root_triangle_free_pair_count=(
                counters.batch_root_triangle_free_pairs
            ),
            batch_root_triangle_uncertain_pair_count=(
                counters.batch_root_triangle_uncertain_pairs
            ),
            batch_plane_monotone_same_side_free_count=(
                counters.batch_plane_monotone_same_side_free
            ),
            shared_plane_gate_root_count=(
                counters.shared_plane_gate_roots
            ),
            parallel_plane_root_task_count=(
                counters.parallel_plane_root_tasks
            ),
            pre_root_spatial_enclosure_group_count=(
                counters.pre_root_spatial_enclosure_groups
            ),
            pre_root_spatial_free_pair_count=(
                counters.pre_root_spatial_free_pairs
            ),
            pre_root_spatial_fully_free_group_count=(
                counters.pre_root_spatial_fully_free_groups
            ),
            parent_certified_free_pair_prune_count=(
                counters.parent_certified_free_pair_prunes
            ),
            parent_certified_root_outside_pair_prune_count=(
                counters.parent_certified_root_outside_pair_prunes
            ),
            parent_certified_root_pair_reuse_count=(
                counters.parent_certified_root_pair_reuses
            ),
            large_exact_batch_temporal_deferral_count=(
                counters.large_exact_batch_temporal_deferrals
            ),
            large_exact_batch_deferred_root_group_count=(
                counters.large_exact_batch_deferred_root_groups
            ),
            swept_face_witness_stage_count=(
                counters.swept_face_witness_stages
            ),
            swept_face_witness_materialized_count=(
                counters.swept_face_witnesses_materialized
            ),
            staged_potential_root_temporal_deferral_count=(
                counters.staged_potential_root_temporal_deferrals
            ),
            staged_potential_root_group_count=(
                counters.staged_potential_root_groups
            ),
            staged_unmaterialized_witness_count=(
                counters.staged_unmaterialized_witnesses
            ),
            parent_frontier_geometry_bypass_count=(
                counters.parent_frontier_geometry_bypasses
            ),
            pre_nearest_aabb_witness_test_count=(
                counters.pre_nearest_aabb_witness_tests
            ),
            pre_nearest_aabb_certified_free_witness_count=(
                counters.pre_nearest_aabb_certified_free_witnesses
            ),
            pre_nearest_aabb_exact_survivor_count=(
                counters.pre_nearest_aabb_exact_survivors
            ),
            pre_nearest_aabb_fast_path_count=(
                counters.pre_nearest_aabb_fast_paths
            ),
            pre_nearest_aabb_fallback_count=(
                counters.pre_nearest_aabb_fallbacks
            ),
            root_interpolation_iteration_count=(
                counters.root_interpolation_iterations
            ),
            interval_newton_iteration_count=(
                counters.interval_newton_iterations
            ),
            root_bisection_iteration_count=(
                counters.root_bisection_iterations
            ),
            whole_path_sphere_screen_segment_count=(
                counters.whole_path_sphere_screen_segments
            ),
            whole_path_sphere_screen_query_count=(
                counters.whole_path_sphere_screen_queries
            ),
            whole_path_sphere_screen_bvh_node_visits=(
                counters.whole_path_sphere_screen_bvh_node_visits
            ),
            whole_path_sphere_screen_triangle_tests=(
                counters.whole_path_sphere_screen_triangle_tests
            ),
            whole_path_sphere_screen_obb_sat_certified_free_node_count=(
                counters.whole_path_sphere_screen_obb_sat_certified_free_nodes
            ),
            whole_path_sphere_screen_obb_sat_triangle_test_count=(
                counters.whole_path_sphere_screen_obb_sat_triangle_tests
            ),
            whole_path_sphere_screen_moving_triangle_sat_certified_free_pair_count=(
                counters.whole_path_sphere_screen_moving_triangle_sat_certified_free_pairs
            ),
            whole_path_sphere_screen_moving_triangle_sat_pair_test_count=(
                counters.whole_path_sphere_screen_moving_triangle_sat_pair_tests
            ),
            whole_path_sphere_screen_temporal_refined_leaf_pair_count=(
                counters.whole_path_sphere_screen_temporal_refined_leaf_pairs
            ),
            whole_path_sphere_screen_temporal_refinement_transform_count=(
                counters.whole_path_sphere_screen_temporal_refinement_transforms
            ),
            whole_path_sphere_screen_maximum_temporal_refinement_depth_reached=(
                counters.whole_path_sphere_screen_maximum_temporal_refinement_depth
            ),
            whole_path_sphere_screen_narrowphase_refinement_used=(
                counters.whole_path_sphere_screen_narrowphase_refinement_used
            ),
            whole_path_sphere_screen_narrowphase_work_budget_exhausted=(
                counters.whole_path_sphere_screen_narrowphase_work_budget_exhausted
            ),
            whole_path_sphere_screen_directional_contact_feasibility_used=(
                counters.whole_path_sphere_screen_directional_contact_feasibility_used
            ),
            whole_path_sphere_screen_directional_bvh_node_pair_test_count=(
                counters.whole_path_sphere_screen_directional_bvh_node_pair_tests
            ),
            whole_path_sphere_screen_directional_bvh_node_pair_rejected_count=(
                counters.whole_path_sphere_screen_directional_bvh_node_pair_rejections
            ),
            whole_path_sphere_screen_directional_leaf_face_pair_test_count=(
                counters.whole_path_sphere_screen_directional_leaf_face_pair_tests
            ),
            whole_path_sphere_screen_directional_leaf_face_pair_rejected_count=(
                counters.whole_path_sphere_screen_directional_leaf_face_pair_rejections
            ),
            whole_path_sphere_screen_directional_interval_witness_motion_evaluation_count=(
                counters.whole_path_sphere_screen_directional_interval_witness_motion_evaluations
            ),
            whole_path_sphere_screen_certified_no_valid_contact=(
                counters.whole_path_sphere_screen_certified_no_valid_contact
            ),
            whole_path_sphere_screen_certified_free=(
                counters.whole_path_sphere_screen_certified_free
            ),
            whole_path_sphere_screen_clearance_lower_bound_m=(
                counters.whole_path_sphere_screen_clearance_lower_bound_m
            ),
            certified_contact_root_count=counters.certified_contact_roots,
            unresolved_witness_face_pair_count=(
                counters.unresolved_witness_face_pairs
            ),
            cofirst_root_count=counters.cofirst_root_count,
            possible_earliest_root_count=(
                counters.possible_earliest_root_count
            ),
            possible_first_contact_set_sha256=(
                None if possible_set is None else possible_set.set_sha256
            ),
            competing_root_order_block_count=(
                counters.competing_root_order_blocks
            ),
            acceptance_ray_call_count=0,
            selected_triangle_index=(
                None if root is None else root.pad_triangle_index
            ),
            selected_witness_index=(
                None if root is None else root.witness_index
            ),
            selected_object_face_index=(
                None if root is None else root.object_face_index
            ),
            selected_normalized_closure=(
                None if certificate is None else certificate.representative_phase
            ),
            selected_normalized_closure_role=(
                None
                if certificate is None
                else certificate.representative_phase_role
            ),
            selected_closure_interval_width=(
                None
                if certificate is None
                else certificate.phase.upper - certificate.phase.lower
            ),
            selected_spatial_error_bound_m=spatial_bound,
            selected_root_phase_lower=(
                None if certificate is None else certificate.phase.lower
            ),
            selected_root_phase_upper=(
                None if certificate is None else certificate.phase.upper
            ),
            selected_pad_approach_lower=(
                None if certificate is None else certificate.pad_approach.lower
            ),
            selected_path_local_free_side_approach_lower=(
                None
                if certificate is None
                else certificate.path_local_free_side_approach.lower
            ),
            selected_object_source_winding_free_side_sign=(
                None
                if certificate is None
                else certificate.object_source_winding_free_side_sign
            ),
            first_contact_classification=resolved_classification,
        )

    def _audit(
        self,
        *,
        budget: _Budget,
        pad_audits: Sequence[PadClosureAudit],
        failure_reason: str | None,
        budget_exhausted: bool = False,
        candidate_role: str = NO_CANDIDATE_ROLE,
    ) -> RayClosureAudit:
        # A successful closing path terminates on an authorised contact
        # surface, so the certified lower bound over the complete closed path
        # is exactly zero.  Reporting a positive number here would silently
        # exclude the endpoint and overstate trajectory clearance.
        trajectory_clearance = 0.0 if failure_reason is None else (
            min(
                row.certified_witness_path_clearance_lower_bound_m
                for row in pad_audits
            )
            if pad_audits
            else 0.0
        )
        return RayClosureAudit(
            method_id=METHOD_ID,
            numerical_policy=NUMERICAL_POLICY,
            witness_rule=WITNESS_RULE,
            interval_rule=INTERVAL_RULE,
            distance_bvh_rule=DISTANCE_BVH_RULE,
            ray_evaluation_policy=RAY_EVALUATION_POLICY,
            feature_root_policy=FEATURE_ROOT_POLICY,
            object_contact_normal_policy=(
                self.object_contact_normal_policy
            ),
            pad_surface_normal_policy=self.pad_surface_normal_policy,
            parameter_layout=self.parameter_layout,
            pad_order=tuple(pad.verified.name for pad in self.prepared_pads),
            full_verified_pad_mesh_used=True,
            pad_face_subset_input_allowed=False,
            independent_actuation_supports=self.independent_actuation_supports,
            closure_parameter_domain_id=CLOSURE_PARAMETER_DOMAIN_ID,
            closure_suffix_dominance_argument=CLOSURE_SUFFIX_DOMINANCE_ARGUMENT,
            preshape_joint_names=self.preshape_joint_names,
            closure_open_joint_positions_rad=self.closure_open_joint_positions_rad,
            maximum_subdivision_intervals=budget.maximum,
            interval_arithmetic_method_id=INTERVAL_KINEMATICS_METHOD_ID,
            interval_decimal_precision=(
                self.interval_arithmetic_options.decimal_precision
            ),
            maximum_root_bisection_iterations=(
                self.interval_arithmetic_options.maximum_root_bisection_iterations
            ),
            subdivision_intervals_used=budget.used,
            subdivision_budget_exhausted=budget_exhausted,
            internal_force_role=INTERNAL_FORCE_ROLE,
            trajectory_clearance_m=trajectory_clearance,
            trajectory_clearance_role=TRAJECTORY_CLEARANCE_ROLE,
            task_frame_source=self.task_frame.source,
            closure_focus_method=CLOSURE_FOCUS_METHOD,
            distance_bvh_node_count=self.distance_bvh.node_count,
            pad_audits=tuple(pad_audits),
            claim_limitations=CLAIM_LIMITATIONS,
            failure_reason=failure_reason,
            model_binding_complete=self.model_binding_complete,
            model_binding_status=self.model_binding_status,
            object_geometry_sha256=self.object_geometry_sha256,
            model_contract_sha256=self.model_contract_sha256,
            pad_geometry_sha256=self.pad_geometry_sha256,
            pad_runtime_geometry_sha256=(
                self.pad_runtime_geometry_sha256
            ),
            pad_link_names=self.pad_link_names,
            closing_directions_physical=(
                self.closing_directions_physical_tuple
            ),
            model_contract_canonical_json=(
                self.model_contract_canonical_json
            ),
            candidate_role=candidate_role,
            candidate_exact_contact_endpoint_certified=False,
            display_approximation_role=DISPLAY_APPROXIMATION_ROLE,
            possible_first_contact_set_sha256=tuple(
                row.possible_first_contact_set_sha256
                for row in pad_audits
                if row.possible_first_contact_set_sha256 is not None
            ),
        )

    def evaluate_unit_parameters(
        self,
        parameters_unit: Sequence[float],
        hand_model: ThreeFingerHandModel | None = None,
    ) -> RayClosureEvaluation:
        return self._evaluate_unit_parameters_with_execution(
            parameters_unit,
            hand_model,
            execution=_GeometryExecutionContext(),
        )

    def _evaluate_unit_parameters_with_execution(
        self,
        parameters_unit: Sequence[float],
        hand_model: ThreeFingerHandModel | None,
        *,
        execution: _GeometryExecutionContext,
        _use_whole_path_sphere_screen: bool = True,
    ) -> RayClosureEvaluation:
        if not isinstance(execution, _GeometryExecutionContext):
            raise RayClosureError("invalid ray-closure geometry execution context")
        if type(_use_whole_path_sphere_screen) is not bool:
            raise RayClosureError(
                "whole-path sphere screen switch must be boolean"
            )
        if execution.witness_state_cache or execution.nearest_batch_cache:
            raise RayClosureError(
                "geometry execution context caches must be empty at evaluation start"
            )
        supplied_hand = self.hand_model if hand_model is None else hand_model
        self._validate_hand(supplied_hand)
        budget = _Budget(self.maximum_subdivision_intervals)
        try:
            (
                q_start,
                _target,
                _rotation,
                object_from_hand,
                hand_extent,
            ) = self._decode_with_object_from_hand(parameters_unit)
        except RayClosureError as error:
            audit = self._audit(
                budget=budget,
                pad_audits=(),
                failure_reason=f"PARAMETER_DOMAIN_REJECTED:{error}",
            )
            return RayClosureEvaluation(None, audit)
        spatial_error = (
            self.intersector.distance_error_bound_m
            + self.distance_bvh.aabb_error_bound_m
            + _FK_ERROR
            * (self.intersector.characteristic_length_m + hand_extent)
        )

        sphere_screens: tuple[WholePathPadSphereScreen, ...] = ()
        if _use_whole_path_sphere_screen:
            sphere_screens = (
                self._classify_whole_path_pad_aabb_hierarchies(
                    self._whole_path_pad_aabb_hierarchy_coverages(
                        q_start=q_start,
                        object_from_hand=object_from_hand,
                        spatial_error_bound_m=spatial_error,
                    ),
                    enable_moving_triangle_refinement=False,
                    enable_directional_contact_feasibility=False,
                )
            )
            certified_free_rows = tuple(
                (row_index, screen)
                for row_index, screen in enumerate(sphere_screens)
                if screen.certified_free
            )
            if certified_free_rows:
                row_index, screen = max(
                    certified_free_rows,
                    key=lambda row: (
                        row[1].minimum_clearance_lower_bound_m,
                        -row[0],
                    ),
                )
                prepared = self.prepared_pads[row_index]
                direction = self.closing_directions_physical[row_index]
                maximum_parameter = self._maximum_path_parameter(
                    q_start, direction
                )
                counters = _PadCounters()
                self._bind_whole_path_sphere_screen_to_counters(
                    counters, screen
                )
                outcome = _PadSearchOutcome(
                    state=_PadSearchState.CERTIFIED_FREE,
                    interval_lower=0.0,
                    interval_upper=maximum_parameter,
                    possible_first_contact_set=None,
                    unresolved_reason=None,
                )
                execution.stats.fail_closed_fingers_skipped = (
                    len(self.prepared_pads) - 1
                )
                audit = self._audit(
                    budget=budget,
                    pad_audits=(
                        self._pad_audit_v9(
                            prepared, counters, outcome
                        ),
                    ),
                    failure_reason=(
                        "NO_FIRST_CONTACT_FOR_PAD:"
                        f"{prepared.verified.name}"
                    ),
                )
                return RayClosureEvaluation(None, audit)

        outcomes: list[_PadSearchOutcome] = []
        pad_audits: list[PadClosureAudit] = []
        try:
            for row_index, prepared in enumerate(self.prepared_pads):
                direction = self.closing_directions_physical[row_index]
                maximum_parameter = self._maximum_path_parameter(q_start, direction)
                counters = _PadCounters()
                if sphere_screens:
                    self._bind_whole_path_sphere_screen_to_counters(
                        counters, sphere_screens[row_index]
                    )
                outcome = self._search_pad_first_contact_v9(
                    prepared=prepared,
                    q_start=q_start,
                    direction=direction,
                    maximum_parameter=maximum_parameter,
                    object_from_hand=object_from_hand,
                    spatial_error_bound_m=spatial_error,
                    budget=budget,
                    counters=counters,
                    execution=execution,
                )
                pad_audits.append(
                    self._pad_audit_v9(prepared, counters, outcome)
                )
                if outcome.state is _PadSearchState.CERTIFIED_FREE:
                    execution.stats.fail_closed_fingers_skipped = (
                        len(self.prepared_pads) - row_index - 1
                    )
                    audit = self._audit(
                        budget=budget,
                        pad_audits=pad_audits,
                        failure_reason=f"NO_FIRST_CONTACT_FOR_PAD:{prepared.verified.name}",
                    )
                    return RayClosureEvaluation(None, audit)
                if outcome.state is _PadSearchState.UNRESOLVED:
                    execution.stats.fail_closed_fingers_skipped = (
                        len(self.prepared_pads) - row_index - 1
                    )
                    audit = self._audit(
                        budget=budget,
                        pad_audits=pad_audits,
                        failure_reason=(
                            "UNRESOLVED_FIRST_CONTACT:"
                            f"{prepared.verified.name}:"
                            f"{outcome.unresolved_reason}"
                        ),
                    )
                    return RayClosureEvaluation(None, audit)
                root_classes = {
                    root.semantic_classification for root in outcome.roots
                }
                if len(root_classes) != 1:
                    execution.stats.fail_closed_fingers_skipped = (
                        len(self.prepared_pads) - row_index - 1
                    )
                    audit = self._audit(
                        budget=budget,
                        pad_audits=pad_audits,
                        failure_reason=(
                            "POSSIBLE_EARLIEST_CONTACT_CLASSIFICATION_CONFLICT:"
                            f"{prepared.verified.name}"
                        ),
                    )
                    return RayClosureEvaluation(None, audit)
                classification = next(iter(root_classes))
                object_planes = {
                    _exact_dyadic_plane_key(
                        self.canonical_object_face_vertices_m[
                            root.object_face_index
                        ]
                    )
                    for root in outcome.roots
                }
                if len(object_planes) != 1:
                    execution.stats.fail_closed_fingers_skipped = (
                        len(self.prepared_pads) - row_index - 1
                    )
                    audit = self._audit(
                        budget=budget,
                        pad_audits=pad_audits,
                        failure_reason=(
                            "POSSIBLE_EARLIEST_CONTACT_NORMAL_SET_UNSUPPORTED:"
                            f"{prepared.verified.name}"
                        ),
                    )
                    return RayClosureEvaluation(None, audit)
                if classification != (
                    "ALLOWED_PATH_LOCAL_FREE_SIDE_TRANSVERSE_CONTACT"
                ):
                    execution.stats.fail_closed_fingers_skipped = (
                        len(self.prepared_pads) - row_index - 1
                    )
                    audit = self._audit(
                        budget=budget,
                        pad_audits=pad_audits,
                        failure_reason=(
                            f"{classification}:{prepared.verified.name}"
                        ),
                    )
                    return RayClosureEvaluation(None, audit)
                outcomes.append(outcome)
        except _SubdivisionBudgetExhausted:
            audit = self._audit(
                budget=budget,
                pad_audits=pad_audits,
                failure_reason="MAXIMUM_SUBDIVISION_INTERVALS_EXHAUSTED",
                budget_exhausted=True,
            )
            return RayClosureEvaluation(None, audit)
        except HandModelError as error:
            audit = self._audit(
                budget=budget,
                pad_audits=pad_audits,
                failure_reason=f"HAND_KINEMATICS_REJECTED:{error}",
            )
            return RayClosureEvaluation(None, audit)

        proposal_q_final = np.array(q_start, copy=True)
        for row_index, outcome in enumerate(outcomes):
            possible_set = outcome.possible_first_contact_set
            if possible_set is None:
                raise RayClosureError(
                    "certified PAD outcome lost its possible-first-contact set"
                )
            proposal_phase = (
                possible_set.display_proposal_root
                .certificate.implicit_root.display_approximation
            )
            proposal_q_final += (
                proposal_phase
                * self.closing_directions_physical[row_index]
            )
        try:
            supplied_hand.resolve_joint_positions(proposal_q_final)
        except HandModelError as error:
            audit = self._audit(
                budget=budget,
                pad_audits=pad_audits,
                failure_reason=f"FINAL_JOINT_CONFIGURATION_REJECTED:{error}",
            )
            return RayClosureEvaluation(None, audit)

        contacts: list[PlannedPadContact] = []
        final_pad_audits: list[PadClosureAudit] = []
        for row_index, (prepared, outcome, prior_audit) in enumerate(
            zip(self.prepared_pads, outcomes, pad_audits)
        ):
            possible_set = outcome.possible_first_contact_set
            if possible_set is None:
                raise RayClosureError(
                    "certified PAD outcome lost its possible-first-contact set"
                )
            root = possible_set.display_proposal_root
            certificate = root.certificate
            recertification_counters = _PadCounters()
            recertification_transform_cache = (
                self.interval_kinematics.new_link_transform_cache()
                if execution.cache_enabled
                else None
            )
            recertified = self._classify_witness_face_pair_v9(
                prepared=prepared,
                witness_flat_index=root.witness_flat_index,
                object_face_index=root.object_face_index,
                q_start=q_start,
                direction=self.closing_directions_physical[row_index],
                lower=certificate.phase.lower,
                upper=certificate.phase.upper,
                object_from_hand=object_from_hand,
                counters=recertification_counters,
                transform_cache=recertification_transform_cache,
            )
            if recertification_transform_cache is not None:
                execution.stats.interval_transform_cache_hits += (
                    recertification_transform_cache.hit_count
                )
                execution.stats.interval_transform_cache_misses += (
                    recertification_transform_cache.miss_count
                )
                execution.stats.interval_transform_cache_peak_entries = max(
                    execution.stats.interval_transform_cache_peak_entries,
                    recertification_transform_cache.entry_count,
                )
                execution.stats.interval_point_cache_hits += (
                    recertification_transform_cache.point_hit_count
                )
                execution.stats.interval_point_cache_misses += (
                    recertification_transform_cache.point_miss_count
                )
                execution.stats.interval_point_cache_nonprimary_bypasses += (
                    recertification_transform_cache.point_nonprimary_bypass_count
                )
                execution.stats.interval_point_cache_peak_entries = max(
                    execution.stats.interval_point_cache_peak_entries,
                    recertification_transform_cache.point_entry_count,
                )
                execution.stats.interval_pad_area_cache_hits += (
                    recertification_transform_cache.pad_area_hit_count
                )
                execution.stats.interval_pad_area_cache_misses += (
                    recertification_transform_cache.pad_area_miss_count
                )
                execution.stats.interval_pad_area_cache_nonprimary_bypasses += (
                    recertification_transform_cache.pad_area_nonprimary_bypass_count
                )
                execution.stats.interval_pad_area_cache_peak_entries = max(
                    execution.stats.interval_pad_area_cache_peak_entries,
                    recertification_transform_cache.pad_area_entry_count,
                )
            recertified_root = recertified.root
            identity_changed = (
                recertified.state is not _PadSearchState.CERTIFIED_ROOT
                or recertified_root is None
                or recertified_root.witness_flat_index
                != root.witness_flat_index
                or recertified_root.object_face_index != root.object_face_index
                or recertified_root.semantic_classification
                != root.semantic_classification
                or recertified_root.certificate.phase.lower
                < certificate.phase.lower
                or recertified_root.certificate.phase.upper
                > certificate.phase.upper
                or recertified_root.certificate.object_source_winding_free_side_sign
                != certificate.object_source_winding_free_side_sign
                or recertified_root.certificate.implicit_root.equation_sha256
                != certificate.implicit_root.equation_sha256
                or recertified_root.certificate.implicit_root.feature_identity_sha256
                != certificate.implicit_root.feature_identity_sha256
            )
            if identity_changed:
                final_pad_audits.append(
                    self._pad_audit_v9(
                        prepared, recertification_counters, outcome,
                        "FINAL_Q_INTERVAL_ROOT_RECERTIFICATION_REJECTED",
                    )
                )
                audit = self._audit(
                    budget=budget,
                    pad_audits=final_pad_audits,
                    failure_reason=f"FINAL_Q_RECHECK_FAILED:{prepared.verified.name}",
                )
                return RayClosureEvaluation(None, audit)
            proposal_contact_q = (
                q_start
                + certificate.implicit_root.display_approximation
                * self.closing_directions_physical[row_index]
            )
            links = self.hand_model.forward_kinematics(
                proposal_contact_q, base_transform=object_from_hand
            )
            link_transform = links[prepared.verified.link_name]
            proposal_witness_point = (
                link_transform[:3, :3]
                @ prepared.witness_points_link_m[root.witness_flat_index]
                + link_transform[:3, 3]
            )
            object_triangle = self.canonical_object_face_vertices_m[
                root.object_face_index
            ]
            display_only_contact = _closest_point_on_triangle(
                proposal_witness_point, object_triangle
            )
            source_area = np.cross(
                object_triangle[1] - object_triangle[0],
                object_triangle[2] - object_triangle[0],
            )
            source_normal = source_area / np.linalg.norm(source_area)
            motion_opposing_normal = (
                certificate.object_source_winding_free_side_sign
                * source_normal
            )
            barycentric = prepared.barycentric_coordinates[
                root.witness_flat_index
            ]
            contacts.append(
                PlannedPadContact(
                    pad_name=prepared.verified.name,
                    position_object_m=tuple(
                        float(value) for value in display_only_contact
                    ),
                    path_local_free_side_normal_object=tuple(
                        float(value) for value in motion_opposing_normal
                    ),
                    surface_coordinates=tuple(
                        float(value)
                        for value in (
                            *barycentric,
                            certificate.implicit_root.display_approximation,
                        )
                    ),
                )
            )
            final_pad_audits.append(prior_audit)

        display_only_candidate = GraspCandidate.from_matrix(
            object_from_hand=object_from_hand,
            independent_joint_positions_rad=proposal_q_final,
            planned_pad_contacts=contacts,
            internal_normal_forces_n=(0.0, 0.0, 0.0),
        )
        audit = self._audit(
            budget=budget,
            pad_audits=final_pad_audits,
            failure_reason=REPRESENTATIVE_PROPOSAL_FAILURE_REASON,
            candidate_role=CANDIDATE_REPRESENTATIVE_ROLE,
        )
        possible_sets = tuple(
            outcome.possible_first_contact_set for outcome in outcomes
        )
        if any(row is None for row in possible_sets):
            raise RayClosureError(
                "complete evaluation lost a possible-first-contact set"
            )
        certified_sets = tuple(
            row for row in possible_sets if row is not None
        )
        return RayClosureEvaluation(
            candidate=None,
            audit=audit,
            possible_first_contact_sets=certified_sets,
            display_only_proposal=DisplayOnlyGraspProposal(
                display_only_candidate
            ),
            sequential_closure_policy=CertifiedSequentialClosurePolicy(
                object_from_hand=tuple(
                    float(value) for value in object_from_hand.ravel()
                ),
                initial_independent_joint_positions_rad=tuple(
                    float(value) for value in q_start
                ),
                independent_joint_names=tuple(
                    self.hand_model.independent_joint_names
                ),
                pad_order=tuple(
                    prepared.verified.name for prepared in self.prepared_pads
                ),
                independent_actuation_supports=(
                    self.independent_actuation_supports
                ),
                closing_directions_physical=(
                    self.closing_directions_physical_tuple
                ),
                possible_first_contact_sets=certified_sets,
                object_geometry_sha256=self.object_geometry_sha256,
                model_contract_sha256=self.model_contract_sha256,
            ),
        )

    def candidate_from_unit_parameters(
        self,
        parameters_unit: np.ndarray,
        hand_model: ThreeFingerHandModel,
    ) -> GraspCandidate | None:
        return self.evaluate_unit_parameters(parameters_unit, hand_model).candidate

    def _parameters_from_candidate(
        self,
        candidate: GraspCandidate,
    ) -> tuple[np.ndarray, float]:
        expected_pad_order = tuple(
            prepared.verified.name for prepared in self.prepared_pads
        )
        contacts = tuple(candidate.planned_pad_contacts)
        if tuple(contact.pad_name for contact in contacts) != expected_pad_order:
            raise RayClosureError(
                "candidate contacts do not follow this model's deterministic PAD order"
            )
        if candidate.internal_normal_forces_n != (0.0, 0.0, 0.0):
            raise RayClosureError(
                "ray-closure candidate must retain zero planning force placeholders"
            )
        if candidate.stiffness_diagonal or candidate.damping_diagonal:
            raise RayClosureError(
                "ray-closure candidate cannot inject controller gains"
            )

        closure_parameters: list[float] = []
        for contact in contacts:
            if len(contact.surface_coordinates) != 4:
                raise RayClosureError(
                    "candidate contact lacks finite-PAD witness closure coordinates"
                )
            barycentric = np.asarray(contact.surface_coordinates[:3], dtype=np.float64)
            if not any(
                np.allclose(
                    barycentric,
                    witness,
                    rtol=0.0,
                    atol=_BARYCENTRIC_ERROR,
                )
                for witness in _BARYCENTRIC_WITNESSES
            ):
                raise RayClosureError(
                    "candidate contact is not one of the registered PAD witnesses"
                )
            closure = float(contact.surface_coordinates[3])
            closure_error = _TIME_ERROR * max(1.0, abs(closure))
            if (
                not math.isfinite(closure)
                or closure < -closure_error
                or closure > 1.0 + closure_error
            ):
                raise RayClosureError(
                    "candidate contact closure coordinate is outside the full-open path"
                )
            closure_parameters.append(min(1.0, max(0.0, closure)))

        q_final = np.asarray(
            candidate.independent_joint_positions_rad, dtype=np.float64
        )
        if q_final.shape != self.lower_joint_limits.shape:
            raise RayClosureError("candidate joint vector has the wrong dimension")
        q_start = np.array(q_final, copy=True)
        for closure, direction in zip(
            closure_parameters, self.closing_directions_physical
        ):
            q_start -= closure * direction
        joint_error = _FK_ERROR * max(
            1.0,
            float(np.linalg.norm(q_start, ord=np.inf)),
            float(np.linalg.norm(self.joint_spans, ord=np.inf)),
        )
        for row_index, joint_index in enumerate(self.closure_support_indices):
            if (
                abs(
                    float(q_start[joint_index])
                    - self.closure_open_joint_positions_rad[row_index]
                )
                > joint_error
            ):
                raise RayClosureError(
                    "candidate does not reconstruct the registered full-open endpoint"
                )
        try:
            self.hand_model.resolve_joint_positions(q_start)
        except HandModelError as error:
            raise RayClosureError(
                f"candidate does not reconstruct a valid pregrasp: {error}"
            ) from error

        focus_result = self._closure_focus_hand(q_start)
        if focus_result is None:
            raise RayClosureError("candidate reconstructs no finite closure focus")
        focus_hand, hand_extent = focus_result
        transform = candidate.object_from_hand_matrix()
        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        coordinate_scale = max(
            1.0,
            self.intersector.characteristic_length_m,
            hand_extent,
            float(np.linalg.norm(translation, ord=np.inf)),
            float(
                np.linalg.norm(
                    self.object_model.assembly_axis_origin_m, ord=np.inf
                )
            ),
        )
        comparison_error_m = (
            self.intersector.distance_error_bound_m
            + self.distance_bvh.aabb_error_bound_m
            + _FK_ERROR * coordinate_scale
        )

        relative_rotation = self.task_basis_object.T @ rotation
        yaw = math.atan2(
            float(relative_rotation[1, 0]),
            float(relative_rotation[0, 0]),
        )
        if yaw < 0.0:
            yaw += 2.0 * math.pi
        canonical_rotation = np.asarray(
            (
                (math.cos(yaw), -math.sin(yaw), 0.0),
                (math.sin(yaw), math.cos(yaw), 0.0),
                (0.0, 0.0, 1.0),
            )
        )
        rotation_error = _FK_ERROR * max(
            1.0, float(np.linalg.norm(relative_rotation, ord=np.inf))
        )
        if not np.allclose(
            relative_rotation,
            canonical_rotation,
            rtol=0.0,
            atol=rotation_error,
        ):
            raise RayClosureError(
                "candidate hand rotation is outside the pre-registered axis-yaw family"
            )

        target = rotation @ focus_hand + translation
        lower, upper = self._placement_coordinate_bounds(q_start, rotation)
        target_coordinates = self.task_basis_object.T @ (
            target - self.object_model.assembly_axis_origin_m
        )
        parameters = np.empty(self.parameter_dimension, dtype=np.float64)
        parameters[0] = yaw / (2.0 * math.pi)
        parameters[1] = _recover_closed_unit_coordinate(
            float(target_coordinates[2]),
            float(lower[2]),
            float(upper[2]),
            absolute_error=comparison_error_m,
            label="candidate axial target",
        )
        parameters[2] = _recover_closed_unit_coordinate(
            float(target_coordinates[0]),
            float(lower[0]),
            float(upper[0]),
            absolute_error=comparison_error_m,
            label="candidate first lateral target",
        )
        parameters[3] = _recover_closed_unit_coordinate(
            float(target_coordinates[1]),
            float(lower[1]),
            float(upper[1]),
            absolute_error=comparison_error_m,
            label="candidate second lateral target",
        )
        for parameter_index, joint_index in enumerate(self.preshape_joint_indices):
            parameters[4 + parameter_index] = _recover_closed_unit_coordinate(
                float(q_start[joint_index]),
                float(self.lower_joint_limits[joint_index]),
                float(self.upper_joint_limits[joint_index]),
                absolute_error=joint_error,
                label=f"candidate preshape joint {joint_index}",
            )
        return parameters, comparison_error_m

    @staticmethod
    def _assert_recomputed_candidate(
        supplied: GraspCandidate,
        recomputed: GraspCandidate,
        *,
        comparison_error_m: float,
    ) -> None:
        supplied_transform = supplied.object_from_hand_matrix()
        recomputed_transform = recomputed.object_from_hand_matrix()
        transform_error = _FK_ERROR * max(
            1.0,
            float(np.linalg.norm(supplied_transform, ord=np.inf)),
            float(np.linalg.norm(recomputed_transform, ord=np.inf)),
        )
        if not np.allclose(
            supplied_transform,
            recomputed_transform,
            rtol=0.0,
            atol=transform_error,
        ):
            raise RayClosureError("candidate transform fails deterministic recomputation")
        supplied_joints = np.asarray(supplied.independent_joint_positions_rad)
        recomputed_joints = np.asarray(recomputed.independent_joint_positions_rad)
        joint_error = _FK_ERROR * max(
            1.0,
            float(np.linalg.norm(supplied_joints, ord=np.inf)),
            float(np.linalg.norm(recomputed_joints, ord=np.inf)),
        )
        if not np.allclose(
            supplied_joints, recomputed_joints, rtol=0.0, atol=joint_error
        ):
            raise RayClosureError("candidate final joints fail deterministic recomputation")
        if (
            supplied.internal_normal_forces_n
            != recomputed.internal_normal_forces_n
            or supplied.stiffness_diagonal != recomputed.stiffness_diagonal
            or supplied.damping_diagonal != recomputed.damping_diagonal
        ):
            raise RayClosureError("candidate planning placeholders were modified")
        for supplied_contact, recomputed_contact in zip(
            supplied.planned_pad_contacts, recomputed.planned_pad_contacts
        ):
            if supplied_contact.pad_name != recomputed_contact.pad_name:
                raise RayClosureError("candidate PAD identity fails recomputation")
            if not np.allclose(
                supplied_contact.position_object_m,
                recomputed_contact.position_object_m,
                rtol=0.0,
                atol=comparison_error_m,
            ):
                raise RayClosureError("candidate contact position fails recomputation")
            if not np.allclose(
                supplied_contact.path_local_free_side_normal_object,
                recomputed_contact.path_local_free_side_normal_object,
                rtol=0.0,
                atol=_DOT_ERROR,
            ):
                raise RayClosureError("candidate contact normal fails recomputation")
            coordinate_scale = max(
                1.0,
                *(abs(value) for value in supplied_contact.surface_coordinates),
                *(abs(value) for value in recomputed_contact.surface_coordinates),
            )
            if not np.allclose(
                supplied_contact.surface_coordinates,
                recomputed_contact.surface_coordinates,
                rtol=0.0,
                atol=_TIME_ERROR * coordinate_scale,
            ):
                raise RayClosureError("candidate PAD witness coordinates fail recomputation")

    def trajectory_clearance_m(
        self,
        candidate: GraspCandidate,
        hand_model: ThreeFingerHandModel,
    ) -> float:
        """Return only the certified complete finite-witness-path lower bound.

        This is a stateless verification hook for TaskWrenchEvaluator.
        Candidate parameters are inverted from immutable candidate fields and
        the entire predictor is rerun.  It is not a full-hand clearance query.
        Since each valid path ends at contact, its complete-path lower bound is
        exactly zero.
        """

        if not isinstance(candidate, GraspCandidate):
            raise RayClosureError("trajectory clearance requires a GraspCandidate")
        self._validate_hand(hand_model)
        parameters, comparison_error_m = self._parameters_from_candidate(candidate)
        recomputed = self.evaluate_unit_parameters(parameters, hand_model)
        if recomputed.candidate is None:
            raise RayClosureError(
                "candidate trajectory cannot be recertified: "
                f"{recomputed.audit.failure_reason}"
            )
        self._assert_recomputed_candidate(
            candidate,
            recomputed.candidate,
            comparison_error_m=comparison_error_m,
        )
        if recomputed.audit.trajectory_clearance_role != TRAJECTORY_CLEARANCE_ROLE:
            raise RayClosureError("trajectory clearance audit role changed unexpectedly")
        return float(recomputed.audit.trajectory_clearance_m)

    @property
    def contract(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "method_id": METHOD_ID,
                "model_binding_complete": self.model_binding_complete,
                "model_binding_status": self.model_binding_status,
                "model_contract_digest_method_id": (
                    MODEL_CONTRACT_DIGEST_METHOD_ID
                ),
                "object_geometry_sha256": self.object_geometry_sha256,
                "model_contract_sha256": self.model_contract_sha256,
                "pad_geometry_sha256": self.pad_geometry_sha256,
                "pad_runtime_geometry_sha256": (
                    self.pad_runtime_geometry_sha256
                ),
                "pad_link_names": self.pad_link_names,
                "closing_directions_physical": (
                    self.closing_directions_physical_tuple
                ),
                "model_contract_canonical_json": (
                    self.model_contract_canonical_json
                ),
                "parameter_layout": self.parameter_layout,
                "closure_focus_method": CLOSURE_FOCUS_METHOD,
                "ray_evaluation_policy": RAY_EVALUATION_POLICY,
                "feature_root_policy": FEATURE_ROOT_POLICY,
                "possible_first_contact_set_method_id": (
                    POSSIBLE_FIRST_CONTACT_SET_METHOD_ID
                ),
                "possible_earliest_ordering_policy": (
                    POSSIBLE_EARLIEST_ORDERING_POLICY
                ),
                "representative_proposal_role": (
                    CANDIDATE_REPRESENTATIVE_ROLE
                ),
                "display_approximation_role": DISPLAY_APPROXIMATION_ROLE,
                "object_contact_normal_policy": (
                    self.object_contact_normal_policy
                ),
                "pad_surface_normal_policy": (
                    self.pad_surface_normal_policy
                ),
                "closure_parameter_domain_id": CLOSURE_PARAMETER_DOMAIN_ID,
                "closure_suffix_dominance_argument": (
                    CLOSURE_SUFFIX_DOMINANCE_ARGUMENT
                ),
                "preshape_joint_names": self.preshape_joint_names,
                "closure_open_joint_positions_rad": (
                    self.closure_open_joint_positions_rad
                ),
                "pad_order": tuple(pad.name for pad in self.verified_pads),
                "maximum_subdivision_intervals": self.maximum_subdivision_intervals,
                "interval_arithmetic_method_id": (
                    INTERVAL_KINEMATICS_METHOD_ID
                ),
                "interval_decimal_precision": (
                    self.interval_arithmetic_options.decimal_precision
                ),
                "maximum_root_bisection_iterations": (
                    self.interval_arithmetic_options.maximum_root_bisection_iterations
                ),
                "full_verified_pad_mesh_used": True,
                "pad_face_subset_input_allowed": False,
                "internal_force_role": INTERNAL_FORCE_ROLE,
                "trajectory_clearance_role": TRAJECTORY_CLEARANCE_ROLE,
                "claim_limitations": CLAIM_LIMITATIONS,
            }
        )


__all__ = [
    "CANDIDATE_REPRESENTATIVE_ROLE",
    "CLAIM_LIMITATIONS",
    "CertifiedContactFeatureRoot",
    "CertifiedSequentialClosurePolicy",
    "CLOSURE_FOCUS_METHOD",
    "CLOSURE_PARAMETER_DOMAIN_ID",
    "CLOSURE_SUFFIX_DOMINANCE_ARGUMENT",
    "DISTANCE_BVH_RULE",
    "DisplayOnlyGraspProposal",
    "FEATURE_ROOT_POLICY",
    "INTERNAL_FORCE_ROLE",
    "INTERVAL_RULE",
    "METHOD_ID",
    "MODEL_BINDING_COMPLETE_STATUS",
    "MODEL_BINDING_UNBOUND_STATUS",
    "MODEL_CONTRACT_DIGEST_METHOD_ID",
    "PARAMETER_LAYOUT_PREFIX",
    "POSSIBLE_EARLIEST_ORDERING_POLICY",
    "POSSIBLE_FIRST_CONTACT_SET_METHOD_ID",
    "PossibleFirstContactSet",
    "RAY_EVALUATION_POLICY",
    "REPRESENTATIVE_PROPOSAL_FAILURE_REASON",
    "SEQUENTIAL_CLOSURE_EXECUTION_SEMANTICS",
    "SEQUENTIAL_CLOSURE_POLICY_METHOD_ID",
    "PadClosureAudit",
    "PreRegisteredTaskFrame",
    "RayClosureAudit",
    "RayClosureError",
    "RayClosureEvaluation",
    "RayClosureSurfaceModel",
    "TRAJECTORY_CLEARANCE_ROLE",
    "WHOLE_PATH_SPHERE_SCREEN_RULE",
    "WITNESS_RULE",
    "WholePathPadSphereScreen",
]
