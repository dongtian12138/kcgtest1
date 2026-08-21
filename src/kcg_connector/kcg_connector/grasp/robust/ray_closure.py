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

from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
import hashlib
import heapq
import json
import math
import re
from types import MappingProxyType
from typing import ClassVar, Mapping, Sequence

import numpy as np
from scipy.optimize import brentq

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
    DISPLAY_APPROXIMATION_ROLE,
    DirectedIntervalKinematics,
    IntervalArithmeticOptions,
    IntervalKinematicsError,
    IntervalRootClassification,
    IntervalRootState,
    IntervalTransverseRootCertificate,
    METHOD_ID as INTERVAL_KINEMATICS_METHOD_ID,
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


def _exact_dyadic_plane_key(
    triangle_m: Sequence[Sequence[float]],
) -> tuple[int, int, int, int]:
    """Return an orientation-free exact plane key for binary64 vertices."""

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

    def nearest_many(
        self, points_m: Sequence[Sequence[float]]
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
        points = points_world - self.centre_m
        count = len(points)
        best_distances = np.full(count, math.inf, dtype=np.float64)
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
                for face_index_value in node.face_indices:
                    face_index = int(face_index_value)
                    closest = _closest_points_on_triangle(
                        points[active], self.triangles[face_index]
                    )
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
    interval_point_motion_evaluations: int = 0
    interval_pair_evaluations: int = 0
    certified_contact_roots: int = 0
    unresolved_witness_face_pairs: int = 0
    cofirst_root_count: int = 0
    possible_earliest_root_count: int = 0
    competing_root_order_blocks: int = 0


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
    triangles = points[faces]
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
    for array in (
        witness_points,
        witness_normals,
        triangle_indices,
        witness_indices,
        barycentric,
        surface_centroid,
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
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return translations where every swept PAD AABB can overlap CAD.

        For object interval ``[o-, o+]`` and a certified outer interval
        ``[p-, p+]`` of one PAD over its complete registered closure path,
        overlap is possible only when the focus coordinate is in
        ``[o- - p+, o+ - p-]``.  Intersecting those intervals over all three
        PADs therefore gives a necessary geometry-derived chart domain.  It
        is not a clearance or collision certificate.
        """

        focus_result = self._closure_focus_hand(q_start)
        if focus_result is None:
            raise RayClosureError(
                "placement domain has no finite full-closed PAD focus"
            )
        focus_hand, _hand_extent = focus_result
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

    def _decode(
        self, parameters_unit: Sequence[float]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        lower, upper = self._placement_coordinate_bounds(joints, rotation)
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
        endpoint = q_start + maximum_parameter * direction
        resolved_start = self.hand_model.resolve_joint_positions(q_start)
        resolved_end = self.hand_model.resolve_joint_positions(endpoint)
        resolved_velocity = self.hand_model.resolve_joint_velocities(
            direction, enforce_limits=False
        )
        ancestor_names = self._ancestor_joint_names(prepared.verified.link_name)
        local_radii = np.linalg.norm(points, axis=1)
        velocity_bounds = np.zeros(len(local_radii), dtype=np.float64)
        for ancestor_index, name in enumerate(ancestor_names):
            joint = self.hand_model.joints[name]
            rate = abs(float(resolved_velocity[name]))
            if joint.joint_type in ("revolute", "continuous"):
                downstream_reach = np.array(local_radii, copy=True)
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
                velocity_bounds += rate * downstream_reach
            elif joint.joint_type == "prismatic":
                velocity_bounds += rate
        velocity_bounds = np.nextafter(
            velocity_bounds * (1.0 + _FK_ERROR), math.inf
        )
        if np.any(velocity_bounds <= 0.0) or not np.all(
            np.isfinite(velocity_bounds)
        ):
            raise RayClosureError(
                f"closing path for {prepared.verified.name} has no finite PAD motion bound"
            )
        velocity_bounds.setflags(write=False)
        return velocity_bounds

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
    ) -> _IntervalGeometry:
        """Return the H2 possible set with certified hierarchy pruning.

        Nodes are visited in increasing lower-bound order.  A node may be
        skipped only after its strict lower bound is greater than zero when a
        possible witness already exists, or greater than the best exact margin
        when the interval is free.  Thus every possible witness and the exact
        free-interval minimum are identical to an all-witness nearest query.
        """

        thresholds = np.asarray(
            enclosure_radii_m + spatial_error_bound_m, dtype=np.float64
        )
        witness_count = len(states)
        possible = np.zeros(witness_count, dtype=bool)
        exact_margins = np.full(witness_count, math.inf, dtype=np.float64)
        nearest_face_indices = np.full(witness_count, -1, dtype=np.int64)
        exact = np.zeros(witness_count, dtype=bool)
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
    ) -> tuple[tuple[int, int], ...]:
        """Enumerate every witness/face pair allowed by interval AABBs."""

        pairs: list[tuple[int, int]] = []
        for witness_index_value in possible_witness_indices:
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
                position_lower,
                position_upper,
            )
            counters.swept_face_candidates += len(face_indices)
            pairs.extend(
                (witness_index, int(face_index))
                for face_index in face_indices
            )
        return tuple(sorted(set(pairs)))

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
                if self.object_model.contact_face_mask[object_face_index]
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

        def search(lower: float, upper: float) -> _PadSearchOutcome:
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
            pairs = self._complete_swept_face_pairs_v9(
                prepared=prepared,
                possible_witness_indices=possible_indices,
                q_start=q_start,
                direction=direction,
                lower=lower,
                upper=upper,
                object_from_hand=object_from_hand,
                counters=counters,
            )
            rows = tuple(
                self._classify_witness_face_pair_v9(
                    prepared=prepared,
                    witness_flat_index=witness_index,
                    object_face_index=face_index,
                    q_start=q_start,
                    direction=direction,
                    lower=lower,
                    upper=upper,
                    object_from_hand=object_from_hand,
                    counters=counters,
                )
                for witness_index, face_index in pairs
            )
            combined = self._combine_pair_classifications_v9(
                rows,
                lower=lower,
                upper=upper,
                counters=counters,
            )
            if combined.state is not _PadSearchState.UNRESOLVED:
                return combined
            if np.nextafter(lower, upper) >= upper:
                return combined
            first = search(lower, midpoint)
            if first.state is not _PadSearchState.CERTIFIED_FREE:
                return first
            return search(midpoint, upper)

        return search(0.0, maximum_parameter)

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
        if not self.object_model.contact_face_mask[object_face_index]:
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
                    if not self.object_model.contact_face_mask[ray.face_index]:
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
    ) -> RayClosureEvaluation:
        if not isinstance(execution, _GeometryExecutionContext):
            raise RayClosureError("invalid ray-closure geometry execution context")
        if execution.witness_state_cache or execution.nearest_batch_cache:
            raise RayClosureError(
                "geometry execution context caches must be empty at evaluation start"
            )
        supplied_hand = self.hand_model if hand_model is None else hand_model
        self._validate_hand(supplied_hand)
        budget = _Budget(self.maximum_subdivision_intervals)
        try:
            q_start, target, rotation = self._decode(parameters_unit)
        except RayClosureError as error:
            audit = self._audit(
                budget=budget,
                pad_audits=(),
                failure_reason=f"PARAMETER_DOMAIN_REJECTED:{error}",
            )
            return RayClosureEvaluation(None, audit)
        transform_result = self._object_from_hand(q_start, target, rotation)
        if transform_result is None:
            audit = self._audit(
                budget=budget,
                pad_audits=(),
                failure_reason="CLOSING_FOCUS_UNDEFINED_FROM_PAD_KINEMATICS",
            )
            return RayClosureEvaluation(None, audit)
        object_from_hand, hand_extent = transform_result
        spatial_error = (
            self.intersector.distance_error_bound_m
            + self.distance_bvh.aabb_error_bound_m
            + _FK_ERROR
            * (self.intersector.characteristic_length_m + hand_extent)
        )

        outcomes: list[_PadSearchOutcome] = []
        pad_audits: list[PadClosureAudit] = []
        try:
            for row_index, prepared in enumerate(self.prepared_pads):
                direction = self.closing_directions_physical[row_index]
                maximum_parameter = self._maximum_path_parameter(q_start, direction)
                counters = _PadCounters()
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
    "WITNESS_RULE",
]
