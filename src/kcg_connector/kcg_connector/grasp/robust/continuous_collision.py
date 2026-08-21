"""Conservative interval separation certificates for triangle surfaces.

The module exposes two distinct proof families: moving-link surface versus a
static-object surface, and one moving-link surface versus another moving-link
surface on the same scalar joint path.  Both require strict Cartesian-axis
separation for every triangle pair.  Any potential contact, tangency,
coplanarity, arithmetic failure, adjacent-float phase, or exhausted
computation budget remains unresolved.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import math
from typing import Sequence

import numpy as np

from kcg_connector.grasp.robust.interval_kinematics import (
    DirectedIntervalKinematics,
    IntervalBounds,
    IntervalKinematicsError,
    METHOD_ID as INTERVAL_KINEMATICS_METHOD_ID,
)


METHOD_ID = "CARTS_MP_INTERVAL_MOVING_STATIC_SURFACE_STRICT_AABB_BVH_V1"
MOVING_PAIR_METHOD_ID = (
    "CARTS_MP_INTERVAL_MOVING_SURFACE_PAIR_RELATIVE_AXIS_V1"
)
CLAIM_LIMITATIONS = (
    "MOVING_LINK_SURFACE_VS_STATIC_OBJECT_SURFACE_ONLY",
    "NOT_SOLID_CONTAINMENT_OR_INTERIOR_EXCLUSION",
    "NOT_HAND_SELF_COLLISION",
    "NOT_ENVIRONMENT_COLLISION",
    "NOT_MULTI_LINK_OR_FULL_HAND_PATH_CERTIFICATE",
    "POTENTIAL_CONTACT_TANGENCY_AND_COPLANARITY_ARE_UNRESOLVED",
    "TRIANGLE_SWEPT_AABB_SEPARATION_SUFFICIENT_NOT_NECESSARY",
)
MOVING_PAIR_CLAIM_LIMITATIONS = (
    "TWO_MOVING_LINK_SURFACES_ONLY",
    "NO_SEMANTIC_COLLISION_PAIR_EXEMPTIONS_APPLIED",
    "NOT_SOLID_CONTAINMENT_OR_INTERIOR_EXCLUSION",
    "NOT_ENVIRONMENT_COLLISION",
    "NOT_FULL_HAND_OR_MULTI_PAIR_PATH_CERTIFICATE",
    "INDEPENDENT_POINT_INTERVALS_DO_NOT_EXPLOIT_FK_CORRELATION",
    "POTENTIAL_CONTACT_TANGENCY_AND_COPLANARITY_ARE_UNRESOLVED",
    "RELATIVE_AXIS_SEPARATION_SUFFICIENT_NOT_NECESSARY",
)


class ContinuousCollisionError(ValueError):
    """Raised when the requested mathematical contract is malformed."""


class ContinuousCollisionState(str, Enum):
    CERTIFIED_FREE = "CERTIFIED_FREE"
    UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True)
class ContinuousCollisionAudit:
    method_id: str
    interval_kinematics_method_id: str
    moving_surface_geometry_sha256: str
    static_surface_geometry_sha256: str
    motion_contract_sha256: str
    link_name: str
    moving_triangle_count: int
    static_triangle_count: int
    pair_count_per_interval: int
    maximum_subdivision_intervals: int
    processed_interval_count: int
    certified_free_leaf_interval_count: int
    subdivided_interval_count: int
    point_motion_evaluation_count: int
    bvh_node_visit_count: int
    bvh_leaf_visit_count: int
    leaf_pair_evaluation_count: int
    pair_universe_count: int
    pair_coverage_count: int
    strictly_separated_pair_count: int
    potential_overlap_pair_observation_count: int
    terminal_unresolved_pair_count: int
    all_processed_pairs_accounted_for: bool
    entire_phase_covered: bool
    unresolved_reason: str
    claim_limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        digests = (
            self.moving_surface_geometry_sha256,
            self.static_surface_geometry_sha256,
            self.motion_contract_sha256,
        )
        if self.method_id != METHOD_ID:
            raise ContinuousCollisionError("collision audit method mismatch")
        if self.interval_kinematics_method_id != INTERVAL_KINEMATICS_METHOD_ID:
            raise ContinuousCollisionError(
                "collision audit interval backend mismatch"
            )
        if any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in digests
        ):
            raise ContinuousCollisionError(
                "collision audit geometry digests are invalid"
            )
        if not str(self.link_name):
            raise ContinuousCollisionError(
                "collision audit link_name cannot be empty"
            )
        integer_fields = (
            self.moving_triangle_count,
            self.static_triangle_count,
            self.pair_count_per_interval,
            self.maximum_subdivision_intervals,
            self.processed_interval_count,
            self.certified_free_leaf_interval_count,
            self.subdivided_interval_count,
            self.point_motion_evaluation_count,
            self.bvh_node_visit_count,
            self.bvh_leaf_visit_count,
            self.leaf_pair_evaluation_count,
            self.pair_universe_count,
            self.pair_coverage_count,
            self.strictly_separated_pair_count,
            self.potential_overlap_pair_observation_count,
            self.terminal_unresolved_pair_count,
        )
        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            for value in integer_fields
        ):
            raise ContinuousCollisionError(
                "collision audit counters must be non-negative integers"
            )
        if (
            self.moving_triangle_count == 0
            or self.static_triangle_count == 0
            or self.pair_count_per_interval
            != self.moving_triangle_count * self.static_triangle_count
            or self.maximum_subdivision_intervals == 0
            or self.processed_interval_count
            > self.maximum_subdivision_intervals
            or self.pair_coverage_count > self.pair_universe_count
            or self.pair_universe_count
            != self.processed_interval_count * self.pair_count_per_interval
            or self.strictly_separated_pair_count
            + self.potential_overlap_pair_observation_count
            != self.pair_coverage_count
            or self.leaf_pair_evaluation_count > self.pair_coverage_count
            or self.bvh_leaf_visit_count > self.bvh_node_visit_count
        ):
            raise ContinuousCollisionError(
                "collision audit coverage arithmetic is inconsistent"
            )
        if self.all_processed_pairs_accounted_for != (
            self.pair_coverage_count == self.pair_universe_count
        ):
            raise ContinuousCollisionError(
                "collision audit pair-accounting flag is inconsistent"
            )
        if (
            self.all_processed_pairs_accounted_for
            and self.point_motion_evaluation_count
            != self.processed_interval_count
            * self.moving_triangle_count
            * 3
        ):
            raise ContinuousCollisionError(
                "collision audit point-motion coverage is incomplete"
            )
        if self.claim_limitations != CLAIM_LIMITATIONS:
            raise ContinuousCollisionError(
                "collision audit limitations were altered"
            )
        if not str(self.unresolved_reason):
            raise ContinuousCollisionError(
                "collision audit unresolved_reason cannot be empty"
            )


@dataclass(frozen=True)
class ContinuousCollisionCertificate:
    state: ContinuousCollisionState
    searched_phase: IntervalBounds
    certified_free_leaf_intervals: tuple[IntervalBounds, ...]
    unresolved_interval: IntervalBounds | None
    audit: ContinuousCollisionAudit

    def __post_init__(self) -> None:
        if not isinstance(self.state, ContinuousCollisionState):
            raise ContinuousCollisionError(
                "collision certificate state is invalid"
            )
        if not isinstance(self.searched_phase, IntervalBounds):
            raise ContinuousCollisionError(
                "collision certificate phase is invalid"
            )
        leaves = tuple(self.certified_free_leaf_intervals)
        if not all(isinstance(value, IntervalBounds) for value in leaves):
            raise ContinuousCollisionError(
                "collision certificate free leaves must be intervals"
            )
        object.__setattr__(self, "certified_free_leaf_intervals", leaves)
        if self.audit.certified_free_leaf_interval_count != len(leaves):
            raise ContinuousCollisionError(
                "collision certificate free-leaf count is inconsistent"
            )
        if self.state == ContinuousCollisionState.CERTIFIED_FREE:
            if (
                self.unresolved_interval is not None
                or not leaves
                or not self.audit.entire_phase_covered
                or not self.audit.all_processed_pairs_accounted_for
                or self.audit.unresolved_reason != "NONE"
            ):
                raise ContinuousCollisionError(
                    "free collision certificate is internally inconsistent"
                )
            if (
                leaves[0].lower != self.searched_phase.lower
                or leaves[-1].upper != self.searched_phase.upper
                or any(
                    first.upper != second.lower
                    for first, second in zip(leaves, leaves[1:])
                )
            ):
                raise ContinuousCollisionError(
                    "free collision leaves do not cover the searched phase"
                )
        elif (
            not isinstance(self.unresolved_interval, IntervalBounds)
            or self.audit.entire_phase_covered
            or self.audit.unresolved_reason == "NONE"
        ):
            raise ContinuousCollisionError(
                "unresolved collision certificate is internally inconsistent"
            )


@dataclass(frozen=True)
class MovingSurfacePairCollisionAudit:
    method_id: str
    interval_kinematics_method_id: str
    first_link_name: str
    second_link_name: str
    first_surface_geometry_sha256: str
    second_surface_geometry_sha256: str
    pair_contract_sha256: str
    first_triangle_count: int
    second_triangle_count: int
    pair_count_per_interval: int
    maximum_subdivision_intervals: int
    processed_interval_count: int
    certified_free_leaf_interval_count: int
    subdivided_interval_count: int
    point_motion_evaluation_count: int
    relative_coordinate_interval_evaluation_count: int
    pair_universe_count: int
    pair_coverage_count: int
    strictly_separated_pair_count: int
    potential_overlap_pair_observation_count: int
    terminal_unresolved_pair_count: int
    all_processed_pairs_accounted_for: bool
    entire_phase_covered: bool
    unresolved_reason: str
    claim_limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.method_id != MOVING_PAIR_METHOD_ID:
            raise ContinuousCollisionError(
                "moving-pair collision audit method mismatch"
            )
        if self.interval_kinematics_method_id != (
            INTERVAL_KINEMATICS_METHOD_ID
        ):
            raise ContinuousCollisionError(
                "moving-pair interval backend mismatch"
            )
        if (
            not str(self.first_link_name)
            or not str(self.second_link_name)
            or self.first_link_name == self.second_link_name
        ):
            raise ContinuousCollisionError(
                "moving-pair audit needs two distinct named links"
            )
        for value in (
            self.first_surface_geometry_sha256,
            self.second_surface_geometry_sha256,
            self.pair_contract_sha256,
        ):
            if len(value) != 64 or any(
                character not in "0123456789abcdef"
                for character in value
            ):
                raise ContinuousCollisionError(
                    "moving-pair audit digest is invalid"
                )
        integer_fields = (
            self.first_triangle_count,
            self.second_triangle_count,
            self.pair_count_per_interval,
            self.maximum_subdivision_intervals,
            self.processed_interval_count,
            self.certified_free_leaf_interval_count,
            self.subdivided_interval_count,
            self.point_motion_evaluation_count,
            self.relative_coordinate_interval_evaluation_count,
            self.pair_universe_count,
            self.pair_coverage_count,
            self.strictly_separated_pair_count,
            self.potential_overlap_pair_observation_count,
            self.terminal_unresolved_pair_count,
        )
        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            for value in integer_fields
        ):
            raise ContinuousCollisionError(
                "moving-pair audit counters must be non-negative integers"
            )
        if (
            self.first_triangle_count == 0
            or self.second_triangle_count == 0
            or self.maximum_subdivision_intervals == 0
            or self.pair_count_per_interval
            != self.first_triangle_count * self.second_triangle_count
            or self.processed_interval_count
            > self.maximum_subdivision_intervals
            or self.pair_universe_count
            != self.processed_interval_count * self.pair_count_per_interval
            or self.pair_coverage_count > self.pair_universe_count
            or self.strictly_separated_pair_count
            + self.potential_overlap_pair_observation_count
            != self.pair_coverage_count
        ):
            raise ContinuousCollisionError(
                "moving-pair audit coverage arithmetic is inconsistent"
            )
        if self.all_processed_pairs_accounted_for != (
            self.pair_coverage_count == self.pair_universe_count
        ):
            raise ContinuousCollisionError(
                "moving-pair accounting flag is inconsistent"
            )
        if self.all_processed_pairs_accounted_for:
            expected_point_evaluations = (
                self.processed_interval_count
                * 3
                * (
                    self.first_triangle_count
                    + self.second_triangle_count
                )
            )
            expected_relative_evaluations = (
                self.processed_interval_count
                * self.pair_count_per_interval
                * 27
            )
            if (
                self.point_motion_evaluation_count
                != expected_point_evaluations
                or self.relative_coordinate_interval_evaluation_count
                != expected_relative_evaluations
            ):
                raise ContinuousCollisionError(
                    "moving-pair directed-interval coverage is incomplete"
                )
        if self.claim_limitations != MOVING_PAIR_CLAIM_LIMITATIONS:
            raise ContinuousCollisionError(
                "moving-pair audit limitations were altered"
            )
        if not str(self.unresolved_reason):
            raise ContinuousCollisionError(
                "moving-pair unresolved_reason cannot be empty"
            )


@dataclass(frozen=True)
class MovingSurfacePairCollisionCertificate:
    state: ContinuousCollisionState
    searched_phase: IntervalBounds
    certified_free_leaf_intervals: tuple[IntervalBounds, ...]
    unresolved_interval: IntervalBounds | None
    audit: MovingSurfacePairCollisionAudit

    def __post_init__(self) -> None:
        if not isinstance(self.state, ContinuousCollisionState):
            raise ContinuousCollisionError(
                "moving-pair certificate state is invalid"
            )
        if not isinstance(self.searched_phase, IntervalBounds):
            raise ContinuousCollisionError(
                "moving-pair searched phase is invalid"
            )
        leaves = tuple(self.certified_free_leaf_intervals)
        if not all(isinstance(value, IntervalBounds) for value in leaves):
            raise ContinuousCollisionError(
                "moving-pair free leaves must be intervals"
            )
        object.__setattr__(self, "certified_free_leaf_intervals", leaves)
        if self.audit.certified_free_leaf_interval_count != len(leaves):
            raise ContinuousCollisionError(
                "moving-pair free-leaf count is inconsistent"
            )
        if self.state == ContinuousCollisionState.CERTIFIED_FREE:
            if (
                self.unresolved_interval is not None
                or not leaves
                or not self.audit.entire_phase_covered
                or not self.audit.all_processed_pairs_accounted_for
                or self.audit.unresolved_reason != "NONE"
            ):
                raise ContinuousCollisionError(
                    "moving-pair free certificate is inconsistent"
                )
            if (
                leaves[0].lower != self.searched_phase.lower
                or leaves[-1].upper != self.searched_phase.upper
                or any(
                    first.upper != second.lower
                    for first, second in zip(leaves, leaves[1:])
                )
            ):
                raise ContinuousCollisionError(
                    "moving-pair free leaves do not cover the phase"
                )
        elif (
            not isinstance(self.unresolved_interval, IntervalBounds)
            or self.audit.entire_phase_covered
            or self.audit.unresolved_reason == "NONE"
        ):
            raise ContinuousCollisionError(
                "moving-pair unresolved certificate is inconsistent"
            )


@dataclass
class _Counters:
    processed_intervals: int = 0
    free_leaf_intervals: int = 0
    subdivisions: int = 0
    point_motion_evaluations: int = 0
    bvh_node_visits: int = 0
    bvh_leaf_visits: int = 0
    leaf_pair_evaluations: int = 0
    pair_universe: int = 0
    pair_coverage: int = 0
    strictly_separated_pairs: int = 0
    potential_overlap_pairs: int = 0


@dataclass
class _MovingPairCounters:
    processed_intervals: int = 0
    free_leaf_intervals: int = 0
    subdivisions: int = 0
    point_motion_evaluations: int = 0
    relative_coordinate_interval_evaluations: int = 0
    pair_universe: int = 0
    pair_coverage: int = 0
    strictly_separated_pairs: int = 0
    potential_overlap_pairs: int = 0


@dataclass(frozen=True)
class _BVHNode:
    lower_m: np.ndarray
    upper_m: np.ndarray
    face_indices: tuple[int, ...]
    subtree_face_count: int
    left: "_BVHNode | None" = None
    right: "_BVHNode | None" = None

    def __post_init__(self) -> None:
        lower = np.array(self.lower_m, dtype=np.float64, copy=True)
        upper = np.array(self.upper_m, dtype=np.float64, copy=True)
        if (
            lower.shape != (3,)
            or upper.shape != (3,)
            or not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper))
            or np.any(lower > upper)
        ):
            raise ContinuousCollisionError("invalid static BVH node bounds")
        lower.setflags(write=False)
        upper.setflags(write=False)
        object.__setattr__(self, "lower_m", lower)
        object.__setattr__(self, "upper_m", upper)
        object.__setattr__(self, "face_indices", tuple(self.face_indices))

    @property
    def leaf(self) -> bool:
        return self.left is None and self.right is None


def _normalize_signed_zero(array: np.ndarray) -> np.ndarray:
    normalized = np.array(array, dtype=np.float64, copy=True)
    normalized[normalized == 0.0] = 0.0
    return normalized


def _update_text(digest: "hashlib._Hash", value: str) -> None:
    payload = str(value).encode("utf-8")
    digest.update(np.asarray((len(payload),), dtype="<u8").tobytes())
    digest.update(payload)


def _motion_contract_sha256(
    *,
    backend: DirectedIntervalKinematics,
    link_name: str,
    q_start: np.ndarray,
    direction: np.ndarray,
    phase: IntervalBounds,
    base_transform: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"CARTS_INTERVAL_LINK_MOTION_CONTRACT_V1\0")
    _update_text(digest, backend.hand_model.base_link)
    _update_text(digest, link_name)
    _update_text(digest, INTERVAL_KINEMATICS_METHOD_ID)
    digest.update(
        np.asarray(
            (backend.options.decimal_precision,),
            dtype="<u8",
        ).tobytes()
    )
    for joint_name in backend.hand_model.joint_order:
        joint = backend.hand_model.joints[joint_name]
        for value in (
            joint.name,
            joint.joint_type,
            joint.parent_link,
            joint.child_link,
        ):
            _update_text(digest, value)
        digest.update(
            np.asarray(
                (
                    *joint.origin_xyz_m,
                    *joint.origin_rpy_rad,
                    *joint.axis,
                ),
                dtype="<f8",
            ).tobytes()
        )
        if joint.limit is None:
            digest.update(b"NO_LIMIT\0")
        else:
            digest.update(b"LIMIT\0")
            digest.update(
                np.asarray(
                    (joint.limit.lower, joint.limit.upper),
                    dtype="<f8",
                ).tobytes()
            )
        if joint.mimic is None:
            digest.update(b"NO_MIMIC\0")
        else:
            digest.update(b"MIMIC\0")
            _update_text(digest, joint.mimic.source_joint)
            digest.update(
                np.asarray(
                    (joint.mimic.multiplier, joint.mimic.offset),
                    dtype="<f8",
                ).tobytes()
            )
    for vector in (
        q_start,
        direction,
        np.asarray((phase.lower, phase.upper)),
        base_transform.reshape(-1),
    ):
        digest.update(
            np.asarray(
                _normalize_signed_zero(vector),
                dtype="<f8",
            ).tobytes()
        )
    return digest.hexdigest()


def _canonical_surface(
    triangles_m: Sequence[Sequence[Sequence[float]]] | np.ndarray,
    *,
    label: str,
) -> tuple[np.ndarray, str]:
    triangles = np.asarray(triangles_m, dtype=np.float64)
    if (
        triangles.ndim != 3
        or triangles.shape[1:] != (3, 3)
        or len(triangles) == 0
        or not np.all(np.isfinite(triangles))
    ):
        raise ContinuousCollisionError(
            f"{label} must be a non-empty finite array with shape (F, 3, 3)"
        )
    rows: list[tuple[bytes, np.ndarray]] = []
    for triangle in triangles:
        canonical = _normalize_signed_zero(triangle)
        order = sorted(
            range(3),
            key=lambda index: tuple(
                float(value) for value in canonical[index]
            ),
        )
        canonical = canonical[np.asarray(order, dtype=np.int64)]
        with np.errstate(over="ignore", invalid="ignore"):
            edges = canonical[1:] - canonical[0]
        if not np.all(np.isfinite(edges)):
            raise ContinuousCollisionError(
                f"{label} triangle edge arithmetic overflowed"
            )
        edge_scale = float(np.max(np.abs(edges)))
        if edge_scale == 0.0:
            raise ContinuousCollisionError(
                f"{label} contains an exactly degenerate triangle"
            )
        scaled_cross = np.cross(
            edges[0] / edge_scale,
            edges[1] / edge_scale,
        )
        if (
            not np.all(np.isfinite(scaled_cross))
            or float(np.linalg.norm(scaled_cross)) == 0.0
        ):
            raise ContinuousCollisionError(
                f"{label} contains an exactly degenerate triangle"
            )
        little_endian = np.asarray(canonical, dtype="<f8")
        rows.append((little_endian.tobytes(order="C"), canonical))
    rows.sort(key=lambda row: row[0])
    ordered = np.stack([row[1] for row in rows])
    ordered.setflags(write=False)
    digest = hashlib.sha256()
    digest.update(b"CARTS_UNORIENTED_UNORDERED_TRIANGLE_SURFACE_V1\0")
    digest.update(np.asarray((len(ordered),), dtype="<u8").tobytes())
    for key, _triangle in rows:
        digest.update(key)
    return ordered, digest.hexdigest()


def _strictly_separated(
    first_lower: np.ndarray,
    first_upper: np.ndarray,
    second_lower: np.ndarray,
    second_upper: np.ndarray,
) -> bool:
    return bool(
        np.any(first_upper < second_lower)
        or np.any(second_upper < first_lower)
    )


class _StaticTriangleBVH:
    def __init__(self, triangles_m: np.ndarray) -> None:
        self.triangles_m = triangles_m
        self.face_lower_m = np.min(triangles_m, axis=1)
        self.face_upper_m = np.max(triangles_m, axis=1)
        self.face_centroid_m = (
            triangles_m[:, 0] / 3.0
            + triangles_m[:, 1] / 3.0
            + triangles_m[:, 2] / 3.0
        )
        self.root = self._build(tuple(range(len(triangles_m))))

    def _build(self, indices: tuple[int, ...]) -> _BVHNode:
        index_array = np.asarray(indices, dtype=np.int64)
        lower = np.min(self.face_lower_m[index_array], axis=0)
        upper = np.max(self.face_upper_m[index_array], axis=0)
        if len(indices) <= 4:
            return _BVHNode(
                lower,
                upper,
                indices,
                len(indices),
            )
        centroid_lower = np.min(self.face_centroid_m[index_array], axis=0)
        centroid_upper = np.max(self.face_centroid_m[index_array], axis=0)
        axis = int(np.argmax(centroid_upper - centroid_lower))
        ordered = tuple(
            sorted(
                indices,
                key=lambda index: (
                    float(self.face_centroid_m[index, axis]),
                    index,
                ),
            )
        )
        middle = len(ordered) // 2
        left = self._build(ordered[:middle])
        right = self._build(ordered[middle:])
        return _BVHNode(
            lower,
            upper,
            (),
            len(indices),
            left,
            right,
        )

    def potential_faces(
        self,
        lower_m: np.ndarray,
        upper_m: np.ndarray,
        counters: _Counters,
    ) -> tuple[tuple[int, ...], int]:
        candidates: list[int] = []
        pruned_face_count = 0
        stack = [self.root]
        while stack:
            node = stack.pop()
            counters.bvh_node_visits += 1
            if _strictly_separated(
                lower_m,
                upper_m,
                node.lower_m,
                node.upper_m,
            ):
                pruned_face_count += node.subtree_face_count
                continue
            if node.leaf:
                counters.bvh_leaf_visits += 1
                candidates.extend(node.face_indices)
                continue
            if node.right is None or node.left is None:
                raise ContinuousCollisionError(
                    "static BVH internal node is incomplete"
                )
            stack.append(node.right)
            stack.append(node.left)
        return tuple(candidates), pruned_face_count


def _proper_se3(value: object) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ContinuousCollisionError(
            "object_from_hand_base must be a finite 4x4 matrix"
        )
    structural_gamma = (
        128.0
        * np.finfo(np.float64).eps
        / (1.0 - 128.0 * np.finfo(np.float64).eps)
    )
    if not np.allclose(
        matrix[3],
        np.asarray((0.0, 0.0, 0.0, 1.0)),
        rtol=0.0,
        atol=structural_gamma,
    ):
        raise ContinuousCollisionError(
            "object_from_hand_base has an invalid homogeneous row"
        )
    rotation = matrix[:3, :3]
    rotation_bound = structural_gamma * max(
        1.0,
        float(np.linalg.norm(rotation, ord=np.inf)),
    )
    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3),
        rtol=0.0,
        atol=rotation_bound,
    ) or not np.isclose(
        np.linalg.det(rotation),
        1.0,
        rtol=0.0,
        atol=rotation_bound,
    ):
        raise ContinuousCollisionError(
            "object_from_hand_base rotation must belong to SO(3)"
        )
    result = np.array(matrix, copy=True)
    result.setflags(write=False)
    return result


def _build_audit(
    *,
    moving_hash: str,
    static_hash: str,
    motion_hash: str,
    link_name: str,
    moving_count: int,
    static_count: int,
    maximum_intervals: int,
    counters: _Counters,
    terminal_unresolved_pairs: int,
    entire_phase_covered: bool,
    unresolved_reason: str,
) -> ContinuousCollisionAudit:
    return ContinuousCollisionAudit(
        method_id=METHOD_ID,
        interval_kinematics_method_id=INTERVAL_KINEMATICS_METHOD_ID,
        moving_surface_geometry_sha256=moving_hash,
        static_surface_geometry_sha256=static_hash,
        motion_contract_sha256=motion_hash,
        link_name=link_name,
        moving_triangle_count=moving_count,
        static_triangle_count=static_count,
        pair_count_per_interval=moving_count * static_count,
        maximum_subdivision_intervals=maximum_intervals,
        processed_interval_count=counters.processed_intervals,
        certified_free_leaf_interval_count=counters.free_leaf_intervals,
        subdivided_interval_count=counters.subdivisions,
        point_motion_evaluation_count=(
            counters.point_motion_evaluations
        ),
        bvh_node_visit_count=counters.bvh_node_visits,
        bvh_leaf_visit_count=counters.bvh_leaf_visits,
        leaf_pair_evaluation_count=counters.leaf_pair_evaluations,
        pair_universe_count=counters.pair_universe,
        pair_coverage_count=counters.pair_coverage,
        strictly_separated_pair_count=counters.strictly_separated_pairs,
        potential_overlap_pair_observation_count=(
            counters.potential_overlap_pairs
        ),
        terminal_unresolved_pair_count=terminal_unresolved_pairs,
        all_processed_pairs_accounted_for=(
            counters.pair_coverage == counters.pair_universe
        ),
        entire_phase_covered=entire_phase_covered,
        unresolved_reason=unresolved_reason,
        claim_limitations=CLAIM_LIMITATIONS,
    )


def certify_moving_link_surface_separated_from_static_surface(
    *,
    backend: DirectedIntervalKinematics,
    link_name: str,
    q_start: Sequence[float] | np.ndarray,
    direction: Sequence[float] | np.ndarray,
    phase: IntervalBounds,
    object_from_hand_base: Sequence[Sequence[float]] | np.ndarray,
    moving_triangles_link_m: (
        Sequence[Sequence[Sequence[float]]] | np.ndarray
    ),
    static_triangles_object_m: (
        Sequence[Sequence[Sequence[float]]] | np.ndarray
    ),
    maximum_subdivision_intervals: int,
) -> ContinuousCollisionCertificate:
    """Certify strict moving/static surface separation on ``phase``.

    At each phase, a rigidly transformed triangle is the convex hull of its
    three transformed vertices, so every Cartesian coordinate extremum occurs
    at a transformed vertex.  Directed point-motion intervals therefore make
    their union AABB an enclosure of the whole swept triangle.  Strict AABB
    separation on one axis proves the corresponding triangle pair disjoint.
    A free result is returned only when the left-first closed bisection leaves
    cover the entire requested phase and every moving/static pair is proven
    separated on every leaf.
    """

    if not isinstance(backend, DirectedIntervalKinematics):
        raise ContinuousCollisionError(
            "continuous collision requires DirectedIntervalKinematics"
        )
    if not str(link_name):
        raise ContinuousCollisionError("link_name cannot be empty")
    if not isinstance(phase, IntervalBounds):
        raise ContinuousCollisionError("phase must be explicit IntervalBounds")
    if (
        not isinstance(maximum_subdivision_intervals, int)
        or isinstance(maximum_subdivision_intervals, bool)
        or maximum_subdivision_intervals <= 0
    ):
        raise ContinuousCollisionError(
            "maximum_subdivision_intervals must be a positive integer"
        )
    joint_count = len(backend.hand_model.independent_joint_names)
    start = np.asarray(q_start, dtype=np.float64)
    path_direction = np.asarray(direction, dtype=np.float64)
    if (
        start.shape != (joint_count,)
        or path_direction.shape != (joint_count,)
        or not np.all(np.isfinite(start))
        or not np.all(np.isfinite(path_direction))
    ):
        raise ContinuousCollisionError(
            "q_start and direction must match independent joint coordinates"
        )
    base = _proper_se3(object_from_hand_base)
    motion_hash = _motion_contract_sha256(
        backend=backend,
        link_name=str(link_name),
        q_start=start,
        direction=path_direction,
        phase=phase,
        base_transform=base,
    )
    moving, moving_hash = _canonical_surface(
        moving_triangles_link_m,
        label="moving link surface",
    )
    static, static_hash = _canonical_surface(
        static_triangles_object_m,
        label="static object surface",
    )
    static_bvh = _StaticTriangleBVH(static)
    static_lower = static_bvh.face_lower_m
    static_upper = static_bvh.face_upper_m
    counters = _Counters()
    free_leaves: list[IntervalBounds] = []
    pending: list[IntervalBounds] = [phase]

    while pending:
        interval = pending.pop()
        if (
            counters.processed_intervals
            >= maximum_subdivision_intervals
        ):
            audit = _build_audit(
                moving_hash=moving_hash,
                static_hash=static_hash,
                motion_hash=motion_hash,
                link_name=str(link_name),
                moving_count=len(moving),
                static_count=len(static),
                maximum_intervals=maximum_subdivision_intervals,
                counters=counters,
                terminal_unresolved_pairs=0,
                entire_phase_covered=False,
                unresolved_reason=(
                    "SUBDIVISION_INTERVAL_BUDGET_EXHAUSTED"
                ),
            )
            return ContinuousCollisionCertificate(
                state=ContinuousCollisionState.UNRESOLVED,
                searched_phase=phase,
                certified_free_leaf_intervals=tuple(free_leaves),
                unresolved_interval=interval,
                audit=audit,
            )
        counters.processed_intervals += 1
        interval_overlap_pairs = 0
        backend_failure_reason: str | None = None
        interval_pair_universe = len(moving) * len(static)
        counters.pair_universe += interval_pair_universe

        for moving_triangle in moving:
            swept_lower = np.full(3, math.inf, dtype=np.float64)
            swept_upper = np.full(3, -math.inf, dtype=np.float64)
            for vertex in moving_triangle:
                try:
                    motion = backend.point_motion(
                        link_name=str(link_name),
                        q_start=start,
                        direction=path_direction,
                        phase_lower=interval.lower,
                        phase_upper=interval.upper,
                        base_transform=base,
                        point_local_m=vertex,
                    )
                except IntervalKinematicsError as error:
                    backend_failure_reason = (
                        "INTERVAL_KINEMATICS_UNRESOLVED:"
                        + type(error).__name__
                    )
                    break
                counters.point_motion_evaluations += 1
                if motion.method_id != INTERVAL_KINEMATICS_METHOD_ID:
                    backend_failure_reason = (
                        "INTERVAL_KINEMATICS_METHOD_MISMATCH"
                    )
                    break
                swept_lower = np.minimum(
                    swept_lower,
                    np.asarray(
                        [bound.lower for bound in motion.position_object_m],
                        dtype=np.float64,
                    ),
                )
                swept_upper = np.maximum(
                    swept_upper,
                    np.asarray(
                        [bound.upper for bound in motion.position_object_m],
                        dtype=np.float64,
                    ),
                )
            if backend_failure_reason is not None:
                break
            candidates, pruned_count = static_bvh.potential_faces(
                swept_lower,
                swept_upper,
                counters,
            )
            counters.pair_coverage += pruned_count
            counters.strictly_separated_pairs += pruned_count
            for face_index in candidates:
                counters.leaf_pair_evaluations += 1
                counters.pair_coverage += 1
                if _strictly_separated(
                    swept_lower,
                    swept_upper,
                    static_lower[face_index],
                    static_upper[face_index],
                ):
                    counters.strictly_separated_pairs += 1
                else:
                    counters.potential_overlap_pairs += 1
                    interval_overlap_pairs += 1

        if backend_failure_reason is not None:
            audit = _build_audit(
                moving_hash=moving_hash,
                static_hash=static_hash,
                motion_hash=motion_hash,
                link_name=str(link_name),
                moving_count=len(moving),
                static_count=len(static),
                maximum_intervals=maximum_subdivision_intervals,
                counters=counters,
                terminal_unresolved_pairs=interval_overlap_pairs,
                entire_phase_covered=False,
                unresolved_reason=backend_failure_reason,
            )
            return ContinuousCollisionCertificate(
                state=ContinuousCollisionState.UNRESOLVED,
                searched_phase=phase,
                certified_free_leaf_intervals=tuple(free_leaves),
                unresolved_interval=interval,
                audit=audit,
            )
        if counters.pair_coverage > counters.pair_universe:
            raise ContinuousCollisionError(
                "continuous collision pair accounting overflowed"
            )
        if interval_overlap_pairs == 0:
            counters.free_leaf_intervals += 1
            free_leaves.append(interval)
            continue
        if counters.processed_intervals >= maximum_subdivision_intervals:
            audit = _build_audit(
                moving_hash=moving_hash,
                static_hash=static_hash,
                motion_hash=motion_hash,
                link_name=str(link_name),
                moving_count=len(moving),
                static_count=len(static),
                maximum_intervals=maximum_subdivision_intervals,
                counters=counters,
                terminal_unresolved_pairs=interval_overlap_pairs,
                entire_phase_covered=False,
                unresolved_reason="SUBDIVISION_INTERVAL_BUDGET_EXHAUSTED",
            )
            return ContinuousCollisionCertificate(
                state=ContinuousCollisionState.UNRESOLVED,
                searched_phase=phase,
                certified_free_leaf_intervals=tuple(free_leaves),
                unresolved_interval=interval,
                audit=audit,
            )
        midpoint = interval.lower + 0.5 * (
            interval.upper - interval.lower
        )
        if not interval.lower < midpoint < interval.upper:
            audit = _build_audit(
                moving_hash=moving_hash,
                static_hash=static_hash,
                motion_hash=motion_hash,
                link_name=str(link_name),
                moving_count=len(moving),
                static_count=len(static),
                maximum_intervals=maximum_subdivision_intervals,
                counters=counters,
                terminal_unresolved_pairs=interval_overlap_pairs,
                entire_phase_covered=False,
                unresolved_reason="ADJACENT_BINARY64_PHASE_ENDPOINTS",
            )
            return ContinuousCollisionCertificate(
                state=ContinuousCollisionState.UNRESOLVED,
                searched_phase=phase,
                certified_free_leaf_intervals=tuple(free_leaves),
                unresolved_interval=interval,
                audit=audit,
            )
        counters.subdivisions += 1
        # Stack is LIFO: pushing right then left makes traversal left-first.
        pending.append(IntervalBounds(midpoint, interval.upper))
        pending.append(IntervalBounds(interval.lower, midpoint))

    audit = _build_audit(
        moving_hash=moving_hash,
        static_hash=static_hash,
        motion_hash=motion_hash,
        link_name=str(link_name),
        moving_count=len(moving),
        static_count=len(static),
        maximum_intervals=maximum_subdivision_intervals,
        counters=counters,
        terminal_unresolved_pairs=0,
        entire_phase_covered=True,
        unresolved_reason="NONE",
    )
    return ContinuousCollisionCertificate(
        state=ContinuousCollisionState.CERTIFIED_FREE,
        searched_phase=phase,
        certified_free_leaf_intervals=tuple(free_leaves),
        unresolved_interval=None,
        audit=audit,
    )


class _MovingPairBackendUnresolved(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = str(reason)


def _moving_pair_contract_sha256(
    *,
    first_link_name: str,
    first_surface_hash: str,
    first_motion_hash: str,
    second_link_name: str,
    second_surface_hash: str,
    second_motion_hash: str,
) -> str:
    entries = sorted(
        (
            (
                first_link_name,
                first_surface_hash,
                first_motion_hash,
            ),
            (
                second_link_name,
                second_surface_hash,
                second_motion_hash,
            ),
        )
    )
    digest = hashlib.sha256()
    digest.update(b"CARTS_MOVING_SURFACE_PAIR_CONTRACT_V1\0")
    for entry in entries:
        for value in entry:
            _update_text(digest, value)
    return digest.hexdigest()


def _enclose_moving_surface_vertices(
    *,
    backend: DirectedIntervalKinematics,
    link_name: str,
    triangles_link_m: np.ndarray,
    q_start: np.ndarray,
    direction: np.ndarray,
    interval: IntervalBounds,
    base_transform: np.ndarray,
    counters: _MovingPairCounters,
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.empty((len(triangles_link_m), 3, 3), dtype=np.float64)
    upper = np.empty_like(lower)
    for face_index, triangle in enumerate(triangles_link_m):
        for vertex_index, vertex in enumerate(triangle):
            try:
                motion = backend.point_motion(
                    link_name=link_name,
                    q_start=q_start,
                    direction=direction,
                    phase_lower=interval.lower,
                    phase_upper=interval.upper,
                    base_transform=base_transform,
                    point_local_m=vertex,
                )
            except IntervalKinematicsError as error:
                raise _MovingPairBackendUnresolved(
                    "INTERVAL_KINEMATICS_UNRESOLVED:"
                    + type(error).__name__
                ) from error
            counters.point_motion_evaluations += 1
            if motion.method_id != INTERVAL_KINEMATICS_METHOD_ID:
                raise _MovingPairBackendUnresolved(
                    "INTERVAL_KINEMATICS_METHOD_MISMATCH"
                )
            lower[face_index, vertex_index] = np.asarray(
                [bound.lower for bound in motion.position_object_m],
                dtype=np.float64,
            )
            upper[face_index, vertex_index] = np.asarray(
                [bound.upper for bound in motion.position_object_m],
                dtype=np.float64,
            )
    return lower, upper


def _relative_triangle_pair_strictly_separated(
    *,
    first_lower: np.ndarray,
    first_upper: np.ndarray,
    second_lower: np.ndarray,
    second_upper: np.ndarray,
    counters: _MovingPairCounters,
) -> bool:
    all_strictly_negative = np.ones(3, dtype=bool)
    all_strictly_positive = np.ones(3, dtype=bool)
    for first_vertex in range(3):
        for second_vertex in range(3):
            with np.errstate(over="ignore", invalid="ignore"):
                relative_lower = np.nextafter(
                    first_lower[first_vertex]
                    - second_upper[second_vertex],
                    -np.inf,
                )
                relative_upper = np.nextafter(
                    first_upper[first_vertex]
                    - second_lower[second_vertex],
                    np.inf,
                )
            counters.relative_coordinate_interval_evaluations += 3
            finite = np.isfinite(relative_lower) & np.isfinite(
                relative_upper
            )
            all_strictly_negative &= finite & (relative_upper < 0.0)
            all_strictly_positive &= finite & (relative_lower > 0.0)
    return bool(
        np.any(all_strictly_negative | all_strictly_positive)
    )


def _build_moving_pair_audit(
    *,
    first_link_name: str,
    second_link_name: str,
    first_surface_hash: str,
    second_surface_hash: str,
    pair_contract_hash: str,
    first_triangle_count: int,
    second_triangle_count: int,
    maximum_intervals: int,
    counters: _MovingPairCounters,
    terminal_unresolved_pairs: int,
    entire_phase_covered: bool,
    unresolved_reason: str,
) -> MovingSurfacePairCollisionAudit:
    return MovingSurfacePairCollisionAudit(
        method_id=MOVING_PAIR_METHOD_ID,
        interval_kinematics_method_id=INTERVAL_KINEMATICS_METHOD_ID,
        first_link_name=first_link_name,
        second_link_name=second_link_name,
        first_surface_geometry_sha256=first_surface_hash,
        second_surface_geometry_sha256=second_surface_hash,
        pair_contract_sha256=pair_contract_hash,
        first_triangle_count=first_triangle_count,
        second_triangle_count=second_triangle_count,
        pair_count_per_interval=(
            first_triangle_count * second_triangle_count
        ),
        maximum_subdivision_intervals=maximum_intervals,
        processed_interval_count=counters.processed_intervals,
        certified_free_leaf_interval_count=counters.free_leaf_intervals,
        subdivided_interval_count=counters.subdivisions,
        point_motion_evaluation_count=counters.point_motion_evaluations,
        relative_coordinate_interval_evaluation_count=(
            counters.relative_coordinate_interval_evaluations
        ),
        pair_universe_count=counters.pair_universe,
        pair_coverage_count=counters.pair_coverage,
        strictly_separated_pair_count=counters.strictly_separated_pairs,
        potential_overlap_pair_observation_count=(
            counters.potential_overlap_pairs
        ),
        terminal_unresolved_pair_count=terminal_unresolved_pairs,
        all_processed_pairs_accounted_for=(
            counters.pair_coverage == counters.pair_universe
        ),
        entire_phase_covered=entire_phase_covered,
        unresolved_reason=unresolved_reason,
        claim_limitations=MOVING_PAIR_CLAIM_LIMITATIONS,
    )


def _moving_pair_certificate(
    *,
    state: ContinuousCollisionState,
    phase: IntervalBounds,
    free_leaves: list[IntervalBounds],
    unresolved_interval: IntervalBounds | None,
    first_link_name: str,
    second_link_name: str,
    first_surface_hash: str,
    second_surface_hash: str,
    pair_contract_hash: str,
    first_triangle_count: int,
    second_triangle_count: int,
    maximum_intervals: int,
    counters: _MovingPairCounters,
    terminal_unresolved_pairs: int,
    entire_phase_covered: bool,
    unresolved_reason: str,
) -> MovingSurfacePairCollisionCertificate:
    audit = _build_moving_pair_audit(
        first_link_name=first_link_name,
        second_link_name=second_link_name,
        first_surface_hash=first_surface_hash,
        second_surface_hash=second_surface_hash,
        pair_contract_hash=pair_contract_hash,
        first_triangle_count=first_triangle_count,
        second_triangle_count=second_triangle_count,
        maximum_intervals=maximum_intervals,
        counters=counters,
        terminal_unresolved_pairs=terminal_unresolved_pairs,
        entire_phase_covered=entire_phase_covered,
        unresolved_reason=unresolved_reason,
    )
    return MovingSurfacePairCollisionCertificate(
        state=state,
        searched_phase=phase,
        certified_free_leaf_intervals=tuple(free_leaves),
        unresolved_interval=unresolved_interval,
        audit=audit,
    )


def certify_moving_link_surfaces_separated_from_each_other(
    *,
    backend: DirectedIntervalKinematics,
    first_link_name: str,
    second_link_name: str,
    q_start: Sequence[float] | np.ndarray,
    direction: Sequence[float] | np.ndarray,
    phase: IntervalBounds,
    object_from_hand_base: Sequence[Sequence[float]] | np.ndarray,
    first_triangles_link_m: (
        Sequence[Sequence[Sequence[float]]] | np.ndarray
    ),
    second_triangles_link_m: (
        Sequence[Sequence[Sequence[float]]] | np.ndarray
    ),
    maximum_subdivision_intervals: int,
) -> MovingSurfacePairCollisionCertificate:
    """Certify strict separation between two moving link surfaces.

    For each triangle pair and coordinate axis, all nine vertex-pair relative
    intervals must have one common strict sign.  This places one complete
    moving triangle convex hull strictly on one side of the other throughout
    that phase interval.  Independent point enclosures intentionally sacrifice
    FK correlation for a conservative proof.
    """

    if not isinstance(backend, DirectedIntervalKinematics):
        raise ContinuousCollisionError(
            "moving-pair collision requires DirectedIntervalKinematics"
        )
    first_link = str(first_link_name)
    second_link = str(second_link_name)
    if not first_link or not second_link or first_link == second_link:
        raise ContinuousCollisionError(
            "moving-pair collision needs two distinct named links"
        )
    available_links = {backend.hand_model.base_link}
    for joint in backend.hand_model.joints.values():
        available_links.add(joint.parent_link)
        available_links.add(joint.child_link)
    if first_link not in available_links or second_link not in available_links:
        raise ContinuousCollisionError(
            "moving-pair link is outside the hand model"
        )
    if not isinstance(phase, IntervalBounds):
        raise ContinuousCollisionError(
            "moving-pair phase must be explicit IntervalBounds"
        )
    if (
        not isinstance(maximum_subdivision_intervals, int)
        or isinstance(maximum_subdivision_intervals, bool)
        or maximum_subdivision_intervals <= 0
    ):
        raise ContinuousCollisionError(
            "maximum_subdivision_intervals must be a positive integer"
        )
    joint_count = len(backend.hand_model.independent_joint_names)
    start = np.asarray(q_start, dtype=np.float64)
    path_direction = np.asarray(direction, dtype=np.float64)
    if (
        start.shape != (joint_count,)
        or path_direction.shape != (joint_count,)
        or not np.all(np.isfinite(start))
        or not np.all(np.isfinite(path_direction))
    ):
        raise ContinuousCollisionError(
            "q_start and direction must match independent joint coordinates"
        )
    base = _proper_se3(object_from_hand_base)
    first_surface, first_hash = _canonical_surface(
        first_triangles_link_m,
        label=f"moving link surface {first_link}",
    )
    second_surface, second_hash = _canonical_surface(
        second_triangles_link_m,
        label=f"moving link surface {second_link}",
    )
    first_motion_hash = _motion_contract_sha256(
        backend=backend,
        link_name=first_link,
        q_start=start,
        direction=path_direction,
        phase=phase,
        base_transform=base,
    )
    second_motion_hash = _motion_contract_sha256(
        backend=backend,
        link_name=second_link,
        q_start=start,
        direction=path_direction,
        phase=phase,
        base_transform=base,
    )
    pair_contract_hash = _moving_pair_contract_sha256(
        first_link_name=first_link,
        first_surface_hash=first_hash,
        first_motion_hash=first_motion_hash,
        second_link_name=second_link,
        second_surface_hash=second_hash,
        second_motion_hash=second_motion_hash,
    )
    counters = _MovingPairCounters()
    free_leaves: list[IntervalBounds] = []
    pending: list[IntervalBounds] = [phase]
    pair_count = len(first_surface) * len(second_surface)

    while pending:
        interval = pending.pop()
        if counters.processed_intervals >= maximum_subdivision_intervals:
            return _moving_pair_certificate(
                state=ContinuousCollisionState.UNRESOLVED,
                phase=phase,
                free_leaves=free_leaves,
                unresolved_interval=interval,
                first_link_name=first_link,
                second_link_name=second_link,
                first_surface_hash=first_hash,
                second_surface_hash=second_hash,
                pair_contract_hash=pair_contract_hash,
                first_triangle_count=len(first_surface),
                second_triangle_count=len(second_surface),
                maximum_intervals=maximum_subdivision_intervals,
                counters=counters,
                terminal_unresolved_pairs=0,
                entire_phase_covered=False,
                unresolved_reason=(
                    "SUBDIVISION_INTERVAL_BUDGET_EXHAUSTED"
                ),
            )
        counters.processed_intervals += 1
        counters.pair_universe += pair_count
        try:
            first_lower, first_upper = _enclose_moving_surface_vertices(
                backend=backend,
                link_name=first_link,
                triangles_link_m=first_surface,
                q_start=start,
                direction=path_direction,
                interval=interval,
                base_transform=base,
                counters=counters,
            )
            second_lower, second_upper = _enclose_moving_surface_vertices(
                backend=backend,
                link_name=second_link,
                triangles_link_m=second_surface,
                q_start=start,
                direction=path_direction,
                interval=interval,
                base_transform=base,
                counters=counters,
            )
        except _MovingPairBackendUnresolved as error:
            return _moving_pair_certificate(
                state=ContinuousCollisionState.UNRESOLVED,
                phase=phase,
                free_leaves=free_leaves,
                unresolved_interval=interval,
                first_link_name=first_link,
                second_link_name=second_link,
                first_surface_hash=first_hash,
                second_surface_hash=second_hash,
                pair_contract_hash=pair_contract_hash,
                first_triangle_count=len(first_surface),
                second_triangle_count=len(second_surface),
                maximum_intervals=maximum_subdivision_intervals,
                counters=counters,
                terminal_unresolved_pairs=0,
                entire_phase_covered=False,
                unresolved_reason=error.reason,
            )

        interval_overlap_pairs = 0
        for first_face_index in range(len(first_surface)):
            for second_face_index in range(len(second_surface)):
                separated = _relative_triangle_pair_strictly_separated(
                    first_lower=first_lower[first_face_index],
                    first_upper=first_upper[first_face_index],
                    second_lower=second_lower[second_face_index],
                    second_upper=second_upper[second_face_index],
                    counters=counters,
                )
                counters.pair_coverage += 1
                if separated:
                    counters.strictly_separated_pairs += 1
                else:
                    counters.potential_overlap_pairs += 1
                    interval_overlap_pairs += 1

        if interval_overlap_pairs == 0:
            counters.free_leaf_intervals += 1
            free_leaves.append(interval)
            continue
        if counters.processed_intervals >= maximum_subdivision_intervals:
            return _moving_pair_certificate(
                state=ContinuousCollisionState.UNRESOLVED,
                phase=phase,
                free_leaves=free_leaves,
                unresolved_interval=interval,
                first_link_name=first_link,
                second_link_name=second_link,
                first_surface_hash=first_hash,
                second_surface_hash=second_hash,
                pair_contract_hash=pair_contract_hash,
                first_triangle_count=len(first_surface),
                second_triangle_count=len(second_surface),
                maximum_intervals=maximum_subdivision_intervals,
                counters=counters,
                terminal_unresolved_pairs=interval_overlap_pairs,
                entire_phase_covered=False,
                unresolved_reason=(
                    "SUBDIVISION_INTERVAL_BUDGET_EXHAUSTED"
                ),
            )
        midpoint = interval.lower + 0.5 * (
            interval.upper - interval.lower
        )
        if not interval.lower < midpoint < interval.upper:
            return _moving_pair_certificate(
                state=ContinuousCollisionState.UNRESOLVED,
                phase=phase,
                free_leaves=free_leaves,
                unresolved_interval=interval,
                first_link_name=first_link,
                second_link_name=second_link,
                first_surface_hash=first_hash,
                second_surface_hash=second_hash,
                pair_contract_hash=pair_contract_hash,
                first_triangle_count=len(first_surface),
                second_triangle_count=len(second_surface),
                maximum_intervals=maximum_subdivision_intervals,
                counters=counters,
                terminal_unresolved_pairs=interval_overlap_pairs,
                entire_phase_covered=False,
                unresolved_reason="ADJACENT_BINARY64_PHASE_ENDPOINTS",
            )
        counters.subdivisions += 1
        pending.append(IntervalBounds(midpoint, interval.upper))
        pending.append(IntervalBounds(interval.lower, midpoint))

    return _moving_pair_certificate(
        state=ContinuousCollisionState.CERTIFIED_FREE,
        phase=phase,
        free_leaves=free_leaves,
        unresolved_interval=None,
        first_link_name=first_link,
        second_link_name=second_link,
        first_surface_hash=first_hash,
        second_surface_hash=second_hash,
        pair_contract_hash=pair_contract_hash,
        first_triangle_count=len(first_surface),
        second_triangle_count=len(second_surface),
        maximum_intervals=maximum_subdivision_intervals,
        counters=counters,
        terminal_unresolved_pairs=0,
        entire_phase_covered=True,
        unresolved_reason="NONE",
    )


__all__ = [
    "CLAIM_LIMITATIONS",
    "ContinuousCollisionAudit",
    "ContinuousCollisionCertificate",
    "ContinuousCollisionError",
    "ContinuousCollisionState",
    "METHOD_ID",
    "MOVING_PAIR_CLAIM_LIMITATIONS",
    "MOVING_PAIR_METHOD_ID",
    "MovingSurfacePairCollisionAudit",
    "MovingSurfacePairCollisionCertificate",
    "certify_moving_link_surfaces_separated_from_each_other",
    "certify_moving_link_surface_separated_from_static_surface",
]
