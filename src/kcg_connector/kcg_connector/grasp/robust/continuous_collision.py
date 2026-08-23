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
    BATCH_POINT_MOTION_METHOD_ID,
    DirectedIntervalKinematics,
    IntervalBounds,
    IntervalKinematicsError,
    METHOD_ID as INTERVAL_KINEMATICS_METHOD_ID,
)


METHOD_ID = (
    "CARTS_MP_INTERVAL_MOVING_STATIC_SURFACE_AABB_BVH_"
    "SWEPT_TRIANGLE_EAGER_PRECOMPUTED_AXIS_FAMILY_FRONTIER_V6"
)
PREPARED_STATIC_SURFACE_METHOD_ID = (
    "CARTS_PREPARED_STATIC_TRIANGLE_SURFACE_BVH_V1"
)
MOVING_PAIR_METHOD_ID = (
    "CARTS_MP_INTERVAL_MOVING_SURFACE_PAIR_RELATIVE_AXIS_V1"
)
INDEPENDENT_MOVING_PAIR_METHOD_ID = (
    "CARTS_MP_INTERVAL_INDEPENDENT_TWO_PHASE_SURFACE_PAIR_V1"
)
CLAIM_LIMITATIONS = (
    "MOVING_LINK_SURFACE_VS_STATIC_OBJECT_SURFACE_ONLY",
    "NOT_SOLID_CONTAINMENT_OR_INTERIOR_EXCLUSION",
    "NOT_HAND_SELF_COLLISION",
    "NOT_ENVIRONMENT_COLLISION",
    "NOT_MULTI_LINK_OR_FULL_HAND_PATH_CERTIFICATE",
    "POTENTIAL_CONTACT_TANGENCY_AND_COPLANARITY_ARE_UNRESOLVED",
    "AABB_AND_SWEPT_TRIANGLE_AXIS_SEPARATION_SUFFICIENT_NOT_NECESSARY",
)


def _gamma(operation_count: int) -> float:
    epsilon = np.finfo(np.float64).eps
    product = float(operation_count) * epsilon
    if operation_count <= 0 or product >= 1.0:
        raise ValueError("operation_count cannot produce a finite gamma_n")
    return product / (1.0 - product)


_SAT_DOT_ERROR = _gamma(128)
_NARROWPHASE_PAIR_PACKET_SIZE = 65536
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
INDEPENDENT_MOVING_PAIR_CLAIM_LIMITATIONS = (
    "TWO_MOVING_LINK_SURFACES_WITH_INDEPENDENT_SCALAR_PHASES_ONLY",
    "FULL_CARTESIAN_PRODUCT_OF_BOTH_REGISTERED_PHASE_INTERVALS",
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
    root_bvh_interval_count: int
    inherited_strictly_separated_pair_count: int
    frontier_pair_evaluation_count: int
    bvh_node_visit_count: int
    bvh_leaf_visit_count: int
    leaf_pair_evaluation_count: int
    narrowphase_pair_evaluation_count: int
    narrowphase_strictly_separated_pair_count: int
    narrowphase_invocation_interval_count: int
    narrowphase_eager_child_interval_count: int
    narrowphase_pair_packet_count: int
    narrowphase_root_pair_packet_count: int
    narrowphase_child_pair_packet_count: int
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
            self.root_bvh_interval_count,
            self.inherited_strictly_separated_pair_count,
            self.frontier_pair_evaluation_count,
            self.bvh_node_visit_count,
            self.bvh_leaf_visit_count,
            self.leaf_pair_evaluation_count,
            self.narrowphase_pair_evaluation_count,
            self.narrowphase_strictly_separated_pair_count,
            self.narrowphase_invocation_interval_count,
            self.narrowphase_eager_child_interval_count,
            self.narrowphase_pair_packet_count,
            self.narrowphase_root_pair_packet_count,
            self.narrowphase_child_pair_packet_count,
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
            or self.root_bvh_interval_count > 1
            or self.root_bvh_interval_count > self.processed_interval_count
            or self.inherited_strictly_separated_pair_count
            > self.strictly_separated_pair_count
            or self.frontier_pair_evaluation_count
            > self.leaf_pair_evaluation_count
            or self.inherited_strictly_separated_pair_count
            + self.leaf_pair_evaluation_count
            > self.pair_coverage_count
            or self.pair_universe_count
            != self.processed_interval_count * self.pair_count_per_interval
            or self.strictly_separated_pair_count
            + self.potential_overlap_pair_observation_count
            != self.pair_coverage_count
            or self.leaf_pair_evaluation_count > self.pair_coverage_count
            or self.narrowphase_pair_evaluation_count
            > self.leaf_pair_evaluation_count
            or self.narrowphase_strictly_separated_pair_count
            > self.narrowphase_pair_evaluation_count
            or self.narrowphase_pair_evaluation_count
            - self.narrowphase_strictly_separated_pair_count
            != self.potential_overlap_pair_observation_count
            or self.narrowphase_invocation_interval_count
            > self.processed_interval_count
            or self.narrowphase_eager_child_interval_count
            > self.narrowphase_invocation_interval_count
            or self.narrowphase_pair_packet_count
            < self.narrowphase_invocation_interval_count
            or self.narrowphase_pair_packet_count
            > self.narrowphase_pair_evaluation_count
            or self.narrowphase_root_pair_packet_count
            + self.narrowphase_child_pair_packet_count
            != self.narrowphase_pair_packet_count
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
            and self.root_bvh_interval_count != 1
        ):
            raise ContinuousCollisionError(
                "collision audit root-BVH coverage is incomplete"
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


@dataclass(frozen=True)
class IndependentMotionPhaseBox:
    """One closed box in two independently varying closure phases."""

    first_phase: IntervalBounds
    second_phase: IntervalBounds

    def __post_init__(self) -> None:
        if not isinstance(self.first_phase, IntervalBounds) or not isinstance(
            self.second_phase, IntervalBounds
        ):
            raise ContinuousCollisionError(
                "independent moving-pair phase box is malformed"
            )


@dataclass(frozen=True)
class IndependentMovingSurfacePairCollisionAudit:
    method_id: str
    interval_kinematics_method_id: str
    first_link_name: str
    second_link_name: str
    first_surface_geometry_sha256: str
    second_surface_geometry_sha256: str
    pair_contract_sha256: str
    first_triangle_count: int
    second_triangle_count: int
    pair_count_per_phase_box: int
    maximum_subdivision_phase_boxes: int
    processed_phase_box_count: int
    certified_free_leaf_phase_box_count: int
    subdivided_phase_box_count: int
    point_motion_evaluation_count: int
    relative_coordinate_interval_evaluation_count: int
    pair_universe_count: int
    pair_coverage_count: int
    strictly_separated_pair_count: int
    potential_overlap_pair_observation_count: int
    terminal_unresolved_pair_count: int
    all_processed_pairs_accounted_for: bool
    entire_phase_product_covered: bool
    unresolved_reason: str
    claim_limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.method_id != INDEPENDENT_MOVING_PAIR_METHOD_ID:
            raise ContinuousCollisionError(
                "independent moving-pair audit method mismatch"
            )
        if self.interval_kinematics_method_id != (
            INTERVAL_KINEMATICS_METHOD_ID
        ):
            raise ContinuousCollisionError(
                "independent moving-pair interval backend mismatch"
            )
        if (
            not str(self.first_link_name)
            or not str(self.second_link_name)
            or self.first_link_name == self.second_link_name
        ):
            raise ContinuousCollisionError(
                "independent moving-pair audit needs distinct links"
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
                    "independent moving-pair digest is invalid"
                )
        integer_fields = (
            self.first_triangle_count,
            self.second_triangle_count,
            self.pair_count_per_phase_box,
            self.maximum_subdivision_phase_boxes,
            self.processed_phase_box_count,
            self.certified_free_leaf_phase_box_count,
            self.subdivided_phase_box_count,
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
                "independent moving-pair counters must be non-negative"
            )
        if (
            self.first_triangle_count == 0
            or self.second_triangle_count == 0
            or self.maximum_subdivision_phase_boxes == 0
            or self.pair_count_per_phase_box
            != self.first_triangle_count * self.second_triangle_count
            or self.processed_phase_box_count
            > self.maximum_subdivision_phase_boxes
            or self.pair_universe_count
            != self.processed_phase_box_count
            * self.pair_count_per_phase_box
            or self.pair_coverage_count > self.pair_universe_count
            or self.strictly_separated_pair_count
            + self.potential_overlap_pair_observation_count
            != self.pair_coverage_count
            or self.certified_free_leaf_phase_box_count
            > self.processed_phase_box_count
        ):
            raise ContinuousCollisionError(
                "independent moving-pair coverage arithmetic is inconsistent"
            )
        if self.all_processed_pairs_accounted_for != (
            self.pair_coverage_count == self.pair_universe_count
        ):
            raise ContinuousCollisionError(
                "independent moving-pair accounting flag is inconsistent"
            )
        if self.all_processed_pairs_accounted_for:
            expected_point_evaluations = (
                self.processed_phase_box_count
                * 3
                * (
                    self.first_triangle_count
                    + self.second_triangle_count
                )
            )
            expected_relative_evaluations = (
                self.processed_phase_box_count
                * self.pair_count_per_phase_box
                * 27
            )
            if (
                self.point_motion_evaluation_count
                != expected_point_evaluations
                or self.relative_coordinate_interval_evaluation_count
                != expected_relative_evaluations
            ):
                raise ContinuousCollisionError(
                    "independent moving-pair interval coverage is incomplete"
                )
        if self.entire_phase_product_covered and (
            self.processed_phase_box_count
            != 2 * self.subdivided_phase_box_count + 1
            or self.certified_free_leaf_phase_box_count
            != self.subdivided_phase_box_count + 1
            or self.terminal_unresolved_pair_count != 0
        ):
            raise ContinuousCollisionError(
                "independent moving-pair free partition is incomplete"
            )
        if (
            self.claim_limitations
            != INDEPENDENT_MOVING_PAIR_CLAIM_LIMITATIONS
            or not str(self.unresolved_reason)
        ):
            raise ContinuousCollisionError(
                "independent moving-pair limitations or reason changed"
            )


@dataclass(frozen=True)
class IndependentMovingSurfacePairCollisionCertificate:
    state: ContinuousCollisionState
    searched_phase_box: IndependentMotionPhaseBox
    certified_free_leaf_phase_boxes: tuple[
        IndependentMotionPhaseBox, ...
    ]
    unresolved_phase_box: IndependentMotionPhaseBox | None
    audit: IndependentMovingSurfacePairCollisionAudit

    def __post_init__(self) -> None:
        if (
            not isinstance(self.state, ContinuousCollisionState)
            or not isinstance(
                self.searched_phase_box, IndependentMotionPhaseBox
            )
        ):
            raise ContinuousCollisionError(
                "independent moving-pair certificate is malformed"
            )
        leaves = tuple(self.certified_free_leaf_phase_boxes)
        if not all(
            isinstance(value, IndependentMotionPhaseBox)
            for value in leaves
        ):
            raise ContinuousCollisionError(
                "independent moving-pair free leaves are malformed"
            )
        object.__setattr__(self, "certified_free_leaf_phase_boxes", leaves)
        if self.audit.certified_free_leaf_phase_box_count != len(leaves):
            raise ContinuousCollisionError(
                "independent moving-pair free-leaf count is inconsistent"
            )
        if self.state is ContinuousCollisionState.CERTIFIED_FREE:
            if (
                self.unresolved_phase_box is not None
                or not leaves
                or not self.audit.entire_phase_product_covered
                or not self.audit.all_processed_pairs_accounted_for
                or self.audit.unresolved_reason != "NONE"
            ):
                raise ContinuousCollisionError(
                    "independent moving-pair free certificate is inconsistent"
                )
        elif (
            not isinstance(
                self.unresolved_phase_box, IndependentMotionPhaseBox
            )
            or self.audit.entire_phase_product_covered
            or self.audit.unresolved_reason == "NONE"
        ):
            raise ContinuousCollisionError(
                "independent moving-pair unresolved certificate is inconsistent"
            )


@dataclass
class _Counters:
    processed_intervals: int = 0
    free_leaf_intervals: int = 0
    subdivisions: int = 0
    point_motion_evaluations: int = 0
    root_bvh_intervals: int = 0
    inherited_strictly_separated_pairs: int = 0
    frontier_pair_evaluations: int = 0
    bvh_node_visits: int = 0
    bvh_leaf_visits: int = 0
    leaf_pair_evaluations: int = 0
    narrowphase_pair_evaluations: int = 0
    narrowphase_strictly_separated_pairs: int = 0
    narrowphase_invocation_intervals: int = 0
    narrowphase_eager_child_intervals: int = 0
    narrowphase_pair_packets: int = 0
    narrowphase_root_pair_packets: int = 0
    narrowphase_child_pair_packets: int = 0
    pair_universe: int = 0
    pair_coverage: int = 0
    strictly_separated_pairs: int = 0
    potential_overlap_pairs: int = 0


_UNRESOLVED_PAIR_FRONTIER_TOKEN = object()


@dataclass(frozen=True)
class _UnresolvedTrianglePairFrontier:
    """Immutable exact moving/static face-pair ids still unresolved."""

    pair_ids: np.ndarray
    moving_triangle_count: int
    static_triangle_count: int
    _construction_token: object

    def __post_init__(self) -> None:
        if (
            self._construction_token is not _UNRESOLVED_PAIR_FRONTIER_TOKEN
            or not isinstance(self.moving_triangle_count, int)
            or isinstance(self.moving_triangle_count, bool)
            or self.moving_triangle_count <= 0
            or not isinstance(self.static_triangle_count, int)
            or isinstance(self.static_triangle_count, bool)
            or self.static_triangle_count <= 0
            or self.moving_triangle_count
            > np.iinfo(np.int64).max // self.static_triangle_count
        ):
            raise ContinuousCollisionError(
                "unresolved triangle-pair frontier contract is invalid"
            )
        pair_ids = np.asarray(self.pair_ids, dtype=np.int64)
        pair_limit = self.moving_triangle_count * self.static_triangle_count
        if (
            pair_ids.ndim != 1
            or np.any(pair_ids < 0)
            or np.any(pair_ids >= pair_limit)
        ):
            raise ContinuousCollisionError(
                "unresolved triangle-pair frontier ids are invalid"
            )
        ordered = np.sort(pair_ids, kind="stable")
        if len(ordered) > 1 and np.any(ordered[1:] == ordered[:-1]):
            raise ContinuousCollisionError(
                "unresolved triangle-pair frontier contains duplicates"
            )
        immutable = np.frombuffer(
            np.asarray(ordered, dtype="<i8").tobytes(order="C"),
            dtype="<i8",
        )
        object.__setattr__(self, "pair_ids", immutable)

    @property
    def pair_count(self) -> int:
        return len(self.pair_ids)


def _unresolved_triangle_pair_frontier(
    pair_ids: Sequence[int] | np.ndarray,
    *,
    moving_triangle_count: int,
    static_triangle_count: int,
) -> _UnresolvedTrianglePairFrontier:
    return _UnresolvedTrianglePairFrontier(
        pair_ids=np.asarray(pair_ids, dtype=np.int64),
        moving_triangle_count=moving_triangle_count,
        static_triangle_count=static_triangle_count,
        _construction_token=_UNRESOLVED_PAIR_FRONTIER_TOKEN,
    )


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


def _moving_projection_interval_bounds(
    *,
    moving_triangles_midpoint_m: np.ndarray,
    moving_vertex_motion_half_extent_upper_m: np.ndarray,
    candidate_axes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project Cartesian vertex boxes onto axes with outward rounding."""

    moving = np.asarray(moving_triangles_midpoint_m, dtype=np.float64)
    half_extent = np.asarray(
        moving_vertex_motion_half_extent_upper_m, dtype=np.float64
    )
    axes = np.asarray(candidate_axes, dtype=np.float64)
    count = len(moving)
    if (
        moving.shape != (count, 3, 3)
        or count == 0
        or half_extent.shape != (count, 3, 3)
        or axes.ndim != 3
        or axes.shape[0] != count
        or axes.shape[2] != 3
        or axes.shape[1] == 0
        or not np.all(np.isfinite(moving))
        or not np.all(np.isfinite(half_extent))
        or not np.all(np.isfinite(axes))
        or np.any(half_extent < 0.0)
    ):
        raise ContinuousCollisionError(
            "moving projection inputs need aligned finite midpoint boxes "
            "and axes"
        )
    projection_center = np.einsum("nvc,nac->nva", moving, axes)
    projection_expansion = np.einsum(
        "nvc,nac->nva", half_extent, np.abs(axes)
    )
    minimum = np.nextafter(
        np.min(projection_center - projection_expansion, axis=1),
        -math.inf,
    )
    maximum = np.nextafter(
        np.max(projection_center + projection_expansion, axis=1),
        math.inf,
    )
    if not np.all(np.isfinite(minimum)) or not np.all(np.isfinite(maximum)):
        raise ContinuousCollisionError(
            "moving projection interval arithmetic overflowed"
        )
    return minimum, maximum


def _axis_strict_separation_mask(
    *,
    moving_triangles_midpoint_m: np.ndarray,
    moving_vertex_motion_half_extent_upper_m: np.ndarray,
    fixed_triangles_m: np.ndarray,
    candidate_axes: np.ndarray,
    moving_projection_minimum: np.ndarray | None = None,
    moving_projection_maximum: np.ndarray | None = None,
    fixed_projection_minimum: np.ndarray | None = None,
    fixed_projection_maximum: np.ndarray | None = None,
    candidate_axis_norm: np.ndarray | None = None,
    coordinate_scale_m: np.ndarray | None = None,
) -> np.ndarray:
    """Return pairs proved strictly separated by at least one given axis."""

    moving = np.asarray(moving_triangles_midpoint_m, dtype=np.float64)
    half_extent = np.asarray(
        moving_vertex_motion_half_extent_upper_m, dtype=np.float64
    )
    fixed = np.asarray(fixed_triangles_m, dtype=np.float64)
    axes = np.asarray(candidate_axes, dtype=np.float64)
    count = len(moving)
    if (
        moving.shape != (count, 3, 3)
        or count == 0
        or half_extent.shape != (count, 3, 3)
        or fixed.shape != (count, 3, 3)
        or axes.ndim != 3
        or axes.shape[0] != count
        or axes.shape[2] != 3
        or axes.shape[1] == 0
        or not np.all(np.isfinite(fixed))
    ):
        raise ContinuousCollisionError(
            "axis separation inputs need aligned finite triangle packets"
        )
    if (moving_projection_minimum is None) != (
        moving_projection_maximum is None
    ):
        raise ContinuousCollisionError(
            "moving projection bounds must be supplied together"
        )
    if moving_projection_minimum is None:
        moving_minimum, moving_maximum = (
            _moving_projection_interval_bounds(
                moving_triangles_midpoint_m=moving,
                moving_vertex_motion_half_extent_upper_m=half_extent,
                candidate_axes=axes,
            )
        )
    else:
        moving_minimum = np.asarray(
            moving_projection_minimum, dtype=np.float64
        )
        moving_maximum = np.asarray(
            moving_projection_maximum, dtype=np.float64
        )
        expected_shape = (count, axes.shape[1])
        if (
            moving_minimum.shape != expected_shape
            or moving_maximum.shape != expected_shape
            or not np.all(np.isfinite(moving_minimum))
            or not np.all(np.isfinite(moving_maximum))
            or np.any(moving_minimum > moving_maximum)
        ):
            raise ContinuousCollisionError(
                "precomputed moving projection bounds are invalid"
            )
    if (fixed_projection_minimum is None) != (
        fixed_projection_maximum is None
    ):
        raise ContinuousCollisionError(
            "fixed projection bounds must be supplied together"
        )
    if fixed_projection_minimum is None:
        fixed_projection = np.einsum("nvc,nac->nva", fixed, axes)
        fixed_minimum = np.min(fixed_projection, axis=1)
        fixed_maximum = np.max(fixed_projection, axis=1)
    else:
        fixed_minimum = np.asarray(
            fixed_projection_minimum, dtype=np.float64
        )
        fixed_maximum = np.asarray(
            fixed_projection_maximum, dtype=np.float64
        )
        expected_shape = (count, axes.shape[1])
        if (
            fixed_minimum.shape != expected_shape
            or fixed_maximum.shape != expected_shape
            or not np.all(np.isfinite(fixed_minimum))
            or not np.all(np.isfinite(fixed_maximum))
            or np.any(fixed_minimum > fixed_maximum)
        ):
            raise ContinuousCollisionError(
                "precomputed fixed projection bounds are invalid"
            )
    if candidate_axis_norm is None:
        axis_norm = np.linalg.norm(axes, axis=2)
    else:
        axis_norm = np.asarray(candidate_axis_norm, dtype=np.float64)
        if axis_norm.shape != (count, axes.shape[1]):
            raise ContinuousCollisionError(
                "precomputed candidate-axis norms are invalid"
            )
    if coordinate_scale_m is None:
        coordinate_scale = (
            np.max(np.abs(moving), axis=(1, 2))
            + np.max(half_extent, axis=(1, 2))
            + np.max(np.abs(fixed), axis=(1, 2))
            + 1.0
        )
    else:
        coordinate_scale = np.asarray(coordinate_scale_m, dtype=np.float64)
        if coordinate_scale.shape != (count,):
            raise ContinuousCollisionError(
                "precomputed coordinate scales are invalid"
            )
    projection_error = np.nextafter(
        _SAT_DOT_ERROR
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
    valid_axis = (
        np.isfinite(axis_norm)
        & (axis_norm > np.finfo(np.float64).tiny)
        & np.isfinite(projection_error)
    )
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


def _moving_triangle_triangle_strict_separation_mask(
    *,
    moving_triangles_midpoint_m: np.ndarray,
    moving_vertex_motion_half_extent_upper_m: np.ndarray,
    fixed_triangles_m: np.ndarray,
) -> np.ndarray:
    """All-axis anisotropic reference for the staged implementation.

    Each moving vertex is enclosed by its Cartesian interval box.  One strict
    separating axis is sufficient; no separating axis remains unresolved.
    """

    moving = np.asarray(moving_triangles_midpoint_m, dtype=np.float64)
    half_extent = np.asarray(
        moving_vertex_motion_half_extent_upper_m, dtype=np.float64
    )
    fixed = np.asarray(fixed_triangles_m, dtype=np.float64)
    count = len(moving)
    if (
        moving.shape != (count, 3, 3)
        or count == 0
        or half_extent.shape != (count, 3, 3)
        or fixed.shape != (count, 3, 3)
        or not np.all(np.isfinite(moving))
        or not np.all(np.isfinite(half_extent))
        or not np.all(np.isfinite(fixed))
        or np.any(half_extent < 0.0)
    ):
        raise ContinuousCollisionError(
            "moving triangle SAT inputs need aligned finite triangles and "
            "nonnegative vertex motion half extents"
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
    fixed_normals = np.cross(fixed_edges[:, 0], -fixed_edges[:, 2])
    edge_cross_axes = np.cross(
        moving_edges[:, :, None, :], fixed_edges[:, None, :, :]
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
    return _axis_strict_separation_mask(
        moving_triangles_midpoint_m=moving,
        moving_vertex_motion_half_extent_upper_m=half_extent,
        fixed_triangles_m=fixed,
        candidate_axes=candidate_axes,
    )


def _moving_triangle_packet_geometry(
    point_lower_m: np.ndarray,
    point_upper_m: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    lower = np.asarray(point_lower_m, dtype=np.float64)
    upper = np.asarray(point_upper_m, dtype=np.float64)
    if (
        lower.ndim != 3
        or lower.shape[1:] != (3, 3)
        or upper.shape != lower.shape
        or len(lower) == 0
        or not np.all(np.isfinite(lower))
        or not np.all(np.isfinite(upper))
        or np.any(lower > upper)
    ):
        raise ContinuousCollisionError(
            "moving triangle point bounds must be finite aligned (F,3,3)"
        )
    triangle_lower = np.min(lower, axis=1)
    triangle_upper = np.max(upper, axis=1)
    midpoint = 0.5 * lower + 0.5 * upper
    if not np.all(np.isfinite(midpoint)):
        raise ContinuousCollisionError(
            "moving triangle midpoint arithmetic overflowed"
        )
    half_extent = np.maximum(
        np.abs(midpoint - lower), np.abs(upper - midpoint)
    )
    half_extent = np.nextafter(half_extent, math.inf)
    moving_edges = np.stack(
        (
            midpoint[:, 1] - midpoint[:, 0],
            midpoint[:, 2] - midpoint[:, 1],
            midpoint[:, 0] - midpoint[:, 2],
        ),
        axis=1,
    )
    moving_normals = np.cross(moving_edges[:, 0], -moving_edges[:, 2])
    moving_in_plane_axes = np.cross(
        moving_edges, moving_normals[:, None, :]
    )
    moving_stage_axes = np.concatenate(
        (moving_normals[:, None, :], moving_in_plane_axes), axis=1
    )
    (
        moving_stage_projection_minimum,
        moving_stage_projection_maximum,
    ) = _moving_projection_interval_bounds(
        moving_triangles_midpoint_m=midpoint,
        moving_vertex_motion_half_extent_upper_m=half_extent,
        candidate_axes=moving_stage_axes,
    )
    moving_stage_axis_norm = np.linalg.norm(moving_stage_axes, axis=2)
    moving_coordinate_scale = (
        np.max(np.abs(midpoint), axis=(1, 2))
        + np.max(half_extent, axis=(1, 2))
    )
    if (
        not np.all(np.isfinite(half_extent))
        or not np.all(np.isfinite(moving_edges))
        or not np.all(np.isfinite(moving_stage_axes))
        or not np.all(np.isfinite(moving_stage_axis_norm))
        or not np.all(np.isfinite(moving_coordinate_scale))
    ):
        raise ContinuousCollisionError(
            "moving triangle anisotropic geometry arithmetic overflowed"
        )
    return (
        triangle_lower,
        triangle_upper,
        midpoint,
        half_extent,
        moving_edges,
        moving_stage_axes,
        moving_stage_projection_minimum,
        moving_stage_projection_maximum,
        moving_stage_axis_norm,
        moving_coordinate_scale,
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
        self.face_edges_m = np.stack(
            (
                triangles_m[:, 1] - triangles_m[:, 0],
                triangles_m[:, 2] - triangles_m[:, 1],
                triangles_m[:, 0] - triangles_m[:, 2],
            ),
            axis=1,
        )
        face_normals = np.cross(
            self.face_edges_m[:, 0], -self.face_edges_m[:, 2]
        )
        face_in_plane_axes = np.cross(
            self.face_edges_m, face_normals[:, None, :]
        )
        self.face_static_stage_axes = np.concatenate(
            (face_normals[:, None, :], face_in_plane_axes), axis=1
        )
        face_static_projection = np.einsum(
            "nvc,nac->nva",
            triangles_m,
            self.face_static_stage_axes,
        )
        self.face_static_projection_minimum = np.min(
            face_static_projection, axis=1
        )
        self.face_static_projection_maximum = np.max(
            face_static_projection, axis=1
        )
        self.face_static_axis_norm = np.linalg.norm(
            self.face_static_stage_axes, axis=2
        )
        self.face_coordinate_scale_m = np.max(
            np.abs(triangles_m), axis=(1, 2)
        )
        self.root = self._build(tuple(range(len(triangles_m))))
        for immutable in (
            self.face_lower_m,
            self.face_upper_m,
            self.face_centroid_m,
            self.face_edges_m,
            self.face_static_stage_axes,
            self.face_static_projection_minimum,
            self.face_static_projection_maximum,
            self.face_static_axis_norm,
            self.face_coordinate_scale_m,
        ):
            immutable.setflags(write=False)

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

    def classify_moving_face_bounds_packet(
        self,
        lower_m: np.ndarray,
        upper_m: np.ndarray,
        counters: _Counters,
    ) -> int:
        """Classify all moving-face AABBs in one shared BVH traversal."""

        lower = np.asarray(lower_m, dtype=np.float64)
        upper = np.asarray(upper_m, dtype=np.float64)
        if (
            lower.ndim != 2
            or lower.shape[1:] != (3,)
            or upper.shape != lower.shape
            or len(lower) == 0
            or not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper))
            or np.any(lower > upper)
        ):
            raise ContinuousCollisionError(
                "moving-face bound packet must be finite aligned (F, 3)"
            )
        overlap_pair_count = 0
        all_indices = np.arange(len(lower), dtype=np.int64)
        stack: list[tuple[_BVHNode, np.ndarray]] = [
            (self.root, all_indices)
        ]
        while stack:
            node, moving_indices = stack.pop()
            counters.bvh_node_visits += len(moving_indices)
            moving_lower = lower[moving_indices]
            moving_upper = upper[moving_indices]
            separated_from_node = np.any(
                (moving_upper < node.lower_m)
                | (node.upper_m < moving_lower),
                axis=1,
            )
            separated_moving_count = int(
                np.count_nonzero(separated_from_node)
            )
            if separated_moving_count:
                pruned_pairs = (
                    separated_moving_count * node.subtree_face_count
                )
                counters.pair_coverage += pruned_pairs
                counters.strictly_separated_pairs += pruned_pairs
            active = moving_indices[~separated_from_node]
            if len(active) == 0:
                continue
            if node.leaf:
                counters.bvh_leaf_visits += len(active)
                static_indices = np.asarray(
                    node.face_indices, dtype=np.int64
                )
                active_lower = lower[active, None, :]
                active_upper = upper[active, None, :]
                face_lower = self.face_lower_m[None, static_indices, :]
                face_upper = self.face_upper_m[None, static_indices, :]
                separated_pairs = np.any(
                    (active_upper < face_lower)
                    | (face_upper < active_lower),
                    axis=2,
                )
                pair_count = len(active) * len(static_indices)
                strictly_separated = int(
                    np.count_nonzero(separated_pairs)
                )
                overlaps = pair_count - strictly_separated
                counters.leaf_pair_evaluations += pair_count
                counters.pair_coverage += pair_count
                counters.strictly_separated_pairs += strictly_separated
                counters.potential_overlap_pairs += overlaps
                overlap_pair_count += overlaps
                continue
            if node.right is None or node.left is None:
                raise ContinuousCollisionError(
                    "static BVH internal node is incomplete"
                )
            stack.append((node.right, active))
            stack.append((node.left, active))
        return overlap_pair_count

    def _precomputed_axis_family_strict_separation_mask(
        self,
        *,
        moving_faces: np.ndarray,
        static_faces: np.ndarray,
        midpoint_m: np.ndarray,
        half_extent_upper_m: np.ndarray,
        moving_edges_m: np.ndarray,
        moving_stage_axes: np.ndarray,
        moving_stage_projection_minimum: np.ndarray,
        moving_stage_projection_maximum: np.ndarray,
        moving_stage_axis_norm: np.ndarray,
        moving_coordinate_scale_m: np.ndarray,
    ) -> np.ndarray:
        """Evaluate all axis families with reusable self-axis projections."""

        moving_indices = np.asarray(moving_faces, dtype=np.int64)
        static_indices = np.asarray(static_faces, dtype=np.int64)
        pair_count = len(moving_indices)
        if (
            pair_count == 0
            or static_indices.shape != (pair_count,)
            or np.any(moving_indices < 0)
            or np.any(moving_indices >= len(midpoint_m))
            or np.any(static_indices < 0)
            or np.any(static_indices >= len(self.triangles_m))
        ):
            raise ContinuousCollisionError(
                "axis-family narrowphase needs non-empty aligned face pairs"
            )
        moving_midpoint = midpoint_m[moving_indices]
        moving_half_extent = half_extent_upper_m[moving_indices]
        fixed_triangles = self.triangles_m[static_indices]
        moving_axes = moving_stage_axes[moving_indices]
        static_axes = self.face_static_stage_axes[static_indices]
        coordinate_scale = (
            moving_coordinate_scale_m[moving_indices]
            + self.face_coordinate_scale_m[static_indices]
            + 1.0
        )
        moving_axis_separated = _axis_strict_separation_mask(
            moving_triangles_midpoint_m=moving_midpoint,
            moving_vertex_motion_half_extent_upper_m=moving_half_extent,
            fixed_triangles_m=fixed_triangles,
            candidate_axes=moving_axes,
            moving_projection_minimum=(
                moving_stage_projection_minimum[moving_indices]
            ),
            moving_projection_maximum=(
                moving_stage_projection_maximum[moving_indices]
            ),
            candidate_axis_norm=moving_stage_axis_norm[moving_indices],
            coordinate_scale_m=coordinate_scale,
        )
        static_axis_separated = _axis_strict_separation_mask(
            moving_triangles_midpoint_m=moving_midpoint,
            moving_vertex_motion_half_extent_upper_m=moving_half_extent,
            fixed_triangles_m=fixed_triangles,
            candidate_axes=static_axes,
            fixed_projection_minimum=(
                self.face_static_projection_minimum[static_indices]
            ),
            fixed_projection_maximum=(
                self.face_static_projection_maximum[static_indices]
            ),
            candidate_axis_norm=self.face_static_axis_norm[static_indices],
            coordinate_scale_m=coordinate_scale,
        )
        edge_cross_axes = np.cross(
            moving_edges_m[moving_indices, :, None, :],
            self.face_edges_m[static_indices, None, :, :],
        ).reshape((pair_count, 9, 3))
        edge_axis_separated = _axis_strict_separation_mask(
            moving_triangles_midpoint_m=moving_midpoint,
            moving_vertex_motion_half_extent_upper_m=moving_half_extent,
            fixed_triangles_m=fixed_triangles,
            candidate_axes=edge_cross_axes,
            coordinate_scale_m=coordinate_scale,
        )
        result = (
            moving_axis_separated
            | static_axis_separated
            | edge_axis_separated
        )
        result.setflags(write=False)
        return result

    def classify_moving_triangles_packet(
        self,
        point_lower_m: np.ndarray,
        point_upper_m: np.ndarray,
        counters: _Counters,
    ) -> _UnresolvedTrianglePairFrontier:
        """Classify packet pairs with BVH AABBs then swept-triangle SAT."""

        lower = np.asarray(point_lower_m, dtype=np.float64)
        (
            triangle_lower,
            triangle_upper,
            midpoint,
            half_extent,
            moving_edges,
            moving_stage_axes,
            moving_stage_projection_minimum,
            moving_stage_projection_maximum,
            moving_stage_axis_norm,
            moving_coordinate_scale,
        ) = _moving_triangle_packet_geometry(point_lower_m, point_upper_m)

        counters.root_bvh_intervals += 1
        aabb_overlap_pair_ids: list[np.ndarray] = []
        all_indices = np.arange(len(lower), dtype=np.int64)
        stack: list[tuple[_BVHNode, np.ndarray]] = [
            (self.root, all_indices)
        ]
        while stack:
            node, moving_indices = stack.pop()
            counters.bvh_node_visits += len(moving_indices)
            moving_lower = triangle_lower[moving_indices]
            moving_upper = triangle_upper[moving_indices]
            separated_from_node = np.any(
                (moving_upper < node.lower_m)
                | (node.upper_m < moving_lower),
                axis=1,
            )
            separated_moving_count = int(
                np.count_nonzero(separated_from_node)
            )
            if separated_moving_count:
                pruned_pairs = (
                    separated_moving_count * node.subtree_face_count
                )
                counters.pair_coverage += pruned_pairs
                counters.strictly_separated_pairs += pruned_pairs
            active = moving_indices[~separated_from_node]
            if len(active) == 0:
                continue
            if node.leaf:
                counters.bvh_leaf_visits += len(active)
                static_indices = np.asarray(
                    node.face_indices, dtype=np.int64
                )
                active_lower = triangle_lower[active, None, :]
                active_upper = triangle_upper[active, None, :]
                face_lower = self.face_lower_m[None, static_indices, :]
                face_upper = self.face_upper_m[None, static_indices, :]
                aabb_separated = np.any(
                    (active_upper < face_lower)
                    | (face_upper < active_lower),
                    axis=2,
                )
                pair_count = len(active) * len(static_indices)
                aabb_separated_count = int(
                    np.count_nonzero(aabb_separated)
                )
                overlap_offsets = np.nonzero(~aabb_separated)
                overlap_count = len(overlap_offsets[0])
                if overlap_count:
                    moving_faces = active[
                        overlap_offsets[0]
                    ]
                    static_faces = static_indices[
                        overlap_offsets[1]
                    ]
                    aabb_overlap_pair_ids.append(
                        moving_faces * len(self.triangles_m)
                        + static_faces
                    )
                counters.leaf_pair_evaluations += pair_count
                counters.pair_coverage += pair_count
                counters.strictly_separated_pairs += aabb_separated_count
                continue
            if node.right is None or node.left is None:
                raise ContinuousCollisionError(
                    "static BVH internal node is incomplete"
                )
            stack.append((node.right, active))
            stack.append((node.left, active))
        aabb_overlap_ids = (
            np.concatenate(aabb_overlap_pair_ids)
            if aabb_overlap_pair_ids
            else np.empty(0, dtype=np.int64)
        )
        aabb_overlap_count = len(aabb_overlap_ids)
        sat_separated_count = 0
        unresolved_pair_ids: list[np.ndarray] = []
        if aabb_overlap_count:
            counters.narrowphase_invocation_intervals += 1
            for start_index in range(
                0,
                aabb_overlap_count,
                _NARROWPHASE_PAIR_PACKET_SIZE,
            ):
                stop_index = min(
                    aabb_overlap_count,
                    start_index + _NARROWPHASE_PAIR_PACKET_SIZE,
                )
                overlap_pair_ids = aabb_overlap_ids[
                    start_index:stop_index
                ]
                overlap_moving_faces = (
                    overlap_pair_ids // len(self.triangles_m)
                )
                overlap_static_faces = (
                    overlap_pair_ids % len(self.triangles_m)
                )
                separated_by_sat = self._precomputed_axis_family_strict_separation_mask(
                    moving_faces=overlap_moving_faces,
                    static_faces=overlap_static_faces,
                    midpoint_m=midpoint,
                    half_extent_upper_m=half_extent,
                    moving_edges_m=moving_edges,
                    moving_stage_axes=moving_stage_axes,
                    moving_stage_projection_minimum=(
                        moving_stage_projection_minimum
                    ),
                    moving_stage_projection_maximum=(
                        moving_stage_projection_maximum
                    ),
                    moving_stage_axis_norm=moving_stage_axis_norm,
                    moving_coordinate_scale_m=moving_coordinate_scale,
                )
                counters.narrowphase_pair_packets += 1
                counters.narrowphase_root_pair_packets += 1
                sat_separated_count += int(
                    np.count_nonzero(separated_by_sat)
                )
                if np.any(~separated_by_sat):
                    unresolved_pair_ids.append(
                        overlap_pair_ids[~separated_by_sat]
                    )
        pair_ids = (
            np.concatenate(unresolved_pair_ids)
            if unresolved_pair_ids
            else np.empty(0, dtype=np.int64)
        )
        counters.narrowphase_pair_evaluations += aabb_overlap_count
        counters.narrowphase_strictly_separated_pairs += (
            sat_separated_count
        )
        counters.strictly_separated_pairs += sat_separated_count
        counters.potential_overlap_pairs += len(pair_ids)
        return _unresolved_triangle_pair_frontier(
            pair_ids,
            moving_triangle_count=len(lower),
            static_triangle_count=len(self.triangles_m),
        )

    def classify_parent_pair_frontier_packet(
        self,
        point_lower_m: np.ndarray,
        point_upper_m: np.ndarray,
        parent_frontier: _UnresolvedTrianglePairFrontier,
        counters: _Counters,
    ) -> _UnresolvedTrianglePairFrontier:
        """AABB-filter a parent frontier and eagerly run full narrowphase."""

        lower = np.asarray(point_lower_m, dtype=np.float64)
        (
            triangle_lower,
            triangle_upper,
            midpoint,
            half_extent,
            moving_edges,
            moving_stage_axes,
            moving_stage_projection_minimum,
            moving_stage_projection_maximum,
            moving_stage_axis_norm,
            moving_coordinate_scale,
        ) = _moving_triangle_packet_geometry(point_lower_m, point_upper_m)
        if (
            parent_frontier.moving_triangle_count != len(lower)
            or parent_frontier.static_triangle_count
            != len(self.triangles_m)
        ):
            raise ContinuousCollisionError(
                "parent unresolved pair frontier geometry mismatch"
            )
        complete_pair_count = len(lower) * len(self.triangles_m)
        inherited_count = complete_pair_count - parent_frontier.pair_count
        counters.inherited_strictly_separated_pairs += inherited_count
        counters.pair_coverage += inherited_count
        counters.strictly_separated_pairs += inherited_count
        counters.frontier_pair_evaluations += parent_frontier.pair_count
        counters.leaf_pair_evaluations += parent_frontier.pair_count

        aabb_overlap_pair_ids: list[np.ndarray] = []
        aabb_separated_count = 0
        for start_index in range(
            0,
            parent_frontier.pair_count,
            _NARROWPHASE_PAIR_PACKET_SIZE,
        ):
            stop_index = min(
                parent_frontier.pair_count,
                start_index + _NARROWPHASE_PAIR_PACKET_SIZE,
            )
            pair_ids = parent_frontier.pair_ids[start_index:stop_index]
            moving_faces = pair_ids // len(self.triangles_m)
            static_faces = pair_ids % len(self.triangles_m)
            separated_by_aabb = np.any(
                (
                    triangle_upper[moving_faces]
                    < self.face_lower_m[static_faces]
                )
                | (
                    self.face_upper_m[static_faces]
                    < triangle_lower[moving_faces]
                ),
                axis=1,
            )
            aabb_separated_count += int(np.count_nonzero(separated_by_aabb))
            overlap_pair_ids = pair_ids[~separated_by_aabb]
            if len(overlap_pair_ids):
                aabb_overlap_pair_ids.append(overlap_pair_ids)

        aabb_overlap_ids = (
            np.concatenate(aabb_overlap_pair_ids)
            if aabb_overlap_pair_ids
            else np.empty(0, dtype=np.int64)
        )
        aabb_overlap_count = len(aabb_overlap_ids)
        sat_separated_count = 0
        unresolved_pair_ids: list[np.ndarray] = []
        if aabb_overlap_count:
            counters.narrowphase_invocation_intervals += 1
            counters.narrowphase_eager_child_intervals += 1
            for start_index in range(
                0,
                aabb_overlap_count,
                _NARROWPHASE_PAIR_PACKET_SIZE,
            ):
                stop_index = min(
                    aabb_overlap_count,
                    start_index + _NARROWPHASE_PAIR_PACKET_SIZE,
                )
                overlap_pair_ids = aabb_overlap_ids[
                    start_index:stop_index
                ]
                overlap_moving_faces = (
                    overlap_pair_ids // len(self.triangles_m)
                )
                overlap_static_faces = (
                    overlap_pair_ids % len(self.triangles_m)
                )
                separated_by_sat = self._precomputed_axis_family_strict_separation_mask(
                    moving_faces=overlap_moving_faces,
                    static_faces=overlap_static_faces,
                    midpoint_m=midpoint,
                    half_extent_upper_m=half_extent,
                    moving_edges_m=moving_edges,
                    moving_stage_axes=moving_stage_axes,
                    moving_stage_projection_minimum=(
                        moving_stage_projection_minimum
                    ),
                    moving_stage_projection_maximum=(
                        moving_stage_projection_maximum
                    ),
                    moving_stage_axis_norm=moving_stage_axis_norm,
                    moving_coordinate_scale_m=moving_coordinate_scale,
                )
                counters.narrowphase_pair_packets += 1
                counters.narrowphase_child_pair_packets += 1
                sat_separated_count += int(
                    np.count_nonzero(separated_by_sat)
                )
                if np.any(~separated_by_sat):
                    unresolved_pair_ids.append(
                        overlap_pair_ids[~separated_by_sat]
                    )
            pair_ids = (
                np.concatenate(unresolved_pair_ids)
                if unresolved_pair_ids
                else np.empty(0, dtype=np.int64)
            )
            counters.narrowphase_pair_evaluations += aabb_overlap_count
            counters.narrowphase_strictly_separated_pairs += (
                sat_separated_count
            )
        else:
            pair_ids = np.empty(0, dtype=np.int64)

        unresolved_count = len(pair_ids)
        counters.pair_coverage += parent_frontier.pair_count
        counters.strictly_separated_pairs += (
            aabb_separated_count + sat_separated_count
        )
        counters.potential_overlap_pairs += unresolved_count
        return _unresolved_triangle_pair_frontier(
            pair_ids,
            moving_triangle_count=len(lower),
            static_triangle_count=len(self.triangles_m),
        )


_PREPARED_STATIC_SURFACE_TOKEN = object()


@dataclass(frozen=True)
class PreparedStaticTriangleSurface:
    """One immutable canonical surface and its reusable static BVH."""

    method_id: str
    geometry_sha256: str
    triangle_count: int
    triangles_object_m: np.ndarray
    _bvh: _StaticTriangleBVH
    _construction_token: object

    def __post_init__(self) -> None:
        triangles = np.asarray(self.triangles_object_m, dtype=np.float64)
        if (
            self._construction_token is not _PREPARED_STATIC_SURFACE_TOKEN
            or self.method_id != PREPARED_STATIC_SURFACE_METHOD_ID
            or len(self.geometry_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.geometry_sha256
            )
            or not isinstance(self.triangle_count, int)
            or isinstance(self.triangle_count, bool)
            or self.triangle_count <= 0
            or triangles.shape != (self.triangle_count, 3, 3)
            or triangles.flags.writeable
            or type(self._bvh) is not _StaticTriangleBVH
            or self._bvh.triangles_m is not self.triangles_object_m
        ):
            raise ContinuousCollisionError(
                "prepared static triangle surface is malformed"
            )


def prepare_static_triangle_surface(
    triangles_object_m: (
        Sequence[Sequence[Sequence[float]]] | np.ndarray
    ),
    *,
    expected_geometry_sha256: str | None = None,
) -> PreparedStaticTriangleSurface:
    """Canonicalize and index one static surface for repeated queries."""

    triangles, geometry_sha256 = _canonical_surface(
        triangles_object_m,
        label="prepared static object surface",
    )
    if expected_geometry_sha256 is not None and (
        len(str(expected_geometry_sha256)) != 64
        or any(
            character not in "0123456789abcdef"
            for character in str(expected_geometry_sha256)
        )
        or str(expected_geometry_sha256) != geometry_sha256
    ):
        raise ContinuousCollisionError(
            "prepared static object surface geometry hash mismatch"
        )
    # bytes owns the storage, so callers cannot turn write access back on.
    immutable = np.frombuffer(
        np.asarray(triangles, dtype="<f8").tobytes(order="C"),
        dtype="<f8",
    ).reshape(triangles.shape)
    bvh = _StaticTriangleBVH(immutable)
    return PreparedStaticTriangleSurface(
        method_id=PREPARED_STATIC_SURFACE_METHOD_ID,
        geometry_sha256=geometry_sha256,
        triangle_count=len(immutable),
        triangles_object_m=immutable,
        _bvh=bvh,
        _construction_token=_PREPARED_STATIC_SURFACE_TOKEN,
    )


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
        root_bvh_interval_count=counters.root_bvh_intervals,
        inherited_strictly_separated_pair_count=(
            counters.inherited_strictly_separated_pairs
        ),
        frontier_pair_evaluation_count=counters.frontier_pair_evaluations,
        bvh_node_visit_count=counters.bvh_node_visits,
        bvh_leaf_visit_count=counters.bvh_leaf_visits,
        leaf_pair_evaluation_count=counters.leaf_pair_evaluations,
        narrowphase_pair_evaluation_count=(
            counters.narrowphase_pair_evaluations
        ),
        narrowphase_strictly_separated_pair_count=(
            counters.narrowphase_strictly_separated_pairs
        ),
        narrowphase_invocation_interval_count=(
            counters.narrowphase_invocation_intervals
        ),
        narrowphase_eager_child_interval_count=(
            counters.narrowphase_eager_child_intervals
        ),
        narrowphase_pair_packet_count=counters.narrowphase_pair_packets,
        narrowphase_root_pair_packet_count=(
            counters.narrowphase_root_pair_packets
        ),
        narrowphase_child_pair_packet_count=(
            counters.narrowphase_child_pair_packets
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
        Sequence[Sequence[Sequence[float]]]
        | np.ndarray
        | PreparedStaticTriangleSurface
    ),
    maximum_subdivision_intervals: int,
) -> ContinuousCollisionCertificate:
    """Certify strict moving/static surface separation on ``phase``.

    At each phase, directed point-motion intervals enclose all moving triangle
    vertices.  Packet BVH AABBs reject broadphase pairs.  Full anisotropic
    triangle-axis checks run at the root, after an amortized frontier halving,
    and before a terminal return; deferred pairs remain unresolved.
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
    if type(static_triangles_object_m) is PreparedStaticTriangleSurface:
        prepared_static = static_triangles_object_m
        static = prepared_static.triangles_object_m
        static_hash = prepared_static.geometry_sha256
        static_bvh = prepared_static._bvh
    else:
        static, static_hash = _canonical_surface(
            static_triangles_object_m,
            label="static object surface",
        )
        static_bvh = _StaticTriangleBVH(static)
    moving_points = moving.reshape((-1, 3))
    counters = _Counters()
    free_leaves: list[IntervalBounds] = []
    pending: list[
        tuple[IntervalBounds, _UnresolvedTrianglePairFrontier | None]
    ] = [(phase, None)]

    while pending:
        interval, parent_frontier = pending.pop()
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
        subdivision_midpoint = interval.lower + 0.5 * (
            interval.upper - interval.lower
        )
        cannot_subdivide = not (
            interval.lower < subdivision_midpoint < interval.upper
        )
        interval_overlap_pairs = 0
        interval_frontier: _UnresolvedTrianglePairFrontier | None = None
        backend_failure_reason: str | None = None
        interval_pair_universe = len(moving) * len(static)
        counters.pair_universe += interval_pair_universe

        try:
            motion = backend.point_motion_many(
                link_name=str(link_name),
                q_start=start,
                direction=path_direction,
                phase_lower=interval.lower,
                phase_upper=interval.upper,
                base_transform=base,
                points_local_m=moving_points,
            )
        except IntervalKinematicsError as error:
            backend_failure_reason = (
                "INTERVAL_KINEMATICS_UNRESOLVED:"
                + type(error).__name__
            )
        else:
            counters.point_motion_evaluations += len(moving_points)
            if motion.method_id != BATCH_POINT_MOTION_METHOD_ID:
                backend_failure_reason = (
                    "INTERVAL_KINEMATICS_METHOD_MISMATCH"
                )
            else:
                point_lower = np.asarray(
                    motion.position_lower_object_m,
                    dtype=np.float64,
                ).reshape((len(moving), 3, 3))
                point_upper = np.asarray(
                    motion.position_upper_object_m,
                    dtype=np.float64,
                ).reshape((len(moving), 3, 3))
        if backend_failure_reason is None:
            if parent_frontier is None:
                interval_frontier = (
                    static_bvh.classify_moving_triangles_packet(
                        point_lower,
                        point_upper,
                        counters,
                    )
                )
            else:
                interval_frontier = (
                    static_bvh.classify_parent_pair_frontier_packet(
                        point_lower,
                        point_upper,
                        parent_frontier,
                        counters,
                    )
                )
            interval_overlap_pairs = interval_frontier.pair_count

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
        if cannot_subdivide:
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
        if interval_frontier is None:
            raise ContinuousCollisionError(
                "unresolved interval lost its exact pair frontier"
            )
        # Stack is LIFO: pushing right then left makes traversal left-first.
        pending.append(
            (
                IntervalBounds(subdivision_midpoint, interval.upper),
                interval_frontier,
            )
        )
        pending.append(
            (
                IntervalBounds(interval.lower, subdivision_midpoint),
                interval_frontier,
            )
        )

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


def _independent_moving_pair_contract_sha256(
    *,
    first_link_name: str,
    first_surface_hash: str,
    first_motion_hash: str,
    second_link_name: str,
    second_surface_hash: str,
    second_motion_hash: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"CARTS_INDEPENDENT_TWO_PHASE_SURFACE_PAIR_CONTRACT_V1\0")
    for value in (
        first_link_name,
        first_surface_hash,
        first_motion_hash,
        second_link_name,
        second_surface_hash,
        second_motion_hash,
    ):
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
    points = triangles_link_m.reshape((-1, 3))
    try:
        motion = backend.point_motion_many(
            link_name=link_name,
            q_start=q_start,
            direction=direction,
            phase_lower=interval.lower,
            phase_upper=interval.upper,
            base_transform=base_transform,
            points_local_m=points,
        )
    except IntervalKinematicsError as error:
        raise _MovingPairBackendUnresolved(
            "INTERVAL_KINEMATICS_UNRESOLVED:"
            + type(error).__name__
        ) from error
    counters.point_motion_evaluations += len(points)
    if motion.method_id != BATCH_POINT_MOTION_METHOD_ID:
        raise _MovingPairBackendUnresolved(
            "INTERVAL_KINEMATICS_METHOD_MISMATCH"
        )
    shape = (len(triangles_link_m), 3, 3)
    return (
        np.asarray(
            motion.position_lower_object_m, dtype=np.float64
        ).reshape(shape),
        np.asarray(
            motion.position_upper_object_m, dtype=np.float64
        ).reshape(shape),
    )


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


def _independent_moving_pair_certificate(
    *,
    state: ContinuousCollisionState,
    searched_phase_box: IndependentMotionPhaseBox,
    free_leaves: list[IndependentMotionPhaseBox],
    unresolved_phase_box: IndependentMotionPhaseBox | None,
    first_link_name: str,
    second_link_name: str,
    first_surface_hash: str,
    second_surface_hash: str,
    pair_contract_hash: str,
    first_triangle_count: int,
    second_triangle_count: int,
    maximum_phase_boxes: int,
    counters: _MovingPairCounters,
    terminal_unresolved_pairs: int,
    entire_phase_product_covered: bool,
    unresolved_reason: str,
) -> IndependentMovingSurfacePairCollisionCertificate:
    audit = IndependentMovingSurfacePairCollisionAudit(
        method_id=INDEPENDENT_MOVING_PAIR_METHOD_ID,
        interval_kinematics_method_id=INTERVAL_KINEMATICS_METHOD_ID,
        first_link_name=first_link_name,
        second_link_name=second_link_name,
        first_surface_geometry_sha256=first_surface_hash,
        second_surface_geometry_sha256=second_surface_hash,
        pair_contract_sha256=pair_contract_hash,
        first_triangle_count=first_triangle_count,
        second_triangle_count=second_triangle_count,
        pair_count_per_phase_box=(
            first_triangle_count * second_triangle_count
        ),
        maximum_subdivision_phase_boxes=maximum_phase_boxes,
        processed_phase_box_count=counters.processed_intervals,
        certified_free_leaf_phase_box_count=(
            counters.free_leaf_intervals
        ),
        subdivided_phase_box_count=counters.subdivisions,
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
        entire_phase_product_covered=entire_phase_product_covered,
        unresolved_reason=unresolved_reason,
        claim_limitations=INDEPENDENT_MOVING_PAIR_CLAIM_LIMITATIONS,
    )
    return IndependentMovingSurfacePairCollisionCertificate(
        state=state,
        searched_phase_box=searched_phase_box,
        certified_free_leaf_phase_boxes=tuple(free_leaves),
        unresolved_phase_box=unresolved_phase_box,
        audit=audit,
    )


def certify_independent_link_motion_surfaces_separated_from_each_other(
    *,
    backend: DirectedIntervalKinematics,
    first_link_name: str,
    second_link_name: str,
    first_q_start: Sequence[float] | np.ndarray,
    first_direction: Sequence[float] | np.ndarray,
    first_phase: IntervalBounds,
    second_q_start: Sequence[float] | np.ndarray,
    second_direction: Sequence[float] | np.ndarray,
    second_phase: IntervalBounds,
    object_from_hand_base: Sequence[Sequence[float]] | np.ndarray,
    first_triangles_link_m: (
        Sequence[Sequence[Sequence[float]]] | np.ndarray
    ),
    second_triangles_link_m: (
        Sequence[Sequence[Sequence[float]]] | np.ndarray
    ),
    maximum_subdivision_phase_boxes: int,
) -> IndependentMovingSurfacePairCollisionCertificate:
    """Certify separation on the Cartesian product of two link paths.

    The two scalar phases vary independently.  Every processed node encloses
    the complete rectangular phase product, never only its diagonal or a
    finite set of samples.  Strict relative-axis separation is sufficient for
    a free leaf; every other leaf is bisected or returned unresolved.
    """

    if not isinstance(backend, DirectedIntervalKinematics):
        raise ContinuousCollisionError(
            "independent moving-pair collision needs interval kinematics"
        )
    first_link = str(first_link_name)
    second_link = str(second_link_name)
    if not first_link or not second_link or first_link == second_link:
        raise ContinuousCollisionError(
            "independent moving-pair collision needs distinct links"
        )
    if not isinstance(first_phase, IntervalBounds) or not isinstance(
        second_phase, IntervalBounds
    ):
        raise ContinuousCollisionError(
            "independent moving-pair phases must be explicit intervals"
        )
    if (
        not isinstance(maximum_subdivision_phase_boxes, int)
        or isinstance(maximum_subdivision_phase_boxes, bool)
        or maximum_subdivision_phase_boxes <= 0
    ):
        raise ContinuousCollisionError(
            "maximum_subdivision_phase_boxes must be positive"
        )
    available_links = {backend.hand_model.base_link}
    for joint in backend.hand_model.joints.values():
        available_links.add(joint.parent_link)
        available_links.add(joint.child_link)
    if first_link not in available_links or second_link not in available_links:
        raise ContinuousCollisionError(
            "independent moving-pair link is outside the hand model"
        )
    joint_count = len(backend.hand_model.independent_joint_names)
    first_start = np.asarray(first_q_start, dtype=np.float64)
    first_path_direction = np.asarray(first_direction, dtype=np.float64)
    second_start = np.asarray(second_q_start, dtype=np.float64)
    second_path_direction = np.asarray(second_direction, dtype=np.float64)
    if any(
        value.shape != (joint_count,) or not np.all(np.isfinite(value))
        for value in (
            first_start,
            first_path_direction,
            second_start,
            second_path_direction,
        )
    ):
        raise ContinuousCollisionError(
            "independent moving-pair paths must match joint coordinates"
        )
    base = _proper_se3(object_from_hand_base)
    first_surface, first_hash = _canonical_surface(
        first_triangles_link_m,
        label=f"independent moving link surface {first_link}",
    )
    second_surface, second_hash = _canonical_surface(
        second_triangles_link_m,
        label=f"independent moving link surface {second_link}",
    )
    first_motion_hash = _motion_contract_sha256(
        backend=backend,
        link_name=first_link,
        q_start=first_start,
        direction=first_path_direction,
        phase=first_phase,
        base_transform=base,
    )
    second_motion_hash = _motion_contract_sha256(
        backend=backend,
        link_name=second_link,
        q_start=second_start,
        direction=second_path_direction,
        phase=second_phase,
        base_transform=base,
    )
    pair_contract_hash = _independent_moving_pair_contract_sha256(
        first_link_name=first_link,
        first_surface_hash=first_hash,
        first_motion_hash=first_motion_hash,
        second_link_name=second_link,
        second_surface_hash=second_hash,
        second_motion_hash=second_motion_hash,
    )
    searched_box = IndependentMotionPhaseBox(first_phase, second_phase)
    counters = _MovingPairCounters()
    free_leaves: list[IndependentMotionPhaseBox] = []
    pending: list[IndependentMotionPhaseBox] = [searched_box]
    pair_count = len(first_surface) * len(second_surface)
    first_total_width = first_phase.upper - first_phase.lower
    second_total_width = second_phase.upper - second_phase.lower

    while pending:
        phase_box = pending.pop()
        if counters.processed_intervals >= maximum_subdivision_phase_boxes:
            return _independent_moving_pair_certificate(
                state=ContinuousCollisionState.UNRESOLVED,
                searched_phase_box=searched_box,
                free_leaves=free_leaves,
                unresolved_phase_box=phase_box,
                first_link_name=first_link,
                second_link_name=second_link,
                first_surface_hash=first_hash,
                second_surface_hash=second_hash,
                pair_contract_hash=pair_contract_hash,
                first_triangle_count=len(first_surface),
                second_triangle_count=len(second_surface),
                maximum_phase_boxes=maximum_subdivision_phase_boxes,
                counters=counters,
                terminal_unresolved_pairs=0,
                entire_phase_product_covered=False,
                unresolved_reason="SUBDIVISION_PHASE_BOX_BUDGET_EXHAUSTED",
            )
        counters.processed_intervals += 1
        counters.pair_universe += pair_count
        try:
            first_lower, first_upper = _enclose_moving_surface_vertices(
                backend=backend,
                link_name=first_link,
                triangles_link_m=first_surface,
                q_start=first_start,
                direction=first_path_direction,
                interval=phase_box.first_phase,
                base_transform=base,
                counters=counters,
            )
            second_lower, second_upper = _enclose_moving_surface_vertices(
                backend=backend,
                link_name=second_link,
                triangles_link_m=second_surface,
                q_start=second_start,
                direction=second_path_direction,
                interval=phase_box.second_phase,
                base_transform=base,
                counters=counters,
            )
        except _MovingPairBackendUnresolved as error:
            return _independent_moving_pair_certificate(
                state=ContinuousCollisionState.UNRESOLVED,
                searched_phase_box=searched_box,
                free_leaves=free_leaves,
                unresolved_phase_box=phase_box,
                first_link_name=first_link,
                second_link_name=second_link,
                first_surface_hash=first_hash,
                second_surface_hash=second_hash,
                pair_contract_hash=pair_contract_hash,
                first_triangle_count=len(first_surface),
                second_triangle_count=len(second_surface),
                maximum_phase_boxes=maximum_subdivision_phase_boxes,
                counters=counters,
                terminal_unresolved_pairs=0,
                entire_phase_product_covered=False,
                unresolved_reason=error.reason,
            )

        overlap_pairs = 0
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
                    overlap_pairs += 1
        if overlap_pairs == 0:
            counters.free_leaf_intervals += 1
            free_leaves.append(phase_box)
            continue
        if counters.processed_intervals >= maximum_subdivision_phase_boxes:
            return _independent_moving_pair_certificate(
                state=ContinuousCollisionState.UNRESOLVED,
                searched_phase_box=searched_box,
                free_leaves=free_leaves,
                unresolved_phase_box=phase_box,
                first_link_name=first_link,
                second_link_name=second_link,
                first_surface_hash=first_hash,
                second_surface_hash=second_hash,
                pair_contract_hash=pair_contract_hash,
                first_triangle_count=len(first_surface),
                second_triangle_count=len(second_surface),
                maximum_phase_boxes=maximum_subdivision_phase_boxes,
                counters=counters,
                terminal_unresolved_pairs=overlap_pairs,
                entire_phase_product_covered=False,
                unresolved_reason="SUBDIVISION_PHASE_BOX_BUDGET_EXHAUSTED",
            )

        first_width = (
            phase_box.first_phase.upper - phase_box.first_phase.lower
        )
        second_width = (
            phase_box.second_phase.upper - phase_box.second_phase.lower
        )
        first_relative_width = (
            first_width / first_total_width
            if first_total_width > 0.0
            else 0.0
        )
        second_relative_width = (
            second_width / second_total_width
            if second_total_width > 0.0
            else 0.0
        )
        split_first = first_relative_width >= second_relative_width
        selected = (
            phase_box.first_phase if split_first else phase_box.second_phase
        )
        midpoint = selected.lower + 0.5 * (
            selected.upper - selected.lower
        )
        if not selected.lower < midpoint < selected.upper:
            return _independent_moving_pair_certificate(
                state=ContinuousCollisionState.UNRESOLVED,
                searched_phase_box=searched_box,
                free_leaves=free_leaves,
                unresolved_phase_box=phase_box,
                first_link_name=first_link,
                second_link_name=second_link,
                first_surface_hash=first_hash,
                second_surface_hash=second_hash,
                pair_contract_hash=pair_contract_hash,
                first_triangle_count=len(first_surface),
                second_triangle_count=len(second_surface),
                maximum_phase_boxes=maximum_subdivision_phase_boxes,
                counters=counters,
                terminal_unresolved_pairs=overlap_pairs,
                entire_phase_product_covered=False,
                unresolved_reason="ADJACENT_BINARY64_PHASE_BOX_ENDPOINTS",
            )
        lower_half = IntervalBounds(selected.lower, midpoint)
        upper_half = IntervalBounds(midpoint, selected.upper)
        counters.subdivisions += 1
        if split_first:
            pending.append(
                IndependentMotionPhaseBox(
                    upper_half, phase_box.second_phase
                )
            )
            pending.append(
                IndependentMotionPhaseBox(
                    lower_half, phase_box.second_phase
                )
            )
        else:
            pending.append(
                IndependentMotionPhaseBox(
                    phase_box.first_phase, upper_half
                )
            )
            pending.append(
                IndependentMotionPhaseBox(
                    phase_box.first_phase, lower_half
                )
            )

    return _independent_moving_pair_certificate(
        state=ContinuousCollisionState.CERTIFIED_FREE,
        searched_phase_box=searched_box,
        free_leaves=free_leaves,
        unresolved_phase_box=None,
        first_link_name=first_link,
        second_link_name=second_link,
        first_surface_hash=first_hash,
        second_surface_hash=second_hash,
        pair_contract_hash=pair_contract_hash,
        first_triangle_count=len(first_surface),
        second_triangle_count=len(second_surface),
        maximum_phase_boxes=maximum_subdivision_phase_boxes,
        counters=counters,
        terminal_unresolved_pairs=0,
        entire_phase_product_covered=True,
        unresolved_reason="NONE",
    )


__all__ = [
    "CLAIM_LIMITATIONS",
    "ContinuousCollisionAudit",
    "ContinuousCollisionCertificate",
    "ContinuousCollisionError",
    "ContinuousCollisionState",
    "INDEPENDENT_MOVING_PAIR_CLAIM_LIMITATIONS",
    "INDEPENDENT_MOVING_PAIR_METHOD_ID",
    "IndependentMotionPhaseBox",
    "IndependentMovingSurfacePairCollisionAudit",
    "IndependentMovingSurfacePairCollisionCertificate",
    "METHOD_ID",
    "MOVING_PAIR_CLAIM_LIMITATIONS",
    "MOVING_PAIR_METHOD_ID",
    "MovingSurfacePairCollisionAudit",
    "MovingSurfacePairCollisionCertificate",
    "PREPARED_STATIC_SURFACE_METHOD_ID",
    "PreparedStaticTriangleSurface",
    "certify_independent_link_motion_surfaces_separated_from_each_other",
    "certify_moving_link_surfaces_separated_from_each_other",
    "certify_moving_link_surface_separated_from_static_surface",
    "prepare_static_triangle_surface",
]
