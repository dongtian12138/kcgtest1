"""Conservative continuous moving-surface/static-surface separation tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import inspect
from pathlib import Path

import numpy as np
import pytest

from kcg_connector.grasp.robust.continuous_collision import (
    CLAIM_LIMITATIONS,
    ContinuousCollisionError,
    ContinuousCollisionState,
    METHOD_ID,
    MOVING_PAIR_CLAIM_LIMITATIONS,
    MOVING_PAIR_METHOD_ID,
    certify_moving_link_surfaces_separated_from_each_other,
    certify_moving_link_surface_separated_from_static_surface,
)
from kcg_connector.grasp.robust.hand_model import (
    GeometrySpec,
    JointLimit,
    JointSpec,
    PadGeometry,
    ThreeFingerHandModel,
)
from kcg_connector.grasp.robust.interval_kinematics import (
    DirectedIntervalKinematics,
    IntervalArithmeticOptions,
    IntervalBounds,
)


def _backend(*, second_origin_y: float = 0.0) -> DirectedIntervalKinematics:
    joints = {}
    pads = {}
    finger_joints = {}
    for name in ("a", "b", "c"):
        joint_name = f"joint_{name}"
        link_name = f"link_{name}"
        finger_name = f"finger_{name}"
        pad_name = f"pad_{name}"
        joints[joint_name] = JointSpec(
            name=joint_name,
            joint_type="prismatic",
            parent_link="hand_base",
            child_link=link_name,
            origin_xyz_m=(0.0, 0.0, 0.0),
            origin_rpy_rad=(0.0, 0.0, 0.0),
            axis=(1.0, 0.0, 0.0),
            limit=JointLimit(-2.0, 2.0),
        )
        pads[pad_name] = PadGeometry(
            name=pad_name,
            finger_name=finger_name,
            link_name=link_name,
            origin_xyz_m=(0.0, 0.0, 0.0),
            origin_rpy_rad=(0.0, 0.0, 0.0),
            geometry=GeometrySpec("box", (1.0, 1.0, 1.0)),
        )
        finger_joints[finger_name] = (joint_name,)
        if name == "b":
            joint = joints[joint_name]
            joints[joint_name] = JointSpec(
                name=joint.name,
                joint_type=joint.joint_type,
                parent_link=joint.parent_link,
                child_link=joint.child_link,
                origin_xyz_m=(0.0, float(second_origin_y), 0.0),
                origin_rpy_rad=joint.origin_rpy_rad,
                axis=joint.axis,
                limit=joint.limit,
            )
    hand = ThreeFingerHandModel(
        base_link="hand_base",
        joints=joints,
        joint_order=tuple(joints),
        finger_joint_names=finger_joints,
        pads=pads,
    )
    return DirectedIntervalKinematics(
        hand,
        IntervalArithmeticOptions(
            decimal_precision=80,
            maximum_root_bisection_iterations=256,
        ),
    )


def _moving_triangle() -> np.ndarray:
    return np.asarray(
        (((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),),
        dtype=np.float64,
    )


def _static_triangle(
    x_coordinate: float,
    *,
    z_offset: float = 0.0,
) -> np.ndarray:
    return np.asarray(
        (
            (
                (x_coordinate, 0.0, z_offset),
                (x_coordinate, 1.0, z_offset),
                (x_coordinate, 0.0, z_offset + 1.0),
            ),
        ),
        dtype=np.float64,
    )


def _certify(
    *,
    static: np.ndarray,
    direction: np.ndarray | None = None,
    phase: IntervalBounds = IntervalBounds(0.0, 1.0),
    maximum_intervals: int = 1,
    moving: np.ndarray | None = None,
    base: np.ndarray | None = None,
):
    return certify_moving_link_surface_separated_from_static_surface(
        backend=_backend(),
        link_name="link_a",
        q_start=np.zeros(3),
        direction=(
            np.asarray((1.0, 0.0, 0.0))
            if direction is None
            else direction
        ),
        phase=phase,
        object_from_hand_base=(np.eye(4) if base is None else base),
        moving_triangles_link_m=(
            _moving_triangle() if moving is None else moving
        ),
        static_triangles_object_m=static,
        maximum_subdivision_intervals=maximum_intervals,
    )


def _certify_pair(
    *,
    second_origin_y: float = 3.0,
    q_start: np.ndarray | None = None,
    direction: np.ndarray | None = None,
    phase: IntervalBounds = IntervalBounds(0.0, 1.0),
    maximum_intervals: int = 1,
    first_surface: np.ndarray | None = None,
    second_surface: np.ndarray | None = None,
    base: np.ndarray | None = None,
    first_link_name: str = "link_a",
    second_link_name: str = "link_b",
):
    return certify_moving_link_surfaces_separated_from_each_other(
        backend=_backend(second_origin_y=second_origin_y),
        first_link_name=first_link_name,
        second_link_name=second_link_name,
        q_start=np.zeros(3) if q_start is None else q_start,
        direction=np.zeros(3) if direction is None else direction,
        phase=phase,
        object_from_hand_base=np.eye(4) if base is None else base,
        first_triangles_link_m=(
            _moving_triangle() if first_surface is None else first_surface
        ),
        second_triangles_link_m=(
            _moving_triangle() if second_surface is None else second_surface
        ),
        maximum_subdivision_intervals=maximum_intervals,
    )


def test_prismatic_path_with_strict_axis_separation_is_free() -> None:
    certificate = _certify(static=_static_triangle(2.0))

    assert certificate.state == ContinuousCollisionState.CERTIFIED_FREE
    assert certificate.unresolved_interval is None
    assert certificate.certified_free_leaf_intervals == (
        IntervalBounds(0.0, 1.0),
    )
    assert certificate.audit.method_id == METHOD_ID
    assert certificate.audit.processed_interval_count == 1
    assert certificate.audit.point_motion_evaluation_count == 3
    assert certificate.audit.pair_universe_count == 1
    assert certificate.audit.pair_count_per_interval == 1
    assert certificate.audit.pair_coverage_count == 1
    assert certificate.audit.strictly_separated_pair_count == 1
    assert certificate.audit.potential_overlap_pair_observation_count == 0
    assert certificate.audit.all_processed_pairs_accounted_for
    assert certificate.audit.entire_phase_covered
    assert certificate.audit.claim_limitations == CLAIM_LIMITATIONS
    with pytest.raises(FrozenInstanceError):
        certificate.state = ContinuousCollisionState.UNRESOLVED


@pytest.mark.parametrize(
    ("static", "direction", "maximum_intervals"),
    (
        (_static_triangle(0.5), np.asarray((1.0, 0.0, 0.0)), 7),
        (_static_triangle(1.0), np.asarray((1.0, 0.0, 0.0)), 3),
        (_static_triangle(0.0), np.zeros(3), 1),
    ),
    ids=("transverse_crossing", "endpoint_touch", "coplanar_path"),
)
def test_crossing_touch_and_coplanarity_remain_unresolved(
    static: np.ndarray,
    direction: np.ndarray,
    maximum_intervals: int,
) -> None:
    certificate = _certify(
        static=static,
        direction=direction,
        maximum_intervals=maximum_intervals,
    )

    assert certificate.state == ContinuousCollisionState.UNRESOLVED
    assert isinstance(certificate.unresolved_interval, IntervalBounds)
    assert not certificate.audit.entire_phase_covered
    assert certificate.audit.unresolved_reason == (
        "SUBDIVISION_INTERVAL_BUDGET_EXHAUSTED"
    )
    assert (
        certificate.audit.potential_overlap_pair_observation_count > 0
    )
    assert certificate.audit.all_processed_pairs_accounted_for


def test_all_faces_are_covered_not_only_initially_nearest_face() -> None:
    decoys_near_but_z_separated = np.concatenate(
        tuple(
            _static_triangle(x_coordinate, z_offset=1.01)
            for x_coordinate in (0.1, 0.2, 0.3, 0.4, 0.5)
        )
    )
    later_path_overlap = _static_triangle(0.9)
    static = np.concatenate(
        (decoys_near_but_z_separated, later_path_overlap)
    )

    certificate = _certify(static=static, maximum_intervals=1)

    assert certificate.state == ContinuousCollisionState.UNRESOLVED
    assert certificate.audit.pair_count_per_interval == 6
    assert certificate.audit.pair_universe_count == 6
    assert certificate.audit.pair_coverage_count == 6
    assert certificate.audit.bvh_node_visit_count > 1
    assert certificate.audit.leaf_pair_evaluation_count == 3
    assert certificate.audit.strictly_separated_pair_count == 5
    assert certificate.audit.potential_overlap_pair_observation_count == 1
    assert certificate.audit.all_processed_pairs_accounted_for


def test_common_proper_se3_preserves_free_certificate_and_counts() -> None:
    static = _static_triangle(10.0)
    reference = _certify(static=static)
    rotation = np.asarray(
        (
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        )
    )
    translation = np.asarray((3.0, -4.0, 2.0))
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    transformed_static = static @ rotation.T + translation

    transformed = _certify(
        static=transformed_static,
        base=transform,
    )

    assert transformed.state == reference.state
    assert (
        transformed.audit.processed_interval_count
        == reference.audit.processed_interval_count
    )
    assert (
        transformed.audit.pair_coverage_count
        == reference.audit.pair_coverage_count
    )
    assert (
        transformed.audit.strictly_separated_pair_count
        == reference.audit.strictly_separated_pair_count
    )


def test_budget_exhaustion_never_claims_unprocessed_path_free() -> None:
    certificate = _certify(
        static=_static_triangle(0.5),
        maximum_intervals=1,
    )

    assert certificate.state == ContinuousCollisionState.UNRESOLVED
    assert certificate.audit.processed_interval_count == 1
    assert certificate.audit.maximum_subdivision_intervals == 1
    assert certificate.audit.unresolved_reason == (
        "SUBDIVISION_INTERVAL_BUDGET_EXHAUSTED"
    )
    assert certificate.audit.terminal_unresolved_pair_count == 1


def test_budget_can_expire_before_an_unprocessed_right_interval() -> None:
    certificate = _certify(
        static=_static_triangle(1.0),
        maximum_intervals=2,
    )

    assert certificate.state == ContinuousCollisionState.UNRESOLVED
    assert certificate.audit.processed_interval_count == 2
    assert certificate.audit.terminal_unresolved_pair_count == 0
    assert certificate.unresolved_interval == IntervalBounds(0.5, 1.0)
    assert certificate.certified_free_leaf_intervals == (
        IntervalBounds(0.0, 0.5),
    )


def test_adjacent_binary64_phase_overlap_is_unresolved() -> None:
    lower = 1.0
    upper = float(np.nextafter(lower, 2.0))
    certificate = _certify(
        static=_static_triangle(0.0),
        direction=np.zeros(3),
        phase=IntervalBounds(lower, upper),
        maximum_intervals=2,
    )

    assert certificate.state == ContinuousCollisionState.UNRESOLVED
    assert certificate.audit.unresolved_reason == (
        "ADJACENT_BINARY64_PHASE_ENDPOINTS"
    )
    assert certificate.unresolved_interval == IntervalBounds(lower, upper)


def test_face_order_and_each_face_winding_do_not_change_certificate() -> None:
    moving = np.concatenate(
        (
            _moving_triangle(),
            _moving_triangle() + np.asarray((0.0, 2.0, 0.0)),
        )
    )
    static = np.concatenate(
        (
            _static_triangle(3.0),
            _static_triangle(4.0, z_offset=2.0),
        )
    )
    reference = _certify(static=static, moving=moving)
    permuted_moving = moving[::-1, ::-1, :]
    permuted_static = static[::-1, (1, 0, 2), :]

    permuted = _certify(
        static=permuted_static,
        moving=permuted_moving,
    )

    assert permuted.state == reference.state
    assert (
        permuted.audit.moving_surface_geometry_sha256
        == reference.audit.moving_surface_geometry_sha256
    )
    assert (
        permuted.audit.static_surface_geometry_sha256
        == reference.audit.static_surface_geometry_sha256
    )
    assert permuted.audit.pair_universe_count == 4
    assert permuted.audit.pair_coverage_count == 4
    assert (
        permuted.audit.strictly_separated_pair_count
        == reference.audit.strictly_separated_pair_count
    )


def test_core_contains_no_allowed_contact_or_metric_angle_gate() -> None:
    implementation = (
        Path(__file__).resolve().parents[2]
        / "kcg_connector/grasp/robust/continuous_collision.py"
    ).read_text(encoding="utf-8")
    forbidden_tokens = (
        "allowed_contact",
        "adjacent_link_exemption",
        "disable_collisions",
        "never_collision",
        "srdf",
        "millimeter",
        "distance_threshold",
        "angle_threshold",
        "degree_threshold",
    )
    assert all(token not in implementation for token in forbidden_tokens)


def test_finite_triangle_edge_overflow_is_rejected() -> None:
    maximum = np.finfo(np.float64).max
    overflowing = np.asarray(
        (
            (
                (-maximum, 0.0, 0.0),
                (maximum, 0.0, 0.0),
                (0.0, 1.0, 0.0),
            ),
        )
    )
    with pytest.raises(ContinuousCollisionError, match="overflowed"):
        _certify(
            static=_static_triangle(2.0),
            moving=overflowing,
        )


def test_two_moving_surfaces_with_strict_relative_axis_are_free() -> None:
    certificate = _certify_pair(
        second_origin_y=3.0,
        direction=np.asarray((1.0, 1.0, 0.0)),
    )

    assert certificate.state == ContinuousCollisionState.CERTIFIED_FREE
    assert certificate.audit.method_id == MOVING_PAIR_METHOD_ID
    assert certificate.audit.first_link_name == "link_a"
    assert certificate.audit.second_link_name == "link_b"
    assert certificate.audit.point_motion_evaluation_count == 6
    assert (
        certificate.audit.relative_coordinate_interval_evaluation_count
        == 27
    )
    assert certificate.audit.pair_count_per_interval == 1
    assert certificate.audit.pair_universe_count == 1
    assert certificate.audit.pair_coverage_count == 1
    assert certificate.audit.strictly_separated_pair_count == 1
    assert certificate.audit.potential_overlap_pair_observation_count == 0
    assert certificate.audit.all_processed_pairs_accounted_for
    assert certificate.audit.entire_phase_covered
    assert (
        certificate.audit.claim_limitations
        == MOVING_PAIR_CLAIM_LIMITATIONS
    )
    assert len(certificate.audit.pair_contract_sha256) == 64


@pytest.mark.parametrize(
    ("q_start", "direction", "maximum_intervals"),
    (
        (
            np.asarray((0.0, 1.0, 0.0)),
            np.asarray((1.0, -1.0, 0.0)),
            7,
        ),
        (
            np.asarray((0.0, 1.0, 0.0)),
            np.asarray((0.5, -0.5, 0.0)),
            3,
        ),
        (np.zeros(3), np.zeros(3), 1),
    ),
    ids=("crossing", "endpoint_touch", "coplanar"),
)
def test_moving_crossing_touch_and_coplanarity_are_unresolved(
    q_start: np.ndarray,
    direction: np.ndarray,
    maximum_intervals: int,
) -> None:
    certificate = _certify_pair(
        second_origin_y=0.0,
        q_start=q_start,
        direction=direction,
        maximum_intervals=maximum_intervals,
    )

    assert certificate.state == ContinuousCollisionState.UNRESOLVED
    assert certificate.audit.unresolved_reason == (
        "SUBDIVISION_INTERVAL_BUDGET_EXHAUSTED"
    )
    assert (
        certificate.audit.potential_overlap_pair_observation_count > 0
    )
    assert certificate.audit.all_processed_pairs_accounted_for
    assert not certificate.audit.entire_phase_covered


def test_moving_pair_budget_never_claims_overlap_free() -> None:
    certificate = _certify_pair(
        second_origin_y=0.0,
        q_start=np.asarray((0.0, 1.0, 0.0)),
        direction=np.asarray((1.0, -1.0, 0.0)),
        maximum_intervals=1,
    )

    assert certificate.state == ContinuousCollisionState.UNRESOLVED
    assert certificate.audit.processed_interval_count == 1
    assert certificate.audit.terminal_unresolved_pair_count == 1
    assert certificate.audit.pair_coverage_count == 1


def test_moving_pair_adjacent_binary64_overlap_is_unresolved() -> None:
    lower = 1.0
    upper = float(np.nextafter(lower, 2.0))
    certificate = _certify_pair(
        second_origin_y=0.0,
        phase=IntervalBounds(lower, upper),
        maximum_intervals=2,
    )

    assert certificate.state == ContinuousCollisionState.UNRESOLVED
    assert certificate.audit.unresolved_reason == (
        "ADJACENT_BINARY64_PHASE_ENDPOINTS"
    )
    assert certificate.unresolved_interval == IntervalBounds(lower, upper)


def test_moving_pair_is_equivariant_under_common_proper_se3() -> None:
    reference = _certify_pair(second_origin_y=4.0)
    rotation = np.asarray(
        (
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        )
    )
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = (3.0, -4.0, 2.0)

    transformed = _certify_pair(
        second_origin_y=4.0,
        base=transform,
    )

    assert transformed.state == reference.state
    assert (
        transformed.audit.pair_coverage_count
        == reference.audit.pair_coverage_count
    )
    assert (
        transformed.audit.strictly_separated_pair_count
        == reference.audit.strictly_separated_pair_count
    )
    assert (
        transformed.audit.relative_coordinate_interval_evaluation_count
        == reference.audit.relative_coordinate_interval_evaluation_count
    )


def test_moving_pair_face_permutations_and_side_swap_are_invariant() -> None:
    first_surface = np.concatenate(
        (
            _moving_triangle(),
            _moving_triangle() + np.asarray((0.0, 2.0, 0.0)),
        )
    )
    second_surface = np.concatenate(
        (
            _moving_triangle() + np.asarray((0.0, 0.0, 2.0)),
            _moving_triangle() + np.asarray((0.0, 3.0, 2.0)),
        )
    )
    reference = _certify_pair(
        second_origin_y=10.0,
        first_surface=first_surface,
        second_surface=second_surface,
    )
    permuted = _certify_pair(
        second_origin_y=10.0,
        first_surface=first_surface[::-1, ::-1, :],
        second_surface=second_surface[::-1, (1, 0, 2), :],
    )
    swapped = _certify_pair(
        second_origin_y=10.0,
        first_link_name="link_b",
        second_link_name="link_a",
        first_surface=second_surface,
        second_surface=first_surface,
    )

    assert reference.state == ContinuousCollisionState.CERTIFIED_FREE
    assert permuted.state == reference.state
    assert swapped.state == reference.state
    assert (
        permuted.audit.pair_contract_sha256
        == reference.audit.pair_contract_sha256
    )
    assert (
        swapped.audit.pair_contract_sha256
        == reference.audit.pair_contract_sha256
    )
    assert reference.audit.pair_count_per_interval == 4
    assert reference.audit.pair_coverage_count == 4
    assert reference.audit.point_motion_evaluation_count == 12
    assert (
        reference.audit.relative_coordinate_interval_evaluation_count
        == 108
    )


def test_moving_pair_rejects_identical_link_names() -> None:
    with pytest.raises(ContinuousCollisionError, match="distinct named links"):
        _certify_pair(
            first_link_name="link_a",
            second_link_name="link_a",
        )


def test_moving_pair_public_api_has_no_semantic_exemption_input() -> None:
    parameters = set(
        inspect.signature(
            certify_moving_link_surfaces_separated_from_each_other
        ).parameters
    )
    forbidden_parameters = {
        "allowed_pairs",
        "disabled_pairs",
        "adjacent_pairs",
        "never_pairs",
        "srdf",
    }
    assert parameters.isdisjoint(forbidden_parameters)
