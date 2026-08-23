"""Conservative continuous moving-surface/static-surface separation tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import inspect
from pathlib import Path

import numpy as np
import pytest

import kcg_connector.grasp.robust.continuous_collision as collision_module

from kcg_connector.grasp.robust.continuous_collision import (
    CLAIM_LIMITATIONS,
    ContinuousCollisionError,
    ContinuousCollisionState,
    INDEPENDENT_MOVING_PAIR_CLAIM_LIMITATIONS,
    INDEPENDENT_MOVING_PAIR_METHOD_ID,
    METHOD_ID,
    MOVING_PAIR_CLAIM_LIMITATIONS,
    MOVING_PAIR_METHOD_ID,
    PREPARED_STATIC_SURFACE_METHOD_ID,
    certify_independent_link_motion_surfaces_separated_from_each_other,
    certify_moving_link_surfaces_separated_from_each_other,
    certify_moving_link_surface_separated_from_static_surface,
    prepare_static_triangle_surface,
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


def _moving_triangle_point_bounds(
    x_lower: float,
    x_upper: float,
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.array(_moving_triangle(), copy=True)
    upper = np.array(_moving_triangle(), copy=True)
    lower[:, :, 0] += x_lower
    upper[:, :, 0] += x_upper
    return lower, upper


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


def _certify_independent_pair(
    *,
    second_origin_y: float = 3.0,
    first_q_start: np.ndarray | None = None,
    first_direction: np.ndarray | None = None,
    first_phase: IntervalBounds = IntervalBounds(0.0, 1.0),
    second_q_start: np.ndarray | None = None,
    second_direction: np.ndarray | None = None,
    second_phase: IntervalBounds = IntervalBounds(0.0, 1.0),
    maximum_phase_boxes: int = 1,
):
    return certify_independent_link_motion_surfaces_separated_from_each_other(
        backend=_backend(second_origin_y=second_origin_y),
        first_link_name="link_a",
        second_link_name="link_b",
        first_q_start=(
            np.zeros(3) if first_q_start is None else first_q_start
        ),
        first_direction=(
            np.asarray((1.0, 0.0, 0.0))
            if first_direction is None
            else first_direction
        ),
        first_phase=first_phase,
        second_q_start=(
            np.zeros(3) if second_q_start is None else second_q_start
        ),
        second_direction=(
            np.asarray((0.0, 1.0, 0.0))
            if second_direction is None
            else second_direction
        ),
        second_phase=second_phase,
        object_from_hand_base=np.eye(4),
        first_triangles_link_m=_moving_triangle(),
        second_triangles_link_m=_moving_triangle(),
        maximum_subdivision_phase_boxes=maximum_phase_boxes,
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


def test_triangle_narrowphase_resolves_coplanar_aabb_overlap() -> None:
    coplanar_disjoint = np.asarray(
        (
            (
                (0.0, 0.8, 0.8),
                (0.0, 1.8, 0.8),
                (0.0, 0.8, 1.8),
            ),
        ),
        dtype=np.float64,
    )

    certificate = _certify(
        static=coplanar_disjoint,
        direction=np.zeros(3),
        maximum_intervals=1,
    )

    assert certificate.state is ContinuousCollisionState.CERTIFIED_FREE
    assert certificate.audit.narrowphase_pair_evaluation_count == 1
    assert (
        certificate.audit.narrowphase_strictly_separated_pair_count == 1
    )
    assert certificate.audit.potential_overlap_pair_observation_count == 0


def test_triangle_narrowphase_never_clears_real_overlap() -> None:
    overlapping = np.asarray(
        (
            (
                (0.0, 0.2, 0.2),
                (0.0, 0.8, 0.2),
                (0.0, 0.2, 0.8),
            ),
        ),
        dtype=np.float64,
    )

    certificate = _certify(
        static=overlapping,
        direction=np.zeros(3),
        maximum_intervals=1,
    )

    assert certificate.state is ContinuousCollisionState.UNRESOLVED
    assert certificate.audit.narrowphase_pair_evaluation_count == 1
    assert (
        certificate.audit.narrowphase_strictly_separated_pair_count == 0
    )
    assert certificate.audit.potential_overlap_pair_observation_count == 1


def test_componentwise_projection_encloses_every_vertex_box_corner() -> None:
    lower = np.asarray(
        (
            (
                (-0.2, -0.1, -0.3),
                (0.8, -0.4, 0.1),
                (-0.5, 0.7, 0.2),
            ),
        ),
        dtype=np.float64,
    )
    upper = np.asarray(
        (
            (
                (0.4, 0.3, 0.2),
                (1.1, 0.2, 0.6),
                (0.2, 1.3, 0.9),
            ),
        ),
        dtype=np.float64,
    )
    (
        _triangle_lower,
        _triangle_upper,
        midpoint,
        half_extent,
        _moving_edges,
        _moving_stage_axes,
        _moving_stage_projection_minimum,
        _moving_stage_projection_maximum,
        _moving_stage_axis_norm,
        _moving_coordinate_scale,
    ) = collision_module._moving_triangle_packet_geometry(lower, upper)
    axes = np.asarray(
        (
            (
                (1.0, 0.0, 0.0),
                (0.0, -2.0, 0.0),
                (0.0, 0.0, 3.0),
                (1.0, -2.0, 0.5),
                (-0.25, 0.75, 1.5),
            ),
        ),
        dtype=np.float64,
    )

    projection_lower, projection_upper = (
        collision_module._moving_projection_interval_bounds(
            moving_triangles_midpoint_m=midpoint,
            moving_vertex_motion_half_extent_upper_m=half_extent,
            candidate_axes=axes,
        )
    )

    corner_selectors = np.asarray(
        tuple(
            (x_upper, y_upper, z_upper)
            for x_upper in (False, True)
            for y_upper in (False, True)
            for z_upper in (False, True)
        ),
        dtype=np.bool_,
    )
    for vertex_index in range(3):
        corners = np.where(
            corner_selectors,
            upper[0, vertex_index],
            lower[0, vertex_index],
        )
        corner_projections = corners @ axes[0].T
        assert np.all(corner_projections >= projection_lower[0])
        assert np.all(corner_projections <= projection_upper[0])


@pytest.mark.parametrize("bounded_motion", (False, True))
def test_precomputed_axis_families_match_all_axis_reference(
    bounded_motion: bool,
) -> None:
    moving = np.concatenate(
        (
            _moving_triangle(),
            _moving_triangle() + np.asarray((0.0, 2.0, 0.0)),
        )
    )
    lower = np.array(moving, copy=True)
    upper = np.array(moving, copy=True)
    if bounded_motion:
        lower -= np.asarray((0.08, 0.03, 0.05))
        upper += np.asarray((0.12, 0.07, 0.09))
    static = np.concatenate(
        (
            _static_triangle(0.0),
            _static_triangle(2.0),
            _static_triangle(0.0, z_offset=3.0),
        )
    )
    prepared = prepare_static_triangle_surface(static)
    (
        _triangle_lower,
        _triangle_upper,
        midpoint,
        half_extent,
        moving_edges,
        moving_stage_axes,
        moving_stage_projection_minimum,
        moving_stage_projection_maximum,
        moving_stage_axis_norm,
        moving_coordinate_scale,
    ) = collision_module._moving_triangle_packet_geometry(lower, upper)
    moving_faces = np.repeat(
        np.arange(len(moving), dtype=np.int64),
        prepared.triangle_count,
    )
    static_faces = np.tile(
        np.arange(prepared.triangle_count, dtype=np.int64),
        len(moving),
    )
    family_split = (
        prepared._bvh._precomputed_axis_family_strict_separation_mask(
            moving_faces=moving_faces,
            static_faces=static_faces,
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
    )
    reference = (
        collision_module._moving_triangle_triangle_strict_separation_mask(
            moving_triangles_midpoint_m=midpoint[moving_faces],
            moving_vertex_motion_half_extent_upper_m=(
                half_extent[moving_faces]
            ),
            fixed_triangles_m=prepared.triangles_object_m[static_faces],
        )
    )

    assert np.array_equal(family_split, reference)


def test_root_aggregates_multiple_bvh_leaves_into_one_pair_packet() -> None:
    static = np.concatenate(
        tuple(_static_triangle(value) for value in np.linspace(0.1, 0.8, 8))
    )
    prepared = prepare_static_triangle_surface(static)
    lower, upper = _moving_triangle_point_bounds(0.0, 1.0)
    counters = collision_module._Counters()

    root = prepared._bvh.classify_moving_triangles_packet(
        lower, upper, counters
    )

    assert prepared.triangle_count == 8
    assert counters.bvh_leaf_visits > 1
    assert counters.leaf_pair_evaluations == 8
    assert counters.narrowphase_pair_evaluations == 8
    assert counters.narrowphase_pair_packets == 1
    assert counters.narrowphase_root_pair_packets == 1
    assert counters.narrowphase_child_pair_packets == 0
    assert counters.pair_coverage == 8
    assert counters.potential_overlap_pairs == 8
    assert root.pair_count == 8
    assert np.array_equal(root.pair_ids, np.arange(8, dtype=np.int64))


def test_child_narrowphase_is_eager_before_subdivision() -> None:
    prepared = prepare_static_triangle_surface(
        np.concatenate(
            (
                _static_triangle(0.25),
                _static_triangle(0.50),
                _static_triangle(0.75),
            )
        )
    )
    root_lower, root_upper = _moving_triangle_point_bounds(0.0, 1.0)
    child_lower, child_upper = _moving_triangle_point_bounds(0.0, 0.6)
    grandchild_lower, grandchild_upper = _moving_triangle_point_bounds(
        0.0, 0.3
    )
    root = prepared._bvh.classify_moving_triangles_packet(
        root_lower, root_upper, collision_module._Counters()
    )
    child_counters = collision_module._Counters()
    child = prepared._bvh.classify_parent_pair_frontier_packet(
        child_lower, child_upper, root, child_counters
    )
    grandchild_counters = collision_module._Counters()
    grandchild = prepared._bvh.classify_parent_pair_frontier_packet(
        grandchild_lower,
        grandchild_upper,
        child,
        grandchild_counters,
    )

    assert root.pair_count == 3
    assert child.pair_count == 2
    assert child_counters.narrowphase_pair_evaluations == 2
    assert child_counters.narrowphase_invocation_intervals == 1
    assert child_counters.narrowphase_eager_child_intervals == 1
    assert child_counters.narrowphase_pair_packets == 1
    assert grandchild.pair_count == 1
    assert grandchild_counters.narrowphase_invocation_intervals == 1
    assert grandchild_counters.narrowphase_eager_child_intervals == 1
    assert grandchild_counters.narrowphase_pair_packets == 1
    assert grandchild_counters.narrowphase_pair_evaluations == 1


def test_eager_child_avoids_h99_sibling_terminal_duplication() -> None:
    coplanar_disjoint = np.asarray(
        (
            (
                (0.0, 0.8, 0.8),
                (0.0, 1.8, 0.8),
                (0.0, 0.8, 1.8),
            ),
        ),
        dtype=np.float64,
    )
    prepared = prepare_static_triangle_surface(coplanar_disjoint)
    moving = _moving_triangle()
    root_lower = moving - np.asarray((0.0, 1.0, 1.0))
    root_upper = moving + np.asarray((0.0, 1.0, 1.0))
    root = prepared._bvh.classify_moving_triangles_packet(
        root_lower, root_upper, collision_module._Counters()
    )
    counters = collision_module._Counters()

    child = prepared._bvh.classify_parent_pair_frontier_packet(
        moving,
        moving,
        root,
        counters,
    )

    assert root.pair_count == 1
    assert child.pair_count == 0
    assert counters.narrowphase_invocation_intervals == 1
    assert counters.narrowphase_eager_child_intervals == 1
    assert counters.narrowphase_pair_packets == 1
    assert counters.narrowphase_pair_evaluations == 1


def test_parent_pair_frontier_is_monotone_and_matches_full_recheck() -> None:
    prepared = prepare_static_triangle_surface(
        np.concatenate((_static_triangle(0.25), _static_triangle(0.75)))
    )
    parent_lower, parent_upper = _moving_triangle_point_bounds(0.0, 1.0)
    child_lower, child_upper = _moving_triangle_point_bounds(0.0, 0.4)
    grandchild_lower, grandchild_upper = _moving_triangle_point_bounds(
        0.0, 0.1
    )

    root_counters = collision_module._Counters()
    root = prepared._bvh.classify_moving_triangles_packet(
        parent_lower, parent_upper, root_counters
    )
    child_counters = collision_module._Counters()
    child = prepared._bvh.classify_parent_pair_frontier_packet(
        child_lower, child_upper, root, child_counters
    )
    child_reference_counters = collision_module._Counters()
    child_reference = prepared._bvh.classify_moving_triangles_packet(
        child_lower, child_upper, child_reference_counters
    )
    grandchild_counters = collision_module._Counters()
    grandchild = prepared._bvh.classify_parent_pair_frontier_packet(
        grandchild_lower,
        grandchild_upper,
        child,
        grandchild_counters,
    )
    grandchild_reference_counters = collision_module._Counters()
    grandchild_reference = prepared._bvh.classify_moving_triangles_packet(
        grandchild_lower,
        grandchild_upper,
        grandchild_reference_counters,
    )

    assert root.pair_count == 2
    assert child.pair_count == 1
    assert grandchild.pair_count == 0
    assert np.array_equal(child.pair_ids, child_reference.pair_ids)
    assert np.array_equal(
        grandchild.pair_ids, grandchild_reference.pair_ids
    )
    assert set(child.pair_ids).issubset(set(root.pair_ids))
    assert set(grandchild.pair_ids).issubset(set(child.pair_ids))
    assert root_counters.root_bvh_intervals == 1
    assert child_counters.root_bvh_intervals == 0
    assert child_counters.frontier_pair_evaluations == 2
    assert grandchild_counters.frontier_pair_evaluations == 1
    assert grandchild_counters.inherited_strictly_separated_pairs == 1
    assert grandchild_counters.pair_coverage == 2
    assert grandchild_counters.strictly_separated_pairs == 2
    assert grandchild_counters.potential_overlap_pairs == 0
    assert not root.pair_ids.flags.writeable


def test_persistent_frontier_matches_full_recheck_certificate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    static = np.concatenate(
        (_static_triangle(0.5), _static_triangle(0.0, z_offset=2.0))
    )
    optimized = _certify(static=static, maximum_intervals=7)
    full_packet = (
        collision_module._StaticTriangleBVH.classify_moving_triangles_packet
    )

    def full_recheck_reference(
        self,
        point_lower_m,
        point_upper_m,
        _parent_frontier,
        counters,
    ):
        root_bvh_count = counters.root_bvh_intervals
        result = full_packet(self, point_lower_m, point_upper_m, counters)
        counters.root_bvh_intervals = root_bvh_count
        return result

    monkeypatch.setattr(
        collision_module._StaticTriangleBVH,
        "classify_parent_pair_frontier_packet",
        full_recheck_reference,
    )
    reference = _certify(static=static, maximum_intervals=7)

    assert optimized.state == reference.state
    assert optimized.unresolved_interval == reference.unresolved_interval
    assert (
        optimized.certified_free_leaf_intervals
        == reference.certified_free_leaf_intervals
    )
    assert (
        optimized.audit.processed_interval_count
        == reference.audit.processed_interval_count
    )
    assert optimized.audit.pair_universe_count == reference.audit.pair_universe_count
    assert optimized.audit.pair_coverage_count == reference.audit.pair_coverage_count
    assert (
        optimized.audit.terminal_unresolved_pair_count
        == reference.audit.terminal_unresolved_pair_count
    )
    assert optimized.audit.root_bvh_interval_count == 1
    assert optimized.audit.inherited_strictly_separated_pair_count > 0
    assert optimized.audit.frontier_pair_evaluation_count > 0
    assert optimized.audit.narrowphase_eager_child_interval_count > 0
    assert optimized.audit.narrowphase_pair_packet_count > 0
    assert optimized.audit.all_processed_pairs_accounted_for
    assert reference.audit.all_processed_pairs_accounted_for
    assert (
        optimized.audit.leaf_pair_evaluation_count
        < reference.audit.leaf_pair_evaluation_count
    )


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


def test_prepared_static_surface_matches_direct_and_rejects_mismatch() -> None:
    static = np.concatenate(
        (_static_triangle(3.0), _static_triangle(4.0, z_offset=2.0))
    )
    direct = _certify(static=static)
    prepared = prepare_static_triangle_surface(
        static[::-1, ::-1, :],
        expected_geometry_sha256=(
            direct.audit.static_surface_geometry_sha256
        ),
    )

    reused = _certify(static=prepared)

    assert prepared.method_id == PREPARED_STATIC_SURFACE_METHOD_ID
    assert prepared.triangle_count == len(static)
    assert reused == direct
    assert not prepared.triangles_object_m.flags.writeable
    with pytest.raises(ValueError):
        prepared.triangles_object_m[0, 0, 0] = 9.0
    with pytest.raises(ContinuousCollisionError, match="hash mismatch"):
        prepare_static_triangle_surface(
            static,
            expected_geometry_sha256="0" * 64,
        )


def test_surface_collision_uses_one_batch_fk_per_link_and_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _backend(second_origin_y=5.0)
    batch_calls: list[tuple[str, int]] = []
    original_batch = backend.point_motion_many

    def counted_batch(**kwargs):
        points = np.asarray(kwargs["points_local_m"])
        batch_calls.append((str(kwargs["link_name"]), len(points)))
        return original_batch(**kwargs)

    def scalar_must_not_run(**_kwargs):
        raise AssertionError("scalar point_motion must not run")

    monkeypatch.setattr(backend, "point_motion_many", counted_batch)
    monkeypatch.setattr(backend, "point_motion", scalar_must_not_run)
    moving = np.concatenate(
        (
            _moving_triangle(),
            _moving_triangle() + np.asarray((0.0, 2.0, 0.0)),
        )
    )

    static_certificate = (
        certify_moving_link_surface_separated_from_static_surface(
            backend=backend,
            link_name="link_a",
            q_start=np.zeros(3),
            direction=np.asarray((1.0, 0.0, 0.0)),
            phase=IntervalBounds(0.0, 1.0),
            object_from_hand_base=np.eye(4),
            moving_triangles_link_m=moving,
            static_triangles_object_m=_static_triangle(10.0),
            maximum_subdivision_intervals=1,
        )
    )
    pair_certificate = certify_moving_link_surfaces_separated_from_each_other(
        backend=backend,
        first_link_name="link_a",
        second_link_name="link_b",
        q_start=np.zeros(3),
        direction=np.zeros(3),
        phase=IntervalBounds(0.0, 1.0),
        object_from_hand_base=np.eye(4),
        first_triangles_link_m=moving,
        second_triangles_link_m=_moving_triangle(),
        maximum_subdivision_intervals=1,
    )

    assert static_certificate.state is ContinuousCollisionState.CERTIFIED_FREE
    assert pair_certificate.state is ContinuousCollisionState.CERTIFIED_FREE
    assert batch_calls == [("link_a", 6), ("link_a", 6), ("link_b", 3)]
    assert static_certificate.audit.point_motion_evaluation_count == 6
    assert pair_certificate.audit.point_motion_evaluation_count == 9


def test_packet_static_bvh_matches_per_face_reference_accounting() -> None:
    static = np.concatenate(
        tuple(
            _static_triangle(x_coordinate, z_offset=z_offset)
            for x_coordinate, z_offset in (
                (0.0, 0.0),
                (1.0, 0.0),
                (2.0, 2.0),
                (3.0, 2.0),
                (4.0, 4.0),
            )
        )
    )
    prepared = prepare_static_triangle_surface(static)
    moving_lower = np.asarray(
        (
            (-0.1, -0.1, -0.1),
            (1.5, 0.2, 0.2),
            (10.0, 10.0, 10.0),
            (2.5, 0.0, 2.0),
        ),
        dtype=np.float64,
    )
    moving_upper = np.asarray(
        (
            (0.1, 0.1, 0.1),
            (2.5, 0.8, 0.8),
            (11.0, 11.0, 11.0),
            (3.5, 1.0, 3.0),
        ),
        dtype=np.float64,
    )
    scalar = collision_module._Counters()
    scalar_overlap_count = 0
    for lower, upper in zip(moving_lower, moving_upper):
        candidates, pruned = prepared._bvh.potential_faces(
            lower, upper, scalar
        )
        scalar.pair_coverage += pruned
        scalar.strictly_separated_pairs += pruned
        for face_index in candidates:
            scalar.leaf_pair_evaluations += 1
            scalar.pair_coverage += 1
            separated = collision_module._strictly_separated(
                lower,
                upper,
                prepared._bvh.face_lower_m[face_index],
                prepared._bvh.face_upper_m[face_index],
            )
            if separated:
                scalar.strictly_separated_pairs += 1
            else:
                scalar.potential_overlap_pairs += 1
                scalar_overlap_count += 1

    packet = collision_module._Counters()
    packet_overlap_count = (
        prepared._bvh.classify_moving_face_bounds_packet(
            moving_lower,
            moving_upper,
            packet,
        )
    )

    assert packet_overlap_count == scalar_overlap_count
    assert packet.bvh_node_visits == scalar.bvh_node_visits
    assert packet.bvh_leaf_visits == scalar.bvh_leaf_visits
    assert packet.leaf_pair_evaluations == scalar.leaf_pair_evaluations
    assert packet.pair_coverage == scalar.pair_coverage
    assert packet.strictly_separated_pairs == scalar.strictly_separated_pairs
    assert packet.potential_overlap_pairs == scalar.potential_overlap_pairs


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


def test_independent_pair_covers_complete_free_phase_product() -> None:
    certificate = _certify_independent_pair(second_origin_y=4.0)

    assert certificate.state is ContinuousCollisionState.CERTIFIED_FREE
    assert certificate.unresolved_phase_box is None
    assert certificate.audit.method_id == INDEPENDENT_MOVING_PAIR_METHOD_ID
    assert certificate.audit.claim_limitations == (
        INDEPENDENT_MOVING_PAIR_CLAIM_LIMITATIONS
    )
    assert certificate.audit.entire_phase_product_covered
    assert certificate.audit.processed_phase_box_count == 1
    assert certificate.audit.certified_free_leaf_phase_box_count == 1
    assert certificate.audit.pair_coverage_count == 1
    assert certificate.audit.point_motion_evaluation_count == 6
    assert (
        certificate.audit.relative_coordinate_interval_evaluation_count
        == 27
    )


def test_independent_pair_checks_product_not_shared_phase_diagonal() -> None:
    shared_diagonal = _certify_pair(
        second_origin_y=0.0,
        q_start=np.asarray((0.0, 1.0, 0.0)),
        direction=np.asarray((1.0, 1.0, 0.0)),
        maximum_intervals=3,
    )
    independent_product = _certify_independent_pair(
        second_origin_y=0.0,
        first_q_start=np.zeros(3),
        first_direction=np.asarray((1.0, 0.0, 0.0)),
        second_q_start=np.asarray((0.0, 1.0, 0.0)),
        second_direction=np.asarray((0.0, 1.0, 0.0)),
        maximum_phase_boxes=1,
    )

    assert shared_diagonal.state is ContinuousCollisionState.CERTIFIED_FREE
    assert independent_product.state is ContinuousCollisionState.UNRESOLVED
    assert independent_product.audit.unresolved_reason == (
        "SUBDIVISION_PHASE_BOX_BUDGET_EXHAUSTED"
    )
    assert independent_product.audit.terminal_unresolved_pair_count == 1


def test_independent_pair_budget_exhaustion_never_claims_free() -> None:
    certificate = _certify_independent_pair(
        second_origin_y=0.0,
        maximum_phase_boxes=1,
    )

    assert certificate.state is ContinuousCollisionState.UNRESOLVED
    assert certificate.unresolved_phase_box is not None
    assert not certificate.audit.entire_phase_product_covered
    assert certificate.audit.processed_phase_box_count == 1
    assert certificate.audit.potential_overlap_pair_observation_count == 1
