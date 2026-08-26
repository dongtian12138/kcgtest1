"""Direct numerical regressions for height-before-budget projection."""

import numpy as np
import pytest

from kcg_connector.grasp.carts_v2.height_projection import (
    intersect_contact_with_table,
    minimum_handbase_z_for_finite_table,
    minimum_z_over_finite_table_top,
    project_height_to_intervals,
    translate_transform_world_z,
)


def test_world_z_translation_is_left_multiplied() -> None:
    pose = np.asarray([
        [0.0, -1.0, 0.0, 0.2],
        [1.0, 0.0, 0.0, -0.3],
        [0.0, 0.0, 1.0, 0.4],
        [0.0, 0.0, 0.0, 1.0],
    ])
    shifted = translate_transform_world_z(pose, 0.025)
    assert shifted[:3, :3] == pytest.approx(pose[:3, :3])
    assert shifted[:3, 3] == pytest.approx((0.2, -0.3, 0.425))


def test_finite_table_ignores_outside_geometry_and_bounds_crossing_triangle() -> None:
    triangles = np.asarray([[  # state, triangle, vertex, xyz
        [[-0.2, 0.0, -0.20], [0.0, 0.2, -0.10], [0.2, 0.0, -0.10]],
        [[10.0, 10.0, -9.0], [10.1, 10.0, -9.0], [10.0, 10.1, -9.0]],
    ]])
    result = minimum_handbase_z_for_finite_table(
        triangles, np.eye(3), (0.0, 0.0),
        np.asarray(((-0.1, 0.1), (-0.1, 0.1))), 0.5,
        required_clearance_m=0.01,
    )
    assert result.minimum_handbase_z_m == pytest.approx(0.71)
    assert result.minimum_relative_z_m == pytest.approx(-0.20)
    assert result.contributing_primitive_index == 0
    assert result.overlapping_primitive_count == 1
    assert result.geometry_kind == "TRIANGLE_AABB_CONSERVATIVE"


def test_negative_table_clearance_is_rejected() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        minimum_handbase_z_for_finite_table(
            np.asarray([[[0.0, 0.0, -0.1]]]), np.eye(3), (0.0, 0.0),
            np.asarray(((-1.0, 1.0), (-1.0, 1.0))), 0.0,
            required_clearance_m=-1.0e-3,
        )


def test_discrete_points_outside_finite_table_impose_no_height() -> None:
    points = np.asarray([[[2.0, 2.0, -100.0], [3.0, 3.0, -200.0]]])
    result = minimum_handbase_z_for_finite_table(
        points, np.eye(3), (0.0, 0.0),
        np.asarray(((-1.0, 1.0), (-1.0, 1.0))), 0.0,
    )
    assert result.minimum_handbase_z_m is None
    assert result.overlapping_primitive_count == 0
    assert result.geometry_kind == "POINT_DISCRETE"


def test_exact_triangle_query_rejects_aabb_only_table_overlap() -> None:
    triangles = np.asarray((
        ((-2.0, 0.4, -4.0), (0.4, 2.0, -3.0), (2.0, 2.0, -2.0)),
    ))
    minimum, index = minimum_z_over_finite_table_top(
        triangles, np.asarray(((-0.1, 0.1), (-0.1, 0.1))))
    assert minimum is None
    assert index is None


def test_exact_triangle_query_uses_clipped_table_crossing_height() -> None:
    triangles = np.asarray((
        ((-1.0, 0.0, -1.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0)),
    ))
    minimum, index = minimum_z_over_finite_table_top(
        triangles, np.asarray(((-0.1, 0.1), (-0.1, 0.1))))
    assert minimum == pytest.approx(-0.1)
    assert index == 0


def test_exact_triangle_query_fast_path_keeps_lowest_fully_inside_face() -> None:
    triangles = np.asarray((
        ((-0.2, 0.0, 0.0), (0.2, 0.0, 2.0), (0.0, 0.2, 2.0)),
        ((-0.05, -0.05, -0.3), (0.05, -0.05, -0.2), (0.0, 0.05, -0.1)),
    ))
    minimum, index = minimum_z_over_finite_table_top(
        triangles, np.asarray(((-0.1, 0.1), (-0.1, 0.1))))
    assert minimum == pytest.approx(-0.3)
    assert index == 1


def test_table_intersection_and_nearest_projection_prefer_higher_tie() -> None:
    feasible = intersect_contact_with_table(((0.0, 1.0), (3.0, 4.0)), 0.5)
    assert feasible == ((0.5, 1.0), (3.0, 4.0))
    projected = project_height_to_intervals(2.0, feasible)
    assert projected.projected_height_m == pytest.approx(3.0)
    assert projected.translation_world_z_m == pytest.approx(1.0)
    assert projected.selected_interval_m == (3.0, 4.0)


def test_empty_contact_table_intersection_fails_closed() -> None:
    feasible = intersect_contact_with_table(((0.0, 0.2),), 0.3)
    assert feasible == ()
    with pytest.raises(ValueError, match="no height satisfies"):
        project_height_to_intervals(0.1, feasible)
