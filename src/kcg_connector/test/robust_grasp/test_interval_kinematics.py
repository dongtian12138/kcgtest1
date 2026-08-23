import math
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
from itertools import product

import numpy as np
import pytest

from kcg_connector.grasp.robust.hand_model import (
    GeometrySpec,
    JointLimit,
    JointSpec,
    MimicSpec,
    PadGeometry,
    ThreeFingerHandModel,
)
from kcg_connector.grasp.robust.interval_kinematics import (
    COMPILED_POINT_PLANE_BINDING_CACHE_CAPACITY,
    DISPLAY_APPROXIMATION_ROLE,
    INTERVAL_GEOMETRIC_JACOBIAN_METHOD_ID,
    INTERVAL_RIGID_TRANSFORM_METHOD_ID,
    CertifiedImplicitRoot,
    DirectedIntervalKinematics,
    IntervalArithmeticOptions,
    IntervalBounds,
    IntervalKinematicsError,
    IntervalRootClassification,
    IntervalRootState,
    METHOD_ID,
    MULTIPHASE_TRANSFORM_CACHE_CAPACITY,
)


def _three_finger_hand(
    *,
    first_joint_type: str,
    first_axis: tuple[float, float, float] | None = None,
) -> ThreeFingerHandModel:
    joints = {}
    pads = {}
    finger_joints = {}
    for index, name in enumerate(("a", "b", "c")):
        joint_name = f"joint_{name}"
        link_name = f"link_{name}"
        finger_name = f"finger_{name}"
        pad_name = f"pad_{name}"
        joint_type = first_joint_type if index == 0 else "prismatic"
        axis = (
            (0.0, 0.0, 1.0)
            if joint_type == "revolute"
            else (1.0, 0.0, 0.0)
        )
        if index == 0 and first_axis is not None:
            axis = first_axis
        joints[joint_name] = JointSpec(
            name=joint_name,
            joint_type=joint_type,
            parent_link="hand_base",
            child_link=link_name,
            origin_xyz_m=(0.0, 0.0, 0.0),
            origin_rpy_rad=(0.0, 0.0, 0.0),
            axis=axis,
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
    return ThreeFingerHandModel(
        base_link="hand_base",
        joints=joints,
        joint_order=tuple(joints),
        finger_joint_names=finger_joints,
        pads=pads,
    )


def _backend(
    *, first_joint_type: str = "prismatic"
) -> DirectedIntervalKinematics:
    return DirectedIntervalKinematics(
        _three_finger_hand(first_joint_type=first_joint_type),
        IntervalArithmeticOptions(
            decimal_precision=80,
            maximum_root_bisection_iterations=256,
        ),
    )


def _positive_x_pad_triangle() -> np.ndarray:
    return np.asarray(
        ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )


def _positive_x_object_triangle(x_coordinate: float) -> np.ndarray:
    return np.asarray(
        (
            (x_coordinate, 0.0, 0.0),
            (x_coordinate, 1.0, 0.0),
            (x_coordinate, 0.0, 1.0),
        ),
        dtype=np.float64,
    )


def _assert_matrix_value_is_enclosed(
    interval_rows: tuple[tuple[IntervalBounds, ...], ...],
    exact_rows: np.ndarray,
) -> None:
    assert exact_rows.shape == (
        len(interval_rows),
        len(interval_rows[0]),
    )
    for row, interval_row in enumerate(interval_rows):
        for column, bounds in enumerate(interval_row):
            assert bounds.lower <= exact_rows[row, column] <= bounds.upper


def _mimic_three_finger_hand() -> ThreeFingerHandModel:
    joints = {
        "joint_a_source": JointSpec(
            name="joint_a_source",
            joint_type="revolute",
            parent_link="hand_base",
            child_link="link_a_mid",
            origin_xyz_m=(0.0, 0.0, 0.0),
            origin_rpy_rad=(0.0, 0.0, 0.0),
            axis=(0.0, 0.0, 1.0),
            limit=JointLimit(-1.0, 1.0),
        ),
        "joint_a_mimic": JointSpec(
            name="joint_a_mimic",
            joint_type="revolute",
            parent_link="link_a_mid",
            child_link="link_a",
            origin_xyz_m=(0.4, 0.0, 0.0),
            origin_rpy_rad=(0.0, 0.0, 0.0),
            axis=(0.0, 0.0, 1.0),
            limit=JointLimit(-2.0, 2.0),
            mimic=MimicSpec(
                source_joint="joint_a_source",
                multiplier=2.0,
                offset=0.1,
            ),
        ),
        "joint_b": JointSpec(
            name="joint_b",
            joint_type="prismatic",
            parent_link="hand_base",
            child_link="link_b",
            origin_xyz_m=(0.0, 0.0, 0.0),
            origin_rpy_rad=(0.0, 0.0, 0.0),
            axis=(1.0, 0.0, 0.0),
            limit=JointLimit(-1.0, 1.0),
        ),
        "joint_c": JointSpec(
            name="joint_c",
            joint_type="prismatic",
            parent_link="hand_base",
            child_link="link_c",
            origin_xyz_m=(0.0, 0.0, 0.0),
            origin_rpy_rad=(0.0, 0.0, 0.0),
            axis=(1.0, 0.0, 0.0),
            limit=JointLimit(-1.0, 1.0),
        ),
    }
    pads = {
        f"pad_{name}": PadGeometry(
            name=f"pad_{name}",
            finger_name=f"finger_{name}",
            link_name=f"link_{name}",
            origin_xyz_m=(0.0, 0.0, 0.0),
            origin_rpy_rad=(0.0, 0.0, 0.0),
            geometry=GeometrySpec("box", (1.0, 1.0, 1.0)),
        )
        for name in ("a", "b", "c")
    }
    return ThreeFingerHandModel(
        base_link="hand_base",
        joints=joints,
        joint_order=tuple(joints),
        finger_joint_names={
            "finger_a": ("joint_a_source", "joint_a_mimic"),
            "finger_b": ("joint_b",),
            "finger_c": ("joint_c",),
        },
        pads=pads,
    )


def test_joint_box_geometric_jacobian_encloses_exact_revolute_samples() -> None:
    backend = _backend(first_joint_type="revolute")
    base_transform = np.asarray(
        (
            (0.0, -1.0, 0.0, 0.2),
            (1.0, 0.0, 0.0, -0.1),
            (0.0, 0.0, 1.0, 0.3),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    joint_box = (
        IntervalBounds(-0.35, 0.45),
        IntervalBounds(0.0, 0.0),
        IntervalBounds(0.0, 0.0),
    )
    point_box = (
        IntervalBounds(-1.0, 1.0),
        IntervalBounds(-1.0, 1.0),
        IntervalBounds(-1.0, 1.0),
    )
    result = backend.geometric_jacobian_bounds(
        link_name="link_a",
        independent_joint_intervals=joint_box,
        point_object_m=point_box,
        base_transform=base_transform,
    )

    point_local = np.asarray((0.3, -0.2, 0.1), dtype=np.float64)
    for sample in np.linspace(joint_box[0].lower, joint_box[0].upper, 17):
        positions = np.asarray((sample, 0.0, 0.0), dtype=np.float64)
        exact = backend.hand_model.geometric_jacobian(
            "link_a",
            positions,
            point_local_m=point_local,
            base_transform=base_transform,
        )
        link_transform = backend.hand_model.forward_kinematics(
            positions, base_transform=base_transform
        )["link_a"]
        point_object = (
            link_transform[:3, :3] @ point_local + link_transform[:3, 3]
        )
        assert all(
            bounds.lower <= value <= bounds.upper
            for bounds, value in zip(point_box, point_object)
        )
        _assert_matrix_value_is_enclosed(result.elements, exact)

    assert result.method_id == INTERVAL_GEOMETRIC_JACOBIAN_METHOD_ID
    assert result.independent_joint_names == ("joint_a", "joint_b", "joint_c")
    assert result.as_dict()["method_id"] == INTERVAL_GEOMETRIC_JACOBIAN_METHOD_ID
    with pytest.raises(FrozenInstanceError):
        result.link_name = "forged"  # type: ignore[misc]


def test_joint_box_geometric_jacobian_encloses_prismatic_and_mimic_samples() -> None:
    prismatic = _backend(first_joint_type="prismatic")
    zero_box = (
        IntervalBounds(-0.2, 0.3),
        IntervalBounds(0.0, 0.0),
        IntervalBounds(0.0, 0.0),
    )
    point_box = tuple(IntervalBounds(-1.0, 1.0) for _axis in range(3))
    prismatic_result = prismatic.geometric_jacobian_bounds(
        link_name="link_a",
        independent_joint_intervals=zero_box,
        point_object_m=point_box,
        base_transform=np.eye(4),
    )
    for sample in (-0.2, 0.0, 0.3):
        exact = prismatic.hand_model.geometric_jacobian(
            "link_a", np.asarray((sample, 0.0, 0.0))
        )
        _assert_matrix_value_is_enclosed(prismatic_result.elements, exact)

    mimic_hand = _mimic_three_finger_hand()
    mimic_backend = DirectedIntervalKinematics(
        mimic_hand,
        IntervalArithmeticOptions(
            decimal_precision=80,
            maximum_root_bisection_iterations=256,
        ),
    )
    mimic_joint_box = (
        IntervalBounds(-0.2, 0.3),
        IntervalBounds(0.0, 0.0),
        IntervalBounds(0.0, 0.0),
    )
    mimic_result = mimic_backend.geometric_jacobian_bounds(
        link_name="link_a",
        independent_joint_intervals=mimic_joint_box,
        point_object_m=point_box,
        base_transform=np.eye(4),
    )
    for sample in np.linspace(-0.2, 0.3, 11):
        positions = np.asarray((sample, 0.0, 0.0), dtype=np.float64)
        exact = mimic_hand.geometric_jacobian(
            "link_a", positions, point_local_m=(0.2, 0.1, 0.0)
        )
        link_transform = mimic_hand.forward_kinematics(positions)["link_a"]
        point_object = (
            link_transform[:3, :3] @ np.asarray((0.2, 0.1, 0.0))
            + link_transform[:3, 3]
        )
        assert all(
            bounds.lower <= value <= bounds.upper
            for bounds, value in zip(point_box, point_object)
        )
        _assert_matrix_value_is_enclosed(mimic_result.elements, exact)


def test_joint_box_rigid_transform_encloses_every_endpoint_combination() -> None:
    backend = _backend(first_joint_type="revolute")
    base_transform = np.asarray(
        (
            (0.0, -1.0, 0.0, 0.2),
            (1.0, 0.0, 0.0, -0.1),
            (0.0, 0.0, 1.0, 0.3),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    joint_box = (
        IntervalBounds(-0.35, 0.45),
        IntervalBounds(-0.2, 0.3),
        IntervalBounds(-0.1, 0.4),
    )
    result = backend.link_transform_over_joint_box(
        link_name="link_a",
        independent_joint_intervals=joint_box,
        base_transform=base_transform,
    )

    for positions in product(
        *((bounds.lower, bounds.upper) for bounds in joint_box)
    ):
        exact = backend.hand_model.forward_kinematics(
            np.asarray(positions, dtype=np.float64),
            base_transform=base_transform,
        )["link_a"]
        _assert_matrix_value_is_enclosed(result.elements, exact[:3])

    assert result.method_id == INTERVAL_RIGID_TRANSFORM_METHOD_ID
    assert result.independent_joint_names == (
        "joint_a",
        "joint_b",
        "joint_c",
    )
    assert result.as_dict()["method_id"] == (
        INTERVAL_RIGID_TRANSFORM_METHOD_ID
    )
    with pytest.raises(FrozenInstanceError):
        result.link_name = "forged"  # type: ignore[misc]


def test_joint_box_rigid_transform_rejects_bad_shape_and_limits() -> None:
    backend = _backend(first_joint_type="revolute")
    with pytest.raises(IntervalKinematicsError, match="match independent joints"):
        backend.link_transform_over_joint_box(
            link_name="link_a",
            independent_joint_intervals=(IntervalBounds(0.0, 0.0),),
            base_transform=np.eye(4),
        )
    with pytest.raises(IntervalKinematicsError, match="limit contract"):
        backend.link_transform_over_joint_box(
            link_name="link_a",
            independent_joint_intervals=(
                IntervalBounds(-2.1, 0.0),
                IntervalBounds(0.0, 0.0),
                IntervalBounds(0.0, 0.0),
            ),
            base_transform=np.eye(4),
        )


def test_joint_box_geometric_jacobian_rejects_unbound_inputs() -> None:
    backend = _backend(first_joint_type="revolute")
    with pytest.raises(IntervalKinematicsError, match="match independent joints"):
        backend.geometric_jacobian_bounds(
            link_name="link_a",
            independent_joint_intervals=(IntervalBounds(0.0, 0.0),),
            point_object_m=tuple(
                IntervalBounds(0.0, 0.0) for _axis in range(3)
            ),
            base_transform=np.eye(4),
        )
    with pytest.raises(IntervalKinematicsError, match="three intervals"):
        backend.geometric_jacobian_bounds(
            link_name="link_a",
            independent_joint_intervals=tuple(
                IntervalBounds(0.0, 0.0) for _joint in range(3)
            ),
            point_object_m=(IntervalBounds(0.0, 0.0),),
            base_transform=np.eye(4),
        )
    with pytest.raises(IntervalKinematicsError, match="limit contract"):
        backend.geometric_jacobian_bounds(
            link_name="link_a",
            independent_joint_intervals=(
                IntervalBounds(1.9, 2.1),
                IntervalBounds(0.0, 0.0),
                IntervalBounds(0.0, 0.0),
            ),
            point_object_m=tuple(
                IntervalBounds(0.0, 0.0) for _axis in range(3)
            ),
            base_transform=np.eye(4),
        )


def test_complete_path_phase_boxes_enclose_every_joint_combination() -> None:
    backend = _backend(first_joint_type="prismatic")
    initial = np.asarray((0.1, -0.2, 0.3), dtype=np.float64)
    directions = np.asarray(
        (
            (0.5, 0.0, 0.0),
            (0.0, -0.25, 0.0),
            (0.0, 0.0, 0.75),
        ),
        dtype=np.float64,
    )
    phases = (
        IntervalBounds(0.2, 0.4),
        IntervalBounds(-0.1, 0.3),
        IntervalBounds(0.0, 0.2),
    )
    result = backend.independent_joint_box_from_paths(
        initial_independent_joint_positions=initial,
        directions=directions,
        phase_intervals=phases,
    )

    for values in product(
        *(
            (bounds.lower, 0.5 * (bounds.lower + bounds.upper), bounds.upper)
            for bounds in phases
        )
    ):
        exact = initial + np.asarray(values, dtype=np.float64) @ directions
        assert all(
            bounds.lower <= value <= bounds.upper
            for bounds, value in zip(result, exact)
        )

    with pytest.raises(IntervalKinematicsError, match="match independent joints"):
        backend.independent_joint_box_from_paths(
            initial_independent_joint_positions=initial,
            directions=((1.0, 0.0),),
            phase_intervals=(IntervalBounds(0.0, 1.0),),
        )


def test_linear_prismatic_crossing_has_strict_geometric_enclosure() -> None:
    backend = _backend()
    motion = backend.point_motion(
        link_name="link_a",
        q_start=np.zeros(3),
        direction=np.asarray((1.0, 0.0, 0.0)),
        phase_lower=0.2,
        phase_upper=0.4,
        base_transform=np.eye(4),
        point_local_m=np.asarray((0.0, 0.25, 0.25)),
    )
    predicates = backend.contact_predicates(
        link_name="link_a",
        q_start=np.zeros(3),
        direction=np.asarray((1.0, 0.0, 0.0)),
        phase_lower=0.2,
        phase_upper=0.4,
        base_transform=np.eye(4),
        witness_point_local_m=np.asarray((0.0, 0.25, 0.25)),
        pad_triangle_local_m=_positive_x_pad_triangle(),
        object_triangle_m=_positive_x_object_triangle(0.3),
    )

    assert predicates.method_id == METHOD_ID
    assert predicates.decimal_precision == 80
    assert motion.method_id == METHOD_ID
    assert motion.position_object_m[0].lower <= 0.2
    assert motion.position_object_m[0].upper >= 0.4
    assert motion.velocity_object_m_per_unit[0].strictly_positive
    assert motion.acceleration_object_m_per_unit_squared[0].contains_zero
    assert predicates.position_object_m[0].lower <= 0.2
    assert predicates.position_object_m[0].upper >= 0.4
    assert predicates.plane_value.lower < 0.0 < predicates.plane_value.upper
    assert predicates.plane_derivative.strictly_positive
    assert predicates.object_plane_transversality.strictly_positive
    assert predicates.plane_second_derivative.contains_zero
    assert all(
        edge.strictly_positive
        for edge in predicates.triangle_edge_halfspaces
    )
    assert predicates.pad_approach.strictly_positive
    left = backend.contact_predicates(
        link_name="link_a",
        q_start=np.zeros(3),
        direction=np.asarray((1.0, 0.0, 0.0)),
        phase_lower=0.2,
        phase_upper=0.2,
        base_transform=np.eye(4),
        witness_point_local_m=np.asarray((0.0, 0.25, 0.25)),
        pad_triangle_local_m=_positive_x_pad_triangle(),
        object_triangle_m=_positive_x_object_triangle(0.3),
    )
    right = backend.contact_predicates(
        link_name="link_a",
        q_start=np.zeros(3),
        direction=np.asarray((1.0, 0.0, 0.0)),
        phase_lower=0.4,
        phase_upper=0.4,
        base_transform=np.eye(4),
        witness_point_local_m=np.asarray((0.0, 0.25, 0.25)),
        pad_triangle_local_m=_positive_x_pad_triangle(),
        object_triangle_m=_positive_x_object_triangle(0.3),
    )
    assert left.plane_value.strictly_negative
    assert right.plane_value.strictly_positive

    motion_oriented_root = backend.certify_transverse_contact_root(
        link_name="link_a",
        q_start=np.zeros(3),
        direction=np.asarray((1.0, 0.0, 0.0)),
        phase_lower=0.2,
        phase_upper=0.4,
        base_transform=np.eye(4),
        witness_point_local_m=np.asarray((0.0, 0.25, 0.25)),
        pad_triangle_local_m=_positive_x_pad_triangle(),
        object_triangle_m=_positive_x_object_triangle(0.3),
    )
    assert motion_oriented_root.state is (
        IntervalRootState.CERTIFIED_PAD_DIRECTIONAL_TRANSVERSE_ROOT
    )
    certificate = motion_oriented_root.certificate
    assert certificate is not None
    assert certificate.path_local_free_side_approach.strictly_positive
    assert certificate.object_source_winding_free_side_sign == -1

    source_triangle = _positive_x_object_triangle(0.3)
    flipped_triangle = source_triangle[[0, 2, 1]]
    flipped_root = backend.certify_transverse_contact_root(
        link_name="link_a",
        q_start=np.zeros(3),
        direction=np.asarray((1.0, 0.0, 0.0)),
        phase_lower=0.2,
        phase_upper=0.4,
        base_transform=np.eye(4),
        witness_point_local_m=np.asarray((0.0, 0.25, 0.25)),
        pad_triangle_local_m=_positive_x_pad_triangle(),
        object_triangle_m=flipped_triangle,
    )
    assert flipped_root.state is (
        IntervalRootState.CERTIFIED_PAD_DIRECTIONAL_TRANSVERSE_ROOT
    )
    flipped_certificate = flipped_root.certificate
    assert flipped_certificate is not None
    assert flipped_certificate.object_source_winding_free_side_sign == 1
    source_normal = np.cross(
        source_triangle[1] - source_triangle[0],
        source_triangle[2] - source_triangle[0],
    )
    source_normal /= np.linalg.norm(source_normal)
    flipped_normal = np.cross(
        flipped_triangle[1] - flipped_triangle[0],
        flipped_triangle[2] - flipped_triangle[0],
    )
    flipped_normal /= np.linalg.norm(flipped_normal)
    source_motion_opposing = (
        certificate.object_source_winding_free_side_sign * source_normal
    )
    flipped_motion_opposing = (
        flipped_certificate.object_source_winding_free_side_sign
        * flipped_normal
    )
    assert np.array_equal(source_motion_opposing, flipped_motion_opposing)

    with pytest.raises(IntervalKinematicsError, match="lacks"):
        IntervalRootClassification(
            state=(
                IntervalRootState.CERTIFIED_PAD_DIRECTION_REJECTED_TRANSVERSE_ROOT
            ),
            searched_phase=motion_oriented_root.searched_phase,
            certificate=certificate,
            reason="FORGED_DIRECTION_TEST",
        )
    with pytest.raises(IntervalKinematicsError, match="outside"):
        IntervalRootClassification(
            state=(
                IntervalRootState.CERTIFIED_PAD_DIRECTIONAL_TRANSVERSE_ROOT
            ),
            searched_phase=IntervalBounds(0.0, 0.1),
            certificate=certificate,
            reason="FORGED_PHASE_TEST",
        )


def test_batch_point_motion_contains_every_scalar_high_precision_result() -> None:
    backend = _backend(first_joint_type="revolute")
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (0.25, -0.125, 0.5),
            (-0.75, 0.625, -0.25),
            (1.0e-12, -2.0e-12, 3.0e-12),
            (-1.25, -0.875, 0.375),
        ),
        dtype=np.float64,
    )
    arguments = {
        "link_name": "link_a",
        "q_start": np.asarray((0.15, -0.2, 0.1), dtype=np.float64),
        "direction": np.asarray((0.8, 0.0, 0.0), dtype=np.float64),
        "phase_lower": -0.25,
        "phase_upper": 0.45,
        "base_transform": np.eye(4),
    }
    transform_cache = backend.new_link_transform_cache()
    batch = backend.point_motion_many(
        **arguments,
        points_local_m=points,
        transform_cache=transform_cache,
    )

    batch_pairs = (
        (
            batch.position_lower_object_m,
            batch.position_upper_object_m,
            "position_object_m",
        ),
        (
            batch.velocity_lower_object_m_per_unit,
            batch.velocity_upper_object_m_per_unit,
            "velocity_object_m_per_unit",
        ),
        (
            batch.acceleration_lower_object_m_per_unit_squared,
            batch.acceleration_upper_object_m_per_unit_squared,
            "acceleration_object_m_per_unit_squared",
        ),
    )
    for point_index, point in enumerate(points):
        scalar = backend.point_motion(
            **arguments,
            point_local_m=point,
            transform_cache=transform_cache,
        )
        for batch_lower, batch_upper, scalar_field in batch_pairs:
            scalar_bounds = getattr(scalar, scalar_field)
            for axis, bounds in enumerate(scalar_bounds):
                assert batch_lower[point_index, axis] <= bounds.lower
                assert batch_upper[point_index, axis] >= bounds.upper

    assert transform_cache.miss_count == 1
    assert transform_cache.hit_count == len(points)
    assert not batch.position_lower_object_m.flags.writeable
    assert not batch.acceleration_upper_object_m_per_unit_squared.flags.writeable

    with pytest.raises(IntervalKinematicsError, match="non-empty shape"):
        backend.point_motion_many(
            **arguments,
            points_local_m=np.empty((0, 3), dtype=np.float64),
        )


def test_batch_point_velocity_vector_matches_motion_and_encloses_samples() -> None:
    backend = _backend(first_joint_type="revolute")
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (0.25, -0.125, 0.5),
            (-0.75, 0.625, -0.25),
        ),
        dtype=np.float64,
    )
    vectors = np.asarray(
        (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, -1.0),
        ),
        dtype=np.float64,
    )
    arguments = {
        "link_name": "link_a",
        "q_start": np.asarray((0.15, -0.2, 0.1), dtype=np.float64),
        "direction": np.asarray((0.8, 0.0, 0.0), dtype=np.float64),
        "phase_lower": -0.25,
        "phase_upper": 0.45,
        "base_transform": np.eye(4),
    }
    complete = backend.point_motion_many(
        **arguments,
        points_local_m=points,
    )
    directional = backend.point_velocity_and_vector_many(
        **arguments,
        points_local_m=points,
        vectors_local=vectors,
    )

    assert np.array_equal(
        directional.point_velocity_lower_object_m_per_unit,
        complete.velocity_lower_object_m_per_unit,
    )
    assert np.array_equal(
        directional.point_velocity_upper_object_m_per_unit,
        complete.velocity_upper_object_m_per_unit,
    )
    for phase in np.linspace(-0.25, 0.45, 9):
        q = arguments["q_start"] + phase * arguments["direction"]
        rotation = backend.hand_model.forward_kinematics(
            q, base_transform=arguments["base_transform"]
        )["link_a"][:3, :3]
        exact_vectors = vectors @ rotation.T
        assert np.all(
            directional.vector_lower_object <= exact_vectors
        )
        assert np.all(
            exact_vectors <= directional.vector_upper_object
        )
    assert not directional.vector_lower_object.flags.writeable
    assert not (
        directional.point_velocity_upper_object_m_per_unit.flags.writeable
    )

    with pytest.raises(IntervalKinematicsError, match="aligned finite"):
        backend.point_velocity_and_vector_many(
            **arguments,
            points_local_m=points,
            vectors_local=vectors[:2],
        )


def test_shared_plane_root_finalization_matches_scalar_triangle_states(
    monkeypatch,
) -> None:
    backend = _backend(first_joint_type="prismatic")

    def assert_same_except_contraction_provenance(
        grouped: IntervalRootClassification,
        scalar: IntervalRootClassification,
    ) -> None:
        assert grouped.state is scalar.state
        assert grouped.searched_phase == scalar.searched_phase
        assert grouped.reason == scalar.reason
        assert (grouped.certificate is None) == (scalar.certificate is None)
        if scalar.certificate is None:
            return
        assert grouped.certificate is not None
        grouped_certificate = grouped.certificate
        scalar_certificate = scalar.certificate

        def assert_overlap(first, second) -> None:
            assert max(first.lower, second.lower) <= min(
                first.upper, second.upper
            )
            assert first.strictly_positive == second.strictly_positive
            assert first.strictly_negative == second.strictly_negative

        grouped_root = grouped_certificate.implicit_root
        scalar_root = scalar_certificate.implicit_root
        assert grouped_root.method_id == scalar_root.method_id
        assert grouped_root.equation_sha256 == scalar_root.equation_sha256
        assert (
            grouped_root.feature_identity_sha256
            == scalar_root.feature_identity_sha256
        )
        assert grouped_root.feature_type == scalar_root.feature_type
        assert grouped_root.isolating_interval == scalar_root.isolating_interval
        assert grouped_root.uniqueness_proven is scalar_root.uniqueness_proven
        assert (
            grouped_root.display_approximation
            == scalar_root.display_approximation
        )
        assert (
            grouped_root.display_approximation_role
            == scalar_root.display_approximation_role
        )
        assert_overlap(grouped_root.value_at_lower, scalar_root.value_at_lower)
        assert_overlap(grouped_root.value_at_upper, scalar_root.value_at_upper)
        assert_overlap(grouped_root.derivative, scalar_root.derivative)
        for grouped_edge, scalar_edge in zip(
            grouped_certificate.triangle_edge_halfspaces,
            scalar_certificate.triangle_edge_halfspaces,
        ):
            assert_overlap(grouped_edge, scalar_edge)
        assert_overlap(
            grouped_certificate.pad_approach,
            scalar_certificate.pad_approach,
        )
        assert_overlap(
            grouped_certificate.path_local_free_side_approach,
            scalar_certificate.path_local_free_side_approach,
        )
        for grouped_position, scalar_position in zip(
            grouped_certificate.position_object_m,
            scalar_certificate.position_object_m,
        ):
            assert_overlap(grouped_position, scalar_position)
        assert (
            grouped_certificate.object_source_winding_free_side_sign
            == scalar_certificate.object_source_winding_free_side_sign
        )
        assert grouped_certificate.method_id == scalar_certificate.method_id
        assert (
            grouped_certificate.decimal_precision
            == scalar_certificate.decimal_precision
        )
        assert (
            grouped_certificate.bisection_iterations
            <= scalar_certificate.bisection_iterations
        )

    interior_arguments = {
        "link_name": "link_a",
        "q_start": np.zeros(3),
        "direction": np.asarray((1.0, 0.0, 0.0)),
        "phase_lower": 0.2,
        "phase_upper": 0.4,
        "base_transform": np.eye(4),
        "witness_point_local_m": np.asarray((0.0, 0.25, 0.25)),
        "pad_triangle_local_m": _positive_x_pad_triangle(),
    }
    representative = _positive_x_object_triangle(0.3)
    plane = backend.certify_transverse_plane_root(
        **interior_arguments,
        object_triangle_m=representative,
    )
    grouped_interior = backend.finalize_transverse_plane_root_for_triangle(
        plane_classification=plane,
        **interior_arguments,
        object_triangle_m=representative,
        representative_object_triangle_m=representative,
    )
    scalar_interior = backend.certify_transverse_contact_root(
        **interior_arguments,
        object_triangle_m=representative,
    )
    assert_same_except_contraction_provenance(
        grouped_interior, scalar_interior
    )
    assert plane.root is not None
    assert (
        plane.root.interpolation_iterations
        + plane.root.interval_newton_iterations
        >= 1
    )
    assert (
        plane.root.bisection_iterations
        <= backend.options.maximum_root_bisection_iterations
    )
    assert backend.compiled_root_transaction_count >= 1
    reference_plane = backend.certify_transverse_plane_root(
        **interior_arguments,
        object_triangle_m=representative,
        _use_interval_newton_contraction=False,
        _use_nominal_root_seed=False,
    )
    assert reference_plane.root is not None
    assert reference_plane.root.interpolation_iterations == 0
    assert reference_plane.root.interval_newton_iterations == 0
    assert (
        plane.root.isolating_interval
        == reference_plane.root.isolating_interval
    )
    assert (
        plane.root.bisection_iterations
        <= reference_plane.root.bisection_iterations
    )

    outside_triangle = representative + np.asarray((0.0, 2.0, 0.0))
    grouped_outside = backend.finalize_transverse_plane_root_for_triangle(
        plane_classification=plane,
        **interior_arguments,
        object_triangle_m=outside_triangle,
        representative_object_triangle_m=representative,
    )
    scalar_outside = backend.certify_transverse_contact_root(
        **interior_arguments,
        object_triangle_m=outside_triangle,
    )
    assert grouped_outside.state is scalar_outside.state
    assert grouped_outside.state is IntervalRootState.CERTIFIED_FREE

    boundary_arguments = dict(interior_arguments)
    boundary_arguments["witness_point_local_m"] = np.asarray(
        (0.0, 0.0, 0.25)
    )
    boundary_plane = backend.certify_transverse_plane_root(
        **boundary_arguments,
        object_triangle_m=representative,
    )
    grouped_boundary = backend.finalize_transverse_plane_root_for_triangle(
        plane_classification=boundary_plane,
        **boundary_arguments,
        object_triangle_m=representative,
        representative_object_triangle_m=representative,
    )
    scalar_boundary = backend.certify_transverse_contact_root(
        **boundary_arguments,
        object_triangle_m=representative,
    )
    assert_same_except_contraction_provenance(
        grouped_boundary, scalar_boundary
    )
    assert grouped_boundary.state is IntervalRootState.UNRESOLVED

    reverse_arguments = dict(interior_arguments)
    reverse_arguments.update(
        {
            "q_start": np.asarray((0.5, 0.0, 0.0)),
            "direction": np.asarray((-1.0, 0.0, 0.0)),
            "phase_lower": 0.1,
            "phase_upper": 0.3,
        }
    )
    reverse_plane = backend.certify_transverse_plane_root(
        **reverse_arguments,
        object_triangle_m=representative,
    )
    grouped_reverse = backend.finalize_transverse_plane_root_for_triangle(
        plane_classification=reverse_plane,
        **reverse_arguments,
        object_triangle_m=representative,
        representative_object_triangle_m=representative,
    )
    scalar_reverse = backend.certify_transverse_contact_root(
        **reverse_arguments,
        object_triangle_m=representative,
    )
    assert_same_except_contraction_provenance(
        grouped_reverse, scalar_reverse
    )
    assert grouped_reverse.state is (
        IntervalRootState.CERTIFIED_PAD_DIRECTION_REJECTED_TRANSVERSE_ROOT
    )

    opposite_winding = representative[[0, 2, 1]]
    grouped_opposite = backend.finalize_transverse_plane_root_for_triangle(
        plane_classification=plane,
        **interior_arguments,
        object_triangle_m=opposite_winding,
        representative_object_triangle_m=representative,
    )
    scalar_opposite = backend.certify_transverse_contact_root(
        **interior_arguments,
        object_triangle_m=opposite_winding,
    )
    assert_same_except_contraction_provenance(
        grouped_opposite, scalar_opposite
    )

    different_plane = representative.copy()
    different_plane[0, 0] += 0.01
    rejected_reuse = backend.finalize_transverse_plane_root_for_triangle(
        plane_classification=plane,
        **interior_arguments,
        object_triangle_m=different_plane,
        representative_object_triangle_m=representative,
    )
    assert rejected_reuse.state is IntervalRootState.UNRESOLVED
    assert rejected_reuse.reason == (
        "REUSED_EXACT_PLANE_ROOT_NOT_CERTIFIED_FOR_TRIANGLE_PLANE"
    )

    assert plane.root is not None
    uncertain_pad_plane = replace(
        plane,
        root=replace(
            plane.root,
            pad_approach=IntervalBounds(-1.0, 1.0),
        ),
    )
    local_pad_calls: list[tuple[float, float]] = []
    original_pad_approach = backend.contact_pad_approach

    def counted_pad_approach(**arguments):
        local_pad_calls.append(
            (arguments["phase_lower"], arguments["phase_upper"])
        )
        return original_pad_approach(**arguments)

    monkeypatch.setattr(
        backend, "contact_pad_approach", counted_pad_approach
    )
    fallback_direction = backend.finalize_transverse_plane_root_for_triangle(
        plane_classification=uncertain_pad_plane,
        **interior_arguments,
        object_triangle_m=representative,
        representative_object_triangle_m=representative,
    )
    assert fallback_direction.state is grouped_interior.state
    assert local_pad_calls == [
        (
            plane.root.isolating_interval.lower,
            plane.root.isolating_interval.upper,
        )
    ]


def test_exact_link_transform_cache_reuses_only_identical_inputs() -> None:
    backend = _backend()
    common = {
        "link_name": "link_a",
        "q_start": np.zeros(3),
        "direction": np.asarray((1.0, 0.0, 0.0)),
        "phase_lower": 0.2,
        "phase_upper": 0.4,
        "base_transform": np.eye(4),
    }
    cache = backend.new_link_transform_cache()
    backend.point_motion(
        **common,
        point_local_m=np.asarray((0.0, 0.25, 0.25)),
        transform_cache=cache,
    )
    cached_second = backend.point_motion(
        **common,
        point_local_m=np.asarray((0.0, 0.5, 0.25)),
        transform_cache=cache,
    )
    uncached_second = backend.point_motion(
        **common,
        point_local_m=np.asarray((0.0, 0.5, 0.25)),
    )

    assert cached_second == uncached_second
    assert cache.miss_count == 1
    assert cache.hit_count == 1
    assert cache.entry_count == 1
    assert cache.point_miss_count == 2
    assert cache.point_hit_count == 0
    assert cache.point_entry_count == 2

    cached_predicates = backend.contact_predicates(
        **common,
        witness_point_local_m=np.asarray((0.0, 0.25, 0.25)),
        pad_triangle_local_m=_positive_x_pad_triangle(),
        object_triangle_m=_positive_x_object_triangle(0.3),
        transform_cache=cache,
    )
    uncached_predicates = backend.contact_predicates(
        **common,
        witness_point_local_m=np.asarray((0.0, 0.25, 0.25)),
        pad_triangle_local_m=_positive_x_pad_triangle(),
        object_triangle_m=_positive_x_object_triangle(0.3),
    )
    assert cached_predicates == uncached_predicates
    assert cache.miss_count == 1
    assert cache.hit_count == 2
    assert cache.point_hit_count == 1
    assert cache.point_miss_count == 2
    assert cache.pad_area_hit_count == 0
    assert cache.pad_area_miss_count == 1
    assert cache.pad_area_entry_count == 1

    cached_other_face = backend.contact_predicates(
        **common,
        witness_point_local_m=np.asarray((0.0, 0.25, 0.25)),
        pad_triangle_local_m=_positive_x_pad_triangle(),
        object_triangle_m=_positive_x_object_triangle(0.35),
        transform_cache=cache,
    )
    uncached_other_face = backend.contact_predicates(
        **common,
        witness_point_local_m=np.asarray((0.0, 0.25, 0.25)),
        pad_triangle_local_m=_positive_x_pad_triangle(),
        object_triangle_m=_positive_x_object_triangle(0.35),
    )
    assert cached_other_face == uncached_other_face
    assert cache.hit_count == 3
    assert cache.point_hit_count == 2
    assert cache.pad_area_hit_count == 1
    assert cache.point_entry_count == 2
    assert cache.pad_area_entry_count == 1

    changed_pad = np.array(_positive_x_pad_triangle(), copy=True)
    changed_pad[2, 2] += 0.125
    cached_changed_pad = backend.contact_predicates(
        **common,
        witness_point_local_m=np.asarray((0.0, 0.25, 0.25)),
        pad_triangle_local_m=changed_pad,
        object_triangle_m=_positive_x_object_triangle(0.3),
        transform_cache=cache,
    )
    uncached_changed_pad = backend.contact_predicates(
        **common,
        witness_point_local_m=np.asarray((0.0, 0.25, 0.25)),
        pad_triangle_local_m=changed_pad,
        object_triangle_m=_positive_x_object_triangle(0.3),
    )
    assert cached_changed_pad == uncached_changed_pad
    assert cache.point_hit_count == 3
    assert cache.pad_area_miss_count == 2
    assert cache.pad_area_entry_count == 2

    changed_phase = backend.point_motion(
        **{
            **common,
            "phase_lower": 0.25,
            "phase_upper": 0.4,
        },
        point_local_m=np.asarray((0.0, 0.25, 0.25)),
        transform_cache=cache,
    )
    uncached_changed_phase = backend.point_motion(
        **{
            **common,
            "phase_lower": 0.25,
            "phase_upper": 0.4,
        },
        point_local_m=np.asarray((0.0, 0.25, 0.25)),
    )
    assert changed_phase == uncached_changed_phase
    assert cache.miss_count == 2
    assert cache.entry_count == 2
    assert cache.nonprimary_transform_bypass_count == 0
    assert cache.point_nonprimary_bypass_count == 0
    assert cache.point_entry_count == 3

    foreign_cache = _backend().new_link_transform_cache()
    with pytest.raises(IntervalKinematicsError, match="another interval backend"):
        backend.point_motion(
            **common,
            point_local_m=np.asarray((0.0, 0.25, 0.25)),
            transform_cache=foreign_cache,
        )


def test_nominal_root_seed_reduces_exact_bisection_without_changing_root() -> None:
    backend = _backend()
    arguments = {
        "link_name": "link_a",
        "q_start": np.zeros(3),
        "direction": np.asarray((3.0, 0.0, 0.0)),
        "phase_lower": 0.0,
        "phase_upper": 0.5,
        "base_transform": np.eye(4),
        "witness_point_local_m": np.asarray((0.0, 0.25, 0.25)),
        "pad_triangle_local_m": _positive_x_pad_triangle(),
        "object_triangle_m": _positive_x_object_triangle(1.0),
        "_use_interval_newton_contraction": False,
        "_use_compiled_root_transaction": False,
    }

    seeded = backend.certify_transverse_plane_root(**arguments)
    reference = backend.certify_transverse_plane_root(
        **arguments,
        _use_nominal_root_seed=False,
    )

    assert seeded.root is not None
    assert reference.root is not None
    assert seeded.root.isolating_interval == reference.root.isolating_interval
    assert seeded.root.value_at_lower == reference.root.value_at_lower
    assert seeded.root.value_at_upper == reference.root.value_at_upper
    assert (
        seeded.root.bisection_iterations
        < reference.root.bisection_iterations
    )


def test_precertified_plane_gate_reuses_rigorous_motion_without_changing_root() -> None:
    backend = _backend()
    common = {
        "link_name": "link_a",
        "q_start": np.zeros(3),
        "direction": np.asarray((3.0, 0.0, 0.0)),
        "base_transform": np.eye(4),
        "witness_point_local_m": np.asarray((0.0, 0.25, 0.25)),
        "object_triangle_m": _positive_x_object_triangle(1.0),
    }
    arguments = {
        **common,
        "phase_lower": 0.0,
        "phase_upper": 0.5,
        "pad_triangle_local_m": _positive_x_pad_triangle(),
    }
    reference = backend.certify_transverse_plane_root(**arguments)
    whole = backend.contact_plane_motion(
        **common,
        phase_lower=0.0,
        phase_upper=0.5,
    )
    lower_value = backend.contact_plane_value(
        **common,
        phase_lower=0.0,
        phase_upper=0.0,
    )
    upper_value = backend.contact_plane_value(
        **common,
        phase_lower=0.5,
        phase_upper=0.5,
    )

    reused = backend.certify_transverse_plane_root(
        **arguments,
        _precertified_plane_derivative=whole.plane_derivative,
        _precertified_lower_value=lower_value,
        _precertified_upper_value=upper_value,
    )

    assert reused == reference
    with pytest.raises(
        IntervalKinematicsError,
        match="requires derivative and both endpoint",
    ):
        backend.certify_transverse_plane_root(
            **arguments,
            _precertified_plane_derivative=whole.plane_derivative,
        )


def test_nominal_root_seed_cannot_override_exact_endpoint_signs(
    monkeypatch,
) -> None:
    backend = _backend()
    arguments = {
        "link_name": "link_a",
        "q_start": np.zeros(3),
        "direction": np.asarray((3.0, 0.0, 0.0)),
        "phase_lower": 0.0,
        "phase_upper": 0.5,
        "base_transform": np.eye(4),
        "witness_point_local_m": np.asarray((0.0, 0.25, 0.25)),
        "pad_triangle_local_m": _positive_x_pad_triangle(),
        "object_triangle_m": _positive_x_object_triangle(1.0),
    }
    reference = backend.certify_transverse_plane_root(
        **arguments,
        _use_nominal_root_seed=False,
    )
    monkeypatch.setattr(
        backend,
        "_nominal_root_seed_bracket",
        lambda **_arguments: (0.1, 0.2),
    )

    result = backend.certify_transverse_plane_root(
        **arguments,
        _use_compiled_root_transaction=False,
    )

    assert result == reference


def test_inward_prismatic_crossing_has_unique_transverse_root() -> None:
    backend = _backend()
    negative_x_pad = np.asarray(
        ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
        dtype=np.float64,
    )
    result = backend.certify_transverse_contact_root(
        link_name="link_a",
        q_start=np.asarray((0.5, 0.0, 0.0)),
        direction=np.asarray((-1.0, 0.0, 0.0)),
        phase_lower=0.0,
        phase_upper=0.4,
        base_transform=np.eye(4),
        witness_point_local_m=np.asarray((0.0, 0.25, 0.25)),
        pad_triangle_local_m=negative_x_pad,
        object_triangle_m=_positive_x_object_triangle(0.3),
    )

    assert result.state is (
        IntervalRootState.CERTIFIED_PAD_DIRECTIONAL_TRANSVERSE_ROOT
    )
    assert result.reason == (
        "STRICT_IVT_MONOTONE_INTERIOR_PAD_DIRECTIONAL_TRANSVERSE_ROOT"
    )
    certificate = result.certificate
    assert certificate is not None
    assert certificate.phase.lower <= 0.2 <= certificate.phase.upper
    assert certificate.plane_value_at_lower.strictly_positive
    assert certificate.plane_value_at_upper.strictly_negative
    assert certificate.plane_derivative.strictly_negative
    assert certificate.path_local_free_side_approach.strictly_positive
    assert certificate.object_source_winding_free_side_sign == 1
    assert certificate.pad_approach.strictly_positive
    assert all(
        edge.strictly_positive
        for edge in certificate.triangle_edge_halfspaces
    )


def test_plane_value_only_matches_full_predicates_exactly() -> None:
    backend = _backend()
    common = {
        "link_name": "link_a",
        "q_start": np.zeros(3),
        "direction": np.asarray((3.0, 0.0, 0.0)),
        "base_transform": np.eye(4),
        "witness_point_local_m": np.asarray((0.0, 0.25, 0.25)),
        "object_triangle_m": _positive_x_object_triangle(1.0),
    }
    for lower, upper in ((0.0, 0.5), (1.0 / 3.0, 1.0 / 3.0)):
        full = backend.contact_predicates(
            **common,
            phase_lower=lower,
            phase_upper=upper,
            pad_triangle_local_m=_positive_x_pad_triangle(),
        )
        plane_only = backend.contact_plane_value(
            **common,
            phase_lower=lower,
            phase_upper=upper,
        )
        assert plane_only == full.plane_value

    revolute = _backend(first_joint_type="revolute")
    revolute_common = {
        "link_name": "link_a",
        "q_start": np.asarray((-0.5, 0.0, 0.0)),
        "direction": np.asarray((1.0, 0.0, 0.0)),
        "base_transform": np.eye(4),
        "witness_point_local_m": np.asarray((1.0, 0.0, 0.0)),
        "object_triangle_m": np.asarray(
            ((1.0, -2.0, -2.0), (1.0, 2.0, -2.0), (1.0, 0.0, 2.0)),
            dtype=np.float64,
        ),
    }
    for lower, upper in ((0.0, 1.0), (0.125, 0.125)):
        full = revolute.contact_predicates(
            **revolute_common,
            phase_lower=lower,
            phase_upper=upper,
            pad_triangle_local_m=_positive_x_pad_triangle(),
        )
        plane_only = revolute.contact_plane_value(
            **revolute_common,
            phase_lower=lower,
            phase_upper=upper,
        )
        assert plane_only == full.plane_value


def test_value_only_link_transform_cache_reuses_identical_motion() -> None:
    backend = _backend(first_joint_type="revolute")
    common = {
        "link_name": "link_a",
        "q_start": np.asarray((-0.5, 0.0, 0.0)),
        "direction": np.asarray((1.0, 0.0, 0.0)),
        "phase_lower": 0.125,
        "phase_upper": 0.125,
        "base_transform": np.eye(4),
        "witness_point_local_m": np.asarray((1.0, 0.0, 0.0)),
    }
    first_triangle = np.asarray(
        ((1.0, -2.0, -2.0), (1.0, 2.0, -2.0), (1.0, 0.0, 2.0)),
        dtype=np.float64,
    )
    second_triangle = np.array(first_triangle, copy=True)
    second_triangle[:, 0] += 0.25
    cache = backend.new_link_transform_cache()

    backend.contact_plane_value(
        **common,
        object_triangle_m=first_triangle,
        transform_cache=cache,
    )
    cached_second = backend.contact_plane_value(
        **common,
        object_triangle_m=second_triangle,
        transform_cache=cache,
    )
    uncached_second = backend.contact_plane_value(
        **common,
        object_triangle_m=second_triangle,
    )

    assert cached_second == uncached_second
    assert cache.miss_count == 1
    assert cache.hit_count == 1
    assert cache.entry_count == 1

    foreign_cache = _backend().new_link_transform_cache()
    with pytest.raises(IntervalKinematicsError, match="another interval backend"):
        backend.contact_plane_value(
            **common,
            object_triangle_m=first_triangle,
            transform_cache=foreign_cache,
        )


def test_static_joint_origins_are_precompiled_for_value_and_jet_paths(
    monkeypatch,
) -> None:
    backend = _backend(first_joint_type="revolute")

    def forbidden_rebuild(*_arguments, **_keywords):
        raise AssertionError("static joint origin was rebuilt")

    monkeypatch.setattr(backend, "_rpy_rotation", forbidden_rebuild)
    monkeypatch.setattr(backend, "_value_rpy_rotation", forbidden_rebuild)
    common = {
        "link_name": "link_a",
        "q_start": np.asarray((-0.5, 0.0, 0.0)),
        "direction": np.asarray((1.0, 0.0, 0.0)),
        "phase_lower": 0.125,
        "phase_upper": 0.25,
        "base_transform": np.eye(4),
    }
    witness = np.asarray((1.0, 0.0, 0.0))

    motion = backend.point_motion(**common, point_local_m=witness)
    plane = backend.contact_plane_value(
        **common,
        witness_point_local_m=witness,
        object_triangle_m=_positive_x_object_triangle(1.0),
    )

    assert motion.phase == IntervalBounds(0.125, 0.25)
    assert isinstance(plane, IntervalBounds)


def test_precompiled_object_plane_is_exactly_triangle_bound() -> None:
    backend = _backend(first_joint_type="revolute")
    first_triangle = _positive_x_object_triangle(1.0)
    second_triangle = _positive_x_object_triangle(1.25)
    plane_data = backend._object_plane_value_data(first_triangle)
    common = {
        "link_name": "link_a",
        "q_start": np.asarray((-0.5, 0.0, 0.0)),
        "direction": np.asarray((1.0, 0.0, 0.0)),
        "phase_lower": 0.125,
        "phase_upper": 0.25,
        "base_transform": np.eye(4),
        "witness_point_local_m": np.asarray((1.0, 0.0, 0.0)),
        "object_triangle_m": first_triangle,
    }

    compiled = backend.contact_plane_value(
        **common,
        _object_plane_value_data=plane_data,
    )
    direct = backend.contact_plane_value(**common)

    assert compiled == direct
    with pytest.raises(IntervalKinematicsError, match="differs"):
        backend.contact_plane_value(
            **{
                **common,
                "object_triangle_m": second_triangle,
            },
            _object_plane_value_data=plane_data,
        )


def test_value_only_multiphase_cache_reuses_and_has_a_hard_capacity() -> None:
    backend = _backend()
    cache = backend.new_link_transform_cache()
    common = {
        "link_name": "link_a",
        "q_start": np.zeros(3),
        "direction": np.asarray((1.0, 0.0, 0.0)),
        "base_transform": np.eye(4),
        "witness_point_local_m": np.asarray((0.0, 0.25, 0.25)),
        "object_triangle_m": _positive_x_object_triangle(0.75),
        "transform_cache": cache,
    }
    backend.contact_plane_value(
        **common,
        phase_lower=0.0,
        phase_upper=1.0,
    )
    phase_count = MULTIPHASE_TRANSFORM_CACHE_CAPACITY + 8
    for index in range(phase_count):
        phase = (index + 1.0) / float(phase_count + 2)
        backend.contact_plane_value(
            **common,
            phase_lower=phase,
            phase_upper=phase,
        )

    final_phase = phase_count / float(phase_count + 2)
    hits_before = cache.hit_count
    backend.contact_plane_value(
        **common,
        phase_lower=final_phase,
        phase_upper=final_phase,
    )

    assert cache.entry_count == MULTIPHASE_TRANSFORM_CACHE_CAPACITY
    assert cache.transform_eviction_count == 9
    assert cache.hit_count == hits_before + 1
    assert cache.nonprimary_transform_bypass_count == 0


@pytest.mark.parametrize(
    "axis",
    (
        (0.0, 0.0, 1.0),
        (0.0, -1.0, 0.0),
        (math.sqrt(0.5), math.sqrt(0.5), 0.0),
    ),
)
def test_compiled_exact_point_plane_matches_mpmath_revolute_paths(
    axis,
) -> None:
    backend = DirectedIntervalKinematics(
        _three_finger_hand(
            first_joint_type="revolute",
            first_axis=axis,
        ),
        IntervalArithmeticOptions(
            decimal_precision=80,
            maximum_root_bisection_iterations=256,
        ),
    )
    common = {
        "link_name": "link_a",
        "q_start": np.asarray((-0.4, 0.0, 0.0)),
        "direction": np.asarray((0.8, 0.0, 0.0)),
        "base_transform": np.eye(4),
        "witness_point_local_m": np.asarray((0.7, -0.2, 0.4)),
        "object_triangle_m": np.asarray(
            ((0.3, -1.0, -1.0), (0.3, 1.0, -1.0), (0.3, 0.0, 1.0)),
            dtype=np.float64,
        ),
    }
    binding = backend._new_compiled_point_plane_binding(**common)
    if binding is None:
        pytest.skip(backend.compiled_point_backend_failure_reason)

    for phase in (0.0, 0.13, 0.5, 0.91, 1.0):
        reference = backend.contact_plane_value(
            **common,
            phase_lower=phase,
            phase_upper=phase,
        )
        compiled = backend.contact_plane_value(
            **common,
            phase_lower=phase,
            phase_upper=phase,
            _compiled_point_evaluator=binding,
        )
        assert max(reference.lower, compiled.lower) <= min(
            reference.upper, compiled.upper
        )

    assert backend.compiled_point_backend_status == "ACTIVE"
    assert backend.compiled_point_evaluation_count == 5

    compiled_plane, compiled_positions = binding.evaluator.evaluate_interval(
        0.13,
        float(np.nextafter(0.13, 1.0)),
    )
    reference_plane = backend.contact_plane_value(
        **common,
        phase_lower=0.13,
        phase_upper=float(np.nextafter(0.13, 1.0)),
    )
    reference_motion = backend.point_motion(
        link_name=common["link_name"],
        q_start=common["q_start"],
        direction=common["direction"],
        phase_lower=0.13,
        phase_upper=float(np.nextafter(0.13, 1.0)),
        base_transform=common["base_transform"],
        point_local_m=common["witness_point_local_m"],
    )
    assert max(reference_plane.lower, compiled_plane[0]) <= min(
        reference_plane.upper, compiled_plane[1]
    )
    for compiled_position, reference_position in zip(
        compiled_positions,
        reference_motion.position_object_m,
    ):
        assert max(reference_position.lower, compiled_position[0]) <= min(
            reference_position.upper, compiled_position[1]
        )


def test_compiled_exact_point_plane_matches_mpmath_mimic_chain() -> None:
    backend = DirectedIntervalKinematics(
        _mimic_three_finger_hand(),
        IntervalArithmeticOptions(
            decimal_precision=80,
            maximum_root_bisection_iterations=256,
        ),
    )
    common = {
        "link_name": "link_a",
        "q_start": np.asarray((-0.25, 0.0, 0.0)),
        "direction": np.asarray((0.4, 0.0, 0.0)),
        "base_transform": np.asarray(
            (
                (0.0, -1.0, 0.0, 0.2),
                (1.0, 0.0, 0.0, -0.1),
                (0.0, 0.0, 1.0, 0.3),
                (0.0, 0.0, 0.0, 1.0),
            ),
            dtype=np.float64,
        ),
        "witness_point_local_m": np.asarray((0.6, 0.1, -0.2)),
        "object_triangle_m": np.asarray(
            ((0.1, -1.0, -1.0), (0.1, 1.0, -1.0), (0.1, 0.0, 1.0)),
            dtype=np.float64,
        ),
    }
    binding = backend._new_compiled_point_plane_binding(**common)
    if binding is None:
        pytest.skip(backend.compiled_point_backend_failure_reason)

    for phase in (0.0, 0.2, 0.6, 1.0):
        reference = backend.contact_plane_value(
            **common,
            phase_lower=phase,
            phase_upper=phase,
        )
        compiled = backend.contact_plane_value(
            **common,
            phase_lower=phase,
            phase_upper=phase,
            _compiled_point_evaluator=binding,
        )
        assert max(reference.lower, compiled.lower) <= min(
            reference.upper, compiled.upper
        )


def test_compiled_point_plane_rebind_reuses_exact_path_and_link_plan() -> None:
    backend = _backend()
    common = {
        "link_name": "link_a",
        "q_start": np.zeros(3),
        "direction": np.asarray((1.0, 0.0, 0.0)),
        "base_transform": np.eye(4),
        "witness_point_local_m": np.asarray((0.0, 0.25, 0.25)),
    }
    first_triangle = _positive_x_object_triangle(0.25)
    second_triangle = _positive_x_object_triangle(0.75)
    first = backend._new_compiled_point_plane_binding(
        **common,
        object_triangle_m=first_triangle,
    )
    if first is None:
        pytest.skip(backend.compiled_point_backend_failure_reason)
    rebound = backend._new_compiled_point_plane_binding(
        **common,
        object_triangle_m=second_triangle,
    )
    assert rebound is first

    reference = backend.contact_plane_value(
        **common,
        object_triangle_m=second_triangle,
        phase_lower=0.5,
        phase_upper=0.5,
    )
    compiled = backend.contact_plane_value(
        **common,
        object_triangle_m=second_triangle,
        phase_lower=0.5,
        phase_upper=0.5,
        _compiled_point_evaluator=rebound,
    )
    assert max(reference.lower, compiled.lower) <= min(
        reference.upper, compiled.upper
    )
    assert compiled.strictly_negative
    assert backend.compiled_point_backend_status == "ACTIVE_REBOUND"
    assert backend.compiled_point_binding_cache_miss_count == 1
    assert backend.compiled_point_binding_cache_hit_count == 1
    assert backend.compiled_point_binding_triangle_rebind_count == 1
    assert rebound.evaluator.triangle_rebind_count == 1
    assert len(backend._compiled_point_plane_binding_cache) == 1
    assert len(backend._compiled_point_plane_link_plan_cache) == 1
    link_plan = backend._compiled_point_plane_link_plan_cache["link_a"]
    assert all(
        not value.flags.writeable
        for value in (
            link_plan.joint_types,
            link_plan.source_indices,
            link_plan.origins_xyz_m,
            link_plan.origins_rpy_rad,
            link_plan.axes,
            link_plan.multipliers,
            link_plan.offsets,
        )
    )


def test_compiled_point_plane_cache_is_bounded_and_evicts_oldest() -> None:
    backend = _backend()
    bindings = []
    for index in range(COMPILED_POINT_PLANE_BINDING_CACHE_CAPACITY + 1):
        binding = backend._new_compiled_point_plane_binding(
            link_name="link_a",
            q_start=np.zeros(3),
            direction=np.asarray((1.0, 0.0, 0.0)),
            base_transform=np.eye(4),
            witness_point_local_m=np.asarray(
                (0.0, 0.25 + index * 1.0e-4, 0.25)
            ),
            object_triangle_m=_positive_x_object_triangle(0.5),
        )
        if binding is None:
            pytest.skip(backend.compiled_point_backend_failure_reason)
        bindings.append(binding)

    assert (
        len(backend._compiled_point_plane_binding_cache)
        == COMPILED_POINT_PLANE_BINDING_CACHE_CAPACITY
    )
    assert backend.compiled_point_binding_cache_miss_count == len(bindings)
    assert backend.compiled_point_binding_cache_eviction_count == 1
    assert bindings[0].evaluator.closed
    assert not bindings[-1].evaluator.closed


def test_plane_root_uses_compiled_exact_point_evaluations() -> None:
    backend = _backend()
    result = backend.certify_transverse_plane_root(
        link_name="link_a",
        q_start=np.zeros(3),
        direction=np.asarray((3.0, 0.0, 0.0)),
        phase_lower=0.0,
        phase_upper=0.5,
        base_transform=np.eye(4),
        witness_point_local_m=np.asarray((0.0, 0.25, 0.25)),
        pad_triangle_local_m=_positive_x_pad_triangle(),
        object_triangle_m=_positive_x_object_triangle(1.0),
    )

    if backend.compiled_point_backend_status == "FALLBACK_MPMATH":
        pytest.skip(backend.compiled_point_backend_failure_reason)
    assert result.root is not None
    assert backend.compiled_point_backend_status == "ACTIVE"
    assert backend.compiled_point_evaluation_count > 0


def test_plane_only_motion_matches_full_contact_plane_quantities() -> None:
    backend = _backend(first_joint_type="revolute")
    common = {
        "link_name": "link_a",
        "q_start": np.asarray((-0.5, 0.0, 0.0)),
        "direction": np.asarray((1.0, 0.0, 0.0)),
        "phase_lower": 0.1,
        "phase_upper": 0.4,
        "base_transform": np.eye(4),
        "witness_point_local_m": np.asarray((1.0, 0.0, 0.0)),
        "object_triangle_m": np.asarray(
            ((0.8, -2.0, -2.0), (0.8, 2.0, -2.0), (0.8, 0.0, 2.0)),
            dtype=np.float64,
        ),
    }
    cache = backend.new_link_transform_cache()
    plane = backend.contact_plane_motion(
        **common,
        transform_cache=cache,
    )
    full = backend.contact_predicates(
        **common,
        pad_triangle_local_m=_positive_x_pad_triangle(),
        transform_cache=cache,
    )

    assert plane.phase == full.phase
    assert plane.plane_value == full.plane_value
    assert plane.plane_derivative == full.plane_derivative


def test_compiled_root_transaction_matches_python_root_and_position() -> None:
    backend = _backend()
    arguments = {
        "link_name": "link_a",
        "q_start": np.zeros(3),
        "direction": np.asarray((3.0, 0.0, 0.0)),
        "phase_lower": 0.0,
        "phase_upper": 0.5,
        "base_transform": np.eye(4),
        "witness_point_local_m": np.asarray((0.0, 0.25, 0.25)),
        "pad_triangle_local_m": _positive_x_pad_triangle(),
        "object_triangle_m": _positive_x_object_triangle(1.0),
    }
    compiled = backend.certify_transverse_plane_root(**arguments)
    reference = backend.certify_transverse_plane_root(
        **arguments,
        _use_compiled_root_transaction=False,
    )

    if backend.compiled_point_backend_status == "FALLBACK_MPMATH":
        pytest.skip(backend.compiled_point_backend_failure_reason)
    assert compiled.root is not None
    assert reference.root is not None
    assert compiled.root.isolating_interval == reference.root.isolating_interval
    assert compiled.root.value_at_lower.strictly_negative
    assert compiled.root.value_at_upper.strictly_positive
    for compiled_position, reference_position in zip(
        compiled.root.position_object_m,
        reference.root.position_object_m,
    ):
        assert max(compiled_position.lower, reference_position.lower) <= min(
            compiled_position.upper, reference_position.upper
        )
    assert backend.compiled_root_transaction_count == 1
    assert backend.compiled_interval_position_evaluation_count == 1
    assert (
        compiled.root.interpolation_iterations
        + compiled.root.interval_newton_iterations
        >= 1
    )
    assert compiled.root.bisection_iterations < 10

    pure_bisection = backend.certify_transverse_plane_root(
        **arguments,
        _use_compiled_root_transaction=False,
        _use_interval_newton_contraction=False,
        _use_nominal_root_seed=False,
    )
    assert pure_bisection.root is not None
    assert (
        compiled.root.bisection_iterations
        < pure_bisection.root.bisection_iterations
    )


def test_compiled_plane_root_skips_full_contact_predicates(monkeypatch) -> None:
    backend = _backend()

    def forbidden_full(**_arguments):
        raise AssertionError("plane-only root must not build full contact predicates")

    monkeypatch.setattr(backend, "contact_predicates", forbidden_full)
    result = backend.certify_transverse_plane_root(
        link_name="link_a",
        q_start=np.zeros(3),
        direction=np.asarray((3.0, 0.0, 0.0)),
        phase_lower=0.0,
        phase_upper=0.5,
        base_transform=np.eye(4),
        witness_point_local_m=np.asarray((0.0, 0.25, 0.25)),
        pad_triangle_local_m=_positive_x_pad_triangle(),
        object_triangle_m=_positive_x_object_triangle(1.0),
    )

    if backend.compiled_point_backend_status == "FALLBACK_MPMATH":
        pytest.skip(backend.compiled_point_backend_failure_reason)
    assert result.root is not None
    assert backend.compiled_root_transaction_count == 1


def test_unique_plane_root_outside_triangle_stops_before_exact_isolation() -> None:
    diagonal = math.sqrt(0.5)
    backend = DirectedIntervalKinematics(
        _three_finger_hand(
            first_joint_type="prismatic",
            first_axis=(diagonal, diagonal, 0.0),
        ),
        IntervalArithmeticOptions(
            decimal_precision=80,
            maximum_root_bisection_iterations=256,
        ),
    )
    cache = backend.new_link_transform_cache()
    result = backend.certify_transverse_contact_root(
        link_name="link_a",
        q_start=np.zeros(3),
        direction=np.asarray((1.0, 0.0, 0.0)),
        phase_lower=0.0,
        phase_upper=0.6,
        base_transform=np.eye(4),
        witness_point_local_m=np.asarray((0.0, 0.5, 0.25)),
        pad_triangle_local_m=_positive_x_pad_triangle(),
        object_triangle_m=_positive_x_object_triangle(0.3),
        transform_cache=cache,
    )

    assert result.state is IntervalRootState.CERTIFIED_FREE
    assert result.reason == (
        "UNIQUE_PLANE_ROOT_BRACKET_STRICTLY_OUTSIDE_TRIANGLE"
    )
    assert result.certificate is None
    assert 1 < cache.entry_count <= MULTIPHASE_TRANSFORM_CACHE_CAPACITY
    assert cache.nonprimary_transform_bypass_count == 0


def test_plane_free_root_classification_skips_full_predicates(
    monkeypatch,
) -> None:
    backend = _backend()
    plane_calls: list[tuple[float, float]] = []
    original_plane = backend.contact_plane_value

    def counted_plane(**arguments):
        plane_calls.append(
            (arguments["phase_lower"], arguments["phase_upper"])
        )
        return original_plane(**arguments)

    def forbidden_full(**_arguments):
        raise AssertionError("plane-free pair must skip full predicates")

    monkeypatch.setattr(backend, "contact_plane_value", counted_plane)
    monkeypatch.setattr(backend, "contact_predicates", forbidden_full)
    result = backend.certify_transverse_contact_root(
        link_name="link_a",
        q_start=np.zeros(3),
        direction=np.asarray((1.0, 0.0, 0.0)),
        phase_lower=0.2,
        phase_upper=0.4,
        base_transform=np.eye(4),
        witness_point_local_m=np.asarray((0.0, 0.25, 0.25)),
        pad_triangle_local_m=_positive_x_pad_triangle(),
        object_triangle_m=_positive_x_object_triangle(0.75),
    )

    assert result.state is IntervalRootState.CERTIFIED_FREE
    assert result.reason == "OBJECT_PLANE_VALUE_EXCLUDES_ZERO"
    assert plane_calls == [(0.2, 0.4)]


def test_nonbinary_root_uses_prefilter_then_full_predicates_for_whole_and_root(
    monkeypatch,
) -> None:
    backend = _backend()
    full_calls: list[tuple[float, float]] = []
    plane_calls: list[tuple[float, float]] = []
    original_full = backend.contact_predicates
    original_plane = backend.contact_plane_value

    def counted_full(**arguments):
        full_calls.append(
            (arguments["phase_lower"], arguments["phase_upper"])
        )
        return original_full(**arguments)

    def counted_plane(**arguments):
        plane_calls.append(
            (arguments["phase_lower"], arguments["phase_upper"])
        )
        return original_plane(**arguments)

    monkeypatch.setattr(backend, "contact_predicates", counted_full)
    monkeypatch.setattr(backend, "contact_plane_value", counted_plane)
    result = backend.certify_transverse_contact_root(
        link_name="link_a",
        q_start=np.zeros(3),
        direction=np.asarray((3.0, 0.0, 0.0)),
        phase_lower=0.0,
        phase_upper=0.5,
        base_transform=np.eye(4),
        witness_point_local_m=np.asarray((0.0, 0.25, 0.25)),
        pad_triangle_local_m=_positive_x_pad_triangle(),
        object_triangle_m=_positive_x_object_triangle(1.0),
    )

    assert result.certificate is not None
    assert len(full_calls) == 2
    assert full_calls[0] == (0.0, 0.5)
    assert full_calls[-1] == (
        result.certificate.phase.lower,
        result.certificate.phase.upper,
    )
    assert plane_calls[0] == (0.0, 0.5)
    assert len(plane_calls) == result.certificate.bisection_iterations + 3


def test_nonbinary_one_third_root_is_an_interval_not_its_display_midpoint() -> None:
    backend = _backend()
    arguments = {
        "link_name": "link_a",
        "q_start": np.zeros(3),
        "direction": np.asarray((3.0, 0.0, 0.0)),
        "phase_lower": 0.0,
        "phase_upper": 0.5,
        "base_transform": np.eye(4),
        "witness_point_local_m": np.asarray((0.0, 0.25, 0.25)),
        "pad_triangle_local_m": _positive_x_pad_triangle(),
        "object_triangle_m": _positive_x_object_triangle(1.0),
    }

    first = backend.certify_transverse_contact_root(**arguments)
    second = backend.certify_transverse_contact_root(**arguments)

    assert first.state is (
        IntervalRootState.CERTIFIED_PAD_DIRECTIONAL_TRANSVERSE_ROOT
    )
    certificate = first.certificate
    repeated_certificate = second.certificate
    assert certificate is not None
    assert repeated_certificate is not None
    implicit = certificate.implicit_root
    assert isinstance(implicit, CertifiedImplicitRoot)
    exact_root = Fraction(1, 3)
    assert Fraction.from_float(implicit.isolating_interval.lower) < exact_root
    assert exact_root < Fraction.from_float(implicit.isolating_interval.upper)
    assert implicit.uniqueness_proven is True
    assert implicit.display_approximation_role == DISPLAY_APPROXIMATION_ROLE
    assert implicit.equation_sha256 == (
        repeated_certificate.implicit_root.equation_sha256
    )
    assert implicit.feature_identity_sha256 == (
        repeated_certificate.implicit_root.feature_identity_sha256
    )

    display_row = backend.contact_predicates(
        **{
            **arguments,
            "phase_lower": implicit.display_approximation,
            "phase_upper": implicit.display_approximation,
        }
    )
    assert not display_row.plane_value.contains_zero
    assert implicit.as_dict()["display_approximation_role"] == (
        DISPLAY_APPROXIMATION_ROLE
    )
    with pytest.raises(FrozenInstanceError):
        implicit.uniqueness_proven = False  # type: ignore[misc]

    limited_backend = DirectedIntervalKinematics(
        _three_finger_hand(first_joint_type="prismatic"),
        IntervalArithmeticOptions(
            decimal_precision=80,
            maximum_root_bisection_iterations=1,
        ),
    )
    exhausted = limited_backend.certify_transverse_contact_root(**arguments)
    assert exhausted.state is IntervalRootState.UNRESOLVED
    assert exhausted.reason == "ROOT_BISECTION_COMPUTATION_BUDGET_EXHAUSTED"


def test_triangle_boundary_root_is_unresolved() -> None:
    backend = _backend()
    negative_x_pad = np.asarray(
        ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
        dtype=np.float64,
    )
    result = backend.certify_transverse_contact_root(
        link_name="link_a",
        q_start=np.asarray((0.5, 0.0, 0.0)),
        direction=np.asarray((-1.0, 0.0, 0.0)),
        phase_lower=0.0,
        phase_upper=0.4,
        base_transform=np.eye(4),
        witness_point_local_m=np.asarray((0.0, 0.0, 0.25)),
        pad_triangle_local_m=negative_x_pad,
        object_triangle_m=_positive_x_object_triangle(0.3),
    )

    assert result.state is IntervalRootState.UNRESOLVED
    assert result.reason == "TRIANGLE_BOUNDARY_NOT_STRICTLY_INTERIOR"


def test_revolute_tangent_is_not_a_transverse_unique_root() -> None:
    backend = _backend(first_joint_type="revolute")
    object_triangle = np.asarray(
        ((1.0, -2.0, -2.0), (1.0, 2.0, -2.0), (1.0, 0.0, 2.0)),
        dtype=np.float64,
    )
    predicates = backend.contact_predicates(
        link_name="link_a",
        q_start=np.asarray((-0.5, 0.0, 0.0)),
        direction=np.asarray((1.0, 0.0, 0.0)),
        phase_lower=0.0,
        phase_upper=1.0,
        base_transform=np.eye(4),
        witness_point_local_m=np.asarray((1.0, 0.0, 0.0)),
        pad_triangle_local_m=_positive_x_pad_triangle(),
        object_triangle_m=object_triangle,
    )

    assert predicates.plane_value.contains_zero
    assert predicates.plane_derivative.contains_zero
    assert predicates.plane_second_derivative.strictly_negative

    root = backend.certify_transverse_contact_root(
        link_name="link_a",
        q_start=np.asarray((-0.5, 0.0, 0.0)),
        direction=np.asarray((1.0, 0.0, 0.0)),
        phase_lower=0.0,
        phase_upper=1.0,
        base_transform=np.eye(4),
        witness_point_local_m=np.asarray((1.0, 0.0, 0.0)),
        pad_triangle_local_m=_positive_x_pad_triangle(),
        object_triangle_m=object_triangle,
    )
    assert root.state is IntervalRootState.UNRESOLVED
    assert root.reason == "NONTRANSVERSE_OR_MULTIPLE_PLANE_ROOTS"


def test_interval_inputs_fail_closed_without_hidden_defaults() -> None:
    with pytest.raises(IntervalKinematicsError, match="positive integer"):
        IntervalArithmeticOptions(
            decimal_precision=0,
            maximum_root_bisection_iterations=256,
        )
    with pytest.raises(IntervalKinematicsError, match="positive integer"):
        IntervalArithmeticOptions(
            decimal_precision=True,
            maximum_root_bisection_iterations=256,
        )
    with pytest.raises(IntervalKinematicsError, match="bisection"):
        IntervalArithmeticOptions(
            decimal_precision=80,
            maximum_root_bisection_iterations=0,
        )

    options = IntervalArithmeticOptions(
        decimal_precision=80,
        maximum_root_bisection_iterations=256,
    )
    assert options.as_dict() == {
        "decimal_precision": 80,
        "maximum_root_bisection_iterations": 256,
    }
    with pytest.raises(IntervalKinematicsError, match="root certificate"):
        IntervalRootClassification(
            state=(
                IntervalRootState.CERTIFIED_PAD_DIRECTIONAL_TRANSVERSE_ROOT
            ),
            searched_phase=IntervalBounds(0.0, 1.0),
            certificate=None,
            reason="FORGERY_TEST",
        )

    low_precision = DirectedIntervalKinematics(
        _three_finger_hand(first_joint_type="prismatic"),
        IntervalArithmeticOptions(
            decimal_precision=15,
            maximum_root_bisection_iterations=64,
        ),
    )
    high_precision = _backend()
    assert low_precision.context.prec == 53
    assert high_precision.context.prec == 269
    assert low_precision.context.prec == 53

    backend = _backend()
    with pytest.raises(IntervalKinematicsError, match="finite and ordered"):
        backend.contact_predicates(
            link_name="link_a",
            q_start=np.zeros(3),
            direction=np.asarray((1.0, 0.0, 0.0)),
            phase_lower=math.inf,
            phase_upper=1.0,
            base_transform=np.eye(4),
            witness_point_local_m=np.zeros(3),
            pad_triangle_local_m=_positive_x_pad_triangle(),
            object_triangle_m=_positive_x_object_triangle(0.3),
        )

    reflection = np.eye(4)
    reflection[0, 0] = -1.0
    with pytest.raises(IntervalKinematicsError, match="preserve orientation"):
        backend.point_motion(
            link_name="link_a",
            q_start=np.zeros(3),
            direction=np.asarray((1.0, 0.0, 0.0)),
            phase_lower=0.0,
            phase_upper=0.5,
            base_transform=reflection,
            point_local_m=np.zeros(3),
        )

    for invalid_rotation in (
        np.diag((100.0, 1.0, 1.0, 1.0)),
        np.asarray(
            (
                (1.0, 1.0e-6, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            )
        ),
    ):
        with pytest.raises(IntervalKinematicsError, match="proper rotation"):
            backend.point_motion(
                link_name="link_a",
                q_start=np.zeros(3),
                direction=np.asarray((1.0, 0.0, 0.0)),
                phase_lower=0.0,
                phase_upper=0.5,
                base_transform=invalid_rotation,
                point_local_m=np.zeros(3),
            )

    with pytest.raises(IntervalKinematicsError, match="limit contract"):
        backend.point_motion(
            link_name="link_a",
            q_start=np.asarray((1.9, 0.0, 0.0)),
            direction=np.asarray((1.0, 0.0, 0.0)),
            phase_lower=0.0,
            phase_upper=0.5,
            base_transform=np.eye(4),
            point_local_m=np.zeros(3),
        )

    with pytest.raises(IntervalKinematicsError, match="limit contract"):
        backend.point_motion(
            link_name="link_a",
            q_start=np.asarray((2.0, 0.0, 0.0)),
            direction=np.asarray((2.0 ** -54, 0.0, 0.0)),
            phase_lower=0.0,
            phase_upper=1.0,
            base_transform=np.eye(4),
            point_local_m=np.zeros(3),
        )

    invalid_axis_hand = _three_finger_hand(
        first_joint_type="prismatic",
        first_axis=(2.0, 0.0, 0.0),
    )
    with pytest.raises(IntervalKinematicsError, match="unit vector"):
        DirectedIntervalKinematics(
            invalid_axis_hand,
            IntervalArithmeticOptions(
                decimal_precision=80,
                maximum_root_bisection_iterations=256,
            ),
        )
