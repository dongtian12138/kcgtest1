import math
from dataclasses import FrozenInstanceError
from fractions import Fraction

import numpy as np
import pytest

from kcg_connector.grasp.robust.hand_model import (
    GeometrySpec,
    JointLimit,
    JointSpec,
    PadGeometry,
    ThreeFingerHandModel,
)
from kcg_connector.grasp.robust.interval_kinematics import (
    DISPLAY_APPROXIMATION_ROLE,
    CertifiedImplicitRoot,
    DirectedIntervalKinematics,
    IntervalArithmeticOptions,
    IntervalBounds,
    IntervalKinematicsError,
    IntervalRootClassification,
    IntervalRootState,
    METHOD_ID,
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
