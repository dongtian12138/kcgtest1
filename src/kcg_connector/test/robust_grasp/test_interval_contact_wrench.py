from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
from itertools import product
import hashlib
import math

import numpy as np
import pytest

from kcg_connector.grasp.robust.interval_contact_wrench import (
    FORMAL_DOMAIN_RULE,
    METHOD_ID,
    IntervalContactWrenchError,
    build_interval_contact_wrench_matrices,
)
from kcg_connector.grasp.robust.interval_kinematics import (
    INTERVAL_GEOMETRIC_JACOBIAN_METHOD_ID,
    IntervalBounds,
    IntervalGeometricJacobian,
)
from kcg_connector.grasp.robust.task_wrench_evaluator import (
    ContactRangeRootWrenchDomain,
)


def _root_domains():
    radius = 0.05
    root_three_over_two = math.sqrt(3.0) / 2.0
    points = (
        (radius, 0.0, 0.0),
        (-0.5 * radius, root_three_over_two * radius, 0.0),
        (-0.5 * radius, -root_three_over_two * radius, 0.0),
    )
    free_side = tuple(
        tuple(float(value / np.linalg.norm(point)) for value in point)
        for point in points
    )
    joint_names = ("joint_a", "joint_b", "joint_c")
    joint_box = tuple(IntervalBounds(0.0, 0.0) for _ in joint_names)
    roots = []
    for contact_index, (point, normal) in enumerate(zip(points, free_side)):
        position = tuple(
            IntervalBounds(value - 1.0e-5, value + 1.0e-5)
            for value in point
        )
        jacobian_rows = []
        for axis in range(6):
            row = []
            for joint_index in range(3):
                center = (
                    0.2 * (axis + 1) * (joint_index + 1)
                    if axis < 3
                    else 0.0
                )
                half_width = 1.0e-6 if axis < 3 else 0.0
                row.append(
                    IntervalBounds(center - half_width, center + half_width)
                )
            jacobian_rows.append(tuple(row))
        jacobian = IntervalGeometricJacobian(
            link_name=f"link_{contact_index}",
            independent_joint_names=joint_names,
            joint_position_intervals=joint_box,
            point_object_m=position,  # type: ignore[arg-type]
            elements=tuple(jacobian_rows),
            method_id=INTERVAL_GEOMETRIC_JACOBIAN_METHOD_ID,
            decimal_precision=80,
        )
        roots.append(
            ContactRangeRootWrenchDomain(
                pad_name=f"pad_{contact_index}",
                formal_root_sha256=hashlib.sha256(
                    f"root:{contact_index}".encode("utf-8")
                ).hexdigest(),
                witness_flat_index=contact_index,
                object_face_index=contact_index,
                semantic_classification=(
                    "ALLOWED_PATH_LOCAL_FREE_SIDE_TRANSVERSE_CONTACT"
                ),
                phase=IntervalBounds(0.1, 0.2),
                position_object_m=position,  # type: ignore[arg-type]
                object_source_winding_free_side_sign=1,
                path_local_free_side_normal_object=normal,  # type: ignore[arg-type]
                pad_link_name=f"link_{contact_index}",
                interval_geometric_jacobian=jacobian,
            )
        )
    return tuple(roots)


def _build(roots=None):
    if roots is None:
        roots = _root_domains()
    return build_interval_contact_wrench_matrices(
        root_domains=roots,
        pad_order=("pad_0", "pad_1", "pad_2"),
        normal_force_caps_n=(20.0, 20.0, 20.0),
        wrench_origin_object_m=(0.0, 0.0, 0.0),
        task_frame_rotation_object=np.eye(3),
        hard_bound_friction_coefficient=0.8,
        maximum_inner_approximation_relative_error=0.1,
        cone_edge_multiplier=1,
    )


def test_physical_root_domains_enclose_position_and_jacobian_corners() -> None:
    roots = _root_domains()
    first = _build(roots)
    second = _build(roots)

    assert first.as_dict() == second.as_dict()
    assert first.method_id == METHOD_ID
    assert first.formal_domain_rule == FORMAL_DOMAIN_RULE
    assert len(first.matrix_sha256) == 64
    assert first.pad_order == ("pad_0", "pad_1", "pad_2")
    assert first.independent_joint_names == (
        "joint_a",
        "joint_b",
        "joint_c",
    )
    assert first.ray_count == 3 * first.edges_per_contact
    assert not first.display_approximation_used_as_formal_evidence
    assert not first.finite_contact_geometry_sampling_used_as_formal_evidence

    for ray_index, owner in enumerate(first.ray_owner):
        force = np.asarray(first.force_ray_vectors_object[ray_index])
        for corner in product(
            *(
                (bounds.lower, bounds.upper)
                for bounds in roots[owner].position_object_m
            )
        ):
            exact_wrench = np.concatenate(
                (force, np.cross(np.asarray(corner), force))
            )
            assert all(
                first.grasp_matrix_intervals[row][ray_index].lower
                <= exact_wrench[row]
                <= first.grasp_matrix_intervals[row][ray_index].upper
                for row in range(6)
            )
        jacobian = roots[owner].interval_geometric_jacobian
        for joint_index in range(3):
            for endpoint_choice in product((0, 1), repeat=3):
                linear_column = np.asarray(
                    [
                        (
                            jacobian.elements[axis][joint_index].lower,
                            jacobian.elements[axis][joint_index].upper,
                        )[endpoint_choice[axis]]
                        for axis in range(3)
                    ]
                )
                exact_torque = sum(
                    Fraction.from_float(float(jacobian_value))
                    * Fraction.from_float(float(force_value))
                    for jacobian_value, force_value in zip(
                        linear_column,
                        force,
                    )
                )
                bounds = first.joint_torque_from_ray_intervals[
                    joint_index
                ][ray_index]
                assert (
                    Fraction.from_float(bounds.lower)
                    <= exact_torque
                    <= Fraction.from_float(bounds.upper)
                )

    with pytest.raises(FrozenInstanceError):
        first.decimal_precision = 20  # type: ignore[misc]


def test_pad_order_mismatch_is_rejected() -> None:
    roots = _root_domains()

    with pytest.raises(IntervalContactWrenchError, match="ordered roots"):
        _build((roots[1], roots[0], roots[2]))


def test_joint_order_and_precision_mismatch_are_rejected() -> None:
    roots = list(_root_domains())
    second_jacobian = roots[1].interval_geometric_jacobian
    roots[1] = replace(
        roots[1],
        interval_geometric_jacobian=replace(
            second_jacobian,
            independent_joint_names=("joint_b", "joint_a", "joint_c"),
        ),
    )
    with pytest.raises(IntervalContactWrenchError, match="ordered interval"):
        _build(tuple(roots))

    roots = list(_root_domains())
    third_jacobian = roots[2].interval_geometric_jacobian
    roots[2] = replace(
        roots[2],
        interval_geometric_jacobian=replace(
            third_jacobian,
            decimal_precision=60,
        ),
    )
    with pytest.raises(IntervalContactWrenchError, match="precision"):
        _build(tuple(roots))
