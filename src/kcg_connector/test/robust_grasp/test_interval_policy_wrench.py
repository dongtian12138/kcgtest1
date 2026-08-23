from dataclasses import FrozenInstanceError, replace
import hashlib
import math

import numpy as np
import pytest

from kcg_connector.grasp.robust.interval_kinematics import (
    INTERVAL_GEOMETRIC_JACOBIAN_METHOD_ID,
    IntervalBounds,
    IntervalGeometricJacobian,
)
from kcg_connector.grasp.robust.interval_policy_wrench import (
    LOAD_RULE,
    METHOD_ID,
    ROOT_PRODUCT_RULE,
    IntervalPolicyWrenchError,
    IntervalPolicyWrenchState,
    certify_declared_interval_policy_wrench_margin,
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
    joint_names = ("joint_a", "joint_b", "joint_c")
    joint_box = tuple(IntervalBounds(0.0, 0.0) for _ in joint_names)
    roots = []
    for contact_index, point in enumerate(points):
        normal = tuple(
            float(value / np.linalg.norm(point)) for value in point
        )
        position = tuple(
            IntervalBounds(value - 1.0e-6, value + 1.0e-6)
            for value in point
        )
        elements = tuple(
            tuple(
                IntervalBounds(
                    (0.02 * (axis + 1) * (joint_index + 1))
                    - (1.0e-7 if axis < 3 else 0.0),
                    (0.02 * (axis + 1) * (joint_index + 1))
                    + (1.0e-7 if axis < 3 else 0.0),
                )
                for joint_index in range(3)
            )
            for axis in range(6)
        )
        jacobian = IntervalGeometricJacobian(
            link_name=f"link_{contact_index}",
            independent_joint_names=joint_names,
            joint_position_intervals=joint_box,
            point_object_m=position,  # type: ignore[arg-type]
            elements=elements,
            method_id=INTERVAL_GEOMETRIC_JACOBIAN_METHOD_ID,
            decimal_precision=80,
        )
        roots.append(
            ContactRangeRootWrenchDomain(
                pad_name=f"pad_{contact_index}",
                formal_root_sha256=hashlib.sha256(
                    f"policy-root:{contact_index}".encode("utf-8")
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


def _vertices() -> np.ndarray:
    vertices = np.zeros((12, 6), dtype=np.float64)
    scales = (0.2, 0.2, 0.2, 0.01, 0.01, 0.01)
    for axis, scale in enumerate(scales):
        vertices[2 * axis, axis] = scale
        vertices[2 * axis + 1, axis] = -scale
    return vertices


def _variant(root: ContactRangeRootWrenchDomain, label: str):
    shifted_position = tuple(
        IntervalBounds(bounds.lower + 2.0e-7, bounds.upper + 2.0e-7)
        for bounds in root.position_object_m
    )
    return replace(
        root,
        formal_root_sha256=hashlib.sha256(label.encode("utf-8")).hexdigest(),
        position_object_m=shifted_position,  # type: ignore[arg-type]
        interval_geometric_jacobian=replace(
            root.interval_geometric_jacobian,
            point_object_m=shifted_position,  # type: ignore[arg-type]
        ),
    )


def _inputs(domains=None):
    roots = _root_domains()
    if domains is None:
        domains = (
            tuple(sorted((roots[0], _variant(roots[0], "variant-a")), key=lambda row: row.formal_root_sha256)),
            (roots[1],),
            tuple(sorted((roots[2], _variant(roots[2], "variant-c")), key=lambda row: row.formal_root_sha256)),
        )
    return {
        "pad_root_domains": domains,
        "pad_order": ("pad_0", "pad_1", "pad_2"),
        "normal_force_caps_n": (20.0, 20.0, 20.0),
        "joint_effort_limits": (100.0, 100.0, 100.0),
        "wrench_origin_object_m": (0.0, 0.0, 0.0),
        "task_frame_rotation_object": np.eye(3),
        "hard_bound_friction_coefficient": 0.8,
        "maximum_inner_approximation_relative_error": 0.1,
        "cone_edge_multiplier": 1,
        "nominal_external_wrench": (0.0, 0.0, -1.0, 0.0, 0.0, 0.0),
        "disturbance_vertices": _vertices(),
        "declared_margin": 0.25,
    }


def test_complete_root_product_and_twelve_loads_are_certified() -> None:
    inputs = _inputs()
    first = certify_declared_interval_policy_wrench_margin(**inputs)
    second = certify_declared_interval_policy_wrench_margin(**inputs)

    assert first.state is (
        IntervalPolicyWrenchState.CERTIFIED_DECLARED_MARGIN_FOR_COMPLETE_DOMAIN
    )
    assert first.as_dict() == second.as_dict()
    assert first.method_id == METHOD_ID
    assert first.root_product_rule == ROOT_PRODUCT_RULE
    assert first.load_rule == LOAD_RULE
    assert first.possible_root_counts == (2, 1, 2)
    assert first.cartesian_product_count == 4
    assert first.evaluated_load_count == 48
    assert first.expected_load_count == 48
    assert first.complete_cartesian_product_and_loads_evaluated
    assert first.certified_margin_lower_bound == 0.25
    assert first.maximum_pad_utilization_upper is not None
    assert first.maximum_pad_utilization_upper < 1.0
    assert first.maximum_joint_torque_utilization_upper is not None
    assert first.maximum_joint_torque_utilization_upper < 1.0
    assert len(first.certificate_sha256) == 64
    with pytest.raises(FrozenInstanceError):
        first.reason = "forged"  # type: ignore[misc]


def test_any_one_sided_root_combination_rejects_whole_margin() -> None:
    roots = tuple(
        replace(
            root,
            path_local_free_side_normal_object=(1.0, 0.0, 0.0),
        )
        for root in _root_domains()
    )
    domains = tuple((root,) for root in roots)

    certificate = certify_declared_interval_policy_wrench_margin(
        **_inputs(domains)
    )

    assert certificate.state is IntervalPolicyWrenchState.NOT_CERTIFIABLE
    assert certificate.certified_margin_lower_bound is None
    assert not certificate.complete_cartesian_product_and_loads_evaluated
    assert certificate.failed_root_combination_index == 0
    assert certificate.failed_disturbance_vertex_index == 0
    assert certificate.reason.startswith("LOAD_NOT_CERTIFIED:")


def test_incomplete_or_duplicate_domain_and_nonpositive_margin_are_rejected() -> None:
    inputs = _inputs()
    inputs["disturbance_vertices"] = _vertices()[:10]
    with pytest.raises(IntervalPolicyWrenchError, match="six nonzero"):
        certify_declared_interval_policy_wrench_margin(**inputs)

    roots = _root_domains()
    inputs = _inputs(((roots[0], roots[0]), (roots[1],), (roots[2],)))
    with pytest.raises(IntervalPolicyWrenchError, match="canonical root"):
        certify_declared_interval_policy_wrench_margin(**inputs)

    inputs = _inputs()
    inputs["declared_margin"] = 0.0
    with pytest.raises(IntervalPolicyWrenchError, match="positive declared"):
        certify_declared_interval_policy_wrench_margin(**inputs)
