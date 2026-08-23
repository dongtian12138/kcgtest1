from dataclasses import FrozenInstanceError
from itertools import product
import math

import numpy as np
import pytest

from kcg_connector.grasp.robust.interval_contact_balance import (
    FORMAL_EVIDENCE_ROLE,
    METHOD_ID,
    MIDPOINT_PROPOSAL_ROLE,
    IntervalContactBalanceError,
    IntervalContactBalanceState,
    certify_interval_contact_balance,
)
from kcg_connector.grasp.robust.interval_kinematics import IntervalBounds
from kcg_connector.grasp.robust.robust_wrench import (
    build_polyhedral_contact_wrench_model,
)


def _outward_bounds(values: np.ndarray) -> IntervalBounds:
    return IntervalBounds(
        float(np.nextafter(np.min(values), -math.inf)),
        float(np.nextafter(np.max(values), math.inf)),
    )


def _three_sided_fixture(position_half_width: float = 1.0e-6):
    radius = 0.05
    root_three_over_two = math.sqrt(3.0) / 2.0
    points = np.asarray(
        (
            (radius, 0.0, 0.0),
            (-0.5 * radius, root_three_over_two * radius, 0.0),
            (-0.5 * radius, -root_three_over_two * radius, 0.0),
        ),
        dtype=np.float64,
    )
    inward = -points / np.linalg.norm(points, axis=1)[:, None]
    tangents = np.tile(np.asarray((0.0, 0.0, 1.0)), (3, 1))
    model = build_polyhedral_contact_wrench_model(
        contact_points_m=points,
        inward_normals=inward,
        tangent_directions=tangents,
        friction_coefficients=0.8,
        normal_force_caps_n=(20.0, 20.0, 20.0),
        wrench_origin_m=(0.0, 0.0, 0.0),
        maximum_inner_approximation_relative_error=0.1,
    )
    position_bounds = tuple(
        tuple(
            IntervalBounds(
                float(value - position_half_width),
                float(value + position_half_width),
            )
            for value in point
        )
        for point in points
    )
    force_by_ray = tuple(
        np.asarray(
            model.contact_force_matrix[
                3 * owner : 3 * owner + 3,
                ray_index,
            ],
            dtype=np.float64,
        )
        for ray_index, owner in enumerate(model.ray_owner)
    )
    grasp_columns: list[tuple[IntervalBounds, ...]] = []
    for ray_index, owner in enumerate(model.ray_owner):
        force = force_by_ray[ray_index]
        torque_corners = np.asarray(
            [
                np.cross(np.asarray(corner), force)
                for corner in product(
                    *(
                        (bounds.lower, bounds.upper)
                        for bounds in position_bounds[owner]
                    )
                )
            ]
        )
        grasp_columns.append(
            tuple(IntervalBounds(float(value), float(value)) for value in force)
            + tuple(
                _outward_bounds(torque_corners[:, axis]) for axis in range(3)
            )
        )
    grasp = tuple(
        tuple(column[row] for column in grasp_columns) for row in range(6)
    )
    joint_torque = tuple(
        tuple(
            IntervalBounds(1.0, 1.0)
            if owner == joint_index
            else IntervalBounds(0.0, 0.0)
            for owner in model.ray_owner
        )
        for joint_index in range(3)
    )
    return {
        "grasp_matrix_intervals": grasp,
        "joint_torque_from_ray_intervals": joint_torque,
        "ray_owner": model.ray_owner,
        "normal_force_caps_n": (20.0, 20.0, 20.0),
        "joint_effort_limits": (100.0, 100.0, 100.0),
        "external_wrench": (0.0, 0.0, -1.0, 0.0, 0.0, 0.0),
        "decimal_precision": 80,
        "points": points,
        "position_bounds": position_bounds,
        "force_by_ray": force_by_ray,
    }


def _certificate_inputs(fixture: dict[str, object]) -> dict[str, object]:
    return {
        key: fixture[key]
        for key in (
            "grasp_matrix_intervals",
            "joint_torque_from_ray_intervals",
            "ray_owner",
            "normal_force_caps_n",
            "joint_effort_limits",
            "external_wrench",
            "decimal_precision",
        )
    }


def test_three_sided_contact_box_is_certified_and_encloses_corners() -> None:
    fixture = _three_sided_fixture()
    inputs = _certificate_inputs(fixture)
    first = certify_interval_contact_balance(**inputs)
    second = certify_interval_contact_balance(**inputs)

    assert first.state is (
        IntervalContactBalanceState.CERTIFIED_ALL_INTERVAL_STATES_BALANCED
    )
    assert first.as_dict() == second.as_dict()
    assert first.method_id == METHOD_ID
    assert first.midpoint_proposal_role == MIDPOINT_PROPOSAL_ROLE
    assert first.formal_evidence_role == FORMAL_EVIDENCE_ROLE
    assert first.basis_ray_indices is not None
    assert first.ray_coefficient_intervals is not None
    assert first.midpoint_proposal is not None
    assert first.pad_normal_force_intervals is not None
    assert first.maximum_pad_utilization_upper is not None
    assert first.maximum_pad_utilization_upper < 1.0
    assert first.maximum_joint_torque_utilization_upper is not None
    assert first.maximum_joint_torque_utilization_upper < 1.0

    basis = first.basis_ray_indices
    nonbasis = tuple(
        index for index in range(first.ray_count) if index not in set(basis)
    )
    owners = fixture["ray_owner"]
    forces = fixture["force_by_ray"]
    position_bounds = fixture["position_bounds"]
    external = np.asarray(fixture["external_wrench"], dtype=np.float64)
    proposal = np.asarray(first.midpoint_proposal)
    for endpoint_choice in product((0, 1), repeat=9):
        positions = np.asarray(
            [
                [
                    (bounds.lower, bounds.upper)[
                        endpoint_choice[3 * contact_index + axis]
                    ]
                    for axis, bounds in enumerate(contact_bounds)
                ]
                for contact_index, contact_bounds in enumerate(position_bounds)
            ]
        )
        exact_grasp = np.zeros((6, first.ray_count), dtype=np.float64)
        for ray_index, owner in enumerate(owners):
            force = forces[ray_index]
            exact_grasp[:3, ray_index] = force
            exact_grasp[3:, ray_index] = np.cross(positions[owner], force)
        coefficients = proposal.copy()
        coefficients[list(basis)] = np.linalg.solve(
            exact_grasp[:, basis],
            -external - exact_grasp[:, nonbasis] @ coefficients[list(nonbasis)],
        )
        assert all(
            bounds.lower <= value <= bounds.upper
            for bounds, value in zip(
                first.ray_coefficient_intervals,
                coefficients,
            )
        )
        np.testing.assert_allclose(
            exact_grasp @ coefficients + external,
            np.zeros(6),
            rtol=0.0,
            atol=1.0e-10,
        )
        for contact_index, cap in enumerate(
            fixture["normal_force_caps_n"]
        ):
            assert sum(
                value
                for value, owner in zip(coefficients, owners)
                if owner == contact_index
            ) <= cap

    with pytest.raises(FrozenInstanceError):
        first.reason = "forged"  # type: ignore[misc]


def test_one_sided_contact_rays_are_rejected() -> None:
    fixture = _three_sided_fixture()
    grasp = [list(row) for row in fixture["grasp_matrix_intervals"]]
    for ray_index in range(len(fixture["ray_owner"])):
        grasp[0][ray_index] = IntervalBounds(-1.0, -1.0)
    inputs = _certificate_inputs(fixture)
    inputs["grasp_matrix_intervals"] = tuple(tuple(row) for row in grasp)

    certificate = certify_interval_contact_balance(**inputs)

    assert certificate.state is IntervalContactBalanceState.NOT_CERTIFIABLE
    assert certificate.reason == "MIDPOINT_STRICT_INTERIOR_PROPOSAL_INFEASIBLE"
    assert certificate.ray_coefficient_intervals is None


def test_large_contact_position_box_fails_closed() -> None:
    fixture = _three_sided_fixture(position_half_width=0.2)

    certificate = certify_interval_contact_balance(
        **_certificate_inputs(fixture)
    )

    assert certificate.state is IntervalContactBalanceState.NOT_CERTIFIABLE
    assert certificate.reason.startswith("INTERVAL_RECOURSE_NOT_CERTIFIED:")
    assert certificate.ray_coefficient_intervals is None


def test_malformed_interval_contact_balance_input_is_rejected() -> None:
    fixture = _three_sided_fixture()
    inputs = _certificate_inputs(fixture)
    inputs["ray_owner"] = tuple(fixture["ray_owner"][:-1])

    with pytest.raises(IntervalContactBalanceError, match=r"G:\(6,M\)"):
        certify_interval_contact_balance(**inputs)
