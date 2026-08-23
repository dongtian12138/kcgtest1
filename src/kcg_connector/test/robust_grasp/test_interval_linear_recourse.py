from dataclasses import FrozenInstanceError
from itertools import product

import numpy as np
import pytest

from kcg_connector.grasp.robust.interval_kinematics import IntervalBounds
from kcg_connector.grasp.robust.interval_linear_recourse import (
    METHOD_ID,
    PRECONDITIONER_ROLE,
    IntervalLinearRecourseError,
    IntervalLinearRecourseState,
    certify_interval_linear_recourse,
)


def _assert_solution_is_enclosed(
    intervals: tuple[IntervalBounds, ...], solution: np.ndarray
) -> None:
    assert all(
        bounds.lower <= value <= bounds.upper
        for bounds, value in zip(intervals, solution)
    )


def test_scalar_interval_family_encloses_every_exact_solution() -> None:
    certificate = certify_interval_linear_recourse(
        coefficient_intervals=((IntervalBounds(2.0, 3.0),),),
        right_hand_side_intervals=(IntervalBounds(4.0, 5.0),),
        decimal_precision=80,
    )

    assert certificate.state is (
        IntervalLinearRecourseState.CERTIFIED_UNIQUE_SOLUTION_ENCLOSURE
    )
    assert certificate.solution_intervals is not None
    assert certificate.contraction_norm_upper is not None
    assert certificate.contraction_norm_upper < 1.0
    for coefficient, right_hand_side in product(
        (2.0, 2.25, 2.5, 2.75, 3.0),
        (4.0, 4.25, 4.5, 4.75, 5.0),
    ):
        _assert_solution_is_enclosed(
            certificate.solution_intervals,
            np.asarray((right_hand_side / coefficient,)),
        )
    assert certificate.method_id == METHOD_ID
    assert certificate.preconditioner_role == PRECONDITIONER_ROLE
    with pytest.raises(FrozenInstanceError):
        certificate.reason = "forged"  # type: ignore[misc]


def test_two_dimensional_interval_family_encloses_exact_grid() -> None:
    coefficient_intervals = (
        (IntervalBounds(1.9, 2.1), IntervalBounds(0.05, 0.15)),
        (IntervalBounds(0.15, 0.25), IntervalBounds(1.4, 1.6)),
    )
    right_hand_side_intervals = (
        IntervalBounds(0.9, 1.1),
        IntervalBounds(1.8, 2.2),
    )
    first = certify_interval_linear_recourse(
        coefficient_intervals=coefficient_intervals,
        right_hand_side_intervals=right_hand_side_intervals,
        decimal_precision=80,
    )
    second = certify_interval_linear_recourse(
        coefficient_intervals=coefficient_intervals,
        right_hand_side_intervals=right_hand_side_intervals,
        decimal_precision=80,
    )

    assert first.state is (
        IntervalLinearRecourseState.CERTIFIED_UNIQUE_SOLUTION_ENCLOSURE
    )
    assert first.solution_intervals is not None
    assert first.as_dict() == second.as_dict()
    value_sets = (
        (1.9, 2.0, 2.1),
        (0.05, 0.1, 0.15),
        (0.15, 0.2, 0.25),
        (1.4, 1.5, 1.6),
        (0.9, 1.0, 1.1),
        (1.8, 2.0, 2.2),
    )
    for values in product(*value_sets):
        matrix = np.asarray(
            ((values[0], values[1]), (values[2], values[3])),
            dtype=np.float64,
        )
        right_hand_side = np.asarray((values[4], values[5]))
        exact_solution = np.linalg.solve(matrix, right_hand_side)
        _assert_solution_is_enclosed(
            first.solution_intervals, exact_solution
        )


def test_family_containing_singular_matrix_fails_closed() -> None:
    certificate = certify_interval_linear_recourse(
        coefficient_intervals=((IntervalBounds(-0.5, 1.5),),),
        right_hand_side_intervals=(IntervalBounds(1.0, 1.0),),
        decimal_precision=80,
    )

    assert certificate.state is IntervalLinearRecourseState.NOT_CERTIFIABLE
    assert certificate.solution_intervals is None
    assert certificate.solution_error_radius_upper is None
    assert certificate.contraction_norm_upper is not None
    assert certificate.contraction_norm_upper >= 1.0
    assert certificate.reason == (
        "BANACH_CONTRACTION_NORM_DOES_NOT_EXCLUDE_ONE"
    )


def test_singular_midpoint_and_malformed_inputs_fail_closed() -> None:
    singular_midpoint = certify_interval_linear_recourse(
        coefficient_intervals=((IntervalBounds(-1.0, 1.0),),),
        right_hand_side_intervals=(IntervalBounds(1.0, 1.0),),
        decimal_precision=80,
    )
    assert singular_midpoint.state is IntervalLinearRecourseState.NOT_CERTIFIABLE
    assert singular_midpoint.reason == "MIDPOINT_MATRIX_NOT_INVERTIBLE"
    assert singular_midpoint.preconditioner is None

    with pytest.raises(IntervalLinearRecourseError, match=r"A:\(N,N\)"):
        certify_interval_linear_recourse(
            coefficient_intervals=(
                (IntervalBounds(1.0, 1.0), IntervalBounds(0.0, 0.0)),
            ),
            right_hand_side_intervals=(IntervalBounds(1.0, 1.0),),
            decimal_precision=80,
        )
    with pytest.raises(IntervalLinearRecourseError, match="positive integer"):
        certify_interval_linear_recourse(
            coefficient_intervals=((IntervalBounds(1.0, 1.0),),),
            right_hand_side_intervals=(IntervalBounds(1.0, 1.0),),
            decimal_precision=True,
        )
