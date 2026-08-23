"""Directed interval proof for adjustable square linear systems.

For every real ``A`` and ``b`` in the supplied interval family, this module
certifies existence and uniqueness of ``x`` in a reported enclosure.  A
binary64 midpoint inverse is only a fixed preconditioner.  The proof uses the
directed interval upper bound

    q = ||I - R[A]||_inf < 1

and the Banach perturbation identity.  No sampled matrix, midpoint solution,
or empirical tolerance is accepted as the certificate.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Sequence

from mpmath.ctx_iv import MPIntervalContext
import numpy as np

from kcg_connector.grasp.robust.interval_kinematics import IntervalBounds


METHOD_ID = "MPMATH_DIRECTED_INTERVAL_BANACH_LINEAR_RECOURSE_V1"
PRECONDITIONER_ROLE = "BINARY64_MIDPOINT_INVERSE_PROPOSAL_ONLY"


class IntervalLinearRecourseError(ValueError):
    """Raised when an interval linear-system input is malformed."""


class IntervalLinearRecourseState(str, Enum):
    CERTIFIED_UNIQUE_SOLUTION_ENCLOSURE = (
        "CERTIFIED_UNIQUE_SOLUTION_ENCLOSURE"
    )
    NOT_CERTIFIABLE = "NOT_CERTIFIABLE"


def _optional_float_hex(value: float | None) -> str | None:
    return None if value is None else float(value).hex()


@dataclass(frozen=True)
class IntervalLinearRecourseCertificate:
    """Immutable Banach certificate for one square interval system family."""

    state: IntervalLinearRecourseState
    dimension: int
    solution_intervals: tuple[IntervalBounds, ...] | None
    contraction_norm_upper: float | None
    preconditioned_residual_norm_upper: float | None
    solution_error_radius_upper: float | None
    midpoint_solution: tuple[float, ...] | None
    preconditioner: tuple[tuple[float, ...], ...] | None
    method_id: str
    preconditioner_role: str
    decimal_precision: int
    reason: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.state, IntervalLinearRecourseState)
            or not isinstance(self.dimension, int)
            or isinstance(self.dimension, bool)
            or self.dimension <= 0
            or self.method_id != METHOD_ID
            or self.preconditioner_role != PRECONDITIONER_ROLE
            or not isinstance(self.decimal_precision, int)
            or isinstance(self.decimal_precision, bool)
            or self.decimal_precision <= 0
            or not self.reason
        ):
            raise IntervalLinearRecourseError(
                "interval linear recourse provenance is malformed"
            )
        midpoint = self.midpoint_solution
        preconditioner = self.preconditioner
        if midpoint is not None and (
            len(midpoint) != self.dimension
            or not all(math.isfinite(value) for value in midpoint)
        ):
            raise IntervalLinearRecourseError(
                "midpoint solution must match the interval system dimension"
            )
        if preconditioner is not None and (
            len(preconditioner) != self.dimension
            or any(
                len(row) != self.dimension
                or not all(math.isfinite(value) for value in row)
                for row in preconditioner
            )
        ):
            raise IntervalLinearRecourseError(
                "preconditioner must be a finite square binary64 matrix"
            )
        for value in (
            self.contraction_norm_upper,
            self.preconditioned_residual_norm_upper,
            self.solution_error_radius_upper,
        ):
            if value is not None and (not math.isfinite(value) or value < 0.0):
                raise IntervalLinearRecourseError(
                    "interval linear recourse bounds must be finite and non-negative"
                )
        if self.state is (
            IntervalLinearRecourseState.CERTIFIED_UNIQUE_SOLUTION_ENCLOSURE
        ):
            if (
                self.solution_intervals is None
                or len(self.solution_intervals) != self.dimension
                or not all(
                    isinstance(bounds, IntervalBounds)
                    for bounds in self.solution_intervals
                )
                or self.contraction_norm_upper is None
                or self.contraction_norm_upper >= 1.0
                or self.preconditioned_residual_norm_upper is None
                or self.solution_error_radius_upper is None
                or midpoint is None
                or preconditioner is None
            ):
                raise IntervalLinearRecourseError(
                    "certified interval linear recourse is incomplete"
                )
        elif (
            self.solution_intervals is not None
            or self.solution_error_radius_upper is not None
        ):
            raise IntervalLinearRecourseError(
                "uncertified interval linear recourse cannot carry solution bounds"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "dimension": self.dimension,
            "solution_intervals": (
                None
                if self.solution_intervals is None
                else [bounds.as_dict() for bounds in self.solution_intervals]
            ),
            "contraction_norm_upper_binary64_hex": _optional_float_hex(
                self.contraction_norm_upper
            ),
            "preconditioned_residual_norm_upper_binary64_hex": (
                _optional_float_hex(self.preconditioned_residual_norm_upper)
            ),
            "solution_error_radius_upper_binary64_hex": _optional_float_hex(
                self.solution_error_radius_upper
            ),
            "midpoint_solution_binary64_hex": (
                None
                if self.midpoint_solution is None
                else [float(value).hex() for value in self.midpoint_solution]
            ),
            "preconditioner_binary64_hex": (
                None
                if self.preconditioner is None
                else [
                    [float(value).hex() for value in row]
                    for row in self.preconditioner
                ]
            ),
            "method_id": self.method_id,
            "preconditioner_role": self.preconditioner_role,
            "decimal_precision": self.decimal_precision,
            "reason": self.reason,
        }


def _outward_bounds(value: object) -> IntervalBounds:
    lower = float(value.a)
    upper = float(value.b)
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise IntervalLinearRecourseError(
            "interval linear arithmetic produced a non-finite enclosure"
        )
    return IntervalBounds(
        float(np.nextafter(lower, -math.inf)),
        float(np.nextafter(upper, math.inf)),
    )


def _interval_midpoint(bounds: IntervalBounds) -> float:
    return float(bounds.lower + 0.5 * (bounds.upper - bounds.lower))


def certify_interval_linear_recourse(
    *,
    coefficient_intervals: Sequence[Sequence[IntervalBounds]],
    right_hand_side_intervals: Sequence[IntervalBounds],
    decimal_precision: int,
) -> IntervalLinearRecourseCertificate:
    """Certify all solutions of ``[A] x = [b]`` with a Banach bound."""

    rows = tuple(tuple(row) for row in coefficient_intervals)
    right_hand_side = tuple(right_hand_side_intervals)
    dimension = len(rows)
    if (
        dimension == 0
        or len(right_hand_side) != dimension
        or any(len(row) != dimension for row in rows)
        or not all(
            isinstance(bounds, IntervalBounds)
            for row in rows
            for bounds in row
        )
        or not all(
            isinstance(bounds, IntervalBounds) for bounds in right_hand_side
        )
    ):
        raise IntervalLinearRecourseError(
            "coefficient and right-hand-side intervals must form A:(N,N), b:(N,)"
        )
    if (
        not isinstance(decimal_precision, int)
        or isinstance(decimal_precision, bool)
        or decimal_precision <= 0
    ):
        raise IntervalLinearRecourseError(
            "decimal_precision must be an explicit positive integer"
        )

    midpoint_matrix = np.asarray(
        [[_interval_midpoint(bounds) for bounds in row] for row in rows],
        dtype=np.float64,
    )
    midpoint_right_hand_side = np.asarray(
        [_interval_midpoint(bounds) for bounds in right_hand_side],
        dtype=np.float64,
    )
    try:
        preconditioner_array = np.linalg.inv(midpoint_matrix)
        midpoint_solution_array = np.linalg.solve(
            midpoint_matrix, midpoint_right_hand_side
        )
    except np.linalg.LinAlgError:
        return IntervalLinearRecourseCertificate(
            state=IntervalLinearRecourseState.NOT_CERTIFIABLE,
            dimension=dimension,
            solution_intervals=None,
            contraction_norm_upper=None,
            preconditioned_residual_norm_upper=None,
            solution_error_radius_upper=None,
            midpoint_solution=None,
            preconditioner=None,
            method_id=METHOD_ID,
            preconditioner_role=PRECONDITIONER_ROLE,
            decimal_precision=decimal_precision,
            reason="MIDPOINT_MATRIX_NOT_INVERTIBLE",
        )
    if not np.all(np.isfinite(preconditioner_array)) or not np.all(
        np.isfinite(midpoint_solution_array)
    ):
        return IntervalLinearRecourseCertificate(
            state=IntervalLinearRecourseState.NOT_CERTIFIABLE,
            dimension=dimension,
            solution_intervals=None,
            contraction_norm_upper=None,
            preconditioned_residual_norm_upper=None,
            solution_error_radius_upper=None,
            midpoint_solution=None,
            preconditioner=None,
            method_id=METHOD_ID,
            preconditioner_role=PRECONDITIONER_ROLE,
            decimal_precision=decimal_precision,
            reason="MIDPOINT_PRECONDITIONER_NOT_FINITE",
        )

    context = MPIntervalContext()
    context.dps = decimal_precision

    def exact(value: float) -> object:
        return context.mpf([float(value), float(value)])

    interval_matrix = tuple(
        tuple(context.mpf([bounds.lower, bounds.upper]) for bounds in row)
        for row in rows
    )
    interval_right_hand_side = tuple(
        context.mpf([bounds.lower, bounds.upper])
        for bounds in right_hand_side
    )
    interval_identity_minus_product: list[list[object]] = []
    for row in range(dimension):
        output_row: list[object] = []
        for column in range(dimension):
            value = exact(1.0 if row == column else 0.0)
            for inner in range(dimension):
                value -= (
                    exact(float(preconditioner_array[row, inner]))
                    * interval_matrix[inner][column]
                )
            output_row.append(value)
        interval_identity_minus_product.append(output_row)

    row_norm_uppers: list[float] = []
    for row in interval_identity_minus_product:
        row_sum = exact(0.0)
        for value in row:
            row_sum += abs(value)
        row_norm_uppers.append(_outward_bounds(row_sum).upper)
    contraction_norm_upper = max(row_norm_uppers)

    interval_residual: list[object] = []
    for row in range(dimension):
        value = interval_right_hand_side[row]
        for column in range(dimension):
            value -= interval_matrix[row][column] * exact(
                float(midpoint_solution_array[column])
            )
        interval_residual.append(value)
    preconditioned_residual: list[object] = []
    for row in range(dimension):
        value = exact(0.0)
        for inner in range(dimension):
            value += (
                exact(float(preconditioner_array[row, inner]))
                * interval_residual[inner]
            )
        preconditioned_residual.append(value)
    preconditioned_residual_norm_upper = max(
        _outward_bounds(abs(value)).upper
        for value in preconditioned_residual
    )

    midpoint_solution = tuple(
        float(value) for value in midpoint_solution_array
    )
    preconditioner = tuple(
        tuple(float(value) for value in row)
        for row in preconditioner_array
    )
    if (
        not math.isfinite(contraction_norm_upper)
        or contraction_norm_upper >= 1.0
    ):
        return IntervalLinearRecourseCertificate(
            state=IntervalLinearRecourseState.NOT_CERTIFIABLE,
            dimension=dimension,
            solution_intervals=None,
            contraction_norm_upper=contraction_norm_upper,
            preconditioned_residual_norm_upper=(
                preconditioned_residual_norm_upper
            ),
            solution_error_radius_upper=None,
            midpoint_solution=midpoint_solution,
            preconditioner=preconditioner,
            method_id=METHOD_ID,
            preconditioner_role=PRECONDITIONER_ROLE,
            decimal_precision=decimal_precision,
            reason="BANACH_CONTRACTION_NORM_DOES_NOT_EXCLUDE_ONE",
        )

    denominator = exact(1.0) - exact(contraction_norm_upper)
    error_radius_interval = (
        exact(preconditioned_residual_norm_upper) / denominator
    )
    solution_error_radius_upper = _outward_bounds(
        error_radius_interval
    ).upper
    solution_intervals = tuple(
        _outward_bounds(
            exact(midpoint_value)
            + context.mpf(
                [-solution_error_radius_upper, solution_error_radius_upper]
            )
        )
        for midpoint_value in midpoint_solution
    )
    return IntervalLinearRecourseCertificate(
        state=(
            IntervalLinearRecourseState.CERTIFIED_UNIQUE_SOLUTION_ENCLOSURE
        ),
        dimension=dimension,
        solution_intervals=solution_intervals,
        contraction_norm_upper=contraction_norm_upper,
        preconditioned_residual_norm_upper=(
            preconditioned_residual_norm_upper
        ),
        solution_error_radius_upper=solution_error_radius_upper,
        midpoint_solution=midpoint_solution,
        preconditioner=preconditioner,
        method_id=METHOD_ID,
        preconditioner_role=PRECONDITIONER_ROLE,
        decimal_precision=decimal_precision,
        reason="STRICT_BANACH_CONTRACTION_PROVES_ALL_UNIQUE_SOLUTIONS_ENCLOSED",
    )


__all__ = [
    "IntervalLinearRecourseCertificate",
    "IntervalLinearRecourseError",
    "IntervalLinearRecourseState",
    "METHOD_ID",
    "PRECONDITIONER_ROLE",
    "certify_interval_linear_recourse",
]
