"""Fail-closed contact-force balance over interval grasp/transmission data.

The midpoint linear program in this module proposes one strictly interior
force allocation and six adjustable ray coefficients.  It is never accepted
as evidence.  Formal evidence comes from ``interval_linear_recourse`` proving
that the six-by-six interval basis remains invertible for every admitted
matrix, followed by directed interval checks of coefficient positivity, PAD
normal-force caps, and independent-joint effort limits.

Each ray coefficient is assumed to equal that ray's normal-force contribution
at its owning PAD, matching CARTS' unit-normal polyhedral friction rays.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Sequence

from mpmath.ctx_iv import MPIntervalContext
import numpy as np
from scipy.linalg import qr
from scipy.optimize import linprog

from kcg_connector.grasp.robust.interval_kinematics import IntervalBounds
from kcg_connector.grasp.robust.interval_linear_recourse import (
    IntervalLinearRecourseCertificate,
    IntervalLinearRecourseState,
    certify_interval_linear_recourse,
)


METHOD_ID = "CARTS_DIRECTED_INTERVAL_ADJUSTABLE_SIX_RAY_BALANCE_V1"
MIDPOINT_PROPOSAL_ROLE = "BINARY64_MIDPOINT_STRICT_INTERIOR_PROPOSAL_ONLY"
FORMAL_EVIDENCE_ROLE = (
    "DIRECTED_INTERVAL_RECOURSE_POSITIVITY_PAD_AND_JOINT_LIMITS_ONLY"
)


class IntervalContactBalanceError(ValueError):
    """Raised when interval contact-balance inputs are malformed."""


class IntervalContactBalanceState(str, Enum):
    CERTIFIED_ALL_INTERVAL_STATES_BALANCED = (
        "CERTIFIED_ALL_INTERVAL_STATES_BALANCED"
    )
    NOT_CERTIFIABLE = "NOT_CERTIFIABLE"


def _optional_float_hex(value: float | None) -> str | None:
    return None if value is None else float(value).hex()


@dataclass(frozen=True)
class IntervalContactBalanceCertificate:
    """Immutable certificate for one exact external wrench."""

    state: IntervalContactBalanceState
    ray_count: int
    contact_count: int
    joint_count: int
    basis_ray_indices: tuple[int, ...] | None
    ray_coefficient_intervals: tuple[IntervalBounds, ...] | None
    pad_normal_force_intervals: tuple[IntervalBounds, ...] | None
    joint_torque_intervals: tuple[IntervalBounds, ...] | None
    midpoint_proposal: tuple[float, ...] | None
    midpoint_common_slack_fraction: float | None
    interval_recourse: IntervalLinearRecourseCertificate | None
    maximum_pad_utilization_upper: float | None
    maximum_joint_torque_utilization_upper: float | None
    method_id: str
    midpoint_proposal_role: str
    formal_evidence_role: str
    decimal_precision: int
    reason: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.state, IntervalContactBalanceState)
            or self.ray_count < 6
            or self.contact_count < 1
            or self.joint_count < 1
            or self.method_id != METHOD_ID
            or self.midpoint_proposal_role != MIDPOINT_PROPOSAL_ROLE
            or self.formal_evidence_role != FORMAL_EVIDENCE_ROLE
            or not isinstance(self.decimal_precision, int)
            or isinstance(self.decimal_precision, bool)
            or self.decimal_precision <= 0
            or not self.reason
        ):
            raise IntervalContactBalanceError(
                "interval contact-balance provenance is malformed"
            )
        basis = self.basis_ray_indices
        if basis is not None and (
            len(basis) != 6
            or len(set(basis)) != 6
            or any(index < 0 or index >= self.ray_count for index in basis)
        ):
            raise IntervalContactBalanceError(
                "adjustable basis must contain six distinct valid rays"
            )
        proposal = self.midpoint_proposal
        if proposal is not None and (
            len(proposal) != self.ray_count
            or not all(math.isfinite(value) and value >= 0.0 for value in proposal)
        ):
            raise IntervalContactBalanceError(
                "midpoint proposal must contain finite non-negative ray loads"
            )
        slack = self.midpoint_common_slack_fraction
        if slack is not None and (not math.isfinite(slack) or slack <= 0.0):
            raise IntervalContactBalanceError(
                "midpoint proposal slack must be finite and strictly positive"
            )
        for value in (
            self.maximum_pad_utilization_upper,
            self.maximum_joint_torque_utilization_upper,
        ):
            if value is not None and (not math.isfinite(value) or value < 0.0):
                raise IntervalContactBalanceError(
                    "contact-balance utilization must be finite and non-negative"
                )
        if self.interval_recourse is not None and (
            not isinstance(
                self.interval_recourse,
                IntervalLinearRecourseCertificate,
            )
            or self.interval_recourse.dimension != 6
        ):
            raise IntervalContactBalanceError(
                "contact balance must bind one six-dimensional recourse proof"
            )
        if self.state is (
            IntervalContactBalanceState.CERTIFIED_ALL_INTERVAL_STATES_BALANCED
        ):
            if (
                basis is None
                or proposal is None
                or slack is None
                or self.interval_recourse is None
                or self.interval_recourse.state
                is not IntervalLinearRecourseState.CERTIFIED_UNIQUE_SOLUTION_ENCLOSURE
                or self.ray_coefficient_intervals is None
                or len(self.ray_coefficient_intervals) != self.ray_count
                or not all(
                    bounds.strictly_positive
                    for bounds in self.ray_coefficient_intervals
                )
                or self.pad_normal_force_intervals is None
                or len(self.pad_normal_force_intervals) != self.contact_count
                or self.joint_torque_intervals is None
                or len(self.joint_torque_intervals) != self.joint_count
                or self.maximum_pad_utilization_upper is None
                or self.maximum_pad_utilization_upper > 1.0
                or self.maximum_joint_torque_utilization_upper is None
                or self.maximum_joint_torque_utilization_upper > 1.0
            ):
                raise IntervalContactBalanceError(
                    "certified interval contact balance is incomplete"
                )
        elif any(
            value is not None
            for value in (
                self.ray_coefficient_intervals,
                self.pad_normal_force_intervals,
                self.joint_torque_intervals,
                self.maximum_pad_utilization_upper,
                self.maximum_joint_torque_utilization_upper,
            )
        ):
            raise IntervalContactBalanceError(
                "uncertified contact balance cannot carry formal limit bounds"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "ray_count": self.ray_count,
            "contact_count": self.contact_count,
            "joint_count": self.joint_count,
            "basis_ray_indices": (
                None
                if self.basis_ray_indices is None
                else list(self.basis_ray_indices)
            ),
            "ray_coefficient_intervals": (
                None
                if self.ray_coefficient_intervals is None
                else [row.as_dict() for row in self.ray_coefficient_intervals]
            ),
            "pad_normal_force_intervals": (
                None
                if self.pad_normal_force_intervals is None
                else [row.as_dict() for row in self.pad_normal_force_intervals]
            ),
            "joint_torque_intervals": (
                None
                if self.joint_torque_intervals is None
                else [row.as_dict() for row in self.joint_torque_intervals]
            ),
            "midpoint_proposal_binary64_hex": (
                None
                if self.midpoint_proposal is None
                else [float(value).hex() for value in self.midpoint_proposal]
            ),
            "midpoint_common_slack_fraction_binary64_hex": (
                _optional_float_hex(self.midpoint_common_slack_fraction)
            ),
            "interval_recourse": (
                None
                if self.interval_recourse is None
                else self.interval_recourse.as_dict()
            ),
            "maximum_pad_utilization_upper_binary64_hex": (
                _optional_float_hex(self.maximum_pad_utilization_upper)
            ),
            "maximum_joint_torque_utilization_upper_binary64_hex": (
                _optional_float_hex(
                    self.maximum_joint_torque_utilization_upper
                )
            ),
            "method_id": self.method_id,
            "midpoint_proposal_role": self.midpoint_proposal_role,
            "formal_evidence_role": self.formal_evidence_role,
            "decimal_precision": self.decimal_precision,
            "reason": self.reason,
        }


def _midpoint(bounds: IntervalBounds) -> float:
    return float(bounds.lower + 0.5 * (bounds.upper - bounds.lower))


def _outward_bounds(value: object) -> IntervalBounds:
    lower = float(value.a)
    upper = float(value.b)
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise IntervalContactBalanceError(
            "directed interval contact arithmetic became non-finite"
        )
    return IntervalBounds(
        float(np.nextafter(lower, -math.inf)),
        float(np.nextafter(upper, math.inf)),
    )


def certify_interval_contact_balance(
    *,
    grasp_matrix_intervals: Sequence[Sequence[IntervalBounds]],
    joint_torque_from_ray_intervals: Sequence[Sequence[IntervalBounds]],
    ray_owner: Sequence[int],
    normal_force_caps_n: Sequence[float],
    joint_effort_limits: Sequence[float],
    external_wrench: Sequence[float],
    decimal_precision: int,
) -> IntervalContactBalanceCertificate:
    """Certify adjustable equilibrium for every supplied interval matrix."""

    grasp = tuple(tuple(row) for row in grasp_matrix_intervals)
    torque = tuple(tuple(row) for row in joint_torque_from_ray_intervals)
    owners = tuple(ray_owner)
    caps = tuple(float(value) for value in normal_force_caps_n)
    efforts = tuple(float(value) for value in joint_effort_limits)
    wrench = np.asarray(external_wrench, dtype=np.float64)
    ray_count = len(owners)
    contact_count = len(caps)
    joint_count = len(efforts)
    if (
        len(grasp) != 6
        or ray_count < 6
        or any(len(row) != ray_count for row in grasp)
        or not all(
            isinstance(bounds, IntervalBounds)
            for row in grasp
            for bounds in row
        )
        or joint_count < 1
        or len(torque) != joint_count
        or any(len(row) != ray_count for row in torque)
        or not all(
            isinstance(bounds, IntervalBounds)
            for row in torque
            for bounds in row
        )
        or contact_count < 1
        or len(owners) != ray_count
        or any(
            not isinstance(owner, int)
            or isinstance(owner, bool)
            or owner < 0
            or owner >= contact_count
            for owner in owners
        )
        or set(owners) != set(range(contact_count))
        or not all(math.isfinite(value) and value > 0.0 for value in caps)
        or not all(math.isfinite(value) and value > 0.0 for value in efforts)
        or wrench.shape != (6,)
        or not np.all(np.isfinite(wrench))
        or not isinstance(decimal_precision, int)
        or isinstance(decimal_precision, bool)
        or decimal_precision <= 0
    ):
        raise IntervalContactBalanceError(
            "contact balance needs G:(6,M), T:(J,M), valid owners, positive limits, wrench:(6,) and explicit precision"
        )

    def failure(
        reason: str,
        *,
        proposal: tuple[float, ...] | None = None,
        slack: float | None = None,
        basis: tuple[int, ...] | None = None,
        recourse: IntervalLinearRecourseCertificate | None = None,
    ) -> IntervalContactBalanceCertificate:
        return IntervalContactBalanceCertificate(
            state=IntervalContactBalanceState.NOT_CERTIFIABLE,
            ray_count=ray_count,
            contact_count=contact_count,
            joint_count=joint_count,
            basis_ray_indices=basis,
            ray_coefficient_intervals=None,
            pad_normal_force_intervals=None,
            joint_torque_intervals=None,
            midpoint_proposal=proposal,
            midpoint_common_slack_fraction=slack,
            interval_recourse=recourse,
            maximum_pad_utilization_upper=None,
            maximum_joint_torque_utilization_upper=None,
            method_id=METHOD_ID,
            midpoint_proposal_role=MIDPOINT_PROPOSAL_ROLE,
            formal_evidence_role=FORMAL_EVIDENCE_ROLE,
            decimal_precision=decimal_precision,
            reason=reason,
        )

    grasp_midpoint = np.asarray(
        [[_midpoint(bounds) for bounds in row] for row in grasp],
        dtype=np.float64,
    )
    torque_midpoint = np.asarray(
        [[_midpoint(bounds) for bounds in row] for row in torque],
        dtype=np.float64,
    )
    owner_matrix = np.zeros((contact_count, ray_count), dtype=np.float64)
    rays_per_contact = np.zeros(contact_count, dtype=np.int64)
    for ray_index, owner in enumerate(owners):
        owner_matrix[owner, ray_index] = 1.0
        rays_per_contact[owner] += 1

    variable_count = ray_count + 1
    inequality_rows: list[np.ndarray] = []
    inequality_bounds: list[float] = []
    for ray_index, owner in enumerate(owners):
        row = np.zeros(variable_count, dtype=np.float64)
        row[ray_index] = -1.0
        row[-1] = caps[owner] / float(rays_per_contact[owner])
        inequality_rows.append(row)
        inequality_bounds.append(0.0)
    for contact_index in range(contact_count):
        row = np.zeros(variable_count, dtype=np.float64)
        row[:ray_count] = owner_matrix[contact_index]
        row[-1] = caps[contact_index]
        inequality_rows.append(row)
        inequality_bounds.append(caps[contact_index])
    for joint_index in range(joint_count):
        for sign in (1.0, -1.0):
            row = np.zeros(variable_count, dtype=np.float64)
            row[:ray_count] = sign * torque_midpoint[joint_index]
            row[-1] = efforts[joint_index]
            inequality_rows.append(row)
            inequality_bounds.append(efforts[joint_index])

    objective = np.zeros(variable_count, dtype=np.float64)
    objective[-1] = -1.0
    equality_matrix = np.zeros((6, variable_count), dtype=np.float64)
    equality_matrix[:, :ray_count] = grasp_midpoint
    proposal_result = linprog(
        objective,
        A_ub=np.asarray(inequality_rows),
        b_ub=np.asarray(inequality_bounds),
        A_eq=equality_matrix,
        b_eq=-wrench,
        bounds=tuple((0.0, None) for _ in range(ray_count)) + ((0.0, 1.0),),
        method="highs",
    )
    if (
        not proposal_result.success
        or proposal_result.x is None
        or proposal_result.x.shape != (variable_count,)
        or not np.all(np.isfinite(proposal_result.x))
    ):
        return failure("MIDPOINT_STRICT_INTERIOR_PROPOSAL_INFEASIBLE")
    proposal_array = np.asarray(proposal_result.x[:ray_count], dtype=np.float64)
    slack = float(proposal_result.x[-1])
    proposal = tuple(float(value) for value in proposal_array)
    if slack <= 0.0 or np.any(proposal_array <= 0.0):
        return failure(
            "MIDPOINT_PROPOSAL_HAS_NO_STRICT_INTERIOR",
            proposal=proposal,
            slack=(slack if slack > 0.0 else None),
        )

    row_scales = np.max(np.abs(grasp_midpoint), axis=1)
    row_scales[row_scales == 0.0] = 1.0
    _, _, pivots = qr(
        grasp_midpoint / row_scales[:, None],
        mode="economic",
        pivoting=True,
    )
    basis = tuple(int(index) for index in pivots[:6])
    nonbasis = tuple(
        index for index in range(ray_count) if index not in set(basis)
    )

    context = MPIntervalContext()
    context.dps = decimal_precision

    def exact(value: float) -> object:
        return context.mpf([float(value), float(value)])

    right_hand_side: list[IntervalBounds] = []
    for row_index in range(6):
        value = -exact(float(wrench[row_index]))
        for ray_index in nonbasis:
            bounds = grasp[row_index][ray_index]
            value -= (
                context.mpf([bounds.lower, bounds.upper])
                * exact(proposal_array[ray_index])
            )
        right_hand_side.append(_outward_bounds(value))
    recourse = certify_interval_linear_recourse(
        coefficient_intervals=tuple(
            tuple(grasp[row_index][ray_index] for ray_index in basis)
            for row_index in range(6)
        ),
        right_hand_side_intervals=tuple(right_hand_side),
        decimal_precision=decimal_precision,
    )
    if recourse.state is not (
        IntervalLinearRecourseState.CERTIFIED_UNIQUE_SOLUTION_ENCLOSURE
    ):
        return failure(
            f"INTERVAL_RECOURSE_NOT_CERTIFIED:{recourse.reason}",
            proposal=proposal,
            slack=slack,
            basis=basis,
            recourse=recourse,
        )
    assert recourse.solution_intervals is not None
    coefficient_intervals_list = [
        IntervalBounds(value, value) for value in proposal
    ]
    for ray_index, bounds in zip(basis, recourse.solution_intervals):
        coefficient_intervals_list[ray_index] = bounds
    coefficient_intervals = tuple(coefficient_intervals_list)
    if any(not bounds.strictly_positive for bounds in coefficient_intervals):
        return failure(
            "ADJUSTABLE_RAY_COEFFICIENT_CAN_REACH_ZERO",
            proposal=proposal,
            slack=slack,
            basis=basis,
            recourse=recourse,
        )

    pad_intervals: list[IntervalBounds] = []
    for contact_index in range(contact_count):
        value = exact(0.0)
        for ray_index, owner in enumerate(owners):
            if owner == contact_index:
                bounds = coefficient_intervals[ray_index]
                value += context.mpf([bounds.lower, bounds.upper])
        pad_bounds = _outward_bounds(value)
        if pad_bounds.upper > caps[contact_index]:
            return failure(
                "PAD_NORMAL_FORCE_CAP_NOT_PROVEN",
                proposal=proposal,
                slack=slack,
                basis=basis,
                recourse=recourse,
            )
        pad_intervals.append(pad_bounds)

    joint_intervals: list[IntervalBounds] = []
    joint_utilizations: list[float] = []
    for joint_index in range(joint_count):
        value = exact(0.0)
        for ray_index in range(ray_count):
            torque_bounds = torque[joint_index][ray_index]
            coefficient_bounds = coefficient_intervals[ray_index]
            value += (
                context.mpf([torque_bounds.lower, torque_bounds.upper])
                * context.mpf(
                    [coefficient_bounds.lower, coefficient_bounds.upper]
                )
            )
        joint_bounds = _outward_bounds(value)
        absolute_upper = max(abs(joint_bounds.lower), abs(joint_bounds.upper))
        utilization = absolute_upper / efforts[joint_index]
        if utilization > 1.0:
            return failure(
                "JOINT_EFFORT_LIMIT_NOT_PROVEN",
                proposal=proposal,
                slack=slack,
                basis=basis,
                recourse=recourse,
            )
        joint_intervals.append(joint_bounds)
        joint_utilizations.append(utilization)

    pad_utilizations = [
        bounds.upper / cap for bounds, cap in zip(pad_intervals, caps)
    ]
    return IntervalContactBalanceCertificate(
        state=(
            IntervalContactBalanceState.CERTIFIED_ALL_INTERVAL_STATES_BALANCED
        ),
        ray_count=ray_count,
        contact_count=contact_count,
        joint_count=joint_count,
        basis_ray_indices=basis,
        ray_coefficient_intervals=coefficient_intervals,
        pad_normal_force_intervals=tuple(pad_intervals),
        joint_torque_intervals=tuple(joint_intervals),
        midpoint_proposal=proposal,
        midpoint_common_slack_fraction=slack,
        interval_recourse=recourse,
        maximum_pad_utilization_upper=max(pad_utilizations),
        maximum_joint_torque_utilization_upper=max(joint_utilizations),
        method_id=METHOD_ID,
        midpoint_proposal_role=MIDPOINT_PROPOSAL_ROLE,
        formal_evidence_role=FORMAL_EVIDENCE_ROLE,
        decimal_precision=decimal_precision,
        reason=(
            "DIRECTED_INTERVAL_RECOURSE_PROVES_BALANCE_POSITIVITY_AND_LIMITS"
        ),
    )


__all__ = [
    "FORMAL_EVIDENCE_ROLE",
    "IntervalContactBalanceCertificate",
    "IntervalContactBalanceError",
    "IntervalContactBalanceState",
    "METHOD_ID",
    "MIDPOINT_PROPOSAL_ROLE",
    "certify_interval_contact_balance",
]
