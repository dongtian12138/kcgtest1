"""Fail-closed joint-torque to tool-wrench estimation.

This module deliberately has no ROS 2 or Isaac Sim dependencies.  It solves

``tau_external = J_tool.T @ wrench_tool``

for a 6 x 7 geometric Jacobian.  The caller must supply the complete modeled
internal joint torque (gravity, payload, inertia, Coriolis/centrifugal,
friction and calibrated bias).  The estimator cannot tell a missing dynamics
term from a real external wrench.

The implementation is an inactive numerical building block: it is not wired
to the residual-v0 observation or action path.  Calibrated wrench scales,
damping, condition and projection-residual limits must be supplied explicitly.
Omitting any of them returns an invalid result instead of silently choosing
deployment thresholds.

Only a single resultant wrench at the Jacobian's known tool frame is
observable under this model.  Seven joint torques cannot localize or separate
multiple hand, payload, environment or arm-link contacts.
"""

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np


TOOL_JACOBIAN_SHAPE = (6, 7)
JOINT_TORQUE_SHAPE = (7,)
TOOL_WRENCH_SHAPE = (6,)
REQUIRED_TASK_RANK = 6


@dataclass(frozen=True)
class JointTorqueWrenchEstimate:
    """A safety-gated wrench estimate and its consistency diagnostics.

    ``wrench`` is ``None`` whenever ``valid`` is false.  For numerical gate
    failures, the seven-dimensional projection diagnostics remain available
    so a safety monitor can explain the rejection.  Structurally invalid or
    non-finite inputs cannot have meaningful projection diagnostics and use
    ``None`` for those arrays.
    """

    wrench: Optional[np.ndarray]
    external_torque: Optional[np.ndarray]
    projected_torque: Optional[np.ndarray]
    projection_residual: Optional[np.ndarray]
    projection_residual_norm_nm: float
    rank: int
    condition_number: float
    valid: bool
    reason: str


def _readonly_copy(values):
    result = np.array(values, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _invalid(
    reason,
    *,
    external_torque=None,
    projected_torque=None,
    projection_residual=None,
    projection_residual_norm_nm=math.inf,
    rank=0,
    condition_number=math.inf,
):
    return JointTorqueWrenchEstimate(
        wrench=None,
        external_torque=(
            None
            if external_torque is None
            else _readonly_copy(external_torque)
        ),
        projected_torque=(
            None
            if projected_torque is None
            else _readonly_copy(projected_torque)
        ),
        projection_residual=(
            None
            if projection_residual is None
            else _readonly_copy(projection_residual)
        ),
        projection_residual_norm_nm=float(
            projection_residual_norm_nm
        ),
        rank=int(rank),
        condition_number=float(condition_number),
        valid=False,
        reason=str(reason),
    )


def _array_or_reason(value, expected_shape, shape_reason, finite_reason):
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return None, shape_reason
    if result.shape != expected_shape:
        return None, shape_reason
    if not np.all(np.isfinite(result)):
        return None, finite_reason
    return result, None


def _scalar_or_reason(value, *, minimum, inclusive, reason):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None, reason
    if not math.isfinite(result):
        return None, reason
    if inclusive:
        accepted = result >= minimum
    else:
        accepted = result > minimum
    if not accepted:
        return None, reason
    return result, None


def estimate_tool_wrench(
    jacobian,
    measured_torque,
    modeled_internal_torque,
    *,
    weights=None,
    wrench_scales=None,
    damping=None,
    maximum_condition_number=None,
    maximum_projection_residual_nm=None,
    rank_tolerance=None,
):
    """Estimate one tool-frame wrench with weighted damped least squares.

    Args:
        jacobian: Geometric tool Jacobian with shape ``(6, 7)``.  Its twist
            and wrench order must be ``[x, y, z, rx, ry, rz]`` and
            ``[Fx, Fy, Fz, Mx, My, Mz]`` respectively.
        measured_torque: Seven measured iiwa joint torques in Nm.
        modeled_internal_torque: Seven caller-modeled non-contact torques in
            Nm.  This input is mandatory; the estimator does not compute robot
            dynamics itself.
        weights: Optional seven positive dimensionless measurement-reliability
            weights.  Larger values give a joint channel more influence.  The
            default is equal weighting.
        wrench_scales: Six positive task-metric scales ordered as
            ``[Fx, Fy, Fz, Mx, My, Mz]``, with force entries in N and moment
            entries in Nm.  The solver uses ``wrench = diag(scales) @ u`` and
            computes rank and condition on the dimensionally normalized
            coordinate ``u``.  This task-specific metric must be fixed and
            calibrated; ``None`` keeps the estimator disabled.
        damping: Positive calibrated singular-value damping.  ``None`` keeps
            this design-only estimator disabled.
        maximum_condition_number: Calibrated upper bound for the weighted
            torque-to-wrench operator's condition number.
        maximum_projection_residual_nm: Calibrated upper bound for the L2 norm
            of ``tau_external - jacobian.T @ wrench``.
        rank_tolerance: Optional non-negative SVD rank tolerance.  NumPy's
            matrix-rank tolerance is reproduced when omitted.

    Returns:
        :class:`JointTorqueWrenchEstimate`.  Every rejected case has
        ``valid=False``, a stable ``reason``, and ``wrench=None``.

    Notes:
        The weighted objective is

        ``||sqrt(W) (J.T S u - tau_external)||^2 + damping^2 ||u||^2``,

        where ``S = diag(wrench_scales)`` and ``wrench = S u``.  Raw force and
        moment coordinates have different units, so a condition number without
        this fixed task metric is not used as a safety gate.

        A small projection residual proves only consistency with one
        tool-frame resultant.  It does not prove that dynamics compensation
        is correct, nor identify a contact point or multiple contacts.
    """
    jacobian_array, reason = _array_or_reason(
        jacobian,
        TOOL_JACOBIAN_SHAPE,
        "invalid_jacobian_shape",
        "nonfinite_jacobian",
    )
    if reason is not None:
        return _invalid(reason)

    measured_array, reason = _array_or_reason(
        measured_torque,
        JOINT_TORQUE_SHAPE,
        "invalid_measured_torque_shape",
        "nonfinite_measured_torque",
    )
    if reason is not None:
        return _invalid(reason)

    modeled_array, reason = _array_or_reason(
        modeled_internal_torque,
        JOINT_TORQUE_SHAPE,
        "invalid_modeled_internal_torque_shape",
        "nonfinite_modeled_internal_torque",
    )
    if reason is not None:
        return _invalid(reason)

    if weights is None:
        weights_array = np.ones(JOINT_TORQUE_SHAPE, dtype=np.float64)
    else:
        weights_array, reason = _array_or_reason(
            weights,
            JOINT_TORQUE_SHAPE,
            "invalid_weights_shape",
            "nonfinite_weights",
        )
        if reason is not None:
            return _invalid(reason)
        if np.any(weights_array <= 0.0):
            return _invalid("nonpositive_weights")

    if wrench_scales is None:
        wrench_scales_array = None
    else:
        wrench_scales_array, reason = _array_or_reason(
            wrench_scales,
            TOOL_WRENCH_SHAPE,
            "invalid_wrench_scales_shape",
            "nonfinite_wrench_scales",
        )
        if reason is not None:
            return _invalid(reason)
        if np.any(wrench_scales_array <= 0.0):
            return _invalid("nonpositive_wrench_scales")

    with np.errstate(over="ignore", invalid="ignore"):
        external_torque = measured_array - modeled_array
    if not np.all(np.isfinite(external_torque)):
        return _invalid("nonfinite_external_torque")

    if (
        wrench_scales_array is None
        or damping is None
        or maximum_condition_number is None
        or maximum_projection_residual_nm is None
    ):
        return _invalid(
            "uncalibrated_safety_limits",
            external_torque=external_torque,
        )

    damping_value, reason = _scalar_or_reason(
        damping,
        minimum=0.0,
        inclusive=False,
        reason="invalid_damping",
    )
    if reason is not None:
        return _invalid(reason, external_torque=external_torque)

    maximum_condition, reason = _scalar_or_reason(
        maximum_condition_number,
        minimum=1.0,
        inclusive=True,
        reason="invalid_maximum_condition_number",
    )
    if reason is not None:
        return _invalid(reason, external_torque=external_torque)

    maximum_residual, reason = _scalar_or_reason(
        maximum_projection_residual_nm,
        minimum=0.0,
        inclusive=True,
        reason="invalid_maximum_projection_residual_nm",
    )
    if reason is not None:
        return _invalid(reason, external_torque=external_torque)

    if rank_tolerance is None:
        tolerance_value = None
    else:
        tolerance_value, reason = _scalar_or_reason(
            rank_tolerance,
            minimum=0.0,
            inclusive=True,
            reason="invalid_rank_tolerance",
        )
        if reason is not None:
            return _invalid(reason, external_torque=external_torque)

    sqrt_weights = np.sqrt(weights_array)
    with np.errstate(over="ignore", invalid="ignore"):
        weighted_operator = sqrt_weights[:, None] * jacobian_array.T
        scaled_weighted_operator = (
            weighted_operator * wrench_scales_array[None, :]
        )
        weighted_external_torque = sqrt_weights * external_torque
    if (
        not np.all(np.isfinite(weighted_operator))
        or not np.all(np.isfinite(scaled_weighted_operator))
        or not np.all(np.isfinite(weighted_external_torque))
    ):
        return _invalid(
            "nonfinite_weighted_input",
            external_torque=external_torque,
        )

    try:
        left_vectors, singular_values, right_vectors_t = np.linalg.svd(
            scaled_weighted_operator,
            full_matrices=False,
        )
    except np.linalg.LinAlgError:
        return _invalid("svd_failed", external_torque=external_torque)
    if not np.all(np.isfinite(singular_values)):
        return _invalid(
            "nonfinite_singular_values",
            external_torque=external_torque,
        )

    if tolerance_value is None:
        tolerance_value = (
            max(scaled_weighted_operator.shape)
            * np.finfo(np.float64).eps
            * singular_values[0]
        )
    rank = int(np.count_nonzero(singular_values > tolerance_value))
    if rank < REQUIRED_TASK_RANK:
        condition_number = math.inf
    else:
        condition_number = float(
            singular_values[0] / singular_values[-1]
        )

    filter_factors = np.zeros_like(singular_values)
    positive = singular_values > 0.0
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        damping_ratios = damping_value / singular_values[positive]
        filter_factors[positive] = 1.0 / (
            singular_values[positive]
            * (1.0 + damping_ratios * damping_ratios)
        )
        normalized_wrench = right_vectors_t.T @ (
            filter_factors * (left_vectors.T @ weighted_external_torque)
        )
        candidate_wrench = wrench_scales_array * normalized_wrench
        projected_torque = jacobian_array.T @ candidate_wrench
        projection_residual = external_torque - projected_torque
    projection_residual_norm = float(np.linalg.norm(projection_residual))

    numerical_outputs = (
        candidate_wrench,
        projected_torque,
        projection_residual,
    )
    if not all(np.all(np.isfinite(value)) for value in numerical_outputs):
        return _invalid(
            "nonfinite_solver_output",
            external_torque=external_torque,
            rank=rank,
            condition_number=condition_number,
        )

    diagnostics = {
        "external_torque": external_torque,
        "projected_torque": projected_torque,
        "projection_residual": projection_residual,
        "projection_residual_norm_nm": projection_residual_norm,
        "rank": rank,
        "condition_number": condition_number,
    }
    if rank < REQUIRED_TASK_RANK:
        return _invalid("rank_below_six", **diagnostics)
    if condition_number > maximum_condition:
        return _invalid("condition_number_exceeded", **diagnostics)
    if projection_residual_norm > maximum_residual:
        return _invalid("projection_residual_exceeded", **diagnostics)

    return JointTorqueWrenchEstimate(
        wrench=_readonly_copy(candidate_wrench),
        external_torque=_readonly_copy(external_torque),
        projected_torque=_readonly_copy(projected_torque),
        projection_residual=_readonly_copy(projection_residual),
        projection_residual_norm_nm=projection_residual_norm,
        rank=rank,
        condition_number=condition_number,
        valid=True,
        reason="ok",
    )


__all__ = [
    "JOINT_TORQUE_SHAPE",
    "REQUIRED_TASK_RANK",
    "TOOL_JACOBIAN_SHAPE",
    "TOOL_WRENCH_SHAPE",
    "JointTorqueWrenchEstimate",
    "estimate_tool_wrench",
]
