"""Truth-isolated execution mathematics for CARTS-Grasp.

The module contains two independent, auditable pieces of the execution stack:

* a six-dimensional object impedance whose storage and dissipation balance is
  reported for every evaluation; and
* a convex grasp-force QP in the ray coordinates of
  :class:`PolyhedralContactWrenchModel`.

The QP uses the same certified inner friction polygons as the robust planner.
Non-negative ray coefficients enforce unilateral contact and the polygonal
friction cones.  The model's linear inequalities enforce per-contact normal
caps and any planner-supplied force constraints.  Optional joint-torque limits
are appended as linear inequalities.  There is no simulator dependency and no
object-specific candidate, coordinate, or acceptance threshold in this file.

All numerical tolerances and iteration budgets are mandatory fields of an
explicit configuration mapping.  They are numerical-resolution contracts, not
physical grasp-pass gates.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.linalg import qr
from scipy.optimize import linprog, minimize
from scipy.spatial.transform import Rotation

from .robust_wrench import PolyhedralContactWrenchModel


_FORCE_QP_CONSTRAINT_SCALING_IMPLEMENTATION = (
    "CALLER_WRENCH_DIAGONAL_THEN_AUGMENTED_ROW_INF_NORM_V1"
)


def _readonly(value: np.ndarray | Sequence[float]) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=np.float64)
    result.setflags(write=False)
    return result


def _finite_array(
    value: Any,
    *,
    shape: tuple[int, ...] | None,
    name: str,
) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if shape is not None and result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _augmented_row_infinity_norms(
    matrix: np.ndarray, target: np.ndarray
) -> np.ndarray:
    """Return exact row divisors from the augmented rows ``[A_i, b_i]``.

    A structurally zero augmented row uses the algebraic identity divisor one.
    This rule contains no empirical cutoff and therefore commutes exactly with
    multiplication of any nonzero constraint row by a scalar.
    """

    coefficient_norms = np.max(np.abs(matrix), axis=1)
    row_norms = np.maximum(coefficient_norms, np.abs(target))
    row_norms[row_norms == 0.0] = 1.0
    return row_norms


def _positive(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


def _nonnegative(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a non-negative finite number")
    return result


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _exact_mapping_keys(
    document: Mapping[str, Any], *, expected: set[str], name: str
) -> None:
    missing = sorted(expected.difference(document))
    unknown = sorted(set(document).difference(expected))
    if missing or unknown:
        raise ValueError(
            f"{name} has missing keys {missing} and unknown keys {unknown}"
        )


@dataclass(frozen=True)
class PassiveImpedanceNumerics:
    """Numerical resolution contract for SE(3) and passivity audits."""

    matrix_symmetry_absolute_tolerance: float
    semidefinite_eigenvalue_tolerance: float
    rotation_orthogonality_tolerance: float
    homogeneous_row_tolerance: float
    passivity_balance_tolerance: float

    def __post_init__(self) -> None:
        for field_name in (
            "matrix_symmetry_absolute_tolerance",
            "semidefinite_eigenvalue_tolerance",
            "rotation_orthogonality_tolerance",
            "homogeneous_row_tolerance",
            "passivity_balance_tolerance",
        ):
            _positive(getattr(self, field_name), name=field_name)

    @classmethod
    def from_mapping(
        cls, document: Mapping[str, Any]
    ) -> "PassiveImpedanceNumerics":
        expected = {
            "matrix_symmetry_absolute_tolerance",
            "semidefinite_eigenvalue_tolerance",
            "rotation_orthogonality_tolerance",
            "homogeneous_row_tolerance",
            "passivity_balance_tolerance",
        }
        _exact_mapping_keys(
            document, expected=expected, name="passive_impedance_numerics"
        )
        return cls(
            matrix_symmetry_absolute_tolerance=_positive(
                document["matrix_symmetry_absolute_tolerance"],
                name="matrix_symmetry_absolute_tolerance",
            ),
            semidefinite_eigenvalue_tolerance=_positive(
                document["semidefinite_eigenvalue_tolerance"],
                name="semidefinite_eigenvalue_tolerance",
            ),
            rotation_orthogonality_tolerance=_positive(
                document["rotation_orthogonality_tolerance"],
                name="rotation_orthogonality_tolerance",
            ),
            homogeneous_row_tolerance=_positive(
                document["homogeneous_row_tolerance"],
                name="homogeneous_row_tolerance",
            ),
            passivity_balance_tolerance=_positive(
                document["passivity_balance_tolerance"],
                name="passivity_balance_tolerance",
            ),
        )


@dataclass(frozen=True)
class ForceAllocationSolverOptions:
    """Mandatory numerical contract for the feasibility LP and convex QP."""

    solver: str
    constraint_scaling: str
    maximum_iterations: int
    objective_tolerance: float
    equality_tolerance: float
    inequality_tolerance: float
    linear_independence_tolerance: float
    feasibility_dual_tolerance: float
    regularization: float
    physical_acceptance_gate: bool

    def __post_init__(self) -> None:
        if self.solver != "SCIPY_SLSQP_WITH_HIGHS_FEASIBILITY":
            raise ValueError(
                "only SCIPY_SLSQP_WITH_HIGHS_FEASIBILITY is supported"
            )
        if self.constraint_scaling != "CALLER_SUPPLIED_WRENCH_SCALES":
            raise ValueError(
                "constraint_scaling must be CALLER_SUPPLIED_WRENCH_SCALES"
            )
        _positive_integer(self.maximum_iterations, name="maximum_iterations")
        for field_name in (
            "objective_tolerance",
            "equality_tolerance",
            "inequality_tolerance",
            "linear_independence_tolerance",
            "feasibility_dual_tolerance",
            "regularization",
        ):
            _positive(getattr(self, field_name), name=field_name)
        if self.physical_acceptance_gate is not False:
            raise ValueError(
                "force-allocation numerical residuals cannot be a physical pass gate"
            )

    @classmethod
    def from_mapping(
        cls, document: Mapping[str, Any]
    ) -> "ForceAllocationSolverOptions":
        expected = {
            "solver",
            "constraint_scaling",
            "maximum_iterations",
            "objective_tolerance",
            "equality_tolerance",
            "inequality_tolerance",
            "linear_independence_tolerance",
            "feasibility_dual_tolerance",
            "regularization",
            "physical_acceptance_gate",
        }
        _exact_mapping_keys(document, expected=expected, name="force_qp")
        return cls(
            solver=str(document["solver"]),
            constraint_scaling=str(document["constraint_scaling"]),
            maximum_iterations=_positive_integer(
                document["maximum_iterations"], name="maximum_iterations"
            ),
            objective_tolerance=_positive(
                document["objective_tolerance"], name="objective_tolerance"
            ),
            equality_tolerance=_positive(
                document["equality_tolerance"], name="equality_tolerance"
            ),
            inequality_tolerance=_positive(
                document["inequality_tolerance"], name="inequality_tolerance"
            ),
            linear_independence_tolerance=_positive(
                document["linear_independence_tolerance"],
                name="linear_independence_tolerance",
            ),
            feasibility_dual_tolerance=_positive(
                document["feasibility_dual_tolerance"],
                name="feasibility_dual_tolerance",
            ),
            regularization=_positive(
                document["regularization"], name="regularization"
            ),
            physical_acceptance_gate=document["physical_acceptance_gate"],
        )


@dataclass(frozen=True)
class OnlineTruthFirewallAudit:
    """Proof-carrying record for the permitted online observation surface."""

    accepted_online_fields: tuple[str, ...]
    estimator_source: str
    ground_truth_pose_used: bool
    contact_point_or_normal_truth_used: bool
    collider_identity_used: bool
    physx_contact_report_used: bool

    def __post_init__(self) -> None:
        if not self.estimator_source:
            raise ValueError("estimator_source cannot be empty")
        if any(
            value is not False
            for value in (
                self.ground_truth_pose_used,
                self.contact_point_or_normal_truth_used,
                self.collider_identity_used,
                self.physx_contact_report_used,
            )
        ):
            raise ValueError("online truth-firewall audit must fail closed")


_ONLINE_REQUIRED_FIELDS = {
    "estimated_pose_world_from_object",
    "estimated_twist_world",
    "estimator_source",
}
_ONLINE_OPTIONAL_FIELDS = {
    "joint_positions",
    "joint_velocities",
    "joint_torques",
    "wrist_wrench",
    "tactile_measurements",
    "timestamp_s",
}
_FORBIDDEN_ONLINE_FRAGMENTS = (
    "ground_truth",
    "groundtruth",
    "contact_point",
    "contact_normal",
    "contact_report",
    "contact_truth",
    "contact_name",
    "collider",
    "collision_prim",
    "physx",
)


def _contains_forbidden_online_fragment(value: str) -> str | None:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    for fragment in _FORBIDDEN_ONLINE_FRAGMENTS:
        if fragment in normalized:
            return fragment
    return None


def _validate_transform(
    value: Any,
    *,
    numerics: PassiveImpedanceNumerics,
    name: str,
) -> np.ndarray:
    transform = _finite_array(value, shape=(4, 4), name=name)
    row_error = float(
        np.max(np.abs(transform[3] - np.asarray((0.0, 0.0, 0.0, 1.0))))
    )
    if row_error > numerics.homogeneous_row_tolerance:
        raise ValueError(f"{name} has an invalid homogeneous row")
    rotation = transform[:3, :3]
    orthogonality_error = float(
        np.max(np.abs(rotation.T @ rotation - np.eye(3)))
    )
    determinant_error = abs(float(np.linalg.det(rotation)) - 1.0)
    if (
        orthogonality_error > numerics.rotation_orthogonality_tolerance
        or determinant_error > numerics.rotation_orthogonality_tolerance
    ):
        raise ValueError(f"{name} must contain a proper rotation")
    return transform


@dataclass(frozen=True)
class OnlineControlObservation:
    """Whitelisted estimator/proprioception input; simulator truth has no slot."""

    estimated_pose_world_from_object: np.ndarray
    estimated_twist_world: np.ndarray
    estimator_source: str
    joint_positions: np.ndarray | None
    joint_velocities: np.ndarray | None
    joint_torques: np.ndarray | None
    wrist_wrench: np.ndarray | None
    tactile_measurements: np.ndarray | None
    timestamp_s: float | None
    truth_firewall_audit: OnlineTruthFirewallAudit

    @classmethod
    def from_mapping(
        cls,
        document: Mapping[str, Any],
        *,
        numerics: PassiveImpedanceNumerics,
    ) -> "OnlineControlObservation":
        if not isinstance(document, Mapping):
            raise ValueError("online observation must be a mapping")
        fields = set(document)
        missing = sorted(_ONLINE_REQUIRED_FIELDS.difference(fields))
        unknown = sorted(
            fields.difference(_ONLINE_REQUIRED_FIELDS | _ONLINE_OPTIONAL_FIELDS)
        )
        rejected_fragments = {
            fragment
            for key in fields
            for fragment in (_contains_forbidden_online_fragment(str(key)),)
            if fragment is not None
        }
        if missing or unknown or rejected_fragments:
            raise ValueError(
                "online observation rejected: "
                f"missing={missing}, unknown={unknown}, "
                f"forbidden_fragments={sorted(rejected_fragments)}"
            )
        source = str(document["estimator_source"])
        source_fragment = _contains_forbidden_online_fragment(source)
        if not source or source_fragment is not None:
            raise ValueError(
                "estimator_source is empty or names a forbidden truth source"
            )

        def optional_vector(field_name: str) -> np.ndarray | None:
            if field_name not in document or document[field_name] is None:
                return None
            value = _finite_array(document[field_name], shape=None, name=field_name)
            if value.ndim != 1:
                raise ValueError(f"{field_name} must be a one-dimensional vector")
            return _readonly(value)

        timestamp: float | None = None
        if "timestamp_s" in document and document["timestamp_s"] is not None:
            if isinstance(document["timestamp_s"], bool):
                raise ValueError("timestamp_s must be finite")
            timestamp = float(document["timestamp_s"])
            if not math.isfinite(timestamp):
                raise ValueError("timestamp_s must be finite")
        audit = OnlineTruthFirewallAudit(
            accepted_online_fields=tuple(sorted(fields)),
            estimator_source=source,
            ground_truth_pose_used=False,
            contact_point_or_normal_truth_used=False,
            collider_identity_used=False,
            physx_contact_report_used=False,
        )
        return cls(
            estimated_pose_world_from_object=_readonly(
                _validate_transform(
                    document["estimated_pose_world_from_object"],
                    numerics=numerics,
                    name="estimated_pose_world_from_object",
                )
            ),
            estimated_twist_world=_readonly(
                _finite_array(
                    document["estimated_twist_world"],
                    shape=(6,),
                    name="estimated_twist_world",
                )
            ),
            estimator_source=source,
            joint_positions=optional_vector("joint_positions"),
            joint_velocities=optional_vector("joint_velocities"),
            joint_torques=optional_vector("joint_torques"),
            wrist_wrench=optional_vector("wrist_wrench"),
            tactile_measurements=optional_vector("tactile_measurements"),
            timestamp_s=timestamp,
            truth_firewall_audit=audit,
        )


def object_pose_error_world(
    *,
    desired_pose_world_from_object: Sequence[Sequence[float]],
    estimated_pose_world_from_object: Sequence[Sequence[float]],
    numerics: PassiveImpedanceNumerics,
) -> np.ndarray:
    """Return world-frame translation and rotation-vector error.

    The rotation vector maps the current estimated orientation to the desired
    orientation and therefore transforms covariantly under a rigid change of
    world coordinates.
    """

    desired = _validate_transform(
        desired_pose_world_from_object,
        numerics=numerics,
        name="desired_pose_world_from_object",
    )
    estimated = _validate_transform(
        estimated_pose_world_from_object,
        numerics=numerics,
        name="estimated_pose_world_from_object",
    )
    translation_error = desired[:3, 3] - estimated[:3, 3]
    rotation_error_world = Rotation.from_matrix(
        desired[:3, :3] @ estimated[:3, :3].T
    ).as_rotvec()
    return _readonly(np.concatenate((translation_error, rotation_error_world)))


@dataclass(frozen=True)
class PassiveImpedanceAudit:
    elastic_storage_j: float
    damping_dissipation_rate_w: float
    controller_power_to_object_w: float
    elastic_storage_rate_w: float
    closed_loop_mechanical_energy_rate_w: float
    passivity_balance_residual_w: float
    passivity_balance_tolerance: float
    passivity_identity_within_numerical_tolerance: bool
    stiffness_minimum_eigenvalue: float
    damping_minimum_eigenvalue_before_projection: float
    damping_projection_frobenius_norm: float


@dataclass(frozen=True)
class PassiveImpedanceResult:
    commanded_object_wrench: np.ndarray
    pose_error: np.ndarray
    object_twist: np.ndarray
    audit: PassiveImpedanceAudit
    truth_firewall_audit: OnlineTruthFirewallAudit | None


class PassiveObjectImpedance:
    """Six-dimensional spring-damper with an explicit passivity identity."""

    def __init__(
        self,
        *,
        stiffness: Sequence[Sequence[float]],
        damping: Sequence[Sequence[float]],
        numerics: PassiveImpedanceNumerics,
    ) -> None:
        self.numerics = numerics
        stiffness_raw = _finite_array(stiffness, shape=(6, 6), name="stiffness")
        damping_raw = _finite_array(damping, shape=(6, 6), name="damping")
        self.stiffness = _readonly(
            self._symmetric_matrix(
                stiffness_raw, name="stiffness", require_positive_definite=True
            )[0]
        )
        damping_validated, damping_minimum, projection_norm = self._symmetric_matrix(
            damping_raw, name="damping", require_positive_definite=False
        )
        self.damping = _readonly(damping_validated)
        self._stiffness_minimum_eigenvalue = float(
            np.min(np.linalg.eigvalsh(self.stiffness))
        )
        self._damping_minimum_before_projection = damping_minimum
        self._damping_projection_norm = projection_norm

    def _symmetric_matrix(
        self,
        matrix: np.ndarray,
        *,
        name: str,
        require_positive_definite: bool,
    ) -> tuple[np.ndarray, float, float]:
        symmetry_error = float(np.max(np.abs(matrix - matrix.T)))
        if symmetry_error > self.numerics.matrix_symmetry_absolute_tolerance:
            raise ValueError(f"{name} must be symmetric")
        symmetric = 0.5 * (matrix + matrix.T)
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
        minimum = float(np.min(eigenvalues))
        if require_positive_definite:
            if minimum <= self.numerics.semidefinite_eigenvalue_tolerance:
                raise ValueError(
                    "stiffness must be positive definite at configured resolution"
                )
            return symmetric, minimum, 0.0
        if minimum < -self.numerics.semidefinite_eigenvalue_tolerance:
            raise ValueError("damping must be positive semidefinite")
        projected_eigenvalues = np.maximum(eigenvalues, 0.0)
        projected = (eigenvectors * projected_eigenvalues) @ eigenvectors.T
        projection_norm = float(np.linalg.norm(projected - symmetric, ord="fro"))
        return projected, minimum, projection_norm

    def evaluate(
        self,
        *,
        pose_error: Sequence[float],
        object_twist: Sequence[float],
        truth_firewall_audit: OnlineTruthFirewallAudit | None,
    ) -> PassiveImpedanceResult:
        error = _finite_array(pose_error, shape=(6,), name="pose_error")
        twist = _finite_array(object_twist, shape=(6,), name="object_twist")
        spring_wrench = self.stiffness @ error
        damping_wrench = self.damping @ twist
        commanded = spring_wrench - damping_wrench
        storage = 0.5 * float(error @ self.stiffness @ error)
        dissipation = float(twist @ self.damping @ twist)
        controller_power = float(commanded @ twist)
        storage_rate = -float(spring_wrench @ twist)
        closed_loop_rate = storage_rate + controller_power
        balance_residual = closed_loop_rate + dissipation
        audit = PassiveImpedanceAudit(
            elastic_storage_j=storage,
            damping_dissipation_rate_w=dissipation,
            controller_power_to_object_w=controller_power,
            elastic_storage_rate_w=storage_rate,
            closed_loop_mechanical_energy_rate_w=closed_loop_rate,
            passivity_balance_residual_w=balance_residual,
            passivity_balance_tolerance=self.numerics.passivity_balance_tolerance,
            passivity_identity_within_numerical_tolerance=(
                abs(balance_residual)
                <= self.numerics.passivity_balance_tolerance
            ),
            stiffness_minimum_eigenvalue=self._stiffness_minimum_eigenvalue,
            damping_minimum_eigenvalue_before_projection=(
                self._damping_minimum_before_projection
            ),
            damping_projection_frobenius_norm=self._damping_projection_norm,
        )
        return PassiveImpedanceResult(
            commanded_object_wrench=_readonly(commanded),
            pose_error=_readonly(error),
            object_twist=_readonly(twist),
            audit=audit,
            truth_firewall_audit=truth_firewall_audit,
        )

    def evaluate_observation(
        self,
        *,
        observation: OnlineControlObservation,
        desired_pose_world_from_object: Sequence[Sequence[float]],
    ) -> PassiveImpedanceResult:
        error = object_pose_error_world(
            desired_pose_world_from_object=desired_pose_world_from_object,
            estimated_pose_world_from_object=(
                observation.estimated_pose_world_from_object
            ),
            numerics=self.numerics,
        )
        return self.evaluate(
            pose_error=error,
            object_twist=observation.estimated_twist_world,
            truth_firewall_audit=observation.truth_firewall_audit,
        )


@dataclass(frozen=True)
class JointTorqueLinearConstraint:
    """Affine joint torques ``tau = map @ stacked_contact_force + bias``."""

    contact_force_to_joint_torque: np.ndarray
    bias_torque: np.ndarray
    lower_torque: np.ndarray
    upper_torque: np.ndarray

    @classmethod
    def from_arrays(
        cls,
        *,
        contact_force_to_joint_torque: Sequence[Sequence[float]],
        bias_torque: Sequence[float],
        lower_torque: Sequence[float],
        upper_torque: Sequence[float],
    ) -> "JointTorqueLinearConstraint":
        matrix = _finite_array(
            contact_force_to_joint_torque,
            shape=None,
            name="contact_force_to_joint_torque",
        )
        if matrix.ndim != 2 or matrix.shape[0] < 1:
            raise ValueError(
                "contact_force_to_joint_torque must have shape (J, 3N), J >= 1"
            )
        joint_count = int(matrix.shape[0])
        bias = _finite_array(
            bias_torque, shape=(joint_count,), name="bias_torque"
        )
        lower = _finite_array(
            lower_torque, shape=(joint_count,), name="lower_torque"
        )
        upper = _finite_array(
            upper_torque, shape=(joint_count,), name="upper_torque"
        )
        if np.any(lower > upper):
            raise ValueError("joint torque lower bounds exceed upper bounds")
        return cls(
            contact_force_to_joint_torque=_readonly(matrix),
            bias_torque=_readonly(bias),
            lower_torque=_readonly(lower),
            upper_torque=_readonly(upper),
        )


@dataclass(frozen=True)
class GraspForceAllocationResult:
    """Force-QP result with separate numerical and physical audit planes.

    Only the dimensionless ``solver_coordinate_*`` residual maxima participate
    in ``solver_success``.  The ``physical_*`` arrays retain their native units
    and row lineage for audit; they are never combined into a mixed-unit scalar
    and are never compared with a numerical solver tolerance.
    """

    solver_success: bool
    solver_status: int
    solver_message: str
    solver_options: ForceAllocationSolverOptions
    constraint_scaling_implementation: str
    equality_rank: int
    wrench_scaling: np.ndarray
    equality_augmented_row_inf_norms_after_wrench_scaling: np.ndarray
    inequality_augmented_row_inf_norms: np.ndarray
    inequality_row_labels: tuple[str, ...]
    ray_coefficients: np.ndarray | None
    contact_forces: np.ndarray | None
    normal_forces: np.ndarray | None
    achieved_object_wrench: np.ndarray | None
    desired_object_wrench: np.ndarray
    joint_torques: np.ndarray | None
    objective_value: float | None
    solver_coordinate_equality_residuals: np.ndarray | None
    solver_coordinate_inequality_residuals: np.ndarray | None
    maximum_solver_coordinate_equality_residual: float | None
    maximum_solver_coordinate_inequality_violation: float | None
    physical_equilibrium_force_residual_n: np.ndarray | None
    physical_equilibrium_moment_residual_nm: np.ndarray | None
    physical_normal_cap_residuals_n: np.ndarray | None
    physical_additional_model_inequality_residuals_by_row: np.ndarray | None
    physical_joint_upper_torque_residuals_nm: np.ndarray | None
    physical_joint_lower_torque_residuals_nm: np.ndarray | None
    physical_ray_nonnegativity_residuals_n: np.ndarray | None
    internal_ray_component: np.ndarray | None
    internal_contact_forces: np.ndarray | None
    solver_coordinate_internal_wrench_residuals: np.ndarray | None
    physical_internal_force_residual_n: np.ndarray | None
    physical_internal_moment_residual_nm: np.ndarray | None
    truth_firewall_audit: OnlineTruthFirewallAudit | None


def _failure_result(
    *,
    status: int,
    message: str,
    options: ForceAllocationSolverOptions,
    equality_rank: int,
    desired: np.ndarray,
    wrench_scaling: np.ndarray,
    equality_row_norms: np.ndarray,
    inequality_row_norms: np.ndarray,
    inequality_row_labels: tuple[str, ...],
    truth_firewall_audit: OnlineTruthFirewallAudit | None,
) -> GraspForceAllocationResult:
    return GraspForceAllocationResult(
        solver_success=False,
        solver_status=status,
        solver_message=message,
        solver_options=options,
        constraint_scaling_implementation=(
            _FORCE_QP_CONSTRAINT_SCALING_IMPLEMENTATION
        ),
        equality_rank=equality_rank,
        wrench_scaling=_readonly(wrench_scaling),
        equality_augmented_row_inf_norms_after_wrench_scaling=_readonly(
            equality_row_norms
        ),
        inequality_augmented_row_inf_norms=_readonly(inequality_row_norms),
        inequality_row_labels=inequality_row_labels,
        ray_coefficients=None,
        contact_forces=None,
        normal_forces=None,
        achieved_object_wrench=None,
        desired_object_wrench=_readonly(desired),
        joint_torques=None,
        objective_value=None,
        solver_coordinate_equality_residuals=None,
        solver_coordinate_inequality_residuals=None,
        maximum_solver_coordinate_equality_residual=None,
        maximum_solver_coordinate_inequality_violation=None,
        physical_equilibrium_force_residual_n=None,
        physical_equilibrium_moment_residual_nm=None,
        physical_normal_cap_residuals_n=None,
        physical_additional_model_inequality_residuals_by_row=None,
        physical_joint_upper_torque_residuals_nm=None,
        physical_joint_lower_torque_residuals_nm=None,
        physical_ray_nonnegativity_residuals_n=None,
        internal_ray_component=None,
        internal_contact_forces=None,
        solver_coordinate_internal_wrench_residuals=None,
        physical_internal_force_residual_n=None,
        physical_internal_moment_residual_nm=None,
        truth_firewall_audit=truth_firewall_audit,
    )


def allocate_grasp_forces(
    model: PolyhedralContactWrenchModel,
    *,
    desired_object_wrench: Sequence[float],
    preferred_normal_forces: Sequence[float],
    contact_force_quadratic_weights: Sequence[float],
    normal_tracking_quadratic_weights: Sequence[float],
    wrench_scaling: Sequence[float],
    joint_torque_constraint: JointTorqueLinearConstraint | None,
    solver_options: ForceAllocationSolverOptions,
    truth_firewall_audit: OnlineTruthFirewallAudit | None,
) -> GraspForceAllocationResult:
    """Allocate a desired object wrench plus grasp-map-nullspace preload.

    The decision variables are non-negative friction-cone ray coefficients.
    The objective minimizes contact-force burden while tracking a planner
    supplied normal preload.  A strictly positive configured regularization
    makes the ray-coordinate QP unique even when cone rays are redundant.
    """

    desired = _finite_array(
        desired_object_wrench, shape=(6,), name="desired_object_wrench"
    )
    contact_count = model.contact_count
    ray_count = model.ray_count
    preferred = _finite_array(
        preferred_normal_forces,
        shape=(contact_count,),
        name="preferred_normal_forces",
    )
    if np.any(preferred < 0.0):
        raise ValueError("preferred_normal_forces must be non-negative")
    force_weights = _finite_array(
        contact_force_quadratic_weights,
        shape=(3 * contact_count,),
        name="contact_force_quadratic_weights",
    )
    normal_weights = _finite_array(
        normal_tracking_quadratic_weights,
        shape=(contact_count,),
        name="normal_tracking_quadratic_weights",
    )
    if np.any(force_weights < 0.0) or np.any(normal_weights < 0.0):
        raise ValueError("quadratic weights must be non-negative")
    scales = _finite_array(wrench_scaling, shape=(6,), name="wrench_scaling")
    if np.any(scales <= 0.0):
        raise ValueError("wrench_scaling must be strictly positive")

    force_matrix = model.contact_force_matrix
    normal_matrix = model.normal_force_matrix
    model_inequality_matrix = np.asarray(
        model.ray_constraint_matrix, dtype=np.float64
    )
    model_inequality_bounds = np.asarray(
        model.ray_constraint_upper_bounds, dtype=np.float64
    )
    if (
        model_inequality_matrix.ndim != 2
        or model_inequality_matrix.shape[1] != ray_count
        or model_inequality_matrix.shape[0] < contact_count
        or model_inequality_bounds.shape
        != (model_inequality_matrix.shape[0],)
    ):
        raise ValueError("contact model inequality arrays have incompatible shapes")

    physical_inequality_rows = [model_inequality_matrix]
    physical_inequality_bounds = [model_inequality_bounds]
    additional_model_row_count = (
        model_inequality_matrix.shape[0] - contact_count
    )
    inequality_row_labels = [
        *(f"normal_cap[{index}]" for index in range(contact_count)),
        *(
            f"model_additional[{index}]"
            for index in range(additional_model_row_count)
        ),
    ]
    joint_ray_matrix: np.ndarray | None = None
    joint_count = 0
    if joint_torque_constraint is not None:
        torque_map = joint_torque_constraint.contact_force_to_joint_torque
        if torque_map.shape[1] != 3 * contact_count:
            raise ValueError(
                "joint torque map column count must equal 3 * contact_count"
            )
        joint_count = int(torque_map.shape[0])
        joint_ray_matrix = torque_map @ force_matrix
        physical_inequality_rows.extend((joint_ray_matrix, -joint_ray_matrix))
        physical_inequality_bounds.extend(
            (
                joint_torque_constraint.upper_torque
                - joint_torque_constraint.bias_torque,
                -joint_torque_constraint.lower_torque
                + joint_torque_constraint.bias_torque,
            )
        )
        inequality_row_labels.extend(
            f"joint_upper[{index}]" for index in range(joint_count)
        )
        inequality_row_labels.extend(
            f"joint_lower[{index}]" for index in range(joint_count)
        )

    # Bounds still declare non-negativity to both solvers, while the explicit
    # rows make every numerical inequality residual auditable in one scaled
    # coordinate system.
    physical_inequality_rows.append(-np.eye(ray_count, dtype=np.float64))
    physical_inequality_bounds.append(
        np.zeros(ray_count, dtype=np.float64)
    )
    inequality_row_labels.extend(
        f"ray_nonnegative[{index}]" for index in range(ray_count)
    )
    physical_inequality_matrix = np.vstack(physical_inequality_rows)
    physical_inequality_target = np.concatenate(physical_inequality_bounds)
    inequality_row_norms = _augmented_row_infinity_norms(
        physical_inequality_matrix, physical_inequality_target
    )
    solver_inequality_matrix = (
        physical_inequality_matrix / inequality_row_norms[:, None]
    )
    solver_inequality_target = (
        physical_inequality_target / inequality_row_norms
    )

    caller_scaled_grasp = model.grasp_matrix / scales[:, None]
    caller_scaled_desired = desired / scales
    equality_row_norms = _augmented_row_infinity_norms(
        caller_scaled_grasp, caller_scaled_desired
    )
    solver_equality_matrix = (
        caller_scaled_grasp / equality_row_norms[:, None]
    )
    solver_equality_target = caller_scaled_desired / equality_row_norms
    left, singular_values, right_t = np.linalg.svd(
        solver_equality_matrix, full_matrices=False
    )
    equality_rank = int(
        np.sum(
            singular_values > solver_options.linear_independence_tolerance
        )
    )
    if equality_rank == 0:
        if (
            float(np.max(np.abs(solver_equality_target)))
            > solver_options.equality_tolerance
        ):
            return _failure_result(
                status=2,
                message="desired wrench is outside a zero-rank grasp map",
                options=solver_options,
                equality_rank=equality_rank,
                desired=desired,
                wrench_scaling=scales,
                equality_row_norms=equality_row_norms,
                inequality_row_norms=inequality_row_norms,
                inequality_row_labels=tuple(inequality_row_labels),
                truth_firewall_audit=truth_firewall_audit,
            )
        reduced_equality = np.zeros((0, ray_count), dtype=np.float64)
        reduced_target = np.zeros(0, dtype=np.float64)
        particular = np.zeros(ray_count, dtype=np.float64)
    else:
        particular = right_t[:equality_rank].T @ (
            (left[:, :equality_rank].T @ solver_equality_target)
            / singular_values[:equality_rank]
        )
        range_residual = (
            solver_equality_matrix @ particular - solver_equality_target
        )
        if (
            float(np.max(np.abs(range_residual)))
            > solver_options.equality_tolerance
        ):
            return _failure_result(
                status=2,
                message="desired wrench lies outside the grasp-map range",
                options=solver_options,
                equality_rank=equality_rank,
                desired=desired,
                wrench_scaling=scales,
                equality_row_norms=equality_row_norms,
                inequality_row_norms=inequality_row_norms,
                inequality_row_labels=tuple(inequality_row_labels),
                truth_firewall_audit=truth_firewall_audit,
            )
        _, _, pivot_rows = qr(
            solver_equality_matrix.T, mode="economic", pivoting=True
        )
        independent_rows = np.asarray(
            pivot_rows[:equality_rank], dtype=np.int64
        )
        reduced_equality = solver_equality_matrix[independent_rows]
        reduced_target = solver_equality_target[independent_rows]

    feasibility = linprog(
        np.zeros(ray_count, dtype=np.float64),
        A_ub=solver_inequality_matrix,
        b_ub=solver_inequality_target,
        A_eq=(reduced_equality if equality_rank else None),
        b_eq=(reduced_target if equality_rank else None),
        bounds=[(0.0, None)] * ray_count,
        method="highs",
        options={
            "maxiter": solver_options.maximum_iterations,
            "primal_feasibility_tolerance": min(
                solver_options.equality_tolerance,
                solver_options.inequality_tolerance,
            ),
            "dual_feasibility_tolerance": (
                solver_options.feasibility_dual_tolerance
            ),
        },
    )
    if not feasibility.success:
        return _failure_result(
            status=int(feasibility.status),
            message=f"feasibility LP failed: {feasibility.message}",
            options=solver_options,
            equality_rank=equality_rank,
            desired=desired,
            wrench_scaling=scales,
            equality_row_norms=equality_row_norms,
            inequality_row_norms=inequality_row_norms,
            inequality_row_labels=tuple(inequality_row_labels),
            truth_firewall_audit=truth_firewall_audit,
        )

    def objective(coefficients: np.ndarray) -> float:
        contact_forces = force_matrix @ coefficients
        normal_error = normal_matrix @ coefficients - preferred
        return 0.5 * float(
            np.sum(force_weights * contact_forces * contact_forces)
            + np.sum(normal_weights * normal_error * normal_error)
            + solver_options.regularization * (coefficients @ coefficients)
        )

    def objective_gradient(coefficients: np.ndarray) -> np.ndarray:
        contact_forces = force_matrix @ coefficients
        normal_error = normal_matrix @ coefficients - preferred
        return (
            force_matrix.T @ (force_weights * contact_forces)
            + normal_matrix.T @ (normal_weights * normal_error)
            + solver_options.regularization * coefficients
        )

    constraints: list[dict[str, Any]] = [
        {
            "type": "ineq",
            "fun": lambda value: solver_inequality_target
            - solver_inequality_matrix @ value,
            "jac": lambda value: -solver_inequality_matrix,
        }
    ]
    if equality_rank:
        constraints.append(
            {
                "type": "eq",
                "fun": lambda value: reduced_equality @ value
                - reduced_target,
                "jac": lambda value: reduced_equality,
            }
        )
    optimized = minimize(
        objective,
        np.asarray(feasibility.x, dtype=np.float64),
        jac=objective_gradient,
        method="SLSQP",
        bounds=[(0.0, None)] * ray_count,
        constraints=constraints,
        options={
            "maxiter": solver_options.maximum_iterations,
            "ftol": solver_options.objective_tolerance,
            "disp": False,
        },
    )
    coefficients = np.asarray(optimized.x, dtype=np.float64)
    contact_forces_flat = force_matrix @ coefficients
    normal_forces = normal_matrix @ coefficients
    achieved = model.grasp_matrix @ coefficients
    solver_equality_residuals = (
        solver_equality_matrix @ coefficients - solver_equality_target
    )
    solver_inequality_residuals = (
        solver_inequality_matrix @ coefficients - solver_inequality_target
    )
    maximum_solver_inequality_violation = float(
        np.max(solver_inequality_residuals, initial=0.0)
    )
    maximum_solver_equality_residual = float(
        np.max(np.abs(solver_equality_residuals), initial=0.0)
    )
    internal = coefficients - particular
    internal_forces = force_matrix @ internal
    physical_internal_wrench_residuals = model.grasp_matrix @ internal
    solver_internal_wrench_residuals = solver_equality_matrix @ internal
    physical_equilibrium_residuals = achieved - desired
    physical_model_inequality_residuals = (
        model_inequality_matrix @ coefficients - model_inequality_bounds
    )
    numerical_success = bool(
        optimized.success
        and maximum_solver_equality_residual
        <= solver_options.equality_tolerance
        and maximum_solver_inequality_violation
        <= solver_options.inequality_tolerance
    )
    joint_torques: np.ndarray | None = None
    if joint_torque_constraint is not None:
        joint_torques = (
            joint_torque_constraint.contact_force_to_joint_torque
            @ contact_forces_flat
            + joint_torque_constraint.bias_torque
        )
    return GraspForceAllocationResult(
        solver_success=numerical_success,
        solver_status=int(optimized.status),
        solver_message=str(optimized.message),
        solver_options=solver_options,
        constraint_scaling_implementation=(
            _FORCE_QP_CONSTRAINT_SCALING_IMPLEMENTATION
        ),
        equality_rank=equality_rank,
        wrench_scaling=_readonly(scales),
        equality_augmented_row_inf_norms_after_wrench_scaling=_readonly(
            equality_row_norms
        ),
        inequality_augmented_row_inf_norms=_readonly(inequality_row_norms),
        inequality_row_labels=tuple(inequality_row_labels),
        ray_coefficients=_readonly(coefficients),
        contact_forces=_readonly(
            contact_forces_flat.reshape(contact_count, 3)
        ),
        normal_forces=_readonly(normal_forces),
        achieved_object_wrench=_readonly(achieved),
        desired_object_wrench=_readonly(desired),
        joint_torques=(
            None if joint_torques is None else _readonly(joint_torques)
        ),
        objective_value=float(objective(coefficients)),
        solver_coordinate_equality_residuals=_readonly(
            solver_equality_residuals
        ),
        solver_coordinate_inequality_residuals=_readonly(
            solver_inequality_residuals
        ),
        maximum_solver_coordinate_equality_residual=(
            maximum_solver_equality_residual
        ),
        maximum_solver_coordinate_inequality_violation=(
            maximum_solver_inequality_violation
        ),
        physical_equilibrium_force_residual_n=_readonly(
            physical_equilibrium_residuals[:3]
        ),
        physical_equilibrium_moment_residual_nm=_readonly(
            physical_equilibrium_residuals[3:]
        ),
        physical_normal_cap_residuals_n=_readonly(
            normal_forces - model.normal_force_caps_n
        ),
        physical_additional_model_inequality_residuals_by_row=_readonly(
            physical_model_inequality_residuals[contact_count:]
        ),
        physical_joint_upper_torque_residuals_nm=_readonly(
            np.zeros(0, dtype=np.float64)
            if joint_torques is None
            else joint_torques - joint_torque_constraint.upper_torque
        ),
        physical_joint_lower_torque_residuals_nm=_readonly(
            np.zeros(0, dtype=np.float64)
            if joint_torques is None
            else joint_torque_constraint.lower_torque - joint_torques
        ),
        physical_ray_nonnegativity_residuals_n=_readonly(-coefficients),
        internal_ray_component=_readonly(internal),
        internal_contact_forces=_readonly(
            internal_forces.reshape(contact_count, 3)
        ),
        solver_coordinate_internal_wrench_residuals=_readonly(
            solver_internal_wrench_residuals
        ),
        physical_internal_force_residual_n=_readonly(
            physical_internal_wrench_residuals[:3]
        ),
        physical_internal_moment_residual_nm=_readonly(
            physical_internal_wrench_residuals[3:]
        ),
        truth_firewall_audit=truth_firewall_audit,
    )


__all__ = [
    "ForceAllocationSolverOptions",
    "GraspForceAllocationResult",
    "JointTorqueLinearConstraint",
    "OnlineControlObservation",
    "OnlineTruthFirewallAudit",
    "PassiveImpedanceAudit",
    "PassiveImpedanceNumerics",
    "PassiveImpedanceResult",
    "PassiveObjectImpedance",
    "allocate_grasp_forces",
    "object_pose_error_world",
]
