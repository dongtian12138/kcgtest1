"""Task-wrench evaluation for object-independent CARTS-Grasp candidates.

Coordinate convention:

* object_from_hand maps hand-base coordinates into the object frame;
* planned positions and outward normals are expressed in the object frame;
* contact forces are forces exerted by the hand on the object;
* a compressive force is opposite the certified path-local free-side normal;
  equivalence to a solid-outward normal requires the separate complete
  external-first-contact collision certificate;
* external gravity points along gravity_direction_object;
* positive lift_acceleration_m_s2 denotes acceleration opposite gravity.

All wrenches are taken about the object's centre of mass.  The nominal
gravity/lift wrench therefore has zero moment, while a contact force contributes
(p_contact - p_COM) cross f.  No object name, stored grasp pose, or historical
candidate lookup participates in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from kcg_connector.grasp.robust.grasp_optimizer import (
    GraspCandidate,
    ObjectSurfaceModel,
    WrenchEvaluation,
)
from kcg_connector.grasp.robust.full_hand_collision import (
    ContactRangePolicyCollisionCertificate,
    FullHandClosureCollisionState,
)
from kcg_connector.grasp.robust.hand_model import (
    HandModelError,
    ThreeFingerHandModel,
)
from kcg_connector.grasp.robust.interval_kinematics import (
    METHOD_ID as INTERVAL_KINEMATICS_METHOD_ID,
    IntervalBounds,
)
from kcg_connector.grasp.robust.object_model import ObjectGraspModel
from kcg_connector.grasp.robust.ray_closure import (
    CANDIDATE_REPRESENTATIVE_ROLE,
    MODEL_BINDING_COMPLETE_STATUS,
    REPRESENTATIVE_PROPOSAL_FAILURE_REASON,
    CertifiedContactFeatureRoot,
    CertifiedSequentialClosurePolicy,
    RayClosureAudit,
    METHOD_ID as RAY_CLOSURE_METHOD_ID,
    _canonical_json,
    _float64_array_hex,
    _hand_model_manifest,
)
from kcg_connector.grasp.robust.robust_wrench import (
    LinearProgramSolverOptions,
    TaskWrenchMarginResult,
    build_polyhedral_contact_wrench_model,
    maximum_task_wrench_polytope_margin,
)


_FLOAT_EPS = np.finfo(np.float64).eps
FRICTION_INTERVAL_ONLY_CERTIFIED_UNCERTAINTY_SCOPE = (
    "CERTIFIED_SET_CONTAINS_ONLY_DECLARED_FRICTION_INTERVAL_"
    "ALL_OTHER_UNCERTAINTIES_REQUIRE_CALIBRATION"
)
COMPLETE_CONTINUOUS_TRAJECTORY_CLEARANCE_SCOPE = (
    "COMPLETE_HAND_OBJECT_ENVIRONMENT_APPROACH_CLOSURE_LIFT_"
    "CONTINUOUS_COLLISION_CERTIFIED_LOWER_BOUND"
)
CONTACT_RANGE_POLICY_WRENCH_METHOD_ID = (
    "CARTS_CONTACT_RANGE_POLICY_WRENCH_DOMAIN_AUDIT_V1"
)
CONTACT_RANGE_POLICY_WRENCH_PRODUCT_RULE = (
    "COMPLETE_CARTESIAN_PRODUCT_OF_ALL_POSSIBLE_EARLIEST_ROOTS"
)
CONTACT_RANGE_POLICY_WRENCH_MANDATORY_BLOCKERS = (
    "INTERVAL_CONTACT_JACOBIAN_CERTIFICATE_UNAVAILABLE",
    "PARAMETRIC_CONTACT_RANGE_WRENCH_LOWER_BOUND_UNAVAILABLE",
    "CALIBRATED_NONFRICTION_UNCERTAINTY_BOUNDS_UNAVAILABLE",
)
CONTACT_RANGE_POLICY_WRENCH_CLAIM_LIMITATIONS = (
    "CONTACT_RANGE_POLICY_ROOT_DOMAIN_BINDING_ONLY",
    "ALL_POSSIBLE_EARLIEST_ROOTS_AND_CARTESIAN_PRODUCT_COUNT_BOUND",
    "NO_DISPLAY_APPROXIMATION_AS_FORMAL_EVIDENCE",
    "NO_FINITE_CONTACT_GEOMETRY_SAMPLING_AS_FORMAL_EVIDENCE",
    "NO_EXACT_FINAL_JOINT_OR_CONTACT_SUBSTITUTION",
    "NO_INTERVAL_CONTACT_JACOBIAN_CERTIFICATE",
    "NO_PARAMETRIC_CONTACT_RANGE_WRENCH_LOWER_BOUND",
    "FRICTION_INTERVAL_ONLY_OTHER_UNCERTAINTIES_UNCALIBRATED",
    "NORMAL_FORCE_CAPACITY_IS_UNCALIBRATED_OPTIMIZATION_BOUND",
    "NOT_COLLISION_OR_DYNAMIC_EVIDENCE",
)
_ALLOWED_POLICY_CONTACT_CLASSIFICATION = (
    "ALLOWED_PATH_LOCAL_FREE_SIDE_TRANSVERSE_CONTACT"
)


class TaskWrenchEvaluationError(RuntimeError):
    """Fail-closed result when a margin cannot be numerically certified."""


class ContactRangePolicyWrenchState(str, Enum):
    """The V1 range consumer cannot yet certify a wrench margin."""

    NOT_CERTIFIABLE = "NOT_CERTIFIABLE"


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=np.float64)
    result.setflags(write=False)
    return result


def _deep_freeze(value: Any) -> Any:
    """Return an immutable diagnostic snapshot with no writable arrays."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(child) for key, child in value.items()}
        )
    if isinstance(value, np.ndarray):
        return _deep_freeze(value.tolist())
    if isinstance(value, (tuple, list)):
        return tuple(_deep_freeze(child) for child in value)
    return value


def _finite_vector(
    value: Sequence[float], *, shape: tuple[int, ...], label: str
) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must have finite shape {shape}")
    return result


def _unit_vector(value: Sequence[float], *, label: str) -> np.ndarray:
    result = _finite_vector(value, shape=(3,), label=label)
    norm = float(np.linalg.norm(result))
    if norm <= _FLOAT_EPS:
        raise ValueError(f"{label} is numerically zero")
    return result / norm


def _proper_rotation(
    value: Sequence[Sequence[float]], *, label: str
) -> np.ndarray:
    rotation = _finite_vector(value, shape=(3, 3), label=label)
    # Roundoff check, not a physical gate.  The multiplier bounds the floating
    # operations in two 3-by-3 products and a determinant.
    roundoff = 64.0 * _FLOAT_EPS
    orthogonality_error = float(
        np.linalg.norm(rotation.T @ rotation - np.eye(3), ord=np.inf)
    )
    determinant = float(np.linalg.det(rotation))
    if (
        orthogonality_error > roundoff
        or determinant <= 0.0
        or abs(determinant - 1.0) > roundoff
    ):
        raise ValueError(f"{label} must be a proper orthonormal rotation")
    return rotation


def _rigid_transform(value: Sequence[float], *, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64).reshape(4, 4)
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} must be finite")
    roundoff = 64.0 * _FLOAT_EPS
    if float(np.linalg.norm(matrix[3] - (0.0, 0.0, 0.0, 1.0))) > roundoff:
        raise ValueError(f"{label} must have a homogeneous final row")
    _proper_rotation(matrix[:3, :3], label=f"{label} rotation")
    return matrix


@dataclass(frozen=True)
class ContactActuationModel:
    """Linear contact-to-independent-joint map in the object frame."""

    torque_from_object_contact_forces: np.ndarray
    independent_joint_effort_limits: np.ndarray
    contact_points_link_m: tuple[tuple[float, float, float], ...]

    def __post_init__(self) -> None:
        torque_map = np.asarray(
            self.torque_from_object_contact_forces, dtype=np.float64
        )
        effort = np.asarray(self.independent_joint_effort_limits, dtype=np.float64)
        if (
            torque_map.ndim != 2
            or torque_map.shape[0] != effort.size
            or torque_map.shape[1] != 3 * len(self.contact_points_link_m)
            or not np.all(np.isfinite(torque_map))
            or effort.shape != (torque_map.shape[0],)
            or not np.all(np.isfinite(effort))
            or np.any(effort <= 0.0)
        ):
            raise ValueError("contact actuation model has inconsistent finite dimensions")
        object.__setattr__(
            self, "torque_from_object_contact_forces", _readonly(torque_map)
        )
        object.__setattr__(
            self, "independent_joint_effort_limits", _readonly(effort)
        )


@dataclass(frozen=True)
class TaskWrenchDefinition:
    """The preregistered nominal wrench and 12-vertex disturbance body."""

    nominal_external_wrench: np.ndarray
    disturbance_vertices: np.ndarray
    force_scale_n: float
    moment_scale_nm: float
    wrench_origin_object_m: np.ndarray

    def __post_init__(self) -> None:
        nominal = _finite_vector(
            self.nominal_external_wrench,
            shape=(6,),
            label="nominal_external_wrench",
        )
        vertices = _finite_vector(
            self.disturbance_vertices,
            shape=(12, 6),
            label="disturbance_vertices",
        )
        origin = _finite_vector(
            self.wrench_origin_object_m,
            shape=(3,),
            label="wrench_origin_object_m",
        )
        if (
            not math.isfinite(self.force_scale_n)
            or self.force_scale_n <= 0.0
            or not math.isfinite(self.moment_scale_nm)
            or self.moment_scale_nm <= 0.0
        ):
            raise ValueError("task wrench scales must be finite and positive")
        object.__setattr__(self, "nominal_external_wrench", _readonly(nominal))
        object.__setattr__(self, "disturbance_vertices", _readonly(vertices))
        object.__setattr__(self, "wrench_origin_object_m", _readonly(origin))


@dataclass(frozen=True)
class TaskWrenchOnlyEvaluation:
    """Collision-independent task-wrench result for one common QMC design.

    This type deliberately has no trajectory-clearance field.  It allows every
    unique static V9 candidate to receive exactly one wrench evaluation even
    while the independent complete-trajectory collision certificate remains
    unresolved.  Consequently it is diagnostic evidence, not by itself a
    formally selectable grasp.
    """

    task_margins: tuple[float, ...]
    hard_bound_minimum_task_margin: float
    peak_normal_force_n: float
    joint_torque_utilization: float
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        margins = tuple(float(value) for value in self.task_margins)
        scalars = (
            float(self.hard_bound_minimum_task_margin),
            float(self.peak_normal_force_n),
            float(self.joint_torque_utilization),
        )
        if not margins or not all(
            math.isfinite(value) for value in margins + scalars
        ):
            raise ValueError(
                "task-wrench-only evaluation must contain finite margins"
            )
        if scalars[0] < 0.0:
            raise ValueError("hard-bound task margin cannot be negative")
        if scalars[1] < 0.0 or scalars[2] < 0.0:
            raise ValueError(
                "task-wrench-only force and torque burdens cannot be negative"
            )
        if not isinstance(self.diagnostics, Mapping):
            raise ValueError("task-wrench-only diagnostics must be a mapping")
        object.__setattr__(self, "task_margins", margins)
        object.__setattr__(
            self, "hard_bound_minimum_task_margin", scalars[0]
        )
        object.__setattr__(self, "peak_normal_force_n", scalars[1])
        object.__setattr__(self, "joint_torque_utilization", scalars[2])
        object.__setattr__(self, "diagnostics", _deep_freeze(self.diagnostics))


@dataclass(frozen=True)
class ContactRangeRootWrenchDomain:
    """One possible exact contact root, represented without a midpoint."""

    pad_name: str
    formal_root_sha256: str
    witness_flat_index: int
    object_face_index: int
    semantic_classification: str
    phase: IntervalBounds
    position_object_m: tuple[IntervalBounds, IntervalBounds, IntervalBounds]
    object_source_winding_free_side_sign: int

    def __post_init__(self) -> None:
        positions = tuple(self.position_object_m)
        if (
            not self.pad_name
            or len(self.formal_root_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.formal_root_sha256
            )
            or not isinstance(self.witness_flat_index, int)
            or isinstance(self.witness_flat_index, bool)
            or self.witness_flat_index < 0
            or not isinstance(self.object_face_index, int)
            or isinstance(self.object_face_index, bool)
            or self.object_face_index < 0
            or self.semantic_classification
            != _ALLOWED_POLICY_CONTACT_CLASSIFICATION
            or not isinstance(self.phase, IntervalBounds)
            or len(positions) != 3
            or not all(isinstance(row, IntervalBounds) for row in positions)
            or self.object_source_winding_free_side_sign not in (-1, 1)
        ):
            raise TaskWrenchEvaluationError(
                "contact-range root wrench domain is malformed"
            )
        object.__setattr__(self, "position_object_m", positions)

    def as_dict(self) -> dict[str, object]:
        return {
            "pad_name": self.pad_name,
            "formal_root_sha256": self.formal_root_sha256,
            "witness_flat_index": self.witness_flat_index,
            "object_face_index": self.object_face_index,
            "semantic_classification": self.semantic_classification,
            "phase": self.phase.as_dict(),
            "position_object_m": [
                row.as_dict() for row in self.position_object_m
            ],
            "object_source_winding_free_side_sign": (
                self.object_source_winding_free_side_sign
            ),
        }


@dataclass(frozen=True)
class ContactRangePadWrenchDomain:
    """Every possible earliest root for one PAD in canonical order."""

    pad_name: str
    roots: tuple[ContactRangeRootWrenchDomain, ...]

    def __post_init__(self) -> None:
        roots = tuple(self.roots)
        root_ids = tuple(row.formal_root_sha256 for row in roots)
        if (
            not self.pad_name
            or not roots
            or any(row.pad_name != self.pad_name for row in roots)
            or root_ids != tuple(sorted(root_ids))
            or len(set(root_ids)) != len(root_ids)
        ):
            raise TaskWrenchEvaluationError(
                "contact-range PAD wrench domain is not canonical"
            )
        object.__setattr__(self, "roots", roots)

    @property
    def domain_sha256(self) -> str:
        return hashlib.sha256(
            _canonical_json(self._evidence_dict()).encode("utf-8")
        ).hexdigest()

    def _evidence_dict(self) -> dict[str, object]:
        return {
            "pad_name": self.pad_name,
            "roots": [row.as_dict() for row in self.roots],
        }

    def as_dict(self) -> dict[str, object]:
        result = self._evidence_dict()
        result["domain_sha256"] = self.domain_sha256
        return result


@dataclass(frozen=True)
class ContactRangePolicyWrenchAudit:
    """Immutable proof of domain consumption, not of wrench capability."""

    method_id: str
    ray_closure_method_id: str
    interval_kinematics_method_id: str
    policy_sha256: str
    v9_model_and_policy_sha256: str
    policy_collision_binding_sha256: str
    task_wrench_contract_sha256: str
    scenario_design_sha256: str
    friction_scenario_sha256: str
    object_geometry_sha256: str
    model_contract_sha256: str
    pad_order: tuple[str, str, str]
    pad_domains: tuple[ContactRangePadWrenchDomain, ...]
    possible_root_counts: tuple[int, int, int]
    total_possible_root_count: int
    cartesian_product_count: int
    cartesian_product_rule: str
    scenario_count: int
    scenario_dimension: int
    hard_bound_friction_coefficient: float
    uncertainty_claim_scope: str
    policy_contact_root_domains_consumed: bool
    complete_cartesian_product_bound: bool
    display_approximation_used_as_formal_evidence: bool
    finite_contact_geometry_sampling_used_as_formal_evidence: bool
    exact_candidate_wrench_invocation_count: int
    contact_range_margin_computed: bool
    interval_contact_jacobian_certificate_present: bool
    parametric_wrench_lower_bound_certificate_present: bool
    formal_selection_allowed: bool
    blockers: tuple[str, ...]
    claim_limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        digests = (
            self.policy_sha256,
            self.v9_model_and_policy_sha256,
            self.policy_collision_binding_sha256,
            self.task_wrench_contract_sha256,
            self.scenario_design_sha256,
            self.friction_scenario_sha256,
            self.object_geometry_sha256,
            self.model_contract_sha256,
        )
        domains = tuple(self.pad_domains)
        pad_order = tuple(self.pad_order)
        counts = tuple(self.possible_root_counts)
        if (
            self.method_id != CONTACT_RANGE_POLICY_WRENCH_METHOD_ID
            or self.ray_closure_method_id != RAY_CLOSURE_METHOD_ID
            or self.interval_kinematics_method_id
            != INTERVAL_KINEMATICS_METHOD_ID
            or any(
                len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in digests
            )
            or len(pad_order) != 3
            or len(set(pad_order)) != 3
            or tuple(row.pad_name for row in domains) != pad_order
            or counts != tuple(len(row.roots) for row in domains)
            or any(value <= 0 for value in counts)
            or self.total_possible_root_count != sum(counts)
            or self.cartesian_product_count != math.prod(counts)
            or self.cartesian_product_rule
            != CONTACT_RANGE_POLICY_WRENCH_PRODUCT_RULE
            or self.scenario_count <= 0
            or self.scenario_dimension != 1
            or not math.isfinite(self.hard_bound_friction_coefficient)
            or self.hard_bound_friction_coefficient < 0.0
            or self.uncertainty_claim_scope
            != FRICTION_INTERVAL_ONLY_CERTIFIED_UNCERTAINTY_SCOPE
        ):
            raise TaskWrenchEvaluationError(
                "contact-range policy wrench audit binding is malformed"
            )
        if (
            self.policy_contact_root_domains_consumed is not True
            or self.complete_cartesian_product_bound is not True
            or self.display_approximation_used_as_formal_evidence
            or self.finite_contact_geometry_sampling_used_as_formal_evidence
            or self.exact_candidate_wrench_invocation_count != 0
            or self.contact_range_margin_computed
            or self.interval_contact_jacobian_certificate_present
            or self.parametric_wrench_lower_bound_certificate_present
            or self.formal_selection_allowed
            or self.blockers != CONTACT_RANGE_POLICY_WRENCH_MANDATORY_BLOCKERS
            or self.claim_limitations
            != CONTACT_RANGE_POLICY_WRENCH_CLAIM_LIMITATIONS
        ):
            raise TaskWrenchEvaluationError(
                "contact-range policy wrench audit overclaims its evidence"
            )
        object.__setattr__(self, "pad_order", pad_order)
        object.__setattr__(self, "pad_domains", domains)
        object.__setattr__(self, "possible_root_counts", counts)

    @property
    def audit_sha256(self) -> str:
        return hashlib.sha256(
            _canonical_json(self._evidence_dict()).encode("utf-8")
        ).hexdigest()

    def _evidence_dict(self) -> dict[str, object]:
        return {
            "method_id": self.method_id,
            "ray_closure_method_id": self.ray_closure_method_id,
            "interval_kinematics_method_id": (
                self.interval_kinematics_method_id
            ),
            "policy_sha256": self.policy_sha256,
            "v9_model_and_policy_sha256": self.v9_model_and_policy_sha256,
            "policy_collision_binding_sha256": (
                self.policy_collision_binding_sha256
            ),
            "task_wrench_contract_sha256": self.task_wrench_contract_sha256,
            "scenario_design_sha256": self.scenario_design_sha256,
            "friction_scenario_sha256": self.friction_scenario_sha256,
            "object_geometry_sha256": self.object_geometry_sha256,
            "model_contract_sha256": self.model_contract_sha256,
            "pad_order": list(self.pad_order),
            "pad_domains": [row.as_dict() for row in self.pad_domains],
            "possible_root_counts": list(self.possible_root_counts),
            "total_possible_root_count": self.total_possible_root_count,
            "cartesian_product_count": self.cartesian_product_count,
            "cartesian_product_rule": self.cartesian_product_rule,
            "scenario_count": self.scenario_count,
            "scenario_dimension": self.scenario_dimension,
            "hard_bound_friction_coefficient_binary64_hex": float(
                self.hard_bound_friction_coefficient
            ).hex(),
            "uncertainty_claim_scope": self.uncertainty_claim_scope,
            "policy_contact_root_domains_consumed": (
                self.policy_contact_root_domains_consumed
            ),
            "complete_cartesian_product_bound": (
                self.complete_cartesian_product_bound
            ),
            "display_approximation_used_as_formal_evidence": (
                self.display_approximation_used_as_formal_evidence
            ),
            "finite_contact_geometry_sampling_used_as_formal_evidence": (
                self.finite_contact_geometry_sampling_used_as_formal_evidence
            ),
            "exact_candidate_wrench_invocation_count": (
                self.exact_candidate_wrench_invocation_count
            ),
            "contact_range_margin_computed": self.contact_range_margin_computed,
            "interval_contact_jacobian_certificate_present": (
                self.interval_contact_jacobian_certificate_present
            ),
            "parametric_wrench_lower_bound_certificate_present": (
                self.parametric_wrench_lower_bound_certificate_present
            ),
            "formal_selection_allowed": self.formal_selection_allowed,
            "blockers": list(self.blockers),
            "claim_limitations": list(self.claim_limitations),
        }

    def as_dict(self) -> dict[str, object]:
        result = self._evidence_dict()
        result["audit_sha256"] = self.audit_sha256
        return result


@dataclass(frozen=True)
class ContactRangePolicyWrenchCertificate:
    """Fail-closed result until a certified range-wrench lower bound exists."""

    state: ContactRangePolicyWrenchState
    audit: ContactRangePolicyWrenchAudit
    task_margins: None = None
    hard_bound_minimum_task_margin: None = None
    peak_normal_force_n: None = None
    joint_torque_utilization: None = None

    def __post_init__(self) -> None:
        if (
            self.state is not ContactRangePolicyWrenchState.NOT_CERTIFIABLE
            or not isinstance(self.audit, ContactRangePolicyWrenchAudit)
            or self.task_margins is not None
            or self.hard_bound_minimum_task_margin is not None
            or self.peak_normal_force_n is not None
            or self.joint_torque_utilization is not None
        ):
            raise TaskWrenchEvaluationError(
                "contact-range policy wrench certificate overclaims a result"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "audit": self.audit.as_dict(),
            "task_margins": None,
            "hard_bound_minimum_task_margin": None,
            "peak_normal_force_n": None,
            "joint_torque_utilization": None,
        }


class TaskWrenchEvaluator:
    """Common-Sobol friction evaluation with finite PAD/actuator constraints.

    The certified set in this evaluator contains only the explicitly declared
    friction interval.  Pose, surface, mass-property, joint-tracking and
    actuator residuals require a separate calibrated contract and are not
    silently represented by zero-width intervals.
    """

    uncertainty_dimension = 1

    def __init__(
        self,
        *,
        object_model: ObjectGraspModel,
        characteristic_radius_m: float,
        friction_coefficient_interval: Sequence[float],
        uncertainty_claim_scope: str,
        gravity_direction_object: Sequence[float],
        task_frame_rotation_object: Sequence[Sequence[float]],
        gravity_acceleration_m_s2: float,
        lift_acceleration_m_s2: float,
        maximum_inner_approximation_relative_error: float,
        cone_edge_multiplier: int,
        solver_options: LinearProgramSolverOptions,
    ) -> None:
        if not isinstance(object_model, ObjectGraspModel):
            raise TypeError("object_model must be an ObjectGraspModel")
        radius = float(characteristic_radius_m)
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("characteristic_radius_m must be finite and positive")
        friction = _finite_vector(
            friction_coefficient_interval,
            shape=(2,),
            label="friction_coefficient_interval",
        )
        if friction[0] < 0.0 or friction[1] < friction[0]:
            raise ValueError("friction interval must satisfy 0 <= lower <= upper")
        if (
            uncertainty_claim_scope
            != FRICTION_INTERVAL_ONLY_CERTIFIED_UNCERTAINTY_SCOPE
        ):
            raise ValueError(
                "uncertainty_claim_scope must explicitly limit certification "
                "to the declared friction interval"
            )
        gravity_acceleration = float(gravity_acceleration_m_s2)
        lift_acceleration = float(lift_acceleration_m_s2)
        if not math.isfinite(gravity_acceleration) or gravity_acceleration <= 0.0:
            raise ValueError("gravity_acceleration_m_s2 must be finite and positive")
        if not math.isfinite(lift_acceleration):
            raise ValueError("lift_acceleration_m_s2 must be finite")
        cone_error = float(maximum_inner_approximation_relative_error)
        if not math.isfinite(cone_error) or not 0.0 < cone_error < 1.0:
            raise ValueError(
                "maximum_inner_approximation_relative_error must lie in (0, 1)"
            )
        if (
            isinstance(cone_edge_multiplier, bool)
            or int(cone_edge_multiplier) != cone_edge_multiplier
            or cone_edge_multiplier < 1
        ):
            raise ValueError("cone_edge_multiplier must be a positive integer")
        if not isinstance(solver_options, LinearProgramSolverOptions):
            raise TypeError("solver_options must be LinearProgramSolverOptions")

        self.object_model = object_model
        self.characteristic_radius_m = radius
        self.friction_coefficient_interval = tuple(float(item) for item in friction)
        self.uncertainty_claim_scope = uncertainty_claim_scope
        self.gravity_direction_object = _readonly(
            _unit_vector(
                gravity_direction_object, label="gravity_direction_object"
            )
        )
        self.task_frame_rotation_object = _readonly(
            _proper_rotation(
                task_frame_rotation_object,
                label="task_frame_rotation_object",
            )
        )
        self.gravity_acceleration_m_s2 = gravity_acceleration
        self.lift_acceleration_m_s2 = lift_acceleration
        self.maximum_inner_approximation_relative_error = cone_error
        self.cone_edge_multiplier = int(cone_edge_multiplier)
        self.solver_options = solver_options
        self.task_wrench_definition = self._build_task_wrench_definition()

    @staticmethod
    def _policy_model_and_path_sha256(
        policy: CertifiedSequentialClosurePolicy,
        audit: RayClosureAudit,
    ) -> str:
        payload = {
            "policy_sha256": policy.policy_sha256,
            "ray_closure_method_id": audit.method_id,
            "model_binding_status": audit.model_binding_status,
            "object_geometry_sha256": audit.object_geometry_sha256,
            "model_contract_sha256": audit.model_contract_sha256,
            "pad_order": list(audit.pad_order),
            "pad_geometry_sha256": list(audit.pad_geometry_sha256),
            "pad_runtime_geometry_sha256": list(
                audit.pad_runtime_geometry_sha256
            ),
            "pad_link_names": list(audit.pad_link_names),
            "closing_directions_physical": _float64_array_hex(
                audit.closing_directions_physical
            ),
            "candidate_role": audit.candidate_role,
            "candidate_exact_contact_endpoint_certified": (
                audit.candidate_exact_contact_endpoint_certified
            ),
        }
        return hashlib.sha256(
            _canonical_json(payload).encode("utf-8")
        ).hexdigest()

    def _task_wrench_contract_sha256(
        self,
        *,
        hand_model: ThreeFingerHandModel,
        model_contract_sha256: str,
    ) -> str:
        definition = self.task_wrench_definition
        options = self.solver_options
        payload = {
            "method_id": CONTACT_RANGE_POLICY_WRENCH_METHOD_ID,
            "model_contract_sha256": model_contract_sha256,
            "object": {
                "geometry_sha256": self.object_model.geometry_sha256,
                "mass_kg": float(self.object_model.mass_kg).hex(),
                "center_of_mass_m": _float64_array_hex(
                    self.object_model.center_of_mass_m
                ),
                "inertia_kg_m2": _float64_array_hex(
                    self.object_model.inertia_kg_m2
                ),
            },
            "hand": _hand_model_manifest(hand_model),
            "task_wrench": {
                "nominal_external_wrench": _float64_array_hex(
                    definition.nominal_external_wrench
                ),
                "disturbance_vertices": _float64_array_hex(
                    definition.disturbance_vertices
                ),
                "force_scale_n": float(definition.force_scale_n).hex(),
                "moment_scale_nm": float(definition.moment_scale_nm).hex(),
                "wrench_origin_object_m": _float64_array_hex(
                    definition.wrench_origin_object_m
                ),
                "characteristic_radius_m": float(
                    self.characteristic_radius_m
                ).hex(),
                "gravity_direction_object": _float64_array_hex(
                    self.gravity_direction_object
                ),
                "task_frame_rotation_object": _float64_array_hex(
                    self.task_frame_rotation_object
                ),
                "gravity_acceleration_m_s2": float(
                    self.gravity_acceleration_m_s2
                ).hex(),
                "lift_acceleration_m_s2": float(
                    self.lift_acceleration_m_s2
                ).hex(),
            },
            "friction": {
                "coefficient_interval": _float64_array_hex(
                    self.friction_coefficient_interval
                ),
                "uncertainty_claim_scope": self.uncertainty_claim_scope,
                "maximum_inner_approximation_relative_error": float(
                    self.maximum_inner_approximation_relative_error
                ).hex(),
                "cone_edge_multiplier": self.cone_edge_multiplier,
            },
            "linear_program": {
                "solver": options.solver,
                "constraint_scaling": options.requested_constraint_scaling,
                "maximum_iterations": options.maximum_iterations,
                "primal_feasibility_tolerance": float(
                    options.primal_feasibility_tolerance
                ).hex(),
                "dual_feasibility_tolerance": float(
                    options.dual_feasibility_tolerance
                ).hex(),
                "ipm_optimality_tolerance": float(
                    options.ipm_optimality_tolerance
                ).hex(),
                "physical_acceptance_gate": options.physical_acceptance_gate,
            },
        }
        return hashlib.sha256(
            _canonical_json(payload).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _policy_collision_binding_sha256(
        certificate: ContactRangePolicyCollisionCertificate,
    ) -> str:
        audit = certificate.audit
        payload = {
            "state": certificate.state.value,
            "method_id": audit.method_id,
            "policy_sha256": audit.policy_sha256,
            "v9_audit_and_policy_sha256": (
                audit.v9_audit_and_policy_sha256
            ),
            "object_source_asset_sha256": (
                audit.object_source_asset_sha256
            ),
            "object_surface_geometry_sha256": (
                audit.object_surface_geometry_sha256
            ),
            "ray_closure_object_geometry_sha256": (
                audit.ray_closure_object_geometry_sha256
            ),
            "ray_model_contract_sha256": audit.ray_model_contract_sha256,
            "link_surface_bindings": [
                list(row) for row in audit.link_surface_bindings
            ],
            "terminal_partition_bindings": [
                list(row) for row in audit.terminal_partition_bindings
            ],
            "self_pair_inventory_sha256": audit.self_pair_inventory_sha256,
            "support_phase_upper_bounds": _float64_array_hex(
                audit.support_phase_upper_bounds
            ),
            "checkable_collision_gates_passed": (
                audit.checkable_collision_gates_passed
            ),
            "blockers": list(audit.blockers),
            "claim_limitations": list(audit.claim_limitations),
        }
        return hashlib.sha256(
            _canonical_json(payload).encode("utf-8")
        ).hexdigest()

    def _validate_contact_range_policy_binding(
        self,
        *,
        policy: CertifiedSequentialClosurePolicy,
        audit: RayClosureAudit,
        hand_model: ThreeFingerHandModel,
        policy_collision_certificate: ContactRangePolicyCollisionCertificate,
    ) -> None:
        if not isinstance(policy, CertifiedSequentialClosurePolicy) or not isinstance(
            audit, RayClosureAudit
        ):
            raise TaskWrenchEvaluationError(
                "contact-range wrench needs a certified policy and V9 audit"
            )
        if not isinstance(hand_model, ThreeFingerHandModel):
            raise TaskWrenchEvaluationError(
                "contact-range wrench needs ThreeFingerHandModel"
            )
        if not isinstance(
            policy_collision_certificate,
            ContactRangePolicyCollisionCertificate,
        ):
            raise TaskWrenchEvaluationError(
                "contact-range wrench needs the policy collision binding"
            )
        if (
            audit.method_id != RAY_CLOSURE_METHOD_ID
            or audit.model_binding_status != MODEL_BINDING_COMPLETE_STATUS
            or not audit.model_binding_complete
            or audit.failure_reason != REPRESENTATIVE_PROPOSAL_FAILURE_REASON
            or audit.candidate_role != CANDIDATE_REPRESENTATIVE_ROLE
            or audit.candidate_exact_contact_endpoint_certified
            or not audit.full_verified_pad_mesh_used
            or audit.pad_face_subset_input_allowed
            or audit.subdivision_budget_exhausted
        ):
            raise TaskWrenchEvaluationError(
                "contact-range wrench needs the registered V9 policy outcome"
            )
        if (
            policy.independent_joint_names
            != tuple(hand_model.independent_joint_names)
            or policy.pad_order != audit.pad_order
            or policy.independent_actuation_supports
            != audit.independent_actuation_supports
            or policy.closing_directions_physical
            != audit.closing_directions_physical
            or policy.object_geometry_sha256 != audit.object_geometry_sha256
            or policy.model_contract_sha256 != audit.model_contract_sha256
            or audit.object_geometry_sha256
            != self.object_model.geometry_sha256
            or len(audit.possible_first_contact_set_sha256) != 3
        ):
            raise TaskWrenchEvaluationError(
                "contact-range policy differs from its wrench model binding"
            )
        collision_audit = policy_collision_certificate.audit
        expected_policy_binding = self._policy_model_and_path_sha256(
            policy, audit
        )
        if (
            policy_collision_certificate.state
            is not FullHandClosureCollisionState.NOT_CERTIFIABLE
            or not collision_audit.checkable_collision_gates_passed
            or not collision_audit.policy_contact_ranges_consumed
            or collision_audit.display_approximation_used_as_formal_evidence
            or collision_audit.policy_sha256 != policy.policy_sha256
            or collision_audit.v9_audit_and_policy_sha256
            != expected_policy_binding
            or collision_audit.ray_closure_object_geometry_sha256
            != audit.object_geometry_sha256
            or collision_audit.ray_model_contract_sha256
            != audit.model_contract_sha256
        ):
            raise TaskWrenchEvaluationError(
                "contact-range wrench differs from its policy collision binding"
            )
        try:
            document = json.loads(audit.model_contract_canonical_json)
        except (TypeError, ValueError) as error:
            raise TaskWrenchEvaluationError(
                "contact-range wrench model contract cannot be decoded"
            ) from error
        if (
            not isinstance(document, Mapping)
            or _canonical_json(document) != audit.model_contract_canonical_json
            or hashlib.sha256(
                audit.model_contract_canonical_json.encode("utf-8")
            ).hexdigest()
            != audit.model_contract_sha256
        ):
            raise TaskWrenchEvaluationError(
                "contact-range wrench model contract digest is not canonical"
            )
        try:
            document_object_hash = document["object"]["geometry_sha256"]
            document_hand = document["hand"]
        except (KeyError, TypeError) as error:
            raise TaskWrenchEvaluationError(
                "contact-range wrench model contract lacks object/hand binding"
            ) from error
        if (
            document_object_hash != self.object_model.geometry_sha256
            or document_hand != _hand_model_manifest(hand_model)
        ):
            raise TaskWrenchEvaluationError(
                "contact-range wrench object or hand differs from V9"
            )
        try:
            hand_model.resolve_joint_positions(
                policy.initial_independent_joint_positions_rad
            )
        except HandModelError as error:
            raise TaskWrenchEvaluationError(
                "contact-range wrench policy initial joints violate the hand"
            ) from error
        if (
            set(policy.pad_order) != set(hand_model.pads)
            or any(
                hand_model.pads[name].normal_force_capacity_n is None
                for name in policy.pad_order
            )
            or any(
                hand_model.independent_joint_limits[name].effort is None
                for name in hand_model.independent_joint_names
            )
        ):
            raise TaskWrenchEvaluationError(
                "contact-range wrench hand lacks PAD force or joint effort bounds"
            )
        for pad_name, contact_set in zip(
            policy.pad_order, policy.possible_first_contact_sets
        ):
            roots = contact_set.possible_earliest_roots
            if (
                contact_set.pad_name != pad_name
                or not roots
                or any(
                    not isinstance(root, CertifiedContactFeatureRoot)
                    or root.pad_name != pad_name
                    or root.semantic_classification
                    != _ALLOWED_POLICY_CONTACT_CLASSIFICATION
                    or root.certificate.method_id
                    != INTERVAL_KINEMATICS_METHOD_ID
                    for root in roots
                )
            ):
                raise TaskWrenchEvaluationError(
                    "contact-range wrench root domain is not a valid earliest set"
                )

    @staticmethod
    def _root_wrench_domain(
        root: CertifiedContactFeatureRoot,
    ) -> ContactRangeRootWrenchDomain:
        return ContactRangeRootWrenchDomain(
            pad_name=root.pad_name,
            formal_root_sha256=(
                CertifiedSequentialClosurePolicy._formal_root_sha256(root)
            ),
            witness_flat_index=root.witness_flat_index,
            object_face_index=root.object_face_index,
            semantic_classification=root.semantic_classification,
            phase=root.certificate.phase,
            position_object_m=root.certificate.position_object_m,
            object_source_winding_free_side_sign=(
                root.certificate.object_source_winding_free_side_sign
            ),
        )

    def evaluate_contact_range_policy(
        self,
        policy: CertifiedSequentialClosurePolicy,
        scenario_parameters_unit: np.ndarray,
        *,
        v9_audit: RayClosureAudit,
        hand_model: ThreeFingerHandModel,
        policy_collision_certificate: ContactRangePolicyCollisionCertificate,
    ) -> ContactRangePolicyWrenchCertificate:
        """Bind the full policy domain and fail closed before a range margin.

        QMC points still describe only the registered friction interval.  They
        never sample contact geometry.  Every possible earliest root remains
        in a per-PAD domain and the complete three-PAD Cartesian-product count
        is recorded without constructing a display-only exact candidate.
        """

        self._validate_contact_range_policy_binding(
            policy=policy,
            audit=v9_audit,
            hand_model=hand_model,
            policy_collision_certificate=policy_collision_certificate,
        )
        scenarios = np.asarray(scenario_parameters_unit, dtype=np.float64)
        friction_values = self.friction_coefficients_from_unit(scenarios)
        pad_domains = tuple(
            ContactRangePadWrenchDomain(
                pad_name=contact_set.pad_name,
                roots=tuple(
                    sorted(
                        (
                            self._root_wrench_domain(root)
                            for root in contact_set.possible_earliest_roots
                        ),
                        key=lambda row: row.formal_root_sha256,
                    )
                ),
            )
            for contact_set in policy.possible_first_contact_sets
        )
        counts = tuple(len(row.roots) for row in pad_domains)
        scenario_design_sha256 = hashlib.sha256(
            np.ascontiguousarray(scenarios, dtype=">f8").tobytes(order="C")
        ).hexdigest()
        friction_scenario_sha256 = hashlib.sha256(
            np.ascontiguousarray(friction_values, dtype=">f8").tobytes(
                order="C"
            )
        ).hexdigest()
        policy_and_model_sha256 = self._policy_model_and_path_sha256(
            policy, v9_audit
        )
        task_contract_sha256 = self._task_wrench_contract_sha256(
            hand_model=hand_model,
            model_contract_sha256=v9_audit.model_contract_sha256,
        )
        audit = ContactRangePolicyWrenchAudit(
            method_id=CONTACT_RANGE_POLICY_WRENCH_METHOD_ID,
            ray_closure_method_id=RAY_CLOSURE_METHOD_ID,
            interval_kinematics_method_id=INTERVAL_KINEMATICS_METHOD_ID,
            policy_sha256=policy.policy_sha256,
            v9_model_and_policy_sha256=policy_and_model_sha256,
            policy_collision_binding_sha256=(
                self._policy_collision_binding_sha256(
                    policy_collision_certificate
                )
            ),
            task_wrench_contract_sha256=task_contract_sha256,
            scenario_design_sha256=scenario_design_sha256,
            friction_scenario_sha256=friction_scenario_sha256,
            object_geometry_sha256=self.object_model.geometry_sha256,
            model_contract_sha256=v9_audit.model_contract_sha256,
            pad_order=policy.pad_order,
            pad_domains=pad_domains,
            possible_root_counts=counts,
            total_possible_root_count=sum(counts),
            cartesian_product_count=math.prod(counts),
            cartesian_product_rule=CONTACT_RANGE_POLICY_WRENCH_PRODUCT_RULE,
            scenario_count=int(scenarios.shape[0]),
            scenario_dimension=int(scenarios.shape[1]),
            hard_bound_friction_coefficient=(
                self.friction_coefficient_interval[0]
            ),
            uncertainty_claim_scope=self.uncertainty_claim_scope,
            policy_contact_root_domains_consumed=True,
            complete_cartesian_product_bound=True,
            display_approximation_used_as_formal_evidence=False,
            finite_contact_geometry_sampling_used_as_formal_evidence=False,
            exact_candidate_wrench_invocation_count=0,
            contact_range_margin_computed=False,
            interval_contact_jacobian_certificate_present=False,
            parametric_wrench_lower_bound_certificate_present=False,
            formal_selection_allowed=False,
            blockers=CONTACT_RANGE_POLICY_WRENCH_MANDATORY_BLOCKERS,
            claim_limitations=CONTACT_RANGE_POLICY_WRENCH_CLAIM_LIMITATIONS,
        )
        return ContactRangePolicyWrenchCertificate(
            state=ContactRangePolicyWrenchState.NOT_CERTIFIABLE,
            audit=audit,
        )

    def _build_task_wrench_definition(self) -> TaskWrenchDefinition:
        mass = self.object_model.mass_kg
        force_scale = mass * self.gravity_acceleration_m_s2
        moment_scale = force_scale * self.characteristic_radius_m
        task_axes = self.task_frame_rotation_object
        vertices = np.zeros((12, 6), dtype=np.float64)
        for axis_index in range(3):
            force_vertex = force_scale * task_axes[:, axis_index]
            moment_vertex = moment_scale * task_axes[:, axis_index]
            vertices[2 * axis_index, :3] = force_vertex
            vertices[2 * axis_index + 1, :3] = -force_vertex
            vertices[6 + 2 * axis_index, 3:] = moment_vertex
            vertices[6 + 2 * axis_index + 1, 3:] = -moment_vertex

        # d points with gravity.  Acceleration -a_lift*d requires contact force
        # -m(g+a_lift)d, so the external load balanced by the hand is positive.
        nominal = np.zeros(6, dtype=np.float64)
        nominal[:3] = (
            mass
            * (self.gravity_acceleration_m_s2 + self.lift_acceleration_m_s2)
            * self.gravity_direction_object
        )
        return TaskWrenchDefinition(
            nominal_external_wrench=nominal,
            disturbance_vertices=vertices,
            force_scale_n=force_scale,
            moment_scale_nm=moment_scale,
            wrench_origin_object_m=self.object_model.center_of_mass_m,
        )

    def friction_coefficients_from_unit(
        self, scenario_parameters_unit: np.ndarray
    ) -> np.ndarray:
        """Map the common one-dimensional Sobol design into declared friction."""

        scenarios = np.asarray(scenario_parameters_unit, dtype=np.float64)
        if (
            scenarios.ndim != 2
            or scenarios.shape[1] != self.uncertainty_dimension
            or scenarios.shape[0] < 1
            or not np.all(np.isfinite(scenarios))
            or np.any(scenarios < 0.0)
            or np.any(scenarios > 1.0)
        ):
            raise ValueError(
                "scenario_parameters_unit must have finite shape (S, 1) in [0, 1]"
            )
        lower, upper = self.friction_coefficient_interval
        return _readonly(lower + scenarios[:, 0] * (upper - lower))

    def _contact_tangent(
        self, inward_normal_object: np.ndarray
    ) -> np.ndarray:
        """Choose a task-frame tangent equivariantly, without an angle gate."""

        for axis_index in range(3):
            axis = self.task_frame_rotation_object[:, axis_index]
            tangent = axis - float(axis @ inward_normal_object) * inward_normal_object
            norm = float(np.linalg.norm(tangent))
            if norm > _FLOAT_EPS:
                return tangent / norm
        raise TaskWrenchEvaluationError(
            "proper task frame produced no finite contact tangent"
        )

    def independent_joint_torque_map(
        self,
        candidate: GraspCandidate,
        hand_model: ThreeFingerHandModel,
    ) -> ContactActuationModel:
        """Return tau = T f_object using independent/mimic-aware Jacobians."""

        transform_object_from_hand = _rigid_transform(
            candidate.object_from_hand,
            label="candidate.object_from_hand",
        )
        rotation_object_from_hand = transform_object_from_hand[:3, :3]
        rotation_hand_from_object = rotation_object_from_hand.T
        translation_object_from_hand = transform_object_from_hand[:3, 3]
        joint_positions = candidate.independent_joint_positions_rad
        try:
            link_transforms = hand_model.forward_kinematics(joint_positions)
        except HandModelError as error:
            raise TaskWrenchEvaluationError(
                f"hand forward kinematics failed: {error}"
            ) from error

        joint_names = tuple(hand_model.independent_joint_names)
        effort_limits: list[float] = []
        for joint_name in joint_names:
            joint_limit = hand_model.independent_joint_limits[joint_name]
            effort = joint_limit.effort
            if effort is None or not math.isfinite(float(effort)) or effort <= 0.0:
                raise TaskWrenchEvaluationError(
                    f"independent joint {joint_name} has no finite positive effort limit"
                )
            effort_limits.append(float(effort))

        torque_map = np.zeros(
            (len(joint_names), 3 * len(candidate.planned_pad_contacts)),
            dtype=np.float64,
        )
        contact_points_link: list[tuple[float, float, float]] = []
        for contact_index, contact in enumerate(candidate.planned_pad_contacts):
            pad = hand_model.pads.get(contact.pad_name)
            if pad is None:
                raise TaskWrenchEvaluationError(
                    f"planned contact names unknown PAD {contact.pad_name!r}"
                )
            point_object = np.asarray(contact.position_object_m, dtype=np.float64)
            point_hand = rotation_hand_from_object @ (
                point_object - translation_object_from_hand
            )
            transform_hand_from_link = np.asarray(
                link_transforms[pad.link_name], dtype=np.float64
            )
            rotation_hand_from_link = transform_hand_from_link[:3, :3]
            point_link = rotation_hand_from_link.T @ (
                point_hand - transform_hand_from_link[:3, 3]
            )
            try:
                jacobian_hand = hand_model.geometric_jacobian(
                    pad.link_name,
                    joint_positions,
                    point_local_m=point_link,
                )
            except HandModelError as error:
                raise TaskWrenchEvaluationError(
                    f"PAD Jacobian failed for {contact.pad_name}: {error}"
                ) from error
            expected_shape = (6, len(joint_names))
            if (
                jacobian_hand.shape != expected_shape
                or not np.all(np.isfinite(jacobian_hand))
            ):
                raise TaskWrenchEvaluationError(
                    f"PAD Jacobian has shape {jacobian_hand.shape}, expected {expected_shape}"
                )
            torque_map[
                :, 3 * contact_index : 3 * contact_index + 3
            ] = jacobian_hand[:3].T @ rotation_hand_from_object
            contact_points_link.append(tuple(float(item) for item in point_link))

        return ContactActuationModel(
            torque_from_object_contact_forces=torque_map,
            independent_joint_effort_limits=np.asarray(effort_limits),
            contact_points_link_m=tuple(contact_points_link),
        )

    def _contact_inputs(
        self,
        candidate: GraspCandidate,
        hand_model: ThreeFingerHandModel,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        contact_names = tuple(
            contact.pad_name for contact in candidate.planned_pad_contacts
        )
        if len(set(contact_names)) != len(contact_names):
            raise TaskWrenchEvaluationError("planned PAD contacts are not unique")
        if set(contact_names) != set(hand_model.pads):
            raise TaskWrenchEvaluationError(
                "candidate must contain exactly one contact for every physical PAD"
            )

        points = np.asarray(
            [contact.position_object_m for contact in candidate.planned_pad_contacts],
            dtype=np.float64,
        )
        free_side = np.asarray(
            [
                contact.path_local_free_side_normal_object
                for contact in candidate.planned_pad_contacts
            ],
            dtype=np.float64,
        )
        compressive_axis = -free_side
        tangents = np.asarray(
            [self._contact_tangent(normal) for normal in compressive_axis],
            dtype=np.float64,
        )
        capacities: list[float] = []
        for name in contact_names:
            capacity = hand_model.pads[name].normal_force_capacity_n
            if (
                capacity is None
                or not math.isfinite(float(capacity))
                or capacity <= 0.0
            ):
                raise TaskWrenchEvaluationError(
                    f"PAD {name} has no finite positive normal-force capacity"
                )
            capacities.append(float(capacity))
        preload = np.asarray(candidate.internal_normal_forces_n, dtype=np.float64)
        capacity_array = np.asarray(capacities, dtype=np.float64)
        if preload.shape != capacity_array.shape or np.any(preload < 0.0):
            raise TaskWrenchEvaluationError(
                "candidate preload must contain one non-negative value per PAD"
            )
        if np.any(preload > capacity_array):
            raise TaskWrenchEvaluationError(
                "candidate preload exceeds a physical PAD normal-force capacity"
            )
        return points, compressive_axis, tangents, capacity_array, preload

    def _constraint_matrix(
        self,
        *,
        actuation: ContactActuationModel,
        inward_normals: np.ndarray,
        preload_normal_forces_n: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        torque_map = actuation.torque_from_object_contact_forces
        effort = actuation.independent_joint_effort_limits
        contact_count = inward_normals.shape[0]
        preload_rows = np.zeros((contact_count, 3 * contact_count), dtype=np.float64)
        for index, normal in enumerate(inward_normals):
            preload_rows[index, 3 * index : 3 * index + 3] = -normal
        matrix = np.vstack((torque_map, -torque_map, preload_rows))
        bounds = np.concatenate((effort, effort, -preload_normal_forces_n))
        return matrix, bounds

    def _solve_margin(
        self,
        *,
        candidate: GraspCandidate,
        hand_model: ThreeFingerHandModel,
        friction_coefficient: float,
        actuation: ContactActuationModel,
        contact_inputs: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> TaskWrenchMarginResult:
        del candidate, hand_model
        points, compressive_axes, tangents, capacities, preload = contact_inputs
        constraint_matrix, constraint_bounds = self._constraint_matrix(
            actuation=actuation,
            inward_normals=compressive_axes,
            preload_normal_forces_n=preload,
        )
        model = build_polyhedral_contact_wrench_model(
            contact_points_m=points,
            inward_normals=compressive_axes,
            tangent_directions=tangents,
            friction_coefficients=float(friction_coefficient),
            normal_force_caps_n=capacities,
            wrench_origin_m=self.object_model.center_of_mass_m,
            maximum_inner_approximation_relative_error=(
                self.maximum_inner_approximation_relative_error
            ),
            cone_edge_multiplier=self.cone_edge_multiplier,
            contact_force_inequality_matrix=constraint_matrix,
            contact_force_inequality_upper_bounds=constraint_bounds,
        )
        torque_from_rays = (
            actuation.torque_from_object_contact_forces
            @ model.contact_force_matrix
        )
        torque_utilization_from_rays = torque_from_rays / (
            actuation.independent_joint_effort_limits[:, None]
        )
        signed_torque_utilization_from_rays = np.vstack(
            (
                torque_utilization_from_rays,
                -torque_utilization_from_rays,
            )
        )
        result = maximum_task_wrench_polytope_margin(
            model,
            nominal_external_wrench=(
                self.task_wrench_definition.nominal_external_wrench
            ),
            disturbance_vertices=self.task_wrench_definition.disturbance_vertices,
            solver_options=self.solver_options,
            lexicographic_ray_load_groups=(
                model.normal_force_matrix,
                signed_torque_utilization_from_rays,
            ),
        )
        if not result.solver_success or result.maximum_margin is None:
            raise TaskWrenchEvaluationError(
                "task-wrench LP failed closed: "
                f"status={result.solver_status}, message={result.solver_message}"
            )
        if (
            result.maximum_scaled_equilibrium_residual is None
            or result.maximum_scaled_inequality_violation is None
            or result.maximum_scaled_equilibrium_residual
            > self.solver_options.primal_feasibility_tolerance
            or result.maximum_scaled_inequality_violation
            > self.solver_options.primal_feasibility_tolerance
        ):
            raise TaskWrenchEvaluationError(
                "task-wrench LP certificate exceeds the explicit primal tolerance"
            )
        if not math.isfinite(result.maximum_margin) or result.maximum_margin < 0.0:
            raise TaskWrenchEvaluationError(
                "task-wrench LP returned an invalid non-finite/negative margin"
            )
        if (
            result.lexicographic_optimal_loads is None
            or len(result.lexicographic_optimal_loads) != 2
            or not all(
                math.isfinite(float(value)) and float(value) >= 0.0
                for value in result.lexicographic_optimal_loads
            )
        ):
            raise TaskWrenchEvaluationError(
                "task-wrench LP omitted its two certified lexicographic loads"
            )
        expected_stage_names = (
            "MAXIMIZE_SHARED_TASK_MARGIN",
            "MINIMIZE_LEXICOGRAPHIC_RAY_LOAD_GROUP_0",
            "MINIMIZE_LEXICOGRAPHIC_RAY_LOAD_GROUP_1",
        )
        stage_results = result.lexicographic_stage_results
        if (
            tuple(stage.stage_name for stage in stage_results)
            != expected_stage_names
            or any(
                not stage.solver_success
                or stage.optimal_value is None
                or stage.maximum_scaled_equilibrium_residual is None
                or stage.maximum_scaled_inequality_violation is None
                or stage.maximum_scaled_equilibrium_residual
                > self.solver_options.primal_feasibility_tolerance
                or stage.maximum_scaled_inequality_violation
                > self.solver_options.primal_feasibility_tolerance
                for stage in stage_results
            )
        ):
            raise TaskWrenchEvaluationError(
                "task-wrench LP omitted a certified lexicographic stage"
            )
        expected_stage_values = (
            float(result.maximum_margin),
            float(result.lexicographic_optimal_loads[0]),
            float(result.lexicographic_optimal_loads[1]),
        )
        if tuple(stage.optimal_value for stage in stage_results) != (
            expected_stage_values
        ):
            raise TaskWrenchEvaluationError(
                "task-wrench LP stage objectives disagree with its certificate"
            )
        if (
            result.ray_coefficients_by_vertex is None
            or result.contact_forces_by_vertex is None
            or result.normal_forces_by_vertex is None
        ):
            raise TaskWrenchEvaluationError(
                "successful lexicographic LP omitted its final-stage allocation"
            )
        ray_coefficients = np.asarray(result.ray_coefficients_by_vertex)
        contact_forces = np.asarray(result.contact_forces_by_vertex)
        normal_forces = np.asarray(result.normal_forces_by_vertex)
        vertex_count = self.task_wrench_definition.disturbance_vertices.shape[0]
        if (
            ray_coefficients.shape != (vertex_count, model.ray_count)
            or contact_forces.shape
            != (vertex_count, model.contact_count, 3)
            or normal_forces.shape != (vertex_count, model.contact_count)
            or not np.all(np.isfinite(ray_coefficients))
            or not np.all(np.isfinite(contact_forces))
            or not np.all(np.isfinite(normal_forces))
        ):
            raise TaskWrenchEvaluationError(
                "task-wrench LP returned malformed final-stage allocations"
            )
        return result

    @staticmethod
    def _trajectory_clearance(
        *,
        surface_model: ObjectSurfaceModel,
        candidate: GraspCandidate,
        hand_model: ThreeFingerHandModel,
    ) -> float:
        scope = getattr(surface_model, "trajectory_clearance_scope", None)
        if scope != COMPLETE_CONTINUOUS_TRAJECTORY_CLEARANCE_SCOPE:
            raise TaskWrenchEvaluationError(
                "surface model trajectory clearance is not certified for the "
                "complete hand-object-environment approach/closure/lift path"
            )
        method = getattr(surface_model, "trajectory_clearance_m", None)
        if not callable(method):
            raise TaskWrenchEvaluationError(
                "surface model must explicitly provide trajectory_clearance_m"
            )
        clearance = float(method(candidate, hand_model))
        if not math.isfinite(clearance):
            raise TaskWrenchEvaluationError(
                "surface trajectory clearance must be finite"
            )
        return clearance

    def evaluate_task_wrench(
        self,
        candidate: GraspCandidate,
        scenario_parameters_unit: np.ndarray,
        *,
        hand_model: ThreeFingerHandModel,
    ) -> TaskWrenchOnlyEvaluation:
        """Evaluate one candidate once, without inventing collision clearance."""

        friction_values = self.friction_coefficients_from_unit(
            scenario_parameters_unit
        )
        contact_inputs = self._contact_inputs(candidate, hand_model)
        actuation = self.independent_joint_torque_map(candidate, hand_model)
        scenario_results = tuple(
            self._solve_margin(
                candidate=candidate,
                hand_model=hand_model,
                friction_coefficient=float(friction),
                actuation=actuation,
                contact_inputs=contact_inputs,
            )
            for friction in friction_values
        )
        lower_bound_result = self._solve_margin(
            candidate=candidate,
            hand_model=hand_model,
            friction_coefficient=self.friction_coefficient_interval[0],
            actuation=actuation,
            contact_inputs=contact_inputs,
        )
        all_results = scenario_results + (lower_bound_result,)
        margins = tuple(float(result.maximum_margin) for result in scenario_results)

        lexicographic_loads = tuple(
            result.lexicographic_optimal_loads for result in all_results
        )
        if any(loads is None for loads in lexicographic_loads):
            raise TaskWrenchEvaluationError(
                "successful LP omitted its lexicographic load certificate"
            )
        peak_normal_force = max(
            float(loads[0]) for loads in lexicographic_loads if loads is not None
        )
        maximum_torque_utilization = max(
            float(loads[1]) for loads in lexicographic_loads if loads is not None
        )

        def stage_diagnostics(
            result: TaskWrenchMarginResult,
        ) -> tuple[Mapping[str, Any], ...]:
            return tuple(
                MappingProxyType(
                    {
                        "stage_name": stage.stage_name,
                        "solver_success": stage.solver_success,
                        "solver_status": stage.solver_status,
                        "solver_message": stage.solver_message,
                        "optimal_value": stage.optimal_value,
                        "maximum_scaled_equilibrium_residual": (
                            stage.maximum_scaled_equilibrium_residual
                        ),
                        "maximum_scaled_inequality_violation": (
                            stage.maximum_scaled_inequality_violation
                        ),
                    }
                )
                for stage in result.lexicographic_stage_results
            )

        diagnostics: Mapping[str, Any] = MappingProxyType(
            {
                "uncertainty_parameter_names": ("friction_coefficient",),
                "certified_uncertainty_scope": self.uncertainty_claim_scope,
                "uncalibrated_uncertainties_excluded_from_certified_set": (
                    "object_pose",
                    "surface_position_and_normal",
                    "mass_center_of_mass_and_inertia",
                    "joint_tracking",
                    "actuator_capability",
                ),
                "friction_coefficients": tuple(
                    float(value) for value in friction_values
                ),
                "hard_bound_friction_coefficient": float(
                    self.friction_coefficient_interval[0]
                ),
                "disturbance_body": "CENTRALLY_SYMMETRIC_6D_CROSS_POLYTOPE",
                "disturbance_vertex_count": 12,
                "force_scale_n": self.task_wrench_definition.force_scale_n,
                "moment_scale_nm": self.task_wrench_definition.moment_scale_nm,
                "nominal_external_wrench": tuple(
                    float(value)
                    for value in self.task_wrench_definition.nominal_external_wrench
                ),
                "wrench_origin_object_m": tuple(
                    float(value)
                    for value in self.task_wrench_definition.wrench_origin_object_m
                ),
                "nominal_moment_about_com_nm": (0.0, 0.0, 0.0),
                "gravity_direction_sign_convention": (
                    "EXTERNAL_GRAVITY_ALONG_DECLARED_DIRECTION;"
                    "POSITIVE_LIFT_ACCELERATION_OPPOSES_GRAVITY"
                ),
                "normal_force_capacity_n": tuple(
                    float(value) for value in contact_inputs[3]
                ),
                "independent_joint_effort_limits": tuple(
                    float(value)
                    for value in actuation.independent_joint_effort_limits
                ),
                "independent_joint_effort_limit_role": (
                    "URDF_DECLARED_UNCALIBRATED_OPTIMIZATION_CONSTRAINT"
                ),
                "independent_joint_torque_map_shape": tuple(
                    int(value)
                    for value in actuation.torque_from_object_contact_forces.shape
                ),
                "lexicographic_load_group_roles": (
                    "PEAK_PAD_NORMAL_FORCE_N",
                    "PEAK_ABSOLUTE_INDEPENDENT_JOINT_TORQUE_UTILIZATION",
                ),
                "scenario_lexicographic_optimal_loads": tuple(
                    tuple(float(value) for value in result.lexicographic_optimal_loads)
                    for result in scenario_results
                    if result.lexicographic_optimal_loads is not None
                ),
                "hard_bound_lexicographic_optimal_loads": tuple(
                    float(value)
                    for value in lower_bound_result.lexicographic_optimal_loads
                ),
                "scenario_lexicographic_stage_reports": tuple(
                    stage_diagnostics(result) for result in scenario_results
                ),
                "hard_bound_lexicographic_stage_reports": stage_diagnostics(
                    lower_bound_result
                ),
                "solver": self.solver_options.solver,
                "requested_constraint_scaling": (
                    self.solver_options.requested_constraint_scaling
                ),
                "actual_constraint_scaling": scenario_results[
                    0
                ].constraint_scaling_implementation,
                "residual_coordinate_system": (
                    "EXPLICIT_EQUILIBRATED_SOLVER_COORDINATES"
                ),
                "primal_feasibility_tolerance": (
                    self.solver_options.primal_feasibility_tolerance
                ),
                "dual_feasibility_tolerance": (
                    self.solver_options.dual_feasibility_tolerance
                ),
                "ipm_optimality_tolerance": (
                    self.solver_options.ipm_optimality_tolerance
                ),
            }
        )
        return TaskWrenchOnlyEvaluation(
            task_margins=margins,
            hard_bound_minimum_task_margin=float(
                lower_bound_result.maximum_margin
            ),
            peak_normal_force_n=peak_normal_force,
            joint_torque_utilization=maximum_torque_utilization,
            diagnostics=diagnostics,
        )

    def evaluate(
        self,
        candidate: GraspCandidate,
        scenario_parameters_unit: np.ndarray,
        *,
        surface_model: ObjectSurfaceModel,
        hand_model: ThreeFingerHandModel,
    ) -> WrenchEvaluation:
        """Compatibility API that combines wrench and certified clearance."""

        wrench = self.evaluate_task_wrench(
            candidate,
            scenario_parameters_unit,
            hand_model=hand_model,
        )
        clearance = self._trajectory_clearance(
            surface_model=surface_model,
            candidate=candidate,
            hand_model=hand_model,
        )
        diagnostics = dict(wrench.diagnostics)
        diagnostics["trajectory_clearance_scope"] = (
            COMPLETE_CONTINUOUS_TRAJECTORY_CLEARANCE_SCOPE
        )
        return WrenchEvaluation(
            task_margins=wrench.task_margins,
            hard_bound_minimum_task_margin=(
                wrench.hard_bound_minimum_task_margin
            ),
            peak_normal_force_n=wrench.peak_normal_force_n,
            joint_torque_utilization=wrench.joint_torque_utilization,
            trajectory_clearance_m=clearance,
            feasible=True,
            diagnostics=_deep_freeze(diagnostics),
        )


__all__ = [
    "COMPLETE_CONTINUOUS_TRAJECTORY_CLEARANCE_SCOPE",
    "CONTACT_RANGE_POLICY_WRENCH_CLAIM_LIMITATIONS",
    "CONTACT_RANGE_POLICY_WRENCH_MANDATORY_BLOCKERS",
    "CONTACT_RANGE_POLICY_WRENCH_METHOD_ID",
    "CONTACT_RANGE_POLICY_WRENCH_PRODUCT_RULE",
    "ContactActuationModel",
    "ContactRangePadWrenchDomain",
    "ContactRangePolicyWrenchAudit",
    "ContactRangePolicyWrenchCertificate",
    "ContactRangePolicyWrenchState",
    "ContactRangeRootWrenchDomain",
    "FRICTION_INTERVAL_ONLY_CERTIFIED_UNCERTAINTY_SCOPE",
    "TaskWrenchOnlyEvaluation",
    "TaskWrenchDefinition",
    "TaskWrenchEvaluationError",
    "TaskWrenchEvaluator",
]
