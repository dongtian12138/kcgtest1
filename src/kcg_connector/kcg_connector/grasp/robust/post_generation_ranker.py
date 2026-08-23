"""Post-generation, rank-only evaluation for frozen CARTS V9 outputs.

This module cannot generate, refine, replace, or refill a candidate.  It
consumes the complete :class:`TopLevelGenerationResult`, validates its exact
V9/model lineage, and evaluates every unique statically accepted exact
candidate or certified contact-range closure policy with one common immutable
Sobol design.  A policy is never collapsed to a display midpoint or invented
exact contact endpoint.

The current task-wrench evaluator certifies only the declared friction
interval, and the current FullHand V1 certificate is deliberately
``NOT_CERTIFIABLE``.  Therefore this V2 rank-only bridge may report a
diagnostic order when a separately typed complete-clearance certificate is
available, but it never creates a formal winner.  ``selected_candidate`` and
``selected_contact_range_policy`` must remain ``None`` until a future
evaluator actually consumes all calibrated non-friction uncertainty bounds
and a complete approach/closure/lift collision certificate.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
import hashlib
import json
import math
import re
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import scipy

from kcg_connector.grasp.robust.full_hand_collision import (
    CONTACT_RANGE_POLICY_METHOD_ID,
    ContactRangePolicyCollisionCertificate,
    FullHandClosureCollisionCertificate,
    FullHandClosureCollisionState,
    METHOD_ID as FULL_HAND_COLLISION_METHOD_ID,
)
from kcg_connector.grasp.robust.grasp_optimizer import (
    GraspCandidate,
    deterministic_sobol,
)
from kcg_connector.grasp.robust.pareto_ranker import (
    CandidateMetrics,
    pareto_layers,
    qmc_lower_tail_mean,
)
from kcg_connector.grasp.robust.ray_closure import (
    CANDIDATE_REPRESENTATIVE_ROLE,
    CLOSURE_PARAMETER_DOMAIN_ID,
    CertifiedSequentialClosurePolicy,
    METHOD_ID as RAY_CLOSURE_METHOD_ID,
    MODEL_BINDING_COMPLETE_STATUS,
    REPRESENTATIVE_PROPOSAL_FAILURE_REASON,
    RayClosureAudit,
)
from kcg_connector.grasp.robust.task_wrench_evaluator import (
    ContactRangePolicyWrenchCertificate,
    ContactRangePolicyWrenchState,
    FRICTION_INTERVAL_ONLY_CERTIFIED_UNCERTAINTY_SCOPE,
    TaskWrenchEvaluator,
    TaskWrenchOnlyEvaluation,
)
from kcg_connector.grasp.robust.top_level_candidate_generator import (
    ALLOWED_TOTAL_ATTEMPT_BUDGETS,
    AttemptStatus,
    CandidateAttemptAudit,
    StaticV9AcceptedCandidate,
    StaticV9AcceptedPolicy,
    TopLevelGenerationResult,
    UniqueV9Evaluation,
    V9InvocationAuditBinding,
    METHOD_ID as TOP_LEVEL_GENERATOR_METHOD_ID,
    canonicalize_v9_parameters,
)


METHOD_ID = "CARTS_POST_GENERATION_COMMON_QMC_LEXICOGRAPHIC_RANKER_V2"
COMPLETE_CLEARANCE_METHOD_ID = (
    "CARTS_COMPLETE_HAND_OBJECT_ENVIRONMENT_APPROACH_CLOSURE_LIFT_"
    "CLEARANCE_CERTIFICATE_V1"
)
COMPLETE_CLEARANCE_SCOPE = (
    "COMPLETE_HAND_OBJECT_ENVIRONMENT_APPROACH_CLOSURE_LIFT_"
    "CONTINUOUS_COLLISION_CERTIFIED_LOWER_BOUND"
)
COMPLETE_POLICY_CLEARANCE_METHOD_ID = (
    "CARTS_COMPLETE_CONTACT_RANGE_POLICY_HAND_OBJECT_ENVIRONMENT_APPROACH_"
    "CLOSURE_LIFT_CLEARANCE_CERTIFICATE_V1"
)
SCENARIO_METHOD_ID = "CARTS_COMMON_SCRAMBLED_SOBOL_FRICTION_DIAGNOSTIC_V1"
SCENARIO_DIMENSION = 1
SCENARIO_COUNT = 128
SCENARIO_SOBOL_SEED = 20260820
SCENARIO_SCIPY_VERSION = "1.8.0"
SCENARIO_DESIGN_SHA256 = (
    "dd1115f4bd4c52c29fa29261037fe7ef96d79bca10dad6c7a8a066eef3f05631"
)
SELECTION_ORDER = (
    "hard_bound_minimum_task_margin",
    "qmc_lower_tail_mean_task_margin",
    "minimum_peak_normal_force_n",
    "minimum_joint_torque_utilization",
    "maximum_trajectory_clearance_m",
)
TIE_BREAK_RULE = "FIRST_TOP_LEVEL_ATTEMPT_INDEX_THEN_V9_PARAMETER_KEY_HEX"
FORMAL_UNCERTAINTY_BLOCKER = (
    "MISSING_CALIBRATED_NONFRICTION_UNCERTAINTY_BOUNDS"
)
POLICY_AWARE_RANKING_STATUS = (
    "POLICY_AWARE_COLLISION_AND_WRENCH_CONSUMERS_ACTIVE"
)
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class PostGenerationRankingError(ValueError):
    """Raised when the shared ranking protocol itself is malformed."""


class CandidateEvaluationState(str, Enum):
    """One terminal, non-overclaiming state for an accepted V9 candidate."""

    FORMALLY_SCORED = "FORMALLY_SCORED"
    CERTIFIED_PHYSICALLY_INFEASIBLE = "CERTIFIED_PHYSICALLY_INFEASIBLE"
    UNRESOLVED_COLLISION = "UNRESOLVED_COLLISION"
    UNRESOLVED_WRENCH = "UNRESOLVED_WRENCH"
    UNCERTAINTY_SCOPE_INCOMPLETE = "UNCERTAINTY_SCOPE_INCOMPLETE"
    PROTOCOL_REJECTED = "PROTOCOL_REJECTED"


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and _HEX_SHA256.fullmatch(value) is not None


def _json_value(value: Any) -> Any:
    """Convert supported evidence into a canonical JSON-compatible value."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PostGenerationRankingError(
                "canonical evidence cannot contain a non-finite float"
            )
        return value
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, bytes):
        return {"binary_hex": value.hex()}
    if isinstance(value, Mapping):
        rows: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise PostGenerationRankingError(
                    "canonical evidence mappings require string keys"
                )
            rows[key] = _json_value(child)
        return rows
    if isinstance(value, (tuple, list)):
        return [_json_value(child) for child in value]
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in fields(value)
        }
    as_dict = getattr(value, "as_dict", None)
    if callable(as_dict):
        return _json_value(as_dict())
    raise PostGenerationRankingError(
        "evidence is not canonically serializable: "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _candidate_document(candidate: GraspCandidate) -> dict[str, Any]:
    return {
        "object_from_hand": list(candidate.object_from_hand),
        "independent_joint_positions_rad": list(
            candidate.independent_joint_positions_rad
        ),
        "planned_pad_contacts": [
            {
                "pad_name": row.pad_name,
                "position_object_m": list(row.position_object_m),
                "path_local_free_side_normal_object": list(
                    row.path_local_free_side_normal_object
                ),
                "surface_coordinates": list(row.surface_coordinates),
            }
            for row in candidate.planned_pad_contacts
        ],
        "internal_normal_forces_n": list(
            candidate.internal_normal_forces_n
        ),
        "stiffness_diagonal": list(candidate.stiffness_diagonal),
        "damping_diagonal": list(candidate.damping_diagonal),
    }


def candidate_sha256(candidate: GraspCandidate) -> str:
    """Return the canonical physical-decision identity used by this stage."""

    if type(candidate) is not GraspCandidate:
        raise PostGenerationRankingError(
            "rank-only input candidate must be exactly GraspCandidate"
        )
    return _sha256(
        {
            "method_id": METHOD_ID,
            "role": "GRASP_CANDIDATE",
            "candidate": _candidate_document(candidate),
        }
    )


def v9_evidence_sha256(
    candidate: GraspCandidate,
    audit: RayClosureAudit,
) -> str:
    """Match the exact V9 evidence payload bound by FullHand collision V1."""

    if (
        type(candidate) is not GraspCandidate
        or type(audit) is not RayClosureAudit
    ):
        raise PostGenerationRankingError(
            "V9 evidence requires exact GraspCandidate and RayClosureAudit"
        )
    return _sha256(
        {
            "audit": audit.as_dict(),
            "candidate": _candidate_document(candidate),
        }
    )


@dataclass(frozen=True)
class CommonScenarioDesign:
    method_id: str
    scipy_version: str
    dimension: int
    count: int
    sobol_seed: int
    scramble: bool
    optimization: None
    identity_encoding: str
    design_sha256: str
    parameters_unit: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if (
            self.method_id != SCENARIO_METHOD_ID
            or self.scipy_version != SCENARIO_SCIPY_VERSION
            or self.dimension != SCENARIO_DIMENSION
            or self.count != SCENARIO_COUNT
            or self.sobol_seed != SCENARIO_SOBOL_SEED
            or self.scramble is not True
            or self.optimization is not None
            or self.identity_encoding != "BIG_ENDIAN_BINARY64_ROW_MAJOR"
            or self.design_sha256 != SCENARIO_DESIGN_SHA256
            or len(self.parameters_unit) != self.count
            or any(len(row) != self.dimension for row in self.parameters_unit)
        ):
            raise PostGenerationRankingError(
                "common scenario design differs from the frozen contract"
            )
        array = np.asarray(self.parameters_unit, dtype=np.float64)
        digest = hashlib.sha256(
            np.asarray(array, dtype=">f8").tobytes(order="C")
        ).hexdigest()
        if digest != self.design_sha256:
            raise PostGenerationRankingError(
                "common scenario tuple contradicts its realized SHA-256"
            )


@dataclass(frozen=True)
class CompleteTrajectoryCollisionCertificate:
    """Strongly typed future complete-clearance bridge.

    FullHand V1 is intentionally not this type.  This record can enable only
    diagnostic five-metric ordering in the present friction-only pipeline; it
    cannot enable a formal result.
    """

    method_id: str
    claim_scope: str
    source_certificate_sha256: str
    candidate_sha256: str
    v9_evidence_sha256: str
    model_contract_sha256: str
    trajectory_clearance_lower_bound_m: float
    blockers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        digests = (
            self.source_certificate_sha256,
            self.candidate_sha256,
            self.v9_evidence_sha256,
            self.model_contract_sha256,
        )
        clearance = float(self.trajectory_clearance_lower_bound_m)
        if (
            self.method_id != COMPLETE_CLEARANCE_METHOD_ID
            or self.claim_scope != COMPLETE_CLEARANCE_SCOPE
            or any(not _valid_sha256(value) for value in digests)
            or not math.isfinite(clearance)
            or clearance < 0.0
            or tuple(self.blockers)
        ):
            raise PostGenerationRankingError(
                "complete trajectory collision certificate is malformed"
            )
        object.__setattr__(
            self, "trajectory_clearance_lower_bound_m", clearance
        )
        object.__setattr__(self, "blockers", ())

    @property
    def certificate_sha256(self) -> str:
        return _sha256(self)


@dataclass(frozen=True)
class CompleteContactRangeTrajectoryCollisionCertificate:
    """Future complete-clearance proof bound to one contact-range policy.

    The embedded current policy certificate supplies the already checked
    closure-range collision and the exact binding required by the wrench
    evaluator.  This stronger wrapper may be created only after the remaining
    PAD, containment, arm, environment, approach, and lift scopes are proven.
    It still cannot remove the independent non-friction uncertainty blocker.
    """

    method_id: str
    claim_scope: str
    source_policy_collision_certificate: ContactRangePolicyCollisionCertificate
    policy_sha256: str
    v9_policy_evidence_sha256: str
    model_contract_sha256: str
    trajectory_clearance_lower_bound_m: float
    blockers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        source = self.source_policy_collision_certificate
        clearance = float(self.trajectory_clearance_lower_bound_m)
        if (
            self.method_id != COMPLETE_POLICY_CLEARANCE_METHOD_ID
            or self.claim_scope != COMPLETE_CLEARANCE_SCOPE
            or type(source) is not ContactRangePolicyCollisionCertificate
            or source.state is not FullHandClosureCollisionState.NOT_CERTIFIABLE
            or not source.audit.checkable_collision_gates_passed
            or source.audit.policy_sha256 != self.policy_sha256
            or source.audit.v9_audit_and_policy_sha256
            != self.v9_policy_evidence_sha256
            or source.audit.ray_model_contract_sha256
            != self.model_contract_sha256
            or any(
                not _valid_sha256(value)
                for value in (
                    self.policy_sha256,
                    self.v9_policy_evidence_sha256,
                    self.model_contract_sha256,
                )
            )
            or not math.isfinite(clearance)
            or clearance < 0.0
            or tuple(self.blockers)
        ):
            raise PostGenerationRankingError(
                "complete contact-range collision certificate is malformed"
            )
        object.__setattr__(
            self, "trajectory_clearance_lower_bound_m", clearance
        )
        object.__setattr__(self, "blockers", ())

    @property
    def source_certificate_sha256(self) -> str:
        return _sha256(self.source_policy_collision_certificate)

    @property
    def certificate_sha256(self) -> str:
        return _sha256(self)


@dataclass(frozen=True)
class GenerationAttemptSnapshot:
    attempt_index: int
    lane: str
    lane_point_index: int
    sobol_seed: int
    sobol_parameters_unit: tuple[float, ...]
    anchor_pad_name: str | None
    proposal_audit_sha256: str | None
    proposal_failure_reason: str | None
    status: str
    v9_parameters_unit: tuple[float, ...] | None
    v9_parameter_key_hex: str | None
    duplicate_of_attempt_index: int | None
    v9_audit_sha256: str | None
    invocation_binding_sha256: str | None
    v9_failure_reason: str | None
    failure_reason: str | None


@dataclass(frozen=True)
class DiagnosticWrenchSummary:
    hard_bound_minimum_task_margin: float
    qmc_lower_tail_mean_task_margin: float
    peak_normal_force_n: float
    joint_torque_utilization: float

    def __post_init__(self) -> None:
        values = tuple(
            float(getattr(self, field.name)) for field in fields(self)
        )
        if not all(math.isfinite(value) for value in values):
            raise PostGenerationRankingError(
                "diagnostic wrench summary must be finite"
            )
        if values[0] < 0.0 or values[2] < 0.0 or values[3] < 0.0:
            raise PostGenerationRankingError(
                "diagnostic wrench summary has an invalid sign"
            )


@dataclass(frozen=True)
class RankCandidateRecord:
    evaluation_index: int
    v9_parameter_key_hex: str
    v9_parameters_unit: tuple[float, ...]
    first_attempt_index: int
    lineage_attempt_indices: tuple[int, ...]
    generation_lineage_sha256: str
    candidate: GraspCandidate
    candidate_sha256: str
    v9_audit_sha256: str
    v9_evidence_sha256: str
    v9_model_contract_sha256: str
    common_scenario_design_sha256: str
    collision_invocation_count: int
    wrench_invocation_count: int
    collision_certificate_sha256: str | None
    collision_state: str
    trajectory_clearance_lower_bound_m: float | None
    wrench_evaluation_sha256: str | None
    certified_uncertainty_scope: str | None
    diagnostic_wrench: DiagnosticWrenchSummary | None
    diagnostic_metrics: CandidateMetrics | None
    formal_metrics: CandidateMetrics | None
    state: CandidateEvaluationState
    blockers: tuple[str, ...]
    diagnostic_rank: int | None = None
    formal_rank: int | None = None
    pareto_layer: int | None = None

    def __post_init__(self) -> None:
        if (
            self.collision_invocation_count != 1
            or self.wrench_invocation_count != 1
            or self.common_scenario_design_sha256
            != SCENARIO_DESIGN_SHA256
            or self.blockers != tuple(sorted(set(self.blockers)))
            or self.formal_metrics is not None
            or self.formal_rank is not None
        ):
            raise PostGenerationRankingError(
                "candidate rank record violates the V2 fail-closed contract"
            )


@dataclass(frozen=True)
class RankContactRangePolicyRecord:
    evaluation_index: int
    v9_parameter_key_hex: str
    v9_parameters_unit: tuple[float, ...]
    first_attempt_index: int
    lineage_attempt_indices: tuple[int, ...]
    generation_lineage_sha256: str
    sequential_closure_policy: CertifiedSequentialClosurePolicy
    policy_sha256: str
    v9_audit_sha256: str
    v9_policy_evidence_sha256: str | None
    v9_model_contract_sha256: str
    common_scenario_design_sha256: str
    collision_invocation_count: int
    wrench_invocation_count: int
    collision_certificate_sha256: str | None
    collision_state: str
    trajectory_clearance_lower_bound_m: float | None
    wrench_evaluation_sha256: str | None
    wrench_state: str | None
    certified_uncertainty_scope: str | None
    diagnostic_wrench: DiagnosticWrenchSummary | None
    diagnostic_metrics: CandidateMetrics | None
    formal_metrics: CandidateMetrics | None
    state: CandidateEvaluationState
    blockers: tuple[str, ...]
    diagnostic_rank: int | None = None
    formal_rank: int | None = None
    pareto_layer: int | None = None

    def __post_init__(self) -> None:
        if (
            type(self.sequential_closure_policy)
            is not CertifiedSequentialClosurePolicy
            or self.policy_sha256
            != self.sequential_closure_policy.policy_sha256
            or not _valid_sha256(self.policy_sha256)
            or not _valid_sha256(self.v9_audit_sha256)
            or self.v9_policy_evidence_sha256 is not None
            and not _valid_sha256(self.v9_policy_evidence_sha256)
            or not _valid_sha256(self.v9_model_contract_sha256)
            or self.collision_invocation_count != 1
            or self.wrench_invocation_count not in (0, 1)
            or self.wrench_invocation_count == 0
            and self.wrench_evaluation_sha256 is not None
            or self.common_scenario_design_sha256
            != SCENARIO_DESIGN_SHA256
            or self.blockers != tuple(sorted(set(self.blockers)))
            or self.formal_metrics is not None
            or self.formal_rank is not None
        ):
            raise PostGenerationRankingError(
                "contact-range policy rank record violates the V2 "
                "fail-closed contract"
            )


@dataclass(frozen=True)
class PostGenerationRankResult:
    method_id: str
    rank_contract_sha256: str
    generation_contract_sha256: str
    generation_result_sha256: str
    common_scenario_design: CommonScenarioDesign
    generation_attempt_lineage: tuple[GenerationAttemptSnapshot, ...]
    candidate_records: tuple[RankCandidateRecord, ...]
    contact_range_policy_records: tuple[RankContactRangePolicyRecord, ...]
    diagnostic_ranked_keys: tuple[str, ...]
    diagnostic_ranked_policy_keys: tuple[str, ...]
    formal_ranked_keys: tuple[str, ...]
    formal_ranked_policy_keys: tuple[str, ...]
    selected_v9_parameter_key_hex: str | None
    selected_candidate: GraspCandidate | None
    selected_contact_range_policy_key_hex: str | None
    selected_contact_range_policy: CertifiedSequentialClosurePolicy | None
    selection_blockers: tuple[str, ...]
    all_unique_candidates_resolved: bool
    canonical_json_bytes: bytes
    canonical_sha256: str

    def __post_init__(self) -> None:
        if (
            self.method_id != METHOD_ID
            or not _valid_sha256(self.rank_contract_sha256)
            or not _valid_sha256(self.generation_contract_sha256)
            or not _valid_sha256(self.generation_result_sha256)
            or self.formal_ranked_keys
            or self.formal_ranked_policy_keys
            or self.selected_v9_parameter_key_hex is not None
            or self.selected_candidate is not None
            or self.selected_contact_range_policy_key_hex is not None
            or self.selected_contact_range_policy is not None
            or self.selection_blockers != tuple(
                sorted(set(self.selection_blockers))
            )
            or hashlib.sha256(self.canonical_json_bytes).hexdigest()
            != self.canonical_sha256
        ):
            raise PostGenerationRankingError(
                "post-generation result violates its immutable V2 contract"
            )


def _audit_sha256(value: Any | None) -> str | None:
    return None if value is None else _sha256(value)


def _invocation_document(binding: V9InvocationAuditBinding) -> dict[str, Any]:
    return {
        "method_id": binding.method_id,
        "parameter_domain_id": binding.parameter_domain_id,
        "parameter_layout": list(binding.parameter_layout),
        "requested_parameters_unit": list(binding.requested_parameters_unit),
        "requested_parameter_key_hex": binding.requested_parameter_key_hex,
        "raw_v9_audit_sha256": _audit_sha256(binding.raw_v9_audit),
    }


def _attempt_snapshot(row: CandidateAttemptAudit) -> GenerationAttemptSnapshot:
    binding = row.invocation_binding
    return GenerationAttemptSnapshot(
        attempt_index=row.attempt_index,
        lane=row.lane.value,
        lane_point_index=row.lane_point_index,
        sobol_seed=row.lineage.sobol_seed,
        sobol_parameters_unit=tuple(row.sobol_parameters_unit),
        anchor_pad_name=row.anchor_pad_name,
        proposal_audit_sha256=_audit_sha256(row.lineage.proposal_audit),
        proposal_failure_reason=row.lineage.proposal_failure_reason,
        status=row.status.value,
        v9_parameters_unit=(
            None
            if row.v9_parameters_unit is None
            else tuple(row.v9_parameters_unit)
        ),
        v9_parameter_key_hex=row.v9_parameter_key_hex,
        duplicate_of_attempt_index=row.duplicate_of_attempt_index,
        v9_audit_sha256=_audit_sha256(row.v9_audit),
        invocation_binding_sha256=(
            None if binding is None else _sha256(_invocation_document(binding))
        ),
        v9_failure_reason=row.v9_failure_reason,
        failure_reason=row.failure_reason,
    )


def _build_common_scenario_design(
) -> tuple[CommonScenarioDesign, np.ndarray]:
    if scipy.__version__ != SCENARIO_SCIPY_VERSION:
        raise PostGenerationRankingError(
            "runtime SciPy differs from the frozen common-scenario contract"
        )
    generated = deterministic_sobol(
        dimension=SCENARIO_DIMENSION,
        count=SCENARIO_COUNT,
        seed=SCENARIO_SOBOL_SEED,
    )
    digest = hashlib.sha256(
        np.asarray(generated, dtype=">f8").tobytes(order="C")
    ).hexdigest()
    if digest != SCENARIO_DESIGN_SHA256:
        raise PostGenerationRankingError(
            "runtime common Sobol design differs from its frozen SHA-256"
        )
    native_bytes = np.asarray(generated, dtype=np.float64).tobytes(order="C")
    immutable_array = np.frombuffer(native_bytes, dtype=np.float64).reshape(
        SCENARIO_COUNT, SCENARIO_DIMENSION
    )
    if immutable_array.flags.writeable:
        raise PostGenerationRankingError(
            "common scenario design must be bytes-backed and read-only"
        )
    rows = tuple(
        tuple(float(value) for value in row) for row in immutable_array
    )
    return (
        CommonScenarioDesign(
            method_id=SCENARIO_METHOD_ID,
            scipy_version=scipy.__version__,
            dimension=SCENARIO_DIMENSION,
            count=SCENARIO_COUNT,
            sobol_seed=SCENARIO_SOBOL_SEED,
            scramble=True,
            optimization=None,
            identity_encoding="BIG_ENDIAN_BINARY64_ROW_MAJOR",
            design_sha256=digest,
            parameters_unit=rows,
        ),
        immutable_array,
    )


class PostGenerationRankOnlyPipeline:
    """Evaluate a frozen top-level result without any candidate search API."""

    def __init__(
        self,
        *,
        expected_generation_contract_sha256: str,
        expected_model_contract_sha256: str,
        wrench_evaluator: TaskWrenchEvaluator,
        hand_model: Any,
        collision_certifier: Callable[[StaticV9AcceptedCandidate], Any],
        policy_collision_certifier: (
            Callable[[StaticV9AcceptedPolicy], Any] | None
        ) = None,
        lower_tail_fraction: float = 0.10,
    ) -> None:
        if not _valid_sha256(expected_generation_contract_sha256):
            raise PostGenerationRankingError(
                "expected generation contract SHA-256 is invalid"
            )
        if not _valid_sha256(expected_model_contract_sha256):
            raise PostGenerationRankingError(
                "expected model contract SHA-256 is invalid"
            )
        if type(wrench_evaluator) is not TaskWrenchEvaluator:
            raise PostGenerationRankingError(
                "formal bridge requires the exact TaskWrenchEvaluator type"
            )
        if wrench_evaluator.uncertainty_dimension != SCENARIO_DIMENSION:
            raise PostGenerationRankingError(
                "task-wrench uncertainty dimension differs from "
                "scenario design"
            )
        if wrench_evaluator.uncertainty_claim_scope != (
            FRICTION_INTERVAL_ONLY_CERTIFIED_UNCERTAINTY_SCOPE
        ):
            raise PostGenerationRankingError(
                "current bridge only recognizes explicit friction-only scope"
            )
        if not callable(collision_certifier):
            raise PostGenerationRankingError(
                "collision_certifier must be one explicit callable"
            )
        if (
            policy_collision_certifier is not None
            and not callable(policy_collision_certifier)
        ):
            raise PostGenerationRankingError(
                "policy_collision_certifier must be callable or None"
            )
        tail = float(lower_tail_fraction)
        if not math.isfinite(tail) or not 0.0 < tail <= 1.0:
            raise PostGenerationRankingError(
                "lower_tail_fraction must lie in (0, 1]"
            )
        self.expected_generation_contract_sha256 = (
            expected_generation_contract_sha256
        )
        self.expected_model_contract_sha256 = expected_model_contract_sha256
        self.wrench_evaluator = wrench_evaluator
        self.hand_model = hand_model
        self.collision_certifier = collision_certifier
        self.policy_collision_certifier = policy_collision_certifier
        self.lower_tail_fraction = tail
        self.common_scenario_design, self._scenario_array = (
            _build_common_scenario_design()
        )
        contract = {
            "method_id": METHOD_ID,
            "generation_contract_sha256": (
                expected_generation_contract_sha256
            ),
            "model_contract_sha256": expected_model_contract_sha256,
            "scenario_design": self.common_scenario_design,
            "selection_order": SELECTION_ORDER,
            "tie_break_rule": TIE_BREAK_RULE,
            "lower_tail_fraction": tail,
            "generation_allowed": False,
            "refinement_allowed": False,
            "replacement_sampling_allowed": False,
            "retry_allowed": False,
            "failed_candidate_drop_allowed": False,
            "collision_invocations_per_unique_accepted_candidate": 1,
            "wrench_invocations_per_unique_accepted_candidate": 1,
            "contact_range_policy_consumption": POLICY_AWARE_RANKING_STATUS,
            "collision_invocations_per_unique_accepted_policy": 1,
            "wrench_invocations_after_valid_policy_collision_binding": 1,
            "display_midpoint_substitution_allowed": False,
            "formal_uncertainty_scope_status": "INCOMPLETE_FRICTION_ONLY",
            "formal_selection_allowed": False,
        }
        self.rank_contract_sha256 = _sha256(contract)

    def _validate_accepted(
        self,
        accepted: StaticV9AcceptedCandidate,
        unique: UniqueV9Evaluation,
    ) -> None:
        if type(accepted) is not StaticV9AcceptedCandidate:
            raise PostGenerationRankingError(
                "accepted rows must be exact StaticV9AcceptedCandidate values"
            )
        candidate = accepted.candidate
        audit = accepted.v9_audit
        binding = accepted.invocation_binding
        if (
            type(candidate) is not GraspCandidate
            or type(audit) is not RayClosureAudit
            or type(binding) is not V9InvocationAuditBinding
        ):
            raise PostGenerationRankingError(
                "accepted row lacks exact candidate, Ray audit, or invocation"
            )
        if (
            audit.method_id != RAY_CLOSURE_METHOD_ID
            or audit.closure_parameter_domain_id
            != CLOSURE_PARAMETER_DOMAIN_ID
            or audit.failure_reason is not None
            or not audit.model_binding_complete
            or audit.model_binding_status != MODEL_BINDING_COMPLETE_STATUS
            or audit.model_contract_sha256
            != self.expected_model_contract_sha256
            or audit.object_geometry_sha256
            != self.wrench_evaluator.object_model.geometry_sha256
        ):
            raise PostGenerationRankingError(
                "accepted V9 audit is not bound to the expected real model"
            )
        if (
            binding.raw_v9_audit is not audit
            or binding.method_id != RAY_CLOSURE_METHOD_ID
            or binding.parameter_domain_id != CLOSURE_PARAMETER_DOMAIN_ID
            or tuple(binding.parameter_layout) != tuple(audit.parameter_layout)
            or tuple(binding.requested_parameters_unit)
            != tuple(accepted.v9_parameters_unit)
            or binding.requested_parameter_key_hex
            != accepted.v9_parameter_key_hex
        ):
            raise PostGenerationRankingError(
                "accepted invocation does not bind the exact V9 request"
            )
        canonical = canonicalize_v9_parameters(
            accepted.v9_parameters_unit,
            parameter_layout=audit.parameter_layout,
        )
        if canonical.exact_key_hex != accepted.v9_parameter_key_hex:
            raise PostGenerationRankingError(
                "accepted V9 key differs from canonical parameters"
            )
        if tuple(
            contact.pad_name for contact in candidate.planned_pad_contacts
        ) != tuple(audit.pad_order):
            raise PostGenerationRankingError(
                "accepted candidate PAD order differs from its Ray audit"
            )
        lineage_indices = tuple(
            row.attempt_index for row in accepted.lineage
        )
        if (
            not lineage_indices
            or lineage_indices != tuple(
                row.attempt_index for row in unique.lineage
            )
            or min(lineage_indices) != unique.first_attempt_index
            or accepted.lineage != unique.lineage
            or accepted.candidate is not unique.candidate
            or accepted.v9_audit is not unique.v9_audit
            or accepted.invocation_binding is not unique.invocation_binding
        ):
            raise PostGenerationRankingError(
                "accepted candidate lineage differs from its unique V9 row"
            )
        try:
            manifest = json.loads(audit.model_contract_canonical_json)
            manifest_joint_names = tuple(
                manifest["hand"]["independent_joint_names"]
            )
            manifest_pad_names = tuple(
                row["name"] for row in manifest["verified_pads"]
            )
            supplied_joint_names = tuple(
                self.hand_model.independent_joint_names
            )
            supplied_pad_names = tuple(self.hand_model.pads)
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            raise PostGenerationRankingError(
                "hand/model manifest binding is incomplete"
            ) from error
        if (
            supplied_joint_names != manifest_joint_names
            or set(supplied_pad_names) != set(manifest_pad_names)
            or len(candidate.independent_joint_positions_rad)
            != len(manifest_joint_names)
        ):
            raise PostGenerationRankingError(
                "task-wrench hand differs from the V9 model contract"
            )

    def _validate_accepted_policy(
        self,
        accepted: StaticV9AcceptedPolicy,
        unique: UniqueV9Evaluation,
    ) -> None:
        if type(accepted) is not StaticV9AcceptedPolicy:
            raise PostGenerationRankingError(
                "accepted policy rows must be exact StaticV9AcceptedPolicy values"
            )
        policy = accepted.sequential_closure_policy
        audit = accepted.v9_audit
        binding = accepted.invocation_binding
        if (
            type(policy) is not CertifiedSequentialClosurePolicy
            or type(audit) is not RayClosureAudit
            or type(binding) is not V9InvocationAuditBinding
        ):
            raise PostGenerationRankingError(
                "accepted policy lacks exact closure policy, Ray audit, or invocation"
            )
        if (
            audit.method_id != RAY_CLOSURE_METHOD_ID
            or audit.closure_parameter_domain_id
            != CLOSURE_PARAMETER_DOMAIN_ID
            or audit.failure_reason != REPRESENTATIVE_PROPOSAL_FAILURE_REASON
            or audit.candidate_role != CANDIDATE_REPRESENTATIVE_ROLE
            or audit.candidate_exact_contact_endpoint_certified
            or not audit.model_binding_complete
            or audit.model_binding_status != MODEL_BINDING_COMPLETE_STATUS
            or audit.model_contract_sha256
            != self.expected_model_contract_sha256
            or audit.object_geometry_sha256
            != self.wrench_evaluator.object_model.geometry_sha256
        ):
            raise PostGenerationRankingError(
                "accepted contact-range policy audit is not bound to the expected model"
            )
        if (
            binding.raw_v9_audit is not audit
            or binding.method_id != RAY_CLOSURE_METHOD_ID
            or binding.parameter_domain_id != CLOSURE_PARAMETER_DOMAIN_ID
            or tuple(binding.parameter_layout) != tuple(audit.parameter_layout)
            or tuple(binding.requested_parameters_unit)
            != tuple(accepted.v9_parameters_unit)
            or binding.requested_parameter_key_hex
            != accepted.v9_parameter_key_hex
        ):
            raise PostGenerationRankingError(
                "accepted policy invocation does not bind the exact V9 request"
            )
        canonical = canonicalize_v9_parameters(
            accepted.v9_parameters_unit,
            parameter_layout=audit.parameter_layout,
        )
        expected_contact_set_hashes = tuple(
            row.set_sha256 for row in policy.possible_first_contact_sets
        )
        if (
            canonical.exact_key_hex != accepted.v9_parameter_key_hex
            or policy.pad_order != audit.pad_order
            or policy.independent_actuation_supports
            != audit.independent_actuation_supports
            or policy.closing_directions_physical
            != audit.closing_directions_physical
            or policy.object_geometry_sha256 != audit.object_geometry_sha256
            or policy.model_contract_sha256 != audit.model_contract_sha256
            or expected_contact_set_hashes
            != audit.possible_first_contact_set_sha256
        ):
            raise PostGenerationRankingError(
                "accepted contact-range policy differs from its Ray audit"
            )
        lineage_indices = tuple(row.attempt_index for row in accepted.lineage)
        if (
            not lineage_indices
            or lineage_indices
            != tuple(row.attempt_index for row in unique.lineage)
            or min(lineage_indices) != unique.first_attempt_index
            or accepted.lineage != unique.lineage
            or accepted.sequential_closure_policy
            is not unique.sequential_closure_policy
            or accepted.v9_audit is not unique.v9_audit
            or accepted.invocation_binding is not unique.invocation_binding
        ):
            raise PostGenerationRankingError(
                "accepted contact-range policy lineage differs from its unique V9 row"
            )
        try:
            manifest = json.loads(audit.model_contract_canonical_json)
            manifest_joint_names = tuple(
                manifest["hand"]["independent_joint_names"]
            )
            manifest_pad_names = tuple(
                row["name"] for row in manifest["verified_pads"]
            )
            supplied_joint_names = tuple(self.hand_model.independent_joint_names)
            supplied_pad_names = tuple(self.hand_model.pads)
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            raise PostGenerationRankingError(
                "policy hand/model manifest binding is incomplete"
            ) from error
        if (
            supplied_joint_names != manifest_joint_names
            or set(supplied_pad_names) != set(manifest_pad_names)
            or policy.independent_joint_names != manifest_joint_names
        ):
            raise PostGenerationRankingError(
                "policy task-wrench hand differs from the V9 model contract"
            )

    def _validate_generation(
        self,
        result: TopLevelGenerationResult,
    ) -> tuple[
        tuple[GenerationAttemptSnapshot, ...],
        tuple[tuple[StaticV9AcceptedCandidate, UniqueV9Evaluation], ...],
        tuple[tuple[StaticV9AcceptedPolicy, UniqueV9Evaluation], ...],
        str,
    ]:
        if type(result) is not TopLevelGenerationResult:
            raise PostGenerationRankingError(
                "rank-only input must be exact TopLevelGenerationResult"
            )
        if (
            result.method_id != TOP_LEVEL_GENERATOR_METHOD_ID
            or result.contract_hash_sha256
            != self.expected_generation_contract_sha256
            or result.total_attempt_budget not in ALLOWED_TOTAL_ATTEMPT_BUDGETS
            or result.attempts_per_lane * 4 != result.total_attempt_budget
            or result.local_refinement_evaluation_budget != 0
            or len(result.attempts) != result.total_attempt_budget
        ):
            raise PostGenerationRankingError(
                "top-level generation result differs from the frozen contract"
            )
        if result.accepted_policies and self.policy_collision_certifier is None:
            raise PostGenerationRankingError(
                "POLICY_COLLISION_CERTIFIER_REQUIRED_FOR_ACCEPTED_POLICIES"
            )
        if any(
            type(row) is not CandidateAttemptAudit
            or row.attempt_index != index
            for index, row in enumerate(result.attempts)
        ):
            raise PostGenerationRankingError(
                "generation attempts are not complete contiguous lineage"
            )
        snapshots = tuple(_attempt_snapshot(row) for row in result.attempts)
        duplicate_count = sum(
            row.status is AttemptStatus.DUPLICATE_CANONICAL_V9_PARAMETERS
            for row in result.attempts
        )
        proposal_failure_count = sum(
            row.status
            in {
                AttemptStatus.PROPOSAL_REJECTED,
                AttemptStatus.PROPOSAL_PROTOCOL_REJECTED,
                AttemptStatus.PROPOSAL_V9_DOMAIN_REJECTED,
            }
            for row in result.attempts
        )
        if (
            duplicate_count != result.duplicate_attempt_count
            or proposal_failure_count != result.proposal_failure_count
            or len(result.unique_v9_evaluations)
            != result.v9_evaluation_count
        ):
            raise PostGenerationRankingError(
                "generation summary counters contradict attempt lineage"
            )
        unique_by_key: dict[str, UniqueV9Evaluation] = {}
        for row in result.unique_v9_evaluations:
            if type(row) is not UniqueV9Evaluation:
                raise PostGenerationRankingError(
                    "unique V9 rows have an unexpected type"
                )
            if row.v9_parameter_key_hex in unique_by_key:
                raise PostGenerationRankingError(
                    "unique V9 parameter key is duplicated"
                )
            canonical = canonicalize_v9_parameters(
                row.v9_parameters_unit,
                parameter_layout=(
                    row.v9_audit.parameter_layout
                    if type(row.v9_audit) is RayClosureAudit
                    else (
                        "assembly_axis_yaw_unit",
                        "axial_target_unit",
                        "lateral_task_x_unit",
                        "lateral_task_y_unit",
                        "preshape_joint_unit:unbound",
                    )
                ),
            )
            if canonical.exact_key_hex != row.v9_parameter_key_hex:
                raise PostGenerationRankingError(
                    "unique V9 row has a noncanonical parameter key"
                )
            lineage_indices = tuple(item.attempt_index for item in row.lineage)
            if (
                not lineage_indices
                or min(lineage_indices) != row.first_attempt_index
                or any(index < 0 or index >= len(result.attempts)
                       for index in lineage_indices)
            ):
                raise PostGenerationRankingError(
                    "unique V9 row has invalid attempt lineage"
                )
            unique_by_key[row.v9_parameter_key_hex] = row
        expected_accepted = tuple(
            row
            for row in result.unique_v9_evaluations
            if row.status is AttemptStatus.STATIC_V9_ACCEPTED
        )
        if len(expected_accepted) != len(result.accepted_candidates):
            raise PostGenerationRankingError(
                "accepted candidates do not cover every static V9 acceptance"
            )
        bound_rows: list[
            tuple[StaticV9AcceptedCandidate, UniqueV9Evaluation]
        ] = []
        for accepted, unique in zip(
            result.accepted_candidates, expected_accepted
        ):
            if accepted.v9_parameter_key_hex != unique.v9_parameter_key_hex:
                raise PostGenerationRankingError(
                    "accepted candidate order differs from unique V9 order"
                )
            self._validate_accepted(accepted, unique)
            bound_rows.append((accepted, unique))
        expected_policies = tuple(
            row
            for row in result.unique_v9_evaluations
            if row.status is AttemptStatus.STATIC_V9_POLICY_ACCEPTED
        )
        if len(expected_policies) != len(result.accepted_policies):
            raise PostGenerationRankingError(
                "accepted policies do not cover every static V9 policy acceptance"
            )
        bound_policy_rows: list[
            tuple[StaticV9AcceptedPolicy, UniqueV9Evaluation]
        ] = []
        for accepted, unique in zip(result.accepted_policies, expected_policies):
            if accepted.v9_parameter_key_hex != unique.v9_parameter_key_hex:
                raise PostGenerationRankingError(
                    "accepted policy order differs from unique V9 order"
                )
            self._validate_accepted_policy(accepted, unique)
            bound_policy_rows.append((accepted, unique))
        result_document = {
            "method_id": result.method_id,
            "contract_hash_sha256": result.contract_hash_sha256,
            "total_attempt_budget": result.total_attempt_budget,
            "attempts_per_lane": result.attempts_per_lane,
            "local_refinement_evaluation_budget": (
                result.local_refinement_evaluation_budget
            ),
            "attempts": snapshots,
            "unique_v9_evaluations": [
                {
                    "v9_parameters_unit": row.v9_parameters_unit,
                    "v9_parameter_key_hex": row.v9_parameter_key_hex,
                    "first_attempt_index": row.first_attempt_index,
                    "lineage_attempt_indices": tuple(
                        item.attempt_index for item in row.lineage
                    ),
                    "candidate_sha256": (
                        candidate_sha256(row.candidate)
                        if type(row.candidate) is GraspCandidate
                        else None
                    ),
                    "contact_range_policy_sha256": (
                        row.sequential_closure_policy.policy_sha256
                        if type(row.sequential_closure_policy)
                        is CertifiedSequentialClosurePolicy
                        else None
                    ),
                    "v9_audit_sha256": _audit_sha256(row.v9_audit),
                    "status": row.status.value,
                    "v9_failure_reason": row.v9_failure_reason,
                }
                for row in result.unique_v9_evaluations
            ],
            "accepted_candidate_keys": tuple(
                row.v9_parameter_key_hex
                for row in result.accepted_candidates
            ),
            "accepted_policy_keys": tuple(
                row.v9_parameter_key_hex for row in result.accepted_policies
            ),
            "v9_evaluation_count": result.v9_evaluation_count,
            "duplicate_attempt_count": result.duplicate_attempt_count,
            "proposal_failure_count": result.proposal_failure_count,
        }
        return (
            snapshots,
            tuple(bound_rows),
            tuple(bound_policy_rows),
            _sha256(result_document),
        )

    def _collision_evidence(
        self,
        accepted: StaticV9AcceptedCandidate,
        expected_candidate_sha256: str,
        expected_v9_evidence_sha256: str,
    ) -> tuple[str, str | None, float | None, tuple[str, ...]]:
        try:
            certificate = self.collision_certifier(accepted)
        except Exception as error:  # exactly one call; never retried
            return (
                "COLLISION_CERTIFIER_EXCEPTION",
                None,
                None,
                (
                    "COLLISION_CERTIFIER_EXCEPTION:"
                    f"{type(error).__name__}:{error}",
                ),
            )
        if type(certificate) is FullHandClosureCollisionCertificate:
            audit = certificate.audit
            if (
                audit.method_id != FULL_HAND_COLLISION_METHOD_ID
                or audit.v9_evidence_sha256
                != expected_v9_evidence_sha256
            ):
                return (
                    "FULL_HAND_V1_PROTOCOL_REJECTED",
                    None,
                    None,
                    ("FULL_HAND_V1_V9_OR_METHOD_BINDING_MISMATCH",),
                )
            blockers = tuple(audit.blockers)
            if certificate.state is FullHandClosureCollisionState.CERTIFIED:
                blockers = blockers + (
                    (
                        "FULL_HAND_V1_HAS_NO_QUANTITATIVE_COMPLETE_"
                        "CLEARANCE_BRIDGE"
                    ),
                )
            return (
                certificate.state.value,
                _sha256(certificate),
                None,
                tuple(sorted(set(blockers))),
            )
        if type(certificate) is CompleteTrajectoryCollisionCertificate:
            if (
                certificate.candidate_sha256
                != expected_candidate_sha256
                or certificate.v9_evidence_sha256
                != expected_v9_evidence_sha256
                or certificate.model_contract_sha256
                != self.expected_model_contract_sha256
            ):
                return (
                    "COMPLETE_CLEARANCE_PROTOCOL_REJECTED",
                    None,
                    None,
                    (
                        "COMPLETE_CLEARANCE_CANDIDATE_OR_MODEL_"
                        "BINDING_MISMATCH",
                    ),
                )
            return (
                "COMPLETE_CLEARANCE_CERTIFIED",
                certificate.certificate_sha256,
                certificate.trajectory_clearance_lower_bound_m,
                (),
            )
        return (
            "COLLISION_CERTIFICATE_TYPE_REJECTED",
            None,
            None,
            (
                "COLLISION_CERTIFICATE_MUST_BE_EXACT_FULL_HAND_V1_OR_"
                "COMPLETE_TYPED_CLEARANCE",
            ),
        )

    def _wrench_evidence(
        self,
        candidate: GraspCandidate,
    ) -> tuple[
        TaskWrenchOnlyEvaluation | None,
        DiagnosticWrenchSummary | None,
        str | None,
        tuple[str, ...],
    ]:
        try:
            evaluation = self.wrench_evaluator.evaluate_task_wrench(
                candidate,
                self._scenario_array,
                hand_model=self.hand_model,
            )
        except Exception as error:  # exactly one call; never retried
            return (
                None,
                None,
                None,
                (
                    "TASK_WRENCH_EVALUATOR_EXCEPTION:"
                    f"{type(error).__name__}:{error}",
                ),
            )
        if type(evaluation) is not TaskWrenchOnlyEvaluation:
            return (
                None,
                None,
                None,
                ("TASK_WRENCH_EVALUATOR_RETURNED_UNEXPECTED_TYPE",),
            )
        if len(evaluation.task_margins) != SCENARIO_COUNT:
            return (
                None,
                None,
                None,
                ("TASK_WRENCH_SCENARIO_COUNT_MISMATCH",),
            )
        scope = evaluation.diagnostics.get("certified_uncertainty_scope")
        if scope != FRICTION_INTERVAL_ONLY_CERTIFIED_UNCERTAINTY_SCOPE:
            return (
                None,
                None,
                None,
                ("TASK_WRENCH_UNCERTAINTY_SCOPE_MISMATCH",),
            )
        summary = DiagnosticWrenchSummary(
            hard_bound_minimum_task_margin=(
                evaluation.hard_bound_minimum_task_margin
            ),
            qmc_lower_tail_mean_task_margin=qmc_lower_tail_mean(
                evaluation.task_margins,
                self.lower_tail_fraction,
            ),
            peak_normal_force_n=evaluation.peak_normal_force_n,
            joint_torque_utilization=evaluation.joint_torque_utilization,
        )
        return evaluation, summary, _sha256(evaluation), ()

    def _policy_collision_evidence(
        self,
        accepted: StaticV9AcceptedPolicy,
    ) -> tuple[
        ContactRangePolicyCollisionCertificate | None,
        str | None,
        str,
        str | None,
        float | None,
        tuple[str, ...],
    ]:
        certifier = self.policy_collision_certifier
        if certifier is None:  # guarded before any evaluation
            raise AssertionError("policy collision certifier disappeared")
        try:
            certificate = certifier(accepted)
        except Exception as error:  # exactly one call; never retried
            return (
                None,
                None,
                "POLICY_COLLISION_CERTIFIER_EXCEPTION",
                None,
                None,
                (
                    "POLICY_COLLISION_CERTIFIER_EXCEPTION:"
                    f"{type(error).__name__}:{error}",
                ),
            )
        clearance: float | None = None
        if type(certificate) is ContactRangePolicyCollisionCertificate:
            source = certificate
            collision_state = certificate.state.value
            collision_sha = _sha256(certificate)
            blockers = tuple(certificate.audit.blockers)
        elif type(certificate) is (
            CompleteContactRangeTrajectoryCollisionCertificate
        ):
            source = certificate.source_policy_collision_certificate
            collision_state = "COMPLETE_POLICY_CLEARANCE_CERTIFIED"
            collision_sha = certificate.certificate_sha256
            clearance = certificate.trajectory_clearance_lower_bound_m
            blockers = ()
        else:
            return (
                None,
                None,
                "POLICY_COLLISION_CERTIFICATE_TYPE_REJECTED",
                None,
                None,
                (
                    "POLICY_COLLISION_CERTIFICATE_MUST_BE_EXACT_RANGE_OR_"
                    "COMPLETE_TYPED_CLEARANCE",
                ),
            )
        audit = source.audit
        policy = accepted.sequential_closure_policy
        v9_audit = accepted.v9_audit
        if (
            audit.method_id != CONTACT_RANGE_POLICY_METHOD_ID
            or not audit.policy_contact_ranges_consumed
            or audit.display_approximation_used_as_formal_evidence
            or not audit.checkable_collision_gates_passed
            or audit.policy_sha256 != policy.policy_sha256
            or audit.ray_closure_object_geometry_sha256
            != v9_audit.object_geometry_sha256
            or audit.ray_model_contract_sha256
            != self.expected_model_contract_sha256
            or type(certificate)
            is CompleteContactRangeTrajectoryCollisionCertificate
            and (
                certificate.policy_sha256 != policy.policy_sha256
                or certificate.v9_policy_evidence_sha256
                != audit.v9_audit_and_policy_sha256
                or certificate.model_contract_sha256
                != self.expected_model_contract_sha256
            )
        ):
            return (
                None,
                None,
                "POLICY_COLLISION_PROTOCOL_REJECTED",
                None,
                None,
                ("POLICY_COLLISION_OR_MODEL_BINDING_MISMATCH",),
            )
        return (
            source,
            audit.v9_audit_and_policy_sha256,
            collision_state,
            collision_sha,
            clearance,
            tuple(sorted(set(blockers))),
        )

    def _policy_wrench_evidence(
        self,
        accepted: StaticV9AcceptedPolicy,
        policy_collision_certificate: (
            ContactRangePolicyCollisionCertificate | None
        ),
    ) -> tuple[
        ContactRangePolicyWrenchCertificate | None,
        DiagnosticWrenchSummary | None,
        str | None,
        tuple[str, ...],
        int,
    ]:
        if policy_collision_certificate is None:
            return (
                None,
                None,
                None,
                ("POLICY_WRENCH_SKIPPED_NO_VALID_COLLISION_BINDING",),
                0,
            )
        try:
            certificate = self.wrench_evaluator.evaluate_contact_range_policy(
                accepted.sequential_closure_policy,
                self._scenario_array,
                v9_audit=accepted.v9_audit,
                hand_model=self.hand_model,
                policy_collision_certificate=policy_collision_certificate,
            )
        except Exception as error:  # exactly one call; never retried
            return (
                None,
                None,
                None,
                (
                    "POLICY_WRENCH_EVALUATOR_EXCEPTION:"
                    f"{type(error).__name__}:{error}",
                ),
                1,
            )
        if type(certificate) is not ContactRangePolicyWrenchCertificate:
            return (
                None,
                None,
                None,
                ("POLICY_WRENCH_EVALUATOR_RETURNED_UNEXPECTED_TYPE",),
                1,
            )
        audit = certificate.audit
        if (
            audit.policy_sha256
            != accepted.sequential_closure_policy.policy_sha256
            or audit.model_contract_sha256
            != self.expected_model_contract_sha256
            or audit.scenario_count != SCENARIO_COUNT
            or audit.scenario_dimension != SCENARIO_DIMENSION
            or audit.scenario_design_sha256 != SCENARIO_DESIGN_SHA256
            or audit.uncertainty_claim_scope
            != FRICTION_INTERVAL_ONLY_CERTIFIED_UNCERTAINTY_SCOPE
            or audit.display_approximation_used_as_formal_evidence
            or audit.finite_contact_geometry_sampling_used_as_formal_evidence
            or audit.exact_candidate_wrench_invocation_count != 0
        ):
            return (
                None,
                None,
                None,
                ("POLICY_WRENCH_OR_SCENARIO_BINDING_MISMATCH",),
                1,
            )
        summary: DiagnosticWrenchSummary | None = None
        if certificate.state is (
            ContactRangePolicyWrenchState.PARAMETRIC_WRENCH_CERTIFIED_NONFRICTION_UNCALIBRATED
        ):
            if certificate.task_margins is None:
                raise AssertionError("certified policy wrench lost its margins")
            summary = DiagnosticWrenchSummary(
                hard_bound_minimum_task_margin=float(
                    certificate.hard_bound_minimum_task_margin
                ),
                qmc_lower_tail_mean_task_margin=qmc_lower_tail_mean(
                    certificate.task_margins,
                    self.lower_tail_fraction,
                ),
                peak_normal_force_n=float(certificate.peak_normal_force_n),
                joint_torque_utilization=float(
                    certificate.joint_torque_utilization
                ),
            )
        return (
            certificate,
            summary,
            _sha256(certificate),
            tuple(sorted(set(audit.blockers))),
            1,
        )

    @staticmethod
    def _diagnostic_rank(
        records: Sequence[RankCandidateRecord | RankContactRangePolicyRecord],
    ) -> tuple[
        tuple[RankCandidateRecord | RankContactRangePolicyRecord, ...],
        tuple[str, ...],
    ]:
        rows = tuple(records)
        eligible = tuple(
            index
            for index, row in enumerate(rows)
            if row.diagnostic_metrics is not None
        )
        if not eligible:
            return rows, ()
        metrics = tuple(rows[index].diagnostic_metrics for index in eligible)
        if any(value is None for value in metrics):  # pragma: no cover
            raise AssertionError("diagnostic eligibility lost its metrics")
        concrete = tuple(value for value in metrics if value is not None)
        layers = pareto_layers(concrete)
        layer_by_position = {
            position: layer_number
            for layer_number, layer in enumerate(layers, start=1)
            for position in layer
        }
        order = sorted(
            range(len(eligible)),
            key=lambda position: (
                concrete[position].lexicographic_key(),
                rows[eligible[position]].first_attempt_index,
                rows[eligible[position]].v9_parameter_key_hex,
            ),
        )
        rank_by_position = {
            position: rank for rank, position in enumerate(order, start=1)
        }
        mutable = list(rows)
        for position, record_index in enumerate(eligible):
            mutable[record_index] = replace(
                mutable[record_index],
                diagnostic_rank=rank_by_position[position],
                pareto_layer=layer_by_position[position],
            )
        ranked_keys = tuple(
            rows[eligible[position]].v9_parameter_key_hex for position in order
        )
        return tuple(mutable), ranked_keys

    def evaluate(
        self,
        generation_result: TopLevelGenerationResult,
    ) -> PostGenerationRankResult:
        """Evaluate each exact candidate or contact-range policy once."""

        (
            snapshots,
            accepted_rows,
            accepted_policy_rows,
            generation_sha256,
        ) = self._validate_generation(generation_result)
        records: list[RankCandidateRecord] = []
        for evaluation_index, (accepted, unique) in enumerate(accepted_rows):
            candidate = accepted.candidate
            audit = accepted.v9_audit
            candidate_digest = candidate_sha256(candidate)
            v9_digest = v9_evidence_sha256(candidate, audit)
            collision_state, collision_sha, clearance, collision_blockers = (
                self._collision_evidence(
                    accepted,
                    candidate_digest,
                    v9_digest,
                )
            )
            wrench, wrench_summary, wrench_sha, wrench_blockers = (
                self._wrench_evidence(candidate)
            )
            blockers = list(collision_blockers + wrench_blockers)
            blockers.append(FORMAL_UNCERTAINTY_BLOCKER)
            diagnostic_metrics: CandidateMetrics | None = None
            if wrench_summary is not None and clearance is not None:
                diagnostic_metrics = CandidateMetrics(
                    hard_bound_minimum_task_margin=(
                        wrench_summary.hard_bound_minimum_task_margin
                    ),
                    qmc_lower_tail_mean_task_margin=(
                        wrench_summary.qmc_lower_tail_mean_task_margin
                    ),
                    peak_normal_force_n=(
                        wrench_summary.peak_normal_force_n
                    ),
                    joint_torque_utilization=(
                        wrench_summary.joint_torque_utilization
                    ),
                    trajectory_clearance_m=clearance,
                )
            if wrench is None:
                state = CandidateEvaluationState.UNRESOLVED_WRENCH
            elif clearance is None:
                state = CandidateEvaluationState.UNRESOLVED_COLLISION
            else:
                state = CandidateEvaluationState.UNCERTAINTY_SCOPE_INCOMPLETE
            lineage_indices = tuple(
                row.attempt_index for row in accepted.lineage
            )
            records.append(
                RankCandidateRecord(
                    evaluation_index=evaluation_index,
                    v9_parameter_key_hex=accepted.v9_parameter_key_hex,
                    v9_parameters_unit=tuple(
                        accepted.v9_parameters_unit
                    ),
                    first_attempt_index=unique.first_attempt_index,
                    lineage_attempt_indices=lineage_indices,
                    generation_lineage_sha256=_sha256(
                        {
                            "v9_parameter_key_hex": (
                                accepted.v9_parameter_key_hex
                            ),
                            "attempt_indices": lineage_indices,
                            "lineage": accepted.lineage,
                        }
                    ),
                    candidate=candidate,
                    candidate_sha256=candidate_digest,
                    v9_audit_sha256=_sha256(audit.as_dict()),
                    v9_evidence_sha256=v9_digest,
                    v9_model_contract_sha256=(
                        audit.model_contract_sha256
                    ),
                    common_scenario_design_sha256=(
                        self.common_scenario_design.design_sha256
                    ),
                    collision_invocation_count=1,
                    wrench_invocation_count=1,
                    collision_certificate_sha256=collision_sha,
                    collision_state=collision_state,
                    trajectory_clearance_lower_bound_m=clearance,
                    wrench_evaluation_sha256=wrench_sha,
                    certified_uncertainty_scope=(
                        None
                        if wrench is None
                        else str(
                            wrench.diagnostics[
                                "certified_uncertainty_scope"
                            ]
                        )
                    ),
                    diagnostic_wrench=wrench_summary,
                    diagnostic_metrics=diagnostic_metrics,
                    formal_metrics=None,
                    state=state,
                    blockers=tuple(sorted(set(blockers))),
                )
            )
        ranked_records, diagnostic_keys = self._diagnostic_rank(records)
        policy_records: list[RankContactRangePolicyRecord] = []
        for policy_index, (accepted, unique) in enumerate(
            accepted_policy_rows
        ):
            policy = accepted.sequential_closure_policy
            audit = accepted.v9_audit
            policy_digest = policy.policy_sha256
            (
                source_collision,
                v9_policy_digest,
                collision_state,
                collision_sha,
                clearance,
                collision_blockers,
            ) = self._policy_collision_evidence(accepted)
            (
                wrench,
                wrench_summary,
                wrench_sha,
                wrench_blockers,
                wrench_invocation_count,
            ) = self._policy_wrench_evidence(
                accepted,
                source_collision,
            )
            blockers = list(collision_blockers + wrench_blockers)
            blockers.append(FORMAL_UNCERTAINTY_BLOCKER)
            diagnostic_metrics: CandidateMetrics | None = None
            if wrench_summary is not None and clearance is not None:
                diagnostic_metrics = CandidateMetrics(
                    hard_bound_minimum_task_margin=(
                        wrench_summary.hard_bound_minimum_task_margin
                    ),
                    qmc_lower_tail_mean_task_margin=(
                        wrench_summary.qmc_lower_tail_mean_task_margin
                    ),
                    peak_normal_force_n=wrench_summary.peak_normal_force_n,
                    joint_torque_utilization=(
                        wrench_summary.joint_torque_utilization
                    ),
                    trajectory_clearance_m=clearance,
                )
            if wrench_summary is None:
                state = CandidateEvaluationState.UNRESOLVED_WRENCH
            elif clearance is None:
                state = CandidateEvaluationState.UNRESOLVED_COLLISION
            else:
                state = CandidateEvaluationState.UNCERTAINTY_SCOPE_INCOMPLETE
            lineage_indices = tuple(
                row.attempt_index for row in accepted.lineage
            )
            policy_records.append(
                RankContactRangePolicyRecord(
                    evaluation_index=len(records) + policy_index,
                    v9_parameter_key_hex=accepted.v9_parameter_key_hex,
                    v9_parameters_unit=tuple(accepted.v9_parameters_unit),
                    first_attempt_index=unique.first_attempt_index,
                    lineage_attempt_indices=lineage_indices,
                    generation_lineage_sha256=_sha256(
                        {
                            "v9_parameter_key_hex": (
                                accepted.v9_parameter_key_hex
                            ),
                            "attempt_indices": lineage_indices,
                            "lineage": accepted.lineage,
                            "contact_range_policy_sha256": policy_digest,
                        }
                    ),
                    sequential_closure_policy=policy,
                    policy_sha256=policy_digest,
                    v9_audit_sha256=_sha256(audit.as_dict()),
                    v9_policy_evidence_sha256=v9_policy_digest,
                    v9_model_contract_sha256=audit.model_contract_sha256,
                    common_scenario_design_sha256=(
                        self.common_scenario_design.design_sha256
                    ),
                    collision_invocation_count=1,
                    wrench_invocation_count=wrench_invocation_count,
                    collision_certificate_sha256=collision_sha,
                    collision_state=collision_state,
                    trajectory_clearance_lower_bound_m=clearance,
                    wrench_evaluation_sha256=wrench_sha,
                    wrench_state=(
                        None if wrench is None else wrench.state.value
                    ),
                    certified_uncertainty_scope=(
                        None
                        if wrench is None
                        else wrench.audit.uncertainty_claim_scope
                    ),
                    diagnostic_wrench=wrench_summary,
                    diagnostic_metrics=diagnostic_metrics,
                    formal_metrics=None,
                    state=state,
                    blockers=tuple(sorted(set(blockers))),
                )
            )
        ranked_policy_records, diagnostic_policy_keys = (
            self._diagnostic_rank(policy_records)
        )
        unresolved_states = {
            CandidateEvaluationState.UNRESOLVED_COLLISION,
            CandidateEvaluationState.UNRESOLVED_WRENCH,
            CandidateEvaluationState.PROTOCOL_REJECTED,
        }
        unresolved = any(
            row.state in unresolved_states for row in ranked_records
        ) or any(
            row.state in unresolved_states for row in ranked_policy_records
        )
        selection_blockers = {
            FORMAL_UNCERTAINTY_BLOCKER,
            "NO_FORMALLY_SCORED_CANDIDATE",
        }
        if ranked_policy_records:
            selection_blockers.add(
                "NO_FORMALLY_SCORED_CONTACT_RANGE_POLICY"
            )
        if unresolved:
            selection_blockers.add(
                "UNRESOLVED_ACCEPTED_OUTPUTS_PREVENT_FINITE_DESIGN_WINNER"
            )
        result_document = {
            "method_id": METHOD_ID,
            "rank_contract_sha256": self.rank_contract_sha256,
            "generation_contract_sha256": (
                self.expected_generation_contract_sha256
            ),
            "generation_result_sha256": generation_sha256,
            "common_scenario_design": self.common_scenario_design,
            "generation_attempt_lineage": snapshots,
            "candidate_records": ranked_records,
            "contact_range_policy_records": ranked_policy_records,
            "diagnostic_ranked_keys": diagnostic_keys,
            "diagnostic_ranked_policy_keys": diagnostic_policy_keys,
            "formal_ranked_keys": (),
            "formal_ranked_policy_keys": (),
            "selected_v9_parameter_key_hex": None,
            "selected_candidate": None,
            "selected_contact_range_policy_key_hex": None,
            "selected_contact_range_policy": None,
            "selection_blockers": tuple(sorted(selection_blockers)),
            "all_unique_candidates_resolved": not unresolved,
        }
        encoded = _canonical_bytes(result_document)
        return PostGenerationRankResult(
            method_id=METHOD_ID,
            rank_contract_sha256=self.rank_contract_sha256,
            generation_contract_sha256=(
                self.expected_generation_contract_sha256
            ),
            generation_result_sha256=generation_sha256,
            common_scenario_design=self.common_scenario_design,
            generation_attempt_lineage=snapshots,
            candidate_records=ranked_records,
            contact_range_policy_records=ranked_policy_records,
            diagnostic_ranked_keys=diagnostic_keys,
            diagnostic_ranked_policy_keys=diagnostic_policy_keys,
            formal_ranked_keys=(),
            formal_ranked_policy_keys=(),
            selected_v9_parameter_key_hex=None,
            selected_candidate=None,
            selected_contact_range_policy_key_hex=None,
            selected_contact_range_policy=None,
            selection_blockers=tuple(sorted(selection_blockers)),
            all_unique_candidates_resolved=not unresolved,
            canonical_json_bytes=encoded,
            canonical_sha256=hashlib.sha256(encoded).hexdigest(),
        )


__all__ = [
    "COMPLETE_CLEARANCE_METHOD_ID",
    "COMPLETE_CLEARANCE_SCOPE",
    "COMPLETE_POLICY_CLEARANCE_METHOD_ID",
    "CandidateEvaluationState",
    "CommonScenarioDesign",
    "CompleteTrajectoryCollisionCertificate",
    "CompleteContactRangeTrajectoryCollisionCertificate",
    "DiagnosticWrenchSummary",
    "GenerationAttemptSnapshot",
    "METHOD_ID",
    "PostGenerationRankOnlyPipeline",
    "PostGenerationRankResult",
    "PostGenerationRankingError",
    "RankCandidateRecord",
    "RankContactRangePolicyRecord",
    "SCENARIO_COUNT",
    "SCENARIO_DESIGN_SHA256",
    "SCENARIO_DIMENSION",
    "SCENARIO_METHOD_ID",
    "SCENARIO_SOBOL_SEED",
    "SELECTION_ORDER",
    "TIE_BREAK_RULE",
    "POLICY_AWARE_RANKING_STATUS",
    "candidate_sha256",
    "v9_evidence_sha256",
]
