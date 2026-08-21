"""Frozen-budget top-level proposals for the CARTS V9 closure certifier.

This module deliberately has a narrow role.  Four independent scrambled
Sobol lanes propose parameters, every proposal is converted to the one V9
five-dimensional parameter domain, and each exact canonical parameter tuple
is delegated to V9 at most once.  A proposal is never contact evidence and a
V9 result from this module is only a static closure result; collision, wrench,
uncertainty and dynamic claims belong to later stages.

The runtime boundary is structural but strict: the production method/domain
identifiers, layouts, hand identity, closure-model identity, and PAD order must
all agree before one design point is generated.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
import scipy

from kcg_connector.grasp.robust.grasp_optimizer import (
    GraspCandidate,
    deterministic_sobol,
)
from kcg_connector.grasp.robust.ray_closure import (
    CANDIDATE_REPRESENTATIVE_ROLE,
    CLOSURE_PARAMETER_DOMAIN_ID as PRODUCTION_V9_PARAMETER_DOMAIN_ID,
    CertifiedSequentialClosurePolicy,
    DisplayOnlyGraspProposal,
    METHOD_ID as PRODUCTION_V9_METHOD_ID,
    MODEL_BINDING_COMPLETE_STATUS,
    MODEL_CONTRACT_DIGEST_METHOD_ID,
    PARAMETER_LAYOUT_PREFIX as PRODUCTION_V9_PARAMETER_LAYOUT_PREFIX,
    REPRESENTATIVE_PROPOSAL_FAILURE_REASON,
)
from kcg_connector.grasp.robust.surface_anchored_closure import (
    FIXED_ANCHOR_METHOD_ID as PRODUCTION_FIXED_ANCHOR_METHOD_ID,
    FIXED_ANCHOR_PARAMETER_DOMAIN_ID
    as PRODUCTION_FIXED_ANCHOR_PARAMETER_DOMAIN_ID,
    FIXED_ANCHOR_PARAMETER_LAYOUT_PREFIX
    as PRODUCTION_FIXED_ANCHOR_PARAMETER_LAYOUT_PREFIX,
)


METHOD_ID = "CARTS_DIRECT_V9_PLUS_FIXED_SINGLE_ANCHOR_STRATIFIED_GENERATOR_V1"
V9_PARAMETER_DOMAIN_ID = PRODUCTION_V9_PARAMETER_DOMAIN_ID
DIRECT_V9_PARAMETER_LAYOUT_PREFIX = PRODUCTION_V9_PARAMETER_LAYOUT_PREFIX
FIXED_ANCHOR_PARAMETER_LAYOUT_PREFIX = (
    PRODUCTION_FIXED_ANCHOR_PARAMETER_LAYOUT_PREFIX
)
FROZEN_V9_PARAMETER_DIMENSION = 5
FROZEN_FIXED_ANCHOR_PARAMETER_DIMENSION = 6
ALLOWED_TOTAL_ATTEMPT_BUDGETS = (128, 256, 512)
MAIN_TOTAL_ATTEMPT_BUDGET = 256
MAXIMUM_POINTS_PER_LANE = max(ALLOWED_TOTAL_ATTEMPT_BUDGETS) // 4
LOCAL_REFINEMENT_EVALUATION_BUDGET = 0
SCHEDULE_RULE = "ROUND_ROBIN_DIRECT_THEN_PAD_A_THEN_PAD_B_THEN_PAD_C"
DEDUPLICATION_RULE = (
    "CANONICAL_V9_FIVE_BIG_ENDIAN_BINARY64_VALUES_EXACT_BYTES_BEFORE_V9"
)
EXACT_CANDIDATE_OUTPUT_CHANNEL = "STATIC_EXACT_GRASP_CANDIDATE"
CONTACT_RANGE_POLICY_OUTPUT_CHANNEL = (
    "STATIC_CERTIFIED_SEQUENTIAL_CLOSURE_POLICY"
)
TOP_LEVEL_OUTPUT_CLAIM = (
    "STATIC_V9_EXACT_CANDIDATE_OR_CONTACT_RANGE_POLICY_ACCEPTANCE_ONLY"
)
CONTACT_RANGE_POLICY_DOWNSTREAM_STATUS = (
    "PENDING_POLICY_AWARE_COLLISION_AND_WRENCH"
)


class TopLevelCandidateGeneratorError(ValueError):
    """Raised when the immutable generator protocol is violated."""


class CandidateLane(str, Enum):
    """Complete formal lane allowlist with no external registry."""

    DIRECT_V9 = "DIRECT_V9"
    SURFACE_PAD_A = "SURFACE_PAD_A"
    SURFACE_PAD_B = "SURFACE_PAD_B"
    SURFACE_PAD_C = "SURFACE_PAD_C"


class AttemptStatus(str, Enum):
    """Mutually exclusive terminal status for one budget-consuming attempt."""

    PROPOSAL_REJECTED = "PROPOSAL_REJECTED"
    PROPOSAL_PROTOCOL_REJECTED = "PROPOSAL_PROTOCOL_REJECTED"
    PROPOSAL_V9_DOMAIN_REJECTED = "PROPOSAL_V9_DOMAIN_REJECTED"
    DUPLICATE_CANONICAL_V9_PARAMETERS = "DUPLICATE_CANONICAL_V9_PARAMETERS"
    V9_REJECTED = "V9_REJECTED"
    V9_PROTOCOL_REJECTED = "V9_PROTOCOL_REJECTED"
    V9_EVALUATOR_EXCEPTION = "V9_EVALUATOR_EXCEPTION"
    STATIC_V9_ACCEPTED = "STATIC_V9_ACCEPTED"
    STATIC_V9_POLICY_ACCEPTED = "STATIC_V9_POLICY_ACCEPTED"


@dataclass(frozen=True)
class CandidateLaneSpec:
    lane: CandidateLane
    dimension: int
    sobol_seed: int
    anchor_pad_ordinal: int | None


LANE_SPECS = (
    CandidateLaneSpec(CandidateLane.DIRECT_V9, 5, 20260820, None),
    CandidateLaneSpec(CandidateLane.SURFACE_PAD_A, 6, 20260821, 0),
    CandidateLaneSpec(CandidateLane.SURFACE_PAD_B, 6, 20260822, 1),
    CandidateLaneSpec(CandidateLane.SURFACE_PAD_C, 6, 20260823, 2),
)


@runtime_checkable
class FixedAnchorSurfaceProposer(Protocol):
    """Protocol reserved for the production fixed-anchor surface mapper."""

    fixed_anchor_method_id: str
    fixed_anchor_parameter_domain_id: str
    fixed_anchor_parameter_layout: Sequence[str]
    prepared_pad_names: Sequence[str]
    hand_model: Any
    closure_model: Any

    def propose_fixed_anchor(
        self,
        parameters6: np.ndarray,
        anchor_pad_name: str,
        hand_model: Any,
    ) -> Any:
        """Return ``v9_parameters_unit`` and ``audit`` fields."""


@runtime_checkable
class V9ClosureEvaluator(Protocol):
    """Structural protocol for V9 static closure evaluation."""

    method_id: str
    closure_parameter_domain_id: str
    parameter_layout: Sequence[str]
    preshape_joint_names: Sequence[str]
    prepared_pads: Sequence[Any]
    hand_model: Any
    model_binding_complete: bool
    model_binding_status: str
    object_geometry_sha256: str
    model_contract_sha256: str
    pad_geometry_sha256: Sequence[str]
    pad_runtime_geometry_sha256: Sequence[str]
    pad_link_names: Sequence[str]
    closing_directions_physical: Sequence[Sequence[float]]
    model_contract_canonical_json: str

    def evaluate_unit_parameters(
        self,
        parameters_unit: np.ndarray,
        hand_model: Any,
    ) -> Any:
        """Return an object with ``candidate`` and ``audit`` fields."""


@dataclass(frozen=True)
class CanonicalV9Parameters:
    """Canonical V9 identity after periodic-yaw and signed-zero handling."""

    values: tuple[float, float, float, float, float]
    exact_key: bytes

    @property
    def exact_key_hex(self) -> str:
        return self.exact_key.hex()


@dataclass(frozen=True)
class CandidateLineage:
    """The complete proposal-side provenance for one attempt."""

    attempt_index: int
    lane: CandidateLane
    lane_point_index: int
    sobol_seed: int
    sobol_parameters_unit: tuple[float, ...]
    anchor_pad_name: str | None
    proposal_audit: Any | None
    proposal_failure_reason: str | None


@dataclass(frozen=True)
class CandidateAttemptAudit:
    """Fail-closed record for exactly one consumed top-level budget slot."""

    lineage: CandidateLineage
    status: AttemptStatus
    v9_parameters_unit: tuple[float, float, float, float, float] | None
    v9_parameter_key_hex: str | None
    duplicate_of_attempt_index: int | None
    v9_audit: Any | None
    invocation_binding: "V9InvocationAuditBinding | None"
    v9_failure_reason: str | None
    failure_reason: str | None

    @property
    def attempt_index(self) -> int:
        return self.lineage.attempt_index

    @property
    def lane(self) -> CandidateLane:
        return self.lineage.lane

    @property
    def lane_point_index(self) -> int:
        return self.lineage.lane_point_index

    @property
    def sobol_parameters_unit(self) -> tuple[float, ...]:
        return self.lineage.sobol_parameters_unit

    @property
    def anchor_pad_name(self) -> str | None:
        return self.lineage.anchor_pad_name


@dataclass(frozen=True)
class UniqueV9Evaluation:
    """One and only one V9 invocation plus every proposal that led to it."""

    v9_parameters_unit: tuple[float, float, float, float, float]
    v9_parameter_key_hex: str
    first_attempt_index: int
    lineage: tuple[CandidateLineage, ...]
    candidate: Any | None
    v9_audit: Any | None
    invocation_binding: "V9InvocationAuditBinding | None"
    status: AttemptStatus
    v9_failure_reason: str | None
    sequential_closure_policy: Any | None = None

    @property
    def accepted_static(self) -> bool:
        return self.status in {
            AttemptStatus.STATIC_V9_ACCEPTED,
            AttemptStatus.STATIC_V9_POLICY_ACCEPTED,
        }

    @property
    def accepted_static_policy(self) -> bool:
        return self.status is AttemptStatus.STATIC_V9_POLICY_ACCEPTED


@dataclass(frozen=True)
class StaticV9AcceptedCandidate:
    """A static V9 acceptance, with no collision/wrench/dynamic implication."""

    v9_parameters_unit: tuple[float, float, float, float, float]
    v9_parameter_key_hex: str
    candidate: Any
    v9_audit: Any
    invocation_binding: "V9InvocationAuditBinding"
    lineage: tuple[CandidateLineage, ...]


@dataclass(frozen=True)
class StaticV9AcceptedPolicy:
    """A contact-range plan before collision, wrench, ranking, or dynamics."""

    v9_parameters_unit: tuple[float, float, float, float, float]
    v9_parameter_key_hex: str
    sequential_closure_policy: CertifiedSequentialClosurePolicy
    v9_audit: Any
    invocation_binding: "V9InvocationAuditBinding"
    lineage: tuple[CandidateLineage, ...]


@dataclass(frozen=True)
class V9InvocationAuditBinding:
    """Top-level proof binding one exact request to one validated V9 audit."""

    method_id: str
    parameter_domain_id: str
    parameter_layout: tuple[str, ...]
    requested_parameters_unit: tuple[float, float, float, float, float]
    requested_parameter_key_hex: str
    raw_v9_audit: Any


@dataclass(frozen=True)
class TopLevelGenerationResult:
    """Complete bounded output of one nested-prefix generation run."""

    method_id: str
    contract_hash_sha256: str
    total_attempt_budget: int
    attempts_per_lane: int
    local_refinement_evaluation_budget: int
    attempts: tuple[CandidateAttemptAudit, ...]
    unique_v9_evaluations: tuple[UniqueV9Evaluation, ...]
    accepted_candidates: tuple[StaticV9AcceptedCandidate, ...]
    v9_evaluation_count: int
    duplicate_attempt_count: int
    proposal_failure_count: int
    accepted_policies: tuple[StaticV9AcceptedPolicy, ...] = ()

    def __post_init__(self) -> None:
        candidate_keys = tuple(
            row.v9_parameter_key_hex for row in self.accepted_candidates
        )
        policy_keys = tuple(
            row.v9_parameter_key_hex for row in self.accepted_policies
        )
        expected_candidate_keys = tuple(
            row.v9_parameter_key_hex
            for row in self.unique_v9_evaluations
            if row.status is AttemptStatus.STATIC_V9_ACCEPTED
        )
        expected_policy_keys = tuple(
            row.v9_parameter_key_hex
            for row in self.unique_v9_evaluations
            if row.status is AttemptStatus.STATIC_V9_POLICY_ACCEPTED
        )
        if len(self.attempts) != self.total_attempt_budget:
            raise TopLevelCandidateGeneratorError(
                "result attempt count does not equal its frozen budget"
            )
        if self.v9_evaluation_count != len(self.unique_v9_evaluations):
            raise TopLevelCandidateGeneratorError(
                "each unique canonical parameter must have exactly one "
                "V9 evaluation"
            )
        if self.local_refinement_evaluation_budget != 0:
            raise TopLevelCandidateGeneratorError(
                "formal V1 cannot execute local refinement"
            )
        if (
            candidate_keys != expected_candidate_keys
            or policy_keys != expected_policy_keys
            or set(candidate_keys).intersection(policy_keys)
            or any(
                type(row) is not StaticV9AcceptedPolicy
                or type(row.sequential_closure_policy)
                is not CertifiedSequentialClosurePolicy
                for row in self.accepted_policies
            )
        ):
            raise TopLevelCandidateGeneratorError(
                "static candidate/policy outputs contradict unique evaluations"
            )


@dataclass(frozen=True)
class ResumableGenerationState:
    """Immutable committed prefix for the one generator transition engine."""

    method_id: str
    contract_hash_sha256: str
    target_total_attempt_budget: int
    attempts: tuple[CandidateAttemptAudit, ...]
    unique_v9_evaluations: tuple[UniqueV9Evaluation, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "attempts", tuple(self.attempts))
        object.__setattr__(
            self,
            "unique_v9_evaluations",
            tuple(self.unique_v9_evaluations),
        )

    @property
    def completed_attempt_count(self) -> int:
        return len(self.attempts)


@dataclass
class _EvaluationAccumulator:
    canonical: CanonicalV9Parameters
    first_attempt_index: int
    lineage: list[CandidateLineage]
    candidate: Any | None
    sequential_closure_policy: Any | None
    v9_audit: Any | None
    invocation_binding: V9InvocationAuditBinding | None
    status: AttemptStatus
    v9_failure_reason: str | None


_MISSING = object()


@dataclass(frozen=True)
class _V9ModelBinding:
    model_binding_complete: bool
    model_binding_status: str
    model_contract_digest_method_id: str
    object_geometry_sha256: str
    model_contract_sha256: str
    pad_geometry_sha256: tuple[str, ...]
    pad_runtime_geometry_sha256: tuple[str, ...]
    pad_link_names: tuple[str, ...]
    closing_directions_physical: tuple[tuple[float, ...], ...]
    closing_directions_binary64_key: bytes
    model_contract_canonical_json: str
    model_contract_canonical_json_bytes: bytes


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, _MISSING)
    return getattr(value, name, _MISSING)


def _required_field(value: Any, name: str) -> Any:
    raw = _field(value, name)
    if raw is _MISSING:
        raise TopLevelCandidateGeneratorError(
            f"protocol audit is missing required field {name}"
        )
    return raw


def _finite_float_tuple(value: Any, *, label: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError, OverflowError) as error:
        raise TopLevelCandidateGeneratorError(
            f"{label} must be a finite numeric sequence"
        ) from error
    if not all(math.isfinite(item) for item in result):
        raise TopLevelCandidateGeneratorError(
            f"{label} must contain only finite values"
        )
    return result


def _implementation_type_id(value: Any) -> str:
    implementation_type = type(value)
    return (
        f"{implementation_type.__module__}."
        f"{implementation_type.__qualname__}"
    )


def _binary64_sequence_key(value: Sequence[float]) -> bytes:
    return np.asarray(tuple(value), dtype=">f8").tobytes(order="C")


def _evidence_equal(first: object, second: object) -> bool:
    if first is second:
        return True
    try:
        comparison = first == second
    except Exception:
        return False
    if isinstance(comparison, np.ndarray):
        return bool(np.all(comparison))
    try:
        return bool(comparison)
    except (TypeError, ValueError):
        return False


def _prepared_pad_names(evaluator: Any) -> tuple[str, ...]:
    prepared_pads = _field(evaluator, "prepared_pads")
    if prepared_pads is _MISSING:
        raise TopLevelCandidateGeneratorError(
            "v9_evaluator must expose prepared_pads"
        )
    names: list[str] = []
    try:
        rows = tuple(prepared_pads)
    except TypeError as error:
        raise TopLevelCandidateGeneratorError(
            "v9_evaluator prepared_pads must be a finite sequence"
        ) from error
    for row in rows:
        verified = _field(row, "verified")
        name = _MISSING if verified is _MISSING else _field(verified, "name")
        if not isinstance(name, str) or not name:
            raise TopLevelCandidateGeneratorError(
                "every v9_evaluator prepared PAD must expose a non-empty "
                "verified.name"
            )
        names.append(name)
    return tuple(names)


def _sha256_text(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TopLevelCandidateGeneratorError(
            f"{label} must be 64 lowercase hexadecimal digits"
        )
    return value


def _string_tuple(
    value: Any,
    *,
    label: str,
    expected_count: int,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise TopLevelCandidateGeneratorError(
            f"{label} must be a finite string sequence"
        )
    try:
        rows = tuple(value)
    except TypeError as error:
        raise TopLevelCandidateGeneratorError(
            f"{label} must be a finite string sequence"
        ) from error
    if (
        len(rows) != expected_count
        or any(not isinstance(item, str) or not item for item in rows)
    ):
        raise TopLevelCandidateGeneratorError(
            f"{label} must contain exactly {expected_count} non-empty strings"
        )
    return rows


def _closing_direction_binding(
    value: Any,
    *,
    label: str,
) -> tuple[tuple[tuple[float, ...], ...], bytes]:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise TopLevelCandidateGeneratorError(
            f"{label} must be a finite binary64 matrix"
        ) from error
    if (
        array.ndim != 2
        or array.shape[0] != 3
        or array.shape[1] == 0
        or not np.all(np.isfinite(array))
        or any(not np.any(row != 0.0) for row in array)
    ):
        raise TopLevelCandidateGeneratorError(
            f"{label} must contain three finite non-zero rows"
        )
    rows = tuple(
        tuple(float(component) for component in row)
        for row in array
    )
    exact_key = np.asarray(array, dtype=">f8").tobytes(order="C")
    return rows, exact_key


def _float64_rows_hex(
    rows: Sequence[Sequence[float]],
) -> list[list[str]]:
    return [
        [float(component).hex() for component in row]
        for row in rows
    ]


def _validated_model_binding(
    value: Any,
    *,
    expected_pad_names: tuple[str, ...],
    label: str,
) -> _V9ModelBinding:
    complete = _required_field(value, "model_binding_complete")
    if complete is not True:
        raise TopLevelCandidateGeneratorError(
            f"{label} model_binding_complete must be production true"
        )
    status = _required_field(value, "model_binding_status")
    if status != MODEL_BINDING_COMPLETE_STATUS:
        raise TopLevelCandidateGeneratorError(
            f"{label} model_binding_status differs from production"
        )
    object_sha256 = _sha256_text(
        _required_field(value, "object_geometry_sha256"),
        label=f"{label} object_geometry_sha256",
    )
    model_sha256 = _sha256_text(
        _required_field(value, "model_contract_sha256"),
        label=f"{label} model_contract_sha256",
    )
    pad_source_sha256 = _string_tuple(
        _required_field(value, "pad_geometry_sha256"),
        label=f"{label} pad_geometry_sha256",
        expected_count=3,
    )
    pad_runtime_sha256 = _string_tuple(
        _required_field(value, "pad_runtime_geometry_sha256"),
        label=f"{label} pad_runtime_geometry_sha256",
        expected_count=3,
    )
    for index, digest in enumerate(pad_source_sha256):
        _sha256_text(
            digest,
            label=f"{label} pad_geometry_sha256[{index}]",
        )
    for index, digest in enumerate(pad_runtime_sha256):
        _sha256_text(
            digest,
            label=f"{label} pad_runtime_geometry_sha256[{index}]",
        )
    pad_link_names = _string_tuple(
        _required_field(value, "pad_link_names"),
        label=f"{label} pad_link_names",
        expected_count=3,
    )
    directions, direction_key = _closing_direction_binding(
        _required_field(value, "closing_directions_physical"),
        label=f"{label} closing_directions_physical",
    )
    canonical_json = _required_field(
        value, "model_contract_canonical_json"
    )
    if not isinstance(canonical_json, str):
        raise TopLevelCandidateGeneratorError(
            f"{label} model_contract_canonical_json must be text"
        )
    try:
        canonical_bytes = canonical_json.encode("utf-8")
        document = json.loads(canonical_json)
        recomputed_json = json.dumps(
            document,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError, UnicodeError) as error:
        raise TopLevelCandidateGeneratorError(
            f"{label} model contract canonical JSON is invalid"
        ) from error
    if recomputed_json.encode("utf-8") != canonical_bytes:
        raise TopLevelCandidateGeneratorError(
            f"{label} model contract JSON is not canonical"
        )
    recomputed_digest = hashlib.sha256(canonical_bytes).hexdigest()
    if recomputed_digest != model_sha256:
        raise TopLevelCandidateGeneratorError(
            f"{label} model contract SHA-256 fails recomputation"
        )
    try:
        verified_pads = document["verified_pads"]
        document_pad_names = tuple(row["name"] for row in verified_pads)
        document_pad_links = tuple(
            row["link_name"] for row in verified_pads
        )
        document_pad_source_sha256 = tuple(
            row["source_mesh_sha256"] for row in verified_pads
        )
        document_pad_runtime_sha256 = tuple(
            row["runtime_geometry_sha256"] for row in verified_pads
        )
        document_directions = document["closure"][
            "closing_directions_physical"
        ]
        if (
            document["schema"] != MODEL_CONTRACT_DIGEST_METHOD_ID
            or document["object"]["geometry_sha256"] != object_sha256
            or document_pad_names != expected_pad_names
            or document_pad_links != pad_link_names
            or document_pad_source_sha256 != pad_source_sha256
            or document_pad_runtime_sha256 != pad_runtime_sha256
            or document_directions != _float64_rows_hex(directions)
            or document["ray_closure"]["method_id"]
            != PRODUCTION_V9_METHOD_ID
            or document["ray_closure"]["closure_parameter_domain_id"]
            != PRODUCTION_V9_PARAMETER_DOMAIN_ID
        ):
            raise TopLevelCandidateGeneratorError(
                f"{label} canonical model contract contradicts evidence"
            )
    except (KeyError, TypeError) as error:
        raise TopLevelCandidateGeneratorError(
            f"{label} canonical model contract is structurally incomplete"
        ) from error
    return _V9ModelBinding(
        model_binding_complete=True,
        model_binding_status=status,
        model_contract_digest_method_id=MODEL_CONTRACT_DIGEST_METHOD_ID,
        object_geometry_sha256=object_sha256,
        model_contract_sha256=model_sha256,
        pad_geometry_sha256=pad_source_sha256,
        pad_runtime_geometry_sha256=pad_runtime_sha256,
        pad_link_names=pad_link_names,
        closing_directions_physical=directions,
        closing_directions_binary64_key=direction_key,
        model_contract_canonical_json=canonical_json,
        model_contract_canonical_json_bytes=canonical_bytes,
    )


def canonicalize_v9_parameters(
    parameters_unit: Sequence[float],
    *,
    parameter_layout: Sequence[str],
) -> CanonicalV9Parameters:
    """Return the sole V9 identity or fail closed on an invalid mapped point.

    Yaw is a periodic unit coordinate and is reduced modulo one.  Placement
    and preshape are closed unit coordinates.  Every signed zero is changed to
    positive zero before a big-endian binary64 exact-byte key is created.
    """

    layout = tuple(parameter_layout)
    if (
        len(layout) != FROZEN_V9_PARAMETER_DIMENSION
        or layout[: len(DIRECT_V9_PARAMETER_LAYOUT_PREFIX)]
        != DIRECT_V9_PARAMETER_LAYOUT_PREFIX
    ):
        raise TopLevelCandidateGeneratorError(
            "V9 parameter layout differs from the production "
            "five-dimensional domain"
        )
    raw_values = _finite_float_tuple(parameters_unit, label="V9 parameters")
    if len(raw_values) != len(layout):
        raise TopLevelCandidateGeneratorError(
            "V9 parameters must contain exactly five values"
        )

    yaw = math.fmod(raw_values[0], 1.0)
    if yaw < 0.0:
        yaw += 1.0
    if yaw >= 1.0:
        yaw = 0.0
    canonical_values = [yaw]
    for label, value in zip(layout[1:], raw_values[1:]):
        if value < 0.0 or value > 1.0:
            raise TopLevelCandidateGeneratorError(
                f"{label} must lie in the closed unit interval"
            )
        canonical_values.append(value)
    canonical_values = [
        0.0 if value == 0.0 else value for value in canonical_values
    ]
    values = tuple(canonical_values)
    if len(values) != 5:  # pragma: no cover - construction invariant
        raise AssertionError("canonical V9 parameter dimension changed")
    exact_key = np.asarray(values, dtype=">f8").tobytes(order="C")
    return CanonicalV9Parameters(
        values=(values[0], values[1], values[2], values[3], values[4]),
        exact_key=exact_key,
    )


class TopLevelCandidateGenerator:
    """Generate a fixed nested design and delegate unique points once to V9."""

    def __init__(
        self,
        *,
        v9_evaluator: V9ClosureEvaluator,
        surface_proposer: FixedAnchorSurfaceProposer,
        hand_model: Any,
        anchor_pad_names: Sequence[str],
    ) -> None:
        evaluate = getattr(v9_evaluator, "evaluate_unit_parameters", None)
        if not callable(evaluate):
            raise TopLevelCandidateGeneratorError(
                "v9_evaluator must expose evaluate_unit_parameters"
            )
        propose = getattr(surface_proposer, "propose_fixed_anchor", None)
        if not callable(propose):
            raise TopLevelCandidateGeneratorError(
                "surface_proposer must expose propose_fixed_anchor"
            )
        if hand_model is None:
            raise TopLevelCandidateGeneratorError("hand_model cannot be None")
        names = tuple(anchor_pad_names)
        if (
            len(names) != 3
            or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != 3
        ):
            raise TopLevelCandidateGeneratorError(
                "anchor_pad_names must contain three distinct non-empty names"
            )

        if _field(v9_evaluator, "method_id") != PRODUCTION_V9_METHOD_ID:
            raise TopLevelCandidateGeneratorError(
                "v9_evaluator method_id differs from the production "
                "V9 certifier"
            )
        if (
            _field(v9_evaluator, "closure_parameter_domain_id")
            != PRODUCTION_V9_PARAMETER_DOMAIN_ID
        ):
            raise TopLevelCandidateGeneratorError(
                "v9_evaluator closure_parameter_domain_id differs "
                "from production"
            )
        if _field(v9_evaluator, "hand_model") is not hand_model:
            raise TopLevelCandidateGeneratorError(
                "v9_evaluator must be bound to the exact supplied "
                "hand_model instance"
            )
        raw_preshape_names = _field(v9_evaluator, "preshape_joint_names")
        if raw_preshape_names is _MISSING:
            raise TopLevelCandidateGeneratorError(
                "v9_evaluator must expose preshape_joint_names"
            )
        preshape_names = tuple(raw_preshape_names)
        if any(
            not isinstance(name, str) or not name
            for name in preshape_names
        ):
            raise TopLevelCandidateGeneratorError(
                "v9_evaluator preshape_joint_names must be non-empty strings"
            )
        v9_layout = tuple(_field(v9_evaluator, "parameter_layout"))
        expected_v9_layout = DIRECT_V9_PARAMETER_LAYOUT_PREFIX + tuple(
            f"preshape_joint_unit:{name}" for name in preshape_names
        )
        if (
            v9_layout != expected_v9_layout
            or len(v9_layout) != FROZEN_V9_PARAMETER_DIMENSION
        ):
            raise TopLevelCandidateGeneratorError(
                "v9_evaluator parameter_layout must be the hand-derived "
                "production five-dimensional layout"
            )
        evaluator_pad_names = _prepared_pad_names(v9_evaluator)
        if evaluator_pad_names != names:
            raise TopLevelCandidateGeneratorError(
                "anchor_pad_names must exactly equal the production V9 "
                "prepared PAD order"
            )
        v9_model_binding = _validated_model_binding(
            v9_evaluator,
            expected_pad_names=names,
            label="v9_evaluator",
        )

        if (
            _field(surface_proposer, "fixed_anchor_method_id")
            != PRODUCTION_FIXED_ANCHOR_METHOD_ID
        ):
            raise TopLevelCandidateGeneratorError(
                "surface_proposer fixed_anchor_method_id differs from "
                "production"
            )
        if (
            _field(surface_proposer, "fixed_anchor_parameter_domain_id")
            != PRODUCTION_FIXED_ANCHOR_PARAMETER_DOMAIN_ID
        ):
            raise TopLevelCandidateGeneratorError(
                "surface_proposer fixed_anchor_parameter_domain_id differs "
                "from production"
            )
        if _field(surface_proposer, "hand_model") is not hand_model:
            raise TopLevelCandidateGeneratorError(
                "surface_proposer must be bound to the exact supplied "
                "hand_model instance"
            )
        if _field(surface_proposer, "closure_model") is not v9_evaluator:
            raise TopLevelCandidateGeneratorError(
                "surface_proposer closure_model must be the exact V9 "
                "evaluator instance"
            )
        surface_pad_names = tuple(
            _field(surface_proposer, "prepared_pad_names")
        )
        if surface_pad_names != names:
            raise TopLevelCandidateGeneratorError(
                "surface_proposer prepared_pad_names must equal the V9 "
                "PAD order"
            )
        fixed_layout = tuple(
            _field(surface_proposer, "fixed_anchor_parameter_layout")
        )
        expected_fixed_layout = FIXED_ANCHOR_PARAMETER_LAYOUT_PREFIX + tuple(
            f"preshape_joint_unit:{name}" for name in preshape_names
        )
        if (
            fixed_layout != expected_fixed_layout
            or len(fixed_layout) != FROZEN_FIXED_ANCHOR_PARAMETER_DIMENSION
        ):
            raise TopLevelCandidateGeneratorError(
                "surface_proposer fixed_anchor_parameter_layout must be the "
                "hand-derived production six-dimensional layout"
            )

        self.v9_evaluator = v9_evaluator
        self.surface_proposer = surface_proposer
        self.hand_model = hand_model
        self.anchor_pad_names = names
        self.preshape_joint_names = preshape_names
        self.v9_parameter_layout = v9_layout
        self.fixed_anchor_parameter_layout = fixed_layout
        self.v9_model_binding = v9_model_binding
        self.v9_model_contract_sha256 = (
            v9_model_binding.model_contract_sha256
        )
        self.v9_implementation_type_id = _implementation_type_id(v9_evaluator)
        self.surface_implementation_type_id = _implementation_type_id(
            surface_proposer
        )
        self._maximum_designs = MappingProxyType(self._designs())
        self._contract_document = self._make_contract_document()
        canonical_json = json.dumps(
            self._contract_document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        self.contract_hash_sha256 = hashlib.sha256(canonical_json).hexdigest()

    def _make_contract_document(self) -> dict[str, Any]:
        lanes = []
        for spec in LANE_SPECS:
            anchor_name = (
                None
                if spec.anchor_pad_ordinal is None
                else self.anchor_pad_names[spec.anchor_pad_ordinal]
            )
            lanes.append(
                {
                    "lane": spec.lane.value,
                    "dimension": spec.dimension,
                    "sobol_seed": spec.sobol_seed,
                    "anchor_pad_ordinal": spec.anchor_pad_ordinal,
                    "anchor_pad_name": anchor_name,
                    "maximum_prefix_design_sha256": hashlib.sha256(
                        np.asarray(
                            self._maximum_designs[spec.lane], dtype=">f8"
                        ).tobytes(order="C")
                    ).hexdigest(),
                }
            )
        return {
            "method_id": METHOD_ID,
            "v9_certifier": {
                "implementation_type_id": self.v9_implementation_type_id,
                "method_id": PRODUCTION_V9_METHOD_ID,
                "parameter_domain_id": PRODUCTION_V9_PARAMETER_DOMAIN_ID,
                "parameter_layout": list(self.v9_parameter_layout),
                "model_binding_complete": (
                    self.v9_model_binding.model_binding_complete
                ),
                "model_binding_status": (
                    self.v9_model_binding.model_binding_status
                ),
                "model_contract_digest_method_id": (
                    self.v9_model_binding.model_contract_digest_method_id
                ),
                "model_contract_sha256": self.v9_model_contract_sha256,
            },
            "fixed_anchor_mapper": {
                "implementation_type_id": self.surface_implementation_type_id,
                "method_id": PRODUCTION_FIXED_ANCHOR_METHOD_ID,
                "parameter_domain_id": (
                    PRODUCTION_FIXED_ANCHOR_PARAMETER_DOMAIN_ID
                ),
                "parameter_layout": list(self.fixed_anchor_parameter_layout),
            },
            "hand_binding": {
                "runtime_object_identity_required": True,
                "preshape_joint_names": list(self.preshape_joint_names),
                "prepared_pad_order": list(self.anchor_pad_names),
            },
            "lanes": lanes,
            "lane_order": [spec.lane.value for spec in LANE_SPECS],
            "schedule_rule": SCHEDULE_RULE,
            "allowed_total_attempt_budgets": list(
                ALLOWED_TOTAL_ATTEMPT_BUDGETS
            ),
            "main_total_attempt_budget": MAIN_TOTAL_ATTEMPT_BUDGET,
            "maximum_points_per_lane": MAXIMUM_POINTS_PER_LANE,
            "proposal_failure_consumes_attempt": True,
            "duplicate_consumes_attempt": True,
            "replacement_sampling_allowed": False,
            "maximum_v9_evaluations_equals_attempt_budget": True,
            "deduplication_rule": DEDUPLICATION_RULE,
            "canonicalization": {
                "yaw_equivalence": (
                    "FINITE_BINARY64_MODULO_ONE_TO_HALF_OPEN_UNIT"
                ),
                "non_yaw_domain": "CLOSED_UNIT_INTERVAL",
                "signed_zero": "NORMALIZE_TO_POSITIVE_ZERO",
                "identity_encoding": "FIVE_BIG_ENDIAN_BINARY64_VALUES",
            },
            "sobol_design": {
                "generator": "SCIPY_STATS_QMC_SOBOL_VIA_DETERMINISTIC_SOBOL",
                "scipy_version": scipy.__version__,
                "scramble": True,
                "optimization": None,
                "generate_common_maximum_then_take_lane_prefix": True,
                "maximum_points_per_lane": MAXIMUM_POINTS_PER_LANE,
                "realized_design_hashes_are_contract_bound": True,
            },
            "v9_acceptance_rule": (
                "EITHER_EXACT_GRASP_CANDIDATE_WITH_NULL_FAILURE_OR_"
                "CERTIFIED_SEQUENTIAL_CLOSURE_POLICY_WITH_REGISTERED_"
                "REPRESENTATIVE_FAILURE_AND_STRICT_PRODUCTION_BINDINGS"
            ),
            "accepted_output_channels": [
                EXACT_CANDIDATE_OUTPUT_CHANNEL,
                CONTACT_RANGE_POLICY_OUTPUT_CHANNEL,
            ],
            "candidate_and_policy_mutually_exclusive": True,
            "display_only_proposal_formal_eligible": False,
            "contact_range_policy_downstream_status": (
                CONTACT_RANGE_POLICY_DOWNSTREAM_STATUS
            ),
            "external_lane_registry_supported": False,
            "local_refinement": {
                "method_id": "CANONICAL_V9_DYADIC_STENCIL_V1",
                "execution_status": "DISABLED_FOR_V1",
                "evaluation_budget": LOCAL_REFINEMENT_EVALUATION_BUDGET,
                "ranking_eligible": False,
            },
            "output_claim": TOP_LEVEL_OUTPUT_CLAIM,
        }

    @property
    def contract_document(self) -> dict[str, Any]:
        """Return an isolated JSON-compatible copy of the frozen contract."""

        return json.loads(
            json.dumps(
                self._contract_document,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
        )

    @staticmethod
    def _designs() -> dict[CandidateLane, np.ndarray]:
        """Generate one common maximum design for true budget prefixes."""

        designs: dict[CandidateLane, np.ndarray] = {}
        for spec in LANE_SPECS:
            design = np.array(
                deterministic_sobol(
                    dimension=spec.dimension,
                    count=MAXIMUM_POINTS_PER_LANE,
                    seed=spec.sobol_seed,
                ),
                dtype=np.float64,
                copy=True,
            )
            design.setflags(write=False)
            designs[spec.lane] = design
        return designs

    def _proposal_for_attempt(
        self,
        *,
        spec: CandidateLaneSpec,
        sobol_point: np.ndarray,
    ) -> tuple[
        Sequence[float] | None,
        Any | None,
        str | None,
        AttemptStatus | None,
    ]:
        if spec.lane is CandidateLane.DIRECT_V9:
            return sobol_point, None, None, None
        if spec.anchor_pad_ordinal is None:  # pragma: no cover
            raise AssertionError("surface lane lacks an anchor ordinal")
        anchor_name = self.anchor_pad_names[spec.anchor_pad_ordinal]
        try:
            proposal = self.surface_proposer.propose_fixed_anchor(
                sobol_point.copy(), anchor_name, self.hand_model
            )
        except Exception as error:  # fail closed; no retry or replacement
            reason = (
                "SURFACE_PROPOSER_EXCEPTION:"
                f"{type(error).__name__}:{error}"
            )
            return None, None, reason, AttemptStatus.PROPOSAL_PROTOCOL_REJECTED
        mapped = _field(proposal, "v9_parameters_unit")
        audit = _field(proposal, "audit")
        if mapped is _MISSING:
            return (
                None,
                None if audit is _MISSING else audit,
                "PROPOSAL_PROTOCOL_MISSING_V9_PARAMETERS_UNIT",
                AttemptStatus.PROPOSAL_PROTOCOL_REJECTED,
            )
        if audit is _MISSING or audit is None:
            return (
                None,
                None,
                "PROPOSAL_PROTOCOL_MISSING_OR_NULL_AUDIT",
                AttemptStatus.PROPOSAL_PROTOCOL_REJECTED,
            )
        try:
            if _required_field(audit, "method_id") != (
                PRODUCTION_FIXED_ANCHOR_METHOD_ID
            ):
                raise TopLevelCandidateGeneratorError(
                    "surface audit method_id differs from production"
                )
            if _required_field(audit, "parameter_domain_id") != (
                PRODUCTION_FIXED_ANCHOR_PARAMETER_DOMAIN_ID
            ):
                raise TopLevelCandidateGeneratorError(
                    "surface audit parameter_domain_id differs from production"
                )
            if tuple(_required_field(audit, "parameter_layout")) != (
                self.fixed_anchor_parameter_layout
            ):
                raise TopLevelCandidateGeneratorError(
                    "surface audit parameter_layout differs from the "
                    "bound mapper"
                )
            audited_parameters = _finite_float_tuple(
                _required_field(audit, "parameters_unit"),
                label="surface audit parameters_unit",
            )
            requested_parameters = tuple(float(value) for value in sobol_point)
            if (
                len(audited_parameters) != len(requested_parameters)
                or _binary64_sequence_key(audited_parameters)
                != _binary64_sequence_key(requested_parameters)
            ):
                raise TopLevelCandidateGeneratorError(
                    "surface audit parameters_unit differs from the "
                    "exact request"
                )
            if _required_field(audit, "anchor_pad_name") != anchor_name:
                raise TopLevelCandidateGeneratorError(
                    "surface audit anchor_pad_name differs from the "
                    "requested PAD"
                )
            raw_failure_field = _required_field(audit, "failure_reason")
            raw_failure = (
                None
                if raw_failure_field is None
                else str(raw_failure_field)
            )
            delegated = _required_field(
                audit, "delegated_volume_parameters_unit"
            )
            if mapped is None:
                if delegated is not None:
                    raise TopLevelCandidateGeneratorError(
                        "rejected surface proposal cannot audit delegated "
                        "V9 parameters"
                    )
            else:
                mapped_values = _finite_float_tuple(
                    mapped, label="surface proposal V9 parameters"
                )
                delegated_values = _finite_float_tuple(
                    delegated,
                    label="surface audit delegated V9 parameters",
                )
                if (
                    len(mapped_values) != len(delegated_values)
                    or _binary64_sequence_key(mapped_values)
                    != _binary64_sequence_key(delegated_values)
                ):
                    raise TopLevelCandidateGeneratorError(
                        "surface audit delegated parameters differ from "
                        "proposal output"
                    )
                mapped = mapped_values
        except Exception as error:
            return (
                None,
                audit,
                f"PROPOSAL_PROTOCOL_AUDIT_BINDING_REJECTED:{error}",
                AttemptStatus.PROPOSAL_PROTOCOL_REJECTED,
            )
        if mapped is None:
            return (
                None,
                audit,
                raw_failure or "PROPOSAL_REJECTED_WITHOUT_FAILURE_REASON",
                AttemptStatus.PROPOSAL_REJECTED,
            )
        if raw_failure is not None:
            return (
                None,
                audit,
                raw_failure,
                AttemptStatus.PROPOSAL_PROTOCOL_REJECTED,
            )
        return mapped, audit, raw_failure, None

    def _validate_persisted_sequential_policy(
        self,
        policy: Any,
        audit: Any,
    ) -> CertifiedSequentialClosurePolicy:
        if type(policy) is not CertifiedSequentialClosurePolicy:
            raise TopLevelCandidateGeneratorError(
                "stored V9 policy has an unexpected type"
            )
        if (
            _required_field(audit, "failure_reason")
            != REPRESENTATIVE_PROPOSAL_FAILURE_REASON
            or _required_field(audit, "candidate_role")
            != CANDIDATE_REPRESENTATIVE_ROLE
            or _required_field(
                audit, "candidate_exact_contact_endpoint_certified"
            )
            is not False
            or tuple(
                _required_field(
                    audit, "possible_first_contact_set_sha256"
                )
            )
            != tuple(
                row.set_sha256
                for row in policy.possible_first_contact_sets
            )
            or policy.pad_order != self.anchor_pad_names
            or policy.closing_directions_physical
            != self.v9_model_binding.closing_directions_physical
            or policy.object_geometry_sha256
            != self.v9_model_binding.object_geometry_sha256
            or policy.model_contract_sha256
            != self.v9_model_binding.model_contract_sha256
        ):
            raise TopLevelCandidateGeneratorError(
                "stored V9 policy differs from its audit/model binding"
            )
        document = json.loads(
            self.v9_model_binding.model_contract_canonical_json
        )
        try:
            expected_joint_names = tuple(
                document["hand"]["independent_joint_names"]
            )
            expected_supports = tuple(
                tuple(row)
                for row in document["closure"][
                    "independent_actuation_supports"
                ]
            )
        except (KeyError, TypeError) as error:
            raise TopLevelCandidateGeneratorError(
                "bound V9 model lacks joint/support policy evidence"
            ) from error
        if (
            policy.independent_joint_names != expected_joint_names
            or policy.independent_actuation_supports != expected_supports
        ):
            raise TopLevelCandidateGeneratorError(
                "stored V9 policy joint/support binding differs"
            )
        return policy

    def _validate_sequential_policy_result(
        self,
        *,
        result: Any,
        policy: Any,
        audit: Any,
        raw_failure: str | None,
    ) -> CertifiedSequentialClosurePolicy:
        policy = self._validate_persisted_sequential_policy(policy, audit)
        possible_sets = _required_field(result, "possible_first_contact_sets")
        display = _required_field(result, "display_only_proposal")
        if (
            raw_failure != REPRESENTATIVE_PROPOSAL_FAILURE_REASON
            or not isinstance(display, DisplayOnlyGraspProposal)
        ):
            raise TopLevelCandidateGeneratorError(
                "V9 policy requires the registered representative audit/display role"
            )
        try:
            possible_rows = tuple(possible_sets)
        except TypeError as error:
            raise TopLevelCandidateGeneratorError(
                "V9 policy possible-contact sets are not a finite sequence"
            ) from error
        if (
            policy.possible_first_contact_sets != possible_rows
            or policy.pad_order != self.anchor_pad_names
            or policy.closing_directions_physical
            != self.v9_model_binding.closing_directions_physical
            or policy.object_geometry_sha256
            != self.v9_model_binding.object_geometry_sha256
            or policy.model_contract_sha256
            != self.v9_model_binding.model_contract_sha256
            or _required_field(audit, "candidate_role")
            != CANDIDATE_REPRESENTATIVE_ROLE
            or _required_field(
                audit, "candidate_exact_contact_endpoint_certified"
            )
            is not False
            or tuple(
                _required_field(
                    audit, "possible_first_contact_set_sha256"
                )
            )
            != tuple(row.set_sha256 for row in possible_rows)
        ):
            raise TopLevelCandidateGeneratorError(
                "V9 policy differs from its contact/model/audit binding"
            )
        document = json.loads(
            self.v9_model_binding.model_contract_canonical_json
        )
        try:
            joint_names = tuple(
                document["hand"]["independent_joint_names"]
            )
            supports = tuple(
                tuple(row)
                for row in document["closure"][
                    "independent_actuation_supports"
                ]
            )
        except (KeyError, TypeError) as error:
            raise TopLevelCandidateGeneratorError(
                "bound V9 model lacks joint/support policy evidence"
            ) from error
        if (
            policy.independent_joint_names != joint_names
            or policy.independent_actuation_supports != supports
        ):
            raise TopLevelCandidateGeneratorError(
                "V9 policy joint/support binding differs from the model contract"
            )
        return policy

    def _evaluate_v9(
        self, canonical: CanonicalV9Parameters
    ) -> tuple[
        Any | None,
        Any | None,
        Any | None,
        V9InvocationAuditBinding | None,
        AttemptStatus,
        str | None,
    ]:
        try:
            result = self.v9_evaluator.evaluate_unit_parameters(
                np.asarray(canonical.values, dtype=np.float64),
                self.hand_model,
            )
        except Exception as error:  # one fail-closed invocation; never retried
            return (
                None,
                None,
                None,
                None,
                AttemptStatus.V9_EVALUATOR_EXCEPTION,
                f"V9_EVALUATOR_EXCEPTION:{type(error).__name__}:{error}",
            )
        candidate = _field(result, "candidate")
        policy_field = _field(result, "sequential_closure_policy")
        policy = None if policy_field is _MISSING else policy_field
        audit = _field(result, "audit")
        if candidate is _MISSING or audit is _MISSING or audit is None:
            return (
                None,
                None,
                None if audit is _MISSING or audit is None else audit,
                None,
                AttemptStatus.V9_PROTOCOL_REJECTED,
                "V9_PROTOCOL_REQUIRES_CANDIDATE_AND_NON_NULL_AUDIT",
            )
        try:
            if _required_field(audit, "method_id") != PRODUCTION_V9_METHOD_ID:
                raise TopLevelCandidateGeneratorError(
                    "V9 audit method_id differs from production"
                )
            if _required_field(audit, "closure_parameter_domain_id") != (
                PRODUCTION_V9_PARAMETER_DOMAIN_ID
            ):
                raise TopLevelCandidateGeneratorError(
                    "V9 audit closure_parameter_domain_id differs from "
                    "production"
                )
            if tuple(_required_field(audit, "parameter_layout")) != (
                self.v9_parameter_layout
            ):
                raise TopLevelCandidateGeneratorError(
                    "V9 audit parameter_layout differs from the bound "
                    "evaluator"
                )
            if tuple(_required_field(audit, "preshape_joint_names")) != (
                self.preshape_joint_names
            ):
                raise TopLevelCandidateGeneratorError(
                    "V9 audit preshape_joint_names differ from the bound hand"
                )
            if tuple(_required_field(audit, "pad_order")) != (
                self.anchor_pad_names
            ):
                raise TopLevelCandidateGeneratorError(
                    "V9 audit pad_order differs from the bound prepared "
                    "PAD order"
                )
            if (
                _required_field(audit, "full_verified_pad_mesh_used")
                is not True
            ):
                raise TopLevelCandidateGeneratorError(
                    "V9 audit must use the full verified PAD mesh"
                )
            if (
                _required_field(audit, "pad_face_subset_input_allowed")
                is not False
            ):
                raise TopLevelCandidateGeneratorError(
                    "V9 audit cannot allow a PAD face subset input"
                )
            current_model_binding = _validated_model_binding(
                self.v9_evaluator,
                expected_pad_names=self.anchor_pad_names,
                label="v9_evaluator current",
            )
            if current_model_binding != self.v9_model_binding:
                raise TopLevelCandidateGeneratorError(
                    "V9 evaluator model binding changed after construction"
                )
            audit_model_binding = _validated_model_binding(
                audit,
                expected_pad_names=self.anchor_pad_names,
                label="V9 audit",
            )
            if audit_model_binding != self.v9_model_binding:
                raise TopLevelCandidateGeneratorError(
                    "V9 audit model binding differs byte-for-byte from "
                    "the bound evaluator"
                )
            raw_failure_field = _required_field(audit, "failure_reason")
            raw_failure = (
                None
                if raw_failure_field is None
                else str(raw_failure_field)
            )
        except Exception as error:
            return (
                None,
                None,
                audit,
                None,
                AttemptStatus.V9_PROTOCOL_REJECTED,
                f"V9_PROTOCOL_AUDIT_BINDING_REJECTED:{error}",
            )
        invocation_binding = V9InvocationAuditBinding(
            method_id=PRODUCTION_V9_METHOD_ID,
            parameter_domain_id=PRODUCTION_V9_PARAMETER_DOMAIN_ID,
            parameter_layout=self.v9_parameter_layout,
            requested_parameters_unit=canonical.values,
            requested_parameter_key_hex=canonical.exact_key_hex,
            raw_v9_audit=audit,
        )
        if policy is not None:
            if candidate is not None:
                return (
                    None,
                    None,
                    audit,
                    invocation_binding,
                    AttemptStatus.V9_PROTOCOL_REJECTED,
                    "V9_PROTOCOL_CANDIDATE_AND_POLICY_ARE_MUTUALLY_EXCLUSIVE",
                )
            try:
                validated_policy = self._validate_sequential_policy_result(
                    result=result,
                    policy=policy,
                    audit=audit,
                    raw_failure=raw_failure,
                )
            except Exception as error:
                return (
                    None,
                    None,
                    audit,
                    invocation_binding,
                    AttemptStatus.V9_PROTOCOL_REJECTED,
                    f"V9_PROTOCOL_POLICY_BINDING_REJECTED:{error}",
                )
            return (
                None,
                validated_policy,
                audit,
                invocation_binding,
                AttemptStatus.STATIC_V9_POLICY_ACCEPTED,
                None,
            )
        if candidate is None:
            if raw_failure is None:
                return (
                    None,
                    None,
                    audit,
                    invocation_binding,
                    AttemptStatus.V9_PROTOCOL_REJECTED,
                    "V9_PROTOCOL_NULL_CANDIDATE_WITHOUT_FAILURE_REASON",
                )
            return (
                None,
                None,
                audit,
                invocation_binding,
                AttemptStatus.V9_REJECTED,
                raw_failure,
            )
        if not isinstance(candidate, GraspCandidate):
            return (
                None,
                None,
                audit,
                invocation_binding,
                AttemptStatus.V9_PROTOCOL_REJECTED,
                "V9_PROTOCOL_CANDIDATE_MUST_BE_GRASP_CANDIDATE",
            )
        if tuple(
            contact.pad_name for contact in candidate.planned_pad_contacts
        ) != self.anchor_pad_names:
            return (
                None,
                None,
                audit,
                invocation_binding,
                AttemptStatus.V9_PROTOCOL_REJECTED,
                "V9_PROTOCOL_CANDIDATE_PAD_ORDER_DIFFERS_FROM_BOUND_ORDER",
            )
        if raw_failure is not None:
            return (
                None,
                None,
                audit,
                invocation_binding,
                AttemptStatus.V9_PROTOCOL_REJECTED,
                raw_failure,
            )
        return (
            candidate,
            None,
            audit,
            invocation_binding,
            AttemptStatus.STATIC_V9_ACCEPTED,
            None,
        )

    @staticmethod
    def _frozen_evaluations(
        evaluations: Mapping[bytes, _EvaluationAccumulator],
    ) -> tuple[UniqueV9Evaluation, ...]:
        return tuple(
            UniqueV9Evaluation(
                v9_parameters_unit=row.canonical.values,
                v9_parameter_key_hex=row.canonical.exact_key_hex,
                first_attempt_index=row.first_attempt_index,
                lineage=tuple(row.lineage),
                candidate=row.candidate,
                sequential_closure_policy=row.sequential_closure_policy,
                v9_audit=row.v9_audit,
                invocation_binding=row.invocation_binding,
                status=row.status,
                v9_failure_reason=row.v9_failure_reason,
            )
            for row in evaluations.values()
        )

    def _mutable_evaluations(
        self,
        state: ResumableGenerationState,
    ) -> dict[bytes, _EvaluationAccumulator]:
        evaluations: dict[bytes, _EvaluationAccumulator] = {}
        for row in state.unique_v9_evaluations:
            canonical = canonicalize_v9_parameters(
                row.v9_parameters_unit,
                parameter_layout=self.v9_parameter_layout,
            )
            evaluations[canonical.exact_key] = _EvaluationAccumulator(
                canonical=canonical,
                first_attempt_index=row.first_attempt_index,
                lineage=list(row.lineage),
                candidate=row.candidate,
                sequential_closure_policy=(
                    row.sequential_closure_policy
                ),
                v9_audit=row.v9_audit,
                invocation_binding=row.invocation_binding,
                status=row.status,
                v9_failure_reason=row.v9_failure_reason,
            )
        return evaluations

    def _advance_one_attempt(
        self,
        *,
        attempts: list[CandidateAttemptAudit],
        evaluations: dict[bytes, _EvaluationAccumulator],
    ) -> None:
        """Commit exactly one scheduled attempt or its terminal failure."""

        attempt_index = len(attempts)
        lane_point_index, lane_ordinal = divmod(
            attempt_index, len(LANE_SPECS)
        )
        spec = LANE_SPECS[lane_ordinal]
        sobol_point = np.array(
            self._maximum_designs[spec.lane][lane_point_index],
            dtype=np.float64,
            copy=True,
        )
        anchor_name = (
            None
            if spec.anchor_pad_ordinal is None
            else self.anchor_pad_names[spec.anchor_pad_ordinal]
        )
        mapped, proposal_audit, proposal_failure, proposal_status = (
            self._proposal_for_attempt(
                spec=spec,
                sobol_point=sobol_point,
            )
        )
        lineage = CandidateLineage(
            attempt_index=attempt_index,
            lane=spec.lane,
            lane_point_index=lane_point_index,
            sobol_seed=spec.sobol_seed,
            sobol_parameters_unit=tuple(
                float(value) for value in sobol_point
            ),
            anchor_pad_name=anchor_name,
            proposal_audit=proposal_audit,
            proposal_failure_reason=proposal_failure,
        )
        if proposal_status is not None:
            attempts.append(
                CandidateAttemptAudit(
                    lineage=lineage,
                    status=proposal_status,
                    v9_parameters_unit=None,
                    v9_parameter_key_hex=None,
                    duplicate_of_attempt_index=None,
                    v9_audit=None,
                    invocation_binding=None,
                    v9_failure_reason=None,
                    failure_reason=proposal_failure,
                )
            )
            return

        if mapped is None:  # pragma: no cover - helper invariant
            raise AssertionError(
                "successful proposal has no mapped parameters"
            )
        try:
            canonical = canonicalize_v9_parameters(
                mapped,
                parameter_layout=self.v9_parameter_layout,
            )
        except TopLevelCandidateGeneratorError as error:
            reason = f"PROPOSAL_V9_DOMAIN_REJECTED:{error}"
            attempts.append(
                CandidateAttemptAudit(
                    lineage=lineage,
                    status=AttemptStatus.PROPOSAL_V9_DOMAIN_REJECTED,
                    v9_parameters_unit=None,
                    v9_parameter_key_hex=None,
                    duplicate_of_attempt_index=None,
                    v9_audit=None,
                    invocation_binding=None,
                    v9_failure_reason=None,
                    failure_reason=reason,
                )
            )
            return

        existing = evaluations.get(canonical.exact_key)
        if existing is not None:
            existing.lineage.append(lineage)
            attempts.append(
                CandidateAttemptAudit(
                    lineage=lineage,
                    status=(
                        AttemptStatus.DUPLICATE_CANONICAL_V9_PARAMETERS
                    ),
                    v9_parameters_unit=canonical.values,
                    v9_parameter_key_hex=canonical.exact_key_hex,
                    duplicate_of_attempt_index=existing.first_attempt_index,
                    v9_audit=existing.v9_audit,
                    invocation_binding=existing.invocation_binding,
                    v9_failure_reason=existing.v9_failure_reason,
                    failure_reason=existing.v9_failure_reason,
                )
            )
            return

        (
            candidate,
            sequential_closure_policy,
            v9_audit,
            invocation_binding,
            status,
            v9_failure,
        ) = self._evaluate_v9(canonical)
        evaluations[canonical.exact_key] = _EvaluationAccumulator(
            canonical=canonical,
            first_attempt_index=attempt_index,
            lineage=[lineage],
            candidate=candidate,
            sequential_closure_policy=sequential_closure_policy,
            v9_audit=v9_audit,
            invocation_binding=invocation_binding,
            status=status,
            v9_failure_reason=v9_failure,
        )
        attempts.append(
            CandidateAttemptAudit(
                lineage=lineage,
                status=status,
                v9_parameters_unit=canonical.values,
                v9_parameter_key_hex=canonical.exact_key_hex,
                duplicate_of_attempt_index=None,
                v9_audit=v9_audit,
                invocation_binding=invocation_binding,
                v9_failure_reason=v9_failure,
                failure_reason=v9_failure,
            )
        )

    def begin_resumable(
        self,
        target_total_attempt_budget: int,
    ) -> ResumableGenerationState:
        """Start an empty immutable run for one allowed frozen budget."""

        if (
            isinstance(target_total_attempt_budget, bool)
            or not isinstance(target_total_attempt_budget, int)
            or target_total_attempt_budget
            not in ALLOWED_TOTAL_ATTEMPT_BUDGETS
        ):
            raise TopLevelCandidateGeneratorError(
                "target_total_attempt_budget must be exactly one of "
                "128, 256 or 512"
            )
        state = ResumableGenerationState(
            method_id=METHOD_ID,
            contract_hash_sha256=self.contract_hash_sha256,
            target_total_attempt_budget=target_total_attempt_budget,
            attempts=(),
            unique_v9_evaluations=(),
        )
        self.validate_resumable_state(state)
        return state

    def validate_resumable_state(
        self,
        state: ResumableGenerationState,
    ) -> None:
        """Recompute schedule, canonical keys, deduplication and lineage."""

        if type(state) is not ResumableGenerationState:
            raise TopLevelCandidateGeneratorError(
                "resumable state has an unexpected type"
            )
        if (
            state.method_id != METHOD_ID
            or state.contract_hash_sha256 != self.contract_hash_sha256
            or state.target_total_attempt_budget
            not in ALLOWED_TOTAL_ATTEMPT_BUDGETS
            or len(state.attempts) > state.target_total_attempt_budget
        ):
            raise TopLevelCandidateGeneratorError(
                "resumable state differs from the bound generator contract"
            )
        unique_by_key: dict[str, UniqueV9Evaluation] = {}
        for row in state.unique_v9_evaluations:
            if type(row) is not UniqueV9Evaluation:
                raise TopLevelCandidateGeneratorError(
                    "resumable unique evaluation has an unexpected type"
                )
            canonical = canonicalize_v9_parameters(
                row.v9_parameters_unit,
                parameter_layout=self.v9_parameter_layout,
            )
            if (
                canonical.exact_key_hex != row.v9_parameter_key_hex
                or row.v9_parameter_key_hex in unique_by_key
            ):
                raise TopLevelCandidateGeneratorError(
                    "resumable unique evaluation key is not canonical/unique"
                )
            unique_by_key[row.v9_parameter_key_hex] = row

        first_key_order: list[str] = []
        lineage_by_key: dict[str, list[CandidateLineage]] = {}
        proposal_statuses = {
            AttemptStatus.PROPOSAL_REJECTED,
            AttemptStatus.PROPOSAL_PROTOCOL_REJECTED,
            AttemptStatus.PROPOSAL_V9_DOMAIN_REJECTED,
        }
        v9_statuses = {
            AttemptStatus.V9_REJECTED,
            AttemptStatus.V9_PROTOCOL_REJECTED,
            AttemptStatus.V9_EVALUATOR_EXCEPTION,
            AttemptStatus.STATIC_V9_ACCEPTED,
            AttemptStatus.STATIC_V9_POLICY_ACCEPTED,
        }
        for index, attempt in enumerate(state.attempts):
            if (
                type(attempt) is not CandidateAttemptAudit
                or type(attempt.lineage) is not CandidateLineage
            ):
                raise TopLevelCandidateGeneratorError(
                    "resumable attempt has an unexpected type"
                )
            lane_point_index, lane_ordinal = divmod(index, len(LANE_SPECS))
            spec = LANE_SPECS[lane_ordinal]
            expected_sobol = self._maximum_designs[spec.lane][
                lane_point_index
            ]
            expected_anchor = (
                None
                if spec.anchor_pad_ordinal is None
                else self.anchor_pad_names[spec.anchor_pad_ordinal]
            )
            lineage = attempt.lineage
            if (
                lineage.attempt_index != index
                or lineage.lane is not spec.lane
                or lineage.lane_point_index != lane_point_index
                or lineage.sobol_seed != spec.sobol_seed
                or lineage.anchor_pad_name != expected_anchor
                or _binary64_sequence_key(lineage.sobol_parameters_unit)
                != _binary64_sequence_key(expected_sobol)
            ):
                raise TopLevelCandidateGeneratorError(
                    "resumable attempt schedule/Sobol lineage changed"
                )
            if attempt.status in proposal_statuses:
                if (
                    attempt.v9_parameters_unit is not None
                    or attempt.v9_parameter_key_hex is not None
                    or attempt.duplicate_of_attempt_index is not None
                    or attempt.v9_audit is not None
                    or attempt.invocation_binding is not None
                    or attempt.v9_failure_reason is not None
                    or attempt.failure_reason is None
                ):
                    raise TopLevelCandidateGeneratorError(
                        "proposal failure state carries contradictory V9 data"
                    )
                continue
            if attempt.v9_parameters_unit is None:
                raise TopLevelCandidateGeneratorError(
                    "V9/duplicate attempt is missing canonical parameters"
                )
            canonical = canonicalize_v9_parameters(
                attempt.v9_parameters_unit,
                parameter_layout=self.v9_parameter_layout,
            )
            key = canonical.exact_key_hex
            if attempt.v9_parameter_key_hex != key:
                raise TopLevelCandidateGeneratorError(
                    "resumable attempt canonical V9 key changed"
                )
            if attempt.status is (
                AttemptStatus.DUPLICATE_CANONICAL_V9_PARAMETERS
            ):
                if key not in lineage_by_key:
                    raise TopLevelCandidateGeneratorError(
                        "duplicate precedes its unique V9 evaluation"
                    )
                unique = unique_by_key[key]
                if (
                    attempt.duplicate_of_attempt_index
                    != unique.first_attempt_index
                    or not _evidence_equal(
                        attempt.v9_audit, unique.v9_audit
                    )
                    or not _evidence_equal(
                        attempt.invocation_binding,
                        unique.invocation_binding,
                    )
                    or attempt.v9_failure_reason
                    != unique.v9_failure_reason
                    or attempt.failure_reason != unique.v9_failure_reason
                ):
                    raise TopLevelCandidateGeneratorError(
                        "duplicate does not preserve its first V9 result"
                    )
                lineage_by_key[key].append(lineage)
                continue
            if attempt.status not in v9_statuses or key in lineage_by_key:
                raise TopLevelCandidateGeneratorError(
                    "resumable V9 evaluation status/order is invalid"
                )
            unique = unique_by_key.get(key)
            if (
                unique is None
                or unique.first_attempt_index != index
                or unique.status is not attempt.status
                or not _evidence_equal(
                    unique.v9_audit, attempt.v9_audit
                )
                or not _evidence_equal(
                    unique.invocation_binding,
                    attempt.invocation_binding,
                )
                or unique.v9_failure_reason != attempt.v9_failure_reason
                or attempt.failure_reason != attempt.v9_failure_reason
                or attempt.duplicate_of_attempt_index is not None
            ):
                raise TopLevelCandidateGeneratorError(
                    "unique V9 row differs from its first attempt"
                )
            if attempt.status is AttemptStatus.STATIC_V9_ACCEPTED:
                if (
                    type(unique.candidate) is not GraspCandidate
                    or unique.sequential_closure_policy is not None
                    or unique.invocation_binding is None
                ):
                    raise TopLevelCandidateGeneratorError(
                        "static acceptance lacks candidate/invocation evidence"
                    )
            elif attempt.status is AttemptStatus.STATIC_V9_POLICY_ACCEPTED:
                if (
                    unique.candidate is not None
                    or unique.invocation_binding is None
                ):
                    raise TopLevelCandidateGeneratorError(
                        "static policy acceptance carries contradictory data"
                    )
                self._validate_persisted_sequential_policy(
                    unique.sequential_closure_policy,
                    unique.v9_audit,
                )
            elif (
                unique.candidate is not None
                or unique.sequential_closure_policy is not None
            ):
                raise TopLevelCandidateGeneratorError(
                    "non-accepted unique V9 row carries candidate/policy data"
                )
            first_key_order.append(key)
            lineage_by_key[key] = [lineage]

        if first_key_order != list(unique_by_key):
            raise TopLevelCandidateGeneratorError(
                "unique V9 insertion order differs from first attempts"
            )
        for key, unique in unique_by_key.items():
            if tuple(lineage_by_key.get(key, ())) != unique.lineage:
                raise TopLevelCandidateGeneratorError(
                    "unique V9 lineage does not cover every proposal"
                )

    def advance_resumable(
        self,
        state: ResumableGenerationState,
        *,
        stop_attempt_index_exclusive: int,
    ) -> ResumableGenerationState:
        """Advance a validated prefix to one explicit attempt boundary."""

        self.validate_resumable_state(state)
        start = state.completed_attempt_count
        stop = stop_attempt_index_exclusive
        if (
            isinstance(stop, bool)
            or not isinstance(stop, int)
            or stop < start
            or stop > state.target_total_attempt_budget
        ):
            raise TopLevelCandidateGeneratorError(
                "stop attempt index must be within the remaining target prefix"
            )
        attempts = list(state.attempts)
        evaluations = self._mutable_evaluations(state)
        while len(attempts) < stop:
            self._advance_one_attempt(
                attempts=attempts,
                evaluations=evaluations,
            )
        advanced = ResumableGenerationState(
            method_id=METHOD_ID,
            contract_hash_sha256=self.contract_hash_sha256,
            target_total_attempt_budget=state.target_total_attempt_budget,
            attempts=tuple(attempts),
            unique_v9_evaluations=self._frozen_evaluations(evaluations),
        )
        self.validate_resumable_state(advanced)
        return advanced

    def extend_target(
        self,
        state: ResumableGenerationState,
        new_target_total_attempt_budget: int,
    ) -> ResumableGenerationState:
        """Monotonically extend one committed prefix to a larger budget."""

        self.validate_resumable_state(state)
        if (
            isinstance(new_target_total_attempt_budget, bool)
            or not isinstance(new_target_total_attempt_budget, int)
            or new_target_total_attempt_budget
            not in ALLOWED_TOTAL_ATTEMPT_BUDGETS
            or new_target_total_attempt_budget
            < state.target_total_attempt_budget
            or new_target_total_attempt_budget
            < state.completed_attempt_count
        ):
            raise TopLevelCandidateGeneratorError(
                "resumable target may only extend through 128, 256, 512"
            )
        extended = ResumableGenerationState(
            method_id=state.method_id,
            contract_hash_sha256=state.contract_hash_sha256,
            target_total_attempt_budget=new_target_total_attempt_budget,
            attempts=state.attempts,
            unique_v9_evaluations=state.unique_v9_evaluations,
        )
        self.validate_resumable_state(extended)
        return extended

    def finalize_prefix(
        self,
        state: ResumableGenerationState,
        total_attempt_budget: int,
    ) -> TopLevelGenerationResult:
        """Materialise one complete allowed prefix without later lineage."""

        self.validate_resumable_state(state)
        if (
            isinstance(total_attempt_budget, bool)
            or not isinstance(total_attempt_budget, int)
            or total_attempt_budget not in ALLOWED_TOTAL_ATTEMPT_BUDGETS
            or total_attempt_budget > state.completed_attempt_count
        ):
            raise TopLevelCandidateGeneratorError(
                "finalized prefix must be a completed 128, 256 or 512 budget"
            )
        attempts = state.attempts[:total_attempt_budget]
        source_by_key = {
            row.v9_parameter_key_hex: row
            for row in state.unique_v9_evaluations
        }
        evaluations: dict[bytes, _EvaluationAccumulator] = {}
        for attempt in attempts:
            if attempt.v9_parameter_key_hex is None:
                continue
            key = bytes.fromhex(attempt.v9_parameter_key_hex)
            if attempt.status is (
                AttemptStatus.DUPLICATE_CANONICAL_V9_PARAMETERS
            ):
                evaluations[key].lineage.append(attempt.lineage)
                continue
            source = source_by_key[attempt.v9_parameter_key_hex]
            canonical = canonicalize_v9_parameters(
                source.v9_parameters_unit,
                parameter_layout=self.v9_parameter_layout,
            )
            evaluations[key] = _EvaluationAccumulator(
                canonical=canonical,
                first_attempt_index=source.first_attempt_index,
                lineage=[attempt.lineage],
                candidate=source.candidate,
                sequential_closure_policy=(
                    source.sequential_closure_policy
                ),
                v9_audit=source.v9_audit,
                invocation_binding=source.invocation_binding,
                status=source.status,
                v9_failure_reason=source.v9_failure_reason,
            )
        unique_evaluations = self._frozen_evaluations(evaluations)
        accepted_candidates = tuple(
            StaticV9AcceptedCandidate(
                v9_parameters_unit=row.v9_parameters_unit,
                v9_parameter_key_hex=row.v9_parameter_key_hex,
                candidate=row.candidate,
                v9_audit=row.v9_audit,
                invocation_binding=row.invocation_binding,
                lineage=row.lineage,
            )
            for row in unique_evaluations
            if row.status is AttemptStatus.STATIC_V9_ACCEPTED
            and row.invocation_binding is not None
        )
        accepted_policies = tuple(
            StaticV9AcceptedPolicy(
                v9_parameters_unit=row.v9_parameters_unit,
                v9_parameter_key_hex=row.v9_parameter_key_hex,
                sequential_closure_policy=row.sequential_closure_policy,
                v9_audit=row.v9_audit,
                invocation_binding=row.invocation_binding,
                lineage=row.lineage,
            )
            for row in unique_evaluations
            if row.status is AttemptStatus.STATIC_V9_POLICY_ACCEPTED
            and type(row.sequential_closure_policy)
            is CertifiedSequentialClosurePolicy
            and row.invocation_binding is not None
        )
        duplicate_count = sum(
            row.status is AttemptStatus.DUPLICATE_CANONICAL_V9_PARAMETERS
            for row in attempts
        )
        proposal_failure_count = sum(
            row.status
            in {
                AttemptStatus.PROPOSAL_REJECTED,
                AttemptStatus.PROPOSAL_PROTOCOL_REJECTED,
                AttemptStatus.PROPOSAL_V9_DOMAIN_REJECTED,
            }
            for row in attempts
        )
        return TopLevelGenerationResult(
            method_id=METHOD_ID,
            contract_hash_sha256=self.contract_hash_sha256,
            total_attempt_budget=total_attempt_budget,
            attempts_per_lane=total_attempt_budget // len(LANE_SPECS),
            local_refinement_evaluation_budget=(
                LOCAL_REFINEMENT_EVALUATION_BUDGET
            ),
            attempts=attempts,
            unique_v9_evaluations=unique_evaluations,
            accepted_candidates=accepted_candidates,
            accepted_policies=accepted_policies,
            v9_evaluation_count=len(unique_evaluations),
            duplicate_attempt_count=int(duplicate_count),
            proposal_failure_count=int(proposal_failure_count),
        )

    def generate(
        self,
        total_attempt_budget: int = MAIN_TOTAL_ATTEMPT_BUDGET,
    ) -> TopLevelGenerationResult:
        """Evaluate one prefix through the resumable transition engine."""

        state = self.begin_resumable(total_attempt_budget)
        state = self.advance_resumable(
            state,
            stop_attempt_index_exclusive=total_attempt_budget,
        )
        return self.finalize_prefix(state, total_attempt_budget)
