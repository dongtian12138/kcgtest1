"""Bounded proposal-and-proof search for a positive interval wrench margin."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
from itertools import product
import json
import math
from typing import Sequence

import numpy as np
from scipy.optimize import linprog

from kcg_connector.grasp.robust.interval_contact_wrench import (
    ContactRootWrenchDomain,
    build_interval_contact_wrench_matrices,
)
from kcg_connector.grasp.robust.interval_policy_wrench import (
    METHOD_ID as INTERVAL_POLICY_WRENCH_METHOD_ID,
    IntervalPolicyWrenchCertificate,
    IntervalPolicyWrenchState,
    certify_declared_interval_policy_wrench_margin,
)


METHOD_ID = "CARTS_BOUNDED_DYADIC_INTERVAL_POLICY_MARGIN_SEARCH_V1"
MIDPOINT_MARGIN_PROPOSAL_ROLE = (
    "COMPLETE_ROOT_PRODUCT_INTERVAL_MATRIX_MIDPOINT_LP_UPPER_PROPOSAL_ONLY"
)
CERTIFIED_MARGIN_SEARCH_RULE = (
    "HALVE_UNTIL_FIRST_COMPLETE_CERTIFICATE_THEN_BISECT_KEEP_HIGHEST_CERTIFIED"
)
MAXIMUM_CERTIFICATION_ATTEMPTS = 12
MAXIMUM_CERTIFICATION_ATTEMPTS_ROLE = (
    "PRE_REGISTERED_COMPUTE_BUDGET_NOT_PHYSICAL_ACCEPTANCE_THRESHOLD"
)
MIDPOINT_PROPOSAL_METHOD_ID = (
    "SCIPY_HIGHS_COMPLETE_ROOT_PRODUCT_MIDPOINT_COMMON_MARGIN_PROPOSAL_V1"
)


class IntervalPolicyMarginError(ValueError):
    """Raised when interval-margin optimization inputs are malformed."""


class IntervalPolicyMarginState(str, Enum):
    CERTIFIED_POSITIVE_LOWER_BOUND = "CERTIFIED_POSITIVE_LOWER_BOUND"
    NOT_CERTIFIABLE = "NOT_CERTIFIABLE"


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


@dataclass(frozen=True)
class IntervalMarginProofAttempt:
    attempt_index: int
    proposed_margin: float
    proof_state: IntervalPolicyWrenchState
    proof_certificate_sha256: str
    evaluated_load_count: int
    expected_load_count: int
    reason: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.attempt_index, int)
            or isinstance(self.attempt_index, bool)
            or self.attempt_index < 0
            or not math.isfinite(self.proposed_margin)
            or self.proposed_margin <= 0.0
            or not isinstance(self.proof_state, IntervalPolicyWrenchState)
            or len(self.proof_certificate_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.proof_certificate_sha256
            )
            or self.evaluated_load_count <= 0
            or self.expected_load_count < self.evaluated_load_count
            or not self.reason
        ):
            raise IntervalPolicyMarginError(
                "interval margin proof-attempt record is malformed"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "attempt_index": self.attempt_index,
            "proposed_margin_binary64_hex": float(self.proposed_margin).hex(),
            "proof_state": self.proof_state.value,
            "proof_certificate_sha256": self.proof_certificate_sha256,
            "evaluated_load_count": self.evaluated_load_count,
            "expected_load_count": self.expected_load_count,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class IntervalPolicyMarginCertificate:
    state: IntervalPolicyMarginState
    possible_root_counts: tuple[int, int, int]
    cartesian_product_count: int
    evaluation_binding_sha256: str
    midpoint_margin_proposals: tuple[float, ...]
    initial_margin_upper_proposal: float | None
    proof_attempts: tuple[IntervalMarginProofAttempt, ...]
    certified_margin_lower_bound: float | None
    final_policy_wrench_certificate: IntervalPolicyWrenchCertificate | None
    failed_midpoint_root_combination_index: int | None
    method_id: str
    midpoint_proposal_method_id: str
    midpoint_margin_proposal_role: str
    certified_margin_search_rule: str
    maximum_certification_attempts: int
    maximum_certification_attempts_role: str
    midpoint_margin_proposal_used_as_formal_evidence: bool
    returned_margin_requires_complete_interval_certificate: bool
    physical_acceptance_threshold_used: bool
    reason: str

    def __post_init__(self) -> None:
        proposals = tuple(float(value) for value in self.midpoint_margin_proposals)
        attempts = tuple(self.proof_attempts)
        if (
            not isinstance(self.state, IntervalPolicyMarginState)
            or len(self.possible_root_counts) != 3
            or any(value <= 0 for value in self.possible_root_counts)
            or self.cartesian_product_count
            != math.prod(self.possible_root_counts)
            or len(self.evaluation_binding_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.evaluation_binding_sha256
            )
            or len(proposals) > self.cartesian_product_count
            or not all(math.isfinite(value) and value > 0.0 for value in proposals)
            or len(attempts) > MAXIMUM_CERTIFICATION_ATTEMPTS
            or tuple(row.attempt_index for row in attempts)
            != tuple(range(len(attempts)))
            or self.method_id != METHOD_ID
            or self.midpoint_proposal_method_id
            != MIDPOINT_PROPOSAL_METHOD_ID
            or self.midpoint_margin_proposal_role
            != MIDPOINT_MARGIN_PROPOSAL_ROLE
            or self.certified_margin_search_rule
            != CERTIFIED_MARGIN_SEARCH_RULE
            or self.maximum_certification_attempts
            != MAXIMUM_CERTIFICATION_ATTEMPTS
            or self.maximum_certification_attempts_role
            != MAXIMUM_CERTIFICATION_ATTEMPTS_ROLE
            or self.midpoint_margin_proposal_used_as_formal_evidence
            or not self.returned_margin_requires_complete_interval_certificate
            or self.physical_acceptance_threshold_used
            or not self.reason
        ):
            raise IntervalPolicyMarginError(
                "interval policy-margin certificate provenance is malformed"
            )
        if self.initial_margin_upper_proposal is not None and (
            not math.isfinite(self.initial_margin_upper_proposal)
            or self.initial_margin_upper_proposal <= 0.0
            or self.initial_margin_upper_proposal != min(proposals)
        ):
            raise IntervalPolicyMarginError(
                "initial margin proposal must be the complete-product minimum"
            )
        if self.state is IntervalPolicyMarginState.CERTIFIED_POSITIVE_LOWER_BOUND:
            final = self.final_policy_wrench_certificate
            if (
                len(proposals) != self.cartesian_product_count
                or self.initial_margin_upper_proposal is None
                or not attempts
                or self.certified_margin_lower_bound is None
                or not math.isfinite(self.certified_margin_lower_bound)
                or self.certified_margin_lower_bound <= 0.0
                or self.certified_margin_lower_bound
                > self.initial_margin_upper_proposal
                or final is None
                or final.state
                is not IntervalPolicyWrenchState.CERTIFIED_DECLARED_MARGIN_FOR_COMPLETE_DOMAIN
                or final.certified_margin_lower_bound
                != self.certified_margin_lower_bound
                or self.failed_midpoint_root_combination_index is not None
                or not any(
                    row.proposed_margin == self.certified_margin_lower_bound
                    and row.proof_state
                    is IntervalPolicyWrenchState.CERTIFIED_DECLARED_MARGIN_FOR_COMPLETE_DOMAIN
                    for row in attempts
                )
            ):
                raise IntervalPolicyMarginError(
                    "certified interval margin lacks its complete final proof"
                )
        elif (
            self.certified_margin_lower_bound is not None
            or self.final_policy_wrench_certificate is not None
            or (
                self.failed_midpoint_root_combination_index is not None
                and attempts
            )
            or (
                self.failed_midpoint_root_combination_index is None
                and len(proposals) != self.cartesian_product_count
            )
        ):
            raise IntervalPolicyMarginError(
                "uncertified interval margin overclaims a lower bound"
            )
        object.__setattr__(self, "midpoint_margin_proposals", proposals)
        object.__setattr__(self, "proof_attempts", attempts)

    @property
    def certificate_sha256(self) -> str:
        return hashlib.sha256(
            _canonical_json(self._evidence_dict()).encode("utf-8")
        ).hexdigest()

    def _evidence_dict(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "possible_root_counts": list(self.possible_root_counts),
            "cartesian_product_count": self.cartesian_product_count,
            "evaluation_binding_sha256": self.evaluation_binding_sha256,
            "midpoint_margin_proposals_binary64_hex": [
                float(value).hex() for value in self.midpoint_margin_proposals
            ],
            "initial_margin_upper_proposal_binary64_hex": (
                None
                if self.initial_margin_upper_proposal is None
                else float(self.initial_margin_upper_proposal).hex()
            ),
            "proof_attempts": [row.as_dict() for row in self.proof_attempts],
            "certified_margin_lower_bound_binary64_hex": (
                None
                if self.certified_margin_lower_bound is None
                else float(self.certified_margin_lower_bound).hex()
            ),
            "final_policy_wrench_certificate": (
                None
                if self.final_policy_wrench_certificate is None
                else self.final_policy_wrench_certificate.as_dict()
            ),
            "failed_midpoint_root_combination_index": (
                self.failed_midpoint_root_combination_index
            ),
            "method_id": self.method_id,
            "midpoint_proposal_method_id": self.midpoint_proposal_method_id,
            "midpoint_margin_proposal_role": (
                self.midpoint_margin_proposal_role
            ),
            "certified_margin_search_rule": self.certified_margin_search_rule,
            "maximum_certification_attempts": (
                self.maximum_certification_attempts
            ),
            "maximum_certification_attempts_role": (
                self.maximum_certification_attempts_role
            ),
            "midpoint_margin_proposal_used_as_formal_evidence": (
                self.midpoint_margin_proposal_used_as_formal_evidence
            ),
            "returned_margin_requires_complete_interval_certificate": (
                self.returned_margin_requires_complete_interval_certificate
            ),
            "physical_acceptance_threshold_used": (
                self.physical_acceptance_threshold_used
            ),
            "reason": self.reason,
        }

    def as_dict(self) -> dict[str, object]:
        result = self._evidence_dict()
        result["certificate_sha256"] = self.certificate_sha256
        return result


def _midpoint_margin_proposal(
    *,
    grasp_matrix: np.ndarray,
    joint_torque_matrix: np.ndarray,
    ray_owner: tuple[int, ...],
    normal_force_caps_n: tuple[float, ...],
    joint_effort_limits: tuple[float, ...],
    nominal_external_wrench: np.ndarray,
    disturbance_vertices: np.ndarray,
) -> float | None:
    ray_count = len(ray_owner)
    load_count = len(disturbance_vertices)
    variable_count = load_count * ray_count + 1
    objective = np.zeros(variable_count, dtype=np.float64)
    objective[-1] = -1.0
    equality = np.zeros((6 * load_count, variable_count), dtype=np.float64)
    equality_bounds = np.tile(-nominal_external_wrench, load_count)
    owner_matrix = np.zeros((3, ray_count), dtype=np.float64)
    for ray_index, owner in enumerate(ray_owner):
        owner_matrix[owner, ray_index] = 1.0
    per_load_inequality = np.vstack(
        (owner_matrix, joint_torque_matrix, -joint_torque_matrix)
    )
    per_load_bounds = np.concatenate(
        (
            np.asarray(normal_force_caps_n),
            np.asarray(joint_effort_limits),
            np.asarray(joint_effort_limits),
        )
    )
    inequality = np.zeros(
        (load_count * len(per_load_bounds), variable_count),
        dtype=np.float64,
    )
    inequality_bounds = np.tile(per_load_bounds, load_count)
    for load_index, vertex in enumerate(disturbance_vertices):
        column_start = load_index * ray_count
        column_stop = column_start + ray_count
        row_start = 6 * load_index
        equality[row_start : row_start + 6, column_start:column_stop] = (
            grasp_matrix
        )
        equality[row_start : row_start + 6, -1] = vertex
        inequality_start = load_index * len(per_load_bounds)
        inequality[
            inequality_start : inequality_start + len(per_load_bounds),
            column_start:column_stop,
        ] = per_load_inequality
    result = linprog(
        objective,
        A_ub=inequality,
        b_ub=inequality_bounds,
        A_eq=equality,
        b_eq=equality_bounds,
        bounds=tuple((0.0, None) for _ in range(variable_count)),
        method="highs",
    )
    if (
        not result.success
        or result.x is None
        or result.x.shape != (variable_count,)
        or not np.all(np.isfinite(result.x))
        or float(result.x[-1]) <= 0.0
    ):
        return None
    return float(result.x[-1])


def certify_interval_policy_wrench_margin_lower_bound(
    *,
    pad_root_domains: Sequence[Sequence[ContactRootWrenchDomain]],
    pad_order: Sequence[str],
    normal_force_caps_n: Sequence[float],
    joint_effort_limits: Sequence[float],
    wrench_origin_object_m: Sequence[float],
    task_frame_rotation_object: Sequence[Sequence[float]],
    hard_bound_friction_coefficient: float,
    maximum_inner_approximation_relative_error: float,
    cone_edge_multiplier: int,
    nominal_external_wrench: Sequence[float],
    disturbance_vertices: Sequence[Sequence[float]],
    evaluation_binding_sha256: str,
) -> IntervalPolicyMarginCertificate:
    """Return only a margin carrying a complete directed interval proof."""

    domains = tuple(tuple(domain) for domain in pad_root_domains)
    names = tuple(pad_order)
    caps = tuple(float(value) for value in normal_force_caps_n)
    efforts = tuple(float(value) for value in joint_effort_limits)
    nominal = np.asarray(nominal_external_wrench, dtype=np.float64)
    vertices = np.asarray(disturbance_vertices, dtype=np.float64)
    if (
        len(domains) != 3
        or len(names) != 3
        or len(set(names)) != 3
        or any(not domain for domain in domains)
        or any(
            any(root.pad_name != name for root in domain)
            for name, domain in zip(names, domains)
        )
        or any(
            tuple(root.formal_root_sha256 for root in domain)
            != tuple(sorted(root.formal_root_sha256 for root in domain))
            or len({root.formal_root_sha256 for root in domain}) != len(domain)
            for domain in domains
        )
        or len(caps) != 3
        or not all(math.isfinite(value) and value > 0.0 for value in caps)
        or not efforts
        or not all(math.isfinite(value) and value > 0.0 for value in efforts)
        or nominal.shape != (6,)
        or not np.all(np.isfinite(nominal))
        or vertices.shape != (12, 6)
        or not np.all(np.isfinite(vertices))
        or not np.array_equal(vertices[0::2], -vertices[1::2])
        or np.linalg.matrix_rank(vertices[0::2]) != 6
        or len(evaluation_binding_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in evaluation_binding_sha256
        )
    ):
        raise IntervalPolicyMarginError(
            "margin search needs three canonical root sets, positive limits, nominal wrench and six +/- disturbance pairs"
        )
    counts = tuple(len(domain) for domain in domains)
    combination_count = math.prod(counts)

    def result(
        *,
        state: IntervalPolicyMarginState,
        proposals: tuple[float, ...],
        initial: float | None,
        attempts: tuple[IntervalMarginProofAttempt, ...],
        certified: float | None,
        final: IntervalPolicyWrenchCertificate | None,
        failed_midpoint: int | None,
        reason: str,
    ) -> IntervalPolicyMarginCertificate:
        return IntervalPolicyMarginCertificate(
            state=state,
            possible_root_counts=counts,  # type: ignore[arg-type]
            cartesian_product_count=combination_count,
            evaluation_binding_sha256=evaluation_binding_sha256,
            midpoint_margin_proposals=proposals,
            initial_margin_upper_proposal=initial,
            proof_attempts=attempts,
            certified_margin_lower_bound=certified,
            final_policy_wrench_certificate=final,
            failed_midpoint_root_combination_index=failed_midpoint,
            method_id=METHOD_ID,
            midpoint_proposal_method_id=MIDPOINT_PROPOSAL_METHOD_ID,
            midpoint_margin_proposal_role=MIDPOINT_MARGIN_PROPOSAL_ROLE,
            certified_margin_search_rule=CERTIFIED_MARGIN_SEARCH_RULE,
            maximum_certification_attempts=MAXIMUM_CERTIFICATION_ATTEMPTS,
            maximum_certification_attempts_role=(
                MAXIMUM_CERTIFICATION_ATTEMPTS_ROLE
            ),
            midpoint_margin_proposal_used_as_formal_evidence=False,
            returned_margin_requires_complete_interval_certificate=True,
            physical_acceptance_threshold_used=False,
            reason=reason,
        )

    proposals: list[float] = []
    for combination_index, roots in enumerate(product(*domains)):
        matrices = build_interval_contact_wrench_matrices(
            root_domains=roots,
            pad_order=names,
            normal_force_caps_n=caps,
            wrench_origin_object_m=wrench_origin_object_m,
            task_frame_rotation_object=task_frame_rotation_object,
            hard_bound_friction_coefficient=hard_bound_friction_coefficient,
            maximum_inner_approximation_relative_error=(
                maximum_inner_approximation_relative_error
            ),
            cone_edge_multiplier=cone_edge_multiplier,
        )
        if len(matrices.independent_joint_names) != len(efforts):
            raise IntervalPolicyMarginError(
                "joint effort limits do not match interval Jacobian order"
            )
        grasp_midpoint = np.asarray(
            [
                [
                    bounds.lower + 0.5 * (bounds.upper - bounds.lower)
                    for bounds in row
                ]
                for row in matrices.grasp_matrix_intervals
            ]
        )
        torque_midpoint = np.asarray(
            [
                [
                    bounds.lower + 0.5 * (bounds.upper - bounds.lower)
                    for bounds in row
                ]
                for row in matrices.joint_torque_from_ray_intervals
            ]
        )
        proposal = _midpoint_margin_proposal(
            grasp_matrix=grasp_midpoint,
            joint_torque_matrix=torque_midpoint,
            ray_owner=matrices.ray_owner,
            normal_force_caps_n=matrices.normal_force_caps_n,
            joint_effort_limits=efforts,
            nominal_external_wrench=nominal,
            disturbance_vertices=vertices,
        )
        if proposal is None:
            return result(
                state=IntervalPolicyMarginState.NOT_CERTIFIABLE,
                proposals=tuple(proposals),
                initial=None,
                attempts=(),
                certified=None,
                final=None,
                failed_midpoint=combination_index,
                reason="MIDPOINT_MARGIN_PROPOSAL_INFEASIBLE",
            )
        proposals.append(proposal)

    initial = min(proposals)
    high = initial
    low = 0.0
    best: IntervalPolicyWrenchCertificate | None = None
    attempts: list[IntervalMarginProofAttempt] = []
    for attempt_index in range(MAXIMUM_CERTIFICATION_ATTEMPTS):
        candidate = high * 0.5 if best is None else low + 0.5 * (high - low)
        if not math.isfinite(candidate) or candidate <= 0.0 or candidate in (low, high):
            break
        proof = certify_declared_interval_policy_wrench_margin(
            pad_root_domains=domains,
            pad_order=names,
            normal_force_caps_n=caps,
            joint_effort_limits=efforts,
            wrench_origin_object_m=wrench_origin_object_m,
            task_frame_rotation_object=task_frame_rotation_object,
            hard_bound_friction_coefficient=hard_bound_friction_coefficient,
            maximum_inner_approximation_relative_error=(
                maximum_inner_approximation_relative_error
            ),
            cone_edge_multiplier=cone_edge_multiplier,
            nominal_external_wrench=nominal,
            disturbance_vertices=vertices,
            declared_margin=candidate,
        )
        attempts.append(
            IntervalMarginProofAttempt(
                attempt_index=attempt_index,
                proposed_margin=candidate,
                proof_state=proof.state,
                proof_certificate_sha256=proof.certificate_sha256,
                evaluated_load_count=proof.evaluated_load_count,
                expected_load_count=proof.expected_load_count,
                reason=proof.reason,
            )
        )
        if proof.state is (
            IntervalPolicyWrenchState.CERTIFIED_DECLARED_MARGIN_FOR_COMPLETE_DOMAIN
        ):
            best = proof
            low = candidate
        else:
            high = candidate

    if best is None:
        return result(
            state=IntervalPolicyMarginState.NOT_CERTIFIABLE,
            proposals=tuple(proposals),
            initial=initial,
            attempts=tuple(attempts),
            certified=None,
            final=None,
            failed_midpoint=None,
            reason="NO_POSITIVE_MARGIN_RECEIVED_A_COMPLETE_INTERVAL_CERTIFICATE",
        )
    assert best.certified_margin_lower_bound is not None
    return result(
        state=IntervalPolicyMarginState.CERTIFIED_POSITIVE_LOWER_BOUND,
        proposals=tuple(proposals),
        initial=initial,
        attempts=tuple(attempts),
        certified=best.certified_margin_lower_bound,
        final=best,
        failed_midpoint=None,
        reason="HIGHEST_ATTEMPTED_COMPLETE_INTERVAL_CERTIFICATE_RETAINED",
    )


__all__ = [
    "CERTIFIED_MARGIN_SEARCH_RULE",
    "MAXIMUM_CERTIFICATION_ATTEMPTS",
    "MAXIMUM_CERTIFICATION_ATTEMPTS_ROLE",
    "METHOD_ID",
    "MIDPOINT_MARGIN_PROPOSAL_ROLE",
    "MIDPOINT_PROPOSAL_METHOD_ID",
    "IntervalMarginProofAttempt",
    "IntervalPolicyMarginCertificate",
    "IntervalPolicyMarginError",
    "IntervalPolicyMarginState",
    "certify_interval_policy_wrench_margin_lower_bound",
]
