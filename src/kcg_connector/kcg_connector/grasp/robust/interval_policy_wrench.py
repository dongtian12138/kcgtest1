"""Complete root-product and twelve-load interval wrench aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
from itertools import product
import json
import math
from typing import Sequence

import numpy as np

from kcg_connector.grasp.robust.interval_contact_balance import (
    METHOD_ID as INTERVAL_CONTACT_BALANCE_METHOD_ID,
    IntervalContactBalanceCertificate,
    IntervalContactBalanceState,
    certify_interval_contact_balance,
)
from kcg_connector.grasp.robust.interval_contact_wrench import (
    METHOD_ID as INTERVAL_CONTACT_WRENCH_METHOD_ID,
    ContactRootWrenchDomain,
    build_interval_contact_wrench_matrices,
)


METHOD_ID = "CARTS_COMPLETE_ROOT_PRODUCT_TWELVE_LOAD_INTERVAL_WRENCH_V1"
ROOT_PRODUCT_RULE = "COMPLETE_CANONICAL_CARTESIAN_PRODUCT_OF_THREE_PAD_ROOT_SETS"
LOAD_RULE = "ALL_SIX_CENTRALLY_SYMMETRIC_DISTURBANCE_PAIRS_AT_DECLARED_MARGIN"
MARGIN_ROLE = "DECLARED_POSITIVE_MARGIN_CERTIFIED_WITHOUT_SEARCH_OR_REDUCTION"


class IntervalPolicyWrenchError(ValueError):
    """Raised when a complete policy-wrench proof input is malformed."""


class IntervalPolicyWrenchState(str, Enum):
    CERTIFIED_DECLARED_MARGIN_FOR_COMPLETE_DOMAIN = (
        "CERTIFIED_DECLARED_MARGIN_FOR_COMPLETE_DOMAIN"
    )
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
class IntervalPolicyLoadCertificate:
    root_combination_index: int
    formal_root_sha256: tuple[str, str, str]
    disturbance_vertex_index: int
    external_wrench: tuple[float, float, float, float, float, float]
    interval_matrix_sha256: str
    balance_certificate: IntervalContactBalanceCertificate

    def __post_init__(self) -> None:
        if (
            not isinstance(self.root_combination_index, int)
            or isinstance(self.root_combination_index, bool)
            or self.root_combination_index < 0
            or len(self.formal_root_sha256) != 3
            or any(
                len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in self.formal_root_sha256
            )
            or not isinstance(self.disturbance_vertex_index, int)
            or isinstance(self.disturbance_vertex_index, bool)
            or not 0 <= self.disturbance_vertex_index < 12
            or len(self.external_wrench) != 6
            or not all(math.isfinite(value) for value in self.external_wrench)
            or len(self.interval_matrix_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.interval_matrix_sha256
            )
            or not isinstance(
                self.balance_certificate,
                IntervalContactBalanceCertificate,
            )
        ):
            raise IntervalPolicyWrenchError(
                "policy load certificate binding is malformed"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "root_combination_index": self.root_combination_index,
            "formal_root_sha256": list(self.formal_root_sha256),
            "disturbance_vertex_index": self.disturbance_vertex_index,
            "external_wrench_binary64_hex": [
                float(value).hex() for value in self.external_wrench
            ],
            "interval_matrix_sha256": self.interval_matrix_sha256,
            "balance_certificate": self.balance_certificate.as_dict(),
        }


@dataclass(frozen=True)
class IntervalPolicyWrenchCertificate:
    state: IntervalPolicyWrenchState
    pad_order: tuple[str, str, str]
    possible_root_counts: tuple[int, int, int]
    cartesian_product_count: int
    disturbance_vertex_count: int
    declared_margin: float
    certified_margin_lower_bound: float | None
    load_certificates: tuple[IntervalPolicyLoadCertificate, ...]
    evaluated_load_count: int
    expected_load_count: int
    complete_cartesian_product_and_loads_evaluated: bool
    failed_root_combination_index: int | None
    failed_disturbance_vertex_index: int | None
    maximum_pad_utilization_upper: float | None
    maximum_joint_torque_utilization_upper: float | None
    method_id: str
    interval_contact_wrench_method_id: str
    interval_contact_balance_method_id: str
    root_product_rule: str
    load_rule: str
    margin_role: str
    display_approximation_used_as_formal_evidence: bool
    finite_contact_geometry_sampling_used_as_formal_evidence: bool
    margin_search_or_reduction_performed: bool
    reason: str

    def __post_init__(self) -> None:
        records = tuple(self.load_certificates)
        if (
            not isinstance(self.state, IntervalPolicyWrenchState)
            or len(self.pad_order) != 3
            or len(set(self.pad_order)) != 3
            or len(self.possible_root_counts) != 3
            or any(value <= 0 for value in self.possible_root_counts)
            or self.cartesian_product_count
            != math.prod(self.possible_root_counts)
            or self.disturbance_vertex_count != 12
            or not math.isfinite(self.declared_margin)
            or self.declared_margin <= 0.0
            or self.evaluated_load_count != len(records)
            or self.expected_load_count != 12 * self.cartesian_product_count
            or self.evaluated_load_count > self.expected_load_count
            or self.method_id != METHOD_ID
            or self.interval_contact_wrench_method_id
            != INTERVAL_CONTACT_WRENCH_METHOD_ID
            or self.interval_contact_balance_method_id
            != INTERVAL_CONTACT_BALANCE_METHOD_ID
            or self.root_product_rule != ROOT_PRODUCT_RULE
            or self.load_rule != LOAD_RULE
            or self.margin_role != MARGIN_ROLE
            or self.display_approximation_used_as_formal_evidence
            or self.finite_contact_geometry_sampling_used_as_formal_evidence
            or self.margin_search_or_reduction_performed
            or not self.reason
        ):
            raise IntervalPolicyWrenchError(
                "interval policy-wrench certificate provenance is malformed"
            )
        if self.state is (
            IntervalPolicyWrenchState.CERTIFIED_DECLARED_MARGIN_FOR_COMPLETE_DOMAIN
        ):
            if (
                self.certified_margin_lower_bound != self.declared_margin
                or self.evaluated_load_count != self.expected_load_count
                or not self.complete_cartesian_product_and_loads_evaluated
                or self.failed_root_combination_index is not None
                or self.failed_disturbance_vertex_index is not None
                or self.maximum_pad_utilization_upper is None
                or not 0.0 <= self.maximum_pad_utilization_upper <= 1.0
                or self.maximum_joint_torque_utilization_upper is None
                or not 0.0
                <= self.maximum_joint_torque_utilization_upper
                <= 1.0
                or any(
                    record.balance_certificate.state
                    is not IntervalContactBalanceState.CERTIFIED_ALL_INTERVAL_STATES_BALANCED
                    for record in records
                )
            ):
                raise IntervalPolicyWrenchError(
                    "certified policy margin is incomplete"
                )
        elif (
            self.certified_margin_lower_bound is not None
            or self.complete_cartesian_product_and_loads_evaluated
            or self.failed_root_combination_index is None
            or self.failed_disturbance_vertex_index is None
            or self.maximum_pad_utilization_upper is not None
            or self.maximum_joint_torque_utilization_upper is not None
            or not records
            or records[-1].balance_certificate.state
            is not IntervalContactBalanceState.NOT_CERTIFIABLE
        ):
            raise IntervalPolicyWrenchError(
                "uncertified policy margin overclaims its partial evaluation"
            )
        object.__setattr__(self, "load_certificates", records)

    @property
    def certificate_sha256(self) -> str:
        return hashlib.sha256(
            _canonical_json(self._evidence_dict()).encode("utf-8")
        ).hexdigest()

    def _evidence_dict(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "pad_order": list(self.pad_order),
            "possible_root_counts": list(self.possible_root_counts),
            "cartesian_product_count": self.cartesian_product_count,
            "disturbance_vertex_count": self.disturbance_vertex_count,
            "declared_margin_binary64_hex": float(self.declared_margin).hex(),
            "certified_margin_lower_bound_binary64_hex": (
                None
                if self.certified_margin_lower_bound is None
                else float(self.certified_margin_lower_bound).hex()
            ),
            "load_certificates": [row.as_dict() for row in self.load_certificates],
            "evaluated_load_count": self.evaluated_load_count,
            "expected_load_count": self.expected_load_count,
            "complete_cartesian_product_and_loads_evaluated": (
                self.complete_cartesian_product_and_loads_evaluated
            ),
            "failed_root_combination_index": (
                self.failed_root_combination_index
            ),
            "failed_disturbance_vertex_index": (
                self.failed_disturbance_vertex_index
            ),
            "maximum_pad_utilization_upper_binary64_hex": (
                None
                if self.maximum_pad_utilization_upper is None
                else float(self.maximum_pad_utilization_upper).hex()
            ),
            "maximum_joint_torque_utilization_upper_binary64_hex": (
                None
                if self.maximum_joint_torque_utilization_upper is None
                else float(self.maximum_joint_torque_utilization_upper).hex()
            ),
            "method_id": self.method_id,
            "interval_contact_wrench_method_id": (
                self.interval_contact_wrench_method_id
            ),
            "interval_contact_balance_method_id": (
                self.interval_contact_balance_method_id
            ),
            "root_product_rule": self.root_product_rule,
            "load_rule": self.load_rule,
            "margin_role": self.margin_role,
            "display_approximation_used_as_formal_evidence": (
                self.display_approximation_used_as_formal_evidence
            ),
            "finite_contact_geometry_sampling_used_as_formal_evidence": (
                self.finite_contact_geometry_sampling_used_as_formal_evidence
            ),
            "margin_search_or_reduction_performed": (
                self.margin_search_or_reduction_performed
            ),
            "reason": self.reason,
        }

    def as_dict(self) -> dict[str, object]:
        result = self._evidence_dict()
        result["certificate_sha256"] = self.certificate_sha256
        return result


def certify_declared_interval_policy_wrench_margin(
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
    declared_margin: float,
) -> IntervalPolicyWrenchCertificate:
    """Certify one positive margin without changing or searching its value."""

    domains = tuple(tuple(domain) for domain in pad_root_domains)
    names = tuple(pad_order)
    caps = tuple(float(value) for value in normal_force_caps_n)
    efforts = tuple(float(value) for value in joint_effort_limits)
    nominal = np.asarray(nominal_external_wrench, dtype=np.float64)
    vertices = np.asarray(disturbance_vertices, dtype=np.float64)
    margin = float(declared_margin)
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
        or not math.isfinite(margin)
        or margin <= 0.0
    ):
        raise IntervalPolicyWrenchError(
            "policy margin needs three canonical root sets, positive limits, one nominal wrench, six nonzero +/- disturbance pairs and a positive declared margin"
        )

    counts = tuple(len(domain) for domain in domains)
    combination_count = math.prod(counts)
    records: list[IntervalPolicyLoadCertificate] = []
    pad_utilizations: list[float] = []
    joint_utilizations: list[float] = []
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
            raise IntervalPolicyWrenchError(
                "joint effort limits do not match interval Jacobian order"
            )
        for vertex_index, vertex in enumerate(vertices):
            external = nominal + margin * vertex
            balance = certify_interval_contact_balance(
                grasp_matrix_intervals=matrices.grasp_matrix_intervals,
                joint_torque_from_ray_intervals=(
                    matrices.joint_torque_from_ray_intervals
                ),
                ray_owner=matrices.ray_owner,
                normal_force_caps_n=matrices.normal_force_caps_n,
                joint_effort_limits=efforts,
                external_wrench=external,
                decimal_precision=matrices.decimal_precision,
            )
            record = IntervalPolicyLoadCertificate(
                root_combination_index=combination_index,
                formal_root_sha256=matrices.formal_root_sha256,
                disturbance_vertex_index=vertex_index,
                external_wrench=tuple(float(value) for value in external),  # type: ignore[arg-type]
                interval_matrix_sha256=matrices.matrix_sha256,
                balance_certificate=balance,
            )
            records.append(record)
            if balance.state is IntervalContactBalanceState.NOT_CERTIFIABLE:
                return IntervalPolicyWrenchCertificate(
                    state=IntervalPolicyWrenchState.NOT_CERTIFIABLE,
                    pad_order=names,  # type: ignore[arg-type]
                    possible_root_counts=counts,  # type: ignore[arg-type]
                    cartesian_product_count=combination_count,
                    disturbance_vertex_count=12,
                    declared_margin=margin,
                    certified_margin_lower_bound=None,
                    load_certificates=tuple(records),
                    evaluated_load_count=len(records),
                    expected_load_count=12 * combination_count,
                    complete_cartesian_product_and_loads_evaluated=False,
                    failed_root_combination_index=combination_index,
                    failed_disturbance_vertex_index=vertex_index,
                    maximum_pad_utilization_upper=None,
                    maximum_joint_torque_utilization_upper=None,
                    method_id=METHOD_ID,
                    interval_contact_wrench_method_id=(
                        INTERVAL_CONTACT_WRENCH_METHOD_ID
                    ),
                    interval_contact_balance_method_id=(
                        INTERVAL_CONTACT_BALANCE_METHOD_ID
                    ),
                    root_product_rule=ROOT_PRODUCT_RULE,
                    load_rule=LOAD_RULE,
                    margin_role=MARGIN_ROLE,
                    display_approximation_used_as_formal_evidence=False,
                    finite_contact_geometry_sampling_used_as_formal_evidence=False,
                    margin_search_or_reduction_performed=False,
                    reason=f"LOAD_NOT_CERTIFIED:{balance.reason}",
                )
            assert balance.maximum_pad_utilization_upper is not None
            assert balance.maximum_joint_torque_utilization_upper is not None
            pad_utilizations.append(balance.maximum_pad_utilization_upper)
            joint_utilizations.append(
                balance.maximum_joint_torque_utilization_upper
            )

    return IntervalPolicyWrenchCertificate(
        state=(
            IntervalPolicyWrenchState.CERTIFIED_DECLARED_MARGIN_FOR_COMPLETE_DOMAIN
        ),
        pad_order=names,  # type: ignore[arg-type]
        possible_root_counts=counts,  # type: ignore[arg-type]
        cartesian_product_count=combination_count,
        disturbance_vertex_count=12,
        declared_margin=margin,
        certified_margin_lower_bound=margin,
        load_certificates=tuple(records),
        evaluated_load_count=len(records),
        expected_load_count=12 * combination_count,
        complete_cartesian_product_and_loads_evaluated=True,
        failed_root_combination_index=None,
        failed_disturbance_vertex_index=None,
        maximum_pad_utilization_upper=max(pad_utilizations),
        maximum_joint_torque_utilization_upper=max(joint_utilizations),
        method_id=METHOD_ID,
        interval_contact_wrench_method_id=INTERVAL_CONTACT_WRENCH_METHOD_ID,
        interval_contact_balance_method_id=INTERVAL_CONTACT_BALANCE_METHOD_ID,
        root_product_rule=ROOT_PRODUCT_RULE,
        load_rule=LOAD_RULE,
        margin_role=MARGIN_ROLE,
        display_approximation_used_as_formal_evidence=False,
        finite_contact_geometry_sampling_used_as_formal_evidence=False,
        margin_search_or_reduction_performed=False,
        reason=(
            "DECLARED_MARGIN_CERTIFIED_FOR_COMPLETE_ROOT_PRODUCT_AND_TWELVE_LOADS"
        ),
    )


__all__ = [
    "LOAD_RULE",
    "MARGIN_ROLE",
    "METHOD_ID",
    "ROOT_PRODUCT_RULE",
    "IntervalPolicyLoadCertificate",
    "IntervalPolicyWrenchCertificate",
    "IntervalPolicyWrenchError",
    "IntervalPolicyWrenchState",
    "certify_declared_interval_policy_wrench_margin",
]
