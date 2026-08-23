from dataclasses import FrozenInstanceError, replace
import hashlib

import pytest

from kcg_connector.grasp.robust.interval_policy_margin import (
    CERTIFIED_MARGIN_SEARCH_RULE,
    MAXIMUM_CERTIFICATION_ATTEMPTS,
    MAXIMUM_CERTIFICATION_ATTEMPTS_ROLE,
    METHOD_ID,
    MIDPOINT_MARGIN_PROPOSAL_ROLE,
    IntervalPolicyMarginState,
    certify_interval_policy_wrench_margin_lower_bound,
)
from kcg_connector.grasp.robust.interval_policy_wrench import (
    IntervalPolicyWrenchState,
)

from test_interval_policy_wrench import _inputs, _root_domains


def _margin_inputs(domains=None):
    inputs = _inputs(domains)
    del inputs["declared_margin"]
    inputs["evaluation_binding_sha256"] = hashlib.sha256(
        b"interval policy margin test binding"
    ).hexdigest()
    return inputs


def test_bounded_search_returns_only_complete_certified_lower_bound() -> None:
    roots = _root_domains()
    inputs = _margin_inputs(tuple((root,) for root in roots))
    first = certify_interval_policy_wrench_margin_lower_bound(**inputs)
    second = certify_interval_policy_wrench_margin_lower_bound(**inputs)

    assert first.state is IntervalPolicyMarginState.CERTIFIED_POSITIVE_LOWER_BOUND
    assert first.as_dict() == second.as_dict()
    assert first.method_id == METHOD_ID
    assert first.midpoint_margin_proposal_role == MIDPOINT_MARGIN_PROPOSAL_ROLE
    assert first.certified_margin_search_rule == CERTIFIED_MARGIN_SEARCH_RULE
    assert first.maximum_certification_attempts == MAXIMUM_CERTIFICATION_ATTEMPTS
    assert first.maximum_certification_attempts_role == (
        MAXIMUM_CERTIFICATION_ATTEMPTS_ROLE
    )
    assert len(first.evaluation_binding_sha256) == 64
    assert len(first.midpoint_margin_proposals) == 1
    assert first.initial_margin_upper_proposal is not None
    assert first.certified_margin_lower_bound is not None
    assert 0.0 < first.certified_margin_lower_bound <= first.initial_margin_upper_proposal
    assert len(first.proof_attempts) == MAXIMUM_CERTIFICATION_ATTEMPTS
    assert first.final_policy_wrench_certificate is not None
    assert first.final_policy_wrench_certificate.state is (
        IntervalPolicyWrenchState.CERTIFIED_DECLARED_MARGIN_FOR_COMPLETE_DOMAIN
    )
    assert first.final_policy_wrench_certificate.certified_margin_lower_bound == (
        first.certified_margin_lower_bound
    )
    assert not first.midpoint_margin_proposal_used_as_formal_evidence
    assert first.returned_margin_requires_complete_interval_certificate
    assert not first.physical_acceptance_threshold_used
    assert len(first.certificate_sha256) == 64
    with pytest.raises(FrozenInstanceError):
        first.reason = "forged"  # type: ignore[misc]


def test_complete_root_product_constrains_midpoint_proposal() -> None:
    inputs = _margin_inputs()

    certificate = certify_interval_policy_wrench_margin_lower_bound(**inputs)

    assert certificate.possible_root_counts == (2, 1, 2)
    assert certificate.cartesian_product_count == 4
    assert len(certificate.midpoint_margin_proposals) == 4
    assert certificate.initial_margin_upper_proposal == min(
        certificate.midpoint_margin_proposals
    )
    assert len(certificate.proof_attempts) <= MAXIMUM_CERTIFICATION_ATTEMPTS


def test_one_sided_policy_returns_no_margin_and_no_formal_proof() -> None:
    roots = tuple(
        replace(
            root,
            path_local_free_side_normal_object=(1.0, 0.0, 0.0),
        )
        for root in _root_domains()
    )
    certificate = certify_interval_policy_wrench_margin_lower_bound(
        **_margin_inputs(tuple((root,) for root in roots))
    )

    assert certificate.state is IntervalPolicyMarginState.NOT_CERTIFIABLE
    assert certificate.certified_margin_lower_bound is None
    assert certificate.final_policy_wrench_certificate is None
    assert certificate.failed_midpoint_root_combination_index == 0
    assert not certificate.proof_attempts
