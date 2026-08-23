from __future__ import annotations

import numpy as np
import pytest

from kcg_connector.grasp.robust.multifidelity_candidate_rank import (
    EVIDENCE_ROLE,
    EXACT_TOP_K,
    MultifidelityCandidateRankError,
    ProxyRankInput,
    rank_screened_candidates,
)
from kcg_connector.grasp.robust.ray_closure import WholePathPadSphereScreen


def _key(parameters: tuple[float, ...]) -> str:
    return np.asarray(parameters, dtype=">f8").tobytes(order="C").hex()


def _input(
    ordinal: int,
    roots: tuple[int, int, int],
    *,
    triangle_tests: int = 3,
    node_queries: int = 9,
    rejected: bool = False,
    skipped: bool = False,
) -> ProxyRankInput:
    parameters = (ordinal / 16.0, 0.5, 0.5, 0.5, 0.5)
    screens = tuple(
        WholePathPadSphereScreen(
            pad_name=f"pad_{index}",
            finger_name=f"finger_{index}",
            segment_count=8,
            nearest_surface_query_count=0,
            distance_bvh_node_visits=node_queries + index,
            distance_triangle_tests=triangle_tests + index,
            minimum_clearance_lower_bound_m=0.0,
            certified_free=rejected and index == 0,
            spatial_node_query_count=node_queries + index,
            obb_sat_triangle_test_count=triangle_tests + index,
            skipped_due_to_other_pad_free=skipped and index > 0,
            root_overlap_segment_count=roots[index],
        )
        for index in range(3)
    )
    return ProxyRankInput(
        canonical_key_hex=_key(parameters),
        parameters_unit=parameters,
        first_attempt_index=ordinal,
        pad_screens=screens,
    )


def test_proxy_rank_is_permutation_stable_and_hard_caps_exact_top_four() -> None:
    rows = (
        _input(1, (8, 8, 8), triangle_tests=1),
        _input(2, (8, 7, 8), triangle_tests=1),
        _input(3, (7, 7, 7), triangle_tests=1),
        _input(4, (8, 8, 8), triangle_tests=8),
        _input(5, (6, 8, 8), triangle_tests=1),
        _input(6, (8, 8, 8), triangle_tests=2),
        _input(7, (8, 8, 8), rejected=True),
    )

    forward = rank_screened_candidates(rows)
    reverse = rank_screened_candidates(tuple(reversed(rows)))

    assert forward.exact_selected_keys == reverse.exact_selected_keys
    assert len(forward.exact_selected_keys) == EXACT_TOP_K
    assert [row.proxy_rank for row in forward.ranked_survivors] == list(
        range(1, len(forward.ranked_survivors) + 1)
    )
    assert sum(
        row.selected_for_exact_v9 for row in forward.ranked_survivors
    ) == EXACT_TOP_K
    assert rows[-1].canonical_key_hex in forward.cheap_rejected_keys
    assert rows[-1].canonical_key_hex not in forward.exact_selected_keys


def test_proxy_rank_uses_three_finger_opportunity_before_cost() -> None:
    balanced = _input(1, (8, 8, 8), triangle_tests=20, node_queries=40)
    cheaper_but_unbalanced = _input(
        2, (8, 7, 8), triangle_tests=1, node_queries=1
    )

    result = rank_screened_candidates((cheaper_but_unbalanced, balanced))

    assert result.ranked_survivors[0].canonical_key_hex == (
        balanced.canonical_key_hex
    )
    document = result.as_dict()
    assert document["evidence_role"] == EVIDENCE_ROLE
    assert document["proxy_certifies_or_rejects"] is False
    assert document["formal_selected_candidate"] is None
    assert document["full_hand_collision_state"] == "NOT_CERTIFIABLE"
    assert document["dynamic_launch_allowed"] is False


def test_proxy_rank_rejects_incomplete_survivor_audit() -> None:
    row = _input(1, (8, 8, 8), skipped=True)

    with pytest.raises(
        MultifidelityCandidateRankError,
        match="complete audits",
    ):
        rank_screened_candidates((row,))
