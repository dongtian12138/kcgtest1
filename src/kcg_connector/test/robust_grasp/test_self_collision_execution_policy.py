from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

import test_candidate_route as route_fixtures

from kcg_connector.grasp.robust.self_collision_execution_policy import (
    EXPECTED_BASE_PAIR_COUNT,
    EXPECTED_FORBIDDEN_PAIR_COUNT,
    EXPECTED_STRUCTURAL_INTERFACE_PAIR_COUNT,
    build_self_collision_execution_policy,
)


real_route_inputs = route_fixtures.real_route_inputs


def test_policy_excludes_only_direct_hash_bound_urdf_parent_child_pairs(
    real_route_inputs,
) -> None:
    aggregate = real_route_inputs[route_fixtures.CURRENT_OBJECT][0]
    policy = build_self_collision_execution_policy(
        kinematic_binding=aggregate.kinematic_binding,
        base_inventory=aggregate.self_pair_inventory,
    )
    collision_links = set(policy.link_names)
    expected_structural = tuple(
        sorted(
            {
                tuple(sorted((joint.parent_link, joint.child_link)))
                for joint in aggregate.kinematic_binding.model.joints.values()
                if joint.parent_link in collision_links
                and joint.child_link in collision_links
            }
        )
    )

    assert policy.all_pair_count == EXPECTED_BASE_PAIR_COUNT == 136
    assert policy.structural_interface_pair_count == (
        EXPECTED_STRUCTURAL_INTERFACE_PAIR_COUNT
    ) == 15
    assert policy.forbidden_pair_count == EXPECTED_FORBIDDEN_PAIR_COUNT == 121
    assert policy.structural_interface_pairs == expected_structural
    assert set(policy.structural_interface_pairs).isdisjoint(
        policy.forbidden_pairs
    )
    assert set(policy.structural_interface_pairs).union(
        policy.forbidden_pairs
    ) == set(policy.all_pairs)
    assert policy.srdf_exemptions_applied is False
    assert policy.manual_pair_allowlist_used is False
    assert policy.online_truth_used is False
    with pytest.raises(FrozenInstanceError):
        policy.forbidden_pair_count = 120  # type: ignore[misc]
