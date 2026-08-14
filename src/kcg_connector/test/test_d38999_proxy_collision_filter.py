"""Pure tests for the narrow D38999 proxy collision exception."""

import sys

import pytest

from kcg_connector.d38999_proxy_collision_filter import (
    build_proxy_collision_filter_plan,
)


def test_plan_is_exactly_the_previously_proven_500_pairs():
    plan = build_proxy_collision_filter_plan(
        "/World/Loose/BodyAssembly",
        "/World/Loose/CouplingNut",
        "/World/Fixed",
    )
    assert len(plan.body_mating_segments) == 20
    assert len(plan.nut_segments) == 24
    assert len(plan.fixed_entry_segments) == 20
    assert len(plan.filtered_pairs) == 500
    assert len(set(plan.filtered_pairs)) == 500

    # All 24x20 nut/entry placeholder pairs are filtered.
    assert (
        "/World/Loose/CouplingNut/Segment_23",
        "/World/Fixed/EntryShell/Segment_19",
    ) in plan.filtered_pairs
    # Mating/entry filtering is deliberately same-index only; real remaining
    # connector, fixture, table and robot collisions are outside the plan.
    assert (
        "/World/Loose/BodyAssembly/MatingShell/Segment_08",
        "/World/Fixed/EntryShell/Segment_08",
    ) in plan.filtered_pairs
    assert (
        "/World/Loose/BodyAssembly/MatingShell/Segment_08",
        "/World/Fixed/EntryShell/Segment_09",
    ) not in plan.filtered_pairs
    assert all("Table" not in pair for pair in plan.filtered_pairs)
    assert all("Fixture" not in pair for pair in plan.filtered_pairs)


def test_plan_rejects_invalid_or_mismatched_topology():
    with pytest.raises(ValueError):
        build_proxy_collision_filter_plan(
            "/B", "/N", "/F", nut_segment_count=0
        )
    with pytest.raises(ValueError):
        build_proxy_collision_filter_plan(
            "/B", "/N", "/F", body_mating_segment_count=19
        )


def test_pure_import_does_not_load_isaac_or_usd():
    for name in ("isaacsim", "omni", "pxr"):
        assert name not in sys.modules
