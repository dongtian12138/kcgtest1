"""Exact collision-filter plan for the segmented D38999 proxy.

The public-dimension proxy represents coaxial circular shells with separate
boxes.  At the 0.30 mm nominal radial running clearance, PhysX convex contact
margins report two known classes of false contact.  This helper disables only
those explicit segment pairs; it does not disable the fixed rear-body stop,
the fixture, the table, the robot, or any other connector collision.

The module deliberately imports no Isaac/pxr package at module import time.
Callers inject the USD bindings so ordinary Python tests can verify the exact
500-pair plan without starting SimulationApp.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProxyCollisionFilterPlan:
    """Resolved prim paths for the narrow proxy-only collision exception."""

    body_mating_segments: tuple[str, ...]
    nut_segments: tuple[str, ...]
    fixed_entry_segments: tuple[str, ...]
    filtered_pairs: tuple[tuple[str, str], ...]


def build_proxy_collision_filter_plan(
    body_root: str,
    nut_root: str,
    fixed_receptacle_root: str,
    *,
    body_mating_segment_count: int = 20,
    nut_segment_count: int = 24,
    fixed_entry_segment_count: int = 20,
) -> ProxyCollisionFilterPlan:
    """Build the same exact filter already proven by the q7 twist probe."""

    counts = (
        body_mating_segment_count,
        nut_segment_count,
        fixed_entry_segment_count,
    )
    if any(type(value) is not int or value <= 0 for value in counts):
        raise ValueError(
            "proxy collision segment counts must be positive ints"
        )
    if body_mating_segment_count != fixed_entry_segment_count:
        raise ValueError(
            "body mating and fixed entry segment counts must match"
        )

    body_segments = tuple(
        f"{body_root}/MatingShell/Segment_{index:02d}"
        for index in range(body_mating_segment_count)
    )
    nut_segments = tuple(
        f"{nut_root}/Segment_{index:02d}"
        for index in range(nut_segment_count)
    )
    fixed_segments = tuple(
        f"{fixed_receptacle_root}/EntryShell/Segment_{index:02d}"
        for index in range(fixed_entry_segment_count)
    )
    # Every nut box is a thread-geometry placeholder and therefore excluded
    # from every fixed entry box.  The mating shells retain all collisions
    # except the 20 same-angle box pairs that falsely close the 0.30 mm bore.
    pairs = tuple(
        (nut_path, fixed_path)
        for nut_path in nut_segments
        for fixed_path in fixed_segments
    ) + tuple(zip(body_segments, fixed_segments))
    return ProxyCollisionFilterPlan(
        body_mating_segments=body_segments,
        nut_segments=nut_segments,
        fixed_entry_segments=fixed_segments,
        filtered_pairs=pairs,
    )


def apply_proxy_collision_filter(
    stage,
    UsdPhysics,
    Sdf,
    plan: ProxyCollisionFilterPlan,
) -> dict[str, object]:
    """Validate and author the exact filter plan before physics starts."""

    required_paths = set(plan.body_mating_segments)
    required_paths.update(plan.nut_segments)
    required_paths.update(plan.fixed_entry_segments)
    prims = {}
    for path in sorted(required_paths):
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"proxy collision prim is missing: {path}")
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            raise RuntimeError(f"proxy prim is not a collider: {path}")
        prims[path] = prim

    relationships = {}
    for source_path, target_path in plan.filtered_pairs:
        relation = relationships.get(source_path)
        if relation is None:
            filtered_api = UsdPhysics.FilteredPairsAPI.Apply(
                prims[source_path]
            )
            relation = filtered_api.CreateFilteredPairsRel()
            relationships[source_path] = relation
        relation.AddTarget(Sdf.Path(target_path))

    return {
        "body_mating_segment_count": len(plan.body_mating_segments),
        "enabled": True,
        "fixed_entry_segment_count": len(plan.fixed_entry_segments),
        "mode": "proxy_false_contacts_only",
        "nut_segment_count": len(plan.nut_segments),
        "pair_count": len(plan.filtered_pairs),
    }


__all__ = [
    "ProxyCollisionFilterPlan",
    "apply_proxy_collision_filter",
    "build_proxy_collision_filter_plan",
]
