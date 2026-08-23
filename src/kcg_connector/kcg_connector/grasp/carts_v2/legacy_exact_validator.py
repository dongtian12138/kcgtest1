"""The only V2 adapter allowed to import the frozen strict collision backend."""

from __future__ import annotations

import math
import time

from kcg_connector.grasp.carts_v2.models import (
    ExactValidationResult,
    SelectedCandidate,
    V2Inputs,
)
from kcg_connector.grasp.robust.candidate_route_collision import (
    BACKEND_LIFT_DISTANCE_M,
    build_candidate_route_collision_certificate,
)


_LEGACY_PUBLIC_BACKEND = build_candidate_route_collision_certificate


def validate_top_candidates(
    inputs: V2Inputs,
    selected: tuple[SelectedCandidate, ...],
) -> tuple[ExactValidationResult, ...]:
    """Fail closed before invoking a backend with incompatible motion scope."""

    requested = float(inputs.config.section("dynamic")["lift_distance_m"])
    mismatch = not math.isclose(
        requested, BACKEND_LIFT_DISTANCE_M, rel_tol=0.0, abs_tol=0.0
    )
    results = []
    for candidate in selected:
        started = time.perf_counter()
        status = (
            "UNRESOLVED_INTERFACE_MISMATCH"
            if mismatch
            else "UNRESOLVED_LEGACY_CONTRACT_ADAPTATION"
        )
        reason = (
            f"V2 requests {requested:.3f} m lift but frozen backend "
            f"{_LEGACY_PUBLIC_BACKEND.__name__} covers "
            f"{BACKEND_LIFT_DISTANCE_M:.3f} m"
            if mismatch
            else "V2 candidate is not bound to the frozen V9 contract graph"
        )
        results.append(
            ExactValidationResult(
                candidate_id=candidate.prediction.seed.candidate_id,
                status=status,
                reason=reason,
                requested_lift_distance_m=requested,
                backend_lift_distance_m=BACKEND_LIFT_DISTANCE_M,
                backend_invoked=False,
                path_minimum_clearance_m=None,
                elapsed_s=time.perf_counter() - started,
            )
        )
    return tuple(results)


__all__ = ["validate_top_candidates"]
