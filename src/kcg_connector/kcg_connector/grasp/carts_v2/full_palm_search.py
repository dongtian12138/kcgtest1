"""Deterministic full-palm candidate cascade; geometry stays in callbacks."""

from __future__ import annotations

from itertools import product
import math
from typing import Callable, Mapping, Sequence

import numpy as np

from kcg_connector.grasp.carts_v2.models import CandidateSeed


_PALM_CONFIGURATION_COUNT = 91
_MAXIMUM_INPUT_PER_PALM = 64
_MAXIMUM_PRECISE_PER_PALM = 8
_MAXIMUM_GLOBAL_EVALUATIONS_PER_SEED = 300
_PREGRASP_PHASE_VALUES = (0.0, 0.1, 0.2)

PhaseTriple = tuple[float, float, float]
Evaluator = Callable[[CandidateSeed, PhaseTriple, int], Mapping[str, object]]


def fixed_pregrasp_phase_combinations() -> tuple[PhaseTriple, ...]:
    """Return the frozen 3^3 normalized per-finger preshape design."""

    return tuple(
        tuple(float(value) for value in values)  # type: ignore[misc]
        for values in product(_PREGRASP_PHASE_VALUES, repeat=3)
    )


def _validated_grid(palm_grid_rad: Sequence[float]) -> np.ndarray:
    grid = np.asarray(palm_grid_rad, dtype=np.float64)
    steps = np.diff(grid)
    if (
        grid.shape != (_PALM_CONFIGURATION_COUNT,)
        or not np.all(np.isfinite(grid))
        or np.any(steps <= 0.0)
        or not np.allclose(steps, steps[0], atol=1.0e-12, rtol=0.0)
    ):
        raise ValueError("palm grid must be 91 finite, increasing, uniform angles")
    return grid


def group_candidates_by_palm(
    candidates: Sequence[CandidateSeed], palm_grid_rad: Sequence[float]
) -> tuple[tuple[float, tuple[CandidateSeed, ...]], ...]:
    """Return all 91 canonical buckets and fail closed on ambiguous identity."""

    grid = _validated_grid(palm_grid_rad)
    buckets: list[list[CandidateSeed]] = [[] for _ in grid]
    identifiers: set[str] = set()
    for seed in candidates:
        if seed.candidate_id in identifiers:
            raise ValueError(f"duplicate candidate ID: {seed.candidate_id}")
        identifiers.add(seed.candidate_id)
        value = seed.palm_configuration_rad
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f"{seed.candidate_id} has no finite palm configuration")
        matches = np.flatnonzero(
            np.isclose(grid, float(value), atol=1.0e-12, rtol=0.0)
        )
        if len(matches) != 1:
            raise ValueError(f"{seed.candidate_id} is outside the registered palm grid")
        bucket = buckets[int(matches[0])]
        bucket.append(seed)
        if len(bucket) > _MAXIMUM_INPUT_PER_PALM:
            raise ValueError("a palm angle exceeds its fixed 64-candidate input budget")
    return tuple(
        (float(angle), tuple(sorted(bucket, key=lambda seed: seed.candidate_id)))
        for angle, bucket in zip(grid, buckets)
    )


def _selection_key(value: object) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)):
        raise ValueError("selection_key must be a numeric sequence")
    key = tuple(float(item) for item in value)
    if any(not math.isfinite(item) for item in key):
        raise ValueError("selection_key values must be finite")
    return key


def _evaluation_count(result: Mapping[str, object], remaining: int) -> int:
    value = result.get("evaluation_count")
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("evaluator must report a positive integer evaluation_count")
    if value > remaining:
        raise ValueError("Stage A+B evaluation budget exceeded for one seed")
    return value


def _validated_budget(value: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= _MAXIMUM_GLOBAL_EVALUATIONS_PER_SEED
    ):
        raise ValueError("global per-seed budget must be an integer in [1, 300]")
    return value


def select_pregrasp_combination(
    seed: CandidateSeed,
    cheap_evaluator: Evaluator,
    *,
    global_budget_per_seed: int = _MAXIMUM_GLOBAL_EVALUATIONS_PER_SEED,
) -> dict[str, object]:
    """Select one cheap-safe preshape; lower selection_key is better."""

    global_budget_per_seed = _validated_budget(global_budget_per_seed)
    used = 0
    best: tuple[tuple[float, ...], PhaseTriple, dict[str, object]] | None = None
    finger_chain_collision_count = 0
    for phases in fixed_pregrasp_phase_combinations():
        remaining = global_budget_per_seed - used
        if remaining <= 0:
            raise ValueError("Stage A exhausted the global per-seed budget")
        result = dict(cheap_evaluator(seed, phases, remaining))
        used += _evaluation_count(result, remaining)
        open_collision = result.get("open_palm_base_table_collision")
        finger_collision = result.get("pregrasp_finger_chain_table_collision")
        accepted = result.get("accepted")
        if not all(type(value) is bool for value in (open_collision, finger_collision, accepted)):
            raise ValueError("cheap evaluator collision and accepted fields must be bool")
        if open_collision:
            return {
                "status": "REJECT",
                "reason": "OPEN_PALM_BASE_TABLE_COLLISION",
                "stage_a_evaluation_count": used,
            }
        if finger_collision:
            finger_chain_collision_count += 1
            continue
        if not accepted:
            continue
        key = _selection_key(result.get("selection_key"))
        ranked = (key, phases, result)
        if best is None or ranked[:2] < best[:2]:
            best = ranked
    if best is None:
        reason = (
            "ALL_27_PREGRASP_FINGER_CHAINS_COLLIDE_TABLE"
            if finger_chain_collision_count == 27
            else "NO_CHEAP_PREGRASP_SURVIVES"
        )
        return {"status": "REJECT", "reason": reason, "stage_a_evaluation_count": used}
    return {
        "status": "SURVIVE",
        "pregrasp_closure_phases": best[1],
        "cheap_result": best[2],
        "cheap_selection_key": best[0],
        "stage_a_evaluation_count": used,
    }


def run_full_palm_cascade(
    candidates: Sequence[CandidateSeed],
    palm_grid_rad: Sequence[float],
    *,
    cheap_evaluator: Evaluator,
    precise_evaluator: Evaluator,
    global_budget_per_seed: int = _MAXIMUM_GLOBAL_EVALUATIONS_PER_SEED,
) -> dict[str, object]:
    """Run per-seed Stage A, per-angle top-8 Stage B, and budget accounting."""

    global_budget_per_seed = _validated_budget(global_budget_per_seed)
    groups = group_candidates_by_palm(candidates, palm_grid_rad)
    rejected: list[dict[str, object]] = []
    deferred: list[dict[str, object]] = []
    selected: list[dict[str, object]] = []
    counts: dict[str, dict[str, int]] = {}
    for palm_angle, seeds in groups:
        survivors = []
        for seed in seeds:
            stage_a = select_pregrasp_combination(
                seed, cheap_evaluator, global_budget_per_seed=global_budget_per_seed
            )
            count_a = int(stage_a["stage_a_evaluation_count"])
            counts[seed.candidate_id] = {"stage_a": count_a, "stage_b": 0, "total": count_a}
            if stage_a["status"] != "SURVIVE":
                rejected.append({"seed": seed, **stage_a})
                continue
            survivors.append((stage_a["cheap_selection_key"], seed.candidate_id, seed, stage_a))
        survivors.sort(key=lambda row: (row[0], row[1]))
        for _key, _identifier, seed, stage_a in survivors[_MAXIMUM_PRECISE_PER_PALM:]:
            deferred.append({"seed": seed, "reason": "PER_ANGLE_PRECISE_TOP8_LIMIT"})
        angle_selected = []
        for _key, _identifier, seed, stage_a in survivors[:_MAXIMUM_PRECISE_PER_PALM]:
            count_a = int(stage_a["stage_a_evaluation_count"])
            remaining = global_budget_per_seed - count_a
            if remaining <= 0:
                raise ValueError("no Stage B budget remains for a selected seed")
            phases = stage_a["pregrasp_closure_phases"]
            result = dict(precise_evaluator(seed, phases, remaining))
            count_b = _evaluation_count(result, remaining)
            counts[seed.candidate_id] = {
                "stage_a": count_a,
                "stage_b": count_b,
                "total": count_a + count_b,
            }
            if type(result.get("accepted")) is not bool:
                raise ValueError("precise evaluator accepted field must be bool")
            if not result["accepted"]:
                rejected.append({"seed": seed, "reason": "PRECISE_REJECT", "precise_result": result})
                continue
            precise_key = _selection_key(result.get("selection_key"))
            angle_selected.append((precise_key, seed.candidate_id, seed, stage_a, result))
        for precise_key, _identifier, seed, stage_a, result in sorted(
            angle_selected, key=lambda row: (row[0], row[1])
        ):
            selected.append({
                "seed": seed,
                "palm_configuration_rad": palm_angle,
                "pregrasp_closure_phases": stage_a["pregrasp_closure_phases"],
                "cheap_result": stage_a["cheap_result"],
                "precise_result": result,
                "precise_selection_key": precise_key,
            })
    return {
        "selected": tuple(selected),
        "rejected": tuple(rejected),
        "deferred": tuple(deferred),
        "evaluation_count_by_candidate": counts,
    }


__all__ = [
    "fixed_pregrasp_phase_combinations",
    "group_candidates_by_palm",
    "run_full_palm_cascade",
    "select_pregrasp_combination",
]
