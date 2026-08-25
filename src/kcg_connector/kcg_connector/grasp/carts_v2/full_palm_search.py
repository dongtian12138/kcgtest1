"""Deterministic full-palm budget allocation and callback cascade."""

from __future__ import annotations

from dataclasses import replace
from itertools import product
import math
from typing import Callable, Mapping, Sequence

import numpy as np

from kcg_connector.grasp.carts_v2.models import (
    CandidateSeed, V2Inputs, joint_positions_for_phases,
)


_PALM_CONFIGURATION_COUNT = 91
_MAXIMUM_INPUT_PER_PALM = 64
_MAXIMUM_PRECISE_PER_PALM = 8
_MAXIMUM_CALLBACK_EVALUATIONS_PER_SEED = 28
_PREGRASP_PHASE_VALUES = (0.0, 0.1, 0.2)

PhaseTriple = tuple[float, float, float]
Variant = tuple[CandidateSeed, PhaseTriple]
PregraspBatchEvaluator = Callable[[tuple[Variant, ...]], Sequence[Mapping[str, object]]]
PreciseEvaluator = Callable[[CandidateSeed], Mapping[str, object]]


def fixed_pregrasp_phase_combinations() -> tuple[PhaseTriple, ...]:
    """Return the frozen 3^3 normalized per-finger preshape design."""

    return tuple(tuple(float(value) for value in values)  # type: ignore[misc]
                 for values in product(_PREGRASP_PHASE_VALUES, repeat=3))


def _validated_grid(palm_grid_rad: Sequence[float]) -> np.ndarray:
    grid = np.asarray(palm_grid_rad, dtype=np.float64)
    steps = np.diff(grid)
    if (grid.shape != (_PALM_CONFIGURATION_COUNT,)
            or not np.all(np.isfinite(grid)) or np.any(steps <= 0.0)
            or not np.allclose(steps, steps[0], atol=1.0e-12, rtol=0.0)):
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
        matches = np.flatnonzero(np.isclose(
            grid, float(value), atol=1.0e-12, rtol=0.0))
        if len(matches) != 1:
            raise ValueError(f"{seed.candidate_id} is outside the registered palm grid")
        bucket = buckets[int(matches[0])]
        bucket.append(seed)
        if len(bucket) > _MAXIMUM_INPUT_PER_PALM:
            raise ValueError("a palm angle exceeds its fixed 64-candidate input budget")
    return tuple((float(angle), tuple(sorted(
        bucket, key=lambda seed: seed.candidate_id)))
        for angle, bucket in zip(grid, buckets))


def _selection_key(value: object, label: str) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)):
        raise ValueError(f"{label} must be a numeric sequence")
    key = tuple(float(item) for item in value)
    if not key or any(not math.isfinite(item) for item in key):
        raise ValueError(f"{label} values must be finite")
    return key


def _budget_candidates(seeds, diagnostics):
    """Reserve each present generator branch once, then physical-key fill."""

    branches: dict[str, list[tuple[object, ...]]] = {}
    for seed in seeds:
        diagnostic = diagnostics.get(seed.candidate_id)
        if not isinstance(diagnostic, Mapping):
            raise ValueError(f"missing budget diagnostic for {seed.candidate_id}")
        branch = str(diagnostic.get("branch", ""))
        score = float(diagnostic.get("generator_score", math.nan))
        if not branch or not math.isfinite(score):
            raise ValueError("budget branch/score identity is invalid")
        physical = _selection_key(diagnostic.get("physical_selection_key"),
                                  "physical_selection_key")
        branches.setdefault(branch, []).append((
            physical, -score, seed.candidate_id, seed, dict(diagnostic)))
    if len(branches) > _MAXIMUM_PRECISE_PER_PALM:
        raise ValueError("generator branch count exceeds per-angle budget")
    ranked = {}
    for branch, rows in branches.items():
        rows.sort(key=lambda row: row[:3])
        ranked[branch] = [(*row, rank, branch) for rank, row in enumerate(rows)]
    retained = [ranked[branch][0] for branch in sorted(ranked)]
    remainder = [row for branch in sorted(ranked) for row in ranked[branch][1:]]
    remainder.sort(key=lambda row: (row[0], row[1], row[6], row[2]))
    retained.extend(remainder[:max(0, _MAXIMUM_PRECISE_PER_PALM - len(retained))])
    retained_ids = {row[2] for row in retained}
    deferred = [{"candidate_id": row[2],
                 "reason": "DEFERRED_BY_PER_ANGLE_PRECISE_BUDGET",
                 "budget_diagnostic": row[4]}
                for branch in sorted(ranked) for row in ranked[branch]
                if row[2] not in retained_ids]
    retained.sort(key=lambda row: row[2])
    return tuple(row[3] for row in retained), deferred


def select_pregrasp_combination(
    seed: CandidateSeed,
    evaluations: Sequence[tuple[PhaseTriple, Mapping[str, object]]],
) -> dict[str, object]:
    """Choose path-clear, contact-near preshape from exactly 27 public results."""

    if (len(evaluations) != 27 or {row[0] for row in evaluations}
            != set(fixed_pregrasp_phase_combinations())):
        raise ValueError("pregrasp evaluator must return all 27 registered phases")
    best = None
    pass_count = 0
    reasons: dict[str, int] = {}
    for phases, raw in evaluations:
        result = dict(raw)
        if type(result.get("accepted")) is not bool:
            raise ValueError("pregrasp accepted field must be bool")
        reported = tuple(float(value) for value in
                         result.get("pregrasp_closure_phases", ()))
        if reported != phases:
            raise ValueError("pregrasp result phase identity changed")
        if not result["accepted"]:
            for reason in result.get("reasons", ("PREGRASP_REJECT",)):
                reasons[str(reason)] = reasons.get(str(reason), 0) + 1
            continue
        raw_table = result.get("minimum_table_clearance_m")
        table = None if raw_table is None else float(raw_table)
        pad_map = result.get("pregrasp_pad_clearance_by_name_m")
        if not isinstance(pad_map, Mapping) or len(pad_map) != 3:
            raise ValueError("accepted pregrasp must report all three PAD distances")
        pad_distances = tuple(float(pad_map[name]) for name in sorted(pad_map))
        if ((table is not None and not math.isfinite(table))
                or any(not math.isfinite(value) for value in pad_distances)):
            raise ValueError("accepted pregrasp metrics must be finite")
        pass_count += 1
        table_key = (0.0, 0.0) if table is None else (1.0, -table)
        ranked = ((max(pad_distances), *table_key, *phases), phases, table,
                  dict(pad_map))
        if best is None or ranked[0] < best[0]:
            best = ranked
    if best is None:
        return {"status": "REJECT", "candidate_id": seed.candidate_id,
                "pregrasp_pass_count": 0, "rejection_reason_counts": reasons}
    return {
        "status": "SURVIVE", "candidate_id": seed.candidate_id,
        "pregrasp_pass_count": pass_count,
        "pregrasp_closure_phases": best[1],
        "minimum_table_clearance_m": best[2],
        "pregrasp_pad_clearance_by_name_m": best[3],
        "worst_pregrasp_pad_clearance_m": max(best[3].values()),
        "rejection_reason_counts": reasons,
    }


def _bound_pregrasp(inputs: V2Inputs, seed: CandidateSeed, phases: PhaseTriple):
    joints = joint_positions_for_phases(
        inputs, phases, reference_joint_positions_rad=seed.pregrasp_joint_positions_rad)
    bound = replace(seed, pregrasp_closure_phases=phases,
                    pregrasp_joint_positions_rad=tuple(float(value) for value in joints))
    palm_index = inputs.hand_model.independent_joint_names.index("f1j1")
    if (bound.palm_configuration_rad is None or not np.isclose(
            joints[palm_index], bound.palm_configuration_rad,
            atol=1.0e-12, rtol=0.0)):
        raise ValueError("PALM_CONFIGURATION_LOST_IN_PIPELINE")
    return bound


def _allocate_groups(groups, diagnostics, combinations):
    plans, variants, deferred = [], [], []
    for palm_angle, seeds in groups:
        retained, angle_deferred = _budget_candidates(seeds, diagnostics)
        start = len(variants)
        variants.extend((seed, phases) for seed in retained for phases in combinations)
        plans.append((palm_angle, seeds, retained, start))
        deferred.extend(angle_deferred)
    return plans, tuple(variants), deferred


def _evaluate_angle(inputs, plan, reports, combinations, precise_evaluator):
    palm_angle, seeds, retained, start = plan
    rejected, precise_rows, callback_counts = [], [], {}
    variant_passes = candidate_passes = closure_passes = fast_passes = 0
    for index, seed in enumerate(retained):
        offset = start + index * len(combinations)
        paired = tuple(zip(combinations, reports[offset:offset + len(combinations)]))
        if any(str(row.get("candidate_id")) != seed.candidate_id for _, row in paired):
            raise ValueError("pregrasp batch result order/identity changed")
        pregrasp = select_pregrasp_combination(seed, paired)
        variant_passes += int(pregrasp["pregrasp_pass_count"])
        callback_counts[seed.candidate_id] = 27
        if pregrasp["status"] != "SURVIVE":
            rejected.append(pregrasp)
            continue
        candidate_passes += 1
        bound = _bound_pregrasp(inputs, seed, pregrasp["pregrasp_closure_phases"])
        precise = dict(precise_evaluator(bound))
        callback_counts[seed.candidate_id] = 28
        for field in ("accepted", "closure_pass", "fast_filter_pass"):
            if type(precise.get(field)) is not bool:
                raise ValueError(f"precise evaluator {field} field must be bool")
        closure_passes += int(precise["closure_pass"])
        fast_passes += int(precise["fast_filter_pass"])
        if not precise["accepted"]:
            rejected.append({
                "candidate_id": seed.candidate_id, "precise_result": precise,
                "reason": "SELECTED_PREGRASP_PRECISE_REJECT:" + str(
                    precise.get("reason", "PRECISE_REJECT")),
                "alternative_path_safe_pregrasp_count": max(
                    0, int(pregrasp["pregrasp_pass_count"]) - 1),
                "alternative_pregrasp_status":
                    "BUDGET_DEFERRED_NOT_PHYSICAL_FAILURE",
            })
            continue
        key = _selection_key(precise.get("selection_key"), "selection_key")
        precise_rows.append((key, seed.candidate_id, bound, pregrasp, precise))
    precise_rows.sort(key=lambda row: (row[0], row[1]))
    reason_counts: dict[str, int] = {}
    for row in rejected:
        nested = row.get("rejection_reason_counts", {})
        for reason, count in nested.items():
            key = str(reason)
            reason_counts[key] = reason_counts.get(key, 0) + int(count)
        if "reason" in row:
            reason = str(row["reason"])
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    audit = {
        "palm_configuration_rad": palm_angle, "input_count": len(seeds),
        "budget_retained_count": len(retained),
        "pregrasp_variant_checked_count": len(retained) * len(combinations),
        "pregrasp_variant_pass_count": variant_passes,
        "pregrasp_candidate_survive_count": candidate_passes,
        "precise_callback_count": sum(value == 28 for value in callback_counts.values()),
        "exact_closure_pass_count": closure_passes,
        "precise_fast_filter_pass_count": fast_passes,
        "rejection_reason_counts": reason_counts,
        "precise_survivor_candidate_ids": [row[1] for row in precise_rows],
    }
    return precise_rows, rejected, callback_counts, audit


def run_full_palm_cascade(
    inputs: V2Inputs,
    candidates: Sequence[CandidateSeed],
    palm_grid_rad: Sequence[float],
    *,
    budget_diagnostics: Mapping[str, Mapping[str, object]],
    pregrasp_evaluator: PregraspBatchEvaluator,
    precise_evaluator: PreciseEvaluator,
) -> tuple[tuple[CandidateSeed, ...], dict[str, object]]:
    """Allocate 64-to-8 per angle and return every exact survivor."""

    combinations = fixed_pregrasp_phase_combinations()
    plans, variants, deferred = _allocate_groups(
        group_candidates_by_palm(candidates, palm_grid_rad),
        budget_diagnostics, combinations)
    reports = tuple(pregrasp_evaluator(variants)) if variants else ()
    if len(reports) != len(variants):
        raise ValueError("pregrasp batch result count differs from requested variants")
    selected, rejected, per_angle = [], [], []
    callback_counts: dict[str, int] = {}
    for plan in plans:
        precise_rows, angle_rejected, angle_counts, angle_audit = _evaluate_angle(
            inputs, plan, reports, combinations, precise_evaluator
        )
        rejected.extend(angle_rejected)
        callback_counts.update(angle_counts)
        selected.extend(row[2] for row in precise_rows)
        per_angle.append(angle_audit)
    audit = {
        "schema_version": "carts_full_palm_cascade_audit_v1",
        "claim_scope": "OFFLINE_CALLBACK_CASCADE_NOT_GRASP_OR_DYNAMIC_SUCCESS",
        "input_candidate_count": len(candidates),
        "task_evaluation_candidate_ids": [seed.candidate_id for seed in selected],
        "task_evaluation_candidates": [{"candidate_id": seed.candidate_id,
            "palm_configuration_rad": seed.palm_configuration_rad,
            "pregrasp_closure_phases": seed.pregrasp_closure_phases,
            "pregrasp_joint_positions_rad": seed.pregrasp_joint_positions_rad}
            for seed in selected],
        "candidate_level_callback_limit_per_seed": _MAXIMUM_CALLBACK_EVALUATIONS_PER_SEED,
        "internal_control_states_count_as_candidate_calls": False,
        "pregrasp_logical_state_count": sum(
            int(row.get("checked_state_count", 0)) for row in reports),
        "pregrasp_physical_query_state_count": sum(
            int(row.get("physical_query_state_count", 0)) for row in reports),
        "pregrasp_reused_identical_state_count": sum(
            int(row.get("reused_identical_state_count", 0)) for row in reports),
        "per_angle": per_angle,
        "deferred": deferred,
        "rejected": rejected,
        "callback_evaluation_count_by_candidate": callback_counts,
    }
    return tuple(selected), audit


__all__ = [
    "fixed_pregrasp_phase_combinations",
    "group_candidates_by_palm",
    "run_full_palm_cascade",
    "select_pregrasp_combination",
]
