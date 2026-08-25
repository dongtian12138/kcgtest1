"""Deterministic height-before-Top-8 full-palm cascade."""

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
_PREGRASP_PHASE_VALUES = (0.0, 0.1, 0.2)

PhaseTriple = tuple[float, float, float]
HeightEvaluator = Callable[
    [CandidateSeed], tuple[Sequence[CandidateSeed], Mapping[str, object]]
]
ProgressCallback = Callable[[Mapping[str, object]], None]


def fixed_pregrasp_phase_combinations() -> tuple[PhaseTriple, ...]:
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
        matches = np.flatnonzero(np.isclose(grid, float(value), atol=1.0e-12, rtol=0.0))
        if len(matches) != 1:
            raise ValueError(f"{seed.candidate_id} is outside the registered palm grid")
        buckets[int(matches[0])].append(seed)
        if len(buckets[int(matches[0])]) > _MAXIMUM_INPUT_PER_PALM:
            raise ValueError("a palm angle exceeds its fixed 64-candidate input budget")
    return tuple((float(angle), tuple(sorted(rows, key=lambda row: row.candidate_id)))
                 for angle, rows in zip(grid, buckets))


def _selection_key(value: object, label: str) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)):
        raise ValueError(f"{label} must be a numeric sequence")
    key = tuple(float(item) for item in value)
    if not key or any(not math.isfinite(item) for item in key):
        raise ValueError(f"{label} values must be finite")
    return key


def bind_pregrasp(inputs: V2Inputs, seed: CandidateSeed, phases: PhaseTriple):
    joints = joint_positions_for_phases(
        inputs, phases, reference_joint_positions_rad=seed.pregrasp_joint_positions_rad)
    bound = replace(seed, pregrasp_closure_phases=phases,
                    pregrasp_joint_positions_rad=tuple(float(value) for value in joints))
    palm_index = inputs.hand_model.independent_joint_names.index("f1j1")
    if (bound.palm_configuration_rad is None or not np.isclose(
            joints[palm_index], bound.palm_configuration_rad, atol=1.0e-12, rtol=0.0)):
        raise ValueError("PALM_CONFIGURATION_LOST_IN_PIPELINE")
    return bound


def _compact_height_audit(value: Mapping[str, object]) -> dict[str, object]:
    audit = dict(value)
    deferred = tuple(audit.pop("deferred", ()))
    audit["pregrasp_budget_not_evaluated_count"] = len(deferred)
    return audit


def _evaluate_seed(seed: CandidateSeed, evaluator: HeightEvaluator):
    raw_survivors, raw_audit = evaluator(seed)
    audit = _compact_height_audit(raw_audit)
    if str(audit.get("candidate_id")) != seed.candidate_id:
        raise ValueError("height evaluator candidate identity changed")
    survivors = tuple(raw_survivors)
    exact_count = int(audit.get("exact_variant_evaluated_count", -1))
    if not 0 <= len(survivors) <= exact_count <= 2:
        raise ValueError("height evaluator exceeded the two-variant budget")
    evaluated = tuple(audit.get("evaluated", ()))
    ranked = []
    for survivor in survivors:
        if (survivor.candidate_id != seed.candidate_id
                or survivor.object_id != seed.object_id
                or survivor.palm_configuration_rad != seed.palm_configuration_rad):
            raise ValueError("projected candidate identity changed")
        phases = tuple(float(value) for value in survivor.pregrasp_closure_phases)
        matches = [dict(row) for row in evaluated
                   if tuple(row.get("pregrasp_closure_phases", ())) == phases
                   and row.get("status") ==
                   "OFFLINE_SAMPLED_HAND_HEIGHT_FEASIBLE_AT_PROJECTED_Z"]
        if len(matches) != 1:
            raise ValueError("projected survivor has no unique accepted height row")
        key = _selection_key(matches[0].get("selection_key"), "height selection_key")
        ranked.append((key, phases, survivor, matches[0]))
    if not ranked:
        reasons = [str(row.get("reason", "HEIGHT_REJECT")) for row in evaluated]
        return None, {"candidate_id": seed.candidate_id,
                      "reason": reasons[0] if reasons else "NO_HEIGHT_VARIANT_SURVIVED",
                      "height_audit": audit}, exact_count
    ranked.sort(key=lambda row: (row[0], row[1]))
    selected = ranked[0]
    return selected[2], {"candidate_id": seed.candidate_id,
                         "physical_selection_key": list(selected[0]),
                         "selected_height_result": selected[3],
                         "height_audit": audit}, exact_count


def _budget_candidates(rows):
    ranked = []
    for seed, evidence in rows:
        score = float(seed.generator_score if seed.generator_score is not None else math.nan)
        if not math.isfinite(score):
            raise ValueError("projected candidate generator score is invalid")
        key = _selection_key(evidence.get("physical_selection_key"),
                             "physical_selection_key")
        ranked.append((key, -score, seed.candidate_id, seed, evidence))
    ranked.sort(key=lambda row: row[:3])
    retained = ranked[:_MAXIMUM_PRECISE_PER_PALM]
    deferred = [{"candidate_id": row[2],
                 "reason": "DEFERRED_BY_POST_HEIGHT_PER_ANGLE_TOP8",
                 "physical_selection_key": list(row[0])}
                for row in ranked[_MAXIMUM_PRECISE_PER_PALM:]]
    return (tuple(row[3] for row in retained),
            [row[4] for row in retained], deferred)


def _candidate_record(seed: CandidateSeed) -> dict[str, object]:
    return {"candidate_id": seed.candidate_id,
            "palm_configuration_rad": seed.palm_configuration_rad,
            "object_from_hand": list(seed.object_from_hand),
            "pregrasp_closure_phases": list(seed.pregrasp_closure_phases),
            "pregrasp_joint_positions_rad": list(seed.pregrasp_joint_positions_rad)}


def _cascade_audit(candidates, selected, rejected, deferred, per_angle,
                   callback_counts, completed):
    return {
        "schema_version": "carts_height_before_top8_cascade_v2",
        "claim_scope": "OFFLINE_HEIGHT_PROJECTED_CASCADE_NOT_DYNAMIC_SUCCESS",
        "ordering": "ALL_SEEDS_HEIGHT_FEASIBILITY_THEN_PER_ANGLE_TOP8",
        "input_candidate_count": len(candidates),
        "input_candidate_ids": [seed.candidate_id for seed in candidates],
        "completed_palm_bucket_count": completed,
        "task_evaluation_candidate_ids": [seed.candidate_id for seed in selected],
        "task_evaluation_candidates": [_candidate_record(seed) for seed in selected],
        "per_angle": per_angle, "deferred": deferred, "rejected": rejected,
        "exact_height_variant_count_by_candidate": callback_counts,
    }


def _resume_cascade(inputs, candidates, groups, resume):
    if resume is None:
        return 0, [], [], [], [], {}
    completed = int(resume.get("completed_palm_bucket_count", -1))
    per_angle = list(resume.get("per_angle", ()))
    expected_ids = [seed.candidate_id for seed in candidates]
    if (resume.get("schema_version") != "carts_height_before_top8_cascade_v2"
            or resume.get("input_candidate_ids") != expected_ids
            or not 0 <= completed <= _PALM_CONFIGURATION_COUNT
            or len(per_angle) != completed):
        raise ValueError("height-before-Top8 checkpoint identity changed")
    expected_angles = [row[0] for row in groups[:completed]]
    if not np.allclose([row.get("palm_configuration_rad") for row in per_angle],
                       expected_angles, atol=1.0e-12, rtol=0.0):
        raise ValueError("height checkpoint buckets are not the completed prefix")
    by_id = {seed.candidate_id: seed for seed in candidates}
    selected = []
    for row in resume.get("task_evaluation_candidates", ()):
        seed = by_id.get(str(row.get("candidate_id")))
        phases = tuple(float(value) for value in row.get("pregrasp_closure_phases", ()))
        pose = tuple(float(value) for value in row.get("object_from_hand", ()))
        if seed is None or len(phases) != 3 or len(pose) != 16:
            raise ValueError("checkpoint projected candidate identity changed")
        bound = replace(bind_pregrasp(inputs, seed, phases), object_from_hand=pose)
        if (bound.palm_configuration_rad not in expected_angles
                or not np.allclose(bound.pregrasp_joint_positions_rad,
                                   row.get("pregrasp_joint_positions_rad", ()),
                                   atol=1.0e-12, rtol=0.0)):
            raise ValueError("checkpoint projected candidate state changed")
        selected.append(bound)
    return (completed, selected, list(resume.get("rejected", ())), per_angle,
            list(resume.get("deferred", ())),
            dict(resume.get("exact_height_variant_count_by_candidate", {})))


def run_full_palm_cascade(
    inputs: V2Inputs,
    candidates: Sequence[CandidateSeed],
    palm_grid_rad: Sequence[float],
    *,
    height_evaluator: HeightEvaluator,
    resume_audit: Mapping[str, object] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> tuple[tuple[CandidateSeed, ...], dict[str, object]]:
    """Evaluate height for every seed before assigning each angle's Top-8."""
    groups = group_candidates_by_palm(candidates, palm_grid_rad)
    completed, selected, rejected, per_angle, deferred, callback_counts = (
        _resume_cascade(inputs, candidates, groups, resume_audit))
    for palm_angle, seeds in groups[completed:]:
        height_rows, angle_rejected = [], []
        for seed in seeds:
            survivor, evidence, exact_count = _evaluate_seed(seed, height_evaluator)
            callback_counts[seed.candidate_id] = exact_count
            (angle_rejected if survivor is None else height_rows).append(
                evidence if survivor is None else (survivor, evidence))
        retained, retained_evidence, angle_deferred = _budget_candidates(height_rows)
        selected.extend(retained)
        rejected.extend(angle_rejected)
        deferred.extend(angle_deferred)
        reason_counts: dict[str, int] = {}
        for row in angle_rejected:
            reason = str(row["reason"])
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        per_angle.append({
            "palm_configuration_rad": palm_angle,
            "input_count": len(seeds),
            "height_evaluated_seed_count": len(seeds),
            "height_projected_seed_survive_count": len(height_rows),
            "budget_retained_count": len(retained),
            "budget_deferred_count": len(angle_deferred),
            "rejection_reason_counts": reason_counts,
            "selected_candidate_ids": [seed.candidate_id for seed in retained],
            "selected_height_evidence": retained_evidence,
        })
        completed += 1
        audit = _cascade_audit(candidates, selected, rejected, deferred, per_angle,
                               callback_counts, completed)
        if progress_callback is not None:
            progress_callback(audit)
    audit = _cascade_audit(candidates, selected, rejected, deferred, per_angle,
                           callback_counts, completed)
    return tuple(selected), audit


__all__ = ["bind_pregrasp", "fixed_pregrasp_phase_combinations",
           "group_candidates_by_palm", "run_full_palm_cascade"]
