"""Apply the one preregistered, unit-safe lexicographic V2 ordering."""

from __future__ import annotations

import math
from typing import Mapping

from kcg_connector.grasp.carts_v2.models import (
    ClosurePrediction,
    FastFilterResult,
    SelectedCandidate,
    TaskQualityResult,
)


def _minimum_metric(value: float | None) -> float:
    return math.inf if value is None else float(value)


def _maximum_metric(value: float | None) -> float:
    return math.inf if value is None else -float(value)


def select_top_candidates(
    predictions: tuple[ClosurePrediction, ...],
    filters: tuple[FastFilterResult, ...],
    qualities: tuple[TaskQualityResult, ...],
    *,
    top_k: int,
    path_clearance_by_id: Mapping[str, float | None] | None = None,
) -> tuple[SelectedCandidate, ...]:
    """Rank by seven explicit fields; no cross-unit weighted sum is formed."""

    if top_k < 1:
        raise ValueError("top_k must be positive")
    prediction_by_id = {row.seed.candidate_id: row for row in predictions}
    filter_by_id = {row.candidate_id: row for row in filters}
    quality_by_id = {row.candidate_id: row for row in qualities}
    if set(prediction_by_id) != set(filter_by_id):
        raise ValueError("prediction and fast-filter candidate sets differ")
    clearances = {} if path_clearance_by_id is None else dict(path_clearance_by_id)
    eligible = []
    for candidate_id, quality in quality_by_id.items():
        fast_filter = filter_by_id.get(candidate_id)
        prediction = prediction_by_id.get(candidate_id)
        if (
            fast_filter is None
            or prediction is None
            or fast_filter.status != "FAST_SURVIVE"
        ):
            continue
        clearance = clearances.get(candidate_id)
        key = (
            _maximum_metric(quality.worst_task_margin),
            _maximum_metric(quality.lower_tail_mean_margin),
            _minimum_metric(quality.required_peak_normal_force_n),
            _minimum_metric(quality.maximum_joint_load_utilization),
            _maximum_metric(clearance),
            _minimum_metric(quality.sensitivity),
            candidate_id,
        )
        eligible.append((key, prediction, fast_filter, quality, clearance))
    eligible.sort(key=lambda row: row[0])
    selected = []
    for rank, (_key, prediction, fast_filter, quality, clearance) in enumerate(
        eligible[:top_k], start=1
    ):
        selected.append(
            SelectedCandidate(
                rank=rank,
                prediction=prediction,
                fast_filter=fast_filter,
                task_quality=quality,
                path_minimum_clearance_m=clearance,
                offline_task_gate_passed=(
                    quality.status == "TASK_SURVIVE"
                    and prediction.status == "CLOSURE_SURVIVE"
                ),
            )
        )
    return tuple(selected)


__all__ = ["select_top_candidates"]
