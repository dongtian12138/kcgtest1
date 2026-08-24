"""Straight-line orchestration for the offline CARTS-Grasp V2 stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np

from kcg_connector.grasp.carts_v2.candidate_generator import (
    generate_raw_candidates,
    select_diverse_predictions,
)
from kcg_connector.grasp.carts_v2.closure_predictor import SequentialClosurePredictor
from kcg_connector.grasp.carts_v2.fast_filter import fast_filter_predictions
from kcg_connector.grasp.carts_v2.legacy_exact_validator import (
    validate_top_candidates,
)
from kcg_connector.grasp.carts_v2.models import (
    CandidateSeed,
    ClosurePrediction,
    FastFilterResult,
    ExactValidationResult,
    SelectedCandidate,
    TaskQualityResult,
    V2Inputs,
    load_v2_inputs,
)
from kcg_connector.grasp.carts_v2.selector import select_candidate_rankings
from kcg_connector.grasp.carts_v2.task_quality import (
    common_uncertainty_design,
    evaluate_task_quality,
)


@dataclass(frozen=True)
class OfflinePipelineResult:
    inputs: V2Inputs
    raw_candidates: tuple[CandidateSeed, ...]
    raw_closure_predictions: tuple[ClosurePrediction, ...]
    raw_fast_filter_results: tuple[FastFilterResult, ...]
    diversity_rejection_reasons: dict[str, str]
    candidates: tuple[CandidateSeed, ...]
    closure_predictions: tuple[ClosurePrediction, ...]
    fast_filter_results: tuple[FastFilterResult, ...]
    task_quality_results: tuple[TaskQualityResult, ...]
    executable_candidates: tuple[SelectedCandidate, ...]
    diagnostic_candidates: tuple[SelectedCandidate, ...]
    exact_validation_results: tuple[ExactValidationResult, ...]
    scenario_design: np.ndarray
    timings_s: dict[str, float]


def run_offline_pipeline(
    repository_root: Path | str,
    *,
    config_path: Path | str,
    object_id: str,
) -> OfflinePipelineResult:
    timings: dict[str, float] = {}
    started = time.perf_counter()
    inputs = load_v2_inputs(
        repository_root, config_path=config_path, object_id=object_id
    )
    timings["load_inputs"] = time.perf_counter() - started

    started = time.perf_counter()
    raw_candidates = generate_raw_candidates(inputs)
    timings["candidate_generation"] = time.perf_counter() - started

    started = time.perf_counter()
    predictor = SequentialClosurePredictor(inputs)
    raw_predictions = tuple(predictor.predict(seed) for seed in raw_candidates)
    timings["closure_prediction"] = time.perf_counter() - started

    started = time.perf_counter()
    raw_filters = fast_filter_predictions(inputs, raw_predictions)
    timings["fast_filter"] = time.perf_counter() - started

    started = time.perf_counter()
    predictions, diversity_rejections = select_diverse_predictions(
        inputs, raw_predictions, raw_filters
    )
    filter_by_id = {row.candidate_id: row for row in raw_filters}
    filters = tuple(filter_by_id[row.seed.candidate_id] for row in predictions)
    candidates = tuple(row.seed for row in predictions)
    timings["diversity_selection"] = time.perf_counter() - started

    started = time.perf_counter()
    design = common_uncertainty_design(inputs)
    qualities = tuple(
        evaluate_task_quality(inputs, prediction, design)
        for prediction, fast_filter in zip(predictions, filters)
        if fast_filter.status == "FAST_SURVIVE"
    )
    timings["task_quality"] = time.perf_counter() - started

    started = time.perf_counter()
    top_k = int(inputs.config.section("exact_validation")["top_k"])
    executable, diagnostic = select_candidate_rankings(
        predictions,
        filters,
        qualities,
        top_k=top_k,
        path_clearance_by_id={
            row.candidate_id: row.minimum_table_clearance_m for row in filters
        },
    )
    timings["selection"] = time.perf_counter() - started

    started = time.perf_counter()
    exact_results = validate_top_candidates(inputs, executable)
    timings["exact_validation"] = time.perf_counter() - started
    timings["total"] = sum(
        value for key, value in timings.items() if key != "total"
    )
    return OfflinePipelineResult(
        inputs=inputs,
        raw_candidates=raw_candidates,
        raw_closure_predictions=raw_predictions,
        raw_fast_filter_results=raw_filters,
        diversity_rejection_reasons=diversity_rejections,
        candidates=candidates,
        closure_predictions=predictions,
        fast_filter_results=filters,
        task_quality_results=qualities,
        executable_candidates=executable,
        diagnostic_candidates=diagnostic,
        exact_validation_results=exact_results,
        scenario_design=design,
        timings_s=timings,
    )


__all__ = ["OfflinePipelineResult", "run_offline_pipeline"]
