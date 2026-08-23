"""Straight-line orchestration for the offline CARTS-Grasp V2 stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np

from kcg_connector.grasp.carts_v2.candidate_generator import generate_candidates
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
from kcg_connector.grasp.carts_v2.selector import select_top_candidates
from kcg_connector.grasp.carts_v2.task_quality import (
    common_uncertainty_design,
    evaluate_task_quality,
)


@dataclass(frozen=True)
class OfflinePipelineResult:
    inputs: V2Inputs
    candidates: tuple[CandidateSeed, ...]
    closure_predictions: tuple[ClosurePrediction, ...]
    fast_filter_results: tuple[FastFilterResult, ...]
    task_quality_results: tuple[TaskQualityResult, ...]
    selected_top: tuple[SelectedCandidate, ...]
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
    candidates = generate_candidates(inputs)
    timings["candidate_generation"] = time.perf_counter() - started

    started = time.perf_counter()
    predictor = SequentialClosurePredictor(inputs)
    predictions = tuple(predictor.predict(seed) for seed in candidates)
    timings["closure_prediction"] = time.perf_counter() - started

    started = time.perf_counter()
    filters = fast_filter_predictions(inputs, predictions)
    timings["fast_filter"] = time.perf_counter() - started

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
    selected = select_top_candidates(
        predictions,
        filters,
        qualities,
        top_k=top_k,
        path_clearance_by_id={
            row.candidate_id: row.sampled_hand_table_clearance_m for row in filters
        },
    )
    timings["selection"] = time.perf_counter() - started

    started = time.perf_counter()
    exact_results = validate_top_candidates(inputs, selected)
    timings["exact_validation"] = time.perf_counter() - started
    timings["total"] = sum(
        value for key, value in timings.items() if key != "total"
    )
    return OfflinePipelineResult(
        inputs=inputs,
        candidates=candidates,
        closure_predictions=predictions,
        fast_filter_results=filters,
        task_quality_results=qualities,
        selected_top=selected,
        exact_validation_results=exact_results,
        scenario_design=design,
        timings_s=timings,
    )


__all__ = ["OfflinePipelineResult", "run_offline_pipeline"]
