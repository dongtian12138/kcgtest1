"""Bounded real-contract profile for the staged CARTS candidate screen only.

This command never calls the exact V9 candidate evaluator, never writes a
generation checkpoint, and never launches Isaac.  It maps the frozen schedule
to canonical V9 parameters, removes exact duplicates, and measures only the
whole-path PAD sphere screen.  Progress is written atomically after every
phase so a supervisor timeout remains auditable.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from .production_candidate_generation import (
    DEVELOPMENT_OBJECT_ID,
    build_production_candidate_generation_runtime,
)
from .multifidelity_candidate_rank import (
    EXACT_TOP_K,
    METHOD_ID as PROXY_RANK_METHOD_ID,
    ProxyRankInput,
    rank_screened_candidates,
)
from .top_level_candidate_generator import (
    LANE_SPECS,
    MAIN_TOTAL_ATTEMPT_BUDGET,
    TopLevelCandidateGeneratorError,
    canonicalize_v9_parameters,
)

_SCREEN_CHUNK_SIZE = MAIN_TOTAL_ATTEMPT_BUDGET
_MAXIMUM_SCREEN_SECONDS = 20.0
_MAXIMUM_PROXY_RANK_SECONDS = 5.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(
            document,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def profile(
    *,
    repository_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    document: dict[str, Any] = {
        "schema_version": "carts_multifidelity_proxy_rank_profile_v8",
        "recorded_at_utc": _utc_now(),
        "status": "STARTED",
        "object_id": DEVELOPMENT_OBJECT_ID,
        "scheduled_attempt_count": MAIN_TOTAL_ATTEMPT_BUDGET,
        "exact_v9_evaluator_called": False,
        "generation_checkpoint_written": False,
        "isaac_launched": False,
    }
    _atomic_write(output_path, document)
    try:
        build_started = time.perf_counter()
        runtime = build_production_candidate_generation_runtime(
            repository_root=repository_root,
            object_id=DEVELOPMENT_OBJECT_ID,
        )
        document.update(
            {
                "status": "MODEL_BUILT",
                "run_id_if_generation_were_authorized": runtime.run_id,
                "model_build_seconds": time.perf_counter() - build_started,
            }
        )
        _atomic_write(output_path, document)

        proposal_started = time.perf_counter()
        canonical_by_key: dict[bytes, object] = {}
        multiplicity_by_key: dict[bytes, int] = {}
        first_lane_by_key: dict[bytes, str] = {}
        first_attempt_index_by_key: dict[bytes, int] = {}
        lane_attempt_keys: dict[str, list[bytes]] = {
            spec.lane.value: [] for spec in LANE_SPECS
        }
        terminal_attempt_count_by_lane: dict[str, int] = {
            spec.lane.value: 0 for spec in LANE_SPECS
        }
        proposal_terminal_attempt_count = 0
        proposal_status_counts: dict[str, int] = {}
        canonical_domain_rejection_count = 0
        duplicate_attempt_count = 0
        proposal_full_closed_focus_reuse_count = 0
        generator = runtime.generator
        for attempt_index in range(MAIN_TOTAL_ATTEMPT_BUDGET):
            lane_point_index, lane_ordinal = divmod(
                attempt_index, len(LANE_SPECS)
            )
            spec = LANE_SPECS[lane_ordinal]
            lane_name = spec.lane.value
            sobol_point = np.array(
                generator._maximum_designs[spec.lane][lane_point_index],
                dtype=np.float64,
                copy=True,
            )
            mapped, _audit, _failure, proposal_status = (
                generator._proposal_for_attempt(
                    spec=spec,
                    sobol_point=sobol_point,
                )
            )
            if spec.anchor_pad_ordinal is not None and mapped is not None:
                proposal_full_closed_focus_reuse_count += 1
            if proposal_status is not None or mapped is None:
                proposal_terminal_attempt_count += 1
                terminal_attempt_count_by_lane[lane_name] += 1
                label = (
                    "MISSING_MAPPED_PARAMETERS"
                    if proposal_status is None
                    else proposal_status.value
                )
                proposal_status_counts[label] = (
                    proposal_status_counts.get(label, 0) + 1
                )
                continue
            try:
                canonical = canonicalize_v9_parameters(
                    mapped,
                    parameter_layout=generator.v9_parameter_layout,
                )
            except TopLevelCandidateGeneratorError:
                proposal_terminal_attempt_count += 1
                canonical_domain_rejection_count += 1
                terminal_attempt_count_by_lane[lane_name] += 1
                continue
            key = canonical.exact_key
            lane_attempt_keys[lane_name].append(key)
            if key in canonical_by_key:
                duplicate_attempt_count += 1
                multiplicity_by_key[key] += 1
            else:
                canonical_by_key[key] = canonical
                multiplicity_by_key[key] = 1
                first_lane_by_key[key] = lane_name
                first_attempt_index_by_key[key] = attempt_index
        proposal_seconds = time.perf_counter() - proposal_started
        parameters = np.asarray(
            [canonical.values for canonical in canonical_by_key.values()],
            dtype=np.float64,
        )
        document.update(
            {
                "status": "SCHEDULE_MAPPED",
                "proposal_mapping_seconds": proposal_seconds,
                "proposal_terminal_attempt_count": (
                    proposal_terminal_attempt_count
                ),
                "proposal_status_counts": dict(
                    sorted(proposal_status_counts.items())
                ),
                "canonical_domain_rejection_count": (
                    canonical_domain_rejection_count
                ),
                "duplicate_attempt_count": duplicate_attempt_count,
                "proposal_full_closed_focus_reuse_count": (
                    proposal_full_closed_focus_reuse_count
                ),
                "unique_canonical_screen_input_count": len(parameters),
                "screenable_attempt_count_by_lane_including_duplicates": {
                    lane: len(keys)
                    for lane, keys in lane_attempt_keys.items()
                },
                "terminal_attempt_count_by_lane": (
                    terminal_attempt_count_by_lane
                ),
            }
        )
        _atomic_write(output_path, document)
        if len(parameters) == 0:
            raise RuntimeError("fixed schedule produced no screenable parameters")

        keys = tuple(canonical_by_key)
        screens: list[tuple[object, ...]] = []
        screen_seconds = 0.0
        screen_chunks: list[dict[str, Any]] = []
        for chunk_start in range(0, len(parameters), _SCREEN_CHUNK_SIZE):
            chunk_stop = min(
                chunk_start + _SCREEN_CHUNK_SIZE, len(parameters)
            )
            chunk_started = time.perf_counter()
            chunk_screens = (
                runtime.closure_model.screen_unit_parameter_batch(
                    parameters[chunk_start:chunk_stop]
                )
            )
            chunk_seconds = time.perf_counter() - chunk_started
            screen_seconds += chunk_seconds
            screens.extend(chunk_screens)
            completed_rejected = sum(
                any(
                    row.certified_free
                    or row.certified_no_valid_contact
                    for row in candidate_rows
                )
                for candidate_rows in screens
            )
            geometric_free_rejected = sum(
                any(row.certified_free for row in candidate_rows)
                for candidate_rows in screens
            )
            directional_rejected = sum(
                any(
                    row.certified_no_valid_contact
                    for row in candidate_rows
                )
                for candidate_rows in screens
            )
            screen_chunks.append(
                {
                    "chunk_start_unique_index": chunk_start,
                    "chunk_stop_unique_index_exclusive": chunk_stop,
                    "chunk_unique_count": chunk_stop - chunk_start,
                    "chunk_screen_seconds": chunk_seconds,
                    "cumulative_screen_compute_seconds": screen_seconds,
                    "cumulative_rejected_unique_count": (
                        completed_rejected
                    ),
                    "cumulative_survivor_unique_count": (
                        len(screens) - completed_rejected
                    ),
                }
            )
            document.update(
                {
                    "status": "SCREENING",
                    "screen_chunk_size": _SCREEN_CHUNK_SIZE,
                    "screen_completed_unique_count": len(screens),
                    "screen_compute_seconds_so_far": screen_seconds,
                    "screen_chunks": screen_chunks,
                    "screen_rejected_unique_count_so_far": (
                        completed_rejected
                    ),
                    "screen_survivor_unique_count_so_far": (
                        len(screens) - completed_rejected
                    ),
                    "screen_geometric_free_rejected_unique_count_so_far": (
                        geometric_free_rejected
                    ),
                    "screen_directional_no_valid_contact_rejected_unique_count_so_far": (
                        directional_rejected
                    ),
                    "screen_full_closed_focus_reuse_count_so_far": len(
                        screens
                    ),
                    "screen_moving_triangle_sat_pair_test_count_so_far": sum(
                        item.moving_triangle_sat_pair_test_count
                        for row in screens
                        for item in row
                    ),
                    "screen_moving_triangle_sat_certified_free_pair_count_so_far": sum(
                        item.moving_triangle_sat_certified_free_pair_count
                        for row in screens
                        for item in row
                    ),
                    "screen_temporal_refined_leaf_pair_count_so_far": sum(
                        item.temporal_refined_leaf_pair_count
                        for row in screens
                        for item in row
                    ),
                    "screen_temporal_refinement_transform_count_so_far": sum(
                        item.temporal_refinement_transform_count
                        for row in screens
                        for item in row
                    ),
                    "screen_narrowphase_refined_unique_count_so_far": sum(
                        any(
                            item.narrowphase_refinement_used
                            for item in row
                        )
                        for row in screens
                    ),
                    "screen_narrowphase_refined_pad_count_so_far": sum(
                        item.narrowphase_refinement_used
                        for row in screens
                        for item in row
                    ),
                    "screen_narrowphase_work_budget_exhausted_pad_count_so_far": sum(
                        item.narrowphase_work_budget_exhausted
                        for row in screens
                        for item in row
                    ),
                    "screen_maximum_temporal_refinement_depth_reached_so_far": max(
                        item.maximum_temporal_refinement_depth_reached
                        for row in screens
                        for item in row
                    ),
                    "screen_directional_contact_feasibility_used_unique_count_so_far": sum(
                        any(
                            item.directional_contact_feasibility_used
                            for item in row
                        )
                        for row in screens
                    ),
                    "screen_directional_contact_feasibility_used_pad_count_so_far": sum(
                        item.directional_contact_feasibility_used
                        for row in screens
                        for item in row
                    ),
                    "screen_directional_bvh_node_pair_test_count_so_far": sum(
                        item.directional_bvh_node_pair_test_count
                        for row in screens
                        for item in row
                    ),
                    "screen_directional_bvh_node_pair_rejected_count_so_far": sum(
                        item.directional_bvh_node_pair_rejected_count
                        for row in screens
                        for item in row
                    ),
                    "screen_directional_leaf_face_pair_test_count_so_far": sum(
                        item.directional_leaf_face_pair_test_count
                        for row in screens
                        for item in row
                    ),
                    "screen_directional_leaf_face_pair_rejected_count_so_far": sum(
                        item.directional_leaf_face_pair_rejected_count
                        for row in screens
                        for item in row
                    ),
                    "screen_directional_interval_witness_motion_evaluation_count_so_far": sum(
                        item.directional_interval_witness_motion_evaluation_count
                        for row in screens
                        for item in row
                    ),
                    "projected_full_screen_compute_seconds": (
                        screen_seconds * len(parameters) / len(screens)
                    ),
                    "projected_full_post_build_schedule_and_screen_seconds": (
                        proposal_seconds
                        + screen_seconds * len(parameters) / len(screens)
                    ),
                    "projected_full_survivor_unique_count": (
                        (len(screens) - completed_rejected)
                        * len(parameters)
                        / len(screens)
                    ),
                }
            )
            _atomic_write(output_path, document)
            if any(
                item.moving_triangle_sat_pair_test_count != 0
                or item.temporal_refined_leaf_pair_count != 0
                or item.narrowphase_refinement_used
                for row in screens
                for item in row
            ):
                raise RuntimeError(
                    "retired moving-triangle refinement entered production screen"
                )
            if any(
                item.directional_contact_feasibility_used
                or item.certified_no_valid_contact
                or item.directional_interval_witness_motion_evaluation_count
                != 0
                for row in screens
                for item in row
            ):
                raise RuntimeError(
                    "retired directional refinement entered production screen"
                )
        rejected_unique_indices = tuple(
            index
            for index, candidate_rows in enumerate(screens)
            if any(
                row.certified_free or row.certified_no_valid_contact
                for row in candidate_rows
            )
        )
        geometric_free_rejected_unique_indices = tuple(
            index
            for index, candidate_rows in enumerate(screens)
            if any(row.certified_free for row in candidate_rows)
        )
        directional_rejected_unique_indices = tuple(
            index
            for index, candidate_rows in enumerate(screens)
            if any(
                row.certified_no_valid_contact
                for row in candidate_rows
            )
        )
        rejected_key_set = {keys[index] for index in rejected_unique_indices}
        rejected_attempt_count = sum(
            multiplicity_by_key[key] for key in rejected_key_set
        )
        survivor_unique_count = len(parameters) - len(rejected_key_set)
        rank_started = time.perf_counter()
        rank_result = rank_screened_candidates(
            tuple(
                ProxyRankInput(
                    canonical_key_hex=key.hex(),
                    parameters_unit=tuple(
                        float(value) for value in parameters[index]
                    ),
                    first_attempt_index=first_attempt_index_by_key[key],
                    pad_screens=tuple(screens[index]),
                )
                for index, key in enumerate(keys)
            )
        )
        proxy_rank_seconds = time.perf_counter() - rank_started
        exact_selected_count = len(rank_result.exact_selected_keys)
        if exact_selected_count > EXACT_TOP_K:
            raise RuntimeError("proxy exact selection exceeded Top-K")
        performance_gate_passed = (
            screen_seconds <= _MAXIMUM_SCREEN_SECONDS
            and proxy_rank_seconds <= _MAXIMUM_PROXY_RANK_SECONDS
        )
        exact_evaluations_avoided = (
            MAIN_TOTAL_ATTEMPT_BUDGET - exact_selected_count
        )
        screen_result_by_lane: dict[str, dict[str, int]] = {}
        for lane_name, lane_keys in lane_attempt_keys.items():
            first_owned_keys = tuple(
                key
                for key, first_lane in first_lane_by_key.items()
                if first_lane == lane_name
            )
            rejected_attempts = sum(
                key in rejected_key_set for key in lane_keys
            )
            rejected_first_owned = sum(
                key in rejected_key_set for key in first_owned_keys
            )
            screen_result_by_lane[lane_name] = {
                "scheduled_attempt_count": (
                    MAIN_TOTAL_ATTEMPT_BUDGET // len(LANE_SPECS)
                ),
                "terminal_before_screen_count": (
                    terminal_attempt_count_by_lane[lane_name]
                ),
                "screenable_attempt_count_including_duplicates": len(
                    lane_keys
                ),
                "rejected_attempt_count_including_duplicates": (
                    rejected_attempts
                ),
                "survivor_attempt_count_including_duplicates": (
                    len(lane_keys) - rejected_attempts
                ),
                "first_owned_unique_count": len(first_owned_keys),
                "rejected_first_owned_unique_count": rejected_first_owned,
                "survivor_first_owned_unique_count": (
                    len(first_owned_keys) - rejected_first_owned
                ),
            }
        clearances = [
            row.minimum_clearance_lower_bound_m
            for candidate_rows in screens
            for row in candidate_rows
        ]
        document.update(
            {
                "status": (
                    "COMPLETED_TOP4_PROXY_BOUND"
                    if performance_gate_passed
                    else "COMPLETED_EVIDENCE_PERFORMANCE_GATE_MISS"
                ),
                "proxy_rank_method_id": PROXY_RANK_METHOD_ID,
                "proxy_rank_seconds": proxy_rank_seconds,
                "maximum_proxy_rank_seconds": _MAXIMUM_PROXY_RANK_SECONDS,
                "maximum_screen_seconds": _MAXIMUM_SCREEN_SECONDS,
                "screen_time_gate_passed": (
                    screen_seconds <= _MAXIMUM_SCREEN_SECONDS
                ),
                "proxy_rank_time_gate_passed": (
                    proxy_rank_seconds <= _MAXIMUM_PROXY_RANK_SECONDS
                ),
                "performance_gate_passed": performance_gate_passed,
                "proxy_rank_certifies_or_rejects": False,
                "proxy_rank_result": rank_result.as_dict(),
                "exact_top_k_ceiling": EXACT_TOP_K,
                "exact_selected_unique_count": exact_selected_count,
                "exact_selected_keys": list(rank_result.exact_selected_keys),
                "worst_case_exact_count_reduction_factor": (
                    MAIN_TOTAL_ATTEMPT_BUDGET / EXACT_TOP_K
                ),
                "screen_seconds": screen_seconds,
                "screen_chunk_size": _SCREEN_CHUNK_SIZE,
                "screen_completed_unique_count": len(screens),
                "screen_chunks": screen_chunks,
                "post_build_schedule_and_screen_seconds": (
                    proposal_seconds + screen_seconds
                ),
                "screened_pad_count": sum(len(row) for row in screens),
                "screen_evaluated_pad_count": sum(
                    not item.skipped_due_to_other_pad_free
                    for row in screens
                    for item in row
                ),
                "screen_skipped_pad_count_after_candidate_rejection": sum(
                    item.skipped_due_to_other_pad_free
                    for row in screens
                    for item in row
                ),
                "screened_sphere_node_count": sum(
                    item.spatial_node_query_count
                    for row in screens
                    for item in row
                ),
                "screen_aabb_certified_free_node_count": sum(
                    item.aabb_certified_free_node_count
                    for row in screens
                    for item in row
                ),
                "screen_exact_distance_query_count": sum(
                    item.exact_distance_query_count
                    for row in screens
                    for item in row
                ),
                "screen_obb_sat_certified_free_node_count": sum(
                    item.obb_sat_certified_free_node_count
                    for row in screens
                    for item in row
                ),
                "screen_obb_sat_triangle_test_count": sum(
                    item.obb_sat_triangle_test_count
                    for row in screens
                    for item in row
                ),
                "screen_moving_triangle_sat_pair_test_count": sum(
                    item.moving_triangle_sat_pair_test_count
                    for row in screens
                    for item in row
                ),
                "screen_moving_triangle_sat_certified_free_pair_count": sum(
                    item.moving_triangle_sat_certified_free_pair_count
                    for row in screens
                    for item in row
                ),
                "screen_temporal_refined_leaf_pair_count": sum(
                    item.temporal_refined_leaf_pair_count
                    for row in screens
                    for item in row
                ),
                "screen_temporal_refinement_transform_count": sum(
                    item.temporal_refinement_transform_count
                    for row in screens
                    for item in row
                ),
                "screen_narrowphase_refined_unique_count": sum(
                    any(
                        item.narrowphase_refinement_used
                        for item in row
                    )
                    for row in screens
                ),
                "screen_narrowphase_refined_pad_count": sum(
                    item.narrowphase_refinement_used
                    for row in screens
                    for item in row
                ),
                "screen_narrowphase_work_budget_exhausted_pad_count": sum(
                    item.narrowphase_work_budget_exhausted
                    for row in screens
                    for item in row
                ),
                "screen_maximum_temporal_refinement_depth_reached": max(
                    item.maximum_temporal_refinement_depth_reached
                    for row in screens
                    for item in row
                ),
                "screen_directional_contact_feasibility_used_unique_count": sum(
                    any(
                        item.directional_contact_feasibility_used
                        for item in row
                    )
                    for row in screens
                ),
                "screen_directional_contact_feasibility_used_pad_count": sum(
                    item.directional_contact_feasibility_used
                    for row in screens
                    for item in row
                ),
                "screen_directional_bvh_node_pair_test_count": sum(
                    item.directional_bvh_node_pair_test_count
                    for row in screens
                    for item in row
                ),
                "screen_directional_bvh_node_pair_rejected_count": sum(
                    item.directional_bvh_node_pair_rejected_count
                    for row in screens
                    for item in row
                ),
                "screen_directional_leaf_face_pair_test_count": sum(
                    item.directional_leaf_face_pair_test_count
                    for row in screens
                    for item in row
                ),
                "screen_directional_leaf_face_pair_rejected_count": sum(
                    item.directional_leaf_face_pair_rejected_count
                    for row in screens
                    for item in row
                ),
                "screen_directional_interval_witness_motion_evaluation_count": sum(
                    item.directional_interval_witness_motion_evaluation_count
                    for row in screens
                    for item in row
                ),
                "screen_directional_certified_no_valid_contact_pad_count": sum(
                    item.certified_no_valid_contact
                    for row in screens
                    for item in row
                ),
                "screen_full_closed_focus_reuse_count": len(screens),
                "screen_maximum_spatial_depth_reached": max(
                    item.maximum_spatial_depth_reached
                    for row in screens
                    for item in row
                ),
                "screen_bvh_node_visits": sum(
                    item.distance_bvh_node_visits
                    for row in screens
                    for item in row
                ),
                "screen_triangle_tests": sum(
                    item.distance_triangle_tests
                    for row in screens
                    for item in row
                ),
                "screen_rejected_unique_count": len(rejected_key_set),
                "screen_geometric_free_rejected_unique_count": len(
                    geometric_free_rejected_unique_indices
                ),
                "screen_directional_no_valid_contact_rejected_unique_count": len(
                    directional_rejected_unique_indices
                ),
                "screen_rejected_attempt_count_including_duplicates": (
                    rejected_attempt_count
                ),
                "screen_survivor_unique_count": survivor_unique_count,
                "survivor_unique_exact_evaluation_count": exact_selected_count,
                "screen_result_by_lane": screen_result_by_lane,
                "exact_evaluations_avoided_count": exact_evaluations_avoided,
                "exact_evaluations_avoided_fraction": (
                    exact_evaluations_avoided
                    / MAIN_TOTAL_ATTEMPT_BUDGET
                ),
                "minimum_sphere_clearance_lower_bound_m": min(clearances),
                "maximum_sphere_clearance_lower_bound_m": max(clearances),
                "total_wall_seconds": time.perf_counter() - started,
                "exact_v9_evaluator_called": False,
                "generation_checkpoint_written": False,
                "isaac_launched": False,
            }
        )
        _atomic_write(output_path, document)
        return document
    except Exception as error:
        document.update(
            {
                "status": "FAILED",
                "failure": f"{type(error).__name__}:{error}",
                "total_wall_seconds": time.perf_counter() - started,
                "exact_v9_evaluator_called": False,
                "generation_checkpoint_written": False,
                "isaac_launched": False,
            }
        )
        _atomic_write(output_path, document)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = profile(
        repository_root=arguments.repository_root.resolve(),
        output_path=arguments.output.resolve(),
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "screen_seconds": result.get(
                    "screen_seconds",
                    result.get("screen_compute_seconds_so_far"),
                ),
                "survivor_unique_exact_evaluation_count": result.get(
                    "survivor_unique_exact_evaluation_count",
                    result.get("screen_survivor_unique_count_so_far"),
                ),
                "output": str(arguments.output.resolve()),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
