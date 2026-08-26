#!/usr/bin/env python3
"""Exact height, closure, task-load and bounded-IK checks for one shortlist shard."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from kcg_connector.grasp.carts_v2.closure_predictor import SequentialClosurePredictor
from kcg_connector.grasp.carts_v2.fast_filter import (
    fast_filter_predictions, fast_filter_pregrasp_paths,
)
from kcg_connector.grasp.carts_v2.height_projected_search import (
    SampledPathEnvelope, contact_height_bounds, sampled_height_path_states,
    search_height_projected_pregrasps,
)
from kcg_connector.grasp.carts_v2.models import CandidateSeed, file_sha256, load_v2_inputs
from kcg_connector.grasp.carts_v2.selector import nominal_research_task_pass
from kcg_connector.grasp.carts_v2.surface_contact import ExactContactSurfaceQuery
from kcg_connector.grasp.carts_v2.task_quality import (
    common_uncertainty_design, evaluate_task_quality,
)
from kcg_connector.grasp.robust.bounded_hand_base_ik import (
    CandidateJointRouteError, solve_bounded_hand_base_ik,
)


ROOT = Path(__file__).resolve().parents[2]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--feature-report", type=Path, required=True)
    parser.add_argument("--shortlist-offset", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _seed(row: dict) -> CandidateSeed:
    values = dict(row)
    for name in ("anchor_position_object_m", "object_from_hand",
                 "pregrasp_joint_positions_rad", "pregrasp_closure_phases"):
        values[name] = tuple(values[name])
    if values.get("approach_direction_object") is not None:
        values["approach_direction_object"] = tuple(values["approach_direction_object"])
    return CandidateSeed(**values)


def _pregrasp_key(inputs, query, seed, cache) -> tuple[float, ...]:
    transforms = inputs.hand_model.forward_kinematics(
        seed.pregrasp_joint_positions_rad,
        base_transform=seed.object_from_hand_matrix())
    maximum = int(inputs.config.section(
        "closure_prediction")["nearest_face_candidate_count"])
    distances = []
    for name, surface in sorted(inputs.task_grip_surfaces.items()):
        transform = np.asarray(transforms[surface.link_name], dtype="<f8")
        key = name, transform.tobytes()
        if key not in cache:
            nearest, _points, _normals = query.query_task_surface_witnesses(
                name, transform, maximum)
            cache[key] = float(np.min(nearest.distance_m))
        distances.append(cache[key])
    return max(distances), sum(distances), *seed.pregrasp_closure_phases


def _project(inputs, seed, predictor, query, cache):
    height = inputs.config.section("height_projection")
    fast = inputs.config.section("fast_filter")
    return search_height_projected_pregrasps(
        inputs, seed, predictor,
        sampled_path_envelope=lambda bound, phases: SampledPathEnvelope(
            tuple(sampled_height_path_states(inputs, bound, phases)),
            "REGISTERED_CONTROL_STEPS_PALM_PRESHAPE_APPROACH_"
            "SEQUENTIAL_CLOSURE_PRELOAD_LIFT_START"),
        pregrasp_contact_key=lambda bound: _pregrasp_key(
            inputs, query, bound, cache),
        pregrasp_path_callback=lambda bound: fast_filter_pregrasp_paths(
            inputs, ((bound, bound.pregrasp_closure_phases),))[0],
        fast_filter_callback=lambda prediction: fast_filter_predictions(
            inputs, (prediction,))[0],
        contact_height_bounds_m=contact_height_bounds(inputs, seed),
        table_numerical_tolerance_m=float(fast["table_penetration_tolerance_m"]),
        required_table_clearance_m=float(height["table_operation_clearance_m"]),
        maximum_exact_variants=1,
        selected_pregrasp_phases=seed.pregrasp_closure_phases)


def _bounded_ik(inputs, seed) -> dict:
    target = inputs.frozen_world_from_object @ seed.object_from_hand_matrix()
    try:
        joints, position, orientation, index = solve_bounded_hand_base_ik(
            inputs.config.section("ik")["solver"], model=inputs.robot_model,
            hand_positions=seed.pregrasp_joint_positions_rad,
            target_world_from_hand_base=target, label=seed.candidate_id)
        return {"status": "BOUNDED_IK_PASS_NOT_PATH_COLLISION",
                "arm_joint_positions_rad": list(joints), "position_error_m": position,
                "orientation_error_rad": orientation, "selected_seed_index": index}
    except CandidateJointRouteError as error:
        return {"status": "IK_REJECT", "code": error.code, "detail": error.detail}


def _evaluate(inputs, seed, predictor, query, cache) -> dict:
    started = time.perf_counter()
    survivors, audit = _project(inputs, seed, predictor, query, cache)
    row = {"candidate_id": seed.candidate_id, "input_seed": asdict(seed),
           "height_search": audit, "survivor_count": len(survivors),
           "sampled_exact_geometry_pass": False, "local_research_candidate": False,
           "geometry_evidence_scope": "RAW_MESH_REGISTERED_CONTROL_STEP_SAMPLES_NOT_CONTINUOUS_CERTIFICATE"}
    if len(survivors) != 1:
        row["elapsed_s"] = time.perf_counter() - started
        return row
    projected = survivors[0]
    prediction = predictor.predict(projected)
    pregrasp = fast_filter_pregrasp_paths(
        inputs, ((projected, projected.pregrasp_closure_phases),))[0]
    filtered = fast_filter_predictions(inputs, (prediction,))[0]
    row.update({"projected_seed": asdict(projected),
                "closure_prediction": asdict(prediction),
                "pregrasp_filter": pregrasp, "closure_filter": asdict(filtered)})
    geometry = bool(prediction.status == "CLOSURE_SURVIVE"
                    and pregrasp.get("accepted") is True
                    and filtered.status == "FAST_SURVIVE")
    row["sampled_exact_geometry_pass"] = geometry
    if geometry:
        quality = evaluate_task_quality(
            inputs, prediction, common_uncertainty_design(inputs))
        row["task_quality"] = asdict(quality)
        row["bounded_ik"] = _bounded_ik(inputs, projected)
        row["local_research_candidate"] = nominal_research_task_pass(quality)
    row["elapsed_s"] = time.perf_counter() - started
    return row


def main() -> int:
    args, started = _arguments(), time.perf_counter()
    root = args.repository_root.resolve()
    feature, output = _resolve(root, args.feature_report), _resolve(root, args.output)
    if output.exists() or not feature.is_file() or not 1 <= args.count <= 8:
        raise ValueError("exact shard output/input/budget is invalid")
    source = json.loads(feature.read_text(encoding="utf-8"))
    config = Path(source["config"])
    inputs = load_v2_inputs(root, config_path=config, object_id=source["object_id"])
    if (file_sha256(config) != source["config_sha256"]
            or inputs.object_contract.model.provenance.source_sha256
            != source["object_mesh_sha256"]):
        raise ValueError("feature report configuration or object identity changed")
    all_seeds = tuple(_seed(row) for row in source["exact_shortlist"])
    selected = all_seeds[args.shortlist_offset:args.shortlist_offset + args.count]
    if len(selected) != args.count:
        raise ValueError("exact shortlist shard exceeds the registered budget")
    predictor, query, cache = (SequentialClosurePredictor(inputs),
                               ExactContactSurfaceQuery(inputs), {})
    result = {"schema_version": "carts_surface_v2_exact_shortlist_shard_v1",
              "claim_scope": "SAMPLED_RAW_MESH_OFFLINE_GEOMETRY_TASK_AND_IK_NOT_CONTINUOUS_OR_DYNAMIC_SUCCESS",
              "hardware_authorized": False, "formal_dynamic_pass": False,
              "research_dynamic_pass": False,
              "feature_report": str(feature), "feature_report_sha256": file_sha256(feature),
              "shortlist_offset": args.shortlist_offset, "requested_count": args.count,
              "completed_count": 0, "candidates": []}
    output.parent.mkdir(parents=True, exist_ok=True)
    for seed in selected:
        result["candidates"].append(_evaluate(inputs, seed, predictor, query, cache))
        result["completed_count"] = len(result["candidates"])
        result["elapsed_s"] = time.perf_counter() - started
        output.write_text(json.dumps(result, indent=2, sort_keys=True,
                                     allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "completed_count": result["completed_count"],
                      "geometry_pass_count": sum(row["sampled_exact_geometry_pass"]
                      for row in result["candidates"]), "elapsed_s": result["elapsed_s"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
