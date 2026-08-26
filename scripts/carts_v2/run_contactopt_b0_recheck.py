#!/usr/bin/env python3
"""Recheck fixed CONTACTOPT poses under B0 semantics before any Isaac launch."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import numpy as np
import yaml

from kcg_connector.grasp.carts_v2.b0_surface_semantics import (
    b0_nominal_pickup_task_pass, b0_sampled_table_clearance_pass,
    b0_surface_audit,
    bind_b0_external_load_bearing_surfaces,
)
from kcg_connector.grasp.carts_v2.closure_predictor import SequentialClosurePredictor
from kcg_connector.grasp.carts_v2.contact_interval_solver import ProxyContactIntervalEvaluator
from kcg_connector.grasp.carts_v2.contact_constrained_optimizer import (
    optimize_contact_constrained_top48,
)
from kcg_connector.grasp.carts_v2.fast_filter import (
    fast_filter_predictions, fast_filter_pregrasp_paths,
)
from kcg_connector.grasp.carts_v2.models import CandidateSeed, file_sha256, load_v2_inputs
from kcg_connector.grasp.carts_v2.selector import nominal_research_task_pass
from kcg_connector.grasp.carts_v2.task_quality import (
    common_uncertainty_design, evaluate_task_quality,
)
from kcg_connector.grasp.robust.bounded_hand_base_ik import (
    CandidateJointRouteError, solve_bounded_hand_base_ik,
)


ROOT = Path(__file__).resolve().parents[2]


def _args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--seed-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--project-table", action="store_true")
    return parser.parse_args()


def _resolve(root, path):
    value = Path(path)
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _seed(row):
    values = dict(row)
    for key in ("anchor_position_object_m", "object_from_hand",
                "pregrasp_joint_positions_rad", "pregrasp_closure_phases"):
        values[key] = tuple(values[key])
    if values.get("approach_direction_object") is not None:
        values["approach_direction_object"] = tuple(values["approach_direction_object"])
    return CandidateSeed(**values)


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


def _bounded_ik(inputs, seed):
    target = inputs.frozen_world_from_object @ seed.object_from_hand_matrix()
    try:
        joints, position, orientation, index = solve_bounded_hand_base_ik(
            inputs.config.section("ik")["solver"], model=inputs.robot_model,
            hand_positions=seed.pregrasp_joint_positions_rad,
            target_world_from_hand_base=target, label=seed.candidate_id)
        result = {"status": "BOUNDED_IK_PASS_NOT_PATH_COLLISION",
                  "arm_joint_positions_rad": list(joints),
                  "position_error_m": position,
                  "orientation_error_rad": orientation,
                  "selected_seed_index": index}
    except CandidateJointRouteError as error:
        result = {"status": "IK_REJECT", "code": error.code,
                  "detail": error.detail}
    return result, target


def _evaluate(inputs, evaluator, predictor, design, seed, specification):
    started = time.perf_counter()
    cheap, interval = evaluator.evaluate(seed, specification)
    row = {"candidate_id": seed.candidate_id, "input_seed": asdict(seed),
           "cheap": cheap, "proxy_interval": interval,
           "sampled_raw_mesh_geometry_pass": False,
           "sampled_table_operation_clearance_pass": False,
           "nominal_12n_task_pass": False,
           "six_axis_disturbance_margin_pass": False,
           "local_isaac_input_ready": False}
    if interval.get("status") != "PROXY_INTERVAL_SURVIVE":
        row["elapsed_s"] = time.perf_counter() - started
        return row
    prediction = predictor.predict(seed)
    pregrasp = fast_filter_pregrasp_paths(
        inputs, ((seed, seed.pregrasp_closure_phases),))[0]
    filtered = fast_filter_predictions(inputs, (prediction,))[0]
    required_table = float(inputs.config.section("height_projection")[
        "table_operation_clearance_m"])
    table_tolerance = float(inputs.config.section("fast_filter")[
        "table_penetration_tolerance_m"])
    table_pass = b0_sampled_table_clearance_pass(
        pregrasp.get("minimum_table_clearance_m"),
        filtered.minimum_table_clearance_m, required_table, table_tolerance)
    geometry = bool(prediction.status == "CLOSURE_SURVIVE"
                    and pregrasp.get("accepted") is True
                    and filtered.status == "FAST_SURVIVE" and table_pass)
    row.update({"closure_prediction": asdict(prediction),
                "pregrasp_filter": pregrasp, "closure_filter": asdict(filtered),
                "sampled_raw_mesh_geometry_pass": geometry,
                "sampled_table_operation_clearance_pass": table_pass,
                "required_table_operation_clearance_m": required_table,
                "table_clearance_numerical_tolerance_m": table_tolerance})
    if geometry:
        quality = evaluate_task_quality(inputs, prediction, design)
        operation_cap = float(inputs.config.section("task_quality")[
            "normal_force_operation_cap_n"])
        task_pass = b0_nominal_pickup_task_pass(quality, operation_cap)
        ik, target = _bounded_ik(inputs, seed) if task_pass else (
            {"status": "NOT_RUN_B0_TASK_INELIGIBLE"},
            inputs.frozen_world_from_object @ seed.object_from_hand_matrix())
        row.update({"task_quality": asdict(quality),
                    "nominal_12n_task_pass": task_pass,
                    "six_axis_disturbance_margin_pass": (
                        nominal_research_task_pass(quality)),
                    "target_world_from_handbase_row_major": list(target.ravel()),
                    "bounded_ik": ik,
                    "full_arm_path_collision_checked": False,
                    "local_isaac_input_ready": bool(task_pass and ik["status"]
                        == "BOUNDED_IK_PASS_NOT_PATH_COLLISION")})
    row["elapsed_s"] = time.perf_counter() - started
    return row


def _project_table(inputs, evaluator, seeds, specifications):
    required, bound = (float(inputs.config.section("height_projection")[
        "table_operation_clearance_m"]), 0.020)
    world_rotation = np.asarray(inputs.frozen_world_from_object[:3, :3])
    projected, records = [], []
    for seed in seeds:
        cheap = evaluator.evaluate_cheap(seed, specifications[seed.candidate_id])
        margin = float(cheap["table_margin_m"])
        rise = max(0.0, required - margin)
        status = "PROJECTED" if rise <= bound else "PROJECTION_EXCEEDS_FIXED_20MM_BOUND"
        row = {"candidate_id": seed.candidate_id,
               "input_table_margin_m": margin, "required_rise_m": rise,
               "translation_bound_m": bound, "status": status}
        if rise <= bound:
            pose = seed.object_from_hand_matrix().copy()
            pose[:3, 3] += world_rotation.T @ np.asarray((0.0, 0.0, rise))
            moved = replace(seed, object_from_hand=tuple(float(value) for value in pose.ravel()))
            after = evaluator.evaluate_cheap(moved, specifications[seed.candidate_id])
            row["projected_table_margin_m"] = after["table_margin_m"]
            projected.append(moved)
        records.append(row)
    return tuple(projected), records


def main():
    args, started = _args(), time.perf_counter()
    root, output = args.repository_root.resolve(), _resolve(args.repository_root.resolve(), args.output)
    manifest_path = _resolve(root, args.seed_manifest)
    _require(manifest_path.is_file() and not output.exists(), "B0 input/output path invalid")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = _resolve(root, manifest["base_physical_config"])
    method = _resolve(root, manifest["method_config"])
    _require(file_sha256(base) == manifest["base_physical_config_sha256"],
             "base physical configuration changed")
    _require(file_sha256(method) == manifest["method_config_sha256"],
             "CONTACTOPT method configuration changed")
    inputs = bind_b0_external_load_bearing_surfaces(load_v2_inputs(
        root, config_path=base, object_id=manifest["object_id"]))
    _require(inputs.object_contract.model.provenance.source_sha256
             == manifest["object_mesh_sha256"], "object mesh identity changed")
    specifications = {row["candidate_id"]: row
                      for row in manifest["audit"]["specifications"]}
    seeds = tuple(_seed(row) for row in manifest["generated_candidates"])
    evaluator = ProxyContactIntervalEvaluator(inputs)
    projection = None
    if args.project_table:
        seeds, projection = _project_table(inputs, evaluator, seeds, specifications)
    optimization = None
    if args.optimize:
        method_values = yaml.safe_load(method.read_text(encoding="utf-8"))
        seeds, optimization = optimize_contact_constrained_top48(
            inputs, seeds, tuple(specifications.values()),
            method_values["contact_optimization"])
    predictor = SequentialClosurePredictor(inputs)
    design, rows = common_uncertainty_design(inputs), []
    result = {"schema_version": "carts_contactopt_b0_recheck_v1",
              "claim_scope": "B0_OFFLINE_PROXY_RAW_MESH_AND_NOMINAL_TASK_NOT_DYNAMIC_SUCCESS",
              "hardware_authorized": False, "formal_dynamic_pass": False,
              "research_dynamic_pass": False,
              "seed_manifest": str(manifest_path),
              "seed_manifest_sha256": file_sha256(manifest_path),
              "base_physical_config": str(base),
              "base_physical_config_sha256": file_sha256(base),
              "method_config": str(method),
              "method_config_sha256": file_sha256(method),
              "object_id": inputs.object_contract.object_id,
              "object_mesh_sha256": inputs.object_contract.model.provenance.source_sha256,
              "b0_surface_audit": b0_surface_audit(inputs),
              "table_projection": projection,
              "optimization": optimization, "candidates": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        rows.append(_evaluate(inputs, evaluator, predictor, design, seed,
                              specifications[seed.candidate_id]))
        result["completed_count"] = len(rows)
        result["elapsed_s"] = time.perf_counter() - started
        output.write_text(json.dumps(_json_ready(result), indent=2, sort_keys=True,
                                     allow_nan=False) + "\n", encoding="utf-8")
    result["local_isaac_input_count"] = sum(
        row["local_isaac_input_ready"] for row in rows)
    result["source"] = {"path": str(Path(__file__).resolve()),
                        "sha256": file_sha256(Path(__file__).resolve())}
    result["elapsed_s"] = time.perf_counter() - started
    output.write_text(json.dumps(_json_ready(result), indent=2, sort_keys=True,
                                 allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "completed": len(rows),
                      "isaac_inputs": result["local_isaac_input_count"],
                      "elapsed_s": result["elapsed_s"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
