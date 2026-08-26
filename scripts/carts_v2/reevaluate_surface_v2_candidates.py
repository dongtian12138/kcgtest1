#!/usr/bin/env python3
"""Re-evaluate fixed legacy opposition poses with Surface-V2 semantics."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from kcg_connector.grasp.carts_v2.closure_predictor import SequentialClosurePredictor
from kcg_connector.grasp.carts_v2.fast_filter import (
    fast_filter_predictions, fast_filter_pregrasp_paths,
)
from kcg_connector.grasp.carts_v2.models import (
    FACE_ROLE_NAMES, CandidateSeed, file_sha256, load_v2_inputs,
)
from kcg_connector.grasp.carts_v2.selector import select_candidate_rankings
from kcg_connector.grasp.carts_v2.surface_contact import ExactContactSurfaceQuery
from kcg_connector.grasp.carts_v2.task_quality import (
    common_uncertainty_design, evaluate_task_quality,
)
from kcg_connector.grasp.robust.bounded_hand_base_ik import (
    CandidateJointRouteError, solve_bounded_hand_base_ik,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--audit-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _candidate(row: dict, suffix: str) -> CandidateSeed:
    return CandidateSeed(
        candidate_id=f"{row['candidate_id']}__surface_v2_{suffix}",
        object_id=row["object_id"], anchor_face_index=int(row["anchor_face_index"]),
        anchor_position_object_m=tuple(row["anchor_position_object_m"]),
        object_from_hand=tuple(row["object_from_hand_row_major"]),
        pregrasp_joint_positions_rad=tuple(row["pregrasp_joint_positions_rad"]),
        pregrasp_closure_phases=tuple(row["pregrasp_closure_phases"]),
        source_sample_index=0, approach_direction_object=tuple(
            row["approach_direction_object"]),
        palm_configuration_rad=float(row["palm_configuration_rad"]),
    )


def _projected_legacy_anchor(report: dict, inputs) -> dict | None:
    anchor = report.get("selected_geometric_anchor")
    evaluated = (report.get("height_search") or {}).get("evaluated") or []
    if not anchor or not evaluated:
        return None
    phases = tuple(float(value) for value in anchor["pregrasp_closure_phases"])
    row = next((item for item in evaluated
                if tuple(item.get("pregrasp_closure_phases", ())) == phases), evaluated[0])
    iterations = row.get("contact_conditioned_iterations") or []
    if not iterations:
        return None
    world = inputs.frozen_world_from_object @ np.asarray(
        anchor["object_from_hand_row_major"], dtype=np.float64).reshape(4, 4)
    world[2, 3] = float(iterations[-1]["projected_handbase_world_z_m"])
    projected = np.linalg.inv(inputs.frozen_world_from_object) @ world
    result = dict(anchor)
    result["object_from_hand_row_major"] = projected.ravel().tolist()
    return result


def _old_row(report: dict, phases: tuple[float, ...]) -> dict:
    variant = next((row for row in report.get("variant_results", ())
                    if tuple(row.get("pregrasp_closure_phases", ())) == phases), {})
    task = next((row for row in report.get("task_and_bounded_ik", ())
                 if row.get("candidate_id") == report.get(
                     "selected_geometric_anchor", {}).get("candidate_id")), {})
    return {"binary_method": "LEGACY_RADIAL_ONLY_BASELINE",
            "contact_status": variant.get("contact_status"),
            "contact_reason": variant.get("contact_reason"),
            "three_contact_count": variant.get("three_contact_count", 0),
            "fast_filter_status": variant.get("fast_filter_status"),
            "task_quality": task.get("task_quality"),
            "bounded_ik": task.get("bounded_ik")}


def _regions(inputs, query, prediction) -> list[dict]:
    transforms = inputs.hand_model.forward_kinematics(
        prediction.final_joint_positions_rad,
        base_transform=prediction.seed.object_from_hand_matrix())
    rows = []
    for contact in prediction.contacts:
        surface = inputs.task_grip_surfaces[contact.pad_name]
        nearest, _points, _normals = query.query_task_surface_witnesses(
            contact.pad_name, transforms[surface.link_name], 16)
        rows.append({
            "finger": contact.pad_name,
            "representative_object_face": contact.object_face_index,
            "representative_role": contact.object_surface_role,
            "hard_forbidden_minimum_gap_m": nearest.forbidden_distance_m,
            "region_witness_count": contact.region_witness_count,
            "region_triangle_area_m2": contact.region_triangle_area_m2,
            "primary_sampled_hand_patch_area_fraction": (
                contact.region_primary_sampled_hand_patch_area_fraction),
            "secondary_sampled_hand_patch_area_fraction": (
                contact.region_secondary_sampled_hand_patch_area_fraction),
            "selected_source_object_normal": list(
                contact.path_local_free_side_normal_object),
            "region_composite_normal_object": contact.region_composite_normal_object,
            "normal_dispersion_rad": contact.region_normal_dispersion_rad,
        })
    return rows


def _evaluate(inputs, seed: CandidateSeed, old: dict) -> dict:
    predictor = SequentialClosurePredictor(inputs)
    prediction = predictor.predict(seed)
    result = {"candidate_id": seed.candidate_id,
              "pregrasp_closure_phases": list(seed.pregrasp_closure_phases),
              "fixed_object_from_hand_row_major": list(seed.object_from_hand),
              "legacy_result": old, "new_closure_status": prediction.status,
              "new_closure_reason": prediction.reason,
              "new_contact_count": len(prediction.contacts),
              "contact_regions": [], "fast_filter": None,
              "task_quality": None, "bounded_ik": None,
              "research_candidate": False}
    if prediction.status != "CLOSURE_SURVIVE":
        return result
    query = ExactContactSurfaceQuery(inputs)
    result["contact_regions"] = _regions(inputs, query, prediction)
    pregrasp = fast_filter_pregrasp_paths(
        inputs, ((seed, seed.pregrasp_closure_phases),))[0]
    fast = fast_filter_predictions(inputs, (prediction,))[0]
    result["fast_filter"] = {"pregrasp": pregrasp,
                             "closure": asdict(fast)}
    if pregrasp.get("accepted") is not True or fast.status != "FAST_SURVIVE":
        return result
    quality = evaluate_task_quality(inputs, prediction,
                                    common_uncertainty_design(inputs))
    result["task_quality"] = asdict(quality)
    research, _formal, _diagnostic = select_candidate_rankings(
        (prediction,), (fast,), (quality,), top_k=1,
        path_clearance_by_id={seed.candidate_id: min(
            float(pregrasp["minimum_table_clearance_m"]),
            float(fast.minimum_table_clearance_m))})
    result["research_candidate"] = bool(research)
    if not research:
        return result
    target = inputs.frozen_world_from_object @ seed.object_from_hand_matrix()
    try:
        joints, position_error, orientation_error, index = solve_bounded_hand_base_ik(
            inputs.config.section("ik")["solver"], model=inputs.robot_model,
            hand_positions=seed.pregrasp_joint_positions_rad,
            target_world_from_hand_base=target, label=seed.candidate_id)
        result["bounded_ik"] = {"status": "BOUNDED_IK_PASS_NOT_PATH_COLLISION",
            "arm_joint_positions_rad": list(joints), "position_error_m": position_error,
            "orientation_error_rad": orientation_error, "selected_seed_index": index}
    except CandidateJointRouteError as error:
        result["bounded_ik"] = {"status": "IK_REJECT", "code": error.code,
                                "detail": error.detail}
    return result


def main() -> int:
    args, started = _arguments(), time.perf_counter()
    root = args.repository_root.resolve()
    config = args.config if args.config.is_absolute() else root / args.config
    output = args.output if args.output.is_absolute() else root / args.output
    reports = [(path if path.is_absolute() else root / path).resolve()
               for path in args.report]
    audit_path = (args.audit_summary if args.audit_summary.is_absolute() else
                  root / args.audit_summary).resolve()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in reports]
    object_ids = {row["selected_geometric_anchor"]["object_id"] for row in payloads}
    if len(object_ids) != 1 or output.exists():
        raise ValueError("reports must bind one object and output must be new")
    inputs = load_v2_inputs(root, config_path=config, object_id=object_ids.pop())
    audit_object = next(row for row in audit["objects"].values()
                        if row["object_id"] == inputs.object_contract.object_id)
    if (audit["source"]["commit"] != "ba724b2f5c0c6f30d99c02130200a113bac3e260"
            or audit_object["mesh_sha256"]
            != inputs.object_contract.model.provenance.source_sha256):
        raise ValueError("surface audit commit or object mesh identity changed")
    candidates, seen = [], set()
    for report_index, report in enumerate(payloads):
        rows = list(report.get("survivor_candidates") or ())
        if not rows:
            projected = _projected_legacy_anchor(report, inputs)
            rows = [] if projected is None else [projected]
        for row in rows:
            key = (tuple(row["object_from_hand_row_major"]),
                   tuple(row["pregrasp_closure_phases"]))
            if key in seen:
                continue
            seen.add(key)
            seed = _candidate(row, f"r{report_index:02d}_{len(candidates):02d}")
            candidates.append((seed, _old_row(
                report, tuple(seed.pregrasp_closure_phases))))
    roles, areas = inputs.face_roles.face_role, inputs.object_contract.model.mesh.face_areas_m2
    result = {"schema_version": "carts_surface_v2_fixed_candidate_reevaluation_v1",
        "claim_scope": "OFFLINE_FIXED_POSE_REEVALUATION_NOT_DYNAMIC_SUCCESS",
        "hardware_authorized": False, "formal_dynamic_pass": False,
        "research_dynamic_pass": False, "config": str(config),
        "config_sha256": file_sha256(config),
        "surface_audit_binding": {"path": str(audit_path),
            "sha256": file_sha256(audit_path),
            "source_commit": audit["source"]["commit"],
            "object_mesh_sha256": audit_object["mesh_sha256"]},
        "source_reports": [{"path": str(path), "sha256": file_sha256(path)}
                           for path in reports],
        "surface_role_counts": {FACE_ROLE_NAMES[code]: int(np.sum(roles == code))
                                for code in range(3)},
        "surface_role_area_fractions": {FACE_ROLE_NAMES[code]: float(
            np.sum(areas[roles == code]) / np.sum(areas)) for code in range(3)},
        "candidate_count": len(candidates),
        "candidates": [_evaluate(inputs, seed, old) for seed, old in candidates]}
    result["elapsed_s"] = time.perf_counter() - started
    result["research_candidate_count"] = sum(
        bool(row["research_candidate"]) for row in result["candidates"])
    eligible = [row for row in result["candidates"] if row["research_candidate"]]
    eligible.sort(key=lambda row: (
        np.inf if row["task_quality"]["worst_task_margin"] is None else
        -float(row["task_quality"]["worst_task_margin"]),
        np.inf if row["task_quality"]["lower_tail_mean_margin"] is None else
        -float(row["task_quality"]["lower_tail_mean_margin"]),
        np.inf if row["task_quality"]["required_peak_normal_force_n"] is None else
        float(row["task_quality"]["required_peak_normal_force_n"]),
        -min(float(region["hard_forbidden_minimum_gap_m"])
             for region in row["contact_regions"]), row["candidate_id"]))
    result["top3_task_lexicographic_candidate_ids"] = [
        row["candidate_id"] for row in eligible[:3]]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True,
                                 allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "candidate_count": len(candidates),
                      "research_candidate_count": result["research_candidate_count"],
                      "elapsed_s": result["elapsed_s"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
