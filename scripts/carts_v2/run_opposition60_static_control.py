#!/usr/bin/env python3
"""Render one FK opposition-anchor control without launching Isaac Sim."""
from __future__ import annotations
import argparse
import cProfile
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
from kcg_connector.grasp.carts_v2.closure_predictor import SequentialClosurePredictor
from kcg_connector.grasp.carts_v2.fast_filter import (
    fast_filter_predictions, fast_filter_pregrasp_paths,
)
from kcg_connector.grasp.carts_v2.height_projected_search import (
    SampledPathEnvelope, contact_height_bounds, sampled_height_path_states,
    search_height_projected_pregrasps,
)
from kcg_connector.grasp.carts_v2.models import load_v2_inputs
from kcg_connector.grasp.carts_v2.opposition_seed_generator import (
    generate_opposition_anchors, task_surface_triangle_geometry,
)
from kcg_connector.grasp.carts_v2.surface_contact import ExactContactSurfaceQuery
from kcg_connector.grasp.carts_v2.selector import select_candidate_rankings
from kcg_connector.grasp.carts_v2.task_quality import (
    common_uncertainty_design, evaluate_task_quality,
)
from kcg_connector.grasp.robust.bounded_hand_base_ik import (
    CandidateJointRouteError, solve_bounded_hand_base_ik,
)
from kcg_connector.grasp.robust.object_model import file_sha256
_DEFAULT_CONFIG = Path("src/kcg_connector/config/carts_nailfree_height_projected.yaml")
_DEFAULT_OBJECT = "te_deutsch_d38999_26fj35pn_step"
_DEFAULT_OUTPUT = Path(
    "artifacts/carts_v2/opposition60_isaac/qp60_static_positive_control"
)
def _exact_variant_budget(value: str) -> int:
    budget = int(value)
    if not 1 <= budget <= 27:
        raise argparse.ArgumentTypeError("maximum exact variants must lie in [1, 27]")
    return budget
def _exact_variant_offset(value: str) -> int:
    offset = int(value)
    if not 0 <= offset < 27:
        raise argparse.ArgumentTypeError("exact variant offset must lie in [0, 26]")
    return offset
def _anchor_index(value: str) -> int:
    index = int(value)
    if not 0 <= index < 12:
        raise argparse.ArgumentTypeError("anchor index must lie in [0, 11]")
    return index
def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument("--object-id", default=_DEFAULT_OBJECT)
    parser.add_argument("--palm-angle-deg", type=float, default=60.0)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--maximum-exact-variants", type=_exact_variant_budget,
                        default=27)
    parser.add_argument("--exact-variant-offset", type=_exact_variant_offset, default=0)
    parser.add_argument("--anchor-index", type=_anchor_index, default=0)
    parser.add_argument("--profile-output", type=Path)
    parser.add_argument("--evaluate-task-ik", action="store_true")
    return parser.parse_args()
def _world(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    return values @ transform[:3, :3].T + transform[:3, 3]
def _sample(values: np.ndarray, maximum: int) -> np.ndarray:
    rows = np.asarray(values, dtype=np.float64)
    if len(rows) <= maximum:
        return rows
    return rows[np.linspace(0, len(rows) - 1, maximum, dtype=np.int64)]
def _equal_axes(axis, rows: list[np.ndarray]) -> None:
    points = np.vstack([row for row in rows if len(row)])
    lower, upper = np.min(points, axis=0), np.max(points, axis=0)
    center = 0.5 * (lower + upper)
    radius = max(float(np.max(upper - lower)) * 0.55, 0.01)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))
def _render_four_views(path: Path, inputs, seed, audit: dict, prediction) -> None:
    pose = seed.object_from_hand_matrix()
    joints = np.asarray(prediction.final_joint_positions_rad, dtype=np.float64)
    transforms = inputs.hand_model.forward_kinematics(joints, base_transform=pose)
    mesh = inputs.object_contract.model.mesh
    allowed_ids = inputs.face_roles.allowed_face_indices
    object_all = _sample(mesh.face_centroids_m, 4500)
    object_allowed = _sample(mesh.face_centroids_m[allowed_ids], 3000)
    hand_rows = []
    for link, triangles in inputs.hand_collision_triangles_by_link.items():
        centers = np.mean(np.asarray(triangles), axis=1)
        hand_rows.append((link, _sample(_world(centers, transforms[link]), 500)))
    task_rows = []
    for name, surface in sorted(inputs.task_grip_surfaces.items()):
        transform = transforms[surface.link_name]
        centers = np.mean(surface.triangles_local_m, axis=1)
        task_rows.append((name, _sample(_world(centers, transform), 700)))
    matches = [row for row in audit["selected"]
               if row["candidate_id"] == seed.candidate_id]
    if len(matches) != 1:
        raise ValueError("render anchor identity is missing or ambiguous")
    selected = matches[0]
    target = np.asarray(selected["target_band_center_object_m"], dtype=np.float64)
    axis_vector = np.asarray(audit["object_grasp_band"]["axis_object"], dtype=np.float64)
    work = np.asarray(task_surface_triangle_geometry(
        inputs, seed.palm_configuration_rad, seed.pregrasp_closure_phases
    )["work_center_handbase_m"])
    mapped_work = _world(work[None, :], pose)[0]
    contact_points = np.asarray(
        [contact.object_position_m for contact in prediction.contacts],
        dtype=np.float64,
    )
    handbase_points = np.asarray([
        row["handbase_object_m"] for row in audit["selected"]
    ], dtype=np.float64)
    views = (
        (0, -90, "front"), (0, 0, "side"),
        (90, -90, "top"), (25, -55, "oblique"),
    )
    figure = plt.figure(figsize=(13, 11), dpi=150)
    for plot_index, (elevation, azimuth, label) in enumerate(views, start=1):
        plot = figure.add_subplot(2, 2, plot_index, projection="3d")
        plot.scatter(*object_all.T, s=0.4, c="#94a3b8", alpha=0.12)
        plot.scatter(*object_allowed.T, s=1.1, c="#f97316", alpha=0.55)
        for link, points in hand_rows:
            color = "#64748b" if link == "handbase_link" else "#2563eb"
            plot.scatter(*points.T, s=0.45, c=color, alpha=0.18)
        for _name, points in task_rows:
            plot.scatter(*points.T, s=1.6, c="#16a34a", alpha=0.75)
        plot.scatter(*handbase_points.T, s=16, c="#7c3aed", alpha=0.7)
        plot.scatter(*target, s=55, c="#dc2626", marker="x")
        plot.scatter(*mapped_work, s=38, c="#06b6d4", marker="o")
        if len(contact_points):
            plot.scatter(*contact_points.T, s=60, c="#facc15", marker="*")
        endpoints = np.vstack((target - 0.04 * axis_vector,
                               target + 0.04 * axis_vector))
        plot.plot(*endpoints.T, color="#dc2626", linewidth=1.4)
        _equal_axes(plot, [object_all, *(row[1] for row in hand_rows)])
        plot.view_init(elev=elevation, azim=azimuth)
        plot.set_title(label)
        plot.set_xlabel("object x / m")
        plot.set_ylabel("object y / m")
        plot.set_zlabel("object z / m")
    figure.suptitle(
        f"Opposition anchor: q_p={math.degrees(seed.palm_configuration_rad):.6f} deg\n"
        "orange object band, green TASK_GRIP_SURFACE, purple retained handbases"
    )
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
def _pregrasp_contact_key(inputs, query, seed, cache) -> tuple[float, ...]:
    transforms = inputs.hand_model.forward_kinematics(
        seed.pregrasp_joint_positions_rad,
        base_transform=seed.object_from_hand_matrix(),
    )
    maximum = int(inputs.config.section(
        "closure_prediction")["nearest_face_candidate_count"])
    distances = []
    for name, surface in sorted(inputs.task_grip_surfaces.items()):
        transform = np.asarray(transforms[surface.link_name], dtype="<f8")
        key = (name, transform.tobytes())
        if key not in cache:
            nearest, _hand_points, _normals = query.query_task_surface_witnesses(
                name, transform, maximum
            )
            cache[key] = float(np.min(nearest.distance_m))
        distances.append(cache[key])
    if len(distances) != 3 or not np.all(np.isfinite(distances)):
        raise ValueError("pregrasp TASK_GRIP_SURFACE distances are invalid")
    return max(distances), sum(distances), *seed.pregrasp_closure_phases
def _handbase_world_z(inputs, seed) -> float:
    return float((inputs.frozen_world_from_object
                  @ seed.object_from_hand_matrix())[2, 3])

def _contact_record(contact, nearest, contact_distance_m) -> dict[str, object]:
    forbidden = nearest.forbidden_distance_m
    guard = bool(forbidden is not None and forbidden <= contact_distance_m
                 and forbidden <= contact.clearance_m
                 + 64.0 * np.finfo(np.float64).eps)
    return {
        "pad_name": contact.pad_name,
        "object_position_m": list(contact.object_position_m),
        "object_face_index": contact.object_face_index,
        "hand_surface_face_index": contact.hand_surface_face_index,
        "legacy_blue_pad": contact.hand_surface_legacy_blue_pad,
        "phase_lower": contact.phase_lower,
        "phase_upper": contact.phase_upper,
        "clearance_m": contact.clearance_m,
        "inward_motion_m_per_phase": contact.inward_motion_m_per_phase,
        "registered_patch_count": nearest.registered_patch_count,
        "finite_patch_witness_count": nearest.finite_patch_witness_count,
        "returned_witness_count": len(nearest.distance_m),
        "forbidden_surface_distance_m": forbidden,
        "forbidden_surface_face_index": nearest.forbidden_face_index,
        "forbidden_first_contact_guard_triggered": guard,
    }

def _minimum_table_record(row: dict, fast_result=None) -> tuple[object, str, str]:
    values = []
    pregrasp = row.get("projected_pregrasp_path", {})
    if isinstance(pregrasp, dict) and pregrasp.get("minimum_table_clearance_m") is not None:
        values.append((float(pregrasp["minimum_table_clearance_m"]),
                       str(pregrasp.get("minimum_clearance_link", "")),
                       str(pregrasp.get("minimum_clearance_stage", ""))))
    if fast_result is not None and fast_result.minimum_table_clearance_m is not None:
        values.append((float(fast_result.minimum_table_clearance_m),
                       fast_result.minimum_clearance_link,
                       fast_result.minimum_clearance_finger_stage))
    history = tuple(row.get("contact_conditioned_iterations", ()))
    if not values and history:
        last = history[-1]
        required = last.get("minimum_table_handbase_z_m")
        projected = last.get("projected_handbase_world_z_m")
        if required is not None and projected is not None:
            values.append((float(projected) - float(required), "",
                           str(last.get("minimum_table_contributing_stage", ""))))
    return (None, "", "") if not values else min(values, key=lambda value: value[0])

def _variant_records(inputs, predictor, query, survivors, audit):
    by_phases = {tuple(seed.pregrasp_closure_phases): seed for seed in survivors}
    records, renders = [], []
    for source in audit["evaluated"]:
        phases = tuple(float(value) for value in source["pregrasp_closure_phases"])
        seed = by_phases.get(phases)
        prediction = None if seed is None else predictor.predict(seed)
        fast_result = None if prediction is None else fast_filter_predictions(
            inputs, (prediction,))[0]
        history = tuple(source.get("contact_conditioned_iterations", ()))
        probes = tuple(source.get("contact_height_probes", ()))
        final_phases = source.get("fresh_final_closure_phases")
        if final_phases is None and history:
            final_phases = history[-1].get("contact_stop_phases")
        projected_z = source.get("projected_handbase_world_z_m")
        if projected_z is None and history:
            projected_z = history[-1].get("projected_handbase_world_z_m")
        closure_status = source.get("fresh_closure_status")
        if not closure_status and probes:
            closure_status = probes[-1].get("closure_status")
        reason = (source.get("revalidation_detail")
                  or source.get("fresh_closure_reason") or source.get("reason") or "")
        clearance, link, stage = _minimum_table_record(source, fast_result)
        complete_state_count = (0 if not history else int(
            history[-1].get("table_path_checked_state_count", 0)))
        contacts = []
        if prediction is not None:
            closure_status, reason = prediction.status, prediction.reason
            final_phases = list(prediction.final_closure_phases)
            projected_z = _handbase_world_z(inputs, seed)
            transforms = inputs.hand_model.forward_kinematics(
                prediction.final_joint_positions_rad,
                base_transform=seed.object_from_hand_matrix())
            maximum = int(inputs.config.section(
                "closure_prediction")["nearest_face_candidate_count"])
            contact_distance = float(inputs.config.section(
                "closure_prediction")["contact_distance_m"])
            for contact in prediction.contacts:
                surface = inputs.task_grip_surfaces[contact.pad_name]
                nearest, _points, _normals = query.query_task_surface_witnesses(
                    contact.pad_name, transforms[surface.link_name], maximum)
                contacts.append(_contact_record(
                    contact, nearest, contact_distance))
            complete_state_count = len(sampled_height_path_states(
                inputs, seed, prediction.final_closure_phases))
        record = {
            "pregrasp_closure_phases": list(phases),
            "height_search_status": source["status"],
            "height_search_reason": source.get("reason", ""),
            "contact_status": closure_status or "NOT_REACHED",
            "contact_reason": reason,
            "three_contact_count": len(contacts),
            "contacts": contacts,
            "final_contact_stop_phases": final_phases,
            "projected_handbase_world_z_m": projected_z,
            "complete_sequential_path_state_count": complete_state_count,
            "minimum_table_clearance_m": clearance,
            "minimum_table_clearance_link": link,
            "minimum_table_clearance_stage": stage,
            "fast_filter_status": None if fast_result is None else fast_result.status,
            "fast_filter_reasons": [] if fast_result is None else list(fast_result.reasons),
            "minimum_nonallowed_surface_clearance_m": source.get(
                "minimum_nonallowed_surface_clearance_m"),
            "nonallowed_surface_gate": source.get("nonallowed_surface_gate",
                "NOT_REACHED_OR_BINARY_ONLY"),
            "forbidden_first_contact_guard_triggered": bool(
                "FORBIDDEN_OBJECT_FIRST_CONTACT" in str(reason)),
            "four_view_image": None,
        }
        records.append(record)
        if seed is not None and prediction is not None:
            renders.append((record, seed, prediction))
    records.sort(key=lambda value: tuple(value["pregrasp_closure_phases"]))
    return records, renders
def _candidate_record(seed) -> dict[str, object]:
    return {
        "candidate_id": seed.candidate_id,
        "object_id": seed.object_id,
        "object_from_hand_row_major": list(seed.object_from_hand),
        "pregrasp_joint_positions_rad": list(seed.pregrasp_joint_positions_rad),
        "pregrasp_closure_phases": list(seed.pregrasp_closure_phases),
        "palm_configuration_rad": seed.palm_configuration_rad,
        "palm_configuration_deg": math.degrees(seed.palm_configuration_rad),
        "approach_direction_object": list(seed.approach_direction_object),
        "anchor_face_index": seed.anchor_face_index,
        "anchor_position_object_m": list(seed.anchor_position_object_m),
    }
def _task_and_bounded_ik(inputs, seed, predictor) -> dict[str, object]:
    prediction = predictor.predict(seed)
    pregrasp = fast_filter_pregrasp_paths(
        inputs, ((seed, seed.pregrasp_closure_phases),))[0]
    fast = fast_filter_predictions(inputs, (prediction,))[0]
    path_values = [float(value) for value in (
        pregrasp.get("minimum_table_clearance_m"),
        fast.minimum_table_clearance_m,
    ) if value is not None]
    if (
        prediction.status != "CLOSURE_SURVIVE"
        or pregrasp.get("accepted") is not True
        or fast.status != "FAST_SURVIVE"
        or not fast.sequential_closure_sweep_pass
        or not path_values
    ):
        raise RuntimeError("fresh geometry gate changed before task evaluation")
    path_minimum = min(path_values)
    design = common_uncertainty_design(inputs)
    quality = evaluate_task_quality(inputs, prediction, design)
    research, formal, diagnostic = select_candidate_rankings(
        (prediction,), (fast,), (quality,), top_k=1,
        path_clearance_by_id={seed.candidate_id: path_minimum})
    ik = {"status": "NOT_RUN_TASK_INELIGIBLE"}
    target = inputs.frozen_world_from_object @ seed.object_from_hand_matrix()
    if research:
        try:
            joints, position_error, orientation_error, seed_index = (
                solve_bounded_hand_base_ik(
                    inputs.config.section("ik")["solver"],
                    model=inputs.robot_model,
                    hand_positions=seed.pregrasp_joint_positions_rad,
                    target_world_from_hand_base=target,
                    label=seed.candidate_id,
                )
            )
            ik = {
                "status": "BOUNDED_IK_PASS_NOT_PATH_COLLISION",
                "arm_joint_positions_rad": list(joints),
                "position_error_m": position_error,
                "orientation_error_rad": orientation_error,
                "selected_seed_index": seed_index,
            }
        except CandidateJointRouteError as error:
            ik = {"status": "IK_REJECT", "code": error.code,
                  "detail": error.detail}
    return {
        "candidate_id": seed.candidate_id,
        "fresh_contact_stop_phases": list(prediction.final_closure_phases),
        "fresh_contact_count": len(prediction.contacts),
        "fresh_object_contact_face_indices": [
            row.object_face_index for row in prediction.contacts],
        "fresh_hand_surface_face_indices": [
            row.hand_surface_face_index for row in prediction.contacts],
        "path_minimum_table_clearance_m": path_minimum,
        "scenario_design_shape": list(design.shape),
        "scenario_design_sha256": hashlib.sha256(
            np.asarray(design, dtype="<f8").tobytes()).hexdigest(),
        "task_quality": asdict(quality),
        "research_task_eligible_not_executable": bool(research),
        "formal_task_eligible_not_executable": bool(formal),
        "diagnostic_only": bool(diagnostic),
        "target_world_from_handbase_row_major": list(target.ravel()),
        "bounded_ik": ik,
        "full_arm_path_collision_checked": False,
        "isaac_started": False,
    }
def main() -> int:
    started = time.perf_counter()
    args = _arguments()
    root = args.repository_root.resolve()
    config = args.config if args.config.is_absolute() else root / args.config
    output = args.output if args.output.is_absolute() else root / args.output
    angle = math.radians(float(args.palm_angle_deg))
    inputs = load_v2_inputs(root, config_path=config, object_id=args.object_id)
    seeds, anchor_audit = generate_opposition_anchors(inputs, (angle,))
    if len(seeds) <= args.anchor_index:
        raise RuntimeError("the static opposition control produced no anchor")
    anchor = seeds[args.anchor_index]
    predictor = SequentialClosurePredictor(inputs)
    query = ExactContactSurfaceQuery(inputs)
    contact_cache = {}
    height_settings = inputs.config.section("height_projection")
    fast_settings = inputs.config.section("fast_filter")
    height_bounds = contact_height_bounds(inputs, anchor)
    profile_path = args.profile_output
    if profile_path is not None:
        profile_path = profile_path if profile_path.is_absolute() else root / profile_path
        profile_path.parent.mkdir(parents=True, exist_ok=True)
    profiler = None if profile_path is None else cProfile.Profile()
    run_search = lambda: search_height_projected_pregrasps(
            inputs, anchor, predictor,
            sampled_path_envelope=lambda bound, final_phases: SampledPathEnvelope(
                tuple(sampled_height_path_states(inputs, bound, final_phases)),
                "REGISTERED_CONTROL_STEPS_PALM_PRESHAPE_APPROACH_"
                "SEQUENTIAL_CLOSURE_PRELOAD_LIFT_START"),
            pregrasp_contact_key=lambda bound: _pregrasp_contact_key(
                inputs, query, bound, contact_cache),
            pregrasp_path_callback=lambda bound: fast_filter_pregrasp_paths(
                inputs, ((bound, bound.pregrasp_closure_phases),))[0],
            fast_filter_callback=lambda prediction: fast_filter_predictions(
                inputs, (prediction,))[0], contact_height_bounds_m=height_bounds,
            table_numerical_tolerance_m=float(
                fast_settings["table_penetration_tolerance_m"]),
            required_table_clearance_m=float(
                height_settings["table_operation_clearance_m"]),
            maximum_exact_variants=args.maximum_exact_variants,
            exact_variant_offset=args.exact_variant_offset)
    try:
        survivors, height_audit = (run_search() if profiler is None
                                   else profiler.runcall(run_search))
    finally:
        if profiler is not None:
            profiler.dump_stats(profile_path)
    variant_records, renders = _variant_records(
        inputs, predictor, query, survivors, height_audit)
    task_ik_records = (
        [_task_and_bounded_ik(inputs, seed, predictor) for seed in survivors]
        if args.evaluate_task_ik else []
    )
    output.mkdir(parents=True, exist_ok=True)
    image_records = []
    for index, (record, survivor, prediction) in enumerate(renders):
        phases = "".join(str(int(round(value * 10.0)))
                         for value in survivor.pregrasp_closure_phases)
        image = output / (
            f"survivor_a{args.anchor_index:02d}_"
            f"o{args.exact_variant_offset:02d}_{index:02d}_"
            f"p{phases}_four_views.png")
        _render_four_views(image, inputs, survivor, anchor_audit, prediction)
        record["four_view_image"] = image.name
        image_records.append({"path": image.name, "sha256": file_sha256(image)})
    result = {
        "schema_version": "carts_opposition60_static_control_v2",
        "claim_scope": (
            f"OFFLINE_ANCHOR_INDEX_{args.anchor_index}_"
            f"PRESHAPE_SHARD_OFFSET_{args.exact_variant_offset}_"
            f"COUNT_{args.maximum_exact_variants}_"
            "EXACT_STATIC_NOT_DYNAMIC_SUCCESS"
        ),
        "requested_palm_angle_deg": float(args.palm_angle_deg),
        "requested_palm_angle_rad": angle,
        "selected_geometric_anchor_index": args.anchor_index,
        "available_geometric_anchor_count": len(seeds),
        "selected_geometric_anchor": _candidate_record(anchor),
        "anchor_generation": anchor_audit,
        "contact_height_bounds_world_z_m": list(height_bounds),
        "height_search": height_audit,
        "variant_results": variant_records,
        "survivor_candidates": [_candidate_record(seed) for seed in survivors],
        "task_and_bounded_ik_requested": bool(args.evaluate_task_ik),
        "task_and_bounded_ik": task_ik_records,
        "survivor_four_views": image_records,
        "registered_patch_count_by_pad": {
            name: int(len(np.unique(surface.patch_indices)))
            for name, surface in sorted(inputs.task_grip_surfaces.items())
        },
        "forbidden_object_first_contact_guard": "ENABLED_FAIL_CLOSED",
        "pregrasp_variant_count": height_audit["pregrasp_variant_count"],
        "exact_variant_evaluated_count": height_audit[
            "exact_variant_evaluated_count"],
        "maximum_exact_variants": args.maximum_exact_variants,
        "exact_variant_offset": args.exact_variant_offset,
        "exact_variant_evaluation_interval": height_audit[
            "exact_variant_evaluation_interval"],
        "profile": (None if profile_path is None else {
            "path": str(profile_path), "sha256": file_sha256(profile_path)}),
        "survivor_count": len(survivors),
        "predictor_executed": True,
        "controller_executed": False,
        "isaac_started": False,
        "connector_moved": False,
        "hardware_authorized": False,
        "dynamic_success": False,
        "pending_gate": (
            "RESEARCH_COLLISION_ASSET_FULL_ARM_PATH_AND_ISAAC"
            if args.evaluate_task_ik else "TASK_QUALITY_BOUNDED_IK_AND_ISAAC"
        ),
        "config": str(config.relative_to(root)),
        "config_sha256": file_sha256(config),
        "script_sha256": file_sha256(Path(__file__).resolve()),
        "elapsed_s": time.perf_counter() - started,
    }
    destination = output / (
        f"opposition60_anchor_a{args.anchor_index:02d}_"
        f"exact_offset_{args.exact_variant_offset:02d}_"
        f"count_{args.maximum_exact_variants:02d}_static_control.json"
    )
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "result": str(destination),
        "result_sha256": file_sha256(destination),
        "exact_variant_evaluated_count": result["exact_variant_evaluated_count"],
        "maximum_exact_variants": result["maximum_exact_variants"],
        "exact_variant_offset": result["exact_variant_offset"],
        "selected_geometric_anchor_index": result[
            "selected_geometric_anchor_index"],
        "survivor_count": result["survivor_count"],
        "elapsed_s": result["elapsed_s"],
        "isaac_started": False,
    }, indent=2))
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
