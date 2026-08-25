#!/usr/bin/env python3
"""Validate file-bound GraspGenX poses and run the existing V2 physics chain."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from kcg_connector.grasp.carts_v2.candidate_generator import generate_raw_candidates
from kcg_connector.grasp.carts_v2.closure_predictor import SequentialClosurePredictor
from kcg_connector.grasp.carts_v2.fast_filter import (
    fast_filter_predictions,
    fast_filter_pregrasp_paths,
)
from kcg_connector.grasp.carts_v2.full_palm_search import run_full_palm_cascade
from kcg_connector.grasp.carts_v2.graspgenx_adapter import (
    AdaptedCandidate,
    load_graspgenx_candidates,
)
from kcg_connector.grasp.carts_v2.models import V2Inputs, load_v2_inputs
from kcg_connector.grasp.carts_v2.height_projected_search import (
    SampledPathEnvelope, sampled_height_path_states,
    search_height_projected_pregrasps,
)
from kcg_connector.grasp.carts_v2.pipeline import run_offline_pipeline
from kcg_connector.grasp.carts_v2.reporting import write_offline_report
from kcg_connector.grasp.robust.object_model import file_sha256
from kcg_connector.grasp.carts_v2.surface_contact import ExactContactSurfaceQuery


def summarize_six_d_coverage(inputs: V2Inputs,
                             candidates: tuple[AdaptedCandidate, ...]) -> dict[str, object]:
    """Diagnose proposal coverage without treating it as grasp evidence."""
    if len(candidates) < 2:
        raise ValueError("at least two candidates are required for coverage")
    poses = np.asarray([row.seed.object_from_hand_matrix() for row in candidates])
    generator_poses = np.asarray([np.asarray(
        row.evidence["object_from_graspgenx_row_major"]).reshape(4, 4) for row in candidates])
    positions = poses[:, :3, 3]
    origin = inputs.object_contract.model.assembly_axis_origin_m
    basis = inputs.object_contract.task_frame_rotation_object
    task_positions = (positions - origin) @ basis
    radial = np.linalg.norm(task_positions[:, :2], axis=1)
    approach = generator_poses[:, :3, 2]
    approach_azimuth = np.arctan2(approach[:, 1], approach[:, 0])
    approach_elevation = np.arcsin(np.clip(approach[:, 2], -1.0, 1.0))
    rotations = Rotation.from_matrix(poses[:, :3, :3])
    rpy = rotations.as_euler("xyz")
    radius_scale = max(float(inputs.object_contract.characteristic_radius_m), 1.0e-9)
    features = np.column_stack((positions / radius_scale, rotations.as_rotvec() / np.pi))
    covariance = np.cov(features, rowvar=False)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    effective_dimension = float(np.sum(eigenvalues) ** 2
                                / max(np.sum(eigenvalues ** 2), 1.0e-18))
    dedup = inputs.config.section("candidate_generation")["deduplication"]
    descriptor_ids = sorted({row.seed.descriptor_id for row in candidates})
    descriptor_counts = {key: sum(row.seed.descriptor_id == key for row in candidates)
                         for key in descriptor_ids}
    palm_azimuth = np.mod(np.arctan2(task_positions[:, 1], task_positions[:, 0]), 2 * np.pi)
    occupied_quadrants = len(np.unique(np.floor(palm_azimuth / (0.5 * np.pi)).astype(int)))
    checks = {
        "at_least_100_candidates": len(candidates) >= 100,
        "multiple_descriptors": len(descriptor_counts) > 1,
        "multiple_palm_sides": occupied_quadrants >= 2,
        "radial_distance_not_constant": float(np.ptp(radial)) > float(dedup["palm_position_m"]),
        "roll_not_constant": float(np.ptp(rpy[:, 0])) > float(dedup["palm_orientation_rad"]),
        "pitch_not_constant": float(np.ptp(rpy[:, 1])) > float(dedup["palm_orientation_rad"]),
    }
    return {
        "schema_version": "graspgenx_carts_6d_coverage_v1",
        "object_id": inputs.object_contract.object_id,
        "candidate_count": len(candidates),
        "position_range_object_m": np.ptp(positions, axis=0).tolist(),
        "position_min_object_m": np.min(positions, axis=0).tolist(),
        "position_max_object_m": np.max(positions, axis=0).tolist(),
        "radial_distance_range_m": [float(np.min(radial)), float(np.max(radial))],
        "approach_azimuth_range_rad": [float(np.min(approach_azimuth)), float(np.max(approach_azimuth))],
        "approach_elevation_range_rad": [float(np.min(approach_elevation)), float(np.max(approach_elevation))],
        "roll_pitch_yaw_range_rad": np.ptp(rpy, axis=0).tolist(),
        "descriptor_counts": descriptor_counts,
        "palm_azimuth_quadrant_count": occupied_quadrants,
        "normalized_pose_covariance": covariance.tolist(),
        "normalized_pose_covariance_eigenvalues": eigenvalues.tolist(),
        "effective_dimension_participation_ratio": effective_dimension,
        "diagnostic_checks": checks,
        "coverage_pass": all(checks.values()),
        "evidence_scope": "GENERATOR_COVERAGE_DIAGNOSTIC_NOT_GRASP_SUCCESS",
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline-config", type=Path, required=True)
    parser.add_argument("--integration-manifest", type=Path, required=True)
    parser.add_argument("--object-manifest", type=Path, required=True)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-coverage-render", action="store_true")
    return parser.parse_args()


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} root must be an object")
    return value


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _require_bound_file(
    root: Path, actual: Path, bound_value: object, expected_sha: object, label: str
) -> None:
    if not isinstance(bound_value, str) or not isinstance(expected_sha, str):
        raise ValueError(f"integration manifest has no frozen {label} identity")
    actual = actual.resolve()
    if actual != _resolve(root, bound_value):
        raise ValueError(f"{label} path differs from integration manifest")
    if file_sha256(actual) != expected_sha:
        raise ValueError(f"{label} SHA-256 differs from integration manifest")


def _object_mesh_path(manifest: dict, object_id: str) -> Path:
    rows = [row for row in manifest.get("objects", ()) if row.get("object_id") == object_id]
    if len(rows) != 1:
        raise ValueError("object manifest must contain the requested object exactly once")
    return Path(rows[0]["standardized_mesh_npz"]).resolve()


def _route_bindings(root: Path, args, manifest: dict) -> tuple[Path, str, str, str]:
    if manifest.get("schema_version") != "carts_full_palm_search_manifest_v1":
        raise ValueError("search manifest schema changed")
    generator = manifest.get("generator", {})
    descriptors = manifest.get("descriptor_manifest", {})
    proposals = manifest.get("proposals", {})
    budgets = manifest.get("cascade_budgets", {})
    expected_budgets = {
        "raw_proposals_per_angle": 128, "score_keep_per_angle": 64,
        "precise_closure_per_angle": 8,
        "pregrasp_phase_values_per_finger": [0, 0.1, 0.2],
        "pregrasp_combination_count": 27,
        "maximum_candidate_level_calls_per_seed": 28,
        "internal_control_states_count_as_candidate_calls": False,
        "cartesian_product_forbidden": True,
    }
    if budgets != expected_budgets:
        raise ValueError("search manifest cascade budget identity changed")
    expected_proposals = {
        "schema_version": "graspgenx_carts_full_palm_proposals_v2",
        "model_load_count": 1, "descriptor_count_per_object": 91,
        "kept_per_descriptor": 64, "kept_count_per_object": 5824,
        "palm_configuration_bound_per_proposal": True,
    }
    if any(proposals.get(key) != value for key, value in expected_proposals.items()):
        raise ValueError("search manifest proposal identity changed")
    if (descriptors.get("schema_version") != "kcg_graspgenx_descriptors_v2"
            or descriptors.get("descriptor_count") != 91):
        raise ValueError("search manifest descriptor identity changed")
    if manifest.get("rescue_refinement_budget", {}).get(
        "maximum_geometry_evaluations_per_seed"
    ) != 300:
        raise ValueError("future rescue budget identity changed")
    _require_bound_file(root, args.object_manifest, *(
        manifest.get("object_manifest", {}).get(key) for key in ("path", "sha256")
    ), "object manifest")
    proposal = proposals.get(args.object_id, {})
    _require_bound_file(root, args.proposal, proposal.get("path"),
                        proposal.get("sha256"), "proposal")
    descriptor_path = _resolve(root, descriptors["path"])
    _require_bound_file(root, descriptor_path, descriptors.get("path"),
                        descriptors.get("sha256"), "descriptor manifest")
    return (descriptor_path, str(generator.get("commit", "")),
            str(generator.get("checkpoint_tree_sha256", "")),
            str(descriptors.get("sha256", "")))


def _palm_grid(descriptor_document: dict, search_manifest: dict) -> tuple[float, ...]:
    rows = descriptor_document.get("descriptors", ())
    values = sorted(float(row["palm_configuration_rad"]) for row in rows)
    palm = search_manifest.get("palm_configuration", {})
    expected = np.linspace(float(palm.get("lower_rad", np.nan)),
                           float(palm.get("upper_rad", np.nan)),
                           int(palm.get("grid_count", -1)))
    if (len(rows) != 91 or len({str(row.get("descriptor_id")) for row in rows}) != 91
            or not np.allclose(values, expected, atol=1.0e-12, rtol=0.0)):
        raise ValueError("descriptor palm grid differs from search manifest")
    return tuple(values)


def _filter_open_hand_from_table(inputs, candidates):
    """Return official coarse scene-PC preference; never a physical rejection."""
    import graspgenx.utils.collision_filter as official
    import trimesh

    cfg = inputs.config.section("candidate_generation")["graspgenx"]["scene_collision_filter"]
    if cfg.get("enabled") is not True or not candidates:
        raise ValueError("official scene-PC filter must be enabled with candidates")
    threshold, sample_count = float(cfg["collision_threshold_m"]), int(cfg["gripper_surface_sample_count"])
    maximum = int(cfg["maximum_table_point_count"])
    bounds = inputs.table_xy_bounds_m
    width, height = bounds[:, 1] - bounds[:, 0]
    nx = max(2, int(np.sqrt(maximum * width / height)))
    ny = max(2, maximum // nx)
    xx, yy = np.meshgrid(np.linspace(*bounds[0], nx), np.linspace(*bounds[1], ny))
    table = np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, inputs.table_top_z_m)))
    preferred_ids, per_descriptor, not_preferred = set(), [], []
    for descriptor_id in sorted({row.seed.descriptor_id for row in candidates}):
        rows = [row for row in candidates if row.seed.descriptor_id == descriptor_id]
        first = rows[0]
        transforms = inputs.hand_model.forward_kinematics(first.seed.pregrasp_joint_positions_rad)
        generator_from_hand = np.asarray(first.evidence["graspgenx_from_handbase_row_major"]).reshape(4, 4)
        triangles = []
        for link_name, local in inputs.hand_collision_triangles_by_link.items():
            link = transforms[link_name]
            hand = local @ link[:3, :3].T + link[:3, 3]
            triangles.append(hand @ generator_from_hand[:3, :3].T + generator_from_hand[:3, 3])
        triangles = np.concatenate(triangles)
        mesh = trimesh.Trimesh(vertices=triangles.reshape(-1, 3), faces=np.arange(triangles.size // 3).reshape(-1, 3), process=False)
        surface, _ = trimesh.sample.sample_surface(mesh, sample_count, seed=int(inputs.config.section("candidate_generation")["random_seed"]))
        poses = np.asarray([
            inputs.frozen_world_from_object @ np.asarray(row.evidence["object_from_graspgenx_row_major"]).reshape(4, 4)
            for row in rows
        ])
        mask = official.filter_colliding_grasps(table, poses, collision_threshold=threshold, num_collision_samples=sample_count, gripper_surface_points=surface)
        preferred_ids.update(row.seed.candidate_id for row, keep in zip(rows, mask) if keep)
        not_preferred.extend({"candidate_id": row.seed.candidate_id, "descriptor_id": descriptor_id, "raw_index": row.seed.source_sample_index} for row, keep in zip(rows, mask) if not keep)
        per_descriptor.append({"descriptor_id": descriptor_id, "input_count": len(rows), "preferred_count": int(mask.sum()), "not_preferred_count": int(len(rows) - mask.sum()), "canonical_surface_sha256": hashlib.sha256(np.asarray(surface, dtype="<f4").tobytes()).hexdigest()})
    preferred = tuple(row for row in candidates if row.seed.candidate_id in preferred_ids)
    evidence = candidates[0].evidence
    audit = {
        "schema_version": "graspgenx_carts_scene_pc_preference_v2", "object_id": inputs.object_contract.object_id,
        "method": "OFFICIAL_COARSE_OPEN_HAND_SCENE_PC_BUDGET_PRIORITY_DIAGNOSTIC",
        "official_function": "graspgenx.utils.collision_filter.filter_colliding_grasps",
        "official_source_sha256": file_sha256(Path(official.__file__)), "generator_commit": evidence["generator_commit"],
        "input_proposal_sha256": evidence["proposal_file_sha256"], "descriptor_manifest_sha256": evidence["descriptor_manifest_sha256"],
        "collision_roster_sha256": file_sha256(inputs.repository_root / inputs.config.section("inputs")["collision_roster"]),
        "collision_threshold_m": threshold, "gripper_surface_sample_count": sample_count,
        "registered_hand_link_count": len(inputs.hand_collision_triangles_by_link),
        "scene_components": ["FINITE_TABLE_TOP"], "target_object_points_included": False,
        "table_xy_bounds_m": bounds.tolist(), "table_top_z_m": inputs.table_top_z_m,
        "table_point_count": len(table), "table_point_cloud_sha256": hashlib.sha256(np.asarray(table, dtype="<f4").tobytes()).hexdigest(),
        "input_count": len(candidates), "preferred_count": len(preferred),
        "not_preferred_count": len(candidates) - len(preferred),
        "per_descriptor": per_descriptor, "not_preferred": not_preferred,
        "claim_scope": "BUDGET_PRIORITY_ONLY_NOT_REJECTION_OR_PATH_SAFETY_PROOF",
    }
    return preferred, audit


def _hand_reach_radius(inputs, seed) -> float:
    transforms = inputs.hand_model.forward_kinematics(
        seed.pregrasp_joint_positions_rad
    )
    maximum = 0.0
    for link_name, triangles in inputs.hand_collision_triangles_by_link.items():
        transform = transforms[link_name]
        points = triangles.reshape(-1, 3) @ transform[:3, :3].T + transform[:3, 3]
        maximum = max(maximum, float(np.max(np.linalg.norm(points, axis=1))))
    if not np.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("registered hand reach radius is invalid")
    return maximum


def _candidate_sequence_sha256(adapted) -> str:
    rows = []
    for row in adapted:
        seed = row.seed
        rows.append({
            "object_id": seed.object_id, "candidate_id": seed.candidate_id,
            "object_from_hand": seed.object_from_hand,
            "pregrasp_joint_positions_rad": seed.pregrasp_joint_positions_rad,
            "palm_configuration_rad": seed.palm_configuration_rad,
            "approach_direction_object": seed.approach_direction_object,
            "maximum_closure_phase": seed.maximum_closure_phase,
        })
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _pregrasp_contact_key(inputs, query, seed) -> tuple[float, ...]:
    transforms = inputs.hand_model.forward_kinematics(
        seed.pregrasp_joint_positions_rad,
        base_transform=seed.object_from_hand_matrix())
    distances = []
    for name, surface in sorted(inputs.task_grip_surfaces.items()):
        nearest, _point, _normal = query.query_pad(name, transforms[surface.link_name])
        distances.append(float(nearest.distance_m[0]))
    if len(distances) != 3 or any(not np.isfinite(value) for value in distances):
        raise ValueError("pregrasp task-surface distances are invalid")
    return max(distances), sum(distances), *seed.pregrasp_closure_phases


def _contact_height_bounds(inputs, seed, reach_cache) -> tuple[float, float]:
    if seed.descriptor_id not in reach_cache:
        reach_cache[seed.descriptor_id] = _hand_reach_radius(inputs, seed)
    vertices = inputs.object_contract.model.mesh.vertices_m
    world = (vertices @ inputs.frozen_world_from_object[:3, :3].T
             + inputs.frozen_world_from_object[:3, 3])
    reach = float(reach_cache[seed.descriptor_id])
    return float(np.min(world[:, 2]) - reach), float(np.max(world[:, 2]) + reach)


def _render_coverage(inputs, adapted, baseline, destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    mesh = inputs.object_contract.model.mesh
    background = mesh.face_centroids_m[:: max(1, len(mesh.faces) // 5000)]
    new_poses = np.asarray([row.seed.object_from_hand_matrix() for row in adapted])
    generator_poses = np.asarray([
        np.asarray(row.evidence["object_from_graspgenx_row_major"]).reshape(4, 4)
        for row in adapted
    ])
    old_poses = np.asarray([row.object_from_hand_matrix() for row in baseline])
    figure = plt.figure(figsize=(13, 6), dpi=150)
    for index, (title, poses, directions) in enumerate((
        ("Old axisymmetric baseline", old_poses, old_poses[:, :3, 2]),
        ("GraspGenX-CARTS 6D proposals", new_poses, generator_poses[:, :3, 2]),
    ), start=1):
        axis = figure.add_subplot(1, 2, index, projection="3d")
        axis.scatter(*background.T, s=0.25, color="#777777", alpha=0.15)
        chosen = np.linspace(0, len(poses) - 1, min(100, len(poses)), dtype=int)
        palm = poses[chosen, :3, 3]
        arrow = directions[chosen]
        axis.scatter(*palm.T, s=8, color="#1976d2", alpha=0.65)
        axis.quiver(*palm.T, *arrow.T, length=0.012, normalize=True, color="#d32f2f")
        axis.set_title(f"{title}\nshown={len(chosen)}")
        axis.set_xlabel("object x / m")
        axis.set_ylabel("object y / m")
        axis.set_zlabel("object z / m")
        axis.set_box_aspect((1, 1, 1))
    figure.tight_layout()
    figure.savefig(destination)
    plt.close(figure)


def _load_bound_candidates(root: Path, args, search_manifest: dict):
    descriptor_path, commit, checkpoint_sha, descriptor_sha = _route_bindings(
        root, args, search_manifest
    )
    object_manifest = _read(args.object_manifest.resolve())
    mesh_path = _object_mesh_path(object_manifest, args.object_id)
    inputs = load_v2_inputs(root, config_path=args.config, object_id=args.object_id)
    settings = inputs.config.section("candidate_generation")
    dedup = settings["deduplication"]
    adapted = load_graspgenx_candidates(
        inputs, args.proposal, descriptor_path, mesh_path,
        expected_generator_commit=commit,
        expected_checkpoint_sha256=checkpoint_sha,
        expected_random_seed=int(settings["random_seed"]),
        expected_descriptor_manifest_sha256=descriptor_sha,
        translation_tolerance_m=float(dedup["palm_position_m"]),
        rotation_tolerance_rad=float(dedup["palm_orientation_rad"]),
        maximum_candidates=int(settings["graspgenx"]["merged_max_per_object"]),
    )
    descriptor_document = _read(descriptor_path)
    return inputs, adapted, descriptor_document


def main() -> int:
    args = _arguments()
    root = args.repository_root.resolve()
    search_manifest = _read(args.integration_manifest.resolve())
    inputs, adapted, descriptor_document = _load_bound_candidates(
        root, args, search_manifest
    )
    coverage = summarize_six_d_coverage(inputs, adapted)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "candidate_coverage.json", coverage)
    _scene_preferred, scene_audit = _filter_open_hand_from_table(inputs, adapted)
    _write_json(args.output_dir / "scene_pc_filter_audit.json", scene_audit)
    if not args.skip_coverage_render:
        baseline_inputs = load_v2_inputs(
            root, config_path=args.baseline_config, object_id=args.object_id
        )
        _render_coverage(
            inputs, adapted, generate_raw_candidates(baseline_inputs),
            args.output_dir / "old_vs_graspgenx_coverage.png",
        )
    evidence = [dict(row.evidence) for row in adapted]
    _write_json(args.output_dir / "generator_binding.json", evidence)
    if coverage["coverage_pass"] is not True:
        print("six-dimensional coverage diagnostic failed closed")
        return 2
    predictor = SequentialClosurePredictor(inputs)
    surface_query = ExactContactSurfaceQuery(inputs)
    height_settings = inputs.config.section("height_projection")
    fast_settings = inputs.config.section("fast_filter")
    pregrasp_settings = inputs.config.section(
        "candidate_generation")["pregrasp_search"]
    reach_cache: dict[str, float] = {}

    def height_evaluator(seed):
        return search_height_projected_pregrasps(
            inputs, seed, predictor,
            sampled_path_envelope=lambda bound: SampledPathEnvelope(
                tuple(sampled_height_path_states(inputs, bound)),
                "REGISTERED_CONTROL_STEPS_PALM_PRESHAPE_APPROACH_"
                "SEQUENTIAL_CLOSURE_PRELOAD_LIFT_START",
            ),
            pregrasp_contact_key=lambda bound: _pregrasp_contact_key(
                inputs, surface_query, bound),
            pregrasp_path_callback=lambda bound: fast_filter_pregrasp_paths(
                inputs, ((bound, bound.pregrasp_closure_phases),))[0],
            fast_filter_callback=lambda prediction: fast_filter_predictions(
                inputs, (prediction,))[0],
            contact_height_bounds_m=_contact_height_bounds(
                inputs, seed, reach_cache),
            coarse_sample_count=int(
                height_settings["contact_search_coarse_sample_count"]),
            boundary_tolerance_m=float(
                height_settings["contact_boundary_tolerance_m"]),
            maximum_bisection_iterations=int(
                height_settings["maximum_bisection_iterations"]),
            table_numerical_tolerance_m=float(
                fast_settings["table_penetration_tolerance_m"]),
            required_table_clearance_m=float(
                height_settings["table_operation_clearance_m"]),
            maximum_exact_variants=int(
                pregrasp_settings["maximum_exact_preclosures_per_seed"]),
        )
    checkpoint_path = args.output_dir / "full_palm_cascade_checkpoint.json"
    resume_audit = _read(checkpoint_path) if checkpoint_path.exists() else None
    checkpoint_binding = {
        name: file_sha256(path) for name, path in {
            "config": args.config.resolve(),
            "runner": Path(__file__).resolve(),
            "cascade": root / "src/kcg_connector/kcg_connector/grasp/carts_v2/full_palm_search.py",
            "height_projection": root / "src/kcg_connector/kcg_connector/grasp/carts_v2/height_projection.py",
            "height_projected_search": root / "src/kcg_connector/kcg_connector/grasp/carts_v2/height_projected_search.py",
            "fast_filter": root / "src/kcg_connector/kcg_connector/grasp/carts_v2/fast_filter.py",
            "closure_predictor": root / "src/kcg_connector/kcg_connector/grasp/carts_v2/closure_predictor.py",
            "surface_contact": root / "src/kcg_connector/kcg_connector/grasp/carts_v2/surface_contact.py",
            "models": root / "src/kcg_connector/kcg_connector/grasp/carts_v2/models.py",
            "search_manifest": args.integration_manifest.resolve(),
        }.items()
    }
    checkpoint_binding.update({
        "object_id": args.object_id,
        "proposal_sha256": file_sha256(args.proposal.resolve()),
        "adapted_candidate_sequence_sha256": _candidate_sequence_sha256(adapted),
    })
    if (resume_audit is not None
            and resume_audit.get("checkpoint_binding") != checkpoint_binding):
        raise ValueError("full-palm checkpoint source/config binding changed")
    def checkpoint(audit):
        compact = dict(audit)
        compact["deferred"] = []
        compact["checkpoint_deferred_recomputed_on_resume"] = True
        compact["checkpoint_binding"] = checkpoint_binding
        _write_json(checkpoint_path, compact)
    selected_seeds, cascade_audit = run_full_palm_cascade(
        inputs, tuple(row.seed for row in adapted),
        _palm_grid(descriptor_document, search_manifest),
        height_evaluator=height_evaluator,
        resume_audit=resume_audit,
        progress_callback=checkpoint,
    )
    if cascade_audit.get("completed_palm_bucket_count") != 91:
        raise RuntimeError("full-palm cascade returned before all 91 buckets")
    cascade_audit["checkpoint_binding"] = checkpoint_binding
    _write_json(args.output_dir / "full_palm_cascade_audit.json", cascade_audit)
    if not selected_seeds:
        print("full-palm cascade found no path/contact candidate")
        return 4
    result = run_offline_pipeline(
        root, config_path=args.config, object_id=args.object_id,
        candidate_seeds=selected_seeds,
    )
    write_offline_report(result, args.output_dir)
    print(
        f"{args.object_id}: {len(selected_seeds)} per-angle path/contact candidates, "
        f"{len(result.research_task_candidates)} nominal-task eligible, "
        "0 executable before arm IK/path"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
