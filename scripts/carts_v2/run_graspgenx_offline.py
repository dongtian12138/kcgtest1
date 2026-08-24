#!/usr/bin/env python3
"""Validate file-bound GraspGenX poses and run the existing V2 physics chain."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from kcg_connector.grasp.carts_v2.candidate_generator import generate_raw_candidates
from kcg_connector.grasp.carts_v2.graspgenx_adapter import (
    load_graspgenx_candidates,
    summarize_six_d_coverage,
)
from kcg_connector.grasp.carts_v2.models import load_v2_inputs
from kcg_connector.grasp.carts_v2.pipeline import run_offline_pipeline
from kcg_connector.grasp.carts_v2.reporting import write_offline_report
from kcg_connector.grasp.robust.object_model import file_sha256


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


def _filter_open_hand_from_table(inputs, candidates):
    """Official coarse scene-PC rejection; target object is deliberately absent."""
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
    kept_ids, per_descriptor, rejected = set(), [], []
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
        kept_ids.update(row.seed.candidate_id for row, keep in zip(rows, mask) if keep)
        rejected.extend({"candidate_id": row.seed.candidate_id, "descriptor_id": descriptor_id, "raw_index": row.seed.source_sample_index} for row, keep in zip(rows, mask) if not keep)
        per_descriptor.append({"descriptor_id": descriptor_id, "input_count": len(rows), "retained_count": int(mask.sum()), "rejected_count": int(len(rows) - mask.sum()), "canonical_surface_sha256": hashlib.sha256(np.asarray(surface, dtype="<f4").tobytes()).hexdigest()})
    filtered = tuple(row for row in candidates if row.seed.candidate_id in kept_ids)
    evidence = candidates[0].evidence
    audit = {
        "schema_version": "graspgenx_carts_scene_pc_filter_v1", "object_id": inputs.object_contract.object_id,
        "method": "OFFICIAL_COARSE_OPEN_HAND_SCENE_PC_REJECT_ONLY",
        "official_function": "graspgenx.utils.collision_filter.filter_colliding_grasps",
        "official_source_sha256": file_sha256(Path(official.__file__)), "generator_commit": evidence["generator_commit"],
        "input_proposal_sha256": evidence["proposal_file_sha256"], "descriptor_manifest_sha256": evidence["descriptor_manifest_sha256"],
        "collision_roster_sha256": file_sha256(inputs.repository_root / inputs.config.section("inputs")["collision_roster"]),
        "collision_threshold_m": threshold, "gripper_surface_sample_count": sample_count,
        "registered_hand_link_count": len(inputs.hand_collision_triangles_by_link),
        "scene_components": ["FINITE_TABLE_TOP"], "target_object_points_included": False,
        "table_xy_bounds_m": bounds.tolist(), "table_top_z_m": inputs.table_top_z_m,
        "table_point_count": len(table), "table_point_cloud_sha256": hashlib.sha256(np.asarray(table, dtype="<f4").tobytes()).hexdigest(),
        "input_count": len(candidates), "retained_count": len(filtered), "rejected_count": len(candidates) - len(filtered),
        "per_descriptor": per_descriptor, "rejected": rejected,
        "claim_scope": "EARLY_PROXIMITY_REJECT_NOT_MESH_OR_PATH_SAFETY_PROOF",
    }
    return filtered, audit


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


def main() -> int:
    args = _arguments()
    root = args.repository_root.resolve()
    integration = _read(args.integration_manifest.resolve())
    object_inputs = integration.get("object_inputs", {})
    _require_bound_file(
        root,
        args.object_manifest,
        object_inputs.get("manifest"),
        object_inputs.get("sha256"),
        "object manifest",
    )
    _require_bound_file(
        root,
        args.proposal,
        object_inputs.get("proposal_path_by_object", {}).get(args.object_id),
        object_inputs.get("proposal_sha256_by_object", {}).get(args.object_id),
        "proposal",
    )
    object_manifest = _read(args.object_manifest.resolve())
    checkpoint_sha = integration.get("checkpoint", {}).get("sha256")
    descriptor_sha = integration.get("descriptors", {}).get("sha256")
    if not checkpoint_sha or not descriptor_sha:
        raise ValueError("integration manifest has no frozen model/descriptor identity")
    descriptor_path = _resolve(root, integration["descriptors"]["manifest"])
    mesh_path = _object_mesh_path(object_manifest, args.object_id)
    inputs = load_v2_inputs(root, config_path=args.config, object_id=args.object_id)
    settings = inputs.config.section("candidate_generation")
    dedup = settings["deduplication"]
    adapted = load_graspgenx_candidates(
        inputs, args.proposal, descriptor_path, mesh_path,
        expected_generator_commit=integration["generator"]["commit"],
        expected_checkpoint_sha256=checkpoint_sha,
        expected_random_seed=int(settings["random_seed"]),
        expected_descriptor_manifest_sha256=descriptor_sha,
        translation_tolerance_m=float(dedup["palm_position_m"]),
        rotation_tolerance_rad=float(dedup["palm_orientation_rad"]),
        maximum_candidates=int(settings["graspgenx"]["merged_max_per_object"]),
    )
    coverage = summarize_six_d_coverage(inputs, adapted)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "candidate_coverage.json").write_text(
        json.dumps(coverage, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    physics_candidates, scene_audit = _filter_open_hand_from_table(inputs, adapted)
    (args.output_dir / "scene_pc_filter_audit.json").write_text(
        json.dumps(scene_audit, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if not args.skip_coverage_render:
        baseline_inputs = load_v2_inputs(
            root, config_path=args.baseline_config, object_id=args.object_id
        )
        _render_coverage(
            inputs, adapted, generate_raw_candidates(baseline_inputs),
            args.output_dir / "old_vs_graspgenx_coverage.png",
        )
    evidence = [dict(row.evidence) for row in adapted]
    (args.output_dir / "generator_binding.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if coverage["coverage_pass"] is not True:
        print("six-dimensional coverage diagnostic failed closed")
        return 2
    if not physics_candidates:
        print("official open-hand scene-PC filter rejected every proposal")
        return 3
    result = run_offline_pipeline(
        root, config_path=args.config, object_id=args.object_id,
        candidate_seeds=tuple(row.seed for row in physics_candidates),
    )
    write_offline_report(result, args.output_dir)
    print(
        f"{args.object_id}: {len(physics_candidates)} scene-filtered 6D candidates, "
        f"{len(result.research_task_candidates)} nominal-task eligible, "
        "0 executable before arm IK/path"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
