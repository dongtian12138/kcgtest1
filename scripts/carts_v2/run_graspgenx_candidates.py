#!/usr/bin/env python3
"""Run official GraspGenX once and export file-bound 6-DOF proposals."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import time

import graspgenx
import numpy as np
import torch
import trimesh

from graspgenx import get_checkpoints_version_dir
from graspgenx.dataset.xgrasp_dataset_utils import (
    filter_xgripper_grasps_by_point_cloud_visibility,
)
from graspgenx.grasp_server import GraspGenXSampler
from graspgenx.samplers import run_planner_on_object
from graspgenx.serving.types import SweepVolumeParams
from graspgenx.utils.checkpoint_io import load_model_cfg


_SCHEMA = "graspgenx_carts_proposals_v1"
_DESCRIPTOR_SCHEMA = "kcg_graspgenx_descriptors_v1"
_OBJECT_SCHEMA = "graspgenx_carts_objects_v1"
_KEEP_METHOD = "FIXED_SIX_APPROACH_STRATA_THEN_SCORE_FILL"
_VISIBILITY_METHOD = "OFFICIAL_OPEN_OR_HALF_SWEEP_POINT_CLOUD_VISIBILITY"
_SAMPLE_METHOD = "TRIMESH_ALLOWED_FACE_SAMPLE_EXPLICIT_SEED"
_CONDITIONING_MODE = "REGISTERED_ALLOWED_SURFACE_ROI_POINT_CLOUD"
_DOWNSTREAM_COLLISION_SCOPE = "FULL_REGISTERED_OBJECT_MESH"
_GRASPMOE_PARAMETERS = {
    "grasp_threshold": 0.7,
    "topk_num_grasps": -1,
    "moe_num_yaws": 36,
    "moe_z_offsets_cm": [-2, 0],
    "moe_outlier_threshold": 0.014,
    "moe_outlier_k": 20,
    "moe_obb_mode": "advanced",
    "moe_skip_obb_rule": "auto",
    "moe_obb_density": "dense-topandside",
    "moe_obb_position_spacing_cm": 1.0,
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--descriptor-manifest", required=True, type=Path)
    parser.add_argument("--object-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--checkpoints", type=Path)
    parser.add_argument("--generator-commit", required=True)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--num-grasps", type=int, default=256)
    parser.add_argument("--keep-per-descriptor", type=int, default=128)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    """Bind every checkpoint byte and relative filename deterministically."""

    files = tuple(sorted(path for path in root.rglob("*") if path.is_file()))
    if not files:
        raise FileNotFoundError(f"checkpoint tree contains no files: {root}")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _official_source_commit() -> tuple[str, str]:
    """Return the imported official repository path and exact Git identity."""

    repository = Path(graspgenx.__file__).resolve().parents[1]
    completed = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip().lower()
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise ValueError("imported GraspGenX source has no valid Git identity")
    return str(repository), commit


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} root must be an object")
    return value


def _allowed_face_domain_sha256(face_count: int, indices: np.ndarray) -> str:
    values = np.asarray(indices, dtype="<u8")
    if (
        values.ndim != 1
        or len(values) == 0
        or np.any(values >= int(face_count))
        or not np.array_equal(values, np.unique(values))
    ):
        raise ValueError("allowed face domain is invalid")
    header = np.asarray((int(face_count), len(values)), dtype="<u8").tobytes()
    return hashlib.sha256(header + values.tobytes()).hexdigest()


def _load_object(
    row: dict, seed: int,
) -> tuple[np.ndarray, np.ndarray, str, np.ndarray, str, str]:
    path = Path(row["standardized_mesh_npz"]).resolve()
    if _sha256(path) != row["standardized_mesh_sha256"]:
        raise ValueError(f"standardized mesh hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        vertices = np.asarray(archive["vertices_m"], dtype=np.float64)
        faces = np.asarray(archive["faces"], dtype=np.int64)
        allowed = np.asarray(archive["allowed_face_indices"], dtype=np.int64)
    domain_sha = _allowed_face_domain_sha256(len(faces), allowed)
    if (
        row.get("face_count") != len(faces)
        or row.get("allowed_face_count") != len(allowed)
        or row.get("allowed_face_domain_sha256") != domain_sha
    ):
        raise ValueError("object allowed-face identity changed")
    inference_from_object = np.asarray(
        row["inference_from_object_row_major"], dtype=np.float64
    ).reshape(4, 4)
    rotation = inference_from_object[:3, :3]
    if (
        row.get("inference_frame") != "FROZEN_SCENE_WORLD"
        or not np.allclose(
            inference_from_object[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-9
        )
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-9)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-9)
    ):
        raise ValueError("object inference frame is not one proper frozen transform")
    vertices_inference = (
        vertices @ rotation.T + inference_from_object[:3, 3]
    )
    mesh = trimesh.Trimesh(
        vertices=vertices_inference, faces=faces, process=False
    )
    if not np.all(np.isfinite(mesh.vertices)) or len(mesh.faces) == 0:
        raise ValueError(f"invalid standardized mesh: {path}")
    allowed_mesh = trimesh.Trimesh(
        vertices=vertices_inference, faces=faces[allowed], process=False
    )
    points, sampled_faces = trimesh.sample.sample_surface(
        allowed_mesh, int(row["sample_point_count"]), seed=int(seed)
    )
    if len(points) != int(row["sample_point_count"]) or np.any(
        (sampled_faces < 0) | (sampled_faces >= len(allowed))
    ):
        raise ValueError("allowed-surface sampling left the registered domain")
    center = np.mean(points, axis=0)
    centered = np.asarray(points - center, dtype=np.float32)
    point_cloud_sha256 = hashlib.sha256(
        centered.tobytes(order="C")
        + np.asarray(center, dtype=np.float64).tobytes(order="C")
    ).hexdigest()
    return (
        centered, center, str(path), np.linalg.inv(inference_from_object),
        point_cloud_sha256, domain_sha,
    )


def _sweep_params(row: dict) -> SweepVolumeParams:
    sweep = row["graspgenx_config"]["sweep_volume"]
    return SweepVolumeParams(
        extents_open=sweep["extents"],
        offset_open=sweep["offset"],
        extents_mid=sweep["extents2"],
        offset_mid=sweep["offset2"],
        gripper_type=2,
        fingertip_depth=float(row["graspgenx_config"]["fingertip"][2]),
    )


def _valid_pose_mask(poses: np.ndarray, scores: np.ndarray) -> np.ndarray:
    finite = np.all(np.isfinite(poses), axis=(1, 2)) & np.isfinite(scores)
    homogeneous = np.max(
        np.abs(poses[:, 3, :] - np.asarray((0.0, 0.0, 0.0, 1.0))), axis=1
    ) <= 1.0e-4
    rotation = poses[:, :3, :3]
    orthogonal = np.max(
        np.abs(np.transpose(rotation, (0, 2, 1)) @ rotation - np.eye(3)),
        axis=(1, 2),
    ) <= 2.0e-3
    proper = np.abs(np.linalg.det(rotation) - 1.0) <= 2.0e-3
    return finite & homogeneous & orthogonal & proper


def _open_or_half_visibility(poses, point_cloud, gripper):
    open_visible = filter_xgripper_grasps_by_point_cloud_visibility(
        poses, point_cloud, gripper
    )
    mid = np.asarray(gripper.sweep_volume_mid, dtype=np.float32)
    half_bounds = np.stack((mid[3:] - 0.5 * mid[:3], mid[3:] + 0.5 * mid[:3]))
    half_visible = filter_xgripper_grasps_by_point_cloud_visibility(
        poses, point_cloud, replace(gripper, grasp_volume=half_bounds)
    )
    masks = []
    for value in (open_visible, half_visible):
        masks.append(
            np.zeros(len(poses), dtype=bool)
            if value is None else np.asarray(value, dtype=bool)
        )
    if any(mask.shape != (len(poses),) for mask in masks):
        raise ValueError("official sweep-volume visibility mask has wrong shape")
    return masks[0], masks[1], masks[0] | masks[1]


def _approach_stratum(tag: str, pose: np.ndarray) -> str:
    if tag != "obb":
        return "diff"
    approach = pose[:3, 2]
    if approach[2] < -0.5:
        return "obb_top"
    azimuth = float(np.mod(np.arctan2(approach[1], approach[0]), 2.0 * np.pi))
    return f"obb_side_{min(int(azimuth / (0.5 * np.pi)), 3)}"


def _direction_stratified_keep(
    poses: np.ndarray, scores: np.ndarray, tags: list[str], valid: np.ndarray,
    keep: int,
) -> tuple[np.ndarray, dict[str, int], dict[str, int]]:
    names = ("diff", "obb_top", "obb_side_0", "obb_side_1", "obb_side_2", "obb_side_3")
    groups = {name: [] for name in names}
    for index in np.flatnonzero(valid):
        groups[_approach_stratum(str(tags[index]), poses[index])].append(int(index))
    selected: list[int] = []
    base, remainder = divmod(int(keep), len(names))
    for position, name in enumerate(names):
        quota = base + int(position < remainder)
        ranked = sorted(groups[name], key=lambda index: (-scores[index], index))
        selected.extend(ranked[:quota])
    selected_set = set(selected)
    remaining = sorted(
        (int(index) for index in np.flatnonzero(valid) if int(index) not in selected_set),
        key=lambda index: (-scores[index], index),
    )
    selected.extend(remaining[: max(0, int(keep) - len(selected))])
    selected = sorted(selected, key=lambda index: (-scores[index], index))
    visible_counts = {name: len(groups[name]) for name in names}
    kept_counts = {
        name: sum(_approach_stratum(str(tags[index]), poses[index]) == name for index in selected)
        for name in names
    }
    return np.asarray(selected, dtype=np.int64), visible_counts, kept_counts


def _infer(
    point_cloud: np.ndarray,
    center: np.ndarray,
    sampler: GraspGenXSampler,
    *,
    object_from_inference: np.ndarray,
    num_grasps: int,
    keep: int,
) -> tuple[list[dict], dict]:
    started = time.perf_counter()
    poses, scores, tags, _obb = run_planner_on_object(
        point_cloud,
        sampler,
        planner="graspmoe",
        num_grasps=num_grasps,
        **_GRASPMOE_PARAMETERS,
    )
    elapsed = time.perf_counter() - started
    poses = np.asarray(poses, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if poses.size == 0:
        return [], {
            "raw_count": 0,
            "invalid_pose_count": 0,
            "kept_count": 0,
            "elapsed_s": elapsed,
        }
    poses = poses.reshape(-1, 4, 4)
    scores = scores.reshape(-1)
    if len(poses) != len(scores) or len(tags) != len(poses):
        raise ValueError("official planner returned inconsistent output lengths")
    rigid = _valid_pose_mask(poses, scores)
    open_visibility, half_visibility, visibility = _open_or_half_visibility(
        poses, point_cloud, sampler.gripper
    )
    valid = rigid & visibility
    order, visible_strata, kept_strata = _direction_stratified_keep(
        poses, scores, list(tags), valid, keep
    )
    rows = []
    for raw_index in order:
        inference_from_generator = np.array(poses[raw_index], copy=True)
        inference_from_generator[:3, 3] += center
        pose = object_from_inference @ inference_from_generator
        rows.append(
            {
                "raw_index": int(raw_index),
                "score": float(scores[raw_index]),
                "branch": str(tags[raw_index]),
                "object_from_graspgenx_row_major": pose.ravel().tolist(),
            }
        )
    audit = {
        "raw_count": int(len(poses)),
        "invalid_pose_count": int(len(poses) - np.count_nonzero(rigid)),
        "rigid_finite_pose_count": int(np.count_nonzero(rigid)),
        "open_sweep_visible_count": int(np.count_nonzero(rigid & open_visibility)),
        "half_sweep_visible_count": int(np.count_nonzero(rigid & half_visibility)),
        "open_or_half_sweep_visible_count": int(np.count_nonzero(rigid & visibility)),
        "open_or_half_sweep_reject_count": int(np.count_nonzero(rigid & ~visibility)),
        "proposal_visibility_method": _VISIBILITY_METHOD,
        "open_sweep_claim_scope": "SEMANTIC_REACHABILITY_FILTER_NOT_COLLISION_PROOF",
        "proposal_keep_method": _KEEP_METHOD,
        "visible_approach_stratum_counts": visible_strata,
        "kept_approach_stratum_counts": kept_strata,
        "kept_count": len(rows),
        "elapsed_s": elapsed,
    }
    return rows, audit


def main() -> int:
    args = _arguments()
    if args.num_grasps != 256 or args.keep_per_descriptor != 128:
        raise ValueError("route1 fixes generation=256 and per-descriptor keep=128")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    descriptors = _read_json(args.descriptor_manifest)
    objects = _read_json(args.object_manifest)
    if (
        descriptors.get("schema_version") != _DESCRIPTOR_SCHEMA
        or descriptors.get("object_independent") is not True
        or not 1 <= len(descriptors.get("descriptors", ())) <= 5
    ):
        raise ValueError("descriptor manifest identity/count changed")
    if objects.get("schema_version") != _OBJECT_SCHEMA or not objects.get("objects"):
        raise ValueError("object manifest identity changed or is empty")
    source_path, source_commit = _official_source_commit()
    if args.generator_commit.lower() != source_commit:
        raise ValueError("requested generator commit differs from imported official source")
    checkpoint_root = (
        args.checkpoints or Path(get_checkpoints_version_dir())
    ).resolve()
    checkpoint_sha256 = _tree_sha256(checkpoint_root)
    cfg = load_model_cfg(
        str(checkpoint_root / "gen"), str(checkpoint_root / "dis"), None, None
    )
    object_data = {
        row["object_id"]: (*_load_object(row, args.seed), row)
        for row in objects["objects"]
    }
    collected = {object_id: [] for object_id in object_data}
    audits = {object_id: [] for object_id in object_data}
    shared_model = None
    for descriptor in descriptors["descriptors"]:
        sampler = GraspGenXSampler.from_sweep_volume(
            cfg, _sweep_params(descriptor), model=shared_model
        )
        shared_model = sampler.model
        for object_id, (
            points, center, _path, object_from_inference, _point_cloud_sha,
            _domain_sha, _row,
        ) in object_data.items():
            rows, audit = _infer(
                points,
                center,
                sampler,
                object_from_inference=object_from_inference,
                num_grasps=args.num_grasps,
                keep=args.keep_per_descriptor,
            )
            for row in rows:
                row["descriptor_id"] = descriptor["descriptor_id"]
            collected[object_id].extend(rows)
            audits[object_id].append(
                {"descriptor_id": descriptor["descriptor_id"], **audit}
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for object_id, (
        _points, _center, path, _object_from_inference, point_cloud_sha,
        domain_sha, object_row,
    ) in object_data.items():
        payload = {
            "schema_version": _SCHEMA,
            "object_id": object_id,
            "generator_commit": source_commit,
            "checkpoint_root": str(checkpoint_root),
            "checkpoint_sha256": checkpoint_sha256,
            "source_mesh_sha256": object_row["source_mesh_sha256"],
            "standardized_mesh": path,
            "standardized_mesh_sha256": object_row["standardized_mesh_sha256"],
            "object_point_cloud_sha256": point_cloud_sha,
            "proposal_conditioning_mode": _CONDITIONING_MODE,
            "downstream_collision_geometry_scope": _DOWNSTREAM_COLLISION_SCOPE,
            "allowed_face_count": int(object_row["allowed_face_count"]),
            "allowed_surface_area_m2": float(object_row["allowed_surface_area_m2"]),
            "allowed_face_domain_sha256": domain_sha,
            "face_role_method": object_row["face_role_method"],
            "inference_frame": object_row["inference_frame"],
            "inference_from_object_row_major": object_row[
                "inference_from_object_row_major"
            ],
            "descriptor_manifest_sha256": _sha256(args.descriptor_manifest),
            "random_seed": args.seed,
            "planner": "graspmoe",
            "inference_parameters": {
                "object_sample_point_count": int(object_row["sample_point_count"]),
                "object_surface_sample_method": _SAMPLE_METHOD,
                "proposal_conditioning_mode": _CONDITIONING_MODE,
                "num_grasps": args.num_grasps,
                "keep_per_descriptor": args.keep_per_descriptor,
                "proposal_keep_method": _KEEP_METHOD,
                "proposal_visibility_method": _VISIBILITY_METHOD,
                **_GRASPMOE_PARAMETERS,
            },
            "model_loaded_once": shared_model is not None,
            "model_load_count": 1,
            "environment": {
                "python": platform.python_version(),
                "pytorch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "graspgenx_source": source_path,
                "graspgenx_module": str(Path(graspgenx.__file__).resolve()),
                "torch_module": str(Path(torch.__file__).resolve()),
            },
            "descriptor_audits": audits[object_id],
            "proposals": collected[object_id],
        }
        destination = args.output_dir / f"{object_id}.json"
        destination.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(f"{object_id}: {len(collected[object_id])} proposals -> {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
