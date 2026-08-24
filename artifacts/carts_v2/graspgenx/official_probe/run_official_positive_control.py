from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[4] / "third_party/GraspGenX"
OUTPUT = Path(__file__).resolve().parent
CHECKPOINT_ROOT = REPOSITORY / "ext/graspgenx_checkpoints/release"
DESCRIPTIONS = REPOSITORY / "ext/gripper_descriptions"
MESH = OUTPUT / "banana.obj"

os.environ["GRASPGENX_CHECKPOINT_DIR"] = str(CHECKPOINT_ROOT.parent)
os.environ["GRASPGENX_GRIPPER_CFG_DIR"] = str(DESCRIPTIONS)
sys.path.insert(0, str(REPOSITORY / "scripts"))
sys.path.insert(0, str(REPOSITORY))

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import torch
import trimesh.transformations as tra

from demo_object_mesh import load_mesh_data
from demo_object_pc import load_model_cfg
from graspgenx.dataset.eval_utils import save_to_isaac_grasp_format
from graspgenx.grasp_server import GraspGenXSampler


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    started = time.perf_counter()
    point_cloud, _, mesh, center_transform = load_mesh_data(str(MESH), 1.0, 3500)
    model_cfg = load_model_cfg(
        str(CHECKPOINT_ROOT / "gen"), str(CHECKPOINT_ROOT / "dis"), None, None
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    sampler = GraspGenXSampler(model_cfg, "robotiq_3f", assets_dir=str(REPOSITORY / "assets"))
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - load_started

    inference_started = time.perf_counter()
    grasps, scores = GraspGenXSampler.run_inference(
        point_cloud,
        sampler,
        grasp_threshold=-1.0,
        num_grasps=20,
        topk_num_grasps=20,
        min_grasps=20,
        max_tries=1,
        remove_outliers=False,
    )
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_started
    poses_centered = grasps.detach().cpu().numpy()
    scores_np = scores.detach().cpu().numpy()
    order = np.argsort(scores_np)[::-1]
    poses_centered = poses_centered[order][:20]
    scores_np = scores_np[order][:20]
    poses = np.asarray([tra.inverse_matrix(center_transform) @ pose for pose in poses_centered])
    if poses.shape != (20, 4, 4) or scores_np.shape != (20,):
        raise RuntimeError(f"Expected exactly 20 poses/scores, got {poses.shape}/{scores_np.shape}")

    np.savez(OUTPUT / "poses_scores.npz", poses=poses, scores=scores_np)
    save_to_isaac_grasp_format(poses, scores_np, str(OUTPUT / "grasps.yaml"))
    rows = [
        {"rank": index + 1, "score": float(score), "pose_row_major": pose.tolist()}
        for index, (pose, score) in enumerate(zip(poses, scores_np))
    ]

    figure = plt.figure(figsize=(8, 7), dpi=150)
    axis = figure.add_subplot(111, projection="3d")
    vertices = np.asarray(mesh.vertices)
    stride = max(1, len(vertices) // 4000)
    shown = vertices[::stride]
    axis.scatter(shown[:, 0], shown[:, 1], shown[:, 2], s=1, c="0.72", alpha=0.35)
    origins = poses_centered[:, :3, 3]
    colors = plt.cm.viridis((scores_np - scores_np.min()) / max(float(np.ptp(scores_np)), 1e-9))
    axis.scatter(origins[:, 0], origins[:, 1], origins[:, 2], s=28, c=colors)
    directions = poses_centered[:, :3, 2] * 0.03
    axis.quiver(origins[:, 0], origins[:, 1], origins[:, 2], directions[:, 0], directions[:, 1], directions[:, 2], length=1.0, normalize=False, color=colors)
    axis.set_title("GraspGenX official robotiq_3f / banana: top 20")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_zlabel("z [m]")
    figure.tight_layout()
    figure.savefig(OUTPUT / "static_top20.png")
    plt.close(figure)

    result = {
        "schema_version": "graspgenx_official_positive_control_v1",
        "status": "OFFICIAL_HEADLESS_POSITIVE_CONTROL_PASS",
        "gripper": "robotiq_3f",
        "gripper_type": "revolute_3f",
        "mesh": str(MESH),
        "mesh_sha256": sha256(MESH),
        "random_seed": 0,
        "requested_candidates": 20,
        "returned_candidates": len(rows),
        "model_load_count": 1,
        "model_load_seconds": model_load_seconds,
        "inference_seconds": inference_seconds,
        "total_seconds": time.perf_counter() - started,
        "score_min": float(scores_np.min()),
        "score_max": float(scores_np.max()),
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
        "checkpoints": {
            "generator_sha256": sha256(CHECKPOINT_ROOT / "gen/epoch_736.pth"),
            "discriminator_sha256": sha256(CHECKPOINT_ROOT / "dis/epoch_1056.pth"),
        },
        "poses_and_scores": rows,
        "evidence_boundary": "Official model headless inference only; no collision, task-load, Isaac dynamic, or hardware validation.",
    }
    (OUTPUT / "positive_control.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: result[key] for key in (
        "status", "returned_candidates", "model_load_count", "model_load_seconds",
        "inference_seconds", "total_seconds", "score_min", "score_max", "gpu"
    )}, indent=2))


if __name__ == "__main__":
    main()
