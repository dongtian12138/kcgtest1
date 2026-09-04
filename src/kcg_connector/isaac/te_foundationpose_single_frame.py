#!/usr/bin/env python3
"""Run one model-based FoundationPose registration on ordinary RGB-D data.

The input mask is externally supplied (SAM-6D in the current pipeline).  This
script never reads simulator object pose, semantic labels, instance labels, or
contact truth.  It performs no robot control.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import trimesh


def _matrix4(values: object, label: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.size != 16:
        raise ValueError(f"{label} must contain 16 values")
    matrix = matrix.reshape(4, 4)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{label} contains non-finite values")
    return matrix


def _load_intrinsics(path: Path) -> np.ndarray:
    document = json.loads(path.read_text(encoding="utf-8"))
    if "cam_K" in document:
        values = document["cam_K"]
    elif "camera_calibration" in document:
        values = document["camera_calibration"]["intrinsics_3x3"]
    else:
        raise ValueError("intrinsics JSON contains neither cam_K nor camera_calibration")
    matrix = np.asarray(values, dtype=np.float64).reshape(3, 3)
    if not np.isfinite(matrix).all() or matrix[2, 2] == 0.0:
        raise ValueError("camera intrinsics are invalid")
    return matrix


def _load_world_from_camera(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    document = json.loads(path.read_text(encoding="utf-8"))
    calibration = document.get("camera_calibration", document)
    values = calibration.get("world_from_camera_cv_row_major")
    if values is None:
        raise ValueError("world-from-camera JSON lacks world_from_camera_cv_row_major")
    return _matrix4(values, "world_from_camera_cv_row_major")


def _save_pipeline_figure(
    output_stem: Path,
    rgb: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    pose_overlay: np.ndarray,
) -> None:
    finite = np.isfinite(depth) & (depth > 0.0)
    depth_display = np.full(depth.shape, np.nan, dtype=np.float32)
    depth_display[finite] = depth[finite]

    mask_overlay = rgb.copy()
    tint = np.zeros_like(mask_overlay)
    tint[..., 1] = 220
    tint[..., 2] = 255
    mask_overlay[mask] = (
        0.48 * mask_overlay[mask].astype(np.float32)
        + 0.52 * tint[mask].astype(np.float32)
    ).astype(np.uint8)

    figure, axes = plt.subplots(1, 4, figsize=(16.0, 4.2), constrained_layout=True)
    axes[0].imshow(rgb)
    axes[0].set_title("(a) Global RGB")
    axes[1].imshow(mask_overlay)
    axes[1].contour(mask.astype(np.uint8), levels=[0.5], colors=["cyan"], linewidths=0.8)
    axes[1].set_title("(b) SAM-6D mask")
    depth_artist = axes[2].imshow(depth_display, cmap="viridis")
    axes[2].contour(mask.astype(np.uint8), levels=[0.5], colors=["white"], linewidths=0.8)
    axes[2].set_title("(c) Metric depth")
    colorbar = figure.colorbar(depth_artist, ax=axes[2], fraction=0.046, pad=0.03)
    colorbar.set_label("Depth [m]")
    axes[3].imshow(pose_overlay)
    axes[3].set_title("(d) FoundationPose 6D")
    for axis in axes:
        axis.set_axis_off()
    figure.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foundationpose-root", type=Path, required=True)
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--depth-npy", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--intrinsics-json", type=Path, required=True)
    parser.add_argument("--world-from-camera-json", type=Path)
    parser.add_argument("--prior-world-from-object-json", type=Path)
    parser.add_argument(
        "--prior-world-pose-key",
        default="planned_world_from_object_row_major",
    )
    parser.add_argument(
        "--initialize-from-prior-only",
        action="store_true",
        help=(
            "initialize FoundationPose pose_last from the transferred world pose "
            "without running a local track_one refinement"
        ),
    )
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--mesh-scale-to-m", type=float, default=1.0)
    parser.add_argument("--maximum-faces", type=int, default=50_000)
    parser.add_argument("--refine-iterations", type=int, default=5)
    parser.add_argument("--foundationpose-debug", type=int, choices=(0, 1), default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.initialize_from_prior_only and args.prior_world_from_object_json is None:
        parser.error("--initialize-from-prior-only requires --prior-world-from-object-json")

    foundationpose_root = args.foundationpose_root.resolve()
    if not (foundationpose_root / "estimater.py").is_file():
        raise FileNotFoundError(f"FoundationPose root is invalid: {foundationpose_root}")
    sys.path.insert(0, str(foundationpose_root))

    # Import the official implementation only after its repository is on sys.path.
    import nvdiffrast.torch as dr  # type: ignore
    from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor  # type: ignore
    from Utils import (  # type: ignore
        draw_posed_3d_box,
        draw_xyz_axis,
        set_logging_format,
        set_seed,
    )

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = output_dir / "foundationpose_debug"
    debug_dir.mkdir()

    rgb_bgr = cv2.imread(str(args.rgb), cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise FileNotFoundError(args.rgb)
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    depth = np.asarray(np.load(args.depth_npy), dtype=np.float32)
    mask_image = cv2.imread(str(args.mask), cv2.IMREAD_GRAYSCALE)
    if mask_image is None:
        raise FileNotFoundError(args.mask)
    mask = mask_image > 0
    if depth.shape != rgb.shape[:2] or mask.shape != rgb.shape[:2]:
        raise ValueError("RGB, depth, and mask shapes do not match")
    depth[~np.isfinite(depth) | (depth <= 0.0)] = 0.0
    valid_mask_depth = mask & (depth >= 0.001)
    if valid_mask_depth.sum() < 4:
        raise ValueError("mask contains fewer than four valid depth pixels")

    camera_matrix = _load_intrinsics(args.intrinsics_json)
    world_from_camera = _load_world_from_camera(args.world_from_camera_json)

    mesh = trimesh.load(args.mesh, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("mesh must load as one Trimesh")
    if not np.isfinite(args.mesh_scale_to_m) or args.mesh_scale_to_m <= 0.0:
        raise ValueError("mesh scale must be finite and positive")
    mesh = mesh.copy()
    mesh.apply_scale(float(args.mesh_scale_to_m))
    original_face_count = int(len(mesh.faces))
    if args.maximum_faces <= 0:
        raise ValueError("maximum faces must be positive")
    if original_face_count > args.maximum_faces:
        mesh = mesh.simplify_quadric_decimation(face_count=args.maximum_faces)
    used_face_count = int(len(mesh.faces))
    used_mesh_path = output_dir / "registration_mesh_used_m.ply"
    mesh.export(used_mesh_path)

    set_logging_format()
    set_seed(0)
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    gl_context = dr.RasterizeCudaContext()
    estimator = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=str(debug_dir),
        debug=args.foundationpose_debug,
        glctx=gl_context,
    )
    prior_world_from_object = None
    prior_camera_from_object = None
    if args.prior_world_from_object_json is None:
        camera_from_object = estimator.register(
            K=camera_matrix,
            rgb=rgb,
            depth=depth,
            ob_mask=mask,
            iteration=args.refine_iterations,
        )
        initialization_method = "BLIND_REGISTER_252_ROTATION_HYPOTHESES"
    else:
        if world_from_camera is None:
            raise ValueError("pose-prior tracking requires world-from-camera")
        prior_document = json.loads(
            args.prior_world_from_object_json.read_text(encoding="utf-8")
        )
        if args.prior_world_pose_key not in prior_document:
            raise ValueError(
                f"prior JSON lacks {args.prior_world_pose_key!r}"
            )
        prior_world_from_object = _matrix4(
            prior_document[args.prior_world_pose_key],
            args.prior_world_pose_key,
        )
        prior_camera_from_object = (
            np.linalg.inv(world_from_camera) @ prior_world_from_object
        )
        import torch

        object_to_center = (
            estimator.get_tf_to_centered_mesh().detach().cpu().numpy()
        )
        estimator.pose_last = torch.as_tensor(
            prior_camera_from_object @ np.linalg.inv(object_to_center),
            device="cuda",
            dtype=torch.float32,
        )
        if args.initialize_from_prior_only:
            camera_from_object = prior_camera_from_object
            initialization_method = "GLOBAL_POSE_TRANSFER_ONLY_NO_LOCAL_REFINEMENT"
        else:
            camera_from_object = estimator.track_one(
                rgb=rgb,
                depth=depth,
                K=camera_matrix,
                iteration=args.refine_iterations,
            )
            initialization_method = "GLOBAL_POSE_TRANSFER_THEN_FOUNDATIONPOSE_TRACK_ONE"
    camera_from_object = _matrix4(camera_from_object, "camera_from_object")
    world_from_object = (
        None if world_from_camera is None else world_from_camera @ camera_from_object
    )

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bounds = np.stack((-extents / 2.0, extents / 2.0), axis=0).reshape(2, 3)
    center_pose = camera_from_object @ np.linalg.inv(to_origin)
    overlay = draw_posed_3d_box(
        camera_matrix,
        img=rgb.copy(),
        ob_in_cam=center_pose,
        bbox=bounds,
        line_color=(0, 255, 0),
        linewidth=3,
    )
    overlay = draw_xyz_axis(
        overlay,
        ob_in_cam=center_pose,
        scale=max(float(np.max(extents)) * 0.7, 0.02),
        K=camera_matrix,
        thickness=3,
        transparency=0,
        is_input_rgb=True,
    )

    imageio.imwrite(output_dir / "global_rgb.png", rgb)
    imageio.imwrite(output_dir / "sam6d_mask.png", (mask.astype(np.uint8) * 255))
    imageio.imwrite(output_dir / "foundationpose_pose_overlay.png", overlay)
    _save_pipeline_figure(
        output_dir / "figure_global_detection_and_pose",
        rgb,
        depth,
        mask,
        overlay,
    )

    result = {
        "schema_version": "kcg_foundationpose_single_frame_v1",
        "method": "NVLABS_FOUNDATIONPOSE_MODEL_BASED",
        "initialization_method": initialization_method,
        "online_inputs": {
            "rgb": str(args.rgb.resolve()),
            "depth_m": str(args.depth_npy.resolve()),
            "mask": str(args.mask.resolve()),
            "camera_intrinsics": str(args.intrinsics_json.resolve()),
            "registration_mesh": str(args.mesh.resolve()),
        },
        "truth_inputs_used": [],
        "mesh_scale_to_m": float(args.mesh_scale_to_m),
        "mesh_original_face_count": original_face_count,
        "mesh_used_face_count": used_face_count,
        "valid_mask_depth_pixel_count": int(valid_mask_depth.sum()),
        "mask_pixel_count": int(mask.sum()),
        "camera_from_object_row_major": camera_from_object.ravel().tolist(),
        "prior_world_from_object_row_major": (
            None
            if prior_world_from_object is None
            else prior_world_from_object.ravel().tolist()
        ),
        "prior_camera_from_object_row_major": (
            None
            if prior_camera_from_object is None
            else prior_camera_from_object.ravel().tolist()
        ),
        "world_from_object_row_major": (
            None if world_from_object is None else world_from_object.ravel().tolist()
        ),
        "refine_iterations": int(args.refine_iterations),
        "local_refinement_performed": not args.initialize_from_prior_only,
        "foundationpose_debug": int(args.foundationpose_debug),
        "foundationpose_repository": str(foundationpose_root),
        "foundationpose_git_commit": subprocess.run(
            ["git", "-C", str(foundationpose_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "interpretation": (
            "single-frame pose initialization/refinement; not an accuracy certification and not "
            "robot-motion authorization"
        ),
    }
    (output_dir / "foundationpose_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    np.savetxt(output_dir / "camera_from_object.txt", camera_from_object, fmt="%.10f")
    if world_from_object is not None:
        np.savetxt(output_dir / "world_from_object.txt", world_from_object, fmt="%.10f")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
