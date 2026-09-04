#!/usr/bin/env python3
"""Keep one official FoundationPose tracker alive across RGB-D frames.

The worker owns no robot or simulator state.  It reads one JSON request per
line from stdin and emits responses prefixed with ``FP_RESPONSE `` so the
Isaac runtime can ignore normal FoundationPose logging.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import trimesh


def _matrix3(values: object, label: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.size != 9:
        raise ValueError(f"{label} must contain nine values")
    matrix = matrix.reshape(3, 3)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{label} is not finite")
    return matrix


def _load_frame(request: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    rgb_bgr = cv2.imread(str(request["rgb"]), cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise FileNotFoundError(str(request["rgb"]))
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    depth = np.asarray(np.load(str(request["depth_npy"])), dtype=np.float32)
    if depth.shape != rgb.shape[:2]:
        raise ValueError("FoundationPose RGB and depth shapes differ")
    depth[~np.isfinite(depth) | (depth <= 0.0)] = 0.0
    return rgb, depth


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foundationpose-root", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--mesh-scale-to-m", type=float, default=0.001)
    parser.add_argument("--maximum-faces", type=int, default=50_000)
    parser.add_argument("--register-iterations", type=int, default=5)
    parser.add_argument("--track-iterations", type=int, default=2)
    parser.add_argument("--debug-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.foundationpose_root.resolve()
    if not (root / "estimater.py").is_file():
        raise FileNotFoundError(f"invalid FoundationPose root: {root}")
    if args.mesh_scale_to_m <= 0.0 or args.maximum_faces <= 0:
        raise ValueError("mesh scale and face limit must be positive")
    sys.path.insert(0, str(root))

    import nvdiffrast.torch as dr  # type: ignore
    from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor  # type: ignore
    from Utils import set_logging_format, set_seed  # type: ignore

    mesh = trimesh.load(args.mesh.resolve(), process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("FoundationPose mesh must be one Trimesh")
    mesh = mesh.copy()
    mesh.apply_scale(float(args.mesh_scale_to_m))
    if len(mesh.faces) > int(args.maximum_faces):
        mesh = mesh.simplify_quadric_decimation(face_count=int(args.maximum_faces))

    debug_dir = args.debug_dir.resolve()
    debug_dir.mkdir(parents=True, exist_ok=False)
    set_logging_format()
    set_seed(0)
    estimator = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=ScorePredictor(),
        refiner=PoseRefinePredictor(),
        debug_dir=str(debug_dir),
        debug=0,
        glctx=dr.RasterizeCudaContext(),
    )
    print("FP_READY", flush=True)

    initialized = False
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            command = str(request.get("command", ""))
            if command == "stop":
                print("FP_STOPPED", flush=True)
                return 0
            if command not in {"register", "initialize_from_pose", "track"}:
                raise ValueError(f"unknown worker command: {command}")
            rgb, depth = _load_frame(request)
            camera = _matrix3(request["camera_matrix"], "camera_matrix")
            started = time.perf_counter()
            if command == "register":
                if initialized:
                    raise RuntimeError("FoundationPose was already registered")
                mask_image = cv2.imread(str(request["mask"]), cv2.IMREAD_GRAYSCALE)
                if mask_image is None:
                    raise FileNotFoundError(str(request["mask"]))
                mask = mask_image > 0
                if mask.shape != depth.shape or int(np.sum(mask & (depth > 0.0))) < 4:
                    raise ValueError("registration mask has insufficient valid depth")
                pose = estimator.register(
                    K=camera,
                    rgb=rgb,
                    depth=depth,
                    ob_mask=mask,
                    iteration=int(args.register_iterations),
                )
                initialized = True
            elif command == "initialize_from_pose":
                if initialized:
                    raise RuntimeError("FoundationPose was already initialized")
                prior = np.asarray(
                    request["camera_from_object_row_major"], dtype=np.float64
                ).reshape(4, 4)
                if not np.isfinite(prior).all():
                    raise ValueError("initial camera pose is not finite")
                import torch

                object_to_center = (
                    estimator.get_tf_to_centered_mesh().detach().cpu().numpy()
                )
                estimator.pose_last = torch.as_tensor(
                    prior @ np.linalg.inv(object_to_center),
                    device="cuda",
                    dtype=torch.float32,
                )
                pose = estimator.track_one(
                    rgb=rgb,
                    depth=depth,
                    K=camera,
                    iteration=int(args.track_iterations),
                )
                initialized = True
            else:
                if not initialized:
                    raise RuntimeError("track requested before register")
                pose = estimator.track_one(
                    rgb=rgb,
                    depth=depth,
                    K=camera,
                    iteration=int(args.track_iterations),
                )
            elapsed = time.perf_counter() - started
            pose = np.asarray(pose, dtype=np.float64).reshape(4, 4)
            if not np.isfinite(pose).all():
                raise RuntimeError("FoundationPose returned a nonfinite pose")
            response = {
                "ok": True,
                "command": command,
                "elapsed_s": elapsed,
                "camera_from_object_row_major": pose.ravel().tolist(),
            }
        except Exception as error:
            response = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        print("FP_RESPONSE " + json.dumps(response, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
