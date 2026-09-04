#!/usr/bin/env python3
"""Remove support-plane shadow pixels from one SAM-6D detection mask."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as mask_util


def _decode_rle(segmentation: dict[str, object]) -> np.ndarray:
    counts = segmentation["counts"]
    if not isinstance(counts, list):
        return mask_util.decode(segmentation).astype(bool)
    height, width = (int(value) for value in segmentation["size"])
    flat = np.empty(height * width, dtype=np.uint8)
    cursor = 0
    value = 0
    for count_value in counts:
        count = int(count_value)
        flat[cursor : cursor + count] = value
        cursor += count
        value = 1 - value
    if cursor != len(flat):
        raise ValueError("uncompressed COCO RLE length differs from mask size")
    return flat.reshape((height, width), order="F").astype(bool)


def _fit_support_plane(
    points: np.ndarray,
    *,
    iterations: int,
    residual_limit_m: float,
) -> tuple[np.ndarray, float, float]:
    if len(points) < 100:
        raise ValueError("fewer than 100 background depth points")
    random = np.random.default_rng(0)
    best_inliers: np.ndarray | None = None
    for _ in range(iterations):
        sample = points[random.choice(len(points), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = float(np.linalg.norm(normal))
        if norm < 1.0e-9:
            continue
        normal /= norm
        offset = -float(normal @ sample[0])
        inliers = np.abs(points @ normal + offset) < residual_limit_m
        if best_inliers is None or int(inliers.sum()) > int(best_inliers.sum()):
            best_inliers = inliers
    if best_inliers is None or int(best_inliers.sum()) < 100:
        raise RuntimeError("dominant support plane was not found")
    plane_points = points[best_inliers]
    center = plane_points.mean(axis=0)
    _, _, right_vectors = np.linalg.svd(plane_points - center, full_matrices=False)
    normal = right_vectors[-1]
    offset = -float(normal @ center)
    # Make the camera origin lie on the positive side of the plane.  Object
    # pixels above the support and closer to the camera then have positive height.
    if offset < 0.0:
        normal = -normal
        offset = -offset
    return normal, offset, float(best_inliers.mean())


def _largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if count < 2:
        raise RuntimeError("depth filtering removed every foreground component")
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    selected = labels == label
    return cv2.morphologyEx(
        selected.astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
    ).astype(bool)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detections-json", type=Path, required=True)
    depth_input = parser.add_mutually_exclusive_group(required=True)
    depth_input.add_argument("--depth-mm", type=Path)
    depth_input.add_argument("--depth-m-npy", type=Path)
    parser.add_argument("--camera-json", type=Path, required=True)
    parser.add_argument("--output-mask", type=Path, required=True)
    parser.add_argument("--output-detections-json", type=Path, required=True)
    parser.add_argument("--output-summary-json", type=Path, required=True)
    parser.add_argument("--minimum-height-m", type=float, default=0.001)
    parser.add_argument("--plane-residual-limit-m", type=float, default=0.0005)
    parser.add_argument("--ransac-iterations", type=int, default=256)
    parser.add_argument("--background-sample-stride", type=int, default=4)
    args = parser.parse_args()

    outputs = (
        args.output_mask,
        args.output_detections_json,
        args.output_summary_json,
    )
    if any(path.exists() for path in outputs):
        raise FileExistsError("refusing to overwrite a depth-filter output")
    if (
        args.minimum_height_m <= 0.0
        or args.plane_residual_limit_m <= 0.0
        or args.ransac_iterations < 1
        or args.background_sample_stride < 1
    ):
        raise ValueError("filter parameters must be positive")

    detections = json.loads(args.detections_json.read_text(encoding="utf-8"))
    if not detections:
        raise ValueError("SAM-6D returned no detections")
    detection = max(detections, key=lambda item: float(item["score"]))
    broad_mask = _decode_rle(detection["segmentation"])
    camera = json.loads(args.camera_json.read_text(encoding="utf-8"))
    intrinsics = np.asarray(camera["cam_K"], dtype=np.float64).reshape(3, 3)
    if args.depth_m_npy is not None:
        depth_m = np.asarray(np.load(args.depth_m_npy), dtype=np.float64)
        depth_source = "FLOAT_METERS_NPY"
    else:
        assert args.depth_mm is not None
        depth_raw = np.asarray(Image.open(args.depth_mm), dtype=np.float64)
        depth_m = depth_raw * float(camera["depth_scale"]) / 1000.0
        depth_source = "INTEGER_MILLIMETERS_PNG"
    if depth_m.shape != broad_mask.shape:
        raise ValueError("depth and SAM mask shapes differ")

    height, width = depth_m.shape
    yy, xx = np.indices(depth_m.shape)
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    stride = int(args.background_sample_stride)
    background = valid & ~broad_mask & (yy % stride == 0) & (xx % stride == 0)
    z = depth_m[background]
    x = (xx[background] - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (yy[background] - intrinsics[1, 2]) * z / intrinsics[1, 1]
    normal, offset, inlier_fraction = _fit_support_plane(
        np.column_stack((x, y, z)),
        iterations=int(args.ransac_iterations),
        residual_limit_m=float(args.plane_residual_limit_m),
    )

    z = depth_m[broad_mask]
    x = (xx[broad_mask] - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (yy[broad_mask] - intrinsics[1, 2]) * z / intrinsics[1, 1]
    signed_height = np.zeros(depth_m.shape, dtype=np.float64)
    signed_height[broad_mask] = np.column_stack((x, y, z)) @ normal + offset
    filtered = broad_mask & valid & (
        signed_height > float(args.minimum_height_m)
    )
    filtered = _largest_component(filtered)

    args.output_mask.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((filtered * 255).astype(np.uint8)).save(args.output_mask)
    encoded = mask_util.encode(np.asfortranarray(filtered.astype(np.uint8)))
    encoded_json = {
        "size": [int(value) for value in encoded["size"]],
        "counts": encoded["counts"].decode("ascii"),
    }
    ys, xs = np.nonzero(filtered)
    bbox = [
        int(xs.min()),
        int(ys.min()),
        int(xs.max() - xs.min() + 1),
        int(ys.max() - ys.min() + 1),
    ]
    filtered_detection = dict(detection)
    filtered_detection["bbox"] = bbox
    filtered_detection["segmentation"] = encoded_json
    args.output_detections_json.write_text(
        json.dumps([filtered_detection], indent=2) + "\n", encoding="utf-8"
    )

    summary = {
        "schema_version": "kcg_sam6d_depth_support_filter_v1",
        "source_detection_score": float(detection["score"]),
        "broad_mask_pixels": int(broad_mask.sum()),
        "filtered_mask_pixels": int(filtered.sum()),
        "removed_pixels": int((broad_mask & ~filtered).sum()),
        "filtered_bbox_xywh": bbox,
        "minimum_image_border_px": int(
            min(xs.min(), width - 1 - xs.max(), ys.min(), height - 1 - ys.max())
        ),
        "filtered_valid_depth_fraction": float(np.mean(valid[filtered])),
        "support_plane_normal_camera": normal.tolist(),
        "support_plane_offset_m": float(offset),
        "support_plane_ransac_inlier_fraction": inlier_fraction,
        "minimum_height_above_support_m": float(args.minimum_height_m),
        "depth_source": depth_source,
        "truth_inputs_used": [],
    }
    args.output_summary_json.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
