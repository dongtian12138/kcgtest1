#!/usr/bin/env python3
"""Estimate the connector origin and directed axis from its visible circular face.

The hidden key and axial yaw are deliberately not estimated.  The method uses
the SAM mask only as a region of interest, fits the dominant visible face plane
from depth, finds the centre of the symmetric pin field in RGB, and shifts that
visible-face centre to the CAD object origin along the fitted axis.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.optimize import least_squares
import trimesh


def _visible_face_geometry(mesh_path: Path) -> tuple[float, float]:
    """Return visible-face radius and its distance to local z=0, in metres."""
    mesh = trimesh.load(mesh_path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"expected one triangle mesh: {mesh_path}")
    vertices = np.asarray(mesh.vertices, dtype=np.float64) * 0.001
    z_min = float(np.min(vertices[:, 2]))
    z_max = float(np.max(vertices[:, 2]))
    if not (z_min < -1.0e-6 and abs(z_max) < 1.0e-6):
        raise ValueError(
            "the plug mesh must place the hidden object origin at local z=0 "
            "and the visible face at negative z"
        )
    face_vertices = vertices[np.abs(vertices[:, 2] - z_min) < 1.0e-8]
    if len(face_vertices) < 16:
        raise ValueError("too few vertices on the visible face")
    radius = float(np.max(np.linalg.norm(face_vertices[:, :2], axis=1)))
    return radius, -z_min


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    for reference in (
        np.asarray((1.0, 0.0, 0.0)),
        np.asarray((0.0, 1.0, 0.0)),
    ):
        first = reference - float(reference @ normal) * normal
        norm = float(np.linalg.norm(first))
        if norm > 1.0e-8:
            first /= norm
            second = np.cross(normal, first)
            second /= np.linalg.norm(second)
            return first, second
    raise ValueError("cannot construct a basis for the face plane")


def _fit_visible_plane(
    points: np.ndarray,
    *,
    residual_limit_m: float,
    iterations: int,
) -> tuple[np.ndarray, float, np.ndarray, float]:
    if len(points) < 1000:
        raise ValueError("fewer than 1000 valid plug depth pixels")
    sampled = points[::4]
    random = np.random.default_rng(7)
    best_count = -1
    best_normal: np.ndarray | None = None
    best_offset = 0.0
    for _ in range(iterations):
        first, second, third = sampled[random.choice(len(sampled), 3, replace=False)]
        normal = np.cross(second - first, third - first)
        norm = float(np.linalg.norm(normal))
        if norm < 1.0e-9:
            continue
        normal /= norm
        offset = -float(normal @ first)
        count = int(
            np.count_nonzero(
                np.abs(sampled @ normal + offset) < residual_limit_m
            )
        )
        if count > best_count:
            best_count = count
            best_normal = normal
            best_offset = offset
    if best_normal is None:
        raise RuntimeError("the visible face plane was not found")

    normal = best_normal
    offset = best_offset
    inliers = np.zeros(len(points), dtype=bool)
    for _ in range(4):
        inliers = np.abs(points @ normal + offset) < residual_limit_m
        if int(inliers.sum()) < 1000:
            raise RuntimeError("too few visible face plane inliers")
        center = points[inliers].mean(axis=0)
        _, _, right_vectors = np.linalg.svd(
            points[inliers] - center, full_matrices=False
        )
        normal = right_vectors[-1]
        offset = -float(normal @ center)

    # The visible surface normal points from the plug toward the camera origin.
    if float(normal @ points.mean(axis=0)) > 0.0:
        normal = -normal
        offset = -offset
    sampled_fraction = best_count / len(sampled)
    return normal, offset, inliers, sampled_fraction


def _coarse_face_center(
    *,
    mask: np.ndarray,
    image_points: tuple[np.ndarray, np.ndarray],
    point_cloud: np.ndarray,
    normal: np.ndarray,
    offset: float,
    intrinsics: np.ndarray,
    face_radius_m: float,
    plane_residual_limit_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    yy, xx = image_points
    residual_image = np.full(mask.shape, np.inf, dtype=np.float64)
    residual_image[yy, xx] = np.abs(point_cloud @ normal + offset)
    component_mask = mask & (
        residual_image < max(plane_residual_limit_m, 0.00075)
    )
    component_mask = cv2.morphologyEx(
        component_mask.astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((9, 9), dtype=np.uint8),
    )
    contours, _ = cv2.findContours(
        component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        raise RuntimeError("the visible face component has no contour")
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    if len(contour) < 100 or cv2.contourArea(contour.astype(np.float32)) < 1000.0:
        raise RuntimeError("the visible face contour is too small")

    rays = np.column_stack(
        (
            (contour[:, 0] - intrinsics[0, 2]) / intrinsics[0, 0],
            (contour[:, 1] - intrinsics[1, 2]) / intrinsics[1, 1],
            np.ones(len(contour)),
        )
    )
    denominators = rays @ normal
    if np.any(np.abs(denominators) < 1.0e-6):
        raise RuntimeError("a face-contour ray is parallel to the face plane")
    boundary = rays * (-offset / denominators)[:, None]
    first, second = _plane_basis(normal)
    plane_origin = -offset * normal
    boundary_2d = np.column_stack(
        ((boundary - plane_origin) @ first, (boundary - plane_origin) @ second)
    )
    initial = np.median(boundary_2d, axis=0)
    solution = least_squares(
        lambda center: np.linalg.norm(boundary_2d - center, axis=1)
        - face_radius_m,
        initial,
        loss="soft_l1",
        f_scale=0.0003,
        max_nfev=200,
    )
    residual = np.linalg.norm(boundary_2d - solution.x, axis=1) - face_radius_m
    return (
        solution.x,
        first,
        second,
        float(np.sqrt(np.mean(residual**2))),
        int(np.count_nonzero(component_mask)),
    )


def estimate_plug_five_dof(
    *,
    rgb: np.ndarray,
    depth_mm: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray,
    mesh_path: Path,
    depth_bin_center_correction_mm: float = 0.5,
    plane_residual_limit_m: float = 0.00065,
    plane_ransac_iterations: int = 700,
) -> dict[str, object]:
    """Estimate camera-from-object while leaving axial yaw unobserved."""
    started = time.perf_counter()
    rgb = np.asarray(rgb)
    depth_mm = np.asarray(depth_mm, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    intrinsics = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
    if rgb.shape[:2] != depth_mm.shape or mask.shape != depth_mm.shape:
        raise ValueError("RGB, depth, and mask image shapes differ")
    if not np.isfinite(intrinsics).all():
        raise ValueError("camera intrinsics are not finite")

    face_radius_m, visible_face_to_origin_m = _visible_face_geometry(mesh_path)
    valid = mask & np.isfinite(depth_mm) & (depth_mm > 0.0)
    yy, xx = np.nonzero(valid)
    z = (depth_mm[yy, xx] + depth_bin_center_correction_mm) / 1000.0
    points = np.column_stack(
        (
            (xx - intrinsics[0, 2]) * z / intrinsics[0, 0],
            (yy - intrinsics[1, 2]) * z / intrinsics[1, 1],
            z,
        )
    )
    normal, offset, plane_inliers, ransac_fraction = _fit_visible_plane(
        points,
        residual_limit_m=plane_residual_limit_m,
        iterations=plane_ransac_iterations,
    )
    coarse_center_2d, first, second, circle_rms, component_pixels = (
        _coarse_face_center(
            mask=valid,
            image_points=(yy, xx),
            point_cloud=points,
            normal=normal,
            offset=offset,
            intrinsics=intrinsics,
            face_radius_m=face_radius_m,
            plane_residual_limit_m=plane_residual_limit_m,
        )
    )

    # The visible pin field is rotationally symmetric about the desired axis.
    # Its local dark-feature centroid is therefore a more accurate centre than
    # the outer silhouette, which also contains rear lugs and side walls.
    plane_origin = -offset * normal
    coarse_center_3d = (
        plane_origin
        + coarse_center_2d[0] * first
        + coarse_center_2d[1] * second
    )
    projected_center = intrinsics @ coarse_center_3d
    projected_center = projected_center[:2] / projected_center[2]
    projected_edge = intrinsics @ (coarse_center_3d + face_radius_m * first)
    projected_edge = projected_edge[:2] / projected_edge[2]
    face_radius_px = float(np.linalg.norm(projected_edge - projected_center))
    margin = int(math.ceil(1.15 * face_radius_px))
    x0 = max(0, int(math.floor(projected_center[0])) - margin)
    x1 = min(rgb.shape[1], int(math.ceil(projected_center[0])) + margin + 1)
    y0 = max(0, int(math.floor(projected_center[1])) - margin)
    y1 = min(rgb.shape[0], int(math.ceil(projected_center[1])) + margin + 1)
    if x1 - x0 < 20 or y1 - y0 < 20:
        raise RuntimeError("the projected visible face is outside the image")

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
    blur_sigma_px = float(np.clip(face_radius_px / 40.0, 2.0, 8.0))
    local_background = cv2.GaussianBlur(gray, (0, 0), blur_sigma_px)
    darkness = np.maximum(local_background - gray, 0.0)
    roi_y, roi_x = np.indices((y1 - y0, x1 - x0))
    roi_x = roi_x + x0
    roi_y = roi_y + y0
    rays = np.stack(
        (
            (roi_x - intrinsics[0, 2]) / intrinsics[0, 0],
            (roi_y - intrinsics[1, 2]) / intrinsics[1, 1],
            np.ones_like(roi_x),
        ),
        axis=-1,
    )
    denominators = np.einsum("hwc,c->hw", rays, normal)
    plane_points = rays * (-offset / denominators)[..., None]
    coordinate_first = np.einsum(
        "hwc,c->hw", plane_points - plane_origin, first
    )
    coordinate_second = np.einsum(
        "hwc,c->hw", plane_points - plane_origin, second
    )
    feature_radius_m = 0.9 * face_radius_m
    inside_pin_field = (
        (coordinate_first - coarse_center_2d[0]) ** 2
        + (coordinate_second - coarse_center_2d[1]) ** 2
        < feature_radius_m**2
    )
    weights = darkness[y0:y1, x0:x1] * inside_pin_field * valid[y0:y1, x0:x1]
    weight_sum = float(weights.sum())
    if not math.isfinite(weight_sum) or weight_sum < 1.0e-3:
        raise RuntimeError("the visible pin field has insufficient RGB contrast")
    face_center_2d = np.asarray(
        (
            float(np.sum(weights * coordinate_first) / weight_sum),
            float(np.sum(weights * coordinate_second) / weight_sum),
        )
    )
    center_shift = float(np.linalg.norm(face_center_2d - coarse_center_2d))
    if center_shift > 0.25 * face_radius_m:
        raise RuntimeError("RGB pin-field centre disagrees with the depth geometry")

    visible_face_center = (
        plane_origin + face_center_2d[0] * first + face_center_2d[1] * second
    )
    directed_axis = -normal
    object_origin = visible_face_center + visible_face_to_origin_m * directed_axis
    rotation_first, rotation_second = _plane_basis(directed_axis)
    camera_from_object = np.eye(4, dtype=np.float64)
    camera_from_object[:3, :3] = np.column_stack(
        (rotation_first, rotation_second, directed_axis)
    )
    camera_from_object[:3, 3] = object_origin

    plane_residuals = np.abs(points @ normal + offset)
    return {
        "camera_from_object": camera_from_object,
        "metrics": {
            "schema_version": "kcg_plug_five_dof_geometry_v1",
            "controlled_pose_components": "POSITION_AND_DIRECTED_AXIS_ONLY",
            "axial_yaw_estimated": False,
            "mask_valid_depth_pixels": int(len(points)),
            "visible_face_component_pixels": component_pixels,
            "visible_face_radius_m": face_radius_m,
            "visible_face_to_object_origin_m": visible_face_to_origin_m,
            "plane_ransac_sampled_inlier_fraction": ransac_fraction,
            "plane_refined_inlier_fraction": float(np.mean(plane_inliers)),
            "plane_residual_median_m": float(np.median(plane_residuals)),
            "plane_residual_p95_m": float(np.quantile(plane_residuals, 0.95)),
            "coarse_circle_residual_rms_m": circle_rms,
            "rgb_pin_feature_weight_sum": weight_sum,
            "rgb_pin_center_shift_from_coarse_m": center_shift,
            "visible_face_radius_px": face_radius_px,
            "depth_bin_center_correction_mm": depth_bin_center_correction_mm,
            "estimated_visible_face_center_camera_m": visible_face_center.tolist(),
            "estimated_directed_axis_camera": directed_axis.tolist(),
            "elapsed_s": time.perf_counter() - started,
            "truth_inputs_used": [],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--depth-mm", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--camera-json", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_json}")
    rgb_bgr = cv2.imread(str(args.rgb), cv2.IMREAD_COLOR)
    depth_mm = cv2.imread(str(args.depth_mm), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(args.mask), cv2.IMREAD_GRAYSCALE)
    if rgb_bgr is None or depth_mm is None or mask is None:
        raise FileNotFoundError("an RGB, depth, or mask input could not be read")
    camera = json.loads(args.camera_json.read_text(encoding="utf-8"))
    result = estimate_plug_five_dof(
        rgb=cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB),
        depth_mm=depth_mm,
        mask=mask > 0,
        intrinsics=np.asarray(camera["cam_K"], dtype=np.float64),
        mesh_path=args.mesh,
    )
    document = {
        "camera_from_object_row_major": np.asarray(
            result["camera_from_object"], dtype=np.float64
        ).ravel().tolist(),
        "metrics": result["metrics"],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(document, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(document, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
