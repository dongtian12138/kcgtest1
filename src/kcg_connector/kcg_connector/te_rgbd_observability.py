"""Truth-free geometric preflight for the TE J35 RGB-D pose provider.

This module deliberately stops before CAD registration.  It answers the
earliest question for a frozen camera view: does an ordinary depth frame
contain enough endpoint samples, and could the smallest TE key feature span
enough pixels to support a unique keyed pose at all?

The optimistic pixel-width test is a fail-closed necessary condition.  It
uses only the measured optical depth, camera intrinsics, frozen camera
calibration, frozen workspaces, and public CAD dimensions.  Passing the test
does not authorize motion; failing it proves that the view cannot resolve all
five keys even under the most favourable in-plane orientation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


MISS_ENDPOINT_NOT_FOUND = "MISS_ENDPOINT_NOT_FOUND"
MISS_KEY_NOT_OBSERVABLE = "MISS_KEY_NOT_OBSERVABLE"
READY_FOR_FULL_CAD_REGISTRATION = "READY_FOR_FULL_CAD_REGISTRATION"


@dataclass(frozen=True)
class EndpointWorkspace:
    """Frozen world-axis-aligned endpoint workspace."""

    name: str
    minimum_world_m: tuple[float, float, float]
    maximum_world_m: tuple[float, float, float]
    minimum_points: int
    smallest_key_chord_m: float

    def __post_init__(self) -> None:
        lower = np.asarray(self.minimum_world_m, dtype=np.float64)
        upper = np.asarray(self.maximum_world_m, dtype=np.float64)
        if lower.shape != (3,) or upper.shape != (3,):
            raise ValueError(f"{self.name}: workspace bounds must be 3-vectors")
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise ValueError(f"{self.name}: workspace bounds must be finite")
        if not np.all(lower < upper):
            raise ValueError(f"{self.name}: workspace minimum must precede maximum")
        if self.minimum_points < 1:
            raise ValueError(f"{self.name}: minimum_points must be positive")
        if not np.isfinite(self.smallest_key_chord_m) or self.smallest_key_chord_m <= 0.0:
            raise ValueError(f"{self.name}: smallest_key_chord_m must be positive")


def world_from_camera_cv(
    eye_world_m: Sequence[float], target_world_m: Sequence[float]
) -> np.ndarray:
    """Return ``T_world_camera`` for CV axes x-right, y-down, z-forward.

    This is the inverse convention of ``fixed_camera_model`` in the existing
    registration module.  It is computed from a frozen calibration contract,
    never from a runtime object transform.
    """

    eye = np.asarray(eye_world_m, dtype=np.float64)
    target = np.asarray(target_world_m, dtype=np.float64)
    if eye.shape != (3,) or target.shape != (3,):
        raise ValueError("camera eye and target must be 3-vectors")
    forward = target - eye
    forward_norm = float(np.linalg.norm(forward))
    if not np.isfinite(forward_norm) or forward_norm <= 0.0:
        raise ValueError("camera eye and target must differ")
    forward /= forward_norm
    right = np.cross(forward, np.asarray((0.0, 0.0, 1.0)))
    right_norm = float(np.linalg.norm(right))
    if not np.isfinite(right_norm) or right_norm <= 1.0e-12:
        raise ValueError("camera view direction cannot be parallel to world Z")
    right /= right_norm
    up = np.cross(right, forward)
    rotation = np.column_stack((right, -up, forward))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = eye
    return transform


def backproject_depth_to_world(
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
    world_from_camera: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Back-project positive finite image-plane depth to world points.

    Returns world points and their integer ``(v, u)`` pixel coordinates.  The
    input depth must be the ordinary ``distance_to_image_plane`` channel.
    """

    depth = np.asarray(depth_m, dtype=np.float64)
    camera_matrix = np.asarray(intrinsics, dtype=np.float64)
    transform = np.asarray(world_from_camera, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError("depth_m must be a 2D image")
    if camera_matrix.shape != (3, 3):
        raise ValueError("intrinsics must be a 3x3 matrix")
    if transform.shape != (4, 4):
        raise ValueError("world_from_camera must be a 4x4 matrix")
    if not np.all(np.isfinite(camera_matrix)) or not np.all(np.isfinite(transform)):
        raise ValueError("camera calibration must be finite")
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("camera focal lengths must be positive")
    valid = np.isfinite(depth) & (depth > 0.0)
    pixel_v, pixel_u = np.nonzero(valid)
    z = depth[valid]
    x = (pixel_u.astype(np.float64) - cx) * z / fx
    y = (pixel_v.astype(np.float64) - cy) * z / fy
    camera_points = np.column_stack((x, y, z))
    world_points = (
        camera_points @ transform[:3, :3].T + transform[:3, 3]
    )
    return world_points, pixel_v, pixel_u


def _workspace_record(
    *,
    world_points: np.ndarray,
    optical_depth_m: np.ndarray,
    workspace: EndpointWorkspace,
    focal_length_px: float,
    minimum_key_width_px: float,
) -> dict[str, Any]:
    lower = np.asarray(workspace.minimum_world_m, dtype=np.float64)
    upper = np.asarray(workspace.maximum_world_m, dtype=np.float64)
    selected = np.all((world_points >= lower) & (world_points <= upper), axis=1)
    count = int(np.sum(selected))
    record: dict[str, Any] = {
        "endpoint": workspace.name,
        "workspace_minimum_world_m": lower.tolist(),
        "workspace_maximum_world_m": upper.tolist(),
        "point_count": count,
        "minimum_point_count": int(workspace.minimum_points),
        "smallest_key_chord_m": float(workspace.smallest_key_chord_m),
        "minimum_key_width_px": float(minimum_key_width_px),
    }
    if count < workspace.minimum_points:
        record.update(
            {
                "status": MISS_ENDPOINT_NOT_FOUND,
                "optimistic_smallest_key_width_px": None,
                "median_optical_depth_m": None,
            }
        )
        return record

    selected_depth = optical_depth_m[selected]
    median_depth = float(np.median(selected_depth))
    nearest_depth = float(np.min(selected_depth))
    # Nearest depth is deliberately used for the gate.  It is the largest
    # possible pixel width in this measured endpoint cloud, so failure is a
    # strict view-resolution failure, not a pessimistic scoring choice.
    optimistic_width = float(
        focal_length_px * workspace.smallest_key_chord_m / nearest_depth
    )
    record.update(
        {
            "median_optical_depth_m": median_depth,
            "nearest_optical_depth_m": nearest_depth,
            "optimistic_smallest_key_width_px": optimistic_width,
            "width_bound_kind": "nearest_measured_depth_optimistic_upper_bound",
        }
    )
    if optimistic_width < minimum_key_width_px:
        record["status"] = MISS_KEY_NOT_OBSERVABLE
    else:
        record["status"] = READY_FOR_FULL_CAD_REGISTRATION
    return record


def evaluate_te_rgbd_observability(
    *,
    rgb: np.ndarray,
    depth_m: np.ndarray,
    static_depth_m: np.ndarray,
    intrinsics: np.ndarray,
    world_from_camera: np.ndarray,
    workspaces: Sequence[EndpointWorkspace],
    minimum_key_width_px: float = 4.0,
    minimum_foreground_depth_delta_m: float = 0.00025,
) -> dict[str, Any]:
    """Evaluate endpoint presence and a necessary keyed-pose resolution gate.

    A ``READY_FOR_FULL_CAD_REGISTRATION`` result is not a pose result and does
    not authorize motion.  It only says that the current pixel scale does not
    make five-key registration impossible before registration even starts.
    """

    rgb_array = np.asarray(rgb)
    depth = np.asarray(depth_m, dtype=np.float64)
    static_depth = np.asarray(static_depth_m, dtype=np.float64)
    camera_matrix = np.asarray(intrinsics, dtype=np.float64)
    if rgb_array.ndim != 3 or rgb_array.shape[2] < 3:
        raise ValueError("rgb must be an HxWx3-or-more array")
    if rgb_array.shape[:2] != depth.shape:
        raise ValueError("rgb and depth dimensions differ")
    if static_depth.shape != depth.shape:
        raise ValueError("static and observed depth dimensions differ")
    if not np.isfinite(minimum_key_width_px) or minimum_key_width_px <= 0.0:
        raise ValueError("minimum_key_width_px must be positive")
    if (
        not np.isfinite(minimum_foreground_depth_delta_m)
        or minimum_foreground_depth_delta_m <= 0.0
    ):
        raise ValueError("minimum_foreground_depth_delta_m must be positive")
    if not workspaces:
        raise ValueError("at least one endpoint workspace is required")

    observed_valid = np.isfinite(depth) & (depth > 0.0)
    static_valid = np.isfinite(static_depth) & (static_depth > 0.0)
    both_valid = observed_valid & static_valid
    depth_delta = np.zeros_like(depth)
    depth_delta[both_valid] = static_depth[both_valid] - depth[both_valid]
    foreground = observed_valid & (
        (~static_valid)
        | (both_valid & (depth_delta >= minimum_foreground_depth_delta_m))
    )
    foreground_depth = np.where(foreground, depth, np.nan)
    world_points, pixel_v, pixel_u = backproject_depth_to_world(
        foreground_depth, camera_matrix, world_from_camera
    )
    optical_depth = depth[pixel_v, pixel_u]
    focal_length_px = float(min(camera_matrix[0, 0], camera_matrix[1, 1]))
    endpoint_records = [
        _workspace_record(
            world_points=world_points,
            optical_depth_m=optical_depth,
            workspace=workspace,
            focal_length_px=focal_length_px,
            minimum_key_width_px=float(minimum_key_width_px),
        )
        for workspace in workspaces
    ]
    failed = [
        record for record in endpoint_records
        if record["status"] != READY_FOR_FULL_CAD_REGISTRATION
    ]
    if failed:
        overall_status = (
            MISS_ENDPOINT_NOT_FOUND
            if any(record["status"] == MISS_ENDPOINT_NOT_FOUND for record in failed)
            else MISS_KEY_NOT_OBSERVABLE
        )
    else:
        overall_status = READY_FOR_FULL_CAD_REGISTRATION
    return {
        "schema_version": "kcg_te_rgbd_observability_v1",
        "status": overall_status,
        "control_authorized": False,
        "full_6d_pose_claimed": False,
        "unique_main_key_claimed": False,
        "provider_inputs": [
            "ordinary_rgb",
            "ordinary_depth",
            "ordinary_static_scene_depth",
            "camera_intrinsics",
            "frozen_world_from_camera_cv",
            "frozen_endpoint_workspaces",
            "public_te_key_dimensions",
        ],
        "truth_inputs_used": [],
        "semantic_or_instance_inputs_used": [],
        "minimum_key_width_px": float(minimum_key_width_px),
        "minimum_foreground_depth_delta_m": float(
            minimum_foreground_depth_delta_m
        ),
        "observed_valid_depth_pixel_count": int(np.sum(observed_valid)),
        "static_valid_depth_pixel_count": int(np.sum(static_valid)),
        "foreground_depth_pixel_count": int(np.sum(foreground)),
        "valid_depth_point_count": int(len(world_points)),
        "endpoints": endpoint_records,
        "interpretation": (
            "necessary_view_gate_only_not_a_pose_or_control_authorization"
        ),
    }


def workspaces_from_config(
    observation_contract: Mapping[str, Any], *, minimum_points: int = 150
) -> tuple[EndpointWorkspace, EndpointWorkspace]:
    """Build the two frozen endpoint workspaces from the observe-only config."""

    def bounds(prefix: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
        value = observation_contract[f"{prefix}_workspace_world_aabb_m"]
        return tuple(value["minimum"]), tuple(value["maximum"])

    plug_min, plug_max = bounds("plug")
    receptacle_min, receptacle_max = bounds("receptacle")
    return (
        EndpointWorkspace(
            name="plug",
            minimum_world_m=plug_min,
            maximum_world_m=plug_max,
            minimum_points=minimum_points,
            smallest_key_chord_m=0.0013208,
        ),
        EndpointWorkspace(
            name="receptacle",
            minimum_world_m=receptacle_min,
            maximum_world_m=receptacle_max,
            minimum_points=minimum_points,
            smallest_key_chord_m=0.00160,
        ),
    )


__all__ = [
    "EndpointWorkspace",
    "MISS_ENDPOINT_NOT_FOUND",
    "MISS_KEY_NOT_OBSERVABLE",
    "READY_FOR_FULL_CAD_REGISTRATION",
    "backproject_depth_to_world",
    "evaluate_te_rgbd_observability",
    "workspaces_from_config",
    "world_from_camera_cv",
]
