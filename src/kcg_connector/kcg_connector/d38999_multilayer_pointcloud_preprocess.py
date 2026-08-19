"""Truth-free RGB-D point-cloud preprocessing for the multilayer model.

Open3D is not installed in the current environment.  This module therefore
implements the required back-projection, invalid-depth filtering and voxel
downsampling with a deterministic NumPy equivalent.  No mask, object pose,
contact data or simulator truth is accepted.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np
import yaml


SCHEMA_VERSION = "kcg_d38999_multilayer_pointcloud_preprocess_v1"
SELECTED_BACKEND = "NUMPY_DETERMINISTIC_EQUIVALENT"
MINIMUM_DEPTH_M = 0.02
MAXIMUM_DEPTH_M = 10.0
FRAME_ID_PATTERN = re.compile(r"^[A-Za-z0-9_./:-]{1,256}$")

FROZEN_SOURCES = {
    "artifacts/agent_control/tasks/EIGHT-HOUR-C4-RGBD-SAVE/"
    "ARCHIVE_CONTRACT_MANIFEST.json": (
        "9c076b68373beb6b62595f3032533add2bcc5a80523bd20cb059e808d6b371b1"
    ),
    "src/kcg_connector/config/d38999_master_model_contract_v1.yaml": (
        "57010b3c6f8d2214712427193ef7b9b57ede3089a882aa88033f89774215c68b"
    ),
    "artifacts/kcg_connector/isaac/d38999_multilayer_v1/MODEL_MAPPING.json": (
        "a31d4c0e7cb911f0371c257b0784506388f0f52d1d07401c1b1c2239fa47a783"
    ),
    "src/kcg_connector/kcg_connector/postgrasp_shadow_estimator.py": (
        "b164a83fa5039d39664cbfa6ac4bca1ca7976158f019f3fe174de71afaaa6a8e"
    ),
}


@dataclass(frozen=True)
class PointCloudPreprocessResult:
    points_camera_m: np.ndarray
    colors_rgb_u8: np.ndarray
    summary: dict[str, Any]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping(path: Path, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise ValueError(f"{label} missing: {path}")
    if path.suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
    else:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _verified_sources(root: Path) -> tuple[dict[str, str], ...]:
    rows = []
    for relative, expected in FROZEN_SOURCES.items():
        path = root / relative
        if not path.is_file():
            raise ValueError(f"frozen point-cloud source missing: {relative}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"frozen point-cloud source hash mismatch: {relative}")
        rows.append({"path": relative, "sha256": actual})
    return tuple(rows)


def build_multilayer_pointcloud_preprocess_contract(
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    sources = _verified_sources(root)
    archive = _mapping(
        root
        / "artifacts/agent_control/tasks/EIGHT-HOUR-C4-RGBD-SAVE/"
        "ARCHIVE_CONTRACT_MANIFEST.json",
        "C4 archive contract",
    )
    master = _mapping(
        root / "src/kcg_connector/config/d38999_master_model_contract_v1.yaml",
        "master model contract",
    )
    model_mapping = _mapping(
        root
        / "artifacts/kcg_connector/isaac/d38999_multilayer_v1/"
        "MODEL_MAPPING.json",
        "multilayer mapping",
    )
    if (
        archive.get("status") != "OFFLINE_PASS"
        or archive.get("current_readiness", {}).get(
            "dynamic_capture_archives_available"
        )
        != 0
        or archive.get("dynamic_rgbd_pass_claimed") is not False
        or archive.get("raw_capture", {}).get("channels_exactly")
        != ["rgb", "distance_to_image_plane"]
    ):
        raise ValueError("C4 archive boundary changed")
    try:
        visual_requirement = master["representation_requirements"][
            "D38999_VISUAL_COMPLETE_V1"
        ]
        visual_mapping = model_mapping["representations"][
            "D38999_VISUAL_COMPLETE_V1"
        ]
    except (KeyError, TypeError):
        raise ValueError("multilayer point-cloud authority missing") from None
    if (
        "point_cloud" not in visual_requirement.get("purpose", [])
        or visual_mapping.get("root") != archive["observation_target"]["root"]
        or visual_mapping.get("visible_geometry_preserved") is not True
    ):
        raise ValueError("multilayer point-cloud target changed")
    open3d_available = importlib.util.find_spec("open3d") is not None
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "OFFLINE_PASS",
        "classification": "TRUTH_FREE_POINTCLOUD_PREPROCESS_INTERFACE",
        "input_archive_contract": (
            "kcg_d38999_multilayer_rgbd_archive_contract_v1"
        ),
        "input_channels_exactly": ["rgb", "distance_to_image_plane"],
        "input_depth_semantics": "CAMERA_OPTICAL_Z_DISTANCE_M",
        "output_frame_semantics": "INPUT_CAMERA_OPTICAL_FRAME",
        "depth_range_m": [MINIMUM_DEPTH_M, MAXIMUM_DEPTH_M],
        "selected_backend": SELECTED_BACKEND,
        "open3d_available": open3d_available,
        "dependency_resolution": "EQUIVALENT_NUMPY_PATH_SELECTED",
        "voxel_size_policy": {
            "project_default_defined": False,
            "caller_must_supply_positive_m": True,
            "acceptance_threshold": False,
            "parameter_search_allowed": False,
        },
        "operations": [
            "pinhole_depth_backprojection",
            "finite_positive_depth_filter",
            "frozen_clipping_range_filter",
            "deterministic_voxel_centroid_downsample",
            "finite_output_assertion",
            "array_sha256_summary",
        ],
        "truth_firewall": {
            "semantic_mask_allowed": False,
            "object_pose_allowed": False,
            "contact_report_allowed": False,
            "collider_identity_allowed": False,
            "contact_normal_allowed": False,
            "postrun_object_pose_write_allowed": False,
        },
        "observation_target": dict(archive["observation_target"]),
        "current_readiness": {
            "offline_fixture_processing_allowed": True,
            "dynamic_archives_available": 0,
            "dynamic_pointcloud_pass_claimed": False,
        },
        "sources": list(sources),
        "simulation_started": False,
        "dynamic_pointcloud_pass_claimed": False,
        "hardware_authorized": False,
    }


def _intrinsics(value: Mapping[str, Any], shape: tuple[int, int]) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != {
        "width",
        "height",
        "fx",
        "fy",
        "cx",
        "cy",
    }:
        raise ValueError("camera_intrinsics keys differ")
    rows = {
        key: float(value[key]) for key in ("width", "height", "fx", "fy", "cx", "cy")
    }
    height, width = shape
    if (
        int(rows["width"]) != width
        or int(rows["height"]) != height
        or rows["width"] != width
        or rows["height"] != height
        or not all(np.isfinite(number) for number in rows.values())
        or rows["fx"] <= 0.0
        or rows["fy"] <= 0.0
    ):
        raise ValueError("camera intrinsics or image dimensions are invalid")
    return rows


def preprocess_rgbd_pointcloud(
    depth_m: np.ndarray,
    rgb: np.ndarray,
    camera_intrinsics: Mapping[str, Any],
    *,
    frame_id: str,
    voxel_size_m: float,
) -> PointCloudPreprocessResult:
    """Back-project and downsample in the declared camera optical frame."""

    depth = np.asarray(depth_m)
    colors = np.asarray(rgb)
    if depth.ndim != 2 or colors.shape != (*depth.shape, 3):
        raise ValueError("RGB-D shapes differ")
    if colors.dtype != np.uint8:
        raise ValueError("RGB must be uint8")
    if FRAME_ID_PATTERN.fullmatch(frame_id) is None:
        raise ValueError("frame_id is empty or unsafe")
    voxel_size = float(voxel_size_m)
    if not np.isfinite(voxel_size) or voxel_size <= 0.0:
        raise ValueError("voxel_size_m must be finite and positive")
    camera = _intrinsics(camera_intrinsics, depth.shape)

    valid = (
        np.isfinite(depth)
        & (depth >= MINIMUM_DEPTH_M)
        & (depth <= MAXIMUM_DEPTH_M)
    )
    if not np.any(valid):
        raise ValueError("RGB-D frame has no valid depth points")
    rows, columns = np.indices(depth.shape, dtype=np.float64)
    z = depth[valid].astype(np.float64)
    x = (columns[valid] - camera["cx"]) * z / camera["fx"]
    y = (rows[valid] - camera["cy"]) * z / camera["fy"]
    points = np.column_stack((x, y, z))
    source_colors = colors[valid]
    if not np.all(np.isfinite(points)):
        raise ValueError("back-projection produced non-finite points")

    voxel_keys = np.floor(points / voxel_size).astype(np.int64)
    unique_keys, inverse = np.unique(voxel_keys, axis=0, return_inverse=True)
    counts = np.bincount(inverse, minlength=len(unique_keys)).astype(np.float64)
    point_sums = np.zeros((len(unique_keys), 3), dtype=np.float64)
    color_sums = np.zeros((len(unique_keys), 3), dtype=np.float64)
    np.add.at(point_sums, inverse, points)
    np.add.at(color_sums, inverse, source_colors.astype(np.float64))
    reduced_points = point_sums / counts[:, None]
    reduced_colors = np.rint(color_sums / counts[:, None]).astype(np.uint8)
    if not np.all(np.isfinite(reduced_points)):
        raise ValueError("voxel downsampling produced non-finite points")

    canonical_points = np.asarray(reduced_points, dtype="<f8")
    canonical_colors = np.asarray(reduced_colors, dtype=np.uint8)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "OFFLINE_PASS",
        "evidence_level": "OFFLINE_FIXTURE_OR_CALLER_SUPPLIED_RGBD",
        "selected_backend": SELECTED_BACKEND,
        "open3d_available": importlib.util.find_spec("open3d") is not None,
        "frame_id": frame_id,
        "point_frame": "camera_optical",
        "depth_range_m": [MINIMUM_DEPTH_M, MAXIMUM_DEPTH_M],
        "voxel_size_m": voxel_size,
        "input_pixel_count": int(depth.size),
        "valid_depth_point_count": int(np.count_nonzero(valid)),
        "filtered_depth_pixel_count": int(depth.size - np.count_nonzero(valid)),
        "output_point_count": int(len(canonical_points)),
        "bounds_min_m": canonical_points.min(axis=0).tolist(),
        "bounds_max_m": canonical_points.max(axis=0).tolist(),
        "centroid_m": canonical_points.mean(axis=0).tolist(),
        "points_sha256": hashlib.sha256(canonical_points.tobytes()).hexdigest(),
        "colors_sha256": hashlib.sha256(canonical_colors.tobytes()).hexdigest(),
        "semantic_mask_used": False,
        "object_pose_truth_used": False,
        "contact_truth_used": False,
        "dynamic_pointcloud_pass_claimed": False,
    }
    return PointCloudPreprocessResult(
        points_camera_m=canonical_points,
        colors_rgb_u8=canonical_colors,
        summary=summary,
    )


__all__ = [
    "FROZEN_SOURCES",
    "PointCloudPreprocessResult",
    "SCHEMA_VERSION",
    "SELECTED_BACKEND",
    "build_multilayer_pointcloud_preprocess_contract",
    "preprocess_rgbd_pointcloud",
]
