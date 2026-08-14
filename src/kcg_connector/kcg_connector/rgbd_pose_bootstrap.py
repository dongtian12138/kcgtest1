"""Strict, Isaac-free helpers for the first D38999 RGB-D milestone.

The module validates the camera contract and converts a semantic mask plus a
depth image into auditable visibility/depth statistics.  It intentionally does
not manufacture a keyed 6D pose: an axially symmetric depth silhouette cannot
observe the connector key yaw.  The Isaac smoke therefore remains fail-closed
at the boundary to :mod:`kcg_connector.connector_pose` until a real detector or
fiducial estimator provides that missing orientation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np
import yaml


RGBD_BOOTSTRAP_SCHEMA_VERSION = "kcg_d38999_rgbd_bootstrap_v1"
DEFAULT_RGBD_BOOTSTRAP_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "config/d38999_rgbd_bootstrap_v1.yaml"
)
_PRIM_PATH = re.compile(r"^/World(?:/[A-Za-z_][A-Za-z0-9_]*)+$")


@dataclass(frozen=True)
class RgbdCamera:
    prim_path: str
    frame_id: str
    eye_m: tuple[float, float, float]
    target_m: tuple[float, float, float]
    resolution: tuple[int, int]
    frequency_hz: int
    warmup_frames: int


@dataclass(frozen=True)
class RgbdLabels:
    taxonomy: str
    loose_plug: str
    fixed_receptacle: str


@dataclass(frozen=True)
class RgbdPositionEstimator:
    kind: str
    mask_center_statistic: str
    loose_plug_registered_model_height_m: float
    loose_plug_registered_model_height_source: str
    fixed_receptacle_registered_model_height_m: float
    fixed_receptacle_registered_model_height_source: str


@dataclass(frozen=True)
class RgbdAcceptance:
    minimum_pixels_per_endpoint: int
    minimum_visible_fraction_per_endpoint: float
    minimum_valid_depth_m: float
    maximum_valid_depth_m: float
    maximum_xy_centroid_error_m: float


@dataclass(frozen=True)
class RgbdOutput:
    directory: str
    rgb_filename: str
    depth_preview_filename: str
    depth_numpy_filename: str
    semantic_preview_filename: str
    report_filename: str


@dataclass(frozen=True)
class D38999RgbdBootstrap:
    schema_version: str
    tabletop_config: str
    pose_contract_config: str
    camera: RgbdCamera
    labels: RgbdLabels
    position_estimator: RgbdPositionEstimator
    acceptance: RgbdAcceptance
    output: RgbdOutput


@dataclass(frozen=True)
class MaskDepthStatistics:
    pixel_count: int
    visible_fraction: float
    valid_depth_count: int
    minimum_depth_m: float
    median_depth_m: float
    maximum_depth_m: float


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _exact_keys(value: Mapping[str, Any], expected, label: str) -> None:
    actual = set(value)
    wanted = set(expected)
    if actual != wanted:
        raise ValueError(
            f"{label} keys differ; missing={sorted(wanted - actual)}, "
            f"extra={sorted(actual - wanted)}"
        )


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be non-empty unpadded text")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _positive(value: Any, label: str) -> float:
    result = _finite(value, label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{label} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return result


def _vector3(value: Any, label: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{label} must contain three finite numbers")
    return tuple(
        _finite(item, f"{label}[{index}]")
        for index, item in enumerate(value)
    )


def _repository_path(value: Any, label: str) -> str:
    result = _text(value, label)
    path = Path(result)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be repository-relative")
    return result


def _filename(value: Any, label: str, suffix: str) -> str:
    result = _text(value, label)
    path = Path(result)
    if path.name != result or not result.endswith(suffix):
        raise ValueError(f"{label} must be a local {suffix} filename")
    return result


def load_rgbd_bootstrap(
    path: Path | str = DEFAULT_RGBD_BOOTSTRAP_CONFIG_PATH,
) -> D38999RgbdBootstrap:
    """Load the versioned RGB-D bootstrap contract with exact schemas."""
    document = _mapping(
        yaml.safe_load(Path(path).read_text(encoding="utf-8")), "config"
    )
    _exact_keys(
        document,
        (
            "schema_version",
            "tabletop_config",
            "pose_contract_config",
            "camera",
            "labels",
            "position_estimator",
            "acceptance",
            "output",
        ),
        "config",
    )
    if document["schema_version"] != RGBD_BOOTSTRAP_SCHEMA_VERSION:
        raise ValueError("unsupported RGB-D bootstrap schema")

    camera_doc = _mapping(document["camera"], "camera")
    _exact_keys(camera_doc, RgbdCamera.__dataclass_fields__, "camera")
    prim_path = _text(camera_doc["prim_path"], "camera.prim_path")
    if not _PRIM_PATH.fullmatch(prim_path):
        raise ValueError("camera.prim_path must be an absolute /World path")
    resolution = camera_doc["resolution"]
    if not isinstance(resolution, (list, tuple)) or len(resolution) != 2:
        raise ValueError("camera.resolution must contain width and height")
    camera = RgbdCamera(
        prim_path=prim_path,
        frame_id=_text(camera_doc["frame_id"], "camera.frame_id"),
        eye_m=_vector3(camera_doc["eye_m"], "camera.eye_m"),
        target_m=_vector3(camera_doc["target_m"], "camera.target_m"),
        resolution=(
            _positive_int(resolution[0], "camera.resolution[0]"),
            _positive_int(resolution[1], "camera.resolution[1]"),
        ),
        frequency_hz=_positive_int(
            camera_doc["frequency_hz"], "camera.frequency_hz"
        ),
        warmup_frames=_positive_int(
            camera_doc["warmup_frames"], "camera.warmup_frames"
        ),
    )
    if camera.eye_m == camera.target_m:
        raise ValueError("camera eye and target must differ")

    labels_doc = _mapping(document["labels"], "labels")
    _exact_keys(labels_doc, RgbdLabels.__dataclass_fields__, "labels")
    labels = RgbdLabels(
        **{
            key: _text(labels_doc[key], f"labels.{key}")
            for key in RgbdLabels.__dataclass_fields__
        }
    )
    if labels.loose_plug == labels.fixed_receptacle:
        raise ValueError("endpoint labels must be distinct")

    estimator_doc = _mapping(
        document["position_estimator"], "position_estimator"
    )
    _exact_keys(
        estimator_doc,
        RgbdPositionEstimator.__dataclass_fields__,
        "position_estimator",
    )
    position_estimator = RgbdPositionEstimator(
        kind=_text(estimator_doc["kind"], "position_estimator.kind"),
        mask_center_statistic=_text(
            estimator_doc["mask_center_statistic"],
            "position_estimator.mask_center_statistic",
        ),
        loose_plug_registered_model_height_m=_positive(
            estimator_doc["loose_plug_registered_model_height_m"],
            "position_estimator.loose_plug_registered_model_height_m",
        ),
        loose_plug_registered_model_height_source=_text(
            estimator_doc["loose_plug_registered_model_height_source"],
            "position_estimator.loose_plug_registered_model_height_source",
        ),
        fixed_receptacle_registered_model_height_m=_positive(
            estimator_doc["fixed_receptacle_registered_model_height_m"],
            "position_estimator.fixed_receptacle_registered_model_height_m",
        ),
        fixed_receptacle_registered_model_height_source=_text(
            estimator_doc[
                "fixed_receptacle_registered_model_height_source"
            ],
            "position_estimator."
            "fixed_receptacle_registered_model_height_source",
        ),
    )
    if position_estimator.kind != "ray_plane_registered_model_height":
        raise ValueError("unsupported RGB-D position estimator")
    if (
        position_estimator.mask_center_statistic
        != "coordinatewise_median_semantic_mask_pixels"
    ):
        raise ValueError("unsupported semantic-mask center statistic")

    acceptance_doc = _mapping(document["acceptance"], "acceptance")
    _exact_keys(
        acceptance_doc,
        RgbdAcceptance.__dataclass_fields__,
        "acceptance",
    )
    acceptance = RgbdAcceptance(
        minimum_pixels_per_endpoint=_positive_int(
            acceptance_doc["minimum_pixels_per_endpoint"],
            "acceptance.minimum_pixels_per_endpoint",
        ),
        minimum_visible_fraction_per_endpoint=_positive(
            acceptance_doc["minimum_visible_fraction_per_endpoint"],
            "acceptance.minimum_visible_fraction_per_endpoint",
        ),
        minimum_valid_depth_m=_positive(
            acceptance_doc["minimum_valid_depth_m"],
            "acceptance.minimum_valid_depth_m",
        ),
        maximum_valid_depth_m=_positive(
            acceptance_doc["maximum_valid_depth_m"],
            "acceptance.maximum_valid_depth_m",
        ),
        maximum_xy_centroid_error_m=_positive(
            acceptance_doc["maximum_xy_centroid_error_m"],
            "acceptance.maximum_xy_centroid_error_m",
        ),
    )
    if acceptance.minimum_visible_fraction_per_endpoint > 1.0:
        raise ValueError("minimum visible fraction cannot exceed one")
    if (
        acceptance.minimum_valid_depth_m
        >= acceptance.maximum_valid_depth_m
    ):
        raise ValueError("depth interval must be increasing")

    output_doc = _mapping(document["output"], "output")
    _exact_keys(output_doc, RgbdOutput.__dataclass_fields__, "output")
    output = RgbdOutput(
        directory=_repository_path(
            output_doc["directory"], "output.directory"
        ),
        rgb_filename=_filename(
            output_doc["rgb_filename"], "output.rgb_filename", ".png"
        ),
        depth_preview_filename=_filename(
            output_doc["depth_preview_filename"],
            "output.depth_preview_filename",
            ".png",
        ),
        depth_numpy_filename=_filename(
            output_doc["depth_numpy_filename"],
            "output.depth_numpy_filename",
            ".npy",
        ),
        semantic_preview_filename=_filename(
            output_doc["semantic_preview_filename"],
            "output.semantic_preview_filename",
            ".png",
        ),
        report_filename=_filename(
            output_doc["report_filename"],
            "output.report_filename",
            ".json",
        ),
    )

    return D38999RgbdBootstrap(
        schema_version=RGBD_BOOTSTRAP_SCHEMA_VERSION,
        tabletop_config=_repository_path(
            document["tabletop_config"], "tabletop_config"
        ),
        pose_contract_config=_repository_path(
            document["pose_contract_config"], "pose_contract_config"
        ),
        camera=camera,
        labels=labels,
        position_estimator=position_estimator,
        acceptance=acceptance,
        output=output,
    )


def semantic_ids_for_label(
    id_to_labels: Mapping[Any, Any], taxonomy: str, label: str
) -> tuple[int, ...]:
    """Resolve renderer semantic IDs without depending on key JSON types."""
    taxonomy = _text(taxonomy, "taxonomy")
    label = _text(label, "label")
    matches = []
    for raw_id, raw_labels in _mapping(
        id_to_labels, "id_to_labels"
    ).items():
        labels = _mapping(raw_labels, f"id_to_labels[{raw_id!r}]")
        if labels.get(taxonomy) == label:
            if isinstance(raw_id, bool):
                raise ValueError("semantic IDs cannot be boolean")
            try:
                semantic_id = int(raw_id)
            except (TypeError, ValueError) as error:
                raise ValueError("semantic ID must be integer-like") from error
            if semantic_id < 0:
                raise ValueError("semantic ID cannot be negative")
            matches.append(semantic_id)
    if not matches:
        raise ValueError(f"semantic label not present: {label!r}")
    return tuple(sorted(set(matches)))


def summarize_mask_depth(
    semantic_ids: np.ndarray,
    depth_m: np.ndarray,
    selected_ids: tuple[int, ...],
) -> tuple[MaskDepthStatistics, np.ndarray]:
    """Validate one mask/depth pair and return finite endpoint statistics."""
    semantic = np.asarray(semantic_ids)
    depth = np.asarray(depth_m, dtype=np.float64)
    if semantic.ndim != 2 or depth.shape != semantic.shape:
        raise ValueError("semantic IDs and depth must be equal 2D images")
    if not selected_ids:
        raise ValueError("selected_ids cannot be empty")
    if any(
        isinstance(value, bool) or int(value) < 0
        for value in selected_ids
    ):
        raise ValueError("selected_ids must be non-negative integers")
    mask = np.isin(semantic, np.asarray(selected_ids, dtype=semantic.dtype))
    pixel_count = int(np.count_nonzero(mask))
    if pixel_count == 0:
        raise ValueError("endpoint semantic mask is empty")
    selected_depth = depth[mask]
    finite_depth = selected_depth[np.isfinite(selected_depth)]
    if finite_depth.size == 0 or np.any(finite_depth <= 0.0):
        raise ValueError("endpoint mask has no valid positive depth")
    return (
        MaskDepthStatistics(
            pixel_count=pixel_count,
            visible_fraction=pixel_count / float(mask.size),
            valid_depth_count=int(finite_depth.size),
            minimum_depth_m=float(np.min(finite_depth)),
            median_depth_m=float(np.median(finite_depth)),
            maximum_depth_m=float(np.max(finite_depth)),
        ),
        mask,
    )


def robust_world_xy_centroid(world_points: np.ndarray) -> tuple[float, float]:
    """Return the median world XY location of finite masked depth points."""
    points = np.asarray(world_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError("world_points must be a non-empty Nx3 array")
    if not np.all(np.isfinite(points)):
        raise ValueError("world_points must be finite")
    median = np.median(points[:, :2], axis=0)
    return (float(median[0]), float(median[1]))


def robust_semantic_mask_center_uv(
    semantic_mask: np.ndarray,
) -> tuple[float, float]:
    """Return the coordinate-wise median pixel center as ``(u, v)``.

    The semantic silhouette can contain an occluded edge or a small detached
    raster island.  A coordinate-wise median keeps a 50 percent breakdown
    point, unlike an arithmetic centroid, while remaining deterministic for
    even pixel counts.  The caller must provide the exact boolean endpoint
    mask; coercing arbitrary numeric arrays here could silently turn NaN into
    a selected pixel.
    """
    mask = np.asarray(semantic_mask)
    if mask.ndim != 2 or mask.dtype.kind != "b":
        raise ValueError("semantic_mask must be a 2D boolean array")
    rows, columns = np.nonzero(mask)
    if rows.size == 0:
        raise ValueError("semantic_mask cannot be empty")
    center = np.median(
        np.column_stack((columns, rows)).astype(np.float64), axis=0
    )
    if not np.all(np.isfinite(center)):
        raise ValueError("semantic-mask center must be finite")
    return (float(center[0]), float(center[1]))


def intersect_camera_ray_with_horizontal_plane(
    camera_origin_world_m: np.ndarray,
    point_on_ray_world_m: np.ndarray,
    registered_model_height_m: float,
    *,
    minimum_abs_ray_z: float = 1.0e-9,
) -> tuple[float, float, float]:
    """Intersect a finite camera ray with a registered world-Z plane."""

    def finite_vector3(value, label):
        vector = np.asarray(value, dtype=np.float64)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"{label} must be a finite three-vector")
        return vector

    origin = finite_vector3(camera_origin_world_m, "camera_origin_world_m")
    ray_point = finite_vector3(point_on_ray_world_m, "point_on_ray_world_m")
    height = _finite(
        registered_model_height_m, "registered_model_height_m"
    )
    parallel_tolerance = _positive(
        minimum_abs_ray_z, "minimum_abs_ray_z"
    )
    direction = ray_point - origin
    if float(np.linalg.norm(direction)) <= parallel_tolerance:
        raise ValueError("camera ray must have nonzero length")
    if abs(float(direction[2])) <= parallel_tolerance:
        raise ValueError("camera ray is parallel to registered height plane")
    scale = (height - float(origin[2])) / float(direction[2])
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("registered height plane must lie in front of camera")
    intersection = origin + scale * direction
    if not np.all(np.isfinite(intersection)):
        raise ValueError("ray-plane intersection must be finite")
    intersection[2] = height
    return tuple(float(value) for value in intersection)
