"""Fail-closed TE J35 full-pose provider boundary for ordinary RGB-D.

The provider consumes only the sanitized ``kcg_te_rgbd_provider_input_v1``
manifest, its ordinary RGB/depth/static-depth files, frozen camera calibration,
frozen workspaces, known static support geometry, and hash-bound official TE
STL meshes.  It never opens a capture report, USD stage, Prim, semantic image,
or simulator pose.

For the current historical Rx180 tabletop observation the earliest decisive
test is key visibility.  The asymmetric axial radius profile chooses the
supplier +Z sign.  The five-key yaw is eligible for scoring only if the narrow
key band that is not hidden by the coupling-nut front edge has measured depth
support.  Missing support returns ``MISS_KEY_NOT_OBSERVABLE`` with all formal
pose fields null; rear-contact or nut features are never substituted for the
wide main key.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Any, Mapping

import numpy as np
from PIL import Image

from kcg_connector.te_rgbd_observability import backproject_depth_to_world


INPUT_SCHEMA_VERSION = "kcg_te_rgbd_provider_input_v1"
RESULT_SCHEMA_VERSION = "kcg_te_rgbd_pose_provider_result_v1"
PROVIDER_SCOPE_PAIR = "PLUG_AND_RECEPTACLE_FULL_POSE"
PROVIDER_SCOPE_TRANSPORT_PLUG_ONLY = "TRANSPORT_PLUG_ONLY"
MISS_KEY_NOT_OBSERVABLE = "MISS_KEY_NOT_OBSERVABLE"
MISS_AXIS_SIGN_AMBIGUOUS = "MISS_AXIS_SIGN_AMBIGUOUS"
MISS_TRANSPORT_RING_NOT_OBSERVABLE = "MISS_TRANSPORT_RING_NOT_OBSERVABLE"
MISS_TRANSPORT_REAR_FACE_RESEARCH_BOUND = (
    "MISS_TRANSPORT_REAR_FACE_RESEARCH_BOUND"
)

_EXPECTED_STL_BOUNDS_M = {
    "te_deutsch_d38999_26fj35pn": (
        (-0.023458918124, -0.023458918124, -0.0310134),
        (0.023458918124, 0.023458918124, 0.0),
    ),
    "te_deutsch_d38999_20fj35sn": (
        (-0.0230124, -0.0230124, -0.031623),
        (0.0230124, 0.0230124, 0.0000000001),
    ),
}

# Official supplier plug geometry in the supplier mating-face frame.
_PLUG_KEY_CENTERS_DEG = (10.0, 90.0, 157.0, 254.0, 308.0)
_PLUG_KEY_WIDTHS_DEG = (
    4.02158,
    7.73810,
    4.02158,
    4.02158,
    4.02158,
)
_PLUG_MAIN_KEY_INDEX = 1
_PLUG_KEY_RADIUS_M = 0.0188214
_PLUG_KEY_LOCAL_Z_INTERVAL_M = (-0.0076454, -0.0007620)
# The nut/front outer envelope covers the key belt rearward of this plane.
_PLUG_EXPOSED_KEY_LOCAL_Z_INTERVAL_M = (-0.0015494, -0.0007620)
_MINIMUM_KEY_PATCH_PIXELS = 12
_MINIMUM_AXIS_PROFILE_IMPROVEMENT = 0.10
_KEY_DEPTH_TOLERANCE_M = 0.001
_YAW_COARSE_STEP_DEG = 0.5
_YAW_FINE_STEP_DEG = 0.02
# Rx180 transport exposes this coupling-nut ring above the table.  The limits
# are supplier-CAD local z values frozen before the translated validation
# capture; scene pose truth is neither needed nor accepted by the provider.
_RX180_TRANSPORT_RING_LOCAL_Z_INTERVAL_M = (-0.0055, -0.0015)
_MINIMUM_RX180_TRANSPORT_RING_POINTS = 1000
_RX180_REAR_FACE_CANDIDATE_DEPTH_M = 0.0012
_RX180_REAR_FACE_CONSENSUS_DISTANCE_M = 0.000025
_MINIMUM_RX180_REAR_FACE_CANDIDATE_POINTS = 1000
_MINIMUM_RX180_REAR_FACE_CONSENSUS_POINTS = 500
_MINIMUM_RX180_REAR_FACE_QUADRANT_POINTS = 100
_MINIMUM_RX180_REAR_FACE_CONSENSUS_FRACTION = 0.60
_MAXIMUM_RX180_REAR_FACE_P99_RESIDUAL_M = 0.000020
_MAXIMUM_RX180_REAR_FACE_ITERATIONS = 8
_RX180_LATERAL_RESEARCH_BOUND_M = 0.000183
_RX180_SUPPORT_Z_RESEARCH_BOUND_M = 0.000471
_RX180_AXIS_TILT_RESEARCH_BOUND_RAD = 0.0112
_RX180_RESEARCH_BOUND_UNIQUE_DEPTH_EVIDENCE = (
    {
        "role": "same_reset_tabletop",
        "provider_input_sha256": (
            "4bbeee5caad6f7d6d7fed5c39360294c188c0e8aa3fdaaa52acc82ab5d50afab"
        ),
        "ordinary_depth_sha256": (
            "e929046a2314a309ba3a02a699eefd777ae7a299226a65068abd609c5f014585"
        ),
    },
    {
        "role": "independent_xy_translated_tabletop",
        "provider_input_sha256": (
            "82b45bae2f54d50f1888dc99753d3496b1759e35ed1c31a7f3ca3e2e075ba022"
        ),
        "ordinary_depth_sha256": (
            "7a812328e3aa8e7910beb2813446cde3b9b601e5c3a35c83c3eb48dc6ddaa6f6"
        ),
    },
    {
        "role": "run08_second_observation_tabletop",
        "provider_input_sha256": (
            "51f6f6870134a733f5f53c6caa33d241471aa290a41f41dd65d77f3b39c29004"
        ),
        "ordinary_depth_sha256": (
            "7b34d3cf5d2171d53a4148301ec1c78ce16a3aeb35fd05bdae722930eeb23fca"
        ),
    },
)


@dataclass(frozen=True)
class BinaryStl:
    faces_m: np.ndarray
    face_normals: np.ndarray
    bounds_min_m: np.ndarray
    bounds_max_m: np.ndarray
    triangle_count: int


@dataclass(frozen=True)
class ProviderInputs:
    repository: Path
    manifest_path: Path
    manifest: Mapping[str, Any]
    rgb: np.ndarray
    depth_m: np.ndarray
    static_depth_m: np.ndarray
    intrinsics: np.ndarray
    world_from_camera: np.ndarray
    cad: Mapping[str, BinaryStl]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _resolve_repository_file(
    repository: Path, value: Any, expected_sha256: Any, label: str
) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} path must be text")
    raw = Path(value)
    path = raw.resolve() if raw.is_absolute() else (repository / raw).resolve()
    try:
        path.relative_to(repository)
    except ValueError as error:
        raise ValueError(f"{label} escapes the repository") from error
    if not path.is_file():
        raise ValueError(f"{label} file is missing")
    if not isinstance(expected_sha256, str) or _sha256(path) != expected_sha256:
        raise ValueError(f"{label} SHA-256 differs")
    return path


def _resolve_manifest_artifact(
    manifest_path: Path, record: Any, label: str
) -> Path:
    document = _mapping(record, label)
    relative = Path(document["path_relative_to_manifest"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} path must be local to the manifest")
    path = (manifest_path.parent / relative).resolve()
    if not path.is_file() or _sha256(path) != document.get("sha256"):
        raise ValueError(f"{label} artifact identity differs")
    return path


def _forbidden_provider_key(value: Any) -> str | None:
    forbidden = {
        "scene_generation_truth",
        "scene_generation_truth_postcapture_record",
        "pose_prim_path",
        "reference_prim_path",
        "object_prim_path",
        "object_pose_truth",
    }
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in forbidden:
                return str(key)
            nested = _forbidden_provider_key(item)
            if nested is not None:
                return nested
    elif isinstance(value, list):
        for item in value:
            nested = _forbidden_provider_key(item)
            if nested is not None:
                return nested
    return None


def load_binary_stl_mm(path: Path, *, model_id: str) -> BinaryStl:
    """Load one binary supplier STL and convert its numeric millimetres to m."""
    size = path.stat().st_size
    with path.open("rb") as stream:
        header = stream.read(80)
        count_bytes = stream.read(4)
        if len(header) != 80 or len(count_bytes) != 4:
            raise ValueError("STL header is truncated")
        triangle_count = struct.unpack("<I", count_bytes)[0]
    expected_size = 84 + 50 * triangle_count
    if size != expected_size or triangle_count < 1:
        raise ValueError("official TE STL is not the expected binary layout")
    dtype = np.dtype(
        [
            ("normal", "<f4", (3,)),
            ("vertices", "<f4", (3, 3)),
            ("attribute", "<u2"),
        ]
    )
    records = np.memmap(path, dtype=dtype, mode="r", offset=84, shape=(triangle_count,))
    faces_m = np.asarray(records["vertices"], dtype=np.float64) * 0.001
    face_normals = np.asarray(records["normal"], dtype=np.float64)
    bounds_min = np.min(faces_m.reshape(-1, 3), axis=0)
    bounds_max = np.max(faces_m.reshape(-1, 3), axis=0)
    expected = _EXPECTED_STL_BOUNDS_M.get(model_id)
    if expected is None or not np.allclose(
        bounds_min, expected[0], rtol=0.0, atol=2.0e-9
    ) or not np.allclose(bounds_max, expected[1], rtol=0.0, atol=2.0e-9):
        raise ValueError(f"{model_id} STL bounds differ from supplier audit")
    return BinaryStl(
        faces_m=faces_m,
        face_normals=face_normals,
        bounds_min_m=bounds_min,
        bounds_max_m=bounds_max,
        triangle_count=int(triangle_count),
    )


def load_provider_inputs(
    provider_input_path: Path | str,
    repository_root: Path | str,
) -> ProviderInputs:
    repository = Path(repository_root).expanduser().resolve()
    manifest_path = Path(provider_input_path).expanduser().resolve()
    try:
        manifest_path.relative_to(repository)
    except ValueError as error:
        raise ValueError("provider input escapes the repository") from error
    manifest = _mapping(
        json.loads(manifest_path.read_text(encoding="utf-8")),
        "provider_input",
    )
    if manifest.get("schema_version") != INPUT_SCHEMA_VERSION:
        raise ValueError("unsupported TE provider input schema")
    leaked_key = _forbidden_provider_key(manifest)
    if leaked_key is not None:
        raise ValueError(f"provider input contains forbidden key {leaked_key}")
    firewall = _mapping(manifest.get("truth_firewall"), "truth_firewall")
    forbidden_inputs = set(firewall.get("forbidden_provider_inputs", ()))
    if not {
        "semantic_segmentation_truth",
        "instance_segmentation_truth",
        "object_pose_truth",
        "object_prim_path",
    }.issubset(forbidden_inputs):
        raise ValueError("provider truth firewall is incomplete")

    rgb_path = _resolve_manifest_artifact(
        manifest_path, manifest["ordinary_rgb"], "ordinary_rgb"
    )
    depth_path = _resolve_manifest_artifact(
        manifest_path, manifest["ordinary_depth"], "ordinary_depth"
    )
    static_depth_path = _resolve_manifest_artifact(
        manifest_path,
        manifest["ordinary_static_scene_depth"],
        "ordinary_static_scene_depth",
    )
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
    depth = np.asarray(np.load(depth_path, allow_pickle=False), dtype=np.float64)
    static_depth = np.asarray(
        np.load(static_depth_path, allow_pickle=False), dtype=np.float64
    )
    if rgb.shape[:2] != depth.shape or static_depth.shape != depth.shape:
        raise ValueError("ordinary RGB/static/observed depth dimensions differ")
    calibration = _mapping(
        manifest.get("camera_calibration"), "camera_calibration"
    )
    intrinsics = np.asarray(calibration["intrinsics_3x3"], dtype=np.float64)
    world_from_camera = np.asarray(
        calibration["world_from_camera_cv_row_major"], dtype=np.float64
    ).reshape(4, 4)
    if intrinsics.shape != (3, 3) or not np.all(np.isfinite(intrinsics)):
        raise ValueError("camera intrinsics are invalid")
    if not np.all(np.isfinite(world_from_camera)):
        raise ValueError("world_from_camera is invalid")

    provider_scope = manifest.get("provider_scope", PROVIDER_SCOPE_PAIR)
    if provider_scope not in {
        PROVIDER_SCOPE_PAIR,
        PROVIDER_SCOPE_TRANSPORT_PLUG_ONLY,
    }:
        raise ValueError("unsupported TE provider scope")
    endpoints = (
        ("plug", "receptacle")
        if provider_scope == PROVIDER_SCOPE_PAIR
        else ("plug",)
    )
    cad_models = _mapping(manifest.get("te_cad_models"), "te_cad_models")
    if set(cad_models) != set(endpoints):
        raise ValueError("TE CAD endpoints differ from provider scope")
    cad: dict[str, BinaryStl] = {}
    for endpoint in endpoints:
        model = _mapping(cad_models[endpoint], f"te_cad_models.{endpoint}")
        path = _resolve_repository_file(
            repository,
            model["registration_cad"],
            model["registration_cad_sha256"],
            f"{endpoint} registration CAD",
        )
        cad[endpoint] = load_binary_stl_mm(path, model_id=model["model_id"])
    return ProviderInputs(
        repository=repository,
        manifest_path=manifest_path,
        manifest=manifest,
        rgb=rgb,
        depth_m=depth,
        static_depth_m=static_depth,
        intrinsics=intrinsics,
        world_from_camera=world_from_camera,
        cad=cad,
    )


def _foreground_points(inputs: ProviderInputs) -> dict[str, dict[str, np.ndarray]]:
    threshold = float(
        inputs.manifest["observability"]["minimum_foreground_depth_delta_m"]
    )
    observed_valid = np.isfinite(inputs.depth_m) & (inputs.depth_m > 0.0)
    static_valid = np.isfinite(inputs.static_depth_m) & (
        inputs.static_depth_m > 0.0
    )
    depth_delta = np.full(inputs.depth_m.shape, -np.inf, dtype=np.float64)
    shared_valid = observed_valid & static_valid
    depth_delta[shared_valid] = (
        inputs.static_depth_m[shared_valid] - inputs.depth_m[shared_valid]
    )
    foreground = observed_valid & (
        (~static_valid)
        | (depth_delta >= threshold)
    )
    points, pixel_v, pixel_u = backproject_depth_to_world(
        np.where(foreground, inputs.depth_m, np.nan),
        inputs.intrinsics,
        inputs.world_from_camera,
    )
    result = {}
    workspaces = inputs.manifest["frozen_endpoint_workspaces_world_aabb_m"]
    if set(workspaces) != set(inputs.cad):
        raise ValueError("provider workspaces differ from loaded CAD endpoints")
    for endpoint in inputs.cad:
        bounds = workspaces[endpoint]
        lower = np.asarray(bounds["minimum"], dtype=np.float64)
        upper = np.asarray(bounds["maximum"], dtype=np.float64)
        selected = np.all((points >= lower) & (points <= upper), axis=1)
        result[endpoint] = {
            "points_world_m": points[selected],
            "pixel_v": pixel_v[selected],
            "pixel_u": pixel_u[selected],
        }
    return result


def _surface_samples(faces: np.ndarray, count: int = 180000) -> np.ndarray:
    first, second, third = faces[:, 0], faces[:, 1], faces[:, 2]
    area = 0.5 * np.linalg.norm(np.cross(second - first, third - first), axis=1)
    valid = np.isfinite(area) & (area > 0.0)
    if int(np.sum(valid)) < 1:
        raise ValueError("official STL has no positive-area triangles")
    faces = faces[valid]
    probability = area[valid] / float(np.sum(area[valid]))
    rng = np.random.default_rng(38999)
    indices = rng.choice(len(faces), size=count, replace=True, p=probability)
    selected = faces[indices]
    first_random = rng.random(count)
    second_random = rng.random(count)
    reflected = first_random + second_random > 1.0
    first_random[reflected] = 1.0 - first_random[reflected]
    second_random[reflected] = 1.0 - second_random[reflected]
    return (
        selected[:, 0]
        + first_random[:, None] * (selected[:, 1] - selected[:, 0])
        + second_random[:, None] * (selected[:, 2] - selected[:, 0])
    )


def _radial_profile(
    points: np.ndarray,
    axial_height: np.ndarray,
    length_m: float,
    *,
    bin_count: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(0.0, length_m, bin_count + 1)
    center_xy = 0.5 * (np.min(points[:, :2], axis=0) + np.max(points[:, :2], axis=0))
    radius = np.linalg.norm(points[:, :2] - center_xy, axis=1)
    values = np.full(bin_count, np.nan, dtype=np.float64)
    counts = np.zeros(bin_count, dtype=np.int64)
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        selected = (axial_height >= lower) & (axial_height < upper)
        counts[index] = int(np.sum(selected))
        if counts[index] >= 20:
            values[index] = float(np.quantile(radius[selected], 0.98))
    finite = np.flatnonzero(np.isfinite(values))
    if len(finite) < 6:
        raise ValueError("axial profile has insufficient populated bins")
    missing = np.flatnonzero(~np.isfinite(values))
    values[missing] = np.interp(missing, finite, values[finite])
    return values, counts


def _axis_sign_from_profile(
    observed_world: np.ndarray,
    cad: BinaryStl,
    support_z_m: float,
) -> dict[str, Any]:
    length = float(-cad.bounds_min_m[2])
    observed_height = observed_world[:, 2] - support_z_m
    observed_profile, observed_counts = _radial_profile(
        observed_world, observed_height, length
    )
    samples = _surface_samples(cad.faces_m)
    negative_profile, _ = _radial_profile(samples, -samples[:, 2], length)
    positive_profile, _ = _radial_profile(
        samples, samples[:, 2] + length, length
    )
    populated = observed_counts >= 100
    negative_rmse = float(
        np.sqrt(np.mean((observed_profile[populated] - negative_profile[populated]) ** 2))
    )
    positive_rmse = float(
        np.sqrt(np.mean((observed_profile[populated] - positive_profile[populated]) ** 2))
    )
    if negative_rmse <= positive_rmse:
        sign = -1
        best, alternative = negative_rmse, positive_rmse
    else:
        sign = 1
        best, alternative = positive_rmse, negative_rmse
    improvement = (alternative - best) / max(alternative, 1.0e-12)
    return {
        "method": "KNOWN_SUPPORT_NORMAL_PLUS_OFFICIAL_CAD_ASYMMETRIC_RADIAL_PROFILE_V1",
        "supplier_plus_z_world_sign": sign,
        "supplier_outward_axis_world": [0.0, 0.0, float(sign)],
        "negative_sign_profile_rmse_m": negative_rmse,
        "positive_sign_profile_rmse_m": positive_rmse,
        "relative_best_profile_improvement": improvement,
        "minimum_required_relative_improvement": _MINIMUM_AXIS_PROFILE_IMPROVEMENT,
        "sign_unique": bool(improvement >= _MINIMUM_AXIS_PROFILE_IMPROVEMENT),
        "observed_profile_bin_count": int(np.sum(populated)),
    }


def _null_pose_endpoint(
    status: str,
    diagnostics: Mapping[str, Any],
    *,
    key_field: str,
) -> dict[str, Any]:
    result = {
        "status": status,
        "position_xyz_m": None,
        "quaternion_xyzw": None,
        "insertion_axis_world": None,
        "uncertainty_6d_3sigma": [None, None, None, None, None, None],
        "confidence": 0.0,
        "occluded": True,
        "key_observed": False,
        "control_allowed": False,
        "diagnostics": dict(diagnostics),
    }
    result[key_field] = None
    return result


def _rx180_transport_ring_center(
    plug_points: np.ndarray,
    support_z_m: float,
    *,
    median_optical_depth_m: float,
    focal_length_px: float,
) -> dict[str, Any]:
    local_z = -(plug_points[:, 2] - support_z_m)
    lower, upper = _RX180_TRANSPORT_RING_LOCAL_Z_INTERVAL_M
    selected = (local_z >= lower) & (local_z <= upper)
    points_xy = plug_points[selected, :2]
    diagnostics: dict[str, Any] = {
        "method": "TE_RX180_CAD_AXIAL_RING_CIRCLE_LEAST_SQUARES_V1",
        "supplier_local_z_interval_m": [lower, upper],
        "supplier_axis_sign_required": -1,
        "point_count": int(len(points_xy)),
        "minimum_point_count": _MINIMUM_RX180_TRANSPORT_RING_POINTS,
        "band_selected_without_object_pose_truth": True,
        "simulation_research_bound_frozen": True,
        "formal_3sigma_or_camera_certification": False,
    }
    if len(points_xy) < _MINIMUM_RX180_TRANSPORT_RING_POINTS:
        diagnostics.update(
            {
                "fit_valid": False,
                "failure_reason": "INSUFFICIENT_CAD_AXIAL_RING_POINTS",
                "center_xy_m": None,
                "candidate_research_bound_per_lateral_component_m": None,
            }
        )
        return diagnostics

    x_value, y_value = points_xy[:, 0], points_xy[:, 1]
    design = np.column_stack(
        (2.0 * x_value, 2.0 * y_value, np.ones(len(points_xy)))
    )
    target = x_value * x_value + y_value * y_value
    solution, _, rank, singular_values = np.linalg.lstsq(
        design, target, rcond=None
    )
    center_xy = solution[:2]
    radius_squared = float(solution[2] + center_xy @ center_xy)
    if (
        int(rank) != 3
        or not np.all(np.isfinite(center_xy))
        or not np.isfinite(radius_squared)
        or radius_squared <= 0.0
    ):
        diagnostics.update(
            {
                "fit_valid": False,
                "failure_reason": "DEGENERATE_RING_CIRCLE_FIT",
                "center_xy_m": None,
                "design_rank": int(rank),
                "design_singular_values": singular_values.tolist(),
                "candidate_research_bound_per_lateral_component_m": None,
            }
        )
        return diagnostics

    radius = math.sqrt(radius_squared)
    signed_residual = np.linalg.norm(points_xy - center_xy, axis=1) - radius
    residual_median = float(np.median(signed_residual))
    residual_absolute = np.abs(signed_residual)
    residual_mad = float(
        np.median(np.abs(signed_residual - residual_median))
    )
    residual_p99 = float(np.quantile(residual_absolute, 0.99))
    pixel_footprint = float(median_optical_depth_m / focal_length_px)
    candidate_bound = pixel_footprint + residual_p99
    if not np.isfinite(residual_median) or not all(
        np.isfinite(value) and value >= 0.0
        for value in (
            radius,
            residual_mad,
            residual_p99,
            pixel_footprint,
            candidate_bound,
        )
    ):
        raise ValueError("Rx180 transport ring diagnostics are not finite")
    diagnostics.update(
        {
            "fit_valid": True,
            "failure_reason": None,
            "center_xy_m": center_xy.tolist(),
            "radius_m": radius,
            "design_rank": int(rank),
            "design_singular_values": singular_values.tolist(),
            "signed_radial_residual_median_m": residual_median,
            "signed_radial_residual_mad_m": residual_mad,
            "absolute_radial_residual_p99_m": residual_p99,
            "median_optical_depth_m": float(median_optical_depth_m),
            "focal_length_px": float(focal_length_px),
            "one_pixel_footprint_at_median_depth_m": pixel_footprint,
            "candidate_research_bound_method": (
                "ONE_PIXEL_FOOTPRINT_PLUS_ABSOLUTE_RING_RESIDUAL_P99"
            ),
            "candidate_research_bound_per_lateral_component_m": (
                candidate_bound
            ),
            "candidate_research_bound_status": (
                "DIAGNOSTIC_ONLY_NOT_THE_FROZEN_LATERAL_ERROR_BOUND"
            ),
            "candidate_research_bound_scope": (
                "SIMULATION_RESEARCH_ONLY_NOT_3SIGMA_NOT_CAMERA_OR_"
                "MANUFACTURING_CERTIFICATION"
            ),
            "candidate_research_bound_excludes": [
                "camera_calibration_systematic_error",
                "physical_settle_height_and_tilt",
                "robot_motion_error",
                "independent_translated_sample_error",
            ],
        }
    )
    return diagnostics


def _transport_grasp_pose(
    plug_points: np.ndarray,
    cad: BinaryStl,
    support_z_m: float,
    axis_sign: Mapping[str, Any],
    *,
    median_optical_depth_m: float,
    focal_length_px: float,
) -> dict[str, Any]:
    """Keep the observable axis pose separate from keyed assembly yaw.

    The coupling nut is axisymmetric for this limited transport purpose.  No
    quaternion is emitted because yaw is not observed; callers must not reuse
    this record for receptacle contact or insertion.
    """
    sign_unique = axis_sign.get("sign_unique") is True
    sign = int(axis_sign["supplier_plus_z_world_sign"])
    ring = (
        _rx180_transport_ring_center(
            plug_points,
            support_z_m,
            median_optical_depth_m=median_optical_depth_m,
            focal_length_px=focal_length_px,
        )
        if sign_unique and sign == -1
        else None
    )
    ring_valid = ring is not None and ring.get("fit_valid") is True
    rear_face_guard: dict[str, Any] | None = None
    if sign_unique and sign == -1 and ring_valid:
        try:
            rear_face_guard = _rx180_rear_face_research_guard(
                plug_points,
                cad,
                support_z_m,
                ring,
            )
        except Exception as error:
            rear_face_guard = {
                "method": "TE_RX180_REAR_FACE_CURRENT_CAPTURE_RESEARCH_GUARD_V1",
                "guard_pass": False,
                "failure_reasons": ["REAR_FACE_FIT_FAILED"],
                "error_type": type(error).__name__,
                "error": str(error),
                "ordinary_rgbd_only": True,
                "repeated_frame_statistics_used": False,
                "object_pose_truth_used": False,
            }
    rear_face_valid = (
        rear_face_guard is not None
        and rear_face_guard.get("guard_pass") is True
    )
    center_xy = (
        np.asarray(ring["center_xy_m"], dtype=np.float64)
        if ring_valid
        else 0.5
        * (
            np.min(plug_points[:, :2], axis=0)
            + np.max(plug_points[:, :2], axis=0)
        )
    )
    length = float(-cad.bounds_min_m[2])
    origin_z = support_z_m if sign == -1 else support_z_m + length
    outward_axis_world = list(axis_sign["supplier_outward_axis_world"])
    transport_observed = bool(
        sign_unique and (sign != -1 or (ring_valid and rear_face_valid))
    )
    status = "OBSERVED_AXIS_POSITION_YAW_FREE"
    if not transport_observed:
        if not sign_unique:
            status = MISS_AXIS_SIGN_AMBIGUOUS
        elif sign == -1 and not ring_valid:
            status = MISS_TRANSPORT_RING_NOT_OBSERVABLE
        elif sign == -1:
            status = MISS_TRANSPORT_REAR_FACE_RESEARCH_BOUND
    return {
        "status": status,
        "target_part": "CouplingNut",
        "pose_reference": "plug_supplier_mating_face_axis",
        "position_xyz_m": [
            float(center_xy[0]),
            float(center_xy[1]),
            float(origin_z),
        ] if transport_observed else None,
        "quaternion_xyzw": None,
        "outward_axis_world": (
            outward_axis_world
            if transport_observed else None
        ),
        "yaw_status": "UNOBSERVED_FREE_FOR_AXISYMMETRIC_NUT_TRANSPORT_ONLY",
        "main_key_required": False,
        "uncertainty_6d_3sigma": [None, None, None, None, None, None],
        "uncertainty_6d_research_bound": (
            [
                _RX180_LATERAL_RESEARCH_BOUND_M,
                _RX180_LATERAL_RESEARCH_BOUND_M,
                _RX180_SUPPORT_Z_RESEARCH_BOUND_M,
                _RX180_AXIS_TILT_RESEARCH_BOUND_RAD,
                _RX180_AXIS_TILT_RESEARCH_BOUND_RAD,
                math.pi,
            ]
            if transport_observed and sign == -1
            else [None, None, None, None, None, None]
        ),
        "uncertainty_6d_research_bound_scope": (
            "FINITE_SIMULATION_RESEARCH_BOUND_NOT_3SIGMA_NOT_CAMERA_OR_"
            "MANUFACTURING_CERTIFICATION"
        ),
        "uncertainty_6d_research_bound_semantics": {
            "component_order": ["x", "y", "z", "roll", "pitch", "yaw"],
            "translation_components_are_absolute_bounds": True,
            "roll_pitch_values_encode_one_combined_axis_tilt_cone": True,
            "roll_pitch_are_not_an_independent_box": True,
            "yaw_is_unobserved_full_range": True,
        },
        "confidence": (
            float(axis_sign["relative_best_profile_improvement"])
            if transport_observed else 0.0
        ),
        "confidence_scope": "AXIS_SIGN_PROFILE_SEPARATION_ONLY_NOT_CALIBRATED_6D",
        "transport_grasp_planning_input_available": transport_observed,
        "transport_grasp_control_allowed": False,
        "receptacle_contact_allowed": False,
        "insertion_allowed": False,
        "locking_allowed": False,
        "in_plane_branch_observed": False,
        "frozen_main_key_propagation_allowed": False,
        "coupling_nut_reference_forbidden": True,
        "robot_command_count": 0,
        "derivation": {
            "lateral_center": (
                "te_rx180_cad_axial_ring_circle_least_squares_v1"
                if sign == -1
                else "foreground_world_xy_aabb_midpoint_non_rx180_diagnostic_only"
            ),
            "axial_origin": "known_static_support_plus_signed_supplier_axial_extent",
            "axis_sign": dict(axis_sign),
            "rx180_axisymmetric_ring": ring,
            "rx180_rear_face_current_capture_guard": rear_face_guard,
            "rear_face_position_or_axis_consumed_by_transport_target": False,
            "transport_target_axis_source": "KNOWN_STATIC_SUPPORT_NORMAL",
            "transport_target_z_source": "KNOWN_STATIC_SUPPORT_GEOMETRY",
            "rear_face_guard_role": (
                "CURRENT_CAPTURE_Z_AND_TILT_RESEARCH_ENVELOPE_ONLY"
            ),
        },
    }


def _fit_identity_rear_face_pose(
    plug_points: np.ndarray,
    cad: BinaryStl,
    support_z_m: float,
) -> dict[str, Any]:
    """Fit a visible PlugBody end plane without inferring keyed yaw."""
    origin_z = support_z_m - float(cad.bounds_min_m[2])
    face = plug_points[
        (plug_points[:, 2] >= origin_z - 0.0005)
        & (plug_points[:, 2] <= origin_z + 0.0003)
    ]
    if len(face) < 500:
        raise ValueError("rear face has insufficient depth points")
    initial_center = 0.5 * (
        np.min(plug_points[:, :2], axis=0)
        + np.max(plug_points[:, :2], axis=0)
    )
    initial_radius = np.linalg.norm(face[:, :2] - initial_center, axis=1)
    ring = face[(initial_radius >= 0.015) & (initial_radius <= 0.021)]
    if len(ring) < 300:
        raise ValueError("rear-face ring is not visible")
    x_value, y_value = ring[:, 0], ring[:, 1]
    design = np.column_stack((2.0 * x_value, 2.0 * y_value, np.ones(len(ring))))
    target = x_value * x_value + y_value * y_value
    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    center_xy = solution[:2]
    fitted_radius = math.sqrt(max(0.0, float(solution[2] + center_xy @ center_xy)))
    radial_residual = np.linalg.norm(ring[:, :2] - center_xy, axis=1) - fitted_radius

    plane_centroid = np.mean(face, axis=0)
    centered = face - plane_centroid
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(centered, rowvar=False))
    axis = eigenvectors[:, int(np.argmin(eigenvalues))]
    if axis[2] < 0.0:
        axis = -axis
    axis /= np.linalg.norm(axis)
    tilt = math.acos(float(np.clip(axis[2], -1.0, 1.0)))
    plane_residual = centered @ axis
    return {
        "origin_world_m": [float(center_xy[0]), float(center_xy[1]), origin_z],
        "axis_world": axis.tolist(),
        "face_point_count": int(len(face)),
        "ring_point_count": int(len(ring)),
        "fitted_ring_radius_m": fitted_radius,
        "ring_radial_rmse_m": float(np.sqrt(np.mean(radial_residual ** 2))),
        "plane_smallest_eigenvalue_m2": float(np.min(eigenvalues)),
        "plane_centroid_world_m": plane_centroid.tolist(),
        "plane_absolute_residual_p99_m": float(
            np.quantile(np.abs(plane_residual), 0.99)
        ),
        "axis_tilt_from_support_normal_rad": tilt,
    }


def _rx180_rear_face_research_guard(
    plug_points: np.ndarray,
    cad: BinaryStl,
    support_z_m: float,
    ring: Mapping[str, Any],
) -> dict[str, Any]:
    """Fit the dominant visible Rx180 rear plane and enforce its scope.

    This is a finite simulation research gate, not a statistical confidence
    interval.  The plane supplies position and insertion-axis tilt, but the
    axisymmetric surface deliberately supplies no keyed-yaw branch.
    """
    center_xy = np.asarray(ring["center_xy_m"], dtype=np.float64)
    pixel_footprint = float(ring["one_pixel_footprint_at_median_depth_m"])
    vertices = cad.faces_m.reshape(-1, 3)
    rear_vertices = vertices[
        np.abs(vertices[:, 2] - float(cad.bounds_min_m[2])) <= 1.0e-9
    ]
    if len(rear_vertices) < 3:
        raise ValueError("official CAD rear-face vertices are unavailable")
    rear_face_radius = float(
        np.max(np.linalg.norm(rear_vertices[:, :2], axis=1))
    )
    if (
        center_xy.shape != (2,)
        or not np.all(np.isfinite(center_xy))
        or not math.isfinite(pixel_footprint)
        or pixel_footprint <= 0.0
        or not math.isfinite(rear_face_radius)
        or rear_face_radius <= 0.0
    ):
        raise ValueError("Rx180 rear-face CAD/image diagnostics are not finite")

    radial_distance = np.linalg.norm(
        plug_points[:, :2] - center_xy, axis=1
    )
    radial_selected = radial_distance <= rear_face_radius + pixel_footprint
    if int(np.sum(radial_selected)) < _MINIMUM_RX180_REAR_FACE_CANDIDATE_POINTS:
        raise ValueError("Rx180 rear-face radial support is insufficient")
    highest_z = float(np.max(plug_points[radial_selected, 2]))
    candidate = plug_points[
        radial_selected
        & (plug_points[:, 2] >= highest_z - _RX180_REAR_FACE_CANDIDATE_DEPTH_M)
    ]
    if len(candidate) < _MINIMUM_RX180_REAR_FACE_CANDIDATE_POINTS:
        raise ValueError("Rx180 rear-face plane candidates are insufficient")

    consensus = candidate[candidate[:, 2] >= np.median(candidate[:, 2])]
    previous_mask: np.ndarray | None = None
    converged = False
    iteration_count = 0
    for iteration_count in range(1, _MAXIMUM_RX180_REAR_FACE_ITERATIONS + 1):
        if len(consensus) < 3:
            break
        centroid = np.mean(consensus, axis=0)
        eigenvalues, eigenvectors = np.linalg.eigh(
            np.cov(consensus - centroid, rowvar=False)
        )
        if not np.all(np.isfinite(eigenvalues)):
            raise ValueError("Rx180 rear-face eigenvalues are not finite")
        axis = eigenvectors[:, int(np.argmin(eigenvalues))]
        if axis[2] < 0.0:
            axis = -axis
        axis_norm = float(np.linalg.norm(axis))
        if not math.isfinite(axis_norm) or axis_norm <= 1.0e-12:
            raise ValueError("Rx180 rear-face plane normal is degenerate")
        axis /= axis_norm
        mask = (
            np.abs((candidate - centroid) @ axis)
            <= _RX180_REAR_FACE_CONSENSUS_DISTANCE_M
        )
        if previous_mask is not None and np.array_equal(mask, previous_mask):
            converged = True
            consensus = candidate[mask]
            break
        previous_mask = mask
        consensus = candidate[mask]

    if len(consensus) < 3:
        raise ValueError("Rx180 rear-face plane consensus is insufficient")
    centroid = np.mean(consensus, axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(
        np.cov(consensus - centroid, rowvar=False)
    )
    if not np.all(np.isfinite(eigenvalues)):
        raise ValueError("Rx180 rear-face final eigenvalues are not finite")
    axis = eigenvectors[:, int(np.argmin(eigenvalues))]
    if axis[2] < 0.0:
        axis = -axis
    axis_norm = float(np.linalg.norm(axis))
    if not math.isfinite(axis_norm) or axis_norm <= 1.0e-12:
        raise ValueError("Rx180 rear-face final plane normal is degenerate")
    axis /= axis_norm
    if (
        not np.all(np.isfinite(centroid))
        or not np.all(np.isfinite(axis))
        or abs(float(axis[2])) < 1.0e-9
    ):
        raise ValueError("Rx180 rear-face plane normal is not finite")
    plane_residual = (consensus - centroid) @ axis
    residual_rmse = float(np.sqrt(np.mean(plane_residual ** 2)))
    residual_p99 = float(np.quantile(np.abs(plane_residual), 0.99))
    measured_tilt = math.acos(float(np.clip(axis[2], -1.0, 1.0)))

    rear_plane_z_at_ring_center = float(
        centroid[2]
        - (
            axis[0] * (center_xy[0] - centroid[0])
            + axis[1] * (center_xy[1] - centroid[1])
        )
        / axis[2]
    )
    supplier_axial_length = float(-cad.bounds_min_m[2])
    inferred_mating_face_z = float(
        rear_plane_z_at_ring_center - supplier_axial_length * axis[2]
    )
    support_z_residual = float(inferred_mating_face_z - support_z_m)
    axial_research_margin = float(
        abs(support_z_residual)
        + 2.0 * pixel_footprint
        + 2.0 * residual_p99
    )
    tilt_margin = float(
        measured_tilt + math.atan(pixel_footprint / rear_face_radius)
    )
    if not all(
        math.isfinite(value)
        for value in (
            rear_plane_z_at_ring_center,
            inferred_mating_face_z,
            support_z_residual,
            residual_rmse,
            residual_p99,
            measured_tilt,
            axial_research_margin,
            tilt_margin,
        )
    ):
        raise ValueError("Rx180 rear-face derived diagnostics are not finite")
    relative_xy = consensus[:, :2] - center_xy
    quadrant_counts = [
        int(np.sum((relative_xy[:, 0] >= 0.0) & (relative_xy[:, 1] >= 0.0))),
        int(np.sum((relative_xy[:, 0] >= 0.0) & (relative_xy[:, 1] < 0.0))),
        int(np.sum((relative_xy[:, 0] < 0.0) & (relative_xy[:, 1] >= 0.0))),
        int(np.sum((relative_xy[:, 0] < 0.0) & (relative_xy[:, 1] < 0.0))),
    ]
    consensus_fraction = float(len(consensus) / len(candidate))
    gates = {
        "deterministic_consensus_converged": converged,
        "minimum_candidate_point_count": (
            len(candidate) >= _MINIMUM_RX180_REAR_FACE_CANDIDATE_POINTS
        ),
        "minimum_consensus_point_count": (
            len(consensus) >= _MINIMUM_RX180_REAR_FACE_CONSENSUS_POINTS
        ),
        "dominant_consensus_fraction": (
            consensus_fraction > _MINIMUM_RX180_REAR_FACE_CONSENSUS_FRACTION
        ),
        "all_four_quadrants_supported": (
            min(quadrant_counts) >= _MINIMUM_RX180_REAR_FACE_QUADRANT_POINTS
        ),
        "plane_p99_residual_inside_bound": (
            residual_p99 <= _MAXIMUM_RX180_REAR_FACE_P99_RESIDUAL_M
        ),
        "axial_research_margin_inside_support_z_bound": (
            axial_research_margin <= _RX180_SUPPORT_Z_RESEARCH_BOUND_M
        ),
        "tilt_margin_inside_axis_tilt_bound": (
            tilt_margin <= _RX180_AXIS_TILT_RESEARCH_BOUND_RAD
        ),
    }
    failure_reasons = [name for name, passed in gates.items() if not passed]
    return {
        "method": "TE_RX180_REAR_FACE_CURRENT_CAPTURE_RESEARCH_GUARD_V1",
        "guard_pass": not failure_reasons,
        "failure_reasons": failure_reasons,
        "ordinary_rgbd_only": True,
        "repeated_frame_statistics_used": False,
        "object_pose_truth_used": False,
        "candidate_point_count": int(len(candidate)),
        "consensus_point_count": int(len(consensus)),
        "consensus_fraction": consensus_fraction,
        "quadrant_point_counts": quadrant_counts,
        "minimum_candidate_point_count": (
            _MINIMUM_RX180_REAR_FACE_CANDIDATE_POINTS
        ),
        "minimum_consensus_point_count": (
            _MINIMUM_RX180_REAR_FACE_CONSENSUS_POINTS
        ),
        "minimum_quadrant_point_count": (
            _MINIMUM_RX180_REAR_FACE_QUADRANT_POINTS
        ),
        "minimum_consensus_fraction": (
            _MINIMUM_RX180_REAR_FACE_CONSENSUS_FRACTION
        ),
        "candidate_depth_below_highest_m": (
            _RX180_REAR_FACE_CANDIDATE_DEPTH_M
        ),
        "consensus_distance_m": _RX180_REAR_FACE_CONSENSUS_DISTANCE_M,
        "iteration_count": iteration_count,
        "converged": converged,
        "official_cad_rear_face_radius_m": rear_face_radius,
        "one_pixel_footprint_at_median_depth_m": pixel_footprint,
        "rear_face_plane_centroid_world_m": centroid.tolist(),
        "rear_face_plane_normal_world": axis.tolist(),
        "supplier_outward_axis_world": (-axis).tolist(),
        "rear_plane_z_at_transport_ring_center_m": rear_plane_z_at_ring_center,
        "inferred_mating_face_z_m": inferred_mating_face_z,
        "known_static_support_z_m": float(support_z_m),
        "inferred_mating_face_support_z_residual_m": support_z_residual,
        "plane_residual_rmse_m": residual_rmse,
        "plane_absolute_residual_p99_m": residual_p99,
        "maximum_plane_absolute_residual_p99_m": (
            _MAXIMUM_RX180_REAR_FACE_P99_RESIDUAL_M
        ),
        "axial_research_margin_method": (
            "ABS_SUPPORT_RESIDUAL_PLUS_TWO_PIXEL_FOOTPRINTS_PLUS_TWO_"
            "PLANE_P99_RESIDUALS"
        ),
        "axial_research_margin_m": axial_research_margin,
        "measured_axis_tilt_from_support_normal_rad": measured_tilt,
        "tilt_research_margin_method": (
            "MEASURED_TILT_PLUS_ATAN_ONE_PIXEL_OVER_CAD_REAR_RADIUS"
        ),
        "tilt_research_margin_rad": tilt_margin,
        "frozen_research_bounds": {
            "lateral_per_component_m": _RX180_LATERAL_RESEARCH_BOUND_M,
            "support_z_absolute_m": _RX180_SUPPORT_Z_RESEARCH_BOUND_M,
            "axis_tilt_cone_rad": _RX180_AXIS_TILT_RESEARCH_BOUND_RAD,
            "yaw_range_rad": [-math.pi, math.pi],
            "scope": (
                "FINITE_SIMULATION_RESEARCH_BOUND_NOT_3SIGMA_NOT_CAMERA_OR_"
                "MANUFACTURING_CERTIFICATION"
            ),
            "unique_ordinary_depth_evidence": list(
                _RX180_RESEARCH_BOUND_UNIQUE_DEPTH_EVIDENCE
            ),
            "unique_depth_count": len(
                _RX180_RESEARCH_BOUND_UNIQUE_DEPTH_EVIDENCE
            ),
            "repeated_reobservations_counted_as_independent": False,
        },
        "lateral_research_bound_basis": (
            "FROZEN_SEALED_TWO_POSITION_POSTHOC_ERROR_PANEL_NOT_CURRENT_"
            "RING_PIXEL_HEURISTIC"
        ),
        "ring_candidate_research_bound_is_formal_uncertainty": False,
        "gates": gates,
        "in_plane_branch_observed": False,
        "frozen_main_key_propagation_allowed": False,
        "secondary_plane_evaluation": "NOT_SEPARATELY_EVALUATED",
        "dominant_plane_basis": (
            "PRIMARY_CONSENSUS_STRICTLY_EXCEEDS_SIXTY_PERCENT_OF_CANDIDATES"
        ),
    }


def _plug_key_patch_faces(cad: BinaryStl) -> tuple[np.ndarray, ...]:
    centroid = np.mean(cad.faces_m, axis=1)
    normal = np.asarray(cad.face_normals, dtype=np.float64)
    normal_xy_norm = np.linalg.norm(normal[:, :2], axis=1)
    unit_normal_xy = normal[:, :2] / np.maximum(normal_xy_norm[:, None], 1.0e-12)
    radial_plane = np.sum(centroid[:, :2] * unit_normal_xy, axis=1)
    base = (
        (centroid[:, 2] >= _PLUG_KEY_LOCAL_Z_INTERVAL_M[0] - 1.0e-8)
        & (centroid[:, 2] <= _PLUG_KEY_LOCAL_Z_INTERVAL_M[1] + 1.0e-8)
        & (np.abs(normal[:, 2]) < 1.0e-4)
        & (normal_xy_norm > 0.8)
        & (np.abs(radial_plane - _PLUG_KEY_RADIUS_M) < 0.000012)
    )
    angle = np.degrees(np.arctan2(centroid[:, 1], centroid[:, 0])) % 360.0
    patches = []
    for center, width in zip(_PLUG_KEY_CENTERS_DEG, _PLUG_KEY_WIDTHS_DEG):
        delta = np.abs((angle - center + 180.0) % 360.0 - 180.0)
        selected = base & (delta <= 0.5 * width + 1.0e-4)
        if int(np.sum(selected)) != 6:
            raise ValueError("official plug key patch triangle label changed")
        patches.append(np.flatnonzero(selected))
    return tuple(patches)


def _triangle_samples(triangles: np.ndarray, subdivisions: int = 18) -> np.ndarray:
    rows = []
    for first_index in range(subdivisions + 1):
        for second_index in range(subdivisions + 1 - first_index):
            first = first_index / float(subdivisions)
            second = second_index / float(subdivisions)
            rows.append((first, second, 1.0 - first - second))
    weights = np.asarray(rows, dtype=np.float64)
    return np.einsum("sa,tac->tsc", weights, triangles).reshape(-1, 3)


def _rotation_from_axis_and_yaw(axis_world: np.ndarray, yaw_rad: float) -> np.ndarray:
    axis = np.asarray(axis_world, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    reference = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
    reference -= float(reference @ axis) * axis
    if np.linalg.norm(reference) < 1.0e-9:
        reference = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
        reference -= float(reference @ axis) * axis
    x_axis = reference / np.linalg.norm(reference)
    y_axis = np.cross(axis, x_axis)
    base = np.column_stack((x_axis, y_axis, axis))
    cosine, sine = math.cos(yaw_rad), math.sin(yaw_rad)
    local_yaw = np.asarray(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    return base @ local_yaw


def _quaternion_xyzw(rotation: np.ndarray) -> list[float]:
    from scipy.spatial.transform import Rotation

    quaternion = Rotation.from_matrix(rotation).as_quat()
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    return quaternion.tolist()


def _foreground_mask(inputs: ProviderInputs) -> np.ndarray:
    threshold = float(
        inputs.manifest["observability"]["minimum_foreground_depth_delta_m"]
    )
    observed_valid = np.isfinite(inputs.depth_m) & (inputs.depth_m > 0.0)
    static_valid = np.isfinite(inputs.static_depth_m) & (
        inputs.static_depth_m > 0.0
    )
    delta = np.full(inputs.depth_m.shape, -np.inf, dtype=np.float64)
    shared = observed_valid & static_valid
    delta[shared] = inputs.static_depth_m[shared] - inputs.depth_m[shared]
    return observed_valid & ((~static_valid) | (delta >= threshold))


def _score_key_patch(
    inputs: ProviderInputs,
    local_samples: np.ndarray,
    local_normal: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    foreground: np.ndarray,
) -> dict[str, Any]:
    world = local_samples @ rotation.T + translation
    camera_from_world = np.linalg.inv(inputs.world_from_camera)
    camera = (
        world @ camera_from_world[:3, :3].T
        + camera_from_world[:3, 3]
    )
    depth = camera[:, 2]
    fx, fy = inputs.intrinsics[0, 0], inputs.intrinsics[1, 1]
    cx, cy = inputs.intrinsics[0, 2], inputs.intrinsics[1, 2]
    u_value = np.rint(fx * camera[:, 0] / depth + cx).astype(np.int64)
    v_value = np.rint(fy * camera[:, 1] / depth + cy).astype(np.int64)
    height, width = inputs.depth_m.shape
    inside = (
        (depth > 0.0)
        & (u_value >= 0)
        & (u_value < width)
        & (v_value >= 0)
        & (v_value < height)
    )
    camera_position = inputs.world_from_camera[:3, 3]
    normal_world = rotation @ local_normal
    view = camera_position - translation
    front_facing = float(normal_world @ view) > 0.0
    per_pixel: dict[tuple[int, int], float] = {}
    for pixel_v, pixel_u, predicted_depth in zip(
        v_value[inside], u_value[inside], depth[inside]
    ):
        key = (int(pixel_v), int(pixel_u))
        previous = per_pixel.get(key)
        if previous is None or predicted_depth < previous:
            per_pixel[key] = float(predicted_depth)
    residuals = []
    consistent = 0
    occluded = 0
    foreground_pixels = 0
    for (pixel_v, pixel_u), predicted_depth in per_pixel.items():
        observed = float(inputs.depth_m[pixel_v, pixel_u])
        if foreground[pixel_v, pixel_u]:
            foreground_pixels += 1
        if not math.isfinite(observed) or observed <= 0.0:
            continue
        residual = observed - predicted_depth
        residuals.append(residual)
        if foreground[pixel_v, pixel_u] and abs(residual) <= _KEY_DEPTH_TOLERANCE_M:
            consistent += 1
        elif observed < predicted_depth - _KEY_DEPTH_TOLERANCE_M:
            occluded += 1
    residual_array = np.asarray(residuals, dtype=np.float64)
    return {
        "front_facing": bool(front_facing),
        "projected_pixel_count": int(len(per_pixel)),
        "foreground_projected_pixel_count": foreground_pixels,
        "depth_consistent_pixel_count": int(consistent if front_facing else 0),
        "occluded_pixel_count": int(occluded),
        "depth_residual_rmse_m": (
            float(np.sqrt(np.mean(residual_array ** 2)))
            if len(residual_array) else None
        ),
        "depth_residual_median_m": (
            float(np.median(residual_array)) if len(residual_array) else None
        ),
    }


def _positive_sign_key_registration(
    inputs: ProviderInputs,
    plug_points: np.ndarray,
    support_z_m: float,
    axis_sign: Mapping[str, Any],
) -> dict[str, Any]:
    pose = _fit_identity_rear_face_pose(
        plug_points, inputs.cad["plug"], support_z_m
    )
    translation = np.asarray(pose["origin_world_m"], dtype=np.float64)
    axis = np.asarray(pose["axis_world"], dtype=np.float64)
    patch_faces = _plug_key_patch_faces(inputs.cad["plug"])
    patch_samples = tuple(
        _triangle_samples(inputs.cad["plug"].faces_m[indices])
        for indices in patch_faces
    )
    patch_normals = tuple(
        np.mean(inputs.cad["plug"].face_normals[indices], axis=0)
        for indices in patch_faces
    )
    foreground = _foreground_mask(inputs)

    def main_support(yaw_rad: float) -> tuple[int, float]:
        rotation = _rotation_from_axis_and_yaw(axis, yaw_rad)
        score = _score_key_patch(
            inputs,
            patch_samples[_PLUG_MAIN_KEY_INDEX],
            patch_normals[_PLUG_MAIN_KEY_INDEX],
            rotation,
            translation,
            foreground,
        )
        residual = score["depth_residual_rmse_m"]
        return score["depth_consistent_pixel_count"], (
            float(residual) if residual is not None else math.inf
        )

    coarse_yaws = np.radians(
        np.arange(0.0, 360.0, _YAW_COARSE_STEP_DEG, dtype=np.float64)
    )
    coarse_rows = [(*main_support(yaw), yaw) for yaw in coarse_yaws]
    coarse_rows.sort(key=lambda row: (-row[0], row[1]))
    coarse_best = float(coarse_rows[0][2])
    fine_offsets = np.radians(
        np.arange(
            -_YAW_COARSE_STEP_DEG,
            _YAW_COARSE_STEP_DEG + 0.5 * _YAW_FINE_STEP_DEG,
            _YAW_FINE_STEP_DEG,
        )
    )
    fine_rows = [
        (*main_support((coarse_best + offset) % (2.0 * math.pi)),
         (coarse_best + offset) % (2.0 * math.pi))
        for offset in fine_offsets
    ]
    fine_rows.sort(key=lambda row: (-row[0], row[1]))
    observed_wide_feature_yaw = float(fine_rows[0][2])

    candidates = []
    main_center = _PLUG_KEY_CENTERS_DEG[_PLUG_MAIN_KEY_INDEX]
    for correspondence, key_center in enumerate(_PLUG_KEY_CENTERS_DEG):
        yaw = (
            observed_wide_feature_yaw
            + math.radians(main_center - key_center)
        ) % (2.0 * math.pi)
        rotation = _rotation_from_axis_and_yaw(axis, yaw)
        patches = [
            _score_key_patch(
                inputs,
                samples,
                normal,
                rotation,
                translation,
                foreground,
            )
            for samples, normal in zip(patch_samples, patch_normals)
        ]
        required_support = [
            patches[index]["depth_consistent_pixel_count"]
            for index in (1, 0, 2)
        ]
        viable = all(value >= _MINIMUM_KEY_PATCH_PIXELS for value in required_support)
        candidates.append(
            {
                "cyclic_correspondence": correspondence,
                "observed_wide_feature_assigned_supplier_key_index": correspondence,
                "yaw_rad": yaw,
                "yaw_deg": math.degrees(yaw),
                "patches": {
                    f"K{int(center)}" + ("_main" if index == 1 else ""): row
                    for index, (center, row) in enumerate(
                        zip(_PLUG_KEY_CENTERS_DEG, patches)
                    )
                },
                "required_support_pixels_K90_K10_K157": required_support,
                "total_required_support_pixels": int(sum(required_support)),
                "viable": bool(viable),
            }
        )
    viable = [row for row in candidates if row["viable"]]
    viable.sort(key=lambda row: -row["total_required_support_pixels"])
    unique = len(viable) == 1
    selected = viable[0] if unique else None
    return {
        "method": "POSITIVE_SIGN_REAR_FACE_FIT_PLUS_FIVE_KEY_DEPTH_PATCH_V1",
        "rear_face_pose": pose,
        "coarse_yaw_step_deg": _YAW_COARSE_STEP_DEG,
        "fine_yaw_step_deg": _YAW_FINE_STEP_DEG,
        "observed_wide_feature_yaw_rad": observed_wide_feature_yaw,
        "minimum_required_patch_pixels": _MINIMUM_KEY_PATCH_PIXELS,
        "five_discrete_yaw_candidates": candidates,
        "viable_candidate_count": len(viable),
        "unique_branch": unique,
        "selected_candidate": selected,
    }


def _positive_full_pose_records(
    inputs: ProviderInputs,
    registration: Mapping[str, Any],
    axis_sign: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    selected = registration.get("selected_candidate")
    if registration.get("unique_branch") is not True or not isinstance(
        selected, Mapping
    ):
        raise ValueError("positive-sign five-key branch is not unique")
    rear = registration["rear_face_pose"]
    position = np.asarray(rear["origin_world_m"], dtype=np.float64)
    axis = np.asarray(rear["axis_world"], dtype=np.float64)
    yaw = float(selected["yaw_rad"])
    rotation = _rotation_from_axis_and_yaw(axis, yaw)
    quaternion = _quaternion_xyzw(rotation)
    main_key = rotation @ np.asarray((0.0, 1.0, 0.0), dtype=np.float64)

    plug_observability = next(
        row for row in inputs.manifest["observability"]["endpoints"]
        if row["endpoint"] == "plug"
    )
    optical_depth = float(plug_observability["median_optical_depth_m"])
    focal_px = float(min(inputs.intrinsics[0, 0], inputs.intrinsics[1, 1]))
    lateral_3sigma = 3.0 * float(rear["ring_radial_rmse_m"])
    axial_3sigma = 3.0 * math.sqrt(
        max(0.0, float(rear["plane_smallest_eigenvalue_m2"]))
    )
    tilt_3sigma = max(
        0.001,
        3.0 * float(rear["axis_tilt_from_support_normal_rad"]),
    )
    yaw_3sigma = 3.0 * math.atan2(
        optical_depth / focal_px, _PLUG_KEY_RADIUS_M
    )
    uncertainty = [
        lateral_3sigma,
        lateral_3sigma,
        axial_3sigma,
        tilt_3sigma,
        tilt_3sigma,
        yaw_3sigma,
    ]
    required_support = selected["required_support_pixels_K90_K10_K157"]
    confidence = min(
        float(axis_sign["relative_best_profile_improvement"]),
        min(1.0, min(required_support) / float(_MINIMUM_KEY_PATCH_PIXELS)),
    )
    status = "PLUG_FULL_KEYED_POSE_OBSERVED_NO_CONTROL"
    pose_record = {
        "status": status,
        "position_xyz_m": position.tolist(),
        "quaternion_xyzw": quaternion,
        "insertion_axis_world": axis.tolist(),
        "unique_main_key_world": main_key.tolist(),
        "uncertainty_6d_3sigma": uncertainty,
        "uncertainty_scope": (
            "SIM_RESEARCH_RING_PLANE_AND_ONE_PIXEL_KEY_RESOLUTION_"
            "NOT_HARDWARE_CALIBRATION"
        ),
        "confidence": confidence,
        "occluded": False,
        "occlusion_note": "far-side K254/K308 excluded by backface/occlusion",
        "key_observed": True,
        "wide_main_key_depth_support_pixels": int(required_support[0]),
        "control_allowed": False,
        "receptacle_contact_allowed": False,
        "diagnostics": {
            "axis_sign": dict(axis_sign),
            "positive_sign_key_registration": dict(registration),
        },
    }

    relation = inputs.manifest["te_cad_models"]["plug"].get(
        "body_rear_face_to_main_key_relation", {}
    )
    rear_origin = position + rotation @ np.asarray(
        (0.0, 0.0, inputs.cad["plug"].bounds_min_m[2]), dtype=np.float64
    )
    freeze_candidate = {
        "schema_version": "kcg_te_body_rear_face_to_main_key_freeze_candidate_v1",
        "status": "CANDIDATE_READY_TO_FREEZE",
        "capture_id": inputs.manifest["capture_id"],
        "relation_id": relation.get(
            "relation_id", "te_body_rear_face_to_main_key_v1"
        ),
        "relation_contract_sha256": relation.get("contract_sha256"),
        "reference_part": "PlugBody",
        "forbidden_reference_part": "CouplingNut",
        "observed_supplier_main_key_index": _PLUG_MAIN_KEY_INDEX,
        "observed_branch_id": "N_POLARIZATION_SUPPLIER_MAIN_KEY_INDEX_1",
        "observed_unwrapped_yaw_rad": yaw,
        "body_rear_face_pose_world": {
            "position_xyz_m": rear_origin.tolist(),
            "quaternion_xyzw": quaternion,
        },
        "main_key_direction_world": main_key.tolist(),
        "direct_key_evidence": {
            "K90_main_depth_consistent_pixels": int(required_support[0]),
            "K10_depth_consistent_pixels": int(required_support[1]),
            "K157_depth_consistent_pixels": int(required_support[2]),
            "unique_viable_branch_count": int(
                registration["viable_candidate_count"]
            ),
        },
        "continuity_policy": {
            "track_only_current_body_rear_face_pose_after_freeze": True,
            "relabel_after_slip_or_disturbance_allowed": False,
            "short_occlusion_with_continuous_branch_requires_relabel": False,
            "recalibration_status_after_complete_branch_loss": (
                "RECALIBRATION_REQUIRED_AFTER_BRANCH_LOSS"
            ),
        },
        "control_allowed": False,
    }
    freeze_candidate["candidate_sha256"] = _mapping_sha256(freeze_candidate)
    assembly_record = dict(pose_record)
    assembly_record.update(
        {
            "reference_part": "PlugBody",
            "relation_freeze_candidate": freeze_candidate,
        }
    )
    return pose_record, assembly_record, freeze_candidate


def estimate_te_pose_pair(inputs: ProviderInputs) -> dict[str, Any]:
    endpoint_points = _foreground_points(inputs)
    plug_points = endpoint_points["plug"]["points_world_m"]
    provider_scope = inputs.manifest.get("provider_scope", PROVIDER_SCOPE_PAIR)
    plug_only = provider_scope == PROVIDER_SCOPE_TRANSPORT_PLUG_ONLY
    receptacle_points = (
        np.empty((0, 3), dtype=np.float64)
        if plug_only
        else endpoint_points["receptacle"]["points_world_m"]
    )
    if len(plug_points) < 150 or (not plug_only and len(receptacle_points) < 150):
        raise ValueError("endpoint foreground point count is below frozen minimum")
    static_geometry = inputs.manifest["known_static_scene_geometry"]
    table = static_geometry["table"]
    table_top = float(table["center_world_m"][2]) + 0.5 * float(table["size_m"][2])
    axis_sign = _axis_sign_from_profile(
        plug_points, inputs.cad["plug"], table_top
    )
    plug_observability = next(
        row
        for row in inputs.manifest["observability"]["endpoints"]
        if row["endpoint"] == "plug"
    )
    transport_pose = _transport_grasp_pose(
        plug_points,
        inputs.cad["plug"],
        table_top,
        axis_sign,
        median_optical_depth_m=float(
            plug_observability["median_optical_depth_m"]
        ),
        focal_length_px=float(
            min(inputs.intrinsics[0, 0], inputs.intrinsics[1, 1])
        ),
    )
    full_pose_record = None
    assembly_pose_record = None
    relation_freeze_candidate = None
    if plug_only:
        status = transport_pose["status"]
        key_visibility = {
            "evaluated": False,
            "reason": "transport plug-only scope does not observe or require main-key yaw",
        }
    elif axis_sign["sign_unique"] is not True:
        status = MISS_AXIS_SIGN_AMBIGUOUS
        key_visibility = {
            "evaluated": False,
            "reason": "axis sign was not unique",
        }
    elif axis_sign["supplier_plus_z_world_sign"] == 1:
        key_visibility = _positive_sign_key_registration(
            inputs, plug_points, table_top, axis_sign
        )
        if key_visibility["unique_branch"] is True:
            status = "PLUG_FULL_KEYED_POSE_OBSERVED_NO_CONTROL"
            (
                full_pose_record,
                assembly_pose_record,
                relation_freeze_candidate,
            ) = _positive_full_pose_records(
                inputs, key_visibility, axis_sign
            )
        else:
            status = "MISS_KEY_AMBIGUOUS"
    else:
        world_band = (
            table_top - _PLUG_EXPOSED_KEY_LOCAL_Z_INTERVAL_M[1],
            table_top - _PLUG_EXPOSED_KEY_LOCAL_Z_INTERVAL_M[0],
        )
        in_band = (
            (plug_points[:, 2] >= world_band[0])
            & (plug_points[:, 2] <= world_band[1])
        )
        band_count = int(np.sum(in_band))
        band_pixels = int(
            len(
                set(
                    zip(
                        endpoint_points["plug"]["pixel_v"][in_band].tolist(),
                        endpoint_points["plug"]["pixel_u"][in_band].tolist(),
                    )
                )
            )
        )
        key_visibility = {
            "method": "OFFICIAL_KEY_BELT_MINUS_COUPLING_NUT_OCCLUSION_DEPTH_SUPPORT_V1",
            "full_key_local_z_interval_m": list(_PLUG_KEY_LOCAL_Z_INTERVAL_M),
            "exposed_key_local_z_interval_m": list(
                _PLUG_EXPOSED_KEY_LOCAL_Z_INTERVAL_M
            ),
            "exposed_key_world_z_interval_m": list(world_band),
            "observed_plug_minimum_world_z_m": float(np.min(plug_points[:, 2])),
            "observed_plug_maximum_world_z_m": float(np.max(plug_points[:, 2])),
            "minimum_observed_gap_above_exposed_band_m": float(
                np.min(plug_points[:, 2]) - world_band[1]
            ),
            "depth_support_point_count": band_count,
            "depth_support_pixel_count": band_pixels,
            "minimum_required_patch_pixels": _MINIMUM_KEY_PATCH_PIXELS,
            "wide_main_key_actually_observed": bool(
                band_pixels >= _MINIMUM_KEY_PATCH_PIXELS
            ),
            "key_centers_supplier_deg": list(_PLUG_KEY_CENTERS_DEG),
            "key_widths_deg": list(_PLUG_KEY_WIDTHS_DEG),
            "main_key_index": _PLUG_MAIN_KEY_INDEX,
            "main_key_radius_m": _PLUG_KEY_RADIUS_M,
            "five_discrete_yaw_candidates": [
                {
                    "cyclic_correspondence": index,
                    "status": (
                        "NOT_SCORABLE_KEY_PATCH_UNOBSERVED"
                        if band_pixels < _MINIMUM_KEY_PATCH_PIXELS
                        else "PENDING_FULL_VISIBLE_PATCH_SCORE"
                    ),
                    "score": None,
                }
                for index in range(5)
            ],
        }
        status = (
            "MISS_FULL_CAD_REGISTRATION_NOT_RUN"
            if key_visibility["wide_main_key_actually_observed"]
            else MISS_KEY_NOT_OBSERVABLE
        )

    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "capture_id": inputs.manifest["capture_id"],
        "provider_scope": provider_scope,
        "same_reset_evidence": inputs.manifest.get("same_reset_evidence"),
        "capture_time": inputs.manifest.get("capture_time"),
        "status": status,
        "model_identity": {
            endpoint: {
                "model_id": inputs.manifest["te_cad_models"][endpoint]["model_id"],
                "identity": inputs.manifest["te_cad_models"][endpoint]["identity"],
                "registration_cad_sha256": inputs.manifest["te_cad_models"][endpoint][
                    "registration_cad_sha256"
                ],
                "stl_unit_conversion": "supplier_mm_times_1e-3_to_m",
                "stl_bounds_m": {
                    "minimum": inputs.cad[endpoint].bounds_min_m.tolist(),
                    "maximum": inputs.cad[endpoint].bounds_max_m.tolist(),
                },
            }
            for endpoint in inputs.cad
        },
        "endpoints": {
            "plug": (
                full_pose_record
                if full_pose_record is not None
                else _null_pose_endpoint(
                    status,
                    {
                        "foreground_point_count": int(len(plug_points)),
                        "axis_sign": axis_sign,
                        "key_visibility": key_visibility,
                    },
                    key_field="unique_main_key_world",
                )
            ),
            "receptacle": _null_pose_endpoint(
                "NOT_PRESENT_IN_TRANSPORT_PLUG_ONLY_PHYSICAL_SCENE"
                if plug_only
                else "NOT_EVALUATED_BY_CURRENT_PLUG_ONLY_PROVIDER_SCOPE",
                {"foreground_point_count": int(len(receptacle_points))},
                key_field="unique_main_keyway_world",
            ),
        },
        "transport_grasp_pose": transport_pose,
        "assembly_key_pose": (
            assembly_pose_record
            if assembly_pose_record is not None
            else {
                "status": status,
                "reference_part": "PlugBody",
                "position_xyz_m": None,
                "quaternion_xyzw": None,
                "insertion_axis_world": None,
                "unique_main_key_world": None,
                "uncertainty_6d_3sigma": [None, None, None, None, None, None],
                "confidence": 0.0,
                "occluded": True,
                "key_observed": False,
                "control_allowed": False,
                "receptacle_contact_allowed": False,
                "branch_continuity": {
                    "relation_id": "te_body_rear_face_to_main_key_v1",
                    "relation_status": "NOT_BOUND_TO_CURRENT_PROVIDER_INPUT",
                    "reference_part": "PlugBody",
                    "coupling_nut_reference_forbidden": True,
                    "relabel_after_freeze_allowed": False,
                    "axisymmetric_face_supplies_in_plane_branch": False,
                    "current_project_relation_state_claimed": False,
                },
            }
        ),
        "relation_freeze_candidate": relation_freeze_candidate,
        "full_6d": False,
        "plug_full_6d": bool(full_pose_record is not None),
        "keyed_orientation_observed": bool(full_pose_record is not None),
        "control_allowed": False,
        "robot_command_count": 0,
        "truth_flags": {
            "uses_semantic_truth": False,
            "uses_instance_truth": False,
            "uses_object_pose_truth": False,
            "uses_prim_transform": False,
            "uses_contact_truth": False,
            "posthoc_truth_read_before_result": False,
        },
        "input_evidence": {
            "provider_input_sha256": _sha256(inputs.manifest_path),
            "ordinary_rgb_sha256": inputs.manifest["ordinary_rgb"]["sha256"],
            "ordinary_depth_sha256": inputs.manifest["ordinary_depth"]["sha256"],
            "ordinary_static_scene_depth_sha256": inputs.manifest[
                "ordinary_static_scene_depth"
            ]["sha256"],
            "camera_calibration_sha256": inputs.manifest["camera_calibration"][
                "calibration_sha256"
            ],
        },
    }
    json.dumps(result, allow_nan=False, sort_keys=True)
    return result


def run_te_rgbd_pose_provider(
    provider_input_path: Path | str,
    repository_root: Path | str,
) -> dict[str, Any]:
    return estimate_te_pose_pair(
        load_provider_inputs(provider_input_path, repository_root)
    )


__all__ = [
    "INPUT_SCHEMA_VERSION",
    "MISS_AXIS_SIGN_AMBIGUOUS",
    "MISS_KEY_NOT_OBSERVABLE",
    "RESULT_SCHEMA_VERSION",
    "estimate_te_pose_pair",
    "load_binary_stl_mm",
    "load_provider_inputs",
    "run_te_rgbd_pose_provider",
]
