"""Pure preparation contract for multi-position D38999 vision.

This module deliberately has no Isaac, ROS, OpenCV, TensorRT, or GPU imports.
It defines deterministic placement sampling, strict observability gates, and
candidate ``object_T_target`` composition without connecting any result to the
current end-to-end controller.  The checked-in proxy lacks a unique key, so a
geometric RGB-D estimate must fail the keyed-yaw and control gates.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from numbers import Integral, Real
from pathlib import Path
import random
import re
from typing import Any, Mapping, Sequence

import yaml


SCHEMA_VERSION = "kcg_d38999_multisite_vision6d_v1"
DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "d38999_multisite_vision6d_v1.yaml"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ClosedRange:
    """Finite inclusive scalar interval."""

    lower: float
    upper: float

    def contains(self, value: float) -> bool:
        return self.lower <= float(value) <= self.upper


@dataclass(frozen=True)
class EndpointPlacementBounds:
    """Tabletop XY/yaw sampling bounds for one endpoint root."""

    x_m: ClosedRange
    y_m: ClosedRange
    origin_z_m: float
    yaw_rad: ClosedRange
    footprint_radius_m: float
    support: str


@dataclass(frozen=True)
class PlacementPairSample:
    """One deterministic loose/fixed placement candidate."""

    seed: int
    loose_position_xyz_m: tuple[float, float, float]
    loose_yaw_rad: float
    fixed_position_xyz_m: tuple[float, float, float]
    fixed_yaw_rad: float
    center_separation_m: float


@dataclass(frozen=True)
class VisibilityGates:
    minimum_center_margin_px: int
    minimum_mask_pixels_per_endpoint: int
    minimum_visible_fraction_per_endpoint: float
    minimum_valid_depth_fraction_in_mask: float
    maximum_occlusion_fraction: float
    required_views_for_translation_axis: int
    required_views_for_key_yaw: int


@dataclass(frozen=True)
class GeometricPoseGates:
    maximum_translation_std_m: float
    maximum_cad_fit_rmse_m: float
    maximum_axis_error_rad: float
    minimum_axis_inlier_fraction: float
    minimum_unique_yaw_score_margin: float


@dataclass(frozen=True)
class TargetTransformCandidate:
    """Unqualified proxy-derived ``object_T_target`` candidate."""

    transform_id: str
    model_id: str
    kind: str
    parent_frame_id: str
    child_frame_id: str
    translation_xyz_m: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    provenance: str
    qualified: bool


@dataclass(frozen=True)
class EndpointPoseEvidence:
    """Algorithm-neutral evidence consumed by the strict pose gate."""

    mask_pixels: int
    visible_fraction: float
    valid_depth_fraction_in_mask: float
    center_margin_px: float
    occlusion_fraction: float
    calibrated_view_count: int
    translation_std_m: float
    cad_fit_rmse_m: float
    axis_error_rad: float
    axis_inlier_fraction: float
    key_feature_observed: bool
    yaw_hypothesis_count: int
    unique_yaw_score_margin: float


@dataclass(frozen=True)
class EndpointPoseGate:
    visibility_passed: bool
    translation_observed: bool
    axis_observed: bool
    keyed_yaw_observed: bool


@dataclass(frozen=True)
class PoseControlGateResult:
    """Pair-level result that explains why control remains unauthorized."""

    loose_plug: EndpointPoseGate
    fixed_receptacle: EndpointPoseGate
    full_6d: bool
    keyed_orientation_observed: bool
    all_target_transforms_qualified: bool
    control_authorized: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ResolvedTargetCandidate:
    """Composed ``world_T_target`` candidate with qualification preserved."""

    transform_id: str
    position_xyz_m: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    qualified: bool


@dataclass(frozen=True)
class D38999MultisiteVision6dContract:
    schema_version: str
    enabled: bool
    status: str
    delivery_window_hours: float
    achievable_in_window: tuple[str, ...]
    not_honestly_claimable_in_window: tuple[str, ...]
    input_paths: dict[str, Path]
    loose_plug: EndpointPlacementBounds
    fixed_receptacle: EndpointPlacementBounds
    minimum_center_separation_m: float
    table_xy_bounds_m: tuple[float, float, float, float]
    required_anchor_pairs: tuple[dict[str, Any], ...]
    visibility: VisibilityGates
    geometry: GeometricPoseGates
    global_camera_available: bool
    wrist_camera_available: bool
    current_proxy_has_unique_polarization_key: bool
    yaw_symmetry_order: int
    equivalent_yaw_period_rad: float
    periodic_component_orders: dict[str, tuple[int, ...]]
    coupling_nut_is_independent_yaw_body: bool
    target_transforms: tuple[TargetTransformCandidate, ...]
    pose_control_current_authorized: bool
    foundationpose_official_sources: dict[str, str]
    foundationpose_required_asset_paths: dict[str, str]
    foundationpose_blockers: tuple[str, ...]
    foundationpose_model_version: str
    foundationpose_model_license: str
    foundationpose_ros_wrapper_license: str
    foundationpose_reference_code_license: str
    boundaries: dict[str, bool]

    def transform(self, transform_id: str) -> TargetTransformCandidate:
        matches = [
            item
            for item in self.target_transforms
            if item.transform_id == transform_id
        ]
        if len(matches) != 1:
            raise ValueError(f"unknown target transform {transform_id!r}")
        return matches[0]


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _exact(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    actual = set(value)
    if actual != keys:
        raise ValueError(
            f"{label} keys differ; missing={sorted(keys - actual)}, "
            f"extra={sorted(actual - keys)}"
        )


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be non-empty unpadded text")
    return value


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean")
    return value


def _real(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _positive(value: Any, label: str) -> float:
    result = _real(value, label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _fraction(value: Any, label: str) -> float:
    result = _real(value, label)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be within [0, 1]")
    return result


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{label} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return result


def _vector(value: Any, size: int, label: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must contain {size} finite numbers")
    try:
        result = tuple(value)
    except TypeError as error:
        raise ValueError(
            f"{label} must contain {size} finite numbers"
        ) from error
    if len(result) != size:
        raise ValueError(f"{label} must contain {size} finite numbers")
    return tuple(
        _real(item, f"{label}[{index}]")
        for index, item in enumerate(result)
    )


def _closed_range(value: Any, label: str) -> ClosedRange:
    lower, upper = _vector(value, 2, label)
    if lower > upper:
        raise ValueError(f"{label} lower bound exceeds upper bound")
    return ClosedRange(lower=lower, upper=upper)


def _string_list(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    result = tuple(
        _text(item, f"{label}[{index}]")
        for index, item in enumerate(value)
    )
    if not result or len(result) != len(set(result)):
        raise ValueError(f"{label} must be non-empty and unique")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_input_files(
    value: Any, *, repository: Path
) -> dict[str, Path]:
    label = "inputs"
    document = _mapping(value, label)
    names = {
        "proxy_config",
        "tabletop_scene",
        "pose_contract",
        "proxy_asset",
    }
    _exact(document, names, label)
    result = {}
    for name in sorted(names):
        item_label = f"{label}.{name}"
        item = _mapping(document[name], item_label)
        _exact(item, {"path", "sha256"}, item_label)
        relative = Path(_text(item["path"], f"{item_label}.path"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"{item_label}.path must be repository-relative")
        expected_hash = _text(item["sha256"], f"{item_label}.sha256")
        if not _SHA256.fullmatch(expected_hash):
            raise ValueError(f"{item_label}.sha256 must be lowercase SHA-256")
        path = (repository / relative).resolve()
        if not path.is_file() or repository not in path.parents:
            raise ValueError(f"{item_label} is missing or outside repository")
        if _sha256(path) != expected_hash:
            raise ValueError(f"{item_label} SHA-256 mismatch")
        result[name] = path
    return result


def _parse_placement_bounds(value: Any, label: str) -> EndpointPlacementBounds:
    document = _mapping(value, label)
    _exact(
        document,
        {
            "x_m",
            "y_m",
            "origin_z_m",
            "yaw_rad",
            "footprint_radius_m",
            "support",
        },
        label,
    )
    yaw = _closed_range(document["yaw_rad"], f"{label}.yaw_rad")
    if yaw.lower < -math.pi or yaw.upper > math.pi:
        raise ValueError(f"{label}.yaw_rad must remain within [-pi, pi]")
    return EndpointPlacementBounds(
        x_m=_closed_range(document["x_m"], f"{label}.x_m"),
        y_m=_closed_range(document["y_m"], f"{label}.y_m"),
        origin_z_m=_real(document["origin_z_m"], f"{label}.origin_z_m"),
        yaw_rad=yaw,
        footprint_radius_m=_positive(
            document["footprint_radius_m"],
            f"{label}.footprint_radius_m",
        ),
        support=_text(document["support"], f"{label}.support"),
    )


def _point_in_endpoint_bounds(
    point: Sequence[float], bounds: EndpointPlacementBounds
) -> bool:
    return bounds.x_m.contains(point[0]) and bounds.y_m.contains(point[1])


def _validate_table_containment(
    bounds: EndpointPlacementBounds,
    table: tuple[float, float, float, float],
    label: str,
) -> None:
    x_min, x_max, y_min, y_max = table
    radius = bounds.footprint_radius_m
    if (
        bounds.x_m.lower - radius < x_min
        or bounds.x_m.upper + radius > x_max
        or bounds.y_m.lower - radius < y_min
        or bounds.y_m.upper + radius > y_max
    ):
        raise ValueError(f"{label} footprint can leave table bounds")


def _parse_visibility(value: Any) -> tuple[VisibilityGates, bool, bool]:
    label = "camera_visibility"
    document = _mapping(value, label)
    _exact(
        document,
        {"existing_global_camera", "future_wrist_camera", "gates"},
        label,
    )
    global_camera = _mapping(
        document["existing_global_camera"],
        f"{label}.existing_global_camera",
    )
    _exact(
        global_camera,
        {"available", "prim_path", "frame_id", "resolution_px"},
        f"{label}.existing_global_camera",
    )
    resolution = _vector(
        global_camera["resolution_px"],
        2,
        f"{label}.existing_global_camera.resolution_px",
    )
    if any(value != int(value) or value <= 0 for value in resolution):
        raise ValueError("global camera resolution must be positive integers")
    wrist = _mapping(
        document["future_wrist_camera"],
        f"{label}.future_wrist_camera",
    )
    _exact(
        wrist,
        {"available", "required_for_unique_key_confirmation", "blocker"},
        f"{label}.future_wrist_camera",
    )
    if wrist["required_for_unique_key_confirmation"] is not True:
        raise ValueError("future wrist view must be required for key yaw")
    gates = _mapping(document["gates"], f"{label}.gates")
    _exact(
        gates,
        {
            "minimum_center_margin_px",
            "minimum_mask_pixels_per_endpoint",
            "minimum_visible_fraction_per_endpoint",
            "minimum_valid_depth_fraction_in_mask",
            "maximum_occlusion_fraction",
            "required_views_for_translation_axis",
            "required_views_for_key_yaw",
        },
        f"{label}.gates",
    )
    result = VisibilityGates(
        minimum_center_margin_px=_integer(
            gates["minimum_center_margin_px"],
            "camera_visibility.gates.minimum_center_margin_px",
            minimum=1,
        ),
        minimum_mask_pixels_per_endpoint=_integer(
            gates["minimum_mask_pixels_per_endpoint"],
            "camera_visibility.gates.minimum_mask_pixels_per_endpoint",
            minimum=1,
        ),
        minimum_visible_fraction_per_endpoint=_fraction(
            gates["minimum_visible_fraction_per_endpoint"],
            "camera_visibility.gates.minimum_visible_fraction_per_endpoint",
        ),
        minimum_valid_depth_fraction_in_mask=_fraction(
            gates["minimum_valid_depth_fraction_in_mask"],
            "camera_visibility.gates.minimum_valid_depth_fraction_in_mask",
        ),
        maximum_occlusion_fraction=_fraction(
            gates["maximum_occlusion_fraction"],
            "camera_visibility.gates.maximum_occlusion_fraction",
        ),
        required_views_for_translation_axis=_integer(
            gates["required_views_for_translation_axis"],
            "camera_visibility.gates.required_views_for_translation_axis",
            minimum=1,
        ),
        required_views_for_key_yaw=_integer(
            gates["required_views_for_key_yaw"],
            "camera_visibility.gates.required_views_for_key_yaw",
            minimum=1,
        ),
    )
    if result.required_views_for_key_yaw <= (
        result.required_views_for_translation_axis
    ):
        raise ValueError("key yaw must require an additional calibrated view")
    return (
        result,
        _boolean(global_camera["available"], "global camera available"),
        _boolean(wrist["available"], "wrist camera available"),
    )


def _parse_geometry(value: Any) -> GeometricPoseGates:
    label = "geometric_rgbd_estimator"
    document = _mapping(value, label)
    _exact(
        document,
        {"enabled", "dependencies", "pipeline", "capability_claims", "gates"},
        label,
    )
    if _boolean(document["enabled"], f"{label}.enabled") is not False:
        raise ValueError("geometric estimator must remain disabled")
    _string_list(document["dependencies"], f"{label}.dependencies")
    _string_list(document["pipeline"], f"{label}.pipeline")
    claims = _mapping(
        document["capability_claims"], f"{label}.capability_claims"
    )
    _exact(
        claims,
        {
            "translation_xyz_candidate",
            "connector_axis_candidate",
            "roll_pitch_candidate",
            "unique_key_yaw_candidate",
        },
        f"{label}.capability_claims",
    )
    if any(claims[name] is not True for name in (
        "translation_xyz_candidate",
        "connector_axis_candidate",
        "roll_pitch_candidate",
    )) or claims["unique_key_yaw_candidate"] is not False:
        raise ValueError("geometric estimator capability claims are dishonest")
    gates = _mapping(document["gates"], f"{label}.gates")
    fields = {
        "maximum_translation_std_m",
        "maximum_cad_fit_rmse_m",
        "maximum_axis_error_rad",
        "minimum_axis_inlier_fraction",
        "minimum_unique_yaw_score_margin",
    }
    _exact(gates, fields, f"{label}.gates")
    return GeometricPoseGates(
        maximum_translation_std_m=_positive(
            gates["maximum_translation_std_m"],
            f"{label}.gates.maximum_translation_std_m",
        ),
        maximum_cad_fit_rmse_m=_positive(
            gates["maximum_cad_fit_rmse_m"],
            f"{label}.gates.maximum_cad_fit_rmse_m",
        ),
        maximum_axis_error_rad=_positive(
            gates["maximum_axis_error_rad"],
            f"{label}.gates.maximum_axis_error_rad",
        ),
        minimum_axis_inlier_fraction=_fraction(
            gates["minimum_axis_inlier_fraction"],
            f"{label}.gates.minimum_axis_inlier_fraction",
        ),
        minimum_unique_yaw_score_margin=_fraction(
            gates["minimum_unique_yaw_score_margin"],
            f"{label}.gates.minimum_unique_yaw_score_margin",
        ),
    )


def _unit_quaternion(
    value: Any, label: str
) -> tuple[float, float, float, float]:
    result = _vector(value, 4, label)
    norm = math.sqrt(sum(item * item for item in result))
    if norm == 0.0 or abs(norm - 1.0) > 1.0e-6:
        raise ValueError(f"{label} must be a unit quaternion")
    normalized = tuple(item / norm for item in result)
    if normalized[3] < 0.0:
        normalized = tuple(-item for item in normalized)
    return normalized  # type: ignore[return-value]


def _parse_transforms(value: Any) -> tuple[TargetTransformCandidate, ...]:
    label = "object_target_transforms"
    document = _mapping(value, label)
    _exact(
        document,
        {"convention", "quaternion_convention", "candidates"},
        label,
    )
    if document["convention"] != "parent_T_child":
        raise ValueError("target transform convention must be parent_T_child")
    if document["quaternion_convention"] != "xyzw":
        raise ValueError("target quaternion convention must be xyzw")
    values = document["candidates"]
    if not isinstance(values, list):
        raise ValueError("target transform candidates must be a list")
    fields = {
        "transform_id",
        "model_id",
        "kind",
        "parent_frame_id",
        "child_frame_id",
        "translation_xyz_m",
        "quaternion_xyzw",
        "provenance",
        "qualified",
    }
    result = []
    for index, raw in enumerate(values):
        item_label = f"{label}.candidates[{index}]"
        item = _mapping(raw, item_label)
        _exact(item, fields, item_label)
        kind = _text(item["kind"], f"{item_label}.kind")
        if kind not in {"grasp", "assembly"}:
            raise ValueError(f"{item_label}.kind is unsupported")
        result.append(
            TargetTransformCandidate(
                transform_id=_text(
                    item["transform_id"], f"{item_label}.transform_id"
                ),
                model_id=_text(item["model_id"], f"{item_label}.model_id"),
                kind=kind,
                parent_frame_id=_text(
                    item["parent_frame_id"],
                    f"{item_label}.parent_frame_id",
                ),
                child_frame_id=_text(
                    item["child_frame_id"],
                    f"{item_label}.child_frame_id",
                ),
                translation_xyz_m=_vector(
                    item["translation_xyz_m"],
                    3,
                    f"{item_label}.translation_xyz_m",
                ),
                quaternion_xyzw=_unit_quaternion(
                    item["quaternion_xyzw"],
                    f"{item_label}.quaternion_xyzw",
                ),
                provenance=_text(
                    item["provenance"], f"{item_label}.provenance"
                ),
                qualified=_boolean(
                    item["qualified"], f"{item_label}.qualified"
                ),
            )
        )
    ids = [item.transform_id for item in result]
    if len(result) != 3 or len(ids) != len(set(ids)):
        raise ValueError("exactly three unique target candidates are required")
    roles = {(item.model_id, item.kind) for item in result}
    if roles != {
        ("d38999_26kj61sn_proxy_v1", "grasp"),
        ("d38999_26kj61sn_proxy_v1", "assembly"),
        ("d38999_20kj61pn_proxy_v1", "assembly"),
    }:
        raise ValueError("target candidates do not cover the D38999 pair")
    if any(item.qualified for item in result):
        raise ValueError("proxy-derived target candidates are not calibrated")
    return tuple(result)


def _parse_foundationpose(value: Any):
    label = "foundationpose"
    document = _mapping(value, label)
    fields = {
        "integration_enabled",
        "preferred_adapter",
        "preferred_environment",
        "host_assessment_date",
        "host_ros_distribution",
        "current_official_isaac_ros_release",
        "current_official_ros_distribution",
        "ngc_model_version",
        "ngc_compressed_size_mb",
        "inference_runtime",
        "minimum_recommended_gpu_memory_gb",
        "model_license",
        "ros_wrapper_license",
        "nvlabs_reference_code_license",
        "official_sources",
        "required_asset_paths",
        "blockers",
    }
    _exact(document, fields, label)
    if document["integration_enabled"] is not False:
        raise ValueError("FoundationPose integration must remain disabled")
    if document["preferred_environment"] != "isolated_isaac_ros_container":
        raise ValueError("FoundationPose must be isolated from the host")
    if document["inference_runtime"] != "TensorRT":
        raise ValueError("official FoundationPose route requires TensorRT")
    _positive(document["ngc_compressed_size_mb"], "FoundationPose size")
    _positive(
        document["minimum_recommended_gpu_memory_gb"],
        "FoundationPose GPU memory",
    )
    sources = _mapping(
        document["official_sources"], f"{label}.official_sources"
    )
    source_keys = {
        "documentation",
        "ros_source",
        "reference_code",
        "model_card",
        "refine_onnx",
        "score_onnx",
    }
    _exact(sources, source_keys, f"{label}.official_sources")
    parsed_sources = {
        name: _text(sources[name], f"{label}.official_sources.{name}")
        for name in source_keys
    }
    if any(not url.startswith("https://") for url in parsed_sources.values()):
        raise ValueError("FoundationPose sources must use HTTPS")
    assets = _mapping(
        document["required_asset_paths"],
        f"{label}.required_asset_paths",
    )
    asset_keys = {
        "refine_onnx",
        "score_onnx",
        "refine_engine",
        "score_engine",
        "object_mesh",
    }
    _exact(assets, asset_keys, f"{label}.required_asset_paths")
    parsed_assets = {
        name: _text(assets[name], f"{label}.required_asset_paths.{name}")
        for name in asset_keys
    }
    if any(
        Path(path).is_absolute() or ".." in Path(path).parts
        for path in parsed_assets.values()
    ):
        raise ValueError(
            "FoundationPose asset paths must be workspace-relative"
        )
    blockers = _string_list(document["blockers"], f"{label}.blockers")
    required_blockers = {
        "isaac_ros_environment_not_installed",
        "tensorrt_and_trtexec_not_available_on_host_path",
        "simplified_obj_mesh_not_exported_and_validated",
        "current_proxy_has_no_unique_polarization_key",
    }
    if not required_blockers.issubset(blockers):
        raise ValueError("FoundationPose blocker list is incomplete")
    return (
        parsed_sources,
        parsed_assets,
        blockers,
        _text(document["ngc_model_version"], f"{label}.ngc_model_version"),
        _text(document["model_license"], f"{label}.model_license"),
        _text(
            document["ros_wrapper_license"],
            f"{label}.ros_wrapper_license",
        ),
        _text(
            document["nvlabs_reference_code_license"],
            f"{label}.nvlabs_reference_code_license",
        ),
    )


def load_d38999_multisite_vision6d_contract(
    path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    repository: str | Path | None = None,
) -> D38999MultisiteVision6dContract:
    """Load and cross-check the disabled multi-position vision contract."""
    config_path = Path(path).expanduser().resolve()
    root = (
        Path(repository).expanduser().resolve()
        if repository is not None
        else config_path.parents[3]
    )
    document = _mapping(
        yaml.safe_load(config_path.read_text(encoding="utf-8")),
        "document",
    )
    top_fields = {
        "schema_version",
        "enabled",
        "status",
        "scope",
        "inputs",
        "placement_randomization",
        "camera_visibility",
        "geometric_rgbd_estimator",
        "symmetry_and_keying",
        "object_target_transforms",
        "pose_control_gate",
        "foundationpose",
        "boundaries",
    }
    _exact(document, top_fields, "document")
    if document["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported multi-position vision schema")
    if document["enabled"] is not False:
        raise ValueError("multi-position vision must remain disabled")
    if document["status"] != "prepared_contract_not_e2e_integrated":
        raise ValueError("multi-position vision status is invalid")

    scope = _mapping(document["scope"], "scope")
    _exact(
        scope,
        {
            "delivery_window_hours",
            "achievable_in_window",
            "not_honestly_claimable_in_window",
        },
        "scope",
    )
    delivery_hours = _positive(
        scope["delivery_window_hours"], "scope.delivery_window_hours"
    )
    if delivery_hours != 8.0:
        raise ValueError("this preparation contract is scoped to eight hours")
    achievable = _string_list(
        scope["achievable_in_window"], "scope.achievable_in_window"
    )
    unavailable = _string_list(
        scope["not_honestly_claimable_in_window"],
        "scope.not_honestly_claimable_in_window",
    )
    input_paths = _parse_input_files(document["inputs"], repository=root)

    placement = _mapping(
        document["placement_randomization"], "placement_randomization"
    )
    _exact(
        placement,
        {
            "seed_policy",
            "loose_plug",
            "fixed_receptacle",
            "paired_constraints",
        },
        "placement_randomization",
    )
    if placement["seed_policy"] != "explicit_integer_seed":
        raise ValueError("placement seed policy must be explicit")
    loose = _parse_placement_bounds(
        placement["loose_plug"], "placement_randomization.loose_plug"
    )
    fixed = _parse_placement_bounds(
        placement["fixed_receptacle"],
        "placement_randomization.fixed_receptacle",
    )
    paired = _mapping(
        placement["paired_constraints"],
        "placement_randomization.paired_constraints",
    )
    _exact(
        paired,
        {
            "minimum_center_separation_m",
            "table_xy_bounds_m",
            "required_anchor_pairs",
        },
        "placement_randomization.paired_constraints",
    )
    minimum_separation = _positive(
        paired["minimum_center_separation_m"],
        "placement_randomization.minimum_center_separation_m",
    )
    table_bounds = _vector(
        paired["table_xy_bounds_m"],
        4,
        "placement_randomization.table_xy_bounds_m",
    )
    if not (
        table_bounds[0] < table_bounds[1]
        and table_bounds[2] < table_bounds[3]
    ):
        raise ValueError("table XY bounds are invalid")
    _validate_table_containment(loose, table_bounds, "loose plug")
    _validate_table_containment(fixed, table_bounds, "fixed receptacle")
    anchors_value = paired["required_anchor_pairs"]
    if not isinstance(anchors_value, list) or len(anchors_value) < 5:
        raise ValueError("at least five anchor pairs are required")
    anchors = []
    anchor_ids = set()
    for index, raw in enumerate(anchors_value):
        label = f"placement anchor {index}"
        item = _mapping(raw, label)
        _exact(item, {"id", "loose_xy_m", "fixed_xy_m"}, label)
        identifier = _text(item["id"], f"{label}.id")
        if identifier in anchor_ids:
            raise ValueError("placement anchor IDs must be unique")
        anchor_ids.add(identifier)
        loose_xy = _vector(item["loose_xy_m"], 2, f"{label}.loose_xy_m")
        fixed_xy = _vector(item["fixed_xy_m"], 2, f"{label}.fixed_xy_m")
        if not _point_in_endpoint_bounds(loose_xy, loose):
            raise ValueError(f"{label} loose point is outside bounds")
        if not _point_in_endpoint_bounds(fixed_xy, fixed):
            raise ValueError(f"{label} fixed point is outside bounds")
        separation = math.dist(loose_xy, fixed_xy)
        if separation < minimum_separation:
            raise ValueError(f"{label} violates endpoint separation")
        anchors.append(
            {
                "id": identifier,
                "loose_xy_m": loose_xy,
                "fixed_xy_m": fixed_xy,
            }
        )

    visibility, global_available, wrist_available = _parse_visibility(
        document["camera_visibility"]
    )
    geometry = _parse_geometry(document["geometric_rgbd_estimator"])

    symmetry = _mapping(document["symmetry_and_keying"], "symmetry_and_keying")
    _exact(
        symmetry,
        {
            "current_proxy_has_unique_polarization_key",
            "body_and_receptacle_minimum_yaw_symmetry_order",
            "equivalent_yaw_period_rad",
            "periodic_component_orders",
            "coupling_nut_is_independent_yaw_body",
            "contact_pattern_is_visual_only",
            "required_before_keyed_yaw_claim",
        },
        "symmetry_and_keying",
    )
    has_key = _boolean(
        symmetry["current_proxy_has_unique_polarization_key"],
        "symmetry current proxy key",
    )
    symmetry_order = _integer(
        symmetry["body_and_receptacle_minimum_yaw_symmetry_order"],
        "symmetry order",
        minimum=2,
    )
    symmetry_period = _positive(
        symmetry["equivalent_yaw_period_rad"], "symmetry period"
    )
    period_error = abs(
        symmetry_period - 2.0 * math.pi / symmetry_order
    )
    if has_key or period_error > 1.0e-12:
        raise ValueError("current proxy yaw symmetry declaration is invalid")
    component_document = _mapping(
        symmetry["periodic_component_orders"],
        "symmetry_and_keying.periodic_component_orders",
    )
    component_names = {
        "loose_body",
        "fixed_receptacle",
        "coupling_nut",
    }
    _exact(
        component_document,
        component_names,
        "symmetry_and_keying.periodic_component_orders",
    )
    component_orders = {}
    for name in sorted(component_names):
        values = component_document[name]
        if not isinstance(values, list) or not values:
            raise ValueError(f"periodic component {name} must be a list")
        orders = tuple(
            _integer(item, f"periodic component {name}", minimum=2)
            for item in values
        )
        component_orders[name] = orders
    for endpoint in ("loose_body", "fixed_receptacle"):
        common_order = math.gcd(*component_orders[endpoint])
        if common_order != symmetry_order:
            raise ValueError(
                f"{endpoint} component symmetry differs from declaration"
            )
    nut_independent = _boolean(
        symmetry["coupling_nut_is_independent_yaw_body"],
        "coupling nut independent yaw",
    )
    if (
        not nut_independent
        or symmetry["contact_pattern_is_visual_only"] is not True
    ):
        raise ValueError("proxy symmetry limitations must remain explicit")
    _string_list(
        symmetry["required_before_keyed_yaw_claim"],
        "symmetry_and_keying.required_before_keyed_yaw_claim",
    )
    transforms = _parse_transforms(document["object_target_transforms"])

    provider = _mapping(document["pose_control_gate"], "pose_control_gate")
    provider_fields = {
        "provider_schema",
        "required_pose_source",
        "require_both_endpoints",
        "require_full_6d",
        "require_keyed_orientation_observed",
        "require_no_truth_position",
        "require_no_truth_orientation",
        "require_all_target_transforms_qualified",
        "current_control_authorized",
    }
    _exact(provider, provider_fields, "pose_control_gate")
    if provider["provider_schema"] != "kcg_connector_pose_provider_sample_v1":
        raise ValueError("pose provider schema differs from shared boundary")
    if provider["required_pose_source"] != "vision":
        raise ValueError("multi-position control requires a vision source")
    required_true = provider_fields - {
        "provider_schema",
        "required_pose_source",
        "current_control_authorized",
    }
    if any(provider[name] is not True for name in required_true):
        raise ValueError("pose control requirements must all be enabled")
    current_authorized = _boolean(
        provider["current_control_authorized"],
        "pose_control_gate.current_control_authorized",
    )
    if current_authorized:
        raise ValueError("vision control is not currently authorized")

    (
        official_sources,
        asset_paths,
        blockers,
        model_version,
        model_license,
        wrapper_license,
        reference_code_license,
    ) = _parse_foundationpose(document["foundationpose"])

    boundaries_doc = _mapping(document["boundaries"], "boundaries")
    boundary_fields = {
        "e2e_integration_allowed",
        "object_pose_writes_after_start_allowed",
        "host_package_install_allowed",
        "model_download_performed",
        "tensorrt_engine_build_performed",
        "calibrated_transform_claimed",
        "unique_key_yaw_claimed",
        "vision_control_authorized",
        "real_assembly_success_claimed",
    }
    _exact(boundaries_doc, boundary_fields, "boundaries")
    boundaries = {
        name: _boolean(boundaries_doc[name], f"boundaries.{name}")
        for name in sorted(boundary_fields)
    }
    if any(boundaries.values()):
        raise ValueError("vision preparation boundaries must all be false")

    return D38999MultisiteVision6dContract(
        schema_version=SCHEMA_VERSION,
        enabled=False,
        status=document["status"],
        delivery_window_hours=delivery_hours,
        achievable_in_window=achievable,
        not_honestly_claimable_in_window=unavailable,
        input_paths=input_paths,
        loose_plug=loose,
        fixed_receptacle=fixed,
        minimum_center_separation_m=minimum_separation,
        table_xy_bounds_m=table_bounds,
        required_anchor_pairs=tuple(anchors),
        visibility=visibility,
        geometry=geometry,
        global_camera_available=global_available,
        wrist_camera_available=wrist_available,
        current_proxy_has_unique_polarization_key=has_key,
        yaw_symmetry_order=symmetry_order,
        equivalent_yaw_period_rad=symmetry_period,
        periodic_component_orders=component_orders,
        coupling_nut_is_independent_yaw_body=nut_independent,
        target_transforms=transforms,
        pose_control_current_authorized=current_authorized,
        foundationpose_official_sources=official_sources,
        foundationpose_required_asset_paths=asset_paths,
        foundationpose_blockers=blockers,
        foundationpose_model_version=model_version,
        foundationpose_model_license=model_license,
        foundationpose_ros_wrapper_license=wrapper_license,
        foundationpose_reference_code_license=reference_code_license,
        boundaries=boundaries,
    )


def sample_multisite_placement(
    contract: D38999MultisiteVision6dContract,
    *,
    seed: int,
) -> PlacementPairSample:
    """Sample one reproducible pair without touching a simulator stage."""
    if not isinstance(contract, D38999MultisiteVision6dContract):
        raise ValueError("contract must be D38999MultisiteVision6dContract")
    seed_value = _integer(seed, "seed")
    generator = random.Random(seed_value)

    def sample_endpoint(bounds: EndpointPlacementBounds):
        return (
            generator.uniform(bounds.x_m.lower, bounds.x_m.upper),
            generator.uniform(bounds.y_m.lower, bounds.y_m.upper),
            bounds.origin_z_m,
            generator.uniform(bounds.yaw_rad.lower, bounds.yaw_rad.upper),
        )

    loose_x, loose_y, loose_z, loose_yaw = sample_endpoint(
        contract.loose_plug
    )
    fixed_x, fixed_y, fixed_z, fixed_yaw = sample_endpoint(
        contract.fixed_receptacle
    )
    separation = math.dist((loose_x, loose_y), (fixed_x, fixed_y))
    if separation < contract.minimum_center_separation_m:
        raise RuntimeError(
            "sampled endpoint pair violates separation contract"
        )
    return PlacementPairSample(
        seed=seed_value,
        loose_position_xyz_m=(loose_x, loose_y, loose_z),
        loose_yaw_rad=loose_yaw,
        fixed_position_xyz_m=(fixed_x, fixed_y, fixed_z),
        fixed_yaw_rad=fixed_yaw,
        center_separation_m=separation,
    )


def _parse_endpoint_evidence(value: Any, label: str) -> EndpointPoseEvidence:
    document = _mapping(value, label)
    fields = {
        "mask_pixels",
        "visible_fraction",
        "valid_depth_fraction_in_mask",
        "center_margin_px",
        "occlusion_fraction",
        "calibrated_view_count",
        "translation_std_m",
        "cad_fit_rmse_m",
        "axis_error_rad",
        "axis_inlier_fraction",
        "key_feature_observed",
        "yaw_hypothesis_count",
        "unique_yaw_score_margin",
    }
    _exact(document, fields, label)
    return EndpointPoseEvidence(
        mask_pixels=_integer(document["mask_pixels"], f"{label}.mask_pixels"),
        visible_fraction=_fraction(
            document["visible_fraction"], f"{label}.visible_fraction"
        ),
        valid_depth_fraction_in_mask=_fraction(
            document["valid_depth_fraction_in_mask"],
            f"{label}.valid_depth_fraction_in_mask",
        ),
        center_margin_px=_real(
            document["center_margin_px"], f"{label}.center_margin_px"
        ),
        occlusion_fraction=_fraction(
            document["occlusion_fraction"], f"{label}.occlusion_fraction"
        ),
        calibrated_view_count=_integer(
            document["calibrated_view_count"],
            f"{label}.calibrated_view_count",
        ),
        translation_std_m=_positive(
            document["translation_std_m"], f"{label}.translation_std_m"
        ),
        cad_fit_rmse_m=_positive(
            document["cad_fit_rmse_m"], f"{label}.cad_fit_rmse_m"
        ),
        axis_error_rad=_real(
            document["axis_error_rad"], f"{label}.axis_error_rad"
        ),
        axis_inlier_fraction=_fraction(
            document["axis_inlier_fraction"],
            f"{label}.axis_inlier_fraction",
        ),
        key_feature_observed=_boolean(
            document["key_feature_observed"],
            f"{label}.key_feature_observed",
        ),
        yaw_hypothesis_count=_integer(
            document["yaw_hypothesis_count"],
            f"{label}.yaw_hypothesis_count",
            minimum=1,
        ),
        unique_yaw_score_margin=_fraction(
            document["unique_yaw_score_margin"],
            f"{label}.unique_yaw_score_margin",
        ),
    )


def evaluate_pose_control_gate(
    contract: D38999MultisiteVision6dContract,
    evidence: Mapping[str, Any],
) -> PoseControlGateResult:
    """Evaluate translation/axis/yaw separately and deny ambiguous control."""
    if not isinstance(contract, D38999MultisiteVision6dContract):
        raise ValueError("contract must be D38999MultisiteVision6dContract")
    document = _mapping(evidence, "evidence")
    _exact(document, {"loose_plug", "fixed_receptacle"}, "evidence")
    parsed = {
        role: _parse_endpoint_evidence(document[role], f"evidence.{role}")
        for role in ("loose_plug", "fixed_receptacle")
    }
    results = {}
    reasons = []
    for role, item in parsed.items():
        visibility = bool(
            item.mask_pixels
            >= contract.visibility.minimum_mask_pixels_per_endpoint
            and item.visible_fraction
            >= contract.visibility.minimum_visible_fraction_per_endpoint
            and item.valid_depth_fraction_in_mask
            >= contract.visibility.minimum_valid_depth_fraction_in_mask
            and item.center_margin_px
            >= contract.visibility.minimum_center_margin_px
            and item.occlusion_fraction
            <= contract.visibility.maximum_occlusion_fraction
            and item.calibrated_view_count
            >= contract.visibility.required_views_for_translation_axis
        )
        translation = bool(
            visibility
            and item.translation_std_m
            <= contract.geometry.maximum_translation_std_m
            and item.cad_fit_rmse_m
            <= contract.geometry.maximum_cad_fit_rmse_m
        )
        axis = bool(
            translation
            and 0.0 <= item.axis_error_rad
            <= contract.geometry.maximum_axis_error_rad
            and item.axis_inlier_fraction
            >= contract.geometry.minimum_axis_inlier_fraction
        )
        keyed_yaw = bool(
            axis
            and contract.current_proxy_has_unique_polarization_key
            and item.key_feature_observed
            and item.calibrated_view_count
            >= contract.visibility.required_views_for_key_yaw
            and item.yaw_hypothesis_count == 1
            and item.unique_yaw_score_margin
            >= contract.geometry.minimum_unique_yaw_score_margin
        )
        results[role] = EndpointPoseGate(
            visibility_passed=visibility,
            translation_observed=translation,
            axis_observed=axis,
            keyed_yaw_observed=keyed_yaw,
        )
        if not visibility:
            reasons.append(f"{role}:visibility_gate_failed")
        elif not translation:
            reasons.append(f"{role}:translation_gate_failed")
        elif not axis:
            reasons.append(f"{role}:axis_gate_failed")
        if not keyed_yaw:
            reasons.append(f"{role}:keyed_yaw_unobservable")

    full_6d = all(item.keyed_yaw_observed for item in results.values())
    all_qualified = all(item.qualified for item in contract.target_transforms)
    if not contract.enabled:
        reasons.append("contract_disabled")
    if not contract.current_proxy_has_unique_polarization_key:
        reasons.append("proxy_unique_key_geometry_absent")
    if not contract.wrist_camera_available:
        reasons.append("second_calibrated_view_unavailable")
    if not all_qualified:
        reasons.append("target_transforms_unqualified")
    authorized = bool(
        contract.enabled
        and contract.pose_control_current_authorized
        and full_6d
        and all_qualified
    )
    return PoseControlGateResult(
        loose_plug=results["loose_plug"],
        fixed_receptacle=results["fixed_receptacle"],
        full_6d=full_6d,
        keyed_orientation_observed=full_6d,
        all_target_transforms_qualified=all_qualified,
        control_authorized=authorized,
        reasons=tuple(dict.fromkeys(reasons)),
    )


def _quaternion_multiply(first, second):
    x1, y1, z1, w1 = first
    x2, y2, z2, w2 = second
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def _rotate_vector(quaternion, vector):
    x_value, y_value, z_value, w_value = quaternion
    vector_quaternion = (vector[0], vector[1], vector[2], 0.0)
    conjugate = (-x_value, -y_value, -z_value, w_value)
    rotated = _quaternion_multiply(
        _quaternion_multiply(quaternion, vector_quaternion), conjugate
    )
    return rotated[:3]


def resolve_target_candidate(
    contract: D38999MultisiteVision6dContract,
    *,
    transform_id: str,
    object_position_xyz_m: Sequence[Real],
    object_quaternion_xyzw: Sequence[Real],
) -> ResolvedTargetCandidate:
    """Compose ``world_T_object * object_T_target`` without upgrading trust."""
    transform = contract.transform(transform_id)
    object_position = _vector(
        object_position_xyz_m, 3, "object_position_xyz_m"
    )
    object_orientation = _unit_quaternion(
        object_quaternion_xyzw, "object_quaternion_xyzw"
    )
    offset = _rotate_vector(
        object_orientation, transform.translation_xyz_m
    )
    target_position = tuple(
        object_position[index] + offset[index] for index in range(3)
    )
    target_orientation = _unit_quaternion(
        _quaternion_multiply(
            object_orientation, transform.quaternion_xyzw
        ),
        "resolved target quaternion",
    )
    return ResolvedTargetCandidate(
        transform_id=transform.transform_id,
        position_xyz_m=target_position,
        quaternion_xyzw=target_orientation,
        qualified=transform.qualified,
    )
