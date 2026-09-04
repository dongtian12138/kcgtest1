#!/usr/bin/env python3

"""Capture one truth-firewalled TE J35 RGB-D observation in Isaac Sim.

The entry authors the already-declared finite table and two official TE visual
assets, then delegates frame acquisition to
``capture_d38999_rgbd_raw_formal``.  It has no robot, semantic annotator, pose
provider, object-pose readback, or control command.  Scene-generation poses
are kept in the capture report but are deliberately omitted from the versioned
``provider_input.json`` interface.

The frozen plug attitude is the historical Rx180 tabletop attitude.  A later
provider must return ``MISS_KEY_NOT_OBSERVABLE`` when the ordinary images do
not uniquely show the wide main key; this entry never fills that missing yaw
from the authored scene transform.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import traceback
from typing import Any, Mapping

import numpy as np
import yaml


SCHEMA_VERSION = "kcg_te_rgbd_observe_only_v1"
VIEW_OVERRIDE_SCHEMA_VERSION = "kcg_te_rgbd_observe_only_view_override_v1"
KEY_OBSERVATION_OVERLAY_SCHEMA_VERSION = (
    "kcg_te_rgbd_observe_only_key_observation_overlay_v1"
)
RX180_TRANSLATION_OVERLAY_SCHEMA_VERSION = (
    "kcg_te_rgbd_observe_only_rx180_translation_overlay_v1"
)
CAMERA_SCHEMA_VERSION = "kcg_te_rgbd_camera_v1"
CAPTURE_REPORT_SCHEMA_VERSION = "kcg_te_rgbd_observe_only_capture_v1"
PROVIDER_INPUT_SCHEMA_VERSION = "kcg_te_rgbd_provider_input_v1"
DEFAULT_CONFIG_RELATIVE = (
    "src/kcg_connector/config/te_rgbd_observe_only_v1.yaml"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_PRIM_PATH = re.compile(r"^/World(?:/[A-Za-z_][A-Za-z0-9_]*)+$")


@dataclass(frozen=True)
class LoadedObserveOnlyConfig:
    path: Path
    document: Mapping[str, Any]
    source_paths: Mapping[str, Path]
    asset_paths: Mapping[str, Mapping[str, Path]]
    tabletop: Any
    rgbd: Any
    te_contract: Mapping[str, Any]
    geometry_audit: Mapping[str, Any]


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _exact_keys(value: Mapping[str, Any], expected, label: str) -> None:
    wanted = set(expected)
    actual = set(value)
    if actual != wanted:
        raise ValueError(
            f"{label} keys differ; missing={sorted(wanted - actual)}, "
            f"extra={sorted(actual - wanted)}"
        )


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be non-empty unpadded text")
    return value


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _vector(value: Any, size: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{label} must contain {size} finite values")
    return tuple(
        _finite(item, f"{label}[{index}]")
        for index, item in enumerate(value)
    )


def _identifier(value: Any, label: str) -> str:
    result = _text(value, label)
    if not _IDENTIFIER.fullmatch(result):
        raise ValueError(f"{label} must be a stable identifier")
    return result


def _prim_path(value: Any, label: str) -> str:
    result = _text(value, label)
    if not _PRIM_PATH.fullmatch(result):
        raise ValueError(f"{label} must be an absolute /World prim path")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _repository_file(repository: Path, value: Any, label: str) -> Path:
    relative = Path(_text(value, label))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} must be repository-relative")
    result = (repository / relative).resolve()
    try:
        result.relative_to(repository)
    except ValueError as error:
        raise ValueError(f"{label} escapes the repository") from error
    if not result.is_file():
        raise ValueError(f"{label} does not exist: {relative}")
    return result


def _load_yaml(path: Path, label: str) -> Mapping[str, Any]:
    return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), label)


def _rotation_xyz_degrees(value: Any, label: str) -> np.ndarray:
    rx_value, ry_value, rz_value = np.radians(_vector(value, 3, label))
    sx, cx = math.sin(rx_value), math.cos(rx_value)
    sy, cy = math.sin(ry_value), math.cos(ry_value)
    sz, cz = math.sin(rz_value), math.cos(rz_value)
    rotation_x = np.asarray(
        ((1.0, 0.0, 0.0), (0.0, cx, -sx), (0.0, sx, cx))
    )
    rotation_y = np.asarray(
        ((cy, 0.0, sy), (0.0, 1.0, 0.0), (-sy, 0.0, cy))
    )
    rotation_z = np.asarray(
        ((cz, -sz, 0.0), (sz, cz, 0.0), (0.0, 0.0, 1.0))
    )
    return rotation_z @ rotation_y @ rotation_x


def _world_aabb_from_supplier_bbox(
    bbox: Mapping[str, Any],
    translation: tuple[float, float, float],
    rotation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    minimum = np.asarray(
        (
            _finite(bbox["xmin"], "bbox.xmin"),
            _finite(bbox["ymin"], "bbox.ymin"),
            _finite(bbox["zmin"], "bbox.zmin"),
        ),
        dtype=np.float64,
    ) / 1000.0
    maximum = np.asarray(
        (
            _finite(bbox["xmax"], "bbox.xmax"),
            _finite(bbox["ymax"], "bbox.ymax"),
            _finite(bbox["zmax"], "bbox.zmax"),
        ),
        dtype=np.float64,
    ) / 1000.0
    corners = np.asarray(
        [
            (x_value, y_value, z_value)
            for x_value in (minimum[0], maximum[0])
            for y_value in (minimum[1], maximum[1])
            for z_value in (minimum[2], maximum[2])
        ],
        dtype=np.float64,
    )
    world = corners @ rotation.T + np.asarray(translation, dtype=np.float64)
    return np.min(world, axis=0), np.max(world, axis=0)


def _parse_aabb(value: Any, label: str) -> tuple[np.ndarray, np.ndarray]:
    document = _mapping(value, label)
    _exact_keys(document, ("minimum", "maximum"), label)
    minimum = np.asarray(_vector(document["minimum"], 3, f"{label}.minimum"))
    maximum = np.asarray(_vector(document["maximum"], 3, f"{label}.maximum"))
    if np.any(minimum >= maximum):
        raise ValueError(f"{label} must have increasing bounds")
    return minimum, maximum


def _resolve_view_override(
    repository: Path,
    path: Path,
    raw_document: Mapping[str, Any],
) -> tuple[Mapping[str, Any], dict[str, Path], Mapping[str, Any] | None]:
    """Resolve a camera/scene overlay without duplicating the base runner."""
    overlay_schema = raw_document.get("schema_version")
    if overlay_schema not in {
        VIEW_OVERRIDE_SCHEMA_VERSION,
        KEY_OBSERVATION_OVERLAY_SCHEMA_VERSION,
        RX180_TRANSLATION_OVERLAY_SCHEMA_VERSION,
    }:
        return raw_document, {}, None
    key_observation_overlay = (
        overlay_schema == KEY_OBSERVATION_OVERLAY_SCHEMA_VERSION
    )
    rx180_translation_overlay = (
        overlay_schema == RX180_TRANSLATION_OVERLAY_SCHEMA_VERSION
    )
    expected_fields = [
        "schema_version",
        "base_observe_config",
        "camera_config",
        "experiment_contract",
    ]
    if key_observation_overlay or rx180_translation_overlay:
        expected_fields.append("scene_generation_truth")
    if key_observation_overlay:
        expected_fields.extend(
            (
                "body_rear_face_to_main_key_contract",
                "geometry_preflight",
            )
        )
    _exact_keys(
        raw_document,
        tuple(expected_fields),
        "view_override",
    )
    base_path = _repository_file(
        repository,
        raw_document["base_observe_config"],
        "view_override.base_observe_config",
    )
    camera_path = _repository_file(
        repository,
        raw_document["camera_config"],
        "view_override.camera_config",
    )
    base = _load_yaml(base_path, "base_observe_config")
    if base.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("view override base schema differs")
    camera_document = _load_yaml(camera_path, "camera_config")
    _exact_keys(
        camera_document,
        ("schema_version", "camera", "geometry_preflight"),
        "camera_config",
    )
    if camera_document["schema_version"] != CAMERA_SCHEMA_VERSION:
        raise ValueError("unsupported TE RGB-D camera schema")
    camera = _mapping(camera_document["camera"], "camera_config.camera")
    _exact_keys(
        camera,
        (
            "view_id",
            "calibration_id",
            "prim_path",
            "frame_id",
            "eye_world_m",
            "target_world_m",
            "resolution_px",
            "frequency_hz",
            "warmup_frames",
            "channels_exactly",
            "camera_cv_axes",
            "focal_length_mm",
            "horizontal_aperture_mm",
            "clipping_range_m",
        ),
        "camera_config.camera",
    )
    preflight = _mapping(
        camera_document["geometry_preflight"],
        "camera_config.geometry_preflight",
    )
    plug_preflight = _mapping(preflight["plug"], "geometry_preflight.plug")
    receptacle_preflight = _mapping(
        preflight["receptacle"], "geometry_preflight.receptacle"
    )
    coarse_global_handoff_only = (
        preflight.get("coarse_global_handoff_only") is True
    )
    if not coarse_global_handoff_only and min(
        _finite(
            plug_preflight["optimistic_smallest_key_width_px"],
            "plug optimistic key width",
        ),
        _finite(
            plug_preflight["optimistic_exposed_key_band_px"],
            "plug exposed key band",
        ),
        _finite(
            receptacle_preflight["optimistic_smallest_key_width_px"],
            "receptacle optimistic key width",
        ),
    ) < 4.4:
        raise ValueError("near-view key projection lacks ten-percent margin")
    if _finite(
        preflight["minimum_image_border_margin_px"],
        "minimum image border margin",
    ) < 16.0:
        raise ValueError("near-view official CAD AABB lacks 16 px border")
    if preflight.get("full_aabb_in_frame") is not True:
        raise ValueError("near-view official CAD AABB is not in frame")
    import copy

    merged = copy.deepcopy(base)
    merged["experiment_contract"] = dict(raw_document["experiment_contract"])
    override_paths = {
        "base_observe_config": base_path,
        "camera_config": camera_path,
    }
    if key_observation_overlay or rx180_translation_overlay:
        merged["scene_generation_truth"] = dict(
            raw_document["scene_generation_truth"]
        )
    if rx180_translation_overlay:
        base_truth = _mapping(
            base["scene_generation_truth"], "base.scene_generation_truth"
        )
        translated_truth = _mapping(
            raw_document["scene_generation_truth"],
            "rx180_translation.scene_generation_truth",
        )
        expected_translation = (
            np.asarray(base_truth["plug_world_translation_m"], dtype=np.float64)
            + np.asarray((0.015, 0.015, 0.0), dtype=np.float64)
        )
        if (
            not np.array_equal(
                np.asarray(
                    translated_truth["plug_world_translation_m"],
                    dtype=np.float64,
                ),
                expected_translation,
            )
            or translated_truth.get("plug_world_rotation_xyz_deg")
            != [180.0, 0.0, 0.0]
            or translated_truth.get("plug_support_surface") != "table_top"
            or any(
                translated_truth.get(name) != base_truth.get(name)
                for name in (
                    "receptacle_world_translation_m",
                    "receptacle_world_rotation_xyz_deg",
                    "receptacle_support_surface",
                )
            )
        ):
            raise ValueError(
                "Rx180 validation overlay must change only plug x/y by +15 mm"
            )
    if key_observation_overlay:
        relation_path = _repository_file(
            repository,
            raw_document["body_rear_face_to_main_key_contract"],
            "view_override.body_rear_face_to_main_key_contract",
        )
        relation = _load_yaml(relation_path, "body_rear_face_to_main_key")
        if relation.get("schema_version") != (
            "kcg_te_body_rear_face_to_main_key_v1"
        ) or relation.get("identity", {}).get("reference_part") != "PlugBody":
            raise ValueError("body-to-key relation is not PlugBody-bound")
        if relation.get("identity", {}).get("forbidden_reference_part") != (
            "CouplingNut"
        ):
            raise ValueError("body-to-key relation does not forbid nut reference")
        geometry = _mapping(
            raw_document["geometry_preflight"],
            "key_observation.geometry_preflight",
        )
        if geometry.get("main_key_directly_visible") is not True or int(
            geometry.get("main_key_depth_consistent_pixels", 0)
        ) < 12:
            raise ValueError("key-observation overlay lacks direct main-key support")
        override_paths["body_rear_face_to_main_key_contract"] = relation_path
    observation = merged["observation_contract"]
    observation.update(
        {
            "view_id": camera["view_id"],
            "calibration_id": camera["calibration_id"],
            "camera_frame_id": camera["frame_id"],
            "camera_eye_world_m": camera["eye_world_m"],
            "camera_target_world_m": camera["target_world_m"],
            "resolution_px": camera["resolution_px"],
            "channels_exactly": camera["channels_exactly"],
            "camera_cv_axes": camera["camera_cv_axes"],
            "focal_length_mm": camera["focal_length_mm"],
            "horizontal_aperture_mm": camera["horizontal_aperture_mm"],
            "clipping_range_m": camera["clipping_range_m"],
        }
    )
    return (
        merged,
        override_paths,
        camera_document,
    )


def load_observe_only_config(
    repository: Path,
    config_path: Path,
) -> LoadedObserveOnlyConfig:
    from kcg_connector.d38999_tabletop_scene import (
        load_d38999_tabletop_scene,
    )
    from kcg_connector.isaac_d38999_rgbd_runtime import (
        RGBD_CAMERA_CLIPPING_RANGE_M,
        RGBD_CAMERA_FOCAL_LENGTH_MM,
        RGBD_CAMERA_HORIZONTAL_APERTURE_MM,
    )
    from kcg_connector.rgbd_pose_bootstrap import load_rgbd_bootstrap
    from kcg_connector.te_rgbd_observability import world_from_camera_cv

    path = config_path.expanduser().resolve()
    try:
        path.relative_to(repository)
    except ValueError as error:
        raise ValueError("observe-only config escapes the repository") from error
    raw_document = _load_yaml(path, "config")
    document, override_paths, camera_override = _resolve_view_override(
        repository, path, raw_document
    )
    _exact_keys(
        document,
        (
            "schema_version",
            "mode",
            "authorization",
            "source_contracts",
            "experiment_contract",
            "assets",
            "scene_generation_truth",
            "observation_contract",
            "formal_gate",
            "truth_firewall",
            "output",
        ),
        "config",
    )
    if document["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported TE RGB-D observe-only schema")
    if document["mode"] != "SIMULATION_ONLY_TE_RGBD_OBSERVE_ONLY":
        raise ValueError("observe-only mode differs from the frozen contract")

    authorization = _mapping(document["authorization"], "authorization")
    _exact_keys(
        authorization,
        (
            "simulation_only",
            "hardware_authorized",
            "robot_present",
            "robot_motion_authorized",
            "visual_control_authorized",
            "insertion_control_authorized",
        ),
        "authorization",
    )
    expected_authorization = {
        "simulation_only": True,
        "hardware_authorized": False,
        "robot_present": False,
        "robot_motion_authorized": False,
        "visual_control_authorized": False,
        "insertion_control_authorized": False,
    }
    for name, expected in expected_authorization.items():
        if _boolean(authorization[name], f"authorization.{name}") is not expected:
            raise ValueError(f"authorization.{name} violates observe-only scope")

    experiment = _mapping(
        document["experiment_contract"], "experiment_contract"
    )
    _exact_keys(
        experiment,
        (
            "current_physical_state",
            "unique_unknown",
            "single_frozen_variable",
            "falsifiable_judgment",
            "miss_branch",
            "ready_branch",
        ),
        "experiment_contract",
    )
    for name, value in experiment.items():
        _text(value, f"experiment_contract.{name}")

    source_contracts = _mapping(
        document["source_contracts"], "source_contracts"
    )
    _exact_keys(
        source_contracts,
        ("tabletop_config", "rgbd_config", "te_contract", "geometry_audit"),
        "source_contracts",
    )
    source_paths = {
        name: _repository_file(
            repository, value, f"source_contracts.{name}"
        )
        for name, value in source_contracts.items()
    }
    source_paths.update(override_paths)
    tabletop = load_d38999_tabletop_scene(source_paths["tabletop_config"])
    rgbd = load_rgbd_bootstrap(source_paths["rgbd_config"])
    if camera_override is not None:
        from dataclasses import replace
        from kcg_connector.rgbd_pose_bootstrap import RgbdCamera

        camera = camera_override["camera"]
        rgbd = replace(
            rgbd,
            camera=RgbdCamera(
                prim_path=_prim_path(camera["prim_path"], "camera.prim_path"),
                frame_id=_text(camera["frame_id"], "camera.frame_id"),
                eye_m=_vector(camera["eye_world_m"], 3, "camera.eye_world_m"),
                target_m=_vector(
                    camera["target_world_m"], 3, "camera.target_world_m"
                ),
                resolution=tuple(
                    int(value)
                    for value in _vector(
                        camera["resolution_px"], 2, "camera.resolution_px"
                    )
                ),
                frequency_hz=int(
                    _finite(camera["frequency_hz"], "camera.frequency_hz")
                ),
                warmup_frames=int(
                    _finite(camera["warmup_frames"], "camera.warmup_frames")
                ),
            ),
        )
    te_contract = _load_yaml(source_paths["te_contract"], "te_contract")
    geometry_audit = _mapping(
        json.loads(source_paths["geometry_audit"].read_text(encoding="utf-8")),
        "geometry_audit",
    )

    assets = _mapping(document["assets"], "assets")
    _exact_keys(assets, ("plug", "receptacle"), "assets")
    asset_paths: dict[str, dict[str, Path]] = {}
    asset_fields = (
        "role",
        "model_id",
        "identity",
        "render_asset",
        "render_asset_sha256",
        "registration_cad",
        "registration_cad_sha256",
        "pose_prim_path",
        "reference_prim_path",
    )
    expected_identity = {
        "plug": ("loose_plug", "D38999/26FJ35PN", "plug_complete"),
        "receptacle": (
            "fixed_receptacle",
            "D38999/20FJ35SN",
            "receptacle",
        ),
    }
    te_identity = _mapping(te_contract.get("identity"), "te_contract.identity")
    te_visual_assets = _mapping(
        te_contract.get("visual_assets"), "te_contract.visual_assets"
    )
    for endpoint, expected in expected_identity.items():
        asset = _mapping(assets[endpoint], f"assets.{endpoint}")
        _exact_keys(asset, asset_fields, f"assets.{endpoint}")
        expected_role, expected_name, te_asset_key = expected
        if asset["role"] != expected_role or asset["identity"] != expected_name:
            raise ValueError(f"assets.{endpoint} identity or role differs")
        te_identity_key = "plug" if endpoint == "plug" else "receptacle"
        if te_identity.get(te_identity_key) != expected_name:
            raise ValueError(f"TE contract {endpoint} identity differs")
        if te_visual_assets.get(te_asset_key) != asset["render_asset"]:
            raise ValueError(f"TE contract {endpoint} visual asset differs")
        render_path = _repository_file(
            repository, asset["render_asset"], f"assets.{endpoint}.render_asset"
        )
        cad_path = _repository_file(
            repository,
            asset["registration_cad"],
            f"assets.{endpoint}.registration_cad",
        )
        for path_key, file_path in (
            ("render_asset_sha256", render_path),
            ("registration_cad_sha256", cad_path),
        ):
            expected_sha = _text(asset[path_key], f"assets.{endpoint}.{path_key}")
            if not _SHA256.fullmatch(expected_sha) or _sha256(file_path) != expected_sha:
                raise ValueError(f"assets.{endpoint}.{path_key} differs")
        pose_path = _prim_path(
            asset["pose_prim_path"], f"assets.{endpoint}.pose_prim_path"
        )
        reference_path = _prim_path(
            asset["reference_prim_path"],
            f"assets.{endpoint}.reference_prim_path",
        )
        if not reference_path.startswith(pose_path + "/"):
            raise ValueError(f"assets.{endpoint} reference is not below pose prim")
        asset_paths[endpoint] = {
            "render_asset": render_path,
            "registration_cad": cad_path,
        }

    scene_truth = _mapping(
        document["scene_generation_truth"], "scene_generation_truth"
    )
    _exact_keys(
        scene_truth,
        (
            "scene_pose_id",
            "plug_world_translation_m",
            "plug_world_rotation_xyz_deg",
            "plug_support_surface",
            "receptacle_world_translation_m",
            "receptacle_world_rotation_xyz_deg",
            "receptacle_support_surface",
        ),
        "scene_generation_truth",
    )
    _identifier(scene_truth["scene_pose_id"], "scene_generation_truth.scene_pose_id")
    if scene_truth["plug_support_surface"] != "table_top":
        raise ValueError("plug support surface must be table_top")
    if scene_truth["receptacle_support_surface"] != "fixture_top":
        raise ValueError("receptacle support surface must be fixture_top")

    observation = _mapping(
        document["observation_contract"], "observation_contract"
    )
    observation_fields = (
        "view_id",
        "calibration_id",
        "world_frame_id",
        "camera_frame_id",
        "camera_eye_world_m",
        "camera_target_world_m",
        "camera_cv_axes",
        "resolution_px",
        "channels_exactly",
        "focal_length_mm",
        "horizontal_aperture_mm",
        "clipping_range_m",
        "plug_workspace_world_aabb_m",
        "receptacle_workspace_world_aabb_m",
        "minimum_endpoint_points",
        "minimum_key_width_px",
        "minimum_foreground_depth_delta_m",
        "capture_rt_subframes",
    )
    _exact_keys(observation, observation_fields, "observation_contract")
    _identifier(observation["view_id"], "observation_contract.view_id")
    _identifier(
        observation["calibration_id"], "observation_contract.calibration_id"
    )
    if observation["world_frame_id"] != "world":
        raise ValueError("observe-only output frame must be world")
    if observation["camera_frame_id"] != rgbd.camera.frame_id:
        raise ValueError("camera frame differs from RGB-D source contract")
    if tuple(_vector(observation["camera_eye_world_m"], 3, "camera eye")) != tuple(
        rgbd.camera.eye_m
    ):
        raise ValueError("camera eye differs from RGB-D source contract")
    if tuple(
        _vector(observation["camera_target_world_m"], 3, "camera target")
    ) != tuple(rgbd.camera.target_m):
        raise ValueError("camera target differs from RGB-D source contract")
    resolution = tuple(
        int(value)
        for value in _vector(observation["resolution_px"], 2, "resolution_px")
    )
    if resolution != tuple(rgbd.camera.resolution):
        raise ValueError("camera resolution differs from RGB-D source contract")
    if observation["camera_cv_axes"] != ["x_right", "y_down", "z_forward"]:
        raise ValueError("camera CV axes differ")
    if observation["channels_exactly"] != ["rgb", "distance_to_image_plane"]:
        raise ValueError("observe-only channels must be ordinary RGB and depth")
    if not math.isclose(
        _finite(observation["focal_length_mm"], "focal_length_mm"),
        RGBD_CAMERA_FOCAL_LENGTH_MM,
        abs_tol=1.0e-12,
    ):
        raise ValueError("camera focal length differs from raw runtime")
    if not math.isclose(
        _finite(
            observation["horizontal_aperture_mm"], "horizontal_aperture_mm"
        ),
        RGBD_CAMERA_HORIZONTAL_APERTURE_MM,
        abs_tol=1.0e-12,
    ):
        raise ValueError("camera aperture differs from raw runtime")
    if tuple(_vector(observation["clipping_range_m"], 2, "clipping_range_m")) != tuple(
        RGBD_CAMERA_CLIPPING_RANGE_M
    ):
        raise ValueError("camera clipping range differs from raw runtime")
    rt_subframes = _finite(
        observation["capture_rt_subframes"], "capture_rt_subframes"
    )
    if not rt_subframes.is_integer() or rt_subframes < 1:
        raise ValueError("capture_rt_subframes must be a positive integer")
    minimum_endpoint_points = _finite(
        observation["minimum_endpoint_points"], "minimum_endpoint_points"
    )
    if not minimum_endpoint_points.is_integer() or minimum_endpoint_points < 1:
        raise ValueError("minimum_endpoint_points must be a positive integer")
    if _finite(
        observation["minimum_key_width_px"], "minimum_key_width_px"
    ) <= 0.0:
        raise ValueError("minimum_key_width_px must be positive")
    if _finite(
        observation["minimum_foreground_depth_delta_m"],
        "minimum_foreground_depth_delta_m",
    ) <= 0.0:
        raise ValueError("minimum_foreground_depth_delta_m must be positive")

    table_top = tabletop.table.center_m[2] + 0.5 * tabletop.table.size_m[2]
    fixture_top = (
        tabletop.fixed_endpoint.fixture_center_m[2]
        + 0.5 * tabletop.fixed_endpoint.fixture_size_m[2]
    )
    products = _mapping(geometry_audit.get("products"), "geometry_audit.products")
    for endpoint, support_z in (("plug", table_top), ("receptacle", fixture_top)):
        product = _mapping(products.get(endpoint), f"geometry_audit.products.{endpoint}")
        bbox = _mapping(product.get("bounding_box_mm"), f"{endpoint}.bounding_box_mm")
        translation = _vector(
            scene_truth[f"{endpoint}_world_translation_m"],
            3,
            f"scene_generation_truth.{endpoint}_world_translation_m",
        )
        rotation = _rotation_xyz_degrees(
            scene_truth[f"{endpoint}_world_rotation_xyz_deg"],
            f"scene_generation_truth.{endpoint}_world_rotation_xyz_deg",
        )
        world_minimum, world_maximum = _world_aabb_from_supplier_bbox(
            bbox, translation, rotation
        )
        if not math.isclose(
            float(world_minimum[2]), support_z, abs_tol=2.0e-7, rel_tol=0.0
        ):
            raise ValueError(f"{endpoint} does not rest on its declared support")
        workspace_minimum, workspace_maximum = _parse_aabb(
            observation[f"{endpoint}_workspace_world_aabb_m"],
            f"observation_contract.{endpoint}_workspace_world_aabb_m",
        )
        if np.any(world_minimum < workspace_minimum) or np.any(
            world_maximum > workspace_maximum
        ):
            raise ValueError(f"{endpoint} CAD bounds escape frozen workspace")

    formal_gate = _mapping(document["formal_gate"], "formal_gate")
    _exact_keys(
        formal_gate,
        (
            "required_outputs",
            "key_observation_required",
            "ambiguous_key_result",
            "unobserved_key_result",
            "missing_endpoint_result",
            "miss_authorizes_motion",
            "pose_result_from_scene_generation_truth_forbidden",
        ),
        "formal_gate",
    )
    if not isinstance(formal_gate["required_outputs"], list) or len(
        formal_gate["required_outputs"]
    ) != 8:
        raise ValueError("formal gate must name the eight endpoint outputs")
    if formal_gate["unobserved_key_result"] != "MISS_KEY_NOT_OBSERVABLE":
        raise ValueError("unobserved key must fail as MISS_KEY_NOT_OBSERVABLE")
    for name in (
        "key_observation_required",
        "pose_result_from_scene_generation_truth_forbidden",
    ):
        if _boolean(formal_gate[name], f"formal_gate.{name}") is not True:
            raise ValueError(f"formal_gate.{name} must be true")
    if _boolean(
        formal_gate["miss_authorizes_motion"],
        "formal_gate.miss_authorizes_motion",
    ) is not False:
        raise ValueError("MISS must not authorize motion")

    firewall = _mapping(document["truth_firewall"], "truth_firewall")
    _exact_keys(
        firewall,
        (
            "provider_inputs_exactly",
            "forbidden_provider_inputs",
            "posthoc_truth_evaluation_allowed",
        ),
        "truth_firewall",
    )
    if "object_pose_truth" not in firewall["forbidden_provider_inputs"]:
        raise ValueError("truth firewall must forbid object pose truth")
    if "semantic_segmentation_truth" not in firewall["forbidden_provider_inputs"]:
        raise ValueError("truth firewall must forbid semantic truth")

    output = _mapping(document["output"], "output")
    _exact_keys(
        output,
        (
            "root_directory",
            "existing_output_policy",
            "capture_report_filename",
            "provider_input_filename",
            "observability_report_filename",
        ),
        "output",
    )
    if output["existing_output_policy"] != "REFUSE_OVERWRITE":
        raise ValueError("observe-only output must refuse overwrite")
    for name in (
        "capture_report_filename",
        "provider_input_filename",
        "observability_report_filename",
    ):
        filename = Path(_text(output[name], f"output.{name}"))
        if filename.name != str(filename) or filename.suffix != ".json":
            raise ValueError(f"output.{name} must be a local JSON filename")

    world_from_camera_cv(
        observation["camera_eye_world_m"],
        observation["camera_target_world_m"],
    )
    return LoadedObserveOnlyConfig(
        path=path,
        document=document,
        source_paths=source_paths,
        asset_paths=asset_paths,
        tabletop=tabletop,
        rgbd=rgbd,
        te_contract=te_contract,
        geometry_audit=geometry_audit,
    )


def _resolve_output_directory(
    repository: Path,
    loaded: LoadedObserveOnlyConfig,
    requested: str,
) -> Path:
    raw_root = Path(loaded.document["output"]["root_directory"])
    if raw_root.is_absolute() or ".." in raw_root.parts:
        raise ValueError("configured output root must be repository-relative")
    root = (repository / raw_root).resolve()
    raw_requested = Path(_text(requested, "output_directory"))
    candidate = (
        raw_requested.resolve()
        if raw_requested.is_absolute()
        else (repository / raw_requested).resolve()
    )
    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise ValueError("output directory is outside the frozen output root") from error
    if not relative.parts:
        raise ValueError("output directory must be one run below the output root")
    if candidate.exists():
        raise ValueError("output directory already exists; overwrite is forbidden")
    return candidate


def _author_te_visual_scene(
    *,
    stage,
    loaded: LoadedObserveOnlyConfig,
    add_reference_to_stage,
    Gf,
    UsdGeom,
) -> dict[str, Any]:
    assets = loaded.document["assets"]
    truth = loaded.document["scene_generation_truth"]
    root_path = "/".join(str(assets["plug"]["pose_prim_path"]).split("/")[:-1])
    if stage.GetPrimAtPath(root_path).IsValid():
        raise RuntimeError("TE observe-only scene root already exists")
    UsdGeom.Xform.Define(stage, root_path)
    records = {}
    for endpoint in ("plug", "receptacle"):
        asset = assets[endpoint]
        pose_path = asset["pose_prim_path"]
        reference_path = asset["reference_prim_path"]
        pose_xform = UsdGeom.Xform.Define(stage, pose_path)
        translation = _vector(
            truth[f"{endpoint}_world_translation_m"],
            3,
            f"{endpoint} translation",
        )
        rotation = _vector(
            truth[f"{endpoint}_world_rotation_xyz_deg"],
            3,
            f"{endpoint} rotation",
        )
        pose_xform.AddTranslateOp().Set(Gf.Vec3d(*translation))
        pose_xform.AddRotateXYZOp().Set(Gf.Vec3f(*rotation))
        add_reference_to_stage(
            str(loaded.asset_paths[endpoint]["render_asset"]),
            reference_path,
        )
        reference_prim = stage.GetPrimAtPath(reference_path)
        if not reference_prim.IsValid():
            raise RuntimeError(f"{endpoint} official visual reference is missing")
        records[endpoint] = {
            "pose_prim_path": pose_path,
            "reference_prim_path": reference_path,
            "scene_generation_transform_write_count": 2,
        }
    return {
        "scene_root_prim_path": root_path,
        "endpoints": records,
        "object_pose_read_calls": 0,
        "object_pose_writes_after_endpoint_observation_start": 0,
    }


def _camera_calibration_record(
    loaded: LoadedObserveOnlyConfig,
    raw_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    from kcg_connector.te_rgbd_observability import world_from_camera_cv

    observation = loaded.document["observation_contract"]
    camera_metrics = _mapping(raw_metrics.get("camera"), "raw_metrics.camera")
    intrinsics = np.asarray(camera_metrics.get("intrinsics"), dtype=np.float64)
    if intrinsics.shape != (3, 3) or not np.all(np.isfinite(intrinsics)):
        raise RuntimeError("captured camera intrinsics are not a finite 3x3 matrix")
    world_from_camera = world_from_camera_cv(
        observation["camera_eye_world_m"],
        observation["camera_target_world_m"],
    )
    camera_from_world = np.linalg.inv(world_from_camera)
    record = {
        "calibration_id": observation["calibration_id"],
        "calibration_scope": "SIMULATION_ONLY_FROZEN_GLOBAL_CAMERA",
        "world_frame_id": observation["world_frame_id"],
        "camera_frame_id": observation["camera_frame_id"],
        "transform_convention": "parent_T_child",
        "camera_cv_axes": list(observation["camera_cv_axes"]),
        "intrinsics_3x3": intrinsics.tolist(),
        "world_from_camera_cv_row_major": world_from_camera.ravel().tolist(),
        "camera_cv_from_world_row_major": camera_from_world.ravel().tolist(),
        "resolution_px": list(observation["resolution_px"]),
        "clipping_range_m": list(observation["clipping_range_m"]),
    }
    record["calibration_sha256"] = _json_sha256(record)
    return record


def _relative_to_repository(repository: Path, path: Path) -> str:
    return str(path.resolve().relative_to(repository))


def _build_provider_input(
    *,
    repository: Path,
    loaded: LoadedObserveOnlyConfig,
    output_directory: Path,
    capture_id: str,
    capture_started_at_utc: str,
    capture_completed_at_utc: str,
    static_capture,
    observed_capture,
    observability: Mapping[str, Any],
) -> dict[str, Any]:
    static_rgb_path = (
        output_directory / "background" / loaded.rgbd.output.rgb_filename
    )
    static_depth_path = (
        output_directory
        / "background"
        / loaded.rgbd.output.depth_numpy_filename
    )
    rgb_path = (
        output_directory / "observation" / loaded.rgbd.output.rgb_filename
    )
    depth_path = (
        output_directory
        / "observation"
        / loaded.rgbd.output.depth_numpy_filename
    )
    if not all(
        path.is_file()
        for path in (static_rgb_path, static_depth_path, rgb_path, depth_path)
    ):
        raise RuntimeError("raw RGB-D artifacts are missing")
    calibration = _camera_calibration_record(loaded, observed_capture.metrics)
    static_calibration = _camera_calibration_record(
        loaded, static_capture.metrics
    )
    if static_calibration != calibration:
        raise RuntimeError("static and observed captures use different calibration")
    observation = loaded.document["observation_contract"]
    te_frames = _mapping(
        loaded.te_contract["assembly_completion_contract"]["supplier_cad_frames"],
        "te supplier frames",
    )
    provider_models = {}
    for endpoint, frame_key, key_name in (
        ("plug", "plug", "unique_main_key_local"),
        ("receptacle", "receptacle", "unique_main_keyway_local"),
    ):
        asset = loaded.document["assets"][endpoint]
        supplier_frame = _mapping(te_frames[frame_key], f"supplier frame {endpoint}")
        provider_models[endpoint] = {
            "role": asset["role"],
            "model_id": asset["model_id"],
            "identity": asset["identity"],
            "registration_cad": asset["registration_cad"],
            "registration_cad_sha256": asset["registration_cad_sha256"],
            "supplier_frame_origin": supplier_frame["origin"],
            "outward_axis_local": list(supplier_frame["outward_axis_local"]),
            key_name: list(supplier_frame[key_name]),
        }
    relation_path = loaded.source_paths.get(
        "body_rear_face_to_main_key_contract"
    )
    if relation_path is not None:
        relation = _load_yaml(relation_path, "body_rear_face_to_main_key")
        provider_models["plug"]["body_rear_face_to_main_key_relation"] = {
            "relation_id": relation["relation_id"],
            "contract_sha256": _sha256(relation_path),
            "reference_part": relation["identity"]["reference_part"],
            "forbidden_reference_part": relation["identity"][
                "forbidden_reference_part"
            ],
            "frames": relation["frames"],
            "freeze_contract": relation["freeze_contract"],
            "branch_continuity": relation["branch_continuity"],
        }
    tabletop = loaded.tabletop
    provider_input = {
        "schema_version": PROVIDER_INPUT_SCHEMA_VERSION,
        "capture_id": capture_id,
        "capture_time": {
            "clock_domain": "host_utc",
            "capture_started_at_utc": capture_started_at_utc,
            "static_frame_timestamp_utc": static_capture.metrics.get(
                "capture_timestamp_utc"
            ),
            "observed_frame_timestamp_utc": observed_capture.metrics.get(
                "capture_timestamp_utc"
            ),
            "capture_completed_at_utc": capture_completed_at_utc,
        },
        "capture_contract_sha256": _sha256(loaded.path),
        "ordinary_rgb": {
            "path_relative_to_manifest": str(
                rgb_path.relative_to(output_directory)
            ),
            "sha256": _sha256(rgb_path),
            "encoding": "png_rgb_uint8",
            "shape": list(np.asarray(observed_capture.rgb).shape),
        },
        "ordinary_depth": {
            "path_relative_to_manifest": str(
                depth_path.relative_to(output_directory)
            ),
            "sha256": _sha256(depth_path),
            "encoding": "npy_distance_to_image_plane_m_float32",
            "shape": list(np.asarray(observed_capture.depth).shape),
        },
        "ordinary_static_scene_depth": {
            "path_relative_to_manifest": str(
                static_depth_path.relative_to(output_directory)
            ),
            "sha256": _sha256(static_depth_path),
            "encoding": "npy_distance_to_image_plane_m_float32",
            "shape": list(np.asarray(static_capture.depth).shape),
        },
        "camera_calibration": calibration,
        "known_static_scene_geometry": {
            "table": {
                "center_world_m": list(tabletop.table.center_m),
                "size_m": list(tabletop.table.size_m),
            },
            "fixture": {
                "center_world_m": list(
                    tabletop.fixed_endpoint.fixture_center_m
                ),
                "size_m": list(tabletop.fixed_endpoint.fixture_size_m),
            },
        },
        "frozen_endpoint_workspaces_world_aabb_m": {
            "plug": observation["plug_workspace_world_aabb_m"],
            "receptacle": observation[
                "receptacle_workspace_world_aabb_m"
            ],
        },
        "te_cad_models": provider_models,
        "formal_gate": dict(loaded.document["formal_gate"]),
        "truth_firewall": dict(loaded.document["truth_firewall"]),
        "provider_callable_contract": {
            "input": (
                "provider_input mapping plus static/observed RGB-D arrays"
            ),
            "output": (
                "one vision pose pair or one formal_gate MISS code"
            ),
            "implementation_status": (
                "OBSERVABILITY_GATE_EXECUTED_FULL_CAD_PROVIDER_NOT_IMPLEMENTED"
            ),
        },
        "observability": dict(observability),
        "pose_result": None,
        "control_authorized": False,
    }
    forbidden_keys = {
        "scene_generation_truth",
        "pose_prim_path",
        "reference_prim_path",
        "plug_world_translation_m",
        "receptacle_world_translation_m",
    }

    def contains_forbidden_key(value: Any) -> bool:
        if isinstance(value, Mapping):
            return any(
                key in forbidden_keys or contains_forbidden_key(child)
                for key, child in value.items()
            )
        if isinstance(value, (list, tuple)):
            return any(contains_forbidden_key(child) for child in value)
        return False

    forbidden_exact_values = {
        str(loaded.document["scene_generation_truth"]["scene_pose_id"]),
        *(
            str(loaded.document["assets"][endpoint][field])
            for endpoint in ("plug", "receptacle")
            for field in ("pose_prim_path", "reference_prim_path")
        ),
    }

    def contains_forbidden_value(value: Any) -> bool:
        if isinstance(value, Mapping):
            return any(contains_forbidden_value(child) for child in value.values())
        if isinstance(value, (list, tuple)):
            return any(contains_forbidden_value(child) for child in value)
        return isinstance(value, str) and value in forbidden_exact_values

    if contains_forbidden_key(provider_input) or contains_forbidden_value(
        provider_input
    ):
        raise RuntimeError("provider input leaked scene-generation truth")
    return provider_input


def _arguments(repository: Path):
    parser = argparse.ArgumentParser(
        description="Capture one official-TE ordinary RGB-D observe-only run"
    )
    parser.add_argument(
        "--config",
        default=str(repository / DEFAULT_CONFIG_RELATIVE),
    )
    parser.add_argument("--output-directory")
    parser.add_argument("--capture-id")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument(
        "--validate-config-only",
        action="store_true",
        help="validate paths, hashes, frames and truth boundary without Isaac",
    )
    arguments = parser.parse_args()
    if not arguments.validate_config_only:
        if not arguments.output_directory:
            parser.error("--output-directory is required for capture")
        if not arguments.capture_id:
            parser.error("--capture-id is required for capture")
    return arguments


def main() -> int:
    repository = Path(__file__).resolve().parents[3]
    arguments = _arguments(repository)
    try:
        loaded = load_observe_only_config(
            repository, Path(arguments.config)
        )
        config_sha = _sha256(loaded.path)
        source_hashes = {
            name: _sha256(path)
            for name, path in loaded.source_paths.items()
        }
        if arguments.validate_config_only:
            validation = {
                "schema_version": SCHEMA_VERSION,
                "config_path": _relative_to_repository(repository, loaded.path),
                "config_sha256": config_sha,
                "source_contract_sha256": source_hashes,
                "asset_sha256": {
                    endpoint: {
                        kind: _sha256(path)
                        for kind, path in paths.items()
                    }
                    for endpoint, paths in loaded.asset_paths.items()
                },
                "hardware_authorized": False,
                "isaac_started": False,
                "passed": True,
            }
            print(json.dumps(validation, allow_nan=False, sort_keys=True))
            return 0
        capture_id = _identifier(arguments.capture_id, "capture_id")
        output_directory = _resolve_output_directory(
            repository, loaded, arguments.output_directory
        )
    except Exception as exception:
        print(
            json.dumps(
                {
                    "passed": False,
                    "isaac_started": False,
                    "error": f"{type(exception).__name__}: {exception}",
                },
                allow_nan=False,
                sort_keys=True,
            ),
            flush=True,
        )
        return 2

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": not arguments.gui,
            "multi_gpu": False,
            "active_gpu": 0,
            "physics_gpu": 0,
        }
    )
    passed = False
    report: dict[str, Any] = {
        "schema_version": CAPTURE_REPORT_SCHEMA_VERSION,
        "capture_id": capture_id,
        "claim_scope": "RAW_RGBD_OBSERVE_ONLY_NO_POSE_OR_CONTROL_CLAIM",
        "hardware_authorized": False,
        "robot_present": False,
        "robot_motion_command_count": 0,
        "semantic_annotator_used": False,
        "object_pose_read_calls": 0,
        "passed": False,
    }
    try:
        from PIL import Image

        from isaacsim.core.api import World
        from isaacsim.core.utils.stage import (
            add_reference_to_stage,
            get_current_stage,
        )
        from isaacsim.sensors.camera import Camera
        import omni.replicator.core as rep
        from omni.physx.scripts import physicsUtils
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

        from kcg_connector.d38999_tabletop_scene import (
            author_d38999_tabletop_environment,
        )
        from kcg_connector.isaac_d38999_rgbd_runtime import (
            capture_d38999_rgbd_raw_formal,
        )
        from kcg_connector.te_rgbd_observability import (
            evaluate_te_rgbd_observability,
            workspaces_from_config,
            world_from_camera_cv,
        )

        World.clear_instance()
        world = World(
            stage_units_in_meters=1.0,
            physics_dt=1.0 / loaded.tabletop.physics.rate_hz,
            rendering_dt=1.0 / 60.0,
            backend="numpy",
            device="cpu",
        )
        stage = get_current_stage()
        environment = author_d38999_tabletop_environment(
            stage,
            loaded.tabletop,
            Gf=Gf,
            Sdf=Sdf,
            UsdGeom=UsdGeom,
            UsdPhysics=UsdPhysics,
            UsdShade=UsdShade,
            physics_utils=physicsUtils,
        )
        world.reset()
        world.get_physics_context().set_gravity(
            loaded.tabletop.physics.gravity_m_s2
        )
        world.pause()
        simulation_app.update()

        capture_started_at_utc = datetime.now(timezone.utc).isoformat()
        capture_bindings = {
            "Camera": Camera,
            "Gf": Gf,
            "Image": Image,
            "Usd": Usd,
            "UsdGeom": UsdGeom,
            "UsdLux": UsdLux,
            "rep": rep,
        }
        observation_contract = loaded.document["observation_contract"]
        static_capture = capture_d38999_rgbd_raw_formal(
            bindings=capture_bindings,
            simulation_app=simulation_app,
            world=world,
            stage=stage,
            tabletop=loaded.tabletop,
            rgbd=loaded.rgbd,
            output_dir=output_directory / "background",
            camera_clipping_range_m=observation_contract[
                "clipping_range_m"
            ],
            rt_subframes=int(observation_contract["capture_rt_subframes"]),
        )
        if static_capture.passed is not True:
            raise RuntimeError("ordinary static-scene RGB-D capture failed")
        authored_scene = _author_te_visual_scene(
            stage=stage,
            loaded=loaded,
            add_reference_to_stage=add_reference_to_stage,
            Gf=Gf,
            UsdGeom=UsdGeom,
        )
        simulation_app.update()
        observed_capture = capture_d38999_rgbd_raw_formal(
            bindings=capture_bindings,
            simulation_app=simulation_app,
            world=world,
            stage=stage,
            tabletop=loaded.tabletop,
            rgbd=loaded.rgbd,
            output_dir=output_directory / "observation",
            camera_clipping_range_m=observation_contract[
                "clipping_range_m"
            ],
            rt_subframes=int(observation_contract["capture_rt_subframes"]),
        )
        capture_completed_at_utc = datetime.now(timezone.utc).isoformat()
        if observed_capture.passed is not True:
            raise RuntimeError("ordinary endpoint RGB-D capture failed")
        static_intrinsics = np.asarray(
            static_capture.metrics["camera"]["intrinsics"], dtype=np.float64
        )
        observed_intrinsics = np.asarray(
            observed_capture.metrics["camera"]["intrinsics"], dtype=np.float64
        )
        if not np.array_equal(static_intrinsics, observed_intrinsics):
            raise RuntimeError("static and observed camera intrinsics differ")
        observability = evaluate_te_rgbd_observability(
            rgb=observed_capture.rgb,
            depth_m=observed_capture.depth,
            static_depth_m=static_capture.depth,
            intrinsics=observed_intrinsics,
            world_from_camera=world_from_camera_cv(
                observation_contract["camera_eye_world_m"],
                observation_contract["camera_target_world_m"],
            ),
            workspaces=workspaces_from_config(
                observation_contract,
                minimum_points=int(
                    observation_contract["minimum_endpoint_points"]
                ),
            ),
            minimum_key_width_px=float(
                observation_contract["minimum_key_width_px"]
            ),
            minimum_foreground_depth_delta_m=float(
                observation_contract["minimum_foreground_depth_delta_m"]
            ),
        )
        observability_path = output_directory / loaded.document["output"][
            "observability_report_filename"
        ]
        observability_path.write_text(
            json.dumps(
                observability, allow_nan=False, indent=2, sort_keys=True
            )
            + "\n",
            encoding="utf-8",
        )
        provider_input = _build_provider_input(
            repository=repository,
            loaded=loaded,
            output_directory=output_directory,
            capture_id=capture_id,
            capture_started_at_utc=capture_started_at_utc,
            capture_completed_at_utc=capture_completed_at_utc,
            static_capture=static_capture,
            observed_capture=observed_capture,
            observability=observability,
        )
        provider_input_path = output_directory / loaded.document["output"][
            "provider_input_filename"
        ]
        provider_input_path.write_text(
            json.dumps(
                provider_input, allow_nan=False, indent=2, sort_keys=True
            )
            + "\n",
            encoding="utf-8",
        )
        report.update(
            {
                "config": {
                    "path": _relative_to_repository(repository, loaded.path),
                    "sha256": config_sha,
                },
                "source_contract_sha256": source_hashes,
                "asset_identity": {
                    endpoint: {
                        "identity": loaded.document["assets"][endpoint][
                            "identity"
                        ],
                        "render_asset": loaded.document["assets"][endpoint][
                            "render_asset"
                        ],
                        "render_asset_sha256": loaded.document["assets"][
                            endpoint
                        ]["render_asset_sha256"],
                        "registration_cad": loaded.document["assets"][endpoint][
                            "registration_cad"
                        ],
                        "registration_cad_sha256": loaded.document["assets"][
                            endpoint
                        ]["registration_cad_sha256"],
                    }
                    for endpoint in ("plug", "receptacle")
                },
                "environment_authoring": environment,
                "scene_authoring": authored_scene,
                "scene_generation_truth_postcapture_record": dict(
                    loaded.document["scene_generation_truth"]
                ),
                "raw_capture": {
                    "static_background": static_capture.metrics,
                    "endpoint_observation": observed_capture.metrics,
                },
                "raw_artifacts": {
                    "static_rgb": {
                        "path": str(
                            (
                                Path("background")
                                / loaded.rgbd.output.rgb_filename
                            )
                        ),
                        "sha256": _sha256(
                            output_directory
                            / "background"
                            / loaded.rgbd.output.rgb_filename
                        ),
                    },
                    "static_depth": {
                        "path": str(
                            Path("background")
                            / loaded.rgbd.output.depth_numpy_filename
                        ),
                        "sha256": _sha256(
                            output_directory
                            / "background"
                            / loaded.rgbd.output.depth_numpy_filename
                        ),
                    },
                    "observed_rgb": {
                        "path": str(
                            Path("observation")
                            / loaded.rgbd.output.rgb_filename
                        ),
                        "sha256": _sha256(
                            output_directory
                            / "observation"
                            / loaded.rgbd.output.rgb_filename
                        ),
                    },
                    "observed_depth": {
                        "path": str(
                            Path("observation")
                            / loaded.rgbd.output.depth_numpy_filename
                        ),
                        "sha256": _sha256(
                            output_directory
                            / "observation"
                            / loaded.rgbd.output.depth_numpy_filename
                        ),
                    },
                },
                "observability": {
                    "path": observability_path.name,
                    "sha256": _sha256(observability_path),
                    "result": observability,
                },
                "provider_input": {
                    "path": provider_input_path.name,
                    "sha256": _sha256(provider_input_path),
                    "scene_generation_truth_included": False,
                },
                "formal_pose_result": {
                    "status": observability["status"],
                    "pose": None,
                    "keyed_orientation_observed": False,
                    "control_authorized": False,
                },
                "capture_started_at_utc": capture_started_at_utc,
                "capture_completed_at_utc": capture_completed_at_utc,
                "object_pose_writes_after_endpoint_observation_start": 0,
                "experiment_contract": dict(
                    loaded.document["experiment_contract"]
                ),
                "passed": True,
            }
        )
        report_path = output_directory / loaded.document["output"][
            "capture_report_filename"
        ]
        report_path.write_text(
            json.dumps(report, allow_nan=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        passed = True
        print(
            json.dumps(
                {
                    "capture_id": capture_id,
                    "output_directory": _relative_to_repository(
                        repository, output_directory
                    ),
                    "raw_rgbd_captured": True,
                    "formal_pose_status": observability["status"],
                    "control_authorized": False,
                    "passed": True,
                },
                allow_nan=False,
                sort_keys=True,
            ),
            flush=True,
        )
    except BaseException as exception:
        report.update(
            {
                "error": f"{type(exception).__name__}: {exception}",
                "traceback": traceback.format_exc(),
                "passed": False,
            }
        )
        if not output_directory.exists():
            output_directory.mkdir(parents=True, exist_ok=False)
        failure_path = output_directory / loaded.document["output"][
            "capture_report_filename"
        ]
        if not failure_path.exists():
            failure_path.write_text(
                json.dumps(report, allow_nan=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
        traceback.print_exc()
        print(
            json.dumps(
                {
                    "capture_id": capture_id,
                    "passed": False,
                    "error": report["error"],
                },
                allow_nan=False,
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        simulation_app.close(exit_code=0 if passed else 1)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
