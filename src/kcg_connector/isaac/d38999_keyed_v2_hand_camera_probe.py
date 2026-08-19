#!/usr/bin/env python3

"""Fixed Palm/Wrist RGB-D probe for the public-spec keyed-v2 asset.

Two fresh stages are used: a repeated fixed Pregrasp view, then the existing
safe Home-to-Pregrasp three-segment path. Both cameras are non-physical
children of handbase and their v5 T_HC is authored once per fresh stage. The
probe stops before grasp/insertion and can never authorize control.

Runtime reads are limited to RGB, planar depth, robot joints for endpoint
tracking, and handbase/camera transforms for T_HC rigidity. Semantic labels,
keyed-object poses, contacts, and collider identity are never requested.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import traceback
from typing import Any, Sequence

import numpy as np
import yaml

from kcg_connector.d38999_keyed_public_spec_v2 import (
    PLUG_MODEL_ID,
    RECOMMENDED_ASSET_NAME,
    ROOT_PRIM,
)
from kcg_connector.d38999_tabletop_pick import interpolate_arm


SCHEMA_VERSION = "kcg_d38999_keyed_v2_hand_camera_probe_v1"
PROBE_SCOPE = "SIMULATION_ONLY_FIXED_HAND_CAMERA_PROBE"
DEFAULT_CONFIG = Path(
    "src/kcg_connector/config/d38999_keyed_v2_hand_camera_probe_v1.yaml"
)
ROBOT_ASSET = Path("artifacts/kcg_connector/isaac/robot/handarm/handarm.usda")
KEYED_ASSET = Path(
    "artifacts/kcg_connector/isaac/keyed_v2_contact_offset_r2"
) / RECOMMENDED_ASSET_NAME
CAMERAS = ("palm", "wrist")
CV_FROM_USD = np.diag((1.0, -1.0, -1.0, 1.0))
V5_EYE_TARGET = {
    "palm": ((0.0, 0.0, 0.315), (0.0, 0.0, 0.448)),
    "wrist": ((-0.150, 0.0, 0.060), (-0.090, 0.0, 0.480)),
}
SAFE_SEGMENTS = (
    (
        "home_to_safe_mid",
        6.2,
        (-0.1133152125, 0.2419650715, -0.1716717785, -0.3550775790,
         0.0852878585, 0.9833095885, -0.0414886690),
    ),
    (
        "safe_mid_to_high_approach",
        3.7,
        (-0.1813043400, 0.3871441144, -0.2746748456, -0.5681241264,
         0.1364605736, 1.5732953416, -0.0663818704),
    ),
    (
        "high_approach_to_pregrasp",
        2.5,
        (-0.2266304250, 0.4839301430, -0.3433435570, -0.7101551580,
         0.1705757170, 1.9666191770, -0.0829773380),
    ),
)
EPISODE_IDS = (
    "stationary_pregrasp_repeat",
    "home_safe_three_segment_to_pregrasp",
)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def safe_new_output_dir(path: Path | str) -> Path:
    output = Path(path).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite probe output: {output}")
    return output


def camera_cv_pose_from_eye_target(
    eye: Sequence[float], target: Sequence[float]
) -> np.ndarray:
    """Exact v5 CV convention: x-right, y-down, z-forward."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if eye.shape != (3,) or target.shape != (3,) or not np.all(
        np.isfinite(np.concatenate((eye, target)))
    ):
        raise ValueError("camera eye/target must be finite 3-vectors")
    forward = target - eye
    norm = float(np.linalg.norm(forward))
    if norm <= 1.0e-9:
        raise ValueError("camera eye and target must differ")
    forward /= norm
    right = np.cross((0.0, 1.0, 0.0), forward)
    right /= np.linalg.norm(right)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.column_stack((right, np.cross(forward, right), forward))
    pose[:3, 3] = eye
    return pose


def camera_cv_pose_to_usd(t_hc: Any) -> np.ndarray:
    pose = np.asarray(t_hc, dtype=np.float64)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError("T_HC must be a finite 4x4 transform")
    return pose @ CV_FROM_USD


def mount_residual(expected_t_hc: Any, t_wh: Any, t_wu: Any) -> dict[str, Any]:
    """Compute camera/hand residual; this API accepts no object transform."""
    expected = np.asarray(expected_t_hc, dtype=np.float64)
    hand = np.asarray(t_wh, dtype=np.float64)
    usd_camera = np.asarray(t_wu, dtype=np.float64)
    if any(value.shape != (4, 4) for value in (expected, hand, usd_camera)):
        raise ValueError("mount inputs must be 4x4 transforms")
    observed = np.linalg.inv(hand) @ usd_camera @ CV_FROM_USD
    relative = expected[:3, :3].T @ observed[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return {
        "translation_m": float(np.linalg.norm(observed[:3, 3] - expected[:3, 3])),
        "rotation_rad": float(math.acos(cosine)),
        "observed_T_HC_cv": observed,
    }


def minimum_jerk_arm(start, target, fraction) -> np.ndarray:
    """Thin array adapter around the existing frozen pick interpolation."""
    first = tuple(float(value) for value in np.asarray(start).ravel())
    second = tuple(float(value) for value in np.asarray(target).ravel())
    return np.asarray(interpolate_arm(first, second, fraction), dtype=np.float64)


def gravity_z_scalar(value: Any) -> float:
    """Translate the config vector into Isaac 6's scalar Z-gravity API."""
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("gravity_m_s2 must be a finite XYZ vector")
    if not np.allclose(vector[:2], 0.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("Isaac scalar gravity API only supports the stage Z axis")
    return float(vector[2])


def json_safe(value: Any) -> Any:
    """Keep failure evidence writable without treating NaN as valid data."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        scalar = float(value)
        return scalar if math.isfinite(scalar) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def evaluate_rgbd_quality(
    endpoints: Sequence[dict[str, Any]],
    required_endpoint_names: Sequence[str],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    """Gate only declared observation endpoints; retain all others as evidence."""
    required = tuple(str(name) for name in required_endpoint_names)
    if not required or len(set(required)) != len(required):
        raise ValueError("RGB-D quality endpoints must be non-empty and unique")
    observed = {}
    for endpoint in endpoints:
        name = str(endpoint["name"])
        if name in observed:
            raise ValueError(f"duplicate captured endpoint: {name}")
        observed[name] = all(
            camera["rgb_dynamic_range"]
            >= thresholds["minimum_rgb_dynamic_range"]
            and camera["rgb_standard_deviation"]
            >= thresholds["minimum_rgb_standard_deviation"]
            and camera["valid_depth_pixels"]
            >= thresholds["minimum_valid_depth_pixels"]
            for camera in endpoint["cameras"].values()
        )
    missing = [name for name in required if name not in observed]
    if missing:
        raise ValueError(f"required RGB-D endpoint was not captured: {missing}")
    return {
        "passed": all(observed[name] for name in required),
        "required_endpoints": list(required),
        "quality_by_captured_endpoint": observed,
    }


def load_probe_contract(path: Path | str | None = None) -> dict[str, Any]:
    """Validate only identity, camera, motion, truth, and authorization gates."""
    path = (repository_root() / DEFAULT_CONFIG) if path is None else Path(path)
    contract = yaml.safe_load(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(contract, dict) or contract.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected hand-camera probe schema")
    if contract.get("enabled") is not True or contract.get("mode") != PROBE_SCOPE:
        raise ValueError("probe must remain enabled and simulation-only")
    assets = contract["assets"]
    if Path(assets["robot"]) != ROBOT_ASSET or Path(assets["keyed_pair"]) != KEYED_ASSET:
        raise ValueError("probe asset identity changed")
    if assets["keyed_asset_root_prim"] != ROOT_PRIM or assets["keyed_model_id"] != PLUG_MODEL_ID:
        raise ValueError("keyed-v2 model/root identity changed")

    rig = contract["camera_rig"]
    if (
        rig["mount_contract"] != "SIM_VISUAL_MOUNT_CANDIDATE_FIXED_T_HC_V5"
        or rig["resolution_px"] != [1280, 720]
        or rig["channels_exactly"] != ["rgb", "distance_to_image_plane"]
        or not math.isclose(float(rig["focal_length_mm"]), 24.0)
        or not math.isclose(
            float(rig["vertical_aperture_mm"]), 20.955 * 720.0 / 1280.0
        )
    ):
        raise ValueError("camera rig must retain the v5 24mm 1280x720 RGB-D contract")
    for name in CAMERAS:
        declared = np.asarray(rig[name]["T_HC_cv"], dtype=np.float64)
        if not np.allclose(
            declared, camera_cv_pose_from_eye_target(*V5_EYE_TARGET[name]),
            rtol=0.0, atol=1.0e-14,
        ):
            raise ValueError(f"{name} explicit T_HC changed")

    motion = contract["motion"]
    actual_segments = tuple(
        (item["name"], float(item["duration_s"]), tuple(item["target_arm_rad"]))
        for item in motion["approach_segments"]
    )
    if int(motion["rate_hz"]) != 240 or actual_segments != SAFE_SEGMENTS:
        raise ValueError("safe approach duration/target changed")
    episodes = contract["episodes"]
    if tuple(item["id"] for item in episodes) != EPISODE_IDS or not all(
        item.get("fresh_stage") is True for item in episodes
    ):
        raise ValueError("two fresh episodes are required")
    for episode in episodes:
        captures = tuple(episode["capture_endpoints"])
        required = tuple(episode["rgbd_quality_required_endpoints"])
        if not required or not set(required).issubset(captures):
            raise ValueError("RGB-D quality endpoints must be captured endpoints")
    capture = contract["capture"]
    if (
        capture["wrist_shadow_allowed"] is not False
        or capture["palm_shadow_occlusion_policy"] != "UNKNOWN_FAIL_CLOSED"
        or capture["palm_shadow_expected_rejection_code"]
        != "KEY_REGION_OCCLUSION_UNKNOWN"
    ):
        raise ValueError("Palm/Wrist shadow boundary changed")
    authorization = contract["authorization"]
    if any(
        authorization[name] is not False
        for name in (
            "control_authorized", "visual_control_authorized",
            "insertion_control_authorized", "grasp_authorized",
            "selected_for_control_allowed",
        )
    ) or contract["truth_firewall"]["image_or_truth_feedback_changes_motion"] is not False:
        raise ValueError("probe authorization/truth firewall must remain false")
    return contract


def build_precommitted_motion_schedule(contract) -> tuple[dict[str, Any], ...]:
    """Freeze every command before Isaac starts or an image is observed."""
    rate = int(contract["motion"]["rate_hz"])
    current = np.asarray(contract["robot"]["home_arm_rad"], dtype=np.float64)
    result = []
    for segment in contract["motion"]["approach_segments"]:
        target = np.asarray(segment["target_arm_rad"], dtype=np.float64)
        count = round(float(segment["duration_s"]) * rate)
        commands = tuple(
            minimum_jerk_arm(current, target, (index + 1) / count)
            for index in range(count)
        )
        result.append({"name": segment["name"], "target": target, "commands": commands})
        current = target
    return tuple(result)


def derive_unknown_occlusion_shadow_inputs(depth_m: Any) -> dict[str, Any]:
    """Image-only ROI; integrated-scene occlusion intentionally stays unknown."""
    depth = np.asarray(depth_m)
    if depth.ndim != 2 or not np.issubdtype(depth.dtype, np.number):
        raise ValueError("depth_m must be a numeric HxW image")
    height, width = depth.shape
    rows, columns = np.indices(depth.shape)
    roi = (
        (rows >= height // 4) & (rows < height - height // 4)
        & (columns >= width // 4) & (columns < width - width // 4)
    )
    valid = np.isfinite(depth) & (depth > 0.0)
    return {
        "connector_face_mask": roi & valid,
        "face_center_uv": ((width - 1) * 0.5, (height - 1) * 0.5),
        "occlusion_mask": None,
        "diagnostics": {
            "mask_source": "CENTRAL_VALID_DEPTH_ROI_IMAGE_ONLY",
            "occlusion_status": "UNKNOWN_FAIL_CLOSED",
            "semantic_or_object_truth_used": False,
        },
    }


def _set_translation(prim, xyz, *, Gf, UsdGeom) -> None:
    xform = UsdGeom.Xformable(prim)
    operations = [
        operation for operation in xform.GetOrderedXformOps()
        if operation.GetOpType() == UsdGeom.XformOp.TypeTranslate
    ]
    if len(operations) > 1:
        raise RuntimeError(f"ambiguous translate ops: {prim.GetPath()}")
    (operations[0] if operations else xform.AddTranslateOp()).Set(Gf.Vec3d(*xyz))


def _author_scene_and_rig(
    *, stage, contract, robot_asset, keyed_asset, add_reference, Gf, UsdGeom,
    UsdLux, UsdPhysics
):
    """Reference both assets and author each non-physical camera exactly once."""
    assets, rig_contract = contract["assets"], contract["camera_rig"]
    add_reference(str(robot_asset), assets["robot_root_prim"])
    pair = stage.DefinePrim(assets["keyed_reference_prim"], "Xform")
    pair.GetReferences().AddReference(str(keyed_asset), ROOT_PRIM)
    for suffix, xyz in (
        ("/FixedReceptacle", assets["fixed_receptacle_origin_world_m"]),
        ("/LoosePlug", assets["loose_plug_origin_world_m"]),
    ):
        prim = stage.GetPrimAtPath(assets["keyed_reference_prim"] + suffix)
        if not prim.IsValid():
            raise RuntimeError(f"missing r2 endpoint: {suffix}")
        _set_translation(prim, xyz, Gf=Gf, UsdGeom=UsdGeom)

    # This probe observes authored views only.  Disable the endpoint rigid
    # bodies before reset so PhysX cannot reinterpret their nested reference
    # transforms, and isolate the unmodeled coupling joint entirely.
    for suffix in ("/LoosePlug/BodyAssembly", "/LoosePlug/CouplingNut"):
        prim = stage.GetPrimAtPath(assets["keyed_reference_prim"] + suffix)
        if not prim.IsValid():
            raise RuntimeError(f"missing keyed visual body: {suffix}")
        UsdPhysics.RigidBodyAPI.Apply(prim).CreateRigidBodyEnabledAttr().Set(False)
        prim.SetCustomDataByKey("kcg:cameraProbePhysics", "disabled_visual_only")
    joint = stage.GetPrimAtPath(
        assets["keyed_reference_prim"] + "/LoosePlug/CouplingNutJoint"
    )
    if not joint.IsValid():
        raise RuntimeError("missing keyed coupling joint")
    joint.SetActive(False)

    scene = contract["scene"]
    table = UsdGeom.Cube.Define(stage, "/World/KeyedV2HandCameraProbe/TableVisual")
    table.CreateSizeAttr(1.0)
    table.CreateDisplayColorAttr([Gf.Vec3f(0.30, 0.27, 0.22)])
    table_xform = UsdGeom.Xformable(table)
    table_xform.AddTranslateOp().Set(Gf.Vec3d(*scene["visual_table_center_m"]))
    table_xform.AddScaleOp().Set(Gf.Vec3f(*scene["visual_table_size_m"]))
    table.GetPrim().SetCustomDataByKey("kcg:physicsRole", "visual_only")
    fill = UsdLux.DomeLight.Define(stage, "/World/KeyedV2HandCameraProbe/Fill")
    fill.CreateIntensityAttr(float(scene["dome_light_intensity"]))
    key_light = UsdLux.DistantLight.Define(stage, "/World/KeyedV2HandCameraProbe/Key")
    key_light.CreateIntensityAttr(float(scene["key_light_intensity"]))
    UsdGeom.Xformable(key_light).AddRotateXYZOp().Set(Gf.Vec3f(-35.0, 20.0, 25.0))

    handbase = assets["handbase_prim"]
    if not stage.GetPrimAtPath(handbase).IsValid():
        raise RuntimeError(f"handbase prim missing: {handbase}")
    rig = {"handbase_prim": handbase, "mount_write_count": {}}
    for name in CAMERAS:
        definition = rig_contract[name]
        path = handbase + definition["prim_suffix"]
        camera = UsdGeom.Camera.Define(stage, path)
        xform = UsdGeom.Xformable(camera)
        xform.ClearXformOpOrder()
        t_hc = np.asarray(definition["T_HC_cv"], dtype=np.float64)
        xform.AddTransformOp().Set(
            Gf.Matrix4d(*camera_cv_pose_to_usd(t_hc).T.ravel().tolist())
        )
        camera.CreateFocalLengthAttr(float(rig_contract["focal_length_mm"]))
        camera.CreateHorizontalApertureAttr(float(rig_contract["horizontal_aperture_mm"]))
        camera.CreateVerticalApertureAttr(float(rig_contract["vertical_aperture_mm"]))
        camera.CreateClippingRangeAttr(Gf.Vec2f(*rig_contract["clipping_range_m"]))
        if any(
            token in schema for schema in camera.GetPrim().GetAppliedSchemas()
            for token in ("RigidBodyAPI", "CollisionAPI")
        ):
            raise RuntimeError(f"{name} Camera unexpectedly has a physics API")
        rig[name] = {"prim": path, "T_HC_cv": t_hc, "role": definition["role"]}
        rig["mount_write_count"][name] = 1
    return rig


class MountMonitor:
    """Continuous T_HC audit over only handbase and its two Camera children."""

    def __init__(self, stage, rig, thresholds, Usd, UsdGeom):
        self.stage, self.rig, self.Usd, self.UsdGeom = stage, rig, Usd, UsdGeom
        self.translation_limit = float(thresholds["maximum_T_HC_translation_residual_m"])
        self.rotation_limit = float(thresholds["maximum_T_HC_rotation_residual_rad"])
        self.count = 0
        self.max_t = {name: 0.0 for name in CAMERAS}
        self.max_r = {name: 0.0 for name in CAMERAS}

    def _world(self, path):
        prim = self.stage.GetPrimAtPath(path)
        return np.asarray(
            self.UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                self.Usd.TimeCode.Default()
            ), dtype=np.float64,
        ).T

    def sample(self):
        t_wh = self._world(self.rig["handbase_prim"])
        for name in CAMERAS:
            value = mount_residual(
                self.rig[name]["T_HC_cv"], t_wh, self._world(self.rig[name]["prim"])
            )
            self.max_t[name] = max(self.max_t[name], value["translation_m"])
            self.max_r[name] = max(self.max_r[name], value["rotation_rad"])
            if value["translation_m"] > self.translation_limit or value[
                "rotation_rad"
            ] > self.rotation_limit:
                raise RuntimeError(f"{name} T_HC rigidity gate failed")
        self.count += 1

    def report(self):
        return {
            "sample_count": self.count,
            "maximum_translation_residual_m": self.max_t,
            "maximum_rotation_residual_rad": self.max_r,
            "passed": self.count > 0,
        }


def _depth_preview(depth):
    valid = np.isfinite(depth) & (depth > 0.0)
    image = np.zeros(depth.shape, dtype=np.uint8)
    if not np.any(valid):
        return image
    low, high = np.quantile(depth[valid], (0.02, 0.98))
    image[valid] = 255 if high <= low else np.round(
        255.0 * np.clip((high - depth[valid]) / (high - low), 0.0, 1.0)
    ).astype(np.uint8)
    return image


def _capture(
    *, output_root, episode_dir, label, target_arm, world, robot, arm_indices,
    streams, monitor, contract, app, rep, shadow_pipeline
):
    """Zero-delta endpoint capture of exactly RGB plus planar depth."""
    from PIL import Image

    was_playing = bool(world.is_playing())
    if was_playing:
        world.pause()
        app.update()
    try:
        for _ in range(int(contract["capture"]["warmup_frames"])):
            rep.orchestrator.step(
                rt_subframes=int(contract["capture"]["rt_subframes"]),
                delta_time=0.0, pause_timeline=True,
            )
            monitor.sample()
        actual = np.asarray(robot.get_joint_positions(joint_indices=arm_indices))
        endpoint_dir = episode_dir / label
        endpoint_dir.mkdir(parents=True, exist_ok=False)
        result = {
            "name": label,
            "maximum_arm_tracking_error_rad": float(
                np.max(np.abs(actual - np.asarray(target_arm)))
            ),
            "cameras": {}, "palm_shadow": None,
        }
        arrays = {}
        width, height = contract["camera_rig"]["resolution_px"]
        for name in CAMERAS:
            rgba = np.asarray(streams[name]["rgb"].get_data())
            depth = np.asarray(streams[name]["depth"].get_data(), dtype=np.float32)
            if rgba.ndim != 3 or rgba.shape[:2] != (height, width) or rgba.shape[2] < 3:
                raise RuntimeError(f"{name} invalid RGB shape: {rgba.shape}")
            if depth.shape != (height, width):
                raise RuntimeError(f"{name} invalid depth shape: {depth.shape}")
            rgb = np.asarray(rgba[:, :, :3], dtype=np.uint8)
            camera_dir = endpoint_dir / name
            camera_dir.mkdir()
            Image.fromarray(rgb, "RGB").save(camera_dir / "rgb.png")
            np.save(camera_dir / "depth_m.npy", depth)
            Image.fromarray(_depth_preview(depth), "L").save(camera_dir / "depth_preview.png")
            valid = np.isfinite(depth) & (depth > 0.0)
            result["cameras"][name] = {
                "rgb": str((camera_dir / "rgb.png").relative_to(output_root)),
                "depth_m": str((camera_dir / "depth_m.npy").relative_to(output_root)),
                "depth_preview": str(
                    (camera_dir / "depth_preview.png").relative_to(output_root)
                ),
                "rgb_dynamic_range": int(rgb.max()) - int(rgb.min()),
                "rgb_standard_deviation": float(np.std(rgb.astype(np.float32))),
                "valid_depth_pixels": int(np.count_nonzero(valid)),
            }
            arrays[name] = rgb
            if name == "palm" and "pregrasp" in label:
                inputs = derive_unknown_occlusion_shadow_inputs(depth)
                shadow = shadow_pipeline(
                    inputs["connector_face_mask"], depth, inputs["face_center_uv"],
                    contract["capture"]["branch_directions_uv"], PLUG_MODEL_ID,
                    occlusion_mask=inputs["occlusion_mask"],
                )
                if (
                    shadow.get("rejection_code")
                    != contract["capture"]["palm_shadow_expected_rejection_code"]
                    or shadow.get("control_authorized") is not False
                    or shadow.get("selected_for_control_allowed") is not False
                ):
                    raise RuntimeError("Palm shadow did not fail closed")
                result["palm_shadow"] = {
                    "rejection_code": shadow["rejection_code"],
                    "input_derivation": inputs["diagnostics"],
                    "control_authorized": False,
                }
        return result, arrays
    finally:
        if was_playing:
            world.play()
            app.update()


def _run_episode(
    *, index, episode, output, contract, assets, app, api, motion_schedule
):
    """Create one fresh stage, run its bounded observation, and stop."""
    World = api["World"]
    World.clear_instance()
    api["omni_usd"].get_context().new_stage()
    app.update()
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=1.0 / contract["motion"]["rate_hz"],
        rendering_dt=1.0 / 60.0,
        backend="numpy", device="cpu",
    )
    stage = api["get_current_stage"]()
    rig = _author_scene_and_rig(
        stage=stage, contract=contract, robot_asset=assets[0], keyed_asset=assets[1],
        add_reference=api["add_reference_to_stage"], Gf=api["Gf"],
        UsdGeom=api["UsdGeom"], UsdLux=api["UsdLux"],
        UsdPhysics=api["UsdPhysics"],
    )
    robot = world.scene.add(api["SingleArticulation"](
        prim_path=contract["assets"]["articulation_prim"],
        name=f"keyed_v2_camera_probe_robot_{index}",
    ))
    world.reset()
    world.get_physics_context().set_gravity(
        gravity_z_scalar(contract["scene"]["gravity_m_s2"])
    )
    names = {name: position for position, name in enumerate(robot.dof_names)}
    arm_indices = np.asarray([names[name] for name in contract["robot"]["arm_joint_names"]])
    hand_indices = np.asarray(
        [names[name] for name in contract["robot"]["active_hand_joint_names"]]
    )
    controlled = np.concatenate((arm_indices, hand_indices)).astype(np.int32)
    gains = robot.get_articulation_controller()
    kps, kds = np.zeros(robot.num_dof), np.zeros(robot.num_dof)
    kps[arm_indices], kds[arm_indices] = (
        contract["robot"]["arm_stiffness"], contract["robot"]["arm_damping"]
    )
    kps[hand_indices], kds[hand_indices] = (
        contract["robot"]["hand_stiffness"], contract["robot"]["hand_damping"]
    )
    gains.set_gains(
        kps=kps.astype(np.float32), kds=kds.astype(np.float32), save_to_usd=False
    )
    home = np.asarray(contract["robot"]["home_arm_rad"])
    opened = np.asarray(contract["robot"]["open_hand_rad"])
    pregrasp = motion_schedule[-1]["target"]
    initial_arm = pregrasp if episode["initial_arm_pose"] == "pregrasp" else home
    positions = np.zeros(robot.num_dof, dtype=np.float32)
    positions[arm_indices] = initial_arm
    robot.set_joint_positions(positions)
    robot.set_joint_velocities(np.zeros(robot.num_dof, dtype=np.float32))

    streams = {}
    resolution = tuple(contract["camera_rig"]["resolution_px"])
    for name in CAMERAS:
        product = api["rep"].create.render_product(
            stage.GetPrimAtPath(rig[name]["prim"]), resolution,
            name=f"D38999KeyedV2{name.title()}ProbeProduct",
        )
        rgb = api["rep"].AnnotatorRegistry.get_annotator("rgb")
        depth = api["rep"].AnnotatorRegistry.get_annotator("distance_to_image_plane")
        rgb.attach([product.path])
        depth.attach([product.path])
        streams[name] = {"product": product.path, "rgb": rgb, "depth": depth}
    monitor = MountMonitor(
        stage, rig, contract["thresholds"], api["Usd"], api["UsdGeom"]
    )
    episode_dir = output / episode["id"]
    episode_dir.mkdir()
    report = {
        "episode_id": episode["id"], "fresh_stage": True, "fresh_world": True,
        "motion_kind": episode["motion_kind"], "control_authorized": False,
        "grasp_or_insertion_entered": False,
        "camera_rig": {
            "parent_prim": rig["handbase_prim"],
            "mount_write_count": rig["mount_write_count"],
            "palm_role": rig["palm"]["role"], "wrist_role": rig["wrist"]["role"],
        },
        "endpoints": [],
    }

    def step(arm):
        robot.apply_action(api["ArticulationAction"](
            joint_positions=np.concatenate((arm, opened)).astype(np.float32),
            joint_indices=controlled,
        ))
        world.step(render=False)
        monitor.sample()

    def hold(arm, duration):
        for _ in range(max(1, round(duration * contract["motion"]["rate_hz"]))):
            step(arm)

    def capture(label, target):
        return _capture(
            output_root=output, episode_dir=episode_dir, label=label,
            target_arm=target, world=world, robot=robot, arm_indices=arm_indices,
            streams=streams, monitor=monitor, contract=contract, app=app,
            rep=api["rep"], shadow_pipeline=api["shadow_pipeline"],
        )

    try:
        # Start every active/passive hand joint consistently at zero, then let
        # the articulation controller move only the active joints to the open
        # target.  Directly teleporting active mimic masters caused non-finite
        # PhysX transforms in the first live probe.
        hold(initial_arm, contract["motion"]["hand_open_settle_s"])
        if episode["id"] == EPISODE_IDS[0]:
            images = []
            for repeat in range(contract["motion"]["stationary_repeat_count"]):
                hold(pregrasp, contract["motion"]["stationary_hold_s"])
                endpoint, arrays = capture(f"pregrasp_repeat_{repeat:02d}", pregrasp)
                report["endpoints"].append(endpoint)
                images.append(arrays)
            report["stationary_repeat_rgb_mean_abs_difference"] = {
                name: float(np.mean(np.abs(
                    images[1][name].astype(np.float32)
                    - images[0][name].astype(np.float32)
                ))) for name in CAMERAS
            }
        else:
            hold(home, contract["motion"]["endpoint_hold_s"])
            endpoint, home_images = capture("home", home)
            report["endpoints"].append(endpoint)
            for segment in motion_schedule:
                for arm in segment["commands"]:
                    step(arm)
                hold(segment["target"], contract["motion"]["endpoint_hold_s"])
                endpoint, images = capture(segment["name"], segment["target"])
                for name in CAMERAS:
                    endpoint["cameras"][name]["rgb_mean_abs_difference_from_home"] = float(
                        np.mean(np.abs(
                            images[name].astype(np.float32)
                            - home_images[name].astype(np.float32)
                        ))
                    )
                report["endpoints"].append(endpoint)

        threshold = contract["thresholds"]
        report["rigidity"] = monitor.report()
        report["endpoint_tracking_passed"] = all(
            item["maximum_arm_tracking_error_rad"]
            <= threshold["maximum_endpoint_arm_tracking_error_rad"]
            for item in report["endpoints"]
        )
        rgbd_quality = evaluate_rgbd_quality(
            report["endpoints"], episode["rgbd_quality_required_endpoints"], threshold
        )
        report["rgbd_quality_passed"] = rgbd_quality["passed"]
        report["rgbd_quality_required_endpoints"] = rgbd_quality[
            "required_endpoints"
        ]
        report["rgbd_quality_by_captured_endpoint"] = rgbd_quality[
            "quality_by_captured_endpoint"
        ]
        report["moving_views_changed"] = None if episode["id"] == EPISODE_IDS[0] else all(
            report["endpoints"][-1]["cameras"][name][
                "rgb_mean_abs_difference_from_home"
            ] >= threshold["minimum_moving_rgb_mean_abs_difference"]
            for name in CAMERAS
        )
        report["ended_at_pregrasp"] = "pregrasp" in report["endpoints"][-1]["name"]
        report["passed"] = bool(
            report["rigidity"]["passed"] and report["endpoint_tracking_passed"]
            and report["rgbd_quality_passed"] and report["ended_at_pregrasp"]
            and report["moving_views_changed"] is not False
        )
        return report
    finally:
        for stream in streams.values():
            stream["rgb"].detach([stream["product"]])
            stream["depth"].detach([stream["product"]])
        world.stop()


def main(argv=None) -> int:
    root = repository_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(root / DEFAULT_CONFIG))
    parser.add_argument("--output-dir")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--run", action="store_true", help="explicit Isaac opt-in")
    arguments = parser.parse_args(argv)
    if not arguments.run:
        parser.error("--run is required")
    contract = load_probe_contract(arguments.config)
    assets = (
        (root / contract["assets"]["robot"]).resolve(),
        (root / contract["assets"]["keyed_pair"]).resolve(),
    )
    if not all(path.is_file() for path in assets) or assets[1].name != RECOMMENDED_ASSET_NAME:
        raise FileNotFoundError("required handarm or r2 keyed-v2 asset is missing")
    default_output = root / contract["directory_contract"]["default_output"]
    output = safe_new_output_dir(arguments.output_dir or default_output)
    output.mkdir(parents=True, exist_ok=False)
    schedule = build_precommitted_motion_schedule(contract)
    report = {
        "schema_version": SCHEMA_VERSION, "probe_scope": PROBE_SCOPE,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "RUNNING", "passed": False, "control_authorized": False,
        "visual_control_authorized": False, "insertion_control_authorized": False,
        "selected_for_control_allowed": False, "simulation_only": True,
        "thread_mode": "UNMODELED_NOT_USED", "viewpoint_human_review_required": True,
        "uses_semantic_or_object_pose_or_contact_truth": False,
        "image_or_truth_feedback_changes_motion": False,
        "motion_schedule_precommitted_before_isaac_start": True,
        "episodes": [],
    }

    from isaacsim import SimulationApp
    app = SimulationApp({
        "headless": not arguments.gui, "multi_gpu": False,
        "active_gpu": 0, "physics_gpu": 0,
    })
    completed = False
    try:
        import omni.replicator.core as rep
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
        from isaacsim.core.utils.types import ArticulationAction
        from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics
        from kcg_connector.d38999_key_shadow_pipeline import run_palm_key_shadow_pipeline

        api = {
            "rep": rep, "omni_usd": omni.usd, "World": World,
            "SingleArticulation": SingleArticulation,
            "add_reference_to_stage": add_reference_to_stage,
            "get_current_stage": get_current_stage,
            "ArticulationAction": ArticulationAction,
            "Gf": Gf, "Usd": Usd, "UsdGeom": UsdGeom, "UsdLux": UsdLux,
            "UsdPhysics": UsdPhysics,
            "shadow_pipeline": run_palm_key_shadow_pipeline,
        }
        for index, episode in enumerate(contract["episodes"]):
            result = _run_episode(
                index=index, episode=episode, output=output, contract=contract,
                assets=assets, app=app, api=api, motion_schedule=schedule,
            )
            report["episodes"].append(result)
            if result["passed"] is not True:
                raise RuntimeError(f"episode failed: {episode['id']}")
        report.update(
            status="TECHNICAL_PROBE_PASSED_HUMAN_VIEW_REVIEW_REQUIRED",
            passed=True, completed_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        completed = True
    except BaseException as error:
        report.update(
            status="FAILED", error=f"{type(error).__name__}: {error}",
            completed_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        traceback.print_exc()
    finally:
        try:
            (output / "report.json").write_text(
                json.dumps(
                    json_safe(report), allow_nan=False, indent=2, sort_keys=True
                ) + "\n",
                encoding="utf-8",
            )
        except BaseException:
            completed = False
            traceback.print_exc()
        app.close(exit_code=0 if completed else 1)
    return 0 if completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
