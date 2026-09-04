#!/usr/bin/env python3
"""Execute one simulation-only move to a precomputed visual handoff pose."""

from __future__ import annotations

import argparse
from dataclasses import replace
import gzip
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().with_name("carts_v2")))
import controller as control  # type: ignore  # noqa: E402
import run_grasp_lift as runner  # type: ignore  # noqa: E402
from engine_health import (  # type: ignore  # noqa: E402
    gpu_backend_record,
    gpu_world_parameters,
    load_runtime_resources,
)
from kcg_connector.grasp.carts_v2.models import load_v2_inputs  # noqa: E402
from kcg_connector.grasp.robust.bounded_hand_base_ik import (  # noqa: E402
    solve_bounded_hand_base_ik,
)
from kcg_connector.te_transport_grasp_target import (  # noqa: E402
    load_transport_grasp_relation,
)
from te_foundationpose_handoff_plan import (  # type: ignore  # noqa: E402
    FullRobotCollisionScene,
    _cylinder_from_mesh,
    _fcl_distance,
)
from te_plug_five_dof_geometry import (  # type: ignore  # noqa: E402
    estimate_plug_five_dof,
)


CV_FROM_USD = np.diag((1.0, -1.0, -1.0, 1.0))
RECEPTACLE_MESH_SCALE_TO_M = 0.001
GLOBAL_PEM_GEOMETRY_MAX_AXIS_DISAGREEMENT_DEG = 5.0

MOVEIT_SOFT_ARM_BOUNDS_RAD = {
    "iiwa_joint_1": (-2.93215314335, 2.93215314335),
    "iiwa_joint_2": (-2.05948851735, 2.05948851735),
    "iiwa_joint_3": (-2.93215314335, 2.93215314335),
    "iiwa_joint_4": (-2.05948851735, 2.05948851735),
    "iiwa_joint_5": (-2.93215314335, 2.93215314335),
    "iiwa_joint_6": (-2.05948851735, 2.05948851735),
    "iiwa_joint_7": (-3.01941960595, 3.01941960595),
}


class _PalmLocalizationOnlyComplete(Exception):
    """Internal control-flow marker after the requested stationary frames."""


class _GlobalCaptureOnlyComplete(Exception):
    """Internal control-flow marker after one requested global RGB-D capture."""


class _PhysicalVisualAuditor:
    """Record physical truth post-run while preserving the wrist safety gate."""

    def __init__(self, ft_auditor: object, truth_auditor: object) -> None:
        self.ft_auditor = ft_auditor
        self.truth_auditor = truth_auditor

    @property
    def samples(self) -> list[dict[str, object]]:
        return self.truth_auditor.samples

    def capture(self, **keywords: object) -> None:
        self.truth_auditor.capture(**keywords)
        self.ft_auditor.capture(**keywords)


def _split_scene_entry(
    repository: Path,
    scene_entry: dict[str, object],
    manifest_path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    """Replace only the plug asset with the bound body-plus-nut model."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "schema_version": "kcg_te_j35_free_split_tabletop_asset_v1",
        "product_id": "D38999/26FJ35PN",
        "hardware_authorized": False,
        "legal_grasp_contact_part": "CouplingNut",
    }
    if any(manifest.get(key) != value for key, value in required.items()):
        raise ValueError("split plug manifest identity or safety scope differs")
    asset = (repository / str(manifest["asset"])).resolve()
    if (
        not asset.is_file()
        or runner.file_sha256(asset) != manifest.get("asset_sha256")
    ):
        raise ValueError("split plug asset differs from its manifest")
    result = dict(scene_entry)
    result.update(
        {
            "scene_kind": runner.FREE_SPLIT_SCENE_KIND,
            "asset": str(manifest["asset"]),
            "manifest": str(manifest_path.relative_to(repository)),
            "reference_prim_path": str(manifest["reference_prim_path"]),
            "body_relative_prim_path": str(
                manifest["body_relative_prim_path"]
            ),
            "coupling_nut_relative_prim_path": str(
                manifest["coupling_nut_relative_prim_path"]
            ),
            "joint_relative_prim_path": str(
                manifest["joint_relative_prim_path"]
            ),
            "split_joint_contract": dict(manifest["joint"]),
            "component_bottom_offsets_m": list(
                manifest["component_bottom_offsets_m"]
            ),
            "legal_grasp_contact_part": "CouplingNut",
        }
    )
    return result, manifest


def _json_ready(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _matrix4(values: object, label: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.size != 16:
        raise ValueError(f"{label} must contain 16 values")
    matrix = matrix.reshape(4, 4)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{label} is not finite")
    return matrix


def _directed_axis_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first /= np.linalg.norm(first)
    second /= np.linalg.norm(second)
    return math.degrees(
        math.acos(np.clip(float(first @ second), -1.0, 1.0))
    )


def _validate_receptacle_visual_binding(
    *,
    visual_path: Path,
    collision_mesh_path: Path,
    position_world_m: np.ndarray,
    fixture: dict[str, object],
    Usd: object,
    UsdGeom: object,
) -> dict[str, object]:
    """Require one metre-authored USD visual matching the collision CAD."""
    if visual_path.suffix.lower() not in (".usd", ".usda", ".usdc"):
        raise ValueError(
            "receptacle visual must be a metre-authored USD asset; raw OBJ "
            "units are ambiguous"
        )
    visual_stage = Usd.Stage.Open(str(visual_path))
    if visual_stage is None:
        raise ValueError(f"receptacle visual could not be opened: {visual_path}")
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(visual_stage))
    if not math.isclose(meters_per_unit, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(
            "receptacle visual must declare exactly one metre per stage unit"
        )
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
    )
    box = cache.ComputeWorldBound(visual_stage.GetPseudoRoot()).ComputeAlignedBox()
    visual_bounds_m = np.asarray(
        (
            tuple(float(value) for value in box.GetMin()),
            tuple(float(value) for value in box.GetMax()),
        ),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(visual_bounds_m)):
        raise ValueError("receptacle visual bounds are not finite")

    import trimesh

    collision_mesh = trimesh.load_mesh(collision_mesh_path, process=False)
    if not isinstance(collision_mesh, trimesh.Trimesh):
        raise ValueError("receptacle collision CAD is not one triangle mesh")
    collision_bounds_m = (
        np.asarray(collision_mesh.bounds, dtype=np.float64)
        * RECEPTACLE_MESH_SCALE_TO_M
    )
    if not np.allclose(
        visual_bounds_m, collision_bounds_m, rtol=0.0, atol=1.0e-6
    ):
        raise ValueError(
            "receptacle visual and collision CAD bounds differ after unit conversion"
        )

    fixture_center = np.asarray(fixture["center_world_m"], dtype=np.float64)
    fixture_size = np.asarray(fixture["size_m"], dtype=np.float64)
    fixture_min = fixture_center - 0.5 * fixture_size
    fixture_max = fixture_center + 0.5 * fixture_size
    installed_min = position_world_m + visual_bounds_m[0]
    installed_max = position_world_m + visual_bounds_m[1]
    if not math.isclose(
        float(installed_min[2]),
        float(fixture_max[2]),
        rel_tol=0.0,
        abs_tol=1.0e-6,
    ):
        raise ValueError("receptacle back face is not seated on the fixture top")
    if np.any(installed_min[:2] < fixture_min[:2]) or np.any(
        installed_max[:2] > fixture_max[:2]
    ):
        raise ValueError("receptacle visual footprint extends outside the fixture")
    return {
        "visual_asset": str(visual_path),
        "collision_mesh": str(collision_mesh_path),
        "meters_per_unit": meters_per_unit,
        "visual_local_bounds_m": visual_bounds_m.tolist(),
        "collision_local_bounds_m": collision_bounds_m.tolist(),
        "installed_world_bounds_m": [
            installed_min.tolist(),
            installed_max.tolist(),
        ],
        "fixture_top_z_m": float(fixture_max[2]),
        "back_face_z_m": float(installed_min[2]),
    }


def _estimate_five_dof_from_float_depth(
    *,
    rgb_path: Path,
    depth_m_path: Path,
    mask_path: Path,
    intrinsics: np.ndarray,
    mesh_path: Path,
) -> dict[str, object]:
    rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    filtered_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if rgb_bgr is None or filtered_mask is None:
        raise RuntimeError("geometry estimator RGB or mask could not be read")
    depth_m = np.asarray(np.load(depth_m_path), dtype=np.float64)
    if depth_m.shape != filtered_mask.shape:
        raise RuntimeError("geometry estimator depth and mask shapes differ")
    geometry_result = estimate_plug_five_dof(
        rgb=cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB),
        depth_mm=1000.0 * depth_m,
        mask=filtered_mask > 0,
        intrinsics=intrinsics,
        mesh_path=mesh_path,
        depth_bin_center_correction_mm=0.0,
    )
    metrics = dict(geometry_result["metrics"])
    metrics["depth_source"] = "FLOAT_METERS_NPY"
    geometry_result["metrics"] = metrics
    return geometry_result


def _pem_geometry_consistency(
    *,
    pem_world_from_object: np.ndarray,
    geometry_world_from_object: np.ndarray,
    geometry_metrics: dict[str, object],
) -> dict[str, object]:
    radius_m = float(geometry_metrics["visible_face_radius_m"])
    visible_face_to_origin_m = float(
        geometry_metrics["visible_face_to_object_origin_m"]
    )
    pem_axis = pem_world_from_object[:3, 2]
    geometry_axis = geometry_world_from_object[:3, 2]
    pem_face_center = (
        pem_world_from_object[:3, 3]
        - visible_face_to_origin_m * pem_axis
    )
    geometry_face_center = (
        geometry_world_from_object[:3, 3]
        - visible_face_to_origin_m * geometry_axis
    )
    face_center_disagreement_m = float(
        np.linalg.norm(pem_face_center - geometry_face_center)
    )
    axis_disagreement_deg = _directed_axis_angle_deg(pem_axis, geometry_axis)
    maximum_face_center_disagreement_m = 0.25 * radius_m
    passed = bool(
        face_center_disagreement_m <= maximum_face_center_disagreement_m
        and axis_disagreement_deg
        <= GLOBAL_PEM_GEOMETRY_MAX_AXIS_DISAGREEMENT_DEG
    )
    return {
        "pass": passed,
        "face_center_disagreement_mm": 1000.0 * face_center_disagreement_m,
        "maximum_face_center_disagreement_mm": (
            1000.0 * maximum_face_center_disagreement_m
        ),
        "directed_axis_disagreement_deg": axis_disagreement_deg,
        "maximum_directed_axis_disagreement_deg": (
            GLOBAL_PEM_GEOMETRY_MAX_AXIS_DISAGREEMENT_DEG
        ),
        "position_limit_derivation": "ONE_QUARTER_VISIBLE_FACE_RADIUS",
        "axis_limit_basis": (
            "CURRENT_SUCCESSFUL_GLOBAL_RUNS_MAXIMUM_1P008_DEG_WITH_"
            "FIVE_DEGREE_GROSS_FAILURE_MARGIN"
        ),
    }


def _camera_cv_pose_from_eye_target(eye: object, target: object) -> np.ndarray:
    eye_vector = np.asarray(eye, dtype=np.float64)
    target_vector = np.asarray(target, dtype=np.float64)
    forward = target_vector - eye_vector
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, (0.0, 0.0, 1.0))
    if np.linalg.norm(right) < 1.0e-8:
        right = np.cross(forward, (0.0, 1.0, 0.0))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    down /= np.linalg.norm(down)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.column_stack((right, down, forward))
    pose[:3, 3] = eye_vector
    return pose


def _author_camera(
    stage: object,
    path: str,
    parent_from_camera_cv: np.ndarray,
    *,
    resolution: tuple[int, int],
    focal_length_mm: float,
    horizontal_aperture_mm: float,
    clipping_range_m: tuple[float, float],
    Gf: object,
    UsdGeom: object,
) -> None:
    camera = UsdGeom.Camera.Define(stage, path)
    xform = UsdGeom.Xformable(camera)
    xform.ClearXformOpOrder()
    parent_from_camera_usd = parent_from_camera_cv @ CV_FROM_USD
    xform.AddTransformOp().Set(
        Gf.Matrix4d(*parent_from_camera_usd.T.ravel().tolist())
    )
    camera.CreateFocalLengthAttr(float(focal_length_mm))
    camera.CreateHorizontalApertureAttr(float(horizontal_aperture_mm))
    camera.CreateVerticalApertureAttr(
        float(horizontal_aperture_mm) * resolution[1] / resolution[0]
    )
    camera.CreateClippingRangeAttr(Gf.Vec2f(*clipping_range_m))


def _capture_rgbd(
    *,
    rep: object,
    resources: dict[tuple[str, tuple[int, int]], tuple[object, tuple[object, object]]],
    camera_path: str,
    resolution: tuple[int, int],
    output_dir: Path,
    warmup_frames: int,
    rt_subframes: int,
) -> dict[str, object]:
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=False)
    resource_key = (camera_path, resolution)
    if resource_key not in resources:
        render_product = rep.create.render_product(
            camera_path,
            resolution,
            name=f"TEVisualHandoffRenderProduct{len(resources):02d}",
        )
        rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        depth_annotator = rep.AnnotatorRegistry.get_annotator(
            "distance_to_image_plane"
        )
        annotators = (rgb_annotator, depth_annotator)
        for annotator in annotators:
            annotator.attach([render_product.path])
        resources[resource_key] = (render_product, annotators)
    render_product, annotators = resources[resource_key]
    rgb_annotator, depth_annotator = annotators
    for _ in range(int(warmup_frames)):
        rep.orchestrator.step(
            rt_subframes=int(rt_subframes),
            delta_time=0.0,
            pause_timeline=True,
        )
    rgba = np.asarray(rgb_annotator.get_data())
    depth = np.asarray(depth_annotator.get_data(), dtype=np.float32)
    if rgba.ndim != 3 or rgba.shape[2] < 3 or rgba.shape[:2] != depth.shape:
        raise RuntimeError("RGB-D capture returned mismatched arrays")
    rgb = np.asarray(rgba[:, :, :3], dtype=np.uint8)
    Image.fromarray(rgb).save(output_dir / "rgb.png")
    np.save(output_dir / "depth_m.npy", depth)
    finite = np.isfinite(depth) & (depth > 0.0)
    preview = np.zeros(depth.shape, dtype=np.uint16)
    preview[finite] = np.clip(depth[finite] * 1000.0, 0.0, 65535.0).astype(
        np.uint16
    )
    Image.fromarray(preview).save(output_dir / "depth_mm.png")
    return {
        "camera_path": camera_path,
        "resolution": list(resolution),
        "finite_positive_depth_pixels": int(np.sum(finite)),
        "depth_valid_fraction": float(np.mean(finite)),
        "rgb_minimum": int(np.min(rgb)),
        "rgb_maximum": int(np.max(rgb)),
        "rgb_standard_deviation": float(np.std(rgb)),
    }


def _close_rgbd_resources(
    resources: dict[tuple[str, tuple[int, int]], tuple[object, tuple[object, object]]],
) -> None:
    for render_product, annotators in reversed(tuple(resources.values())):
        for annotator in reversed(annotators):
            try:
                annotator.detach()
            except Exception:
                pass
        try:
            render_product.destroy()
        except Exception:
            pass
    resources.clear()


def _world_from_prim_cv(stage: object, path: str, *, Usd: object, UsdGeom: object) -> np.ndarray:
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"camera prim is missing: {path}")
    world_from_usd = np.asarray(
        UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        ),
        dtype=np.float64,
    ).T
    return world_from_usd @ CV_FROM_USD


def _world_from_prim(stage: object, path: str, *, Usd: object, UsdGeom: object) -> np.ndarray:
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"prim is missing: {path}")
    return np.asarray(
        UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        ),
        dtype=np.float64,
    ).T


def _yaw_free_object_frame(
    position_world: np.ndarray,
    outward_axis_world: np.ndarray,
) -> np.ndarray:
    z_axis = np.asarray(outward_axis_world, dtype=np.float64)
    z_axis /= np.linalg.norm(z_axis)
    for reference in (
        np.asarray((1.0, 0.0, 0.0), dtype=np.float64),
        np.asarray((0.0, 1.0, 0.0), dtype=np.float64),
    ):
        x_axis = reference - float(reference @ z_axis) * z_axis
        norm = float(np.linalg.norm(x_axis))
        if norm > 1.0e-8:
            x_axis /= norm
            break
    else:  # pragma: no cover
        raise ValueError("cannot construct yaw-free object frame")
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    frame = np.eye(4, dtype=np.float64)
    frame[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
    frame[:3, 3] = np.asarray(position_world, dtype=np.float64)
    return frame


def _rotation_angle(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(first, dtype=np.float64).T @ np.asarray(
        second, dtype=np.float64
    )
    return float(
        math.acos(np.clip(0.5 * (np.trace(relative) - 1.0), -1.0, 1.0))
    )


def _bounded_pose_step(
    current: np.ndarray,
    target: np.ndarray,
    *,
    maximum_translation_m: float,
    maximum_rotation_rad: float,
) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    translation = target[:3, 3] - current[:3, 3]
    translation_norm = float(np.linalg.norm(translation))
    translation_scale = (
        1.0
        if translation_norm <= maximum_translation_m
        else maximum_translation_m / translation_norm
    )
    result[:3, 3] = current[:3, 3] + translation_scale * translation
    rotation_vector = Rotation.from_matrix(
        target[:3, :3] @ current[:3, :3].T
    ).as_rotvec()
    rotation_norm = float(np.linalg.norm(rotation_vector))
    rotation_scale = (
        1.0
        if rotation_norm <= maximum_rotation_rad
        else maximum_rotation_rad / rotation_norm
    )
    result[:3, :3] = (
        Rotation.from_rotvec(rotation_scale * rotation_vector).as_matrix()
        @ current[:3, :3]
    )
    return result


def _child_environment() -> dict[str, str]:
    environment = dict(os.environ)
    isaac_library = "/home/noob/WorkPlace/isaacsim/.conda-env/lib"
    entries = [
        item
        for item in environment.get("LD_LIBRARY_PATH", "").split(":")
        if item and item != isaac_library
    ]
    environment["LD_LIBRARY_PATH"] = ":".join(entries)
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    environment["PYTHONUNBUFFERED"] = "1"
    return environment


def _run_logged_process(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
) -> float:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        env=_child_environment(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    elapsed = time.perf_counter() - started
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"subprocess failed ({completed.returncode}); see {log_path}"
        )
    return elapsed


def _translated_frame(frame: np.ndarray, local_z_m: float) -> np.ndarray:
    offset = np.eye(4, dtype=np.float64)
    offset[2, 3] = float(local_z_m)
    return np.asarray(frame, dtype=np.float64) @ offset


def _moveit_collision_documents(
    *,
    collision_contract: dict[str, object],
    known_scene_geometry: dict[str, object],
    world_from_object: np.ndarray,
    receptacle_position_m: object,
) -> list[dict[str, object]]:
    def box(identifier: str, row: dict[str, object]) -> dict[str, object]:
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = np.asarray(row["center_world_m"], dtype=np.float64)
        return {
            "id": identifier,
            "type": "box",
            "dimensions_m": list(map(float, row["size_m"])),
            "world_from_primitive_row_major": pose.ravel().tolist(),
        }

    def cylinder(
        identifier: str,
        bounds: dict[str, object],
        frame: np.ndarray,
    ) -> dict[str, object]:
        z_min = float(bounds["z_min_m"])
        z_max = float(bounds["z_max_m"])
        centered = _translated_frame(frame, 0.5 * (z_min + z_max))
        return {
            "id": identifier,
            "type": "cylinder",
            "dimensions_m": [z_max - z_min, float(bounds["radius_m"])],
            "world_from_primitive_row_major": centered.ravel().tolist(),
        }

    receptacle = np.eye(4, dtype=np.float64)
    receptacle[:3, 3] = np.asarray(receptacle_position_m, dtype=np.float64)
    return [
        box("table", dict(known_scene_geometry["table"])),
        box("fixture", dict(known_scene_geometry["fixture"])),
        cylinder(
            "plug",
            dict(collision_contract["plug_yaw_swept_cylinder"]),
            world_from_object,
        ),
        cylinder(
            "receptacle",
            dict(collision_contract["receptacle_cylinder"]),
            receptacle,
        ),
    ]


def _run_external_pose_plan(
    *,
    backend: str,
    repository: Path,
    ros_domain_id: int,
    stage_name: str,
    output_dir: Path,
    start_arm: np.ndarray,
    hand_positions: np.ndarray,
    target_world_from_hand: np.ndarray,
    collision_objects: list[dict[str, object]],
    physics_dt_s: float,
    ik_timeout_s: float,
    planning_timeout_s: float,
) -> tuple[list[np.ndarray], dict[str, object]]:
    if backend not in ("moveit", "tesseract"):
        raise ValueError(f"unsupported external planning backend: {backend}")
    stage_dir = output_dir / f"{backend}_plans" / stage_name
    stage_dir.mkdir(parents=True, exist_ok=False)
    request_path = stage_dir / "request.json"
    plan_path = stage_dir / "plan.json"
    log_path = stage_dir / "client.log"
    start = np.asarray(start_arm, dtype=np.float64)
    hand = np.asarray(hand_positions, dtype=np.float64)
    target = np.asarray(target_world_from_hand, dtype=np.float64)
    if (
        start.shape != (7,)
        or hand.shape != (4,)
        or target.shape != (4, 4)
        or not all(np.all(np.isfinite(row)) for row in (start, hand, target))
    ):
        raise ValueError("external planning inputs are invalid")
    request = {
        "frame_id": "world",
        "group_name": "kuka",
        "target_link": "handbase_link",
        "start_joint_names": list(
            control.ARM_JOINT_NAMES + control.ACTIVE_HAND_JOINT_NAMES
        ),
        "start_joint_positions_rad": np.concatenate((start, hand)).tolist(),
        "world_from_target_link_row_major": target.ravel().tolist(),
        "position_tolerance_m": 1.0e-4,
        "orientation_tolerance_rad": 5.0e-4,
        "joint_goal_tolerance_rad": 1.0e-4,
        "ik_timeout_s": float(ik_timeout_s),
        "allowed_planning_time_s": float(planning_timeout_s),
        "num_planning_attempts": 4,
        "maximum_velocity_scaling_factor": 0.1,
        "maximum_acceleration_scaling_factor": 0.1,
        "planner_id": "RRTConnect",
        "path_joint_bounds_rad": {
            name: list(bounds)
            for name, bounds in MOVEIT_SOFT_ARM_BOUNDS_RAD.items()
        },
        "collision_objects": collision_objects,
    }
    request_path.write_text(
        json.dumps(request, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    setup = repository / "install" / "setup.bash"
    if backend == "moveit":
        local_pick_ik = (
            repository / ".deps" / "pick_ik" / "opt" / "ros" / "humble"
        )
        planner = (
            repository / "src" / "kcg_moveit1" / "scripts" / "plan_visual_pose.py"
        )
        if not all(path.exists() for path in (setup, local_pick_ik, planner)):
            raise RuntimeError(
                "MoveIt workspace or local pick_ik installation is missing"
            )
        shell_command = " && ".join(
            (
                "source /opt/ros/humble/setup.bash",
                f"source {shlex.quote(str(setup))}",
                "export AMENT_PREFIX_PATH="
                f"{shlex.quote(str(local_pick_ik))}:$AMENT_PREFIX_PATH",
                "export LD_LIBRARY_PATH="
                f"{shlex.quote(str(local_pick_ik / 'lib'))}:$LD_LIBRARY_PATH",
                f"exec /usr/bin/python3 {shlex.quote(str(planner))} "
                f"--request {shlex.quote(str(request_path))} "
                f"--output {shlex.quote(str(plan_path))}",
            )
        )
        timeout_s = float(ik_timeout_s + planning_timeout_s + 30.0)
        planner_label = "MoveIt"
    else:
        planner = (
            repository
            / "src"
            / "kcg_moveit1"
            / "scripts"
            / "plan_visual_pose_tesseract.py"
        )
        python = repository / ".venv" / "bin" / "python"
        isaac_library = Path(
            "/home/noob/WorkPlace/isaacsim/.conda-env/lib"
        )
        if not all(path.exists() for path in (setup, planner, python, isaac_library)):
            raise RuntimeError("Tesseract planner environment is missing")
        shell_command = " && ".join(
            (
                "source /opt/ros/humble/setup.bash",
                f"source {shlex.quote(str(setup))}",
                "export LD_LIBRARY_PATH="
                f"{shlex.quote(str(isaac_library))}:$LD_LIBRARY_PATH",
                f"exec {shlex.quote(str(python))} {shlex.quote(str(planner))} "
                f"--request {shlex.quote(str(request_path))} "
                f"--output {shlex.quote(str(plan_path))}",
            )
        )
        timeout_s = float(planning_timeout_s + 30.0)
        planner_label = "Tesseract"
    environment = _child_environment()
    environment["ROS_DOMAIN_ID"] = str(int(ros_domain_id))
    started = time.perf_counter()
    completed = subprocess.run(
        ["/bin/bash", "-lc", shell_command],
        cwd=str(repository),
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_s,
    )
    elapsed = time.perf_counter() - started
    log_path.write_text(completed.stdout, encoding="utf-8")
    if not plan_path.is_file():
        raise RuntimeError(
            f"{planner_label} client produced no plan for {stage_name}; "
            f"see {log_path}"
        )
    document = json.loads(plan_path.read_text(encoding="utf-8"))
    document["client_wall_s"] = elapsed
    document["client_return_code"] = int(completed.returncode)
    if completed.returncode != 0 or document.get("success") is not True:
        raise RuntimeError(
            f"{planner_label} planning failed at "
            f"{document.get('failure_stage')} with code "
            f"{document.get('error_code')}; see {plan_path}"
        )
    trajectory = document["trajectory"]
    names = tuple(trajectory["joint_names"])
    expected = tuple(control.ARM_JOINT_NAMES)
    if set(names) != set(expected):
        raise RuntimeError(
            f"{planner_label} trajectory arm joints differ from the robot"
        )
    order = [names.index(name) for name in expected]
    points = trajectory["points"]
    source_time = np.asarray(
        [row["time_from_start_s"] for row in points], dtype=np.float64
    )
    source_positions = np.asarray(
        [[row["positions_rad"][index] for index in order] for row in points],
        dtype=np.float64,
    )
    if (
        len(points) < 2
        or source_positions.shape != (len(points), 7)
        or not np.all(np.isfinite(source_time))
        or not np.all(np.isfinite(source_positions))
        or abs(float(source_time[0])) > 1.0e-12
        or np.any(np.diff(source_time) <= 0.0)
        or not np.allclose(source_positions[0], start, rtol=0.0, atol=1.0e-6)
    ):
        raise RuntimeError(f"{planner_label} returned an invalid timed trajectory")
    soft_bounds = np.asarray(
        [MOVEIT_SOFT_ARM_BOUNDS_RAD[name] for name in expected],
        dtype=np.float64,
    )
    path_soft_limit_margin = np.minimum(
        source_positions - soft_bounds[None, :, 0],
        soft_bounds[None, :, 1] - source_positions,
    )
    minimum_path_soft_limit_margin_rad = float(np.min(path_soft_limit_margin))
    if minimum_path_soft_limit_margin_rad < -1.0e-9:
        raise RuntimeError(
            f"{planner_label} returned a path outside the configured soft limits"
        )
    count = max(1, int(math.ceil(float(source_time[-1]) / physics_dt_s)))
    target_time = np.linspace(0.0, float(source_time[-1]), count + 1)
    resampled = np.column_stack(
        [
            np.interp(target_time, source_time, source_positions[:, index])
            for index in range(7)
        ]
    )
    document["execution_resampling"] = {
        "physics_dt_upper_bound_s": float(physics_dt_s),
        "state_count": len(resampled),
        "duration_s": float(target_time[-1]),
        "maximum_joint_step_rad": float(
            np.max(np.abs(np.diff(resampled, axis=0)))
        ),
        "minimum_path_soft_limit_margin_rad": (
            minimum_path_soft_limit_margin_rad
        ),
    }
    return [row.copy() for row in resampled], document


def _run_sam6d_frame(
    *,
    repository: Path,
    sam6d_root: Path,
    sam6d_python: Path,
    templates: Path,
    cad_mm: Path,
    rgb: Path,
    depth_m: Path,
    depth_mm: Path,
    camera_json: Path,
    output_dir: Path,
    run_pem: bool = True,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "templates").symlink_to(templates.resolve(), target_is_directory=True)
    results = output_dir / "sam6d_results"
    results.mkdir()
    ism_elapsed = _run_logged_process(
        [
            str(sam6d_python),
            "run_inference_custom.py",
            "--segmentor_model",
            "sam",
            "--output_dir",
            str(output_dir),
            "--cad_path",
            str(cad_mm),
            "--rgb_path",
            str(rgb),
            "--depth_path",
            str(depth_mm),
            "--cam_path",
            str(camera_json),
        ],
        cwd=sam6d_root / "Instance_Segmentation_Model",
        log_path=output_dir / "sam6d_ism.log",
    )
    filter_elapsed = _run_logged_process(
        [
            str(sam6d_python),
            str(repository / "src/kcg_connector/isaac/te_sam6d_depth_support_filter.py"),
            "--detections-json",
            str(results / "detection_ism.json"),
            "--depth-m-npy",
            str(depth_m),
            "--camera-json",
            str(camera_json),
            "--output-mask",
            str(results / "best_mask_depth_support_filtered.png"),
            "--output-detections-json",
            str(results / "detection_ism_depth_support_filtered.json"),
            "--output-summary-json",
            str(results / "depth_support_filter_summary.json"),
        ],
        cwd=repository,
        log_path=output_dir / "depth_support_filter.log",
    )
    depth_filter = json.loads(
        (results / "depth_support_filter_summary.json").read_text(
            encoding="utf-8"
        )
    )
    if not run_pem:
        return {
            "score": float(depth_filter["source_detection_score"]),
            "mask": results / "best_mask_depth_support_filtered.png",
            "timing_s": {
                "ism": ism_elapsed,
                "depth_filter": filter_elapsed,
                "total": ism_elapsed + filter_elapsed,
            },
            "depth_filter": depth_filter,
        }
    pem_elapsed = _run_logged_process(
        [
            str(sam6d_python),
            "run_inference_custom.py",
            "--output_dir",
            str(output_dir),
            "--cad_path",
            str(cad_mm),
            "--rgb_path",
            str(rgb),
            "--depth_path",
            str(depth_mm),
            "--cam_path",
            str(camera_json),
            "--seg_path",
            str(results / "detection_ism_depth_support_filtered.json"),
        ],
        cwd=sam6d_root / "Pose_Estimation_Model",
        log_path=output_dir / "sam6d_pem.log",
    )
    detections = json.loads(
        (results / "detection_pem.json").read_text(encoding="utf-8")
    )
    if not detections:
        raise RuntimeError("SAM-6D returned no pose")
    detection = max(detections, key=lambda row: float(row["score"]))
    score = float(detection["score"])
    rotation = np.asarray(detection["R"], dtype=np.float64)
    translation_mm = np.asarray(detection["t"], dtype=np.float64)
    if (
        not math.isfinite(score)
        or rotation.shape != (3, 3)
        or translation_mm.shape != (3,)
        or not np.all(np.isfinite(rotation))
        or not np.all(np.isfinite(translation_mm))
        or not np.allclose(
            rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1.0e-3
        )
        or not math.isclose(
            float(np.linalg.det(rotation)), 1.0, rel_tol=0.0, abs_tol=1.0e-3
        )
    ):
        raise RuntimeError("SAM-6D returned a malformed rigid pose")
    camera_from_object = np.eye(4, dtype=np.float64)
    camera_from_object[:3, :3] = rotation
    camera_from_object[:3, 3] = translation_mm / 1000.0
    return {
        "camera_from_object": camera_from_object,
        "score": score,
        "mask": results / "best_mask_depth_support_filtered.png",
        "timing_s": {
            "ism": ism_elapsed,
            "depth_filter": filter_elapsed,
            "pem": pem_elapsed,
            "total": ism_elapsed + filter_elapsed + pem_elapsed,
        },
        "depth_filter": depth_filter,
    }


def _run_geometry_frame(
    *,
    repository: Path,
    sam6d_root: Path,
    sam6d_python: Path,
    templates: Path,
    cad_mm: Path,
    rgb: Path,
    depth_m: Path,
    depth_mm: Path,
    camera_json: Path,
    camera_matrix: np.ndarray,
    output_dir: Path,
) -> dict[str, object]:
    """Run one independent SAM mask and Five-DoF geometry observation."""
    sam_result = _run_sam6d_frame(
        repository=repository,
        sam6d_root=sam6d_root,
        sam6d_python=sam6d_python,
        templates=templates,
        cad_mm=cad_mm,
        rgb=rgb,
        depth_m=depth_m,
        depth_mm=depth_mm,
        camera_json=camera_json,
        output_dir=output_dir,
        run_pem=False,
    )
    geometry_result = _estimate_five_dof_from_float_depth(
        rgb_path=rgb,
        depth_m_path=depth_m,
        mask_path=Path(sam_result["mask"]),
        intrinsics=camera_matrix,
        mesh_path=cad_mm,
    )
    geometry_elapsed = float(dict(geometry_result["metrics"])["elapsed_s"])
    timing = {
        **dict(sam_result["timing_s"]),
        "geometry": geometry_elapsed,
        "total": float(sam_result["timing_s"]["total"]) + geometry_elapsed,
    }
    return {
        "camera_from_object": np.asarray(
            geometry_result["camera_from_object"], dtype=np.float64
        ),
        "geometry": geometry_result,
        "sam": sam_result,
        "timing_s": timing,
    }


def _first_discrete_collision(
    collision_scene: object,
    arm_positions: np.ndarray,
    hand_positions: np.ndarray,
    obstacles: dict[str, object],
) -> dict[str, object] | None:
    """Return the first collision at one exact commanded joint state."""
    import fcl

    transforms = collision_scene.inputs.robot_model.forward_kinematics(
        tuple(np.concatenate((arm_positions, hand_positions))),
        enforce_limits=False,
    )
    for name, collision_object in collision_scene.objects.items():
        transform = np.asarray(transforms[name], dtype=np.float64)
        collision_object.setTransform(
            fcl.Transform(transform[:3, :3], transform[:3, 3])
        )
    request = fcl.CollisionRequest(num_max_contacts=1, enable_contact=False)
    for first, second in collision_scene.pairs:
        if fcl.collide(
            collision_scene.objects[first],
            collision_scene.objects[second],
            request,
            fcl.CollisionResult(),
        ):
            return {"kind": "self", "pair": [first, second]}
    for obstacle_name, obstacle in obstacles.items():
        for link_name, link_object in collision_scene.objects.items():
            if fcl.collide(
                link_object,
                obstacle,
                request,
                fcl.CollisionResult(),
            ):
                return {
                    "kind": "environment",
                    "obstacle": obstacle_name,
                    "link": link_name,
                }
    return None


def _start_foundationpose_worker(
    *,
    worker_script: Path,
    foundationpose_python: Path,
    foundationpose_root: Path,
    mesh_mm: Path,
    output_dir: Path,
) -> tuple[subprocess.Popen[str], list[str]]:
    command = [
        str(foundationpose_python),
        str(worker_script),
        "--foundationpose-root",
        str(foundationpose_root),
        "--mesh",
        str(mesh_mm),
        "--mesh-scale-to-m",
        "0.001",
        "--register-iterations",
        "5",
        "--track-iterations",
        "2",
        "--debug-dir",
        str(output_dir / "foundationpose_debug"),
    ]
    process = subprocess.Popen(
        command,
        cwd=str(foundationpose_root),
        env=_child_environment(),
        text=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert process.stdout is not None
    log_lines: list[str] = []
    while True:
        line = process.stdout.readline()
        if not line:
            raise RuntimeError("FoundationPose worker exited before becoming ready")
        log_lines.append(line)
        if line.strip() == "FP_READY":
            return process, log_lines


def _foundationpose_request(
    process: subprocess.Popen[str],
    log_lines: list[str],
    request: dict[str, object],
) -> dict[str, object]:
    assert process.stdin is not None and process.stdout is not None
    process.stdin.write(json.dumps(_json_ready(request), allow_nan=False) + "\n")
    process.stdin.flush()
    while True:
        line = process.stdout.readline()
        if not line:
            raise RuntimeError("FoundationPose worker exited without a response")
        log_lines.append(line)
        if line.startswith("FP_RESPONSE "):
            response = json.loads(line[len("FP_RESPONSE ") :])
            if response.get("ok") is not True:
                raise RuntimeError(
                    f"FoundationPose failed: {response.get('error_type')}: "
                    f"{response.get('error')}"
                )
            return response


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--runtime-resources", type=Path, required=True)
    parser.add_argument("--ft-config", type=Path, required=True)
    parser.add_argument("--global-camera-config", type=Path, required=True)
    parser.add_argument("--hand-camera-config", type=Path, required=True)
    parser.add_argument("--receptacle-visual", type=Path, required=True)
    parser.add_argument("--receptacle-position-m", nargs=3, type=float, required=True)
    parser.add_argument(
        "--stop-after-global-capture",
        action="store_true",
        help="capture one global RGB-D frame and stop before perception or motion",
    )
    parser.add_argument(
        "--stop-after-global-localization",
        action="store_true",
        help="run global perception and stop before any robot motion",
    )
    parser.add_argument("--palm-capture-count", type=int, default=3)
    parser.add_argument("--palm-capture-hold-s", type=float, default=0.10)
    parser.add_argument(
        "--stop-after-palm-localization",
        action="store_true",
        help="save and score the stationary palm frames without planning onward",
    )
    parser.add_argument(
        "--servo-estimator",
        choices=("sam6d", "foundationpose", "geometry"),
        help="run the optional no-contact palm-camera servo after handoff",
    )
    parser.add_argument("--sam6d-root", type=Path)
    parser.add_argument("--sam6d-python", type=Path)
    parser.add_argument("--sam6d-templates", type=Path)
    parser.add_argument("--foundationpose-root", type=Path)
    parser.add_argument("--foundationpose-python", type=Path)
    parser.add_argument("--servo-maximum-iterations", type=int, default=20)
    parser.add_argument("--servo-maximum-translation-step-m", type=float, default=0.005)
    parser.add_argument("--servo-maximum-rotation-step-deg", type=float, default=1.0)
    parser.add_argument("--servo-step-duration-s", type=float, default=1.0)
    parser.add_argument("--servo-position-tolerance-m", type=float, default=0.001)
    parser.add_argument("--servo-axis-tolerance-deg", type=float, default=1.0)
    parser.add_argument("--servo-consecutive-required", type=int, default=2)
    parser.add_argument("--servo-collision-samples", type=int, default=11)
    parser.add_argument(
        "--servo-control-policy",
        choices=(
            "simultaneous",
            "axis_then_position",
            "camera_orbit_then_position",
            "frozen_plan",
        ),
        default="simultaneous",
    )
    parser.add_argument("--servo-maximum-estimated-position-offset-m", type=float, default=0.020)
    parser.add_argument("--servo-maximum-estimated-axis-offset-deg", type=float, default=15.0)
    parser.add_argument(
        "--execute-grasp-lift",
        action="store_true",
        help=(
            "after a frozen geometry target reaches pregrasp, execute the "
            "relation-bound finite three-finger grasp, lift, and hold"
        ),
    )
    parser.add_argument(
        "--free-split-object-manifest",
        type=Path,
        help="bound body-plus-rotating-nut simulation asset",
    )
    parser.add_argument(
        "--physical-preflight-evaluation",
        type=Path,
        help="independent q70/4.5 physical binding used for contact-surface audit",
    )
    parser.add_argument("--postgrasp-disturbance-panel", type=Path)
    parser.add_argument("--postgrasp-disturbance-condition")
    parser.add_argument(
        "--nominal-grasp-qualification-evaluation",
        type=Path,
        help="prior successful visual-grasp evaluation authorizing disturbance",
    )
    parser.add_argument(
        "--moveit-pick-ik-rrtconnect",
        action="store_true",
        help=(
            "solve each visual hand-base target with pick_ik global and plan "
            "the arm path with MoveIt RRTConnect"
        ),
    )
    parser.add_argument(
        "--tesseract-kdl-ompl-rrtconnect",
        action="store_true",
        help=(
            "solve each visual hand-base target with Tesseract KDL, use the "
            "direct segment when collision-free, otherwise use OMPL RRTConnect"
        ),
    )
    parser.add_argument("--moveit-ros-domain-id", type=int, default=73)
    parser.add_argument("--moveit-ik-timeout-s", type=float, default=2.0)
    parser.add_argument("--moveit-planning-timeout-s", type=float, default=8.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()

    repository = Path(__file__).resolve().parents[3]
    output = args.output_dir.resolve()
    if args.moveit_pick_ik_rrtconnect and args.tesseract_kdl_ompl_rrtconnect:
        raise ValueError("select exactly one external motion planner")
    external_planner_backend = (
        "moveit"
        if args.moveit_pick_ik_rrtconnect
        else "tesseract"
        if args.tesseract_kdl_ompl_rrtconnect
        else None
    )
    if args.palm_capture_count < 1 or args.palm_capture_count > 5:
        raise ValueError("palm capture count must be between one and five")
    if not math.isfinite(args.palm_capture_hold_s) or args.palm_capture_hold_s <= 0.0:
        raise ValueError("palm capture hold must be positive and finite")
    if args.execute_grasp_lift and (
        args.servo_control_policy != "frozen_plan"
        or args.servo_estimator != "geometry"
        or args.free_split_object_manifest is None
    ):
        raise ValueError(
            "physical continuation requires frozen geometry planning and the "
            "bound split-plug manifest"
        )
    disturbance_arguments = (
        args.postgrasp_disturbance_panel,
        args.postgrasp_disturbance_condition,
        args.nominal_grasp_qualification_evaluation,
    )
    if any(value is not None for value in disturbance_arguments) and not all(
        value is not None for value in disturbance_arguments
    ):
        raise ValueError(
            "postgrasp disturbance requires panel, condition, and nominal "
            "qualification"
        )
    if any(value is not None for value in disturbance_arguments) and (
        not args.execute_grasp_lift
        or args.physical_preflight_evaluation is None
    ):
        raise ValueError(
            "postgrasp disturbance requires physical grasp execution and its "
            "independent preflight binding"
        )
    if args.stop_after_palm_localization and (
        args.servo_control_policy != "frozen_plan"
        or args.servo_estimator != "geometry"
        or args.execute_grasp_lift
    ):
        raise ValueError(
            "palm-localization-only mode requires frozen geometry without grasp"
        )
    diagnostic_stops = (
        args.stop_after_global_capture,
        args.stop_after_global_localization,
        args.stop_after_palm_localization,
    )
    if sum(bool(value) for value in diagnostic_stops) > 1:
        raise ValueError("select only one diagnostic stopping point")
    if args.stop_after_global_localization and (
        args.servo_control_policy != "frozen_plan"
        or args.servo_estimator != "geometry"
    ):
        raise ValueError(
            "global-localization-only mode requires frozen geometry mode"
        )
    if external_planner_backend is not None and (
        args.servo_control_policy != "frozen_plan"
        or args.servo_estimator != "geometry"
        or (
            external_planner_backend == "moveit"
            and (
                args.moveit_ros_domain_id < 0
                or args.moveit_ros_domain_id > 232
            )
        )
        or not math.isfinite(args.moveit_ik_timeout_s)
        or args.moveit_ik_timeout_s <= 0.0
        or not math.isfinite(args.moveit_planning_timeout_s)
        or args.moveit_planning_timeout_s <= 0.0
    ):
        raise ValueError(
            "external planning requires frozen geometry mode and positive "
            "planning timeouts; MoveIt also requires a valid ROS domain"
        )
    if args.servo_estimator is not None:
        if (
            args.servo_control_policy == "frozen_plan"
            and args.servo_estimator != "geometry"
        ):
            raise ValueError("frozen_plan requires the geometry estimator")
        required_paths = {
            "sam6d_root": args.sam6d_root,
            "sam6d_python": args.sam6d_python,
            "sam6d_templates": args.sam6d_templates,
        }
        if args.servo_estimator == "foundationpose":
            required_paths.update(
                {
                    "foundationpose_root": args.foundationpose_root,
                    "foundationpose_python": args.foundationpose_python,
                }
            )
        missing = sorted(name for name, value in required_paths.items() if value is None)
        if missing:
            raise ValueError(f"servo estimator paths are missing: {missing}")
        positive = (
            args.servo_maximum_iterations,
            args.servo_maximum_translation_step_m,
            args.servo_maximum_rotation_step_deg,
            args.servo_step_duration_s,
            args.servo_position_tolerance_m,
            args.servo_axis_tolerance_deg,
            args.servo_consecutive_required,
            args.servo_collision_samples,
            args.servo_maximum_estimated_position_offset_m,
            args.servo_maximum_estimated_axis_offset_deg,
        )
        if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in positive):
            raise ValueError("servo limits must be positive and finite")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite runtime output: {output}")
    output.mkdir(parents=True)

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    collision = plan["collision"]
    if not (
        collision.get("collision_free_at_all_discrete_states") is True
        and int(collision.get("robot_collision_link_count", 0)) == 17
        and plan.get("hardware_authorized") is False
        and plan.get("motion_authorized_by_this_file") is False
    ):
        raise ValueError("handoff plan lacks the complete collision-free contract")
    handoff_arm = np.asarray(plan["handoff_arm_joint_target_rad"], dtype=np.float64)
    handoff_hand = np.asarray(plan["handoff_hand_joint_target_rad"], dtype=np.float64)
    handoff_pose = _matrix4(
        plan["world_from_hand_handoff_row_major"], "handoff world pose"
    )
    if handoff_arm.shape != (7,) or handoff_hand.shape != (4,):
        raise ValueError("handoff joint targets have invalid shape")
    nominal_pregrasp_pose = _matrix4(
        plan["world_from_hand_pregrasp_row_major"], "pregrasp world pose"
    )
    reference_object_pose = _matrix4(
        plan["planned_world_from_object_row_major"], "reference object pose"
    )
    reference_object_to_handoff = (
        np.linalg.inv(reference_object_pose) @ handoff_pose
    )
    reference_object_to_pregrasp = (
        np.linalg.inv(reference_object_pose) @ nominal_pregrasp_pose
    )
    same_run_global_task_frame: np.ndarray | None = None
    provider = json.loads(
        Path(plan["inputs"]["scene_provider_json"]).read_text(
            encoding="utf-8"
        )
    )
    known_scene_geometry = dict(provider["known_static_scene_geometry"])
    relation, relation_path = load_transport_grasp_relation(
        Path(plan["inputs"]["grasp_relation"]), repository
    )
    object_from_hand = _matrix4(
        relation["transform"]["object_from_hand_base_row_major"],
        "object-from-hand relation",
    )

    inputs = load_v2_inputs(
        repository,
        config_path=args.config.resolve(),
        object_id=args.object_id,
    )
    dynamic = dict(inputs.config.section("dynamic"))
    configured_contact_coordination_mode = str(
        dynamic.get("contact_coordination_mode", "sequential")
    )
    relation_hand = relation["hand_contract"]
    physical_grasp = None
    physical_hand_tilt_audit = None
    if args.execute_grasp_lift:
        if relation.get("relation_id") != "te_q70_rear19_equal_normal_4p5_v1":
            raise ValueError("physical continuation is not bound to q70/4.5")
        required_effort = relation_hand.get("required_closing_joint_effort_nm")
        if not isinstance(required_effort, dict) or tuple(required_effort) != (
            "f1j2",
            "f2j1",
            "f3j2",
        ):
            raise ValueError("q70/4.5 relation omits ordered finger efforts")
        dynamic["required_closing_joint_effort_nm"] = [
            float(required_effort[name]) for name in ("f1j2", "f2j1", "f3j2")
        ]
        dynamic["finger_preload_scales"] = [1.0, 1.0, 1.0]
        dynamic["contact_coordination_mode"] = "parallel_contact_latch"
        physical_grasp = runner._registered_grasp(
            inputs,
            args.object_id,
            dynamic,
            float(relation_hand["pregrasp_joint_positions_rad"][0]),
            np.asarray(
                relation_hand["approach_high_seed_arm_positions_rad"],
                dtype=np.float64,
            ),
            float(relation_hand["grasp_axis_position_m"]),
            float(relation_hand["hand_yaw_rad"]),
            tuple(relation_hand["closing_order"]),
        )
        physical_grasp, physical_hand_tilt_audit = (
            runner._apply_hand_tilt_about_object_pivot(
                inputs, physical_grasp, {}
            )
        )
        generated_control = physical_grasp["control_plan"]
        relation_fields = {
            "object_from_hand_row_major": relation["transform"][
                "object_from_hand_base_row_major"
            ],
            "approach_direction_object": relation["transform"][
                "approach_direction_object"
            ],
            "pregrasp_joint_positions_rad": relation_hand[
                "pregrasp_joint_positions_rad"
            ],
            "final_joint_positions_rad": relation_hand[
                "final_joint_positions_rad"
            ],
            "approach_high_seed_arm_positions_rad": relation_hand[
                "approach_high_seed_arm_positions_rad"
            ],
            "closing_order": relation_hand["closing_order"],
        }
        if any(
            not np.allclose(
                np.asarray(generated_control[key], dtype=np.float64),
                np.asarray(value, dtype=np.float64),
                rtol=0.0,
                atol=1.0e-12,
            )
            for key, value in relation_fields.items()
            if key != "closing_order"
        ) or list(generated_control["closing_order"]) != list(
            relation_fields["closing_order"]
        ):
            raise ValueError("q70 relation differs from generated grasp geometry")
    scene_entry = dict(dynamic["object_scenes"][args.object_id])
    split_manifest = None
    if args.free_split_object_manifest is not None:
        manifest_path = args.free_split_object_manifest.resolve()
        scene_entry, split_manifest = _split_scene_entry(
            repository, scene_entry, manifest_path
        )
    robot_asset = (repository / dynamic["robot_asset"]).resolve()
    resources = load_runtime_resources(args.runtime_resources.resolve())
    ft_document = json.loads(args.ft_config.read_text(encoding="utf-8"))
    ft_contract = ft_document["wrist_ft_safety"]
    camera_document = yaml.safe_load(
        args.global_camera_config.read_text(encoding="utf-8")
    )["camera"]
    hand_camera_document = yaml.safe_load(
        args.hand_camera_config.read_text(encoding="utf-8")
    )["camera_rig"]

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": not args.gui,
            "multi_gpu": False,
            "active_gpu": 0,
            "physics_gpu": 0,
            "fast_shutdown": True,
        }
    )
    engine_log_path = runner.current_engine_log_path()
    result: dict[str, object] = {
        "schema_version": (
            "kcg_frozen_target_visual_grasp_runtime_v1"
            if args.execute_grasp_lift
            else "kcg_frozen_target_pregrasp_runtime_v1"
            if args.servo_control_policy == "frozen_plan"
            else (
                "kcg_palm_visual_servo_runtime_v1"
                if args.servo_estimator is not None
                else "kcg_visual_handoff_runtime_v1"
            )
        ),
        "simulation_only": True,
        "hardware_authorized": False,
        "contact_authorized": bool(args.execute_grasp_lift),
        "finger_closure_authorized": bool(args.execute_grasp_lift),
        "lift_authorized": bool(args.execute_grasp_lift),
        "postgrasp_disturbance_authorized": bool(
            args.postgrasp_disturbance_panel is not None
        ),
        "plan": str(args.plan.resolve()),
        "transport_grasp_relation": str(relation_path),
        "servo_estimator": args.servo_estimator,
        "motion_mode": args.servo_control_policy,
        "motion_planner": (
            "PICK_IK_GLOBAL_THEN_MOVEIT2_OMPL_RRTCONNECT"
            if external_planner_backend == "moveit"
            else "TESSERACT_KDL_DIRECT_IF_CLEAR_ELSE_OMPL_RRTCONNECT"
            if external_planner_backend == "tesseract"
            else "LEGACY_BOUNDED_IK_AND_INTERPOLATION"
        ),
        "controlled_pose_components": "POSITION_AND_OUTWARD_AXIS_ONLY",
        "axial_yaw_consumed": False,
    }
    exit_code = 1
    rgbd_resources: dict[
        tuple[str, tuple[int, int]], tuple[object, tuple[object, object]]
    ] = {}
    try:
        import carb.settings
        import fcl
        from isaacsim.core.api import World
        from isaacsim.core.experimental.prims import RigidPrim as TensorRigidPrim
        from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
        from isaacsim.core.simulation_manager import SimulationManager
        from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
        import omni.replicator.core as rep
        from omni.physx import (
            get_physx_interface,
            get_physx_simulation_interface,
        )
        from omni.physx.bindings._physx import SETTING_DISABLE_CONTACT_PROCESSING
        from pxr import (
            Gf,
            PhysxSchema,
            PhysicsSchemaTools,
            Usd,
            UsdGeom,
            UsdLux,
            UsdPhysics,
            UsdShade,
        )

        settings = carb.settings.get_settings()
        settings.set_bool(SETTING_DISABLE_CONTACT_PROCESSING, False)
        World.clear_instance()
        SimulationManager.set_physics_sim_device("cuda:0")
        world = World(
            stage_units_in_meters=1.0,
            physics_dt=float(dynamic["physics_dt_s"]),
            rendering_dt=1.0 / 60.0,
            **gpu_world_parameters(resources),
        )
        context = world.get_physics_context()
        stage = get_current_stage()
        physics_scene_api = PhysxSchema.PhysxSceneAPI.Apply(
            stage.GetPrimAtPath(context.prim_path)
        )
        physics_scene_api.CreateMinVelocityIterationCountAttr().Set(8)
        scene = runner.prepare_dynamic_scene(
            repository,
            stage,
            scene_entry,
            add_reference_to_stage,
            (
                {"contact_friction_coefficient": 0.45}
                if args.execute_grasp_lift
                else {}
            ),
        )
        add_reference_to_stage(str(robot_asset), runner.ROBOT_ROOT)
        if args.execute_grasp_lift:
            runner._apply_contact_friction_perturbation(
                stage, scene, Usd, UsdPhysics, UsdShade, PhysxSchema
            )
            runner._apply_object_mass_perturbation(
                stage, scene, Gf, UsdPhysics
            )
            runner._apply_center_of_mass_perturbation(
                stage, scene, Gf, UsdGeom, UsdPhysics
            )

        plug_mesh = Path(plan["inputs"]["plug_mesh"]).resolve()
        receptacle_mesh = Path(plan["inputs"]["receptacle_mesh"]).resolve()
        fixture = dict(known_scene_geometry["fixture"])
        receptacle_position_m = np.asarray(
            args.receptacle_position_m, dtype=np.float64
        )
        receptacle_visual_binding = _validate_receptacle_visual_binding(
            visual_path=args.receptacle_visual.resolve(),
            collision_mesh_path=receptacle_mesh,
            position_world_m=receptacle_position_m,
            fixture=fixture,
            Usd=Usd,
            UsdGeom=UsdGeom,
        )
        receptacle_root = "/World/TEVisualHandoff/FixedReceptaclePose"
        receptacle_pose = UsdGeom.Xform.Define(stage, receptacle_root)
        receptacle_pose.AddTranslateOp().Set(
            Gf.Vec3d(*receptacle_position_m)
        )
        add_reference_to_stage(
            str(args.receptacle_visual.resolve()),
            receptacle_root + "/OfficialVisual",
        )
        result["receptacle_geometry_binding"] = receptacle_visual_binding
        table_bounds = np.asarray(inputs.table_xy_bounds_m, dtype=np.float64)
        table_size = np.asarray(
            (
                table_bounds[0, 1] - table_bounds[0, 0],
                table_bounds[1, 1] - table_bounds[1, 0],
                1.0,
            ),
            dtype=np.float64,
        )
        table_center = np.asarray(
            (
                np.mean(table_bounds[0]),
                np.mean(table_bounds[1]),
                inputs.table_top_z_m - 0.5,
            ),
            dtype=np.float64,
        )
        static_obstacles: dict[str, object] = {
            "table": fcl.CollisionObject(
                fcl.Box(*table_size), fcl.Transform(table_center)
            )
        }
        static_obstacles["fixture"] = fcl.CollisionObject(
            fcl.Box(*np.asarray(fixture["size_m"], dtype=np.float64)),
            fcl.Transform(np.asarray(fixture["center_world_m"], dtype=np.float64)),
        )
        receptacle_world = np.eye(4, dtype=np.float64)
        receptacle_world[:3, 3] = receptacle_position_m
        receptacle_obstacle, runtime_receptacle_cylinder = _cylinder_from_mesh(
            receptacle_mesh, RECEPTACLE_MESH_SCALE_TO_M, receptacle_world
        )
        planned_receptacle_cylinder = dict(collision["receptacle_cylinder"])
        if any(
            not math.isclose(
                float(runtime_receptacle_cylinder[key]),
                float(planned_receptacle_cylinder[key]),
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
            for key in ("radius_m", "z_min_m", "z_max_m")
        ):
            raise ValueError(
                "runtime receptacle collision geometry differs from the plan"
            )
        result["receptacle_geometry_binding"]["collision_cylinder"] = (
            runtime_receptacle_cylinder
        )
        static_obstacles["receptacle"] = receptacle_obstacle
        collision_scene = FullRobotCollisionScene(inputs)
        plug_bounds = collision["plug_yaw_swept_cylinder"]
        plug_center_local_z_m = 0.5 * (
            float(plug_bounds["z_min_m"]) + float(plug_bounds["z_max_m"])
        )
        planned_contact_lever_m = float(
            np.linalg.norm(object_from_hand[:3, 3])
            + float(plug_bounds["radius_m"])
        )
        planned_contact_wrist_torque_limit_nm = float(
            ft_contract["maximum_resultant_force_n"]
        ) * planned_contact_lever_m

        resolution = tuple(int(value) for value in camera_document["resolution_px"])
        global_path = str(camera_document["prim_path"])
        global_world_from_camera = _camera_cv_pose_from_eye_target(
            camera_document["eye_world_m"], camera_document["target_world_m"]
        )
        _author_camera(
            stage,
            global_path,
            global_world_from_camera,
            resolution=resolution,
            focal_length_mm=float(camera_document["focal_length_mm"]),
            horizontal_aperture_mm=float(camera_document["horizontal_aperture_mm"]),
            clipping_range_m=tuple(camera_document["clipping_range_m"]),
            Gf=Gf,
            UsdGeom=UsdGeom,
        )
        palm_definition = hand_camera_document["palm"]
        palm_path = runner.HAND_BASE_PATH + str(palm_definition["prim_suffix"])
        palm_t_hc = np.asarray(palm_definition["T_HC_cv"], dtype=np.float64)
        _author_camera(
            stage,
            palm_path,
            palm_t_hc,
            resolution=tuple(hand_camera_document["resolution_px"]),
            focal_length_mm=float(hand_camera_document["focal_length_mm"]),
            horizontal_aperture_mm=float(hand_camera_document["horizontal_aperture_mm"]),
            clipping_range_m=tuple(hand_camera_document["clipping_range_m"]),
            Gf=Gf,
            UsdGeom=UsdGeom,
        )
        light_root = "/World/TEFoundationPoseHandoff/Lights"
        dome = UsdLux.DomeLight.Define(stage, light_root + "/Fill")
        dome.CreateIntensityAttr(900.0)
        key = UsdLux.DistantLight.Define(stage, light_root + "/Key")
        key.CreateIntensityAttr(700.0)
        UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(-35.0, 20.0, 25.0))

        ft_articulation = world.scene.add(
            SingleArticulation(
                prim_path=runner.ARTICULATION_PATH,
                name="visual_handoff_ft_reader",
            )
        )
        object_parts: tuple[object, ...] = ()
        tensor_contact_prim = None
        tensor_contact_sensor_paths: tuple[str, ...] = ()
        contact_report_complete = False
        physical_engine_monitor = None
        if args.execute_grasp_lift:
            rigid_body_prims = []
            contact_report_prims = []
            for prim in stage.Traverse():
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    path = str(prim.GetPath())
                    rigid_body_prims.append(path)
                    PhysxSchema.PhysxContactReportAPI.Apply(
                        prim
                    ).CreateThresholdAttr().Set(0.0)
                    contact_report_prims.append(path)
            hand_base_prim = stage.GetPrimAtPath(runner.HAND_BASE_PATH)
            if not hand_base_prim.IsValid():
                raise RuntimeError("physical visual grasp hand base is missing")
            object_parts = tuple(
                world.scene.add(
                    SingleRigidPrim(
                        prim_path=path,
                        name=f"visual_grasp_object_part_{index}",
                    )
                )
                for index, path in enumerate(scene["part_prim_paths"])
            )
            robot_contact_paths = tuple(
                path
                for path in rigid_body_prims
                if path == runner.ROBOT_ROOT
                or path.startswith(runner.ROBOT_ROOT + "/")
            )
            object_contact_paths = tuple(map(str, scene["part_prim_paths"]))
            tensor_contact_sensor_paths = (
                robot_contact_paths + object_contact_paths
            )
            tensor_contact_prim = TensorRigidPrim(
                list(tensor_contact_sensor_paths),
                resolve_paths=False,
                contact_filter_paths=list(object_contact_paths),
                max_contact_count=runner.TENSOR_CONTACT_MAX_COUNT,
            )
            result["physical_contact_setup"] = {
                "object_part_paths": list(object_contact_paths),
                "legal_grasp_contact_paths": list(
                    scene["legal_grasp_contact_paths"]
                ),
                "tensor_contact_sensor_path_count": len(
                    tensor_contact_sensor_paths
                ),
            }
        context.set_gravity(float(scene["gravity_m_s2"]))
        world.reset()
        if args.execute_grasp_lift:
            assert tensor_contact_prim is not None
            if not tensor_contact_prim.is_physics_tensor_entity_valid():
                raise RuntimeError("physical visual grasp contact view is invalid")
            rigid_after = [
                str(prim.GetPath())
                for prim in stage.Traverse()
                if prim.HasAPI(UsdPhysics.RigidBodyAPI)
            ]
            reports_after = [
                str(prim.GetPath())
                for prim in stage.Traverse()
                if prim.HasAPI(PhysxSchema.PhysxContactReportAPI)
            ]
            contact_report_complete = bool(
                set(rigid_body_prims)
                == set(contact_report_prims)
                == set(rigid_after)
                == set(reports_after)
            )
            if not contact_report_complete:
                raise RuntimeError("physical visual grasp contact reporters differ")
            assert split_manifest is not None
            result["physical_scene_binding"] = {
                "split_manifest": str(args.free_split_object_manifest.resolve()),
                "total_mass_kg": float(
                    split_manifest["mass_model"]["total_mass_kg"]
                ),
                "contact_friction_coefficient": 0.45,
                "unloaded_nut_resistance_nm": float(
                    split_manifest["joint"]["rotational_resistance"][
                        "assumed_resisting_torque_nm"
                    ]
                ),
                "wrist_force_limit_n": float(
                    ft_contract["maximum_resultant_force_n"]
                ),
                "wrist_torque_limit_nm": float(
                    ft_contract["maximum_resultant_torque_nm"]
                ),
                "planned_contact_wrist_torque_limit_nm": (
                    planned_contact_wrist_torque_limit_nm
                ),
                "planned_contact_lever_m": planned_contact_lever_m,
                "planned_contact_limit_derivation": (
                    "FREE_SPACE_FORCE_LIMIT_TIMES_OBJECT_ORIGIN_DISTANCE_"
                    "PLUS_YAW_SWEPT_RADIUS"
                ),
                "finger_effort_abort_nm": float(
                    dynamic["measured_effort_abort_nm"]
                ),
            }
        result["physics_backend"] = gpu_backend_record(world, context)
        if result["physics_backend"]["pass"] is not True:
            raise RuntimeError("GPU physics backend audit failed")

        world.pause()
        simulation_app.update()
        result["global_before"] = _capture_rgbd(
            rep=rep,
            resources=rgbd_resources,
            camera_path=global_path,
            resolution=resolution,
            output_dir=output / "global_before",
            warmup_frames=int(camera_document["warmup_frames"]),
            rt_subframes=4,
        )
        if (
            int(result["global_before"]["finite_positive_depth_pixels"]) < 100
            or float(result["global_before"]["rgb_standard_deviation"]) < 1.0
        ):
            raise RuntimeError("global RGB-D capture is unusable")
        if args.stop_after_global_capture:
            result["result_scope"] = (
                "one corrected global RGB-D scene capture; no perception, "
                "planning, finger motion, contact, lift, or disturbance"
            )
            result["truth_inputs_used_for_control"] = []
            raise _GlobalCaptureOnlyComplete
        if args.servo_control_policy == "frozen_plan":
            if external_planner_backend is None:
                raise RuntimeError(
                    "same-run global localization requires an external planner"
                )
            assert args.sam6d_root is not None
            assert args.sam6d_python is not None
            assert args.sam6d_templates is not None
            global_resolution = tuple(camera_document["resolution_px"])
            global_focal_pixels = (
                global_resolution[0]
                * float(camera_document["focal_length_mm"])
                / float(camera_document["horizontal_aperture_mm"])
            )
            global_camera_matrix = np.asarray(
                (
                    (global_focal_pixels, 0.0, 0.5 * global_resolution[0]),
                    (0.0, global_focal_pixels, 0.5 * global_resolution[1]),
                    (0.0, 0.0, 1.0),
                ),
                dtype=np.float64,
            )
            global_camera_json = output / "global_camera.json"
            global_camera_json.write_text(
                json.dumps(
                    {
                        "cam_K": global_camera_matrix.ravel().tolist(),
                        "depth_scale": 1,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            global_perception = _run_sam6d_frame(
                repository=repository,
                sam6d_root=args.sam6d_root.resolve(),
                sam6d_python=args.sam6d_python,
                templates=args.sam6d_templates.resolve(),
                cad_mm=Path(plan["inputs"]["plug_mesh"]).resolve(),
                rgb=output / "global_before" / "rgb.png",
                depth_m=output / "global_before" / "depth_m.npy",
                depth_mm=output / "global_before" / "depth_mm.png",
                camera_json=global_camera_json,
                output_dir=output / "perception" / "global_current",
                run_pem=True,
            )
            global_world_from_camera = _world_from_prim_cv(
                stage, global_path, Usd=Usd, UsdGeom=UsdGeom
            )
            global_world_from_object_pem = (
                global_world_from_camera
                @ np.asarray(
                    global_perception["camera_from_object"], dtype=np.float64
                )
            )
            global_geometry = _estimate_five_dof_from_float_depth(
                rgb_path=output / "global_before" / "rgb.png",
                depth_m_path=output / "global_before" / "depth_m.npy",
                mask_path=Path(global_perception["mask"]),
                intrinsics=global_camera_matrix,
                mesh_path=Path(plan["inputs"]["plug_mesh"]).resolve(),
            )
            global_world_from_object_geometry = (
                global_world_from_camera
                @ np.asarray(
                    global_geometry["camera_from_object"], dtype=np.float64
                )
            )
            global_consistency = _pem_geometry_consistency(
                pem_world_from_object=global_world_from_object_pem,
                geometry_world_from_object=global_world_from_object_geometry,
                geometry_metrics=dict(global_geometry["metrics"]),
            )
            global_geometry_metrics = dict(global_geometry["metrics"])
            global_geometry_position = global_world_from_object_geometry[:3, 3]
            global_geometry_axis = global_world_from_object_geometry[:3, 2]
            maximum_support_offset_m = 0.25 * float(
                global_geometry_metrics["visible_face_radius_m"]
            )
            inside_table_xy = bool(
                table_bounds[0, 0] <= global_geometry_position[0]
                <= table_bounds[0, 1]
                and table_bounds[1, 0] <= global_geometry_position[1]
                <= table_bounds[1, 1]
            )
            support_offset_m = abs(
                float(global_geometry_position[2]) - float(inputs.table_top_z_m)
            )
            plug_receptacle_xy_distance_m = float(
                np.linalg.norm(
                    global_geometry_position[:2] - receptacle_position_m[:2]
                )
            )
            minimum_plug_receptacle_xy_distance_m = float(
                collision["plug_yaw_swept_cylinder"]["radius_m"]
                + collision["receptacle_cylinder"]["radius_m"]
            )
            global_scene_consistency = {
                "pass": bool(
                    inside_table_xy
                    and support_offset_m <= maximum_support_offset_m
                    and plug_receptacle_xy_distance_m
                    > minimum_plug_receptacle_xy_distance_m
                    and float(global_geometry_axis[2]) < 0.0
                ),
                "inside_table_xy": inside_table_xy,
                "object_origin_support_offset_mm": 1000.0 * support_offset_m,
                "maximum_object_origin_support_offset_mm": (
                    1000.0 * maximum_support_offset_m
                ),
                "plug_receptacle_xy_distance_mm": (
                    1000.0 * plug_receptacle_xy_distance_m
                ),
                "minimum_plug_receptacle_xy_distance_mm": (
                    1000.0 * minimum_plug_receptacle_xy_distance_m
                ),
                "axis_points_from_visible_face_toward_table": bool(
                    float(global_geometry_axis[2]) < 0.0
                ),
                "support_limit_derivation": "ONE_QUARTER_VISIBLE_FACE_RADIUS",
                "known_scene_inputs": [
                    "table_xy_bounds",
                    "table_top_z",
                    "fixed_receptacle_position",
                    "CAD_collision_radii",
                ],
                "simulator_object_truth_used": False,
            }
            same_run_global_task_frame = _yaw_free_object_frame(
                global_world_from_object_geometry[:3, 3],
                global_world_from_object_geometry[:3, 2],
            )
            handoff_pose = (
                same_run_global_task_frame @ reference_object_to_handoff
            )
            nominal_pregrasp_pose = (
                same_run_global_task_frame @ reference_object_to_pregrasp
            )
            reference_object_pose = same_run_global_task_frame
            result["global_localization"] = {
                "pose_source": (
                    "CURRENT_RUN_SAM_MASK_FLOAT_DEPTH_FIVE_DOF_GEOMETRY"
                ),
                "sam6d_score": float(global_perception["score"]),
                "timing_s": dict(global_perception["timing_s"]),
                "depth_filter": dict(global_perception["depth_filter"]),
                "pem_geometry_consistency": global_consistency,
                "known_scene_consistency": global_scene_consistency,
                "world_from_camera_cv_row_major": (
                    global_world_from_camera.ravel().tolist()
                ),
                "pem_world_from_object_row_major": (
                    global_world_from_object_pem.ravel().tolist()
                ),
                "geometry_world_from_object_raw_row_major": (
                    global_world_from_object_geometry.ravel().tolist()
                ),
                "world_from_object_raw_row_major": (
                    global_world_from_object_geometry.ravel().tolist()
                ),
                "world_from_object_yaw_free_row_major": (
                    same_run_global_task_frame.ravel().tolist()
                ),
                "controlled_pose_components": "POSITION_AND_DIRECTED_AXIS_ONLY",
                "axial_yaw_consumed": False,
                "pem_used_for_control": False,
                "geometry_metrics": global_geometry_metrics,
                "saved_global_pose_used_as_current_object_pose": False,
                "truth_inputs_used_for_control": [],
            }
            if global_consistency["pass"] is not True:
                raise RuntimeError(
                    "global SAM-6D PEM disagrees with mask/depth geometry; "
                    "refusing to plan motion"
                )
            if global_scene_consistency["pass"] is not True:
                raise RuntimeError(
                    "global mask/depth geometry is inconsistent with the known "
                    "tabletop plug workspace; refusing to plan motion"
                )
            if args.stop_after_global_localization:
                result["result_scope"] = (
                    "one corrected global RGB-D localization with SAM mask, "
                    "float-depth geometry, and PEM consistency checking; no "
                    "robot motion, finger motion, contact, lift, or disturbance"
                )
                result["truth_inputs_used_for_control"] = []
                raise _GlobalCaptureOnlyComplete
        world.play()

        command_counter: dict[str, int] = {}
        robot_data = control.create_native_gravity_compensated_robot(
            runner.ARTICULATION_PATH,
            runner.EXPECTED_DOF_NAMES,
            dynamic,
            command_api_counter=command_counter,
        )
        robot, active_indices, arm_indices, lower, upper, drive_audit = robot_data
        metadata = ft_articulation._articulation_view._metadata
        joint_indices = dict(metadata.joint_indices)
        reaction_row = int(joint_indices[ft_contract["source_joint"]]) + 1
        hand_inertials = runner._load_frozen_hand_inertials(
            repository / ft_document["source_evidence"]["hand_inertial_source"]["path"]
        )
        ft_auditor = runner._HighObservationWristFtAuditor(
            ft_articulation=ft_articulation,
            reaction_row=reaction_row,
            robot_model=inputs.robot_model,
            hand_inertials=hand_inertials,
            gravity_m_s2=float(scene["gravity_m_s2"]),
            physics_dt_s=float(dynamic["physics_dt_s"]),
            task_rotation_world=handoff_pose[:3, :3],
            force_limit_n=float(ft_contract["maximum_resultant_force_n"]),
            torque_limit_nm=float(ft_contract["maximum_resultant_torque_nm"]),
            planned_contact_torque_limit_nm=(
                planned_contact_wrist_torque_limit_nm
                if args.execute_grasp_lift
                else None
            ),
            dynamic_inertia_compensation_enabled=True,
            frozen_hand_positions=handoff_hand,
            dynamic_inertia_enabled_phases=(
                "approach_above",
                "wait_above_settled",
                "observation_wait_hold",
                "visual_servo",
                "frozen_target_plan",
                "settle",
                "pregrasp_hold",
                "tare",
            ),
        )
        payload_model = None
        if args.execute_grasp_lift:
            assert split_manifest is not None
            part_rows = tuple(split_manifest["parts"].values())
            part_masses = np.asarray(
                [row["mass_properties"]["mass_kg"] for row in part_rows],
                dtype=np.float64,
            )
            part_centers = np.asarray(
                [
                    row["mass_properties"]["center_of_mass_m"]
                    for row in part_rows
                ],
                dtype=np.float64,
            )
            total_mass = float(np.sum(part_masses))
            center_object = np.average(
                part_centers, axis=0, weights=part_masses
            )
            center_hand = (
                np.linalg.inv(object_from_hand)
                @ np.concatenate((center_object, (1.0,)))
            )[:3]
            payload_model = {
                "method": (
                    "SPLIT_CAD_MASS_COM_JACOBIAN_FEEDFORWARD_RAMPED_"
                    "OVER_TABLE_CLEARANCE"
                ),
                "mass_kg": total_mass,
                "gravity_m_s2": abs(float(scene["gravity_m_s2"])),
                "center_of_mass_object_m": center_object.tolist(),
                "center_of_mass_from_hand_m": center_hand.tolist(),
                "transfer_distance_m": float(
                    dynamic["table_release_clearance_m"]
                ),
                "source": "BOUND_SPLIT_ASSET_MANIFEST",
                "online_object_truth_used": False,
            }
        stepper = control.JointSignalStepper(
            robot=robot,
            world=world,
            auditor=ft_auditor,
            active_indices=active_indices,
            arm_indices=arm_indices,
            arm_lower_limits=lower,
            arm_upper_limits=upper,
            settings=dynamic,
            render=False,
            robot_model=(inputs.robot_model if args.execute_grasp_lift else None),
            payload_model=payload_model,
            command_api_counter=command_counter,
        )
        ft_auditor.stepper = stepper
        dt = float(dynamic["physics_dt_s"])
        home_arm = np.zeros(7, dtype=np.float64)
        home_hand = np.zeros(4, dtype=np.float64)
        tare_steps = round(float(ft_contract["tare_duration_s"]) / dt)
        for _ in stepper.active_steps(tare_steps):
            stepper.advance("ft_free_space_tare", home_arm, home_hand)
        if stepper.abort_reason is None:
            ft_auditor.finalize_tare(int(ft_contract["minimum_tare_samples"]))

        handoff_planned_states = None
        if external_planner_backend is not None:
            handoff_collision_objects = _moveit_collision_documents(
                collision_contract=collision,
                known_scene_geometry=known_scene_geometry,
                world_from_object=reference_object_pose,
                receptacle_position_m=args.receptacle_position_m,
            )
            handoff_planned_states, handoff_plan_record = (
                _run_external_pose_plan(
                    backend=external_planner_backend,
                    repository=repository,
                    ros_domain_id=args.moveit_ros_domain_id,
                    stage_name="global_to_palm_handoff",
                    output_dir=output,
                    start_arm=home_arm,
                    hand_positions=handoff_hand,
                    target_world_from_hand=handoff_pose,
                    collision_objects=handoff_collision_objects,
                    physics_dt_s=dt,
                    ik_timeout_s=args.moveit_ik_timeout_s,
                    planning_timeout_s=args.moveit_planning_timeout_s,
                )
            )
            result["handoff_motion_plan"] = _json_ready(
                handoff_plan_record
            )
            reference_plug_obstacle, _ = _cylinder_from_mesh(
                plug_mesh, 0.001, reference_object_pose
            )
            handoff_obstacles = {
                **static_obstacles,
                "plug": reference_plug_obstacle,
            }
            first_handoff_collision = None
            for state_index, state in enumerate(handoff_planned_states):
                collision_at_state = _first_discrete_collision(
                    collision_scene, state, handoff_hand, handoff_obstacles
                )
                if collision_at_state is not None:
                    first_handoff_collision = {
                        "state_index": state_index,
                        **collision_at_state,
                    }
                    break
            handoff_arm = np.asarray(
                handoff_planned_states[-1], dtype=np.float64
            )
            planned_handoff_fk = np.asarray(
                inputs.robot_model.forward_kinematics(
                    tuple(np.concatenate((handoff_arm, handoff_hand))),
                    enforce_limits=False,
                )["handbase_link"],
                dtype=np.float64,
            )
            hard_margin = float(
                min(
                    np.min(np.minimum(state - lower, upper - state))
                    for state in handoff_planned_states
                )
            )
            handoff_plan_record["execution_fcl_check"] = {
                "all_240hz_states_checked": True,
                "state_count": len(handoff_planned_states),
                "first_collision": first_handoff_collision,
                "minimum_hard_joint_limit_margin_rad": hard_margin,
                "endpoint_position_error_m": float(
                    np.linalg.norm(
                        planned_handoff_fk[:3, 3] - handoff_pose[:3, 3]
                    )
                ),
                "endpoint_orientation_error_rad": float(
                    np.linalg.norm(
                        Rotation.from_matrix(
                            handoff_pose[:3, :3].T
                            @ planned_handoff_fk[:3, :3]
                        ).as_rotvec()
                    )
                ),
            }
            result["handoff_motion_plan"] = _json_ready(
                handoff_plan_record
            )
            if first_handoff_collision is not None:
                raise RuntimeError(
                    "external handoff path failed the authoritative FCL check"
                )

        motion_plan = {
            "approach_arm_waypoints_rad": (handoff_arm,),
            "pregrasp_arm_positions_rad": handoff_arm,
            "pregrasp_hand_positions_rad": handoff_hand,
        }
        handoff_result = None
        if stepper.abort_reason is None:
            handoff_result = control.run_pregrasp_sequence(
                stepper,
                motion_plan,
                dynamic,
                stop_after_approach_high=True,
                observation_wait_hold_duration_s=0.5,
                approach_high_motion_duration_s=float(
                    plan["trajectory"]["approach_duration_s"]
                ),
                approach_high_arm_states=handoff_planned_states,
            )
        actual = robot.get_dof_positions(indices=0).numpy()[0][active_indices]
        actual_fk = np.asarray(
            inputs.robot_model.forward_kinematics(
                tuple(actual), enforce_limits=False
            )["handbase_link"],
            dtype=np.float64,
        )
        result.update(
            {
                "abort_reason": stepper.abort_reason,
                "handoff_result": _json_ready(handoff_result),
                "completed_command_steps": int(stepper.step_index),
                "command_api_counts": command_counter,
                "maximum_joint_speed_rad_s": float(stepper.maximum_speed),
                "maximum_arm_tracking_error_rad": float(stepper.maximum_arm_error),
                "arrival_arm_joint_error_rad": float(
                    np.max(np.abs(actual[:7] - handoff_arm))
                ),
                "arrival_position_error_m": float(
                    np.linalg.norm(actual_fk[:3, 3] - handoff_pose[:3, 3])
                ),
                "ft": ft_auditor.summary(),
                "native_drive_audit": _json_ready(drive_audit),
            }
        )
        reached = bool(
            stepper.abort_reason is None
            and handoff_result is not None
            and handoff_result.get("high_wait_reached_without_abort") is True
        )
        result["handoff_reached_and_stopped"] = reached
        if not reached:
            raise RuntimeError(f"handoff motion failed: {stepper.abort_reason}")

        world.pause()
        simulation_app.update()
        result["global_handoff"] = _capture_rgbd(
            rep=rep,
            resources=rgbd_resources,
            camera_path=global_path,
            resolution=resolution,
            output_dir=output / "global_handoff",
            warmup_frames=int(camera_document["warmup_frames"]),
            rt_subframes=4,
        )
        palm_capture_path = "/World/TEVisualHandoff/PalmCaptureCamera"
        palm_resolution = tuple(hand_camera_document["resolution_px"])
        capture_hold_steps = round(float(args.palm_capture_hold_s) / dt)
        if capture_hold_steps < 1:
            raise ValueError("palm capture hold is shorter than one physics step")
        palm_frames: list[dict[str, object]] = []
        palm_world_from_camera = None
        for frame_index in range(int(args.palm_capture_count)):
            if frame_index > 0:
                world.play()
                for _ in stepper.active_steps(capture_hold_steps):
                    stepper.advance(
                        "handoff_capture_hold", handoff_arm, handoff_hand
                    )
                if stepper.abort_reason is not None:
                    raise RuntimeError(
                        f"handoff hold failed: {stepper.abort_reason}"
                    )
                world.pause()
                simulation_app.update()
            actual = robot.get_dof_positions(indices=0).numpy()[0][active_indices]
            actual_fk = np.asarray(
                inputs.robot_model.forward_kinematics(
                    tuple(actual), enforce_limits=False
                )["handbase_link"],
                dtype=np.float64,
            )
            palm_world_from_camera = actual_fk @ palm_t_hc
            _author_camera(
                stage,
                palm_capture_path,
                palm_world_from_camera,
                resolution=palm_resolution,
                focal_length_mm=float(hand_camera_document["focal_length_mm"]),
                horizontal_aperture_mm=float(
                    hand_camera_document["horizontal_aperture_mm"]
                ),
                clipping_range_m=tuple(hand_camera_document["clipping_range_m"]),
                Gf=Gf,
                UsdGeom=UsdGeom,
            )
            simulation_app.update()
            frame_metrics = _capture_rgbd(
                rep=rep,
                resources=rgbd_resources,
                camera_path=palm_capture_path,
                resolution=palm_resolution,
                output_dir=output / "palm_handoff" / f"frame_{frame_index:03d}",
                warmup_frames=(
                    int(camera_document["warmup_frames"])
                    if frame_index == 0
                    else 1
                ),
                rt_subframes=4,
            )
            frame_metrics["frame_index"] = frame_index
            frame_metrics["world_from_camera_cv_row_major"] = (
                palm_world_from_camera.ravel().tolist()
            )
            frame_metrics["maximum_arm_joint_error_rad"] = float(
                np.max(np.abs(actual[:7] - handoff_arm))
            )
            palm_frames.append(frame_metrics)
        result["palm_handoff_frames"] = palm_frames
        for label in ("global_before", "global_handoff"):
            metrics = result[label]
            if (
                int(metrics["finite_positive_depth_pixels"]) < 100
                or float(metrics["rgb_standard_deviation"]) < 1.0
            ):
                raise RuntimeError(f"{label} RGB-D capture is unusable")
        for metrics in palm_frames:
            if (
                int(metrics["finite_positive_depth_pixels"]) < 100
                or float(metrics["rgb_standard_deviation"]) < 1.0
            ):
                raise RuntimeError(
                    f"palm frame {metrics['frame_index']} RGB-D capture is unusable"
                )
        result["global_world_from_camera_cv_row_major"] = (
            _world_from_prim_cv(stage, global_path, Usd=Usd, UsdGeom=UsdGeom)
            .ravel()
            .tolist()
        )
        result["palm_world_from_camera_cv_row_major"] = (
            palm_world_from_camera.ravel().tolist()
        )
        focal_length = float(hand_camera_document["focal_length_mm"])
        horizontal_aperture = float(
            hand_camera_document["horizontal_aperture_mm"]
        )
        focal_pixels = palm_resolution[0] * focal_length / horizontal_aperture
        result["palm_camera_intrinsics_3x3"] = [
            [focal_pixels, 0.0, 0.5 * palm_resolution[0]],
            [0.0, focal_pixels, 0.5 * palm_resolution[1]],
            [0.0, 0.0, 1.0],
        ]
        result["palm_camera_pose_source"] = (
            "measured_joint_positions_to_robot_fk_times_fixed_hand_eye_extrinsic"
        )
        handoff_actual = np.array(actual, dtype=np.float64, copy=True)
        handoff_actual_fk = np.array(actual_fk, dtype=np.float64, copy=True)

        if args.servo_estimator is not None:
            import fcl

            assert args.sam6d_root is not None
            assert args.sam6d_python is not None
            assert args.sam6d_templates is not None
            camera_matrix = np.asarray(
                result["palm_camera_intrinsics_3x3"], dtype=np.float64
            )
            palm_camera_json = output / "palm_camera.json"
            palm_camera_json.write_text(
                json.dumps(
                    {
                        "cam_K": camera_matrix.ravel().tolist(),
                        "depth_scale": 1,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            plug_mesh = Path(plan["inputs"]["plug_mesh"]).resolve()
            receptacle_mesh = Path(plan["inputs"]["receptacle_mesh"]).resolve()
            provider = json.loads(
                Path(plan["inputs"]["scene_provider_json"]).read_text(
                    encoding="utf-8"
                )
            )
            geometry = provider["known_static_scene_geometry"]
            table_bounds = np.asarray(inputs.table_xy_bounds_m, dtype=np.float64)
            table_size = np.asarray(
                (
                    table_bounds[0, 1] - table_bounds[0, 0],
                    table_bounds[1, 1] - table_bounds[1, 0],
                    1.0,
                ),
                dtype=np.float64,
            )
            table_center = np.asarray(
                (
                    np.mean(table_bounds[0]),
                    np.mean(table_bounds[1]),
                    inputs.table_top_z_m - 0.5,
                ),
                dtype=np.float64,
            )
            static_obstacles: dict[str, object] = {
                "table": fcl.CollisionObject(
                    fcl.Box(*table_size), fcl.Transform(table_center)
                )
            }
            fixture = geometry["fixture"]
            static_obstacles["fixture"] = fcl.CollisionObject(
                fcl.Box(*np.asarray(fixture["size_m"], dtype=np.float64)),
                fcl.Transform(np.asarray(fixture["center_world_m"], dtype=np.float64)),
            )
            receptacle_world = np.eye(4, dtype=np.float64)
            receptacle_world[:3, 3] = np.asarray(
                args.receptacle_position_m, dtype=np.float64
            )
            receptacle_obstacle, _ = _cylinder_from_mesh(
                receptacle_mesh, 0.001, receptacle_world
            )
            static_obstacles["receptacle"] = receptacle_obstacle
            collision_scene = FullRobotCollisionScene(inputs)
            plug_bounds = collision["plug_yaw_swept_cylinder"]
            plug_center_local_z_m = 0.5 * (
                float(plug_bounds["z_min_m"]) + float(plug_bounds["z_max_m"])
            )
            camera_from_hand = np.linalg.inv(palm_t_hc)

            worker_process: subprocess.Popen[str] | None = None
            worker_log_lines: list[str] = []
            geometry_world_history: list[tuple[np.ndarray, np.ndarray]] = []

            servo_frames: list[dict[str, object]] = []
            termination = "UNSET"
            consecutive = 0
            motion_count = 0
            servo_started = time.perf_counter()
            frozen_plan_record: dict[str, object] | None = None
            full_physical_motion_plan: dict[str, object] | None = None
            physical_precontact_hand_pose: np.ndarray | None = None
            physical_precontact_truth_world_from_object: np.ndarray | None = None
            reference_axis = np.asarray(
                reference_object_pose[:3, 2], dtype=np.float64
            )
            reference_axis /= np.linalg.norm(reference_axis)
            maximum_rotation_step = math.radians(
                float(args.servo_maximum_rotation_step_deg)
            )
            axis_tolerance = math.radians(float(args.servo_axis_tolerance_deg))
            maximum_axis_offset = math.radians(
                float(args.servo_maximum_estimated_axis_offset_deg)
            )
            servo_capture_path = "/World/TEVisualHandoff/PalmServoCaptureCamera"

            try:
                if args.servo_control_policy == "frozen_plan":
                    if len(palm_frames) < 3:
                        raise ValueError(
                            "frozen_plan requires at least three stationary palm frames"
                        )
                    actual = robot.get_dof_positions(indices=0).numpy()[0][active_indices]
                    actual_fk = np.asarray(
                        inputs.robot_model.forward_kinematics(
                            tuple(actual), enforce_limits=False
                        )["handbase_link"],
                        dtype=np.float64,
                    )

                    def observe_frozen_frame(
                        *,
                        frame_index: int,
                        frame_dir: Path,
                        camera_world: np.ndarray,
                        role: str,
                        perception_name: str,
                        actual_hand_pose: np.ndarray,
                    ) -> tuple[np.ndarray, dict[str, object]]:
                        observation = _run_geometry_frame(
                            repository=repository,
                            sam6d_root=args.sam6d_root.resolve(),
                            sam6d_python=args.sam6d_python,
                            templates=args.sam6d_templates.resolve(),
                            cad_mm=plug_mesh,
                            rgb=frame_dir / "rgb.png",
                            depth_m=frame_dir / "depth_m.npy",
                            depth_mm=frame_dir / "depth_mm.png",
                            camera_json=palm_camera_json,
                            camera_matrix=camera_matrix,
                            output_dir=output / "perception" / perception_name,
                        )
                        camera_from_object = np.asarray(
                            observation["camera_from_object"], dtype=np.float64
                        )
                        unfused_world = camera_world @ camera_from_object
                        raw_axis = np.asarray(unfused_world[:3, 2], dtype=np.float64)
                        raw_axis /= np.linalg.norm(raw_axis)
                        if float(raw_axis @ reference_axis) < 0.0:
                            raise RuntimeError(
                                "palm float-depth geometry returned a directed "
                                "axis opposite to the validated global axis"
                            )
                        geometry_world_history.append(
                            (unfused_world[:3, 3].copy(), raw_axis)
                        )
                        fused_position = np.median(
                            np.stack([item[0] for item in geometry_world_history]),
                            axis=0,
                        )
                        fused_axis = np.median(
                            np.stack([item[1] for item in geometry_world_history]),
                            axis=0,
                        )
                        fused_axis /= np.linalg.norm(fused_axis)
                        fused_task = _yaw_free_object_frame(
                            fused_position, fused_axis
                        )
                        sam_result = dict(observation["sam"])
                        geometry_result = dict(observation["geometry"])
                        record: dict[str, object] = {
                            "iteration": frame_index,
                            "observation_role": role,
                            "frame_dir": str(frame_dir.resolve()),
                            "pose_source": (
                                "SAM_MASK_DEPTH_RGB_FIVE_DOF_GEOMETRY_"
                                "CUMULATIVE_MEDIAN"
                            ),
                            "camera_world_from_cv_row_major": (
                                camera_world.ravel().tolist()
                            ),
                            "camera_from_object_raw_row_major": (
                                camera_from_object.ravel().tolist()
                            ),
                            "estimated_world_from_object_raw_row_major": (
                                fused_task.ravel().tolist()
                            ),
                            "estimated_world_from_object_yaw_free_row_major": (
                                fused_task.ravel().tolist()
                            ),
                            "actual_world_from_hand_row_major": (
                                actual_hand_pose.ravel().tolist()
                            ),
                            "perception_timing_s": observation["timing_s"],
                            "sam6d_initialization": {
                                "score": sam_result["score"],
                                "mask": str(Path(sam_result["mask"]).resolve()),
                                "depth_filter": sam_result["depth_filter"],
                            },
                            "geometry_estimation": {
                                "metrics": geometry_result["metrics"],
                                "unfused_world_from_object_row_major": (
                                    unfused_world.ravel().tolist()
                                ),
                                "fusion_window": len(geometry_world_history),
                                "fusion_method": (
                                    "CUMULATIVE_COMPONENT_MEDIAN_IN_WORLD"
                                ),
                            },
                        }
                        return fused_task, record

                    frozen_task_frame = None
                    for frame_index, capture in enumerate(palm_frames):
                        frame_dir = (
                            output / "palm_handoff" / f"frame_{frame_index:03d}"
                        )
                        camera_world = _matrix4(
                            capture["world_from_camera_cv_row_major"],
                            "stationary palm camera pose",
                        )
                        frozen_task_frame, frame_record = observe_frozen_frame(
                            frame_index=frame_index,
                            frame_dir=frame_dir,
                            camera_world=camera_world,
                            role="INITIAL_STATIONARY_LOCALIZATION",
                            perception_name=f"frozen_initial_{frame_index:03d}",
                            actual_hand_pose=actual_fk,
                        )
                        servo_frames.append(frame_record)
                    assert frozen_task_frame is not None
                    if args.stop_after_palm_localization:
                        termination = "PALM_LOCALIZATION_ONLY_COMPLETED"
                        raise _PalmLocalizationOnlyComplete

                    frozen_position_offset = float(
                        np.linalg.norm(
                            frozen_task_frame[:3, 3]
                            - reference_object_pose[:3, 3]
                        )
                    )
                    frozen_axis_offset = math.acos(
                        np.clip(
                            float(
                                frozen_task_frame[:3, 2]
                                @ reference_object_pose[:3, 2]
                            ),
                            -1.0,
                            1.0,
                        )
                    )
                    if frozen_position_offset > float(
                        args.servo_maximum_estimated_position_offset_m
                    ):
                        raise RuntimeError(
                            "frozen local position is outside the safety envelope"
                        )
                    if frozen_axis_offset > maximum_axis_offset:
                        raise RuntimeError(
                            "frozen local axis is outside the safety envelope"
                        )

                    frozen_target_hand = frozen_task_frame @ object_from_hand
                    frozen_planned_states = None
                    if external_planner_backend is not None:
                        frozen_collision_objects = _moveit_collision_documents(
                            collision_contract=collision,
                            known_scene_geometry=known_scene_geometry,
                            world_from_object=frozen_task_frame,
                            receptacle_position_m=args.receptacle_position_m,
                        )
                        frozen_planned_states, frozen_plan_record = (
                            _run_external_pose_plan(
                                backend=external_planner_backend,
                                repository=repository,
                                ros_domain_id=args.moveit_ros_domain_id,
                                stage_name="palm_frozen_target_to_pregrasp",
                                output_dir=output,
                                start_arm=np.asarray(actual[:7], dtype=np.float64),
                                hand_positions=handoff_hand,
                                target_world_from_hand=frozen_target_hand,
                                collision_objects=frozen_collision_objects,
                                physics_dt_s=dt,
                                ik_timeout_s=args.moveit_ik_timeout_s,
                                planning_timeout_s=args.moveit_planning_timeout_s,
                            )
                        )
                        result["frozen_pregrasp_motion_plan"] = _json_ready(
                            frozen_plan_record
                        )
                        solved_arm = np.asarray(
                            frozen_planned_states[-1], dtype=np.float64
                        )
                        solved_fk = np.asarray(
                            inputs.robot_model.forward_kinematics(
                                tuple(
                                    np.concatenate((solved_arm, handoff_hand))
                                ),
                                enforce_limits=False,
                            )["handbase_link"],
                            dtype=np.float64,
                        )
                        ik_position_error = float(
                            np.linalg.norm(
                                solved_fk[:3, 3] - frozen_target_hand[:3, 3]
                            )
                        )
                        ik_rotation_error = float(
                            np.linalg.norm(
                                Rotation.from_matrix(
                                    frozen_target_hand[:3, :3].T
                                    @ solved_fk[:3, :3]
                                ).as_rotvec()
                            )
                        )
                        seed_index = "PICK_IK_GLOBAL"
                        if args.execute_grasp_lift:
                            assert physical_grasp is not None
                            control_plan = physical_grasp["control_plan"]
                            final_hand = tuple(
                                map(
                                    float,
                                    control_plan[
                                        "final_joint_positions_rad"
                                    ],
                                )
                            )
                            closing_order, closing_numbers = (
                                control.normalized_closing_order(
                                    control_plan.get(
                                        "closing_order",
                                        inputs.config.section(
                                            "closure_prediction"
                                        )["closing_order"],
                                    )
                                )
                            )
                            lift_rows, lift_errors = (
                                control._solve_lift_waypoints(
                                    inputs,
                                    inputs.robot_model,
                                    inputs.config.section("ik")["solver"],
                                    final_hand,
                                    frozen_target_hand,
                                    solved_arm,
                                )
                            )
                            direction_object = control_plan.get(
                                "approach_direction_object"
                            )
                            approach_direction_world = (
                                np.asarray((0.0, 0.0, -1.0))
                                if direction_object is None
                                else frozen_task_frame[:3, :3]
                                @ np.asarray(direction_object)
                            )
                            full_physical_motion_plan = {
                                "arm_joint_names": control.ARM_JOINT_NAMES,
                                "active_hand_joint_names": (
                                    control.ACTIVE_HAND_JOINT_NAMES
                                ),
                                "approach_arm_waypoints_rad": (solved_arm,),
                                "pregrasp_arm_positions_rad": solved_arm,
                                "pregrasp_hand_positions_rad": tuple(
                                    map(
                                        float,
                                        control_plan[
                                            "pregrasp_joint_positions_rad"
                                        ],
                                    )
                                ),
                                "final_hand_positions_rad": final_hand,
                                "closing_order": closing_order,
                                "closing_order_finger_numbers": (
                                    closing_numbers
                                ),
                                "lift_arm_waypoints_rad": tuple(lift_rows),
                                "world_from_hand_base_target": tuple(
                                    map(float, frozen_target_hand.ravel())
                                ),
                                "approach_direction_world": tuple(
                                    map(float, approach_direction_world)
                                ),
                                "approach_seed_index": "PICK_IK_GLOBAL",
                                "pregrasp_seed_index": "PICK_IK_GLOBAL",
                                "maximum_ik_position_error_m": max(
                                    [ik_position_error]
                                    + [row[0] for row in lift_errors]
                                ),
                                "maximum_ik_orientation_error_rad": max(
                                    [ik_rotation_error]
                                    + [row[1] for row in lift_errors]
                                ),
                                "maximum_lift_joint_step_rad": max(
                                    float(
                                        np.max(
                                            np.abs(
                                                np.asarray(right)
                                                - np.asarray(left)
                                            )
                                        )
                                    )
                                    for left, right in zip(
                                        lift_rows, lift_rows[1:]
                                    )
                                ),
                                "maximum_approach_joint_step_rad": 0.0,
                                "online_signals": (
                                    "joint_position",
                                    "joint_velocity",
                                    "joint_target_error",
                                    "tare_subtracted_measured_joint_effort",
                                    "robot_model_gravity_compensation_from_joint_state",
                                ),
                            }
                    elif args.execute_grasp_lift:
                        assert physical_grasp is not None
                        full_physical_motion_plan = dict(
                            control.build_joint_motion_plan(
                                repository,
                                inputs,
                                physical_grasp["control_plan"],
                                frozen_task_frame,
                                include_lift=True,
                            )
                        )
                        planned_target = _matrix4(
                            full_physical_motion_plan[
                                "world_from_hand_base_target"
                            ],
                            "physical frozen pregrasp target",
                        )
                        if not np.allclose(
                            planned_target,
                            frozen_target_hand,
                            rtol=0.0,
                            atol=1.0e-12,
                        ):
                            raise RuntimeError(
                                "physical motion plan changed the visual target"
                            )
                        solved_arm = np.asarray(
                            full_physical_motion_plan[
                                "pregrasp_arm_positions_rad"
                            ],
                            dtype=np.float64,
                        )
                        ik_position_error = float(
                            full_physical_motion_plan[
                                "maximum_ik_position_error_m"
                            ]
                        )
                        ik_rotation_error = float(
                            full_physical_motion_plan[
                                "maximum_ik_orientation_error_rad"
                            ]
                        )
                        seed_index = full_physical_motion_plan[
                            "pregrasp_seed_index"
                        ]
                    else:
                        solved, ik_position_error, ik_rotation_error, seed_index = (
                            solve_bounded_hand_base_ik(
                                inputs.config.section("ik")["solver"],
                                model=inputs.robot_model,
                                hand_positions=handoff_hand,
                                target_world_from_hand_base=frozen_target_hand,
                                seed_arm_positions=(actual[:7],),
                                label="FROZEN_GEOMETRY_PREGRASP",
                            )
                        )
                        solved_arm = np.asarray(solved, dtype=np.float64)
                    for frame_record in servo_frames:
                        frame_record["target_world_from_hand_row_major"] = (
                            frozen_target_hand.ravel().tolist()
                        )
                        frame_record["position_error_to_estimated_target_m"] = float(
                            np.linalg.norm(
                                frozen_target_hand[:3, 3] - actual_fk[:3, 3]
                            )
                        )
                        frame_record["axis_error_to_estimated_target_deg"] = (
                            math.degrees(
                                math.acos(
                                    np.clip(
                                        float(
                                            frozen_target_hand[:3, 2]
                                            @ actual_fk[:3, 2]
                                        ),
                                        -1.0,
                                        1.0,
                                    )
                                )
                            )
                        )

                    start_arm = np.asarray(actual[:7], dtype=np.float64)
                    if frozen_planned_states is not None:
                        planned_arm_states = frozen_planned_states
                        frozen_motion_steps = len(planned_arm_states) - 1
                        frozen_motion_duration_s = float(
                            result["frozen_pregrasp_motion_plan"][
                                "execution_resampling"
                            ]["duration_s"]
                        )
                    else:
                        frozen_motion_duration_s = float(
                            plan["trajectory"]["approach_duration_s"]
                        )
                        frozen_motion_steps = max(
                            1, round(frozen_motion_duration_s / dt)
                        )
                        planned_arm_states = []
                        for state_index in range(frozen_motion_steps + 1):
                            blend = control.minimum_jerk_blend(
                                state_index / frozen_motion_steps
                            )
                            planned_arm_states.append(
                                (1.0 - blend) * start_arm + blend * solved_arm
                            )

                    plug_obstacle, _ = _cylinder_from_mesh(
                        plug_mesh, 0.001, frozen_task_frame
                    )
                    frozen_obstacles = {**static_obstacles, "plug": plug_obstacle}
                    first_collision = None
                    collision_started = time.perf_counter()
                    for state_index, state in enumerate(planned_arm_states):
                        collision_at_state = _first_discrete_collision(
                            collision_scene,
                            state,
                            handoff_hand,
                            frozen_obstacles,
                        )
                        if collision_at_state is not None:
                            first_collision = {
                                "state_index": state_index,
                                **collision_at_state,
                            }
                            break
                    collision_boolean_elapsed = time.perf_counter() - collision_started

                    sampled_indices = np.linspace(
                        0,
                        frozen_motion_steps,
                        min(101, frozen_motion_steps + 1),
                    ).round().astype(int)
                    collision_minimum: dict[str, float] = {
                        "self": math.inf,
                        **{name: math.inf for name in frozen_obstacles},
                    }
                    collision_limiting_link: dict[str, str | None] = {
                        name: None for name in frozen_obstacles
                    }
                    limiting_self_pair = None
                    for state_index in sampled_indices:
                        state = planned_arm_states[int(state_index)]
                        self_clearance, self_pair = collision_scene.state_clearance(
                            state, handoff_hand
                        )
                        if self_clearance < collision_minimum["self"]:
                            collision_minimum["self"] = self_clearance
                            limiting_self_pair = self_pair
                        for obstacle_name, obstacle in frozen_obstacles.items():
                            minimum, limiting_link = min(
                                (
                                    _fcl_distance(link_object, obstacle),
                                    link_name,
                                )
                                for link_name, link_object in (
                                    collision_scene.objects.items()
                                )
                            )
                            if minimum < collision_minimum[obstacle_name]:
                                collision_minimum[obstacle_name] = minimum
                                collision_limiting_link[obstacle_name] = limiting_link

                    frozen_plan_record = {
                        "schema_version": "kcg_frozen_target_plan_v1",
                        "visual_updates_during_motion": 0,
                        "frozen_world_from_object_row_major": (
                            frozen_task_frame.ravel().tolist()
                        ),
                        "target_world_from_hand_row_major": (
                            frozen_target_hand.ravel().tolist()
                        ),
                        "target_arm_joint_positions_rad": solved_arm.tolist(),
                        "ik": {
                            "position_error_m": ik_position_error,
                            "rotation_error_rad": ik_rotation_error,
                            "seed_index": seed_index,
                        },
                        "trajectory": {
                            "interpolation": "MINIMUM_JERK_JOINT_SPACE",
                            "duration_s": frozen_motion_duration_s,
                            "rate_hz": int(round(1.0 / dt)),
                            "state_count": len(planned_arm_states),
                            "maximum_joint_step_rad": float(
                                np.max(
                                    np.abs(np.diff(np.stack(planned_arm_states), axis=0))
                                )
                            ),
                        },
                        "collision_check": {
                            "all_command_states_checked": True,
                            "command_state_count": len(planned_arm_states),
                            "first_collision": first_collision,
                            "boolean_check_elapsed_s": collision_boolean_elapsed,
                            "clearance_sample_count": int(len(sampled_indices)),
                            "sampled_minimum_clearance_m": collision_minimum,
                            "limiting_self_pair": (
                                None
                                if limiting_self_pair is None
                                else list(limiting_self_pair)
                            ),
                            "limiting_environment_link": collision_limiting_link,
                            "continuous_proof_claimed": False,
                        },
                        "final_reobservation_used_for_control": False,
                        "correction_replan_executed": False,
                    }

                    if first_collision is not None:
                        termination = "FROZEN_PLAN_COLLISION_CHECK_FAILED"
                    else:
                        world.play()
                        for motion_step in stepper.active_steps(
                            frozen_motion_steps
                        ):
                            stepper.advance(
                                "frozen_target_plan",
                                planned_arm_states[motion_step + 1],
                                handoff_hand,
                            )
                        world.pause()
                        simulation_app.update()
                        if stepper.abort_reason is not None:
                            termination = (
                                f"SAFETY_ABORT:{stepper.abort_reason}"
                            )
                        else:
                            termination = "FROZEN_PLAN_REACHED"
                            motion_count = 1

                    if termination == "FROZEN_PLAN_REACHED":
                        final_active_for_observation = (
                            robot.get_dof_positions(indices=0)
                            .numpy()[0][active_indices]
                        )
                        final_hand_for_observation = np.asarray(
                            inputs.robot_model.forward_kinematics(
                                tuple(final_active_for_observation),
                                enforce_limits=False,
                            )["handbase_link"],
                            dtype=np.float64,
                        )
                        final_camera_world = (
                            final_hand_for_observation @ palm_t_hc
                        )
                        final_camera_path = (
                            "/World/TEVisualHandoff/FrozenPlanFinalCamera"
                        )
                        _author_camera(
                            stage,
                            final_camera_path,
                            final_camera_world,
                            resolution=palm_resolution,
                            focal_length_mm=float(
                                hand_camera_document["focal_length_mm"]
                            ),
                            horizontal_aperture_mm=float(
                                hand_camera_document["horizontal_aperture_mm"]
                            ),
                            clipping_range_m=tuple(
                                hand_camera_document["clipping_range_m"]
                            ),
                            Gf=Gf,
                            UsdGeom=UsdGeom,
                        )
                        simulation_app.update()
                        final_frame_dir = output / "frozen_plan_final" / "frame_000"
                        final_capture = _capture_rgbd(
                            rep=rep,
                            resources=rgbd_resources,
                            camera_path=final_camera_path,
                            resolution=palm_resolution,
                            output_dir=final_frame_dir,
                            warmup_frames=1,
                            rt_subframes=4,
                        )
                        try:
                            final_task_frame, final_frame_record = (
                                observe_frozen_frame(
                                    frame_index=len(servo_frames),
                                    frame_dir=final_frame_dir,
                                    camera_world=final_camera_world,
                                    role=(
                                        "FINAL_STATIC_REOBSERVATION_NOT_CONTROL"
                                    ),
                                    perception_name="frozen_final_000",
                                    actual_hand_pose=final_hand_for_observation,
                                )
                            )
                            final_target_from_reobservation = (
                                final_task_frame @ object_from_hand
                            )
                            final_position_error = float(
                                np.linalg.norm(
                                    final_target_from_reobservation[:3, 3]
                                    - final_hand_for_observation[:3, 3]
                                )
                            )
                            final_axis_error = math.acos(
                                np.clip(
                                    float(
                                        final_target_from_reobservation[:3, 2]
                                        @ final_hand_for_observation[:3, 2]
                                    ),
                                    -1.0,
                                    1.0,
                                )
                            )
                            final_frame_record[
                                "target_world_from_hand_row_major"
                            ] = final_target_from_reobservation.ravel().tolist()
                            final_frame_record[
                                "position_error_to_estimated_target_m"
                            ] = final_position_error
                            final_frame_record[
                                "axis_error_to_estimated_target_deg"
                            ] = math.degrees(final_axis_error)
                            servo_frames.append(final_frame_record)
                            frozen_plan_record["final_reobservation"] = {
                                "capture": final_capture,
                                "position_error_to_reobserved_target_m": (
                                    final_position_error
                                ),
                                "axis_error_to_reobserved_target_deg": (
                                    math.degrees(final_axis_error)
                                ),
                                "within_original_servo_tolerance": bool(
                                    final_position_error
                                    <= float(args.servo_position_tolerance_m)
                                    and final_axis_error <= axis_tolerance
                                ),
                                "used_for_control": False,
                            }
                        except Exception as final_error:
                            termination = (
                                "FROZEN_PLAN_REACHED_FINAL_REOBSERVATION_FAILED"
                            )
                            frozen_plan_record["final_reobservation"] = {
                                "capture": final_capture,
                                "error_type": type(final_error).__name__,
                                "error": str(final_error),
                                "used_for_control": False,
                            }

                    if args.execute_grasp_lift and termination in (
                        "FROZEN_PLAN_REACHED",
                        "FROZEN_PLAN_REACHED_FINAL_REOBSERVATION_FAILED",
                    ):
                        final_reobservation_warning = (
                            termination
                            == "FROZEN_PLAN_REACHED_FINAL_REOBSERVATION_FAILED"
                        )
                        assert physical_grasp is not None
                        assert full_physical_motion_plan is not None
                        assert tensor_contact_prim is not None
                        assert split_manifest is not None
                        precontact_active = (
                            robot.get_dof_positions(indices=0)
                            .numpy()[0][active_indices]
                        )
                        physical_precontact_hand_pose = np.asarray(
                            inputs.robot_model.forward_kinematics(
                                tuple(precontact_active),
                                enforce_limits=False,
                            )["handbase_link"],
                            dtype=np.float64,
                        )
                        physical_precontact_truth_world_from_object = (
                            _world_from_prim(
                                stage,
                                str(scene["roots"]["object"]),
                                Usd=Usd,
                                UsdGeom=UsdGeom,
                            )
                        )
                        physical_engine_monitor = runner.PhysxStatsMonitor(
                            context
                        )
                        truth_auditor = runner.TruthAuditRecorder(
                            object_parts=object_parts,
                            hand_base_prim=hand_base_prim,
                            robot_model=inputs.robot_model,
                            stage_modules=(Gf, Usd, UsdGeom),
                            contact_interface=get_physx_simulation_interface(),
                            path_decoder=PhysicsSchemaTools.intToSdfPath,
                            roots={"robot": runner.ROBOT_ROOT, **scene["roots"]},
                            expected_total_mass_kg=float(
                                split_manifest["mass_model"]["total_mass_kg"]
                            ),
                            part_bottom_offsets_m=scene[
                                "part_bottom_offsets_m"
                            ],
                            table_top_z_m=float(scene["table_top_z_m"]),
                            physics_dt_s=dt,
                            engine_monitor=physical_engine_monitor,
                            physics_step_interface=get_physx_interface(),
                            tensor_contact_prim=tensor_contact_prim,
                            tensor_contact_sensor_paths=(
                                tensor_contact_sensor_paths
                            ),
                            tensor_contact_max_count=(
                                runner.TENSOR_CONTACT_MAX_COUNT
                            ),
                        )
                        physical_auditor = _PhysicalVisualAuditor(
                            ft_auditor, truth_auditor
                        )
                        stepper.auditor = physical_auditor
                        physical_pregrasp = {
                            "arm": np.asarray(
                                full_physical_motion_plan[
                                    "pregrasp_arm_positions_rad"
                                ],
                                dtype=np.float64,
                            ),
                            "hand": np.asarray(
                                full_physical_motion_plan[
                                    "pregrasp_hand_positions_rad"
                                ],
                                dtype=np.float64,
                            ),
                            "above_settled": True,
                            "above_final_error_rad": float(
                                np.max(
                                    np.abs(
                                        robot.get_dof_positions(indices=0)
                                        .numpy()[0][active_indices][:7]
                                        - np.asarray(
                                            full_physical_motion_plan[
                                                "pregrasp_arm_positions_rad"
                                            ],
                                            dtype=np.float64,
                                        )
                                    )
                                )
                            ),
                            "visual_target_frozen_before_contact": True,
                        }
                        world.play()
                        stepper.advance(
                            "settle",
                            physical_pregrasp["arm"],
                            physical_pregrasp["hand"],
                        )
                        physical_grasp_result = control.run_grasp_lift_sequence(
                            stepper,
                            full_physical_motion_plan,
                            dynamic,
                            physical_pregrasp,
                        )
                        visual_consumption = {
                            "provided": True,
                            "motion_plan_pose_source": (
                                "FIVE_STATIONARY_PALM_RGBD_FRAMES_"
                                "SAM_MASK_DEPTH_GEOMETRY_MEDIAN"
                            ),
                            "planned_world_from_hand_base_target_row_major": (
                                frozen_target_hand.ravel().tolist()
                            ),
                            "assembly_key_pose_consumed": False,
                            "scene_contract_pose_used_for_motion_plan": False,
                            "online_object_or_semantic_truth_used": False,
                            "controller_execution_observed": True,
                        }
                        physical_arguments = SimpleNamespace(
                            object_id=args.object_id,
                            mode="grasp-lift",
                            initialize_at_pregrasp=False,
                            visual_transport_target_consumption=(
                                visual_consumption
                            ),
                            robustness_scenario_name="friction_lower_0p45",
                            robustness_perturbation={
                                "contact_friction_coefficient": 0.45
                            },
                            hand_tilt_about_object_pivot_audit={
                                **physical_hand_tilt_audit
                            },
                            postgrasp_disturbance=None,
                            postgrasp_disturbance_panel=(
                                None
                                if args.postgrasp_disturbance_panel is None
                                else str(args.postgrasp_disturbance_panel.resolve())
                            ),
                            postgrasp_disturbance_condition=(
                                args.postgrasp_disturbance_condition
                            ),
                            nominal_grasp_qualification_evaluation=(
                                None
                                if args.nominal_grasp_qualification_evaluation
                                is None
                                else str(
                                    args.nominal_grasp_qualification_evaluation.resolve()
                                )
                            ),
                            free_split_object_manifest=(
                                str(args.free_split_object_manifest.resolve())
                            ),
                            runtime_resources_path=args.runtime_resources.resolve(),
                            runtime_resources_document=json.loads(
                                args.runtime_resources.read_text(encoding="utf-8")
                            ),
                            robot_asset_path=robot_asset,
                            preflight_evaluation_path=(
                                None
                                if args.physical_preflight_evaluation is None
                                else args.physical_preflight_evaluation.resolve()
                            ),
                            preflight_document=(
                                {}
                                if args.physical_preflight_evaluation is None
                                else json.loads(
                                    args.physical_preflight_evaluation.read_text(
                                        encoding="utf-8"
                                    )
                                )
                            ),
                            configured_preload_increment_rad=float(
                                dynamic["preload_increment_rad"]
                            ),
                            effective_preload_increment_rad=float(
                                dynamic["preload_increment_rad"]
                            ),
                            configured_lift_arm_damping_nm_s_rad=float(
                                dynamic["lift_arm_damping_nm_s_rad"]
                            ),
                            effective_lift_arm_damping_nm_s_rad=float(
                                dynamic["lift_arm_damping_nm_s_rad"]
                            ),
                            configured_finger_preload_scales=[1.0, 1.0, 1.0],
                            effective_finger_preload_scales=[1.0, 1.0, 1.0],
                            required_closing_joint_effort_nm=list(
                                dynamic["required_closing_joint_effort_nm"]
                            ),
                            dynamic_settings=dynamic,
                            configured_closing_order=list(
                                relation_hand["closing_order"]
                            ),
                            effective_closing_order=list(
                                relation_hand["closing_order"]
                            ),
                            configured_contact_coordination_mode=(
                                configured_contact_coordination_mode
                            ),
                            effective_contact_coordination_mode=(
                                "parallel_contact_latch"
                            ),
                            configured_palm_joint_position_rad=float(
                                relation_hand[
                                    "pregrasp_joint_positions_rad"
                                ][0]
                            ),
                            effective_palm_joint_position_rad=float(
                                relation_hand[
                                    "pregrasp_joint_positions_rad"
                                ][0]
                            ),
                            configured_approach_high_seed_arm_positions_rad=list(
                                relation_hand[
                                    "approach_high_seed_arm_positions_rad"
                                ]
                            ),
                            effective_approach_high_seed_arm_positions_rad=list(
                                relation_hand[
                                    "approach_high_seed_arm_positions_rad"
                                ]
                            ),
                            finger_joint_target_audit={
                                "applied": False,
                                "joint_name": "f3j2",
                                "requested_offset_rad": 0.0,
                                "before_pregrasp_rad": float(
                                    relation_hand[
                                        "pregrasp_joint_positions_rad"
                                    ][3]
                                ),
                                "observed_pregrasp_target_rad": float(
                                    relation_hand[
                                        "pregrasp_joint_positions_rad"
                                    ][3]
                                ),
                                "before_final_rad": float(
                                    relation_hand[
                                        "final_joint_positions_rad"
                                    ][3]
                                ),
                                "observed_final_target_rad": float(
                                    relation_hand[
                                        "final_joint_positions_rad"
                                    ][3]
                                ),
                            },
                            visual_transport_target_binding=None,
                        )
                        if args.postgrasp_disturbance_panel is not None:
                            runner._load_postgrasp_disturbance(
                                repository,
                                physical_arguments,
                                inputs,
                                physical_grasp,
                                scene_entry,
                            )
                            disturbance_execution = (
                                runner._run_postgrasp_disturbance(
                                    {
                                        "object_parts": object_parts,
                                        "scene": scene,
                                        "auditor": physical_auditor,
                                    },
                                    physical_arguments,
                                    stepper,
                                    physical_grasp_result,
                                )
                            )
                        else:
                            disturbance_execution = {
                                "requested": False,
                                "started": False,
                                "completed": False,
                                "failure_reason": None,
                            }
                        physical_outcome = control.controller_outcome(
                            stepper,
                            mode="grasp-lift",
                            native_drive_audit=drive_audit,
                            pregrasp=physical_pregrasp,
                            grasp=physical_grasp_result,
                        )
                        physical_trace = runner._initial_trace(
                            physical_arguments,
                            inputs,
                            physical_grasp,
                            full_physical_motion_plan,
                            dynamic,
                        )
                        physical_trace.update(
                            {
                                "controller_outcome": physical_outcome,
                                "postgrasp_disturbance_execution": (
                                    disturbance_execution
                                ),
                                "samples": truth_auditor.samples,
                                "contact_report_api_audit": {
                                    "complete": contact_report_complete
                                },
                                "tensor_contact_view_audit": {
                                    "robot_sensor_paths": list(
                                        robot_contact_paths
                                    ),
                                    "object_sensor_paths": list(
                                        object_contact_paths
                                    ),
                                    "contact_filter_paths": list(
                                        object_contact_paths
                                    ),
                                    "sensor_paths": list(
                                        tensor_contact_sensor_paths
                                    ),
                                    "max_contact_count": (
                                        runner.TENSOR_CONTACT_MAX_COUNT
                                    ),
                                    "valid_after_reset": True,
                                },
                                "audit_roots": {
                                    "robot": runner.ROBOT_ROOT,
                                    **scene["roots"],
                                },
                                "accepted_preflight_bound": False,
                                "offline_task_gate_passed": False,
                                "online_object_or_contact_truth_used": False,
                                "truth_audit_data_returned_to_controller": (
                                    False
                                ),
                                "object_pose_writes_after_start": 0,
                            }
                        )
                        current_evidence_binding = runner._evidence_binding(
                            repository,
                            physical_arguments,
                            inputs,
                            physical_grasp,
                            scene,
                            robot_asset,
                        )
                        physical_trace["evidence_binding"] = (
                            current_evidence_binding
                        )
                        if args.physical_preflight_evaluation is not None:
                            preflight_binding = physical_arguments.preflight_document.get(
                                "evidence_binding"
                            )
                            if not isinstance(preflight_binding, dict):
                                raise ValueError(
                                    "physical preflight evidence binding is absent"
                                )
                            clamp_only_binding_keys = {
                                "registered_grasp_sha256",
                                "required_closing_joint_effort_nm",
                                "predicted_unit_task_closing_joint_effort_nm",
                            }
                            mismatched_binding_keys = sorted(
                                key
                                for key in (
                                    set(preflight_binding)
                                    | set(current_evidence_binding)
                                )
                                - clamp_only_binding_keys
                                if preflight_binding.get(key)
                                != current_evidence_binding.get(key)
                            )
                            if mismatched_binding_keys:
                                raise ValueError(
                                    "physical preflight differs from the current "
                                    "visual grasp outside clamp-only fields: "
                                    f"{mismatched_binding_keys}"
                                )
                            physical_trace[
                                "preflight_clamp_effort_reuse_audit"
                            ] = {
                                "preflight_required_closing_joint_effort_nm": (
                                    preflight_binding.get(
                                        "required_closing_joint_effort_nm"
                                    )
                                ),
                                "current_required_closing_joint_effort_nm": (
                                    current_evidence_binding.get(
                                        "required_closing_joint_effort_nm"
                                    )
                                ),
                                "ignored_binding_keys": sorted(
                                    clamp_only_binding_keys
                                ),
                                "all_other_binding_keys_match": True,
                            }
                            physical_trace["accepted_preflight_bound"] = True
                            physical_trace[
                                "accepted_preflight_evaluation_sha256"
                            ] = runner.file_sha256(
                                args.physical_preflight_evaluation.resolve()
                            )
                        controller_path = (
                            output / "physical_controller_outcome.json"
                        )
                        controller_path.write_text(
                            json.dumps(
                                _json_ready(physical_outcome),
                                ensure_ascii=False,
                                indent=2,
                            )
                            + "\n",
                            encoding="utf-8",
                        )
                        raw_trace_path = output / "physical_truth_samples.json.gz"
                        with gzip.open(
                            raw_trace_path, "wt", encoding="utf-8"
                        ) as raw_stream:
                            json.dump(
                                _json_ready(truth_auditor.samples),
                                raw_stream,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            )
                            raw_stream.write("\n")
                        result["physical_raw_evidence"] = {
                            "controller_outcome": str(controller_path),
                            "truth_samples_gzip": str(raw_trace_path),
                            "sample_count": len(truth_auditor.samples),
                            "written_before_evaluation": True,
                        }
                        if physical_engine_monitor is None:
                            raise RuntimeError(
                                "physical engine monitor was not initialized"
                            )
                        physical_runtime_record = {
                            "scene": scene,
                            "robot_asset": robot_asset,
                            "registered_grasp": physical_grasp,
                            "control_plan": physical_grasp["control_plan"],
                            "runtime_resources_path": (
                                args.runtime_resources.resolve()
                            ),
                            "capacity_audit_sha256": (
                                physical_arguments.runtime_resources_document[
                                    "capacity_audit_sha256"
                                ]
                            ),
                            "postgrasp_disturbance": (
                                physical_arguments.postgrasp_disturbance
                            ),
                        }
                        physical_trace["runtime"] = runner._runtime_record(
                            repository, inputs, physical_runtime_record
                        )
                        physical_trace["identity_hash_check_pass"] = (
                            runner.identity_hashes_match(physical_trace)
                        )
                        physical_evaluation = runner.evaluate_trace(
                            physical_trace,
                            robot_asset_path=robot_asset,
                            inputs=inputs,
                        )
                        split_relative_motion = (
                            runner._split_plug_relative_motion_summary(
                                {
                                    "scene": scene,
                                    "auditor": physical_auditor,
                                }
                            )
                        )
                        physical_evaluation["split_plug_relative_motion"] = (
                            split_relative_motion
                        )
                        split_contact_policy = (
                            runner._split_plug_contact_policy_summary(
                                {"scene": scene}, physical_evaluation
                            )
                        )
                        physical_evaluation[
                            "split_plug_grasp_contact_policy"
                        ] = split_contact_policy
                        runner._apply_split_plug_contact_policy(
                            physical_evaluation, split_contact_policy
                        )
                        engine_runtime = physical_engine_monitor.summary()
                        engine_runtime["gpu_backend_pass"] = bool(
                            result["physics_backend"]["pass"]
                        )
                        engine_runtime["engine_log_sync"] = (
                            runner.synchronize_engine_log(engine_log_path)
                        )
                        physical_evaluation = (
                            runner.finalize_engine_evaluation(
                                physical_evaluation,
                                engine_runtime,
                                engine_log_path,
                            )
                        )
                        evaluation_path = output / "physical_grasp_evaluation.json"
                        evaluation_path.write_text(
                            json.dumps(
                                _json_ready(physical_evaluation),
                                ensure_ascii=False,
                                indent=2,
                            )
                            + "\n",
                            encoding="utf-8",
                        )
                        physical_success = bool(
                            physical_evaluation.get(
                                "nominal_diagnostic_pass"
                            )
                            is True
                            and split_contact_policy.get(
                                "all_three_fingers_contacted_coupling_nut_only"
                            )
                            is True
                        )
                        result["physical_grasp"] = {
                            "evaluation": str(evaluation_path),
                            "success": physical_success,
                            "controller_failure_reason": (
                                physical_evaluation.get(
                                    "controller_failure_reason"
                                )
                            ),
                            "maximum_lift_m": physical_evaluation.get(
                                "maximum_lift_m"
                            ),
                            "hold_duration_s": physical_evaluation.get(
                                "hold_duration_s"
                            ),
                            "three_terminal_link_contacts_observed": (
                                physical_evaluation.get(
                                    "three_terminal_link_contacts_observed"
                                )
                            ),
                            "nut_only": split_contact_policy.get(
                                "all_three_fingers_contacted_coupling_nut_only"
                            ),
                            "hand_object_full_relative_pose": (
                                physical_evaluation.get(
                                    "hand_object_full_relative_pose"
                                )
                            ),
                            "hand_grasp_part_relative_pose": (
                                physical_evaluation.get(
                                    "hand_grasp_part_relative_pose"
                                )
                            ),
                            "split_plug_relative_motion": split_relative_motion,
                            "postgrasp_disturbance": physical_evaluation.get(
                                "postgrasp_disturbance"
                            ),
                            "final_reobservation_warning": (
                                final_reobservation_warning
                            ),
                            "truth_used_for_control": False,
                        }
                        termination = (
                            "FROZEN_PLAN_REACHED_GRASP_LIFT_COMPLETED"
                            if physical_success
                            else "FROZEN_PLAN_REACHED_GRASP_LIFT_FAILED"
                        )

                iteration_values = (
                    ()
                    if args.servo_control_policy == "frozen_plan"
                    else range(int(args.servo_maximum_iterations) + 2)
                )
                for iteration in iteration_values:
                    actual = robot.get_dof_positions(indices=0).numpy()[0][active_indices]
                    actual_fk = np.asarray(
                        inputs.robot_model.forward_kinematics(
                            tuple(actual), enforce_limits=False
                        )["handbase_link"],
                        dtype=np.float64,
                    )
                    palm_world_from_camera = actual_fk @ palm_t_hc
                    if iteration == 0:
                        frame_dir = output / "palm_handoff" / "frame_000"
                        capture_metrics = dict(palm_frames[0])
                    else:
                        _author_camera(
                            stage,
                            servo_capture_path,
                            palm_world_from_camera,
                            resolution=palm_resolution,
                            focal_length_mm=float(hand_camera_document["focal_length_mm"]),
                            horizontal_aperture_mm=float(
                                hand_camera_document["horizontal_aperture_mm"]
                            ),
                            clipping_range_m=tuple(
                                hand_camera_document["clipping_range_m"]
                            ),
                            Gf=Gf,
                            UsdGeom=UsdGeom,
                        )
                        simulation_app.update()
                        frame_dir = (
                            output / "visual_servo_frames" / f"frame_{iteration:03d}"
                        )
                        capture_metrics = _capture_rgbd(
                            rep=rep,
                            resources=rgbd_resources,
                            camera_path=servo_capture_path,
                            resolution=palm_resolution,
                            output_dir=frame_dir,
                            warmup_frames=1,
                            rt_subframes=4,
                        )
                    if (
                        int(capture_metrics["finite_positive_depth_pixels"]) < 100
                        or float(capture_metrics["rgb_standard_deviation"]) < 1.0
                    ):
                        termination = "RGBD_CAPTURE_INVALID"
                        break

                    sam_result: dict[str, object] | None = None
                    geometry_result: dict[str, object] | None = None
                    if args.servo_estimator in ("sam6d", "geometry") or iteration == 0:
                        sam_result = _run_sam6d_frame(
                            repository=repository,
                            sam6d_root=args.sam6d_root.resolve(),
                            sam6d_python=args.sam6d_python,
                            templates=args.sam6d_templates.resolve(),
                            cad_mm=plug_mesh,
                            rgb=frame_dir / "rgb.png",
                            depth_m=frame_dir / "depth_m.npy",
                            depth_mm=frame_dir / "depth_mm.png",
                            camera_json=palm_camera_json,
                            output_dir=output
                            / "perception"
                            / f"frame_{iteration:03d}_sam6d",
                            run_pem=args.servo_estimator != "geometry",
                        )

                    if args.servo_estimator == "sam6d":
                        assert sam_result is not None
                        camera_from_object = np.asarray(
                            sam_result["camera_from_object"], dtype=np.float64
                        )
                        perception_timing = dict(sam_result["timing_s"])
                        pose_source = "SAM6D_ISM_DEPTH_FILTER_PEM"
                    elif args.servo_estimator == "geometry":
                        assert sam_result is not None
                        geometry_result = _estimate_five_dof_from_float_depth(
                            rgb_path=frame_dir / "rgb.png",
                            depth_m_path=frame_dir / "depth_m.npy",
                            mask_path=Path(sam_result["mask"]),
                            intrinsics=camera_matrix,
                            mesh_path=plug_mesh,
                        )
                        camera_from_object = np.asarray(
                            geometry_result["camera_from_object"], dtype=np.float64
                        )
                        geometry_elapsed = float(
                            dict(geometry_result["metrics"])["elapsed_s"]
                        )
                        perception_timing = {
                            **dict(sam_result["timing_s"]),
                            "geometry": geometry_elapsed,
                            "total": float(sam_result["timing_s"]["total"])
                            + geometry_elapsed,
                        }
                        pose_source = "SAM_MASK_DEPTH_RGB_FIVE_DOF_GEOMETRY"
                    else:
                        if worker_process is None:
                            assert args.foundationpose_root is not None
                            assert args.foundationpose_python is not None
                            worker_process, worker_log_lines = (
                                _start_foundationpose_worker(
                                    worker_script=Path(__file__).resolve().with_name(
                                        "te_foundationpose_tracker_worker.py"
                                    ),
                                    foundationpose_python=args.foundationpose_python,
                                    foundationpose_root=args.foundationpose_root.resolve(),
                                    mesh_mm=plug_mesh,
                                    output_dir=output,
                                )
                            )
                        request: dict[str, object] = {
                            "command": (
                                "initialize_from_pose" if iteration == 0 else "track"
                            ),
                            "rgb": str((frame_dir / "rgb.png").resolve()),
                            "depth_npy": str((frame_dir / "depth_m.npy").resolve()),
                            "camera_matrix": camera_matrix.ravel().tolist(),
                        }
                        if iteration == 0:
                            assert sam_result is not None
                            request["camera_from_object_row_major"] = np.asarray(
                                sam_result["camera_from_object"], dtype=np.float64
                            ).ravel().tolist()
                        fp_response = _foundationpose_request(
                            worker_process, worker_log_lines, request
                        )
                        camera_from_object = _matrix4(
                            fp_response["camera_from_object_row_major"],
                            "FoundationPose camera pose",
                        )
                        perception_timing = {
                            "foundationpose": float(fp_response["elapsed_s"]),
                            "sam6d_initialization": (
                                None
                                if sam_result is None
                                else float(sam_result["timing_s"]["total"])
                            ),
                            "total": float(fp_response["elapsed_s"])
                            + (
                                0.0
                                if sam_result is None
                                else float(sam_result["timing_s"]["total"])
                            ),
                        }
                        pose_source = (
                            "SAM6D_POSE_FOUNDATIONPOSE_TRACK_INITIALIZATION"
                            if iteration == 0
                            else "FOUNDATIONPOSE_TRACK_ONE"
                        )

                    estimated_world_from_object_raw = (
                        palm_world_from_camera @ camera_from_object
                    )
                    geometry_unfused_world: np.ndarray | None = None
                    geometry_fusion_window = 0
                    if args.servo_estimator == "geometry":
                        geometry_unfused_world = estimated_world_from_object_raw.copy()
                        raw_axis = np.asarray(
                            geometry_unfused_world[:3, 2], dtype=np.float64
                        )
                        raw_axis /= np.linalg.norm(raw_axis)
                        if float(raw_axis @ reference_axis) < 0.0:
                            raw_axis = -raw_axis
                        geometry_world_history.append(
                            (geometry_unfused_world[:3, 3].copy(), raw_axis)
                        )
                        # The plug is static throughout the no-contact pregrasp
                        # phase.  Keep the well-observed oblique handoff views in
                        # the estimate instead of allowing several nearly frontal,
                        # depth-quantized frames to rotate the axis.
                        recent = geometry_world_history
                        geometry_fusion_window = len(recent)
                        estimated_world_from_object_raw[:3, 3] = np.median(
                            np.stack([item[0] for item in recent]), axis=0
                        )
                        fused_axis = np.median(
                            np.stack([item[1] for item in recent]), axis=0
                        )
                        fused_axis /= np.linalg.norm(fused_axis)
                        estimated_world_from_object_raw[:3, 2] = fused_axis
                    estimated_axis = np.asarray(
                        estimated_world_from_object_raw[:3, 2], dtype=np.float64
                    )
                    estimated_axis /= np.linalg.norm(estimated_axis)
                    axis_sign_flipped = bool(float(estimated_axis @ reference_axis) < 0.0)
                    if axis_sign_flipped:
                        estimated_axis = -estimated_axis
                    estimated_task_frame = _yaw_free_object_frame(
                        estimated_world_from_object_raw[:3, 3], estimated_axis
                    )
                    target_hand_pose = estimated_task_frame @ object_from_hand
                    estimated_position_offset = float(
                        np.linalg.norm(
                            estimated_task_frame[:3, 3]
                            - reference_object_pose[:3, 3]
                        )
                    )
                    estimated_axis_offset = float(
                        math.acos(
                            np.clip(
                                float(estimated_axis @ reference_axis), -1.0, 1.0
                            )
                        )
                    )
                    position_error = float(
                        np.linalg.norm(target_hand_pose[:3, 3] - actual_fk[:3, 3])
                    )
                    axis_error = float(
                        math.acos(
                            np.clip(
                                float(
                                    target_hand_pose[:3, 2] @ actual_fk[:3, 2]
                                ),
                                -1.0,
                                1.0,
                            )
                        )
                    )
                    full_rotation_error = _rotation_angle(
                        actual_fk[:3, :3], target_hand_pose[:3, :3]
                    )
                    frame_record: dict[str, object] = {
                        "iteration": iteration,
                        "frame_dir": str(frame_dir.resolve()),
                        "pose_source": pose_source,
                        "camera_world_from_cv_row_major": (
                            palm_world_from_camera.ravel().tolist()
                        ),
                        "camera_from_object_raw_row_major": (
                            camera_from_object.ravel().tolist()
                        ),
                        "estimated_world_from_object_raw_row_major": (
                            estimated_world_from_object_raw.ravel().tolist()
                        ),
                        "estimated_world_from_object_yaw_free_row_major": (
                            estimated_task_frame.ravel().tolist()
                        ),
                        "actual_world_from_hand_row_major": actual_fk.ravel().tolist(),
                        "target_world_from_hand_row_major": target_hand_pose.ravel().tolist(),
                        "position_error_to_estimated_target_m": position_error,
                        "axis_error_to_estimated_target_deg": math.degrees(axis_error),
                        "full_rotation_error_to_estimated_target_deg": math.degrees(
                            full_rotation_error
                        ),
                        "estimated_position_offset_from_global_m": estimated_position_offset,
                        "estimated_axis_offset_from_global_deg": math.degrees(
                            estimated_axis_offset
                        ),
                        "axis_sign_flipped_to_match_global": axis_sign_flipped,
                        "perception_timing_s": perception_timing,
                        "sam6d_initialization": (
                            None
                            if sam_result is None
                            else {
                                "score": sam_result["score"],
                                "mask": str(Path(sam_result["mask"]).resolve()),
                                "depth_filter": sam_result["depth_filter"],
                            }
                        ),
                        "geometry_estimation": (
                            None
                            if geometry_result is None
                            else {
                                "metrics": geometry_result["metrics"],
                                "unfused_world_from_object_row_major": (
                                    geometry_unfused_world.ravel().tolist()
                                    if geometry_unfused_world is not None
                                    else None
                                ),
                                "fusion_window": geometry_fusion_window,
                                "fusion_method": (
                                    "CUMULATIVE_COMPONENT_MEDIAN_IN_WORLD"
                                ),
                            }
                        ),
                    }
                    servo_frames.append(frame_record)

                    if (
                        estimated_position_offset
                        > float(args.servo_maximum_estimated_position_offset_m)
                    ):
                        termination = "ESTIMATED_POSITION_OUTSIDE_SAFETY_ENVELOPE"
                        break
                    if estimated_axis_offset > maximum_axis_offset:
                        termination = "ESTIMATED_AXIS_OUTSIDE_SAFETY_ENVELOPE"
                        break

                    within_target = bool(
                        position_error <= float(args.servo_position_tolerance_m)
                        and axis_error <= axis_tolerance
                    )
                    consecutive = consecutive + 1 if within_target else 0
                    frame_record["within_online_target"] = within_target
                    frame_record["online_target_consecutive_count"] = consecutive
                    if consecutive >= int(args.servo_consecutive_required):
                        termination = "ONLINE_TARGET_REACHED"
                        break
                    if iteration >= int(args.servo_maximum_iterations):
                        termination = "MAXIMUM_ITERATIONS_REACHED"
                        break

                    if within_target:
                        next_hand_pose = actual_fk.copy()
                        solved_arm = np.asarray(actual[:7], dtype=np.float64)
                        frame_record["motion_skipped_for_confirmation"] = True
                    else:
                        step_target = target_hand_pose
                        control_phase = "SIMULTANEOUS_POSITION_AND_AXIS"
                        if (
                            args.servo_control_policy == "axis_then_position"
                            and axis_error > axis_tolerance
                        ):
                            step_target = actual_fk.copy()
                            step_target[:3, :3] = target_hand_pose[:3, :3]
                            control_phase = "AXIS_ALIGNMENT_AT_FIXED_HAND_POSITION"
                        elif args.servo_control_policy == "axis_then_position":
                            control_phase = "POSITION_APPROACH_AFTER_AXIS_ALIGNMENT"
                        if (
                            args.servo_control_policy
                            == "camera_orbit_then_position"
                            and axis_error > axis_tolerance
                        ):
                            target_camera_pose = target_hand_pose @ palm_t_hc
                            camera_rotation_target = palm_world_from_camera.copy()
                            camera_rotation_target[:3, :3] = (
                                target_camera_pose[:3, :3]
                            )
                            orbit_camera_pose = _bounded_pose_step(
                                palm_world_from_camera,
                                camera_rotation_target,
                                maximum_translation_m=float(
                                    args.servo_maximum_translation_step_m
                                ),
                                maximum_rotation_rad=maximum_rotation_step,
                            )
                            plug_center_world = (
                                estimated_task_frame
                                @ np.asarray(
                                    (0.0, 0.0, plug_center_local_z_m, 1.0),
                                    dtype=np.float64,
                                )
                            )[:3]
                            viewing_distance = float(
                                np.linalg.norm(
                                    plug_center_world
                                    - palm_world_from_camera[:3, 3]
                                )
                            )
                            orbit_camera_pose[:3, 3] = (
                                plug_center_world
                                - orbit_camera_pose[:3, 2] * viewing_distance
                            )
                            unbounded_orbit_hand = (
                                orbit_camera_pose @ camera_from_hand
                            )
                            next_hand_pose = _bounded_pose_step(
                                actual_fk,
                                unbounded_orbit_hand,
                                maximum_translation_m=float(
                                    args.servo_maximum_translation_step_m
                                ),
                                maximum_rotation_rad=maximum_rotation_step,
                            )
                            control_phase = "CAMERA_ORBIT_AXIS_ALIGNMENT"
                            frame_record["orbit_viewing_distance_m"] = (
                                viewing_distance
                            )
                        else:
                            if (
                                args.servo_control_policy
                                == "camera_orbit_then_position"
                            ):
                                control_phase = (
                                    "POSITION_APPROACH_AFTER_CAMERA_ORBIT"
                                )
                            next_hand_pose = _bounded_pose_step(
                                actual_fk,
                                step_target,
                                maximum_translation_m=float(
                                    args.servo_maximum_translation_step_m
                                ),
                                maximum_rotation_rad=maximum_rotation_step,
                            )
                        frame_record["control_phase"] = control_phase
                        solved, ik_position_error, ik_rotation_error, seed_index = (
                            solve_bounded_hand_base_ik(
                                inputs.config.section("ik")["solver"],
                                model=inputs.robot_model,
                                hand_positions=handoff_hand,
                                target_world_from_hand_base=next_hand_pose,
                                seed_arm_positions=(actual[:7],),
                                label=f"PALM_VISUAL_SERVO_{iteration:03d}",
                            )
                        )
                        solved_arm = np.asarray(solved, dtype=np.float64)
                        frame_record["ik"] = {
                            "position_error_m": ik_position_error,
                            "rotation_error_rad": ik_rotation_error,
                            "seed_index": seed_index,
                        }

                    plug_obstacle, _ = _cylinder_from_mesh(
                        plug_mesh, 0.001, estimated_task_frame
                    )
                    obstacles = {**static_obstacles, "plug": plug_obstacle}
                    collision_minimum = {
                        "self": math.inf,
                        **{name: math.inf for name in obstacles},
                    }
                    collision_limiting_link: dict[str, str | None] = {
                        name: None for name in obstacles
                    }
                    collision_detected: dict[str, object] | None = None
                    for sample_index in range(int(args.servo_collision_samples)):
                        fraction = (
                            sample_index / (int(args.servo_collision_samples) - 1)
                            if int(args.servo_collision_samples) > 1
                            else 1.0
                        )
                        sample_arm = (1.0 - fraction) * actual[:7] + fraction * solved_arm
                        self_clearance, self_pair = collision_scene.state_clearance(
                            sample_arm, handoff_hand
                        )
                        collision_minimum["self"] = min(
                            collision_minimum["self"], self_clearance
                        )
                        if self_clearance == 0.0 and collision_detected is None:
                            collision_detected = {
                                "kind": "self",
                                "sample_index": sample_index,
                                "pair": None if self_pair is None else list(self_pair),
                            }
                        for obstacle_name, obstacle in obstacles.items():
                            minimum, limiting_link = min(
                                (
                                    _fcl_distance(link_object, obstacle),
                                    link_name,
                                )
                                for link_name, link_object in (
                                    collision_scene.objects.items()
                                )
                            )
                            if minimum < collision_minimum[obstacle_name]:
                                collision_minimum[obstacle_name] = minimum
                                collision_limiting_link[obstacle_name] = limiting_link
                            if minimum == 0.0 and collision_detected is None:
                                collision_detected = {
                                    "kind": "environment",
                                    "sample_index": sample_index,
                                    "obstacle": obstacle_name,
                                    "link": limiting_link,
                                }
                    frame_record["collision_check"] = {
                        "sample_count": int(args.servo_collision_samples),
                        "minimum_clearance_m": collision_minimum,
                        "limiting_environment_link": collision_limiting_link,
                        "first_collision": collision_detected,
                        "continuous_proof_claimed": False,
                    }
                    if collision_detected is not None:
                        termination = "PREDICTED_COLLISION"
                        break

                    world.play()
                    motion_steps = max(
                        1, round(float(args.servo_step_duration_s) / dt)
                    )
                    start_arm = np.asarray(actual[:7], dtype=np.float64)
                    for motion_step in stepper.active_steps(motion_steps):
                        blend = control.minimum_jerk_blend(
                            (motion_step + 1) / motion_steps
                        )
                        command_arm = (1.0 - blend) * start_arm + blend * solved_arm
                        stepper.advance("visual_servo", command_arm, handoff_hand)
                    if stepper.abort_reason is not None:
                        termination = f"SAFETY_ABORT:{stepper.abort_reason}"
                        world.pause()
                        break
                    hold_steps = max(1, round(float(args.palm_capture_hold_s) / dt))
                    for _ in stepper.active_steps(hold_steps):
                        stepper.advance("visual_servo", solved_arm, handoff_hand)
                    world.pause()
                    simulation_app.update()
                    if stepper.abort_reason is not None:
                        termination = f"SAFETY_ABORT:{stepper.abort_reason}"
                        break
                    if not within_target:
                        motion_count += 1
            except _PalmLocalizationOnlyComplete:
                pass
            finally:
                if worker_process is not None:
                    try:
                        assert worker_process.stdin is not None
                        worker_process.stdin.write('{"command":"stop"}\n')
                        worker_process.stdin.flush()
                        worker_process.wait(timeout=10.0)
                    except Exception:
                        worker_process.terminate()
                    (output / "foundationpose_worker.log").write_text(
                        "".join(worker_log_lines), encoding="utf-8"
                    )

            final_active = robot.get_dof_positions(indices=0).numpy()[0][active_indices]
            final_hand_pose = np.asarray(
                inputs.robot_model.forward_kinematics(
                    tuple(final_active), enforce_limits=False
                )["handbase_link"],
                dtype=np.float64,
            )
            if (
                args.execute_grasp_lift
                and physical_precontact_hand_pose is not None
                and physical_precontact_truth_world_from_object is not None
            ):
                final_hand_pose = physical_precontact_hand_pose
                posthoc_truth_world_from_object = (
                    physical_precontact_truth_world_from_object
                )
            else:
                posthoc_truth_world_from_object = _world_from_prim(
                    stage,
                    str(scene["roots"]["object"]),
                    Usd=Usd,
                    UsdGeom=UsdGeom,
                )
            truth_axis = np.asarray(
                posthoc_truth_world_from_object[:3, 2], dtype=np.float64
            )
            truth_axis /= np.linalg.norm(truth_axis)
            truth_task_frame = _yaw_free_object_frame(
                posthoc_truth_world_from_object[:3, 3], truth_axis
            )
            if same_run_global_task_frame is not None:
                global_localization = result["global_localization"]
                global_localization["posthoc_position_error_mm"] = (
                    1000.0
                    * float(
                        np.linalg.norm(
                            same_run_global_task_frame[:3, 3]
                            - truth_task_frame[:3, 3]
                        )
                    )
                )
                global_localization["posthoc_axis_error_deg"] = math.degrees(
                    math.acos(
                        np.clip(
                            float(
                                same_run_global_task_frame[:3, 2]
                                @ truth_task_frame[:3, 2]
                            ),
                            -1.0,
                            1.0,
                        )
                    )
                )
                global_localization[
                    "posthoc_truth_world_from_object_row_major"
                ] = posthoc_truth_world_from_object.ravel().tolist()
                global_localization["posthoc_only_not_used_for_control"] = True
            truth_target_hand = truth_task_frame @ object_from_hand
            for frame_record in servo_frames:
                estimated = _matrix4(
                    frame_record["estimated_world_from_object_yaw_free_row_major"],
                    "stored estimated task frame",
                )
                frame_record["posthoc_pose_error"] = {
                    "position_mm": 1000.0
                    * float(
                        np.linalg.norm(
                            estimated[:3, 3] - truth_task_frame[:3, 3]
                        )
                    ),
                    "axis_deg": math.degrees(
                        math.acos(
                            np.clip(
                                float(estimated[:3, 2] @ truth_task_frame[:3, 2]),
                                -1.0,
                                1.0,
                            )
                        )
                    ),
                }
            posthoc_final_hand_target_position_error = float(
                np.linalg.norm(
                    final_hand_pose[:3, 3] - truth_target_hand[:3, 3]
                )
            )
            posthoc_final_hand_target_axis_error = math.acos(
                np.clip(
                    float(
                        final_hand_pose[:3, 2] @ truth_target_hand[:3, 2]
                    ),
                    -1.0,
                    1.0,
                )
            )
            if (
                args.servo_control_policy == "frozen_plan"
                and frozen_plan_record is not None
            ):
                posthoc_control_estimate = _matrix4(
                    frozen_plan_record["frozen_world_from_object_row_major"],
                    "frozen visual estimate used for control",
                )
                posthoc_control_estimate_role = (
                    "FROZEN_STATIONARY_VISUAL_ESTIMATE_USED_FOR_CONTROL"
                )
            else:
                posthoc_control_estimate = _matrix4(
                    servo_frames[-1][
                        "estimated_world_from_object_yaw_free_row_major"
                    ],
                    "last visual estimate used for control",
                )
                posthoc_control_estimate_role = (
                    "LAST_ONLINE_VISUAL_ESTIMATE_USED_FOR_CONTROL"
                )
            posthoc_control_visual_position_error = float(
                np.linalg.norm(
                    posthoc_control_estimate[:3, 3]
                    - truth_task_frame[:3, 3]
                )
            )
            posthoc_control_visual_axis_error = math.acos(
                np.clip(
                    float(
                        posthoc_control_estimate[:3, 2]
                        @ truth_task_frame[:3, 2]
                    ),
                    -1.0,
                    1.0,
                )
            )
            final_reobservation_error = None
            if (
                servo_frames
                and servo_frames[-1].get("observation_role")
                == "FINAL_STATIC_REOBSERVATION_NOT_CONTROL"
            ):
                final_reobservation_error = dict(
                    servo_frames[-1]["posthoc_pose_error"]
                )
            servo_result = {
                "schema_version": (
                    "kcg_frozen_target_visual_grasp_trace_v1"
                    if args.execute_grasp_lift
                    else "kcg_frozen_target_pregrasp_trace_v1"
                    if args.servo_control_policy == "frozen_plan"
                    else "kcg_palm_visual_servo_trace_v1"
                ),
                "estimator": args.servo_estimator,
                "controller": (
                    "FROZEN_FIVE_DOF_MINIMUM_JERK_PLAN_V1"
                    if args.servo_control_policy == "frozen_plan"
                    else (
                        "YAW_FREE_CAMERA_ORBIT_THEN_POSITION_SERVO_V1"
                        if args.servo_control_policy
                        == "camera_orbit_then_position"
                        else (
                            "YAW_FREE_AXIS_THEN_POSITION_BOUNDED_STEP_SERVO_V1"
                            if args.servo_control_policy == "axis_then_position"
                            else "YAW_FREE_BOUNDED_STEP_POSITION_AXIS_SERVO_V1"
                        )
                    )
                ),
                "termination": termination,
                "online_target_reached": (
                    termination.startswith("FROZEN_PLAN_REACHED")
                    if args.servo_control_policy == "frozen_plan"
                    else termination == "ONLINE_TARGET_REACHED"
                ),
                "motion_count": motion_count,
                "capture_count": len(servo_frames),
                "elapsed_wall_s": time.perf_counter() - servo_started,
                "limits": {
                    "maximum_translation_step_m": float(
                        args.servo_maximum_translation_step_m
                    ),
                    "maximum_rotation_step_deg": float(
                        args.servo_maximum_rotation_step_deg
                    ),
                    "position_tolerance_m": float(args.servo_position_tolerance_m),
                    "axis_tolerance_deg": float(args.servo_axis_tolerance_deg),
                    "consecutive_required": int(args.servo_consecutive_required),
                },
                "frames": servo_frames,
                "final_world_from_hand_row_major": final_hand_pose.ravel().tolist(),
                "posthoc_truth_world_from_object_row_major": (
                    posthoc_truth_world_from_object.ravel().tolist()
                ),
                "posthoc_truth_target_world_from_hand_row_major": (
                    truth_target_hand.ravel().tolist()
                ),
                "posthoc_control_visual_estimate_error": {
                    "position_mm": (
                        1000.0 * posthoc_control_visual_position_error
                    ),
                    "axis_deg": math.degrees(
                        posthoc_control_visual_axis_error
                    ),
                    "estimate_role": posthoc_control_estimate_role,
                },
                "posthoc_final_reobservation_error": (
                    final_reobservation_error
                ),
                "posthoc_final_hand_target_execution_error": {
                    "position_mm": (
                        1000.0 * posthoc_final_hand_target_position_error
                    ),
                    "axis_deg": math.degrees(
                        posthoc_final_hand_target_axis_error
                    ),
                    "meaning": (
                        "ACTUAL_HAND_POSE_VERSUS_HAND_TARGET_COMPUTED_FROM_"
                        "POSTHOC_OBJECT_TRUTH"
                    ),
                },
                "posthoc_final_position_error_mm": (
                    1000.0 * posthoc_final_hand_target_position_error
                ),
                "posthoc_final_axis_error_deg": math.degrees(
                    posthoc_final_hand_target_axis_error
                ),
                "legacy_posthoc_final_error_fields_meaning": (
                    "HAND_TARGET_EXECUTION_ERROR_NOT_VISUAL_ESTIMATION_ERROR"
                ),
                "truth_inputs_used_for_control": [],
                "axial_yaw_consumed": False,
                "contact_or_finger_motion_authorized": bool(
                    args.execute_grasp_lift
                ),
                "frozen_plan": frozen_plan_record,
            }
            trace_name = (
                "frozen_target_plan_trace.json"
                if args.servo_control_policy == "frozen_plan"
                else "visual_servo_trace.json"
            )
            result_key = (
                "frozen_target_plan"
                if args.servo_control_policy == "frozen_plan"
                else "visual_servo"
            )
            result[result_key] = servo_result
            (output / trace_name).write_text(
                json.dumps(_json_ready(servo_result), ensure_ascii=False, indent=2)
                + "\n",
                encoding="utf-8",
            )
            actual = final_active
            actual_fk = final_hand_pose

        result.update(
            {
                "abort_reason": stepper.abort_reason,
                "completed_command_steps": int(stepper.step_index),
                "command_api_counts": command_counter,
                "maximum_joint_speed_rad_s": float(stepper.maximum_speed),
                "maximum_arm_tracking_error_rad": float(stepper.maximum_arm_error),
                "arrival_arm_joint_error_rad": float(
                    np.max(np.abs(handoff_actual[:7] - handoff_arm))
                ),
                "arrival_position_error_m": float(
                    np.linalg.norm(
                        handoff_actual_fk[:3, 3] - handoff_pose[:3, 3]
                    )
                ),
                "ft": ft_auditor.summary(),
            }
        )
        result["truth_inputs_used_for_control"] = []
        result["result_scope"] = (
            "same-reset global SAM mask plus float-depth geometry position/axis "
            "localization with independent PEM consistency checking, "
            "stationary palm localization, frozen target planning, "
            "q70/4.5 equal-normal three-finger grasp, lift, and hold"
            + (
                ", followed by one frozen body-COM disturbance condition"
                if args.postgrasp_disturbance_panel is not None
                else ""
            )
            + "; no visual "
            "updates were consumed during pregrasp motion or contact; saved E50 "
            "data defined only the camera-to-object observation relation"
            if args.execute_grasp_lift
            else "same-reset global SAM mask plus float-depth geometry "
            "position/axis localization with independent PEM consistency checking, "
            "stationary palm localization, frozen target planning, "
            "no-contact execution, and final static reobservation; no visual "
            "updates were consumed during motion; saved E50 data defined only "
            "the camera-to-object observation relation"
            if args.servo_control_policy == "frozen_plan"
            else (
                "same-reset handoff-to-pregrasp no-contact visual servo; global pose "
                "was transferred from the selected E50 run, so this is not a same-reset "
                "global-perception-to-servo claim"
                if args.servo_estimator is not None
                else "separate-reset handoff from the selected E50 global SAM-6D pose; "
                "not a same-reset end-to-end claim"
            )
        )
        (output / "ft_samples.json").write_text(
            json.dumps(_json_ready(ft_auditor.samples), ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        physical_result = result.get("physical_grasp", {})
        disturbance_result = physical_result.get(
            "postgrasp_disturbance", {}
        )
        exit_code = (
            0
            if not args.execute_grasp_lift
            else 0
            if physical_result.get("success") is True
            and (
                args.postgrasp_disturbance_panel is None
                or disturbance_result.get("core_condition_pass") is True
            )
            else 2
        )
    except _GlobalCaptureOnlyComplete:
        exit_code = 0
    except Exception as error:
        result["error_type"] = type(error).__name__
        result["error"] = str(error)
        result["traceback"] = traceback.format_exc()
        exit_code = 1
    finally:
        _close_rgbd_resources(rgbd_resources)
        (output / "runtime_result.json").write_text(
            json.dumps(_json_ready(result), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        simulation_app.close(exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
