"""One-shot DIAGNOSTIC_MOUNT_SEARCH_ONLY raw RGB-D runner."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from kcg_connector.d38999_cad_registration import (
    fixed_camera_model,
    project,
    shell25j_plug_cad_profile,
)
from kcg_connector.d38999_inhand_multiview import matrix_pose, pose_matrix
from kcg_connector.postgrasp_shadow_view_planner import (
    DIAGNOSTIC_MOUNT_CANDIDATES,
    DIAG_MOUNT_SCHEMA_VERSION,
    DIAG_MOUNT_TARGET_P,
    diagnostic_hp_envelope_samples,
    diagnostic_mount_hard_gates,
    diagnostic_optical_axis_angle_deg,
)

DIAG_CLIP_NEAR_M = 0.02
DIAG_CLIP_FAR_M = 0.30
DIAG_DEPTH_BAND_M = 0.005
DIAG_MIN_SHELL_PIXELS = 100
DIAG_MIN_SOCKET_PIXELS = 20
DIAG_MAX_CONDITION_5D = 1.0e6


def _sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gf_matrix_rows(matrix) -> list[list[float]]:
    return [
        [float(matrix[row][column]) for column in range(4)]
        for row in range(4)
    ]


def cv_camera_pose_from_usd_row_xform(matrix_rows) -> np.ndarray:
    """Convert USD -Z-forward/+Y-up camera axes to CV +Z/+Y-down."""
    usd = np.asarray(matrix_rows, dtype=np.float64)
    if usd.shape != (4, 4) or not np.all(np.isfinite(usd)):
        raise ValueError("USD camera transform must be a finite 4x4 matrix")
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = usd[:3, :3].T @ np.diag((1.0, -1.0, -1.0))
    pose[:3, 3] = usd[3, :3]
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError("USD camera rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError("USD camera rotation is not proper")
    return pose


def _t_wc_from_camera_model(camera) -> np.ndarray:
    rows = np.asarray(camera.world_to_camera, dtype=np.float64)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rows.T
    pose[:3, 3] = np.asarray(camera.position_world, dtype=np.float64)
    return pose


def _projected_pixel_count(camera, points):
    uv, depth = project(camera, points)
    valid = (
        (depth > DIAG_CLIP_NEAR_M)
        & (depth < DIAG_CLIP_FAR_M)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < camera.width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < camera.height)
    )
    if not np.any(valid):
        return 0
    pixels = np.rint(uv[valid]).astype(np.int32)
    return int(len(np.unique(pixels, axis=0)))


def _projection_condition_5d(camera, points, base_pose):
    steps = np.asarray((0.0002, 0.0002, 0.0002, math.radians(0.15), math.radians(0.15)))
    scale = np.asarray((0.0005, 0.0005, 0.0010, math.radians(2), math.radians(2)))
    base = np.asarray(base_pose, dtype=np.float64)

    def pixels(pose6):
        transform = pose_matrix(pose6)
        xyz = points.xyz @ transform[:3, :3].T + transform[:3, 3]
        return project(camera, xyz)[0].ravel()

    jac = np.column_stack(
        tuple(
            (pixels(base + np.eye(6)[i] * steps[i]) - pixels(base - np.eye(6)[i] * steps[i]))
            / (2.0 * steps[i])
            for i in range(5)
        )
    )
    values = np.linalg.svd(jac * scale, compute_uv=False)
    return float(values[0] / max(values[-1], 1.0e-12))


def _relative_nominal_plug_pose(nominal_hp, perturbed_hp):
    return np.linalg.inv(pose_matrix(nominal_hp)) @ pose_matrix(perturbed_hp)


def prefilter_diagnostic_candidate(nominal_hp, eye_plug, target_plug):
    """CPU hard gates over the deterministic 17-point 6D envelope."""
    nominal_hp = np.asarray(nominal_hp, dtype=np.float64)
    if nominal_hp.shape == (4, 4):
        nominal_pose = matrix_pose(nominal_hp)
    elif nominal_hp.shape == (6,):
        nominal_pose = nominal_hp
    else:
        raise ValueError("nominal_hp must be a 4x4 matrix or 6D pose")
    camera = fixed_camera_model(
        eye=tuple(float(v) for v in eye_plug),
        target=tuple(float(v) for v in target_plug),
        resolution=(640, 360),
    )
    shell = shell25j_plug_cad_profile(feature_set="shell_only").plug_mating
    socket = shell25j_plug_cad_profile(feature_set="socket_only").plug_mating
    samples = []
    for sample_hp in diagnostic_hp_envelope_samples(nominal_pose):
        rel = _relative_nominal_plug_pose(nominal_pose, sample_hp)
        rel_pose = matrix_pose(rel)
        shell_world = _transform_cad(shell, rel)
        socket_world = _transform_cad(socket, rel)
        shell_pixels = _projected_pixel_count(camera, shell_world.xyz)
        socket_pixels = _projected_pixel_count(camera, socket_world.xyz)
        condition = _projection_condition_5d(camera, socket, rel_pose)
        passed = bool(
            shell_pixels >= DIAG_MIN_SHELL_PIXELS
            and socket_pixels >= DIAG_MIN_SOCKET_PIXELS
            and condition <= DIAG_MAX_CONDITION_5D
        )
        samples.append(
            {
                "shell_pixels": shell_pixels,
                "socket_pixels": socket_pixels,
                "condition_5d": condition,
                "passed": passed,
            }
        )
    return {
        "passed": all(item["passed"] for item in samples),
        "envelope_sample_count": len(samples),
        "samples": samples,
    }


def _transform_cad(cad, matrix):
    from kcg_connector.d38999_cad_registration import CadPoints

    xyz = cad.xyz @ matrix[:3, :3].T + matrix[:3, 3]
    normal = cad.normal @ matrix[:3, :3].T
    return CadPoints(xyz, normal, cad.label, cad.edge)


def _formal_raw_metrics(rgb, depth, camera):
    depth = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(depth) & (depth > 0.0)
    height, width = depth.shape
    h0, h1 = int(height * 0.2), int(height * 0.8)
    w0, w1 = int(width * 0.2), int(width * 0.8)
    central = finite[h0:h1, w0:w1]
    metrics = {
        "depth_valid_fraction": float(np.mean(finite)),
        "central_depth_fraction": float(np.mean(central)) if central.size else 0.0,
        "finite_positive_depth_pixels": int(np.sum(finite)),
        "projected_depth_identity": "PROJECTED_DEPTH_CONSISTENCY_ONLY_NOT_FEATURE_IDENTITY",
        "mechanical_feasibility": "MECHANICAL_FEASIBILITY_UNVERIFIED",
    }
    shell = shell25j_plug_cad_profile(feature_set="shell_only").plug_mating
    socket = shell25j_plug_cad_profile(feature_set="socket_only").plug_mating
    for name, cad in (("shell", shell), ("socket", socket)):
        # Both the diagnostic camera model and CAD are expressed in the
        # nominal Plug frame.  Raw depth is camera-local, so mixing world CAD
        # points with this camera would silently corrupt the support metric.
        uv, predicted = project(camera, cad.xyz)
        u = np.rint(uv[:, 0]).astype(np.int32)
        v = np.rint(uv[:, 1]).astype(np.int32)
        valid = (
            (predicted > DIAG_CLIP_NEAR_M)
            & (predicted < DIAG_CLIP_FAR_M)
            & (u >= 0)
            & (u < width)
            & (v >= 0)
            & (v < height)
        )
        observed = np.full(len(predicted), np.inf)
        observed[valid] = depth[v[valid], u[valid]]
        obs_valid = np.isfinite(observed) & (observed > 0.0)
        consistent = obs_valid & (np.abs(predicted - observed) <= DIAG_DEPTH_BAND_M)
        behind = obs_valid & (predicted - observed > DIAG_DEPTH_BAND_M)
        metrics[f"projected_{name}_depth_support"] = float(
            np.mean(consistent[valid]) if int(np.sum(valid)) else 0.0
        )
        metrics[f"projected_{name}_foreground_occlusion"] = float(
            np.mean(behind[valid]) if int(np.sum(valid)) else 0.0
        )
    gray = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 30, 90)
    edge_distance = cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 3)
    edge_points = shell.xyz[shell.edge]
    uv, predicted = project(camera, edge_points)
    edge_valid = (
        (predicted > DIAG_CLIP_NEAR_M)
        & (predicted < DIAG_CLIP_FAR_M)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < height)
    )
    sampled = cv2.remap(
        edge_distance.astype(np.float32),
        uv[:, 0].astype(np.float32).reshape(-1, 1),
        uv[:, 1].astype(np.float32).reshape(-1, 1),
        cv2.INTER_LINEAR,
    ).ravel()
    metrics["edge_support_fraction"] = float(
        np.mean(sampled[edge_valid] < 5.0) if np.any(edge_valid) else 0.0
    )
    socket_rel = matrix_pose(np.eye(4))
    metrics["condition_number_5d"] = _projection_condition_5d(
        camera,
        _transform_cad(socket, np.eye(4)),
        socket_rel,
    )
    metrics["foreground_occlusion_fraction"] = float(
        0.5
        * (
            metrics["projected_shell_foreground_occlusion"]
            + metrics["projected_socket_foreground_occlusion"]
        )
    )
    return metrics


def diagnostic_camera_world_transform(
    current_hand_transform, nominal_hand_to_plug, eye_plug, target_plug
):
    """Place one fixed mount from current robot FK and nominal grasp CAD."""
    camera_in_plug = fixed_camera_model(
        eye=tuple(float(value) for value in eye_plug),
        target=tuple(float(value) for value in target_plug),
        resolution=(1280, 720),
    )
    return (
        np.asarray(current_hand_transform, dtype=np.float64)
        @ np.asarray(nominal_hand_to_plug, dtype=np.float64)
        @ _t_wc_from_camera_model(camera_in_plug)
    )


def run_diagnostic_mount_search(
    *,
    repository: Path,
    arguments,
    Gf,
    Usd,
    UsdGeom,
    UsdLux,
    stage,
    world,
    simulation_app,
    tabletop,
    current_hand_target,
    rate_hz,
    global_step,
    current_hand_transform,
    nominal_hand_to_plug,
    source_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    output_root = (
        Path(arguments.output_dir).expanduser().resolve()
        / "postgrasp_diag_mount_search"
    )
    result = {
        "mode": "DIAGNOSTIC_MOUNT_SEARCH_ONLY",
        "control_authorized": False,
        "formal_estimator_input": False,
        "semantic_present": False,
        "posthoc_identity_audit": "POSTHOC_IDENTITY_AUDIT_NOT_IMPLEMENTED",
        "mount_formally_accepted": False,
        "status": "STARTED",
    }
    if output_root.exists():
        result["status"] = "FAIL_CLOSED_OUTPUT_EXISTS"
        return result
    output_root.mkdir(parents=True, exist_ok=False)
    try:
        from isaacsim.sensors.camera import Camera
        import omni.replicator.core as rep
        from PIL import Image

        import kcg_connector.isaac_d38999_rgbd_runtime as raw_runtime

        from kcg_connector.rgbd_pose_bootstrap import load_rgbd_bootstrap

        rgbd_base = load_rgbd_bootstrap(
            Path(arguments.rgbd_config).expanduser().resolve()
        )
        bound_source_hashes = dict(source_hashes or {})
        bound_source_hashes.update(
            {
                "postgrasp_diag_mount_search_sha256": _sha256_file(__file__),
                "isaac_d38999_rgbd_runtime_sha256": _sha256_file(
                    raw_runtime.__file__
                ),
                "rgbd_config_sha256": _sha256_file(
                    Path(arguments.rgbd_config).expanduser().resolve()
                ),
            }
        )
        records = []
        failures = []
        nominal_hp = np.asarray(nominal_hand_to_plug, dtype=np.float64)
        current_hand = np.asarray(current_hand_transform, dtype=np.float64)
        for candidate_id, eye_plug in DIAGNOSTIC_MOUNT_CANDIDATES:
            record = {
                "candidate_id": candidate_id,
                "eye_plug_m": list(eye_plug),
                "target_plug_m": list(DIAG_MOUNT_TARGET_P),
                "optical_axis_angle_deg": diagnostic_optical_axis_angle_deg(
                    eye_plug, DIAG_MOUNT_TARGET_P
                ),
                "near_clip_m": DIAG_CLIP_NEAR_M,
                "far_clip_m": DIAG_CLIP_FAR_M,
            }
            gate = diagnostic_mount_hard_gates(eye_plug, DIAG_MOUNT_TARGET_P)
            record["hard_gates"] = gate
            if not gate["passed"]:
                record["capture_status"] = "HARD_GATE_REJECTED"
                failures.append(candidate_id)
                records.append(record)
                continue
            prefilter = prefilter_diagnostic_candidate(
                nominal_hp, eye_plug, DIAG_MOUNT_TARGET_P
            )
            record["envelope_prefilter"] = prefilter
            if not prefilter["passed"]:
                record["capture_status"] = "ENVELOPE_PREFILTER_REJECTED"
                failures.append(candidate_id)
                records.append(record)
                continue
            camera_in_plug = fixed_camera_model(
                eye=tuple(float(v) for v in eye_plug),
                target=tuple(float(v) for v in DIAG_MOUNT_TARGET_P),
                resolution=(1280, 720),
            )
            t_wc = diagnostic_camera_world_transform(
                current_hand,
                nominal_hp,
                eye_plug,
                DIAG_MOUNT_TARGET_P,
            )
            t_hc = nominal_hp @ _t_wc_from_camera_model(camera_in_plug)
            eye_world = tuple(float(v) for v in t_wc[:3, 3])
            forward_world = t_wc[:3, :3] @ np.asarray((0.0, 0.0, 1.0))
            target_world = tuple(
                float(v) for v in (np.asarray(t_wc[:3, 3]) + forward_world)
            )
            camera_path = f"/World/PostgraspDiagCamera_{candidate_id}"
            camera_prim = stage.GetPrimAtPath(camera_path)
            if camera_prim is None or not camera_prim.IsValid():
                camera_prim = UsdGeom.Camera.Define(stage, camera_path)
            direction = np.asarray(target_world) - np.asarray(eye_world)
            direction = direction / np.linalg.norm(direction)
            rotation = Gf.Rotation(
                Gf.Vec3d(0.0, 0.0, -1.0), Gf.Vec3d(*direction)
            )
            matrix = Gf.Matrix4d(1.0)
            matrix.SetRotate(rotation)
            matrix.SetTranslateOnly(Gf.Vec3d(*eye_world))
            UsdGeom.Xformable(camera_prim).ClearXformOpOrder()
            UsdGeom.Xformable(camera_prim).AddTransformOp().Set(matrix)
            camera_prim.CreateFocalLengthAttr(24.0)
            camera_prim.CreateHorizontalApertureAttr(20.955)
            camera_prim.CreateVerticalApertureAttr(20.955 * 720.0 / 1280.0)
            camera_prim.CreateClippingRangeAttr(
                Gf.Vec2f(DIAG_CLIP_NEAR_M, DIAG_CLIP_FAR_M)
            )
            diag_rgbd = replace(
                rgbd_base,
                camera=replace(
                    rgbd_base.camera,
                    prim_path=camera_path,
                    frame_id=f"diag_mount_{candidate_id.lower()}",
                    eye_m=eye_world,
                    target_m=target_world,
                    resolution=(1280, 720),
                ),
            )
            capture_dir = output_root / f"raw_{candidate_id.lower()}"
            timestamp = datetime.now(timezone.utc).isoformat()
            capture = raw_runtime.capture_d38999_rgbd_raw_formal(
                bindings={
                    "Camera": Camera,
                    "Gf": Gf,
                    "Image": Image,
                    "Usd": Usd,
                    "UsdGeom": UsdGeom,
                    "UsdLux": UsdLux,
                    "rep": rep,
                },
                simulation_app=simulation_app,
                world=world,
                stage=stage,
                tabletop=tabletop,
                rgbd=diag_rgbd,
                output_dir=capture_dir,
                camera_clipping_range_m=(
                    DIAG_CLIP_NEAR_M,
                    DIAG_CLIP_FAR_M,
                ),
            )
            if not capture.passed:
                record["capture_status"] = "RAW_CAPTURE_FAILED"
                failures.append(candidate_id)
                records.append(record)
                continue
            capture_camera = capture.metrics.get("camera")
            required_camera_fields = {
                "clipping_range_m",
                "focal_length_mm",
                "frame_id",
                "horizontal_aperture_mm",
                "intrinsics",
                "resolution",
            }
            if not isinstance(capture_camera, dict) or not (
                required_camera_fields <= set(capture_camera)
            ):
                record["capture_status"] = "RAW_CAPTURE_CONTRACT_INCOMPLETE"
                record["missing_camera_fields"] = sorted(
                    required_camera_fields - set(capture_camera or {})
                )
                failures.append(candidate_id)
                records.append(record)
                continue
            usd_camera_xform = UsdGeom.Xformable(
                camera_prim
            ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            usd_camera_rows = _gf_matrix_rows(usd_camera_xform)
            capture_t_wc = cv_camera_pose_from_usd_row_xform(
                usd_camera_rows
            )
            capture_t_hc = np.linalg.inv(current_hand) @ capture_t_wc
            formal = _formal_raw_metrics(
                capture.rgb, capture.depth, camera_in_plug
            )
            finite_metrics = all(
                np.isfinite(float(formal[key]))
                for key in (
                    "projected_shell_depth_support",
                    "projected_socket_depth_support",
                    "foreground_occlusion_fraction",
                    "edge_support_fraction",
                    "condition_number_5d",
                )
            )
            if not finite_metrics:
                record["capture_status"] = "NON_FINITE_METRICS"
                failures.append(candidate_id)
                records.append({"record": record, "formal_metrics": formal})
                continue
            record.update(
                {
                    "capture_status": "CAPTURED",
                    "timestamp_utc": timestamp,
                    "capture_dir": str(capture_dir),
                    "global_physics_step": int(global_step),
                    "capture_camera": capture_camera,
                    "capture_T_WH": current_hand.tolist(),
                    "capture_T_HC": capture_t_hc.tolist(),
                    "capture_T_WC": capture_t_wc.tolist(),
                    "design_T_HC": t_hc.tolist(),
                    "design_T_WC": t_wc.tolist(),
                    "capture_T_convention": (
                        "COLUMN_VECTOR_CV_CAMERA_POSITIVE_Z_FORWARD_Y_DOWN"
                    ),
                    "capture_usd_camera_xform_row_major": usd_camera_rows,
                    "capture_usd_camera_convention": (
                        "USD_GF_ROW_MAJOR_CAMERA_LOCAL_NEGATIVE_Z_FORWARD"
                    ),
                    "formal_metrics": formal,
                    "score": None,
                }
            )
            records.append(record)
        manifest = {
            "schema_version": DIAG_MOUNT_SCHEMA_VERSION,
            "mode": "DIAGNOSTIC_MOUNT_SEARCH_ONLY",
            "virtual_candidates": True,
            "control_authorized": False,
            "formal_estimator_input": False,
            "semantic_present": False,
            "posthoc_identity_audit": "POSTHOC_IDENTITY_AUDIT_NOT_IMPLEMENTED",
            "mount_formally_accepted": False,
            "records": records,
            "failures": failures,
            "source_hashes": bound_source_hashes,
        }
        (output_root / "diag_manifest.json").write_text(
            json.dumps(manifest, allow_nan=False, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        result.update(
            {
                "records": records,
                "failures": failures,
                "manifest_path": str(output_root / "diag_manifest.json"),
                "status": (
                    "COMPLETED_DIAGNOSTIC_RAW_ONLY"
                    if not failures
                    else "COMPLETED_WITH_CANDIDATE_FAILURES"
                ),
            }
        )
    except Exception as exception:
        result["status"] = "DIAG_ABORT_SAFE"
        result["error"] = f"{type(exception).__name__}: {exception}"
    return result


__all__ = [
    "cv_camera_pose_from_usd_row_xform",
    "diagnostic_camera_world_transform",
    "prefilter_diagnostic_candidate",
    "run_diagnostic_mount_search",
]
