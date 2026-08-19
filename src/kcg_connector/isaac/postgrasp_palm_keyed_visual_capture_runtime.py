"""Thin same-episode Palm RGB-D capture adapter for the keyed visual gate.

The adapter reuses the already-authored canonical ``PalmCamera`` child prim.  It
does not define a camera, edit an xform, read a camera world pose, or accept an
endpoint pose.  Physics is paused while Replicator renders zero-delta frames.

The first integration stage intentionally derives only a central valid-depth
ROI and supplies ``occlusion_mask=None``.  The downstream evaluator must then
return ``KEY_REGION_OCCLUSION_UNKNOWN`` with every plan/control authorization
false.  This proves the live capture path without pretending that the current
ROI is a reliable keyed connector segmentation.
"""

from __future__ import annotations

import importlib.util
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = "kcg_d38999_postgrasp_palm_keyed_visual_capture_v1"
MODE = "REUSE_EXISTING_PALM_PRIM_RGBD_THEN_FAIL_CLOSED_OCCLUSION_UNKNOWN"
RESOLUTION_PX = (1280, 720)
ANNOTATORS_EXACTLY = ("rgb", "distance_to_image_plane")
PALM_PRIM_SUFFIX = "/PalmCamera"
WARMUP_FRAMES = 2
RT_SUBFRAMES = 2
MAXIMUM_CAPTURE_Q_DRIFT_RAD = 0.002
REPORT_FILENAME = "report.json"
RGB_FILENAME = "rgb.png"
DEPTH_FILENAME = "depth_m.npy"


def _load_visual_evaluator():
    module_path = Path(__file__).with_name(
        "postgrasp_palm_keyed_visual_runtime.py"
    )
    spec = importlib.util.spec_from_file_location(
        "postgrasp_palm_keyed_visual_runtime_for_capture", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Palm keyed visual evaluator cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_VISUAL = _load_visual_evaluator()
evaluate_postgrasp_palm_keyed_visual_control = (
    _VISUAL.evaluate_postgrasp_palm_keyed_visual_control
)


def _base_result(output_subdir: Any, palm_prim_path: Any) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": MODE,
        "status": "STARTED",
        "reason": None,
        "rejection_code": None,
        "capture_passed": False,
        "observation_passed": False,
        "plan_authorized": False,
        "simulation_prealign_target_authorized": False,
        "control_authorized": False,
        "simulation_prealign_control_authorized": False,
        "simulation_insertion_control_authorized": False,
        "hardware_control_authorized": False,
        "safe_stop_required": True,
        "palm_prim_path": palm_prim_path if isinstance(palm_prim_path, str) else None,
        "output_subdir": str(output_subdir) if output_subdir is not None else None,
        "annotators": list(ANNOTATORS_EXACTLY),
        "resolution_px": list(RESOLUTION_PX),
        "timeline": {},
        "joint_capture": {},
        "input_derivation": {},
        "visual_evaluator": None,
        "artifacts": {},
        "resource_cleanup": {},
        "camera_pose_written": False,
        "object_pose_read": False,
        "contact_or_collider_read": False,
        "semantic_annotator_requested": False,
    }


def _abort(
    result: Mapping[str, Any],
    code: str,
    reason: str,
) -> dict[str, Any]:
    aborted = dict(result)
    aborted.update(
        status="ABORT_SAFE",
        reason=reason,
        rejection_code=code,
        plan_authorized=False,
        simulation_prealign_target_authorized=False,
        control_authorized=False,
        simulation_prealign_control_authorized=False,
        simulation_insertion_control_authorized=False,
        hardware_control_authorized=False,
        safe_stop_required=True,
    )
    return aborted


def _transform(value: Any, label: str) -> np.ndarray:
    try:
        transform = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f"{label} must be a finite 4x4 transform")
    if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-9):
        raise ValueError(f"{label} last row is invalid")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-6):
        raise ValueError(f"{label} rotation is not orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1.0e-6):
        raise ValueError(f"{label} rotation determinant is not +1")
    return transform


def _joint_positions(robot: Any, arm_indices: Sequence[int]) -> np.ndarray:
    values = np.asarray(robot.get_joint_positions(), dtype=np.float64).ravel()
    indices = np.asarray(arm_indices)
    if indices.shape != (7,) or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("arm_indices must contain exactly seven integer indices")
    indices = indices.astype(np.int64, copy=False)
    if len(set(int(value) for value in indices)) != 7:
        raise ValueError("arm_indices must be unique")
    if np.any(indices < 0) or np.any(indices >= values.size):
        raise ValueError("arm_indices are outside the robot joint vector")
    selected = values[indices]
    if not np.all(np.isfinite(selected)):
        raise ValueError("actual arm joint positions must be finite")
    return selected


def _valid_camera_prim(stage: Any, palm_prim_path: str):
    if not isinstance(palm_prim_path, str) or not palm_prim_path.endswith(
        PALM_PRIM_SUFFIX
    ):
        raise ValueError(
            f"palm_prim_path must identify an existing {PALM_PRIM_SUFFIX} child"
        )
    prim = stage.GetPrimAtPath(palm_prim_path)
    if prim is None or not prim.IsValid():
        raise RuntimeError(f"existing Palm camera prim is unavailable: {palm_prim_path}")
    get_type_name = getattr(prim, "GetTypeName", None)
    if callable(get_type_name) and get_type_name() != "Camera":
        raise RuntimeError("existing Palm child prim is not a Camera")
    get_attribute = getattr(prim, "GetAttribute", None)
    if not callable(get_attribute):
        raise RuntimeError("existing Palm camera cannot expose clippingRange read-only")
    clipping_attribute = get_attribute("clippingRange")
    if clipping_attribute is None or not clipping_attribute.IsValid():
        raise RuntimeError("existing Palm camera clippingRange is unavailable")
    clipping = np.asarray(clipping_attribute.Get(), dtype=np.float64).ravel()
    if (
        clipping.shape != (2,)
        or not np.all(np.isfinite(clipping))
        or not math.isclose(float(clipping[0]), 0.02, abs_tol=1.0e-9)
        or float(clipping[1]) <= float(clipping[0])
    ):
        raise RuntimeError("existing Palm camera clippingRange must begin at 0.02 m")
    return prim


def _normalize_capture_arrays(rgb_value: Any, depth_value: Any):
    rgba = np.asarray(rgb_value)
    depth = np.asarray(depth_value)
    height, width = RESOLUTION_PX[1], RESOLUTION_PX[0]
    if rgba.ndim != 3 or rgba.shape[:2] != (height, width) or rgba.shape[2] < 3:
        raise RuntimeError("Palm rgb annotator must return 720x1280 with >=3 channels")
    if depth.shape != (height, width) or not np.issubdtype(depth.dtype, np.number):
        raise RuntimeError("Palm distance annotator must return numeric 720x1280")
    rgb = np.asarray(rgba[:, :, :3], dtype=np.uint8)
    depth = depth.astype(np.float32, copy=False)
    valid = np.isfinite(depth) & (depth > 0.0)
    if not np.any(valid):
        raise RuntimeError("Palm depth contains no finite positive pixels")
    return rgb, depth, valid


def _central_valid_depth_roi(depth: np.ndarray) -> dict[str, Any]:
    height, width = depth.shape
    rows, columns = np.indices(depth.shape)
    central = (
        (rows >= height // 4)
        & (rows < height - height // 4)
        & (columns >= width // 4)
        & (columns < width - width // 4)
    )
    valid = np.isfinite(depth) & (depth > 0.0)
    face = central & valid
    face_pixels = int(np.count_nonzero(face))
    if face_pixels == 0:
        raise RuntimeError("central valid-depth ROI is empty")
    face_rows, face_columns = np.nonzero(face)
    center = (float(np.mean(face_columns)), float(np.mean(face_rows)))
    return {
        "connector_face_mask": face,
        "face_center_uv": center,
        "occlusion_mask": None,
        "diagnostics": {
            "face_mask_source": "CENTRAL_VALID_DEPTH_ROI_IMAGE_ONLY_BOOTSTRAP",
            "face_pixels": face_pixels,
            "valid_depth_pixels": int(np.count_nonzero(valid)),
            "face_center_source": "CENTRAL_VALID_DEPTH_ROI_CENTROID",
            "occlusion_status": "UNKNOWN_FAIL_CLOSED",
            "semantic_or_object_truth_used": False,
            "reliable_key_segmentation_claimed": False,
        },
    }


def _camera_intrinsics_contract() -> dict[str, float | int]:
    width, height = RESOLUTION_PX
    focal_length_mm = 24.0
    horizontal_aperture_mm = 20.955
    focal_px = focal_length_mm * width / horizontal_aperture_mm
    return {
        "fx": focal_px,
        "fy": focal_px,
        "cx": (width - 1) * 0.5,
        "cy": (height - 1) * 0.5,
        "width": width,
        "height": height,
    }


def _evaluator_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "schema_version",
        "status",
        "reason",
        "rejection_code",
        "observation_passed",
        "plan_authorized",
        "simulation_prealign_target_authorized",
        "control_authorized",
        "simulation_prealign_control_authorized",
        "simulation_insertion_control_authorized",
        "hardware_control_authorized",
        "safe_stop_required",
    )
    return {field: value.get(field) for field in fields}


def _cleanup_resources(
    annotators: Sequence[Any],
    render_product: Any,
    render_product_path: str | None,
) -> dict[str, Any]:
    errors: list[str] = []
    detached = 0
    for annotator in reversed(tuple(annotators)):
        try:
            if not isinstance(render_product_path, str) or not render_product_path:
                raise RuntimeError("render product path missing during annotator cleanup")
            annotator.detach([render_product_path])
            detached += 1
        except Exception as exc:  # pragma: no cover - Isaac boundary
            errors.append(f"annotator.detach: {type(exc).__name__}: {exc}")
    render_product_destroyed = render_product is None
    if render_product is not None:
        try:
            render_product.destroy()
            render_product_destroyed = True
        except Exception as exc:  # pragma: no cover - Isaac boundary
            errors.append(f"render_product.destroy: {type(exc).__name__}: {exc}")
    return {
        "annotator_detach_count": detached,
        "render_product_destroyed": render_product_destroyed,
        "errors": errors,
        "resources_released": bool(
            detached == len(tuple(annotators))
            and render_product_destroyed
            and not errors
        ),
        "stage_prims_removed": 0,
        "camera_prim_preserved": True,
    }


def _restore_timeline(
    *, world: Any, simulation_app: Any, was_playing: bool
) -> dict[str, Any]:
    playing_before_restore = bool(world.is_playing())
    if was_playing and not playing_before_restore:
        world.play()
        simulation_app.update()
    elif not was_playing and playing_before_restore:
        world.pause()
        simulation_app.update()
    playing_after_restore = bool(world.is_playing())
    return {
        "playing_before_capture": was_playing,
        "playing_before_restore": playing_before_restore,
        "playing_after_restore": playing_after_restore,
        "restored": playing_after_restore is was_playing,
    }


def _write_report(output_dir: Path, result: Mapping[str, Any]) -> None:
    (output_dir / REPORT_FILENAME).write_text(
        json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_postgrasp_palm_keyed_visual_capture(
    *,
    output_subdir: Path | str,
    stage: Any,
    world: Any,
    simulation_app: Any,
    robot: Any,
    arm_indices: Sequence[int],
    rep: Any,
    palm_prim_path: str,
    scene_schema_version: str,
    scene_profile_id: str,
    fixed_orientation_token: str,
    keyed_model_id: str,
    T_HC_frozen_configured: Any,
    T_WH_from_actual_q: Any,
    T_WR_fixed_configured: Any,
    T_RP_target_configured: Any,
) -> dict[str, Any]:
    """Capture one frozen Palm frame and exercise the fail-closed evaluator.

    All exceptions are converted to ``ABORT_SAFE``.  This first-stage adapter
    intentionally has no C2/accuracy inputs because unknown occlusion must
    reject before those later-stage contracts are considered.
    """

    result = _base_result(output_subdir, palm_prim_path)
    output_dir: Path | None = None
    render_product = None
    render_product_path: str | None = None
    annotators: list[Any] = []
    was_playing: bool | None = None
    timeline_was_paused = False
    output_dir_created = False
    try:
        output_dir = Path(output_subdir).expanduser().resolve()
        if output_dir.exists():
            return _abort(
                result,
                "OUTPUT_SUBDIR_ALREADY_EXISTS",
                "capture output subdirectory already exists",
            )
        output_dir.mkdir(parents=True, exist_ok=False)
        output_dir_created = True

        # Validate all configured transforms without applying any of them to
        # the stage.  Their names make the runtime/config boundary explicit.
        t_hc = _transform(T_HC_frozen_configured, "T_HC_frozen_configured")
        t_wh = _transform(T_WH_from_actual_q, "T_WH_from_actual_q")
        t_wr = _transform(T_WR_fixed_configured, "T_WR_fixed_configured")
        t_rp = _transform(T_RP_target_configured, "T_RP_target_configured")

        palm_prim = _valid_camera_prim(stage, palm_prim_path)
        q_before = _joint_positions(robot, arm_indices)
        was_playing = bool(world.is_playing())
        if was_playing:
            world.pause()
            timeline_was_paused = True
            simulation_app.update()
        else:
            timeline_was_paused = True
        if bool(world.is_playing()):
            raise RuntimeError("timeline did not pause before Palm capture")

        render_product = rep.create.render_product(
            palm_prim,
            RESOLUTION_PX,
            name="D38999PostgraspPalmKeyedVisualProduct",
        )
        render_product_path = render_product.path
        for name in ANNOTATORS_EXACTLY:
            annotator = rep.AnnotatorRegistry.get_annotator(name)
            annotator.attach([render_product_path])
            annotators.append(annotator)

        for _ in range(WARMUP_FRAMES):
            rep.orchestrator.step(
                rt_subframes=RT_SUBFRAMES,
                delta_time=0.0,
                pause_timeline=True,
            )
        rgb, depth, valid_depth = _normalize_capture_arrays(
            annotators[0].get_data(), annotators[1].get_data()
        )
        q_after = _joint_positions(robot, arm_indices)
        q_drift = float(np.max(np.abs(q_after - q_before)))
        result["joint_capture"] = {
            "actual_arm_q_before_capture_rad": q_before.tolist(),
            "actual_arm_q_after_capture_rad": q_after.tolist(),
            "maximum_absolute_drift_rad": q_drift,
            "maximum_allowed_drift_rad": MAXIMUM_CAPTURE_Q_DRIFT_RAD,
        }
        if q_drift > MAXIMUM_CAPTURE_Q_DRIFT_RAD:
            raise RuntimeError(
                "CAPTURE_Q_DRIFT_ABOVE_LIMIT: robot moved during Palm capture"
            )

        from PIL import Image

        Image.fromarray(rgb).save(output_dir / RGB_FILENAME)
        np.save(output_dir / DEPTH_FILENAME, depth)
        result["artifacts"] = {
            "rgb": RGB_FILENAME,
            "depth_m": DEPTH_FILENAME,
            "report": REPORT_FILENAME,
        }
        result["capture_passed"] = True
        result["capture"] = {
            "camera_prim_reused": True,
            "camera_prim_path": palm_prim_path,
            "render_product_path": render_product_path,
            "resolution_px": list(RESOLUTION_PX),
            "annotators": list(ANNOTATORS_EXACTLY),
            "warmup_frames": WARMUP_FRAMES,
            "rt_subframes": RT_SUBFRAMES,
            "render_delta_time_s": 0.0,
            "finite_positive_depth_pixels": int(np.count_nonzero(valid_depth)),
            "rgb_dtype": str(rgb.dtype),
            "depth_dtype": str(depth.dtype),
        }

        inputs = _central_valid_depth_roi(depth)
        result["input_derivation"] = inputs["diagnostics"]
        evaluator = evaluate_postgrasp_palm_keyed_visual_control(
            scene_schema_version=scene_schema_version,
            scene_profile_id=scene_profile_id,
            fixed_orientation_token=fixed_orientation_token,
            keyed_model_id=keyed_model_id,
            camera_contract={
                "camera_name": "Palm",
                "parent_frame": "handbase_link",
                "resolution_px": list(RESOLUTION_PX),
                "channels_exactly": list(ANNOTATORS_EXACTLY),
                "near_clip_m": 0.02,
                "T_HC_source": "FROZEN_SCENE_CONFIG",
                "camera_contract_id": _VISUAL.EXPECTED_CAMERA_CONTRACT_ID,
            },
            rgb=rgb,
            depth_m=depth,
            connector_face_mask=inputs["connector_face_mask"],
            face_center_uv=inputs["face_center_uv"],
            occlusion_mask=inputs["occlusion_mask"],
            image_mask_quality=None,
            camera_intrinsics=_camera_intrinsics_contract(),
            c2_candidates=(),
            pose_quality=None,
            accuracy_gate=None,
            yaw_acceptance=None,
            actual_arm_q_before_capture=q_before,
            actual_arm_q_after_capture=q_after,
            T_HC_frozen_configured=t_hc,
            T_WH_from_actual_q=t_wh,
            T_WR_fixed_configured=t_wr,
            T_RP_target_configured=t_rp,
        )
        summary = _evaluator_summary(evaluator)
        result["visual_evaluator"] = summary
        if summary.get("rejection_code") != "KEY_REGION_OCCLUSION_UNKNOWN":
            raise RuntimeError(
                "first-stage evaluator did not reject unknown occlusion exactly"
            )
        forbidden_true = (
            "observation_passed",
            "plan_authorized",
            "simulation_prealign_target_authorized",
            "control_authorized",
            "simulation_prealign_control_authorized",
            "simulation_insertion_control_authorized",
            "hardware_control_authorized",
        )
        if any(summary.get(name) is not False for name in forbidden_true):
            raise RuntimeError("unknown-occlusion evaluator authorization relaxed")
        result.update(
            status="CAPTURED_EVALUATOR_SAFE_STOP",
            reason="occlusion remains unknown in first-stage integrated capture",
            rejection_code="KEY_REGION_OCCLUSION_UNKNOWN",
        )
    except Exception as exc:  # noqa: BLE001 - runtime must return safe structure
        message = f"{type(exc).__name__}: {exc}"
        code = (
            "CAPTURE_Q_DRIFT_ABOVE_LIMIT"
            if "CAPTURE_Q_DRIFT_ABOVE_LIMIT" in str(exc)
            else "CAPTURE_RUNTIME_EXCEPTION"
        )
        result = _abort(result, code, message)
    finally:
        cleanup = _cleanup_resources(
            annotators, render_product, render_product_path
        )
        result["resource_cleanup"] = cleanup
        if cleanup["resources_released"] is not True:
            result = _abort(
                result,
                "CAPTURE_RESOURCE_CLEANUP_FAILED",
                "; ".join(cleanup["errors"]),
            )
        if was_playing is not None and timeline_was_paused:
            try:
                timeline = _restore_timeline(
                    world=world,
                    simulation_app=simulation_app,
                    was_playing=was_playing,
                )
                result["timeline"] = timeline
                if timeline["restored"] is not True:
                    result = _abort(
                        result,
                        "TIMELINE_RESTORE_FAILED",
                        "timeline state was not restored after Palm capture",
                    )
            except Exception as exc:  # pragma: no cover - Isaac boundary
                result = _abort(
                    result,
                    "TIMELINE_RESTORE_EXCEPTION",
                    f"{type(exc).__name__}: {exc}",
                )
        if output_dir_created and output_dir is not None and output_dir.is_dir():
            try:
                _write_report(output_dir, result)
            except Exception as exc:  # pragma: no cover - filesystem boundary
                result = _abort(
                    result,
                    "REPORT_WRITE_FAILED",
                    f"{type(exc).__name__}: {exc}",
                )
    return result


__all__ = [
    "ANNOTATORS_EXACTLY",
    "DEPTH_FILENAME",
    "MAXIMUM_CAPTURE_Q_DRIFT_RAD",
    "MODE",
    "PALM_PRIM_SUFFIX",
    "REPORT_FILENAME",
    "RESOLUTION_PX",
    "RGB_FILENAME",
    "SCHEMA_VERSION",
    "run_postgrasp_palm_keyed_visual_capture",
]
