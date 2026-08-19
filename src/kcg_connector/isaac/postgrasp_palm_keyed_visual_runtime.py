"""Single fail-closed Palm RGB-D observation and control-planning gate.

This module deliberately stops at a pure target-pose plan.  It never commands
the robot and it never accepts a simulator object pose, collision result, or
an episode-specific camera pose.  The selected ``T_HP`` is produced only from
the frozen hand-camera transform and the image-derived ``T_CP`` C2 estimate.

Transform notation follows ``T_AB`` = pose of frame B expressed in frame A.
The configured prealignment target is therefore::

    T_WP_target = T_WR_configured @ T_RP_target_configured
    T_WH_target = T_WP_target @ inverse(T_HP_selected)

The returned ``plan_authorized`` field is scoped to simulation visual
prealignment target planning.  ``control_authorized`` stays false until a
downstream collision-checked path gate accepts the target; insertion and
hardware control stay disabled here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

import numpy as np

from kcg_connector.d38999_key_yaw_acceptance import (
    DEFAULT_SIMULATION_MINIMUM_SAMPLES,
    NUMERIC_EQUALITY_TOLERANCE_DEG,
    SIMULATION_SCHEMA_VERSION as YAW_ACCEPTANCE_SCHEMA_VERSION,
    SIMULATION_THRESHOLD_LABEL as YAW_ACCEPTANCE_THRESHOLD_LABEL,
)
from kcg_connector.d38999_key_yaw_benchmark import (
    EXPECTED_C2_IDS,
    refine_keyed_axial_yaw_from_rgbd,
)
from kcg_connector.d38999_keyed_public_spec_v2 import (
    PAIR_MODEL_ID,
    PLUG_MODEL_ID,
    load_keyed_public_spec_v2,
)


SCHEMA_VERSION = "kcg_d38999_postgrasp_palm_keyed_visual_control_v1"
MODE = "PALM_RGBD_KEYED_OBSERVATION_AND_SIM_PREALIGN_PLAN"

EXPECTED_SCENE_SCHEMA_VERSION = "kcg_d38999_keyed_v2_tabletop_scene_v1"
EXPECTED_SCENE_PROFILE_ID = PAIR_MODEL_ID
EXPECTED_FIXED_ORIENTATION_TOKEN = (
    "MATING_FACE_UP_RX_180_FOR_DOWNWARD_INSERTION"
)
EXPECTED_FIXED_ROTATION = np.diag((1.0, -1.0, -1.0))

EXPECTED_CAMERA_NAME = "Palm"
EXPECTED_CAMERA_PARENT_FRAME = "handbase_link"
EXPECTED_RESOLUTION_PX = (1280, 720)
EXPECTED_CHANNELS = ("rgb", "distance_to_image_plane")
EXPECTED_NEAR_CLIP_M = 0.02
EXPECTED_T_HC_SOURCE = "FROZEN_SCENE_CONFIG"
EXPECTED_CAMERA_CONTRACT_ID = "PALM_1280X720_RGBD_0P02_FIXED_T_HC_V1"

MASK_QUALITY_SCHEMA_VERSION = "kcg_palm_rgb_image_mask_quality_v1"
MASK_SOURCE = "PALM_RGB_IMAGE_ONLY"
POSE_QUALITY_SCHEMA_VERSION = "kcg_palm_rgbd_c2_pose_quality_v1"
POSE_SOURCE = "PALM_RGBD_C2_ESTIMATOR"
ACCURACY_GATE_SCHEMA_VERSION = "kcg_palm_keyed_visual_accuracy_gate_v1"

MAXIMUM_CAPTURE_Q_DRIFT_RAD = 0.002
KEY_AXIS_PROJECTION_LENGTH_M = 0.010


def _base_result(
    scene_schema_version: Any,
    scene_profile_id: Any,
    keyed_model_id: Any,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": MODE,
        "status": "REJECTED_SAFE_STOP",
        "reason": None,
        "rejection_code": None,
        "observation_passed": False,
        "plan_authorized": False,
        "simulation_prealign_target_authorized": False,
        "continue_to_path_planner_authorized": False,
        "control_authorized": False,
        "simulation_prealign_control_authorized": False,
        "simulation_insertion_control_authorized": False,
        "hardware_control_authorized": False,
        "safe_stop_required": True,
        "scene_schema_version": (
            scene_schema_version if isinstance(scene_schema_version, str) else None
        ),
        "scene_profile_id": (
            scene_profile_id if isinstance(scene_profile_id, str) else None
        ),
        "keyed_model_id": keyed_model_id if isinstance(keyed_model_id, str) else None,
        "selected_hypothesis_id": None,
        "estimated_axial_yaw_rad": None,
        "T_HP_selected": None,
        "T_WP_target": None,
        "T_WH_target": None,
        "target_plan": None,
        "keyed_yaw_observation": None,
        "keyed_branch_projections": None,
        "capture_q_drift_max_rad": None,
        "gate_diagnostics": {},
        "actuator_command_issued": False,
        "truth_correction_applied": False,
    }


def _reject(
    base: Mapping[str, Any],
    code: str,
    reason: str,
    *,
    observation_passed: bool = False,
    **diagnostics: Any,
) -> dict[str, Any]:
    result = dict(base)
    result.update(
        status=(
            "OBSERVATION_ONLY_CONTROL_REJECTED_SAFE_STOP"
            if observation_passed
            else "REJECTED_SAFE_STOP"
        ),
        reason=reason,
        rejection_code=code,
        observation_passed=observation_passed,
        gate_diagnostics=diagnostics,
    )
    return result


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


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


def _arm_q(value: Any, label: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if result.shape != (7,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be a finite 7-vector")
    return result


def _camera_contract_error(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return "CAMERA_CONTRACT_MISSING"
    expected_fields = {
        "camera_name",
        "parent_frame",
        "resolution_px",
        "channels_exactly",
        "near_clip_m",
        "T_HC_source",
        "camera_contract_id",
    }
    if set(value) != expected_fields:
        return "CAMERA_CONTRACT_FIELDS_NOT_EXACT"
    if value["camera_name"] != EXPECTED_CAMERA_NAME:
        return "NOT_PALM_CAMERA"
    if value["parent_frame"] != EXPECTED_CAMERA_PARENT_FRAME:
        return "PALM_CAMERA_PARENT_NOT_HANDBASE"
    try:
        resolution = tuple(int(item) for item in value["resolution_px"])
    except (TypeError, ValueError):
        return "CAMERA_RESOLUTION_INVALID"
    if resolution != EXPECTED_RESOLUTION_PX:
        return "CAMERA_RESOLUTION_NOT_1280X720"
    try:
        channels = tuple(value["channels_exactly"])
    except TypeError:
        return "CAMERA_CHANNELS_INVALID"
    if channels != EXPECTED_CHANNELS:
        return "CAMERA_CHANNELS_NOT_EXACT_RGB_DISTANCE"
    try:
        near_clip = _finite_number(value["near_clip_m"], "near_clip_m")
    except ValueError:
        return "CAMERA_NEAR_CLIP_INVALID"
    if not math.isclose(near_clip, EXPECTED_NEAR_CLIP_M, abs_tol=1.0e-12):
        return "CAMERA_NEAR_CLIP_NOT_0P02_M"
    if value["T_HC_source"] != EXPECTED_T_HC_SOURCE:
        return "T_HC_NOT_FROZEN_SCENE_CONFIG"
    if value["camera_contract_id"] != EXPECTED_CAMERA_CONTRACT_ID:
        return "PALM_CAMERA_CONTRACT_ID_MISMATCH"
    return None


def _mask_quality_error(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return "IMAGE_MASK_QUALITY_MISSING"
    expected_fields = {
        "schema_version",
        "source",
        "status",
        "passed",
        "confidence_calibrated",
        "observed_confidence",
        "minimum_confidence",
        "camera_contract_id",
    }
    if set(value) != expected_fields:
        return "IMAGE_MASK_QUALITY_FIELDS_NOT_EXACT"
    if value["schema_version"] != MASK_QUALITY_SCHEMA_VERSION:
        return "IMAGE_MASK_QUALITY_SCHEMA_MISMATCH"
    if value["source"] != MASK_SOURCE:
        return "FACE_MASK_NOT_FROM_PALM_RGB_ONLY"
    if value["camera_contract_id"] != EXPECTED_CAMERA_CONTRACT_ID:
        return "IMAGE_MASK_CAMERA_CONTRACT_MISMATCH"
    if value["status"] != "ACCEPTED" or value["passed"] is not True:
        return "IMAGE_MASK_QUALITY_REJECTED"
    if value["confidence_calibrated"] is not True:
        return "IMAGE_MASK_CONFIDENCE_UNCALIBRATED"
    try:
        observed = _finite_number(
            value["observed_confidence"], "observed_confidence"
        )
        minimum = _finite_number(
            value["minimum_confidence"], "minimum_confidence"
        )
    except ValueError:
        return "IMAGE_MASK_CONFIDENCE_INVALID"
    if not (0.0 <= minimum <= 1.0 and 0.0 <= observed <= 1.0):
        return "IMAGE_MASK_CONFIDENCE_OUT_OF_RANGE"
    if observed < minimum:
        return "IMAGE_MASK_CONFIDENCE_BELOW_LIMIT"
    return None


def _pose_quality_error(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return "POSE_QUALITY_MISSING"
    expected_fields = {
        "schema_version",
        "source",
        "status",
        "pose_valid",
        "quality_gate_passed",
        "covariance_calibration_status",
        "calibration_id",
        "scene_profile_id",
        "camera_contract_id",
    }
    if set(value) != expected_fields:
        return "POSE_QUALITY_FIELDS_NOT_EXACT"
    if value["schema_version"] != POSE_QUALITY_SCHEMA_VERSION:
        return "POSE_QUALITY_SCHEMA_MISMATCH"
    if value["source"] != POSE_SOURCE:
        return "POSE_NOT_FROM_PALM_RGBD_C2_ESTIMATOR"
    if value["scene_profile_id"] != EXPECTED_SCENE_PROFILE_ID:
        return "POSE_QUALITY_SCENE_PROFILE_MISMATCH"
    if value["camera_contract_id"] != EXPECTED_CAMERA_CONTRACT_ID:
        return "POSE_QUALITY_CAMERA_CONTRACT_MISMATCH"
    if value["status"] != "ACCEPTED":
        return "POSE_QUALITY_REJECTED"
    if value["pose_valid"] is not True or value["quality_gate_passed"] is not True:
        return "POSE_QUALITY_GATE_NOT_PASSED"
    if value["covariance_calibration_status"] != "CALIBRATED":
        return "POSE_COVARIANCE_UNCALIBRATED"
    calibration_id = value["calibration_id"]
    if not isinstance(calibration_id, str) or not calibration_id.strip():
        return "POSE_CALIBRATION_ID_MISSING"
    return None


def _yaw_acceptance_error(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return "PUBLIC_SPEC_YAW_P95_GATE_MISSING"
    model = load_keyed_public_spec_v2()
    checks = (
        (value.get("schema_version") == YAW_ACCEPTANCE_SCHEMA_VERSION,
         "PUBLIC_SPEC_YAW_P95_SCHEMA_MISMATCH"),
        (value.get("mode") == "OFFLINE_WITHHELD_SYNTHETIC_YAW_EVALUATION_ONLY",
         "PUBLIC_SPEC_YAW_P95_MODE_MISMATCH"),
        (value.get("status") == "PASSED_PUBLIC_SPEC_SIMULATION_SHADOW_ONLY",
         "PUBLIC_SPEC_YAW_P95_STATUS_NOT_PASSED"),
        (value.get("keyed_model_id") == PLUG_MODEL_ID,
         "PUBLIC_SPEC_YAW_P95_MODEL_MISMATCH"),
        (value.get("profile_name") == model.simulation_acceptance_profile,
         "PUBLIC_SPEC_YAW_P95_PROFILE_MISMATCH"),
        (value.get("threshold_label") == YAW_ACCEPTANCE_THRESHOLD_LABEL,
         "PUBLIC_SPEC_YAW_P95_THRESHOLD_LABEL_MISMATCH"),
        (value.get("withheld_truth") is True,
         "PUBLIC_SPEC_YAW_P95_NOT_WITHHELD"),
        (value.get("passed") is True and value.get("shadow_authorized") is True,
         "PUBLIC_SPEC_YAW_P95_GATE_NOT_PASSED"),
        (value.get("selected_for_control_allowed") is False,
         "UPSTREAM_YAW_GATE_AUTHORIZATION_BOUNDARY_RELAXED"),
        (value.get("simulation_insertion_control_authorized") is False,
         "UPSTREAM_INSERTION_AUTHORIZATION_BOUNDARY_RELAXED"),
        (value.get("robot_control_authorized") is False,
         "UPSTREAM_ROBOT_AUTHORIZATION_BOUNDARY_RELAXED"),
        (value.get("hardware_control_authorized") is False,
         "UPSTREAM_HARDWARE_AUTHORIZATION_BOUNDARY_RELAXED"),
        (value.get("drawing_specified_mechanical_yaw_clearance") is False,
         "PUBLIC_SPEC_YAW_CLEARANCE_SCOPE_MISMATCH"),
        (value.get("real_measured_clearance_deg") is None,
         "PUBLIC_SPEC_GATE_CANNOT_CLAIM_MEASURED_CLEARANCE"),
    )
    for passed, code in checks:
        if not passed:
            return code
    sample_count = value.get("sample_count")
    if isinstance(sample_count, bool) or not isinstance(sample_count, (int, np.integer)):
        return "PUBLIC_SPEC_YAW_SAMPLE_COUNT_INVALID"
    if int(sample_count) < DEFAULT_SIMULATION_MINIMUM_SAMPLES:
        return "PUBLIC_SPEC_YAW_SAMPLE_COUNT_INSUFFICIENT"
    dataset_tag = value.get("dataset_tag")
    if not isinstance(dataset_tag, str) or not dataset_tag.strip():
        return "PUBLIC_SPEC_YAW_DATASET_TAG_MISSING"
    try:
        observed = _finite_number(
            value.get("observed_yaw_error_p95_deg"), "observed_yaw_error_p95_deg"
        )
        required = _finite_number(
            value.get("required_yaw_error_p95_deg"), "required_yaw_error_p95_deg"
        )
    except ValueError:
        return "PUBLIC_SPEC_YAW_P95_NUMERIC_VALUE_INVALID"
    if observed < 0.0 or required <= 0.0:
        return "PUBLIC_SPEC_YAW_P95_VALUE_OUT_OF_RANGE"
    if not observed < required - NUMERIC_EQUALITY_TOLERANCE_DEG:
        return "PUBLIC_SPEC_YAW_P95_NOT_STRICTLY_BELOW_HALF_GAP"
    return None


def _accuracy_gate_error(
    value: Any,
    yaw_acceptance: Mapping[str, Any] | None,
) -> str | None:
    """Bind a calibrated live-input gate to the exact offline p95 artifact."""
    if not isinstance(value, Mapping):
        return "SAME_CAMERA_ACCURACY_GATE_MISSING"
    expected_fields = {
        "schema_version",
        "status",
        "accuracy_gate_passed",
        "camera_contract_id",
        "scene_profile_id",
        "keyed_model_id",
        "calibration_id",
        "yaw_dataset_tag",
    }
    if set(value) != expected_fields:
        return "SAME_CAMERA_ACCURACY_GATE_FIELDS_NOT_EXACT"
    if value["schema_version"] != ACCURACY_GATE_SCHEMA_VERSION:
        return "SAME_CAMERA_ACCURACY_GATE_SCHEMA_MISMATCH"
    if value["status"] != "ACCEPTED" or value["accuracy_gate_passed"] is not True:
        return "SAME_CAMERA_ACCURACY_GATE_NOT_PASSED"
    if value["camera_contract_id"] != EXPECTED_CAMERA_CONTRACT_ID:
        return "SAME_CAMERA_ACCURACY_GATE_CAMERA_MISMATCH"
    if value["scene_profile_id"] != EXPECTED_SCENE_PROFILE_ID:
        return "SAME_CAMERA_ACCURACY_GATE_SCENE_MISMATCH"
    if value["keyed_model_id"] != PLUG_MODEL_ID:
        return "SAME_CAMERA_ACCURACY_GATE_MODEL_MISMATCH"
    calibration_id = value["calibration_id"]
    if not isinstance(calibration_id, str) or not calibration_id.strip():
        return "SAME_CAMERA_ACCURACY_GATE_CALIBRATION_ID_MISSING"
    dataset_tag = value["yaw_dataset_tag"]
    if not isinstance(dataset_tag, str) or not dataset_tag.strip():
        return "SAME_CAMERA_ACCURACY_GATE_DATASET_TAG_MISSING"
    if not isinstance(yaw_acceptance, Mapping):
        return "SAME_CAMERA_ACCURACY_GATE_P95_REPORT_MISSING"
    if yaw_acceptance.get("dataset_tag") != dataset_tag:
        return "SAME_CAMERA_ACCURACY_GATE_P95_DATASET_MISMATCH"
    return None


def _image_arrays(
    rgb: Any,
    depth_m: Any,
    connector_face_mask: Any,
    occlusion_mask: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rgb_array = np.asarray(rgb)
    depth = np.asarray(depth_m)
    face = np.asarray(connector_face_mask)
    occlusion = np.asarray(occlusion_mask)
    height, width = EXPECTED_RESOLUTION_PX[1], EXPECTED_RESOLUTION_PX[0]
    if rgb_array.shape != (height, width, 3) or rgb_array.dtype != np.uint8:
        raise ValueError("rgb must be uint8 with shape (720, 1280, 3)")
    if depth.shape != (height, width) or not np.issubdtype(depth.dtype, np.number):
        raise ValueError("depth_m must be numeric with shape (720, 1280)")
    if face.shape != (height, width) or face.dtype != np.bool_:
        raise ValueError("connector_face_mask must be bool with shape (720, 1280)")
    if occlusion.shape != (height, width) or occlusion.dtype != np.bool_:
        raise ValueError("occlusion_mask must be bool with shape (720, 1280)")
    if int(np.count_nonzero(face)) == 0:
        raise ValueError("connector_face_mask must not be empty")
    return rgb_array, depth.astype(np.float64, copy=False), face, occlusion


def _rz(angle_rad: float) -> np.ndarray:
    cosine, sine = math.cos(angle_rad), math.sin(angle_rad)
    return np.asarray(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )


def _project_uv(point: np.ndarray, intrinsics: Mapping[str, Any]) -> np.ndarray:
    if point[2] <= 1.0e-9:
        raise ValueError("candidate projection lies behind Palm camera")
    return np.asarray(
        (
            float(intrinsics["fx"]) * point[0] / point[2]
            + float(intrinsics["cx"]),
            float(intrinsics["fy"]) * point[1] / point[2]
            + float(intrinsics["cy"]),
        ),
        dtype=np.float64,
    )


def _keyed_branch_projections(
    c2_candidates: Sequence[Mapping[str, Any]],
    camera_intrinsics: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not isinstance(c2_candidates, Sequence) or isinstance(
        c2_candidates, (str, bytes)
    ):
        raise ValueError("c2_candidates must be a sequence")
    records: list[dict[str, Any]] = []
    for candidate in c2_candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError("each C2 candidate must be a mapping")
        if set(candidate) != {"hypothesis_id", "T_camera_plug", "axial_yaw_rad"}:
            raise ValueError("each C2 candidate must contain only strict pose fields")
        hypothesis_id = candidate["hypothesis_id"]
        if hypothesis_id not in EXPECTED_C2_IDS:
            raise ValueError("C2 hypothesis ID must be YAW_0 or YAW_PI")
        transform = _transform(candidate["T_camera_plug"], "T_camera_plug")
        center = transform[:3, 3]
        endpoint = center + KEY_AXIS_PROJECTION_LENGTH_M * transform[:3, 0]
        center_uv = _project_uv(center, camera_intrinsics)
        endpoint_uv = _project_uv(endpoint, camera_intrinsics)
        direction = endpoint_uv - center_uv
        norm = float(np.linalg.norm(direction))
        if norm <= 1.0e-9:
            raise ValueError("keyed +X branch projection is degenerate")
        direction /= norm
        records.append(
            {
                "hypothesis_id": hypothesis_id,
                "source_axis": "KEYED_PLUG_FRAME_PLUS_X",
                "center_uv": center_uv.tolist(),
                "direction_uv": direction.tolist(),
            }
        )
    records.sort(key=lambda item: EXPECTED_C2_IDS.index(item["hypothesis_id"]))
    if tuple(item["hypothesis_id"] for item in records) != EXPECTED_C2_IDS:
        raise ValueError("C2 candidates must contain exactly YAW_0 and YAW_PI")
    dot = float(
        np.dot(records[0]["direction_uv"], records[1]["direction_uv"])
    )
    if dot > -math.cos(math.radians(2.0)):
        raise ValueError("projected keyed +X C2 directions are not antipodal")
    return records


def evaluate_postgrasp_palm_keyed_visual_control(
    *,
    scene_schema_version: str,
    scene_profile_id: str,
    fixed_orientation_token: str,
    keyed_model_id: str,
    camera_contract: Mapping[str, Any],
    rgb: Any,
    depth_m: Any,
    connector_face_mask: Any,
    face_center_uv: Sequence[float],
    occlusion_mask: Any,
    image_mask_quality: Mapping[str, Any],
    camera_intrinsics: Mapping[str, Any],
    c2_candidates: Sequence[Mapping[str, Any]],
    pose_quality: Mapping[str, Any],
    accuracy_gate: Mapping[str, Any] | None,
    yaw_acceptance: Mapping[str, Any] | None,
    actual_arm_q_before_capture: Sequence[float],
    actual_arm_q_after_capture: Sequence[float],
    T_HC_frozen_configured: Any,
    T_WH_from_actual_q: Any,
    T_WR_fixed_configured: Any,
    T_RP_target_configured: Any,
) -> dict[str, Any]:
    """Observe keyed yaw and gate a simulation prealignment target atomically.

    Malformed or unaccepted runtime evidence returns a structured safe stop.
    The only offline truth-derived input is the aggregate public-spec yaw-p95
    acceptance report; no per-frame truth is accepted by this API.
    """

    base = _base_result(scene_schema_version, scene_profile_id, keyed_model_id)
    if scene_schema_version != EXPECTED_SCENE_SCHEMA_VERSION:
        return _reject(
            base, "KEYED_SCENE_SCHEMA_MISMATCH", "exact keyed-v2 scene required"
        )
    if scene_profile_id != EXPECTED_SCENE_PROFILE_ID:
        return _reject(
            base, "KEYED_SCENE_PROFILE_MISMATCH", "exact keyed-v2 profile required"
        )
    if fixed_orientation_token != EXPECTED_FIXED_ORIENTATION_TOKEN:
        return _reject(
            base,
            "FIXED_ORIENTATION_CONTRACT_MISMATCH",
            "fixed receptacle orientation is not the keyed-v2 downward-insertion config",
        )
    if keyed_model_id != PLUG_MODEL_ID:
        return _reject(
            base, "KEYED_MODEL_ID_MISMATCH", "exact public-spec keyed plug required"
        )
    camera_error = _camera_contract_error(camera_contract)
    if camera_error is not None:
        return _reject(base, camera_error, "Palm capture contract rejected")
    if occlusion_mask is None:
        return _reject(
            base,
            "KEY_REGION_OCCLUSION_UNKNOWN",
            "Palm occlusion mask is required before keyed observation",
        )

    try:
        _, depth, face, occlusion = _image_arrays(
            rgb, depth_m, connector_face_mask, occlusion_mask
        )
        q_before = _arm_q(actual_arm_q_before_capture, "actual q before capture")
        q_after = _arm_q(actual_arm_q_after_capture, "actual q after capture")
        t_hc = _transform(T_HC_frozen_configured, "T_HC_frozen_configured")
        t_wh = _transform(T_WH_from_actual_q, "T_WH_from_actual_q")
        t_wr = _transform(T_WR_fixed_configured, "T_WR_fixed_configured")
        t_rp_target = _transform(T_RP_target_configured, "T_RP_target_configured")
    except ValueError as exc:
        return _reject(base, "RUNTIME_INPUT_INVALID", str(exc))

    if not np.allclose(t_wr[:3, :3], EXPECTED_FIXED_ROTATION, atol=1.0e-6):
        return _reject(
            base,
            "FIXED_CONFIG_ROTATION_MISMATCH",
            "configured fixed receptacle must use Rx=180 degrees",
        )
    q_drift = float(np.max(np.abs(q_after - q_before)))
    base["capture_q_drift_max_rad"] = q_drift
    if q_drift > MAXIMUM_CAPTURE_Q_DRIFT_RAD:
        return _reject(
            base,
            "CAPTURE_Q_DRIFT_ABOVE_LIMIT",
            "robot moved while Palm RGB-D was captured",
            observed_q_drift_rad=q_drift,
            maximum_q_drift_rad=MAXIMUM_CAPTURE_Q_DRIFT_RAD,
        )

    try:
        projections = _keyed_branch_projections(c2_candidates, camera_intrinsics)
        keyed_yaw = refine_keyed_axial_yaw_from_rgbd(
            depth_m=depth,
            connector_face_mask=face,
            face_center_uv=face_center_uv,
            camera_intrinsics=camera_intrinsics,
            c2_candidates=c2_candidates,
            keyed_model_id=keyed_model_id,
            occlusion_mask=occlusion,
        )
    except (KeyError, TypeError, ValueError) as exc:
        return _reject(base, "C2_OR_IMAGE_CONTRACT_INVALID", str(exc))
    base["keyed_branch_projections"] = projections
    base["keyed_yaw_observation"] = keyed_yaw
    if keyed_yaw.get("passed") is not True:
        return _reject(
            base,
            str(keyed_yaw.get("rejection_code") or "KEYED_YAW_OBSERVATION_REJECTED"),
            str(keyed_yaw.get("reason") or "Palm keyed-yaw observation rejected"),
            keyed_yaw_status=keyed_yaw.get("status"),
        )

    selected_id = keyed_yaw.get("selected_hypothesis_id")
    estimated_yaw = keyed_yaw.get("estimated_axial_yaw_rad")
    if selected_id not in EXPECTED_C2_IDS or not isinstance(estimated_yaw, Real):
        return _reject(
            base,
            "KEYED_YAW_SUCCESS_CONTRACT_INVALID",
            "passed keyed-yaw result lacks a finite strict C2 selection",
            observation_passed=True,
        )
    estimated_yaw = float(estimated_yaw)
    if not math.isfinite(estimated_yaw):
        return _reject(
            base,
            "KEYED_YAW_SUCCESS_CONTRACT_INVALID",
            "passed keyed-yaw result contains non-finite yaw",
            observation_passed=True,
        )

    mask_error = _mask_quality_error(image_mask_quality)
    pose_error = _pose_quality_error(pose_quality)
    accuracy_error = _accuracy_gate_error(accuracy_gate, yaw_acceptance)
    yaw_gate_error = _yaw_acceptance_error(yaw_acceptance)
    gate_errors = [
        error
        for error in (mask_error, pose_error, yaw_gate_error, accuracy_error)
        if error is not None
    ]
    if gate_errors:
        base["selected_hypothesis_id"] = selected_id
        base["estimated_axial_yaw_rad"] = estimated_yaw
        return _reject(
            base,
            gate_errors[0],
            "Palm observation succeeded but calibrated control gates are incomplete",
            observation_passed=True,
            failed_control_gates=gate_errors,
        )

    selected_candidates = [
        item for item in c2_candidates if item.get("hypothesis_id") == selected_id
    ]
    if len(selected_candidates) != 1:
        return _reject(
            base,
            "SELECTED_C2_CANDIDATE_NOT_UNIQUE",
            "selected strict C2 candidate is not unique",
            observation_passed=True,
        )
    try:
        t_cp = _transform(
            selected_candidates[0]["T_camera_plug"], "selected T_camera_plug"
        ).copy()
        candidate_yaw = _finite_number(
            selected_candidates[0]["axial_yaw_rad"], "selected axial_yaw_rad"
        )
        correction = (estimated_yaw - candidate_yaw + math.pi) % (
            2.0 * math.pi
        ) - math.pi
        t_cp[:3, :3] = t_cp[:3, :3] @ _rz(correction)
        t_hp = _transform(t_hc @ t_cp, "T_HP_selected")
        t_wp_observed = _transform(t_wh @ t_hp, "T_WP_observed")
        t_wp_target = _transform(t_wr @ t_rp_target, "T_WP_target")
        t_wh_target = _transform(
            t_wp_target @ np.linalg.inv(t_hp), "T_WH_target"
        )
    except (KeyError, ValueError, np.linalg.LinAlgError) as exc:
        return _reject(
            base,
            "VISUAL_TARGET_PLAN_INVALID",
            str(exc),
            observation_passed=True,
        )

    plan = {
        "scope": "SIMULATION_VISUAL_PREALIGN_TARGET_POSE_ONLY",
        "scene_profile_id": scene_profile_id,
        "selected_hypothesis_id": selected_id,
        "T_HP_selected": t_hp.tolist(),
        "T_WP_observed": t_wp_observed.tolist(),
        "T_WP_target": t_wp_target.tolist(),
        "T_WH_target": t_wh_target.tolist(),
        "requires_collision_checked_path_planner": True,
        "plan_authorized": True,
        "control_authorized": False,
        "insertion_motion_included": False,
        "truth_correction_applied": False,
        "actuator_command_issued": False,
    }
    result = dict(base)
    result.update(
        status="VISUAL_OBSERVATION_AND_CONTROL_PLAN_ACCEPTED",
        reason=None,
        rejection_code=None,
        observation_passed=True,
        plan_authorized=True,
        simulation_prealign_target_authorized=True,
        continue_to_path_planner_authorized=True,
        control_authorized=False,
        simulation_prealign_control_authorized=False,
        safe_stop_required=False,
        selected_hypothesis_id=selected_id,
        estimated_axial_yaw_rad=estimated_yaw,
        T_HP_selected=t_hp.tolist(),
        T_WP_target=t_wp_target.tolist(),
        T_WH_target=t_wh_target.tolist(),
        target_plan=plan,
        gate_diagnostics={
            "image_mask_quality_gate": "PASSED_CALIBRATED",
            "pose_quality_gate": "PASSED_CALIBRATED",
            "same_camera_accuracy_gate": "PASSED",
            "public_spec_yaw_p95_gate": "PASSED",
            "capture_q_drift_gate": "PASSED",
        },
    )
    return result


__all__ = [
    "ACCURACY_GATE_SCHEMA_VERSION",
    "EXPECTED_CAMERA_CONTRACT_ID",
    "EXPECTED_CAMERA_NAME",
    "EXPECTED_CHANNELS",
    "EXPECTED_FIXED_ORIENTATION_TOKEN",
    "EXPECTED_NEAR_CLIP_M",
    "EXPECTED_RESOLUTION_PX",
    "EXPECTED_SCENE_PROFILE_ID",
    "EXPECTED_SCENE_SCHEMA_VERSION",
    "MASK_QUALITY_SCHEMA_VERSION",
    "MAXIMUM_CAPTURE_Q_DRIFT_RAD",
    "POSE_QUALITY_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "evaluate_postgrasp_palm_keyed_visual_control",
]
