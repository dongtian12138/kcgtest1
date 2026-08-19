"""Pure-CPU continuous keyed-yaw refinement and local benchmark metrics.

Inference consumes only image-derived RGB-D observations, calibrated camera
intrinsics, and two estimated C2 pose candidates.  It fits the observed face
plane, intersects the centre/master-key pixel rays with that plane, and uses
the resulting 3-D tangent direction to select and refine one C2 branch.  There
is deliberately no truth, semantic, contact, or collider input in the
inference API.

Caller-presented synthetic truth is consumed only by the separate metric
helper at the end of an experiment.  Rejected VISIBLE_VALID samples receive a
180-degree error, preventing selective rejection from improving p95.  Both
the observed p95 and a deterministic bootstrap one-sided 95% upper bound must
pass for ``metric_gates_passed``.  Same-account files and timestamps cannot
prove withholding, so the top-level benchmark remains blocked until an
OS-isolated evaluator exists; it never authorizes shadow or robot control.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np

from kcg_connector.d38999_key_yaw_acceptance import (
    DEFAULT_SIMULATION_MINIMUM_SAMPLES,
    NUMERIC_EQUALITY_TOLERANCE_DEG,
    evaluate_public_spec_sim_key_yaw_acceptance,
)
from kcg_connector.d38999_keyed_public_spec_v2 import PLUG_MODEL_ID
from kcg_connector.d38999_key_region_detector import (
    SCHEMA_VERSION as KEY_DETECTOR_SCHEMA_VERSION,
    detect_key_region_from_palm_rgbd,
)


SCHEMA_VERSION = "kcg_d38999_key_yaw_refinement_v1"
BENCHMARK_SCHEMA_VERSION = "kcg_d38999_key_yaw_benchmark_v1"
MODE = "PALM_RGBD_C2_CONTINUOUS_AXIAL_YAW_SHADOW_ONLY"
BENCHMARK_MODE = "OFFLINE_LOCAL_SYNTHETIC_YAW_METRICS_FORMAL_WITHHELD_BLOCKED"
EXPECTED_C2_IDS = ("YAW_0", "YAW_PI")
DEFAULT_STRESS_PROFILE = "adversarial_gdt_stress"
MINIMUM_FACE_PIXELS = 200
MINIMUM_VALID_FACE_DEPTH_FRACTION = 0.98
MAXIMUM_PLANE_RMSE_M = 0.0010
MAXIMUM_CANDIDATE_NORMAL_ERROR_DEG = 10.0
MINIMUM_BRANCH_MARGIN_DEG = 2.0
MINIMUM_KEY_RING_RADIUS_RATIO = 0.60
MAXIMUM_KEY_RING_RADIUS_RATIO = 1.15
MAXIMUM_C2_TRANSLATION_DISAGREEMENT_M = 0.002
MAXIMUM_C2_NORMAL_DISAGREEMENT_DEG = 2.0
MAXIMUM_C2_ANTIPODAL_ERROR_DEG = 2.0
MAXIMUM_C2_YAW_CONSISTENCY_ERROR_DEG = 0.10
DEFAULT_MINIMUM_VISIBLE_YIELD = 0.99
DEFAULT_MINIMUM_STRATUM_SAMPLES = 30
DEFAULT_MINIMUM_MUST_REJECT_SAMPLES = 1024
DEFAULT_MINIMUM_MUST_REJECT_PER_CLASS = 256
REQUIRED_REJECTION_CLASSES = (
    "OCCLUDED",
    "OUT_OF_FRAME",
    "DEPTH_MISSING",
    "LOW_CONFIDENCE",
)
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260816
LOCAL_MANIFEST_SCHEMA_VERSION = "kcg_d38999_key_yaw_local_manifest_v1"
CLAIMED_REVEAL_SCHEMA_VERSION = "kcg_d38999_key_yaw_claimed_reveal_v1"
LOCAL_CONSISTENCY_CHECK = "LOCAL_SELF_CONSISTENCY_AND_ACCIDENTAL_MUTATION_CHECK"
FORMAL_WITHHELD_BLOCKED_STATUS = "BLOCKED_REQUIRES_OS_ISOLATED_EVALUATOR"
REQUIRED_STRATUM_AXES = ("yaw", "light", "pose")
REQUIRED_DISTINCT_YAW_VALUES = 64
MINIMUM_DECLARED_LIGHT_STRATA = 2
MINIMUM_DECLARED_POSE_STRATA = 2
MAXIMUM_CANDIDATE_FACE_CENTER_ERROR_M = 0.002
PROJECT_STRESS_INTERPRETATION = (
    "PROJECT_ADVERSARIAL_GDT_STRESS_ASSUMPTION_NOT_DRAWING_SPECIFIED_CLEARANCE"
)


def _wrap_rad(value: float | np.ndarray) -> float | np.ndarray:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _bootstrap_p95_upper_bound_deg(errors_deg: np.ndarray, seed: int) -> float | None:
    """Return the deterministic bootstrap one-sided 95% upper p95 bound."""
    if errors_deg.size == 0 or not np.all(np.isfinite(errors_deg)):
        return None
    rng = np.random.default_rng(seed)
    bootstrapped_p95 = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    batch_size = 128
    for start in range(0, BOOTSTRAP_REPLICATES, batch_size):
        stop = min(start + batch_size, BOOTSTRAP_REPLICATES)
        indices = rng.integers(
            0,
            errors_deg.size,
            size=(stop - start, errors_deg.size),
        )
        bootstrapped_p95[start:stop] = np.percentile(
            errors_deg[indices], 95.0, axis=1
        )
    ordered = np.sort(bootstrapped_p95)
    upper_index = min(
        ordered.size - 1,
        max(0, math.ceil(0.95 * ordered.size) - 1),
    )
    return float(ordered[upper_index])


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _boolean_image(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape (H, W)")
    if array.dtype != np.bool_:
        raise ValueError(f"{name} must be a boolean image")
    return array


def _numeric_image(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape (H, W)")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be numeric")
    return array.astype(np.float64, copy=False)


def _uv(value: Any, label: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if result.shape != (2,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be a finite UV pair")
    return result


def _intrinsics(value: Any, shape: tuple[int, int]) -> dict[str, float | int]:
    if not isinstance(value, Mapping):
        raise ValueError("camera_intrinsics must be a mapping")
    expected = {"fx", "fy", "cx", "cy", "width", "height"}
    if set(value) != expected:
        raise ValueError(
            "camera_intrinsics must contain exactly fx/fy/cx/cy/width/height"
        )
    fx = _finite_number(value["fx"], "camera_intrinsics.fx")
    fy = _finite_number(value["fy"], "camera_intrinsics.fy")
    cx = _finite_number(value["cx"], "camera_intrinsics.cx")
    cy = _finite_number(value["cy"], "camera_intrinsics.cy")
    width = value["width"]
    height = value["height"]
    if isinstance(width, bool) or not isinstance(width, (int, np.integer)):
        raise ValueError("camera_intrinsics.width must be an integer")
    if isinstance(height, bool) or not isinstance(height, (int, np.integer)):
        raise ValueError("camera_intrinsics.height must be an integer")
    width, height = int(width), int(height)
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("camera focal lengths must be positive")
    if (height, width) != shape:
        raise ValueError("camera dimensions must match depth_m")
    if not (0.0 <= cx < width and 0.0 <= cy < height):
        raise ValueError("camera principal point must lie inside the image")
    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "width": width, "height": height}


def _candidate_records(value: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("c2_candidates must contain exactly two records")
    if len(value) != 2:
        raise ValueError("c2_candidates must contain exactly two records")
    parsed = []
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise ValueError(f"c2_candidates[{index}] must be a mapping")
        expected = {"hypothesis_id", "T_camera_plug", "axial_yaw_rad"}
        if set(raw) != expected:
            raise ValueError(
                f"c2_candidates[{index}] must contain exactly "
                "hypothesis_id/T_camera_plug/axial_yaw_rad"
            )
        hypothesis_id = raw["hypothesis_id"]
        if hypothesis_id not in EXPECTED_C2_IDS:
            raise ValueError(f"c2_candidates[{index}] has an invalid hypothesis_id")
        try:
            transform = np.asarray(raw["T_camera_plug"], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"c2_candidates[{index}] pose must be numeric") from exc
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise ValueError(f"c2_candidates[{index}] pose must be finite 4x4")
        if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-9):
            raise ValueError(f"c2_candidates[{index}] pose last row is invalid")
        rotation = transform[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-6):
            raise ValueError(f"c2_candidates[{index}] rotation is not orthonormal")
        if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1.0e-6):
            raise ValueError(f"c2_candidates[{index}] rotation determinant is not +1")
        parsed.append(
            {
                "hypothesis_id": hypothesis_id,
                "T_camera_plug": transform,
                "axial_yaw_rad": _finite_number(
                    raw["axial_yaw_rad"], f"c2_candidates[{index}].axial_yaw_rad"
                ),
            }
        )
    parsed.sort(key=lambda item: EXPECTED_C2_IDS.index(item["hypothesis_id"]))
    if {item["hypothesis_id"] for item in parsed} != set(EXPECTED_C2_IDS):
        raise ValueError("c2_candidates must contain one YAW_0 and one YAW_PI")
    first, second = parsed
    translation_difference = float(
        np.linalg.norm(
            first["T_camera_plug"][:3, 3] - second["T_camera_plug"][:3, 3]
        )
    )
    if translation_difference > MAXIMUM_C2_TRANSLATION_DISAGREEMENT_M:
        raise ValueError("C2 candidate translations do not share a 5-DOF solution")
    first_normal = first["T_camera_plug"][:3, 2]
    second_normal = second["T_camera_plug"][:3, 2]
    normal_angle = math.degrees(
        math.acos(float(np.clip(np.dot(first_normal, second_normal), -1.0, 1.0)))
    )
    if normal_angle > MAXIMUM_C2_NORMAL_DISAGREEMENT_DEG:
        raise ValueError("C2 candidate face normals do not agree")
    common_normal = first_normal + second_normal
    common_normal /= np.linalg.norm(common_normal)
    plane_x_axes = []
    for item in (first, second):
        axis = item["T_camera_plug"][:3, 0].copy()
        axis -= common_normal * float(np.dot(axis, common_normal))
        axis /= np.linalg.norm(axis)
        plane_x_axes.append(axis)
    x_dot = float(np.clip(np.dot(*plane_x_axes), -1.0, 1.0))
    actual_x_separation_deg = math.degrees(math.acos(x_dot))
    if abs(180.0 - actual_x_separation_deg) > MAXIMUM_C2_ANTIPODAL_ERROR_DEG:
        raise ValueError("C2 candidate in-plane +X axes must be antipodal")
    yaw_difference = float(
        _wrap_rad(second["axial_yaw_rad"] - first["axial_yaw_rad"])
    )
    if not math.isclose(abs(yaw_difference), math.pi, abs_tol=1.0e-3):
        raise ValueError("C2 candidate axial yaw values must differ by pi")
    transform_difference = math.atan2(
        float(np.dot(common_normal, np.cross(plane_x_axes[0], plane_x_axes[1]))),
        x_dot,
    )
    consistency_error_deg = abs(
        math.degrees(float(_wrap_rad(transform_difference - yaw_difference)))
    )
    if consistency_error_deg > MAXIMUM_C2_YAW_CONSISTENCY_ERROR_DEG:
        raise ValueError("C2 transform rotation disagrees with axial_yaw difference")
    return first, second


def _base_result(keyed_model_id: Any) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": MODE,
        "status": "REJECTED",
        "reason": None,
        "rejection_code": None,
        "passed": False,
        "shadow_only": True,
        "shadow_authorized": False,
        "control_authorized": False,
        "selected_for_control_allowed": False,
        "keyed_model_id": keyed_model_id if isinstance(keyed_model_id, str) else None,
        "selected_hypothesis_id": None,
        "estimated_axial_yaw_rad": None,
        "quality_diagnostics": {},
    }


def _reject(
    base: dict[str, Any],
    status: str,
    reason: str,
    rejection_code: str,
    **diagnostics: Any,
) -> dict[str, Any]:
    result = dict(base)
    result.update(
        status=status,
        reason=reason,
        rejection_code=rejection_code,
        quality_diagnostics=diagnostics,
    )
    return result


def _fit_face_plane(
    depth: np.ndarray,
    face: np.ndarray,
    occlusion: np.ndarray,
    camera: Mapping[str, float | int],
) -> tuple[np.ndarray, float, dict[str, Any]] | tuple[None, None, dict[str, Any]]:
    face_pixels = int(np.count_nonzero(face))
    valid_depth = np.isfinite(depth) & (depth > 0.0)
    valid_face = face & valid_depth & ~occlusion
    valid_pixels = int(np.count_nonzero(valid_face))
    valid_fraction = float(valid_pixels / face_pixels) if face_pixels else 0.0
    diagnostics: dict[str, Any] = {
        "face_pixels": face_pixels,
        "valid_face_depth_pixels": valid_pixels,
        "valid_face_depth_fraction": valid_fraction,
        "occluded_face_pixels": int(np.count_nonzero(face & occlusion)),
    }
    if face_pixels < MINIMUM_FACE_PIXELS:
        diagnostics["plane_failure"] = "FACE_SUPPORT_TOO_SMALL"
        return None, None, diagnostics
    if valid_fraction < MINIMUM_VALID_FACE_DEPTH_FRACTION:
        diagnostics["plane_failure"] = "FACE_DEPTH_INCOMPLETE"
        return None, None, diagnostics
    rows, columns = np.nonzero(valid_face)
    z = depth[valid_face]
    x = (columns.astype(np.float64) - float(camera["cx"])) * z / float(camera["fx"])
    y = (rows.astype(np.float64) - float(camera["cy"])) * z / float(camera["fy"])
    points = np.column_stack((x, y, z))
    # Deterministic decimation bounds CPU/memory without changing the image
    # information source.
    if points.shape[0] > 30000:
        indices = np.linspace(0, points.shape[0] - 1, 30000).astype(np.int64)
        points = points[indices]

    def fit(values: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        centroid = np.mean(values, axis=0)
        covariance = (values - centroid).T @ (values - centroid) / values.shape[0]
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        normal = eigenvectors[:, int(np.argmin(eigenvalues))]
        normal /= np.linalg.norm(normal)
        offset = -float(np.dot(normal, centroid))
        return normal, offset, centroid

    normal, offset, _ = fit(points)
    residuals = np.abs(points @ normal + offset)
    median = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median)))
    inlier_threshold = max(0.00020, median + 4.0 * 1.4826 * mad)
    inliers = residuals <= inlier_threshold
    if int(np.count_nonzero(inliers)) < MINIMUM_FACE_PIXELS:
        diagnostics["plane_failure"] = "PLANE_INLIER_SUPPORT_TOO_SMALL"
        return None, None, diagnostics
    normal, offset, centroid = fit(points[inliers])
    final_residuals = points[inliers] @ normal + offset
    rmse = float(np.sqrt(np.mean(final_residuals**2)))
    diagnostics.update(
        {
            "plane_input_points": int(points.shape[0]),
            "plane_inlier_points": int(np.count_nonzero(inliers)),
            "plane_inlier_fraction": float(np.mean(inliers)),
            "plane_inlier_threshold_m": inlier_threshold,
            "plane_rmse_m": rmse,
            "plane_centroid_camera_m": centroid.tolist(),
        }
    )
    if not math.isfinite(rmse) or rmse > MAXIMUM_PLANE_RMSE_M:
        diagnostics["plane_failure"] = "PLANE_RMSE_ABOVE_LIMIT"
        return None, None, diagnostics
    return normal, offset, diagnostics


def _ray_plane_point(
    uv: np.ndarray,
    camera: Mapping[str, float | int],
    normal: np.ndarray,
    offset: float,
) -> np.ndarray | None:
    ray = np.asarray(
        (
            (uv[0] - float(camera["cx"])) / float(camera["fx"]),
            (uv[1] - float(camera["cy"])) / float(camera["fy"]),
            1.0,
        ),
        dtype=np.float64,
    )
    denominator = float(np.dot(normal, ray))
    if abs(denominator) <= 1.0e-9:
        return None
    scale = -offset / denominator
    if not math.isfinite(scale) or scale <= 0.0:
        return None
    return scale * ray


def refine_keyed_axial_yaw_from_rgbd(
    depth_m: Any,
    connector_face_mask: Any,
    face_center_uv: Sequence[float],
    camera_intrinsics: Mapping[str, Any],
    c2_candidates: Sequence[Mapping[str, Any]],
    keyed_model_id: str | None,
    *,
    occlusion_mask: Any | None,
) -> dict[str, Any]:
    """Select/refine one C2 pose using an observed 3-D face tangent.

    ``T_camera_plug`` maps plug-frame points into the optical camera frame.
    ``axial_yaw_rad`` is the candidate's estimated plug-frame Rz coordinate,
    not truth.  The master-key centroid is derived internally from the exact
    five-key/N-pattern detector output; callers cannot inject a direction.
    """
    depth = _numeric_image("depth_m", depth_m)
    face = _boolean_image("connector_face_mask", connector_face_mask)
    if face.shape != depth.shape:
        raise ValueError("connector_face_mask shape must match depth_m")
    camera = _intrinsics(camera_intrinsics, depth.shape)
    center = _uv(face_center_uv, "face_center_uv")
    candidates = _candidate_records(c2_candidates)
    base = _base_result(keyed_model_id)

    if keyed_model_id != PLUG_MODEL_ID:
        return _reject(
            base,
            "MODEL_NOT_KEYED_V2",
            "KEYED_MODEL_ID_MISSING_OR_UNREGISTERED",
            "KEYED_MODEL_ID_UNAVAILABLE",
        )
    if occlusion_mask is None:
        return _reject(
            base,
            "OCCLUSION_UNKNOWN",
            "OCCLUSION_MASK_MISSING",
            "KEY_REGION_OCCLUSION_UNKNOWN",
        )
    occlusion = _boolean_image("occlusion_mask", occlusion_mask)
    if occlusion.shape != face.shape:
        raise ValueError("occlusion_mask shape must match depth_m")

    height, width = depth.shape
    if not (0.0 <= center[0] < width and 0.0 <= center[1] < height):
        return _reject(
            base,
            "OUT_OF_FRAME",
            "FACE_CENTER_OUTSIDE_IMAGE",
            "CONNECTOR_FACE_OUT_OF_FRAME",
        )
    center_pixel = (
        min(height - 1, max(0, int(round(center[1])))),
        min(width - 1, max(0, int(round(center[0])))),
    )
    if not face[center_pixel]:
        return _reject(
            base,
            "LOW_CONFIDENCE",
            "FACE_CENTER_OUTSIDE_FACE_MASK",
            "KEY_REGION_LOW_CONFIDENCE",
        )
    if np.any(face[0, :]) or np.any(face[-1, :]) or np.any(face[:, 0]) or np.any(face[:, -1]):
        return _reject(
            base,
            "OUT_OF_FRAME",
            "CONNECTOR_FACE_TOUCHES_IMAGE_BORDER",
            "CONNECTOR_FACE_OUT_OF_FRAME",
        )
    if np.any(face & occlusion):
        return _reject(
            base,
            "OCCLUDED",
            "OCCLUSION_INTERSECTS_CONNECTOR_FACE",
            "KEY_REGION_OCCLUDED",
            occluded_face_pixels=int(np.count_nonzero(face & occlusion)),
        )

    detector = detect_key_region_from_palm_rgbd(
        face,
        depth,
        center,
        keyed_model_id,
        occlusion_mask=occlusion,
    )
    if detector.get("passed") is not True:
        detector_status = detector.get("status")
        detector_reason = detector.get("reason")
        detector_code = detector.get("rejection_code")
        return _reject(
            base,
            detector_status if isinstance(detector_status, str) else "LOW_CONFIDENCE",
            detector_reason if isinstance(detector_reason, str) else "KEY_DETECTOR_REJECTED",
            detector_code
            if isinstance(detector_code, str)
            else "KEY_REGION_LOW_CONFIDENCE",
            key_detector_schema_version=detector.get("schema_version"),
            key_detector_status=detector_status,
        )
    detector_quality = detector.get("quality_diagnostics")
    detector_probability = np.asarray(detector.get("key_probability"))
    detector_direction = _uv(detector.get("key_direction_uv"), "detector.key_direction_uv")
    detector_contract_ok = bool(
        detector.get("schema_version") == KEY_DETECTOR_SCHEMA_VERSION
        and detector.get("status") == "KEY_DIRECTION_DETECTED_SHADOW_ONLY"
        and detector.get("passed") is True
        and detector.get("shadow_only") is True
        and detector.get("control_authorized") is False
        and detector.get("keyed_model_id") == keyed_model_id
        and isinstance(detector_quality, Mapping)
        and detector_quality.get("candidate_count") == 5
    )
    if not detector_contract_ok:
        return _reject(
            base,
            "LOW_CONFIDENCE",
            "KEY_DETECTOR_SUCCESS_CONTRACT_MISMATCH",
            "KEY_REGION_LOW_CONFIDENCE",
        )
    detection_confidence = _finite_number(
        detector_quality.get("detection_confidence"),
        "detector.detection_confidence",
    )
    minimum_detection_confidence = _finite_number(
        detector_quality.get("minimum_detection_confidence"),
        "detector.minimum_detection_confidence",
    )
    pattern_error_deg = _finite_number(
        detector_quality.get("maximum_minor_pattern_angle_error_deg"),
        "detector.maximum_minor_pattern_angle_error_deg",
    )
    maximum_pattern_error_deg = _finite_number(
        detector_quality.get("maximum_allowed_minor_pattern_angle_error_deg"),
        "detector.maximum_allowed_minor_pattern_angle_error_deg",
    )
    if (
        detection_confidence < minimum_detection_confidence
        or pattern_error_deg > maximum_pattern_error_deg
        or detector_probability.shape != face.shape
        or not np.issubdtype(detector_probability.dtype, np.number)
        or not np.all(np.isfinite(detector_probability))
        or np.any(detector_probability < 0.0)
        or np.any(detector_probability > 1.0)
    ):
        return _reject(
            base,
            "LOW_CONFIDENCE",
            "KEY_DETECTOR_EVIDENCE_BELOW_CONTRACT",
            "KEY_REGION_LOW_CONFIDENCE",
        )
    key_support = detector_probability > 0.0
    key_rows, key_columns = np.nonzero(key_support)
    if key_rows.size == 0:
        return _reject(
            base,
            "LOW_CONFIDENCE",
            "KEY_DETECTOR_PROBABILITY_SUPPORT_EMPTY",
            "KEY_REGION_LOW_CONFIDENCE",
        )
    key_weights = detector_probability[key_support].astype(np.float64)
    key_uv = np.asarray(
        (
            np.average(key_columns, weights=key_weights),
            np.average(key_rows, weights=key_weights),
        ),
        dtype=np.float64,
    )
    key_source = "EXACT_FIVE_KEY_N_PATTERN_DETECTOR_PROBABILITY_CENTROID"
    if not (0.0 <= key_uv[0] < width and 0.0 <= key_uv[1] < height):
        return _reject(
            base,
            "OUT_OF_FRAME",
            "KEY_OBSERVATION_OUTSIDE_IMAGE",
            "KEY_REGION_OUT_OF_FRAME",
        )
    key_offset_uv = key_uv - center
    key_radius_px = float(np.linalg.norm(key_offset_uv))
    if key_radius_px <= 1.0e-12:
        return _reject(
            base,
            "LOW_CONFIDENCE",
            "KEY_OBSERVATION_AT_FACE_CENTER",
            "KEY_REGION_LOW_CONFIDENCE",
        )
    radial_direction = key_offset_uv / key_radius_px
    detector_direction_norm = float(np.linalg.norm(detector_direction))
    if detector_direction_norm <= 1.0e-12:
        return _reject(
            base,
            "LOW_CONFIDENCE",
            "KEY_DETECTOR_DIRECTION_DEGENERATE",
            "KEY_REGION_LOW_CONFIDENCE",
        )
    detector_direction /= detector_direction_norm
    direction_coherence_deg = math.degrees(
        math.acos(
            float(np.clip(np.dot(radial_direction, detector_direction), -1.0, 1.0))
        )
    )
    if direction_coherence_deg > 5.0:
        return _reject(
            base,
            "LOW_CONFIDENCE",
            "KEY_CENTROID_AND_DIRECTION_DISAGREE",
            "KEY_REGION_LOW_CONFIDENCE",
            key_centroid_direction_error_deg=direction_coherence_deg,
        )
    face_rows, face_columns = np.nonzero(face)
    face_offsets = np.column_stack(
        (face_columns.astype(np.float64) - center[0], face_rows.astype(np.float64) - center[1])
    )
    along = face_offsets @ radial_direction
    lateral_direction = np.asarray((-radial_direction[1], radial_direction[0]))
    lateral = np.abs(face_offsets @ lateral_direction)
    equivalent_radius_px = math.sqrt(float(face_rows.size) / math.pi)
    ray_band_px = max(2.0, 0.15 * equivalent_radius_px)
    directional_support = along[(along > 0.0) & (lateral <= ray_band_px)]
    if directional_support.size == 0:
        return _reject(
            base,
            "LOW_CONFIDENCE",
            "FACE_MASK_HAS_NO_SUPPORT_TOWARD_KEY",
            "KEY_REGION_LOW_CONFIDENCE",
        )
    directional_face_radius_px = float(np.max(directional_support))
    key_ring_radius_ratio = key_radius_px / directional_face_radius_px
    key_pixel = (
        min(height - 1, max(0, int(round(key_uv[1])))),
        min(width - 1, max(0, int(round(key_uv[0])))),
    )
    if occlusion[key_pixel]:
        return _reject(
            base,
            "OCCLUDED",
            "OCCLUSION_INTERSECTS_KEY_OBSERVATION",
            "KEY_REGION_OCCLUDED",
        )
    key_row, key_column = key_pixel
    valid_depth = np.isfinite(depth) & (depth > 0.0)
    local_rows = slice(max(0, key_row - 1), min(height, key_row + 2))
    local_columns = slice(max(0, key_column - 1), min(width, key_column + 2))
    local_face = face[local_rows, local_columns]
    local_valid_depth = valid_depth[local_rows, local_columns]
    key_support_depth_complete = bool(np.all(valid_depth[key_support]))
    local_key_depth_complete = bool(
        valid_depth[key_pixel]
        and np.count_nonzero(local_face) > 0
        and np.all(local_valid_depth[local_face])
    )
    if not key_support_depth_complete or not local_key_depth_complete:
        return _reject(
            base,
            "DEPTH_MISSING",
            "KEY_PIXEL_OR_LOCAL_RING_DEPTH_INCOMPLETE",
            "KEY_REGION_DEPTH_MISSING",
            key_pixel_uv=[key_column, key_row],
            key_support_depth_complete=key_support_depth_complete,
            local_key_depth_complete=local_key_depth_complete,
        )
    local_face_support = bool(
        np.any(
            face[
                max(0, key_row - 1) : min(height, key_row + 2),
                max(0, key_column - 1) : min(width, key_column + 2),
            ]
        )
    )
    if (
        key_ring_radius_ratio < MINIMUM_KEY_RING_RADIUS_RATIO
        or key_ring_radius_ratio > MAXIMUM_KEY_RING_RADIUS_RATIO
        or not local_face_support
    ):
        return _reject(
            base,
            "LOW_CONFIDENCE",
            "KEY_OBSERVATION_OUTSIDE_FACE_KEY_RING",
            "KEY_REGION_LOW_CONFIDENCE",
            key_radius_px=key_radius_px,
            directional_face_radius_px=directional_face_radius_px,
            key_ring_radius_ratio=key_ring_radius_ratio,
            minimum_key_ring_radius_ratio=MINIMUM_KEY_RING_RADIUS_RATIO,
            maximum_key_ring_radius_ratio=MAXIMUM_KEY_RING_RADIUS_RATIO,
            local_face_support=local_face_support,
        )

    fitted_normal, plane_offset, plane_diagnostics = _fit_face_plane(
        depth, face, occlusion, camera
    )
    if fitted_normal is None or plane_offset is None:
        failure = plane_diagnostics.get("plane_failure")
        code = (
            "KEY_REGION_DEPTH_MISSING"
            if failure == "FACE_DEPTH_INCOMPLETE"
            else "KEY_REGION_LOW_CONFIDENCE"
        )
        return _reject(
            base,
            "DEPTH_MISSING" if code == "KEY_REGION_DEPTH_MISSING" else "LOW_CONFIDENCE",
            str(failure),
            code,
            **plane_diagnostics,
        )

    candidate_normal = np.mean(
        [item["T_camera_plug"][:3, 2] for item in candidates], axis=0
    )
    candidate_normal /= np.linalg.norm(candidate_normal)
    if float(np.dot(fitted_normal, candidate_normal)) < 0.0:
        fitted_normal = -fitted_normal
        plane_offset = -plane_offset
    normal_error_deg = math.degrees(
        math.acos(
            float(np.clip(np.dot(fitted_normal, candidate_normal), -1.0, 1.0))
        )
    )
    diagnostics = {
        **plane_diagnostics,
        "fitted_face_normal_camera": fitted_normal.tolist(),
        "candidate_face_normal_camera": candidate_normal.tolist(),
        "candidate_to_fitted_normal_error_deg": normal_error_deg,
        "maximum_candidate_normal_error_deg": MAXIMUM_CANDIDATE_NORMAL_ERROR_DEG,
        "key_observation_source": key_source,
        "face_center_uv": center.tolist(),
        "key_observation_uv": key_uv.tolist(),
        "key_radius_px": key_radius_px,
        "directional_face_radius_px": directional_face_radius_px,
        "key_ring_radius_ratio": key_ring_radius_ratio,
        "minimum_key_ring_radius_ratio": MINIMUM_KEY_RING_RADIUS_RATIO,
        "maximum_key_ring_radius_ratio": MAXIMUM_KEY_RING_RADIUS_RATIO,
        "local_face_support": local_face_support,
        "key_detector_schema_version": detector.get("schema_version"),
        "key_detector_status": detector.get("status"),
        "key_detector_candidate_count": detector_quality["candidate_count"],
        "key_detector_confidence": detection_confidence,
        "key_detector_minimum_confidence": minimum_detection_confidence,
        "key_detector_n_pattern_error_deg": pattern_error_deg,
        "key_detector_maximum_n_pattern_error_deg": maximum_pattern_error_deg,
        "key_detector_support_pixels": int(key_rows.size),
        "key_centroid_direction_error_deg": direction_coherence_deg,
        "key_support_depth_complete": key_support_depth_complete,
        "local_key_depth_complete": local_key_depth_complete,
    }
    if normal_error_deg > MAXIMUM_CANDIDATE_NORMAL_ERROR_DEG:
        return _reject(
            base,
            "LOW_CONFIDENCE",
            "C2_CANDIDATE_NORMAL_DISAGREES_WITH_DEPTH_PLANE",
            "KEY_REGION_LOW_CONFIDENCE",
            **diagnostics,
        )

    center_3d = _ray_plane_point(center, camera, fitted_normal, plane_offset)
    key_3d = _ray_plane_point(key_uv, camera, fitted_normal, plane_offset)
    if center_3d is None or key_3d is None:
        return _reject(
            base,
            "LOW_CONFIDENCE",
            "PIXEL_RAY_PLANE_INTERSECTION_FAILED",
            "KEY_DIRECTION_DEGENERATE",
            **diagnostics,
        )
    candidate_center_errors_m = [
        float(np.linalg.norm(item["T_camera_plug"][:3, 3] - center_3d))
        for item in candidates
    ]
    diagnostics.update(
        {
            "observed_face_center_camera_m": center_3d.tolist(),
            "candidate_face_center_errors_m": candidate_center_errors_m,
            "maximum_candidate_face_center_error_m": (
                MAXIMUM_CANDIDATE_FACE_CENTER_ERROR_M
            ),
        }
    )
    if max(candidate_center_errors_m) > MAXIMUM_CANDIDATE_FACE_CENTER_ERROR_M:
        return _reject(
            base,
            "LOW_CONFIDENCE",
            "C2_TRANSLATION_DISAGREES_WITH_OBSERVED_FACE_CENTER",
            "KEY_REGION_LOW_CONFIDENCE",
            **diagnostics,
        )
    observed = key_3d - center_3d
    observed -= fitted_normal * float(np.dot(observed, fitted_normal))
    observed_norm = float(np.linalg.norm(observed))
    if observed_norm <= 1.0e-9:
        return _reject(
            base,
            "LOW_CONFIDENCE",
            "OBSERVED_KEY_TANGENT_DEGENERATE",
            "KEY_DIRECTION_DEGENERATE",
            **diagnostics,
        )
    observed /= observed_norm

    branch_records = []
    for item in candidates:
        x_axis = item["T_camera_plug"][:3, 0].copy()
        x_axis -= fitted_normal * float(np.dot(x_axis, fitted_normal))
        x_norm = float(np.linalg.norm(x_axis))
        if x_norm <= 1.0e-9:
            return _reject(
                base,
                "LOW_CONFIDENCE",
                "C2_KEY_AXIS_DEGENERATE_IN_FACE_PLANE",
                "KEY_DIRECTION_DEGENERATE",
                **diagnostics,
            )
        x_axis /= x_norm
        y_axis = np.cross(fitted_normal, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        correction = math.atan2(
            float(np.dot(observed, y_axis)),
            float(np.dot(observed, x_axis)),
        )
        branch_records.append(
            {
                "hypothesis_id": item["hypothesis_id"],
                "signed_yaw_correction_rad": correction,
                "absolute_yaw_correction_deg": abs(math.degrees(correction)),
                "candidate_key_axis_camera": x_axis.tolist(),
            }
        )
    branch_records.sort(
        key=lambda item: (item["absolute_yaw_correction_deg"], item["hypothesis_id"])
    )
    best, second = branch_records
    margin_deg = float(
        second["absolute_yaw_correction_deg"] - best["absolute_yaw_correction_deg"]
    )
    diagnostics.update(
        {
            "face_center_camera_m": center_3d.tolist(),
            "key_observation_camera_m": key_3d.tolist(),
            "observed_key_tangent_camera": observed.tolist(),
            "c2_branch_diagnostics": branch_records,
            "best_branch_margin_deg": margin_deg,
            "minimum_branch_margin_deg": MINIMUM_BRANCH_MARGIN_DEG,
        }
    )
    if margin_deg < MINIMUM_BRANCH_MARGIN_DEG:
        return _reject(
            base,
            "AMBIGUOUS",
            "C2_BRANCH_MARGIN_BELOW_LIMIT",
            "KEY_BRANCH_AMBIGUOUS",
            **diagnostics,
        )

    selected = next(
        item for item in candidates if item["hypothesis_id"] == best["hypothesis_id"]
    )
    correction = float(best["signed_yaw_correction_rad"])
    estimated_yaw = float(_wrap_rad(selected["axial_yaw_rad"] + correction))
    diagnostics.update(
        {
            "selected_candidate_axial_yaw_rad": float(selected["axial_yaw_rad"]),
            "signed_yaw_correction_rad": correction,
            "signed_yaw_correction_deg": math.degrees(correction),
            "estimated_axial_yaw_wrapped_rad": estimated_yaw,
            "yaw_definition": "C2_CANDIDATE_AXIAL_RZ_PLUS_FACE_PLANE_KEY_CORRECTION",
        }
    )
    result = dict(base)
    result.update(
        {
            "status": "CONTINUOUS_KEYED_YAW_ESTIMATED_SHADOW_ONLY",
            "passed": True,
            "selected_hypothesis_id": selected["hypothesis_id"],
            "estimated_axial_yaw_rad": estimated_yaw,
            "quality_diagnostics": diagnostics,
        }
    )
    return result


def _prediction_records(value: Any) -> dict[str, Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("predictions must be a sequence")
    records: dict[str, Mapping[str, Any]] = {}
    required = {
        "sample_id",
        "passed",
        "estimated_axial_yaw_rad",
        "selected_hypothesis_id",
        "shadow_only",
        "control_authorized",
    }
    for index, item in enumerate(value):
        if not isinstance(item, Mapping) or set(item) != required:
            raise ValueError(
                f"predictions[{index}] must contain exactly the local prediction fields"
            )
        sample_id = item["sample_id"]
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"predictions[{index}].sample_id is invalid")
        if sample_id in records:
            raise ValueError(f"duplicate prediction sample_id: {sample_id}")
        if type(item["passed"]) is not bool:
            raise ValueError(f"predictions[{index}].passed must be boolean")
        if item["shadow_only"] is not True or item["control_authorized"] is not False:
            raise ValueError("prediction authorization boundary was relaxed")
        estimate = item["estimated_axial_yaw_rad"]
        if item["passed"]:
            _finite_number(estimate, f"predictions[{index}].estimated_axial_yaw_rad")
            if item["selected_hypothesis_id"] not in EXPECTED_C2_IDS:
                raise ValueError(f"predictions[{index}] passed without a C2 hypothesis")
        elif estimate is not None or item["selected_hypothesis_id"] is not None:
            raise ValueError(
                f"predictions[{index}] rejected but retained a yaw/branch estimate"
            )
        records[sample_id] = item
    return records


def _truth_records(value: Any) -> dict[str, Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("withheld_truth_records must be a sequence")
    records: dict[str, Mapping[str, Any]] = {}
    required = {
        "sample_id",
        "expected_outcome",
        "axial_yaw_truth_rad",
        "expected_hypothesis_id",
        "strata",
    }
    for index, item in enumerate(value):
        if not isinstance(item, Mapping) or not required.issubset(item):
            raise ValueError(f"withheld_truth_records[{index}] is missing required fields")
        sample_id = item["sample_id"]
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"withheld_truth_records[{index}].sample_id is invalid")
        if sample_id in records:
            raise ValueError(f"duplicate withheld truth sample_id: {sample_id}")
        outcome = item["expected_outcome"]
        if outcome not in {"VISIBLE_VALID", "MUST_REJECT"}:
            raise ValueError(f"withheld_truth_records[{index}] outcome is invalid")
        if not isinstance(item["strata"], Mapping):
            raise ValueError(f"withheld_truth_records[{index}].strata must be a mapping")
        if outcome == "VISIBLE_VALID":
            if set(item["strata"]) != set(REQUIRED_STRATUM_AXES):
                raise ValueError(
                    f"withheld_truth_records[{index}] strata must be yaw/light/pose"
                )
            _finite_number(
                item["axial_yaw_truth_rad"],
                f"withheld_truth_records[{index}].axial_yaw_truth_rad",
            )
            if item["expected_hypothesis_id"] not in EXPECTED_C2_IDS:
                raise ValueError(
                    f"withheld_truth_records[{index}] has invalid expected hypothesis"
                )
        else:
            rejection_class = item.get("rejection_class")
            if rejection_class not in REQUIRED_REJECTION_CLASSES:
                raise ValueError(
                    f"withheld_truth_records[{index}] has invalid rejection_class"
                )
            if (
                item["axial_yaw_truth_rad"] is not None
                or item["expected_hypothesis_id"] is not None
            ):
                raise ValueError(
                    f"withheld_truth_records[{index}] MUST_REJECT yaw/branch must be null"
                )
        records[sample_id] = item
    return records


def _utc_timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{label} must be an ISO UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{label} is invalid") from exc
    if parsed.utcoffset() != timedelta(0):
        raise ValueError(f"{label} must be UTC")
    return parsed


def _declared_coverage(
    yaw_values_rad: Any,
    strata: Any,
) -> tuple[list[float], dict[str, list[str]]]:
    if not isinstance(yaw_values_rad, Sequence) or isinstance(
        yaw_values_rad, (str, bytes)
    ):
        raise ValueError("declared_yaw_values_rad must be a sequence")
    yaw_values = sorted(
        float(_wrap_rad(_finite_number(value, "declared yaw value")))
        for value in yaw_values_rad
    )
    if len(yaw_values) != REQUIRED_DISTINCT_YAW_VALUES or len(
        {round(value, 12) for value in yaw_values}
    ) != REQUIRED_DISTINCT_YAW_VALUES:
        raise ValueError("exactly 64 distinct yaw values must be declared")
    if not isinstance(strata, Mapping) or set(strata) != set(REQUIRED_STRATUM_AXES):
        raise ValueError("declared_strata must contain exactly yaw/light/pose")
    normalized: dict[str, list[str]] = {}
    for axis in REQUIRED_STRATUM_AXES:
        labels = strata[axis]
        if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)):
            raise ValueError(f"declared_strata.{axis} must be a sequence")
        values = list(labels)
        if any(not isinstance(label, str) or not label for label in values):
            raise ValueError(f"declared_strata.{axis} labels must be non-empty text")
        if len(values) != len(set(values)):
            raise ValueError(f"declared_strata.{axis} labels must be unique")
        normalized[axis] = sorted(values)
    if len(normalized["yaw"]) != REQUIRED_DISTINCT_YAW_VALUES:
        raise ValueError("exactly 64 yaw strata must be declared")
    if len(normalized["light"]) < MINIMUM_DECLARED_LIGHT_STRATA:
        raise ValueError("at least two light strata must be declared")
    if len(normalized["pose"]) < MINIMUM_DECLARED_POSE_STRATA:
        raise ValueError("at least two pose strata must be declared")
    return yaw_values, normalized


def write_local_key_yaw_prediction_artifact(
    predictions: Sequence[Mapping[str, Any]],
    *,
    prediction_artifact_path: Path | str,
    prediction_manifest_path: Path | str,
    dataset_tag: str,
    run_id: str,
    declared_yaw_values_rad: Sequence[float],
    declared_strata: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Write a non-overwriting local artifact; this is not a trust boundary."""
    records = _prediction_records(predictions)
    if not isinstance(dataset_tag, str) or not dataset_tag.strip():
        raise ValueError("dataset_tag must be non-empty text")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be non-empty text")
    yaw_values, strata = _declared_coverage(
        declared_yaw_values_rad, declared_strata
    )
    artifact = Path(prediction_artifact_path).expanduser().resolve()
    manifest_path = Path(prediction_manifest_path).expanduser().resolve()
    if artifact == manifest_path:
        raise ValueError("prediction artifact and manifest paths must differ")
    if not artifact.parent.is_dir() or not manifest_path.parent.is_dir():
        raise ValueError("prediction artifact and manifest parent directories must exist")
    if artifact.exists() or manifest_path.exists():
        raise FileExistsError("local prediction paths are non-overwritable")
    ordered_records = [records[sample_id] for sample_id in sorted(records)]
    with artifact.open("x", encoding="utf-8") as stream:
        for record in ordered_records:
            serializable = dict(record)
            if serializable["estimated_axial_yaw_rad"] is not None:
                serializable["estimated_axial_yaw_rad"] = float(
                    serializable["estimated_axial_yaw_rad"]
                )
            stream.write(json.dumps(serializable, sort_keys=True) + "\n")
    artifact_stat = artifact.stat()
    completed_at = datetime.now(timezone.utc).isoformat(
        timespec="microseconds"
    ).replace("+00:00", "Z")
    manifest = {
        "schema_version": LOCAL_MANIFEST_SCHEMA_VERSION,
        "status": "LOCAL_PREDICTION_ARTIFACT_COMPLETE",
        "run_id": run_id.strip(),
        "prediction_manifest_id": f"{run_id.strip()}:prediction-manifest",
        "dataset_tag": dataset_tag.strip(),
        "prediction_artifact_path": str(artifact),
        "prediction_manifest_path": str(manifest_path),
        "prediction_record_count": len(ordered_records),
        "prediction_sample_ids": sorted(records),
        "prediction_completed_at_utc": completed_at,
        "prediction_artifact_created_exclusive": True,
        "artifact_size_bytes": int(artifact_stat.st_size),
        "artifact_mtime_ns": int(artifact_stat.st_mtime_ns),
        "declared_yaw_values_rad": yaw_values,
        "declared_strata": strata,
    }
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, sort_keys=True, indent=2)
        stream.write("\n")
    return manifest


def _load_local_predictions(
    prediction_manifest_path: Path | str,
    dataset_tag: str,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Any]]:
    raw_manifest_path = Path(prediction_manifest_path).expanduser()
    if raw_manifest_path.is_symlink():
        raise ValueError("prediction manifest path must not be a symlink")
    manifest_path = raw_manifest_path.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"prediction manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fields = {
        "schema_version",
        "status",
        "run_id",
        "prediction_manifest_id",
        "dataset_tag",
        "prediction_artifact_path",
        "prediction_manifest_path",
        "prediction_record_count",
        "prediction_sample_ids",
        "prediction_completed_at_utc",
        "prediction_artifact_created_exclusive",
        "artifact_size_bytes",
        "artifact_mtime_ns",
        "declared_yaw_values_rad",
        "declared_strata",
    }
    if not isinstance(manifest, Mapping) or set(manifest) != fields:
        raise ValueError("local prediction manifest schema is not exact")
    if (
        manifest["schema_version"] != LOCAL_MANIFEST_SCHEMA_VERSION
        or manifest["status"] != "LOCAL_PREDICTION_ARTIFACT_COMPLETE"
        or manifest["prediction_artifact_created_exclusive"] is not True
    ):
        raise ValueError("local prediction artifact is not complete")
    if manifest["dataset_tag"] != dataset_tag.strip():
        raise ValueError("prediction manifest dataset_tag does not match")
    if manifest["prediction_manifest_path"] != str(manifest_path):
        raise ValueError("prediction manifest path does not match local path")
    _utc_timestamp(
        manifest["prediction_completed_at_utc"],
        "prediction_completed_at_utc",
    )
    yaw_values, strata = _declared_coverage(
        manifest["declared_yaw_values_rad"], manifest["declared_strata"]
    )
    artifact_raw = Path(manifest["prediction_artifact_path"]).expanduser()
    if artifact_raw.is_symlink():
        raise ValueError("prediction artifact path must not be a symlink")
    artifact = artifact_raw.resolve()
    if str(artifact) != manifest["prediction_artifact_path"] or not artifact.is_file():
        raise ValueError("local prediction artifact path is invalid")
    before = artifact.stat()
    if (
        int(before.st_size) != manifest["artifact_size_bytes"]
        or int(before.st_mtime_ns) != manifest["artifact_mtime_ns"]
    ):
        raise ValueError("prediction artifact changed after local recording")
    raw_records = []
    for line_number, line in enumerate(
        artifact.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, Mapping):
            raise ValueError(f"prediction artifact line {line_number} is not an object")
        raw_records.append(record)
    after = artifact.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("prediction artifact changed while being read")
    records = _prediction_records(raw_records)
    if (
        manifest["prediction_record_count"] != len(records)
        or manifest["prediction_sample_ids"] != sorted(records)
    ):
        raise ValueError("prediction artifact does not match local manifest")
    local_checked = dict(manifest)
    local_checked["declared_yaw_values_rad"] = yaw_values
    local_checked["declared_strata"] = strata
    return records, local_checked


def _validate_claimed_reveal_metadata(
    value: Any,
    manifest: Mapping[str, Any],
    truth_record_count: int,
) -> dict[str, Any]:
    fields = {
        "schema_version",
        "status",
        "run_id",
        "dataset_tag",
        "truth_reveal_id",
        "truth_record_count",
        "truth_revealed_at_utc",
        "prediction_manifest_path",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("exact claimed_reveal_metadata is required")
    if (
        value["schema_version"] != CLAIMED_REVEAL_SCHEMA_VERSION
        or value["status"] != "CALLER_CLAIMS_TRUTH_AVAILABLE_FOR_EVALUATION"
    ):
        raise ValueError("claimed reveal metadata is not evaluation-complete")
    if (
        value["run_id"] != manifest["run_id"]
        or value["dataset_tag"] != manifest["dataset_tag"]
        or value["prediction_manifest_path"] != manifest["prediction_manifest_path"]
    ):
        raise ValueError("claimed reveal does not match local prediction run")
    if not isinstance(value["truth_reveal_id"], str) or not value["truth_reveal_id"]:
        raise ValueError("truth_reveal_id must be non-empty text")
    if value["truth_record_count"] != truth_record_count:
        raise ValueError("claimed reveal record count does not match")
    completed_at = _utc_timestamp(
        manifest["prediction_completed_at_utc"], "prediction_completed_at_utc"
    )
    revealed_at = _utc_timestamp(
        value["truth_revealed_at_utc"], "truth_revealed_at_utc"
    )
    if not completed_at < revealed_at:
        raise ValueError("claimed reveal time must follow local prediction completion")
    return dict(value)


def evaluate_local_key_yaw_benchmark_metrics(
    prediction_manifest_path: Path | str,
    claimed_truth_records: Sequence[Mapping[str, Any]],
    *,
    keyed_model_id: str,
    dataset_tag: str,
    claimed_reveal_metadata: Mapping[str, Any] | None = None,
    profile_name: str = DEFAULT_STRESS_PROFILE,
    minimum_samples: int = DEFAULT_SIMULATION_MINIMUM_SAMPLES,
    minimum_visible_yield: float = DEFAULT_MINIMUM_VISIBLE_YIELD,
    required_stratum_axes: Sequence[str] = ("yaw", "light", "pose"),
    minimum_stratum_samples: int = DEFAULT_MINIMUM_STRATUM_SAMPLES,
    minimum_must_reject_samples: int = DEFAULT_MINIMUM_MUST_REJECT_SAMPLES,
    minimum_must_reject_per_class: int = DEFAULT_MINIMUM_MUST_REJECT_PER_CLASS,
) -> dict[str, Any]:
    """Compute local metrics without claiming formal withheld evidence."""
    if keyed_model_id != PLUG_MODEL_ID:
        raise ValueError(f"keyed_model_id must be exactly {PLUG_MODEL_ID}")
    if profile_name != DEFAULT_STRESS_PROFILE:
        raise ValueError("profile_name cannot weaken the adversarial stress contract")
    if (
        isinstance(minimum_samples, bool)
        or not isinstance(minimum_samples, (int, np.integer))
        or int(minimum_samples) < DEFAULT_SIMULATION_MINIMUM_SAMPLES
    ):
        raise ValueError("minimum_samples cannot weaken the 1000-sample contract")
    minimum_yield = _finite_number(minimum_visible_yield, "minimum_visible_yield")
    if not DEFAULT_MINIMUM_VISIBLE_YIELD <= minimum_yield <= 1.0:
        raise ValueError("minimum_visible_yield cannot weaken the 0.99 contract")
    if (
        isinstance(minimum_stratum_samples, bool)
        or not isinstance(minimum_stratum_samples, (int, np.integer))
        or int(minimum_stratum_samples) < DEFAULT_MINIMUM_STRATUM_SAMPLES
    ):
        raise ValueError("minimum_stratum_samples cannot weaken the 30-sample contract")
    if (
        isinstance(minimum_must_reject_samples, bool)
        or not isinstance(minimum_must_reject_samples, (int, np.integer))
        or int(minimum_must_reject_samples) < DEFAULT_MINIMUM_MUST_REJECT_SAMPLES
    ):
        raise ValueError(
            "minimum_must_reject_samples cannot weaken the 1024-sample contract"
        )
    if (
        isinstance(minimum_must_reject_per_class, bool)
        or not isinstance(minimum_must_reject_per_class, (int, np.integer))
        or int(minimum_must_reject_per_class)
        < DEFAULT_MINIMUM_MUST_REJECT_PER_CLASS
    ):
        raise ValueError(
            "minimum_must_reject_per_class cannot weaken the 256-sample contract"
        )
    axes = tuple(required_stratum_axes)
    if axes != REQUIRED_STRATUM_AXES:
        raise ValueError("required_stratum_axes must remain exactly yaw/light/pose")

    prediction_by_id, local_manifest = _load_local_predictions(
        prediction_manifest_path, dataset_tag
    )
    truth_by_id = _truth_records(claimed_truth_records)
    if set(prediction_by_id) != set(truth_by_id):
        raise ValueError("prediction and withheld-truth sample IDs differ")
    claimed_reveal = _validate_claimed_reveal_metadata(
        claimed_reveal_metadata,
        local_manifest,
        len(truth_by_id),
    )

    visible_truth = []
    visible_estimates = []
    visible_records = []
    visible_passed = 0
    c2_misselection_count = 0
    must_reject_count = 0
    must_reject_false_accept_count = 0
    rejection_class_counts = {name: 0 for name in REQUIRED_REJECTION_CLASSES}
    for sample_id in sorted(truth_by_id):
        truth_record = truth_by_id[sample_id]
        prediction = prediction_by_id[sample_id]
        if truth_record["expected_outcome"] == "MUST_REJECT":
            must_reject_count += 1
            rejection_class_counts[truth_record["rejection_class"]] += 1
            if prediction["passed"]:
                must_reject_false_accept_count += 1
            continue
        truth_yaw = float(truth_record["axial_yaw_truth_rad"])
        visible_truth.append(truth_yaw)
        if prediction["passed"]:
            visible_passed += 1
            estimate = float(prediction["estimated_axial_yaw_rad"])
            if prediction["selected_hypothesis_id"] != truth_record["expected_hypothesis_id"]:
                c2_misselection_count += 1
        else:
            # A rejected valid sample receives the maximum wrapped error.
            estimate = float(_wrap_rad(truth_yaw + math.pi))
        visible_estimates.append(estimate)
        visible_records.append((truth_record, estimate, truth_yaw))

    actual_yaw_values = sorted(
        {round(float(_wrap_rad(value)), 12) for value in visible_truth}
    )
    declared_yaw_values = local_manifest["declared_yaw_values_rad"]
    if (
        len(actual_yaw_values) != REQUIRED_DISTINCT_YAW_VALUES
        or not np.allclose(
            actual_yaw_values,
            declared_yaw_values,
            atol=1.0e-10,
            rtol=0.0,
        )
    ):
        raise ValueError("withheld truth must match the 64 declared yaw values")
    observed_strata = {
        axis: sorted(
            {record["strata"][axis] for record, _, _ in visible_records}
        )
        for axis in REQUIRED_STRATUM_AXES
    }
    if observed_strata != local_manifest["declared_strata"]:
        raise ValueError("withheld truth strata do not match declared coverage")
    yaw_label_values: dict[str, set[float]] = {}
    for record, _, truth_yaw in visible_records:
        yaw_label_values.setdefault(record["strata"]["yaw"], set()).add(
            round(float(_wrap_rad(truth_yaw)), 12)
        )
    if any(len(values) != 1 for values in yaw_label_values.values()):
        raise ValueError("each yaw stratum must identify exactly one actual yaw value")

    truth_array = np.asarray(visible_truth, dtype=np.float64)
    estimate_array = np.asarray(visible_estimates, dtype=np.float64)
    p95_gate = dict(
        evaluate_public_spec_sim_key_yaw_acceptance(
            estimate_array,
            truth_array,
            keyed_model_id=keyed_model_id,
            profile_name=profile_name,
            minimum_samples=minimum_samples,
            dataset_tag=dataset_tag,
            withheld_truth=False,
        )
    )
    observed_p95_deg = p95_gate["observed_yaw_error_p95_deg"]
    local_p95_metric_passed = bool(
        p95_gate["sample_count"] >= int(minimum_samples)
        and observed_p95_deg is not None
        and observed_p95_deg
        < p95_gate["required_yaw_error_p95_deg"]
        - NUMERIC_EQUALITY_TOLERANCE_DEG
    )
    p95_gate.update(
        mode="LOCAL_NUMERIC_P95_METRIC_FORMAL_WITHHELD_BLOCKED",
        status=(
            "LOCAL_P95_METRIC_PASSED_FORMAL_WITHHELD_BLOCKED"
            if local_p95_metric_passed
            else "LOCAL_P95_METRIC_REJECTED_FORMAL_WITHHELD_BLOCKED"
        ),
        reason=(None if local_p95_metric_passed else "LOCAL_P95_METRIC_GATE_FAILED"),
        passed=local_p95_metric_passed,
        withheld_truth=False,
        formal_withheld_evidence_verified=False,
        formal_withheld_evidence_status=FORMAL_WITHHELD_BLOCKED_STATUS,
    )
    p95_gate["shadow_authorized"] = False
    p95_gate["control_authorized"] = False
    required_p95_deg = float(p95_gate["required_yaw_error_p95_deg"])
    wrapped_errors_deg = np.degrees(
        np.abs(_wrap_rad(estimate_array - truth_array))
    )
    p95_upper_bound_deg = _bootstrap_p95_upper_bound_deg(
        wrapped_errors_deg, BOOTSTRAP_SEED
    )
    p95_upper_bound_gate_passed = bool(
        p95_upper_bound_deg is not None
        and p95_upper_bound_deg
        < required_p95_deg - NUMERIC_EQUALITY_TOLERANCE_DEG
    )
    visible_count = int(truth_array.size)
    visible_yield = float(visible_passed / visible_count) if visible_count else 0.0

    stratum_reports = []
    all_strata_passed = True
    bootstrap_group_index = 0
    for axis in axes:
        groups: dict[str, list[float]] = {}
        for truth_record, estimate, truth_yaw in visible_records:
            strata = truth_record["strata"]
            if axis not in strata or not isinstance(strata[axis], str) or not strata[axis]:
                raise ValueError(f"visible truth record is missing stratum axis {axis}")
            error_deg = abs(math.degrees(float(_wrap_rad(estimate - truth_yaw))))
            groups.setdefault(strata[axis], []).append(error_deg)
        for label in sorted(groups):
            errors = np.asarray(groups[label], dtype=np.float64)
            p95_deg = float(np.percentile(errors, 95.0))
            bootstrap_group_index += 1
            upper_bound_deg = _bootstrap_p95_upper_bound_deg(
                errors, BOOTSTRAP_SEED + bootstrap_group_index
            )
            enough = bool(errors.size >= int(minimum_stratum_samples))
            passed = bool(
                enough
                and p95_deg
                < required_p95_deg - NUMERIC_EQUALITY_TOLERANCE_DEG
                and upper_bound_deg is not None
                and upper_bound_deg
                < required_p95_deg - NUMERIC_EQUALITY_TOLERANCE_DEG
            )
            all_strata_passed = all_strata_passed and passed
            stratum_reports.append(
                {
                    "axis": axis,
                    "label": label,
                    "sample_count": int(errors.size),
                    "observed_yaw_error_p95_deg": p95_deg,
                    "bootstrap_p95_one_sided_95_upper_bound_deg": upper_bound_deg,
                    "required_yaw_error_p95_deg": required_p95_deg,
                    "minimum_stratum_samples": int(minimum_stratum_samples),
                    "passed": passed,
                }
            )

    yield_passed = bool(visible_yield >= minimum_yield)
    c2_passed = c2_misselection_count == 0
    minimum_must_reject_samples = int(minimum_must_reject_samples)
    minimum_must_reject_per_class = int(minimum_must_reject_per_class)
    must_reject_count_gate_passed = bool(
        must_reject_count >= minimum_must_reject_samples
    )
    rejection_class_reports = [
        {
            "rejection_class": rejection_class,
            "sample_count": rejection_class_counts[rejection_class],
            "minimum_samples": minimum_must_reject_per_class,
            "passed": bool(
                rejection_class_counts[rejection_class]
                >= minimum_must_reject_per_class
            ),
        }
        for rejection_class in REQUIRED_REJECTION_CLASSES
    ]
    rejection_class_quotas_passed = all(
        item["passed"] for item in rejection_class_reports
    )
    rejection_passed = bool(
        must_reject_false_accept_count == 0
        and must_reject_count_gate_passed
        and rejection_class_quotas_passed
    )
    metric_gates_passed = bool(
        p95_gate.get("passed") is True
        and p95_upper_bound_gate_passed
        and yield_passed
        and c2_passed
        and rejection_passed
        and all_strata_passed
    )
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "mode": BENCHMARK_MODE,
        "status": (
            "METRIC_GATES_PASSED_FORMAL_WITHHELD_EVIDENCE_BLOCKED"
            if metric_gates_passed
            else "METRIC_GATES_REJECTED_FORMAL_WITHHELD_EVIDENCE_BLOCKED"
        ),
        "passed": False,
        "metric_gates_passed": metric_gates_passed,
        "formal_withheld_evidence_status": FORMAL_WITHHELD_BLOCKED_STATUS,
        "formal_withheld_evidence_verified": False,
        "shadow_only": True,
        "shadow_authorized": False,
        "control_authorized": False,
        "selected_for_control_allowed": False,
        "simulation_insertion_control_authorized": False,
        "robot_control_authorized": False,
        "hardware_control_authorized": False,
        "keyed_model_id": keyed_model_id,
        "dataset_tag": dataset_tag,
        "withheld_protocol": LOCAL_CONSISTENCY_CHECK,
        "two_stage_withheld_protocol_verified": False,
        "local_artifact_accidental_mutation_check_passed": True,
        "local_prediction_manifest": {
            "schema_version": local_manifest["schema_version"],
            "status": local_manifest["status"],
            "run_id": local_manifest["run_id"],
            "prediction_manifest_id": local_manifest["prediction_manifest_id"],
            "prediction_artifact_path": local_manifest[
                "prediction_artifact_path"
            ],
            "prediction_manifest_path": local_manifest[
                "prediction_manifest_path"
            ],
            "prediction_record_count": local_manifest[
                "prediction_record_count"
            ],
            "prediction_completed_at_utc": local_manifest[
                "prediction_completed_at_utc"
            ],
        },
        "caller_claimed_reveal_metadata": claimed_reveal,
        "caller_claimed_reveal_trusted": False,
        "coverage_self_consistency_passed": True,
        "actual_distinct_yaw_value_count": len(actual_yaw_values),
        "observed_strata": observed_strata,
        "threshold_interpretation": PROJECT_STRESS_INTERPRETATION,
        "drawing_specified_mechanical_yaw_clearance": False,
        "real_measured_clearance_deg": None,
        "visible_valid_sample_count": visible_count,
        "visible_valid_passed_count": visible_passed,
        "visible_valid_yield": visible_yield,
        "minimum_visible_yield": minimum_yield,
        "visible_yield_gate_passed": yield_passed,
        "rejected_visible_samples_penalty_deg": 180.0,
        "c2_misselection_count": c2_misselection_count,
        "c2_misselection_gate_passed": c2_passed,
        "must_reject_sample_count": must_reject_count,
        "minimum_must_reject_samples": minimum_must_reject_samples,
        "must_reject_count_gate_passed": must_reject_count_gate_passed,
        "must_reject_false_accept_count": must_reject_false_accept_count,
        "minimum_must_reject_per_class": minimum_must_reject_per_class,
        "rejection_class_reports": rejection_class_reports,
        "rejection_class_quotas_passed": rejection_class_quotas_passed,
        "must_reject_gate_passed": rejection_passed,
        "observed_penalized_yaw_error_p95_deg": (
            None
            if wrapped_errors_deg.size == 0
            else float(np.percentile(wrapped_errors_deg, 95.0))
        ),
        "bootstrap_method": "FIXED_SEED_NONPARAMETRIC_P95_ONE_SIDED_95_UPPER",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_p95_one_sided_95_upper_bound_deg": p95_upper_bound_deg,
        "bootstrap_p95_upper_bound_gate_passed": p95_upper_bound_gate_passed,
        "p95_gate": p95_gate,
        "stratum_reports": stratum_reports,
        "all_strata_passed": all_strata_passed,
    }


__all__ = [
    "BENCHMARK_SCHEMA_VERSION",
    "CLAIMED_REVEAL_SCHEMA_VERSION",
    "DEFAULT_STRESS_PROFILE",
    "FORMAL_WITHHELD_BLOCKED_STATUS",
    "LOCAL_CONSISTENCY_CHECK",
    "LOCAL_MANIFEST_SCHEMA_VERSION",
    "PROJECT_STRESS_INTERPRETATION",
    "SCHEMA_VERSION",
    "evaluate_local_key_yaw_benchmark_metrics",
    "refine_keyed_axial_yaw_from_rgbd",
    "write_local_key_yaw_prediction_artifact",
]
