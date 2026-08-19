"""Pure-CPU palm-view detector for the five D38999 radial keys.

The detector is a deliberately conservative simulation baseline.  It consumes
only an observed connector-face mask, depth image, observed face centre, an
explicit occlusion mask, and the exact keyed-v2 model identity.  In particular
it has no semantic/object-pose/collider input and it never authorizes control.

The outer face contour is represented in polar bins around ``face_center_uv``.
Five radial protrusions are required.  The master key is selected only when its
angular/tangential width is uniquely larger than the four minor keys, following
the public 2.54 mm versus 1.32 mm nominal-width ratio.  These dimensions are
used only as a dimensionless ratio: camera intrinsics are intentionally not
invented here.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np


SCHEMA_VERSION = "kcg_d38999_key_region_detector_v1"
MODE = "PALM_RGBD_POLAR_CONTOUR_KEY_SHADOW_ONLY"
SUPPORTED_KEYED_MODEL_ID = "d38999_26kj61sn_keyed_proxy_v2"
MASTER_KEY_WIDTH_MM = 2.54
MINOR_KEY_WIDTH_MM = 1.32
EXPECTED_MASTER_TO_MINOR_WIDTH_RATIO = MASTER_KEY_WIDTH_MM / MINOR_KEY_WIDTH_MM
EXPECTED_MINOR_OFFSETS_DEG = (80.0, 142.0, 196.0, 293.0)
EXPECTED_MIRRORED_MINOR_OFFSETS_DEG = tuple(
    sorted((-value) % 360.0 for value in EXPECTED_MINOR_OFFSETS_DEG)
)

# These are conservative pixel-domain gates for the simulation baseline, not
# calibrated probabilities or real-hardware limits.
IMAGE_BORDER_MARGIN_PX = 2
MINIMUM_FACE_PIXELS = 200
MINIMUM_BODY_RADIUS_PX = 12.0
MINIMUM_POLAR_COVERAGE_FRACTION = 0.92
MINIMUM_PROTRUSION_EXCESS_PX = 2.0
MINIMUM_CANDIDATE_PEAK_EXCESS_PX = 3.0
MINIMUM_CANDIDATE_WIDTH_DEG = 2.0
MAXIMUM_CANDIDATE_WIDTH_DEG = 32.0
MAXIMUM_MINOR_WIDTH_CV = 0.28
MINIMUM_MASTER_TO_SECOND_WIDTH_RATIO = 1.35
MINIMUM_MASTER_TO_MEDIAN_MINOR_RATIO = 1.45
MAXIMUM_MASTER_TO_MEDIAN_MINOR_RATIO = 2.70
MINIMUM_DETECTION_CONFIDENCE = 0.72
MAXIMUM_MINOR_PATTERN_ANGLE_ERROR_DEG = 8.0


def _base_result(shape: tuple[int, int], keyed_model_id: Any) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": MODE,
        "status": "REJECTED",
        "reason": None,
        "rejection_code": None,
        "passed": False,
        "shadow_only": True,
        "control_authorized": False,
        "confidence_calibrated": False,
        "keyed_model_id": keyed_model_id if isinstance(keyed_model_id, str) else None,
        "key_probability": np.zeros(shape, dtype=np.float64),
        "key_direction_uv": None,
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
        {
            "status": status,
            "reason": reason,
            "rejection_code": rejection_code,
            "quality_diagnostics": diagnostics,
        }
    )
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


def _finite_uv(value: Sequence[float]) -> np.ndarray:
    try:
        center = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("face_center_uv must be numeric") from exc
    if center.shape != (2,) or not np.all(np.isfinite(center)):
        raise ValueError("face_center_uv must be a finite UV pair")
    return center


def _touches_border(mask: np.ndarray, margin_px: int) -> bool:
    band = max(1, int(margin_px))
    return bool(
        np.any(mask[:band, :])
        or np.any(mask[-band:, :])
        or np.any(mask[:, :band])
        or np.any(mask[:, -band:])
    )


def _circular_smooth(values: np.ndarray, half_width: int = 1) -> np.ndarray:
    neighbours = [np.roll(values, offset) for offset in range(-half_width, half_width + 1)]
    return np.median(np.stack(neighbours, axis=0), axis=0)


def _cyclic_runs(flags: np.ndarray) -> list[np.ndarray]:
    """Return contiguous true runs, with index zero joined cyclically to N-1."""
    if flags.ndim != 1:
        raise ValueError("flags must be one-dimensional")
    if not np.any(flags):
        return []
    if np.all(flags):
        return [np.arange(flags.size, dtype=np.int64)]
    starts = np.flatnonzero(flags & ~np.roll(flags, 1))
    runs: list[np.ndarray] = []
    for start_value in starts:
        start = int(start_value)
        indices: list[int] = []
        index = start
        while flags[index]:
            indices.append(index)
            index = (index + 1) % flags.size
        runs.append(np.asarray(indices, dtype=np.int64))
    return runs


def _fill_short_cyclic_gaps(flags: np.ndarray, maximum_gap: int) -> np.ndarray:
    filled = flags.copy()
    for gap in _cyclic_runs(~flags):
        if gap.size <= maximum_gap:
            filled[gap] = True
    return filled


def _polar_boundary(
    face: np.ndarray, center: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    rows, columns = np.nonzero(face)
    du = columns.astype(np.float64) - center[0]
    dv = rows.astype(np.float64) - center[1]
    radii = np.hypot(du, dv)
    approximate_radius = math.sqrt(float(face.sum()) / math.pi)
    bin_count = int(np.clip(round(2.0 * math.pi * approximate_radius), 180, 720))
    angles = np.mod(np.arctan2(dv, du), 2.0 * math.pi)
    pixel_bins = np.floor(angles * bin_count / (2.0 * math.pi)).astype(np.int64)
    pixel_bins = np.minimum(pixel_bins, bin_count - 1)

    boundary = np.full(bin_count, np.nan, dtype=np.float64)
    observed = np.zeros(bin_count, dtype=np.bool_)
    for index in range(bin_count):
        selected = pixel_bins == index
        if np.any(selected):
            boundary[index] = float(np.max(radii[selected]))
            observed[index] = True

    valid_indices = np.flatnonzero(observed)
    if valid_indices.size >= 2:
        extended_indices = np.concatenate(
            (valid_indices - bin_count, valid_indices, valid_indices + bin_count)
        )
        extended_values = np.tile(boundary[valid_indices], 3)
        boundary = np.interp(np.arange(bin_count), extended_indices, extended_values)
    coverage = float(valid_indices.size / bin_count)
    return boundary, observed, pixel_bins, radii, coverage


def _candidate_diagnostic(
    indices: np.ndarray,
    excess: np.ndarray,
    bin_count: int,
) -> dict[str, Any]:
    angular_width_deg = float(indices.size * 360.0 / bin_count)
    weights = np.maximum(excess[indices], 1.0e-9)
    angles = (indices.astype(np.float64) + 0.5) * 2.0 * math.pi / bin_count
    complex_mean = np.sum(weights * np.exp(1j * angles))
    centre_angle = float(np.mod(np.angle(complex_mean), 2.0 * math.pi))
    return {
        "bin_indices": indices,
        "angular_width_deg": angular_width_deg,
        "centre_angle_rad_uv": centre_angle,
        "peak_excess_px": float(np.max(excess[indices])),
        "mean_excess_px": float(np.mean(excess[indices])),
    }


def detect_key_region_from_palm_rgbd(
    connector_face_mask: Any,
    depth_m: Any,
    face_center_uv: Sequence[float],
    keyed_model_id: str | None,
    *,
    occlusion_mask: Any | None,
) -> dict[str, Any]:
    """Detect the master-key region and direction, for shadow use only.

    Invalid image shapes/dtypes are programming errors and raise ``ValueError``.
    Missing or unsafe observations return a structured fail-closed result.
    ``occlusion_mask`` is keyword-only so callers cannot accidentally omit its
    meaning while supplying another positional image.
    """

    face = _boolean_image("connector_face_mask", connector_face_mask)
    depth = _numeric_image("depth_m", depth_m)
    if depth.shape != face.shape:
        raise ValueError("depth_m shape must match connector_face_mask")
    center = _finite_uv(face_center_uv)
    base = _base_result(face.shape, keyed_model_id)

    if keyed_model_id != SUPPORTED_KEYED_MODEL_ID:
        reason = (
            "KEYED_MODEL_ID_MISSING"
            if not isinstance(keyed_model_id, str) or not keyed_model_id.strip()
            else "OLD_OR_UNREGISTERED_MODEL_ID"
        )
        return _reject(
            base,
            "MODEL_NOT_KEYED_V2",
            reason,
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
        raise ValueError("occlusion_mask shape must match connector_face_mask")

    height, width = face.shape
    if not (0.0 <= center[0] < width and 0.0 <= center[1] < height):
        return _reject(
            base,
            "OUT_OF_FRAME",
            "FACE_CENTER_OUTSIDE_IMAGE",
            "CONNECTOR_FACE_OUT_OF_FRAME",
        )
    face_pixels = int(np.count_nonzero(face))
    if face_pixels < MINIMUM_FACE_PIXELS:
        return _reject(
            base,
            "LOW_CONFIDENCE",
            "FACE_SUPPORT_TOO_SMALL",
            "KEY_REGION_LOW_CONFIDENCE",
            face_pixels=face_pixels,
        )
    center_u = int(np.clip(round(float(center[0])), 0, width - 1))
    center_v = int(np.clip(round(float(center[1])), 0, height - 1))
    if not face[center_v, center_u]:
        return _reject(
            base,
            "LOW_CONFIDENCE",
            "FACE_CENTER_OUTSIDE_FACE_MASK",
            "KEY_REGION_LOW_CONFIDENCE",
            face_pixels=face_pixels,
        )
    if _touches_border(face, IMAGE_BORDER_MARGIN_PX):
        return _reject(
            base,
            "OUT_OF_FRAME",
            "CONNECTOR_FACE_TOUCHES_IMAGE_BORDER",
            "CONNECTOR_FACE_OUT_OF_FRAME",
            face_pixels=face_pixels,
            image_border_margin_px=IMAGE_BORDER_MARGIN_PX,
        )

    rows, columns = np.indices(face.shape, dtype=np.float64)
    radius_image = np.hypot(columns - center[0], rows - center[1])
    face_radii = radius_image[face]
    observed_outer_radius = float(np.max(face_radii))
    conservative_roi_radius = observed_outer_radius + max(3.0, 0.10 * observed_outer_radius)
    occlusion_roi = radius_image <= conservative_roi_radius
    occluded_roi_pixels = int(np.count_nonzero(occlusion & occlusion_roi))
    if occluded_roi_pixels:
        return _reject(
            base,
            "OCCLUDED",
            "OCCLUSION_INTERSECTS_CONNECTOR_ROI",
            "KEY_REGION_OCCLUDED",
            face_pixels=face_pixels,
            occluded_connector_roi_pixels=occluded_roi_pixels,
            conservative_occlusion_roi_radius_px=conservative_roi_radius,
        )

    valid_depth = np.isfinite(depth) & (depth > 0.0)
    valid_face_depth_pixels = int(np.count_nonzero(valid_depth & face))
    valid_face_depth_fraction = float(valid_face_depth_pixels / face_pixels)
    if valid_face_depth_pixels != face_pixels:
        return _reject(
            base,
            "DEPTH_MISSING",
            "CONNECTOR_FACE_DEPTH_INCOMPLETE",
            "KEY_REGION_DEPTH_MISSING",
            face_pixels=face_pixels,
            valid_face_depth_pixels=valid_face_depth_pixels,
            valid_face_depth_fraction=valid_face_depth_fraction,
        )

    boundary, observed_bins, pixel_bins, pixel_radii, coverage = _polar_boundary(
        face, center
    )
    bin_count = int(boundary.size)
    if coverage < MINIMUM_POLAR_COVERAGE_FRACTION:
        return _reject(
            base,
            "LOW_CONFIDENCE",
            "POLAR_BOUNDARY_COVERAGE_TOO_LOW",
            "KEY_REGION_LOW_CONFIDENCE",
            polar_bin_count=bin_count,
            observed_polar_bins=int(np.count_nonzero(observed_bins)),
            polar_coverage_fraction=coverage,
        )

    smoothed_boundary = _circular_smooth(boundary)
    body_radius = float(np.median(smoothed_boundary))
    if body_radius < MINIMUM_BODY_RADIUS_PX:
        return _reject(
            base,
            "LOW_CONFIDENCE",
            "FACE_RADIUS_TOO_SMALL",
            "KEY_REGION_LOW_CONFIDENCE",
            body_radius_px=body_radius,
        )
    body_residual = smoothed_boundary - body_radius
    lower_residuals = body_residual[body_residual <= np.median(body_residual)]
    contour_noise_mad = float(np.median(np.abs(lower_residuals)))
    protrusion_threshold = max(
        MINIMUM_PROTRUSION_EXCESS_PX,
        4.0 * 1.4826 * contour_noise_mad,
    )
    protrusion_bins = body_residual >= protrusion_threshold
    protrusion_bins = _fill_short_cyclic_gaps(protrusion_bins, maximum_gap=2)
    raw_runs = _cyclic_runs(protrusion_bins)
    candidates: list[dict[str, Any]] = []
    for run in raw_runs:
        candidate = _candidate_diagnostic(run, body_residual, bin_count)
        if (
            candidate["angular_width_deg"] >= MINIMUM_CANDIDATE_WIDTH_DEG
            and candidate["angular_width_deg"] <= MAXIMUM_CANDIDATE_WIDTH_DEG
            and candidate["peak_excess_px"] >= MINIMUM_CANDIDATE_PEAK_EXCESS_PX
        ):
            candidates.append(candidate)

    public_candidates = [
        {key: value for key, value in candidate.items() if key != "bin_indices"}
        for candidate in candidates
    ]
    common_diagnostics = {
        "face_pixels": face_pixels,
        "mean_face_depth_m": float(np.mean(depth[face])),
        "valid_face_depth_fraction": valid_face_depth_fraction,
        "polar_bin_count": bin_count,
        "observed_polar_bins": int(np.count_nonzero(observed_bins)),
        "polar_coverage_fraction": coverage,
        "body_radius_px": body_radius,
        "contour_noise_mad_px": contour_noise_mad,
        "protrusion_threshold_px": protrusion_threshold,
        "candidate_count": len(candidates),
        "candidates": public_candidates,
        "reference_master_key_width_mm": MASTER_KEY_WIDTH_MM,
        "reference_minor_key_width_mm": MINOR_KEY_WIDTH_MM,
        "reference_master_to_minor_width_ratio": (
            EXPECTED_MASTER_TO_MINOR_WIDTH_RATIO
        ),
    }
    if len(candidates) != 5:
        return _reject(
            base,
            "LOW_CONFIDENCE",
            "KEY_CANDIDATE_COUNT_NOT_FIVE",
            "KEY_REGION_LOW_CONFIDENCE",
            **common_diagnostics,
        )

    ordered = sorted(
        range(5),
        key=lambda index: candidates[index]["angular_width_deg"],
        reverse=True,
    )
    master_index = int(ordered[0])
    master_width = float(candidates[master_index]["angular_width_deg"])
    second_width = float(candidates[ordered[1]]["angular_width_deg"])
    minor_widths = np.asarray(
        [
            candidates[index]["angular_width_deg"]
            for index in range(5)
            if index != master_index
        ],
        dtype=np.float64,
    )
    median_minor_width = float(np.median(minor_widths))
    minor_width_cv = float(np.std(minor_widths) / np.mean(minor_widths))
    master_to_second_ratio = master_width / second_width
    master_to_median_minor_ratio = master_width / median_minor_width
    width_diagnostics = {
        **common_diagnostics,
        "master_candidate_index": master_index,
        "master_width_deg": master_width,
        "second_width_deg": second_width,
        "minor_widths_deg": minor_widths.tolist(),
        "median_minor_width_deg": median_minor_width,
        "minor_width_cv": minor_width_cv,
        "master_to_second_width_ratio": master_to_second_ratio,
        "master_to_median_minor_width_ratio": master_to_median_minor_ratio,
    }
    if (
        master_to_second_ratio < MINIMUM_MASTER_TO_SECOND_WIDTH_RATIO
        or master_to_median_minor_ratio < MINIMUM_MASTER_TO_MEDIAN_MINOR_RATIO
        or master_to_median_minor_ratio > MAXIMUM_MASTER_TO_MEDIAN_MINOR_RATIO
    ):
        return _reject(
            base,
            "AMBIGUOUS",
            "MASTER_KEY_WIDTH_NOT_UNIQUE",
            "KEY_DIRECTION_AMBIGUOUS",
            **width_diagnostics,
        )
    if minor_width_cv > MAXIMUM_MINOR_WIDTH_CV:
        return _reject(
            base,
            "LOW_CONFIDENCE",
            "MINOR_KEY_WIDTHS_INCONSISTENT",
            "KEY_REGION_LOW_CONFIDENCE",
            **width_diagnostics,
        )

    master_angle_deg = math.degrees(
        float(candidates[master_index]["centre_angle_rad_uv"])
    )
    observed_minor_offsets_deg = sorted(
        (
            math.degrees(float(candidates[index]["centre_angle_rad_uv"]))
            - master_angle_deg
        )
        % 360.0
        for index in range(5)
        if index != master_index
    )
    direct_pattern_errors_deg = [
        abs(observed - expected)
        for observed, expected in zip(
            observed_minor_offsets_deg, EXPECTED_MINOR_OFFSETS_DEG
        )
    ]
    mirrored_pattern_errors_deg = [
        abs(observed - expected)
        for observed, expected in zip(
            observed_minor_offsets_deg, EXPECTED_MIRRORED_MINOR_OFFSETS_DEG
        )
    ]
    if max(direct_pattern_errors_deg) <= max(mirrored_pattern_errors_deg):
        pattern_chirality = "DIRECT_IMAGE_UV"
        selected_expected_offsets_deg = EXPECTED_MINOR_OFFSETS_DEG
        minor_pattern_errors_deg = direct_pattern_errors_deg
    else:
        pattern_chirality = "MIRRORED_IMAGE_UV"
        selected_expected_offsets_deg = EXPECTED_MIRRORED_MINOR_OFFSETS_DEG
        minor_pattern_errors_deg = mirrored_pattern_errors_deg
    maximum_minor_pattern_error_deg = max(minor_pattern_errors_deg)
    width_diagnostics.update(
        {
            "observed_minor_offsets_from_master_deg": observed_minor_offsets_deg,
            "expected_minor_offsets_from_master_deg": list(
                selected_expected_offsets_deg
            ),
            "direct_n_minor_offsets_from_master_deg": list(
                EXPECTED_MINOR_OFFSETS_DEG
            ),
            "mirrored_n_minor_offsets_from_master_deg": list(
                EXPECTED_MIRRORED_MINOR_OFFSETS_DEG
            ),
            "image_pattern_chirality": pattern_chirality,
            "direct_pattern_angle_errors_deg": direct_pattern_errors_deg,
            "mirrored_pattern_angle_errors_deg": mirrored_pattern_errors_deg,
            "minor_pattern_angle_errors_deg": minor_pattern_errors_deg,
            "maximum_minor_pattern_angle_error_deg": (
                maximum_minor_pattern_error_deg
            ),
            "maximum_allowed_minor_pattern_angle_error_deg": (
                MAXIMUM_MINOR_PATTERN_ANGLE_ERROR_DEG
            ),
        }
    )
    if maximum_minor_pattern_error_deg > MAXIMUM_MINOR_PATTERN_ANGLE_ERROR_DEG:
        return _reject(
            base,
            "LOW_CONFIDENCE",
            "N_POLARIZATION_KEY_PATTERN_INCONSISTENT",
            "KEY_REGION_LOW_CONFIDENCE",
            **width_diagnostics,
        )

    separation_score = float(
        np.clip(
            (master_to_second_ratio - MINIMUM_MASTER_TO_SECOND_WIDTH_RATIO)
            / (
                EXPECTED_MASTER_TO_MINOR_WIDTH_RATIO
                - MINIMUM_MASTER_TO_SECOND_WIDTH_RATIO
            ),
            0.0,
            1.0,
        )
    )
    ratio_score = float(
        np.clip(
            1.0
            - abs(
                master_to_median_minor_ratio
                - EXPECTED_MASTER_TO_MINOR_WIDTH_RATIO
            )
            / (0.50 * EXPECTED_MASTER_TO_MINOR_WIDTH_RATIO),
            0.0,
            1.0,
        )
    )
    minor_consistency_score = float(
        np.clip(1.0 - minor_width_cv / MAXIMUM_MINOR_WIDTH_CV, 0.0, 1.0)
    )
    minimum_peak = min(float(item["peak_excess_px"]) for item in candidates)
    radial_contrast_score = float(
        np.clip(
            (minimum_peak - MINIMUM_CANDIDATE_PEAK_EXCESS_PX)
            / (2.0 * MINIMUM_CANDIDATE_PEAK_EXCESS_PX),
            0.0,
            1.0,
        )
    )
    pattern_consistency_score = float(
        np.clip(
            1.0
            - maximum_minor_pattern_error_deg
            / MAXIMUM_MINOR_PATTERN_ANGLE_ERROR_DEG,
            0.0,
            1.0,
        )
    )
    # Pixel contours add a roughly fixed edge-bin width to every protrusion,
    # which biases the measured wide/narrow ratio down at finite resolution.
    # The hard uniqueness/range gates above still fail closed; this score then
    # combines the four independent quality signals without treating that
    # known raster bias as an automatic rejection.
    detection_confidence = (
        0.25 * separation_score
        + 0.25 * ratio_score
        + 0.15 * minor_consistency_score
        + 0.10 * radial_contrast_score
        + 0.25 * pattern_consistency_score
    )
    confidence_diagnostics = {
        **width_diagnostics,
        "separation_score": separation_score,
        "reference_ratio_score": ratio_score,
        "minor_consistency_score": minor_consistency_score,
        "radial_contrast_score": radial_contrast_score,
        "n_pattern_consistency_score": pattern_consistency_score,
        "detection_confidence": detection_confidence,
        "minimum_detection_confidence": MINIMUM_DETECTION_CONFIDENCE,
    }
    if detection_confidence < MINIMUM_DETECTION_CONFIDENCE:
        return _reject(
            base,
            "LOW_CONFIDENCE",
            "MASTER_KEY_CONFIDENCE_BELOW_LIMIT",
            "KEY_REGION_LOW_CONFIDENCE",
            **confidence_diagnostics,
        )

    master = candidates[master_index]
    angle = float(master["centre_angle_rad_uv"])
    key_direction = np.asarray((math.cos(angle), math.sin(angle)), dtype=np.float64)
    master_bins = np.zeros(bin_count, dtype=np.bool_)
    master_bins[master["bin_indices"]] = True
    # Only the protruding part, not the circular body beneath it, is exposed as
    # the key probability region for the downstream shadow selector.
    master_region_on_face = (
        master_bins[pixel_bins]
        & (pixel_radii >= body_radius + 0.5 * protrusion_threshold)
    )
    key_probability = np.zeros(face.shape, dtype=np.float64)
    face_rows, face_columns = np.nonzero(face)
    key_probability[face_rows[master_region_on_face], face_columns[master_region_on_face]] = (
        detection_confidence
    )
    support_pixels = int(np.count_nonzero(key_probability))
    confidence_diagnostics.update(
        {
            "master_key_support_pixels": support_pixels,
            "key_direction_method": "POLAR_CONTOUR_UNIQUE_WIDEST_OF_FIVE",
        }
    )
    if support_pixels == 0:
        return _reject(
            base,
            "LOW_CONFIDENCE",
            "MASTER_KEY_REGION_EMPTY",
            "KEY_REGION_LOW_CONFIDENCE",
            **confidence_diagnostics,
        )

    result = dict(base)
    result.update(
        {
            "status": "KEY_DIRECTION_DETECTED_SHADOW_ONLY",
            "reason": None,
            "rejection_code": None,
            "passed": True,
            "key_probability": key_probability,
            "key_direction_uv": key_direction.tolist(),
            "quality_diagnostics": confidence_diagnostics,
        }
    )
    return result


# Short public alias for callers while retaining the explicit palm/RGB-D name.
detect_key_region = detect_key_region_from_palm_rgbd


__all__ = [
    "EXPECTED_MASTER_TO_MINOR_WIDTH_RATIO",
    "EXPECTED_MINOR_OFFSETS_DEG",
    "EXPECTED_MIRRORED_MINOR_OFFSETS_DEG",
    "MASTER_KEY_WIDTH_MM",
    "MINOR_KEY_WIDTH_MM",
    "MODE",
    "SCHEMA_VERSION",
    "SUPPORTED_KEYED_MODEL_ID",
    "detect_key_region",
    "detect_key_region_from_palm_rgbd",
]
