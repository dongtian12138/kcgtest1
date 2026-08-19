"""Fail-closed D38999 key-direction selector for the two C2 branches.

This module is deliberately CPU-only and shadow-only.  It consumes image
observations, never simulator/object truth, and it never authorizes control.
The defaults below are candidates for simulation tuning; they are not measured
hardware limits and they are not confidence-calibrated.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


SCHEMA_VERSION = "kcg_d38999_key_branch_selector_v1"
THRESHOLD_LABEL = "SIM_TUNING_ONLY_CANDIDATE"
BRANCH_IDS = ("C2_LINKED_BRANCH_0", "C2_LINKED_BRANCH_PI")
SHADOW_HYPOTHESIS_IDS = ("YAW_0", "YAW_PI")
SUPPORTED_KEYED_PLUG_MODEL_IDS = frozenset(
    {"d38999_26kj61sn_keyed_proxy_v2"}
)
BLOCKED_KEY_STATUSES = frozenset(
    {
        "KEYED_MODEL_ID_UNAVAILABLE",
        "KEYED_GEOMETRY_UNAVAILABLE",
    }
)

DEFAULT_THRESHOLDS = {
    "minimum_key_probability": 0.65,
    "minimum_mean_key_probability": 0.72,
    "minimum_key_support_pixels": 12,
    "minimum_key_area_fraction": 0.002,
    "maximum_key_area_fraction": 0.15,
    "maximum_secondary_component_mass_ratio": 0.35,
    "maximum_occlusion_fraction": 0.10,
    "minimum_valid_key_depth_fraction": 0.80,
    "image_border_margin_px": 2,
    "minimum_radial_length_px": 5.0,
    "maximum_branch_angle_error_deg": 25.0,
    "minimum_branch_margin_deg": 60.0,
    "threshold_label": THRESHOLD_LABEL,
}


def _base_result(keyed_model_id: Any, thresholds: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "PALM_RGBD_KEY_C2_BRANCH_SHADOW_ONLY",
        "status": "REJECTED",
        "reason": None,
        "rejection_code": None,
        "passed": False,
        "shadow_only": True,
        "control_authorized": False,
        "confidence_calibrated": False,
        "selected_for_shadow": None,
        "shadow_selected_hypothesis_id": None,
        "selected_branch_index": None,
        "keyed_model_id": keyed_model_id if isinstance(keyed_model_id, str) else None,
        "threshold_label": thresholds["threshold_label"],
    }


def _reject(
    base: dict[str, Any], status: str, reason: str, **diagnostics: Any
) -> dict[str, Any]:
    if "rejection_code" not in diagnostics:
        if status == "MODEL_NOT_KEYED_V2":
            rejection_code = "KEYED_MODEL_ID_UNAVAILABLE"
        elif status == "OUT_OF_FRAME":
            rejection_code = (
                "KEY_REGION_OUT_OF_FRAME"
                if reason.startswith("KEY_")
                else "CONNECTOR_FACE_OUT_OF_FRAME"
            )
        elif status == "OCCLUDED":
            rejection_code = "KEY_REGION_OCCLUDED"
        elif status == "OCCLUSION_UNKNOWN":
            rejection_code = "KEY_REGION_OCCLUSION_UNKNOWN"
        elif status == "DEPTH_MISSING":
            rejection_code = "KEY_REGION_DEPTH_MISSING"
        elif status == "LOW_CONFIDENCE":
            rejection_code = "KEY_REGION_LOW_CONFIDENCE"
        elif status == "AMBIGUOUS":
            rejection_code = (
                "KEY_DIRECTION_DEGENERATE"
                if "DEGENERATE" in reason
                else "KEY_BRANCH_AMBIGUOUS"
            )
        else:
            rejection_code = reason
        diagnostics["rejection_code"] = rejection_code
    result = dict(base)
    result.update({"status": status, "reason": reason})
    result.update(diagnostics)
    return result


def blocked_key_branch_selection(
    status: str,
    reason: str,
    *,
    keyed_model_id: str | None = None,
    current_model_id: str | None = None,
) -> dict[str, Any]:
    """Build a shadow-only result when keyed observation cannot yet be run.

    This avoids manufacturing an image observation while the exact keyed-v2
    identity or its key geometry is unavailable.  ``status`` is intentionally
    restricted to explicit pre-observation blockers.
    """
    if status not in BLOCKED_KEY_STATUSES:
        raise ValueError(
            "status must be KEYED_MODEL_ID_UNAVAILABLE or "
            "KEYED_GEOMETRY_UNAVAILABLE"
        )
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be a non-empty string")
    if keyed_model_id is not None and (
        not isinstance(keyed_model_id, str) or not keyed_model_id.strip()
    ):
        raise ValueError("keyed_model_id must be a non-empty string or None")
    if current_model_id is not None and (
        not isinstance(current_model_id, str) or not current_model_id.strip()
    ):
        raise ValueError("current_model_id must be a non-empty string or None")
    result = _base_result(keyed_model_id, DEFAULT_THRESHOLDS)
    result.update(
        {
            "status": status,
            "reason": reason.strip(),
            "blocked_before_observation": True,
            "current_model_id": current_model_id,
            "rejection_code": status,
        }
    )
    return result


def _resolved_thresholds(overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    values = dict(DEFAULT_THRESHOLDS)
    if overrides is not None:
        if not isinstance(overrides, Mapping):
            raise ValueError("thresholds must be a mapping")
        unknown = set(overrides) - set(DEFAULT_THRESHOLDS)
        if unknown:
            raise ValueError(f"unknown thresholds: {sorted(unknown)}")
        values.update(overrides)

    numeric_rules = {
        "minimum_key_probability": (0.0, 1.0, False),
        "minimum_mean_key_probability": (0.0, 1.0, False),
        "minimum_key_area_fraction": (0.0, 1.0, True),
        "maximum_key_area_fraction": (0.0, 1.0, False),
        "maximum_secondary_component_mass_ratio": (0.0, 1.0, False),
        "maximum_occlusion_fraction": (0.0, 1.0, False),
        "minimum_valid_key_depth_fraction": (0.0, 1.0, False),
        "minimum_radial_length_px": (0.0, math.inf, True),
        "maximum_branch_angle_error_deg": (0.0, 180.0, True),
        "minimum_branch_margin_deg": (0.0, 180.0, False),
    }
    for name, (lower, upper, strict_lower) in numeric_rules.items():
        value = values[name]
        if isinstance(value, bool):
            raise ValueError(f"{name} must be numeric")
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be numeric") from exc
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        lower_ok = value > lower if strict_lower else value >= lower
        if not lower_ok or value > upper:
            interval = "(" if strict_lower else "["
            raise ValueError(f"{name} must be in {interval}{lower}, {upper}]")
        values[name] = value

    integer_rules = {
        "minimum_key_support_pixels": 2,
        "image_border_margin_px": 0,
    }
    for name, minimum in integer_rules.items():
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        if int(value) < minimum:
            raise ValueError(f"{name} must be at least {minimum}")
        values[name] = int(value)

    if values["minimum_key_probability"] > values["minimum_mean_key_probability"]:
        raise ValueError(
            "minimum_key_probability must not exceed minimum_mean_key_probability"
        )
    if values["minimum_key_area_fraction"] >= values["maximum_key_area_fraction"]:
        raise ValueError(
            "minimum_key_area_fraction must be smaller than maximum_key_area_fraction"
        )
    if values["threshold_label"] != THRESHOLD_LABEL:
        raise ValueError(f"threshold_label must remain {THRESHOLD_LABEL}")
    return values


def _as_image(name: str, value: Any, *, boolean: bool = False) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape (H, W)")
    if boolean:
        if array.dtype != np.bool_:
            raise ValueError(f"{name} must be a boolean image")
        return array
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be numeric")
    return array.astype(np.float64, copy=False)


def _touches_image_border(mask: np.ndarray, margin_px: int) -> bool:
    # A zero margin still rejects actual edge contact. A positive value keeps
    # that many pixels clear on every side of the image.
    band = max(1, int(margin_px))
    return bool(
        np.any(mask[:band, :])
        or np.any(mask[-band:, :])
        or np.any(mask[:, :band])
        or np.any(mask[:, -band:])
    )


def _largest_weighted_component(
    mask: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, float]:
    """Return the strongest component and runner-up/strongest mass ratio."""
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=np.bool_)
    best_pixels: list[tuple[int, int]] = []
    best_mass = -1.0
    component_masses: list[float] = []
    for start_v, start_u in np.argwhere(mask):
        v0, u0 = int(start_v), int(start_u)
        if visited[v0, u0]:
            continue
        stack = [(v0, u0)]
        visited[v0, u0] = True
        pixels: list[tuple[int, int]] = []
        mass = 0.0
        while stack:
            v, u = stack.pop()
            pixels.append((v, u))
            mass += float(weights[v, u])
            for dv in (-1, 0, 1):
                for du in (-1, 0, 1):
                    if dv == 0 and du == 0:
                        continue
                    nv, nu = v + dv, u + du
                    if (
                        0 <= nv < height
                        and 0 <= nu < width
                        and mask[nv, nu]
                        and not visited[nv, nu]
                    ):
                        visited[nv, nu] = True
                        stack.append((nv, nu))
        component_masses.append(mass)
        if mass > best_mass:
            best_mass = mass
            best_pixels = pixels

    component = np.zeros_like(mask, dtype=np.bool_)
    if best_pixels:
        rows, columns = zip(*best_pixels)
        component[np.asarray(rows), np.asarray(columns)] = True
    ordered_masses = sorted(component_masses, reverse=True)
    secondary_mass_ratio = (
        float(ordered_masses[1] / ordered_masses[0])
        if len(ordered_masses) > 1 and ordered_masses[0] > 0.0
        else 0.0
    )
    return component, secondary_mass_ratio


def _unit_branch_directions(branch_directions_uv: Any) -> np.ndarray:
    try:
        directions = np.asarray(branch_directions_uv, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("branch_directions_uv must be numeric") from exc
    if directions.shape != (2, 2):
        raise ValueError("branch_directions_uv must contain exactly two UV vectors")
    if not np.all(np.isfinite(directions)):
        raise ValueError("branch_directions_uv must be finite")
    norms = np.linalg.norm(directions, axis=1)
    if np.any(norms <= 1.0e-12):
        raise ValueError("branch direction vectors must be non-zero")
    return directions / norms[:, None]


def _angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    cosine = float(np.clip(np.dot(first, second), -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))


def select_key_branch_from_rgbd(
    key_probability: Any,
    face_mask: Any,
    depth_m: Any,
    face_center_uv: Sequence[float],
    branch_directions_uv: Sequence[Sequence[float]],
    keyed_model_id: str | None,
    *,
    occlusion_mask: Any | None = None,
    thresholds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Select one of exactly two C2 branches for shadow evaluation only.

    Invalid array shapes or threshold definitions raise ``ValueError`` because
    they are programming/configuration errors.  Missing or weak observations
    return a structured rejection and do not raise.
    """
    limits = _resolved_thresholds(thresholds)
    probability = _as_image("key_probability", key_probability)
    face = _as_image("face_mask", face_mask, boolean=True)
    depth = _as_image("depth_m", depth_m)
    if probability.shape != face.shape or depth.shape != face.shape:
        raise ValueError("key_probability, face_mask, and depth_m shapes must match")

    try:
        center = np.asarray(face_center_uv, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("face_center_uv must be numeric") from exc
    if center.shape != (2,) or not np.all(np.isfinite(center)):
        raise ValueError("face_center_uv must be a finite UV pair")
    branches = _unit_branch_directions(branch_directions_uv)
    base = _base_result(keyed_model_id, limits)

    if not isinstance(keyed_model_id, str) or not keyed_model_id.strip():
        return _reject(
            base,
            "MODEL_NOT_KEYED_V2",
            "KEYED_MODEL_ID_MISSING",
        )
    model_id = keyed_model_id.strip()
    if model_id not in SUPPORTED_KEYED_PLUG_MODEL_IDS:
        normalized_model_id = model_id.lower().replace("-", "_")
        reason = (
            "OLD_OR_UNKEYED_MODEL_ID"
            if "keyed" not in normalized_model_id or "_v1" in normalized_model_id
            else "UNREGISTERED_KEYED_V2_MODEL_ID"
        )
        return _reject(
            base,
            "MODEL_NOT_KEYED_V2",
            reason,
            rejection_code="KEYED_MODEL_ID_UNAVAILABLE",
        )

    if occlusion_mask is None:
        return _reject(
            base,
            "OCCLUSION_UNKNOWN",
            "OCCLUSION_MASK_MISSING",
            rejection_code="KEY_REGION_OCCLUSION_UNKNOWN",
        )
    occlusion = _as_image("occlusion_mask", occlusion_mask, boolean=True)
    if occlusion.shape != face.shape:
        raise ValueError("occlusion_mask shape must match face_mask")

    height, width = face.shape
    if not (0.0 <= center[0] < width and 0.0 <= center[1] < height):
        return _reject(
            base,
            "OUT_OF_FRAME",
            "FACE_CENTER_OUTSIDE_IMAGE",
            rejection_code="CONNECTOR_FACE_OUT_OF_FRAME",
        )
    face_pixels = int(np.count_nonzero(face))
    if face_pixels == 0:
        return _reject(base, "LOW_CONFIDENCE", "FACE_MASK_EMPTY")
    center_u = min(width - 1, max(0, int(round(float(center[0])))))
    center_v = min(height - 1, max(0, int(round(float(center[1])))))
    if not face[center_v, center_u]:
        return _reject(
            base,
            "LOW_CONFIDENCE",
            "FACE_CENTER_OUTSIDE_FACE_MASK",
            rejection_code="KEY_REGION_LOW_CONFIDENCE",
        )
    border_margin = limits["image_border_margin_px"]
    if _touches_image_border(face, border_margin):
        border_key = (
            face
            & np.isfinite(probability)
            & (probability >= limits["minimum_key_probability"])
        )
        if _touches_image_border(border_key, border_margin):
            return _reject(
                base,
                "OUT_OF_FRAME",
                "KEY_TOUCHES_IMAGE_BORDER",
                rejection_code="KEY_REGION_OUT_OF_FRAME",
            )
        return _reject(
            base,
            "OUT_OF_FRAME",
            "FACE_TOUCHES_IMAGE_BORDER",
            rejection_code="CONNECTOR_FACE_OUT_OF_FRAME",
        )

    face_occlusion_fraction = float(np.count_nonzero(face & occlusion) / face_pixels)
    if face_occlusion_fraction > limits["maximum_occlusion_fraction"]:
        return _reject(
            base,
            "OCCLUDED",
            "FACE_OCCLUSION_ABOVE_LIMIT",
            face_occlusion_fraction=face_occlusion_fraction,
        )

    face_probabilities = probability[face]
    if (
        not np.all(np.isfinite(face_probabilities))
        or np.any(face_probabilities < 0.0)
        or np.any(face_probabilities > 1.0)
    ):
        return _reject(base, "LOW_CONFIDENCE", "KEY_PROBABILITY_INVALID")

    raw_key = face & (probability >= limits["minimum_key_probability"])
    key, secondary_component_mass_ratio = _largest_weighted_component(
        raw_key, probability
    )
    key_pixels = int(np.count_nonzero(key))
    key_area_fraction = float(key_pixels / face_pixels)
    if key_pixels < limits["minimum_key_support_pixels"]:
        return _reject(
            base,
            "LOW_CONFIDENCE",
            "KEY_SUPPORT_TOO_SMALL",
            key_support_pixels=key_pixels,
            key_area_fraction=key_area_fraction,
        )
    if (
        secondary_component_mass_ratio
        > limits["maximum_secondary_component_mass_ratio"]
    ):
        return _reject(
            base,
            "LOW_CONFIDENCE",
            "MULTIPLE_COMPARABLE_KEY_COMPONENTS",
            key_support_pixels=key_pixels,
            key_area_fraction=key_area_fraction,
            secondary_component_mass_ratio=secondary_component_mass_ratio,
        )
    if _touches_image_border(key, border_margin):
        return _reject(
            base,
            "OUT_OF_FRAME",
            "KEY_TOUCHES_IMAGE_BORDER",
            rejection_code="KEY_REGION_OUT_OF_FRAME",
        )

    key_mean_probability = float(np.mean(probability[key]))
    if key_mean_probability < limits["minimum_mean_key_probability"]:
        return _reject(
            base,
            "LOW_CONFIDENCE",
            "KEY_MEAN_PROBABILITY_TOO_LOW",
            key_support_pixels=key_pixels,
            key_mean_probability=key_mean_probability,
            key_area_fraction=key_area_fraction,
        )
    if key_area_fraction < limits["minimum_key_area_fraction"]:
        return _reject(
            base,
            "LOW_CONFIDENCE",
            "KEY_AREA_FRACTION_TOO_SMALL",
            key_support_pixels=key_pixels,
            key_mean_probability=key_mean_probability,
            key_area_fraction=key_area_fraction,
        )
    if key_area_fraction > limits["maximum_key_area_fraction"]:
        return _reject(
            base,
            "LOW_CONFIDENCE",
            "KEY_AREA_FRACTION_TOO_LARGE",
            key_support_pixels=key_pixels,
            key_mean_probability=key_mean_probability,
            key_area_fraction=key_area_fraction,
        )

    key_occlusion_fraction = float(np.count_nonzero(key & occlusion) / key_pixels)
    if key_occlusion_fraction > limits["maximum_occlusion_fraction"]:
        return _reject(
            base,
            "OCCLUDED",
            "KEY_OCCLUSION_ABOVE_LIMIT",
            face_occlusion_fraction=face_occlusion_fraction,
            key_occlusion_fraction=key_occlusion_fraction,
        )

    valid_key_depth = key & np.isfinite(depth) & (depth > 0.0)
    valid_key_depth_pixels = int(np.count_nonzero(valid_key_depth))
    valid_key_depth_fraction = float(valid_key_depth_pixels / key_pixels)
    if valid_key_depth_fraction < limits["minimum_valid_key_depth_fraction"]:
        return _reject(
            base,
            "DEPTH_MISSING",
            "VALID_KEY_DEPTH_BELOW_LIMIT",
            valid_key_depth_pixels=valid_key_depth_pixels,
            valid_key_depth_fraction=valid_key_depth_fraction,
        )

    rows, columns = np.nonzero(key)
    points_uv = np.column_stack((columns, rows)).astype(np.float64)
    weights = probability[key].astype(np.float64)
    weight_sum = float(np.sum(weights))
    centroid = np.sum(points_uv * weights[:, None], axis=0) / weight_sum
    radial = centroid - center
    radial_length = float(np.linalg.norm(radial))
    if radial_length < limits["minimum_radial_length_px"]:
        return _reject(
            base,
            "AMBIGUOUS",
            "RADIAL_LENGTH_DEGENERATE",
            key_centroid_uv=centroid.tolist(),
            radial_length_px=radial_length,
        )
    radial_direction = radial / radial_length

    centered = points_uv - centroid
    covariance = (centered * weights[:, None]).T @ centered / weight_sum
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    principal_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    if not np.all(np.isfinite(principal_axis)) or float(np.max(eigenvalues)) <= 1.0e-12:
        return _reject(
            base,
            "AMBIGUOUS",
            "PCA_DIRECTION_DEGENERATE",
            key_centroid_uv=centroid.tolist(),
            radial_length_px=radial_length,
        )
    if float(np.dot(principal_axis, radial_direction)) < 0.0:
        principal_axis = -principal_axis

    principal_axis = principal_axis / np.linalg.norm(principal_axis)
    # Branch choice comes from where the key region lies around the face.
    # Real notches may be radial, tangential or nearly square, so the PCA long
    # axis is diagnostic only and must not decide the discrete C2 branch.
    key_direction = radial_direction
    angle_errors = [_angle_deg(key_direction, branch) for branch in branches]
    ordered = sorted(range(2), key=lambda index: (angle_errors[index], index))
    best_index, second_index = ordered
    best_error = float(angle_errors[best_index])
    branch_margin = float(angle_errors[second_index] - best_error)

    bbox = {
        "u_min": int(np.min(columns)),
        "v_min": int(np.min(rows)),
        "u_max": int(np.max(columns)),
        "v_max": int(np.max(rows)),
    }
    diagnostics = {
        "key_region": {
            "bounding_box_uv": bbox,
            "support_pixels": key_pixels,
            "mean_probability": key_mean_probability,
            "face_area_fraction": key_area_fraction,
            "valid_depth_fraction": valid_key_depth_fraction,
            "mean_depth_m": float(np.average(depth[valid_key_depth], weights=probability[valid_key_depth])),
            "occlusion_fraction": key_occlusion_fraction,
            "secondary_component_mass_ratio": secondary_component_mass_ratio,
        },
        "key_centroid_uv": centroid.tolist(),
        "radial_length_px": radial_length,
        "radial_direction_uv": radial_direction.tolist(),
        "pca_principal_axis_uv": principal_axis.tolist(),
        "pca_eigenvalues": eigenvalues.tolist(),
        "pca_radial_alignment_deg": _angle_deg(principal_axis, radial_direction),
        "key_direction_uv": key_direction.tolist(),
        "key_direction_method": "FACE_CENTER_TO_WEIGHTED_KEY_CENTROID",
        "branch_angle_errors_deg": angle_errors,
        "best_branch_angle_error_deg": best_error,
        "branch_margin_deg": branch_margin,
    }
    if best_error > limits["maximum_branch_angle_error_deg"]:
        return _reject(
            base,
            "AMBIGUOUS",
            "BEST_BRANCH_ANGLE_ERROR_ABOVE_LIMIT",
            **diagnostics,
        )
    if branch_margin < limits["minimum_branch_margin_deg"]:
        return _reject(
            base,
            "AMBIGUOUS",
            "BRANCH_MARGIN_BELOW_LIMIT",
            **diagnostics,
        )

    result = dict(base)
    result.update(diagnostics)
    result.update(
        {
            "status": "SHADOW_BRANCH_SELECTED",
            "reason": None,
            "passed": True,
            "selected_for_shadow": BRANCH_IDS[best_index],
            "shadow_selected_hypothesis_id": SHADOW_HYPOTHESIS_IDS[best_index],
            "selected_branch_index": int(best_index),
        }
    )
    return result


# Short public name for callers; both names retain the shadow-only contract.
select_key_branch = select_key_branch_from_rgbd


__all__ = [
    "BLOCKED_KEY_STATUSES",
    "BRANCH_IDS",
    "DEFAULT_THRESHOLDS",
    "SCHEMA_VERSION",
    "SHADOW_HYPOTHESIS_IDS",
    "SUPPORTED_KEYED_PLUG_MODEL_IDS",
    "THRESHOLD_LABEL",
    "blocked_key_branch_selection",
    "select_key_branch",
    "select_key_branch_from_rgbd",
]
