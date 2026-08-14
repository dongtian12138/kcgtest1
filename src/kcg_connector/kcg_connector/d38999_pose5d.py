"""Truth-free semantic-depth Pose5D and C2 hypothesis estimation.

The estimator consumes only registered depth points belonging to one semantic
mask plus an optional axis prior.  The prior is permitted to come from robot
FK or a calibrated fixture and is used only to choose the sign/eigenvector
branch; it cannot supply the measured center or overwrite the fitted axis.
Simulation truth is intentionally absent from every public function here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml


SCHEMA_VERSION = "kcg_d38999_pose5d_v1"
DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "config/d38999_pose5d_v1.yaml"
)


@dataclass(frozen=True)
class Pose5dEstimate:
    object_id: str
    frame_id: str
    timestamp: str
    capture_id: str
    xyz_m: tuple[float, float, float]
    axis_vector: tuple[float, float, float]
    roll_rad: float
    pitch_rad: float
    yaw_hypotheses: tuple[Mapping[str, float], Mapping[str, float]]
    covariance_6x6: tuple[tuple[float, ...], ...]
    lateral_position_std_m: float
    axis_angle_std_rad: float
    point_count: int
    depth_valid_ratio: float
    radial_fit_rmse_m: float
    planar_anisotropy: float
    confidence: float
    control_authorized: bool
    reject_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_pose5d_config(path: Path | str = DEFAULT_CONFIG_PATH) -> Mapping[str, Any]:
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, Mapping) or document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"pose5d schema must be {SCHEMA_VERSION}")
    estimator = document.get("estimator")
    boundaries = document.get("boundaries")
    symmetry = document.get("symmetry")
    if not all(isinstance(item, Mapping) for item in (estimator, boundaries, symmetry)):
        raise ValueError("pose5d estimator, symmetry, and boundaries must be mappings")
    if document.get("enabled") is not True:
        raise ValueError("pose5d config is disabled")
    if symmetry.get("order") != 2 or symmetry.get("unique_yaw_claimed") is not False:
        raise ValueError("pose5d must preserve two C2 yaw hypotheses")
    for name in (
        "object_truth_allowed_in_estimator",
        "physx_contact_normal_allowed_in_estimator",
        "collider_identity_allowed_in_estimator",
    ):
        if boundaries.get(name) is not False:
            raise ValueError(f"pose5d forbidden boundary {name} must be false")
    return document


def _unit(vector: np.ndarray, label: str) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{label} must be a finite vector3")
    norm = float(np.linalg.norm(value))
    if norm < 1.0e-12:
        raise ValueError(f"{label} is degenerate")
    return value / norm


def _robust_points(points: np.ndarray, quantile: float, maximum: int) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("world_points_m must have shape Nx3")
    values = values[np.all(np.isfinite(values), axis=1)]
    if len(values) == 0:
        return values
    median = np.median(values, axis=0)
    radius = np.linalg.norm(values - median, axis=1)
    cutoff = float(np.quantile(radius, quantile))
    values = values[radius <= cutoff]
    if len(values) > maximum:
        # Deterministic stratified subsampling preserves the full silhouette.
        indices = np.linspace(0, len(values) - 1, maximum, dtype=np.int64)
        values = values[indices]
    return values


def _basis_for_axis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray((1.0, 0.0, 0.0))
    if abs(float(np.dot(reference, axis))) > 0.90:
        reference = np.asarray((0.0, 1.0, 0.0))
    u = _unit(reference - float(np.dot(reference, axis)) * axis, "axis basis")
    return u, np.cross(axis, u)


def _circle_center(points_2d: np.ndarray) -> tuple[np.ndarray, float]:
    """Algebraic circle center with two robust residual-refit passes."""
    values = np.asarray(points_2d, dtype=np.float64)
    center = np.median(values, axis=0)
    for _ in range(3):
        shifted = values - center
        design = np.column_stack((2.0 * shifted[:, 0], 2.0 * shifted[:, 1], np.ones(len(values))))
        target = np.sum(shifted * shifted, axis=1)
        solution, *_ = np.linalg.lstsq(design, target, rcond=None)
        center = center + solution[:2]
        radial = np.linalg.norm(values - center, axis=1)
        residual = np.abs(radial - np.median(radial))
        cutoff = float(np.quantile(residual, 0.90))
        kept = residual <= max(cutoff, 1.0e-9)
        if int(np.sum(kept)) < 20:
            break
        values = values[kept]
    radial = np.linalg.norm(values - center, axis=1)
    rmse = float(np.sqrt(np.mean((radial - np.median(radial)) ** 2)))
    return center, rmse


def _fit_once(
    points: np.ndarray,
    axis_prior: np.ndarray,
    *,
    axial_low_quantile: float,
    axial_registered_offset_m: float,
    visible_cap_quantile: float,
    minimum_cap_point_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    centroid = np.mean(points, axis=0)
    # For the fixed overhead RGB-D view the most positive prior-axis band is
    # a collection of connector end faces.  Its plane normal is much less
    # biased by the nut/flange silhouette than whole-object PCA.  The prior
    # selects the visible band and normal sign only; the depth plane supplies
    # the actual measured axis.
    prior_coordinate = points @ axis_prior
    threshold = float(np.quantile(prior_coordinate, visible_cap_quantile))
    cap = points[prior_coordinate >= threshold]
    if len(cap) < minimum_cap_point_count:
        raise ValueError("visible cap has insufficient points")
    cap_centered = cap - np.mean(cap, axis=0)
    cap_covariance = cap_centered.T @ cap_centered / max(1, len(cap) - 1)
    _, cap_vectors = np.linalg.eigh(cap_covariance)
    axis = cap_vectors[:, 0]
    if float(np.dot(axis, axis_prior)) < 0.0:
        axis = -axis
    axis = _unit(axis, "fitted axis")
    u, v = _basis_for_axis(axis)
    planar = np.column_stack((points @ u, points @ v))
    # Multiple concentric proxy radii are retained here.  In the fixed-camera
    # trials they provide a less occlusion-biased center than the visible cap
    # alone; the cap remains responsible for the axis normal above.
    center_2d, radial_rmse = _circle_center(planar)
    axial_center = float(
        np.quantile(points @ axis, axial_low_quantile)
        + axial_registered_offset_m
    )
    center = center_2d[0] * u + center_2d[1] * v + axial_center * axis
    planar_centered = planar - center_2d
    planar_covariance = np.cov(planar_centered, rowvar=False)
    values, vectors = np.linalg.eigh(planar_covariance)
    major = vectors[:, int(np.argmax(values))]
    c2_reference = _unit(major[0] * u + major[1] * v, "C2 reference")
    denominator = max(float(np.max(values)), 1.0e-12)
    anisotropy = float((np.max(values) - np.min(values)) / denominator)
    return center, axis, c2_reference, radial_rmse, anisotropy


def _axis_angle(a: np.ndarray, b: np.ndarray) -> float:
    return math.acos(float(np.clip(np.dot(a, b), -1.0, 1.0)))


def estimate_pose5d(
    world_points_m: np.ndarray,
    *,
    object_id: str,
    frame_id: str,
    capture_id: str,
    axis_prior: tuple[float, float, float] | np.ndarray,
    depth_valid_ratio: float,
    lateral_authorization_gate_m: float,
    axis_authorization_gate_rad: float,
    timestamp: str | None = None,
    config: Mapping[str, Any] | None = None,
) -> Pose5dEstimate:
    """Estimate center, axis, roll/pitch and a non-averaged C2 yaw pair."""
    document = load_pose5d_config() if config is None else config
    settings = document["estimator"]
    model = document.get("models", {}).get(object_id)
    if not isinstance(model, Mapping):
        raise ValueError(f"pose5d model registration is missing for {object_id}")
    axial_offset = float(model["registered_offset_m"])
    axial_quantile = float(settings["axial_low_quantile"])
    prior = _unit(np.asarray(axis_prior, dtype=np.float64), "axis_prior")
    raw = np.asarray(world_points_m, dtype=np.float64)
    raw_valid = int(np.sum(np.all(np.isfinite(raw), axis=1))) if raw.ndim == 2 else 0
    points = _robust_points(
        raw,
        float(settings["robust_radius_quantile"]),
        int(settings["maximum_point_count"]),
    )
    minimum = int(settings["minimum_point_count"])
    now = timestamp or datetime.now(timezone.utc).isoformat()
    if len(points) < minimum:
        return _rejected(object_id, frame_id, capture_id, now, len(points), depth_valid_ratio, "INSUFFICIENT_POINTS")

    center, axis, reference, radial_rmse, anisotropy = _fit_once(
        points,
        prior,
        axial_low_quantile=axial_quantile,
        axial_registered_offset_m=axial_offset,
        visible_cap_quantile=float(settings["visible_cap_quantile"]),
        minimum_cap_point_count=int(settings["minimum_cap_point_count"]),
    )
    prior_angle = _axis_angle(axis, prior)
    rng = np.random.default_rng(int(settings["random_seed"]) + sum(ord(c) for c in capture_id + object_id))
    centers = []
    axes = []
    sample_size = max(minimum, int(round(float(settings["bootstrap_fraction"]) * len(points))))
    for _ in range(int(settings["bootstrap_samples"])):
        indices = rng.integers(0, len(points), size=sample_size)
        try:
            sample_center, sample_axis, _, _, _ = _fit_once(
                points[indices],
                prior,
                axial_low_quantile=axial_quantile,
                axial_registered_offset_m=axial_offset,
                visible_cap_quantile=float(settings["visible_cap_quantile"]),
                minimum_cap_point_count=int(settings["minimum_cap_point_count"]),
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
        centers.append(sample_center)
        axes.append(sample_axis)
    if len(centers) < 8:
        return _rejected(object_id, frame_id, capture_id, now, len(points), depth_valid_ratio, "BOOTSTRAP_DEGENERATE")
    centers_array = np.asarray(centers)
    center_covariance = np.cov(centers_array, rowvar=False)
    lateral_variance = max(
        0.0,
        float(np.trace(center_covariance) - axis @ center_covariance @ axis),
    )
    lateral_std = math.sqrt(lateral_variance)
    angle_samples = np.asarray([_axis_angle(item, axis) for item in axes])
    axis_std = float(np.std(angle_samples, ddof=1))
    rotation_variance = axis_std * axis_std
    covariance6 = np.zeros((6, 6), dtype=np.float64)
    covariance6[:3, :3] = center_covariance
    covariance6[3, 3] = rotation_variance
    covariance6[4, 4] = rotation_variance
    # C2 yaw is explicitly ambiguous, so yaw variance is one half-turn squared.
    covariance6[5, 5] = (0.5 * math.pi) ** 2

    roll = math.atan2(-float(axis[1]), max(float(axis[2]), 1.0e-12))
    pitch = math.atan2(float(axis[0]), math.hypot(float(axis[1]), float(axis[2])))
    u, v = _basis_for_axis(axis)
    yaw = math.atan2(float(np.dot(reference, v)), float(np.dot(reference, u))) % (2.0 * math.pi)
    yaw_confidence = float(np.clip(anisotropy, 0.0, 1.0))
    hypotheses = (
        {"yaw_rad": yaw, "confidence": 0.5 * yaw_confidence},
        {"yaw_rad": (yaw + math.pi) % (2.0 * math.pi), "confidence": 0.5 * yaw_confidence},
    )
    reject = []
    if not 0.0 < lateral_authorization_gate_m:
        reject.append("LATERAL_GATE_UNAVAILABLE")
    elif lateral_std > float(settings["maximum_lateral_std_fraction_of_gate"]) * lateral_authorization_gate_m:
        reject.append("LATERAL_UNCERTAINTY_HIGH")
    if not 0.0 < axis_authorization_gate_rad:
        reject.append("AXIS_GATE_UNAVAILABLE")
    elif axis_std > float(settings["maximum_axis_std_fraction_of_gate"]) * axis_authorization_gate_rad:
        reject.append("AXIS_UNCERTAINTY_HIGH")
    if depth_valid_ratio < float(settings["minimum_depth_valid_ratio"]):
        reject.append("DEPTH_VALID_RATIO_LOW")
    if prior_angle > float(settings["maximum_axis_prior_angle_rad"]):
        reject.append("AXIS_PRIOR_DISAGREEMENT")
    if radial_rmse > float(settings["maximum_cylinder_radial_rmse_m"]):
        reject.append("CYLINDER_FIT_RMSE_HIGH")
    confidence_terms = (
        min(1.0, len(points) / max(1.0, 4.0 * minimum)),
        max(0.0, 1.0 - radial_rmse / float(settings["maximum_cylinder_radial_rmse_m"])),
        max(0.0, 1.0 - prior_angle / float(settings["maximum_axis_prior_angle_rad"])),
        max(0.0, min(1.0, depth_valid_ratio)),
    )
    confidence = float(np.prod(confidence_terms) ** (1.0 / len(confidence_terms)))
    return Pose5dEstimate(
        object_id=object_id,
        frame_id=frame_id,
        timestamp=now,
        capture_id=capture_id,
        xyz_m=tuple(float(value) for value in center),
        axis_vector=tuple(float(value) for value in axis),
        roll_rad=roll,
        pitch_rad=pitch,
        yaw_hypotheses=hypotheses,
        covariance_6x6=tuple(tuple(float(value) for value in row) for row in covariance6),
        lateral_position_std_m=lateral_std,
        axis_angle_std_rad=axis_std,
        point_count=int(len(points)),
        depth_valid_ratio=float(depth_valid_ratio),
        radial_fit_rmse_m=radial_rmse,
        planar_anisotropy=anisotropy,
        confidence=confidence,
        control_authorized=not reject,
        reject_reason=";".join(reject) if reject else None,
    )


def _rejected(object_id: str, frame_id: str, capture_id: str, timestamp: str, point_count: int, depth_valid_ratio: float, reason: str) -> Pose5dEstimate:
    nan3 = (math.nan, math.nan, math.nan)
    covariance = tuple(tuple(0.0 if row != column else math.inf for column in range(6)) for row in range(6))
    return Pose5dEstimate(
        object_id, frame_id, timestamp, capture_id, nan3, nan3, math.nan, math.nan,
        ({"yaw_rad": math.nan, "confidence": 0.0}, {"yaw_rad": math.nan, "confidence": 0.0}),
        covariance, math.inf, math.inf, point_count, float(depth_valid_ratio), math.inf, 0.0, 0.0, False, reason,
    )


def relative_pose5d(plug: Pose5dEstimate, receptacle: Pose5dEstimate) -> dict[str, Any]:
    """Build a same-capture relative observation without world truth."""
    same_capture = plug.capture_id == receptacle.capture_id
    if not same_capture:
        return {"control_authorized": False, "reject_reason": "UNSYNCHRONIZED_ENDPOINTS"}
    plug_center = np.asarray(plug.xyz_m)
    receptacle_center = np.asarray(receptacle.xyz_m)
    plug_axis = np.asarray(plug.axis_vector)
    receptacle_axis = np.asarray(receptacle.axis_vector)
    valid = bool(np.all(np.isfinite(plug_center)) and np.all(np.isfinite(receptacle_center)))
    axis_error = _axis_angle(plug_axis, receptacle_axis) if valid else math.inf
    delta = receptacle_center - plug_center if valid else np.full(3, math.nan)
    axial = float(np.dot(delta, receptacle_axis)) if valid else math.nan
    lateral = delta - axial * receptacle_axis if valid else np.full(3, math.nan)
    authorized = bool(valid and plug.control_authorized and receptacle.control_authorized)
    reasons = [item for item in (plug.reject_reason, receptacle.reject_reason) if item]
    return {
        "schema_version": "kcg_d38999_relative_pose5d_v1",
        "capture_id": plug.capture_id,
        "frame_id": plug.frame_id,
        "translation_receptacle_minus_plug_m": delta.tolist(),
        "lateral_vector_m": lateral.tolist(),
        "lateral_error_m": float(np.linalg.norm(lateral)),
        "axial_separation_m": axial,
        "axis_error_rad": axis_error,
        "control_authorized": authorized,
        "reject_reason": ";".join(reasons) if reasons else None,
        "truth_inputs": [],
    }


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "Pose5dEstimate",
    "estimate_pose5d",
    "load_pose5d_config",
    "relative_pose5d",
]
