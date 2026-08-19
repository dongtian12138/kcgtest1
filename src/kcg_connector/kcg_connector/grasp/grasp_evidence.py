"""Post-episode pose errors and paired grasp summaries.

This module is an evaluator.  It is intentionally not imported by the
sensor-bounded contact controller.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class PoseError6D:
    dx_m: float
    dy_m: float
    dz_m: float
    drx_rad: float
    dry_rad: float
    drz_rad: float

    def as_dict(self) -> dict[str, float]:
        return {
            "dx_m": self.dx_m,
            "dy_m": self.dy_m,
            "dz_m": self.dz_m,
            "drx_rad": self.drx_rad,
            "dry_rad": self.dry_rad,
            "drz_rad": self.drz_rad,
        }


def _transform(value: Sequence[Sequence[float]], label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (4, 4) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be one finite 4x4 transform")
    if not np.allclose(result[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-9):
        raise ValueError(f"{label} has an invalid homogeneous row")
    rotation = result[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-6):
        raise ValueError(f"{label} rotation is not orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1.0e-6):
        raise ValueError(f"{label} rotation is improper")
    return result


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    cosine = max(-1.0, min(1.0, (float(np.trace(rotation)) - 1.0) * 0.5))
    angle = math.acos(cosine)
    if angle < 1.0e-10:
        return np.zeros(3, dtype=np.float64)
    if math.pi - angle < 1.0e-6:
        eigenvalues, eigenvectors = np.linalg.eig(rotation)
        index = int(np.argmin(np.abs(eigenvalues - 1.0)))
        axis = np.real(eigenvectors[:, index])
        axis /= np.linalg.norm(axis)
        return axis * angle
    axis = np.asarray(
        (
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ),
        dtype=np.float64,
    ) / (2.0 * math.sin(angle))
    return axis * angle


def relative_pose_error(
    nominal_hand_to_plug: Sequence[Sequence[float]],
    actual_hand_to_plug: Sequence[Sequence[float]],
) -> PoseError6D:
    nominal = _transform(nominal_hand_to_plug, "nominal_hand_to_plug")
    actual = _transform(actual_hand_to_plug, "actual_hand_to_plug")
    delta = np.linalg.inv(nominal) @ actual
    rotation = _rotation_vector(delta[:3, :3])
    return PoseError6D(
        dx_m=float(delta[0, 3]),
        dy_m=float(delta[1, 3]),
        dz_m=float(delta[2, 3]),
        drx_rad=float(rotation[0]),
        dry_rad=float(rotation[1]),
        drz_rad=float(rotation[2]),
    )


def summarize(values: Iterable[float]) -> dict[str, float | int | None]:
    data = np.asarray(tuple(float(value) for value in values), dtype=np.float64)
    if data.size == 0:
        return {"count": 0, "mean": None, "median": None, "p95": None, "maximum": None}
    if not np.all(np.isfinite(data)):
        raise ValueError("summary data contains non-finite values")
    return {
        "count": int(data.size),
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "p95": float(np.percentile(data, 95)),
        "maximum": float(np.max(data)),
    }


def summarize_reports(reports: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = tuple(reports)
    successes = tuple(row for row in rows if row.get("grasp_success") is True)
    result: dict[str, Any] = {
        "episode_count": len(rows),
        "success_count": len(successes),
        "success_rate": len(successes) / len(rows) if rows else None,
        "failure_reasons": {},
    }
    for row in rows:
        if row.get("grasp_success") is True:
            continue
        reason = str(row.get("failure_reason") or "unknown")
        result["failure_reasons"][reason] = result["failure_reasons"].get(reason, 0) + 1
    for key in ("dx_m", "dy_m", "dz_m", "drx_rad", "dry_rad", "drz_rad"):
        result[key] = summarize(
            row["posthoc_pose_error"][key]
            for row in successes
            if isinstance(row.get("posthoc_pose_error"), Mapping)
            and row["posthoc_pose_error"].get(key) is not None
        )
    result["translation_norm_m"] = summarize(
        math.sqrt(sum(float(row["posthoc_pose_error"][key]) ** 2 for key in ("dx_m", "dy_m", "dz_m")))
        for row in successes
        if isinstance(row.get("posthoc_pose_error"), Mapping)
    )
    result["rotation_norm_rad"] = summarize(
        math.sqrt(sum(float(row["posthoc_pose_error"][key]) ** 2 for key in ("drx_rad", "dry_rad", "drz_rad")))
        for row in successes
        if isinstance(row.get("posthoc_pose_error"), Mapping)
    )
    return result


def load_json_reports(paths: Iterable[Path | str]) -> list[dict[str, Any]]:
    reports = []
    for raw in paths:
        path = Path(raw)
        reports.append(json.loads(path.read_text(encoding="utf-8")))
    return reports
