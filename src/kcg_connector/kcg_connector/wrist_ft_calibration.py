"""Pure analysis helpers for virtual wrist-wrench axis calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


WRENCH_AXIS_NAMES = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")
_CASE_NAMES = ("plus_full", "minus_full", "plus_half", "minus_half")


@dataclass(frozen=True)
class AxisCalibrationThresholds:
    """Acceptance limits for the signed axis-calibration response."""

    minimum_absolute_gain: float = 0.85
    maximum_absolute_gain: float = 1.15
    maximum_cross_axis_ratio: float = 0.05
    maximum_odd_symmetry_error_ratio: float = 0.05
    maximum_linearity_error_ratio: float = 0.05


def _vector(values: Sequence[float], label: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (6,):
        raise ValueError(f"{label} must have shape (6,), got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must contain only finite values")
    return result


def analyze_axis_calibration(
    case_deltas: Mapping[str, Mapping[str, Sequence[float]]],
    *,
    force_magnitude_n: float,
    torque_magnitude_nm: float,
    half_scale: float = 0.5,
    thresholds: AxisCalibrationThresholds | None = None,
) -> dict:
    """Infer raw-to-canonical sign/permutation and validate +/- loads.

    ``case_deltas`` contains the measured raw reaction-wrench change for four
    physically applied load cases on each canonical axis.  Force columns are
    normalized by ``force_magnitude_n`` and torque columns by
    ``torque_magnitude_nm``.  Cross-axis ratios are evaluated only within the
    force or torque triplet so that unlike physical units are never divided.
    """

    limits = thresholds or AxisCalibrationThresholds()
    magnitudes = np.asarray(
        [force_magnitude_n] * 3 + [torque_magnitude_nm] * 3,
        dtype=np.float64,
    )
    if not np.all(np.isfinite(magnitudes)) or np.any(magnitudes <= 0.0):
        raise ValueError(
            "force and torque magnitudes must be finite and positive"
        )
    if not np.isfinite(half_scale) or not 0.0 < half_scale < 1.0:
        raise ValueError("half_scale must be finite and between zero and one")
    if set(case_deltas) != set(WRENCH_AXIS_NAMES):
        raise ValueError(
            "case_deltas must contain exactly the six wrench axes"
        )

    full_sensitivity = np.zeros((6, 6), dtype=np.float64)
    half_sensitivity = np.zeros((6, 6), dtype=np.float64)
    full_even = np.zeros((6, 6), dtype=np.float64)
    half_even = np.zeros((6, 6), dtype=np.float64)
    for column, axis_name in enumerate(WRENCH_AXIS_NAMES):
        axis_cases = case_deltas[axis_name]
        if set(axis_cases) != set(_CASE_NAMES):
            raise ValueError(
                f"{axis_name} must contain exactly {', '.join(_CASE_NAMES)}"
            )
        vectors = {
            name: _vector(values, f"{axis_name}.{name}")
            for name, values in axis_cases.items()
        }
        magnitude = magnitudes[column]
        full_sensitivity[:, column] = (
            vectors["plus_full"] - vectors["minus_full"]
        ) / (2.0 * magnitude)
        half_sensitivity[:, column] = (
            vectors["plus_half"] - vectors["minus_half"]
        ) / (2.0 * half_scale * magnitude)
        full_even[:, column] = (
            vectors["plus_full"] + vectors["minus_full"]
        ) / (2.0 * magnitude)
        half_even[:, column] = (
            vectors["plus_half"] + vectors["minus_half"]
        ) / (2.0 * half_scale * magnitude)

    mapping = np.zeros((6, 6), dtype=np.float64)
    records = []
    dominant_rows = []
    for column, axis_name in enumerate(WRENCH_AXIS_NAMES):
        group_start = 0 if column < 3 else 3
        group = np.arange(group_start, group_start + 3)
        group_values = np.abs(full_sensitivity[group, column])
        dominant_row = int(group[int(np.argmax(group_values))])
        dominant_rows.append(dominant_row)
        dominant_gain = float(full_sensitivity[dominant_row, column])
        absolute_gain = abs(dominant_gain)
        mapping[column, dominant_row] = 1.0 if dominant_gain > 0.0 else -1.0
        other_rows = group[group != dominant_row]
        denominator = max(absolute_gain, np.finfo(np.float64).eps)
        cross_axis_ratio = float(
            np.max(np.abs(full_sensitivity[other_rows, column]))
            / denominator
        )
        odd_symmetry_error_ratio = float(
            np.max(np.abs(full_even[group, column])) / denominator
        )
        half_odd_symmetry_error_ratio = float(
            np.max(np.abs(half_even[group, column])) / denominator
        )
        linearity_error_ratio = float(
            np.max(
                np.abs(
                    half_sensitivity[group, column]
                    - full_sensitivity[group, column]
                )
            )
            / denominator
        )
        opposite_group = np.arange(3, 6) if column < 3 else np.arange(0, 3)
        opposite_kind_max = float(
            np.max(np.abs(full_sensitivity[opposite_group, column]))
        )
        axis_passed = bool(
            limits.minimum_absolute_gain <= absolute_gain
            <= limits.maximum_absolute_gain
            and cross_axis_ratio <= limits.maximum_cross_axis_ratio
            and odd_symmetry_error_ratio
            <= limits.maximum_odd_symmetry_error_ratio
            and half_odd_symmetry_error_ratio
            <= limits.maximum_odd_symmetry_error_ratio
            and linearity_error_ratio <= limits.maximum_linearity_error_ratio
        )
        records.append(
            {
                "canonical_axis": axis_name,
                "dominant_raw_axis": WRENCH_AXIS_NAMES[dominant_row],
                "raw_gain": dominant_gain,
                "raw_to_canonical_sign": (
                    1 if dominant_gain > 0.0 else -1
                ),
                "same_kind_cross_axis_ratio": cross_axis_ratio,
                "full_odd_symmetry_error_ratio": odd_symmetry_error_ratio,
                "half_odd_symmetry_error_ratio": (
                    half_odd_symmetry_error_ratio
                ),
                "half_vs_full_linearity_error_ratio": (
                    linearity_error_ratio
                ),
                "opposite_kind_max_sensitivity": opposite_kind_max,
                "passed": axis_passed,
            }
        )

    force_rows_unique = len(set(dominant_rows[:3])) == 3
    torque_rows_unique = len(set(dominant_rows[3:])) == 3
    mapping_is_signed_permutation = bool(
        force_rows_unique
        and torque_rows_unique
        and np.all(np.sum(np.abs(mapping), axis=0) == 1.0)
        and np.all(np.sum(np.abs(mapping), axis=1) == 1.0)
    )
    passed = bool(
        mapping_is_signed_permutation
        and all(record["passed"] for record in records)
    )
    absolute_gains = [abs(record["raw_gain"]) for record in records]
    return {
        "axis_names": list(WRENCH_AXIS_NAMES),
        "raw_from_applied_full_sensitivity": full_sensitivity.tolist(),
        "raw_from_applied_half_sensitivity": half_sensitivity.tolist(),
        "canonical_from_raw_sign_permutation": mapping.tolist(),
        "axis_records": records,
        "minimum_absolute_gain": min(absolute_gains),
        "maximum_absolute_gain": max(absolute_gains),
        "maximum_same_kind_cross_axis_ratio": max(
            record["same_kind_cross_axis_ratio"] for record in records
        ),
        "maximum_odd_symmetry_error_ratio": max(
            max(
                record["full_odd_symmetry_error_ratio"],
                record["half_odd_symmetry_error_ratio"],
            )
            for record in records
        ),
        "maximum_linearity_error_ratio": max(
            record["half_vs_full_linearity_error_ratio"]
            for record in records
        ),
        "mapping_is_signed_permutation": mapping_is_signed_permutation,
        "thresholds": {
            "minimum_absolute_gain": limits.minimum_absolute_gain,
            "maximum_absolute_gain": limits.maximum_absolute_gain,
            "maximum_cross_axis_ratio": limits.maximum_cross_axis_ratio,
            "maximum_odd_symmetry_error_ratio": (
                limits.maximum_odd_symmetry_error_ratio
            ),
            "maximum_linearity_error_ratio": (
                limits.maximum_linearity_error_ratio
            ),
        },
        "passed": passed,
    }
