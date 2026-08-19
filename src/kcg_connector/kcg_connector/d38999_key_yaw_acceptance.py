"""Pure-CPU acceptance gates for D38999 keyed-yaw estimates.

Both gates are evaluation-only and can never authorize robot control.  The
original hardware gate accepts only a measured mechanical clearance.  A
separate public-spec simulation gate reads the explicitly derived keyed-v2
tolerance profiles, so a simulation result can never masquerade as a measured
hardware result.
"""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any

import numpy as np

from kcg_connector.d38999_keyed_public_spec_v2 import (
    PLUG_MODEL_ID,
    load_keyed_public_spec_v2,
)


SCHEMA_VERSION = "kcg_d38999_key_yaw_acceptance_v1"
THRESHOLD_LABEL = "REAL_MEASURED_CLEARANCE_REQUIRED"
SIMULATION_SCHEMA_VERSION = "kcg_d38999_public_spec_key_yaw_acceptance_v1"
SIMULATION_THRESHOLD_LABEL = (
    "SPEC_TOLERANCE_DERIVED_CONSERVATIVE_SIMULATION_ONLY"
)
DEFAULT_MINIMUM_SAMPLES = 30
DEFAULT_SIMULATION_MINIMUM_SAMPLES = 1000
NUMERIC_EQUALITY_TOLERANCE_DEG = 1.0e-12


def _clearance_threshold(clearance_deg: Any) -> float | None:
    if clearance_deg is None:
        return None
    if isinstance(clearance_deg, bool) or not isinstance(clearance_deg, Real):
        raise ValueError("measured_allowable_clearance_deg must be a finite number or None")
    clearance = float(clearance_deg)
    if not math.isfinite(clearance) or clearance <= 0.0:
        raise ValueError("measured_allowable_clearance_deg must be finite and positive")
    return clearance / 2.0


def _minimum_sample_count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("minimum_samples must be an integer")
    count = int(value)
    if count < DEFAULT_MINIMUM_SAMPLES:
        raise ValueError(
            f"minimum_samples must be at least {DEFAULT_MINIMUM_SAMPLES}"
        )
    return count


def _yaw_arrays(estimated_yaw_rad: Any, truth_yaw_rad: Any) -> tuple[np.ndarray, np.ndarray]:
    try:
        estimated = np.asarray(estimated_yaw_rad, dtype=np.float64)
        truth = np.asarray(truth_yaw_rad, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("estimated_yaw_rad and truth_yaw_rad must be numeric") from exc
    if estimated.ndim != 1 or truth.ndim != 1:
        raise ValueError("estimated_yaw_rad and truth_yaw_rad must be one-dimensional")
    if estimated.shape != truth.shape:
        raise ValueError("estimated_yaw_rad and truth_yaw_rad must have the same shape")
    return estimated, truth


def _base_result(
    *,
    required_p95_deg: float | None,
    observed_p95_deg: float | None,
    sample_count: int,
    dataset_tag: str,
    withheld_truth: bool,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "OFFLINE_WITHHELD_YAW_EVALUATION_ONLY",
        "status": "REJECTED",
        "reason": None,
        "required_yaw_error_p95_deg": required_p95_deg,
        "observed_yaw_error_p95_deg": observed_p95_deg,
        "sample_count": sample_count,
        "dataset_tag": dataset_tag,
        "withheld_truth": withheld_truth,
        "passed": False,
        "shadow_authorized": False,
        "control_authorized": False,
        "threshold_label": THRESHOLD_LABEL,
        "authorization_scope": "EVALUATION_ONLY_NO_CONTROL",
    }


def evaluate_key_yaw_acceptance(
    estimated_yaw_rad: Any,
    truth_yaw_rad: Any,
    measured_allowable_clearance_deg: float | None,
    *,
    minimum_samples: int = DEFAULT_MINIMUM_SAMPLES,
    dataset_tag: str,
    withheld_truth: bool,
) -> dict[str, Any]:
    """Evaluate wrapped absolute yaw error against half the real clearance.

    Array shape/type mistakes and invalid configuration raise ``ValueError``.
    Evidence-quality failures return a structured, fail-closed result.  The
    threshold comparison is deliberately strict: ``p95 < clearance / 2``.
    """
    required_p95_deg = _clearance_threshold(measured_allowable_clearance_deg)
    required_samples = _minimum_sample_count(minimum_samples)
    if not isinstance(dataset_tag, str) or not dataset_tag.strip():
        raise ValueError("dataset_tag must be non-empty text")
    if type(withheld_truth) is not bool:
        raise ValueError("withheld_truth must be boolean")

    estimated, truth = _yaw_arrays(estimated_yaw_rad, truth_yaw_rad)
    sample_count = int(estimated.size)

    finite = bool(np.all(np.isfinite(estimated)) and np.all(np.isfinite(truth)))
    if finite and sample_count:
        wrapped_error_rad = (estimated - truth + math.pi) % (2.0 * math.pi) - math.pi
        observed_p95_deg = float(
            np.percentile(np.degrees(np.abs(wrapped_error_rad)), 95.0)
        )
    else:
        observed_p95_deg = None

    result = _base_result(
        required_p95_deg=required_p95_deg,
        observed_p95_deg=observed_p95_deg,
        sample_count=sample_count,
        dataset_tag=dataset_tag.strip(),
        withheld_truth=withheld_truth,
    )

    if not finite:
        result.update(
            status="REJECTED_NONFINITE_YAW",
            reason="YAW_INPUT_CONTAINS_NAN_OR_INFINITY",
        )
        return result
    if required_p95_deg is None:
        result.update(
            status="BLOCKED_REAL_CLEARANCE_UNKNOWN",
            reason="MEASURED_ALLOWABLE_CLEARANCE_REQUIRED",
        )
        return result
    if sample_count < required_samples:
        result.update(
            status="REJECTED_INSUFFICIENT_SAMPLES",
            reason="MINIMUM_SAMPLE_COUNT_NOT_MET",
        )
        return result
    if not withheld_truth:
        result.update(
            status="REJECTED_NOT_WITHHELD_TRUTH",
            reason="EVALUATION_DATASET_MUST_BE_WITHHELD",
        )
        return result

    # Degree/radian conversion can put a mathematically equal value a few ULPs
    # below the threshold.  Treat that numerical equality as failure so the
    # required strict inequality cannot be bypassed by floating-point noise.
    passed = bool(
        observed_p95_deg
        < required_p95_deg - NUMERIC_EQUALITY_TOLERANCE_DEG
    )
    result.update(
        status=("PASSED_EVALUATION_ONLY" if passed else "REJECTED_YAW_P95_THRESHOLD"),
        reason=(None if passed else "P95_MUST_BE_STRICTLY_BELOW_HALF_CLEARANCE"),
        passed=passed,
        shadow_authorized=passed,
    )
    return result


def evaluate_public_spec_sim_key_yaw_acceptance(
    estimated_yaw_rad: Any,
    truth_yaw_rad: Any,
    *,
    keyed_model_id: str,
    profile_name: str | None = None,
    minimum_samples: int = DEFAULT_SIMULATION_MINIMUM_SAMPLES,
    dataset_tag: str,
    withheld_truth: bool,
) -> dict[str, Any]:
    """Evaluate synthetic keyed-v2 yaw without inventing hardware evidence.

    ``profile_name`` defaults to the contract's adversarial GDT stress
    profile.  Passing authorizes only shadow evidence for this simulation
    identity; every control and hardware-qualification field stays false.
    """

    if keyed_model_id != PLUG_MODEL_ID:
        raise ValueError(f"keyed_model_id must be exactly {PLUG_MODEL_ID}")
    required_samples = _minimum_sample_count(minimum_samples)
    if required_samples < DEFAULT_SIMULATION_MINIMUM_SAMPLES:
        raise ValueError(
            "public-spec simulation minimum_samples must be at least "
            f"{DEFAULT_SIMULATION_MINIMUM_SAMPLES}"
        )
    if not isinstance(dataset_tag, str) or not dataset_tag.strip():
        raise ValueError("dataset_tag must be non-empty text")
    if type(withheld_truth) is not bool:
        raise ValueError("withheld_truth must be boolean")

    model = load_keyed_public_spec_v2()
    selected_profile = profile_name or model.simulation_acceptance_profile
    if not isinstance(selected_profile, str) or not selected_profile:
        raise ValueError("profile_name must be non-empty text or None")
    try:
        profile = model.clearance_profile(selected_profile)
    except KeyError as exc:
        raise ValueError(f"unknown public-spec yaw profile: {selected_profile}") from exc

    estimated, truth = _yaw_arrays(estimated_yaw_rad, truth_yaw_rad)
    sample_count = int(estimated.size)
    finite = bool(np.all(np.isfinite(estimated)) and np.all(np.isfinite(truth)))
    if finite and sample_count:
        wrapped_error_rad = (estimated - truth + math.pi) % (2.0 * math.pi) - math.pi
        observed_p95_deg = float(
            np.percentile(np.degrees(np.abs(wrapped_error_rad)), 95.0)
        )
    else:
        observed_p95_deg = None

    result: dict[str, Any] = {
        "schema_version": SIMULATION_SCHEMA_VERSION,
        "mode": "OFFLINE_WITHHELD_SYNTHETIC_YAW_EVALUATION_ONLY",
        "status": "REJECTED",
        "reason": None,
        "keyed_model_id": keyed_model_id,
        "profile_name": selected_profile,
        "clearance_kind": "peak_to_peak_virtual_yaw_window",
        "derived_peak_to_peak_clearance_deg": profile.peak_to_peak_deg,
        "required_yaw_error_p95_deg": profile.required_p95_deg,
        "clearance_derivation_kind": profile.derivation_kind,
        "drawing_specified_mechanical_yaw_clearance": (
            profile.drawing_specified_clearance
        ),
        "observed_yaw_error_p95_deg": observed_p95_deg,
        "sample_count": sample_count,
        "dataset_tag": dataset_tag.strip(),
        "withheld_truth": withheld_truth,
        "passed": False,
        "shadow_authorized": False,
        "selected_for_control_allowed": False,
        "simulation_insertion_control_authorized": False,
        "robot_control_authorized": False,
        "hardware_control_authorized": False,
        "real_measured_clearance_deg": None,
        "space_qualification_claimed": False,
        "threshold_label": SIMULATION_THRESHOLD_LABEL,
        "authorization_scope": "PUBLIC_SPEC_SIMULATION_SHADOW_ONLY_NO_CONTROL",
    }

    if not finite:
        result.update(
            status="REJECTED_NONFINITE_YAW",
            reason="YAW_INPUT_CONTAINS_NAN_OR_INFINITY",
        )
        return result
    if sample_count < required_samples:
        result.update(
            status="REJECTED_INSUFFICIENT_SAMPLES",
            reason="PUBLIC_SPEC_SIMULATION_MINIMUM_SAMPLE_COUNT_NOT_MET",
        )
        return result
    if not withheld_truth:
        result.update(
            status="REJECTED_NOT_WITHHELD_TRUTH",
            reason="SYNTHETIC_EVALUATION_DATASET_MUST_BE_WITHHELD",
        )
        return result

    passed = bool(
        observed_p95_deg
        < profile.required_p95_deg - NUMERIC_EQUALITY_TOLERANCE_DEG
    )
    result.update(
        status=(
            "PASSED_PUBLIC_SPEC_SIMULATION_SHADOW_ONLY"
            if passed
            else "REJECTED_PUBLIC_SPEC_SIM_YAW_P95_THRESHOLD"
        ),
        reason=(
            None
            if passed
            else "P95_MUST_BE_STRICTLY_BELOW_HALF_DERIVED_SIM_CLEARANCE"
        ),
        passed=passed,
        shadow_authorized=passed,
    )
    return result


__all__ = [
    "DEFAULT_MINIMUM_SAMPLES",
    "DEFAULT_SIMULATION_MINIMUM_SAMPLES",
    "NUMERIC_EQUALITY_TOLERANCE_DEG",
    "SCHEMA_VERSION",
    "SIMULATION_SCHEMA_VERSION",
    "SIMULATION_THRESHOLD_LABEL",
    "THRESHOLD_LABEL",
    "evaluate_key_yaw_acceptance",
    "evaluate_public_spec_sim_key_yaw_acceptance",
]
