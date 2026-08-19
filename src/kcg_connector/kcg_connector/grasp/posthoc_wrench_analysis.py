"""Post-hoc E0 wrench diagnostics for the tabletop physical grasp.

Every function in this module is a pure, post-hoc diagnostic.  Nothing here
drives a command, feeds a safety gate, or changes an episode PASS result.
The formal runner may *record* the candidate scalars below as evidence, but
they are never allowed to gate the staged lift.

Frame honesty: the formal runner consumes ``canonical_wrench_sensor``, which
is the raw hand2arm reaction wrench with only the sign correction
``canonical_from_raw = -I`` applied.  That quantity lives in the handbase_link
sensor frame.  The ``connector_task_frame`` transform (``transform_wrench_to_task``)
is NOT applied by the formal lift path, so every wrench label here is
sensor-frame unless explicitly stated otherwise.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

WRIST_INDEX_FORCE = (0, 1, 2)
WRIST_INDEX_MOMENT = (3, 4, 5)

WRENCH_FRAME_NOTE = (
    "handbase_link canonical sensor frame (canonical_from_raw=-I sign "
    "correction only); connector_task_frame transform is NOT applied by the "
    "formal lift path"
)


def _wrench(value: Any, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (6,):
        raise ValueError(f"{label} must be one finite 6-vector")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be finite")
    return result


def force_norm(wrench: Sequence[float]) -> float:
    """Absolute sensor-frame force norm of one 6D wrench."""
    values = _wrench(wrench, "wrench")
    return float(np.linalg.norm(values[WRIST_INDEX_FORCE,]))


def moment_norm(wrench: Sequence[float]) -> float:
    """Absolute sensor-frame moment norm of one 6D wrench."""
    values = _wrench(wrench, "wrench")
    return float(np.linalg.norm(values[WRIST_INDEX_MOMENT,]))


def signed_delta(
    current: Sequence[float], reference: Sequence[float]
) -> np.ndarray:
    """Per-channel signed difference current - reference."""
    return _wrench(current, "current") - _wrench(reference, "reference")


def moment_magnitude_increase(
    current: Sequence[float], reference: Sequence[float]
) -> float:
    """Candidate gate quantity max(0, ||M_current|| - ||M_reference||).

    Diagnostic only.  This intentionally returns zero when the absolute
    moment norm decreases, so a normal unload (0.566 N*m -> 0.262 N*m) is
    not flagged by this candidate.
    """
    return max(
        0.0,
        moment_norm(current) - moment_norm(reference),
    )


def perpendicular_moment_delta(
    current: Sequence[float], reference: Sequence[float]
) -> dict[str, float | None]:
    """Decompose the moment delta into reference-radial and perpendicular parts.

    Returns a JSON-safe dictionary with the signed radial component (along the
    reference moment direction), the perpendicular component norm, and the
    angle between the delta and the reference moment direction.  When the
    reference moment norm is below the floor the decomposition degenerates to
    the full delta norm as the perpendicular part.
    """
    current_moment = _wrench(current, "current")[WRIST_INDEX_MOMENT,]
    reference_moment = _wrench(reference, "reference")[WRIST_INDEX_MOMENT,]
    delta = current_moment - reference_moment
    reference_norm = float(np.linalg.norm(reference_moment))
    if reference_norm <= 1.0e-9:
        return {
            "reference_moment_norm_nm": 0.0,
            "radial_component_nm": 0.0,
            "perpendicular_component_norm_nm": float(
                np.linalg.norm(delta)
            ),
            "delta_to_reference_angle_rad": None,
        }
    unit = reference_moment / reference_norm
    radial_scalar = float(delta @ unit)
    perpendicular = delta - radial_scalar * unit
    delta_norm = float(np.linalg.norm(delta))
    cosine = 0.0 if delta_norm <= 1.0e-12 else max(
        -1.0, min(1.0, radial_scalar / delta_norm)
    )
    return {
        "reference_moment_norm_nm": reference_norm,
        "radial_component_nm": radial_scalar,
        "perpendicular_component_norm_nm": float(
            np.linalg.norm(perpendicular)
        ),
        "delta_to_reference_angle_rad": math.acos(cosine),
    }


def reference_window_statistics(
    samples: Sequence[Sequence[float]],
    *,
    force_drift_bound_n: float | None = None,
    moment_drift_bound_nm: float | None = None,
) -> dict[str, Any]:
    """Per-channel and absolute-norm statistics over a reference window.

    Records per-channel mean/std, first-half vs second-half mean and their
    per-sample slope, and mean/std/extreme absolute force and moment norms.
    The bounded-drift indicators are evidence only: they never gate control.
    """
    data = np.asarray(samples, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 6:
        raise ValueError("reference samples must be an N x 6 array")
    if data.shape[0] < 2:
        raise ValueError("reference window needs at least two samples")
    if not np.all(np.isfinite(data)):
        raise ValueError("reference samples must be finite")
    count = int(data.shape[0])
    split = count // 2
    first_half = np.mean(data[:split], axis=0)
    second_half = np.mean(data[split:], axis=0)
    drift = second_half - first_half
    slope = drift / float(count - 1)
    force_norms = np.linalg.norm(data[:, WRIST_INDEX_FORCE,], axis=1)
    moment_norms = np.linalg.norm(data[:, WRIST_INDEX_MOMENT,], axis=1)
    result: dict[str, Any] = {
        "window_steps": count,
        "frame": WRENCH_FRAME_NOTE,
        "evidence_only": True,
        "per_channel": {
            "mean": [float(value) for value in np.mean(data, axis=0)],
            "std": [
                float(value)
                for value in np.std(data, axis=0, ddof=1 if count > 1 else 0)
            ],
        },
        "first_half_mean": [float(value) for value in first_half],
        "second_half_mean": [float(value) for value in second_half],
        "first_to_second_half_drift": [float(value) for value in drift],
        "first_to_second_half_slope_per_sample": [
            float(value) for value in slope
        ],
        "absolute_force_norm_n": _norm_statistics(force_norms),
        "absolute_moment_norm_nm": _norm_statistics(moment_norms),
    }
    indicators: dict[str, Any] = {}
    if force_drift_bound_n is not None:
        maximum = float(np.max(np.abs(drift[WRIST_INDEX_FORCE,])))
        indicators["force"] = {
            "bound_n": float(force_drift_bound_n),
            "maximum_absolute_drift_n": maximum,
            "within_bound": bool(maximum <= force_drift_bound_n),
        }
    if moment_drift_bound_nm is not None:
        maximum = float(np.max(np.abs(drift[WRIST_INDEX_MOMENT,])))
        indicators["moment"] = {
            "bound_nm": float(moment_drift_bound_nm),
            "maximum_absolute_drift_nm": maximum,
            "within_bound": bool(maximum <= moment_drift_bound_nm),
        }
    if indicators:
        indicators["gates_control"] = False
        result["bounded_drift_indicators"] = indicators
    return result


def channel_window_statistics(
    samples: Sequence[Sequence[float]],
    *,
    channel_names: Sequence[str] = ("f1", "f2", "f3"),
    frame: str = "finger_root_torque_proxy_post_tare",
    source: str = "sequential_consolidation_final_window",
    threshold_label: str = "SIM_TUNING_ONLY_A_CANDIDATE",
) -> dict[str, Any]:
    """Per-channel signed statistics for an N x 3 proxy window.

    Strictly for the three signed finger-root torque-proxy channels of the
    sequential consolidation final window: two-dimensional, at least two
    samples, exactly three finite channels.  Outputs per-channel
    mean/std/min/max, first/second-half means, their drift and per-sample
    slope plus window metadata.  No force/moment norms are computed here:
    wrist-specific quantities stay in reference_window_statistics.
    """
    data = np.asarray(samples, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("channel samples must be an N x 3 array")
    if data.shape[1] != 3:
        raise ValueError("channel samples must have exactly 3 channels")
    if data.shape[0] < 2:
        raise ValueError("channel window needs at least two samples")
    if not np.all(np.isfinite(data)):
        raise ValueError("channel samples must be finite")
    if len(tuple(channel_names)) != 3:
        raise ValueError("channel_names must contain three names")
    count = int(data.shape[0])
    split = count // 2
    first_half = np.mean(data[:split], axis=0)
    second_half = np.mean(data[split:], axis=0)
    drift = second_half - first_half
    slope = drift / float(count - 1)
    per_channel: dict[str, Any] = {}
    for index, name in enumerate(channel_names):
        values = data[:, index]
        per_channel[name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1 if count > 1 else 0)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "first_half_mean": float(first_half[index]),
            "second_half_mean": float(second_half[index]),
            "first_to_second_half_drift": float(drift[index]),
            "first_to_second_half_slope_per_sample": float(slope[index]),
        }
    return {
        "window_steps": count,
        "frame": str(frame),
        "source": str(source),
        "threshold_label": str(threshold_label),
        "evidence_only": True,
        "per_channel": per_channel,
    }


def window_statistics_block(
    samples: Sequence[Sequence[float]],
    *,
    channel_names: Sequence[str],
    frame: str,
    source: str,
    threshold_label: str = "SIM_TUNING_ONLY",
    baseline_subtraction: str | None = None,
    norm_summaries: bool = False,
) -> dict[str, Any]:
    """Evidence-only per-channel statistics over an existing sample window.

    Pure statistics over samples the formal runner already collected in its
    tare window: this helper never triggers a physics step, never changes
    sampling order, and its output is written to the report only -- it is
    never consumed by a controller, detector, recovery path, or PASS gate.
    std is the sample standard deviation (ddof=1).  baseline_subtraction
    records whether the input was centered first ("window_mean") or is the
    raw window (None).  Every emitted number is a finite JSON-safe float.
    """
    data = np.asarray(samples, dtype=np.float64)
    names = tuple(channel_names)
    if data.ndim != 2 or data.shape[1] != len(names):
        raise ValueError(
            "window samples must be an N x C array with C = "
            f"len(channel_names) = {len(names)}"
        )
    if data.shape[0] < 2:
        raise ValueError("window statistics need at least two samples")
    if not np.all(np.isfinite(data)):
        raise ValueError("window samples must be finite")
    if len(set(names)) != len(names) or not all(names):
        raise ValueError("channel names must be unique and non-empty")
    count = int(data.shape[0])
    split = count // 2
    first_half = np.mean(data[:split], axis=0)
    second_half = np.mean(data[split:], axis=0)
    per_channel: dict[str, Any] = {}
    for index, name in enumerate(names):
        column = data[:, index]
        per_channel[name] = {
            "mean": float(np.mean(column)),
            "std": float(np.std(column, ddof=1)),
            "rms": float(np.sqrt(np.mean(column * column))),
            "min": float(np.min(column)),
            "max": float(np.max(column)),
            "first_half_mean": float(first_half[index]),
            "second_half_mean": float(second_half[index]),
            "first_to_second_half_drift": float(
                second_half[index] - first_half[index]
            ),
        }
    result: dict[str, Any] = {
        "sample_count": count,
        "std_ddof": 1,
        "std_ddof_note": (
            "sample standard deviation (ddof=1) over the window"
        ),
        "threshold_label": threshold_label,
        "evidence_only": True,
        "baseline_subtraction": baseline_subtraction,
        "frame": frame,
        "source": source,
        "per_channel": per_channel,
    }
    if norm_summaries:
        if data.shape[1] != 6:
            raise ValueError(
                "norm summaries require exactly six wrench channels"
            )
        result["absolute_force_norm_n"] = _norm_statistics(
            np.linalg.norm(data[:, WRIST_INDEX_FORCE,], axis=1)
        )
        result["absolute_moment_norm_nm"] = _norm_statistics(
            np.linalg.norm(data[:, WRIST_INDEX_MOMENT,], axis=1)
        )
    return result


def _norm_statistics(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def _reference_wrench(report: Mapping[str, Any]) -> np.ndarray:
    value = report.get("formal_payload_wrist_reference")
    if value is None:
        raise ValueError(
            "report has no formal_payload_wrist_reference; cannot run E0 "
            "offline diagnostics"
        )
    return _wrench(value, "formal_payload_wrist_reference")


def _current_wrench(
    report: Mapping[str, Any],
) -> tuple[np.ndarray, int | None]:
    failure = report.get("formal_lift_failure")
    if not isinstance(failure, Mapping) or failure.get(
        "wrist_wrench_canonical"
    ) is None:
        raise ValueError(
            "report has no formal_lift_failure wrist_wrench_canonical; E0 "
            "requires the failed step's sensor-frame wrench"
        )
    global_step = failure.get("global_step")
    return (
        _wrench(
            failure["wrist_wrench_canonical"],
            "formal_lift_failure.wrist_wrench_canonical",
        ),
        int(global_step) if global_step is not None else None,
    )


def per_step_series(steps: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Extract diagnostic candidate series from formal lift step records."""
    series = []
    for record in steps:
        phase = record.get("phase")
        if not isinstance(phase, str) or not phase.startswith(
            "physical_grip_lift"
        ):
            continue
        canonical = record.get("wrist_wrench_canonical")
        if canonical is None:
            continue
        series.append((record, canonical))
    return [
        {
            "global_step": record.get("global_step"),
            "phase": record.get("phase"),
            "lift_stage": record.get("lift_stage"),
            "lift_stage_step": record.get("lift_stage_step"),
            "absolute_force_norm_n": force_norm(canonical),
            "absolute_moment_norm_nm": moment_norm(canonical),
        }
        for record, canonical in series
    ]


def analyze_episode(
    report: Mapping[str, Any],
    steps: Sequence[Mapping[str, Any]],
    *,
    plug_nut_mass_kg: float,
    gravity_m_s2: float = 9.81,
) -> dict[str, Any]:
    """Compute the frozen E0 diagnostics for one episode.

    Never mutates the inputs and never changes the original PASS result.
    """
    if (
        isinstance(plug_nut_mass_kg, bool)
        or not math.isfinite(float(plug_nut_mass_kg))
        or float(plug_nut_mass_kg) <= 0.0
    ):
        raise ValueError("plug_nut_mass_kg must be positive and finite")
    if not math.isfinite(float(gravity_m_s2)) or float(gravity_m_s2) <= 0.0:
        raise ValueError("gravity_m_s2 must be positive and finite")
    reference = _reference_wrench(report)
    current, failure_global_step = _current_wrench(report)
    delta = signed_delta(current, reference)
    series = per_step_series(steps)
    failure = report.get("formal_lift_failure")
    recorded_statistics = report.get(
        "formal_payload_wrist_reference_statistics"
    )
    plug_nut_weight = float(plug_nut_mass_kg) * float(gravity_m_s2)
    return {
        "posthoc_diagnostics_only": True,
        "changes_pass": False,
        "original_pass_unchanged": report.get("passed"),
        "wrench_frame_note": WRENCH_FRAME_NOTE,
        "reference_wrench_sensor_frame": [
            float(value) for value in reference
        ],
        "reference_absolute_force_norm_n": force_norm(reference),
        "reference_absolute_moment_norm_nm": moment_norm(reference),
        "current_wrench_sensor_frame": [
            float(value) for value in current
        ],
        "current_absolute_force_norm_n": force_norm(current),
        "current_absolute_moment_norm_nm": moment_norm(current),
        "signed_delta_sensor_frame": [float(value) for value in delta],
        "vector_delta_force_norm_n": float(
            np.linalg.norm(delta[WRIST_INDEX_FORCE,])
        ),
        "vector_delta_moment_norm_nm": float(
            np.linalg.norm(delta[WRIST_INDEX_MOMENT,])
        ),
        "moment_magnitude_increase_candidate_nm": (
            moment_magnitude_increase(current, reference)
        ),
        "perpendicular_moment_delta": perpendicular_moment_delta(
            current, reference
        ),
        "delta_fz_n": float(delta[2]),
        "plug_nut_mass_kg": float(plug_nut_mass_kg),
        "gravity_m_s2": float(gravity_m_s2),
        "plug_nut_weight_n": plug_nut_weight,
        "delta_fz_to_plug_nut_weight_ratio": float(
            delta[2] / plug_nut_weight
        ),
        "failure_global_step": failure_global_step,
        "failure_reason": (
            failure.get("reason")
            if isinstance(failure, Mapping)
            else None
        ),
        "reference_statistics_from_report": (
            recorded_statistics
            if isinstance(recorded_statistics, Mapping)
            else None
        ),
        "reference_statistics_available": isinstance(
            recorded_statistics, Mapping
        ),
        "per_step_series_source": "controller_steps.jsonl",
        "per_step_series": series,
    }
