"""B-V2-H16 bumpless restoration of the anchored arm position stiffness."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any, Mapping, Sequence

import numpy as np

from .pre_lift_arm_drive_compliance import minimum_jerk_blend


CONFIG_KEYS = (
    "enabled",
    "threshold_label",
    "source_h6_run_id",
    "source_h13_run_id",
    "transition_steps",
    "stability_window_steps",
    "expected_initial_stiffness_nm_rad",
    "expected_restored_stiffness_nm_rad",
    "expected_damping_nm_s_rad",
    "maximum_stiffness_step_nm_rad",
    "target_bias_step_guard",
    "offline_reference_maximum_target_bias_step_rad",
    "maximum_target_bias_rad",
    "maximum_position_effort_residual_nm",
    "maximum_gain_readback_error",
    "maximum_entry_moment_score_nm",
    "maximum_entry_load_imbalance",
)


@dataclass(frozen=True)
class PreLiftArmStiffnessRestorationConfig:
    """The one evidence-derived H16 diagnostic parameter set."""

    enabled: bool
    threshold_label: str
    source_h6_run_id: str
    source_h13_run_id: str
    transition_steps: int
    stability_window_steps: int
    expected_initial_stiffness_nm_rad: float
    expected_restored_stiffness_nm_rad: float
    expected_damping_nm_s_rad: float
    maximum_stiffness_step_nm_rad: float
    target_bias_step_guard: str
    offline_reference_maximum_target_bias_step_rad: float
    maximum_target_bias_rad: float
    maximum_position_effort_residual_nm: float
    maximum_gain_readback_error: float
    maximum_entry_moment_score_nm: float
    maximum_entry_load_imbalance: float

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("H16 enabled must be boolean")
        if self.threshold_label != "SIM_TUNING_ONLY_B_V2_H16":
            raise ValueError("H16 must stay SIM_TUNING_ONLY_B_V2_H16")
        if self.source_h6_run_id != "B-V2-GRASP-07":
            raise ValueError("H16 H6 source must stay B-V2-GRASP-07")
        if self.source_h13_run_id != "B-V2-GRASP-13":
            raise ValueError("H16 H13 source must stay B-V2-GRASP-13")
        if self.transition_steps != 240 or self.stability_window_steps != 240:
            raise ValueError("H16 transition and stability windows must each be 240 steps")
        if (
            self.target_bias_step_guard
            != "FRESH_ENTRY_PRELOAD_EXACT_FROZEN_SCHEDULE"
        ):
            raise ValueError(
                "H16 target-bias step guard must use the fresh entry preload "
                "and frozen schedule"
            )
        exact = {
            "expected_initial_stiffness_nm_rad": 6000.0,
            "expected_restored_stiffness_nm_rad": 24000.0,
            "expected_damping_nm_s_rad": 875.29978045654,
            "maximum_stiffness_step_nm_rad": 140.62,
            "offline_reference_maximum_target_bias_step_rad": 5.3e-5,
            "maximum_target_bias_rad": 0.01,
            "maximum_position_effort_residual_nm": 1.0e-6,
            "maximum_gain_readback_error": 1.0e-3,
            "maximum_entry_moment_score_nm": 0.24,
            "maximum_entry_load_imbalance": 0.18,
        }
        for name, expected in exact.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{name} must be numeric")
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            if not math.isclose(
                float(value), expected, rel_tol=0.0, abs_tol=0.0
            ):
                raise ValueError(
                    f"H16 {name} is frozen at the evidence-derived value {expected}"
                )
        if self.maximum_entry_moment_score_nm >= 0.30:
            raise ValueError("H16 entry moment score must stay below 0.30 N*m")


def load_pre_lift_arm_stiffness_restoration_config(
    value: Any,
) -> PreLiftArmStiffnessRestorationConfig:
    """Load H16 strictly; absence keeps historical contracts disabled."""

    defaults = {
        "enabled": False,
        "threshold_label": "SIM_TUNING_ONLY_B_V2_H16",
        "source_h6_run_id": "B-V2-GRASP-07",
        "source_h13_run_id": "B-V2-GRASP-13",
        "transition_steps": 240,
        "stability_window_steps": 240,
        "expected_initial_stiffness_nm_rad": 6000.0,
        "expected_restored_stiffness_nm_rad": 24000.0,
        "expected_damping_nm_s_rad": 875.29978045654,
        "maximum_stiffness_step_nm_rad": 140.62,
        "target_bias_step_guard": (
            "FRESH_ENTRY_PRELOAD_EXACT_FROZEN_SCHEDULE"
        ),
        "offline_reference_maximum_target_bias_step_rad": 5.3e-5,
        "maximum_target_bias_rad": 0.01,
        "maximum_position_effort_residual_nm": 1.0e-6,
        "maximum_gain_readback_error": 1.0e-3,
        "maximum_entry_moment_score_nm": 0.24,
        "maximum_entry_load_imbalance": 0.18,
    }
    if value is None:
        return PreLiftArmStiffnessRestorationConfig(**defaults)
    if not isinstance(value, Mapping):
        raise ValueError("pre_lift_arm_stiffness_restoration must be a mapping")
    unknown = sorted(set(value) - set(CONFIG_KEYS))
    missing = sorted(set(CONFIG_KEYS) - set(value))
    if unknown or missing:
        raise ValueError(
            "pre_lift_arm_stiffness_restoration has unknown keys "
            f"{unknown} and/or missing keys {missing}"
        )
    loaded = dict(value)
    for name in CONFIG_KEYS[6:]:
        if name != "target_bias_step_guard":
            loaded[name] = float(loaded[name])
    loaded["threshold_label"] = str(loaded["threshold_label"])
    loaded["source_h6_run_id"] = str(loaded["source_h6_run_id"])
    loaded["source_h13_run_id"] = str(loaded["source_h13_run_id"])
    loaded["target_bias_step_guard"] = str(
        loaded["target_bias_step_guard"]
    )
    return PreLiftArmStiffnessRestorationConfig(**loaded)


def _seven(values: Sequence[Real], label: str) -> np.ndarray:
    result = np.asarray(tuple(values), dtype=np.float64)
    if result.shape != (7,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must contain seven finite values")
    return result


def _validate_runtime_gains(
    initial_stiffness_nm_rad: Real,
    restored_stiffness_nm_rad: Real,
    damping_nm_s_rad: Real,
    config: PreLiftArmStiffnessRestorationConfig,
) -> tuple[float, float, float]:
    initial = float(initial_stiffness_nm_rad)
    restored = float(restored_stiffness_nm_rad)
    damping = float(damping_nm_s_rad)
    expected = (
        config.expected_initial_stiffness_nm_rad,
        config.expected_restored_stiffness_nm_rad,
        config.expected_damping_nm_s_rad,
    )
    actual = (initial, restored, damping)
    if not all(math.isfinite(value) for value in actual):
        raise ValueError("H16 runtime gains must be finite")
    if actual != expected:
        raise ValueError("H16 runtime gains do not match the one authorized set")
    return actual


def derive_runtime_target_bias_step_envelope(
    position_preload_nm: Sequence[Real],
    initial_stiffness_nm_rad: Real,
    restored_stiffness_nm_rad: Real,
    damping_nm_s_rad: Real,
    config: PreLiftArmStiffnessRestorationConfig,
) -> dict[str, Any]:
    """Derive and freeze the exact H16 bias-step envelope at fresh entry.

    The historical 5.3e-5 rad value came from a different process entry
    preload and is retained only as evidence.  The runtime envelope is not a
    tunable parameter: it is the exact maximum produced by the fresh preload
    and the already frozen 6000->24000 N*m/rad minimum-jerk schedule.  A
    second, preload-independent ceiling is derived from the frozen 0.01 rad
    total-bias limit, so the per-run derivation cannot relax that limit.
    """

    initial, restored, _ = _validate_runtime_gains(
        initial_stiffness_nm_rad,
        restored_stiffness_nm_rad,
        damping_nm_s_rad,
        config,
    )
    preload = _seven(position_preload_nm, "position_preload_nm")
    entry_bias = preload / initial
    maximum_entry_bias = float(np.max(np.abs(entry_bias)))
    if maximum_entry_bias > config.maximum_target_bias_rad:
        raise ValueError("H16 fresh entry bias exceeds the absolute bound")

    previous_bias = entry_bias
    maximum_step = 0.0
    peak_transition_step = -1
    maximum_stiffness_step = 0.0
    previous_stiffness = initial
    absolute_preload = initial * config.maximum_target_bias_rad
    previous_absolute_bias = absolute_preload / initial
    absolute_maximum_step = 0.0
    absolute_peak_transition_step = -1
    for transition_step in range(config.transition_steps):
        fraction = float(transition_step + 1) / float(
            config.transition_steps
        )
        blend = minimum_jerk_blend(fraction)
        stiffness = initial + (restored - initial) * blend
        stiffness_step = stiffness - previous_stiffness
        maximum_stiffness_step = max(
            maximum_stiffness_step, stiffness_step
        )
        bias = preload / stiffness
        step = float(np.max(np.abs(bias - previous_bias)))
        if step > maximum_step:
            maximum_step = step
            peak_transition_step = transition_step
        absolute_bias = absolute_preload / stiffness
        absolute_step = abs(absolute_bias - previous_absolute_bias)
        if absolute_step > absolute_maximum_step:
            absolute_maximum_step = absolute_step
            absolute_peak_transition_step = transition_step
        previous_bias = bias
        previous_absolute_bias = absolute_bias
        previous_stiffness = stiffness

    if maximum_stiffness_step > config.maximum_stiffness_step_nm_rad:
        raise ValueError("H16 frozen schedule exceeds its stiffness-step bound")
    roundoff_tolerance = float(
        16.0 * np.spacing(config.maximum_target_bias_rad)
    )
    if maximum_step > absolute_maximum_step + roundoff_tolerance:
        raise ValueError("H16 runtime bias-step envelope exceeds its absolute ceiling")
    return {
        "guard_kind": config.target_bias_step_guard,
        "entry_position_preload_nm": preload.tolist(),
        "maximum_entry_target_bias_rad": maximum_entry_bias,
        "runtime_target_bias_step_envelope_rad": maximum_step,
        "runtime_envelope_peak_transition_step": peak_transition_step,
        "absolute_schedule_target_bias_step_ceiling_rad": (
            absolute_maximum_step
        ),
        "absolute_ceiling_peak_transition_step": (
            absolute_peak_transition_step
        ),
        "comparison_roundoff_tolerance_rad": roundoff_tolerance,
        "offline_reference_maximum_target_bias_step_rad": (
            config.offline_reference_maximum_target_bias_step_rad
        ),
        "offline_reference_exceeded": bool(
            maximum_step
            > config.offline_reference_maximum_target_bias_step_rad
        ),
        "maximum_stiffness_step_nm_rad": maximum_stiffness_step,
    }


def derive_stiffness_restoration_step(
    realized_arm_rad: Sequence[Real],
    position_preload_nm: Sequence[Real],
    initial_stiffness_nm_rad: Real,
    restored_stiffness_nm_rad: Real,
    damping_nm_s_rad: Real,
    transition_step: int,
    previous_target_bias_rad: Sequence[Real],
    config: PreLiftArmStiffnessRestorationConfig,
    *,
    runtime_target_bias_step_envelope_rad: Real,
) -> dict[str, Any]:
    """Return one bumpless H16 stiffness step around current robot state."""

    if type(transition_step) is not int or not (
        0 <= transition_step < config.transition_steps
    ):
        raise ValueError("transition_step is outside the configured H16 transition")
    initial, restored, damping = _validate_runtime_gains(
        initial_stiffness_nm_rad,
        restored_stiffness_nm_rad,
        damping_nm_s_rad,
        config,
    )
    realized = _seven(realized_arm_rad, "realized_arm_rad")
    preload = _seven(position_preload_nm, "position_preload_nm")
    previous_bias = _seven(
        previous_target_bias_rad, "previous_target_bias_rad"
    )
    envelope = derive_runtime_target_bias_step_envelope(
        preload,
        initial,
        restored,
        damping,
        config,
    )
    runtime_envelope = float(runtime_target_bias_step_envelope_rad)
    if not math.isfinite(runtime_envelope) or runtime_envelope <= 0.0:
        raise ValueError("H16 runtime target-bias step envelope must be positive")
    if not math.isclose(
        runtime_envelope,
        float(envelope["runtime_target_bias_step_envelope_rad"]),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("H16 runtime target-bias step envelope changed after entry")
    fraction = float(transition_step + 1) / float(config.transition_steps)
    blend = minimum_jerk_blend(fraction)
    stiffness = initial + (restored - initial) * blend
    previous_fraction = float(transition_step) / float(config.transition_steps)
    previous_blend = minimum_jerk_blend(previous_fraction)
    previous_stiffness = initial + (restored - initial) * previous_blend
    stiffness_step = stiffness - previous_stiffness
    if stiffness_step > config.maximum_stiffness_step_nm_rad:
        raise ValueError("H16 stiffness step exceeds its derived bound")
    bias = preload / stiffness
    maximum_bias = float(np.max(np.abs(bias)))
    if maximum_bias > config.maximum_target_bias_rad:
        raise ValueError("H16 target bias exceeds the H6 internal bound")
    bias_step = bias - previous_bias
    maximum_bias_step = float(np.max(np.abs(bias_step)))
    if maximum_bias_step > runtime_envelope + float(
        envelope["comparison_roundoff_tolerance_rad"]
    ):
        raise ValueError("H16 target bias step exceeds its derived bound")
    target = realized + bias
    effort = stiffness * (target - realized)
    residual = effort - preload
    maximum_residual = float(np.max(np.abs(residual)))
    if maximum_residual > config.maximum_position_effort_residual_nm:
        raise ValueError("H16 position-effort continuity residual exceeds its bound")
    return {
        "fraction": fraction,
        "minimum_jerk_blend": blend,
        "applied_stiffness_nm_rad": stiffness,
        "applied_damping_nm_s_rad": damping,
        "stiffness_step_nm_rad": stiffness_step,
        "path_arm_rad": realized.tolist(),
        "target_arm_rad": target.tolist(),
        "target_bias_rad": bias.tolist(),
        "target_bias_step_rad": bias_step.tolist(),
        "maximum_target_bias_rad": maximum_bias,
        "maximum_target_bias_step_rad": maximum_bias_step,
        "runtime_target_bias_step_envelope_rad": runtime_envelope,
        "absolute_schedule_target_bias_step_ceiling_rad": envelope[
            "absolute_schedule_target_bias_step_ceiling_rad"
        ],
        "position_preload_nm": preload.tolist(),
        "position_effort_nm": effort.tolist(),
        "position_effort_residual_nm": residual.tolist(),
        "maximum_position_effort_residual_nm": maximum_residual,
    }


def restored_nominal_drive_target(
    path_arm_rad: Sequence[Real],
    position_preload_nm: Sequence[Real],
    restored_stiffness_nm_rad: Real,
    damping_nm_s_rad: Real,
    config: PreLiftArmStiffnessRestorationConfig,
) -> dict[str, Any]:
    """Freeze one anchored nominal-stiffness target after H16 transition."""

    _, restored, damping = _validate_runtime_gains(
        config.expected_initial_stiffness_nm_rad,
        restored_stiffness_nm_rad,
        damping_nm_s_rad,
        config,
    )
    path = _seven(path_arm_rad, "path_arm_rad")
    preload = _seven(position_preload_nm, "position_preload_nm")
    bias = preload / restored
    target = path + bias
    residual = restored * (target - path) - preload
    maximum_residual = float(np.max(np.abs(residual)))
    if maximum_residual > config.maximum_position_effort_residual_nm:
        raise ValueError("H16 restored position-effort residual exceeds its bound")
    return {
        "path_arm_rad": path.tolist(),
        "target_arm_rad": target.tolist(),
        "target_bias_rad": bias.tolist(),
        "maximum_target_bias_rad": float(np.max(np.abs(bias))),
        "applied_stiffness_nm_rad": restored,
        "applied_damping_nm_s_rad": damping,
        "position_preload_nm": preload.tolist(),
        "position_effort_residual_nm": residual.tolist(),
        "maximum_position_effort_residual_nm": maximum_residual,
    }


__all__ = [
    "CONFIG_KEYS",
    "PreLiftArmStiffnessRestorationConfig",
    "derive_runtime_target_bias_step_envelope",
    "derive_stiffness_restoration_step",
    "load_pre_lift_arm_stiffness_restoration_config",
    "restored_nominal_drive_target",
]
