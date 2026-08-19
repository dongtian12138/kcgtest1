"""Truth-free H23 differential-finger preload diagnostic mathematics.

The runtime performs the physical probe and raw safety monitoring.  This
module only locks the evidence-derived schedule and analyzes synchronized
baseline/probe means in the zero-common-mode three-finger subspace.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .pre_lift_arm_drive_compliance import minimum_jerk_blend
from .pre_lift_wrench_centering import PreLiftWrenchCenteringConfig
from .three_finger_sequential_grasp import SequentialGraspConfig


THRESHOLD_LABEL = "SIM_TUNING_ONLY_B_V2_H23_DIAGNOSTIC"
CONFIG_KEYS = (
    "enabled",
    "threshold_label",
    "source_h22_run_id",
    "diagnostic_only",
    "finger_probe_order",
    "probe_increment_source",
    "move_steps_source",
    "settle_steps_source",
    "sample_steps_source",
    "minimum_probe_response_source",
    "arm_target_fixed",
    "vertical_feedforward_zero_required",
    "payload_reference_rebase_forbidden",
    "raw_sensor_hard_gate_unchanged",
    "hard_gate_detection_delay_steps",
    "object_contact_event_truth_forbidden",
    "correction_applied_in_diagnostic",
)


@dataclass(frozen=True)
class DifferentialFingerPreloadDiagnosticConfig:
    enabled: bool
    threshold_label: str
    source_h22_run_id: str
    diagnostic_only: bool
    finger_probe_order: tuple[str, str, str]
    probe_increment_source: str
    move_steps_source: str
    settle_steps_source: str
    sample_steps_source: str
    minimum_probe_response_source: str
    arm_target_fixed: bool
    vertical_feedforward_zero_required: bool
    payload_reference_rebase_forbidden: bool
    raw_sensor_hard_gate_unchanged: bool
    hard_gate_detection_delay_steps: int
    object_contact_event_truth_forbidden: bool
    correction_applied_in_diagnostic: bool

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("H23 diagnostic enabled must be boolean")
        expected_text = {
            "threshold_label": THRESHOLD_LABEL,
            "source_h22_run_id": "B-V2-H22-PREUNLOADING-NULLING-01",
            "probe_increment_source": "SEQUENTIAL_PROBE_INCREMENT_RAD",
            "move_steps_source": "H4_PROBE_MOVE_STEPS",
            "settle_steps_source": "SEQUENTIAL_PROBE_SETTLE_STEPS",
            "sample_steps_source": "H4_PROBE_SAMPLE_STEPS",
            "minimum_probe_response_source": (
                "SEQUENTIAL_MINIMUM_PROBE_RESPONSE_NM"
            ),
        }
        for name, expected in expected_text.items():
            if getattr(self, name) != expected:
                raise ValueError(f"H23 diagnostic {name} must remain {expected}")
        if self.finger_probe_order != ("f1", "f2", "f3"):
            raise ValueError("H23 diagnostic finger order must remain f1,f2,f3")
        for name in (
            "diagnostic_only",
            "arm_target_fixed",
            "vertical_feedforward_zero_required",
            "payload_reference_rebase_forbidden",
            "raw_sensor_hard_gate_unchanged",
            "object_contact_event_truth_forbidden",
        ):
            if getattr(self, name) is not True:
                raise ValueError(f"H23 diagnostic {name} must remain true")
        if self.correction_applied_in_diagnostic is not False:
            raise ValueError(
                "H23 diagnostic cannot apply the prospective correction"
            )
        if type(self.hard_gate_detection_delay_steps) is not int:
            raise ValueError("H23 diagnostic hard-gate delay must be integer")
        if self.hard_gate_detection_delay_steps != 0:
            raise ValueError("H23 diagnostic raw hard-gate delay must remain zero")


def load_differential_finger_preload_diagnostic_config(
    value: Any,
) -> DifferentialFingerPreloadDiagnosticConfig:
    defaults = {
        "enabled": False,
        "threshold_label": THRESHOLD_LABEL,
        "source_h22_run_id": "B-V2-H22-PREUNLOADING-NULLING-01",
        "diagnostic_only": True,
        "finger_probe_order": ("f1", "f2", "f3"),
        "probe_increment_source": "SEQUENTIAL_PROBE_INCREMENT_RAD",
        "move_steps_source": "H4_PROBE_MOVE_STEPS",
        "settle_steps_source": "SEQUENTIAL_PROBE_SETTLE_STEPS",
        "sample_steps_source": "H4_PROBE_SAMPLE_STEPS",
        "minimum_probe_response_source": (
            "SEQUENTIAL_MINIMUM_PROBE_RESPONSE_NM"
        ),
        "arm_target_fixed": True,
        "vertical_feedforward_zero_required": True,
        "payload_reference_rebase_forbidden": True,
        "raw_sensor_hard_gate_unchanged": True,
        "hard_gate_detection_delay_steps": 0,
        "object_contact_event_truth_forbidden": True,
        "correction_applied_in_diagnostic": False,
    }
    if value is None:
        return DifferentialFingerPreloadDiagnosticConfig(**defaults)
    if not isinstance(value, Mapping):
        raise ValueError(
            "pre_lift_differential_finger_preload_diagnostic must be a mapping"
        )
    unknown = sorted(set(value) - set(CONFIG_KEYS))
    missing = sorted(set(CONFIG_KEYS) - set(value))
    if unknown or missing:
        raise ValueError(
            "pre_lift_differential_finger_preload_diagnostic has unknown keys "
            f"{unknown} and/or missing keys {missing}"
        )
    order = value["finger_probe_order"]
    if not isinstance(order, (list, tuple)) or len(order) != 3:
        raise ValueError("H23 diagnostic finger_probe_order must have 3 entries")
    return DifferentialFingerPreloadDiagnosticConfig(
        enabled=value["enabled"],
        threshold_label=str(value["threshold_label"]),
        source_h22_run_id=str(value["source_h22_run_id"]),
        diagnostic_only=value["diagnostic_only"],
        finger_probe_order=tuple(str(item) for item in order),
        probe_increment_source=str(value["probe_increment_source"]),
        move_steps_source=str(value["move_steps_source"]),
        settle_steps_source=str(value["settle_steps_source"]),
        sample_steps_source=str(value["sample_steps_source"]),
        minimum_probe_response_source=str(
            value["minimum_probe_response_source"]
        ),
        arm_target_fixed=value["arm_target_fixed"],
        vertical_feedforward_zero_required=(
            value["vertical_feedforward_zero_required"]
        ),
        payload_reference_rebase_forbidden=(
            value["payload_reference_rebase_forbidden"]
        ),
        raw_sensor_hard_gate_unchanged=(
            value["raw_sensor_hard_gate_unchanged"]
        ),
        hard_gate_detection_delay_steps=(
            value["hard_gate_detection_delay_steps"]
        ),
        object_contact_event_truth_forbidden=(
            value["object_contact_event_truth_forbidden"]
        ),
        correction_applied_in_diagnostic=(
            value["correction_applied_in_diagnostic"]
        ),
    )


def derive_probe_contract(
    config: DifferentialFingerPreloadDiagnosticConfig,
    sequential: SequentialGraspConfig,
    h4: PreLiftWrenchCenteringConfig,
) -> dict[str, Any]:
    """Derive every numeric H23 value from an already checked-in constant."""

    if not config.enabled:
        raise ValueError("H23 probe contract requires enabled diagnostic")
    if h4.enabled:
        raise ValueError("H23 cannot reactivate the rejected H4 controller")
    amplitude = float(sequential.probe_increment_rad)
    move_steps = int(h4.probe_move_steps)
    settle_steps = int(sequential.probe_settle_steps)
    sample_steps = int(h4.probe_sample_steps)
    minimum_response = float(sequential.minimum_probe_response_nm)
    if min(amplitude, minimum_response) <= 0.0:
        raise ValueError("H23 inherited probe values must be positive")
    total_steps = sample_steps + 3 * (
        move_steps + sample_steps + move_steps + settle_steps
    )
    return {
        "finger_probe_order": list(config.finger_probe_order),
        "probe_increment_rad": amplitude,
        "probe_increment_source": config.probe_increment_source,
        "probe_move_steps": move_steps,
        "probe_move_steps_source": config.move_steps_source,
        "settle_steps": settle_steps,
        "settle_steps_source": config.settle_steps_source,
        "sample_steps": sample_steps,
        "sample_steps_source": config.sample_steps_source,
        "initial_baseline_steps": sample_steps,
        "minimum_probe_response_nm": minimum_response,
        "minimum_probe_response_source": (
            config.minimum_probe_response_source
        ),
        "objective_force_scale_n": float(h4.objective_force_scale_n),
        "objective_moment_scale_nm": float(h4.objective_moment_scale_nm),
        "damping_ratio": float(h4.damping_ratio),
        "prospective_correction_norm_limit_rad": amplitude,
        "total_expected_steps": total_steps,
        "required_differential_rank": 2,
        "arm_target_fixed": True,
        "vertical_feedforward_n": 0.0,
        "raw_sensor_hard_gate_unchanged": True,
        "hard_gate_detection_delay_steps": 0,
        "diagnostic_only": True,
        "correction_applied": False,
    }


def probe_offset_rad(
    step_index: int,
    move_steps: int,
    amplitude_rad: float,
    *,
    returning: bool,
) -> dict[str, float | int | bool]:
    """Return a bounded minimum-jerk positive probe or its exact return."""

    if type(step_index) is not int or type(move_steps) is not int:
        raise ValueError("H23 probe indices must be integers")
    if not 0 <= step_index < move_steps or move_steps <= 0:
        raise ValueError("H23 probe step is outside its move interval")
    if not math.isfinite(float(amplitude_rad)) or amplitude_rad <= 0.0:
        raise ValueError("H23 probe amplitude must be positive and finite")
    fraction = float(step_index + 1) / float(move_steps)
    blend = minimum_jerk_blend(fraction)
    offset = float(amplitude_rad) * (1.0 - blend if returning else blend)
    return {
        "step_index": step_index,
        "move_steps": move_steps,
        "minimum_jerk_blend": blend,
        "offset_rad": offset,
        "positive_closure_only": True,
        "returning": returning,
    }


def _matrix(
    values: Sequence[Sequence[float]], shape: tuple[int, int], label: str
) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be a finite {shape} matrix")
    return result


def analyze_differential_probe(
    *,
    baseline_objectives: Sequence[Sequence[float]],
    probe_objectives: Sequence[Sequence[float]],
    baseline_root_loads_nm: Sequence[Sequence[float]],
    probe_root_loads_nm: Sequence[Sequence[float]],
    probe_increment_rad: float | Sequence[float],
    minimum_probe_response_nm: float,
    damping_ratio: float,
    prospective_correction_norm_limit_rad: float,
) -> dict[str, Any]:
    """Analyze three one-sided probes without applying any correction."""

    baselines = _matrix(baseline_objectives, (4, 4), "H23 baselines")
    probes = _matrix(probe_objectives, (3, 4), "H23 probes")
    baseline_loads = _matrix(
        baseline_root_loads_nm, (4, 3), "H23 baseline root loads"
    )
    probe_loads = _matrix(
        probe_root_loads_nm, (3, 3), "H23 probe root loads"
    )
    increment_values = np.asarray(probe_increment_rad, dtype=np.float64)
    if increment_values.ndim == 0:
        probe_increments = np.full(3, float(increment_values), dtype=np.float64)
    elif increment_values.shape == (3,):
        probe_increments = increment_values.copy()
    else:
        raise ValueError(
            "H23 probe_increment_rad must be one scalar or three values"
        )
    if not np.all(np.isfinite(probe_increments)) or np.any(
        probe_increments <= 0.0
    ):
        raise ValueError(
            "H23 probe_increment_rad values must be positive and finite"
        )
    for name, value in (
        ("minimum_probe_response_nm", minimum_probe_response_nm),
        ("damping_ratio", damping_ratio),
        (
            "prospective_correction_norm_limit_rad",
            prospective_correction_norm_limit_rad,
        ),
    ):
        if not math.isfinite(float(value)) or value <= 0.0:
            raise ValueError(f"H23 {name} must be positive and finite")

    adjacent_baseline = 0.5 * (baselines[:-1] + baselines[1:])
    jacobian = (
        (probes - adjacent_baseline) / probe_increments[:, np.newaxis]
    ).T
    adjacent_load_baseline = 0.5 * (
        baseline_loads[:-1] + baseline_loads[1:]
    )
    root_response = np.asarray(
        [
            abs(probe_loads[index, index] - adjacent_load_baseline[index, index])
            for index in range(3)
        ],
        dtype=np.float64,
    )
    minimum_response_passed = bool(
        np.all(root_response >= minimum_probe_response_nm)
    )

    sqrt2 = math.sqrt(2.0)
    sqrt6 = math.sqrt(6.0)
    differential_basis = np.asarray(
        (
            (1.0 / sqrt2, 1.0 / sqrt6),
            (-1.0 / sqrt2, 1.0 / sqrt6),
            (0.0, -2.0 / sqrt6),
        ),
        dtype=np.float64,
    )
    differential_jacobian = jacobian @ differential_basis
    left, singular_values, right_t = np.linalg.svd(
        differential_jacobian, full_matrices=False
    )
    rank_threshold = (
        float(singular_values[0]) * 1.0e-6
        if singular_values[0] > 0.0
        else 0.0
    )
    rank = int(np.count_nonzero(singular_values > rank_threshold))
    condition_number = (
        float(singular_values[0] / singular_values[-1])
        if singular_values[-1] > 0.0
        else math.inf
    )

    final_objective = baselines[-1]
    damping = float(damping_ratio) * float(singular_values[0])
    if rank == 2:
        gains = singular_values / (
            singular_values * singular_values + damping * damping
        )
        differential_coordinates = -(
            right_t.T @ (gains * (left.T @ final_objective))
        )
        prospective = differential_basis @ differential_coordinates
    else:
        prospective = np.zeros(3, dtype=np.float64)
    unbounded_norm = float(np.linalg.norm(prospective))
    clipped = False
    if unbounded_norm > prospective_correction_norm_limit_rad:
        prospective *= (
            prospective_correction_norm_limit_rad / unbounded_norm
        )
        clipped = True
    predicted = final_objective + jacobian @ prospective
    return {
        "objective_components": [
            "task_Fx_over_8N",
            "task_Fy_over_8N",
            "sensor_origin_Mx_increment_over_0.30Nm",
            "sensor_origin_My_increment_over_0.30Nm",
        ],
        "realized_probe_increments_rad": probe_increments.tolist(),
        "probe_increment_normalization": "per_finger_realized_increment",
        "jacobian_per_rad": jacobian.tolist(),
        "differential_basis": differential_basis.tolist(),
        "differential_jacobian_per_rad": differential_jacobian.tolist(),
        "singular_values_per_rad": singular_values.tolist(),
        "rank_threshold": rank_threshold,
        "differential_rank": rank,
        "differential_condition_number": condition_number,
        "root_probe_response_nm": root_response.tolist(),
        "minimum_probe_response_nm": float(minimum_probe_response_nm),
        "minimum_probe_response_passed": minimum_response_passed,
        "final_baseline_objective": final_objective.tolist(),
        "prospective_correction_closure_rad": prospective.tolist(),
        "prospective_correction_sum_rad": float(np.sum(prospective)),
        "prospective_correction_unbounded_norm_rad": unbounded_norm,
        "prospective_correction_norm_rad": float(np.linalg.norm(prospective)),
        "prospective_correction_norm_limit_rad": float(
            prospective_correction_norm_limit_rad
        ),
        "prospective_correction_clipped": clipped,
        "prospective_predicted_objective": predicted.tolist(),
        "prospective_predicted_objective_norm": float(
            np.linalg.norm(predicted)
        ),
        "diagnostic_supported": bool(
            minimum_response_passed and rank == 2
        ),
        "diagnostic_only": True,
        "correction_applied": False,
    }


__all__ = [
    "DifferentialFingerPreloadDiagnosticConfig",
    "analyze_differential_probe",
    "derive_probe_contract",
    "load_differential_finger_preload_diagnostic_config",
    "probe_offset_rad",
]
