"""Bounded per-finger absolute root-load hold during staged lift."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from .three_finger_sequential_grasp import SequentialGraspConfig


THRESHOLD_LABEL = "SIM_TUNING_ONLY_B_V2_H17"
REFERENCE_WINDOW_SOURCE = (
    "TRAILING_H13_TRANSITION_USING_EXISTING_REFERENCE_WINDOW_STEPS"
)
CONFIG_KEYS = (
    "enabled",
    "threshold_label",
    "source_h11_run_id",
    "source_h16_run_id",
    "reference_window_source",
    "reuse_sequential_control_parameters",
    "independent_absolute_root_load_hold",
    "immediately_preceding_root_torque_only",
)


@dataclass(frozen=True)
class LiftFingerAbsoluteLoadHoldConfig:
    enabled: bool
    threshold_label: str
    source_h11_run_id: str
    source_h16_run_id: str
    reference_window_source: str
    reuse_sequential_control_parameters: bool
    independent_absolute_root_load_hold: bool
    immediately_preceding_root_torque_only: bool

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("absolute root-load hold enabled must be boolean")
        for name in (
            "threshold_label",
            "source_h11_run_id",
            "source_h16_run_id",
            "reference_window_source",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"absolute root-load hold {name} must be text")
        for name in (
            "reuse_sequential_control_parameters",
            "independent_absolute_root_load_hold",
            "immediately_preceding_root_torque_only",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"absolute root-load hold {name} must be boolean")
        if self.enabled:
            if self.threshold_label != THRESHOLD_LABEL:
                raise ValueError(
                    "enabled H17 absolute root-load hold must keep its "
                    "SIM_TUNING_ONLY label"
                )
            if self.reference_window_source != REFERENCE_WINDOW_SOURCE:
                raise ValueError(
                    "enabled H17 must use the trailing H13 transition and the "
                    "existing reference window"
                )
            if not all(
                (
                    self.reuse_sequential_control_parameters,
                    self.independent_absolute_root_load_hold,
                    self.immediately_preceding_root_torque_only,
                )
            ):
                raise ValueError(
                    "enabled H17 must reuse the bounded sequential parameters, "
                    "hold each load independently and use only the preceding "
                    "root-torque sample"
                )


def _default_config() -> LiftFingerAbsoluteLoadHoldConfig:
    return LiftFingerAbsoluteLoadHoldConfig(
        enabled=False,
        threshold_label=THRESHOLD_LABEL,
        source_h11_run_id="HISTORICAL_CONTRACT_DISABLED",
        source_h16_run_id="HISTORICAL_CONTRACT_DISABLED",
        reference_window_source=REFERENCE_WINDOW_SOURCE,
        reuse_sequential_control_parameters=True,
        independent_absolute_root_load_hold=True,
        immediately_preceding_root_torque_only=True,
    )


def load_lift_finger_absolute_load_hold_config(
    value: Any,
) -> LiftFingerAbsoluteLoadHoldConfig:
    """Load the strict H17 section; historical contracts default disabled."""

    if value is None:
        return _default_config()
    if not isinstance(value, Mapping):
        raise ValueError("lift_finger_absolute_load_hold must be a mapping")
    unknown = sorted(set(value) - set(CONFIG_KEYS))
    missing = sorted(set(CONFIG_KEYS) - set(value))
    if unknown or missing:
        raise ValueError(
            "lift_finger_absolute_load_hold has unknown keys "
            f"{unknown} and/or missing keys {missing}"
        )
    return LiftFingerAbsoluteLoadHoldConfig(
        enabled=value["enabled"],
        threshold_label=value["threshold_label"],
        source_h11_run_id=value["source_h11_run_id"],
        source_h16_run_id=value["source_h16_run_id"],
        reference_window_source=value["reference_window_source"],
        reuse_sequential_control_parameters=value[
            "reuse_sequential_control_parameters"
        ],
        independent_absolute_root_load_hold=value[
            "independent_absolute_root_load_hold"
        ],
        immediately_preceding_root_torque_only=value[
            "immediately_preceding_root_torque_only"
        ],
    )


def _three_finite(values: Sequence[float], label: str) -> tuple[float, ...]:
    if len(values) != 3:
        raise ValueError(f"{label} must contain exactly three values")
    result = tuple(float(value) for value in values)
    if any(not math.isfinite(value) for value in result):
        raise ValueError(f"{label} must contain only finite values")
    return result


class LiftFingerAbsoluteLoadHold:
    """H17 controller around one fresh, pre-lift three-load reference."""

    def __init__(
        self,
        config: LiftFingerAbsoluteLoadHoldConfig,
        sequential: SequentialGraspConfig,
        *,
        base_targets_rad: Sequence[float],
        open_targets_rad: Sequence[float],
        closed_targets_rad: Sequence[float],
        initial_balance_total_rad: Sequence[float],
        reference_root_torque_nm: Sequence[float],
    ) -> None:
        if not config.enabled:
            raise ValueError("absolute root-load hold requires enabled config")
        self.config = config
        self.sequential = sequential
        self.base_targets = _three_finite(base_targets_rad, "base targets")
        self.open_targets = _three_finite(open_targets_rad, "open targets")
        self.closed_targets = _three_finite(closed_targets_rad, "closed targets")
        self.initial_balance_total = _three_finite(
            initial_balance_total_rad, "initial balance totals"
        )
        signed_reference = _three_finite(
            reference_root_torque_nm, "reference root torque"
        )
        self.reference_absolute_load = tuple(abs(value) for value in signed_reference)
        self.reference_normalized_load = tuple(
            load / scale
            for load, scale in zip(
                self.reference_absolute_load, self.sequential.load_scale_nm
            )
        )
        if any(
            value < self.sequential.stable_minimum_normalized_load
            for value in self.reference_normalized_load
        ):
            raise ValueError(
                "H17 reference root load is below the frozen stable-contact floor"
            )

        self.direction = tuple(
            1.0 if closed > opened else -1.0
            for opened, closed in zip(self.open_targets, self.closed_targets)
        )
        if any(
            opened == closed
            for opened, closed in zip(self.open_targets, self.closed_targets)
        ):
            raise ValueError("each H17 finger needs a closure direction")
        if any(
            not min(opened, closed) <= base <= max(opened, closed)
            for base, opened, closed in zip(
                self.base_targets, self.open_targets, self.closed_targets
            )
        ):
            raise ValueError("H17 base target lies outside its joint interval")

        maximum_total = self.sequential.maximum_balance_total_rad
        if any(abs(value) > maximum_total for value in self.initial_balance_total):
            raise ValueError("initial balance total exceeds the sequential budget")
        cumulative_lower = []
        cumulative_upper = []
        for base, opened, closed, direction, initial in zip(
            self.base_targets,
            self.open_targets,
            self.closed_targets,
            self.direction,
            self.initial_balance_total,
        ):
            physical_lower = direction * (opened - base)
            physical_upper = direction * (closed - base)
            cumulative_lower.append(max(physical_lower, -maximum_total - initial))
            cumulative_upper.append(min(physical_upper, maximum_total - initial))
        if any(lo > 0.0 or hi < 0.0 for lo, hi in zip(cumulative_lower, cumulative_upper)):
            raise ValueError("initial H17 target has no feasible trim interval")
        self.cumulative_lower = tuple(cumulative_lower)
        self.cumulative_upper = tuple(cumulative_upper)
        self.cumulative_trim = [0.0, 0.0, 0.0]
        self.steps = 0
        self.maximum_abs_step = 0.0
        self.maximum_abs_cumulative = 0.0
        self.maximum_abs_load_error_nm = 0.0
        self.maximum_abs_common_mode_step = 0.0
        self.step_bound_active_count = 0
        self.cumulative_or_joint_bound_active_count = 0

    def update(self, root_torque_nm: Sequence[float]) -> dict[str, Any]:
        signed_load = _three_finite(root_torque_nm, "root torque")
        absolute_load = tuple(abs(value) for value in signed_load)
        normalized = tuple(
            load / scale
            for load, scale in zip(absolute_load, self.sequential.load_scale_nm)
        )
        normalized_error = tuple(
            reference - current
            for reference, current in zip(
                self.reference_normalized_load, normalized
            )
        )
        requested = tuple(
            self.sequential.balance_gain_rad_per_load * error
            for error in normalized_error
        )
        maximum_step = self.sequential.maximum_balance_step_rad
        lower = tuple(
            max(-maximum_step, lo - total)
            for lo, total in zip(self.cumulative_lower, self.cumulative_trim)
        )
        upper = tuple(
            min(maximum_step, hi - total)
            for hi, total in zip(self.cumulative_upper, self.cumulative_trim)
        )
        applied = tuple(
            min(max(value, lo), hi)
            for value, lo, hi in zip(requested, lower, upper)
        )
        step_bound_active = any(
            abs(value) >= maximum_step - 1.0e-15 for value in applied
        )
        cumulative_bound_active = any(
            abs(total + delta - lo) <= 1.0e-15
            or abs(total + delta - hi) <= 1.0e-15
            for total, delta, lo, hi in zip(
                self.cumulative_trim,
                applied,
                self.cumulative_lower,
                self.cumulative_upper,
            )
        )
        self.cumulative_trim = [
            total + delta
            for total, delta in zip(self.cumulative_trim, applied)
        ]
        targets = tuple(
            base + direction * total
            for base, direction, total in zip(
                self.base_targets, self.direction, self.cumulative_trim
            )
        )
        combined_total = tuple(
            initial + trim
            for initial, trim in zip(
                self.initial_balance_total, self.cumulative_trim
            )
        )
        load_error_nm = tuple(
            reference - current
            for reference, current in zip(
                self.reference_absolute_load, absolute_load
            )
        )
        common_mode_step = sum(applied) / 3.0
        self.steps += 1
        self.maximum_abs_step = max(
            self.maximum_abs_step, max(abs(value) for value in applied)
        )
        self.maximum_abs_cumulative = max(
            self.maximum_abs_cumulative,
            max(abs(value) for value in self.cumulative_trim),
        )
        self.maximum_abs_load_error_nm = max(
            self.maximum_abs_load_error_nm,
            max(abs(value) for value in load_error_nm),
        )
        self.maximum_abs_common_mode_step = max(
            self.maximum_abs_common_mode_step, abs(common_mode_step)
        )
        self.step_bound_active_count += int(step_bound_active)
        self.cumulative_or_joint_bound_active_count += int(
            cumulative_bound_active
        )
        return {
            "input_root_torque_signed_nm": list(signed_load),
            "input_root_torque_absolute_nm": list(absolute_load),
            "reference_root_torque_absolute_nm": list(
                self.reference_absolute_load
            ),
            "normalized_loads": list(normalized),
            "reference_normalized_loads": list(
                self.reference_normalized_load
            ),
            "load_error_nm": list(load_error_nm),
            "normalized_load_error": list(normalized_error),
            "requested_delta_closure_rad": list(requested),
            "applied_delta_closure_rad": list(applied),
            "common_mode_applied_delta_closure_rad": float(common_mode_step),
            "cumulative_trim_closure_rad": list(self.cumulative_trim),
            "combined_balance_total_rad": list(combined_total),
            "output_targets_rad": list(targets),
            "step_bound_active": step_bound_active,
            "cumulative_or_joint_bound_active": cumulative_bound_active,
            "independent_absolute_load_targets": True,
            "mean_closure_target_allowed_to_change": True,
            "object_truth_used": False,
            "contact_truth_used": False,
            "event_truth_used": False,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "record_count": self.steps,
            "reference_root_torque_absolute_nm": list(
                self.reference_absolute_load
            ),
            "reference_normalized_loads": list(
                self.reference_normalized_load
            ),
            "balance_gain_rad_per_load": (
                self.sequential.balance_gain_rad_per_load
            ),
            "load_scale_nm": list(self.sequential.load_scale_nm),
            "maximum_balance_step_rad": (
                self.sequential.maximum_balance_step_rad
            ),
            "maximum_balance_total_rad": (
                self.sequential.maximum_balance_total_rad
            ),
            "maximum_abs_applied_step_rad": self.maximum_abs_step,
            "maximum_abs_cumulative_trim_rad": self.maximum_abs_cumulative,
            "maximum_abs_load_error_nm": self.maximum_abs_load_error_nm,
            "maximum_abs_common_mode_step_rad": (
                self.maximum_abs_common_mode_step
            ),
            "step_bound_active_count": self.step_bound_active_count,
            "cumulative_or_joint_bound_active_count": (
                self.cumulative_or_joint_bound_active_count
            ),
            "final_cumulative_trim_rad": list(self.cumulative_trim),
            "independent_absolute_load_targets": True,
            "mean_closure_target_allowed_to_change": True,
        }


__all__ = [
    "LiftFingerAbsoluteLoadHold",
    "LiftFingerAbsoluteLoadHoldConfig",
    "load_lift_finger_absolute_load_hold_config",
]
