"""Bounded zero-mean three-finger load balancing during staged lift."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from .three_finger_sequential_grasp import SequentialGraspConfig


THRESHOLD_LABEL = "SIM_TUNING_ONLY_B_V2_H11"
CONFIG_KEYS = (
    "enabled",
    "threshold_label",
    "source_run_id",
    "reuse_sequential_balance_parameters",
    "zero_mean_closure_coordinate_trim",
    "immediately_preceding_root_torque_only",
)


@dataclass(frozen=True)
class LiftFingerLoadBalanceConfig:
    enabled: bool
    threshold_label: str
    source_run_id: str
    reuse_sequential_balance_parameters: bool
    zero_mean_closure_coordinate_trim: bool
    immediately_preceding_root_torque_only: bool

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("lift finger load balance enabled must be boolean")
        if not isinstance(self.threshold_label, str) or not self.threshold_label:
            raise ValueError("lift finger load balance threshold_label must be text")
        if not isinstance(self.source_run_id, str) or not self.source_run_id:
            raise ValueError("lift finger load balance source_run_id must be text")
        for name in (
            "reuse_sequential_balance_parameters",
            "zero_mean_closure_coordinate_trim",
            "immediately_preceding_root_torque_only",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"lift finger load balance {name} must be boolean")
        if self.enabled:
            if self.threshold_label != THRESHOLD_LABEL:
                raise ValueError(
                    "enabled H11 load balance must keep its SIM_TUNING_ONLY label"
                )
            if not all(
                (
                    self.reuse_sequential_balance_parameters,
                    self.zero_mean_closure_coordinate_trim,
                    self.immediately_preceding_root_torque_only,
                )
            ):
                raise ValueError(
                    "enabled H11 load balance must reuse the sequential bounds, "
                    "remain zero-mean and use only the preceding root-torque sample"
                )


def _default_config() -> LiftFingerLoadBalanceConfig:
    return LiftFingerLoadBalanceConfig(
        enabled=False,
        threshold_label=THRESHOLD_LABEL,
        source_run_id="HISTORICAL_CONTRACT_DISABLED",
        reuse_sequential_balance_parameters=True,
        zero_mean_closure_coordinate_trim=True,
        immediately_preceding_root_torque_only=True,
    )


def load_lift_finger_load_balance_config(
    value: Any,
) -> LiftFingerLoadBalanceConfig:
    """Load the strict H11 section; historical contracts default disabled."""

    if value is None:
        return _default_config()
    if not isinstance(value, Mapping):
        raise ValueError("lift_finger_load_balance must be a mapping")
    unknown = sorted(set(value) - set(CONFIG_KEYS))
    missing = sorted(set(CONFIG_KEYS) - set(value))
    if unknown or missing:
        raise ValueError(
            "lift_finger_load_balance has unknown keys "
            f"{unknown} and/or missing keys {missing}"
        )
    return LiftFingerLoadBalanceConfig(
        enabled=value["enabled"],
        threshold_label=value["threshold_label"],
        source_run_id=value["source_run_id"],
        reuse_sequential_balance_parameters=value[
            "reuse_sequential_balance_parameters"
        ],
        zero_mean_closure_coordinate_trim=value[
            "zero_mean_closure_coordinate_trim"
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


def _project_zero_sum_box(
    requested: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
) -> tuple[float, float, float]:
    """Project three requested increments onto box bounds and sum(x)=0."""

    requested_values = _three_finite(requested, "requested increments")
    lower_values = _three_finite(lower, "lower bounds")
    upper_values = _three_finite(upper, "upper bounds")
    if any(lo > hi for lo, hi in zip(lower_values, upper_values)):
        raise ValueError("zero-mean projection bounds are inverted")
    if sum(lower_values) > 0.0 or sum(upper_values) < 0.0:
        raise ValueError("zero-mean projection is infeasible")

    lambda_low = min(
        value - hi
        for value, hi in zip(requested_values, upper_values)
    )
    lambda_high = max(
        value - lo
        for value, lo in zip(requested_values, lower_values)
    )
    projected = [0.0, 0.0, 0.0]
    # Fixed iterations are a numerical implementation detail, not a control
    # parameter.  The bounds and gain all come from SequentialGraspConfig.
    for _ in range(80):
        level = 0.5 * (lambda_low + lambda_high)
        projected = [
            min(max(value - level, lo), hi)
            for value, lo, hi in zip(
                requested_values, lower_values, upper_values
            )
        ]
        if sum(projected) > 0.0:
            lambda_low = level
        else:
            lambda_high = level

    residual = sum(projected)
    if residual != 0.0:
        if residual > 0.0:
            for index in range(3):
                available = projected[index] - lower_values[index]
                correction = min(residual, available)
                projected[index] -= correction
                residual -= correction
                if residual <= 0.0:
                    break
        else:
            for index in range(3):
                available = upper_values[index] - projected[index]
                correction = min(-residual, available)
                projected[index] += correction
                residual += correction
                if residual >= 0.0:
                    break
    # Remove the final floating residual without leaving the box.  Feasibility
    # was proven above, so at least one component has the required room.
    residual = sum(projected)
    if residual != 0.0:
        for index in range(3):
            candidate = projected[index] - residual
            if lower_values[index] <= candidate <= upper_values[index]:
                projected[index] = candidate
                break
    return tuple(float(value) for value in projected)


class LiftFingerLoadBalance:
    """Stateful H11 overlay around the frozen stable-grasp targets."""

    def __init__(
        self,
        config: LiftFingerLoadBalanceConfig,
        sequential: SequentialGraspConfig,
        *,
        base_targets_rad: Sequence[float],
        open_targets_rad: Sequence[float],
        closed_targets_rad: Sequence[float],
        initial_balance_total_rad: Sequence[float],
    ) -> None:
        if not config.enabled:
            raise ValueError("lift finger load balance controller requires enabled config")
        self.config = config
        self.sequential = sequential
        self.base_targets = _three_finite(base_targets_rad, "base targets")
        self.open_targets = _three_finite(open_targets_rad, "open targets")
        self.closed_targets = _three_finite(closed_targets_rad, "closed targets")
        self.initial_balance_total = _three_finite(
            initial_balance_total_rad, "initial balance totals"
        )
        self.direction = tuple(
            1.0 if closed > opened else -1.0
            for opened, closed in zip(self.open_targets, self.closed_targets)
        )
        if any(
            opened == closed
            for opened, closed in zip(self.open_targets, self.closed_targets)
        ):
            raise ValueError("each lift-balanced finger needs a closure direction")
        if any(
            not min(opened, closed) <= base <= max(opened, closed)
            for base, opened, closed in zip(
                self.base_targets, self.open_targets, self.closed_targets
            )
        ):
            raise ValueError("base target lies outside its physical joint interval")
        maximum_total = self.sequential.maximum_balance_total_rad
        if any(abs(value) > maximum_total for value in self.initial_balance_total):
            raise ValueError("initial balance total already exceeds sequential budget")

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
            raise ValueError("initial lift target has no feasible zero-trim interval")
        self.cumulative_lower = tuple(cumulative_lower)
        self.cumulative_upper = tuple(cumulative_upper)
        self.cumulative_trim = [0.0, 0.0, 0.0]
        self.steps = 0
        self.maximum_abs_step = 0.0
        self.maximum_abs_cumulative = 0.0
        self.maximum_abs_zero_sum_residual = 0.0
        self.saturated_step_count = 0
        self.saturated_cumulative_count = 0
        self.maximum_normalized_load_imbalance = 0.0

    def update(self, root_torque_nm: Sequence[float]) -> dict[str, Any]:
        signed_load = _three_finite(root_torque_nm, "root torque")
        absolute_load = tuple(abs(value) for value in signed_load)
        normalized = tuple(
            load / scale
            for load, scale in zip(
                absolute_load, self.sequential.load_scale_nm
            )
        )
        mean_load = sum(normalized) / 3.0
        requested = tuple(
            self.sequential.balance_gain_rad_per_load
            * (mean_load - value)
            for value in normalized
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
        applied = _project_zero_sum_box(requested, lower, upper)
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
        if any(
            target < min(opened, closed) - 1.0e-12
            or target > max(opened, closed) + 1.0e-12
            for target, opened, closed in zip(
                targets, self.open_targets, self.closed_targets
            )
        ):
            raise RuntimeError("lift finger load balance escaped a joint interval")

        zero_sum_residual = sum(applied)
        cumulative_zero_sum_residual = sum(self.cumulative_trim)
        imbalance = max(normalized) - min(normalized)
        self.steps += 1
        self.maximum_abs_step = max(
            self.maximum_abs_step, max(abs(value) for value in applied)
        )
        self.maximum_abs_cumulative = max(
            self.maximum_abs_cumulative,
            max(abs(value) for value in self.cumulative_trim),
        )
        self.maximum_abs_zero_sum_residual = max(
            self.maximum_abs_zero_sum_residual,
            abs(zero_sum_residual),
            abs(cumulative_zero_sum_residual),
        )
        self.maximum_normalized_load_imbalance = max(
            self.maximum_normalized_load_imbalance, imbalance
        )
        self.saturated_step_count += int(step_bound_active)
        self.saturated_cumulative_count += int(cumulative_bound_active)

        return {
            "input_root_torque_signed_nm": list(signed_load),
            "input_root_torque_absolute_nm": list(absolute_load),
            "normalized_loads": list(normalized),
            "normalized_load_imbalance": float(imbalance),
            "requested_delta_closure_rad": list(requested),
            "applied_delta_closure_rad": list(applied),
            "cumulative_trim_closure_rad": list(self.cumulative_trim),
            "combined_balance_total_rad": [
                initial + trim
                for initial, trim in zip(
                    self.initial_balance_total, self.cumulative_trim
                )
            ],
            "output_targets_rad": list(targets),
            "zero_sum_residual_rad": float(zero_sum_residual),
            "cumulative_zero_sum_residual_rad": float(
                cumulative_zero_sum_residual
            ),
            "step_bound_active": bool(step_bound_active),
            "cumulative_or_joint_bound_active": bool(
                cumulative_bound_active
            ),
            "object_truth_used": False,
            "contact_truth_used": False,
            "event_truth_used": False,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "record_count": int(self.steps),
            "balance_gain_rad_per_load": (
                self.sequential.balance_gain_rad_per_load
            ),
            "maximum_balance_step_rad": (
                self.sequential.maximum_balance_step_rad
            ),
            "maximum_balance_total_rad": (
                self.sequential.maximum_balance_total_rad
            ),
            "load_scale_nm": list(self.sequential.load_scale_nm),
            "base_targets_rad": list(self.base_targets),
            "initial_balance_total_rad": list(self.initial_balance_total),
            "cumulative_trim_lower_rad": list(self.cumulative_lower),
            "cumulative_trim_upper_rad": list(self.cumulative_upper),
            "final_cumulative_trim_rad": list(self.cumulative_trim),
            "final_targets_rad": [
                base + direction * total
                for base, direction, total in zip(
                    self.base_targets, self.direction, self.cumulative_trim
                )
            ],
            "maximum_abs_applied_step_rad": self.maximum_abs_step,
            "maximum_abs_cumulative_trim_rad": self.maximum_abs_cumulative,
            "maximum_abs_zero_sum_residual_rad": (
                self.maximum_abs_zero_sum_residual
            ),
            "maximum_normalized_load_imbalance": (
                self.maximum_normalized_load_imbalance
            ),
            "step_bound_active_count": int(self.saturated_step_count),
            "cumulative_or_joint_bound_active_count": int(
                self.saturated_cumulative_count
            ),
            "mean_closure_target_change_rad": (
                sum(self.cumulative_trim) / 3.0
            ),
            "object_truth_used": False,
            "contact_truth_used": False,
            "event_truth_used": False,
        }
