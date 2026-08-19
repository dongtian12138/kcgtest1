"""Sensor-only pre-lift and lift safety monitor.

GraspStabilityMonitor.update receives the CURRENT absolute canonical 6D
wrist wrench; the payload reference is fixed at construction.  Internally
the force gate keeps the frozen reference-delta semantics
(norm(F_now - F_ref) > 8 N), while the moment gate uses the frozen
three_component_decomposition_v1 quantity (magnitude increase,
perpendicular, reversal; strictly greater than 0.30 N*m triggers).  The
legacy moment delta norm is tracked as peak evidence only and never gates.
The empty-hand diagnostic recorder keeps its frozen legacy increment-norm
gate and records the candidate decomposition as evidence only.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence


def wrist_payload_increment(
    canonical_wrench: Sequence[float],
    payload_reference: Sequence[float],
) -> tuple[float, ...]:
    """Return the six-dimensional wrist increment relative to the payload.

    Both inputs are canonical sensor-frame reaction wrenches.  The payload
    reference is the quasi-static mean captured with the plug grasped and
    before lift starts; the increment is the quantity the lift gates compare.
    """

    if len(canonical_wrench) != 6 or len(payload_reference) != 6:
        raise ValueError("canonical wrench and payload reference need 6 values")
    canonical = tuple(float(value) for value in canonical_wrench)
    reference = tuple(float(value) for value in payload_reference)
    if not all(math.isfinite(value) for value in canonical + reference):
        raise ValueError("wrist payload increment inputs must be finite")
    return tuple(
        value - base for value, base in zip(canonical, reference)
    )


WRIST_MOMENT_SEMANTICS = "three_component_decomposition_v1"
WRIST_MOMENT_EPSILON = 1e-12
TRIGGER_COMPONENT_PRIORITY = (
    "magnitude_increase",
    "perpendicular",
    "reversal",
)


def _moment_vector(value: Any, label: str) -> tuple[float, ...]:
    if isinstance(value, bool) or not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be a 3-vector")
    if len(value) != 3:
        raise ValueError(f"{label} must be a 3-vector")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} must be finite")
    return result


def evaluate_wrist_moment_safety(
    current_moment_nm: Sequence[float],
    reference_moment_nm: Sequence[float],
    limit_nm: float,
) -> dict[str, Any]:
    """Frozen three-component wrist-moment safety evaluation.

    Decomposes the current moment relative to the payload reference
    direction: magnitude increase (total-norm growth), perpendicular
    (lateral change), and reversal (projection crossing zero backwards).
    A large unloading along the reference direction no longer trips the
    gate.  The legacy delta norm is still reported as evidence only.
    Exact equality with the limit passes; strictly greater triggers.
    """
    current = _moment_vector(current_moment_nm, "current moment")
    reference = _moment_vector(reference_moment_nm, "reference moment")
    if isinstance(limit_nm, bool) or not isinstance(limit_nm, (int, float)):
        raise ValueError("limit must be a number")
    limit = float(limit_nm)
    if not math.isfinite(limit) or limit <= 0.0:
        raise ValueError("limit must be positive and finite")
    norm_m = math.sqrt(sum(value * value for value in current))
    norm_r = math.sqrt(sum(value * value for value in reference))
    legacy = math.sqrt(
        sum(
            (now - base) * (now - base)
            for now, base in zip(current, reference)
        )
    )
    if norm_r <= WRIST_MOMENT_EPSILON:
        magnitude_increase = norm_m
        perpendicular = norm_m
        parallel = None
        reversal = 0.0
    else:
        unit = tuple(value / norm_r for value in reference)
        parallel = sum(now * base for now, base in zip(current, unit))
        magnitude_increase = max(0.0, norm_m - norm_r)
        projection = tuple(parallel * value for value in unit)
        perpendicular = math.sqrt(
            sum(
                (now - base) * (now - base)
                for now, base in zip(current, projection)
            )
        )
        reversal = max(0.0, -parallel)
    score = max(magnitude_increase, perpendicular, reversal)
    triggered = bool(score > limit)
    trigger_component = None
    if triggered:
        components = {
            "magnitude_increase": magnitude_increase,
            "perpendicular": perpendicular,
            "reversal": reversal,
        }
        for name in TRIGGER_COMPONENT_PRIORITY:
            if components[name] == score:
                trigger_component = name
                break
    return {
        "semantics": WRIST_MOMENT_SEMANTICS,
        "reference_moment_norm_nm": norm_r,
        "current_moment_norm_nm": norm_m,
        "legacy_delta_norm_nm": legacy,
        "magnitude_increase_nm": magnitude_increase,
        "parallel_current_nm": parallel,
        "perpendicular_nm": perpendicular,
        "reversal_nm": reversal,
        "gate_score_nm": score,
        "triggered": triggered,
        "trigger_component": trigger_component,
        "gate_limit_nm": limit,
    }


@dataclass(frozen=True)
class GraspStabilityConfig:
    maximum_root_torque_delta_nm: float
    minimum_retained_load_fraction: float
    maximum_normalized_load_imbalance: float
    maximum_load_rate_nm_s: float
    maximum_wrist_force_n: float
    maximum_wrist_moment_nm: float
    maximum_arm_tracking_error_rad: float
    maximum_finger_speed_rad_s: float
    loss_confirm_steps: int

    def __post_init__(self) -> None:
        for name in (
            "maximum_root_torque_delta_nm",
            "minimum_retained_load_fraction",
            "maximum_normalized_load_imbalance",
            "maximum_load_rate_nm_s",
            "maximum_wrist_force_n",
            "maximum_wrist_moment_nm",
            "maximum_arm_tracking_error_rad",
            "maximum_finger_speed_rad_s",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if not 0.0 < self.minimum_retained_load_fraction < 1.0:
            raise ValueError("minimum_retained_load_fraction must be in (0,1)")
        if self.loss_confirm_steps < 2:
            raise ValueError("load loss must be confirmed over multiple steps")


class GraspStabilityMonitor:
    def __init__(
        self,
        config: GraspStabilityConfig,
        *,
        reference_load_nm: Sequence[float],
        load_scale_nm: Sequence[float],
        sample_period_s: float,
        wrist_reference: Sequence[float],
    ):
        if len(reference_load_nm) != 3 or len(load_scale_nm) != 3:
            raise ValueError("three reference loads and scales are required")
        if sample_period_s <= 0.0:
            raise ValueError("sample_period_s must be positive")
        self.config = config
        self.reference = tuple(abs(float(value)) for value in reference_load_nm)
        self.scale = tuple(float(value) for value in load_scale_nm)
        if any(value <= 0.0 for value in self.reference + self.scale):
            raise ValueError("reference loads and scales must be positive")
        # Accept any real one-dimensional numeric sequence of length 6,
        # including numpy ndarrays (the runner passes np.ndarray at
        # runtime).  Reject bools, strings/bytes, scalars, non-iterables,
        # two-dimensional arrays ((1,6) and (6,1) alike), non-numeric and
        # non-finite entries.  The ndim duck-type guard keeps this module
        # free of a NumPy dependency.
        if isinstance(wrist_reference, bool) or isinstance(
            wrist_reference, (str, bytes)
        ):
            raise ValueError("wrist_reference must be a 6-vector")
        array_ndim = getattr(wrist_reference, "ndim", None)
        if array_ndim is not None and int(array_ndim) != 1:
            raise ValueError("wrist_reference must be a 6-vector")
        try:
            wrist = tuple(float(value) for value in wrist_reference)
        except (TypeError, ValueError):
            raise ValueError("wrist_reference must be a 6-vector") from None
        if len(wrist) != 6:
            raise ValueError("wrist_reference must be a 6-vector")
        if not all(math.isfinite(value) for value in wrist):
            raise ValueError("wrist_reference must be finite")
        self.wrist_reference = wrist
        self.dt = sample_period_s
        self.previous = self.reference
        self.loss_steps = [0, 0, 0]
        self.failed = False
        self.failure_reason: str | None = None
        # Wrist-input peak evidence.  The caller feeds the absolute
        # canonical wrench; increments and the legacy delta norm are
        # computed internally and stay evidence-only.
        self.step_count = 0
        self.peak_wrist_force_increment_n = 0.0
        self.peak_wrist_moment_increment_nm = 0.0
        self.peak_per_channel_increment = [0.0] * 6
        self.last_increment = tuple(0.0 for _ in range(6))
        self.peak_moment_magnitude_increase_nm = 0.0
        self.peak_moment_perpendicular_nm = 0.0
        self.peak_moment_reversal_nm = 0.0
        self.peak_moment_safety_score_nm = 0.0
        self.moment_trigger_component: str | None = None
        self.last_moment_safety_evidence: dict[str, Any] = {}

    def update(
        self,
        root_torque_delta_nm: Sequence[float],
        wrist_wrench: Sequence[float],
        *,
        arm_tracking_error_rad: float,
        finger_velocities_rad_s: Sequence[float],
    ) -> bool:
        """Consume one lift step with the ABSOLUTE canonical wrist wrench.

        The force gate keeps the frozen reference-delta semantics; the
        moment gate uses the frozen three-component decomposition of the
        current moment against the payload reference.  The legacy moment
        delta norm is tracked as peak evidence and never gates.
        """
        if len(root_torque_delta_nm) != 3 or len(wrist_wrench) != 6 or len(finger_velocities_rad_s) != 3:
            raise ValueError("monitor input dimensions are invalid")
        loads = tuple(abs(float(value)) for value in root_torque_delta_nm)
        wrench = tuple(float(value) for value in wrist_wrench)
        values = loads + wrench + (float(arm_tracking_error_rad),) + tuple(
            float(value) for value in finger_velocities_rad_s
        )
        if not all(math.isfinite(value) for value in values):
            return self._fail("nonfinite_sensor_or_robot_state")
        self.step_count += 1
        increment = tuple(
            value - base for value, base in zip(wrench, self.wrist_reference)
        )
        self.last_increment = increment
        force_increment = math.sqrt(
            sum(value * value for value in increment[:3])
        )
        moment_evidence = evaluate_wrist_moment_safety(
            wrench[3:],
            self.wrist_reference[3:],
            self.config.maximum_wrist_moment_nm,
        )
        self.last_moment_safety_evidence = moment_evidence
        self.peak_wrist_force_increment_n = max(
            self.peak_wrist_force_increment_n, force_increment
        )
        self.peak_wrist_moment_increment_nm = max(
            self.peak_wrist_moment_increment_nm,
            moment_evidence["legacy_delta_norm_nm"],
        )
        self.peak_per_channel_increment = [
            max(old, abs(value))
            for old, value in zip(self.peak_per_channel_increment, increment)
        ]
        self.peak_moment_magnitude_increase_nm = max(
            self.peak_moment_magnitude_increase_nm,
            moment_evidence["magnitude_increase_nm"],
        )
        self.peak_moment_perpendicular_nm = max(
            self.peak_moment_perpendicular_nm,
            moment_evidence["perpendicular_nm"],
        )
        self.peak_moment_reversal_nm = max(
            self.peak_moment_reversal_nm, moment_evidence["reversal_nm"]
        )
        self.peak_moment_safety_score_nm = max(
            self.peak_moment_safety_score_nm, moment_evidence["gate_score_nm"]
        )
        if max(loads) > self.config.maximum_root_torque_delta_nm:
            return self._fail("root_torque_limit")
        if force_increment > self.config.maximum_wrist_force_n:
            return self._fail("wrist_force_limit")
        if moment_evidence["triggered"]:
            self.moment_trigger_component = moment_evidence[
                "trigger_component"
            ]
            return self._fail("wrist_moment_limit")
        if arm_tracking_error_rad > self.config.maximum_arm_tracking_error_rad:
            return self._fail("arm_tracking_limit")
        if max(abs(value) for value in finger_velocities_rad_s) > self.config.maximum_finger_speed_rad_s:
            return self._fail("finger_speed_limit")
        rates = tuple(abs(now - old) / self.dt for now, old in zip(loads, self.previous))
        if max(rates) > self.config.maximum_load_rate_nm_s:
            return self._fail("root_torque_rate_limit")
        normalized = tuple(value / scale for value, scale in zip(loads, self.scale))
        if max(normalized) - min(normalized) > self.config.maximum_normalized_load_imbalance:
            return self._fail("load_imbalance_limit")
        for index, (value, reference) in enumerate(zip(loads, self.reference)):
            if value < self.config.minimum_retained_load_fraction * reference:
                self.loss_steps[index] += 1
            else:
                self.loss_steps[index] = 0
            if self.loss_steps[index] >= self.config.loss_confirm_steps:
                return self._fail(f"f{index + 1}_load_lost")
        self.previous = loads
        return True

    def _fail(self, reason: str) -> bool:
        self.failed = True
        self.failure_reason = reason
        return False

    def fail_closed(self, reason: str) -> bool:
        """Force a sensor-side failure without consuming a stale sample.

        Used when the wrist sensor itself is lost mid-lift: the frozen last
        reading must not pass the increment gates, and the summary must show
        the failure instead of a misleading healthy step.
        """

        return self._fail(reason)

    def summary(self) -> dict[str, Any]:
        """Return JSON-safe evidence plus the unchanged gate limits."""

        return {
            "steps": self.step_count,
            "failed": self.failed,
            "failure_reason": self.failure_reason,
            "wrist_moment_semantics": WRIST_MOMENT_SEMANTICS,
            "peak_wrist_force_increment_n": self.peak_wrist_force_increment_n,
            "peak_wrist_moment_increment_nm": (
                self.peak_wrist_moment_increment_nm
            ),
            "peak_moment_magnitude_increase_nm": (
                self.peak_moment_magnitude_increase_nm
            ),
            "peak_moment_perpendicular_nm": (
                self.peak_moment_perpendicular_nm
            ),
            "peak_moment_reversal_nm": self.peak_moment_reversal_nm,
            "peak_moment_safety_score_nm": self.peak_moment_safety_score_nm,
            "moment_trigger_component": self.moment_trigger_component,
            "last_moment_safety_evidence": dict(
                self.last_moment_safety_evidence
            ),
            "peak_per_channel_increment": list(
                self.peak_per_channel_increment
            ),
            "last_increment": list(self.last_increment),
            "force_gate_n": self.config.maximum_wrist_force_n,
            "moment_gate_nm": self.config.maximum_wrist_moment_nm,
            "legacy_moment_delta_is_evidence_only": True,
        }


class EmptyHandLiftDiagnosticMonitor:
    """Wrist/robot-only diagnostic recorder for empty-hand lift replays.

    Deliberately NOT a GraspStabilityMonitor: the replay has no grasped
    payload reference, so the root-load retained/imbalance/rate gates do
    not apply and no fake reference load is injected.  It enforces only the
    frozen wrist force/moment increment gates (8 N / 0.30 N*m on the
    current increment-norm semantics), the arm-tracking gate, the finger
    speed gate, and finite inputs; a gate crossing is recorded as a
    diagnostic observation, never as a grasp failure, and this recorder
    can never grant any PASS.
    """

    def __init__(
        self,
        config: GraspStabilityConfig,
        *,
        reference_wrench: Sequence[float],
    ):
        if len(reference_wrench) != 6:
            raise ValueError("diagnostic wrist reference needs 6 values")
        reference = tuple(float(value) for value in reference_wrench)
        if not all(math.isfinite(value) for value in reference):
            raise ValueError("diagnostic wrist reference must be finite")
        self.config = config
        self.reference = reference
        self.failed = False
        self.failure_reason: str | None = None
        self.step_count = 0
        self.peak_wrist_force_increment_n = 0.0
        self.peak_wrist_moment_increment_nm = 0.0
        self.peak_per_channel_increment = [0.0] * 6
        self.last_increment = tuple(0.0 for _ in range(6))
        self.last_moment_safety_evidence: dict[str, Any] = {}

    def update(
        self,
        wrist_wrench_canonical: Sequence[float],
        *,
        arm_tracking_error_rad: float,
        finger_velocities_rad_s: Sequence[float],
    ) -> bool:
        """Record one step; return False only when a gate is observed.

        The wrist input is the absolute canonical wrench; the increment is
        computed against the open-hand reference captured before the move,
        matching the formal lift's increment semantics exactly.
        """
        if len(wrist_wrench_canonical) != 6 or len(finger_velocities_rad_s) != 3:
            raise ValueError("diagnostic monitor input dimensions are invalid")
        wrench = tuple(float(value) for value in wrist_wrench_canonical)
        values = wrench + (float(arm_tracking_error_rad),) + tuple(
            float(value) for value in finger_velocities_rad_s
        )
        if not all(math.isfinite(value) for value in values):
            self.failed = True
            self.failure_reason = "empty_hand_nonfinite_sensor_or_robot_state"
            return False
        increment = tuple(
            value - base for value, base in zip(wrench, self.reference)
        )
        self.step_count += 1
        self.last_increment = increment
        # Candidate three-component decomposition recorded as evidence only:
        # the diagnostic stop/exit semantics stay on the frozen legacy gate.
        self.last_moment_safety_evidence = evaluate_wrist_moment_safety(
            wrench[3:],
            self.reference[3:],
            self.config.maximum_wrist_moment_nm,
        )
        force_increment = math.sqrt(
            sum(value * value for value in increment[:3])
        )
        moment_increment = math.sqrt(
            sum(value * value for value in increment[3:])
        )
        self.peak_wrist_force_increment_n = max(
            self.peak_wrist_force_increment_n, force_increment
        )
        self.peak_wrist_moment_increment_nm = max(
            self.peak_wrist_moment_increment_nm, moment_increment
        )
        self.peak_per_channel_increment = [
            max(old, abs(value))
            for old, value in zip(self.peak_per_channel_increment, increment)
        ]
        # Strictly-greater comparisons: exact equality with a gate passes.
        if force_increment > self.config.maximum_wrist_force_n:
            self.failed = True
            self.failure_reason = "empty_hand_wrist_force_gate_observed"
            return False
        if moment_increment > self.config.maximum_wrist_moment_nm:
            self.failed = True
            self.failure_reason = "empty_hand_wrist_moment_gate_observed"
            return False
        if arm_tracking_error_rad > self.config.maximum_arm_tracking_error_rad:
            self.failed = True
            self.failure_reason = "empty_hand_arm_tracking_gate_observed"
            return False
        if max(abs(value) for value in finger_velocities_rad_s) > self.config.maximum_finger_speed_rad_s:
            self.failed = True
            self.failure_reason = "empty_hand_finger_speed_gate_observed"
            return False
        return True

    def fail_closed(self, reason: str) -> bool:
        """Sensor-side failure without consuming a stale sample."""
        self.failed = True
        self.failure_reason = reason
        return False

    def summary(self) -> dict[str, Any]:
        return {
            "diagnostic_only": True,
            "mode": "EMPTY_HAND_FIRST_STAGE_REPLAY_DIAGNOSTIC_ONLY",
            "steps": self.step_count,
            "failed": self.failed,
            "failure_reason": self.failure_reason,
            "reference_wrench": list(self.reference),
            "peak_wrist_force_increment_n": self.peak_wrist_force_increment_n,
            "peak_wrist_moment_increment_nm": (
                self.peak_wrist_moment_increment_nm
            ),
            "peak_per_channel_increment": list(
                self.peak_per_channel_increment
            ),
            "last_increment": list(self.last_increment),
            "force_gate_n": self.config.maximum_wrist_force_n,
            "moment_gate_nm": self.config.maximum_wrist_moment_nm,
            "arm_tracking_gate_rad": (
                self.config.maximum_arm_tracking_error_rad
            ),
            "finger_speed_gate_rad_s": (
                self.config.maximum_finger_speed_rad_s
            ),
            "root_load_gates_applied": False,
            "last_moment_safety_evidence": dict(
                self.last_moment_safety_evidence
            ),
            "moment_candidate_evidence_only": True,
        }
