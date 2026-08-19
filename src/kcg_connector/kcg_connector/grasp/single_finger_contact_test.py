'''Pure single-finger contact/release characterization controller.

This controller characterizes one finger at a time using only the finger-root
torque proxy and joint q/qd/history.  It has no geometry, object pose,
collider, or contact-report input.  The other two fingers stay open; the
runner guarantees that outside this module.  The controller outputs only the
selected finger's target, stiffness scale, state and JSON-safe evidence.

The controller can report `detector_test_passed=True` after a bounded
SOFT_HOLD window and a confirmed commanded release, but that is a detector
characterization result, never a grasp PASS.
'''

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

from .finger_contact_detector import (
    FingerContactDetector,
    FingerContactDetectorConfig,
    FingerContactObservation,
    FingerContactState,
)

SOFT_HOLD_STIFFNESS_SCALE = 0.35
MAXIMUM_RELEASE_STEPS_CEILING = 100_000

FINGERS = ("f1", "f2", "f3")


@dataclass(frozen=True)
class SingleFingerContactConfig:
    '''Bounded pure-control parameters for one single-finger test.'''

    threshold_label: str = "SIM_TUNING_ONLY"
    soft_hold_steps: int = 24
    minimum_release_travel_rad: float = 0.05
    maximum_release_tracking_error_rad: float = 0.05
    maximum_release_steps: int = 720
    maximum_approach_steps: int = 1440
    approach_rate_rad_s: float = 0.18
    release_rate_rad_s: float = 0.18

    def __post_init__(self) -> None:
        if self.threshold_label != "SIM_TUNING_ONLY":
            raise ValueError(
                "single-finger thresholds must remain SIM_TUNING_ONLY"
            )
        if (
            isinstance(self.soft_hold_steps, bool)
            or not isinstance(self.soft_hold_steps, int)
            or self.soft_hold_steps < 2
        ):
            raise ValueError("soft_hold_steps must be an integer >= 2")
        if (
            isinstance(self.maximum_release_steps, bool)
            or not isinstance(self.maximum_release_steps, int)
            or self.maximum_release_steps <= 0
            or self.maximum_release_steps > MAXIMUM_RELEASE_STEPS_CEILING
        ):
            raise ValueError(
                "maximum_release_steps must be a positive bounded integer"
            )
        if (
            isinstance(self.maximum_approach_steps, bool)
            or not isinstance(self.maximum_approach_steps, int)
            or self.maximum_approach_steps <= 0
        ):
            raise ValueError(
                "maximum_approach_steps must be a positive integer"
            )
        for name in (
            "minimum_release_travel_rad",
            "maximum_release_tracking_error_rad",
            "approach_rate_rad_s",
            "release_rate_rad_s",
        ):
            value = getattr(self, name)
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be positive and finite")


@dataclass(frozen=True)
class SingleFingerCommand:
    '''One JSON-safe controller step for the selected finger.'''

    finger: str
    target_rad: float
    stiffness_scale: float
    state: FingerContactState
    observation: FingerContactObservation | None
    soft_hold_step: int
    soft_hold_steps_configured: int
    release_step: int
    release_confirm_steps_configured: int
    release_conditions: Mapping[str, Any]
    failed: bool
    failure_reason: str | None
    detector_test_passed: bool
    evidence: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReleaseBudgetEvidence:
    '''JSON-safe release budget feasibility breakdown (frozen 018 formula).'''

    travel_steps: int
    filter_tail_steps: int
    tracking_lag_steps: int
    confirm_steps: int
    required_steps: int
    configured_steps: int
    headroom_steps: int
    feasible: bool
    step_rad: float
    maximum_span_rad: float
    minimum_possible_release_threshold_nm: float
    maximum_torque_delta_gate_nm: float
    lowpass_alpha: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "travel_steps": self.travel_steps,
            "filter_tail_steps": self.filter_tail_steps,
            "tracking_lag_steps": self.tracking_lag_steps,
            "confirm_steps": self.confirm_steps,
            "required_steps": self.required_steps,
            "configured_steps": self.configured_steps,
            "headroom_steps": self.headroom_steps,
            "feasible": self.feasible,
            "step_rad": self.step_rad,
            "maximum_span_rad": self.maximum_span_rad,
            "minimum_possible_release_threshold_nm": (
                self.minimum_possible_release_threshold_nm
            ),
            "maximum_torque_delta_gate_nm": (
                self.maximum_torque_delta_gate_nm
            ),
            "lowpass_alpha": self.lowpass_alpha,
        }


def _finite_positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def release_budget_feasibility(
    *,
    closed_targets_rad: Sequence[float],
    open_targets_rad: Sequence[float],
    release_rate_rad_s: float,
    sample_period_s: float,
    lowpass_alpha: float,
    minimum_release_delta_nm: float,
    release_ratio: float,
    minimum_contact_delta_nm: float,
    maximum_release_tracking_error_rad: float,
    release_confirm_steps: int,
    maximum_torque_delta_gate_nm: float,
    configured_steps: int,
) -> ReleaseBudgetEvidence:
    '''Compute the frozen discrete release-budget lower bound (018 formula).

    travel_steps = ceil(max|closed-open| / (rate*dt))
    filter_tail  = ceil(log(threshold / gate) / log(1 - alpha))
    tracking_lag = ceil(tracking_error / (rate*dt))
    required     = travel + filter_tail + tracking_lag + confirm

    Every input is a parameter; nothing here is hardcoded to the current
    targets, 240 Hz or the 18-step confirm window.  A configured budget
    below the required bound is reported as infeasible (never silently).
    '''
    rate = _finite_positive(release_rate_rad_s, "release_rate_rad_s")
    period = _finite_positive(sample_period_s, "sample_period_s")
    alpha = float(lowpass_alpha)
    if isinstance(alpha, bool) or not math.isfinite(alpha):
        raise ValueError("lowpass_alpha must be a finite number")
    if not 0.0 < alpha < 1.0:
        raise ValueError("lowpass_alpha must lie in (0, 1)")
    ratio = float(release_ratio)
    if isinstance(ratio, bool) or not math.isfinite(ratio):
        raise ValueError("release_ratio must be a finite number")
    if not 0.0 < ratio < 1.0:
        raise ValueError("release_ratio must lie in (0, 1)")
    minimum_release_delta = float(minimum_release_delta_nm)
    minimum_contact_delta = float(minimum_contact_delta_nm)
    if (
        isinstance(minimum_release_delta_nm, bool)
        or isinstance(minimum_contact_delta_nm, bool)
        or not math.isfinite(minimum_release_delta)
        or not math.isfinite(minimum_contact_delta)
        or minimum_release_delta < 0.0
        or minimum_contact_delta < 0.0
    ):
        raise ValueError(
            "release thresholds must be finite non-negative numbers"
        )
    tracking_error = _finite_positive(
        maximum_release_tracking_error_rad,
        "maximum_release_tracking_error_rad",
    )
    gate = _finite_positive(
        maximum_torque_delta_gate_nm, "maximum_torque_delta_gate_nm"
    )
    if isinstance(release_confirm_steps, bool) or not isinstance(
        release_confirm_steps, int
    ) or release_confirm_steps < 1:
        raise ValueError(
            "release_confirm_steps must be a positive integer"
        )
    if isinstance(configured_steps, bool) or not isinstance(
        configured_steps, int
    ) or configured_steps < 1:
        raise ValueError("configured_steps must be a positive integer")
    closed = tuple(float(value) for value in closed_targets_rad)
    open_ = tuple(float(value) for value in open_targets_rad)
    if not closed or len(closed) != len(open_):
        raise ValueError("closed/open target vectors need equal length")
    if not all(math.isfinite(value) for value in closed + open_):
        raise ValueError("targets must be finite")
    threshold = max(minimum_release_delta, ratio * minimum_contact_delta)
    if threshold <= 0.0:
        raise ValueError(
            "minimum possible release threshold must be positive"
        )
    if gate <= threshold:
        raise ValueError(
            "maximum_torque_delta_gate_nm must exceed the minimum possible "
            "release threshold"
        )
    step = rate * period
    if step <= 0.0:
        raise ValueError("release step must be positive")
    maximum_span = max(abs(c - o) for c, o in zip(closed, open_))
    travel_steps = math.ceil(maximum_span / step)
    filter_tail_steps = math.ceil(
        math.log(threshold / gate) / math.log(1.0 - alpha)
    )
    tracking_lag_steps = math.ceil(tracking_error / step)
    required_steps = (
        travel_steps
        + filter_tail_steps
        + tracking_lag_steps
        + release_confirm_steps
    )
    headroom_steps = configured_steps - required_steps
    return ReleaseBudgetEvidence(
        travel_steps=travel_steps,
        filter_tail_steps=filter_tail_steps,
        tracking_lag_steps=tracking_lag_steps,
        confirm_steps=release_confirm_steps,
        required_steps=required_steps,
        configured_steps=configured_steps,
        headroom_steps=headroom_steps,
        feasible=bool(configured_steps >= required_steps),
        step_rad=step,
        maximum_span_rad=maximum_span,
        minimum_possible_release_threshold_nm=threshold,
        maximum_torque_delta_gate_nm=gate,
        lowpass_alpha=alpha,
    )


class SingleFingerContactTest:
    '''Baseline -> approach -> confirm -> SOFT_HOLD -> commanded release.'''

    def __init__(
        self,
        config: SingleFingerContactConfig,
        detector_config: FingerContactDetectorConfig,
        *,
        finger: str,
        open_target_rad: float,
        closed_target_rad: float,
    ):
        if finger not in FINGERS:
            raise ValueError(f"unknown finger {finger!r}")
        if not math.isfinite(float(open_target_rad)) or not math.isfinite(
            float(closed_target_rad)
        ):
            raise ValueError("open/closed targets must be finite")
        if float(open_target_rad) == float(closed_target_rad):
            raise ValueError("single-finger test needs a closure direction")
        self.config = config
        self.finger = finger
        self.open_target = float(open_target_rad)
        self.closed_target = float(closed_target_rad)
        self.direction = 1.0 if self.closed_target > self.open_target else -1.0
        self.release_direction = -self.direction
        self.detector = FingerContactDetector(detector_config, name=finger)
        self.target = self.open_target
        self.state = FingerContactState.APPROACH
        self.step = 0
        self.soft_hold_step = 0
        self.release_armed = False
        self.release_step = 0
        self.release_confirm = 0
        self.release_start_target: float | None = None
        self.transition_events: list[dict[str, Any]] = []
        self._step_events: list[dict[str, Any]] = []
        self.failed = False
        self.failure_reason: str | None = None

    def calibrate(self, baseline_samples_nm: Sequence[float]) -> None:
        self.detector.calibrate(baseline_samples_nm)
        self.state = FingerContactState.APPROACH

    @property
    def detector_test_passed(self) -> bool:
        return bool(
            not self.failed
            and self.state == FingerContactState.RELEASE_CONFIRMED
        )

    def _fail(self, reason: str) -> None:
        detector_before = self.detector.state.value
        self.failed = True
        self.failure_reason = reason
        self.detector.fail()
        self.state = FingerContactState.FAILED
        self._record_transition(
            detector_before, FingerContactState.FAILED.value
        )

    def _bounded_target(self, value: float) -> float:
        lower = min(self.open_target, self.closed_target)
        upper = max(self.open_target, self.closed_target)
        return min(max(value, lower), upper)

    def _record_transition(
        self, from_state: str, to_state: str
    ) -> None:
        if from_state == to_state:
            return
        event = {
            "step": self.step,
            "finger": self.finger,
            "from": from_state,
            "to": to_state,
        }
        self.transition_events.append(event)
        self._step_events.append(event)

    def _command(
        self,
        pre_state: str,
        observation: FingerContactObservation | None,
        release_conditions: Mapping[str, Any] | None = None,
    ) -> SingleFingerCommand:
        events = list(self._step_events)
        stiffness = (
            SOFT_HOLD_STIFFNESS_SCALE
            if self.state == FingerContactState.SOFT_HOLD
            else 1.0
        )
        return SingleFingerCommand(
            finger=self.finger,
            target_rad=self.target,
            stiffness_scale=stiffness,
            state=self.state,
            observation=observation,
            soft_hold_step=self.soft_hold_step,
            soft_hold_steps_configured=self.config.soft_hold_steps,
            release_step=self.release_step,
            release_confirm_steps_configured=(
                self.detector.config.release_confirm_steps
            ),
            release_conditions=dict(release_conditions or {}),
            failed=self.failed,
            failure_reason=self.failure_reason,
            detector_test_passed=self.detector_test_passed,
            evidence={
                "pre_state": pre_state,
                "post_state": self.state.value,
                "transition_events": events,
                "step": self.step,
                "release_confirm": self.release_confirm,
                "release_start_target_rad": self.release_start_target,
            },
        )

    def update(
        self,
        measured_torque_nm: float,
        joint_position_rad: float,
        joint_velocity_rad_s: float,
        *,
        timestamp_s: float,
    ) -> SingleFingerCommand:
        pre_state = self.state.value
        self._step_events = []
        if self.failed:
            return self._command(pre_state, None)
        self.step += 1
        detector_pre_state = self.detector.state.value
        try:
            observation = self.detector.update(
                measured_torque_nm,
                joint_position_rad=joint_position_rad,
                joint_velocity_rad_s=joint_velocity_rad_s,
                commanded_position_rad=self.target,
                timestamp_s=timestamp_s,
            )
        except ValueError as error:
            self._record_transition(
                detector_pre_state, self.detector.state.value
            )
            self._fail(f"nonfinite_or_stale_input: {error}")
            return self._command(pre_state, None)
        # Preserve the detector's own intra-step transition (for example
        # CONTACT_CANDIDATE -> CONTACT_CONFIRMED) before any controller-driven
        # transition is appended.
        self._record_transition(
            detector_pre_state, self.detector.state.value
        )

        if observation.state in (
            FingerContactState.SLIP_SUSPECTED,
            FingerContactState.FAILED,
        ):
            self._fail(
                f"{observation.state.value.lower()}_during_characterization"
            )
            return self._command(pre_state, observation)

        if self.state == FingerContactState.APPROACH:
            if observation.state == FingerContactState.CONTACT_CONFIRMED:
                # Latch the confirmed target and enter SOFT_HOLD immediately;
                # the hold window counts from the following updates.
                self.detector.mark_soft_hold()
                self.state = FingerContactState.SOFT_HOLD
                self.soft_hold_step = 0
                self._record_transition(
                    FingerContactState.CONTACT_CONFIRMED.value,
                    FingerContactState.SOFT_HOLD.value,
                )
            else:
                self._advance_approach()
            if (
                not self.failed
                and self.target == self.closed_target
                and observation.state
                not in (
                    FingerContactState.CONTACT_CONFIRMED,
                    FingerContactState.SOFT_HOLD,
                )
            ):
                self._fail("approach_closed_limit_without_contact")
            if (
                self.step >= self.config.maximum_approach_steps
                and not self.failed
            ):
                self._fail("approach_step_budget_exhausted")
            return self._command(pre_state, observation)

        if (
            self.state == FingerContactState.SOFT_HOLD
            and self.release_armed
        ):
            # The 24th subsequent SOFT_HOLD update completed on the previous
            # step; only now does the commanded release start, with open-loop
            # motion beginning in this very update.
            self.detector.begin_commanded_release()
            self.state = FingerContactState.RELEASE_COMMANDED
            self.release_step = 0
            self.release_confirm = 0
            self.release_start_target = self.target
            self.release_armed = False
            self._record_transition(
                FingerContactState.SOFT_HOLD.value,
                FingerContactState.RELEASE_COMMANDED.value,
            )

        if self.state == FingerContactState.SOFT_HOLD:
            self.soft_hold_step += 1
            if self.soft_hold_step >= self.config.soft_hold_steps:
                # Still SOFT_HOLD for this output: the release command is
                # armed but must not shorten the hold by one update.
                self.release_armed = True
            return self._command(pre_state, observation)

        if self.state == FingerContactState.RELEASE_COMMANDED:
            self.release_step += 1
            self.target = self._bounded_target(
                self.target
                + self.release_direction
                * self.config.release_rate_rad_s
                * self.detector.config.sample_period_s
            )
            assert self.release_start_target is not None
            load_ok = bool(
                observation.absolute_load_nm
                <= self.detector.release_threshold_nm
            )
            travel = abs(self.target - self.release_start_target)
            travel_ok = bool(
                travel >= self.config.minimum_release_travel_rad
            )
            tracking_ok = bool(
                abs(self.target - joint_position_rad)
                <= self.config.maximum_release_tracking_error_rad
            )
            conditions: dict[str, Any] = {
                "load_ok": load_ok,
                "travel_ok": travel_ok,
                "tracking_ok": tracking_ok,
                "absolute_load_nm": observation.absolute_load_nm,
                "release_threshold_nm": self.detector.release_threshold_nm,
                "travel_rad": travel,
                "tracking_error_rad": abs(
                    self.target - joint_position_rad
                ),
            }
            if load_ok and travel_ok and tracking_ok:
                self.release_confirm += 1
            else:
                self.release_confirm = 0
            if (
                self.release_confirm
                >= self.detector.config.release_confirm_steps
            ):
                self.detector.confirm_commanded_release()
                self.state = FingerContactState.RELEASE_CONFIRMED
                self._record_transition(
                    FingerContactState.RELEASE_COMMANDED.value,
                    FingerContactState.RELEASE_CONFIRMED.value,
                )
            elif self.release_step >= self.config.maximum_release_steps:
                self._fail("release_step_budget_exhausted")
            return self._command(pre_state, observation, conditions)

        # RELEASE_CONFIRMED: the finger stays at its open command.
        return self._command(pre_state, observation)

    def _advance_approach(self) -> None:
        self.target = self._bounded_target(
            self.target
            + self.direction
            * self.config.approach_rate_rad_s
            * self.detector.config.sample_period_s
        )
