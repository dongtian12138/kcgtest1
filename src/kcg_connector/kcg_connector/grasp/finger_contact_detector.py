"""Contact detection from one finger-root joint reaction proxy.

The detector deliberately has no geometry, object pose, collider, or contact
report input.  Its thresholds are simulation tuning values, not calibrated
hardware force thresholds and not fingertip tactile measurements.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any, Iterable


class FingerContactState(str, Enum):
    APPROACH = "APPROACH"
    CONTACT_CANDIDATE = "CONTACT_CANDIDATE"
    CONTACT_CONFIRMED = "CONTACT_CONFIRMED"
    SOFT_HOLD = "SOFT_HOLD"
    LOAD_BUILD = "LOAD_BUILD"
    STABLE_CONTACT = "STABLE_CONTACT"
    SLIP_SUSPECTED = "SLIP_SUSPECTED"
    RELEASE_COMMANDED = "RELEASE_COMMANDED"
    RELEASE_CONFIRMED = "RELEASE_CONFIRMED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class FingerContactDetectorConfig:
    sample_period_s: float
    lowpass_alpha: float
    derivative_alpha: float
    contact_sigma_multiplier: float
    minimum_contact_delta_nm: float
    release_ratio: float
    minimum_release_delta_nm: float
    minimum_rise_rate_nm_s: float
    maximum_stall_velocity_rad_s: float
    minimum_tracking_error_rad: float
    confirm_steps: int
    release_confirm_steps: int
    maximum_sample_gap_s: float
    position_velocity_window_steps: int
    threshold_label: str = "SIM_TUNING_ONLY"

    def __post_init__(self) -> None:
        positive = (
            "sample_period_s",
            "contact_sigma_multiplier",
            "minimum_contact_delta_nm",
            "minimum_release_delta_nm",
            "minimum_rise_rate_nm_s",
            "maximum_stall_velocity_rad_s",
            "minimum_tracking_error_rad",
            "maximum_sample_gap_s",
        )
        for name in positive:
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("lowpass_alpha", "derivative_alpha", "release_ratio"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be in (0, 1)")
        if self.confirm_steps < 2 or self.release_confirm_steps < 2:
            raise ValueError("contact/release confirmation must span multiple steps")
        if self.maximum_sample_gap_s < self.sample_period_s:
            raise ValueError("maximum_sample_gap_s is below one sample period")
        if (
            isinstance(self.position_velocity_window_steps, bool)
            or not isinstance(self.position_velocity_window_steps, int)
            or not 1 <= self.position_velocity_window_steps <= 24
        ):
            raise ValueError(
                "position_velocity_window_steps must be an integer in [1, 24]"
            )
        if self.threshold_label != "SIM_TUNING_ONLY":
            raise ValueError("detector thresholds must remain SIM_TUNING_ONLY")


@dataclass(frozen=True)
class FingerContactObservation:
    state: FingerContactState
    raw_delta_nm: float
    filtered_delta_nm: float
    filtered_rate_nm_s: float
    absolute_load_nm: float
    contact_threshold_nm: float
    release_threshold_nm: float
    load_score: float
    stalled: bool
    reported_joint_velocity_rad_s: float
    position_derived_velocity_rad_s: float | None
    velocity_disagreement_rad_s: float | None
    position_velocity_sample_count: int
    position_velocity_window_span_s: float | None
    stall_velocity_source: str
    candidate_steps: int
    release_steps: int
    step: int

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["state"] = self.state.value
        return result


class FingerContactDetector:
    """Stateful baseline, filter, hysteresis and debounce for one channel."""

    def __init__(self, config: FingerContactDetectorConfig, *, name: str):
        if not name:
            raise ValueError("finger detector name is required")
        self.config = config
        self.name = name
        self.state = FingerContactState.APPROACH
        self.baseline_mean_nm: float | None = None
        self.baseline_std_nm: float | None = None
        self._filtered = 0.0
        self._filtered_rate = 0.0
        self._previous_filtered = 0.0
        self._previous_time: float | None = None
        self._candidate_steps = 0
        self._release_steps = 0
        self._step = 0
        self._position_history: deque[tuple[float, float]] = deque(
            maxlen=self.config.position_velocity_window_steps + 1
        )

    def calibrate(self, samples_nm: Iterable[float]) -> None:
        samples = tuple(float(value) for value in samples_nm)
        if len(samples) < max(8, self.config.confirm_steps * 2):
            raise ValueError("insufficient no-contact baseline samples")
        if not all(math.isfinite(value) for value in samples):
            raise ValueError("baseline contains non-finite samples")
        mean = sum(samples) / len(samples)
        variance = sum((value - mean) ** 2 for value in samples) / len(samples)
        self.baseline_mean_nm = mean
        self.baseline_std_nm = math.sqrt(max(0.0, variance))
        self._filtered = 0.0
        self._previous_filtered = 0.0
        self._filtered_rate = 0.0
        self._previous_time = None
        self._candidate_steps = 0
        self._release_steps = 0
        self._step = 0
        self._position_history.clear()
        self.state = FingerContactState.APPROACH

    @property
    def calibrated(self) -> bool:
        return self.baseline_mean_nm is not None and self.baseline_std_nm is not None

    @property
    def contact_threshold_nm(self) -> float:
        if not self.calibrated:
            raise RuntimeError("detector is not baseline calibrated")
        return max(
            self.config.minimum_contact_delta_nm,
            self.config.contact_sigma_multiplier * float(self.baseline_std_nm),
        )

    @property
    def release_threshold_nm(self) -> float:
        return max(
            self.config.minimum_release_delta_nm,
            self.config.release_ratio * self.contact_threshold_nm,
        )

    def _transition(self, state: FingerContactState) -> None:
        if self.state in (FingerContactState.FAILED, FingerContactState.SLIP_SUSPECTED):
            return
        self.state = state

    def mark_soft_hold(self) -> None:
        if self.state != FingerContactState.CONTACT_CONFIRMED:
            raise RuntimeError(f"{self.name}: SOFT_HOLD before contact confirmation")
        self._transition(FingerContactState.SOFT_HOLD)

    def begin_commanded_release(self) -> None:
        """Enter the commanded-release phase from a completed SOFT_HOLD.

        During RELEASE_COMMANDED the slip detector is deliberately disarmed:
        the finger is opening by command, so load decay and tracking motion
        are expected and must not be judged as SLIP_SUSPECTED.  Release
        confirmation is owned by the single-finger controller, not here.
        """
        if self.state != FingerContactState.SOFT_HOLD:
            raise RuntimeError(
                f"{self.name}: commanded release before SOFT_HOLD completion"
            )
        self._transition(FingerContactState.RELEASE_COMMANDED)

    def confirm_commanded_release(self) -> None:
        if self.state != FingerContactState.RELEASE_COMMANDED:
            raise RuntimeError(
                f"{self.name}: release confirmation before release command"
            )
        self._transition(FingerContactState.RELEASE_CONFIRMED)

    def mark_load_build(self) -> None:
        if self.state not in (
            FingerContactState.SOFT_HOLD,
            FingerContactState.CONTACT_CONFIRMED,
        ):
            raise RuntimeError(f"{self.name}: LOAD_BUILD from {self.state.value}")
        self._transition(FingerContactState.LOAD_BUILD)

    def mark_stable(self) -> None:
        if self.state != FingerContactState.LOAD_BUILD:
            raise RuntimeError(f"{self.name}: STABLE_CONTACT from {self.state.value}")
        self._transition(FingerContactState.STABLE_CONTACT)

    def fail(self) -> None:
        self.state = FingerContactState.FAILED

    def update(
        self,
        measured_torque_nm: float,
        *,
        joint_position_rad: float,
        joint_velocity_rad_s: float,
        commanded_position_rad: float,
        timestamp_s: float,
    ) -> FingerContactObservation:
        if not self.calibrated:
            raise RuntimeError("detector is not baseline calibrated")
        values = (
            measured_torque_nm,
            joint_position_rad,
            joint_velocity_rad_s,
            commanded_position_rad,
            timestamp_s,
        )
        if not all(math.isfinite(float(value)) for value in values):
            self.fail()
            raise ValueError(f"{self.name}: non-finite detector input")
        if self._previous_time is not None:
            dt = timestamp_s - self._previous_time
            if dt <= 0.0 or dt > self.config.maximum_sample_gap_s:
                self.fail()
                raise ValueError(f"{self.name}: stale or non-monotonic torque sample")
        else:
            dt = self.config.sample_period_s

        raw_delta = float(measured_torque_nm) - float(self.baseline_mean_nm)
        self._previous_filtered = self._filtered
        self._filtered += self.config.lowpass_alpha * (raw_delta - self._filtered)
        raw_rate = (self._filtered - self._previous_filtered) / dt
        self._filtered_rate += self.config.derivative_alpha * (
            raw_rate - self._filtered_rate
        )
        self._previous_time = float(timestamp_s)
        self._step += 1

        absolute_load = abs(self._filtered)
        contact_threshold = self.contact_threshold_nm
        release_threshold = self.release_threshold_nm
        tracking_error = abs(commanded_position_rad - joint_position_rad)

        self._position_history.append(
            (float(timestamp_s), float(joint_position_rad))
        )
        position_velocity = None
        window_span = None
        sample_count = len(self._position_history)
        if sample_count >= self.config.position_velocity_window_steps + 1:
            oldest_time, oldest_position = self._position_history[0]
            newest_time, newest_position = self._position_history[-1]
            window_span = float(newest_time - oldest_time)
            if not math.isfinite(window_span) or window_span <= 0.0:
                self.fail()
                raise ValueError(
                    f"{self.name}: non-positive position window span"
                )
            position_velocity = float(
                (newest_position - oldest_position) / window_span
            )
            if not math.isfinite(position_velocity):
                self.fail()
                raise ValueError(
                    f"{self.name}: non-finite position-derived velocity"
                )
        disagreement = (
            None
            if position_velocity is None
            else float(abs(float(joint_velocity_rad_s) - position_velocity))
        )
        if disagreement is not None and not math.isfinite(disagreement):
            self.fail()
            raise ValueError(
                f"{self.name}: non-finite velocity disagreement"
            )
        stall_source = (
            "position_history"
            if position_velocity is not None
            else "insufficient_history"
        )
        stalled = bool(
            position_velocity is not None
            and abs(position_velocity)
            <= self.config.maximum_stall_velocity_rad_s
            and tracking_error >= self.config.minimum_tracking_error_rad
        )
        rising = abs(self._filtered_rate) >= self.config.minimum_rise_rate_nm_s
        candidate = absolute_load >= contact_threshold and (stalled or rising)

        if self.state == FingerContactState.APPROACH:
            if candidate:
                self._candidate_steps = 1
                self._transition(FingerContactState.CONTACT_CANDIDATE)
        elif self.state == FingerContactState.CONTACT_CANDIDATE:
            if candidate:
                self._candidate_steps += 1
                if self._candidate_steps >= self.config.confirm_steps:
                    self._transition(FingerContactState.CONTACT_CONFIRMED)
                    self._release_steps = 0
            else:
                self._candidate_steps = 0
                if absolute_load <= release_threshold:
                    self._transition(FingerContactState.APPROACH)
        elif self.state in (
            FingerContactState.CONTACT_CONFIRMED,
            FingerContactState.SOFT_HOLD,
            FingerContactState.LOAD_BUILD,
            FingerContactState.STABLE_CONTACT,
        ):
            # RELEASE_COMMANDED/RELEASE_CONFIRMED are intentionally absent:
            # slip detection is disarmed during a commanded opening.
            if absolute_load <= release_threshold:
                self._release_steps += 1
                if self._release_steps >= self.config.release_confirm_steps:
                    self._transition(FingerContactState.SLIP_SUSPECTED)
            else:
                self._release_steps = 0

        return FingerContactObservation(
            state=self.state,
            raw_delta_nm=raw_delta,
            filtered_delta_nm=self._filtered,
            filtered_rate_nm_s=self._filtered_rate,
            absolute_load_nm=absolute_load,
            contact_threshold_nm=contact_threshold,
            release_threshold_nm=release_threshold,
            load_score=absolute_load / contact_threshold,
            stalled=stalled,
            reported_joint_velocity_rad_s=float(joint_velocity_rad_s),
            position_derived_velocity_rad_s=position_velocity,
            velocity_disagreement_rad_s=disagreement,
            position_velocity_sample_count=sample_count,
            position_velocity_window_span_s=window_span,
            stall_velocity_source=stall_source,
            candidate_steps=self._candidate_steps,
            release_steps=self._release_steps,
            step=self._step,
        )
