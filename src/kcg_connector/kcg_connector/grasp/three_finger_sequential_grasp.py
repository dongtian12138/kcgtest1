"""Bounded sequential-compliant closing for the D38999 three-finger hand."""

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


FINGERS = ("f1", "f2", "f3")

CONTROLLER_PHASE_APPROACH = "approach"
CONTROLLER_PHASE_STABLE_HOLD = "stable_hold"
CONTROLLER_PHASE_CONSOLIDATION_RAMP = "consolidation_ramp"
CONTROLLER_PHASE_CONSOLIDATION_WINDOW = "consolidation_window"
CONTROLLER_PHASE_LIFT_READY = "lift_ready"

# First falsifiable lift-readiness candidate (SIM_TUNING_ONLY_A_CANDIDATE):
# after STABLE_CONTACT the finger targets freeze exactly and the stiffness
# ramps 0.35 -> 1.0 over consolidation_ramp_steps physics steps, then
# holds consolidation_window_steps steps at the final scale.  Only a full
# clean pass under the frozen lift gates yields LIFT_READY.  These are not
# hardware thresholds and not a safety law.
CONSOLIDATION_THRESHOLD_LABEL = "SIM_TUNING_ONLY_A_CANDIDATE"
DEFAULT_SOFT_HOLD_STIFFNESS_SCALE = 0.35
DEFAULT_CONSOLIDATION_FINAL_STIFFNESS_SCALE = 1.0
DEFAULT_CONSOLIDATION_RAMP_STEPS = 120
DEFAULT_CONSOLIDATION_WINDOW_STEPS = 240


@dataclass(frozen=True)
class SequentialGraspConfig:
    detector: FingerContactDetectorConfig
    sample_period_s: float
    approach_rate_rad_s: float
    soft_hold_preload_rad: float
    load_build_rate_rad_s: float
    balance_gain_rad_per_load: float
    maximum_balance_step_rad: float
    maximum_balance_total_rad: float
    probe_increment_rad: float
    probe_settle_steps: int
    minimum_probe_response_nm: float
    maximum_probe_cross_coupling_ratio: float
    load_scale_nm: tuple[float, float, float]
    stable_minimum_normalized_load: float
    maximum_normalized_load_imbalance: float
    stable_confirm_steps: int
    maximum_approach_steps: int
    maximum_load_build_steps: int
    soft_hold_window_steps: int = 24
    soft_hold_stiffness_scale: float = DEFAULT_SOFT_HOLD_STIFFNESS_SCALE
    consolidation_final_stiffness_scale: float = (
        DEFAULT_CONSOLIDATION_FINAL_STIFFNESS_SCALE
    )
    consolidation_ramp_steps: int = DEFAULT_CONSOLIDATION_RAMP_STEPS
    consolidation_window_steps: int = DEFAULT_CONSOLIDATION_WINDOW_STEPS
    consolidation_threshold_label: str = CONSOLIDATION_THRESHOLD_LABEL
    probe_mode: str = "per_finger"

    def __post_init__(self) -> None:
        for name in (
            "sample_period_s",
            "approach_rate_rad_s",
            "soft_hold_preload_rad",
            "load_build_rate_rad_s",
            "balance_gain_rad_per_load",
            "maximum_balance_step_rad",
            "maximum_balance_total_rad",
            "probe_increment_rad",
            "minimum_probe_response_nm",
            "maximum_probe_cross_coupling_ratio",
            "stable_minimum_normalized_load",
            "maximum_normalized_load_imbalance",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if len(self.load_scale_nm) != 3 or any(
            not math.isfinite(value) or value <= 0.0
            for value in self.load_scale_nm
        ):
            raise ValueError("load_scale_nm must contain three positive values")
        if self.stable_confirm_steps < 2:
            raise ValueError("stable confirmation must span multiple steps")
        if self.probe_settle_steps < 2:
            raise ValueError("probe settling must span multiple steps")
        if self.probe_mode not in ("per_finger", "collective"):
            raise ValueError(
                "probe_mode must be per_finger or collective"
            )
        for name in ("maximum_approach_steps", "maximum_load_build_steps"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(
                    f"{name} must be a positive integer step budget"
                )
        if (
            isinstance(self.soft_hold_stiffness_scale, bool)
            or not math.isfinite(self.soft_hold_stiffness_scale)
            or not 0.0 < self.soft_hold_stiffness_scale <= 1.0
        ):
            raise ValueError(
                "soft_hold_stiffness_scale must be finite in (0, 1]"
            )
        if (
            isinstance(self.consolidation_final_stiffness_scale, bool)
            or not math.isfinite(self.consolidation_final_stiffness_scale)
            or not (
                self.soft_hold_stiffness_scale
                < self.consolidation_final_stiffness_scale
                <= 1.0
            )
        ):
            raise ValueError(
                "consolidation_final_stiffness_scale must satisfy "
                "soft_hold < final <= 1.0 and be finite"
            )
        for name in ("consolidation_ramp_steps", "consolidation_window_steps"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 2
            ):
                raise ValueError(f"{name} must be a positive integer >= 2")
        if self.consolidation_threshold_label != CONSOLIDATION_THRESHOLD_LABEL:
            raise ValueError(
                "consolidation thresholds must remain "
                "SIM_TUNING_ONLY_A_CANDIDATE"
            )
        if isinstance(self.soft_hold_window_steps, bool) or (
            not isinstance(self.soft_hold_window_steps, int)
        ) or self.soft_hold_window_steps < 2:
            raise ValueError(
                "soft_hold_window_steps must be a positive integer >= 2"
            )


@dataclass(frozen=True)
class SequentialGraspCommand:
    finger_targets_rad: tuple[float, float, float]
    finger_stiffness_scale: tuple[float, float, float]
    observations: Mapping[str, FingerContactObservation]
    contact_order: tuple[str, ...]
    stable: bool
    failed: bool
    failure_reason: str | None
    normalized_loads: tuple[float, float, float]
    normalized_load_imbalance: float
    probe_response_nm: tuple[float, ...]
    lift_ready: bool = False
    controller_phase: str = ""
    evidence: Mapping[str, Any] = field(default_factory=dict)


class ThreeFingerSequentialGrasp:
    """Three independent detectors plus bounded target/load balancing."""

    def __init__(
        self,
        config: SequentialGraspConfig,
        *,
        open_targets_rad: Sequence[float],
        closed_targets_rad: Sequence[float],
        start_delay_steps: Sequence[int] = (0, 0, 0),
    ):
        if len(open_targets_rad) != 3 or len(closed_targets_rad) != 3:
            raise ValueError("exactly three finger targets are required")
        if len(start_delay_steps) != 3 or any(value < 0 for value in start_delay_steps):
            raise ValueError("start_delay_steps must be three nonnegative integers")
        self.config = config
        self.open_targets = tuple(float(value) for value in open_targets_rad)
        self.closed_targets = tuple(float(value) for value in closed_targets_rad)
        self.direction = tuple(
            1.0 if end > start else -1.0
            for start, end in zip(self.open_targets, self.closed_targets)
        )
        if any(start == end for start, end in zip(self.open_targets, self.closed_targets)):
            raise ValueError("each mapped finger joint needs a closure direction")
        self.start_delay_steps = tuple(int(value) for value in start_delay_steps)
        self.detectors = {
            name: FingerContactDetector(config.detector, name=name)
            for name in FINGERS
        }
        self.targets = list(self.open_targets)
        self.contact_order: list[str] = []
        self.contact_targets: dict[str, float] = {}
        self.balance_total = [0.0, 0.0, 0.0]
        self.step = 0
        self.load_build_step = 0
        self.stable_steps = 0
        self.probe_index = 0
        self.probe_step = 0
        self.probe_initial_settle_steps = config.probe_settle_steps
        self.probe_baseline_loads: tuple[float, float, float] | None = None
        self.probe_response_nm: list[float] = []
        self.probe_aggregate_response_nm: float | None = None
        self.soft_hold_window_step = 0
        self.soft_hold_window_armed = False
        self.soft_hold_window_complete = False
        self.transition_events: list[dict[str, Any]] = []
        self._step_events: list[dict[str, Any]] = []
        self.balance_delta_rad = [0.0, 0.0, 0.0]
        self.failed = False
        self.failure_reason: str | None = None
        self.controller_phase = CONTROLLER_PHASE_APPROACH
        self.consolidation_armed = False
        self.consolidation_complete = False
        self.consolidation_ramp_step = 0
        self.consolidation_window_step = 0
        self.consolidation_stiffness_scale = float(
            config.soft_hold_stiffness_scale
        )
        self._consolidation_scale_previous: float | None = None
        self._consolidation_scale_min: float | None = None
        self._consolidation_scale_max: float | None = None
        self._consolidation_scale_monotonic = True
        self._frozen_targets: tuple[float, float, float] | None = None

    def begin_consolidation(self) -> None:
        """Arm the bounded lift-readiness consolidation exactly once.

        Allowed only after the three fingers reached STABLE_CONTACT and the
        controller has not failed.  The finger targets are frozen bit-exact
        from here on; the stiffness ramps soft -> final over
        consolidation_ramp_steps updates and then holds for
        consolidation_window_steps updates before lift_ready flips true.
        """
        if self.failed:
            raise RuntimeError(
                "cannot arm consolidation after controller failure"
            )
        if not self.stable:
            raise RuntimeError(
                "consolidation requires all three fingers STABLE_CONTACT"
            )
        if self.consolidation_armed:
            raise RuntimeError("consolidation can only be armed once")
        self.consolidation_armed = True
        self.consolidation_ramp_step = 0
        self.consolidation_window_step = 0
        self.consolidation_stiffness_scale = float(
            self.config.soft_hold_stiffness_scale
        )
        self.controller_phase = CONTROLLER_PHASE_CONSOLIDATION_RAMP
        if self._frozen_targets is None:
            self._frozen_targets = tuple(
                float(value) for value in self.targets
            )

    @property
    def lift_ready(self) -> bool:
        return self.consolidation_complete and not self.failed

    def _advance_consolidation(self) -> None:
        if self.consolidation_complete:
            return
        soft = float(self.config.soft_hold_stiffness_scale)
        final = float(self.config.consolidation_final_stiffness_scale)
        if self.consolidation_ramp_step < self.config.consolidation_ramp_steps:
            self.consolidation_ramp_step += 1
            scale = soft + (final - soft) * (
                float(self.consolidation_ramp_step)
                / float(self.config.consolidation_ramp_steps)
            )
            if self.consolidation_ramp_step >= (
                self.config.consolidation_ramp_steps
            ):
                scale = final
            scale = min(max(scale, soft), final)
            self.consolidation_stiffness_scale = scale
            self.controller_phase = CONTROLLER_PHASE_CONSOLIDATION_RAMP
        else:
            self.consolidation_window_step += 1
            self.consolidation_stiffness_scale = final
            self.controller_phase = CONTROLLER_PHASE_CONSOLIDATION_WINDOW
            if (
                self.consolidation_window_step
                >= self.config.consolidation_window_steps
            ):
                self.consolidation_complete = True
                self.controller_phase = CONTROLLER_PHASE_LIFT_READY
        if self._consolidation_scale_previous is not None and (
            self.consolidation_stiffness_scale + 1.0e-12
            < self._consolidation_scale_previous
        ):
            self._consolidation_scale_monotonic = False
        self._consolidation_scale_previous = (
            self.consolidation_stiffness_scale
        )
        self._consolidation_scale_min = (
            self.consolidation_stiffness_scale
            if self._consolidation_scale_min is None
            else min(
                self._consolidation_scale_min,
                self.consolidation_stiffness_scale,
            )
        )
        self._consolidation_scale_max = (
            self.consolidation_stiffness_scale
            if self._consolidation_scale_max is None
            else max(
                self._consolidation_scale_max,
                self.consolidation_stiffness_scale,
            )
        )

    def calibrate(self, baseline_samples_nm: Mapping[str, Sequence[float]]) -> None:
        if set(baseline_samples_nm) != set(FINGERS):
            raise ValueError("baseline samples must contain f1, f2 and f3")
        for name in FINGERS:
            self.detectors[name].calibrate(baseline_samples_nm[name])

    def _bounded_target(self, index: int, value: float) -> float:
        start = self.open_targets[index]
        end = self.closed_targets[index]
        return min(max(value, min(start, end)), max(start, end))

    def _record_transition(
        self, finger: str, from_state: str, to_state: str
    ) -> None:
        if from_state == to_state:
            return
        event = {
            "step": self.step,
            "finger": finger,
            "from": from_state,
            "to": to_state,
        }
        self.transition_events.append(event)
        self._step_events.append(event)

    def _fail(self, reason: str, finger: str | None = None) -> None:
        if finger is not None:
            before = self.detectors[finger].state.value
        else:
            before = None
        self.failed = True
        self.failure_reason = reason
        for detector in self.detectors.values():
            detector.fail()
        if finger is not None:
            self._record_transition(
                finger, before, FingerContactState.FAILED.value
            )

    def update(
        self,
        torque_nm: Sequence[float],
        joint_positions_rad: Sequence[float],
        joint_velocities_rad_s: Sequence[float],
        *,
        timestamp_s: float,
    ) -> SequentialGraspCommand:
        if any(len(values) != 3 for values in (torque_nm, joint_positions_rad, joint_velocities_rad_s)):
            raise ValueError("three torque, position and velocity values are required")
        pre_states = {name: detector.state.value for name, detector in self.detectors.items()}
        self._step_events = []
        if self.failed:
            return self._command({}, (0.0, 0.0, 0.0), pre_states)
        self.step += 1
        self.balance_delta_rad = [0.0, 0.0, 0.0]
        observations: dict[str, FingerContactObservation] = {}
        for index, name in enumerate(FINGERS):
            detector_pre_state = self.detectors[name].state.value
            observations[name] = self.detectors[name].update(
                float(torque_nm[index]),
                joint_position_rad=float(joint_positions_rad[index]),
                joint_velocity_rad_s=float(joint_velocities_rad_s[index]),
                commanded_position_rad=float(self.targets[index]),
                timestamp_s=timestamp_s,
            )
            # Preserve the detector's own intra-step transition (for example
            # CONTACT_CANDIDATE -> CONTACT_CONFIRMED) before any
            # controller-driven transition is appended.
            self._record_transition(
                name, detector_pre_state, self.detectors[name].state.value
            )
            state = observations[name].state
            if state == FingerContactState.CONTACT_CONFIRMED:
                if name not in self.contact_order:
                    self.contact_order.append(name)
                    # Lock the target at the confirmation step (with the
                    # configured one-shot preload) and keep it locked while
                    # the other fingers complete their own confirmations.
                    preload = self.direction[index] * self.config.soft_hold_preload_rad
                    self.contact_targets[name] = self._bounded_target(
                        index, self.targets[index] + preload
                    )
                self.targets[index] = self.contact_targets[name]
                self.detectors[name].mark_soft_hold()
                self._record_transition(
                    name,
                    FingerContactState.CONTACT_CONFIRMED.value,
                    FingerContactState.SOFT_HOLD.value,
                )
            elif state in (FingerContactState.SLIP_SUSPECTED, FingerContactState.FAILED):
                self._fail(
                    f"{name}_{state.value.lower()}", finger=name
                )
            if self.failed:
                # A slip/failure already failed the whole controller:
                # do not let later fingers overwrite the first failure reason.
                break

        all_contacted = all(
            detector.state
            in (
                FingerContactState.SOFT_HOLD,
                FingerContactState.LOAD_BUILD,
                FingerContactState.STABLE_CONTACT,
            )
            for detector in self.detectors.values()
        )
        if not all_contacted and not self.failed:
            increment = self.config.approach_rate_rad_s * self.config.sample_period_s
            for index, name in enumerate(FINGERS):
                if self.step <= self.start_delay_steps[index]:
                    continue
                if self.detectors[name].state in (
                    FingerContactState.APPROACH,
                    FingerContactState.CONTACT_CANDIDATE,
                ):
                    self.targets[index] = self._bounded_target(
                        index, self.targets[index] + self.direction[index] * increment
                    )
                    if self.targets[index] == self.closed_targets[index]:
                        self._fail(
                            f"{name}_closed_limit_without_contact",
                            finger=name,
                        )
                        break
            if self.step >= self.config.maximum_approach_steps and not self.failed:
                self._fail("approach_step_budget_exhausted")

        normalized = tuple(
            observations.get(name).absolute_load_nm / self.config.load_scale_nm[index]
            if name in observations
            else 0.0
            for index, name in enumerate(FINGERS)
        )
        imbalance = max(normalized) - min(normalized)

        if all_contacted and not self.failed:
            if not self.soft_hold_window_complete:
                # The update that confirms the last finger does not count:
                # the independent SOFT_HOLD window starts on the following
                # updates and must complete soft_hold_window_steps full
                # physics steps before any LOAD_BUILD/probe activity.  During
                # the window every detector keeps updating at low stiffness,
                # and any slip or failure fails closed.
                if not self.soft_hold_window_armed:
                    self.soft_hold_window_armed = True
                    return self._command(observations, normalized, pre_states)
                self.soft_hold_window_step += 1
                if (
                    self.soft_hold_window_step
                    < self.config.soft_hold_window_steps
                ):
                    return self._command(
                        observations, normalized, pre_states
                    )
                self.soft_hold_window_complete = True
                for detector_name, detector in self.detectors.items():
                    if detector.state == FingerContactState.SOFT_HOLD:
                        detector.mark_load_build()
                        self._record_transition(
                            detector_name,
                            FingerContactState.SOFT_HOLD.value,
                            FingerContactState.LOAD_BUILD.value,
                        )
                # The marking update performs no probe work: the existing
                # probe initial settle runs independently afterwards.
                return self._command(observations, normalized, pre_states)
            if (
                self.consolidation_armed
                or self.consolidation_complete
                or self.stable
            ):
                # Frozen-hold / consolidation: the three targets are
                # bit-identical to the frozen set captured when the hold
                # began.  No load-build base closure, no balance, no probe
                # work.  The detectors above already consumed this step's
                # timestamp/loads and any slip or failure already failed
                # the controller closed.
                if self._frozen_targets is None:
                    self._frozen_targets = tuple(
                        float(value) for value in self.targets
                    )
                elif tuple(float(value) for value in self.targets) != (
                    self._frozen_targets
                ):
                    self._fail("consolidation_target_invariant_broken")
                if self.consolidation_armed and not self.failed:
                    self._advance_consolidation()
                if not self.consolidation_armed and not self.failed:
                    self.controller_phase = CONTROLLER_PHASE_STABLE_HOLD
                return self._command(observations, normalized, pre_states)
            absolute_loads = tuple(
                observations[name].absolute_load_nm for name in FINGERS
            )
            if self.probe_index < 3:
                if self.probe_initial_settle_steps > 0:
                    self.probe_initial_settle_steps -= 1
                    return self._command(observations, normalized, pre_states)
                if self.config.probe_mode == "collective":
                    if self.probe_step == 0:
                        self.probe_baseline_loads = absolute_loads
                        for index in range(3):
                            self.targets[index] = self._bounded_target(
                                index,
                                self.targets[index]
                                + self.direction[index]
                                * self.config.probe_increment_rad,
                            )
                        self.probe_step = 1
                    elif self.probe_step < self.config.probe_settle_steps:
                        self.probe_step += 1
                    else:
                        assert self.probe_baseline_loads is not None
                        responses = tuple(
                            after - before
                            for after, before in zip(
                                absolute_loads,
                                self.probe_baseline_loads,
                            )
                        )
                        aggregate_response = sum(responses)
                        self.probe_aggregate_response_nm = (
                            aggregate_response
                        )
                        minimum_aggregate_response = (
                            3.0 * self.config.minimum_probe_response_nm
                        )
                        if aggregate_response < minimum_aggregate_response:
                            self._fail(
                                "collective_probe_response_invalid"
                            )
                        else:
                            self.probe_response_nm.extend(responses)
                            self.probe_index = 3
                            self.probe_step = 0
                            self.probe_baseline_loads = None
                    return self._command(
                        observations, normalized, pre_states
                    )
                if self.probe_step == 0:
                    self.probe_baseline_loads = absolute_loads
                    self.targets[self.probe_index] = self._bounded_target(
                        self.probe_index,
                        self.targets[self.probe_index]
                        + self.direction[self.probe_index]
                        * self.config.probe_increment_rad,
                    )
                    self.probe_step = 1
                elif self.probe_step < self.config.probe_settle_steps:
                    self.probe_step += 1
                else:
                    assert self.probe_baseline_loads is not None
                    response = absolute_loads[self.probe_index] - (
                        self.probe_baseline_loads[self.probe_index]
                    )
                    cross = max(
                        abs(after - before)
                        for index, (after, before) in enumerate(
                            zip(absolute_loads, self.probe_baseline_loads)
                        )
                        if index != self.probe_index
                    )
                    if response < self.config.minimum_probe_response_nm:
                        self._fail(
                            f"f{self.probe_index + 1}_probe_response_invalid",
                            finger=f"f{self.probe_index + 1}",
                        )
                    elif (
                        cross / response
                        > self.config.maximum_probe_cross_coupling_ratio
                    ):
                        self._fail(
                            f"f{self.probe_index + 1}_probe_cross_coupling",
                            finger=f"f{self.probe_index + 1}",
                        )
                    else:
                        self.probe_response_nm.append(response)
                        self.probe_index += 1
                        self.probe_step = 0
                        self.probe_baseline_loads = None
                return self._command(observations, normalized, pre_states)
            self.load_build_step += 1
            mean_load = sum(normalized) / 3.0
            base_step = self.config.load_build_rate_rad_s * self.config.sample_period_s
            for index in range(3):
                balance = self.config.balance_gain_rad_per_load * (
                    mean_load - normalized[index]
                )
                balance = max(
                    -self.config.maximum_balance_step_rad,
                    min(self.config.maximum_balance_step_rad, balance),
                )
                remaining = self.config.maximum_balance_total_rad - abs(
                    self.balance_total[index]
                )
                balance = max(-remaining, min(remaining, balance))
                self.balance_total[index] += balance
                self.balance_delta_rad[index] = balance
                delta = self.direction[index] * base_step + balance * self.direction[index]
                self.targets[index] = self._bounded_target(index, self.targets[index] + delta)

            stable_now = bool(
                min(normalized) >= self.config.stable_minimum_normalized_load
                and imbalance <= self.config.maximum_normalized_load_imbalance
            )
            self.stable_steps = self.stable_steps + 1 if stable_now else 0
            if self.stable_steps >= self.config.stable_confirm_steps:
                for detector_name, detector in self.detectors.items():
                    if detector.state == FingerContactState.LOAD_BUILD:
                        detector.mark_stable()
                        self._record_transition(
                            detector_name,
                            FingerContactState.LOAD_BUILD.value,
                            FingerContactState.STABLE_CONTACT.value,
                        )
            if self.load_build_step >= self.config.maximum_load_build_steps and not self.stable:
                self._fail("load_build_step_budget_exhausted")

        return self._command(observations, normalized, pre_states)

    @property
    def stable(self) -> bool:
        return all(
            detector.state == FingerContactState.STABLE_CONTACT
            for detector in self.detectors.values()
        )

    def _command(
        self,
        observations: Mapping[str, FingerContactObservation],
        normalized: Sequence[float],
        pre_states: Mapping[str, str],
    ) -> SequentialGraspCommand:
        values = tuple(float(value) for value in normalized)
        imbalance = max(values) - min(values) if len(values) == 3 else math.inf
        if self.consolidation_armed or self.consolidation_complete:
            hold_scale = float(self.consolidation_stiffness_scale)
            stiffness = tuple(hold_scale for _name in FINGERS)
        else:
            soft_scale = float(self.config.soft_hold_stiffness_scale)
            stiffness = tuple(
                soft_scale
                if self.detectors[name].state
                in (
                    FingerContactState.SOFT_HOLD,
                    FingerContactState.LOAD_BUILD,
                    FingerContactState.STABLE_CONTACT,
                )
                else 1.0
                for name in FINGERS
            )
        post_states = {
            name: detector.state.value
            for name, detector in self.detectors.items()
        }
        # Transition events are recorded during update() in execution order,
        # preserving detector-internal transitions (e.g. CANDIDATE ->
        # CONFIRMED) and the controller-driven ones (e.g. CONFIRMED ->
        # SOFT_HOLD) as separate ordered events within the same step.
        step_events = list(self._step_events)
        evidence: dict[str, Any] = {
            "pre_states": dict(pre_states),
            "post_states": dict(post_states),
            "transition_events": step_events,
            "soft_hold_window_step": self.soft_hold_window_step,
            "soft_hold_window_armed": self.soft_hold_window_armed,
            "soft_hold_window_complete": self.soft_hold_window_complete,
            "soft_hold_window_steps_configured": (
                self.config.soft_hold_window_steps
            ),
            "contact_targets_rad": {
                name: self.contact_targets.get(name)
                for name in FINGERS
            },
            "probe_index": self.probe_index,
            "probe_step": self.probe_step,
            "probe_settle_remaining": self.probe_initial_settle_steps,
            "probe_baseline_loads_nm": (
                None
                if self.probe_baseline_loads is None
                else [float(value) for value in self.probe_baseline_loads]
            ),
            "probe_response_nm": [
                float(value) for value in self.probe_response_nm
            ],
            "probe_mode": self.config.probe_mode,
            "probe_aggregate_response_nm": (
                self.probe_aggregate_response_nm
            ),
            "probe_aggregate_minimum_response_nm": (
                3.0 * self.config.minimum_probe_response_nm
                if self.config.probe_mode == "collective"
                else None
            ),
            "balance_delta_rad": [
                float(value) for value in self.balance_delta_rad
            ],
            "balance_total_rad": [
                float(value) for value in self.balance_total
            ],
            "balance_budget_remaining_rad": [
                float(
                    self.config.maximum_balance_total_rad
                    - abs(self.balance_total[index])
                )
                for index in range(3)
            ],
            "controller_phase": self.controller_phase,
            "lift_ready": self.lift_ready,
            "consolidation_armed": self.consolidation_armed,
            "consolidation_complete": self.consolidation_complete,
            "consolidation_ramp_step": self.consolidation_ramp_step,
            "consolidation_window_step": self.consolidation_window_step,
            "consolidation_stiffness_scale": float(
                self.consolidation_stiffness_scale
            ),
            "consolidation_ramp_steps_configured": (
                self.config.consolidation_ramp_steps
            ),
            "consolidation_window_steps_configured": (
                self.config.consolidation_window_steps
            ),
            "consolidation_final_stiffness_scale_configured": float(
                self.config.consolidation_final_stiffness_scale
            ),
            "soft_hold_stiffness_scale_configured": float(
                self.config.soft_hold_stiffness_scale
            ),
            "consolidation_threshold_label": (
                self.config.consolidation_threshold_label
            ),
            "consolidation_scale_monotonic": (
                self._consolidation_scale_monotonic
            ),
            "consolidation_scale_min": self._consolidation_scale_min,
            "consolidation_scale_max": self._consolidation_scale_max,
            "frozen_targets_rad": (
                None
                if self._frozen_targets is None
                else [float(value) for value in self._frozen_targets]
            ),
            "targets_match_frozen": (
                self._frozen_targets is not None
                and tuple(float(value) for value in self.targets)
                == self._frozen_targets
            ),
        }
        return SequentialGraspCommand(
            finger_targets_rad=tuple(self.targets),
            finger_stiffness_scale=stiffness,
            observations=observations,
            contact_order=tuple(self.contact_order),
            stable=self.stable,
            failed=self.failed,
            failure_reason=self.failure_reason,
            normalized_loads=values,
            normalized_load_imbalance=imbalance,
            probe_response_nm=tuple(self.probe_response_nm),
            lift_ready=self.lift_ready,
            controller_phase=self.controller_phase,
            evidence=evidence,
        )
