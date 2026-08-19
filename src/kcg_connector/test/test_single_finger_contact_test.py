'''Pure tests for the single-finger contact/release state machine.'''

from __future__ import annotations

import pytest

from kcg_connector.grasp.finger_contact_detector import (
    FingerContactDetectorConfig,
    FingerContactState,
)
from kcg_connector.grasp.single_finger_contact_test import (
    MAXIMUM_RELEASE_STEPS_CEILING,
    SingleFingerContactConfig,
    SingleFingerContactTest,
)

CONTACT_POSITION = 0.05
OPEN_TARGET = 0.0
CLOSED_TARGET = 0.5
GAIN = 50.0


def _detector_config():
    return FingerContactDetectorConfig(
        sample_period_s=0.01,
        lowpass_alpha=0.9,
        derivative_alpha=0.9,
        contact_sigma_multiplier=6.0,
        minimum_contact_delta_nm=0.05,
        release_ratio=0.45,
        minimum_release_delta_nm=0.02,
        minimum_rise_rate_nm_s=0.01,
        maximum_stall_velocity_rad_s=0.02,
        minimum_tracking_error_rad=0.005,
        confirm_steps=3,
        release_confirm_steps=3,
        maximum_sample_gap_s=0.02,
        position_velocity_window_steps=6,
    )


def _config(**overrides):
    values = {
        "threshold_label": "SIM_TUNING_ONLY",
        "soft_hold_steps": 24,
        "minimum_release_travel_rad": 0.05,
        "maximum_release_tracking_error_rad": 0.05,
        "maximum_release_steps": 100,
        "maximum_approach_steps": 60,
        "approach_rate_rad_s": 0.18,
        "release_rate_rad_s": 0.18,
    }
    values.update(overrides)
    return SingleFingerContactConfig(**values)


def _controller(finger="f1", config=None):
    return SingleFingerContactTest(
        config or _config(),
        _detector_config(),
        finger=finger,
        open_target_rad=OPEN_TARGET,
        closed_target_rad=CLOSED_TARGET,
    )


def _step(controller, torque, position, velocity):
    # Timestamps continue contiguously from the controller step counter so
    # the detector never sees an artificial stale gap.
    step = controller.step + 1
    return controller.update(
        torque, position, velocity, timestamp_s=step * 0.01
    )


def _drive_approach(controller, torque_sign):
    command = None
    for _ in range(80):
        command = _step(
            controller,
            torque_sign * GAIN * max(0.0, controller.target - CONTACT_POSITION),
            min(controller.target, CONTACT_POSITION),
            1.0 if controller.target < CONTACT_POSITION else 0.0,
        )
        if command.state in (
            FingerContactState.SOFT_HOLD,
            FingerContactState.RELEASE_COMMANDED,
            FingerContactState.RELEASE_CONFIRMED,
        ) or command.failed:
            break
    return command


@pytest.mark.parametrize("sign", (1.0, -1.0))
def test_positive_and_negative_torque_contact_confirm(sign):
    controller = _controller()
    controller.calibrate([0.0] * 16)
    command = _drive_approach(controller, sign)
    assert command.state == FingerContactState.SOFT_HOLD
    assert not command.failed
    # The confirmation update itself: target lock plus immediate low
    # stiffness, counted separately from the subsequent hold outputs.
    assert command.stiffness_scale == 0.35
    latched = controller.target
    subsequent_soft_hold_outputs = 0
    for _ in range(80):
        command = _step(
            controller,
            sign * GAIN * (controller.target - CONTACT_POSITION),
            min(controller.target, CONTACT_POSITION),
            0.0,
        )
        if command.state == FingerContactState.SOFT_HOLD:
            assert controller.target == latched
            subsequent_soft_hold_outputs += 1
            assert command.stiffness_scale == 0.35
        if command.state == FingerContactState.RELEASE_COMMANDED:
            break
    # Exactly 24 subsequent updates must output SOFT_HOLD at low stiffness;
    # the next (25th subsequent) update enters RELEASE_COMMANDED at full
    # stiffness and starts the open-loop release motion.
    assert subsequent_soft_hold_outputs == 24
    assert controller.soft_hold_step == 24
    assert command.state == FingerContactState.RELEASE_COMMANDED
    assert command.stiffness_scale == 1.0


def test_intra_step_transition_evidence_is_complete_and_ordered():
    controller = _controller()
    controller.calibrate([0.0] * 16)
    command = _drive_approach(controller, 1.0)
    assert command.state == FingerContactState.SOFT_HOLD
    for _ in range(300):
        if controller.state == FingerContactState.SOFT_HOLD:
            torque = GAIN * (controller.target - CONTACT_POSITION)
            position = min(controller.target, CONTACT_POSITION)
        else:
            torque = 0.0
            position = controller.target
        command = _step(controller, torque, position, 0.5)
        if command.failed or command.state == FingerContactState.RELEASE_CONFIRMED:
            break
    assert command.state == FingerContactState.RELEASE_CONFIRMED
    events = controller.transition_events
    pairs = [(event["from"], event["to"]) for event in events]
    for expected in (
        ("APPROACH", "CONTACT_CANDIDATE"),
        ("CONTACT_CANDIDATE", "CONTACT_CONFIRMED"),
        ("CONTACT_CONFIRMED", "SOFT_HOLD"),
        ("SOFT_HOLD", "RELEASE_COMMANDED"),
        ("RELEASE_COMMANDED", "RELEASE_CONFIRMED"),
    ):
        assert expected in pairs, f"missing transition {expected}"
    steps = [event["step"] for event in events]
    assert steps == sorted(steps)
    # The confirmation step carries both intra-step transitions in order:
    # the detector's CANDIDATE -> CONFIRMED first, then the controller's
    # CONFIRMED -> SOFT_HOLD.
    confirmation_step = next(
        event["step"]
        for event in events
        if (event["from"], event["to"])
        == ("CONTACT_CANDIDATE", "CONTACT_CONFIRMED")
    )
    same_step = [
        (event["from"], event["to"])
        for event in events
        if event["step"] == confirmation_step
    ]
    assert same_step == [
        ("CONTACT_CANDIDATE", "CONTACT_CONFIRMED"),
        ("CONTACT_CONFIRMED", "SOFT_HOLD"),
    ]
    # SOFT_HOLD -> RELEASE_COMMANDED lands exactly one step after the 24th
    # subsequent hold output: the release starts on the next update.
    release_step = next(
        event["step"]
        for event in events
        if (event["from"], event["to"])
        == ("SOFT_HOLD", "RELEASE_COMMANDED")
    )
    assert release_step - confirmation_step == 1 + 24


def test_no_contact_exhausts_approach_budget_and_fails_closed():
    controller = _controller()
    controller.calibrate([0.0] * 16)
    command = None
    for _ in range(80):
        command = _step(controller, 0.0, controller.target, 1.0)
        if command.failed:
            break
    assert command is not None and command.failed
    assert command.failure_reason == "approach_step_budget_exhausted"
    assert not controller.detector_test_passed


def test_successful_release_confirms_after_soft_hold_window():
    controller = _controller()
    controller.calibrate([0.0] * 16)
    command = _drive_approach(controller, 1.0)
    assert command.state == FingerContactState.SOFT_HOLD
    confirmed = None
    for _ in range(200):
        if controller.state == FingerContactState.SOFT_HOLD:
            torque = GAIN * (controller.target - CONTACT_POSITION)
            position = min(controller.target, CONTACT_POSITION)
            velocity = 0.0
        else:
            torque = 0.0
            position = controller.target
            velocity = 1.0
        command = _step(controller, torque, position, velocity)
        if command.failed or command.state == FingerContactState.RELEASE_CONFIRMED:
            confirmed = command
            break
    assert confirmed is not None
    assert not confirmed.failed
    assert confirmed.state == FingerContactState.RELEASE_CONFIRMED
    assert confirmed.detector_test_passed
    assert controller.detector_test_passed


def test_stuck_opening_fails_release_budget_without_confirmation():
    controller = _controller()
    controller.calibrate([0.0] * 16)
    command = _drive_approach(controller, 1.0)
    assert command.state == FingerContactState.SOFT_HOLD
    final = None
    for _ in range(300):
        if controller.state == FingerContactState.SOFT_HOLD:
            torque = GAIN * (controller.target - CONTACT_POSITION)
            position = min(controller.target, CONTACT_POSITION)
        else:
            # Stuck: position frozen at the contact pose and load stays high.
            torque = GAIN * 0.2
            position = CONTACT_POSITION
        command = _step(controller, torque, position, 0.0)
        if command.failed:
            final = command
            break
    assert final is not None
    assert final.failed
    assert final.failure_reason == "release_step_budget_exhausted"
    assert not final.detector_test_passed
    assert final.state != FingerContactState.RELEASE_CONFIRMED


def test_filter_tail_delays_but_does_not_forge_confirmation():
    controller = _controller(
        config=_config(minimum_release_travel_rad=0.01)
    )
    controller.calibrate([0.0] * 16)
    command = _drive_approach(controller, 1.0)
    assert command.state == FingerContactState.SOFT_HOLD
    release_command_seen = False
    final = None
    for _ in range(300):
        if controller.state == FingerContactState.SOFT_HOLD:
            torque = GAIN * (controller.target - CONTACT_POSITION)
            position = min(controller.target, CONTACT_POSITION)
        else:
            release_command_seen = True
            torque = 0.0
            position = controller.target
        command = _step(controller, torque, position, 0.5)
        if controller.state in (
            FingerContactState.RELEASE_COMMANDED,
            FingerContactState.RELEASE_CONFIRMED,
        ):
            # Slip stays disarmed during the commanded opening, even while
            # the low-pass tail still shows load.
            assert command.state != FingerContactState.SLIP_SUSPECTED
        if command.failed or command.state == FingerContactState.RELEASE_CONFIRMED:
            final = command
            break
    assert release_command_seen
    assert final is not None
    assert final.state == FingerContactState.RELEASE_CONFIRMED
    assert final.detector_test_passed


def test_active_opening_spike_resets_confirm_and_never_arms_slip():
    controller = _controller(
        config=_config(minimum_release_travel_rad=0.01)
    )
    controller.calibrate([0.0] * 16)
    command = _drive_approach(controller, 1.0)
    assert command.state == FingerContactState.SOFT_HOLD
    spike_done = False
    confirm_before_spike = None
    final = None
    for _ in range(300):
        if controller.state == FingerContactState.SOFT_HOLD:
            torque = GAIN * (controller.target - CONTACT_POSITION)
            position = min(controller.target, CONTACT_POSITION)
        elif (
            not spike_done
            and controller.state == FingerContactState.RELEASE_COMMANDED
        ):
            # A single active-opening transient load spike.
            spike_done = True
            torque = 0.5
            position = controller.target
        else:
            torque = 0.0
            position = controller.target
        command = _step(controller, torque, position, 0.5)
        if (
            confirm_before_spike is None
            and controller.state == FingerContactState.RELEASE_COMMANDED
        ):
            confirm_before_spike = command.evidence["release_confirm"]
        if command.failed or command.state == FingerContactState.RELEASE_CONFIRMED:
            final = command
            break
    assert spike_done
    assert confirm_before_spike is not None
    assert confirm_before_spike == 0
    assert "SLIP_SUSPECTED" not in [
        event["to"] for event in controller.transition_events
    ]
    assert final is not None
    assert final.state == FingerContactState.RELEASE_CONFIRMED


def test_release_requires_all_three_conditions_simultaneously():
    controller = _controller(
        config=_config(minimum_release_travel_rad=0.05)
    )
    controller.calibrate([0.0] * 16)
    command = _drive_approach(controller, 1.0)
    assert command.state == FingerContactState.SOFT_HOLD
    for _ in range(300):
        if controller.state == FingerContactState.SOFT_HOLD:
            torque = GAIN * (controller.target - CONTACT_POSITION)
            position = min(controller.target, CONTACT_POSITION)
        else:
            torque = 0.0
            position = controller.target
        command = _step(controller, torque, position, 0.0)
        if controller.state == FingerContactState.RELEASE_COMMANDED:
            conditions = command.release_conditions
            if not conditions:
                # The transition update itself carries no release
                # conditions yet; release motion starts on the next update.
                continue
            travel = conditions["travel_rad"]
            if travel < controller.config.minimum_release_travel_rad:
                assert conditions["travel_ok"] is False
                assert command.evidence["release_confirm"] == 0
        if command.failed or command.state == FingerContactState.RELEASE_CONFIRMED:
            break
    assert command.state == FingerContactState.RELEASE_CONFIRMED
    assert command.detector_test_passed


def test_unconfirmed_cannot_fake_release_pass():
    controller = _controller()
    controller.calibrate([0.0] * 16)
    assert not controller.detector_test_passed
    # Only a completed SOFT_HOLD window may begin the commanded release.
    with pytest.raises(RuntimeError, match="SOFT_HOLD"):
        controller.detector.begin_commanded_release()


def test_stale_and_nonfinite_inputs_fail_closed():
    controller = _controller()
    controller.calibrate([0.0] * 16)
    controller.update(0.0, 0.0, 0.0, timestamp_s=0.01)
    command = controller.update(0.0, 0.0, 0.0, timestamp_s=0.99)
    assert command.failed
    assert "stale" in command.failure_reason
    assert not command.detector_test_passed
    assert command.state == FingerContactState.FAILED

    controller = _controller()
    controller.calibrate([0.0] * 16)
    command = controller.update(
        float("nan"), 0.0, 0.0, timestamp_s=0.01
    )
    assert command.failed
    assert "nonfinite_or_stale_input" in command.failure_reason
    assert command.state == FingerContactState.FAILED


def test_config_validation():
    with pytest.raises(ValueError, match="SIM_TUNING_ONLY"):
        _config(threshold_label="HARDWARE_CALIBRATED")
    with pytest.raises(ValueError, match="soft_hold_steps"):
        _config(soft_hold_steps=1)
    with pytest.raises(ValueError, match="maximum_release_steps"):
        _config(maximum_release_steps=0)
    with pytest.raises(ValueError, match="maximum_release_steps"):
        _config(maximum_release_steps=MAXIMUM_RELEASE_STEPS_CEILING + 1)
    with pytest.raises(ValueError, match="minimum_release_travel_rad"):
        _config(minimum_release_travel_rad=0.0)
    with pytest.raises(ValueError, match="tracking"):
        _config(maximum_release_tracking_error_rad=float("inf"))


def test_release_command_is_open_loop_bounded():
    controller = _controller(config=_config(release_rate_rad_s=0.18))
    controller.calibrate([0.0] * 16)
    command = _drive_approach(controller, 1.0)
    assert command.state == FingerContactState.SOFT_HOLD
    previous = None
    for _ in range(300):
        if controller.state == FingerContactState.SOFT_HOLD:
            torque = GAIN * (controller.target - CONTACT_POSITION)
            position = min(controller.target, CONTACT_POSITION)
        else:
            torque = 0.0
            position = controller.target
        command = _step(controller, torque, position, 0.5)
        if controller.state == FingerContactState.RELEASE_COMMANDED:
            if previous is not None:
                # Fixed-rate open-loop step, bounded by the open target:
                # either the full rate step or clamped at the open command.
                step_size = abs(command.target_rad - previous)
                assert step_size == pytest.approx(
                    0.18 * 0.01, abs=1.0e-9
                ) or command.target_rad == OPEN_TARGET
            previous = command.target_rad
        if command.failed or command.state == FingerContactState.RELEASE_CONFIRMED:
            break
    assert command.state == FingerContactState.RELEASE_CONFIRMED
    assert 0.0 <= command.target_rad <= CLOSED_TARGET


from kcg_connector.grasp.single_finger_contact_test import (
    ReleaseBudgetEvidence,
    release_budget_feasibility,
)


def test_release_budget_formal_config_is_exactly_1133_of_1200():
    from kcg_connector.d38999_tabletop_pick import (
        load_d38999_tabletop_pick_config,
    )
    from kcg_connector.grasp.physical_grasp_config import (
        load_physical_grasp_experiment_config,
    )

    pick = load_d38999_tabletop_pick_config(
        "src/kcg_connector/config/d38999_tabletop_pick_v1.yaml"
    )
    physical = load_physical_grasp_experiment_config(
        "src/kcg_connector/config/d38999_tabletop_physical_grasp_v1.yaml"
    )
    detector = physical.sequential.detector
    evidence = release_budget_feasibility(
        closed_targets_rad=[
            float(pick.motion.grasp_hand_rad[index])
            for index in (1, 2, 3)
        ],
        open_targets_rad=[
            float(pick.robot.open_hand_rad[index])
            for index in (1, 2, 3)
        ],
        release_rate_rad_s=physical.single_finger.release_rate_rad_s,
        sample_period_s=detector.sample_period_s,
        lowpass_alpha=detector.lowpass_alpha,
        minimum_release_delta_nm=detector.minimum_release_delta_nm,
        release_ratio=detector.release_ratio,
        minimum_contact_delta_nm=detector.minimum_contact_delta_nm,
        maximum_release_tracking_error_rad=(
            physical.single_finger.maximum_release_tracking_error_rad
        ),
        release_confirm_steps=detector.release_confirm_steps,
        maximum_torque_delta_gate_nm=(
            pick.sensing.maximum_absolute_torque_delta_nm
        ),
        configured_steps=physical.single_finger.maximum_release_steps,
    )
    assert evidence.travel_steps == 1020
    assert evidence.filter_tail_steps == 28
    assert evidence.tracking_lag_steps == 67
    assert evidence.confirm_steps == 18
    assert evidence.required_steps == 1133
    assert evidence.configured_steps == 1200
    assert evidence.headroom_steps == 67
    assert evidence.feasible is True
    assert evidence.maximum_torque_delta_gate_nm == pytest.approx(2.0)
    assert evidence.step_rad == pytest.approx(0.00075)
    assert evidence.maximum_span_rad == pytest.approx(0.765)
    assert evidence.minimum_possible_release_threshold_nm == pytest.approx(
        0.009
    )
    assert isinstance(evidence.as_dict(), dict)


def test_release_budget_1132_is_infeasible_and_1133_feasible():
    from kcg_connector.d38999_tabletop_pick import (
        load_d38999_tabletop_pick_config,
    )
    from kcg_connector.grasp.physical_grasp_config import (
        load_physical_grasp_experiment_config,
    )

    pick = load_d38999_tabletop_pick_config(
        "src/kcg_connector/config/d38999_tabletop_pick_v1.yaml"
    )
    physical = load_physical_grasp_experiment_config(
        "src/kcg_connector/config/d38999_tabletop_physical_grasp_v1.yaml"
    )
    detector = physical.sequential.detector
    evidence = release_budget_feasibility(
        closed_targets_rad=[0.765, 0.595, 0.765],
        open_targets_rad=[0.0, 0.0, 0.0],
        release_rate_rad_s=physical.single_finger.release_rate_rad_s,
        sample_period_s=detector.sample_period_s,
        lowpass_alpha=detector.lowpass_alpha,
        minimum_release_delta_nm=detector.minimum_release_delta_nm,
        release_ratio=detector.release_ratio,
        minimum_contact_delta_nm=detector.minimum_contact_delta_nm,
        maximum_release_tracking_error_rad=(
            physical.single_finger.maximum_release_tracking_error_rad
        ),
        release_confirm_steps=detector.release_confirm_steps,
        maximum_torque_delta_gate_nm=2.0,
        configured_steps=1132,
    )
    assert evidence.required_steps == 1133
    assert evidence.feasible is False
    assert evidence.headroom_steps == -1
    feasible_evidence = release_budget_feasibility(
        closed_targets_rad=[0.765, 0.595, 0.765],
        open_targets_rad=[0.0, 0.0, 0.0],
        release_rate_rad_s=physical.single_finger.release_rate_rad_s,
        sample_period_s=detector.sample_period_s,
        lowpass_alpha=detector.lowpass_alpha,
        minimum_release_delta_nm=detector.minimum_release_delta_nm,
        release_ratio=detector.release_ratio,
        minimum_contact_delta_nm=detector.minimum_contact_delta_nm,
        maximum_release_tracking_error_rad=(
            physical.single_finger.maximum_release_tracking_error_rad
        ),
        release_confirm_steps=detector.release_confirm_steps,
        maximum_torque_delta_gate_nm=2.0,
        configured_steps=1133,
    )
    assert feasible_evidence.feasible is True
    assert feasible_evidence.headroom_steps == 0


def test_release_budget_grows_with_span_and_gate_is_parameterized():
    base = dict(
        closed_targets_rad=[0.765],
        open_targets_rad=[0.0],
        release_rate_rad_s=0.18,
        sample_period_s=1.0 / 240.0,
        lowpass_alpha=0.18,
        minimum_release_delta_nm=0.008,
        release_ratio=0.45,
        minimum_contact_delta_nm=0.020,
        maximum_release_tracking_error_rad=0.05,
        release_confirm_steps=18,
        maximum_torque_delta_gate_nm=2.0,
        configured_steps=1200,
    )
    evidence = release_budget_feasibility(**base)
    wider = release_budget_feasibility(
        **{**base, "closed_targets_rad": [0.9]}
    )
    assert wider.required_steps > evidence.required_steps
    assert wider.travel_steps > evidence.travel_steps
    # The gate is a parameter, never hardcoded: a larger gate lengthens
    # the filter tail (the formal pick sensing gate is 2.0 N*m).
    larger_gate = release_budget_feasibility(
        **{**base, "maximum_torque_delta_gate_nm": 8.0}
    )
    assert larger_gate.filter_tail_steps > evidence.filter_tail_steps


def test_release_budget_rejects_malformed_inputs():
    base = dict(
        closed_targets_rad=[0.765],
        open_targets_rad=[0.0],
        release_rate_rad_s=0.18,
        sample_period_s=1.0 / 240.0,
        lowpass_alpha=0.18,
        minimum_release_delta_nm=0.008,
        release_ratio=0.45,
        minimum_contact_delta_nm=0.020,
        maximum_release_tracking_error_rad=0.05,
        release_confirm_steps=18,
        maximum_torque_delta_gate_nm=8.0,
        configured_steps=1200,
    )
    with pytest.raises(ValueError, match="length"):
        release_budget_feasibility(
            **{**base, "open_targets_rad": []}
        )
    with pytest.raises(ValueError, match="finite"):
        release_budget_feasibility(
            **{**base, "closed_targets_rad": [float("nan")]}
        )
    with pytest.raises(ValueError, match="lowpass_alpha"):
        release_budget_feasibility(
            **{**base, "lowpass_alpha": 0.0}
        )
    with pytest.raises(ValueError, match="release_ratio"):
        release_budget_feasibility(
            **{**base, "release_ratio": 1.2}
        )
    with pytest.raises(ValueError, match="release_confirm_steps"):
        release_budget_feasibility(
            **{**base, "release_confirm_steps": 0}
        )
    with pytest.raises(ValueError, match="configured_steps"):
        release_budget_feasibility(
            **{**base, "configured_steps": True}
        )
    with pytest.raises(ValueError, match="threshold"):
        release_budget_feasibility(
            **{
                **base,
                "minimum_release_delta_nm": 0.009,
                "maximum_torque_delta_gate_nm": 0.009,
            }
        )
    with pytest.raises(ValueError, match="release_rate"):
        release_budget_feasibility(
            **{**base, "release_rate_rad_s": 0.0}
        )
