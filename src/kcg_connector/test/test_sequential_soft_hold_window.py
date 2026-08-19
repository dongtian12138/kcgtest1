'''Pure tests for the sequential SOFT_HOLD window semantics.'''

from __future__ import annotations

from kcg_connector.grasp.finger_contact_detector import (
    FingerContactDetectorConfig,
)
from kcg_connector.grasp.three_finger_sequential_grasp import (
    SequentialGraspConfig,
    ThreeFingerSequentialGrasp,
)


def _config(soft_hold_window_steps=3):
    return SequentialGraspConfig(
        detector=FingerContactDetectorConfig(
            sample_period_s=0.01,
            lowpass_alpha=0.9,
            derivative_alpha=0.9,
            contact_sigma_multiplier=6.0,
            minimum_contact_delta_nm=0.05,
            release_ratio=0.45,
            minimum_release_delta_nm=0.01,
            minimum_rise_rate_nm_s=0.01,
            maximum_stall_velocity_rad_s=0.02,
            minimum_tracking_error_rad=0.002,
            confirm_steps=3,
            release_confirm_steps=3,
            maximum_sample_gap_s=0.02,
            position_velocity_window_steps=6,
        ),
        sample_period_s=0.01,
        approach_rate_rad_s=1.0,
        soft_hold_preload_rad=0.01,
        load_build_rate_rad_s=0.1,
        balance_gain_rad_per_load=0.005,
        maximum_balance_step_rad=0.002,
        maximum_balance_total_rad=0.02,
        probe_increment_rad=0.004,
        probe_settle_steps=3,
        minimum_probe_response_nm=0.02,
        maximum_probe_cross_coupling_ratio=0.5,
        load_scale_nm=(0.4, 0.4, 0.4),
        stable_minimum_normalized_load=0.4,
        maximum_normalized_load_imbalance=0.5,
        stable_confirm_steps=3,
        maximum_approach_steps=100,
        maximum_load_build_steps=30,
        soft_hold_window_steps=soft_hold_window_steps,
    )


def _controller(soft_hold_window_steps=3):
    controller = ThreeFingerSequentialGrasp(
        _config(soft_hold_window_steps),
        open_targets_rad=(0.0, 0.0, 0.0),
        closed_targets_rad=(1.0, 1.0, 1.0),
    )
    controller.calibrate(
        {"f1": [0.0] * 16, "f2": [0.0] * 16, "f3": [0.0] * 16}
    )
    return controller


CONTACTS = (0.10, 0.20, 0.30)


def _update(controller, torques=None, step=1):
    targets = tuple(controller.targets)
    positions = tuple(
        min(target, contact)
        for target, contact in zip(targets, CONTACTS)
    )
    if torques is None:
        torques = tuple(
            20.0 * max(0.0, target - contact)
            for target, contact in zip(targets, CONTACTS)
        )
    velocities = tuple(
        0.0 if target >= contact else 1.0
        for target, contact in zip(targets, CONTACTS)
    )
    return controller.update(
        torques, positions, velocities, timestamp_s=step * 0.01
    )


def _drive(controller, steps, torques_fn=None):
    command = None
    for step in range(1, steps + 1):
        torques = torques_fn() if torques_fn is not None else None
        command = _update(controller, torques, step)
        if command.failed or command.stable:
            break
    return command


def test_last_confirmed_finger_soft_hold_spans_the_full_window():
    controller = _controller(soft_hold_window_steps=5)
    confirmation_step = None
    soft_hold_updates = 0
    load_build_step = None
    command = None
    for step in range(1, 200):
        command = _update(controller, step=step)
        post = command.evidence["post_states"]
        if (
            confirmation_step is None
            and all(
                value in ("SOFT_HOLD", "LOAD_BUILD", "STABLE_CONTACT")
                for value in post.values()
            )
        ):
            confirmation_step = step
        if confirmation_step is not None:
            if all(value == "SOFT_HOLD" for value in post.values()):
                soft_hold_updates += 1
            elif "LOAD_BUILD" in post.values():
                load_build_step = step
                break
        if command.failed:
            break
    assert command is not None
    assert not command.failed, command.failure_reason
    assert confirmation_step is not None
    assert load_build_step is not None
    # SOFT_HOLD post-states span the confirmation update plus the four
    # counted updates; LOAD_BUILD is marked at the end of the fifth counted
    # update (window_steps=5).
    assert soft_hold_updates == 5
    assert load_build_step - confirmation_step == soft_hold_updates
    assert command.evidence["soft_hold_window_complete"] is True
    assert command.evidence["soft_hold_window_step"] == 5


def test_confirmed_finger_target_is_latched_and_stiffness_low_immediately():
    controller = _controller()
    latched = None
    command = None
    for step in range(1, 40):
        command = _update(controller, step=step)
        if command.contact_order and latched is None:
            latched = command.finger_targets_rad[0]
        if latched is not None:
            assert command.finger_targets_rad[0] == latched
            assert command.finger_stiffness_scale[0] == 0.35
            if len(command.contact_order) == 3:
                break
    assert command is not None
    assert command.contact_order == ("f1", "f2", "f3")
    assert latched is not None


def test_probe_settle_is_independent_of_the_soft_hold_window():
    controller = _controller(soft_hold_window_steps=5)
    settle_remaining = []
    window_complete = False
    for step in range(1, 120):
        command = _update(controller, step=step)
        settle_remaining.append(
            command.evidence["probe_settle_remaining"]
        )
        if command.evidence["soft_hold_window_complete"]:
            window_complete = True
        if command.stable or command.failed:
            break
    assert window_complete
    assert settle_remaining[-1] == 0
    assert all(
        after <= before
        for before, after in zip(settle_remaining, settle_remaining[1:])
    )


def test_load_loss_during_soft_hold_window_fails_closed():
    controller = _controller(soft_hold_window_steps=5)
    contacted = False
    command = None

    def torques():
        nonlocal contacted
        states = [
            controller.detectors[name].state.value
            for name in ("f1", "f2", "f3")
        ]
        if all(
            state in ("SOFT_HOLD", "LOAD_BUILD", "STABLE_CONTACT")
            for state in states
        ):
            contacted = True
            return (0.0, 0.0, 0.0)
        return None

    command = _drive(controller, 120, torques_fn=torques)
    assert contacted
    assert command is not None
    assert command.failed
    assert "slip_suspected" in command.failure_reason


def test_transition_event_steps_are_monotonic_and_ordered():
    controller = _controller()
    _drive(controller, 200)
    steps = [event["step"] for event in controller.transition_events]
    assert steps == sorted(steps)
    assert all(
        event["from"] != event["to"]
        for event in controller.transition_events
    )
    states = [event["to"] for event in controller.transition_events]
    assert "SOFT_HOLD" in states
    assert "LOAD_BUILD" in states


def test_intra_step_transitions_preserve_candidate_confirmed_soft_hold():
    controller = _controller()
    _drive(controller, 200)
    events = controller.transition_events
    # Every finger must expose its full ordered chain, including the
    # detector-internal CANDIDATE -> CONFIRMED and the controller-driven
    # CONFIRMED -> SOFT_HOLD on the same step.
    for finger in ("f1", "f2", "f3"):
        pairs = [
            (event["from"], event["to"])
            for event in events
            if event["finger"] == finger
        ]
        for expected in (
            ("APPROACH", "CONTACT_CANDIDATE"),
            ("CONTACT_CANDIDATE", "CONTACT_CONFIRMED"),
            ("CONTACT_CONFIRMED", "SOFT_HOLD"),
            ("SOFT_HOLD", "LOAD_BUILD"),
            ("LOAD_BUILD", "STABLE_CONTACT"),
        ):
            assert expected in pairs, f"{finger} missing {expected}"
        # The confirmation step shows the two intra-step transitions in the
        # recorded order.
        confirmed_step = next(
            event["step"]
            for event in events
            if event["finger"] == finger
            and (event["from"], event["to"])
            == ("CONTACT_CANDIDATE", "CONTACT_CONFIRMED")
        )
        same_step = [
            (event["from"], event["to"])
            for event in events
            if event["finger"] == finger
            and event["step"] == confirmed_step
        ]
        assert same_step == [
            ("CONTACT_CANDIDATE", "CONTACT_CONFIRMED"),
            ("CONTACT_CONFIRMED", "SOFT_HOLD"),
        ]


def test_window_step_counter_is_configured_and_reported():
    controller = _controller(soft_hold_window_steps=4)
    command = _drive(controller, 200)
    assert command is not None and (command.stable or command.failed)
    assert controller.config.soft_hold_window_steps == 4
    for event in controller.transition_events:
        assert event["finger"] in ("f1", "f2", "f3")
        assert isinstance(event["step"], int)
