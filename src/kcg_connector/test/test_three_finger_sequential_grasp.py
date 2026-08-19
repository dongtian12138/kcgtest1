from kcg_connector.grasp.finger_contact_detector import (
    FingerContactDetectorConfig,
    FingerContactState,
)
from kcg_connector.grasp.three_finger_sequential_grasp import (
    ThreeFingerSequentialGrasp,
    SequentialGraspConfig,
)
import dataclasses


def _config() -> SequentialGraspConfig:
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
        soft_hold_window_steps=3,
    )


def test_first_contact_soft_holds_while_other_fingers_continue_and_all_stabilize():
    controller = ThreeFingerSequentialGrasp(
        _config(),
        open_targets_rad=(0.0, 0.0, 0.0),
        closed_targets_rad=(1.0, 1.0, 1.0),
    )
    controller.calibrate({"f1": [0.0] * 16, "f2": [0.0] * 16, "f3": [0.0] * 16})
    contacts = (0.10, 0.20, 0.30)
    first_hold_target = None
    command = None
    for step in range(1, 90):
        targets = tuple(controller.targets)
        positions = tuple(min(target, contact) for target, contact in zip(targets, contacts))
        torques = tuple(20.0 * max(0.0, target - contact) for target, contact in zip(targets, contacts))
        velocities = tuple(0.0 if target >= contact else 1.0 for target, contact in zip(targets, contacts))
        command = controller.update(
            torques,
            positions,
            velocities,
            timestamp_s=step * 0.01,
        )
        if command.contact_order and first_hold_target is None:
            first_hold_target = command.finger_targets_rad[0]
        if first_hold_target is not None and len(command.contact_order) < 3:
            assert command.finger_targets_rad[0] == first_hold_target
            assert command.finger_targets_rad[2] >= command.finger_targets_rad[0]
        if command.stable or command.failed:
            break

    assert command is not None
    assert not command.failed, command.failure_reason
    assert command.stable
    assert command.contact_order == ("f1", "f2", "f3")
    assert len(command.probe_response_nm) == 3
    assert all(
        detector.state == FingerContactState.STABLE_CONTACT
        for detector in controller.detectors.values()
    )


def test_load_balance_relieves_overloaded_finger_with_bounded_increment():
    config = _config()
    controller = ThreeFingerSequentialGrasp(
        config,
        open_targets_rad=(0.0, 0.0, 0.0),
        closed_targets_rad=(1.0, 1.0, 1.0),
    )
    controller.calibrate({"f1": [0.0] * 16, "f2": [0.0] * 16, "f3": [0.0] * 16})
    controller.targets[:] = [0.5, 0.5, 0.5]
    controller.probe_index = 3
    controller.soft_hold_window_armed = True
    controller.soft_hold_window_complete = True
    for detector in controller.detectors.values():
        detector.state = FingerContactState.LOAD_BUILD
    before = tuple(controller.targets)
    command = controller.update(
        (0.40, 0.20, 0.20),
        (0.49, 0.49, 0.49),
        (0.0, 0.0, 0.0),
        timestamp_s=0.01,
    )
    increments = tuple(after - old for after, old in zip(command.finger_targets_rad, before))
    assert increments[0] < increments[1]
    assert increments[0] < increments[2]
    assert max(abs(value) for value in controller.balance_total) <= config.maximum_balance_total_rad


def test_collective_probe_moves_all_fingers_and_preserves_total_gate():
    config = dataclasses.replace(
        _config(), probe_mode="collective", probe_settle_steps=2
    )
    controller = ThreeFingerSequentialGrasp(
        config,
        open_targets_rad=(0.0, 0.0, 0.0),
        closed_targets_rad=(1.0, 1.0, 1.0),
    )
    controller.calibrate(
        {"f1": [0.0] * 16, "f2": [0.0] * 16, "f3": [0.0] * 16}
    )
    controller.targets[:] = [0.5, 0.5, 0.5]
    controller.probe_initial_settle_steps = 0
    controller.soft_hold_window_armed = True
    controller.soft_hold_window_complete = True
    for detector in controller.detectors.values():
        detector.state = FingerContactState.LOAD_BUILD

    baseline = controller.update(
        (0.20, 0.20, 0.20),
        (0.49, 0.49, 0.49),
        (0.0, 0.0, 0.0),
        timestamp_s=0.01,
    )
    assert baseline.finger_targets_rad == (0.504, 0.504, 0.504)
    assert baseline.evidence["probe_mode"] == "collective"
    controller.update(
        (0.30, 0.30, 0.30),
        (0.49, 0.49, 0.49),
        (0.0, 0.0, 0.0),
        timestamp_s=0.02,
    )
    result = controller.update(
        (0.30, 0.30, 0.30),
        (0.49, 0.49, 0.49),
        (0.0, 0.0, 0.0),
        timestamp_s=0.03,
    )
    assert not result.failed
    assert len(result.probe_response_nm) == 3
    assert result.evidence["probe_aggregate_response_nm"] >= 0.06
    assert result.evidence["probe_aggregate_minimum_response_nm"] == 0.06


def test_missing_contact_exhausts_bounded_approach_and_fails_closed():
    controller = ThreeFingerSequentialGrasp(
        _config(),
        open_targets_rad=(0.0, 0.0, 0.0),
        closed_targets_rad=(1.0, 1.0, 1.0),
    )
    controller.calibrate({"f1": [0.0] * 16, "f2": [0.0] * 16, "f3": [0.0] * 16})
    command = None
    for step in range(1, 101):
        command = controller.update(
            (0.0, 0.0, 0.0),
            tuple(controller.targets),
            (1.0, 1.0, 1.0),
            timestamp_s=step * 0.01,
        )
        if command.failed:
            break
    assert command is not None and command.failed
    assert command.failure_reason == "f1_closed_limit_without_contact"


def test_step_budgets_reject_booleans_and_non_integers():
    # The approach/load-build budgets are physics-step counters; a boolean
    # or a non-integer must never be accepted as a budget (bool-as-int
    # would silently collapse the bound to 1 step).
    import dataclasses

    import pytest

    for name in ("maximum_approach_steps", "maximum_load_build_steps"):
        for bad in (True, False, 12.5):
            with pytest.raises(ValueError):
                dataclasses.replace(_config(), **{name: bad})
