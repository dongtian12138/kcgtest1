'''Controller contract tests for the LIFT_READY consolidation (030).

After STABLE_CONTACT the three finger targets freeze bit-exact; an
explicit one-shot begin_consolidation() arms a bounded stiffness ramp
soft -> final over consolidation_ramp_steps updates and a fixed-scale
quiet window of consolidation_window_steps updates.  Only a complete,
failure-free window flips lift_ready.  All thresholds are falsifiable
SIM_TUNING_ONLY_A_CANDIDATE candidates.
'''

from __future__ import annotations

import pytest

from kcg_connector.grasp.finger_contact_detector import (
    FingerContactDetectorConfig,
)
from kcg_connector.grasp.three_finger_sequential_grasp import (
    CONTROLLER_PHASE_CONSOLIDATION_RAMP,
    CONTROLLER_PHASE_CONSOLIDATION_WINDOW,
    CONTROLLER_PHASE_LIFT_READY,
    CONTROLLER_PHASE_STABLE_HOLD,
    CONSOLIDATION_THRESHOLD_LABEL,
    DEFAULT_CONSOLIDATION_FINAL_STIFFNESS_SCALE,
    DEFAULT_CONSOLIDATION_RAMP_STEPS,
    DEFAULT_CONSOLIDATION_WINDOW_STEPS,
    DEFAULT_SOFT_HOLD_STIFFNESS_SCALE,
    SequentialGraspConfig,
    ThreeFingerSequentialGrasp,
)


def _config(**overrides) -> SequentialGraspConfig:
    values = dict(
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
    values.update(overrides)
    return SequentialGraspConfig(**values)


def _controller(**config_overrides) -> ThreeFingerSequentialGrasp:
    controller = ThreeFingerSequentialGrasp(
        _config(**config_overrides),
        open_targets_rad=(0.0, 0.0, 0.0),
        closed_targets_rad=(1.0, 1.0, 1.0),
    )
    controller.calibrate(
        {"f1": [0.0] * 16, "f2": [0.0] * 16, "f3": [0.0] * 16}
    )
    return controller


def _drive_to_stable(controller, contact=(0.10, 0.20, 0.30)):
    command = None
    last_step = 0
    for step in range(1, 90):
        targets = tuple(controller.targets)
        positions = tuple(
            min(target, c) for target, c in zip(targets, contact)
        )
        torques = tuple(
            20.0 * max(0.0, target - c)
            for target, c in zip(targets, contact)
        )
        velocities = tuple(
            0.0 if target >= c else 1.0
            for target, c in zip(targets, contact)
        )
        command = controller.update(
            torques,
            positions,
            velocities,
            timestamp_s=step * 0.01,
        )
        last_step = step
        if command.stable:
            break
    assert command is not None and command.stable and not command.failed
    return command, last_step


def _step(controller, step_index, contact=(0.10, 0.20, 0.30)):
    targets = tuple(controller.targets)
    positions = tuple(
        min(target, c) for target, c in zip(targets, contact)
    )
    torques = tuple(
        20.0 * max(0.0, target - c)
        for target, c in zip(targets, contact)
    )
    velocities = tuple(
        0.0 if target >= c else 1.0
        for target, c in zip(targets, contact)
    )
    return controller.update(
        torques,
        positions,
        velocities,
        timestamp_s=step_index * 0.01,
    )


def test_stable_hold_freezes_targets_and_keeps_detectors_fresh():
    controller = _controller()
    command, last_step = _drive_to_stable(controller)
    frozen = tuple(command.finger_targets_rad)
    first_detector_step = {
        name: command.observations[name].step for name in ("f1", "f2", "f3")
    }
    for extra in range(1, 6):
        command = _step(controller, last_step + extra)
        assert tuple(command.finger_targets_rad) == frozen
        assert command.finger_stiffness_scale == (
            DEFAULT_SOFT_HOLD_STIFFNESS_SCALE,
        ) * 3
        assert command.lift_ready is False
        assert command.controller_phase == CONTROLLER_PHASE_STABLE_HOLD
    for name in ("f1", "f2", "f3"):
        assert (
            command.observations[name].step
            == first_detector_step[name] + 5
        )


def test_begin_consolidation_only_after_stable_and_once():
    controller = _controller()
    with pytest.raises(RuntimeError):
        controller.begin_consolidation()
    _drive_to_stable(controller)
    controller.begin_consolidation()
    with pytest.raises(RuntimeError):
        controller.begin_consolidation()


def test_ramp_monotonic_bounded_from_soft_to_final():
    ramp_steps = 8
    controller = _controller(
        consolidation_ramp_steps=ramp_steps,
        consolidation_window_steps=4,
    )
    _command, last_step = _drive_to_stable(controller)
    controller.begin_consolidation()
    command = None
    scales = []
    for index in range(1, ramp_steps + 1):
        command = _step(controller, last_step + index)
        assert not command.failed
        assert command.lift_ready is False
        assert command.controller_phase == CONTROLLER_PHASE_CONSOLIDATION_RAMP
        assert len(set(command.finger_stiffness_scale)) == 1
        scales.append(command.finger_stiffness_scale[0])
        assert command.evidence["consolidation_ramp_step"] == index
        assert command.evidence["consolidation_window_step"] == 0
    assert scales[0] == pytest.approx(
        DEFAULT_SOFT_HOLD_STIFFNESS_SCALE
        + (
            DEFAULT_CONSOLIDATION_FINAL_STIFFNESS_SCALE
            - DEFAULT_SOFT_HOLD_STIFFNESS_SCALE
        )
        / ramp_steps
    )
    assert scales[-1] == DEFAULT_CONSOLIDATION_FINAL_STIFFNESS_SCALE
    assert all(
        later + 1e-12 >= earlier
        for earlier, later in zip(scales, scales[1:])
    )
    assert min(scales) >= DEFAULT_SOFT_HOLD_STIFFNESS_SCALE
    assert max(scales) <= DEFAULT_CONSOLIDATION_FINAL_STIFFNESS_SCALE
    assert command.evidence["consolidation_scale_monotonic"] is True


def test_window_fixed_final_scale_and_lift_ready_only_at_end():
    ramp_steps = 8
    window_steps = 6
    controller = _controller(
        consolidation_ramp_steps=ramp_steps,
        consolidation_window_steps=window_steps,
    )
    _command, last_step = _drive_to_stable(controller)
    controller.begin_consolidation()
    command = None
    for index in range(1, ramp_steps + window_steps + 1):
        command = _step(controller, last_step + index)
        assert not command.failed
        assert command.finger_stiffness_scale[0] == (
            DEFAULT_CONSOLIDATION_FINAL_STIFFNESS_SCALE
            if index > ramp_steps
            else command.finger_stiffness_scale[0]
        )
    assert command.lift_ready is True
    assert command.controller_phase == CONTROLLER_PHASE_LIFT_READY
    assert command.evidence["consolidation_window_step"] == window_steps
    assert command.evidence["consolidation_ramp_step"] == ramp_steps


def test_lift_ready_requires_full_window():
    ramp_steps = 4
    window_steps = 5
    controller = _controller(
        consolidation_ramp_steps=ramp_steps,
        consolidation_window_steps=window_steps,
    )
    _command, last_step = _drive_to_stable(controller)
    controller.begin_consolidation()
    command = None
    for index in range(1, ramp_steps + window_steps):
        command = _step(controller, last_step + index)
    assert command.lift_ready is False
    assert command.controller_phase == CONTROLLER_PHASE_CONSOLIDATION_WINDOW
    assert (
        command.finger_stiffness_scale[0]
        == DEFAULT_CONSOLIDATION_FINAL_STIFFNESS_SCALE
    )
    command = _step(controller, last_step + ramp_steps + window_steps)
    assert command.lift_ready is True
    assert command.controller_phase == CONTROLLER_PHASE_LIFT_READY


def test_targets_bit_frozen_through_ramp_and_window():
    controller = _controller(
        consolidation_ramp_steps=4,
        consolidation_window_steps=4,
    )
    _command, last_step = _drive_to_stable(controller)
    frozen = tuple(controller.targets)
    controller.begin_consolidation()
    command = None
    for index in range(1, 9):
        command = _step(controller, last_step + index)
        assert tuple(command.finger_targets_rad) == frozen
        assert command.evidence["targets_match_frozen"] is True
        assert command.evidence["frozen_targets_rad"] == list(frozen)


def test_slip_during_consolidation_fails_closed():
    controller = _controller(
        consolidation_ramp_steps=4,
        consolidation_window_steps=4,
    )
    _command, last_step = _drive_to_stable(controller)
    controller.begin_consolidation()
    command = _step(controller, last_step + 1)
    assert not command.failed
    for extra in range(1, 22):
        command = controller.update(
            (0.0, 0.0, 0.0),
            tuple(controller.targets),
            (1.0, 1.0, 1.0),
            timestamp_s=(last_step + 1 + extra) * 0.01,
        )
        if command.failed:
            break
    assert command.failed
    assert command.lift_ready is False
    assert command.failure_reason is not None


def test_begin_consolidation_rejects_failed_controller():
    controller = _controller(
        consolidation_ramp_steps=4,
        consolidation_window_steps=4,
    )
    _command, last_step = _drive_to_stable(controller)
    controller.begin_consolidation()
    for extra in range(1, 22):
        command = controller.update(
            (0.0, 0.0, 0.0),
            tuple(controller.targets),
            (1.0, 1.0, 1.0),
            timestamp_s=(last_step + extra) * 0.01,
        )
        if command.failed:
            break
    with pytest.raises(RuntimeError):
        controller.begin_consolidation()


def test_extra_updates_after_lift_ready_hold_state():
    controller = _controller(
        consolidation_ramp_steps=3,
        consolidation_window_steps=3,
    )
    _command, last_step = _drive_to_stable(controller)
    controller.begin_consolidation()
    frozen = tuple(controller.targets)
    command = None
    for index in range(1, 7):
        command = _step(controller, last_step + index)
    assert command.lift_ready is True
    for extra in range(1, 4):
        command = _step(controller, last_step + 6 + extra)
        assert command.lift_ready is True
        assert tuple(command.finger_targets_rad) == frozen
        assert (
            command.finger_stiffness_scale[0]
            == DEFAULT_CONSOLIDATION_FINAL_STIFFNESS_SCALE
        )
        assert command.controller_phase == CONTROLLER_PHASE_LIFT_READY


@pytest.mark.parametrize(
    "overrides",
    [
        {"consolidation_ramp_steps": True},
        {"consolidation_window_steps": 12.5},
        {"soft_hold_stiffness_scale": 0.0},
        {
            "soft_hold_stiffness_scale": 0.8,
            "consolidation_final_stiffness_scale": 0.6,
        },
        {"consolidation_final_stiffness_scale": 1.1},
        {"consolidation_threshold_label": "SIM_TUNING_ONLY"},
    ],
)
def test_consolidation_config_rejects_invalid_values(overrides):
    import dataclasses

    with pytest.raises(ValueError):
        dataclasses.replace(_config(), **overrides)


def test_consolidation_defaults_are_frozen_candidates():
    config = _config()
    assert config.soft_hold_stiffness_scale == 0.35
    assert config.consolidation_final_stiffness_scale == 1.0
    assert config.consolidation_ramp_steps == DEFAULT_CONSOLIDATION_RAMP_STEPS
    assert config.consolidation_window_steps == (
        DEFAULT_CONSOLIDATION_WINDOW_STEPS
    )
    assert config.consolidation_threshold_label == (
        CONSOLIDATION_THRESHOLD_LABEL
    )
    assert DEFAULT_CONSOLIDATION_RAMP_STEPS == 120
    assert DEFAULT_CONSOLIDATION_WINDOW_STEPS == 240


def test_command_carries_consolidation_evidence():
    controller = _controller(
        consolidation_ramp_steps=4, consolidation_window_steps=4
    )
    _command, last_step = _drive_to_stable(controller)
    controller.begin_consolidation()
    command = _step(controller, last_step + 1)
    evidence = command.evidence
    for key in (
        "controller_phase",
        "lift_ready",
        "consolidation_armed",
        "consolidation_ramp_step",
        "consolidation_window_step",
        "consolidation_stiffness_scale",
        "consolidation_scale_monotonic",
        "consolidation_final_stiffness_scale_configured",
        "soft_hold_stiffness_scale_configured",
        "consolidation_threshold_label",
        "targets_match_frozen",
    ):
        assert key in evidence, key
    assert evidence["consolidation_threshold_label"] == (
        "SIM_TUNING_ONLY_A_CANDIDATE"
    )


def test_final_scale_one_legal_and_above_upper_bound_rejected():
    import dataclasses

    config = dataclasses.replace(
        _config(), consolidation_final_stiffness_scale=1.0
    )
    assert config.consolidation_final_stiffness_scale == 1.0
    with pytest.raises(ValueError):
        dataclasses.replace(
            _config(), consolidation_final_stiffness_scale=1.01
        )
    with pytest.raises(ValueError):
        dataclasses.replace(
            _config(), consolidation_final_stiffness_scale=0.35
        )
