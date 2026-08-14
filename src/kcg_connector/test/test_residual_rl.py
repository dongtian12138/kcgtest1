import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from kcg_connector.residual_rl import (
    RESIDUAL_ACTION_NAMES,
    RESIDUAL_ACTION_SIZE,
    RESIDUAL_OBSERVATION_NAMES,
    RESIDUAL_OBSERVATION_SIZE,
    ConnectorResidualState,
    calculate_residual_reward,
    decode_residual_action,
    evaluate_residual_state,
    load_connector_residual_config,
    residual_observation,
)


CONFIG_PATH = Path(__file__).parents[1] / "config" / "connector_task.yaml"
CONFIG = load_connector_residual_config(CONFIG_PATH)


def state(**overrides):
    values = {
        "phase_progress": 0.0,
        "q7_position_rad": -0.05,
        "q7_tracking_error_rad": 0.0,
        "q7_velocity_rad_s": (
            CONFIG.tightening_direction * CONFIG.nominal_q7_speed_rad_s
        ),
        "nut_angle_rad": 0.0,
        "nut_angular_velocity_rad_s": CONFIG.nominal_q7_speed_rad_s,
        "axial_travel_m": 0.0,
        "axial_velocity_m_s": CONFIG.maximum_axial_speed_m_s * 0.9,
        "grasp_translation_error_m": (0.0, 0.0, 0.0),
        "grasp_rotation_error_rad": (0.0, 0.0, 0.0),
        "finger_torques_nm": (0.10, 0.12, 0.11),
        "finger_torque_deltas_nm": (0.0, 0.0, 0.0),
        "clamp_positions_rad": CONFIG.clamp_nominal_positions_rad,
    }
    values.update(overrides)
    return ConnectorResidualState(**values)


def test_contract_is_four_actions_and_twenty_four_observations():
    assert RESIDUAL_ACTION_SIZE == 4
    assert RESIDUAL_OBSERVATION_SIZE == 24
    assert len(set(RESIDUAL_ACTION_NAMES)) == 4
    assert len(set(RESIDUAL_OBSERVATION_NAMES)) == 24
    assert not any("tactile" in name for name in RESIDUAL_OBSERVATION_NAMES)


def test_zero_action_is_safe_nominal_q7_and_clamp_command():
    decoded = decode_residual_action(np.zeros(4), CONFIG)
    assert decoded.q7_velocity_target_rad_s == pytest.approx(
        CONFIG.tightening_direction * CONFIG.nominal_q7_speed_rad_s
    )
    assert decoded.clamp_position_targets_rad == pytest.approx(
        CONFIG.clamp_nominal_positions_rad
    )


@pytest.mark.parametrize("q7_action", [-1.0, 1.0])
def test_q7_residual_cannot_reverse_or_exceed_speed_limit(q7_action):
    decoded = decode_residual_action([q7_action, 0.0, 0.0, 0.0], CONFIG)
    speed = decoded.q7_velocity_target_rad_s
    assert math.copysign(1.0, speed) == CONFIG.tightening_direction
    assert abs(speed) <= CONFIG.maximum_q7_speed_rad_s
    assert abs(speed) > 0.0


def test_clamp_residuals_are_independently_bounded():
    decoded = decode_residual_action([0.0, -2.0, 0.5, 2.0], CONFIG)
    expected = np.asarray(CONFIG.clamp_nominal_positions_rad) + np.asarray(
        [-1.0, 0.5, 1.0]
    ) * np.asarray(CONFIG.clamp_position_residual_limits_rad)
    assert decoded.clamp_position_targets_rad == pytest.approx(expected)


def test_observation_is_finite_bounded_float32_and_versioned():
    observation = residual_observation(
        state(
            phase_progress=0.5,
            nut_angle_rad=CONFIG.target_angle_rad / 2.0,
            axial_travel_m=CONFIG.expected_axial_travel_m / 2.0,
        ),
        CONFIG,
    )
    assert CONFIG.interface_version == "kcg_connector_twist_residual_v0"
    assert observation.shape == (24,)
    assert observation.dtype == np.float32
    assert np.all(np.isfinite(observation))
    assert np.all(observation >= -1.0)
    assert np.all(observation <= 1.0)


def test_nonfinite_observation_is_rejected():
    with pytest.raises(ValueError, match="finite"):
        residual_observation(state(nut_angle_rad=math.nan), CONFIG)


def test_success_uses_measured_nut_angle_and_helical_travel():
    complete = state(
        phase_progress=1.0,
        nut_angle_rad=CONFIG.target_angle_rad,
        q7_velocity_rad_s=0.0,
        nut_angular_velocity_rad_s=0.0,
        axial_travel_m=CONFIG.expected_axial_travel_m,
        axial_velocity_m_s=0.0,
        stable_hold_seconds=CONFIG.success_hold_duration_s,
    )
    result = evaluate_residual_state(complete, CONFIG)
    assert result.terminated
    assert result.success
    assert result.reason == "success"


def test_q7_motion_without_measured_nut_progress_is_not_success():
    result = evaluate_residual_state(
        state(
            phase_progress=1.0,
            q7_position_rad=-2.1,
            q7_tracking_error_rad=0.0,
            nut_angle_rad=0.0,
            axial_travel_m=0.0,
        ),
        CONFIG,
    )
    assert not result.terminated
    assert not result.success


def test_inconsistent_measured_helix_fails_as_cross_thread():
    result = evaluate_residual_state(
        state(
            nut_angle_rad=0.75 * CONFIG.target_angle_rad,
            axial_travel_m=0.0,
        ),
        CONFIG,
    )
    assert result.terminated
    assert result.reason == "cross_thread"


def test_short_probe_requires_at_least_seventy_five_percent_axial_motion():
    result = evaluate_residual_state(
        state(
            nut_angle_rad=CONFIG.target_angle_rad,
            axial_travel_m=0.50 * CONFIG.expected_axial_travel_m,
        ),
        CONFIG,
    )
    assert result.terminated
    assert result.reason == "cross_thread"


def test_stage_threshold_controls_cross_thread_and_success_gates():
    strict = replace(CONFIG, minimum_axial_progress_fraction=0.90)
    expected = strict.expected_axial_travel_m
    insufficient = evaluate_residual_state(
        state(
            nut_angle_rad=strict.target_angle_rad,
            axial_travel_m=0.89 * expected,
            q7_velocity_rad_s=0.0,
            nut_angular_velocity_rad_s=0.0,
            axial_velocity_m_s=0.0,
            stable_hold_seconds=strict.success_hold_duration_s,
        ),
        strict,
    )
    complete = evaluate_residual_state(
        state(
            nut_angle_rad=strict.target_angle_rad,
            axial_travel_m=0.90 * expected,
            q7_velocity_rad_s=0.0,
            nut_angular_velocity_rad_s=0.0,
            axial_velocity_m_s=0.0,
            stable_hold_seconds=strict.success_hold_duration_s,
        ),
        strict,
    )
    assert insufficient.reason == "cross_thread"
    assert complete.success


def test_eighteen_degrees_is_not_a_complete_twenty_degree_probe():
    angle = math.radians(18.0)
    result = evaluate_residual_state(
        state(
            nut_angle_rad=angle,
            axial_travel_m=(
                CONFIG.helical_lead_m * angle / (2.0 * math.pi)
            ),
        ),
        CONFIG,
    )
    assert not result.terminated
    assert not result.success


def test_overtwist_is_an_explicit_failure():
    angle = CONFIG.target_angle_rad + math.radians(1.0)
    result = evaluate_residual_state(
        state(
            nut_angle_rad=angle,
            axial_travel_m=(
                CONFIG.helical_lead_m * angle / (2.0 * math.pi)
            ),
        ),
        CONFIG,
    )
    assert result.terminated
    assert result.reason == "overtwist"


def test_target_while_still_moving_enters_hold_without_success():
    result = evaluate_residual_state(
        state(
            phase_progress=1.0,
            nut_angle_rad=CONFIG.target_angle_rad,
            axial_travel_m=CONFIG.expected_axial_travel_m,
            stable_hold_seconds=CONFIG.success_hold_duration_s,
        ),
        CONFIG,
    )
    assert not result.terminated
    assert result.reason == "holding"


def test_stopped_target_requires_continuous_half_second_hold():
    result = evaluate_residual_state(
        state(
            phase_progress=1.0,
            nut_angle_rad=CONFIG.target_angle_rad,
            q7_velocity_rad_s=0.0,
            nut_angular_velocity_rad_s=0.0,
            axial_travel_m=CONFIG.expected_axial_travel_m,
            axial_velocity_m_s=0.0,
            stable_hold_seconds=CONFIG.success_hold_duration_s - 0.1,
        ),
        CONFIG,
    )
    assert not result.terminated
    assert result.reason == "holding"


def test_lost_torque_channels_and_overload_fail_explicitly():
    lost = evaluate_residual_state(
        state(finger_torques_nm=(0.0, 0.0, 0.10)), CONFIG
    )
    overloaded = evaluate_residual_state(
        state(finger_torques_nm=(0.10, 1.10, 0.10)), CONFIG
    )
    assert lost.reason == "lost_grasp"
    assert overloaded.reason == "finger_overload"


def test_reward_uses_physical_nut_progress_not_policy_action():
    initial = state()
    unchanged_reward = calculate_residual_reward(
        initial, initial, [1.0, 0.0, 0.0, 0.0], CONFIG
    )
    advanced_angle = math.radians(2.0)
    advanced = state(
        nut_angle_rad=advanced_angle,
        axial_travel_m=(
            CONFIG.helical_lead_m * advanced_angle / (2.0 * math.pi)
        ),
    )
    progress_reward = calculate_residual_reward(
        initial, advanced, np.zeros(4), CONFIG
    )
    assert unchanged_reward.terms["nut_progress"] == 0.0
    assert progress_reward.terms["nut_progress"] > 0.0
    assert progress_reward.total > unchanged_reward.total


def test_zero_progress_has_nonpositive_reward():
    stationary = state(
        q7_velocity_rad_s=0.0,
        nut_angular_velocity_rad_s=0.0,
        axial_velocity_m_s=0.0,
    )
    reward = calculate_residual_reward(
        stationary, stationary, np.zeros(4), CONFIG
    )
    assert reward.total <= 0.0


def test_large_grasp_rotation_is_lost_grasp():
    result = evaluate_residual_state(
        state(grasp_rotation_error_rad=(0.0, 0.0, math.pi)), CONFIG
    )
    assert result.terminated
    assert result.reason == "lost_grasp"
