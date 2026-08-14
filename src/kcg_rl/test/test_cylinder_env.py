import numpy as np

from kcg_rl.cylinder_env import (
    APPROACH_POSITIONS,
    HAND_LOWER_BOUNDS,
    HAND_UPPER_BOUNDS,
    LIFT_POSITIONS,
    VALIDATED_CLOSED_HAND_POSITIONS,
    calculate_reward,
    decode_macro_action,
    expand_hand_positions,
    interpolate_arm_positions,
)


def test_expand_hand_positions_maps_all_coupled_joints():
    active = [1.0, 0.75, 0.50, 0.70]
    expanded = expand_hand_positions(active)
    assert expanded.tolist() == [
        1.0,
        0.75,
        0.50,
        0.70,
        1.0,
        0.75,
        0.50,
        0.70,
    ]


def test_normalized_macro_action_maps_bounds_and_lift_trigger():
    hand, lift_requested, action = decode_macro_action(
        [-2.0, -1.0, 0.0, 1.0, 0.75], 0.5
    )
    expected = [
        HAND_LOWER_BOUNDS[0],
        HAND_LOWER_BOUNDS[1],
        0.5 * (HAND_LOWER_BOUNDS[2] + HAND_UPPER_BOUNDS[2]),
        HAND_UPPER_BOUNDS[3],
    ]
    assert np.allclose(hand, expected)
    assert lift_requested
    assert np.allclose(action, [-1.0, -1.0, 0.0, 1.0, 0.75])


def test_zero_hand_action_is_the_validated_curriculum_center():
    hand, lift_requested, _ = decode_macro_action(
        [0.0, 0.0, 0.0, 0.0, -1.0], 0.5
    )
    assert np.allclose(hand, VALIDATED_CLOSED_HAND_POSITIONS)
    assert not lift_requested


def test_arm_interpolation_uses_validated_endpoints():
    assert np.allclose(interpolate_arm_positions(0.0), APPROACH_POSITIONS)
    assert np.allclose(interpolate_arm_positions(1.0), LIFT_POSITIONS)
    assert np.allclose(
        interpolate_arm_positions(0.5),
        0.5 * (APPROACH_POSITIONS + LIFT_POSITIONS),
    )


def test_reward_has_success_bonus_and_height_progress():
    previous = {"height_gain": 0.06, "grasp_distance": 0.01}
    metrics = {
        "height_gain": 0.08,
        "grasp_distance": 0.005,
        "loaded_torque_channels": 3,
    }
    reward, terms = calculate_reward(
        previous, metrics, np.zeros(5), "success"
    )
    assert terms["height_progress"] > 0.0
    assert terms["terminal"] == 25.0
    assert reward > 25.0
