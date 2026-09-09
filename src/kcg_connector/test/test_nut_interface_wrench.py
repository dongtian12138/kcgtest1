"""Analytical force balances for the connector-origin wrench used in turning."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "isaac"))
from te_body_nut_rotation import interface_wrench_from_wrist, interface_wrench_from_sensor_sample


@pytest.mark.parametrize("scene_gravity", [-9.81, 9.81])
def test_freely_suspended_offset_payload_has_zero_interface_load(scene_gravity):
    # A 2 kg payload at (0, 0, .02), measured about (.1, -.2, .5).
    # Its downward weight produces (-3.924, -1.962, 0) N m at the wrist.
    result = interface_wrench_from_wrist(
        [0., 0., -19.62, -3.924, -1.962, 0.], [.1, -.2, .5],
        [0., 0., 0.], [0., 0., .02], 2., scene_gravity, np.eye(3))
    np.testing.assert_allclose(result, np.zeros(6), atol=1e-12)


@pytest.mark.parametrize("scene_gravity", [-9.81, 9.81])
def test_known_support_force_and_twist_at_plug_origin(scene_gravity):
    # Add (1, 0, 4) N at the plug origin and a +.3 N m world-z couple.
    # The socket x axis points along world y, so the x force is socket -y.
    rotation = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
    result = interface_wrench_from_wrist(
        [1., 0., -15.62, -3.124, -2.062, .1], [.1, -.2, .5],
        [0., 0., 0.], [0., 0., .02], 2., scene_gravity, rotation)
    np.testing.assert_allclose(result, [0., -1., 4., 0., 0., .3], atol=1e-12)


@pytest.mark.parametrize("scene_gravity", [-9.81, 9.81])
def test_pure_couple_is_preserved_in_tilted_socket_frame(scene_gravity):
    # A +.04 N m world-y couple is -.04 N m about socket z after Rx(90).
    rotation = [[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]]
    result = interface_wrench_from_wrist(
        [0., 0., -19.62, -3.924, -1.922, 0.], [.1, -.2, .5],
        [0., 0., 0.], [0., 0., .02], 2., scene_gravity, rotation)
    np.testing.assert_allclose(result, [0., 0., 0., 0., 0., -.04], atol=1e-12)


@pytest.mark.parametrize("inertia_already_applied", [False, True])
def test_saved_sensor_frames_recover_known_physical_support(inertia_already_applied):
    hand = np.eye(4)
    hand[:3, :3] = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
    hand[:3, 3] = [.1, -.2, .5]
    task = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    # The same independent 2 kg payload / (1,0,4) N support / .3 N m couple
    # balance above, now measured in a different task frame with known inertia.
    balanced = np.array([1., 0., -15.62, -3.124, -2.062, .1])
    inertia = np.array([.2, .3, -.4, .01, -.02, .03])
    measured = balanced if inertia_already_applied else balanced-inertia
    sample = {
        "gravity_and_dynamic_compensated_task_wrench": np.r_[task.T@measured[:3], task.T@measured[3:]],
        "dynamic_inertia_prediction": {
            "ready": True, "applied_to_safety_residual": inertia_already_applied,
            "predicted_positive_inertia_wrench_sensor": np.r_[hand[:3, :3].T@inertia[:3], hand[:3, :3].T@inertia[3:]],
        },
    }
    result = interface_wrench_from_sensor_sample(
        sample, task, hand, [0., 0., 0.], [0., 0., .02], 2., -9.81, hand[:3, :3])
    np.testing.assert_allclose(result, [0., -1., 4., 0., 0., .3], atol=1e-12)
