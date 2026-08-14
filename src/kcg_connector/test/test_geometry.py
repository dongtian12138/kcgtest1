import math

import numpy as np
import pytest

from kcg_connector.geometry import (
    axis_angle_error,
    helical_travel,
    relative_pose,
    split_axial_error,
    unwrap_angle,
)


IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])


def test_relative_pose_uses_source_frame():
    quarter_turn = np.array([0.0, 0.0, math.sin(math.pi / 4.0), math.cos(math.pi / 4.0)])
    position, orientation = relative_pose(
        [1.0, 2.0, 3.0], quarter_turn, [1.0, 3.0, 3.0], quarter_turn
    )
    assert np.allclose(position, [1.0, 0.0, 0.0], atol=1.0e-9)
    assert np.allclose(orientation, IDENTITY, atol=1.0e-9)


def test_axis_angle_error_is_directed():
    flipped = np.array([1.0, 0.0, 0.0, 0.0])
    assert axis_angle_error(IDENTITY, IDENTITY) == pytest.approx(0.0)
    assert axis_angle_error(IDENTITY, flipped) == pytest.approx(math.pi)


def test_split_axial_error():
    axial, lateral = split_axial_error([0.003, 0.004, -0.010])
    assert axial == pytest.approx(-0.010)
    assert lateral == pytest.approx(0.005)


def test_helical_travel_for_one_turn():
    assert helical_travel(2.0 * math.pi, 0.004) == pytest.approx(0.004)


def test_unwrap_angle_crosses_positive_pi_continuously():
    previous = math.radians(179.0)
    current = math.radians(-179.0)
    assert unwrap_angle(previous, current) == pytest.approx(math.radians(181.0))


def test_zero_quaternion_is_rejected():
    with pytest.raises(ValueError):
        relative_pose([0.0] * 3, [0.0] * 4, [0.0] * 3, IDENTITY)
