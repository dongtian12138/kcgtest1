"""Retiring a load reference must not disable independent signal stops."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "isaac/carts_v2"))
from controller import JointSignalStepper


def check(action="record_only", *, invalid_position=False, speed=0., tracking=0.):
    stepper = JointSignalStepper.__new__(JointSignalStepper)
    stepper.settings = {"measured_effort_abort_action": action, "measured_effort_abort_nm": .9,
                        "maximum_joint_speed_rad_s": 3., "maximum_arm_tracking_error_rad": .2}
    position, velocity, effort = np.zeros(11), np.zeros(11), np.zeros(11)
    effort[8] = 1.2284153699874878  # Actual previously stopping CPU sample.
    position[0] = np.nan if invalid_position else tracking
    velocity[0] = speed
    stepper.latest = position, velocity, effort
    stepper.abort_reason = None
    stepper._apply_signal_aborts(np.zeros(7), position, velocity, effort)
    return stepper.abort_reason


def test_legacy_load_action_and_record_only_are_distinct():
    assert check("abort") == "HAND_MEASURED_EFFORT_ABORT"
    assert check() is None


@pytest.mark.parametrize("kwargs, expected", [
    ({"invalid_position": True}, "NONFINITE_JOINT_SIGNAL_ABORT"),
    ({"speed": 3.1}, "JOINT_SPEED_ABORT"),
    ({"tracking": .3}, "ARM_TRACKING_ERROR_ABORT"),
])
def test_independent_stops_remain_active(kwargs, expected):
    assert check(**kwargs) == expected
