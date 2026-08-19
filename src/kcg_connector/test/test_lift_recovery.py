import numpy as np
import pytest

from kcg_connector.grasp.lift_recovery import (
    LiftRecoveryConfig,
    plan_recovery_open,
    plan_recovery_return,
)


def _config(**overrides):
    values = {
        "return_steps_per_waypoint": 2,
        "settle_steps": 3,
        "open_duration_s": 3.5,
    }
    values.update(overrides)
    return LiftRecoveryConfig(**values)


def test_recovery_return_reverses_repeats_then_settles_at_prelift():
    traversed = [np.full(7, 1.0), np.full(7, 2.0), np.full(7, 3.0)]
    plan = plan_recovery_return(traversed, np.full(7, 0.0), _config())
    expected = [3.0, 3.0, 2.0, 2.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    assert len(plan) == len(expected)
    assert [float(waypoint[0]) for waypoint in plan] == expected
    for waypoint in plan:
        assert waypoint.shape == (7,)
        assert np.allclose(waypoint, waypoint[0])


def test_recovery_return_uses_half_speed_replay_for_single_step():
    traversed = [np.full(7, 0.25)]
    plan = plan_recovery_return(traversed, np.zeros(7), _config(settle_steps=1))
    assert len(plan) == 2 + 1
    assert [float(waypoint[0]) for waypoint in plan] == [0.25, 0.25, 0.0]


def test_recovery_return_empty_traversal_is_settle_only():
    # Zero-lift hold gate trigger: the arm never moved during the hold, so
    # the only safe return is settling at the pre-lift target.
    plan = plan_recovery_return([], np.zeros(7), _config())
    assert len(plan) == 3
    for waypoint in plan:
        assert waypoint.shape == (7,)
        assert np.allclose(waypoint, np.zeros(7))


def test_recovery_return_validates_inputs():
    with pytest.raises(ValueError, match="7-vector"):
        plan_recovery_return([np.zeros(6)], np.zeros(7), _config())
    with pytest.raises(ValueError, match="7-vector"):
        plan_recovery_return([np.full(7, np.nan)], np.zeros(7), _config())
    with pytest.raises(ValueError, match="pre-lift"):
        plan_recovery_return([np.zeros(7)], np.full(7, np.inf), _config())


def test_recovery_open_is_bounded_monotone_and_ends_open():
    held = np.zeros(4)
    open_target = np.full(4, 0.8)
    plan = plan_recovery_open(held, open_target, 0.5, 20)
    assert len(plan) == 10
    assert np.all(plan[0] >= held)
    assert np.all(plan[0] < open_target)
    assert np.allclose(plan[-1], open_target)
    for before, after in zip(plan, plan[1:]):
        assert np.all(after >= before)
        assert np.all(after <= open_target)


def test_recovery_open_validates_inputs():
    with pytest.raises(ValueError, match="4-vector"):
        plan_recovery_open(np.zeros(3), np.zeros(4), 1.0, 100.0)
    with pytest.raises(ValueError, match="duration"):
        plan_recovery_open(np.zeros(4), np.zeros(4), 0.0, 100.0)
    with pytest.raises(ValueError, match="rate"):
        plan_recovery_open(np.zeros(4), np.zeros(4), 1.0, 0.0)


def test_recovery_config_validation():
    with pytest.raises(ValueError):
        LiftRecoveryConfig(0, 1, 1.0)
    with pytest.raises(ValueError):
        LiftRecoveryConfig(1, 0, 1.0)
    with pytest.raises(ValueError):
        LiftRecoveryConfig(1, 1, 0.0)
    with pytest.raises(ValueError):
        LiftRecoveryConfig(1, 1, float("inf"))
