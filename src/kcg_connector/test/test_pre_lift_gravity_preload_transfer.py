import numpy as np
import pytest

from kcg_connector.grasp.pre_lift_gravity_preload_transfer import (
    PreLiftGravityPreloadTransferConfig,
    derive_gravity_preload_transfer_step,
    gravity_supported_path_drive_target,
)


def _config():
    return PreLiftGravityPreloadTransferConfig(
        enabled=True,
        threshold_label="SIM_TUNING_ONLY_B_V2_H15",
        transition_steps=240,
        stability_window_steps=240,
        maximum_gravity_feedforward_nm=300.0,
        maximum_feedforward_step_nm=2.5,
        maximum_target_bias_rad=0.01,
        maximum_effort_continuity_residual_nm=1.0e-6,
        maximum_effort_readback_error_nm=1.0e-5,
        maximum_entry_moment_score_nm=0.24,
        maximum_entry_load_imbalance=0.18,
    )


def _states():
    path = np.asarray([0.1, -0.4, 0.2, -1.1, 0.1, 1.6, -0.1])
    preload = np.asarray([0.9, -41.0, -3.2, 21.7, 0.18, -0.06, 0.10])
    gravity = np.asarray([0.1, -38.0, -0.2, 19.0, 0.05, 0.4, 0.01])
    return path, preload, gravity


def test_h15_final_transfer_preserves_total_effort_and_reduces_position_bias():
    path, preload, gravity = _states()
    result = derive_gravity_preload_transfer_step(
        path,
        preload,
        gravity,
        6000.0,
        239,
        np.asarray(gravity) * (1.0 - 1.0e-8),
        _config(),
    )
    assert np.asarray(result["gravity_feedforward_nm"]) == pytest.approx(gravity)
    assert np.asarray(result["total_effort_nm"]) == pytest.approx(preload)
    assert result["maximum_effort_continuity_residual_nm"] < 1.0e-10
    assert result["maximum_target_bias_rad"] < np.max(np.abs(preload / 6000.0))


def test_h15_minimum_jerk_transfer_is_bounded_and_continuous():
    path, preload, gravity = _states()
    previous = np.zeros(7)
    blends = []
    for step in range(240):
        result = derive_gravity_preload_transfer_step(
            path,
            preload,
            gravity,
            6000.0,
            step,
            previous,
            _config(),
        )
        previous = np.asarray(result["gravity_feedforward_nm"])
        blends.append(result["minimum_jerk_blend"])
        assert result["maximum_feedforward_step_nm"] <= 2.5
        assert np.asarray(result["total_effort_nm"]) == pytest.approx(preload)
    assert blends[0] > 0.0
    assert blends[-1] == pytest.approx(1.0)
    assert all(left <= right for left, right in zip(blends, blends[1:]))


def test_h15_full_gravity_path_target_keeps_only_residual_position_preload():
    path, preload, gravity = _states()
    result = gravity_supported_path_drive_target(
        path,
        preload,
        gravity,
        6000.0,
        gravity,
        _config(),
    )
    target = np.asarray(result["target_arm_rad"])
    assert target - path == pytest.approx((preload - gravity) / 6000.0)
    assert np.asarray(result["total_effort_nm"]) == pytest.approx(preload)


def test_h15_rejects_unbounded_feedforward_and_parameter_search():
    path, preload, gravity = _states()
    excessive = gravity.copy()
    excessive[1] = 301.0
    with pytest.raises(ValueError, match="actuator bound"):
        gravity_supported_path_drive_target(
            path, preload, excessive, 6000.0, excessive, _config()
        )
    values = dict(_config().__dict__)
    values["maximum_feedforward_step_nm"] = 3.0
    with pytest.raises(ValueError, match="frozen"):
        PreLiftGravityPreloadTransferConfig(**values)
