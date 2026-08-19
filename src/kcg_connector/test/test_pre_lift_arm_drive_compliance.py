import numpy as np
import pytest

from kcg_connector.grasp.pre_lift_arm_drive_compliance import (
    PreLiftArmDriveComplianceConfig,
    capture_position_preload_nm,
    compliant_path_drive_target,
    derive_bumpless_drive_step,
)


def _config():
    return PreLiftArmDriveComplianceConfig(
        enabled=True,
        threshold_label="SIM_TUNING_ONLY_B_V2_H6",
        stiffness_scale=0.25,
        damping_scale=1.0,
        transition_steps=240,
        stability_window_steps=240,
        maximum_bumpless_target_delta_rad=0.010,
        maximum_position_effort_residual_nm=1.0e-6,
        maximum_entry_moment_score_nm=0.24,
        maximum_entry_load_imbalance=0.18,
    )


def _run06_states():
    realized = np.asarray(
        [
            -0.12934480607509613,
            0.4167536497116089,
            -0.4029049277305603,
            -1.1201527118682861,
            0.15923148393630981,
            1.636252999305725,
            -0.10705240070819855,
        ]
    )
    commanded = np.asarray(
        [
            -0.129340616881,
            0.415154160813,
            -0.403066037272,
            -1.119324443931,
            0.159214032504,
            1.636235747425,
            -0.107043622479,
        ]
    )
    return commanded, realized


def test_h6_final_gain_step_preserves_position_effort_exactly():
    commanded, realized = _run06_states()
    preload = capture_position_preload_nm(commanded, realized, 24000.0)
    result = derive_bumpless_drive_step(
        realized,
        preload,
        24000.0,
        400.0,
        239,
        _config(),
    )
    target = np.asarray(result["target_arm_rad"])
    assert result["applied_stiffness"] == pytest.approx(6000.0)
    assert result["applied_damping"] == pytest.approx(400.0)
    assert result["maximum_target_bias_rad"] == pytest.approx(
        0.006397955594435567
    )
    assert result["maximum_position_effort_residual_nm"] < 1.0e-10
    assert np.allclose(6000.0 * (target - realized), preload, atol=1.0e-10)


def test_h6_transition_is_minimum_jerk_and_monotonic():
    commanded, realized = _run06_states()
    preload = capture_position_preload_nm(commanded, realized, 24000.0)
    values = [
        derive_bumpless_drive_step(
            realized, preload, 24000.0, 400.0, step, _config()
        )
        for step in range(240)
    ]
    stiffness = [value["applied_stiffness"] for value in values]
    assert stiffness[0] < 24000.0
    assert stiffness[-1] == pytest.approx(6000.0)
    assert all(a >= b for a, b in zip(stiffness, stiffness[1:]))
    assert max(value["maximum_target_bias_rad"] for value in values) < 0.010


def test_h6_path_target_retains_same_bounded_bias():
    commanded, realized = _run06_states()
    preload = capture_position_preload_nm(commanded, realized, 24000.0)
    path = realized + np.asarray([0.0, 0.001, 0.0, -0.001, 0.0, 0.0, 0.0])
    result = compliant_path_drive_target(path, preload, 24000.0, _config())
    assert np.allclose(
        np.asarray(result["target_arm_rad"]) - path,
        preload / 6000.0,
    )
    assert result["maximum_target_bias_rad"] < 0.010


def test_h6_rejects_parameter_search_surface():
    values = dict(_config().__dict__)
    values["stiffness_scale"] = 0.30
    with pytest.raises(ValueError, match="frozen at the derived value"):
        PreLiftArmDriveComplianceConfig(**values)
