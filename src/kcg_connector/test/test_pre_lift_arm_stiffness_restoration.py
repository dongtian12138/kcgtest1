import numpy as np
import pytest

from kcg_connector.grasp.pre_lift_arm_stiffness_restoration import (
    PreLiftArmStiffnessRestorationConfig,
    derive_runtime_target_bias_step_envelope,
    derive_stiffness_restoration_step,
    restored_nominal_drive_target,
)


def _config():
    return PreLiftArmStiffnessRestorationConfig(
        enabled=True,
        threshold_label="SIM_TUNING_ONLY_B_V2_H16",
        source_h6_run_id="B-V2-GRASP-07",
        source_h13_run_id="B-V2-GRASP-13",
        transition_steps=240,
        stability_window_steps=240,
        expected_initial_stiffness_nm_rad=6000.0,
        expected_restored_stiffness_nm_rad=24000.0,
        expected_damping_nm_s_rad=875.29978045654,
        maximum_stiffness_step_nm_rad=140.62,
        target_bias_step_guard=(
            "FRESH_ENTRY_PRELOAD_EXACT_FROZEN_SCHEDULE"
        ),
        offline_reference_maximum_target_bias_step_rad=5.3e-5,
        maximum_target_bias_rad=0.01,
        maximum_position_effort_residual_nm=1.0e-6,
        maximum_gain_readback_error=1.0e-3,
        maximum_entry_moment_score_nm=0.24,
        maximum_entry_load_imbalance=0.18,
    )


def _state():
    q = np.asarray([-0.13, 0.417, -0.403, -1.12, 0.159, 1.636, -0.107])
    preload = np.asarray([0.8923, -41.0385, -3.2215, 21.7038, 0.1805, -0.0621, 0.0991])
    return q, preload


def test_h16_restores_existing_nominal_stiffness_and_preserves_effort():
    q, preload = _state()
    envelope = derive_runtime_target_bias_step_envelope(
        preload, 6000.0, 24000.0, 875.29978045654, _config()
    )
    previous_bias = preload / 6000.0
    final = None
    for step in range(240):
        final = derive_stiffness_restoration_step(
            q,
            preload,
            6000.0,
            24000.0,
            875.29978045654,
            step,
            previous_bias,
            _config(),
            runtime_target_bias_step_envelope_rad=envelope[
                "runtime_target_bias_step_envelope_rad"
            ],
        )
        previous_bias = np.asarray(final["target_bias_rad"])
        assert final["maximum_position_effort_residual_nm"] < 1.0e-9
    assert final is not None
    assert final["applied_stiffness_nm_rad"] == pytest.approx(24000.0)
    assert np.asarray(final["target_bias_rad"]) == pytest.approx(
        preload / 24000.0
    )


def test_h16_minimum_jerk_steps_obey_derived_stiffness_and_bias_bounds():
    q, preload = _state()
    envelope = derive_runtime_target_bias_step_envelope(
        preload, 6000.0, 24000.0, 875.29978045654, _config()
    )
    previous_bias = preload / 6000.0
    stiffness = []
    for step in range(240):
        result = derive_stiffness_restoration_step(
            q, preload, 6000.0, 24000.0, 875.29978045654,
            step, previous_bias, _config(),
            runtime_target_bias_step_envelope_rad=envelope[
                "runtime_target_bias_step_envelope_rad"
            ],
        )
        previous_bias = np.asarray(result["target_bias_rad"])
        stiffness.append(result["applied_stiffness_nm_rad"])
        assert result["stiffness_step_nm_rad"] <= 140.62
        assert result["maximum_target_bias_step_rad"] <= envelope[
            "runtime_target_bias_step_envelope_rad"
        ]
    assert all(left <= right for left, right in zip(stiffness, stiffness[1:]))


def test_h16_fresh_entry_envelope_is_exact_and_below_absolute_ceiling():
    fresh_process_preload = np.asarray(
        [
            1.3190682137731446,
            -41.50016890749475,
            -3.576056351273227,
            21.12154496630403,
            0.08774550051082741,
            -0.4405095996826702,
            0.5195785723136614,
        ]
    )
    envelope = derive_runtime_target_bias_step_envelope(
        fresh_process_preload,
        6000.0,
        24000.0,
        875.29978045654,
        _config(),
    )
    assert envelope["runtime_target_bias_step_envelope_rad"] == pytest.approx(
        5.323036507711484e-5, rel=0.0, abs=1.0e-18
    )
    assert envelope["runtime_envelope_peak_transition_step"] == 62
    assert envelope["offline_reference_exceeded"] is True
    assert (
        envelope["runtime_target_bias_step_envelope_rad"]
        < envelope["absolute_schedule_target_bias_step_ceiling_rad"]
    )


def test_h16_runtime_envelope_cannot_be_inflated_or_changed_after_entry():
    q, preload = _state()
    envelope = derive_runtime_target_bias_step_envelope(
        preload, 6000.0, 24000.0, 875.29978045654, _config()
    )
    with pytest.raises(
        ValueError, match="envelope changed after entry"
    ):
        derive_stiffness_restoration_step(
            q,
            preload,
            6000.0,
            24000.0,
            875.29978045654,
            0,
            preload / 6000.0,
            _config(),
            runtime_target_bias_step_envelope_rad=(
                envelope["runtime_target_bias_step_envelope_rad"] * 1.01
            ),
        )


def test_h16_fixed_anchor_target_keeps_nominal_stiffness_preload():
    q, preload = _state()
    result = restored_nominal_drive_target(
        q, preload, 24000.0, 875.29978045654, _config()
    )
    assert np.asarray(result["target_arm_rad"]) - q == pytest.approx(
        preload / 24000.0
    )
    assert result["maximum_position_effort_residual_nm"] < 1.0e-9


@pytest.mark.parametrize(
    "field,value",
    [
        ("expected_initial_stiffness_nm_rad", 6001.0),
        ("expected_restored_stiffness_nm_rad", 23999.0),
        ("expected_damping_nm_s_rad", 875.0),
        ("transition_steps", 241),
        ("maximum_stiffness_step_nm_rad", 141.0),
    ],
)
def test_h16_rejects_a_second_parameter_set(field, value):
    values = dict(_config().__dict__)
    values[field] = value
    with pytest.raises(ValueError):
        PreLiftArmStiffnessRestorationConfig(**values)
