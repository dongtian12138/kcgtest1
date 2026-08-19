import pytest

from kcg_connector.grasp.pre_lift_xy_nyquist_suppression import (
    load_pre_lift_xy_nyquist_suppression_config,
    two_sample_xy_control_input,
)


def enabled_document():
    return {
        "enabled": True,
        "threshold_label": "SIM_TUNING_ONLY_B_V2_H21",
        "source_h20_run_id": "B-V2-H20-PRELIFT-XY-ADMITTANCE-01",
        "two_sample_arithmetic_mean_required": True,
        "xy_control_input_only": True,
        "raw_sensor_hard_gate_unchanged": True,
        "raw_peaks_recorded": True,
        "hard_gate_detection_delay_steps": 0,
        "no_new_gain_force_or_geometry_parameter": True,
    }


def test_h21_defaults_disabled_and_loads_strict_enabled_contract():
    assert load_pre_lift_xy_nyquist_suppression_config(None).enabled is False
    config = load_pre_lift_xy_nyquist_suppression_config(enabled_document())
    assert config.enabled is True
    assert config.hard_gate_detection_delay_steps == 0


def test_two_sample_mean_exactly_rejects_nyquist_alternation():
    result = two_sample_xy_control_input((1.0, -2.0), (-1.0, 2.0))
    assert result["filtered_control_task_force_xy_n"] == pytest.approx((0.0, 0.0))
    assert result["nyquist_alternating_component_gain"] == 0.0
    assert result["hard_gate_detection_delay_steps"] == 0
    assert result["raw_peaks_recorded"] is True


def test_first_sample_duplicates_itself_without_zero_start_transient():
    result = two_sample_xy_control_input((0.4, -0.2), None)
    assert result["filtered_control_task_force_xy_n"] == pytest.approx((0.4, -0.2))
    assert result["initialized_by_current_sample_duplication"] is True
    assert result["oldest_control_sample_age_steps"] == 0


@pytest.mark.parametrize(
    "key",
    (
        "two_sample_arithmetic_mean_required",
        "xy_control_input_only",
        "raw_sensor_hard_gate_unchanged",
        "raw_peaks_recorded",
        "no_new_gain_force_or_geometry_parameter",
    ),
)
def test_h21_truth_and_parameter_boundaries_fail_closed(key):
    document = enabled_document()
    document[key] = False
    with pytest.raises(ValueError, match="must remain true"):
        load_pre_lift_xy_nyquist_suppression_config(document)


def test_h21_rejects_hard_gate_delay_unknown_fields_and_nonfinite_inputs():
    document = enabled_document()
    document["hard_gate_detection_delay_steps"] = 1
    with pytest.raises(ValueError, match="must remain zero"):
        load_pre_lift_xy_nyquist_suppression_config(document)
    document = enabled_document()
    document["gain"] = 0.5
    with pytest.raises(ValueError, match="unknown keys"):
        load_pre_lift_xy_nyquist_suppression_config(document)
    with pytest.raises(ValueError, match="finite"):
        two_sample_xy_control_input((float("nan"), 0.0), None)
