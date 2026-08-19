import pytest

from kcg_connector.grasp.pre_lift_vertical_force_xy_admittance import (
    load_pre_lift_vertical_force_xy_admittance_config,
)


def enabled_document():
    return {
        "enabled": True,
        "threshold_label": "SIM_TUNING_ONLY_B_V2_H20",
        "source_h14_run_id": "B-V2-GRASP-15-IFIX01",
        "source_h18_run_id": "B-V2-H18-VERTICAL-FORCE-RAMP-01",
        "reuse_h14_xy_parameters": True,
        "reuse_h18_force_profile": True,
        "immediately_preceding_task_wrench_only": True,
        "vertical_position_target_fixed": True,
        "sensor_origin_hard_gate_unchanged": True,
        "no_new_numeric_control_parameter": True,
    }


def test_h20_loads_strictly_and_absence_defaults_disabled():
    assert load_pre_lift_vertical_force_xy_admittance_config(None).enabled is False
    config = load_pre_lift_vertical_force_xy_admittance_config(enabled_document())
    assert config.enabled is True
    assert config.reuse_h14_xy_parameters is True
    assert config.reuse_h18_force_profile is True


@pytest.mark.parametrize(
    "key",
    (
        "reuse_h14_xy_parameters",
        "reuse_h18_force_profile",
        "immediately_preceding_task_wrench_only",
        "vertical_position_target_fixed",
        "sensor_origin_hard_gate_unchanged",
        "no_new_numeric_control_parameter",
    ),
)
def test_h20_truth_and_parameter_boundaries_fail_closed(key):
    document = enabled_document()
    document[key] = False
    with pytest.raises(ValueError, match="must remain true"):
        load_pre_lift_vertical_force_xy_admittance_config(document)


def test_h20_rejects_unknown_fields():
    document = enabled_document()
    document["new_gain"] = 1.0
    with pytest.raises(ValueError, match="unknown keys"):
        load_pre_lift_vertical_force_xy_admittance_config(document)
