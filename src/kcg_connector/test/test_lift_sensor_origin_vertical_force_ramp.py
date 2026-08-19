import numpy as np
import pytest

from kcg_connector.d38999_tabletop_pick import iiwa14_grasp_tcp_transform
from kcg_connector.grasp.lift_sensor_origin_vertical_force_ramp import (
    load_lift_sensor_origin_vertical_force_ramp_config,
    sensor_origin_from_grasp_tcp_transform,
)
from kcg_connector.grasp.lift_task_space_vertical_force_ramp import (
    derive_vertical_force_step,
    numeric_tcp_translation_jacobian,
)


ARM = (
    -0.12951016426086426,
    0.4169462323188782,
    -0.40288954973220825,
    -1.120123028755188,
    0.15929490327835083,
    1.6363224983215332,
    -0.1071062907576561,
)


def enabled_document():
    return {
        "enabled": True,
        "threshold_label": "SIM_TUNING_ONLY_B_V2_H19",
        "source_h12_run_id": "B-V2-H12-ZERO-LIFT-01",
        "source_h13_run_id": "B-V2-GRASP-14",
        "source_h17_run_id": "B-V2-H17-ABSOLUTE-LOAD-HOLD-01",
        "source_h18_run_id": "B-V2-H18-VERTICAL-FORCE-RAMP-01",
        "force_target_source": "FROZEN_BODY_PLUS_NUT_WEIGHT",
        "ramp_steps_source": "H13_TRANSITION_STEPS",
        "world_force_axis": [0.0, 0.0, 1.0],
        "force_application_frame": "handbase_link_sensor_origin",
        "hard_gate_reference_frame": "handbase_link_sensor_origin",
        "handbase_to_tcp_source": "FROZEN_PICK_GEOMETRY_CANDIDATE",
        "keep_position_target_fixed_during_ramp": True,
        "immediately_preceding_arm_state_only": True,
        "recompute_mapping_during_staged_lift": True,
    }


def sensor_fk(arm):
    return sensor_origin_from_grasp_tcp_transform(
        arm, 0.400, iiwa14_grasp_tcp_transform
    )


def test_h19_is_strict_and_historical_absence_defaults_disabled():
    assert load_lift_sensor_origin_vertical_force_ramp_config(None).enabled is False
    config = load_lift_sensor_origin_vertical_force_ramp_config(
        enabled_document()
    )
    assert config.enabled is True
    assert config.force_application_frame == "handbase_link_sensor_origin"
    assert config.hard_gate_reference_frame == "handbase_link_sensor_origin"


@pytest.mark.parametrize(
    ("key", "value", "match"),
    (
        ("force_application_frame", "grasp_tcp", "handbase_link_sensor_origin"),
        ("hard_gate_reference_frame", "grasp_tcp", "handbase_link_sensor_origin"),
        ("world_force_axis", [0.0, 0.0, -1.0], "world \\+Z"),
        ("keep_position_target_fixed_during_ramp", False, "must remain true"),
    ),
)
def test_h19_semantic_boundaries_fail_closed(key, value, match):
    document = enabled_document()
    document[key] = value
    with pytest.raises(ValueError, match=match):
        load_lift_sensor_origin_vertical_force_ramp_config(document)


def test_sensor_origin_is_exactly_400_mm_behind_grasp_tcp():
    world_tcp = np.asarray(iiwa14_grasp_tcp_transform(ARM), dtype=np.float64)
    world_sensor = np.asarray(sensor_fk(ARM), dtype=np.float64)
    expected = np.eye(4, dtype=np.float64)
    expected[2, 3] = 0.400
    assert world_sensor @ expected == pytest.approx(world_tcp, abs=1.0e-12)
    assert world_sensor[:3, :3] == pytest.approx(world_tcp[:3, :3], abs=1.0e-12)


def test_h19_same_force_has_distinct_collocated_jacobian_and_virtual_work():
    tcp_jacobian = numeric_tcp_translation_jacobian(
        ARM, iiwa14_grasp_tcp_transform
    )
    sensor_jacobian = numeric_tcp_translation_jacobian(ARM, sensor_fk)
    assert np.max(np.abs(sensor_jacobian - tcp_jacobian)) > 0.10
    result = derive_vertical_force_step(
        ARM,
        (0.11, -0.07, 0.05, -0.03, 0.02, -0.01, 0.04),
        0.23,
        0.08,
        -9.81,
        8.0,
        1.0,
        0.0,
        (0.0,) * 7,
        (300.0,) * 7,
        sensor_fk,
    )
    assert result["world_up_force_n"] == pytest.approx(3.0411)
    assert result["virtual_work_residual_w"] < 1.0e-15
    assert result["maximum_abs_joint_effort_nm"] == pytest.approx(
        1.6518508257350906, abs=1.0e-9
    )
    assert result["maximum_abs_joint_effort_nm"] < 300.0
    assert result["object_truth_used"] is False
    assert result["contact_truth_used"] is False
    assert result["event_truth_used"] is False
    assert result["object_pose_written"] is False
