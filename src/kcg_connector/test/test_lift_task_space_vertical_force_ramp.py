import numpy as np
import pytest

from kcg_connector.d38999_tabletop_pick import iiwa14_grasp_tcp_transform
from kcg_connector.grasp.lift_task_space_vertical_force_ramp import (
    NUMERIC_JACOBIAN_STEP_RAD,
    derive_vertical_force_step,
    frozen_payload_weight_force_n,
    load_lift_task_space_vertical_force_ramp_config,
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
        "threshold_label": "SIM_TUNING_ONLY_B_V2_H18",
        "source_h12_run_id": "B-V2-H12-ZERO-LIFT-01",
        "source_h13_run_id": "B-V2-GRASP-14",
        "source_h17_run_id": "B-V2-H17-ABSOLUTE-LOAD-HOLD-01",
        "force_target_source": "FROZEN_BODY_PLUS_NUT_WEIGHT",
        "ramp_steps_source": "H13_TRANSITION_STEPS",
        "world_force_axis": [0.0, 0.0, 1.0],
        "keep_position_target_fixed_during_ramp": True,
        "immediately_preceding_arm_state_only": True,
        "recompute_mapping_during_staged_lift": True,
    }


def test_historical_absence_is_disabled_and_strict_h18_loads():
    assert load_lift_task_space_vertical_force_ramp_config(None).enabled is False
    config = load_lift_task_space_vertical_force_ramp_config(enabled_document())
    assert config.enabled is True
    assert config.world_force_axis == (0.0, 0.0, 1.0)


@pytest.mark.parametrize(
    ("key", "value", "match"),
    (
        ("world_force_axis", [0.0, 0.0, -1.0], "world \\+Z"),
        ("force_target_source", "TUNED", "FROZEN_BODY_PLUS_NUT_WEIGHT"),
        ("keep_position_target_fixed_during_ramp", False, "must remain true"),
        ("immediately_preceding_arm_state_only", False, "must remain true"),
        ("recompute_mapping_during_staged_lift", False, "must remain true"),
    ),
)
def test_h18_semantic_boundaries_fail_closed(key, value, match):
    document = enabled_document()
    document[key] = value
    with pytest.raises(ValueError, match=match):
        load_lift_task_space_vertical_force_ramp_config(document)


def test_frozen_payload_weight_is_the_only_force_target_and_below_gate():
    assert frozen_payload_weight_force_n(0.23, 0.08, -9.81, 8.0) == pytest.approx(
        3.0411
    )
    with pytest.raises(ValueError, match="below the force gate"):
        frozen_payload_weight_force_n(0.50, 0.50, -9.81, 8.0)


def test_full_seven_joint_translation_jacobian_matches_h17_derivation():
    jacobian = numeric_tcp_translation_jacobian(
        ARM, iiwa14_grasp_tcp_transform
    )
    assert jacobian.shape == (3, 7)
    assert NUMERIC_JACOBIAN_STEP_RAD == pytest.approx(1.0e-6)
    assert np.linalg.svd(jacobian, compute_uv=False) == pytest.approx(
        (0.9498911501556699, 0.8684469601120927, 0.4836586909506691),
        abs=1.0e-9,
    )


def test_h18_full_force_mapping_matches_derivation_and_virtual_work():
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
        iiwa14_grasp_tcp_transform,
    )
    assert result["world_up_force_n"] == pytest.approx(3.0411)
    assert result["joint_effort_nm"] == pytest.approx(
        (
            0.0,
            -1.648713792160933,
            -0.1735609330874882,
            1.1945069134406017,
            -0.0007632119367995127,
            0.004121026360684454,
            0.0,
        ),
        abs=1.0e-9,
    )
    assert result["virtual_work_residual_w"] < 1.0e-15
    assert result["object_truth_used"] is False
    assert result["contact_truth_used"] is False
    assert result["object_pose_written"] is False


def test_h18_ramp_is_monotonic_and_actuator_limit_fails_closed():
    first = derive_vertical_force_step(
        ARM,
        (0.0,) * 7,
        0.23,
        0.08,
        -9.81,
        8.0,
        1.0 / 240.0,
        0.0,
        (0.0,) * 7,
        (300.0,) * 7,
        iiwa14_grasp_tcp_transform,
    )
    assert 0.0 < first["world_up_force_n"] < 3.0411
    with pytest.raises(ValueError, match="exceeds actuator"):
        derive_vertical_force_step(
            ARM,
            (0.0,) * 7,
            0.23,
            0.08,
            -9.81,
            8.0,
            1.0,
            0.0,
            (0.0,) * 7,
            (0.1,) * 7,
            iiwa14_grasp_tcp_transform,
        )
