'''Pure tests for the log-only terminal evaluator snapshot builder.'''

from __future__ import annotations

import math

import numpy as np
import pytest

from kcg_connector.grasp.terminal_evaluator import build_terminal_snapshot

IDENTITY = (1.0, 0.0, 0.0, 0.0)


def _snapshot(**overrides):
    values = {
        "reason": "formal_lift_gate_triggered",
        "global_step": 9722,
        "phase": "physical_grip_lift_stage_1",
        "plug_body_position_world": (0.43, -0.08, 0.25),
        "plug_body_orientation_world": IDENTITY,
        "nut_position_world": (0.43, -0.08, 0.257),
        "nut_orientation_world": IDENTITY,
        "hand_position_world": (0.43, -0.08, 0.60),
        "hand_orientation_world": IDENTITY,
        "settled_plug_position_world": (0.43, -0.08, 0.24),
        "settled_plug_orientation_world": IDENTITY,
        "settled_plug_z_m": 0.24,
        "lift_started_dz_m": 0.0005,
        "contact_audit": {
            "finger_body_group_records": {
                "f1": {"body": 2, "nut": 1},
                "f2": {"body": 3, "nut": 0},
                "f3": {"body": 1, "nut": 2},
            },
            "plug_table_records": 0,
        },
    }
    values.update(overrides)
    return build_terminal_snapshot(**values)


def test_snapshot_is_log_only_and_controller_terminal():
    snapshot = _snapshot()
    assert snapshot["posthoc_truth_evaluation_only"] is True
    assert snapshot["sampled_with_controller_terminal"] is True
    assert snapshot["controller_terminal"] is True
    assert snapshot["sinks"] == "report_log_only"
    assert snapshot["consumed_by"] == []


def test_t_hand_plug_is_derived_correctly():
    snapshot = _snapshot(
        hand_position_world=(0.0, 0.0, 0.0),
        plug_body_position_world=(0.1, 0.2, 0.3),
    )
    t_hand_plug = np.asarray(snapshot["t_hand_plug"])
    assert np.allclose(t_hand_plug[:3, 3], (0.1, 0.2, 0.3))
    assert np.allclose(t_hand_plug[:3, :3], np.eye(3))


def test_relative_settled_xyz_and_yaw():
    snapshot = _snapshot(
        plug_body_position_world=(0.53, -0.10, 0.245),
        settled_plug_position_world=(0.43, -0.08, 0.24),
    )
    assert snapshot["relative_settled_plug"]["xyz_m"] == pytest.approx(
        (0.1, -0.02, 0.005)
    )
    assert snapshot["relative_settled_plug"]["yaw_rad"] == pytest.approx(0.0)


def test_plug_lift_started_flag_uses_configured_threshold():
    assert _snapshot().get("plug_lift_started") is True
    resting = _snapshot(
        plug_body_position_world=(0.43, -0.08, 0.2401),
        settled_plug_z_m=0.24,
    )
    assert resting["plug_lift_started"] is False
    assert resting["plug_z_m"] == pytest.approx(0.2401)


def test_contact_audit_passthrough():
    audit = {"plug_table_records": 7, "robot_loose_plug_records": 3}
    snapshot = _snapshot(contact_audit=audit)
    assert snapshot["episode_terminal_contact_audit"] == audit


def test_quaternion_normalization_is_tolerated():
    snapshot = _snapshot(plug_body_orientation_world=(2.0, 0.0, 0.0, 0.0))
    orientation = snapshot["plug_body_pose_world"]["orientation_wxyz"]
    assert np.allclose(orientation, (1.0, 0.0, 0.0, 0.0))


def test_yaw_delta_wraps_to_pi_range():
    # 90 degrees around world Z.
    quarter = (math.cos(math.pi / 4.0), 0.0, 0.0, math.sin(math.pi / 4.0))
    snapshot = _snapshot(
        plug_body_orientation_world=quarter,
        settled_plug_orientation_world=IDENTITY,
    )
    assert snapshot["relative_settled_plug"]["yaw_rad"] == pytest.approx(
        math.pi / 2.0
    )


def test_snapshot_validates_inputs():
    with pytest.raises(ValueError, match="3-vector"):
        _snapshot(plug_body_position_world=(0.0, 0.0))
    with pytest.raises(ValueError, match="quaternion"):
        _snapshot(plug_body_orientation_world=(1.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="global_step"):
        _snapshot(global_step=1.5)
    with pytest.raises(ValueError, match="lift_started_dz_m"):
        _snapshot(lift_started_dz_m=-0.1)
    with pytest.raises(ValueError, match="contact_audit"):
        _snapshot(contact_audit=[1, 2, 3])
    with pytest.raises(ValueError, match="finite"):
        _snapshot(nut_position_world=(0.0, float("nan"), 0.0))
