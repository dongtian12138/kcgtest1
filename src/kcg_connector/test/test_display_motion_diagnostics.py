import numpy as np
import pytest

from kcg_connector.display_motion_diagnostics import (
    DisplayMotionRingBuffer,
    atomic_write_json,
    build_failure_report,
    evaluate_display_sensor_gates,
    evaluate_display_wrist_evidence,
    evaluate_waypoint_path_quality,
    joint_target_limit_violations,
    numeric_tcp_jacobian,
)


def _identity_fk(joints):
    matrix = np.eye(4)
    q = np.asarray(joints, dtype=np.float64).ravel()
    matrix[:3, 3] = np.array([q[0], q[1], q[2] + 0.4])
    return matrix


@pytest.mark.parametrize("phase", ["motion", "hold"])
def test_formal_moment_gate_triggers_at_point_three_plus_epsilon(phase):
    evidence = evaluate_display_wrist_evidence(
        current_wrench=[0.0, 0.0, 0.0, 0.0, 0.0, 0.3000001],
        reference_wrench=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        previous_raw_wrench=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ema_wrench=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    assert evidence["formal_gate_triggered"] is True
    assert evidence["moment_formal_triggered"] is True
    assert "formal_moment" in "|".join(evidence["triggered_gates"])
    # The same evidence object is used by both motion and hold branches.
    assert phase in ("motion", "hold")


def test_formal_gate_remains_when_ema_candidate_is_off():
    evidence = evaluate_display_wrist_evidence(
        current_wrench=[0.0, 0.0, 9.0, 0.0, 0.0, 0.0],
        reference_wrench=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        previous_raw_wrench=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ema_wrench=[0.0, 0.0, 9.0, 0.0, 0.0, 0.0],
        ema_force_limit_n=999.0,
        ema_moment_limit_nm=999.0,
    )
    assert evidence["force_formal_triggered"] is True
    assert evidence["ema_candidate_triggered"] is False
    assert evidence["formal_gate_triggered"] is True


def test_constant_step_is_ema_residual_not_rate():
    current = [0.1, 0.2, 0.3, 0.01, 0.02, 0.03]
    evidence = evaluate_display_wrist_evidence(
        current_wrench=current,
        reference_wrench=[0.0] * 6,
        previous_raw_wrench=current,
        ema_wrench=current,
    )
    assert "rate" not in str(evidence)
    assert evidence["adjacent_raw_force_delta_n"] == 0.0
    assert evidence["ema_force_residual_n"] == 0.0
    assert evidence["ema_force_residual_n"] is not None


def test_ring_buffer_and_failure_evidence_are_written(tmp_path):
    buffer = DisplayMotionRingBuffer(capacity=3)
    for index in range(4):
        buffer.append({"global_step": index, "phase": "motion"})
    trace_path = tmp_path / "motion_trace.jsonl"
    buffer.write_jsonl(trace_path)
    lines = trace_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert "global_step" in lines[-1]
    report = build_failure_report(
        error="formal_moment:perpendicular",
        status="DISPLAY_SAFETY_FAIL_CLOSED",
        trace_records=buffer.records(),
        path_quality_records=[{"view_id": "F1", "reject": False}],
    )
    report_path = tmp_path / "failure_report.json"
    atomic_write_json(report_path, report)
    loaded = __import__("json").loads(report_path.read_text())
    assert loaded["status"] == "DISPLAY_SAFETY_FAIL_CLOSED"
    assert loaded["control_authorized"] is False
    assert loaded["formal_estimator_input"] is False


def test_failure_report_never_authorizes_control():
    report = build_failure_report(
        error="test",
        status="TEST",
        trace_records=[],
        path_quality_records=[],
    )
    assert report["control_authorized"] is False
    assert report["formal_estimator_input"] is False


def test_path_quality_rejects_branch_jump_before_motion():
    start = np.zeros(7)
    waypoints = [start + np.array([0.2, 0, 0, 0, 0, 0, 0])]
    report = evaluate_waypoint_path_quality(
        waypoints,
        forward_kinematics=_identity_fk,
        physics_rate_hz=240.0,
        steps_per_waypoint=60,
        start_q=start,
        table_top_z_m=0.2,
        fixture_center_m=(0.55, 0.185, 0.22),
        fixture_half_extent_m=(0.07, 0.07, 0.02),
        max_abs_dq_per_waypoint=0.12,
    )
    assert report["reject"] is True
    assert any("_dq=" in reason for reason in report["reasons"])


def test_path_quality_rejects_high_qdd_estimate_before_motion():
    start = np.zeros(7)
    # Two waypoints with increasing velocity produce a large qdd estimate.
    waypoints = [
        start + np.array([0.001, 0, 0, 0, 0, 0, 0]),
        start + np.array([0.020, 0, 0, 0, 0, 0, 0]),
    ]
    report = evaluate_waypoint_path_quality(
        waypoints,
        forward_kinematics=_identity_fk,
        physics_rate_hz=240.0,
        steps_per_waypoint=1,
        start_q=start,
        table_top_z_m=0.2,
        fixture_center_m=(0.55, 0.185, 0.22),
        fixture_half_extent_m=(0.07, 0.07, 0.02),
        max_abs_qdd_est_rad_s2=1.0,
    )
    assert report["reject"] is True
    assert report["peak_abs_qdd_est_rad_s2"] > 1.0


def test_numeric_tcp_jacobian_shape():
    jacobian = numeric_tcp_jacobian(np.zeros(7), _identity_fk)
    assert jacobian.shape == (6, 6)
    assert np.all(np.isfinite(jacobian))


def test_path_quality_rejects_ill_conditioned_jacobian_before_motion():
    def rank_deficient_fk(joints):
        q = np.asarray(joints, dtype=np.float64).ravel()
        matrix = np.eye(4)
        # Joints 0 and 1 move the TCP identically, so the Jacobian is rank
        # deficient at every waypoint.
        matrix[:3, 3] = [q[0] + q[1], 0.0, 0.4]
        return matrix

    start = np.zeros(7)
    waypoints = [start + np.array([0.001, 0.001, 0, 0, 0, 0, 0])]
    report = evaluate_waypoint_path_quality(
        waypoints,
        forward_kinematics=rank_deficient_fk,
        physics_rate_hz=240.0,
        steps_per_waypoint=60,
        start_q=start,
        table_top_z_m=0.2,
        fixture_center_m=(0.55, 0.185, 0.22),
        fixture_half_extent_m=(0.07, 0.07, 0.02),
        min_jacobian_singular_value=0.02,
    )
    assert report["reject"] is True
    assert any("jacobian_smin" in reason for reason in report["reasons"])


JOINT_LIMITS = [
    (-2.967, 2.967),
    (-2.094, 2.094),
    (-2.967, 2.967),
    (-2.094, 2.094),
    (-2.967, 2.967),
    (-2.0943951024, 2.0943951024),
    (-2.967, 2.967),
]


def test_q6_target_above_limit_is_rejected_before_motion():
    q = np.zeros(7)
    q[5] = 2.1517
    violations = joint_target_limit_violations(
        q, JOINT_LIMITS, margin_rad=0.010
    )
    assert any(v["joint_index"] == 5 for v in violations)

    report = evaluate_waypoint_path_quality(
        [q],
        forward_kinematics=_identity_fk,
        physics_rate_hz=240.0,
        steps_per_waypoint=60,
        start_q=np.zeros(7),
        table_top_z_m=0.2,
        fixture_center_m=(0.55, 0.185, 0.22),
        fixture_half_extent_m=(0.07, 0.07, 0.02),
        joint_limits=JOINT_LIMITS,
        joint_limit_margin_rad=0.010,
    )
    assert report["reject"] is True
    assert any("joint_limit_margin" in r for r in report["reasons"])


def test_q6_target_inside_limit_margin_passes():
    q = np.zeros(7)
    q[5] = 2.0943951024 - 0.010 - 0.001
    violations = joint_target_limit_violations(
        q, JOINT_LIMITS, margin_rad=0.010
    )
    assert violations == []


def test_joint_saturation_with_increasing_target_triggers_tracking_gate():
    desired = np.zeros(7)
    desired[5] = 2.1517
    actual = np.zeros(7)
    actual[5] = 2.0943951024
    gates = evaluate_display_sensor_gates(
        desired_arm_q=desired,
        actual_q=actual,
        velocities=np.zeros(7),
        torque=np.zeros(3),
        joint_limits=JOINT_LIMITS,
        joint_limit_margin_rad=0.010,
        max_arm_tracking_error_rad=0.030,
    )
    assert gates["ok"] is False
    assert "joint_target_limit_margin" in gates["reasons"]
    assert "arm_tracking_limit" in gates["reasons"]
    assert gates["arm_tracking_error_rad"] > 0.030
