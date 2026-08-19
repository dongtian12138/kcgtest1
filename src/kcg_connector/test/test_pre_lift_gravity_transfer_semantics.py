"""Static safety boundaries for the one H15 dynamic diagnostic path."""

from pathlib import Path


RUNNER = (
    Path(__file__).resolve().parents[1]
    / "isaac"
    / "d38999_tabletop_pick_smoke.py"
)


def _h15_region() -> str:
    source = RUNNER.read_text(encoding="utf-8")
    start = source.index("if gravity_transfer_diagnostic_mode:")
    end = source.index("if (\n                physical_grasp.pre_lift_centering.enabled", start)
    return source[start:end]


def test_h15_is_characterization_only_and_never_claims_dynamic_pass():
    region = _h15_region()
    assert '"diagnostic_only": True' in region
    assert '"dynamic_pass_claimed": False' in region
    assert '"physical_lift_started": False' in region
    assert 'metrics["passed"] = False' in region
    assert "DIAGNOSTIC_ONLY episode is not a grasp PASS" in region


def test_h15_uses_robot_dynamics_and_reads_back_the_same_effort_action():
    region = _h15_region()
    assert "get_generalized_gravity_forces(" in region
    assert "robot.get_applied_joint_efforts(" in region
    assert "maximum_effort_readback_error_nm" in region
    assert "position_plus_effort_same_action" in region
    assert "arm_feedforward_effort_nm" in RUNNER.read_text(encoding="utf-8")


def test_h15_keeps_truth_and_pose_write_firewalls_explicit():
    region = _h15_region()
    assert '"object_truth_used": False' in region
    assert '"contact_truth_used": False' in region
    assert '"event_truth_used": False' in region
    assert '"object_pose_written": False' in region
    assert "contact_snapshot(" not in region
    assert "get_world_pose(" not in region
    assert "set_world_pose(" not in region


def test_h15_runs_before_any_staged_lift_and_uses_the_existing_hard_monitor():
    source = RUNNER.read_text(encoding="utf-8")
    h15 = source.index("if gravity_transfer_diagnostic_mode:")
    staged = source.index("# Staged mode below", h15)
    assert h15 < staged
    region = source[h15:staged]
    assert "formal_lift_monitor.update(" in region
    assert "run_formal_failure_recovery()" in region
    assert "sensor_origin_hard_gate_unchanged" in region
