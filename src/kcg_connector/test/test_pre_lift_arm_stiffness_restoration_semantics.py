"""Static safety boundaries for the one H16 dynamic diagnostic path."""

from pathlib import Path


RUNNER = (
    Path(__file__).resolve().parents[1]
    / "isaac"
    / "d38999_tabletop_pick_smoke.py"
)


def _h16_region() -> str:
    source = RUNNER.read_text(encoding="utf-8")
    start = source.index("if stiffness_restore_diagnostic_mode:")
    end = source.index("lift_start_tcp = np.asarray(", start)
    return source[start:end]


def test_h16_is_characterization_only_and_never_claims_dynamic_pass():
    region = _h16_region()
    assert '"diagnostic_only": True' in region
    assert '"dynamic_pass_claimed": False' in region
    assert '"physical_lift_started": False' in region
    assert 'metrics["passed"] = False' in region
    assert "DIAGNOSTIC_ONLY episode is not a grasp PASS" in region


def test_h16_restores_the_existing_drive_and_reads_back_every_gain_step():
    region = _h16_region()
    guard = region.index("derive_runtime_target_bias_step_envelope(")
    transition = region.index(
        "for h16_step in range(restoration.transition_steps):"
    )
    assert guard < transition
    assert "derive_stiffness_restoration_step(" in region
    assert "restored_nominal_drive_target(" in region
    assert "controller.set_gains(" in region
    assert "controller.get_gains()" in region
    assert "maximum_gain_readback_error" in region
    assert "capture_position_preload_nm(" in region
    assert '"runtime_target_bias_step_envelope_rad"' in region
    assert '"absolute_schedule_target_bias_step_ceiling_rad"' in region
    assert '"feedforward_effort_used": False' in region


def test_h16_fixed_anchor_hold_uses_robot_dynamics_and_original_hard_gate():
    region = _h16_region()
    assert "fixed_anchor_target" in region
    assert "formal_lift_monitor.update(" in region
    assert "run_formal_failure_recovery()" in region
    assert '"sensor_origin_hard_gate_unchanged": True' in region
    assert "observe_and_step(" in region
    assert "formal_lift_traversed_arm_targets = []" in region


def test_h16_keeps_truth_and_pose_write_firewalls_explicit():
    region = _h16_region()
    assert '"object_truth_used": False' in region
    assert '"contact_truth_used": False' in region
    assert '"event_truth_used": False' in region
    assert '"object_pose_written": False' in region
    assert "contact_snapshot(" not in region
    assert "get_world_pose(" not in region
    assert "set_world_pose(" not in region


def test_h16_runs_after_h13_and_before_any_staged_lift():
    source = RUNNER.read_text(encoding="utf-8")
    h13 = source.index("if lift_phase_arm_damping.enabled:")
    h16 = source.index("if stiffness_restore_diagnostic_mode:", h13)
    staged = source.index(
        "for stage_index, lift_stage in enumerate(", h16
    )
    assert h13 < h16 < staged
