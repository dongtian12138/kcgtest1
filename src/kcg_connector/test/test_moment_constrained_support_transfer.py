from pathlib import Path

import pytest
import yaml

from kcg_connector.grasp.moment_constrained_support_transfer import (
    MomentConstrainedSupportTransfer,
    PHASE_BREAKAWAY_COMMIT,
    PHASE_BUMPLESS_CONTROL_TRANSFER,
    PHASE_INTERNAL_FORCE_CENTERING,
    PHASE_LIFT_READY,
    PHASE_QUASISTATIC_UNWEIGHT,
    load_moment_constrained_support_transfer_config,
    verify_evidence_bindings,
)


REPOSITORY = Path(__file__).resolve().parents[3]
CONFIG = (
    REPOSITORY
    / "src/kcg_connector/config/d38999_moment_constrained_support_transfer_v1.yaml"
)


def _controller():
    return MomentConstrainedSupportTransfer(
        load_moment_constrained_support_transfer_config(CONFIG),
        base_targets_rad=(
            0.7495704666819744,
            0.5878985422036598,
            0.7584590646397895,
        ),
        open_targets_rad=(0.0, 0.0, 0.0),
        closed_targets_rad=(0.765, 0.595, 0.765),
    )


def _safe_step(controller, step):
    return controller.update(
        task_force_xy_n=(0.0, 0.0),
        sensor_origin_moment_increment_xyz_nm=(0.0, 0.0, 0.0),
        finger_root_torque_nm=(0.23, 0.23, 0.23),
        raw_moment_gate_score_nm=0.0,
        raw_hard_gate_triggered=False,
        input_global_step=step,
    )


def test_strict_config_and_all_evidence_hashes_are_bound():
    config = load_moment_constrained_support_transfer_config(CONFIG)
    assert config.enabled is True
    assert config.support_force_target_n == pytest.approx(3.0411)
    assert config.support_profile_samples == 240
    assert config.stable_confirm_steps == 48
    assert config.support_advance_moment_score_limit_nm == pytest.approx(
        0.30 - 0.2690785287299935
    )
    results = verify_evidence_bindings(config, REPOSITORY)
    assert len(results) == 4
    assert all(result["verified"] for result in results)


@pytest.mark.parametrize(
    ("section", "key", "value", "match"),
    (
        ("safety", "moment_component_gate_nm", 0.31, "0.30"),
        ("safety", "force_component_gate_n", 8.1, "8 N"),
        ("safety", "hard_gate_detection_delay_steps", 1, "delay"),
        ("truth_firewall", "object_pose_write_forbidden", False, "must remain true"),
        ("internal_force", "zero_sum_required", False, "must remain true"),
        ("support_transfer", "support_profile_samples", 239, "240-sample"),
    ),
)
def test_contract_changes_fail_closed(tmp_path, section, key, value, match):
    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    document[section][key] = value
    candidate = tmp_path / "candidate.yaml"
    candidate.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_moment_constrained_support_transfer_config(candidate)


def test_internal_force_step_is_zero_sum_rate_bounded_and_truth_free():
    controller = _controller()
    result = controller.update(
        task_force_xy_n=(0.8, -0.3),
        sensor_origin_moment_increment_xyz_nm=(0.20, 0.12, 0.0),
        finger_root_torque_nm=(0.23, 0.24, 0.22),
        raw_moment_gate_score_nm=0.22,
        raw_hard_gate_triggered=False,
        input_global_step=10,
    )
    assert result["phase_before"] == PHASE_INTERNAL_FORCE_CENTERING
    assert sum(result["applied_delta_closure_rad"]) == pytest.approx(0.0, abs=1e-14)
    assert max(abs(value) for value in result["applied_delta_closure_rad"]) <= (
        0.004 / 48.0 + 1.0e-15
    )
    assert result["support_profile_fraction"] == 0.0
    assert result["raw_sensor_hard_gate_unchanged"] is True
    assert result["hard_gate_detection_delay_steps"] == 0
    assert result["object_truth_used"] is False
    assert result["contact_truth_used"] is False
    assert result["contact_normal_used"] is False
    assert result["event_truth_used"] is False
    assert result["object_pose_written"] is False


def test_support_does_not_advance_without_derived_moment_margin():
    controller = _controller()
    for step in range(200):
        result = controller.update(
            task_force_xy_n=(0.0, 0.0),
            sensor_origin_moment_increment_xyz_nm=(0.0, 0.0, 0.0),
            finger_root_torque_nm=(0.23, 0.23, 0.23),
            raw_moment_gate_score_nm=0.031,
            raw_hard_gate_triggered=False,
            input_global_step=step,
        )
    assert result["phase_after"] == PHASE_INTERNAL_FORCE_CENTERING
    assert result["support_profile_index"] == 0


def test_raw_hard_gate_fails_without_filter_or_delay():
    controller = _controller()
    with pytest.raises(RuntimeError, match="hard gate"):
        controller.update(
            task_force_xy_n=(0.0, 0.0),
            sensor_origin_moment_increment_xyz_nm=(0.31, 0.0, 0.0),
            finger_root_torque_nm=(0.23, 0.23, 0.23),
            raw_moment_gate_score_nm=0.3000001,
            raw_hard_gate_triggered=True,
            input_global_step=1,
        )


def test_complete_state_machine_uses_240_quasistatic_support_levels():
    controller = _controller()
    step = 0
    result = None
    for _ in range(48):
        result = _safe_step(controller, step)
        step += 1
    assert result["phase_after"] == PHASE_BUMPLESS_CONTROL_TRANSFER
    result = _safe_step(controller, step)
    step += 1
    assert result["phase_after"] == PHASE_QUASISTATIC_UNWEIGHT
    previous_fraction = 0.0
    while controller.phase == PHASE_QUASISTATIC_UNWEIGHT:
        result = _safe_step(controller, step)
        step += 1
        assert result["support_profile_fraction"] >= previous_fraction
        previous_fraction = result["support_profile_fraction"]
    assert controller.phase == PHASE_BREAKAWAY_COMMIT
    assert controller.support_profile_index == 240
    assert controller.support_advance_count == 240
    for _ in range(48):
        result = _safe_step(controller, step)
        step += 1
    assert result["phase_after"] == PHASE_LIFT_READY
    assert result["commit_lift"] is True
    summary = controller.summary()
    assert summary["maximum_abs_target_step_rad"] <= 0.004 / 48.0 + 1.0e-15
    assert summary["maximum_abs_cumulative_rad"] <= 0.030 + 1.0e-15
    assert summary["maximum_zero_sum_residual_rad"] <= 1.0e-14


def test_root_load_loss_blocks_all_phase_progress():
    controller = _controller()
    for step in range(100):
        result = controller.update(
            task_force_xy_n=(0.0, 0.0),
            sensor_origin_moment_increment_xyz_nm=(0.0, 0.0, 0.0),
            finger_root_torque_nm=(0.23, 0.23, 0.01),
            raw_moment_gate_score_nm=0.0,
            raw_hard_gate_triggered=False,
            input_global_step=step,
        )
    assert result["phase_after"] == PHASE_INTERNAL_FORCE_CENTERING
    assert result["advance_safe"] is False
    assert result["support_profile_index"] == 0
