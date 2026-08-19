import ast
from pathlib import Path

import pytest

from kcg_connector.grasp.pre_lift_quasistatic_lateral_preload_nulling import (
    load_pre_lift_quasistatic_lateral_preload_nulling_config,
)


REPOSITORY = Path(__file__).resolve().parents[3]
RUNNER = REPOSITORY / "src/kcg_connector/isaac/d38999_tabletop_pick_smoke.py"


def enabled_document():
    return {
        "enabled": True,
        "threshold_label": "SIM_TUNING_ONLY_B_V2_H22",
        "source_h21_run_id": "B-V2-H21-NYQUIST-SUPPRESSION-01",
        "steps_source": "H13_TRANSITION_STEPS",
        "zero_vertical_feedforward_required": True,
        "reuse_h14_xy_parameters": True,
        "reuse_h21_two_sample_input": True,
        "retain_correction_into_vertical_ramp": True,
        "correction_total_bound_reset_forbidden": True,
        "payload_reference_rebase_forbidden": True,
        "raw_sensor_hard_gate_unchanged": True,
        "hard_gate_detection_delay_steps": 0,
        "no_new_gain_force_or_geometry_parameter": True,
    }


def test_h22_defaults_disabled_and_loads_strict_enabled_contract():
    default = load_pre_lift_quasistatic_lateral_preload_nulling_config(None)
    assert default.enabled is False
    config = load_pre_lift_quasistatic_lateral_preload_nulling_config(
        enabled_document()
    )
    assert config.enabled is True
    assert config.steps_source == "H13_TRANSITION_STEPS"
    assert config.hard_gate_detection_delay_steps == 0


@pytest.mark.parametrize(
    "key",
    (
        "zero_vertical_feedforward_required",
        "reuse_h14_xy_parameters",
        "reuse_h21_two_sample_input",
        "retain_correction_into_vertical_ramp",
        "correction_total_bound_reset_forbidden",
        "payload_reference_rebase_forbidden",
        "raw_sensor_hard_gate_unchanged",
        "no_new_gain_force_or_geometry_parameter",
    ),
)
def test_h22_authorization_and_safety_boundaries_fail_closed(key):
    document = enabled_document()
    document[key] = False
    with pytest.raises(ValueError, match="must remain true"):
        load_pre_lift_quasistatic_lateral_preload_nulling_config(document)


def test_h22_rejects_new_numeric_parameter_and_hard_gate_delay():
    document = enabled_document()
    document["gain"] = 0.5
    with pytest.raises(ValueError, match="unknown keys"):
        load_pre_lift_quasistatic_lateral_preload_nulling_config(document)

    document = enabled_document()
    document["hard_gate_detection_delay_steps"] = 1
    with pytest.raises(ValueError, match="must remain zero"):
        load_pre_lift_quasistatic_lateral_preload_nulling_config(document)


@pytest.mark.parametrize(
    ("key", "value"),
    (
        ("threshold_label", "SIM_TUNING_ONLY_B_V2_H23"),
        ("source_h21_run_id", "another-run"),
        ("steps_source", "NEW_STEPS"),
    ),
)
def test_h22_rejects_changed_lineage_or_duration_source(key, value):
    document = enabled_document()
    document[key] = value
    with pytest.raises(ValueError, match="must remain"):
        load_pre_lift_quasistatic_lateral_preload_nulling_config(document)


def test_h22_runner_reuses_frozen_chain_and_h13_duration():
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    h22_blocks = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and "pre_lift_quasistatic_lateral_preload_nulling.enabled"
        in ast.unparse(node.test)
        and any(
            isinstance(child, ast.For)
            and "h22_step_index" in ast.unparse(child.target)
            for child in node.body
        )
    ]
    assert len(h22_blocks) == 1
    block = h22_blocks[0]
    block_source = ast.unparse(block)
    assert "range(lift_phase_arm_damping.transition_steps)" in block_source
    assert "h24_h17_control_input(root_delta" in block_source
    assert (
        "lift_finger_absolute_controller.update(h17_control_input)"
        in block_source
    )
    assert "formal_lift_monitor.update(root_delta" in block_source
    assert "two_sample_xy_control_input(" in block_source
    assert "derive_lift_xy_force_admittance_step(" in block_source
    assert "formal_lift_monitor.update(" in block_source
    assert "observe_and_step(" in block_source


def test_h22_runner_does_not_rebase_reference_or_assign_vertical_force():
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    block = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and "pre_lift_quasistatic_lateral_preload_nulling.enabled"
        in ast.unparse(node.test)
        and any(
            isinstance(child, ast.For)
            and "h22_step_index" in ast.unparse(child.target)
            for child in node.body
        )
    )
    assigned_names = {
        target.id
        for node in ast.walk(block)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (
            node.targets if isinstance(node, ast.Assign) else (node.target,)
        )
        if isinstance(target, ast.Name)
    }
    assert "formal_wrist_payload_reference" not in assigned_names
    assert "formal_arm_feedforward_effort" not in assigned_names
    block_source = ast.unparse(block)
    for forbidden in (
        "contact_snapshot(",
        "get_world_pose(",
        "set_world_pose(",
        "get_contact_report(",
        "get_contact_normal(",
    ):
        assert forbidden not in block_source
    assert "'vertical_feedforward_zero': True" in block_source
    assert "'payload_reference_rebased': False" in block_source
    assert "'correction_total_bound_reset': False" in block_source
    assert "'object_truth_used': False" in block_source
    assert "'contact_truth_used': False" in block_source
    assert "'event_truth_used': False" in block_source
    assert "'object_pose_written': False" in block_source
