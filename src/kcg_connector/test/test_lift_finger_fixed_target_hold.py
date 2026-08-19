from pathlib import Path

import pytest
import yaml

from kcg_connector.grasp.lift_finger_fixed_target_hold import (
    LiftFingerFixedTargetHold,
    SOURCE_DERIVATION_SHA256,
    load_lift_finger_fixed_target_hold_config,
)
from kcg_connector.grasp.physical_grasp_config import (
    load_physical_grasp_experiment_config,
)


REPOSITORY = Path(__file__).resolve().parents[3]
CONFIG = (
    REPOSITORY
    / "src/kcg_connector/config/d38999_multilayer_tabletop_physical_grasp_v2.yaml"
)
RUNNER = REPOSITORY / "src/kcg_connector/isaac/d38999_tabletop_pick_smoke.py"
DERIVATION = (
    REPOSITORY
    / "artifacts/agent_control/tasks/"
    "D38999-AUTONOMOUS-DYNAMIC-CLOSEOUT-V2/"
    "DYN-B-GRASP-LIFT-RECOVERY-V2/"
    "B_H25_FIXED_POST_H23_FINGER_TARGET_HOLD_DERIVATION.json"
)
FROZEN_TARGETS = (
    0.7495674840655498,
    0.5879039328849862,
    0.7584564545523949,
)


def _section():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))[
        "lift_finger_fixed_target_hold"
    ]


def test_h25_section_is_strict_enabled_and_derivation_bound():
    section = _section()
    config = load_lift_finger_fixed_target_hold_config(section)
    physical = load_physical_grasp_experiment_config(CONFIG)
    assert config.enabled is True
    assert physical.lift_finger_fixed_target_hold == config
    assert config.source_derivation_sha256 == SOURCE_DERIVATION_SHA256
    assert DERIVATION.is_file()
    import hashlib

    assert hashlib.sha256(DERIVATION.read_bytes()).hexdigest() == (
        SOURCE_DERIVATION_SHA256
    )
    assert not any(
        key in section
        for key in (
            "target_rad",
            "gain",
            "force_n",
            "geometry",
            "coefficient",
        )
    )


def test_h25_holds_exact_targets_and_records_raw_load_without_feedback():
    controller = LiftFingerFixedTargetHold(
        load_lift_finger_fixed_target_hold_config(_section()),
        FROZEN_TARGETS,
    )
    result = controller.update(
        FROZEN_TARGETS,
        (0.24, 0.25, 0.23),
        100,
        "pre_lift_h25_fixed_finger_targets_vertical_force_ramp",
        0,
    )
    assert result["output_targets_rad"] == list(FROZEN_TARGETS)
    assert result["raw_finger_root_torque_nm"] == pytest.approx(
        (0.24, 0.25, 0.23)
    )
    assert result["maximum_abs_target_error_rad"] == 0.0
    assert result["h17_update_executed"] is False
    assert result["finger_target_modified"] is False
    assert result["raw_sensor_hard_gate_unchanged"] is True
    assert result["hard_gate_detection_delay_steps"] == 0
    assert result["object_truth_used"] is False
    assert result["contact_truth_used"] is False
    assert result["event_truth_used"] is False
    assert controller.summary()["h17_updates_executed"] == 0


def test_h25_rejects_target_drift_unknown_fields_and_weakened_boundaries():
    controller = LiftFingerFixedTargetHold(
        load_lift_finger_fixed_target_hold_config(_section()),
        FROZEN_TARGETS,
    )
    with pytest.raises(ValueError, match="drifted"):
        controller.update(
            (FROZEN_TARGETS[0] + 1.0e-9, *FROZEN_TARGETS[1:]),
            (0.24, 0.25, 0.23),
            100,
            "ramp",
            0,
        )
    section = _section()
    section["gain"] = 1.0
    with pytest.raises(ValueError, match="unknown keys"):
        load_lift_finger_fixed_target_hold_config(section)
    section = _section()
    section["raw_sensor_hard_gate_unchanged"] = False
    with pytest.raises(ValueError, match="must remain true"):
        load_lift_finger_fixed_target_hold_config(section)
    section = _section()
    section["formal_b_pass_claimed_from_hold"] = True
    with pytest.raises(ValueError, match="cannot claim formal B pass"):
        load_lift_finger_fixed_target_hold_config(section)


def test_h25_runner_bypasses_h17_only_after_h23_and_keeps_raw_hard_gate():
    source = RUNNER.read_text(encoding="utf-8")
    assert source.count("LiftFingerFixedTargetHold(") == 1
    assert source.count("lift_finger_fixed_target_controller.update(") == 2
    assert "h25_controller_not_activated" in source
    assert "h25_target_handoff_mismatch" in source
    assert "SUPERSEDED_BY_H25_AFTER_H23_STABILITY_WINDOW" in source
    assert '"raw_hard_gate_sample_filtered": False' in source
    assert '"raw_hard_gate_detection_delay_steps": 0' in source
    assert '"h18_start_finger_targets_rad"' in source
    assert '"h17_target_update_call_count_after_activation"' in source
    assert '"h17_cumulative_closure_after_step_rad"' in source
    assert '"h17_cumulative_closure_change_after_activation_rad"' in source
    assert "formal_lift_monitor.update(\n                            root_delta," in source
    assert '"object_truth_used": False' in source
    assert '"contact_truth_used": False' in source
    assert '"event_truth_used": False' in source
    assert '"post_physics_object_pose_writes": 0' in source
