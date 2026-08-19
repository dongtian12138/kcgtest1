import ast
import math
from pathlib import Path

import pytest

from kcg_connector.grasp.physical_grasp_config import (
    load_physical_grasp_experiment_config,
)
from kcg_connector.grasp.pre_lift_differential_finger_preload_correction import (
    FIXED_CORRECTION_CLOSURE_RAD,
    SOURCE_ANALYSIS_SHA256,
    apply_closure_correction_to_targets,
    correction_step,
    derive_fixed_correction_contract,
    load_differential_finger_preload_correction_config,
)


REPOSITORY = Path(__file__).resolve().parents[3]
CONFIG = (
    REPOSITORY
    / "src/kcg_connector/config/d38999_multilayer_tabletop_physical_grasp_v2.yaml"
)
RUNNER = REPOSITORY / "src/kcg_connector/isaac/d38999_tabletop_pick_smoke.py"


def enabled_document():
    return {
        "enabled": True,
        "threshold_label": "SIM_TUNING_ONLY_B_V2_H23_FIXED_CORRECTION",
        "source_run_id": (
            "B-V2-H23-DIFFERENTIAL-FINGER-PRELOAD-DIAGNOSTIC-IFIX01"
        ),
        "source_analysis_sha256": SOURCE_ANALYSIS_SHA256,
        "fixed_correction_closure_rad": list(FIXED_CORRECTION_CLOSURE_RAD),
        "correction_norm_limit_source": "SEQUENTIAL_PROBE_INCREMENT_RAD",
        "transition_steps_source": "H13_TRANSITION_STEPS",
        "stability_steps_source": "REFERENCE_WINDOW_STEPS",
        "zero_sum_required": True,
        "arm_target_fixed_during_correction": True,
        "vertical_feedforward_zero_required": True,
        "wrist_payload_reference_rebase_forbidden": True,
        "root_reference_refresh_from_trailing_stability_window": True,
        "persistent_via_fresh_h17_equilibrium": True,
        "raw_sensor_hard_gate_unchanged": True,
        "hard_gate_detection_delay_steps": 0,
        "object_contact_event_truth_forbidden": True,
        "second_parameter_set_allowed": False,
        "formal_b_pass_claimed_from_fix_window": False,
    }


def test_h23_fixed_correction_defaults_off_and_is_exact():
    assert load_differential_finger_preload_correction_config(None).enabled is False
    config = load_differential_finger_preload_correction_config(enabled_document())
    assert config.enabled is True
    assert config.source_analysis_sha256 == SOURCE_ANALYSIS_SHA256
    assert config.fixed_correction_closure_rad == pytest.approx(
        FIXED_CORRECTION_CLOSURE_RAD
    )
    assert sum(config.fixed_correction_closure_rad) == pytest.approx(0.0, abs=1e-15)
    assert math.sqrt(
        sum(value * value for value in config.fixed_correction_closure_rad)
    ) == pytest.approx(0.004)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    (
        ("source_analysis_sha256", "0" * 64, "source_analysis_sha256"),
        ("fixed_correction_closure_rad", [0.001, 0.0, -0.001], "single diagnostic"),
        ("hard_gate_detection_delay_steps", 1, "remain zero"),
        ("second_parameter_set_allowed", True, "second parameter"),
        ("formal_b_pass_claimed_from_fix_window", True, "formal B pass"),
    ),
)
def test_h23_fixed_correction_rejects_drift(key, value, message):
    document = enabled_document()
    document[key] = value
    with pytest.raises(ValueError, match=message):
        load_differential_finger_preload_correction_config(document)


def test_h23_contract_reuses_existing_steps_and_probe_norm():
    physical = load_physical_grasp_experiment_config(CONFIG)
    result = derive_fixed_correction_contract(
        physical.pre_lift_differential_finger_preload_correction,
        physical.sequential,
        transition_steps=physical.lift_phase_arm_damping.transition_steps,
        stability_steps=physical.reference_window_steps,
    )
    assert result["transition_steps"] == 240
    assert physical.reference_window_steps == 120
    assert result["stability_steps"] == physical.reference_window_steps
    assert result["correction_norm_rad"] == pytest.approx(0.004)
    assert result["correction_norm_limit_rad"] == pytest.approx(0.004)
    assert result["hard_gate_detection_delay_steps"] == 0
    assert result["second_parameter_set_allowed"] is False


def test_h23_correction_is_minimum_jerk_zero_sum_and_bounded():
    steps = [
        correction_step(index, 240, FIXED_CORRECTION_CLOSURE_RAD)
        for index in range(240)
    ]
    assert 0.0 < steps[0]["minimum_jerk_blend"] < 1.0
    assert steps[-1]["minimum_jerk_blend"] == pytest.approx(1.0)
    assert steps[-1]["applied_correction_closure_rad"] == pytest.approx(
        FIXED_CORRECTION_CLOSURE_RAD
    )
    assert all(
        abs(step["applied_correction_sum_rad"]) <= 1.0e-15
        for step in steps
    )
    overlay = apply_closure_correction_to_targets(
        [0.7489653960413496, 0.5854216140613789, 0.761541063422695],
        [0.0, 0.0, 0.0],
        [0.765, 0.595, 0.765],
        FIXED_CORRECTION_CLOSURE_RAD,
    )
    assert overlay["inside_frozen_open_closed_bounds"] is True
    assert overlay["corrected_targets_rad"] == pytest.approx(
        [0.7495704666819743, 0.5878985422036598, 0.7584590646397895]
    )


def _correction_runner_block():
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    blocks = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and "pre_lift_differential_finger_preload_correction.enabled"
        in ast.unparse(node.test)
        and "h23_fix_schedule" in ast.unparse(node)
    ]
    assert len(blocks) == 1
    return blocks[0]


def test_h23_runner_applies_one_fixed_correction_and_refreshes_h17():
    source = ast.unparse(_correction_runner_block())
    assert "derive_fixed_correction_contract(" in source
    assert "correction_step(" in source
    assert "apply_closure_correction_to_targets(" in source
    assert "observe_and_step(" in source
    assert "formal_lift_monitor.update(" in source
    assert "LiftFingerAbsoluteLoadHold(" in source
    assert "h23_fix_stability_root_samples" in source
    assert "formal_wrist_payload_reference" in source
    assert "'formal_b_pass_claimed': False" in source
    assert "'hard_gate_detection_delay_steps': 0" in source


def test_h23_runner_fixed_correction_has_no_object_or_contact_truth_control():
    source = ast.unparse(_correction_runner_block())
    for forbidden in (
        "contact_snapshot(",
        "get_world_pose(",
        "set_world_pose(",
        "get_contact_report(",
        "get_contact_normal(",
        "analyze_differential_probe(",
    ):
        assert forbidden not in source
    assert "'object_truth_used': False" in source
    assert "'contact_truth_used': False" in source
    assert "'event_truth_used': False" in source
    assert "'object_pose_written': False" in source
