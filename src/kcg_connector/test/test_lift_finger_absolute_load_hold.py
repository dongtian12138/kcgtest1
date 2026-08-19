from pathlib import Path

import pytest
import yaml

from kcg_connector.grasp.lift_finger_absolute_load_hold import (
    LiftFingerAbsoluteLoadHold,
    load_lift_finger_absolute_load_hold_config,
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
REFERENCE_NM = (
    0.23768424335867194,
    0.24899487116684516,
    0.24385290443897226,
)


def _controller():
    physical = load_physical_grasp_experiment_config(CONFIG)
    return LiftFingerAbsoluteLoadHold(
        physical.lift_finger_absolute_load_hold,
        physical.sequential,
        base_targets_rad=(
            0.7502400231904544,
            0.5853430799499983,
            0.761291896859542,
        ),
        open_targets_rad=(0.0, 0.0, 0.0),
        closed_targets_rad=(0.765, 0.595, 0.765),
        initial_balance_total_rad=(
            -0.008968310142879386,
            0.01063474661667164,
            -0.0016664364737922581,
        ),
        reference_root_torque_nm=REFERENCE_NM,
    )


def test_h17_section_reuses_existing_numeric_parameters_only():
    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    section = document["lift_finger_absolute_load_hold"]
    assert section == {
        "enabled": True,
        "threshold_label": "SIM_TUNING_ONLY_B_V2_H17",
        "source_h11_run_id": "B-V2-GRASP-13",
        "source_h16_run_id": "B-V2-H16-STIFFNESS-RESTORE-IFIX01",
        "reference_window_source": (
            "TRAILING_H13_TRANSITION_USING_EXISTING_REFERENCE_WINDOW_STEPS"
        ),
        "reuse_sequential_control_parameters": True,
        "independent_absolute_root_load_hold": True,
        "immediately_preceding_root_torque_only": True,
    }
    config = load_lift_finger_absolute_load_hold_config(section)
    assert config.enabled is True
    assert not any(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in section.values()
    )


def test_h17_changes_common_closure_to_restore_all_underloaded_fingers():
    step = _controller().update((0.210, 0.220, 0.215))
    applied = step["applied_delta_closure_rad"]
    assert all(value > 0.0 for value in applied)
    assert step["common_mode_applied_delta_closure_rad"] > 0.0
    assert step["mean_closure_target_allowed_to_change"] is True
    assert max(abs(value) for value in applied) <= 0.001 + 1.0e-15
    assert step["object_truth_used"] is False
    assert step["contact_truth_used"] is False
    assert step["event_truth_used"] is False


def test_h17_opens_all_overloaded_fingers_and_respects_existing_total_bounds():
    controller = _controller()
    first = controller.update((0.270, 0.280, 0.275))
    assert all(value < 0.0 for value in first["applied_delta_closure_rad"])
    last = first
    for _ in range(500):
        last = controller.update((0.050, 0.050, 0.050))
    assert all(
        abs(value) <= 0.030 + 1.0e-15
        for value in last["combined_balance_total_rad"]
    )
    assert all(
        low - 1.0e-15 <= target <= high + 1.0e-15
        for target, low, high in zip(
            last["output_targets_rad"],
            (0.0, 0.0, 0.0),
            (0.765, 0.595, 0.765),
        )
    )
    summary = controller.summary()
    assert summary["maximum_abs_applied_step_rad"] <= 0.001 + 1.0e-15
    assert summary["mean_closure_target_allowed_to_change"] is True


def test_h17_rejects_nonfinite_sensor_or_reference_input():
    with pytest.raises(ValueError, match="finite"):
        _controller().update((0.2, float("nan"), 0.2))
    physical = load_physical_grasp_experiment_config(CONFIG)
    with pytest.raises(ValueError, match="finite"):
        LiftFingerAbsoluteLoadHold(
            physical.lift_finger_absolute_load_hold,
            physical.sequential,
            base_targets_rad=(0.70, 0.55, 0.70),
            open_targets_rad=(0.0, 0.0, 0.0),
            closed_targets_rad=(0.765, 0.595, 0.765),
            initial_balance_total_rad=(0.0, 0.0, 0.0),
            reference_root_torque_nm=(0.2, float("inf"), 0.2),
        )


def test_h17_runner_uses_fresh_h13_samples_and_preserves_truth_firewall():
    source = RUNNER.read_text(encoding="utf-8")
    assert "h13_root_samples[-reference_count:]" in source
    assert "np.mean(\n                    np.abs(reference_samples), axis=0" in source
    assert "LiftFingerAbsoluteLoadHold" in source
    assert '"lift_finger_absolute_load_hold"' in source
    assert '"h11_runtime_controller_instantiated": False' in source
    assert '"input_is_immediately_preceding_sample"' in source
    assert '"arm_stiffness_modified_by_h17": False' in source
    assert '"arm_damping_modified_by_h17": False' in source
    assert '"vertical_trajectory_modified_by_h17": False' in source
    assert '"sensor_origin_hard_gate_unchanged": True' in source
    assert '"object_truth_used": False' in source
    assert '"contact_truth_used": False' in source
    assert '"event_truth_used": False' in source
    assert '"object_pose_written": False' in source
