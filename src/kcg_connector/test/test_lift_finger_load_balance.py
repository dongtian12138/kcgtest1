from pathlib import Path

import pytest
import yaml

from kcg_connector.grasp.lift_finger_load_balance import (
    LiftFingerLoadBalance,
    load_lift_finger_load_balance_config,
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


def _controller():
    physical = load_physical_grasp_experiment_config(CONFIG)
    return LiftFingerLoadBalance(
        physical.lift_finger_load_balance,
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
    )


def test_h11_uses_only_existing_sequential_numeric_parameters():
    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    section = document["lift_finger_load_balance"]
    assert section == {
        "enabled": True,
        "threshold_label": "SIM_TUNING_ONLY_B_V2_H11",
        "source_run_id": "B-V2-GRASP-12",
        "reuse_sequential_balance_parameters": True,
        "zero_mean_closure_coordinate_trim": True,
        "immediately_preceding_root_torque_only": True,
    }
    config = load_lift_finger_load_balance_config(section)
    assert config.enabled is True
    assert not any(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in section.values()
    )


def test_h11_relieves_overloaded_fingers_without_increasing_mean_closure():
    controller = _controller()
    step = controller.update((0.2423124485, 0.2471886119, 0.2057836487))
    applied = step["applied_delta_closure_rad"]
    assert applied[0] < 0.0
    assert applied[1] < 0.0
    assert applied[2] > 0.0
    assert sum(applied) == pytest.approx(0.0, abs=1.0e-15)
    assert max(abs(value) for value in applied) <= 0.001 + 1.0e-15
    assert sum(step["cumulative_trim_closure_rad"]) == pytest.approx(
        0.0, abs=1.0e-15
    )
    assert step["object_truth_used"] is False
    assert step["contact_truth_used"] is False
    assert step["event_truth_used"] is False


def test_h11_stays_inside_joint_and_combined_balance_budgets_when_saturated():
    controller = _controller()
    last = None
    for _ in range(500):
        last = controller.update((0.30, 0.30, 0.10))
    assert last is not None
    assert last["output_targets_rad"][2] <= 0.765 + 1.0e-15
    assert all(
        abs(value) <= 0.030 + 1.0e-15
        for value in last["combined_balance_total_rad"]
    )
    assert sum(last["cumulative_trim_closure_rad"]) == pytest.approx(
        0.0, abs=1.0e-14
    )
    summary = controller.summary()
    assert summary["record_count"] == 500
    assert summary["maximum_abs_applied_step_rad"] <= 0.001 + 1.0e-15
    assert summary["mean_closure_target_change_rad"] == pytest.approx(
        0.0, abs=1.0e-14
    )


def test_h11_rejects_nonfinite_sensor_input():
    with pytest.raises(ValueError, match="finite"):
        _controller().update((0.2, float("nan"), 0.2))


def test_runner_records_preceding_step_causality_and_truth_isolation():
    source = RUNNER.read_text(encoding="utf-8")
    assert "LiftFingerLoadBalance" in source
    assert '"lift_finger_load_balance"' in source
    assert '"input_is_immediately_preceding_sample"' in source
    assert '"mean_closure_target_unchanged": True' in source
    assert '"sensor_origin_hard_gate_unchanged": True' in source
    assert '"object_truth_used": False' in source
    assert '"contact_truth_used": False' in source
    assert '"event_truth_used": False' in source
    assert '"object_pose_written": False' in source
