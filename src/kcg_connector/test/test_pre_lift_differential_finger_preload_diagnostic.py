import ast
import math
from pathlib import Path

import numpy as np
import pytest

from kcg_connector.grasp.pre_lift_differential_finger_preload_diagnostic import (
    analyze_differential_probe,
    load_differential_finger_preload_diagnostic_config,
    probe_offset_rad,
)


REPOSITORY = Path(__file__).resolve().parents[3]
RUNNER = REPOSITORY / "src/kcg_connector/isaac/d38999_tabletop_pick_smoke.py"


def enabled_document():
    return {
        "enabled": True,
        "threshold_label": "SIM_TUNING_ONLY_B_V2_H23_DIAGNOSTIC",
        "source_h22_run_id": "B-V2-H22-PREUNLOADING-NULLING-01",
        "diagnostic_only": True,
        "finger_probe_order": ["f1", "f2", "f3"],
        "probe_increment_source": "SEQUENTIAL_PROBE_INCREMENT_RAD",
        "move_steps_source": "H4_PROBE_MOVE_STEPS",
        "settle_steps_source": "SEQUENTIAL_PROBE_SETTLE_STEPS",
        "sample_steps_source": "H4_PROBE_SAMPLE_STEPS",
        "minimum_probe_response_source": (
            "SEQUENTIAL_MINIMUM_PROBE_RESPONSE_NM"
        ),
        "arm_target_fixed": True,
        "vertical_feedforward_zero_required": True,
        "payload_reference_rebase_forbidden": True,
        "raw_sensor_hard_gate_unchanged": True,
        "hard_gate_detection_delay_steps": 0,
        "object_contact_event_truth_forbidden": True,
        "correction_applied_in_diagnostic": False,
    }


def test_h23_config_is_strict_and_defaults_disabled():
    assert load_differential_finger_preload_diagnostic_config(None).enabled is False
    config = load_differential_finger_preload_diagnostic_config(
        enabled_document()
    )
    assert config.enabled is True
    assert config.finger_probe_order == ("f1", "f2", "f3")
    assert config.hard_gate_detection_delay_steps == 0


@pytest.mark.parametrize(
    "key",
    (
        "diagnostic_only",
        "arm_target_fixed",
        "vertical_feedforward_zero_required",
        "payload_reference_rebase_forbidden",
        "raw_sensor_hard_gate_unchanged",
        "object_contact_event_truth_forbidden",
    ),
)
def test_h23_safety_flags_fail_closed(key):
    document = enabled_document()
    document[key] = False
    with pytest.raises(ValueError, match="must remain true"):
        load_differential_finger_preload_diagnostic_config(document)


def test_h23_rejects_correction_delay_unknown_and_changed_order():
    document = enabled_document()
    document["correction_applied_in_diagnostic"] = True
    with pytest.raises(ValueError, match="cannot apply"):
        load_differential_finger_preload_diagnostic_config(document)
    document = enabled_document()
    document["hard_gate_detection_delay_steps"] = 1
    with pytest.raises(ValueError, match="remain zero"):
        load_differential_finger_preload_diagnostic_config(document)
    document = enabled_document()
    document["gain"] = 1.0
    with pytest.raises(ValueError, match="unknown keys"):
        load_differential_finger_preload_diagnostic_config(document)
    document = enabled_document()
    document["finger_probe_order"] = ["f2", "f1", "f3"]
    with pytest.raises(ValueError, match="must remain f1,f2,f3"):
        load_differential_finger_preload_diagnostic_config(document)


def test_probe_offset_is_positive_bounded_and_exactly_returns():
    outward = [
        probe_offset_rad(index, 48, 0.004, returning=False)["offset_rad"]
        for index in range(48)
    ]
    returning = [
        probe_offset_rad(index, 48, 0.004, returning=True)["offset_rad"]
        for index in range(48)
    ]
    assert 0.0 < outward[0] < outward[-1]
    assert outward[-1] == pytest.approx(0.004)
    assert returning[0] < 0.004
    assert returning[-1] == pytest.approx(0.0, abs=1e-15)
    assert all(0.0 <= value <= 0.004 for value in outward + returning)


def test_probe_analysis_recovers_rank_two_zero_sum_subspace():
    amplitude = 0.004
    jacobian = np.asarray(
        (
            (2.0, -1.0, -1.0),
            (0.0, 1.5, -1.5),
            (1.0, -0.5, -0.5),
            (-0.25, 0.75, -0.5),
        ),
        dtype=np.float64,
    )
    baseline = np.asarray((0.1, -0.2, 0.3, -0.1), dtype=np.float64)
    baselines = np.stack((baseline, baseline, baseline, baseline))
    probes = np.stack(
        [baseline + amplitude * jacobian[:, index] for index in range(3)]
    )
    root_baselines = np.full((4, 3), 0.24, dtype=np.float64)
    root_probes = root_baselines[:3].copy()
    for index in range(3):
        root_probes[index, index] += 0.003
    result = analyze_differential_probe(
        baseline_objectives=baselines,
        probe_objectives=probes,
        baseline_root_loads_nm=root_baselines,
        probe_root_loads_nm=root_probes,
        probe_increment_rad=amplitude,
        minimum_probe_response_nm=0.002,
        damping_ratio=0.10,
        prospective_correction_norm_limit_rad=amplitude,
    )
    assert np.asarray(result["jacobian_per_rad"]) == pytest.approx(jacobian)
    assert result["differential_rank"] == 2
    assert result["minimum_probe_response_passed"] is True
    assert result["diagnostic_supported"] is True
    assert result["prospective_correction_sum_rad"] == pytest.approx(
        0.0, abs=1e-15
    )
    assert result["prospective_correction_norm_rad"] <= amplitude
    assert math.isfinite(result["differential_condition_number"])


def test_probe_analysis_uses_each_realized_increment_for_its_column():
    increments = np.asarray((0.004, 0.004, 0.003458936577305005))
    jacobian = np.asarray(
        (
            (2.0, -1.0, -1.0),
            (0.0, 1.5, -1.5),
            (1.0, -0.5, -0.5),
            (-0.25, 0.75, -0.5),
        ),
        dtype=np.float64,
    )
    baseline = np.asarray((0.1, -0.2, 0.3, -0.1), dtype=np.float64)
    baselines = np.stack((baseline, baseline, baseline, baseline))
    probes = np.stack(
        [
            baseline + increments[index] * jacobian[:, index]
            for index in range(3)
        ]
    )
    root_baselines = np.full((4, 3), 0.24, dtype=np.float64)
    root_probes = root_baselines[:3].copy()
    for index in range(3):
        root_probes[index, index] += 0.003
    result = analyze_differential_probe(
        baseline_objectives=baselines,
        probe_objectives=probes,
        baseline_root_loads_nm=root_baselines,
        probe_root_loads_nm=root_probes,
        probe_increment_rad=increments,
        minimum_probe_response_nm=0.002,
        damping_ratio=0.10,
        prospective_correction_norm_limit_rad=0.004,
    )
    assert result["realized_probe_increments_rad"] == pytest.approx(
        increments
    )
    assert result["probe_increment_normalization"] == (
        "per_finger_realized_increment"
    )
    assert np.asarray(result["jacobian_per_rad"]) == pytest.approx(jacobian)
    assert result["differential_rank"] == 2
    assert result["diagnostic_supported"] is True


@pytest.mark.parametrize(
    "increments",
    ((0.004, 0.004), (0.004, 0.0, 0.004), (0.004, float("nan"), 0.004)),
)
def test_probe_analysis_rejects_invalid_realized_increment_vectors(increments):
    with pytest.raises(ValueError, match="probe_increment_rad"):
        analyze_differential_probe(
            baseline_objectives=np.zeros((4, 4)),
            probe_objectives=np.zeros((3, 4)),
            baseline_root_loads_nm=np.zeros((4, 3)),
            probe_root_loads_nm=np.zeros((3, 3)),
            probe_increment_rad=increments,
            minimum_probe_response_nm=0.002,
            damping_ratio=0.10,
            prospective_correction_norm_limit_rad=0.004,
        )


def test_probe_analysis_rejects_missing_response_and_rank():
    zeros4 = np.zeros((4, 4), dtype=np.float64)
    zeros3 = np.zeros((3, 4), dtype=np.float64)
    loads4 = np.full((4, 3), 0.24, dtype=np.float64)
    loads3 = np.full((3, 3), 0.24, dtype=np.float64)
    result = analyze_differential_probe(
        baseline_objectives=zeros4,
        probe_objectives=zeros3,
        baseline_root_loads_nm=loads4,
        probe_root_loads_nm=loads3,
        probe_increment_rad=0.004,
        minimum_probe_response_nm=0.002,
        damping_ratio=0.10,
        prospective_correction_norm_limit_rad=0.004,
    )
    assert result["differential_rank"] == 0
    assert result["minimum_probe_response_passed"] is False
    assert result["diagnostic_supported"] is False
    assert result["correction_applied"] is False


def _h23_runner_block():
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    blocks = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and "pre_lift_differential_finger_preload_diagnostic.enabled"
        in ast.unparse(node.test)
        and any(
            isinstance(child, ast.For)
            and "h23_step_index" in ast.unparse(child.target)
            for child in node.body
        )
    ]
    assert len(blocks) == 1
    return blocks[0]


def test_h23_runner_executes_one_bounded_diagnostic_and_no_correction():
    source = ast.unparse(_h23_runner_block())
    assert "derive_probe_contract(" in source
    assert "probe_offset_rad(" in source
    assert "analyze_differential_probe(" in source
    assert "sequential_controller._bounded_target(" in source
    assert "h23_realized_probe_increments_rad.tolist()" in source
    assert "h23_probe_margin_insufficient" not in source
    assert "formal_lift_monitor.update(" in source
    assert "observe_and_step(h23_arm_target, current_hand_target, True, h23_zero_feedforward)" in source
    assert "lift_finger_absolute_controller.update(" not in source
    assert "derive_lift_xy_force_admittance_step(" not in source
    assert "'diagnostic_only': True" in source
    assert "'formal_b_pass_claimed': False" in source
    assert "'correction_applied': False" in source
    assert "'h17_controller_updated_during_probe': False" in source
    assert "'arm_xy_controller_updated_during_probe': False" in source
    assert "h23_diagnostic_complete_supported" in source
    assert "h23_diagnostic_complete_unsupported" in source


def test_h23_runner_keeps_arm_feedforward_reference_and_truth_frozen():
    block = _h23_runner_block()
    assigned_names = {
        target.id
        for node in ast.walk(block)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (
            node.targets if isinstance(node, ast.Assign) else (node.target,)
        )
        if isinstance(target, ast.Name)
    }
    assert "current_arm_target" not in assigned_names
    assert "formal_arm_feedforward_effort" not in assigned_names
    assert "formal_wrist_payload_reference" not in assigned_names
    source = ast.unparse(block)
    for forbidden in (
        "contact_snapshot(",
        "get_world_pose(",
        "set_world_pose(",
        "get_contact_report(",
        "get_contact_normal(",
    ):
        assert forbidden not in source
    assert "'raw_sensor_hard_gate_unchanged': True" in source
    assert "'hard_gate_detection_delay_steps': 0" in source
    assert "'object_truth_used': False" in source
    assert "'contact_truth_used': False" in source
    assert "'event_truth_used': False" in source
    assert "'object_pose_written': False" in source
