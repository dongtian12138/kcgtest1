import json
import math
from pathlib import Path

import pytest

from kcg_connector.grasp.grasp_stability_monitor import (
    GraspStabilityConfig,
    GraspStabilityMonitor,
    wrist_payload_increment,
)


def _monitor() -> GraspStabilityMonitor:
    return GraspStabilityMonitor(
        GraspStabilityConfig(
            maximum_root_torque_delta_nm=2.0,
            minimum_retained_load_fraction=0.5,
            maximum_normalized_load_imbalance=0.8,
            maximum_load_rate_nm_s=20.0,
            maximum_wrist_force_n=8.0,
            maximum_wrist_moment_nm=0.3,
            maximum_arm_tracking_error_rad=0.03,
            maximum_finger_speed_rad_s=0.8,
            loss_confirm_steps=3,
        ),
        reference_load_nm=(0.3, 0.3, 0.3),
        load_scale_nm=(0.3, 0.3, 0.3),
        sample_period_s=0.01,
        wrist_reference=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )


def _update(monitor, loads, wrench=(0.0,) * 6):
    return monitor.update(
        loads,
        wrench,
        arm_tracking_error_rad=0.001,
        finger_velocities_rad_s=(0.0, 0.0, 0.0),
    )


def test_stable_lift_signals_pass_without_object_truth():
    monitor = _monitor()
    assert _update(monitor, (0.30, 0.30, 0.30))
    assert _update(monitor, (0.29, 0.31, 0.30))
    assert not monitor.failed


def test_load_loss_is_debounced_then_fails_closed():
    monitor = _monitor()
    assert _update(monitor, (0.20, 0.30, 0.30))
    assert _update(monitor, (0.14, 0.30, 0.30))
    assert _update(monitor, (0.14, 0.30, 0.30))
    assert not _update(monitor, (0.14, 0.30, 0.30))
    assert monitor.failure_reason == "f1_load_lost"


def test_wrist_force_and_arm_tracking_gates_fail_closed():
    monitor = _monitor()
    assert not _update(monitor, (0.30, 0.30, 0.30), (9.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    assert monitor.failure_reason == "wrist_force_limit"

    monitor = _monitor()
    assert not monitor.update(
        (0.30, 0.30, 0.30),
        (0.0,) * 6,
        arm_tracking_error_rad=0.04,
        finger_velocities_rad_s=(0.0, 0.0, 0.0),
    )
    assert monitor.failure_reason == "arm_tracking_limit"


def test_wrist_payload_increment_math_and_validation():
    canonical = (1.0, -2.0, 3.0, 4.0, 5.0, 6.0)
    reference = (0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
    assert wrist_payload_increment(canonical, reference) == (
        0.5,
        -2.5,
        2.5,
        3.5,
        4.5,
        5.5,
    )
    with pytest.raises(ValueError):
        wrist_payload_increment((1.0, 2.0, 3.0), reference)
    with pytest.raises(ValueError):
        wrist_payload_increment(
            (1.0, 2.0, 3.0, 4.0, 5.0, float("nan")), reference
        )


def test_static_held_payload_does_not_trip_gates_when_input_is_increment():
    # rep03 evidence: the pre-lift canonical wrench already contains the
    # static held-payload load.  The monitor must receive the increment
    # relative to the payload reference, so a quasi-static first lift step
    # with a tiny motion delta passes the unchanged 8 N / 0.30 Nm gates.
    payload_reference = (0.994, -0.637, 20.194, 0.308, 0.476, 0.0016)
    first_step_canonical = (0.996, -0.639, 20.190, 0.311, 0.479, 0.0017)
    increment = wrist_payload_increment(
        first_step_canonical, payload_reference
    )
    monitor = _monitor()
    assert _update(monitor, (0.30, 0.30, 0.30), increment)
    assert not monitor.failed


def test_payload_increment_moment_gate_still_fails_closed():
    monitor = _monitor()
    increment = (0.0, 0.0, 0.0, 0.25, 0.20, 0.0)
    assert not _update(monitor, (0.30, 0.30, 0.30), increment)
    assert monitor.failure_reason == "wrist_moment_limit"
    summary = monitor.summary()
    assert summary["peak_wrist_moment_increment_nm"] == pytest.approx(
        math.hypot(0.25, 0.20)
    )
    assert summary["peak_per_channel_increment"] == pytest.approx(
        [0.0, 0.0, 0.0, 0.25, 0.20, 0.0]
    )


def test_fail_closed_forces_sensor_loss_without_stale_sample():
    monitor = _monitor()
    assert _update(monitor, (0.30, 0.30, 0.30))
    assert monitor.step_count == 1
    assert not monitor.fail_closed("wrist_ft_sensor_error")
    assert monitor.failed
    assert monitor.failure_reason == "wrist_ft_sensor_error"
    assert monitor.step_count == 1
    assert monitor.summary()["failed"] is True
    assert monitor.summary()["failure_reason"] == "wrist_ft_sensor_error"


def test_summary_tracks_increment_peaks_and_unchanged_gate_limits():
    monitor = _monitor()
    assert _update(monitor, (0.30, 0.30, 0.30), (1.0, -2.0, 0.5, 0.1, 0.0, 0.0))
    assert _update(monitor, (0.30, 0.30, 0.30), (0.0, 1.0, 0.0, 0.0, 0.2, 0.0))
    summary = monitor.summary()
    assert summary["steps"] == 2
    assert summary["failed"] is False
    assert summary["failure_reason"] is None
    assert summary["peak_wrist_force_increment_n"] == pytest.approx(
        math.hypot(2.0, 1.0, 0.5)
    )
    assert summary["peak_wrist_moment_increment_nm"] == pytest.approx(
        math.hypot(0.2)
    )
    assert summary["peak_per_channel_increment"] == pytest.approx(
        [1.0, 2.0, 0.5, 0.1, 0.2, 0.0]
    )
    assert summary["last_increment"] == pytest.approx(
        [0.0, 1.0, 0.0, 0.0, 0.2, 0.0]
    )
    assert summary["force_gate_n"] == 8.0
    assert summary["moment_gate_nm"] == 0.3


from kcg_connector.grasp.grasp_stability_monitor import (
    EmptyHandLiftDiagnosticMonitor,
)


def _diagnostic_config() -> GraspStabilityConfig:
    return GraspStabilityConfig(
        maximum_root_torque_delta_nm=2.0,
        minimum_retained_load_fraction=0.45,
        maximum_normalized_load_imbalance=0.80,
        maximum_load_rate_nm_s=25.0,
        maximum_wrist_force_n=8.0,
        maximum_wrist_moment_nm=0.30,
        maximum_arm_tracking_error_rad=0.030,
        maximum_finger_speed_rad_s=0.80,
        loss_confirm_steps=18,
    )


def _diagnostic_reference():
    return (1.1677, 0.0519, 19.7246, 0.0426, 0.5519, 0.0182)


def _diagnostic_update(monitor, wrench, *, tracking=0.001, fingers=(0.0, 0.0, 0.0)):
    return monitor.update(
        wrench,
        arm_tracking_error_rad=tracking,
        finger_velocities_rad_s=fingers,
    )


def test_empty_hand_monitor_passes_below_gates_and_records_peaks():
    monitor = EmptyHandLiftDiagnosticMonitor(
        _diagnostic_config(), reference_wrench=_diagnostic_reference()
    )
    reference = _diagnostic_reference()
    small = tuple(value + 0.001 for value in reference)
    assert _diagnostic_update(monitor, small) is True
    summary = monitor.summary()
    assert summary["failed"] is False
    assert summary["failure_reason"] is None
    assert summary["diagnostic_only"] is True
    assert summary["root_load_gates_applied"] is False
    assert summary["steps"] == 1
    assert summary["peak_wrist_force_increment_n"] == pytest.approx(
        math.sqrt(3) * 0.001
    )
    assert summary["peak_wrist_moment_increment_nm"] == pytest.approx(
        math.sqrt(3) * 0.001
    )


def test_empty_hand_moment_gate_exact_equality_passes_strictly_greater_stops():
    monitor = EmptyHandLiftDiagnosticMonitor(
        _diagnostic_config(), reference_wrench=_diagnostic_reference()
    )
    reference = _diagnostic_reference()
    exact = (
        reference[0], reference[1], reference[2],
        reference[3], reference[4] + 0.30, reference[5],
    )
    assert _diagnostic_update(monitor, exact) is True
    monitor2 = EmptyHandLiftDiagnosticMonitor(
        _diagnostic_config(), reference_wrench=_diagnostic_reference()
    )
    over = (
        reference[0], reference[1], reference[2],
        reference[3], reference[4] + 0.30 + 1e-9, reference[5],
    )
    assert _diagnostic_update(monitor2, over) is False
    assert monitor2.failure_reason == "empty_hand_wrist_moment_gate_observed"
    assert monitor2.summary()["peak_wrist_moment_increment_nm"] == (
        pytest.approx(0.30 + 1e-9)
    )


def test_empty_hand_force_gate_exact_equality_passes_strictly_greater_stops():
    monitor = EmptyHandLiftDiagnosticMonitor(
        _diagnostic_config(), reference_wrench=_diagnostic_reference()
    )
    reference = _diagnostic_reference()
    exact = (reference[0] + 8.0, reference[1], reference[2],) + tuple(
        reference[3:]
    )
    assert _diagnostic_update(monitor, exact) is True
    over = (reference[0] + 8.0 + 1e-9, reference[1], reference[2],) + tuple(
        reference[3:]
    )
    assert _diagnostic_update(monitor, over) is False
    assert monitor.failure_reason == "empty_hand_wrist_force_gate_observed"


def test_empty_hand_arm_tracking_and_finger_speed_gates():
    monitor = EmptyHandLiftDiagnosticMonitor(
        _diagnostic_config(), reference_wrench=_diagnostic_reference()
    )
    assert _diagnostic_update(
        monitor, _diagnostic_reference(), tracking=0.030 + 1e-9
    ) is False
    assert monitor.failure_reason == "empty_hand_arm_tracking_gate_observed"
    monitor2 = EmptyHandLiftDiagnosticMonitor(
        _diagnostic_config(), reference_wrench=_diagnostic_reference()
    )
    assert _diagnostic_update(
        monitor2, _diagnostic_reference(), fingers=(0.0, 0.80 + 1e-9, 0.0)
    ) is False
    assert monitor2.failure_reason == "empty_hand_finger_speed_gate_observed"


def test_empty_hand_monitor_nonfinite_fails_closed_without_raising():
    monitor = EmptyHandLiftDiagnosticMonitor(
        _diagnostic_config(), reference_wrench=_diagnostic_reference()
    )
    wrench = list(_diagnostic_reference())
    wrench[3] = float("nan")
    assert _diagnostic_update(monitor, wrench) is False
    assert monitor.failure_reason == (
        "empty_hand_nonfinite_sensor_or_robot_state"
    )
    assert monitor.failed is True


def test_empty_hand_monitor_has_no_root_load_gates():
    import inspect

    parameters = inspect.signature(EmptyHandLiftDiagnosticMonitor.update)
    names = set(parameters.parameters)
    assert "root_torque" not in " ".join(names)
    assert "reference_load" not in " ".join(names)
    summary = EmptyHandLiftDiagnosticMonitor(
        _diagnostic_config(), reference_wrench=_diagnostic_reference()
    ).summary()
    assert summary["root_load_gates_applied"] is False


def test_empty_hand_monitor_reference_validation():
    with pytest.raises(ValueError):
        EmptyHandLiftDiagnosticMonitor(
            _diagnostic_config(), reference_wrench=(1.0, 2.0, 3.0)
        )
    with pytest.raises(ValueError):
        EmptyHandLiftDiagnosticMonitor(
            _diagnostic_config(), reference_wrench=(1.0, 2.0, 3.0, 4.0, 5.0, float("inf"))
        )


def test_empty_hand_monitor_summary_is_json_safe():
    import json

    monitor = EmptyHandLiftDiagnosticMonitor(
        _diagnostic_config(), reference_wrench=_diagnostic_reference()
    )
    _diagnostic_update(monitor, _diagnostic_reference())
    json.dumps(monitor.summary(), allow_nan=False)



from kcg_connector.grasp.grasp_stability_monitor import (
    evaluate_wrist_moment_safety,
)


def _eval(m, rvec, limit=0.30):
    return evaluate_wrist_moment_safety(m, rvec, limit)


def test_moment_safety_zero_reference_uses_absolute_magnitude():
    evidence = _eval([0.31, 0.0, 0.0], [0.0, 0.0, 0.0])
    assert evidence["magnitude_increase_nm"] == pytest.approx(0.31)
    assert evidence["perpendicular_nm"] == pytest.approx(0.31)
    assert evidence["reversal_nm"] == 0.0
    assert evidence["parallel_current_nm"] is None
    assert evidence["triggered"] is True
    assert evidence["trigger_component"] == "magnitude_increase"


def test_moment_safety_tiny_reference_no_division_by_zero():
    evidence = _eval([0.5, 0.0, 0.0], [1e-13, 0.0, 0.0])
    assert evidence["magnitude_increase_nm"] == pytest.approx(0.5)
    assert evidence["triggered"] is True


def test_moment_safety_parallel_increase_fires():
    reference = [0.0, 0.55, 0.0]
    evidence = _eval([0.0, 0.86, 0.0], reference)
    assert evidence["magnitude_increase_nm"] == pytest.approx(0.31)
    assert evidence["triggered"] is True
    assert evidence["trigger_component"] == "magnitude_increase"


def test_moment_safety_parallel_decrease_passes():
    reference = [0.0, 0.55, 0.0]
    evidence = _eval([0.0, 0.24, 0.0], reference)
    assert evidence["magnitude_increase_nm"] == 0.0
    assert evidence["perpendicular_nm"] == pytest.approx(0.0, abs=1e-12)
    assert evidence["reversal_nm"] == 0.0
    assert evidence["triggered"] is False
    assert evidence["legacy_delta_norm_nm"] == pytest.approx(0.31)


def test_moment_safety_perpendicular_fires():
    reference = [0.0, 0.55, 0.0]
    evidence = _eval([0.31, 0.55, 0.0], reference)
    assert evidence["perpendicular_nm"] == pytest.approx(0.31)
    assert evidence["triggered"] is True
    assert evidence["trigger_component"] == "perpendicular"


def test_moment_safety_diagonal_combination():
    reference = [0.0, 0.55, 0.0]
    passing = _eval([0.25, 0.75, 0.0], reference)
    assert passing["magnitude_increase_nm"] == pytest.approx(
        math.sqrt(0.25**2 + 0.75**2) - 0.55
    )
    assert passing["perpendicular_nm"] == pytest.approx(0.25)
    assert passing["triggered"] is False
    failing = _eval([0.31, 0.75, 0.0], reference)
    assert failing["triggered"] is True
    assert failing["trigger_component"] == "perpendicular"


def test_moment_safety_reversal_fires_only_past_zero():
    reference = [0.0, 0.55, 0.0]
    before = _eval([0.0, 0.05, 0.0], reference)
    assert before["reversal_nm"] == 0.0
    small = _eval([0.0, -0.2, 0.0], reference)
    assert small["reversal_nm"] == pytest.approx(0.2)
    assert small["triggered"] is False
    large = _eval([0.0, -0.31, 0.0], reference)
    assert large["reversal_nm"] == pytest.approx(0.31)
    assert large["triggered"] is True
    assert large["trigger_component"] == "reversal"


def test_moment_safety_180_degree_reversal():
    reference = [0.0, 0.55, 0.0]
    small = _eval([0.0, -0.2, 0.0], reference)
    assert small["triggered"] is False
    large = _eval([0.0, -0.55, 0.0], reference)
    assert large["reversal_nm"] == pytest.approx(0.55)
    assert large["triggered"] is True


@pytest.mark.parametrize(
    "m, rvec, limit",
    [
        (["bad", 0, 0], [0, 0.5, 0], 0.3),
        ([0, 0, 0], [0, 0.5], 0.3),
        ([0, 0, 0], [0, 0.5, 0], "bad"),
        ([0, 0, 0], [0, 0.5, 0], True),
        ([0, 0, 0], [0, 0.5, 0], -1.0),
        ([0, 0, 0], [0, 0.5, 0], float("nan")),
        ([0, float("nan"), 0], [0, 0.5, 0], 0.3),
        (True, [0, 0.5, 0], 0.3),
        ([0, 0, 0], True, 0.3),
    ],
)
def test_moment_safety_validation_rejects_bad_inputs(m, rvec, limit):
    with pytest.raises(ValueError):
        _eval(m, rvec, limit)


def test_moment_safety_exact_limit_passes_strictly_greater_fires():
    reference = [0.0, 0.55, 0.0]
    exact = _eval([0.0, 0.85, 0.0], reference)
    assert exact["magnitude_increase_nm"] == pytest.approx(0.30)
    assert exact["triggered"] is False
    over = _eval([0.0, 0.85 + 1e-9, 0.0], reference)
    assert over["triggered"] is True


def test_moment_safety_component_tie_uses_frozen_priority():
    reference = [0.0, 0.55, 0.0]
    # magnitude_increase and perpendicular both exactly 0.31: the frozen
    # priority order magnitude_increase < perpendicular < reversal decides.
    lateral = 0.31
    axial = math.sqrt((0.55 + 0.31) ** 2 - lateral * lateral)
    evidence = _eval([lateral, axial, 0.0], reference)
    assert evidence["magnitude_increase_nm"] == pytest.approx(0.31)
    assert evidence["perpendicular_nm"] == pytest.approx(0.31)
    assert evidence["triggered"] is True
    assert evidence["trigger_component"] == "magnitude_increase"


def test_moment_safety_legacy_unload_does_not_trigger_new_score():
    reference = [0.0, 0.55, 0.0]
    evidence = _eval([0.0, 0.23, 0.0], reference)
    assert evidence["legacy_delta_norm_nm"] == pytest.approx(0.32)
    assert evidence["gate_score_nm"] == pytest.approx(0.0, abs=1e-12)
    assert evidence["triggered"] is False


def test_monitor_summary_is_json_safe_and_legacy_evidence_only():
    import json

    monitor = _monitor()
    _update(monitor, (0.3, 0.3, 0.3), (1.0, 0.0, 20.0, 0.0, 0.5, 0.0))
    summary = monitor.summary()
    json.dumps(summary, allow_nan=False)
    assert summary["wrist_moment_semantics"] == (
        "three_component_decomposition_v1"
    )
    assert summary["legacy_moment_delta_is_evidence_only"] is True
    assert summary["peak_wrist_moment_increment_nm"] == pytest.approx(0.5)
    assert "last_moment_safety_evidence" in summary



REPOSITORY = Path(__file__).resolve().parents[3]
GRASPED_TRACE = (
    REPOSITORY
    / "artifacts/kcg_connector/tabletop_physical_grasp_v1/synchronous_grasp/"
    "seed000_headless_staged_rep02_diagnostic_source_codex/controller_steps.jsonl"
)
EMPTY_TRACE = (
    REPOSITORY
    / "artifacts/kcg_connector/tabletop_physical_grasp_v1/diagnostics/"
    "empty_hand_first_stage_seed000_headless_rep01_codex/controller_steps.jsonl"
)


def _replay_moment_safety(trace_path: Path, reference):
    rows = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    steps = []
    for record in rows:
        if str(record.get("phase", "")).startswith(
            "physical_grip_lift_stage"
        ) or (
            record.get("diagnostic_empty_hand")
            and record.get("lift_stage") == 1
        ):
            evidence = evaluate_wrist_moment_safety(
                record["wrist_wrench_canonical"][3:], reference[3:], 0.30
            )
            steps.append((record.get("lift_stage_step"), evidence))
    return steps


@pytest.mark.skipif(not GRASPED_TRACE.is_file(), reason="artifact missing")
def test_real_grasped_rep02_moment_replay_matches_offline_counterfactual():
    import json

    report_path = GRASPED_TRACE.parent / "nominal_physics_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    reference = report["formal_payload_wrist_reference"]
    steps = _replay_moment_safety(GRASPED_TRACE, reference)
    assert len(steps) == 32
    legacy_peak = max(step[1]["legacy_delta_norm_nm"] for step in steps)
    legacy_step = next(
        step[0] for step in steps
        if step[1]["legacy_delta_norm_nm"] == legacy_peak
    )
    assert legacy_peak == pytest.approx(0.32128, abs=1e-4)
    assert legacy_step == 31
    score_peak = max(step[1]["gate_score_nm"] for step in steps)
    score_step = next(
        step[0] for step in steps
        if step[1]["gate_score_nm"] == score_peak
    )
    assert score_peak == pytest.approx(0.06467, abs=1e-4)
    assert score_step == 25
    assert all(step[1]["gate_score_nm"] <= 0.30 for step in steps)
    assert all(
        not step[1]["triggered"] for step in steps
    )
    assert [step[0] for step in steps if step[1]["legacy_delta_norm_nm"] > 0.30] == [31]


@pytest.mark.skipif(not EMPTY_TRACE.is_file(), reason="artifact missing")
def test_real_empty_hand_moment_replay_passes_with_large_margin():
    import json

    report_path = EMPTY_TRACE.parent / "nominal_physics_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    reference = report["formal_payload_wrist_reference"]
    steps = _replay_moment_safety(EMPTY_TRACE, reference)
    assert len(steps) == 385
    score_peak = max(step[1]["gate_score_nm"] for step in steps)
    assert score_peak == pytest.approx(0.00106, abs=1e-4)
    assert all(step[1]["gate_score_nm"] <= 0.30 for step in steps)
    assert all(not step[1]["triggered"] for step in steps)



def test_monitor_moment_gate_uses_decomposition_and_records_component():
    config = GraspStabilityConfig(
        maximum_root_torque_delta_nm=2.0,
        minimum_retained_load_fraction=0.45,
        maximum_normalized_load_imbalance=0.80,
        maximum_load_rate_nm_s=25.0,
        maximum_wrist_force_n=8.0,
        maximum_wrist_moment_nm=0.30,
        maximum_arm_tracking_error_rad=0.030,
        maximum_finger_speed_rad_s=0.80,
        loss_confirm_steps=18,
    )
    monitor = GraspStabilityMonitor(
        config,
        reference_load_nm=(0.3, 0.3, 0.3),
        load_scale_nm=(0.3, 0.3, 0.3),
        sample_period_s=0.01,
        wrist_reference=(1.0, 0.0, 20.0, 0.0, 0.55, 0.0),
    )
    # A perpendicular-only moment of 0.31 on top of the reference: the
    # decomposition fires with the frozen reason and component label.
    okay = monitor.update(
        (0.3, 0.3, 0.3),
        (1.0, 0.0, 20.0, 0.31, 0.55, 0.0),
        arm_tracking_error_rad=0.001,
        finger_velocities_rad_s=(0.0, 0.0, 0.0),
    )
    assert okay is False
    assert monitor.failure_reason == "wrist_moment_limit"
    assert monitor.moment_trigger_component == "perpendicular"
    assert monitor.last_moment_safety_evidence["perpendicular_nm"] == (
        pytest.approx(0.31)
    )
    assert monitor.last_moment_safety_evidence["legacy_delta_norm_nm"] == (
        pytest.approx(0.31)
    )


def test_monitor_force_gate_with_nonzero_reference():
    config = GraspStabilityConfig(
        maximum_root_torque_delta_nm=2.0,
        minimum_retained_load_fraction=0.45,
        maximum_normalized_load_imbalance=0.80,
        maximum_load_rate_nm_s=25.0,
        maximum_wrist_force_n=8.0,
        maximum_wrist_moment_nm=0.30,
        maximum_arm_tracking_error_rad=0.030,
        maximum_finger_speed_rad_s=0.80,
        loss_confirm_steps=18,
    )
    reference = (1.0, 0.0, 20.0, 0.0, 0.55, 0.0)
    exact = GraspStabilityMonitor(
        config,
        reference_load_nm=(0.3, 0.3, 0.3),
        load_scale_nm=(0.3, 0.3, 0.3),
        sample_period_s=0.01,
        wrist_reference=reference,
    )
    assert exact.update(
        (0.3, 0.3, 0.3),
        (1.0 + 8.0, 0.0, 20.0, 0.0, 0.55, 0.0),
        arm_tracking_error_rad=0.001,
        finger_velocities_rad_s=(0.0, 0.0, 0.0),
    ) is True
    over = GraspStabilityMonitor(
        config,
        reference_load_nm=(0.3, 0.3, 0.3),
        load_scale_nm=(0.3, 0.3, 0.3),
        sample_period_s=0.01,
        wrist_reference=reference,
    )
    assert over.update(
        (0.3, 0.3, 0.3),
        (1.0 + 8.0 + 1e-9, 0.0, 20.0, 0.0, 0.55, 0.0),
        arm_tracking_error_rad=0.001,
        finger_velocities_rad_s=(0.0, 0.0, 0.0),
    ) is False
    assert over.failure_reason == "wrist_force_limit"



def _reference_config():
    return GraspStabilityConfig(
        maximum_root_torque_delta_nm=2.0,
        minimum_retained_load_fraction=0.45,
        maximum_normalized_load_imbalance=0.80,
        maximum_load_rate_nm_s=25.0,
        maximum_wrist_force_n=8.0,
        maximum_wrist_moment_nm=0.30,
        maximum_arm_tracking_error_rad=0.030,
        maximum_finger_speed_rad_s=0.80,
        loss_confirm_steps=18,
    )


def test_monitor_accepts_numpy_ndarray_wrist_reference():
    import numpy as np

    reference = np.asarray(
        [1.0, 0.0, 20.0, 0.0, 0.55, 0.0],
        dtype=np.float64,
    )
    monitor = GraspStabilityMonitor(
        _reference_config(),
        reference_load_nm=(0.3, 0.3, 0.3),
        load_scale_nm=(0.3, 0.3, 0.3),
        sample_period_s=0.01,
        wrist_reference=reference,
    )
    assert monitor.wrist_reference == tuple(reference)
    # A step with a perpendicular-only moment on top of the ndarray
    # reference still decomposes correctly.
    assert monitor.update(
        (0.3, 0.3, 0.3),
        (1.0, 0.0, 20.0, 0.31, 0.55, 0.0),
        arm_tracking_error_rad=0.001,
        finger_velocities_rad_s=(0.0, 0.0, 0.0),
    ) is False
    assert monitor.failure_reason == "wrist_moment_limit"
    assert monitor.moment_trigger_component == "perpendicular"


def test_monitor_rejects_bad_wrist_reference_types():
    import numpy as np

    bad_values = [
        "not-a-vector",
        b"not-a-vector",
        5.0,
        None,
        True,
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [1.0, 2.0, 3.0, 4.0, 5.0, float("nan")],
        [1.0, 2.0, 3.0, 4.0, 5.0, float("inf")],
        np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]),
        np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]),
        np.array(5.0),
    ]
    for bad in bad_values:
        with pytest.raises(ValueError):
            GraspStabilityMonitor(
                _reference_config(),
                reference_load_nm=(0.3, 0.3, 0.3),
                load_scale_nm=(0.3, 0.3, 0.3),
                sample_period_s=0.01,
                wrist_reference=bad,
            )

