'''Contract tests for the empty-hand first-stage diagnostic replay.'''

from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


REPOSITORY = Path(__file__).resolve().parents[3]
RUNNER = (
    REPOSITORY
    / "src/kcg_connector/isaac/d38999_tabletop_pick_smoke.py"
)
MONITOR = (
    REPOSITORY
    / "src/kcg_connector/kcg_connector/grasp/grasp_stability_monitor.py"
)
SOURCE_ROOT = str(REPOSITORY / "src" / "kcg_connector")

TRUTH_TOKENS = (
    "get_world_pose",
    "contact_snapshot",
    "get_full_contact_report",
    "collider",
    "settled_body",
    "settled_nut",
    "body_in_tcp_frame",
    "contact_normal",
)


def _source():
    return RUNNER.read_text(encoding="utf-8")


def _run_cli(*arguments):
    environment = dict(os.environ)
    environment["PYTHONPATH"] = SOURCE_ROOT + os.pathsep + environment.get(
        "PYTHONPATH", ""
    )
    return subprocess.run(
        [sys.executable, str(RUNNER), *arguments],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


@pytest.mark.parametrize(
    "extra",
    [
        ("--physical-grasp-method", "single-finger", "--single-finger", "f1"),
        ("--physical-grasp-method", "sequential-compliant"),
        ("--formal-lift-mode", "zero-lift-hold"),
    ],
)
def test_cli_diagnostic_mutex_rejected_before_isaac(extra):
    result = _run_cli(
        "--physical-grasp-method", "synchronous",
        "--formal-lift-mode", "staged",
        "--empty-hand-first-stage-diagnostic",
        *extra,
    )
    assert result.returncode == 2
    assert "empty-hand-first-stage-diagnostic" in result.stderr


def test_cli_help_lists_diagnostic_flag():
    result = _run_cli("--help")
    assert result.returncode == 0
    assert "--empty-hand-first-stage-diagnostic" in result.stdout


def _diagnostic_region():
    source = _source()
    start = source.index(
        "if arguments.empty_hand_first_stage_diagnostic:",
        source.index("if zero_lift_hold_mode:") - 200000 if False else 0,
    )
    # The lift-flow diagnostic branch is the LAST occurrence of the flag
    # guard (the closure block also references the flag).
    occurrences = []
    cursor = 0
    while True:
        index = source.find(
            "if arguments.empty_hand_first_stage_diagnostic:", cursor
        )
        if index < 0:
            break
        occurrences.append(index)
        cursor = index + 1
    assert len(occurrences) >= 2
    start = occurrences[-1]
    end = source.index("if zero_lift_hold_mode:", start)
    return source[start:end]


def test_diagnostic_region_reads_no_truth_tokens_before_terminal_snapshot():
    region = _diagnostic_region()
    snapshot_index = region.rfind("capture_terminal_snapshot(")
    assert snapshot_index > 0
    control_part = region[:snapshot_index]
    for token in TRUTH_TOKENS:
        assert token not in control_part, (
            f"diagnostic control window reads {token}"
        )
    # The terminal posthoc truth read is allowed only at the end, after
    # the bounded arm return completes.
    assert "empty_hand_diagnostic_return" in region[:snapshot_index]


def test_diagnostic_region_never_grants_pass():
    region = _diagnostic_region()
    assert 'metrics["passed"] = False' in region
    assert 'metrics["grasp_success_claimed"] = False' in region
    assert '"granted_grasp_pass": False' in region
    assert '"diagnostic_only": True' in region
    assert 'process_exit_code = int(classification["exit_code"])' in region
    assert 'metrics["passed"] = True' not in region
    assert '"grasp_success_claimed": True' not in region


def test_diagnostic_region_reuses_stage1_target_path():
    region = _diagnostic_region()
    assert "solve_fixed_q7_tcp_pose(" in region
    assert "interpolate_arm(" in region
    assert "realized_randomization.lift_speed_scale" in region
    assert "plan_recovery_return(" in region
    assert "iiwa14_grasp_tcp_transform(" in region


def test_diagnostic_region_keeps_hand_open_and_skips_closure():
    source = _source()
    closure_start = source.index(
        "if arguments.empty_hand_first_stage_diagnostic:",
    )
    closure_end = source.index(
        "phase = \"closed_hand_seating\"", closure_start
    )
    closure_region = source[closure_start:closure_end]
    assert "total_closure_steps = 0" in closure_region
    assert "diagnostic_empty_hand_no_closure" in closure_region
    region = _diagnostic_region()
    assert '"diagnostic_empty_hand": True' in region
    assert '"attempted": False' in region
    assert '"not_attempted_reason": None' in region
    assert 'sensor_stream_unavailable_for_safe_return' in region


def test_diagnostic_region_has_no_latch_magnet_or_pose_writes():
    source = _source()
    closure_start = source.index(
        "if arguments.empty_hand_first_stage_diagnostic:",
    )
    closure_end = source.index(
        "phase = \"closed_hand_seating\"", closure_start
    )
    region = source[closure_start:closure_end] + _diagnostic_region()
    for token in ("magnet", "latch", "set_world_pose"):
        assert token not in region, f"diagnostic path touches {token}"


def test_diagnostic_recorder_has_no_root_load_gates():
    source = MONITOR.read_text(encoding="utf-8")
    start = source.index("class EmptyHandLiftDiagnosticMonitor:")
    class_source = source[start:]
    for token in (
        "maximum_root_torque_delta_nm",
        "minimum_retained_load_fraction",
        "maximum_normalized_load_imbalance",
        "maximum_load_rate_nm_s",
    ):
        assert token not in class_source, (
            f"diagnostic recorder references {token}"
        )
    assert "root_load_gates_applied" in class_source


def test_diagnostic_recorder_inputs_are_wrist_and_robot_only():
    from kcg_connector.grasp.grasp_stability_monitor import (
        EmptyHandLiftDiagnosticMonitor,
    )

    names = set(
        inspect.signature(EmptyHandLiftDiagnosticMonitor.update).parameters
    )
    assert names == {
        "self",
        "wrist_wrench_canonical",
        "arm_tracking_error_rad",
        "finger_velocities_rad_s",
    }



def test_diagnostic_region_writes_failure_and_boundary_fields():
    region = _diagnostic_region()
    assert 'metrics["formal_lift_failure"] = {' in region
    assert '"diagnostic_gate_observation": bool(' in region
    assert '"sensor_or_robot_fault": bool(' in region
    assert 'metrics["formal_lift_stages"] = formal_lift_stage_records' in region
    assert 'metrics["formal_lift_monitor"] = monitor_summary' in region
    assert 'metrics["control_reads_object_truth"] = False' in region
    assert 'metrics["control_reads_contact_report"] = False' in region
    assert (
        'metrics['
        in region
        and '"empty_hand_diagnostic_posthoc_consumed_by_control"'
        in region
    )
    assert 'metrics["empty_hand_diagnostic_termination"] = {' in region
    assert '"fault_category": classification["fault_category"]' in region
    assert 'metrics["empty_hand_diagnostic_return"] = return_record' in region
    assert 'metrics["formal_lift_terminal"] = True' in region



def test_diagnostic_region_uses_loop_exhaustion_and_status_mapping():
    region = _diagnostic_region()
    assert "stage_loop_completed_cleanly = False" in region
    assert "stage_loop_completed_cleanly = True" in region
    assert "stage_record_failure_evidence(termination)" in region
    assert "stage1_completed = stage1_completed_flag(" in region
    assert "diagnostic_report_status(" in region
    assert 'metrics["empty_hand_diagnostic_completed"] = bool(' in region
    assert "report_status[\"wrist_status\"]" in region
    assert "report_status[\"console_label\"]" in region



def test_runner_formal_monitor_uses_canonical_wrench_and_reference():
    source = _source()
    assert "wrist_reference=formal_wrist_payload_reference" in source
    assert (
        "formal_latest_wrist_payload_increment,\n"
        "                            arm_tracking_error_rad=arm_tracking"
    ) not in source
    assert source.count(
        "root_delta,\n                            formal_latest_wrist_canonical,"
    ) == 3
    centering_start = source.index(
        "def centering_control_step("
    )
    centering_end = source.index(
        "def centering_move(", centering_start
    )
    centering = source[centering_start:centering_end]
    assert "formal_lift_monitor.update(" in centering
    assert "formal_latest_wrist_canonical" in centering
    assert source.count("triggered=") == 0
