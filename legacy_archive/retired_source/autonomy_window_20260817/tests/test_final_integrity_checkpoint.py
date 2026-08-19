from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from kcg_connector.final_integrity_checkpoint import (
    G7_G8_CLOSURE_PATHS,
    G9_SELF_INPUT_PATHS,
    build_final_integrity_checkpoint,
    render_checkpoint_markdown,
    write_new,
)
from kcg_connector.stable_closeout_integrity import check_stable_manifest


ROOT = Path(__file__).resolve().parents[3]


def _report():
    return build_final_integrity_checkpoint(
        repository_root=ROOT,
        work_queue_path="artifacts/agent_control/WORK_QUEUE.yaml",
        master_state_path="artifacts/agent_control/MASTER_STATE.json",
        blocker_ledger_path="artifacts/agent_control/BLOCKER_LEDGER.jsonl",
        readiness_report_path="artifacts/agent_control/tasks/EIGHT-HOUR-G2-DYNAMIC-READINESS-DAG/DYNAMIC_READINESS_DAG.json",
        preflight_path="artifacts/agent_control/tasks/EIGHT-HOUR-G3-FINAL-REVIEW-PREFLIGHT/FINAL_REVIEW_PREFLIGHT.json",
        stable_baseline_path="artifacts/agent_control/tasks/EIGHT-HOUR-G7-CLOSEOUT-INTEGRITY-MONITOR/STABLE_SOURCE_BASELINE.json",
        generated_at_utc="2026-08-18T01:45:00Z",
    )


def test_g7_original_baseline_remains_zero_drift():
    report = _report()
    assert report["result"] == "OFFLINE_PASS"
    assert report["g7_baseline_source_count"] == 244
    assert report["g7_checked_source_count"] == 244
    assert report["g7_missing_source_count"] == 0
    assert report["g7_drift_source_count"] == 0
    assert report["g7_current_manifest_digest"] == report["g7_baseline_manifest_digest"]


def test_g7_g8_fourteen_sources_are_in_package_closure():
    report = _report()
    assert len(G7_G8_CLOSURE_PATHS) == 14
    assert report["g7_g8_expected_closure_source_count"] == 14
    assert report["g7_g8_closed_source_count"] == 14
    assert report["g7_g8_missing_from_package_closure_count"] == 0
    assert {row["path"] for row in report["g7_g8_closure_sources"]} == set(G7_G8_CLOSURE_PATHS)


def test_g9_three_inputs_are_frozen_for_g10():
    report = _report()
    assert len(G9_SELF_INPUT_PATHS) == 3
    assert report["g9_self_input_count"] == 3
    assert report["g9_self_closure_deferred_to_g10"] is True
    assert all(len(row["sha256"]) == 64 for row in report["g9_self_inputs"])


def test_current_final_builder_view_is_complete_but_not_finalized():
    report = _report()
    assert report["current_task_table_row_count"] == 50
    assert report["current_package_source_count"] > 240
    assert report["current_code_module_count"] == 36
    assert report["current_test_file_count"] == 35
    assert report["formal_final_output_count"] == 0


def test_dynamic_and_authorization_boundaries_remain_false():
    report = _report()
    assert report["dynamic_passed_task_count"] == 0
    assert report["current_frontier_state"] == "HOME"
    assert report["historical_tests_rerun"] is False
    assert report["repository_walk_performed"] is False
    assert report["simulation_started"] is False
    assert report["robot_commands_emitted"] == 0
    assert report["assembly_success_claimed"] is False
    assert report["formal_r12_generated"] is False
    assert report["control_authorized"] is False
    assert report["hardware_authorized"] is False


def test_all_snapshot_rows_are_unique_existing_files():
    report = _report()
    rows = report["g7_g8_closure_sources"] + report["g9_self_inputs"]
    paths = [row["path"] for row in rows]
    assert len(paths) == len(set(paths))
    assert all((ROOT / row["path"]).is_file() for row in rows)
    assert all(row["size_bytes"] > 0 for row in rows)


def test_tampered_g7_baseline_is_detected_without_source_write():
    path = ROOT / "artifacts/agent_control/tasks/EIGHT-HOUR-G7-CLOSEOUT-INTEGRITY-MONITOR/STABLE_SOURCE_BASELINE.json"
    baseline = json.loads(path.read_text())
    tampered = copy.deepcopy(baseline)
    tampered["stable_sources"][0]["sha256"] = "0" * 64
    check = check_stable_manifest(
        repository_root=ROOT,
        baseline=tampered,
        checked_at_utc="2026-08-18T01:45:00Z",
    )
    assert check["result"] == "FAIL"
    assert check["drift_source_count"] == 1


def test_invalid_generated_time_fails_closed():
    kwargs = {
        "repository_root": ROOT,
        "work_queue_path": "artifacts/agent_control/WORK_QUEUE.yaml",
        "master_state_path": "artifacts/agent_control/MASTER_STATE.json",
        "blocker_ledger_path": "artifacts/agent_control/BLOCKER_LEDGER.jsonl",
        "readiness_report_path": "artifacts/agent_control/tasks/EIGHT-HOUR-G2-DYNAMIC-READINESS-DAG/DYNAMIC_READINESS_DAG.json",
        "preflight_path": "artifacts/agent_control/tasks/EIGHT-HOUR-G3-FINAL-REVIEW-PREFLIGHT/FINAL_REVIEW_PREFLIGHT.json",
        "stable_baseline_path": "artifacts/agent_control/tasks/EIGHT-HOUR-G7-CLOSEOUT-INTEGRITY-MONITOR/STABLE_SOURCE_BASELINE.json",
        "generated_at_utc": "not-utc",
    }
    with pytest.raises(ValueError, match="explicit UTC"):
        build_final_integrity_checkpoint(**kwargs)


def test_markdown_exposes_self_closure_boundary():
    markdown = render_checkpoint_markdown(_report())
    assert markdown.startswith("# 最终完整性检查点")
    assert "G7/G8包源闭包 | 14/14 | PASS" in markdown
    assert "交G10闭包复核" in markdown
    assert "动态通过0" in markdown


def test_outputs_are_immutable(tmp_path):
    output = tmp_path / "checkpoint.json"
    write_new(output, "{}\n")
    with pytest.raises(FileExistsError, match="immutable"):
        write_new(output, "{}\n")
