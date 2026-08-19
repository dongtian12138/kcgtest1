from __future__ import annotations

from pathlib import Path

import pytest

from kcg_connector.closeout_monitor import (
    G10_SELF_INPUT_PATHS,
    G9_COMPLETION_PATHS,
    build_closeout_monitor,
    render_monitor_markdown,
    write_new,
)


ROOT = Path(__file__).resolve().parents[3]


def _report():
    return build_closeout_monitor(
        repository_root=ROOT,
        work_queue_path="artifacts/agent_control/WORK_QUEUE.yaml",
        master_state_path="artifacts/agent_control/MASTER_STATE.json",
        blocker_ledger_path="artifacts/agent_control/BLOCKER_LEDGER.jsonl",
        readiness_report_path="artifacts/agent_control/tasks/EIGHT-HOUR-G2-DYNAMIC-READINESS-DAG/DYNAMIC_READINESS_DAG.json",
        preflight_path="artifacts/agent_control/tasks/EIGHT-HOUR-G3-FINAL-REVIEW-PREFLIGHT/FINAL_REVIEW_PREFLIGHT.json",
        generated_at_utc="2026-08-18T01:55:00Z",
    )


def test_g9_seven_completion_sources_are_closed():
    report = _report()
    assert report["result"] == "OFFLINE_PASS"
    assert len(G9_COMPLETION_PATHS) == 7
    assert report["g9_expected_closure_source_count"] == 7
    assert report["g9_closed_source_count"] == 7
    assert report["g9_missing_from_package_closure_count"] == 0


def test_current_task_table_and_source_counts_are_exact():
    report = _report()
    assert report["task_table_row_count"] == 51
    assert report["package_source_count"] == 268
    assert report["code_module_count"] == 37
    assert report["test_file_count"] == 36


def test_in_memory_report_begins_with_table_and_has_fourteen_summaries():
    report = _report()
    assert report["report_begins_with_required_table"] is True
    assert report["summary_item_count"] == 14
    assert report["summary_item_numbers"] == list(range(1, 15))
    assert len(report["in_memory_report_sha256"]) == 64
    assert report["in_memory_report_size_bytes"] > 1000


def test_g10_three_inputs_are_frozen_for_g11():
    report = _report()
    assert len(G10_SELF_INPUT_PATHS) == 3
    assert report["g10_self_input_count"] == 3
    assert report["g10_self_closure_deferred_to_g11"] is True
    assert all(len(row["sha256"]) == 64 for row in report["g10_self_inputs"])


def test_static_check_total_is_complete():
    report = _report()
    assert report["passed_static_or_offline_check_count"] == report["declared_static_or_offline_check_total"]
    assert report["passed_static_or_offline_check_count"] > 1200


def test_no_formal_dynamic_or_authorization_claims():
    report = _report()
    assert report["formal_final_output_count"] == 0
    assert report["formal_report_written"] is False
    assert report["dynamic_passed_task_count"] == 0
    assert report["current_frontier_state"] == "HOME"
    assert report["historical_tests_rerun"] is False
    assert report["simulation_started"] is False
    assert report["robot_commands_emitted"] == 0
    assert report["assembly_success_claimed"] is False
    assert report["formal_r12_generated"] is False
    assert report["control_authorized"] is False
    assert report["hardware_authorized"] is False


def test_all_g9_closure_rows_are_unique_existing_files():
    rows = _report()["g9_closure_sources"]
    paths = [row["path"] for row in rows]
    assert len(paths) == len(set(paths))
    assert all((ROOT / row["path"]).is_file() for row in rows)
    assert all(row["size_bytes"] > 0 for row in rows)


def test_invalid_generated_time_fails_closed():
    with pytest.raises(ValueError, match="explicit UTC"):
        build_closeout_monitor(
            repository_root=ROOT,
            work_queue_path="artifacts/agent_control/WORK_QUEUE.yaml",
            master_state_path="artifacts/agent_control/MASTER_STATE.json",
            blocker_ledger_path="artifacts/agent_control/BLOCKER_LEDGER.jsonl",
            readiness_report_path="artifacts/agent_control/tasks/EIGHT-HOUR-G2-DYNAMIC-READINESS-DAG/DYNAMIC_READINESS_DAG.json",
            preflight_path="artifacts/agent_control/tasks/EIGHT-HOUR-G3-FINAL-REVIEW-PREFLIGHT/FINAL_REVIEW_PREFLIGHT.json",
            generated_at_utc="bad-time",
        )


def test_markdown_records_predeadline_boundary():
    markdown = render_monitor_markdown(_report())
    assert markdown.startswith("# 截止前最终报告闭包监控")
    assert "G9包源闭包 | 7/7 | PASS" in markdown
    assert "最终汇总 | 14项" in markdown
    assert "最终报告仅在内存渲染" in markdown


def test_outputs_are_immutable(tmp_path):
    output = tmp_path / "monitor.json"
    write_new(output, "{}\n")
    with pytest.raises(FileExistsError, match="immutable"):
        write_new(output, "{}\n")
