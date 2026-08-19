from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from kcg_connector.final_review_preflight import (
    FINAL_SUMMARY_FIELDS_CN,
    FINAL_TABLE_COLUMNS_CN,
    PLANNED_CLOSEOUT_MEMBERS,
    SCHEMA_VERSION,
    build_final_review_preflight,
    render_preflight_markdown,
    validate_bundle_name,
    write_preflight_pair,
)


ROOT = Path(__file__).resolve().parents[3]


def _build():
    return build_final_review_preflight(
        repository_root=ROOT,
        work_queue_path="artifacts/agent_control/WORK_QUEUE.yaml",
        master_state_path="artifacts/agent_control/MASTER_STATE.json",
        blocker_ledger_path="artifacts/agent_control/BLOCKER_LEDGER.jsonl",
        gate_ledger_path="artifacts/agent_control/GATE_LEDGER.csv",
        decision_log_path="artifacts/agent_control/DECISION_LOG.jsonl",
        status_history_path="artifacts/agent_control/STATUS_HISTORY.jsonl",
        task_graph_path="artifacts/agent_control/TASK_GRAPH.yaml",
        readiness_report_path=(
            "artifacts/agent_control/tasks/EIGHT-HOUR-G2-DYNAMIC-READINESS-DAG/"
            "DYNAMIC_READINESS_DAG.json"
        ),
        generated_at_utc="2026-08-18T00:55:00Z",
        planned_bundle_name="EIGHT_HOUR_ASSEMBLY_PROGRESS_20260818T031449Z.zip",
    )


def test_current_preflight_has_exact_report_shape():
    report = _build()
    assert report["schema_version"] == SCHEMA_VERSION
    assert report["result"] == "OFFLINE_PASS"
    assert report["report_blueprint"]["first_content_must_be_table"] is True
    assert report["report_blueprint"]["table_columns_cn"] == list(FINAL_TABLE_COLUMNS_CN)
    assert report["report_blueprint"]["task_row_count"] == 41
    assert report["report_blueprint"]["summary_field_count"] == 14
    assert [row["field_cn"] for row in report["report_blueprint"]["summary_source_bindings"]] == list(FINAL_SUMMARY_FIELDS_CN)


def test_dynamic_boundary_and_home_are_not_promoted():
    report = _build()
    values = report["summary_values_snapshot"]
    assert values["dynamic_passed_task_count"] == 0
    assert values["current_frontier_state"] == "HOME"
    assert report["static_or_offline_promoted_to_dynamic_count"] == 0
    assert all(row["true_dynamic_pass"] is False for row in report["report_blueprint"]["task_rows"])


def test_diagnostic_isaac_process_is_counted_but_not_formal_run():
    values = _build()["summary_values_snapshot"]
    assert values["isaac_process_run_count"] == 1
    assert values["formal_or_acceptance_isaac_run_count"] == 0
    assert values["isaac_cumulative_runtime_seconds"] == 96.0
    assert values["isaac_explicit_physics_step_count"] == 0


def test_absent_real_performance_stays_null():
    values = _build()["summary_values_snapshot"]
    assert values["peak_vram_mib"] is None
    assert values["physics_fps"] is None
    assert values["render_fps"] is None
    assert "no real measurement" in values["performance_missing_reason"]


def test_top_three_and_time_budget_are_evidence_bound():
    values = _build()["summary_values_snapshot"]
    assert values["top_three_priority_root_blockers"] == ["A1", "B1", "C8"]
    assert values["over_45_minute_task_count"] == 0
    assert values["over_45_minute_tasks"] == []


def test_known_forbidden_claims_are_zero_and_scoped():
    values = _build()["summary_values_snapshot"]
    assert values["known_forbidden_claim_count"] == 0
    assert values["known_forbidden_claims"] == []


def test_completed_modules_and_checks_have_narrow_counting_rules():
    values = _build()["summary_values_snapshot"]
    assert values["resolved_problem_count"] == 8
    assert len(values["resolved_problem_tasks"]) == 8
    assert values["completed_code_module_count"] > 25
    assert values["completed_test_file_count"] > 25
    assert values["passed_static_or_offline_check_count"] == values["declared_static_or_offline_check_total"]
    assert values["passed_static_or_offline_check_count"] > 1100


def test_package_members_are_direct_existing_paths_or_planned_closeout():
    package = _build()["package_blueprint"]
    assert package["existing_member_count"] > 80
    assert package["path_escape_count"] == 0
    assert package["future_members_hashed_early"] is False
    assert package["planned_closeout_members"] == list(PLANNED_CLOSEOUT_MEMBERS)
    assert all(len(row["sha256"]) == 64 for row in package["existing_members"])


def test_bundle_name_rejects_ambiguous_or_wrong_names():
    validate_bundle_name("EIGHT_HOUR_ASSEMBLY_PROGRESS_20260818T031449Z.zip")
    for value in ("success.zip", "latest.zip", "final_final.zip", "TASK_PASS.zip"):
        with pytest.raises(ValueError, match="naming rule"):
            validate_bundle_name(value)


def test_absolute_input_path_is_rejected():
    kwargs = dict(
        repository_root=ROOT,
        work_queue_path="/tmp/queue.yaml",
        master_state_path="artifacts/agent_control/MASTER_STATE.json",
        blocker_ledger_path="artifacts/agent_control/BLOCKER_LEDGER.jsonl",
        gate_ledger_path="artifacts/agent_control/GATE_LEDGER.csv",
        decision_log_path="artifacts/agent_control/DECISION_LOG.jsonl",
        status_history_path="artifacts/agent_control/STATUS_HISTORY.jsonl",
        task_graph_path="artifacts/agent_control/TASK_GRAPH.yaml",
        readiness_report_path="artifacts/agent_control/tasks/EIGHT-HOUR-G2-DYNAMIC-READINESS-DAG/DYNAMIC_READINESS_DAG.json",
        generated_at_utc="2026-08-18T00:55:00Z",
        planned_bundle_name="EIGHT_HOUR_ASSEMBLY_PROGRESS_20260818T031449Z.zip",
    )
    with pytest.raises(ValueError, match="absolute path"):
        build_final_review_preflight(**kwargs)


def test_markdown_states_missing_metrics_and_no_final_output():
    markdown = render_preflight_markdown(_build())
    assert markdown.startswith("# 八小时最终报告与审查包预检\n\n| 检查 |")
    assert "正式/验收 Isaac 运行：0 次" in markdown
    assert "显存峰值、物理帧率、渲染帧率保持空值" in markdown
    assert "不生成最终报告或压缩包" in markdown


def test_preflight_outputs_are_immutable_and_not_final_claims(tmp_path):
    report = _build()
    assert report["final_report_generated"] is False
    assert report["review_bundle_generated"] is False
    assert report["final_status_claimed"] is False
    assert report["simulation_started_by_preflight"] is False
    paths = [tmp_path / "preflight.json", tmp_path / "preflight.md"]
    write_preflight_pair(report, *paths)
    assert json.loads(paths[0].read_text())["result"] == "OFFLINE_PASS"
    with pytest.raises(FileExistsError, match="immutable"):
        write_preflight_pair(report, *paths)
