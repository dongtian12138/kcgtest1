from __future__ import annotations

import json
from pathlib import Path

import pytest

from kcg_connector.eight_hour_final_artifacts import (
    FINAL_OUTPUTS,
    FINAL_STATUS,
    FINAL_USER_ACTION,
    SCHEMA_VERSION,
    build_code_snapshot_patch,
    build_final_report_data,
    build_review_zip,
    collect_actual_commands,
    render_final_report,
    verify_review_zip,
)


ROOT = Path(__file__).resolve().parents[3]


def _data(closeout="2026-08-18T01:05:00Z"):
    return build_final_report_data(
        repository_root=ROOT,
        work_queue_path="artifacts/agent_control/WORK_QUEUE.yaml",
        master_state_path="artifacts/agent_control/MASTER_STATE.json",
        blocker_ledger_path="artifacts/agent_control/BLOCKER_LEDGER.jsonl",
        readiness_report_path=(
            "artifacts/agent_control/tasks/EIGHT-HOUR-G2-DYNAMIC-READINESS-DAG/"
            "DYNAMIC_READINESS_DAG.json"
        ),
        preflight_path=(
            "artifacts/agent_control/tasks/EIGHT-HOUR-G3-FINAL-REVIEW-PREFLIGHT/"
            "FINAL_REVIEW_PREFLIGHT.json"
        ),
        closeout_at_utc=closeout,
    )


def test_current_data_keeps_dynamic_and_physical_boundary():
    data = _data()
    assert data["schema_version"] == SCHEMA_VERSION
    assert data["metrics"]["dynamic_passed_task_count"] == 0
    assert data["metrics"]["current_frontier_state"] == "HOME"
    assert data["static_or_offline_promoted_to_dynamic_count"] == 0
    assert data["assembly_success_claimed"] is False
    assert data["formal_r12_generated"] is False
    assert data["hardware_authorized"] is False


def test_final_report_starts_with_required_table_and_has_all_summaries():
    report = render_final_report(_data())
    assert report.startswith("# 八小时自主任务最终报告\n\n| 任务 | 开始状态 | 结束状态 |")
    assert "## 十四项最终汇总" in report
    for number in range(1, 15):
        assert f"{number}." in report
    assert "物理动态否" in report
    assert "无实测值" in report


def test_task_rows_include_main_queue_and_completed_post_queue():
    data = _data()
    keys = [row["task"] for row in data["task_rows"]]
    assert keys[:4] == ["A1", "A2", "A3", "A4"]
    assert keys[40] == "F3"
    assert "QUEUE_AUDIT" in keys
    assert "G1" in keys
    assert "G2" in keys
    assert "G3" in keys
    assert "G4" not in keys


def test_metrics_are_recomputed_and_missing_performance_stays_null():
    metrics = _data()["metrics"]
    assert metrics["resolved_problem_count"] == 8
    assert metrics["completed_code_module_count"] >= 31
    assert metrics["completed_test_file_count"] >= 30
    assert metrics["passed_static_or_offline_check_count"] == metrics["declared_static_or_offline_check_total"]
    assert metrics["passed_static_or_offline_check_count"] >= 1218
    assert metrics["peak_vram_mib"] is None
    assert metrics["physics_fps"] is None
    assert metrics["render_fps"] is None


def test_isaac_process_and_formal_run_counts_are_separate():
    metrics = _data()["metrics"]
    assert metrics["isaac_process_run_count"] == 1
    assert metrics["formal_or_acceptance_isaac_run_count"] == 0
    assert metrics["isaac_cumulative_runtime_seconds"] == 96.0
    assert metrics["isaac_explicit_physics_step_count"] == 0


def test_priority_time_and_forbidden_rules_are_fail_closed():
    metrics = _data()["metrics"]
    assert metrics["top_three_priority_root_blockers"] == ["A1", "B1", "C8"]
    assert metrics["over_45_minute_task_count"] == 0
    assert metrics["known_forbidden_rule_violation_count"] == 0


def test_package_sources_are_existing_direct_paths_and_include_code():
    data = _data()
    assert len(data["package_source_paths"]) > 190
    assert len(data["package_source_paths"]) == len(set(data["package_source_paths"]))
    assert "src/kcg_connector/kcg_connector/dynamic_readiness_dag.py" in data["package_source_paths"]
    assert "src/kcg_connector/kcg_connector/final_review_preflight.py" in data["package_source_paths"]
    assert all((ROOT / path).is_file() for path in data["package_source_paths"])


def test_actual_command_collection_is_source_labeled():
    data = _data()
    text = collect_actual_commands(ROOT, data["package_source_paths"])
    assert text.startswith("# 八小时窗口实际命令记录")
    assert "EIGHT-HOUR-A1-INIT-DATUM/ACTUAL_COMMANDS.txt" in text
    assert "test_dynamic_readiness_dag.py" in text
    assert "test_final_review_preflight.py" in text


def test_code_snapshot_patch_is_not_misattributed():
    data = _data()
    patch = build_code_snapshot_patch(
        ROOT, data["code_modules"] + data["test_files"]
    )
    assert patch.startswith("# 八小时任务代码快照补丁")
    assert "不声称工作树全部改动均由本任务创建" in patch
    assert "+++ src/kcg_connector/kcg_connector/dynamic_readiness_dag.py" in patch


def test_review_zip_has_unique_members_and_verified_sha256(tmp_path):
    data = _data()
    closeout = []
    for relative in FINAL_OUTPUTS:
        path = tmp_path / relative.name
        path.write_text(f"fixture:{relative}\n", encoding="utf-8")
        closeout.append(path)
    # The real builder restricts bundle location; exercise the verifier using a
    # repository-local temporary review path that is removed by the test.
    review = ROOT / "artifacts/agent_control/review/EIGHT_HOUR_ASSEMBLY_PROGRESS_20990101T000000Z.zip"
    try:
        result = build_review_zip(
            root=ROOT,
            source_paths=data["package_source_paths"][:12],
            closeout_paths=[],
            bundle_path=review,
        )
        assert result["sha256_verified"] is True
        assert result["member_count"] == result["manifest_entry_count"] + 1
        assert verify_review_zip(review)["sha256_verified"] is True
    finally:
        review.unlink(missing_ok=True)


def test_review_zip_rejects_wrong_name_and_overwrite(tmp_path):
    bad = ROOT / "artifacts/agent_control/review/latest.zip"
    with pytest.raises(ValueError, match="naming rule"):
        build_review_zip(root=ROOT, source_paths=[], closeout_paths=[], bundle_path=bad)
    existing = ROOT / "artifacts/agent_control/review/EIGHT_HOUR_ASSEMBLY_PROGRESS_20990101T000001Z.zip"
    existing.write_bytes(b"existing")
    try:
        with pytest.raises(FileExistsError, match="immutable"):
            build_review_zip(root=ROOT, source_paths=[], closeout_paths=[], bundle_path=existing)
    finally:
        existing.unlink(missing_ok=True)


def test_predeadline_data_build_does_not_write_formal_outputs():
    before = {str(path): (ROOT / path).exists() for path in FINAL_OUTPUTS}
    data = _data("2026-08-18T01:10:00Z")
    after = {str(path): (ROOT / path).exists() for path in FINAL_OUTPUTS}
    assert data["final_status_requested"] == FINAL_STATUS
    assert data["final_user_action_requested"] == FINAL_USER_ACTION
    assert before == after
