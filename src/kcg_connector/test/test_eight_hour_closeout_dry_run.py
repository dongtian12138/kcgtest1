from __future__ import annotations

import json
from pathlib import Path

import pytest

from kcg_connector.eight_hour_closeout_dry_run import (
    DRAFT_NAMES,
    SCHEMA_VERSION,
    build_temporary_dry_run_zip,
)
from kcg_connector.eight_hour_final_artifacts import (
    FINAL_OUTPUTS,
    build_final_report_data,
    render_final_report,
)


ROOT = Path(__file__).resolve().parents[3]


def _data():
    return build_final_report_data(
        repository_root=ROOT,
        work_queue_path="artifacts/agent_control/WORK_QUEUE.yaml",
        master_state_path="artifacts/agent_control/MASTER_STATE.json",
        blocker_ledger_path="artifacts/agent_control/BLOCKER_LEDGER.jsonl",
        readiness_report_path="artifacts/agent_control/tasks/EIGHT-HOUR-G2-DYNAMIC-READINESS-DAG/DYNAMIC_READINESS_DAG.json",
        preflight_path="artifacts/agent_control/tasks/EIGHT-HOUR-G3-FINAL-REVIEW-PREFLIGHT/FINAL_REVIEW_PREFLIGHT.json",
        closeout_at_utc="2026-08-18T01:10:00Z",
    )


def test_current_dry_run_data_is_non_dynamic_and_home():
    data = _data()
    assert data["schema_version"].endswith("_v1")
    assert data["metrics"]["dynamic_passed_task_count"] == 0
    assert data["metrics"]["current_frontier_state"] == "HOME"
    assert data["metrics"]["peak_vram_mib"] is None


def test_draft_report_has_table_and_fourteen_items():
    report = render_final_report(_data())
    assert report.startswith("# 八小时自主任务最终报告\n\n| 任务 |")
    assert "## 十四项最终汇总" in report
    assert all(f"{number}." in report for number in range(1, 15))


def test_formal_outputs_are_absent_before_dry_run():
    assert all(not (ROOT / relative).exists() for relative in FINAL_OUTPUTS)


def test_draft_names_do_not_overlap_formal_outputs():
    assert len(DRAFT_NAMES) == len(set(DRAFT_NAMES)) == 6
    assert not {path.name for path in FINAL_OUTPUTS} & set(DRAFT_NAMES)


def test_temporary_zip_verifies_and_is_removed(tmp_path):
    drafts = []
    for index in range(3):
        path = ROOT / f"artifacts/agent_control/tasks/EIGHT-HOUR-G5-CLOSEOUT-DRY-RUN/.test_draft_{index}"
        path.write_text(f"draft {index}\n", encoding="utf-8")
        drafts.append(path)
    try:
        receipt = build_temporary_dry_run_zip(
            root=ROOT,
            source_paths=_data()["package_source_paths"][:10],
            draft_paths=drafts,
        )
        assert receipt["sha256_verified"] is True
        assert receipt["temporary_bundle_removed"] is True
        assert receipt["member_count"] == receipt["manifest_entry_count"] + 1
    finally:
        for path in drafts:
            path.unlink(missing_ok=True)


def test_temporary_zip_rejects_outside_draft(tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("outside")
    with pytest.raises(ValueError, match="outside repository"):
        build_temporary_dry_run_zip(
            root=ROOT,
            source_paths=[],
            draft_paths=[outside],
        )


def test_package_sources_have_no_duplicates_and_are_present():
    paths = _data()["package_source_paths"]
    assert len(paths) == len(set(paths))
    assert len(paths) > 190
    assert all((ROOT / path).is_file() for path in paths)


def test_schema_and_boundary_constants_are_explicit():
    assert SCHEMA_VERSION == "kcg_eight_hour_closeout_dry_run_v1"
    data = _data()
    assert data["simulation_started_by_builder"] is False
    assert data["robot_commands_emitted_by_builder"] == 0
    assert data["assembly_success_claimed"] is False
    assert data["formal_r12_generated"] is False
