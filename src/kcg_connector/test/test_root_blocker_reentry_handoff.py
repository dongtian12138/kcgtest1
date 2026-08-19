from __future__ import annotations

import json
from pathlib import Path

import pytest

from kcg_connector.root_blocker_reentry_handoff import (
    EXPECTED_CLASSIFICATIONS,
    ROOT_ORDER,
    SCHEMA_VERSION,
    build_root_blocker_handoff,
    render_handoff_markdown,
    write_handoff_pair,
)


ROOT = Path(__file__).resolve().parents[3]


def _build():
    return build_root_blocker_handoff(
        repository_root=ROOT,
        blocker_ledger_path="artifacts/agent_control/BLOCKER_LEDGER.jsonl",
        readiness_report_path="artifacts/agent_control/tasks/EIGHT-HOUR-G2-DYNAMIC-READINESS-DAG/DYNAMIC_READINESS_DAG.json",
        generated_at_utc="2026-08-18T01:15:00Z",
    )


def test_five_root_blockers_and_top_three_are_exact():
    report = _build()
    assert report["schema_version"] == SCHEMA_VERSION
    assert report["root_blocker_count"] == 5
    assert [row["task_key"] for row in report["entries"]] == list(ROOT_ORDER)
    assert report["top_three_priority_root_blockers"] == ["A1", "B1", "C8"]


def test_classifications_match_frozen_evidence():
    for row in _build()["entries"]:
        assert row["classification"] == EXPECTED_CLASSIFICATIONS[row["task_key"]]
        assert len(row["task_result_sha256"]) == 64
        assert (ROOT / row["task_result_path"]).is_file()


def test_every_entry_has_descendants_and_reentry_evidence():
    for row in _build()["entries"]:
        assert row["blocked_descendant_count"] > 0
        assert len(row["blocked_descendants"]) == row["blocked_descendant_count"]
        assert len(row["reentry_prerequisites"]) >= 3
        assert row["next_window_first_evidence_action"]
        assert row["acceptance_proof_required"]


def test_no_execution_or_command_is_authorized_this_window():
    report = _build()
    assert report["current_window_action"] == "NO_REVISIT_OR_EXECUTION"
    assert report["all_execution_authorized_this_window"] is False
    assert report["all_proposed_commands_null"] is True
    assert all(row["execution_authorized_this_window"] is False for row in report["entries"])
    assert all(row["proposed_command"] is None for row in report["entries"])


def test_a1_third_visit_is_explicitly_forbidden():
    a1 = _build()["entries"][0]
    assert a1["task_key"] == "A1"
    assert a1["third_visit_forbidden_this_window"] is True
    assert "不得重复显式contactOffset/restOffset假设" in a1["reentry_prerequisites"]


def test_no_geometry_threshold_or_force_shortcut():
    for row in _build()["entries"]:
        assert row["geometry_change_authorized"] is False
        assert row["threshold_change_authorized"] is False
        assert row["higher_force_or_moment_authorized"] is False
        assert row["revisit_count_incremented"] is False


def test_c8_and_e6_preserve_missing_authority():
    rows = {row["task_key"]: row for row in _build()["entries"]}
    assert "FoundationPose容器或等效官方运行时可用" in rows["C8"]["reentry_prerequisites"]
    assert "输入来自真实RGB-D证据而不是仿真真值" in rows["C8"]["reentry_prerequisites"]
    assert "防松齿绝对相位原点有权威来源或明确等效授权" in rows["E6"]["reentry_prerequisites"]
    assert "不得裁剪" in rows["E6"]["reentry_prerequisites"][2]


def test_markdown_states_all_safety_boundaries():
    markdown = render_handoff_markdown(_build())
    assert markdown.startswith("# 根阻塞重入交接\n\n| 顺序 |")
    assert "本文件不授权本窗口重访" in markdown
    assert "proposed_command` 均为 null" in markdown
    assert "8牛力分量上限" in markdown
    assert "0.30牛米力矩分量上限" in markdown


def test_outputs_are_immutable(tmp_path):
    paths = [tmp_path / "handoff.json", tmp_path / "handoff.md"]
    write_handoff_pair(_build(), *paths)
    assert json.loads(paths[0].read_text())["result"] == "OFFLINE_PASS"
    with pytest.raises(FileExistsError, match="immutable"):
        write_handoff_pair(_build(), *paths)


def test_builder_has_no_dynamic_or_hardware_claim():
    report = _build()
    assert report["simulation_started"] is False
    assert report["robot_commands_emitted"] == 0
    assert report["assembly_success_claimed"] is False
    assert report["formal_r12_generated"] is False
    assert report["control_authorized"] is False
    assert report["hardware_authorized"] is False
