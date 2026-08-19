from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from kcg_connector.code_test_traceability import (
    DIRECT_TEST_LINK,
    STATIC_VALIDATION_ONLY,
    build_code_test_traceability,
    render_traceability_markdown,
    write_new,
)


ROOT = Path(__file__).resolve().parents[3]


def _report():
    return build_code_test_traceability(
        repository_root=ROOT,
        work_queue_path="artifacts/agent_control/WORK_QUEUE.yaml",
        generated_at_utc="2026-08-18T01:35:00Z",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_current_queue_has_expected_traceability_counts():
    report = _report()
    assert report["result"] == "OFFLINE_PASS"
    assert report["passing_task_count"] == 42
    assert report["production_module_count"] == 35
    assert report["direct_test_link_count"] == 34
    assert report["static_validation_only_count"] == 1


def test_every_production_module_is_hash_verified_and_classified():
    report = _report()
    assert report["missing_file_count"] == 0
    assert report["sha256_drift_count"] == 0
    assert report["unclassified_production_module_count"] == 0
    for row in report["traceability_rows"]:
        path = ROOT / row["production_path"]
        assert path.is_file()
        assert _sha(path) == row["production_sha256"]
        assert row["traceability_mode"] in {DIRECT_TEST_LINK, STATIC_VALIDATION_ONLY}


def test_direct_links_are_same_task_declared_test_files():
    rows = [row for row in _report()["traceability_rows"] if row["traceability_mode"] == DIRECT_TEST_LINK]
    assert len(rows) == 34
    assert all(row["direct_test_files"] for row in rows)
    assert all(not row["static_validation_sources"] for row in rows)
    assert all((ROOT / test["path"]).is_file() for row in rows for test in row["direct_test_files"])


def test_a2_is_the_only_static_validation_only_module():
    rows = [row for row in _report()["traceability_rows"] if row["traceability_mode"] == STATIC_VALIDATION_ONLY]
    assert len(rows) == 1
    assert rows[0]["task_key"] == "A2"
    assert rows[0]["production_path"] == "src/kcg_connector/isaac/d38999_multilayer_nominal_bench.py"
    assert len(rows[0]["static_validation_sources"]) == 3
    assert sum(source["status"] == "PASS" for source in rows[0]["static_validation_sources"]) == 2


def test_tasks_without_declared_production_are_explicit_not_inferred():
    report = _report()
    keys = {row["task_key"] for row in report["tasks_without_declared_production_code"]}
    assert report["tasks_without_declared_production_code_count"] == 7
    assert keys == {"A3", "A4", "B5", "B6", "C1", "D5", "QUEUE_AUDIT"}


def test_no_coverage_dynamic_or_runtime_claims():
    report = _report()
    assert report["coverage_metric"] is None
    assert report["coverage_claimed"] is False
    assert report["dynamic_passed_task_count"] == 0
    assert report["historical_tests_rerun"] is False
    assert report["repository_walk_performed"] is False
    assert report["simulation_started"] is False
    assert report["robot_commands_emitted"] == 0
    assert report["assembly_success_claimed"] is False


def test_declared_production_hash_drift_fails_closed(tmp_path):
    production = tmp_path / "module.py"
    test_file = tmp_path / "test_module.py"
    production.write_text("VALUE = 1\n")
    test_file.write_text("def test_value(): pass\n")
    result = tmp_path / "result.json"
    result.write_text(json.dumps({
        "task_id": "T1",
        "code_files": [
            {"path": str(production.relative_to(tmp_path)), "sha256": "0" * 64},
            {"path": str(test_file.relative_to(tmp_path)), "sha256": _sha(test_file)},
        ],
    }))
    queue = tmp_path / "queue.yaml"
    queue.write_text(yaml.safe_dump({
        "groups": {"X": {"tasks": {"T1": {"status": "OFFLINE_PASS", "evidence": "result.json"}}}},
        "post_queue_tasks": {},
    }))
    with pytest.raises(ValueError, match="sha256 drift"):
        build_code_test_traceability(
            repository_root=tmp_path,
            work_queue_path="queue.yaml",
            generated_at_utc="2026-08-18T01:35:00Z",
        )


def test_production_without_test_or_static_evidence_fails_closed(tmp_path):
    production = tmp_path / "module.py"
    production.write_text("VALUE = 1\n")
    result = tmp_path / "result.json"
    result.write_text(json.dumps({
        "task_id": "T1",
        "code_files": [{"path": "module.py", "sha256": _sha(production)}],
    }))
    queue = tmp_path / "queue.yaml"
    queue.write_text(yaml.safe_dump({
        "groups": {"X": {"tasks": {"T1": {"status": "STATIC_PASS", "evidence": "result.json"}}}},
        "post_queue_tasks": {},
    }))
    with pytest.raises(ValueError, match="lacks test or static evidence"):
        build_code_test_traceability(
            repository_root=tmp_path,
            work_queue_path="queue.yaml",
            generated_at_utc="2026-08-18T01:35:00Z",
        )


def test_markdown_states_traceability_boundary_and_counts():
    markdown = render_traceability_markdown(_report())
    assert markdown.startswith("# 代码测试可追溯审计")
    assert "不代表行、分支或行为覆盖率" in markdown
    assert "- 生产模块：35。" in markdown
    assert "- 直接测试链接：34。" in markdown
    assert "- 仅静态验证：1。" in markdown
    assert "动态通过声明：0" in markdown


def test_outputs_are_immutable(tmp_path):
    output = tmp_path / "report.json"
    write_new(output, "{}\n")
    with pytest.raises(FileExistsError, match="immutable"):
        write_new(output, "{}\n")
