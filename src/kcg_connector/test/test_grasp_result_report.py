from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from kcg_connector.grasp_result_report import (
    TASK_KEYS,
    build_grasp_result_report,
    write_grasp_result_report,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
WORK_QUEUE = REPOSITORY_ROOT / "artifacts/agent_control/WORK_QUEUE.yaml"


def _queue() -> dict:
    return yaml.safe_load(WORK_QUEUE.read_text(encoding="utf-8"))


def _build_fixture(tmp_path: Path, document: dict):
    root = tmp_path / "repository"
    for task in document["groups"]["B"]["tasks"].values():
        value = task.get("evidence")
        if not isinstance(value, str):
            continue
        relative = Path(value)
        if relative.is_absolute():
            continue
        source = (REPOSITORY_ROOT / relative).resolve()
        if not source.is_relative_to(REPOSITORY_ROOT) or not source.is_file():
            continue
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    path = root / "artifacts/agent_control/WORK_QUEUE.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(document, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return build_grasp_result_report(
        repository_root=root,
        work_queue_path=path,
        generated_at_utc="2026-08-17T20:10:00Z",
    )


def _build(path: Path = WORK_QUEUE):
    return build_grasp_result_report(
        repository_root=REPOSITORY_ROOT,
        work_queue_path=path,
        generated_at_utc="2026-08-17T20:10:00Z",
    )


def test_current_queue_reports_real_parked_chain_without_dynamic_claim():
    report = _build()
    assert report["report_tool_status"] == "OFFLINE_PASS"
    assert report["grasp_chain_status"] == "PARKED"
    assert report["tasks"]["B1"]["status"] == "PARKED"
    assert report["tasks"]["B5"]["status"] == "OFFLINE_PASS"
    assert report["dynamic_grasp_pass_claimed"] is False
    assert report["formal_grasp_pass_claimed"] is False
    assert report["assembly_success_claimed"] is False


def test_all_dynamic_measurements_remain_null_without_runtime_evidence():
    report = _build()
    assert report["dynamic_measurements_available"] is False
    assert report["synthetic_measurements_used"] is False
    assert all(value is None for value in report["dynamic_measurements"].values())


def test_each_evidence_path_has_hash_and_matching_status():
    report = _build()
    rows = report["source_manifest"]["task_evidence"]
    assert {row["task"] for row in rows} == {"B1", "B5"}
    assert all(len(row["sha256"]) == 64 for row in rows)
    assert all(row["status"] == report["tasks"][row["task"]]["status"] for row in rows)


def test_dynamic_pass_input_is_rejected_not_preserved(tmp_path):
    queue = _queue()
    queue["groups"]["B"]["tasks"]["B2"]["status"] = "DYNAMIC_PASS"
    with pytest.raises(ValueError, match="cannot be granted or preserved"):
        _build_fixture(tmp_path, queue)


def test_unknown_status_is_rejected(tmp_path):
    queue = _queue()
    queue["groups"]["B"]["tasks"]["B2"]["status"] = "SUCCESS"
    with pytest.raises(ValueError, match="unsupported status"):
        _build_fixture(tmp_path, queue)


def test_missing_evidence_for_offline_pass_is_rejected(tmp_path):
    queue = _queue()
    queue["groups"]["B"]["tasks"]["B5"].pop("evidence")
    with pytest.raises(ValueError, match="requires evidence"):
        _build_fixture(tmp_path, queue)


def test_queue_and_evidence_status_mismatch_is_rejected(tmp_path):
    queue = _queue()
    queue["groups"]["B"]["tasks"]["B5"]["status"] = "STATIC_PASS"
    with pytest.raises(ValueError, match="status mismatch"):
        _build_fixture(tmp_path, queue)


def test_absolute_and_escaping_evidence_paths_are_rejected(tmp_path):
    for value in ("/etc/passwd", "../../etc/passwd"):
        queue = copy.deepcopy(_queue())
        queue["groups"]["B"]["tasks"]["B5"]["evidence"] = value
        with pytest.raises(ValueError, match="evidence path"):
            _build_fixture(tmp_path, queue)


def test_every_required_task_is_present(tmp_path):
    queue = _queue()
    del queue["groups"]["B"]["tasks"]["B3"]
    with pytest.raises(ValueError, match="lacks B3"):
        _build_fixture(tmp_path, queue)


def test_report_write_is_json_safe(tmp_path):
    report = _build()
    output = tmp_path / "report.json"
    write_grasp_result_report(report, output)
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == "kcg_eight_hour_grasp_result_report_v1"
    assert tuple(loaded["tasks"]) == TASK_KEYS
