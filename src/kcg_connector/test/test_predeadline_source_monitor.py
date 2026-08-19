from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from kcg_connector.predeadline_source_monitor import (
    SCHEMA_VERSION,
    SELF_PATHS,
    build_predeadline_baseline,
    check_predeadline_baseline,
    monitor_predeadline_sources,
    write_new_json,
)
from kcg_connector.stable_closeout_integrity import MUTABLE_CONTROL_PATHS


ROOT = Path(__file__).resolve().parents[3]


def _baseline():
    return build_predeadline_baseline(
        repository_root=ROOT,
        work_queue_path="artifacts/agent_control/WORK_QUEUE.yaml",
        master_state_path="artifacts/agent_control/MASTER_STATE.json",
        blocker_ledger_path="artifacts/agent_control/BLOCKER_LEDGER.jsonl",
        readiness_report_path="artifacts/agent_control/tasks/EIGHT-HOUR-G2-DYNAMIC-READINESS-DAG/DYNAMIC_READINESS_DAG.json",
        preflight_path="artifacts/agent_control/tasks/EIGHT-HOUR-G3-FINAL-REVIEW-PREFLIGHT/FINAL_REVIEW_PREFLIGHT.json",
        generated_at_utc="2026-08-18T01:50:00Z",
    )


class _FakeClock:
    def __init__(self):
        self.value = 0.0

    def monotonic(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


def test_baseline_is_large_direct_and_schema_bound():
    baseline = _baseline()
    assert baseline["schema_version"] == SCHEMA_VERSION
    assert baseline["result"] == "BASELINE_CREATED"
    assert baseline["stable_source_count"] > 250
    assert len(baseline["stable_manifest_digest"]) == 64


def test_mutable_controls_excluded_and_self_sources_included():
    baseline = _baseline()
    paths = {row["path"] for row in baseline["stable_manifest"]["stable_sources"]}
    assert not paths & MUTABLE_CONTROL_PATHS
    assert set(SELF_PATHS) <= paths
    assert baseline["mutable_control_path_count"] == 9


def test_immediate_check_has_zero_missing_drift_and_formal_outputs():
    check = check_predeadline_baseline(
        repository_root=ROOT,
        baseline=_baseline(),
        checked_at_utc="2026-08-18T01:51:00Z",
    )
    assert check["result"] == "PASS"
    assert check["missing_source_count"] == 0
    assert check["drift_source_count"] == 0
    assert check["formal_final_output_count"] == 0


def test_tampered_digest_fails_closed_without_editing_source():
    baseline = copy.deepcopy(_baseline())
    baseline["stable_manifest"]["stable_sources"][0]["sha256"] = "0" * 64
    check = check_predeadline_baseline(
        repository_root=ROOT,
        baseline=baseline,
        checked_at_utc="2026-08-18T01:51:00Z",
    )
    assert check["result"] == "FAIL"
    assert check["drift_source_count"] == 1


def test_fake_clock_monitor_has_exact_checkpoints(tmp_path):
    clock = _FakeClock()
    summary = monitor_predeadline_sources(
        repository_root=ROOT,
        baseline=_baseline(),
        duration_seconds=3,
        interval_seconds=1,
        heartbeat_path=tmp_path / "heartbeat.jsonl",
        monotonic_fn=clock.monotonic,
        sleep_fn=clock.sleep,
        utc_now_fn=lambda: "2026-08-18T01:52:00Z",
    )
    assert summary["result"] == "OFFLINE_PASS"
    assert summary["expected_checkpoint_count"] == 3
    assert summary["completed_checkpoint_count"] == 3
    assert summary["elapsed_seconds"] == 3
    assert len((tmp_path / "heartbeat.jsonl").read_text().splitlines()) == 3


def test_monitor_stops_on_first_failed_checkpoint(tmp_path):
    baseline = copy.deepcopy(_baseline())
    baseline["stable_manifest"]["stable_sources"][0]["size_bytes"] += 1
    clock = _FakeClock()
    summary = monitor_predeadline_sources(
        repository_root=ROOT,
        baseline=baseline,
        duration_seconds=3,
        interval_seconds=1,
        heartbeat_path=tmp_path / "heartbeat.jsonl",
        monotonic_fn=clock.monotonic,
        sleep_fn=clock.sleep,
        utc_now_fn=lambda: "2026-08-18T01:52:00Z",
    )
    assert summary["result"] == "FAIL"
    assert summary["completed_checkpoint_count"] == 1
    assert summary["maximum_drift_source_count"] == 1


def test_duration_must_be_bounded_and_divisible(tmp_path):
    with pytest.raises(ValueError, match="exceeds"):
        monitor_predeadline_sources(
            repository_root=ROOT, baseline=_baseline(), duration_seconds=1501,
            interval_seconds=1, heartbeat_path=tmp_path / "a.jsonl"
        )
    with pytest.raises(ValueError, match="exact multiple"):
        monitor_predeadline_sources(
            repository_root=ROOT, baseline=_baseline(), duration_seconds=5,
            interval_seconds=2, heartbeat_path=tmp_path / "b.jsonl"
        )


def test_heartbeat_and_summary_outputs_are_immutable(tmp_path):
    heartbeat = tmp_path / "heartbeat.jsonl"
    heartbeat.write_text("existing\n")
    with pytest.raises(FileExistsError, match="immutable"):
        monitor_predeadline_sources(
            repository_root=ROOT, baseline=_baseline(), duration_seconds=1,
            interval_seconds=1, heartbeat_path=heartbeat
        )
    output = tmp_path / "summary.json"
    write_new_json({"result": "fixture"}, output)
    with pytest.raises(FileExistsError, match="immutable"):
        write_new_json({"result": "fixture"}, output)


def test_baseline_rows_are_unique_existing_relative_paths():
    rows = _baseline()["stable_manifest"]["stable_sources"]
    paths = [row["path"] for row in rows]
    assert len(paths) == len(set(paths))
    assert all(not Path(path).is_absolute() for path in paths)
    assert all((ROOT / path).is_file() for path in paths)


def test_no_dynamic_acceptance_or_hardware_claim():
    baseline = _baseline()
    assert baseline["simulation_started"] is False
    assert baseline["robot_commands_emitted"] == 0
    assert baseline["assembly_success_claimed"] is False
    assert baseline["formal_r12_generated"] is False
    assert baseline["control_authorized"] is False
    assert baseline["hardware_authorized"] is False
