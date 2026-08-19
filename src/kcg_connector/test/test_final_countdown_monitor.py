from __future__ import annotations

import copy
from pathlib import Path

import pytest

from kcg_connector.final_countdown_monitor import (
    ALLOWED_PREDECESSORS,
    CODE_SELF_PATHS,
    SCHEMA_VERSION,
    build_countdown_baseline,
    check_countdown_baseline,
    monitor_countdown_sources,
    write_new_json,
)
from kcg_connector.stable_closeout_integrity import MUTABLE_CONTROL_PATHS


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "EIGHT-HOUR-G12-PREDEADLINE-SOURCE-MONITOR"
PREDECESSOR = (
    "artifacts/agent_control/tasks/"
    "EIGHT-HOUR-G11-PREDEADLINE-SOURCE-MONITOR/TASK_RESULT.json"
)


def _baseline():
    return build_countdown_baseline(
        repository_root=ROOT,
        task_id=TASK_ID,
        predecessor_task_result_path=PREDECESSOR,
        work_queue_path="artifacts/agent_control/WORK_QUEUE.yaml",
        master_state_path="artifacts/agent_control/MASTER_STATE.json",
        blocker_ledger_path="artifacts/agent_control/BLOCKER_LEDGER.jsonl",
        readiness_report_path=(
            "artifacts/agent_control/tasks/"
            "EIGHT-HOUR-G2-DYNAMIC-READINESS-DAG/DYNAMIC_READINESS_DAG.json"
        ),
        preflight_path=(
            "artifacts/agent_control/tasks/"
            "EIGHT-HOUR-G3-FINAL-REVIEW-PREFLIGHT/FINAL_REVIEW_PREFLIGHT.json"
        ),
        generated_at_utc="2026-08-18T02:24:00Z",
    )


class _FakeClock:
    def __init__(self):
        self.value = 0.0

    def monotonic(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


def test_baseline_is_large_schema_bound_and_closes_predecessor():
    baseline = _baseline()
    assert baseline["schema_version"] == SCHEMA_VERSION
    assert baseline["task_id"] == TASK_ID
    assert baseline["stable_source_count"] > 270
    assert len(baseline["stable_manifest_digest"]) == 64
    assert baseline["predecessor_closure_source_count"] > 0
    assert baseline["predecessor_missing_from_baseline_count"] == 0


def test_only_g12_and_g13_are_authorized_with_exact_predecessors():
    assert set(ALLOWED_PREDECESSORS) == {
        "EIGHT-HOUR-G12-PREDEADLINE-SOURCE-MONITOR",
        "EIGHT-HOUR-G13-FINAL-COUNTDOWN-MONITOR",
    }
    with pytest.raises(ValueError, match="not authorized"):
        build_countdown_baseline(
            repository_root=ROOT,
            task_id="EIGHT-HOUR-G14-UNAUTHORIZED",
            predecessor_task_result_path=PREDECESSOR,
            work_queue_path="artifacts/agent_control/WORK_QUEUE.yaml",
            master_state_path="artifacts/agent_control/MASTER_STATE.json",
            blocker_ledger_path="artifacts/agent_control/BLOCKER_LEDGER.jsonl",
            readiness_report_path=(
                "artifacts/agent_control/tasks/"
                "EIGHT-HOUR-G2-DYNAMIC-READINESS-DAG/DYNAMIC_READINESS_DAG.json"
            ),
            preflight_path=(
                "artifacts/agent_control/tasks/"
                "EIGHT-HOUR-G3-FINAL-REVIEW-PREFLIGHT/FINAL_REVIEW_PREFLIGHT.json"
            ),
            generated_at_utc="2026-08-18T02:24:00Z",
        )


def test_mutable_controls_excluded_and_current_self_sources_included():
    baseline = _baseline()
    paths = {row["path"] for row in baseline["stable_manifest"]["stable_sources"]}
    assert not paths & MUTABLE_CONTROL_PATHS
    assert set(CODE_SELF_PATHS) <= paths
    assert f"artifacts/agent_control/tasks/{TASK_ID}/RUN_PLAN.json" in paths


def test_immediate_check_has_zero_missing_drift_and_formal_outputs():
    check = check_countdown_baseline(
        repository_root=ROOT,
        baseline=_baseline(),
        checked_at_utc="2026-08-18T02:25:00Z",
    )
    assert check["result"] == "PASS"
    assert check["missing_source_count"] == 0
    assert check["drift_source_count"] == 0
    assert check["formal_final_output_count"] == 0


def test_tampered_digest_fails_closed_without_source_edit():
    baseline = copy.deepcopy(_baseline())
    baseline["stable_manifest"]["stable_sources"][0]["sha256"] = "0" * 64
    check = check_countdown_baseline(
        repository_root=ROOT,
        baseline=baseline,
        checked_at_utc="2026-08-18T02:25:00Z",
    )
    assert check["result"] == "FAIL"
    assert check["drift_source_count"] == 1


def test_fake_clock_monitor_has_exact_checkpoints(tmp_path):
    clock = _FakeClock()
    summary = monitor_countdown_sources(
        repository_root=ROOT,
        baseline=_baseline(),
        duration_seconds=3,
        interval_seconds=1,
        heartbeat_path=tmp_path / "heartbeat.jsonl",
        monotonic_fn=clock.monotonic,
        sleep_fn=clock.sleep,
        utc_now_fn=lambda: "2026-08-18T02:26:00Z",
    )
    assert summary["result"] == "OFFLINE_PASS"
    assert summary["completed_checkpoint_count"] == 3
    assert summary["elapsed_seconds"] == 3
    assert len((tmp_path / "heartbeat.jsonl").read_text().splitlines()) == 3


def test_monitor_stops_on_first_failed_checkpoint(tmp_path):
    baseline = copy.deepcopy(_baseline())
    baseline["stable_manifest"]["stable_sources"][0]["size_bytes"] += 1
    clock = _FakeClock()
    summary = monitor_countdown_sources(
        repository_root=ROOT,
        baseline=baseline,
        duration_seconds=3,
        interval_seconds=1,
        heartbeat_path=tmp_path / "heartbeat.jsonl",
        monotonic_fn=clock.monotonic,
        sleep_fn=clock.sleep,
        utc_now_fn=lambda: "2026-08-18T02:26:00Z",
    )
    assert summary["result"] == "FAIL"
    assert summary["completed_checkpoint_count"] == 1
    assert summary["maximum_drift_source_count"] == 1


def test_duration_is_positive_bounded_and_divisible(tmp_path):
    for duration, interval, message in ((0, 1, "positive"), (1501, 1, "exceeds"), (5, 2, "exact multiple")):
        with pytest.raises(ValueError, match=message):
            monitor_countdown_sources(
                repository_root=ROOT,
                baseline=_baseline(),
                duration_seconds=duration,
                interval_seconds=interval,
                heartbeat_path=tmp_path / f"{duration}-{interval}.jsonl",
            )


def test_outputs_are_immutable(tmp_path):
    heartbeat = tmp_path / "heartbeat.jsonl"
    heartbeat.write_text("existing\n")
    with pytest.raises(FileExistsError, match="immutable"):
        monitor_countdown_sources(
            repository_root=ROOT,
            baseline=_baseline(),
            duration_seconds=1,
            interval_seconds=1,
            heartbeat_path=heartbeat,
        )
    output = tmp_path / "summary.json"
    write_new_json({"result": "fixture"}, output)
    with pytest.raises(FileExistsError, match="immutable"):
        write_new_json({"result": "fixture"}, output)


def test_no_dynamic_acceptance_or_hardware_claim():
    baseline = _baseline()
    assert baseline["dynamic_passed_task_count"] == 0
    assert baseline["current_frontier_state"] == "HOME"
    assert baseline["simulation_started"] is False
    assert baseline["robot_commands_emitted"] == 0
    assert baseline["assembly_success_claimed"] is False
    assert baseline["formal_r12_generated"] is False
    assert baseline["control_authorized"] is False
    assert baseline["hardware_authorized"] is False
