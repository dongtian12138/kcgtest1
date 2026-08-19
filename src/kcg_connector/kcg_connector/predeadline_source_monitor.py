"""Fail-closed wall-clock monitor for stable pre-deadline package sources."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

from .eight_hour_final_artifacts import FINAL_OUTPUTS
from .final_review_preflight import _load_mapping, _resolve_inside, _sha256
from .stable_closeout_integrity import (
    MUTABLE_CONTROL_PATHS,
    build_stable_manifest,
    check_stable_manifest,
)


SCHEMA_VERSION = "kcg_eight_hour_predeadline_source_monitor_v1"
TASK_ID = "EIGHT-HOUR-G11-PREDEADLINE-SOURCE-MONITOR"
SELF_PATHS = (
    "src/kcg_connector/kcg_connector/predeadline_source_monitor.py",
    "src/kcg_connector/test/test_predeadline_source_monitor.py",
    "artifacts/agent_control/tasks/EIGHT-HOUR-G11-PREDEADLINE-SOURCE-MONITOR/RUN_PLAN.json",
)


def _digest(rows: Sequence[Mapping[str, Any]]) -> str:
    value = json.dumps(
        list(rows), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_predeadline_baseline(
    *,
    repository_root: str | Path,
    work_queue_path: str | Path,
    master_state_path: str | Path,
    blocker_ledger_path: str | Path,
    readiness_report_path: str | Path,
    preflight_path: str | Path,
    generated_at_utc: str,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    stable = build_stable_manifest(
        repository_root=root,
        work_queue_path=work_queue_path,
        master_state_path=master_state_path,
        blocker_ledger_path=blocker_ledger_path,
        readiness_report_path=readiness_report_path,
        preflight_path=preflight_path,
        generated_at_utc=generated_at_utc,
    )
    rows_by_path = {row["path"]: dict(row) for row in stable["stable_sources"]}
    for relative in SELF_PATHS:
        path = _resolve_inside(root, relative, "G11 self source")
        if not path.is_file():
            raise ValueError(f"G11 self source missing: {relative}")
        rows_by_path[relative] = {
            "path": relative,
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
    rows = [rows_by_path[path] for path in sorted(rows_by_path)]
    stable["stable_sources"] = rows
    stable["stable_source_count"] = len(rows)
    stable["stable_manifest_digest"] = _digest(rows)
    formal_outputs = [relative for relative in FINAL_OUTPUTS if (root / relative).exists()]
    if formal_outputs:
        raise ValueError(f"formal outputs appeared before deadline: {formal_outputs}")
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "generated_at_utc": generated_at_utc,
        "result": "BASELINE_CREATED",
        "stable_manifest": stable,
        "stable_source_count": len(rows),
        "stable_manifest_digest": stable["stable_manifest_digest"],
        "mutable_control_path_count": len(MUTABLE_CONTROL_PATHS),
        "mutable_control_paths_excluded": sorted(MUTABLE_CONTROL_PATHS),
        "self_source_count": len(SELF_PATHS),
        "self_sources": list(SELF_PATHS),
        "formal_final_output_count": 0,
        "simulation_started": False,
        "robot_commands_emitted": 0,
        "assembly_success_claimed": False,
        "formal_r12_generated": False,
        "control_authorized": False,
        "hardware_authorized": False,
    }


def check_predeadline_baseline(
    *, repository_root: str | Path, baseline: Mapping[str, Any], checked_at_utc: str
) -> dict[str, Any]:
    if baseline.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("predeadline baseline schema mismatch")
    stable = baseline.get("stable_manifest")
    if not isinstance(stable, Mapping):
        raise ValueError("predeadline baseline lacks stable manifest")
    check = check_stable_manifest(
        repository_root=repository_root,
        baseline=stable,
        checked_at_utc=checked_at_utc,
    )
    passed = check["result"] == "PASS" and check["formal_final_output_count"] == 0
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "checked_at_utc": checked_at_utc,
        "result": "PASS" if passed else "FAIL",
        "baseline_source_count": check["baseline_source_count"],
        "checked_source_count": check["checked_source_count"],
        "missing_source_count": check["missing_source_count"],
        "missing_sources": check["missing_sources"],
        "drift_source_count": check["drift_source_count"],
        "drift_sources": check["drift_sources"],
        "baseline_manifest_digest": check["baseline_manifest_digest"],
        "current_manifest_digest": check["current_manifest_digest"],
        "formal_final_output_count": check["formal_final_output_count"],
        "simulation_started": False,
        "robot_commands_emitted": 0,
    }


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()


def monitor_predeadline_sources(
    *,
    repository_root: str | Path,
    baseline: Mapping[str, Any],
    duration_seconds: int,
    interval_seconds: int,
    heartbeat_path: str | Path,
    monotonic_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
    utc_now_fn: Callable[[], str] = _utc_now,
) -> dict[str, Any]:
    if duration_seconds <= 0 or interval_seconds <= 0:
        raise ValueError("duration and interval must be positive")
    if duration_seconds > 1500:
        raise ValueError("monitor duration exceeds single-command limit")
    if duration_seconds % interval_seconds:
        raise ValueError("duration must be an exact multiple of interval")
    heartbeat = Path(heartbeat_path)
    if heartbeat.exists():
        raise FileExistsError("heartbeat output is immutable")
    expected = duration_seconds // interval_seconds
    started_at_utc = utc_now_fn()
    start = monotonic_fn()
    checks: list[dict[str, Any]] = []
    for index in range(1, expected + 1):
        target = start + index * interval_seconds
        remaining = target - monotonic_fn()
        if remaining > 0:
            sleep_fn(remaining)
        elapsed = monotonic_fn() - start
        check = check_predeadline_baseline(
            repository_root=repository_root,
            baseline=baseline,
            checked_at_utc=utc_now_fn(),
        )
        row = {
            "checkpoint_index": index,
            "expected_checkpoint_count": expected,
            "elapsed_seconds": elapsed,
            **check,
        }
        _append_jsonl(heartbeat, row)
        checks.append(row)
        if check["result"] != "PASS":
            break
    elapsed = monotonic_fn() - start
    all_passed = (
        len(checks) == expected
        and all(row["result"] == "PASS" for row in checks)
        and elapsed >= duration_seconds
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "started_at_utc": started_at_utc,
        "completed_at_utc": utc_now_fn(),
        "result": "OFFLINE_PASS" if all_passed else "FAIL",
        "requested_duration_seconds": duration_seconds,
        "heartbeat_interval_seconds": interval_seconds,
        "elapsed_seconds": elapsed,
        "expected_checkpoint_count": expected,
        "completed_checkpoint_count": len(checks),
        "passing_checkpoint_count": sum(row["result"] == "PASS" for row in checks),
        "baseline_source_count": baseline.get("stable_source_count"),
        "baseline_manifest_digest": baseline.get("stable_manifest_digest"),
        "maximum_missing_source_count": max((row["missing_source_count"] for row in checks), default=0),
        "maximum_drift_source_count": max((row["drift_source_count"] for row in checks), default=0),
        "maximum_formal_final_output_count": max((row["formal_final_output_count"] for row in checks), default=0),
        "dynamic_passed_task_count": 0,
        "current_frontier_state": "HOME",
        "simulation_started": False,
        "robot_commands_emitted": 0,
        "assembly_success_claimed": False,
        "formal_r12_generated": False,
        "control_authorized": False,
        "hardware_authorized": False,
    }


def write_new_json(value: Mapping[str, Any], output: str | Path) -> None:
    path = Path(output)
    if path.exists():
        raise FileExistsError("predeadline-monitor output is immutable")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--create-baseline", action="store_true")
    mode.add_argument("--monitor", action="store_true")
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--work-queue")
    parser.add_argument("--master-state")
    parser.add_argument("--blocker-ledger")
    parser.add_argument("--readiness-report")
    parser.add_argument("--preflight")
    parser.add_argument("--generated-at-utc")
    parser.add_argument("--baseline")
    parser.add_argument("--duration-seconds", type=int, default=1200)
    parser.add_argument("--interval-seconds", type=int, default=60)
    parser.add_argument("--heartbeat")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    root = Path(args.repository_root).resolve()
    output = _resolve_inside(root, args.output, "G11 output")
    output_root = (root / "artifacts/agent_control/tasks" / TASK_ID).resolve()
    if not output.is_relative_to(output_root):
        raise PermissionError("G11 outputs must remain inside the G11 task directory")
    if args.create_baseline:
        required = (
            args.work_queue,
            args.master_state,
            args.blocker_ledger,
            args.readiness_report,
            args.preflight,
            args.generated_at_utc,
        )
        if any(value is None for value in required):
            raise ValueError("baseline mode requires all source arguments")
        value = build_predeadline_baseline(
            repository_root=root,
            work_queue_path=args.work_queue,
            master_state_path=args.master_state,
            blocker_ledger_path=args.blocker_ledger,
            readiness_report_path=args.readiness_report,
            preflight_path=args.preflight,
            generated_at_utc=args.generated_at_utc,
        )
    else:
        if args.baseline is None or args.heartbeat is None:
            raise ValueError("monitor mode requires baseline and heartbeat")
        baseline_path = _resolve_inside(root, args.baseline, "G11 baseline")
        heartbeat = _resolve_inside(root, args.heartbeat, "G11 heartbeat")
        if not heartbeat.is_relative_to(output_root):
            raise PermissionError("G11 heartbeat must remain inside task directory")
        baseline = _load_mapping(baseline_path, "G11 baseline")
        value = monitor_predeadline_sources(
            repository_root=root,
            baseline=baseline,
            duration_seconds=args.duration_seconds,
            interval_seconds=args.interval_seconds,
            heartbeat_path=heartbeat,
        )
        if value["result"] != "OFFLINE_PASS":
            raise SystemExit("predeadline stable-source monitor failed")
    write_new_json(value, output)


if __name__ == "__main__":
    main()


__all__ = [
    "SCHEMA_VERSION",
    "SELF_PATHS",
    "TASK_ID",
    "build_predeadline_baseline",
    "check_predeadline_baseline",
    "monitor_predeadline_sources",
    "write_new_json",
]
