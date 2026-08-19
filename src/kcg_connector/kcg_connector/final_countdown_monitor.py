"""Fail-closed stable-source monitor for the final pre-deadline countdown."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

from .final_review_preflight import _load_mapping, _resolve_inside, _sha256
from .predeadline_source_monitor import build_predeadline_baseline
from .stable_closeout_integrity import MUTABLE_CONTROL_PATHS, check_stable_manifest


SCHEMA_VERSION = "kcg_eight_hour_final_countdown_monitor_v1"
ALLOWED_PREDECESSORS = {
    "EIGHT-HOUR-G12-PREDEADLINE-SOURCE-MONITOR":
        "EIGHT-HOUR-G11-PREDEADLINE-SOURCE-MONITOR",
    "EIGHT-HOUR-G13-FINAL-COUNTDOWN-MONITOR":
        "EIGHT-HOUR-G12-PREDEADLINE-SOURCE-MONITOR",
}
CODE_SELF_PATHS = (
    "src/kcg_connector/kcg_connector/final_countdown_monitor.py",
    "src/kcg_connector/test/test_final_countdown_monitor.py",
)


def _digest(rows: Sequence[Mapping[str, Any]]) -> str:
    encoded = json.dumps(
        list(rows), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _task_self_paths(task_id: str) -> tuple[str, ...]:
    if task_id not in ALLOWED_PREDECESSORS:
        raise ValueError(f"countdown task is not authorized: {task_id}")
    return CODE_SELF_PATHS + (
        f"artifacts/agent_control/tasks/{task_id}/RUN_PLAN.json",
    )


def _predecessor_paths(
    predecessor_result: Mapping[str, Any], predecessor_result_path: str
) -> list[str]:
    paths = {predecessor_result_path}
    evidence = predecessor_result.get("evidence")
    code_files = predecessor_result.get("code_files")
    if not isinstance(evidence, list) or not isinstance(code_files, list):
        raise ValueError("predecessor result lacks evidence or code_files")
    for relative in evidence:
        if not isinstance(relative, str):
            raise ValueError("predecessor evidence path must be a string")
        paths.add(relative)
    for row in code_files:
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
            raise ValueError("predecessor code_files row is invalid")
        paths.add(row["path"])
    return sorted(paths)


def build_countdown_baseline(
    *,
    repository_root: str | Path,
    task_id: str,
    predecessor_task_result_path: str | Path,
    work_queue_path: str | Path,
    master_state_path: str | Path,
    blocker_ledger_path: str | Path,
    readiness_report_path: str | Path,
    preflight_path: str | Path,
    generated_at_utc: str,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    expected_predecessor = ALLOWED_PREDECESSORS.get(task_id)
    if expected_predecessor is None:
        raise ValueError(f"countdown task is not authorized: {task_id}")
    predecessor_path = _resolve_inside(
        root, predecessor_task_result_path, "countdown predecessor task result"
    )
    predecessor = _load_mapping(predecessor_path, "countdown predecessor task result")
    if predecessor.get("task_id") != expected_predecessor:
        raise ValueError("countdown predecessor task_id mismatch")
    if predecessor.get("outcome") != "OFFLINE_PASS":
        raise ValueError("countdown predecessor did not close OFFLINE_PASS")

    base = build_predeadline_baseline(
        repository_root=root,
        work_queue_path=work_queue_path,
        master_state_path=master_state_path,
        blocker_ledger_path=blocker_ledger_path,
        readiness_report_path=readiness_report_path,
        preflight_path=preflight_path,
        generated_at_utc=generated_at_utc,
    )
    stable = base["stable_manifest"]
    rows_by_path = {row["path"]: dict(row) for row in stable["stable_sources"]}
    predecessor_relative = predecessor_path.relative_to(root).as_posix()
    closure_paths = _predecessor_paths(predecessor, predecessor_relative)
    self_paths = _task_self_paths(task_id)
    for relative in sorted(set(closure_paths) | set(self_paths)):
        path = _resolve_inside(root, relative, "countdown stable source")
        if not path.is_file():
            raise ValueError(f"countdown stable source missing: {relative}")
        rows_by_path[relative] = {
            "path": relative,
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
    rows = [rows_by_path[path] for path in sorted(rows_by_path)]
    stable["stable_sources"] = rows
    stable["stable_source_count"] = len(rows)
    stable["stable_manifest_digest"] = _digest(rows)
    missing_from_baseline = [path for path in closure_paths if path not in rows_by_path]
    if missing_from_baseline:
        raise ValueError(
            f"predecessor closure missing from baseline: {missing_from_baseline}"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": task_id,
        "generated_at_utc": generated_at_utc,
        "result": "BASELINE_CREATED",
        "predecessor_task_id": expected_predecessor,
        "predecessor_task_result_path": predecessor_relative,
        "predecessor_closure_source_count": len(closure_paths),
        "predecessor_missing_from_baseline_count": 0,
        "predecessor_closure_sources": [rows_by_path[path] for path in closure_paths],
        "stable_manifest": stable,
        "stable_source_count": len(rows),
        "stable_manifest_digest": stable["stable_manifest_digest"],
        "mutable_control_path_count": len(MUTABLE_CONTROL_PATHS),
        "mutable_control_paths_excluded": sorted(MUTABLE_CONTROL_PATHS),
        "self_source_count": len(self_paths),
        "self_sources": list(self_paths),
        "formal_final_output_count": 0,
        "dynamic_passed_task_count": 0,
        "current_frontier_state": "HOME",
        "simulation_started": False,
        "robot_commands_emitted": 0,
        "assembly_success_claimed": False,
        "formal_r12_generated": False,
        "control_authorized": False,
        "hardware_authorized": False,
    }


def check_countdown_baseline(
    *,
    repository_root: str | Path,
    baseline: Mapping[str, Any],
    checked_at_utc: str,
) -> dict[str, Any]:
    if baseline.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("countdown baseline schema mismatch")
    task_id = baseline.get("task_id")
    if task_id not in ALLOWED_PREDECESSORS:
        raise ValueError("countdown baseline task_id is not authorized")
    stable = baseline.get("stable_manifest")
    if not isinstance(stable, Mapping):
        raise ValueError("countdown baseline lacks stable manifest")
    check = check_stable_manifest(
        repository_root=repository_root,
        baseline=stable,
        checked_at_utc=checked_at_utc,
    )
    passed = check["result"] == "PASS" and check["formal_final_output_count"] == 0
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": task_id,
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


def monitor_countdown_sources(
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
        raise FileExistsError("countdown heartbeat output is immutable")
    expected = duration_seconds // interval_seconds
    started_at_utc = utc_now_fn()
    start = monotonic_fn()
    checks: list[dict[str, Any]] = []
    for index in range(1, expected + 1):
        target = start + index * interval_seconds
        remaining = target - monotonic_fn()
        if remaining > 0:
            sleep_fn(remaining)
        check = check_countdown_baseline(
            repository_root=repository_root,
            baseline=baseline,
            checked_at_utc=utc_now_fn(),
        )
        row = {
            "checkpoint_index": index,
            "expected_checkpoint_count": expected,
            "elapsed_seconds": monotonic_fn() - start,
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
        "task_id": baseline.get("task_id"),
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
        "predecessor_closure_source_count": baseline.get(
            "predecessor_closure_source_count"
        ),
        "maximum_missing_source_count": max(
            (row["missing_source_count"] for row in checks), default=0
        ),
        "maximum_drift_source_count": max(
            (row["drift_source_count"] for row in checks), default=0
        ),
        "maximum_formal_final_output_count": max(
            (row["formal_final_output_count"] for row in checks), default=0
        ),
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
        raise FileExistsError("countdown-monitor output is immutable")
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
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--predecessor-task-result")
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
    if args.task_id not in ALLOWED_PREDECESSORS:
        raise ValueError(f"countdown task is not authorized: {args.task_id}")
    output = _resolve_inside(root, args.output, "countdown output")
    output_root = (root / "artifacts/agent_control/tasks" / args.task_id).resolve()
    if not output.is_relative_to(output_root):
        raise PermissionError("countdown outputs must remain inside current task directory")
    if args.create_baseline:
        required = (
            args.predecessor_task_result,
            args.work_queue,
            args.master_state,
            args.blocker_ledger,
            args.readiness_report,
            args.preflight,
            args.generated_at_utc,
        )
        if any(value is None for value in required):
            raise ValueError("baseline mode requires all source arguments")
        value = build_countdown_baseline(
            repository_root=root,
            task_id=args.task_id,
            predecessor_task_result_path=args.predecessor_task_result,
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
        baseline_path = _resolve_inside(root, args.baseline, "countdown baseline")
        heartbeat = _resolve_inside(root, args.heartbeat, "countdown heartbeat")
        if not heartbeat.is_relative_to(output_root):
            raise PermissionError("countdown heartbeat must remain inside task directory")
        baseline = _load_mapping(baseline_path, "countdown baseline")
        if baseline.get("task_id") != args.task_id:
            raise ValueError("CLI task_id and baseline task_id mismatch")
        value = monitor_countdown_sources(
            repository_root=root,
            baseline=baseline,
            duration_seconds=args.duration_seconds,
            interval_seconds=args.interval_seconds,
            heartbeat_path=heartbeat,
        )
        if value["result"] != "OFFLINE_PASS":
            raise SystemExit("countdown stable-source monitor failed")
    write_new_json(value, output)


if __name__ == "__main__":
    main()


__all__ = [
    "ALLOWED_PREDECESSORS",
    "CODE_SELF_PATHS",
    "SCHEMA_VERSION",
    "build_countdown_baseline",
    "check_countdown_baseline",
    "monitor_countdown_sources",
    "write_new_json",
]
