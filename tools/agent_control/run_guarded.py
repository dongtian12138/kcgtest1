#!/usr/bin/env python3
"""TASK-R12-005 bounded diagnostic command runner.

This wrapper enforces the task's count, driver-kind, timeout, heartbeat, and
preflight-record requirements. It deliberately does not interpret physics
success; downstream evidence extraction must do that from raw artifacts.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any


TASK_ID = "TASK-R12-005"
ALLOWED_DRIVERS = {
    "bounded_axial_integral",
    "bounded_axial_force_feedforward",
    "body_push_nut_torque",
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_required(repo: Path) -> tuple[str, dict[str, Any], str]:
    charter_path = repo / "artifacts/agent_control/PROJECT_CHARTER_CN.md"
    state_path = repo / "artifacts/agent_control/MASTER_STATE.json"
    task_path = repo / "artifacts/agent_control/CURRENT_TASK.md"
    charter = charter_path.read_text(encoding="utf-8")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    task = task_path.read_text(encoding="utf-8")
    if state.get("task_id") != TASK_ID or TASK_ID not in charter or TASK_ID not in task:
        raise RuntimeError("控制文件与 TASK-R12-005 不一致")
    return charter, state, task


def write_json_atomic(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def append_jsonl(path: Path, payload: Any) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot(paths: list[Path]) -> dict[str, str]:
    return {str(path): sha256_file(path) for path in paths if path.is_file()}


def update_status(status_path: Path, lines: list[str]) -> None:
    with status_path.open("a", encoding="utf-8") as stream:
        stream.write("\n## 运行守卫更新\n\n")
        for line in lines:
            stream.write(f"- {line}\n")


def terminate_group(process: subprocess.Popen[Any]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--hypothesis-id", required=True)
    parser.add_argument("--driver-kind", required=True, choices=sorted(ALLOWED_DRIVERS))
    parser.add_argument("--unique-change", required=True)
    parser.add_argument("--expected-result", required=True)
    parser.add_argument("--failure-criterion", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=1500)
    parser.add_argument("--protected-path", action="append", type=Path, default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("必须在 -- 后提供命令")
    if not 1 <= args.timeout_seconds <= 1500:
        parser.error("timeout 必须在 1..1500 秒内")
    return args


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    _, state, _ = read_required(repo)
    control = repo / "artifacts/agent_control"
    records = control / "runs"
    records.mkdir(parents=True, exist_ok=True)
    existing = []
    for path in records.glob("run_*.json"):
        try:
            existing.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            raise RuntimeError(f"无法验证既有运行记录：{path}")
    started = [item for item in existing if item.get("task_id") == TASK_ID]
    if len(started) >= int(state["diagnostic_run_limit"]):
        raise RuntimeError("诊断运行总数已达到上限")
    if any(item.get("driver_kind") == args.driver_kind for item in started):
        raise RuntimeError(f"驱动类型 {args.driver_kind} 已运行过")

    protected = [(repo / path).resolve() if not path.is_absolute() else path.resolve() for path in args.protected_path]
    before = snapshot(protected)
    run_number = len(started) + 1
    started_at = utc_now()
    slug = f"run_{run_number:02d}_{args.driver_kind}"
    record_path = records / f"{slug}.json"
    log_path = records / f"{slug}.log"
    record: dict[str, Any] = {
        "task_id": TASK_ID,
        "run_number": run_number,
        "hypothesis_id": args.hypothesis_id,
        "driver_kind": args.driver_kind,
        "unique_change": args.unique_change,
        "expected_result": args.expected_result,
        "failure_criterion": args.failure_criterion,
        "command": args.command,
        "cwd": str(repo),
        "started_at_utc": started_at,
        "timeout_seconds": args.timeout_seconds,
        "protected_sha256_before": before,
        "status": "STARTED",
        "exit_code": None,
    }
    write_json_atomic(record_path, record)
    append_jsonl(control / "DECISION_LOG.jsonl", {
        "timestamp_utc": started_at,
        "task_id": TASK_ID,
        "kind": "DIAGNOSTIC_PREFLIGHT",
        "hypothesis_id": args.hypothesis_id,
        "driver_kind": args.driver_kind,
        "unique_change": args.unique_change,
        "expected_result": args.expected_result,
        "failure_criterion": args.failure_criterion,
        "run_number": run_number,
    })
    state["diagnostic_runs_started"] = run_number
    state["phase"] = "DIAGNOSTIC_RUNNING"
    state["status"] = "RUNNING"
    state["last_updated_at_utc"] = started_at
    write_json_atomic(control / "MASTER_STATE.json", state)
    update_status(control / "CURRENT_STATUS_CN.md", [
        f"更新时间（UTC）：{started_at}",
        f"开始诊断 {run_number}/2：{args.hypothesis_id} / {args.driver_kind}",
        f"唯一改变：{args.unique_change}",
    ])

    deadline = time.monotonic() + args.timeout_seconds
    next_heartbeat = time.monotonic() + 60
    next_status = time.monotonic() + 600
    exit_code: int | None = None
    timed_out = False
    with log_path.open("wb") as log_stream:
        process = subprocess.Popen(
            args.command,
            cwd=repo,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        while True:
            exit_code = process.poll()
            now = time.monotonic()
            if exit_code is not None:
                break
            if now >= deadline:
                timed_out = True
                terminate_group(process)
                exit_code = process.poll()
                break
            if now >= next_heartbeat:
                elapsed = int(args.timeout_seconds - max(0, deadline - now))
                print(f"[{utc_now()}] 存活：诊断 {run_number}/2 已运行 {elapsed}s", flush=True)
                next_heartbeat += 60
            if now >= next_status:
                update_status(control / "CURRENT_STATUS_CN.md", [
                    f"更新时间（UTC）：{utc_now()}",
                    f"诊断 {run_number}/2 仍在运行，驱动：{args.driver_kind}",
                ])
                next_status += 600
            time.sleep(1)

    after = snapshot(protected)
    changed = sorted(path for path in set(before) | set(after) if before.get(path) != after.get(path))
    ended_at = utc_now()
    record.update({
        "ended_at_utc": ended_at,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "protected_sha256_after": after,
        "protected_paths_changed": changed,
        "log_path": str(log_path.relative_to(repo)),
        "status": "INVALID_PROTECTED_CHANGE" if changed else ("TIMED_OUT" if timed_out else "COMPLETED"),
    })
    write_json_atomic(record_path, record)
    state = read_required(repo)[1]
    state["diagnostic_runs_completed"] = run_number
    state["phase"] = "DIAGNOSTIC_REVIEW"
    state["status"] = record["status"]
    state["last_updated_at_utc"] = ended_at
    write_json_atomic(control / "MASTER_STATE.json", state)
    update_status(control / "CURRENT_STATUS_CN.md", [
        f"更新时间（UTC）：{ended_at}",
        f"诊断 {run_number}/2 结束，exit_code={exit_code}，timed_out={timed_out}",
        f"受保护文件变化：{changed if changed else '无'}",
    ])
    if changed:
        print("受保护文件在运行期间发生变化，证据无效", file=sys.stderr)
        return 86
    if timed_out:
        return 124
    return int(exit_code or 0)


if __name__ == "__main__":
    raise SystemExit(main())

