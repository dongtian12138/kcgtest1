#!/usr/bin/env python3
"""Run the one TASK-R12-006B formal P1 command under a fail-closed guard."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from typing import Any

import yaml


TASK_ID = "TASK-R12-006B"
RESULT_SHA256 = "4551c1a9900b421e85c53e7dabd9821cb8349ada80fb5ab95a1522951b79febb"
EXPECTED_COMMAND = [
    "src/kcg_connector/isaac/run_isaac_python.sh",
    "src/kcg_connector/isaac/d38999_physical_r7_p1_nominal_bench.py",
    "--run",
    "--output-dir",
    "artifacts/agent_control/tasks/TASK-R12-006B/P1_FORMAL",
    "--scene-config",
    "artifacts/agent_control/tasks/TASK-R12-006B/candidate/scene.yaml",
    "--model-contract",
    "src/kcg_connector/config/d38999_keyed_v3_physical_model_contract_r12_v1.yaml",
    "--acceptance-config",
    "src/kcg_connector/config/d38999_keyed_v3_physical_acceptance_r12_v1.yaml",
    "--authorized-local-candidate-result",
    "artifacts/agent_control/tasks/TASK-R12-006B/candidate/CANDIDATE_BUILD_RESULT.json",
    "--authorized-local-candidate-result-sha256",
    RESULT_SHA256,
    "--candidate-index",
    "2",
    "--kit-portable-root",
    "/tmp/task-r12-006b-p1-formal",
]
OUTPUT_REL = Path("artifacts/agent_control/tasks/TASK-R12-006B/P1_FORMAL")
PORTABLE_ROOT = Path("/tmp/task-r12-006b-p1-formal")
TASK_DIR_REL = Path("artifacts/agent_control/tasks/TASK-R12-006B")
RECORD_REL = TASK_DIR_REL / "FORMAL_RUN_RECORD.json"
LOG_REL = TASK_DIR_REL / "FORMAL_RUN.log"
HEARTBEAT_REL = TASK_DIR_REL / "FORMAL_HEARTBEAT.json"
PROTECTED_RELATIVE_PATHS = (
    Path("artifacts/kcg_connector/isaac/keyed_v3_physical_r12/candidates/r12_candidate_02/r12_candidate_02.usda"),
    Path("artifacts/agent_control/tasks/TASK-R12-006B/candidate/task_r12_006b_local_candidate_01.usda"),
    Path("artifacts/agent_control/tasks/TASK-R12-006B/candidate/scene.yaml"),
    Path("artifacts/agent_control/tasks/TASK-R12-006B/candidate/CANDIDATE_BUILD_RESULT.json"),
    Path("src/kcg_connector/config/d38999_keyed_v3_physical_model_contract_r12_v1.yaml"),
    Path("src/kcg_connector/config/d38999_keyed_v3_physical_acceptance_r12_v1.yaml"),
    Path("src/kcg_connector/kcg_connector/d38999_r12_local_candidate.py"),
    Path("src/kcg_connector/kcg_connector/d38999_tabletop_scene.py"),
    Path("src/kcg_connector/isaac/validate_physical_r7_composed_scene.py"),
    Path("src/kcg_connector/isaac/validate_physical_r11_cooked_geometry.py"),
    Path("src/kcg_connector/isaac/d38999_physical_r7_p1_nominal_bench.py"),
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_text_atomic(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def append_status(path: Path, title: str, rows: list[str]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(f"\n## {title}\n\n")
        for row in rows:
            stream.write(f"- {row}\n")


def protected_snapshot(repository: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in PROTECTED_RELATIVE_PATHS:
        path = repository / relative
        if not path.is_file():
            raise FileNotFoundError(f"受保护文件不存在：{relative}")
        result[str(relative)] = sha256_file(path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--timeout-seconds", type=int, default=1450)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    if arguments.command and arguments.command[0] == "--":
        arguments.command = arguments.command[1:]
    if arguments.command != EXPECTED_COMMAND:
        parser.error("命令与 TASK-R12-006B 任务图中的唯一正式 P1 命令不一致")
    if not 1 <= arguments.timeout_seconds <= 1450:
        parser.error("子命令超时必须在 1..1450 秒内")
    return arguments


def preflight(repository: Path, command: list[str]) -> tuple[dict[str, Any], str]:
    control = repository / "artifacts/agent_control"
    charter = (control / "PROJECT_CHARTER_CN.md").read_text(encoding="utf-8")
    graph_text = (control / "TASK_GRAPH.yaml").read_text(encoding="utf-8")
    graph = yaml.safe_load(graph_text)
    state = json.loads((control / "MASTER_STATE.json").read_text(encoding="utf-8"))
    task = (control / "CURRENT_TASK.md").read_text(encoding="utf-8")
    with (control / "GATE_LEDGER.csv").open(encoding="utf-8", newline="") as stream:
        gates = list(csv.DictReader(stream))
    node = graph["nodes"][TASK_ID]
    if TASK_ID not in charter or TASK_ID not in task:
        raise RuntimeError("章程或当前任务不是 TASK-R12-006B")
    if graph.get("current_task") != TASK_ID or state.get("task_id") != TASK_ID:
        raise RuntimeError("任务图或主状态不是 TASK-R12-006B")
    if state.get("phase") != "STATIC_VALIDATION_PASSED_FORMAL_P1_PENDING":
        raise RuntimeError("主状态尚未到达正式 P1 待运行阶段")
    if state.get("local_candidates_created") != 1 or node.get("local_candidates_created") != 1:
        raise RuntimeError("唯一候选计数不是 1/1")
    if state.get("formal_p1_run_count") != 0 or node.get("formal_p1_runs") != 0:
        raise RuntimeError("正式 P1 已经启动过")
    if node.get("static_validation_status") != "PASS":
        raise RuntimeError("静态 A2 门尚未通过")
    if not any(
        row.get("gate_id") == "P1-006B-A2" and row.get("status") == "PASS"
        for row in gates
    ):
        raise RuntimeError("门账缺少 P1-006B-A2 通过记录")
    if shlex.split(str(node.get("acceptance_command", ""))) != command:
        raise RuntimeError("运行命令与任务图 acceptance_command 不一致")
    candidate = state.get("local_candidate", {})
    if candidate.get("build_result_sha256") != RESULT_SHA256:
        raise RuntimeError("主状态中的候选清单 SHA-256 pin 改变")
    if (repository / OUTPUT_REL).exists():
        raise FileExistsError("正式 P1 输出目录已存在")
    if PORTABLE_ROOT.exists():
        raise FileExistsError("正式 P1 portable root 已存在")
    for relative in (RECORD_REL, LOG_REL, HEARTBEAT_REL):
        if (repository / relative).exists():
            raise FileExistsError(f"正式运行证据已存在：{relative}")
    return state, graph_text


def terminate_group(process: subprocess.Popen[Any]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def main() -> int:
    arguments = parse_args()
    repository = arguments.repo.expanduser().resolve()
    state, graph_text = preflight(repository, arguments.command)
    before = protected_snapshot(repository)
    print("TASK-R12-006B 正式 P1 守卫预检通过", flush=True)
    if arguments.check_only:
        return 0

    control = repository / "artifacts/agent_control"
    record_path = repository / RECORD_REL
    log_path = repository / LOG_REL
    heartbeat_path = repository / HEARTBEAT_REL
    started_at = utc_now()
    record: dict[str, Any] = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "run_number": 1,
        "run_limit": 1,
        "run_kind": "FORMAL_P1",
        "diagnostic_only": False,
        "command": arguments.command,
        "cwd": str(repository),
        "started_at_utc": started_at,
        "timeout_seconds": arguments.timeout_seconds,
        "protected_sha256_before": before,
        "status": "STARTED",
        "exit_code": None,
        "timed_out": False,
    }
    write_json_atomic(record_path, record)
    state["formal_p1_run_count"] = 1
    state["phase"] = "FORMAL_P1_RUNNING"
    state["last_updated_at_utc"] = started_at
    state["formal_p1_process"] = {
        "started_at_utc": started_at,
        "run_record": str(RECORD_REL),
        "log": str(LOG_REL),
        "heartbeat": str(HEARTBEAT_REL),
    }
    write_json_atomic(control / "MASTER_STATE.json", state)
    marker = "    formal_p1_runs: 0\n"
    if graph_text.count(marker) != 1:
        raise RuntimeError("任务图正式 P1 计数标记不唯一")
    write_text_atomic(
        control / "TASK_GRAPH.yaml",
        graph_text.replace(marker, "    formal_p1_runs: 1\n", 1),
    )
    append_jsonl(
        control / "DECISION_LOG.jsonl",
        {
            "timestamp_utc": started_at,
            "task_id": TASK_ID,
            "kind": "FORMAL_P1_PREFLIGHT",
            "run_number": 1,
            "run_limit": 1,
            "diagnostic_only": False,
            "unique_change": "none_after_single_local_candidate",
            "expected_result": "前四事件保持且后三事件依次发生并满足冻结P1全部判据",
            "failure_criterion": "任一事件缺失或乱序、位置超差、力矩越限、硬穿透、错误接触、求解器错误、位姿写入或同一阻塞再现",
            "command": arguments.command,
        },
    )
    append_status(
        control / "CURRENT_STATUS_CN.md",
        "TASK-R12-006B 正式 P1 守卫启动",
        [
            f"更新时间（UTC）：{started_at}",
            "状态：运行中",
            "正式 P1 运行次数：1/1",
            "当前需要用户做什么：无需操作",
        ],
    )

    child_environment = dict(os.environ)
    package_path = str(repository / "src/kcg_connector")
    prior_pythonpath = child_environment.get("PYTHONPATH")
    child_environment["PYTHONPATH"] = (
        package_path if not prior_pythonpath else package_path + os.pathsep + prior_pythonpath
    )
    deadline = time.monotonic() + arguments.timeout_seconds
    next_heartbeat = time.monotonic() + 60.0
    next_status = time.monotonic() + 600.0
    exit_code: int | None = None
    timed_out = False
    with log_path.open("wb") as log_stream:
        process = subprocess.Popen(
            arguments.command,
            cwd=repository,
            env=child_environment,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        record["child_pid"] = process.pid
        write_json_atomic(record_path, record)
        while True:
            exit_code = process.poll()
            now = time.monotonic()
            elapsed = int(arguments.timeout_seconds - max(0.0, deadline - now))
            if exit_code is not None:
                break
            if now >= deadline:
                timed_out = True
                terminate_group(process)
                exit_code = process.poll()
                break
            if now >= next_heartbeat:
                heartbeat = {
                    "task_id": TASK_ID,
                    "run_number": 1,
                    "status": "RUNNING",
                    "timestamp_utc": utc_now(),
                    "elapsed_seconds": elapsed,
                    "child_pid": process.pid,
                }
                write_json_atomic(heartbeat_path, heartbeat)
                print(
                    f"[{heartbeat['timestamp_utc']}] 存活：正式 P1 已运行 {elapsed}s",
                    flush=True,
                )
                next_heartbeat += 60.0
            if now >= next_status:
                append_status(
                    control / "CURRENT_STATUS_CN.md",
                    "TASK-R12-006B 正式 P1 十分钟状态",
                    [
                        f"更新时间（UTC）：{utc_now()}",
                        f"正式 P1 仍在运行，已运行 {elapsed} 秒",
                        "当前需要用户做什么：无需操作",
                    ],
                )
                next_status += 600.0
            time.sleep(1.0)

    ended_at = utc_now()
    after = protected_snapshot(repository)
    changed = sorted(
        path for path in before.keys() | after.keys() if before.get(path) != after.get(path)
    )
    final_status = (
        "INVALID_PROTECTED_CHANGE"
        if changed
        else ("TIMED_OUT" if timed_out else "COMPLETED")
    )
    record.update(
        {
            "ended_at_utc": ended_at,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "protected_sha256_after": after,
            "protected_paths_changed": changed,
            "log_path": str(LOG_REL),
            "status": final_status,
        }
    )
    write_json_atomic(record_path, record)
    write_json_atomic(
        heartbeat_path,
        {
            "task_id": TASK_ID,
            "run_number": 1,
            "status": final_status,
            "timestamp_utc": ended_at,
            "exit_code": exit_code,
            "timed_out": timed_out,
        },
    )
    current_state = json.loads(
        (control / "MASTER_STATE.json").read_text(encoding="utf-8")
    )
    current_state["phase"] = "FORMAL_P1_REVIEW"
    current_state["last_updated_at_utc"] = ended_at
    current_state["formal_p1_process"].update(
        {
            "ended_at_utc": ended_at,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "protected_paths_changed": changed,
        }
    )
    write_json_atomic(control / "MASTER_STATE.json", current_state)
    append_status(
        control / "CURRENT_STATUS_CN.md",
        "TASK-R12-006B 正式 P1 进程结束",
        [
            f"更新时间（UTC）：{ended_at}",
            f"进程退出码：{exit_code}",
            f"是否超时：{str(timed_out).lower()}",
            f"受保护文件变化：{changed if changed else '无'}",
            "说明：尚未根据原始报告宣布 P1 通过或失败",
            "当前需要用户做什么：无需操作",
        ],
    )
    print(
        f"正式 P1 子进程结束：exit_code={exit_code} timed_out={timed_out} protected_changes={changed}",
        flush=True,
    )
    if changed:
        return 86
    if timed_out:
        return 124
    return int(exit_code or 0)


if __name__ == "__main__":
    raise SystemExit(main())
