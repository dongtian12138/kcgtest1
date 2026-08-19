"""Deterministic final report and review-bundle builder for the eight-hour window.

Pure helper functions may be exercised in temporary directories before the
deadline.  The CLI refuses to write formal outputs before the frozen deadline
or unless the control files already carry EIGHT_HOUR_WINDOW_COMPLETE.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Mapping, Sequence
import zipfile

from .final_review_preflight import (
    CONTROL_MEMBERS,
    EXPECTED_TASK_KEYS,
    FINAL_SUMMARY_FIELDS_CN,
    FINAL_TABLE_COLUMNS_CN,
    PASS_STATUSES,
    _check_fraction,
    _declared_evidence_paths,
    _document_status,
    _duration_seconds,
    _flatten_queue,
    _integer_prefix,
    _load_jsonl,
    _load_mapping,
    _resolve_inside,
    _sha256,
    _true_forbidden_claims,
    validate_bundle_name,
)


SCHEMA_VERSION = "kcg_eight_hour_final_artifacts_v1"
TASK_ID = "EIGHT-HOUR-G4-FINAL-ARTIFACT-BUILDER"
FINAL_STATUS = "EIGHT_HOUR_WINDOW_COMPLETE"
FINAL_USER_ACTION = "需要上传审查包"
FINAL_REPORT = Path("artifacts/agent_control/EIGHT_HOUR_FINAL_REPORT_CN.md")
FINAL_RESULT = Path("artifacts/agent_control/EIGHT_HOUR_FINAL_RESULT.json")
FINAL_COMMANDS = Path("artifacts/agent_control/EIGHT_HOUR_ACTUAL_COMMANDS.txt")
FINAL_MANIFEST = Path("artifacts/agent_control/EIGHT_HOUR_FILE_MANIFEST.json")
FINAL_DIFF = Path("artifacts/agent_control/EIGHT_HOUR_CODE_DIFF.patch")
FINAL_WORKTREE = Path("artifacts/agent_control/EIGHT_HOUR_WORKTREE_STATUS.txt")
FINAL_OUTPUTS = (
    FINAL_REPORT, FINAL_RESULT, FINAL_COMMANDS, FINAL_MANIFEST,
    FINAL_DIFF, FINAL_WORKTREE,
)


def _parse_utc(value: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("timestamp must be explicit UTC")
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _task_records(
    root: Path, queue: Mapping[str, Any]
) -> list[tuple[str, Mapping[str, Any], Mapping[str, Any], str]]:
    records: list[tuple[str, Mapping[str, Any], Mapping[str, Any], str]] = []
    for key, task in _flatten_queue(queue):
        value = task.get("evidence")
        if not isinstance(value, str):
            raise ValueError(f"{key} lacks result evidence")
        path = _resolve_inside(root, value, f"{key} result")
        document = _load_mapping(path, f"{key} result")
        if _document_status(document) != task.get("status"):
            raise ValueError(f"{key} queue/evidence status mismatch")
        records.append((key, task, document, str(path.relative_to(root))))
    post_queue = queue.get("post_queue_tasks")
    if not isinstance(post_queue, Mapping):
        raise ValueError("work queue lacks post_queue_tasks")
    for key, task in post_queue.items():
        if not isinstance(task, Mapping):
            raise ValueError(f"post-queue task {key} is invalid")
        value = task.get("evidence")
        if value is None:
            continue
        if not isinstance(value, str):
            raise ValueError(f"post-queue task {key} evidence is invalid")
        path = _resolve_inside(root, value, f"post-queue {key} result")
        document = _load_mapping(path, f"post-queue {key} result")
        if _document_status(document) != task.get("status"):
            raise ValueError(f"post-queue {key} queue/evidence status mismatch")
        records.append((str(key), task, document, str(path.relative_to(root))))
    return records


def _hash_verified_code_files(
    root: Path,
    records: Sequence[tuple[str, Mapping[str, Any], Mapping[str, Any], str]],
) -> tuple[list[str], list[str]]:
    modules: set[str] = set()
    tests: set[str] = set()
    for key, _task, document, _result_path in records:
        if _document_status(document) not in PASS_STATUSES:
            continue
        for row in document.get("code_files", []):
            if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
                raise ValueError(f"{key} has invalid code_files row")
            path = _resolve_inside(root, row["path"], f"{key} code file")
            expected = row.get("sha256")
            if not isinstance(expected, str) or not path.is_file() or _sha256(path) != expected:
                raise ValueError(f"{key} hash-verified code file drift")
            collection = tests if "/test/" in row["path"] else modules
            collection.add(row["path"])
    return sorted(modules), sorted(tests)


def _active_parked_rows(blockers: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in blockers:
        if row.get("status") not in {
            "PARKED", "PARKED_FINAL_FOR_THIS_WINDOW", "BLOCKED_EXTERNAL"
        }:
            continue
        rows.append(
            {
                "task_id": row.get("task_id"),
                "classification": row.get("classification") or row.get("blocker_id"),
                "status": row.get("status"),
                "evidence": row.get("evidence"),
            }
        )
    return rows


def _task_table_row(
    key: str,
    task: Mapping[str, Any],
    document: Mapping[str, Any],
    result_path: str,
    is_main_queue: bool,
) -> dict[str, Any]:
    status = _document_status(document)
    fraction = _check_fraction(document)
    checks = f"{fraction[0]}/{fraction[1]}" if fraction else "未统一声明"
    if status == "DYNAMIC_PASS":
        true_pass = "是（动态）"
        reason = "—"
    elif status in {"STATIC_PASS", "OFFLINE_PASS"}:
        true_pass = "实现/静态或离线通过；物理动态否"
        reason = "上游动态门未通过或本节点仅授权离线证据"
    else:
        true_pass = "否"
        reason = str(document.get("classification") or task.get("classification") or status)
    if key == "A1":
        start_status = "PARKED（窗口入口问题，授权唯一重访）"
    elif is_main_queue:
        start_status = "未单独冻结；不得反推为动态状态"
    else:
        start_status = "NOT_STARTED（按自动交接进入）"
    next_step = document.get("next_task") or task.get("next_if_pass") or "依G2根阻塞顺序重入"
    return {
        "task": key,
        "name": task.get("name"),
        "start_status": start_status,
        "end_status": status,
        "work_done": document.get("classification") or task.get("name"),
        "key_values_or_checks": checks,
        "truly_passed": true_pass,
        "failure_reason": reason,
        "next_step": next_step,
        "result_path": result_path,
    }


def collect_package_sources(
    root: Path,
    records: Sequence[tuple[str, Mapping[str, Any], Mapping[str, Any], str]],
    code_modules: Sequence[str],
    test_files: Sequence[str],
) -> list[str]:
    paths: set[str] = set(CONTROL_MEMBERS)
    for key, _task, document, result_path in records:
        paths.add(result_path)
        for value in _declared_evidence_paths(document):
            path = _resolve_inside(root, value, f"{key} declared evidence")
            if not path.is_file():
                raise ValueError(f"declared evidence missing: {value}")
            paths.add(str(path.relative_to(root)))
    paths.update(code_modules)
    paths.update(test_files)
    a1_files = (
        "artifacts/agent_control/tasks/EIGHT-HOUR-A1-INIT-DATUM/RUN_RECORD.json",
        "artifacts/agent_control/tasks/EIGHT-HOUR-A1-INIT-DATUM/ACTUAL_COMMANDS.txt",
    )
    paths.update(a1_files)
    for relative in sorted(paths):
        path = _resolve_inside(root, relative, "package source")
        if not path.is_file():
            raise ValueError(f"package source missing: {relative}")
    return sorted(paths)


def build_final_report_data(
    *,
    repository_root: str | Path,
    work_queue_path: str | Path,
    master_state_path: str | Path,
    blocker_ledger_path: str | Path,
    readiness_report_path: str | Path,
    preflight_path: str | Path,
    closeout_at_utc: str,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    queue_path = _resolve_inside(root, work_queue_path, "work queue")
    master_path = _resolve_inside(root, master_state_path, "master state")
    blocker_path = _resolve_inside(root, blocker_ledger_path, "blocker ledger")
    readiness_path = _resolve_inside(root, readiness_report_path, "readiness report")
    preflight_file = _resolve_inside(root, preflight_path, "final preflight")
    queue = _load_mapping(queue_path, "work queue")
    master = _load_mapping(master_path, "master state")
    blockers = _load_jsonl(blocker_path, "blocker ledger")
    readiness = _load_mapping(readiness_path, "readiness report")
    preflight = _load_mapping(preflight_file, "final preflight")
    closeout = _parse_utc(closeout_at_utc)
    start = _parse_utc(str(queue.get("window_started_at_utc")))
    deadline = _parse_utc(str(queue.get("window_deadline_utc")))
    if closeout < start:
        raise ValueError("closeout precedes the autonomous window")
    if (
        readiness.get("formal_dynamic_pass_count") != 0
        or readiness.get("current_frontier_state") != "HOME"
        or preflight.get("report_blueprint", {}).get("summary_field_count") != 14
    ):
        raise ValueError("frozen dynamic/report boundary changed")
    records = _task_records(root, queue)
    modules, tests = _hash_verified_code_files(root, records)
    rows = []
    fixed_problem_tasks = []
    passed_checks = 0
    total_checks = 0
    over_budget_tasks = []
    forbidden_claims = []
    main_keys = set(EXPECTED_TASK_KEYS)
    for key, task, document, result_path in records:
        rows.append(
            _task_table_row(key, task, document, result_path, key in main_keys)
        )
        status = _document_status(document)
        fraction = _check_fraction(document)
        if status in PASS_STATUSES and fraction:
            if fraction[0] != fraction[1]:
                raise ValueError(f"passing task {key} has incomplete checks")
            passed_checks += fraction[0]
            total_checks += fraction[1]
        if status in PASS_STATUSES and _integer_prefix(document.get("targeted_fix_count")):
            fixed_problem_tasks.append(key)
        task_start = document.get("started_at_utc")
        task_end = document.get("completed_at_utc")
        if isinstance(task_start, str) and isinstance(task_end, str):
            if _duration_seconds(task_start, task_end) > 2700:
                over_budget_tasks.append(key)
        forbidden_claims.extend(
            f"{key}:{path}" for path in _true_forbidden_claims(document)
        )
    a1_record_path = root / "artifacts/agent_control/tasks/EIGHT-HOUR-A1-INIT-DATUM/RUN_RECORD.json"
    a1_record = _load_mapping(a1_record_path, "A1 run record")
    isaac_seconds = _duration_seconds(
        str(a1_record["started_at_utc"]), str(a1_record["ended_at_utc"])
    )
    task_status_counts = Counter(row["end_status"] for row in rows)
    parked = _active_parked_rows(blockers)
    top_three = [
        row["task_key"] for row in readiness["top_three_priority_root_blockers"]
    ]
    if top_three != ["A1", "B1", "C8"]:
        raise ValueError("top-three blockers changed")
    source_paths = collect_package_sources(root, records, modules, tests)
    elapsed = (min(closeout, deadline) - start).total_seconds()
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "window_started_at_utc": queue.get("window_started_at_utc"),
        "window_deadline_utc": queue.get("window_deadline_utc"),
        "closeout_at_utc": closeout_at_utc,
        "window_elapsed_seconds_at_report": elapsed,
        "task_rows": rows,
        "task_status_counts": dict(sorted(task_status_counts.items())),
        "metrics": {
            "resolved_problem_count": len(fixed_problem_tasks),
            "resolved_problem_tasks": fixed_problem_tasks,
            "completed_code_module_count": len(modules),
            "completed_test_file_count": len(tests),
            "passed_static_or_offline_check_count": passed_checks,
            "declared_static_or_offline_check_total": total_checks,
            "dynamic_passed_task_count": readiness["formal_dynamic_pass_count"],
            "isaac_process_run_count": 1,
            "formal_or_acceptance_isaac_run_count": 0,
            "isaac_cumulative_runtime_seconds": isaac_seconds,
            "isaac_explicit_physics_step_count": 0,
            "peak_vram_mib": None,
            "physics_fps": None,
            "render_fps": None,
            "performance_missing_reason": "A4仅离线通过，未启动真实测量",
            "current_frontier_state": readiness["current_frontier_state"],
            "parked_problem_count": len(parked),
            "parked_problems": parked,
            "top_three_priority_root_blockers": top_three,
            "over_45_minute_task_count": len(over_budget_tasks),
            "over_45_minute_tasks": over_budget_tasks,
            "known_forbidden_rule_violation_count": len(forbidden_claims),
            "known_forbidden_rule_violations": forbidden_claims,
        },
        "code_modules": modules,
        "test_files": tests,
        "package_source_paths": source_paths,
        "source_manifest": {
            "work_queue": {"path": str(queue_path.relative_to(root)), "sha256": _sha256(queue_path)},
            "master_state": {"path": str(master_path.relative_to(root)), "sha256": _sha256(master_path)},
            "blocker_ledger": {"path": str(blocker_path.relative_to(root)), "sha256": _sha256(blocker_path)},
            "readiness_report": {"path": str(readiness_path.relative_to(root)), "sha256": _sha256(readiness_path)},
            "preflight": {"path": str(preflight_file.relative_to(root)), "sha256": _sha256(preflight_file)},
        },
        "current_control_status": master.get("status"),
        "final_status_requested": FINAL_STATUS,
        "final_user_action_requested": FINAL_USER_ACTION,
        "simulation_started_by_builder": False,
        "robot_commands_emitted_by_builder": 0,
        "static_or_offline_promoted_to_dynamic_count": 0,
        "assembly_success_claimed": False,
        "formal_r12_generated": False,
        "control_authorized": False,
        "hardware_authorized": False,
    }


def _cell(value: Any) -> str:
    return str(value if value is not None else "无实测值").replace("|", "\\|").replace("\n", " ")


def render_final_report(data: Mapping[str, Any]) -> str:
    metrics = data["metrics"]
    lines = [
        "# 八小时自主任务最终报告",
        "",
        "| " + " | ".join(FINAL_TABLE_COLUMNS_CN) + " |",
        "|" + "---|" * len(FINAL_TABLE_COLUMNS_CN),
    ]
    for row in data["task_rows"]:
        values = (
            f"{row['task']} {row['name']}", row["start_status"], row["end_status"],
            row["work_done"], row["key_values_or_checks"], row["truly_passed"],
            row["failure_reason"], row["next_step"],
        )
        lines.append("| " + " | ".join(_cell(value) for value in values) + " |")
    parked_lines = [
        f"{row['task_id']}:{row['classification']}" for row in metrics["parked_problems"]
    ]
    lines.extend(
        [
            "",
            "## 十四项最终汇总",
            "",
            f"1. 八小时内实际解决的问题：{metrics['resolved_problem_count']} 个；口径为有 `targeted_fix_count` 且最终静态/离线通过的节点，节点为 {metrics['resolved_problem_tasks']}。",
            f"2. 完成的代码模块：{metrics['completed_code_module_count']} 个；另有 {metrics['completed_test_file_count']} 个摘要匹配的测试文件。",
            f"3. 通过的静态/离线检查：{metrics['passed_static_or_offline_check_count']}/{metrics['declared_static_or_offline_check_total']}；只统计最终通过节点明确声明的最终检查分数。",
            f"4. 通过的动态测试：{metrics['dynamic_passed_task_count']}。",
            f"5. Isaac 进程启动：{metrics['isaac_process_run_count']} 次，其中正式/验收运行 {metrics['formal_or_acceptance_isaac_run_count']} 次；唯一一次为 A1 诊断性初始化对照。",
            f"6. 累计 Isaac 进程时间：{metrics['isaac_cumulative_runtime_seconds']:.0f} 秒；显式物理步 {metrics['isaac_explicit_physics_step_count']}。",
            "7. 5070Ti 最高显存：无实测值。",
            "8. 物理帧率：无实测值。",
            "9. 渲染帧率：无实测值。",
            f"10. 当前最前面的完整任务链状态：`{metrics['current_frontier_state']}`。",
            f"11. PARKED/外部阻塞账本项：{metrics['parked_problem_count']} 项；" + "；".join(parked_lines) + "。",
            f"12. 三个最高优先问题：{metrics['top_three_priority_root_blockers']}；按 A 至 F 主链首次出现位置排序。",
            f"13. 原地打转超过45分钟：{'是' if metrics['over_45_minute_task_count'] else '否'}；记录节点 {metrics['over_45_minute_tasks']}。",
            f"14. 已记录的禁止规则违反：{metrics['known_forbidden_rule_violation_count']}；该结论限于状态文件和任务结果中的显式字段。",
            "",
            "## 证据边界",
            "",
            "- `STATIC_PASS`/`OFFLINE_PASS` 只表示实现或离线证据，不是物理动态通过。",
            "- A2 正式名义插入未运行，完整状态机仍停在 `HOME`，机器人命令为 0。",
            "- 没有真实显存、物理帧率、渲染帧率或视频证据；未用 0 伪装缺失值。",
            "- 正式 R12 未生成；高精细参考只读；本窗口新连接器几何候选为 0；硬件授权为 false。",
            "",
        ]
    )
    return "\n".join(lines)


def collect_actual_commands(root: Path, source_paths: Sequence[str]) -> str:
    rows = [
        "# 八小时窗口实际命令记录",
        "# 仅收录 ACTUAL_COMMANDS 文本以及验证/运行结果文件中的显式 command 字段。",
        "",
    ]
    seen: set[str] = set()
    for relative in source_paths:
        path = root / relative
        if path.name == "ACTUAL_COMMANDS.txt":
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if text and text not in seen:
                rows.extend([f"## {relative}", text, ""])
                seen.add(text)
            continue
        if path.suffix != ".json" or not any(
            token in path.name for token in ("VALIDATION", "RUN_RECORD")
        ):
            continue
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(document, Mapping):
            continue
        commands = []
        for key, value in document.items():
            if key == "command" and isinstance(value, str):
                commands.append(value)
            elif key == "commands" and isinstance(value, list):
                commands.extend(item for item in value if isinstance(item, str))
        for command in commands:
            if command in seen:
                continue
            rows.extend([f"## {relative}", command, ""])
            seen.add(command)
    rows.append(f"# 唯一命令记录数：{len(seen)}")
    return "\n".join(rows) + "\n"


def build_code_snapshot_patch(root: Path, code_paths: Sequence[str]) -> str:
    lines = [
        "# 八小时任务代码快照补丁",
        "# 每个文件均由通过节点的 TASK_RESULT SHA-256 约束；这是当前快照，不声称工作树全部改动均由本任务创建。",
        "",
    ]
    for relative in sorted(set(code_paths)):
        path = _resolve_inside(root, relative, "code snapshot")
        try:
            current = path.read_text(encoding="utf-8").splitlines(keepends=True)
        except UnicodeDecodeError:
            lines.append(f"# binary omitted: {relative} sha256={_sha256(path)}\n")
            continue
        lines.extend(
            difflib.unified_diff(
                [], current, fromfile="/dev/null", tofile=relative, lineterm=""
            )
        )
        lines.append("")
    return "\n".join(lines) + "\n"


def _write_new(path: Path, content: str | bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite final output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")


def build_review_zip(
    *,
    root: Path,
    source_paths: Sequence[str],
    closeout_paths: Sequence[Path],
    bundle_path: Path,
) -> dict[str, Any]:
    validate_bundle_name(bundle_path.name)
    expected_parent = (root / "artifacts/agent_control/review").resolve()
    bundle = bundle_path.resolve()
    if bundle.parent != expected_parent:
        raise ValueError("review bundle must be inside artifacts/agent_control/review")
    if bundle.exists():
        raise FileExistsError("review bundle output is immutable")
    members: dict[str, Path] = {}
    for relative in source_paths:
        path = _resolve_inside(root, relative, "bundle source")
        members[relative] = path
    for path in closeout_paths:
        resolved = path.resolve()
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise ValueError("closeout bundle member is missing or outside repository")
        members[str(resolved.relative_to(root))] = resolved
    bundle.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="eight_hour_bundle_") as temp_name:
        staging = Path(temp_name)
        for relative, source in sorted(members.items()):
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        staged = sorted(path for path in staging.rglob("*") if path.is_file())
        sums = "\n".join(
            f"{_sha256(path)}  {path.relative_to(staging)}" for path in staged
        ) + "\n"
        (staging / "SHA256SUMS.txt").write_text(sums, encoding="utf-8")
        with zipfile.ZipFile(bundle, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(item for item in staging.rglob("*") if item.is_file()):
                archive.write(path, path.relative_to(staging))
    verification = verify_review_zip(bundle)
    return {
        "path": str(bundle.relative_to(root)),
        "sha256": _sha256(bundle),
        "size_bytes": bundle.stat().st_size,
        **verification,
    }


def verify_review_zip(path: str | Path) -> dict[str, Any]:
    bundle = Path(path)
    with zipfile.ZipFile(bundle, "r") as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("review bundle contains duplicate member names")
        if "SHA256SUMS.txt" not in names:
            raise ValueError("review bundle lacks SHA256SUMS.txt")
        expected: dict[str, str] = {}
        for line in archive.read("SHA256SUMS.txt").decode("utf-8").splitlines():
            digest, name = line.split("  ", 1)
            expected[name] = digest
        actual_names = set(names) - {"SHA256SUMS.txt"}
        if actual_names != set(expected):
            raise ValueError("review bundle manifest/member mismatch")
        mismatches = [
            name for name, digest in expected.items()
            if hashlib.sha256(archive.read(name)).hexdigest() != digest
        ]
        if mismatches:
            raise ValueError(f"review bundle SHA-256 mismatch: {mismatches}")
    return {
        "member_count": len(names),
        "manifest_entry_count": len(expected),
        "sha256_verified": True,
    }


def finalize_artifacts(
    *,
    repository_root: str | Path,
    data: Mapping[str, Any],
    bundle_name: str,
    worktree_status: str,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    queue = _load_mapping(root / "artifacts/agent_control/WORK_QUEUE.yaml", "work queue")
    master = _load_mapping(root / "artifacts/agent_control/MASTER_STATE.json", "master state")
    if _parse_utc(str(data["closeout_at_utc"])) < _parse_utc(str(queue["window_deadline_utc"])):
        raise PermissionError("formal final artifacts are forbidden before the window deadline")
    if queue.get("status") != FINAL_STATUS or master.get("status") != FINAL_STATUS:
        raise PermissionError("control files must declare EIGHT_HOUR_WINDOW_COMPLETE first")
    outputs = [root / relative for relative in FINAL_OUTPUTS]
    if any(path.exists() for path in outputs):
        raise FileExistsError("one or more formal final outputs already exist")
    report_text = render_final_report(data)
    commands_text = collect_actual_commands(root, data["package_source_paths"])
    patch_text = build_code_snapshot_patch(
        root, list(data["code_modules"]) + list(data["test_files"])
    )
    _write_new(root / FINAL_REPORT, report_text)
    _write_new(root / FINAL_COMMANDS, commands_text)
    _write_new(root / FINAL_DIFF, patch_text)
    _write_new(root / FINAL_WORKTREE, worktree_status)
    result = {
        "schema_version": SCHEMA_VERSION,
        "task_id": "EIGHT-HOUR-ASSEMBLY-20260817T191449Z",
        "status": FINAL_STATUS,
        "completed_at_utc": data["closeout_at_utc"],
        "user_action": FINAL_USER_ACTION,
        "metrics": data["metrics"],
        "task_status_counts": data["task_status_counts"],
        "final_report": str(FINAL_REPORT),
        "final_report_sha256": _sha256(root / FINAL_REPORT),
        "commands_sha256": _sha256(root / FINAL_COMMANDS),
        "code_snapshot_patch_sha256": _sha256(root / FINAL_DIFF),
        "worktree_status_sha256": _sha256(root / FINAL_WORKTREE),
        "review_bundle_name": bundle_name,
        "review_bundle_sha256_self_reference": None,
        "review_bundle_hash_reported_after_generation": True,
        "simulation_started_by_finalizer": False,
        "robot_commands_emitted_by_finalizer": 0,
        "assembly_success_claimed": False,
        "formal_r12_generated": False,
        "control_authorized": False,
        "hardware_authorized": False,
    }
    _write_new(
        root / FINAL_RESULT,
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True,
                   allow_nan=False) + "\n",
    )
    manifest_sources = list(data["package_source_paths"]) + [
        str(FINAL_REPORT), str(FINAL_RESULT), str(FINAL_COMMANDS),
        str(FINAL_DIFF), str(FINAL_WORKTREE),
    ]
    manifest = {
        "schema_version": 1,
        "generated_at_utc": data["closeout_at_utc"],
        "self_excluded_to_avoid_recursive_hash": True,
        "files": [
            {
                "path": relative,
                "sha256": _sha256(root / relative),
                "size_bytes": (root / relative).stat().st_size,
            }
            for relative in sorted(set(manifest_sources))
        ],
    }
    _write_new(
        root / FINAL_MANIFEST,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True,
                   allow_nan=False) + "\n",
    )
    bundle = root / "artifacts/agent_control/review" / bundle_name
    bundle_result = build_review_zip(
        root=root,
        source_paths=data["package_source_paths"],
        closeout_paths=[root / relative for relative in FINAL_OUTPUTS],
        bundle_path=bundle,
    )
    return {
        "status": FINAL_STATUS,
        "user_action": FINAL_USER_ACTION,
        "final_report": str(FINAL_REPORT),
        "final_result": str(FINAL_RESULT),
        "file_manifest": str(FINAL_MANIFEST),
        "bundle": bundle_result,
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--work-queue", required=True)
    parser.add_argument("--master-state", required=True)
    parser.add_argument("--blocker-ledger", required=True)
    parser.add_argument("--readiness-report", required=True)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--closeout-at-utc", required=True)
    parser.add_argument("--bundle-name", required=True)
    args = parser.parse_args()
    if not args.finalize:
        parser.error("formal outputs require --finalize")
    return args


def main() -> None:
    args = _arguments()
    root = Path(args.repository_root).resolve()
    data = build_final_report_data(
        repository_root=root,
        work_queue_path=args.work_queue,
        master_state_path=args.master_state,
        blocker_ledger_path=args.blocker_ledger,
        readiness_report_path=args.readiness_report,
        preflight_path=args.preflight,
        closeout_at_utc=args.closeout_at_utc,
    )
    status = subprocess.run(
        ["git", "status", "--short", "--branch"], cwd=root,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    worktree = f"$ git status --short --branch\nexit={status.returncode}\n{status.stdout}"
    result = finalize_artifacts(
        repository_root=root,
        data=data,
        bundle_name=args.bundle_name,
        worktree_status=worktree,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "FINAL_OUTPUTS",
    "FINAL_STATUS",
    "FINAL_USER_ACTION",
    "SCHEMA_VERSION",
    "build_code_snapshot_patch",
    "build_final_report_data",
    "build_review_zip",
    "collect_actual_commands",
    "collect_package_sources",
    "finalize_artifacts",
    "render_final_report",
    "verify_review_zip",
]
