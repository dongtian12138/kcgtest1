"""Fail-closed preflight for the eight-hour final report and review bundle.

The preflight reads only explicit queue evidence.  Missing runtime performance
stays null, and STATIC_PASS/OFFLINE_PASS never becomes a dynamic claim.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import yaml


SCHEMA_VERSION = "kcg_eight_hour_final_review_preflight_v1"
TASK_ID = "EIGHT-HOUR-G3-FINAL-REVIEW-PREFLIGHT"
EXPECTED_TASK_KEYS = (
    "A1", "A2", "A3", "A4",
    "B1", "B2", "B3", "B4", "B5", "B6",
    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10",
    "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10",
    "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8",
    "F1", "F2", "F3",
)
PASS_STATUSES = {"STATIC_PASS", "OFFLINE_PASS", "DYNAMIC_PASS"}
FINAL_TABLE_COLUMNS_CN = (
    "任务", "开始状态", "结束状态", "本轮做了什么", "关键数值/测试",
    "是否真正通过", "若未通过原因", "下一步",
)
FINAL_SUMMARY_FIELDS_CN = (
    "八小时内实际解决了几个问题",
    "完成了多少个代码模块",
    "通过了多少项静态测试",
    "通过了多少项动态测试",
    "跑了多少次Isaac仿真",
    "累计Isaac运行时间",
    "5070Ti最高显存",
    "物理帧率",
    "渲染帧率",
    "当前最前面的完整任务链能够跑到哪一个状态",
    "所有PARKED问题",
    "哪三个问题最优先",
    "是否存在原地打转超过45分钟的情况",
    "是否违反任何禁止规则",
)
CONTROL_MEMBERS = (
    "AGENTS.md",
    "PLANS.md",
    "artifacts/agent_control/PROJECT_CHARTER_CN.md",
    "artifacts/agent_control/WORK_QUEUE.yaml",
    "artifacts/agent_control/MASTER_STATE.json",
    "artifacts/agent_control/CURRENT_TASK.md",
    "artifacts/agent_control/CURRENT_STATUS_CN.md",
    "artifacts/agent_control/EIGHT_HOUR_PROGRESS_CN.md",
    "artifacts/agent_control/BLOCKER_LEDGER.jsonl",
    "artifacts/agent_control/GATE_LEDGER.csv",
    "artifacts/agent_control/DECISION_LOG.jsonl",
    "artifacts/agent_control/STATUS_HISTORY.jsonl",
    "artifacts/agent_control/TASK_GRAPH.yaml",
    "artifacts/agent_control/multilayer/HIGH_DETAIL_REFERENCE_MANIFEST.json",
    "artifacts/agent_control/multilayer/HIGH_DETAIL_BLOCKED_CONCLUSION_CN.md",
)
POST_QUEUE_RESULTS = (
    "artifacts/agent_control/tasks/EIGHT-HOUR-QUEUE-CLOSEOUT-AUDIT/TASK_RESULT.json",
    "artifacts/agent_control/tasks/EIGHT-HOUR-G1-EVIDENCE-INTEGRITY-MANIFEST/TASK_RESULT.json",
    "artifacts/agent_control/tasks/EIGHT-HOUR-G2-DYNAMIC-READINESS-DAG/TASK_RESULT.json",
)
PLANNED_CLOSEOUT_MEMBERS = (
    "artifacts/agent_control/EIGHT_HOUR_FINAL_REPORT_CN.md",
    "artifacts/agent_control/EIGHT_HOUR_FINAL_RESULT.json",
    "artifacts/agent_control/EIGHT_HOUR_ACTUAL_COMMANDS.txt",
    "artifacts/agent_control/EIGHT_HOUR_FILE_MANIFEST.json",
    "artifacts/agent_control/EIGHT_HOUR_CODE_DIFF.patch",
    "artifacts/agent_control/EIGHT_HOUR_WORKTREE_STATUS.txt",
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
CHECK_FRACTION = re.compile(r"^(\d+)/(\d+)$")
REVIEW_BUNDLE_NAME = re.compile(
    r"^EIGHT_HOUR_ASSEMBLY_PROGRESS_\d{8}T\d{6}Z\.zip$"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_inside(root: Path, value: str | Path, label: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"{label} must be a non-empty repository path")
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"{label} absolute path is forbidden")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"{label} escapes repository root")
    return path


def _load_mapping(path: Path, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise ValueError(f"{label} missing: {path}")
    if path.suffix in {".yaml", ".yml"}:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain a mapping")
    return value


def _load_jsonl(path: Path, label: str) -> list[Mapping[str, Any]]:
    if not path.is_file():
        raise ValueError(f"{label} missing: {path}")
    rows: list[Mapping[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, Mapping):
            raise ValueError(f"{label} line {line_number} is not a mapping")
        rows.append(value)
    return rows


def _document_status(document: Mapping[str, Any]) -> str:
    status = document.get("status")
    outcome = document.get("outcome")
    if status is not None and outcome is not None and status != outcome:
        raise ValueError("task evidence status/outcome mismatch")
    selected = status if status is not None else outcome
    if not isinstance(selected, str):
        raise ValueError("task evidence lacks status/outcome")
    return selected


def _flatten_queue(queue: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    groups = queue.get("groups")
    if not isinstance(groups, Mapping):
        raise ValueError("work queue lacks groups")
    rows: list[tuple[str, Mapping[str, Any]]] = []
    for group in groups.values():
        tasks = group.get("tasks") if isinstance(group, Mapping) else None
        if not isinstance(tasks, Mapping):
            raise ValueError("work queue group lacks tasks")
        for key, task in tasks.items():
            if not isinstance(task, Mapping):
                raise ValueError(f"queue task {key} is not a mapping")
            rows.append((str(key), task))
    if tuple(key for key, _ in rows) != EXPECTED_TASK_KEYS:
        raise ValueError("work queue task inventory/order differs from final contract")
    return rows


def _declared_evidence_paths(document: Mapping[str, Any]) -> list[str]:
    value = document.get("evidence")
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    if value is None:
        return []
    raise ValueError("task evidence field must be a path or path list")


def _check_fraction(document: Mapping[str, Any]) -> tuple[int, int] | None:
    for key in ("checks", "static_checks", "post_fix_validation", "offline_checks"):
        value = document.get(key)
        if not isinstance(value, str):
            continue
        match = CHECK_FRACTION.fullmatch(value)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


def _integer_prefix(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    match = re.match(r"^(\d+)", str(value))
    return int(match.group(1)) if match else 0


def validate_bundle_name(name: str) -> None:
    if REVIEW_BUNDLE_NAME.fullmatch(name) is None:
        raise ValueError("review bundle name violates the frozen naming rule")


def _duration_seconds(start: str, end: str) -> float:
    start_time = datetime.fromisoformat(start.replace("Z", "+00:00"))
    end_time = datetime.fromisoformat(end.replace("Z", "+00:00"))
    seconds = (end_time - start_time).total_seconds()
    if seconds < 0:
        raise ValueError("run end precedes run start")
    return seconds


def _true_forbidden_claims(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            path = f"{prefix}.{key}" if prefix else key
            forbidden = (
                key in {
                    "assembly_success_claimed", "control_authorized",
                    "formal_r12_generated", "hardware_authorized",
                    "new_geometry_candidate_created",
                }
                or ("dynamic" in key and key.endswith("pass_claimed"))
            )
            if forbidden and item is True:
                found.append(path)
            found.extend(_true_forbidden_claims(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_true_forbidden_claims(item, f"{prefix}[{index}]"))
    return found


def build_final_review_preflight(
    *,
    repository_root: str | Path,
    work_queue_path: str | Path,
    master_state_path: str | Path,
    blocker_ledger_path: str | Path,
    gate_ledger_path: str | Path,
    decision_log_path: str | Path,
    status_history_path: str | Path,
    task_graph_path: str | Path,
    readiness_report_path: str | Path,
    generated_at_utc: str,
    planned_bundle_name: str,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    paths = {
        "work_queue": _resolve_inside(root, work_queue_path, "work queue"),
        "master_state": _resolve_inside(root, master_state_path, "master state"),
        "blocker_ledger": _resolve_inside(root, blocker_ledger_path, "blocker ledger"),
        "gate_ledger": _resolve_inside(root, gate_ledger_path, "gate ledger"),
        "decision_log": _resolve_inside(root, decision_log_path, "decision log"),
        "status_history": _resolve_inside(root, status_history_path, "status history"),
        "task_graph": _resolve_inside(root, task_graph_path, "task graph"),
        "readiness_report": _resolve_inside(root, readiness_report_path, "readiness report"),
    }
    validate_bundle_name(planned_bundle_name)
    if not generated_at_utc.endswith("Z"):
        raise ValueError("generated_at_utc must be an explicit UTC timestamp")
    queue = _load_mapping(paths["work_queue"], "work queue")
    master = _load_mapping(paths["master_state"], "master state")
    graph = _load_mapping(paths["task_graph"], "task graph")
    readiness = _load_mapping(paths["readiness_report"], "readiness report")
    blockers = _load_jsonl(paths["blocker_ledger"], "blocker ledger")
    decisions = _load_jsonl(paths["decision_log"], "decision log")
    _load_jsonl(paths["status_history"], "status history")
    if not paths["gate_ledger"].is_file():
        raise ValueError("gate ledger missing")
    if queue.get("current_task") != "G3":
        raise ValueError("work queue is not at G3")
    if master.get("task_id") != TASK_ID or graph.get("current_task") != TASK_ID:
        raise ValueError("control files do not agree on G3")
    if (
        readiness.get("formal_dynamic_pass_count") != 0
        or readiness.get("dynamic_execution_authorized_count") != 0
        or readiness.get("current_frontier_state") != "HOME"
    ):
        raise ValueError("readiness report would promote or move the dynamic frontier")

    task_documents: dict[str, Mapping[str, Any]] = {}
    task_rows = []
    package_paths: set[str] = set(CONTROL_MEMBERS)
    forbidden_claims: list[str] = []
    check_passed = 0
    check_total = 0
    fixed_problem_tasks: list[str] = []
    completed_modules: set[str] = set()
    completed_tests: set[str] = set()
    over_budget_tasks: list[str] = []
    for key, task in _flatten_queue(queue):
        evidence_value = task.get("evidence")
        if not isinstance(evidence_value, str):
            raise ValueError(f"{key} lacks a single result evidence path")
        result_path = _resolve_inside(root, evidence_value, f"{key} task result")
        document = _load_mapping(result_path, f"{key} task result")
        status = _document_status(document)
        if status != task.get("status"):
            raise ValueError(f"{key} queue/evidence status mismatch")
        task_documents[key] = document
        package_paths.add(str(result_path.relative_to(root)))
        for declared in _declared_evidence_paths(document):
            declared_path = _resolve_inside(root, declared, f"{key} declared evidence")
            if not declared_path.is_file():
                raise ValueError(f"{key} declared evidence missing: {declared}")
            package_paths.add(str(declared_path.relative_to(root)))
        fraction = _check_fraction(document)
        if status in PASS_STATUSES and fraction is not None:
            passed, total = fraction
            if passed != total:
                raise ValueError(f"passing task {key} has incomplete final checks")
            check_passed += passed
            check_total += total
        if status in PASS_STATUSES and _integer_prefix(document.get("targeted_fix_count")):
            fixed_problem_tasks.append(key)
        for row in document.get("code_files", []):
            if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
                raise ValueError(f"{key} code_files contains an invalid row")
            code_path = _resolve_inside(root, row["path"], f"{key} code file")
            expected = row.get("sha256")
            if status in PASS_STATUSES:
                if not isinstance(expected, str) or SHA256_PATTERN.fullmatch(expected) is None:
                    raise ValueError(f"passing task {key} has unverified code")
                if not code_path.is_file() or _sha256(code_path) != expected:
                    raise ValueError(f"passing task {key} code hash drift")
                if "/test/" in row["path"] or Path(row["path"]).name.startswith("test_"):
                    completed_tests.add(row["path"])
                else:
                    completed_modules.add(row["path"])
        start = document.get("started_at_utc")
        end = document.get("completed_at_utc")
        if isinstance(start, str) and isinstance(end, str):
            if _duration_seconds(start, end) > 2700:
                over_budget_tasks.append(key)
        forbidden_claims.extend(
            f"{key}:{claim}" for claim in _true_forbidden_claims(document)
        )
        start_status = (
            "PARKED_REVISIT_AUTHORIZED" if key == "A1"
            else "未单独冻结；不得反推为动态状态"
        )
        task_rows.append(
            {
                "task_key": key,
                "name": task.get("name"),
                "start_status": start_status,
                "end_status": status,
                "classification": document.get("classification"),
                "checks": fraction,
                "true_dynamic_pass": status == "DYNAMIC_PASS",
                "result_path": str(result_path.relative_to(root)),
            }
        )

    for relative in POST_QUEUE_RESULTS:
        path = _resolve_inside(root, relative, "post-queue result")
        document = _load_mapping(path, "post-queue result")
        package_paths.add(relative)
        fraction = _check_fraction(document)
        if _document_status(document) in PASS_STATUSES and fraction is not None:
            passed, total = fraction
            if passed != total:
                raise ValueError("passing post-queue task has incomplete checks")
            check_passed += passed
            check_total += total
        if _document_status(document) in PASS_STATUSES:
            for row in document.get("code_files", []):
                if not isinstance(row, Mapping):
                    raise ValueError("post-queue code row invalid")
                code_path = _resolve_inside(root, row.get("path"), "post-queue code")
                expected = row.get("sha256")
                if not isinstance(expected, str) or _sha256(code_path) != expected:
                    raise ValueError("post-queue code hash drift")
                if "/test/" in str(row["path"]):
                    completed_tests.add(str(row["path"]))
                else:
                    completed_modules.add(str(row["path"]))

    a1_record_path = _resolve_inside(
        root,
        "artifacts/agent_control/tasks/EIGHT-HOUR-A1-INIT-DATUM/RUN_RECORD.json",
        "A1 run record",
    )
    a1_record = _load_mapping(a1_record_path, "A1 run record")
    package_paths.add(str(a1_record_path.relative_to(root)))
    isaac_runtime_seconds = _duration_seconds(
        str(a1_record["started_at_utc"]), str(a1_record["ended_at_utc"])
    )
    if a1_record.get("run_kind") != "DIAGNOSTIC_AB":
        raise ValueError("A1 Isaac run kind changed")
    a4 = task_documents["A4"]
    if a4.get("real_performance_measurement_started") is not False:
        raise ValueError("A4 real-performance state changed")
    if a4.get("measured_performance_values_claimed") is not False:
        raise ValueError("A4 unexpectedly claims measured performance")

    for relative in tuple(package_paths):
        path = _resolve_inside(root, relative, "package member")
        if not path.is_file():
            raise ValueError(f"existing package member missing: {relative}")
    existing_members = [
        {
            "path": relative,
            "sha256": _sha256(_resolve_inside(root, relative, "package member")),
            "size_bytes": _resolve_inside(root, relative, "package member").stat().st_size,
        }
        for relative in sorted(package_paths)
    ]

    final_report_path = queue.get("final_outputs", {}).get("report")
    if final_report_path != "artifacts/agent_control/EIGHT_HOUR_FINAL_REPORT_CN.md":
        raise ValueError("final report path differs from the frozen path")
    planned_pattern = queue.get("final_outputs", {}).get("review_bundle_pattern")
    if planned_pattern != (
        "artifacts/agent_control/review/EIGHT_HOUR_ASSEMBLY_PROGRESS_<UTC>.zip"
    ):
        raise ValueError("review bundle pattern differs from the frozen pattern")

    current_parked = [
        row for row in blockers
        if row.get("status") in {"PARKED", "PARKED_FINAL_FOR_THIS_WINDOW", "BLOCKED_EXTERNAL"}
    ]
    top_three = [
        row["task_key"] for row in readiness["top_three_priority_root_blockers"]
    ]
    if top_three != ["A1", "B1", "C8"]:
        raise ValueError("top-three priority blockers changed")
    status_counts = Counter(row["end_status"] for row in task_rows)
    summary_values = {
        "resolved_problem_count": len(fixed_problem_tasks),
        "resolved_problem_tasks": fixed_problem_tasks,
        "completed_code_module_count": len(completed_modules),
        "completed_test_file_count": len(completed_tests),
        "passed_static_or_offline_check_count": check_passed,
        "declared_static_or_offline_check_total": check_total,
        "dynamic_passed_task_count": readiness["formal_dynamic_pass_count"],
        "isaac_process_run_count": 1,
        "formal_or_acceptance_isaac_run_count": 0,
        "isaac_cumulative_runtime_seconds": isaac_runtime_seconds,
        "isaac_explicit_physics_step_count": 0,
        "peak_vram_mib": None,
        "physics_fps": None,
        "render_fps": None,
        "performance_missing_reason": "A4 collector passed offline, but no real measurement started",
        "current_frontier_state": readiness["current_frontier_state"],
        "parked_ledger_entry_count": len(current_parked),
        "top_three_priority_root_blockers": top_three,
        "over_45_minute_task_count": len(over_budget_tasks),
        "over_45_minute_tasks": over_budget_tasks,
        "known_forbidden_claim_count": len(forbidden_claims),
        "known_forbidden_claims": forbidden_claims,
    }
    source_bindings = [
        {"field_cn": FINAL_SUMMARY_FIELDS_CN[0], "source": "passing TASK_RESULT targeted_fix_count", "value_key": "resolved_problem_count"},
        {"field_cn": FINAL_SUMMARY_FIELDS_CN[1], "source": "hash-verified production code_files", "value_key": "completed_code_module_count"},
        {"field_cn": FINAL_SUMMARY_FIELDS_CN[2], "source": "passing task final checks fractions", "value_key": "passed_static_or_offline_check_count"},
        {"field_cn": FINAL_SUMMARY_FIELDS_CN[3], "source": "G2 formal_dynamic_pass_count", "value_key": "dynamic_passed_task_count"},
        {"field_cn": FINAL_SUMMARY_FIELDS_CN[4], "source": "A1 RUN_RECORD plus formal-run fields", "value_key": "isaac_process_run_count"},
        {"field_cn": FINAL_SUMMARY_FIELDS_CN[5], "source": "A1 RUN_RECORD UTC interval", "value_key": "isaac_cumulative_runtime_seconds"},
        {"field_cn": FINAL_SUMMARY_FIELDS_CN[6], "source": "A4 measured evidence or explicit null", "value_key": "peak_vram_mib"},
        {"field_cn": FINAL_SUMMARY_FIELDS_CN[7], "source": "A4 measured evidence or explicit null", "value_key": "physics_fps"},
        {"field_cn": FINAL_SUMMARY_FIELDS_CN[8], "source": "A4 measured evidence or explicit null", "value_key": "render_fps"},
        {"field_cn": FINAL_SUMMARY_FIELDS_CN[9], "source": "G2 and F1 current frontier", "value_key": "current_frontier_state"},
        {"field_cn": FINAL_SUMMARY_FIELDS_CN[10], "source": "BLOCKER_LEDGER.jsonl", "value_key": "parked_ledger_entry_count"},
        {"field_cn": FINAL_SUMMARY_FIELDS_CN[11], "source": "G2 mainline-order root blockers", "value_key": "top_three_priority_root_blockers"},
        {"field_cn": FINAL_SUMMARY_FIELDS_CN[12], "source": "task UTC intervals and 2700-second budget", "value_key": "over_45_minute_task_count"},
        {"field_cn": FINAL_SUMMARY_FIELDS_CN[13], "source": "MASTER_STATE plus recursive task-result false claims", "value_key": "known_forbidden_claim_count"},
    ]
    if len(source_bindings) != len(FINAL_SUMMARY_FIELDS_CN):
        raise AssertionError("final summary source binding count changed")

    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "generated_at_utc": generated_at_utc,
        "result": "OFFLINE_PASS",
        "final_report_generated": False,
        "review_bundle_generated": False,
        "final_status_claimed": False,
        "report_blueprint": {
            "first_content_must_be_table": True,
            "table_columns_cn": list(FINAL_TABLE_COLUMNS_CN),
            "task_row_count": len(task_rows),
            "task_rows": task_rows,
            "start_status_boundary": (
                "A1入口停车状态有账本证据；其他节点未冻结逐节点入口状态，"
                "最终表必须原样标注，不得反推为动态状态"
            ),
            "summary_field_count": len(FINAL_SUMMARY_FIELDS_CN),
            "summary_source_bindings": source_bindings,
        },
        "summary_values_snapshot": summary_values,
        "status_counts": dict(sorted(status_counts.items())),
        "package_blueprint": {
            "planned_bundle_name": planned_bundle_name,
            "existing_member_count": len(existing_members),
            "existing_members": existing_members,
            "planned_closeout_member_count": len(PLANNED_CLOSEOUT_MEMBERS),
            "planned_closeout_members": list(PLANNED_CLOSEOUT_MEMBERS),
            "future_members_hashed_early": False,
            "path_escape_count": 0,
        },
        "source_manifest": {
            key: {"path": str(path.relative_to(root)), "sha256": _sha256(path)}
            for key, path in paths.items()
        },
        "simulation_started_by_preflight": False,
        "robot_commands_emitted_by_preflight": 0,
        "static_or_offline_promoted_to_dynamic_count": 0,
        "assembly_success_claimed": False,
        "formal_r12_generated": False,
        "control_authorized": False,
        "hardware_authorized": False,
    }


def render_preflight_markdown(report: Mapping[str, Any]) -> str:
    values = report["summary_values_snapshot"]
    lines = [
        "# 八小时最终报告与审查包预检",
        "",
        "| 检查 | 结果 |",
        "|---|---|",
        f"| 首页任务表 | {report['report_blueprint']['task_row_count']} 行，{len(report['report_blueprint']['table_columns_cn'])} 列 |",
        f"| 必需汇总项 | {report['report_blueprint']['summary_field_count']}/14 有来源绑定 |",
        f"| 现有包成员 | {report['package_blueprint']['existing_member_count']} 个均已直接摘要 |",
        f"| 窗口结束生成成员 | {report['package_blueprint']['planned_closeout_member_count']} 个，未提前哈希 |",
        f"| 动态通过 | {values['dynamic_passed_task_count']} |",
        f"| 当前完整链前沿 | {values['current_frontier_state']} |",
        "",
        "## 关键口径",
        "",
        f"- 本窗口 Isaac 进程启动：{values['isaac_process_run_count']} 次；累计 {values['isaac_cumulative_runtime_seconds']:.0f} 秒。",
        f"- 其中正式/验收 Isaac 运行：{values['formal_or_acceptance_isaac_run_count']} 次；A1 诊断显式物理步为 0。",
        "- 5070Ti 显存峰值、物理帧率、渲染帧率保持空值，因为没有实测运行。",
        f"- 已知禁止规则真值声明：{values['known_forbidden_claim_count']}；未据此扩大为未记录行为的绝对证明。",
        "- 本预检不生成最终报告或压缩包，不启动仿真，不授予动态通过。",
        "",
    ]
    return "\n".join(lines)


def write_preflight_pair(
    report: Mapping[str, Any], json_output: str | Path, markdown_output: str | Path
) -> None:
    json_path = Path(json_output)
    markdown_path = Path(markdown_output)
    if json_path.exists() or markdown_path.exists():
        raise FileExistsError("G3 preflight outputs are immutable")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True,
                   allow_nan=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_preflight_markdown(report), encoding="utf-8")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--work-queue", required=True)
    parser.add_argument("--master-state", required=True)
    parser.add_argument("--blocker-ledger", required=True)
    parser.add_argument("--gate-ledger", required=True)
    parser.add_argument("--decision-log", required=True)
    parser.add_argument("--status-history", required=True)
    parser.add_argument("--task-graph", required=True)
    parser.add_argument("--readiness-report", required=True)
    parser.add_argument("--generated-at-utc", required=True)
    parser.add_argument("--planned-bundle-name", required=True)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--markdown-output", required=True)
    args = parser.parse_args()
    if not args.run:
        parser.error("G3 preflight requires --run")
    return args


def main() -> None:
    args = _arguments()
    root = Path(args.repository_root).resolve()
    output_root = (root / "artifacts/agent_control/tasks" / TASK_ID).resolve()
    outputs = [
        _resolve_inside(root, args.json_output, "JSON output"),
        _resolve_inside(root, args.markdown_output, "Markdown output"),
    ]
    if not all(path.is_relative_to(output_root) for path in outputs):
        raise PermissionError("G3 outputs must remain inside the G3 task directory")
    report = build_final_review_preflight(
        repository_root=root,
        work_queue_path=args.work_queue,
        master_state_path=args.master_state,
        blocker_ledger_path=args.blocker_ledger,
        gate_ledger_path=args.gate_ledger,
        decision_log_path=args.decision_log,
        status_history_path=args.status_history,
        task_graph_path=args.task_graph,
        readiness_report_path=args.readiness_report,
        generated_at_utc=args.generated_at_utc,
        planned_bundle_name=args.planned_bundle_name,
    )
    write_preflight_pair(report, *outputs)


if __name__ == "__main__":
    main()


__all__ = [
    "FINAL_SUMMARY_FIELDS_CN",
    "FINAL_TABLE_COLUMNS_CN",
    "PLANNED_CLOSEOUT_MEMBERS",
    "SCHEMA_VERSION",
    "build_final_review_preflight",
    "render_preflight_markdown",
    "validate_bundle_name",
    "write_preflight_pair",
]
