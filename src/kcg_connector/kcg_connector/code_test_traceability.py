"""Evidence-bound code/test traceability audit for the eight-hour queue.

This module proves declared-file and validation linkage only.  It deliberately
does not calculate or claim line, branch, or behavioural test coverage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import yaml

from .final_review_preflight import _load_mapping, _resolve_inside, _sha256


SCHEMA_VERSION = "kcg_eight_hour_code_test_traceability_v1"
TASK_ID = "EIGHT-HOUR-G8-CODE-TEST-TRACEABILITY"
PASS_STATUSES = frozenset({"STATIC_PASS", "OFFLINE_PASS", "DYNAMIC_PASS"})
DIRECT_TEST_LINK = "DIRECT_TEST_LINK"
STATIC_VALIDATION_ONLY = "STATIC_VALIDATION_ONLY"


def _is_test_path(relative: str) -> bool:
    path = PurePosixPath(relative)
    return "test" in path.parts or path.name.startswith("test_")


def _verified_declared_file(
    root: Path, row: Mapping[str, Any], label: str
) -> dict[str, Any]:
    relative = row.get("path")
    expected = row.get("sha256")
    if not isinstance(relative, str) or not isinstance(expected, str):
        raise ValueError(f"{label} lacks path or sha256")
    path = _resolve_inside(root, relative, label)
    if not path.is_file():
        raise ValueError(f"{label} missing: {relative}")
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{label} sha256 drift: {relative}")
    return {"path": relative, "sha256": actual}


def _queue_tasks(queue: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    rows: list[tuple[str, Mapping[str, Any]]] = []
    groups = queue.get("groups")
    if not isinstance(groups, Mapping):
        raise ValueError("work queue lacks groups")
    for group in groups.values():
        if not isinstance(group, Mapping) or not isinstance(group.get("tasks"), Mapping):
            raise ValueError("work queue group lacks task mapping")
        rows.extend((str(key), value) for key, value in group["tasks"].items())
    post = queue.get("post_queue_tasks", {})
    if not isinstance(post, Mapping):
        raise ValueError("post_queue_tasks must be a mapping")
    rows.extend((str(key), value) for key, value in post.items())
    if any(not isinstance(value, Mapping) for _key, value in rows):
        raise ValueError("work queue task row must be a mapping")
    return rows


def build_code_test_traceability(
    *,
    repository_root: str | Path,
    work_queue_path: str | Path,
    generated_at_utc: str,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    queue_path = _resolve_inside(root, work_queue_path, "work queue")
    queue = yaml.safe_load(queue_path.read_text(encoding="utf-8"))
    if not isinstance(queue, Mapping):
        raise ValueError("work queue must be a mapping")
    if not isinstance(generated_at_utc, str) or not generated_at_utc.endswith("Z"):
        raise ValueError("generated_at_utc must be explicit UTC")

    passing_tasks = [
        (key, task)
        for key, task in _queue_tasks(queue)
        if task.get("status") in PASS_STATUSES
    ]
    traceability_rows: list[dict[str, Any]] = []
    tasks_without_production: list[dict[str, Any]] = []
    seen_production_paths: set[str] = set()
    all_test_paths: set[str] = set()
    verified_code_file_count = 0

    for task_key, task in passing_tasks:
        evidence = task.get("evidence")
        if not isinstance(evidence, str):
            raise ValueError(f"passing task {task_key} lacks task-result evidence")
        result_path = _resolve_inside(root, evidence, f"{task_key} task result")
        result = _load_mapping(result_path, f"{task_key} task result")
        declared = result.get("code_files", [])
        if not isinstance(declared, list):
            raise ValueError(f"{task_key} code_files must be a list")
        verified = [
            _verified_declared_file(root, row, f"{task_key} declared code")
            for row in declared
            if isinstance(row, Mapping)
        ]
        if len(verified) != len(declared):
            raise ValueError(f"{task_key} code_files contains a non-mapping row")
        verified_code_file_count += len(verified)
        tests = [row for row in verified if _is_test_path(row["path"])]
        production = [row for row in verified if not _is_test_path(row["path"])]
        all_test_paths.update(row["path"] for row in tests)
        if tests and not production:
            raise ValueError(f"{task_key} declares tests without production code")
        if not production:
            tasks_without_production.append(
                {
                    "task_key": task_key,
                    "task_result_path": evidence,
                    "reason": "NO_PRODUCTION_CODE_DECLARED_IN_TASK_RESULT",
                }
            )
            continue

        static_sources: list[dict[str, Any]] = []
        if not tests:
            declared_static = result.get("static_sources")
            if not isinstance(declared_static, list) or not declared_static:
                raise ValueError(f"{task_key} production code lacks test or static evidence")
            for source in declared_static:
                if not isinstance(source, Mapping):
                    raise ValueError(f"{task_key} static source row invalid")
                checked = _verified_declared_file(root, source, f"{task_key} static source")
                checked["status"] = source.get("status")
                static_sources.append(checked)
            if not any(row.get("status") == "PASS" for row in static_sources):
                raise ValueError(f"{task_key} lacks a passing static source")

        for code in production:
            if code["path"] in seen_production_paths:
                raise ValueError(f"production path declared by multiple tasks: {code['path']}")
            seen_production_paths.add(code["path"])
            mode = DIRECT_TEST_LINK if tests else STATIC_VALIDATION_ONLY
            traceability_rows.append(
                {
                    "task_key": task_key,
                    "task_id": result.get("task_id"),
                    "task_status": task.get("status"),
                    "task_result_path": evidence,
                    "task_result_sha256": _sha256(result_path),
                    "production_path": code["path"],
                    "production_sha256": code["sha256"],
                    "traceability_mode": mode,
                    "direct_test_files": tests,
                    "static_validation_sources": static_sources,
                    "coverage_claimed": False,
                    "dynamic_pass_claimed": False,
                }
            )

    traceability_rows.sort(key=lambda row: (row["task_key"], row["production_path"]))
    direct_count = sum(
        row["traceability_mode"] == DIRECT_TEST_LINK for row in traceability_rows
    )
    static_count = sum(
        row["traceability_mode"] == STATIC_VALIDATION_ONLY
        for row in traceability_rows
    )
    dynamic_count = sum(task.get("status") == "DYNAMIC_PASS" for _key, task in passing_tasks)
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "generated_at_utc": generated_at_utc,
        "result": "OFFLINE_PASS",
        "audit_scope": "WORK_QUEUE_EXPLICIT_TASK_RESULT_CODE_FILES_ONLY",
        "passing_task_count": len(passing_tasks),
        "task_with_declared_production_code_count": len(traceability_rows),
        "production_module_count": len(traceability_rows),
        "direct_test_link_count": direct_count,
        "static_validation_only_count": static_count,
        "unique_direct_test_file_count": len(all_test_paths),
        "verified_declared_code_file_count": verified_code_file_count,
        "tasks_without_declared_production_code_count": len(tasks_without_production),
        "tasks_without_declared_production_code": tasks_without_production,
        "missing_file_count": 0,
        "sha256_drift_count": 0,
        "unclassified_production_module_count": 0,
        "dynamic_passed_task_count": dynamic_count,
        "traceability_rows": traceability_rows,
        "coverage_metric": None,
        "coverage_claimed": False,
        "traceability_disclaimer_cn": "直接文件链接或静态证据仅证明可追溯性，不代表行、分支或行为覆盖率。",
        "historical_tests_rerun": False,
        "repository_walk_performed": False,
        "simulation_started": False,
        "robot_commands_emitted": 0,
        "assembly_success_claimed": False,
        "formal_r12_generated": False,
        "control_authorized": False,
        "hardware_authorized": False,
    }


def render_traceability_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# 代码测试可追溯审计",
        "",
        "> 本报告只证明声明文件与验证证据的直接链接，不代表行、分支或行为覆盖率。",
        "",
        "| 任务 | 生产模块 | 分类 | 直接测试数 | 静态证据数 | 动态通过 |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in report["traceability_rows"]:
        lines.append(
            f"| {row['task_key']} | `{row['production_path']}` | "
            f"{row['traceability_mode']} | {len(row['direct_test_files'])} | "
            f"{len(row['static_validation_sources'])} | 否 |"
        )
    lines.extend(
        [
            "",
            "## 汇总",
            "",
            f"- 已通过任务：{report['passing_task_count']}。",
            f"- 生产模块：{report['production_module_count']}。",
            f"- 直接测试链接：{report['direct_test_link_count']}。",
            f"- 仅静态验证：{report['static_validation_only_count']}。",
            f"- 缺失/摘要漂移/未分类：{report['missing_file_count']}/"
            f"{report['sha256_drift_count']}/{report['unclassified_production_module_count']}。",
            "- 历史测试重跑：否；仿真：否；机器人命令：0；动态通过声明：0。",
            "",
        ]
    )
    return "\n".join(lines)


def write_new(path: str | Path, content: str) -> None:
    output = Path(path)
    if output.exists():
        raise FileExistsError("G8 output is immutable")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--work-queue", required=True)
    parser.add_argument("--generated-at-utc", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    root = Path(args.repository_root).resolve()
    output_root = (root / "artifacts/agent_control/tasks" / TASK_ID).resolve()
    json_path = _resolve_inside(root, args.output_json, "G8 JSON output")
    markdown_path = _resolve_inside(root, args.output_markdown, "G8 Markdown output")
    if not json_path.is_relative_to(output_root) or not markdown_path.is_relative_to(output_root):
        raise PermissionError("G8 outputs must remain inside the G8 task directory")
    report = build_code_test_traceability(
        repository_root=root,
        work_queue_path=args.work_queue,
        generated_at_utc=args.generated_at_utc,
    )
    write_new(
        json_path,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
    )
    write_new(markdown_path, render_traceability_markdown(report))


if __name__ == "__main__":
    main()


__all__ = [
    "DIRECT_TEST_LINK",
    "SCHEMA_VERSION",
    "STATIC_VALIDATION_ONLY",
    "build_code_test_traceability",
    "render_traceability_markdown",
    "write_new",
]
