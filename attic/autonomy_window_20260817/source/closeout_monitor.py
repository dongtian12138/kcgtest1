"""Pre-deadline package closure and in-memory final-report monitor."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .eight_hour_final_artifacts import (
    FINAL_OUTPUTS,
    build_final_report_data,
    render_final_report,
)
from .final_review_preflight import _resolve_inside, _sha256


SCHEMA_VERSION = "kcg_eight_hour_closeout_monitor_v1"
TASK_ID = "EIGHT-HOUR-G10-CLOSEOUT-MONITOR"
G9_COMPLETION_PATHS = (
    "src/kcg_connector/kcg_connector/final_integrity_checkpoint.py",
    "src/kcg_connector/test/test_final_integrity_checkpoint.py",
    "artifacts/agent_control/tasks/EIGHT-HOUR-G9-FINAL-INTEGRITY-CHECKPOINT/RUN_PLAN.json",
    "artifacts/agent_control/tasks/EIGHT-HOUR-G9-FINAL-INTEGRITY-CHECKPOINT/VALIDATION.json",
    "artifacts/agent_control/tasks/EIGHT-HOUR-G9-FINAL-INTEGRITY-CHECKPOINT/FINAL_INTEGRITY_CHECKPOINT.json",
    "artifacts/agent_control/tasks/EIGHT-HOUR-G9-FINAL-INTEGRITY-CHECKPOINT/FINAL_INTEGRITY_CHECKPOINT_CN.md",
    "artifacts/agent_control/tasks/EIGHT-HOUR-G9-FINAL-INTEGRITY-CHECKPOINT/TASK_RESULT.json",
)
G10_SELF_INPUT_PATHS = (
    "src/kcg_connector/kcg_connector/closeout_monitor.py",
    "src/kcg_connector/test/test_closeout_monitor.py",
    "artifacts/agent_control/tasks/EIGHT-HOUR-G10-CLOSEOUT-MONITOR/RUN_PLAN.json",
)


def _snapshot(root: Path, paths: Sequence[str], label: str) -> list[dict[str, Any]]:
    if len(paths) != len(set(paths)):
        raise ValueError(f"{label} contains duplicate paths")
    rows = []
    for relative in paths:
        path = _resolve_inside(root, relative, label)
        if not path.is_file():
            raise ValueError(f"{label} missing: {relative}")
        rows.append(
            {"path": relative, "sha256": _sha256(path), "size_bytes": path.stat().st_size}
        )
    return rows


def build_closeout_monitor(
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
    if not isinstance(generated_at_utc, str) or not generated_at_utc.endswith("Z"):
        raise ValueError("generated_at_utc must be explicit UTC")
    data = build_final_report_data(
        repository_root=root,
        work_queue_path=work_queue_path,
        master_state_path=master_state_path,
        blocker_ledger_path=blocker_ledger_path,
        readiness_report_path=readiness_report_path,
        preflight_path=preflight_path,
        closeout_at_utc=generated_at_utc,
    )
    package_paths = set(data["package_source_paths"])
    g9_rows = _snapshot(root, G9_COMPLETION_PATHS, "G9 completion source")
    missing = [row["path"] for row in g9_rows if row["path"] not in package_paths]
    if missing:
        raise ValueError(f"G9 source missing from package closure: {missing}")
    self_rows = _snapshot(root, G10_SELF_INPUT_PATHS, "G10 self input")

    rendered = render_final_report(data)
    rendered_lines = rendered.splitlines()
    summary_numbers = [
        int(match.group(1))
        for match in re.finditer(r"^(\d+)\. ", rendered, flags=re.MULTILINE)
    ]
    if rendered_lines[:3] != [
        "# 八小时自主任务最终报告",
        "",
        "| 任务 | 开始状态 | 结束状态 | 本轮做了什么 | 关键数值/测试 | 是否真正通过 | 若未通过原因 | 下一步 |",
    ]:
        raise ValueError("final report does not begin with the required table")
    if summary_numbers != list(range(1, 15)):
        raise ValueError("final report does not contain exactly fourteen ordered summaries")
    formal_outputs = [
        str(relative) for relative in FINAL_OUTPUTS if (root / relative).exists()
    ]
    if formal_outputs:
        raise ValueError(f"formal outputs appeared before deadline: {formal_outputs}")
    metrics = data["metrics"]
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "generated_at_utc": generated_at_utc,
        "result": "OFFLINE_PASS",
        "g9_expected_closure_source_count": len(g9_rows),
        "g9_closed_source_count": len(g9_rows),
        "g9_missing_from_package_closure_count": 0,
        "g9_closure_sources": g9_rows,
        "g10_self_input_count": len(self_rows),
        "g10_self_inputs": self_rows,
        "g10_self_closure_deferred_to_g11": True,
        "task_table_row_count": len(data["task_rows"]),
        "summary_item_count": len(summary_numbers),
        "summary_item_numbers": summary_numbers,
        "report_begins_with_required_table": True,
        "in_memory_report_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        "in_memory_report_size_bytes": len(rendered.encode("utf-8")),
        "package_source_count": len(package_paths),
        "code_module_count": metrics["completed_code_module_count"],
        "test_file_count": metrics["completed_test_file_count"],
        "passed_static_or_offline_check_count": metrics["passed_static_or_offline_check_count"],
        "declared_static_or_offline_check_total": metrics["declared_static_or_offline_check_total"],
        "dynamic_passed_task_count": metrics["dynamic_passed_task_count"],
        "current_frontier_state": metrics["current_frontier_state"],
        "formal_final_output_count": 0,
        "formal_final_outputs": [],
        "formal_report_written": False,
        "historical_tests_rerun": False,
        "repository_walk_performed": False,
        "simulation_started": False,
        "robot_commands_emitted": 0,
        "assembly_success_claimed": False,
        "formal_r12_generated": False,
        "control_authorized": False,
        "hardware_authorized": False,
    }


def render_monitor_markdown(report: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# 截止前最终报告闭包监控",
            "",
            "| 检查 | 数值 | 结果 |",
            "|---|---:|---|",
            f"| G9包源闭包 | {report['g9_closed_source_count']}/{report['g9_expected_closure_source_count']} | PASS |",
            f"| 首页任务表 | {report['task_table_row_count']}行 | 首屏为表 |",
            f"| 最终汇总 | {report['summary_item_count']}项 | 1至14完整 |",
            f"| 代码/测试 | {report['code_module_count']}/{report['test_file_count']} | 摘要约束 |",
            f"| 正式结束输出 | {report['formal_final_output_count']} | 截止前保持0 |",
            "",
            "- 最终报告仅在内存渲染，未写入正式路径。",
            "- G10自身完成源由G11复核闭包。",
            "- 动态通过0；完整链HOME；正式R12未生成；控制与硬件授权false。",
            "",
        ]
    )


def write_new(path: str | Path, content: str) -> None:
    output = Path(path)
    if output.exists():
        raise FileExistsError("G10 output is immutable")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--work-queue", required=True)
    parser.add_argument("--master-state", required=True)
    parser.add_argument("--blocker-ledger", required=True)
    parser.add_argument("--readiness-report", required=True)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--generated-at-utc", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    root = Path(args.repository_root).resolve()
    output_root = (root / "artifacts/agent_control/tasks" / TASK_ID).resolve()
    json_path = _resolve_inside(root, args.output_json, "G10 JSON output")
    markdown_path = _resolve_inside(root, args.output_markdown, "G10 Markdown output")
    if not json_path.is_relative_to(output_root) or not markdown_path.is_relative_to(output_root):
        raise PermissionError("G10 outputs must remain inside the G10 task directory")
    report = build_closeout_monitor(
        repository_root=root,
        work_queue_path=args.work_queue,
        master_state_path=args.master_state,
        blocker_ledger_path=args.blocker_ledger,
        readiness_report_path=args.readiness_report,
        preflight_path=args.preflight,
        generated_at_utc=args.generated_at_utc,
    )
    write_new(
        json_path,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
    )
    write_new(markdown_path, render_monitor_markdown(report))


if __name__ == "__main__":
    main()


__all__ = [
    "G10_SELF_INPUT_PATHS",
    "G9_COMPLETION_PATHS",
    "SCHEMA_VERSION",
    "build_closeout_monitor",
    "render_monitor_markdown",
    "write_new",
]
