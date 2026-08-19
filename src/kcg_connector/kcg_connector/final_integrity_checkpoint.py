"""Post-queue integrity checkpoint without creating formal closeout files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .eight_hour_final_artifacts import FINAL_OUTPUTS, build_final_report_data
from .final_review_preflight import _load_mapping, _resolve_inside, _sha256
from .stable_closeout_integrity import check_stable_manifest


SCHEMA_VERSION = "kcg_eight_hour_final_integrity_checkpoint_v1"
TASK_ID = "EIGHT-HOUR-G9-FINAL-INTEGRITY-CHECKPOINT"
G7_G8_CLOSURE_PATHS = (
    "src/kcg_connector/kcg_connector/stable_closeout_integrity.py",
    "src/kcg_connector/test/test_stable_closeout_integrity.py",
    "artifacts/agent_control/tasks/EIGHT-HOUR-G7-CLOSEOUT-INTEGRITY-MONITOR/RUN_PLAN.json",
    "artifacts/agent_control/tasks/EIGHT-HOUR-G7-CLOSEOUT-INTEGRITY-MONITOR/VALIDATION.json",
    "artifacts/agent_control/tasks/EIGHT-HOUR-G7-CLOSEOUT-INTEGRITY-MONITOR/STABLE_SOURCE_BASELINE.json",
    "artifacts/agent_control/tasks/EIGHT-HOUR-G7-CLOSEOUT-INTEGRITY-MONITOR/INITIAL_CHECKPOINT.json",
    "artifacts/agent_control/tasks/EIGHT-HOUR-G7-CLOSEOUT-INTEGRITY-MONITOR/TASK_RESULT.json",
    "src/kcg_connector/kcg_connector/code_test_traceability.py",
    "src/kcg_connector/test/test_code_test_traceability.py",
    "artifacts/agent_control/tasks/EIGHT-HOUR-G8-CODE-TEST-TRACEABILITY/RUN_PLAN.json",
    "artifacts/agent_control/tasks/EIGHT-HOUR-G8-CODE-TEST-TRACEABILITY/VALIDATION.json",
    "artifacts/agent_control/tasks/EIGHT-HOUR-G8-CODE-TEST-TRACEABILITY/CODE_TEST_TRACEABILITY.json",
    "artifacts/agent_control/tasks/EIGHT-HOUR-G8-CODE-TEST-TRACEABILITY/CODE_TEST_TRACEABILITY_CN.md",
    "artifacts/agent_control/tasks/EIGHT-HOUR-G8-CODE-TEST-TRACEABILITY/TASK_RESULT.json",
)
G9_SELF_INPUT_PATHS = (
    "src/kcg_connector/kcg_connector/final_integrity_checkpoint.py",
    "src/kcg_connector/test/test_final_integrity_checkpoint.py",
    "artifacts/agent_control/tasks/EIGHT-HOUR-G9-FINAL-INTEGRITY-CHECKPOINT/RUN_PLAN.json",
)


def _snapshot_paths(root: Path, paths: Sequence[str], label: str) -> list[dict[str, Any]]:
    if len(paths) != len(set(paths)):
        raise ValueError(f"{label} contains duplicate paths")
    rows = []
    for relative in paths:
        path = _resolve_inside(root, relative, label)
        if not path.is_file():
            raise ValueError(f"{label} missing: {relative}")
        rows.append(
            {
                "path": relative,
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return rows


def build_final_integrity_checkpoint(
    *,
    repository_root: str | Path,
    work_queue_path: str | Path,
    master_state_path: str | Path,
    blocker_ledger_path: str | Path,
    readiness_report_path: str | Path,
    preflight_path: str | Path,
    stable_baseline_path: str | Path,
    generated_at_utc: str,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    if not isinstance(generated_at_utc, str) or not generated_at_utc.endswith("Z"):
        raise ValueError("generated_at_utc must be explicit UTC")
    baseline_path = _resolve_inside(root, stable_baseline_path, "G7 stable baseline")
    baseline = _load_mapping(baseline_path, "G7 stable baseline")
    stable_check = check_stable_manifest(
        repository_root=root,
        baseline=baseline,
        checked_at_utc=generated_at_utc,
    )
    if stable_check["result"] != "PASS":
        raise ValueError("G7 stable baseline has missing or drifted sources")

    final_data = build_final_report_data(
        repository_root=root,
        work_queue_path=work_queue_path,
        master_state_path=master_state_path,
        blocker_ledger_path=blocker_ledger_path,
        readiness_report_path=readiness_report_path,
        preflight_path=preflight_path,
        closeout_at_utc=generated_at_utc,
    )
    package_paths = set(final_data["package_source_paths"])
    closure_rows = _snapshot_paths(root, G7_G8_CLOSURE_PATHS, "G7/G8 closure source")
    missing_from_closure = [
        row["path"] for row in closure_rows if row["path"] not in package_paths
    ]
    if missing_from_closure:
        raise ValueError(f"G7/G8 source missing from package closure: {missing_from_closure}")
    self_rows = _snapshot_paths(root, G9_SELF_INPUT_PATHS, "G9 self input")

    formal_outputs = [
        str(relative) for relative in FINAL_OUTPUTS if (root / relative).exists()
    ]
    if formal_outputs:
        raise ValueError(f"formal outputs appeared before deadline: {formal_outputs}")
    metrics = final_data["metrics"]
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "generated_at_utc": generated_at_utc,
        "result": "OFFLINE_PASS",
        "g7_baseline_source_count": stable_check["baseline_source_count"],
        "g7_checked_source_count": stable_check["checked_source_count"],
        "g7_missing_source_count": stable_check["missing_source_count"],
        "g7_drift_source_count": stable_check["drift_source_count"],
        "g7_baseline_manifest_digest": stable_check["baseline_manifest_digest"],
        "g7_current_manifest_digest": stable_check["current_manifest_digest"],
        "g7_g8_expected_closure_source_count": len(closure_rows),
        "g7_g8_closed_source_count": len(closure_rows),
        "g7_g8_missing_from_package_closure_count": 0,
        "g7_g8_closure_sources": closure_rows,
        "g9_self_input_count": len(self_rows),
        "g9_self_inputs": self_rows,
        "g9_self_closure_deferred_to_g10": True,
        "current_task_table_row_count": len(final_data["task_rows"]),
        "current_package_source_count": len(package_paths),
        "current_code_module_count": metrics["completed_code_module_count"],
        "current_test_file_count": metrics["completed_test_file_count"],
        "dynamic_passed_task_count": metrics["dynamic_passed_task_count"],
        "current_frontier_state": metrics["current_frontier_state"],
        "formal_final_output_count": 0,
        "formal_final_outputs": [],
        "historical_tests_rerun": False,
        "repository_walk_performed": False,
        "simulation_started": False,
        "robot_commands_emitted": 0,
        "assembly_success_claimed": False,
        "formal_r12_generated": False,
        "control_authorized": False,
        "hardware_authorized": False,
    }


def render_checkpoint_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# 最终完整性检查点",
        "",
        "| 检查 | 数值 | 结果 |",
        "|---|---:|---|",
        f"| G7原始稳定源 | {report['g7_baseline_source_count']} | 缺失 {report['g7_missing_source_count']}，漂移 {report['g7_drift_source_count']} |",
        f"| G7/G8包源闭包 | {report['g7_g8_closed_source_count']}/{report['g7_g8_expected_closure_source_count']} | PASS |",
        f"| G9自身输入冻结 | {report['g9_self_input_count']} | 交G10闭包复核 |",
        f"| 当前任务表 | {report['current_task_table_row_count']} | 仅静态/离线证据 |",
        f"| 正式结束输出 | {report['formal_final_output_count']} | 截止前保持0 |",
        "",
        "## 边界",
        "",
        "- G9不能在自身完成前证明自身结果已进入包源闭包；该项明确交由G10复核。",
        "- 历史测试未重跑；Isaac未启动；机器人命令0；动态通过0；完整链仍在HOME。",
        "- 正式R12未生成；控制与硬件授权均为false。",
        "",
    ]
    return "\n".join(lines)


def write_new(path: str | Path, content: str) -> None:
    output = Path(path)
    if output.exists():
        raise FileExistsError("G9 output is immutable")
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
    parser.add_argument("--stable-baseline", required=True)
    parser.add_argument("--generated-at-utc", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    root = Path(args.repository_root).resolve()
    output_root = (root / "artifacts/agent_control/tasks" / TASK_ID).resolve()
    json_path = _resolve_inside(root, args.output_json, "G9 JSON output")
    markdown_path = _resolve_inside(root, args.output_markdown, "G9 Markdown output")
    if not json_path.is_relative_to(output_root) or not markdown_path.is_relative_to(output_root):
        raise PermissionError("G9 outputs must remain inside the G9 task directory")
    report = build_final_integrity_checkpoint(
        repository_root=root,
        work_queue_path=args.work_queue,
        master_state_path=args.master_state,
        blocker_ledger_path=args.blocker_ledger,
        readiness_report_path=args.readiness_report,
        preflight_path=args.preflight,
        stable_baseline_path=args.stable_baseline,
        generated_at_utc=args.generated_at_utc,
    )
    write_new(
        json_path,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
    )
    write_new(markdown_path, render_checkpoint_markdown(report))


if __name__ == "__main__":
    main()


__all__ = [
    "G7_G8_CLOSURE_PATHS",
    "G9_SELF_INPUT_PATHS",
    "SCHEMA_VERSION",
    "build_final_integrity_checkpoint",
    "render_checkpoint_markdown",
    "write_new",
]
