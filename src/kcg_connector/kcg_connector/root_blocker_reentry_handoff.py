"""Evidence-bound re-entry handoff for the five dynamic root blockers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from .final_review_preflight import (
    _load_jsonl,
    _load_mapping,
    _resolve_inside,
    _sha256,
)


SCHEMA_VERSION = "kcg_eight_hour_root_blocker_reentry_handoff_v1"
TASK_ID = "EIGHT-HOUR-G6-ROOT-BLOCKER-REENTRY-HANDOFF"
ROOT_ORDER = ("A1", "B1", "C8", "E3", "E6")
EXPECTED_CLASSIFICATIONS = {
    "A1": "PARKED_FINAL_FOR_THIS_WINDOW",
    "B1": "AUTHORITATIVE_GRASP_BOUNDARY_MISSING",
    "C8": "FOUNDATIONPOSE_RUNTIME_AND_CURRENT_MODEL_INPUTS_NOT_READY",
    "E3": "LEGACY_REGRASP_INPUT_HASH_DRIFT",
    "E6": "RELATIVE_PERIODIC_RESISTANCE_DERIVED_RUNTIME_PHASE_FAIL_CLOSED",
}
RESULT_PATHS = {
    "A1": "artifacts/agent_control/tasks/EIGHT-HOUR-A1-INIT-DATUM/TASK_RESULT.json",
    "B1": "artifacts/agent_control/tasks/EIGHT-HOUR-B1-GRASP-REGION-COM/TASK_RESULT.json",
    "C8": "artifacts/agent_control/tasks/EIGHT-HOUR-C8-FOUNDATIONPOSE-ADAPTER/TASK_RESULT.json",
    "E3": "artifacts/agent_control/tasks/EIGHT-HOUR-E3-REGRASP-NUT/TASK_RESULT.json",
    "E6": "artifacts/agent_control/tasks/EIGHT-HOUR-E6-ANTI-DECOUPLING-RESISTANCE/TASK_RESULT.json",
}
LEDGER_TASK_IDS = {
    "A1": "A1",
    "B1": "B1",
    "C8": "EIGHT-HOUR-C8-FOUNDATIONPOSE-ADAPTER",
    "E3": "E3",
    "E6": "E6_DYNAMIC",
}
REENTRY = {
    "A1": {
        "prerequisites": [
            "出现新的直接证据、此前遗漏的冻结基准参数，或能区分不同根因的试验",
            "下一窗口重新授予一次访问预算",
            "不得重复显式contactOffset/restOffset假设",
        ],
        "first_action": "只读核对主合同、生成映射与初始化报告中的datum定义；只有找到新的冻结参数才设计一次不同试验",
        "proof_required": "把226.026微米误差降到50微米内，且不是运行后位姿写入或禁用碰撞",
        "third_visit_forbidden_this_window": True,
    },
    "B1": {
        "prerequisites": [
            "出现权威抓取区域边界",
            "出现权威禁抓功能区域边界",
            "出现区域到控制模型prim的明确映射",
        ],
        "first_action": "对新权威来源做只读来源和摘要审计；三项边界齐备后再生成逐指目标",
        "proof_required": "每个接触目标可追溯且不侵入键、螺纹、密封、插针或锁紧螺母功能面",
        "third_visit_forbidden_this_window": False,
    },
    "C8": {
        "prerequisites": [
            "当前多层外观模型OBJ输入存在并摘要冻结",
            "FoundationPose容器或等效官方运行时可用",
            "图像派生实例掩码桥可用",
            "输入来自真实RGB-D证据而不是仿真真值",
        ],
        "first_action": "先做只读运行时与输入清单复核；四项同时满足后才允许运行一次推理",
        "proof_required": "独立进程推理产生可追溯姿态，控制输入不含物体真值、接触名或接触法向",
        "third_visit_forbidden_this_window": False,
    },
    "E3": {
        "prerequisites": [
            "冻结wrist-FT配置摘要与当前用户文件变化得到明确来源决议",
            "当前多层CouplingNut目标权威输入可用",
            "不覆盖或吸收用户未提交修改",
        ],
        "first_action": "先生成两份摘要的只读语义差异，不写配置；只有来源决议后刷新依赖链",
        "proof_required": "重抓适配器完整加载且目标来自允许的视觉/腕力证据，不使用世界真值",
        "third_visit_forbidden_this_window": False,
    },
    "E6": {
        "prerequisites": [
            "E5螺纹轴向随动取得动态通过",
            "防松齿绝对相位原点有权威来源或明确等效授权",
            "反向0.5287860809763352牛米峰值不得裁剪成0.30牛米以内",
        ],
        "first_action": "只读审计绝对相位来源；未授权时保持相对周期模型离线状态",
        "proof_required": "动态防松关系在0.30牛米分量门内可验证，且不靠裁剪、位姿写入或门限放宽",
        "third_visit_forbidden_this_window": False,
    },
}


def _ledger_row(
    rows: list[Mapping[str, Any]], task_id: str, expected: str
) -> Mapping[str, Any]:
    matches = [row for row in rows if row.get("task_id") == task_id]
    for row in reversed(matches):
        if row.get("classification") == expected:
            return row
    # A1's final ledger entry intentionally uses result rather than a
    # classification field; its status itself is the frozen classification.
    if task_id == "A1":
        for row in reversed(matches):
            if row.get("status") == expected:
                return row
    raise ValueError(f"ledger lacks expected classification for {task_id}")


def build_root_blocker_handoff(
    *,
    repository_root: str | Path,
    blocker_ledger_path: str | Path,
    readiness_report_path: str | Path,
    generated_at_utc: str,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    ledger_path = _resolve_inside(root, blocker_ledger_path, "blocker ledger")
    readiness_path = _resolve_inside(root, readiness_report_path, "readiness report")
    ledger = _load_jsonl(ledger_path, "blocker ledger")
    readiness = _load_mapping(readiness_path, "readiness report")
    if not isinstance(generated_at_utc, str) or not generated_at_utc.endswith("Z"):
        raise ValueError("generated_at_utc must be explicit UTC")
    root_rows = readiness.get("root_blockers")
    if not isinstance(root_rows, list):
        raise ValueError("readiness report lacks root blockers")
    by_key = {row.get("task_key"): row for row in root_rows if isinstance(row, Mapping)}
    if tuple(row.get("task_key") for row in root_rows) != ROOT_ORDER:
        raise ValueError("root blocker order changed")
    top_three = [
        row.get("task_key")
        for row in readiness.get("top_three_priority_root_blockers", [])
        if isinstance(row, Mapping)
    ]
    if top_three != ["A1", "B1", "C8"]:
        raise ValueError("top-three root blocker order changed")

    entries = []
    for rank, key in enumerate(ROOT_ORDER, 1):
        expected = EXPECTED_CLASSIFICATIONS[key]
        readiness_row = by_key[key]
        if readiness_row.get("classification") != expected:
            raise ValueError(f"{key} readiness classification drift")
        result_path = _resolve_inside(root, RESULT_PATHS[key], f"{key} task result")
        result = _load_mapping(result_path, f"{key} task result")
        result_classification = result.get("classification")
        if key == "C8":
            # C8's TASK_RESULT classifies static interface readiness; the dynamic
            # blocker is intentionally recorded in the blocker ledger.
            if result_classification != "FOUNDATIONPOSE_ADAPTER_STATIC_READY_ONLY":
                raise ValueError("C8 static result classification drift")
        elif result_classification != expected:
            raise ValueError(f"{key} task result classification drift")
        ledger_row = _ledger_row(ledger, LEDGER_TASK_IDS[key], expected)
        reentry = REENTRY[key]
        entries.append(
            {
                "task_key": key,
                "priority_tier": "TOP_THREE" if rank <= 3 else "SECONDARY_ROOT",
                "mainline_rank": rank,
                "classification": expected,
                "blocked_descendant_count": readiness_row.get("blocked_descendant_count"),
                "blocked_descendants": readiness_row.get("blocked_descendants"),
                "task_result_path": str(result_path.relative_to(root)),
                "task_result_sha256": _sha256(result_path),
                "ledger_task_id": LEDGER_TASK_IDS[key],
                "ledger_timestamp_utc": ledger_row.get("timestamp_utc"),
                "reentry_prerequisites": reentry["prerequisites"],
                "next_window_first_evidence_action": reentry["first_action"],
                "acceptance_proof_required": reentry["proof_required"],
                "execution_authorized_this_window": False,
                "proposed_command": None,
                "revisit_count_incremented": False,
                "third_visit_forbidden_this_window": reentry["third_visit_forbidden_this_window"],
                "geometry_change_authorized": False,
                "threshold_change_authorized": False,
                "higher_force_or_moment_authorized": False,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "generated_at_utc": generated_at_utc,
        "result": "OFFLINE_PASS",
        "root_blocker_count": len(entries),
        "top_three_priority_root_blockers": top_three,
        "priority_basis": "A至F主链首次出现位置，不使用主观严重度评分",
        "entries": entries,
        "current_window_action": "NO_REVISIT_OR_EXECUTION",
        "all_execution_authorized_this_window": False,
        "all_proposed_commands_null": all(row["proposed_command"] is None for row in entries),
        "source_manifest": {
            "blocker_ledger": {
                "path": str(ledger_path.relative_to(root)),
                "sha256": _sha256(ledger_path),
            },
            "readiness_report": {
                "path": str(readiness_path.relative_to(root)),
                "sha256": _sha256(readiness_path),
            },
        },
        "simulation_started": False,
        "robot_commands_emitted": 0,
        "assembly_success_claimed": False,
        "formal_r12_generated": False,
        "control_authorized": False,
        "hardware_authorized": False,
    }


def render_handoff_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# 根阻塞重入交接",
        "",
        "| 顺序 | 根阻塞 | 分类 | 阻塞后代数 | 本窗口执行授权 | 下一窗口首个证据动作 |",
        "|---:|---|---|---:|---:|---|",
    ]
    for row in report["entries"]:
        lines.append(
            f"| {row['mainline_rank']} | {row['task_key']} | "
            f"{row['classification']} | {row['blocked_descendant_count']} | "
            f"{row['execution_authorized_this_window']} | "
            f"{row['next_window_first_evidence_action']} |"
        )
    lines.extend(
        [
            "",
            "## 边界",
            "",
            "- 本文件不授权本窗口重访、运行或安装外部资源。",
            "- 所有 `proposed_command` 均为 null；先满足证据前提，再由下一窗口冻结唯一命令。",
            "- A1 本窗口第三次访问明确禁止；B1 不得猜测抓取边界；C8 不得用真值输入替代运行时/模型输入。",
            "- E3 不覆盖用户现有配置；E6 不裁剪0.5287860809763352牛米反向峰值，也不放宽0.30牛米门限。",
            "- 不修改几何、正式门限、8牛力分量上限或0.30牛米力矩分量上限。",
            "",
        ]
    )
    return "\n".join(lines)


def write_handoff_pair(
    report: Mapping[str, Any], json_output: str | Path, markdown_output: str | Path
) -> None:
    paths = [Path(json_output), Path(markdown_output)]
    if any(path.exists() for path in paths):
        raise FileExistsError("G6 handoff outputs are immutable")
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    paths[0].write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True,
                   allow_nan=False) + "\n",
        encoding="utf-8",
    )
    paths[1].write_text(render_handoff_markdown(report), encoding="utf-8")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--blocker-ledger", required=True)
    parser.add_argument("--readiness-report", required=True)
    parser.add_argument("--generated-at-utc", required=True)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--markdown-output", required=True)
    args = parser.parse_args()
    if not args.run:
        parser.error("G6 handoff requires --run")
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
        raise PermissionError("G6 outputs must remain inside the G6 task directory")
    report = build_root_blocker_handoff(
        repository_root=root,
        blocker_ledger_path=args.blocker_ledger,
        readiness_report_path=args.readiness_report,
        generated_at_utc=args.generated_at_utc,
    )
    write_handoff_pair(report, *outputs)


if __name__ == "__main__":
    main()


__all__ = [
    "EXPECTED_CLASSIFICATIONS",
    "ROOT_ORDER",
    "SCHEMA_VERSION",
    "build_root_blocker_handoff",
    "render_handoff_markdown",
    "write_handoff_pair",
]
