"""Fail-closed dynamic-readiness DAG for the eight-hour assembly queue.

This module only inventories frozen queue/evidence state.  It does not run a
simulator, authorize control, or promote STATIC_PASS/OFFLINE_PASS evidence to a
dynamic result.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


SCHEMA_VERSION = "kcg_eight_hour_dynamic_readiness_dag_v1"
TASK_ID = "EIGHT-HOUR-G2-DYNAMIC-READINESS-DAG"
EXPECTED_TASK_KEYS = (
    "A1", "A2", "A3", "A4",
    "B1", "B2", "B3", "B4", "B5", "B6",
    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10",
    "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10",
    "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8",
    "F1", "F2", "F3",
)
IMPLEMENTATION_READY_STATUSES = {"STATIC_PASS", "OFFLINE_PASS", "DYNAMIC_PASS"}
ALLOWED_STATUSES = {
    "NOT_STARTED", "IMPLEMENTING", "STATIC_PASS", "OFFLINE_PASS",
    "DYNAMIC_PASS", "PARKED", "BLOCKED_EXTERNAL",
}

# Concrete dynamic prerequisites.  The raw queue descriptions remain in every
# node as source text; these edges are the frozen, auditable interpretation used
# for the dynamic main line and F1's normal sequence.
DEFAULT_DEPENDENCIES: Mapping[str, tuple[str, ...]] = {
    "A1": (),
    "A2": ("A1",),
    "A3": ("A2",),
    "A4": (),
    "B1": (),
    "B2": ("B1",),
    "B3": ("B2",),
    "B4": ("B3",),
    "B5": (),
    "B6": ("B2", "B3", "B4", "B5"),
    "C1": (),
    "C2": (),
    "C3": ("B4",),
    "C4": ("C1", "C2", "C3"),
    "C5": ("C4",),
    "C6": ("C5",),
    "C7": ("C6",),
    "C8": ("C4",),
    "C9": ("C3", "C4", "C5", "C6", "C7", "C8"),
    "C10": ("C9",),
    "D1": ("C10",),
    "D2": ("D1", "C10"),
    "D3": ("D2",),
    "D4": ("D3",),
    "D5": ("D4",),
    "D6": ("D5",),
    "D7": ("A2", "D6"),
    "D8": ("D7",),
    "D9": ("D8", "C3"),
    "D10": ("D8", "D9"),
    "E1": ("D7",),
    "E2": ("E1",),
    "E3": ("E2",),
    "E4": ("E3",),
    "E5": ("E4",),
    "E6": ("E5",),
    "E7": ("E6",),
    "E8": ("E7",),
    "F1": ("A2", "B6", "C10", "D7", "D10", "E8"),
    "F2": ("F1", "D10"),
    "F3": ("F1", "F2", "A4"),
}

ROOT_BLOCKER_SPECS = (
    ("A1", "A1", "PARKED_FINAL_FOR_THIS_WINDOW"),
    ("B1", "B1", "AUTHORITATIVE_GRASP_BOUNDARY_MISSING"),
    (
        "C8", "EIGHT-HOUR-C8-FOUNDATIONPOSE-ADAPTER",
        "FOUNDATIONPOSE_RUNTIME_AND_CURRENT_MODEL_INPUTS_NOT_READY",
    ),
    ("E3", "E3", "LEGACY_REGRASP_INPUT_HASH_DRIFT"),
    (
        "E6", "E6_DYNAMIC",
        "RELATIVE_PERIODIC_RESISTANCE_DERIVED_RUNTIME_PHASE_FAIL_CLOSED",
    ),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_inside(root: Path, value: str | Path, label: str) -> Path:
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


def _document_status(document: Mapping[str, Any], label: str) -> str:
    status = document.get("status")
    outcome = document.get("outcome")
    if status is not None and outcome is not None and status != outcome:
        raise ValueError(f"{label} status/outcome mismatch")
    selected = status if status is not None else outcome
    if selected not in ALLOWED_STATUSES:
        raise ValueError(f"{label} lacks a supported status/outcome")
    return str(selected)


def _load_blocker_rows(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, Mapping):
            raise ValueError(f"blocker ledger line {line_number} is not a mapping")
        rows.append(value)
    return rows


def _flatten_tasks(queue: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    groups = queue.get("groups")
    if not isinstance(groups, Mapping):
        raise ValueError("work queue lacks groups")
    flattened: list[tuple[str, Mapping[str, Any]]] = []
    for group_name, group in groups.items():
        tasks = group.get("tasks") if isinstance(group, Mapping) else None
        if not isinstance(tasks, Mapping):
            raise ValueError(f"work queue group {group_name} lacks tasks")
        for key, task in tasks.items():
            if not isinstance(task, Mapping):
                raise ValueError(f"work queue task {key} must be a mapping")
            flattened.append((str(key), task))
    keys = tuple(key for key, _ in flattened)
    if keys != EXPECTED_TASK_KEYS:
        raise ValueError("work queue task inventory/order differs from G2 contract")
    return flattened


def topological_order(
    dependencies: Mapping[str, Sequence[str]],
    expected_keys: Sequence[str] = EXPECTED_TASK_KEYS,
) -> list[str]:
    expected = tuple(expected_keys)
    if tuple(dependencies) != expected:
        raise ValueError("dependency inventory/order differs from task contract")
    expected_set = set(expected)
    indegree = {key: 0 for key in expected}
    children = {key: [] for key in expected}
    for child, parents in dependencies.items():
        if len(tuple(parents)) != len(set(parents)):
            raise ValueError(f"duplicate dependency for {child}")
        for parent in parents:
            if parent not in expected_set:
                raise ValueError(f"unknown dependency {parent!r} for {child}")
            indegree[child] += 1
            children[parent].append(child)
    queue = deque(key for key in expected if indegree[key] == 0)
    ordered: list[str] = []
    while queue:
        key = queue.popleft()
        ordered.append(key)
        for child in children[key]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if len(ordered) != len(expected):
        raise ValueError("dynamic-readiness dependency graph contains a cycle")
    return ordered


def _descendants(
    dependencies: Mapping[str, Sequence[str]], key: str
) -> list[str]:
    children = {item: [] for item in dependencies}
    for child, parents in dependencies.items():
        for parent in parents:
            children[parent].append(child)
    found: set[str] = set()
    pending = list(children[key])
    while pending:
        child = pending.pop()
        if child in found:
            continue
        found.add(child)
        pending.extend(children[child])
    return [item for item in EXPECTED_TASK_KEYS if item in found]


def _strict_dynamic_evidence(document: Mapping[str, Any]) -> bool:
    return (
        document.get("simulation_started") is True
        and document.get("dynamic_pass_claimed") is True
        and document.get("independent_process") is True
        and document.get("controller_truth_used") is False
        and document.get("post_run_pose_write_count") == 0
        and document.get("hardware_authorized") is False
    )


def build_dynamic_readiness_dag(
    *,
    repository_root: str | Path,
    work_queue_path: str | Path,
    blocker_ledger_path: str | Path,
    gate_ledger_path: str | Path,
    generated_at_utc: str,
    dependencies: Mapping[str, Sequence[str]] = DEFAULT_DEPENDENCIES,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    queue_path = _resolve_inside(root, work_queue_path, "work queue")
    blocker_path = _resolve_inside(root, blocker_ledger_path, "blocker ledger")
    gate_path = _resolve_inside(root, gate_ledger_path, "gate ledger")
    queue = _load_mapping(queue_path, "work queue")
    blocker_rows = _load_blocker_rows(blocker_path)
    if not gate_path.is_file():
        raise ValueError(f"gate ledger missing: {gate_path}")
    if not isinstance(generated_at_utc, str) or not generated_at_utc.endswith("Z"):
        raise ValueError("generated_at_utc must be an explicit UTC timestamp")
    ordered = topological_order(dependencies)
    tasks = _flatten_tasks(queue)
    order_index = {key: index for index, key in enumerate(EXPECTED_TASK_KEYS)}

    nodes: list[dict[str, Any]] = []
    evidence_documents: dict[str, Mapping[str, Any]] = {}
    for key, task in tasks:
        status = task.get("status")
        if status not in ALLOWED_STATUSES:
            raise ValueError(f"{key} has unsupported status {status!r}")
        evidence_value = task.get("evidence")
        if not isinstance(evidence_value, str):
            raise ValueError(f"{key} lacks a single evidence path")
        evidence_path = _resolve_inside(root, evidence_value, f"{key} evidence")
        evidence = _load_mapping(evidence_path, f"{key} evidence")
        evidence_status = _document_status(evidence, f"{key} evidence")
        if evidence_status != status:
            raise ValueError(
                f"{key} queue/evidence status mismatch: {status!r} != {evidence_status!r}"
            )
        evidence_documents[key] = evidence
        dynamic_pass = status == "DYNAMIC_PASS"
        if dynamic_pass and not _strict_dynamic_evidence(evidence):
            raise ValueError(f"{key} DYNAMIC_PASS lacks strict independent evidence")
        requested_authorization = task.get("execution_authorized", False)
        if requested_authorization is True and not dynamic_pass:
            raise ValueError(f"{key} non-dynamic task claims execution authorization")
        unresolved = [
            parent
            for parent in dependencies[key]
            if dict(tasks)[parent].get("status") != "DYNAMIC_PASS"
        ]
        nodes.append(
            {
                "task_key": key,
                "name": task.get("name"),
                "queue_status": status,
                "queue_dynamic_status": task.get("dynamic_status"),
                "implementation_ready": status in IMPLEMENTATION_READY_STATUSES,
                "formal_dynamic_passed": dynamic_pass,
                "dynamic_execution_authorized": (
                    requested_authorization is True and dynamic_pass
                ),
                "raw_dynamic_dependency": task.get("dynamic_dependency"),
                "dynamic_dependencies": list(dependencies[key]),
                "unresolved_dynamic_dependencies": unresolved,
                "dynamic_ready_now": dynamic_pass and not unresolved,
                "evidence_path": str(evidence_path.relative_to(root)),
                "evidence_sha256": _sha256(evidence_path),
                "evidence_status": evidence_status,
            }
        )

    ledger_by_task: dict[str, Mapping[str, Any]] = {}
    for row in blocker_rows:
        task_id = row.get("task_id")
        if isinstance(task_id, str):
            ledger_by_task[task_id] = row
    root_blockers = []
    for key, ledger_task, expected_classification in ROOT_BLOCKER_SPECS:
        evidence = evidence_documents[key]
        ledger = ledger_by_task.get(ledger_task)
        evidence_classification = evidence.get("classification")
        if evidence_classification == expected_classification:
            source_kind = "task_evidence"
            source_path = next(
                node["evidence_path"] for node in nodes if node["task_key"] == key
            )
        elif ledger is not None and ledger.get("classification") == expected_classification:
            source_kind = "blocker_ledger"
            source_path = str(blocker_path.relative_to(root))
        else:
            raise ValueError(
                f"root blocker {key} classification is absent or changed: "
                f"{expected_classification}"
            )
        descendants = _descendants(dependencies, key)
        root_blockers.append(
            {
                "task_key": key,
                "classification": expected_classification,
                "source_kind": source_kind,
                "source_path": source_path,
                "mainline_order": order_index[key],
                "blocked_descendant_count": len(descendants),
                "blocked_descendants": descendants,
            }
        )
    root_blockers.sort(key=lambda row: row["mainline_order"])

    status_counts = Counter(node["queue_status"] for node in nodes)
    f1 = evidence_documents["F1"]
    if f1.get("current_state") != "HOME":
        raise ValueError("F1 current frontier differs from frozen HOME evidence")
    a2 = next(node for node in nodes if node["task_key"] == "A2")
    if a2["formal_dynamic_passed"]:
        raise ValueError("A2 unexpectedly carries dynamic-pass evidence")

    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "generated_at_utc": generated_at_utc,
        "result": "OFFLINE_PASS",
        "queue_id": queue.get("queue_id"),
        "queue_current_task": queue.get("current_task"),
        "node_count": len(nodes),
        "edge_count": sum(len(value) for value in dependencies.values()),
        "topological_order": ordered,
        "status_counts": {
            key: status_counts.get(key, 0) for key in sorted(ALLOWED_STATUSES)
        },
        "implementation_ready_count": sum(
            node["implementation_ready"] for node in nodes
        ),
        "parked_queue_node_count": status_counts.get("PARKED", 0),
        "formal_dynamic_pass_count": sum(
            node["formal_dynamic_passed"] for node in nodes
        ),
        "dynamic_execution_authorized_count": sum(
            node["dynamic_execution_authorized"] for node in nodes
        ),
        "current_frontier_state": "HOME",
        "earliest_unmet_formal_dynamic_gate": {
            "task_key": "A2",
            "name": a2["name"],
            "direct_root_blocker": "A1",
            "classification": "PARKED_FINAL_FOR_THIS_WINDOW",
        },
        "root_blockers": root_blockers,
        "top_three_priority_root_blockers": root_blockers[:3],
        "nodes": nodes,
        "source_manifest": {
            "work_queue": {
                "path": str(queue_path.relative_to(root)),
                "sha256": _sha256(queue_path),
            },
            "blocker_ledger": {
                "path": str(blocker_path.relative_to(root)),
                "sha256": _sha256(blocker_path),
            },
            "gate_ledger": {
                "path": str(gate_path.relative_to(root)),
                "sha256": _sha256(gate_path),
            },
        },
        "static_or_offline_promoted_to_dynamic_count": 0,
        "simulation_started_by_builder": False,
        "robot_commands_emitted_by_builder": 0,
        "assembly_success_claimed": False,
        "control_authorized": False,
        "formal_r12_generated": False,
        "hardware_authorized": False,
    }


def render_dot(report: Mapping[str, Any]) -> str:
    lines = ["digraph dynamic_readiness {", "  rankdir=LR;"]
    for node in report["nodes"]:
        color = (
            "green" if node["formal_dynamic_passed"]
            else "red" if node["queue_status"] == "PARKED"
            else "orange"
        )
        label = f"{node['task_key']}\\n{node['queue_status']}"
        lines.append(f'  "{node["task_key"]}" [label="{label}", color={color}];')
    for node in report["nodes"]:
        for parent in node["dynamic_dependencies"]:
            lines.append(f'  "{parent}" -> "{node["task_key"]}";')
    lines.append("}")
    return "\n".join(lines) + "\n"


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# 动态就绪依赖图",
        "",
        "| 节点 | 队列状态 | 实现就绪 | 动态通过 | 动态授权 | 未满足动态依赖 |",
        "|---|---|---:|---:|---:|---|",
    ]
    for node in report["nodes"]:
        unresolved = ", ".join(node["unresolved_dynamic_dependencies"]) or "—"
        lines.append(
            f"| {node['task_key']} | {node['queue_status']} | "
            f"{node['implementation_ready']} | {node['formal_dynamic_passed']} | "
            f"{node['dynamic_execution_authorized']} | {unresolved} |"
        )
    gate = report["earliest_unmet_formal_dynamic_gate"]
    lines.extend(
        [
            "",
            "## 结论",
            "",
            f"- 当前完整链前沿：`{report['current_frontier_state']}`。",
            f"- 实现就绪：{report['implementation_ready_count']}/41；"
            f"队列停车：{report['parked_queue_node_count']}/41。",
            f"- 正式动态通过：{report['formal_dynamic_pass_count']}/41；"
            f"显式动态授权：{report['dynamic_execution_authorized_count']}/41。",
            f"- 最早未满足正式动态门：{gate['task_key']}（{gate['name']}）；"
            f"直接根阻塞：{gate['direct_root_blocker']}。",
            "- 前三优先根问题："
            + "、".join(
                f"{row['task_key']}:{row['classification']}"
                for row in report["top_three_priority_root_blockers"]
            )
            + "。",
            "- 排序依据是 A 至 F 主链中首次出现位置；不使用主观严重度评分。",
            "- STATIC_PASS/OFFLINE_PASS 仅表示实现证据，不构成动态验收或控制授权。",
            "- 本生成器未启动仿真、未发机器人命令、未生成正式 R12。",
            "",
        ]
    )
    return "\n".join(lines)


def write_report_triplet(
    report: Mapping[str, Any],
    json_output: str | Path,
    dot_output: str | Path,
    markdown_output: str | Path,
) -> None:
    paths = [Path(json_output), Path(dot_output), Path(markdown_output)]
    if any(path.exists() for path in paths):
        raise FileExistsError("G2 report outputs are immutable")
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    paths[0].write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True,
                   allow_nan=False) + "\n",
        encoding="utf-8",
    )
    paths[1].write_text(render_dot(report), encoding="utf-8")
    paths[2].write_text(render_markdown(report), encoding="utf-8")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--work-queue", required=True)
    parser.add_argument("--blocker-ledger", required=True)
    parser.add_argument("--gate-ledger", required=True)
    parser.add_argument("--generated-at-utc", required=True)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--dot-output", required=True)
    parser.add_argument("--markdown-output", required=True)
    args = parser.parse_args()
    if not args.run:
        parser.error("G2 report generation requires --run")
    return args


def main() -> None:
    args = _arguments()
    root = Path(args.repository_root).resolve()
    output_root = (root / "artifacts/agent_control/tasks" / TASK_ID).resolve()
    outputs = [
        _resolve_inside(root, args.json_output, "JSON output"),
        _resolve_inside(root, args.dot_output, "DOT output"),
        _resolve_inside(root, args.markdown_output, "Markdown output"),
    ]
    if not all(path.is_relative_to(output_root) for path in outputs):
        raise PermissionError("G2 outputs must remain inside the G2 task directory")
    report = build_dynamic_readiness_dag(
        repository_root=root,
        work_queue_path=args.work_queue,
        blocker_ledger_path=args.blocker_ledger,
        gate_ledger_path=args.gate_ledger,
        generated_at_utc=args.generated_at_utc,
    )
    write_report_triplet(report, *outputs)


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_DEPENDENCIES",
    "EXPECTED_TASK_KEYS",
    "SCHEMA_VERSION",
    "TASK_ID",
    "build_dynamic_readiness_dag",
    "render_dot",
    "render_markdown",
    "topological_order",
    "write_report_triplet",
]
