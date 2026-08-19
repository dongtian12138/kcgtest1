"""Evidence-graded report builder for the eight-hour assembly work queue.

The reporter inventories existing evidence.  It never runs Isaac, creates
video, samples performance, grants control, or promotes an offline result to a
dynamic pass.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import yaml


SCHEMA_VERSION = "kcg_eight_hour_assembly_evidence_report_v1"
VIDEO_MANIFEST_SCHEMA = "kcg_eight_hour_video_manifest_v1"
TASK_ID = "EIGHT-HOUR-F3-EVIDENCE-REPORT"
EXPECTED_TASK_KEYS = (
    "A1", "A2", "A3", "A4",
    "B1", "B2", "B3", "B4", "B5", "B6",
    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10",
    "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10",
    "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8",
    "F1", "F2", "F3",
)
ALLOWED_STATUSES = {
    "NOT_STARTED",
    "IMPLEMENTING",
    "STATIC_PASS",
    "OFFLINE_PASS",
    "DYNAMIC_PASS",
    "PARKED",
    "BLOCKED_EXTERNAL",
}
EVIDENCE_EXPECTED_STATUSES = {
    "STATIC_PASS",
    "OFFLINE_PASS",
    "DYNAMIC_PASS",
    "PARKED",
    "BLOCKED_EXTERNAL",
}
PERFORMANCE_ROOT = Path(
    "artifacts/agent_control/tasks/EIGHT-HOUR-A4-PERFORMANCE-MEASURED"
)
VIDEO_ROOT = Path("artifacts/agent_control/runtime_evidence/videos")
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mkv", ".webm"}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _resolve_inside(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"{label} must be a non-empty repository path")
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"{label} absolute path is forbidden")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"{label} escapes repository root")
    return path


def _document_status(document: Mapping[str, Any], label: str) -> str:
    status = document.get("status")
    outcome = document.get("outcome")
    if status is not None and outcome is not None and status != outcome:
        raise ValueError(f"{label} status/outcome mismatch")
    selected = status if status is not None else outcome
    if selected not in ALLOWED_STATUSES:
        raise ValueError(f"{label} lacks a supported status/outcome")
    return str(selected)


def _true_claim_paths(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            path = f"{prefix}.{key}" if prefix else key
            dynamic_claim = (
                key in {
                    "assembly_success_claimed",
                    "control_authorized",
                    "formal_acceptance_claimed",
                    "formal_physics_pass_claimed",
                    "formal_r12_generated",
                    "hardware_authorized",
                }
                or ("dynamic" in key and key.endswith("pass_claimed"))
            )
            if dynamic_claim and item is True:
                found.append(path)
            found.extend(_true_claim_paths(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_true_claim_paths(item, f"{prefix}[{index}]"))
    return found


def _validate_declared_code_files(
    root: Path,
    document: Mapping[str, Any],
    document_status: str,
) -> tuple[int, int]:
    rows = document.get("code_files")
    if rows is None:
        return 0, 0
    if not isinstance(rows, list):
        raise ValueError("code_files must be a list when present")
    checked = 0
    unverified = 0
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"code_files[{index}] must be a mapping")
        path = _resolve_inside(root, row.get("path"), f"code_files[{index}]")
        if not path.is_file():
            raise ValueError(f"code_files[{index}] is incomplete")
        expected = row.get("sha256")
        if expected is None:
            validation_status = row.get("validation_status")
            if (
                document_status == "PARKED"
                and isinstance(validation_status, str)
                and validation_status
            ):
                unverified += 1
                continue
            raise ValueError(f"code_files[{index}] is incomplete")
        if not isinstance(expected, str):
            raise ValueError(f"code_files[{index}] SHA-256 is invalid")
        if SHA256_PATTERN.fullmatch(expected) is None:
            raise ValueError(f"code_files[{index}] SHA-256 is invalid")
        if _sha256(path) != expected:
            raise ValueError(f"declared code file hash drift: {row.get('path')}")
        checked += 1
    return checked, unverified


def _validate_dynamic_envelope(document: Mapping[str, Any], task_key: str) -> None:
    required = {
        "evidence_level": "DYNAMIC_INDEPENDENT_PROCESS",
        "independent_process": True,
        "simulation_started": True,
        "dynamic_pass_claimed": True,
        "controller_truth_used": False,
        "post_run_pose_write_count": 0,
        "hardware_authorized": False,
    }
    for key, expected in required.items():
        if document.get(key) != expected:
            raise ValueError(
                f"{task_key} DYNAMIC_PASS lacks strict {key}={expected!r}"
            )
    revision = document.get("source_revision_sha256")
    if not isinstance(revision, str) or SHA256_PATTERN.fullmatch(revision) is None:
        raise ValueError(f"{task_key} DYNAMIC_PASS lacks source revision SHA-256")


def _task_evidence(
    root: Path,
    task_key: str,
    task: Mapping[str, Any],
) -> dict[str, Any]:
    status = task.get("status")
    evidence_value = task.get("evidence")
    if evidence_value is None:
        return {
            "state": (
                "MISSING_EXPECTED_EVIDENCE"
                if status in EVIDENCE_EXPECTED_STATUSES
                else "NOT_APPLICABLE_YET"
            ),
            "path": None,
            "sha256": None,
            "document_status": None,
            "classification": task.get("classification"),
            "checks": task.get("checks"),
            "declared_code_files_checked": 0,
            "declared_code_files_unverified": 0,
            "true_dynamic_claim_paths": [],
            "simulation_started": False,
            "current_state": None,
            "current_rejection_code": None,
        }
    path = _resolve_inside(root, evidence_value, f"{task_key} evidence")
    document = _load_mapping(path, f"{task_key} evidence")
    document_status = _document_status(document, f"{task_key} evidence")
    if document_status != status:
        raise ValueError(
            f"{task_key} queue/evidence status mismatch: "
            f"{status!r} != {document_status!r}"
        )
    true_claims = _true_claim_paths(document)
    if status == "DYNAMIC_PASS":
        _validate_dynamic_envelope(document, task_key)
    elif true_claims:
        raise ValueError(
            f"{task_key} non-dynamic evidence contains true claims: {true_claims}"
        )
    checked, unverified = _validate_declared_code_files(
        root, document, document_status
    )
    return {
        "state": "VERIFIED_PRESENT",
        "path": str(path.relative_to(root)),
        "sha256": _sha256(path),
        "document_status": document_status,
        "classification": document.get("classification"),
        "checks": document.get("checks"),
        "declared_code_files_checked": checked,
        "declared_code_files_unverified": unverified,
        "true_dynamic_claim_paths": true_claims,
        "simulation_started": document.get("simulation_started", False),
        "current_state": document.get("current_state"),
        "current_rejection_code": document.get("current_rejection_code"),
    }


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite numeric data")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite numeric data")
    return result


def _performance_inventory(root: Path, value: Any) -> dict[str, Any]:
    empty = {
        "available": False,
        "path": None,
        "sha256": None,
        "gpu_name": None,
        "target_process_vram_peak_mib": None,
        "physics_steps_per_wall_second": None,
        "render_fps": None,
        "measurement_kind": None,
    }
    if value is None:
        return empty
    path = _resolve_inside(root, value, "performance summary")
    expected_root = (root / PERFORMANCE_ROOT).resolve()
    if not path.is_relative_to(expected_root):
        raise ValueError("performance summary is outside the measured A4 root")
    document = _load_mapping(path, "performance summary")
    if (
        document.get("status") != "MEASURED"
        or document.get("measurement_kind") != "REAL"
        or document.get("offline_fixture") is not False
        or document.get("gpu_name") != "NVIDIA GeForce RTX 5070 Ti"
        or document.get("dynamic_pass_claimed") is not False
        or document.get("hardware_authorized") is not False
    ):
        raise ValueError("performance summary is not real process-bound evidence")
    physics = document.get("physics_steps_per_wall_second")
    render = document.get("render")
    if not isinstance(physics, Mapping) or not isinstance(render, Mapping):
        raise ValueError("performance summary lacks physics or render statistics")
    values = physics.get("values")
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError("performance summary requires exactly three physics rates")
    rates = [_finite(item, "physics rate") for item in values]
    return {
        "available": True,
        "path": str(path.relative_to(root)),
        "sha256": _sha256(path),
        "gpu_name": document["gpu_name"],
        "target_process_vram_peak_mib": _finite(
            document.get("target_process_vram_peak_mib"), "VRAM peak"
        ),
        "physics_steps_per_wall_second": rates,
        "render_fps": _finite(render.get("overall_fps"), "render FPS"),
        "measurement_kind": "REAL",
    }


def _video_inventory(root: Path, value: Any) -> dict[str, Any]:
    if value is None:
        return {
            "available": False,
            "manifest_path": None,
            "manifest_sha256": None,
            "count": 0,
            "files": [],
            "playback_content_verified": False,
        }
    manifest_path = _resolve_inside(root, value, "video manifest")
    manifest = _load_mapping(manifest_path, "video manifest")
    if manifest.get("schema_version") != VIDEO_MANIFEST_SCHEMA:
        raise ValueError("video manifest schema is unsupported")
    rows = manifest.get("videos")
    if not isinstance(rows, list):
        raise ValueError("video manifest videos must be a list")
    files = []
    expected_root = (root / VIDEO_ROOT).resolve()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"video row {index} must be a mapping")
        path = _resolve_inside(root, row.get("path"), f"video row {index}")
        if not path.is_relative_to(expected_root):
            raise ValueError("video file is outside the runtime video root")
        if path.suffix.lower() not in ALLOWED_VIDEO_SUFFIXES or not path.is_file():
            raise ValueError("video file is missing or has an unsupported suffix")
        expected = row.get("sha256")
        if not isinstance(expected, str) or _sha256(path) != expected:
            raise ValueError("video SHA-256 mismatch")
        if (
            row.get("source") != "isaac_runtime_capture"
            or row.get("synthetic") is not False
            or row.get("simulation_started") is not True
            or not isinstance(row.get("source_process_pid"), int)
            or row.get("source_process_pid") <= 0
            or path.stat().st_size <= 0
        ):
            raise ValueError("video row is not attributable runtime evidence")
        files.append(
            {
                "path": str(path.relative_to(root)),
                "sha256": expected,
                "size_bytes": path.stat().st_size,
                "role": row.get("role"),
                "source_process_pid": row.get("source_process_pid"),
            }
        )
    return {
        "available": bool(files),
        "manifest_path": str(manifest_path.relative_to(root)),
        "manifest_sha256": _sha256(manifest_path),
        "count": len(files),
        "files": files,
        "playback_content_verified": False,
    }


def _verify_prior_report(
    root: Path,
    value: Any,
    task_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if value is None:
        return {"checked": False, "path": None, "common_evidence_count": 0}
    path = _resolve_inside(root, value, "prior report")
    prior = _load_mapping(path, "prior report")
    if prior.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("prior report schema is unsupported")
    previous = {
        row.get("task_key"): row.get("evidence", {}).get("sha256")
        for row in prior.get("tasks", [])
        if isinstance(row, Mapping) and isinstance(row.get("evidence"), Mapping)
    }
    current = {
        row.get("task_key"): row.get("evidence", {}).get("sha256")
        for row in task_rows
        if isinstance(row.get("evidence"), Mapping)
    }
    common = sorted(
        key for key in previous.keys() & current.keys()
        if previous[key] is not None and current[key] is not None
    )
    drift = [key for key in common if previous[key] != current[key]]
    if drift:
        raise ValueError(f"task evidence hash drift relative to prior report: {drift}")
    return {
        "checked": True,
        "path": str(path.relative_to(root)),
        "sha256": _sha256(path),
        "common_evidence_count": len(common),
    }


def build_assembly_evidence_report(
    *,
    repository_root: str | Path,
    work_queue_path: str | Path,
    generated_at_utc: str,
    video_manifest_path: str | Path | None = None,
    performance_summary_path: str | Path | None = None,
    prior_report_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    queue_path = _resolve_inside(root, work_queue_path, "work queue")
    queue = _load_mapping(queue_path, "work queue")
    if not isinstance(generated_at_utc, str) or not generated_at_utc.endswith("Z"):
        raise ValueError("generated_at_utc must be an explicit UTC timestamp")
    groups = queue.get("groups")
    if not isinstance(groups, Mapping):
        raise ValueError("work queue lacks groups")
    flattened: list[tuple[str, Mapping[str, Any]]] = []
    for group_key, group in groups.items():
        tasks = group.get("tasks") if isinstance(group, Mapping) else None
        if not isinstance(tasks, Mapping):
            raise ValueError(f"work queue group {group_key} lacks tasks")
        for task_key, task in tasks.items():
            if not isinstance(task, Mapping):
                raise ValueError(f"work queue task {task_key} must be a mapping")
            flattened.append((str(task_key), task))
    keys = tuple(key for key, _ in flattened)
    if keys != EXPECTED_TASK_KEYS:
        raise ValueError("work queue task inventory/order differs from F3 contract")

    rows = []
    for task_key, task in flattened:
        status = task.get("status")
        if status not in ALLOWED_STATUSES:
            raise ValueError(f"{task_key} has unsupported status {status!r}")
        evidence = _task_evidence(root, task_key, task)
        rows.append(
            {
                "task_key": task_key,
                "name": task.get("name"),
                "queue_status": status,
                "dynamic_dependency": task.get("dynamic_dependency"),
                "implementation_only": task.get("implementation_only", False),
                "evidence": evidence,
                "dynamic_pass_evidence_present": status == "DYNAMIC_PASS",
            }
        )

    counts = Counter(row["queue_status"] for row in rows)
    missing_expected = [
        row["task_key"]
        for row in rows
        if row["evidence"]["state"] == "MISSING_EXPECTED_EVIDENCE"
    ]
    f1_row = next(row for row in rows if row["task_key"] == "F1")
    dynamic_tasks = [
        row["task_key"] for row in rows if row["queue_status"] == "DYNAMIC_PASS"
    ]
    video = _video_inventory(root, video_manifest_path)
    performance = _performance_inventory(root, performance_summary_path)
    prior = _verify_prior_report(root, prior_report_path, rows)
    full_chain_dynamic = (
        f1_row["queue_status"] == "DYNAMIC_PASS"
        and f1_row["evidence"]["state"] == "VERIFIED_PRESENT"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "generated_at_utc": generated_at_utc,
        "report_tool_status": "OFFLINE_PASS",
        "queue_id": queue.get("queue_id"),
        "queue_status": queue.get("status"),
        "queue_current_task": queue.get("current_task"),
        "task_count": len(rows),
        "status_counts": {key: counts.get(key, 0) for key in sorted(ALLOWED_STATUSES)},
        "tasks": rows,
        "task_evidence_present_count": sum(
            row["evidence"]["state"] == "VERIFIED_PRESENT" for row in rows
        ),
        "missing_expected_task_evidence": missing_expected,
        "dynamic_task_count": len(dynamic_tasks),
        "dynamic_tasks": dynamic_tasks,
        "full_chain_dynamic_pass_evidence_present": full_chain_dynamic,
        "current_frontier_state": f1_row["evidence"].get("current_state"),
        "video": video,
        "performance": performance,
        "source_manifest": {
            "work_queue": {
                "path": str(queue_path.relative_to(root)),
                "sha256": _sha256(queue_path),
            },
            "prior_report": prior,
        },
        "simulation_started_by_reporter": False,
        "robot_commands_emitted_by_reporter": 0,
        "synthetic_dynamic_evidence_used": False,
        "dynamic_acceptance_authority": "not_this_offline_reporter",
        "assembly_success_claimed": False,
        "formal_r12_generated": False,
        "control_authorized": False,
        "hardware_authorized": False,
    }


def render_report_markdown(report: Mapping[str, Any]) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|")

    lines = [
        "# 完整装配证据、视频与统计报告",
        "",
        "| 任务 | 队列状态 | 证据状态 | 动态通过证据 | 证据路径 |",
        "|---|---|---|---|---|",
    ]
    for row in report["tasks"]:
        evidence = row["evidence"]
        lines.append(
            "| "
            + " | ".join(
                cell(value)
                for value in (
                    row["task_key"],
                    row["queue_status"],
                    evidence["state"],
                    row["dynamic_pass_evidence_present"],
                    evidence["path"] or "—",
                )
            )
            + " |"
        )
    performance = report["performance"]
    lines.extend(
        [
            "",
            "## 当前边界",
            "",
            f"- 当前完整任务状态：{report['current_frontier_state'] or '未知'}。",
            f"- 动态通过节点数：{report['dynamic_task_count']}。",
            f"- 可追溯视频数：{report['video']['count']}。",
            "- 实测性能："
            + (
                f"显存峰值 {performance['target_process_vram_peak_mib']} MiB，"
                f"渲染 {performance['render_fps']} FPS。"
                if performance["available"]
                else "未提供。"
            ),
            f"- 缺失的预期任务证据：{report['missing_expected_task_evidence']}。",
            "- 本报告器不启动仿真、不发机器人命令，也不授予动态通过或控制权限。",
            "",
        ]
    )
    return "\n".join(lines)


def write_report_pair(
    report: Mapping[str, Any],
    json_output: str | Path,
    markdown_output: str | Path,
) -> None:
    json_path = Path(json_output)
    markdown_path = Path(markdown_output)
    if json_path.exists() or markdown_path.exists():
        raise FileExistsError("F3 report outputs are immutable")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True,
                   allow_nan=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_report_markdown(report), encoding="utf-8")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--work-queue", required=True)
    parser.add_argument("--generated-at-utc", required=True)
    parser.add_argument("--video-manifest")
    parser.add_argument("--performance-summary")
    parser.add_argument("--prior-report")
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--markdown-output", required=True)
    args = parser.parse_args()
    if not args.run:
        parser.error("report generation requires --run")
    return args


def main() -> None:
    args = _arguments()
    root = Path(args.repository_root).resolve()
    output_root = (root / "artifacts/agent_control/tasks" / TASK_ID).resolve()
    json_output = _resolve_inside(root, args.json_output, "JSON output")
    markdown_output = _resolve_inside(root, args.markdown_output, "Markdown output")
    if not json_output.is_relative_to(output_root) or not markdown_output.is_relative_to(output_root):
        raise PermissionError("F3 outputs must remain inside the F3 task directory")
    report = build_assembly_evidence_report(
        repository_root=root,
        work_queue_path=args.work_queue,
        generated_at_utc=args.generated_at_utc,
        video_manifest_path=args.video_manifest,
        performance_summary_path=args.performance_summary,
        prior_report_path=args.prior_report,
    )
    write_report_pair(report, json_output, markdown_output)


if __name__ == "__main__":
    main()


__all__ = [
    "EXPECTED_TASK_KEYS",
    "SCHEMA_VERSION",
    "VIDEO_MANIFEST_SCHEMA",
    "build_assembly_evidence_report",
    "render_report_markdown",
    "write_report_pair",
]
