"""Evidence-bound offline report for the eight-hour grasp work group.

This module indexes existing B1-B5 task evidence.  It cannot grant a
dynamic grasp pass: dynamic acceptance remains the responsibility of a
future independent-process runtime acceptance artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


SCHEMA_VERSION = "kcg_eight_hour_grasp_result_report_v1"
TASK_KEYS = ("B1", "B2", "B3", "B4", "B5")
ALLOWED_STATUSES = {
    "NOT_STARTED",
    "IMPLEMENTING",
    "STATIC_PASS",
    "OFFLINE_PASS",
    "DYNAMIC_PASS",
    "PARKED",
    "BLOCKED_EXTERNAL",
}
EVIDENCE_REQUIRED_STATUSES = {
    "STATIC_PASS",
    "OFFLINE_PASS",
    "PARKED",
    "BLOCKED_EXTERNAL",
}
DYNAMIC_MEASUREMENT_FIELDS = (
    "contact_order",
    "first_two_fingers",
    "third_finger",
    "lift_stage_positions_m",
    "actual_lift_m",
    "peak_force_n",
    "peak_moment_nm",
    "fixed_receptacle_drift_m",
    "post_run_pose_write_count",
)


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


def _resolve_evidence(repository_root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("evidence path must be a non-empty repository path")
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError("absolute evidence paths are forbidden")
    path = (repository_root / relative).resolve()
    if not path.is_relative_to(repository_root):
        raise ValueError("evidence path escapes repository root")
    return path


def _task_evidence(
    repository_root: Path,
    task_key: str,
    task: Mapping[str, Any],
) -> dict[str, Any] | None:
    status = task.get("status")
    if status == "DYNAMIC_PASS":
        raise ValueError(
            f"{task_key} DYNAMIC_PASS cannot be granted or preserved by "
            "the offline B6 reporter"
        )
    evidence_value = task.get("evidence")
    if status in EVIDENCE_REQUIRED_STATUSES and evidence_value is None:
        raise ValueError(f"{task_key} {status} requires evidence")
    if evidence_value is None:
        return None
    path = _resolve_evidence(repository_root, evidence_value)
    document = _load_mapping(path, f"{task_key} evidence")
    if document.get("status") != status:
        raise ValueError(
            f"{task_key} queue/evidence status mismatch: "
            f"{status!r} != {document.get('status')!r}"
        )
    for claim in (
        "dynamic_grasp_pass_claimed",
        "formal_grasp_pass_claimed",
        "formal_physics_pass_claimed",
    ):
        if document.get(claim) is True:
            raise ValueError(
                f"{task_key} non-dynamic evidence contains true {claim}"
            )
    return {
        "path": str(path.relative_to(repository_root)),
        "sha256": _sha256(path),
        "status": document.get("status"),
        "classification": document.get("classification"),
        "simulation_started": document.get("simulation_started"),
        "dynamic_grasp_pass_claimed": document.get(
            "dynamic_grasp_pass_claimed", False
        ),
        "formal_physics_pass_claimed": document.get(
            "formal_physics_pass_claimed", False
        ),
    }


def _aggregate_status(tasks: Mapping[str, Mapping[str, Any]]) -> str:
    statuses = [tasks[key]["status"] for key in TASK_KEYS]
    if "BLOCKED_EXTERNAL" in statuses:
        return "BLOCKED_EXTERNAL"
    if "PARKED" in statuses:
        return "PARKED"
    if "IMPLEMENTING" in statuses or "NOT_STARTED" in statuses:
        return "NOT_STARTED"
    return "OFFLINE_PASS"


def build_grasp_result_report(
    *,
    repository_root: str | Path,
    work_queue_path: str | Path,
    generated_at_utc: str,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    queue_path = Path(work_queue_path)
    if not queue_path.is_absolute():
        queue_path = root / queue_path
    queue_path = queue_path.resolve()
    if not queue_path.is_relative_to(root):
        raise ValueError("work queue must be inside repository root")
    queue = _load_mapping(queue_path, "work queue")
    try:
        raw_tasks = queue["groups"]["B"]["tasks"]
    except (KeyError, TypeError):
        raise ValueError("work queue lacks groups.B.tasks") from None
    if not isinstance(raw_tasks, Mapping):
        raise ValueError("groups.B.tasks must be a mapping")

    tasks: dict[str, dict[str, Any]] = {}
    evidence_manifest: list[dict[str, Any]] = []
    for task_key in TASK_KEYS:
        raw = raw_tasks.get(task_key)
        if not isinstance(raw, Mapping):
            raise ValueError(f"work queue lacks {task_key}")
        status = raw.get("status")
        if status not in ALLOWED_STATUSES:
            raise ValueError(f"{task_key} has unsupported status {status!r}")
        evidence = _task_evidence(root, task_key, raw)
        if evidence is not None:
            evidence_manifest.append({"task": task_key, **evidence})
        tasks[task_key] = {
            "name": raw.get("name"),
            "status": status,
            "classification": raw.get("classification"),
            "dynamic_dependency": raw.get("dynamic_dependency"),
            "evidence": evidence,
        }

    chain_status = _aggregate_status(tasks)
    measurements = {
        field: None for field in DYNAMIC_MEASUREMENT_FIELDS
    }
    b5_evidence = tasks["B5"]["evidence"] or {}
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": generated_at_utc,
        "report_tool_status": "OFFLINE_PASS",
        "grasp_chain_status": chain_status,
        "tasks": tasks,
        "source_manifest": {
            "work_queue": {
                "path": str(queue_path.relative_to(root)),
                "sha256": _sha256(queue_path),
            },
            "task_evidence": evidence_manifest,
        },
        "safety_monitor": {
            "task_status": tasks["B5"]["status"],
            "evidence_path": b5_evidence.get("path"),
            "dynamic_integration_observed": False,
        },
        "dynamic_measurements_available": False,
        "dynamic_measurements": measurements,
        "dynamic_acceptance_authority": "not_this_offline_reporter",
        "dynamic_grasp_pass_claimed": False,
        "formal_grasp_pass_claimed": False,
        "assembly_success_claimed": False,
        "simulation_started_by_reporter": False,
        "synthetic_measurements_used": False,
        "current_user_action": "无需操作",
    }


def write_grasp_result_report(
    report: Mapping[str, Any], output_path: str | Path
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True,
                   allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--work-queue", required=True)
    parser.add_argument("--generated-at-utc", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not args.run:
        parser.error("report generation requires --run")
    return args


def main() -> None:
    args = _arguments()
    report = build_grasp_result_report(
        repository_root=args.repository_root,
        work_queue_path=args.work_queue,
        generated_at_utc=args.generated_at_utc,
    )
    write_grasp_result_report(report, args.output)


if __name__ == "__main__":
    main()
