"""Direct-path integrity monitor for stable eight-hour closeout sources.

Unlike the parked G1 recursive reference classifier, this monitor consumes the
already validated package-source list and excludes an explicit set of mutable
control files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .eight_hour_final_artifacts import FINAL_OUTPUTS, build_final_report_data
from .final_review_preflight import _load_mapping, _resolve_inside, _sha256


SCHEMA_VERSION = "kcg_eight_hour_stable_closeout_integrity_v1"
TASK_ID = "EIGHT-HOUR-G7-CLOSEOUT-INTEGRITY-MONITOR"
MUTABLE_CONTROL_PATHS = frozenset(
    {
        "artifacts/agent_control/WORK_QUEUE.yaml",
        "artifacts/agent_control/MASTER_STATE.json",
        "artifacts/agent_control/CURRENT_TASK.md",
        "artifacts/agent_control/CURRENT_STATUS_CN.md",
        "artifacts/agent_control/EIGHT_HOUR_PROGRESS_CN.md",
        "artifacts/agent_control/GATE_LEDGER.csv",
        "artifacts/agent_control/DECISION_LOG.jsonl",
        "artifacts/agent_control/STATUS_HISTORY.jsonl",
        "artifacts/agent_control/TASK_GRAPH.yaml",
    }
)
G7_SELF_PATHS = (
    "src/kcg_connector/kcg_connector/stable_closeout_integrity.py",
    "src/kcg_connector/test/test_stable_closeout_integrity.py",
    "artifacts/agent_control/tasks/EIGHT-HOUR-G7-CLOSEOUT-INTEGRITY-MONITOR/RUN_PLAN.json",
)
HIGH_DETAIL_MANIFEST = Path(
    "artifacts/agent_control/multilayer/HIGH_DETAIL_REFERENCE_MANIFEST.json"
)


def _canonical_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    encoded = json.dumps(
        list(rows), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _high_detail_assets(root: Path) -> list[tuple[str, str, str]]:
    manifest_path = root / HIGH_DETAIL_MANIFEST
    manifest = _load_mapping(manifest_path, "high-detail reference manifest")
    baseline = manifest.get("authoritative_high_detail_baseline")
    variant = manifest.get("rejected_local_variant_retained_as_evidence")
    if not isinstance(baseline, Mapping) or not isinstance(variant, Mapping):
        raise ValueError("high-detail manifest lacks frozen asset mappings")
    rows = [
        (str(baseline.get("path")), str(baseline.get("sha256")), "high_detail_baseline"),
        (str(variant.get("path")), str(variant.get("sha256")), "rejected_local_variant"),
        (
            str(variant.get("build_result_path")),
            str(variant.get("build_result_sha256")),
            "rejected_variant_build_result",
        ),
    ]
    for relative, expected, label in rows:
        path = _resolve_inside(root, relative, label)
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"frozen high-detail asset drift: {label}")
    return rows


def build_stable_manifest(
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
    data = build_final_report_data(
        repository_root=root,
        work_queue_path=work_queue_path,
        master_state_path=master_state_path,
        blocker_ledger_path=blocker_ledger_path,
        readiness_report_path=readiness_report_path,
        preflight_path=preflight_path,
        closeout_at_utc=generated_at_utc,
    )
    direct_paths = {
        relative
        for relative in data["package_source_paths"]
        if relative not in MUTABLE_CONTROL_PATHS
    }
    direct_paths.update(G7_SELF_PATHS)
    frozen_assets = _high_detail_assets(root)
    direct_paths.update(relative for relative, _digest, _label in frozen_assets)
    rows = []
    for relative in sorted(direct_paths):
        path = _resolve_inside(root, relative, "stable source")
        if not path.is_file():
            raise ValueError(f"stable source missing: {relative}")
        rows.append(
            {
                "path": relative,
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    if len(rows) != len({row["path"] for row in rows}):
        raise ValueError("stable source path duplication")
    frozen_by_path = {relative: digest for relative, digest, _label in frozen_assets}
    for row in rows:
        expected = frozen_by_path.get(row["path"])
        if expected is not None and row["sha256"] != expected:
            raise ValueError("frozen high-detail digest mismatch in stable rows")
    formal_outputs = [
        str(relative) for relative in FINAL_OUTPUTS if (root / relative).exists()
    ]
    if formal_outputs:
        raise ValueError(f"formal outputs appeared before deadline: {formal_outputs}")
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "generated_at_utc": generated_at_utc,
        "result": "BASELINE_CREATED",
        "stable_source_count": len(rows),
        "stable_sources": rows,
        "stable_manifest_digest": _canonical_digest(rows),
        "mutable_control_path_count": len(MUTABLE_CONTROL_PATHS),
        "mutable_control_paths_excluded": sorted(MUTABLE_CONTROL_PATHS),
        "frozen_high_detail_asset_count": len(frozen_assets),
        "frozen_high_detail_assets": [
            {"path": relative, "sha256": digest, "role": label}
            for relative, digest, label in frozen_assets
        ],
        "formal_final_output_count": 0,
        "g1_recursive_reference_classifier_used": False,
        "simulation_started": False,
        "robot_commands_emitted": 0,
        "assembly_success_claimed": False,
        "formal_r12_generated": False,
        "control_authorized": False,
        "hardware_authorized": False,
    }


def check_stable_manifest(
    *, repository_root: str | Path, baseline: Mapping[str, Any], checked_at_utc: str
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    if baseline.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("stable baseline schema mismatch")
    rows = baseline.get("stable_sources")
    if not isinstance(rows, list):
        raise ValueError("stable baseline lacks source rows")
    current_rows = []
    missing = []
    drift = []
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
            raise ValueError("stable baseline row invalid")
        path = _resolve_inside(root, row["path"], "stable source check")
        if not path.is_file():
            missing.append(row["path"])
            continue
        current = {
            "path": row["path"],
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        current_rows.append(current)
        if current["sha256"] != row.get("sha256") or current["size_bytes"] != row.get("size_bytes"):
            drift.append(row["path"])
    result = "PASS" if not missing and not drift else "FAIL"
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "checked_at_utc": checked_at_utc,
        "result": result,
        "baseline_source_count": len(rows),
        "checked_source_count": len(current_rows),
        "missing_source_count": len(missing),
        "missing_sources": missing,
        "drift_source_count": len(drift),
        "drift_sources": drift,
        "current_manifest_digest": _canonical_digest(current_rows),
        "baseline_manifest_digest": baseline.get("stable_manifest_digest"),
        "formal_final_output_count": sum(
            (root / relative).exists() for relative in FINAL_OUTPUTS
        ),
        "simulation_started": False,
        "robot_commands_emitted": 0,
    }


def write_new_json(value: Mapping[str, Any], output: str | Path) -> None:
    path = Path(output)
    if path.exists():
        raise FileExistsError("stable-integrity output is immutable")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                   allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--create-baseline", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--work-queue")
    parser.add_argument("--master-state")
    parser.add_argument("--blocker-ledger")
    parser.add_argument("--readiness-report")
    parser.add_argument("--preflight")
    parser.add_argument("--generated-at-utc", required=True)
    parser.add_argument("--baseline")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    root = Path(args.repository_root).resolve()
    output = _resolve_inside(root, args.output, "integrity output")
    output_root = (root / "artifacts/agent_control/tasks" / TASK_ID).resolve()
    if not output.is_relative_to(output_root):
        raise PermissionError("G7 outputs must remain inside the G7 task directory")
    if args.create_baseline:
        required = (
            args.work_queue, args.master_state, args.blocker_ledger,
            args.readiness_report, args.preflight,
        )
        if any(value is None for value in required):
            raise ValueError("baseline mode requires all source arguments")
        value = build_stable_manifest(
            repository_root=root,
            work_queue_path=args.work_queue,
            master_state_path=args.master_state,
            blocker_ledger_path=args.blocker_ledger,
            readiness_report_path=args.readiness_report,
            preflight_path=args.preflight,
            generated_at_utc=args.generated_at_utc,
        )
    else:
        if args.baseline is None:
            raise ValueError("check mode requires --baseline")
        baseline_path = _resolve_inside(root, args.baseline, "stable baseline")
        baseline = _load_mapping(baseline_path, "stable baseline")
        value = check_stable_manifest(
            repository_root=root,
            baseline=baseline,
            checked_at_utc=args.generated_at_utc,
        )
        if value["result"] != "PASS" or value["formal_final_output_count"] != 0:
            raise SystemExit("stable source integrity check failed")
    write_new_json(value, output)


if __name__ == "__main__":
    main()


__all__ = [
    "G7_SELF_PATHS",
    "MUTABLE_CONTROL_PATHS",
    "SCHEMA_VERSION",
    "build_stable_manifest",
    "check_stable_manifest",
    "write_new_json",
]
