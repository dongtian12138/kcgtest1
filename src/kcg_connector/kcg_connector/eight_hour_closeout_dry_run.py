"""Pre-deadline dry run for the eight-hour closeout artifacts.

Drafts are restricted to the G5 task directory.  The complete ZIP exists only
inside a temporary directory and is deleted after SHA-256 verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence
import zipfile

from .eight_hour_final_artifacts import (
    FINAL_OUTPUTS,
    build_code_snapshot_patch,
    build_final_report_data,
    collect_actual_commands,
    render_final_report,
    verify_review_zip,
)
from .final_review_preflight import _resolve_inside, _sha256


SCHEMA_VERSION = "kcg_eight_hour_closeout_dry_run_v1"
TASK_ID = "EIGHT-HOUR-G5-CLOSEOUT-DRY-RUN"
DRAFT_NAMES = (
    "DRAFT_FINAL_DATA.json",
    "DRAFT_FINAL_REPORT_CN.md",
    "DRAFT_ACTUAL_COMMANDS.txt",
    "DRAFT_CODE_SNAPSHOT.patch",
    "DRY_RUN_FILE_MANIFEST.json",
    "DRY_RUN_VALIDATION.json",
)


def _write_new(path: Path, content: str) -> None:
    if path.exists():
        raise FileExistsError(f"G5 draft output is immutable: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_temporary_dry_run_zip(
    *,
    root: Path,
    source_paths: Sequence[str],
    draft_paths: Sequence[Path],
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="eight_hour_g5_") as temp_name:
        temporary_root = Path(temp_name)
        staging = temporary_root / "staging"
        staging.mkdir()
        members: dict[str, Path] = {}
        for relative in source_paths:
            source = _resolve_inside(root, relative, "G5 source")
            members[relative] = source
        for source in draft_paths:
            resolved = source.resolve()
            if not resolved.is_relative_to(root) or not resolved.is_file():
                raise ValueError("G5 draft member missing or outside repository")
            members[str(resolved.relative_to(root))] = resolved
        for relative, source in sorted(members.items()):
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        staged = sorted(path for path in staging.rglob("*") if path.is_file())
        sums = "\n".join(
            f"{_sha256(path)}  {path.relative_to(staging)}" for path in staged
        ) + "\n"
        (staging / "SHA256SUMS.txt").write_text(sums, encoding="utf-8")
        bundle = temporary_root / "dry_run_bundle.zip"
        with zipfile.ZipFile(bundle, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(item for item in staging.rglob("*") if item.is_file()):
                archive.write(path, path.relative_to(staging))
        verification = verify_review_zip(bundle)
        receipt = {
            "temporary_bundle_sha256": _sha256(bundle),
            "temporary_bundle_size_bytes": bundle.stat().st_size,
            **verification,
        }
    receipt["temporary_bundle_removed"] = not bundle.exists()
    if receipt["temporary_bundle_removed"] is not True:
        raise RuntimeError("G5 temporary bundle cleanup failed")
    return receipt


def run_closeout_dry_run(
    *,
    repository_root: str | Path,
    work_queue_path: str | Path,
    master_state_path: str | Path,
    blocker_ledger_path: str | Path,
    readiness_report_path: str | Path,
    preflight_path: str | Path,
    generated_at_utc: str,
    output_directory: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    output = _resolve_inside(root, output_directory, "G5 output directory")
    required_root = (root / "artifacts/agent_control/tasks" / TASK_ID).resolve()
    if output != required_root:
        raise PermissionError("G5 outputs must use the exact G5 task directory")
    if any((root / relative).exists() for relative in FINAL_OUTPUTS):
        raise PermissionError("formal final output appeared before G5 dry run")
    planned_paths = [output / name for name in DRAFT_NAMES]
    if any(path.exists() for path in planned_paths):
        raise FileExistsError("G5 dry-run outputs are immutable")
    data = build_final_report_data(
        repository_root=root,
        work_queue_path=work_queue_path,
        master_state_path=master_state_path,
        blocker_ledger_path=blocker_ledger_path,
        readiness_report_path=readiness_report_path,
        preflight_path=preflight_path,
        closeout_at_utc=generated_at_utc,
    )
    data_path, report_path, commands_path, patch_path, manifest_path, validation_path = planned_paths
    _write_new(
        data_path,
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True,
                   allow_nan=False) + "\n",
    )
    _write_new(report_path, render_final_report(data))
    _write_new(commands_path, collect_actual_commands(root, data["package_source_paths"]))
    _write_new(
        patch_path,
        build_code_snapshot_patch(
            root, list(data["code_modules"]) + list(data["test_files"])
        ),
    )
    manifest_rows = [
        {
            "path": relative,
            "sha256": _sha256(root / relative),
            "size_bytes": (root / relative).stat().st_size,
        }
        for relative in data["package_source_paths"]
    ]
    draft_rows = [
        {
            "path": str(path.relative_to(root)),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in (data_path, report_path, commands_path, patch_path)
    ]
    manifest = {
        "schema_version": 1,
        "generated_at_utc": generated_at_utc,
        "source_file_count": len(manifest_rows),
        "draft_file_count_before_manifest": len(draft_rows),
        "source_files": manifest_rows,
        "draft_files": draft_rows,
    }
    _write_new(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True,
                   allow_nan=False) + "\n",
    )
    zip_receipt = build_temporary_dry_run_zip(
        root=root,
        source_paths=data["package_source_paths"],
        draft_paths=[data_path, report_path, commands_path, patch_path, manifest_path],
    )
    metrics = data["metrics"]
    validation = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "generated_at_utc": generated_at_utc,
        "result": "OFFLINE_PASS",
        "draft_report_sha256": _sha256(report_path),
        "draft_data_sha256": _sha256(data_path),
        "draft_commands_sha256": _sha256(commands_path),
        "draft_code_snapshot_sha256": _sha256(patch_path),
        "dry_run_manifest_sha256": _sha256(manifest_path),
        "source_file_count": len(manifest_rows),
        "draft_file_count": 5,
        "task_table_row_count": len(data["task_rows"]),
        "summary_field_count": 14,
        "dynamic_passed_task_count": metrics["dynamic_passed_task_count"],
        "current_frontier_state": metrics["current_frontier_state"],
        "peak_vram_mib": metrics["peak_vram_mib"],
        "physics_fps": metrics["physics_fps"],
        "render_fps": metrics["render_fps"],
        "formal_final_output_count": sum(
            (root / relative).exists() for relative in FINAL_OUTPUTS
        ),
        "temporary_zip": zip_receipt,
        "simulation_started": False,
        "robot_commands_emitted": 0,
        "assembly_success_claimed": False,
        "formal_r12_generated": False,
        "control_authorized": False,
        "hardware_authorized": False,
    }
    if (
        validation["dynamic_passed_task_count"] != 0
        or validation["current_frontier_state"] != "HOME"
        or validation["formal_final_output_count"] != 0
        or validation["temporary_zip"]["temporary_bundle_removed"] is not True
    ):
        raise ValueError("G5 dry-run boundary verification failed")
    _write_new(
        validation_path,
        json.dumps(validation, ensure_ascii=False, indent=2, sort_keys=True,
                   allow_nan=False) + "\n",
    )
    return validation


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--work-queue", required=True)
    parser.add_argument("--master-state", required=True)
    parser.add_argument("--blocker-ledger", required=True)
    parser.add_argument("--readiness-report", required=True)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--generated-at-utc", required=True)
    parser.add_argument("--output-directory", required=True)
    args = parser.parse_args()
    if not args.run:
        parser.error("G5 dry run requires --run")
    return args


def main() -> None:
    args = _arguments()
    result = run_closeout_dry_run(
        repository_root=args.repository_root,
        work_queue_path=args.work_queue,
        master_state_path=args.master_state,
        blocker_ledger_path=args.blocker_ledger,
        readiness_report_path=args.readiness_report,
        preflight_path=args.preflight,
        generated_at_utc=args.generated_at_utc,
        output_directory=args.output_directory,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "DRAFT_NAMES",
    "SCHEMA_VERSION",
    "TASK_ID",
    "build_temporary_dry_run_zip",
    "run_closeout_dry_run",
]
