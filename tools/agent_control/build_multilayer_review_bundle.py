#!/usr/bin/env python3

"""Build and verify the fixed TASK-R12-MULTILAYER-004 blocked review bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import zipfile


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_RELATIVE = Path(
    "artifacts/agent_control/review/"
    "TASK_R12-MULTILAYER-004_BLOCKED_20260817T171653Z.zip"
)
TASK_DIR = Path("artifacts/agent_control/tasks/TASK-R12-MULTILAYER-004")

ROOT_FILES = (
    "artifacts/agent_control/PROJECT_CHARTER_CN.md",
    "artifacts/agent_control/TASK_GRAPH.yaml",
    "artifacts/agent_control/MASTER_STATE.json",
    "artifacts/agent_control/CURRENT_TASK.md",
    "artifacts/agent_control/CURRENT_STATUS_CN.md",
    "artifacts/agent_control/STATUS_HISTORY.jsonl",
    "artifacts/agent_control/DECISION_LOG.jsonl",
    "artifacts/agent_control/GATE_LEDGER.csv",
    "AGENTS.md",
    "PLANS.md",
)

TASK_ROOT_FILES = {
    f"{TASK_DIR}/REVIEW_REQUEST_CN.md": "REVIEW_REQUEST_CN.md",
    f"{TASK_DIR}/TASK_RESULT.json": "TASK_RESULT.json",
    f"{TASK_DIR}/ACTUAL_COMMANDS.txt": "ACTUAL_COMMANDS.txt",
    f"{TASK_DIR}/STATIC_ATTEMPT_LOG_CN.md": "STATIC_ATTEMPT_LOG_CN.md",
    f"{TASK_DIR}/VALIDATION_PLAN_CN.md": "VALIDATION_PLAN_CN.md",
}

EVIDENCE_FILES = (
    "artifacts/agent_control/tasks/TASK-R12-MULTILAYER-001/VALIDATION.json",
    "artifacts/agent_control/multilayer/HIGH_DETAIL_REFERENCE_MANIFEST.json",
    "artifacts/agent_control/multilayer/HIGH_DETAIL_BLOCKED_CONCLUSION_CN.md",
    "artifacts/agent_control/tasks/TASK-R12-MULTILAYER-002/VALIDATION.json",
    "artifacts/agent_control/tasks/TASK-R12-MULTILAYER-002/SOURCE_EXTRACTION.json",
    "artifacts/agent_control/tasks/TASK-R12-MULTILAYER-003/GENERATION_PLAN_CN.md",
    "artifacts/agent_control/tasks/TASK-R12-MULTILAYER-003/BUILD_RESULT.json",
    "artifacts/agent_control/tasks/TASK-R12-MULTILAYER-003/VALIDATION.json",
    "src/kcg_connector/config/d38999_master_model_contract_v1.yaml",
    "src/kcg_connector/isaac/build_d38999_multilayer_models.py",
    "tools/agent_control/validate_multilayer_model.py",
    "tools/agent_control/build_multilayer_review_bundle.py",
    "artifacts/kcg_connector/isaac/d38999_multilayer_v1/D38999_VISUAL_COMPLETE_V1.usda",
    "artifacts/kcg_connector/isaac/d38999_multilayer_v1/D38999_ASSEMBLY_CONTROL_V1.usda",
    "artifacts/kcg_connector/isaac/d38999_multilayer_v1/D38999_LOCAL_CONTACT_REFERENCE_V1.usda",
    "artifacts/kcg_connector/isaac/d38999_multilayer_v1/MODEL_MAPPING.json",
)

CODE_PATHS = (
    "AGENTS.md",
    "PLANS.md",
    "src/kcg_connector/config/d38999_master_model_contract_v1.yaml",
    "src/kcg_connector/isaac/build_d38999_multilayer_models.py",
    "tools/agent_control/validate_multilayer_model.py",
    "tools/agent_control/build_multilayer_review_bundle.py",
)

REQUIRED_ARCHIVE_MEMBERS = (
    "REVIEW_REQUEST_CN.md",
    "TASK_RESULT.json",
    "ACTUAL_COMMANDS.txt",
    "STATIC_ATTEMPT_LOG_CN.md",
    "VALIDATION_PLAN_CN.md",
    "state/artifacts/agent_control/MASTER_STATE.json",
    "state/artifacts/agent_control/TASK_GRAPH.yaml",
    "state/artifacts/agent_control/GATE_LEDGER.csv",
    "state/artifacts/agent_control/DECISION_LOG.jsonl",
    "state/artifacts/agent_control/CURRENT_STATUS_CN.md",
    "TASK_CODE_DIFF.patch",
    "WORKTREE_STATUS.txt",
    "FILE_SUMMARY.json",
    "BUNDLE_METADATA.json",
    "SHA256SUMS.txt",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_git(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=WORKSPACE_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def _path_is_tracked(relative: str) -> bool:
    result = _run_git(["ls-files", "--error-unmatch", "--", relative])
    return result.returncode == 0


def _code_diff() -> str:
    sections: list[str] = []
    for relative in CODE_PATHS:
        source = WORKSPACE_ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        if _path_is_tracked(relative):
            result = _run_git(["diff", "--no-ext-diff", "--binary", "--", relative])
            if result.returncode != 0:
                raise RuntimeError(f"git diff failed for {relative}: {result.stdout}")
        else:
            result = _run_git(
                ["diff", "--no-index", "--no-ext-diff", "--binary", "--", "/dev/null", relative]
            )
            if result.returncode not in (0, 1):
                raise RuntimeError(f"git diff --no-index failed for {relative}: {result.stdout}")
        sections.append(
            f"# path={relative}\n# tracked={str(_path_is_tracked(relative)).lower()}\n"
            f"{result.stdout.rstrip()}\n"
        )
    return "\n".join(sections)


def _copy(source_relative: str, destination: Path) -> None:
    source = WORKSPACE_ROOT / source_relative
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source.read_bytes())


def _summary(staging: Path) -> dict[str, object]:
    files = sorted(path for path in staging.rglob("*") if path.is_file())
    return {
        "schema_version": "kcg_review_file_summary_v1",
        "task_id": "TASK-R12-MULTILAYER-004",
        "outcome": "BLOCKED",
        "file_count_before_summary_and_hash_manifest": len(files),
        "files": [
            {
                "path": path.relative_to(staging).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in files
        ],
        "static_consistency_result_present": False,
        "simulation_started": False,
        "formal_p1_pass_claimed": False,
        "formal_r12_generated": False,
        "hardware_authorized": False,
    }


def main() -> int:
    output = WORKSPACE_ROOT / OUTPUT_RELATIVE
    if output.exists():
        raise FileExistsError(f"refusing to overwrite review bundle: {output}")
    result = json.loads((WORKSPACE_ROOT / TASK_DIR / "TASK_RESULT.json").read_text())
    state = json.loads(
        (WORKSPACE_ROOT / "artifacts/agent_control/MASTER_STATE.json").read_text()
    )
    if result.get("task_id") != "TASK-R12-MULTILAYER-004":
        raise ValueError("task result identity differs")
    if result.get("outcome") != "BLOCKED":
        raise ValueError("review builder is authorized only for the blocked result")
    if state.get("task_id") != "TASK-R12-MULTILAYER-004" or state.get("status") != "BLOCKED":
        raise ValueError("master state is not the matching blocked task")
    static_result = WORKSPACE_ROOT / TASK_DIR / "STATIC_CONSISTENCY.json"
    if static_result.exists():
        raise ValueError("unexpected static result exists; blocked premise changed")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="task_r12_multilayer_004_review_") as temporary:
        staging = Path(temporary)
        for source_relative, archive_name in TASK_ROOT_FILES.items():
            _copy(source_relative, staging / archive_name)
        for source_relative in ROOT_FILES:
            _copy(source_relative, staging / "state" / source_relative)
        for source_relative in EVIDENCE_FILES:
            _copy(source_relative, staging / "evidence" / source_relative)

        (staging / "TASK_CODE_DIFF.patch").write_text(
            _code_diff(), encoding="utf-8"
        )
        worktree = _run_git(["status", "--short", "--branch"])
        (staging / "WORKTREE_STATUS.txt").write_text(
            f"$ git status --short --branch\nexit={worktree.returncode}\n{worktree.stdout}",
            encoding="utf-8",
        )
        metadata = {
            "schema_version": "kcg_multilayer_blocked_review_bundle_v1",
            "task_id": "TASK-R12-MULTILAYER-004",
            "outcome": "BLOCKED",
            "classification": result["classification"],
            "model_static_consistency_conclusion": "NOT_EVALUATED",
            "bundle_path": OUTPUT_RELATIVE.as_posix(),
            "static_acceptance_attempts": 2,
            "targeted_fix_count": 1,
            "targeted_fix_limit": 1,
            "static_consistency_result_present": False,
            "related_process_count_at_closeout": 0,
            "simulation_started": False,
            "formal_p1_pass_claimed": False,
            "formal_r12_generated": False,
            "hardware_authorized": False,
        }
        (staging / "BUNDLE_METADATA.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (staging / "FILE_SUMMARY.json").write_text(
            json.dumps(_summary(staging), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        files_before_hashes = sorted(
            path for path in staging.rglob("*") if path.is_file()
        )
        (staging / "SHA256SUMS.txt").write_text(
            "\n".join(
                f"{_sha256(path)}  {path.relative_to(staging).as_posix()}"
                for path in files_before_hashes
            )
            + "\n",
            encoding="utf-8",
        )
        all_files = sorted(path for path in staging.rglob("*") if path.is_file())
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in all_files:
                archive.write(path, path.relative_to(staging).as_posix())

    with zipfile.ZipFile(output, "r") as archive:
        names = set(archive.namelist())
        missing = sorted(set(REQUIRED_ARCHIVE_MEMBERS) - names)
        bad_member = archive.testzip()
        if missing or bad_member is not None:
            raise RuntimeError(
                f"review bundle verification failed: missing={missing}, bad={bad_member}"
            )
    print(
        json.dumps(
            {
                "output": OUTPUT_RELATIVE.as_posix(),
                "sha256": _sha256(output),
                "size_bytes": output.stat().st_size,
                "member_count": len(names),
                "required_members_missing": [],
                "zip_crc_error": None,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
