#!/usr/bin/env python3
"""Build the fixed TASK-R12-005 review ZIP with a SHA-256 manifest."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import zipfile


REQUIRED = [
    "REVIEW_REQUEST_CN.md",
    "MASTER_STATE.json",
    "TASK_RESULT.json",
    "GATE_LEDGER.csv",
    "DECISION_LOG.jsonl",
    "CURRENT_STATUS_CN.md",
]

BASELINE_RUNNER = (
    "artifacts/kcg_connector/handoff_to_gpt56pro/"
    "r12_direct_closeout_20260817T103626Z/SOURCE/src/kcg_connector/isaac/"
    "d38999_physical_r7_p1_nominal_bench.py"
)
ACTIVE_RUNNER = "src/kcg_connector/isaac/d38999_physical_r7_p1_nominal_bench.py"
TASK_CREATED_CODE = [
    "AGENTS.md",
    "PLANS.md",
    "tools/agent_control/run_guarded.py",
    "tools/agent_control/build_review_bundle.py",
    "tools/agent_control/analyze_r12_005.py",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_capture(repo: Path, args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
    )
    return f"$ git {' '.join(args)}\nexit={result.returncode}\n{result.stdout}"


def no_index_diff(repo: Path, before: str, after: str) -> str:
    result = subprocess.run(
        ["git", "diff", "--no-index", "--binary", "--", before, after],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode not in (0, 1):
        raise SystemExit(
            f"无法生成任务差异：{before} -> {after}\n{result.stdout}"
        )
    return (
        f"# git diff --no-index --binary -- {before} {after}\n"
        f"# exit={result.returncode}\n{result.stdout}"
    )


def task_code_diff(repo: Path) -> str:
    baseline = repo / BASELINE_RUNNER
    active = repo / ACTIVE_RUNNER
    if not baseline.is_file() or not active.is_file():
        raise SystemExit("缺少 runner 基线或当前文件，不能构建审查差异")
    sections = [no_index_diff(repo, BASELINE_RUNNER, ACTIVE_RUNNER)]
    for relative in TASK_CREATED_CODE:
        path = repo / relative
        if not path.is_file():
            raise SystemExit(f"缺少本轮应纳入差异的文件：{relative}")
        sections.append(no_index_diff(repo, "/dev/null", relative))
    return "\n".join(sections)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--outcome", required=True, choices=["PASS", "BLOCKED"])
    parser.add_argument("--evidence", action="append", type=Path, default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    control = repo / "artifacts/agent_control"
    missing = [name for name in REQUIRED if not (control / name).is_file()]
    if missing:
        raise SystemExit(f"缺少必需文件：{missing}")
    result = json.loads((control / "TASK_RESULT.json").read_text(encoding="utf-8"))
    if result.get("task_id") != "TASK-R12-005" or not result.get("final_classification"):
        raise SystemExit("TASK_RESULT.json 尚无有效最终分类")

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    review_dir = control / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    output = review_dir / f"TASK_R12_005_{args.outcome}_{stamp}.zip"
    with tempfile.TemporaryDirectory(prefix="task_r12_005_") as tmp_name:
        staging = Path(tmp_name)
        for name in REQUIRED:
            (staging / name).write_bytes((control / name).read_bytes())
        (staging / "TASK_CODE_DIFF.patch").write_text(
            task_code_diff(repo), encoding="utf-8"
        )
        (staging / "WORKTREE_STATUS.txt").write_text(
            git_capture(repo, ["status", "--short", "--branch"]), encoding="utf-8"
        )
        for path_arg in args.evidence:
            source = (repo / path_arg).resolve() if not path_arg.is_absolute() else path_arg.resolve()
            try:
                source.relative_to(repo)
            except ValueError as exc:
                raise SystemExit(f"证据必须位于仓库内：{source}") from exc
            if not source.is_file():
                raise SystemExit(f"证据文件不存在：{source}")
            target = staging / "evidence" / source.relative_to(repo)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())

        files = sorted(path for path in staging.rglob("*") if path.is_file())
        manifest_lines = [f"{sha256_file(path)}  {path.relative_to(staging)}" for path in files]
        (staging / "SHA256SUMS.txt").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(staging.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(staging))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
