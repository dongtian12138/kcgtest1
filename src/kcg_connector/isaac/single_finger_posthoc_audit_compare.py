#!/usr/bin/env python3

'''Offline capture/skip posthoc audit comparator (pure Python, no Isaac).

Reads two episode outputs (capture + skip) and writes ONE independent
summary JSON.  It never rewrites the episode reports and never launches
Isaac.  Frozen exit semantics: 0 validation passed, 1 contact gate failed,
2 input/provenance contract mismatch, 3 inconclusive.
'''

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kcg_connector.grasp.single_finger_posthoc_audit import compare_episodes


def _load_steps(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="capture/skip single-finger posthoc audit comparator"
    )
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--skip-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    arguments = parser.parse_args()
    capture_dir = Path(arguments.capture_dir).expanduser().resolve()
    skip_dir = Path(arguments.skip_dir).expanduser().resolve()
    output_dir = Path(arguments.output_dir).expanduser().resolve()
    capture_report_path = capture_dir / "nominal_physics_report.json"
    skip_report_path = skip_dir / "nominal_physics_report.json"
    capture_steps_path = capture_dir / "controller_steps.jsonl"
    skip_steps_path = skip_dir / "controller_steps.jsonl"
    if not all(
        path.is_file()
        for path in (
            capture_report_path,
            skip_report_path,
            capture_steps_path,
            skip_steps_path,
        )
    ):
        print(
            "input contract mismatch: missing report/steps files",
            flush=True,
        )
        return 2
    capture_report = json.loads(
        capture_report_path.read_text(encoding="utf-8")
    )
    skip_report = json.loads(skip_report_path.read_text(encoding="utf-8"))
    try:
        summary = compare_episodes(
            capture_report,
            _load_steps(capture_steps_path),
            skip_report,
            _load_steps(skip_steps_path),
        )
    except ValueError as error:
        print(f"input contract mismatch: {error}", flush=True)
        return 2
    capture_resolved = capture_dir.resolve()
    skip_resolved = skip_dir.resolve()
    output_resolved = output_dir.resolve()
    for episode_dir in (capture_resolved, skip_resolved):
        if output_resolved == episode_dir or episode_dir in output_resolved.parents:
            print(
                "refusing to write the comparison inside an episode "
                "directory",
                flush=True,
            )
            return 2
    if output_resolved.exists() and any(output_resolved.iterdir()):
        print(
            "refusing to overwrite a non-empty output directory",
            flush=True,
        )
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "posthoc_audit_comparison.json").write_text(
        json.dumps(
            summary,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "exit_code": summary["exit_code"],
                "single_finger_validation_passed": summary[
                    "single_finger_validation_passed"
                ],
                "failure_reason": summary["failure_reason"],
            },
            indent=2,
        ),
        flush=True,
    )
    return int(summary["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
