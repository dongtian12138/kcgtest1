#!/usr/bin/env python3

'''Offline GUI/headless single-finger consistency comparator (pure, no Isaac).

Reads one headless capture episode and one GUI capture episode and writes ONE
independent summary JSON.  It never rewrites episode reports and never
launches Isaac.  Frozen exit semantics:
  0 = functional/structural GUI-headless consistency passed;
      quantitative equivalence explicitly not claimed
  1 = valid evidence but physical/functional structure mismatch
  2 = input/schema/provenance contract invalid
  3 = evidence incomplete/inconclusive, or the caller requests an
      unregistered quantitative equivalence claim
'''

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path

from kcg_connector.grasp.single_finger_gui_consistency import (
    compare_gui_headless_consistency,
)


def _load_steps(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_safe(value):
    """Replace every non-finite float before fail-closed JSON output."""

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description=
            "headless/GUI single-finger functional/structural "
            "consistency comparator"
    )
    parser.add_argument("--headless-dir", required=True)
    parser.add_argument("--gui-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--request-quantitative-equivalence",
        action="store_true",
        help=(
            "request a quantitative equivalence claim; rejected with "
            "exit 3 until quantitative gates are registered"
        ),
    )
    arguments = parser.parse_args()
    headless_dir = Path(arguments.headless_dir).expanduser().resolve()
    gui_dir = Path(arguments.gui_dir).expanduser().resolve()
    output_dir = Path(arguments.output_dir).expanduser().resolve()
    headless_report_path = headless_dir / "nominal_physics_report.json"
    gui_report_path = gui_dir / "nominal_physics_report.json"
    headless_steps_path = headless_dir / "controller_steps.jsonl"
    gui_steps_path = gui_dir / "controller_steps.jsonl"
    if not all(
        path.is_file()
        for path in (
            headless_report_path,
            gui_report_path,
            headless_steps_path,
            gui_steps_path,
        )
    ):
        print(
            "input contract mismatch: missing report/steps files",
            flush=True,
        )
        return 2
    for episode_dir in (headless_dir, gui_dir):
        if (
            output_dir == episode_dir
            or episode_dir in output_dir.parents
        ):
            print(
                "refusing to write the comparison inside an episode "
                "directory",
                flush=True,
            )
            return 2
    if output_dir.exists() and any(output_dir.iterdir()):
        print(
            "refusing to overwrite a non-empty output directory",
            flush=True,
        )
        return 2
    try:
        headless_report = json.loads(
            headless_report_path.read_text(encoding="utf-8")
        )
        gui_report = json.loads(
            gui_report_path.read_text(encoding="utf-8")
        )
    except (ValueError, OSError) as error:
        print(f"input contract mismatch: {error}", flush=True)
        return 2
    if not isinstance(headless_report, dict) or not isinstance(
        gui_report, dict
    ):
        print(
            "input contract mismatch: reports must be JSON objects",
            flush=True,
        )
        return 2
    try:
        headless_steps = _load_steps(headless_steps_path)
        gui_steps = _load_steps(gui_steps_path)
    except (ValueError, OSError) as error:
        print(f"input contract mismatch: {error}", flush=True)
        return 2
    try:
        summary = compare_gui_headless_consistency(
            headless_report,
            headless_steps,
            gui_report,
            gui_steps,
            request_quantitative_gates=
            arguments.request_quantitative_equivalence,
        )
    except Exception as error:
        print(
            f"input contract mismatch: unexpected evaluation error: "
            f"{type(error).__name__}",
            flush=True,
        )
        return 2
    import kcg_connector.grasp.single_finger_gui_consistency as evaluator
    summary["source"] = {
        "cli_sha256": _file_sha256(Path(__file__).resolve()),
        "evaluator_module_sha256": _file_sha256(
            Path(inspect.getsourcefile(evaluator)).resolve()
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "gui_headless_consistency_comparison.json").write_text(
        json.dumps(
            _json_safe(summary),
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
                "functional_structural_gui_headless_consistency_passed": (
                    summary[
                        "functional_structural_gui_headless_"
                        "consistency_passed"
                    ]
                ),
                "quantitative_equivalence_claimed": summary[
                    "quantitative_equivalence_claimed"
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
