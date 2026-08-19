#!/usr/bin/env python3

"""Run one independently isolated repeat of the frozen multilayer nominal bench.

This entry point deliberately reuses ``d38999_multilayer_nominal_bench._runtime``
instead of copying its physical model.  It only opens a repeat output path after
an independently accepted A2 result and the dedicated A3 dynamic authorization
are both present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Sequence

import d38999_multilayer_nominal_bench as nominal


SCHEMA_VERSION = "kcg_d38999_multilayer_repeat_bench_v1"
TASK_ID = "EIGHT-HOUR-A3-REPEAT-DYNAMIC"
A2_TASK_ID = "TASK-R12-MULTILAYER-005"
A2_RESULT_RELATIVE_PATH = Path(
    "artifacts/agent_control/tasks/TASK-R12-MULTILAYER-005/TASK_RESULT.json"
)
A2_REPORT_RELATIVE_PATH = Path(
    "artifacts/agent_control/tasks/TASK-R12-MULTILAYER-005/NOMINAL/report.json"
)
REPEAT_ROOT_RELATIVE_PATH = Path(
    "artifacts/agent_control/tasks/EIGHT-HOUR-A3-REPEAT-DYNAMIC"
)
PORTABLE_ROOT_PREFIX = "/tmp/kcg-eight-hour-a3-repeat-run-"


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", required=True)
    parser.add_argument("--repeat-index", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--kit-portable-root", required=True)
    parser.add_argument("--a2-acceptance", required=True)
    return parser.parse_args(argv)


def _repository() -> Path:
    return Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _exact_file(raw: str, relative: Path, label: str) -> Path:
    actual = Path(raw).expanduser().resolve()
    expected = (_repository() / relative).resolve()
    if actual != expected:
        raise PermissionError(f"{label} path is frozen: {actual} != {expected}")
    if not actual.is_file():
        raise FileNotFoundError(actual)
    return actual


def validate_a2_acceptance(raw: str) -> dict[str, Any]:
    acceptance_path = _exact_file(raw, A2_RESULT_RELATIVE_PATH, "A2 acceptance")
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    required = {
        "task_id": A2_TASK_ID,
        "outcome": "PASS",
        "status": "DYNAMIC_PASS",
        "formal_nominal_pass_claimed": True,
        "formal_runs": "1/1",
        "hardware_authorized": False,
    }
    for key, expected in required.items():
        if acceptance.get(key) != expected:
            raise PermissionError(
                f"A2 acceptance {key} is not authoritative: "
                f"{acceptance.get(key)!r} != {expected!r}"
            )
    report_path = (_repository() / A2_REPORT_RELATIVE_PATH).resolve()
    if acceptance.get("formal_report") != str(A2_REPORT_RELATIVE_PATH):
        raise PermissionError("A2 acceptance does not bind the frozen nominal report path")
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report_sha256 = _sha256(report_path)
    if acceptance.get("formal_report_sha256") != report_sha256:
        raise PermissionError("A2 formal report SHA-256 does not match")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report_required = {
        "bench_id": nominal.BENCH_ID,
        "mode": "formal_multilayer_nominal",
        "status": "PASS",
        "passed": True,
        "simulation_started": True,
        "solver_error_count": 0,
        "object_pose_write_after_physics_start_count": 0,
        "formal_r12_generated": False,
        "hardware_authorized": False,
    }
    for key, expected in report_required.items():
        if report.get(key) != expected:
            raise PermissionError(
                f"A2 report {key} is not accepted: {report.get(key)!r} != {expected!r}"
            )
    for role, expected in nominal.EXPECTED_SHA256.items():
        if report.get("input_sha256", {}).get(role) != expected:
            raise PermissionError(f"A2 report frozen {role} SHA-256 does not match")
    return {
        "path": str(acceptance_path),
        "sha256": _sha256(acceptance_path),
        "formal_report": str(report_path),
        "formal_report_sha256": report_sha256,
    }


def _authorize(arguments: argparse.Namespace, output: Path, portable: Path) -> dict[str, Any]:
    repository = _repository()
    expected_output = (
        repository / REPEAT_ROOT_RELATIVE_PATH / f"RUN_{arguments.repeat_index:02d}"
    ).resolve()
    expected_portable = Path(
        f"{PORTABLE_ROOT_PREFIX}{arguments.repeat_index:02d}"
    ).resolve()
    if output != expected_output:
        raise PermissionError(f"repeat output path is frozen: {output} != {expected_output}")
    if portable != expected_portable:
        raise PermissionError(
            f"repeat portable root is frozen: {portable} != {expected_portable}"
        )
    state = json.loads(
        (repository / "artifacts/agent_control/MASTER_STATE.json").read_text(
            encoding="utf-8"
        )
    )
    autonomy = state.get("eight_hour_autonomy", {})
    repeat = autonomy.get("a3_repeat_execution", {})
    if state.get("task_id") != TASK_ID or state.get("status") != "IMPLEMENTING":
        raise PermissionError("A3 repeat runtime is not the current implementing task")
    if autonomy.get("current_queue_task") != "A3":
        raise PermissionError("work queue is not positioned at A3")
    if repeat.get("execution_authorized") is not True:
        raise PermissionError("A3 dynamic repeat execution is not authorized")
    if repeat.get("run_limit") != 3:
        raise PermissionError("A3 repeat run limit must be exactly three")
    return validate_a2_acceptance(arguments.a2_acceptance)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _arguments(argv)
    frozen = nominal._load_frozen_inputs(arguments.contract, arguments.model)
    output = Path(arguments.output_dir).expanduser().resolve()
    portable = Path(arguments.kit_portable_root).expanduser().resolve()
    a2_evidence = _authorize(arguments, output, portable)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite repeat evidence: {output}")
    if portable.exists():
        raise FileExistsError(f"refusing to reuse Kit portable root: {portable}")
    output.mkdir(parents=True, exist_ok=False)
    portable.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("WARP_CACHE_PATH", str(portable / "warp-cache"))
    sys.argv = [sys.argv[0], "--portable-root", str(portable)]
    from isaacsim import SimulationApp

    application = SimulationApp(
        {
            "headless": True,
            "hide_ui": True,
            "renderer": "Minimal",
            "multi_gpu": False,
            "fast_shutdown": True,
            "enable_crashreporter": False,
        }
    )
    status = 1
    runtime_arguments = argparse.Namespace(
        run=True,
        initialize_only=False,
        contract=arguments.contract,
        model=arguments.model,
        output_dir=arguments.output_dir,
        kit_portable_root=arguments.kit_portable_root,
    )
    try:
        report = nominal._runtime(runtime_arguments, frozen, application)
        report.update(
            {
                "schema_version": SCHEMA_VERSION,
                "repeat_task_id": TASK_ID,
                "repeat_index": arguments.repeat_index,
                "independent_process_pid": os.getpid(),
                "input_sha256": frozen["input_sha256"],
                "a2_acceptance": a2_evidence,
                "nominal_runtime_source_sha256": _sha256(
                    Path(nominal.__file__).resolve()
                ),
                "repeat_run_passed": report.get("passed") is True,
                "formal_repeat_milestone_claimed": False,
            }
        )
        status = 0 if report.get("passed") is True else 1
    except BaseException as error:
        traceback.print_exc()
        report = {
            "schema_version": SCHEMA_VERSION,
            "repeat_task_id": TASK_ID,
            "repeat_index": arguments.repeat_index,
            "status": "ERROR",
            "passed": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "simulation_started": True,
            "formal_repeat_milestone_claimed": False,
            "formal_r12_generated": False,
            "hardware_authorized": False,
        }
    finally:
        (output / "report.json").write_text(
            json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        application.close()
    print(json.dumps(report, allow_nan=False, sort_keys=True))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
