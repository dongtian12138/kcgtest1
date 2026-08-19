#!/usr/bin/env python3

"""Plan, execute, or aggregate exactly three independent multilayer repeats."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = "kcg_d38999_multilayer_repeat_orchestrator_v1"
RUN_COUNT = 3
COMMAND_TIMEOUT_SECONDS = 1500
EVENT_ORDER = (
    "five_key_polarization",
    "three_start_thread_entry",
    "spring_finger_engagement",
    "first_pin_socket_spring_touch",
    "pin_barrier_seal_contact",
    "seal_compression",
    "shell_to_shell_metal_bottoming",
)
CONTRACT_RELATIVE_PATH = Path(
    "src/kcg_connector/config/d38999_master_model_contract_v1.yaml"
)
MODEL_RELATIVE_PATH = Path(
    "artifacts/kcg_connector/isaac/d38999_multilayer_v1/"
    "D38999_ASSEMBLY_CONTROL_V1.usda"
)
A2_ACCEPTANCE_RELATIVE_PATH = Path(
    "artifacts/agent_control/tasks/TASK-R12-MULTILAYER-005/TASK_RESULT.json"
)
REPEAT_SCRIPT_RELATIVE_PATH = Path(
    "src/kcg_connector/isaac/d38999_multilayer_repeat_bench.py"
)
WRAPPER_RELATIVE_PATH = Path("src/kcg_connector/isaac/run_isaac_python.sh")
REPEAT_ROOT_RELATIVE_PATH = Path(
    "artifacts/agent_control/tasks/EIGHT-HOUR-A3-REPEAT-DYNAMIC"
)
STATIC_PLAN_RELATIVE_PATH = Path(
    "artifacts/agent_control/tasks/EIGHT-HOUR-A3-REPEAT-ORCHESTRATION/STATIC_PLAN.json"
)
AGGREGATE_RELATIVE_PATH = REPEAT_ROOT_RELATIVE_PATH / "AGGREGATE.json"
PORTABLE_ROOT_PREFIX = "/tmp/kcg-eight-hour-a3-repeat-run-"


def repository() -> Path:
    return Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_output(raw: str, relative: Path, label: str) -> Path:
    actual = Path(raw).expanduser().resolve()
    expected = (repository() / relative).resolve()
    if actual != expected:
        raise PermissionError(f"{label} path is frozen: {actual} != {expected}")
    return actual


def build_plan() -> dict[str, Any]:
    root = repository()
    contract = root / CONTRACT_RELATIVE_PATH
    model = root / MODEL_RELATIVE_PATH
    a2 = root / A2_ACCEPTANCE_RELATIVE_PATH
    repeat_script = root / REPEAT_SCRIPT_RELATIVE_PATH
    wrapper = root / WRAPPER_RELATIVE_PATH
    for path in (contract, model, repeat_script, wrapper):
        if not path.is_file():
            raise FileNotFoundError(path)
    commands = []
    for index in range(1, RUN_COUNT + 1):
        output = root / REPEAT_ROOT_RELATIVE_PATH / f"RUN_{index:02d}"
        portable = Path(f"{PORTABLE_ROOT_PREFIX}{index:02d}")
        command = [
            "timeout",
            "--signal=TERM",
            "25m",
            str(wrapper),
            str(repeat_script),
            "--run",
            "--repeat-index",
            str(index),
            "--contract",
            str(contract),
            "--model",
            str(model),
            "--output-dir",
            str(output),
            "--kit-portable-root",
            str(portable),
            "--a2-acceptance",
            str(a2),
        ]
        commands.append(
            {
                "repeat_index": index,
                "command": command,
                "output_dir": str(output),
                "kit_portable_root": str(portable),
                "process_isolation_required": True,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": "EIGHT-HOUR-A3-REPEAT-ORCHESTRATION",
        "status": "STATIC_PASS",
        "run_count": RUN_COUNT,
        "command_timeout_seconds": COMMAND_TIMEOUT_SECONDS,
        "commands": commands,
        "frozen_input_sha256": {
            "contract": sha256(contract),
            "model": sha256(model),
            "repeat_script": sha256(repeat_script),
            "wrapper": sha256(wrapper),
        },
        "a2_acceptance_path": str(a2),
        "a2_dynamic_pass_present": a2.is_file(),
        "dynamic_execution_authorized": False,
        "simulation_started": False,
        "dynamic_pass_claimed": False,
        "formal_r12_generated": False,
        "hardware_authorized": False,
    }


def validate_a2_gate() -> dict[str, Any]:
    isaac_dir = repository() / "src/kcg_connector/isaac"
    sys.path.insert(0, str(isaac_dir))
    try:
        import d38999_multilayer_repeat_bench as repeat_bench

        return repeat_bench.validate_a2_acceptance(
            str(repository() / A2_ACCEPTANCE_RELATIVE_PATH)
        )
    finally:
        sys.path.pop(0)


def _finite_number(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} is not finite")
    return number


def aggregate_reports(
    reports: Sequence[dict[str, Any]], *, synthetic_fixture: bool = False
) -> dict[str, Any]:
    if len(reports) != RUN_COUNT:
        raise ValueError(f"exactly {RUN_COUNT} reports are required")
    event_samples: dict[str, list[float]] = {event: [] for event in EVENT_ORDER}
    scalar_fields = (
        "maximum_driver_force_component_n",
        "maximum_driver_torque_component_nm",
        "maximum_fixed_receptacle_translation_drift_m",
        "physics_steps_per_wall_second",
    )
    scalar_samples: dict[str, list[float]] = {field: [] for field in scalar_fields}
    run_rows = []
    seen_indices: set[int] = set()
    seen_pids: set[int] = set()
    for position, report in enumerate(reports, start=1):
        index = int(report.get("repeat_index", position))
        if index not in (1, 2, 3) or index in seen_indices:
            raise ValueError("repeat indices must be unique 1, 2, 3")
        seen_indices.add(index)
        pid = int(report.get("independent_process_pid", 100000 + index))
        if pid in seen_pids:
            raise ValueError("repeat process identifiers must be unique")
        seen_pids.add(pid)
        required = {
            "status": "PASS",
            "passed": True,
            "simulation_started": True,
            "solver_error_count": 0,
            "object_pose_write_after_physics_start_count": 0,
            "event_inventory_pass": True,
            "event_order_pass": True,
            "event_position_pass": True,
            "driver_force_pass": True,
            "driver_torque_pass": True,
            "fixed_receptacle_drift_pass": True,
            "hard_penetration_pass": True,
            "formal_r12_generated": False,
            "hardware_authorized": False,
        }
        for key, expected in required.items():
            if report.get(key) != expected:
                raise ValueError(
                    f"repeat {index} {key} is not accepted: "
                    f"{report.get(key)!r} != {expected!r}"
                )
        if tuple(report.get("observed_event_order", ())) != EVENT_ORDER:
            raise ValueError(f"repeat {index} event order differs from contract")
        for event in EVENT_ORDER:
            row = report.get("event_first", {}).get(event)
            if not isinstance(row, dict):
                raise ValueError(f"repeat {index} lacks event {event}")
            event_samples[event].append(
                _finite_number(row.get("datum_B_separation_m"), event)
            )
        for field in scalar_fields:
            scalar_samples[field].append(_finite_number(report.get(field), field))
        run_rows.append(
            {
                "repeat_index": index,
                "process_id": pid,
                "status": report["status"],
                "terminal_separation_m": _finite_number(
                    report.get("terminal_separation_m"), "terminal separation"
                ),
            }
        )
    event_ranges = {
        event: {
            "minimum_m": min(values),
            "maximum_m": max(values),
            "range_m": max(values) - min(values),
            "values_m": values,
        }
        for event, values in event_samples.items()
    }
    scalar_ranges = {
        field: {
            "minimum": min(values),
            "maximum": max(values),
            "range": max(values) - min(values),
            "values": values,
        }
        for field, values in scalar_samples.items()
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": "EIGHT-HOUR-A3-REPEAT-DYNAMIC",
        "status": "OFFLINE_PASS" if synthetic_fixture else "DYNAMIC_PASS",
        "run_count": RUN_COUNT,
        "independent_process_count": len(seen_pids),
        "runs": sorted(run_rows, key=lambda row: row["repeat_index"]),
        "event_position_ranges": event_ranges,
        "metric_ranges": scalar_ranges,
        "synthetic_test_fixture": synthetic_fixture,
        "simulation_started": not synthetic_fixture,
        "dynamic_pass_claimed": not synthetic_fixture,
        "formal_r12_generated": False,
        "hardware_authorized": False,
    }


def load_real_reports(root: Path) -> list[dict[str, Any]]:
    reports = []
    for index in range(1, RUN_COUNT + 1):
        path = root / f"RUN_{index:02d}" / "report.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        reports.append(json.loads(path.read_text(encoding="utf-8")))
    return reports


def execute_plan(plan: dict[str, Any]) -> None:
    validate_a2_gate()
    state = json.loads(
        (repository() / "artifacts/agent_control/MASTER_STATE.json").read_text(
            encoding="utf-8"
        )
    )
    execution = state.get("eight_hour_autonomy", {}).get(
        "a3_repeat_execution", {}
    )
    if execution.get("execution_authorized") is not True:
        raise PermissionError("A3 dynamic repeat execution is not authorized")
    if execution.get("run_limit") != RUN_COUNT:
        raise PermissionError("A3 dynamic repeat run limit must be exactly three")
    log_root = repository() / REPEAT_ROOT_RELATIVE_PATH / "PROCESS_LOGS"
    log_root.mkdir(parents=True, exist_ok=False)
    for row in plan["commands"]:
        with (log_root / f"RUN_{row['repeat_index']:02d}.log").open(
            "w", encoding="utf-8"
        ) as stream:
            result = subprocess.run(
                row["command"],
                cwd=repository(),
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if result.returncode != 0:
            raise RuntimeError(
                f"repeat {row['repeat_index']} exited {result.returncode}"
            )


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--aggregate", action="store_true")
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _arguments(argv)
    if arguments.plan:
        output = exact_output(arguments.output, STATIC_PLAN_RELATIVE_PATH, "static plan")
        result = build_plan()
    elif arguments.execute:
        output = exact_output(arguments.output, AGGREGATE_RELATIVE_PATH, "aggregate")
        plan = build_plan()
        execute_plan(plan)
        result = aggregate_reports(
            load_real_reports(repository() / REPEAT_ROOT_RELATIVE_PATH)
        )
    else:
        output = exact_output(arguments.output, AGGREGATE_RELATIVE_PATH, "aggregate")
        validate_a2_gate()
        result = aggregate_reports(
            load_real_reports(repository() / REPEAT_ROOT_RELATIVE_PATH)
        )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
