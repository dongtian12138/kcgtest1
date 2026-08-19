#!/usr/bin/env python3

"""Offline-only static validation for the A3 repeat runner and aggregator."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Sequence


def _repository() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_orchestrator():
    path = _repository() / "tools/agent_control/run_multilayer_repeats.py"
    spec = importlib.util.spec_from_file_location("run_multilayer_repeats", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load repeat orchestrator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_report(index: int, events: Sequence[str]) -> dict[str, Any]:
    base = 0.0055
    event_first = {
        event: {"datum_B_separation_m": base + position * 0.001 + index * 1.0e-7}
        for position, event in enumerate(events)
    }
    return {
        "repeat_index": index,
        "independent_process_pid": 90000 + index,
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
        "observed_event_order": list(events),
        "event_first": event_first,
        "maximum_driver_force_component_n": 7.0 + index * 0.01,
        "maximum_driver_torque_component_nm": 0.2 + index * 0.001,
        "maximum_fixed_receptacle_translation_drift_m": 4.0e-6 + index * 1.0e-8,
        "physics_steps_per_wall_second": 900.0 + index,
        "terminal_separation_m": 0.01505,
    }


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _arguments(argv)
    root = _repository()
    output = Path(arguments.output).expanduser().resolve()
    expected_output = (
        root
        / "artifacts/agent_control/tasks/EIGHT-HOUR-A3-REPEAT-ORCHESTRATION/"
        "STATIC_VALIDATION.json"
    ).resolve()
    if output != expected_output:
        raise PermissionError(f"output path is frozen: {output} != {expected_output}")
    if output.exists():
        raise FileExistsError(output)
    orchestrator = _load_orchestrator()
    plan = orchestrator.build_plan()
    checks: dict[str, bool] = {}
    commands = plan["commands"]
    checks["exactly_three_commands"] = len(commands) == 3
    checks["indices_are_1_2_3"] = [row["repeat_index"] for row in commands] == [1, 2, 3]
    checks["output_dirs_unique"] = len({row["output_dir"] for row in commands}) == 3
    checks["portable_roots_unique"] = len(
        {row["kit_portable_root"] for row in commands}
    ) == 3
    checks["all_commands_have_25m_timeout"] = all(
        row["command"][:3] == ["timeout", "--signal=TERM", "25m"]
        for row in commands
    )
    checks["all_commands_are_separate_processes"] = all(
        row["process_isolation_required"] is True for row in commands
    )
    checks["plan_does_not_claim_dynamic_pass"] = (
        plan["dynamic_pass_claimed"] is False
        and plan["simulation_started"] is False
        and plan["dynamic_execution_authorized"] is False
    )
    checks["a2_gate_closed_now"] = plan["a2_dynamic_pass_present"] is False
    try:
        orchestrator.validate_a2_gate()
    except (FileNotFoundError, PermissionError):
        checks["missing_a2_evidence_rejected"] = True
    else:
        checks["missing_a2_evidence_rejected"] = False
    reports = [
        _synthetic_report(index, orchestrator.EVENT_ORDER)
        for index in range(1, 4)
    ]
    aggregate = orchestrator.aggregate_reports(reports, synthetic_fixture=True)
    checks["synthetic_fixture_stays_offline"] = (
        aggregate["status"] == "OFFLINE_PASS"
        and aggregate["synthetic_test_fixture"] is True
        and aggregate["simulation_started"] is False
        and aggregate["dynamic_pass_claimed"] is False
    )
    checks["aggregate_keeps_all_seven_events"] = (
        set(aggregate["event_position_ranges"]) == set(orchestrator.EVENT_ORDER)
    )
    try:
        orchestrator.aggregate_reports(reports[:2], synthetic_fixture=True)
    except ValueError:
        checks["incomplete_report_set_rejected"] = True
    else:
        checks["incomplete_report_set_rejected"] = False
    repeat_source = (
        root / "src/kcg_connector/isaac/d38999_multilayer_repeat_bench.py"
    ).read_text(encoding="utf-8")
    checks["repeat_reuses_nominal_runtime"] = "nominal._runtime(" in repeat_source
    checks["repeat_has_truth_independent_runtime"] = all(
        token not in repeat_source
        for token in (
            "get_ground_truth_pose",
            "contact_normal_used = True",
            "event_truth_used = True",
        )
    )
    failed = [name for name, passed in checks.items() if not passed]
    result = {
        "schema_version": 1,
        "task_id": "EIGHT-HOUR-A3-REPEAT-ORCHESTRATION",
        "status": "STATIC_PASS" if not failed else "FAIL",
        "checks": checks,
        "check_count": len(checks),
        "passed_check_count": len(checks) - len(failed),
        "failed_checks": failed,
        "synthetic_test_fixture_used": True,
        "synthetic_fixture_persisted_as_dynamic_evidence": False,
        "simulation_started": False,
        "dynamic_pass_claimed": False,
        "formal_r12_generated": False,
        "hardware_authorized": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
