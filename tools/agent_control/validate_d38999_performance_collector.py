#!/usr/bin/env python3

"""Offline-only validation of the D38999 performance collector."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Sequence


def _repository() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_collector():
    path = _repository() / "tools/agent_control/d38999_performance_collector.py"
    spec = importlib.util.spec_from_file_location("d38999_performance_collector", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load performance collector")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _gpu_samples() -> list[dict[str, Any]]:
    return [
        {
            "source": "offline-test-fixture",
            "target_pid": 4242,
            "target_process_found": True,
            "gpu_name": "NVIDIA GeForce RTX 5070 Ti",
            "gpu_uuid": "GPU-offline-fixture",
            "target_process_memory_mib": value,
            "device_memory_used_mib": value + 500.0,
            "device_utilization_percent": utilization,
        }
        for value, utilization in ((1024.0, 70.0), (1536.0, 80.0), (1400.0, 75.0))
    ]


def _physics_reports() -> list[dict[str, Any]]:
    return [
        {
            "status": "PASS",
            "passed": True,
            "simulation_started": True,
            "physics_steps_per_wall_second": value,
        }
        for value in (790.0, 800.0, 810.0)
    ]


def _render_timestamps() -> list[dict[str, Any]]:
    return [
        {
            "source": "offline-test-fixture",
            "target_pid": 4242,
            "representation": "D38999_VISUAL_COMPLETE_V1",
            "full_visual_render": True,
            "frame_index": index,
            "timestamp_monotonic_s": index * 0.02,
        }
        for index in range(11)
    ]


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _arguments(argv)
    root = _repository()
    output = Path(arguments.output).expanduser().resolve()
    expected = (
        root
        / "artifacts/agent_control/tasks/EIGHT-HOUR-A4-PERFORMANCE-TOOL/"
        "STATIC_VALIDATION.json"
    ).resolve()
    if output != expected:
        raise PermissionError(f"output path is frozen: {output} != {expected}")
    if output.exists():
        raise FileExistsError(output)
    collector = _load_collector()
    checks: dict[str, bool] = {}
    inventory = collector.gpu_inventory()
    checks["nvidia_smi_gpu_inventory_available"] = len(inventory) >= 1
    checks["expected_5070ti_present"] = any(
        row["gpu_name"] == collector.EXPECTED_GPU_NAME for row in inventory
    )
    summary = collector.summarize_data(
        _gpu_samples(),
        _physics_reports(),
        _render_timestamps(),
        offline_fixture=True,
    )
    checks["offline_fixture_status_is_offline_pass"] = summary["status"] == "OFFLINE_PASS"
    checks["offline_fixture_never_claims_dynamic_pass"] = (
        summary["offline_fixture"] is True
        and summary["dynamic_pass_claimed"] is False
        and summary["simulation_started_by_collector"] is False
    )
    checks["pid_bound_process_vram_peak"] = (
        summary["target_pid"] == 4242
        and summary["target_process_vram_peak_mib"] == 1536.0
    )
    checks["device_vram_peak_calculated"] = summary["device_vram_peak_mib"] == 2036.0
    checks["physics_rate_three_reports"] = (
        summary["physics_steps_per_wall_second"]["values"] == [790.0, 800.0, 810.0]
        and summary["physics_steps_per_wall_second"]["range"] == 20.0
    )
    checks["render_fps_from_monotonic_timestamps"] = abs(
        summary["render"]["overall_fps"] - 50.0
    ) < 1.0e-9
    checks["full_visual_representation_preserved"] = (
        summary["render"]["representation"] == "D38999_VISUAL_COMPLETE_V1"
    )
    mixed_pid = _gpu_samples()
    mixed_pid[-1] = {**mixed_pid[-1], "target_pid": 4343}
    try:
        collector.summarize_data(
            mixed_pid, _physics_reports(), _render_timestamps(), offline_fixture=True
        )
    except ValueError:
        checks["mixed_gpu_pid_rejected"] = True
    else:
        checks["mixed_gpu_pid_rejected"] = False
    missing_process = _gpu_samples()
    missing_process[-1] = {**missing_process[-1], "target_process_found": False}
    try:
        collector.summarize_data(
            missing_process,
            _physics_reports(),
            _render_timestamps(),
            offline_fixture=True,
        )
    except ValueError:
        checks["missing_target_process_rejected"] = True
    else:
        checks["missing_target_process_rejected"] = False
    bad_frames = _render_timestamps()
    bad_frames[5] = {**bad_frames[5], "timestamp_monotonic_s": 0.05}
    try:
        collector.summarize_data(
            _gpu_samples(), _physics_reports(), bad_frames, offline_fixture=True
        )
    except ValueError:
        checks["non_monotonic_render_time_rejected"] = True
    else:
        checks["non_monotonic_render_time_rejected"] = False
    try:
        collector.summarize_data(
            _gpu_samples(),
            _physics_reports()[:2],
            _render_timestamps(),
            offline_fixture=True,
        )
    except ValueError:
        checks["incomplete_physics_reports_rejected"] = True
    else:
        checks["incomplete_physics_reports_rejected"] = False
    wrong_visual = _render_timestamps()
    wrong_visual[-1] = {**wrong_visual[-1], "representation": "D38999_ASSEMBLY_CONTROL_V1"}
    try:
        collector.summarize_data(
            _gpu_samples(), _physics_reports(), wrong_visual, offline_fixture=True
        )
    except ValueError:
        checks["non_visual_render_source_rejected"] = True
    else:
        checks["non_visual_render_source_rejected"] = False
    checks["hardware_authorization_remains_false"] = summary["hardware_authorized"] is False
    failed = [name for name, passed in checks.items() if not passed]
    result = {
        "schema_version": 1,
        "task_id": "EIGHT-HOUR-A4-PERFORMANCE-TOOL",
        "status": "OFFLINE_PASS" if not failed else "FAIL",
        "checks": checks,
        "check_count": len(checks),
        "passed_check_count": len(checks) - len(failed),
        "failed_checks": failed,
        "gpu_inventory": inventory,
        "fixture_summary": summary,
        "synthetic_test_fixture_used": True,
        "real_performance_measurement_started": False,
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
