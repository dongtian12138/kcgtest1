#!/usr/bin/env python3

"""Collect process-bound GPU samples and summarize D38999 performance evidence."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any, Sequence


SCHEMA_VERSION = "kcg_d38999_performance_collector_v1"
TASK_ID = "EIGHT-HOUR-A4-PERFORMANCE-MEASURED"
MEASURED_ROOT_RELATIVE_PATH = Path(
    "artifacts/agent_control/tasks/EIGHT-HOUR-A4-PERFORMANCE-MEASURED"
)
EXPECTED_GPU_NAME = "NVIDIA GeForce RTX 5070 Ti"
EXPECTED_PHYSICS_REPORT_COUNT = 3
EXPECTED_VISUAL_REPRESENTATION = "D38999_VISUAL_COMPLETE_V1"


def repository() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_nvidia_smi(fields: str, query_kind: str) -> list[list[str]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--query-{query_kind}={fields}",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi {query_kind} query failed: {result.stderr.strip()}")
    return [
        [cell.strip() for cell in row]
        for row in csv.reader(result.stdout.splitlines())
        if row
    ]


def gpu_inventory() -> list[dict[str, Any]]:
    rows = _run_nvidia_smi(
        "index,name,uuid,memory.total,memory.used,utilization.gpu", "gpu"
    )
    inventory = []
    for row in rows:
        if len(row) != 6:
            raise RuntimeError(f"unexpected nvidia-smi GPU row: {row!r}")
        inventory.append(
            {
                "gpu_index": int(row[0]),
                "gpu_name": row[1],
                "gpu_uuid": row[2],
                "device_memory_total_mib": float(row[3]),
                "device_memory_used_mib": float(row[4]),
                "device_utilization_percent": float(row[5]),
            }
        )
    if not inventory:
        raise RuntimeError("nvidia-smi returned no GPUs")
    return inventory


def process_gpu_rows() -> list[dict[str, Any]]:
    rows = _run_nvidia_smi(
        "pid,gpu_uuid,process_name,used_gpu_memory", "compute-apps"
    )
    processes = []
    for row in rows:
        if len(row) != 4:
            raise RuntimeError(f"unexpected nvidia-smi process row: {row!r}")
        processes.append(
            {
                "pid": int(row[0]),
                "gpu_uuid": row[1],
                "process_name": row[2],
                "used_gpu_memory_mib": float(row[3]),
            }
        )
    return processes


def query_process_gpu_sample(target_pid: int) -> dict[str, Any]:
    if target_pid <= 0:
        raise ValueError("target PID must be positive")
    inventory = gpu_inventory()
    processes = process_gpu_rows()
    matches = [row for row in processes if row["pid"] == target_pid]
    uuids = {row["gpu_uuid"] for row in matches}
    if len(uuids) > 1:
        raise RuntimeError("target process spans more than one GPU")
    selected = (
        next((gpu for gpu in inventory if gpu["gpu_uuid"] in uuids), None)
        if uuids
        else inventory[0]
    )
    if selected is None:
        raise RuntimeError("target process GPU UUID is absent from device inventory")
    return {
        "schema_version": SCHEMA_VERSION,
        "source": "nvidia-smi-live-query",
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "monotonic_s": time.monotonic(),
        "target_pid": target_pid,
        "target_process_found": bool(matches),
        "target_process_names": sorted({row["process_name"] for row in matches}),
        "target_process_memory_mib": sum(
            row["used_gpu_memory_mib"] for row in matches
        ),
        **selected,
        "measurement_kind": "REAL_PROCESS_BOUND_SAMPLE",
    }


def _exact_measured_path(raw: str, label: str) -> Path:
    actual = Path(raw).expanduser().resolve()
    root = (repository() / MEASURED_ROOT_RELATIVE_PATH).resolve()
    if not actual.is_relative_to(root):
        raise PermissionError(f"{label} must be below {root}")
    return actual


def _authorize_sampling() -> None:
    state = json.loads(
        (repository() / "artifacts/agent_control/MASTER_STATE.json").read_text(
            encoding="utf-8"
        )
    )
    performance = state.get("eight_hour_autonomy", {}).get(
        "a4_performance_measurement", {}
    )
    if state.get("task_id") != TASK_ID or state.get("status") != "IMPLEMENTING":
        raise PermissionError("measured performance collection is not the current task")
    if performance.get("execution_authorized") is not True:
        raise PermissionError("measured performance collection is not authorized")


def sample_gpu_process(
    target_pid: int, duration_s: float, interval_s: float, output: Path
) -> int:
    _authorize_sampling()
    if not (0.5 <= duration_s <= 1500.0):
        raise ValueError("duration must be between 0.5 and 1500 seconds")
    if not (0.1 <= interval_s <= 10.0):
        raise ValueError("sample interval must be between 0.1 and 10 seconds")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + duration_s
    count = 0
    with output.open("x", encoding="utf-8") as stream:
        while True:
            sample = query_process_gpu_sample(target_pid)
            stream.write(json.dumps(sample, allow_nan=False, sort_keys=True) + "\n")
            stream.flush()
            count += 1
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            time.sleep(min(interval_s, remaining))
    return count


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate a percentile of no values")
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_data(
    gpu_samples: Sequence[dict[str, Any]],
    physics_reports: Sequence[dict[str, Any]],
    render_timestamps: Sequence[dict[str, Any]],
    *,
    offline_fixture: bool,
) -> dict[str, Any]:
    if len(gpu_samples) < 2:
        raise ValueError("at least two GPU samples are required")
    if len(physics_reports) != EXPECTED_PHYSICS_REPORT_COUNT:
        raise ValueError(
            f"exactly {EXPECTED_PHYSICS_REPORT_COUNT} physics reports are required"
        )
    if len(render_timestamps) < 3:
        raise ValueError("at least three render timestamps are required")
    expected_gpu_source = "offline-test-fixture" if offline_fixture else "nvidia-smi-live-query"
    target_pids = {int(row.get("target_pid", -1)) for row in gpu_samples}
    gpu_uuids = {str(row.get("gpu_uuid", "")) for row in gpu_samples}
    if len(target_pids) != 1 or next(iter(target_pids)) <= 0:
        raise ValueError("GPU samples must bind one positive target PID")
    if len(gpu_uuids) != 1 or not next(iter(gpu_uuids)):
        raise ValueError("GPU samples must bind one GPU UUID")
    if any(row.get("source") != expected_gpu_source for row in gpu_samples):
        raise ValueError("GPU sample source does not match measurement kind")
    if any(row.get("target_process_found") is not True for row in gpu_samples):
        raise ValueError("every GPU sample must find the target process")
    gpu_names = {str(row.get("gpu_name", "")) for row in gpu_samples}
    if gpu_names != {EXPECTED_GPU_NAME}:
        raise ValueError(f"GPU identity differs from {EXPECTED_GPU_NAME}")
    process_memory = [
        _finite(row.get("target_process_memory_mib"), "process memory")
        for row in gpu_samples
    ]
    device_memory = [
        _finite(row.get("device_memory_used_mib"), "device memory")
        for row in gpu_samples
    ]
    device_utilization = [
        _finite(row.get("device_utilization_percent"), "device utilization")
        for row in gpu_samples
    ]
    physics_rates = []
    for index, report in enumerate(physics_reports, start=1):
        if report.get("status") != "PASS" or report.get("passed") is not True:
            raise ValueError(f"physics report {index} is not PASS")
        if report.get("simulation_started") is not True:
            raise ValueError(f"physics report {index} did not start simulation")
        physics_rates.append(
            _finite(
                report.get("physics_steps_per_wall_second"),
                f"physics report {index} rate",
            )
        )
    expected_render_source = (
        "offline-test-fixture" if offline_fixture else "isaac-render-loop-monotonic"
    )
    render_pid = next(iter(target_pids))
    frame_indices = []
    timestamps = []
    for row in render_timestamps:
        if row.get("source") != expected_render_source:
            raise ValueError("render timestamp source does not match measurement kind")
        if row.get("representation") != EXPECTED_VISUAL_REPRESENTATION:
            raise ValueError("render timestamps are not from the complete visual model")
        if row.get("full_visual_render") is not True:
            raise ValueError("render timestamp does not attest full visual rendering")
        if int(row.get("target_pid", -1)) != render_pid:
            raise ValueError("render timestamps and GPU samples use different PIDs")
        frame_indices.append(int(row.get("frame_index", -1)))
        timestamps.append(_finite(row.get("timestamp_monotonic_s"), "render timestamp"))
    if frame_indices != list(range(frame_indices[0], frame_indices[0] + len(frame_indices))):
        raise ValueError("render frame indices are not consecutive")
    intervals = [later - earlier for earlier, later in zip(timestamps, timestamps[1:])]
    if any(interval <= 0.0 for interval in intervals):
        raise ValueError("render timestamps are not strictly increasing")
    overall_render_fps = (len(timestamps) - 1) / (timestamps[-1] - timestamps[0])
    result = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "status": "OFFLINE_PASS" if offline_fixture else "MEASURED",
        "measurement_kind": "OFFLINE_TEST_FIXTURE" if offline_fixture else "REAL",
        "target_pid": render_pid,
        "gpu_name": next(iter(gpu_names)),
        "gpu_uuid": next(iter(gpu_uuids)),
        "gpu_sample_count": len(gpu_samples),
        "target_process_vram_peak_mib": max(process_memory),
        "device_vram_peak_mib": max(device_memory),
        "device_utilization_peak_percent": max(device_utilization),
        "physics_steps_per_wall_second": {
            "values": physics_rates,
            "minimum": min(physics_rates),
            "maximum": max(physics_rates),
            "mean": statistics.fmean(physics_rates),
            "range": max(physics_rates) - min(physics_rates),
        },
        "render": {
            "representation": EXPECTED_VISUAL_REPRESENTATION,
            "frame_count": len(timestamps),
            "overall_fps": overall_render_fps,
            "frame_time_p50_s": _percentile(intervals, 0.50),
            "frame_time_p95_s": _percentile(intervals, 0.95),
            "minimum_instantaneous_fps": 1.0 / max(intervals),
        },
        "offline_fixture": offline_fixture,
        "simulation_started_by_collector": False,
        "dynamic_pass_claimed": False,
        "formal_r12_generated": False,
        "hardware_authorized": False,
    }
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--sample-gpu", action="store_true")
    mode.add_argument("--summarize", action="store_true")
    parser.add_argument("--pid", type=int)
    parser.add_argument("--duration-s", type=float)
    parser.add_argument("--interval-s", type=float, default=0.5)
    parser.add_argument("--gpu-samples")
    parser.add_argument("--physics-report", action="append", default=[])
    parser.add_argument("--render-timestamps")
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _arguments(argv)
    output = _exact_measured_path(arguments.output, "performance output")
    if arguments.sample_gpu:
        if arguments.pid is None or arguments.duration_s is None:
            raise ValueError("--pid and --duration-s are required for GPU sampling")
        count = sample_gpu_process(
            arguments.pid, arguments.duration_s, arguments.interval_s, output
        )
        print(json.dumps({"sample_count": count, "output": str(output)}, sort_keys=True))
        return 0
    if arguments.gpu_samples is None or arguments.render_timestamps is None:
        raise ValueError("GPU samples and render timestamps are required")
    if len(arguments.physics_report) != EXPECTED_PHYSICS_REPORT_COUNT:
        raise ValueError("exactly three --physics-report arguments are required")
    if output.exists():
        raise FileExistsError(output)
    gpu_samples = _read_jsonl(Path(arguments.gpu_samples).expanduser().resolve())
    render_timestamps = _read_jsonl(
        Path(arguments.render_timestamps).expanduser().resolve()
    )
    physics_reports = [
        json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
        for path in arguments.physics_report
    ]
    summary = summarize_data(
        gpu_samples, physics_reports, render_timestamps, offline_fixture=False
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
