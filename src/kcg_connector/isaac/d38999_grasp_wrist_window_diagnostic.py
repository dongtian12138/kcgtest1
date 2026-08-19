#!/usr/bin/env python3
"""Read-only streaming diagnostic over controller_steps.jsonl.

Segments episodes by controller phase and computes per-stream statistics
for every wrist stream that is actually present (raw sensor frame,
canonical, empty-baseline-compensated, payload reference and payload
increment) plus the three finger-root proxies.  Optional pre-lift
overlap comparison aligns two episodes by global_step and compares each
stream/field sample-by-sample with explicit availability counts.  Never
imports Isaac, never rewrites reports, never infers missing curves from
posthoc endpoints.  Evidence only: can never alter a PASS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SCHEMA_VERSION = "kcg_grasp_wrist_window_diagnostic_v1"

ROOT_CHANNELS = ("f1", "f2", "f3")

# Near-constant tolerance semantics: a channel is treated as
# near-constant when its standard deviation is at most
# NEAR_CONSTANT_ABS_TOL + NEAR_CONSTANT_REL_TOL * abs(mean) of that
# channel.  mean/std/rms are always reported; only correlation and
# dominant frequency become null with available=false.
NEAR_CONSTANT_ABS_TOL = 1e-12
NEAR_CONSTANT_REL_TOL = 1e-9

WRIST_STREAMS = (
    ("wrist_wrench_raw_sensor_frame", "handbase_link_sensor_frame"),
    ("wrist_wrench_canonical", "handbase_link_canonical_sensor_frame"),
    (
        "wrist_wrench_empty_baseline_compensated",
        "handbase_link_canonical_sensor_frame",
    ),
    ("wrist_wrench_payload_reference", "handbase_link_canonical_sensor_frame"),
    (
        "wrist_wrench_payload_reference_increment",
        "handbase_link_canonical_sensor_frame",
    ),
)

WRIST_CHANNELS = ("fx_n", "fy_n", "fz_n", "tx_nm", "ty_nm", "tz_nm")

KNOWN_PHASES = (
    "physical_hand_closure",
    "closed_hand_seating",
    "physical_grip_preload",
    "physical_grip_consolidation",
    "physical_grip_lift_stage_1",
    "physical_grip_lift_stage_2",
    "physical_grip_lift_stage_3",
    "unsupported_final_hold",
)


def _clean(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    if isinstance(value, list):
        return [_clean(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    return value


def _series_stats(values: Sequence[float]) -> dict[str, Any]:
    data = np.asarray([float(v) for v in values], dtype=np.float64)
    count = int(data.size)
    split = count // 2
    first = data[:split]
    second = data[split:]
    mean = float(np.mean(data))
    drift = (
        float(np.mean(second)) - float(np.mean(first))
        if first.size and second.size
        else None
    )
    detrended = data - mean
    dominant = None
    near_constant = float(np.std(data)) <= (
        NEAR_CONSTANT_ABS_TOL + NEAR_CONSTANT_REL_TOL * abs(mean)
    )
    if detrended.size >= 4 and not near_constant:
        spectrum = np.abs(np.fft.rfft(detrended))[1:]
        if spectrum.size:
            dominant = float(np.argmax(spectrum) + 1) / float(detrended.size)
    return {
        "count": count,
        "mean": mean,
        "std": float(np.std(data)),
        "rms": float(np.sqrt(np.mean(data ** 2))),
        "first_half_mean": float(np.mean(first)) if first.size else None,
        "second_half_mean": float(np.mean(second)) if second.size else None,
        "first_to_second_half_drift": drift,
        "peak_abs": float(np.max(np.abs(data))),
        "dominant_cycles_per_step": dominant,
        "near_constant": near_constant,
    }


def _stream_stats(records, field: str, rate_hz) -> dict[str, Any]:
    rows = []
    for record in records:
        value = record.get(field)
        if isinstance(value, (list, tuple)) and len(value) == 6:
            if all(
                isinstance(v, (int, float))
                and not isinstance(v, bool)
                and math.isfinite(float(v))
                for v in value
            ):
                rows.append([float(v) for v in value])
    if not rows:
        return {
            "available": False,
            "sample_count": 0,
            "missing_or_invalid": len(records),
            "unavailable_and_not_inferred": True,
        }
    matrix = np.asarray(rows, dtype=np.float64)
    per_channel = {
        name: _series_stats(matrix[:, index])
        for index, name in enumerate(WRIST_CHANNELS)
    }
    for name, stats in per_channel.items():
        dominant = stats["dominant_cycles_per_step"]
        stats["dominant_hz"] = (
            dominant * float(rate_hz)
            if dominant is not None and rate_hz is not None
            else None
        )
        stats["dominant_hz_available"] = (
            dominant is not None and rate_hz is not None
        )

    def near_constant(column):
        return float(np.std(column)) <= (
            NEAR_CONSTANT_ABS_TOL
            + NEAR_CONSTANT_REL_TOL * abs(float(np.mean(column)))
        )

    correlation = {}
    for index_a, name_a in enumerate(WRIST_CHANNELS):
        correlation[name_a] = {}
        for index_b, name_b in enumerate(WRIST_CHANNELS):
            column_a = matrix[:, index_a]
            column_b = matrix[:, index_b]
            if near_constant(column_a) or near_constant(column_b):
                correlation[name_a][name_b] = None
                correlation[name_a][name_b + "_available"] = False
            else:
                correlation[name_a][name_b] = float(
                    np.corrcoef(column_a, column_b)[0, 1]
                )
                correlation[name_a][name_b + "_available"] = True
    return {
        "available": True,
        "sample_count": int(matrix.shape[0]),
        "missing_or_invalid": len(records) - int(matrix.shape[0]),
        "per_channel": per_channel,
        "correlation_matrix": correlation,
    }


def _segment_of(record: Mapping[str, Any]) -> str:
    phase = record.get("phase")
    if phase == "physical_grip_consolidation":
        evidence = record.get("controller_evidence") or {}
        window_step = evidence.get("consolidation_window_step", 0)
        if isinstance(window_step, int) and window_step > 0:
            return "consolidation_window"
        return "consolidation_ramp"
    if phase in KNOWN_PHASES:
        return str(phase)
    return f"other:{phase}"


def _load_records(episode_dir: Path):
    path = episode_dir / "controller_steps.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"missing controller steps: {path}")
    records = []
    problems = []
    previous_step = None
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                problems.append(f"line {line_number}: {error}")
                continue
            step = record.get("global_step")
            if isinstance(step, bool) or not isinstance(step, int):
                problems.append(
                    f"line {line_number}: global_step not a non-bool integer"
                )
            elif previous_step is not None and step <= previous_step:
                problems.append(
                    f"line {line_number}: global_step not strictly increasing"
                )
            if isinstance(step, int) and not isinstance(step, bool):
                previous_step = step
            records.append(record)
    return records, problems


def _segment_stats(records, rate_hz) -> dict[str, Any]:
    result: dict[str, Any] = {
        "step_count": len(records),
        "wrist_streams": {
            name: _stream_stats(records, name, rate_hz)
            for name, _frame in WRIST_STREAMS
        },
        "root": None,
        "nut_pose_velocity_per_step": "unavailable_and_not_inferred",
    }
    root_rows = []
    for record in records:
        root = record.get("finger_root_torque_proxy_nm")
        if isinstance(root, Mapping) and set(root) >= set(ROOT_CHANNELS):
            if all(
                isinstance(root[c], (int, float))
                and not isinstance(root[c], bool)
                and math.isfinite(float(root[c]))
                for c in ROOT_CHANNELS
            ):
                root_rows.append([float(root[c]) for c in ROOT_CHANNELS])
    if root_rows:
        matrix = np.asarray(root_rows, dtype=np.float64)
        result["root"] = {
            "available": True,
            "sample_count": int(matrix.shape[0]),
            "missing_or_invalid": len(records) - int(matrix.shape[0]),
            "per_channel": {
                name: _series_stats(matrix[:, index])
                for index, name in enumerate(ROOT_CHANNELS)
            },
        }
    else:
        result["root"] = {
            "available": False,
            "sample_count": 0,
            "missing_or_invalid": len(records),
            "unavailable_and_not_inferred": True,
        }
    return result


def _overlap(records_a, records_b) -> dict[str, Any]:
    prelift_phases = (
        "physical_hand_closure",
        "closed_hand_seating",
        "physical_grip_preload",
        "consolidation_ramp",
        "consolidation_window",
    )
    by_step_a = {
        r["global_step"]: r
        for r in records_a
        if isinstance(r.get("global_step"), int)
        and _segment_of(r) in prelift_phases
    }
    by_step_b = {
        r["global_step"]: r
        for r in records_b
        if isinstance(r.get("global_step"), int)
        and _segment_of(r) in prelift_phases
    }
    common_steps = sorted(set(by_step_a) & set(by_step_b))
    fields = {}
    extractors = [
        (
            "wrist_wrench_raw_sensor_frame",
            lambda r: r.get("wrist_wrench_raw_sensor_frame"),
        ),
        ("wrist_wrench_canonical", lambda r: r.get("wrist_wrench_canonical")),
        (
            "wrist_wrench_empty_baseline_compensated",
            lambda r: r.get("wrist_wrench_empty_baseline_compensated"),
        ),
        (
            "wrist_wrench_payload_reference",
            lambda r: r.get("wrist_wrench_payload_reference"),
        ),
        (
            "wrist_wrench_payload_reference_increment",
            lambda r: r.get("wrist_wrench_payload_reference_increment"),
        ),
        (
            "finger_root_torque_proxy_nm",
            lambda r: (
                [
                    r.get("finger_root_torque_proxy_nm", {}).get(c)
                    for c in ROOT_CHANNELS
                ]
                if isinstance(r.get("finger_root_torque_proxy_nm"), Mapping)
                else None
            ),
        ),
        ("finger_targets_rad", lambda r: r.get("finger_targets_rad")),
    ]
    for label, extract in extractors:
        seq_a = []
        seq_b = []
        for step in common_steps:
            value_a = extract(by_step_a[step])
            value_b = extract(by_step_b[step])
            if (
                isinstance(value_a, (list, tuple))
                and isinstance(value_b, (list, tuple))
                and len(value_a) == len(value_b)
                and all(
                    isinstance(v, (int, float))
                    and not isinstance(v, bool)
                    and math.isfinite(float(v))
                    for v in list(value_a) + list(value_b)
                )
            ):
                seq_a.append([float(v) for v in value_a])
                seq_b.append([float(v) for v in value_b])
        if not seq_a:
            fields[label] = {
                "identical": False,
                "available": False,
                "unavailable_and_not_inferred": True,
            }
            continue
        matrix_a = np.asarray(seq_a, dtype=np.float64)
        matrix_b = np.asarray(seq_b, dtype=np.float64)
        max_diff = float(np.max(np.abs(matrix_a - matrix_b)))
        fields[label] = {
            "identical": bool(max_diff == 0.0),
            "available": True,
            "max_abs_diff": max_diff,
            "sample_count": int(matrix_a.shape[0]),
            "channel_count": int(matrix_a.shape[1]),
            "sha256_a": hashlib.sha256(matrix_a.tobytes()).hexdigest(),
            "sha256_b": hashlib.sha256(matrix_b.tobytes()).hexdigest(),
        }
    return {
        "prelift_steps_a": len(by_step_a),
        "prelift_steps_b": len(by_step_b),
        "common_step_count": len(common_steps),
        "fields": fields,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episode", action="append", required=True,
        help="episode directory (repeatable)",
    )
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument(
        "--rate-hz", type=float, default=None,
        help=(
            "explicit physics rate in Hz for frequency conversion "
            "(a rate, not the step period)"
        ),
    )
    parser.add_argument("--compare-a", default=None)
    parser.add_argument("--compare-b", default=None)
    arguments = parser.parse_args(argv)

    if arguments.rate_hz is not None:
        if not math.isfinite(arguments.rate_hz) or arguments.rate_hz <= 0:
            parser.error("--rate-hz must be finite and positive")
    if (arguments.compare_a is None) != (arguments.compare_b is None):
        parser.error("--compare-a and --compare-b must be given together")
    episode_dirs = [Path(raw).resolve() for raw in arguments.episode]
    output_dir = Path(arguments.output).resolve()
    if output_dir in episode_dirs:
        parser.error("output directory must not be an episode directory")

    diagnostics = []
    problems = []
    for episode_dir in episode_dirs:
        try:
            records, load_problems = _load_records(episode_dir)
        except FileNotFoundError as error:
            problems.append(f"{episode_dir}: {error}")
            continue
        problems.extend(load_problems)
        segments: dict[str, list[Mapping[str, Any]]] = {}
        for record in records:
            segments.setdefault(_segment_of(record), []).append(record)
        diagnostics.append(
            {
                "directory": str(episode_dir),
                "segments": {
                    name: _segment_stats(rows, arguments.rate_hz)
                    for name, rows in segments.items()
                },
                "final_hold_note": (
                    "present"
                    if "unsupported_final_hold" in segments
                    else (
                        "absent_in_this_schema: the stage-3 hold is inside "
                        "physical_grip_lift_stage_3 records"
                    )
                ),
                "load_problems": list(load_problems),
            }
        )

    overlap = None
    if arguments.compare_a is not None:
        try:
            records_a, problems_a = _load_records(Path(arguments.compare_a))
            records_b, problems_b = _load_records(Path(arguments.compare_b))
            problems.extend(problems_a)
            problems.extend(problems_b)
            overlap = _overlap(records_a, records_b)
        except FileNotFoundError as error:
            problems.append(f"overlap: {error}")

    document = _clean(
        {
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "rate_hz": arguments.rate_hz,
            "frequency_note": (
                "Hz reported only with explicit --rate-hz; otherwise "
                "dominant_cycles_per_step only and dominant_hz unavailable"
            ),
            "read_only_note": (
                "read-only diagnostic; never alters any report or PASS"
            ),
            "nut_note": (
                "per-step nut pose/velocity is absent from controller_steps "
                "and reported unavailable_and_not_inferred, never fabricated"
            ),
            "episodes": diagnostics,
            "prelift_overlap": overlap,
            "problems": problems,
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "wrist_window_diagnostic.json"
    md_path = output_dir / "SUMMARY_CN.md"
    json_path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    lines = ["# 腕力窗口只读诊断（不影响 PASS）", ""]
    for entry in diagnostics:
        lines.append("## " + Path(entry["directory"]).name)
        for name, stats in entry["segments"].items():
            lines.append(
                "- %s: steps=%d; final hold=%s"
                % (name, stats["step_count"], entry["final_hold_note"])
            )
    if overlap:
        lines.append("## Pre-lift 重叠逐样本比对")
        lines.append(
            "公共步数=%d，A=%d，B=%d"
            % (
                overlap["common_step_count"],
                overlap["prelift_steps_a"],
                overlap["prelift_steps_b"],
            )
        )
        for label, field in overlap["fields"].items():
            lines.append(
                "- %s: available=%s identical=%s samples=%s max_abs_diff=%s"
                % (
                    label,
                    field.get("available"),
                    field.get("identical"),
                    field.get("sample_count"),
                    field.get("max_abs_diff"),
                )
            )
    if problems:
        lines.append("## 问题")
        for problem in problems:
            lines.append("- " + problem)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    try:
        reloaded = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(reloaded, dict):
            raise ValueError("diagnostic json not an object")
    except (ValueError, OSError) as error:
        print(f"SELF-CHECK FAILED: {error}")
        return 1
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
