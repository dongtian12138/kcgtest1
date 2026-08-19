#!/usr/bin/env python3
"""Offline paired summarizer for G4 synchronous/sequential pairs.

Pure read-only: consumes pair manifests produced by
d38999_g4_paired_batch.py (or equivalent), re-verifies every recorded
hash, enforces the fairness hard gates, and emits per-method statistics
plus paired deltas (sequential - synchronous) with explicit paired
sample counts.  The full sequential success contract remains the
responsibility of the accepted G4 summarizer; this tool only performs
pair integrity, fairness and generic statistics, and never recomputes
PASS from posthoc truth.
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

SCHEMA_VERSION = "kcg_g4_paired_summary_v1"
MANIFEST_SCHEMA_VERSION = "kcg_g4_paired_input_manifest_v1"

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
BATCH_RUNNER_PATH = (
    REPOSITORY_ROOT / "src/kcg_connector/isaac/d38999_g4_paired_batch.py"
)

SOURCE_FILES = {
    "runner_sha256": "src/kcg_connector/isaac/d38999_tabletop_pick_smoke.py",
    "wrapper_sha256": (
        "src/kcg_connector/isaac/d38999_tabletop_physical_grasp_v1.py"
    ),
    "three_finger_sequential_grasp_sha256": (
        "src/kcg_connector/kcg_connector/grasp/"
        "three_finger_sequential_grasp.py"
    ),
    "finger_contact_detector_sha256": (
        "src/kcg_connector/kcg_connector/grasp/finger_contact_detector.py"
    ),
    "grasp_stability_monitor_sha256": (
        "src/kcg_connector/kcg_connector/grasp/grasp_stability_monitor.py"
    ),
    "physical_grasp_config_loader_sha256": (
        "src/kcg_connector/kcg_connector/grasp/physical_grasp_config.py"
    ),
    "physical_grasp_config_sha256": (
        "src/kcg_connector/config/d38999_tabletop_physical_grasp_v1.yaml"
    ),
    "pick_config_sha256": (
        "src/kcg_connector/config/d38999_tabletop_pick_v1.yaml"
    ),
    "tabletop_scene_config_sha256": (
        "src/kcg_connector/config/d38999_tabletop_scene_v1.yaml"
    ),
}

GPU_CONSOLE_REQUIRED_MARKERS = (
    "NVIDIA GeForce RTX 5070 Ti",
    "Yes: 0",
    "\"cuda:0\"",
    "CUDA Toolkit",
)

GPU_CONSOLE_FORBIDDEN_MARKERS = (
    "Failed to create any GPU devices",
    "CPU fallback",
    "cpu_fallback",
    "Warp initialized on cpu",
    "warp CPU backend",
)

SIDE_METHODS = ("synchronous", "sequential-compliant")
FINGERS = ("f1", "f2", "f3")
RATE_HZ_DEFAULT = 240.0

DIRECT_FAIRNESS_FIELDS = [
    "provenance source/config SHA-256 fields",
    "provenance.payload_sha256",
    "realized_randomization.canonical_payload (excluding method)",
    "realized_arm_targets",
    "formal_lift_mode",
    "gui",
    "seed",
    "normalized_argv (excluding method/output-dir)",
]

INDIRECT_FAIRNESS_FIELDS = [
    "physics_dt",
    "physics_substeps",
    "solver_settings",
]


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"missing file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reject_nonfinite_json(token: str) -> None:
    raise ValueError(f"non-finite JSON constant {token}")


def _load_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_nonfinite_json,
    )


def current_source_hashes() -> dict[str, str]:
    return {
        key: _sha256_file(REPOSITORY_ROOT / relative)
        for key, relative in SOURCE_FILES.items()
    }


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
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


def _stats(values: Sequence[float]) -> dict[str, Any] | None:
    data = np.asarray([float(v) for v in values], dtype=np.float64)
    if data.size == 0 or not np.all(np.isfinite(data)):
        return None
    return {
        "count": int(data.size),
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "p95": float(np.percentile(data, 95)),
        "maximum": float(np.max(data)),
    }


def _load_side(manifest_side: Mapping[str, Any]):
    report_path = Path(manifest_side["report_file"])
    trace_path = Path(manifest_side["trace_file"])
    report = _load_json(report_path)
    if not isinstance(report, Mapping):
        raise ValueError("report is not a JSON object")
    records = []
    trace_problems = []
    previous = None
    with trace_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(
                    line, parse_constant=_reject_nonfinite_json
                )
            except (json.JSONDecodeError, ValueError) as error:
                trace_problems.append(f"line {line_number}: {error}")
                continue
            if not isinstance(record, Mapping):
                trace_problems.append(
                    f"line {line_number}: record not an object"
                )
                continue
            step = record.get("global_step")
            if isinstance(step, bool) or not isinstance(step, int):
                trace_problems.append(f"line {line_number}: step not int")
            elif previous is not None and step <= previous:
                trace_problems.append(
                    f"line {line_number}: step not strictly increasing"
                )
            if isinstance(step, int):
                previous = step
            records.append(record)
    if not records:
        trace_problems.append("trace is empty")
    return report, records, trace_problems


def _verify_manifest_content_hash(manifest: Mapping[str, Any]) -> bool:
    declared = manifest.get("manifest_content_sha256")
    if not isinstance(declared, str) or not declared:
        return False
    canonical = json.dumps(
        {
            "seed": manifest.get("seed"),
            "gui": manifest.get("gui"),
            "sides": manifest.get("sides"),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest() == declared


SIDE_FILE_KEYS = (
    ("report_sha256", "report_file"),
    ("trace_sha256", "trace_file"),
    ("side_console_sha256", "side_console"),
)


def verify_pair(
    manifest: Mapping[str, Any], disk_hashes: Mapping[str, str]
) -> tuple[list[str], dict[str, Any]]:
    problems: list[str] = []
    sides = {}
    if manifest.get("schema_version") != "kcg_g4_paired_batch_v1":
        problems.append("pair manifest schema mismatch")
    if not _verify_manifest_content_hash(manifest):
        problems.append("manifest content hash invalid")
    if manifest.get("source_hashes") != disk_hashes:
        problems.append("manifest source hashes differ from disk")
    batch_runner_sha256 = manifest.get("batch_runner_sha256")
    if batch_runner_sha256 != _sha256_file(BATCH_RUNNER_PATH):
        problems.append("batch runner hash missing or differs from disk")
    if manifest.get("gui") is not False:
        problems.append("formal paired summary requires gui=false")
    manifest_sides = manifest.get("sides")
    if not isinstance(manifest_sides, list):
        manifest_sides = []
        problems.append("manifest sides must be a list")
    listed_methods = [
        entry.get("method")
        for entry in manifest_sides
        if isinstance(entry, Mapping)
    ]
    if listed_methods != list(SIDE_METHODS):
        problems.append("manifest must contain exactly two ordered sides")
    for method in SIDE_METHODS:
        entry = next(
            (
                side
                for side in manifest_sides
                if isinstance(side, Mapping) and side.get("method") == method
            ),
            None,
        )
        if entry is None:
            problems.append(f"missing {method} side")
            sides[method] = {"available": False}
            continue
        for hash_key, file_key in SIDE_FILE_KEYS:
            path = Path(entry.get(file_key) or "")
            if not path.is_file():
                problems.append(f"{method} {file_key} file missing")
            elif _sha256_file(path) != entry.get(hash_key):
                problems.append(f"{method} {hash_key} hash mismatch")
        try:
            report, records, trace_problems = _load_side(entry)
        except (
            FileNotFoundError,
            json.JSONDecodeError,
            ValueError,
        ) as error:
            problems.append(f"{method}: {error}")
            sides[method] = {"available": False}
            continue
        problems.extend(f"{method} trace: {p}" for p in trace_problems)
        console = Path(entry["side_console"]).read_text(
            encoding="utf-8", errors="replace"
        )
        for marker in GPU_CONSOLE_REQUIRED_MARKERS:
            if marker not in console:
                problems.append(
                    f"{method} console missing GPU marker {marker!r}"
                )
        for marker in GPU_CONSOLE_FORBIDDEN_MARKERS:
            if marker in console:
                problems.append(
                    f"{method} console contains forbidden GPU marker "
                    f"{marker!r}"
                )
        if report.get("seed") != manifest.get("seed"):
            problems.append(f"{method} seed mismatch")
        if report.get("physical_grasp_method") != method:
            problems.append(f"{method} report method mismatch")
        if report.get("formal_lift_mode") != "staged":
            problems.append(f"{method} not staged")
        if report.get("gui") is not False:
            problems.append(f"{method} gui mismatch")
        if report.get("control_reads_object_truth") is not False:
            problems.append(f"{method} control reads object truth")
        if report.get("control_reads_contact_report") is not False:
            problems.append(f"{method} control reads contact report")
        if report.get("object_pose_writes_after_start") != 0:
            problems.append(f"{method} pose writes")
        if report.get("attachment") != "none":
            problems.append(f"{method} attachment")
        if report.get("posthoc_truth_evaluation_only") is not True:
            problems.append(f"{method} posthoc truth not evaluation-only")
        provenance = report.get("provenance") or {}
        for key, expected in disk_hashes.items():
            if provenance.get(key) != expected:
                problems.append(f"{method} {key} source mismatch")
        acceptance = report.get("formal_acceptance") or {}
        physical_success = bool(
            report.get("passed") is True
            and report.get("process_exit_code") == 0
            and isinstance(acceptance, Mapping)
            and acceptance.get("passed") is True
        )
        normalized = entry.get("normalized_argv")
        if not (
            isinstance(normalized, list)
            and normalized
            and all(isinstance(item, str) for item in normalized)
        ):
            problems.append(f"{method} normalized argv invalid")
        sides[method] = {
            "available": True,
            "report": report,
            "records": records,
            "manifest_side": entry,
            "physical_success": physical_success,
        }
    if all(sides.get(m, {}).get("available") for m in SIDE_METHODS):
        sync = sides["synchronous"]["report"]
        seq = sides["sequential-compliant"]["report"]
        if sync["provenance"].get("payload_sha256") != seq["provenance"].get(
            "payload_sha256"
        ):
            problems.append("payload_sha256 mismatch across sides")
        sync_canonical = dict(
            (sync.get("realized_randomization") or {}).get(
                "canonical_payload", {}
            )
        )
        seq_canonical = dict(
            (seq.get("realized_randomization") or {}).get(
                "canonical_payload", {}
            )
        )
        sync_canonical.pop("method", None)
        seq_canonical.pop("method", None)
        if sync_canonical != seq_canonical:
            problems.append("canonical_payload differs beyond method")
        if (sync.get("realized_arm_targets") or {}) != (
            seq.get("realized_arm_targets") or {}
        ):
            problems.append("realized_arm_targets differ")
        sync_argv = tuple(
            sides["synchronous"]["manifest_side"].get("normalized_argv") or []
        )
        seq_argv = tuple(
            sides["sequential-compliant"]["manifest_side"].get(
                "normalized_argv"
            )
            or []
        )
        if not sync_argv or not seq_argv:
            problems.append("normalized argv unavailable")
            sides["_argv_comparison"] = "invalid"
        elif sync_argv != seq_argv:
            problems.append("normalized argv differs across sides")
    return problems, sides


def _side_metrics(side: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    if not side.get("available"):
        return {}, []
    report = side["report"]
    unavailable: list[str] = []
    metrics: dict[str, Any] = {}
    tst = report.get("table_stage") or {}
    pose = report.get("posthoc_pose_error") or {}
    slip = report.get("posthoc_lift_relative_slip") or {}
    monitor = report.get("formal_lift_monitor") or {}
    grasp = report.get("grasp_controller") or {}
    if _finite(tst.get("translation_xy_m")):
        metrics["table_xy_mm"] = float(tst["translation_xy_m"]) * 1000.0
    else:
        unavailable.append("table_xy_mm")
    if _finite(tst.get("yaw_delta_rad")):
        metrics["abs_table_yaw_deg"] = (
            abs(float(tst["yaw_delta_rad"])) * 180.0 / math.pi
        )
    else:
        unavailable.append("abs_table_yaw_deg")
    for channel in ("dx_m", "dy_m", "dz_m"):
        if _finite(pose.get(channel)):
            metrics[f"posthoc_abs_{channel}_mm"] = (
                abs(float(pose[channel])) * 1000.0
            )
        else:
            unavailable.append(f"posthoc_abs_{channel}_mm")
    for channel in ("drx_rad", "dry_rad", "drz_rad"):
        if _finite(pose.get(channel)):
            metrics[f"posthoc_abs_{channel}_deg"] = (
                abs(float(pose[channel])) * 180.0 / math.pi
            )
        else:
            unavailable.append(f"posthoc_abs_{channel}_deg")
    xyz = ("dx_m", "dy_m", "dz_m")
    rpy = ("drx_rad", "dry_rad", "drz_rad")
    if all(_finite(pose.get(c)) for c in xyz):
        metrics["posthoc_translation_norm_mm"] = (
            math.sqrt(sum(float(pose[c]) ** 2 for c in xyz)) * 1000.0
        )
    else:
        unavailable.append("posthoc_translation_norm_mm")
    if all(_finite(pose.get(c)) for c in rpy):
        metrics["posthoc_rotation_norm_deg"] = (
            math.sqrt(sum(float(pose[c]) ** 2 for c in rpy)) * 180.0 / math.pi
        )
    else:
        unavailable.append("posthoc_rotation_norm_deg")
    if all(_finite(slip.get(c)) for c in xyz):
        metrics["lift_slip_translation_norm_mm"] = (
            math.sqrt(sum(float(slip[c]) ** 2 for c in xyz)) * 1000.0
        )
    else:
        unavailable.append("lift_slip_translation_norm_mm")
    if all(_finite(slip.get(c)) for c in rpy):
        metrics["lift_slip_rotation_norm_deg"] = (
            math.sqrt(sum(float(slip[c]) ** 2 for c in rpy)) * 180.0 / math.pi
        )
    else:
        unavailable.append("lift_slip_rotation_norm_deg")
    for key, name in (
        ("peak_wrist_force_increment_n", "wrist_force_peak_n"),
        ("peak_moment_safety_score_nm", "moment_score_peak_nm"),
    ):
        if _finite(monitor.get(key)):
            metrics[name] = float(monitor[key])
        else:
            unavailable.append(name)
    pre_lift = report.get("pre_lift_grasp_controller_evidence") or {}
    summary = pre_lift.get("sequential_final_summary") or {}
    if _finite(summary.get("normalized_load_imbalance")):
        metrics["normalized_load_imbalance"] = float(
            summary["normalized_load_imbalance"]
        )
    else:
        unavailable.append("normalized_load_imbalance")
    phase_steps = report.get("phase_steps") or {}
    if isinstance(phase_steps, Mapping) and phase_steps:
        total_steps = sum(
            int(v)
            for v in phase_steps.values()
            if isinstance(v, int) and not isinstance(v, bool)
        )
        metrics["duration_s"] = total_steps / RATE_HZ_DEFAULT
    else:
        unavailable.append("duration_s")
    contact_steps = grasp.get("contact_global_steps") or {}
    if set(contact_steps) == set(FINGERS):
        for finger in FINGERS:
            metrics[f"contact_step_{finger}"] = int(contact_steps[finger])
        ordered = grasp.get("contact_order") or []
        metrics["contact_order"] = "-".join(ordered) if ordered else None
    else:
        unavailable.append("contact_timing")
    return metrics, unavailable


def _failure_reason(report: Mapping[str, Any]) -> str | None:
    if not report:
        return None
    monitor = report.get("formal_lift_monitor") or {}
    recovery = report.get("formal_recovery") or {}
    lift_failure = report.get("formal_lift_failure") or {}
    for source in (
        monitor.get("failure_reason"),
        recovery.get("original_failure_reason"),
        lift_failure.get("reason"),
    ):
        if isinstance(source, str) and source.strip():
            return source
    error = report.get("error")
    if isinstance(error, str) and error.strip():
        return error
    return None


def _contact_order_distribution(pairs, method):
    distribution: dict[str, int] = {}
    for pair in pairs:
        metrics = pair["sides"].get(method) or {}
        order = metrics.get("contact_order")
        if isinstance(order, str) and order:
            distribution[order] = distribution.get(order, 0) + 1
    return distribution


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair-dir", action="append", default=[],
        help="pair directory (repeatable)",
    )
    parser.add_argument(
        "--batch-dir", default=None,
        help="batch directory containing seedXXX/pair_manifest.json",
    )
    parser.add_argument(
        "--require-complete-pairs", action="store_true",
        help="exit 1 when any pair is incomplete",
    )
    parser.add_argument(
        "--require-all-pass", action="store_true",
        help="exit 2 when structure is valid but a side physically failed",
    )
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args(argv)

    pair_dirs = [Path(raw).resolve() for raw in arguments.pair_dir]
    if arguments.batch_dir:
        batch = Path(arguments.batch_dir).resolve()
        if not batch.is_dir():
            parser.error(f"batch dir missing: {batch}")
        for manifest in sorted(batch.glob("seed*/pair_manifest.json")):
            pair_dirs.append(manifest.parent)
    if not pair_dirs:
        parser.error("no pair directories")
    output_dir = Path(arguments.output).resolve()
    if output_dir in pair_dirs:
        parser.error("output must not be a pair directory")
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error("output directory already exists and is not empty")

    disk_hashes = current_source_hashes()
    cli_sha256 = _sha256_file(Path(__file__).resolve())
    pairs = []
    structural_problems_total = 0
    for pair_dir in pair_dirs:
        manifest_path = pair_dir / "pair_manifest.json"
        manifest = None
        problems: list[str] = []
        sides = {}
        try:
            manifest = _load_json(manifest_path)
            problems, sides = verify_pair(manifest, disk_hashes)
        except (
            FileNotFoundError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            problems = [f"{type(error).__name__}: {error}"]
        seed = manifest.get("seed") if isinstance(manifest, Mapping) else None
        pairs.append(
            {
                "directory": str(pair_dir),
                "seed": seed,
                "manifest": manifest if isinstance(manifest, Mapping) else {},
                "structural_ok": not problems,
                "problems": problems,
                "argv_comparison": sides.pop("_argv_comparison", "verified"),
                "sides": {
                    m: _side_metrics(sides.get(m, {}))[0]
                    for m in SIDE_METHODS
                },
                "unavailable": {
                    m: _side_metrics(sides.get(m, {}))[1]
                    for m in SIDE_METHODS
                },
                "physical_success": {
                    m: bool(
                        (sides.get(m) or {}).get("physical_success", False)
                    )
                    for m in SIDE_METHODS
                },
                "failure_reasons": {
                    m: _failure_reason(
                        (sides.get(m) or {}).get("report", {})
                    )
                    for m in SIDE_METHODS
                },
            }
        )
        structural_problems_total += len(problems)

    seeds = [p["seed"] for p in pairs]
    if len(set(seeds)) != len(seeds):
        structural_problems_total += 1

    valid = [p for p in pairs if p["structural_ok"]]
    method_stats = {}
    for method in SIDE_METHODS:
        rows = [p["sides"][method] for p in valid if p["sides"].get(method)]
        metric_names = sorted(
            {key for row in rows for key in row if row.get(key) is not None}
        )
        blocks = {}
        for name in metric_names:
            values = [row[name] for row in rows if _finite(row.get(name))]
            stats = _stats(values)
            if stats:
                stats["available"] = True
                stats["sample_count"] = len(values)
            else:
                stats = {"available": False, "sample_count": 0}
            blocks[name] = stats
        success_count = sum(1 for p in valid if p["physical_success"][method])
        failure_distribution: dict[str, int] = {}
        for p in valid:
            if not p["physical_success"][method]:
                reason = p["failure_reasons"][method] or "unknown"
                failure_distribution[reason] = (
                    failure_distribution.get(reason, 0) + 1
                )
        method_stats[method] = {
            "pair_count": len(valid),
            "success_count": success_count,
            "drop_count": sum(
                count
                for reason, count in failure_distribution.items()
                if "load_lost" in reason
            ),
            "failure_reason_distribution": failure_distribution,
            "metrics": blocks,
            "contact_order_distribution": _contact_order_distribution(
                valid, method
            ),
        }

    paired_deltas = {}
    metric_names = sorted(
        set(method_stats["synchronous"]["metrics"])
        | set(method_stats["sequential-compliant"]["metrics"])
    )
    for name in metric_names:
        deltas = []
        for p in valid:
            seq_value = p["sides"]["sequential-compliant"].get(name)
            sync_value = p["sides"]["synchronous"].get(name)
            if _finite(seq_value) and _finite(sync_value):
                deltas.append(float(seq_value) - float(sync_value))
        stats = _stats(deltas)
        paired_deltas[name] = {
            "available": stats is not None,
            "paired_sample_count": len(deltas),
            "stats": stats,
            "semantics": "sequential - synchronous",
        }

    summary = _clean(
        {
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "generator_source_sha256": cli_sha256,
            "structural_valid": structural_problems_total == 0,
            "all_sides_physical_pass": bool(
                valid
                and all(
                    p["physical_success"][m]
                    for p in valid
                    for m in SIDE_METHODS
                )
            ),
            "direct_verified_fields": DIRECT_FAIRNESS_FIELDS,
            "indirect_same_source_config_fields": {
                field: (
                    "indirect: fixed by identical source/config hashes and "
                    "normalized argv; not per-field verified in reports"
                )
                for field in INDIRECT_FAIRNESS_FIELDS
            },
            "p95_method": "numpy_linear_default_np_percentile_95",
            "sample_size_note": (
                "n>=30 satisfies the sample threshold only; it does not "
                "claim G5/G6 or superiority"
            ),
            "regrasp": "unavailable_and_not_inferred",
            "posthoc_note": (
                "posthoc pose/table/nut data are evaluation-only and never "
                "alter PASS or commands"
            ),
            "pairs": [
                {
                    "directory": p["directory"],
                    "seed": p["seed"],
                    "structural_ok": p["structural_ok"],
                    "problems": p["problems"],
                    "argv_comparison": p["argv_comparison"],
                    "sides": p["sides"],
                    "unavailable": p["unavailable"],
                    "physical_success": p["physical_success"],
                    "failure_reasons": p["failure_reasons"],
                }
                for p in pairs
            ],
            "method_statistics": method_stats,
            "paired_deltas": paired_deltas,
            "preliminary": len(valid) < 30,
        }
    )

    manifest_inputs = []
    for pair in pairs:
        manifest_path = Path(pair["directory"]) / "pair_manifest.json"
        if manifest_path.is_file():
            manifest_inputs.append(
                {
                    "role": "pair_manifest",
                    "path": str(manifest_path),
                    "sha256": _sha256_file(manifest_path),
                }
            )
        side_entries = pair.get("manifest", {}).get("sides", []) or []
        if not isinstance(side_entries, list):
            side_entries = []
        for method in SIDE_METHODS:
            entry = next(
                (
                    s
                    for s in side_entries
                    if isinstance(s, Mapping) and s.get("method") == method
                ),
                None,
            )
            if not entry:
                continue
            for hash_key, file_key in SIDE_FILE_KEYS:
                raw_path = entry.get(file_key)
                if not isinstance(raw_path, str):
                    continue
                path = Path(raw_path)
                if not path.is_file():
                    continue
                manifest_inputs.append(
                    {
                        "role": f"{method}_{file_key}",
                        "path": str(path),
                        "sha256": _sha256_file(path),
                    }
                )
    for key, relative in SOURCE_FILES.items():
        manifest_inputs.append(
            {
                "role": "source_file",
                "key": key,
                "path": str(REPOSITORY_ROOT / relative),
                "sha256": disk_hashes[key],
            }
        )
    manifest_inputs.append(
        {
            "role": "paired_batch_runner_source",
            "path": str(BATCH_RUNNER_PATH),
            "sha256": _sha256_file(BATCH_RUNNER_PATH),
        }
    )
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generator_source_sha256": cli_sha256,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": manifest_inputs,
    }
    canonical = json.dumps(
        {"inputs": manifest["inputs"]}, sort_keys=True, ensure_ascii=False
    )
    manifest_sha256 = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    manifest["manifest_content_sha256"] = manifest_sha256
    summary["input_manifest_sha256"] = manifest_sha256

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    lines = ["# G4 成对汇总（只读，不影响 PASS）", ""]
    lines.append("结构有效：%s" % summary["structural_valid"])
    for method in SIDE_METHODS:
        stats = method_stats[method]
        lines.append(
            "%s：成功 %d/%d，drop=%d"
            % (
                method,
                stats["success_count"],
                stats["pair_count"],
                stats["drop_count"],
            )
        )
    lines.append("")
    lines.append("paired delta 语义：sequential - synchronous")
    lines.append("regrasp：unavailable_and_not_inferred")
    lines.append("posthoc 数据仅评价，绝不重算 PASS。")
    (output_dir / "SUMMARY_CN.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    try:
        reloaded = json.loads(
            (output_dir / "summary.json").read_text(encoding="utf-8")
        )
        if reloaded.get("input_manifest_sha256") != manifest_sha256:
            raise ValueError("summary manifest hash mismatch")
    except (ValueError, OSError) as error:
        print(f"SELF-CHECK FAILED: {error}")
        return 1

    if structural_problems_total != 0:
        return 1
    incomplete = any(not p["structural_ok"] for p in pairs)
    if arguments.require_complete_pairs and incomplete:
        return 1
    if arguments.require_all_pass and not summary["all_sides_physical_pass"]:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
