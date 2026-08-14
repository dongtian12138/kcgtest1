#!/usr/bin/env python3

"""Aggregate three synchronized four-view tooth captures, fail closed.

This module intentionally does not start Isaac Sim.  It re-validates the
hash-bound PNG capture bundles with :mod:`d38999_tooth_sync_analysis`, measures
every usable fixed view (not only the priority view used by the primary
analyzer), and separates two very different conclusions:

* the 240 Hz physics trace can exclude independent motion of every authored
  tooth above the diagnostic transform thresholds; and
* RGB evidence can only exclude an *unknown* visibly jittering tooth when all
  24 tooth identities are measurable at every sampled phase-local transition.

Even complete visual coverage does not authorize a "no render jitter" claim
without a residual acceptance threshold registered before inspecting the
captures.  The output therefore remains ``VALID_LIMITED`` and records that
limitation explicitly instead of turning successful parsing into a visual
stability result.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from kcg_connector import d38999_tooth_sync_analysis as analysis


SCHEMA_VERSION = "kcg_d38999_tooth_sync_evidence_v1"
MANIFEST_SCHEMA_VERSION = "kcg_d38999_tooth_sync_evidence_manifest_v1"
RUN_IDS = ("baseline", "rtx_history_512", "segment00_normalized")
COMPARISON_IDS = (
    "baseline_vs_rtx_history_512",
    "baseline_vs_segment00_normalized",
)
EXPECTED_SEGMENTS = tuple(f"Segment_{index:02d}" for index in range(24))
EXPECTED_SEGMENT_SET = frozenset(EXPECTED_SEGMENTS)


class EvidenceError(RuntimeError):
    """Raised when the three-run evidence cannot be aggregated safely."""


def sha256_file(path: str | Path) -> str:
    """Hash an evidence file without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_file(path: str | Path, repository: Path, label: str) -> Path:
    target = Path(path).expanduser().resolve()
    repository = Path(repository).expanduser().resolve()
    if not target.is_file():
        raise EvidenceError(f"{label} is missing: {target}")
    if repository != target and repository not in target.parents:
        raise EvidenceError(f"{label} escapes repository: {target}")
    return target


def _repo_directory(path: str | Path, repository: Path, label: str) -> Path:
    target = Path(path).expanduser().resolve()
    repository = Path(repository).expanduser().resolve()
    if not target.is_dir():
        raise EvidenceError(f"{label} is missing: {target}")
    if repository != target and repository not in target.parents:
        raise EvidenceError(f"{label} escapes repository: {target}")
    return target


def file_binding(path: str | Path, repository: str | Path) -> dict[str, Any]:
    """Return a repository-relative, size-and-SHA binding."""

    repository = Path(repository).expanduser().resolve()
    target = _repo_file(path, repository, "bound artifact")
    return {
        "path": str(target.relative_to(repository)),
        "sha256": sha256_file(target),
        "size_bytes": target.stat().st_size,
    }


def _finite_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise EvidenceError(f"{label} must be finite and non-negative")
    return result


def validate_physics_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Validate all 24 physical tooth transforms and diagnostic thresholds."""

    if report.get("schema_version") != analysis.PHYSICS_SCHEMA_VERSION:
        raise EvidenceError("physics report schema mismatch")
    steps = report.get("steps")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise EvidenceError("physics report steps must be a positive integer")
    if report.get("anomaly_steps") != 0:
        raise EvidenceError("physics report contains tooth anomaly steps")
    thresholds = report.get("thresholds")
    if not isinstance(thresholds, Mapping):
        raise EvidenceError("physics report thresholds are missing")
    translation_limit = _finite_nonnegative(
        thresholds.get("translation_m"), "translation threshold"
    )
    rotation_limit = _finite_nonnegative(
        thresholds.get("rotation_rad"), "rotation threshold"
    )
    if translation_limit != 1.0e-6 or rotation_limit != 1.0e-5:
        raise EvidenceError("physics diagnostic thresholds changed")
    segments = report.get("segment_aggregate")
    if not isinstance(segments, Mapping) or set(segments) != EXPECTED_SEGMENT_SET:
        raise EvidenceError("physics report must contain exactly 24 segments")
    maxima = {
        "maximum_local_translation_error_m": 0.0,
        "maximum_parent_relative_translation_error_m": 0.0,
        "maximum_local_rotation_error_rad": 0.0,
        "maximum_parent_relative_rotation_error_rad": 0.0,
    }
    for segment_name in EXPECTED_SEGMENTS:
        segment = segments[segment_name]
        if not isinstance(segment, Mapping):
            raise EvidenceError(f"{segment_name} aggregate is not a mapping")
        for key, limit in (
            ("maximum_local_translation_error_m", translation_limit),
            (
                "maximum_parent_relative_translation_error_m",
                translation_limit,
            ),
            ("maximum_local_rotation_error_rad", rotation_limit),
            ("maximum_parent_relative_rotation_error_rad", rotation_limit),
        ):
            value = _finite_nonnegative(
                segment.get(key), f"{segment_name}.{key}"
            )
            if value > limit:
                raise EvidenceError(f"{segment_name}.{key} exceeds threshold")
            maxima[key] = max(maxima[key], value)
    return {
        "all_24_segments_tracked": True,
        "anomaly_steps": 0,
        "maximum_errors": maxima,
        "relative_motion_below_diagnostic_threshold": True,
        "steps": steps,
        "thresholds": {
            "rotation_rad": rotation_limit,
            "translation_m": translation_limit,
        },
    }


def validate_treatment(run_id: str, bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Require the exact baseline, renderer-history and schema A/B treatments."""

    report = bundle["physics_report"]
    render = bundle["manifest"].get("render_settings", {})
    normalization = report.get("normalization_ab", {})
    schema = report.get("segment00_schema", {})
    if run_id == "baseline":
        expected_mode = "baseline"
        if render.get("requested") != {}:
            raise EvidenceError("baseline requested render settings changed")
        if normalization != {
            "authored_in_session_layer": False,
            "changed_missing_rotate_z": False,
        }:
            raise EvidenceError("baseline unexpectedly normalizes Segment_00")
        if schema.get("schema_outlier") is not True:
            raise EvidenceError("baseline Segment_00 schema is not the outlier")
    elif run_id == "rtx_history_512":
        expected_mode = "rtx_history_512"
        if render.get("requested") != {
            "/rtx/scenedb/maxHistoryTransformCount": 512
        }:
            raise EvidenceError("RTX history treatment differs from 512")
        if normalization != {
            "authored_in_session_layer": False,
            "changed_missing_rotate_z": False,
        }:
            raise EvidenceError("RTX history run unexpectedly normalizes Segment_00")
        if schema.get("schema_outlier") is not True:
            raise EvidenceError("RTX history Segment_00 schema is not the outlier")
    elif run_id == "segment00_normalized":
        expected_mode = "baseline"
        if render.get("requested") != {}:
            raise EvidenceError("normalized run changed renderer settings")
        if (
            normalization.get("authored_in_session_layer") is not True
            or normalization.get("changed_missing_rotate_z") is not True
            or normalization.get("rotate_z_degrees") != 0
            or schema.get("schema_outlier") is not False
            or schema.get("explicit_rotate_z") is not True
            or schema.get("explicit_rotate_z_degrees") != 0
        ):
            raise EvidenceError("Segment_00 normalization treatment is invalid")
    else:
        raise EvidenceError(f"unexpected run ID: {run_id}")
    if (
        render.get("mode") != expected_mode
        or render.get("exact_match") is not True
        or render.get("validated_after_simulation_app_start") is not True
        or render.get("mismatches") != []
    ):
        raise EvidenceError(f"{run_id} render setting readback failed")
    return {
        "normalization_ab": normalization,
        "render_mode": render["mode"],
        "render_settings_exactly_read_back": True,
        "segment00_schema": schema,
    }


def _frame_key(frame: Mapping[str, Any]) -> tuple[int, str, int]:
    return (
        int(frame["global_step"]),
        str(frame["phase"]),
        int(frame["phase_step"]),
    )


def _phase_local_transitions(frames):
    previous = None
    for current in frames:
        if previous is not None and previous["phase"] == current["phase"]:
            yield previous, current
        previous = current


def _coverage_summary(transition_records) -> dict[str, Any]:
    records = list(transition_records)
    if not records:
        raise EvidenceError("visual evidence contains no phase-local transitions")
    phase_records = defaultdict(list)
    counts = Counter()
    overall_union = set()
    for record in records:
        observed = set(record["segments"])
        overall_union.update(observed)
        counts.update(observed)
        phase_records[record["phase"]].append(record)

    def one_scope(items):
        union = set().union(*(set(item["segments"]) for item in items))
        minimum = min(len(item["segments"]) for item in items)
        return {
            "complete_24_segment_transitions": sum(
                set(item["segments"]) == EXPECTED_SEGMENT_SET for item in items
            ),
            "identity_union_all_24": union == EXPECTED_SEGMENT_SET,
            "minimum_segments_per_transition": minimum,
            "missing_from_identity_union": sorted(EXPECTED_SEGMENT_SET - union),
            "segments_in_identity_union": sorted(union),
            "transitions": len(items),
        }

    per_phase = {
        phase: one_scope(items) for phase, items in sorted(phase_records.items())
    }
    every_transition = all(
        set(record["segments"]) == EXPECTED_SEGMENT_SET for record in records
    )
    return {
        **one_scope(records),
        "every_transition_all_24": every_transition,
        "per_phase": per_phase,
        "per_segment_transition_counts": {
            segment: counts[segment] for segment in EXPECTED_SEGMENTS
        },
    }


def collect_all_view_run_evidence(
    run_id: str, result: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Measure every view and report true 24-ID temporal coverage."""

    rows = []
    transitions = []
    for previous, current in _phase_local_transitions(result["frames"]):
        observed_union = set()
        usable_views = []
        for view_id in analysis.VIEW_IDS:
            common = analysis._common_transition_segments(  # noqa: SLF001
                previous, current, view_id
            )
            if len(common) < analysis.MINIMUM_VISIBLE_TEETH:
                continue
            usable_views.append(view_id)
            observed_union.update(common)
            measured = analysis._measure_transition(  # noqa: SLF001
                previous, current, view_id, common
            )
            rows.extend({"run_id": run_id, **row} for row in measured)
        if not usable_views:
            raise EvidenceError(
                f"{run_id} has no usable view at global_step="
                f"{current['global_step']}"
            )
        transitions.append(
            {
                "global_step": current["global_step"],
                "phase": current["phase"],
                "phase_step": current["phase_step"],
                "segments": sorted(observed_union),
                "usable_views": usable_views,
            }
        )
    return rows, _coverage_summary(transitions)


def collect_all_view_ab_evidence(
    comparison_id: str,
    first_result: Mapping[str, Any],
    second_result: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Measure aligned A/B residuals in every common usable camera view."""

    first_frames = first_result["frames"]
    second_frames = second_result["frames"]
    if [_frame_key(frame) for frame in first_frames] != [
        _frame_key(frame) for frame in second_frames
    ]:
        raise EvidenceError(f"{comparison_id} frame keys differ")
    rows = []
    transitions = []
    for index in range(1, len(first_frames)):
        first_previous = first_frames[index - 1]
        first_current = first_frames[index]
        second_previous = second_frames[index - 1]
        second_current = second_frames[index]
        if first_previous["phase"] != first_current["phase"]:
            continue
        observed_union = set()
        usable_views = []
        for view_id in analysis.VIEW_IDS:
            views = (
                first_previous["views"],
                first_current["views"],
                second_previous["views"],
                second_current["views"],
            )
            if not all(view_id in item for item in views):
                continue
            common = sorted(
                set.intersection(
                    *(set(item[view_id]["points"]) for item in views)
                )
            )
            if len(common) < analysis.MINIMUM_VISIBLE_TEETH:
                continue
            usable_views.append(view_id)
            observed_union.update(common)
            first_rows = analysis._measure_transition(  # noqa: SLF001
                first_previous, first_current, view_id, common
            )
            second_rows = analysis._measure_transition(  # noqa: SLF001
                second_previous, second_current, view_id, common
            )
            for left, right in zip(first_rows, second_rows):
                if left["segment"] != right["segment"]:
                    raise EvidenceError("A/B segment row order differs")
                rows.append(
                    {
                        "comparison_id": comparison_id,
                        "global_step": left["global_step"],
                        "phase": left["phase"],
                        "phase_step": left["phase_step"],
                        "segment": left["segment"],
                        "view_id": view_id,
                        "first_residual_pitch_fraction": left[
                            "residual_pitch_fraction"
                        ],
                        "second_residual_pitch_fraction": right[
                            "residual_pitch_fraction"
                        ],
                        "second_minus_first_pitch_fraction": right[
                            "residual_pitch_fraction"
                        ]
                        - left["residual_pitch_fraction"],
                    }
                )
        if not usable_views:
            raise EvidenceError(
                f"{comparison_id} has no common usable view at "
                f"global_step={first_current['global_step']}"
            )
        transitions.append(
            {
                "global_step": first_current["global_step"],
                "phase": first_current["phase"],
                "phase_step": first_current["phase_step"],
                "segments": sorted(observed_union),
                "usable_views": usable_views,
            }
        )
    return rows, _coverage_summary(transitions)


def _distribution(values) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=float)
    if not len(array) or not np.all(np.isfinite(array)):
        raise EvidenceError("residual statistics are empty or non-finite")
    return {
        "maximum": float(np.max(array)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "samples": int(len(array)),
    }


def _write_csv(path: Path, rows) -> None:
    rows = list(rows)
    if not rows:
        raise EvidenceError(f"cannot write empty evidence CSV: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_three_run_evidence(
    *,
    repository: str | Path,
    captures: Mapping[str, str | Path],
    capture_helper: str | Path,
    runner_source: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Re-validate, aggregate and hash-bind one formal three-run artifact."""

    repository = Path(repository).expanduser().resolve()
    if set(captures) != set(RUN_IDS):
        raise EvidenceError(f"captures must contain exactly {RUN_IDS!r}")
    helper = _repo_file(capture_helper, repository, "capture helper")
    runner = _repo_file(runner_source, repository, "regrasp runner")
    analyzer_source = _repo_file(analysis.__file__, repository, "analyzer source")
    output = Path(output_directory).expanduser().resolve()
    if output.exists():
        raise EvidenceError(f"output directory already exists: {output}")
    if repository != output and repository not in output.parents:
        raise EvidenceError("output directory escapes repository")

    bundles = {}
    results = {}
    physics = {}
    treatments = {}
    capture_directories = {}
    run_rows = []
    run_coverage = {}
    for run_id in RUN_IDS:
        capture = _repo_directory(
            captures[run_id], repository, f"{run_id} capture"
        )
        bundle = analysis.validate_capture_bundle(capture, helper)
        result = analysis.analyze_validated_capture(bundle)
        bundles[run_id] = bundle
        results[run_id] = result
        physics[run_id] = validate_physics_report(bundle["physics_report"])
        treatments[run_id] = validate_treatment(run_id, bundle)
        capture_directories[run_id] = capture
        rows, coverage = collect_all_view_run_evidence(run_id, result)
        run_rows.extend(rows)
        run_coverage[run_id] = coverage

    trace_hashes = {
        bundle["physics_trace_sha256"] for bundle in bundles.values()
    }
    if len(trace_hashes) != 1:
        raise EvidenceError("three runs do not have identical physics traces")
    baseline_bundle = bundles["baseline"]
    baseline_result = results["baseline"]
    comparisons = (
        (
            "baseline_vs_rtx_history_512",
            "rtx_history_512",
        ),
        (
            "baseline_vs_segment00_normalized",
            "segment00_normalized",
        ),
    )
    ab_rows = []
    ab_coverage = {}
    priority_comparison_rows = {}
    for comparison_id, second_id in comparisons:
        # This call preserves the analyzer's no-residual-shopping, same-view
        # authorization gate.  The all-view rows below are additional coverage
        # diagnostics, not a replacement for that primary comparison.
        authorized_rows = analysis.compare_aligned_runs(
            baseline_bundle,
            baseline_result,
            bundles[second_id],
            results[second_id],
        )
        priority_comparison_rows[comparison_id] = len(authorized_rows)
        rows, coverage = collect_all_view_ab_evidence(
            comparison_id, baseline_result, results[second_id]
        )
        ab_rows.extend(rows)
        ab_coverage[comparison_id] = coverage

    strict_visual_coverage = bool(
        all(item["every_transition_all_24"] for item in run_coverage.values())
        and all(item["every_transition_all_24"] for item in ab_coverage.values())
    )
    claim_blockers = [
        "no_visual_residual_acceptance_threshold_was_preregistered",
        "30_hz_sampling_cannot_exclude_between_sample_render_artifacts",
    ]
    if not strict_visual_coverage:
        claim_blockers.insert(
            0,
            "not_all_24_segment_ids_are_measurable_at_every_sampled_transition",
        )
    report = {
        "schema_version": SCHEMA_VERSION,
        "classification": "VALID_LIMITED_VISUAL_JITTER_UNRESOLVED",
        "evidence_valid": True,
        "physics": {
            "all_three_runs_exclude_independent_tooth_motion_above_diagnostic_threshold": True,
            "identical_physics_trace_sha256": next(iter(trace_hashes)),
            "runs": physics,
        },
        "treatments": treatments,
        "visual": {
            "all_three_capture_bundles_hash_and_frame_validated": True,
            "all_three_single_run_analyses_authorized": True,
            "both_priority_view_ab_comparisons_authorized": True,
            "priority_view_ab_row_counts": priority_comparison_rows,
            "run_all_view_coverage": run_coverage,
            "ab_all_view_coverage": ab_coverage,
            "strict_unknown_single_tooth_temporal_coverage_passed": strict_visual_coverage,
            "render_jitter_absence_claim_authorized": False,
            "claim_blockers": claim_blockers,
            "run_residual_pitch_fraction": _distribution(
                row["residual_pitch_fraction"] for row in run_rows
            ),
            "ab_absolute_delta_pitch_fraction": _distribution(
                abs(row["second_minus_first_pitch_fraction"])
                for row in ab_rows
            ),
        },
        "scope": {
            "capture_rate_hz": 30,
            "fixed_views": list(analysis.VIEW_IDS),
            "physics_rate_hz": 240,
            "phases": list(analysis.CAPTURE_PHASES),
            "tooth_ids": list(EXPECTED_SEGMENTS),
        },
        "limitations": [
            "valid_analysis_is_not_a_no_jitter_threshold_pass",
            "visual_coverage_union_is_reported_separately_from_every_transition_coverage",
            "runner_source_is_a_bound_snapshot_but_not_attested_by_the_capture_manifest",
            "physical_transform_thresholds_are_diagnostic_not_contact_safety_limits",
            "this_prepared_twist_probe_does_not_validate_full_end_to_end_assembly",
        ],
    }

    output.mkdir(parents=True, exist_ok=False)
    run_csv = output / "all_view_per_tooth_residuals.csv"
    ab_csv = output / "all_view_ab_residuals.csv"
    report_path = output / "report.json"
    _write_csv(run_csv, run_rows)
    _write_csv(ab_csv, ab_rows)
    report_path.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    capture_bindings = {}
    for run_id, bundle in bundles.items():
        capture = capture_directories[run_id]
        capture_bindings[run_id] = {
            "capture_directory": str(capture.relative_to(repository)),
            "capture_manifest": file_binding(
                capture / "video_capture_manifest.json", repository
            ),
            "physics_report": file_binding(
                bundle["physics_report_path"], repository
            ),
            "physics_summary": file_binding(
                bundle["physics_summary_path"], repository
            ),
            "sync_csv": file_binding(
                capture / "video_frame_sync.csv", repository
            ),
        }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "HASH_SIZE_SCHEMA_BOUND",
        "capture_bundles": capture_bindings,
        "indirect_frame_binding": {
            "all_png_hashes_revalidated": True,
            "mechanism": "capture_manifest_sha256_plus_per_png_sha256_map",
        },
        "outputs": {
            "all_view_ab_residuals": file_binding(ab_csv, repository),
            "all_view_per_tooth_residuals": file_binding(run_csv, repository),
            "report": file_binding(report_path, repository),
        },
        "sources": {
            "analysis_source": file_binding(analyzer_source, repository),
            "capture_helper": file_binding(helper, repository),
            "runner_source_snapshot": file_binding(runner, repository),
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"manifest": manifest, "report": report}


def _arguments(argv=None):
    repository = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description="Aggregate strict four-view D38999 tooth evidence"
    )
    parser.add_argument("--repository", type=Path, default=repository)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--rtx-history-512", type=Path, required=True)
    parser.add_argument("--segment00-normalized", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--capture-helper",
        type=Path,
        default=(
            repository
            / "src/kcg_connector/isaac/d38999_tooth_sync_capture.py"
        ),
    )
    parser.add_argument(
        "--runner-source",
        type=Path,
        default=(
            repository
            / "src/kcg_connector/isaac/d38999_nut_regrasp_smoke.py"
        ),
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    arguments = _arguments(argv)
    try:
        result = aggregate_three_run_evidence(
            repository=arguments.repository,
            captures={
                "baseline": arguments.baseline,
                "rtx_history_512": arguments.rtx_history_512,
                "segment00_normalized": arguments.segment00_normalized,
            },
            capture_helper=arguments.capture_helper,
            runner_source=arguments.runner_source,
            output_directory=arguments.output,
        )
    except (EvidenceError, analysis.EvidenceError, OSError, ValueError) as error:
        print(json.dumps({"evidence_valid": False, "error": str(error)}))
        return 2
    print(json.dumps(result["report"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COMPARISON_IDS",
    "EXPECTED_SEGMENTS",
    "EvidenceError",
    "MANIFEST_SCHEMA_VERSION",
    "RUN_IDS",
    "SCHEMA_VERSION",
    "aggregate_three_run_evidence",
    "collect_all_view_ab_evidence",
    "collect_all_view_run_evidence",
    "file_binding",
    "main",
    "sha256_file",
    "validate_physics_report",
    "validate_treatment",
]
