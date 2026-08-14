"""Fail-closed aggregation of repeated virtual wrist-FT monitor runs.

This module is intentionally Isaac-free.  It verifies immutable runner and
monitor-config hashes, then aggregates at least three completed headless E2E
logs/reports.  The result is observation-only reproducibility evidence: it
cannot derive safety thresholds, claim calibration, or mark training ready.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence

import yaml

from kcg_connector.virtual_wrist_ft_runtime import (
    HOME_TARE_PHASE,
    PAYLOAD_CAPTURE_PHASE,
    PROTECTED_PHASES,
    SCHEMA_VERSION as MONITOR_SCHEMA_VERSION,
    WRENCH_ORDER,
    load_virtual_wrist_ft_monitor_config,
    verify_virtual_wrist_ft_monitor_inputs,
)


REQUEST_SCHEMA_VERSION = "kcg_d38999_wrist_ft_repeatability_request_v1"
ARTIFACT_SCHEMA_VERSION = "kcg_d38999_wrist_ft_repeatability_artifact_v1"
MINIMUM_RUNS = 3
RUNNER_SOURCE_PATH = (
    "src/kcg_connector/isaac/d38999_tabletop_pick_smoke.py"
)
MONITOR_CONFIG_PATH = (
    "src/kcg_connector/config/d38999_wrist_ft_monitor_v1.yaml"
)
METRICS_KINDS = frozenset(
    {"headless_e2e_jsonl_log", "headless_e2e_metrics_json"}
)
RESULT_BANNER = "ISAAC D38999 END TO END V1 PASSED"
SCALAR_NAMES = (
    "lateral_force_n",
    "axial_force_n",
    "bending_torque_nm",
    "tightening_torque_nm",
)

_REQUEST_KEYS = {
    "schema_version",
    "minimum_runs",
    "policy",
    "runs",
}
_POLICY_KEYS = {
    "monitor_only",
    "statistics_only",
    "generate_safety_thresholds",
    "claim_calibration",
    "claim_training_ready",
}
_RUN_KEYS = {
    "run_id",
    "metrics_artifact",
    "runner_source",
    "monitor_config",
}
_FILE_KEYS = {"path", "sha256"}
_METRICS_FILE_KEYS = {"kind", "path", "sha256"}
_REPORT_KEYS = {
    "schema_version",
    "status",
    "measurement_joint",
    "reaction_row_index",
    "metadata_joint_index_offset",
    "raw_frame",
    "task_frame",
    "canonical_from_raw",
    "home_empty_baseline_canonical",
    "payload_baseline_canonical",
    "payload_increment_estimate_canonical",
    "phase_sample_counts",
    "protected_phase_peaks",
    "last_sample",
    "compensation_mode",
    "dynamic_inertia_compensation_complete",
    "orientation_dependent_gravity_compensation_complete",
    "same_scene_threshold_calibration_status",
    "calibrated_safety_limits",
    "monitor_only",
    "modifies_e2e_pass_gate",
    "residual_v1_enabled",
    "safety_gate_claimed",
    "assembly_success_claimed_from_wrench",
}
_SAMPLE_BASE_KEYS = {
    "global_step",
    "timestamp_s",
    "runtime_phase",
    "policy_phase",
    "source_frame",
    "target_frame",
    "raw_wrench",
    "canonical_wrench_sensor",
}
_SAMPLE_CONTACT_KEYS = {
    "compensated_wrench_sensor",
    "compensated_wrench_task",
    "task_scalars",
}


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _exact_keys(
    value: Mapping[str, Any], expected: set[str], label: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} keys are invalid: "
            f"missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be non-empty text")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
    ):
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _vector(value: Any, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != len(WRENCH_ORDER):
        raise ValueError(f"{label} must be a six-element list")
    return [_number(item, f"{label}[]") for item in value]


def _sha256_text(value: Any, label: str) -> str:
    digest = _text(value, label)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{label} must be lowercase SHA-256")
    return digest


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_file(
    repository: Path,
    raw_path: Any,
    label: str,
    *,
    repository_relative: bool,
) -> Path:
    text = _text(raw_path, label)
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = repository / candidate
    resolved = candidate.resolve()
    if repository_relative:
        try:
            resolved.relative_to(repository)
        except ValueError as error:
            raise ValueError(
                f"{label} must remain repository-relative"
            ) from error
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _verify_bound_file(
    document: Any,
    label: str,
    repository: Path,
    expected_relative_path: str,
) -> dict[str, str]:
    binding = _mapping(document, label)
    _exact_keys(binding, _FILE_KEYS, label)
    relative_path = _text(binding["path"], f"{label}.path")
    if relative_path != expected_relative_path:
        raise ValueError(
            f"{label}.path must equal {expected_relative_path}"
        )
    expected_digest = _sha256_text(binding["sha256"], f"{label}.sha256")
    path = _resolve_file(
        repository,
        relative_path,
        f"{label}.path",
        repository_relative=True,
    )
    if _digest(path) != expected_digest:
        raise ValueError(f"{label} SHA-256 mismatch")
    return {"path": relative_path, "sha256": expected_digest}


def _validate_policy(document: Any) -> None:
    policy = _mapping(document, "policy")
    _exact_keys(policy, _POLICY_KEYS, "policy")
    expected = {
        "monitor_only": True,
        "statistics_only": True,
        "generate_safety_thresholds": False,
        "claim_calibration": False,
        "claim_training_ready": False,
    }
    for key, expected_value in expected.items():
        if policy[key] is not expected_value:
            raise ValueError(f"policy.{key} must remain {expected_value}")


def _validate_sample(
    document: Any,
    label: str,
    *,
    raw_frame: str,
    task_frame: str,
) -> dict[str, Any]:
    sample = _mapping(document, label)
    keys = set(sample)
    if keys not in (
        _SAMPLE_BASE_KEYS,
        _SAMPLE_BASE_KEYS | _SAMPLE_CONTACT_KEYS,
    ):
        allowed = _SAMPLE_BASE_KEYS | _SAMPLE_CONTACT_KEYS
        raise ValueError(
            f"{label} keys are invalid: "
            f"missing={sorted(_SAMPLE_BASE_KEYS - keys)}, "
            f"extra={sorted(keys - allowed)}"
        )
    _integer(sample["global_step"], f"{label}.global_step")
    timestamp = _number(sample["timestamp_s"], f"{label}.timestamp_s")
    if timestamp < 0.0:
        raise ValueError(f"{label}.timestamp_s must be non-negative")
    _text(sample["runtime_phase"], f"{label}.runtime_phase")
    policy_phase = _text(sample["policy_phase"], f"{label}.policy_phase")
    if sample["source_frame"] != raw_frame:
        raise ValueError(f"{label}.source_frame does not match report")
    if sample["target_frame"] != task_frame:
        raise ValueError(f"{label}.target_frame does not match report")
    _vector(sample["raw_wrench"], f"{label}.raw_wrench")
    canonical = _vector(
        sample["canonical_wrench_sensor"],
        f"{label}.canonical_wrench_sensor",
    )
    if _SAMPLE_CONTACT_KEYS <= keys:
        if policy_phase not in PROTECTED_PHASES:
            raise ValueError(
                f"{label} has contact fields outside protected phase"
            )
        _vector(
            sample["compensated_wrench_sensor"],
            f"{label}.compensated_wrench_sensor",
        )
        _vector(
            sample["compensated_wrench_task"],
            f"{label}.compensated_wrench_task",
        )
        scalars = _mapping(sample["task_scalars"], f"{label}.task_scalars")
        _exact_keys(scalars, set(SCALAR_NAMES), f"{label}.task_scalars")
        for name in SCALAR_NAMES:
            _number(scalars[name], f"{label}.task_scalars.{name}")
    return {"canonical_wrench_sensor": canonical}


def _validate_peak_record(
    document: Any,
    label: str,
    *,
    raw_frame: str,
    task_frame: str,
) -> float:
    record = _mapping(document, label)
    _exact_keys(
        record,
        {"absolute_peak", "signed_value_at_peak", "sample"},
        label,
    )
    absolute_peak = _number(record["absolute_peak"], f"{label}.absolute_peak")
    signed_peak = _number(
        record["signed_value_at_peak"],
        f"{label}.signed_value_at_peak",
    )
    if absolute_peak < 0.0 or not math.isclose(
        absolute_peak,
        abs(signed_peak),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{label} absolute and signed peaks disagree")
    _validate_sample(
        record["sample"],
        f"{label}.sample",
        raw_frame=raw_frame,
        task_frame=task_frame,
    )
    return absolute_peak


def _validate_monitor_report(
    document: Any, config: Any, label: str
) -> dict[str, Any]:
    report = _mapping(document, label)
    _exact_keys(report, _REPORT_KEYS, label)
    if report["schema_version"] != MONITOR_SCHEMA_VERSION:
        raise ValueError(f"{label}.schema_version is unsupported")
    if report["status"] != "MONITOR_ONLY":
        raise ValueError(f"{label}.status must equal MONITOR_ONLY")
    if report["monitor_only"] is not True:
        raise ValueError(f"{label}.monitor_only must remain true")
    if report["measurement_joint"] != config.measurement_joint:
        raise ValueError(f"{label}.measurement_joint does not match config")
    _integer(
        report["reaction_row_index"],
        f"{label}.reaction_row_index",
        minimum=1,
    )
    if (
        report["metadata_joint_index_offset"]
        != config.metadata_joint_index_offset
    ):
        raise ValueError(
            f"{label}.metadata_joint_index_offset does not match config"
        )
    if report["raw_frame"] != config.raw_frame:
        raise ValueError(f"{label}.raw_frame does not match config")
    if report["task_frame"] != config.task_frame_id:
        raise ValueError(f"{label}.task_frame does not match config")
    expected_matrix = [list(row) for row in config.canonical_from_raw]
    if report["canonical_from_raw"] != expected_matrix:
        raise ValueError(f"{label}.canonical_from_raw does not match config")

    vectors = {}
    for key in (
        "home_empty_baseline_canonical",
        "payload_baseline_canonical",
        "payload_increment_estimate_canonical",
    ):
        vectors[key] = _vector(report[key], f"{label}.{key}")

    phase_counts = _mapping(
        report["phase_sample_counts"], f"{label}.phase_sample_counts"
    )
    required_phases = {
        HOME_TARE_PHASE,
        PAYLOAD_CAPTURE_PHASE,
        *PROTECTED_PHASES,
    }
    unexpected_phases = set(phase_counts) - required_phases - {"OTHER"}
    if unexpected_phases:
        raise ValueError(
            f"{label}.phase_sample_counts has unknown phases: "
            f"{sorted(unexpected_phases)}"
        )
    if not required_phases <= set(phase_counts):
        raise ValueError(f"{label}.phase_sample_counts is incomplete")
    for phase, count in phase_counts.items():
        minimum = (
            config.minimum_capture_samples
            if phase in {HOME_TARE_PHASE, PAYLOAD_CAPTURE_PHASE}
            else 1
        )
        _integer(
            count,
            f"{label}.phase_sample_counts.{phase}",
            minimum=minimum,
        )

    peak_document = _mapping(
        report["protected_phase_peaks"],
        f"{label}.protected_phase_peaks",
    )
    _exact_keys(
        peak_document,
        set(PROTECTED_PHASES),
        f"{label}.protected_phase_peaks",
    )
    peaks: dict[str, dict[str, float]] = {}
    for phase in PROTECTED_PHASES:
        phase_document = _mapping(
            peak_document[phase],
            f"{label}.protected_phase_peaks.{phase}",
        )
        _exact_keys(
            phase_document,
            set(SCALAR_NAMES),
            f"{label}.protected_phase_peaks.{phase}",
        )
        peaks[phase] = {
            name: _validate_peak_record(
                phase_document[name],
                f"{label}.protected_phase_peaks.{phase}.{name}",
                raw_frame=config.raw_frame,
                task_frame=config.task_frame_id,
            )
            for name in SCALAR_NAMES
        }

    last_sample = _validate_sample(
        report["last_sample"],
        f"{label}.last_sample",
        raw_frame=config.raw_frame,
        task_frame=config.task_frame_id,
    )
    if report["compensation_mode"] != "captured_payload_quasistatic_baseline":
        raise ValueError(f"{label}.compensation_mode is unsupported")
    for key in (
        "dynamic_inertia_compensation_complete",
        "orientation_dependent_gravity_compensation_complete",
        "modifies_e2e_pass_gate",
        "residual_v1_enabled",
        "safety_gate_claimed",
        "assembly_success_claimed_from_wrench",
    ):
        if report[key] is not False:
            raise ValueError(f"{label}.{key} must remain false")
    if (
        report["same_scene_threshold_calibration_status"]
        != config.threshold_status
    ):
        raise ValueError(
            f"{label}.same_scene_threshold_calibration_status "
            "does not match config"
        )
    if report["calibrated_safety_limits"] is not None:
        raise ValueError(f"{label}.calibrated_safety_limits must remain null")
    return {
        "vectors": vectors,
        "last_sample": last_sample["canonical_wrench_sensor"],
        "peaks": peaks,
    }


def _read_metrics(path: Path, kind: str, label: str) -> Mapping[str, Any]:
    content = path.read_text(encoding="utf-8")
    if kind == "headless_e2e_metrics_json":
        try:
            return _mapping(json.loads(content), label)
        except json.JSONDecodeError as error:
            raise ValueError(f"{label} is not valid JSON") from error

    candidates = []
    banner_present = False
    for line_number, raw_line in enumerate(content.splitlines(), start=1):
        line = raw_line.strip()
        if line == RESULT_BANNER:
            banner_present = True
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{label} has malformed JSON on line {line_number}"
            ) from error
        if isinstance(value, Mapping) and "virtual_wrist_ft_monitor" in value:
            candidates.append(value)
    if not banner_present:
        raise ValueError(f"{label} is missing the exact PASS banner")
    if len(candidates) != 1:
        raise ValueError(
            f"{label} must contain exactly one final wrist-FT metrics object"
        )
    return candidates[0]


def _validate_metrics(
    metrics: Mapping[str, Any], config: Any, label: str
) -> dict[str, Any]:
    expected_true = (
        "passed",
        "end_to_end_probe_requested",
        "wrist_ft_monitor_requested",
    )
    for key in expected_true:
        if metrics.get(key) is not True:
            raise ValueError(f"{label}.{key} must be true")
    if metrics.get("gui") is not False:
        raise ValueError(f"{label}.gui must be false for headless evidence")
    if "error" in metrics:
        raise ValueError(f"{label}.error must be absent")
    end_to_end = _mapping(metrics.get("end_to_end"), f"{label}.end_to_end")
    if end_to_end.get("passed") is not True:
        raise ValueError(f"{label}.end_to_end.passed must be true")
    return _validate_monitor_report(
        metrics.get("virtual_wrist_ft_monitor"),
        config,
        f"{label}.virtual_wrist_ft_monitor",
    )


def _scalar_statistics(values: Sequence[float]) -> dict[str, Any]:
    return {
        "sample_count": len(values),
        "mean": statistics.fmean(values),
        "sample_standard_deviation": statistics.stdev(values),
        "minimum_observed": min(values),
        "maximum_observed": max(values),
        "observed_span": max(values) - min(values),
    }


def _vector_statistics(vectors: Sequence[Sequence[float]]) -> dict[str, Any]:
    per_axis = {
        axis: _scalar_statistics([vector[index] for vector in vectors])
        for index, axis in enumerate(WRENCH_ORDER)
    }
    pairwise_distances = []
    for left_index, left in enumerate(vectors):
        for right in vectors[left_index + 1:]:
            pairwise_distances.append(
                math.sqrt(
                    sum(
                        (left_value - right_value) ** 2
                        for left_value, right_value in zip(left, right)
                    )
                )
            )
    return {
        "sample_count": len(vectors),
        "axis_order": list(WRENCH_ORDER),
        "per_axis_observed": per_axis,
        "maximum_observed_pairwise_l2_distance": max(pairwise_distances),
    }


def aggregate_repeatability(
    manifest_path: str | Path,
    repository: str | Path,
) -> dict[str, Any]:
    """Validate headless artifacts and return statistics-only evidence."""

    root = Path(repository).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    manifest = Path(manifest_path).expanduser().resolve()
    request = _mapping(
        yaml.safe_load(manifest.read_text(encoding="utf-8")),
        "root",
    )
    _exact_keys(request, _REQUEST_KEYS, "root")
    if request["schema_version"] != REQUEST_SCHEMA_VERSION:
        raise ValueError("unexpected repeatability request schema_version")
    if request["minimum_runs"] != MINIMUM_RUNS:
        raise ValueError(f"minimum_runs must equal {MINIMUM_RUNS}")
    _validate_policy(request["policy"])
    runs = request["runs"]
    if not isinstance(runs, list) or len(runs) < MINIMUM_RUNS:
        raise ValueError(f"runs must contain at least {MINIMUM_RUNS} entries")

    config_path = root / MONITOR_CONFIG_PATH
    config = load_virtual_wrist_ft_monitor_config(config_path)
    verify_virtual_wrist_ft_monitor_inputs(config, root)

    run_ids = set()
    metrics_paths = set()
    metrics_digests = set()
    source_binding = None
    config_binding = None
    validated_runs = []
    provenance_runs = []
    for index, raw_run in enumerate(runs):
        label = f"runs[{index}]"
        run = _mapping(raw_run, label)
        _exact_keys(run, _RUN_KEYS, label)
        run_id = _text(run["run_id"], f"{label}.run_id")
        if run_id in run_ids:
            raise ValueError("run_id values must be unique")
        run_ids.add(run_id)

        current_source = _verify_bound_file(
            run["runner_source"],
            f"{label}.runner_source",
            root,
            RUNNER_SOURCE_PATH,
        )
        current_config = _verify_bound_file(
            run["monitor_config"],
            f"{label}.monitor_config",
            root,
            MONITOR_CONFIG_PATH,
        )
        if source_binding is None:
            source_binding = current_source
            config_binding = current_config
        elif (
            current_source != source_binding
            or current_config != config_binding
        ):
            raise ValueError("all runs must bind the same source and config")

        artifact = _mapping(
            run["metrics_artifact"], f"{label}.metrics_artifact"
        )
        _exact_keys(
            artifact,
            _METRICS_FILE_KEYS,
            f"{label}.metrics_artifact",
        )
        kind = _text(artifact["kind"], f"{label}.metrics_artifact.kind")
        if kind not in METRICS_KINDS:
            raise ValueError(f"{label}.metrics_artifact.kind is unsupported")
        metrics_path = _resolve_file(
            root,
            artifact["path"],
            f"{label}.metrics_artifact.path",
            repository_relative=False,
        )
        metrics_digest = _sha256_text(
            artifact["sha256"], f"{label}.metrics_artifact.sha256"
        )
        if _digest(metrics_path) != metrics_digest:
            raise ValueError(f"{label}.metrics_artifact SHA-256 mismatch")
        path_identity = str(metrics_path)
        if path_identity in metrics_paths or metrics_digest in metrics_digests:
            raise ValueError("run metrics artifacts must be distinct")
        metrics_paths.add(path_identity)
        metrics_digests.add(metrics_digest)

        metrics = _read_metrics(
            metrics_path, kind, f"{label}.metrics_artifact"
        )
        validated = _validate_metrics(metrics, config, label)
        validated_runs.append(validated)
        canonical_report = json.dumps(
            metrics["virtual_wrist_ft_monitor"],
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        provenance_runs.append(
            {
                "run_id": run_id,
                "metrics_artifact": {
                    "kind": kind,
                    "path": _text(
                        artifact["path"], f"{label}.metrics_artifact.path"
                    ),
                    "sha256": metrics_digest,
                },
                "monitor_report_sha256": hashlib.sha256(
                    canonical_report
                ).hexdigest(),
            }
        )

    vector_names = (
        "home_empty_baseline_canonical",
        "payload_baseline_canonical",
        "payload_increment_estimate_canonical",
    )
    vector_statistics = {
        name: _vector_statistics(
            [run["vectors"][name] for run in validated_runs]
        )
        for name in vector_names
    }
    vector_statistics["last_sample_canonical_wrench_sensor"] = (
        _vector_statistics([run["last_sample"] for run in validated_runs])
    )
    peak_statistics = {
        phase: {
            name: _scalar_statistics(
                [run["peaks"][phase][name] for run in validated_runs]
            )
            for name in SCALAR_NAMES
        }
        for phase in PROTECTED_PHASES
    }
    result = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "MONITOR_ONLY_REPEATABILITY_EVIDENCE",
        "run_count": len(validated_runs),
        "minimum_required_runs": MINIMUM_RUNS,
        "axis_order": list(WRENCH_ORDER),
        "provenance": {
            "runner_source": source_binding,
            "monitor_config": config_binding,
            "run_artifacts": provenance_runs,
            "all_hashes_verified": True,
            "duplicate_artifacts_rejected": True,
        },
        "observed_statistics": {
            "wrench_vectors": vector_statistics,
            "protected_phase_absolute_peaks": peak_statistics,
        },
        "claims": {
            "monitor_only": True,
            "statistics_only": True,
            "safety_thresholds_generated": False,
            "calibration_claimed": False,
            "training_ready_claimed": False,
            "safety_gate_enabled": False,
            "e2e_gate_modified": False,
        },
        "calibrated_safety_limits": None,
        "generated_safety_thresholds": None,
    }
    # Reject non-finite values before callers persist an evidence artifact.
    json.dumps(result, allow_nan=False, sort_keys=True)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate at least three hash-bound headless virtual wrist-FT "
            "runs into monitor-only statistics."
        )
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--repository", default=Path.cwd(), type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="New JSON artifact path; existing files are never overwritten.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = aggregate_repeatability(
            arguments.manifest, arguments.repository
        )
        encoded = json.dumps(
            result, allow_nan=False, indent=2, sort_keys=True
        )
        if arguments.output is None:
            print(encoded)
        else:
            output = arguments.output.expanduser().resolve()
            if not output.parent.is_dir():
                raise FileNotFoundError(output.parent)
            # Exclusive creation prevents an earlier evidence artifact from
            # being silently replaced by a later run set.
            with output.open("x", encoding="utf-8") as stream:
                stream.write(encoded + "\n")
            print(output)
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(
            f"wrist FT repeatability aggregation FAILED: {error}",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "MINIMUM_RUNS",
    "MONITOR_CONFIG_PATH",
    "REQUEST_SCHEMA_VERSION",
    "RUNNER_SOURCE_PATH",
    "aggregate_repeatability",
    "main",
]
