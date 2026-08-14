#!/usr/bin/env python3

"""Aggregate one baseline and one ghost-fingers synchronized tooth run.

The GPU runtime owns only render visibility.  This CPU module independently
revalidates every PNG/hash/sync row, the ghost runtime manifest and cleanup,
three physical-equivalence fingerprints, and all-view tooth coverage.  A
successful evidence parse is deliberately separate from the strict 24/24
occlusion-control gate: partial visibility improvement remains a valid but
blocked result, never a visual-jitter PASS.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from kcg_connector import d38999_tooth_occlusion_control as control
from kcg_connector import d38999_tooth_sync_analysis as analysis
from kcg_connector import d38999_tooth_sync_evidence as sync_evidence


SCHEMA_VERSION = "kcg_d38999_tooth_occlusion_evidence_v1"
MANIFEST_SCHEMA_VERSION = (
    "kcg_d38999_tooth_occlusion_evidence_manifest_v1"
)
RUNTIME_MANIFEST_SCHEMA_VERSION = "kcg_d38999_tooth_ghost_manifest_v1"
EXPECTED_STATE_TRACE_SHA256 = (
    "af4b1d6f10fe7b1eae875a9335aa92a3c89ce750b77b37d768cdeebd78d66e00"
)
EXPECTED_CONTACT_TRACE_SHA256 = (
    "35c04f4a23b2795eabce6f3e65213ce5e6fbb56ea50f080f3955f748441a3e15"
)
EXPECTED_CONTACT_DYNAMICS_SHA256 = (
    "b6d9d6533c9eccf0b1460432f6ed9c111b4790ae4c308a4c1a367819a5a582da"
)


class EvidenceError(RuntimeError):
    """Raised when an occlusion-control input is not safely attributable."""


def sha256_file(path: str | Path) -> str:
    """Hash one evidence file without loading it fully into memory."""

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


def file_binding(path: str | Path, repository: Path) -> dict[str, Any]:
    """Return a repository-relative path, SHA and exact size."""

    repository = Path(repository).resolve()
    target = _repo_file(path, repository, "bound file")
    return {
        "path": str(target.relative_to(repository)),
        "sha256": sha256_file(target),
        "size_bytes": target.stat().st_size,
    }


def validate_external_binding(
    binding: Mapping[str, Any], expected_path: Path, label: str
) -> None:
    """Validate one absolute runtime binding without trusting its path."""

    if not isinstance(binding, Mapping):
        raise EvidenceError(f"{label} binding is not a mapping")
    expected = Path(expected_path).resolve()
    try:
        bound = Path(binding["path"]).expanduser().resolve()
    except (KeyError, TypeError, ValueError) as error:
        raise EvidenceError(f"{label} path is invalid") from error
    if bound != expected:
        raise EvidenceError(f"{label} path differs from expected input")
    if (
        binding.get("sha256") != sha256_file(expected)
        or binding.get("size_bytes") != expected.stat().st_size
    ):
        raise EvidenceError(f"{label} size/SHA binding differs")


def contact_trace_sha256(summary_path: str | Path) -> str:
    """Hash exact per-step contact counts independently of runtime code."""

    columns = (
        "global_step",
        "phase",
        "phase_step",
        "segment_contact_records",
    )
    digest = hashlib.sha256()
    rows = 0
    with Path(summary_path).open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        reader = csv.DictReader(stream)
        if not set(columns).issubset(reader.fieldnames or ()):
            raise EvidenceError("contact trace columns are incomplete")
        digest.update(("\x1f".join(columns) + "\n").encode("utf-8"))
        for row in reader:
            digest.update(
                ("\x1f".join(row[name] for name in columns) + "\n").encode(
                    "utf-8"
                )
            )
            rows += 1
    if rows == 0:
        raise EvidenceError("contact trace is empty")
    return digest.hexdigest()


def validate_runtime_bundle(
    *,
    repository: Path,
    ghost_root: Path,
    capture_manifest_path: Path,
    physics_report_path: Path,
    physics_summary_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Revalidate the runtime sidecar and every file/source binding."""

    runtime_directory = ghost_root / "ghost"
    manifest_path = runtime_directory / "manifest.json"
    sidecar_path = runtime_directory / "visibility_sidecar.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != RUNTIME_MANIFEST_SCHEMA_VERSION:
        raise EvidenceError("ghost runtime manifest schema mismatch")
    if manifest.get("status") != "HASH_SIZE_SCHEMA_BOUND":
        raise EvidenceError("ghost runtime manifest status mismatch")
    inputs = manifest.get("inputs", {})
    outputs = manifest.get("outputs", {})
    sources = manifest.get("sources", {})
    validate_external_binding(
        inputs.get("capture_manifest", {}),
        capture_manifest_path,
        "capture manifest",
    )
    validate_external_binding(
        inputs.get("physics_report", {}),
        physics_report_path,
        "physics report",
    )
    validate_external_binding(
        inputs.get("physics_summary", {}),
        physics_summary_path,
        "physics summary",
    )
    validate_external_binding(
        outputs.get("visibility_sidecar", {}),
        sidecar_path,
        "visibility sidecar",
    )
    expected_sources = {
        "contact_fingerprint_contract": (
            repository
            / "src/kcg_connector/kcg_connector/"
            "d38999_tooth_occlusion_control.py"
        ),
        "prepared_tooth_runner": (
            repository
            / "src/kcg_connector/isaac/d38999_nut_regrasp_smoke.py"
        ),
        "runtime": (
            repository
            / "src/kcg_connector/isaac/d38999_tooth_ghost_runtime.py"
        ),
    }
    if set(sources) != set(expected_sources):
        raise EvidenceError("ghost runtime source binding names differ")
    for name, path in expected_sources.items():
        validate_external_binding(sources[name], path, name)
    runtime = control.validate_runtime_sidecar(sidecar)
    return {"manifest": manifest, "sidecar": sidecar}, runtime


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise EvidenceError(f"cannot write empty CSV: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _identity_count(coverage: Mapping[str, Any]) -> int:
    return len(coverage.get("segments_in_identity_union", ()))


def _strict_blocker(error: Exception) -> str:
    message = str(error)
    allowed = (
        "ghost visual identity union is not 24/24",
        "ghost does not expose all 24 IDs at every transition",
    )
    if not any(text in message for text in allowed):
        raise EvidenceError(
            "strict evaluation failed before visual coverage: " + message
        ) from error
    return message


def aggregate_occlusion_evidence(
    *,
    repository: str | Path,
    baseline_root: str | Path,
    ghost_root: str | Path,
    capture_helper: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Build one immutable baseline-vs-ghost occlusion evidence artifact."""

    repository = Path(repository).expanduser().resolve()
    baseline_root = _repo_directory(
        baseline_root, repository, "baseline root"
    )
    ghost_root = _repo_directory(ghost_root, repository, "ghost root")
    helper = _repo_file(capture_helper, repository, "capture helper")
    output = Path(output_directory).expanduser().resolve()
    if output.exists():
        raise EvidenceError(f"output already exists: {output}")
    if repository != output and repository not in output.parents:
        raise EvidenceError("output escapes repository")

    baseline_capture = baseline_root / "capture"
    ghost_capture = ghost_root / "capture"
    baseline_bundle = analysis.validate_capture_bundle(
        baseline_capture, helper
    )
    ghost_bundle = analysis.validate_capture_bundle(ghost_capture, helper)
    baseline_result = analysis.analyze_validated_capture(baseline_bundle)
    ghost_result = analysis.analyze_validated_capture(ghost_bundle)
    baseline_rows, baseline_coverage = (
        sync_evidence.collect_all_view_run_evidence(
            "opaque_fingers_baseline", baseline_result
        )
    )
    ghost_rows, ghost_coverage = (
        sync_evidence.collect_all_view_run_evidence(
            "ghost_fingers_baseline", ghost_result
        )
    )

    baseline_report_path = baseline_bundle["physics_report_path"]
    baseline_summary_path = baseline_bundle["physics_summary_path"]
    ghost_report_path = ghost_bundle["physics_report_path"]
    ghost_summary_path = ghost_bundle["physics_summary_path"]
    runtime_bundle, runtime = validate_runtime_bundle(
        repository=repository,
        ghost_root=ghost_root,
        capture_manifest_path=ghost_capture
        / "video_capture_manifest.json",
        physics_report_path=ghost_report_path,
        physics_summary_path=ghost_summary_path,
    )
    baseline_report = baseline_bundle["physics_report"]
    ghost_report = ghost_bundle["physics_report"]
    baseline_state = baseline_bundle["physics_trace_sha256"]
    ghost_state = ghost_bundle["physics_trace_sha256"]
    baseline_contact_trace = contact_trace_sha256(baseline_summary_path)
    ghost_contact_trace = contact_trace_sha256(ghost_summary_path)
    baseline_dynamics = control.contact_dynamics_fingerprint(
        baseline_report
    )
    ghost_dynamics = control.contact_dynamics_fingerprint(ghost_report)
    expected = {
        "state_trace_sha256": EXPECTED_STATE_TRACE_SHA256,
        "contact_trace_sha256": EXPECTED_CONTACT_TRACE_SHA256,
        "contact_dynamics_sha256": EXPECTED_CONTACT_DYNAMICS_SHA256,
    }
    observed_pairs = {
        "state_trace_sha256": (baseline_state, ghost_state),
        "contact_trace_sha256": (
            baseline_contact_trace,
            ghost_contact_trace,
        ),
        "contact_dynamics_sha256": (
            baseline_dynamics,
            ghost_dynamics,
        ),
    }
    for name, pair in observed_pairs.items():
        if pair != (expected[name], expected[name]):
            raise EvidenceError(f"baseline/ghost {name} differs")
    sidecar_bindings = runtime_bundle["sidecar"].get("bindings", {})
    sidecar_expected = {
        "physics_state_trace_sha256": expected["state_trace_sha256"],
        "physics_contact_trace_sha256": expected["contact_trace_sha256"],
        "contact_dynamics_sha256": expected["contact_dynamics_sha256"],
    }
    for name, value in sidecar_expected.items():
        if sidecar_bindings.get(name) != value:
            raise EvidenceError(f"runtime sidecar {name} differs")

    strict_passed = True
    strict_error = None
    try:
        control.evaluate_occlusion_control(
            baseline_capture_manifest=baseline_bundle["manifest"],
            ghost_capture_manifest=ghost_bundle["manifest"],
            baseline_physics_report=baseline_report,
            ghost_physics_report=ghost_report,
            baseline_physics_trace_sha256=baseline_state,
            ghost_physics_trace_sha256=ghost_state,
            baseline_contact_trace_sha256=baseline_contact_trace,
            ghost_contact_trace_sha256=ghost_contact_trace,
            ghost_runtime_sidecar=runtime_bundle["sidecar"],
            visual_coverage=ghost_coverage,
        )
    except control.OcclusionControlError as error:
        strict_passed = False
        strict_error = _strict_blocker(error)

    baseline_segments = set(
        baseline_coverage["segments_in_identity_union"]
    )
    ghost_segments = set(ghost_coverage["segments_in_identity_union"])
    report = {
        "schema_version": SCHEMA_VERSION,
        "classification": (
            "VALID_STRICT_24_OCCLUSION_CONTROL_PASS"
            if strict_passed
            else "VALID_OCCLUSION_REDUCED_STRICT_24_BLOCKED"
        ),
        "evidence_valid": True,
        "strict_occlusion_control_passed": strict_passed,
        "strict_evaluation_error": strict_error,
        "physics": {
            "baseline_equals_ghost": True,
            "contact_dynamics_sha256": ghost_dynamics,
            "contact_trace_sha256": ghost_contact_trace,
            "state_trace_sha256": ghost_state,
        },
        "runtime": {
            **runtime,
            "cleanup": runtime_bundle["sidecar"]["cleanup"],
            "mutation_audit": runtime_bundle["sidecar"][
                "mutation_audit"
            ],
            "passed": runtime_bundle["sidecar"]["passed"],
        },
        "visual": {
            "baseline": baseline_coverage,
            "ghost": ghost_coverage,
            "identity_count_before": _identity_count(baseline_coverage),
            "identity_count_after": _identity_count(ghost_coverage),
            "identity_count_gain": (
                _identity_count(ghost_coverage)
                - _identity_count(baseline_coverage)
            ),
            "newly_measurable_segments": sorted(
                ghost_segments - baseline_segments
            ),
            "still_missing_segments": sorted(
                set(control.EXPECTED_SEGMENTS) - ghost_segments
            ),
            "render_jitter_absence_claim_authorized": False,
        },
        "scope": {
            "capture_rate_hz": 30,
            "physics_rate_hz": 240,
            "rgb_frames_per_run": ghost_result["rgb_frames_validated"],
            "sampled_transitions": ghost_coverage["transitions"],
            "views": list(analysis.VIEW_IDS),
        },
        "limitations": [
            "ghost_visibility_improvement_is_not_strict_24_tooth_coverage",
            "no_visual_residual_acceptance_threshold_was_preregistered",
            "30_hz_sampling_cannot_exclude_between_sample_render_artifacts",
            "prepared_twist_probe_is_not_full_end_to_end_assembly",
        ],
    }

    output.mkdir(parents=True, exist_ok=False)
    baseline_csv = output / "baseline_all_view_residuals.csv"
    ghost_csv = output / "ghost_all_view_residuals.csv"
    report_path = output / "report.json"
    _write_csv(baseline_csv, baseline_rows)
    _write_csv(ghost_csv, ghost_rows)
    report_path.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    source_path = Path(__file__).resolve()
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "HASH_SIZE_SCHEMA_BOUND",
        "inputs": {
            "baseline_capture_manifest": file_binding(
                baseline_capture / "video_capture_manifest.json", repository
            ),
            "baseline_physics_report": file_binding(
                baseline_report_path, repository
            ),
            "baseline_physics_summary": file_binding(
                baseline_summary_path, repository
            ),
            "ghost_capture_manifest": file_binding(
                ghost_capture / "video_capture_manifest.json", repository
            ),
            "ghost_physics_report": file_binding(
                ghost_report_path, repository
            ),
            "ghost_physics_summary": file_binding(
                ghost_summary_path, repository
            ),
            "ghost_runtime_manifest": file_binding(
                ghost_root / "ghost/manifest.json", repository
            ),
            "ghost_visibility_sidecar": file_binding(
                ghost_root / "ghost/visibility_sidecar.json", repository
            ),
        },
        "indirect_frame_binding": {
            "all_png_hashes_revalidated": True,
            "mechanism": "capture_manifest_per_png_sha256_maps",
            "rgb_frames_revalidated": (
                baseline_result["rgb_frames_validated"]
                + ghost_result["rgb_frames_validated"]
            ),
        },
        "outputs": {
            "baseline_all_view_residuals": file_binding(
                baseline_csv, repository
            ),
            "ghost_all_view_residuals": file_binding(
                ghost_csv, repository
            ),
            "report": file_binding(report_path, repository),
        },
        "sources": {
            "analysis": file_binding(Path(analysis.__file__), repository),
            "capture_helper": file_binding(helper, repository),
            "occlusion_contract": file_binding(
                Path(control.__file__), repository
            ),
            "occlusion_evidence": file_binding(source_path, repository),
            "sync_evidence": file_binding(
                Path(sync_evidence.__file__), repository
            ),
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"manifest": manifest, "report": report}


def _arguments(argv=None):
    repository = Path(__file__).resolve().parents[3]
    root = repository / "artifacts/kcg_connector/d38999_nut_tooth_jitter"
    parser = argparse.ArgumentParser(
        description="Aggregate strict D38999 finger-ghost tooth evidence"
    )
    parser.add_argument("--repository", type=Path, default=repository)
    parser.add_argument(
        "--baseline-root", type=Path, default=root / "four_synced_baseline"
    )
    parser.add_argument("--ghost-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--capture-helper",
        type=Path,
        default=(
            repository
            / "src/kcg_connector/isaac/d38999_tooth_sync_capture.py"
        ),
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    arguments = _arguments(argv)
    try:
        result = aggregate_occlusion_evidence(
            repository=arguments.repository,
            baseline_root=arguments.baseline_root,
            ghost_root=arguments.ghost_root,
            capture_helper=arguments.capture_helper,
            output_directory=arguments.output,
        )
    except (
        EvidenceError,
        analysis.EvidenceError,
        control.OcclusionControlError,
        OSError,
        ValueError,
    ) as error:
        print(json.dumps({"evidence_valid": False, "error": str(error)}))
        return 2
    print(json.dumps(result["report"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EvidenceError",
    "MANIFEST_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "aggregate_occlusion_evidence",
    "contact_trace_sha256",
    "file_binding",
    "main",
    "sha256_file",
    "validate_external_binding",
    "validate_runtime_bundle",
]
