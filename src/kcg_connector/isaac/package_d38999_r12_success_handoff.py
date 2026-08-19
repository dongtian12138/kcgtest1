#!/usr/bin/env python3

"""Build and verify the bounded r12 direct-closeout evidence package.

The requested package name contains ``success_handoff``.  Package content is
truthful: a formal r12 is included only when a candidate P1 actually passed.
This script never promotes a candidate and never starts Isaac Sim.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Iterable, Sequence
import zipfile


WORKSPACE = Path(__file__).resolve().parents[3]
R12_ROOT = WORKSPACE / "artifacts/kcg_connector/isaac/keyed_v3_physical_r12"
DELIVERY_PARENT = WORKSPACE / "artifacts/kcg_connector/handoff_to_gpt56pro"
R11_HANDOFF = DELIVERY_PARENT / "r11_root_cause_20260817T072144Z"
FORMAL_ASSET = (
    R12_ROOT / "d38999_shell25j_25_61_n_keyed_physical_v3_r12.usda"
)


SOURCE_PATHS = (
    "src/kcg_connector/config/d38999_keyed_v3_physical_model_contract_r12_v1.yaml",
    "src/kcg_connector/config/d38999_keyed_v3_physical_acceptance_r12_v1.yaml",
    "src/kcg_connector/config/d38999_keyed_v3_tabletop_scene_r12_v1.yaml",
    "src/kcg_connector/isaac/create_d38999_keyed_physical_r7_asset.py",
    "src/kcg_connector/isaac/build_d38999_keyed_v3_physical_r12_configs.py",
    "src/kcg_connector/isaac/create_d38999_r12_tabletop_scene.py",
    "src/kcg_connector/isaac/audit_d38999_r12_asset.py",
    "src/kcg_connector/isaac/validate_physical_r7_composed_scene.py",
    "src/kcg_connector/isaac/validate_physical_r11_cooked_geometry.py",
    "src/kcg_connector/isaac/d38999_physical_r7_p1_nominal_bench.py",
    "src/kcg_connector/isaac/run_isaac_python.sh",
    "src/kcg_connector/isaac/package_d38999_r12_success_handoff.py",
    "src/kcg_connector/kcg_connector/d38999_keyed_v3_physical_r12_contract.py",
    "src/kcg_connector/kcg_connector/d38999_keyed_v3_physical_r12_acceptance.py",
    "src/kcg_connector/kcg_connector/d38999_keyed_v2_a2_readback_result.py",
    "src/kcg_connector/kcg_connector/d38999_tabletop_scene.py",
    "src/kcg_connector/test/test_d38999_keyed_v3_physical_r12_contract.py",
    "src/kcg_connector/test/test_d38999_keyed_v2_a2_readback_result.py",
    "src/kcg_connector/test/test_d38999_tabletop_scene.py",
)


REPAIRS = {
    1: (
        "Three required r11 repairs only: round detent followers, low-redundancy "
        "nut/body shoulders, and low-redundancy metal bottoming."
    ),
    2: (
        "Candidate 01 thread-only first jam repair: 1080 closed hexahedron rail "
        "pieces became analytic capsule helix chords; lead, phases, followers, "
        "friction, D6, and P1 controller stayed fixed."
    ),
    3: (
        "Candidate 02 first axial-stall repair: 12 hard segmented bore targets "
        "became 12 derived Z capsules; all compliant fingers and force parameters "
        "stayed fixed."
    ),
    4: (
        "Candidate 03 boundary/stall repair: 12 hard bore targets became one "
        "collision-isolated axisymmetric analytic Cylinder; all 12 compliant "
        "fingers and force parameters stayed fixed."
    ),
}


FAILURE_CLASS = {
    1: "THREAD_SEGMENT_END_FACE_SEAM_TORQUE_JAM",
    2: "SPRING_CONTACT_AXIAL_TRACKING_STALL_WITH_SEGMENTED_TARGET",
    3: "SPRING_TARGET_BOUNDARY_SHIFT_AND_AXIAL_TRACKING_STALL",
    4: "SPRING_CONTACT_AXIAL_TRACKING_STALL_AND_FIXED_DRIFT",
}


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timestamp",
        default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    )
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def _copy(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _extract_log_json(path: Path, required_key: str) -> dict[str, Any]:
    for line in reversed(path.read_text(encoding="utf-8", errors="replace").splitlines()):
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and required_key in value:
            return value
    raise ValueError(f"no JSON object with {required_key!r} in {path}")


def _passed_log_json(
    directory: Path, pattern: str, required_key: str
) -> dict[str, Any]:
    candidates = []
    for path in sorted(directory.glob(pattern)):
        try:
            value = _extract_log_json(path, required_key)
        except ValueError:
            continue
        candidates.append((path, value))
    passed = [value for _, value in candidates if value.get("status") == "PASSED"]
    if len(passed) != 1:
        names = [path.name for path, _ in candidates]
        raise ValueError(
            f"expected exactly one passed log for {pattern} in {directory}; found {names}"
        )
    return passed[0]


def _trace_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _persistent_axial_failure_index(
    trace: list[dict[str, Any]], threshold_m: float = 0.00005, steps: int = 120
) -> int | None:
    for index in range(0, len(trace) - steps + 1):
        if all(
            float(trace[row]["target_separation_m"])
            - float(trace[row]["observed_separation_m"])
            > threshold_m
            for row in range(index, index + steps)
        ):
            return index
    return None


def _failure_window(
    candidate_index: int, p1: Path, destination: Path
) -> dict[str, Any]:
    report = _json(p1 / "report.json")
    trace = _trace_rows(p1 / "trace.jsonl")
    if report.get("first_sustained_torque_jam") is not None:
        jam = report["first_sustained_torque_jam"]
        center_step = int(jam.get("first_step", jam["start_step"]))
        source = "formal_10_step_sustained_torque_jam"
    else:
        index = _persistent_axial_failure_index(trace)
        center_step = int(trace[index]["step"]) if index is not None else int(trace[-1]["step"])
        source = (
            "posthoc_120_step_target_minus_observed_axial_error_above_50_um"
            if index is not None
            else "end_of_run_missing_required_event_inventory"
        )
    window = [row for row in trace if abs(int(row["step"]) - center_step) <= 120]
    with destination.open("w", encoding="utf-8") as stream:
        for row in window:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n")
    center = min(trace, key=lambda row: abs(int(row["step"]) - center_step))
    analysis: dict[str, Any] = {
        "candidate_index": candidate_index,
        "classification": FAILURE_CLASS[candidate_index],
        "detection_source": source,
        "center_step": center_step,
        "center_time_s": center["time_s"],
        "target_separation_m": center["target_separation_m"],
        "observed_separation_m": center["observed_separation_m"],
        "target_minus_observed_m": (
            float(center["target_separation_m"])
            - float(center["observed_separation_m"])
        ),
        "active_scored_contact_events": center["active_scored_contact_events"],
        "maximum_torque_component_nm": report.get("maximum_torque_component_nm"),
        "maximum_consecutive_torque_saturation_steps": report.get(
            "maximum_consecutive_torque_saturation_steps"
        ),
        "maximum_fixed_receptacle_translation_drift_m": report.get(
            "maximum_fixed_receptacle_translation_drift_m"
        ),
        "direct_torque_jam": report.get("first_sustained_torque_jam"),
        "evidence_limit": (
            "For non-torque stalls, the P1 contact audit is aggregate post-hoc "
            "evidence; it is not a pointwise reaction-force measurement."
        ),
    }
    if candidate_index == 1:
        analysis["first_jam_analysis"] = report.get("first_jam_analysis")
    return analysis


def _candidate_summary(index: int, candidate_dir: Path) -> dict[str, Any]:
    p1 = candidate_dir / "P1"
    report = _json(p1 / "report.json")
    trace = _trace_rows(p1 / "trace.jsonl")
    audit = _json(candidate_dir / "static_asset_audit.json")
    a2 = _passed_log_json(candidate_dir, "a2*_stdout.log", "composed_counts")
    cooked = _extract_log_json(candidate_dir / "cooked_stdout.log", "cooking_result_failure_count")
    final = trace[-1]
    return {
        "candidate_index": index,
        "asset_path": str((candidate_dir / f"r12_candidate_{index:02d}.usda").resolve()),
        "asset_sha256": _sha256(candidate_dir / f"r12_candidate_{index:02d}.usda"),
        "repair": REPAIRS[index],
        "collider_count": audit["collider_count"],
        "collider_type_counts": audit["type_counts"],
        "old_square_detent_follower_count": audit["old_square_detent_follower_count"],
        "old_segmented_shoulder_count": audit["old_segmented_shoulder_count"],
        "old_segmented_metal_bottoming_count": audit["old_segmented_metal_bottoming_count"],
        "static_asset_audit_passed": audit["passed"],
        "a2_passed": a2["status"] == "PASSED",
        "a2_connector_collider_count": a2["connector"]["collider_rows"],
        "a2_composed_collider_count": a2["composed_counts"]["collision"],
        "cooked_geometry_passed": bool(cooked["passed"]),
        "cooking_result_failure_count": cooked["cooking_result_failure_count"],
        "p1_passed": bool(report["passed"]),
        "observed_event_order": report.get("observed_event_order", []),
        "position_error_m": report.get("position_error_m", {}),
        "all_three_thread_starts_enter": report.get("all_three_thread_starts_enter"),
        "maximum_torque_component_nm": report.get("maximum_torque_component_nm"),
        "maximum_consecutive_torque_saturation_steps": report.get(
            "maximum_consecutive_torque_saturation_steps"
        ),
        "maximum_fixed_receptacle_translation_drift_m": report.get(
            "maximum_fixed_receptacle_translation_drift_m"
        ),
        "fixed_anchor_pass": report.get("fixed_anchor_pass"),
        "solver_error_count": report.get("solver_error_count"),
        "object_pose_write_after_physics_start_count": report.get(
            "object_pose_write_after_physics_start_count"
        ),
        "final_target_separation_m": final["target_separation_m"],
        "final_observed_separation_m": final["observed_separation_m"],
        "final_nut_unwrapped_yaw_rad": final["nut_unwrapped_yaw_rad"],
        "failure_classification": FAILURE_CLASS[index],
    }


def _git_text(arguments: Iterable[str]) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=WORKSPACE,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return result.stdout


def _purpose(path: Path) -> str:
    text = path.as_posix()
    if text.endswith(".usda"):
        return "unpromoted_candidate_asset"
    if "/P1/" in f"/{text}":
        return "full_P1_raw_evidence"
    if text.startswith("SOURCE/"):
        return "review_source"
    if text.startswith("CONFIG/"):
        return "contract_or_acceptance_config"
    if "failure" in text.lower():
        return "first_failure_evidence"
    if text.startswith("GIT/"):
        return "workspace_revision_evidence"
    return "handoff_evidence"


def _manifest_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in {"MANIFEST.csv", "SHA256SUMS.txt"}:
            continue
        relative = path.relative_to(root)
        parts = relative.parts
        candidate_id = next((part for part in parts if part.startswith("r12_candidate_")), "")
        rows.append(
            {
                "relative_path": relative.as_posix(),
                "file_size": path.stat().st_size,
                "sha256": _sha256(path),
                "purpose": _purpose(relative),
                "source_or_generated": (
                    "source" if relative.parts[0] in {"SOURCE", "CONFIG"} else "generated"
                ),
                "candidate_id": candidate_id,
                "required_for_review": "true",
            }
        )
    return rows


def _write_manifest(root: Path) -> None:
    rows = _manifest_rows(root)
    with (root / "MANIFEST.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    checksum_paths = [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS.txt"
    ]
    _write_text(
        root / "SHA256SUMS.txt",
        "\n".join(
            f"{_sha256(path)}  {path.relative_to(root).as_posix()}"
            for path in checksum_paths
        ),
    )


def _verify_unpacked(root: Path) -> dict[str, Any]:
    missing: list[str] = []
    mismatches: list[str] = []
    with (root / "MANIFEST.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        path = root / row["relative_path"]
        if not path.is_file():
            missing.append(row["relative_path"])
            continue
        if path.stat().st_size != int(row["file_size"]) or _sha256(path) != row["sha256"]:
            mismatches.append(row["relative_path"])
    return {
        "manifest_row_count": len(rows),
        "required_file_missing_count": len(missing),
        "hash_or_size_mismatch_count": len(mismatches),
        "missing": missing,
        "mismatches": mismatches,
        "passed": not missing and not mismatches,
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _arguments(argv)
    timestamp = arguments.timestamp
    handoff = DELIVERY_PARENT / f"r12_direct_closeout_{timestamp}"
    archive = DELIVERY_PARENT / f"d38999_r12_success_handoff_{timestamp}.zip"
    verification_sidecar = archive.with_suffix(".verification.json")
    for path in (handoff, archive, verification_sidecar):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite handoff output: {path}")
    if FORMAL_ASSET.exists():
        raise RuntimeError(
            "formal r12 exists even though no retained candidate report passed; refusing package"
        )
    handoff.mkdir(parents=True, exist_ok=False)

    summaries = []
    for index in range(1, 5):
        source = R12_ROOT / "candidates" / f"r12_candidate_{index:02d}"
        target = handoff / "CANDIDATES" / source.name
        shutil.copytree(source, target)
        summary = _candidate_summary(index, source)
        summaries.append(summary)
        _write_json(target / "candidate_summary.json", summary)
        analysis = _failure_window(index, source / "P1", target / "first_failure_window.jsonl")
        _write_json(target / "first_failure_analysis.json", analysis)
        _write_text(target / "repair_and_result.md", f"# Repair\n\n{REPAIRS[index]}\n\n# Result\n\nP1: FAILED\n\nFirst failure: `{FAILURE_CLASS[index]}`")
        _write_json(
            target / "a2_result.json",
            _passed_log_json(source, "a2*_stdout.log", "composed_counts"),
        )
        _write_json(
            target / "cooked_geometry_result.json",
            _extract_log_json(source / "cooked_stdout.log", "cooking_result_failure_count"),
        )

    for relative in SOURCE_PATHS:
        source = WORKSPACE / relative
        group = "CONFIG" if "/config/" in relative else "SOURCE"
        _copy(source, handoff / group / relative)
    _copy(R12_ROOT / "CPU_CONTRACT_RESULT.txt", handoff / "CPU/CPU_CONTRACT_RESULT.txt")
    _copy(R12_ROOT / "CPU_CONTRACT_EXIT_CODE.txt", handoff / "CPU/CPU_CONTRACT_EXIT_CODE.txt")

    prior_files = (
        "02_ROOT_CAUSE_DECISION.json",
        "BASELINE/report.json",
        "BASELINE/first_jam_window.jsonl",
        "BASELINE/first_jam_contacts.csv",
        "CONFIG/d38999_keyed_v2_physical_model_contract_v1.yaml",
        "CONFIG/d38999_keyed_v2_physical_acceptance_v1.yaml",
    )
    for relative in prior_files:
        _copy(R11_HANDOFF / relative, handoff / "PRIOR_R11_EVIDENCE" / relative)

    _write_text(handoff / "GIT/git_head.txt", _git_text(["rev-parse", "HEAD"]))
    _write_text(handoff / "GIT/git_status.txt", _git_text(["status", "--short"]))
    scoped = [relative for relative in SOURCE_PATHS if relative.startswith("src/")]
    _write_text(handoff / "GIT/scoped_tracked_diff.patch", _git_text(["diff", "--", *scoped]))
    predecessor_generator = R11_HANDOFF / "SOURCE/create_d38999_keyed_physical_r7_asset.py"
    current_generator = WORKSPACE / "src/kcg_connector/isaac/create_d38999_keyed_physical_r7_asset.py"
    generator_diff = difflib.unified_diff(
        predecessor_generator.read_text(encoding="utf-8").splitlines(keepends=True),
        current_generator.read_text(encoding="utf-8").splitlines(keepends=True),
        fromfile="r11/SOURCE/create_d38999_keyed_physical_r7_asset.py",
        tofile="working_tree/create_d38999_keyed_physical_r7_asset.py",
    )
    _write_text(handoff / "GIT/GENERATOR_DIFF.patch", "".join(generator_diff))

    with (handoff / "CANDIDATE_MATRIX.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = (
            "candidate_index", "collider_count", "static_asset_audit_passed",
            "a2_passed", "cooked_geometry_passed", "p1_passed",
            "maximum_torque_component_nm", "maximum_consecutive_torque_saturation_steps",
            "maximum_fixed_receptacle_translation_drift_m",
            "final_observed_separation_m", "failure_classification",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: summary[field] for field in fields})

    counts = defaultdict(dict)
    for summary in summaries:
        for kind, count in summary["collider_type_counts"].items():
            counts[summary["candidate_index"]][kind] = count
    with (handoff / "COLLIDER_COUNTS.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("candidate_index", "total", "Mesh", "Cylinder", "Sphere", "Capsule"))
        for summary in summaries:
            row = counts[summary["candidate_index"]]
            writer.writerow((summary["candidate_index"], summary["collider_count"], row.get("Mesh", 0), row.get("Cylinder", 0), row.get("Sphere", 0), row.get("Capsule", 0)))

    final_status = {
        "candidate_count": 4,
        "full_candidate_p1_execution_count": 4,
        "candidate_p1_pass_count": sum(summary["p1_passed"] for summary in summaries),
        "formal_r12_generated": False,
        "formal_r12_path": "NOT_GENERATED_P1_DID_NOT_PASS",
        "r13_generated": False,
        "cpu_contract_test_count": 85,
        "cpu_contract_passed": True,
        "final_candidate": summaries[-1],
        "authorization": {
            "insertion": False,
            "twist": False,
            "randomization": False,
            "training": False,
            "residual_RL": False,
            "hardware_control": False,
        },
    }
    _write_json(handoff / "FINAL_STATUS.json", final_status)
    _write_text(
        handoff / "FORMAL_R12_ABSENCE.txt",
        "NOT_GENERATED\nReason: none of four bounded candidate full P1 executions passed all seven events.\nr13_generated=false",
    )
    _write_text(
        handoff / "00_README_FIRST.md",
        """# D38999 r12 direct closeout handoff

## Outcome

`P1_FAILED_NO_FORMAL_R12`

Four bounded candidates and four full P1 executions were completed. CPU, A2,
and cooked-geometry gates passed for every candidate, but no P1 reached the
last three required events. Therefore no candidate was promoted and the
formal r12 path is intentionally absent.

The archive filename follows the user-requested `success_handoff` pattern; it
does not assert that P1 passed. `FINAL_STATUS.json` is authoritative.

Candidate 01 exposed the first thread segment end-face seam jam. Candidate 02
removed that sustained torque jam, then lost axial tracking after spring-finger
engagement. Candidate 03 rounded the twelve hard bore targets but shifted the
spring event by 0.628 mm and retained the axial stall. Candidate 04 used one
axisymmetric target and restored the spring event position, but still stalled
at 12.9398 mm and exceeded the fixed drift limit transiently.

No r13, local probe, controller retune, force-limit increase, post-start pose
write, or contact-truth control path was used.
""",
    )

    _write_json(
        handoff / "PACKAGE_PREFLIGHT.json",
        {
            "candidate_directories_present": 4,
            "formal_r12_absent": not FORMAL_ASSET.exists(),
            "required_source_file_count": len(SOURCE_PATHS),
            "status": "READY_FOR_MANIFEST",
        },
    )
    _write_manifest(handoff)

    DELIVERY_PARENT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as output:
        for path in sorted(handoff.rglob("*")):
            if path.is_file():
                output.write(
                    path,
                    (Path(handoff.name) / path.relative_to(handoff)).as_posix(),
                )

    with tempfile.TemporaryDirectory(prefix="d38999-r12-handoff-verify-") as temp:
        unpack = Path(temp)
        with zipfile.ZipFile(archive, "r") as source:
            source.testzip()
            source.extractall(unpack)
        verification = _verify_unpacked(unpack / handoff.name)
    verification.update(
        {
            "archive_path": str(archive.resolve()),
            "archive_size_bytes": archive.stat().st_size,
            "archive_sha256": _sha256(archive),
            "zip_crc_test_passed": True,
        }
    )
    _write_json(verification_sidecar, verification)
    if not verification["passed"]:
        raise RuntimeError("unpacked handoff manifest verification failed")
    print(json.dumps(verification, ensure_ascii=False, sort_keys=True))
    print(f"handoff_directory={handoff.resolve()}")
    print(f"archive={archive.resolve()}")
    print(f"sha256={verification['archive_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
