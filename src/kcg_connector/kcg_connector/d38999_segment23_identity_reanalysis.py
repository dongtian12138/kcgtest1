#!/usr/bin/env python3

"""Geometry-anchored, fail-closed reanalysis of D38999 Segment_23.

The captured 24-colour HSV wheel is cyclic and the rendered hue of the rear
tooth nearest Segment_23 is quantized as Segment_22.  This CPU-only analysis
does not widen the hue gate or recolour pixels.  Instead, it binds the exact
asset, camera, parent-pose trace, manifests and PNGs, projects the authored
24-tooth geometry, and accepts an identity only when the observed component
and projected tooth are mutual nearest neighbours with a strict pitch margin.

Recovering an identity is not a visual no-jitter result.  The output keeps the
physics transform result, the post-hoc visual identity result, and the absent
render-jitter acceptance contract as three separate claims.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

import cv2
import numpy as np

from kcg_connector import d38999_tooth_sync_analysis as sync_analysis


SCHEMA_VERSION = "kcg_d38999_segment23_identity_reanalysis_v1"
MANIFEST_SCHEMA_VERSION = (
    "kcg_d38999_segment23_identity_reanalysis_manifest_v1"
)
UPSTREAM_SCHEMA_VERSION = "kcg_d38999_tooth_axial_evidence_v1"
UPSTREAM_MANIFEST_SCHEMA_VERSION = (
    "kcg_d38999_tooth_axial_evidence_manifest_v1"
)
AXIAL_CAPTURE_SCHEMA_VERSION = "kcg_d38999_tooth_axial_capture_v1"
TARGET_VIEW = "axial_segment23"
TARGET_SEGMENT = "Segment_23"
NEIGHBOUR_SEGMENTS = ("Segment_22", "Segment_00")
CAMERA_HORIZONTAL_APERTURE_MM = 20.955
EXPECTED_TOOTH_COUNT = 24
EXPECTED_TOOTH_RADIUS_M = 0.02175
EXPECTED_TOOTH_CENTER_Z_M = 0.0155
EXPECTED_TOOTH_SCALE_M = (0.0045, 0.0051816646, 0.017)
TARGET_MAXIMUM_PITCH_FRACTION = 1.0 / 3.0
TARGET_MINIMUM_MARGIN_PITCH_FRACTION = 0.50


class EvidenceError(RuntimeError):
    """Raised when an input or geometric identity gate is not satisfied."""


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 for one evidence file."""

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


def _repo_directory(
    path: str | Path, repository: Path, label: str
) -> Path:
    target = Path(path).expanduser().resolve()
    repository = Path(repository).expanduser().resolve()
    if not target.is_dir():
        raise EvidenceError(f"{label} is missing: {target}")
    if repository != target and repository not in target.parents:
        raise EvidenceError(f"{label} escapes repository: {target}")
    return target


def file_binding(path: str | Path, repository: Path) -> dict[str, Any]:
    """Bind one repository file by relative path, size, and SHA-256."""

    target = _repo_file(path, repository, "bound file")
    return {
        "path": str(target.relative_to(repository)),
        "sha256": sha256_file(target),
        "size_bytes": target.stat().st_size,
    }


def _validate_relative_binding(
    binding: Mapping[str, Any],
    expected_path: Path,
    repository: Path,
    label: str,
) -> None:
    if not isinstance(binding, Mapping):
        raise EvidenceError(f"{label} binding is not a mapping")
    target = _repo_file(expected_path, repository, label)
    try:
        bound = (repository / str(binding["path"])).resolve()
    except (KeyError, TypeError, ValueError) as error:
        raise EvidenceError(f"{label} binding path is invalid") from error
    if bound != target:
        raise EvidenceError(f"{label} binding path differs")
    if (
        binding.get("sha256") != sha256_file(target)
        or binding.get("size_bytes") != target.stat().st_size
    ):
        raise EvidenceError(f"{label} binding size/SHA differs")


def _balanced_block(text: str, marker: str, label: str) -> str:
    """Extract the first brace-balanced USDA block after ``marker``."""

    start = text.find(marker)
    if start < 0:
        raise EvidenceError(f"asset is missing {label}")
    opening = text.find("{", start + len(marker))
    if opening < 0:
        raise EvidenceError(f"asset {label} has no body")
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[opening + 1:index]
    raise EvidenceError(f"asset {label} block is unterminated")


def _number_tuple(block: str, attribute: str, count: int) -> tuple[float, ...]:
    match = re.search(
        rf"(?:float3|double3)\s+{re.escape(attribute)}\s*=\s*\(([^)]*)\)",
        block,
    )
    if match is None:
        raise EvidenceError(f"asset tooth is missing {attribute}")
    try:
        values = tuple(
            float(item.strip()) for item in match.group(1).split(",")
        )
    except ValueError as error:
        raise EvidenceError(f"asset {attribute} is not numeric") from error
    if len(values) != count or not all(math.isfinite(item) for item in values):
        raise EvidenceError(f"asset {attribute} has invalid arity/data")
    return values


def parse_authored_tooth_geometry(asset_path: str | Path) -> dict[str, Any]:
    """Parse and validate the exact cyclic CouplingNut tooth layout."""

    text = Path(asset_path).read_text(encoding="utf-8")
    nut = _balanced_block(text, 'def Xform "CouplingNut"', "CouplingNut")
    centers: dict[str, tuple[float, float, float]] = {}
    rotations = {}
    for index in range(EXPECTED_TOOTH_COUNT):
        name = f"Segment_{index:02d}"
        block = _balanced_block(nut, f'def Cube "{name}"', name)
        center = _number_tuple(block, "xformOp:translate", 3)
        scale = _number_tuple(block, "xformOp:scale", 3)
        if not np.allclose(
            scale,
            EXPECTED_TOOTH_SCALE_M,
            rtol=0.0,
            atol=1.0e-10,
        ):
            raise EvidenceError(f"asset {name} scale differs from contract")
        rotate = re.search(
            r"float\s+xformOp:rotateZ\s*=\s*([-+0-9.eE]+)", block
        )
        angle = 0.0 if rotate is None and index == 0 else None
        if rotate is not None:
            angle = float(rotate.group(1))
        if angle is None or not math.isclose(
            angle, 15.0 * index, rel_tol=0.0, abs_tol=1.0e-9
        ):
            raise EvidenceError(f"asset {name} rotation differs from index")
        radius = math.hypot(center[0], center[1])
        observed_angle = math.degrees(math.atan2(center[1], center[0])) % 360.0
        if (
            not math.isclose(
                radius,
                EXPECTED_TOOTH_RADIUS_M,
                rel_tol=0.0,
                abs_tol=1.0e-10,
            )
            or not math.isclose(
                center[2],
                EXPECTED_TOOTH_CENTER_Z_M,
                rel_tol=0.0,
                abs_tol=1.0e-10,
            )
            or not math.isclose(
                observed_angle,
                (15.0 * index) % 360.0,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
        ):
            raise EvidenceError(f"asset {name} center breaks cyclic order")
        centers[name] = center
        rotations[name] = angle
    if len(centers) != EXPECTED_TOOTH_COUNT:
        raise EvidenceError("asset does not contain exactly 24 indexed teeth")
    return {
        "centers_local_m": centers,
        "rotations_degrees": rotations,
        "segment_count": len(centers),
    }


def _camera_model(view: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "view_id": TARGET_VIEW,
        "target_segment": TARGET_SEGMENT,
        "eye_m": [0.66, 0.075, 0.47],
        "target_m": [0.55, 0.185, 0.28],
        "focal_length_mm": 50.0,
        "resolution": [960, 720],
        "fixed_before_play": True,
    }
    if any(view.get(key) != value for key, value in expected.items()):
        raise EvidenceError("Segment_23 axial camera contract differs")
    eye = np.asarray(view["eye_m"], dtype=float)
    target = np.asarray(view["target_m"], dtype=float)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray((0.0, 0.0, 1.0)))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    width, height = (int(item) for item in view["resolution"])
    focal_px = width * float(view["focal_length_mm"]) / (
        CAMERA_HORIZONTAL_APERTURE_MM
    )
    return {
        "eye": eye,
        "forward": forward,
        "right": right,
        "up": up,
        "focal_px": focal_px,
        "principal": np.asarray((width / 2.0, height / 2.0)),
    }


def _rotate_quaternion_wxyz(
    quaternion: np.ndarray, vector: np.ndarray
) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=float)
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm <= 0.0:
        raise EvidenceError("physics parent quaternion is invalid")
    quaternion = quaternion / norm
    scalar = quaternion[0]
    imaginary = quaternion[1:]
    return vector + 2.0 * np.cross(
        imaginary,
        np.cross(imaginary, vector) + scalar * vector,
    )


def project_tooth_centers(
    *,
    centers_local_m: Mapping[str, tuple[float, float, float]],
    parent_row: Mapping[str, Any],
    camera: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Project authored local centers using the recorded parent pose."""

    try:
        position = np.asarray(
            [
                float(parent_row[name])
                for name in ("parent_px_m", "parent_py_m", "parent_pz_m")
            ]
        )
        quaternion = np.asarray(
            [
                float(parent_row[name])
                for name in (
                    "parent_qw",
                    "parent_qx",
                    "parent_qy",
                    "parent_qz",
                )
            ]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise EvidenceError("physics parent pose is malformed") from error
    if not np.all(np.isfinite(position)) or not np.all(
        np.isfinite(quaternion)
    ):
        raise EvidenceError("physics parent pose is not finite")
    projected = {}
    for name, local in centers_local_m.items():
        world = position + _rotate_quaternion_wxyz(
            quaternion, np.asarray(local, dtype=float)
        )
        relative = world - camera["eye"]
        depth = float(np.dot(relative, camera["forward"]))
        if not math.isfinite(depth) or depth <= 0.0:
            raise EvidenceError("authored tooth projects behind axial camera")
        projected[name] = camera["principal"] + np.asarray(
            (
                camera["focal_px"]
                * float(np.dot(relative, camera["right"]))
                / depth,
                -camera["focal_px"]
                * float(np.dot(relative, camera["up"]))
                / depth,
            )
        )
    return projected


def mutual_nearest_assignment(
    *,
    projected: Mapping[str, np.ndarray],
    observed: Mapping[str, np.ndarray],
    target: str = TARGET_SEGMENT,
) -> dict[str, Any]:
    """Assign ``target`` without treating an observed hue label as identity."""

    if target not in projected or not observed:
        raise EvidenceError("target projection or observed components missing")
    target_point = np.asarray(projected[target], dtype=float)
    observed_distances = sorted(
        (
            float(np.linalg.norm(np.asarray(point) - target_point)),
            label,
        )
        for label, point in observed.items()
    )
    candidate_distance, candidate_label = observed_distances[0]
    candidate = np.asarray(observed[candidate_label], dtype=float)
    projected_distances = sorted(
        (
            float(np.linalg.norm(point - candidate)),
            label,
        )
        for label, point in projected.items()
    )
    if projected_distances[0][1] != target:
        raise EvidenceError("Segment_23 correspondence is not mutual nearest")
    neighbour_pitch = min(
        float(np.linalg.norm(target_point - projected[name]))
        for name in NEIGHBOUR_SEGMENTS
    )
    if not math.isfinite(neighbour_pitch) or neighbour_pitch <= 0.0:
        raise EvidenceError("projected Segment_23 pitch is invalid")
    error_fraction = candidate_distance / neighbour_pitch
    projected_margin = (
        projected_distances[1][0] - projected_distances[0][0]
    )
    margin_fraction = projected_margin / neighbour_pitch
    if error_fraction >= TARGET_MAXIMUM_PITCH_FRACTION:
        raise EvidenceError(
            "Segment_23 projection error reaches one-third pitch"
        )
    if margin_fraction <= TARGET_MINIMUM_MARGIN_PITCH_FRACTION:
        raise EvidenceError(
            "Segment_23 projected identity margin is too small"
        )
    return {
        "candidate_hue_label": candidate_label,
        "candidate_xy_px": [float(item) for item in candidate],
        "projected_xy_px": [float(item) for item in target_point],
        "projection_error_px": candidate_distance,
        "projection_error_pitch_fraction": error_fraction,
        "projected_identity_margin_px": projected_margin,
        "projected_identity_margin_pitch_fraction": margin_fraction,
        "projected_neighbour_pitch_px": neighbour_pitch,
    }


def _read_physics_rows(path: Path) -> dict[int, dict[str, str]]:
    required = {
        "global_step",
        "phase",
        "phase_step",
        "parent_px_m",
        "parent_py_m",
        "parent_pz_m",
        "parent_qw",
        "parent_qx",
        "parent_qy",
        "parent_qz",
    }
    rows = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not required.issubset(reader.fieldnames or ()):
            raise EvidenceError("physics summary lacks parent-pose columns")
        for row in reader:
            try:
                step = int(row["global_step"])
            except (TypeError, ValueError) as error:
                raise EvidenceError(
                    "physics global step is invalid"
                ) from error
            if step in rows:
                raise EvidenceError("physics summary repeats a global step")
            rows[step] = row
    if not rows:
        raise EvidenceError("physics summary is empty")
    return rows


def _read_axial_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != sync_analysis.SYNC_FIELDS:
            raise EvidenceError("axial sync columns differ from contract")
        raw = list(reader)
    rows = []
    for item in raw:
        try:
            rows.append(
                {
                    **item,
                    "frame_index": int(item["frame_index"]),
                    "sample_index": int(item["sample_index"]),
                    "global_step": int(item["global_step"]),
                    "phase_step": int(item["phase_step"]),
                }
            )
        except (KeyError, TypeError, ValueError) as error:
            raise EvidenceError("axial sync row is malformed") from error
    if not rows:
        raise EvidenceError("axial sync CSV is empty")
    return rows


def _load_main_report(run_log: Path) -> dict[str, Any]:
    reports = []
    content = run_log.read_text(encoding="utf-8", errors="strict")
    for line in content.splitlines():
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("scene") == (
            "kcg_d38999_nut_regrasp_physx_v1"
        ):
            reports.append(value)
    if len(reports) != 1:
        raise EvidenceError("run log lacks exactly one prepared main report")
    return reports[0]


def _statistics(values: list[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=float)
    if not len(data) or not np.all(np.isfinite(data)):
        raise EvidenceError("cannot summarize empty or non-finite values")
    return {
        "maximum": float(np.max(data)),
        "median": float(np.median(data)),
        "minimum": float(np.min(data)),
    }


def _validate_upstream(
    *, repository: Path, run_root: Path, evidence_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = evidence_root / "manifest.json"
    report_path = evidence_root / "report.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != UPSTREAM_MANIFEST_SCHEMA_VERSION
        or manifest.get("status") != "HASH_SIZE_SCHEMA_BOUND"
        or report.get("schema_version") != UPSTREAM_SCHEMA_VERSION
        or report.get("evidence_valid") is not True
    ):
        raise EvidenceError("upstream six-view evidence is not valid/bound")
    _validate_relative_binding(
        manifest.get("outputs", {}).get("report", {}),
        report_path,
        repository,
        "upstream report",
    )
    expected_input_paths = {
        "axial_capture_manifest": run_root
        / "axial/axial_capture_manifest.json",
        "physics_report": run_root / "physics/report.json",
        "physics_summary": run_root / "physics/summary.csv",
        "run_log": run_root / "run.log",
    }
    for name, path in expected_input_paths.items():
        _validate_relative_binding(
            manifest.get("inputs", {}).get(name, {}),
            path,
            repository,
            f"upstream {name}",
        )
    expected_source_paths = {
        "axial_capture": repository
        / "src/kcg_connector/isaac/d38999_tooth_axial_capture.py",
        "base_analysis": Path(sync_analysis.__file__).resolve(),
    }
    for name, path in expected_source_paths.items():
        _validate_relative_binding(
            manifest.get("sources", {}).get(name, {}),
            path,
            repository,
            f"upstream {name} source",
        )
    coverage = report.get("visual", {}).get("six_view_coverage", {})
    if (
        coverage.get("missing_from_identity_union") != [TARGET_SEGMENT]
        or coverage.get("per_segment_transition_counts", {}).get(
            TARGET_SEGMENT
        )
        != 0
        or report.get("physics", {}).get("anomaly_steps") != 0
        or report.get("visual", {}).get(
            "render_jitter_absence_claim_authorized"
        )
        is not False
    ):
        raise EvidenceError("upstream Segment_23 evidence boundary differs")
    return manifest, report


def reanalyze_segment23_identity(
    *,
    repository: str | Path,
    run_root: str | Path,
    upstream_evidence_root: str | Path,
    connector_asset: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Recover Segment_23 identity from existing, hash-bound RGB evidence."""

    repository = Path(repository).expanduser().resolve()
    run_root = _repo_directory(run_root, repository, "six-view run root")
    evidence_root = _repo_directory(
        upstream_evidence_root, repository, "upstream evidence root"
    )
    asset = _repo_file(connector_asset, repository, "connector asset")
    output = Path(output_directory).expanduser().resolve()
    if output.exists():
        raise EvidenceError(f"output already exists: {output}")
    if repository != output and repository not in output.parents:
        raise EvidenceError("output escapes repository")

    upstream_manifest, upstream_report = _validate_upstream(
        repository=repository,
        run_root=run_root,
        evidence_root=evidence_root,
    )
    del upstream_report
    axial_manifest_path = run_root / "axial/axial_capture_manifest.json"
    axial_manifest = json.loads(
        axial_manifest_path.read_text(encoding="utf-8")
    )
    if (
        axial_manifest.get("schema_version")
        != AXIAL_CAPTURE_SCHEMA_VERSION
        or axial_manifest.get("passed") is not True
    ):
        raise EvidenceError("axial capture manifest is not passed")
    views = axial_manifest.get("camera_rig", {}).get("views", [])
    matching_views = [
        view for view in views if view.get("view_id") == TARGET_VIEW
    ]
    if len(matching_views) != 1:
        raise EvidenceError("axial Segment_23 camera declaration differs")
    camera = _camera_model(matching_views[0])

    physics_report_path = run_root / "physics/report.json"
    physics_summary_path = run_root / "physics/summary.csv"
    run_log_path = run_root / "run.log"
    physics_report = json.loads(
        physics_report_path.read_text(encoding="utf-8")
    )
    main_report = _load_main_report(run_log_path)
    asset_sha = sha256_file(asset)
    if (
        main_report.get("d38999_authoring", {}).get("asset_sha256")
        != asset_sha
        or main_report.get("nut_tooth_jitter_probe", {}).get(
            "anomaly_steps"
        )
        != 0
        or physics_report.get("anomaly_steps") != 0
    ):
        raise EvidenceError("asset or zero-anomaly run provenance differs")
    segment_aggregate = physics_report.get("segment_aggregate", {})
    expected_segments = {
        f"Segment_{index:02d}" for index in range(EXPECTED_TOOTH_COUNT)
    }
    if set(segment_aggregate) != expected_segments:
        raise EvidenceError("physics report does not cover all 24 teeth")
    segment23_physics = segment_aggregate.get(TARGET_SEGMENT, {})
    relative_fields = (
        "maximum_local_rotation_error_rad",
        "maximum_local_translation_error_m",
        "maximum_parent_relative_rotation_error_rad",
        "maximum_parent_relative_translation_error_m",
    )
    if any(segment23_physics.get(name) != 0.0 for name in relative_fields):
        raise EvidenceError("physics trace reports Segment_23 relative motion")
    geometry = parse_authored_tooth_geometry(asset)
    palette = physics_report.get("color_identification", {}).get(
        "colors_rgb"
    )
    if palette != sync_analysis.deterministic_segment_colors():
        raise EvidenceError("captured 24-colour palette differs")

    sync_path = run_root / "axial/video_frame_sync.csv"
    if sha256_file(sync_path) != axial_manifest.get("sync_csv_sha256"):
        raise EvidenceError("axial sync CSV SHA differs")
    all_rows = _read_axial_rows(sync_path)
    target_rows = [row for row in all_rows if row["view_id"] == TARGET_VIEW]
    frame_summary = axial_manifest.get("frame_capture", {})
    if (
        len(target_rows)
        != frame_summary.get("frames_per_view", {}).get(TARGET_VIEW)
        or len(target_rows) != frame_summary.get("sample_count")
        or [row["sample_index"] for row in target_rows]
        != list(range(len(target_rows)))
    ):
        raise EvidenceError("axial Segment_23 frame schedule differs")
    frame_hashes = axial_manifest.get("frame_files_sha256", {})
    if not isinstance(frame_hashes, Mapping):
        raise EvidenceError("axial PNG hash map is malformed")
    physics_rows = _read_physics_rows(physics_summary_path)

    assignments = []
    previous_residual = None
    phase_residual_steps: dict[str, list[float]] = {}
    for row in target_rows:
        relative = Path(row["rgb_filename"])
        if relative.is_absolute() or ".." in relative.parts:
            raise EvidenceError("axial PNG path is not capture-relative")
        frame_path = (run_root / "axial" / relative).resolve()
        if run_root / "axial" not in frame_path.parents:
            raise EvidenceError("axial PNG escapes capture directory")
        expected_hash = frame_hashes.get(row["rgb_filename"])
        if expected_hash is None or sha256_file(frame_path) != expected_hash:
            raise EvidenceError("axial Segment_23 PNG SHA differs")
        parent = physics_rows.get(row["global_step"])
        if parent is None or (
            parent["phase"] != row["phase"]
            or int(parent["phase_step"]) != row["phase_step"]
        ):
            raise EvidenceError("axial frame and physics pose key differ")
        image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        try:
            observed, areas = sync_analysis.extract_tooth_centroids(
                image, palette
            )
        except sync_analysis.EvidenceError as error:
            raise EvidenceError(
                "existing Segment_23 frame has no usable components"
            ) from error
        projected = project_tooth_centers(
            centers_local_m=geometry["centers_local_m"],
            parent_row=parent,
            camera=camera,
        )
        assignment = mutual_nearest_assignment(
            projected=projected,
            observed=observed,
        )
        assignment.update(
            {
                "candidate_area_px": int(
                    areas[assignment["candidate_hue_label"]]
                ),
                "global_step": row["global_step"],
                "phase": row["phase"],
                "phase_step": row["phase_step"],
                "sample_index": row["sample_index"],
            }
        )
        residual = np.asarray(assignment["candidate_xy_px"]) - np.asarray(
            assignment["projected_xy_px"]
        )
        if (
            previous_residual is not None
            and previous_residual[0] == row["phase"]
        ):
            phase_residual_steps.setdefault(row["phase"], []).append(
                float(np.linalg.norm(residual - previous_residual[1]))
            )
        previous_residual = (row["phase"], residual)
        assignments.append(assignment)

    hue_labels = sorted(
        {assignment["candidate_hue_label"] for assignment in assignments}
    )
    if hue_labels == [TARGET_SEGMENT]:
        raise EvidenceError(
            "geometry reanalysis did not encounter hue aliasing"
        )
    errors = [item["projection_error_px"] for item in assignments]
    error_fractions = [
        item["projection_error_pitch_fraction"] for item in assignments
    ]
    margins = [
        item["projected_identity_margin_pitch_fraction"]
        for item in assignments
    ]
    areas = [float(item["candidate_area_px"]) for item in assignments]
    report = {
        "schema_version": SCHEMA_VERSION,
        "classification": (
            "VALID_GEOMETRY_RECOVERED_SEGMENT23_IDENTITY_"
            "RENDER_JITTER_UNRESOLVED"
        ),
        "evidence_valid": True,
        "identity_result": {
            "segment": TARGET_SEGMENT,
            "existing_frame_count": len(assignments),
            "all_frames_mutual_nearest": True,
            "all_frames_within_one_third_projected_pitch": True,
            "all_frames_with_more_than_half_pitch_identity_margin": True,
            "geometry_identity_recovered": True,
            "hue_identity_recovered": False,
            "candidate_hue_labels": hue_labels,
            "projection_error_px": _statistics(errors),
            "projection_error_pitch_fraction": _statistics(error_fractions),
            "projected_identity_margin_pitch_fraction": _statistics(margins),
            "candidate_area_px": _statistics(areas),
            "method": (
                "asset_local_centers_plus_recorded_parent_pose_plus_fixed_"
                "camera_mutual_nearest_with_projected_pitch_margin"
            ),
            "modality": (
                "physics_and_CAD_projection_assisted_posthoc_identity_"
                "recovery_not_RGB_only"
            ),
            "per_frame_actual_physical_parent_pose_used": True,
            "correspondence_gate": {
                "assignment": (
                    "observed_component_nearest_to_projected_Segment23_and_"
                    "Segment23_nearest_projected_center_to_that_component"
                ),
                "conflict_policy": "reject_on_any_non_mutual_frame",
                "maximum_projection_error_pitch_fraction_exclusive": (
                    TARGET_MAXIMUM_PITCH_FRACTION
                ),
                "minimum_projected_identity_margin_pitch_fraction_"
                "exclusive": TARGET_MINIMUM_MARGIN_PITCH_FRACTION,
                "pitch_definition": (
                    "minimum_projected_center_distance_from_Segment23_to_"
                    "authored_cyclic_neighbours_Segment22_and_Segment00"
                ),
            },
            "hue_gate_widened": False,
            "pixels_relabelled_or_modified": False,
        },
        "physics_result": {
            "all_24_segments_one_rigid_parent_trace": True,
            "anomaly_steps": 0,
            "independent_physical_segment23_motion_observed": False,
            "segment23_maximum_relative_errors": {
                name: segment23_physics[name] for name in relative_fields
            },
        },
        "visual_diagnostics_only": {
            "phase_local_residual_step_px": {
                phase: _statistics(values)
                for phase, values in sorted(phase_residual_steps.items())
                if values
            },
            "render_jitter_absence_claim_authorized": False,
            "identity_match_is_render_no_jitter_claim": False,
            "reason": (
                "identity thresholds were designed posthoc and 30Hz RGB "
                "cannot exclude between-sample or sub-centroid render jitter"
            ),
        },
        "fresh_capture_recommendation": {
            "required_for_identity_only": False,
            "required_for_strong_visual_jitter_exclusion": True,
            "minimum_change": (
                "presentation-only non-cyclic 24-code codebook or integer "
                "instance-ID annotator at the same completed physics steps"
            ),
            "preserve": [
                "existing physics runner",
                "object and robot pose write count",
                "fixed camera timing",
                "per-PNG hash and sync bindings",
            ],
        },
        "limitations": [
            "posthoc_geometry_identity_is_not_preregistered_jitter_acceptance",
            "centroid_tracking_does_not_test_every_rendered_edge_pixel",
            "30_hz_sampling_cannot_exclude_between_sample_render_artifacts",
            "physics_zero_anomaly_does_not_prove_renderer_zero_jitter",
        ],
    }

    output.mkdir(parents=True, exist_ok=False)
    assignments_path = output / "segment23_assignments.csv"
    with assignments_path.open("w", encoding="utf-8", newline="") as stream:
        fields = list(assignments[0])
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in assignments:
            writer.writerow(item)
    report_path = output / "report.json"
    report_path.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "HASH_SIZE_SCHEMA_BOUND",
        "inputs": {
            "axial_capture_manifest": file_binding(
                axial_manifest_path, repository
            ),
            "connector_asset": file_binding(asset, repository),
            "physics_report": file_binding(physics_report_path, repository),
            "physics_summary": file_binding(physics_summary_path, repository),
            "run_log": file_binding(run_log_path, repository),
            "upstream_manifest": file_binding(
                evidence_root / "manifest.json", repository
            ),
            "upstream_report": file_binding(
                evidence_root / "report.json", repository
            ),
        },
        "indirect_frame_binding": {
            "mechanism": "axial_manifest_per_png_sha256_map",
            "segment23_png_hashes_revalidated": len(assignments),
            "sync_csv_sha256_revalidated": True,
        },
        "outputs": {
            "assignments": file_binding(assignments_path, repository),
            "report": file_binding(report_path, repository),
        },
        "sources": {
            "identity_reanalysis": file_binding(Path(__file__), repository),
            "upstream_base_analysis": upstream_manifest["sources"][
                "base_analysis"
            ],
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
    base = repository / "artifacts/kcg_connector/d38999_nut_tooth_jitter"
    parser = argparse.ArgumentParser(
        description="Reanalyze existing Segment_23 RGB identity by geometry"
    )
    parser.add_argument("--repository", type=Path, default=repository)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--upstream-evidence-root", type=Path, required=True)
    parser.add_argument(
        "--connector-asset",
        type=Path,
        default=(
            repository
            / "artifacts/kcg_connector/isaac/"
            "d38999_shell25j_61_pair_proxy_v1.usda"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=base / "segment23_identity_reanalysis_v1",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    arguments = _arguments(argv)
    try:
        result = reanalyze_segment23_identity(
            repository=arguments.repository,
            run_root=arguments.run_root,
            upstream_evidence_root=arguments.upstream_evidence_root,
            connector_asset=arguments.connector_asset,
            output_directory=arguments.output,
        )
    except (EvidenceError, OSError, ValueError) as error:
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
    "mutual_nearest_assignment",
    "parse_authored_tooth_geometry",
    "project_tooth_centers",
    "reanalyze_segment23_identity",
]
