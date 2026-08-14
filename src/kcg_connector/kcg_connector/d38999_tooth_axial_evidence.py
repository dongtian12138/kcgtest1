#!/usr/bin/env python3

"""Fail-closed six-view evidence for the D38999 tooth jitter probe.

The Isaac extension records two steep axial views on the exact sample keys of
the existing four-view ghost-fingers capture.  This CPU-only postprocessor
revalidates every source, manifest, CSV and PNG hash before reusing the
existing color-centroid and similarity-fit implementation.  Identity-union
coverage is intentionally kept separate from coverage at every transition;
neither result is interpreted as proof that render jitter is absent.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from kcg_connector import d38999_tooth_occlusion_evidence as occlusion
from kcg_connector import d38999_tooth_sync_analysis as analysis
from kcg_connector import d38999_tooth_sync_evidence as sync_evidence


SCHEMA_VERSION = "kcg_d38999_tooth_axial_evidence_v1"
MANIFEST_SCHEMA_VERSION = "kcg_d38999_tooth_axial_evidence_manifest_v1"
AXIAL_CAPTURE_SCHEMA_VERSION = "kcg_d38999_tooth_axial_capture_v1"
AXIAL_BUNDLE_SCHEMA_VERSION = "kcg_d38999_tooth_axial_ghost_bundle_v1"
AXIAL_VIEW_IDS = ("axial_segment13", "axial_segment23")
ALL_VIEW_IDS = (*analysis.VIEW_IDS, *AXIAL_VIEW_IDS)
TARGET_BY_VIEW = {
    "axial_segment13": "Segment_13",
    "axial_segment23": "Segment_23",
}
EXPECTED_SEGMENTS = tuple(f"Segment_{index:02d}" for index in range(24))
EXPECTED_SEGMENT_SET = set(EXPECTED_SEGMENTS)


class EvidenceError(RuntimeError):
    """Raised when any six-view input is missing or not attributable."""


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


def _binding(path: str | Path, repository: Path) -> dict[str, Any]:
    return occlusion.file_binding(path, repository)


def _validate_binding(
    binding: Mapping[str, Any], path: Path, label: str
) -> None:
    """Validate an absolute capture binding without trusting its path."""

    occlusion.validate_external_binding(binding, path, label)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    """Parse the shared sync schema and reject malformed numeric values."""

    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != analysis.SYNC_FIELDS:
            raise EvidenceError("axial sync CSV columns differ from contract")
        raw_rows = list(reader)
    rows = []
    for row in raw_rows:
        try:
            rows.append(
                {
                    "frame_index": int(row["frame_index"]),
                    "sample_index": int(row["sample_index"]),
                    "view_id": row["view_id"],
                    "global_step": int(row["global_step"]),
                    "phase": row["phase"],
                    "phase_step": int(row["phase_step"]),
                    "simulation_time_s": float(row["simulation_time_s"]),
                    "timestamp_s": float(row["timestamp_s"]),
                    "rgb_filename": row["rgb_filename"],
                }
            )
        except (KeyError, TypeError, ValueError) as error:
            raise EvidenceError(
                "axial sync CSV contains invalid data"
            ) from error
    if not rows:
        raise EvidenceError("axial sync CSV is empty")
    return rows


def _validate_axial_rig(manifest: Mapping[str, Any]) -> None:
    """Require the two declared targets and immutable fixed-camera geometry."""

    rig = manifest.get("camera_rig", {})
    if rig.get("view_priority") != list(AXIAL_VIEW_IDS):
        raise EvidenceError("axial view priority differs from contract")
    if rig.get("same_completed_physics_step_as_base_four_views") is not True:
        raise EvidenceError("axial rig is not declared same-step")
    if rig.get("axis_semantics") != (
        "prepared CouplingNut local +Z equals world +Z"
    ):
        raise EvidenceError("axial rig axis semantics differ")
    views = rig.get("views")
    if not isinstance(views, list) or len(views) != len(AXIAL_VIEW_IDS):
        raise EvidenceError("axial camera list differs from contract")
    expected = {
        "axial_segment13": {
            "eye_m": [0.44, 0.075, 0.47],
            "segment_angle_degrees": 195,
            "target_segment": "Segment_13",
        },
        "axial_segment23": {
            "eye_m": [0.66, 0.075, 0.47],
            "segment_angle_degrees": 345,
            "target_segment": "Segment_23",
        },
    }
    for view in views:
        view_id = view.get("view_id")
        if view_id not in expected:
            raise EvidenceError("axial camera ID differs from contract")
        required = {
            **expected[view_id],
            "fixed_before_play": True,
            "focal_length_mm": 50,
            "resolution": [960, 720],
            "target_m": [0.55, 0.185, 0.28],
        }
        if any(view.get(key) != value for key, value in required.items()):
            raise EvidenceError(f"{view_id} geometry differs from contract")
        exposure = view.get("analytic_target_exposure", {})
        if (
            not math.isfinite(exposure.get("axial_top_face_cosine", math.nan))
            or exposure["axial_top_face_cosine"] <= 0.77
            or not math.isfinite(
                exposure.get("radial_outer_face_cosine", math.nan)
            )
            or exposure["radial_outer_face_cosine"] <= 0.54
        ):
            raise EvidenceError(f"{view_id} analytic exposure gate failed")


def _validate_axial_capture(
    *,
    repository: Path,
    run_root: Path,
    base_bundle: Mapping[str, Any],
    axial_helper: Path,
    wrapper: Path,
    runner: Path,
) -> dict[str, Any]:
    """Validate the axial manifest, schedule and all 530 PNG hashes."""

    axial_root = run_root / "axial"
    manifest_path = axial_root / "axial_capture_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != AXIAL_CAPTURE_SCHEMA_VERSION:
        raise EvidenceError("axial capture manifest schema differs")
    if manifest.get("passed") is not True:
        raise EvidenceError("axial capture manifest is not passed")
    _validate_axial_rig(manifest)

    source = manifest.get("capture_source", {})
    helper_sha = occlusion.sha256_file(axial_helper)
    if not (
        source.get("sha256_at_import")
        == source.get("sha256_at_start")
        == source.get("sha256_at_finalize")
        == helper_sha
    ):
        raise EvidenceError("axial capture source provenance differs")
    if Path(source.get("path", "")).resolve() != axial_helper:
        raise EvidenceError("axial capture source path differs")
    provenance = manifest.get("provenance", {})
    _validate_binding(provenance.get("prepared_runner", {}), runner, "runner")
    _validate_binding(provenance.get("wrapper", {}), wrapper, "wrapper")
    if (
        provenance.get("runner_sha256_at_start")
        != occlusion.sha256_file(runner)
        or provenance.get("wrapper_sha256_at_start")
        != occlusion.sha256_file(wrapper)
        or provenance.get("unchanged_during_capture") is not True
    ):
        raise EvidenceError("axial runner/wrapper provenance differs")

    cleanup = manifest.get("cleanup", {})
    required_cleanup = {
        "annotators_detached_count": 2,
        "camera_prims_removed_count": 2,
        "errors": [],
        "object_pose_writes": 0,
        "physics_steps": 0,
        "render_products_destroyed_count": 2,
        "resources_released": True,
        "view_count": 2,
    }
    if any(
        cleanup.get(key) != value
        for key, value in required_cleanup.items()
    ):
        raise EvidenceError("axial cleanup or zero-write gate failed")
    if manifest.get("same_sample_keys_as_base_four_views") is not True:
        raise EvidenceError("axial manifest does not assert equal sample keys")
    if manifest.get("render_settings") != base_bundle["manifest"].get(
        "render_settings"
    ):
        raise EvidenceError("axial/base render settings differ")

    base_manifest_path = run_root / "capture/video_capture_manifest.json"
    _validate_binding(
        manifest.get("base_four_view_binding", {}),
        base_manifest_path,
        "axial base capture",
    )
    physics = manifest.get("physics_evidence", {})
    _validate_binding(
        physics.get("report", {}),
        Path(base_bundle["physics_report_path"]),
        "axial physics report",
    )
    _validate_binding(
        physics.get("summary", {}),
        Path(base_bundle["physics_summary_path"]),
        "axial physics summary",
    )

    sampling = manifest.get("sampling", {})
    if sampling != base_bundle["manifest"].get("sampling"):
        raise EvidenceError("axial/base sampling contracts differ")
    if tuple(manifest.get("sync_columns", ())) != analysis.SYNC_FIELDS:
        raise EvidenceError("axial manifest sync columns differ")
    if manifest.get("sync_csv") != "video_frame_sync.csv":
        raise EvidenceError("axial sync filename differs")
    sync_path = axial_root / "video_frame_sync.csv"
    if occlusion.sha256_file(sync_path) != manifest.get("sync_csv_sha256"):
        raise EvidenceError("axial sync CSV SHA differs")
    rows = _read_rows(sync_path)
    frame_hashes = manifest.get("frame_files_sha256", {})
    if not isinstance(frame_hashes, Mapping):
        raise EvidenceError("axial PNG hash map is invalid")
    if set(frame_hashes) != {row["rgb_filename"] for row in rows}:
        raise EvidenceError("axial PNG set differs from sync CSV")

    samples = []
    previous_timestamp = -math.inf
    for expected_index, row in enumerate(rows):
        view_id = AXIAL_VIEW_IDS[expected_index % len(AXIAL_VIEW_IDS)]
        sample_index = expected_index // len(AXIAL_VIEW_IDS)
        if (
            row["frame_index"] != expected_index
            or row["sample_index"] != sample_index
            or row["view_id"] != view_id
        ):
            raise EvidenceError("axial fixed view order is incomplete")
        if (
            not math.isfinite(row["timestamp_s"])
            or row["timestamp_s"] <= previous_timestamp
        ):
            raise EvidenceError("axial timestamps are not increasing")
        expected_time = row["global_step"] / float(
            sampling["physics_rate_hz"]
        )
        if not math.isclose(
            row["simulation_time_s"], expected_time, rel_tol=0.0, abs_tol=1e-12
        ):
            raise EvidenceError("axial simulation time differs from step")
        relative = Path(row["rgb_filename"])
        if relative.is_absolute() or ".." in relative.parts:
            raise EvidenceError("axial PNG path is not capture-relative")
        frame_path = (axial_root / relative).resolve()
        if axial_root not in frame_path.parents or not frame_path.is_file():
            raise EvidenceError("axial PNG is missing or escapes capture")
        if (
            occlusion.sha256_file(frame_path)
            != frame_hashes[row["rgb_filename"]]
        ):
            raise EvidenceError("axial PNG SHA differs")
        row["frame_path"] = frame_path
        if view_id == AXIAL_VIEW_IDS[0]:
            samples.append(
                {
                    "global_step": row["global_step"],
                    "phase": row["phase"],
                    "phase_step": row["phase_step"],
                    "sample_index": row["sample_index"],
                    "simulation_time_s": row["simulation_time_s"],
                    "views": {view_id: row},
                }
            )
        else:
            sample = samples[-1]
            for key in (
                "global_step",
                "phase",
                "phase_step",
                "sample_index",
                "simulation_time_s",
            ):
                if sample[key] != row[key]:
                    raise EvidenceError(
                        "axial views have different sample keys"
                    )
            sample["views"][view_id] = row
        previous_timestamp = row["timestamp_s"]
    if len(rows) % len(AXIAL_VIEW_IDS) or any(
        tuple(sample["views"]) != AXIAL_VIEW_IDS for sample in samples
    ):
        raise EvidenceError("axial final sample is incomplete")

    base_keys = [
        (
            sample["global_step"],
            sample["phase"],
            sample["phase_step"],
            sample["sample_index"],
            sample["simulation_time_s"],
        )
        for sample in base_bundle["samples"]
    ]
    axial_keys = [
        (
            sample["global_step"],
            sample["phase"],
            sample["phase_step"],
            sample["sample_index"],
            sample["simulation_time_s"],
        )
        for sample in samples
    ]
    if axial_keys != base_keys:
        raise EvidenceError("axial/base sample keys differ on revalidation")
    expected_frame_capture = {
        "first_global_step": samples[0]["global_step"],
        "frame_count": len(rows),
        "frames_per_view": {view: len(samples) for view in AXIAL_VIEW_IDS},
        "last_global_step": samples[-1]["global_step"],
        "phases": list(analysis.CAPTURE_PHASES),
        "sample_count": len(samples),
        "view_order": list(AXIAL_VIEW_IDS),
    }
    if manifest.get("frame_capture") != expected_frame_capture:
        raise EvidenceError("axial frame summary differs from rows")
    actual_pngs = {
        str(path.relative_to(axial_root))
        for path in (axial_root / "frames").rglob("*.png")
    }
    if actual_pngs != set(frame_hashes):
        raise EvidenceError("axial frames contain missing or stale PNGs")
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "rows": rows,
        "samples": samples,
    }


def _validate_posthoc_bundle(
    *, repository: Path, run_root: Path, axial_manifest_path: Path
) -> dict[str, Any]:
    """Validate the pure post-run axial/ghost binding artifact."""

    path = run_root / "axial/axial_ghost_bundle_manifest.json"
    if not path.is_file():
        raise EvidenceError("posthoc axial ghost bundle is missing")
    bundle = json.loads(path.read_text(encoding="utf-8"))
    if bundle.get("schema_version") != AXIAL_BUNDLE_SCHEMA_VERSION:
        raise EvidenceError("axial ghost bundle schema differs")
    required_true = (
        "passed",
        "same_base_capture",
        "same_prepared_runner",
        "visibility_only_zero_physics_or_pose_writes",
    )
    if any(bundle.get(key) is not True for key in required_true):
        raise EvidenceError("axial ghost bundle gate failed")
    inputs = bundle.get("inputs", {})
    expected = {
        "axial_capture_manifest": axial_manifest_path,
        "base_four_view_capture_manifest": (
            run_root / "capture/video_capture_manifest.json"
        ),
        "ghost_manifest": run_root / "ghost/manifest.json",
        "ghost_visibility_sidecar": run_root / "ghost/visibility_sidecar.json",
    }
    if set(inputs) != set(expected):
        raise EvidenceError("axial ghost bundle input names differ")
    for name, input_path in expected.items():
        _validate_binding(inputs[name], input_path, name)
    return {"bundle": bundle, "path": path}


def _extract_axial_frames(
    samples: list[dict[str, Any]], palette: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the existing centroid/pitch gates without requiring both views."""

    frames = []
    valid_counts = Counter()
    target_frame_counts = Counter()
    target_color_components: dict[str, list[dict[str, Any]]] = {
        view_id: [] for view_id in AXIAL_VIEW_IDS
    }
    errors = {view_id: Counter() for view_id in AXIAL_VIEW_IDS}
    for sample in samples:
        frame = {**sample, "views": {}, "view_errors": {}}
        for view_id in AXIAL_VIEW_IDS:
            row = sample["views"][view_id]
            image = cv2.imread(str(row["frame_path"]), cv2.IMREAD_COLOR)
            target_color_components[view_id].extend(
                _target_color_components(
                    image,
                    palette,
                    TARGET_BY_VIEW[view_id],
                    sample_index=sample["sample_index"],
                )
            )
            try:
                points, areas = analysis.extract_tooth_centroids(
                    image, palette
                )
                points, pitches = analysis.select_pitch_qualified_cluster(
                    points
                )
            except analysis.EvidenceError as error:
                message = str(error)
                frame["view_errors"][view_id] = message
                errors[view_id][message] += 1
                continue
            frame["views"][view_id] = {
                "areas": {name: areas[name] for name in points},
                "frame_path": row["frame_path"],
                "points": points,
                "pitches": pitches,
                "rgb_filename": row["rgb_filename"],
            }
            valid_counts[view_id] += 1
            if TARGET_BY_VIEW[view_id] in points:
                target_frame_counts[view_id] += 1
        frames.append(frame)
    color_profile = {}
    for view_id in AXIAL_VIEW_IDS:
        components = target_color_components[view_id]
        areas = np.asarray(
            [component["area_px"] for component in components], dtype=float
        )
        qualified = target_frame_counts[view_id]
        color_profile[view_id] = {
            "broad_hue_component_count": len(components),
            "broad_hue_component_frames": len(
                {component["sample_index"] for component in components}
            ),
            "broad_hue_component_area_px": (
                {
                    "maximum": float(np.max(areas)),
                    "median": float(np.median(areas)),
                    "minimum": float(np.min(areas)),
                }
                if len(areas)
                else None
            ),
            "qualified_identity_frames": qualified,
            "diagnostic_only": True,
            "adjacent_hue_overlap_can_make_broad_mask_non_identifying": True,
            "detector_geometry_or_identity_mismatch": bool(
                components and qualified == 0
            ),
        }
    return frames, {
        "target_color_component_profile": color_profile,
        "invalid_reasons_by_view": {
            view: dict(sorted(errors[view].items())) for view in AXIAL_VIEW_IDS
        },
        "target_frames_by_view": {
            view: target_frame_counts[view] for view in AXIAL_VIEW_IDS
        },
        "valid_samples_by_view": {
            view: valid_counts[view] for view in AXIAL_VIEW_IDS
        },
    }


def _target_color_components(
    image_bgr,
    palette: Mapping[str, Any],
    target: str,
    *,
    sample_index: int,
) -> list[dict[str, Any]]:
    """Profile broad target-hue components; never promote them to identity.

    The palette is cyclic, so a +/-6 hue band overlaps neighbouring teeth.
    These components diagnose whether the rear-view cluster gate discarded a
    large target-coloured region, but they are deliberately not returned to
    the centroid analyzer and cannot improve formal identity coverage.
    """

    if image_bgr is None or image_bgr.ndim != 3:
        raise EvidenceError("axial RGB frame cannot be decoded")
    height, width = image_bgr.shape[:2]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    index = int(target.rsplit("_", 1)[1])
    target_hue = analysis._palette_hues(palette)[index]  # noqa: SLF001
    distance = np.abs(hue.astype(float) - target_hue)
    distance = np.minimum(distance, 180.0 - distance)
    mask = (
        (saturation >= 45) & (value >= 40) & (distance <= 6.0)
    ).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, _, stats, centers = cv2.connectedComponentsWithStats(mask)
    components = []
    for component in range(1, count):
        x, y, component_width, component_height, area = (
            int(item) for item in stats[component]
        )
        center_x, center_y = (float(item) for item in centers[component])
        # This is the fixed axial nut ROI.  Its deliberately broad bounds are
        # diagnostic, while the area/aspect gates reject small highlights.
        if (
            area >= 200
            and area <= int(0.025 * width * height)
            and component_width <= int(0.28 * min(width, height))
            and component_height <= int(0.28 * min(width, height))
            and component_height >= 0.80 * component_width
            and 240.0 <= center_x <= 720.0
            and 280.0 <= center_y <= 650.0
        ):
            components.append(
                {
                    "area_px": area,
                    "bbox_xywh": [
                        x,
                        y,
                        component_width,
                        component_height,
                    ],
                    "center_xy_px": [center_x, center_y],
                    "sample_index": int(sample_index),
                }
            )
    return components


def _validate_run_log(path: Path) -> dict[str, Any]:
    """Bind the process success marker absent from the physics sidecar."""

    content = path.read_text(encoding="utf-8", errors="strict")
    passed_marker = "ISAAC D38999 Q7 TWIST PROBE V1 PASSED"
    if content.count(passed_marker) != 1:
        raise EvidenceError("run log must contain exactly one PASS marker")
    if "Traceback (most recent call last)" in content or " FAILED" in content:
        raise EvidenceError("run log contains a failure marker")
    main_reports = []
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
            main_reports.append(value)
    if len(main_reports) != 1:
        raise EvidenceError(
            "run log must contain exactly one main JSON report"
        )
    main = main_reports[0]
    if (
        main.get("passed") is not True
        or main.get("twist_probe", {}).get("passed") is not True
        or main.get("nut_tooth_jitter_probe", {}).get("anomaly_steps") != 0
        or main.get("zero_forbidden_contacts") is not True
        or main.get("object_pose_writes_after_start") != 0
    ):
        raise EvidenceError("run log main JSON PASS gates differ")
    return {
        "exactly_one_main_json": True,
        "exactly_one_pass_marker": True,
        "main_passed": True,
        "no_failed_or_traceback_marker": True,
        "twist_probe_passed": True,
    }


def _frame_key(frame: Mapping[str, Any]) -> tuple[int, str, int]:
    return (
        int(frame["global_step"]),
        str(frame["phase"]),
        int(frame["phase_step"]),
    )


def merge_same_key_frames(
    base_frames: list[dict[str, Any]], axial_frames: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Merge view maps only when every ordered physics sample key matches."""

    if [_frame_key(frame) for frame in base_frames] != [
        _frame_key(frame) for frame in axial_frames
    ]:
        raise EvidenceError("analyzed base/axial frame keys differ")
    merged = []
    for base, axial in zip(base_frames, axial_frames):
        overlap = set(base["views"]) & set(axial["views"])
        if overlap:
            raise EvidenceError(
                f"base/axial view IDs overlap: {sorted(overlap)}"
            )
        merged.append(
            {
                **base,
                "views": {**base["views"], **axial["views"]},
                "axial_view_errors": axial["view_errors"],
            }
        )
    return merged


def collect_dynamic_view_coverage(
    frames: list[dict[str, Any]], view_ids: tuple[str, ...]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Measure all views and retain empty axial-only transitions explicitly."""

    rows = []
    transitions = []
    previous = None
    for current in frames:
        if previous is None or previous["phase"] != current["phase"]:
            previous = current
            continue
        observed = set()
        usable = []
        for view_id in view_ids:
            common = analysis._common_transition_segments(  # noqa: SLF001
                previous, current, view_id
            )
            if len(common) < analysis.MINIMUM_VISIBLE_TEETH:
                continue
            usable.append(view_id)
            observed.update(common)
            rows.extend(
                analysis._measure_transition(  # noqa: SLF001
                    previous, current, view_id, common
                )
            )
        transitions.append(
            {
                "global_step": current["global_step"],
                "phase": current["phase"],
                "phase_step": current["phase_step"],
                "segments": sorted(observed),
                "usable_views": usable,
            }
        )
        previous = current
    if not transitions:
        raise EvidenceError("six-view evidence has no phase-local transitions")
    return rows, sync_evidence._coverage_summary(transitions)  # noqa: SLF001


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise EvidenceError(f"cannot write empty evidence CSV: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_axial_evidence(
    *,
    repository: str | Path,
    run_root: str | Path,
    output_directory: str | Path,
    base_capture_helper: str | Path,
    axial_capture_helper: str | Path,
    wrapper_source: str | Path,
    runner_source: str | Path,
) -> dict[str, Any]:
    """Build one immutable six-view ghost/axial evidence artifact."""

    repository = Path(repository).expanduser().resolve()
    run_root = _repo_directory(run_root, repository, "six-view run root")
    output = Path(output_directory).expanduser().resolve()
    if output.exists():
        raise EvidenceError(f"output already exists: {output}")
    if repository != output and repository not in output.parents:
        raise EvidenceError("output escapes repository")
    base_helper = _repo_file(base_capture_helper, repository, "base helper")
    axial_helper = _repo_file(axial_capture_helper, repository, "axial helper")
    wrapper = _repo_file(wrapper_source, repository, "axial wrapper")
    runner = _repo_file(runner_source, repository, "prepared runner")

    base_bundle = analysis.validate_capture_bundle(
        run_root / "capture", base_helper
    )
    base_result = analysis.analyze_validated_capture(base_bundle)
    axial_bundle = _validate_axial_capture(
        repository=repository,
        run_root=run_root,
        base_bundle=base_bundle,
        axial_helper=axial_helper,
        wrapper=wrapper,
        runner=runner,
    )
    posthoc = _validate_posthoc_bundle(
        repository=repository,
        run_root=run_root,
        axial_manifest_path=axial_bundle["manifest_path"],
    )
    runtime_bundle, runtime = occlusion.validate_runtime_bundle(
        repository=repository,
        ghost_root=run_root,
        capture_manifest_path=run_root / "capture/video_capture_manifest.json",
        physics_report_path=Path(base_bundle["physics_report_path"]),
        physics_summary_path=Path(base_bundle["physics_summary_path"]),
    )
    physics = sync_evidence.validate_physics_report(
        base_bundle["physics_report"]
    )
    run_log_path = _repo_file(run_root / "run.log", repository, "run log")
    process_result = _validate_run_log(run_log_path)
    axial_frames, extraction = _extract_axial_frames(
        axial_bundle["samples"], base_bundle["palette"]
    )
    merged_frames = merge_same_key_frames(base_result["frames"], axial_frames)
    axial_rows, axial_coverage = collect_dynamic_view_coverage(
        axial_frames, AXIAL_VIEW_IDS
    )
    six_rows, six_coverage = collect_dynamic_view_coverage(
        merged_frames, ALL_VIEW_IDS
    )
    target_transition_counts = {
        target: axial_coverage["per_segment_transition_counts"][target]
        for target in TARGET_BY_VIEW.values()
    }
    strict_passed = six_coverage["every_transition_all_24"]
    report = {
        "schema_version": SCHEMA_VERSION,
        "classification": (
            "VALID_STRICT_SIX_VIEW_24_TRANSITION_COVERAGE"
            if strict_passed
            else "VALID_SIX_VIEW_PARTIAL_COVERAGE_JITTER_UNRESOLVED"
        ),
        "evidence_valid": True,
        "physics": physics,
        "process_result": process_result,
        "capture": {
            "all_1590_png_hashes_revalidated": True,
            "axial_rgb_frames": len(axial_bundle["rows"]),
            "base_rgb_frames": len(base_bundle["rows"]),
            "same_265_sample_keys_revalidated": True,
            "views": list(ALL_VIEW_IDS),
        },
        "ghost_runtime": {
            **runtime,
            "cleanup": runtime_bundle["sidecar"]["cleanup"],
            "mutation_audit": runtime_bundle["sidecar"]["mutation_audit"],
            "passed": runtime_bundle["sidecar"]["passed"],
        },
        "posthoc_binding": {
            "bundle_origin": "posthoc_existing_pure_function",
            "passed": posthoc["bundle"]["passed"],
            "wrapper_created_bundle_claim_authorized": False,
            "wrapper_design_gap": (
                "prepared runner fast-shutdown exits before wrapper "
                "post-return finalizer can execute"
            ),
        },
        "visual": {
            "axial_extraction": extraction,
            "axial_only_coverage": axial_coverage,
            "six_view_coverage": six_coverage,
            "target_transition_counts": target_transition_counts,
            "strict_every_transition_all_24": strict_passed,
            "render_jitter_absence_claim_authorized": False,
        },
        "limitations": [
            "identity_union_is_not_every_transition_coverage",
            "segment13_or_segment23_detection_is_not_a_no_jitter_result",
            "no_visual_residual_acceptance_threshold_was_preregistered",
            "30_hz_sampling_cannot_exclude_between_sample_render_artifacts",
            "axial_ghost_bundle_was_created_posthoc_not_by_the_wrapper",
            "prepared_twist_probe_is_not_full_end_to_end_assembly",
        ],
    }

    output.mkdir(parents=True, exist_ok=False)
    axial_csv = output / "axial_all_view_residuals.csv"
    six_csv = output / "six_view_residuals.csv"
    report_path = output / "report.json"
    _write_csv(axial_csv, axial_rows)
    _write_csv(six_csv, six_rows)
    report_path.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "HASH_SIZE_SCHEMA_BOUND",
        "inputs": {
            "axial_capture_manifest": _binding(
                axial_bundle["manifest_path"], repository
            ),
            "axial_ghost_bundle": _binding(posthoc["path"], repository),
            "base_capture_manifest": _binding(
                run_root / "capture/video_capture_manifest.json", repository
            ),
            "ghost_manifest": _binding(
                run_root / "ghost/manifest.json", repository
            ),
            "ghost_visibility_sidecar": _binding(
                run_root / "ghost/visibility_sidecar.json", repository
            ),
            "physics_report": _binding(
                base_bundle["physics_report_path"], repository
            ),
            "physics_summary": _binding(
                base_bundle["physics_summary_path"], repository
            ),
            "run_log": _binding(run_log_path, repository),
        },
        "indirect_frame_binding": {
            "all_png_hashes_revalidated": True,
            "mechanism": "base_and_axial_manifest_per_png_sha256_maps",
            "rgb_frames_revalidated": (
                len(base_bundle["rows"]) + len(axial_bundle["rows"])
            ),
        },
        "outputs": {
            "axial_all_view_residuals": _binding(axial_csv, repository),
            "report": _binding(report_path, repository),
            "six_view_residuals": _binding(six_csv, repository),
        },
        "sources": {
            "axial_capture": _binding(axial_helper, repository),
            "axial_evidence": _binding(Path(__file__), repository),
            "axial_wrapper": _binding(wrapper, repository),
            "base_analysis": _binding(Path(analysis.__file__), repository),
            "base_capture": _binding(base_helper, repository),
            "occlusion_evidence": _binding(
                Path(occlusion.__file__), repository
            ),
            "prepared_runner": _binding(runner, repository),
            "sync_evidence": _binding(
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
    source = repository / "src/kcg_connector"
    parser = argparse.ArgumentParser(
        description="Aggregate hash-bound D38999 six-view tooth evidence"
    )
    parser.add_argument("--repository", type=Path, default=repository)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--base-capture-helper",
        type=Path,
        default=source / "isaac/d38999_tooth_sync_capture.py",
    )
    parser.add_argument(
        "--axial-capture-helper",
        type=Path,
        default=source / "isaac/d38999_tooth_axial_capture.py",
    )
    parser.add_argument(
        "--wrapper-source",
        type=Path,
        default=source / "isaac/d38999_nut_regrasp_axial_smoke.py",
    )
    parser.add_argument(
        "--runner-source",
        type=Path,
        default=source / "isaac/d38999_nut_regrasp_smoke.py",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    arguments = _arguments(argv)
    try:
        result = aggregate_axial_evidence(
            repository=arguments.repository,
            run_root=arguments.run_root,
            output_directory=arguments.output,
            base_capture_helper=arguments.base_capture_helper,
            axial_capture_helper=arguments.axial_capture_helper,
            wrapper_source=arguments.wrapper_source,
            runner_source=arguments.runner_source,
        )
    except (
        EvidenceError,
        analysis.EvidenceError,
        occlusion.EvidenceError,
        sync_evidence.EvidenceError,
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
    "ALL_VIEW_IDS",
    "AXIAL_VIEW_IDS",
    "EvidenceError",
    "MANIFEST_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "aggregate_axial_evidence",
    "collect_dynamic_view_coverage",
    "main",
    "merge_same_key_frames",
]
