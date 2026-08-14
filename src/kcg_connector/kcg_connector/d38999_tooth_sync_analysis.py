#!/usr/bin/env python3

"""Fail-closed CPU analysis for synchronized D38999 tooth RGB captures.

Unlike desktop-video exploration, this analyzer accepts only frame sets whose
manifest binds every PNG to a physics report, physics summary, capture helper,
view ID and explicit ``global_step/phase/phase_step`` row.  Fixed rear and
front camera pairs cover both sides of the nut as well as left/right gripper
occlusion.  View choice follows manifest priority, never observed residual
size.  A two-run A/B requires one common valid view at each aligned transition.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import colorsys
import csv
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

from kcg_connector.d38999_tooth_video_analysis import fit_similarity_2d


SCHEMA_VERSION = "kcg_d38999_tooth_sync_analysis_v3"
CAPTURE_SCHEMA_VERSION = "kcg_d38999_tooth_sync_capture_v3"
PHYSICS_SCHEMA_VERSION = "kcg_d38999_nut_tooth_jitter_probe_v1"
CAPTURE_PHASES = (
    "nut_only_final_hold",
    "q7_twist_probe_motion",
    "q7_twist_probe_hold",
)
SYNC_FIELDS = (
    "frame_index",
    "sample_index",
    "view_id",
    "global_step",
    "phase",
    "phase_step",
    "simulation_time_s",
    "timestamp_s",
    "rgb_filename",
)
VIEW_IDS = (
    "rear_left",
    "rear_right",
    "front_left",
    "front_right",
)
EXPECTED_CAMERA_RIG = {
    "same_completed_physics_step": True,
    "selection_contract": {
        "ab_requires_same_view_per_transition": True,
        "minimum_adjacent_pitch_px": 12.0,
        "minimum_adjacent_pairs": 3,
        "minimum_visible_teeth": 4,
        "policy": "first_valid_view_in_manifest_priority",
    },
    "view_priority": list(VIEW_IDS),
    "views": [
        {
            "eye_m": [0.30, -0.46, 0.39],
            "focal_length_mm": 50.0,
            "prim_path": (
                "/World/D38999NutRegrasp/ToothSyncCameraRearLeft"
            ),
            "fixed_oblique_before_play": True,
            "render_product_name": (
                "D38999ToothSyncRenderProductRearLeft"
            ),
            "resolution": [960, 720],
            "target_m": [0.55, 0.185, 0.265],
            "view_id": "rear_left",
        },
        {
            "eye_m": [0.80, -0.46, 0.39],
            "focal_length_mm": 50.0,
            "prim_path": (
                "/World/D38999NutRegrasp/ToothSyncCameraRearRight"
            ),
            "fixed_oblique_before_play": True,
            "render_product_name": (
                "D38999ToothSyncRenderProductRearRight"
            ),
            "resolution": [960, 720],
            "target_m": [0.55, 0.185, 0.265],
            "view_id": "rear_right",
        },
        {
            "eye_m": [0.30, 0.8300000000000001, 0.39],
            "focal_length_mm": 50.0,
            "prim_path": (
                "/World/D38999NutRegrasp/ToothSyncCameraFrontLeft"
            ),
            "fixed_oblique_before_play": True,
            "render_product_name": (
                "D38999ToothSyncRenderProductFrontLeft"
            ),
            "resolution": [960, 720],
            "target_m": [0.55, 0.185, 0.265],
            "view_id": "front_left",
        },
        {
            "eye_m": [0.80, 0.8300000000000001, 0.39],
            "focal_length_mm": 50.0,
            "prim_path": (
                "/World/D38999NutRegrasp/ToothSyncCameraFrontRight"
            ),
            "fixed_oblique_before_play": True,
            "render_product_name": (
                "D38999ToothSyncRenderProductFrontRight"
            ),
            "resolution": [960, 720],
            "target_m": [0.55, 0.185, 0.265],
            "view_id": "front_right",
        },
    ],
}
MINIMUM_TOOTH_PITCH_PX = 12.0
MINIMUM_VISIBLE_TEETH = 4
MINIMUM_ADJACENT_PAIRS = 3
MINIMUM_RELATIVE_COMPONENT_AREA = 0.075
_NUMERIC_TOLERANCE = 1.0e-9


class EvidenceError(RuntimeError):
    """Raised when synchronized evidence cannot authorize an analysis."""


def sha256_file(path):
    """Hash a potentially large evidence file without loading it at once."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_segment_colors(count=24):
    """Reconstruct the exact session-layer tooth color contract."""

    return {
        f"Segment_{index:02d}": [
            round(value, 6)
            for value in colorsys.hsv_to_rgb(
                float(index) / float(count), 0.72, 0.95
            )
        ]
        for index in range(count)
    }


def physics_trace_sha256(summary_path):
    """Hash all physics/state fields except verbose contact-record text."""

    digest = hashlib.sha256()
    with Path(summary_path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = [
            field
            for field in (reader.fieldnames or ())
            if field != "segment_contact_records"
        ]
        digest.update(("\x1f".join(fields) + "\n").encode("utf-8"))
        for row in reader:
            digest.update(
                ("\x1f".join(row[field] for field in fields) + "\n").encode(
                    "utf-8"
                )
            )
    return digest.hexdigest()


def _require_exact_mapping(actual, expected, label):
    if set(actual) != set(expected):
        raise EvidenceError(f"{label} keys differ from contract")
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if isinstance(expected_value, list):
            if not np.allclose(
                np.asarray(actual_value, dtype=float),
                np.asarray(expected_value, dtype=float),
                rtol=0.0,
                atol=_NUMERIC_TOLERANCE,
            ):
                raise EvidenceError(f"{label}.{key} differs from contract")
        elif actual_value != expected_value:
            raise EvidenceError(f"{label}.{key} differs from contract")


def _safe_bound_file(path, expected_sha256, label):
    target = Path(path).expanduser().resolve()
    if not target.is_file():
        raise EvidenceError(f"{label} is missing: {target}")
    actual = sha256_file(target)
    if actual != expected_sha256:
        raise EvidenceError(f"{label} SHA-256 mismatch")
    return target


def _read_sync_rows(path):
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != SYNC_FIELDS:
            raise EvidenceError("sync CSV columns differ from contract")
        rows = list(reader)
    if not rows:
        raise EvidenceError("sync CSV is empty")
    parsed = []
    for row in rows:
        try:
            parsed.append(
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
        except (TypeError, ValueError) as exception:
            raise EvidenceError(
                "sync CSV contains invalid numeric data"
            ) from exception
    return parsed


def _expected_phase_schedule(total_steps, interval):
    multiples = list(range(interval, total_steps + 1, interval))
    return multiples if multiples and multiples[0] == 1 else [1, *multiples]


def _physics_step_lookup(summary_path):
    lookup = {}
    with Path(summary_path).open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            key = int(row["global_step"])
            if key in lookup:
                raise EvidenceError("physics summary repeats a global step")
            lookup[key] = (row["phase"], int(row["phase_step"]))
    if not lookup:
        raise EvidenceError("physics summary is empty")
    return lookup


def validate_capture_bundle(capture_directory, capture_helper_path):
    """Validate every hash, schedule and fixed-camera input before pixels."""

    capture = Path(capture_directory).expanduser().resolve()
    helper = Path(capture_helper_path).expanduser().resolve()
    manifest_path = capture / "video_capture_manifest.json"
    if not manifest_path.is_file():
        raise EvidenceError("video_capture_manifest.json is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != CAPTURE_SCHEMA_VERSION:
        raise EvidenceError("capture manifest schema mismatch")
    if manifest.get("passed") is not True:
        raise EvidenceError("capture manifest is not passed")
    if manifest.get("camera_rig") != EXPECTED_CAMERA_RIG:
        raise EvidenceError("fixed four-camera rig differs from contract")

    source = manifest.get("capture_source", {})
    if (
        source.get("sha256_at_import") != source.get("sha256_at_start")
        or source.get("sha256_at_start") != sha256_file(helper)
    ):
        raise EvidenceError(
            "capture helper SHA-256 differs from current helper"
        )
    if (
        source.get("sha256_at_finalize") != source.get("sha256_at_start")
        or source.get("unchanged_during_capture") is not True
    ):
        raise EvidenceError("capture helper changed during capture")
    if Path(source.get("path", "")).name != helper.name:
        raise EvidenceError("capture helper filename differs from contract")

    cleanup = manifest.get("cleanup", {})
    required_cleanup = {
        "annotator_detached": True,
        "annotators_detached_count": len(VIEW_IDS),
        "camera_prim_removed": True,
        "camera_prims_removed_count": len(VIEW_IDS),
        "object_pose_writes": 0,
        "render_product_destroyed": True,
        "render_products_destroyed_count": len(VIEW_IDS),
        "resources_released": True,
        "stage_cleared": False,
        "view_count": len(VIEW_IDS),
        "world_reset": False,
    }
    for key, expected in required_cleanup.items():
        if cleanup.get(key) != expected:
            raise EvidenceError(f"capture cleanup gate failed: {key}")
    if cleanup.get("errors") != []:
        raise EvidenceError("capture cleanup contains errors")

    physics = manifest.get("physics_evidence", {})
    if physics.get("tooth_color_ids_authored") is not True:
        raise EvidenceError("24-tooth color IDs were not proven authored")
    if physics.get("tooth_color_id_count") != 24:
        raise EvidenceError("tooth color ID count is not 24")
    report_path = _safe_bound_file(
        physics.get("report_path", ""),
        physics.get("report_sha256"),
        "physics report",
    )
    summary_path = _safe_bound_file(
        physics.get("summary_path", ""),
        physics.get("summary_sha256"),
        "physics summary",
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != PHYSICS_SCHEMA_VERSION:
        raise EvidenceError("physics report schema mismatch")
    color_report = report.get("color_identification", {})
    if color_report.get("authored_in_session_layer") is not True:
        raise EvidenceError("physics report did not author color IDs")
    palette = color_report.get("colors_rgb")
    if palette != deterministic_segment_colors():
        raise EvidenceError(
            "physics report tooth palette differs from contract"
        )
    if report.get("render_ab_launch") != manifest.get("render_settings"):
        raise EvidenceError("physics and capture render settings differ")

    sampling = manifest.get("sampling", {})
    physics_rate = sampling.get("physics_rate_hz")
    capture_rate = sampling.get("capture_rate_hz")
    interval = sampling.get("physics_steps_per_frame")
    if (
        not all(
            type(value) is int
            for value in (physics_rate, capture_rate, interval)
        )
        or physics_rate <= 0
        or capture_rate <= 0
        or interval <= 0
        or physics_rate != capture_rate * interval
        or sampling.get("sampling_kind")
        != "fixed_integer_physics_step_decimation"
    ):
        raise EvidenceError("capture sampling contract is invalid")
    if tuple(manifest.get("sync_columns", ())) != SYNC_FIELDS:
        raise EvidenceError("manifest sync columns differ from contract")
    if manifest.get("sync_csv") != "video_frame_sync.csv":
        raise EvidenceError("manifest sync CSV name differs from contract")
    sync_path = capture / "video_frame_sync.csv"
    if sha256_file(sync_path) != manifest.get("sync_csv_sha256"):
        raise EvidenceError("sync CSV SHA-256 mismatch")
    rows = _read_sync_rows(sync_path)

    phase_totals = report.get("phase_steps", {})
    lookup = _physics_step_lookup(summary_path)
    observed_by_phase = defaultdict(list)
    previous_sample_global_step = -1
    previous_timestamp = -math.inf
    samples = []
    frame_hashes = manifest.get("frame_files_sha256", {})
    if set(frame_hashes) != {row["rgb_filename"] for row in rows}:
        raise EvidenceError("manifest frame set differs from sync CSV")
    for expected_index, row in enumerate(rows):
        if row["frame_index"] != expected_index:
            raise EvidenceError("frame indices are not contiguous from zero")
        expected_sample = expected_index // len(VIEW_IDS)
        expected_view = VIEW_IDS[expected_index % len(VIEW_IDS)]
        if (
            row["sample_index"] != expected_sample
            or row["view_id"] != expected_view
        ):
            raise EvidenceError(
                "sync rows do not contain the complete fixed view order"
            )
        if (
            not math.isfinite(row["timestamp_s"])
            or row["timestamp_s"] <= previous_timestamp
        ):
            raise EvidenceError(
                "sync timestamps are not finite and increasing"
            )
        expected_time = row["global_step"] / float(physics_rate)
        if not math.isclose(
            row["simulation_time_s"],
            expected_time,
            rel_tol=0.0,
            abs_tol=_NUMERIC_TOLERANCE,
        ):
            raise EvidenceError("sync simulation timestamp differs from step")
        if lookup.get(row["global_step"]) != (
            row["phase"],
            row["phase_step"],
        ):
            raise EvidenceError("sync row differs from physics summary")
        if row["phase"] not in CAPTURE_PHASES:
            raise EvidenceError("sync row contains an unexpected phase")
        relative = Path(row["rgb_filename"])
        if relative.is_absolute() or ".." in relative.parts:
            raise EvidenceError("frame path is not capture-relative")
        frame_path = (capture / relative).resolve()
        if capture not in frame_path.parents or not frame_path.is_file():
            raise EvidenceError(
                "synchronized frame is missing or escapes capture"
            )
        if sha256_file(frame_path) != frame_hashes[row["rgb_filename"]]:
            raise EvidenceError("synchronized frame SHA-256 mismatch")
        row["frame_path"] = frame_path
        if row["view_id"] == VIEW_IDS[0]:
            if row["global_step"] <= previous_sample_global_step:
                raise EvidenceError(
                    "sync sample global steps are not strictly increasing"
                )
            samples.append(
                {
                    "global_step": row["global_step"],
                    "phase": row["phase"],
                    "phase_step": row["phase_step"],
                    "sample_index": row["sample_index"],
                    "simulation_time_s": row["simulation_time_s"],
                    "views": {row["view_id"]: row},
                }
            )
            observed_by_phase[row["phase"]].append(row["phase_step"])
            previous_sample_global_step = row["global_step"]
        else:
            sample = samples[-1]
            for field in (
                "global_step",
                "phase",
                "phase_step",
                "sample_index",
                "simulation_time_s",
            ):
                if row[field] != sample[field]:
                    raise EvidenceError(
                        "views in one sample differ in physics mapping"
                    )
            sample["views"][row["view_id"]] = row
        previous_timestamp = row["timestamp_s"]

    if len(rows) % len(VIEW_IDS) != 0:
        raise EvidenceError("final synchronized sample is missing a view")
    if any(tuple(sample["views"]) != VIEW_IDS for sample in samples):
        raise EvidenceError("a synchronized sample is missing a fixed view")

    for phase in CAPTURE_PHASES:
        if phase not in phase_totals:
            raise EvidenceError(f"physics report is missing phase {phase}")
        expected = _expected_phase_schedule(int(phase_totals[phase]), interval)
        if observed_by_phase[phase] != expected:
            raise EvidenceError(
                f"synchronized frame plan is incomplete for {phase}"
            )
    frame_capture = manifest.get("frame_capture", {})
    expected_frame_capture = {
        "first_global_step": samples[0]["global_step"],
        "frame_count": len(rows),
        "frames_per_view": {
            view_id: len(samples) for view_id in VIEW_IDS
        },
        "last_global_step": samples[-1]["global_step"],
        "phases": list(CAPTURE_PHASES),
        "sample_count": len(samples),
        "view_order": list(VIEW_IDS),
    }
    if frame_capture != expected_frame_capture:
        raise EvidenceError("manifest frame_capture summary differs from CSV")
    actual_frames = {
        str(path.relative_to(capture))
        for path in (capture / "frames").rglob("*.png")
    }
    if actual_frames != set(frame_hashes):
        raise EvidenceError(
            "capture frames directory contains missing or stale PNGs"
        )
    return {
        "capture_directory": capture,
        "manifest": manifest,
        "manifest_sha256": sha256_file(manifest_path),
        "palette": palette,
        "physics_report": report,
        "physics_report_path": report_path,
        "physics_summary_path": summary_path,
        "physics_trace_sha256": physics_trace_sha256(summary_path),
        "rows": rows,
        "samples": samples,
    }


def _palette_hues(palette):
    rgb = np.uint8(
        [
            [
                np.asarray(palette[f"Segment_{index:02d}"]) * 255
                for index in range(24)
            ]
        ]
    )
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[0, :, 0].astype(float)


def extract_tooth_centroids(image_bgr, palette):
    """Extract one spatially clustered component per visible tooth color ID."""

    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise EvidenceError("synchronized RGB frame cannot be decoded")
    height, width = image_bgr.shape[:2]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    hues = _palette_hues(palette)
    distances = np.abs(hue[:, :, None].astype(float) - hues[None, None, :])
    distances = np.minimum(distances, 180.0 - distances)
    labels = np.argmin(distances, axis=2)
    nearest = np.min(distances, axis=2)
    chromatic = (saturation >= 45) & (value >= 40) & (nearest <= 6.0)
    candidates = []
    maximum_area = int(0.025 * width * height)
    maximum_dimension = int(0.28 * min(width, height))
    kernel = np.ones((3, 3), dtype=np.uint8)
    for index in range(24):
        mask = ((labels == index) & chromatic).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        count, _, stats, centers = cv2.connectedComponentsWithStats(mask)
        for component in range(1, count):
            x, y, component_width, component_height, area = (
                int(item) for item in stats[component]
            )
            if (
                12 <= area <= maximum_area
                and component_width <= maximum_dimension
                and component_height <= maximum_dimension
                # The authored tooth faces are tall radial strips in this
                # fixed rear view.  Broad blobs are gripper/table reflections
                # that happen to share a palette hue, not tooth identities.
                and component_height >= 1.35 * component_width
            ):
                candidates.append(
                    {
                        "area": area,
                        "center": np.asarray(centers[component], dtype=float),
                        "segment": f"Segment_{index:02d}",
                    }
                )
    if not candidates:
        raise EvidenceError("no tooth-color components were found")
    cluster_radius = 0.25 * min(width, height)
    best_cluster = None
    best_score = None
    for anchor in candidates:
        nearby = [
            item
            for item in candidates
            if np.linalg.norm(item["center"] - anchor["center"])
            <= cluster_radius
        ]
        score = (
            len({item["segment"] for item in nearby}),
            sum(item["area"] for item in nearby),
        )
        if best_score is None or score > best_score:
            best_score = score
            best_cluster = nearby
    if best_score[0] < MINIMUM_VISIBLE_TEETH:
        raise EvidenceError(
            "fewer than four clustered tooth color IDs are visible"
        )
    # Tiny chromatic highlights elsewhere in the scene can share a palette
    # hue.  A usable tooth ID must therefore also have a non-trivial area
    # relative to the largest tooth surface in the selected spatial cluster.
    # This rejects isolated specks without weakening the independent 12 px
    # adjacent-centroid pitch gate.
    largest_area = max(item["area"] for item in best_cluster)
    reliable_area = MINIMUM_RELATIVE_COMPONENT_AREA * largest_area
    best_cluster = [
        item for item in best_cluster if item["area"] >= reliable_area
    ]
    if len({item["segment"] for item in best_cluster}) < MINIMUM_VISIBLE_TEETH:
        raise EvidenceError(
            "fewer than four reliable tooth color IDs are visible"
        )
    cluster_center = np.median(
        np.stack([item["center"] for item in best_cluster]), axis=0
    )
    points = {}
    areas = {}
    for segment in {item["segment"] for item in best_cluster}:
        options = [
            item for item in best_cluster if item["segment"] == segment
        ]
        chosen = min(
            options,
            key=lambda item: (
                np.linalg.norm(item["center"] - cluster_center),
                -item["area"],
            ),
        )
        points[segment] = chosen["center"]
        areas[segment] = chosen["area"]
    return points, areas


def adjacent_tooth_pitches(points):
    """Return center distances for every visible cyclic adjacent tooth pair."""

    pitches = {}
    for index in range(24):
        first = f"Segment_{index:02d}"
        second = f"Segment_{(index + 1) % 24:02d}"
        if first in points and second in points:
            pitches[f"{first}->{second}"] = float(
                np.linalg.norm(points[second] - points[first])
            )
    return pitches


def select_pitch_qualified_cluster(points):
    """Select a contiguous tooth-ID cluster whose every edge is >= 12 px."""

    adjacency = defaultdict(set)
    pitches = adjacent_tooth_pitches(points)
    for edge, pitch in pitches.items():
        if pitch < MINIMUM_TOOTH_PITCH_PX:
            continue
        first, second = edge.split("->")
        adjacency[first].add(second)
        adjacency[second].add(first)
    components = []
    remaining = set(adjacency)
    while remaining:
        pending = [remaining.pop()]
        component = set(pending)
        while pending:
            current = pending.pop()
            for neighbor in adjacency[current]:
                if neighbor not in component:
                    component.add(neighbor)
                    remaining.discard(neighbor)
                    pending.append(neighbor)
        components.append(component)
    if not components:
        raise EvidenceError("no adjacent tooth pair has at least 12px pitch")
    chosen = max(components, key=lambda item: (len(item), sorted(item)))
    selected = {name: points[name] for name in chosen}
    selected_pitches = adjacent_tooth_pitches(selected)
    if (
        len(selected) < MINIMUM_VISIBLE_TEETH
        or len(selected_pitches) < MINIMUM_ADJACENT_PAIRS
    ):
        raise EvidenceError(
            "fewer than four contiguous teeth have at least 12px pitch"
        )
    return selected, selected_pitches


def _extract_sample_views(sample, palette, view_priority):
    """Extract every independently valid view without residual-based choice."""

    valid = {}
    errors = {}
    for view_id in view_priority:
        row = sample["views"][view_id]
        image = cv2.imread(str(row["frame_path"]), cv2.IMREAD_COLOR)
        try:
            points, areas = extract_tooth_centroids(image, palette)
            points, pitches = select_pitch_qualified_cluster(points)
        except EvidenceError as error:
            errors[view_id] = str(error)
            continue
        valid[view_id] = {
            "areas": {name: areas[name] for name in points},
            "frame_path": row["frame_path"],
            "points": points,
            "pitches": pitches,
            "rgb_filename": row["rgb_filename"],
        }
    if not valid:
        raise EvidenceError(
            "neither fixed view passes the tooth gate at sample_index="
            f"{sample['sample_index']} global_step={sample['global_step']} "
            f"phase={sample['phase']} phase_step={sample['phase_step']} "
            f"view_errors={json.dumps(errors, sort_keys=True)}"
        )
    return {**sample, "view_errors": errors, "views": valid}


def _common_transition_segments(previous, current, view_id):
    if view_id not in previous["views"] or view_id not in current["views"]:
        return []
    return sorted(
        set(previous["views"][view_id]["points"])
        & set(current["views"][view_id]["points"])
    )


def _select_transition_view(previous, current, view_priority):
    """Choose the first prior-ranked view valid across one transition."""

    common_by_view = {}
    for view_id in view_priority:
        common = _common_transition_segments(previous, current, view_id)
        common_by_view[view_id] = common
        if len(common) >= MINIMUM_VISIBLE_TEETH:
            return view_id, common
    raise EvidenceError(
        "no prioritized view shares four teeth across adjacent samples at "
        f"global_step={current['global_step']} phase={current['phase']} "
        f"phase_step={current['phase_step']} previous_global_step="
        f"{previous['global_step']} common_by_view="
        f"{json.dumps(common_by_view, sort_keys=True)}"
    )


def _measure_transition(previous, current, view_id, segments):
    """Fit one view/segment set and return one residual row per tooth."""

    previous_view = previous["views"][view_id]
    current_view = current["views"][view_id]
    reference = np.stack(
        [previous_view["points"][name] for name in segments]
    )
    observed = np.stack(
        [current_view["points"][name] for name in segments]
    )
    fit = fit_similarity_2d(reference, observed)
    pitch = min(
        min(previous_view["pitches"].values()),
        min(current_view["pitches"].values()),
    )
    rows = []
    for index, segment in enumerate(segments):
        vector = fit["residual_vectors"][index]
        residual = float(np.linalg.norm(vector))
        rows.append(
            {
                "global_step": current["global_step"],
                "phase": current["phase"],
                "phase_step": current["phase_step"],
                "previous_global_step": previous["global_step"],
                "segment": segment,
                "view_id": view_id,
                "residual_x_px": float(vector[0]),
                "residual_y_px": float(vector[1]),
                "residual_px": residual,
                "tooth_pitch_px": pitch,
                "residual_pitch_fraction": residual / pitch,
                "registered_scale": float(fit["scale"]),
            }
        )
    return rows


def analyze_validated_capture(bundle):
    """Measure transitions using fixed-priority complementary camera views."""

    priority = tuple(bundle["manifest"]["camera_rig"]["view_priority"])
    frames = [
        _extract_sample_views(sample, bundle["palette"], priority)
        for sample in bundle["samples"]
    ]
    residual_rows = []
    view_usage = {view_id: 0 for view_id in priority}
    previous = None
    for current in frames:
        if previous is None or previous["phase"] != current["phase"]:
            previous = current
            continue
        view_id, common = _select_transition_view(
            previous, current, priority
        )
        residual_rows.extend(
            _measure_transition(previous, current, view_id, common)
        )
        view_usage[view_id] += 1
        previous = current
    if not residual_rows:
        raise EvidenceError("capture has no phase-local frame transitions")
    valid_counts = {
        view_id: sum(view_id in frame["views"] for frame in frames)
        for view_id in priority
    }
    used_pitches = [row["tooth_pitch_px"] for row in residual_rows]
    return {
        "frames": frames,
        "frames_analyzed": len(frames),
        "minimum_tooth_pitch_px": min(used_pitches),
        "residual_rows": residual_rows,
        "rgb_frames_validated": len(bundle["rows"]),
        "segments_observed": sorted(
            {row["segment"] for row in residual_rows}
        ),
        "valid_sample_counts_by_view": valid_counts,
        "view_usage_by_transition": view_usage,
    }


def summarize_residuals(rows):
    """Aggregate per-phase and per-tooth residual magnitudes."""

    groups = defaultdict(list)
    for row in rows:
        groups[(row["phase"], row["view_id"], row["segment"])].append(
            row["residual_pitch_fraction"]
        )
    result = []
    for (phase, view_id, segment), values in sorted(groups.items()):
        array = np.asarray(values, dtype=float)
        result.append(
            {
                "phase": phase,
                "segment": segment,
                "samples": len(array),
                "view_id": view_id,
                "residual_pitch_fraction_median": float(np.median(array)),
                "residual_pitch_fraction_p95": float(np.quantile(array, 0.95)),
                "residual_pitch_fraction_maximum": float(np.max(array)),
            }
        )
    return result


def compare_aligned_runs(
    first_bundle, first_result, second_bundle, second_result
):
    """Compare runs only when physics and every residual key align."""

    if (
        first_bundle["physics_trace_sha256"]
        != second_bundle["physics_trace_sha256"]
    ):
        raise EvidenceError("A/B physics state traces differ")
    for key in ("camera_rig", "sampling"):
        if first_bundle["manifest"].get(key) != second_bundle[
            "manifest"
        ].get(key):
            raise EvidenceError(f"A/B {key} contracts differ")
    if first_bundle["palette"] != second_bundle["palette"]:
        raise EvidenceError("A/B color palettes differ")
    first_treatment = {
        "render_mode": first_bundle["manifest"]["render_settings"].get(
            "mode"
        ),
        "normalization_ab": first_bundle["physics_report"].get(
            "normalization_ab"
        ),
        "segment00_schema": first_bundle["physics_report"].get(
            "segment00_schema"
        ),
    }
    second_treatment = {
        "render_mode": second_bundle["manifest"]["render_settings"].get(
            "mode"
        ),
        "normalization_ab": second_bundle["physics_report"].get(
            "normalization_ab"
        ),
        "segment00_schema": second_bundle["physics_report"].get(
            "segment00_schema"
        ),
    }
    if first_treatment == second_treatment:
        raise EvidenceError("A/B render and schema treatments are identical")

    def sample_keys(result):
        return [
            (
                frame["global_step"],
                frame["phase"],
                frame["phase_step"],
            )
            for frame in result["frames"]
        ]

    if sample_keys(first_result) != sample_keys(second_result):
        raise EvidenceError("A/B synchronized sample keys differ")
    priority = tuple(
        first_bundle["manifest"]["camera_rig"]["view_priority"]
    )
    rows = []
    for index in range(1, len(first_result["frames"])):
        first_previous = first_result["frames"][index - 1]
        first_current = first_result["frames"][index]
        second_previous = second_result["frames"][index - 1]
        second_current = second_result["frames"][index]
        if first_previous["phase"] != first_current["phase"]:
            continue
        selected_view = None
        selected_segments = None
        common_by_view = {}
        for view_id in priority:
            frame_views = (
                first_previous["views"],
                first_current["views"],
                second_previous["views"],
                second_current["views"],
            )
            if not all(view_id in item for item in frame_views):
                common_by_view[view_id] = []
                continue
            common = sorted(
                set.intersection(
                    *[
                        set(item[view_id]["points"])
                        for item in frame_views
                    ]
                )
            )
            common_by_view[view_id] = common
            if len(common) >= MINIMUM_VISIBLE_TEETH:
                selected_view = view_id
                selected_segments = common
                break
        if selected_view is None:
            raise EvidenceError(
                "A/B has no common prioritized view with four teeth at "
                f"global_step={first_current['global_step']} "
                f"phase={first_current['phase']} phase_step="
                f"{first_current['phase_step']} common_by_view="
                f"{json.dumps(common_by_view, sort_keys=True)}"
            )
        first_rows = _measure_transition(
            first_previous,
            first_current,
            selected_view,
            selected_segments,
        )
        second_rows = _measure_transition(
            second_previous,
            second_current,
            selected_view,
            selected_segments,
        )
        for left, right in zip(first_rows, second_rows):
            rows.append(
                {
                    "global_step": left["global_step"],
                    "phase": left["phase"],
                    "phase_step": left["phase_step"],
                    "segment": left["segment"],
                    "view_id": selected_view,
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
    return rows


def summarize_ab_residuals(rows):
    """Aggregate aligned A/B residuals without discarding phase or tooth ID."""

    groups = defaultdict(list)
    for row in rows:
        groups[(row["phase"], row["view_id"], row["segment"])].append(row)
    summaries = []
    for (phase, view_id, segment), values in sorted(groups.items()):
        first = np.asarray(
            [item["first_residual_pitch_fraction"] for item in values],
            dtype=float,
        )
        second = np.asarray(
            [item["second_residual_pitch_fraction"] for item in values],
            dtype=float,
        )
        delta = second - first
        summaries.append(
            {
                "phase": phase,
                "segment": segment,
                "samples": len(values),
                "view_id": view_id,
                "first_residual_pitch_fraction_p95": float(
                    np.quantile(first, 0.95)
                ),
                "second_residual_pitch_fraction_p95": float(
                    np.quantile(second, 0.95)
                ),
                "second_minus_first_pitch_fraction_median": float(
                    np.median(delta)
                ),
                "second_minus_first_pitch_fraction_p95_absolute": float(
                    np.quantile(np.abs(delta), 0.95)
                ),
            }
        )
    return summaries


def _write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_analysis(capture, output, helper, compare=None):
    """Run one strict capture analysis and an optional aligned two-run A/B."""

    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    artifact_names = (
        "report.json",
        "per_tooth_residuals.csv",
        "per_tooth_summary.csv",
        "second_per_tooth_residuals.csv",
        "second_per_tooth_summary.csv",
        "ab_aligned_residuals.csv",
        "ab_per_tooth_summary.csv",
    )
    # A failed rerun must not leave previously authorized CSVs beside its
    # rejection report.  Only this analyzer's fixed filenames inside the
    # explicitly requested output directory are replaced.
    for name in artifact_names:
        (output / name).unlink(missing_ok=True)
    first_bundle = validate_capture_bundle(capture, helper)
    first_result = analyze_validated_capture(first_bundle)
    first_summary = summarize_residuals(first_result["residual_rows"])
    report = {
        "schema_version": SCHEMA_VERSION,
        "passed": True,
        "comparison_authorized": False,
        "analysis_source": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "strict_gates": {
            "capture_helper_sha256": sha256_file(helper),
            "fixed_four_camera_rig_exact": True,
            "minimum_adjacent_pitch_px": MINIMUM_TOOTH_PITCH_PX,
            "minimum_adjacent_pairs": MINIMUM_ADJACENT_PAIRS,
            "minimum_visible_teeth": MINIMUM_VISIBLE_TEETH,
            "physics_report_hash_bound": True,
            "physics_summary_hash_bound": True,
        },
        "single_run": {
            "capture_directory": str(first_bundle["capture_directory"]),
            "capture_manifest_sha256": first_bundle["manifest_sha256"],
            "frames_analyzed": first_result["frames_analyzed"],
            "minimum_tooth_pitch_px": first_result["minimum_tooth_pitch_px"],
            "physics_trace_sha256": first_bundle["physics_trace_sha256"],
            "rgb_frames_validated": first_result[
                "rgb_frames_validated"
            ],
            "segments_observed": first_result["segments_observed"],
            "strict_global_step_phase_alignment": True,
            "summary": first_summary,
            "valid_sample_counts_by_view": first_result[
                "valid_sample_counts_by_view"
            ],
            "view_usage_by_transition": first_result[
                "view_usage_by_transition"
            ],
        },
        "comparison": None,
    }
    second_result = None
    second_summary = None
    if compare is not None:
        second_bundle = validate_capture_bundle(compare, helper)
        second_result = analyze_validated_capture(second_bundle)
        second_summary = summarize_residuals(second_result["residual_rows"])
        comparison_rows = compare_aligned_runs(
            first_bundle, first_result, second_bundle, second_result
        )
        comparison_summary = summarize_ab_residuals(comparison_rows)
        deltas = np.asarray(
            [
                row["second_minus_first_pitch_fraction"]
                for row in comparison_rows
            ]
        )
        report["comparison"] = {
            "authorized": True,
            "first_capture": str(first_bundle["capture_directory"]),
            "first_render_mode": first_bundle["manifest"][
                "render_settings"
            ]["mode"],
            "keys_compared": len(comparison_rows),
            "median_second_minus_first_pitch_fraction": float(
                np.median(deltas)
            ),
            "p95_absolute_second_minus_first_pitch_fraction": float(
                np.quantile(np.abs(deltas), 0.95)
            ),
            "physics_trace_identical": True,
            "second_capture": str(second_bundle["capture_directory"]),
            "second_render_mode": second_bundle["manifest"][
                "render_settings"
            ]["mode"],
            "strict_global_step_phase_alignment": True,
            "summary": comparison_summary,
            "view_usage_by_residual": dict(
                sorted(
                    {
                        view_id: sum(
                            row["view_id"] == view_id
                            for row in comparison_rows
                        )
                        for view_id in VIEW_IDS
                    }.items()
                )
            ),
        }
        report["comparison_authorized"] = True
    _write_csv(
        output / "per_tooth_residuals.csv", first_result["residual_rows"]
    )
    _write_csv(output / "per_tooth_summary.csv", first_summary)
    if compare is not None:
        _write_csv(
            output / "second_per_tooth_residuals.csv",
            second_result["residual_rows"],
        )
        _write_csv(
            output / "second_per_tooth_summary.csv", second_summary
        )
        _write_csv(output / "ab_aligned_residuals.csv", comparison_rows)
        _write_csv(
            output / "ab_per_tooth_summary.csv", comparison_summary
        )
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _arguments(argv=None):
    repository = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description="Analyze hash-bound synchronized D38999 tooth PNG frames"
    )
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument(
        "--compare",
        type=Path,
        help="optional second synchronized capture, for example history512",
    )
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


def main(argv=None):
    arguments = _arguments(argv)
    try:
        report = run_analysis(
            arguments.capture,
            arguments.output,
            arguments.capture_helper,
            compare=arguments.compare,
        )
    except (EvidenceError, OSError, ValueError, json.JSONDecodeError) as error:
        failure = {
            "schema_version": SCHEMA_VERSION,
            "passed": False,
            "comparison_authorized": False,
            "analysis_source": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "capture_requested": str(arguments.capture.resolve()),
            "compare_requested": (
                str(arguments.compare.resolve())
                if arguments.compare is not None
                else None
            ),
            "error": str(error),
        }
        if arguments.capture_helper.is_file():
            failure["capture_helper_sha256"] = sha256_file(
                arguments.capture_helper
            )
        try:
            arguments.output.mkdir(parents=True, exist_ok=True)
            (arguments.output / "report.json").write_text(
                json.dumps(failure, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except OSError:
            pass
        print(json.dumps(failure, sort_keys=True))
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
