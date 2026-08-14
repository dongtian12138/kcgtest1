#!/usr/bin/env python3

"""Strict, CPU-only audit of D38999 colour-ID GUI recordings.

The GUI recordings are useful visual evidence, but a recording tail is not a
physics phase.  This module therefore keeps two results separate:

* an exploratory colour-segmentation measurement, after registering the
  visible nut teeth as one 2-D rigid/similarity body; and
* a strict comparison gate which requires a hash-bound frame/global-step
  sidecar before it permits a render-mode A/B conclusion.

OpenCV is imported only by the video-reading functions.  The registration,
phase-audit and comparison-gate contracts remain pure NumPy/Python and are
covered without Isaac Sim or a GPU.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA_VERSION = "kcg_d38999_tooth_video_analysis_v1"
RUN_MODES = ("baseline", "rtx_history512", "fabric_disabled")
TRACK_SEGMENTS = (
    "Segment_21",
    "Segment_22",
    "Segment_23",
    "Segment_00",
    "Segment_01",
)
SYNC_COLUMNS = ("video_frame_index", "global_step", "phase", "phase_step")


def sha256_file(path):
    """Return the SHA-256 of *path* without loading a large video in memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def physics_trace_sha256(summary_path):
    """Hash physics/state columns, excluding verbose contact-record text."""

    digest = hashlib.sha256()
    with Path(summary_path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = [
            name
            for name in (reader.fieldnames or ())
            if name != "segment_contact_records"
        ]
        digest.update(("\x1f".join(columns) + "\n").encode("utf-8"))
        for row in reader:
            digest.update(
                ("\x1f".join(row[name] for name in columns) + "\n").encode(
                    "utf-8"
                )
            )
    return digest.hexdigest()


def ordered_phase_ranges(summary_path):
    """Read contiguous physics-phase ranges from a jitter summary CSV."""

    ranges = []
    with Path(summary_path).open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            step = int(row["global_step"])
            phase = row["phase"]
            phase_step = int(row["phase_step"])
            if not ranges or ranges[-1]["phase"] != phase:
                ranges.append(
                    {
                        "phase": phase,
                        "first_global_step": step,
                        "last_global_step": step,
                        "steps": phase_step,
                    }
                )
            else:
                ranges[-1]["last_global_step"] = step
                ranges[-1]["steps"] = phase_step
    return ranges


def fit_similarity_2d(reference, observed):
    """Fit the least-squares 2-D similarity mapping reference to observed.

    Returning residual vectors rather than only a scalar makes it possible to
    distinguish coherent nut/camera motion from one tooth moving relative to
    the rest of the ring.
    """

    source = np.asarray(reference, dtype=float)
    target = np.asarray(observed, dtype=float)
    if (
        source.shape != target.shape
        or source.ndim != 2
        or source.shape[1] != 2
    ):
        raise ValueError("reference and observed must both have shape (N, 2)")
    if source.shape[0] < 2:
        raise ValueError("at least two points are required for registration")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_zero = source - source_mean
    target_zero = target - target_mean
    denominator = float(np.sum(source_zero * source_zero))
    if denominator <= np.finfo(float).eps:
        raise ValueError("reference points are degenerate")
    left, singular, right_t = np.linalg.svd(source_zero.T @ target_zero)
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_t
    scale = float(np.sum(singular) / denominator)
    translation = target_mean - scale * source_mean @ rotation
    predicted = scale * source @ rotation + translation
    return {
        "scale": scale,
        "rotation": rotation,
        "translation": translation,
        "residual_vectors": target - predicted,
    }


def _median_pitch(points, ordered_segments=TRACK_SEGMENTS):
    distances = []
    for first, second in zip(ordered_segments[:-1], ordered_segments[1:]):
        if first in points and second in points:
            distances.append(
                float(
                    np.linalg.norm(np.asarray(points[second]) - points[first])
                )
            )
    return float(np.median(distances)) if distances else 0.0


def registered_relative_residual_rows(
    frames,
    *,
    min_common_segments=4,
    minimum_pitch_px=12.0,
    maximum_adjacent_scale_change=0.20,
):
    """Register adjacent frames and return per-tooth relative residual rows.

    Adjacent registration tolerates smooth GUI camera navigation.  Resolution
    and scale-jump gates remain explicit because a six-pixel tooth or a sudden
    scroll-wheel zoom cannot support a sub-pixel jitter claim.
    """

    result = []
    for previous, current in zip(frames[:-1], frames[1:]):
        common = [
            segment
            for segment in TRACK_SEGMENTS
            if segment in previous["points"] and segment in current["points"]
        ]
        if len(common) < min_common_segments:
            continue
        reference = np.asarray([previous["points"][name] for name in common])
        observed = np.asarray([current["points"][name] for name in common])
        try:
            fit = fit_similarity_2d(reference, observed)
        except ValueError:
            continue
        pitch = 0.5 * (
            _median_pitch(previous["points"])
            + _median_pitch(current["points"])
        )
        scale_jump = abs(fit["scale"] - 1.0)
        accepted = bool(
            pitch >= minimum_pitch_px
            and scale_jump <= maximum_adjacent_scale_change
        )
        refusal = []
        if pitch < minimum_pitch_px:
            refusal.append("visible_tooth_pitch_below_resolution_gate")
        if scale_jump > maximum_adjacent_scale_change:
            refusal.append("adjacent_camera_or_scene_scale_jump")
        for index, segment in enumerate(common):
            vector = fit["residual_vectors"][index]
            residual = float(np.linalg.norm(vector))
            result.append(
                {
                    "previous_frame": int(previous["frame_index"]),
                    "video_frame_index": int(current["frame_index"]),
                    "time_s": float(current["time_s"]),
                    "segment": segment,
                    "residual_x_px": float(vector[0]),
                    "residual_y_px": float(vector[1]),
                    "residual_px": residual,
                    "tooth_pitch_px": pitch,
                    "residual_pitch_fraction": (
                        residual / pitch if pitch else None
                    ),
                    "registered_scale": float(fit["scale"]),
                    "accepted_resolution_and_motion": accepted,
                    "measurement_refusal_reasons": ";".join(refusal),
                }
            )
    return result


def summarize_residual_rows(rows):
    """Summarize only rows which passed image-resolution/motion gates."""

    accepted = [row for row in rows if row["accepted_resolution_and_motion"]]
    values = np.asarray(
        [row["residual_pitch_fraction"] for row in accepted], dtype=float
    )
    pitches = np.asarray(
        [row["tooth_pitch_px"] for row in accepted], dtype=float
    )
    return {
        "transition_rows": len(rows),
        "accepted_transition_rows": len(accepted),
        "accepted_fraction": len(accepted) / len(rows) if rows else 0.0,
        "median_tooth_pitch_px": (
            float(np.median(pitches)) if len(pitches) else None
        ),
        "relative_residual_pitch_fraction_median": (
            float(np.median(values)) if len(values) else None
        ),
        "relative_residual_pitch_fraction_p95": (
            float(np.quantile(values, 0.95)) if len(values) else None
        ),
        "strict_phase_aligned_measurement": False,
    }


def explicit_sync_status(run_directory, hashes):
    """Validate a future hash-bound video-frame/global-step sidecar.

    The existing captures intentionally fail this contract.  A plain CSV is
    insufficient: the manifest must bind the video, physics report and sidecar
    hashes so that files from separate runs cannot be accidentally mixed.
    """

    directory = Path(run_directory)
    manifest_path = directory / "video_capture_manifest.json"
    sidecar_path = directory / "video_frame_sync.csv"
    reasons = []
    if not manifest_path.is_file():
        reasons.append("missing_video_capture_manifest.json")
    if not sidecar_path.is_file():
        reasons.append("missing_video_frame_sync.csv")
    if reasons:
        return {"valid": False, "reasons": reasons}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        with sidecar_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            columns = tuple(reader.fieldnames or ())
            rows = list(reader)
    except (OSError, ValueError, csv.Error) as exc:
        return {"valid": False, "reasons": [f"invalid_sync_sidecar:{exc}"]}
    missing = [column for column in SYNC_COLUMNS if column not in columns]
    if missing:
        reasons.append("missing_sync_columns:" + ",".join(missing))
    expected = {
        "video_sha256": hashes["video_sha256"],
        "report_sha256": hashes["report_sha256"],
        "frame_sync_sha256": sha256_file(sidecar_path),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            reasons.append(f"manifest_hash_mismatch:{key}")
    if not rows:
        reasons.append("empty_video_frame_sync.csv")
    return {
        "valid": not reasons,
        "reasons": reasons,
        "rows": len(rows),
        "manifest": str(manifest_path),
        "sidecar": str(sidecar_path),
    }


def comparison_refusal_reasons(run_audits):
    """Return strict cross-mode comparison blockers."""

    reasons = []
    phase_signatures = {
        json.dumps(run["physics"]["phase_ranges"], sort_keys=True)
        for run in run_audits.values()
    }
    if len(phase_signatures) != 1:
        reasons.append("physics_phase_sequences_differ")
    physics_traces = {
        run["hashes"]["physics_trace_sha256"]
        for run in run_audits.values()
    }
    if len(physics_traces) != 1:
        reasons.append("physics_state_traces_differ_across_render_modes")
    missing_sync = [
        mode for mode, run in run_audits.items() if not run["sync"]["valid"]
    ]
    if missing_sync:
        reasons.append(
            "no_hash_bound_frame_to_global_step_sync:"
            + ",".join(missing_sync)
        )
    poor_visual = [
        mode
        for mode, run in run_audits.items()
        if run["exploratory_visual"]["accepted_fraction"] < 0.80
    ]
    if poor_visual:
        reasons.append("visual_quality_gate_failed:" + ",".join(poor_visual))
    return reasons


def _video_metadata(path):
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise RuntimeError(
            "OpenCV is required to read GUI recordings"
        ) from exc
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if fps <= 0.0 or frames <= 0:
        raise RuntimeError(f"invalid video metadata: {path}")
    return {
        "fps": fps,
        "frames": frames,
        "duration_s": frames / fps,
        "width": width,
        "height": height,
    }


def _colour_points(frame):
    """Find robust visible tooth centroids in the fixed GUI viewport region."""

    import cv2

    height, width = frame.shape[:2]
    # The crop contains the lower viewport, excluding the orange background
    # sculpture and nearly all desktop/UI colour.  Coordinates remain absolute.
    x0, x1 = int(0.085 * width), int(0.47 * width)
    y0, y1 = int(0.37 * height), int(0.67 * height)
    hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    # Palette hues are uniformly spaced.  Nearest-bin assignment is exclusive;
    # overlapping hue masks would let adjacent red teeth share the same pixels.
    labels = np.floor((hue.astype(float) + 3.75) / 7.5).astype(np.int16) % 24
    valid = (saturation >= 50) & (value >= 50)
    points = {}
    for segment in TRACK_SEGMENTS:
        index = int(segment.rsplit("_", 1)[1])
        mask = ((labels == index) & valid).astype(np.uint8) * 255
        count, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
        candidates = []
        for component in range(1, count):
            x, y, component_width, component_height, area = (
                int(item) for item in stats[component]
            )
            if (
                area >= 10
                and component_height >= 4
                and component_width <= 150
                and component_height <= 220
                and component_height >= 0.60 * component_width
            ):
                candidates.append((area, component))
        if candidates:
            component = max(candidates)[1]
            center = centroids[component]
            points[segment] = (float(center[0] + x0), float(center[1] + y0))
    return points


def exploratory_terminal_tracks(
    video_path, metadata, seconds=3.0, end_margin_s=0.6
):
    """Extract an explicitly non-phase-aligned terminal feasibility window."""

    import cv2

    fps = metadata["fps"]
    end_frame = max(1, metadata["frames"] - int(round(end_margin_s * fps)))
    start_frame = max(0, end_frame - int(round(seconds * fps)))
    capture = cv2.VideoCapture(str(video_path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for frame_index in range(start_frame, end_frame):
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(
            {
                "frame_index": frame_index,
                "time_s": frame_index / fps,
                "points": _colour_points(frame),
            }
        )
    capture.release()
    return frames


def _physics_summary(report, summary_path):
    aggregates = report["segment_aggregate"]
    return {
        "steps": int(report["steps"]),
        "phase_ranges": ordered_phase_ranges(summary_path),
        "anomaly_steps": int(report["anomaly_steps"]),
        "maximum_parent_relative_translation_error_m": max(
            float(value["maximum_parent_relative_translation_error_m"])
            for value in aggregates.values()
        ),
        "maximum_parent_relative_rotation_error_rad": max(
            float(value["maximum_parent_relative_rotation_error_rad"])
            for value in aggregates.values()
        ),
    }


def audit_run(run_directory, exploratory_seconds=3.0):
    """Audit one recording/report/summary and probe visual feasibility."""

    directory = Path(run_directory)
    video_path = directory / "gui_fullscreen.mp4"
    report_path = directory / "report.json"
    summary_path = directory / "summary.csv"
    for path in (video_path, report_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    hashes = {
        "video_sha256": sha256_file(video_path),
        "report_sha256": sha256_file(report_path),
        "summary_sha256": sha256_file(summary_path),
        "physics_trace_sha256": physics_trace_sha256(summary_path),
    }
    report = json.loads(report_path.read_text(encoding="utf-8"))
    metadata = _video_metadata(video_path)
    frames = exploratory_terminal_tracks(
        video_path, metadata, seconds=exploratory_seconds
    )
    residuals = registered_relative_residual_rows(frames)
    visual = summarize_residual_rows(residuals)
    visual.update(
        {
            "window_kind": "exploratory_unsynchronized_terminal_window",
            "window_seconds": exploratory_seconds,
            "cross_mode_use_authorized": False,
            "warning": (
                "This window is not bound to a physics phase and must not be "
                "used to rank render modes."
            ),
        }
    )
    return {
        "hashes": hashes,
        "video": metadata,
        "physics": _physics_summary(report, summary_path),
        "render_mode": report["render_ab_launch"]["mode"],
        "sync": explicit_sync_status(directory, hashes),
        "exploratory_visual": visual,
        "residual_rows": residuals,
    }


def minimum_recapture_contract():
    """Describe the smallest recapture which can close the evidence gap."""

    return {
        "camera": (
            "fixed rear-oblique camera, authored before play, with Segment_00 "
            "and its neighbours visible at >=12 px tooth pitch"
        ),
        "capture": (
            "direct RenderProduct RGB plus semantic/instance IDs; no GUI "
            "camera "
            "navigation or desktop screen recording"
        ),
        "phases": ["nut_only_final_hold", "q7_twist_probe_hold"],
        "sidecar": {
            "csv": "video_frame_sync.csv",
            "required_columns": list(SYNC_COLUMNS),
            "manifest": "video_capture_manifest.json",
            "required_hashes": [
                "video_sha256",
                "report_sha256",
                "frame_sync_sha256",
            ],
        },
        "comparison": (
            "register all visible nut teeth first, then compare each tooth's "
            "relative residual on identical named phase ranges"
        ),
    }


def _write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze_root(root, output_directory, exploratory_seconds=3.0):
    """Audit all three modes and write strict JSON/CSV evidence artifacts."""

    root = Path(root).expanduser().resolve()
    output = Path(output_directory).expanduser().resolve()
    audits = {}
    residual_rows = []
    for mode in RUN_MODES:
        run = audit_run(root / f"render_{mode}", exploratory_seconds)
        audits[mode] = run
        for row in run.pop("residual_rows"):
            residual_rows.append(
                {
                    "mode": mode,
                    "phase_alignment": "none_exploratory_only",
                    **row,
                }
            )
    refusals = comparison_refusal_reasons(audits)
    report = {
        "schema_version": SCHEMA_VERSION,
        "source_root": str(root),
        "comparison_authorized": not refusals,
        "comparison_refusal_reasons": refusals,
        "runs": audits,
        "minimum_recapture_contract": minimum_recapture_contract(),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    comparison_rows = []
    for mode, run in audits.items():
        visual = run["exploratory_visual"]
        comparison_rows.append(
            {
                "mode": mode,
                "video_sha256": run["hashes"]["video_sha256"],
                "duration_s": run["video"]["duration_s"],
                "physics_steps": run["physics"]["steps"],
                "physics_anomaly_steps": run["physics"]["anomaly_steps"],
                "frame_step_sync_valid": run["sync"]["valid"],
                "exploratory_accepted_fraction": visual["accepted_fraction"],
                "exploratory_median_tooth_pitch_px": visual[
                    "median_tooth_pitch_px"
                ],
                "exploratory_residual_pitch_fraction_p95": visual[
                    "relative_residual_pitch_fraction_p95"
                ],
                "cross_mode_comparison_authorized": not refusals,
            }
        )
    _write_csv(
        output / "comparison.csv", list(comparison_rows[0]), comparison_rows
    )
    if residual_rows:
        _write_csv(
            output / "exploratory_frame_residuals.csv",
            list(residual_rows[0]),
            residual_rows,
        )
    return report


def _parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Audit three D38999 GUI jitter captures without treating "
            "unmatched "
            "video tails as a phase-aligned A/B comparison."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/kcg_connector/d38999_nut_tooth_jitter"),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--exploratory-seconds", type=float, default=3.0)
    parser.add_argument(
        "--require-comparable",
        action="store_true",
        help="exit 2 when strict hash-bound phase alignment is unavailable",
    )
    arguments = parser.parse_args(argv)
    if arguments.exploratory_seconds <= 0.5:
        parser.error("--exploratory-seconds must be greater than 0.5")
    if arguments.output is None:
        arguments.output = arguments.root / "offline_visual_analysis_v1"
    return arguments


def main(argv=None):
    arguments = _parse_arguments(argv)
    report = analyze_root(
        arguments.root,
        arguments.output,
        exploratory_seconds=arguments.exploratory_seconds,
    )
    print(json.dumps(report, sort_keys=True))
    if arguments.require_comparable and not report["comparison_authorized"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
