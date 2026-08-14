"""CPU-only strict evidence tests for synchronized tooth-frame analysis."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from kcg_connector import d38999_tooth_sync_analysis as analysis


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_frame(path, palette, offset_x, tooth_jitter_y):
    image = np.zeros((720, 960, 3), dtype=np.uint8)
    for position, index in enumerate((21, 22, 23, 0, 1)):
        rgb = np.asarray(palette[f"Segment_{index:02d}"]) * 255.0
        bgr = tuple(int(round(value)) for value in rgb[::-1])
        center_x = 330 + 30 * position + offset_x
        center_y = 350 + (tooth_jitter_y if index == 0 else 0)
        cv2.rectangle(
            image,
            (center_x - 8, center_y - 30),
            (center_x + 8, center_y + 30),
            bgr,
            thickness=-1,
        )
    assert cv2.imwrite(str(path), image)


def _make_bundle(
    tmp_path,
    name,
    helper,
    jitter_px=3,
    render_mode="baseline",
    invalid_views=(),
):
    root = tmp_path / name
    capture = root / "capture"
    physics = root / "physics"
    frames = capture / "frames"
    frames.mkdir(parents=True)
    physics.mkdir(parents=True)
    palette = analysis.deterministic_segment_colors()
    history_count = 512 if render_mode == "rtx_history_512" else 8
    render_settings = {
        "actual": {
            "/app/useFabricSceneDelegate": True,
            "/rtx/scenedb/maxHistoryTransformCount": history_count,
        },
        "exact_match": True,
        "extra_args": [],
        "mismatches": [],
        "mode": render_mode,
        "requested": (
            {"/rtx/scenedb/maxHistoryTransformCount": 512}
            if render_mode == "rtx_history_512"
            else {}
        ),
        "validated_after_simulation_app_start": True,
    }
    phase_totals = {phase: 16 for phase in analysis.CAPTURE_PHASES}
    report = {
        "schema_version": analysis.PHYSICS_SCHEMA_VERSION,
        "steps": 48,
        "phase_steps": phase_totals,
        "color_identification": {
            "authored_in_session_layer": True,
            "colors_rgb": palette,
        },
        "render_ab_launch": render_settings,
    }
    report_path = physics / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_path = physics / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "global_step",
                "phase",
                "phase_step",
                "parent_px_m",
                "segment_contact_records",
            ),
        )
        writer.writeheader()
        global_step = 0
        for phase in analysis.CAPTURE_PHASES:
            for phase_step in range(1, 17):
                global_step += 1
                writer.writerow(
                    {
                        "global_step": global_step,
                        "phase": phase,
                        "phase_step": phase_step,
                        "parent_px_m": "0.55",
                        "segment_contact_records": "0",
                    }
                )

    sync_rows = []
    frame_hashes = {}
    frame_index = 0
    sample_index = 0
    invalid_views = set(invalid_views)
    for phase_index, phase in enumerate(analysis.CAPTURE_PHASES):
        for local_index, phase_step in enumerate((1, 8, 16)):
            global_step = phase_index * 16 + phase_step
            for view_index, view_id in enumerate(analysis.VIEW_IDS):
                relative = (
                    f"frames/{view_id}/frame_{sample_index:06d}.png"
                )
                frame_path = capture / relative
                frame_path.parent.mkdir(exist_ok=True)
                if (sample_index, view_id) in invalid_views:
                    blank = np.zeros((720, 960, 3), dtype=np.uint8)
                    assert cv2.imwrite(str(frame_path), blank)
                else:
                    jitter_y = jitter_px if local_index == 1 else 0
                    _write_frame(
                        frame_path,
                        palette,
                        offset_x=local_index + 5 * view_index,
                        tooth_jitter_y=jitter_y,
                    )
                frame_hashes[relative] = _sha256(frame_path)
                sync_rows.append(
                    {
                        "frame_index": frame_index,
                        "sample_index": sample_index,
                        "view_id": view_id,
                        "global_step": global_step,
                        "phase": phase,
                        "phase_step": phase_step,
                        "simulation_time_s": global_step / 240.0,
                        "timestamp_s": 100.0 + frame_index,
                        "rgb_filename": relative,
                    }
                )
                frame_index += 1
            sample_index += 1
    sync_path = capture / "video_frame_sync.csv"
    with sync_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=analysis.SYNC_FIELDS)
        writer.writeheader()
        writer.writerows(sync_rows)
    manifest = {
        "schema_version": analysis.CAPTURE_SCHEMA_VERSION,
        "passed": True,
        "camera_rig": analysis.EXPECTED_CAMERA_RIG,
        "capture_source": {
            "path": str(helper),
            "sha256_at_import": _sha256(helper),
            "sha256_at_start": _sha256(helper),
            "sha256_at_finalize": _sha256(helper),
            "unchanged_during_capture": True,
        },
        "cleanup": {
            "annotator_detached": True,
            "annotators_detached_count": len(analysis.VIEW_IDS),
            "camera_prim_removed": True,
            "camera_prims_removed_count": len(analysis.VIEW_IDS),
            "errors": [],
            "object_pose_writes": 0,
            "render_product_destroyed": True,
            "render_products_destroyed_count": len(analysis.VIEW_IDS),
            "resources_released": True,
            "stage_cleared": False,
            "view_count": len(analysis.VIEW_IDS),
            "world_reset": False,
        },
        "frame_capture": {
            "first_global_step": 1,
            "frame_count": len(sync_rows),
            "frames_per_view": {
                view_id: sample_index for view_id in analysis.VIEW_IDS
            },
            "last_global_step": 48,
            "phases": list(analysis.CAPTURE_PHASES),
            "sample_count": sample_index,
            "view_order": list(analysis.VIEW_IDS),
        },
        "frame_files_sha256": frame_hashes,
        "physics_evidence": {
            "report_path": str(report_path),
            "report_sha256": _sha256(report_path),
            "summary_path": str(summary_path),
            "summary_sha256": _sha256(summary_path),
            "tooth_color_ids_authored": True,
            "tooth_color_id_count": 24,
        },
        "render_settings": render_settings,
        "sampling": {
            "capture_rate_hz": 30,
            "physics_rate_hz": 240,
            "physics_steps_per_frame": 8,
            "sampling_kind": "fixed_integer_physics_step_decimation",
        },
        "step_mapping_semantics": (
            "frame read immediately after world.step(render=True); mapped to "
            "the most recently completed physics global_step"
        ),
        "sync_columns": list(analysis.SYNC_FIELDS),
        "sync_csv": "video_frame_sync.csv",
        "sync_csv_sha256": _sha256(sync_path),
    }
    manifest_path = capture / "video_capture_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return capture, manifest_path


@pytest.fixture
def helper(tmp_path):
    path = tmp_path / "d38999_tooth_sync_capture.py"
    path.write_text("# frozen capture helper\n", encoding="utf-8")
    return path


def test_valid_bundle_is_hash_bound_and_residuals_find_jitter(
    tmp_path, helper
):
    capture, _ = _make_bundle(tmp_path, "baseline", helper)
    bundle = analysis.validate_capture_bundle(capture, helper)
    result = analysis.analyze_validated_capture(bundle)
    assert result["frames_analyzed"] == 9
    assert result["rgb_frames_validated"] == 9 * len(analysis.VIEW_IDS)
    assert result["minimum_tooth_pitch_px"] >= 12.0
    assert "Segment_00" in result["segments_observed"]
    by_segment = {}
    for row in result["residual_rows"]:
        by_segment.setdefault(row["segment"], []).append(row["residual_px"])
    assert max(by_segment["Segment_00"]) > max(
        value
        for segment, values in by_segment.items()
        if segment != "Segment_00"
        for value in values
    )


def test_pitch_cluster_excludes_short_occluded_edge():
    points = {
        "Segment_13": np.asarray([0.0, 0.0]),
        "Segment_14": np.asarray([10.0, 0.0]),
        "Segment_16": np.asarray([50.0, 0.0]),
        "Segment_17": np.asarray([70.0, 0.0]),
        "Segment_18": np.asarray([90.0, 0.0]),
        "Segment_19": np.asarray([110.0, 0.0]),
        "Segment_20": np.asarray([130.0, 0.0]),
    }
    selected, pitches = analysis.select_pitch_qualified_cluster(points)
    assert set(selected) == {
        "Segment_16",
        "Segment_17",
        "Segment_18",
        "Segment_19",
        "Segment_20",
    }
    assert min(pitches.values()) >= analysis.MINIMUM_TOOTH_PITCH_PX


def test_fixed_priority_falls_back_to_complementary_view(tmp_path, helper):
    capture, _ = _make_bundle(
        tmp_path,
        "fallback",
        helper,
        invalid_views=((1, "rear_left"),),
    )
    bundle = analysis.validate_capture_bundle(capture, helper)
    result = analysis.analyze_validated_capture(bundle)
    assert result["valid_sample_counts_by_view"] == {
        "rear_left": 8,
        "rear_right": 9,
        "front_left": 9,
        "front_right": 9,
    }
    assert result["view_usage_by_transition"]["rear_right"] == 2
    assert result["view_usage_by_transition"]["rear_left"] == 4
    assert result["view_usage_by_transition"]["front_left"] == 0
    assert result["view_usage_by_transition"]["front_right"] == 0


def test_ab_uses_same_common_view_without_residual_shopping(
    tmp_path, helper
):
    first_capture, _ = _make_bundle(
        tmp_path,
        "first",
        helper,
        invalid_views=((1, "rear_left"),),
    )
    second_capture, _ = _make_bundle(
        tmp_path,
        "second",
        helper,
        render_mode="rtx_history_512",
    )
    first_bundle = analysis.validate_capture_bundle(first_capture, helper)
    second_bundle = analysis.validate_capture_bundle(second_capture, helper)
    first_result = analysis.analyze_validated_capture(first_bundle)
    second_result = analysis.analyze_validated_capture(second_bundle)
    rows = analysis.compare_aligned_runs(
        first_bundle, first_result, second_bundle, second_result
    )
    affected = [
        row
        for row in rows
        if row["global_step"] in (8, 16)
        and row["phase"] == "nut_only_final_hold"
    ]
    assert affected
    assert {row["view_id"] for row in affected} == {"rear_right"}


@pytest.mark.parametrize("tamper", ("camera", "helper", "frame"))
def test_bundle_tampering_fails_closed(tmp_path, helper, tamper):
    capture, manifest_path = _make_bundle(tmp_path, tamper, helper)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if tamper == "camera":
        manifest["camera_rig"]["views"][0]["eye_m"][0] += 0.01
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        expected = "four-camera rig"
    elif tamper == "helper":
        helper.write_text("# changed helper\n", encoding="utf-8")
        expected = "capture helper SHA-256"
    else:
        target = capture / "frames/rear_left/frame_000000.png"
        target.write_bytes(b"tampered")
        expected = "frame SHA-256"
    with pytest.raises(analysis.EvidenceError, match=expected):
        analysis.validate_capture_bundle(capture, helper)


def test_incomplete_phase_frame_plan_fails_even_when_csv_rehashed(
    tmp_path, helper
):
    capture, manifest_path = _make_bundle(tmp_path, "missing", helper)
    sync_path = capture / "video_frame_sync.csv"
    with sync_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    start = len(analysis.VIEW_IDS)
    removed_rows = rows[start : start + len(analysis.VIEW_IDS)]  # noqa: E203
    del rows[start : start + len(analysis.VIEW_IDS)]  # noqa: E203
    for index, row in enumerate(rows):
        row["frame_index"] = index
        row["sample_index"] = index // len(analysis.VIEW_IDS)
    with sync_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=analysis.SYNC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sync_csv_sha256"] = _sha256(sync_path)
    manifest["frame_capture"]["frame_count"] -= len(analysis.VIEW_IDS)
    manifest["frame_capture"]["sample_count"] -= 1
    for view_id in analysis.VIEW_IDS:
        manifest["frame_capture"]["frames_per_view"][view_id] -= 1
    for removed_row in removed_rows:
        del manifest["frame_files_sha256"][removed_row["rgb_filename"]]
        (capture / removed_row["rgb_filename"]).unlink()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(analysis.EvidenceError, match="frame plan"):
        analysis.validate_capture_bundle(capture, helper)


def test_two_run_comparison_requires_identical_physics_trace(
    tmp_path, helper
):
    first_capture, _ = _make_bundle(tmp_path, "first", helper, jitter_px=3)
    second_capture, second_manifest_path = _make_bundle(
        tmp_path,
        "second",
        helper,
        jitter_px=1,
        render_mode="rtx_history_512",
    )
    first_bundle = analysis.validate_capture_bundle(first_capture, helper)
    second_bundle = analysis.validate_capture_bundle(second_capture, helper)
    first_result = analysis.analyze_validated_capture(first_bundle)
    second_result = analysis.analyze_validated_capture(second_bundle)
    rows = analysis.compare_aligned_runs(
        first_bundle, first_result, second_bundle, second_result
    )
    assert rows
    assert all(
        set(row) >= {
            "global_step",
            "phase",
            "phase_step",
            "segment",
            "second_minus_first_pitch_fraction",
        }
        for row in rows
    )

    manifest = json.loads(
        second_manifest_path.read_text(encoding="utf-8")
    )
    summary_path = Path(manifest["physics_evidence"]["summary_path"])
    text = summary_path.read_text(encoding="utf-8").replace(
        "0.55", "0.56", 1
    )
    summary_path.write_text(text, encoding="utf-8")
    manifest["physics_evidence"]["summary_sha256"] = _sha256(summary_path)
    second_manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    changed_bundle = analysis.validate_capture_bundle(second_capture, helper)
    with pytest.raises(analysis.EvidenceError, match="physics state traces"):
        analysis.compare_aligned_runs(
            first_bundle, first_result, changed_bundle, second_result
        )


def test_run_analysis_explicitly_authorizes_valid_ab(tmp_path, helper):
    first_capture, _ = _make_bundle(tmp_path, "first", helper)
    second_capture, _ = _make_bundle(
        tmp_path,
        "second",
        helper,
        jitter_px=1,
        render_mode="rtx_history_512",
    )
    output = tmp_path / "output"
    report = analysis.run_analysis(
        first_capture,
        output,
        helper,
        compare=second_capture,
    )
    assert report["comparison_authorized"] is True
    assert report["comparison"]["authorized"] is True
    assert (output / "ab_per_tooth_summary.csv").is_file()


def test_failed_ab_does_not_leave_stale_authorized_csv(tmp_path, helper):
    first_capture, _ = _make_bundle(tmp_path, "first", helper)
    second_capture, _ = _make_bundle(tmp_path, "second", helper)
    output = tmp_path / "output"
    output.mkdir()
    stale = output / "ab_aligned_residuals.csv"
    stale.write_text("stale authorized result\n", encoding="utf-8")
    with pytest.raises(
        analysis.EvidenceError, match="treatments are identical"
    ):
        analysis.run_analysis(
            first_capture,
            output,
            helper,
            compare=second_capture,
        )
    assert not stale.exists()


def test_cli_reserves_optional_second_capture(tmp_path):
    arguments = analysis._arguments(
        [
            "--capture",
            str(tmp_path / "baseline"),
            "--compare",
            str(tmp_path / "history512"),
            "--output",
            str(tmp_path / "analysis"),
        ]
    )
    assert arguments.compare == tmp_path / "history512"
