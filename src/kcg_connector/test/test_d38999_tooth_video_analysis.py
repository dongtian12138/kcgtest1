"""Pure CPU contracts for strict D38999 GUI-video jitter analysis."""

from __future__ import annotations

import csv
import importlib
from pathlib import Path
import sys

import numpy as np
import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
try:
    analysis = importlib.import_module(
        "kcg_connector.d38999_tooth_video_analysis"
    )
finally:
    sys.path.remove(str(PACKAGE_ROOT))


def test_similarity_registration_removes_common_rigid_camera_motion():
    reference = np.asarray(
        [[0.0, 0.0], [2.0, 0.0], [0.0, 1.0], [2.0, 1.0]]
    )
    angle = 0.31
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    observed = 1.7 * reference @ rotation + np.asarray([12.0, -4.0])
    fit = analysis.fit_similarity_2d(reference, observed)
    assert fit["scale"] == pytest.approx(1.7)
    assert np.max(np.abs(fit["residual_vectors"])) < 1.0e-12


def test_relative_residual_exposes_one_tooth_after_whole_nut_registration():
    names = analysis.TRACK_SEGMENTS
    first = {
        name: (float(index * 20), float((index % 2) * 5))
        for index, name in enumerate(names)
    }
    second = {
        name: (point[0] + 3.0, point[1] - 2.0)
        for name, point in first.items()
    }
    second["Segment_00"] = (
        second["Segment_00"][0],
        second["Segment_00"][1] + 3.0,
    )
    rows = analysis.registered_relative_residual_rows(
        [
            {"frame_index": 10, "time_s": 1.0, "points": first},
            {"frame_index": 11, "time_s": 1.1, "points": second},
        ]
    )
    by_segment = {row["segment"]: row for row in rows}
    assert all(row["accepted_resolution_and_motion"] for row in rows)
    assert by_segment["Segment_00"]["residual_px"] > max(
        row["residual_px"]
        for name, row in by_segment.items()
        if name != "Segment_00"
    )


def test_resolution_and_scale_jump_are_fail_closed():
    names = analysis.TRACK_SEGMENTS
    tiny = {name: (float(index * 4), 0.0) for index, name in enumerate(names)}
    zoomed = {
        name: (float(index * 8), float(index % 2))
        for index, name in enumerate(names)
    }
    rows = analysis.registered_relative_residual_rows(
        [
            {"frame_index": 1, "time_s": 0.0, "points": tiny},
            {"frame_index": 2, "time_s": 0.1, "points": zoomed},
        ]
    )
    assert rows
    assert not any(row["accepted_resolution_and_motion"] for row in rows)
    assert all(
        "visible_tooth_pitch_below_resolution_gate"
        in row["measurement_refusal_reasons"]
        for row in rows
    )
    assert all(
        "adjacent_camera_or_scene_scale_jump"
        in row["measurement_refusal_reasons"]
        for row in rows
    )


def test_phase_ranges_preserve_csv_execution_order(tmp_path):
    path = tmp_path / "summary.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("global_step", "phase", "phase_step")
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "global_step": 1,
                    "phase": "hold_before_twist",
                    "phase_step": 1,
                },
                {
                    "global_step": 2,
                    "phase": "hold_before_twist",
                    "phase_step": 2,
                },
                {"global_step": 3, "phase": "twist", "phase_step": 1},
                {
                    "global_step": 4,
                    "phase": "hold_after_twist",
                    "phase_step": 1,
                },
            ]
        )
    ranges = analysis.ordered_phase_ranges(path)
    assert [item["phase"] for item in ranges] == [
        "hold_before_twist",
        "twist",
        "hold_after_twist",
    ]
    assert ranges[0]["steps"] == 2


def test_missing_hash_bound_sync_refuses_cross_mode_comparison(tmp_path):
    hashes = {
        "video_sha256": "a" * 64,
        "report_sha256": "b" * 64,
        "summary_sha256": "c" * 64,
    }
    status = analysis.explicit_sync_status(tmp_path, hashes)
    assert status == {
        "valid": False,
        "reasons": [
            "missing_video_capture_manifest.json",
            "missing_video_frame_sync.csv",
        ],
    }
    run = {
        "hashes": {"physics_trace_sha256": "d" * 64},
        "physics": {"phase_ranges": [{"phase": "hold", "steps": 10}]},
        "sync": status,
        "exploratory_visual": {"accepted_fraction": 1.0},
    }
    reasons = analysis.comparison_refusal_reasons(
        {mode: run for mode in analysis.RUN_MODES}
    )
    assert reasons == [
        "no_hash_bound_frame_to_global_step_sync:baseline,rtx_history512,"
        "fabric_disabled"
    ]


def test_different_physics_traces_block_render_only_attribution():
    run = {
        "hashes": {"physics_trace_sha256": "a" * 64},
        "physics": {"phase_ranges": [{"phase": "hold", "steps": 10}]},
        "sync": {"valid": True},
        "exploratory_visual": {"accepted_fraction": 1.0},
    }
    fabric = {
        **run,
        "hashes": {"physics_trace_sha256": "b" * 64},
    }
    reasons = analysis.comparison_refusal_reasons(
        {
            "baseline": run,
            "rtx_history512": run,
            "fabric_disabled": fabric,
        }
    )
    assert reasons == ["physics_state_traces_differ_across_render_modes"]


def test_minimum_recapture_contract_requires_rear_view_ids_and_frame_steps():
    contract = analysis.minimum_recapture_contract()
    assert "Segment_00" in contract["camera"]
    assert "semantic/instance IDs" in contract["capture"]
    assert contract["phases"] == [
        "nut_only_final_hold",
        "q7_twist_probe_hold",
    ]
    assert contract["sidecar"]["required_columns"] == list(
        analysis.SYNC_COLUMNS
    )


def test_cli_rejects_too_short_exploratory_window():
    with pytest.raises(SystemExit):
        analysis._parse_arguments(["--exploratory-seconds", "0.5"])
