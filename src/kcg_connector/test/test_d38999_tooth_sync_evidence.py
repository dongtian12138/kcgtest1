"""Pure tests for the four-view tooth evidence aggregation contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kcg_connector import d38999_tooth_sync_analysis as analysis
from kcg_connector import d38999_tooth_sync_evidence as evidence


def _view(segment_indices, offset_x=0.0, jitter_segment=None):
    points = {}
    for local_index, segment_index in enumerate(segment_indices):
        point = np.asarray(
            [40.0 * local_index + offset_x, 20.0 * (local_index % 2)],
            dtype=float,
        )
        if segment_index == jitter_segment:
            point[1] += 3.0
        points[f"Segment_{segment_index:02d}"] = point
    return {
        "points": points,
        "pitches": {"synthetic_adjacent_pitch": 40.0},
    }


def _frames(*, missing_segment=None, offset=0.0, jitter_segment=None):
    frames = []
    view_segments = {
        "rear_left": tuple(range(0, 6)),
        "rear_right": tuple(range(6, 12)),
        "front_left": tuple(range(12, 18)),
        "front_right": tuple(range(18, 24)),
    }
    global_step = 0
    for phase in analysis.CAPTURE_PHASES:
        for phase_step in (1, 8):
            global_step += 8
            views = {}
            for view_id, indices in view_segments.items():
                if missing_segment is not None:
                    indices = tuple(
                        index for index in indices if index != missing_segment
                    )
                views[view_id] = _view(
                    indices,
                    offset_x=offset + phase_step,
                    jitter_segment=(
                        jitter_segment if phase_step == 8 else None
                    ),
                )
            frames.append(
                {
                    "global_step": global_step,
                    "phase": phase,
                    "phase_step": phase_step,
                    "views": views,
                }
            )
    return frames


def _physics_report():
    segment = {
        "maximum_local_translation_error_m": 0.0,
        "maximum_parent_relative_translation_error_m": 0.0,
        "maximum_local_rotation_error_rad": 0.0,
        "maximum_parent_relative_rotation_error_rad": 0.0,
    }
    return {
        "schema_version": analysis.PHYSICS_SCHEMA_VERSION,
        "steps": 5590,
        "anomaly_steps": 0,
        "thresholds": {
            "translation_m": 1.0e-6,
            "rotation_rad": 1.0e-5,
        },
        "segment_aggregate": {
            name: dict(segment) for name in evidence.EXPECTED_SEGMENTS
        },
    }


def _treatment_bundle(run_id):
    baseline_normalization = {
        "authored_in_session_layer": False,
        "changed_missing_rotate_z": False,
    }
    render = {
        "mode": "baseline",
        "requested": {},
        "actual": {},
        "exact_match": True,
        "validated_after_simulation_app_start": True,
        "mismatches": [],
    }
    report = {
        "normalization_ab": baseline_normalization,
        "segment00_schema": {
            "schema_outlier": True,
            "explicit_rotate_z": False,
            "explicit_rotate_z_degrees": None,
        },
    }
    if run_id == "rtx_history_512":
        render["mode"] = "rtx_history_512"
        render["requested"] = {
            "/rtx/scenedb/maxHistoryTransformCount": 512
        }
    elif run_id == "segment00_normalized":
        report["normalization_ab"] = {
            "authored_in_session_layer": True,
            "changed_missing_rotate_z": True,
            "rotate_z_degrees": 0,
        }
        report["segment00_schema"] = {
            "schema_outlier": False,
            "explicit_rotate_z": True,
            "explicit_rotate_z_degrees": 0,
        }
    return {"manifest": {"render_settings": render}, "physics_report": report}


def test_all_view_union_requires_24_at_every_transition():
    rows, coverage = evidence.collect_all_view_run_evidence(
        "baseline", {"frames": _frames(jitter_segment=7)}
    )
    assert rows
    assert coverage["identity_union_all_24"] is True
    assert coverage["every_transition_all_24"] is True
    assert coverage["minimum_segments_per_transition"] == 24
    assert all(
        item["identity_union_all_24"] is True
        for item in coverage["per_phase"].values()
    )
    segment_07 = [
        row["residual_pitch_fraction"]
        for row in rows
        if row["segment"] == "Segment_07"
    ]
    assert max(segment_07) > 0.0


def test_identity_union_is_not_enough_when_one_tooth_is_never_measurable():
    _, coverage = evidence.collect_all_view_run_evidence(
        "baseline", {"frames": _frames(missing_segment=23)}
    )
    assert coverage["identity_union_all_24"] is False
    assert coverage["every_transition_all_24"] is False
    assert coverage["minimum_segments_per_transition"] == 23
    assert coverage["missing_from_identity_union"] == ["Segment_23"]


def test_ab_coverage_intersects_both_runs_and_both_transition_frames():
    first = {"frames": _frames()}
    second = {"frames": _frames(offset=1.0, jitter_segment=13)}
    rows, coverage = evidence.collect_all_view_ab_evidence(
        "baseline_vs_rtx_history_512", first, second
    )
    assert rows
    assert coverage["every_transition_all_24"] is True
    assert {
        row["comparison_id"] for row in rows
    } == {"baseline_vs_rtx_history_512"}


def test_physics_gate_requires_all_24_and_zero_anomaly_steps():
    result = evidence.validate_physics_report(_physics_report())
    assert result["all_24_segments_tracked"] is True
    assert result["relative_motion_below_diagnostic_threshold"] is True
    bad = _physics_report()
    bad["anomaly_steps"] = 1
    with pytest.raises(evidence.EvidenceError, match="anomaly steps"):
        evidence.validate_physics_report(bad)
    bad = _physics_report()
    del bad["segment_aggregate"]["Segment_23"]
    with pytest.raises(evidence.EvidenceError, match="exactly 24"):
        evidence.validate_physics_report(bad)


@pytest.mark.parametrize("run_id", evidence.RUN_IDS)
def test_all_three_treatments_have_distinct_exact_contracts(run_id):
    result = evidence.validate_treatment(run_id, _treatment_bundle(run_id))
    assert result["render_settings_exactly_read_back"] is True


def test_normalized_treatment_cannot_silently_reuse_baseline():
    with pytest.raises(evidence.EvidenceError, match="normalization treatment"):
        evidence.validate_treatment(
            "segment00_normalized", _treatment_bundle("baseline")
        )


def test_file_binding_is_repo_relative_and_rejects_escape(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    inside = repository / "evidence.json"
    inside.write_text("{}\n", encoding="utf-8")
    binding = evidence.file_binding(inside, repository)
    assert binding["path"] == "evidence.json"
    assert binding["size_bytes"] == 3
    outside = tmp_path / "outside.json"
    outside.write_text("{}\n", encoding="utf-8")
    with pytest.raises(evidence.EvidenceError, match="escapes repository"):
        evidence.file_binding(outside, repository)


def test_contract_never_authorizes_no_jitter_from_parser_success():
    source = Path(evidence.__file__).read_text(encoding="utf-8")
    assert '"render_jitter_absence_claim_authorized": False' in source
    assert "no_visual_residual_acceptance_threshold_was_preregistered" in source
