"""CPU-only fail-closed contracts for six-view axial tooth evidence."""

from __future__ import annotations

import copy

import pytest

from kcg_connector import d38999_tooth_axial_evidence as evidence


def _frame(step, phase="hold", views=()):
    return {
        "global_step": step,
        "phase": phase,
        "phase_step": step,
        "sample_index": step,
        "simulation_time_s": step / 240.0,
        "views": {view: {"points": {}} for view in views},
        "view_errors": {},
    }


def test_merge_requires_identical_ordered_sample_keys():
    base = [_frame(1, views=("rear_left",)), _frame(9, views=("rear_left",))]
    axial = [
        _frame(1, views=("axial_segment13",)),
        _frame(9, views=("axial_segment13",)),
    ]
    merged = evidence.merge_same_key_frames(base, axial)
    assert tuple(merged[0]["views"]) == (
        "rear_left",
        "axial_segment13",
    )
    bad = copy.deepcopy(axial)
    bad[1]["global_step"] = 10
    with pytest.raises(evidence.EvidenceError, match="frame keys differ"):
        evidence.merge_same_key_frames(base, bad)


def test_merge_rejects_view_id_overlap():
    frame = _frame(1, views=("rear_left",))
    with pytest.raises(evidence.EvidenceError, match="view IDs overlap"):
        evidence.merge_same_key_frames([frame], [copy.deepcopy(frame)])


def test_dynamic_coverage_keeps_union_separate_from_every_transition(
    monkeypatch,
):
    views = evidence.AXIAL_VIEW_IDS
    frames = [_frame(index, views=views) for index in (1, 9, 17)]

    def common(previous, current, view_id):
        del previous
        if current["global_step"] == 9:
            start = 0 if view_id == views[0] else 12
            return [
                f"Segment_{index:02d}"
                for index in range(start, start + 12)
            ]
        start = 12 if view_id == views[0] else 0
        return [
            f"Segment_{index:02d}" for index in range(start, start + 12)
        ]

    monkeypatch.setattr(
        evidence.analysis, "_common_transition_segments", common
    )
    monkeypatch.setattr(
        evidence.analysis,
        "_measure_transition",
        lambda previous, current, view, segments: [
            {
                "global_step": current["global_step"],
                "phase": current["phase"],
                "phase_step": current["phase_step"],
                "previous_global_step": previous["global_step"],
                "segment": segment,
                "view_id": view,
            }
            for segment in segments
        ],
    )
    _, coverage = evidence.collect_dynamic_view_coverage(frames, views)
    assert coverage["identity_union_all_24"] is True
    assert coverage["every_transition_all_24"] is True

    # Removing one half from the second transition still leaves a 24-ID
    # temporal union, but must never be upgraded to every-transition coverage.
    def partial(previous, current, view_id):
        result = common(previous, current, view_id)
        if current["global_step"] == 17 and view_id == views[1]:
            return []
        return result

    monkeypatch.setattr(
        evidence.analysis, "_common_transition_segments", partial
    )
    _, coverage = evidence.collect_dynamic_view_coverage(frames, views)
    assert coverage["identity_union_all_24"] is True
    assert coverage["every_transition_all_24"] is False
    assert coverage["minimum_segments_per_transition"] == 12


def test_axial_rig_rejects_target_substitution():
    manifest = {
        "camera_rig": {
            "axis_semantics": (
                "prepared CouplingNut local +Z equals world +Z"
            ),
            "same_completed_physics_step_as_base_four_views": True,
            "view_priority": list(evidence.AXIAL_VIEW_IDS),
            "views": [
                {
                    "view_id": "axial_segment13",
                    "eye_m": [0.44, 0.075, 0.47],
                    "segment_angle_degrees": 195,
                    "target_segment": "Segment_13",
                    "fixed_before_play": True,
                    "focal_length_mm": 50,
                    "resolution": [960, 720],
                    "target_m": [0.55, 0.185, 0.28],
                    "analytic_target_exposure": {
                        "axial_top_face_cosine": 0.78,
                        "radial_outer_face_cosine": 0.55,
                    },
                },
                {
                    "view_id": "axial_segment23",
                    "eye_m": [0.66, 0.075, 0.47],
                    "segment_angle_degrees": 345,
                    "target_segment": "Segment_23",
                    "fixed_before_play": True,
                    "focal_length_mm": 50,
                    "resolution": [960, 720],
                    "target_m": [0.55, 0.185, 0.28],
                    "analytic_target_exposure": {
                        "axial_top_face_cosine": 0.78,
                        "radial_outer_face_cosine": 0.55,
                    },
                },
            ],
        }
    }
    evidence._validate_axial_rig(manifest)  # noqa: SLF001
    manifest["camera_rig"]["views"][1]["target_segment"] = "Segment_22"
    with pytest.raises(evidence.EvidenceError, match="geometry differs"):
        evidence._validate_axial_rig(manifest)  # noqa: SLF001


def test_module_explicitly_forbids_union_and_no_jitter_overclaims():
    source = evidence.Path(evidence.__file__).read_text(encoding="utf-8")
    assert "identity_union_is_not_every_transition_coverage" in source
    assert '"render_jitter_absence_claim_authorized": False' in source
    assert '"wrapper_created_bundle_claim_authorized": False' in source
