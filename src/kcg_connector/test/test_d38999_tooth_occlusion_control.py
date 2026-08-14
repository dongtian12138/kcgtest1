"""Pure CPU tests for the render-only tooth occlusion-control contract."""

from __future__ import annotations

from copy import deepcopy

import pytest

from kcg_connector import d38999_tooth_occlusion_control as control


def _physics_report():
    segment = {
        "contact_counterparts": ["/World/HandArm/finger"],
        "contact_records": 10,
        "maximum_contact_impulse_norm": 0.02,
        "maximum_local_rotation_error_rad": 0.0,
        "maximum_local_translation_error_m": 0.0,
        "maximum_parent_relative_rotation_error_rad": 0.0,
        "maximum_parent_relative_translation_error_m": 0.0,
        "minimum_contact_separation_m": -1.0e-5,
    }
    return {
        "schema_version": control.PHYSICS_SCHEMA_VERSION,
        "anomaly_steps": 0,
        "phase_steps": {
            "nut_only_final_hold": 960,
            "q7_twist_probe_hold": 180,
            "q7_twist_probe_motion": 960,
        },
        "segment_aggregate": {
            name: deepcopy(segment) for name in control.EXPECTED_SEGMENTS
        },
        "steps": 5590,
        "thresholds": {"rotation_rad": 1.0e-5, "translation_m": 1.0e-6},
    }


def _capture_manifest():
    source_hash = "a" * 64
    return {
        "schema_version": control.CAPTURE_SCHEMA_VERSION,
        "passed": True,
        "camera_rig": {
            "same_completed_physics_step": True,
            "view_priority": [
                "rear_left",
                "rear_right",
                "front_left",
                "front_right",
            ],
            "views": [{"view_id": name} for name in (
                "rear_left",
                "rear_right",
                "front_left",
                "front_right",
            )],
        },
        "capture_source": {
            "path": "/repo/d38999_tooth_sync_capture.py",
            "sha256_at_import": source_hash,
            "sha256_at_start": source_hash,
            "sha256_at_finalize": source_hash,
            "unchanged_during_capture": True,
        },
        "cleanup": {"object_pose_writes": 0},
        "frame_capture": {"sample_count": 265, "frame_count": 1060},
        "render_settings": {"mode": "baseline"},
        "sampling": {"physics_rate_hz": 240, "capture_rate_hz": 30},
        "step_mapping_semantics": "same completed physics step",
        "sync_columns": ["frame_index", "global_step"],
    }


def _runtime_sidecar():
    prefix = (
        "/World/HandArm/Geometry/world/iiwa_link_0/iiwa_link_1/"
        "iiwa_link_2/iiwa_link_3/iiwa_link_4/iiwa_link_5/iiwa_link_6/"
        "iiwa_link_7/iiwa_link_ee/handbase_link"
    )
    return {
        "schema_version": control.RUNTIME_SCHEMA_VERSION,
        "passed": True,
        "authoring": {
            "edit_layer": "anonymous_session_layer",
            "robot_root": "/World/HandArm",
            "prim_paths": [
                prefix + "/f1Link1",
                prefix + "/f2Link1",
                prefix + "/f3Link1",
            ],
            "properties": ["visibility"],
            "visibility_token": "invisible",
            "authored_before_timeline_play": True,
            "authored_before_world_reset": True,
            "computed_descendants_invisible": True,
        },
        "mutation_audit": {
            "collision_api_writes": 0,
            "material_writes": 0,
            "object_pose_writes": 0,
            "physics_api_writes": 0,
            "xform_writes": 0,
        },
        "cleanup": {
            "session_visibility_opinions_removed": True,
            "effective_visibility_restored_to_pre_author_state": True,
        },
        "source": {
            "runner_sha256_at_finalize": "e" * 64,
            "runner_sha256_at_start": "e" * 64,
            "runner_unchanged_during_run": True,
            "sha256_at_finalize": "f" * 64,
            "sha256_at_import": "f" * 64,
            "sha256_at_start": "f" * 64,
            "unchanged_during_run": True,
        },
    }


def _coverage():
    counts = {name: 10 for name in control.EXPECTED_SEGMENTS}
    phase = {
        "identity_union_all_24": True,
        "missing_from_identity_union": [],
    }
    return {
        "identity_union_all_24": True,
        "every_transition_all_24": True,
        "segments_in_identity_union": list(control.EXPECTED_SEGMENTS),
        "missing_from_identity_union": [],
        "per_segment_transition_counts": counts,
        "per_phase": {
            "nut_only_final_hold": dict(phase),
            "q7_twist_probe_motion": dict(phase),
            "q7_twist_probe_hold": dict(phase),
        },
    }


def _evaluate(**changes):
    arguments = {
        "baseline_capture_manifest": _capture_manifest(),
        "ghost_capture_manifest": _capture_manifest(),
        "baseline_physics_report": _physics_report(),
        "ghost_physics_report": _physics_report(),
        "baseline_physics_trace_sha256": "b" * 64,
        "ghost_physics_trace_sha256": "b" * 64,
        "baseline_contact_trace_sha256": "c" * 64,
        "ghost_contact_trace_sha256": "c" * 64,
        "ghost_runtime_sidecar": _runtime_sidecar(),
        "visual_coverage": _coverage(),
    }
    arguments.update(changes)
    return control.evaluate_occlusion_control(**arguments)


def test_valid_ghost_run_requires_exact_physics_contact_and_24_id_coverage():
    result = _evaluate()
    assert result["passed"] is True
    assert result["physics"]["state_trace_identical"] is True
    assert result["physics"]["contact_trace_identical"] is True
    assert result["physics"]["contact_dynamics_identical"] is True
    assert result["coverage"]["every_sampled_transition_all_24"] is True
    assert result["claim_boundaries"]["safety_thresholds_changed"] is False


def test_contact_fingerprint_catches_render_run_that_changed_finger_contact():
    ghost = _physics_report()
    ghost["segment_aggregate"]["Segment_21"]["contact_records"] += 1
    with pytest.raises(
        control.OcclusionControlError, match="contact dynamics"
    ):
        _evaluate(ghost_physics_report=ghost)


def test_per_step_contact_trace_must_match_even_when_aggregates_match():
    with pytest.raises(
        control.OcclusionControlError, match="per-step contact trace"
    ):
        _evaluate(ghost_contact_trace_sha256="d" * 64)


def test_union_only_is_not_enough_for_a_rotating_hidden_tooth():
    coverage = _coverage()
    coverage["every_transition_all_24"] = False
    with pytest.raises(
        control.OcclusionControlError, match="every transition"
    ):
        _evaluate(visual_coverage=coverage)


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("object_pose_writes", "object_pose_writes"),
        ("physics_api_writes", "physics_api_writes"),
        ("xform_writes", "xform_writes"),
    ),
)
def test_runtime_ghost_rejects_any_pose_or_physics_mutation(field, message):
    sidecar = _runtime_sidecar()
    sidecar["mutation_audit"][field] = 1
    with pytest.raises(control.OcclusionControlError, match=message):
        _evaluate(ghost_runtime_sidecar=sidecar)


def test_capture_must_reuse_exact_four_view_schedule_and_no_pose_writes():
    ghost = _capture_manifest()
    ghost["camera_rig"]["view_priority"][-1] = "top"
    with pytest.raises(control.OcclusionControlError, match="camera_rig"):
        _evaluate(ghost_capture_manifest=ghost)
    ghost = _capture_manifest()
    ghost["cleanup"]["object_pose_writes"] = 1
    with pytest.raises(control.OcclusionControlError, match="object pose"):
        _evaluate(ghost_capture_manifest=ghost)
