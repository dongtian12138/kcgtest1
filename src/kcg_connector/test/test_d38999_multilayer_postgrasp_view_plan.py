from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from kcg_connector.d38999_multilayer_postgrasp_view_plan import (
    EXPECTED_DELTAS,
    FROZEN_SOURCES,
    build_multilayer_postgrasp_view_plan,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _plan():
    return build_multilayer_postgrasp_view_plan(REPOSITORY_ROOT)


def _copy_sources(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    for relative in FROZEN_SOURCES:
        source = REPOSITORY_ROOT / relative
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    return root


def test_current_plan_is_hash_bound_and_static_only():
    plan = _plan()
    assert plan["status"] == "STATIC_PASS"
    assert plan["dynamic_status"] == "PARKED"
    assert len(plan["sources"]) == 7
    assert all(len(row["sha256"]) == 64 for row in plan["sources"])


def test_exact_three_existing_candidates_are_precommitted():
    plan = _plan()
    assert plan["candidate_view_count"] == 3
    assert plan["candidate_sequence"] == ["V0", "V1", "V2"]
    assert [row["sequence_index"] for row in plan["candidate_views"]] == [0, 1, 2]
    assert [row["tcp_delta_xyz_rpy"] for row in plan["candidate_views"]] == [
        EXPECTED_DELTAS[view_id] for view_id in ("V0", "V1", "V2")
    ]


def test_required_pair_and_optional_third_are_not_relabelled():
    by_id = {row["view_id"]: row for row in _plan()["candidate_views"]}
    assert by_id["V0"]["required_by_legacy_contract"] is True
    assert by_id["V1"]["required_by_legacy_contract"] is True
    assert by_id["V2"]["required_by_legacy_contract"] is False
    assert by_id["V2"]["optional_third_view"] is True


def test_motion_and_candidate_limits_remain_exact():
    plan = _plan()
    assert plan["motion_limits"] == {
        "move_duration_s": 2.0,
        "settle_duration_s": 0.5,
        "per_command_max_joint_delta_rad": 0.05,
        "planned_max_joint_inf_rad": 0.05,
        "episode_max_joint_inf_rad": 0.20,
        "threshold_label": "SIM_TUNING_ONLY_CANDIDATE",
    }
    assert plan["candidate_limits"]["maximum_translation_m"] == 0.040
    assert plan["candidate_limits"]["maximum_rotation_rad"] == (
        0.20943951023931953
    )


def test_known_v1_v2_failures_are_preserved_fail_closed():
    by_id = {row["view_id"]: row for row in _plan()["candidate_views"]}
    assert by_id["V0"]["known_legacy_seed0_gate"] == "CAPTURE_PATH_ONLY"
    assert by_id["V1"]["known_legacy_seed0_gate"] == (
        "PLANNED_MAX_JOINT_INF_RAD_EXCEEDED"
    )
    assert by_id["V2"]["known_legacy_seed0_gate"] == "IK_FAILURE"
    assert all(
        row["current_multilayer_execution_proven"] is False
        and row["dynamic_execution_authorized"] is False
        for row in by_id.values()
    )


def test_camera_binding_does_not_promote_wrist_or_select_a_camera():
    binding = _plan()["camera_binding"]
    assert binding["current_wrist_role"] == "WRIST_LAYOUT_EVIDENCE_ONLY"
    assert binding["current_palm_role"] == "PALM_RGBD_SHADOW_CAPABLE"
    assert binding["selected_camera_for_dynamic_capture"] is None
    assert binding["observation_target"]["representation"] == (
        "D38999_VISUAL_COMPLETE_V1"
    )


def test_comoving_hand_camera_is_not_counted_as_independent_t_hp_view():
    semantics = _plan()["view_independence_semantics"]
    assert semantics == {
        "hand_camera_and_grasped_plug_comoving": True,
        "arm_motion_adds_independent_T_HP_view": False,
        "fixed_world_camera_can_add_independent_T_HP_view": True,
    }


def test_b4_and_dynamic_camera_evidence_remain_required():
    readiness = _plan()["dynamic_readiness"]
    assert readiness["current_multilayer_dynamic_views_proven"] == 0
    assert readiness["formal_capture_dependency"] == "B4_DYNAMIC_PASS"
    assert readiness["b4_dynamic_pass_evidence_present"] is False
    assert readiness["formal_capture_authorized"] is False
    assert readiness["dynamic_view_plan_pass_claimed"] is False


def test_truth_firewall_keeps_plan_precommitted():
    firewall = _plan()["truth_firewall"]
    assert {
        "object_truth",
        "contact_report",
        "collider_identity",
    } <= set(firewall["forbidden_fields"])
    assert firewall["runtime_image_changes_precommitted_motion"] is False
    assert firewall["object_pose_truth_changes_motion"] is False
    assert firewall["contact_name_or_normal_changes_motion"] is False
    assert firewall["postrun_object_pose_write_allowed"] is False


def test_interface_never_claims_motion_render_or_dynamic_pass():
    plan = _plan()
    assert plan["simulation_started"] is False
    assert plan["robot_motion_started"] is False
    assert plan["render_capture_performed"] is False
    assert plan["dynamic_visual_pass_claimed"] is False
    assert plan["formal_postgrasp_capture_pass_claimed"] is False
    assert plan["hardware_authorized"] is False
    json.dumps(plan, allow_nan=False)


def test_frozen_source_tamper_is_rejected(tmp_path):
    root = _copy_sources(tmp_path)
    config = root / "src/kcg_connector/config/d38999_postgrasp_shadow_v1.yaml"
    config.write_text(
        config.read_text().replace(
            "postgrasp_inhand_views: [V0, V1]",
            "postgrasp_inhand_views: [V0, V2]",
        )
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        build_multilayer_postgrasp_view_plan(root)


def test_public_builder_accepts_no_runtime_or_truth_inputs():
    names = set(inspect.signature(build_multilayer_postgrasp_view_plan).parameters)
    assert names == {"repository_root"}
    assert names.isdisjoint(
        {
            "object_pose",
            "contact_name",
            "contact_normal",
            "event_truth",
            "rgb",
            "depth",
            "arm_q",
            "b4_pass",
        }
    )
