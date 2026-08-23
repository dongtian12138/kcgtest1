"""Focused regressions for the two-object CARTS-Grasp V2 offline path."""

from pathlib import Path

import numpy as np

from kcg_connector.grasp.carts_v2.models import load_v2_config
from kcg_connector.grasp.carts_v2.pipeline import run_offline_pipeline


ROOT = Path(__file__).resolve().parents[4]
CONFIG = ROOT / "src/kcg_connector/config/carts_grasp_v2.yaml"
OBJECT_A = "current_d38999_26kj61sn_public_spec"
OBJECT_B = "te_deutsch_d38999_26fj35pn_step"


def test_dynamic_truth_and_motion_boundaries_remain_frozen() -> None:
    config = load_v2_config(CONFIG)
    assert config.values["hardware_authorized"] is False
    dynamic = config.section("dynamic")
    assert dynamic["lift_distance_m"] == 0.05
    assert dynamic["hold_duration_s"] >= 2.0
    assert dynamic["online_object_truth_allowed"] is False
    assert dynamic["online_contact_truth_allowed"] is False
    assert dynamic["object_pose_write_after_start_allowed"] is False


def test_transfer_scene_reuses_the_finite_table_author() -> None:
    config = load_v2_config(CONFIG)
    scene = config.section("dynamic")["object_scenes"][OBJECT_B]
    assert scene["scene_kind"] == "FREE_SINGLE_RIGID_ON_SHARED_FINITE_TABLE"
    runner = (
        ROOT / "src/kcg_connector/isaac/carts_v2/run_grasp_lift.py"
    ).read_text(encoding="utf-8")
    assert "author_d38999_tabletop_environment" in runner
    assert "add_default_ground_plane" not in runner


def test_same_pipeline_preserves_surface_contacts_and_cross_object_outcome() -> None:
    summaries = {}
    for object_id in (OBJECT_A, OBJECT_B):
        result = run_offline_pipeline(
            ROOT, config_path=CONFIG, object_id=object_id
        )
        assert len(result.candidates) == 48
        assert result.scenario_design.shape == (16, 26)
        assert all(
            np.unique(result.scenario_design[:, index]).size == 16
            for index in range(26)
        )
        assert len(result.selected_top) == 3
        assert len(result.exact_validation_results) == 3
        assert all(
            row.status == "UNRESOLVED_INTERFACE_MISMATCH"
            and not row.backend_invoked
            and row.requested_lift_distance_m == 0.05
            and row.backend_lift_distance_m == 0.04
            for row in result.exact_validation_results
        )
        survivors = [
            row
            for row in result.closure_predictions
            if row.status == "CLOSURE_SURVIVE"
        ]
        assert survivors
        for prediction in survivors:
            assert len(prediction.contacts) == 3
            assert all(
                result.inputs.face_roles.face_is_allowed[contact.object_face_index]
                for contact in prediction.contacts
            )
        fast_survivors = [
            row for row in result.fast_filter_results if row.status == "FAST_SURVIVE"
        ]
        assert len(fast_survivors) >= 3
        assert all(
            row.sampled_hand_table_clearance_m is not None
            and row.sampled_hand_table_clearance_m >= -1.0e-5
            for row in fast_survivors
        )
        assert any(
            "HAND_TABLE_PENETRATION" in row.reasons
            for row in result.fast_filter_results
        )
        assert all(not row.offline_task_gate_passed for row in result.selected_top)
        summaries[object_id] = result.selected_top[0]

    best_a = summaries[OBJECT_A]
    assert best_a.task_quality.worst_task_margin == 0.0
    assert "ZERO_IS_RANKING_LOWER_BOUND" in best_a.task_quality.failure_reason
    assert not best_a.offline_task_gate_passed
    best_b = summaries[OBJECT_B]
    assert 0.0 < best_b.task_quality.worst_task_margin < 1.0
    assert best_b.task_quality.required_peak_normal_force_n is None
    assert best_b.task_quality.maximum_joint_load_utilization is None
    assert best_b.task_quality.maximum_generalized_joint_torque_nm is None
    assert not best_b.offline_task_gate_passed
