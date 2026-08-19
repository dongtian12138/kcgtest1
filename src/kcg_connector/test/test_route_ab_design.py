from pathlib import Path

import numpy as np
import pytest

from kcg_connector.d38999_assembly_baseline import load_d38999_assembly_baseline
from kcg_connector.d38999_physical_insertion import (
    load_d38999_physical_insertion,
    solve_fixed_q7_tcp_pose,
)
from kcg_connector.d38999_tabletop_pick import (
    iiwa14_grasp_tcp_transform,
    load_d38999_tabletop_pick_config,
)
from kcg_connector.fixed_camera_t_hp_display_search import (
    search_fixed_camera_t_hp_display_candidates,
)
from kcg_connector.postgrasp_snapshot_gate import load_snapshot_gate_document
from kcg_connector.wrist_receptacle_view_design import (
    design_wrist_receptacle_views,
    frozen_wrist_camera_hand_transform,
    normalize_observability_jacobian,
    receptacle_world_pose,
)


def _inputs(repository):
    pick = load_d38999_tabletop_pick_config(
        repository / "src/kcg_connector/config/d38999_tabletop_pick_v1.yaml"
    )
    insertion = load_d38999_physical_insertion(
        repository / "src/kcg_connector/config/d38999_physical_insertion_v1.yaml"
    )
    assembly = load_d38999_assembly_baseline(
        repository / "src/kcg_connector/config/d38999_assembly_baseline_v1.yaml"
    )
    nominal_tcp = np.asarray(
        iiwa14_grasp_tcp_transform(tuple(pick.motion.grasp_arm_rad))
    )
    tcp_from_handbase = np.eye(4)
    tcp_from_handbase[2, 3] = -float(
        pick.geometry_candidate.handbase_to_tcp_m
    )
    nominal_plug = np.eye(4)
    nominal_plug[:3, 3] = np.asarray(
        pick.geometry_candidate.loose_settled_origin_m
    )
    nominal_hand_to_plug = (
        np.linalg.inv(nominal_tcp @ tcp_from_handbase) @ nominal_plug
    )
    joint_limits = [
        (-2.967, 2.967),
        (-2.094, 2.094),
        (-2.967, 2.967),
        (-2.094, 2.094),
        (-2.967, 2.967),
        (-2.0943951024, 2.0943951024),
        (-2.967, 2.967),
    ]
    return pick, insertion, assembly, tcp_from_handbase, nominal_hand_to_plug, joint_limits


def test_wrist_receptacle_views_are_true_different_views_and_screen_passes():
    repository = Path(__file__).resolve().parents[3]
    (
        _,
        insertion,
        assembly,
        tcp_from_handbase,
        nominal_hand_to_plug,
        joint_limits,
    ) = _inputs(repository)
    base_q = np.asarray(insertion.motion.preinsert_arm_rad)
    receptacle = receptacle_world_pose(
        assembly.datums.fixed.position_world_m,
        assembly.datums.fixed.axis_world,
    )
    import hashlib

    extrinsic_path = (
        repository
        / "src/kcg_connector/config/d38999_postgrasp_shadow_v1.yaml"
    )
    extrinsic_sha256 = hashlib.sha256(
        extrinsic_path.read_bytes()
    ).hexdigest()
    tcp_to_camera = frozen_wrist_camera_hand_transform(
        nominal_hand_to_plug
    )
    report = design_wrist_receptacle_views(
        base_arm_q=base_q,
        solve_arm=solve_fixed_q7_tcp_pose,
        tcp_from_handbase=tcp_from_handbase,
        nominal_hand_to_plug=nominal_hand_to_plug,
        tcp_to_camera=tcp_to_camera,
        receptacle_world=receptacle,
        joint_limits=joint_limits,
        extrinsic_source={
            "path": str(extrinsic_path),
            "sha256": extrinsic_sha256,
            "mount_eye_plug_m": [0.120, 0.0, 0.060],
            "mount_target_plug_m": [0.0, 0.0, 0.006],
        },
    )
    assert report["selected_view_ids"] == ["W_R0", "W_R1"]
    assert report["observability_screen"]["passed"] is True
    assert report["condition"]["observable_5d"] is True
    assert report["condition"]["condition_5d"] < 1.0e6
    assert len(report["c2"]["branches"]) == 2
    assert report["c2"]["cross_product_hypotheses"] == 0
    assert report["c2"]["averaged"] is False
    assert report["covariance"]["status"] == "UNVALIDATED"
    assert report["control_authorized"] is False
    assert report["formal_estimator_input"] is False


def test_wrist_camera_motion_does_not_change_plug_relative_view():
    from kcg_connector.postgrasp_shadow_view_planner import (
        plug_relative_camera_pose,
    )

    first_hand = np.eye(4)
    second_hand = np.eye(4)
    second_hand[:3, 3] = np.array([0.1, -0.05, 0.02])
    t_hc = np.eye(4)
    t_hp = np.eye(4)
    t_hp[:3, 3] = np.array([0.0, 0.0, 0.05])
    first = plug_relative_camera_pose(first_hand, t_hc, t_hp)
    second = plug_relative_camera_pose(second_hand, t_hc, t_hp)
    assert np.allclose(first, second)


def test_fixed_camera_b_route_reports_observability_rejected_when_no_80px_candidate():
    repository = Path(__file__).resolve().parents[3]
    snapshot_path = (
        repository
        / "artifacts/kcg_connector/d38999_postgrasp_visual_ft_e2e_v1"
        / "phase1_snapshot_gate_v3/seed000"
        / "postgrasp_snapshot_gate/snapshot_gate.json"
    )
    if not snapshot_path.is_file():
        pytest.skip("snapshot gate v3 artifact not present")
    (
        _,
        _,
        _,
        tcp_from_handbase,
        nominal_hand_to_plug,
        joint_limits,
    ) = _inputs(repository)
    snapshot = load_snapshot_gate_document(snapshot_path)
    q0 = np.asarray(snapshot["robot_state"]["q_rad"][:7])
    report = search_fixed_camera_t_hp_display_candidates(
        current_arm_q=q0,
        solve_arm=solve_fixed_q7_tcp_pose,
        tcp_from_handbase=tcp_from_handbase,
        nominal_hand_to_plug=nominal_hand_to_plug,
        joint_limits=joint_limits,
        max_candidates=10,
        max_wall_seconds=8.0,
    )
    assert report["status"] == "T_HP_OBSERVABILITY_REJECTED"
    assert report["visual_authorization"] is False
    assert report["control_authorized"] is False
    assert report["third_camera_added"] is False


def test_display_active_control_loop_does_not_read_truth_or_contact():
    source = (
        Path(__file__).resolve().parents[1]
        / "isaac/d38999_postgrasp_fresh_replay_smoke.py"
    ).read_text(encoding="utf-8")
    lines = source.splitlines()
    waypoint_line = next(
        index for index, line in enumerate(lines) if "for waypoint_index" in line
    )
    hold_line = next(
        index for index, line in enumerate(lines)
        if "for hold_index in range" in line
    )
    motion_block = "\n".join(lines[waypoint_line:hold_line])
    hold_block = "\n".join(
        lines[hold_line:hold_line + 220]
    )
    for block_name, block in (
        ("display_move_loop", motion_block),
        ("display_hold_loop", hold_block),
    ):
        assert "body.get_world_pose" not in block, block_name
        assert "nut.get_world_pose" not in block, block_name
        assert "get_full_contact_report" not in block, block_name
        assert "append_posthoc_sidecar_record(" not in block, block_name

    # Final audit exists and is explicitly after control termination.
    assert "posthoc_audit_after_control_termination" in source
    assert 'metrics["control_reads_object_truth"] = False' in source
    assert 'metrics["control_reads_contact_report"] = False' in source


def test_condition_normalization_is_invariant_to_mm_rescaling():
    rng = np.random.default_rng(4)
    jacobian = rng.normal(size=(240, 6))
    jacobian[:, :3] *= 1000.0
    jacobian_mm = jacobian.copy()
    jacobian_mm[:, :3] /= 1000.0
    first = normalize_observability_jacobian(
        jacobian, position_scale_m=0.001, angle_scale_rad=0.001
    )
    second = normalize_observability_jacobian(
        jacobian_mm, position_scale_m=1.0, angle_scale_rad=0.001
    )
    assert np.allclose(first, second)


def test_b_single_candidate_without_pair_fails_closed():
    from kcg_connector.fixed_camera_t_hp_display_search import (
        classify_b_search_status,
    )

    status, reason = classify_b_search_status(
        [{"candidate_index": 1}], None
    )
    assert status == "T_HP_OBSERVABILITY_REJECTED"
    assert "unpaired" in reason
    status, reason = classify_b_search_status(
        [], None
    )
    assert status == "T_HP_OBSERVABILITY_REJECTED"
