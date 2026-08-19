from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pytest

from kcg_connector.d38999_cad_registration import (
    PLUG_MATING,
    fixed_camera_model,
    proxy_cad_points,
    render_points,
    transform_points,
)
from kcg_connector.d38999_tabletop_pick import iiwa14_grasp_tcp_transform
from kcg_connector.d38999_inhand_multiview import matrix_pose, pose_matrix
from kcg_connector.postgrasp_shadow_estimator import (
    FormalView,
    estimate_grouped_views,
)
from kcg_connector.postgrasp_snapshot_truth import (
    RestoreRejected,
    probe_restore_api,
    restore_snapshot,
)
import importlib.util

RUNTIME_PATH = (
    Path(__file__).parents[1]
    / "isaac"
    / "postgrasp_shadow_capture_runtime.py"
)
_spec = importlib.util.spec_from_file_location("postgrasp_shadow_capture_runtime", RUNTIME_PATH)
shadow_rt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(shadow_rt)
AB_SCRIPT = (
    Path(__file__).parents[1]
    / "isaac"
    / "d38999_postgrasp_visibility_ab.py"
)


def _authorized_t_hp_result():
    return {
        "capture_status": "GPU_CAPTURE_VALID",
        "pose_valid": True,
        "control_authorized": True,
        "covariance_calibration_status": "CALIBRATED",
        "c2_resolution": "C2_RESOLVED_BY_OBSERVATION",
        "real_keying_modeled": True,
        "keying_model_id": "d38999_26kj61sn_keyed_proxy_v2",
        "selected_c2_hypothesis_id": "YAW_0",
        "key_branch_selection": {
            "status": "SHADOW_BRANCH_SELECTED",
            "passed": True,
            "selected_for_shadow": "C2_LINKED_BRANCH_0",
            "shadow_selected_hypothesis_id": "YAW_0",
            "control_authorized": False,
        },
        "key_yaw_acceptance": {
            "status": "PASSED_EVALUATION_ONLY",
            "passed": True,
            "withheld_truth": True,
            "threshold_label": "REAL_MEASURED_CLEARANCE_REQUIRED",
            "observed_yaw_error_p95_deg": 0.4,
            "required_yaw_error_p95_deg": 0.5,
            "control_authorized": False,
        },
        "c2_hypotheses": [
            {
                "id": "YAW_0",
                "T_hand_plug_xyz_rpy": [
                    0.001,
                    -0.002,
                    0.448,
                    3.14,
                    0.01,
                    -0.02,
                ],
            },
            {
                "id": "OTHER_BRANCH",
                "T_hand_plug_xyz_rpy": [0.0] * 6,
            },
        ],
    }


def test_cv_optical_pose_is_converted_to_usd_without_losing_roll():
    cv_pose = pose_matrix((0.1, -0.2, 0.3, 0.4, -0.3, 0.2))
    usd_pose = shadow_rt._camera_cv_pose_to_usd(cv_pose)
    assert np.allclose(usd_pose[:3, 3], cv_pose[:3, 3])
    assert np.allclose(
        usd_pose[:3, :3] @ np.asarray((0.0, 0.0, -1.0)),
        cv_pose[:3, :3] @ np.asarray((0.0, 0.0, 1.0)),
    )
    assert np.allclose(
        usd_pose[:3, :3] @ np.asarray((0.0, 1.0, 0.0)),
        cv_pose[:3, :3] @ np.asarray((0.0, -1.0, 0.0)),
    )
    assert np.isclose(np.linalg.det(usd_pose[:3, :3]), 1.0)


def test_fixed_hand_camera_mounts_are_finite_right_handed_and_look_forward():
    for eye, target in (
        ((0.0, 0.0, 0.315), (0.0, 0.0, 0.448)),
        ((-0.150, 0.0, 0.060), (-0.090, 0.0, 0.480)),
    ):
        pose = shadow_rt._camera_cv_pose_from_eye_target(eye, target)
        expected_forward = np.asarray(target) - np.asarray(eye)
        expected_forward = expected_forward / np.linalg.norm(expected_forward)
        assert np.all(np.isfinite(pose))
        assert np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3))
        assert np.isclose(np.linalg.det(pose[:3, :3]), 1.0)
        assert np.allclose(pose[:3, 2], expected_forward)


def test_t_hp_control_gate_is_hard_disabled_during_shadow_only_stage():
    result = _authorized_t_hp_result()
    assert shadow_rt.KEYED_INSERTION_CONTROL_PROMOTION_ENABLED is False
    assert shadow_rt.selected_t_hp_control_pose(result) is None


def test_future_control_gate_requires_explicit_selected_keyed_pose(monkeypatch):
    monkeypatch.setattr(
        shadow_rt, "KEYED_INSERTION_CONTROL_PROMOTION_ENABLED", True
    )
    result = _authorized_t_hp_result()
    selected = shadow_rt.selected_t_hp_control_pose(result)
    assert np.allclose(
        selected,
        result["c2_hypotheses"][0]["T_hand_plug_xyz_rpy"],
    )


@pytest.mark.parametrize(
    "field,bad_value",
    (
        ("capture_status", "GPU_PASS"),
        ("pose_valid", False),
        ("control_authorized", False),
        ("covariance_calibration_status", "UNVALIDATED"),
        ("c2_resolution", "C2_UNRESOLVED"),
        ("real_keying_modeled", False),
        ("keying_model_id", None),
        ("selected_c2_hypothesis_id", None),
    ),
)
def test_t_hp_control_gate_rejects_failed_or_unvalidated_fields(
    field, bad_value, monkeypatch
):
    monkeypatch.setattr(
        shadow_rt, "KEYED_INSERTION_CONTROL_PROMOTION_ENABLED", True
    )
    result = _authorized_t_hp_result()
    result[field] = bad_value
    assert shadow_rt.selected_t_hp_control_pose(result) is None
    result.pop(field)
    assert shadow_rt.selected_t_hp_control_pose(result) is None


@pytest.mark.parametrize(
    "hypotheses",
    (
        [],
        [
            {
                "id": "WRONG_BRANCH",
                "T_hand_plug_xyz_rpy": [0.0] * 6,
            }
        ],
        [
            {
                "id": "KEYED_BRANCH",
                "T_hand_plug_xyz_rpy": [0.0] * 6,
            },
            {
                "id": "KEYED_BRANCH",
                "T_hand_plug_xyz_rpy": [0.0] * 6,
            },
        ],
        [
            {
                "id": "KEYED_BRANCH",
                "T_hand_plug_xyz_rpy": [0.0] * 5,
            }
        ],
        [
            {
                "id": "KEYED_BRANCH",
                "T_hand_plug_xyz_rpy": [0.0, 0.0, np.nan, 0.0, 0.0, 0.0],
            }
        ],
    ),
)
def test_t_hp_control_gate_rejects_ambiguous_or_invalid_hypotheses(
    hypotheses, monkeypatch
):
    monkeypatch.setattr(
        shadow_rt, "KEYED_INSERTION_CONTROL_PROMOTION_ENABLED", True
    )
    result = _authorized_t_hp_result()
    result["c2_hypotheses"] = hypotheses
    assert shadow_rt.selected_t_hp_control_pose(result) is None


def test_shadow_branch_field_cannot_substitute_for_control_selection(monkeypatch):
    monkeypatch.setattr(
        shadow_rt, "KEYED_INSERTION_CONTROL_PROMOTION_ENABLED", True
    )
    result = _authorized_t_hp_result()
    result["selected_c2_hypothesis_id"] = None

    assert result["key_branch_selection"]["shadow_selected_hypothesis_id"] == (
        "YAW_0"
    )
    assert shadow_rt.selected_t_hp_control_pose(result) is None


def test_current_proxy_block_is_explicit_and_never_selects_control():
    result = shadow_rt._current_proxy_key_branch_block()

    assert result["status"] == "KEYED_GEOMETRY_UNAVAILABLE"
    assert result["current_model_id"] == "d38999_shell25j_proxy_v1"
    assert result["selected_for_shadow"] is None
    assert result["shadow_selected_hypothesis_id"] is None
    assert result["control_authorized"] is False
    assert "selected_for_control" not in result


def test_view_motion_safety_gates_are_frozen():
    ok_positions = np.zeros(11)
    ok_velocities = np.zeros(11)
    base = {
        "positions": ok_positions,
        "velocities": ok_velocities,
        "arm_target": np.zeros(7),
        "arm_indices": np.arange(7),
        "measured_efforts": np.zeros(3),
        "tare_efforts": np.zeros(3),
        "wrist_canonical": np.zeros(6),
        "wrist_payload_reference": np.zeros(6),
        "wrist_ft_monitor_error": None,
        "tracking_limit_rad": 0.03,
        "torque_limit_nm": 2.0,
    }
    assert shadow_rt.evaluate_view_motion_safety(**base) is None
    bad = dict(base)
    bad["measured_efforts"] = np.array([3.0, 0.0, 0.0])
    assert shadow_rt.evaluate_view_motion_safety(**bad) == "FINGER_TORQUE_GATE"
    bad = dict(base)
    bad["wrist_canonical"] = np.array([9.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert shadow_rt.evaluate_view_motion_safety(**bad) == "WRIST_FORCE_GATE"
    bad = dict(base)
    bad["wrist_canonical"] = np.array([0.0, 0.0, 0.0, 0.4, 0.0, 0.0])
    assert shadow_rt.evaluate_view_motion_safety(**bad) == "WRIST_MOMENT_GATE"
    bad = dict(base)
    bad["wrist_ft_monitor_error"] = "stale"
    assert shadow_rt.evaluate_view_motion_safety(**bad) == "WRIST_FT_STALE_OR_FAILED"


def test_planned_view_budget_rejects_seed0_v1_without_relaxing_contract():
    motion_config = {
        "per_command_max_joint_delta_rad": 0.05,
        "planned_max_joint_inf_rad": 0.05,
        "episode_max_joint_inf_rad": 0.20,
    }
    # mountfix1 seed000: V1 solved target from actual H0 arm target.
    start = np.array(
        [-0.14715726785463334, 0.43583816474146025, -0.4169313299190262,
         -0.9489236605390955, 0.17593552427907197, 1.7868300702790016,
         -0.107043622479]
    )
    target = np.array(
        [0.4352739, 0.23900739, -0.98063337, -1.2327644, 0.15125828,
         1.58907002, -0.10704362]
    )
    with pytest.raises(shadow_rt.ShadowViewMotionAbort):
        shadow_rt.validate_planned_view_motion(
            start, target, start, motion_config, phase_label="V1"
        )
    # A genuinely small local branch remains admissible.
    small = start + np.array([0.01, -0.01, 0.01, 0.005, -0.005, 0.01, 0.0])
    diagnostic = shadow_rt.validate_planned_view_motion(
        start, small, start, motion_config, phase_label="LOCAL"
    )
    assert diagnostic["planned_max_abs_dq_rad"] <= 0.05
    assert diagnostic["planned_cumulative_max_abs_from_h0_rad"] <= 0.20


def test_v2_ik_failure_is_fail_closed_motion_abort():
    from kcg_connector.d38999_tabletop_pick import iiwa14_grasp_tcp_transform
    from kcg_connector.d38999_inhand_multiview import pose_matrix

    start = np.array(
        [-0.14715726785463334, 0.43583816474146025, -0.4169313299190262,
         -0.9489236605390955, 0.17593552427907197, 1.7868300702790016,
         -0.107043622479]
    )
    tcp = np.asarray(
        iiwa14_grasp_tcp_transform(tuple(float(v) for v in start))
    )
    desired = tcp @ pose_matrix(
        np.array([-0.012, 0.006, -0.030, np.radians(-4), np.radians(10), 0.0])
    )
    with pytest.raises(shadow_rt.ShadowViewMotionAbort):
        shadow_rt._solve_arm(
            start, desired[:3, 3], desired[:3, :3]
        )


def test_return_to_h0_success_and_exception_paths():
    success = {"attempted": False, "completed": False, "error": None}
    shadow_rt._attempt_return_to_h0(
        lambda target, phase: None, np.zeros(7), np.zeros(7), success
    )
    assert success == {"attempted": True, "completed": True, "error": None}
    failure = {"attempted": False, "completed": False, "error": None}

    def bad_move(target, phase):
        raise RuntimeError("sensor failed")

    shadow_rt._attempt_return_to_h0(
        bad_move, np.zeros(7), np.zeros(7), failure
    )
    assert failure["attempted"] is True
    assert failure["completed"] is False
    assert "sensor failed" in failure["error"]


def test_fixed_camera_visibility_ab_uses_posthoc_frames_only(tmp_path):
    from kcg_connector.postgrasp_shadow_view_planner import (
        run_fixed_camera_visibility_ab,
    )

    base = Path(__file__).parents[3] / (
        "artifacts/kcg_connector/d38999_postgrasp_visual_ft_e2e_v1"
        "/phase1_codex_shadow_smoke_mountfix1/seed000"
    )
    result = run_fixed_camera_visibility_ab(
        report_path=base / "nominal_physics_report.json",
        controller_steps_path=base / "controller_steps.jsonl",
        pick_config_path=Path(__file__).parents[1]
        / "config/d38999_tabletop_pick_v1.yaml",
        rgbd_config_path=Path(__file__).parents[1]
        / "config/d38999_rgbd_bootstrap_v1.yaml",
        output_path=tmp_path / "fixed_camera_ab.json",
    )
    assert result["truth_scope"] == "POSTHOC_TRUTH_ONLY"
    assert result["decision"] == "FIXED_CAMERA_PLUG_VISIBLE_OFFLINE"
    assert result["T_HP_observation_division"]["fixed_world_camera"] == (
        "independent_T_HP_view_source"
    )
    assert "do_not_count_as_multiview" in result[
        "T_HP_observation_division"
    ]["wrist_comoving_views"]


def test_partial_formal_archive_preserves_captured_views(tmp_path):
    from kcg_connector.d38999_cad_registration import fixed_camera_model
    from kcg_connector.postgrasp_shadow_estimator import FormalView

    camera = fixed_camera_model(
        eye=(0.64, -0.21, 0.31),
        target=(0.62, -0.21, 0.25),
        resolution=(16, 12),
    )
    rows = np.asarray(camera.world_to_camera, dtype=np.float64)
    t_wc = np.eye(4)
    t_wc[:3, :3] = rows.T
    t_wc[:3, 3] = camera.position_world
    view = FormalView(
        view_id="V0",
        timestamp_utc="2026-08-15T00:00:00Z",
        rgb=np.zeros((12, 16, 3), dtype=np.uint8),
        depth=np.full((12, 16), 0.3, dtype=np.float32),
        camera=camera,
        T_WH=np.eye(4),
        T_HC=np.eye(4),
        T_WC=t_wc,
    )
    result = shadow_rt._preserve_partial_formal_archive(
        tmp_path / "postgrasp_shadow", [view]
    )
    assert result.endswith("formal_views")
    assert (tmp_path / "postgrasp_shadow" / "formal_views" / "formal_manifest.json").is_file()
    status = json.loads(
        (tmp_path / "postgrasp_shadow" / "partial_shadow_archive_status.json").read_text()
    )
    assert status["status"] == "PARTIAL_VIEWS_PRESERVED"


def test_shadow_runtime_uses_stability_tracking_field_not_missing_lift():
    source = RUNTIME_PATH.read_text(encoding="utf-8")
    assert "physical_grasp.stability.maximum_arm_tracking_error_rad" in source
    assert "physical_grasp.lift" not in source


def test_fresh_wrist_getter_detects_overlimit_and_stale_second_step():
    steps = iter([1, 2, 2])
    canonical_values = [
        np.zeros(6),
        np.array([0.0, 0.0, 0.0, 0.4, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.4, 0.0, 0.0]),
    ]

    def getter():
        index = next(steps) - 1
        return {
            "global_step": index + 1,
            "canonical": canonical_values[index],
            "error": None,
        }

    step1, canonical1, error1 = shadow_rt._read_latest_wrist_state(getter, 0)
    assert error1 is None
    assert (
        shadow_rt.evaluate_view_motion_safety(
            positions=np.zeros(11),
            velocities=np.zeros(11),
            arm_target=np.zeros(7),
            arm_indices=np.arange(7),
            measured_efforts=np.zeros(3),
            tare_efforts=np.zeros(3),
            wrist_canonical=canonical1,
            wrist_payload_reference=np.zeros(6),
            wrist_ft_monitor_error=error1,
            tracking_limit_rad=0.03,
            torque_limit_nm=2.0,
        )
        is None
    )
    step2, canonical2, error2 = shadow_rt._read_latest_wrist_state(
        getter, step1
    )
    assert error2 is None
    assert (
        shadow_rt.evaluate_view_motion_safety(
            positions=np.zeros(11),
            velocities=np.zeros(11),
            arm_target=np.zeros(7),
            arm_indices=np.arange(7),
            measured_efforts=np.zeros(3),
            tare_efforts=np.zeros(3),
            wrist_canonical=canonical2,
            wrist_payload_reference=np.zeros(6),
            wrist_ft_monitor_error=error2,
            tracking_limit_rad=0.03,
            torque_limit_nm=2.0,
        )
        == "WRIST_MOMENT_GATE"
    )
    with pytest.raises(shadow_rt.ShadowViewMotionAbort):
        shadow_rt._read_latest_wrist_state(getter, step2)


def test_comoving_wrist_has_zero_plug_viewpoint_change_but_fixed_camera_does_not():
    from kcg_connector.postgrasp_shadow_view_planner import (
        fixed_world_camera_plug_pose,
        plug_relative_camera_pose,
    )

    t_hc = np.eye(4)
    t_hc[:3, 3] = [0.12, 0.0, 0.06]
    t_hp = np.eye(4)
    t_wh_a = np.eye(4)
    t_wh_a[:3, 3] = [0.6, -0.2, 0.3]
    t_wh_b = np.eye(4)
    t_wh_b[:3, 3] = [0.58, -0.21, 0.35]
    t_wh_b[:3, :3] = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.996, -0.087], [0.0, 0.087, 0.996]]
    )
    cp_a = plug_relative_camera_pose(t_wh_a, t_hc, t_hp)
    cp_b = plug_relative_camera_pose(t_wh_b, t_hc, t_hp)
    assert np.max(np.abs(cp_a - cp_b)) < 1.0e-12
    fixed_camera = np.eye(4)
    fixed_camera[:3, 3] = [0.55, -0.85, 0.72]
    cp_fixed_a = fixed_world_camera_plug_pose(fixed_camera, t_wh_a, t_hp)
    cp_fixed_b = fixed_world_camera_plug_pose(fixed_camera, t_wh_b, t_hp)
    assert np.max(np.abs(cp_fixed_a - cp_fixed_b)) > 1.0e-6


def test_t_hc_calibrated_composes_with_handbase_fk():
    nominal_hp = pose_matrix((0.0, 0.0, 0.44848, 0.0, np.pi, 0.0))
    camera_in_plug = shadow_rt._camera_model_from_eye_target(
        (0.12, 0.0, 0.06), (0.0, 0.0, 0.006), (16, 12)
    )
    t_hc = shadow_rt._calibrated_hand_camera_from_nominal_plug(
        nominal_hp,
        (0.12, 0.0, 0.06),
        (0.0, 0.0, 0.006),
        (16, 12),
    )
    assert np.max(
        np.abs(t_hc - nominal_hp @ shadow_rt._t_wc_from_camera_model(camera_in_plug))
    ) < 1.0e-12


def test_seed0_posthoc_visibility_refutes_wrong_mount_chain():
    """Posthoc-only regression fixture from phase0 seed000, never control input."""
    arm_q = np.asarray(
        (
            -0.14716605842113495,
            0.43761807680130005,
            -0.4167623519897461,
            -0.9498232007026672,
            0.17594848573207855,
            1.7868399620056152,
            -0.10703408718109131,
        )
    )
    nominal_hp = np.asarray(
        (
            (-0.9272416730212333, 0.37446345591072444, -4.25308457105829e-07, 2.0619240093111557e-08),
            (0.374463455910726, 0.927241673021327, 7.88773244080411e-08, -3.823982343106983e-09),
            (4.239004007695941e-07, -8.612413240836635e-08, -0.9999999999999063, 0.448480000000106),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    actual_hp_posthoc = np.asarray(
        (
            (-0.9055019020960267, 0.42434185316490897, -0.0005449339845074361, -0.00041324919115848235),
            (0.4243401771485323, 0.9054928641537954, -0.004252884210594307, 0.0001616728810817722),
            (-0.0013112429328129905, -0.004082232125607444, -0.9999908079691745, 0.4471821095382753),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    tcp_from_handbase = np.eye(4)
    tcp_from_handbase[2, 3] = -0.4
    t_wh = (
        np.asarray(iiwa14_grasp_tcp_transform(tuple(arm_q)))
        @ tcp_from_handbase
    )
    t_hc = shadow_rt._calibrated_hand_camera_from_nominal_plug(
        nominal_hp,
        (0.12, 0.0, 0.06),
        (0.0, 0.0, 0.006),
        (1280, 720),
    )
    t_wc = t_wh @ t_hc
    t_wp_actual = t_wh @ actual_hp_posthoc
    eye = t_wc[:3, 3]
    forward = t_wc[:3, :3] @ np.asarray((0.0, 0.0, 1.0))
    plug_direction = t_wp_actual[:3, 3] - eye
    view_angle_deg = np.degrees(
        np.arccos(
            np.clip(
                np.dot(forward, plug_direction)
                / (np.linalg.norm(forward) * np.linalg.norm(plug_direction)),
                -1.0,
                1.0,
            )
        )
    )
    assert eye[2] > 0.25
    assert view_angle_deg < 45.0

    camera = fixed_camera_model(
        eye=eye,
        target=shadow_rt._camera_target_from_t_wc(t_wc),
        resolution=(1280, 720),
    )
    plug_cad, _ = proxy_cad_points()
    observation = render_points(
        camera, (transform_points(plug_cad, matrix_pose(t_wp_actual)),)
    )
    assert int(np.sum(observation["label"] == PLUG_MATING)) >= 100
    h0, h1 = 216, 504
    w0, w1 = 384, 896
    central_depth = observation["depth"][h0:h1, w0:w1]
    finite_positive = np.isfinite(central_depth) & (central_depth > 0.0)
    assert float(np.mean(finite_positive)) >= 0.01


def test_h0_runtime_captures_fixed_world_view_without_arm_motion():
    source = RUNTIME_PATH.read_text(encoding="utf-8")
    assert "raw_rgbd_fixed_world_camera" in source
    assert "fixed_world_camera_views" in source
    assert "fixed_camera_config_T_WC" in source
    assert "capture_d38999_rgbd_runtime(" not in source
    assert "DEFAULT_POSTGRASP_PLANS[1:]" not in source
    assert "move_guarded(" not in source.split(
        "capture_view(0, DEFAULT_POSTGRASP_PLANS[0])", 1
    )[1].split("formal_root", 1)[0]


def test_shadow_runtime_passes_image_binding_to_raw_capture():
    source = RUNTIME_PATH.read_text(encoding="utf-8")
    assert "from PIL import Image" in source
    assert '"Image": Image,' in source
    assert 'capture_d38999_rgbd_raw_formal(' in source


def test_runtime_does_not_backderive_handeye_from_camera_world_pose():
    source = RUNTIME_PATH.read_text(encoding="utf-8")
    assert "ComputeLocalToWorldTransform" not in source
    assert "camera.get_world_pose" not in source
    assert "T_WC = T_WH @ T_HC_fixed_mount_candidate" in source
    assert "capture_d38999_rgbd_raw_formal" in source
    assert "capture_d38999_rgbd_runtime" not in source


def _camera_pose(camera):
    rows = np.asarray(camera.world_to_camera, dtype=np.float64)
    matrix = np.eye(4)
    matrix[:3, :3] = rows.T
    matrix[:3, 3] = np.asarray(camera.position_world, dtype=np.float64)
    return matrix


def _synthetic_views(state):
    plug_cad, receptacle_cad = proxy_cad_points()
    hp, rp = state[:6], state[6:]
    views = []
    for index, eye in enumerate(((0.55, -0.85, 0.72), (0.30, -0.85, 0.72))):
        camera = fixed_camera_model(eye=eye, target=(0.535, -0.0125, 0.231), resolution=(320, 180))
        T_WC = _camera_pose(camera)
        T_WH = np.eye(4)
        T_WP = T_WH @ pose_matrix(hp)
        T_WR = T_WP @ np.linalg.inv(pose_matrix(rp))
        observation = render_points(
            camera,
            (
                transform_points(plug_cad, matrix_pose(T_WP)),
                transform_points(receptacle_cad, matrix_pose(T_WR)),
            ),
        )
        views.append(
            FormalView(
                view_id=f"V{index}",
                timestamp_utc="2026-08-15T00:00:00Z",
                rgb=observation["rgb"],
                depth=observation["depth"],
                camera=camera,
                T_WH=T_WH,
                T_HC=np.linalg.inv(T_WH) @ T_WC,
                T_WC=T_WC,
            )
        )
    return views


def test_h0_shadow_estimates_t_hp_only_and_never_claims_t_rp():
    state = np.zeros(12)
    state[8] = -0.02
    views = _synthetic_views(state)
    result = estimate_grouped_views(
        postgrasp_inhand_views=views,
        final_preinsert_views=(),
        initial_state=state,
    )
    assert result["status"] == "REJECTED_T_HP_POSE_INVALID"
    assert result["success"] is False
    assert result["pose_valid"] is False
    assert "T_hand_plug_xyz_rpy" not in result
    assert len(result["c2"]["hypotheses"]) == 2
    assert result["T_receptacle_plug_status"] == "UNAVAILABLE_NO_RECEPTACLE_VIEWS"


def test_unsupported_restore_remains_fail_closed():
    probe = probe_restore_api({})
    assert probe.status == "API_UNSUPPORTED"
    snapshot = {
        "schema_version": "kcg_d38999_postgrasp_snapshot_v1",
        "role": "truth_restore",
        "scope": "snapshot_restore_and_posthoc_evaluation_only",
        "snapshot_id": "s",
        "timestamp_utc": "2026-08-15T00:00:00Z",
        "episode": "seed000000",
        "seed": 0,
        "global_step": 100,
        "plug_root_state": {
            "position_m": [0.0, 0.0, 0.0],
            "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
            "linear_velocity_m_s": [0.0, 0.0, 0.0],
            "angular_velocity_rad_s": [0.0, 0.0, 0.0],
        },
        "coupling_nut_joint_state": None,
        "robot_state": {"q_rad": [0.0] * 11, "qd_rad_s": [0.0] * 11},
        "frozen_command": {},
        "source_hashes": {},
    }
    with pytest.raises(RestoreRejected):
        restore_snapshot(snapshot, {})


def test_ab_tool_without_posthoc_frame_data_is_insufficient(tmp_path):
    output = tmp_path / "ab.json"
    env = dict(__import__("os").environ)
    env["PYTHONPATH"] = str(Path(__file__).parents[1])
    proc = subprocess.run(
        [sys.executable, str(AB_SCRIPT), "--output", str(output)],
        cwd=Path(__file__).parents[3],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    document = __import__("json").loads(output.read_text(encoding="utf-8"))
    assert document["decision"] == "INSUFFICIENT_POSTHOC_FRAME_DATA"
    assert document["truth_scope"] == "POSTHOC_TRUTH_ONLY"
