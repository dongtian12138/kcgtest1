from pathlib import Path

import numpy as np
import pytest

from kcg_connector.postgrasp_snapshot_gate import (
    SnapshotContractError,
    build_replay_bundle_manifest,
    capture_snapshot_gate_document,
    load_snapshot_gate_document,
    sha256_file,
    write_replay_bundle_manifest,
    write_snapshot_gate_document,
)


def _runtime_module():
    import importlib.util

    path = (
        Path(__file__).resolve().parents[1]
        / "isaac/postgrasp_fresh_replay_runtime.py"
    )
    spec = importlib.util.spec_from_file_location("fresh_rt", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _snapshot_document(*, include_tare=True):
    document = capture_snapshot_gate_document(
        snapshot_id="mock_snap",
        timestamp_utc="2026-08-15T00:00:00Z",
        episode="seed000000",
        seed=0,
        global_step=1000,
        physics_step=1000,
        robot_getters={
            "q": lambda: np.zeros(15),
            "qd": lambda: np.zeros(15),
        },
        finger_getters={
            "hand_q_actual": lambda: np.zeros(4),
            "hand_q_target": lambda: np.zeros(4),
            "finger_root_torque_proxy": lambda: np.array([0.2, 0.3, 0.2]),
            "finger_root_tare_efforts": lambda: np.array([0.1, 0.2, 0.1]),
        },
        plug_root_getters={
            "position": lambda: np.array([0.5, -0.2, 0.3]),
            "orientation": lambda: np.array([1.0, 0.0, 0.0, 0.0]),
            "linear_velocity": lambda: np.zeros(3),
            "angular_velocity": lambda: np.zeros(3),
        },
        nut_joint_getters=None,
        nut_posthoc_getters={
            "position": lambda: np.array([0.5, -0.2, 0.33]),
            "orientation": lambda: np.array([1.0, 0.0, 0.0, 0.0]),
            "linear_velocity": lambda: np.zeros(3),
            "angular_velocity": lambda: np.zeros(3),
        },
        frozen_command={
            "arm_q_target_rad": np.zeros(7).tolist(),
            "hand_q_target_rad": np.zeros(4).tolist(),
            "wrist_ft_payload_reference": np.zeros(6).tolist(),
            "wrist_ft_snapshot_reference": np.zeros(6).tolist(),
            "wrist_ft_snapshot_global_step": 1000,
        },
        provenance={"runner_sha256": "a" * 64},
    )
    if not include_tare:
        del document["finger_state"]["finger_root_tare_efforts_nm"]
    return document


def _write_bundle(tmp_path, *, include_tare=True):
    gate_dir = tmp_path / "postgrasp_snapshot_gate"
    gate_dir.mkdir(parents=True, exist_ok=True)
    snapshot = _snapshot_document(include_tare=include_tare)
    write_snapshot_gate_document(gate_dir / "snapshot_gate.json", snapshot)
    (gate_dir / "replay_stage.usda").write_text(
        "#usda 1.0\n", encoding="utf-8"
    )
    manifest = build_replay_bundle_manifest(
        seed=0,
        snapshot_file="snapshot_gate.json",
        snapshot_sha256=sha256_file(gate_dir / "snapshot_gate.json"),
        stage_file="replay_stage.usda",
        stage_sha256=sha256_file(gate_dir / "replay_stage.usda"),
        source_hashes={"runner_sha256": "a" * 64},
    )
    write_replay_bundle_manifest(
        gate_dir / "replay_bundle_manifest.json", manifest
    )
    return gate_dir


class _Counter:
    def __init__(self):
        self.value = 0

    def increment(self):
        self.value += 1


def _bindings(snapshot, *, extra_write_on_step=None):
    counter = _Counter()
    positions = {
        "plug": np.asarray(snapshot["plug_root_state"]["position_m"]),
        "nut": np.asarray(
            snapshot["nut_rigid_state_restore_only"]["position_m"]
        ),
    }
    orientations = {
        "plug": np.asarray(snapshot["plug_root_state"]["orientation_wxyz"]),
        "nut": np.asarray(
            snapshot["nut_rigid_state_restore_only"]["orientation_wxyz"]
        ),
    }
    state = {"step": -1}

    def restore_rigid_body(name, rigid_state):
        positions[name] = np.asarray(
            rigid_state["position_m"], dtype=np.float64
        )
        orientations[name] = np.asarray(
            rigid_state["orientation_wxyz"], dtype=np.float64
        )
        counter.increment()

    def observe_and_step(_arm, _hand):
        state["step"] += 1
        if extra_write_on_step is not None:
            extra_write_on_step(state["step"], counter, positions)
        return (
            np.asarray(snapshot["robot_state"]["q_rad"]),
            np.asarray(snapshot["robot_state"]["qd_rad_s"]),
        )

    return counter, positions, orientations, restore_rigid_body, observe_and_step, state


def _run(tmp_path, snapshot, **overrides):
    rt = _runtime_module()
    extra_write_on_step = overrides.pop("extra_write_on_step", None)
    (
        counter,
        positions,
        orientations,
        restore_rigid_body,
        observe_and_step,
        state,
    ) = _bindings(snapshot, extra_write_on_step=extra_write_on_step)
    stage_path = tmp_path / "postgrasp_snapshot_gate" / "replay_stage.usda"
    arguments = {
        "snapshot": snapshot,
        "stage_path": stage_path,
        "open_stage": lambda path: True,
        "reset_world": lambda: None,
        "robot_set_q": lambda value: None,
        "robot_set_qd": lambda value: None,
        "restore_rigid_body": restore_rigid_body,
        "observe_and_step": observe_and_step,
        "sample_finger_torque_proxy": lambda: np.asarray(
            snapshot["finger_state"]["finger_root_torque_proxy_nm"]
        ),
        "get_latest_wrist_state": lambda: {
            "global_step": state["step"],
            "canonical": np.asarray(
                snapshot["frozen_command"]["wrist_ft_snapshot_reference"]
            ),
            "error": None,
        },
        "restored_plug_position": positions["plug"],
        "restored_plug_orientation": orientations["plug"],
        "restored_nut_position": positions["nut"],
        "restored_nut_orientation": orientations["nut"],
        "object_write_counter": counter,
    }
    arguments.update(overrides)
    return rt.run_fresh_replay_restore_settle(**arguments), counter, state


def test_bundle_loader_roundtrip_with_v3_tare(tmp_path):
    rt = _runtime_module()
    gate_dir = _write_bundle(tmp_path)
    loaded = rt.load_fresh_replay_bundle(
        gate_dir / "replay_bundle_manifest.json", expected_seed=0
    )
    assert loaded["manifest"]["seed"] == 0
    assert loaded["snapshot"]["seed"] == 0
    assert loaded["stage_path"].name == "replay_stage.usda"
    assert loaded["snapshot"]["finger_state"][
        "finger_root_tare_efforts_nm"
    ] == [0.1, 0.2, 0.1]


def test_bundle_loader_rejects_snapshot_without_tare(tmp_path):
    rt = _runtime_module()
    gate_dir = _write_bundle(tmp_path, include_tare=False)
    with pytest.raises(SnapshotContractError):
        rt.load_fresh_replay_bundle(
            gate_dir / "replay_bundle_manifest.json"
        )


def test_bundle_loader_rejects_seed_mismatch(tmp_path):
    rt = _runtime_module()
    gate_dir = _write_bundle(tmp_path)
    with pytest.raises(SnapshotContractError):
        rt.load_fresh_replay_bundle(
            gate_dir / "replay_bundle_manifest.json", expected_seed=7
        )


def test_restore_settle_nominal_verified(tmp_path):
    snapshot = _snapshot_document()
    gate_dir = _write_bundle(tmp_path)
    write_snapshot_gate_document(gate_dir / "snapshot_gate.json", snapshot)
    # Rebuild the manifest so the mutable in-memory snapshot above matches
    # the on-disk bytes used by the nominal restore test.
    manifest = build_replay_bundle_manifest(
        seed=0,
        snapshot_file="snapshot_gate.json",
        snapshot_sha256=sha256_file(gate_dir / "snapshot_gate.json"),
        stage_file="replay_stage.usda",
        stage_sha256=sha256_file(gate_dir / "replay_stage.usda"),
        source_hashes={"runner_sha256": "a" * 64},
    )
    write_replay_bundle_manifest(
        gate_dir / "replay_bundle_manifest.json", manifest
    )
    result, counter, _ = _run(tmp_path, snapshot)
    assert result["status"] == "FRESH_REPLAY_RESTORE_VERIFIED"
    assert result["restore_boundary_object_pose_writes"] == 2
    assert result["object_pose_writes_after_restore"] == 0
    assert result["control_authorized"] is False
    assert result["formal_estimator_input"] is False
    assert counter.value == 2


def test_restore_rejects_extra_object_write_during_settle(tmp_path):
    snapshot = _snapshot_document()
    _write_bundle(tmp_path)

    def extra_write(step, counter, positions):
        if step == 3:
            positions["plug"] = positions["plug"].copy()
            counter.increment()

    result, counter, _ = _run(
        tmp_path, snapshot, extra_write_on_step=extra_write
    )
    assert result["status"] == "FRESH_REPLAY_ABORT_SAFE"
    assert "object pose writes" in result["error"]
    assert counter.value == 3


def test_restore_rejects_missing_stage(tmp_path):
    snapshot = _snapshot_document()
    _write_bundle(tmp_path)
    (tmp_path / "postgrasp_snapshot_gate" / "replay_stage.usda").unlink()
    result, _, _ = _run(tmp_path, snapshot)
    assert result["status"] == "FRESH_REPLAY_ABORT_SAFE"
    assert "stage missing" in result["error"]


def test_restore_rejects_nonfinite_torque(tmp_path):
    snapshot = _snapshot_document()
    _write_bundle(tmp_path)
    result, _, _ = _run(
        tmp_path,
        snapshot,
        sample_finger_torque_proxy=lambda: np.array([np.nan, 0.0, 0.0]),
    )
    assert result["status"] == "FRESH_REPLAY_ABORT_SAFE"
    assert "finite 3-channel" in result["error"]


def test_legacy_v2_artifact_is_rejected_as_fresh_replay_bundle():
    base = (
        Path(__file__).resolve().parents[3]
        / "artifacts/kcg_connector/d38999_postgrasp_visual_ft_e2e_v1"
        / "phase1_snapshot_gate_v2/seed000"
    )
    snapshot_path = (
        base / "postgrasp_snapshot_gate/snapshot_gate.json"
    )
    if not snapshot_path.is_file():
        pytest.skip("snapshot gate v2 artifact not present")
    legacy = load_snapshot_gate_document(snapshot_path)
    assert (
        "finger_root_tare_efforts_nm"
        not in legacy["finger_state"]
    )
    with pytest.raises(SnapshotContractError):
        _runtime_module().load_fresh_replay_bundle(
            snapshot_path
        )


def test_two_fixed_camera_display_poses_are_distinct_and_bounded():
    base = (
        Path(__file__).resolve().parents[3]
        / "artifacts/kcg_connector/d38999_postgrasp_visual_ft_e2e_v1"
        / "phase1_snapshot_gate_v3/seed000"
    )
    snapshot_path = (
        base / "postgrasp_snapshot_gate/snapshot_gate.json"
    )
    if not snapshot_path.is_file():
        pytest.skip("snapshot gate v3 artifact not present")
    from kcg_connector.d38999_physical_insertion import solve_fixed_q7_tcp_pose
    from kcg_connector.d38999_tabletop_pick import (
        iiwa14_grasp_tcp_transform,
        load_d38999_tabletop_pick_config,
    )
    from kcg_connector.postgrasp_shadow_view_planner import (
        plan_two_fixed_camera_display_feasibility_poses,
    )

    repository = Path(__file__).resolve().parents[3]
    pick = load_d38999_tabletop_pick_config(
        repository / "src/kcg_connector/config/d38999_tabletop_pick_v1.yaml"
    )
    snapshot = load_snapshot_gate_document(snapshot_path)
    q0 = np.asarray(snapshot["robot_state"]["q_rad"][:7])
    nominal_world_tcp = np.asarray(
        iiwa14_grasp_tcp_transform(
            tuple(float(value) for value in pick.motion.grasp_arm_rad)
        )
    )
    tcp_from_handbase = np.eye(4)
    tcp_from_handbase[2, 3] = -float(
        pick.geometry_candidate.handbase_to_tcp_m
    )
    nominal_world_handbase = nominal_world_tcp @ tcp_from_handbase
    nominal_world_plug = np.eye(4)
    nominal_world_plug[:3, 3] = np.asarray(
        pick.geometry_candidate.loose_settled_origin_m
    )
    nominal_hand_to_plug = (
        np.linalg.inv(nominal_world_handbase) @ nominal_world_plug
    )
    poses, direction_deg = plan_two_fixed_camera_display_feasibility_poses(
        q0,
        solve_arm=solve_fixed_q7_tcp_pose,
        handbase_to_tcp_m=pick.geometry_candidate.handbase_to_tcp_m,
        nominal_hand_to_plug=nominal_hand_to_plug,
        fixed_camera_eye=(0.55, -0.85, 0.72),
    )
    assert len(poses) == 2
    assert poses[0]["view_id"] != poses[1]["view_id"]
    assert poses[0]["max_abs_dq_rad"] <= 1.25
    assert poses[1]["max_abs_dq_rad"] <= 1.25
    assert direction_deg >= 15.0


def test_two_robot_side_camera_display_poses_are_feasible():
    base = (
        Path(__file__).resolve().parents[3]
        / "artifacts/kcg_connector/d38999_postgrasp_visual_ft_e2e_v1"
        / "phase1_snapshot_gate_v3/seed000"
    )
    snapshot_path = (
        base / "postgrasp_snapshot_gate/snapshot_gate.json"
    )
    if not snapshot_path.is_file():
        pytest.skip("snapshot gate v3 artifact not present")
    from kcg_connector.d38999_physical_insertion import solve_fixed_q7_tcp_pose
    from kcg_connector.d38999_tabletop_pick import (
        iiwa14_grasp_tcp_transform,
        load_d38999_tabletop_pick_config,
    )
    from kcg_connector.postgrasp_shadow_view_planner import (
        plan_two_robot_side_camera_display_poses,
    )

    repository = Path(__file__).resolve().parents[3]
    pick = load_d38999_tabletop_pick_config(
        repository / "src/kcg_connector/config/d38999_tabletop_pick_v1.yaml"
    )
    snapshot = load_snapshot_gate_document(snapshot_path)
    q0 = np.asarray(snapshot["robot_state"]["q_rad"][:7])
    nominal_world_tcp = np.asarray(
        iiwa14_grasp_tcp_transform(
            tuple(float(value) for value in pick.motion.grasp_arm_rad)
        )
    )
    tcp_from_handbase = np.eye(4)
    tcp_from_handbase[2, 3] = -float(
        pick.geometry_candidate.handbase_to_tcp_m
    )
    nominal_world_handbase = nominal_world_tcp @ tcp_from_handbase
    nominal_world_plug = np.eye(4)
    nominal_world_plug[:3, 3] = np.asarray(
        pick.geometry_candidate.loose_settled_origin_m
    )
    nominal_hand_to_plug = (
        np.linalg.inv(nominal_world_handbase) @ nominal_world_plug
    )
    poses, direction_deg = plan_two_robot_side_camera_display_poses(
        q0,
        solve_arm=solve_fixed_q7_tcp_pose,
        handbase_to_tcp_m=pick.geometry_candidate.handbase_to_tcp_m,
        nominal_hand_to_plug=nominal_hand_to_plug,
        fixed_camera_eye=(0.52, -0.21, 0.62),
    )
    assert len(poses) == 2
    assert poses[0]["max_abs_dq_rad"] <= 1.25
    assert poses[1]["max_abs_dq_rad"] <= 1.25
    assert direction_deg >= 15.0
    assert poses[0]["camera_distance_m"] < 0.40
    assert poses[1]["camera_distance_m"] < 0.40


def test_robot_side_f1_path_quality_passes_before_motion():
    base = (
        Path(__file__).resolve().parents[3]
        / "artifacts/kcg_connector/d38999_postgrasp_visual_ft_e2e_v1"
        / "phase1_snapshot_gate_v3/seed000"
    )
    snapshot_path = (
        base / "postgrasp_snapshot_gate/snapshot_gate.json"
    )
    if not snapshot_path.is_file():
        pytest.skip("snapshot gate v3 artifact not present")
    from kcg_connector.d38999_physical_insertion import solve_fixed_q7_tcp_pose
    from kcg_connector.d38999_tabletop_pick import (
        iiwa14_grasp_tcp_transform,
        load_d38999_tabletop_pick_config,
    )
    from kcg_connector.display_motion_diagnostics import (
        evaluate_waypoint_path_quality,
    )
    from kcg_connector.postgrasp_shadow_view_planner import (
        plan_cartesian_tcp_waypoints,
        plan_two_robot_side_camera_display_poses,
    )

    repository = Path(__file__).resolve().parents[3]
    pick = load_d38999_tabletop_pick_config(
        repository / "src/kcg_connector/config/d38999_tabletop_pick_v1.yaml"
    )
    snapshot = load_snapshot_gate_document(snapshot_path)
    q0 = np.asarray(snapshot["robot_state"]["q_rad"][:7])
    nominal_world_tcp = np.asarray(
        iiwa14_grasp_tcp_transform(
            tuple(float(value) for value in pick.motion.grasp_arm_rad)
        )
    )
    tcp_from_handbase = np.eye(4)
    tcp_from_handbase[2, 3] = -float(
        pick.geometry_candidate.handbase_to_tcp_m
    )
    nominal_world_handbase = nominal_world_tcp @ tcp_from_handbase
    nominal_world_plug = np.eye(4)
    nominal_world_plug[:3, 3] = np.asarray(
        pick.geometry_candidate.loose_settled_origin_m
    )
    nominal_hand_to_plug = (
        np.linalg.inv(nominal_world_handbase) @ nominal_world_plug
    )
    poses, _ = plan_two_robot_side_camera_display_poses(
        q0,
        solve_arm=solve_fixed_q7_tcp_pose,
        handbase_to_tcp_m=pick.geometry_candidate.handbase_to_tcp_m,
        nominal_hand_to_plug=nominal_hand_to_plug,
        fixed_camera_eye=(0.52, -0.21, 0.62),
    )
    f0_q = np.asarray(poses[0]["arm_q_rad"])
    f1_waypoints = plan_cartesian_tcp_waypoints(
        f0_q,
        np.asarray(poses[1]["tcp_target"]),
        solve_arm=solve_fixed_q7_tcp_pose,
    )
    distance = float(
        np.linalg.norm(
            np.asarray(poses[1]["tcp_target"])[:3, 3]
            - np.asarray(poses[0]["tcp_target"])[:3, 3]
        )
    )
    steps_per_waypoint = max(
        1, round(max(4.0, distance / 0.010) * 240.0 / len(f1_waypoints))
    )
    limits = [
        (-2.967, 2.967),
        (-2.094, 2.094),
        (-2.967, 2.967),
        (-2.094, 2.094),
        (-2.967, 2.967),
        (-2.0943951024, 2.0943951024),
        (-2.967, 2.967),
    ]
    quality = evaluate_waypoint_path_quality(
        f1_waypoints,
        forward_kinematics=iiwa14_grasp_tcp_transform,
        physics_rate_hz=240.0,
        steps_per_waypoint=steps_per_waypoint,
        start_q=f0_q,
        table_top_z_m=0.20,
        fixture_center_m=(0.55, 0.185, 0.22),
        fixture_half_extent_m=(0.07, 0.07, 0.02),
        joint_limits=limits,
        joint_limit_margin_rad=0.010,
    )
    assert quality["reject"] is False
    assert quality["minimum_jacobian_singular_value"] > 0.02
    assert max(float(w[5]) for w in f1_waypoints) < 2.0943951024 - 0.010
