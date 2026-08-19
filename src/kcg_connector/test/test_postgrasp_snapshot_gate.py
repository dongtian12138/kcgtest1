from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from kcg_connector.postgrasp_snapshot_gate import (
    GATE_ROLE,
    GATE_SCHEMA_VERSION,
    SettleNotComplete,
    SnapshotContractError,
    REPLAY_BUNDLE_SCHEMA_VERSION,
    build_replay_bundle_manifest,
    capture_allowed_after_settle,
    capture_snapshot_gate_document,
    evaluate_restore_consistency,
    load_replay_bundle_manifest,
    load_snapshot_gate_document,
    sha256_file,
    validate_replay_bundle_manifest,
    write_replay_bundle_manifest,
    write_snapshot_gate_document,
)


def _getters():
    return {
        "robot_getters": {
            "q": lambda: np.zeros(11),
            "qd": lambda: np.zeros(11),
        },
        "finger_getters": {
            "hand_q_actual": lambda: np.zeros(4),
            "hand_q_target": lambda: np.zeros(4),
            "finger_root_torque_proxy": lambda: np.array([0.25, 0.26, 0.25]),
            "finger_root_tare_efforts": lambda: np.array([0.10, 0.11, 0.12]),
        },
        "plug_root_getters": {
            "position": lambda: np.array([0.5, -0.2, 0.3]),
            "orientation": lambda: np.array([1.0, 0.0, 0.0, 0.0]),
            "linear_velocity": lambda: np.zeros(3),
            "angular_velocity": lambda: np.zeros(3),
        },
        "nut_joint_getters": {"q": lambda: 0.01, "qd": lambda: 0.0},
        "nut_posthoc_getters": {
            "position": lambda: np.array([0.5, -0.2, 0.33]),
            "orientation": lambda: np.array([1.0, 0.0, 0.0, 0.0]),
            "linear_velocity": lambda: np.zeros(3),
            "angular_velocity": lambda: np.zeros(3),
        },
    }


def _document():
    getters = _getters()
    return capture_snapshot_gate_document(
        snapshot_id="s",
        timestamp_utc="2026-08-15T00:00:00Z",
        episode="seed000000",
        seed=0,
        global_step=12000,
        physics_step=12000,
        frozen_command={
            "arm_q_target_rad": np.zeros(7).tolist(),
            "hand_q_target_rad": np.zeros(4).tolist(),
            "wrist_ft_payload_reference": np.zeros(6).tolist(),
            "wrist_ft_snapshot_reference": np.zeros(6).tolist(),
            "wrist_ft_snapshot_global_step": 12000,
        },
        provenance={"runner": "abc"},
        **getters,
    )


def test_snapshot_roundtrip_and_role(tmp_path):
    document = _document()
    path = tmp_path / "snapshot.json"
    write_snapshot_gate_document(path, document)
    loaded = load_snapshot_gate_document(path)
    assert loaded["schema_version"] == GATE_SCHEMA_VERSION
    assert loaded["role"] == GATE_ROLE
    assert loaded["nut_rigid_state_restore_only"]["position_m"][2] == 0.33
    assert loaded["finger_state"]["finger_root_tare_efforts_nm"] == [
        0.10,
        0.11,
        0.12,
    ]


def test_settle_gate_before_120_steps(tmp_path):
    with pytest.raises(SettleNotComplete):
        capture_allowed_after_settle(119)
    assert capture_allowed_after_settle(120) is True


def test_consistency_verified_for_nominal_tail(tmp_path):
    document = _document()
    result = evaluate_restore_consistency(
        snapshot=document,
        tail_robot_q=np.zeros((60, 11)),
        tail_robot_qd=np.zeros((60, 11)),
        tail_finger_torque=np.tile([0.25, 0.26, 0.25], (60, 1)),
        tail_wrist_wrench=np.zeros((60, 6)),
        restored_plug_position=np.array([0.5, -0.2, 0.3]),
        restored_plug_orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        restored_nut_position=np.array([0.5, -0.2, 0.33]),
        restored_nut_orientation=np.array([1.0, 0.0, 0.0, 0.0]),
    )
    assert result.verified is True


def test_consistency_fails_on_load_drift(tmp_path):
    document = _document()
    result = evaluate_restore_consistency(
        snapshot=document,
        tail_robot_q=np.zeros((60, 11)),
        tail_robot_qd=np.zeros((60, 11)),
        tail_finger_torque=np.tile([0.8, 0.8, 0.8], (60, 1)),
        tail_wrist_wrench=np.zeros((60, 6)),
        restored_plug_position=np.array([0.5015, -0.2, 0.3]),
        restored_plug_orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        restored_nut_position=np.array([0.5, -0.2, 0.33]),
        restored_nut_orientation=np.array([1.0, 0.0, 0.0, 0.0]),
    )
    assert result.verified is False
    assert "FINGER_LOAD_INCONSISTENT" in result.reasons
    assert "PLUG_POSITION_POSTHOC_INCONSISTENT" in result.reasons


def test_missing_field_fails(tmp_path):
    document = _document()
    document.pop("finger_state")
    with pytest.raises(Exception):
        write_snapshot_gate_document(tmp_path / "bad.json", document)


def test_pick_smoke_snapshot_gate_contract():
    from pathlib import Path
    source = (
        Path(__file__).resolve().parents[1]
        / "isaac/d38999_tabletop_pick_smoke.py"
    ).read_text(encoding="utf-8")
    assert "--postgrasp-snapshot-gate" in source
    assert "run_postgrasp_snapshot_gate(" in source
    assert "snapshot_restore_verified" in source
    assert "process_exit_code = 3" in source
    assert "stage_exporter" in source


def _runtime_module():
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "isaac/postgrasp_snapshot_gate_runtime.py"
    spec = importlib.util.spec_from_file_location("snap_rt", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runtime_restores_two_bodies_only_at_boundary(tmp_path):
    rt = _runtime_module()

    class RigidBody:
        def __init__(self, z):
            self.position = np.array([0.5, -0.2, z], dtype=float)
            self.orientation = np.array([1.0, 0.0, 0.0, 0.0])
            self.linear = np.zeros(3)
            self.angular = np.zeros(3)
            self.pose_writes = 0

        def get_world_pose(self):
            return self.position.copy(), self.orientation.copy()

        def set_world_pose(self, *, position, orientation):
            self.pose_writes += 1
            self.position = np.asarray(position).copy()
            self.orientation = np.asarray(orientation).copy()

        def get_linear_velocity(self):
            return self.linear.copy()

        def get_angular_velocity(self):
            return self.angular.copy()

        def set_linear_velocity(self, value):
            self.linear = np.asarray(value).copy()

        def set_angular_velocity(self, value):
            self.angular = np.asarray(value).copy()

    class Robot:
        def get_joint_positions(self):
            return np.zeros(11)

        def get_joint_velocities(self):
            return np.zeros(11)

    body = RigidBody(0.3)
    nut = RigidBody(0.33)
    state = {"step": 10}

    def observe_and_step(_arm, _hand, _allow):
        state["step"] += 1
        return np.zeros(11), np.zeros(11)

    q_writes = []
    qd_writes = []
    result = rt.run_postgrasp_snapshot_gate(
        arguments=SimpleNamespace(output_dir=str(tmp_path), seed=0),
        global_step=10,
        physics_step=10,
        body=body,
        nut=nut,
        robot=Robot(),
        hand_indices=np.array([7, 8, 9, 10]),
        robot_set_q=lambda value: q_writes.append(np.asarray(value)),
        robot_set_qd=lambda value: qd_writes.append(np.asarray(value)),
        current_arm_target=np.zeros(7),
        current_hand_target=np.zeros(4),
        observe_and_step=observe_and_step,
        sample_post_tare_efforts=lambda: np.zeros(3),
        formal_wrist_payload_reference=np.zeros(6),
        get_latest_wrist_state=lambda: {
            "global_step": state["step"],
            "canonical": np.zeros(6),
            "error": None,
        },
        tare_efforts=np.zeros(3),
        stage_exporter=lambda path: (
            Path(path).write_text("#usda 1.0\n", encoding="utf-8") or True
        ),
        source_hashes={"runner": "abc"},
    )
    assert result["status"] == "SNAPSHOT_GATE_VERIFIED"
    assert result["object_pose_writes_at_restore_boundary"] == 2
    assert result["object_pose_writes_during_settle"] == 0
    assert body.pose_writes == 1
    assert nut.pose_writes == 1
    assert len(q_writes) == len(qd_writes) == 1
    gate_dir = tmp_path / "postgrasp_snapshot_gate"
    assert (gate_dir / "snapshot_gate.json").is_file()
    assert (gate_dir / "replay_stage.usda").is_file()
    assert (gate_dir / "replay_bundle_manifest.json").is_file()
    bundle = result["replay_bundle"]
    assert bundle["restore_truth_scope"] == "INITIALIZATION_ONLY"
    assert bundle["formal_estimator_input"] is False
    assert bundle["control_authorized"] is False
    assert bundle["stage_export_scope"] == "REPLAY_ONLY_NOT_CONTROL_OR_ESTIMATOR"


def test_snapshot_wrist_reference_wins_over_older_payload_reference():
    document = _document()
    document["frozen_command"]["wrist_ft_payload_reference"] = [
        0.362886, -0.163855, 19.928412, 0.113355, 0.192235, 0.003107
    ]
    document["frozen_command"]["wrist_ft_snapshot_reference"] = [
        -0.058404, 0.004655, 21.814596, 0.034045, 0.031012, 0.000067
    ]
    tail = np.tile(
        [-0.058404, 0.004655, 21.814596, 0.034045, 0.031012, 0.000067],
        (60, 1),
    )
    result = evaluate_restore_consistency(
        snapshot=document,
        tail_robot_qd=np.zeros((60, 11)),
        tail_robot_q=np.zeros((60, 11)),
        tail_finger_torque=np.tile([0.25, 0.26, 0.25], (60, 1)),
        tail_wrist_wrench=tail,
        restored_plug_position=np.array([0.5, -0.2, 0.3]),
        restored_plug_orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        restored_nut_position=np.array([0.5, -0.2, 0.33]),
        restored_nut_orientation=np.array([1.0, 0.0, 0.0, 0.0]),
    )
    assert result.verified is True


def test_snapshot_step_must_equal_global_step():
    document = _document()
    document["frozen_command"]["wrist_ft_snapshot_global_step"] = 11999
    with pytest.raises(Exception):
        write_snapshot_gate_document(Path(__file__).parent / "bad_sync.json", document)


def test_force_and_moment_gates_are_separate():
    document = _document()
    base = np.asarray(document["frozen_command"]["wrist_ft_snapshot_reference"])
    tail_force = np.tile(base + np.array([1.01, 0, 0, 0, 0, 0]), (60, 1))
    tail_moment = np.tile(base + np.array([0, 0, 0, 0.11, 0, 0]), (60, 1))
    common = dict(
        snapshot=document,
        tail_robot_qd=np.zeros((60, 11)),
        tail_robot_q=np.zeros((60, 11)),
        tail_finger_torque=np.tile([0.25, 0.26, 0.25], (60, 1)),
        restored_plug_position=np.array([0.5, -0.2, 0.3]),
        restored_plug_orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        restored_nut_position=np.array([0.5, -0.2, 0.33]),
        restored_nut_orientation=np.array([1.0, 0.0, 0.0, 0.0]),
    )
    force_result = evaluate_restore_consistency(tail_wrist_wrench=tail_force, **common)
    assert "WRIST_FORCE_LOAD_INCONSISTENT" in force_result.reasons
    assert "WRIST_MOMENT_LOAD_INCONSISTENT" not in force_result.reasons
    moment_result = evaluate_restore_consistency(tail_wrist_wrench=tail_moment, **common)
    assert "WRIST_MOMENT_LOAD_INCONSISTENT" in moment_result.reasons
    assert "WRIST_FORCE_LOAD_INCONSISTENT" not in moment_result.reasons


def test_live_wrist_changes_are_sampled_and_stale_fails():
    rt = _runtime_module()
    state = {"step": 0}
    state["canonical"] = np.zeros(6)
    state["error"] = None
    state["step"] += 1
    step, wrist = rt._read_live_wrist(lambda: {
        "global_step": state["step"], "canonical": state["canonical"], "error": None
    }, 0)
    assert step == 1
    state["canonical"] = np.array([9.0, 0, 0, 0, 0, 0])
    with pytest.raises(RuntimeError):
        rt._read_live_wrist(lambda: {
            "global_step": state["step"], "canonical": state["canonical"], "error": None
        }, 1)
    # stale same step fails
    with pytest.raises(RuntimeError):
        rt._read_live_wrist(lambda: {
            "global_step": 1, "canonical": np.zeros(6), "error": None
        }, 1)


def test_finger_tare_missing_legacy_loads_but_replay_bundle_rejects(tmp_path):
    document = _document()
    del document["finger_state"]["finger_root_tare_efforts_nm"]
    legacy_path = tmp_path / "legacy_v2_snapshot.json"
    # Historical evidence may still be inspected through the raw loader.
    write_snapshot_gate_document(legacy_path, document)
    loaded_legacy = load_snapshot_gate_document(legacy_path)
    assert "finger_root_tare_efforts_nm" not in loaded_legacy["finger_state"]

    gate_dir = tmp_path / "postgrasp_snapshot_gate"
    gate_dir.mkdir(parents=True, exist_ok=True)
    write_snapshot_gate_document(gate_dir / "snapshot_gate.json", document)
    (gate_dir / "replay_stage.usda").write_text("#usda 1.0\n", encoding="utf-8")
    manifest = build_replay_bundle_manifest(
        seed=0,
        snapshot_file="snapshot_gate.json",
        snapshot_sha256=sha256_file(gate_dir / "snapshot_gate.json"),
        stage_file="replay_stage.usda",
        stage_sha256=sha256_file(gate_dir / "replay_stage.usda"),
        source_hashes={"runner": "abc"},
    )
    write_replay_bundle_manifest(gate_dir / "replay_bundle_manifest.json", manifest)
    with pytest.raises(SnapshotContractError):
        load_replay_bundle_manifest(gate_dir / "replay_bundle_manifest.json")


def test_finger_tare_wrong_shape_fails(tmp_path):
    for bad in ([0.1, 0.2], [0.1, 0.2, 0.3, 0.4]):
        document = _document()
        document["finger_state"]["finger_root_tare_efforts_nm"] = bad
        with pytest.raises(Exception):
            write_snapshot_gate_document(tmp_path / "bad_tare_shape.json", document)


def test_finger_tare_non_finite_fails(tmp_path):
    for bad in ([0.1, np.nan, 0.3], [0.1, np.inf, 0.3]):
        document = _document()
        document["finger_state"]["finger_root_tare_efforts_nm"] = bad
        with pytest.raises(Exception):
            write_snapshot_gate_document(tmp_path / "bad_tare_finite.json", document)


class _GateRigidBody:
    def __init__(self, z):
        self.position = np.array([0.5, -0.2, z], dtype=float)
        self.orientation = np.array([1.0, 0.0, 0.0, 0.0])
        self.linear = np.zeros(3)
        self.angular = np.zeros(3)
        self.pose_writes = 0

    def get_world_pose(self):
        return self.position.copy(), self.orientation.copy()

    def set_world_pose(self, *, position, orientation):
        self.pose_writes += 1
        self.position = np.asarray(position).copy()
        self.orientation = np.asarray(orientation).copy()

    def get_linear_velocity(self):
        return self.linear.copy()

    def get_angular_velocity(self):
        return self.angular.copy()

    def set_linear_velocity(self, value):
        self.linear = np.asarray(value).copy()

    def set_angular_velocity(self, value):
        self.angular = np.asarray(value).copy()


class _GateRobot:
    def get_joint_positions(self):
        return np.zeros(11)

    def get_joint_velocities(self):
        return np.zeros(11)


def _run_gate(tmp_path, *, stage_exporter, tare_efforts, source_hashes=None):
    rt = _runtime_module()
    body = _GateRigidBody(0.3)
    nut = _GateRigidBody(0.33)
    state = {"step": 10}

    def observe_and_step(_arm, _hand, _allow):
        state["step"] += 1
        return np.zeros(11), np.zeros(11)

    result = rt.run_postgrasp_snapshot_gate(
        arguments=SimpleNamespace(output_dir=str(tmp_path), seed=0),
        global_step=10,
        physics_step=10,
        body=body,
        nut=nut,
        robot=_GateRobot(),
        hand_indices=np.array([7, 8, 9, 10]),
        robot_set_q=lambda value: None,
        robot_set_qd=lambda value: None,
        current_arm_target=np.zeros(7),
        current_hand_target=np.zeros(4),
        observe_and_step=observe_and_step,
        sample_post_tare_efforts=lambda: np.zeros(3),
        formal_wrist_payload_reference=np.zeros(6),
        get_latest_wrist_state=lambda: {
            "global_step": state["step"],
            "canonical": np.zeros(6),
            "error": None,
        },
        tare_efforts=tare_efforts,
        stage_exporter=stage_exporter,
        source_hashes=(
            {"runner": "abc"} if source_hashes is None else source_hashes
        ),
    )
    return result, tmp_path / "postgrasp_snapshot_gate"


def test_runtime_stage_export_false_fails_closed(tmp_path):
    result, gate_dir = _run_gate(
        tmp_path, stage_exporter=lambda path: False, tare_efforts=np.zeros(3)
    )
    assert result["status"] == "SNAPSHOT_GATE_ABORT_SAFE"
    assert result["snapshot_restore_verified"] is False
    assert result["control_authorized"] is False
    assert result["formal_estimator_input"] is False
    assert not (gate_dir / "replay_bundle_manifest.json").exists()


def test_runtime_stage_export_missing_file_fails_closed(tmp_path):
    result, _ = _run_gate(
        tmp_path, stage_exporter=lambda path: True, tare_efforts=np.zeros(3)
    )
    assert result["status"] == "SNAPSHOT_GATE_ABORT_SAFE"
    assert result["control_authorized"] is False


def test_runtime_stage_export_empty_file_fails_closed(tmp_path):
    def exporter(path):
        Path(path).write_text("", encoding="utf-8")
        return True

    result, _ = _run_gate(tmp_path, stage_exporter=exporter, tare_efforts=np.zeros(3))
    assert result["status"] == "SNAPSHOT_GATE_ABORT_SAFE"
    assert result["control_authorized"] is False


def test_runtime_tare_wrong_shape_fails_closed(tmp_path):
    def exporter(path):
        Path(path).write_text("#usda 1.0\n", encoding="utf-8")
        return True

    result, _ = _run_gate(tmp_path, stage_exporter=exporter, tare_efforts=np.zeros(2))
    assert result["status"] == "SNAPSHOT_GATE_ABORT_SAFE"


def test_runtime_tare_missing_fails_closed(tmp_path):
    def exporter(path):
        Path(path).write_text("#usda 1.0\n", encoding="utf-8")
        return True

    result, _ = _run_gate(tmp_path, stage_exporter=exporter, tare_efforts=None)
    assert result["status"] == "SNAPSHOT_GATE_ABORT_SAFE"
    assert result["control_authorized"] is False


def test_runtime_mixed_provenance_keeps_only_string_hashes(tmp_path):
    def exporter(path):
        Path(path).write_text("#usda 1.0\n", encoding="utf-8")
        return True

    result, gate_dir = _run_gate(
        tmp_path,
        stage_exporter=exporter,
        tare_efforts=np.zeros(3),
        source_hashes={
            "runner_sha256": "a" * 64,
            "seed": 0,
            "audit_mode": None,
        },
    )
    assert result["status"] == "SNAPSHOT_GATE_VERIFIED"
    manifest = load_replay_bundle_manifest(
        gate_dir / "replay_bundle_manifest.json"
    )
    assert manifest["source_hashes"] == {"runner_sha256": "a" * 64}
    snapshot = load_snapshot_gate_document(gate_dir / "snapshot_gate.json")
    assert snapshot["provenance"] == {"runner_sha256": "a" * 64}


def _write_bundle(tmp_path):
    gate_dir = tmp_path / "postgrasp_snapshot_gate"
    gate_dir.mkdir(parents=True, exist_ok=True)
    write_snapshot_gate_document(gate_dir / "snapshot_gate.json", _document())
    (gate_dir / "replay_stage.usda").write_text("#usda 1.0\n", encoding="utf-8")
    manifest = build_replay_bundle_manifest(
        seed=0,
        snapshot_file="snapshot_gate.json",
        snapshot_sha256=sha256_file(gate_dir / "snapshot_gate.json"),
        stage_file="replay_stage.usda",
        stage_sha256=sha256_file(gate_dir / "replay_stage.usda"),
        source_hashes={"runner": "abc"},
    )
    write_replay_bundle_manifest(gate_dir / "replay_bundle_manifest.json", manifest)
    return gate_dir


def test_replay_bundle_manifest_roundtrip(tmp_path):
    gate_dir = _write_bundle(tmp_path)
    loaded = load_replay_bundle_manifest(gate_dir / "replay_bundle_manifest.json")
    assert loaded["schema_version"] == REPLAY_BUNDLE_SCHEMA_VERSION
    assert loaded["restore_truth_scope"] == "INITIALIZATION_ONLY"
    assert loaded["formal_estimator_input"] is False
    assert loaded["control_authorized"] is False
    assert loaded["seed"] == 0
    assert loaded["snapshot_file"] == "snapshot_gate.json"
    assert loaded["stage_file"] == "replay_stage.usda"


def test_replay_bundle_manifest_hash_mismatch_fails(tmp_path):
    gate_dir = _write_bundle(tmp_path)
    with open(gate_dir / "snapshot_gate.json", "a", encoding="utf-8") as handle:
        handle.write("tampered")
    with pytest.raises(Exception):
        load_replay_bundle_manifest(gate_dir / "replay_bundle_manifest.json")
    write_snapshot_gate_document(gate_dir / "snapshot_gate.json", _document())
    with open(gate_dir / "replay_stage.usda", "a", encoding="utf-8") as handle:
        handle.write("tampered")
    with pytest.raises(Exception):
        load_replay_bundle_manifest(gate_dir / "replay_bundle_manifest.json")


def test_replay_bundle_manifest_missing_artifact_fails(tmp_path):
    gate_dir = _write_bundle(tmp_path)
    (gate_dir / "replay_stage.usda").unlink()
    with pytest.raises(Exception):
        load_replay_bundle_manifest(gate_dir / "replay_bundle_manifest.json")


def test_replay_bundle_manifest_rejects_truth_authorization():
    base = build_replay_bundle_manifest(
        seed=0,
        snapshot_file="snapshot_gate.json",
        snapshot_sha256="a" * 64,
        stage_file="replay_stage.usda",
        stage_sha256="b" * 64,
        source_hashes={},
    )
    for field in ("control_authorized", "formal_estimator_input"):
        tampered = dict(base)
        tampered[field] = True
        with pytest.raises(Exception):
            validate_replay_bundle_manifest(tampered)
    tampered = dict(base)
    tampered["restore_truth_scope"] = "FORMAL_CONTROL"
    with pytest.raises(Exception):
        validate_replay_bundle_manifest(tampered)
