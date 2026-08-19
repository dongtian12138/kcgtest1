import numpy as np
import pytest

from kcg_connector.postgrasp_snapshot_truth import (
    POSTGRASP_SNAPSHOT_SCHEMA_VERSION,
    RestoreRejected,
    SettleNotComplete,
    SnapshotContractError,
    TruthFirewallViolation,
    capture_allowed_after_step,
    load_truth_snapshot,
    probe_restore_api,
    restore_snapshot,
    settle_tail_diagnostic,
    validate_truth_snapshot,
    write_truth_snapshot,
)


def _document(joint_state=True):
    return {
        "schema_version": POSTGRASP_SNAPSHOT_SCHEMA_VERSION,
        "role": "truth_restore",
        "scope": "snapshot_restore_and_posthoc_evaluation_only",
        "snapshot_id": "snap0",
        "timestamp_utc": "2026-08-15T00:00:00Z",
        "episode": "seed000000",
        "seed": 0,
        "global_step": 13426,
        "plug_root_state": {
            "position_m": [0.5, -0.2, 0.25],
            "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
            "linear_velocity_m_s": [0.0, 0.0, 0.0],
            "angular_velocity_rad_s": [0.0, 0.0, 0.0],
        },
        "coupling_nut_joint_state": (
            {"q_rad": 0.01, "qd_rad_s": 0.0} if joint_state else None
        ),
        "robot_state": {
            "q_rad": [0.0, 0.1, -0.2, 0.3, 0.0, 0.1, 0.0, 0.2, 0.3, 0.2, 0.1],
            "qd_rad_s": [0.0] * 11,
        },
        "frozen_command": {
            "arm_q_target_rad": [0.0] * 7,
            "hand_q_target_rad": [0.2] * 4,
        },
        "source_hashes": {"runner": "abc"},
    }


def test_truth_document_roundtrip_and_role_firewall(tmp_path):
    document = _document()
    validated = validate_truth_snapshot(document)
    assert validated["role"] == "truth_restore"
    path = tmp_path / "truth.json"
    write_truth_snapshot(path, validated)
    assert load_truth_snapshot(path) == validated
    bad = dict(document)
    bad["role"] = "formal_observation"
    with pytest.raises(TruthFirewallViolation):
        validate_truth_snapshot(bad)


def test_sha_mismatch_fails(tmp_path):
    path = tmp_path / "truth.json"
    write_truth_snapshot(path, _document())
    with pytest.raises(SnapshotContractError):
        load_truth_snapshot(path, expected_sha256="0" * 64)


def test_probe_and_restore_root_joint_contract():
    calls = []
    bindings = {
        "plug_root_set_pose": lambda p, q: calls.append(("pose", p, q)),
        "plug_root_set_linear_velocity": lambda v: calls.append(("lv", v)),
        "plug_root_set_angular_velocity": lambda w: calls.append(("av", w)),
        "nut_joint_set_state": lambda q, qd: calls.append(("joint", q, qd)),
    }
    probe = probe_restore_api(bindings)
    assert probe.mode == "ROOT_BODY_JOINT_STATE"
    restored = restore_snapshot(_document(), bindings)
    assert restored["mode"] == "ROOT_BODY_JOINT_STATE"
    assert len(calls) == 4


def test_restore_rejects_linked_nut_world_pose():
    bindings = {
        "plug_root_set_pose": lambda p, q: None,
        "plug_root_set_linear_velocity": lambda v: None,
        "plug_root_set_angular_velocity": lambda w: None,
        "nut_joint_set_state": lambda q, qd: None,
        "nut_set_world_pose": lambda p, q: None,
    }
    with pytest.raises(RestoreRejected):
        restore_snapshot(_document(), bindings)


def test_restore_missing_joint_state_fails():
    bindings = {
        "plug_root_set_pose": lambda p, q: None,
        "plug_root_set_linear_velocity": lambda v: None,
        "plug_root_set_angular_velocity": lambda w: None,
        "nut_joint_set_state": lambda q, qd: None,
    }
    with pytest.raises(RestoreRejected):
        restore_snapshot(_document(joint_state=False), bindings)


def test_settle_gate_and_tail_diagnostic():
    with pytest.raises(SettleNotComplete):
        capture_allowed_after_step(119)
    assert capture_allowed_after_step(120) is True
    tail = np.random.default_rng(0).normal(0.1, 0.01, size=(60, 6))
    diag = settle_tail_diagnostic(
        tail, np.zeros(6), np.full(6, 1.0e-6)
    )
    assert diag["pass_gate"] is None
    assert diag["diagnostic_only"] is True
    assert len(diag["z_score"]) == 6
