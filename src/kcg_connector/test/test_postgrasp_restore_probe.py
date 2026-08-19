import pytest

from kcg_connector.postgrasp_snapshot_truth import (
    POSTGRASP_SNAPSHOT_SCHEMA_VERSION,
    RestoreRejected,
    SettleNotComplete,
    capture_allowed_after_step,
    probe_restore_api,
    restore_snapshot,
)


def _snapshot():
    return {
        "schema_version": POSTGRASP_SNAPSHOT_SCHEMA_VERSION,
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
        "coupling_nut_joint_state": {"q_rad": 0.0, "qd_rad_s": 0.0},
        "robot_state": {"q_rad": [0.0] * 11, "qd_rad_s": [0.0] * 11},
        "frozen_command": {},
        "source_hashes": {},
    }


def test_api_probe_unsupported_fails_closed():
    probe = probe_restore_api({})
    assert probe.status == "API_UNSUPPORTED"
    with pytest.raises(RestoreRejected):
        restore_snapshot(_snapshot(), {})


def test_api_probe_native_mode():
    class Native:
        def restore(self, document):
            assert document["plug_root_state"]["position_m"] == [0.0, 0.0, 0.0]

    probe = probe_restore_api({"native_articulation_state": Native()})
    assert probe.mode == "NATIVE_ARTICULATION_STATE"
    result = restore_snapshot(_snapshot(), {"native_articulation_state": Native()})
    assert result["restore_status"] == "RESTORED"


def test_linked_body_write_is_forbidden_even_with_root_joint_api():
    bindings = {
        "plug_root_set_pose": lambda *args: None,
        "plug_root_set_linear_velocity": lambda *args: None,
        "plug_root_set_angular_velocity": lambda *args: None,
        "nut_joint_set_state": lambda *args: None,
        "nut_set_world_pose": lambda *args: None,
    }
    with pytest.raises(RestoreRejected):
        restore_snapshot(_snapshot(), bindings)


def test_capture_before_settle_raises():
    with pytest.raises(SettleNotComplete):
        capture_allowed_after_step(119, settle_steps=120)
    assert capture_allowed_after_step(120, settle_steps=120)
