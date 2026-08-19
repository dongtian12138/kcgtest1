from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from kcg_connector.d38999_multilayer_body_release import (
    BodyReleaseReadiness,
    FROZEN_SOURCES,
    build_body_release_contract,
    evaluate_body_release_gate,
)

ROOT = Path(__file__).resolve().parents[3]
CONTRACT = {
    "hand_joint_order": ["f1j1", "f1j2", "f2j1", "f3j1"],
    "open_hand_rad": [1.0, 0.0, 0.0, 0.0],
    "release_duration_s": 2.5,
    "maximum_joint_speed_rad_s": 1.0,
    "hard_stop_nm": 2.0,
    "maximum_sample_gap_s": 0.0125,
}


def _ready(**kw):
    data = dict(e1_postinsert_hold_dynamic_pass=True, wrist_guard_safe=True,
                wrist_guard_fault_latched=False, hand_joint_state_calibrated=True,
                e1_evidence_id="e1-dynamic-1")
    data.update(kw)
    return BodyReleaseReadiness(**data)


def _eval(readiness=None, **kw):
    data = dict(current_hand_positions_rad=[1.0, 0.7, 0.5, 0.7],
                current_hand_efforts_nm=[0.1] * 4, sample_timestamp_s=1.0,
                now_s=1.01, contract=CONTRACT)
    data.update(kw)
    return evaluate_body_release_gate(readiness or _ready(), **data)


def test_contract_extracts_only_hand_space_release():
    contract = build_body_release_contract(ROOT)
    assert contract["open_hand_rad"] == [1.0, 0.0, 0.0, 0.0]
    assert contract["release_duration_s"] == pytest.approx(2.5)
    assert contract["maximum_joint_speed_rad_s"] == pytest.approx(1.0)
    assert contract["hard_stop_nm"] == pytest.approx(2.0)
    assert contract["current_decision"]["rejection_code"] == "E1_POSTINSERT_HOLD_NOT_DYNAMIC"
    assert "body_root_world_m" in contract["excluded_legacy_fields"]


def test_valid_fixture_produces_minimum_jerk_candidate_only():
    result = _eval()
    plan = result["release_plan_candidate"]
    assert result["status"] == "OFFLINE_BODY_RELEASE_PLAN_CANDIDATE"
    assert plan["target_rad"] == [1.0, 0.0, 0.0, 0.0]
    assert plan["duration_s"] == pytest.approx(2.5)
    assert plan["profile"] == "minimum_jerk"
    assert max(plan["peak_speed_rad_s"]) < 1.0


@pytest.mark.parametrize("kw,code", [
    ({"e1_postinsert_hold_dynamic_pass": False}, "E1_POSTINSERT_HOLD_NOT_DYNAMIC"),
    ({"e1_evidence_id": None}, "E1_EVIDENCE_ID_MISSING"),
    ({"hand_joint_state_calibrated": False}, "HAND_JOINT_STATE_NOT_CALIBRATED"),
    ({"wrist_guard_safe": False}, "WRIST_MOMENT_GUARD_REJECTED"),
    ({"wrist_guard_fault_latched": True}, "WRIST_MOMENT_GUARD_REJECTED"),
])
def test_readiness_fails_closed(kw, code):
    assert _eval(_ready(**kw))["rejection_code"] == code


@pytest.mark.parametrize("name,value", [
    ("current_hand_positions_rad", [0.0] * 3),
    ("current_hand_positions_rad", [0.0, 0.0, float("nan"), 0.0]),
    ("current_hand_efforts_nm", [0.0] * 3),
    ("current_hand_efforts_nm", [0.0, 0.0, float("inf"), 0.0]),
])
def test_invalid_hand_vectors_fail_closed(name, value):
    assert _eval(**{name: value})["rejection_code"] == "INVALID_HAND_SAMPLE"


def test_exact_effort_hard_stop_passes_and_over_limit_fails():
    assert _eval(current_hand_efforts_nm=[2.0, 0, 0, 0])["release_plan_candidate"]
    assert _eval(current_hand_efforts_nm=[2.000001, 0, 0, 0])["rejection_code"] == "FINGER_EFFORT_HARD_STOP"


def test_exact_sample_age_passes_and_stale_fails():
    assert _eval(now_s=1.0125)["release_plan_candidate"]
    assert _eval(now_s=1.012501)["rejection_code"] == "INVALID_OR_STALE_HAND_SAMPLE"


@pytest.mark.parametrize("kw", [
    {"sample_timestamp_s": 2.0}, {"now_s": float("nan")}, {"sample_timestamp_s": True}
])
def test_invalid_sample_times_fail_closed(kw):
    assert _eval(**kw)["rejection_code"] in {"INVALID_HAND_SAMPLE_TIME", "INVALID_OR_STALE_HAND_SAMPLE"}


def test_speed_limit_is_computed_for_profile():
    contract = {**CONTRACT, "release_duration_s": 0.1}
    assert _eval(current_hand_positions_rad=[0.0] * 4, contract=contract)["rejection_code"] == "RELEASE_PROFILE_SPEED_LIMIT"


def test_strict_boolean_readiness():
    assert _eval(_ready(wrist_guard_safe=1))["rejection_code"] == "INVALID_READINESS_SNAPSHOT"


def test_no_path_emits_commands_or_claims():
    for result in (_eval(), _eval(_ready(e1_postinsert_hold_dynamic_pass=False))):
        assert result["robot_motion_command_emitted"] is False
        assert result["finger_command_emitted"] is False
        assert result["control_authorized"] is False
        assert result["dynamic_body_release_pass_claimed"] is False


def test_truth_firewall_signature():
    names = set(inspect.signature(evaluate_body_release_gate).parameters)
    assert names.isdisjoint({"object_pose", "contact_name", "contact_normal", "event_truth"})


def test_source_drift_rejected(tmp_path):
    root = tmp_path / "repo"
    for relative in FROZEN_SOURCES:
        dst = root / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((ROOT / relative).read_bytes())
    target = root / next(iter(FROZEN_SOURCES))
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        build_body_release_contract(root)


def test_strict_json():
    json.dumps(_eval(), sort_keys=True, allow_nan=False)
