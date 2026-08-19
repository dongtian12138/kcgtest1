from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from kcg_connector.d38999_multilayer_nut_regrasp import (
    FROZEN_SOURCES, NutRegraspReadiness, build_nut_regrasp_contract,
    evaluate_nut_regrasp_gate,
)

ROOT = Path(__file__).resolve().parents[3]
NUT = "/World/D38999MultilayerV1/D38999_ASSEMBLY_CONTROL_V1/D38999Pair/LoosePlug/CouplingNut"
AUTH = {
    "source_path": "authority.json", "source_sha256": "a" * 64,
    "plan_artifact_path": "plan.json", "plan_artifact_sha256": "b" * 64,
    "coupling_nut_path": NUT, "visual_tactile_plan_ready": True,
}


def _ready(**kw):
    data = dict(e2_body_release_dynamic_pass=True, visual_pose_accepted=True,
                tactile_ready_latched=True, wrist_guard_safe=True,
                wrist_guard_fault_latched=False, e2_evidence_id="e2-dynamic-1")
    data.update(kw)
    return NutRegraspReadiness(**data)


def _eval(readiness=None, **kw):
    data = dict(current_target_authority=AUTH, coupling_nut_path=NUT)
    data.update(kw)
    return evaluate_nut_regrasp_gate(readiness or _ready(), **data)


def test_contract_loads_legacy_adapter_but_does_not_migrate_world_target():
    contract = build_nut_regrasp_contract(ROOT)
    assert contract["coupling_nut_path"] == NUT
    assert contract["legacy_adapter_status"].endswith("not_gpu_validated")
    assert contract["legacy_world_target_auto_migration_allowed"] is False
    assert contract["current_multilayer_target_authority_available"] is False
    assert contract["current_decision"]["rejection_code"] == "E2_BODY_RELEASE_NOT_DYNAMIC"


def test_valid_fixture_yields_path_and_digest_request_without_pose():
    result = _eval()
    request = result["regrasp_request_candidate"]
    assert result["status"] == "OFFLINE_NUT_REGRASP_REQUEST_CANDIDATE"
    assert request["coupling_nut_path"] == NUT
    assert request["pose_values_embedded"] is False
    assert set(request).isdisjoint({"tcp_position", "object_pose", "contact_normal"})


@pytest.mark.parametrize("kw,code", [
    ({"e2_body_release_dynamic_pass": False}, "E2_BODY_RELEASE_NOT_DYNAMIC"),
    ({"visual_pose_accepted": False}, "VISUAL_POSE_NOT_ACCEPTED"),
    ({"tactile_ready_latched": False}, "TACTILE_READY_NOT_LATCHED"),
    ({"wrist_guard_safe": False}, "WRIST_MOMENT_GUARD_REJECTED"),
    ({"wrist_guard_fault_latched": True}, "WRIST_MOMENT_GUARD_REJECTED"),
    ({"e2_evidence_id": None}, "E2_EVIDENCE_ID_MISSING"),
])
def test_readiness_gates_fail_closed(kw, code):
    assert _eval(_ready(**kw))["rejection_code"] == code


def test_missing_current_target_authority_fails_closed():
    assert _eval(current_target_authority=None)["rejection_code"] == "CURRENT_MULTILAYER_TARGET_AUTHORITY_MISSING"


@pytest.mark.parametrize("authority", [
    {}, {**AUTH, "source_sha256": "bad"},
    {**AUTH, "coupling_nut_path": "/World/wrong"},
    {**AUTH, "visual_tactile_plan_ready": False},
])
def test_invalid_current_target_authority_fails_closed(authority):
    assert _eval(current_target_authority=authority)["rejection_code"] == "CURRENT_MULTILAYER_TARGET_AUTHORITY_INVALID"


def test_invalid_current_path_fails_closed():
    assert _eval(coupling_nut_path="relative")["rejection_code"] == "CURRENT_COUPLING_NUT_PATH_INVALID"


def test_strict_booleans():
    assert _eval(_ready(visual_pose_accepted=1))["rejection_code"] == "INVALID_READINESS_SNAPSHOT"


def test_no_execution_or_claim_on_any_path():
    for result in (_eval(), _eval(_ready(e2_body_release_dynamic_pass=False))):
        assert result["robot_motion_command_emitted"] is False
        assert result["finger_command_emitted"] is False
        assert result["control_authorized"] is False
        assert result["dynamic_nut_regrasp_pass_claimed"] is False


def test_truth_firewall_signature():
    names = set(inspect.signature(evaluate_nut_regrasp_gate).parameters)
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
        build_nut_regrasp_contract(root)


def test_strict_json():
    json.dumps(_eval(), sort_keys=True, allow_nan=False)
