from __future__ import annotations

from dataclasses import replace
import ast
import inspect
import math
from pathlib import Path

import pytest

from kcg_connector.d38999_multilayer_segmented_twist import (
    AUTHORIZED_MOMENT_COMPONENT_LIMIT_NM,
    E3_RESULT_PATH,
    SegmentedTwistReadiness,
    build_segmented_twist_request,
    current_readiness,
    derive_segmented_twist_schedule,
    evaluate_segmented_twist_gate,
    load_segmented_twist_contract,
)


ROOT = Path(__file__).resolve().parents[3]
MODULE = (
    ROOT
    / "src/kcg_connector/kcg_connector/"
    "d38999_multilayer_segmented_twist.py"
)


def _contract():
    return load_segmented_twist_contract(ROOT)


def _ready(contract, **overrides):
    values = {
        "e3_evidence_path": E3_RESULT_PATH,
        "e3_evidence_sha256": contract.current_e3_evidence_sha256,
        "e3_dynamic_nut_regrasp_passed": True,
        "wrist_guard_safe_to_continue": True,
        "wrist_fault_latched": False,
    }
    values.update(overrides)
    return SegmentedTwistReadiness(**values)


def test_contract_binds_exact_sources_and_current_parked_e3():
    contract = _contract()
    assert len(contract.source_rows) == 6
    assert all(len(digest) == 64 for _, digest in contract.source_rows)
    assert contract.current_e3_outcome == "PARKED"
    assert contract.current_e3_dynamic_nut_regrasp_passed is False


def test_authoritative_schedule_and_safety_numbers_are_unchanged():
    contract = _contract()
    assert contract.stroke_count == 3
    assert contract.rewind_count == 2
    assert math.degrees(contract.stroke_angle_rad) == pytest.approx(120.0)
    assert contract.tightening_direction == -1
    assert contract.moment_component_limit_nm == (
        AUTHORIZED_MOMENT_COMPONENT_LIMIT_NM
    )


def test_master_lead_replaces_but_does_not_overwrite_legacy_proxy_lead():
    contract = _contract()
    assert contract.master_lead_m_per_revolution == pytest.approx(0.00762)
    assert contract.legacy_proxy_lead_m_per_revolution == pytest.approx(0.004)
    assert contract.master_lead_m_per_revolution != (
        contract.legacy_proxy_lead_m_per_revolution
    )


def test_schedule_is_exactly_three_negative_q7_strokes_and_two_rewinds():
    schedule = derive_segmented_twist_schedule(
        _contract(), initial_q7_rad=0.650482794
    )
    actions = [stage["action"] for stage in schedule["stages"]]
    assert actions == [
        "GRIP", "TWIST", "RELEASE", "REWIND", "REGRIP",
        "TWIST", "RELEASE", "REWIND", "REGRIP", "TWIST",
    ]
    twists = [stage for stage in schedule["stages"] if stage["action"] == "TWIST"]
    rewinds = [stage for stage in schedule["stages"] if stage["action"] == "REWIND"]
    assert len(twists) == 3
    assert len(rewinds) == 2
    assert [math.degrees(stage["q7_delta_rad"]) for stage in twists] == pytest.approx(
        [-120.0, -120.0, -120.0]
    )
    assert [math.degrees(stage["nut_progress_rad"]) for stage in twists] == pytest.approx(
        [120.0, 120.0, 120.0]
    )
    assert all(stage["nut_progress_rad"] == 0.0 for stage in rewinds)


def test_schedule_uses_master_lead_and_defers_physical_follow_to_e5():
    schedule = derive_segmented_twist_schedule(
        _contract(), initial_q7_rad=0.650482794
    )
    assert schedule["target_nut_progress_rad"] == pytest.approx(math.tau)
    assert schedule["master_expected_axial_travel_m"] == pytest.approx(-0.00762)
    assert schedule["legacy_proxy_lead_used"] is False
    assert schedule["e5_axial_follow_required"] is True
    assert schedule["robot_commands_emitted"] == 0
    assert schedule["control_authorized"] is False


def test_current_parked_e3_produces_zero_output():
    contract = _contract()
    request = build_segmented_twist_request(
        contract,
        current_readiness(contract),
        initial_q7_rad=0.650482794,
    )
    assert request["request_ready"] is False
    assert request["rejection_code"] == "E3_NUT_REGRASP_NOT_DYNAMIC"
    assert request["schedule"] is None
    assert request["robot_commands_emitted"] == 0
    assert request["control_authorized"] is False
    assert request["dynamic_segmented_twist_pass_claimed"] is False


@pytest.mark.parametrize(
    "overrides,code",
    [
        ({"e3_evidence_path": "wrong"}, "E3_EVIDENCE_ID_MISMATCH"),
        ({"e3_evidence_sha256": "0" * 64}, "E3_EVIDENCE_ID_MISMATCH"),
        ({"e3_dynamic_nut_regrasp_passed": False}, "E3_NUT_REGRASP_NOT_DYNAMIC"),
        ({"wrist_fault_latched": True}, "WRIST_MOMENT_FAULT_LATCHED"),
        ({"wrist_guard_safe_to_continue": False}, "WRIST_MOMENT_GUARD_UNSAFE"),
    ],
)
def test_each_gate_fails_closed(overrides, code):
    contract = replace(_contract(), current_e3_dynamic_nut_regrasp_passed=True)
    assert evaluate_segmented_twist_gate(contract, _ready(contract, **overrides)) == code


def test_logical_ready_fixture_only_returns_nonexecuting_schedule_request():
    contract = replace(_contract(), current_e3_dynamic_nut_regrasp_passed=True)
    request = build_segmented_twist_request(
        contract, _ready(contract), initial_q7_rad=0.650482794
    )
    assert request["request_ready"] is True
    assert request["schedule"]["stroke_count"] == 3
    assert request["robot_commands_emitted"] == 0
    assert request["control_authorized"] is False
    assert request["dynamic_segmented_twist_pass_claimed"] is False


@pytest.mark.parametrize("initial", [float("nan"), float("inf"), True, -2.6, 2.6])
def test_schedule_rejects_invalid_or_out_of_window_q7(initial):
    with pytest.raises(ValueError):
        derive_segmented_twist_schedule(_contract(), initial_q7_rad=initial)


def test_public_gate_inputs_exclude_privileged_simulation_truth():
    names = set(inspect.signature(build_segmented_twist_request).parameters)
    assert names == {"contract", "readiness", "initial_q7_rad"}
    assert names.isdisjoint(
        {"object_pose", "contact_name", "contact_normal", "event_truth", "collider_path"}
    )


def test_module_is_cpu_only_and_contains_no_pose_write_or_dynamic_claim():
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    assert roots.isdisjoint({"isaacsim", "omni", "pxr", "rclpy", "torch"})
    source = MODULE.read_text(encoding="utf-8")
    assert "set_world_pose" not in source
    assert "set_local_pose" not in source

