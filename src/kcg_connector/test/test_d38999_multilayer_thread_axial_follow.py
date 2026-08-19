from __future__ import annotations

from dataclasses import replace
import ast
import copy
import inspect
import math
from pathlib import Path

import pytest

from kcg_connector.d38999_multilayer_segmented_twist import (
    derive_segmented_twist_schedule,
    load_segmented_twist_contract,
)
from kcg_connector.d38999_multilayer_thread_axial_follow import (
    E4_RESULT_PATH,
    ThreadAxialFollowReadiness,
    build_thread_axial_follow_request,
    current_readiness,
    derive_thread_axial_follow,
    evaluate_thread_axial_follow_gate,
    load_thread_axial_follow_contract,
)


ROOT = Path(__file__).resolve().parents[3]
MODULE = (
    ROOT
    / "src/kcg_connector/kcg_connector/"
    "d38999_multilayer_thread_axial_follow.py"
)


def _contract():
    return load_thread_axial_follow_contract(ROOT)


def _schedule():
    return derive_segmented_twist_schedule(
        load_segmented_twist_contract(ROOT), initial_q7_rad=0.650482794
    )


def _ready(contract, **overrides):
    values = {
        "e4_evidence_path": E4_RESULT_PATH,
        "e4_evidence_sha256": contract.current_e4_evidence_sha256,
        "e4_dynamic_segmented_twist_passed": True,
        "physical_constraint_runtime_ready": True,
    }
    values.update(overrides)
    return ThreadAxialFollowReadiness(**values)


def test_contract_binds_master_relation_and_current_e4_evidence():
    contract = _contract()
    assert len(contract.source_rows) == 4
    assert contract.relation == "axial_advance_m=-nut_rotation_rad*0.00762/(2*pi)"
    assert contract.current_e4_outcome == "OFFLINE_PASS"
    assert contract.current_e4_dynamic_segmented_twist_passed is False


def test_each_stroke_advances_2p54_mm_and_each_rewind_advances_zero():
    relation = derive_thread_axial_follow(_contract(), _schedule())
    strokes = [row for row in relation["stages"] if row["action"] == "TWIST"]
    rewinds = [row for row in relation["stages"] if row["action"] == "REWIND"]
    assert len(strokes) == 3
    assert len(rewinds) == 2
    assert [row["insertion_advance_m"] for row in strokes] == pytest.approx(
        [0.00254, 0.00254, 0.00254]
    )
    assert [row["axial_coordinate_delta_m"] for row in strokes] == pytest.approx(
        [-0.00254, -0.00254, -0.00254]
    )
    assert all(row["insertion_advance_m"] == 0.0 for row in rewinds)
    assert all(row["axial_coordinate_delta_m"] == 0.0 for row in rewinds)


def test_one_revolution_equals_exactly_one_master_lead():
    relation = derive_thread_axial_follow(_contract(), _schedule())
    assert relation["total_insertion_advance_m"] == pytest.approx(0.00762)
    assert relation["total_axial_coordinate_delta_m"] == pytest.approx(-0.00762)
    assert relation["legacy_0p003_m_lead_used"] is False
    assert relation["legacy_0p004_m_lead_used"] is False


def test_relation_is_constraint_only_and_never_writes_pose_or_force():
    relation = derive_thread_axial_follow(_contract(), _schedule())
    assert relation["physical_model_internal_constraint_only"] is True
    assert relation["software_pose_write_requested"] is False
    assert relation["force_command_requested"] is False
    assert relation["robot_commands_emitted"] == 0
    assert relation["control_authorized"] is False
    assert relation["dynamic_thread_follow_pass_claimed"] is False


def test_current_e4_static_only_state_returns_zero_output():
    contract = _contract()
    request = build_thread_axial_follow_request(
        contract, current_readiness(contract), initial_q7_rad=0.650482794
    )
    assert request["request_ready"] is False
    assert request["rejection_code"] == "E4_SEGMENTED_TWIST_NOT_DYNAMIC"
    assert request["relation"] is None
    assert request["software_pose_write_requested"] is False
    assert request["robot_commands_emitted"] == 0


@pytest.mark.parametrize(
    "overrides,code",
    [
        ({"e4_evidence_path": "wrong"}, "E4_EVIDENCE_ID_MISMATCH"),
        ({"e4_evidence_sha256": "0" * 64}, "E4_EVIDENCE_ID_MISMATCH"),
        ({"e4_dynamic_segmented_twist_passed": False}, "E4_SEGMENTED_TWIST_NOT_DYNAMIC"),
        ({"physical_constraint_runtime_ready": False}, "PHYSICAL_THREAD_CONSTRAINT_RUNTIME_NOT_READY"),
    ],
)
def test_each_readiness_gate_fails_closed(overrides, code):
    contract = replace(_contract(), current_e4_dynamic_segmented_twist_passed=True)
    assert evaluate_thread_axial_follow_gate(contract, _ready(contract, **overrides)) == code


def test_logical_ready_fixture_only_returns_nonexecuting_constraint_request():
    contract = replace(_contract(), current_e4_dynamic_segmented_twist_passed=True)
    request = build_thread_axial_follow_request(
        contract, _ready(contract), initial_q7_rad=0.650482794
    )
    assert request["request_ready"] is True
    assert request["relation"]["total_insertion_advance_m"] == pytest.approx(0.00762)
    assert request["software_pose_write_requested"] is False
    assert request["force_command_requested"] is False
    assert request["robot_commands_emitted"] == 0
    assert request["control_authorized"] is False


@pytest.mark.parametrize("mutation", ["action", "twist_progress", "rewind_progress", "nan", "legacy"])
def test_malformed_or_proxy_schedule_is_rejected(mutation):
    schedule = copy.deepcopy(_schedule())
    if mutation == "action":
        schedule["stages"][1]["action"] = "REWIND"
    elif mutation == "twist_progress":
        schedule["stages"][1]["nut_progress_rad"] = math.radians(119.0)
    elif mutation == "rewind_progress":
        schedule["stages"][3]["nut_progress_rad"] = 1e-6
    elif mutation == "nan":
        schedule["stages"][1]["nut_progress_rad"] = float("nan")
    else:
        schedule["legacy_proxy_lead_used"] = True
    with pytest.raises(ValueError):
        derive_thread_axial_follow(_contract(), schedule)


def test_public_request_inputs_exclude_privileged_simulation_truth():
    names = set(inspect.signature(build_thread_axial_follow_request).parameters)
    assert names == {"contract", "readiness", "initial_q7_rad"}
    assert names.isdisjoint(
        {"object_pose", "contact_name", "contact_normal", "event_truth", "collider_path"}
    )


def test_module_is_cpu_only_and_has_no_pose_write_api():
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

