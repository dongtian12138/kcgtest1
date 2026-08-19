from __future__ import annotations

from dataclasses import replace
import ast
import copy
import inspect
from pathlib import Path

import pytest

from kcg_connector.d38999_multilayer_arrival_decision import (
    E7_RESULT_PATH,
    ArrivalDecisionReadiness,
    audit_posthoc_arrival_report,
    build_arrival_decision_request,
    current_readiness,
    evaluate_arrival_readiness,
    load_arrival_decision_contract,
)


ROOT = Path(__file__).resolve().parents[3]
MODULE = (
    ROOT
    / "src/kcg_connector/kcg_connector/"
    "d38999_multilayer_arrival_decision.py"
)


def _contract():
    return load_arrival_decision_contract(ROOT)


def _ready(contract, **overrides):
    values = {
        "e7_evidence_path": E7_RESULT_PATH,
        "e7_evidence_sha256": contract.current_e7_evidence_sha256,
        "e7_dynamic_metal_stop_passed": True,
        "nominal_report_available": True,
        "posthoc_audit_only": True,
        "report_truth_routed_to_controller": False,
    }
    values.update(overrides)
    return ArrivalDecisionReadiness(**values)


def _report(contract):
    event_first = {}
    for step, (name, position) in enumerate(
        contract.nominal_event_positions_m, start=1
    ):
        event_first[name] = {
            "step": step,
            "time_s": step / 240.0,
            "datum_B_separation_m": position,
            "source": (
                "physx_continuous_real_collision"
                if name == "shell_to_shell_metal_bottoming"
                else "model_internal_continuous_effect_activation"
            ),
        }
    return {
        "schema_version": "kcg_d38999_multilayer_nominal_bench_v1",
        "bench_id": "TASK-R12-MULTILAYER-005",
        "mode": "formal_multilayer_nominal",
        "status": "PASS",
        "passed": True,
        "event_first": event_first,
        "observed_event_order": list(contract.event_order),
        "maximum_driver_force_component_n": 7.5,
        "maximum_driver_torque_component_nm": 0.29,
        "maximum_fixed_receptacle_translation_drift_m": 4.0e-6,
        "hard_penetrations_over_limit": [],
        "solver_error_count": 0,
        "object_pose_write_after_physics_start_count": 0,
        "contact_pairs": [
            {
                "collider_paths": [
                    contract.fixed_stop_path,
                    contract.plug_stop_path,
                ],
                "active_step_count": 2,
            }
        ],
        "control_input_policy": {
            "predeclared_time_schedule": True,
            "rigid_body_state_and_velocity_only": True,
            "contact_object_name_used": False,
            "contact_normal_used": False,
            "contact_manifold_used": False,
            "event_truth_used": False,
            "posthoc_contact_truth_for_scoring_only": True,
        },
        "simulation_started": True,
        "formal_p1_pass_claimed": False,
        "formal_r12_generated": False,
        "hardware_authorized": False,
    }


def test_contract_binds_four_sources_seven_events_and_frozen_limits():
    contract = _contract()
    assert len(contract.source_rows) == 4
    assert len(contract.event_order) == 7
    assert contract.event_order[-1] == "shell_to_shell_metal_bottoming"
    assert dict(contract.nominal_event_positions_m)[contract.event_order[-1]] == pytest.approx(0.01505)
    assert contract.event_position_tolerance_m == pytest.approx(50e-6)
    assert contract.force_component_limit_n_per_driven_body == pytest.approx(8.0)
    assert contract.torque_component_limit_nm == pytest.approx(0.30)
    assert contract.maximum_fixed_receptacle_translation_drift_m == pytest.approx(5e-6)
    assert contract.maximum_noncompliant_hard_penetration_m == pytest.approx(50e-6)
    assert contract.current_e7_dynamic_metal_stop_passed is False


def test_force_limit_semantics_are_per_driven_body_component_not_wrist_total():
    contract = _contract()
    assert contract.force_limit_semantics == (
        "each_driven_body_each_component_not_whole_robot_wrist_total"
    )


def test_current_e7_static_only_state_returns_zero_output():
    contract = _contract()
    request = build_arrival_decision_request(contract, current_readiness(contract))
    assert request["request_ready"] is False
    assert request["rejection_code"] == "E7_METAL_STOP_NOT_DYNAMIC"
    assert request["posthoc_audit_requested"] is False
    assert request["assembly_arrival_claimed"] is False
    assert request["robot_commands_emitted"] == 0


@pytest.mark.parametrize(
    "overrides,code",
    [
        ({"e7_evidence_path": "wrong"}, "E7_EVIDENCE_ID_MISMATCH"),
        ({"e7_evidence_sha256": "0" * 64}, "E7_EVIDENCE_ID_MISMATCH"),
        ({"e7_dynamic_metal_stop_passed": False}, "E7_METAL_STOP_NOT_DYNAMIC"),
        ({"nominal_report_available": False}, "NOMINAL_DYNAMIC_REPORT_MISSING"),
        ({"posthoc_audit_only": False}, "ARRIVAL_DECISION_MUST_BE_POSTHOC_ONLY"),
        ({"report_truth_routed_to_controller": True}, "REPORT_TRUTH_TO_CONTROLLER_FORBIDDEN"),
    ],
)
def test_each_readiness_gate_fails_closed(overrides, code):
    contract = replace(_contract(), current_e7_dynamic_metal_stop_passed=True)
    assert evaluate_arrival_readiness(contract, _ready(contract, **overrides)) == code


def test_self_reported_pass_and_structurally_valid_offline_fixture_are_rejected():
    contract = _contract()
    audit = audit_posthoc_arrival_report(
        contract, _report(contract), evidence_is_offline_fixture=True
    )
    assert audit["reported_status"] == "PASS"
    assert audit["reported_passed"] is True
    assert audit["self_report_trusted_without_recomputation"] is False
    assert audit["accepted"] is False
    assert audit["assembly_arrival_proven"] is False
    assert audit["rejection_codes"] == ["OFFLINE_FIXTURE_NOT_DYNAMIC_EVIDENCE"]


@pytest.mark.parametrize(
    "mutation,code",
    [
        ("bench", "BENCH_ID_MISMATCH"),
        ("mode", "REPORT_MODE_MISMATCH"),
        ("simulation", "SIMULATION_NOT_STARTED"),
        ("missing_event", "EVENT_MISSING:seal_compression"),
        ("order", "EVENT_ORDER_INVALID"),
        ("position", "EVENT_POSITION_OUT_OF_TOLERANCE:pin_barrier_seal_contact"),
        ("precedence", "PRECEDENCE_VIOLATION:seal_compression->shell_to_shell_metal_bottoming"),
        ("metal_source", "METAL_STOP_SOURCE_NOT_PHYSICAL_COLLISION"),
        ("metal_pair", "EXACT_PHYSICAL_METAL_STOP_PAIR_NOT_PROVEN"),
        ("force", "DRIVER_FORCE_COMPONENT_LIMIT_EXCEEDED"),
        ("torque", "DRIVER_TORQUE_COMPONENT_LIMIT_EXCEEDED"),
        ("drift", "FIXED_RECEPTACLE_DRIFT_LIMIT_EXCEEDED"),
        ("penetration", "HARD_PENETRATION_GATE_FAILED"),
        ("solver", "SOLVER_ERROR_PRESENT"),
        ("pose_write", "POST_START_POSE_WRITE_PRESENT"),
        ("truth", "CONTROL_TRUTH_FIREWALL_FAILED"),
        ("posthoc", "POSTHOC_ONLY_BOUNDARY_FAILED"),
        ("p1_claim", "FORMAL_P1_CLAIM_FORBIDDEN_IN_E8"),
        ("r12_claim", "FORMAL_R12_CLAIM_FORBIDDEN_IN_E8"),
        ("hardware", "HARDWARE_AUTHORIZATION_FORBIDDEN"),
    ],
)
def test_each_missing_or_unsafe_dynamic_fact_fails_independent_audit(mutation, code):
    contract = _contract()
    report = copy.deepcopy(_report(contract))
    if mutation == "bench":
        report["bench_id"] = "wrong"
    elif mutation == "mode":
        report["mode"] = "initialize_only"
    elif mutation == "simulation":
        report["simulation_started"] = False
    elif mutation == "missing_event":
        del report["event_first"]["seal_compression"]
    elif mutation == "order":
        report["observed_event_order"][0:2] = reversed(report["observed_event_order"][0:2])
    elif mutation == "position":
        report["event_first"]["pin_barrier_seal_contact"]["datum_B_separation_m"] += 51e-6
    elif mutation == "precedence":
        report["event_first"]["shell_to_shell_metal_bottoming"]["step"] = 6
    elif mutation == "metal_source":
        report["event_first"]["shell_to_shell_metal_bottoming"]["source"] = "position_boolean"
    elif mutation == "metal_pair":
        report["contact_pairs"] = []
    elif mutation == "force":
        report["maximum_driver_force_component_n"] = 8.000001
    elif mutation == "torque":
        report["maximum_driver_torque_component_nm"] = 0.300001
    elif mutation == "drift":
        report["maximum_fixed_receptacle_translation_drift_m"] = 5.000001e-6
    elif mutation == "penetration":
        report["hard_penetrations_over_limit"] = [{"depth_m": 51e-6}]
    elif mutation == "solver":
        report["solver_error_count"] = 1
    elif mutation == "pose_write":
        report["object_pose_write_after_physics_start_count"] = 1
    elif mutation == "truth":
        report["control_input_policy"]["event_truth_used"] = True
    elif mutation == "posthoc":
        report["control_input_policy"]["posthoc_contact_truth_for_scoring_only"] = False
    elif mutation == "p1_claim":
        report["formal_p1_pass_claimed"] = True
    elif mutation == "r12_claim":
        report["formal_r12_generated"] = True
    else:
        report["hardware_authorized"] = True
    audit = audit_posthoc_arrival_report(
        contract, report, evidence_is_offline_fixture=False
    )
    assert audit["accepted"] is False
    assert code in audit["rejection_codes"]
    assert audit["control_authorized"] is False


@pytest.mark.parametrize(
    "field,value,code",
    [
        ("maximum_driver_force_component_n", float("nan"), "INVALID_NUMERIC_FIELD:maximum_driver_force_component_n"),
        ("maximum_driver_torque_component_nm", "0.1", "INVALID_NUMERIC_FIELD:maximum_driver_torque_component_nm"),
        ("maximum_fixed_receptacle_translation_drift_m", float("inf"), "INVALID_NUMERIC_FIELD:maximum_fixed_receptacle_translation_drift_m"),
        ("solver_error_count", True, "SOLVER_ERROR_COUNT_INVALID"),
        ("object_pose_write_after_physics_start_count", -1, "POSE_WRITE_COUNT_INVALID"),
    ],
)
def test_malformed_report_numerics_fail_closed(field, value, code):
    contract = _contract()
    report = _report(contract)
    report[field] = value
    audit = audit_posthoc_arrival_report(
        contract, report, evidence_is_offline_fixture=False
    )
    assert code in audit["rejection_codes"]


def test_logical_dynamic_fixture_can_only_authorize_posthoc_audit_request():
    contract = replace(_contract(), current_e7_dynamic_metal_stop_passed=True)
    request = build_arrival_decision_request(contract, _ready(contract))
    assert request["request_ready"] is True
    assert request["posthoc_audit_requested"] is True
    assert request["event_or_contact_truth_routed_to_controller"] is False
    assert request["position_boolean_requested"] is False
    assert request["assembly_arrival_claimed"] is False
    assert request["control_authorized"] is False


def test_public_request_inputs_exclude_report_and_privileged_truth():
    names = set(inspect.signature(build_arrival_decision_request).parameters)
    assert names == {"contract", "readiness"}
    assert names.isdisjoint(
        {
            "report",
            "object_pose",
            "separation",
            "contact_name",
            "contact_normal",
            "event_truth",
        }
    )


def test_module_is_cpu_only_and_has_no_pose_or_command_api():
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
    assert "apply_force" not in source
    assert "apply_torque" not in source
