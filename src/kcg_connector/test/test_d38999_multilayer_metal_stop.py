from __future__ import annotations

from dataclasses import replace
import ast
import inspect
from pathlib import Path

import pytest

from kcg_connector.d38999_multilayer_metal_stop import (
    E6_RESULT_PATH,
    MetalStopReadiness,
    PosthocMetalStopEvidence,
    audit_posthoc_metal_stop_evidence,
    build_metal_stop_request,
    current_readiness,
    evaluate_metal_stop_gate,
    load_metal_stop_contract,
)


ROOT = Path(__file__).resolve().parents[3]
MODULE = (
    ROOT
    / "src/kcg_connector/kcg_connector/"
    "d38999_multilayer_metal_stop.py"
)


def _contract():
    return load_metal_stop_contract(ROOT)


def _ready(contract, **overrides):
    values = {
        "e6_evidence_path": E6_RESULT_PATH,
        "e6_evidence_sha256": contract.current_e6_evidence_sha256,
        "e6_dynamic_anti_decoupling_passed": True,
        "physical_collision_runtime_ready": True,
        "posthoc_contact_audit_channel_ready": True,
        "controller_consumes_contact_truth": False,
        "controller_uses_pose_boolean": False,
    }
    values.update(overrides)
    return MetalStopReadiness(**values)


def _evidence(contract, **overrides):
    values = {
        "evidence_kind": "posthoc_physx_contact_audit",
        "run_id": "offline-fixture-e7",
        "fixed_stop_path": contract.fixed_stop_path,
        "plug_stop_path": contract.plug_stop_path,
        "physical_contact_active": True,
        "normal_impulse_n_s": 1.0e-4,
        "measured_separation_m": contract.nominal_bottoming_separation_m,
        "maximum_hard_penetration_m": 1.0e-6,
        "solver_error_count": 0,
        "post_run_pose_write_count": 0,
        "controller_consumed_contact_truth": False,
        "controller_used_pose_boolean": False,
        "offline_fixture_only": True,
    }
    values.update(overrides)
    return PosthocMetalStopEvidence(**values)


def test_contract_binds_five_frozen_sources_and_master_boundaries():
    contract = _contract()
    assert len(contract.source_rows) == 5
    assert contract.source_class == "equivalent_assumption"
    assert contract.definition == (
        "physical_receptacle_engaging_shell_to_plug_internal_engaging_shell_contact"
    )
    assert contract.determined_by_physical_collision_not_pose_or_boolean is True
    assert contract.continuous_real_collision_required is True
    assert contract.calibration_range_is_public_bottoming_depth_claim is False


def test_stop_position_geometry_and_event_are_exactly_master_derived():
    contract = _contract()
    assert contract.nominal_bottoming_separation_m == pytest.approx(0.01505)
    assert contract.calibration_range_m == pytest.approx((0.01501, 0.01509))
    assert contract.fixed_cap_radius_m == pytest.approx(0.01695)
    assert contract.fixed_cap_axial_thickness_m == pytest.approx(0.0003)
    assert contract.plug_stop_distribution_radius_m == pytest.approx(0.016)
    assert contract.event_name == "shell_to_shell_metal_bottoming"
    assert contract.event_ordinal == 7
    assert contract.event_position_tolerance_m == pytest.approx(50e-6)


def test_paths_and_collision_roles_are_specific_and_continuous():
    contract = _contract()
    assert contract.fixed_stop_path.endswith(
        "/FixedReceptacle/MatingShell/MetalStop"
    )
    assert contract.plug_stop_path.endswith(
        "/LoosePlug/BodyAssembly/InternalMatingShell/MetalStop"
    )
    assert contract.fixed_collision_role == "continuous_real_metal_stop_fixed"
    assert contract.plug_collision_role == "continuous_real_metal_stop_plug"
    assert contract.fixed_stop_path != contract.plug_stop_path


def test_current_e6_static_only_state_returns_zero_output():
    contract = _contract()
    request = build_metal_stop_request(contract, current_readiness(contract))
    assert request["request_ready"] is False
    assert request["rejection_code"] == "E6_ANTI_DECOUPLING_NOT_DYNAMIC"
    assert request["fixed_stop_path"] is None
    assert request["plug_stop_path"] is None
    assert request["physical_collision_pair_requested"] is False
    assert request["robot_commands_emitted"] == 0


@pytest.mark.parametrize(
    "overrides,code",
    [
        ({"e6_evidence_path": "wrong"}, "E6_EVIDENCE_ID_MISMATCH"),
        ({"e6_evidence_sha256": "0" * 64}, "E6_EVIDENCE_ID_MISMATCH"),
        ({"e6_dynamic_anti_decoupling_passed": False}, "E6_ANTI_DECOUPLING_NOT_DYNAMIC"),
        ({"physical_collision_runtime_ready": False}, "PHYSICAL_METAL_STOP_RUNTIME_NOT_READY"),
        ({"posthoc_contact_audit_channel_ready": False}, "POSTHOC_CONTACT_AUDIT_CHANNEL_NOT_READY"),
        ({"controller_consumes_contact_truth": True}, "CONTROLLER_CONTACT_TRUTH_FORBIDDEN"),
        ({"controller_uses_pose_boolean": True}, "POSE_BOOLEAN_METAL_STOP_FORBIDDEN"),
    ],
)
def test_each_runtime_gate_fails_closed(overrides, code):
    contract = replace(_contract(), current_e6_dynamic_anti_decoupling_passed=True)
    assert evaluate_metal_stop_gate(contract, _ready(contract, **overrides)) == code


def test_logical_ready_fixture_requests_collision_but_never_claims_event():
    contract = replace(_contract(), current_e6_dynamic_anti_decoupling_passed=True)
    request = build_metal_stop_request(contract, _ready(contract))
    assert request["request_ready"] is True
    assert request["physical_collision_pair_requested"] is True
    assert request["nominal_separation_is_audit_metadata_only"] is True
    assert request["position_boolean_requested"] is False
    assert request["contact_truth_routed_to_controller"] is False
    assert request["metal_stop_event_claimed"] is False
    assert request["software_pose_write_requested"] is False
    assert request["force_or_moment_command_requested"] is False
    assert request["control_authorized"] is False


def test_offline_posthoc_fixture_can_never_prove_dynamic_contact():
    contract = _contract()
    audit = audit_posthoc_metal_stop_evidence(contract, _evidence(contract))
    assert audit["accepted"] is False
    assert audit["metal_stop_event_proven"] is False
    assert "OFFLINE_FIXTURE_NOT_DYNAMIC_EVIDENCE" in audit["rejection_codes"]
    assert audit["controller_input_allowed"] is False


@pytest.mark.parametrize(
    "overrides,code",
    [
        ({"fixed_stop_path": "/wrong"}, "METAL_STOP_PAIR_ID_MISMATCH"),
        ({"physical_contact_active": False}, "PHYSICAL_CONTACT_NOT_PROVEN"),
        ({"normal_impulse_n_s": 0.0}, "PHYSICAL_CONTACT_NOT_PROVEN"),
        ({"measured_separation_m": 0.0149}, "BOTTOMING_SEPARATION_OUTSIDE_CALIBRATION_RANGE"),
        ({"maximum_hard_penetration_m": 51e-6}, "HARD_PENETRATION_LIMIT_EXCEEDED"),
        ({"solver_error_count": 1}, "SOLVER_ERROR_PRESENT"),
        ({"post_run_pose_write_count": 1}, "POST_RUN_POSE_WRITE_PRESENT"),
        ({"controller_consumed_contact_truth": True}, "CONTROLLER_CONTACT_TRUTH_FORBIDDEN"),
        ({"controller_used_pose_boolean": True}, "POSE_BOOLEAN_METAL_STOP_FORBIDDEN"),
    ],
)
def test_posthoc_evidence_rejects_each_unsafe_or_unproven_case(overrides, code):
    contract = _contract()
    audit = audit_posthoc_metal_stop_evidence(contract, _evidence(contract, **overrides))
    assert audit["accepted"] is False
    assert code in audit["rejection_codes"]
    assert audit["control_authorized"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("normal_impulse_n_s", float("nan")),
        ("measured_separation_m", float("inf")),
        ("maximum_hard_penetration_m", "0"),
        ("solver_error_count", -1),
        ("post_run_pose_write_count", True),
    ],
)
def test_malformed_posthoc_numeric_fields_are_rejected(field, value):
    contract = _contract()
    with pytest.raises(ValueError):
        audit_posthoc_metal_stop_evidence(
            contract, _evidence(contract, **{field: value})
        )


def test_public_request_inputs_exclude_privileged_simulation_truth():
    names = set(inspect.signature(build_metal_stop_request).parameters)
    assert names == {"contract", "readiness"}
    assert names.isdisjoint(
        {
            "object_pose",
            "separation",
            "contact_name",
            "contact_normal",
            "event_truth",
            "collider_path",
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
