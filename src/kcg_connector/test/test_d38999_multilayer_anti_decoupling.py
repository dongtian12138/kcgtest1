from __future__ import annotations

from dataclasses import replace
import ast
import inspect
import math
from pathlib import Path

import pytest

from kcg_connector.d38999_multilayer_anti_decoupling import (
    E5_RESULT_PATH,
    HIGH_DETAIL_FORWARD_PROXY_PEAK_NM,
    AntiDecouplingReadiness,
    build_anti_decoupling_request,
    current_readiness,
    derive_directional_resistance,
    derive_relative_periodic_profile,
    evaluate_anti_decoupling_gate,
    load_anti_decoupling_contract,
)


ROOT = Path(__file__).resolve().parents[3]
MODULE = (
    ROOT
    / "src/kcg_connector/kcg_connector/"
    "d38999_multilayer_anti_decoupling.py"
)


def _contract():
    return load_anti_decoupling_contract(ROOT)


def _ready(contract, **overrides):
    values = {
        "e5_evidence_path": E5_RESULT_PATH,
        "e5_evidence_sha256": contract.current_e5_evidence_sha256,
        "e5_dynamic_thread_follow_passed": True,
        "physical_detent_runtime_ready": True,
        "absolute_phase_origin_authorized": True,
    }
    values.update(overrides)
    return AntiDecouplingReadiness(**values)


def test_contract_binds_four_sources_and_preserves_provenance_boundary():
    contract = _contract()
    assert len(contract.source_rows) == 4
    assert contract.source_class == "equivalent_assumption"
    assert contract.hardware_curve_claimed is False
    assert contract.absolute_phase_origin_authorized is False
    assert contract.current_e5_outcome == "OFFLINE_PASS"
    assert contract.current_e5_dynamic_thread_follow_passed is False


def test_master_period_and_derived_profile_spans_are_exactly_reviewable():
    contract = _contract()
    assert contract.cycle_count_per_revolution == 36
    assert contract.follower_count == 3
    assert math.degrees(contract.pitch_rad) == pytest.approx(10.0)
    assert math.degrees(contract.forward_span_rad) == pytest.approx(0.9265469426540623)
    assert math.degrees(contract.reverse_span_rad) == pytest.approx(0.09117940164241296)
    assert math.degrees(contract.dwell_span_rad) == pytest.approx(8.982273655703526)


def test_relative_profile_is_periodic_and_has_no_absolute_phase_claim():
    contract = _contract()
    first = derive_relative_periodic_profile(contract, 0.123)
    wrapped = derive_relative_periodic_profile(contract, 0.123 + 19 * contract.pitch_rad)
    assert first["relative_progress_rad"] == pytest.approx(
        wrapped["relative_progress_rad"]
    )
    assert first["profile_branch"] == wrapped["profile_branch"]
    assert first["branch_fraction"] == pytest.approx(wrapped["branch_fraction"])
    assert first["radial_deflection_m"] == pytest.approx(
        wrapped["radial_deflection_m"]
    )
    assert first["absolute_phase_used"] is False
    assert first["hardware_curve_claimed"] is False


def test_profile_branches_preserve_preload_rise_and_continuity():
    contract = _contract()
    dwell = derive_relative_periodic_profile(contract, 0.5 * contract.dwell_span_rad)
    ascent = derive_relative_periodic_profile(
        contract, contract.dwell_span_rad + 0.5 * contract.forward_span_rad
    )
    peak = derive_relative_periodic_profile(
        contract,
        contract.dwell_span_rad + contract.forward_span_rad,
    )
    assert dwell["profile_branch"] == "base_dwell"
    assert dwell["radial_deflection_m"] == pytest.approx(1.0e-6)
    assert ascent["profile_branch"] == "shallow_ascent"
    assert ascent["radial_deflection_m"] == pytest.approx(26.0e-6)
    assert peak["profile_branch"] == "steep_reverse_drop"
    assert peak["radial_deflection_m"] == pytest.approx(51.0e-6)


def test_forward_resistance_is_below_limit_and_never_clipped():
    contract = _contract()
    initial = derive_directional_resistance(
        contract,
        direction="positive_coupling",
        radial_deflection_m=contract.nominal_radial_preload_m,
    )
    peak = derive_directional_resistance(
        contract,
        direction="positive_coupling",
        radial_deflection_m=contract.maximum_radial_deflection_m,
    )
    assert initial["resistance_magnitude_nm"] == pytest.approx(0.0010203264599393618)
    assert peak["resistance_magnitude_nm"] == pytest.approx(0.05203664945690745)
    assert peak["within_authorized_moment_component"] is True
    assert peak["clipped_to_limit"] is False


def test_reverse_resistance_over_limit_is_reported_not_clipped():
    contract = _contract()
    peak = derive_directional_resistance(
        contract,
        direction="reverse_decoupling",
        radial_deflection_m=contract.maximum_radial_deflection_m,
    )
    assert peak["resistance_magnitude_nm"] == pytest.approx(0.5287860809763352)
    assert peak["within_authorized_moment_component"] is False
    assert peak["clipped_to_limit"] is False


def test_high_detail_peak_is_cross_evidence_not_hardware_or_master_override():
    contract = _contract()
    assert HIGH_DETAIL_FORWARD_PROXY_PEAK_NM == pytest.approx(0.060021022609)
    assert contract.maximum_forward_resistance_nm == pytest.approx(0.05203664945690745)
    assert contract.hardware_curve_claimed is False


def test_current_e5_static_only_state_returns_zero_output():
    contract = _contract()
    request = build_anti_decoupling_request(
        contract,
        current_readiness(contract),
        relative_progress_rad=0.0,
        direction="positive_coupling",
    )
    assert request["request_ready"] is False
    assert request["rejection_code"] == "E5_THREAD_AXIAL_FOLLOW_NOT_DYNAMIC"
    assert request["relative_profile"] is None
    assert request["resistance"] is None
    assert request["robot_commands_emitted"] == 0


@pytest.mark.parametrize(
    "overrides,code",
    [
        ({"e5_evidence_path": "wrong"}, "E5_EVIDENCE_ID_MISMATCH"),
        ({"e5_evidence_sha256": "0" * 64}, "E5_EVIDENCE_ID_MISMATCH"),
        ({"e5_dynamic_thread_follow_passed": False}, "E5_THREAD_AXIAL_FOLLOW_NOT_DYNAMIC"),
        ({"physical_detent_runtime_ready": False}, "PHYSICAL_DETENT_RUNTIME_NOT_READY"),
        ({"absolute_phase_origin_authorized": False}, "DETENT_ABSOLUTE_PHASE_ORIGIN_UNAUTHORIZED"),
    ],
)
def test_each_readiness_gate_fails_closed(overrides, code):
    contract = replace(
        _contract(),
        current_e5_dynamic_thread_follow_passed=True,
        absolute_phase_origin_authorized=True,
    )
    assert evaluate_anti_decoupling_gate(contract, _ready(contract, **overrides)) == code


def test_logical_e5_ready_still_rejects_missing_master_absolute_phase():
    contract = replace(_contract(), current_e5_dynamic_thread_follow_passed=True)
    request = build_anti_decoupling_request(
        contract,
        _ready(contract),
        relative_progress_rad=0.0,
        direction="positive_coupling",
    )
    assert request["rejection_code"] == "DETENT_ABSOLUTE_PHASE_ORIGIN_UNAUTHORIZED"
    assert request["physical_model_internal_constraint_requested"] is False
    assert request["control_authorized"] is False


def test_hypothetical_reverse_peak_trips_safety_instead_of_commanding():
    contract = replace(
        _contract(),
        current_e5_dynamic_thread_follow_passed=True,
        absolute_phase_origin_authorized=True,
    )
    progress = contract.dwell_span_rad + contract.forward_span_rad
    request = build_anti_decoupling_request(
        contract,
        _ready(contract),
        relative_progress_rad=progress,
        direction="reverse_decoupling",
    )
    assert request["rejection_code"] == "DETENT_RESISTANCE_EXCEEDS_AUTHORIZED_MOMENT"
    assert request["resistance"]["resistance_magnitude_nm"] > 0.30
    assert request["force_or_moment_command_requested"] is False
    assert request["robot_commands_emitted"] == 0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), "0"])
def test_nonfinite_or_nonnumeric_relative_progress_is_rejected(bad):
    with pytest.raises(ValueError):
        derive_relative_periodic_profile(_contract(), bad)


@pytest.mark.parametrize("bad", [0.0, 0.000052, float("nan"), "1e-6"])
def test_out_of_contract_or_invalid_deflection_is_rejected(bad):
    with pytest.raises(ValueError):
        derive_directional_resistance(
            _contract(), direction="positive_coupling", radial_deflection_m=bad
        )


def test_public_request_inputs_exclude_privileged_simulation_truth():
    names = set(inspect.signature(build_anti_decoupling_request).parameters)
    assert names == {"contract", "readiness", "relative_progress_rad", "direction"}
    assert names.isdisjoint(
        {"object_pose", "contact_name", "contact_normal", "event_truth", "collider_path"}
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
