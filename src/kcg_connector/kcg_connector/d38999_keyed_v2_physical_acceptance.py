"""Pure-CPU validator for the D38999 A2-A5 physical acceptance matrix.

The matrix freezes thresholds only.  It neither launches Isaac Sim nor turns
an unexecuted bench into a passing result, and it computes no file fingerprint.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from kcg_connector.d38999_keyed_v2_frozen_contract_snapshot import (
    FROZEN_ACCEPTANCE_BENCHES,
    FROZEN_ACCEPTANCE_IDENTITY_AND_EVIDENCE,
    FROZEN_ACCEPTANCE_PHASE_RELEASE,
    FROZEN_ACCEPTANCE_SHARED_NUMERIC_PROFILE,
)

from kcg_connector.d38999_keyed_v2_physical_model_contract import (
    NOMINAL_EVENT_B_SEPARATION_MM,
    REQUIRED_BENCH_IDS,
    REQUIRED_SEQUENCE_PRECEDENCE,
    SUCCESSOR_ASSET_NAME,
    SUCCESSOR_REVISION,
    load_physical_model_contract,
)


SCHEMA_VERSION = "kcg_d38999_keyed_v2_physical_acceptance_v1"
DEFAULT_ACCEPTANCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "config/d38999_keyed_v2_physical_acceptance_v1.yaml"
)
MODEL_CONTRACT_REPOSITORY_PATH = (
    "src/kcg_connector/config/d38999_keyed_v2_physical_model_contract_v1.yaml"
)
NOMINAL_R7_EVENT_ORDER = (
    "five_key_polarization",
    "three_start_thread_entry",
    "spring_finger_engagement",
    "first_pin_socket_spring_touch",
    "pin_barrier_seal_contact",
    "seal_compression",
    "shell_to_shell_metal_bottoming",
)
ALLOWED_CONTROLLER_INPUTS = (
    "rgbd",
    "joint_state",
    "tcp_forward_kinematics",
    "wrist_6d_wrench",
    "controller_history",
)
EXPECTED_BENCH_NAMES = {
    "P1": "correct_n_key_and_three_start_entry",
    "P2": "full_yaw_wrong_key_and_boundary_sweep",
    "P3": "xy_tilt_capture_and_blocking_map",
    "P4": "passive_sequence_compliance_and_energy",
    "P5": "all_61_same_label_contacts_with_bad_contact_positive_control",
    "P6": "three_start_theta_z_lead",
    "P7": "physical_metal_bottoming_and_hold",
    "P8": "friction_torque_and_energy",
    "P9": "self_lock_and_component_disengagement",
    "P10": "receptacle_fixture_world_load_path",
    "P11": "robot_hand_self_collision_and_positive_control",
    "P12": "wrist_ft_camera_frames_and_timing",
    "P13": "numerical_stability_three_fresh_processes",
    "P14": "minimal_sensor_only_robot_in_loop",
}
EXPECTED_BENCH_MODES = {
    **{f"P{index}": "connector_component_only" for index in range(1, 8)},
    "P8": "component_then_robot_in_loop",
    "P9": "component_then_robot_in_loop",
    "P10": "fixture_component_only",
    "P11": "robot_in_loop",
    "P12": "robot_sensor_bench",
    "P13": "connector_then_robot_in_loop",
    "P14": "robot_in_loop",
}
FORBIDDEN_METADATA_KEY_PARTS = ("sha256", "checksum", "digest", "hash")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _walk_mapping_keys(value: Any, prefix: str = "document") -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{prefix} contains a non-text key")
            path = f"{prefix}.{key}"
            yield path
            yield from _walk_mapping_keys(child, path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_mapping_keys(child, f"{prefix}[{index}]")


def _reject_fingerprint_metadata(document: Mapping[str, Any]) -> None:
    for path in _walk_mapping_keys(document):
        leaf = path.rsplit(".", 1)[-1].lower().replace("-", "_")
        if any(part in leaf for part in FORBIDDEN_METADATA_KEY_PARTS):
            raise ValueError(
                "acceptance evidence uses semantic identity and resolved "
                f"readback, not fingerprint metadata: {path}"
            )


def _expect_number(
    mapping: Mapping[str, Any], field: str, expected: float, label: str
) -> None:
    actual = _finite(mapping.get(field), f"{label}.{field}")
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-15):
        raise ValueError(f"{label}.{field} changed from the frozen threshold")


def _expect_true(mapping: Mapping[str, Any], field: str, label: str) -> None:
    if mapping.get(field) is not True:
        raise ValueError(f"{label}.{field} must remain true")


def _validate_identity_and_evidence(document: Mapping[str, Any]) -> None:
    if {
        "model_contract": document.get("model_contract"),
        "evidence_policy": document.get("evidence_policy"),
    } != FROZEN_ACCEPTANCE_IDENTITY_AND_EVIDENCE:
        raise ValueError(
            "acceptance identity/evidence policy differs from its frozen snapshot"
        )
    if document.get("status") != "THRESHOLDS_FROZEN_BENCHES_NOT_RUN":
        raise ValueError("acceptance document must not claim that benches ran")
    identity = _mapping(document.get("model_contract"), "model_contract")
    if identity != {
        "path": MODEL_CONTRACT_REPOSITORY_PATH,
        "required_successor_revision": SUCCESSOR_REVISION,
        "required_successor_asset": SUCCESSOR_ASSET_NAME,
    }:
        raise ValueError("acceptance/model contract identity changed")
    evidence = _mapping(document.get("evidence_policy"), "evidence_policy")
    if evidence.get("identity_basis") != (
        "semantic_ids_exact_paths_schema_and_resolved_readback"
    ):
        raise ValueError("evidence identity basis changed")
    if evidence.get("cryptographic_fingerprints_allowed") is not False:
        raise ValueError("cryptographic fingerprints are forbidden")
    for field in (
        "fail_closed_on_missing_field_or_trace",
        "controller_truth_firewall_applies",
        "posthoc_truth_cannot_override_a_formal_failure",
    ):
        _expect_true(evidence, field, "evidence_policy")
    if evidence.get("fresh_process_replays_per_deterministic_case") != 3:
        raise ValueError("deterministic benches require three fresh processes")


def _validate_shared_profile(document: Mapping[str, Any]) -> None:
    shared = _mapping(document.get("shared_numeric_profile"), "shared_numeric_profile")
    if shared != FROZEN_ACCEPTANCE_SHARED_NUMERIC_PROFILE:
        raise ValueError("complete shared numeric profile differs from its frozen snapshot")
    expected = {
        "physics_rate_hz": 240.0,
        "physics_dt_s": 1.0 / 240.0,
        "fine_contact_offset_m": 1.0e-5,
        "rest_offset_m": 0.0,
        "max_hard_stop_gap_or_penetration_m": 5.0e-5,
        "slow_linear_speed_m_s": 5.0e-4,
        "slow_angular_speed_deg_s": 5.0,
        "no_drive_hold_s": 5.0,
        "robot_formal_perpendicular_moment_max_nm": 0.30,
    }
    if set(shared) != set(expected) | {
        "first_formal_exceedance_fails_episode",
        "hard_penetration_excludes_declared_compliant_physical_deflection",
        "compliant_exclusion_requires_resolved_intended_collider_pair",
        "compliant_exclusion_requires_effective_material_binding",
        "compliant_exclusion_requires_positive_resolved_stiffness",
        "missing_or_wrong_compliant_binding_is_scored_as_hard_penetration",
        "compliant_energy_reference",
        "energy_balance_includes_external_work",
        "energy_state_equation",
        "energy_residual_equation",
        "unexplained_energy_gain_definition",
        "energy_trace_required_fields",
    }:
        raise ValueError("shared numeric profile fields changed")
    for field, value in expected.items():
        _expect_number(shared, field, value, "shared_numeric_profile")
    _expect_true(
        shared, "first_formal_exceedance_fails_episode", "shared_numeric_profile"
    )
    _expect_true(
        shared,
        "hard_penetration_excludes_declared_compliant_physical_deflection",
        "shared_numeric_profile",
    )
    _expect_true(
        shared, "energy_balance_includes_external_work", "shared_numeric_profile"
    )
    for field in (
        "compliant_exclusion_requires_resolved_intended_collider_pair",
        "compliant_exclusion_requires_effective_material_binding",
        "compliant_exclusion_requires_positive_resolved_stiffness",
        "missing_or_wrong_compliant_binding_is_scored_as_hard_penetration",
    ):
        _expect_true(shared, field, "shared_numeric_profile")
    if shared.get("compliant_energy_reference") != (
        "after_gravity_free_preload_settle_before_external_probe"
    ):
        raise ValueError("compliant preload energy reference changed")
    if (
        shared.get("energy_state_equation")
        != "E=K+Ug+sum(0.5*k_eff*deflection^2)"
        or shared.get("energy_residual_equation")
        != "R=delta_E-W_external+E_dissipation"
        or shared.get("unexplained_energy_gain_definition") != "max(0,R)"
        or shared.get("energy_trace_required_fields")
        != [
            "timestamp_s",
            "kinetic_energy_j",
            "gravitational_potential_energy_j",
            "compliant_energy_by_role_j",
            "external_applied_work_j",
            "kinematic_or_constraint_work_j",
            "friction_dissipation_j",
            "damping_dissipation_j",
            "residual_j",
        ]
    ):
        raise ValueError("executable preload-energy accounting contract changed")


def _bench(document: Mapping[str, Any], bench_id: str) -> Mapping[str, Any]:
    benches = _mapping(document.get("benches"), "benches")
    return _mapping(benches.get(bench_id), f"benches.{bench_id}")


def _validate_bench_inventory(document: Mapping[str, Any]) -> None:
    benches = _mapping(document.get("benches"), "benches")
    if tuple(benches) != REQUIRED_BENCH_IDS:
        raise ValueError("benches must contain ordered P1 through P14")
    for bench_id, expected_name in EXPECTED_BENCH_NAMES.items():
        bench = _mapping(benches.get(bench_id), f"benches.{bench_id}")
        if bench.get("name") != expected_name:
            raise ValueError(f"{bench_id} name changed")
        if bench.get("mode") != EXPECTED_BENCH_MODES[bench_id]:
            raise ValueError(f"{bench_id} mode changed")
        _mapping(bench.get("inputs"), f"benches.{bench_id}.inputs")
        _mapping(bench.get("pass"), f"benches.{bench_id}.pass")


def _validate_connector_benches(document: Mapping[str, Any]) -> None:
    p1 = _bench(document, "P1")
    p1_inputs = _mapping(p1["inputs"], "P1.inputs")
    p1_pass = _mapping(p1["pass"], "P1.pass")
    if p1_inputs.get("thread_start_phases_deg") != [0.0, 120.0, 240.0]:
        raise ValueError("P1 must cover all three thread starts")
    driver = _mapping(
        p1_inputs.get("component_driver_profile"),
        "P1 component driver profile",
    )
    expected_driver = {
        "schema_version": "kcg_d38999_p1_component_driver_v1",
        "gravity_magnitude_m_s2": 0.0,
        "gravity_set_before_world_reset": True,
        "settle_steps": 120,
        "hold_steps": 240,
        "translation_position_gain_n_m": 600.0,
        "translation_velocity_gain_n_s_m": 8.0,
        "translation_force_component_limit_n": 8.0,
        "roll_pitch_position_gain_nm_rad": 1.2,
        "body_yaw_position_gain_nm_rad": 0.8,
        "nut_yaw_position_gain_nm_rad": 0.8,
        "angular_velocity_gain_nm_s_rad": 0.01,
        "torque_component_limit_nm": 0.30,
        "body_yaw_target_rad": 0.0,
        "nut_yaw_target_equation":
        "-2*pi*max(0,separation-thread_entry_separation)/thread_lead",
        "body_and_nut_translation_target_is_same_frozen_axial_target": True,
        "forces_and_torques_applied_in_world_frame": True,
        "force_and_torque_clamps_are_componentwise": True,
        "object_pose_write_after_physics_start_allowed": False,
    }
    if dict(driver) != expected_driver:
        raise ValueError("P1 component driver profile changed")
    precedence = tuple(
        tuple(edge) for edge in p1_pass.get("required_precedence_edges", ())
    )
    if precedence != REQUIRED_SEQUENCE_PRECEDENCE:
        raise ValueError("P1 physical mating precedence changed")
    if tuple(p1_pass.get("nominal_r7_expected_order", ())) != NOMINAL_R7_EVENT_ORDER:
        raise ValueError("P1 nominal r7 event order changed")
    positions = _mapping(
        p1_pass.get("nominal_event_datum_B_separation_mm"),
        "P1 nominal event positions",
    )
    if {
        event: _finite(positions.get(event), f"P1 event {event}")
        for event in NOMINAL_R7_EVENT_ORDER
    } != NOMINAL_EVENT_B_SEPARATION_MM:
        raise ValueError("P1 nominal event positions changed")
    _expect_number(
        p1_pass, "nominal_event_position_tolerance_m", 5.0e-5, "P1.pass"
    )
    _expect_true(
        p1_pass, "observed_event_order_must_be_recorded", "P1.pass"
    )
    _expect_true(p1_pass, "all_three_thread_starts_enter", "P1.pass")
    if p1_pass.get("false_bottoming_count") != 0:
        raise ValueError("P1 false bottoming tolerance must remain zero")

    p2 = _bench(document, "P2")
    p2_inputs = _mapping(p2["inputs"], "P2.inputs")
    p2_pass = _mapping(p2["pass"], "P2.pass")
    expected_sweep = {
        "yaw_start_deg": 0,
        "yaw_stop_deg_inclusive": 359,
        "yaw_step_deg": 1,
        "mandatory_wrong_yaw_deg": 180,
    }
    if any(p2_inputs.get(key) != value for key, value in expected_sweep.items()):
        raise ValueError("P2 no longer covers the full one-degree yaw sweep")
    for field in (
        "outside_geometry_admissible_n_key_window_blocked",
        "block_before_thread_entry",
        "block_before_first_electrical_contact",
    ):
        _expect_true(p2_pass, field, "P2.pass")
    if p2_pass.get("false_bottoming_count") != 0:
        raise ValueError("P2 false bottoming tolerance must remain zero")

    p3 = _bench(document, "P3")
    p3_inputs = _mapping(p3["inputs"], "P3.inputs")
    p3_pass = _mapping(p3["pass"], "P3.pass")
    if p3_inputs.get("x_offsets_m") != [
        0.0, -0.00005, 0.00005, -0.00010, 0.00010, -0.00020, 0.00020
    ] or p3_inputs.get("y_offsets_m") != p3_inputs.get("x_offsets_m"):
        raise ValueError("P3 XY sweep changed")
    if p3_inputs.get("tilt_x_rad") != [
        0.0, -0.002, 0.002, -0.004, 0.004, -0.008, 0.008
    ] or p3_inputs.get("tilt_y_rad") != p3_inputs.get("tilt_x_rad"):
        raise ValueError("P3 tilt sweep changed")
    _expect_number(p3_pass, "minimum_passive_capture_abs_xy_m", 5.0e-5, "P3.pass")
    _expect_number(p3_pass, "minimum_passive_capture_abs_tilt_rad", 0.002, "P3.pass")
    _expect_number(
        p3_pass,
        "max_noncompliant_hard_surface_penetration_m",
        5.0e-5,
        "P3.pass",
    )

    p4 = _bench(document, "P4")
    p4_inputs = _mapping(p4["inputs"], "P4.inputs")
    p4_pass = _mapping(p4["pass"], "P4.pass")
    for field in (
        "single_contact_then_full_array",
        "load_and_unload",
        "complete_pair_axial_force_probe",
        "spring_finger_force_reported_for_complete_pair",
        "record_active_collider_pair_and_contact_point_counts",
        "calibration_uses_resolved_normal_force_and_overlap",
        "geometry_or_topology_change_for_calibration_forbidden",
    ):
        _expect_true(p4_inputs, field, "P4.inputs")
    if p4_inputs.get("seal_physical_deflection_probe_m") != [
        0.000280, 0.000435, 0.000590
    ]:
        raise ValueError("P4 Series-III seal-deflection probes changed")
    _expect_true(
        p4_inputs, "energy_reference_after_preload_settle", "P4.inputs"
    )
    if p4_inputs.get("force_slope_calibration_roles") != [
        "socket_petal_per_petal", "spring_finger_per_finger",
        "pin_barrier_per_barrier", "peripheral_seal_per_segment",
    ]:
        raise ValueError("P4 realized contact-slope calibration roles changed")
    _expect_number(
        p4_pass,
        "spring_finger_lead_before_first_electrical_contact_m_min",
        0.00102,
        "P4.pass",
    )
    _expect_number(
        p4_pass,
        "shell25_spring_finger_complete_connector_axial_force_n_min",
        2.0,
        "P4.pass",
    )
    _expect_number(
        p4_pass,
        "shell25_spring_finger_complete_connector_axial_force_n_max",
        156.0,
        "P4.pass",
    )
    _expect_number(
        p4_pass, "socket_sleeve_physical_deflection_m_max", 0.000075, "P4.pass"
    )
    if p4_pass.get("pin_barrier_contact_count") != 61:
        raise ValueError("P4 must retain all 61 interfacial pin barriers")
    _expect_number(
        p4_pass,
        "pin_barrier_nominal_first_touch_datum_B_separation_m",
        0.014305,
        "P4.pass",
    )
    _expect_number(
        p4_pass,
        "pin_barrier_post_contact_axial_travel_to_bottoming_m",
        0.000745,
        "P4.pass",
    )
    _expect_number(
        p4_pass,
        "pin_barrier_resolved_normal_deflection_m_max",
        0.000335,
        "P4.pass",
    )
    _expect_number(
        p4_pass, "spring_finger_physical_deflection_m_max", 0.000250, "P4.pass"
    )
    _expect_number(
        p4_pass, "seal_nominal_physical_deflection_m", 0.000435, "P4.pass"
    )
    _expect_number(
        p4_pass, "seal_physical_deflection_m_max", 0.000590, "P4.pass"
    )
    _expect_number(
        p4_pass,
        "max_noncompliant_hard_surface_penetration_m",
        5.0e-5,
        "P4.pass",
    )
    _expect_number(p4_pass, "dissipated_energy_j_min", 0.0, "P4.pass")
    _expect_number(
        p4_pass, "unexplained_energy_gain_j_max", 1.0e-6, "P4.pass"
    )
    if p4_pass.get("realized_normal_force_slope_target_n_m") != {
        "socket_petal_per_petal": 4000.0,
        "spring_finger_per_finger": 12000.0,
        "pin_barrier_per_barrier": 250.0,
        "peripheral_seal_per_segment": 800.0,
    }:
        raise ValueError("P4 realized normal-force slope targets changed")
    _expect_number(
        p4_pass, "realized_normal_force_slope_relative_error_max", 0.10,
        "P4.pass",
    )
    _expect_number(
        p4_pass,
        "band_or_seam_force_discontinuity_relative_to_full_scale_max",
        0.05,
        "P4.pass",
    )
    for field in (
        "declared_compliant_deflection_excluded_from_hard_penetration_gate",
        "force_nonnegative_along_compression",
        "unload_returns_without_residual_growth",
        "initial_preload_energy_accounted_separately",
        "external_work_accounted_in_energy_balance",
        "active_pair_and_contact_point_trace_complete",
        "authored_material_stiffness_is_not_accepted_as_effective_slope_proof",
    ):
        _expect_true(p4_pass, field, "P4.pass")
    if p4_pass.get("force_curve_claim") != "simulation_proxy_only":
        raise ValueError("P4 force curves cannot be promoted to hardware truth")

    p5_inputs = _mapping(_bench(document, "P5")["inputs"], "P5.inputs")
    p5_pass = _mapping(_bench(document, "P5")["pass"], "P5.pass")
    if p5_inputs.get("geometry_variants") != [
        "minimum_interference", "nominal", "maximum_interference"
    ] or p5_inputs.get("geometry_variant_source") != (
        "model_contract.series_III_size20_interface_detail."
        "r7_collision_blueprint.deterministic_geometry_variants"
    ):
        raise ValueError("P5 linked Figure-14 geometry variants changed")
    _expect_true(
        p5_inputs, "independent_cartesian_public_extremes_forbidden", "P5.inputs"
    )
    if (
        p5_pass.get("nominal_same_label_contact_count") != 61
        or p5_pass.get("nominal_cross_label_contact_count") != 0
        or p5_pass.get("bad_control_complete_contact_count_max") != 60
    ):
        raise ValueError("P5 must prove 61 same-label contacts and a failing control")
    _expect_number(
        p5_pass,
        "max_hard_annulus_or_wrong_surface_penetration_m",
        5.0e-5,
        "P5.pass",
    )
    _expect_number(
        p5_pass,
        "compliant_sleeve_physical_deflection_m_max",
        0.000075,
        "P5.pass",
    )

    p6_inputs = _mapping(_bench(document, "P6")["inputs"], "P6.inputs")
    p6_pass = _mapping(_bench(document, "P6")["pass"], "P6.pass")
    _expect_number(p6_inputs, "expected_lead_m_per_revolution", 0.00762, "P6.inputs")
    _expect_number(p6_pass, "fitted_lead_relative_error_max", 0.01, "P6.pass")
    _expect_number(p6_pass, "full_revolution_axial_error_m_max", 5.0e-5, "P6.pass")
    _expect_number(p6_pass, "inter_start_axial_difference_m_max", 5.0e-5, "P6.pass")
    _expect_true(p6_pass, "no_runtime_engagement_switch_or_axial_pose_write", "P6.pass")
    if p6_pass.get("loaded_rail_interval_from_receptacle_B_m") != [
        0.00912, 0.01674
    ]:
        raise ValueError("P6 one-lead loaded rail interval changed")
    _expect_number(
        p6_pass,
        "nominal_loaded_entry_plane_from_receptacle_B_m",
        0.00912,
        "P6.pass",
    )
    if (
        p6_pass.get("nut_body_joint_type") != "UsdPhysics.Joint"
        or p6_pass.get("nut_body_transZ_backup_limits_m")
        != [-0.00015, 0.00015]
    ):
        raise ValueError("P6 nut/body D6 bearing contract changed")
    _expect_true(p6_pass, "nut_body_joint_collision_enabled_readback", "P6.pass")
    _expect_true(p6_pass, "physical_shoulder_contact_before_joint_limit", "P6.pass")

    p7_pass = _mapping(_bench(document, "P7")["pass"], "P7.pass")
    _expect_number(p7_pass, "max_shell_gap_m", 5.0e-5, "P7.pass")
    _expect_number(p7_pass, "max_shell_penetration_m", 5.0e-5, "P7.pass")
    _expect_number(
        p7_pass, "full_mate_datum_B_separation_m", 0.01505, "P7.pass"
    )
    _expect_true(
        p7_pass, "both_stop_surface_assignments_are_simulation_proxies", "P7.pass"
    )
    _expect_true(p7_pass, "physical_shell_stop_pair_required", "P7.pass")
    if p7_pass.get("same_label_contact_count") != 61:
        raise ValueError("P7 must retain all 61 contacts at bottoming")
    _expect_true(p7_pass, "red_band_is_not_a_pass_signal", "P7.pass")
    if p7_pass.get("spontaneous_release_count") != 0:
        raise ValueError("P7 spontaneous release tolerance must remain zero")


def _validate_force_fixture_robot_benches(document: Mapping[str, Any]) -> None:
    p8_inputs = _mapping(_bench(document, "P8")["inputs"], "P8.inputs")
    p8_pass = _mapping(_bench(document, "P8")["pass"], "P8.pass")
    if p8_inputs.get("material_parameter_scales") != [0.5, 1.0, 1.5]:
        raise ValueError("P8 material scale sweep changed")
    _expect_number(p8_inputs, "rotation_speed_deg_s", 5.0, "P8.inputs")
    for field in (
        "complete_pair_coupling_probe",
        "complete_pair_probe_runs_without_robot",
        "detent_component_is_reported_separately",
        "settle_and_rebase_preload_energy_for_each_material_scale",
        "detent_force_overlap_component_calibration",
        "detent_all_three_followers_individual_and_aggregate_trace",
        "detent_active_collider_pair_and_contact_point_trace",
        "detent_tooth_seam_crossing_probe",
        "detent_damping_trace",
    ):
        _expect_true(p8_inputs, field, "P8.inputs")
    _expect_number(p8_pass, "complete_pair_coupling_torque_nm_max", 4.6, "P8.pass")
    _expect_number(
        p8_pass, "detent_initial_forward_component_nm_min", 0.0005, "P8.pass"
    )
    _expect_number(
        p8_pass, "detent_full_forward_ramp_component_nm_max", 0.10, "P8.pass"
    )
    _expect_number(
        p8_pass, "detent_physical_deflection_m_max", 0.000051, "P8.pass"
    )
    _expect_number(
        p8_pass, "detent_resolved_force_overlap_slope_target_n_m", 110000.0,
        "P8.pass",
    )
    _expect_number(
        p8_pass, "detent_resolved_force_overlap_slope_relative_error_max", 0.10,
        "P8.pass",
    )
    _expect_number(
        p8_pass,
        "detent_tooth_seam_force_discontinuity_relative_to_full_scale_max",
        0.05,
        "P8.pass",
    )
    _expect_number(
        p8_pass, "unexplained_energy_gain_j_max", 1.0e-6, "P8.pass"
    )
    for field in (
        "torque_changes_monotonically_with_friction_or_normal_load",
        "detent_result_cannot_be_substituted_for_complete_pair_result",
        "initial_preload_energy_accounted_separately",
        "external_work_accounted_in_energy_balance",
        "kinematic_or_constraint_work_accounted_in_energy_balance",
        "friction_and_damping_dissipation_traced",
        "detent_force_overlap_curve_finite_nonnegative_and_monotonic",
        "detent_active_pair_and_contact_point_trace_complete",
        "detent_resolved_damping_trace_complete",
        "authored_detent_material_stiffness_is_not_effective_slope_proof",
    ):
        _expect_true(p8_pass, field, "P8.pass")
    _expect_number(p8_pass, "robot_formal_perpendicular_moment_nm_max", 0.30, "P8.pass")
    _expect_true(p8_pass, "first_robot_limit_exceedance_fails_episode", "P8.pass")
    _expect_true(p8_pass, "posthoc_filter_cannot_restore_failure", "P8.pass")

    p9_inputs = _mapping(_bench(document, "P9")["inputs"], "P9.inputs")
    p9_pass = _mapping(_bench(document, "P9")["pass"], "P9.pass")
    _expect_number(p9_inputs, "no_drive_hold_s", 5.0, "P9.inputs")
    _expect_number(
        p9_inputs, "robot_reverse_disturbance_nm_max", 0.30, "P9.inputs"
    )
    for field in (
        "component_only_reverse_probe_allowed",
        "complete_pair_disengagement_probe",
        "complete_pair_probe_runs_without_robot",
    ):
        _expect_true(p9_inputs, field, "P9.inputs")
    _expect_number(
        p9_pass, "complete_pair_disengagement_torque_nm_min", 0.6, "P9.pass"
    )
    _expect_number(
        p9_pass, "complete_pair_disengagement_torque_nm_max", 4.6, "P9.pass"
    )
    _expect_number(p9_pass, "hold_rotation_deg_max", 1.0, "P9.pass")
    _expect_number(p9_pass, "hold_axial_retreat_m_max", 5.0e-5, "P9.pass")
    _expect_number(
        p9_pass, "robot_disturbance_rotation_deg_max", 1.0, "P9.pass"
    )
    _expect_number(
        p9_pass, "robot_disturbance_axial_retreat_m_max", 5.0e-5, "P9.pass"
    )
    _expect_true(p9_pass, "component_limit_cannot_relax_robot_limit", "P9.pass")
    _expect_true(
        p9_pass, "complete_pair_limits_do_not_apply_to_detent_alone", "P9.pass"
    )
    _expect_true(p9_pass, "reverse_motion_follows_negative_thread_lead", "P9.pass")

    p10_inputs = _mapping(_bench(document, "P10")["inputs"], "P10.inputs")
    p10_pass = _mapping(_bench(document, "P10")["pass"], "P10.pass")
    for field, value in {
        "axial_force_n": 5.0,
        "lateral_force_n": 2.0,
        "bending_moment_nm": 0.18,
        "torsional_moment_nm": 0.30,
    }.items():
        _expect_number(p10_inputs, field, value, "P10.inputs")
    _expect_number(p10_pass, "translation_drift_m_max", 1.0e-6, "P10.pass")
    _expect_number(p10_pass, "rotation_drift_rad_max", 1.0e-5, "P10.pass")
    for field in (
        "receptacle_to_fixture_joint_required",
        "fixture_to_world_joint_required",
        "direct_world_static_receptacle_forbidden",
        "fixture_explicit_mass_api_required",
        "receptacle_explicit_mass_api_required",
    ):
        _expect_true(p10_pass, field, "P10.pass")
    if (
        p10_pass.get("receptacle_to_fixture_joint_path")
        != "/World/D38999TabletopV1/Joints/ReceptacleToFixture"
        or p10_pass.get("fixture_to_world_joint_path")
        != "/World/D38999TabletopV1/Joints/FixtureToWorld"
        or p10_pass.get("fixture_to_world_body0_relationship_target_count") != 0
        or p10_pass.get("fixture_to_world_localPos0_m")
        != [0.550, 0.185, 0.220]
        or p10_pass.get("fixture_to_world_localRot0_wxyz")
        != [1.0, 0.0, 0.0, 0.0]
        or p10_pass.get("fixture_to_world_localPos1_m") != [0.0, 0.0, 0.0]
        or p10_pass.get("fixture_to_world_localRot1_wxyz")
        != [1.0, 0.0, 0.0, 0.0]
        or p10_pass.get("receptacle_to_fixture_localPos0_m")
        != [0.0, 0.0, 0.052]
        or p10_pass.get("receptacle_to_fixture_localRot0_wxyz")
        != [0.0, 1.0, 0.0, 0.0]
        or p10_pass.get("receptacle_to_fixture_localPos1_m")
        != [0.0, 0.0, 0.0]
        or p10_pass.get("receptacle_to_fixture_localRot1_wxyz")
        != [1.0, 0.0, 0.0, 0.0]
        or p10_pass.get("fixture_diagonal_inertia_kg_m2")
        != [0.0088333333333, 0.0088333333333, 0.0163333333333]
    ):
        raise ValueError("P10 fixture path or mass-property contract changed")
    if p10_pass.get("fixture_or_table_penetration_count") != 0:
        raise ValueError("P10 fixture/table penetration tolerance must remain zero")

    p11_pass = _mapping(_bench(document, "P11")["pass"], "P11.pass")
    _expect_true(p11_pass, "self_collision_enabled_readback", "P11.pass")
    if (
        p11_pass.get("excluded_pair_count_initial") != 16
        or p11_pass.get("sampled_never_pair_exclusion_count") != 0
        or p11_pass.get("nonexcluded_path_collision_count") != 0
        or p11_pass.get("unexpected_environment_contact_count") != 0
    ):
        raise ValueError("P11 self-collision acceptance changed")
    _expect_true(p11_pass, "deliberate_positive_control_detected", "P11.pass")
    _expect_true(
        p11_pass, "excluded_pair_set_exactly_matches_contract", "P11.pass"
    )
    if p11_pass.get("successor_regenerated_from_frozen_source_path") != (
        "src/iiwa_description/urdf/handarm.urdf.xacro"
    ) or p11_pass.get("successor_robot_asset_path") != (
        "artifacts/kcg_connector/isaac/robot/handarm_keyed_v3_physical_r7/handarm.usda"
    ) or p11_pass.get("arm_mass_inertia_source_path") != (
        "src/iiwa_description/urdf/iiwa14.xacro"
    ) or p11_pass.get("hand_mass_inertia_source_path") != (
        "src/iiwa_description/urdf/hand.xacro"
    ) or p11_pass.get("predecessor_physics_usda_reused_as_successor") is not False:
        raise ValueError("P11 successor robot source identity changed")
    if (
        p11_pass.get("robot_collision_link_count") != 17
        or p11_pass.get("unassigned_material_role_collider_count") != 0
        or p11_pass.get("unassigned_response_role_collider_count") != 0
        or p11_pass.get("fingertip_role_links")
        != ["f1Link3", "f2Link2", "f3Link3"]
        or p11_pass.get("nonterminal_finger_role_violation_count") != 0
        or p11_pass.get("resolved_high_friction_robot_links")
        != ["f1Link3", "f2Link2", "f3Link3"]
    ):
        raise ValueError("P11 robot collision/material role inventory changed")
    for field in (
        "full_robot_material_role_mapping_exact",
        "every_robot_collision_link_has_collision_api",
        "every_robot_collision_link_uses_exact_frozen_source_mesh",
        "every_robot_link_mass_com_inertia_matches_frozen_source",
    ):
        _expect_true(p11_pass, field, "P11.pass")

    p12_pass = _mapping(_bench(document, "P12")["pass"], "P12.pass")
    for field, value in {
        "gain_min": 0.99,
        "gain_max": 1.01,
        "same_kind_crosstalk_fraction_max": 0.01,
        "odd_symmetry_error_fraction_max": 0.01,
        "half_amplitude_linearity_error_fraction_max": 0.01,
        "sample_age_s_max": 0.020,
    }.items():
        _expect_number(p12_pass, field, value, "P12.pass")
    if p12_pass.get("fingertip_tactile_input_count") != 0:
        raise ValueError("P12 cannot invent fingertip tactile input")
    for field in (
        "camera_has_mass_rigid_body_or_collision_api_count",
        "wrist_ft_independent_sensor_body_or_collider_count",
        "grasp_tcp_forbidden_physics_api_count",
    ):
        if p12_pass.get(field) != 0:
            raise ValueError(f"P12.{field} tolerance must remain zero")
    for field in (
        "tare_payload_gravity_and_inertia_compensation_required",
        "safety_threshold_fields_may_not_be_null",
        "camera_extrinsics_constant_during_episode",
    ):
        _expect_true(p12_pass, field, "P12.pass")
    if p12_pass.get("wrist_ft_source_joint_semantic_id") != "hand2arm":
        raise ValueError("P12 wrist FT source joint changed")
    if (
        p12_pass.get("palm_camera_canonical_suffix") != "/PalmCamera"
        or p12_pass.get("wrist_camera_canonical_suffix") != "/WristCamera"
        or p12_pass.get("duplicate_live_view_camera_prim_count") != 0
        or p12_pass.get("camera_resolution_px") != [1280, 720]
        or p12_pass.get("camera_clipping_range_m") != [0.02, 10.0]
        or p12_pass.get("wrist_ft_canonical_joint_path")
        != "/World/HandArm/Physics/hand2arm"
        or p12_pass.get("wrist_ft_raw_frame") != "handbase_link"
        or p12_pass.get("wrist_ft_canonical_from_raw") != "negate_all_six_components"
        or p12_pass.get("grasp_tcp_canonical_suffix") != "/grasp_tcp"
    ):
        raise ValueError("P12 canonical camera, FT, or TCP identity changed")
    _expect_true(
        p12_pass, "grasp_tcp_is_massless_noncolliding_virtual_frame", "P12.pass"
    )
    if p12_pass.get("grasp_tcp_handbase_translation_m") != [0.0, 0.0, 0.400]:
        raise ValueError("P12 grasp TCP translation changed")
    if p12_pass.get("grasp_tcp_handbase_rotation_wxyz") != [1.0, 0.0, 0.0, 0.0]:
        raise ValueError("P12 grasp TCP rotation changed")

    p13_inputs = _mapping(_bench(document, "P13")["inputs"], "P13.inputs")
    p13_pass = _mapping(_bench(document, "P13")["pass"], "P13.pass")
    if p13_inputs.get("fresh_process_replays") != 3:
        raise ValueError("P13 requires three fresh processes")
    _expect_number(p13_pass, "full_mate_translation_drift_m_max", 5.0e-5, "P13.pass")
    _expect_number(p13_pass, "full_mate_rotation_drift_deg_max", 0.1, "P13.pass")
    for zero_field in (
        "nan_count",
        "traceback_count",
        "solver_explosion_count",
        "spontaneous_unlock_count",
        "pose_write_after_initialization_count",
        "intended_compliant_thread_detent_pair_filtered_count",
        "resolved_compliant_material_error_count",
        "nonconvex_dynamic_collider_count",
        "automatic_collision_approximation_count",
        "ancestor_filter_covering_intended_pair_count",
        "compliant_pair_with_zero_or_two_compliant_sides_count",
    ):
        if p13_pass.get(zero_field) != 0:
            raise ValueError(f"P13.{zero_field} tolerance must remain zero")

    p14_inputs = _mapping(_bench(document, "P14")["inputs"], "P14.inputs")
    p14_pass = _mapping(_bench(document, "P14")["pass"], "P14.pass")
    if tuple(p14_inputs.get("controller_inputs", ())) != ALLOWED_CONTROLLER_INPUTS:
        raise ValueError("P14 controller input boundary changed")
    _expect_number(p14_pass, "robot_formal_perpendicular_moment_nm_max", 0.30, "P14.pass")
    _expect_true(p14_pass, "first_formal_exceedance_fails_episode", "P14.pass")
    if p14_pass.get("forbidden_controller_input_count") != 0:
        raise ValueError("P14 forbidden controller input tolerance must remain zero")
    if p14_pass.get("forbidden_mechanism_count") != 0:
        raise ValueError("P14 forbidden mechanism tolerance must remain zero")


def _validate_release(document: Mapping[str, Any]) -> None:
    release = _mapping(document.get("phase_release"), "phase_release")
    if release != FROZEN_ACCEPTANCE_PHASE_RELEASE:
        raise ValueError("complete phase-release contract differs from its frozen snapshot")
    if release.get("A3_requires") != list(REQUIRED_BENCH_IDS):
        raise ValueError("A3 release must require every P1-P14 bench")
    if release.get("downstream_release_before_A5") is not False:
        raise ValueError("downstream work cannot release before A5")
    a2_required = set(release.get("A2_requires", []))
    if a2_required != {
        "structural_inventory_pass",
        "resolved_stage_property_readback",
        "geometry_derived_from_collision_not_metadata",
        "material_role_partition_pass",
        "intended_contact_pair_filter_audit_pass",
        "mass_inertia_and_joint_readback_pass",
        "complete_resolved_readback_inventory_pass",
        "realized_robot_asset_contract_pass",
        "connector_blueprint_inventory_pass",
        "explicit_convex_topology_and_bounds_pass",
        "response_role_and_compliance_side_pass",
        "leaf_filter_source_audit_pass",
        "canonical_camera_fixture_path_pass",
    }:
        raise ValueError("A2 release requirements changed")
    a2_result = _mapping(release.get("A2_result_contract"), "A2 result contract")
    model = load_physical_model_contract()
    solver = _mapping(model.document.get("solver_profile"), "model solver profile")
    model_result = _mapping(
        solver.get("resolved_readback_result_contract"),
        "model resolved readback result contract",
    )
    expected_sources = {
        "collider_rows": "solver_profile.resolved_readback_required_fields",
        "property_rows": "solver_profile.resolved_property_readback_required_fields",
        "family_pair_rows": "solver_profile.resolved_family_pair_readback_required_fields",
        "filter_source_rows": "solver_profile.resolved_filter_source_row_required_fields",
    }
    expected_counts = {
        "collider_rows": 15037,
        "property_rows": 22,
        "family_pair_rows": 406,
        "filter_source_rows": 387,
    }
    if (
        a2_result.get("schema_version") != model_result.get("schema_version")
        or a2_result.get("generator_id") != model_result.get("generator_id")
        or a2_result.get("contract_revision") != model_result.get("contract_revision")
        or a2_result.get("model_solver_contract_must_match_exactly") is not True
        or a2_result.get("model_contract_is_only_expected_inventory_authority") is not True
        or a2_result.get("external_expected_inventory_allowed") is not False
        or a2_result.get("production_release_API_accepts_composed_asset_path_only") is not True
        or a2_result.get("caller_supplied_mapping_is_candidate_only") is not True
        or a2_result.get("required_top_level_fields")
        != model_result.get("required_top_level_fields")
        or a2_result.get("row_field_sources") != expected_sources
        or a2_result.get("expected_row_counts") != expected_counts
        or a2_result.get("complete_inventory_required_for_every_collection") is not True
        or a2_result.get("claimed_summary_must_equal_internal_recomputation") is not True
        or a2_result.get("all_expected_semantic_inventory_present") is not True
        or a2_result.get("asset_path_must_equal_authorized_A2_output") is not True
    ):
        raise ValueError("A2 realized readback authority or collection contract changed")
    if any(
        a2_result.get(field) != expected_zero
        for field, expected_zero in model_result["required_summary_counts"].items()
    ):
        raise ValueError("A2 acceptance zero-summary contract differs from the model")
    if {
        "collider_rows": model_result.get("expected_collider_row_count"),
        "property_rows": model_result.get("expected_property_row_count"),
        "family_pair_rows": model_result.get("expected_family_pair_row_count"),
        "filter_source_rows": model_result.get("expected_filter_source_row_count"),
    } != expected_counts:
        raise ValueError("A2 model and acceptance row counts differ")
    a5_required = set(release.get("A5_requires", []))
    if a5_required != {
        "all_a2_a3_a4_requirements_pass",
        "proxy_parameters_frozen",
        "material_roles_frozen",
        "solver_profile_frozen",
        "semantic_asset_identity_frozen",
    }:
        raise ValueError("A5 release requirements changed")


@dataclass(frozen=True)
class PhysicalAcceptanceMatrix:
    path: Path
    document: Mapping[str, Any]

    @property
    def bench_ids(self) -> tuple[str, ...]:
        return tuple(self.document["benches"])

    @property
    def benches_have_run(self) -> bool:
        return self.document["status"] != "THRESHOLDS_FROZEN_BENCHES_NOT_RUN"


def load_physical_acceptance_matrix(
    path: Path | str = DEFAULT_ACCEPTANCE_PATH,
) -> PhysicalAcceptanceMatrix:
    acceptance_path = Path(path).expanduser().resolve()
    document = _mapping(
        yaml.safe_load(acceptance_path.read_text(encoding="utf-8")), "document"
    )
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    _reject_fingerprint_metadata(document)
    model = load_physical_model_contract()
    if model.document["identity"]["successor_revision"] != SUCCESSOR_REVISION:
        raise ValueError("active model contract successor revision changed")
    _validate_identity_and_evidence(document)
    _validate_shared_profile(document)
    _validate_bench_inventory(document)
    _validate_connector_benches(document)
    _validate_force_fixture_robot_benches(document)
    if document["benches"] != FROZEN_ACCEPTANCE_BENCHES:
        raise ValueError(
            "complete P1-P14 bench matrix differs from its frozen snapshot"
        )
    _validate_release(document)
    return PhysicalAcceptanceMatrix(path=acceptance_path, document=document)


__all__ = [
    "ALLOWED_CONTROLLER_INPUTS",
    "DEFAULT_ACCEPTANCE_PATH",
    "EXPECTED_BENCH_NAMES",
    "PhysicalAcceptanceMatrix",
    "NOMINAL_R7_EVENT_ORDER",
    "SCHEMA_VERSION",
    "load_physical_acceptance_matrix",
]
