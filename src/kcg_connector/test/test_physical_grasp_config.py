from pathlib import Path

import pytest
import yaml

from kcg_connector.grasp.physical_grasp_config import (
    load_physical_grasp_experiment_config,
)


REPOSITORY = Path(__file__).resolve().parents[3]
CONFIG = REPOSITORY / "src/kcg_connector/config/d38999_tabletop_physical_grasp_v1.yaml"
MULTILAYER_V1_CONFIG = (
    REPOSITORY
    / "src/kcg_connector/config/d38999_multilayer_tabletop_physical_grasp_v1.yaml"
)
MULTILAYER_V2_CONFIG = (
    REPOSITORY
    / "src/kcg_connector/config/d38999_multilayer_tabletop_physical_grasp_v2.yaml"
)


def test_checked_in_physical_grasp_contract_is_strict_and_proxy_free():
    config = load_physical_grasp_experiment_config(CONFIG)
    assert config.physics_rate_hz == 240
    assert config.sequential.detector.threshold_label == "SIM_TUNING_ONLY"
    assert len(config.lift_stages) == 3
    assert config.post_grasp_stabilization_proxy_enabled is False


def test_lift_wrist_gates_remain_8n_and_0p30nm_and_recovery_is_bounded():
    config = load_physical_grasp_experiment_config(CONFIG)
    assert config.stability.maximum_wrist_force_n == 8.0
    assert config.stability.maximum_wrist_moment_nm == 0.30
    assert config.recovery.return_steps_per_waypoint >= 1
    assert config.recovery.settle_steps >= 1
    assert config.recovery.open_duration_s > 0.0


def test_dynamic_closeout_v2_adds_only_evidence_derived_grasp_controls():
    old_document = yaml.safe_load(
        MULTILAYER_V1_CONFIG.read_text(encoding="utf-8")
    )
    new_document = yaml.safe_load(
        MULTILAYER_V2_CONFIG.read_text(encoding="utf-8")
    )
    assert old_document["sequential"][
        "consolidation_final_stiffness_scale"
    ] == pytest.approx(1.0)
    assert new_document["sequential"][
        "consolidation_final_stiffness_scale"
    ] == pytest.approx(0.85)
    old_document["sequential"][
        "consolidation_final_stiffness_scale"
    ] = 0.85
    centering = new_document.pop("pre_lift_centering")
    rebase = new_document.pop("pre_lift_realized_state_rebase")
    compliance = new_document.pop("pre_lift_arm_drive_compliance")
    gravity_transfer = new_document.pop("pre_lift_gravity_preload_transfer")
    stiffness_restoration = new_document.pop(
        "pre_lift_arm_stiffness_restoration"
    )
    finger_damping = new_document.pop("post_contact_finger_damping")
    lift_x_admittance = new_document.pop("lift_x_force_admittance")
    lift_xy_admittance = new_document.pop("lift_xy_force_admittance")
    lift_finger_balance = new_document.pop("lift_finger_load_balance")
    lift_finger_absolute = new_document.pop(
        "lift_finger_absolute_load_hold"
    )
    lift_finger_root_two_sample = new_document.pop(
        "lift_finger_root_load_two_sample_suppression"
    )
    lift_finger_fixed_target = new_document.pop(
        "lift_finger_fixed_target_hold"
    )
    lift_phase_arm_damping = new_document.pop("lift_phase_arm_damping")
    lift_vertical_force_ramp = new_document.pop(
        "lift_task_space_vertical_force_ramp"
    )
    lift_sensor_origin_vertical_force_ramp = new_document.pop(
        "lift_sensor_origin_vertical_force_ramp"
    )
    pre_lift_vertical_force_xy_admittance = new_document.pop(
        "pre_lift_vertical_force_xy_admittance"
    )
    pre_lift_xy_nyquist_suppression = new_document.pop(
        "pre_lift_xy_nyquist_suppression"
    )
    pre_lift_quasistatic_lateral_preload_nulling = new_document.pop(
        "pre_lift_quasistatic_lateral_preload_nulling"
    )
    pre_lift_differential_finger_preload_diagnostic = new_document.pop(
        "pre_lift_differential_finger_preload_diagnostic"
    )
    pre_lift_differential_finger_preload_correction = new_document.pop(
        "pre_lift_differential_finger_preload_correction"
    )
    assert new_document == old_document
    assert centering["enabled"] is False
    assert centering["maximum_entry_moment_score_nm"] == pytest.approx(0.24)
    assert centering["objective_force_scale_n"] == pytest.approx(8.0)
    assert centering["objective_moment_scale_nm"] == pytest.approx(0.30)
    assert rebase["enabled"] is False
    assert rebase["maximum_entry_moment_score_nm"] == pytest.approx(0.24)
    assert rebase["maximum_entry_load_imbalance"] == pytest.approx(0.18)
    assert compliance["enabled"] is True
    assert compliance["stiffness_scale"] == pytest.approx(0.25)
    assert compliance["damping_scale"] == pytest.approx(1.0)
    assert gravity_transfer["enabled"] is True
    assert gravity_transfer["transition_steps"] == 240
    assert gravity_transfer["maximum_gravity_feedforward_nm"] == pytest.approx(
        300.0
    )
    assert gravity_transfer["maximum_feedforward_step_nm"] == pytest.approx(2.5)
    assert stiffness_restoration["enabled"] is True
    assert stiffness_restoration[
        "expected_initial_stiffness_nm_rad"
    ] == pytest.approx(6000.0)
    assert stiffness_restoration[
        "expected_restored_stiffness_nm_rad"
    ] == pytest.approx(24000.0)
    assert stiffness_restoration["expected_damping_nm_s_rad"] == pytest.approx(
        875.29978045654
    )
    assert finger_damping["enabled"] is True
    assert finger_damping["source_run_id"] == "B-V2-GRASP-08"
    assert finger_damping["include_preshape_joint"] is True
    assert finger_damping["preshape_joint_name"] == "f1j1"
    assert (
        finger_damping["preshape_extension_source_run_id"]
        == "B-V2-GRASP-11-IFIX02"
    )
    assert finger_damping["final_damping_nm_s_rad"] == pytest.approx(
        1.8695969008869397
    )
    assert lift_x_admittance["enabled"] is False
    assert lift_x_admittance["source_run_id"] == "B-V2-GRASP-05"
    assert lift_x_admittance["maximum_total_correction_m"] == pytest.approx(
        15.0e-6
    )
    assert lift_xy_admittance["enabled"] is True
    assert lift_xy_admittance["source_run_id"] == "B-V2-GRASP-14"
    assert lift_xy_admittance["source_h8_run_id"] == "B-V2-GRASP-05"
    assert lift_xy_admittance[
        "maximum_total_correction_norm_m"
    ] == pytest.approx(30.0e-6)
    assert lift_finger_balance["enabled"] is True
    assert lift_finger_balance["source_run_id"] == "B-V2-GRASP-12"
    assert lift_finger_balance["reuse_sequential_balance_parameters"] is True
    assert lift_finger_balance["zero_mean_closure_coordinate_trim"] is True
    assert lift_finger_balance["immediately_preceding_root_torque_only"] is True
    assert lift_finger_absolute["enabled"] is True
    assert (
        lift_finger_absolute["reference_window_source"]
        == "TRAILING_H13_TRANSITION_USING_EXISTING_REFERENCE_WINDOW_STEPS"
    )
    assert lift_finger_absolute["reuse_sequential_control_parameters"] is True
    assert lift_finger_absolute["independent_absolute_root_load_hold"] is True
    assert lift_phase_arm_damping["enabled"] is True
    assert lift_phase_arm_damping["source_lift_run_id"] == "B-V2-GRASP-13"
    assert (
        lift_phase_arm_damping["source_hold_run_id"]
        == "B-V2-H12-ZERO-LIFT-01"
    )
    assert lift_phase_arm_damping["source_median_velocity_ratio"] == pytest.approx(
        2.18824945114135
    )
    assert lift_phase_arm_damping["final_damping_nm_s_rad"] == pytest.approx(
        875.29978045654
    )
    assert lift_vertical_force_ramp["enabled"] is True
    assert (
        lift_vertical_force_ramp["force_target_source"]
        == "FROZEN_BODY_PLUS_NUT_WEIGHT"
    )
    assert lift_vertical_force_ramp["world_force_axis"] == [0.0, 0.0, 1.0]
    assert lift_sensor_origin_vertical_force_ramp["enabled"] is False
    assert (
        lift_sensor_origin_vertical_force_ramp["source_h18_run_id"]
        == "B-V2-H18-VERTICAL-FORCE-RAMP-01"
    )
    assert (
        lift_sensor_origin_vertical_force_ramp["force_application_frame"]
        == "handbase_link_sensor_origin"
    )
    assert pre_lift_vertical_force_xy_admittance["enabled"] is False
    assert pre_lift_vertical_force_xy_admittance[
        "no_new_numeric_control_parameter"
    ] is True
    assert pre_lift_xy_nyquist_suppression["enabled"] is True
    assert pre_lift_xy_nyquist_suppression[
        "hard_gate_detection_delay_steps"
    ] == 0
    assert pre_lift_quasistatic_lateral_preload_nulling["enabled"] is True
    assert (
        pre_lift_quasistatic_lateral_preload_nulling["steps_source"]
        == "H13_TRANSITION_STEPS"
    )
    assert pre_lift_quasistatic_lateral_preload_nulling[
        "correction_total_bound_reset_forbidden"
    ] is True
    assert pre_lift_differential_finger_preload_diagnostic["enabled"] is False
    assert pre_lift_differential_finger_preload_diagnostic[
        "probe_increment_source"
    ] == "SEQUENTIAL_PROBE_INCREMENT_RAD"
    assert pre_lift_differential_finger_preload_diagnostic[
        "correction_applied_in_diagnostic"
    ] is False
    assert pre_lift_differential_finger_preload_correction["enabled"] is True
    assert pre_lift_differential_finger_preload_correction[
        "source_analysis_sha256"
    ] == "719b942a1739400300ca5a6a7de7a0f8dd13de259934c726452bbfdcd3530946"
    assert pre_lift_differential_finger_preload_correction[
        "fixed_correction_closure_rad"
    ] == pytest.approx(
        [0.00060507064062471, 0.002476928142280831, -0.003081998782905541]
    )
    assert pre_lift_differential_finger_preload_correction[
        "second_parameter_set_allowed"
    ] is False
    assert lift_finger_root_two_sample["enabled"] is False
    assert (
        lift_finger_root_two_sample["source_derivation_sha256"]
        == "b2d12112af3241dfe3301f9de1aac6e827f84874a1f0dbcc5229c6f0675a9b55"
    )
    assert lift_finger_root_two_sample[
        "two_sample_arithmetic_mean_required"
    ] is True
    assert lift_finger_root_two_sample["h17_control_input_only"] is True
    assert lift_finger_root_two_sample[
        "raw_sensor_hard_gate_unchanged"
    ] is True
    assert lift_finger_root_two_sample[
        "hard_gate_detection_delay_steps"
    ] == 0
    assert lift_finger_fixed_target["enabled"] is True
    assert (
        lift_finger_fixed_target["source_derivation_sha256"]
        == "1ef9abedea61d0eb529cb3282a2f3ef6152aa461f2d81cfa27c7dd12c68c9051"
    )
    assert lift_finger_fixed_target[
        "disable_h17_updates_during_vertical_force_ramp"
    ] is True
    assert lift_finger_fixed_target[
        "disable_h17_updates_during_staged_lift"
    ] is True
    assert lift_finger_fixed_target[
        "raw_sensor_hard_gate_unchanged"
    ] is True
    assert lift_finger_fixed_target[
        "hard_gate_detection_delay_steps"
    ] == 0

    config = load_physical_grasp_experiment_config(MULTILAYER_V2_CONFIG)
    assert config.sequential.consolidation_final_stiffness_scale == pytest.approx(
        0.85
    )
    assert config.stability.maximum_wrist_force_n == pytest.approx(8.0)
    assert config.stability.maximum_wrist_moment_nm == pytest.approx(0.30)
    assert config.lift_finger_root_load_two_sample_suppression.enabled is False
    assert config.lift_finger_fixed_target_hold.enabled is True
    assert config.pre_lift_centering.enabled is False
    assert config.pre_lift_centering.maximum_total_offset_m == pytest.approx(
        0.00050
    )
    assert config.pre_lift_realized_state_rebase.enabled is False
    assert config.pre_lift_realized_state_rebase.reference_window_steps == 240
    assert config.pre_lift_realized_state_rebase.maximum_rebase_joint_delta_rad == pytest.approx(0.005)
    assert config.pre_lift_arm_drive_compliance.enabled is True
    assert config.pre_lift_arm_drive_compliance.transition_steps == 240
    assert config.pre_lift_arm_drive_compliance.stiffness_scale == pytest.approx(0.25)
    assert config.pre_lift_gravity_preload_transfer.enabled is True
    assert config.pre_lift_gravity_preload_transfer.transition_steps == 240
    assert config.pre_lift_arm_stiffness_restoration.enabled is True
    assert config.pre_lift_arm_stiffness_restoration.transition_steps == 240
    assert config.post_contact_finger_damping.enabled is True
    assert config.post_contact_finger_damping.transition_steps == 240
    assert config.post_contact_finger_damping.include_preshape_joint is True
    assert config.lift_x_force_admittance.enabled is False
    assert config.lift_x_force_admittance.task_x_compliance_m_n == pytest.approx(
        2.4375595445442518e-05
    )
    assert config.lift_xy_force_admittance.enabled is True
    assert config.lift_xy_force_admittance.task_xy_compliance_m_n == pytest.approx(
        2.4375595445442518e-05
    )
    assert config.lift_finger_load_balance.enabled is True
    assert config.lift_finger_load_balance.source_run_id == "B-V2-GRASP-12"
    assert config.lift_finger_absolute_load_hold.enabled is True
    assert (
        config.lift_finger_absolute_load_hold.source_h16_run_id
        == "B-V2-H16-STIFFNESS-RESTORE-IFIX01"
    )
    assert config.lift_finger_root_load_two_sample_suppression.enabled is False
    assert config.lift_finger_fixed_target_hold.enabled is True
    assert (
        config.lift_finger_root_load_two_sample_suppression
        .source_derivation_sha256
        == "b2d12112af3241dfe3301f9de1aac6e827f84874a1f0dbcc5229c6f0675a9b55"
    )
    assert config.lift_phase_arm_damping.enabled is True
    assert config.lift_phase_arm_damping.transition_steps == 240
    assert config.lift_task_space_vertical_force_ramp.enabled is True
    assert (
        config.lift_task_space_vertical_force_ramp.source_h17_run_id
        == "B-V2-H17-ABSOLUTE-LOAD-HOLD-01"
    )
    assert config.lift_sensor_origin_vertical_force_ramp.enabled is False
    assert (
        config.lift_sensor_origin_vertical_force_ramp.source_h18_run_id
        == "B-V2-H18-VERTICAL-FORCE-RAMP-01"
    )
    assert config.pre_lift_vertical_force_xy_admittance.enabled is False
    assert config.pre_lift_xy_nyquist_suppression.enabled is True
    assert config.pre_lift_quasistatic_lateral_preload_nulling.enabled is True
    assert config.pre_lift_differential_finger_preload_diagnostic.enabled is False
    assert (
        config.pre_lift_differential_finger_preload_diagnostic.diagnostic_only
        is True
    )
    assert config.pre_lift_differential_finger_preload_correction.enabled is True
    assert (
        config.pre_lift_differential_finger_preload_correction
        .source_analysis_sha256
        == "719b942a1739400300ca5a6a7de7a0f8dd13de259934c726452bbfdcd3530946"
    )
    assert config.post_grasp_stabilization_proxy_enabled is False


def test_historical_contracts_keep_pre_lift_centering_disabled():
    assert load_physical_grasp_experiment_config(
        MULTILAYER_V1_CONFIG
    ).pre_lift_centering.enabled is False
    assert load_physical_grasp_experiment_config(
        MULTILAYER_V1_CONFIG
    ).pre_lift_realized_state_rebase.enabled is False
    assert load_physical_grasp_experiment_config(
        MULTILAYER_V1_CONFIG
    ).pre_lift_arm_drive_compliance.enabled is False
    assert load_physical_grasp_experiment_config(
        MULTILAYER_V1_CONFIG
    ).pre_lift_gravity_preload_transfer.enabled is False
    assert load_physical_grasp_experiment_config(
        MULTILAYER_V1_CONFIG
    ).pre_lift_arm_stiffness_restoration.enabled is False
    assert load_physical_grasp_experiment_config(
        MULTILAYER_V1_CONFIG
    ).post_contact_finger_damping.enabled is False
    assert load_physical_grasp_experiment_config(
        MULTILAYER_V1_CONFIG
    ).lift_x_force_admittance.enabled is False
    assert load_physical_grasp_experiment_config(
        MULTILAYER_V1_CONFIG
    ).lift_xy_force_admittance.enabled is False
    assert load_physical_grasp_experiment_config(
        MULTILAYER_V1_CONFIG
    ).lift_finger_load_balance.enabled is False
    assert load_physical_grasp_experiment_config(
        MULTILAYER_V1_CONFIG
    ).lift_finger_absolute_load_hold.enabled is False
    assert load_physical_grasp_experiment_config(
        MULTILAYER_V1_CONFIG
    ).lift_phase_arm_damping.enabled is False
    assert load_physical_grasp_experiment_config(
        MULTILAYER_V1_CONFIG
    ).lift_task_space_vertical_force_ramp.enabled is False
    assert load_physical_grasp_experiment_config(
        MULTILAYER_V1_CONFIG
    ).lift_sensor_origin_vertical_force_ramp.enabled is False
    assert load_physical_grasp_experiment_config(
        MULTILAYER_V1_CONFIG
    ).pre_lift_vertical_force_xy_admittance.enabled is False
    assert load_physical_grasp_experiment_config(
        MULTILAYER_V1_CONFIG
    ).pre_lift_xy_nyquist_suppression.enabled is False
    assert load_physical_grasp_experiment_config(
        MULTILAYER_V1_CONFIG
    ).pre_lift_quasistatic_lateral_preload_nulling.enabled is False
    assert load_physical_grasp_experiment_config(
        MULTILAYER_V1_CONFIG
    ).pre_lift_differential_finger_preload_correction.enabled is False


def test_h7_damping_section_is_strict_and_requires_h6(tmp_path):
    document = yaml.safe_load(MULTILAYER_V2_CONFIG.read_text(encoding="utf-8"))
    document["post_contact_finger_damping"]["bogus"] = 1
    changed = tmp_path / "unknown_h7.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        load_physical_grasp_experiment_config(changed)

    document = yaml.safe_load(MULTILAYER_V2_CONFIG.read_text(encoding="utf-8"))
    document["pre_lift_arm_drive_compliance"]["maximum_entry_moment_score_nm"] = 0.30
    changed = tmp_path / "weakened_compliance.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="below the 0.30"):
        load_physical_grasp_experiment_config(changed)


def test_h8_section_is_strict_and_requires_h6_h7(tmp_path):
    document = yaml.safe_load(MULTILAYER_V2_CONFIG.read_text(encoding="utf-8"))
    document["lift_x_force_admittance"]["bogus"] = 1
    changed = tmp_path / "unknown_h8.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        load_physical_grasp_experiment_config(changed)

    document = yaml.safe_load(MULTILAYER_V2_CONFIG.read_text(encoding="utf-8"))
    document["post_contact_finger_damping"]["enabled"] = False
    changed = tmp_path / "h8_without_h7.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="requires"):
        load_physical_grasp_experiment_config(changed)


def test_h11_section_is_strict_and_requires_h6_h7_h8(tmp_path):
    document = yaml.safe_load(MULTILAYER_V2_CONFIG.read_text(encoding="utf-8"))
    document["lift_finger_load_balance"]["bogus"] = 1
    changed = tmp_path / "unknown_h11.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        load_physical_grasp_experiment_config(changed)

    document = yaml.safe_load(MULTILAYER_V2_CONFIG.read_text(encoding="utf-8"))
    document["lift_x_force_admittance"]["enabled"] = False
    document["lift_xy_force_admittance"]["enabled"] = False
    changed = tmp_path / "h11_without_h8.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="H11.*requires"):
        load_physical_grasp_experiment_config(changed)


def test_h14_section_is_strict_and_mutually_exclusive_with_h8(tmp_path):
    document = yaml.safe_load(MULTILAYER_V2_CONFIG.read_text(encoding="utf-8"))
    document["lift_xy_force_admittance"]["bogus"] = 1
    changed = tmp_path / "unknown_h14.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        load_physical_grasp_experiment_config(changed)

    document = yaml.safe_load(MULTILAYER_V2_CONFIG.read_text(encoding="utf-8"))
    document["lift_x_force_admittance"]["enabled"] = True
    changed = tmp_path / "h8_and_h14.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="mutually exclusive"):
        load_physical_grasp_experiment_config(changed)

    document = yaml.safe_load(MULTILAYER_V2_CONFIG.read_text(encoding="utf-8"))
    document["lift_finger_load_balance"][
        "zero_mean_closure_coordinate_trim"
    ] = False
    changed = tmp_path / "h11_without_zero_mean.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="zero-mean"):
        load_physical_grasp_experiment_config(changed)

    document = yaml.safe_load(MULTILAYER_V2_CONFIG.read_text(encoding="utf-8"))
    document["pre_lift_arm_drive_compliance"]["enabled"] = False
    changed = tmp_path / "h7_without_h6.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="requires"):
        load_physical_grasp_experiment_config(changed)


def test_h13_section_is_strict_and_requires_h6_h7_h8_h11(tmp_path):
    document = yaml.safe_load(MULTILAYER_V2_CONFIG.read_text(encoding="utf-8"))
    document["lift_phase_arm_damping"]["bogus"] = 1
    changed = tmp_path / "unknown_h13.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        load_physical_grasp_experiment_config(changed)

    document = yaml.safe_load(MULTILAYER_V2_CONFIG.read_text(encoding="utf-8"))
    document["lift_finger_load_balance"]["enabled"] = False
    changed = tmp_path / "h13_without_h11.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="H13.*requires"):
        load_physical_grasp_experiment_config(changed)


def test_h17_section_is_strict_and_requires_the_frozen_h6_to_h13_chain(
    tmp_path,
):
    document = yaml.safe_load(MULTILAYER_V2_CONFIG.read_text(encoding="utf-8"))
    document["lift_finger_absolute_load_hold"]["bogus"] = 1
    changed = tmp_path / "unknown_h17.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        load_physical_grasp_experiment_config(changed)

    document = yaml.safe_load(MULTILAYER_V2_CONFIG.read_text(encoding="utf-8"))
    document["lift_finger_absolute_load_hold"][
        "independent_absolute_root_load_hold"
    ] = False
    changed = tmp_path / "h17_without_absolute_hold.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="hold each load independently"):
        load_physical_grasp_experiment_config(changed)

    document = yaml.safe_load(MULTILAYER_V2_CONFIG.read_text(encoding="utf-8"))
    document["lift_phase_arm_damping"]["enabled"] = False
    document["pre_lift_arm_stiffness_restoration"]["enabled"] = False
    changed = tmp_path / "h17_without_h13.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="H17.*requires"):
        load_physical_grasp_experiment_config(changed)


def test_h18_section_is_strict_and_requires_h15_interface_and_h17(tmp_path):
    document = yaml.safe_load(MULTILAYER_V2_CONFIG.read_text(encoding="utf-8"))
    document["lift_task_space_vertical_force_ramp"]["bogus"] = 1
    changed = tmp_path / "unknown_h18.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        load_physical_grasp_experiment_config(changed)

    document = yaml.safe_load(MULTILAYER_V2_CONFIG.read_text(encoding="utf-8"))
    document["lift_task_space_vertical_force_ramp"]["enabled"] = True
    document["lift_sensor_origin_vertical_force_ramp"]["enabled"] = False
    document["lift_finger_absolute_load_hold"]["enabled"] = False
    changed = tmp_path / "h18_without_h17.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="H18.*requires"):
        load_physical_grasp_experiment_config(changed)

    document = yaml.safe_load(MULTILAYER_V2_CONFIG.read_text(encoding="utf-8"))
    document["lift_task_space_vertical_force_ramp"]["enabled"] = True
    document["lift_sensor_origin_vertical_force_ramp"]["enabled"] = False
    document["pre_lift_gravity_preload_transfer"]["enabled"] = False
    changed = tmp_path / "h18_without_h15_interface.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="H18.*requires"):
        load_physical_grasp_experiment_config(changed)


def test_h19_section_is_strict_mutually_exclusive_and_requires_h17(tmp_path):
    document = yaml.safe_load(MULTILAYER_V2_CONFIG.read_text(encoding="utf-8"))
    document["lift_sensor_origin_vertical_force_ramp"]["bogus"] = 1
    changed = tmp_path / "unknown_h19.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        load_physical_grasp_experiment_config(changed)

    document = yaml.safe_load(MULTILAYER_V2_CONFIG.read_text(encoding="utf-8"))
    document["lift_sensor_origin_vertical_force_ramp"]["enabled"] = True
    changed = tmp_path / "h18_h19_together.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="mutually exclusive"):
        load_physical_grasp_experiment_config(changed)

    document = yaml.safe_load(MULTILAYER_V2_CONFIG.read_text(encoding="utf-8"))
    document["lift_task_space_vertical_force_ramp"]["enabled"] = False
    document["lift_sensor_origin_vertical_force_ramp"]["enabled"] = True
    document["pre_lift_vertical_force_xy_admittance"]["enabled"] = False
    document["lift_finger_absolute_load_hold"]["enabled"] = False
    changed = tmp_path / "h19_without_h17.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="H19.*requires"):
        load_physical_grasp_experiment_config(changed)


def test_h20_requires_exact_h18_h14_chain(tmp_path):
    document = yaml.safe_load(MULTILAYER_V2_CONFIG.read_text(encoding="utf-8"))
    document["pre_lift_vertical_force_xy_admittance"]["new_gain"] = 1.0
    changed = tmp_path / "h20_new_gain.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        load_physical_grasp_experiment_config(changed)

    document = yaml.safe_load(MULTILAYER_V2_CONFIG.read_text(encoding="utf-8"))
    document["pre_lift_vertical_force_xy_admittance"]["enabled"] = True
    document["pre_lift_xy_nyquist_suppression"]["enabled"] = False
    document["lift_x_force_admittance"]["enabled"] = True
    document["lift_xy_force_admittance"]["enabled"] = False
    changed = tmp_path / "h20_without_h14.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="H20 requires"):
        load_physical_grasp_experiment_config(changed)


def test_h21_is_strict_mutually_exclusive_and_requires_h18_h14(tmp_path):
    document = yaml.safe_load(MULTILAYER_V2_CONFIG.read_text(encoding="utf-8"))
    document["pre_lift_xy_nyquist_suppression"]["gain"] = 0.5
    changed = tmp_path / "h21_new_gain.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        load_physical_grasp_experiment_config(changed)

    document = yaml.safe_load(MULTILAYER_V2_CONFIG.read_text(encoding="utf-8"))
    document["pre_lift_vertical_force_xy_admittance"]["enabled"] = True
    changed = tmp_path / "h20_h21_together.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="mutually exclusive"):
        load_physical_grasp_experiment_config(changed)

    document = yaml.safe_load(MULTILAYER_V2_CONFIG.read_text(encoding="utf-8"))
    document["lift_x_force_admittance"]["enabled"] = True
    document["lift_xy_force_admittance"]["enabled"] = False
    changed = tmp_path / "h21_without_h14.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="H21 requires"):
        load_physical_grasp_experiment_config(changed)


def test_h22_is_strict_and_requires_the_exact_h21_chain(tmp_path):
    document = yaml.safe_load(MULTILAYER_V2_CONFIG.read_text(encoding="utf-8"))
    document["pre_lift_quasistatic_lateral_preload_nulling"]["gain"] = 0.5
    changed = tmp_path / "h22_new_gain.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        load_physical_grasp_experiment_config(changed)

    document = yaml.safe_load(MULTILAYER_V2_CONFIG.read_text(encoding="utf-8"))
    document["pre_lift_xy_nyquist_suppression"]["enabled"] = False
    changed = tmp_path / "h22_without_h21.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="H22 requires"):
        load_physical_grasp_experiment_config(changed)

    document = yaml.safe_load(MULTILAYER_V2_CONFIG.read_text(encoding="utf-8"))
    document["lift_task_space_vertical_force_ramp"]["enabled"] = False
    changed = tmp_path / "h22_without_h18.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="H21 requires|H22 requires"):
        load_physical_grasp_experiment_config(changed)


def test_h23_diagnostic_is_strict_and_requires_exact_h22_chain(tmp_path):
    document = yaml.safe_load(MULTILAYER_V2_CONFIG.read_text(encoding="utf-8"))
    document["pre_lift_differential_finger_preload_diagnostic"]["gain"] = 0.5
    changed = tmp_path / "h23_new_gain.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        load_physical_grasp_experiment_config(changed)

    document = yaml.safe_load(MULTILAYER_V2_CONFIG.read_text(encoding="utf-8"))
    document["pre_lift_differential_finger_preload_diagnostic"]["enabled"] = True
    document["pre_lift_differential_finger_preload_correction"]["enabled"] = False
    document["pre_lift_quasistatic_lateral_preload_nulling"]["enabled"] = False
    changed = tmp_path / "h23_without_h22.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="H23 diagnostic requires"):
        load_physical_grasp_experiment_config(changed)

    document = yaml.safe_load(MULTILAYER_V2_CONFIG.read_text(encoding="utf-8"))
    document["pre_lift_differential_finger_preload_diagnostic"]["enabled"] = True
    document["pre_lift_differential_finger_preload_correction"]["enabled"] = False
    document["pre_lift_centering"]["enabled"] = True
    document["pre_lift_arm_drive_compliance"]["enabled"] = False
    document["post_contact_finger_damping"]["enabled"] = False
    document["pre_lift_gravity_preload_transfer"]["enabled"] = False
    document["pre_lift_arm_stiffness_restoration"]["enabled"] = False
    document["lift_finger_absolute_load_hold"]["enabled"] = False
    document["lift_finger_load_balance"]["enabled"] = False
    document["lift_phase_arm_damping"]["enabled"] = False
    document["lift_xy_force_admittance"]["enabled"] = False
    document["lift_task_space_vertical_force_ramp"]["enabled"] = False
    document["pre_lift_xy_nyquist_suppression"]["enabled"] = False
    document["pre_lift_quasistatic_lateral_preload_nulling"]["enabled"] = False
    changed = tmp_path / "h23_reactivates_h4.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="H23 diagnostic requires"):
        load_physical_grasp_experiment_config(changed)

def test_arm_drive_compliance_section_is_strict_and_below_hard_gates(tmp_path):
    document = yaml.safe_load(MULTILAYER_V2_CONFIG.read_text(encoding="utf-8"))
    document["pre_lift_arm_drive_compliance"]["bogus"] = 1
    changed = tmp_path / "unknown_compliance.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        load_physical_grasp_experiment_config(changed)


def test_h15_section_is_strict_bounded_and_requires_h6(tmp_path):
    document = yaml.safe_load(MULTILAYER_V2_CONFIG.read_text(encoding="utf-8"))
    document["pre_lift_gravity_preload_transfer"]["bogus"] = 1
    changed = tmp_path / "unknown_h15.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        load_physical_grasp_experiment_config(changed)

    document = yaml.safe_load(MULTILAYER_V2_CONFIG.read_text(encoding="utf-8"))
    document["pre_lift_gravity_preload_transfer"][
        "maximum_gravity_feedforward_nm"
    ] = 301.0
    changed = tmp_path / "weakened_h15.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen at 300"):
        load_physical_grasp_experiment_config(changed)

    document = yaml.safe_load(MULTILAYER_V2_CONFIG.read_text(encoding="utf-8"))
    document["pre_lift_arm_drive_compliance"]["enabled"] = False
    document["post_contact_finger_damping"]["enabled"] = False
    document["lift_xy_force_admittance"]["enabled"] = False
    document["lift_finger_load_balance"]["enabled"] = False
    document["lift_phase_arm_damping"]["enabled"] = False
    changed = tmp_path / "h15_without_h6.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="H15.*requires"):
        load_physical_grasp_experiment_config(changed)


def test_h16_section_is_strict_and_requires_h6_h13(tmp_path):
    document = yaml.safe_load(MULTILAYER_V2_CONFIG.read_text(encoding="utf-8"))
    document["pre_lift_arm_stiffness_restoration"]["bogus"] = 1
    changed = tmp_path / "unknown_h16.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        load_physical_grasp_experiment_config(changed)

    document = yaml.safe_load(MULTILAYER_V2_CONFIG.read_text(encoding="utf-8"))
    document["pre_lift_arm_stiffness_restoration"][
        "expected_restored_stiffness_nm_rad"
    ] = 23000.0
    changed = tmp_path / "second_h16_parameter_set.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen"):
        load_physical_grasp_experiment_config(changed)

    document = yaml.safe_load(MULTILAYER_V2_CONFIG.read_text(encoding="utf-8"))
    document["lift_phase_arm_damping"]["enabled"] = False
    changed = tmp_path / "h16_without_h13.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="H16.*requires"):
        load_physical_grasp_experiment_config(changed)

def test_realized_state_rebase_section_is_strict_and_below_hard_gates(tmp_path):
    document = yaml.safe_load(MULTILAYER_V2_CONFIG.read_text(encoding="utf-8"))
    document["pre_lift_realized_state_rebase"]["bogus"] = 1
    changed = tmp_path / "unknown_rebase.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        load_physical_grasp_experiment_config(changed)

    document = yaml.safe_load(MULTILAYER_V2_CONFIG.read_text(encoding="utf-8"))
    document["pre_lift_realized_state_rebase"]["maximum_entry_moment_score_nm"] = 0.30
    changed = tmp_path / "weakened_rebase.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="below the 0.30"):
        load_physical_grasp_experiment_config(changed)


def test_pre_lift_centering_section_is_strict_and_preserves_hard_gates(tmp_path):
    document = yaml.safe_load(MULTILAYER_V2_CONFIG.read_text(encoding="utf-8"))
    document["pre_lift_centering"]["bogus"] = 1
    changed = tmp_path / "unknown_centering.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        load_physical_grasp_experiment_config(changed)

    document = yaml.safe_load(MULTILAYER_V2_CONFIG.read_text(encoding="utf-8"))
    document["pre_lift_centering"]["objective_moment_scale_nm"] = 0.31
    changed = tmp_path / "weakened_centering.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="preserve the 0.30"):
        load_physical_grasp_experiment_config(changed)


def test_randomization_section_loads_strictly_and_matches_yaml():
    config = load_physical_grasp_experiment_config(CONFIG)
    randomization = config.randomization
    assert randomization.plug_x_offset_m.low == pytest.approx(-0.0005)
    assert randomization.plug_yaw_deg.high == pytest.approx(5.0)
    assert randomization.finger_start_delay_steps == (0, 5, 12, 24, 36)
    assert randomization.table_static_friction.high == pytest.approx(1.00)
    assert randomization.plug_mass_scale.low == pytest.approx(0.90)
    assert randomization.lift_speed_scale.high == pytest.approx(1.10)
    assert randomization.center_of_mass_offset_m.low == pytest.approx(-0.001)


def test_single_finger_section_loads_frozen_sim_tuning_values():
    config = load_physical_grasp_experiment_config(CONFIG)
    single = config.single_finger
    assert single.threshold_label == "SIM_TUNING_ONLY"
    assert single.soft_hold_steps == 24
    assert single.minimum_release_travel_rad == pytest.approx(0.05)
    assert single.maximum_release_tracking_error_rad == pytest.approx(0.05)
    assert single.maximum_release_steps == 1200
    assert single.approach_rate_rad_s == pytest.approx(0.18)
    assert single.release_rate_rad_s == pytest.approx(0.18)
    assert config.sequential.soft_hold_window_steps == 24


def test_randomization_and_single_finger_sections_are_strict(tmp_path):
    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    del document["randomization"]["plug_yaw_deg"]
    changed = tmp_path / "missing.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        load_physical_grasp_experiment_config(changed)

    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    document["single_finger"]["bogus_key"] = 1
    changed = tmp_path / "unknown.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        load_physical_grasp_experiment_config(changed)


def test_unsolvable_friction_and_invalid_delay_sets_are_rejected(tmp_path):
    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    document["randomization"]["table_static_friction"] = [0.50, 0.60]
    changed = tmp_path / "friction.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="static"):
        load_physical_grasp_experiment_config(changed)

    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    document["randomization"]["finger_start_delay_steps"] = []
    changed = tmp_path / "delays.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty"):
        load_physical_grasp_experiment_config(changed)


def test_randomization_validation_section_loads_frozen_gates():
    config = load_physical_grasp_experiment_config(CONFIG)
    validation = config.randomization_validation
    assert validation.threshold_label == "SIM_TUNING_ONLY"
    assert validation.maximum_arm_joint_delta_rad == pytest.approx(0.05)
    assert validation.maximum_fk_position_error_m == pytest.approx(1.0e-7)
    assert validation.maximum_fk_rotation_error_rad == pytest.approx(1.0e-7)


def test_randomization_validation_section_is_strict_and_bounded(tmp_path):
    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    del document["randomization_validation"]["maximum_fk_position_error_m"]
    changed = tmp_path / "missing_validation.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        load_physical_grasp_experiment_config(changed)

    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    document["randomization_validation"]["bogus"] = 1.0
    changed = tmp_path / "unknown_validation.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        load_physical_grasp_experiment_config(changed)

    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    document["randomization_validation"]["maximum_arm_joint_delta_rad"] = 0.0
    changed = tmp_path / "zero_gate.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="maximum_arm_joint_delta_rad"):
        load_physical_grasp_experiment_config(changed)

    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    document["randomization_validation"][
        "maximum_fk_rotation_error_rad"
    ] = True
    changed = tmp_path / "bool_gate.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="numeric"):
        load_physical_grasp_experiment_config(changed)

    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    document["randomization_validation"]["threshold_label"] = "HARDWARE"
    changed = tmp_path / "label_gate.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="SIM_TUNING_ONLY"):
        load_physical_grasp_experiment_config(changed)


def test_single_finger_gates_must_stay_sim_tuning_and_bounded(tmp_path):
    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    document["single_finger"]["threshold_label"] = "HARDWARE"
    changed = tmp_path / "label.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="SIM_TUNING_ONLY"):
        load_physical_grasp_experiment_config(changed)

    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    document["single_finger"]["soft_hold_steps"] = 1
    changed = tmp_path / "hold.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="soft_hold_steps"):
        load_physical_grasp_experiment_config(changed)


def test_interval_and_delay_loaders_reject_booleans(tmp_path):
    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    document["randomization"]["plug_x_offset_m"] = [True, 0.0005]
    changed = tmp_path / "bool_interval.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="booleans"):
        load_physical_grasp_experiment_config(changed)

    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    document["randomization"]["finger_start_delay_steps"] = [0, True, 12]
    changed = tmp_path / "bool_delay.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="non-negative"):
        load_physical_grasp_experiment_config(changed)


def test_delay_set_must_contain_zero_and_unique_members(tmp_path):
    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    document["randomization"]["finger_start_delay_steps"] = [5, 12, 24]
    changed = tmp_path / "no_zero.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="contain 0"):
        load_physical_grasp_experiment_config(changed)

    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    document["randomization"]["finger_start_delay_steps"] = [0, 12, 12]
    changed = tmp_path / "duplicate.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        load_physical_grasp_experiment_config(changed)


def test_truth_or_latch_boundary_cannot_be_enabled(tmp_path):
    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    document["boundaries"]["contact_truth_in_controller_allowed"] = True
    changed = tmp_path / "changed.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="boundary"):
        load_physical_grasp_experiment_config(changed)


def test_detector_position_velocity_window_loads_strictly():
    config = load_physical_grasp_experiment_config(CONFIG)
    assert config.sequential.detector.position_velocity_window_steps == 6


def test_detector_position_velocity_window_missing_fails_closed(tmp_path):
    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    del document["detector"]["position_velocity_window_steps"]
    changed = tmp_path / "missing_window.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="position_velocity_window_steps"):
        load_physical_grasp_experiment_config(changed)


@pytest.mark.parametrize("bad", [0, -2, 25, True, 1.5])
def test_detector_position_velocity_window_rejects_bad_values(tmp_path, bad):
    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    document["detector"]["position_velocity_window_steps"] = bad
    changed = tmp_path / "bad_window.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError):
        load_physical_grasp_experiment_config(changed)
