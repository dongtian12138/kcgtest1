from pathlib import Path

import yaml


CONTRACT_PATH = (
    Path(__file__).parents[1] / "config" / "wrist_ft_v1_contract.yaml"
)


def contract():
    with CONTRACT_PATH.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def test_wrist_ft_design_is_disabled_and_preserves_historical_v0_shape():
    document = contract()
    compatibility = document["compatibility"]
    assert document["status"] == "design_only"
    assert document["enabled"] is False
    assert compatibility == {
        "active_interface_version": "kcg_connector_twist_residual_v0",
        "active_action_size": 4,
        "active_observation_size": 24,
        "modifies_active_interface": False,
        "modifies_robot_asset": False,
        "modifies_tcp": False,
    }
    assert compatibility["active_action_size"] == 4
    assert compatibility["active_observation_size"] == 24


def test_virtual_boundary_reuses_zero_transform_hand2arm_joint():
    boundary = contract()["virtual_measurement_boundary"]
    assert boundary["mode"] == "existing_fixed_joint_reaction_wrench"
    assert boundary["available_by_default"] is True
    assert boundary["parent_link"] == "iiwa_link_ee"
    assert boundary["measurement_joint"] == "hand2arm"
    assert boundary["child_link"] == "handbase_link"
    assert boundary["sensor_frame"] == "handbase_link"
    assert boundary["preserve_task_tcp"] == "grasp_tcp"
    assert all(
        boundary[name] is True
        for name in ("zero_translation", "zero_rotation", "zero_thickness")
    )
    assert boundary["zero_mass"] is True
    assert all(
        boundary[name] is False
        for name in (
            "adds_links",
            "adds_joints",
            "adds_geometry",
            "changes_kinematics",
        )
    )


def test_ros_and_isaac_wrench_order_is_exactly_six_dimensional():
    document = contract()
    expected = [
        "force.x",
        "force.y",
        "force.z",
        "torque.x",
        "torque.y",
        "torque.z",
    ]
    assert document["ros2_control"]["state_interfaces"] == expected
    assert document["isaac"]["reaction_wrench_order"] == expected
    assert document["isaac"]["reaction_wrench_joint"] == "hand2arm"
    assert document["isaac"]["reaction_wrench_frame"] == (
        "child_joint_frame"
    )
    assert document["isaac"]["merge_fixed_joints_required"] is False
    assert document["isaac"]["reaction_wrench_joint_index_offset"] == 1
    assert document["isaac"]["privileged_contact_wrench_in_actor"] is False


def test_recorded_virtual_boundary_smoke_preserved_tcp_and_asset():
    evidence = contract()["validation_evidence"]
    assert evidence["marker"] == "ISAAC VIRTUAL WRIST FT PASSED"
    assert evidence["reaction_row_index"] == (
        evidence["metadata_joint_index"] + 1
    )
    assert evidence["selected_wrench_shape"] == [1, 6]
    assert evidence["raw_frame"] == "handbase_link"
    assert evidence["gravity_force_response_n"] > 0.1
    assert evidence["tcp_offset_m"] == [0.0, 0.0, 0.4]
    assert evidence["tcp_offset_change_m"] < 1.0e-12
    assert evidence["asset_file_count"] == 9


def test_recorded_six_axis_calibration_fixes_raw_sign_without_permutation():
    document = contract()
    evidence = document["validation_evidence"]["axis_calibration"]
    expected = [
        [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
    ]
    assert document["frames"]["canonical_from_isaac_raw"] == (
        "wrench_canonical = -wrench_raw"
    )
    assert document["frames"][
        "canonical_from_isaac_raw_axis_permutation"
    ] is False
    assert evidence["canonical_from_raw_sign_permutation"] == expected
    assert 0.99 < evidence["minimum_absolute_gain"] <= 1.0
    assert evidence["maximum_absolute_gain"] < 1.01
    assert evidence["maximum_same_kind_cross_axis_ratio"] < 0.001
    assert evidence["maximum_odd_symmetry_error_ratio"] < 0.001
    assert evidence["maximum_half_full_linearity_error_ratio"] < 0.001
    assert evidence["teardown_articulation_assignment_warning_count"] == 0
    assert evidence["runtime_joint_warning_count"] == 0
    assert evidence["traceback_count"] == 0


def test_joint_torque_estimator_declares_rank_and_contact_limits():
    estimator = contract()["joint_torque_wrench_estimator"]
    assert estimator["integration_enabled"] is False
    assert estimator["implementation_module"] == (
        "kcg_connector.joint_torque_wrench"
    )
    assert estimator["implementation_function"] == "estimate_tool_wrench"
    assert estimator["jacobian_shape"] == [6, 7]
    assert estimator["required_task_rank"] == 6
    assert estimator["nullspace_dimension_at_full_rank"] == 1
    assert estimator["monitor_nullspace_projection_residual"] is True
    assert estimator["contact_distribution_observable"] is False
    assert (
        estimator["arm_link_contact_localization_from_single_wrench"]
        is False
    )
    assert estimator["deployed_solver"] == "weighted_damped_least_squares"
    assert estimator["condition_number_metric"] == (
        "sqrt(W) * transpose(J) * diag(wrench_scales)"
    )
    assert len(estimator["source_joint_names"]) == 7
    assert len(set(estimator["source_joint_names"])) == 7


def test_retired_residual_v1_contract_shape_remains_auditable():
    residual_v1 = contract()["residual_v1"]
    appended = tuple(residual_v1["appended_observation_names"])
    assert residual_v1["action_size"] == 4
    assert residual_v1["base_interface_version"] == (
        "kcg_connector_twist_residual_v0"
    )
    assert residual_v1["observation_size"] == 24 + len(appended)
    assert residual_v1["observation_size"] == 30
    assert len(appended) == len(set(appended)) == 6
    assert residual_v1["observation_source"] == (
        "compensated_connector_task_frame_wrench"
    )


def test_unknown_operational_limits_prevent_accidental_activation():
    document = contract()
    estimator = document["joint_torque_wrench_estimator"]
    scales = document["residual_v1"]["normalization_scales"]
    limits = document["safety_limits"]
    assert all(value is None for value in scales.values())
    assert all(value is None for value in limits.values())
    assert estimator["damping"] is None
    assert estimator["wrench_scales"] is None
    assert estimator["maximum_condition_number"] is None
    assert estimator["maximum_projection_residual_nm"] is None
    assert estimator["calibrated_covariance"] is None
    assert document["required_before_enable"]


def test_calibration_updates_are_separated_from_contact_phases():
    compensation = contract()["compensation"]
    forbidden = set(compensation["tare_forbidden_phases"])
    bias_allowed = set(
        compensation["electronic_bias_update_allowed_phases"]
    )
    payload_allowed = set(compensation["payload_capture_allowed_phases"])
    assert forbidden == {"INSERT", "ENGAGE", "SCREW", "HOLD"}
    assert bias_allowed == {"HOME_FREE_SPACE_EMPTY_HAND"}
    assert payload_allowed == {"POST_GRASP_FREE_SPACE"}
    assert not bias_allowed.intersection(forbidden)
    assert not payload_allowed.intersection(forbidden)


def test_success_cannot_be_declared_from_wrist_torque_alone():
    required = set(contract()["success_requires_all"])
    assert {
        "measured_nut_progress",
        "measured_axial_progress",
        "connector_depth_or_gap_in_tolerance",
        "tightening_torque_in_terminal_band",
        "lateral_force_and_bending_in_tolerance",
        "finger_base_torques_loaded_and_stable",
        "stable_hold",
    } == required
