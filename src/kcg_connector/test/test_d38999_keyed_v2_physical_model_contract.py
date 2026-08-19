from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from kcg_connector.d38999_keyed_v2_physical_model_contract import (
    REQUIRED_BENCH_IDS,
    REQUIRED_COMPONENT_IDS,
    REQUIRED_FORCE_PROXY_ROLES,
    REQUIRED_MATERIAL_ROLES,
    REQUIRED_SEQUENCE_PRECEDENCE,
    REQUIRED_SELF_COLLISION_EXCLUSIONS,
    SUCCESSOR_ASSET_NAME,
    SUCCESSOR_REVISION,
    SUCCESSOR_ROOT_PRIM,
    load_physical_model_contract,
    safe_successor_asset_output,
)


def _write_mutated(tmp_path: Path, document, name: str = "mutated.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def test_contract_is_new_immutable_identity_and_not_rejected_r10_relabeling():
    model = load_physical_model_contract()
    identity = model.document["identity"]

    assert identity["successor_revision"] == SUCCESSOR_REVISION
    assert identity["root_prim"] == SUCCESSOR_ROOT_PRIM
    assert identity["recommended_asset_name"] == SUCCESSOR_ASSET_NAME
    assert identity["predecessor_revision"] == "keyed_v3_physical_r10"
    assert identity["predecessor_role"] == (
        "rejected_runtime_detent_settle_margin_baseline_never_modify"
    )
    assert identity["overwrite_existing"] is False
    assert identity["immutable_after_a5"] is True


def test_a0_is_frozen_but_all_simulation_and_downstream_work_remains_closed():
    model = load_physical_model_contract()

    assert model.document["status"] == "A0_FROZEN_A2_AUTHORIZED"
    assert model.phase_gates == {
        "A0": "FROZEN",
        "A1": "AUDIT_COMPLETE",
        "A2": "NOT_STARTED",
        "A3": "NOT_RUN",
        "A4": "NOT_REVIEWED",
        "A5": "NOT_FROZEN",
    }
    assert model.a2_asset_authoring_allowed is True
    assert model.downstream_authorized is False
    assert model.unresolved_a0_blockers == ()
    assert model.document["authorization"]["connector_component_benches_allowed"] is False
    assert model.document["authorization"]["robot_in_loop_bench_allowed"] is False
    assert all(
        model.document["authorization"][field] is False
        for field in (
            "grasp_allowed",
            "camera_dataset_allowed",
            "visual_control_allowed",
            "insertion_allowed",
            "twist_allowed",
            "randomization_allowed",
            "training_allowed",
            "rl_allowed",
            "hardware_control_allowed",
        )
    )


def test_corrected_shell25_public_dimensions_replace_r6_hard_coding():
    geometry = load_physical_model_contract().document["public_geometry"]
    receptacle = geometry["receptacle_shell_25_class_k"]
    plug = geometry["plug_shell_25_class_k"]

    assert receptacle["mating_shell_f_diameter_mm"] == {
        "min": 39.02,
        "max": 39.42,
    }
    assert receptacle["bore_h_diameter_mm"] == {
        "min": 35.84,
        "max": 35.99,
    }
    assert plug["inner_u_diameter_mm"] == {"min": 33.99, "max": 34.34}
    assert plug["between_keys_w_diameter_mm"] == {
        "min": 35.61,
        "max": 35.77,
    }
    assert plug["overall_z_max_mm"] == 31.0
    assert 2.0 * 23.0 != pytest.approx(
        sum(receptacle["mating_shell_f_diameter_mm"].values()) / 2.0
    )
    assert 2.0 * 14.5 != pytest.approx(
        sum(plug["inner_u_diameter_mm"].values()) / 2.0
    )


def test_all_force_bearing_targets_are_physical_and_bench_bound():
    document = load_physical_model_contract().document
    components = {
        item["id"]: item for item in document["component_completeness"]
    }

    assert set(components) == set(REQUIRED_COMPONENT_IDS)
    for component in components.values():
        if component["affects_force_or_motion"]:
            assert component["a5_representation"].startswith("physical_")
        assert component["acceptance"]
        assert set(component["acceptance"]).issubset(REQUIRED_BENCH_IDS)
    assert components["pins_61"]["current_r6"] == "visual_only"
    assert components["three_start_thread"]["current_r6"] == "unmodeled"
    assert components["red_band"]["affects_force_or_motion"] is False


def test_contact_thread_sequence_and_proxy_boundary_are_explicit():
    document = load_physical_model_contract().document
    geometry = document["public_geometry"]

    assert geometry["contact_pattern_25_61"]["exact_contact_count"] == 61
    assert geometry["contact_pattern_25_61"]["same_label_pairing_required"] is True
    assert geometry["thread_shell_25"]["starts"] == 3
    assert geometry["thread_shell_25"]["pitch_mm"] == pytest.approx(2.54)
    assert geometry["thread_shell_25"]["lead_mm_per_revolution"] == pytest.approx(
        7.62
    )
    assert geometry["thread_shell_25"]["lead_in_per_revolution"] == pytest.approx(
        0.3
    )
    assert geometry["mating_sequence"]["events"][0] == (
        "five_key_polarization"
    )
    assert geometry["mating_sequence"]["events"][-1] == (
        "shell_to_shell_metal_bottoming"
    )
    assert tuple(
        tuple(edge)
        for edge in geometry["mating_sequence"]["required_precedence_edges"]
    ) == REQUIRED_SEQUENCE_PRECEDENCE
    boundaries = document["physical_proxy_boundaries"]
    assert "removable_size20_socket_contact" in boundaries[
        "no_public_force_curve_available_for"
    ]
    assert boundaries["mass_policy"]["origin"] == (
        "inherited_proxy_target_mass_not_hardware_truth"
    )


def test_figure3_axial_chain_does_not_treat_plug_y_or_z_as_direct_B_coordinates():
    axial = load_physical_model_contract().document["public_geometry"][
        "axial_interface"
    ]
    receptacle = axial["receptacle_from_B_depth_mm"]
    plug = axial["plug_control_chain_mm"]
    nominal = plug["nominal_for_r7"]

    assert axial["current_pair_contact_polarity"]["fixed_receptacle"] == "pin"
    assert axial["current_pair_contact_polarity"]["loose_plug"] == "socket"
    assert receptacle["M_pin_tip"]["min"] == pytest.approx(9.50)
    assert receptacle["N_pin_insert_face"]["max"] == pytest.approx(15.46)
    assert receptacle["derived_pin_exposed_length"] == {
        "min": 4.50,
        "max": 5.96,
        "source_kind": "PUBLIC_SPEC_DERIVED",
    }
    assert plug["L_B_to_common_rear_control_plane"]["min"] == pytest.approx(15.01)
    assert plug["Y_socket_spring_touch_to_rear_plane"]["min"] == pytest.approx(
        12.45
    )
    assert nominal["B_to_socket_spring_touch"] == pytest.approx(
        nominal["L_B_to_common_rear_control_plane"]
        - nominal["Y_socket_spring_touch_to_rear_plane"]
    )
    assert axial["metal_bottoming"][
        "determined_by_physical_collision_not_pose_or_boolean"
    ] is True
    assert axial["red_band_coverage"]["physical_pass_gate"] is False


def test_force_transmitting_proxy_roles_are_bounded_passive_and_not_hardware_truth():
    boundaries = load_physical_model_contract().document[
        "physical_proxy_boundaries"
    ]
    roles = boundaries["force_parameters"]

    assert set(roles) == set(REQUIRED_FORCE_PROXY_ROLES)
    for role_name in (
        "socket_contact_sleeves_61",
        "shell_spring_fingers",
        "interfacial_pin_barriers_61",
        "peripheral_seal",
        "anti_decoupling_detent",
    ):
        role = roles[role_name]
        assert role["compliant_contact_acceleration_spring"] is False
        for field in ("stiffness_n_m", "damping_n_s_m"):
            values = role[field]
            assert 0.0 < values["min"] <= values["nominal"] <= values["max"]
    assert roles["socket_contact_sleeves_61"]["petal_count_per_socket"] == 6
    assert roles["socket_contact_sleeves_61"]["force_curve_claim"] == (
        "simulation_proxy_only"
    )
    assert roles["anti_decoupling_detent"]["initial_forward_component_proxy_target_nm"][
        "max"
    ] < 0.30
    assert roles["anti_decoupling_detent"][
        "complete_pair_public_torque_limits_are_not_detent_limits"
    ] is True
    assert roles["peripheral_seal"]["physical_deflection_range_m"] == {
        "min": 0.000280,
        "nominal": 0.000435,
        "max": 0.000590,
    }
    assert roles["coupling_thread"]["runtime_engagement_switch_allowed"] is False
    assert roles["coupling_thread"]["software_axial_pose_write_allowed"] is False


def test_materials_are_role_specific_and_do_not_blanket_bind_the_plug():
    materials = load_physical_model_contract().document["material_roles"]

    assert materials["binding_policy"] == (
        "explicit_role_metadata_only_no_path_blanket_binding"
    )
    assert set(materials["roles"]) == set(REQUIRED_MATERIAL_ROLES)
    assert materials["roles"]["fingertip_pad"]["static_friction"] == 1.40
    assert materials["roles"]["plug_shell_and_keys"]["static_friction"] == 0.35
    assert materials["roles"]["coupling_thread"]["static_friction"] == 0.20
    assert materials["roles"]["table"]["static_friction"] == 0.90
    assert materials["robot_collision_link_roles"]["fingertip_pad"] == [
        "f1Link3",
        "f2Link2",
        "f3Link3",
    ]
    assert materials["unassigned_collision_prim_count_allowed"] == 0


def test_robot_sensor_and_fixture_proxy_choices_are_frozen_now():
    document = load_physical_model_contract().document
    sensors = document["sensor_and_robot_boundaries"]
    decisions = {
        item["name"]: item["decision"]
        for item in document["a0_source_freeze"]["resolved_decisions"]
    }

    assert sensors["robot_inertia"]["active_path"] == (
        "src/iiwa_description/urdf/handarm.urdf.xacro"
    )
    assert sensors["robot_inertia"]["arm_source"] == (
        "src/iiwa_description/urdf/iiwa14.xacro"
    )
    assert sensors["robot_inertia"]["hand_source"] == (
        "src/iiwa_description/urdf/hand.xacro"
    )
    assert sensors["robot_inertia"]["alternate_local_files_authorized"] is False
    assert sensors["finger_channels"]["fingertip_tactile_exists"] is False
    assert sensors["wrist_ft"]["physical_sensor_body_exists"] is False
    assert sensors["cameras"]["physical_housing_exists"] is False
    assert sensors["robot_inertia"]["expected_physical_collision_link_count"] == 17
    assert sensors["grasp_tcp"]["handbase_translation_m"] == [0.0, 0.0, 0.400]
    assert "explicit receptacle-to-fixture-to-world" in decisions["fixture_load_path"]
    assert "runtime engagement switch" in decisions["coupling_thread_architecture"]
    assert "all 61 pins" in decisions["electrical_contact_architecture"]


def test_self_collision_starts_with_only_16_adjacent_pairs():
    document = load_physical_model_contract().document
    policy = document["self_collision_filter"]
    normalized = {
        tuple(sorted(pair)) for pair in policy["excluded_pairs"]
    }

    assert policy["policy"] == "topology_adjacent_only_initially"
    assert policy["sampled_never_pairs_authorized"] is False
    assert normalized == set(REQUIRED_SELF_COLLISION_EXCLUSIONS)
    assert len(normalized) == 16
    assert tuple(sorted(("f1Link1", "f2Link1"))) not in normalized
    assert tuple(sorted(("f1Link3", "iiwa_link_2"))) not in normalized


def test_solver_api_names_values_and_readback_gate_are_frozen():
    solver = load_physical_model_contract().document["solver_profile"]
    authored = solver["authored_attribute_contract"]

    assert solver["schema_versions"] == {
        "omni_usd_schema_physx": "110.1.13",
        "omni_usd_schema_newton": "1.2.1",
    }
    assert authored["physics_scene"]["physxScene:solverType"] == "TGS"
    assert authored["physics_scene"]["physxScene:enableEnhancedDeterminism"] is True
    assert authored["dynamic_rigid_bodies"][
        "physxRigidBody:solverPositionIterationCount"
    ] == 32
    assert authored["fine_connector_colliders"][
        "physxCollision:contactOffset"
    ] == pytest.approx(1.0e-5)
    assert authored["materials"]["physxMaterial:frictionCombineMode"] == "max"
    assert authored["compliant_materials"][
        "physxMaterial:compliantContactAccelerationSpring"
    ] is False
    assert authored["compliant_materials"]["authoring_api"] == (
        "PhysxSchema.PhysxMaterialAPI"
    )
    assert authored["nut_body_D6_joint"]["physics:collisionEnabled"] is True
    assert authored["robot_articulation"]["newton:selfCollisionEnabled"] is True
    assert solver["resolved_api_readback_status"] == "REQUIRED_DURING_A2_AND_A3"


def test_component_torque_spec_cannot_relax_robot_safety_gate():
    safety = load_physical_model_contract().document["safety_contract"]

    assert safety["robot_in_loop"]["formal_perpendicular_moment_max_nm"] == 0.30
    assert safety["robot_in_loop"]["first_exceedance_fails_episode"] is True
    assert (
        safety["robot_in_loop"]["may_be_relaxed_by_component_specification"]
        is False
    )
    assert safety["connector_component_bench"][
        "complete_pair_min_disengagement_nm"
    ] == 0.6
    assert safety["connector_component_bench"][
        "complete_pair_max_coupling_or_disengagement_nm"
    ] == 4.6
    assert safety["connector_component_bench"][
        "limits_apply_to_complete_plug_receptacle_pair_not_detent_alone"
    ] is True
    assert safety["connector_component_bench"]["may_run_without_robot_only"] is True


def test_bottoming_seal_thread_and_bearing_share_one_consistent_geometry_chain():
    document = load_physical_model_contract().document
    axial = document["public_geometry"]["axial_interface"]
    bottoming = axial["metal_bottoming"]
    stack = bottoming["nominal_stack_mm"]
    thread = axial["thread_axial_controls"]["r7_force_transmitting_geometry"]
    bearing = document["physical_proxy_boundaries"]["nut_body_bearing"]

    assert stack["seal_physical_deflection_at_bottoming"] == pytest.approx(
        bottoming["full_mate_datum_B_separation_mm"]
        - stack["seal_first_touch_separation"]
    )
    assert thread["nominal_active_axial_travel_mm"] == pytest.approx(7.62)
    assert (
        thread["nominal_bottoming_datum_B_separation_mm"]
        - thread["nominal_thread_entry_datum_B_separation_mm"]
    ) == pytest.approx(7.62)
    assert bearing["physics_collision_enabled"] is True
    assert bearing["physical_shoulder_contact_endplay_m"] == {
        "low": -0.00005,
        "high": 0.00005,
    }


def test_series_iii_seal_cannot_regress_to_unrelated_0p61mm_value(tmp_path):
    source = load_physical_model_contract()
    document = deepcopy(source.document)
    document["physical_proxy_boundaries"]["force_parameters"][
        "peripheral_seal"
    ]["physical_deflection_range_m"]["nominal"] = 0.00061

    with pytest.raises(ValueError, match="peripheral seal.*nominal is outside|seal deflection range changed"):
        load_physical_model_contract(_write_mutated(tmp_path, document))


def test_detent_stiffness_must_still_close_its_own_torque_budget(tmp_path):
    source = load_physical_model_contract()
    document = deepcopy(source.document)
    document["physical_proxy_boundaries"]["force_parameters"][
        "anti_decoupling_detent"
    ]["stiffness_n_m"]["nominal"] = 60000.0

    with pytest.raises(ValueError, match="anti-decoupling detent.*nominal is outside|detent per-follower stiffness range changed"):
        load_physical_model_contract(_write_mutated(tmp_path, document))


def test_nut_joint_cannot_silence_physical_shoulders_or_detent(tmp_path):
    source = load_physical_model_contract()
    document = deepcopy(source.document)
    document["physical_proxy_boundaries"]["nut_body_bearing"][
        "physics_collision_enabled"
    ] = False

    with pytest.raises(ValueError, match="cylindrical bearing architecture changed"):
        load_physical_model_contract(_write_mutated(tmp_path, document))


def test_barrier_effective_compliance_is_divided_over_24_angular_wedges(tmp_path):
    source = load_physical_model_contract()
    barriers = source.document["physical_proxy_boundaries"]["force_parameters"][
        "interfacial_pin_barriers_61"
    ]
    assert barriers["authored_per_angular_wedge_stiffness_n_m"]["nominal"] * 24 == pytest.approx(
        barriers["stiffness_n_m"]["nominal"]
    )

    document = deepcopy(source.document)
    document["physical_proxy_boundaries"]["force_parameters"][
        "interfacial_pin_barriers_61"
    ]["authored_per_angular_wedge_stiffness_n_m"]["nominal"] = 250.0
    with pytest.raises(
        ValueError,
        match="outside its A3 range|per-wedge stiffness is not exactly 1/24",
    ):
        load_physical_model_contract(_write_mutated(tmp_path, document))


def test_thread_blueprint_cannot_change_piece_count_or_filter_by_start(tmp_path):
    source = load_physical_model_contract()
    document = deepcopy(source.document)
    document["a2_collision_authoring_blueprint"]["thread"]["segments_per_start"] = 180
    with pytest.raises(ValueError, match="thread authoring blueprint"):
        load_physical_model_contract(_write_mutated(tmp_path, document, "thread.yaml"))

    document = deepcopy(source.document)
    document["a2_collision_authoring_blueprint"]["thread"][
        "same_start_index_filtering_allowed"
    ] = True
    with pytest.raises(ValueError, match="thread authoring blueprint"):
        load_physical_model_contract(_write_mutated(tmp_path, document, "thread_filter.yaml"))


def test_connector_internal_root_filter_cannot_hide_detent_or_shoulders(tmp_path):
    source = load_physical_model_contract()
    document = deepcopy(source.document)
    document["a2_collision_authoring_blueprint"]["filtering"][
        "FilteredPairsAPI_on_BodyAssembly_or_CouplingNut_root_allowed"
    ] = True
    with pytest.raises(ValueError, match="leaf-only connector collision filtering"):
        load_physical_model_contract(_write_mutated(tmp_path, document))


def test_socket_chamfer_and_spring_keyway_clearance_are_geometry_locked(tmp_path):
    source = load_physical_model_contract()
    document = deepcopy(source.document)
    detail = document["public_geometry"]["contact_pattern_25_61"][
        "series_III_size20_interface_detail"
    ]["r7_collision_blueprint"]["socket_entry"]
    detail["axial_profile_bands"][1]["depth_end_mm"] = 0.810
    with pytest.raises(ValueError, match="socket-entry convex axial profile bands"):
        load_physical_model_contract(_write_mutated(tmp_path, document, "chamfer.yaml"))

    document = deepcopy(source.document)
    document["a2_collision_authoring_blueprint"]["spring_fingers"][
        "finger_phase_formula_deg"
    ] = "15+30*segment_index"
    with pytest.raises(ValueError, match="spring fingers blueprint"):
        load_physical_model_contract(_write_mutated(tmp_path, document, "spring.yaml"))


def test_resolved_readback_requires_collider_identity_bounds_and_response(tmp_path):
    source = load_physical_model_contract()
    required = set(source.document["solver_profile"]["resolved_readback_required_fields"])
    assert {
        "collider_index", "local_bounds", "world_bounds_at_canonical_pose",
        "physics_approximation", "responseRole", "filteredPairs_sources",
    }.issubset(required)
    pair_required = set(
        source.document["solver_profile"][
            "resolved_family_pair_readback_required_fields"
        ]
    )
    assert {
        "decision_rule_id", "left_family", "right_family",
        "resolved_decision", "matched_final_rule_count",
    }.issubset(pair_required)

    document = deepcopy(source.document)
    document["solver_profile"]["resolved_readback_required_fields"].remove(
        "collider_index"
    )
    with pytest.raises(ValueError, match="readback field inventory"):
        load_physical_model_contract(_write_mutated(tmp_path, document))


def test_robot_source_mesh_and_canonical_camera_identity_are_locked(tmp_path):
    source = load_physical_model_contract()
    document = deepcopy(source.document)
    rows = document["realized_robot_hand_fixture_blueprint"]["collision_inventory"][
        "per_link_source_inventory"
    ]
    rows[0]["mesh_uri"] = "wrong.stl"
    with pytest.raises(ValueError, match="collision source or role changed"):
        load_physical_model_contract(_write_mutated(tmp_path, document, "robot_mesh.yaml"))

    document = deepcopy(source.document)
    document["realized_robot_hand_fixture_blueprint"]["semantic_frames_and_sensors"][
        "cameras"
    ]["duplicate_live_view_camera_prims_allowed"] = True
    with pytest.raises(ValueError, match="camera blueprint"):
        load_physical_model_contract(_write_mutated(tmp_path, document, "camera.yaml"))


def test_a0_cannot_be_claimed_frozen_while_source_blockers_remain(tmp_path):
    source = load_physical_model_contract()
    document = deepcopy(source.document)
    document["a0_source_freeze"]["unresolved_source_mappings"].append(
        {"id": "DELIBERATE_TEST_BLOCKER", "blocking": True}
    )

    with pytest.raises(ValueError, match="source blockers remain"):
        load_physical_model_contract(_write_mutated(tmp_path, document))


def test_force_bearing_component_cannot_fall_back_to_visual_only(tmp_path):
    source = load_physical_model_contract()
    document = deepcopy(source.document)
    for component in document["component_completeness"]:
        if component["id"] == "three_start_thread":
            component["a5_representation"] = "visual_only"
            break

    with pytest.raises(ValueError, match="lacks a physical A5 target"):
        load_physical_model_contract(_write_mutated(tmp_path, document))


def test_downstream_permission_cannot_open_before_a5(tmp_path):
    source = load_physical_model_contract()
    document = deepcopy(source.document)
    document["authorization"]["grasp_allowed"] = True

    with pytest.raises(ValueError, match="authorization latch"):
        load_physical_model_contract(_write_mutated(tmp_path, document))


def test_fingerprint_metadata_is_rejected_without_computing_it(tmp_path):
    source = load_physical_model_contract()
    document = deepcopy(source.document)
    document["identity"]["checksum"] = "not-computed"

    with pytest.raises(ValueError, match="fingerprint metadata"):
        load_physical_model_contract(_write_mutated(tmp_path, document))


def test_successor_asset_output_is_now_allowed_only_at_the_frozen_new_path(tmp_path):
    model = load_physical_model_contract()
    identity = model.document["identity"]
    expected = (
        Path(__file__).resolve().parents[3]
        / identity["recommended_asset_directory"]
        / SUCCESSOR_ASSET_NAME
    ).resolve()

    if expected.exists():
        with pytest.raises(FileExistsError, match="refusing to overwrite"):
            safe_successor_asset_output(expected, model)
    else:
        assert safe_successor_asset_output(expected, model) == expected
    with pytest.raises(ValueError, match="must be exactly"):
        safe_successor_asset_output(tmp_path / SUCCESSOR_ASSET_NAME, model)
