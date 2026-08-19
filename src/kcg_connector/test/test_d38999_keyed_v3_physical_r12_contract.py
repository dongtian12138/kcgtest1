from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from kcg_connector.d38999_keyed_v2_a2_readback_result import (
    _trusted_collider_inventory,
    _trusted_family_algebra,
)
from kcg_connector.d38999_keyed_v2_physical_acceptance import (
    load_physical_acceptance_matrix,
)
from kcg_connector.d38999_keyed_v2_physical_model_contract import (
    load_physical_model_contract,
)
from kcg_connector.d38999_keyed_v3_physical_r12_acceptance import (
    load_r12_physical_acceptance_matrix,
)
from kcg_connector.d38999_keyed_v3_physical_r12_contract import (
    R12_EXPECTED_ANALYTIC_COLLIDER_COUNT,
    R12_EXPECTED_COLLIDER_COUNT,
    R12_EXPECTED_MESH_COLLIDER_COUNT,
    candidate_model,
    load_r12_physical_model_contract,
)
from kcg_connector.d38999_tabletop_scene import load_d38999_tabletop_scene


def test_r12_exact_collider_partition_and_no_old_segment_paths():
    model = load_r12_physical_model_contract()
    inventory = _trusted_collider_inventory(model)
    types = [row["typeName"] for row in inventory.values()]
    assert len(inventory) == R12_EXPECTED_COLLIDER_COUNT == 14761
    assert types.count("Mesh") == R12_EXPECTED_MESH_COLLIDER_COUNT == 14684
    assert len(types) - types.count("Mesh") == R12_EXPECTED_ANALYTIC_COLLIDER_COUNT == 77
    assert types.count("Sphere") == 12
    assert types.count("Cylinder") == 65
    repaired = {
        "detent_followers_3": (3, "Sphere"),
        "fixed_metal_stop_48": (1, "Cylinder"),
        "plug_metal_stop_48": (3, "Sphere"),
        "shoulder_positive_body0_48": (1, "Cylinder"),
        "shoulder_positive_body1_48": (3, "Sphere"),
        "shoulder_negative_body0_48": (1, "Cylinder"),
        "shoulder_negative_body1_48": (3, "Sphere"),
    }
    for family, (count, type_name) in repaired.items():
        rows = [row for row in inventory.values() if row["family"] == family]
        assert len(rows) == count
        assert {row["typeName"] for row in rows} == {type_name}
        assert not any("/Seg_" in row["prim_path"] for row in rows)
    pairs, filters = _trusted_family_algebra(model)
    assert len(pairs) == 406
    assert len(filters) == 387


def test_r12_three_repair_geometries_are_exact():
    document = load_r12_physical_model_contract().document
    blueprint = document["a2_collision_authoring_blueprint"]
    detent = blueprint["anti_decoupling_detent"]
    assert detent["follower_phases_deg"] == [-4.491137, 115.508863, 235.508863]
    assert detent["follower_radius_m"] == 0.000102
    assert detent["follower_center_radius_m"] == 0.022076
    assert detent["follower_local_center_z_m"] == 0.020000
    assert detent["old_radial_tangent_box_collider_count"] == 0
    assert detent["accepted_probe_measured_peak_nm"] == 0.060021022609
    assert abs(
        detent["accepted_probe_measured_peak_nm"]
        - detent["nominal_forward_component_target_nm"]
    ) <= detent["forward_component_absolute_tolerance_nm"]

    shoulder = document["physical_proxy_boundaries"]["nut_body_bearing"][
        "physical_shoulder_collision_geometry"
    ]
    assert shoulder["body0_cap_radius_m"] == 0.01960
    assert shoulder["body0_cap_axial_thickness_m"] == 0.00030
    assert shoulder["body1_sphere_radius_m"] == 0.00050
    assert shoulder["body1_sphere_distribution_radius_m"] == 0.01850
    assert shoulder["positive_transZ_stop"]["body0_collider_center_local_z_m"] == 0.03015
    assert shoulder["positive_transZ_stop"]["body1_sphere_center_local_z_m"] == 0.02945
    assert shoulder["negative_transZ_stop"]["body0_collider_center_local_z_m"] == 0.00985
    assert shoulder["negative_transZ_stop"]["body1_sphere_center_local_z_m"] == 0.01055

    bottom = blueprint["metal_bottoming"]
    assert bottom["fixed_cap_radius_m"] == 0.01695
    assert bottom["fixed_cap_center_local_z_m"] == 0.00015
    assert bottom["plug_sphere_radius_m"] == 0.00050
    assert bottom["plug_sphere_distribution_radius_m"] == 0.01600
    assert bottom["plug_sphere_center_local_z_m"] == 0.01555
    assert bottom["validated_no_contact_separation_m"] == 0.015049
    assert bottom["validated_contact_separation_m"] == 0.015051
    assert bottom["old_segmented_collider_count"] == 0


def test_r12_protected_thread_joint_controller_and_authorization_are_unchanged():
    r11 = load_physical_model_contract().document
    r12 = load_r12_physical_model_contract().document
    assert r12["a2_collision_authoring_blueprint"]["thread"] == r11[
        "a2_collision_authoring_blueprint"
    ]["thread"]
    assert r12["solver_profile"]["authored_attribute_contract"][
        "nut_body_D6_joint"
    ] == r11["solver_profile"]["authored_attribute_contract"]["nut_body_D6_joint"]
    r11_acceptance = load_physical_acceptance_matrix().document
    r12_acceptance = load_r12_physical_acceptance_matrix().document
    assert r12_acceptance["benches"] == r11_acceptance["benches"]
    driver = r12_acceptance["benches"]["P1"]["inputs"]["component_driver_profile"]
    assert driver["torque_component_limit_nm"] == 0.30
    assert driver["angular_velocity_gain_nm_s_rad"] == 0.01
    for field in (
        "insertion_allowed",
        "twist_allowed",
        "randomization_allowed",
        "training_allowed",
        "rl_allowed",
        "hardware_control_allowed",
    ):
        assert r12["authorization"][field] is False


def test_r12_scene_profile_has_two_level_fixed_threshold_and_counts():
    path = (
        Path(__file__).resolve().parents[1]
        / "config/d38999_keyed_v3_tabletop_scene_r12_v1.yaml"
    )
    scene = load_d38999_tabletop_scene(path)
    assert scene.asset_profile.expected_body_collider_count == 7438
    assert scene.asset_profile.expected_nut_collider_count == 204
    assert scene.physics.maximum_fixed_translation_drift_m == 5.0e-6


def test_r12_contract_rejects_any_unapproved_mutation(tmp_path):
    model = load_r12_physical_model_contract()
    document = deepcopy(model.document)
    document["a2_collision_authoring_blueprint"]["thread"][
        "segment_count_per_start"
    ] = 359
    path = tmp_path / "mutated-r12.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="authorized r11 transformation"):
        load_r12_physical_model_contract(path)


def test_candidate_02_only_replaces_thread_rail_piece_representation():
    base = load_r12_physical_model_contract()
    candidate = candidate_model(base, 2)
    base_thread = base.document["a2_collision_authoring_blueprint"]["thread"]
    thread = candidate.document["a2_collision_authoring_blueprint"]["thread"]
    for field in (
        "start_count",
        "start_phases_deg",
        "segments_per_start",
        "segment_angle_deg",
        "contact_surface_equation_mm",
        "axial_interval_m",
        "pitch_radius_m",
        "follower_radial_interval_m",
        "follower_primitive_recipe_id",
        "follower_local_center_angles_deg",
        "coupling_joint_rotZ_sign_for_positive_coupling",
    ):
        assert thread[field] == base_thread[field]
    inventory = _trusted_collider_inventory(candidate)
    rail_rows = [
        row for row in inventory.values() if row["family"] == "thread_rails_3"
    ]
    follower_rows = [
        row for row in inventory.values() if row["family"] == "thread_followers_3"
    ]
    assert len(rail_rows) == 1080
    assert {row["typeName"] for row in rail_rows} == {"Capsule"}
    assert len(follower_rows) == 3
    assert {row["typeName"] for row in follower_rows} == {"Mesh"}
    assert thread["rail_capsule_radius_m"] == 0.00015


def test_candidate_03_only_adds_derived_rounded_spring_targets():
    base = load_r12_physical_model_contract()
    candidate_02 = candidate_model(base, 2)
    candidate_03 = candidate_model(base, 3)
    spring_02 = candidate_02.document["a2_collision_authoring_blueprint"][
        "spring_fingers"
    ]
    spring_03 = candidate_03.document["a2_collision_authoring_blueprint"][
        "spring_fingers"
    ]
    for field in (
        "finger_count",
        "finger_path_template",
        "target_piece_path_template",
        "target_segment_count",
        "target_phase_formula_deg",
        "target_tangential_width_m",
        "target_inner_bore_radius_m",
        "target_local_z_interval_m",
        "finger_material_role",
        "target_material_role",
        "nominal_response_k_n_m",
        "nominal_response_c_n_s_m",
        "nominal_resolved_dynamic_friction_coefficient",
    ):
        assert spring_03[field] == spring_02[field]
    assert spring_03["target_capsule_radius_m"] == 0.0008
    assert spring_03["target_capsule_center_radius_m"] == 0.0187575
    assert spring_03["target_capsule_center_local_z_m"] == 0.003
    assert spring_03["target_capsule_cylinder_height_m"] == 0.0044
    inventory = _trusted_collider_inventory(candidate_03)
    targets = [
        row
        for row in inventory.values()
        if row["family"] == "receptacle_bore_targets_12"
    ]
    assert len(targets) == 12
    assert {row["typeName"] for row in targets} == {"Capsule"}


def test_candidate_04_has_one_continuous_spring_target_and_exact_count_delta():
    base = load_r12_physical_model_contract()
    candidate = candidate_model(base, 4)
    spring = candidate.document["a2_collision_authoring_blueprint"][
        "spring_fingers"
    ]
    assert spring["finger_count"] == 12
    assert spring["target_segment_count"] == 1
    assert spring["target_cylinder_radius_m"] == 0.0179575
    assert spring["target_cylinder_height_m"] == 0.006
    assert spring["target_cylinder_center_local_z_m"] == 0.003
    inventory = _trusted_collider_inventory(candidate)
    assert len(inventory) == 14750
    targets = [
        row
        for row in inventory.values()
        if row["family"] == "receptacle_bore_targets_12"
    ]
    fingers = [
        row for row in inventory.values() if row["family"] == "spring_fingers_12"
    ]
    assert len(targets) == 1
    assert {row["typeName"] for row in targets} == {"Cylinder"}
    assert len(fingers) == 12
