"""Deterministic r12 structural repair contract derived from immutable r11.

Only the three collision representations proven faulty by the retained r11
evidence are changed: detent followers, nut/body shoulders, and metal
bottoming.  Everything else is inherited byte-for-value from the validated
r11 contract.  This module is pure CPU and imports neither Isaac Sim nor USD.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml

from kcg_connector.d38999_keyed_v2_physical_model_contract import (
    PhysicalModelContract,
    SUCCESSOR_ROOT_PRIM,
    WORKSPACE_ROOT,
    load_physical_model_contract,
)


R12_SCHEMA_VERSION = "kcg_d38999_keyed_v3_physical_model_contract_r12_v1"
R12_SUCCESSOR_SCHEMA = "kcg_d38999_keyed_physical_v3_r12_v1"
R12_SUCCESSOR_REVISION = "keyed_v3_physical_r12"
R12_ASSET_NAME = "d38999_shell25j_25_61_n_keyed_physical_v3_r12.usda"
R12_ASSET_DIRECTORY = "artifacts/kcg_connector/isaac/keyed_v3_physical_r12"
R12_PAIR_MODEL_ID = "d38999_shell25j_25_61_n_keyed_physical_pair_v3_r12"
R12_CONTRACT_PATH = (
    Path(__file__).resolve().parents[1]
    / "config/d38999_keyed_v3_physical_model_contract_r12_v1.yaml"
)
R12_ACCEPTANCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "config/d38999_keyed_v3_physical_acceptance_r12_v1.yaml"
)
R12_SCENE_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "config/d38999_keyed_v3_tabletop_scene_r12_v1.yaml"
)
R12_EXPECTED_COLLIDER_COUNT = 14761
R12_EXPECTED_MESH_COLLIDER_COUNT = 14684
R12_EXPECTED_ANALYTIC_COLLIDER_COUNT = 77
R12_CANDIDATE_LIMIT = 4


def candidate_asset_relative_path(candidate_index: int) -> Path:
    if candidate_index not in range(1, R12_CANDIDATE_LIMIT + 1):
        raise ValueError("r12 candidate index must be in 1..4")
    label = f"r12_candidate_{candidate_index:02d}"
    return Path(R12_ASSET_DIRECTORY) / "candidates" / label / f"{label}.usda"


def candidate_scene_relative_path(candidate_index: int) -> Path:
    if candidate_index not in range(1, R12_CANDIDATE_LIMIT + 1):
        raise ValueError("r12 candidate index must be in 1..4")
    return Path(R12_ASSET_DIRECTORY) / "candidates" / f"r12_candidate_{candidate_index:02d}" / "scene.yaml"


def _range_domain(size: int, field: str) -> dict[str, str]:
    return {field: f"0..{size - 1}"}


def _family_geometry(
    *, owner: str, path_template: str, count: int, index_field: str | None,
    material_role: str, response_role: str,
) -> dict[str, Any]:
    domains = {} if index_field is None else _range_domain(count, index_field)
    return {
        "owner": owner,
        "path_templates": [path_template],
        "index_domains": domains,
        "expected_leaf_count_nominal": count,
        "material_role": material_role,
        "response_role": response_role,
    }


def _canonical_pair(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left <= right else (right, left)


def _pair_closure(document: Mapping[str, Any]) -> tuple[int, int]:
    filtering = document["a2_collision_authoring_blueprint"]["filtering"]
    families = filtering["primitive_family_definitions"]
    primitive_names = set(families)
    composites = filtering["composite_family_definitions"]
    counts = {
        name: int(row["expected_leaf_count_nominal"])
        for name, row in families.items()
    }

    def resolve(name: str) -> set[str]:
        if name in primitive_names:
            return {name}
        definition = composites[name]
        included = (
            set(primitive_names)
            if definition.get("include_all_primitive_families")
            else set(definition.get("include", []))
        )
        return included - set(definition.get("exclude", []))

    explicit: set[tuple[str, str]] = set()
    declared = filtering["rule_expansion_contract"]["declared_cross_pairs"]
    for rule in filtering["family_pair_rules"]:
        expansion = rule["expansion"]
        if expansion == "unordered_complement_of_all_prior_explicit_rules":
            continue
        if expansion == "cartesian":
            pairs = {
                _canonical_pair(left, right)
                for left in resolve(rule["left"])
                for right in resolve(rule["right"])
            }
        elif expansion == "declared_cross_pairs":
            pairs = {
                _canonical_pair(left, right)
                for left, right in declared[rule["rule_id"]]
            }
        else:
            raise ValueError(f"unsupported r12 rule expansion {expansion}")
        if explicit & pairs:
            raise ValueError("r12 explicit family rules overlap")
        explicit |= pairs

    all_pairs = {
        _canonical_pair(left, right)
        for left in primitive_names
        for right in primitive_names
    }

    def concrete(pair: tuple[str, str]) -> int:
        left, right = pair
        if left == right:
            return counts[left] * (counts[left] - 1) // 2
        return counts[left] * counts[right]

    return (
        sum(concrete(pair) for pair in explicit),
        sum(concrete(pair) for pair in all_pairs - explicit),
    )


def build_r12_document(r11_document: Mapping[str, Any]) -> dict[str, Any]:
    """Return the one authorized r11 -> r12 contract transformation."""

    document = deepcopy(dict(r11_document))
    document["schema_version"] = R12_SCHEMA_VERSION
    identity = document["identity"]
    identity.update(
        {
            "predecessor_revision": "keyed_v3_physical_r11",
            "predecessor_role": "immutable_confirmed_first_jam_evidence_never_modify",
            "successor_schema": R12_SUCCESSOR_SCHEMA,
            "successor_revision": R12_SUCCESSOR_REVISION,
            "pair_model_id": R12_PAIR_MODEL_ID,
            "recommended_asset_name": R12_ASSET_NAME,
            "recommended_asset_directory": R12_ASSET_DIRECTORY,
        }
    )

    cooking = document["convex_cooking_representation"]
    cooking["analytic_cylinder_exception_families"] = [
        "pins_61",
        "detent_cam_continuous_base_1",
        "fixed_metal_stop_48",
        "shoulder_positive_body0_48",
        "shoulder_negative_body0_48",
    ]
    cooking["analytic_sphere_exception_families"] = [
        "detent_followers_3",
        "plug_metal_stop_48",
        "shoulder_positive_body1_48",
        "shoulder_negative_body1_48",
    ]
    cooking["r12_structural_repair_input"] = (
        "retained_r11_contact_evidence_confirms_tangential_side_face_lock_in_"
        "box_detent_followers_and_segmented_shoulders;validated_cap_plus_three_"
        "sphere_bottoming_boundary_is_reused_without_new_probe"
    )

    proxy_detent = document["physical_proxy_boundaries"]["force_parameters"][
        "anti_decoupling_detent"
    ]
    proxy_detent.update(
        {
            "representation": "continuous_cam_plus_36_teeth_and_three_analytic_sphere_followers",
            "follower_shape": "analytic_sphere",
            "follower_primitive_recipe_id": "analytic_sphere_v1",
            "follower_contact_surface": "continuous_spherical_surface_no_tangential_flat_side",
            "follower_radius_m": 0.000102,
            "follower_center_radius_m": 0.022076,
            "follower_local_center_z_m": 0.020000,
            "nominal_forward_component_target_nm": 0.060,
            "forward_component_absolute_tolerance_nm": 0.00006,
            "accepted_probe_measured_peak_nm": 0.060021022609,
        }
    )
    for obsolete in (
        "follower_axial_width_m",
        "follower_tangential_width_m",
        "follower_radial_interval_m",
    ):
        proxy_detent.pop(obsolete, None)

    shoulder_geometry = document["physical_proxy_boundaries"]["nut_body_bearing"][
        "physical_shoulder_collision_geometry"
    ]
    shoulder_geometry.clear()
    shoulder_geometry.update(
        {
            "representation": "two_analytic_axial_caps_each_against_three_analytic_spheres",
            "body0_primitive_recipe_id": "analytic_cylinder_v1",
            "body1_primitive_recipe_id": "analytic_sphere_v1",
            "body0_cap_radius_m": 0.01960,
            "body0_cap_axial_thickness_m": 0.00030,
            "body1_sphere_count_per_direction": 3,
            "body1_sphere_radius_m": 0.00050,
            "body1_sphere_distribution_radius_m": 0.01850,
            "body1_sphere_phases_deg": [0.0, 120.0, 240.0],
            "positive_transZ_stop": {
                "body0_collider_suffix": "/LoosePlug/BodyAssembly/NutBearingShoulders/PositiveStop",
                "body1_collider_suffix": "/LoosePlug/CouplingNut/NutBearingShoulders/PositiveStop",
                "body0_collider_center_local_z_m": 0.03015,
                "body1_sphere_center_local_z_m": 0.02945,
                "body0_contact_face_bound": "minZ",
                "body1_contact_face_bound": "maxZ",
                "expected_contact_transZ_m": 0.00005,
            },
            "negative_transZ_stop": {
                "body0_collider_suffix": "/LoosePlug/BodyAssembly/NutBearingShoulders/NegativeStop",
                "body1_collider_suffix": "/LoosePlug/CouplingNut/NutBearingShoulders/NegativeStop",
                "body0_collider_center_local_z_m": 0.00985,
                "body1_sphere_center_local_z_m": 0.01055,
                "body0_contact_face_bound": "maxZ",
                "body1_contact_face_bound": "minZ",
                "expected_contact_transZ_m": -0.00005,
            },
            "cap_center_fill_collides_only_with_corresponding_three_spheres": True,
            "minimum_sphere_to_cap_outer_edge_clearance_m": 0.00060,
            "collider_owner_must_match_named_body": True,
            "contact_normals_must_be_axial": True,
        }
    )

    blueprint = document["a2_collision_authoring_blueprint"]
    detent = blueprint["anti_decoupling_detent"]
    detent.update(
        {
            "representation": "continuous_analytic_base_cylinder_plus_36_convex_tooth_prisms_and_three_analytic_sphere_followers",
            "follower_shape": "analytic_sphere",
            "follower_primitive_recipe_id": "analytic_sphere_v1",
            "follower_contact_surface": "continuous_spherical_surface_no_tangential_flat_side",
            "follower_radius_m": 0.000102,
            "follower_center_radius_m": 0.022076,
            "follower_local_center_z_m": 0.020000,
            "nominal_forward_component_target_nm": 0.060,
            "forward_component_absolute_tolerance_nm": 0.00006,
            "accepted_probe_measured_peak_nm": 0.060021022609,
            "old_radial_tangent_box_collider_count": 0,
        }
    )
    for obsolete in (
        "follower_axial_width_m",
        "follower_tangential_width_m",
        "follower_radial_interval_m",
    ):
        detent.pop(obsolete, None)

    blueprint["metal_bottoming"] = {
        "source_kind": "FROZEN_SIMULATION_PROXY",
        "representation": "one_analytic_fixed_cap_plus_three_analytic_plug_spheres",
        "fixed_owner": "FixedReceptacle",
        "plug_owner": "BodyAssembly",
        "fixed_piece_path": f"{SUCCESSOR_ROOT_PRIM}/FixedReceptacle/MatingShell/MetalStop/AnalyticCap",
        "plug_piece_path_template": f"{SUCCESSOR_ROOT_PRIM}/LoosePlug/BodyAssembly/InternalMatingShell/MetalStop/AnalyticSphere_{{sphere_index}}",
        "fixed_cap_radius_m": 0.01695,
        "fixed_cap_axial_thickness_m": 0.00030,
        "fixed_cap_center_local_z_m": 0.00015,
        "plug_sphere_count": 3,
        "plug_sphere_radius_m": 0.00050,
        "plug_sphere_distribution_radius_m": 0.01600,
        "plug_sphere_center_local_z_m": 0.01555,
        "plug_sphere_phases_deg": [0.0, 120.0, 240.0],
        "nominal_bottoming_separation_m": 0.015050,
        "validated_no_contact_separation_m": 0.015049,
        "validated_contact_separation_m": 0.015051,
        "fixed_primitive_recipe_id": "analytic_cylinder_v1",
        "plug_primitive_recipe_id": "analytic_sphere_v1",
        "fixed_material_role": "fixture_and_receptacle",
        "plug_material_role": "plug_shell_and_keys",
        "response_role": "hard_metal_bottoming",
        "old_segmented_collider_count": 0,
        "only_named_stop_group_to_stop_group_pairs_enabled": True,
    }

    blueprint["nut_body_shoulders"] = {
        "source_kind": "FROZEN_SIMULATION_PROXY",
        "blueprint_reference": "physical_proxy_boundaries.nut_body_bearing.physical_shoulder_collision_geometry",
        "representation": "two_analytic_axial_caps_each_against_three_analytic_spheres",
        "group_count": 4,
        "actual_collider_count_by_family": {
            "shoulder_positive_body0_48": 1,
            "shoulder_positive_body1_48": 3,
            "shoulder_negative_body0_48": 1,
            "shoulder_negative_body1_48": 3,
        },
        "expected_total_collider_count": 8,
        "old_segmented_collider_count": 0,
        "body0_primitive_recipe_id": "analytic_cylinder_v1",
        "body1_primitive_recipe_id": "analytic_sphere_v1",
        "material_role": "coupling_bearing_and_shoulder",
        "response_role": "hard_nut_body_shoulder",
        "exact_enabled_pairs": [
            "positive_body0_to_positive_body1",
            "negative_body0_to_negative_body1",
        ],
        "exact_filtered_cross_pairs": [
            "positive_body0_to_negative_body1",
            "negative_body0_to_positive_body1",
        ],
    }

    filtering = blueprint["filtering"]
    families = filtering["primitive_family_definitions"]
    families["detent_followers_3"] = _family_geometry(
        owner="CouplingNut",
        path_template=detent["follower_path_template"],
        count=3,
        index_field="follower_index",
        material_role="anti_decoupling_detent",
        response_role="compliant_detent_follower",
    )
    bottom = blueprint["metal_bottoming"]
    families["fixed_metal_stop_48"] = _family_geometry(
        owner="FixedReceptacle",
        path_template=bottom["fixed_piece_path"],
        count=1,
        index_field=None,
        material_role="fixture_and_receptacle",
        response_role="hard_metal_bottoming",
    )
    families["plug_metal_stop_48"] = _family_geometry(
        owner="BodyAssembly",
        path_template=bottom["plug_piece_path_template"],
        count=3,
        index_field="sphere_index",
        material_role="plug_shell_and_keys",
        response_role="hard_metal_bottoming",
    )
    shoulder_families = {
        "shoulder_positive_body0_48": (
            "BodyAssembly",
            f"{SUCCESSOR_ROOT_PRIM}/LoosePlug/BodyAssembly/NutBearingShoulders/PositiveStop/AnalyticCap",
            1,
            None,
        ),
        "shoulder_positive_body1_48": (
            "CouplingNut",
            f"{SUCCESSOR_ROOT_PRIM}/LoosePlug/CouplingNut/NutBearingShoulders/PositiveStop/AnalyticSphere_{{sphere_index}}",
            3,
            "sphere_index",
        ),
        "shoulder_negative_body0_48": (
            "BodyAssembly",
            f"{SUCCESSOR_ROOT_PRIM}/LoosePlug/BodyAssembly/NutBearingShoulders/NegativeStop/AnalyticCap",
            1,
            None,
        ),
        "shoulder_negative_body1_48": (
            "CouplingNut",
            f"{SUCCESSOR_ROOT_PRIM}/LoosePlug/CouplingNut/NutBearingShoulders/NegativeStop/AnalyticSphere_{{sphere_index}}",
            3,
            "sphere_index",
        ),
    }
    for family, (owner, path_template, count, index_field) in shoulder_families.items():
        families[family] = _family_geometry(
            owner=owner,
            path_template=path_template,
            count=count,
            index_field=index_field,
            material_role="coupling_bearing_and_shoulder",
            response_role="hard_nut_body_shoulder",
        )

    leaf = filtering["realized_leaf_readback_contract"]
    overrides = leaf["analytic_primitive_overrides"]
    for family in cooking["analytic_cylinder_exception_families"]:
        overrides[family] = {
            "typeName": "Cylinder",
            "geometry_type": "analytic_cylinder",
            "physics_approximation": "none",
        }
    for family in cooking["analytic_sphere_exception_families"]:
        overrides[family] = {
            "typeName": "Sphere",
            "geometry_type": "analytic_sphere",
            "physics_approximation": "none",
        }

    total = sum(
        int(row["expected_leaf_count_nominal"]) for row in families.values()
    )
    partition = filtering["primitive_family_partition"]
    partition["expected_total_leaf_count_nominal"] = total
    partition["expected_total_unordered_distinct_leaf_pair_count_nominal"] = (
        total * (total - 1) // 2
    )
    explicit_count, default_count = _pair_closure(document)
    expansion = filtering["rule_expansion_contract"]
    expansion["expected_explicit_concrete_leaf_pair_count"] = explicit_count
    expansion["expected_default_concrete_leaf_pair_count"] = default_count

    result = document["solver_profile"]["resolved_readback_result_contract"]
    result.update(
        {
            "schema_version": "kcg_d38999_keyed_physical_r12_resolved_readback_v1",
            "contract_revision": "d38999_keyed_v3_r12_family_algebra_v1",
            "expected_collider_row_count": total,
        }
    )
    return document


def _expand_count(document: Mapping[str, Any]) -> tuple[int, int]:
    families = document["a2_collision_authoring_blueprint"]["filtering"][
        "primitive_family_definitions"
    ]
    analytic = document["a2_collision_authoring_blueprint"]["filtering"][
        "realized_leaf_readback_contract"
    ]["analytic_primitive_overrides"]
    total = sum(int(row["expected_leaf_count_nominal"]) for row in families.values())
    analytic_count = sum(
        int(families[name]["expected_leaf_count_nominal"]) for name in analytic
    )
    return total - analytic_count, analytic_count


def _validate_r12_document(document: Mapping[str, Any]) -> None:
    if document["schema_version"] != R12_SCHEMA_VERSION:
        raise ValueError("r12 contract schema changed")
    identity = document["identity"]
    expected_identity = {
        "successor_schema": R12_SUCCESSOR_SCHEMA,
        "successor_revision": R12_SUCCESSOR_REVISION,
        "pair_model_id": R12_PAIR_MODEL_ID,
        "recommended_asset_name": R12_ASSET_NAME,
        "recommended_asset_directory": R12_ASSET_DIRECTORY,
        "overwrite_existing": False,
    }
    for key, expected in expected_identity.items():
        if identity.get(key) != expected:
            raise ValueError(f"r12 identity.{key} changed")
    result = document["solver_profile"]["resolved_readback_result_contract"]
    if result["expected_collider_row_count"] != R12_EXPECTED_COLLIDER_COUNT:
        raise ValueError("r12 collider count is not 14761")
    mesh_count, analytic_count = _expand_count(document)
    if (mesh_count, analytic_count) != (
        R12_EXPECTED_MESH_COLLIDER_COUNT,
        R12_EXPECTED_ANALYTIC_COLLIDER_COUNT,
    ):
        raise ValueError("r12 mesh/analytic collider partition does not close")
    families = document["a2_collision_authoring_blueprint"]["filtering"][
        "primitive_family_definitions"
    ]
    expected_counts = {
        "detent_followers_3": 3,
        "fixed_metal_stop_48": 1,
        "plug_metal_stop_48": 3,
        "shoulder_positive_body0_48": 1,
        "shoulder_positive_body1_48": 3,
        "shoulder_negative_body0_48": 1,
        "shoulder_negative_body1_48": 3,
    }
    for family, expected in expected_counts.items():
        if families[family]["expected_leaf_count_nominal"] != expected:
            raise ValueError(f"r12 {family} count changed")
        if any("/Seg_" in template for template in families[family]["path_templates"]):
            raise ValueError(f"r12 {family} retains segmented paths")
    detent = document["a2_collision_authoring_blueprint"]["anti_decoupling_detent"]
    measured = float(detent["accepted_probe_measured_peak_nm"])
    nominal = float(detent["nominal_forward_component_target_nm"])
    tolerance = float(detent["forward_component_absolute_tolerance_nm"])
    if abs(measured - nominal) > tolerance:
        raise ValueError("accepted detent measurement exceeds r12 absolute tolerance")
    authorization = document["authorization"]
    for field in (
        "insertion_allowed",
        "twist_allowed",
        "randomization_allowed",
        "training_allowed",
        "rl_allowed",
        "hardware_control_allowed",
    ):
        if authorization.get(field) is not False:
            raise ValueError(f"r12 authorization.{field} must remain false")


def load_r12_physical_model_contract(
    path: Path | str = R12_CONTRACT_PATH,
) -> PhysicalModelContract:
    contract_path = Path(path).expanduser().resolve()
    r11 = load_physical_model_contract()
    expected = build_r12_document(r11.document)
    actual = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    if actual != expected:
        raise ValueError("r12 contract differs from the one authorized r11 transformation")
    _validate_r12_document(actual)
    return PhysicalModelContract(path=contract_path, document=actual)


def candidate_model(
    model: PhysicalModelContract, candidate_index: int
) -> PhysicalModelContract:
    document = deepcopy(dict(model.document))
    if candidate_index >= 2:
        _apply_thread_capsule_repair(document)
    if candidate_index >= 3:
        _apply_spring_target_capsule_repair(document)
    if candidate_index >= 4:
        _apply_continuous_spring_bore_cylinder_repair(document)
    relative = candidate_asset_relative_path(candidate_index)
    document["identity"]["recommended_asset_directory"] = str(relative.parent)
    document["identity"]["recommended_asset_name"] = relative.name
    return PhysicalModelContract(path=model.path, document=document)


def _apply_thread_capsule_repair(document: dict[str, Any]) -> None:
    """Remove the tangential end faces proven to jam candidate 01."""

    thread = document["a2_collision_authoring_blueprint"]["thread"]
    thread.update(
        {
            "rail_piece_shape": "analytic_capsule_chain_along_original_helix_chords",
            "rail_primitive_recipe_id": "analytic_capsule_helix_chord_v1",
            "rail_capsule_axis": "segment_centerline_chord",
            "rail_capsule_radius_m": 0.5 * float(thread["rail_axial_thickness_m"]),
            "rail_capsule_centerline_radius_m": float(thread["pitch_radius_m"]),
            "rail_capsule_centerline_axial_offset_m": 0.5
            * float(thread["rail_axial_thickness_m"]),
            "rail_capsule_cylinder_height_formula": "distance_between_original_segment_endpoint_centers",
            "original_closed_hexahedron_rail_count": 0,
            "tangential_planar_segment_end_face_count": 0,
            "candidate_01_direct_evidence": (
                "all_three_followers_jammed_at_Seg_164_165_with_"
                "abs_normal_dot_tangent_approximately_1"
            ),
            "candidate_02_only_change": (
                "thread_rail_piece_representation; follower geometry, three-start "
                "phase, lead, material, friction, D6, and P1 controller unchanged"
            ),
        }
    )
    cooking = document["convex_cooking_representation"]
    cooking["analytic_capsule_exception_families"] = ["thread_rails_3"]
    overrides = document["a2_collision_authoring_blueprint"]["filtering"][
        "realized_leaf_readback_contract"
    ]["analytic_primitive_overrides"]
    overrides["thread_rails_3"] = {
        "typeName": "Capsule",
        "geometry_type": "analytic_capsule",
        "physics_approximation": "none",
    }


def _apply_spring_target_capsule_repair(document: dict[str, Any]) -> None:
    """Remove the spring-target axial end face found by candidate 02.

    The target inner radius, tangential width, axial span, material, response,
    path identity, and 12-way phase layout remain unchanged.  Radius and
    center radius are derived from the frozen target width so the rounded
    target has the same inner contact bound without a planar entry face.
    """

    spring = document["a2_collision_authoring_blueprint"]["spring_fingers"]
    width = float(spring["target_tangential_width_m"])
    radius = 0.5 * width
    inner_radius = float(spring["target_inner_bore_radius_m"])
    z0, z1 = (float(value) for value in spring["target_local_z_interval_m"])
    total_length = z1 - z0
    cylinder_height = total_length - 2.0 * radius
    if cylinder_height <= 0.0:
        raise ValueError("spring target capsule axial span is not positive")
    spring.update(
        {
            "target_piece_shape": "analytic_z_capsule_no_planar_entry_face",
            "target_primitive_recipe_id": "analytic_capsule_spring_target_v1",
            "target_capsule_axis": "Z",
            "target_capsule_radius_m": radius,
            "target_capsule_center_radius_m": inner_radius + radius,
            "target_capsule_center_local_z_m": 0.5 * (z0 + z1),
            "target_capsule_cylinder_height_m": cylinder_height,
            "target_inner_contact_bound_m": inner_radius,
            "target_total_axial_span_m": total_length,
            "old_planar_axial_entry_face_count": 0,
            "candidate_02_direct_evidence": (
                "spring_contact_started_at_11.0791666667_s_with_"
                "abs_normal_dot_axis_0.9966959953; twelve_same_index_pairs_"
                "each_generated_about_four_contact_records_per_step"
            ),
            "candidate_03_only_change": (
                "spring_bore_target_piece_representation; all twelve fingers, "
                "target paths, phase, inner contact bound, axial span, material, "
                "stiffness, damping, friction, D6, and P1 controller unchanged"
            ),
        }
    )
    cooking = document["convex_cooking_representation"]
    analytic = list(cooking.get("analytic_capsule_exception_families", []))
    if "receptacle_bore_targets_12" not in analytic:
        analytic.append("receptacle_bore_targets_12")
    cooking["analytic_capsule_exception_families"] = analytic
    overrides = document["a2_collision_authoring_blueprint"]["filtering"][
        "realized_leaf_readback_contract"
    ]["analytic_primitive_overrides"]
    overrides["receptacle_bore_targets_12"] = {
        "typeName": "Capsule",
        "geometry_type": "analytic_capsule",
        "physics_approximation": "none",
    }


def _refresh_candidate_counts(document: dict[str, Any]) -> None:
    """Close collider and pair counts after a bounded candidate mutation."""

    filtering = document["a2_collision_authoring_blueprint"]["filtering"]
    families = filtering["primitive_family_definitions"]
    total = sum(
        int(row["expected_leaf_count_nominal"]) for row in families.values()
    )
    partition = filtering["primitive_family_partition"]
    partition["expected_total_leaf_count_nominal"] = total
    partition["expected_total_unordered_distinct_leaf_pair_count_nominal"] = (
        total * (total - 1) // 2
    )
    explicit_count, default_count = _pair_closure(document)
    expansion = filtering["rule_expansion_contract"]
    expansion["expected_explicit_concrete_leaf_pair_count"] = explicit_count
    expansion["expected_default_concrete_leaf_pair_count"] = default_count
    document["solver_profile"]["resolved_readback_result_contract"][
        "expected_collider_row_count"
    ] = total


def _apply_continuous_spring_bore_cylinder_repair(
    document: dict[str, Any],
) -> None:
    """Replace 12 hard target pieces with one collision-isolated cylinder."""

    blueprint = document["a2_collision_authoring_blueprint"]
    spring = blueprint["spring_fingers"]
    z0, z1 = (float(value) for value in spring["target_local_z_interval_m"])
    path = (
        f"{SUCCESSOR_ROOT_PRIM}/FixedReceptacle/MatingShell/"
        "InnerBoreTarget/ContinuousCylinder"
    )
    spring.update(
        {
            "target_piece_path_template": path,
            "target_segment_count": 1,
            "target_phase_formula_deg": "not_applicable_continuous_axisymmetric_target",
            "target_piece_shape": "one_collision_isolated_analytic_cylinder",
            "target_primitive_recipe_id": "analytic_cylinder_v1",
            "target_cylinder_radius_m": float(
                spring["target_inner_bore_radius_m"]
            ),
            "target_cylinder_height_m": z1 - z0,
            "target_cylinder_center_local_z_m": 0.5 * (z0 + z1),
            "target_has_tangential_segment_seams": False,
            "old_target_piece_count": 0,
            "candidate_04_only_change": (
                "twelve hard bore target pieces become one axisymmetric analytic "
                "cylinder; all twelve compliant fingers, finger identity, phase, "
                "material, stiffness, damping, friction, D6, and P1 controller "
                "remain unchanged"
            ),
        }
    )
    for obsolete in (
        "target_capsule_axis",
        "target_capsule_radius_m",
        "target_capsule_center_radius_m",
        "target_capsule_center_local_z_m",
        "target_capsule_cylinder_height_m",
        "target_inner_contact_bound_m",
        "target_total_axial_span_m",
        "old_planar_axial_entry_face_count",
    ):
        spring.pop(obsolete, None)

    families = blueprint["filtering"]["primitive_family_definitions"]
    original = families["receptacle_bore_targets_12"]
    families["receptacle_bore_targets_12"] = _family_geometry(
        owner="FixedReceptacle",
        path_template=path,
        count=1,
        index_field=None,
        material_role=original["material_role"],
        response_role=original["response_role"],
    )
    cooking = document["convex_cooking_representation"]
    capsules = list(cooking.get("analytic_capsule_exception_families", []))
    cooking["analytic_capsule_exception_families"] = [
        family for family in capsules if family != "receptacle_bore_targets_12"
    ]
    cylinders = list(cooking.get("analytic_cylinder_exception_families", []))
    if "receptacle_bore_targets_12" not in cylinders:
        cylinders.append("receptacle_bore_targets_12")
    cooking["analytic_cylinder_exception_families"] = cylinders
    overrides = blueprint["filtering"]["realized_leaf_readback_contract"][
        "analytic_primitive_overrides"
    ]
    overrides["receptacle_bore_targets_12"] = {
        "typeName": "Cylinder",
        "geometry_type": "analytic_cylinder",
        "physics_approximation": "none",
    }
    _refresh_candidate_counts(document)


def authorized_asset_path(model: PhysicalModelContract) -> Path:
    identity = model.document["identity"]
    return (
        WORKSPACE_ROOT
        / identity["recommended_asset_directory"]
        / identity["recommended_asset_name"]
    ).resolve()


__all__ = [
    "R12_ACCEPTANCE_PATH",
    "R12_ASSET_DIRECTORY",
    "R12_ASSET_NAME",
    "R12_CANDIDATE_LIMIT",
    "R12_CONTRACT_PATH",
    "R12_EXPECTED_ANALYTIC_COLLIDER_COUNT",
    "R12_EXPECTED_COLLIDER_COUNT",
    "R12_EXPECTED_MESH_COLLIDER_COUNT",
    "R12_PAIR_MODEL_ID",
    "R12_SCENE_CONFIG_PATH",
    "R12_SCHEMA_VERSION",
    "R12_SUCCESSOR_REVISION",
    "authorized_asset_path",
    "build_r12_document",
    "candidate_asset_relative_path",
    "candidate_model",
    "candidate_scene_relative_path",
    "load_r12_physical_model_contract",
]
