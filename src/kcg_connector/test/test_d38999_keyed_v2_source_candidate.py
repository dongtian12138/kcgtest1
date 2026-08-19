from pathlib import Path
import math

import yaml

from kcg_connector.d38999_key_branch_selector import (
    DEFAULT_THRESHOLDS,
    SUPPORTED_KEYED_PLUG_MODEL_IDS,
)


CONFIG = (
    Path(__file__).parents[1]
    / "config"
    / "d38999_keyed_v2_source_candidate.yaml"
)
KEY_REGION_CONFIG = (
    Path(__file__).parents[1]
    / "config"
    / "d38999_key_region_shadow_v1.yaml"
)


def _document():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def _key_region_document():
    return yaml.safe_load(KEY_REGION_CONFIG.read_text(encoding="utf-8"))


def test_keyed_v2_identity_is_side_by_side_and_external_cad_audit_disabled():
    document = _document()
    identity = document["side_by_side_model_identity"]
    assert document["enabled"] is False
    assert identity["migration_mode"] == "new_ids_no_in_place_upgrade"
    assert identity["reserved_keyed_v2_model_ids"] == {
        "loose_plug": "d38999_26kj61sn_keyed_proxy_v2",
        "fixed_receptacle": "d38999_20kj61pn_keyed_proxy_v2",
    }
    assert identity["existing_v1_model_ids_reused"] is False
    assert identity["existing_v1_assets_modified"] is False
    assert identity["keyed_v2_geometry_created"] is True
    assert identity["keyed_v2_runtime_registered"] is False


def test_public_spec_simulation_geometry_supersedes_the_external_cad_gap():
    document = _document()
    public_spec = document["public_spec_simulation_geometry"]
    assert document["status"] == "SUPERSEDED_FOR_SIMULATION_BY_PUBLIC_SPEC_KEYED_V2"
    assert public_spec["exact_25_61_public_standard_coordinates_available"] is True
    assert public_spec["five_key_n_polarization_public_standard_geometry_available"] is True
    assert public_spec["matching_keyway_collision_proxy_available"] is True
    assert public_spec["manufacturer_internal_interface_cad_claimed"] is False
    assert public_spec["hardware_metrology_claimed"] is False
    assert public_spec["formal_runtime_registered"] is False


def test_customer_view_step_cannot_claim_internal_key_geometry():
    document = _document()
    audit = document["cad_scope_audit"]
    assert audit["source_kind"] == "vendor_customer_view_external_geometry"
    assert audit["exact_25_61_insert_geometry_available"] is False
    assert audit["exact_master_and_secondary_key_geometry_available"] is False
    assert audit["exact_matching_keyway_geometry_available"] is False
    assert audit["sufficient_for_key_region_training"] is False
    assert audit["sufficient_for_keyway_collision"] is False
    assert audit["sufficient_for_unique_keyed_yaw_claim"] is False
    assert audit["integrity_binding_status"] == (
        "deferred_until_source_freeze_milestone"
    )


def test_unknown_real_clearance_keeps_yaw_and_control_blocked():
    document = _document()
    keying = document["keying_reference"]
    authorization = document["authorization"]
    assert keying["allowable_mating_yaw_clearance_deg"] is None
    assert keying["yaw_p95_acceptance_limit_deg"] is None
    assert keying["control_allowed_before_clearance_is_known"] is False
    assert authorization["shadow_key_region_development_allowed"] is True
    assert authorization["unique_key_yaw_claimed"] is False
    assert authorization["insertion_control_authorized"] is False
    assert authorization["robot_control_authorized"] is False
    assert authorization["real_keying_modeled"] is False


def test_selected_pair_is_exact_25_61_n_but_not_physical_unit_verified():
    document = _document()
    pair = document["identity"]["selected_pair"]
    assert pair["loose_plug"]["dod_pin"] == "D38999/26KJ61SN"
    assert pair["fixed_receptacle"]["dod_pin"] == "D38999/20KJ61PN"
    for endpoint in pair.values():
        assert endpoint["insert_arrangement"] == "25-61"
        assert endpoint["contact_count"] == 61
        assert endpoint["contact_size"] == 20
        assert endpoint["polarization"] == "N"
    traceability = document["identity"]["physical_unit_traceability"]
    assert not any(traceability.values())


def test_key_region_stage_is_shadow_only_and_rejects_bad_observations():
    document = _key_region_document()
    assert document["enabled"] is False
    assert document["isolated_simulation_probe_enabled"] is True
    assert document["model_identity"][
        "current_c2_proxy_allowed_as_keyed_model"
    ] is False
    assert document["model_identity"][
        "public_spec_simulation_key_geometry_available"
    ] is True
    assert document["model_identity"][
        "manufacturer_internal_key_geometry_available"
    ] is False
    assert document["purpose"]["input_hypothesis_count"] == 2
    assert document["purpose"]["output"] == (
        "select_one_c2_branch_for_shadow_only"
    )
    required = set(document["required_rejections"])
    assert {
        "CONNECTOR_FACE_OUT_OF_FRAME",
        "KEY_REGION_OUT_OF_FRAME",
        "KEY_REGION_OCCLUDED",
        "KEY_REGION_DEPTH_MISSING",
        "KEY_REGION_LOW_CONFIDENCE",
        "KEY_BRANCH_AMBIGUOUS",
        "KEYED_MODEL_ID_UNAVAILABLE",
        "KEYED_GEOMETRY_UNAVAILABLE",
        "KEY_REGION_OCCLUSION_UNKNOWN",
    } <= required
    assert document["authorization"]["selected_for_control_field_allowed"] is False
    assert document["authorization"][
        "isolated_simulation_shadow_selection_authorized"
    ] is False
    assert document["authorization"]["control_authorized"] is False


def test_key_region_yaml_and_cpu_selector_use_the_same_candidate_gates():
    document = _key_region_document()
    gates = document["quality_gates"]
    assert SUPPORTED_KEYED_PLUG_MODEL_IDS == {
        document["model_identity"]["required_model_id"]
    }
    assert gates["key_probability_threshold"] == DEFAULT_THRESHOLDS[
        "minimum_key_probability"
    ]
    assert gates["minimum_key_pixels"] == DEFAULT_THRESHOLDS[
        "minimum_key_support_pixels"
    ]
    assert gates["minimum_mean_key_probability"] == DEFAULT_THRESHOLDS[
        "minimum_mean_key_probability"
    ]
    assert gates["minimum_key_region_fraction_of_face"] == DEFAULT_THRESHOLDS[
        "minimum_key_area_fraction"
    ]
    assert gates["maximum_key_region_fraction_of_face"] == DEFAULT_THRESHOLDS[
        "maximum_key_area_fraction"
    ]
    assert gates["maximum_secondary_component_mass_ratio"] == DEFAULT_THRESHOLDS[
        "maximum_secondary_component_mass_ratio"
    ]
    assert gates["minimum_valid_key_depth_fraction"] == DEFAULT_THRESHOLDS[
        "minimum_valid_key_depth_fraction"
    ]
    assert gates["maximum_occluded_key_fraction"] == DEFAULT_THRESHOLDS[
        "maximum_occlusion_fraction"
    ]
    assert gates["maximum_occluded_face_fraction"] == DEFAULT_THRESHOLDS[
        "maximum_occlusion_fraction"
    ]
    assert gates["image_border_margin_px"] == DEFAULT_THRESHOLDS[
        "image_border_margin_px"
    ]
    assert gates["minimum_key_radial_distance_px"] == DEFAULT_THRESHOLDS[
        "minimum_radial_length_px"
    ]
    assert math.isclose(
        math.degrees(gates["maximum_branch_angular_error_rad"]),
        DEFAULT_THRESHOLDS["maximum_branch_angle_error_deg"]
    )
    assert math.isclose(
        math.degrees(gates["minimum_branch_angular_margin_rad"]),
        DEFAULT_THRESHOLDS["minimum_branch_margin_deg"]
    )


def test_key_region_accuracy_gate_cannot_invent_a_real_clearance():
    accuracy = _key_region_document()["accuracy_gate"]
    assert accuracy["measured_allowable_mating_yaw_clearance_deg"] is None
    assert accuracy["required_yaw_error_p95_deg"] is None
    assert accuracy["evaluation_dataset_status"] == "NOT_AVAILABLE"
    assert accuracy["control_allowed_before_gate_passes"] is False
