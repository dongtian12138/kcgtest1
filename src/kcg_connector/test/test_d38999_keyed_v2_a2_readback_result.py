from copy import deepcopy

import pytest

from kcg_connector.d38999_keyed_v2_a2_readback_result import (
    CONTRACT_REVISION,
    GENERATOR_ID,
    SCHEMA_VERSION,
    _trusted_collider_inventory,
    _trusted_family_algebra,
    _trusted_property_inventory,
    validate_a2_composed_asset_release,
    validate_a2_resolved_readback_result,
)
from kcg_connector.d38999_keyed_v2_physical_model_contract import (
    PhysicalModelContract,
    SUCCESSOR_ROOT_PRIM,
    WORKSPACE_ROOT,
    load_physical_model_contract,
)


def _valid_candidate():
    model = load_physical_model_contract()
    colliders = _trusted_collider_inventory(model)
    properties = _trusted_property_inventory(model)
    pairs, filters = _trusted_family_algebra(model)
    response_roles = model.document["material_roles"]["response_roles"]

    collider_rows = []
    for expected in colliders.values():
        response = response_roles[expected["responseRole"]]
        if response["class"] == "hard":
            stiffness = response["compliant_stiffness_n_m"]
            damping = response["compliant_damping_n_s_m"]
        else:
            stiffness = response["nominal_stiffness_n_m"]
            damping = response["nominal_damping_n_s_m"]
        collider_rows.append(
            {
                "family": expected["family"],
                "prim_path": expected["prim_path"],
                "collider_index": expected["collider_index"],
                "typeName": expected["typeName"],
                "appliedSchemas": ["PhysicsCollisionAPI"],
                "collisionEnabled": True,
                "local_bounds": [[-0.001, -0.001, -0.001], [0.001, 0.001, 0.001]],
                "world_bounds_at_canonical_pose": [[-0.001, -0.001, -0.001], [0.001, 0.001, 0.001]],
                "geometry_type": expected["geometry_type"],
                "physics_approximation": expected["physics_approximation"],
                "closed_manifold": True,
                "positive_volume": True,
                "convex": True,
                "topology_signature": f"semantic:{expected['family']}:recipe-v1",
                "materialRole": expected["materialRole"],
                "responseRole": expected["responseRole"],
                "effective_physics_material_binding": f"/World/Materials/{expected['materialRole']}",
                "material_binding_source_prim": expected["prim_path"],
                "resolved_compliant_stiffness_n_m": stiffness,
                "resolved_compliant_damping_n_s_m": damping,
                "resolved_accelerationSpring": False,
                "nearest_rigid_body_owner": expected["owner"],
                "owner_rigidBodyEnabled": True,
                "owner_kinematicEnabled": False,
                "offset_class": "fine_connector",
                "contactOffset_m": 0.00001,
                "restOffset_m": 0.0,
                "collision_group_memberships": [expected["collision_group"]],
                "filteredPairs_sources": [],
                "pass": True,
            }
        )
    property_rows = [
        {**expected, "resolved_value": expected["expected_value"], "pass": True}
        for expected in properties.values()
    ]
    family_pair_rows = [
        {
            **expected,
            "resolved_decision": expected["expected_decision"],
            "resolved_left_member_count": expected["expected_left_member_count"],
            "resolved_right_member_count": expected["expected_right_member_count"],
            "resolved_concrete_leaf_pair_count": expected["expected_concrete_leaf_pair_count"],
            "pass": True,
        }
        for expected in pairs.values()
    ]
    filter_rows = [
        {
            **expected,
            "resolved_effect": expected["expected_effect"],
            "resolved_concrete_leaf_pair_count": expected["expected_concrete_leaf_pair_count"],
            "pass": True,
        }
        for expected in filters.values()
    ]
    identity = model.document["identity"]
    asset_path = str(
        (
            WORKSPACE_ROOT
            / identity["recommended_asset_directory"]
            / identity["recommended_asset_name"]
        ).resolve()
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_id": GENERATOR_ID,
        "contract_revision": CONTRACT_REVISION,
        "asset_path": asset_path,
        "root_prim": SUCCESSOR_ROOT_PRIM,
        "collider_rows": collider_rows,
        "property_rows": property_rows,
        "family_pair_rows": family_pair_rows,
        "filter_source_rows": filter_rows,
        "summary": dict(
            model.document["solver_profile"]["resolved_readback_result_contract"][
                "required_summary_counts"
            ]
        ),
    }


def test_complete_candidate_is_checked_against_internal_frozen_inventory():
    result = validate_a2_resolved_readback_result(_valid_candidate())
    assert result.collider_row_count == 15037
    assert result.property_row_count == 22
    assert result.family_pair_row_count == 406
    assert result.filter_source_row_count == 387
    assert result.release_evidence is False


def test_external_expected_inventory_is_not_an_api_argument():
    candidate = _valid_candidate()
    with pytest.raises(TypeError):
        validate_a2_resolved_readback_result(candidate, candidate)


def test_all_zero_claim_cannot_hide_a_missing_collider():
    candidate = _valid_candidate()
    candidate["collider_rows"].pop()
    with pytest.raises(ValueError, match="summary differs"):
        validate_a2_resolved_readback_result(candidate)


def test_pass_claim_cannot_hide_a_property_mismatch():
    candidate = _valid_candidate()
    candidate["property_rows"][0]["resolved_value"] = "wrong"
    with pytest.raises(ValueError, match="untrusted pass claim"):
        validate_a2_resolved_readback_result(candidate)


def test_release_entrypoint_stays_closed_while_a0_is_open():
    candidate = _valid_candidate()
    active = load_physical_model_contract()
    document = deepcopy(active.document)
    document["authorization"]["a2_asset_authoring_allowed"] = False
    blocked = PhysicalModelContract(path=active.path, document=document)
    with pytest.raises(PermissionError, match="A0 source freeze"):
        validate_a2_composed_asset_release(candidate["asset_path"], model=blocked)
