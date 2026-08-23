from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from kcg_connector.grasp.robust.aggregate_collision_inputs import (
    build_carts_aggregate_collision_runtime_inputs,
)
from kcg_connector.grasp.robust.candidate_route import (
    CLAIM_LIMITATIONS,
    EXPECTED_STAGE_ORDER,
    CandidateRouteError,
    build_candidate_route_state_contract,
)
from kcg_connector.grasp.robust.collision_geometry_binding import (
    certify_carts_collision_geometry_bindings,
)
from kcg_connector.grasp.robust.collision_roster import (
    load_authoritative_collision_link_roster,
)
from kcg_connector.grasp.robust.hand_contract import load_carts_hand_contract
from kcg_connector.grasp.robust.interval_kinematics import (
    DISPLAY_APPROXIMATION_ROLE,
    IMPLICIT_ROOT_FEATURE_TYPE,
    IMPLICIT_ROOT_METHOD_ID,
    METHOD_ID as INTERVAL_METHOD_ID,
    CertifiedImplicitRoot,
    IntervalBounds,
    IntervalTransverseRootCertificate,
)
from kcg_connector.grasp.robust.object_contract import load_object_contract
from kcg_connector.grasp.robust.object_world_pose import (
    certify_settled_object_world_pose,
)
from kcg_connector.grasp.robust.ray_closure import (
    CANDIDATE_REPRESENTATIVE_ROLE,
    CLOSURE_PARAMETER_DOMAIN_ID,
    CertifiedContactFeatureRoot,
    CertifiedSequentialClosurePolicy,
    METHOD_ID as V9_METHOD_ID,
    MODEL_BINDING_COMPLETE_STATUS,
    MODEL_CONTRACT_DIGEST_METHOD_ID,
    PARAMETER_LAYOUT_PREFIX,
    PossibleFirstContactSet,
    REPRESENTATIVE_PROPOSAL_FAILURE_REASON,
)
from kcg_connector.grasp.robust.shared_environment import (
    load_shared_table_fixture_world,
)
from kcg_connector.grasp.robust.top_level_candidate_generator import (
    CandidateLane,
    CandidateLineage,
    StaticV9AcceptedPolicy,
    V9InvocationAuditBinding,
    canonicalize_v9_parameters,
)


REPOSITORY = Path(__file__).resolve().parents[4]
HAND_CONTRACT = REPOSITORY / "src/kcg_connector/config/carts_hand_contact_v1.yaml"
COLLISION_ROSTER = REPOSITORY / "src/kcg_connector/config/carts_collision_roster_v1.yaml"
OBJECT_CONTRACT = REPOSITORY / "src/kcg_connector/config/carts_grasp_objects_v1.yaml"
SHARED_ENVIRONMENT = (
    REPOSITORY / "src/kcg_connector/config/carts_shared_table_fixture_world_v1.yaml"
)
PLACEMENT_CONTRACT = (
    REPOSITORY / "src/kcg_connector/config/carts_shared_object_placement_v1.yaml"
)
ROUTE_CONTRACT = REPOSITORY / "src/kcg_connector/config/carts_candidate_route_v1.yaml"
CURRENT_OBJECT = "current_d38999_26kj61sn_public_spec"
TRANSFER_OBJECT = "te_deutsch_d38999_26fj35pn_step"
HAND_JOINTS = ("f1j1", "f1j2", "f2j1", "f3j2")
SUPPORTS = (("f1j2",), ("f2j1",), ("f3j2",))
V9_LAYOUT = PARAMETER_LAYOUT_PREFIX + ("preshape_joint_unit:f1j1",)


@pytest.fixture(scope="module")
def real_route_inputs():
    hand = load_carts_hand_contract(HAND_CONTRACT, repository_root=REPOSITORY)
    roster = load_authoritative_collision_link_roster(
        COLLISION_ROSTER, repository_root=REPOSITORY
    )
    geometry = certify_carts_collision_geometry_bindings(hand, roster)
    environment = load_shared_table_fixture_world(
        SHARED_ENVIRONMENT, repository_root=REPOSITORY
    )
    from kcg_connector.grasp.robust.interval_kinematics import (
        IntervalArithmeticOptions,
    )

    options = IntervalArithmeticOptions(
        decimal_precision=80,
        maximum_root_bisection_iterations=256,
    )
    result = {}
    for object_id in (CURRENT_OBJECT, TRANSFER_OBJECT):
        object_contract = load_object_contract(
            OBJECT_CONTRACT,
            object_id=object_id,
            repository_root=REPOSITORY,
        )
        aggregate = build_carts_aggregate_collision_runtime_inputs(
            hand_contract=hand,
            collision_roster=roster,
            geometry_binding=geometry,
            object_contract=object_contract,
            shared_environment=environment,
            interval_options=options,
        )
        pose = certify_settled_object_world_pose(
            PLACEMENT_CONTRACT,
            aggregate_inputs=aggregate,
            repository_root=REPOSITORY,
        )
        result[object_id] = (aggregate, pose)
    return result


def _possible_contact_set(
    pad_name: str,
    ordinal: int,
) -> PossibleFirstContactSet:
    lower = 0.20 + 0.05 * ordinal
    upper = lower + 0.01
    implicit = CertifiedImplicitRoot(
        method_id=IMPLICIT_ROOT_METHOD_ID,
        equation_sha256=hashlib.sha256(
            f"route-equation:{pad_name}".encode("utf-8")
        ).hexdigest(),
        feature_identity_sha256=hashlib.sha256(
            f"route-feature:{pad_name}".encode("utf-8")
        ).hexdigest(),
        feature_type=IMPLICIT_ROOT_FEATURE_TYPE,
        isolating_interval=IntervalBounds(lower, upper),
        value_at_lower=IntervalBounds(0.10, 0.20),
        value_at_upper=IntervalBounds(-0.20, -0.10),
        derivative=IntervalBounds(-2.0, -1.0),
        uniqueness_proven=True,
        display_approximation=lower + 0.5 * (upper - lower),
        display_approximation_role=DISPLAY_APPROXIMATION_ROLE,
    )
    certificate = IntervalTransverseRootCertificate(
        implicit_root=implicit,
        triangle_edge_halfspaces=(
            IntervalBounds(0.10, 0.20),
            IntervalBounds(0.10, 0.20),
            IntervalBounds(0.10, 0.20),
        ),
        pad_approach=IntervalBounds(0.50, 1.00),
        path_local_free_side_approach=IntervalBounds(0.50, 1.00),
        object_source_winding_free_side_sign=1,
        position_object_m=(
            IntervalBounds(float(ordinal), float(ordinal) + 0.01),
            IntervalBounds(-0.01, 0.01),
            IntervalBounds(-0.01, 0.01),
        ),
        bisection_iterations=8,
        method_id=INTERVAL_METHOD_ID,
        decimal_precision=80,
    )
    return PossibleFirstContactSet.from_certified_roots(
        (
            CertifiedContactFeatureRoot(
                pad_name=pad_name,
                witness_flat_index=ordinal,
                pad_triangle_index=0,
                witness_index=ordinal,
                object_face_index=ordinal,
                semantic_classification=(
                    "ALLOWED_PATH_LOCAL_FREE_SIDE_TRANSVERSE_CONTACT"
                ),
                certificate=certificate,
            ),
        )
    )


def _accepted_policy(aggregate) -> StaticV9AcceptedPolicy:
    pad_names = tuple(row.pad_name for row in aggregate.terminal_roles)
    pad_links = tuple(row.link_name for row in aggregate.terminal_roles)
    directions = (
        (0.0, 0.1, 0.0, 0.0),
        (0.0, 0.0, 0.1, 0.0),
        (0.0, 0.0, 0.0, 0.1),
    )
    hand_model = aggregate.kinematic_binding.model
    initial = tuple(
        0.5
        * (
            hand_model.independent_joint_limits[name].lower
            + hand_model.independent_joint_limits[name].upper
        )
        for name in HAND_JOINTS
    )
    contact_sets = tuple(
        _possible_contact_set(name, index)
        for index, name in enumerate(pad_names)
    )
    manifest = {
        "schema": MODEL_CONTRACT_DIGEST_METHOD_ID,
        "hand": {
            "base_link": "handbase_link",
            "independent_joint_names": list(HAND_JOINTS),
        },
        "verified_pads": [
            {"name": name, "link_name": link}
            for name, link in zip(pad_names, pad_links)
        ],
    }
    canonical_manifest = json.dumps(
        manifest,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    model_sha = hashlib.sha256(canonical_manifest.encode("utf-8")).hexdigest()
    policy = CertifiedSequentialClosurePolicy(
        object_from_hand=tuple(float(value) for value in np.eye(4).ravel()),
        initial_independent_joint_positions_rad=initial,
        independent_joint_names=HAND_JOINTS,
        pad_order=pad_names,
        independent_actuation_supports=SUPPORTS,
        closing_directions_physical=directions,
        possible_first_contact_sets=contact_sets,
        object_geometry_sha256=(
            aggregate.object_surface.ray_closure_object_geometry_sha256
        ),
        model_contract_sha256=model_sha,
    )
    audit = SimpleNamespace(
        method_id=V9_METHOD_ID,
        closure_parameter_domain_id=CLOSURE_PARAMETER_DOMAIN_ID,
        parameter_layout=V9_LAYOUT,
        failure_reason=REPRESENTATIVE_PROPOSAL_FAILURE_REASON,
        model_binding_complete=True,
        model_binding_status=MODEL_BINDING_COMPLETE_STATUS,
        object_geometry_sha256=policy.object_geometry_sha256,
        model_contract_sha256=model_sha,
        pad_order=pad_names,
        pad_link_names=pad_links,
        independent_actuation_supports=SUPPORTS,
        closing_directions_physical=directions,
        possible_first_contact_set_sha256=tuple(
            row.set_sha256 for row in contact_sets
        ),
        candidate_role=CANDIDATE_REPRESENTATIVE_ROLE,
        candidate_exact_contact_endpoint_certified=False,
        full_verified_pad_mesh_used=True,
        pad_face_subset_input_allowed=False,
        subdivision_budget_exhausted=False,
        model_contract_canonical_json=canonical_manifest,
    )
    parameters = (0.1, 0.2, 0.3, 0.4, 0.5)
    key = canonicalize_v9_parameters(
        parameters, parameter_layout=V9_LAYOUT
    ).exact_key_hex
    lineage = CandidateLineage(
        attempt_index=0,
        lane=CandidateLane.DIRECT_V9,
        lane_point_index=0,
        sobol_seed=20260820,
        sobol_parameters_unit=parameters,
        anchor_pad_name=None,
        proposal_audit=None,
        proposal_failure_reason=None,
    )
    binding = V9InvocationAuditBinding(
        method_id=V9_METHOD_ID,
        parameter_domain_id=CLOSURE_PARAMETER_DOMAIN_ID,
        parameter_layout=V9_LAYOUT,
        requested_parameters_unit=parameters,
        requested_parameter_key_hex=key,
        raw_v9_audit=audit,
    )
    return StaticV9AcceptedPolicy(
        v9_parameters_unit=parameters,
        v9_parameter_key_hex=key,
        sequential_closure_policy=policy,
        v9_audit=audit,
        invocation_binding=binding,
        lineage=(lineage,),
    )


def _build(object_id: str, real_route_inputs):
    aggregate, pose = real_route_inputs[object_id]
    accepted = _accepted_policy(aggregate)
    contract = build_candidate_route_state_contract(
        ROUTE_CONTRACT,
        accepted_policy=accepted,
        aggregate_inputs=aggregate,
        object_world_pose=pose,
        repository_root=REPOSITORY,
    )
    return aggregate, pose, accepted, contract


def test_both_objects_bind_the_same_six_stage_route_rules(
    real_route_inputs,
) -> None:
    contracts = {}
    for object_id in (CURRENT_OBJECT, TRANSFER_OBJECT):
        aggregate, pose, accepted, contract = _build(
            object_id, real_route_inputs
        )
        contracts[object_id] = contract
        assert contract.object_id == object_id
        assert contract.stage_order == EXPECTED_STAGE_ORDER
        assert contract.complete_joint_names == (
            aggregate.kinematic_binding.independent_joint_names
        )
        assert contract.home_independent_joint_positions_rad[:7] == (0.0,) * 7
        assert contract.home_independent_joint_positions_rad[7:] == (
            accepted.sequential_closure_policy.initial_independent_joint_positions_rad
        )
        np.testing.assert_array_equal(
            contract.world_from_hand_pregrasp_target,
            pose.world_from_object,
        )
        lift_delta = (
            contract.world_from_hand_lift_target[:3, 3]
            - contract.world_from_hand_pregrasp_target[:3, 3]
        )
        np.testing.assert_allclose(
            lift_delta,
            (0.0, 0.0, 0.04),
            rtol=0.0,
            atol=(
                8.0
                * np.finfo(np.float64).eps
                * max(1.0, float(np.max(np.abs(lift_delta))))
            ),
        )
        assert contract.route_state_binding_complete is True
        assert contract.arm_ik_solution_complete is False
        assert contract.candidate_specific_motion_binding_complete is False
        assert contract.continuous_collision_complete is False
        assert contract.formal_selection_input_used is False
        assert contract.selection_occurs_after_route_evaluation is True
        assert contract.claim_limitations == CLAIM_LIMITATIONS
    assert (
        contracts[CURRENT_OBJECT].certificate_sha256
        != contracts[TRANSFER_OBJECT].certificate_sha256
    )


def test_closure_sweep_and_contact_endpoints_remain_intervals(
    real_route_inputs,
) -> None:
    _aggregate, _pose, accepted, contract = _build(
        CURRENT_OBJECT, real_route_inputs
    )
    initial = accepted.sequential_closure_policy.initial_independent_joint_positions_rad
    swept = contract.closure_swept_hand_joint_intervals_rad
    contact = contract.closure_contact_hand_joint_intervals_rad
    assert swept[0] == IntervalBounds(initial[0], initial[0])
    assert contact[0] == IntervalBounds(initial[0], initial[0])
    for index in (1, 2, 3):
        assert swept[index].lower == initial[index]
        assert swept[index].upper > swept[index].lower
        assert contact[index].lower > swept[index].lower
        assert contact[index].upper == swept[index].upper


def test_empty_display_or_cross_object_input_fails_closed(
    real_route_inputs,
) -> None:
    current_aggregate, current_pose = real_route_inputs[CURRENT_OBJECT]
    transfer_pose = real_route_inputs[TRANSFER_OBJECT][1]
    accepted = _accepted_policy(current_aggregate)
    for invalid in (None, object(), SimpleNamespace(display_only=True)):
        with pytest.raises(CandidateRouteError) as error:
            build_candidate_route_state_contract(
                ROUTE_CONTRACT,
                accepted_policy=invalid,
                aggregate_inputs=current_aggregate,
                object_world_pose=current_pose,
                repository_root=REPOSITORY,
            )
        assert error.value.code == "STATIC_ACCEPTED_POLICY_REQUIRED"
    with pytest.raises(CandidateRouteError) as cross_object:
        build_candidate_route_state_contract(
            ROUTE_CONTRACT,
            accepted_policy=accepted,
            aggregate_inputs=current_aggregate,
            object_world_pose=transfer_pose,
            repository_root=REPOSITORY,
        )
    assert cross_object.value.code == "CROSS_OBJECT_OR_SCENE_BINDING_MISMATCH"


def test_v9_lineage_drift_is_rejected_and_contract_is_immutable(
    real_route_inputs,
) -> None:
    aggregate, _pose, accepted, contract = _build(
        CURRENT_OBJECT, real_route_inputs
    )
    with pytest.raises(CandidateRouteError) as drift:
        build_candidate_route_state_contract(
            ROUTE_CONTRACT,
            accepted_policy=replace(
                accepted, v9_parameter_key_hex="0" * 80
            ),
            aggregate_inputs=aggregate,
            object_world_pose=real_route_inputs[CURRENT_OBJECT][1],
            repository_root=REPOSITORY,
        )
    assert drift.value.code == "ACCEPTED_POLICY_LINEAGE_MISMATCH"
    assert not contract.world_from_hand_pregrasp_target.flags.writeable
    with pytest.raises(FrozenInstanceError):
        contract.object_id = TRANSFER_OBJECT  # type: ignore[misc]


def test_route_source_has_no_legacy_waypoint_or_runtime_truth_shortcut() -> None:
    source = (
        REPOSITORY
        / "src/kcg_connector/kcg_connector/grasp/robust/candidate_route.py"
    ).read_text(encoding="utf-8")
    assert "connector_home_to_pregrasp_v1.yaml" not in source
    assert "proposed_clearance_arm_rad" not in source
    assert "grasp_arm_rad" not in source
    assert "formal_selected_contact_range_policy" not in source
    assert "get_world_pose" not in source
    assert "set_world_pose" not in source
