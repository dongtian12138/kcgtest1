from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import test_candidate_route as route_fixtures

from kcg_connector.grasp.robust import candidate_route_collision as collision
from kcg_connector.grasp.robust.candidate_joint_route import (
    build_candidate_joint_route_contract,
)
from kcg_connector.grasp.robust.candidate_route_collision import (
    CandidateRouteCollisionError,
    CandidateRouteCollisionState,
    build_candidate_route_collision_certificate,
)
from kcg_connector.grasp.robust.hand_contract import load_carts_hand_contract


REPOSITORY = route_fixtures.REPOSITORY
ROUTE_COLLISION_CONTRACT = (
    REPOSITORY
    / "src/kcg_connector/config/carts_candidate_route_collision_v1.yaml"
)
JOINT_ROUTE_CONTRACT = (
    REPOSITORY
    / "src/kcg_connector/config/carts_candidate_joint_route_v1.yaml"
)
real_route_inputs = route_fixtures.real_route_inputs


def _triangle(*rows: tuple[float, float, float]) -> np.ndarray:
    return np.asarray((rows,), dtype=np.float64)


def test_fixed_axis_surface_check_accepts_separation_but_not_overlap() -> None:
    first = collision._expanded_exact_surface(
        "first",
        _triangle((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        1,
    )
    coplanar_disjoint = collision._expanded_exact_surface(
        "coplanar_disjoint",
        _triangle((1.0, 1.0, 0.0), (1.0, 0.6, 0.0), (0.6, 1.0, 0.0)),
        1,
    )
    overlapping = collision._expanded_exact_surface(
        "overlapping",
        _triangle((0.2, 0.2, 0.0), (0.8, 0.2, 0.0), (0.2, 0.8, 0.0)),
        1,
    )

    free, reason = collision._surfaces_strictly_separated(
        first, coplanar_disjoint
    )
    assert free is True
    assert reason == "NONE"
    unresolved, reason = collision._surfaces_strictly_separated(
        first, overlapping
    )
    assert unresolved is False
    assert reason.startswith("NO_STRICT_FIXED_AXIS_TRIANGLE_SEPARATION")


def test_real_route_fails_closed_at_a_named_collision_scope(
    real_route_inputs,
) -> None:
    aggregate, pose, accepted, route_state = route_fixtures._build(
        route_fixtures.CURRENT_OBJECT, real_route_inputs
    )
    joint_route = build_candidate_joint_route_contract(
        JOINT_ROUTE_CONTRACT,
        route_state=route_state,
        aggregate_inputs=aggregate,
        repository_root=REPOSITORY,
    )
    hand = load_carts_hand_contract(
        route_fixtures.HAND_CONTRACT, repository_root=REPOSITORY
    )
    possible_roots, authorized_roots, pad_blockers = (
        collision._pad_root_binding(accepted, aggregate, hand)
    )
    assert possible_roots > 0
    assert authorized_roots == possible_roots
    assert not pad_blockers
    result = build_candidate_route_collision_certificate(
        ROUTE_COLLISION_CONTRACT,
        accepted_policy=accepted,
        route_state=route_state,
        joint_route=joint_route,
        aggregate_inputs=aggregate,
        object_world_pose=pose,
        hand_contract=hand,
        repository_root=REPOSITORY,
    )

    assert result.state is CandidateRouteCollisionState.NOT_CERTIFIABLE
    assert result.complete_route_collision_coverage is False
    assert result.dynamic_launch_allowed is False
    assert result.controller_execution_authorized is False
    assert result.first_failure is not None
    assert result.first_failure.stage_name
    assert result.first_failure.first_body
    assert result.first_failure.second_body
    assert result.first_failure.reason
    assert result.processed_joint_box_count > 0
    assert result.self_pair_count == 136
    assert result.structural_interface_pair_count == 15
    assert result.route_checked_self_pair_count == 121
    assert result.authorized_full_pad_root_count == (
        result.possible_earliest_pad_root_count
    )
    assert result.possible_earliest_pad_roots_bound_to_authorized_full_pad
    assert result.blockers
    assert result.audit["dynamic_launch_allowed"] is False
    with pytest.raises(FrozenInstanceError):
        result.object_id = "forged"  # type: ignore[misc]


def test_cross_object_route_collision_input_is_rejected(real_route_inputs) -> None:
    current_aggregate, _pose, accepted, route_state = route_fixtures._build(
        route_fixtures.CURRENT_OBJECT, real_route_inputs
    )
    transfer_aggregate, transfer_pose = real_route_inputs[
        route_fixtures.TRANSFER_OBJECT
    ]
    joint_route = build_candidate_joint_route_contract(
        JOINT_ROUTE_CONTRACT,
        route_state=route_state,
        aggregate_inputs=current_aggregate,
        repository_root=REPOSITORY,
    )
    hand = load_carts_hand_contract(
        route_fixtures.HAND_CONTRACT, repository_root=REPOSITORY
    )
    with pytest.raises(CandidateRouteCollisionError) as error:
        build_candidate_route_collision_certificate(
            ROUTE_COLLISION_CONTRACT,
            accepted_policy=accepted,
            route_state=route_state,
            joint_route=joint_route,
            aggregate_inputs=transfer_aggregate,
            object_world_pose=transfer_pose,
            hand_contract=hand,
            repository_root=REPOSITORY,
        )
    assert error.value.code == "CROSS_ROUTE_OBJECT_MODEL_BINDING_MISMATCH"


def test_route_collision_source_contains_no_runtime_truth_shortcut() -> None:
    source = (
        REPOSITORY
        / "src/kcg_connector/kcg_connector/grasp/robust/"
        "candidate_route_collision.py"
    ).read_text(encoding="utf-8")
    for forbidden in (
        "get_world_pose",
        "set_world_pose",
        "get_contact_report",
        "PhysxContactReportAPI",
        "formal_selected_candidate",
        "display_approximation_role",
        "POSSIBLE_EARLIEST_PAD_ROOT_NOT_IN_ALLOWED_COLLISION_FACE",
        "allowed_hashes",
    ):
        assert forbidden not in source
