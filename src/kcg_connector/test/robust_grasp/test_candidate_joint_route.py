from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import test_candidate_route as route_fixtures

from kcg_connector.grasp.robust.candidate_joint_route import (
    CLAIM_LIMITATIONS,
    CandidateJointRouteError,
    build_candidate_joint_route_contract,
)
from kcg_connector.grasp.robust.candidate_route import (
    build_candidate_route_state_contract,
)


REPOSITORY = route_fixtures.REPOSITORY
JOINT_ROUTE_CONTRACT = (
    REPOSITORY
    / "src/kcg_connector/config/carts_candidate_joint_route_v1.yaml"
)
real_route_inputs = route_fixtures.real_route_inputs


def _build(object_id: str, inputs):
    aggregate, _pose, _accepted, route_state = route_fixtures._build(
        object_id, inputs
    )
    joint_route = build_candidate_joint_route_contract(
        JOINT_ROUTE_CONTRACT,
        route_state=route_state,
        aggregate_inputs=aggregate,
        repository_root=REPOSITORY,
    )
    return aggregate, route_state, joint_route


def _orientation_error_rad(actual: np.ndarray, target: np.ndarray) -> float:
    return float(
        np.linalg.norm(
            Rotation.from_matrix(
                target[:3, :3].T @ actual[:3, :3]
            ).as_rotvec()
        )
    )


def test_both_objects_get_bounded_complete_eleven_joint_routes(
    real_route_inputs,
) -> None:
    certificates = {}
    for object_id in (
        route_fixtures.CURRENT_OBJECT,
        route_fixtures.TRANSFER_OBJECT,
    ):
        aggregate, route_state, contract = _build(
            object_id, real_route_inputs
        )
        model = aggregate.kinematic_binding.model
        certificates[object_id] = contract.certificate_sha256
        assert contract.object_id == object_id
        assert contract.complete_joint_names == tuple(
            model.independent_joint_names
        )
        assert contract.arm_ik_solution_complete is True
        assert contract.candidate_specific_motion_binding_complete is True
        assert contract.complete_eleven_joint_route_binding_complete is True
        assert contract.contact_triggered_hand_intervals_preserved is True
        assert contract.joint_limits_complete is True
        assert contract.joint_step_bound_complete is True
        assert contract.continuous_collision_complete is False
        assert contract.controller_execution_authorized is False
        assert contract.formal_selection_input_used is False
        assert contract.claim_limitations == CLAIM_LIMITATIONS

        approach = contract.approach_independent_joint_waypoints_rad
        assert approach[0] == route_state.home_independent_joint_positions_rad
        assert approach[-1] == (
            contract.pregrasp_independent_joint_positions_rad
        )
        assert (
            contract.maximum_observed_approach_arm_joint_step_rad
            <= contract.approach_maximum_arm_joint_step_rad
        )
        for waypoint in approach:
            assert model.within_joint_limits(waypoint)

        for target, intervals, position_error, orientation_error in zip(
            contract.lift_world_from_hand_targets,
            contract.lift_independent_joint_interval_waypoints_rad,
            contract.lift_position_errors_m,
            contract.lift_orientation_errors_rad,
        ):
            arm = tuple(interval.lower for interval in intervals[:7])
            assert all(
                interval.lower == interval.upper
                for interval in intervals[:7]
            )
            actual = model.forward_kinematics(
                arm + route_state.pregrasp_hand_joint_positions_rad
            )[route_state.hand_base_link]
            assert (
                float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
                <= contract.position_tolerance_m
            )
            assert (
                _orientation_error_rad(actual, target)
                <= contract.orientation_tolerance_rad
            )
            assert position_error <= contract.position_tolerance_m
            assert orientation_error <= contract.orientation_tolerance_rad
        assert (
            contract.maximum_observed_lift_arm_joint_step_rad
            <= contract.lift_maximum_arm_joint_step_rad
        )

        swept = (
            contract.closure_stage_swept_independent_joint_intervals_rad
        )
        endpoints = (
            contract.closure_stage_endpoint_independent_joint_intervals_rad
        )
        for stage_index, hand_index in enumerate((1, 2, 3)):
            complete_index = 7 + hand_index
            assert swept[stage_index][complete_index] == (
                route_state.closure_swept_hand_joint_intervals_rad[hand_index]
            )
            assert endpoints[stage_index][complete_index] == (
                route_state.closure_contact_hand_joint_intervals_rad[
                    hand_index
                ]
            )
            assert (
                endpoints[stage_index][complete_index].upper
                > endpoints[stage_index][complete_index].lower
            )
        assert tuple(
            contract.lift_independent_joint_interval_waypoints_rad[0][
                7 + index
            ]
            for index in range(4)
        ) == route_state.closure_contact_hand_joint_intervals_rad

    assert (
        certificates[route_fixtures.CURRENT_OBJECT]
        != certificates[route_fixtures.TRANSFER_OBJECT]
    )


def test_joint_route_is_binary64_deterministic_and_immutable(
    real_route_inputs,
) -> None:
    aggregate, route_state, first = _build(
        route_fixtures.CURRENT_OBJECT, real_route_inputs
    )
    second = build_candidate_joint_route_contract(
        JOINT_ROUTE_CONTRACT,
        route_state=route_state,
        aggregate_inputs=aggregate,
        repository_root=REPOSITORY,
    )
    assert first.certificate_sha256 == second.certificate_sha256
    assert (
        first.approach_independent_joint_waypoints_rad
        == second.approach_independent_joint_waypoints_rad
    )
    assert (
        first.lift_independent_joint_interval_waypoints_rad
        == second.lift_independent_joint_interval_waypoints_rad
    )
    assert all(
        not matrix.flags.writeable
        for matrix in first.lift_world_from_hand_targets
    )
    with pytest.raises(FrozenInstanceError):
        first.object_id = route_fixtures.TRANSFER_OBJECT  # type: ignore[misc]


def test_cross_object_and_unverified_route_inputs_fail_closed(
    real_route_inputs,
) -> None:
    current_aggregate, _pose, _accepted, current_route = (
        route_fixtures._build(
            route_fixtures.CURRENT_OBJECT, real_route_inputs
        )
    )
    transfer_aggregate = real_route_inputs[
        route_fixtures.TRANSFER_OBJECT
    ][0]
    with pytest.raises(CandidateJointRouteError) as cross_object:
        build_candidate_joint_route_contract(
            JOINT_ROUTE_CONTRACT,
            route_state=current_route,
            aggregate_inputs=transfer_aggregate,
            repository_root=REPOSITORY,
        )
    assert cross_object.value.code == "ROUTE_AND_AGGREGATE_BINDING_MISMATCH"

    for invalid in (None, object()):
        with pytest.raises(CandidateJointRouteError) as unverified:
            build_candidate_joint_route_contract(
                JOINT_ROUTE_CONTRACT,
                route_state=invalid,
                aggregate_inputs=current_aggregate,
                repository_root=REPOSITORY,
            )
        assert unverified.value.code == "CANDIDATE_ROUTE_STATE_REQUIRED"


def test_unreachable_target_is_rejected_without_clamping(
    real_route_inputs,
    tmp_path: Path,
) -> None:
    aggregate, pose = real_route_inputs[route_fixtures.CURRENT_OBJECT]
    accepted = route_fixtures._accepted_policy(aggregate)
    far_transform = np.eye(4, dtype=np.float64)
    far_transform[0, 3] = 10.0
    far_policy = replace(
        accepted.sequential_closure_policy,
        object_from_hand=tuple(float(value) for value in far_transform.ravel()),
    )
    far_accepted = replace(
        accepted,
        sequential_closure_policy=far_policy,
    )
    route_state = build_candidate_route_state_contract(
        route_fixtures.ROUTE_CONTRACT,
        accepted_policy=far_accepted,
        aggregate_inputs=aggregate,
        object_world_pose=pose,
        repository_root=REPOSITORY,
    )
    fast_reject_config = tmp_path / "joint_route_fast_reject.yaml"
    fast_reject_config.write_text(
        JOINT_ROUTE_CONTRACT.read_text(encoding="utf-8").replace(
            "maximum_function_evaluations: 2000",
            "maximum_function_evaluations: 20",
        ),
        encoding="utf-8",
    )
    with pytest.raises(CandidateJointRouteError) as unreachable:
        build_candidate_joint_route_contract(
            fast_reject_config,
            route_state=route_state,
            aggregate_inputs=aggregate,
            repository_root=tmp_path,
        )
    assert unreachable.value.code == "IK_TARGET_UNREACHABLE"


def test_joint_route_source_has_no_legacy_or_runtime_truth_shortcut() -> None:
    source = (
        REPOSITORY
        / "src/kcg_connector/kcg_connector/grasp/robust/"
        "candidate_joint_route.py"
    ).read_text(encoding="utf-8")
    assert "connector_home_to_pregrasp_v1.yaml" not in source
    assert "proposed_clearance_arm_rad" not in source
    assert "grasp_arm_rad" not in source
    assert "get_world_pose" not in source
    assert "set_world_pose" not in source
