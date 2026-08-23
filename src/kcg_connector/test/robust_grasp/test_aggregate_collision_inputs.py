from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest

from kcg_connector.grasp.robust.aggregate_collision_inputs import (
    CLAIM_LIMITATIONS,
    EXPECTED_INDEPENDENT_JOINTS,
    REMAINING_BLOCKERS,
    AggregateCollisionInputError,
    build_carts_aggregate_collision_runtime_inputs,
)
from kcg_connector.grasp.robust.collision_geometry_binding import (
    certify_carts_collision_geometry_bindings,
)
from kcg_connector.grasp.robust.collision_roster import (
    load_authoritative_collision_link_roster,
)
from kcg_connector.grasp.robust.hand_contract import load_carts_hand_contract
from kcg_connector.grasp.robust.interval_kinematics import (
    IntervalArithmeticOptions,
)
from kcg_connector.grasp.robust.object_contract import load_object_contract
from kcg_connector.grasp.robust.shared_environment import (
    load_shared_table_fixture_world,
)


REPOSITORY = Path(__file__).resolve().parents[4]
HAND_CONTRACT = REPOSITORY / "src/kcg_connector/config/carts_hand_contact_v1.yaml"
COLLISION_ROSTER = REPOSITORY / "src/kcg_connector/config/carts_collision_roster_v1.yaml"
OBJECT_CONTRACT = REPOSITORY / "src/kcg_connector/config/carts_grasp_objects_v1.yaml"
SHARED_ENVIRONMENT = (
    REPOSITORY / "src/kcg_connector/config/carts_shared_table_fixture_world_v1.yaml"
)
CURRENT_OBJECT = "current_d38999_26kj61sn_public_spec"
TRANSFER_OBJECT = "te_deutsch_d38999_26fj35pn_step"


@pytest.fixture(scope="module")
def real_runtime_inputs():
    hand = load_carts_hand_contract(HAND_CONTRACT, repository_root=REPOSITORY)
    roster = load_authoritative_collision_link_roster(
        COLLISION_ROSTER,
        repository_root=REPOSITORY,
    )
    geometry = certify_carts_collision_geometry_bindings(hand, roster)
    object_contract = load_object_contract(
        OBJECT_CONTRACT,
        object_id=CURRENT_OBJECT,
        repository_root=REPOSITORY,
    )
    options = IntervalArithmeticOptions(
        decimal_precision=80,
        maximum_root_bisection_iterations=256,
    )
    environment = load_shared_table_fixture_world(
        SHARED_ENVIRONMENT,
        repository_root=REPOSITORY,
    )
    result = build_carts_aggregate_collision_runtime_inputs(
        hand_contract=hand,
        collision_roster=roster,
        geometry_binding=geometry,
        object_contract=object_contract,
        shared_environment=environment,
        interval_options=options,
    )
    return hand, roster, geometry, object_contract, options, environment, result


def test_verified_sources_build_one_world_to_terminal_kinematic_tree(
    real_runtime_inputs,
) -> None:
    _hand, roster, _geometry, _object, _options, _environment, result = (
        real_runtime_inputs
    )
    kinematics = result.kinematic_binding
    assert kinematics.model.base_link == "world"
    assert kinematics.independent_joint_names == EXPECTED_INDEPENDENT_JOINTS
    assert kinematics.independent_joint_count == 11
    assert kinematics.collision_link_names == roster.link_names
    assert kinematics.collision_link_count == 17
    assert kinematics.every_collision_link_connected_to_world is True

    positions = tuple(
        0.5
        * (
            kinematics.model.independent_joint_limits[name].lower
            + kinematics.model.independent_joint_limits[name].upper
        )
        for name in kinematics.model.independent_joint_names
    )
    transforms = kinematics.model.forward_kinematics(positions)
    assert set(roster.link_names) <= set(transforms)
    assert all(
        transform.shape == (4, 4) and np.all(np.isfinite(transform))
        for name, transform in transforms.items()
        if name in roster.link_names
    )
    backend = kinematics.new_interval_backend()
    assert backend.hand_model is kinematics.model
    assert backend.options.decimal_precision == 80
    assert backend.options.maximum_root_bisection_iterations == 256


def test_all_real_collision_surfaces_and_terminal_roles_are_runtime_bound(
    real_runtime_inputs,
) -> None:
    _hand, roster, geometry, _object, _options, _environment, result = (
        real_runtime_inputs
    )
    assert result.collision_link_count == 17
    assert result.terminal_role_count == 3
    assert result.self_pair_count == 136
    assert tuple(row.link_name for row in result.link_surfaces) == roster.link_names
    assert result.self_pair_inventory.all_pairs == roster.all_self_pairs

    material_by_name = {
        row.link_name: row
        for row in geometry.collision_link_material_bindings
    }
    for surface in result.link_surfaces:
        material = material_by_name[surface.link_name]
        assert surface.source_asset_sha256 == material.collision_mesh_sha256
        assert len(surface.triangles_link_m) == (
            material.material_boundary.source_face_count
        )

    assert tuple(row.link_name for row in result.terminal_roles) == (
        "f1Link3",
        "f2Link2",
        "f3Link3",
    )
    for terminal in result.terminal_roles:
        assert len(terminal.full_surface.triangles_link_m) == 1100
        assert len(terminal.allowed_pad_surface.triangles_link_m) == 16
        assert len(terminal.forbidden_nonpad_surface.triangles_link_m) == 1084
        assert terminal.terminal_certificate.formal_terminal_role_binding_eligible


def test_current_object_surface_and_material_boundary_are_both_bound(
    real_runtime_inputs,
) -> None:
    _hand, _roster, _geometry, object_contract, _options, _environment, result = (
        real_runtime_inputs
    )
    assert result.object_id == CURRENT_OBJECT
    assert result.object_surface.source_asset_sha256 == (
        object_contract.verified_source_sha256["planning_geometry"]
    )
    assert len(result.object_surface.triangles_object_m) == 145588
    assert result.object_planning_surface_binding_complete is True
    assert result.object_material_boundary is (
        object_contract.material_boundary_evidence
    )
    assert result.object_material_boundary.certificate.source_face_count == 145588
    assert result.object_material_boundary_binding_complete is True
    assert result.audit["object_material_boundary_binding_complete"] is True


def test_shared_environment_is_bound_while_route_remains_fail_closed(
    real_runtime_inputs,
) -> None:
    _hand, _roster, _geometry, _object, _options, environment, result = (
        real_runtime_inputs
    )
    assert result.aggregate_robot_kinematics_binding_complete is True
    assert result.runtime_link_surface_binding_complete is True
    assert result.terminal_runtime_role_binding_complete is True
    assert result.candidate_specific_motion_binding_complete is False
    assert result.shared_environment is environment
    assert result.table_fixture_environment_binding_complete is True
    assert result.shared_environment.root_frame == "world"
    assert result.shared_environment.robot_base_origin_m == (0.0, 0.0, 0.0)
    assert result.shared_environment.obstacle_count == 2
    assert sum(
        len(obstacle.triangles_world_m)
        for obstacle in result.shared_environment.obstacles
    ) == 24
    assert result.continuous_pad_contact_binding_complete is False
    assert result.formal_complete_collision_input_eligible is False
    assert result.remaining_blockers == REMAINING_BLOCKERS
    assert result.claim_limitations == CLAIM_LIMITATIONS
    assert result.audit["formal_complete_collision_input_eligible"] is False
    assert len(result.certificate_sha256) == 64

    with pytest.raises(FrozenInstanceError):
        result.candidate_specific_motion_binding_complete = True  # type: ignore[misc]
    with pytest.raises(ValueError):
        replace(
            result,
            candidate_specific_motion_binding_complete=True,
            formal_complete_collision_input_eligible=True,
            remaining_blockers=(),
        )


def test_runtime_binding_is_deterministic_and_rejects_unverified_types(
    real_runtime_inputs,
) -> None:
    hand, roster, geometry, object_contract, options, environment, result = (
        real_runtime_inputs
    )
    repeated = build_carts_aggregate_collision_runtime_inputs(
        hand_contract=hand,
        collision_roster=roster,
        geometry_binding=geometry,
        object_contract=object_contract,
        shared_environment=environment,
        interval_options=options,
    )
    assert repeated.certificate_sha256 == result.certificate_sha256
    assert repeated.kinematic_binding.model_sha256 == (
        result.kinematic_binding.model_sha256
    )
    assert tuple(row.geometry_sha256 for row in repeated.link_surfaces) == tuple(
        row.geometry_sha256 for row in result.link_surfaces
    )

    with pytest.raises(AggregateCollisionInputError) as error:
        build_carts_aggregate_collision_runtime_inputs(
            hand_contract=object(),  # type: ignore[arg-type]
            collision_roster=roster,
            geometry_binding=geometry,
            object_contract=object_contract,
            shared_environment=environment,
            interval_options=options,
        )
    assert error.value.code == "VERIFIED_HAND_CONTRACT_REQUIRED"

    with pytest.raises(AggregateCollisionInputError) as environment_error:
        build_carts_aggregate_collision_runtime_inputs(
            hand_contract=hand,
            collision_roster=roster,
            geometry_binding=geometry,
            object_contract=object_contract,
            shared_environment=object(),  # type: ignore[arg-type]
            interval_options=options,
        )
    assert environment_error.value.code == "VERIFIED_SHARED_ENVIRONMENT_REQUIRED"


def test_transfer_object_uses_the_same_robot_and_shared_environment(
    real_runtime_inputs,
) -> None:
    hand, roster, geometry, _current, options, environment, current_result = (
        real_runtime_inputs
    )
    transfer = load_object_contract(
        OBJECT_CONTRACT,
        object_id=TRANSFER_OBJECT,
        repository_root=REPOSITORY,
    )
    transfer_result = build_carts_aggregate_collision_runtime_inputs(
        hand_contract=hand,
        collision_roster=roster,
        geometry_binding=geometry,
        object_contract=transfer,
        shared_environment=environment,
        interval_options=options,
    )

    assert transfer_result.object_id == TRANSFER_OBJECT
    assert transfer_result.object_material_boundary_binding_complete is True
    assert transfer_result.table_fixture_environment_binding_complete is True
    assert transfer_result.shared_environment is environment
    assert transfer_result.kinematic_binding.model_sha256 == (
        current_result.kinematic_binding.model_sha256
    )
    assert transfer_result.remaining_blockers == REMAINING_BLOCKERS
    assert transfer_result.formal_complete_collision_input_eligible is False
