from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest

from kcg_connector.grasp.robust.aggregate_collision_inputs import (
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
from kcg_connector.grasp.robust.object_world_pose import (
    CLAIM_LIMITATIONS,
    ObjectWorldPoseError,
    certify_settled_object_world_pose,
)
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
PLACEMENT_CONTRACT = (
    REPOSITORY / "src/kcg_connector/config/carts_shared_object_placement_v1.yaml"
)
CURRENT_OBJECT = "current_d38999_26kj61sn_public_spec"
TRANSFER_OBJECT = "te_deutsch_d38999_26fj35pn_step"


@pytest.fixture(scope="module")
def real_pose_certificates():
    hand = load_carts_hand_contract(HAND_CONTRACT, repository_root=REPOSITORY)
    roster = load_authoritative_collision_link_roster(
        COLLISION_ROSTER, repository_root=REPOSITORY
    )
    geometry = certify_carts_collision_geometry_bindings(hand, roster)
    environment = load_shared_table_fixture_world(
        SHARED_ENVIRONMENT, repository_root=REPOSITORY
    )
    options = IntervalArithmeticOptions(
        decimal_precision=80,
        maximum_root_bisection_iterations=256,
    )
    results = {}
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
        results[object_id] = (aggregate, pose)
    return results


def test_current_object_reproduces_recorded_settled_pose_from_geometry(
    real_pose_certificates,
) -> None:
    _aggregate, pose = real_pose_certificates[CURRENT_OBJECT]
    np.testing.assert_allclose(
        pose.world_from_object[:3, 3],
        (0.520, -0.210, 0.2305),
        rtol=0.0,
        atol=2.0e-9,
    )
    np.testing.assert_allclose(
        pose.orientation_degrees_xyz,
        (180.0, 0.0, 0.0),
        rtol=0.0,
        atol=0.0,
    )
    assert pose.transformed_bounds_world_m[0][2] == pytest.approx(
        0.200, abs=2.0e-16
    )
    assert pose.audit["static_settled_pose_binding_complete"] is True


def test_both_objects_use_one_station_and_geometry_derived_height(
    real_pose_certificates,
) -> None:
    current = real_pose_certificates[CURRENT_OBJECT][1]
    transfer = real_pose_certificates[TRANSFER_OBJECT][1]
    assert current.station_xy_m == transfer.station_xy_m == (0.520, -0.210)
    np.testing.assert_array_equal(
        current.world_from_object[:3, :3], transfer.world_from_object[:3, :3]
    )
    assert current.world_from_object[2, 3] != transfer.world_from_object[2, 3]
    assert transfer.world_from_object[2, 3] == pytest.approx(0.200, abs=2.0e-16)
    for pose in (current, transfer):
        assert pose.transformed_bounds_world_m[0][2] == pytest.approx(
            pose.table_top_z_m, abs=2.0e-16
        )
        assert pose.table_contact_gap_m == 0.0
        assert pose.candidate_route_included is False
        assert pose.isaac_dynamic_state_included is False
        assert pose.hardware_state_included is False
        assert pose.post_start_object_pose_write_allowed is False
        assert pose.claim_limitations == CLAIM_LIMITATIONS


def test_pose_certificate_is_deterministic_and_immutable(
    real_pose_certificates,
) -> None:
    aggregate, pose = real_pose_certificates[CURRENT_OBJECT]
    repeated = certify_settled_object_world_pose(
        PLACEMENT_CONTRACT,
        aggregate_inputs=aggregate,
        repository_root=REPOSITORY,
    )
    assert repeated.certificate_sha256 == pose.certificate_sha256
    assert not pose.world_from_object.flags.writeable
    with pytest.raises(FrozenInstanceError):
        pose.object_id = TRANSFER_OBJECT  # type: ignore[misc]
    with pytest.raises(ValueError):
        replace(pose, candidate_route_included=True)


def test_loader_rejects_unverified_aggregate_and_changed_source_hash(
    real_pose_certificates,
    tmp_path: Path,
) -> None:
    with pytest.raises(ObjectWorldPoseError) as type_error:
        certify_settled_object_world_pose(
            PLACEMENT_CONTRACT,
            aggregate_inputs=object(),  # type: ignore[arg-type]
            repository_root=REPOSITORY,
        )
    assert type_error.value.code == "VERIFIED_AGGREGATE_INPUT_REQUIRED"

    aggregate, _pose = real_pose_certificates[CURRENT_OBJECT]
    changed = tmp_path / "placement.yaml"
    changed.write_text(
        PLACEMENT_CONTRACT.read_text(encoding="utf-8").replace(
            "5600da0ff12c8ca00afa0a4e46f40d2edfde684b231bf580b2bb26678e4e0457",
            "0" * 64,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ObjectWorldPoseError) as hash_error:
        certify_settled_object_world_pose(
            changed,
            aggregate_inputs=aggregate,
            repository_root=REPOSITORY,
        )
    assert hash_error.value.code == "SOURCE_SHA256_MISMATCH"


def test_placement_path_does_not_import_old_candidate_or_runtime_truth() -> None:
    source = (
        REPOSITORY
        / "src/kcg_connector/kcg_connector/grasp/robust/object_world_pose.py"
    ).read_text(encoding="utf-8")
    assert "d38999_keyed_v2_tabletop_pick_v1.yaml" not in source
    assert "proposed_clearance_arm_rad" not in source
    assert "grasp_arm_rad" not in source
    assert "get_world_pose" not in source
    assert "set_world_pose" not in source
