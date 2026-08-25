from pathlib import Path

import numpy as np
import pytest

from kcg_connector.grasp.carts_v2.gripper_descriptor_builder import (
    build_kcg_graspgenx_descriptors,
    build_legacy_kcg_graspgenx_descriptors,
    inner_work_aabb,
    legacy_shared_preshape_grid,
    pad_points_in_graspgenx,
    select_preshape_values,
    shared_preshape_grid,
)
from kcg_connector.grasp.robust.hand_contract import load_carts_hand_contract


ROOT = Path(__file__).resolve().parents[4]
HAND_CONTRACT = ROOT / "src/kcg_connector/config/carts_hand_contact_v1.yaml"


def _hand_inputs():
    contract = load_carts_hand_contract(HAND_CONTRACT, repository_root=ROOT)
    return contract, contract.build_hand_model()


def _uniform_conditioning_phases(hand, value=0.50):
    return {preshape: value for preshape in shared_preshape_grid(hand)}


def test_full_palm_grid_has_endpoints_anchors_and_legacy_baseline() -> None:
    contract, hand = _hand_inputs()
    grid = shared_preshape_grid(hand)
    limit = hand.independent_joint_limits["f1j1"]
    assert len(grid) == 91
    assert grid[0] == limit.lower
    assert grid[-1] == limit.upper
    assert np.allclose(
        np.asarray(grid)[[0, 30, 45, 60, 90]],
        limit.lower
        + (limit.upper - limit.lower) * np.asarray((0.0, 1 / 3, 0.5, 2 / 3, 1.0)),
    )
    with pytest.raises(ValueError, match="exactly 91"):
        shared_preshape_grid(hand, sample_count=90)

    incomplete = _uniform_conditioning_phases(hand)
    incomplete.pop(grid[-1])
    with pytest.raises(ValueError, match="all 91"):
        build_kcg_graspgenx_descriptors(
            contract,
            hand,
            conditioning_close_phase_by_palm=incomplete,
        )

    legacy_grid = legacy_shared_preshape_grid(hand)
    observed = []
    from_callback = select_preshape_values(
        hand,
        self_collision_free=lambda value: observed.append(value) is None,
    )
    first = select_preshape_values(hand, legal_samples_rad=legacy_grid)
    second = select_preshape_values(hand, legal_samples_rad=legacy_grid)
    assert tuple(observed) == legacy_grid
    assert from_callback == first == second
    assert len(first) == 5
    assert legacy_grid[
        int(np.argmin(np.abs(np.asarray(legacy_grid) - 0.70)))
    ] in first
    legacy = build_legacy_kcg_graspgenx_descriptors(
        contract,
        hand,
        closure_phase_by_preshape={value: 0.5 for value in legacy_grid},
        legal_samples_rad=legacy_grid,
    )
    assert len(legacy) == 5
    assert [row.descriptor_id for row in legacy] == [
        f"kcg_3f_preshape_{index:02d}" for index in range(5)
    ]


def test_all_91_descriptor_frames_are_explicit_inverse_and_right_handed() -> None:
    contract, hand = _hand_inputs()
    descriptors = build_kcg_graspgenx_descriptors(
        contract,
        hand,
        conditioning_close_phase_by_palm=_uniform_conditioning_phases(hand),
    )
    grid = shared_preshape_grid(hand)
    assert len(descriptors) == 91
    for index, descriptor in enumerate(descriptors):
        assert descriptor.descriptor_id == f"kcg_3f_palm_{index:03d}"
        assert descriptor.palm_configuration_rad == grid[index]
        assert descriptor.conditioning_close_phase == 0.5
        forward = descriptor.frame.handbase_from_graspgenx
        inverse = descriptor.frame.graspgenx_from_handbase
        assert np.allclose(forward @ inverse, np.eye(4), atol=1e-12)
        assert np.allclose(forward[:3, :3].T @ forward[:3, :3], np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(forward[:3, :3]), 1.0, atol=1e-12)


def test_descriptor_origin_is_registered_proximal_joint_plane() -> None:
    contract, hand = _hand_inputs()
    descriptors = build_kcg_graspgenx_descriptors(
        contract,
        hand,
        conditioning_close_phase_by_palm=_uniform_conditioning_phases(hand),
    )
    for descriptor in descriptors[::15]:
        forward = descriptor.frame.handbase_from_graspgenx
        links = hand.forward_kinematics(descriptor.open_joint_positions_rad)
        origins = []
        for finger in hand.fingers.values():
            joint = hand.joints[finger.joint_names[0]]
            origins.append(
                (links[joint.parent_link] @ joint.origin_transform())[:3, 3]
            )
        z_axis = forward[:3, 2]
        expected = z_axis * float(np.mean(np.asarray(origins) @ z_axis))
        assert np.allclose(forward[:3, 3], expected, atol=1e-12)
        assert np.allclose(descriptor.fingertip_graspgenx_m[:2], 0.0, atol=1e-12)
        assert 0.07 < descriptor.fingertip_graspgenx_m[2] < 0.13
        pad_depths = []
        transforms = hand.pad_transforms(descriptor.open_joint_positions_rad)
        for pad in contract.pads:
            points_handbase = (
                pad.points_local_m @ transforms[pad.name][:3, :3].T
                + transforms[pad.name][:3, 3]
            )
            points_generator = (
                points_handbase
                @ descriptor.frame.graspgenx_from_handbase[:3, :3].T
                + descriptor.frame.graspgenx_from_handbase[:3, 3]
            )
            pad_depths.append(float(np.max(points_generator[:, 2])))
        assert np.isclose(
            descriptor.fingertip_graspgenx_m[2], np.mean(pad_depths), atol=1e-12
        )


def test_mimic_midpoint_and_conditioning_aabbs_bind_real_pad_points() -> None:
    contract, hand = _hand_inputs()
    descriptor = build_kcg_graspgenx_descriptors(
        contract,
        hand,
        conditioning_close_phase_by_palm=_uniform_conditioning_phases(hand),
    )
    assert len(descriptor) == 91
    for row in descriptor:
        for state in (
            row.open_joint_positions_rad,
            row.half_joint_positions_rad,
            row.conditioning_close_joint_positions_rad,
        ):
            assert state["f1j1"] == row.palm_configuration_rad
            assert state["f3j1"] == state["f1j1"]
            assert state["f1j3"] == state["f1j2"]
            assert state["f2j2"] == state["f2j1"]
            assert state["f3j3"] == state["f3j2"]
            hand.resolve_joint_positions(state, enforce_limits=True)
    descriptor = descriptor[45]
    for name in descriptor.open_joint_positions_rad:
        assert np.isclose(
            descriptor.half_joint_positions_rad[name],
            0.5
            * (
                descriptor.open_joint_positions_rad[name]
                + descriptor.conditioning_close_joint_positions_rad[name]
            ),
        )
    for state, extents, offset in (
        (
            descriptor.open_joint_positions_rad,
            descriptor.open_aabb_extents_m,
            descriptor.open_aabb_offset_m,
        ),
        (
            descriptor.half_joint_positions_rad,
            descriptor.half_aabb_extents_m,
            descriptor.half_aabb_offset_m,
        ),
    ):
        points = pad_points_in_graspgenx(contract, hand, state, descriptor.frame)
        lower = np.asarray(offset) - 0.5 * np.asarray(extents)
        upper = np.asarray(offset) + 0.5 * np.asarray(extents)
        assert np.all(np.asarray(extents) > 0.0)
        assert np.all(points[:, 1:] >= lower[1:] - 1e-12)
        assert np.all(points[:, 1:] <= upper[1:] + 1e-12)
        whole_extents = np.ptp(points, axis=0)
        assert extents[0] < whole_extents[0]
        assert np.allclose(extents[1:], whole_extents[1:])
        assert inner_work_aabb(contract, hand, state, descriptor.frame) == (
            extents,
            offset,
        )

    config = descriptor.to_official_config()
    assert not hasattr(descriptor, "maximum_closure_phase")
    assert config["close"] == dict(
        descriptor.conditioning_close_joint_positions_rad
    )
    assert config["type"] == "revolute_3f"
    assert config["symmetric"] is False
    assert set(config["sweep_volume"]) == {"extents", "offset", "extents2", "offset2"}
