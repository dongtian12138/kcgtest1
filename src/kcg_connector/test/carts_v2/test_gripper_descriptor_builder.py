from pathlib import Path

import numpy as np

from kcg_connector.grasp.carts_v2.gripper_descriptor_builder import (
    build_kcg_graspgenx_descriptors,
    inner_work_aabb,
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


def test_fixed_grid_filters_then_selects_at_most_five_deterministically() -> None:
    _contract, hand = _hand_inputs()
    grid = shared_preshape_grid(hand)
    assert len(grid) == 9
    observed = []
    from_callback = select_preshape_values(
        hand,
        self_collision_free=lambda value: observed.append(value) is None,
    )
    first = select_preshape_values(hand, legal_samples_rad=grid)
    second = select_preshape_values(hand, legal_samples_rad=grid)
    assert tuple(observed) == grid
    assert from_callback == first == second
    assert len(first) == 5
    assert grid[int(np.argmin(np.abs(np.asarray(grid) - 0.70)))] in first


def test_descriptor_frame_is_inverse_orthonormal_and_right_handed() -> None:
    contract, hand = _hand_inputs()
    descriptors = build_kcg_graspgenx_descriptors(
        contract,
        hand,
        maximum_closure_phase=0.50,
        legal_samples_rad=shared_preshape_grid(hand),
    )
    assert len(descriptors) == 5
    for descriptor in descriptors:
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
        maximum_closure_phase=0.50,
        legal_samples_rad=shared_preshape_grid(hand),
    )
    for descriptor in descriptors:
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


def test_mimic_midpoint_and_conditioning_aabbs_bind_real_pad_points() -> None:
    contract, hand = _hand_inputs()
    descriptor = build_kcg_graspgenx_descriptors(
        contract,
        hand,
        maximum_closure_phase=0.50,
        legal_samples_rad=shared_preshape_grid(hand),
    )[0]
    for state in (
        descriptor.open_joint_positions_rad,
        descriptor.half_joint_positions_rad,
        descriptor.close_joint_positions_rad,
    ):
        assert state["f3j1"] == state["f1j1"]
        assert state["f1j3"] == state["f1j2"]
        assert state["f2j2"] == state["f2j1"]
        assert state["f3j3"] == state["f3j2"]
        hand.resolve_joint_positions(state, enforce_limits=True)
    for name in descriptor.open_joint_positions_rad:
        assert np.isclose(
            descriptor.half_joint_positions_rad[name],
            0.5
            * (
                descriptor.open_joint_positions_rad[name]
                + descriptor.close_joint_positions_rad[name]
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
    assert config["type"] == "revolute_3f"
    assert config["symmetric"] is False
    assert set(config["sweep_volume"]) == {"extents", "offset", "extents2", "offset2"}
