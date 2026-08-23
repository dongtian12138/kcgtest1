from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from kcg_connector.grasp.robust.collision_geometry_binding import (
    CLAIM_LIMITATIONS,
    REMAINING_BLOCKERS,
    CollisionGeometryBindingError,
    certify_carts_collision_geometry_bindings,
)
from kcg_connector.grasp.robust.collision_roster import (
    load_authoritative_collision_link_roster,
)
from kcg_connector.grasp.robust.hand_contract import load_carts_hand_contract


REPOSITORY = Path(__file__).resolve().parents[4]
HAND_CONTRACT_PATH = REPOSITORY / "src/kcg_connector/config/carts_hand_contact_v1.yaml"
COLLISION_ROSTER_PATH = (
    REPOSITORY / "src/kcg_connector/config/carts_collision_roster_v1.yaml"
)
EXPECTED_MATERIAL_CERTIFICATE_SHA256 = (
    "a2243016fd2539600cf002fdc6402bb5322b84415f0e8df922e6635ea41bac29",
    "29d0efdba5c0f12a9b7741bc652320610c6657b1386889a7e4b3d66b7cb71ca6",
    "b8ef9078c9657d54e97c609b4ce60e196e91a6b8a3e9126f1d52ed1a1e4ff5c3",
    "6ecd76f79f4ae1bdd81abd2a4b7ee0cc9e684bcf7fadbc2bcd1a1c321696b088",
    "44045a9b922f1ce4252b9fb4a8d5e6f9fa4ce7a9319c68defcc8115749ee4736",
    "718852efd30f000df1920a381ea76c44b304e1f86a3cfae2366e98a7554ebcc1",
    "158353e0e3103bb7b569e188f67ee57d50638d520bd3d89b5d1560adc93a1031",
    "d489204f3f12ed22275c11dbef0cd504e2866ec820aefa385e9699fb9ff9d196",
    "d1b04eb19cc9d5bcb6dce8dcab46273fdb4ffe45fed1f491d05a3105e8449177",
    "5f68ac9b78018dfd6e2f3b0a312998822427e4d3a8acbf556ba69d395fa2cb64",
    "3d8b531894f921cf9fc878f9bfb71ed81f5eefdcfc49c93bd1bda634d336f5d4",
    "5bad61d22ac947e1f182bb457556c9ee1caeaf9a32d58c8414ca662e80170cbf",
    "1139b91ae0f792e1d7a9c1bad72cbd6e6a9fa3f1ab765984558f10018db4d260",
    "6a1645ef9529b2f624e4bf660940b13e14960d10fb3a08888e62a5003b219990",
    "82c0550dc9e324958f4099fa5fdbb71315d9b9265239bf00dc7fb8acaf03d0e7",
    "795fd52eecf9687125c6bfd32be23d00178192740d9a3a9c1e979e86c8e24c47",
    "ba9a28b04b1dc51784948580adefc3369009955afdefe3b6a9b950a8eba6ee08",
)


@pytest.fixture(scope="module")
def verified_binding():
    hand = load_carts_hand_contract(
        HAND_CONTRACT_PATH, repository_root=REPOSITORY
    )
    roster = load_authoritative_collision_link_roster(
        COLLISION_ROSTER_PATH, repository_root=REPOSITORY
    )
    binding = certify_carts_collision_geometry_bindings(hand, roster)
    return hand, roster, binding


def test_all_seventeen_material_boundaries_and_three_terminal_roles_are_bound(
    verified_binding,
) -> None:
    _hand, roster, binding = verified_binding
    assert binding.collision_link_count == 17
    assert binding.self_pair_count == 136
    assert binding.verified_material_boundary_count == 17
    assert binding.verified_terminal_role_binding_count == 3
    assert tuple(
        row.link_name for row in binding.collision_link_material_bindings
    ) == roster.link_names
    assert binding.all_registered_collision_links_bound is True
    assert binding.solid_boundary_binding_complete is True
    assert binding.terminal_pad_role_binding_complete is True


def test_real_material_child_certificates_are_frozen_to_roster_order(
    verified_binding,
) -> None:
    _hand, _roster, binding = verified_binding
    assert tuple(
        row.material_boundary.certificate_sha256
        for row in binding.collision_link_material_bindings
    ) == EXPECTED_MATERIAL_CERTIFICATE_SHA256
    assert all(
        row.material_boundary.formal_material_boundary_eligible
        for row in binding.collision_link_material_bindings
    )
    terminal_by_name = {
        row.link_name: row for row in binding.terminal_role_bindings
    }
    material_by_name = {
        row.link_name: row for row in binding.collision_link_material_bindings
    }
    assert all(
        terminal.material_boundary_certificate_sha256
        == material_by_name[name].material_boundary.certificate_sha256
        for name, terminal in terminal_by_name.items()
    )


def test_motion_and_environment_remain_explicit_blockers(
    verified_binding,
) -> None:
    _hand, _roster, binding = verified_binding
    assert binding.motion_binding_complete is False
    assert binding.environment_binding_complete is False
    assert binding.formal_complete_collision_input_eligible is False
    assert binding.remaining_blockers == REMAINING_BLOCKERS
    assert binding.claim_limitations == CLAIM_LIMITATIONS
    assert binding.audit["formal_complete_collision_input_eligible"] is False
    assert binding.audit["remaining_blockers"] == list(REMAINING_BLOCKERS)


def test_binding_is_deterministic_immutable_and_cannot_claim_motion(
    verified_binding,
) -> None:
    hand, roster, binding = verified_binding
    repeated = certify_carts_collision_geometry_bindings(hand, roster)
    assert binding == repeated
    assert len(binding.certificate_sha256) == 64
    with pytest.raises(FrozenInstanceError):
        binding.motion_binding_complete = True  # type: ignore[misc]
    with pytest.raises(ValueError):
        replace(
            binding,
            motion_binding_complete=True,
            formal_complete_collision_input_eligible=True,
            remaining_blockers=(),
        )


def test_unverified_input_types_cannot_enter_roster_wide_certificate(
    verified_binding,
) -> None:
    hand, roster, _binding = verified_binding
    with pytest.raises(CollisionGeometryBindingError) as hand_error:
        certify_carts_collision_geometry_bindings(object(), roster)  # type: ignore[arg-type]
    assert hand_error.value.code == "VERIFIED_HAND_CONTRACT_REQUIRED"
    with pytest.raises(CollisionGeometryBindingError) as roster_error:
        certify_carts_collision_geometry_bindings(hand, object())  # type: ignore[arg-type]
    assert roster_error.value.code == "VERIFIED_COLLISION_ROSTER_REQUIRED"
