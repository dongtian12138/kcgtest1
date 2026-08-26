"""Regressions for the user-corrected three-role connector surface semantics."""
from pathlib import Path

import numpy as np
import pytest

from kcg_connector.grasp.carts_v2.models import (
    FACE_ROLE_NAMES, HARD_FORBIDDEN, PRIMARY_GRIP, SECONDARY_GRIP,
    load_v2_inputs,
)


ROOT = Path(__file__).resolve().parents[4]
CONFIG = ROOT / "src/kcg_connector/config/carts_surface_v2_fast6h.yaml"
OBJECTS = {
    "current_d38999_26kj61sn_public_spec": (10390, 8054, 127144),
    "te_deutsch_d38999_26fj35pn_step": (7269, 132, 679635),
}


@pytest.mark.parametrize("object_id, expected_counts", OBJECTS.items())
def test_three_roles_partition_every_registered_source_face(
    object_id: str, expected_counts: tuple[int, int, int]
) -> None:
    inputs = load_v2_inputs(ROOT, config_path=CONFIG, object_id=object_id)
    role_map = inputs.face_roles
    counts = tuple(int(np.sum(role_map.face_role == code)) for code in range(3))
    assert counts == expected_counts
    assert np.array_equal(role_map.face_is_allowed,
                          role_map.face_role != HARD_FORBIDDEN)
    assert np.array_equal(role_map.legacy_radial_only_face_is_allowed,
                          role_map.face_role == PRIMARY_GRIP)
    assert np.array_equal(role_map.reason_code == 0, role_map.face_is_allowed)


def test_tangential_outer_wall_is_secondary_but_old_axial_faces_remain_hard() -> None:
    inputs = load_v2_inputs(
        ROOT, config_path=CONFIG, object_id="te_deutsch_d38999_26fj35pn_step")
    roles = inputs.face_roles.face_role
    assert FACE_ROLE_NAMES[int(roles[21232])] == "SECONDARY_GRIP"
    assert roles[21232] == SECONDARY_GRIP
    assert inputs.face_roles.legacy_radial_only_face_is_allowed[21232] == 0
    assert all(roles[index] == HARD_FORBIDDEN for index in (
        33484, 33457, 33462, 29383, 33432, 33408, 30940))
