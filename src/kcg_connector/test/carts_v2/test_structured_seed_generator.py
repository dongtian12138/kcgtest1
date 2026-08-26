"""Direct regressions for the fixed CONTACTOPT structured seed design."""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from kcg_connector.grasp.carts_v2.models import PRIMARY_GRIP
from kcg_connector.grasp.carts_v2.structured_seed_generator import (
    _compatible_target_triplet,
    _radial_support_target_mask,
    _reject_record,
    structured_seed_specifications,
)
from kcg_connector.grasp.carts_v2.three_contact_pose_initializer import (
    contact_coordinates,
    initialize_three_contact_pose,
    kabsch_rigid_alignment,
    resolve_palm_configuration_rad,
)


def test_strict_1040_plus_448_design_is_complete_and_deterministic() -> None:
    first, second = structured_seed_specifications(), structured_seed_specifications()
    assert len(first) == 1040 + 448 == 1488
    assert [row.candidate_id for row in first] == [row.candidate_id for row in second]
    assert len({row.candidate_id for row in first}) == 1488
    assert Counter(row.family for row in first) == {"GLOBAL": 1040, "DENSE": 448}
    assert sorted({row.qp_deg for row in first if row.family == "GLOBAL"}) == [
        0.0, 15.0, 30.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0,
        75.0, 82.5, 90.0,
    ]
    assert Counter(row.preshape_id for row in first) == {
        "P0": 1040, "P1": 224, "P2": 224,
    }
    dense_groups = Counter(
        (row.qp_index, row.axial_index, row.preshape_index)
        for row in first if row.family == "DENSE"
    )
    assert set(dense_groups.values()) == {8}


def test_kabsch_then_normal_alignment_recovers_known_pose() -> None:
    hand = np.asarray(((0.0, 0.0, 0.0), (0.03, 0.0, 0.0),
                       (0.0, 0.02, 0.0)))
    hand_normals = np.asarray(((0.0, 0.0, 1.0),) * 3)
    rotation = Rotation.from_euler("xyz", (0.2, -0.1, 0.3)).as_matrix()
    translation = np.asarray((0.1, -0.2, 0.3))
    target = hand @ rotation.T + translation
    object_normals = -(hand_normals @ rotation.T)
    kabsch = kabsch_rigid_alignment(hand, target)
    result = initialize_three_contact_pose(
        hand, hand_normals, target, object_normals)
    pose = np.asarray(result["object_from_hand"])
    assert kabsch[:3, :3] == pytest.approx(rotation, abs=1.0e-12)
    assert pose[:3, :3] == pytest.approx(rotation, abs=1.0e-12)
    assert pose[:3, 3] == pytest.approx(translation, abs=1.0e-12)
    assert result["maximum_point_residual_m"] < 1.0e-12
    assert result["maximum_normal_residual_rad"] < 1.0e-7
    alpha, axial = contact_coordinates(
        np.asarray(((0.03, 0.0, 0.37), (-0.01, 0.025, 0.39),
                    (-0.01, -0.025, 0.38))),
        np.asarray((0.0, 0.0, 1.0)),
    )
    assert len(alpha) == 3
    assert sum(axial) == pytest.approx(0.0, abs=1.0e-15)
    assert np.ptp(axial) > 1.0e-3


def test_geometry_reject_record_preserves_spec_without_fabricating_pose() -> None:
    spec = structured_seed_specifications()[1040]
    row = _reject_record(spec, "NO_USABLE_OBJECT_REGION_IN_TARGET_WINDOW")
    assert row["candidate_id"] == spec.candidate_id
    assert row["family"] == "DENSE"
    assert row["status"] == "SEED_GEOMETRY_REJECT"
    assert row["reason"] == "NO_USABLE_OBJECT_REGION_IN_TARGET_WINDOW"
    assert "object_from_hand" not in row


def test_exact_90_degree_label_uses_rounded_legal_urdf_endpoint() -> None:
    hand = SimpleNamespace(
        independent_joint_names=("f1j1", "f1j2"),
        joint_limit_vectors=lambda: (np.asarray((0.0, 0.0)),
                                     np.asarray((1.57, 1.0))),
    )
    inputs = SimpleNamespace(hand_model=hand)
    effective = resolve_palm_configuration_rad(inputs, np.pi / 2.0)
    assert effective == pytest.approx(1.57)
    assert effective <= hand.joint_limit_vectors()[1][0]
    assert len([row for row in structured_seed_specifications()
                if row.family == "GLOBAL" and row.qp_deg == 90.0]) == 80


def test_joint_triplet_selection_recovers_edge_compatible_correspondence() -> None:
    hand = np.asarray(((0.0, 0.0, 0.0), (0.03, 0.0, 0.0),
                       (0.0, 0.02, 0.0)))
    normals = np.asarray(((0.0, 0.0, 1.0),) * 3)
    exact = hand + np.asarray((0.1, -0.2, 0.3))

    def row(face, point):
        return {"face": face, "point": np.asarray(point), "normal": -normals[0],
                "role": PRIMARY_GRIP,
                "region_area_m2": 1.0e-5, "target_axial_m": 0.0,
                "angle_error_rad": 0.01, "axial_error_m": 0.0001}

    targets = (
        (row(10, exact[0] + (0.004, 0.0, 0.0)), row(11, exact[0])),
        (row(20, exact[1] + (0.0, 0.004, 0.0)), row(21, exact[1])),
        (row(30, exact[2] + (0.004, 0.004, 0.0)), row(31, exact[2])),
    )
    chosen, alignment, audit = _compatible_target_triplet(
        hand, normals, targets, 0.00075)
    assert [item["face"] for item in chosen] == [11, 21, 31]
    assert alignment["maximum_point_residual_m"] < 1.0e-12
    assert audit["target_candidate_counts"] == [2, 2, 2]


def test_joint_triplet_selection_fails_closed_without_edge_compatible_faces() -> None:
    hand = np.asarray(((0.0, 0.0, 0.0), (0.03, 0.0, 0.0),
                       (0.0, 0.02, 0.0)))
    normals = np.asarray(((0.0, 0.0, 1.0),) * 3)

    def row(face, point):
        return {"face": face, "point": np.asarray(point), "normal": -normals[0],
                "role": PRIMARY_GRIP,
                "region_area_m2": 1.0e-5, "target_axial_m": 0.0,
                "angle_error_rad": 0.0, "axial_error_m": 0.0}

    targets = tuple((row(index, point),) for index, point in enumerate(
        ((0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.1, 0.0))))
    with pytest.raises(ValueError, match="NO_EDGE_COMPATIBLE_TARGET_TRIPLET"):
        _compatible_target_triplet(
            hand, normals, targets, 0.00075)


def test_radial_support_uses_outermost_face_and_deterministic_id_tie_break() -> None:
    mask = _radial_support_target_mask(
        np.ones(5, dtype=np.bool_), np.ones(5),
        np.asarray((1.0, 2.0, 2.0, 1.5, 3.0)),
        np.asarray((0.1, 0.1, 0.1, 3.2, 3.2)),
        np.asarray((0.2, 0.2, 0.2, 0.8, 0.8)),
        (0.0, 1.0), 2, 2,
    )
    assert np.flatnonzero(mask).tolist() == [1, 4]
