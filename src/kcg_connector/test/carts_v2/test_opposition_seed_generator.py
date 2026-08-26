"""Regressions for deterministic real-FK opposition anchors."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from kcg_connector.grasp.carts_v2.full_palm_search import (
    fixed_pregrasp_phase_combinations,
)
from kcg_connector.grasp.carts_v2.models import load_v2_inputs
from kcg_connector.grasp.carts_v2.opposition_seed_generator import (
    extract_object_grasp_band,
    generate_opposition_anchors,
    task_surface_triangle_geometry,
)


ROOT = Path(__file__).resolve().parents[4]
CONFIG = ROOT / "src/kcg_connector/config/carts_nailfree_height_projected.yaml"
OBJECT_B = "te_deutsch_d38999_26fj35pn_step"
EXACT_SIXTY_RAD = math.radians(60.0)


@pytest.fixture(scope="module")
def inputs():
    return load_v2_inputs(ROOT, config_path=CONFIG, object_id=OBJECT_B)


def test_object_band_and_real_fk_triangle_are_finite(inputs) -> None:
    band = extract_object_grasp_band(inputs)
    geometry = task_surface_triangle_geometry(
        inputs, EXACT_SIXTY_RAD, (0.1, 0.1, 0.1)
    )
    axis = np.asarray(band["axis_object"])
    frame = np.asarray(geometry["hand_frame_from_opposition"])
    centers = np.asarray(geometry["task_surface_centers_handbase_m"])
    assert np.linalg.norm(axis) == pytest.approx(1.0, abs=1.0e-12)
    assert band["axial_range_m"][1] > band["axial_range_m"][0]
    assert band["allowed_face_count"] > 0
    assert centers.shape == (3, 3)
    assert np.linalg.det(frame) == pytest.approx(1.0, abs=1.0e-12)
    assert frame.T @ frame == pytest.approx(np.eye(3), abs=1.0e-12)
    assert min(geometry["side_lengths_m"]) > 0.0
    assert 0.0 < geometry["triangle_quality"] <= 1.0


def test_exact_sixty_design_enumerates_all_27_then_retains_12(inputs) -> None:
    seeds, audit = generate_opposition_anchors(inputs, (EXACT_SIXTY_RAD,))
    expected_phases = fixed_pregrasp_phase_combinations()
    angle = audit["per_angle"][0]
    assert audit["raw_anchor_count"] == 27 * 12 * 3 == 972
    assert audit["preshape_count"] == len(expected_phases) == 27
    assert tuple(tuple(row) for row in
                 angle["evaluated_pregrasp_closure_phases"]) == expected_phases
    assert audit["retained_anchor_count"] == len(seeds) == 12
    assert [row["azimuth_index"] for row in audit["selected"]] == list(range(12))
    assert angle["retained_axial_counts"] == {"0": 4, "1": 4, "2": 4}
    assert {row["axial_index"] for row in audit["selected"]} == {0, 1, 2}
    assert len({seed.candidate_id for seed in seeds}) == 12
    assert all(seed.candidate_id.count("_p") == 1 for seed in seeds)
    assert all(seed.palm_configuration_rad == EXACT_SIXTY_RAD for seed in seeds)
    assert all(seed.approach_direction_object == pytest.approx(
        audit["object_grasp_band"]["axis_object"], abs=1.0e-12
    ) for seed in seeds)
    assert max(row["work_center_alignment_error_m"]
               for row in audit["selected"]) < 1.0e-12
    palm_index = inputs.hand_model.independent_joint_names.index("f1j1")
    assert all(seed.pregrasp_joint_positions_rad[palm_index]
               == pytest.approx(EXACT_SIXTY_RAD, abs=1.0e-12) for seed in seeds)


def test_anchor_identity_and_pose_are_deterministic(inputs) -> None:
    first, _first_audit = generate_opposition_anchors(inputs, (EXACT_SIXTY_RAD,))
    second, _second_audit = generate_opposition_anchors(inputs, (EXACT_SIXTY_RAD,))
    assert [seed.candidate_id for seed in first] == [
        seed.candidate_id for seed in second
    ]
    for left, right in zip(first, second):
        left_pose = left.object_from_hand_matrix()
        right_pose = right.object_from_hand_matrix()
        assert left_pose == pytest.approx(right_pose, abs=0.0)
        assert left_pose[3] == pytest.approx((0.0, 0.0, 0.0, 1.0), abs=0.0)
        assert left_pose[:3, :3].T @ left_pose[:3, :3] == pytest.approx(
            np.eye(3), abs=1.0e-12
        )
        assert np.linalg.det(left_pose[:3, :3]) == pytest.approx(1.0, abs=1.0e-12)


def test_preregistered_thirty_degree_mid_anchor_identity(inputs) -> None:
    seeds, audit = generate_opposition_anchors(inputs, (EXACT_SIXTY_RAD,))
    seed = seeds[1]
    selected = audit["selected"][1]
    assert seed.candidate_id == "opposition_p1047198_a01_z1"
    assert seed.pregrasp_closure_phases == (0.2, 0.2, 0.2)
    assert selected["azimuth_index"] == 1
    assert selected["axial_index"] == 1
