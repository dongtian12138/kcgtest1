"""CPU tests for the fixed hand-occluder CAD model."""
from pathlib import Path

import numpy as np

from kcg_connector.hand_occluder_cad import (
    HAND_OCCLUDER_LABEL,
    build_hand_occluder_cad,
)


def test_hand_occluder_cad_builds_and_stays_in_hand_frame():
    repository = Path(__file__).resolve().parents[3]
    urdf = repository / "artifacts/kcg_connector/urdf/handarm.urdf"
    mesh_dir = repository / "src/iiwa_description/meshes/hand"
    hand_q = (1.0, 0.7587, 1.0, 0.5721, 0.5721, 1.0, 0.7601, 0.7601)
    cad = build_hand_occluder_cad(hand_q, urdf, mesh_dir)
    assert len(cad.xyz) > 5000
    assert np.all(cad.label == HAND_OCCLUDER_LABEL)
    # hand-frame sanity: fingers hang below the handbase along hand z
    assert cad.xyz[:, 2].max() < 0.5
    assert cad.xyz[:, 2].min() > -0.05
    assert np.linalg.norm(cad.xyz[:, :2], axis=1).max() < 0.25
    assert np.all(np.isfinite(cad.xyz))


def test_hand_occluder_cad_rejects_bad_angles():
    repository = Path(__file__).resolve().parents[3]
    urdf = repository / "artifacts/kcg_connector/urdf/handarm.urdf"
    mesh_dir = repository / "src/iiwa_description/meshes/hand"
    try:
        build_hand_occluder_cad((0.0, 0.0, 0.0), urdf, mesh_dir)
    except ValueError:
        return
    raise AssertionError("expected ValueError for wrong-length hand_q")

