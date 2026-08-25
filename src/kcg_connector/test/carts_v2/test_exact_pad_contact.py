"""Regressions for full PAD mesh contact and fail-closed motion compatibility."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from kcg_connector.grasp.carts_v2.candidate_generator import generate_raw_candidates
from kcg_connector.grasp.carts_v2.closure_predictor import (
    SequentialClosurePredictor,
)
from kcg_connector.grasp.carts_v2.models import load_v2_inputs
from kcg_connector.grasp.carts_v2.surface_contact import (
    ExactPadSurfaceQuery,
    NearestSurface,
    nearest_motion_compatible_index,
)
from kcg_connector.grasp.robust.object_model import TriangleMesh


ROOT = Path(__file__).resolve().parents[4]
CONFIG = ROOT / "src/kcg_connector/config/carts_grasp_v2.yaml"
OBJECT_A = "current_d38999_26kj61sn_public_spec"


def test_exact_pad_mesh_query_returns_full_triangle_witness_and_normal() -> None:
    pytest.importorskip("fcl")
    vertices = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), dtype=float)
    faces = np.asarray(((0, 1, 2),), dtype=np.int64)
    mesh = TriangleMesh(vertices, faces, ("allowed",))
    orientation_sha = "a" * 64
    inputs = SimpleNamespace(
        object_contract=SimpleNamespace(
            model=SimpleNamespace(mesh=mesh),
            orientation_certificate=SimpleNamespace(
                positive_volume_winding_sign_by_source_face=(1,),
                canonical_sha256=orientation_sha,
            ),
            material_boundary_evidence=SimpleNamespace(
                formal_material_boundary_eligible=True,
                certificate=SimpleNamespace(
                    orientation_certificate_sha256=orientation_sha
                ),
            ),
        ),
        hand_contract=SimpleNamespace(
            pads=(SimpleNamespace(name="pad", points_local_m=vertices, faces=faces),)
        ),
    )
    index = ExactPadSurfaceQuery(inputs)
    transform = np.eye(4)
    transform[2, 3] = 0.1
    nearest, pad_point, normal = index.query_pad("pad", transform)
    assert nearest.face_index.tolist() == [0]
    assert nearest.distance_m[0] == pytest.approx(0.1)
    assert pad_point[2] == pytest.approx(0.1)
    assert np.allclose(normal, (0.0, 0.0, 1.0))
    touching, _pad_point, _normal = index.query_pad("pad", np.eye(4))
    assert touching.intersecting is True


def test_outward_normal_sign_is_bound_to_material_certificate() -> None:
    pytest.importorskip("fcl")
    vertices = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), dtype=float)
    faces = np.asarray(((0, 2, 1),), dtype=np.int64)
    mesh = TriangleMesh(vertices, faces, ("allowed",))
    orientation_sha = "b" * 64
    inputs = SimpleNamespace(
        object_contract=SimpleNamespace(
            model=SimpleNamespace(mesh=mesh),
            orientation_certificate=SimpleNamespace(
                positive_volume_winding_sign_by_source_face=(-1,),
                canonical_sha256=orientation_sha,
            ),
            material_boundary_evidence=SimpleNamespace(
                formal_material_boundary_eligible=True,
                certificate=SimpleNamespace(
                    orientation_certificate_sha256=orientation_sha
                ),
            ),
        ),
        hand_contract=SimpleNamespace(
            pads=(SimpleNamespace(name="pad", points_local_m=vertices, faces=faces),)
        ),
    )
    assert np.allclose(ExactPadSurfaceQuery(inputs).normal(0), (0.0, 0.0, 1.0))


def test_second_nearest_motion_compatible_witness_is_not_lost() -> None:
    selected = nearest_motion_compatible_index(
        np.asarray((1.0e-4, 2.0e-4)),
        np.asarray((-1.0e-3, 1.0e-3)),
        1.0e-5,
    )
    assert selected == 1
    assert nearest_motion_compatible_index(
        np.asarray((1.0e-4, 2.0e-4)),
        np.asarray((-1.0e-3, 0.0)),
        1.0e-5,
    ) == -1


def test_empty_motion_compatible_contact_fails_closed(monkeypatch) -> None:
    inputs = load_v2_inputs(ROOT, config_path=CONFIG, object_id=OBJECT_A)
    seed = generate_raw_candidates(inputs)[0]
    predictor = SequentialClosurePredictor(inputs)
    nearest = NearestSurface(
        point_m=np.asarray(((0.01, 0.0, 0.0),)),
        distance_m=np.asarray((1.0e-4,)),
        face_index=np.asarray((int(inputs.face_roles.allowed_face_indices[0]),)),
    )
    monkeypatch.setattr(predictor, "_initial_clearance", lambda *_args: 0.002)
    monkeypatch.setattr(
        predictor,
        "_contact_at_phase",
        lambda *_args: (-1, nearest, np.asarray(((1.0, 0.0, 0.0),)), np.asarray((-1.0,))),
    )
    prediction = predictor.predict(seed)
    assert prediction.status == "CLOSURE_REJECT"
    assert prediction.reason == "NO_MOTION_COMPATIBLE_PAD_POINT_finger_1_pad"
