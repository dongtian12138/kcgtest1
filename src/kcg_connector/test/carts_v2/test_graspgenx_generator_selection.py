from __future__ import annotations

import importlib.util
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[4]
RUNNER = ROOT / "scripts/carts_v2/run_graspgenx_candidates.py"


def _runner():
    pytest.importorskip("graspgenx")
    spec = importlib.util.spec_from_file_location("graspgenx_generator_test", RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_visibility_is_audit_only_and_top_128_is_score_stable(monkeypatch) -> None:
    runner = _runner()
    poses = np.repeat(np.eye(4)[None], 130, axis=0)
    poses[:, 0, 3] = np.arange(130) * 1.0e-4
    scores = np.concatenate((np.full(2, 0.9), np.linspace(0.8, 0.1, 128)))
    tags = ["diff"] * 130
    monkeypatch.setattr(
        runner,
        "run_planner_on_object",
        lambda *_args, **_kwargs: (poses, scores, tags, None),
    )
    open_visible = np.zeros(130, dtype=bool)
    open_visible[-2:] = True
    monkeypatch.setattr(
        runner,
        "_open_or_half_visibility",
        lambda *_args: (open_visible, np.zeros(130, dtype=bool), open_visible),
    )

    rows, audit = runner._infer(
        np.zeros((16, 3)), np.zeros(3), SimpleNamespace(gripper=object()),
        object_from_inference=np.eye(4), num_grasps=256, keep=128,
    )

    assert [row["raw_index"] for row in rows] == list(range(128))
    assert audit["open_sweep_visible_count"] == 2
    assert audit["open_or_half_sweep_not_visible_count"] == 128
    assert audit["proposal_visibility_selection_role"] == "AUDIT_ONLY_NOT_SELECTION_GATE"


def test_object_conditioning_samples_all_registered_faces(tmp_path, monkeypatch) -> None:
    runner = _runner()
    mesh_path = tmp_path / "object.npz"
    vertices = np.asarray(
        ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)), dtype=np.float64
    )
    faces = np.asarray(((0, 1, 2), (0, 1, 3)), dtype=np.int64)
    allowed = np.asarray((0,), dtype=np.int64)
    np.savez(mesh_path, vertices_m=vertices, faces=faces, allowed_face_indices=allowed)
    observed = {}

    def sample_surface(mesh, count, seed):
        observed["face_count"] = len(mesh.faces)
        return np.repeat([[0.25, 0.25, 0.0]], count, axis=0), np.zeros(count, dtype=int)

    monkeypatch.setattr(runner.trimesh.sample, "sample_surface", sample_surface)
    row = {
        "standardized_mesh_npz": str(mesh_path),
        "standardized_mesh_sha256": hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
        "face_count": 2,
        "allowed_face_count": 1,
        "allowed_face_domain_sha256": runner._allowed_face_domain_sha256(2, allowed),
        "sample_point_count": 4,
        "inference_frame": "FROZEN_SCENE_WORLD",
        "inference_from_object_row_major": np.eye(4).ravel().tolist(),
    }

    points, *_rest = runner._load_object(row, 20260824)

    assert observed["face_count"] == 2
    assert points.shape == (4, 3)
