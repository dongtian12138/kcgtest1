import hashlib
import json
from pathlib import Path

import numpy as np

from kcg_connector.grasp.carts_v2.task_grip_surface import (
    TaskGripSurface,
    load_task_grip_surfaces,
    motion_faces_current_workspace,
)
from kcg_connector.grasp.robust.object_model import load_stl_mesh


IDENTITIES = (
    ("finger_1_pad", "finger_1_task_grip_surface", "f1", "f1Link3"),
    ("finger_2_pad", "finger_2_task_grip_surface", "f2", "f2Link2"),
    ("finger_3_pad", "finger_3_task_grip_surface", "f3", "f3Link3"),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_ascii_stl(path: Path, name: str) -> None:
    path.write_text(
        f"solid {name}\n"
        "facet normal 0 0 1\nouter loop\n"
        "vertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\n"
        f"endloop\nendfacet\nendsolid {name}\n",
        encoding="ascii",
    )


def _fixture(root: Path) -> tuple[Path, str]:
    links = []
    for pad_name, surface_name, finger_name, link_name in IDENTITIES:
        source = root / f"{link_name}.stl"
        arrays = root / f"{link_name}.npz"
        _write_ascii_stl(source, link_name)
        mesh, _ = load_stl_mesh(source, unit="m", orient_outward=False)
        np.savez(
            arrays,
            points_local_m=np.asarray(mesh.vertices_m),
            faces=np.asarray(mesh.faces),
            source_face_indices=np.asarray((0,), dtype=np.int64),
            face_normals_local=np.asarray(mesh.face_normals),
            patch_indices=np.asarray((0,), dtype=np.int64),
            legacy_blue_pad_face_mask=np.asarray((True,), dtype=np.bool_),
        )
        links.append({
            "surface_name": surface_name,
            "pad_name": pad_name,
            "finger_name": finger_name,
            "link_name": link_name,
            "source_mesh": source.name,
            "source_mesh_sha256": _sha256(source),
            "source_face_count": 1,
            "surface_npz": arrays.name,
            "surface_npz_sha256": _sha256(arrays),
            "task_face_count": 1,
            "patch_count": 1,
            "legacy_blue_pad_face_count": 1,
        })
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": "carts_task_grip_surface_v2",
        "hand_variant": "CONNECTOR_GRASP_NO_NAIL",
        "semantic": "TASK_GRIP_SURFACE",
        "hardware_authorized": False,
        "online_control_use_allowed": False,
        "object_specific_selection_used": False,
        "links": links,
    }), encoding="utf-8")
    return manifest, _sha256(manifest)


def test_loads_three_hash_bound_task_surfaces(tmp_path: Path) -> None:
    manifest, digest = _fixture(tmp_path)
    surfaces = load_task_grip_surfaces(tmp_path, manifest, digest)
    assert tuple(surfaces) == tuple(row[0] for row in IDENTITIES)
    assert all(len(surface.faces) == 1 for surface in surfaces.values())
    assert all(surface.legacy_blue_pad_face_mask[0] for surface in surfaces.values())


def test_manifest_or_source_drift_fails_closed(tmp_path: Path) -> None:
    manifest, digest = _fixture(tmp_path)
    (tmp_path / "f1Link3.stl").write_text("changed", encoding="ascii")
    try:
        load_task_grip_surfaces(tmp_path, manifest, digest)
    except ValueError as exc:
        assert "source hash changed" in str(exc)
    else:
        raise AssertionError("source drift must fail closed")


def test_current_workspace_and_closing_motion_both_gate_task_face() -> None:
    triangle = np.asarray(((0.0, -0.1, -0.1), (0.0, 0.1, -0.1),
                           (0.0, 0.0, 0.1)))
    faces = np.asarray(((0, 1, 2),), dtype=np.int64)
    surfaces = {}
    for index, (pad, _name, finger, link) in enumerate(IDENTITIES):
        shifted = triangle + (float(index), 0.0, 0.0)
        surfaces[pad] = TaskGripSurface(
            surface_name=f"finger_{index + 1}_task_grip_surface",
            pad_name=pad, finger_name=finger, link_name=link,
            points_local_m=shifted, faces=faces,
            source_face_indices=np.asarray((index,), dtype=np.int64),
            face_normals_local=np.asarray(((1.0, 0.0, 0.0),)),
            patch_indices=np.asarray((0,), dtype=np.int64),
            legacy_blue_pad_face_mask=np.asarray((True,), dtype=np.bool_),
            source_mesh_path=Path(f"{link}.stl"), source_mesh_sha256="0" * 64,
        )
    class Hand:
        @staticmethod
        def forward_kinematics(_joints, base_transform):
            return {row[3]: base_transform for row in IDENTITIES}
    point = np.asarray((-1.0, 0.0, 0.0))
    assert motion_faces_current_workspace(
        surfaces, Hand(), (), np.eye(4), "finger_1_pad", point,
        np.asarray((1.0, 0.0, 0.0)), np.asarray((1.0, 0.0, 0.0)), 1.0e-5,
    )
    assert not motion_faces_current_workspace(
        surfaces, Hand(), (), np.eye(4), "finger_1_pad", point,
        np.asarray((-1.0, 0.0, 0.0)), np.asarray((1.0, 0.0, 0.0)), 1.0e-5,
    )
