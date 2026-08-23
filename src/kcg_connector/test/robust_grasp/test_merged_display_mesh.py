from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


REPOSITORY = Path(__file__).resolve().parents[4]
SOURCE = (
    REPOSITORY
    / "src/kcg_connector/isaac/robust_grasp/export_merged_display_mesh.py"
)


def _module():
    specification = importlib.util.spec_from_file_location(
        "carts_merged_display_export", SOURCE
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tetrahedron(offset: float) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(
        (
            (offset + 0.0, 0.0, 0.0),
            (offset + 1.0, 0.0, 0.0),
            (offset + 0.0, 1.0, 0.0),
            (offset + 0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    faces = np.asarray(
        ((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)), dtype=np.int64
    )
    return vertices, faces


def _fixture(tmp_path: Path) -> dict[str, Path]:
    first_vertices, first_faces = _tetrahedron(0.0)
    second_vertices, second_faces = _tetrahedron(3.0)
    third_vertices, third_faces = _tetrahedron(6.0)
    vertices = np.vstack((first_vertices, second_vertices, third_vertices))
    faces = np.vstack((first_faces, second_faces + 4, third_faces + 8))
    mesh = tmp_path / "mesh.npz"
    np.savez_compressed(
        mesh,
        vertices_m=vertices,
        faces=faces,
        source_prim_paths=np.asarray(
            (
                "/Composed/LoosePlug/A",
                "/Composed/LoosePlug/B",
                "/Composed/LoosePlug/C",
            ),
            dtype=np.str_,
        ),
    )
    source_usda = tmp_path / "source.usda"
    source_usda.write_text(
        """#usda 1.0
def Xform "Source"
{
    def Xform "LoosePlug"
    {
        def Mesh "A"
        {
            color3f[] primvars:displayColor = [(0.1, 0.2, 0.3)]
        }
        def Mesh "B"
        {
            color3f[] primvars:displayColor = [(0.1, 0.2, 0.3)]
        }
        def Sphere "C"
        {
            color3f[] primvars:displayColor = [(0.7, 0.8, 0.9)]
        }
    }
}
""",
        encoding="utf-8",
    )
    source_stage = tmp_path / "stage.usda"
    source_stage.write_text("#usda 1.0\n", encoding="utf-8")
    source_manifest = tmp_path / "source_manifest.json"
    source_manifest.write_text(
        json.dumps(
            {
                "output_sha256": _sha256(mesh),
                "source_stage_sha256": _sha256(source_stage),
                "source_mesh_prim_count": 3,
                "source_gprim_type_counts": {"Mesh": 2, "Sphere": 1},
                "analytic_primitive_tessellation": {
                    "maximum_relative_sagitta_error": 0.001,
                    "circle_edge_count": 71,
                },
            }
        ),
        encoding="utf-8",
    )
    return {
        "mesh": mesh,
        "source_usda": source_usda,
        "source_stage": source_stage,
        "source_manifest": source_manifest,
    }


def test_merges_same_color_components_without_physics_or_face_loss(
    tmp_path: Path,
) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    output = tmp_path / "merged.usda"
    manifest = tmp_path / "merged_manifest.json"
    before = {name: _sha256(path) for name, path in fixture.items()}

    report = module.export_merged_display_mesh(
        mesh_npz=fixture["mesh"],
        source_usda=fixture["source_usda"],
        source_stage=fixture["source_stage"],
        source_manifest=fixture["source_manifest"],
        composed_prefix="/Composed",
        source_prefix="/Source",
        output_path=output,
        manifest_path=manifest,
    )

    assert report["source_prim_count"] == 3
    assert report["source_triangle_count"] == 12
    assert report["merged_mesh_count"] == 2
    assert report["merged_face_counts"] == [8, 4]
    assert report["all_source_faces_assigned_exactly_once"] is True
    assert report["unquantized_triangle_coordinates_preserved_exactly"] is True
    assert report["physics_authored"] is False
    assert report["collision_eligible"] is False
    assert report["formal_geometry_candidate"] is False
    assert report["render_performance_measured"] is False
    assert report["isaac_load_validated"] is False
    assert {name: _sha256(path) for name, path in fixture.items()} == before

    text = output.read_text(encoding="utf-8")
    assert text.count('def Mesh "Color_') == 2
    assert "physics:" not in text
    assert "PhysicsCollisionAPI" not in text
    assert "PhysicsRigidBodyAPI" not in text
    assert json.loads(manifest.read_text(encoding="utf-8")) == report


def test_rejects_ambiguous_component_to_source_path_mapping(tmp_path: Path) -> None:
    module = _module()
    fixture = _fixture(tmp_path)
    with np.load(fixture["mesh"], allow_pickle=False) as archive:
        vertices = archive["vertices_m"]
        faces = archive["faces"]
    broken_mesh = tmp_path / "broken_mesh.npz"
    np.savez_compressed(
        broken_mesh,
        vertices_m=vertices,
        faces=faces,
        source_prim_paths=np.asarray(
            ("/Composed/LoosePlug/A", "/Composed/LoosePlug/B"), dtype=np.str_
        ),
    )
    source_document = json.loads(
        fixture["source_manifest"].read_text(encoding="utf-8")
    )
    source_document["output_sha256"] = _sha256(broken_mesh)
    source_document["source_mesh_prim_count"] = 2
    fixture["source_manifest"].write_text(
        json.dumps(source_document), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="component count does not match"):
        module.export_merged_display_mesh(
            mesh_npz=broken_mesh,
            source_usda=fixture["source_usda"],
            source_stage=fixture["source_stage"],
            source_manifest=fixture["source_manifest"],
            composed_prefix="/Composed",
            source_prefix="/Source",
            output_path=tmp_path / "never.usda",
            manifest_path=tmp_path / "never.json",
        )


def test_rejects_source_gprim_without_constant_display_color(tmp_path: Path) -> None:
    module = _module()
    path = tmp_path / "missing_color.usda"
    path.write_text(
        '#usda 1.0\ndef Xform "World"\n{\n    def Mesh "NoColor"\n    {\n    }\n}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must have one constant displayColor"):
        module.parse_usda_gprim_display_colors(path)
