"""Exact raw-STL PAD lineage tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from kcg_connector.grasp.robust.collision_contract import (
    CoverageMode,
    load_exact_terminal_triangle_partition,
)
from kcg_connector.grasp.robust.object_model import file_sha256, load_stl_mesh
from kcg_connector.grasp.robust.terminal_pad_source import (
    TerminalPadSourceError,
    extract_exact_terminal_pad_source,
    manifold_edge_face_components,
)


REPOSITORY = Path(__file__).resolve().parents[4]
ASSET_ROOT = (
    REPOSITORY
    / "artifacts/agent_control/tasks/CARTS-GRASP-CROSS-OBJECT-V1/"
    "TERMINAL_PAD_EXACT_SOURCE_V2"
)
CASES = (
    (
        "f1Link3",
        "7a33a6ab46729a2237dd13d99be3bcefb92bb3d4b77bbf9e69d884509cffcdb0",
    ),
    (
        "f2Link2",
        "1758619f7ef1369fc3342c7032edee07222f9bdccc187c33830f9fa59bd508b3",
    ),
    (
        "f3Link3",
        "93645443cff113b8c6e5a0280e3270192831d04246233cc45d9745c6e3c7d16e",
    ),
)


@pytest.mark.parametrize("link_name,source_sha256", CASES)
def test_real_terminal_pad_is_selected_from_exact_raw_source_faces(
    link_name: str,
    source_sha256: str,
) -> None:
    source = REPOSITORY / f"src/iiwa_description/meshes/hand/{link_name}.STL"
    pad = extract_exact_terminal_pad_source(
        link_name=link_name,
        source_stl_path=source,
        source_stl_sha256=source_sha256,
        expected_source_face_count=14192,
        expected_pad_face_count=2479,
        expected_pad_vertex_count=1250,
        expected_component_area_rank=1,
    )
    assert pad.source_component_count == 158
    assert pad.source_nonmanifold_edge_count == 163
    assert pad.component_area_rank == 1
    assert pad.face_count == 2479
    assert pad.vertex_count == 1250
    assert pad.boundary_edge_count == 19
    assert pad.nonmanifold_edge_count == 0
    assert pad.winding_conflict_count == 0
    assert pad.audit["coordinate_tolerance_used"] is False
    assert pad.audit["source_vertex_changed"] is False
    assert not pad.source_face_indices.flags.writeable
    assert not pad.points_local_m.flags.writeable
    assert not pad.faces.flags.writeable
    source_mesh, _ = load_stl_mesh(source, unit="m", orient_outward=False)
    np.testing.assert_array_equal(
        pad.points_local_m[pad.faces],
        source_mesh.face_vertices_m[pad.source_face_indices],
    )


def test_nonmanifold_edge_does_not_merge_distinct_surface_sheets() -> None:
    faces = np.asarray(
        (
            (0, 1, 2),
            (1, 0, 3),
            (0, 1, 4),
            (5, 6, 7),
            (6, 5, 8),
        ),
        dtype=np.int64,
    )
    components = manifold_edge_face_components(faces)
    assert tuple(tuple(row) for row in components) == (
        (0,),
        (1,),
        (2,),
        (3, 4),
    )


def test_source_hash_change_fails_closed(tmp_path: Path) -> None:
    source = REPOSITORY / "src/iiwa_description/meshes/hand/f1Link3.STL"
    copied = tmp_path / source.name
    copied.write_bytes(source.read_bytes() + b"changed")
    with pytest.raises(TerminalPadSourceError, match="SHA-256"):
        extract_exact_terminal_pad_source(
            link_name="f1Link3",
            source_stl_path=copied,
            source_stl_sha256=CASES[0][1],
            expected_source_face_count=14192,
            expected_pad_face_count=2479,
            expected_pad_vertex_count=1250,
            expected_component_area_rank=1,
        )


@pytest.mark.parametrize("link_name,source_sha256", CASES)
def test_generated_pad_asset_has_zero_orphan_source_faces(
    link_name: str,
    source_sha256: str,
) -> None:
    source = REPOSITORY / f"src/iiwa_description/meshes/hand/{link_name}.STL"
    pad_path = ASSET_ROOT / f"{link_name}_PAD_BODY_raw_source_local_m.npz"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_sha256
    partition = load_exact_terminal_triangle_partition(
        asset_id=f"{link_name}_authored_visual",
        link_name=link_name,
        source_stl_path=source,
        source_stl_sha256=source_sha256,
        source_unit="m",
        source_coverage_mode=CoverageMode.AUTHORED_VISUAL_SURFACE,
        pad_npz_path=pad_path,
        pad_npz_sha256=file_sha256(pad_path),
        local_transform=np.eye(4),
    )
    assert partition.source_face_count == 14192
    assert partition.pad_face_count == 2479
    assert len(partition.pad_allowed) == 2479
    assert len(partition.nonpad_forbidden) == 11713
    assert partition.orphan_pad_face_count == 0
    assert partition.ambiguous_source_face_count == 0
    assert partition.winding_mismatch_face_count == 0
    assert partition.complete_pad_identity_verified is True
    assert partition.formal_collision_eligible is False
