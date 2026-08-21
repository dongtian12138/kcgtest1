from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from kcg_connector.grasp.robust.collision_contract import (
    CollisionContractError,
    CoverageMode,
    DisabledCollisionAssertion,
    PairPolicy,
    SelfCollisionPairInventory,
    TerminalTrianglePartition,
    VerifiedCollisionMesh,
    build_self_collision_pair_inventory,
    build_synthetic_terminal_triangle_partition,
    canonical_oriented_triangle_bytes,
    canonical_unoriented_triangle_bytes,
    load_exact_terminal_triangle_partition,
    parse_srdf_disabled_collision_assertions,
    triangle_instance_keys,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
PAD_ASSET_ROOT = REPOSITORY_ROOT / (
    "artifacts/agent_control/tasks/"
    "D38999-AUTONOMOUS-DYNAMIC-CLOSEOUT-V2/"
    "DYN-B-V5-PAD-FORCE-CLOSURE-MULTI-GRASP/"
    "BLUE_PAD_SDF_SOURCE_ASSET_V2"
)
TERMINAL_CASES = (
    (
        "f1Link3",
        "src/iiwa_description/meshes/hand/f1Link3.STL",
        "7a33a6ab46729a2237dd13d99be3bcefb92bb3d4b77bbf9e69d884509cffcdb0",
        "f1Link3_PAD_BODY_exact_source_local_m.npz",
        "3357a6397e9061d06cb38f7b2ca325472fe3f79bff4e2bca190f0f3fbfc65fcf",
    ),
    (
        "f2Link2",
        "src/iiwa_description/meshes/hand/f2Link2.STL",
        "1758619f7ef1369fc3342c7032edee07222f9bdccc187c33830f9fa59bd508b3",
        "f2Link2_PAD_BODY_exact_source_local_m.npz",
        "34da43778c4a2bbf0b968571c5c940db9559efc7637e13badf5eaa2d13d89a6a",
    ),
    (
        "f3Link3",
        "src/iiwa_description/meshes/hand/f3Link3.STL",
        "93645443cff113b8c6e5a0280e3270192831d04246233cc45d9745c6e3c7d16e",
        "f3Link3_PAD_BODY_exact_source_local_m.npz",
        "3357a6397e9061d06cb38f7b2ca325472fe3f79bff4e2bca190f0f3fbfc65fcf",
    ),
)
COLLISION_LINKS = (
    "iiwa_link_0",
    "iiwa_link_1",
    "iiwa_link_2",
    "iiwa_link_3",
    "iiwa_link_4",
    "iiwa_link_5",
    "iiwa_link_6",
    "iiwa_link_7",
    "handbase_link",
    "f1Link1",
    "f1Link2",
    "f1Link3",
    "f2Link1",
    "f2Link2",
    "f3Link1",
    "f3Link2",
    "f3Link3",
)
ORPHAN_PAD_FACE_INDICES = (
    647,
    664,
    987,
    1113,
    1253,
    1648,
    1649,
    1667,
    1668,
    2210,
    2302,
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_mesh(tmp_path: Path) -> VerifiedCollisionMesh:
    path = tmp_path / "source.bin"
    path.write_bytes(b"immutable synthetic mesh identity")
    return VerifiedCollisionMesh(
        asset_id="synthetic_source",
        link_name="synthetic_link",
        path=path,
        sha256=_digest(path),
        unit="m",
        local_transform=np.eye(4),
        coverage_mode=CoverageMode.SYNTHETIC_ARRAY_FIXTURE,
    )


def _pad_file(tmp_path: Path, name: str = "pad.bin") -> tuple[Path, str]:
    path = tmp_path / name
    path.write_bytes(b"immutable synthetic PAD identity")
    return path, _digest(path)


def _triangle() -> np.ndarray:
    return np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        dtype=np.float64,
    )


def test_exact_triangle_identity_preserves_bits_and_winding() -> None:
    triangle = _triangle()
    cyclic = triangle[[1, 2, 0]]
    reversed_winding = triangle[[0, 2, 1]]
    signed_zero = triangle.copy()
    signed_zero[0, 0] = -0.0

    assert canonical_unoriented_triangle_bytes(triangle) == (
        canonical_unoriented_triangle_bytes(cyclic)
    )
    assert canonical_oriented_triangle_bytes(triangle) == (
        canonical_oriented_triangle_bytes(cyclic)
    )
    assert canonical_unoriented_triangle_bytes(triangle) == (
        canonical_unoriented_triangle_bytes(reversed_winding)
    )
    assert canonical_oriented_triangle_bytes(triangle) != (
        canonical_oriented_triangle_bytes(reversed_winding)
    )
    assert canonical_unoriented_triangle_bytes(triangle) != (
        canonical_unoriented_triangle_bytes(signed_zero)
    )


def test_triangle_occurrences_are_not_collapsed_to_a_set() -> None:
    triangle = _triangle()
    keys = triangle_instance_keys(
        np.asarray((triangle, triangle)), source_mesh_sha256="0" * 64
    )

    assert len(keys) == 2
    assert keys[0].canonical_geometry_sha256 == (
        keys[1].canonical_geometry_sha256
    )
    assert (keys[0].occurrence_index, keys[1].occurrence_index) == (0, 1)


def test_duplicate_geometry_is_forbidden_dominant(tmp_path: Path) -> None:
    triangle = _triangle()
    pad_path, pad_sha = _pad_file(tmp_path)
    partition = build_synthetic_terminal_triangle_partition(
        source_mesh=_fixture_mesh(tmp_path),
        source_triangles_m=np.asarray((triangle, triangle)),
        pad_source_path=pad_path,
        pad_source_sha256=pad_sha,
        pad_triangles_m=np.asarray((triangle,)),
    )

    assert len(partition.pad_allowed) == 0
    assert len(partition.nonpad_forbidden) == 2
    assert partition.ambiguous_source_face_count == 2
    assert partition.orphan_pad_face_count == 0
    assert partition.same_winding_match_count == 1
    assert partition.winding_mismatch_face_count == 0
    assert partition.complete_pad_identity_verified is False
    assert partition.formal_collision_eligible is False


def test_reversed_pad_winding_blocks_complete_identity(tmp_path: Path) -> None:
    triangle = _triangle()
    pad_path, pad_sha = _pad_file(tmp_path)
    partition = build_synthetic_terminal_triangle_partition(
        source_mesh=_fixture_mesh(tmp_path),
        source_triangles_m=np.asarray((triangle,)),
        pad_source_path=pad_path,
        pad_source_sha256=pad_sha,
        pad_triangles_m=np.asarray((triangle[[0, 2, 1]],)),
    )

    assert len(partition.pad_allowed) == 1
    assert partition.same_winding_match_count == 0
    assert partition.winding_mismatch_face_count == 1
    assert partition.complete_pad_identity_verified is False
    assert partition.formal_collision_eligible is False


@pytest.mark.parametrize(
    "link_name,source_relative,source_sha,pad_name,pad_sha",
    TERMINAL_CASES,
)
def test_real_terminal_partition_quarantines_eleven_orphans(
    link_name: str,
    source_relative: str,
    source_sha: str,
    pad_name: str,
    pad_sha: str,
) -> None:
    partition = load_exact_terminal_triangle_partition(
        asset_id=f"{link_name}_authored_visual",
        link_name=link_name,
        source_stl_path=REPOSITORY_ROOT / source_relative,
        source_stl_sha256=source_sha,
        source_unit="m",
        source_coverage_mode=CoverageMode.AUTHORED_VISUAL_SURFACE,
        pad_npz_path=PAD_ASSET_ROOT / pad_name,
        pad_npz_sha256=pad_sha,
        local_transform=np.eye(4),
    )

    assert partition.source_face_count == 14192
    assert partition.pad_face_count == 2479
    assert len(partition.pad_allowed) == 2468
    assert len(partition.nonpad_forbidden) == 11724
    assert partition.orphan_pad_face_count == 11
    assert partition.ambiguous_source_face_count == 0
    assert partition.same_winding_match_count == 2468
    assert partition.winding_mismatch_face_count == 0
    assert partition.orphan_pad_face_indices == ORPHAN_PAD_FACE_INDICES
    assert partition.winding_mismatch_pad_face_indices == ()
    assert partition.exact_cover_verified is True
    assert partition.complete_pad_identity_verified is False
    assert partition.formal_collision_eligible is False
    assert not (set(partition.pad_allowed) & set(partition.nonpad_forbidden))
    assert len(partition.partition_sha256) == 64


def test_real_srdf_is_metadata_and_all_pairs_restart_forbidden() -> None:
    assertions = parse_srdf_disabled_collision_assertions(
        REPOSITORY_ROOT / "src/kcg_moveit1/config/handarm.srdf"
    )
    inventory = build_self_collision_pair_inventory(
        link_names=COLLISION_LINKS,
        srdf_assertions=assertions,
    )
    audit = dict(inventory.audit)

    assert len(assertions) == 96
    assert audit["srdf_reason_counts"] == {"Adjacent": 16, "Never": 80}
    assert len(inventory.link_names) == 17
    assert len(inventory.all_pairs) == 136
    assert len(inventory.restarted_pair_policies) == 136
    assert all(
        policy is PairPolicy.FORBIDDEN
        for _pair, policy in inventory.restarted_pair_policies
    )
    assert audit["srdf_exemptions_applied"] is False
    assert audit["restarted_forbidden_count"] == 136


def test_srdf_unknown_duplicate_and_unknown_link_fail_closed() -> None:
    with pytest.raises(CollisionContractError, match="not registered"):
        DisabledCollisionAssertion("a", "b", "Sometimes")
    with pytest.raises(CollisionContractError, match="duplicate"):
        build_self_collision_pair_inventory(
            link_names=("a", "b"),
            srdf_assertions=(
                DisabledCollisionAssertion("a", "b", "Never"),
                DisabledCollisionAssertion("b", "a", "Adjacent"),
            ),
        )
    with pytest.raises(CollisionContractError, match="outside"):
        build_self_collision_pair_inventory(
            link_names=("a", "b"),
            srdf_assertions=(
                DisabledCollisionAssertion("a", "unknown", "Never"),
            ),
        )


def test_asset_hash_and_transform_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "mesh.bin"
    path.write_bytes(b"mesh")
    with pytest.raises(CollisionContractError, match="SHA-256 mismatch"):
        VerifiedCollisionMesh(
            asset_id="mesh",
            link_name="link",
            path=path,
            sha256="0" * 64,
            unit="m",
            local_transform=np.eye(4),
            coverage_mode=CoverageMode.AUTHORED_VISUAL_SURFACE,
        )
    reflection = np.eye(4)
    reflection[0, 0] = -1.0
    with pytest.raises(CollisionContractError, match="not proper"):
        VerifiedCollisionMesh(
            asset_id="mesh",
            link_name="link",
            path=path,
            sha256=_digest(path),
            unit="m",
            local_transform=reflection,
            coverage_mode=CoverageMode.AUTHORED_VISUAL_SURFACE,
        )
    overflow = np.eye(4)
    overflow[:3, :3] = np.diag((1.0e200, 1.0e200, 1.0e200))
    with pytest.raises(CollisionContractError, match="overflowed"):
        VerifiedCollisionMesh(
            asset_id="mesh",
            link_name="link",
            path=path,
            sha256=_digest(path),
            unit="m",
            local_transform=overflow,
            coverage_mode=CoverageMode.AUTHORED_VISUAL_SURFACE,
        )

    class ForgedCoverage:
        value = "FORGED_FORMAL_SOLID"
        formal_solid_coverage = True

    with pytest.raises(CollisionContractError, match="registered enum"):
        VerifiedCollisionMesh(
            asset_id="mesh",
            link_name="link",
            path=path,
            sha256=_digest(path),
            unit="m",
            local_transform=np.eye(4),
            coverage_mode=ForgedCoverage(),  # type: ignore[arg-type]
        )


def test_file_lineage_cannot_be_claimed_by_public_array_builder(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.bin"
    source_path.write_bytes(b"source")
    source = VerifiedCollisionMesh(
        asset_id="claimed_real_source",
        link_name="link",
        path=source_path,
        sha256=_digest(source_path),
        unit="m",
        local_transform=np.eye(4),
        coverage_mode=CoverageMode.AUTHORED_VISUAL_SURFACE,
    )
    pad_path, pad_sha = _pad_file(tmp_path)

    with pytest.raises(CollisionContractError, match="synthetic"):
        build_synthetic_terminal_triangle_partition(
            source_mesh=source,
            source_triangles_m=np.asarray((_triangle(),)),
            pad_source_path=pad_path,
            pad_source_sha256=pad_sha,
            pad_triangles_m=np.asarray((_triangle(),)),
        )
    with pytest.raises(TypeError):
        TerminalTrianglePartition()


def test_duplicate_pair_policy_and_arithmetic_overflow_fail_closed() -> None:
    pair = ("a", "b")
    with pytest.raises(CollisionContractError, match="every self-collision"):
        SelfCollisionPairInventory(
            link_names=("a", "b"),
            all_pairs=(pair,),
            srdf_assertions=(),
            restarted_pair_policies=(
                (pair, PairPolicy.FIRST_CONTACT_ENDPOINT_ONLY),
                (pair, PairPolicy.FORBIDDEN),
            ),
            inventory_sha256="0" * 64,
        )
    overflow_triangle = np.asarray(
        ((-1.0e308, 0.0, 0.0), (1.0e308, 0.0, 0.0), (0.0, 1.0, 0.0))
    )
    with pytest.raises(CollisionContractError, match="overflowed"):
        canonical_unoriented_triangle_bytes(overflow_triangle)
    for invalid_links in ((), ("only_one",), ("", "a")):
        with pytest.raises(CollisionContractError, match="at least two"):
            SelfCollisionPairInventory(
                link_names=invalid_links,
                all_pairs=(),
                srdf_assertions=(),
                restarted_pair_policies=(),
                inventory_sha256="0" * 64,
            )


def test_partition_digest_binds_unit_and_transform(tmp_path: Path) -> None:
    triangle = _triangle()
    pad_path, pad_sha = _pad_file(tmp_path)
    first_source = _fixture_mesh(tmp_path)
    translated = np.eye(4)
    translated[:3, 3] = (1.0, 2.0, 3.0)
    second_source = VerifiedCollisionMesh(
        asset_id=first_source.asset_id,
        link_name=first_source.link_name,
        path=first_source.path,
        sha256=first_source.sha256,
        unit="mm",
        local_transform=translated,
        coverage_mode=CoverageMode.SYNTHETIC_ARRAY_FIXTURE,
    )
    first = build_synthetic_terminal_triangle_partition(
        source_mesh=first_source,
        source_triangles_m=np.asarray((triangle,)),
        pad_source_path=pad_path,
        pad_source_sha256=pad_sha,
        pad_triangles_m=np.asarray((triangle,)),
    )
    second = build_synthetic_terminal_triangle_partition(
        source_mesh=second_source,
        source_triangles_m=np.asarray((triangle,)),
        pad_source_path=pad_path,
        pad_source_sha256=pad_sha,
        pad_triangles_m=np.asarray((triangle,)),
    )

    assert first.partition_sha256 != second.partition_sha256
    assert first.formal_collision_eligible is False
    assert second.formal_collision_eligible is False
