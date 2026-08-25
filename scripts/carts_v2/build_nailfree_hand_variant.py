#!/usr/bin/env python3
"""Build a provenance-bound, nail-free distal-link simulation variant.

The removed assembly is the confirmed tool shell plus its two symmetric
mounting posts.  This script never edits the authored hand or binds Isaac.
"""

from __future__ import annotations
import argparse
import importlib.metadata
import json
import struct
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
import trimesh

from kcg_connector.grasp.robust.object_model import file_sha256, load_stl_mesh
from kcg_connector.grasp.robust.terminal_pad_source import manifold_edge_face_components
_TIP_FACE_COUNT = 1076
_TIP_VERTEX_COUNT = 534
_TIP_AREA_RANK_ZERO_BASED = 4
_POST_FACE_COUNT = 640
_POST_VERTEX_COUNTS = {329, 330}
_POST_AREA_RANKS = {6, 7}
_REMOVED_FACE_COUNT = 2356
_SOURCE_FACE_COUNT = 14192
_SOURCE_COMPONENT_COUNT = 158
_VOXEL_PITCH_M = 2.5e-4
_MANIFOLD_SIMPLIFY_TOLERANCE_M = 5.0e-5
_ESTIMATE_METHOD = (
    "SIMULATION_ESTIMATE_SUBTRACT_REMOVED_ASSEMBLY_FROM_BOUND_OLD_URDF_"
    "INERTIA_USING_MATCHED_VOXEL_UNION_MASS_RATIO_NOT_HARDWARE_CALIBRATION"
)
_SEMANTIC_DIR = Path(
    "artifacts/agent_control/tasks/D38999-AUTONOMOUS-DYNAMIC-CLOSEOUT-V2/"
    "DYN-B-V5-PAD-FORCE-CLOSURE-MULTI-GRASP/BLUE_PAD_SDF_SOURCE_ASSET_V2"
)
_BINDINGS = {
    "f1Link3": {
        "source_sha256": "7a33a6ab46729a2237dd13d99be3bcefb92bb3d4b77bbf9e69d884509cffcdb0",
        "tip_npz_sha256": "2cc23d2ae5982e322414dca9befe2f15e18d2752b2701caa20078b130241a1a6",
        "old_com_m": [-0.016803, -0.020891, 0.011999],
        "old_inertia_kg_m2": [
            [1.6904e-5, 9.7694e-6, 9.1101e-9],
            [9.7694e-6, 1.2703e-5, 1.2096e-8],
            [9.1101e-9, 1.2096e-8, 2.4528e-5],
        ],
    },
    "f2Link2": {
        "source_sha256": "1758619f7ef1369fc3342c7032edee07222f9bdccc187c33830f9fa59bd508b3",
        "tip_npz_sha256": "90a7051e40368f4b463f2b5a158a36055aa6ac955344f7432bc11cd297026af4",
        "old_com_m": [-0.012481, -0.023727, -0.012001],
        "old_inertia_kg_m2": [
            [2.0429e-5, 8.2585e-6, 6.6195e-9],
            [8.2585e-6, 9.1773e-6, 1.3619e-8],
            [6.6195e-9, 1.3619e-8, 2.4528e-5],
        ],
    },
    "f3Link3": {
        "source_sha256": "93645443cff113b8c6e5a0280e3270192831d04246233cc45d9745c6e3c7d16e",
        "tip_npz_sha256": "2cc23d2ae5982e322414dca9befe2f15e18d2752b2701caa20078b130241a1a6",
        "old_com_m": [-0.016803, -0.020891, 0.011999],
        "old_inertia_kg_m2": [
            [1.6904e-5, 9.7694e-6, 9.1101e-9],
            [9.7694e-6, 1.2703e-5, 1.2096e-8],
            [9.1101e-9, 1.2096e-8, 2.4528e-5],
        ],
    },
}
_OLD_MASS_KG = 0.057879
def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/iiwa_description/meshes/hand/connector_no_nail"),
    )
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=Path("artifacts/carts_v2/nailfree_height_projected/hand_model_audit"),
    )
    return parser.parse_args()
def _surface_area(triangles: np.ndarray) -> float:
    return float(
        0.5
        * np.linalg.norm(
            np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
            axis=1,
        ).sum()
    )
def _indexed_component(mesh, face_indices: np.ndarray) -> trimesh.Trimesh:
    source_vertices = np.unique(mesh.faces[face_indices])
    inverse = np.full(len(mesh.vertices_m), -1, dtype=np.int64)
    inverse[source_vertices] = np.arange(len(source_vertices), dtype=np.int64)
    result = trimesh.Trimesh(
        vertices=mesh.vertices_m[source_vertices],
        faces=inverse[mesh.faces[face_indices]],
        process=False,
    )
    if float(result.volume) < 0.0:
        result.invert()
    return result
def _triangle_keys(triangles: np.ndarray) -> set[bytes]:
    keys: set[bytes] = set()
    for triangle in np.asarray(triangles, dtype="<f8"):
        keys.add(b"".join(sorted(vertex.tobytes() for vertex in triangle)))
    return keys
def _identify_removable_nail(
    root: Path, link: str
) -> tuple[object, np.ndarray, trimesh.Trimesh, dict]:
    source = root / "src/iiwa_description/meshes/hand" / f"{link}.STL"
    binding = _BINDINGS[link]
    if file_sha256(source) != binding["source_sha256"]:
        raise ValueError(f"{link}: authored STL hash changed")
    mesh, provenance = load_stl_mesh(source, unit="m", orient_outward=False)
    components = manifold_edge_face_components(mesh.faces)
    if len(mesh.faces) != _SOURCE_FACE_COUNT or len(components) != _SOURCE_COMPONENT_COUNT:
        raise ValueError(f"{link}: source face/component identity changed")
    areas = np.asarray([_surface_area(mesh.face_vertices_m[row]) for row in components])
    area_order = np.argsort(-areas, kind="stable")
    shell_matches, post_matches = [], []
    for component_index, row in enumerate(components):
        vertex_count = len(np.unique(mesh.faces[row]))
        rank = int(np.flatnonzero(area_order == component_index)[0])
        if (len(row) == _TIP_FACE_COUNT and vertex_count == _TIP_VERTEX_COUNT
                and rank == _TIP_AREA_RANK_ZERO_BASED):
            shell_matches.append((component_index, np.asarray(row), rank))
        if (
            len(row) == _POST_FACE_COUNT
            and vertex_count in _POST_VERTEX_COUNTS
            and rank in _POST_AREA_RANKS
        ):
            post_matches.append((component_index, np.asarray(row), rank))
    if len(shell_matches) != 1 or len(post_matches) != 2:
        raise ValueError(f"{link}: removable nail component identity is not unique")
    shell_index, shell_faces, shell_rank = shell_matches[0]
    shell = _indexed_component(mesh, shell_faces)
    if not shell.is_watertight or not shell.is_winding_consistent:
        raise ValueError(f"{link}: bound TIP is no longer a closed consistent component")
    rows = [(shell_index, shell_faces, shell_rank), *post_matches]
    rows.sort(key=lambda item: int(item[1][0]))
    if any(not np.array_equal(row, np.arange(row[0], row[-1] + 1)) for _, row, _ in rows):
        raise ValueError(f"{link}: removable nail face streams are no longer contiguous")
    post_distances = []
    for _, row, _ in rows[1:]:
        post = _indexed_component(mesh, row)
        post_distances.append(float(np.min(
            trimesh.proximity.closest_point(shell, post.vertices)[1]
        )))
    if max(post_distances) > 2.0e-5:
        raise ValueError(f"{link}: mounting posts detached from tool shell")
    face_indices = np.concatenate([row for _, row, _ in rows])
    if len(face_indices) != _REMOVED_FACE_COUNT:
        raise ValueError(f"{link}: removable nail face count changed")
    removed = _indexed_component(mesh, face_indices)
    remaining = np.ones(len(mesh.faces), dtype=bool)
    remaining[face_indices] = False
    if np.intersect1d(np.unique(mesh.faces[face_indices]), np.unique(mesh.faces[remaining])).size:
        raise ValueError(f"{link}: TIP now shares authored vertices with retained geometry")
    tip_npz = root / _SEMANTIC_DIR / f"{link}_TIP_exact_source_local_m.npz"
    if file_sha256(tip_npz) != binding["tip_npz_sha256"]:
        raise ValueError(f"{link}: prior user-confirmed TIP artifact hash changed")
    with np.load(tip_npz, allow_pickle=False) as archive:
        prior_triangles = archive["points_local_m"][archive["faces"]]
    if _triangle_keys(prior_triangles) != _triangle_keys(mesh.face_vertices_m[shell_faces]):
        raise ValueError(f"{link}: tool shell differs from user-confirmed TIP geometry")
    component_rows = []
    for position, (index, row, rank) in enumerate(rows):
        part = _indexed_component(mesh, row)
        component_rows.append({
            "role": "TOOL_SHELL" if position == 0 else "MOUNTING_POST",
            "source_component_tuple_index": int(index),
            "area_rank_zero_based": int(rank),
            "source_face_count": int(len(row)),
            "source_vertex_count": int(len(part.vertices)),
            "source_face_index_range_inclusive": [int(row[0]), int(row[-1])],
            "bounds_m": part.bounds.tolist(),
            "centroid_m": part.centroid.tolist(),
        })
    evidence = {
        "source_path": str(source.relative_to(root)),
        "source_sha256": provenance.source_sha256,
        "source_face_count": len(mesh.faces),
        "source_component_count": len(components),
        "source_is_watertight": bool(trimesh.Trimesh(
            vertices=mesh.vertices_m, faces=mesh.faces, process=False).is_watertight),
        "removed_component_count": len(rows),
        "removed_components": component_rows,
        "removed_source_face_count": int(len(face_indices)),
        "removed_source_vertex_count": int(len(removed.vertices)),
        "removed_area_m2": float(removed.area),
        "removed_effective_volume_m3": float(abs(removed.volume)),
        "mounting_post_to_shell_minimum_distances_m": post_distances,
        "mounting_identity_tolerance_m": 2.0e-5,
        "prior_tip_semantic_covers_tool_shell_only": True,
        "tip_prior_semantic_npz": str(tip_npz.relative_to(root)),
        "tip_prior_semantic_npz_sha256": binding["tip_npz_sha256"],
        "retained_face_count": int(np.count_nonzero(remaining)),
    }
    return mesh, face_indices, removed, evidence
def _write_binary_stl(path: Path, triangles_m: np.ndarray) -> None:
    triangles = np.asarray(triangles_m, dtype=np.float64)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths == 0.0) or not np.all(np.isfinite(triangles)):
        raise ValueError(f"cannot write degenerate or non-finite STL: {path}")
    normals /= lengths[:, None]
    header = b"CARTS NAILFREE EXACT SOURCE FACES V1".ljust(80, b"\0")
    with path.open("wb") as stream:
        stream.write(header)
        stream.write(struct.pack("<I", len(triangles)))
        for normal, triangle in zip(normals, triangles):
            stream.write(struct.pack("<12fH", *(normal.tolist() + triangle.reshape(-1).tolist()), 0))
def _closed_union(triangles: np.ndarray) -> tuple[trimesh.Trimesh, trimesh.Trimesh]:
    import manifold3d

    raw = trimesh.Trimesh(
        vertices=np.asarray(triangles).reshape(-1, 3),
        faces=np.arange(np.asarray(triangles).size // 3).reshape(-1, 3),
        process=True,
    )
    voxels = raw.voxelized(_VOXEL_PITCH_M, method="subdivide").fill()
    marching = voxels.marching_cubes
    marching.apply_transform(voxels.transform)
    source = manifold3d.Mesh(
        vert_properties=np.asarray(marching.vertices, dtype=np.float32),
        tri_verts=np.asarray(marching.faces, dtype=np.uint32),
    )
    value = manifold3d.Manifold(source).simplify(_MANIFOLD_SIMPLIFY_TOLERANCE_M)
    output = value.to_mesh()
    closed = trimesh.Trimesh(
        vertices=np.asarray(output.vert_properties)[:, :3],
        faces=np.asarray(output.tri_verts), process=True,
    )
    if not closed.is_watertight or not closed.is_winding_consistent:
        raise ValueError("voxel-union physical mesh is not closed and consistent")
    if closed.volume < 0.0:
        closed.invert()
    return raw, closed


def _surface_deviation(raw: trimesh.Trimesh, closed: trimesh.Trimesh) -> dict:
    sample = lambda mesh: np.vstack((mesh.vertices, mesh.triangles_center))
    _, closed_to_raw, _ = trimesh.proximity.closest_point(raw, sample(closed))
    _, raw_to_closed, _ = trimesh.proximity.closest_point(closed, sample(raw))
    stats = lambda value: {
        "maximum_m": float(np.max(value)),
        "p95_m": float(np.percentile(value, 95.0)),
        "rms_m": float(np.sqrt(np.mean(np.square(value)))),
    }
    return {"closed_to_raw": stats(closed_to_raw), "raw_to_closed": stats(raw_to_closed)}


def _parallel_axis(center: np.ndarray) -> np.ndarray:
    return float(np.dot(center, center)) * np.eye(3) - np.outer(center, center)


def _mass_estimate(
    old_closed: trimesh.Trimesh,
    new_closed: trimesh.Trimesh,
    removed_closed: trimesh.Trimesh,
    link: str,
) -> dict:
    old_volume, new_volume = float(old_closed.volume), float(new_closed.volume)
    if not 0.0 < new_volume < old_volume:
        raise ValueError(f"{link}: matched closed-union volumes do not show nail removal")
    removed_mass = _OLD_MASS_KG * (old_volume - new_volume) / old_volume
    new_mass = _OLD_MASS_KG - removed_mass
    removed_com = np.asarray(removed_closed.center_mass, dtype=np.float64)
    removed_inertia = (
        np.asarray(removed_closed.mass_properties.inertia, dtype=np.float64)
        * removed_mass
        / float(removed_closed.volume)
    )
    old_com = np.asarray(_BINDINGS[link]["old_com_m"], dtype=np.float64)
    old_inertia = np.asarray(_BINDINGS[link]["old_inertia_kg_m2"], dtype=np.float64)
    new_com = (_OLD_MASS_KG * old_com - removed_mass * removed_com) / new_mass
    new_inertia = (
        old_inertia
        + _OLD_MASS_KG * _parallel_axis(old_com)
        - removed_inertia
        - removed_mass * _parallel_axis(removed_com)
        - new_mass * _parallel_axis(new_com)
    )
    eigenvalues = np.linalg.eigvalsh(new_inertia)
    if (
        not np.all(np.isfinite(new_inertia))
        or new_mass <= 0.0
        or np.any(eigenvalues <= 0.0)
        or float(eigenvalues[0] + eigenvalues[1]) < float(eigenvalues[2]) - 1.0e-12
    ):
        raise ValueError(f"{link}: estimated nail-free inertia is not physically valid")
    return {
        "method": _ESTIMATE_METHOD,
        "hardware_calibration_claimed": False,
        "runtime_binding_accepted": False,
        "old_urdf_com_m": old_com.tolist(),
        "old_urdf_inertia_at_com_kg_m2": old_inertia.tolist(),
        "old_closed_union_volume_m3": old_volume,
        "new_closed_union_volume_m3": new_volume,
        "old_mass_kg": _OLD_MASS_KG,
        "removed_nail_assembly_mass_kg": float(removed_mass),
        "removed_shape_volume_m3": float(removed_closed.volume),
        "removed_shape_com_m": removed_com.tolist(),
        "removed_shape_inertia_at_com_kg_m2": removed_inertia.tolist(),
        "new_mass_kg": float(new_mass),
        "new_com_m": new_com.tolist(),
        "new_inertia_at_com_kg_m2": new_inertia.tolist(),
        "new_inertia_eigenvalues_kg_m2": eigenvalues.tolist(),
    }
def _poly(ax, triangles: np.ndarray, axes: tuple[int, int], color, alpha: float) -> None:
    collection = PolyCollection(
        np.asarray(triangles)[:, :, axes] * 1000.0,
        facecolors=color,
        edgecolors="none",
        linewidths=0.0,
        alpha=alpha,
    )
    ax.add_collection(collection)


def _render(path: Path, rows: list[dict], mode: str) -> None:
    projections = ((0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ"))
    figure, axes = plt.subplots(3, 3, figsize=(15, 13), constrained_layout=True)
    for row_index, row in enumerate(rows):
        all_triangles = np.concatenate((row["body_triangles"], row["removed_triangles"]))
        for column, (first, second, label) in enumerate(projections):
            axis = axes[row_index, column]
            if mode == "nailfree":
                _poly(axis, row["body_triangles"], (first, second), "#b8b8b8", 0.85)
                _poly(axis, row["pad_triangles"], (first, second), "#1874d1", 0.95)
            else:
                _poly(axis, row["body_triangles"], (first, second), "#b8b8b8", 0.45)
                _poly(axis, row["removed_triangles"], (first, second), "#d62728", 1.0)
            points = all_triangles.reshape(-1, 3)[:, (first, second)] * 1000.0
            low, high = points.min(axis=0), points.max(axis=0)
            margin = max(float(np.ptp(points, axis=0).max()) * 0.04, 0.5)
            axis.set_xlim(low[0] - margin, high[0] + margin)
            axis.set_ylim(low[1] - margin, high[1] + margin)
            axis.set_aspect("equal", adjustable="box")
            axis.set_title(f"{row['link']} {label}")
            axis.set_xlabel("mm")
            axis.set_ylabel("mm")
            axis.grid(alpha=0.2)
    titles = {
        "nailfree": "Nail-free distal links; retained allowed PAD is blue",
        "removed_nail": "Removed tool shell and two mounting posts are red",
    }
    figure.suptitle(titles[mode])
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _build_link(
    root: Path, output: Path, physical_output: Path, link: str
) -> tuple[dict, dict]:
    mesh, removed_faces, removed, evidence = _identify_removable_nail(root, link)
    keep = np.ones(len(mesh.faces), dtype=bool)
    keep[removed_faces] = False
    body_triangles = mesh.face_vertices_m[keep]
    visual_path = output / f"{link}_nailfree.stl"
    _write_binary_stl(visual_path, body_triangles)
    reloaded, _ = load_stl_mesh(visual_path, unit="m", orient_outward=False)
    if (len(reloaded.faces) != _SOURCE_FACE_COUNT - _REMOVED_FACE_COUNT or
            _triangle_keys(reloaded.face_vertices_m) != _triangle_keys(body_triangles)):
        raise ValueError(f"{link}: written nail-free visual changed retained triangles")
    raw_body, physical = _closed_union(body_triangles)
    _, old_physical = _closed_union(mesh.face_vertices_m)
    deviation = _surface_deviation(raw_body, physical)
    removed_samples = np.vstack((removed.vertices, removed.triangles_center))
    inside = physical.contains(removed_samples)
    _, removed_to_body, _ = trimesh.proximity.closest_point(raw_body, removed_samples)
    _, removed_closed = _closed_union(removed.triangles)
    physical_path = physical_output / f"{link}_nailfree_closed.stl"
    physical.export(physical_path, file_type="stl")
    physical_reloaded = trimesh.load_mesh(physical_path, force="mesh", process=True)
    if not physical_reloaded.is_watertight or not physical_reloaded.is_winding_consistent:
        raise ValueError(f"{link}: written closed physical mesh is invalid")
    pad_path = root / _SEMANTIC_DIR / f"{link}_PAD_BODY_exact_source_local_m.npz"
    with np.load(pad_path, allow_pickle=False) as archive:
        pad_triangles = archive["points_local_m"][archive["faces"]]
    evidence.update({
        "visual_output": str(visual_path.relative_to(root)),
        "visual_output_sha256": file_sha256(visual_path),
        "closed_physical_output": str(physical_path.relative_to(root)),
        "closed_physical_output_sha256": file_sha256(physical_path),
        "closed_physical_face_count": int(len(physical.faces)),
        "closed_physical_watertight": True,
        "closed_union_method": {
            "voxel_pitch_m": _VOXEL_PITCH_M,
            "manifold_simplify_tolerance_m": _MANIFOLD_SIMPLIFY_TOLERANCE_M,
            "interpretation": "NUMERICAL_CLOSED_VOLUME_ESTIMATE_NOT_COLLISION_ACCEPTANCE",
            "manifold3d_version": importlib.metadata.version("manifold3d"),
            "trimesh_version": importlib.metadata.version("trimesh"),
        },
        "closed_physical_surface_deviation": deviation,
        "removed_surface_sample_count": int(len(removed_samples)),
        "removed_surface_samples_inside_closed_physical": int(np.count_nonzero(inside)),
        "inside_removed_sample_distance_to_retained_raw_max_m": float(
            np.max(removed_to_body[inside]) if np.any(inside) else 0.0
        ),
        "mass_properties": _mass_estimate(old_physical, physical, removed_closed, link),
    })
    render = {
        "link": link,
        "body_triangles": body_triangles,
        "removed_triangles": removed.triangles,
        "pad_triangles": pad_triangles,
    }
    return evidence, render


def main() -> int:
    arguments = _arguments()
    root = arguments.repository_root.resolve()
    output = (root / arguments.output_dir).resolve()
    audit_dir = (root / arguments.audit_dir).resolve()
    physical_output = audit_dir / "volume_estimate_sources"
    output.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    physical_output.mkdir(parents=True, exist_ok=True)
    records, render_rows = [], []
    for link in _BINDINGS:
        record, render = _build_link(root, output, physical_output, link)
        records.append(record)
        render_rows.append(render)
    images = {
        "three_views": audit_dir / "nailfree_three_views.png",
        "removed_nail_highlight": audit_dir / "removed_nail_highlight.png",
    }
    _render(images["three_views"], render_rows, "nailfree")
    _render(images["removed_nail_highlight"], render_rows, "removed_nail")
    report = {
        "schema_version": "carts_nailfree_hand_model_audit_v2",
        "status": "STATIC_VISUAL_COMPLETE_PHYSICAL_ESTIMATE_PROVISIONAL_NOT_RUNTIME_BOUND",
        "evidence_level": "STATIC_GEOMETRY_ONLY",
        "dynamic_grasp_claimed": False,
        "formal_dynamic_pass": False,
        "hardware_authorized": False,
        "hand_variant": "CONNECTOR_GRASP_NO_NAIL",
        "removed_role": "USER_CONFIRMED_REMOVABLE_TOOL_SHELL_AND_MOUNTING_POSTS",
        "source_assets_modified": False,
        "xacro_modified": False,
        "mass_property_interpretation": _ESTIMATE_METHOD,
        "collision_runtime_binding_accepted": False,
        "links": records,
        "audit_images": {
            name: {"path": str(path.relative_to(root)), "sha256": file_sha256(path)}
            for name, path in images.items()
        },
        "limitations": [
            "Authored STL is non-manifold; the matched watertight voxel unions are simulation estimates.",
            "Exact retained visual faces carry semantics and offline triangle checks.",
            "The watertight voxel unions are numerical volume estimates, not exact or conservative collision surfaces.",
            "Voxel pitch and simplification tolerance are numerical choices without a hardware-tolerance claim.",
            "Mass, COM and inertia subtract the estimated removed assembly from the bound old URDF baseline.",
            "The estimated inertial values remain provisional until runtime model review and binding.",
            "The two mounting posts are simulation geometry identities, not a hardware BOM claim.",
            "Collision decomposition is a separate fail-closed milestone.",
            "No URDF/Xacro runtime binding is performed by this builder.",
        ],
    }
    report_path = audit_dir / "NAILFREE_HAND_MODEL_AUDIT.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "report": str(report_path),
        "links": [
            {
                "link": row["source_path"].rsplit("/", 1)[-1].removesuffix(".STL"),
                "new_mass_kg": row["mass_properties"]["new_mass_kg"],
                "removed_nail_assembly_mass_kg": row["mass_properties"][
                    "removed_nail_assembly_mass_kg"
                ],
                "retained_faces": row["retained_face_count"],
            }
            for row in records
        ],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
