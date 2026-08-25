#!/usr/bin/env python3
"""Fail-closed f1Link3 topology-normalization audit.

The experiment is deliberately bounded.  It removes only exact duplicate
triangles, never moves an authored retained vertex, and refuses to invoke a
boolean operation unless every non-manifold edge has an even, uniquely
pairable face incidence.  No Isaac API is imported or started.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import trimesh


REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
AUDIT_DIR = Path(__file__).resolve().parent
INPUT_PATH = REPOSITORY_ROOT / (
    "src/iiwa_description/meshes/hand/connector_no_nail/"
    "f1Link3_nailfree.stl"
)
SOURCE_PATH = REPOSITORY_ROOT / "src/iiwa_description/meshes/hand/f1Link3.STL"
REPORT_PATH = AUDIT_DIR / "TOPOLOGY_NORMALIZATION_AUDIT.json"
EXPECTED_INPUT_SHA256 = (
    "965d327c466bec40b898fc4228f8ca240386bab3e8a79af6b48c798db1a0071a"
)
EXPECTED_SOURCE_SHA256 = (
    "7a33a6ab46729a2237dd13d99be3bcefb92bb3d4b77bbf9e69d884509cffcdb0"
)
REMOVED_FACE_RANGES_INCLUSIVE = ((11836, 12911), (12912, 13551), (13552, 14191))
SURFACE_ERROR_LIMIT_M = 1.0e-3
REMOVED_EXCLUSIVE_BOUNDARY_TOLERANCE_M = 1.0e-5
RETAINED_SURFACE_COVER_RADIUS_M = 2.5e-4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def registered_indexed_triangles(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    mesh = trimesh.load_mesh(path, force="mesh", process=False)
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    if triangles.shape != (11836, 3, 3) or not np.all(np.isfinite(triangles)):
        raise ValueError(f"unexpected retained triangle array: {triangles.shape}")
    flat = triangles.reshape(-1, 3)
    topology_keys = np.round(flat, decimals=8)
    _, first, inverse = np.unique(
        topology_keys, axis=0, return_index=True, return_inverse=True
    )
    vertices = flat[first]
    maximum_group_diameter_m = 0.0
    nonidentical_group_count = 0
    for group_index in range(len(vertices)):
        group = flat[inverse == group_index]
        if len(group) < 2:
            continue
        diameter = float(
            np.max(np.linalg.norm(group[:, None, :] - group[None, :, :], axis=2))
        )
        maximum_group_diameter_m = max(maximum_group_diameter_m, diameter)
        nonidentical_group_count += int(diameter > 0.0)
    registration = {
        "method": "ROUND_COORDINATES_TO_8_DECIMALS_FOR_TOPOLOGY_KEYS_ONLY",
        "coordinates_written_or_modified": False,
        "nonidentical_topology_group_count": nonidentical_group_count,
        "maximum_topology_group_diameter_m": maximum_group_diameter_m,
    }
    return vertices, inverse.reshape(-1, 3), registration


def remove_exact_duplicate_faces(
    faces: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for face_index, face in enumerate(faces):
        groups[tuple(sorted(int(value) for value in face))].append(face_index)
    duplicate_groups = [rows for rows in groups.values() if len(rows) > 1]
    retained = np.asarray([rows[0] for rows in groups.values()], dtype=np.int64)
    retained.sort()
    evidence = [
        {"retained_face_index": int(rows[0]), "removed_face_indices": rows[1:]}
        for rows in duplicate_groups
    ]
    return faces[retained], evidence


def edge_incidence(faces: np.ndarray) -> dict[tuple[int, int], list[int]]:
    result: dict[tuple[int, int], list[int]] = defaultdict(list)
    for face_index, face in enumerate(faces):
        for first, second in (
            (face[0], face[1]),
            (face[1], face[2]),
            (face[2], face[0]),
        ):
            result[tuple(sorted((int(first), int(second))))].append(face_index)
    return result


def face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = vertices[faces]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths == 0.0):
        raise ValueError("degenerate retained triangle")
    return normals / lengths[:, None]


def edge_record(
    edge: tuple[int, int],
    incident_faces: list[int],
    vertices: np.ndarray,
    normals: np.ndarray,
) -> dict[str, object]:
    selected = normals[np.asarray(incident_faces, dtype=np.int64)]
    incidence = len(incident_faces)
    return {
        "edge_vertex_indices": list(edge),
        "edge_endpoints_m": vertices[np.asarray(edge)].tolist(),
        "incident_face_count": incidence,
        "incident_face_indices_after_deduplication": incident_faces,
        "incident_face_normals": selected.tolist(),
        "normal_dot_matrix": (selected @ selected.T).tolist(),
        "perfect_pairing_possible_without_face_deletion_or_duplication": incidence % 2 == 0,
        "pairing_status": (
            "IMPOSSIBLE_ODD_FACE_INCIDENCE"
            if incidence % 2
            else "NOT_EVALUATED_AFTER_GLOBAL_ODD_INCIDENCE_FAILURE"
        ),
    }


def main() -> int:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    input_hash = sha256(INPUT_PATH)
    source_hash = sha256(SOURCE_PATH)
    if input_hash != EXPECTED_INPUT_SHA256 or source_hash != EXPECTED_SOURCE_SHA256:
        raise ValueError("bound f1Link3 input identity changed")

    vertices, faces, topology_registration = registered_indexed_triangles(INPUT_PATH)
    deduplicated_faces, duplicate_groups = remove_exact_duplicate_faces(faces)
    incidence = edge_incidence(deduplicated_faces)
    normals = face_normals(vertices, deduplicated_faces)
    nonmanifold = sorted(
        (edge, rows) for edge, rows in incidence.items() if len(rows) > 2
    )
    boundary_count = sum(len(rows) == 1 for rows in incidence.values())
    odd = [(edge, rows) for edge, rows in nonmanifold if len(rows) % 2]
    even = [(edge, rows) for edge, rows in nonmanifold if not len(rows) % 2]

    audit_mesh = trimesh.Trimesh(
        vertices=vertices, faces=deduplicated_faces, process=False
    )
    unique_pairing_pass = not odd
    boolean_performed = False
    production_mesh_written = False
    failure_reasons = []
    if odd:
        failure_reasons.append("ODD_NONMANIFOLD_EDGE_FACE_INCIDENCE_NOT_PAIRABLE")
    if nonmanifold:
        failure_reasons.append("NONMANIFOLD_EDGES_REMAIN_AFTER_EXACT_DUPLICATE_REMOVAL")

    topology_rows = [
        edge_record(edge, rows, vertices, normals) for edge, rows in nonmanifold
    ]
    gates = {
        "exact_input_identity": {"pass": True},
        "retained_vertices_not_moved": {
            "pass": True,
            "maximum_vertex_displacement_m": 0.0,
        },
        "unique_nonmanifold_edge_pairing": {
            "pass": unique_pairing_pass,
            "odd_incidence_edge_count": len(odd),
            "even_incidence_edge_count": len(even),
        },
        "watertight": {
            "pass": bool(audit_mesh.is_watertight),
            "status": "INPUT_AFTER_EXACT_DEDUPLICATION_ONLY",
        },
        "winding_consistent": {
            "pass": bool(audit_mesh.is_winding_consistent),
            "status": "INPUT_AFTER_EXACT_DEDUPLICATION_ONLY",
        },
        "nonmanifold_edge_count_zero": {
            "pass": len(nonmanifold) == 0,
            "count": len(nonmanifold),
        },
        "self_intersection_free": {
            "pass": False,
            "status": "NOT_EVALUATED_UNIQUE_PAIRING_FAILED_NO_CANDIDATE_SOLID",
        },
        "bidirectional_surface_error_with_cover_radius_le_1mm": {
            "pass": False,
            "status": "NOT_EVALUATED_NO_CANDIDATE_SOLID",
            "limit_m": SURFACE_ERROR_LIMIT_M,
            "retained_surface_cover_radius_m": RETAINED_SURFACE_COVER_RADIUS_M,
        },
        "removed_exclusive_occupancy_zero": {
            "pass": False,
            "status": "NOT_EVALUATED_NO_CANDIDATE_SOLID",
            "required_occupied_sample_count": 0,
            "boundary_tolerance_m": REMOVED_EXCLUSIVE_BOUNDARY_TOLERANCE_M,
        },
    }
    report = {
        "schema_version": "carts_nailfree_f1_topology_normalization_audit_v1",
        "status": "FAIL_CLOSED_NONMANIFOLD_PAIRING_NOT_UNIQUE",
        "evidence_level": "STATIC_GEOMETRY_ONLY",
        "hardware_authorized": False,
        "formal_dynamic_pass": False,
        "research_dynamic_pass": False,
        "runtime_binding_accepted": False,
        "isaac_started": False,
        "parameter_scan_performed": False,
        "boolean_backend": {
            "library": "manifold3d",
            "version": importlib.metadata.version("manifold3d"),
            "performed": boolean_performed,
            "reason_not_performed": (
                "INPUT_CANNOT_BE_UNIQUELY_SPLIT_INTO_MANIFOLD_SHELLS"
            ),
        },
        "input": {
            "path": str(INPUT_PATH.relative_to(REPOSITORY_ROOT)),
            "sha256": input_hash,
            "source_path": str(SOURCE_PATH.relative_to(REPOSITORY_ROOT)),
            "source_sha256": source_hash,
            "removed_source_face_ranges_inclusive": REMOVED_FACE_RANGES_INCLUSIVE,
        },
        "script": {
            "path": str(Path(__file__).resolve().relative_to(REPOSITORY_ROOT)),
            "sha256": sha256(Path(__file__).resolve()),
            "trimesh_version": trimesh.__version__,
            "numpy_version": np.__version__,
        },
        "exact_duplicate_removal": {
            "input_face_count": len(faces),
            "output_face_count": len(deduplicated_faces),
            "duplicate_face_count_removed": len(faces) - len(deduplicated_faces),
            "duplicate_groups": duplicate_groups,
            "vertex_count": len(vertices),
            "vertex_coordinates_modified": False,
            "topology_vertex_registration": topology_registration,
        },
        "topology": {
            "boundary_edge_count": boundary_count,
            "nonmanifold_edge_count": len(nonmanifold),
            "nonmanifold_incidence_histogram": dict(
                sorted(Counter(len(rows) for _, rows in nonmanifold).items())
            ),
            "odd_incidence_edge_count": len(odd),
            "even_incidence_edge_count": len(even),
            "all_nonmanifold_edges": topology_rows,
        },
        "gates": gates,
        "failure_reasons": failure_reasons,
        "output": {
            "production_mesh_written": production_mesh_written,
            "path": None,
            "sha256": None,
        },
        "conclusion": (
            "Fifty-six retained edges have three incident faces after removing "
            "all 64 exact duplicate triangles.  They cannot be partitioned "
            "into two-face manifold edge pairs without deleting or duplicating "
            "a retained face, so deterministic topology normalization is not "
            "uniquely defined and is rejected before boolean construction."
        ),
    }
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": report["status"],
        "duplicate_face_count_removed": len(faces) - len(deduplicated_faces),
        "nonmanifold_edge_count": len(nonmanifold),
        "odd_incidence_edge_count": len(odd),
        "production_mesh_written": production_mesh_written,
        "report": str(REPORT_PATH.relative_to(REPOSITORY_ROOT)),
    }, indent=2))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
