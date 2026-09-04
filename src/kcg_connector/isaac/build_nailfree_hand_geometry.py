#!/usr/bin/env python3
"""Build one conservative watertight nail-free mother mesh per terminal link.

The authored SolidWorks STL contains a separate nail shell and two mounting
posts, but the retained finger body is non-manifold.  This builder therefore:

1. removes the three bound nail face streams;
2. repairs the retained body through a fixed-resolution implicit volume;
3. subtracts a closed nail assembly (closed shell plus the minimum convex
   solids enclosing the two open mounting-post surfaces); and
4. fails closed unless every removed-exclusive sample is outside the result.

The result is a simulation geometry repair, not an exact CAD solid and not a
grasp-success claim.  The same emitted mesh is intended to be referenced by
both the visual and collision URDF elements.  PhysX collision cooking is
audited separately after USD import.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import manifold3d
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh


_BODY_FACE_STOP = 11836
_NAIL_FACE_RANGES = {
    "shell": (11836, 12912),
    "post_a": (12912, 13552),
    "post_b": (13552, 14192),
}
_SOURCE_FACE_COUNT = 14192
_OLD_MASS_KG = 0.057879
_SOURCE_BINDINGS = {
    "f1Link3": "7a33a6ab46729a2237dd13d99be3bcefb92bb3d4b77bbf9e69d884509cffcdb0",
    "f2Link2": "1758619f7ef1369fc3342c7032edee07222f9bdccc187c33830f9fa59bd508b3",
    "f3Link3": "93645443cff113b8c6e5a0280e3270192831d04246233cc45d9745c6e3c7d16e",
}
_CONTAINMENT_DIRECTIONS = np.asarray(
    (
        (0.4395064455, 0.617598629942, 0.652231566745),
        (-0.331006941, 0.823487771, 0.460911227),
        (0.706121318, -0.212702194, 0.675395214),
    ),
    dtype=np.float64,
)
_CONTAINMENT_DIRECTIONS /= np.linalg.norm(
    _CONTAINMENT_DIRECTIONS, axis=1, keepdims=True
)


def _arguments() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description="Build and audit the three watertight nail-free terminal meshes."
    )
    parser.add_argument("--repository-root", type=Path, default=repository)
    parser.add_argument(
        "--asset-dir",
        type=Path,
        default=Path(
            "src/iiwa_description/meshes/hand/connector_no_nail_watertight"
        ),
    )
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=Path(
            "artifacts/kcg_connector/isaac/"
            "cad_robust_grasp_goal_20260828/nailfree_geometry_run04"
        ),
    )
    parser.add_argument("--voxel-pitch-m", type=float, default=1.25e-4)
    parser.add_argument("--simplify-tolerance-m", type=float, default=5.0e-5)
    parser.add_argument("--exclusive-clearance-m", type=float, default=1.0e-5)
    parser.add_argument("--collateral-distance-m", type=float, default=5.0e-5)
    parser.add_argument("--worker-rss-limit-gib", type=float, default=45.0)
    parser.add_argument("--audit-chunk-points", type=int, default=32)
    parser.add_argument(
        "--implicit-worker-link",
        choices=tuple(_SOURCE_BINDINGS),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--implicit-worker-output", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _triangle_mesh(triangles: np.ndarray) -> trimesh.Trimesh:
    triangles = np.asarray(triangles, dtype=np.float64)
    mesh = trimesh.Trimesh(
        vertices=triangles.reshape(-1, 3),
        faces=np.arange(triangles.size // 3, dtype=np.int64).reshape(-1, 3),
        process=True,
    )
    mesh.remove_unreferenced_vertices()
    return mesh


def _edge_topology(mesh: trimesh.Trimesh) -> dict[str, int | bool]:
    edges = np.sort(np.asarray(mesh.edges, dtype=np.int64), axis=1)
    _, incidence = np.unique(edges, axis=0, return_counts=True)
    return {
        "vertex_count": int(len(mesh.vertices)),
        "face_count": int(len(mesh.faces)),
        "boundary_edge_count": int(np.count_nonzero(incidence == 1)),
        "nonmanifold_edge_count": int(np.count_nonzero(incidence > 2)),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
    }


def _to_manifold(mesh: trimesh.Trimesh) -> manifold3d.Manifold:
    oriented = mesh.copy()
    if float(oriented.volume) < 0.0:
        oriented.invert()
    value = manifold3d.Manifold(
        manifold3d.Mesh(
            vert_properties=np.asarray(oriented.vertices, dtype=np.float32),
            tri_verts=np.asarray(oriented.faces, dtype=np.uint32),
        )
    )
    if value.is_empty():
        raise ValueError("manifold3d rejected a supposedly closed mesh")
    return value


def _from_manifold(value: manifold3d.Manifold) -> trimesh.Trimesh:
    output = value.to_mesh()
    mesh = trimesh.Trimesh(
        vertices=np.asarray(output.vert_properties, dtype=np.float64)[:, :3],
        faces=np.asarray(output.tri_verts, dtype=np.int64),
        process=True,
    )
    if float(mesh.volume) < 0.0:
        mesh.invert()
    return mesh


def _implicit_repair(
    triangles: np.ndarray, pitch_m: float, simplify_m: float
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    raw = _triangle_mesh(triangles)
    voxels = raw.voxelized(pitch_m, method="subdivide").fill()
    marching = voxels.marching_cubes
    marching.apply_transform(voxels.transform)
    repaired = _from_manifold(_to_manifold(marching).simplify(simplify_m))
    metadata = {
        "voxel_shape": list(map(int, voxels.shape)),
        "filled_voxel_count": int(len(voxels.points)),
        "pitch_m": float(pitch_m),
        "simplify_tolerance_m": float(simplify_m),
    }
    del voxels, marching
    gc.collect()
    return repaired, metadata


def _implicit_worker(arguments: argparse.Namespace) -> int:
    if arguments.implicit_worker_output is None:
        raise ValueError("implicit worker requires an output path")
    root = arguments.repository_root.expanduser().resolve()
    link = arguments.implicit_worker_link
    source_path = root / "src/iiwa_description/meshes/hand" / f"{link}.STL"
    if _sha256(source_path) != _SOURCE_BINDINGS[link]:
        raise ValueError(f"{link}: authored source hash changed")
    source = trimesh.load_mesh(source_path, force="mesh", process=False)
    triangles = np.asarray(source.triangles, dtype=np.float64)
    if len(triangles) != _SOURCE_FACE_COUNT:
        raise ValueError(f"{link}: source face count changed")
    output = arguments.implicit_worker_output.expanduser().resolve()
    metadata_path = output.with_suffix(".json")
    if output.exists() or metadata_path.exists():
        raise FileExistsError(f"implicit worker refuses to overwrite {output}")
    repaired, metadata = _implicit_repair(
        triangles[:_BODY_FACE_STOP],
        arguments.voxel_pitch_m,
        arguments.simplify_tolerance_m,
    )
    repaired.export(output, file_type="stl")
    metadata.update(
        {
            "link": link,
            "output": str(output),
            "output_sha256": _sha256(output),
            "topology": _edge_topology(repaired),
        }
    )
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"link": link, "output": str(output)}), flush=True)
    return 0


def _worker_rss_kb(process_id: int) -> int:
    status = Path(f"/proc/{process_id}/status")
    try:
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except (FileNotFoundError, ProcessLookupError):
        return 0
    return 0


def _run_bounded_implicit_worker(
    arguments: argparse.Namespace,
    root: Path,
    link: str,
    output: Path,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--repository-root",
        str(root),
        "--voxel-pitch-m",
        str(arguments.voxel_pitch_m),
        "--simplify-tolerance-m",
        str(arguments.simplify_tolerance_m),
        "--implicit-worker-link",
        link,
        "--implicit-worker-output",
        str(output),
    ]
    limit_kb = int(arguments.worker_rss_limit_gib * 1024 * 1024)
    if limit_kb <= 0:
        raise ValueError("worker RSS limit must be positive")
    process = subprocess.Popen(
        command,
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    maximum_rss_kb = 0
    limit_hit = False
    while process.poll() is None:
        resident_kb = _worker_rss_kb(process.pid)
        maximum_rss_kb = max(maximum_rss_kb, resident_kb)
        if resident_kb > limit_kb:
            limit_hit = True
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            break
        time.sleep(0.25)
    stdout = process.stdout.read() if process.stdout is not None else ""
    if limit_hit:
        raise MemoryError(
            f"{link}: implicit worker exceeded {arguments.worker_rss_limit_gib} GiB; "
            f"maximum observed RSS {maximum_rss_kb} KiB"
        )
    if process.returncode != 0:
        raise RuntimeError(
            f"{link}: implicit worker failed with {process.returncode}: {stdout}"
        )
    metadata_path = output.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "worker_command": command,
            "worker_stdout": stdout.strip(),
            "worker_maximum_rss_kb": int(maximum_rss_kb),
            "worker_rss_limit_kb": int(limit_kb),
            "worker_rss_limit_hit": False,
        }
    )
    return metadata


def _nail_delete_solid(
    triangles: np.ndarray,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    solids: list[manifold3d.Manifold] = []
    rows: list[dict[str, Any]] = []
    for role, (start, stop) in _NAIL_FACE_RANGES.items():
        source = _triangle_mesh(triangles[start:stop])
        solid = source if role == "shell" else source.convex_hull
        if not solid.is_watertight or not solid.is_winding_consistent:
            raise ValueError(f"{role} deletion solid is not closed and consistently wound")
        solids.append(_to_manifold(solid))
        rows.append(
            {
                "role": role,
                "source_face_range_half_open": [int(start), int(stop)],
                "source_topology": _edge_topology(source),
                "closure_method": (
                    "EXACT_CLOSED_SOURCE_SHELL"
                    if role == "shell"
                    else "MINIMUM_CONVEX_SOLID_OF_OPEN_MOUNTING_POST"
                ),
                "solid_vertex_count": int(len(solid.vertices)),
                "solid_face_count": int(len(solid.faces)),
                "solid_volume_m3": float(abs(solid.volume)),
                "bounds_m": np.asarray(solid.bounds).tolist(),
            }
        )
    union = solids[0] + solids[1] + solids[2]
    return _from_manifold(union), {"parts": rows}


def _surface_samples(mesh: trimesh.Trimesh) -> np.ndarray:
    return np.vstack(
        (
            np.asarray(mesh.vertices),
            np.asarray(mesh.triangles_center),
            np.asarray(mesh.vertices)[np.asarray(mesh.edges_unique)].mean(axis=1),
        )
    )


def _distance_stats(distance: np.ndarray) -> dict[str, float | int]:
    distance = np.asarray(distance, dtype=np.float64)
    return {
        "sample_count": int(len(distance)),
        "maximum_m": float(np.max(distance)) if len(distance) else 0.0,
        "p95_m": float(np.percentile(distance, 95.0)) if len(distance) else 0.0,
        "p50_m": float(np.percentile(distance, 50.0)) if len(distance) else 0.0,
        "rms_m": (
            float(np.sqrt(np.mean(np.square(distance)))) if len(distance) else 0.0
        ),
    }


def _bounded_samples(points: np.ndarray, maximum: int = 30000) -> np.ndarray:
    points = np.asarray(points)
    if len(points) <= maximum:
        return points
    indices = np.linspace(0, len(points) - 1, maximum, dtype=np.int64)
    return points[indices]


def _closest_distance_chunked(
    mesh: trimesh.Trimesh, points: np.ndarray, chunk_points: int
) -> np.ndarray:
    """Return every closest-surface distance without a whole-query candidate array."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError("closest-distance points must have shape (n, 3)")
    if chunk_points <= 0:
        raise ValueError("audit chunk size must be positive")
    distance = np.empty(len(points), dtype=np.float64)
    for start in range(0, len(points), chunk_points):
        stop = min(start + chunk_points, len(points))
        _, distance[start:stop], _ = trimesh.proximity.closest_point(
            mesh, points[start:stop]
        )
    return distance


def _ray_parity_state(
    mesh: trimesh.Trimesh, points: np.ndarray, direction: np.ndarray
) -> np.ndarray:
    """Classify points as outside=-1, ambiguous=0, or inside=1 for one axis."""
    points = np.asarray(points, dtype=np.float64)
    state = np.full(len(points), -1, dtype=np.int8)
    if not len(points):
        return state
    inside_aabb = trimesh.bounds.contains(mesh.bounds, points)
    if not np.any(inside_aabb):
        return state
    active = points[inside_aabb]
    rays = np.tile(np.asarray(direction, dtype=np.float64), (len(active), 1))
    _, ray_index, _ = mesh.ray.intersects_location(
        np.vstack((active, active)),
        np.vstack((rays, -rays)),
        multiple_hits=True,
    )
    hits = np.bincount(ray_index, minlength=len(active) * 2).reshape((2, -1))
    parity = np.mod(hits, 2) == 1
    agree = parity[0] == parity[1]
    one_direction_free = np.any(hits == 0, axis=0)
    active_state = np.zeros(len(active), dtype=np.int8)
    active_state[agree & parity[0]] = 1
    active_state[agree & ~parity[0]] = -1
    active_state[~agree & one_direction_free] = -1
    state[np.flatnonzero(inside_aabb)] = active_state
    return state


def _contains_state_chunked(
    mesh: trimesh.Trimesh, points: np.ndarray, chunk_points: int
) -> np.ndarray:
    """Deterministic multi-axis containment with explicit numerical ambiguity."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError("containment points must have shape (n, 3)")
    if chunk_points <= 0:
        raise ValueError("audit chunk size must be positive")
    combined = np.empty(len(points), dtype=np.int8)
    for start in range(0, len(points), chunk_points):
        stop = min(start + chunk_points, len(points))
        direction_state = np.vstack(
            [
                _ray_parity_state(mesh, points[start:stop], direction)
                for direction in _CONTAINMENT_DIRECTIONS
            ]
        )
        has_inside = np.any(direction_state == 1, axis=0)
        has_outside = np.any(direction_state == -1, axis=0)
        ambiguous = np.any(direction_state == 0, axis=0) | (
            has_inside & has_outside
        )
        state = np.zeros(stop - start, dtype=np.int8)
        state[~ambiguous & has_inside] = 1
        state[~ambiguous & has_outside] = -1
        combined[start:stop] = state
    return combined


def _audit_link(
    link: str,
    triangles: np.ndarray,
    result: trimesh.Trimesh,
    retained_raw: trimesh.Trimesh,
    delete_solid: trimesh.Trimesh,
    old_closed: trimesh.Trimesh,
    parameters: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    removed_raw = _triangle_mesh(triangles[_BODY_FACE_STOP:])
    removed_samples = _surface_samples(removed_raw)
    distance_to_retained = _closest_distance_chunked(
        retained_raw, removed_samples, parameters.audit_chunk_points
    )
    exclusive_mask = distance_to_retained > parameters.exclusive_clearance_m
    exclusive = removed_samples[exclusive_mask]
    exclusive_to_result = _closest_distance_chunked(
        result, exclusive, parameters.audit_chunk_points
    )
    exclusive_state = _contains_state_chunked(
        result, exclusive, parameters.audit_chunk_points
    )
    occupied = exclusive_state == 1
    exclusive_ambiguous = exclusive_state == 0
    exclusive_clearance_violation = (
        exclusive_to_result <= parameters.exclusive_clearance_m
    )
    exclusive_gate_violation = (
        occupied | exclusive_ambiguous | exclusive_clearance_violation
    )

    retained_samples = np.vstack(
        (np.asarray(retained_raw.vertices), np.asarray(retained_raw.triangles_center))
    )
    retained_to_result = _closest_distance_chunked(
        result, retained_samples, parameters.audit_chunk_points
    )
    retained_state = _contains_state_chunked(
        result, retained_samples, parameters.audit_chunk_points
    )
    retained_inside = (retained_state == 1) | (
        retained_to_result <= trimesh.constants.tol.merge
    )
    collateral = (retained_state != 1) & (
        retained_to_result > parameters.collateral_distance_m
    )
    collateral_points = retained_samples[collateral]
    collateral_to_delete = _closest_distance_chunked(
        delete_solid, collateral_points, parameters.audit_chunk_points
    )
    collateral_delete_state = _contains_state_chunked(
        delete_solid, collateral_points, parameters.audit_chunk_points
    )
    collateral_inside_delete = collateral_delete_state == 1
    localization_limit = (
        float(np.sqrt(3.0) * parameters.voxel_pitch_m)
        + parameters.simplify_tolerance_m
    )
    localized = collateral_inside_delete | (
        collateral_to_delete <= localization_limit
    )

    final_samples = _bounded_samples(
        np.vstack((np.asarray(result.vertices), np.asarray(result.triangles_center)))
    )
    final_to_raw = _closest_distance_chunked(
        retained_raw, final_samples, parameters.audit_chunk_points
    )

    density = _OLD_MASS_KG / float(old_closed.volume)
    mass_properties = result.mass_properties
    inertia = np.asarray(mass_properties.inertia, dtype=np.float64) * density
    eigenvalues = np.linalg.eigvalsh(inertia)
    if np.any(eigenvalues <= 0.0):
        raise ValueError(f"{link}: reconstructed inertia is not positive definite")

    topology = _edge_topology(result)
    components = result.split(only_watertight=False)
    geometry_gate = bool(
        topology["watertight"]
        and topology["winding_consistent"]
        and len(components) == 1
        and float(result.volume) > 0.0
        and not np.any(exclusive_gate_violation)
        and np.all(localized)
    )
    audit = {
        "link": link,
        "source_sha256": _SOURCE_BINDINGS[link],
        "source_face_count": int(len(triangles)),
        "source_topology_after_vertex_merge": _edge_topology(
            _triangle_mesh(triangles)
        ),
        "retained_raw_topology": _edge_topology(retained_raw),
        "result_topology": topology,
        "result_connected_component_count": int(len(components)),
        "result_bounds_m": np.asarray(result.bounds).tolist(),
        "result_volume_m3": float(result.volume),
        "removed_exclusive_audit": {
            "definition": (
                "removed shell/post vertices, triangle centers, and edge midpoints "
                "whose distance to retained raw surface exceeds epsilon"
            ),
            "epsilon_m": float(parameters.exclusive_clearance_m),
            "candidate_sample_count": int(len(removed_samples)),
            "exclusive_sample_count": int(len(exclusive)),
            "occupied_exclusive_sample_count": int(np.count_nonzero(occupied)),
            "ambiguous_exclusive_sample_count": int(
                np.count_nonzero(exclusive_ambiguous)
            ),
            "required_result_clearance_m": float(
                parameters.exclusive_clearance_m
            ),
            "insufficient_clearance_sample_count": int(
                np.count_nonzero(exclusive_clearance_violation)
            ),
            "gate_violation_sample_count": int(
                np.count_nonzero(exclusive_gate_violation)
            ),
            "minimum_distance_to_result_m": (
                float(np.min(exclusive_to_result)) if len(exclusive) else None
            ),
            "maximum_occupied_depth_m": (
                float(np.max(exclusive_to_result[occupied]))
                if np.any(occupied)
                else 0.0
            ),
        },
        "retained_surface_collateral_audit": {
            "sample_count": int(len(retained_samples)),
            "outside_sample_count": int(np.count_nonzero(retained_state == -1)),
            "ambiguous_sample_count": int(np.count_nonzero(retained_state == 0)),
            "collateral_distance_m": float(parameters.collateral_distance_m),
            "outside_beyond_collateral_distance_count": int(
                np.count_nonzero(collateral)
            ),
            "maximum_outside_distance_m": (
                float(np.max(retained_to_result[~retained_inside]))
                if np.any(~retained_inside)
                else 0.0
            ),
            "collateral_inside_delete_solid_count": int(
                np.count_nonzero(collateral_inside_delete)
            ),
            "collateral_ambiguous_against_delete_solid_count": int(
                np.count_nonzero(collateral_delete_state == 0)
            ),
            "collateral_localization_limit_m": float(localization_limit),
            "collateral_not_localized_to_nail_overlap_count": int(
                np.count_nonzero(~localized)
            ),
            "interpretation": (
                "conservative loss is accepted only inside or within the fixed "
                "implicit-repair band of the nail/post deletion solid"
            ),
        },
        "surface_distance": {
            "retained_raw_to_result": _distance_stats(retained_to_result),
            "result_to_retained_raw_bounded_sample": _distance_stats(final_to_raw),
            "warning": (
                "retained raw contains authored internal/open surfaces; these distances "
                "are geometry diagnostics, not an exterior Hausdorff certificate"
            ),
        },
        "uniform_density_simulation_estimate": {
            "hardware_calibration_claimed": False,
            "old_runtime_mass_kg": _OLD_MASS_KG,
            "old_repaired_volume_m3": float(old_closed.volume),
            "estimated_density_kg_m3": float(density),
            "new_mass_kg": float(density * result.volume),
            "new_center_of_mass_m": np.asarray(
                mass_properties.center_mass
            ).tolist(),
            "new_inertia_at_center_of_mass_kg_m2": inertia.tolist(),
            "new_inertia_eigenvalues_kg_m2": eigenvalues.tolist(),
            "runtime_binding_accepted": False,
        },
        "offline_geometry_gate_pass": geometry_gate,
        "deterministic_containment": {
            "state_definition": "outside=-1, numerical_ambiguity=0, inside=1",
            "directions": _CONTAINMENT_DIRECTIONS.tolist(),
            "chunk_points": int(parameters.audit_chunk_points),
            "surface_policy": (
                "removed-exclusive samples require outside state and more than "
                "exclusive_clearance_m from the result; ambiguity fails closed"
            ),
        },
    }
    samples = {
        "removed_exclusive_samples_m": exclusive,
        "removed_exclusive_occupied": occupied.astype(np.uint8),
        "removed_exclusive_containment_state": exclusive_state,
        "removed_exclusive_gate_violation": exclusive_gate_violation.astype(
            np.uint8
        ),
        "removed_exclusive_distance_to_result_m": exclusive_to_result,
        "retained_samples_m": retained_samples,
        "retained_distance_to_result_m": retained_to_result,
        "retained_inside_result": retained_inside.astype(np.uint8),
        "retained_containment_state": retained_state,
        "result_samples_m": final_samples,
        "result_distance_to_retained_raw_m": final_to_raw,
    }
    return audit, samples


def _render_overlay(path: Path, rows: list[dict[str, Any]]) -> None:
    views = ((0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ"))
    figure, axes = plt.subplots(3, 3, figsize=(15, 13), constrained_layout=True)
    for row_index, row in enumerate(rows):
        body = _bounded_samples(row["body"], 9000)
        removed = _bounded_samples(row["removed"], 5000)
        result = _bounded_samples(row["result"], 12000)
        all_points = np.vstack((body, removed, result))
        for column, (first, second, label) in enumerate(views):
            axis = axes[row_index, column]
            axis.scatter(
                body[:, first] * 1000.0,
                body[:, second] * 1000.0,
                s=0.35,
                c="#888888",
                alpha=0.30,
                label="retained raw",
            )
            axis.scatter(
                result[:, first] * 1000.0,
                result[:, second] * 1000.0,
                s=0.35,
                c="#1769aa",
                alpha=0.45,
                label="watertight result",
            )
            axis.scatter(
                removed[:, first] * 1000.0,
                removed[:, second] * 1000.0,
                s=0.5,
                c="#d62728",
                alpha=0.60,
                label="removed nail assembly",
            )
            projected = all_points[:, [first, second]] * 1000.0
            low, high = projected.min(axis=0), projected.max(axis=0)
            margin = max(float(np.ptp(projected, axis=0).max()) * 0.04, 0.5)
            axis.set_xlim(low[0] - margin, high[0] + margin)
            axis.set_ylim(low[1] - margin, high[1] + margin)
            axis.set_aspect("equal", adjustable="box")
            axis.grid(alpha=0.2)
            axis.set_title(f"{row['link']} {label}")
            axis.set_xlabel("mm")
            axis.set_ylabel("mm")
    axes[0, 0].legend(loc="best", markerscale=5)
    figure.suptitle("Source-retained, removed nail assembly, and watertight result")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _render_distance_sections(path: Path, rows: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(3, 2, figsize=(13, 13), constrained_layout=True)
    for index, row in enumerate(rows):
        distance_mm = row["retained_distance"] * 1000.0
        axes[index, 0].hist(distance_mm, bins=80, color="#1769aa", alpha=0.85)
        axes[index, 0].set_title(f"{row['link']} retained raw to result")
        axes[index, 0].set_xlabel("nearest surface distance (mm)")
        axes[index, 0].set_ylabel("sample count")
        axes[index, 0].grid(alpha=0.2)

        points = row["result"]
        center_z = float(np.mean(np.asarray(row["bounds"])[:, 2]))
        half_width = row["pitch"] * 1.1
        section = points[np.abs(points[:, 2] - center_z) <= half_width]
        axes[index, 1].scatter(
            section[:, 0] * 1000.0,
            section[:, 1] * 1000.0,
            s=1.0,
            c="#1769aa",
            alpha=0.7,
        )
        axes[index, 1].set_title(
            f"{row['link']} mid-Z section, z={center_z * 1000.0:.3f} mm"
        )
        axes[index, 1].set_xlabel("x (mm)")
        axes[index, 1].set_ylabel("y (mm)")
        axes[index, 1].set_aspect("equal", adjustable="box")
        axes[index, 1].grid(alpha=0.2)
    figure.suptitle("Watertight-repair distance and cross-section diagnostics")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> int:
    arguments = _arguments()
    if arguments.implicit_worker_link is not None:
        return _implicit_worker(arguments)
    root = arguments.repository_root.expanduser().resolve()
    asset_dir = (root / arguments.asset_dir).resolve()
    evidence_dir = (root / arguments.evidence_dir).resolve()
    if asset_dir.exists() or evidence_dir.exists():
        raise FileExistsError(
            "refusing to overwrite an existing asset/evidence directory: "
            f"{asset_dir} or {evidence_dir}"
        )
    if not 0.0 < arguments.voxel_pitch_m <= 2.5e-4:
        raise ValueError("voxel pitch must be positive and no coarser than 0.25 mm")
    if not 0.0 <= arguments.simplify_tolerance_m <= arguments.voxel_pitch_m:
        raise ValueError("simplification tolerance must be within the voxel pitch")
    if arguments.audit_chunk_points <= 0:
        raise ValueError("audit chunk size must be positive")

    evidence_dir.mkdir(parents=True, exist_ok=False)
    candidate_dir = evidence_dir / "candidate_meshes"
    candidate_dir.mkdir()
    links: list[dict[str, Any]] = []
    render_rows: list[dict[str, Any]] = []
    all_pass = True
    for link, expected_hash in _SOURCE_BINDINGS.items():
        source_path = root / "src/iiwa_description/meshes/hand" / f"{link}.STL"
        actual_hash = _sha256(source_path)
        if actual_hash != expected_hash:
            raise ValueError(f"{link}: authored source hash changed")
        source = trimesh.load_mesh(source_path, force="mesh", process=False)
        triangles = np.asarray(source.triangles, dtype=np.float64)
        if len(triangles) != _SOURCE_FACE_COUNT:
            raise ValueError(f"{link}: source face count changed")

        retained_raw = _triangle_mesh(triangles[:_BODY_FACE_STOP])
        repaired_path = candidate_dir / f"{link}_retained_repaired.stl"
        repair_metadata = _run_bounded_implicit_worker(
            arguments,
            root,
            link,
            repaired_path,
        )
        repaired_body = trimesh.load_mesh(
            repaired_path, force="mesh", process=True, validate=False
        )
        delete_solid, deletion_metadata = _nail_delete_solid(triangles)
        repaired_manifold = _to_manifold(repaired_body)
        delete_manifold = _to_manifold(delete_solid)
        result = _from_manifold(repaired_manifold - delete_manifold)
        old_closed = _from_manifold(repaired_manifold + delete_manifold)
        audit, samples = _audit_link(
            link,
            triangles,
            result,
            retained_raw,
            delete_solid,
            old_closed,
            arguments,
        )
        audit["source_path"] = str(source_path.relative_to(root))
        audit["retained_repair"] = repair_metadata
        audit["old_full_proxy_for_mass_estimate"] = {
            "method": (
                "UNION_OF_SAME_PITCH_REPAIRED_RETAINED_BODY_AND_CLOSED_NAIL_"
                "DELETE_SOLID"
            ),
            "second_voxel_field_created": False,
            "hardware_calibration_claimed": False,
        }
        audit["nail_delete_solid"] = deletion_metadata
        all_pass &= bool(audit["offline_geometry_gate_pass"])

        candidate_path = candidate_dir / f"{link}_nailfree_watertight.stl"
        result.export(candidate_path, file_type="stl")
        reloaded = trimesh.load_mesh(
            candidate_path, force="mesh", process=True, validate=True
        )
        if not reloaded.is_watertight or not reloaded.is_winding_consistent:
            raise ValueError(f"{link}: serialized candidate lost closed topology")
        audit["candidate_mesh"] = str(candidate_path.relative_to(root))
        audit["candidate_mesh_sha256"] = _sha256(candidate_path)
        sample_path = evidence_dir / f"{link}_geometry_samples.npz"
        np.savez_compressed(sample_path, **samples)
        audit["raw_sample_data"] = str(sample_path.relative_to(root))
        audit["raw_sample_data_sha256"] = _sha256(sample_path)
        links.append(audit)
        render_rows.append(
            {
                "link": link,
                "body": np.asarray(retained_raw.vertices),
                "removed": np.asarray(_triangle_mesh(triangles[_BODY_FACE_STOP:]).vertices),
                "result": np.asarray(result.vertices),
                "retained_distance": samples["retained_distance_to_result_m"],
                "bounds": np.asarray(result.bounds),
                "pitch": arguments.voxel_pitch_m,
            }
        )
        del (
            source,
            triangles,
            retained_raw,
            repaired_body,
            repaired_manifold,
            delete_solid,
            delete_manifold,
            result,
            old_closed,
            reloaded,
        )
        gc.collect()

    overlay_path = evidence_dir / "source_removed_result_overlay.png"
    section_path = evidence_dir / "distance_and_cross_sections.png"
    _render_overlay(overlay_path, render_rows)
    _render_distance_sections(section_path, render_rows)

    published_assets: list[dict[str, str]] = []
    if all_pass:
        asset_dir.mkdir(parents=True, exist_ok=False)
        for row in links:
            source = root / row["candidate_mesh"]
            destination = asset_dir / source.name
            destination.write_bytes(source.read_bytes())
            published_assets.append(
                {
                    "path": str(destination.relative_to(root)),
                    "sha256": _sha256(destination),
                }
            )

    audit_path = evidence_dir / "NAILFREE_GEOMETRY_AUDIT.json"
    document = {
        "schema_version": "kcg_nailfree_watertight_geometry_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "OFFLINE_GEOMETRY_PASS" if all_pass else "OFFLINE_GEOMETRY_REJECTED",
        "claim_scope": (
            "CAD_DERIVED_REPAIRED_SIMULATION_GEOMETRY_ONLY_NOT_USD_COLLISION_"
            "READBACK_OR_DYNAMIC_GRASP_EVIDENCE"
        ),
        "hardware_authorized": False,
        "formal_dynamic_pass": False,
        "runtime_binding_accepted": False,
        "parameters": {
            "voxel_pitch_m": arguments.voxel_pitch_m,
            "simplify_tolerance_m": arguments.simplify_tolerance_m,
            "exclusive_clearance_m": arguments.exclusive_clearance_m,
            "collateral_distance_m": arguments.collateral_distance_m,
            "worker_rss_limit_gib": arguments.worker_rss_limit_gib,
            "audit_chunk_points": arguments.audit_chunk_points,
            "manifold3d_version": "3.5.2",
            "trimesh_version": trimesh.__version__,
        },
        "set_definition": {
            "B_raw": "retained source triangles with face indices [0, 11836)",
            "R_p_B": "watertight implicit repair of B_raw at pitch p",
            "N": (
                "exact closed nail shell union minimum convex solids of the two "
                "open mounting-post surfaces"
            ),
            "G": "R_p_B set-minus N",
            "D_exclusive": (
                "sampled N points farther than epsilon from B_raw; every such point "
                "must lie outside G"
            ),
            "future_allowed_pad": (
                "a subset of boundary(G) outside the nail/post deletion-overlap set; "
                "its contact-role mapping is audited before USD preflight"
            ),
        },
        "links": links,
        "published_assets": published_assets,
        "figures": [
            str(overlay_path.relative_to(root)),
            str(section_path.relative_to(root)),
        ],
        "reproduction_command": (
            ".venv/bin/python "
            "src/kcg_connector/isaac/build_nailfree_hand_geometry.py"
        ),
    }
    audit_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"audit": str(audit_path), "status": document["status"]}))
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
