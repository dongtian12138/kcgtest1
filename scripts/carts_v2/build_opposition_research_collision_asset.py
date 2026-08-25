#!/usr/bin/env python3
"""Build and fail-closed audit the nail-free compound-convex research asset."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
import trimesh
from kcg_connector.grasp.carts_v2.models import joint_positions_for_phases, load_v2_inputs
from kcg_connector.grasp.robust.object_model import file_sha256, load_stl_mesh
_HAND_AUDIT = Path("artifacts/carts_v2/nailfree_height_projected/hand_model_audit/NAILFREE_HAND_MODEL_AUDIT.json")
_TASK_SURFACES = Path("artifacts/carts_v2/nailfree_height_projected/task_grip_surface_audit/TASK_GRIP_SURFACE_MANIFEST.json")
_CONFIG = Path("src/kcg_connector/config/carts_full_palm_search.yaml")
_OUTPUT = Path("artifacts/carts_v2/opposition60_isaac/research_collision_asset")
_OBJECT = "te_deutsch_d38999_26fj35pn_step"
_SOURCE_FACE_COUNT = 11836
_REMOVED_FACE_COUNT = 2356
_EXPECTED = {
    "f1Link3": "965d327c466bec40b898fc4228f8ca240386bab3e8a79af6b48c798db1a0071a",
    "f2Link2": "3d11ab9797c2ed6e4c622c3ba6c63b2c9fb8258dbf968326828046efc788e893",
    "f3Link3": "62d7aa934e7516f83d884adfe6f518e446d3db612d41bd81edfee96ed2f7e27b",
}
_COACD = {
    "method_id": "REGISTERED_TE_J35_COACD_V1",
    "threshold": 0.05, "max_convex_hull": 64,
    "preprocess_mode": "auto", "preprocess_resolution": 30,
    "resolution": 1000, "mcts_nodes": 10, "mcts_iterations": 50,
    "mcts_max_depth": 3, "pca": False, "merge": True, "decimate": True,
    "max_ch_vertex": 128,
    "seed": 20260823,
}
_PHYSX = {
    "max_convex_hulls": 64, "max_hull_vertices": 64,
    "voxel_resolution": 1_000_000, "error_percentage": 0.5, "shrink_wrap": True,
}
_PHYSX_SUBDIR, _PHYSX_CLI = ("physx_vhacd_shrinkwrap",
                              Path("scripts/carts_v2/run_physx_vhacd_shrinkwrap.py"))
_EXCLUSIVE_DISTANCE_M = 1.0e-5
_BOUNDARY_OCCUPANCY_TOLERANCE_M = 1.0e-9
_EXTERIOR_PROBE_OFFSET_M = 1.0e-7
_RAW_SAMPLE_COUNT, _REGION_SAMPLE_COUNT, _COMPOUND_SAMPLE_COUNT = 30000, 12000, 30000
def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=_OUTPUT)
    parser.add_argument("--backend", choices=("COACD", "PHYSX_VHACD_SHRINK_WRAP"),
                        default="COACD")
    return parser.parse_args()
def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
def _trimesh(mesh) -> trimesh.Trimesh:
    return trimesh.Trimesh(vertices=np.asarray(mesh.vertices_m),
                           faces=np.asarray(mesh.faces), process=False)
def _removed_indices(record: dict) -> np.ndarray:
    rows = [
        np.arange(int(low), int(high) + 1, dtype=np.int64)
        for low, high in (
            item["source_face_index_range_inclusive"]
            for item in record["removed_components"]
        )
    ]
    result = np.concatenate(rows)
    if len(result) != _REMOVED_FACE_COUNT or len(np.unique(result)) != len(result):
        raise ValueError("removed nail face identity changed")
    return result
def _load_bound_geometry(root: Path, link: str, audit: dict, surface: dict) -> dict:
    raw_path = root / audit["visual_output"]
    original_path = root / audit["source_path"]
    surface_path = root / surface["surface_npz"]
    for path in (raw_path, original_path, surface_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if (file_sha256(raw_path) != _EXPECTED[link]
            or audit["visual_output_sha256"] != _EXPECTED[link]
            or surface["source_mesh_sha256"] != _EXPECTED[link]):
        raise ValueError(f"{link}: 11836-face nail-free source hash changed")
    raw, raw_provenance = load_stl_mesh(raw_path, unit="m", orient_outward=False)
    original, original_provenance = load_stl_mesh(original_path, unit="m", orient_outward=False)
    if (len(raw.faces) != _SOURCE_FACE_COUNT
            or raw_provenance.source_sha256 != _EXPECTED[link]
            or original_provenance.source_sha256 != audit["source_sha256"]):
        raise ValueError(f"{link}: source face count or provenance changed")
    removed = _removed_indices(audit)
    retained = np.ones(len(original.faces), dtype=np.bool_)
    retained[removed] = False
    if not np.array_equal(raw.face_vertices_m, original.face_vertices_m[retained]):
        raise ValueError(f"{link}: nail-free mesh is not the exact retained face stream")
    if file_sha256(surface_path) != surface["surface_npz_sha256"]:
        raise ValueError(f"{link}: TASK_GRIP_SURFACE hash changed")
    with np.load(surface_path, allow_pickle=False) as archive:
        task_faces = np.asarray(archive["source_face_indices"], dtype=np.int64)
        task_triangles = np.asarray(archive["points_local_m"])[archive["faces"]]
    if (len(task_faces) != int(surface["task_face_count"])
            or not np.array_equal(raw.face_vertices_m[task_faces], task_triangles)):
        raise ValueError(f"{link}: TASK_GRIP_SURFACE no longer maps to source faces")
    return {"raw": raw, "original": original, "removed_indices": removed,
            "task_faces": task_faces, "raw_path": raw_path, "surface_path": surface_path}
def _qp60_table_masks(root: Path, geometries: dict) -> tuple[dict, dict]:
    inputs = load_v2_inputs(root, config_path=root / _CONFIG, object_id=_OBJECT)
    names = inputs.hand_model.independent_joint_names
    reference = np.asarray([inputs.hand_model.joints[name].limit.lower for name in names],
                           dtype=np.float64)
    reference[names.index("f1j1")] = np.deg2rad(60.0)
    joints = joint_positions_for_phases(inputs, (0.1, 0.1, 0.1),
                                        reference_joint_positions_rad=reference)
    transforms = inputs.hand_model.forward_kinematics(joints)
    masks = {}
    for link, geometry in geometries.items():
        rotation = transforms[link][:3, :3]
        normals_handbase = geometry["raw"].face_normals @ rotation.T
        masks[link] = normals_handbase[:, 2] < 0.0
        if not np.any(masks[link]):
            raise ValueError(f"{link}: q_p=60 table-facing surface is empty")
    reference_record = {
        "palm_configuration_rad": float(np.deg2rad(60.0)),
        "preshape_phases": [0.1, 0.1, 0.1],
        "joint_positions_rad": np.asarray(joints).tolist(),
        "table_facing_rule": "FACE_NORMAL_IN_HANDBASE_DOT_NEGATIVE_Z_IS_POSITIVE",
        "transforms_handbase_from_link": {
            link: np.asarray(transforms[link]).tolist() for link in geometries},
    }
    return masks, reference_record
def _run_coacd(raw) -> tuple[list[trimesh.Trimesh], float]:
    import coacd
    start = time.perf_counter()
    result = coacd.run_coacd(
        coacd.Mesh(np.asarray(raw.vertices_m, dtype=np.float64),
                   np.asarray(raw.faces, dtype=np.int32)),
        **{key: value for key, value in _COACD.items() if key != "method_id"},
    )
    elapsed = time.perf_counter() - start
    hulls = []
    for vertices, faces in result:
        hull = trimesh.Trimesh(vertices=vertices, faces=faces, process=True, validate=True)
        if float(hull.volume) < 0.0:
            hull.invert()
        hulls.append(hull)
    return hulls, elapsed
def _run_physx_batch(root: Path, output: Path, geometries: dict) -> tuple[dict, dict]:
    builder_path, runner_path = Path(__file__).resolve(), root / _PHYSX_CLI
    request = {
        "schema_version": "physx_vhacd_shrinkwrap_request_v2",
        "output_dir": str(output),
        "parameters": _PHYSX,
        "executed_source_chain": {"builder_sha256": file_sha256(builder_path),
                                  "runner_sha256": file_sha256(runner_path)},
        "links": [{
            "link": link, "source_mesh": str(geometry["raw_path"]),
            "source_mesh_sha256": _EXPECTED[link],
            "source_triangle_count": _SOURCE_FACE_COUNT,
        } for link, geometry in geometries.items()],
    }
    request_path = output / "PHYSX_VHACD_REQUEST.json"
    request_path.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    command = [
        str(root / "src/kcg_connector/isaac/run_isaac_python.sh"),
        str(root / _PHYSX_CLI), "--request", str(request_path),
    ]
    completed = subprocess.run(command, cwd=root, text=True, capture_output=True, check=False)
    (output / "physx_stdout.log").write_text(completed.stdout, encoding="utf-8")
    (output / "physx_stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"PhysX VHACD batch failed with exit {completed.returncode}")
    manifest = _json(output / "PHYSX_VHACD_BATCH.json")
    binding = manifest.get("executed_source", {})
    if (binding.get("sha256") != request["executed_source_chain"]["runner_sha256"] or
            Path(binding.get("script", "")).resolve() != runner_path.resolve() or
            manifest.get("executed_source_chain_request") != request["executed_source_chain"] or
            file_sha256(builder_path) != request["executed_source_chain"]["builder_sha256"]):
        raise RuntimeError("executed source chain changed during PhysX batch")
    rows = {}
    for record in manifest["links"]:
        hulls = [trimesh.load_mesh(Path(row["path"]), force="mesh", process=True)
                 for row in record["hulls"]]
        rows[record["link"]] = (hulls, float(record["elapsed_s"]))
    return rows, manifest
def _hull_gate(hulls: list[trimesh.Trimesh], max_hulls: int,
               max_vertices: int | None = None) -> tuple[bool, list[dict]]:
    records = []
    for index, hull in enumerate(hulls):
        record = {
            "index": index,
            "watertight": bool(hull.is_watertight),
            "winding_consistent": bool(hull.is_winding_consistent),
            "convex": bool(hull.is_convex),
            "finite": bool(np.all(np.isfinite(hull.vertices))),
            "volume_m3": float(hull.volume),
            "vertex_count": int(len(hull.vertices)),
            "triangle_count": int(len(hull.faces)),
        }
        record["vertex_limit_pass"] = max_vertices is None or record["vertex_count"] <= max_vertices
        record["pass"] = bool(record["watertight"] and record["winding_consistent"]
                              and record["convex"] and record["finite"]
                              and record["volume_m3"] > 0.0 and record["vertex_limit_pass"])
        records.append(record)
    passed = bool(hulls) and len(hulls) <= max_hulls
    return passed and all(row["pass"] for row in records), records
def _sample_faces(mesh: trimesh.Trimesh, face_mask: np.ndarray, count: int,
                  seed: int) -> tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(mesh.area_faces) * np.asarray(face_mask, dtype=np.float64)
    if not np.any(weights > 0.0):
        raise ValueError("surface sampling region is empty")
    points, faces = trimesh.sample.sample_surface(
        mesh, count=count, face_weight=weights, seed=seed
    )
    return np.asarray(points), np.asarray(faces, dtype=np.int64)
def _exterior_compound_samples(
    hulls: list[trimesh.Trimesh], count: int, seed: int
) -> np.ndarray:
    areas = np.asarray([hull.area for hull in hulls], dtype=np.float64)
    quotas = np.maximum(16, np.floor(count * areas / areas.sum()).astype(np.int64))
    points, probes, owners = [], [], []
    for index, (hull, quota) in enumerate(zip(hulls, quotas)):
        sample, faces = trimesh.sample.sample_surface(
            hull, int(quota), seed=seed + index
        )
        normal = hull.face_normals[np.asarray(faces, dtype=np.int64)]
        points.append(sample)
        probes.append(sample + _EXTERIOR_PROBE_OFFSET_M * normal)
        owners.append(np.full(len(sample), index, dtype=np.int64))
    points_array = np.vstack(points)
    probe_array = np.vstack(probes)
    owner_array = np.concatenate(owners)
    internal = np.zeros(len(points_array), dtype=np.bool_)
    for index, hull in enumerate(hulls):
        eligible = (~internal) & (owner_array != index)
        if np.any(eligible):
            internal[eligible] |= hull.contains(probe_array[eligible])
    exterior = points_array[~internal]
    if not len(exterior):
        raise ValueError("compound convex union has no sampled exterior surface")
    return exterior
def _distribution(values: np.ndarray) -> dict:
    value = np.asarray(values, dtype=np.float64)
    if not len(value) or not np.all(np.isfinite(value)):
        raise ValueError("distance distribution is empty or non-finite")
    maximum = float(np.max(value))
    edges = np.linspace(0.0, max(maximum, 1.0e-12), 21)
    counts, edges = np.histogram(value, bins=edges)
    return {
        "sample_count": int(len(value)),
        "mean_m": float(np.mean(value)),
        "rms_m": float(np.sqrt(np.mean(np.square(value)))),
        "p50_m": float(np.percentile(value, 50.0)),
        "p90_m": float(np.percentile(value, 90.0)),
        "p95_m": float(np.percentile(value, 95.0)),
        "p99_m": float(np.percentile(value, 99.0)),
        "maximum_m": maximum,
        "histogram_edges_m": edges.tolist(),
        "histogram_counts": counts.tolist(),
    }
def _distance_audit(
    raw: trimesh.Trimesh,
    hulls: list[trimesh.Trimesh],
    task_mask: np.ndarray,
    table_mask: np.ndarray,
    seed: int,
) -> dict:
    compound = trimesh.util.concatenate(hulls)
    exterior = _exterior_compound_samples(hulls, _COMPOUND_SAMPLE_COUNT, seed + 100)
    _, compound_to_raw, nearest_raw_face = trimesh.proximity.closest_point(raw, exterior)
    regions = {
        "global": np.ones(len(raw.faces), dtype=np.bool_),
        "task_grip_surface": task_mask,
        "table_facing_at_qp60": table_mask,
    }
    result = {}
    for offset, (name, mask) in enumerate(regions.items()):
        count = _RAW_SAMPLE_COUNT if name == "global" else _REGION_SAMPLE_COUNT
        raw_points, _ = _sample_faces(raw, mask, count, seed + offset)
        _, raw_to_compound, _ = trimesh.proximity.closest_point(compound, raw_points)
        compound_region = mask[np.asarray(nearest_raw_face, dtype=np.int64)]
        result[name] = {
            "raw_to_compound": _distribution(raw_to_compound),
            "compound_to_raw": _distribution(compound_to_raw[compound_region]),
            "compound_region_rule": "EXTERIOR_SAMPLE_CLASSIFIED_BY_NEAREST_RAW_FACE",
        }
    result["compound_exterior_sample_count"] = int(len(exterior))
    return result
def _removed_exclusive_audit(
    geometry: dict, hulls: list[trimesh.Trimesh], output: Path, link: str
) -> tuple[dict, np.ndarray, np.ndarray]:
    removed = geometry["original"].face_vertices_m[geometry["removed_indices"]]
    points = np.vstack(
        (
            removed.reshape(-1, 3),
            removed.mean(axis=1),
            (removed[:, 0] + removed[:, 1]) * 0.5,
            (removed[:, 1] + removed[:, 2]) * 0.5,
            (removed[:, 2] + removed[:, 0]) * 0.5,
        )
    )
    points = np.unique(points, axis=0)
    raw = _trimesh(geometry["raw"])
    _, distance_to_raw, _ = trimesh.proximity.closest_point(raw, points)
    exclusive = points[distance_to_raw > _EXCLUSIVE_DISTANCE_M]
    exclusive_distance = distance_to_raw[distance_to_raw > _EXCLUSIVE_DISTANCE_M]
    occupied = np.zeros(len(exclusive), dtype=np.bool_)
    first_hull = np.full(len(exclusive), -1, dtype=np.int64)
    per_hull = []
    for index, hull in enumerate(hulls):
        inside = hull.contains(exclusive)
        _, boundary_distance, _ = trimesh.proximity.closest_point(hull, exclusive)
        hit = inside | (boundary_distance <= _BOUNDARY_OCCUPANCY_TOLERANCE_M)
        newly = hit & ~occupied
        first_hull[newly] = index
        occupied |= hit
        per_hull.append(int(np.count_nonzero(hit)))
    sample_path = output / f"{link}_removed_exclusive_samples.npz"
    np.savez(
        sample_path,
        points_local_m=exclusive,
        distance_to_retained_raw_m=exclusive_distance,
        occupied_by_compound=occupied,
        first_occupying_hull=first_hull,
    )
    record = {
        "selection": "REMOVED_VERTICES_CENTERS_AND_EDGE_MIDPOINTS_GT_10UM_FROM_RETAINED_RAW",
        "separation_from_retained_raw_m": _EXCLUSIVE_DISTANCE_M,
        "boundary_occupancy_tolerance_m": _BOUNDARY_OCCUPANCY_TOLERANCE_M,
        "candidate_sample_count": int(len(points)),
        "exclusive_sample_count": int(len(exclusive)),
        "occupied_exclusive_sample_count": int(np.count_nonzero(occupied)),
        "occupied_exclusive_fraction": float(np.mean(occupied)) if len(occupied) else 1.0,
        "occupied_sample_count_by_hull": per_hull,
        "sample_npz": str(sample_path),
        "sample_npz_sha256": file_sha256(sample_path),
        "pass": bool(len(exclusive) and not np.any(occupied)),
    }
    return record, removed, exclusive[occupied]
def _poly(axis, triangles: np.ndarray, axes: tuple[int, int], color, alpha: float) -> None:
    axis.add_collection(
        PolyCollection(
            triangles[:, :, axes] * 1000.0,
            facecolors=color,
            edgecolors="none",
            alpha=alpha,
            rasterized=True,
        )
    )
def _render_overlay(
    path: Path,
    link: str,
    raw: trimesh.Trimesh,
    hulls: list[trimesh.Trimesh],
    removed: np.ndarray,
    occupied: np.ndarray,
) -> None:
    compound = np.vstack([hull.triangles for hull in hulls])
    projections = ((0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ"))
    figure, axes = plt.subplots(1, 4, figsize=(18, 4.5), constrained_layout=True)
    for axis, (first, second, label) in zip(axes[:3], projections):
        _poly(axis, raw.triangles, (first, second), "#808080", 0.24)
        _poly(axis, compound, (first, second), "#13a8c7", 0.16)
        points = raw.vertices[:, (first, second)] * 1000.0
        low, high = points.min(axis=0), points.max(axis=0)
        margin = max(float(np.ptp(points, axis=0).max()) * 0.04, 0.5)
        axis.set_xlim(low[0] - margin, high[0] + margin)
        axis.set_ylim(low[1] - margin, high[1] + margin)
        axis.set_title(label)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.2)
    extents = np.ptp(removed.reshape(-1, 3), axis=0)
    pairs = ((0, 1), (0, 2), (1, 2))
    first, second = max(pairs, key=lambda pair: float(extents[list(pair)].prod()))
    zoom = axes[3]
    _poly(zoom, raw.triangles, (first, second), "#808080", 0.25)
    _poly(zoom, compound, (first, second), "#13a8c7", 0.18)
    _poly(zoom, removed, (first, second), "#d62728", 0.20)
    if len(occupied):
        zoom.scatter(
            occupied[:, first] * 1000.0,
            occupied[:, second] * 1000.0,
            s=4,
            c="#8b0000",
            label="occupied exclusive",
        )
    points = removed.reshape(-1, 3)[:, (first, second)] * 1000.0
    low, high = points.min(axis=0), points.max(axis=0)
    margin = max(float(np.ptp(points, axis=0).max()) * 0.15, 0.5)
    zoom.set_xlim(low[0] - margin, high[0] + margin)
    zoom.set_ylim(low[1] - margin, high[1] + margin)
    zoom.set_title(f"deleted region {first}{second}")
    zoom.set_aspect("equal", adjustable="box")
    zoom.grid(alpha=0.2)
    figure.suptitle(
        f"{link}: exact raw gray, compound convex cyan, deleted nail red"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)
def _build_link(
    root: Path,
    output: Path,
    link: str,
    geometry: dict,
    table_mask: np.ndarray,
    decomposition: tuple[list[trimesh.Trimesh], float] | None,
    backend: str,
) -> dict:
    hulls, elapsed = decomposition or _run_coacd(geometry["raw"])
    maximum = _PHYSX["max_convex_hulls"] if decomposition else _COACD["max_convex_hull"]
    vertex_limit = _PHYSX["max_hull_vertices"] if decomposition else None
    hull_gate, hull_records = _hull_gate(hulls, int(maximum), vertex_limit)
    link_output = output / link
    link_output.mkdir(parents=True, exist_ok=False)
    hull_files = []
    for index, hull in enumerate(hulls):
        path = link_output / f"{link}_{backend.lower()}_hull_{index:02d}.stl"
        hull.export(path, file_type="stl")
        reloaded = trimesh.load_mesh(path, force="mesh", process=True)
        if not (
            reloaded.is_watertight
            and reloaded.is_winding_consistent
            and reloaded.is_convex
            and float(reloaded.volume) > 0.0
        ):
            hull_gate = False
        hull_files.append({"path": str(path), "sha256": file_sha256(path)})
    task_mask = np.zeros(_SOURCE_FACE_COUNT, dtype=np.bool_)
    task_mask[geometry["task_faces"]] = True
    distance = _distance_audit(
        _trimesh(geometry["raw"]), hulls, task_mask, table_mask, _COACD["seed"]
    )
    exclusive, removed, occupied = _removed_exclusive_audit(
        geometry, hulls, link_output, link
    )
    overlay = link_output / f"{link}_raw_compound_deleted_overlay.png"
    _render_overlay(
        overlay, link, _trimesh(geometry["raw"]), hulls, removed, occupied
    )
    passed = bool(hull_gate and exclusive["pass"])
    return {
        "link": link,
        "status": "STATIC_GEOMETRY_ASSET_CANDIDATE" if passed else "STATIC_GEOMETRY_ASSET_REJECTED",
        "static_geometry_asset_candidate": passed,
        "runtime_binding_accepted": False,
        "source_mesh": str(geometry["raw_path"].relative_to(root)),
        "source_mesh_sha256": _EXPECTED[link],
        "source_triangle_count": _SOURCE_FACE_COUNT,
        "decomposition_backend": backend,
        "decomposition_elapsed_s": elapsed,
        "hull_count": len(hulls),
        "hull_count_within_limit": bool(len(hulls) <= maximum),
        "hull_validation_pass": hull_gate,
        "hulls": hull_records,
        "hull_files": hull_files,
        "distance_audit": distance,
        "removed_nail_exclusive_audit": exclusive,
        "overlay_png": str(overlay),
        "overlay_png_sha256": file_sha256(overlay),
    }
load_bound_geometry, qp60_table_masks, audit_compound_link = (
    _load_bound_geometry, _qp60_table_masks, _build_link)
def main() -> int:
    arguments = _arguments()
    root = arguments.repository_root.resolve()
    builder_path, executed_builder_sha256 = Path(__file__).resolve(), file_sha256(Path(__file__).resolve())
    output = (root / arguments.output_dir).resolve()
    if arguments.backend == "PHYSX_VHACD_SHRINK_WRAP":
        output /= _PHYSX_SUBDIR
    if output.exists():
        raise FileExistsError(f"refusing to overwrite one-shot output: {output}")
    output.mkdir(parents=True)
    audit_path, surface_path = root / _HAND_AUDIT, root / _TASK_SURFACES
    audit, surfaces = _json(audit_path), _json(surface_path)
    if (
        audit.get("hand_variant") != "CONNECTOR_GRASP_NO_NAIL"
        or surfaces.get("semantic") != "TASK_GRIP_SURFACE"
        or audit.get("hardware_authorized") is not False
        or surfaces.get("hardware_authorized") is not False
    ):
        raise ValueError("nail-free hand or task-surface identity is not accepted")
    audit_by_link = {Path(row["source_path"]).stem: row for row in audit["links"]}
    surface_by_link = {row["link_name"]: row for row in surfaces["links"]}
    if set(audit_by_link) != set(_EXPECTED) or set(surface_by_link) != set(_EXPECTED):
        raise ValueError("three distal link identities changed")
    geometries = {
        link: _load_bound_geometry(
            root, link, audit_by_link[link], surface_by_link[link]
        )
        for link in _EXPECTED
    }
    table_masks, table_reference = _qp60_table_masks(root, geometries)
    physx_rows, physx_manifest = ({}, None)
    if arguments.backend == "PHYSX_VHACD_SHRINK_WRAP":
        physx_rows, physx_manifest = _run_physx_batch(root, output, geometries)
    records = [
        _build_link(
            root, output, link, geometries[link], table_masks[link],
            physx_rows.get(link), arguments.backend,
        )
        for link in _EXPECTED
    ]
    candidate = all(row["static_geometry_asset_candidate"] for row in records)
    manifest = {
        "schema_version": "carts_opposition_research_collision_asset_v2",
        "status": "STATIC_GEOMETRY_ASSET_CANDIDATE" if candidate else "STATIC_GEOMETRY_ASSET_REJECTED",
        "evidence_level": "STATIC_COMPOUND_CONVEX_AUDIT_ONLY",
        "static_geometry_asset_candidate": candidate, "runtime_binding_accepted": False,
        "runtime_binding_required_gates": ["RUNTIME_IMPORT", "INITIAL_PENETRATION",
                                           "OPPOSITION60_REPLAY", "PHYSX_HEALTH"],
        "runtime_binding_gate_status": "NOT_EVALUATED_BY_STATIC_BUILDER",
        "formal_collision_claimed": False, "dynamic_grasp_claimed": False,
        "formal_dynamic_pass": False, "hardware_authorized": False,
        "hand_variant": "CONNECTOR_GRASP_NO_NAIL",
        "source_hand_audit": str(_HAND_AUDIT),
        "source_hand_audit_sha256": file_sha256(audit_path),
        "task_surface_manifest": str(_TASK_SURFACES),
        "task_surface_manifest_sha256": file_sha256(surface_path),
        "builder": str(builder_path.relative_to(root)),
        "builder_sha256": executed_builder_sha256,
        "executed_source": {"script": str(builder_path.relative_to(root)), "sha256": executed_builder_sha256},
        "decomposition_backend": arguments.backend,
        "decomposition_parameters": _PHYSX if physx_rows else _COACD,
        "decomposition_parameter_attempt_count_per_link": 1,
        "physx_batch_manifest": physx_manifest,
        "versions": {
            name: importlib.metadata.version(name)
            for name in (("trimesh", "numpy", "matplotlib") if physx_rows else
                         ("coacd", "trimesh", "numpy", "matplotlib"))
        },
        "table_facing_reference": table_reference,
        "links": records,
        "limitations": ["Compound convex is a research PhysX collision representation only.",
                        "Exact nail-free triangles remain the offline and post-run truth representation.",
                        "Distance audit is sampled and is not a continuous collision certificate.",
                        "Runtime import, q_p=60 closure replay, and Isaac motion are not performed here."],
    }
    manifest_path = output / "RESEARCH_COLLISION_ASSET_MANIFEST.json"
    if file_sha256(builder_path) != executed_builder_sha256:
        raise RuntimeError("builder source changed during execution")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"manifest": str(manifest_path), "status": manifest["status"],
                      "links": [{"link": row["link"], "hulls": row["hull_count"],
                                  "deleted_exclusive_occupied": row["removed_nail_exclusive_audit"]["occupied_exclusive_sample_count"],
                                  "static_candidate": row["static_geometry_asset_candidate"]}
                                 for row in records]}, indent=2))
    return 0 if candidate else 2
if __name__ == "__main__":
    raise SystemExit(main())
