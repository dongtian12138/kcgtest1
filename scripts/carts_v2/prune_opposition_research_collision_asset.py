#!/usr/bin/env python3
"""Prune nail-occupying PhysX hulls and rerun the registered static audit."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
from pathlib import Path
import time
import numpy as np
import trimesh
from build_opposition_research_collision_asset import (
    audit_compound_link,
    load_bound_geometry,
    qp60_table_masks,
)
from kcg_connector.grasp.robust.object_model import file_sha256
_BASELINE = Path(
    "artifacts/carts_v2/opposition60_isaac/research_collision_asset/"
    "physx_vhacd_shrinkwrap/RESEARCH_COLLISION_ASSET_MANIFEST.json"
)
_OUTPUT = Path(
    "artifacts/carts_v2/opposition60_isaac/research_collision_asset/"
    "physx_vhacd_residual_repaired_nailfree"
)
_PARAMETERS = {
    "max_convex_hulls": 64,
    "max_hull_vertices": 64,
    "voxel_resolution": 1_000_000,
    "error_percentage": 0.5,
    "shrink_wrap": True,
}
_EXPECTED_PRUNED_HULL_INDICES = {
    "f1Link3": [24, 31],
    "f2Link2": [49],
    "f3Link3": [39, 56],
}
_EXPECTED_PRUNE_COUNTS = {
    link: len(indices) for link, indices in _EXPECTED_PRUNED_HULL_INDICES.items()
}
_P95_LIMIT_M = 0.002
_P95_DEGRADATION_LIMIT_M = 0.00025
_BOUNDARY_OCCUPANCY_TOLERANCE_M = 1.0e-9
_MAX_RESIDUAL_HULL_FACES = 2 * _PARAMETERS["max_hull_vertices"] - 4
_REGIONS = ("global", "task_grip_surface", "table_facing_at_qp60")
def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
def _verify_baseline_source_chain(root: Path, baseline: dict) -> None:
    builder = baseline.get("executed_source")
    batch = baseline.get("physx_batch_manifest") or {}
    runner = batch.get("executed_source")
    chain = batch.get("executed_source_chain_request") or {}
    if not builder or not runner:
        raise ValueError("EXECUTED_SOURCE_SNAPSHOT_UNAVAILABLE")
    for binding in (builder, runner):
        path = Path(binding["script"])
        path = path if path.is_absolute() else root / path
        if not path.is_file() or file_sha256(path) != binding.get("sha256"):
            raise ValueError("executed source SHA is unavailable or changed")
    if (chain.get("builder_sha256") != builder["sha256"] or
            chain.get("runner_sha256") != runner["sha256"]):
        raise ValueError("baseline executed source chain is inconsistent")
def _pruned_indices(row: dict) -> list[int]:
    occupancy = row["removed_nail_exclusive_audit"]["occupied_sample_count_by_hull"]
    return [index for index, count in enumerate(occupancy) if int(count) > 0]


def _verify_prune_identity(
    rows: dict[str, dict]
) -> tuple[dict[str, int], dict[str, list[int]]]:
    observed_indices = {link: _pruned_indices(row) for link, row in rows.items()}
    if observed_indices != _EXPECTED_PRUNED_HULL_INDICES:
        raise ValueError(
            "fresh baseline occupied hull identity differs from replicated baseline: "
            f"{observed_indices}")
    observed_counts = {
        link: len(indices) for link, indices in observed_indices.items()
    }
    return observed_counts, observed_indices


def _safe_residual_patches(raw, hulls, pruned, exclusive) -> tuple[list, list[dict]]:
    compound = trimesh.util.concatenate(hulls)
    owners = np.concatenate([
        np.full(len(hull.faces), index, dtype=np.int64)
        for index, hull in enumerate(hulls)
    ])
    _, _, nearest = trimesh.proximity.closest_point(
        compound, raw.triangles_center)
    face_owner = owners[np.asarray(nearest, dtype=np.int64)]
    adjacency = np.asarray(raw.face_adjacency, dtype=np.int64)
    patches, records = [], []
    for source_hull in pruned:
        faces = np.flatnonzero(face_owner == source_hull)
        selected = np.zeros(len(raw.faces), dtype=np.bool_)
        selected[faces] = True
        edges = adjacency[np.all(selected[adjacency], axis=1)]
        components = trimesh.graph.connected_components(edges, nodes=faces)
        eligible = []
        for component in components:
            component = np.sort(np.asarray(component, dtype=np.int64))
            points = np.unique(raw.triangles[component].reshape(-1, 3), axis=0)
            if len(points) < 4 or np.linalg.matrix_rank(points - points[0]) < 3:
                continue
            patch = trimesh.convex.convex_hull(points)
            source_vertex_count = int(len(patch.vertices))
            source_face_count = int(len(patch.faces))
            simplified = source_vertex_count > _PARAMETERS["max_hull_vertices"]
            if simplified:
                patch = patch.simplify_quadric_decimation(
                    face_count=_MAX_RESIDUAL_HULL_FACES
                ).convex_hull
            valid = bool(
                patch.is_convex and patch.is_watertight
                and np.all(np.isfinite(patch.vertices)) and patch.volume > 0.0
                and len(patch.vertices) <= _PARAMETERS["max_hull_vertices"])
            if not valid:
                continue
            inside = patch.contains(exclusive)
            _, distance, _ = trimesh.proximity.closest_point(patch, exclusive)
            if np.any(inside | (distance <= _BOUNDARY_OCCUPANCY_TOLERANCE_M)):
                continue
            area = float(np.sum(raw.area_faces[component]))
            eligible.append((
                area, -int(component[0]), component, patch,
                source_vertex_count, source_face_count, simplified,
            ))
        if not eligible:
            raise ValueError(f"no safe connected residual patch for hull {source_hull}")
        (area, _tie_break, component, patch, source_vertex_count,
         source_face_count, simplified) = max(
             eligible, key=lambda item: item[:2])
        patches.append(patch)
        records.append({
            "source_hull_index": int(source_hull),
            "connected_component_count": len(components),
            "eligible_component_count": len(eligible),
            "selected_face_count": int(len(component)),
            "selected_face_area_m2": area,
            "minimum_selected_face_index": int(component[0]),
            "maximum_selected_face_index": int(component[-1]),
            "selection": "MAXIMUM_RETAINED_RAW_FACE_AREA_THEN_MINIMUM_FACE_INDEX",
            "source_convex_vertex_count": source_vertex_count,
            "source_convex_triangle_count": source_face_count,
            "runtime_convex_vertex_count": int(len(patch.vertices)),
            "runtime_convex_triangle_count": int(len(patch.faces)),
            "vertex_limit": _PARAMETERS["max_hull_vertices"],
            "vertex_limit_simplification_applied": simplified,
            "vertex_limit_simplification": "TRIMESH_QUADRIC_FACE124_THEN_CONVEX_HULL",
            "removed_exclusive_occupancy_count": 0,
        })
    return patches, records


def _normalized_repaired_hulls(
    row: dict, geometry: dict
) -> tuple[list, list[int], list[int], list[dict], float]:
    pruned = _pruned_indices(row)
    retained = [index for index in range(len(row["hull_files"])) if index not in pruned]
    started = time.perf_counter()
    baseline_hulls = []
    for index, source in enumerate(row["hull_files"]):
        if file_sha256(Path(source["path"])) != source["sha256"]:
            raise ValueError(f"{row['link']}: baseline hull {index} SHA changed")
        hull = trimesh.load_mesh(source["path"], force="mesh", process=True).convex_hull
        if float(hull.volume) < 0.0:
            hull.invert()
        baseline_hulls.append(hull)
    exclusive_record = row["removed_nail_exclusive_audit"]
    sample_path = Path(exclusive_record["sample_npz"])
    if (exclusive_record["boundary_occupancy_tolerance_m"]
            != _BOUNDARY_OCCUPANCY_TOLERANCE_M
            or file_sha256(sample_path) != exclusive_record["sample_npz_sha256"]):
        raise ValueError(f"{row['link']}: removed-exclusive evidence changed")
    with np.load(sample_path, allow_pickle=False) as archive:
        exclusive = np.asarray(archive["points_local_m"], dtype=np.float64)
    patches, repairs = _safe_residual_patches(
        trimesh.Trimesh(vertices=geometry["raw"].vertices_m,
                        faces=geometry["raw"].faces, process=False),
        baseline_hulls, pruned, exclusive)
    hulls = [baseline_hulls[index] for index in retained] + patches
    return hulls, pruned, retained, repairs, time.perf_counter() - started
def _metric_gates(baseline: dict, result: dict) -> tuple[dict, bool]:
    rows = {}
    for region in _REGIONS:
        before = baseline["distance_audit"][region]["raw_to_compound"]["p95_m"]
        after = result["distance_audit"][region]["raw_to_compound"]["p95_m"]
        rows[region] = {
            "baseline_p95_m": before,
            "pruned_p95_m": after,
            "degradation_m": after - before,
            "absolute_limit_m": _P95_LIMIT_M,
            "degradation_limit_m": _P95_DEGRADATION_LIMIT_M,
            "pass": bool(
                after <= _P95_LIMIT_M
                and after - before <= _P95_DEGRADATION_LIMIT_M
            ),
        }
    return rows, all(row["pass"] for row in rows.values())
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--baseline", type=Path, default=_BASELINE)
    parser.add_argument("--output", type=Path, default=_OUTPUT)
    arguments = parser.parse_args()
    root = arguments.repository_root.resolve()
    baseline_path = (arguments.baseline if arguments.baseline.is_absolute()
                     else root / arguments.baseline)
    output = arguments.output if arguments.output.is_absolute() else root / arguments.output
    if output.exists():
        raise FileExistsError(f"refusing to overwrite one-shot output: {output}")
    baseline = _json(baseline_path)
    _verify_baseline_source_chain(root, baseline)
    if (
        baseline["decomposition_backend"] != "PHYSX_VHACD_SHRINK_WRAP"
        or baseline["decomposition_parameters"] != _PARAMETERS
        or baseline["hardware_authorized"] is not False
        or len(baseline["links"]) != 3
    ):
        raise ValueError("fixed PhysX shrink-wrap baseline identity changed")
    baseline_by_link = {row["link"]: row for row in baseline["links"]}
    observed_prune_counts, observed_prune_indices = _verify_prune_identity(
        baseline_by_link)
    output.mkdir(parents=True)
    audit_path = root / baseline["source_hand_audit"]
    surface_path = root / baseline["task_surface_manifest"]
    audit, surfaces = _json(audit_path), _json(surface_path)
    audit_by_link = {Path(row["source_path"]).stem: row for row in audit["links"]}
    surface_by_link = {row["link_name"]: row for row in surfaces["links"]}
    geometries = {
        link: load_bound_geometry(
            root, link, audit_by_link[link], surface_by_link[link]
        )
        for link in baseline_by_link
    }
    table_masks, table_reference = qp60_table_masks(root, geometries)
    records = []
    for link, baseline_row in baseline_by_link.items():
        hulls, pruned, retained, repairs, elapsed = _normalized_repaired_hulls(
            baseline_row, geometries[link])
        result = audit_compound_link(
            root, output, link, geometries[link], table_masks[link],
            (hulls, elapsed), "PHYSX_VHACD_RESIDUAL_REPAIRED_NAILFREE",
        )
        metrics, metric_pass = _metric_gates(baseline_row, result)
        count_pass = bool(
            len(pruned) == len(repairs)
            and result["hull_count"] == len(baseline_row["hull_files"]))
        hard_pass = bool(
            count_pass
            and result["hull_validation_pass"]
            and result["removed_nail_exclusive_audit"]["pass"]
        )
        expectation_match = len(pruned) == _EXPECTED_PRUNE_COUNTS[link]
        candidate = hard_pass and metric_pass and expectation_match
        result.update({
            "source_baseline_hull_count": len(baseline_row["hull_files"]),
            "pruned_source_hull_indices": pruned,
            "retained_source_hull_indices": retained,
            "residual_repairs": repairs,
            "replacement_hull_count": len(repairs),
            "delete_add_count_match": count_pass,
            "expected_pruned_hull_count": _EXPECTED_PRUNE_COUNTS[link],
            "pruned_hull_count_expectation_match": expectation_match,
            "normalization": "TRIMESH_CONVEX_HULL_PLUS_MAX_AREA_SAFE_RESIDUAL_V1",
            "registered_p95_gates": metrics,
            "static_geometry_asset_candidate": candidate,
            "research_runtime_asset_gate_pass": False,
            "runtime_binding_accepted": False,
            "status": (
                "STATIC_GEOMETRY_ASSET_CANDIDATE"
                if candidate else "STATIC_GEOMETRY_ASSET_REJECTED"
            ),
        })
        records.append(result)
    candidate = all(row["static_geometry_asset_candidate"] for row in records)
    artifact_hashes = {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(output.rglob("*")) if path.is_file()
    }
    manifest = {
        "schema_version": "carts_physx_vhacd_residual_repaired_nailfree_v1",
        "status": (
            "STATIC_GEOMETRY_ASSET_CANDIDATE"
            if candidate else "STATIC_GEOMETRY_ASSET_REJECTED"
        ),
        "evidence_level": "STATIC_PRUNED_COMPOUND_CONVEX_RESEARCH_AUDIT_ONLY",
        "static_geometry_asset_candidate": candidate,
        "runtime_binding_accepted": False,
        "runtime_binding_required_gates": ["RUNTIME_IMPORT", "INITIAL_PENETRATION",
                                           "OPPOSITION60_REPLAY", "PHYSX_HEALTH"],
        "runtime_binding_gate_status": "NOT_EVALUATED_BY_STATIC_PRUNER",
        "formal_collision_claimed": False,
        "formal_dynamic_pass": False,
        "hardware_authorized": False,
        "source_baseline_manifest": str(baseline_path),
        "source_baseline_manifest_sha256": file_sha256(baseline_path),
        "registered_pruned_hull_counts": _EXPECTED_PRUNE_COUNTS,
        "observed_pruned_hull_counts": observed_prune_counts,
        "registered_pruned_hull_indices": _EXPECTED_PRUNED_HULL_INDICES,
        "observed_pruned_hull_indices": observed_prune_indices,
        "postprocessor": str(Path(__file__).resolve().relative_to(root)),
        "postprocessor_sha256": file_sha256(Path(__file__).resolve()),
        "executed_source": {"script": str(Path(__file__).resolve().relative_to(root)),
                            "sha256": file_sha256(Path(__file__).resolve())},
        "selection_rule": "DELETE_OCCUPYING_HULL_AND_ADD_ONE_MAX_AREA_SAFE_CONNECTED_RAW_RESIDUAL",
        "normalization_rule": "TRIMESH_CONVEX_HULL_ON_RETAINED_HULLS_AND_RESIDUAL_PATCHES",
        "fast_simplification_version": importlib.metadata.version("fast-simplification"),
        "registered_research_limits": {
            "raw_to_compound_p95_absolute_m": _P95_LIMIT_M,
            "raw_to_compound_p95_degradation_m": _P95_DEGRADATION_LIMIT_M,
            "classification": "PRE_REGISTERED_SIMULATION_RESEARCH_ASSET_LIMIT",
            "hardware_accuracy_claimed": False,
        },
        "table_facing_reference": table_reference,
        "links": records,
        "artifact_sha256_before_manifest": artifact_hashes,
    }
    manifest_path = output / "RESIDUAL_REPAIRED_COLLISION_ASSET_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "manifest": str(manifest_path), "status": manifest["status"],
        "links": [{
            "link": row["link"], "pruned": row["pruned_source_hull_indices"],
            "retained": row["hull_count"],
            "replacements": row["replacement_hull_count"],
            "deleted_exclusive_occupied": row["removed_nail_exclusive_audit"]["occupied_exclusive_sample_count"],
            "static_candidate": row["static_geometry_asset_candidate"],
        } for row in records],
    }, indent=2))
    return 0 if candidate else 2
if __name__ == "__main__":
    raise SystemExit(main())
