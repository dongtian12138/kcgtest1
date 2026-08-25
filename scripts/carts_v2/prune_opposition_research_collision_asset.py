#!/usr/bin/env python3
"""Prune nail-occupying PhysX hulls and rerun the registered static audit."""
from __future__ import annotations
import argparse
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
    "physx_vhacd_pruned_nailfree"
)
_PARAMETERS = {
    "max_convex_hulls": 64,
    "max_hull_vertices": 64,
    "voxel_resolution": 1_000_000,
    "error_percentage": 0.5,
    "shrink_wrap": True,
}
_EXPECTED_PRUNE_COUNTS = {"f1Link3": 2, "f2Link2": 1, "f3Link3": 3}
_P95_LIMIT_M = 0.002
_P95_DEGRADATION_LIMIT_M = 0.00025
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
def _normalized_retained_hulls(row: dict) -> tuple[list, list[int], list[int], float]:
    occupancy = row["removed_nail_exclusive_audit"]["occupied_sample_count_by_hull"]
    pruned = [index for index, count in enumerate(occupancy) if int(count) > 0]
    retained = [index for index in range(len(row["hull_files"])) if index not in pruned]
    started = time.perf_counter()
    hulls = []
    for index in retained:
        source = row["hull_files"][index]
        if file_sha256(Path(source["path"])) != source["sha256"]:
            raise ValueError(f"{row['link']}: baseline hull {index} SHA changed")
        hull = trimesh.load_mesh(source["path"], force="mesh", process=True).convex_hull
        if float(hull.volume) < 0.0:
            hull.invert()
        hulls.append(hull)
    return hulls, pruned, retained, time.perf_counter() - started
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
    arguments = parser.parse_args()
    root = arguments.repository_root.resolve()
    baseline_path, output = root / _BASELINE, root / _OUTPUT
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
    output.mkdir(parents=True)
    audit_path = root / baseline["source_hand_audit"]
    surface_path = root / baseline["task_surface_manifest"]
    audit, surfaces = _json(audit_path), _json(surface_path)
    audit_by_link = {Path(row["source_path"]).stem: row for row in audit["links"]}
    surface_by_link = {row["link_name"]: row for row in surfaces["links"]}
    baseline_by_link = {row["link"]: row for row in baseline["links"]}
    geometries = {
        link: load_bound_geometry(
            root, link, audit_by_link[link], surface_by_link[link]
        )
        for link in baseline_by_link
    }
    table_masks, table_reference = qp60_table_masks(root, geometries)
    records = []
    for link, baseline_row in baseline_by_link.items():
        hulls, pruned, retained, elapsed = _normalized_retained_hulls(baseline_row)
        result = audit_compound_link(
            root, output, link, geometries[link], table_masks[link],
            (hulls, elapsed), "PHYSX_VHACD_PRUNED_NAILFREE",
        )
        metrics, metric_pass = _metric_gates(baseline_row, result)
        hard_pass = bool(
            result["hull_validation_pass"]
            and result["removed_nail_exclusive_audit"]["pass"]
        )
        expectation_match = len(pruned) == _EXPECTED_PRUNE_COUNTS[link]
        candidate = hard_pass and metric_pass and expectation_match
        result.update({
            "source_baseline_hull_count": len(baseline_row["hull_files"]),
            "pruned_source_hull_indices": pruned,
            "retained_source_hull_indices": retained,
            "expected_pruned_hull_count": _EXPECTED_PRUNE_COUNTS[link],
            "pruned_hull_count_expectation_match": expectation_match,
            "normalization": "TRIMESH_CONVEX_HULL_PER_RETAINED_SOURCE_HULL",
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
        "schema_version": "carts_physx_vhacd_pruned_nailfree_v2",
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
        "source_baseline_manifest": str(_BASELINE),
        "source_baseline_manifest_sha256": file_sha256(baseline_path),
        "postprocessor": str(Path(__file__).resolve().relative_to(root)),
        "postprocessor_sha256": file_sha256(Path(__file__).resolve()),
        "executed_source": {"script": str(Path(__file__).resolve().relative_to(root)),
                            "sha256": file_sha256(Path(__file__).resolve())},
        "selection_rule": "PRUNE_EVERY_BASELINE_HULL_WITH_REMOVED_EXCLUSIVE_OCCUPANCY_GT_ZERO",
        "normalization_rule": "TRIMESH_CONVEX_HULL_ON_EACH_RETAINED_HULL",
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
    manifest_path = output / "PRUNED_NAILFREE_COLLISION_ASSET_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "manifest": str(manifest_path), "status": manifest["status"],
        "links": [{
            "link": row["link"], "pruned": row["pruned_source_hull_indices"],
            "retained": row["hull_count"],
            "deleted_exclusive_occupied": row["removed_nail_exclusive_audit"]["occupied_exclusive_sample_count"],
            "static_candidate": row["static_geometry_asset_candidate"],
        } for row in records],
    }, indent=2))
    return 0 if candidate else 2
if __name__ == "__main__":
    raise SystemExit(main())
