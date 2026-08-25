#!/usr/bin/env python3
"""Cook one bound three-link batch with PhysX VHACD shrink-wrap."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import trimesh

from isaacsim import SimulationApp


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _triangulate(convex) -> trimesh.Trimesh:
    vertices = np.asarray([[v.x, v.y, v.z] for v in convex.vertices], dtype=np.float64)
    triangles = []
    for polygon in convex.polygons:
        ring = list(convex.indices[polygon.index_base:polygon.index_base + polygon.num_vertices])
        triangles.extend((ring[0], ring[index], ring[index + 1]) for index in range(1, len(ring) - 1))
    mesh = trimesh.Trimesh(vertices=vertices, faces=np.asarray(triangles), process=True)
    if float(mesh.volume) < 0.0:
        mesh.invert()
    return mesh


def _author_source(stage, path: str, source: trimesh.Trimesh, parameters: dict):
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    geometry = UsdGeom.Mesh.Define(stage, path)
    geometry.CreatePointsAttr([Gf.Vec3f(*point) for point in source.vertices])
    geometry.CreateFaceVertexCountsAttr([3] * len(source.faces))
    geometry.CreateFaceVertexIndicesAttr(np.asarray(source.faces).reshape(-1).tolist())
    geometry.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    prim = geometry.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(
        UsdPhysics.Tokens.convexDecomposition
    )
    api = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
    api.CreateMaxConvexHullsAttr(parameters["max_convex_hulls"])
    api.CreateHullVertexLimitAttr(parameters["max_hull_vertices"])
    api.CreateVoxelResolutionAttr(parameters["voxel_resolution"])
    api.CreateErrorPercentageAttr(parameters["error_percentage"])
    api.CreateShrinkWrapAttr(parameters["shrink_wrap"])
    return prim, {
        "max_convex_hulls": int(api.GetMaxConvexHullsAttr().Get()),
        "max_hull_vertices": int(api.GetHullVertexLimitAttr().Get()),
        "voxel_resolution": int(api.GetVoxelResolutionAttr().Get()),
        "error_percentage": float(api.GetErrorPercentageAttr().Get()),
        "shrink_wrap": bool(api.GetShrinkWrapAttr().Get()),
    }


def _cook(stage, prim, output: Path, link: str) -> tuple[list[dict], float, str]:
    from omni.physx import get_physx_cooking_interface
    from omni.physx.bindings._physx import PhysxCollisionRepresentationResult
    from pxr import PhysicsSchemaTools, UsdUtils

    returned = {"result": None, "convexes": None}

    def receive(result, convexes):
        returned.update(result=result, convexes=convexes)

    started = time.perf_counter()
    get_physx_cooking_interface().request_convex_collision_representation(
        stage_id=UsdUtils.StageCache.Get().GetId(stage).ToLongInt(),
        collision_prim_id=PhysicsSchemaTools.sdfPathToInt(prim.GetPath()),
        run_asynchronously=False,
        on_result=receive,
    )
    elapsed = time.perf_counter() - started
    result_name = str(returned["result"])
    if returned["result"] != PhysxCollisionRepresentationResult.RESULT_VALID:
        raise RuntimeError(f"{link}: PhysX cooking failed: {result_name}")
    rows = []
    link_output = output / "physx_cooked" / link
    link_output.mkdir(parents=True, exist_ok=False)
    convexes = sorted(
        returned["convexes"],
        key=lambda item: (
            len(item.indices), len(item.vertices), len(item.polygons),
            item.vertices[0].x, item.vertices[0].y, item.vertices[0].z,
        ),
    )
    for index, convex in enumerate(convexes):
        mesh = _triangulate(convex)
        path = link_output / f"{link}_physx_vhacd_hull_{index:02d}.stl"
        mesh.export(path, file_type="stl")
        rows.append({
            "index": index, "path": str(path), "sha256": _sha256(path),
            "vertex_count": int(len(mesh.vertices)), "triangle_count": int(len(mesh.faces)),
            "reported_polygon_count": int(len(convex.polygons)),
        })
    return rows, elapsed, result_name


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    arguments = parser.parse_args()
    request = json.loads(arguments.request.resolve().read_text(encoding="utf-8"))
    output = Path(request["output_dir"]).resolve()
    executed_script = Path(__file__).resolve()
    executed_sha256 = _sha256(executed_script)
    if request.get("executed_source_chain", {}).get("runner_sha256") != executed_sha256:
        raise ValueError("runner source SHA is missing or does not match request")
    app = SimulationApp({
        "headless": True, "multi_gpu": False, "active_gpu": 0,
        "physics_gpu": 0, "fast_shutdown": True,
    })
    try:
        import omni.kit.app
        import omni.usd
        from pxr import UsdGeom

        manager = omni.kit.app.get_app().get_extension_manager()
        manager.set_extension_enabled_immediate("omni.physx", True)
        app.update()
        context = omni.usd.get_context()
        context.new_stage()
        app.update()
        stage = context.get_stage()
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        authored, sources = {}, {}
        for index, row in enumerate(request["links"]):
            path = Path(row["source_mesh"]).resolve()
            if _sha256(path) != row["source_mesh_sha256"]:
                raise ValueError(f"{row['link']}: source SHA changed")
            source = trimesh.load_mesh(path, force="mesh", process=False)
            if len(source.faces) != row["source_triangle_count"]:
                raise ValueError(f"{row['link']}: source triangle count changed")
            prim, readback = _author_source(
                stage, f"/World/NailFree_{index}_{row['link']}", source, request["parameters"]
            )
            if readback != request["parameters"]:
                raise RuntimeError(f"{row['link']}: PhysX parameter readback mismatch: {readback}")
            authored[row["link"]] = (prim, readback)
            sources[row["link"]] = row
        links = []
        for link, (prim, readback) in authored.items():
            hulls, elapsed, result = _cook(stage, prim, output, link)
            links.append({
                "link": link, "source_mesh_sha256": sources[link]["source_mesh_sha256"],
                "source_triangle_count": sources[link]["source_triangle_count"],
                "parameter_readback": readback, "cooking_result": result,
                "elapsed_s": elapsed, "hull_count": len(hulls), "hulls": hulls,
            })
        manifest = {
            "schema_version": "physx_vhacd_shrinkwrap_batch_v2",
            "status": "STATIC_DECOMPOSITION_OUTPUT_ONLY",
            "backend": "PHYSX_VHACD_SHRINK_WRAP", "batch_attempt_count": 1,
            "parameters": request["parameters"], "links": links,
            "python_version": platform.python_version(),
            "isaac_extension": "omni.physx", "hardware_authorized": False,
            "executed_source": {
                "script": str(executed_script), "sha256": executed_sha256,
            },
            "executed_source_chain_request": request["executed_source_chain"],
            "static_geometry_asset_candidate": False,
            "runtime_binding_accepted": False,
            "runtime_gate_evidence": "NOT_EVALUATED_BY_DECOMPOSITION_RUNNER",
        }
        if _sha256(executed_script) != executed_sha256:
            raise RuntimeError("runner source changed during execution")
        path = output / "PHYSX_VHACD_BATCH.json"
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"manifest": str(path), "links": [
            {"link": row["link"], "hulls": row["hull_count"], "elapsed_s": row["elapsed_s"]}
            for row in links
        ]}, indent=2))
        return 0
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
