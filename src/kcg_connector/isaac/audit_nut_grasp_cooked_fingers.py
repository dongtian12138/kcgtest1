#!/usr/bin/env python3
"""Inspect PhysX-cooked fingertip shapes at a previously recorded stop point."""

from pathlib import Path
import argparse
import json
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--max-convex-hulls", type=int)
parser.add_argument("--output-name", default="cooked_finger_inspection")
args = parser.parse_args()

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "multi_gpu": False, "fast_shutdown": True})

from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils, PhysicsSchemaTools, PhysxSchema
from scipy.spatial import ConvexHull
from omni.physx import get_physx_cooking_interface

repo = Path(__file__).resolve().parents[3]
root = repo / "artifacts/kcg_connector/isaac/te_full_assembly_20260905/nut_regrasp_01"
output = root / args.output_name
output.mkdir(parents=True, exist_ok=False)
previous = json.loads((root / "finger_collision_representation_posthoc.json").read_text())["rows"]
asset = repo / "artifacts/kcg_connector/isaac/te_nail_tip_body_grasp_v1/handarm_original_nails_source_decomposition.usda"
stage = Usd.Stage.Open(str(asset))
stage_id = UsdUtils.StageCache.Get().Insert(stage).ToLongInt()
cache = UsdGeom.XformCache()
cooking = get_physx_cooking_interface()
records = []
for item in previous:
    name = item["link"]
    link = next(p for p in stage.Traverse() if p.GetName() == name)
    link_world = np.asarray(cache.GetLocalToWorldTransform(link)).T
    point = np.asarray(item["witness_in_link_m"])
    for prim in Usd.PrimRange(link):
        if not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            continue
        if args.max_convex_hulls is not None:
            PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim).CreateMaxConvexHullsAttr(args.max_convex_hulls)
        mesh_to_link = np.linalg.inv(link_world) @ np.asarray(cache.GetLocalToWorldTransform(prim)).T
        received = {}

        def on_result(result, convexes):
            received["result"] = str(result)
            received["convexes"] = []
            for index, convex in enumerate(convexes):
                vertices = np.asarray([list(p) for p in convex.vertices], dtype=np.float64)
                local = vertices @ mesh_to_link[:3, :3].T + mesh_to_link[:3, 3]
                hull = ConvexHull(local)
                violation = float(np.max(hull.equations[:, :3] @ point + hull.equations[:, 3]))
                path = output / f"{name}_convex_{index:03d}.npz"
                np.savez_compressed(path, vertices_link_m=local, faces=hull.simplices)
                received["convexes"].append({"index": index, "mesh_path": str(path),
                    "vertex_count": len(local), "bbox_link_m": [local.min(axis=0).tolist(), local.max(axis=0).tolist()],
                    "witness_plane_violation_m": violation})

        cooking.request_convex_collision_representation(
            stage_id, PhysicsSchemaTools.sdfPathToInt(prim.GetPath()), False, on_result)
        if not received or not received.get("convexes"):
            raise RuntimeError(f"cooked shape unavailable for {name}: {received}")
        record = {"link": name, "prim_path": str(prim.GetPath()),
                  "approximation": UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get(), **received}
        record["minimum_witness_plane_violation_m"] = min(c["witness_plane_violation_m"] for c in record["convexes"])
        records.append(record)
        print("COOKED_FINGER", name, len(record["convexes"]), record["minimum_witness_plane_violation_m"], flush=True)
result = {"scope": "OFFLINE_AUTHORED_COLLISION_COOKING_AND_POSTRUN_WITNESS_QUERY",
          "trial_max_convex_hulls": args.max_convex_hulls,
          "robot_motion_commands": 0, "physics_time_advanced_s": 0.0,
          "source_asset_modified": False, "source_asset": str(asset), "records": records}
(output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
app.close()
