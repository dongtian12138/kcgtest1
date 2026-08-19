#!/usr/bin/env python3

"""Probe installed PhysX convex-hull cooking without touching project assets.

The probe authors the same thin box three ways: with the current implicit
convex-hull defaults, with an explicit 1 mm minimum thickness, and with an
explicit zero minimum thickness.  It reports authored and cooked bounds only;
it writes no artifact and computes no file fingerprint.
"""

from __future__ import annotations

import json
import os
import traceback
from typing import Any


def _emit(value: Any) -> None:
    os.write(1, (json.dumps(value, allow_nan=False, sort_keys=True) + "\n").encode())


def _bounds(points: list[tuple[float, float, float]]) -> dict[str, list[float]]:
    return {
        "minimum": [min(point[axis] for point in points) for axis in range(3)],
        "maximum": [max(point[axis] for point in points) for axis in range(3)],
        "extent": [
            max(point[axis] for point in points)
            - min(point[axis] for point in points)
            for axis in range(3)
        ],
    }


def _runtime_probe(
    authored_points: list[tuple[float, float, float]],
    counts: list[int],
    indices: list[int],
) -> dict[str, Any]:
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.prims import RigidPrim
    from isaacsim.core.utils.stage import get_current_stage
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    World.clear_instance()
    omni.usd.get_context().new_stage()
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=1.0 / 240.0,
        rendering_dt=1.0 / 60.0,
        backend="numpy",
        device="cpu",
    )
    stage = get_current_stage()

    def add_mesh(
        path: str,
        *,
        center: tuple[float, float, float],
        point_scale: float,
        xform_scale: float,
        dynamic: bool,
    ) -> None:
        owner = UsdGeom.Xform.Define(stage, path)
        owner.AddTranslateOp().Set(Gf.Vec3d(*center))
        if dynamic:
            UsdPhysics.RigidBodyAPI.Apply(owner.GetPrim()).CreateRigidBodyEnabledAttr(True)
            UsdPhysics.MassAPI.Apply(owner.GetPrim()).CreateMassAttr(1.0)
        mesh = UsdGeom.Mesh.Define(stage, path + "/Collision")
        mesh.CreatePointsAttr(
            [
                Gf.Vec3f(*(component * point_scale for component in point))
                for point in authored_points
            ]
        )
        mesh.CreateFaceVertexCountsAttr(counts)
        mesh.CreateFaceVertexIndicesAttr(indices)
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        if xform_scale != 1.0:
            mesh.AddScaleOp().Set(Gf.Vec3f(xform_scale))
        prim = mesh.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(
            UsdPhysics.Tokens.convexHull
        )
        PhysxSchema.PhysxConvexHullCollisionAPI.Apply(
            prim
        ).CreateMinThicknessAttr(0.001)
        physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        physx_collision.CreateContactOffsetAttr(1.0e-5)
        physx_collision.CreateRestOffsetAttr(0.0)

    # Each authored pair has a 0.40 mm surface gap.  The implicit metre-local
    # convex expansion closes that gap; the millimetre-local scaled hull must
    # remain separated.
    cases = {
        "metre_local": {"x": 0.0, "point_scale": 1.0, "xform_scale": 1.0},
        "millimetre_local_scaled": {
            "x": 0.03,
            "point_scale": 1000.0,
            "xform_scale": 0.001,
        },
    }
    views: dict[str, RigidPrim] = {}
    for name, parameters in cases.items():
        root = "/Runtime/" + name
        x = float(parameters["x"])
        add_mesh(
            root + "/Bottom",
            center=(x, 0.0, 0.0),
            point_scale=float(parameters["point_scale"]),
            xform_scale=float(parameters["xform_scale"]),
            dynamic=False,
        )
        add_mesh(
            root + "/Top",
            center=(x, 0.0, 0.0007),
            point_scale=float(parameters["point_scale"]),
            xform_scale=float(parameters["xform_scale"]),
            dynamic=True,
        )
        views[name] = RigidPrim(
            prim_paths_expr=root + "/Top",
            name="probe_" + name,
            reset_xform_properties=False,
        )

    world.get_physics_context().set_gravity(0.0)
    world.reset()
    initial: dict[str, list[float]] = {}
    for name, view in views.items():
        view.initialize()
        position, _orientation = view.get_world_poses()
        initial[name] = [float(value) for value in position[0]]
    for _step in range(30):
        world.step(render=False)
    result: dict[str, Any] = {}
    for name, view in views.items():
        position, _orientation = view.get_world_poses()
        velocity = view.get_velocities()
        final = [float(value) for value in position[0]]
        result[name] = {
            "initial_position_m": initial[name],
            "final_position_m": final,
            "delta_position_m": [
                final[axis] - initial[name][axis] for axis in range(3)
            ],
            "final_velocity": [float(value) for value in velocity[0]],
        }
    world.stop()
    World.clear_instance()
    return result


def _run() -> dict[str, Any]:
    from omni.physx import get_physx_cooking_interface
    from omni.physx.bindings._physx import PhysxCollisionRepresentationResult
    from pxr import Gf, PhysicsSchemaTools, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdUtils

    stage = Usd.Stage.CreateInMemory("physx_convex_min_thickness_probe.usda")
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    cache = UsdUtils.StageCache.Get()
    cache_id = cache.Insert(stage)
    stage_id = cache_id.ToLongInt()
    authored_points = [
        (-0.005, -0.005, -0.00015),
        (+0.005, -0.005, -0.00015),
        (+0.005, +0.005, -0.00015),
        (-0.005, +0.005, -0.00015),
        (-0.005, -0.005, +0.00015),
        (+0.005, -0.005, +0.00015),
        (+0.005, +0.005, +0.00015),
        (-0.005, +0.005, +0.00015),
    ]
    counts = [4, 4, 4, 4, 4, 4]
    indices = [
        3, 2, 1, 0,
        4, 5, 6, 7,
        0, 1, 5, 4,
        1, 2, 6, 5,
        2, 3, 7, 6,
        3, 0, 4, 7,
    ]
    cases = {
        "implicit_default": {
            "minimum_thickness": None,
            "point_scale": 1.0,
            "xform_scale": 1.0,
        },
        "explicit_one_millimetre": {
            "minimum_thickness": 0.001,
            "point_scale": 1.0,
            "xform_scale": 1.0,
        },
        "explicit_zero": {
            "minimum_thickness": 0.0,
            "point_scale": 1.0,
            "xform_scale": 1.0,
        },
        "millimetre_local_mesh_scale": {
            "minimum_thickness": 0.001,
            "point_scale": 1000.0,
            "xform_scale": 0.001,
        },
    }
    results: dict[str, Any] = {}
    cooking = get_physx_cooking_interface()
    for name, parameters in cases.items():
        cooking.release_local_mesh_cache()
        minimum_thickness = parameters["minimum_thickness"]
        point_scale = float(parameters["point_scale"])
        xform_scale = float(parameters["xform_scale"])
        path = f"/Probe/{name}"
        mesh = UsdGeom.Mesh.Define(stage, path)
        local_points = [
            tuple(component * point_scale for component in point)
            for point in authored_points
        ]
        mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in local_points])
        mesh.CreateFaceVertexCountsAttr(counts)
        mesh.CreateFaceVertexIndicesAttr(indices)
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        if xform_scale != 1.0:
            mesh.AddScaleOp().Set(Gf.Vec3f(xform_scale))
        prim = mesh.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(
            UsdPhysics.Tokens.convexHull
        )
        if minimum_thickness is not None:
            PhysxSchema.PhysxConvexHullCollisionAPI.Apply(
                prim
            ).CreateMinThicknessAttr(minimum_thickness)

        callback: dict[str, Any] = {}

        def on_result(result: Any, convexes: list[Any]) -> None:
            callback["result"] = result
            callback["convexes"] = convexes

        cooking.request_convex_collision_representation(
            stage_id=stage_id,
            collision_prim_id=PhysicsSchemaTools.sdfPathToInt(path),
            run_asynchronously=False,
            on_result=on_result,
        )
        if callback.get("result") != PhysxCollisionRepresentationResult.RESULT_VALID:
            raise RuntimeError(f"convex cooking failed for {name}: {callback.get('result')}")
        convexes = callback["convexes"]
        if len(convexes) != 1:
            raise RuntimeError(f"unexpected convex count for {name}: {len(convexes)}")
        cooked_points = [
            (float(vertex.x), float(vertex.y), float(vertex.z))
            for vertex in convexes[0].vertices
        ]
        results[name] = {
            "authored_min_thickness": minimum_thickness,
            "applied_extended_api": minimum_thickness is not None,
            "authored_local_bounds": _bounds(local_points),
            "authored_xform_scale": xform_scale,
            "cooked_vertex_count": len(cooked_points),
            "cooked_polygon_count": len(convexes[0].polygons),
            "cooked_bounds_m": _bounds(cooked_points),
        }
    return {
        "status": "PASSED",
        "stage_meters_per_unit": 1.0,
        "authored_bounds_m": _bounds(authored_points),
        "cases": results,
        "runtime": _runtime_probe(authored_points, counts, indices),
    }


def main() -> int:
    from isaacsim import SimulationApp

    application = SimulationApp(
        {
            "headless": True,
            "hide_ui": True,
            "renderer": "Minimal",
            "multi_gpu": False,
            "fast_shutdown": True,
            "enable_crashreporter": False,
        }
    )
    status = 1
    try:
        _emit(_run())
        status = 0
    except BaseException as error:
        traceback.print_exc()
        _emit(
            {
                "status": "FAILED",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
    finally:
        application.close()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
