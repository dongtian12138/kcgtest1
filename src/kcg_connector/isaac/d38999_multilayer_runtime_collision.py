#!/usr/bin/env python3

"""Deterministic runtime collision representation for the plug guide ring.

The frozen USD keeps the visible single annular mesh.  PhysX cannot use that
concave mesh directly on a dynamic body, while automatic convex decomposition
introduced false penetration.  This module replaces only that collider on the
in-memory stage with 64 exact convex sectors authored in millimetre-local
coordinates and scaled back to metres.  It never writes the source asset.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


ROOT = "/World/D38999MultilayerV1/D38999_ASSEMBLY_CONTROL_V1"
BODY_PATH = ROOT + "/D38999Pair/LoosePlug/BodyAssembly"
CONTINUOUS_PLUG_GUIDE_PATH = BODY_PATH + "/MatingShell/ContinuousPlugGuide"
PROXY_ROOT_PATH = BODY_PATH + "/MatingShell/ContinuousPlugGuideRuntimeConvexSegments"
SEGMENT_COUNT = 64
LOCAL_TO_STAGE_SCALE = 0.001
FACE_VERTEX_COUNTS = (4, 4, 4, 4, 4, 4)
FACE_VERTEX_INDICES = (
    0, 3, 2, 1,
    4, 5, 6, 7,
    0, 1, 5, 4,
    1, 2, 6, 5,
    2, 3, 7, 6,
    3, 0, 4, 7,
)


def annular_convex_segment_specs(
    *,
    inner_radius_m: float,
    outer_radius_m: float,
    z0_m: float,
    z1_m: float,
    segment_count: int = SEGMENT_COUNT,
) -> list[dict[str, Any]]:
    """Return exact convex sectors of the polygonal annulus, without USD."""

    if not (
        0.0 < inner_radius_m < outer_radius_m
        and z0_m < z1_m
        and segment_count >= 8
    ):
        raise ValueError("invalid annular convex-segment dimensions")
    rows: list[dict[str, Any]] = []
    for index in range(segment_count):
        angle0 = 2.0 * math.pi * index / segment_count
        angle1 = 2.0 * math.pi * (index + 1) / segment_count
        c0, s0 = math.cos(angle0), math.sin(angle0)
        c1, s1 = math.cos(angle1), math.sin(angle1)
        points_m = (
            (inner_radius_m * c0, inner_radius_m * s0, z0_m),
            (outer_radius_m * c0, outer_radius_m * s0, z0_m),
            (outer_radius_m * c1, outer_radius_m * s1, z0_m),
            (inner_radius_m * c1, inner_radius_m * s1, z0_m),
            (inner_radius_m * c0, inner_radius_m * s0, z1_m),
            (outer_radius_m * c0, outer_radius_m * s0, z1_m),
            (outer_radius_m * c1, outer_radius_m * s1, z1_m),
            (inner_radius_m * c1, inner_radius_m * s1, z1_m),
        )
        rows.append(
            {
                "index": index,
                "points_m": points_m,
                "points_local_mm": tuple(
                    tuple(component / LOCAL_TO_STAGE_SCALE for component in point)
                    for point in points_m
                ),
                "face_vertex_counts": FACE_VERTEX_COUNTS,
                "face_vertex_indices": FACE_VERTEX_INDICES,
                "xform_scale": LOCAL_TO_STAGE_SCALE,
            }
        )
    return rows


def _contract_dimensions(contract: Mapping[str, Any]) -> dict[str, float]:
    geometry = contract["keying"]["nominal_collision_geometry"]
    stop = contract["metal_stop"]
    inner, outer = [float(value) for value in geometry["plug_shell_radial_interval_m"]]
    return {
        "inner_radius_m": inner,
        "outer_radius_m": outer,
        "z0_m": 0.0,
        "z1_m": float(stop["nominal_bottoming_separation_m"]),
        "fixed_clear_bore_radius_m": float(geometry["receptacle_clear_bore_radius_m"]),
        "fixed_stop_outer_radius_m": float(
            stop["assembly_control_collision"]["fixed_cap_radius_m"]
        ),
    }


def _realized_bounds(points: Sequence[Any]) -> dict[str, float]:
    radii = [math.hypot(float(point[0]), float(point[1])) for point in points]
    z_values = [float(point[2]) for point in points]
    return {
        "minimum_radius_m": min(radii),
        "maximum_radius_m": max(radii),
        "minimum_z_m": min(z_values),
        "maximum_z_m": max(z_values),
    }


def configure_continuous_plug_guide_runtime_collision(
    stage: Any,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Install the one authorized in-memory collider representation fix."""

    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics

    dimensions = _contract_dimensions(contract)
    if dimensions["fixed_clear_bore_radius_m"] <= dimensions["outer_radius_m"]:
        raise RuntimeError("contract no longer has radial shell guidance clearance")
    if dimensions["inner_radius_m"] <= dimensions["fixed_stop_outer_radius_m"]:
        raise RuntimeError("contract no longer has radial plug-guide/stop clearance")
    original = stage.GetPrimAtPath(CONTINUOUS_PLUG_GUIDE_PATH)
    if not original:
        raise RuntimeError(f"missing {CONTINUOUS_PLUG_GUIDE_PATH}")
    if stage.GetPrimAtPath(PROXY_ROOT_PATH):
        raise RuntimeError(f"runtime proxy already exists: {PROXY_ROOT_PATH}")
    approximation = original.GetAttribute("physics:approximation")
    if not approximation or str(approximation.Get()) != "none":
        raise RuntimeError("frozen plug-guide approximation is not the expected none")
    collision_enabled = original.GetAttribute("physics:collisionEnabled")
    if not collision_enabled or bool(collision_enabled.Get()) is not True:
        raise RuntimeError("frozen plug-guide collision is not enabled")
    samples = original.GetAttribute("kcg:circumferentialSamples")
    if not samples or int(samples.Get()) != SEGMENT_COUNT:
        raise RuntimeError("frozen plug-guide sample count changed")
    source_points = UsdGeom.Mesh(original).GetPointsAttr().Get()
    if source_points is None or len(source_points) != 4 * SEGMENT_COUNT:
        raise RuntimeError("frozen plug-guide point count changed")
    source_bounds = _realized_bounds(source_points)
    expected_bounds = {
        "minimum_radius_m": dimensions["inner_radius_m"],
        "maximum_radius_m": dimensions["outer_radius_m"],
        "minimum_z_m": dimensions["z0_m"],
        "maximum_z_m": dimensions["z1_m"],
    }
    if any(
        not math.isclose(source_bounds[key], value, rel_tol=0.0, abs_tol=2.0e-8)
        for key, value in expected_bounds.items()
    ):
        raise RuntimeError(
            f"frozen plug-guide bounds differ from contract: {source_bounds}"
        )

    specs = annular_convex_segment_specs(
        inner_radius_m=dimensions["inner_radius_m"],
        outer_radius_m=dimensions["outer_radius_m"],
        z0_m=dimensions["z0_m"],
        z1_m=dimensions["z1_m"],
    )
    collision_enabled.Set(False)
    proxy_root = UsdGeom.Xform.Define(stage, PROXY_ROOT_PATH).GetPrim()
    proxy_root.CreateAttribute(
        "kcg:runtimeCollisionProxyOf", Sdf.ValueTypeNames.String, custom=True
    ).Set(CONTINUOUS_PLUG_GUIDE_PATH)
    proxy_root.CreateAttribute(
        "kcg:exactConvexSegmentCount", Sdf.ValueTypeNames.Int, custom=True
    ).Set(SEGMENT_COUNT)
    proxy_root.CreateAttribute(
        "kcg:sourceAssetWriteAllowed", Sdf.ValueTypeNames.Bool, custom=True
    ).Set(False)

    for spec in specs:
        path = PROXY_ROOT_PATH + f"/Segment_{spec['index']:02d}"
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr(
            [Gf.Vec3f(*point) for point in spec["points_local_mm"]]
        )
        mesh.CreateFaceVertexCountsAttr(list(spec["face_vertex_counts"]))
        mesh.CreateFaceVertexIndicesAttr(list(spec["face_vertex_indices"]))
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.AddScaleOp().Set(Gf.Vec3f(LOCAL_TO_STAGE_SCALE))
        mesh.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        prim = mesh.GetPrim()
        prim.CreateAttribute(
            "kcg:collisionRole", Sdf.ValueTypeNames.String, custom=True
        ).Set("continuous_shell_and_guidance")
        prim.CreateAttribute(
            "kcg:runtimeProxySegmentIndex", Sdf.ValueTypeNames.Int, custom=True
        ).Set(int(spec["index"]))
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(
            UsdPhysics.Tokens.convexHull
        )
        PhysxSchema.PhysxConvexHullCollisionAPI.Apply(
            prim
        ).CreateMinThicknessAttr(0.001)

    return {
        "classification": "DETERMINISTIC_EXACT_CONVEX_SEGMENTS_RUNTIME_FIX",
        "source_path": CONTINUOUS_PLUG_GUIDE_PATH,
        "proxy_root_path": PROXY_ROOT_PATH,
        "source_collision_disabled_in_memory": True,
        "proxy_segment_count": len(specs),
        "local_to_stage_scale": LOCAL_TO_STAGE_SCALE,
        "source_bounds_m": source_bounds,
        "contract_bounds_m": expected_bounds,
        "guide_radial_clearance_m": (
            dimensions["fixed_clear_bore_radius_m"] - dimensions["outer_radius_m"]
        ),
        "fixed_stop_radial_clearance_m": (
            dimensions["inner_radius_m"] - dimensions["fixed_stop_outer_radius_m"]
        ),
        "geometry_points_changed": False,
        "visual_mesh_changed": False,
        "physical_guidance_effect_removed": False,
        "source_asset_written": False,
    }
