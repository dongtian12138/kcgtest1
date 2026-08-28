#!/usr/bin/env python3
"""Author a three-finger hand asset directly from the user-provided STL."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


STL_SCALE_M_PER_UNIT = 0.001
HAND_BASE_SUFFIX = (
    "/Geometry/world/iiwa_link_0/iiwa_link_1/iiwa_link_2/iiwa_link_3/"
    "iiwa_link_4/iiwa_link_5/iiwa_link_6/iiwa_link_7/iiwa_link_ee/"
    "handbase_link"
)
LINK_SPECS = (
    {
        "link": "f1Link3",
        "path": "/f1Link1/f1Link2/f1Link3",
        "old_visual": "f1Link3",
        "old_collision": "f1Link3_convex",
        "yaw_rad": 2.34686878808,
        "translation_m": (0.02049125144, -0.00300734189, 0.0),
    },
    {
        "link": "f2Link2",
        "path": "/f2Link1/f2Link2",
        "old_visual": "f2Link2",
        "old_collision": "f2Link2_convex",
        "yaw_rad": 2.539984764379217,
        "translation_m": (
            0.020687502006,
            0.000981199056,
            -0.023999999874,
        ),
    },
    {
        "link": "f3Link3",
        "path": "/f3Link1/f3Link2/f3Link3",
        "old_visual": "f3Link3",
        "old_collision": "f3Link3_convex",
        "yaw_rad": 2.34686878808,
        "translation_m": (0.02049125144, -0.00300734189, 0.0),
    },
)


def _arguments() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-asset",
        default=str(
            repository
            / "artifacts/kcg_connector/isaac/robot/"
            "handarm_keyed_v3_physical_r7/handarm.usda"
        ),
    )
    parser.add_argument("--stl", default="/home/noob/Downloads/指尖无指甲.STL")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _transform_points(points, yaw_rad: float, translation_m):
    import numpy as np

    yaw = float(yaw_rad)
    rotation = np.asarray(
        (
            (math.cos(yaw), -math.sin(yaw), 0.0),
            (math.sin(yaw), math.cos(yaw), 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    return (
        points * STL_SCALE_M_PER_UNIT
    ) @ rotation.T + np.asarray(translation_m, dtype=np.float64)


def _author_meshes(stage, link_path: str, points, faces, source_stl: Path) -> None:
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics

    point_values = [Gf.Vec3f(*map(float, row)) for row in points]
    face_counts = [3] * len(faces)
    face_indices = faces.reshape(-1).tolist()

    visual = UsdGeom.Mesh.Define(stage, link_path + "/nailfree_visual")
    visual.CreatePointsAttr(point_values)
    visual.CreateFaceVertexCountsAttr(face_counts)
    visual.CreateFaceVertexIndicesAttr(face_indices)
    visual.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    visual.CreateDoubleSidedAttr(True)
    visual.CreateDisplayColorAttr([Gf.Vec3f(0.18, 0.62, 0.92)])

    collision = UsdGeom.Mesh.Define(stage, link_path + "/nailfree_collision")
    collision.CreatePointsAttr(point_values)
    collision.CreateFaceVertexCountsAttr(face_counts)
    collision.CreateFaceVertexIndicesAttr(face_indices)
    collision.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    collision.CreateDoubleSidedAttr(True)
    collision.CreatePurposeAttr(UsdGeom.Tokens.guide)
    collision.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    UsdPhysics.CollisionAPI.Apply(collision.GetPrim()).CreateCollisionEnabledAttr(
        True
    )
    mesh_api = UsdPhysics.MeshCollisionAPI.Apply(collision.GetPrim())
    mesh_api.CreateApproximationAttr().Set("convexDecomposition")
    decomposition_api = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(
        collision.GetPrim()
    )
    decomposition_api.CreateShrinkWrapAttr().Set(True)
    physx_api = PhysxSchema.PhysxCollisionAPI.Apply(collision.GetPrim())
    physx_api.CreateContactOffsetAttr(0.00005)
    physx_api.CreateRestOffsetAttr(0.0)
    collision.GetPrim().CreateRelationship("material:binding:physics").SetTargets(
        [Sdf.Path("/HandArm/PhysicsMaterials/fingertip_pad")]
    )

    for prim, role in (
        (visual.GetPrim(), "nailfree_fingertip_visual"),
        (collision.GetPrim(), "nailfree_fingertip_collision"),
    ):
        prim.CreateAttribute(
            "kcg:sourceMeshUri", Sdf.ValueTypeNames.String, custom=True
        ).Set(str(source_stl))
        prim.CreateAttribute(
            "kcg:geometryRole", Sdf.ValueTypeNames.String, custom=True
        ).Set(role)
    collision.GetPrim().CreateAttribute(
        "kcg:materialRole", Sdf.ValueTypeNames.String, custom=True
    ).Set("fingertip_pad")


def main() -> int:
    import numpy as np
    import trimesh

    arguments = _arguments()
    source_asset = Path(arguments.source_asset).expanduser().resolve()
    source_stl = Path(arguments.stl).expanduser().resolve()
    output = Path(arguments.output).expanduser().resolve()
    if not source_asset.is_file():
        raise FileNotFoundError(source_asset)
    if not source_stl.is_file():
        raise FileNotFoundError(source_stl)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    raw = trimesh.load_mesh(source_stl, force="mesh", process=False)
    if len(raw.faces) == 0 or len(raw.vertices) == 0:
        raise ValueError("supplied STL contains no triangles")
    source_points = np.asarray(raw.vertices, dtype=np.float64)
    faces = np.asarray(raw.faces, dtype=np.int64)

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {"headless": True, "multi_gpu": False, "active_gpu": 0, "physics_gpu": 0}
    )
    from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/HandArm")
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().GetReferences().AddReference(str(source_asset))

    rows = []
    for spec in LINK_SPECS:
        link_path = "/HandArm" + HAND_BASE_SUFFIX + spec["path"]
        link_prim = stage.GetPrimAtPath(link_path)
        if not link_prim.IsValid() or not link_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(f"missing expected rigid link: {link_path}")

        old_visual_path = link_path + "/" + spec["old_visual"]
        old_collision_path = link_path + "/" + spec["old_collision"]
        for old_path in (old_visual_path, old_collision_path):
            old_prim = stage.GetPrimAtPath(old_path)
            if not old_prim.IsValid():
                raise RuntimeError(f"missing expected old fingertip prim: {old_path}")
            old_prim.SetActive(False)

        transformed = _transform_points(
            source_points, spec["yaw_rad"], spec["translation_m"]
        )
        _author_meshes(stage, link_path, transformed, faces, source_stl)
        PhysxSchema.PhysxContactReportAPI.Apply(
            link_prim
        ).CreateThresholdAttr().Set(0.0)

        active_colliders = []
        for prim in Usd.PrimRange(link_prim):
            if prim.IsActive() and prim.HasAPI(UsdPhysics.CollisionAPI):
                active_colliders.append(str(prim.GetPath()))
        expected_collider = link_path + "/nailfree_collision"
        if active_colliders != [expected_collider]:
            raise RuntimeError(
                f"{spec['link']}: unexpected active colliders {active_colliders}"
            )
        rows.append(
            {
                "link": spec["link"],
                "formula": "p_link = Rz(yaw) * (0.001 * p_stl) + t",
                "yaw_rad": spec["yaw_rad"],
                "translation_m": list(spec["translation_m"]),
                "bounds_link_m": transformed.min(axis=0).tolist()
                + transformed.max(axis=0).tolist(),
                "old_visual_active": stage.GetPrimAtPath(
                    old_visual_path
                ).IsActive(),
                "old_collision_active": stage.GetPrimAtPath(
                    old_collision_path
                ).IsActive(),
                "active_colliders": active_colliders,
                "convex_decomposition_shrink_wrap": bool(
                    stage.GetPrimAtPath(expected_collider)
                    .GetAttribute(
                        "physxConvexDecompositionCollision:shrinkWrap"
                    )
                    .Get()
                ),
            }
        )

    stage.GetRootLayer().Save()
    print(
        json.dumps(
            {
                "source_asset": str(source_asset),
                "source_stl": str(source_stl),
                "output": str(output),
                "source_face_count": int(len(faces)),
                "links": rows,
                "formal_dynamic_pass": False,
                "hardware_authorized": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
