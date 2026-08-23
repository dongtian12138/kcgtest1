#!/usr/bin/env python3
"""Build one free D38999/26FJ35PN rigid body from the supplier STEP mesh."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml

from kcg_connector.grasp.robust.object_model import file_sha256


MM_TO_M = 1.0e-3
COACD_SEED = 20260823
COACD_MAX_HULLS = 64
COACD_THRESHOLD = 0.05


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description="Build a free tabletop plug with same-source convex collision"
    )
    parser.add_argument(
        "--source-stl",
        type=Path,
        default=repository
        / "artifacts/kcg_connector/isaac/te_j35_engineering_v1/visual/"
        "D38999_26FJ35PN_VISUAL.stl",
    )
    parser.add_argument(
        "--visual-usd",
        type=Path,
        default=repository
        / "artifacts/kcg_connector/isaac/te_j35_engineering_v1/visual/"
        "D38999_26FJ35PN_VISUAL.usdc",
    )
    parser.add_argument(
        "--simulation-config",
        type=Path,
        default=repository / "src/kcg_connector/config/te_j35_simulation_v1.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository
        / "artifacts/kcg_connector/isaac/te_j35_free_tabletop_v1",
    )
    return parser.parse_args(argv)


def _load_mass_kg(config_path: Path) -> float:
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    identity = document["identity"]
    if identity["plug"] != "D38999/26FJ35PN":
        raise ValueError("simulation config is not the TE/DEUTSCH 26FJ35PN plug")
    mass = document["mass"]
    value = float(mass["plug_body_kg"]) + float(mass["coupling_nut_kg"])
    if value <= 0.0:
        raise ValueError("simulation mass must be positive")
    return value


def _load_source_mesh(path: Path, mass_kg: float):
    import trimesh

    mesh = trimesh.load_mesh(path, force="mesh", process=True)
    if not mesh.is_watertight or not mesh.is_winding_consistent:
        raise ValueError("supplier STEP tessellation is not a closed consistent mesh")
    source_vertices_mm = np.asarray(mesh.vertices, dtype=np.float64)
    source_faces = np.asarray(mesh.faces, dtype=np.int32)
    mesh.apply_scale(MM_TO_M)
    mesh.density = mass_kg / float(mesh.volume)
    return mesh, source_vertices_mm, source_faces


def _decompose(vertices_mm: np.ndarray, faces: np.ndarray):
    import coacd

    source = coacd.Mesh(vertices_mm, faces)
    return coacd.run_coacd(
        source,
        threshold=COACD_THRESHOLD,
        max_convex_hull=COACD_MAX_HULLS,
        preprocess_mode="auto",
        preprocess_resolution=30,
        resolution=1000,
        mcts_nodes=10,
        mcts_iterations=50,
        mcts_max_depth=3,
        pca=False,
        merge=True,
        decimate=True,
        max_ch_vertex=128,
        seed=COACD_SEED,
    )


def _custom(prim, Sdf, name: str, type_name, value) -> None:
    prim.CreateAttribute(f"kcg:{name}", type_name, custom=True).Set(value)


def _author_asset(
    output_path: Path,
    visual_path: Path,
    hulls,
    mesh,
    source_sha256: str,
) -> None:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/TE_J35FreePlug")
    stage.SetDefaultPrim(root.GetPrim())
    prim = root.GetPrim()
    _custom(prim, Sdf, "productId", Sdf.ValueTypeNames.String, "D38999/26FJ35PN")
    _custom(prim, Sdf, "hardwareAuthorized", Sdf.ValueTypeNames.Bool, False)
    _custom(prim, Sdf, "sourceStlSha256", Sdf.ValueTypeNames.String, source_sha256)
    _custom(
        prim,
        Sdf,
        "collisionRepresentation",
        Sdf.ValueTypeNames.String,
        "COACD_CONVEX_PARTS_FROM_COMPLETE_SUPPLIER_STEP_TESSELLATION",
    )
    _custom(prim, Sdf, "collisionHullCount", Sdf.ValueTypeNames.Int, len(hulls))

    rigid = UsdPhysics.RigidBodyAPI.Apply(prim)
    rigid.CreateRigidBodyEnabledAttr(True)
    mass = UsdPhysics.MassAPI.Apply(prim)
    mass.CreateMassAttr(float(mesh.mass))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(*mesh.center_mass))
    mass.CreateDiagonalInertiaAttr(
        Gf.Vec3f(*np.diag(np.asarray(mesh.moment_inertia, dtype=np.float64)))
    )

    visual = UsdGeom.Xform.Define(stage, "/TE_J35FreePlug/Visual")
    relative_visual = Path(
        Path(visual_path).resolve().relative_to(output_path.parent.parent)
    )
    visual.GetPrim().GetReferences().AddReference(f"../{relative_visual.as_posix()}")

    UsdGeom.Scope.Define(stage, "/TE_J35FreePlug/Collision")
    for index, (vertices_mm, faces) in enumerate(hulls):
        path = f"/TE_J35FreePlug/Collision/Hull_{index:03d}"
        collider = UsdGeom.Mesh.Define(stage, path)
        points = np.asarray(vertices_mm, dtype=np.float64) * MM_TO_M
        triangles = np.asarray(faces, dtype=np.int32)
        collider.CreatePointsAttr([Gf.Vec3f(*row) for row in points])
        collider.CreateFaceVertexCountsAttr([3] * len(triangles))
        collider.CreateFaceVertexIndicesAttr(triangles.ravel().tolist())
        collider.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        collider.CreateExtentAttr(
            UsdGeom.PointBased.ComputeExtent(collider.GetPointsAttr().Get())
        )
        collider.CreatePurposeAttr(UsdGeom.Tokens.guide)
        collider.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
        mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(collider.GetPrim())
        mesh_collision.CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
    stage.GetRootLayer().Save()


def main(argv: Sequence[str] | None = None) -> int:
    repository = Path(__file__).resolve().parents[3]
    args = _arguments(argv)
    source = args.source_stl.resolve()
    visual = args.visual_usd.resolve()
    config = args.simulation_config.resolve()
    for path in (source, visual, config):
        if not path.is_file():
            raise FileNotFoundError(path)

    mass_kg = _load_mass_kg(config)
    mesh, vertices_mm, faces = _load_source_mesh(source, mass_kg)
    hulls = _decompose(vertices_mm, faces)
    if not hulls or len(hulls) > COACD_MAX_HULLS:
        raise RuntimeError("same-source convex decomposition did not close")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    asset_path = output_dir / "TE_J35_FREE_PLUG_V1.usdc"
    _author_asset(asset_path, visual, hulls, mesh, file_sha256(source))
    manifest = {
        "schema_version": "kcg_te_j35_free_tabletop_asset_v1",
        "product_id": "D38999/26FJ35PN",
        "hardware_authorized": False,
        "source_step_class": "OFFICIAL_TE_CUSTOMER_VIEW_MODEL_STEP",
        "source_stl": source.relative_to(repository).as_posix(),
        "source_stl_sha256": file_sha256(source),
        "visual_usd": visual.relative_to(repository).as_posix(),
        "visual_usd_sha256": file_sha256(visual),
        "source_mesh": {
            "watertight": True,
            "winding_consistent": True,
            "vertex_count": int(len(mesh.vertices)),
            "triangle_count": int(len(mesh.faces)),
            "bounds_m": np.asarray(mesh.bounds).tolist(),
        },
        "mass_properties": {
            "mass_kg": float(mesh.mass),
            "source": "INITIAL_SIMULATION_VALUE_NOT_VENDOR_MEASUREMENT",
            "center_of_mass_m": np.asarray(mesh.center_mass).tolist(),
            "inertia_kg_m2": np.asarray(mesh.moment_inertia).tolist(),
            "method": "UNIFORM_DENSITY_EQUIVALENT_FROM_WATERTIGHT_STEP_TESSELLATION",
        },
        "collision": {
            "method": "COACD_FROM_COMPLETE_SUPPLIER_STEP_TESSELLATION",
            "seed": COACD_SEED,
            "threshold": COACD_THRESHOLD,
            "maximum_hulls": COACD_MAX_HULLS,
            "hull_count": len(hulls),
            "vertex_count": sum(len(vertices) for vertices, _ in hulls),
            "triangle_count": sum(len(faces) for _, faces in hulls),
            "primitive_proxy_used": False,
        },
        "asset": asset_path.relative_to(repository).as_posix(),
        "asset_sha256": file_sha256(asset_path),
    }
    manifest_path = output_dir / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
