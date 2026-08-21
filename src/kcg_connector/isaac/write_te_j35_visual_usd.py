#!/usr/bin/env python3
"""Convert the two high-resolution binary STL meshes to visual-only USD."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import struct
from typing import Callable, Sequence


MM_TO_M = 1.0e-3


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plug-stl", type=Path, required=True)
    parser.add_argument("--receptacle-stl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _read_binary_stl(path: Path):
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError(f"STL is too short: {path}")
    triangle_count = struct.unpack_from("<I", data, 80)[0]
    required_size = 84 + triangle_count * 50
    if len(data) != required_size:
        raise ValueError(f"expected binary STL, got incompatible size: {path}")

    points: list[tuple[float, float, float]] = []
    indices: list[int] = []
    normals: list[tuple[float, float, float]] = []
    point_lookup: dict[tuple[float, float, float], int] = {}
    offset = 84
    for _ in range(triangle_count):
        values = struct.unpack_from("<12fH", data, offset)
        normal = tuple(float(value) for value in values[0:3])
        normals.append(normal)
        for start in (3, 6, 9):
            vertex_mm = tuple(float(value) for value in values[start : start + 3])
            vertex_m = tuple(value * MM_TO_M for value in vertex_mm)
            index = point_lookup.get(vertex_m)
            if index is None:
                index = len(points)
                point_lookup[vertex_m] = index
                points.append(vertex_m)
            indices.append(index)
        offset += 50
    return points, indices, normals, triangle_count


def _filtered_mesh(
    points,
    indices,
    normals,
    keep_triangle: Callable[[tuple[tuple[float, float, float], ...]], bool],
):
    filtered_points: list[tuple[float, float, float]] = []
    filtered_indices: list[int] = []
    filtered_normals: list[tuple[float, float, float]] = []
    point_lookup: dict[tuple[float, float, float], int] = {}
    for triangle_index, normal in enumerate(normals):
        source_indices = indices[3 * triangle_index : 3 * triangle_index + 3]
        triangle = tuple(points[index] for index in source_indices)
        if not keep_triangle(triangle):
            continue
        filtered_normals.append(normal)
        for vertex in triangle:
            index = point_lookup.get(vertex)
            if index is None:
                index = len(filtered_points)
                point_lookup[vertex] = index
                filtered_points.append(vertex)
            filtered_indices.append(index)
    return filtered_points, filtered_indices, filtered_normals


def _is_coupling_nut_triangle(triangle) -> bool:
    center_z = sum(vertex[2] for vertex in triangle) / 3.0
    center_radius = sum(math.hypot(vertex[0], vertex[1]) for vertex in triangle) / 3.0
    return -0.0218 <= center_z <= -0.0014 and center_radius >= 0.0190


def _author_component(
    stl_path: Path,
    output_path: Path,
    product_id: str,
    *,
    keep_triangle: Callable[[tuple[tuple[float, float, float], ...]], bool] | None = None,
) -> dict:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    points, indices, normals, triangle_count = _read_binary_stl(stl_path)
    if keep_triangle is not None:
        points, indices, normals = _filtered_mesh(
            points, indices, normals, keep_triangle
        )
        triangle_count = len(normals)
    stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, "/TE_J35")
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().SetCustomDataByKey("productId", product_id)
    root.GetPrim().SetCustomDataByKey("sourceStepTessellation", stl_path.name)
    root.GetPrim().SetCustomDataByKey("representation", "visual_only")

    mesh = UsdGeom.Mesh.Define(stage, "/TE_J35/Geometry")
    mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in points])
    mesh.CreateFaceVertexCountsAttr([3] * triangle_count)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateNormalsAttr([Gf.Vec3f(*normal) for normal in normals])
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.uniform)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(False)
    mesh.CreateExtentAttr(UsdGeom.PointBased.ComputeExtent(mesh.GetPointsAttr().Get()))
    mesh.GetPrim().CreateAttribute(
        "kcg:collisionEnabled", Sdf.ValueTypeNames.Bool, custom=True
    ).Set(False)

    material = UsdShade.Material.Define(stage, "/TE_J35/Looks/Metal")
    shader = UsdShade.Shader.Define(stage, "/TE_J35/Looks/Metal/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.58, 0.61, 0.66)
    )
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.72)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.28)
    material.CreateSurfaceOutput().ConnectToSource(
        shader.ConnectableAPI(), "surface"
    )
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()
    return {
        "product_id": product_id,
        "triangle_count": triangle_count,
        "point_count": len(points),
        "output": str(output_path),
    }


def _author_review_stage(output_dir: Path, plug_name: str, receptacle_name: str) -> Path:
    from pxr import Gf, Usd, UsdGeom

    output_path = output_dir / "TE_J35_VISUAL_BASELINE.usda"
    stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/TE_J35VisualBaseline")
    stage.SetDefaultPrim(root.GetPrim())

    plug = UsdGeom.Xform.Define(stage, "/TE_J35VisualBaseline/Plug")
    plug.GetPrim().GetReferences().AddReference(f"./{plug_name}")
    plug.AddTranslateOp().Set(Gf.Vec3d(-0.03, 0.0, 0.0))

    receptacle = UsdGeom.Xform.Define(
        stage, "/TE_J35VisualBaseline/Receptacle"
    )
    receptacle.GetPrim().GetReferences().AddReference(f"./{receptacle_name}")
    receptacle.AddTranslateOp().Set(Gf.Vec3d(0.03, 0.0, 0.0))
    stage.GetRootLayer().Save()
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _arguments(argv)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    plug_name = "D38999_26FJ35PN_VISUAL.usdc"
    receptacle_name = "D38999_20FJ35SN_VISUAL.usdc"
    records = [
        _author_component(
            arguments.plug_stl,
            arguments.output_dir / plug_name,
            "D38999/26FJ35PN",
        ),
        _author_component(
            arguments.receptacle_stl,
            arguments.output_dir / receptacle_name,
            "D38999/20FJ35SN",
        ),
        _author_component(
            arguments.plug_stl,
            arguments.output_dir / "D38999_26FJ35PN_BODY_VISUAL.usdc",
            "D38999/26FJ35PN body assembly",
            keep_triangle=lambda triangle: not _is_coupling_nut_triangle(triangle),
        ),
        _author_component(
            arguments.plug_stl,
            arguments.output_dir / "D38999_26FJ35PN_COUPLING_NUT_VISUAL.usdc",
            "D38999/26FJ35PN coupling nut",
            keep_triangle=_is_coupling_nut_triangle,
        ),
    ]
    review_stage = _author_review_stage(
        arguments.output_dir, plug_name, receptacle_name
    )
    for record in records:
        print(
            f"{record['product_id']}: {record['triangle_count']} triangles, "
            f"{record['point_count']} points -> {record['output']}"
        )
    print(f"review stage -> {review_stage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
