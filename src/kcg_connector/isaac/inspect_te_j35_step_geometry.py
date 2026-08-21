#!/usr/bin/env python3
"""Produce a deterministic, geometry-only audit of the two TE J35 STEP files.

This utility intentionally imports neither Isaac Sim nor Pixar USD.  It keeps
the supplier STEP topology and the future simulation-authoring decisions in
separate evidence layers.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plug-step", type=Path, required=True)
    parser.add_argument("--receptacle-step", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rounded(value: float) -> float:
    result = round(float(value), 9)
    return 0.0 if result == -0.0 else result


def _bbox_document(box: Any) -> dict[str, float]:
    return {
        "xmin": _rounded(box.xmin),
        "xmax": _rounded(box.xmax),
        "ymin": _rounded(box.ymin),
        "ymax": _rounded(box.ymax),
        "zmin": _rounded(box.zmin),
        "zmax": _rounded(box.zmax),
        "size_x": _rounded(box.xlen),
        "size_y": _rounded(box.ylen),
        "size_z": _rounded(box.zlen),
    }


def _inspect_step(path: Path) -> dict[str, Any]:
    import cadquery as cq
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane

    if not path.is_file():
        raise FileNotFoundError(path)
    model = cq.importers.importStep(str(path))
    solids = model.solids().vals()
    if not solids:
        raise ValueError(f"STEP contains no solid: {path}")
    shape = model.val()
    faces = shape.Faces()
    face_types = Counter(face.geomType() for face in faces)
    cylinder_groups: dict[float, dict[str, float | int]] = defaultdict(
        lambda: {
            "face_count": 0,
            "zmin_mm": math.inf,
            "zmax_mm": -math.inf,
            "area_mm2": 0.0,
        }
    )
    axial_plane_groups: dict[float, dict[str, float | int]] = defaultdict(
        lambda: {
            "face_count": 0,
            "maximum_abs_xy_mm": 0.0,
            "area_mm2": 0.0,
        }
    )
    for face in faces:
        box = face.BoundingBox()
        adaptor = BRepAdaptor_Surface(face.wrapped)
        surface_type = adaptor.GetType()
        if surface_type == GeomAbs_Cylinder:
            radius = _rounded(adaptor.Cylinder().Radius())
            group = cylinder_groups[radius]
            group["face_count"] = int(group["face_count"]) + 1
            group["zmin_mm"] = min(float(group["zmin_mm"]), box.zmin)
            group["zmax_mm"] = max(float(group["zmax_mm"]), box.zmax)
            group["area_mm2"] = float(group["area_mm2"]) + face.Area()
        elif surface_type == GeomAbs_Plane:
            center = face.Center()
            normal = face.normalAt(center)
            if abs(normal.z) >= 0.999999:
                z_value = _rounded(center.z)
                group = axial_plane_groups[z_value]
                group["face_count"] = int(group["face_count"]) + 1
                group["maximum_abs_xy_mm"] = max(
                    float(group["maximum_abs_xy_mm"]),
                    abs(box.xmin),
                    abs(box.xmax),
                    abs(box.ymin),
                    abs(box.ymax),
                )
                group["area_mm2"] = float(group["area_mm2"]) + face.Area()

    cylinders = []
    for radius, group in cylinder_groups.items():
        cylinders.append(
            {
                "radius_mm": radius,
                "face_count": int(group["face_count"]),
                "zmin_mm": _rounded(float(group["zmin_mm"])),
                "zmax_mm": _rounded(float(group["zmax_mm"])),
                "area_mm2": _rounded(float(group["area_mm2"])),
            }
        )
    cylinders.sort(key=lambda item: (-item["area_mm2"], item["radius_mm"]))

    axial_planes = []
    for z_value, group in axial_plane_groups.items():
        axial_planes.append(
            {
                "z_mm": z_value,
                "face_count": int(group["face_count"]),
                "maximum_abs_xy_mm": _rounded(
                    float(group["maximum_abs_xy_mm"])
                ),
                "area_mm2": _rounded(float(group["area_mm2"])),
            }
        )
    axial_planes.sort(key=lambda item: (-item["area_mm2"], item["z_mm"]))

    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "solid_count": len(solids),
        "face_count": len(faces),
        "face_type_counts": dict(sorted(face_types.items())),
        "bounding_box_mm": _bbox_document(shape.BoundingBox()),
        "cylindrical_faces_by_radius": cylinders,
        "axial_planar_faces_by_z": axial_planes,
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _arguments(argv)
    report = {
        "schema_version": "kcg_te_j35_step_geometry_audit_v1",
        "scope": "supplier_step_topology_only",
        "isaac_sim_loaded": False,
        "usd_authored": False,
        "assembly_validated": False,
        "products": {
            "plug": _inspect_step(arguments.plug_step),
            "receptacle": _inspect_step(arguments.receptacle_step),
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
