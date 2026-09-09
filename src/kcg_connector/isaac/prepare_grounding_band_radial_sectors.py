#!/usr/bin/env python3
"""Partition the unchanged band envelope into radial compliance elements.

Element count is a numerical discretization, not the manufacturer's finger count.
No stage is modified. Rest-state total mass, COM and inertia are preserved by
subtracting the allocated sector properties from the original Body properties.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
from pxr import Usd, UsdPhysics


def offset_inertia(mass, center):
    return mass*(np.dot(center, center)*np.eye(3)-np.outer(center, center))


def mass_properties(mesh, density):
    return float(mesh.volume*density), np.asarray(mesh.center_mass), np.asarray(mesh.moment_inertia*density)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--elements", type=int, default=140)
    args = parser.parse_args()
    if not 8 <= args.elements <= 240:
        raise ValueError("bounded numerical element count must leave room for the original articulation")
    args.output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[3]
    root = repo / "artifacts/kcg_connector/isaac/te_full_assembly_20260905"
    partition = json.loads((root / "grounding_band_geometry_03/geometry_manifest.json").read_text())
    entry = partition["meshes"]["circumferential_band"]
    with np.load(entry["path"]) as data:
        band = trimesh.Trimesh(data["vertices_m"], data["faces"], process=False)
    source_asset = repo / "artifacts/kcg_connector/isaac/te_j35_free_split_tabletop_real_mass_resistance_0p020_v2/TE_J35_FREE_SPLIT_PLUG_V1.usdc"
    source_stage = Usd.Stage.Open(str(source_asset))
    api = UsdPhysics.MassAPI(source_stage.GetPrimAtPath("/TE_J35FreeSplitPlug/Body"))
    original_mass = float(api.GetMassAttr().Get())
    original_center = np.asarray(api.GetCenterOfMassAttr().Get())
    quat = api.GetPrincipalAxesAttr().Get()
    axes = Rotation.from_quat([*quat.GetImaginary(), quat.GetReal()]).as_matrix()
    original_inertia = axes @ np.diag(api.GetDiagonalInertiaAttr().Get()) @ axes.T
    # A mass allocation rule only. It does not replace BeCu material density or
    # claim that the original heterogeneous Body is uniformly manufactured.
    allocation_density = original_mass/partition["original_volume_m3"]
    angle_step = 2*np.pi/args.elements
    records, parts = [], []
    for i in range(args.elements):
        a = (i+.25)*angle_step; b = a+angle_step
        xy = np.array([[0., 0.], [.04*np.cos(a), .04*np.sin(a)], [.04*np.cos(b), .04*np.sin(b)]])
        wedge = trimesh.convex.convex_hull(np.vstack((np.c_[xy, np.full(3, -.02)],
                                                     np.c_[xy, np.full(3, .001)])))
        part = trimesh.boolean.intersection([band, wedge], engine="manifold")
        if not part.is_volume or np.any(part.area_faces <= 0.):
            raise ValueError(f"element {i} is not a closed positive source partition")
        mass, center, inertia = mass_properties(part, allocation_density)
        path = args.output / f"sector_{i:03d}.npz"
        np.savez_compressed(path, vertices_m=part.vertices, faces=np.asarray(part.faces, np.int32))
        records.append({"index": i, "angle_center_rad": float((a+b)/2),
                        "angle_span_rad": angle_step, "path": str(path.resolve()),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "mass_kg": mass, "center_of_mass_body_m": center.tolist(),
                        "inertia_about_sector_com_in_body_axes_kg_m2": inertia.tolist(),
                        "volume_m3": float(part.volume)})
        parts.append(part)
    sector_mass = sum(r["mass_kg"] for r in records)
    core_mass = original_mass-sector_mass
    core_center = (original_mass*original_center-sum(r["mass_kg"]*np.asarray(r["center_of_mass_body_m"]) for r in records))/core_mass
    sector_origin_inertia = sum(np.asarray(r["inertia_about_sector_com_in_body_axes_kg_m2"])
                               + offset_inertia(r["mass_kg"], np.asarray(r["center_of_mass_body_m"])) for r in records)
    core_inertia = original_inertia+offset_inertia(original_mass, original_center)-sector_origin_inertia-offset_inertia(core_mass, core_center)
    if core_mass <= 0 or np.linalg.eigvalsh(core_inertia).min() <= 0:
        raise ValueError("mass-preserving allocation has a nonphysical rigid-core inertia")
    combined = trimesh.util.concatenate(parts)
    _, distance, _ = trimesh.proximity.closest_point(combined, band.vertices)
    volume_error = abs(sum(r["volume_m3"] for r in records)-band.volume)
    if distance.max() > 5e-8 or volume_error > band.volume*1e-5:
        raise ValueError("sector rest envelope does not preserve the source band")
    reconstructed_inertia = core_inertia+offset_inertia(core_mass, core_center)+sector_origin_inertia-offset_inertia(original_mass, original_center)
    result = {"scope": "OFFLINE_RADIAL_COMPLIANCE_DISCRETIZATION_NOT_TE_FINGER_RECONSTRUCTION",
              "element_count": args.elements, "element_count_is_physical_finger_count": False,
              "source_partition_manifest": str(root / "grounding_band_geometry_03/geometry_manifest.json"),
              "source_mass_asset": str(source_asset), "source_assets_modified": False,
              "installed_in_assembly": False, "physics_response_evaluated": False,
              "mass_allocation_rule": "Source band volume fraction of original Body mass; residual core properties preserve original rest-state aggregate mass, COM and inertia",
              "allocation_density_is_actual_BeCu_density": False,
              "allocation_density_kg_m3": allocation_density,
              "original_body_mass_kg": original_mass, "allocated_band_mass_kg": sector_mass,
              "original_body_center_of_mass_m": original_center.tolist(),
              "original_body_inertia_about_com_kg_m2": original_inertia.tolist(),
              "rigid_core_mass_kg": core_mass, "rigid_core_center_of_mass_m": core_center.tolist(),
              "rigid_core_inertia_about_com_kg_m2": core_inertia.tolist(),
              "rest_total_inertia_reconstruction_max_error_kg_m2": float(np.max(abs(reconstructed_inertia-original_inertia))),
              "rest_band_surface_vertex_count_checked": len(band.vertices),
              "maximum_source_band_vertex_distance_m": float(distance.max()),
              "band_partition_volume_error_m3": float(volume_error),
              "source_rigid_core_geometry": partition["meshes"]["rigid_body"],
              "sectors": records}
    (args.output / "geometry_manifest.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("sectors", "source_rigid_core_geometry")},indent=2))


if __name__ == "__main__":
    main()
