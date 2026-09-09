"""Install the explicitly documented representative nut thread before reset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def install_representative_inner_thread(repository, stage, assembly_report, manifest_path):
    """Change the nut's mating mesh, preserving mass, joint and material state.

    The original external appearance uses the same modified mesh as collision.
    This function only authors a scene and makes no dynamic success claim.
    """
    import omni.timeline
    from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade

    timeline = omni.timeline.get_timeline_interface()
    if timeline.is_playing() or timeline.get_current_time() != 0:
        raise RuntimeError("representative thread geometry must be installed before physics starts")
    repository = Path(repository).resolve()
    manifest_path = (repository / manifest_path).resolve()
    manifest_path.relative_to(repository)
    manifest = json.loads(manifest_path.read_text())
    mesh_path = Path(manifest["output_mesh"]).resolve()
    mesh_path.relative_to(repository)
    if (manifest["schema_version"] != "te_representative_internal_thread_v1"
            or not manifest["watertight"] or not manifest["positive_volume"]
            or manifest["mass_inertia_material_joint_modified"]
            or manifest["online_pose_writes_or_thread_constraints_added"]
            or hashlib.sha256(mesh_path.read_bytes()).hexdigest() != manifest["output_sha256"]):
        raise ValueError("the documented thread mesh binding or model boundary differs")
    collision_path = assembly_report["collision"]["nut_collision"]
    collision = UsdGeom.Mesh.Get(stage, collision_path)
    if not collision or not collision.GetPrim().HasAPI(UsdPhysics.CollisionAPI):
        raise ValueError("the current source nut SDF collision is unavailable")
    nut_root = stage.GetPrimAtPath(collision.GetPath().GetParentPath())
    if not nut_root:
        raise ValueError("the original nut rigid body is unavailable")
    mass_api = UsdPhysics.MassAPI(nut_root)
    mass_before = {name: attr.Get() for name, attr in (
        ("mass", mass_api.GetMassAttr()), ("centerOfMass", mass_api.GetCenterOfMassAttr()),
        ("diagonalInertia", mass_api.GetDiagonalInertiaAttr()), ("principalAxes", mass_api.GetPrincipalAxesAttr()))}
    data = np.load(mesh_path)
    # Test the actual float coordinates sent to USD/PhysX as well as the
    # builder's topological assertion. No tolerance welding is performed here.
    vertices = np.asarray(data["vertices_m"], dtype=np.float32)
    faces = np.asarray(data["faces"], dtype=np.int32)
    if len(np.unique(vertices, axis=0)) != len(vertices):
        raise ValueError("the thread mesh contains coincident PhysX vertices")
    triangles = vertices[faces].astype(np.float64)
    if np.any(np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                     triangles[:, 2] - triangles[:, 0]), axis=1) == 0):
        raise ValueError("the thread mesh contains zero-area PhysX triangles")
    points = [Gf.Vec3f(*map(float, row)) for row in data["vertices_m"]]
    counts = [3] * len(data["faces"])
    indices = np.asarray(data["faces"], dtype=np.int32).ravel().tolist()
    collision.GetPointsAttr().Set(points)
    collision.GetFaceVertexCountsAttr().Set(counts)
    collision.GetFaceVertexIndicesAttr().Set(indices)
    # Default size-based welding can collapse the boolean mesh's short edges
    # and create non-manifold adjacency. Exact duplicates were removed in the
    # builder; retain all remaining source coordinates during PhysX cooking.
    PhysxSchema.PhysxTriangleMeshCollisionAPI.Apply(collision.GetPrim()).CreateWeldToleranceAttr(0.0)
    # Both surfaces describe the same mesh. Only the generated visual mesh is
    # displayed; the supplier file remains untouched on disk.
    original_visuals = [prim for prim in Usd.PrimRange(nut_root)
                        if prim.IsA(UsdGeom.Mesh) and not prim.HasAPI(UsdPhysics.CollisionAPI)]
    if not original_visuals:
        raise ValueError("the source nut has no visual surface to replace")
    material, _ = UsdShade.MaterialBindingAPI(original_visuals[0]).ComputeBoundMaterial()
    display_color = UsdGeom.Gprim(original_visuals[0]).GetDisplayColorAttr().Get()
    for prim in original_visuals:
        UsdGeom.Imageable(prim).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    visual_path = str(nut_root.GetPath()) + "/RepresentativeThreadVisual"
    visual = UsdGeom.Mesh.Define(stage, visual_path)
    visual.CreatePointsAttr(points)
    visual.CreateFaceVertexCountsAttr(counts)
    visual.CreateFaceVertexIndicesAttr(indices)
    visual.CreateSubdivisionSchemeAttr("none")
    visual.CreateVisibilityAttr(UsdGeom.Tokens.inherited)
    if material:
        UsdShade.MaterialBindingAPI.Apply(visual.GetPrim()).Bind(material)
    elif display_color:
        visual.CreateDisplayColorAttr(display_color)
    mass_after = {name: attr.Get() for name, attr in (
        ("mass", mass_api.GetMassAttr()), ("centerOfMass", mass_api.GetCenterOfMassAttr()),
        ("diagonalInertia", mass_api.GetDiagonalInertiaAttr()), ("principalAxes", mass_api.GetPrincipalAxesAttr()))}
    if mass_before != mass_after:
        raise RuntimeError("thread installation changed original mass or inertia attributes")
    result = {"manifest_path": str(manifest_path), "mesh_path": str(mesh_path),
        "mesh_sha256": manifest["output_sha256"], "collision_path": collision_path,
        "visual_path": visual_path, "original_visuals_hidden": [str(p.GetPath()) for p in original_visuals],
        "source_assets_modified": False, "mass_inertia_joint_material_changed": False,
        "post_start_pose_writes": False, "thread_motion_constraint_added": False,
        "representative_internal_thread_geometry_authored": True,
        "triangle_weld_tolerance_m": 0.0,
        "thread_rotation_to_axial_progress_dynamically_verified": False,
        "supplier_unverified_details": manifest["supplier_unverified_details"]}
    assembly_report["representative_inner_thread"] = result
    assembly_report["collision"]["nut_source_missing_internal_thread_geometry_reconstructed"] = True
    assembly_report["collision"]["nut_collision_provenance"] = "SOURCE_OUTER_CAD_WITH_STANDARD_BOUNDED_REPRESENTATIVE_INNER_THREAD"
    return result
