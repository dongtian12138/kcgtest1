"""Pre-reset radial-band model from source geometry and a declared spring law.

This authoring function is not yet called by the assembly entry point. It
returns every physical component for the caller's mass and contact auditing.
No world joint, runtime pose update or commanded radial target is introduced.
"""
from pathlib import Path
import hashlib
import json

import numpy as np


def install_radial_band(stage, body_path, geometry_manifest, *, total_stiffness_n_m,
                        physics_dt_s, position_iterations):
    import omni.timeline
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade, Vt
    from scipy.spatial.transform import Rotation

    timeline = omni.timeline.get_timeline_interface()
    if timeline.is_playing() or timeline.get_current_time() != 0.:
        raise RuntimeError("radial band may only be authored before physics starts")
    geometry_manifest = Path(geometry_manifest).resolve()
    data = json.loads(geometry_manifest.read_text())
    elements = data["sectors"]
    if not (len(elements) == data["element_count"] and data["source_assets_modified"] is False):
        raise ValueError("invalid source band partition")
    if not (np.isfinite(total_stiffness_n_m) and total_stiffness_n_m > 0
            and physics_dt_s > 0 and 1 <= position_iterations <= 255):
        raise ValueError("finite radial law and solver settings are required")
    body = stage.GetPrimAtPath(body_path)
    core = UsdGeom.Mesh(stage.GetPrimAtPath(body_path + "/SourceCadCollision"))
    if not core or not body.HasAPI(UsdPhysics.ArticulationRootAPI):
        raise ValueError("expected original floating Body/Nut articulation and source collider")
    if abs(float(UsdPhysics.MassAPI(body).GetMassAttr().Get()) - data["original_body_mass_kg"]) > 1e-9:
        raise ValueError("radial allocation must match the current original Body mass")
    material, _ = UsdShade.MaterialBindingAPI(core).ComputeBoundMaterial("physics")
    if not material:
        raise ValueError("ordinary connector material must be assigned before band authoring")
    body_transform = UsdGeom.Xformable(body).GetLocalTransformation()
    root = str(body.GetParent().GetPath())
    if stage.GetPrimAtPath(root + "/GroundingBandSector000").IsValid():
        raise ValueError("radial band already exists")

    def read_mesh(entry):
        path = Path(entry["path"])
        if entry.get("sha256") and hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"source partition changed: {path}")
        with np.load(path) as mesh:
            return mesh["vertices_m"].copy(), mesh["faces"].copy()

    def assign_mesh(mesh, vertices, faces, scale):
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy((vertices*scale).astype(np.float32)))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, np.int32)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.astype(np.int32).ravel()))
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateVisibilityAttr("invisible")
        if scale != 1.:
            mesh.AddScaleOp().Set(Gf.Vec3f(1./scale))

    def assign_mass(prim, mass, center, inertia):
        tensor = np.asarray(inertia, dtype=float)
        values, axes = np.linalg.eigh(tensor)
        if mass <= 0 or min(values) <= 0 or not np.allclose(tensor, tensor.T, atol=1e-16):
            raise ValueError("invalid allocated inertia")
        if np.linalg.det(axes) < 0:
            axes[:, -1] *= -1
        q = Rotation.from_matrix(axes).as_quat()
        api = UsdPhysics.MassAPI.Apply(prim)
        api.CreateMassAttr(float(mass))
        api.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, center)))
        api.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, values)))
        api.CreatePrincipalAxesAttr(Gf.Quatf(float(q[3]), Gf.Vec3f(*map(float, q[:3]))))

    vertices, faces = read_mesh(data["source_rigid_core_geometry"])
    assign_mesh(core, vertices, faces, 1.)
    assign_mass(body, data["rigid_core_mass_kg"], data["rigid_core_center_of_mass_m"],
                data["rigid_core_inertia_about_com_kg_m2"])
    PhysxSchema.PhysxArticulationAPI.Apply(body).CreateEnabledSelfCollisionsAttr(False)
    stiffness = float(total_stiffness_n_m)/len(elements)
    travel = .0004
    h = float(physics_dt_s)/position_iterations
    records = []
    for element in elements:
        index = int(element["index"])
        path = root + f"/GroundingBandSector{index:03d}"
        leaf = UsdGeom.Xform.Define(stage, path)
        leaf.AddTransformOp().Set(body_transform)
        prim = leaf.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim).CreateRigidBodyEnabledAttr(True)
        assign_mass(prim, element["mass_kg"], element["center_of_mass_body_m"],
                    element["inertia_about_sector_com_in_body_axes_kg_m2"])
        shape = UsdGeom.Mesh.Define(stage, path + "/SourceEnvelope")
        vertices, faces = read_mesh(element)
        assign_mesh(shape, vertices, faces, 1000.)
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim()).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(shape.GetPrim()).CreateApproximationAttr("convexHull")
        hull = PhysxSchema.PhysxConvexHullCollisionAPI.Apply(shape.GetPrim())
        hull.CreateHullVertexLimitAttr(255)
        hull.CreateMinThicknessAttr(.001)
        contact = PhysxSchema.PhysxCollisionAPI.Apply(shape.GetPrim())
        contact.CreateContactOffsetAttr(.00005)
        contact.CreateRestOffsetAttr(0.)
        UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(
            material, UsdShade.Tokens.strongerThanDescendants, "physics")

        joint_path = path + "RadialSpring"
        joint = UsdPhysics.PrismaticJoint.Define(stage, joint_path)
        joint.CreateBody0Rel().SetTargets([body.GetPath()])
        joint.CreateBody1Rel().SetTargets([prim.GetPath()])
        joint.CreateAxisAttr("X")
        center = Gf.Vec3f(*map(float, element["center_of_mass_body_m"]))
        q = Rotation.from_euler("z", element["angle_center_rad"]).as_quat()
        rotation = Gf.Quatf(float(q[3]), Gf.Vec3f(*map(float, q[:3])))
        joint.CreateLocalPos0Attr(center); joint.CreateLocalPos1Attr(center)
        joint.CreateLocalRot0Attr(rotation); joint.CreateLocalRot1Attr(rotation)
        joint.CreateLowerLimitAttr(-travel); joint.CreateUpperLimitAttr(0.)
        damping = 2.*np.sqrt(stiffness*element["mass_kg"])
        denominator = 1.-(h*damping+h*h*stiffness)/element["mass_kg"]
        if denominator <= 0:
            raise ValueError("effective solver step is outside the validated inverse-drive mapping")
        force_cap = stiffness*travel+damping*.003
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(float(stiffness/denominator))
        drive.CreateDampingAttr(float(damping/denominator))
        drive.CreateMaxForceAttr(float(force_cap))
        drive.CreateTargetPositionAttr(0.); drive.CreateTargetVelocityAttr(0.)
        records.append({"body_path": path, "joint_path": joint_path, "sector_index": index,
                        "mass_kg": element["mass_kg"], "physical_k_n_m": stiffness,
                        "physical_d_ns_m": float(damping), "drive_mapping_denominator": float(denominator),
                        "finite_drive_force_cap_n": float(force_cap)})

    return {"scope": "REPRESENTATIVE_PASSIVE_RADIAL_BAND_NOT_TE_LEAF_RECONSTRUCTION",
            "geometry_manifest": str(geometry_manifest), "core_path": body_path,
            "physical_components": records, "rest_aggregate_body_mass_kg": data["original_body_mass_kg"],
            "rest_aggregate_body_com_m": data["original_body_center_of_mass_m"],
            "rest_aggregate_body_inertia_kg_m2": data["original_body_inertia_about_com_kg_m2"],
            "zero_radial_targets_authored_before_start": True, "world_joint_added": False,
            "post_start_pose_or_passive_target_writes": False,
            "source_outer_visual_mesh_retained_without_deformation": True,
            "whole_hull_collision_approximates_source_sector": True,
            "requires_component_mass_contact_and_joint_audits_in_caller": True,
            "manufacturer_constitutive_properties_identified": False,
            "assembly_integration_or_physical_success_verified": False}
