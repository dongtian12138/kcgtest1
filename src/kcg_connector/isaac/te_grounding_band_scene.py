"""Author a declared equivalent contact law for the source grounding-band envelope.

This is a local constitutive model, not a reconstruction of TE spring fingers.
Compression is represented by native compliant contact separation. The rigid
body, original keys, mass properties and Body--Nut joint are retained. Call only
before physics starts and AFTER the ordinary connector material assignment.
"""

from pathlib import Path
import hashlib
import json

import numpy as np


def install_grounding_band_contact_model(stage, body_path, manifest_path, *,
                                         stiffness_n_m, damping_ns_m,
                                         socket_collision_path=None,
                                         non_socket_contact_paths=()):
    import omni.timeline
    from pxr import PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

    timeline = omni.timeline.get_timeline_interface()
    if timeline.is_playing() or timeline.get_current_time() != 0.:
        raise RuntimeError("grounding-band law may only be authored before physics")
    if not (np.isfinite(stiffness_n_m) and stiffness_n_m > 0
            and np.isfinite(damping_ns_m) and damping_ns_m >= 0):
        raise ValueError("a finite positive stiffness and nonnegative damping are required")
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("original_asset_modified") is not False
            or manifest.get("original_key_face_count") != 106
            or manifest.get("maximum_key_vertex_to_rigid_partition_distance_m") != 0.):
        raise ValueError("expected verified source band partition retaining every key")
    meshes = {}
    for label in ("rigid_body", "circumferential_band"):
        entry = manifest["meshes"][label]
        path = Path(__file__).resolve().parents[3] / entry["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"partition geometry differs: {label}")
        with np.load(path) as data:
            meshes[label] = (data["vertices_m"].copy(), data["faces"].copy())

    body = stage.GetPrimAtPath(body_path)
    core = stage.GetPrimAtPath(body_path + "/SourceCadCollision")
    band_path = body_path + "/CompliantGroundingBandCollision"
    if (not body.HasAPI(UsdPhysics.RigidBodyAPI) or not core.IsA(UsdGeom.Mesh)
            or stage.GetPrimAtPath(band_path).IsValid()):
        raise ValueError("expected one existing unpartitioned source Body collider")
    original = core
    original_points = UsdGeom.Mesh(original).GetPointsAttr().Get()
    original_faces = UsdGeom.Mesh(original).GetFaceVertexIndicesAttr().Get()
    if socket_collision_path is not None:
        socket = stage.GetPrimAtPath(socket_collision_path)
        if not socket.HasAPI(UsdPhysics.CollisionAPI):
            raise ValueError("pair-specific compliance requires the actual socket collider")
        core_path = body_path + "/SocketRigidCoreCollision"
        if stage.GetPrimAtPath(core_path).IsValid():
            raise ValueError("socket-specific rigid core already exists")
        layer = stage.GetEditTarget().GetLayer()
        if not Sdf.CopySpec(layer, original.GetPath(), layer, Sdf.Path(core_path)):
            raise RuntimeError("could not retain the original source collider while copying the socket core")
        core = stage.GetPrimAtPath(core_path)
    mass_names = ("physics:mass", "physics:centerOfMass", "physics:diagonalInertia",
                  "physics:principalAxes")
    before_mass = {name: body.GetAttribute(name).Get() for name in mass_names}
    original_material, _ = UsdShade.MaterialBindingAPI(core).ComputeBoundMaterial("physics")
    if not original_material:
        raise ValueError("the ordinary connector material must already be resolved")
    material_api = UsdPhysics.MaterialAPI(original_material.GetPrim())
    friction = [material_api.GetStaticFrictionAttr().Get(),
                material_api.GetDynamicFrictionAttr().Get()]
    if not np.isfinite(friction).all() or min(friction) < 0:
        raise ValueError("invalid existing connector friction")

    # A prior stronger-than-descendants root binding would otherwise silently
    # suppress the local spring material. Preserve each old resolved binding.
    prior_bindings = {}
    for prim in Usd.PrimRange(body):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
            if not material:
                raise ValueError(f"unbound existing collider: {prim.GetPath()}")
            prior_bindings[str(prim.GetPath())] = str(material.GetPath())
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                material, UsdShade.Tokens.strongerThanDescendants, "physics")
    UsdShade.MaterialBindingAPI.Apply(body).Bind(
        original_material, UsdShade.Tokens.weakerThanDescendants, "physics")

    def set_mesh(shape, geometry):
        vertices, faces = geometry
        shape.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32)))
        shape.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, np.int32)))
        shape.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.astype(np.int32).ravel()))
        shape.CreateSubdivisionSchemeAttr("none")
        shape.CreateVisibilityAttr("invisible")

    set_mesh(UsdGeom.Mesh(core), meshes["rigid_body"])
    band = UsdGeom.Mesh.Define(stage, band_path)
    set_mesh(band, meshes["circumferential_band"])
    prim = band.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("sdf")
    PhysxSchema.PhysxCollisionAPI.Apply(prim)
    PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim)
    PhysxSchema.PhysxTriangleMeshCollisionAPI.Apply(prim)
    for attribute in core.GetAttributes():
        name = attribute.GetName()
        if (attribute.HasAuthoredValueOpinion()
                and name.startswith(("physxCollision:", "physxSDFMeshCollision:"))):
            prim.CreateAttribute(name, attribute.GetTypeName()).Set(attribute.Get())
    PhysxSchema.PhysxTriangleMeshCollisionAPI(prim).CreateWeldToleranceAttr(0.)

    soft = UsdShade.Material.Define(stage, "/World/RepresentativeGroundingBandMaterial")
    soft_api = UsdPhysics.MaterialAPI.Apply(soft.GetPrim())
    soft_api.CreateStaticFrictionAttr(float(friction[0]))
    soft_api.CreateDynamicFrictionAttr(float(friction[1]))
    soft_api.CreateRestitutionAttr(0.)
    native = PhysxSchema.PhysxMaterialAPI.Apply(soft.GetPrim())
    combine = PhysxSchema.PhysxMaterialAPI(original_material.GetPrim()).GetFrictionCombineModeAttr().Get()
    if combine:
        native.CreateFrictionCombineModeAttr(combine)
    native.CreateCompliantContactStiffnessAttr(float(stiffness_n_m))
    native.CreateCompliantContactDampingAttr(float(damping_ns_m))
    native.CreateCompliantContactAccelerationSpringAttr(False)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        soft, UsdShade.Tokens.strongerThanDescendants, "physics")
    resolved, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
    if resolved.GetPath() != soft.GetPath():
        raise RuntimeError("local compliant material was overridden")
    for path, material_path in prior_bindings.items():
        resolved, _ = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(path)).ComputeBoundMaterial("physics")
        if str(resolved.GetPath()) != material_path:
            raise RuntimeError(f"unintended material change: {path}")
    if before_mass != {name: body.GetAttribute(name).Get() for name in mass_names}:
        raise RuntimeError("body mass properties changed during local contact authoring")
    pair_coverage = None
    if socket_collision_path is not None:
        non_socket = tuple(dict.fromkeys(map(str, non_socket_contact_paths)))
        if body_path in non_socket or socket_collision_path in non_socket:
            raise ValueError("the socket core must not filter its own body or the socket")
        if any(not stage.GetPrimAtPath(path).IsValid() for path in non_socket):
            raise ValueError("a declared non-socket contact target is missing")
        # Each physical interaction retains one complete source envelope:
        # original SDF for grasp/table; rigid core + compliant band for socket.
        # The filters remove duplicate representations, not physical coverage.
        full_filter = UsdPhysics.FilteredPairsAPI.Apply(original).CreateFilteredPairsRel()
        full_filter.SetTargets([socket_collision_path])
        for shape in (core, prim):
            relation = UsdPhysics.FilteredPairsAPI.Apply(shape).CreateFilteredPairsRel()
            relation.SetTargets(non_socket)
            if tuple(map(str, relation.GetTargets())) != non_socket:
                raise RuntimeError("socket-only collision filtering did not read back")
        if (UsdGeom.Mesh(original).GetPointsAttr().Get() != original_points
                or UsdGeom.Mesh(original).GetFaceVertexIndicesAttr().Get() != original_faces
                or not UsdPhysics.CollisionAPI(original).GetCollisionEnabledAttr().Get()
                or list(map(str, full_filter.GetTargets())) != [socket_collision_path]):
            raise RuntimeError("original grasp SDF geometry or collision coverage changed")
        pair_coverage = {
            "non_socket_full_source_collision": str(original.GetPath()),
            "socket_rigid_core_collision": str(core.GetPath()),
            "socket_compliant_band_collision": band_path,
            "socket_collision": socket_collision_path,
            "non_socket_contact_paths": list(non_socket),
            "original_source_vertex_and_face_arrays_retained_exactly": True,
            "source_union_coverage_evidence": str(manifest_path),
            "dynamic_filter_behavior_requires_contact_evidence": True,
        }
    return {
        "scope": "REPRESENTATIVE_HOMOGENIZED_COMPLIANT_CONTACT_NOT_TE_LEAF_RECONSTRUCTION",
        "geometry_manifest": str(manifest_path), "body_path": body_path,
        "rigid_core_collision": str(core.GetPath()), "compliant_band_collision": band_path,
        "material_path": str(soft.GetPath()),
        "native_per_contact_stiffness_n_m": float(stiffness_n_m),
        "native_per_contact_damping_ns_m": float(damping_ns_m),
        "friction_retained": list(map(float, friction)),
        "mass_inertia_and_existing_joint_parameters_changed": False,
        "original_keys_retained_rigid": True, "post_start_object_pose_writes": False,
        "band_compression_representation": "NATIVE_COMPLIANT_CONTACT_SEPARATION",
        "visual_surface_deformation_computed": False,
        "manufacturer_force_deflection_curve_identified": False,
        "point_count_pose_and_resolution_sensitivity_requires_evaluation": True,
        "pair_specific_collision_coverage": pair_coverage,
    }
