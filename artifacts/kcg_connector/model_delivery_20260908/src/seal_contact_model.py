"""Source-preserving, finite interfacial-seal contact for an isolated Isaac scene.

Author only before physics. The source envelope is retained; penetration of the
separate seal collider represents homogenized elastomer deformation, not CAD
deletion. This is a declared representative law, not an identified TE material.
No body pose, mass, inertia, joint or robot limit is changed here.
"""
from pathlib import Path
import hashlib
import json
import numpy as np


def install_front_seal(repository, stage, report, *, stiffness_n_m=100.,
                       damping_ns_m=0., manifest_path=None):
    """Install on an already-authored grounding-band + socket-interior scene.

    `report` is assembly_scene.json. The default point stiffness is an explicit
    initial numerical mapping value, requiring the short compression validation
    described in references/seal_model_basis.md before any frozen delivery.
    """
    import omni.timeline
    from pxr import PhysxSchema, UsdGeom, UsdPhysics, UsdShade, Vt

    repo = Path(repository).resolve()
    if manifest_path is None:
        manifest_path = repo / "artifacts/kcg_connector/isaac/te_full_assembly_20260907/interfacial_seal_geometry_02/geometry_manifest.json"
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    timeline = omni.timeline.get_timeline_interface()
    if timeline.is_playing() or timeline.get_current_time() != 0.:
        raise RuntimeError("front seal may only be authored before physics")
    if not np.isfinite([stiffness_n_m, damping_ns_m]).all() or stiffness_n_m <= 0 or damping_ns_m < 0:
        raise ValueError("finite positive stiffness and nonnegative damping required")
    if (manifest.get("source_assets_modified") is not False
            or manifest.get("source_metal_pin_vertices_checked") != 17920
            or manifest.get("source_key_faces_checked") != 106
            or manifest.get("maximum_critical_vertex_distance_to_rigid_partition_m", 1.) > 1e-8):
        raise ValueError("expected checked seal partition retaining original metal pins and keys")
    coverage = report["grounding_band_contact_model"]["pair_specific_collision_coverage"]
    core_path = coverage["socket_rigid_core_collision"]
    original_path = coverage["non_socket_full_source_collision"]
    body_path = str(Path(core_path).parent)
    body = stage.GetPrimAtPath(body_path)
    core = stage.GetPrimAtPath(core_path)
    original = stage.GetPrimAtPath(original_path)
    socket_path = coverage["socket_collision"]
    socket = stage.GetPrimAtPath(socket_path)
    seal_path = body_path + "/CompliantInterfacialSealCollision"
    if (not body.HasAPI(UsdPhysics.RigidBodyAPI) or not core.IsA(UsdGeom.Mesh)
            or not original.IsA(UsdGeom.Mesh) or not socket.HasAPI(UsdPhysics.CollisionAPI)
            or stage.GetPrimAtPath(seal_path).IsValid()):
        raise ValueError("expected original full Body and socket-only core; seal must not already exist")

    def load_checked(path, digest):
        path = Path(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"source partition was changed: {path}")
        with np.load(path) as data:
            return data["vertices_m"].copy(), data["faces"].copy()

    src_v, src_f = load_checked(manifest["source"], manifest["source_sha256"])
    core_mesh = UsdGeom.Mesh(core)
    old_v = np.asarray(core_mesh.GetPointsAttr().Get(), np.float64)
    old_f = np.asarray(core_mesh.GetFaceVertexIndicesAttr().Get(), np.int32).reshape(-1, 3)
    if old_f.shape != src_f.shape or not np.array_equal(old_f, src_f) or not np.allclose(old_v, src_v, atol=3e-9, rtol=0):
        raise ValueError("existing core differs from the source geometry used by the seal partition")
    meshes = {name: load_checked(entry["path"], entry["sha256"])
              for name, entry in manifest["meshes"].items()}
    mass_names = ("physics:mass", "physics:centerOfMass", "physics:diagonalInertia", "physics:principalAxes")
    before_mass = {key: str(body.GetAttribute(key).Get()) for key in mass_names}
    original_arrays = (UsdGeom.Mesh(original).GetPointsAttr().Get(),
                       UsdGeom.Mesh(original).GetFaceVertexIndicesAttr().Get())
    original_filter = list(UsdPhysics.FilteredPairsAPI(original).GetFilteredPairsRel().GetTargets())
    core_filter = list(UsdPhysics.FilteredPairsAPI(core).GetFilteredPairsRel().GetTargets())
    if socket.GetPath() not in original_filter:
        raise ValueError("full source Body must already be pair-deduplicated against Socket")
    for path in report.get("representative_mating_contacts", {}).get("clip_paths", []):
        if stage.GetPrimAtPath(path).GetPath() not in original_filter:
            raise ValueError("full source Body duplicates the socket clip interaction")
    material, _ = UsdShade.MaterialBindingAPI(core).ComputeBoundMaterial("physics")
    if not material:
        raise ValueError("ordinary core physics material must already resolve")
    old_material_path = str(material.GetPath())
    m = UsdPhysics.MaterialAPI(material.GetPrim())
    friction = [m.GetStaticFrictionAttr().Get(), m.GetDynamicFrictionAttr().Get()]
    if not np.isfinite(friction).all() or min(friction) < 0:
        raise ValueError("existing finite friction required")

    def set_mesh(shape, data):
        vertices, faces = data
        shape.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32)))
        shape.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, np.int32)))
        shape.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.astype(np.int32).ravel()))
        shape.CreateSubdivisionSchemeAttr("none")
        shape.CreateVisibilityAttr("invisible")

    set_mesh(core_mesh, meshes["rigid_core_without_front_seal"])
    shape = UsdGeom.Mesh.Define(stage, seal_path)
    set_mesh(shape, meshes["front_seal_envelope"])
    prim = shape.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("sdf")
    PhysxSchema.PhysxCollisionAPI.Apply(prim)
    PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim)
    PhysxSchema.PhysxTriangleMeshCollisionAPI.Apply(prim).CreateWeldToleranceAttr(0.)
    for attr in core.GetAttributes():
        if attr.HasAuthoredValueOpinion() and attr.GetName().startswith(("physxCollision:", "physxSDFMeshCollision:")):
            prim.CreateAttribute(attr.GetName(), attr.GetTypeName()).Set(attr.Get())
    # Duplicate coverage only: source Body remains the collider for hand/table;
    # this exact seal envelope replaces its rigid socket-facing representation.
    UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel().SetTargets(core_filter)
    soft = UsdShade.Material.Define(stage, body_path + "/RepresentativeInterfacialSealMaterial")
    sm = UsdPhysics.MaterialAPI.Apply(soft.GetPrim())
    sm.CreateStaticFrictionAttr(float(friction[0]))
    sm.CreateDynamicFrictionAttr(float(friction[1]))
    sm.CreateRestitutionAttr(0.)
    native = PhysxSchema.PhysxMaterialAPI.Apply(soft.GetPrim())
    mode = PhysxSchema.PhysxMaterialAPI(material.GetPrim()).GetFrictionCombineModeAttr().Get()
    if mode:
        native.CreateFrictionCombineModeAttr(mode)
    native.CreateCompliantContactStiffnessAttr(float(stiffness_n_m))
    native.CreateCompliantContactDampingAttr(float(damping_ns_m))
    native.CreateCompliantContactAccelerationSpringAttr(False)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(soft, UsdShade.Tokens.strongerThanDescendants, "physics")
    resolved, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
    retained, _ = UsdShade.MaterialBindingAPI(core).ComputeBoundMaterial("physics")
    if resolved.GetPath() != soft.GetPath() or str(retained.GetPath()) != old_material_path:
        raise RuntimeError("seal material binding overrode or was overridden by the source material")
    if before_mass != {key: str(body.GetAttribute(key).Get()) for key in mass_names}:
        raise RuntimeError("source Body mass/inertia changed")
    if (original_arrays != (UsdGeom.Mesh(original).GetPointsAttr().Get(),
                           UsdGeom.Mesh(original).GetFaceVertexIndicesAttr().Get())
            or original_filter != list(UsdPhysics.FilteredPairsAPI(original).GetFilteredPairsRel().GetTargets())
            or core_filter != list(UsdPhysics.FilteredPairsAPI(core).GetFilteredPairsRel().GetTargets())):
        raise RuntimeError("unrelated source collision arrays or filters changed")
    record = {
        "scope": "SOURCE_ENVELOPE_FINITE_INTERFACIAL_SEAL_CONTACT_NOT_EXACT_TE_CALIBRATION",
        "geometry_manifest": str(manifest_path), "body_path": body_path,
        "rigid_core_collision": core_path, "seal_collision": seal_path,
        "material_path": str(soft.GetPath()), "socket_collision": socket_path,
        "native_per_contact_stiffness_n_m": float(stiffness_n_m),
        "native_per_contact_damping_ns_m": float(damping_ns_m),
        "friction_retained": list(map(float, friction)), "friction_is_TE_calibrated": False,
        "source_visual_and_full_grasp_envelope_unchanged": True,
        "source_metal_pins_keys_and_final_shell_stop_retained": True,
        "source_mass_inertia_and_joints_unchanged": True,
        "source_seal_material_region_is_inferred": True,
        "deformation_representation": "NATIVE_CONTACT_SEPARATION_IN_EXACT_SOURCE_ENVELOPE",
        "visual_deformation_or_volume_conservation_simulated": False,
        "post_start_pose_write_or_extra_object_support": False,
        "non_socket_duplicate_pairs_filtered": list(map(str, core_filter)),
        "dynamic_validation_required_before_frozen_delivery": True,
        "basis_file": str(repo / "artifacts/kcg_connector/model_delivery_20260908/references/seal_model_basis.md"),
    }
    report["interfacial_seal_contact_model"] = record
    return record
