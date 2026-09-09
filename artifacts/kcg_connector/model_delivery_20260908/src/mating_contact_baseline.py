"""Small, explicit geometric proxy for a compliant 22-size contact clip.

The six leaves are a homogenized envelope, not a reconstruction of a TE part.
The original pin and outer cavity are retained by callers. Force scale must be
measured in a separate coupon before this geometry is used in assembly.
"""
import numpy as np
from scipy.spatial import ConvexHull


def install_socket_contact_interior(repository, stage, prepared, manifest_path):
    """Install the checked cavity/clip proxy after ordinary material assignment.

    The socket stays on its original fixture. Only its missing interior is
    replaced; the plug remains free. No robot limit or source mass is changed.
    """
    import json
    from pathlib import Path
    import omni.timeline
    from pxr import PhysxSchema,Usd,UsdGeom,UsdPhysics,UsdShade,Vt

    timeline=omni.timeline.get_timeline_interface()
    if timeline.is_playing() or timeline.get_current_time()!=0.:
        raise RuntimeError("socket interior must be authored before physics")
    repository=Path(repository).resolve();manifest_path=Path(manifest_path)
    if not manifest_path.is_absolute():manifest_path=repository/manifest_path
    manifest_path=manifest_path.resolve();manifest_path.relative_to(repository)
    manifest=json.loads(manifest_path.read_text())
    if manifest.get("contact_count")!=128 or manifest.get("source_assets_or_plug_mass_inertia_modified") is not False:
        raise ValueError("expected the preserved-source 128-contact interior manifest")
    deep_manifest_path=Path(manifest["deep_cavity_manifest"])
    deep=json.loads(deep_manifest_path.read_text())
    if not (deep.get("unchanged_exterior_openings_and_lead_ins") and deep.get("unchanged_original_mesh_topology")):
        raise ValueError("the interior must preserve original exterior and cavity topology")
    data=np.load(manifest["deep_cavity_mesh"])
    vertices=np.asarray(data["vertices_m"],np.float64);faces=np.asarray(data["faces"],np.int32)
    moved=np.asarray(data["moved_source_vertex_indices"],np.int64)
    socket=UsdGeom.Mesh.Get(stage,prepared["receptacle_collision_path"])
    root=stage.GetPrimAtPath(prepared["receptacle_root"])
    if not socket or not root.IsValid():raise ValueError("the existing fixed socket is required")
    cache=UsdGeom.XformCache()
    root_pose=np.asarray(cache.GetLocalToWorldTransform(root)).T
    mesh_pose=np.asarray(cache.GetLocalToWorldTransform(socket.GetPrim())).T
    if not np.allclose(np.linalg.inv(root_pose)@mesh_pose,np.eye(4),atol=1e-12,rtol=0):
        raise ValueError("source socket mesh must retain its verified metre canonical frame")
    original=np.asarray(socket.GetPointsAttr().Get(),np.float64)
    original_faces=np.asarray(socket.GetFaceVertexIndicesAttr().Get(),np.int32).reshape(-1,3)
    expected=vertices.copy();expected[moved,2]=float(deep["original_floor_z_m"])
    if original_faces.shape!=faces.shape or not np.allclose(
            original[original_faces],expected[faces],atol=3e-9,rtol=0):
        raise ValueError("candidate cavity triangles differ from the existing source surface")
    # The OBJ recipe is triangle soup while the original USD shares vertices.
    # Transfer its compatible deformation to those original shared vertices,
    # retaining every native face index and checking all incident triangles.
    replacement=np.zeros_like(original);incidence=np.zeros(len(original))
    np.add.at(replacement,original_faces.ravel(),vertices[faces].reshape(-1,3))
    np.add.at(incidence,original_faces.ravel(),1.)
    used=incidence>0;replacement[used]/=incidence[used,None]
    replacement[~used]=original[~used]
    if not np.allclose(replacement[original_faces],vertices[faces],atol=3e-9,rtol=0):
        raise ValueError("cavity extension is incompatible with the source shared-vertex topology")
    parts=prepared["scene"]["part_prim_paths"]
    mass_names=("physics:mass","physics:centerOfMass","physics:diagonalInertia","physics:principalAxes")
    before={path:{n:str(stage.GetPrimAtPath(path).GetAttribute(n).Get()) for n in mass_names} for path in parts}
    band=prepared["report"].get("grounding_band_contact_model",{})
    coverage=band.get("pair_specific_collision_coverage")
    if not coverage:
        raise ValueError("the source full-Body and socket-specific rigid-core coverage is required")
    original_body=stage.GetPrimAtPath(coverage["non_socket_full_source_collision"])
    core=stage.GetPrimAtPath(coverage["socket_rigid_core_collision"])
    if not core.IsValid():raise ValueError("socket-specific original rigid core is missing")
    law=manifest["contact_law"]
    K=float(law["native_per_contact_stiffness_n_m"]);D=float(law["native_per_contact_damping_ns_m"])
    mu=np.asarray([law["static_friction"],law["dynamic_friction"]],float)
    if not np.isfinite([K,D,*mu]).all() or K<=0 or D<0 or np.any(mu<0):
        raise ValueError("finite passive contact parameters are required")
    # Keep existing resolved materials if a parent binding would override the
    # new local spring material. The existing socket material is not retuned.
    prior={}
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            material,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
            if material:
                prior[str(prim.GetPath())]=str(material.GetPath())
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(material,UsdShade.Tokens.strongerThanDescendants,"physics")
    root_binding=UsdShade.MaterialBindingAPI(root)
    root_material,_=root_binding.ComputeBoundMaterial("physics")
    if root_material:root_binding.Bind(root_material,UsdShade.Tokens.weakerThanDescendants,"physics")
    material=UsdShade.Material.Define(stage,prepared["receptacle_root"]+"/RepresentativeClipMaterial")
    m=UsdPhysics.MaterialAPI.Apply(material.GetPrim());m.CreateStaticFrictionAttr(float(mu[0]))
    m.CreateDynamicFrictionAttr(float(mu[1]));m.CreateRestitutionAttr(0.)
    px=PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim());px.CreateFrictionCombineModeAttr("min")
    px.CreateCompliantContactStiffnessAttr(K);px.CreateCompliantContactDampingAttr(D)
    px.CreateCompliantContactAccelerationSpringAttr(False)
    clip=np.load(manifest["clip_geometry_npz"])
    local=np.asarray(clip["vertices_m"],np.float64);clip_faces=np.asarray(clip["faces"],np.int32)
    centers=np.asarray(clip["contact_centers_socket_xy_m"],np.float64)
    if centers.shape!=(128,2) or not np.isfinite(centers).all():raise ValueError("original 128 contact centres required")
    clip_paths=[]
    for i,center in enumerate(centers):
        path=prepared["receptacle_root"]+f"/RepresentativeContacts/Clip{i:03d}"
        if stage.GetPrimAtPath(path).IsValid():raise ValueError("contact proxy already authored")
        points=local.copy();points[:,:2]+=center
        shape=UsdGeom.Mesh.Define(stage,path)
        shape.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
        shape.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(clip_faces),3,np.int32)))
        shape.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(clip_faces.ravel()))
        shape.CreateSubdivisionSchemeAttr("none");shape.CreateDisplayColorAttr([(0.55,0.43,0.16)])
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim()).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(shape.GetPrim()).CreateApproximationAttr("none")
        c=PhysxSchema.PhysxCollisionAPI.Apply(shape.GetPrim());c.CreateContactOffsetAttr(.00005);c.CreateRestOffsetAttr(0.)
        PhysxSchema.PhysxTriangleMeshCollisionAPI.Apply(shape.GetPrim()).CreateWeldToleranceAttr(0.)
        UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(material,UsdShade.Tokens.strongerThanDescendants,"physics")
        bound,_=UsdShade.MaterialBindingAPI(shape.GetPrim()).ComputeBoundMaterial("physics")
        if bound.GetPath()!=material.GetPath():raise RuntimeError("clip spring material overridden")
        clip_paths.append(path)
    socket.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(replacement.astype(np.float32)))
    # The untouched full Body collider remains for the hand and table. The
    # socket-specific source core supplies the same original pins to each clip.
    filtered=UsdPhysics.FilteredPairsAPI.Apply(original_body).CreateFilteredPairsRel()
    for path in clip_paths:filtered.AddTarget(path)
    for path,old_material in prior.items():
        bound,_=UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(path)).ComputeBoundMaterial("physics")
        if str(bound.GetPath())!=old_material:raise RuntimeError("original socket material changed")
    after={path:{n:str(stage.GetPrimAtPath(path).GetAttribute(n).Get()) for n in mass_names} for path in parts}
    if before!=after:raise RuntimeError("plug mass properties changed")
    record={"scope":"REPRESENTATIVE_FIXED_SOCKET_INTERIOR_WITH_FINITE_PASSIVE_CONTACTS",
            "manifest_path":str(manifest_path),"deep_cavity_mesh":manifest["deep_cavity_mesh"],
            "clip_paths":clip_paths,"contact_count":128,"contact_law":law,
            "source_socket_exterior_and_original_topology_preserved":True,
            "source_shared_vertices_retained":len(original),
            "recipe_triangle_soup_vertex_count":len(vertices),
            "source_face_indices_unchanged":True,
            "source_plug_mass_inertia_and_geometry_unchanged":True,
            "new_world_joint_or_plug_support_added":False,"post_start_pose_writes":False,
            "source_full_body_pair_filtered_from_clips":True,"source_core_collides_with_clips":True,
            "exact_TE_internal_calibration":False,"full_assembly_validated":False}
    prepared["report"]["representative_mating_contacts"]=record
    recording=prepared["contact_recording"]
    recording["additional_required_contact_filter_paths"]=list(dict.fromkeys([
        *recording["additional_required_contact_filter_paths"],prepared["receptacle_collision_path"],*clip_paths]))
    recording["minimum_contact_records"]=8192
    prepared["scene"]["evidence_paths"]=tuple(dict.fromkeys([
        *prepared["scene"]["evidence_paths"],manifest_path,deep_manifest_path,
        Path(manifest["deep_cavity_mesh"]),Path(manifest["clip_geometry_npz"]),Path(__file__).resolve()]))
    return record


def representative_clip_mesh(*, pin_radius_m=.00037211, bore_radius_m=.00046355,
                             radial_interference_m=.00005, leaf_count=6,
                             leaf_width_m=.00018, mouth_z_m=-.00120015,
                             entry_slope=.675, contact_band_length_m=.0002):
    if not (0 < radial_interference_m < pin_radius_m < bore_radius_m
            and 0 < leaf_width_m < pin_radius_m and entry_slope > 0
            and contact_band_length_m > 0 and leaf_count == 6):
        raise ValueError("invalid bounded representative six-leaf contact geometry")
    inner=pin_radius_m-radial_interference_m
    entry=bore_radius_m-.00002
    knee=mouth_z_m-(entry-inner)/entry_slope
    bottom=knee-contact_band_length_m
    section=np.array([[bore_radius_m,mouth_z_m],[entry,mouth_z_m],
                      [inner,knee],[inner,bottom],[bore_radius_m,bottom]])
    local=np.array([[r,y,z] for y in [-leaf_width_m/2,leaf_width_m/2] for r,z in section])
    hull=ConvexHull(local*1000.)
    triangles=hull.simplices.copy()
    for i,t in enumerate(triangles):
        n=np.cross(local[t[1]]-local[t[0]],local[t[2]]-local[t[0]])
        if n@hull.equations[i,:3]<0:triangles[i]=t[[0,2,1]]
    vertices=[];faces=[]
    for index in range(leaf_count):
        angle=2*np.pi*index/leaf_count;c,s=np.cos(angle),np.sin(angle)
        R=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
        faces.append(triangles+len(vertices)*len(local));vertices.append(local@R.T)
    report={"scope":"REPRESENTATIVE_COMPLIANT_CONTACT_ENVELOPE_NOT_TE_INTERNAL_CAD",
            "leaf_count":leaf_count,"pin_radius_m":pin_radius_m,
            "original_bore_radius_m":bore_radius_m,"radial_interference_m":radial_interference_m,
            "leaf_width_m":leaf_width_m,"mouth_z_m":mouth_z_m,
            "contact_band_start_z_m":knee,"contact_band_end_z_m":bottom,
            "entry_slope":entry_slope,"contact_band_length_m":contact_band_length_m,
            "geometry_parameters_manufacturer_calibrated":False,
            "source_pin_or_exterior_geometry_modified":False,
            "mechanism_reference":"PRECI-DIP AS39029 reversed BeCu clip, six or eight contact fingers",
            "force_reference_role":"Other-manufacturer historical measured means, not exact TE calibration"}
    return np.concatenate(vertices),np.concatenate(faces),report
