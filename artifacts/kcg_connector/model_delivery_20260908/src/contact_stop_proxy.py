"""Declared convex stop-sector candidate INSIDE the retained source metal.

Prototype only: no caller installs this automatically. Original CAD, keys,
contacts, body mass and existing articulation remain untouched. A later dynamic
lab test must establish whether this convex collision representation fixes the
observed source-SDF hard-stop miss; no joint/pose constraint is introduced.
"""
import numpy as np
from scipy.spatial import ConvexHull


def install_stop_boxes(stage, body_path, *, rigid_core_path, record):
    """Analytic boxes strictly inside the independently checked metal annulus.

    Native boxes avoid the observed cooking fallback that inflated the very
    small convex wedge into a tetrahedron. They introduce no outside volume.
    """
    import omni.timeline
    from pxr import Gf,PhysxSchema,UsdGeom,UsdPhysics,UsdShade
    timeline=omni.timeline.get_timeline_interface()
    if timeline.is_playing() or timeline.get_current_time()!=0.:
        raise RuntimeError('Hard-stop cells are authored before physics only')
    if not record.get('source_contains_check_passed'):
        raise ValueError('Source annular metal evidence required')
    from pathlib import Path
    import json
    depth_evidence=Path(__file__).resolve().parents[1]/'references/path_stop_box_depth1mm_evidence.json'
    evidence=json.loads(depth_evidence.read_text())
    if not evidence['boolean_result_empty'] or evidence['new_occupied_volume_outside_source']:
        raise ValueError('The deeper stop cells lack source-volume containment')
    radius=.0175; half_radial=.00012; half_tangent=.0008; depth=.001; number=32
    radial_min=radius-half_radial
    radial_max=float(np.hypot(radius+half_radial,half_tangent))
    if radial_min<record['inner_radius_m'] or radial_max>record['outer_radius_m'] or depth>evidence['only_proposed_change']['new_depth_m']:
        raise ValueError('Analytic boxes protrude from the verified source-metal annular prism')
    body=stage.GetPrimAtPath(body_path);core=stage.GetPrimAtPath(rigid_core_path)
    material,_=UsdShade.MaterialBindingAPI(core).ComputeBoundMaterial('physics')
    stiffness=material.GetPrim().GetAttribute('physxMaterial:compliantContactStiffness')
    if stiffness and stiffness.Get() not in (None,0.,0):raise ValueError('Hard stop needs a hard material')
    fields=('physics:mass','physics:centerOfMass','physics:diagonalInertia','physics:principalAxes')
    before={k:str(body.GetAttribute(k).Get()) for k in fields}
    relation=UsdPhysics.FilteredPairsAPI(core).GetFilteredPairsRel()
    excluded=relation.GetTargets() if relation else []
    paths=[]
    for index in range(number):
        angle=2*np.pi*index/number;path=f'{body_path}/SourceMetalStopBox{index:02d}'
        if stage.GetPrimAtPath(path):raise ValueError('Stop boxes already authored')
        shape=UsdGeom.Cube.Define(stage,path);shape.CreateSizeAttr(2.)
        shape.AddTranslateOp().Set(Gf.Vec3d(radius*np.cos(angle),radius*np.sin(angle),-depth/2))
        shape.AddOrientOp().Set(Gf.Quatf(float(np.cos(angle/2)),Gf.Vec3f(0.,0.,float(np.sin(angle/2)))))
        shape.AddScaleOp().Set(Gf.Vec3f(half_radial,half_tangent,depth/2))
        shape.CreateVisibilityAttr('invisible');prim=shape.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
        collision=PhysxSchema.PhysxCollisionAPI.Apply(prim)
        collision.CreateContactOffsetAttr(.00005);collision.CreateRestOffsetAttr(0.)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material,UsdShade.Tokens.strongerThanDescendants,'physics')
        UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel().SetTargets(excluded)
        resolved,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
        if resolved.GetPath()!=material.GetPath():raise RuntimeError('Stop box hard material overridden')
        paths.append(path)
    if before!={k:str(body.GetAttribute(k).Get()) for k in fields}:raise RuntimeError('Source mass/inertia changed')
    return {**record,'scope':'SOURCE_METAL_INSCRIBED_ANALYTIC_BOX_STOP_CANDIDATE',
            'representation':'NATIVE_BOX_GEOMETRY_NO_CONVEX_COOKING',
            'installed_paths':paths,'box_count':number,'center_radius_m':radius,
            'half_extents_m':[half_radial,half_tangent,depth/2],
            'analytic_radial_bounds_m':[radial_min,radial_max],
            'native_front_plane_body_z_m':0.,'hard_material':str(material.GetPath()),
            'geometry_containment_basis':'Analytic entire-box radial/z bounds lie inside the independently checked source-metal annular prism',
            'depth_containment_evidence':str(depth_evidence),
            'mass_inertia_unchanged':True,'source_rigid_core_retained':True,
            'candidate_only_not_installed':False,'dynamic_success':False}


def install_socket_stop_boxes(stage, *, socket_collision_path, body_box_paths, body_paths):
    """Use source-contained Box--Box stop contact, retaining all source metal."""
    import json
    from pathlib import Path
    from pxr import Gf,PhysxSchema,Sdf,Usd,UsdGeom,UsdPhysics,UsdShade
    evidence_path=Path(__file__).resolve().parents[1]/'references/path_socket_stop_boxes_1mm_evidence.json'
    evidence=json.loads(evidence_path.read_text())
    if not evidence['boolean_difference_empty'] or evidence['outside_volume_m3']!=0:
        raise ValueError('Socket boxes need whole-volume source containment')
    source=stage.GetPrimAtPath(socket_collision_path);parent=source.GetParent()
    if not np.allclose(np.asarray(UsdGeom.Xformable(source).ComputeLocalToWorldTransform(Usd.TimeCode.Default())),
                       np.asarray(UsdGeom.Xformable(parent).ComputeLocalToWorldTransform(Usd.TimeCode.Default())),atol=1e-12,rtol=0):
        raise ValueError('Verified Socket mesh frame must equal the authoring parent frame')
    material,_=UsdShade.MaterialBindingAPI(source).ComputeBoundMaterial('physics')
    body_boxes=set(body_box_paths)
    excluded=[p.GetPath() for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)
              and any(str(p.GetPath()).startswith(root+'/') for root in body_paths)
              and str(p.GetPath()) not in body_boxes]
    paths=[]
    for index in range(evidence['box_count']):
        angle=2*np.pi*index/evidence['box_count'];radius=evidence['box_center_radius_m']
        path=str(parent.GetPath())+f'/SourceMetalStopBox{index:02d}'
        if stage.GetPrimAtPath(path):raise ValueError('Socket stop box already exists')
        box=UsdGeom.Cube.Define(stage,path);box.CreateSizeAttr(2.)
        box.AddTranslateOp().Set(Gf.Vec3d(radius*np.cos(angle),radius*np.sin(angle),evidence['socket_local_center_z_m']))
        box.AddOrientOp().Set(Gf.Quatf(float(np.cos(angle/2)),Gf.Vec3f(0.,0.,float(np.sin(angle/2)))))
        box.AddScaleOp().Set(Gf.Vec3f(*evidence['half_extents_m']));box.CreateVisibilityAttr('invisible')
        prim=box.GetPrim();UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
        collision=PhysxSchema.PhysxCollisionAPI.Apply(prim)
        collision.CreateContactOffsetAttr(.00005);collision.CreateRestOffsetAttr(0.)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material,UsdShade.Tokens.strongerThanDescendants,'physics')
        UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel().SetTargets(excluded)
        paths.append(path)
    for path in body_box_paths:
        relation=UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(path)).CreateFilteredPairsRel()
        targets=list(relation.GetTargets());targets.append(Sdf.Path(socket_collision_path))
        relation.SetTargets(list(dict.fromkeys(targets)))
    return {'scope':'SOURCE_CONTAINED_SOCKET_BOXES_PAIR_WITH_EXISTING_BODY_BOXES_ONLY',
            'socket_pair_collision_paths':paths,'body_box_paths':body_box_paths,
            'body_boxes_original_socket_pair_disabled':socket_collision_path,
            'socket_boxes_exclude_other_body_and_nut_colliders':list(map(str,excluded)),
            'source_core_socket_contact_unchanged':True,'source_geometry_material_mass_unchanged':True,
            'source_containment_evidence':str(evidence_path),'not_a_new_joint_or_support':True}


def build_stop_sectors(*, inner_radius_m=.01730, outer_radius_m=.01770,
                       front_z_m=0., metal_depth_m=.00010, sectors=32):
    """Inscribed convex wedges; the inner chord never protrudes into the bore.

    Source evidence (full_path_03): Body front annulus z=0, radius
    17.106899..17.837151 mm; matching Socket z=-14.605 mm and radius
    17.224357..19.100801 mm. The selected annulus is within their overlap.
    Source-ray metal depth is >=7.6454 mm at the selected radii; the proposed
    0.10 mm extrusion goes ONLY behind the original front face into that metal.
    """
    if sectors < 8 or not 0 < inner_radius_m < outer_radius_m or metal_depth_m <= 0:
        raise ValueError('Finite annular metal sector dimensions required')
    half=np.pi/sectors
    inner_vertex_radius=inner_radius_m/np.cos(half)
    if inner_vertex_radius>=outer_radius_m:raise ValueError('Too few sectors for this annulus width')
    vertices=[];triangles=[]
    for index in range(sectors):
        center=2*np.pi*index/sectors
        corners=np.array([[inner_vertex_radius*np.cos(center-half),inner_vertex_radius*np.sin(center-half)],
                          [outer_radius_m*np.cos(center-half),outer_radius_m*np.sin(center-half)],
                          [outer_radius_m*np.cos(center+half),outer_radius_m*np.sin(center+half)],
                          [inner_vertex_radius*np.cos(center+half),inner_vertex_radius*np.sin(center+half)]])
        v=np.array([[x,y,z] for z in (front_z_m-metal_depth_m,front_z_m) for x,y in corners])
        hull=ConvexHull(v*1000.);f=hull.simplices.copy()
        for j,t in enumerate(f):
            if np.cross(v[t[1]]-v[t[0]],v[t[2]]-v[t[0]])@hull.equations[j,:3]<0:f[j]=t[[0,2,1]]
        vertices.append(v);triangles.append(f)
    return np.stack(vertices),np.stack(triangles),{
        'scope':'SOURCE_METAL_INSCRIBED_CONVEX_HARD_STOP_CANDIDATE_NOT_DYNAMICALLY_VALIDATED',
        'inner_radius_m':inner_radius_m,'inner_vertex_radius_m':float(inner_vertex_radius),
        'outer_radius_m':outer_radius_m,'front_z_m':front_z_m,'metal_depth_m':metal_depth_m,'sectors':sectors,
        'source_front_plane_m':0.,'source_body_stop_radial_span_m':[.017106899,.017837151],
        'source_socket_stop_plane_m':-.014605,'source_socket_stop_radial_span_m':[.017224357,.019100801],
        'inner_chords_outside_original_bore':True,'outer_chords_inside_source_outer_envelope':True,
        'source_mass_or_geometry_replaced':False,'extra_joint_or_post_start_pose_write':False,
        'physical_scope':'Supplemental hard collision cells contained inside existing source CAD metal. No new occupied external volume; original source SDF remains.',
        'source_contains_check_required':True,'dynamic_validation_required':True,
    }


def install_stop_sectors(stage, body_path, *, rigid_core_path, vertices, faces, record):
    """Explicit pre-physics installation of an already checked candidate.

    Caller supplies offline-approved geometry. The original full Body/source
    rigid core are not disabled or edited. Their old non-socket pair filtering
    is copied so the new cells do not create duplicate robot/table contacts.
    """
    import omni.timeline
    from pxr import PhysxSchema,PhysicsSchemaTools,UsdGeom,UsdPhysics,UsdShade,UsdUtils,Vt
    from omni.physx import get_physx_cooking_interface
    timeline=omni.timeline.get_timeline_interface()
    if timeline.is_playing() or timeline.get_current_time()!=0.:
        raise RuntimeError('Hard-stop collision authoring is pre-physics only')
    if not record.get('source_contains_check_passed'):
        raise ValueError('Explicit source-metal containment evidence required')
    body=stage.GetPrimAtPath(body_path);core=stage.GetPrimAtPath(rigid_core_path)
    if not body.HasAPI(UsdPhysics.RigidBodyAPI) or not core.HasAPI(UsdPhysics.CollisionAPI):
        raise ValueError('Existing dynamic Body and unchanged source rigid collider required')
    material,_=UsdShade.MaterialBindingAPI(core).ComputeBoundMaterial('physics')
    if not material or not material.GetPrim().HasAPI(UsdPhysics.MaterialAPI):raise ValueError('Existing hard physical material required')
    stiffness=material.GetPrim().GetAttribute('physxMaterial:compliantContactStiffness')
    if stiffness and stiffness.Get() not in (None,0.,0):raise ValueError('The stop must use the hard material, not seal/band compliance')
    mass_names=('physics:mass','physics:centerOfMass','physics:diagonalInertia','physics:principalAxes')
    before={name:str(body.GetAttribute(name).Get()) for name in mass_names}
    filter_rel=UsdPhysics.FilteredPairsAPI(core).GetFilteredPairsRel()
    non_socket=filter_rel.GetTargets() if filter_rel else []
    paths=[];cooked_checks=[]
    stage_id=UsdUtils.StageCache.Get().Insert(stage).ToLongInt()
    cooking=get_physx_cooking_interface()
    for index,(v,f) in enumerate(zip(vertices,faces)):
        path=f'{body_path}/SourceMetalStopSector{index:02d}'
        if stage.GetPrimAtPath(path):raise ValueError('Stop proxy already present')
        mesh=UsdGeom.Mesh.Define(stage,path);prim=mesh.GetPrim()
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(v,np.float32)))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(f),3,np.int32)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(f,np.int32).ravel()))
        mesh.CreateSubdivisionSchemeAttr('none');mesh.CreateVisibilityAttr('invisible')
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('convexHull')
        # USD's default 1 mm minimum thickness would inflate this 0.1 mm
        # source-metal cell beyond the original stop plane. Disable inflation.
        hull=PhysxSchema.PhysxConvexHullCollisionAPI.Apply(prim)
        hull.CreateMinThicknessAttr(0.)
        hull.CreateHullVertexLimitAttr(64)
        collider=PhysxSchema.PhysxCollisionAPI.Apply(prim)
        collider.CreateContactOffsetAttr(.00005);collider.CreateRestOffsetAttr(0.)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material,UsdShade.Tokens.strongerThanDescendants,'physics')
        UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel().SetTargets(non_socket)
        resolved,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
        if resolved.GetPath()!=material.GetPath():raise RuntimeError('Hard-stop material overridden')
        received={}
        def on_cooked(status,convexes):
            received['status']=str(status)
            received['vertices']=[np.asarray([list(p) for p in c.vertices],dtype=np.float64) for c in convexes]
        cooking.request_convex_collision_representation(stage_id,PhysicsSchemaTools.sdfPathToInt(prim.GetPath()),False,on_cooked)
        if len(received.get('vertices',[]))!=1:raise RuntimeError('Source stop cell did not cook to one convex')
        actual=received['vertices'][0];planes=ConvexHull(np.asarray(v,np.float64)).equations
        excess=float(np.max(actual@planes[:,:3].T+planes[:,3]))
        if excess>1e-7 or float(actual[:,2].max())>float(record['front_z_m'])+1e-8:
            raise RuntimeError('Convex cooking differs from source metal: '+str({
                'index':index,'plane_excess_m':excess,'authored_bounds_m':[np.min(v,axis=0).tolist(),np.max(v,axis=0).tolist()],
                'native_bounds_m':[actual.min(0).tolist(),actual.max(0).tolist()],
                'native_vertices':actual.tolist()}))
        cooked_checks.append({'path':path,'native_vertex_count':len(actual),
                              'maximum_authored_convex_plane_excess_m':excess,
                              'native_bounds_m':[actual.min(0).tolist(),actual.max(0).tolist()],
                              'minimum_thickness_m':0.})
        paths.append(path)
    after={name:str(body.GetAttribute(name).Get()) for name in mass_names}
    if before!=after:raise RuntimeError('Source mass properties changed')
    return {**record,'installed_paths':paths,'hard_material':str(material.GetPath()),
            'candidate_only_not_installed':False,'native_cooking_checks':cooked_checks,
            'mass_inertia_unchanged':True,'source_rigid_core_retained':True,
            'dynamic_success':False}
