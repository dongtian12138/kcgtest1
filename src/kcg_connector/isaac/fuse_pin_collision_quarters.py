"""Combine each source pin's four convex quarters, before CPU physics starts.

The four closed convex sets occupy disjoint quadrants. Their summed volume is
checked against the convex hull of their union; source vertices are brought to
one local frame by signed quarter-turns, with world-coordinate error checked.
Materials and the symmetric exclusion graph are
checked before authoring. This changes manifold discretization, so contact
forces and complete assembly acceptance must be revalidated independently.
"""
import re


def fuse_pin_quarters(stage, *, validate_only=False):
    import numpy as np
    from scipy.spatial import ConvexHull
    from pxr import Sdf, UsdGeom, UsdPhysics, UsdShade, Vt

    pattern = re.compile(r'(.*/OfficialPinWholeQuarter_(\d{3}))_([0-3])$')
    pins = {}
    for prim in stage.Traverse():
        match = pattern.fullmatch(str(prim.GetPath()))
        if match:
            pins.setdefault(int(match[2]), {})[int(match[3])] = prim
    if set(pins) != set(range(128)) or any(set(v) != {0,1,2,3} for v in pins.values()):
        raise ValueError('The declared source128pins with four quarters each are required')

    def canonical(path):
        return pattern.sub(r'\1_0', str(path))

    relationships = []
    for prim in stage.Traverse():
        rel = prim.GetRelationship('physics:filteredPairs')
        if not rel:
            continue
        original = rel.GetTargets()
        masks = {}
        for target in original:
            match = pattern.fullmatch(str(target))
            if match:
                masks.setdefault(match[1], set()).add(int(match[3]))
        if any(mask != {0,1,2,3} for mask in masks.values()):
            raise ValueError('A source exclusion distinguishes individual pin quarters')
        mapped = list(dict.fromkeys(Sdf.Path(canonical(p)) for p in original))
        if mapped != original:
            relationships.append((rel, original, mapped))

    plans = []
    max_volume_relative_error = 0.
    maximum_vertex_displacement_m = 0.
    for number, quarters in sorted(pins.items()):
        meshes = [UsdGeom.Mesh(quarters[i]) for i in range(4)]
        source_vertices = [np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64) for mesh in meshes]
        transforms = [np.asarray(UsdGeom.Xformable(quarters[i]).GetLocalTransformation()) for i in range(4)]
        if not all(np.array_equal(t[3],transforms[0][3]) for t in transforms):
            raise ValueError('Source pin quarters must share their local origin')
        vertices = []
        pin_vertex_error = 0.
        for points, transform in zip(source_vertices, transforms):
            rotation = transform[:3,:3] @ np.linalg.inv(transforms[0][:3,:3])
            quarter_turn = np.rint(rotation)
            if (not np.allclose(rotation,quarter_turn,rtol=0,atol=1e-14)
                    or not np.array_equal(quarter_turn@quarter_turn.T,np.eye(3))
                    or np.linalg.det(quarter_turn)!=1.):
                raise ValueError('Only exact signed quarter-turn coordinate changes are supported')
            mapped = points @ quarter_turn
            original_world = (np.c_[points,np.ones(len(points))] @ transform)[:,:3]
            mapped_world = (np.c_[mapped,np.ones(len(mapped))] @ transforms[0])[:,:3]
            error = float(np.max(np.linalg.norm(mapped_world-original_world,axis=1)))
            if error>1e-12:
                raise ValueError('Pin vertex world-coordinate displacement exceeds1pm')
            pin_vertex_error=max(pin_vertex_error,error)
            vertices.append(mapped)
        maximum_vertex_displacement_m=max(maximum_vertex_displacement_m,pin_vertex_error)
        filters = []
        materials = []
        collision_attributes = []
        for i, prim in quarters.items():
            if (not prim.HasAPI(UsdPhysics.CollisionAPI)
                    or UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not True
                    or prim.GetAttribute('physics:approximation').Get() != 'convexHull'
                    or prim.GetChildren() or prim.HasAPI(UsdPhysics.RigidBodyAPI)):
                raise ValueError('Only active source convex collision leaves may be combined')
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
            materials.append(str(material.GetPath()))
            collision_attributes.append({a.GetName(): str(a.Get()) for a in prim.GetAttributes()
                                         if a.GetName().startswith('physxCollision:')})
            filters.append(frozenset(canonical(t) for t in prim.GetRelationship('physics:filteredPairs').GetTargets()))
            sx, sy = ((1,1),(-1,1),(-1,-1),(1,-1))[i]
            # This partition proves interiors do not overlap. It also rejects
            # a nonconvex stepped/grooved pin instead of filling its cavities.
            if np.min(vertices[i][:,0]*sx) < -1e-12 or np.min(vertices[i][:,1]*sy) < -1e-12:
                raise ValueError('Source quarters do not occupy disjoint declared quadrants')
        if len(set(materials)) != 1 or not materials[0]:
            raise ValueError('Source quarter materials differ')
        if any(f != filters[0] for f in filters) or any(a != collision_attributes[0] for a in collision_attributes):
            raise ValueError('Source quarter collision semantics differ')
        combined = np.concatenate(vertices)
        hulls = [ConvexHull(v) for v in vertices]
        hull = ConvexHull(combined)
        original_volume = sum(h.volume for h in hulls)
        relative_error = abs(hull.volume-original_volume)/original_volume
        max_volume_relative_error = max(max_volume_relative_error, relative_error)
        if relative_error > 1e-10 or len(hull.vertices)>255:
            raise ValueError('Pin union is not the same supported convex solid')
        if not np.array_equal(combined.astype(np.float32).astype(np.float64),combined):
            raise ValueError('Combining source vertices would change a source coordinate')
        faces = hull.simplices.copy()
        normals = np.cross(combined[faces[:,1]]-combined[faces[:,0]],combined[faces[:,2]]-combined[faces[:,0]])
        reverse = np.einsum('ij,ij->i',normals,hull.equations[:,:3])<0
        faces[reverse] = faces[reverse][:,[0,2,1]]
        plans.append((number,quarters,meshes[0],combined,faces,{
            'pin':number,'retained_collider':str(quarters[0].GetPath()),
            'source_quarter_volumes_local_units3':[h.volume for h in hulls],
            'union_hull_volume_local_units3':hull.volume,'volume_relative_error':relative_error,
            'maximum_authored_world_vertex_displacement_m':pin_vertex_error,
            'hull_vertex_count':len(hull.vertices),'material':materials[0]}))
    report = {'scope':'CPU_SOURCE_PIN_CONVEX_PARTITION_FUSION_REQUIRES_PHYSICAL_REVALIDATION',
              'source_pin_count':len(plans),'active_pin_colliders_before':4*len(plans),
              'active_pin_colliders_after_if_applied':len(plans),
              'maximum_authored_world_vertex_displacement_m':maximum_vertex_displacement_m,
              'source_files_saved':False,'mass_inertia_material_or_pose_authored':False,
              'source_exclusions_preserved_under_quarter_identification':True,
              'maximum_union_volume_relative_error':max_volume_relative_error,
              'manifold_sampling_changed':not validate_only,'runtime_fusion_applied':not validate_only,
              'original_assembly_acceptance_transferred':False,'pins':[row[-1] for row in plans]}
    if validate_only:
        return report
    from pxr import PhysxSchema
    # No authoring happens until every geometry and filtering check has passed.
    for rel,original,mapped in relationships:
        if rel.GetTargets()!=original:
            raise RuntimeError('Source exclusions changed during pin validation')
        rel.SetTargets(mapped)
    for _, quarters, mesh, vertices, faces, _ in plans:
        mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32)))
        mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.full(len(faces),3,np.int32)))
        mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(faces.astype(np.int32).ravel()))
        mesh.CreateSubdivisionSchemeAttr('none')
        # The source four56-vertex pieces fit GPU's64-vertex limit. The exact
        # whole pin has180vertices; CPU PhysX supports the explicit255cap.
        PhysxSchema.PhysxConvexHullCollisionAPI.Apply(quarters[0]).CreateHullVertexLimitAttr(255)
        for i in (1,2,3):
            UsdPhysics.CollisionAPI(quarters[i]).GetCollisionEnabledAttr().Set(False)
    return report


def fuse_pin_shafts(stage, *, validate_only=False):
    """Exact axial partition: one long shaft and the original four nose sectors."""
    import numpy as np
    from scipy.spatial import ConvexHull
    from pxr import Sdf, UsdGeom, UsdPhysics, Vt

    proof=fuse_pin_quarters(stage,validate_only=True)
    plans=[]
    for pin in proof['pins']:
        root=pin['retained_collider'].rsplit('_',1)[0]
        prims=[stage.GetPrimAtPath(f'{root}_{i}') for i in range(4)]
        source=[np.asarray(UsdGeom.Mesh(p).GetPointsAttr().Get(),float) for p in prims]
        matrices=[np.asarray(UsdGeom.Xformable(p).GetLocalTransformation()) for p in prims]
        shaft_quarters=[];tips=[];volumes=[]
        for i,v in enumerate(source):
            levels=np.unique(v[:,2])
            if len(levels)<3:raise ValueError('Source pin has no distinct cylindrical shaft and rounded nose')
            lo,split=levels[:2]
            bottom=v[v[:,2]==lo];shoulder=v[v[:,2]==split]
            bottom_xy=np.unique(bottom[np.linalg.norm(bottom[:,:2],axis=1)>1e-12,:2],axis=0)
            shoulder_xy=np.unique(shoulder[np.linalg.norm(shoulder[:,:2],axis=1)>1e-12,:2],axis=0)
            if not np.array_equal(bottom_xy,shoulder_xy):
                raise ValueError('The selected source interval is not a constant-section shaft')
            lower=np.vstack((bottom,shoulder,[0.,0.,split]))
            tip=np.vstack((v[v[:,2]>=split],[0.,0.,split]))
            original=ConvexHull(v).volume
            a,b=ConvexHull(lower),ConvexHull(tip)
            if abs(a.volume+b.volume-original)/original>1e-10:
                raise ValueError('Axial partition changes a source quarter solid')
            rotation=np.rint(matrices[i][:3,:3]@np.linalg.inv(matrices[0][:3,:3]))
            shaft_quarters.append(lower@rotation)
            tips.append(tip);volumes.append(a.volume)
        shaft=np.unique(np.concatenate(shaft_quarters),axis=0)
        hull=ConvexHull(shaft)
        if (len(hull.vertices)>64 or abs(hull.volume-sum(volumes))/sum(volumes)>1e-10
                or any(len(ConvexHull(v).vertices)>64 for v in tips)):
            raise ValueError('Exact source shaft/nose partition exceeds the native64vertex limit')
        geometries=[*tips,shaft]
        for v in geometries:
            if not np.array_equal(v.astype(np.float32).astype(float),v):
                raise ValueError('Axial partition introduces coordinate quantization')
        paths=[f'{root}_{i}' for i in range(5)]
        plans.append((prims,paths,geometries))
        pin.update(cooked_colliders=paths,shaft_interval_local=[float(lo),float(split)],
                   native_hull_input_vertex_counts=[len(ConvexHull(v).vertices) for v in geometries])
    proof.update(scope='CPU_EXACT_SOURCE_PIN_SHAFT_AND_NOSE_PARTITION_REQUIRES_REVALIDATION',
                 active_pin_colliders_after_if_applied=5*128,
                 active_long_shaft_colliders_before=4*128,active_long_shaft_colliders_after_if_applied=128,
                 native_convex_vertex_cap=64,runtime_fusion_applied=not validate_only,
                 manifold_sampling_changed=not validate_only,
                 axial_partition_checked_for_all128pins=True)
    if validate_only:return proof
    pattern=re.compile(r'(.*/OfficialPinWholeQuarter_\d{3})_0$')
    for prim in stage.Traverse():
        rel=prim.GetRelationship('physics:filteredPairs')
        if rel:
            original=rel.GetTargets();mapped=[]
            for target in original:
                mapped.append(target)
                match=pattern.fullmatch(str(target))
                if match:mapped.append(Sdf.Path(match[1]+'_4'))
            if mapped!=original:rel.SetTargets(list(dict.fromkeys(mapped)))
    # Copy the original quarter's material, collision settings and exact local
    # frame before replacing its geometry; only the runtime layer is authored.
    source_layer=stage.Flatten();target_layer=stage.GetEditTarget().GetLayer()
    for prims,paths,geometries in plans:
        if stage.GetPrimAtPath(paths[4]):raise ValueError('Source shaft destination already exists')
        if not Sdf.CopySpec(source_layer,prims[0].GetPath(),target_layer,Sdf.Path(paths[4])):
            raise RuntimeError('Could not preserve source pin collision properties')
        for path,v in zip(paths,geometries):
            hull=ConvexHull(v);faces=hull.simplices.copy()
            normals=np.cross(v[faces[:,1]]-v[faces[:,0]],v[faces[:,2]]-v[faces[:,0]])
            reverse=np.einsum('ij,ij->i',normals,hull.equations[:,:3])<0
            faces[reverse]=faces[reverse][:,[0,2,1]]
            mesh=UsdGeom.Mesh(stage.GetPrimAtPath(path))
            mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(v.astype(np.float32)))
            mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.full(len(faces),3,np.int32)))
            mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(faces.astype(np.int32).ravel()))
    return proof


def inspect_cooked_pin_hulls(stage, fusion_report):
    """Record actual SDK convex cooking, without stepping or rewriting physics."""
    import numpy as np
    from scipy.spatial import ConvexHull
    from pxr import PhysicsSchemaTools, UsdGeom, UsdUtils
    from omni.physx import get_physx_cooking_interface
    from omni.physx.bindings._physx import PhysxCollisionRepresentationResult

    cooking = get_physx_cooking_interface()
    stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
    result = {'scope':'READ_ONLY_SDK_COOKED_FUSED_PIN_GEOMETRY', 'pins':[],
              'maximum_source_outside_cooked_plane_m':0.,
              'maximum_cooked_outside_source_plane_m':0.}
    for pin,path in [(p,path) for p in fusion_report['pins']
                     for path in p.get('cooked_colliders',[p['retained_collider']])]:
        captured = {}
        def receive(status, hulls):
            captured['status'] = status
            captured['hulls'] = [np.asarray([[p.x,p.y,p.z] for p in hull.vertices],dtype=float)
                                 for hull in hulls]
        cooking.request_convex_collision_representation(stage_id=stage_id,
            collision_prim_id=PhysicsSchemaTools.sdfPathToInt(path),
            run_asynchronously=False,on_result=receive)
        if (captured.get('status') != PhysxCollisionRepresentationResult.RESULT_VALID
                or len(captured.get('hulls',[])) != 1):
            raise RuntimeError('Fused source pin did not cook into exactly one native convex hull')
        prim = stage.GetPrimAtPath(path)
        source = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(),dtype=float)
        cooked = captured['hulls'][0]
        transform = np.asarray(UsdGeom.Xformable(prim).GetLocalTransformation())
        scales = np.linalg.norm(transform[:3,:3],axis=1)
        if not np.allclose(scales,scales[0],rtol=0,atol=1e-14):
            raise RuntimeError('Cooked pin comparison requires the declared uniform source scale')
        a,b = ConvexHull(source),ConvexHull(cooked)
        outside_a=max(0.,float(np.max(source@b.equations[:,:3].T+b.equations[:,3])))*scales[0]
        outside_b=max(0.,float(np.max(cooked@a.equations[:,:3].T+a.equations[:,3])))*scales[0]
        result['maximum_source_outside_cooked_plane_m']=max(result['maximum_source_outside_cooked_plane_m'],outside_a)
        result['maximum_cooked_outside_source_plane_m']=max(result['maximum_cooked_outside_source_plane_m'],outside_b)
        result['pins'].append({'pin':pin['pin'],'collider':path,'native_vertex_count':len(cooked),
                               'native_vertices_local':cooked.tolist(),
                               'source_outside_cooked_plane_m':outside_a,
                               'cooked_outside_source_plane_m':outside_b})
    result['all_declared_native_hulls_read']=len(result['pins'])==sum(
        len(p.get('cooked_colliders',[p['retained_collider']])) for p in fusion_report['pins'])
    return result
