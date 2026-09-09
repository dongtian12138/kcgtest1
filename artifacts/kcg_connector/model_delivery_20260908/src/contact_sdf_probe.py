"""Small read-only native SDF vs declared mesh check; no simulation startup.

Call run_native_sdf_probe(stage, active_physics_sim_view) between steps AFTER
reset. The validated installed tensor layout is gradientXYZ, distance, despite
the installed high-level docstring saying distance first. Offset samples retain
an explicit layout/normal check. No resolution, geometry or state is changed.
"""
import numpy as np

BODY_CORE='/World/TE_J35FreeSplitPlug/Body/SocketRigidCoreCollision'
NUT='/World/TE_J35FreeSplitPlug/CouplingNut/SourceCadCollision'
# Keep the original five indices for shallow comparisons, then add exactly
# the three requested inward depths. This is a geometry query, not a sweep.
OFFSETS=np.array([-50e-6,-20e-6,0.,20e-6,50e-6,-100e-6,-300e-6,-700e-6])


def _mesh_faces(stage,path):
    from pxr import UsdGeom
    mesh=UsdGeom.Mesh.Get(stage,path)
    if not mesh:raise ValueError(f'Missing existing source collision mesh: {path}')
    vertices=np.asarray(mesh.GetPointsAttr().Get(),float)
    counts=np.asarray(mesh.GetFaceVertexCountsAttr().Get(),int)
    if not np.all(counts==3):raise ValueError('Current source mesh must remain triangular')
    faces=np.asarray(mesh.GetFaceVertexIndicesAttr().Get(),int).reshape(-1,3)
    triangles=vertices[faces];centers=triangles.mean(1)
    normal=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    length=np.linalg.norm(normal,axis=1);valid=length>1e-18
    normal[valid]/=length[valid,None]
    return vertices,faces,triangles,centers,normal,valid


def prepare_mesh_normal_samples(stage,*,body_core_path=BODY_CORE,nut_collision_path=NUT):
    """12 source stop triangles and up to24 inner thread-flank triangles.

    Selects geometry only: no loaded position, force, outcome or parameter scan.
    Each source face centroid gets -50,-20,0,+20,+50,-100,-300,-700 um.
    Negative offsets point into the source material along the outward normal.
    Face-normal offsets are local tangent-plane expectations, not a claim that
    the same face remains nearest across sharp corners. Native data is retained.
    """
    records=[]
    for kind,path in [('body_axial_stop',body_core_path),('nut_inner_thread_flanks',nut_collision_path)]:
        v,f,tri,c,n,valid=_mesh_faces(stage,path)
        radius=np.linalg.norm(c[:,:2],axis=1)
        if kind=='body_axial_stop':
            mask=valid&(np.abs(tri[:,:,2]).max(1)<5e-8)&(n[:,2]>.999)&(radius>.01710)&(radius<.01785)
            pool=np.flatnonzero(mask)
            if len(pool)<12:raise ValueError('The verified source z=0 Body stop ring is missing')
            angle=np.arctan2(c[pool,1],c[pool,0]);pool=pool[np.argsort(angle)]
            selected=pool[np.linspace(0,len(pool)-1,12,dtype=int)]
            selection='Exact source z=0 face, outward+Z, original17.1069..17.8372mm annulus;12 angular samples'
        else:
            radial_dot=(n[:,:2]*c[:,:2]).sum(1)/np.maximum(radius,1e-12)
            mask=valid&(radius>.0199640)&(radius<.0208160)&(radial_dot<-.2)&(abs(n[:,2])>.25)&(abs(n[:,2])<.95)
            pool=np.flatnonzero(mask)
            if len(pool)<24:raise ValueError('Declared inner-thread flank faces could not be located')
            selected=[]
            # Opposing flanks, three axial bands, four angular samples each.
            z_edges=np.quantile(c[pool,2],[0,1/3,2/3,1])
            for sign in (-1,1):
                for band in range(3):
                    candidates=pool[(np.sign(n[pool,2])==sign)&(c[pool,2]>=z_edges[band])&(c[pool,2]<=z_edges[band+1])]
                    if not len(candidates):continue
                    angle=np.arctan2(c[candidates,1],c[candidates,0]);candidates=candidates[np.argsort(angle)]
                    selected.extend(candidates[np.linspace(0,len(candidates)-1,min(4,len(candidates)),dtype=int)].tolist())
            selected=np.asarray(list(dict.fromkeys(selected)),int)
            selection='Source inner flank radius19.964..20.816mm, inward radial normal, both axial normal signs,3 source-z bands x4 angles'
        surface=c[selected];directions=n[selected]
        points=(surface[:,None,:]+OFFSETS[None,:,None]*directions[:,None,:]).reshape(-1,3)
        records.append({'kind':kind,'collision_path':path,'source_face_indices':selected,
                        'source_centroids_mesh_local_m':surface,'source_outward_normals_mesh_local':directions,
                        'normal_offsets_m':OFFSETS.copy(),'points_mesh_local_m':points,'selection':selection,
                        'source_bounds_m':np.array([v.min(0),v.max(0)]),
                        'mesh_sdf_resolution':stage.GetPrimAtPath(path).GetAttribute('physxSDFMeshCollision:sdfResolution').Get(),
                        'mesh_sdf_narrow_band_thickness':stage.GetPrimAtPath(path).GetAttribute('physxSDFMeshCollision:sdfNarrowBandThickness').Get(),
                        'mesh_sdf_subgrid_resolution':stage.GetPrimAtPath(path).GetAttribute('physxSDFMeshCollision:sdfSubgridResolution').Get()})
    return records



def exact_mesh_reference(stage,path,points):
    """Closest points and signed distance of the actual source triangles.

    The nearest triangle may switch when entering a thread tooth or reaching
    another side of the annulus. We therefore never use the normal offset as
    the exact signed distance. Gradient is the outward closest-point direction
    where differentiable; at an edge/medial tie it is one valid local reference,
    not proof that a different native gradient is wrong.
    """
    import trimesh
    from pxr import UsdGeom
    mesh_prim=UsdGeom.Mesh.Get(stage,path)
    v=np.asarray(mesh_prim.GetPointsAttr().Get(),float)
    f=np.asarray(mesh_prim.GetFaceVertexIndicesAttr().Get(),int).reshape(-1,3)
    # Thread triangles are tiny in metre coordinates; use millimetres only
    # inside the offline geometric oracle to avoid closest-point tolerance
    # misclassification of their face interiors. Native SDF inputs stay metres.
    oracle_scale=1000.
    mesh=trimesh.Trimesh(vertices=v*oracle_scale,faces=f,process=False)
    nearest,unsigned,nearest_ids=trimesh.proximity.closest_point(mesh,points*oracle_scale)
    nearest/=oracle_scale;unsigned/=oracle_scale
    closed=bool(mesh.is_watertight and mesh.is_winding_consistent and mesh.volume>0)
    near_surface=unsigned<=1e-9
    signed=None;gradient=None;inside=None
    if closed:
        inside=mesh.contains(points*oracle_scale)
        sign=np.where(inside,-1.,1.)
        signed=unsigned*sign;signed[near_surface]=0.
        gradient=sign[:,None]*(points-nearest)/np.maximum(unsigned[:,None],1e-30)
        gradient[near_surface]=mesh.face_normals[nearest_ids[near_surface]]
    return {'watertight_consistent_positive_volume':closed,
            'closest_points_mesh_local_m':nearest,
            'nearest_source_face_indices':nearest_ids,
            'unsigned_distance_m':unsigned,'signed_distance_m':signed,
            'inside_source_mesh':inside,'outward_gradient_reference':gradient,
            'reference_surface_tolerance_m':1e-9,'offline_oracle_coordinate_scale':oracle_scale,
            'sign_convention':'negative inside, positive outside; nearest-surface distance, not normal travel',
            'gradient_limitation':'Nearest-surface gradient is not unique on medial axes/edges; triangle switches are explicitly recorded.'}


def _host(value):
    if hasattr(value,'detach'):return value.detach().cpu().numpy()
    if hasattr(value,'numpy'):return value.numpy()
    return np.asarray(value)


def _frontend_array(physics_sim_view,points,device):
    name=type(physics_sim_view._frontend).__name__.lower()
    if 'warp' in name:
        import warp as wp
        return wp.array(points.astype(np.float32),dtype=wp.float32,device=device)
    if 'torch' in name:
        import torch
        return torch.as_tensor(points,dtype=torch.float32,device=device)
    if 'numpy' in name:return points.astype(np.float32)
    raise ValueError(f'Unsupported explicit tensor frontend: {name}')


def run_native_sdf_probe(stage,physics_sim_view,*,body_core_path=BODY_CORE,
                         nut_collision_path=NUT,device='cuda:0'):
    """Return JSON-compatible raw values and local errors without stepping.

    Intended input: existing World.physics_sim_view (Torch) OR
    SimulationManager._physics_sim_view__warp. Create no new SimulationApp,
    World or SimulationView. Bind only existing shapes and actors.
    """
    samples=prepare_mesh_normal_samples(stage,body_core_path=body_core_path,nut_collision_path=nut_collision_path)
    actors=[str(stage.GetPrimAtPath(p).GetParent().GetPath()) for p in (body_core_path,nut_collision_path)]
    actor_view=physics_sim_view.create_rigid_body_view(actors)
    before=_host(actor_view.get_transforms()).copy()
    output=[]
    for record in samples:
        points=record['points_mesh_local_m'].astype(np.float32).astype(np.float64)
        reference=exact_mesh_reference(stage,record['collision_path'],points)
        view=physics_sim_view.create_sdf_shape_view(record['collision_path'],len(points))
        if view.count!=1:raise RuntimeError('Native SDF did not bind exactly the requested existing shape')
        raw=_host(view.get_sdf_and_gradients(_frontend_array(physics_sim_view,points[None],device))).copy()[0]
        if raw.shape!=(len(points),4) or not np.isfinite(raw).all():raise RuntimeError('Invalid native SDF output shape/values')
        values=raw.reshape(-1,len(OFFSETS),4);distance=values[:,:,3]
        derivative20=(distance[:,3]-distance[:,1])/40e-6
        derivative50=(distance[:,4]-distance[:,0])/100e-6
        gradient=values[:,2,:3];gradnorm=np.linalg.norm(gradient,axis=1)
        alignment=(gradient*record['source_outward_normals_mesh_local']).sum(1)
        # Do not fail just because the suspect SDF differs. Preserve that result
        # as the intended diagnostic; layout uncertainty is explicitly reported.
        row={key:value.tolist() if isinstance(value,np.ndarray) else value for key,value in record.items()}
        row.update(raw_native_gradient_xyz_distance=values.tolist(),source_surface_distance_m=distance[:,2].tolist(),
                   surface_absolute_distance_max_m=float(abs(distance[:,2]).max()),
                   offset_derivative_20um=derivative20.tolist(),offset_derivative_50um=derivative50.tolist(),
                   surface_gradient_norm=gradnorm.tolist(),gradient_dot_source_normal=alignment.tolist(),
                   offset_signed_plane_residual_m=(distance-OFFSETS[None,:]).tolist(),
                   median_offset_derivative_20um=float(np.median(derivative20)),
                   median_offset_derivative_50um=float(np.median(derivative50)),
                   interpretation='Exact source centroids have geometric distance0; ±offset is local face-plane reference. Sharp-edge/corner effects and SDF errors remain distinguishable in saved samples; this is not a global bound or PASS.')
        shape=(len(record['source_face_indices']),len(OFFSETS))
        grad_all=values[:,:,:3];grad_norm_all=np.linalg.norm(grad_all,axis=2)
        nearest_face=reference['nearest_source_face_indices'].reshape(shape)
        seed_faces=record['source_face_indices'][:,None]
        ref_signed=reference['signed_distance_m']
        comparison=[]
        for offset_index,offset in enumerate(OFFSETS):
            entry={'normal_travel_m':float(offset),
                   'native_distance_m':distance[:,offset_index].tolist(),
                   'native_gradient_norm_min_median_max':[float(grad_norm_all[:,offset_index].min()),float(np.median(grad_norm_all[:,offset_index])),float(grad_norm_all[:,offset_index].max())],
                   'native_near_zero_gradient_count':int((grad_norm_all[:,offset_index]<1e-6).sum()),
                   'nearest_face_differs_from_seed_count':int((nearest_face[:,offset_index]!=seed_faces[:,0]).sum())}
            if ref_signed is not None:
                exact=ref_signed.reshape(shape)[:,offset_index]
                error=distance[:,offset_index]-exact
                away=abs(exact)>reference['reference_surface_tolerance_m']
                mismatch=away&(distance[:,offset_index]*exact<0.)
                exact_gradient=reference['outward_gradient_reference'].reshape(*shape,3)[:,offset_index]
                entry.update(source_mesh_signed_distance_m=exact.tolist(),native_minus_exact_mesh_distance_m=error.tolist(),
                             maximum_absolute_distance_error_m=float(abs(error).max()),
                             sign_disagreement_count=int(mismatch.sum()),
                             native_gradient_dot_exact_reference=np.sum(grad_all[:,offset_index]*exact_gradient,axis=1).tolist())
            comparison.append(entry)
        row['actual_float32_query_points_mesh_local_m']=points.tolist()
        row['exact_source_mesh_reference']={key:value.tolist() if isinstance(value,np.ndarray) else value for key,value in reference.items()}
        row['comparisons_by_normal_travel']=comparison
        row['normal_travel_is_not_assumed_equal_to_exact_signed_distance']=True
        row['interpretation']='The exact source mesh supplies signed nearest distance when closed. Deep normal travel may reach another tooth face or annular side; nearest-face switches are expected and retained. Compare native sign/distance/gradient to that geometric reference, not to the travel amount. Near-zero gradients alone are not an error proof at nondifferentiable medial locations. No automatic narrow-band change or PASS.'
        output.append(row)
    after=_host(actor_view.get_transforms()).copy()
    delta=float(abs(after-before).max())
    return {'scope':'READ_ONLY_NATIVE_SDF_VS_SOURCE_STOP_AND_THREAD_FACE_SAMPLES',
            'field_layout':'VALIDATED_INSTALLED_LAYOUT_GRADIENT_XYZ_THEN_SIGNED_DISTANCE_METERS',
            'source_point_space':'COLLISION_MESH_LOCAL_METERS','maximum_actor_pose_change_during_queries':delta,
            'no_physics_step_or_geometry_parameter_change':True,'samples':output,
            'global_collision_correctness_or_assembly_pass':False}
