#!/usr/bin/env python3
"""Declared source rigid-core/Socket coupon, excluding the nut/thread mechanism.

This is a component experiment, not robot assembly. Only a finite native axial
drive acts on the source Body; pose/contact observations never drive control.
"""
import argparse
import json
from pathlib import Path
import time

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--model',type=Path,required=True)
p.add_argument('--pose',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--physics-hz',type=int,choices=(960,1920),default=960)
p.add_argument('--source-convex-pins',action='store_true')
p.add_argument('--source-local-pin-sdfs',action='store_true')
p.add_argument('--source-pin-cylinders',action='store_true')
p.add_argument('--sdf-query-file',type=Path)
p.add_argument('--sdf-query-only',action='store_true')
args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
if sum((args.source_convex_pins,args.source_local_pin_sdfs,args.source_pin_cylinders))>1:
    raise ValueError('select only one declared pin representation')
(out/'driver_snapshot.py').write_bytes(Path(__file__).read_bytes())
(out/'launch.json').write_text(json.dumps({k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},indent=2))
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'multi_gpu':False,'fast_shutdown':True})
failed=False;started=time.perf_counter()
try:
    import numpy as np
    import omni.usd
    import omni.replicator.core as rep
    from pxr import Usd,UsdGeom,UsdLux,UsdPhysics,UsdShade,PhysxSchema,Sdf,Gf
    from scipy.spatial.transform import Rotation
    from isaacsim.core.api import World
    from isaacsim.core.experimental.prims import RigidPrim
    from isaacsim.core.simulation_manager import SimulationManager
    from te_foundationpose_handoff_runtime import _author_camera,_camera_cv_pose_from_eye_target
    from omni.physx.bindings._physx import SETTING_DISABLE_CONTACT_PROCESSING
    import carb,cv2
    dt=1/args.physics_hz;SimulationManager.set_physics_sim_device('cpu')
    carb.settings.get_settings().set_bool(SETTING_DISABLE_CONTACT_PROCESSING,False)
    world=World(stage_units_in_meters=1.,physics_dt=dt,rendering_dt=dt,
                backend='numpy',device='cpu',sim_params={'use_gpu_pipeline':False})
    stage=omni.usd.get_context().get_stage();layer=stage.GetRootLayer()
    UsdGeom.Xform.Define(stage,'/World')
    physics=PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath(world.get_physics_context().prim_path))
    physics.CreateEnableGPUDynamicsAttr(False);physics.CreateBroadphaseTypeAttr('MBP');physics.CreateSolverTypeAttr('TGS')
    physics.CreateMinVelocityIterationCountAttr(1);physics.CreateMaxVelocityIterationCountAttr(1)
    physics.CreateEnableExternalForcesEveryIterationAttr(True)
    source=Usd.Stage.Open(str(args.model.resolve()))
    body_source='/World/TE_J35FreeSplitPlug/Body'
    socket_source='/World/TEVisualHandoff/FixedReceptaclePose/OfficialVisual/Geometry'
    body_path='/World/PinCarrier';socket_path='/World/Receptacle'
    for old,new in [(body_source,body_path),(socket_source,socket_path)]:
        if not Sdf.CopySpec(source.GetRootLayer(),old,layer,new):raise RuntimeError('source copy failed')
    body=stage.GetPrimAtPath(body_path);socket=stage.GetPrimAtPath(socket_path)
    for child in list(body.GetChildren()):
        if (child.GetName()!='SocketRigidCoreCollision'
                and not (args.source_convex_pins and child.GetName().startswith('SourcePinConvex_'))
                and not (args.source_local_pin_sdfs and child.GetName().startswith('SourcePinSdf_'))
                and not (args.source_pin_cylinders and child.GetName().startswith('SourcePinCylinder_'))):
            stage.RemovePrim(child.GetPath())
    # Recreate each physical material binding without changing coefficients.
    copied_materials=[]
    for prim in list(Usd.PrimRange(body))+[socket]:
        for rel in prim.GetRelationships():
            if rel.GetName().startswith('material:binding'):
                for target in rel.GetTargets():
                    material=source.GetPrimAtPath(target)
                    if not material:raise RuntimeError('missing source material '+str(target))
                    Sdf.CreatePrimInLayer(layer,target.GetParentPath())
                    if not Sdf.CopySpec(source.GetRootLayer(),target,layer,target):raise RuntimeError('material copy failed')
                    copied_materials.append(str(target))
        if prim.HasAPI(UsdPhysics.FilteredPairsAPI):
            UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().SetTargets([])
    def transform(prim):return np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
    recipe=json.load(args.pose.open());position=np.asarray(recipe['positions_world_m'][0]);q=np.asarray(recipe['quaternions_wxyz'][0])
    frame=Rotation.from_quat(q[[1,2,3,0]]).as_matrix();axis=frame[:,2]
    xf=UsdGeom.Xformable(body);xf.ClearXformOpOrder();xf.AddTranslateOp().Set(Gf.Vec3d(*position))
    xf.AddOrientOp().Set(Gf.Quatf(float(q[0]),Gf.Vec3f(*q[1:])))
    S=transform(source.GetPrimAtPath(socket_source));sq=Rotation.from_matrix(S[:3,:3]).as_quat()
    xf=UsdGeom.Xformable(socket);xf.ClearXformOpOrder();xf.AddTranslateOp().Set(Gf.Vec3d(*S[:3,3]))
    xf.AddOrientOp().Set(Gf.Quatf(float(sq[3]),Gf.Vec3f(*sq[:3])))
    for api in (UsdPhysics.ArticulationRootAPI,PhysxSchema.PhysxArticulationAPI):
        if body.HasAPI(api):body.RemoveAPI(api)
    UsdPhysics.RigidBodyAPI(body).CreateVelocityAttr(Gf.Vec3f(0.));UsdPhysics.RigidBodyAPI(body).CreateAngularVelocityAttr(Gf.Vec3f(0.))
    rb=PhysxSchema.PhysxRigidBodyAPI.Apply(body);rb.CreateSleepThresholdAttr(0.)
    rb.CreateSolverPositionIterationCountAttr(128);rb.CreateSolverVelocityIterationCountAttr(1)
    UsdGeom.Imageable(stage.GetPrimAtPath(body_path+'/SocketRigidCoreCollision')).MakeVisible()
    pin_paths=[str(c.GetPath()) for c in body.GetChildren() if c.GetName().startswith('SourcePinConvex_')]
    if args.source_convex_pins and len(pin_paths)!=384:raise RuntimeError('all source pin pieces are required')
    for path in pin_paths:UsdGeom.Imageable(stage.GetPrimAtPath(path)).MakeVisible()
    sdf_pin_paths=[str(c.GetPath()) for c in body.GetChildren() if c.GetName().startswith('SourcePinSdf_')]
    if args.source_local_pin_sdfs:
        represented=[]
        for path in sdf_pin_paths:
            a=stage.GetPrimAtPath(path).GetAttribute('kcg:sourcePinIndices')
            represented.extend(list(a.Get()) if a and a.HasAuthoredValueOpinion() else [int(path.rsplit('_',1)[-1])])
        if sorted(represented)!=list(range(128)):raise RuntimeError('all 128 source pins must be represented exactly once')
    for path in sdf_pin_paths:UsdGeom.Imageable(stage.GetPrimAtPath(path)).MakeVisible()
    cylinder_paths=[str(c.GetPath()) for c in body.GetChildren() if c.GetName().startswith('SourcePinCylinder_')]
    if args.source_pin_cylinders and len(cylinder_paths)!=256:raise RuntimeError('both source cylindrical sections of all 128 pins are required')
    for path in cylinder_paths:UsdGeom.Imageable(stage.GetPrimAtPath(path)).MakeVisible()
    mass_fields=['physics:mass','physics:centerOfMass','physics:diagonalInertia','physics:principalAxes']
    if any(str(body.GetAttribute(k).Get())!=str(source.GetPrimAtPath(body_source).GetAttribute(k).Get()) for k in mass_fields):
        raise RuntimeError('source Body mass or inertia changed')
    for old,new in [(body_source+'/SocketRigidCoreCollision',body_path+'/SocketRigidCoreCollision'),(socket_source,socket_path)]:
        for name in ('points','faceVertexIndices','faceVertexCounts'):
            if not np.array_equal(np.asarray(source.GetPrimAtPath(old).GetAttribute(name).Get()),np.asarray(stage.GetPrimAtPath(new).GetAttribute(name).Get())):
                raise RuntimeError('source mesh changed')
    guide=UsdPhysics.Joint.Define(stage,'/World/DeclaredAxialTestGuide')
    guide.CreateBody1Rel().SetTargets([body.GetPath()]);guide.CreateExcludeFromArticulationAttr(True)
    guide.CreateLocalPos0Attr(Gf.Vec3f(*position));guide.CreateLocalRot0Attr(Gf.Quatf(float(q[0]),Gf.Vec3f(*q[1:])))
    guide.CreateLocalPos1Attr(Gf.Vec3f(0.));guide.CreateLocalRot1Attr(Gf.Quatf(1.))
    for name in ('transX','transY','rotX','rotY','rotZ'):
        limit=UsdPhysics.LimitAPI.Apply(guide.GetPrim(),name);limit.CreateLowAttr(1.);limit.CreateHighAttr(-1.)
    drive=UsdPhysics.DriveAPI.Apply(guide.GetPrim(),'transZ');drive.CreateTypeAttr('force')
    drive.CreateStiffnessAttr(10000.);drive.CreateDampingAttr(20.);drive.CreateMaxForceAttr(3.0400615)
    drive.CreateTargetPositionAttr(0.);drive.CreateTargetVelocityAttr(0.)
    PhysxSchema.PhysxContactReportAPI.Apply(body).CreateThresholdAttr(0.)
    contacts=RigidPrim([body_path],resolve_paths=False,contact_filter_paths=[socket_path],max_contact_count=32768)
    camera='/World/CouponCamera';target=S[:3,3]+[0,0,-.0012]
    _author_camera(stage,camera,_camera_cv_pose_from_eye_target(target+[.05,-.06,.035],target),
        resolution=(960,720),focal_length_mm=45.,horizontal_aperture_mm=36.,clipping_range_m=(.001,5.),Gf=Gf,UsdGeom=UsdGeom)
    UsdLux.DomeLight.Define(stage,'/World/Light').CreateIntensityAttr(1200.)
    product=rep.create.render_product(camera,(960,720));rgb=rep.AnnotatorRegistry.get_annotator('rgb');rgb.attach([product.path])
    layer.Export(str(out/'coupon_before_physics.usdc'))
    world.reset()
    # Preserve startup diagnostics and errors; this warning is already retained
    # in the source/reference runs and repeats once per SDF contact point.
    import carb.logging
    carb.logging.acquire_logging().set_level_threshold_for_source(
        'omni.physx.plugin',carb.logging.LogSettingBehavior.OVERRIDE,carb.logging.LEVEL_ERROR)
    (out/'logging_scope.json').write_text(json.dumps(dict(source='omni.physx.plugin',after_reset_level='ERROR',
        reason='Repeated CPU SDF material-face warning retained in preceding runs; numerical records and errors unchanged'),indent=2))
    if args.source_pin_cylinders:
        from omni.physx import get_physx_scene_query_interface
        query=get_physx_scene_query_interface();positions,quaternions=(x.numpy() if hasattr(x,'numpy') else np.asarray(x) for x in contacts.get_world_poses())
        R=Rotation.from_quat(quaternions[0][[1,2,3,0]]).as_matrix();centre_world=positions[0];ray_records=[]
        for index in (0,64,103):
            path=f'{body_path}/SourcePinCylinder_{index:03d}_1';prim=stage.GetPrimAtPath(path)
            radius=float(UsdGeom.Cylinder(prim).GetRadiusAttr().Get());local=np.asarray(UsdGeom.Xformable(prim).GetLocalTransformation()).T
            centre=centre_world+R@local[:3,3]
            for angle_deg in (7.,23.,41.,89.):
                a=np.deg2rad(angle_deg);normal=R@np.array([np.cos(a),np.sin(a),0.]);origin=centre+normal*(radius+.001)
                hit=query.raycast_closest(carb.Float3(*origin),carb.Float3(*(-normal)),.0015)
                if not hit.get('hit',bool(hit.get('collision'))):raise RuntimeError('analytic pin ray missed')
                nr=np.asarray(hit['normal']);angle_error=float(np.arccos(np.clip(nr@normal,-1,1)))
                distance_error=abs(float(hit['distance'])-.001)
                surface_error=float(np.linalg.norm(np.asarray(hit['position'])-(centre+normal*radius)))
                ray_records.append(dict(path=path,angle_deg=angle_deg,hit_path=hit.get('collision'),
                    distance_error_m=distance_error,reported_distance_m=float(hit['distance']),
                    hit_position_world_m=list(hit['position']),expected_surface_position_world_m=(centre+normal*radius).tolist(),
                    surface_point_error_m=surface_error,normal_error_rad=angle_error))
        (out/'native_cylinder_ray_audit.json').write_text(json.dumps(ray_records,indent=2))
        # PxGjkQuery returns its ray advancement parameter separately from the
        # support-surface witness. Retain their discrepancy; do not compensate
        # source radius by the ray parameter's convergence tolerance.
        if any(x['hit_path']!=x['path'] or x['surface_point_error_m']>1e-7 or x['normal_error_rad']>1e-3 for x in ray_records):
            raise RuntimeError('native cylinder ray response does not match analytic source dimensions')
    if args.sdf_query_file:
        from te_pin_sdf_geometry_query import query_pin_contact_sdf
        try:
            query_result=query_pin_contact_sdf(stage,SimulationManager._physics_sim_view__warp,args.sdf_query_file,
                body_path=body_path,local_pin_paths=sdf_pin_paths)
        except Exception as error:
            query_result={'scope':'NATIVE_SDF_GEOMETRY_QUERY_ONLY','status':'UNAVAILABLE','error':str(error)}
        (out/'native_pin_sdf_query.json').write_text(json.dumps(query_result,indent=2))
        print('NATIVE_PIN_SDF_QUERY',json.dumps(query_result),flush=True)
        if args.sdf_query_only:
            (out/'result.json').write_text(json.dumps({'scope':'SDF_GEOMETRY_QUERY_ONLY_NOT_DYNAMIC_BENCHMARK',
                'status':query_result['status'],'commanded_experiment_steps':0,'time_after_world_reset_s':float(world.current_time)},indent=2))
            raise SystemExit(0)
    if args.source_convex_pins:
        from pxr import UsdUtils,PhysicsSchemaTools
        from omni.physx import get_physx_cooking_interface
        from omni.physx.bindings._physx import PhysxCollisionRepresentationResult
        from scipy.spatial import ConvexHull
        cooking=get_physx_cooking_interface();stage_id=UsdUtils.StageCache.Get().Insert(stage).ToLongInt();cooked_audits=[]
        for path in pin_paths:
            found=[]
            def receive(result,convexes):
                if result!=PhysxCollisionRepresentationResult.RESULT_VALID:raise RuntimeError('native pin cooking failed')
                found.extend(convexes)
            cooking.request_convex_collision_representation(stage_id=stage_id,collision_prim_id=PhysicsSchemaTools.sdfPathToInt(path),run_asynchronously=False,on_result=receive)
            if len(found)!=1:raise RuntimeError('each source pin piece must cook to one convex')
            cooked=np.array([[v.x,v.y,v.z] for v in found[0].vertices],float)
            mesh=UsdGeom.Mesh.Get(stage,path)
            requested=np.array(mesh.GetPointsAttr().Get(),float)
            local=np.asarray(UsdGeom.Xformable(mesh).GetLocalTransformation()).T
            scale=np.linalg.norm(local[:3,:3],axis=0)
            if not np.allclose(scale,scale[0],rtol=1e-6):raise RuntimeError('pin meshes require uniform coordinate scaling')
            eq=ConvexHull(requested).equations;eqc=ConvexHull(cooked).equations
            outward=float(np.max(cooked@eq[:,:3].T+eq[:,3])*scale[0]);lost=float(np.max(requested@eqc[:,:3].T+eqc[:,3])*scale[0])
            cooked_audits.append(dict(path=path,requested_vertices=len(requested),cooked_vertices=len(cooked),coordinate_scale_to_m=float(scale[0]),outward_plane_error_m=outward,inward_plane_error_m=lost))
        (out/'cooked_pin_hull_audit.json').write_text(json.dumps(cooked_audits,indent=2))
        if max(max(a['outward_plane_error_m'],a['inward_plane_error_m']) for a in cooked_audits)>5e-8:
            raise RuntimeError('cooked source pin geometry exceeds 0.05 micrometre plane error')
    def host(x):return x.numpy() if hasattr(x,'numpy') else np.asarray(x)
    def capture(label):
        import omni.timeline,omni.kit.app
        def state():
            return np.concatenate([host(x).ravel() for x in (*contacts.get_world_poses(),*contacts.get_velocities())])
        before=state().copy();t=float(world.current_time);playing=world.is_playing()
        if SimulationManager.is_fabric_enabled():
            from omni.physxfabric import get_physx_fabric_interface
            get_physx_fabric_interface().force_update(dt,t)
        timeline=omni.timeline.get_timeline_interface();auto=timeline.is_auto_updating()
        settings=carb.settings.get_settings();old=settings.get('/app/player/playSimulations')
        try:
            timeline.set_auto_update(False);timeline.commit_silently();settings.set('/app/player/playSimulations',False)
            for _ in range(3):omni.kit.app.get_app().update()
            settings.set('/app/player/playSimulations',old)
            rep.orchestrator.step(rt_subframes=1,delta_time=0.,pause_timeline=not playing)
        finally:
            settings.set('/app/player/playSimulations',old);timeline.set_auto_update(auto);timeline.commit_silently()
        after=state()
        if world.current_time!=t or np.max(abs(before-after))>0:raise RuntimeError('capture changed native state')
        pixels=np.asarray(rgb.get_data())
        if pixels.size==0:raise RuntimeError('empty coupon image')
        cv2.imwrite(str(out/(label+'.png')),cv2.cvtColor(pixels[:,:,:3],cv2.COLOR_RGB2BGR))
        with (out/'image_audit.jsonl').open('a') as f:
            f.write(json.dumps(dict(label=label,time_s=t,native_state_delta=float(np.max(abs(before-after))),time_delta_s=world.current_time-t))+'\n')
        if playing and not world.is_playing():world.play()
    stream=(out/'samples.jsonl').open('x',buffering=1);rows=[];sample_index=0;previous_loaded=False
    timings={'drive_update_s':0.,'world_step_s':0.,'native_readback_s':0.,'recording_s':0.,'evidence_s':0.}
    for phase,duration in [('initial_hold',.25),('insertion',2.5),('hold',.5)]:
        for i in range(round(duration/dt)):
            u=(i+1)/round(duration/dt);blend=10*u**3-15*u**4+6*u**5
            command=.0004*blend if phase=='insertion' else .0004 if phase=='hold' else 0.
            tick=time.perf_counter();drive.GetTargetPositionAttr().Set(command)
            timings['drive_update_s']+=time.perf_counter()-tick;tick=time.perf_counter()
            world.step(render=False)
            timings['world_step_s']+=time.perf_counter()-tick;tick=time.perf_counter()
            positions,quats=(host(x).copy() for x in contacts.get_world_poses())
            linear,angular=(host(x).copy() for x in contacts.get_velocities())
            f,pt,n,sep,cnt,start,_=contacts.get_raw_contact_data()
            f,pt,n,sep,cnt,start=(host(x).copy() for x in (f,pt,n,sep,cnt,start))
            timings['native_readback_s']+=time.perf_counter()-tick;tick=time.perf_counter()
            count=int(cnt.ravel()[0]);offset=int(start.ravel()[0]);ix=slice(offset,offset+count)
            impulse=f.ravel()[ix];normal=n[ix];points=pt[ix];separation=sep.ravel()[ix]
            wrench=np.r_[(impulse[:,None]*normal).sum(0),np.cross(points-positions[0],impulse[:,None]*normal).sum(0)]/dt
            loaded=bool(np.any(impulse!=0.))
            full_contacts=(sample_index%max(1,round(.05/dt))==0 or (loaded and not previous_loaded))
            selected=np.ones(len(impulse),bool) if full_contacts else impulse!=0.
            row=dict(time_s=float(world.current_time),phase=phase,command_m=command,
                body_depth_m=float(S[2,3]-positions[0,2]),positions_world_m=positions.tolist(),quaternions_wxyz=quats.tolist(),
                native_linear_velocity_m_s=linear.tolist(),native_angular_velocity_rad_s=angular.tolist(),
                normal_sum_n=float(np.maximum(impulse,0).sum()/dt),normal_wrench=wrench.tolist(),
                raw_contact_count=len(impulse),nonzero_contact_count=int(np.count_nonzero(impulse)),
                full_contact_snapshot=full_contacts,
                contacts=[dict(position_m=a.tolist(),normal=b.tolist(),impulse_n_s=float(c),separation_m=float(d))
                          for a,b,c,d in zip(points[selected],normal[selected],impulse[selected],separation[selected])])
            stream.write(json.dumps(row,separators=(',',':'))+'\n');rows.append(row)
            sample_index+=1;previous_loaded=loaded
            timings['recording_s']+=time.perf_counter()-tick
            if i==round(duration/dt)-1:
                print(json.dumps({k:row[k] for k in ('time_s','phase','body_depth_m','normal_sum_n')}),flush=True)
                tick=time.perf_counter();capture(phase);timings['evidence_s']+=time.perf_counter()-tick
                (out/'timing_progress.json').write_text(json.dumps({'phase':phase,'sample_count':sample_index,'totals':timings},indent=2))
    stream.close();world.pause()
    result=dict(scope='SOURCE_RIGID_CORE_AND_SOCKET_COMPONENT_ONLY; NOT ROBOT OR COMPLETE CONNECTOR ASSEMBLY',
        recipe=recipe,physics_hz=args.physics_hz,source_geometry_unchanged=True,source_body_mass_inertia_unchanged=True,
        source_convex_pins=args.source_convex_pins,
        source_local_pin_sdfs=args.source_local_pin_sdfs,
        source_pin_cylinders=args.source_pin_cylinders,
        excluded_components=['CouplingNut and Body-Nut joint','compliant grounding band','interfacial seal','representative spring contacts','hand/arm'],
        finite_axial_drive_cap_n=3.0400615,source_material_paths=sorted(set(copied_materials)),
        commanded_pose_writes_after_start=False,pose_truth_used_online=False,
        contact_trace_policy='All nonzero impulses every step; all near-field candidates every 0.05 s and on first contact. Unfiltered counts and force sums every step. Earlier full logs retained.',
        final_depth_m=rows[-1]['body_depth_m'],initial_depth_m=rows[0]['body_depth_m'],
        maximum_normal_sum_n=max(r['normal_sum_n'] for r in rows),
        cumulative_normal_impulse_n_s=sum(r['normal_sum_n']*dt for r in rows),
        wall_time_components_s=timings,
        total_simulation_s=rows[-1]['time_s'],wall_s=time.perf_counter()-started)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
except Exception as error:
    import traceback
    failed=True;(out/'failure.json').write_text(json.dumps({'error':str(error),'traceback':traceback.format_exc()},indent=2))
    traceback.print_exc()
finally:
    app.close()
raise SystemExit(1 if failed else 0)
