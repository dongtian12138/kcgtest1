#!/usr/bin/env python3
"""Validate the new laboratory sensor through its external clamp with known loads."""
import argparse,json,time
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
p.add_argument('--position-iterations',type=int,choices=(32,128),default=128)
args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
(out/'driver_snapshot.py').write_bytes(Path(__file__).read_bytes())
(out/'guide_snapshot.py').write_bytes(Path(__file__).with_name('te_instrumented_rotary_guide.py').read_bytes())
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'multi_gpu':False,'fast_shutdown':True});failed=False;started=time.perf_counter()
try:
    import numpy as np,carb,cv2
    import omni.usd,omni.replicator.core as rep
    from pxr import Usd,UsdGeom,UsdLux,UsdPhysics,PhysxSchema,Gf
    from isaacsim.core.api import World
    from isaacsim.core.experimental.prims import RigidPrim
    from isaacsim.core.simulation_manager import SimulationManager
    from te_instrumented_rotary_guide import author_instrumented_guide,read_instrumented_guide
    from te_foundationpose_handoff_runtime import _author_camera,_camera_cv_pose_from_eye_target
    dt=1/960;SimulationManager.set_physics_sim_device('cpu')
    world=World(stage_units_in_meters=1.,physics_dt=dt,rendering_dt=dt,backend='numpy',device='cpu',sim_params={'use_gpu_pipeline':False})
    stage=omni.usd.get_context().get_stage();UsdGeom.Xform.Define(stage,'/World')
    scene=stage.GetPrimAtPath(world.get_physics_context().prim_path);ps=PhysxSchema.PhysxSceneAPI.Apply(scene)
    ps.CreateEnableGPUDynamicsAttr(False);ps.CreateBroadphaseTypeAttr('MBP');ps.CreateSolverTypeAttr('TGS')
    ps.CreateMinVelocityIterationCountAttr(1);ps.CreateMaxVelocityIterationCountAttr(1);ps.CreateEnableExternalForcesEveryIterationAttr(True)
    UsdPhysics.Scene(scene).CreateGravityDirectionAttr(Gf.Vec3f(0.,0.,-1.))
    UsdPhysics.Scene(scene).CreateGravityMagnitudeAttr(9.81)
    gravity=float(UsdPhysics.Scene(scene).GetGravityMagnitudeAttr().Get())
    if not np.isfinite(gravity) or gravity<=0:raise ValueError('known-load reference requires explicit finite gravity')
    T=np.diag([1.,-1.,-1.,1.]);T[:3,3]=[.5,0,.3];axis=T[:3,2]
    cube=UsdGeom.Cube.Define(stage,'/World/KnownPayload');cube.CreateSizeAttr(.025)
    cube.AddTranslateOp().Set(Gf.Vec3d(*T[:3,3]));cube.AddOrientOp().Set(Gf.Quatf(0.,Gf.Vec3f(1,0,0)))
    body=cube.GetPrim();UsdPhysics.RigidBodyAPI.Apply(body)
    m=UsdPhysics.MassAPI.Apply(body);m.CreateMassAttr(.05);m.CreateCenterOfMassAttr(Gf.Vec3f(0.))
    m.CreateDiagonalInertiaAttr(Gf.Vec3f(.05*.025**2/6));mass=float(m.GetMassAttr().Get())
    phys=PhysxSchema.PhysxRigidBodyAPI.Apply(body);phys.CreateSleepThresholdAttr(0.)
    phys.CreateSolverPositionIterationCountAttr(args.position_iterations);phys.CreateSolverVelocityIterationCountAttr(1)
    phys.CreateLinearDampingAttr(0.);phys.CreateAngularDampingAttr(0.)
    guide,drive,axial,spec=author_instrumented_guide(stage,world,str(body.GetPath()),T,torque_cap_nm=.25,position_iterations=args.position_iterations)
    view=RigidPrim([str(body.GetPath())],resolve_paths=False)
    camera='/World/ReferenceCamera';_author_camera(stage,camera,_camera_cv_pose_from_eye_target(T[:3,3]+[.1,-.1,.07],T[:3,3]),
        resolution=(640,480),focal_length_mm=35.,horizontal_aperture_mm=36.,clipping_range_m=(.001,5.),Gf=Gf,UsdGeom=UsdGeom)
    UsdLux.DomeLight.Define(stage,'/World/Light').CreateIntensityAttr(1000.)
    product=rep.create.render_product(camera,(640,480));rgb=rep.AnnotatorRegistry.get_annotator('rgb');rgb.attach([product.path])
    stage.GetRootLayer().Export(str(out/'known_load_fixture.usdc'));world.reset()
    rows=[];stream=(out/'samples.jsonl').open('x',buffering=1)
    for phase,duration,torque in [('initial_hold',.5,0.),('known_torque',.75,.1),('unloaded',.5,0.)]:
        for i in range(round(duration/dt)):
            view.apply_forces_and_torques_at_pos(torques=[(axis*torque).tolist()],indices=[0])
            world.step(render=False);r=read_instrumented_guide(spec)
            r.update(time_s=float(world.current_time),phase=phase,elapsed_phase_s=(i+1)*dt,
                known_external_axis_torque_nm=torque,expected_external_axis_force_n=mass*gravity)
            rows.append(r);stream.write(json.dumps(r,separators=(',',':'))+'\n')
    stream.close()
    checks=[]
    for phase in ['initial_hold','known_torque','unloaded']:
        rr=[x for x in rows if x['phase']==phase];rr=rr[-round(.2/dt):]
        checks.append(dict(phase=phase,maximum_torque_error_nm=max(abs(x['external_axis_torque_nm']-x['known_external_axis_torque_nm']) for x in rr),
            maximum_axial_force_error_n=max(abs(x['external_wrench_sensor'][2]-x['expected_external_axis_force_n']) for x in rr)))
    import omni.timeline,omni.kit.app
    before=np.r_[np.asarray(view.get_world_poses()[0]).ravel(),np.asarray(view.get_world_poses()[1]).ravel()];now=float(world.current_time)
    if SimulationManager.is_fabric_enabled():
        from omni.physxfabric import get_physx_fabric_interface
        get_physx_fabric_interface().force_update(dt,now)
    tl=omni.timeline.get_timeline_interface();auto=tl.is_auto_updating();settings=carb.settings.get_settings();old=settings.get('/app/player/playSimulations')
    try:
        tl.set_auto_update(False);tl.commit_silently();settings.set('/app/player/playSimulations',False)
        for _ in range(3):omni.kit.app.get_app().update()
        settings.set('/app/player/playSimulations',old);rep.orchestrator.step(rt_subframes=1,delta_time=0.,pause_timeline=False)
    finally:
        settings.set('/app/player/playSimulations',old);tl.set_auto_update(auto);tl.commit_silently()
    after=np.r_[np.asarray(view.get_world_poses()[0]).ravel(),np.asarray(view.get_world_poses()[1]).ravel()]
    image=np.asarray(rgb.get_data())
    if world.current_time!=now or np.max(abs(after-before))>0 or image.size==0:raise RuntimeError('reference image audit failed')
    cv2.imwrite(str(out/'after_load_release.png'),cv2.cvtColor(image[:,:,:3],cv2.COLOR_RGB2BGR))
    valid=all(x['maximum_torque_error_nm']<1e-4 and x['maximum_axial_force_error_n']<1e-3 for x in checks)
    result=dict(scope='KNOWN_WEIGHT_AND_TORQUE_THROUGH_THE_EXTERNAL_LAB_CLAMP; NOT SOURCE_CONNECTOR_OR_HAND_CAPACITY',passed=valid,
        physics_hz=960,position_iterations=args.position_iterations,known_mass_kg=mass,known_torque_nm=.1,checks=checks,fixture=spec['metadata'],loaded_tare_used=False,
        image_physics_state_and_time_unchanged=True,wall_s=time.perf_counter()-started)
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
    if not valid:raise RuntimeError('instrumented guide known-load verification failed')
except Exception as e:
    import traceback
    failed=True;(out/'failure.json').write_text(json.dumps(dict(error=str(e),traceback=traceback.format_exc()),indent=2));traceback.print_exc()
finally:
    app.close(exit_code=1 if failed else 0)
