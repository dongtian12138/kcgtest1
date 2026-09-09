"""Short CPU reference-load check of the same fixed-joint wrist-force API.

Each laboratory payload has a declared mass and centre-of-mass lever arm.
No contact-force getter is used to generate the reference. No source robot or
connector is edited; these explicitly separate fixtures are not assembly.
"""
from pathlib import Path
import json
import time


def run_known_load_calibration(output, *, velocity_iterations=1):
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'multi_gpu': False, 'fast_shutdown': True})
    output = Path(output)
    (output/'calibration_source.py').write_bytes(Path(__file__).read_bytes())
    started = time.perf_counter()
    failed = False
    try:
        import numpy as np
        import omni.usd
        from scipy.spatial.transform import Rotation
        from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.simulation_manager import SimulationManager

        dt = 1/960.
        SimulationManager.set_physics_sim_device('cpu')
        world = World(stage_units_in_meters=1., physics_dt=dt, rendering_dt=dt,
                      backend='numpy', device='cpu', sim_params={'use_gpu_pipeline': False})
        stage = omni.usd.get_context().get_stage()
        scene_prim = stage.GetPrimAtPath(world.get_physics_context().prim_path)
        scene = PhysxSchema.PhysxSceneAPI.Apply(scene_prim)
        scene.CreateEnableGPUDynamicsAttr(False)
        scene.CreateBroadphaseTypeAttr('MBP')
        scene.CreateSolverTypeAttr('TGS')
        scene.CreateMinVelocityIterationCountAttr(velocity_iterations)
        scene.CreateMaxVelocityIterationCountAttr(velocity_iterations)
        scene.CreateEnableExternalForcesEveryIterationAttr(True)
        gravity = UsdPhysics.Scene(scene_prim)
        gravity.CreateGravityDirectionAttr(Gf.Vec3f(0., 0., -1.))
        gravity.CreateGravityMagnitudeAttr(9.81)
        gravity_value = float(gravity.GetGravityMagnitudeAttr().Get())

        def quat(q):
            return Gf.Quatf(float(q[3]), Gf.Vec3f(*map(float, q[:3])))

        def rigid(path, origin, q, mass, com):
            x = UsdGeom.Xform.Define(stage, path)
            x.AddTranslateOp().Set(Gf.Vec3d(*map(float, origin)))
            x.AddOrientOp().Set(quat(q))
            prim = x.GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(prim)
            m = UsdPhysics.MassAPI.Apply(prim)
            m.CreateMassAttr(mass)
            m.CreateCenterOfMassAttr(Gf.Vec3f(*com))
            m.CreateDiagonalInertiaAttr(Gf.Vec3f(mass*.02**2/6))
            p = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            p.CreateSleepThresholdAttr(0.)
            p.CreateLinearDampingAttr(0.)
            p.CreateAngularDampingAttr(0.)
            p.CreateSolverPositionIterationCountAttr(128)
            p.CreateSolverVelocityIterationCountAttr(velocity_iterations)
            visual = UsdGeom.Cube.Define(stage, path+'/ReferenceVisual')
            visual.CreateSizeAttr(.02)
            visual.AddTranslateOp().Set(Gf.Vec3d(*com))
            visual.CreateDisplayColorAttr([Gf.Vec3f(.2, .65, .85)])
            return prim

        cases = []
        angles = [(0,0,0),(180,0,0),(0,90,0),(0,-90,0),(90,0,0),(-90,0,0),(20,30,40)]
        masses = [.1,.15,.2,.12,.18,.25,.16]
        levers = [(.06,.02,.03),(.04,-.03,.02),(.02,.05,.04),(-.03,.04,-.02),(.05,.02,-.04),(-.04,.03,.05),(.04,-.03,.02)]
        for i,(euler,mass,com) in enumerate(zip(angles,masses,levers)):
            root = f'/World/Reference{i}'
            UsdGeom.Xform.Define(stage,root)
            R = Rotation.from_euler('xyz',euler,degrees=True)
            q = R.as_quat(); position = np.array([i*.18,0.,.25])
            for label,m,offset in [('Base',1.,(0.,0.,0.)),('Carrier',.1,(0.,0.,0.)),('Payload',mass,com)]:
                rigid(root+'/'+label,position,q,m,offset)
            clamp = UsdPhysics.FixedJoint.Define(stage,root+'/WorldClamp')
            clamp.CreateBody1Rel().SetTargets([root+'/Base'])
            clamp.CreateLocalPos0Attr(Gf.Vec3f(*map(float,position)))
            clamp.CreateLocalRot0Attr(quat(q))
            UsdPhysics.ArticulationRootAPI.Apply(clamp.GetPrim())
            art = PhysxSchema.PhysxArticulationAPI.Apply(clamp.GetPrim())
            art.CreateSolverPositionIterationCountAttr(128)
            art.CreateSolverVelocityIterationCountAttr(velocity_iterations)
            art.CreateSleepThresholdAttr(0.)
            slide = UsdPhysics.PrismaticJoint.Define(stage,root+'/CarrierSlide')
            slide.CreateBody0Rel().SetTargets([root+'/Base'])
            slide.CreateBody1Rel().SetTargets([root+'/Carrier'])
            slide.CreateAxisAttr('Z')
            slide.CreateLowerLimitAttr(-.005); slide.CreateUpperLimitAttr(.005)
            drive = UsdPhysics.DriveAPI.Apply(slide.GetPrim(),'linear')
            drive.CreateTypeAttr('force'); drive.CreateStiffnessAttr(10000.)
            drive.CreateDampingAttr(100.); drive.CreateMaxForceAttr(5.)
            drive.CreateTargetPositionAttr(0.); drive.CreateTargetVelocityAttr(0.)
            sensor = UsdPhysics.FixedJoint.Define(stage,root+'/WristSensor')
            sensor.CreateBody0Rel().SetTargets([root+'/Carrier'])
            sensor.CreateBody1Rel().SetTargets([root+'/Payload'])
            tree = world.scene.add(SingleArticulation(root,name=f'reference{i}',reset_xform_properties=False))
            m = float(UsdPhysics.MassAPI(stage.GetPrimAtPath(root+'/Payload')).GetMassAttr().Get())
            arm = np.array(UsdPhysics.MassAPI(stage.GetPrimAtPath(root+'/Payload')).GetCenterOfMassAttr().Get())
            f_local = R.as_matrix().T@np.array([0.,0.,-m*gravity_value])
            expected = np.r_[f_local,np.cross(arm,f_local)]
            cases.append({'id':i,'tree':tree,'euler_xyz_deg':euler,'mass_kg':m,
                          'com_sensor_m':arm.tolist(),'expected_sensor_wrench':expected.tolist()})
        stage.GetRootLayer().Export(str(output/'calibration_before_physics.usda'))
        world.reset()
        for case in cases:
            case['row'] = case['tree']._articulation_view._metadata.joint_indices['WristSensor']+1
        def host(v):
            if hasattr(v,'detach'):return v.detach().cpu().numpy()
            return v.numpy() if hasattr(v,'numpy') else np.asarray(v)
        samples = []
        for step in range(round(1.5/dt)):
            world.step(render=False)
            sample = []
            for case in cases:
                raw = host(case['tree'].get_measured_joint_forces())[case['row']].copy()
                sample.append(raw)
            samples.append(sample)
        samples = np.asarray(samples)
        results = []
        for i,case in enumerate(cases):
            expected = np.asarray(case['expected_sensor_wrench'])
            external = -samples[-480:,i,:]
            error = external-expected
            record = {k:v for k,v in case.items() if k!='tree'}
            record.update(mean_external_sensor_wrench=external.mean(0).tolist(),
                          maximum_abs_error=np.max(abs(error),axis=0).tolist(),
                          maximum_force_error_n=float(np.max(abs(error[:,:3]))),
                          maximum_torque_error_nm=float(np.max(abs(error[:,3:]))))
            results.append(record)
        report = {'scope':'SYNTHETIC_KNOWN_WEIGHT_AND_LEVER_WRIST_API_CHECK_NOT_REAL_HAND_OR_CONTACT_MODEL_VALIDATION',
                  'cpu_tgs_hz':960,'position_iterations':128,'velocity_iterations':velocity_iterations,
                  'gravity_m_s2':gravity_value,'contact_force_data_used_as_reference':False,
                  'post_start_object_pose_or_mass_writes':False,'cases':results,
                  'maximum_force_error_n':max(r['maximum_force_error_n'] for r in results),
                  'maximum_torque_error_nm':max(r['maximum_torque_error_nm'] for r in results),
                  'wall_seconds':time.perf_counter()-started}
        np.savez_compressed(output/'raw_sensor_samples.npz',raw_incoming_sensor_wrenches=samples,dt_s=dt)
        (output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)
    except Exception:
        import traceback
        failed = True
        error = traceback.format_exc()
        (output/'error.txt').write_text(error)
        print(error,flush=True)
    finally:
        app.close(exit_code=1 if failed else 0)
    return 1 if failed else 0
