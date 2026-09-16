"""Isolated current-hand torque test with an explicitly declared rotary bearing.

The bearing locates the original Nut but provides no axial rotation drive.
Known external torque loads the grip. Hand control uses only hand encoders,
the existing root-moment signal and the finite self-lock motor interface.
Nut pose/contact records are for postrun capacity evaluation, not hand control.
This is a laboratory grip test, never an assembly-success demonstration.
"""
from pathlib import Path
import argparse
import json
import math
import os
import shutil
import time
import traceback
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from probe_source_finger_fourbar import _origin, _binary_stl, _host


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--preload-mode',choices=('current_root','elastic_3p25'),default='current_root')
    parser.add_argument('--load-limit-nm',type=float,default=2.2)
    parser.add_argument('--hold-input-after-preload',action='store_true',
        help='After the same preload, use existing worm self-lock instead of continued root-moment regulation.')
    args=parser.parse_args()
    if not 0<args.load_limit_nm<=4.6:parser.error('the declared lab torque range is (0,4.6] Nm')
    root=Path(__file__).resolve().parents[3];out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    for name in ('probe_three_finger_torque_capacity.py','te_hand_mechanism_runtime.py','te_worm_drive.py','te_hand_fourbar.py'):
        shutil.copy2(Path(__file__).with_name(name),out/name)
    geometry=json.loads(args.geometry.read_text())
    if geometry['finger_mechanism_id']!='source_nails_fourbar_20260912':
        raise ValueError('current measured fourbar geometry is required')
    (out/'geometry_plan.json').write_text(json.dumps(geometry,indent=2)+'\n')
    from kcg_connector.grasp.robust.hand_contract import load_carts_hand_contract
    from kcg_connector.grasp.robust.collision_roster import load_authoritative_collision_link_roster
    from kcg_connector.grasp.carts_v2.models import _build_verified_robot_model
    from te_hand_mechanism_runtime import author_hand_mechanism,HandMechanismRuntime,ACTIVE_HAND
    from te_three_finger_wrench_observer import FingerRootMomentObserver
    from te_worm_drive import finger_force_motor_targets
    import yaml
    model=_build_verified_robot_model(
        load_carts_hand_contract('src/kcg_connector/config/carts_hand_contact_v1.yaml',repository_root=root),
        load_authoritative_collision_link_roster('src/kcg_connector/config/carts_collision_roster_v1.yaml',repository_root=root),
        finger_mechanism_path=root/'src/kcg_connector/config/hand_fourbar_20260912.json')
    source_state=json.loads((root/'artifacts/rotation_diagnosis_20260913/native_sparse_states.json').read_text())['395162']
    arm=np.asarray(source_state['q'][:7]);opened=np.asarray(geometry['open_hand_positions_rad'])
    first_contact=np.asarray(geometry['first_contact_hand_positions_rad'])
    full=np.r_[arm,opened];fk={n:np.asarray(T) for n,T in model.forward_kinematics(full,enforce_limits=False).items()}
    hand_pose=fk['handbase_link'];nut_pose=hand_pose@np.linalg.inv(geometry['canonical_body_from_hand_for_nut_grasp'])
    axis=nut_pose[:3,2]
    document=ET.parse(root/'src/iiwa_description/urdf/hand.xacro').getroot()
    link_names=['handbase_link','f1Link1','f1Link2','f1Link3','f2Link1','f2Link2','f3Link1','f3Link2','f3Link3']
    terminal=('f1Link3','f2Link2','f3Link3')
    joint_names=['f1j1','f1j2','f1j3','f2j1','f2j2','f3j1','f3j2','f3j3']
    joints={n:document.find(f"joint[@name='{n}']") for n in joint_names}
    initial={n:float(full[model.independent_joint_names.index(n)]) for n in ACTIVE_HAND}
    initial['f3j1']=initial['f1j1']
    for follower,coupling in model.fourbar_couplings.items():
        initial[follower]=coupling.position_and_derivative(initial[coupling.source_joint])[0]
    levels=[v for v in (.3,.6,.9,1.2,1.5,1.9,2.2,2.8,3.5,4.6) if v<=args.load_limit_nm+1e-12]
    if not levels or abs(levels[-1]-args.load_limit_nm)>1e-10:levels.append(args.load_limit_nm)
    current=yaml.safe_load((root/'src/kcg_connector/config/visual_assembly_v1_task.yaml').read_text())
    root_targets=np.asarray(current['nut_regrasp']['root_moment_preload']['task_refinement']['targets_nm'])
    requested=root_targets if args.preload_mode=='current_root' else np.full(3,3.25)
    dt=1/960.;app=None;runtime=None;stream=None;frames=[];records=[];abort=None;started=time.monotonic()
    deadline=float(os.environ.get('KCG_EXPERIMENT_ACTION_DEADLINE',started+270.))
    result={'scope':'ISOLATED_CURRENT_HAND_GRIP_CAPACITY_NOT_ASSEMBLY','hardware_authorized':False,
        'geometry':str(args.geometry),'preload_mode':args.preload_mode,'preload_references_nm':requested.tolist(),
        'torque_levels_nm':levels,'bearing_resists_rotation_about_test_axis':False,
        'external_torque_direction':'minus original Body Z; grip reaction is assembly-tightening plus Body Z',
        'source_hand_geometry_mass_and_fourbar_preserved':True,'post_start_object_pose_writes':False,
        'online_nut_pose_or_contact_truth_used_for_hand_control':False,'physics_hz':960,
        'finger_elastic_boundary_nm':3.5,'palm_elastic_boundary_nm':1.,'friction':.45,
        'hold_input_after_preload':args.hold_input_after_preload}
    try:
        from isaacsim import SimulationApp
        app=SimulationApp({'headless':True,'multi_gpu':False,'fast_shutdown':True})
        import carb
        import omni.usd
        from pxr import Usd,UsdGeom,UsdPhysics,UsdShade,PhysxSchema,Gf,Sdf,Vt
        from isaacsim.core.api import World
        from isaacsim.core.experimental.prims import Articulation,RigidPrim
        from isaacsim.core.simulation_manager import SimulationManager
        SimulationManager.set_physics_sim_device('cpu')
        world=World(stage_units_in_meters=1.,physics_dt=dt,rendering_dt=1/60,backend='numpy',device='cpu')
        stage=omni.usd.get_context().get_stage();scene=UsdPhysics.Scene(stage.GetPrimAtPath(world.get_physics_context().prim_path))
        scene.CreateGravityDirectionAttr(Gf.Vec3f(0,0,-1));scene.CreateGravityMagnitudeAttr(9.81)
        physics=PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim());physics.CreateEnableGPUDynamicsAttr(False)
        physics.CreateBroadphaseTypeAttr('MBP');physics.CreateSolverTypeAttr('TGS')
        physics.CreateEnableExternalForcesEveryIterationAttr(True)
        for a,b,v in ((physics.CreateMinPositionIterationCountAttr,physics.CreateMaxPositionIterationCountAttr,64),
                      (physics.CreateMinVelocityIterationCountAttr,physics.CreateMaxVelocityIterationCountAttr,4)):
            a(v);b(v)
        carb.settings.get_settings().set('/physics/disableContactProcessing',False)
        mat=UsdShade.Material.Define(stage,'/World/LabContactMaterial')
        m=UsdPhysics.MaterialAPI.Apply(mat.GetPrim());m.CreateStaticFrictionAttr(.45);m.CreateDynamicFrictionAttr(.45);m.CreateRestitutionAttr(0.)
        PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim()).CreateFrictionCombineModeAttr('average')
        def quat(R):
            q=Rotation.from_matrix(R).as_quat();return Gf.Quatf(float(q[3]),Gf.Vec3f(*q[:3]))
        def collider(prim,sdf=False):
            UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('sdf' if sdf else 'convexHull')
            c=PhysxSchema.PhysxCollisionAPI.Apply(prim);c.CreateContactOffsetAttr(.00005);c.CreateRestOffsetAttr(0.)
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat,materialPurpose='physics')
            if sdf:
                s=PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim);s.CreateSdfResolutionAttr(1024)
                s.CreateSdfSubgridResolutionAttr(6);s.CreateSdfNarrowBandThicknessAttr(.002);s.CreateSdfTriangleCountReductionFactorAttr(1.)
                PhysxSchema.PhysxTriangleMeshCollisionAPI.Apply(prim).CreateWeldToleranceAttr(0.)
        def mesh(path,points,faces,transform=None,color=(.4,.5,.6)):
            obj=UsdGeom.Mesh.Define(stage,path)
            if transform is not None:obj.AddTransformOp().Set(Gf.Matrix4d(*transform.T.ravel().tolist()))
            obj.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(points,np.float32)))
            obj.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces),3,np.int32)))
            obj.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(faces,np.int32).ravel()))
            obj.CreateSubdivisionSchemeAttr('none');obj.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            return obj
        hand_root='/World/Hand';joint_root=hand_root+'/Physics'
        hroot=UsdGeom.Xform.Define(stage,hand_root);UsdPhysics.ArticulationRootAPI.Apply(hroot.GetPrim())
        art=PhysxSchema.PhysxArticulationAPI.Apply(hroot.GetPrim());art.CreateSolverPositionIterationCountAttr(64);art.CreateSolverVelocityIterationCountAttr(4)
        source_records=[]
        for name in link_names:
            link=document.find(f"link[@name='{name}']");path=hand_root+'/Geometry/'+name
            obj=UsdGeom.Xform.Define(stage,path);obj.AddTranslateOp().Set(Gf.Vec3d(*fk[name][:3,3]));obj.AddOrientOp().Set(quat(fk[name][:3,:3]))
            UsdPhysics.RigidBodyAPI.Apply(obj.GetPrim())
            i=link.find('inertial');origin=_origin(i.find('origin'));x=i.find('inertia')
            I=np.array([[float(x.get('ixx')),float(x.get('ixy')),float(x.get('ixz'))],
                [float(x.get('ixy')),float(x.get('iyy')),float(x.get('iyz'))],
                [float(x.get('ixz')),float(x.get('iyz')),float(x.get('izz'))]])
            I=origin[:3,:3]@I@origin[:3,:3].T;values,axes=np.linalg.eigh(I)
            if np.linalg.det(axes)<0:axes[:,0]*=-1
            mass=UsdPhysics.MassAPI.Apply(obj.GetPrim());mass.CreateMassAttr(float(i.find('mass').get('value')))
            mass.CreateCenterOfMassAttr(Gf.Vec3f(*origin[:3,3]));mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*values));mass.CreatePrincipalAxesAttr(quat(axes))
            for k,visual in enumerate(link.findall('visual')):
                spec=visual.find('geometry/mesh');file=root/'src/iiwa_description'/spec.get('filename').split('package://iiwa_description/')[1]
                points,normals=_binary_stl(file);points*=np.fromstring(spec.get('scale','1 1 1'),sep=' ').astype(np.float32)
                objmesh=mesh(path+f'/Visual{k}',points,np.arange(len(points)).reshape(-1,3),_origin(visual.find('origin')),
                    color=(.18,.44,.68) if name in terminal else (.52,.56,.60))
                if name in terminal:collider(objmesh.GetPrim(),sdf=True)
            if name not in terminal:
                for k,col in enumerate(link.findall('collision')):
                    spec=col.find('geometry/mesh');file=root/'src/iiwa_description'/spec.get('filename').split('package://iiwa_description/')[1]
                    points,normals=_binary_stl(file);points*=np.fromstring(spec.get('scale','1 1 1'),sep=' ').astype(np.float32)
                    cmesh=mesh(path+f'/Collision{k}',points,np.arange(len(points)).reshape(-1,3),_origin(col.find('origin')))
                    cmesh.GetVisibilityAttr().Set('invisible');collider(cmesh.GetPrim())
            PhysxSchema.PhysxContactReportAPI.Apply(obj.GetPrim()).CreateThresholdAttr(0.)
            source_records.append({'link':name,'mass_kg':float(i.find('mass').get('value')),'inertia_link_kg_m2':I.tolist()})
        # A fixed sensor link makes the handbase an internal articulated link:
        # incoming-force readback on a world-fixed articulation root is zero.
        # Both mount frames are coincident; the hand's source mass is unchanged.
        sensor_path=hand_root+'/Geometry/DeclaredSensorMount'
        sensor=UsdGeom.Xform.Define(stage,sensor_path)
        sensor.AddTranslateOp().Set(Gf.Vec3d(*hand_pose[:3,3]));sensor.AddOrientOp().Set(quat(hand_pose[:3,:3]))
        UsdPhysics.RigidBodyAPI.Apply(sensor.GetPrim());sensor_mass=UsdPhysics.MassAPI.Apply(sensor.GetPrim())
        sensor_mass.CreateMassAttr(.01);sensor_mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-6))
        ground=UsdPhysics.FixedJoint.Define(stage,joint_root+'/DeclaredBenchGround')
        ground.CreateBody1Rel().SetTargets([sensor_path]);ground.CreateLocalPos0Attr(Gf.Vec3f(*hand_pose[:3,3]));ground.CreateLocalRot0Attr(quat(hand_pose[:3,:3]))
        fixed=UsdPhysics.FixedJoint.Define(stage,joint_root+'/DeclaredBenchMount')
        fixed.CreateBody0Rel().SetTargets([sensor_path]);fixed.CreateBody1Rel().SetTargets([hand_root+'/Geometry/handbase_link'])
        fixed.CreateLocalPos0Attr(Gf.Vec3f(0));fixed.CreateLocalRot0Attr(Gf.Quatf(1))
        fixed.CreateLocalPos1Attr(Gf.Vec3f(0));fixed.CreateLocalRot1Attr(Gf.Quatf(1))
        for name,spec in joints.items():
            if not np.allclose(np.fromstring(spec.find('axis').get('xyz'),sep=' '),[0,0,1]):raise ValueError('unexpected source hinge axis')
            j=UsdPhysics.RevoluteJoint.Define(stage,joint_root+'/'+name);frame=_origin(spec.find('origin'))
            j.CreateBody0Rel().SetTargets([hand_root+'/Geometry/'+spec.find('parent').get('link')])
            j.CreateBody1Rel().SetTargets([hand_root+'/Geometry/'+spec.find('child').get('link')])
            j.CreateAxisAttr('Z');j.CreateLocalPos0Attr(Gf.Vec3f(*frame[:3,3]));j.CreateLocalRot0Attr(quat(frame[:3,:3]))
            j.CreateLocalPos1Attr(Gf.Vec3f(0));j.CreateLocalRot1Attr(Gf.Quatf(1));j.CreateCollisionEnabledAttr(False)
            limit=spec.find('limit');j.CreateLowerLimitAttr(math.degrees(float(limit.get('lower'))));j.CreateUpperLimitAttr(math.degrees(float(limit.get('upper'))))
            PhysxSchema.PhysxJointAPI.Apply(j.GetPrim()).CreateMaxJointVelocityAttr(math.degrees(float(limit.get('velocity'))))
        setup=author_hand_mechanism(stage,root,'src/kcg_connector/config/hand_mechanism_runtime_v1.json',joint_root,initial)
        source_path=root/'artifacts/kcg_connector/te_connector_contact_repaired_20260911/connector_model.usdc'
        source=Usd.Stage.Open(str(source_path));source_nut=source.GetPrimAtPath('/World/TE_J35FreeSplitPlug/CouplingNut')
        nut_path='/World/Nut';obj=UsdGeom.Xform.Define(stage,nut_path)
        obj.AddTranslateOp().Set(Gf.Vec3d(*nut_pose[:3,3]));obj.AddOrientOp().Set(quat(nut_pose[:3,:3]))
        UsdPhysics.RigidBodyAPI.Apply(obj.GetPrim());srcmass=UsdPhysics.MassAPI(source_nut);mass=UsdPhysics.MassAPI.Apply(obj.GetPrim())
        mass.CreateMassAttr(srcmass.GetMassAttr().Get());mass.CreateCenterOfMassAttr(srcmass.GetCenterOfMassAttr().Get())
        mass.CreateDiagonalInertiaAttr(srcmass.GetDiagonalInertiaAttr().Get());mass.CreatePrincipalAxesAttr(srcmass.GetPrincipalAxesAttr().Get())
        ext=UsdGeom.Mesh(source.GetPrimAtPath(str(source_nut.GetPath())+'/ExternalSurfaceContact'))
        if UsdGeom.Xformable(ext).GetOrderedXformOps():raise ValueError('unexpected original Nut external mesh transform')
        objmesh=mesh(nut_path+'/OriginalExternalSurface',np.asarray(ext.GetPointsAttr().Get()),np.asarray(ext.GetFaceVertexIndicesAttr().Get()).reshape(-1,3),color=(.72,.64,.39));collider(objmesh.GetPrim(),True)
        bearing=UsdPhysics.RevoluteJoint.Define(stage,'/World/DeclaredNutRotaryBearing')
        bearing.CreateBody1Rel().SetTargets([nut_path]);bearing.CreateAxisAttr('Z')
        bearing.CreateLocalPos0Attr(Gf.Vec3f(*nut_pose[:3,3]));bearing.CreateLocalRot0Attr(quat(nut_pose[:3,:3]))
        bearing.CreateLocalPos1Attr(Gf.Vec3f(0));bearing.CreateLocalRot1Attr(Gf.Quatf(1));bearing.CreateCollisionEnabledAttr(False)
        PhysxSchema.PhysxJointAPI.Apply(bearing.GetPrim()).CreateJointFrictionAttr(0.)
        # The visible bearing marker is a test fixture, with no collision and
        # no rigid link or drive acting about the measured rotation axis.
        marker=UsdGeom.Cylinder.Define(stage,'/World/VisibleLabBearing');marker.CreateRadiusAttr(.008);marker.CreateHeightAttr(.05)
        marker.AddTranslateOp().Set(Gf.Vec3d(*(nut_pose[:3,3]+axis*.04)));marker.CreateDisplayColorAttr([Gf.Vec3f(.4,.2,.6)])
        (out/'source_authoring.json').write_text(json.dumps({'source_links':source_records,'source_nut':str(source_path),
            'nut_mass_kg':srcmass.GetMassAttr().Get(),'declared_fixed_hand_mount':True,'declared_free_rotary_nut_bearing':True,
            'omitted_from_capacity_test':['robot arm motion','connector Body and Socket','internal mating contacts'],
            'nut_external_mesh_points':len(ext.GetPointsAttr().Get()),'fingertip_sdf_resolution':1024},indent=2)+'\n')
        import omni.replicator.core as rep
        camera=UsdGeom.Camera.Define(stage,'/World/LabCamera');focus=nut_pose[:3,3]+np.array([0.,0.,.045])
        camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*(focus+np.array([.16,-.22,.12]))),Gf.Vec3d(*focus),Gf.Vec3d(0,0,1)).GetInverse())
        camera.CreateFocalLengthAttr(28.);camera.CreateHorizontalApertureAttr(36.);camera.CreateVerticalApertureAttr(28.)
        camera.CreateClippingRangeAttr(Gf.Vec2f(.01,3.))
        UsdLux=__import__('pxr.UsdLux',fromlist=['UsdLux']);UsdLux.DomeLight.Define(stage,'/World/Light').CreateIntensityAttr(700.)
        product=rep.create.render_product('/World/LabCamera',(900,700));rgb=rep.AnnotatorRegistry.get_annotator('rgb');rgb.attach([product.path])
        hand_paths=[hand_root+'/Geometry/'+n for n in link_names]
        contact=RigidPrim(hand_paths,contact_filter_paths=[nut_path],max_contact_count=4096)
        nut=RigidPrim([nut_path],resolve_paths=False)
        stage.GetRootLayer().Export(str(out/'lab_scene.usdc'))
        print('LAB_RESET_BEGIN',flush=True);world.reset();robot=Articulation(hand_root)
        print('LAB_RESET_END',list(robot.dof_names),flush=True)
        if set(robot.dof_names)!=set(joint_names):raise ValueError('hand-only DOF identity differs')
        runtime=HandMechanismRuntime(world,robot,setup,out,active_effort_caps=[1.,3.5,3.5,3.5])
        names=list(robot.dof_names);indices=[names.index(n) for n in ACTIVE_HAND];roots=indices[1:]
        root_link_index=list(robot.link_names).index('handbase_link')
        observer=FingerRootMomentObserver(root,model);tare_q=[];tare_r=[]
        low=np.array([setup['intervals'][n][0] for n in ACTIVE_HAND]);high=np.array([setup['intervals'][n][1] for n in ACTIVE_HAND])
        hand_target=opened.copy();stream=(out/'samples.jsonl').open('x',buffering=1)
        def frame(label):
            import omni.timeline,omni.kit.app
            from PIL import Image
            world.pause();timeline=omni.timeline.get_timeline_interface();auto=timeline.is_auto_updating()
            settings=carb.settings.get_settings();play=settings.get('/app/player/playSimulations')
            before=float(world.current_time);q_before=_host(robot.get_dof_positions(indices=0)).copy();np_before=_host(nut.get_world_poses()[0]).copy()
            try:
                timeline.set_auto_update(False);timeline.commit_silently();settings.set('/app/player/playSimulations',False)
                for _ in range(3):omni.kit.app.get_app().update()
                rep.orchestrator.step(rt_subframes=1,delta_time=0.,pause_timeline=False)
                data=np.asarray(rgb.get_data()).copy();file=out/(label+'.png');Image.fromarray(data[:,:,:3].astype(np.uint8)).save(file)
                frames.append({'label':label,'time_s':before,'path':str(file)})
            finally:
                settings.set('/app/player/playSimulations',play);timeline.set_auto_update(auto);timeline.commit_silently();world.play()
            if world.current_time!=before or np.max(abs(_host(robot.get_dof_positions(indices=0))-q_before))>1e-10 or np.max(abs(_host(nut.get_world_poses()[0])-np_before))>1e-10:
                raise RuntimeError('same-run rendering changed physical state')
        def step(phase,load,reference=None):
            nonlocal hand_target
            if time.monotonic()>deadline:raise RuntimeError('LAB_WALL_BUDGET')
            if (out/'STOP_REQUEST').exists():raise RuntimeError('USER_STOP_REQUEST')
            q=_host(robot.get_dof_positions(indices=0))[0];effort=_host(robot.get_dof_projected_joint_forces(indices=0))[0]
            observed=None
            if observer.tare_reaction is not None:
                observed=observer.control_moments(np.r_[arm,q[indices]],effort[roots],runtime.steps,dt)
            elastic=np.array([runtime.drives[n].reference.transmission_stiffness*(runtime.drives[n].input_angle-q[j]) for n,j in zip(ACTIVE_HAND[1:],roots)])
            input_hold=(args.hold_input_after_preload and reference is not None and phase!='preload')
            if input_hold:
                # Position-reference equal to the current motor input encoder
                # gives zero commanded motor effort. The existing transmission
                # friction holds it; physical finger outputs remain free.
                hand_target[1:]=[runtime.drives[n].input_angle for n in ACTIVE_HAND[1:]]
            elif reference is not None:
                measured=observed if args.preload_mode=='current_root' else elastic
                hand_target,_=finger_force_motor_targets(runtime,q[roots],hand_target,reference,measured,dt,120.,1/6,.15,.02,low,high)
            nut.apply_forces_and_torques_at_pos(torques=(-float(load)*axis)[None,:],indices=[0],local_frame=False)
            runtime.submit(hand_target,phase);world.step(render=False)
            q,v,effort=runtime.last_native_state
            root_force,root_torque=robot.get_link_incoming_joint_force(indices=[0],link_indices=[root_link_index])
            root_force,root_torque=_host(root_force).reshape(-1,3)[0],_host(root_torque).reshape(-1,3)[0]
            pp,rr=nut.get_world_poses();linear,angular=nut.get_velocities();pp,rr,linear,angular=map(_host,(pp,rr,linear,angular))
            contact_data=[_host(x) for x in contact.get_contact_force_data(dt=dt)]
            force,points,normals,separation,counts,starts=contact_data
            contacts=[]
            for i,name in enumerate(link_names):
                count=int(counts.ravel()[i]);offset=int(starts.ravel()[i]);sl=slice(offset,offset+count)
                contacts.append({'link':name,'normal_sum_n':float(force[sl].sum()),'points_m':points[sl].tolist(),
                    'normals':normals[sl].tolist(),'normal_forces_n':force[sl].ravel().tolist()})
            elast=[float(runtime.drives[n].reference.transmission_stiffness*(runtime.drives[n].input_angle-q[j])) for n,j in zip(ACTIVE_HAND,indices)]
            record={'step':runtime.steps-1,'time_s':float(world.current_time),'phase':phase,'applied_resisting_torque_nm':float(load),
                'active_hand_q_rad':q[indices].tolist(),'active_hand_velocity_rad_s':v[indices].tolist(),
                'root_measured_moments_nm':None if observed is None else observed.tolist(),
                'elastic_efforts_nm':elast,'motor_targets_rad':hand_target.tolist(),
                'input_self_lock_hold_commanded':input_hold,
                'bench_support_force_hand_frame_n':root_force.tolist(),'bench_support_torque_hand_frame_nm':root_torque.tolist(),
                'nut_position_m':pp[0].tolist(),'nut_orientation_wxyz':rr[0].tolist(),
                'nut_linear_velocity_m_s':linear[0].tolist(),'nut_angular_velocity_rad_s':angular[0].tolist(),
                'contacts':contacts}
            records.append(record);stream.write(json.dumps(record,separators=(',',':'))+'\n')
            if runtime.steps%240==0:
                print('LAB_PROGRESS',phase,round(world.current_time,3),load,flush=True)
                (out/'progress.json').write_text(json.dumps({k:record[k] for k in ['step','time_s','phase','applied_resisting_torque_nm','elastic_efforts_nm']})+'\n')
            return q,effort,elast
        for _ in range(round(.25/dt)):
            q,effort,_=step('free_space_tare',0.)
            tare_q.append(np.r_[arm,q[indices]]);tare_r.append(effort[roots])
        observer.calibrate_free_space(tare_q[-120:],tare_r[-120:]);frame('00_open')
        close=first_contact.copy();close[1:]+=.004;duration=max(1.,1.875*np.max(abs(close-opened))/.15)
        for k in range(round(duration/dt)):
            u=(k+1)/round(duration/dt);f=10*u**3-15*u**4+6*u**5;hand_target=opened+f*(close-opened);step('closing',0.)
        q=_host(robot.get_dof_positions(indices=0))[0];eff=_host(robot.get_dof_projected_joint_forces(indices=0))[0]
        initial_load=(observer.observe(np.r_[arm,q[indices]],eff[roots]) if args.preload_mode=='current_root' else
            np.array([120.*(runtime.drives[n].input_angle-q[j]) for n,j in zip(ACTIVE_HAND[1:],roots)]))
        for k in range(round(2.5/dt)):
            u=min(1.,(k+1)*dt/1.5);blend=10*u**3-15*u**4+6*u**5
            step('preload',0.,initial_load+blend*(requested-initial_load))
        frame('01_preloaded')
        window=records[-240:]
        values=np.array([r['root_measured_moments_nm'] if args.preload_mode=='current_root' else r['elastic_efforts_nm'][1:] for r in window])
        result['preload_achieved_mean_nm']=values.mean(0).tolist()
        if np.any(abs(values.mean(0)-requested)>.08*requested):raise RuntimeError('PRELOAD_REFERENCE_NOT_REACHED')
        last_load=0.;capacity_stop=None
        for index,level in enumerate(levels):
            for k in range(round(.2/dt)):
                u=(k+1)/round(.2/dt);f=10*u**3-15*u**4+6*u**5
                _,_,elastic=step(f'load_ramp_{level:g}',last_load+f*(level-last_load),requested)
                if min(np.array([1.,3.5,3.5,3.5])-np.abs(elastic))<.05:
                    capacity_stop='TRANSMISSION_RESERVE';break
            if capacity_stop:break
            for _ in range(round(.3/dt)):
                _,_,elastic=step(f'load_hold_{level:g}',level,requested)
                if min(np.array([1.,3.5,3.5,3.5])-np.abs(elastic))<.05:
                    capacity_stop='TRANSMISSION_RESERVE';break
            if capacity_stop:break
            last_load=level
        result['load_stop']=capacity_stop
        if last_load>0:
            previous_load=records[-1]['applied_resisting_torque_nm']
            for k in range(round(.2/dt)):
                u=(k+1)/round(.2/dt);f=10*u**3-15*u**4+6*u**5
                step('return_to_highest_complete_load',previous_load+f*(last_load-previous_load),requested)
            for _ in range(round(1./dt)):step('confirm_highest_complete_load',last_load,requested)
            result['confirmation_torque_nm']=last_load
        frame('02_end_of_loading')
        for _ in range(round(.5/dt)):step('unloaded_grip',0.,requested)
        start_hand=hand_target.copy();duration=max(1.,1.875*np.max(abs(start_hand-opened))/.15)
        for k in range(round(duration/dt)):
            u=(k+1)/round(duration/dt);f=10*u**3-15*u**4+6*u**5;hand_target=start_hand+f*(opened-start_hand);step('opening',0.)
        for _ in range(round(.3/dt)):step('released',0.)
        frame('03_released')
        # Post-release free-bearing witness: a tiny known torque should move
        # the Nut. It is evaluated later, so no Nut truth enters the controller.
        for _ in range(round(.05/dt)):step('free_bearing_witness',.0001)
        result['protocol_completed']=True
    except Exception as error:
        abort=str(error);result['traceback']=traceback.format_exc();print(result['traceback'],flush=True)
    finally:
        if stream:stream.close()
        if runtime:runtime.close()
        result.update(abort=abort,samples=len(records),wall_s=time.monotonic()-started,frames=frames,
            nut_world_from_source_body=nut_pose.tolist(),hand_pose_world=hand_pose.tolist(),source_arm_q_rad=arm.tolist())
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print('LAB_RESULT',json.dumps({k:result.get(k) for k in ['abort','load_stop','samples','wall_s','protocol_completed']}),flush=True)
        if app:
            app.close()
    return 2 if abort else 0


if __name__=='__main__':
    raise SystemExit(main())
