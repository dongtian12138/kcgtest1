#!/usr/bin/env python3
"""Declared connector test apparatus; no robot, axial screw constraint or pose servo.

Only the insertion phase may use a finite axial drive. Rotation uses a finite
native rotary drive; all object state readings below are evaluation-only.
"""
import argparse
import json
from pathlib import Path
import sys
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--repository', type=Path, default=Path.cwd())
p.add_argument('--source', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--initial-pose', type=Path)
p.add_argument('--frozen-model',type=Path)
p.add_argument('--frozen-model-report',type=Path)
p.add_argument('--hard-stop-boxes',action='store_true')
p.add_argument('--paired-stop-boxes',action='store_true')
p.add_argument('--verify-installer-roundtrip',action='store_true')
p.add_argument('--angle-deg', type=float, default=300.)
p.add_argument('--turn-duration-s', type=float, default=60.)
p.add_argument('--torque-cap-nm', type=float, default=1.6)
p.add_argument('--thread-friction', type=float, default=.45)
p.add_argument('--seal-stiffness', type=float, default=100.)
p.add_argument('--without-seal', action='store_true')
p.add_argument('--insert-distance-m', type=float, default=0.)
p.add_argument('--insert-duration-s', type=float, default=10.)
p.add_argument('--release-duration-s', type=float, default=3.)
p.add_argument('--loaded-hold-duration-s',type=float,default=1.)
p.add_argument('--initial-hold-duration-s',type=float,default=.5)
p.add_argument('--physics-hz', type=int, default=240)
p.add_argument('--position-iterations',type=int,default=32)
p.add_argument('--velocity-iterations',type=int,default=1)
p.add_argument('--external-forces-every-iteration',action=argparse.BooleanOptionalAction,default=True)
p.add_argument('--solver',choices=('TGS','PGS'),default='TGS')
p.add_argument('--rigid-contact-diagnostic',action='store_true')
p.add_argument('--query-native-sdf',action='store_true')
p.add_argument('--sdf-query-only',action='store_true')
p.add_argument('--rotation-driver',choices=('external_d6','internal_test_motor','external_torque'),default='external_d6')
p.add_argument('--direct-torque-nm',type=float,default=.2)
p.add_argument('--test-rotary-damping',type=float,default=0.)
p.add_argument('--maximal-passive-joint',action='store_true')
p.add_argument('--physics-device',choices=('cuda:0','cpu'),default='cuda:0')
p.add_argument('--full-shape-contact-report',action=argparse.BooleanOptionalAction,default=True)
p.add_argument('--keep-awake',action='store_true')
p.add_argument('--diagnostic-thread-friction',type=float)
p.add_argument('--video-fps', type=float, default=2.)
p.add_argument('--instrumented-guide-calibration',type=Path)
p.add_argument('--profile-python',action='store_true',help='Record call-time profile without changing physics commands.')
p.add_argument('--defer-usd-output',action='store_true',help='Publish native CPU poses to USD only for evidence; verify native state invariance.')
p.add_argument('--free-engagement-duration-s',type=float,
               help='Short free-connector diagnostic; all external apparatus disabled BEFORE reset.')
p.add_argument('--diagnostic-disable-grounding-band',action='store_true',
               help='Causal comparison only, never a delivery model or assembly result.')
p.add_argument('--torque-only-guide',action='store_true',
               help='Apply the finite rotary drive without lateral, axial or tilt constraints.')
args = p.parse_args()
if args.torque_only_guide and (args.insert_distance_m or args.instrumented_guide_calibration
                              or args.rotation_driver!='external_d6'
                              or args.free_engagement_duration_s is not None):
    p.error('Torque-only apparatus cannot have insertion, instrumentation or a free-settling override')
if args.free_engagement_duration_s is not None:
    if (not 0 < args.free_engagement_duration_s <= .1 or not args.initial_pose
            or not args.frozen_model or args.instrumented_guide_calibration
            or args.rotation_driver != 'external_d6' or args.insert_distance_m
            or args.rigid_contact_diagnostic or args.diagnostic_thread_friction is not None):
        p.error('Free engagement requires a frozen model, declared initial pose, <=0.1 s and no other apparatus/contact changes')
elif args.diagnostic_disable_grounding_band:
    p.error('Grounding-band removal is restricted to the short free-engagement causal diagnostic')
repo=args.repository.resolve(); out=args.output.resolve()
out.mkdir(parents=True, exist_ok=False)
(out/'driver_snapshot.py').write_bytes(Path(__file__).read_bytes())
sys.path.insert(0,str(repo/'src/kcg_connector/isaac'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'multi_gpu':False,'fast_shutdown':True})
failed=False; wall_start=time.perf_counter();profiler=None
if args.profile_python:
    import cProfile
    profiler=cProfile.Profile();profiler.enable()
try:
    import carb
    import numpy as np
    import omni.usd
    import omni.replicator.core as rep
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade
    from scipy.spatial.transform import Rotation
    from isaacsim.core.api import World
    from isaacsim.core.experimental.prims import RigidPrim
    from isaacsim.core.simulation_manager import SimulationManager
    from omni.physx.bindings._physx import SETTING_DISABLE_CONTACT_PROCESSING
    from te_foundationpose_handoff_runtime import _author_camera, _camera_cv_pose_from_eye_target
    from contact_reading import (read_filtered_contacts, read_shape_contact_pairs, group_shape_pairs,
                                 create_link_velocity_reader, read_link_velocity_crosscheck)
    from omni.physx import get_physx_simulation_interface
    import cv2

    dt=1./args.physics_hz
    if not 0 < args.torque_cap_nm <= 4.6 or not 0 < args.thread_friction <= 1:
        raise ValueError('finite declared lab load and material parameters required')
    report=json.loads(args.source.with_name('assembly_scene.json').read_text())
    source=Usd.Stage.Open(str(args.source)); source_layer=source.GetRootLayer()
    SimulationManager.set_physics_sim_device(args.physics_device)
    carb.settings.get_settings().set_bool(SETTING_DISABLE_CONTACT_PROCESSING,False)
    world=World(stage_units_in_meters=1.,physics_dt=dt,rendering_dt=dt,
                backend='numpy',device=args.physics_device,sim_params={'use_gpu_pipeline':args.physics_device!='cpu'})
    stage=omni.usd.get_context().get_stage(); layer=stage.GetRootLayer()
    UsdGeom.Xform.Define(stage,'/World')
    copied=[]
    for prim in source.GetPrimAtPath('/World').GetChildren():
        if prim.GetName() in ('HandArm','FixtureMaterial'): continue
        if not Sdf.CopySpec(source_layer,prim.GetPath(),layer,prim.GetPath()):
            raise RuntimeError(f'copy failed: {prim.GetPath()}')
        copied.append(str(prim.GetPath()))
    physics=PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath(world.get_physics_context().prim_path))
    physics.CreateEnableGPUDynamicsAttr(args.physics_device!='cpu')
    physics.CreateBroadphaseTypeAttr('MBP' if args.physics_device=='cpu' else 'GPU')
    physics.CreateSolverTypeAttr(args.solver)
    physics.CreateEnableExternalForcesEveryIterationAttr(args.external_forces_every_iteration)
    if not 0 <= args.velocity_iterations <= 255:
        raise ValueError('Velocity iterations out of PhysX range')
    physics.CreateMinVelocityIterationCountAttr(args.velocity_iterations)
    physics.CreateMaxVelocityIterationCountAttr(args.velocity_iterations)
    body_path='/World/TE_J35FreeSplitPlug/Body'
    nut_path='/World/TE_J35FreeSplitPlug/CouplingNut'
    parts=[stage.GetPrimAtPath(body_path),stage.GetPrimAtPath(nut_path)]
    mass_fields=('physics:mass','physics:centerOfMass','physics:diagonalInertia','physics:principalAxes')
    mass_before={str(x.GetPath()):{k:str(x.GetAttribute(k).Get()) for k in mass_fields} for x in parts}

    installed_model=None;seal_record=None
    if args.frozen_model:
        import importlib.util
        installer=args.frozen_model.resolve().with_name('install_model.py')
        if not installer.is_file():
            installer=Path(__file__).resolve().parents[1]/'package/install_model.py'
        spec=importlib.util.spec_from_file_location('delivery_install_model',installer)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        installed_model=module.install_model(stage,model_path=args.frozen_model)
        frozen_report=json.loads((args.frozen_model_report or args.frozen_model.with_name('assembly_scene.json')).read_text())
        for key in ('interfacial_seal_contact_model','hard_stop_contact_model','paired_stop_contact_model',
                    'passive_joint_representation','grounding_band_contact_model',
                    'representative_mating_contacts','representative_inner_thread','native_joint_friction'):
            if key in frozen_report:report[key]=frozen_report[key]
        material_map={x['source']:x['installed'] for x in installed_model['private_materials']}
        def remap_materials(value):
            if isinstance(value,dict):return {k:remap_materials(v) for k,v in value.items()}
            if isinstance(value,list):return [remap_materials(v) for v in value]
            return material_map.get(value,value) if isinstance(value,str) else value
        report=remap_materials(report)
        report['frozen_model_installation']=installed_model
        seal_record=report.get('interfacial_seal_contact_model')
    else:
        # Close the source's otherwise implicit Socket friction. This is a
        # declared representative assumption, not a measured TE coefficient.
        material=UsdShade.Material.Define(stage,'/World/ConnectorDeliverySocketMaterial')
        m=UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        m.CreateStaticFrictionAttr(args.thread_friction); m.CreateDynamicFrictionAttr(args.thread_friction)
        m.CreateRestitutionAttr(0.)
        PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim()).CreateFrictionCombineModeAttr('max')
        socket_shape=stage.GetPrimAtPath(report['collision']['receptacle_collision'])
        UsdShade.MaterialBindingAPI.Apply(socket_shape).Bind(material,UsdShade.Tokens.strongerThanDescendants,'physics')
        resolved,_=UsdShade.MaterialBindingAPI(socket_shape).ComputeBoundMaterial('physics')
        if resolved.GetPath()!=material.GetPath(): raise RuntimeError('socket material overridden')
        if not args.without_seal:
            from seal_contact_model import install_front_seal
            seal_record=install_front_seal(repo,stage,report,stiffness_n_m=args.seal_stiffness,damping_ns_m=0.)
    if args.diagnostic_thread_friction is not None:
        if not 0 <= args.diagnostic_thread_friction <= 1:
            raise ValueError('Diagnostic friction outside finite range')
        for label,path,coefficient in (
                ('Nut',report['collision']['nut_collision'],args.diagnostic_thread_friction),
                ('Socket',report['collision']['receptacle_collision'],args.thread_friction)):
            prim=stage.GetPrimAtPath(path)
            original,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
            destination=f'/World/DiagnosticThread{label}Material'
            if not Sdf.CopySpec(stage.GetRootLayer(),original.GetPath(),stage.GetRootLayer(),Sdf.Path(destination)):
                raise RuntimeError('Diagnostic material copy failed')
            material=UsdShade.Material.Get(stage,destination)
            api=UsdPhysics.MaterialAPI.Apply(material.GetPrim())
            api.CreateStaticFrictionAttr(coefficient);api.CreateDynamicFrictionAttr(coefficient)
            PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim()).CreateFrictionCombineModeAttr('min')
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material,UsdShade.Tokens.strongerThanDescendants,'physics')
            resolved,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
            if resolved.GetPath()!=material.GetPath():raise RuntimeError('Diagnostic material binding overridden')
        report['diagnostic_thread_friction']={
            'coefficient':args.diagnostic_thread_friction,'not_a_delivery_model':True,
            'scope':'Nut-Socket pair only: both use min; other Body materials retain max and coefficient .45',
            'purpose':'Check actual friction response separately from incomplete contact force reporting'}
    if args.hard_stop_boxes:
        from contact_stop_proxy import install_stop_boxes
        stop_metadata=Path(__file__).resolve().parents[1]/'references/contact_stop_proxy_candidate.json'
        report['hard_stop_contact_model']=install_stop_boxes(stage,body_path,
            rigid_core_path=report['grounding_band_contact_model']['rigid_core_collision'],
            record=json.loads(stop_metadata.read_text()))
    if args.rigid_contact_diagnostic:
        diagnostic_disabled=[report['grounding_band_contact_model']['compliant_band_collision'],
                             *report['representative_mating_contacts']['clip_paths']]
        if seal_record:diagnostic_disabled.append(seal_record['seal_collision'])
        for path in diagnostic_disabled:
            UsdPhysics.CollisionAPI(stage.GetPrimAtPath(path)).CreateCollisionEnabledAttr(False)
        report['diagnostic_only']={'disabled_compliant_contacts':diagnostic_disabled,
            'not_a_delivery_model':True,'purpose':'Isolate rigid keys, threads and stop from compliant loads'}
    if args.diagnostic_disable_grounding_band:
        path=report['grounding_band_contact_model']['compliant_band_collision']
        UsdPhysics.CollisionAPI(stage.GetPrimAtPath(path)).CreateCollisionEnabledAttr(False)
        report['diagnostic_only']={'disabled_compliant_contacts':[path],
            'not_a_delivery_model':True,'purpose':'Causal comparison of the first free engagement transient only'}
    if args.paired_stop_boxes:
        if not args.hard_stop_boxes:raise ValueError('Paired Socket boxes require the existing Body stop boxes')
        from contact_stop_proxy import install_socket_stop_boxes
        report['paired_stop_contact_model']=install_socket_stop_boxes(stage,
            socket_collision_path=report['collision']['receptacle_collision'],
            body_box_paths=report['hard_stop_contact_model']['installed_paths'],body_paths=[body_path,nut_path])
    if not 1 <= args.position_iterations <= 255:raise ValueError('Position iterations out of PhysX range')
    PhysxSchema.PhysxArticulationAPI.Apply(parts[0]).CreateSolverPositionIterationCountAttr(args.position_iterations)
    PhysxSchema.PhysxArticulationAPI.Apply(parts[0]).CreateSolverVelocityIterationCountAttr(args.velocity_iterations)
    for part in parts:
        PhysxSchema.PhysxRigidBodyAPI.Apply(part).CreateSolverPositionIterationCountAttr(args.position_iterations)
        PhysxSchema.PhysxRigidBodyAPI.Apply(part).CreateSolverVelocityIterationCountAttr(args.velocity_iterations)
        if args.keep_awake:PhysxSchema.PhysxRigidBodyAPI.Apply(part).CreateSleepThresholdAttr(0.)
    if args.keep_awake:
        PhysxSchema.PhysxArticulationAPI.Apply(parts[0]).CreateSleepThresholdAttr(0.)
    if args.initial_pose:
        pose=json.loads(args.initial_pose.read_text())
        # Source snapshots carry a non-identity plug root. The requested recipe
        # is in world coordinates, so do not apply that root transform twice.
        UsdGeom.Xformable(parts[0].GetParent()).ClearXformOpOrder()
        for prim,position,quaternion in zip(parts,pose['positions_world_m'],pose['quaternions_wxyz']):
            xf=UsdGeom.Xformable(prim); xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(*position))
            xf.AddOrientOp().Set(Gf.Quatf(quaternion[0],Gf.Vec3f(*quaternion[1:])))
            UsdPhysics.RigidBodyAPI(prim).CreateVelocityAttr(Gf.Vec3f(0.))
            UsdPhysics.RigidBodyAPI(prim).CreateAngularVelocityAttr(Gf.Vec3f(0.))
            actual=np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
            expected_rotation=Rotation.from_quat(np.roll(quaternion,-1)).as_matrix()
            if (not np.allclose(actual[:3,3],position,atol=1e-8,rtol=0)
                    or not np.allclose(actual[:3,:3],expected_rotation,atol=1e-6,rtol=0)):
                raise RuntimeError('authored initial world pose differs from declared recipe')
        joint=stage.GetPrimAtPath('/World/TE_J35FreeSplitPlug/Joints/CouplingNutRevolute')
        rb=Rotation.from_quat(np.roll(pose['quaternions_wxyz'][0],-1)).as_matrix()
        rn=Rotation.from_quat(np.roll(pose['quaternions_wxyz'][1],-1)).as_matrix()
        relative=rb.T@rn
        angle=float(np.rad2deg(np.arctan2(relative[1,0],relative[0,0])))
        joint.GetAttribute('state:angular:physics:position').Set(angle)
        joint.GetAttribute('state:angular:physics:velocity').Set(0.)
    if args.maximal_passive_joint:
        if args.rotation_driver=='internal_test_motor':
            raise ValueError('Maximal passive joint cannot use an internal test motor')
        for part in parts:
            if part.HasAPI(UsdPhysics.ArticulationRootAPI):part.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            if part.HasAPI(PhysxSchema.PhysxArticulationAPI):part.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
        joint=stage.GetPrimAtPath('/World/TE_J35FreeSplitPlug/Joints/CouplingNutRevolute')
        source_static=float(joint.GetAttribute('physxJointAxis:angular:staticFrictionEffort').Get())
        source_dynamic=float(joint.GetAttribute('physxJointAxis:angular:dynamicFrictionEffort').Get())
        if abs(source_static-source_dynamic)>1e-8:
            raise ValueError('This passive regularization requires equal source static/dynamic effort')
        friction=UsdPhysics.DriveAPI.Apply(joint,'angular')
        friction.CreateTypeAttr('force');friction.CreateStiffnessAttr(0.)
        friction.CreateDampingAttr(100.*np.pi/180.);friction.CreateMaxForceAttr(source_dynamic)
        friction.CreateTargetVelocityAttr(0.)
        report['passive_joint_representation']={
            'kind':'REGULAR_REVOLUTE_SAME_FRAMES_AND_MASSES',
            'source_axis_static_dynamic_effort_nm':[source_static,source_dynamic],
            'passive_resistance':'Native zero-speed damper capped at source dry-friction torque; regularized near zero velocity',
            'damping_nm_s_rad':100.,'maximum_resistance_nm':source_dynamic,
            'near_zero_regularization_rad_s':source_dynamic/100.,
            'not_a_position_drive_or_additional_world_support':True}
    def transform(prim):
        return np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
    initial_nut=transform(parts[1]); origin=initial_nut[:3,3]; frame=initial_nut[:3,:3]; axis=frame[:,2]
    socket=np.asarray(report['socket_initial_position_world_m'])
    mass_after={str(x.GetPath()):{k:str(x.GetAttribute(k).Get()) for k in mass_fields} for x in parts}
    if mass_before!=mass_after: raise RuntimeError('source mass/inertia changed')
    # Export the actual candidate WITHOUT the external test apparatus.
    stage.Flatten().Export(str(out/'connector_candidate.usdc'))
    if args.verify_installer_roundtrip:
        import importlib.util
        installer=Path(__file__).resolve().parents[1]/'package/install_model.py'
        spec=importlib.util.spec_from_file_location('delivery_install_model_roundtrip',installer)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        installed_model=module.install_model(stage,model_path=out/'connector_candidate.usdc')
        report['frozen_model_installation']=installed_model
        (out/'installer_snapshot.py').write_bytes(installer.read_bytes())
    (out/'assembly_scene.json').write_text(json.dumps(report,indent=2)+'\n')
    q0=Rotation.from_matrix(frame).as_quat()
    guide=UsdPhysics.Joint.Define(stage,'/World/DeclaredModelVerificationGuide')
    guide.CreateBody1Rel().SetTargets([parts[1].GetPath()]); guide.CreateExcludeFromArticulationAttr(True)
    guide.CreateLocalPos0Attr(Gf.Vec3f(*origin)); guide.CreateLocalRot0Attr(Gf.Quatf(q0[3],Gf.Vec3f(*q0[:3])))
    guide.CreateLocalPos1Attr(Gf.Vec3f(0.)); guide.CreateLocalRot1Attr(Gf.Quatf(1.))
    for name in (() if args.torque_only_guide else ('transX','transY','rotX','rotY')):
        limit=UsdPhysics.LimitAPI.Apply(guide.GetPrim(),name); limit.CreateLowAttr(1.); limit.CreateHighAttr(-1.)
    drive=UsdPhysics.DriveAPI.Apply(guide.GetPrim(),'rotZ')
    K=30.; D=.1
    drive.CreateTypeAttr('force'); drive.CreateStiffnessAttr(K*np.pi/180.)
    drive.CreateDampingAttr(D*np.pi/180.); drive.CreateMaxForceAttr(args.torque_cap_nm)
    drive.CreateTargetPositionAttr(0.); drive.CreateTargetVelocityAttr(0.)
    direct_torque=args.rotation_driver=='external_torque'
    if direct_torque:
        if not 0 < abs(args.direct_torque_nm) <= args.torque_cap_nm:
            raise ValueError('Direct test torque magnitude must be positive and within the declared cap')
        drive.CreateStiffnessAttr(0.);drive.CreateDampingAttr(0.);drive.CreateMaxForceAttr(0.)
        if args.test_rotary_damping:
            if not 0 < args.test_rotary_damping <= 2.:
                raise ValueError('Finite diagnostic rotary damping required')
            drive.CreateDampingAttr(args.test_rotary_damping*np.pi/180.)
            drive.CreateMaxForceAttr(args.torque_cap_nm)
    external_drive=drive;drive_offset_deg=0.;test_motor=None;motor_reference_deg=0.
    if args.rotation_driver=='internal_test_motor':
        original_joint=stage.GetPrimAtPath('/World/TE_J35FreeSplitPlug/Joints/CouplingNutRevolute')
        test_motor=UsdPhysics.DriveAPI.Apply(original_joint,'angular')
        rb=transform(parts[0])[:3,:3];rn=transform(parts[1])[:3,:3];relative=rb.T@rn
        motor_reference_deg=float(np.rad2deg(np.arctan2(relative[1,0],relative[0,0])))
        # Test-apparatus motor only. The previously exported model has its
        # original passive drive disabled; no motor is shipped in that asset.
        test_motor.CreateTypeAttr('force');test_motor.CreateStiffnessAttr(0.)
        test_motor.CreateDampingAttr(0.);test_motor.CreateMaxForceAttr(0.)
        test_motor.CreateTargetPositionAttr(motor_reference_deg);test_motor.CreateTargetVelocityAttr(0.)
    axial=None
    if args.insert_distance_m:
        axial=UsdPhysics.DriveAPI.Apply(guide.GetPrim(),'transZ')
        axial.CreateTypeAttr('force'); axial.CreateStiffnessAttr(10000.); axial.CreateDampingAttr(20.)
        axial.CreateMaxForceAttr(3.0400615); axial.CreateTargetPositionAttr(0.); axial.CreateTargetVelocityAttr(0.)
    instrumentation=None
    if args.instrumented_guide_calibration:
        calibration=json.loads(args.instrumented_guide_calibration.read_text())
        if (not calibration.get('passed') or calibration.get('physics_hz')!=960
                or args.physics_device!='cpu' or args.physics_hz!=960
                or args.position_iterations!=int(calibration.get('position_iterations',128)) or args.velocity_iterations!=1
                or args.rotation_driver!='external_d6'):
            raise ValueError('instrumented guide requires its passed CPU 960 Hz reference and retained numerical configuration')
        from te_instrumented_rotary_guide import author_instrumented_guide,read_instrumented_guide
        stage.RemovePrim(guide.GetPath())
        guide,drive,instrumented_axial,instrumentation=author_instrumented_guide(
            stage,world,nut_path,initial_nut,torque_cap_nm=args.torque_cap_nm,position_iterations=args.position_iterations)
        if args.insert_distance_m:
            axial=instrumented_axial
        else:
            instrumented_axial.GetMaxForceAttr().Set(0.);instrumented_axial.GetStiffnessAttr().Set(0.);instrumented_axial.GetDampingAttr().Set(0.)
            axial=None
        external_drive=drive
        (out/'instrumented_guide.json').write_text(json.dumps(instrumentation['metadata'],indent=2)+'\n')
        (out/'instrumented_guide_snapshot.py').write_bytes((repo/'src/kcg_connector/isaac/te_instrumented_rotary_guide.py').read_bytes())
    if args.free_engagement_duration_s is not None:
        guide.CreateJointEnabledAttr(False)
        drive.GetMaxForceAttr().Set(0.);drive.GetStiffnessAttr().Set(0.);drive.GetDampingAttr().Set(0.)
        report['free_engagement_diagnostic']={
            'duration_s':args.free_engagement_duration_s,
            'external_guide_disabled_before_world_reset':True,
            'external_drive_cap_nm':0.,'post_start_pose_writes':False,
            'grounding_band_disabled_for_comparison':args.diagnostic_disable_grounding_band,
            'not_robot_assembly_or_delivery_validation':True}
        (out/'assembly_scene.json').write_text(json.dumps(report,indent=2)+'\n')
    if args.torque_only_guide:
        report['torque_only_apparatus']={
            'lateral_translation_constrained':False,'axial_translation_constrained':False,
            'tilt_constrained':False,'only_rotZ_force_drive_authored':True,
            'maximum_rotary_drive_nm':args.torque_cap_nm,
            'scope':'DECLARED_TEST_ACTUATOR_NOT_ROBOT_ASSEMBLY'}
        (out/'assembly_scene.json').write_text(json.dumps(report,indent=2)+'\n')
    # The CPU tensor contact reader also needs this API even when the optional
    # per-shape event query below is not used.
    for x in parts:PhysxSchema.PhysxContactReportAPI.Apply(x).CreateThresholdAttr(0.)
    filters=[report['collision']['receptacle_collision'],*report['representative_mating_contacts']['clip_paths']]
    filters.extend(report.get('paired_stop_contact_model',{}).get('socket_pair_collision_paths',[]))
    contacts=RigidPrim([body_path,nut_path],resolve_paths=False,contact_filter_paths=filters,max_contact_count=32768)
    contact_interface=get_physx_simulation_interface()
    band_report=report['grounding_band_contact_model']
    own_groups={band_report['rigid_core_collision']:'rigid_core',
                band_report['compliant_band_collision']:'grounding'}
    for prim in stage.Traverse():
        path=str(prim.GetPath())
        if path.startswith(band_report['compliant_band_collision']+'_Sector_'):
            own_groups[path]='grounding'
    if seal_record:own_groups[seal_record['seal_collision']]='seal'
    other_groups={path:'contacts128' for path in report['representative_mating_contacts']['clip_paths']}
    camera_path='/World/ModelVerificationCamera'
    _author_camera(stage,camera_path,_camera_cv_pose_from_eye_target(socket+[.09,-.13,.075],socket+[0,0,.007]),
        resolution=(960,720),focal_length_mm=45.,horizontal_aperture_mm=36.,clipping_range_m=(.01,5.),Gf=Gf,UsdGeom=UsdGeom)
    UsdLux.DomeLight.Define(stage,'/World/ModelVerificationLight').CreateIntensityAttr(1200.)
    product=rep.create.render_product(camera_path,(960,720)); rgb=rep.AnnotatorRegistry.get_annotator('rgb'); rgb.attach([product.path])
    layer.Export(str(out/'test_apparatus_before_physics.usdc'))
    import faulthandler
    reset_started=time.perf_counter()
    (out/'initialization_status.json').write_text(json.dumps({
        'stage':'WORLD_RESET_ENTERED','elapsed_since_app_ready_s':reset_started-wall_start})+'\n')
    with (out/'initialization_stacks.txt').open('w') as reset_stacks:
        faulthandler.dump_traceback_later(45.,repeat=True,file=reset_stacks)
        try:
            world.reset()
        finally:
            faulthandler.cancel_dump_traceback_later()
    (out/'initialization_status.json').write_text(json.dumps({
        'stage':'WORLD_RESET_COMPLETED','world_reset_wall_s':time.perf_counter()-reset_started,
        'physics_time_after_reset_s':float(world.current_time)})+'\n')
    backend_record={'requested_device':args.physics_device,
        'gpu_dynamics_enabled':bool(physics.GetEnableGPUDynamicsAttr().Get()),
        'broadphase':str(physics.GetBroadphaseTypeAttr().Get()),
        'tensor_device':str(SimulationManager.get_physics_sim_device()),
        'solver':str(physics.GetSolverTypeAttr().Get()),'physics_dt_s':float(world.get_physics_dt())}
    (out/'effective_physics_backend.json').write_text(json.dumps(backend_record,indent=2)+'\n')
    if backend_record['gpu_dynamics_enabled']!=(args.physics_device!='cpu'):
        raise RuntimeError('Requested and effective physics backend differ')
    if args.defer_usd_output:
        if args.physics_device!='cpu':raise ValueError('deferred USD output is restricted to this CPU diagnostic')
        output_settings=carb.settings.get_settings()
        output_record={'scope':'OUTPUT_SCHEDULING_ONLY; NATIVE PHYSICS AND CONTROLS UNCHANGED',
            'previous_update_to_usd':output_settings.get('/physics/updateToUsd'),
            'previous_update_velocities_to_usd':output_settings.get('/physics/updateVelocitiesToUsd')}
        output_settings.set_bool('/physics/updateToUsd',False)
        output_settings.set_bool('/physics/updateVelocitiesToUsd',False)
        (out/'deferred_output_configuration.json').write_text(json.dumps(output_record,indent=2))
    if args.physics_device=='cpu':
        import carb.logging
        carb.logging.acquire_logging().set_level_threshold_for_source(
            'omni.physx.plugin',carb.logging.LogSettingBehavior.OVERRIDE,carb.logging.LEVEL_ERROR)
        (out/'logging_scope.json').write_text(json.dumps({'source':'omni.physx.plugin','after_reset_level':'ERROR',
            'reason':'Repeated CPU material-face warning preserved in baseline logs; physical observations and errors retained'},indent=2))
    if args.physics_device=='cpu' and args.maximal_passive_joint:
        # CPU SDF contact-report material lookup emits millions of identical
        # invalid-face-index warnings. Prior complete warning logs are kept.
        # Keep errors visible; this changes logging only, not contact handling.
        import carb.logging
        carb.logging.acquire_logging().set_level_threshold_for_source(
            'omni.physx.plugin',carb.logging.LogSettingBehavior.OVERRIDE,carb.logging.LEVEL_ERROR)
        (out/'logging_scope.json').write_text(json.dumps({
            'source':'omni.physx.plugin','after_reset_level':'ERROR',
            'reason':'Repeated CPU SDF getMaterialFromInternalFaceIndex(0xFFFFFFFF) report warning',
            'full_prior_logs':['late_stop_maximal_cpu_02.log','late_stop_maximal_cpu_03.log'],
            'physics_parameters_or_contact_processing_modified':False})+'\n')
    link_view=None;link_indices=None
    if not args.maximal_passive_joint:
        link_view,link_indices=create_link_velocity_reader(SimulationManager._physics_sim_view__warp,
                                                          body_path,[body_path,nut_path])
        compliant,combine=link_view.get_compliant_material_properties()
        (out/'native_material_readback.json').write_text(json.dumps({
            'shape_count':link_view.max_shapes,'link_paths':link_view.link_paths,
            'friction_restitution':link_view.get_material_properties().numpy().tolist(),
            'compliant_stiffness_damping':compliant.numpy().tolist(),
            'compliant_combine_modes':combine.numpy().tolist()},indent=2)+'\n')
    import warp as wp
    motor_indices={device:wp.array(np.asarray([0],np.int32),dtype=wp.int32,device=device) for device in ('cpu','cuda:0')}
    motor_buffers={name:wp.zeros((1,1),dtype=wp.float32,device='cuda:0' if name=='set_dof_position_targets' else 'cpu') for name in
                   ('set_dof_stiffnesses','set_dof_dampings','set_dof_max_forces','set_dof_position_targets')}
    def set_native_motor(name,value):
        buffer=motor_buffers[name];buffer.assign(np.asarray([[value]],np.float32))
        getattr(link_view,name)(buffer,motor_indices[str(buffer.device)])
    motor_readbacks=[]
    def record_motor(phase):
        record={'phase':phase,'stiffness_nm_rad':link_view.get_dof_stiffnesses().numpy().tolist(),
                'damping_nm_s_rad':link_view.get_dof_dampings().numpy().tolist(),
                'maximum_force_nm':link_view.get_dof_max_forces().numpy().tolist(),
                'target_rad':link_view.get_dof_position_targets().numpy().tolist(),
                'external_guide_enabled':bool(guide.GetJointEnabledAttr().Get())}
        motor_readbacks.append(record)
        (out/'test_motor_readbacks.json').write_text(json.dumps(motor_readbacks,indent=2)+'\n')
    if args.query_native_sdf:
        from contact_sdf_probe import run_native_sdf_probe
        sdf_record=run_native_sdf_probe(stage,SimulationManager._physics_sim_view__warp)
        (out/'native_sdf_readback.json').write_text(json.dumps(sdf_record,indent=2)+'\n')
        if args.sdf_query_only:
            print('Native SDF query completed; no assembly actuation requested.',flush=True)
            raise SystemExit(0)
    def host(x):
        return x.detach().cpu().numpy() if hasattr(x,'detach') else x.numpy() if hasattr(x,'numpy') else np.asarray(x)
    from pxr import PhysicsSchemaTools,UsdUtils
    from omni.physx.bindings._physx import acquire_physx_statistics_interface,PhysicsSceneStats
    statistics=acquire_physx_statistics_interface()
    statistics.enable_carb_stats_upload(True)
    statistics_stage_id=UsdUtils.StageCache.Get().Insert(stage).ToLongInt()
    statistics_scene_path=PhysicsSchemaTools.sdfPathToInt(stage.GetPrimAtPath(world.get_physics_context().prim_path).GetPath())
    def native_scene_statistics():
        stats=PhysicsSceneStats()
        for stage_id in (statistics_stage_id,0):
            if statistics.get_physx_scene_statistics(stage_id,statistics_scene_path,stats):
                return {'valid':True,'stage_id':stage_id,
                        'active_constraints':stats.nb_active_constraints,
                        'active_dynamic_rigids':stats.nb_active_dynamic_rigids,
                        'articulations':stats.nb_articulations,
                        'axis_constraints':stats.nb_axis_solver_constaints}
        return {'valid':False}
    def poses():return np.concatenate([host(x).ravel() for x in contacts.get_world_poses()])
    def audit_state():
        current_positions=host(contacts.get_world_poses()[0]).copy()
        velocity=np.concatenate([host(x).ravel().copy() for x in contacts.get_velocities()])
        f,pt,n,sep,cnt,start,_=contacts.get_raw_contact_data()
        f,pt,n,cnt,start=(host(x).copy() for x in (f,pt,n,cnt,start))
        f=f.ravel().astype(np.float64); nw=np.zeros((2,6));fw=np.zeros((2,6))
        for sensor,(count,offset) in enumerate(zip(cnt.ravel(),start.ravel())):
            if not count:continue
            ix=slice(int(offset),int(offset+count)); F=f[ix,None]*n[ix].astype(np.float64)/dt
            nw[sensor]=np.r_[F.sum(0),np.cross(pt[ix]-current_positions[sensor],F).sum(0)]
        fr,fp,fc,fs=(host(x).copy() for x in contacts.get_friction_data())
        fr=fr.astype(np.float64)/dt
        for sensor in range(2):
            chunks=[np.arange(int(offset),int(offset+count)) for count,offset in
                    zip(fc[sensor].ravel(),fs[sensor].ravel()) if count]
            if chunks:
                ix=np.concatenate(chunks)
                fw[sensor]=np.r_[fr[ix].sum(0),np.cross(fp[ix]-current_positions[sensor],fr[ix]).sum(0)]
        # Native buffers contain unused capacity, which is not physical state.
        # Compare per-actor live contact wrenches, not those unspecified slots.
        return {'poses':poses().copy(),'velocity':velocity,'normal_wrench':nw,'friction_wrench':fw}
    video=cv2.VideoWriter(str(out/'assembly_model_test.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),args.video_fps,(960,720))
    frame_audit=(out/'frame_audit.jsonl').open('x',buffering=1)
    images=[]
    def capture(phase,step):
        import omni.timeline
        before=audit_state(); now=float(world.current_time); playing=bool(world.is_playing())
        if args.defer_usd_output:
            from omni.physx import get_physx_interface
            get_physx_interface().update_transformations(False,True,True,False)
        if SimulationManager.is_fabric_enabled():
            from omni.physxfabric import get_physx_fabric_interface
            get_physx_fabric_interface().force_update(dt,now)
        timeline=omni.timeline.get_timeline_interface(); auto=timeline.is_auto_updating()
        settings=carb.settings.get_settings(); old=settings.get('/app/player/playSimulations')
        try:
            timeline.set_auto_update(False); timeline.commit_silently(); settings.set('/app/player/playSimulations',False)
            for _ in range(3):omni.kit.app.get_app().update()
            settings.set('/app/player/playSimulations',old)
            rep.orchestrator.step(rt_subframes=1,delta_time=0.,pause_timeline=not playing)
        finally:
            settings.set('/app/player/playSimulations',old); timeline.set_auto_update(auto); timeline.commit_silently()
        rgba=np.asarray(rgb.get_data()); after=audit_state()
        deltas={k:float(np.max(abs(after[k]-before[k]))) if before[k].size else 0. for k in before}
        delta=deltas['poses']
        audit={'step':step,'phase':phase,'time_s':now,'time_delta_s':float(world.current_time)-now,
               'native_state_deltas':deltas,'image_shape':list(rgba.shape)}
        frame_audit.write(json.dumps(audit)+'\n')
        if (rgba.ndim!=3 or not rgba.size or delta>0 or deltas['velocity']>0
                or deltas['normal_wrench']>1e-6 or deltas['friction_wrench']>1e-6
                or float(world.current_time)!=now):
            raise RuntimeError('evidence render audit failed: '+json.dumps(audit))
        pixels=cv2.cvtColor(rgba[:,:,:3],cv2.COLOR_RGB2BGR)
        cv2.putText(pixels,f'{phase} | t={now:.3f}s | step={step}',(15,28),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
        video.write(pixels)
        if not images or images[-1]['phase']!=phase:
            cv2.imwrite(str(out/f'{len(images):03d}_{phase}.png'),pixels)
        images.append({'frame':len(images),'phase':phase,'step':step,'time_s':now,'native_state_deltas':deltas})
        if playing and not world.is_playing():world.play()
    sequence=[('initial_hold',args.initial_hold_duration_s)]
    if axial:sequence.append(('key_insertion',args.insert_duration_s));sequence.append(('insertion_settle',.5))
    sequence += [('rotation',args.turn_duration_s),('loaded_hold',args.loaded_hold_duration_s),('free_hold',args.release_duration_s)]
    if args.free_engagement_duration_s is not None:
        sequence=[('free_hold',args.free_engagement_duration_s)]
    stream=(out/'samples.jsonl').open('x',buffering=1)
    rows=[]; previous=0.; unwrapped=0.; step=0; peaks={'contact_torque_nm':0.,'body_axial_n':0.}
    previous_part_rotations=None
    measured_times={'world_step_s':0.,'readback_and_reduction_s':0.,'recording_s':0.,'evidence_s':0.}
    raw_depth_milestones=[.0113,.0144,.0145,.0146]
    phase_releases=[]
    for phase,duration in sequence:
        if phase=='rotation' and axial:
            axial.GetMaxForceAttr().Set(0.); axial.GetStiffnessAttr().Set(0.); axial.GetDampingAttr().Set(0.)
        if phase=='rotation' and test_motor:
            external_drive.GetMaxForceAttr().Set(0.);external_drive.GetStiffnessAttr().Set(0.);external_drive.GetDampingAttr().Set(0.)
            guide.CreateJointEnabledAttr(False)
            contact_interface.flush_changes()
            set_native_motor('set_dof_stiffnesses',K)
            set_native_motor('set_dof_dampings',D)
            set_native_motor('set_dof_max_forces',args.torque_cap_nm)
            set_native_motor('set_dof_position_targets',np.deg2rad(motor_reference_deg))
            drive=test_motor;drive_offset_deg=motor_reference_deg
            record_motor(phase)
        if phase=='free_hold':
            if test_motor:
                set_native_motor('set_dof_stiffnesses',0.)
                set_native_motor('set_dof_dampings',0.)
                set_native_motor('set_dof_max_forces',0.)
                record_motor(phase)
            else:
                drive.GetMaxForceAttr().Set(0.); drive.GetStiffnessAttr().Set(0.); drive.GetDampingAttr().Set(0.)
            guide.CreateJointEnabledAttr(False)
            contact_interface.flush_changes()
            phase_releases.append({'step':step,'all_external_guidance_and_drives_disabled':True})
        for i in range(round(duration/dt)):
            u=(i+1)/round(duration/dt); fraction=10*u**3-15*u**4+6*u**5
            if phase=='key_insertion':axial.GetTargetPositionAttr().Set(args.insert_distance_m*fraction)
            if phase=='rotation':
                if test_motor:set_native_motor('set_dof_position_targets',np.deg2rad(drive_offset_deg+args.angle_deg*fraction))
                elif not direct_torque:drive.GetTargetPositionAttr().Set(drive_offset_deg+args.angle_deg*fraction)
            applied_torque_nm=(args.direct_torque_nm*fraction if phase=='rotation' else
                               args.direct_torque_nm if phase=='loaded_hold' else 0.) if direct_torque else 0.
            if direct_torque:
                contacts.apply_forces_and_torques_at_pos(torques=[(axis*applied_torque_nm).tolist()],indices=[1])
            measured_tick=time.perf_counter();world.step(render=False)
            measured_times['world_step_s']+=time.perf_counter()-measured_tick;measured_tick=time.perf_counter()
            positions,quats=(host(x) for x in contacts.get_world_poses())
            linear,angular=(host(x) for x in contacts.get_velocities())
            part_rotations=Rotation.from_quat(quats[:,[1,2,3,0]]).as_matrix()
            pose_increment_speed=(None if previous_part_rotations is None else
                Rotation.from_matrix(part_rotations@np.swapaxes(previous_part_rotations,-1,-2)).magnitude()/dt)
            previous_part_rotations=part_rotations.copy()
            rn=part_rotations[1]; relative=frame.T@rn
            wrapped=float(np.arctan2(relative[1,0],relative[0,0]))
            unwrapped+=float(np.arctan2(np.sin(wrapped-previous),np.cos(wrapped-previous))); previous=wrapped
            f,points,normals,separations,counts,starts,ids=contacts.get_raw_contact_data()
            f,points,normals,separations,counts,starts=(host(x) for x in (f,points,normals,separations,counts,starts))
            f=f.ravel();separations=separations.ravel(); nw=[]; penetration=[]; normal_load=[]
            for body_index,(count,start) in enumerate(zip(counts.ravel(),starts.ravel())):
                count,start=int(count),int(start); sl=slice(start,start+count); F=f[sl,None]*normals[sl]/dt
                nw.append(np.r_[F.sum(0),np.cross(points[sl]-positions[body_index],F).sum(0)])
                penetration.append(float(np.maximum(-separations[sl],0).max()) if count else 0.)
                normal_load.append(float(np.linalg.norm(F,axis=1).sum()))
            fr,fp,fc,fs=(host(x) for x in contacts.get_friction_data()); fr=fr/dt; fw=np.zeros((2,6))
            for body_index in range(2):
                chunks=[np.arange(int(start),int(start+count)) for count,start in
                        zip(fc[body_index].ravel(),fs[body_index].ravel()) if count]
                if chunks:
                    ix=np.concatenate(chunks)
                    fw[body_index]=np.r_[fr[ix].sum(0),np.cross(fp[ix]-positions[body_index],fr[ix]).sum(0)]
            w=np.asarray(nw)+fw
            command=(args.angle_deg*fraction if phase=='rotation' else args.angle_deg if phase in ('loaded_hold','free_hold') else 0.) if test_motor else float(drive.GetTargetPositionAttr().Get())-drive_offset_deg
            depth=float(socket[2]-positions[0,2]); contact_torque=float(w[1,3:]@axis)
            row={'step':step,'time_s':float(world.current_time),'phase':phase,'command_deg':command,
                 'nut_angle_deg':float(np.rad2deg(unwrapped)),'body_depth_m':depth,
                 'external_applied_torque_world_nm':(axis*applied_torque_nm).tolist(),
                 'positions_world_m':positions.tolist(),'quaternions_wxyz':quats.tolist(),
                 'native_linear_velocity_m_s':linear.tolist(),'native_angular_velocity_rad_s':angular.tolist(),
                 'pose_increment_angular_speed_rad_s':None if pose_increment_speed is None else pose_increment_speed.tolist(),
                 'pose_speed_used_only_for_independent_abort_not_actuator_feedback':True,
                 'native_link_velocity_crosscheck':(read_link_velocity_crosscheck(link_view,link_indices) if link_view is not None else None),
                 'normal_wrenches_n_nm':np.asarray(nw).tolist(),'friction_wrenches_n_nm':fw.tolist(),
                 'normal_counts':counts.ravel().tolist(),'normal_loads_n':normal_load,
                 'native_net_contact_force_n':host(contacts.get_net_contact_forces(dt=dt)).tolist(),
                 'max_contact_penetration_m':penetration,'nut_contact_axis_torque_nm':contact_torque,
                 'drive_spring_only_estimate_nm':(None if test_motor or direct_torque else float(np.clip(K*(np.deg2rad(command)-unwrapped),-args.torque_cap_nm,args.torque_cap_nm)) if phase!='free_hold' else 0.),
                 'external_guide_enabled':bool(guide.GetJointEnabledAttr().Get()),
                 'axial_drive_cap_n':float(axial.GetMaxForceAttr().Get()) if axial else 0.}
            if instrumentation is not None:
                row['instrumented_guide_wrench']=read_instrumented_guide(instrumentation)
                row['instrumented_tightening_resistance_nm']=-row['instrumented_guide_wrench']['external_axis_torque_nm']
            if step%max(1,round(.05/dt))==0:
                row['native_scene_statistics']=native_scene_statistics()
                row['native_filtered_normal_force_sum_n']=host(contacts.get_contact_force_matrix(dt=dt)).sum(axis=1).tolist()
                actor_ids=host(ids).ravel()
                row['raw_normal_actor_groups']=[]
                for sensor,(count,start) in enumerate(zip(counts.ravel(),starts.ravel())):
                    ix=np.arange(int(start),int(start+count));unique=np.unique(actor_ids[ix])
                    if len(unique):
                        paths=contacts._physics_rigid_contact_view.get_other_actor_paths_from_ids(
                            wp.array(unique,dtype=wp.uint64,device='cpu'))
                        row['raw_normal_actor_groups'].append([
                            {'actor':path,'force_n':(f[ix[actor_ids[ix]==ident],None]*normals[ix[actor_ids[ix]==ident]]/dt).sum(0).tolist()}
                            for ident,path in zip(unique,paths)])
                    else:row['raw_normal_actor_groups'].append([])
                row['filtered_contacts']=read_filtered_contacts(contacts,dt,positions,active_only=True)
                if args.full_shape_contact_report:
                    pairs=read_shape_contact_pairs(contact_interface,dt,body_path,positions[0])
                    row['body_shape_contacts']=group_shape_pairs(pairs,own_groups,other_groups)
                    row['body_shape_contact_pairs']=pairs
            if raw_depth_milestones and depth>=raw_depth_milestones[0]:
                milestone=raw_depth_milestones.pop(0)
                np.savez_compressed(out/f'raw_depth_{milestone*1000:.1f}mm.npz',
                    step=step,positions_world_m=positions,quaternions_wxyz=quats,
                    normal_impulse_ns=f,points_world_m=points,normals_world=normals,
                    separations_m=separations,counts=counts,starts=starts,
                    friction_force_world_n=fr,friction_points_world_m=fp,
                    friction_counts=fc,friction_starts=fs)
            measured_times['readback_and_reduction_s']+=time.perf_counter()-measured_tick;measured_tick=time.perf_counter()
            stream.write(json.dumps(row,separators=(',',':'))+'\n');rows.append(row)
            measured_times['recording_s']+=time.perf_counter()-measured_tick
            peaks['contact_torque_nm']=max(peaks['contact_torque_nm'],abs(contact_torque))
            peaks['body_axial_n']=max(peaks['body_axial_n'],abs(float(w[0,:3]@axis)))
            if step%max(1,round(1/args.video_fps/dt))==0:
                measured_tick=time.perf_counter();capture(phase,step);measured_times['evidence_s']+=time.perf_counter()-measured_tick
            if step%round(2/dt)==0:
                print(json.dumps({'phase':phase,'t':row['time_s'],'depth_mm':depth*1000,'nut_deg':row['nut_angle_deg'],'torque':contact_torque}),flush=True)
                (out/'progress.json').write_text(json.dumps(row,indent=2)+'\n')
                (out/'timing_progress.json').write_text(json.dumps(measured_times,indent=2)+'\n')
            if not np.isfinite(positions).all():
                raise RuntimeError('native observation is not finite')
            if ((direct_torque or args.torque_only_guide) and pose_increment_speed is not None
                    and float(np.max(pose_increment_speed))>5.):
                raise RuntimeError('Torque-only diagnostic stopped at independent 5 rad/s speed limit')
            step+=1
    capture('final',step);video.release();stream.close();frame_audit.close();world.pause()
    (out/'video_frames.json').write_text(json.dumps(images,indent=2)+'\n')
    free=[r for r in rows if r['phase']=='free_hold']
    result={'scope':'COMPLETE_CONNECTOR_MODEL_APPARATUS_TEST_NOT_ROBOT_OR_HARDWARE',
            'source_scene':str(args.source),'configuration':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            'source_mass_properties':mass_before,'seal':seal_record,'copied_roots':copied,
            'instrumented_guide':instrumentation['metadata'] if instrumentation else None,
            'frozen_model_installation':installed_model,
            'local_com_m':[list(UsdPhysics.MassAPI(x).GetCenterOfMassAttr().Get()) for x in parts],
            'post_start_object_pose_writes':False,'screw_lead_constraint':False,
            'drive_estimate_scope':'SMALL_ANGLE_TRACKING_ESTIMATE_NOT_NATIVE_ACTUATOR_REACTION',
            'force_signal_scope':('CPU_REPORTED_NORMAL_AND_FRICTION_CHANNELS_NOT_A_VERIFIED_TOTAL_WRENCH_OR_HAND_LOAD_REQUIREMENT'
                                  if args.physics_device=='cpu' else 'NATIVE_REPORTED_CONTACT_CHANNELS_NOT_ACTUATOR_TORQUE'),
            'test_motor_scope':('Temporary actuator between Body and Nut, reaction torque on Body; exported model remains passive; not a robot torque certificate' if test_motor else None),
            'axial_drive_during_rotation':False,'releases':phase_releases,
            'initial_body_depth_m':rows[0]['body_depth_m'],'final_body_depth_m':rows[-1]['body_depth_m'],
            'actual_nut_angle_deg':rows[-1]['nut_angle_deg'],'maximum_measured':peaks,
            'free_hold_depth_range_m':max(r['body_depth_m'] for r in free)-min(r['body_depth_m'] for r in free),
            'free_hold_depth_change_m':free[-1]['body_depth_m']-free[0]['body_depth_m'],
            'wall_seconds':time.perf_counter()-wall_start,'success_requires_postrun_physical_review':True}
    result['measured_wall_time_components_s']=measured_times
    if args.torque_only_guide:
        result['torque_only_apparatus']=report['torque_only_apparatus']
    if args.free_engagement_duration_s is not None:
        result['scope']='SHORT_FREE_ENGAGEMENT_CAUSAL_DIAGNOSTIC_NOT_ASSEMBLY_OR_DELIVERY_VALIDATION'
        result['free_engagement_diagnostic']=report['free_engagement_diagnostic']
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
except Exception:
    import traceback
    failed=True;error=traceback.format_exc();(out/'error.txt').write_text(error);print(error,flush=True)
    (out/'failure_timing.json').write_text(json.dumps({
        'elapsed_since_app_ready_s':time.perf_counter()-wall_start,
        'components_s':globals().get('measured_times'),
        'completed_loop_steps':globals().get('step')},indent=2)+'\n')
    if 'video' in globals():video.release()
    if 'stream' in globals():stream.close()
    if 'frame_audit' in globals():frame_audit.close()
finally:
    if profiler is not None:
        profiler.disable();profiler.dump_stats(str(out/'python_call_times.pstats'))
    app.close(exit_code=1 if failed else 0)
raise SystemExit(1 if failed else 0)
