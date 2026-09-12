#!/usr/bin/env python3
"""Short open-loop lateral-load / loose-nut diagnostic of a frozen connector.

Only the socket is fixed. The optional finite rotary test actuator has no
lateral, axial or tilt constraints and is disabled during disturbance loading.
The force schedule is an explicit laboratory load, not hand control or a Body
pose servo. No state-dependent pose correction is applied.
Run through timeout with a wall limit below 180 s, including startup/output.
"""
import argparse
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--initial-pose', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--install-into-source', type=Path,
                        help='Exercise the production frozen-model installer on the original scene before physics.')
    parser.add_argument('--first-turn-deg', type=float, default=0.,
                        help='First short physical turn, starting at the declared thread-entry pose.')
    parser.add_argument('--first-turn-duration-s', type=float, default=.7)
    parser.add_argument('--first-turn-torque-cap-nm', type=float, default=.12)
    parser.add_argument('--oscillation-deg', type=float, default=5.)
    parser.add_argument('--release-hold-s', type=float, default=.1,
                        help='Passive hold after the first turn and at the end, up to 3 s.')
    parser.add_argument('--loaded-retention', action='store_true',
                        help='Continue tightening, release the rotary tool, then apply a fixed six-axis load basis to Body.')
    parser.add_argument('--tighten-deg', type=float, default=230.)
    parser.add_argument('--tighten-duration-s', type=float, default=2.5)
    parser.add_argument('--retention-load-set', choices=('all','body_torsion','side_pair','challenge','reverse_torsion'), default='all')
    parser.add_argument('--retention-hold-s', type=float, default=.06,
                        help='Full-load plateau, with the original30ms loading/unloading ramps.')
    parser.add_argument('--velocity-iterations', type=int, choices=(1, 4), default=1,
                        help='Bounded solver-convergence diagnostic; changes no material or load.')
    parser.add_argument('--continue-tightening', action='store_true',
                        help='After the existing 260deg profile, diagnose another 70deg with a 0.6Nm cap; stop at 5N total pin contact.')
    parser.add_argument('--continued-retention', action='store_true',
                        help='After reviewing first pin entry, continue only 30deg then test Body retention; keep a 20N pin-load review stop.')
    parser.add_argument('--record-pin-entry', action='store_true',
                        help='After offline entry review, record its load-threshold event without treating it as a physical jam; all input/state/time limits remain.')
    parser.add_argument('--direct-retention', action='store_true',
                        help='One bounded velocity stroke with nominal40-to-290deg travel and0.6Nm cap, then the declared retention test; avoids position-error catch-up.')
    args = parser.parse_args()
    if not (0 <= args.first_turn_deg <= 60 and .3 <= args.first_turn_duration_s <= 1.5
            and 0 < args.first_turn_torque_cap_nm <= .25 and 0 < args.oscillation_deg <= 10
            and .1 <= args.release_hold_s <= 3.):
        parser.error('Finite first-turn load and duration required')
    if args.loaded_retention and not (args.first_turn_deg==40. and 60.<=args.tighten_deg<=(290. if args.direct_retention else 260.)
                                     and 1.5<=args.tighten_duration_s<=3. and .06<=args.retention_hold_s<=.6):
        parser.error('Loaded retention requires the existing first40deg and a finite additional profile')
    if args.continue_tightening and not (args.loaded_retention and args.tighten_deg==260.):
        parser.error('Continuation requires the existing 40deg then 260deg profile')
    if args.continued_retention and not args.continue_tightening:
        parser.error('Continued retention requires the finite continuation profile')
    if args.record_pin_entry and not args.continue_tightening:
        parser.error('Pin-entry event recording is only for the reviewed continuation diagnostic')
    if args.direct_retention and not (args.loaded_retention and args.tighten_deg==290. and not args.continue_tightening):
        parser.error('Direct retention requires a single loaded 40-to-290deg stroke')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'driver_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (args.output/'contact_reader_snapshot.py').write_bytes(Path(__file__).with_name('te_fast_contact_reading.py').read_bytes())
    started = time.monotonic()
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'multi_gpu': False, 'fast_shutdown': True})
    failed = False
    try:
        import carb
        import cv2
        import numpy as np
        import omni.usd
        import omni.replicator.core as rep
        import omni.timeline
        from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, PhysicsSchemaTools
        from scipy.spatial.transform import Rotation
        from isaacsim.core.api import World
        from isaacsim.core.experimental.prims import RigidPrim
        from isaacsim.core.simulation_manager import SimulationManager
        from omni.physx import get_physx_simulation_interface
        from omni.physx.bindings._physx import SETTING_DISABLE_CONTACT_PROCESSING
        from te_foundationpose_handoff_runtime import _author_camera, _camera_cv_pose_from_eye_target
        repo = Path(__file__).resolve().parents[3]
        sys.path.insert(0, str(repo/'artifacts/kcg_connector/model_delivery_20260908/src'))
        from contact_reading import read_shape_contact_pairs
        from te_fast_contact_reading import read_shape_contact_pairs_fast

        dt = 1/960.
        SimulationManager.set_physics_sim_device('cpu')
        carb.settings.get_settings().set_bool(SETTING_DISABLE_CONTACT_PROCESSING, False)
        world = World(stage_units_in_meters=1., physics_dt=dt, rendering_dt=dt,
                      backend='numpy', device='cpu', sim_params={'use_gpu_pipeline': False})
        stage = omni.usd.get_context().get_stage()
        UsdGeom.Xform.Define(stage, '/World')
        source = Usd.Stage.Open(str((args.install_into_source or args.model).resolve()))
        source_layer = source.Flatten()
        for child in source.GetPrimAtPath('/World').GetChildren():
            if child.IsA(UsdPhysics.Scene) or child.GetName() in ('HandArm', 'FixtureMaterial'):
                continue
            if not Sdf.CopySpec(source_layer, child.GetPath(), stage.GetRootLayer(), child.GetPath()):
                raise RuntimeError('Could not copy frozen model')
        if args.install_into_source:
            import importlib.util
            adapter=args.model.resolve().with_name('install_model.py')
            spec=importlib.util.spec_from_file_location('engagement_model_installer',adapter)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            installed=module.install_model(stage,model_path=args.model)
            (args.output/'frozen_model_installation.json').write_text(json.dumps(installed,indent=2)+'\n')
        scene = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath(world.get_physics_context().prim_path))
        scene.CreateEnableGPUDynamicsAttr(False)
        scene.CreateBroadphaseTypeAttr('MBP')
        scene.CreateSolverTypeAttr('TGS')
        scene.CreateEnableExternalForcesEveryIterationAttr(True)
        scene.CreateMinVelocityIterationCountAttr(args.velocity_iterations)
        scene.CreateMaxVelocityIterationCountAttr(args.velocity_iterations)
        body = '/World/TE_J35FreeSplitPlug/Body'
        nut = '/World/TE_J35FreeSplitPlug/CouplingNut'
        socket = '/World/TEVisualHandoff/FixedReceptaclePose/OfficialVisual/Geometry'
        pose = json.loads(args.initial_pose.read_text())
        UsdGeom.Xformable(stage.GetPrimAtPath(body).GetParent()).ClearXformOpOrder()
        mass_before = {}
        for path, position, quat in zip((body, nut), pose['positions_world_m'], pose['quaternions_wxyz']):
            prim = stage.GetPrimAtPath(path)
            mass_before[path] = {n: str(prim.GetAttribute(n).Get()) for n in
                                ('physics:mass', 'physics:centerOfMass', 'physics:diagonalInertia', 'physics:principalAxes')}
            xf = UsdGeom.Xformable(prim); xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(*position))
            xf.AddOrientOp().Set(Gf.Quatf(quat[0], Gf.Vec3f(*quat[1:])))
            UsdPhysics.RigidBodyAPI(prim).CreateVelocityAttr(Gf.Vec3f(0.))
            UsdPhysics.RigidBodyAPI(prim).CreateAngularVelocityAttr(Gf.Vec3f(0.))
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI): prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            if prim.HasAPI(PhysxSchema.PhysxArticulationAPI): prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
            rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            rb.CreateSolverPositionIterationCountAttr(128); rb.CreateSolverVelocityIterationCountAttr(args.velocity_iterations)
            rb.CreateSleepThresholdAttr(0.)
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
        joint = stage.GetPrimAtPath('/World/TE_J35FreeSplitPlug/Joints/CouplingNutRevolute')
        captive_play=joint.GetAttribute('kcg:captiveNutAxialPlayM').Get()
        resistance = float(joint.GetAttribute('kcg:passiveResistanceNm' if captive_play else 'physxJointAxis:angular:dynamicFrictionEffort').Get())
        if abs(resistance-.02) > 1e-6: raise ValueError('Unexpected passive resistance')
        bearing_rotation_axis = joint.GetAttribute('kcg:rotationCoordinate').Get() or ('rotZ' if captive_play else 'angular')
        drive = UsdPhysics.DriveAPI.Apply(joint, bearing_rotation_axis)
        drive.CreateTypeAttr('force'); drive.CreateStiffnessAttr(0.)
        drive.CreateDampingAttr(100.*np.pi/180.); drive.CreateMaxForceAttr(resistance)
        drive.CreateTargetVelocityAttr(0.)
        for p in stage.Traverse():
            if p.IsA(UsdPhysics.Joint) and p.GetPath() != joint.GetPath():
                targets = [str(t) for rel in (UsdPhysics.Joint(p).GetBody0Rel(), UsdPhysics.Joint(p).GetBody1Rel()) for t in rel.GetTargets()]
                if body in targets or nut in targets: raise ValueError('Undeclared connector attachment')
        socket_transform = np.asarray(UsdGeom.Xformable(stage.GetPrimAtPath(socket)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
        socket_origin = socket_transform[:3, 3]
        turn_joint = None
        if args.first_turn_deg:
            # Explicit laboratory torque actuator, with all translation and
            # swing coordinates free. It is disabled throughout side loading.
            turn_joint = UsdPhysics.Joint.Define(stage, '/World/DeclaredFirstTurnTorqueActuator')
            turn_joint.CreateBody1Rel().SetTargets([stage.GetPrimAtPath(nut).GetPath()])
            turn_joint.CreateExcludeFromArticulationAttr(True)
            turn_joint.CreateLocalPos0Attr(Gf.Vec3f(*pose['positions_world_m'][1]))
            quat = pose['quaternions_wxyz'][1]
            turn_joint.CreateLocalRot0Attr(Gf.Quatf(quat[0], Gf.Vec3f(*quat[1:])))
            turn_joint.CreateLocalPos1Attr(Gf.Vec3f(0.)); turn_joint.CreateLocalRot1Attr(Gf.Quatf(1.))
            turn_drive = UsdPhysics.DriveAPI.Apply(turn_joint.GetPrim(), 'rotZ')
            turn_drive.CreateTypeAttr('force'); turn_drive.CreateStiffnessAttr(0.)
            turn_drive.CreateDampingAttr(0.); turn_drive.CreateMaxForceAttr(0.)
            turn_drive.CreateTargetPositionAttr(0.); turn_drive.CreateTargetVelocityAttr(0.)
            turn_joint.CreateJointEnabledAttr(False)
        view = RigidPrim([body, nut], resolve_paths=False, contact_filter_paths=[socket], max_contact_count=32768)
        camera = '/World/EngagementEvidenceCamera'
        _author_camera(stage, camera, _camera_cv_pose_from_eye_target(socket_origin+[.11, -.13, .085], socket_origin+[0, 0, .006]),
                       resolution=(800, 600), focal_length_mm=32., horizontal_aperture_mm=36., clipping_range_m=(.01, 5.), Gf=Gf, UsdGeom=UsdGeom)
        UsdLux.DomeLight.Define(stage, '/World/EngagementEvidenceLight').CreateIntensityAttr(1000.)
        product = rep.create.render_product(camera, (800, 600))
        rgb = rep.AnnotatorRegistry.get_annotator('rgb'); rgb.attach([product.path])
        stage.Flatten().Export(str(args.output/'before_physics.usdc'))
        world.reset()
        native_shapes = {'actor_paths': [body, nut],
                         'max_shapes_per_actor': view._physics_rigid_body_view.max_shapes,
                         'usd_extra_source_key_colliders': sum(p.GetName().startswith('SourceGuideKey_') for p in stage.Traverse())}
        (args.output/'native_shape_count.json').write_text(json.dumps(native_shapes,indent=2)+'\n')
        print(json.dumps(native_shapes),flush=True)
        carb.logging.acquire_logging().set_level_threshold_for_source('omni.physx.plugin', carb.logging.LogSettingBehavior.OVERRIDE, carb.logging.LEVEL_ERROR)
        interface = get_physx_simulation_interface()
        # Reuse one complete native contact snapshot for both actors. This
        # removes duplicate reads/path decoding, not contact processing or samples.
        class ContactSnapshot:
            data = None
            def get_full_contact_report(self): return self.data
        snapshot=ContactSnapshot();decoded_paths={}
        contact_reader_checks=[]
        def decode_path(value):
            key=int(value)
            if key not in decoded_paths:decoded_paths[key]=str(PhysicsSchemaTools.intToSdfPath(value))
            return decoded_paths[key]
        def host(value): return value.numpy() if hasattr(value, 'numpy') else np.asarray(value)
        def state(): return np.concatenate([host(a).ravel() for a in (*view.get_world_poses(), *view.get_velocities())])
        image_rows = []
        def capture(phase):
            before = state().copy(); before_time = float(world.current_time)
            timeline = omni.timeline.get_timeline_interface(); auto = timeline.is_auto_updating()
            settings = carb.settings.get_settings(); play = settings.get('/app/player/playSimulations')
            try:
                timeline.set_auto_update(False); timeline.commit_silently(); settings.set('/app/player/playSimulations', False)
                for _ in range(3): omni.kit.app.get_app().update()
                settings.set('/app/player/playSimulations', play)
                rep.orchestrator.step(rt_subframes=1, delta_time=0., pause_timeline=False)
            finally:
                settings.set('/app/player/playSimulations', play); timeline.set_auto_update(auto); timeline.commit_silently()
            pixels = np.asarray(rgb.get_data())
            if pixels.ndim != 3 or not np.array_equal(before, state()) or before_time != float(world.current_time):
                raise RuntimeError('Evidence capture changed native state or produced no image')
            path = args.output/f'{phase}.png'
            cv2.imwrite(str(path), cv2.cvtColor(pixels[:, :, :3], cv2.COLOR_RGB2BGR))
            image_rows.append({'phase': phase, 'time_s': before_time, 'path': str(path.resolve())})
        # One prescribed wrench, equivalent to 10 N applied 20 mm above the
        # Body prim origin. The native API's omitted position is the link
        # transform location, not the center of mass.
        # Both components ramp together; no position feedback and no gain sweep.
        sequence = [('settle', .05), ('side_load', .12), ('release', .10), ('nut_forward', .10), ('nut_reverse', .10), ('free_hold', .10)]
        if turn_joint:
            sequence = [('settle', .05), ('first_turn', args.first_turn_duration_s),
                        ('after_turn_release', args.release_hold_s), ('side_load', .12), ('release', .10),
                        ('nut_reverse', .18), ('nut_forward', .18), ('free_hold', args.release_hold_s)]
        load_cases={}
        if args.loaded_retention:
            sequence=[('settle',.05),('first_turn',args.first_turn_duration_s),('first_turn_hold',.10),
                      ('tighten',args.tighten_duration_s),('tighten_hold',.20),('preload_release',.15)]
            kinds=[('side',10.)] if args.retention_load_set=='side_pair' else [('force',10.),('moment',.2)]
            for kind,amplitude in kinds:
                for axis in range(2 if kind=='side' else 3):
                    if args.retention_load_set=='body_torsion' and (kind!='moment' or axis!=2):continue
                    directions=([(-1.,'minus'),(1.,'plus')] if args.retention_load_set=='body_torsion' else [(1.,'plus'),(-1.,'minus')])
                    for sign,label in directions:
                        name=f'{kind}_{"xyz"[axis]}_{label}'
                        load_cases[name]=(kind,axis,sign*amplitude)
                        sequence.extend([(name,.06+args.retention_hold_s),(name+'_unload',.06)])
            if args.retention_load_set in ('challenge','reverse_torsion'):
                load_cases=({'side_x_plus':('side',0,10.),'moment_z_plus':('moment',2,.2)}
                            if args.retention_load_set=='challenge' else {'moment_z_plus':('moment',2,.2)})
                sequence=sequence[:6]
                for name in load_cases:sequence.extend([(name,.06+args.retention_hold_s),(name+'_unload',.06)])
            sequence.append(('free_hold',.20))
        if args.continue_tightening:
            # A 1.5 s quintic 70deg stroke has a 1.53 rad/s maximum target
            # speed. The initial 0.75 s diagnostic exceeded the independent
            # 5 rad/s measured-speed stop during stick/slip, before pin entry.
            continuation_delta=30. if args.continued_retention else 70.
            continuation_duration=.75 if args.continued_retention else 1.5
            pin_review_stop=20. if args.continued_retention else 5.
            sequence=sequence[:6]+[('continue_tighten',continuation_duration),('continue_hold',.10)]
            if args.continued_retention:
                sequence.append(('continued_preload_release',.15))
                load_cases={'moment_z_plus':('moment',2,.2)}
                sequence.extend([('moment_z_plus',.06+args.retention_hold_s),('moment_z_plus_unload',.06)])
            else:load_cases={}
            sequence.append(('free_hold',.20))
            (args.output/'continuation_scope.json').write_text(json.dumps({
                'scope':'FINITE_CONTINUATION_DIAGNOSTIC_NOT_MATING_QUALIFICATION',
                'unchanged_prefix_target_deg':260.,'continuation_target_deg':260.+continuation_delta,
                'continuation_duration_s':continuation_duration,
                'continuation_torque_cap_nm':.6,'total_pin_normal_load_review_stop_n':pin_review_stop,
                'retention_after_reviewed_pin_entry':args.continued_retention,
                'pin_threshold_is_record_only':args.record_pin_entry,
                'cap_basis':'Bounded diagnostic input below the project primary-source MIL-DTL-38999N table VI shell25 4.6Nm maximum; 0.6Nm is not a recommended mating torque or specimen calibration.',
                'reference':'artifacts/kcg_connector/reference/full_assembly_20260905/dtl38999.txt',
                'geometry_material_joint_mass_and_robot_controls_changed':False},indent=2)+'\n')
        rows = []; previous = None
        pin_entry_event_recorded=False
        previous_nut_wrapped=0.;continuous_nut_angle=0.
        initial_nut_rotation=Rotation.from_quat(np.asarray(pose['quaternions_wxyz'][1])[[1,2,3,0]]).as_matrix()
        measured_wall={'physics_s':0.,'readback_s':0.,'evidence_s':0.}
        with (args.output/'samples.jsonl').open('x', buffering=1) as stream:
            for phase, duration in sequence:
                if turn_joint:
                    rotating = phase in ('first_turn', 'first_turn_hold', 'tighten', 'tighten_hold', 'continue_tighten', 'continue_hold', 'nut_reverse', 'nut_forward')
                    velocity_stroke=args.direct_retention and phase in ('tighten','tighten_hold')
                    turn_joint.CreateJointEnabledAttr(rotating)
                    turn_drive.GetStiffnessAttr().Set(30.*np.pi/180. if rotating and not velocity_stroke else 0.)
                    turn_drive.GetDampingAttr().Set((10. if velocity_stroke else .1)*np.pi/180. if rotating else 0.)
                    turn_drive.GetTargetVelocityAttr().Set(0.)
                    cap=(.6 if phase in ('continue_tighten','continue_hold') or (args.direct_retention and phase in ('tighten','tighten_hold')) else .25 if phase in ('tighten','tighten_hold') else args.first_turn_torque_cap_nm if phase in ('first_turn','first_turn_hold') else .05)
                    turn_drive.GetMaxForceAttr().Set(cap if rotating else 0.)
                    interface.flush_changes()
                for i in range(round(duration/dt)):
                    ramp = min(1., (i+1)*dt/.03)
                    forces = np.zeros((2, 3)); torques = np.zeros((2, 3))
                    if phase == 'side_load': forces[0, 0] = 10.*ramp; torques[0, 1] = .2*ramp
                    if phase in load_cases:
                        kind,axis,value=load_cases[phase]
                        # 30ms ramp up, declared plateau, 30ms ramp down; no
                        # retightening or pose reset between the prescribed cases.
                        elapsed=(i+1)*dt
                        scale=min(1.,elapsed/.03,max(0.,(duration-elapsed)/.03))
                        if kind=='side':
                            forces[0,axis]=value*scale
                            torques[0]=np.cross([0.,0.,.02],forces[0])
                        else:
                            (forces if kind=='force' else torques)[0,axis]=value*scale
                    if turn_joint:
                        u = (i+1)/round(duration/dt); blend = 10*u**3-15*u**4+6*u**5
                        if phase == 'first_turn': target = args.first_turn_deg*blend
                        elif phase == 'tighten': target = args.first_turn_deg+(args.tighten_deg-args.first_turn_deg)*blend
                        elif phase == 'continue_tighten': target = args.tighten_deg+continuation_delta*blend
                        elif phase == 'nut_reverse': target = args.first_turn_deg-args.oscillation_deg*blend
                        elif phase == 'nut_forward': target = args.first_turn_deg-args.oscillation_deg+args.oscillation_deg*blend
                        else: target = float(turn_drive.GetTargetPositionAttr().Get())
                        turn_drive.GetTargetPositionAttr().Set(target)
                        if velocity_stroke and phase=='tighten':
                            # Time-only finite motor command, no accumulated
                            # position error after a frictional pause. Its
                            # integrated free travel is the declared angle;
                            # actual advance still depends on native contacts.
                            elapsed=(i+1)*dt;edge=min(1.,elapsed/.1,max(0.,(duration-elapsed)/.1))
                            blend_velocity=edge*edge*(3.-2.*edge)
                            turn_drive.GetTargetVelocityAttr().Set((args.tighten_deg-args.first_turn_deg)/(duration-.1)*blend_velocity)
                    elif phase in ('nut_forward', 'nut_reverse'): torques[1, 2] = (-1 if phase == 'nut_forward' else 1)*.05*ramp
                    view.apply_forces_and_torques_at_pos(forces=forces, torques=torques)
                    tick=time.monotonic()
                    world.step(render=False)
                    measured_wall['physics_s']+=time.monotonic()-tick;tick=time.monotonic()
                    pos, quat = (host(x).copy() for x in view.get_world_poses())
                    rotation = Rotation.from_quat(quat[:, [1, 2, 3, 0]])
                    speed = None if previous is None else (rotation*previous.inv()).magnitude()/dt
                    previous = rotation
                    relative = rotation[0].inv()*rotation[1]
                    nut_from_initial=initial_nut_rotation.T@rotation[1].as_matrix()
                    nut_angle=float(np.degrees(np.arctan2(nut_from_initial[1,0],nut_from_initial[0,0])))
                    if args.continue_tightening:
                        continuous_nut_angle+=(nut_angle-previous_nut_wrapped+180.)%360.-180.
                        previous_nut_wrapped=nut_angle;nut_angle=continuous_nut_angle
                    elif args.loaded_retention and nut_angle< -30.:nut_angle+=360.
                    angle_error=(float(turn_drive.GetTargetPositionAttr().Get())-nut_angle) if turn_joint else 0.
                    spring_estimate=(float(np.clip(30.*2.*np.sin(np.radians(angle_error)/2.),
                                      -float(turn_drive.GetMaxForceAttr().Get()),float(turn_drive.GetMaxForceAttr().Get())))
                                     if turn_joint and turn_joint.GetJointEnabledAttr().Get() and not velocity_stroke else 0.)
                    snapshot.data=interface.get_full_contact_report()
                    shape_pairs=[read_shape_contact_pairs_fast(snapshot,dt,path,pos[j],decode_path=decode_path)
                                 for j,path in enumerate((body,nut))]
                    if i==0:
                        difference=0.
                        for j,path in enumerate((body,nut)):
                            legacy=read_shape_contact_pairs(snapshot,dt,path,pos[j],decode_path=decode_path)
                            if len(legacy)!=len(shape_pairs[j]):raise RuntimeError('Contact reader lost a shape pair')
                            for old,new in zip(legacy,shape_pairs[j]):
                                for key,value in old.items():
                                    if key in ('normal_wrench_n_nm','friction_wrench_n_nm','normal_load_n'):
                                        difference=max(difference,float(np.max(np.abs(np.asarray(value)-new[key]))))
                                    elif value!=new[key]:raise RuntimeError('Contact reader metadata changed: '+key)
                        if difference>1e-9:raise RuntimeError('Contact reader wrench mismatch')
                        contact_reader_checks.append({'phase':phase,'maximum_absolute_numeric_difference':difference})
                        (args.output/'contact_reader_crosscheck.json').write_text(json.dumps(contact_reader_checks,indent=2)+'\n')
                    row = {'time_s': float(world.current_time), 'phase': phase, 'positions_world_m': pos.tolist(), 'quaternions_wxyz': quat.tolist(),
                           'applied_forces_world_n': forces.tolist(), 'applied_torques_world_nm': torques.tolist(),
                           'lateral_m': float(np.linalg.norm(pos[0, :2]-socket_origin[:2])), 'depth_m': float(socket_origin[2]-pos[0, 2]),
                           'tilt_deg': float(np.degrees(np.arccos(np.clip(-rotation[0].as_matrix()[2, 2], -1, 1)))),
                           'nut_relative_deg': float(np.degrees(np.arctan2(relative.as_matrix()[1, 0], relative.as_matrix()[0, 0]))),
                           'nut_axial_offset_in_body_m': float((rotation[0].as_matrix().T@(pos[1]-pos[0]))[2]),
                           'shape_contacts': shape_pairs[0],
                           'nut_shape_contacts': shape_pairs[1],
                           'nut_angle_about_socket_axis_deg':nut_angle,
                           'rotary_spring_effort_estimate_nm':spring_estimate,
                           'effort_scope':'Position-spring term only (zero during velocity stroke); excludes damping/implicit effects, not a force sensor',
                           'pose_increment_speed_rad_s':None if speed is None else speed.tolist(),
                           'native_linear_velocity_m_s': host(view.get_velocities()[0]).tolist(),
                           'native_angular_velocity_rad_s': host(view.get_velocities()[1]).tolist(),
                           'rotary_actuator_enabled': bool(turn_joint.GetJointEnabledAttr().Get()) if turn_joint else False,
                           'rotary_actuator_mode': 'velocity' if turn_joint and velocity_stroke else 'position' if turn_joint and rotating else 'disabled',
                           'rotary_actuator_target_velocity_deg_s':float(turn_drive.GetTargetVelocityAttr().Get()) if turn_joint else 0.,
                           'rotary_actuator_target_deg': float(turn_drive.GetTargetPositionAttr().Get()) if turn_joint else None,
                           'rotary_actuator_cap_nm': float(turn_drive.GetMaxForceAttr().Get()) if turn_joint else 0.}
                    stream.write(json.dumps(row, separators=(',', ':'))+'\n'); rows.append(row)
                    measured_wall['readback_s']+=time.monotonic()-tick
                    if args.continue_tightening:
                        pin_load=sum(c['normal_load_n'] for c in row['shape_contacts'] if 'SourcePinSdf_' in c['own_collider'])
                        if pin_load>pin_review_stop and not pin_entry_event_recorded:
                            capture('pin_contact_review_stop')
                            (args.output/'pin_contact_review_stop.json').write_text(json.dumps({
                                'time_s':row['time_s'],'phase':phase,'pin_normal_load_sum_n':pin_load,'review_stop_n':pin_review_stop,
                                'body_depth_mm':row['depth_m']*1000.,'nut_angle_deg':nut_angle,
                                'record_only':args.record_pin_entry,
                                'reason':'Pin-entry load reached the declared diagnostic review threshold; not a hardware rating or successful mating.'},indent=2)+'\n')
                            pin_entry_event_recorded=True
                            if not args.record_pin_entry:raise RuntimeError('Declared pin-entry contact review stop')
                    if not np.isfinite(pos).all() or (speed is not None and max(speed)>5.) or row['lateral_m']>.002 or row['tilt_deg']>5.:
                        if np.isfinite(pos).all():capture('independent_state_stop')
                        raise RuntimeError('Independent finite state / speed / displacement stop')
                    if time.monotonic()-started > 155.: raise RuntimeError('Internal wall-clock stop; reserve time for output')
                    if phase in load_cases and i==round((duration-.03)/dt)-1:
                        tick=time.monotonic();capture(phase+'_peak');measured_wall['evidence_s']+=time.monotonic()-tick
                tick=time.monotonic();capture(phase);measured_wall['evidence_s']+=time.monotonic()-tick
                print(json.dumps({k: rows[-1][k] for k in ('phase', 'depth_m', 'lateral_m', 'tilt_deg', 'nut_relative_deg','nut_axial_offset_in_body_m','rotary_spring_effort_estimate_nm')}), flush=True)
        result = {'scope': 'SHORT_CONNECTOR_ENGAGEMENT_LOAD_TEST_NOT_ASSEMBLY_OR_HARDWARE', 'configuration': {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  'socket_origin_world_m': socket_origin.tolist(), 'mass_properties': mass_before, 'physics_hz': 960, 'position_iterations': 128,
                  'external_lateral_axial_or_tilt_constraint': False,
                  'finite_rotary_test_actuator': bool(turn_joint), 'post_start_pose_writes': False, 'prescribed_force_cap_n': 10., 'prescribed_bending_torque_cap_nm': .2,
                  'nut_torque_cap_nm': .05, 'wall_seconds': time.monotonic()-started, 'images': image_rows,
                  'maximum_lateral_m': max(r['lateral_m'] for r in rows), 'maximum_tilt_deg': max(r['tilt_deg'] for r in rows)}
        result['measured_wall_time_s']=measured_wall
        result['loaded_retention_protocol']={'enabled':args.loaded_retention,'tightening_torque_cap_nm':.6 if args.direct_retention else .25,
            'load_cases':load_cases,'rotary_actuator_disabled_during_all_loads':True,
            'state_reset_or_retighten_between_load_cases':False,
            'preload_scope':'AXIAL_ENDSTOP_AND_FINITE_TURN_DRIVER_ONLY_NOT_CALIBRATED_AXIAL_PRELOAD_OR_FULL_MATING',
            'preload_eligibility_requires_postrun_review':True}
        (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    except Exception:
        import traceback
        failed = True; error = traceback.format_exc()
        (args.output/'error.txt').write_text(error); print(error, flush=True)
        (args.output/'failure_timing.json').write_text(json.dumps({
            'elapsed_since_start_s':time.monotonic()-started,
            'components_s':locals().get('measured_wall'),
            'recorded_steps':len(locals().get('rows',[]))},indent=2)+'\n')
    finally:
        app.close(exit_code=1 if failed else 0)
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
