#!/usr/bin/env python3
"""Short open-loop lateral-load / loose-nut diagnostic of a frozen connector.

Only the socket is fixed. No external joint, pose correction or feedback drive
is added. The force schedule is an explicit laboratory load, not hand control.
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
    args = parser.parse_args()
    if not (0 <= args.first_turn_deg <= 60 and .3 <= args.first_turn_duration_s <= 1.5
            and 0 < args.first_turn_torque_cap_nm <= .25 and 0 < args.oscillation_deg <= 10):
        parser.error('Finite first-turn load and duration required')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'driver_snapshot.py').write_bytes(Path(__file__).read_bytes())
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
        from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics
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
        scene.CreateMinVelocityIterationCountAttr(1)
        scene.CreateMaxVelocityIterationCountAttr(1)
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
            rb.CreateSolverPositionIterationCountAttr(128); rb.CreateSolverVelocityIterationCountAttr(1)
            rb.CreateSleepThresholdAttr(0.)
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
        joint = stage.GetPrimAtPath('/World/TE_J35FreeSplitPlug/Joints/CouplingNutRevolute')
        captive_play=joint.GetAttribute('kcg:captiveNutAxialPlayM').Get()
        resistance = float(joint.GetAttribute('kcg:passiveResistanceNm' if captive_play else 'physxJointAxis:angular:dynamicFrictionEffort').Get())
        if abs(resistance-.02) > 1e-6: raise ValueError('Unexpected passive resistance')
        drive = UsdPhysics.DriveAPI.Apply(joint, 'rotZ' if captive_play else 'angular')
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
                        ('after_turn_release', .10), ('side_load', .12), ('release', .10),
                        ('nut_reverse', .18), ('nut_forward', .18), ('free_hold', .10)]
        rows = []; previous = None
        with (args.output/'samples.jsonl').open('x', buffering=1) as stream:
            for phase, duration in sequence:
                if turn_joint:
                    rotating = phase in ('first_turn', 'nut_reverse', 'nut_forward')
                    turn_joint.CreateJointEnabledAttr(rotating)
                    turn_drive.GetStiffnessAttr().Set(30.*np.pi/180. if rotating else 0.)
                    turn_drive.GetDampingAttr().Set(.1*np.pi/180. if rotating else 0.)
                    turn_drive.GetMaxForceAttr().Set((args.first_turn_torque_cap_nm if phase == 'first_turn' else .05) if rotating else 0.)
                    interface.flush_changes()
                for i in range(round(duration/dt)):
                    ramp = min(1., (i+1)*dt/.03)
                    forces = np.zeros((2, 3)); torques = np.zeros((2, 3))
                    if phase == 'side_load': forces[0, 0] = 10.*ramp; torques[0, 1] = .2*ramp
                    if turn_joint:
                        u = (i+1)/round(duration/dt); blend = 10*u**3-15*u**4+6*u**5
                        if phase == 'first_turn': target = args.first_turn_deg*blend
                        elif phase == 'nut_reverse': target = args.first_turn_deg-args.oscillation_deg*blend
                        elif phase == 'nut_forward': target = args.first_turn_deg-args.oscillation_deg+args.oscillation_deg*blend
                        else: target = float(turn_drive.GetTargetPositionAttr().Get())
                        turn_drive.GetTargetPositionAttr().Set(target)
                    elif phase in ('nut_forward', 'nut_reverse'): torques[1, 2] = (-1 if phase == 'nut_forward' else 1)*.05*ramp
                    view.apply_forces_and_torques_at_pos(forces=forces, torques=torques)
                    world.step(render=False)
                    pos, quat = (host(x).copy() for x in view.get_world_poses())
                    rotation = Rotation.from_quat(quat[:, [1, 2, 3, 0]])
                    speed = None if previous is None else (rotation*previous.inv()).magnitude()/dt
                    previous = rotation
                    relative = rotation[0].inv()*rotation[1]
                    row = {'time_s': float(world.current_time), 'phase': phase, 'positions_world_m': pos.tolist(), 'quaternions_wxyz': quat.tolist(),
                           'applied_forces_world_n': forces.tolist(), 'applied_torques_world_nm': torques.tolist(),
                           'lateral_m': float(np.linalg.norm(pos[0, :2]-socket_origin[:2])), 'depth_m': float(socket_origin[2]-pos[0, 2]),
                           'tilt_deg': float(np.degrees(np.arccos(np.clip(-rotation[0].as_matrix()[2, 2], -1, 1)))),
                           'nut_relative_deg': float(np.degrees(np.arctan2(relative.as_matrix()[1, 0], relative.as_matrix()[0, 0]))),
                           'nut_axial_offset_in_body_m': float((rotation[0].as_matrix().T@(pos[1]-pos[0]))[2]),
                           'shape_contacts': read_shape_contact_pairs(interface, dt, body, pos[0]),
                           'nut_shape_contacts': read_shape_contact_pairs(interface, dt, nut, pos[1]),
                           'native_linear_velocity_m_s': host(view.get_velocities()[0]).tolist(),
                           'native_angular_velocity_rad_s': host(view.get_velocities()[1]).tolist(),
                           'rotary_actuator_enabled': bool(turn_joint.GetJointEnabledAttr().Get()) if turn_joint else False,
                           'rotary_actuator_target_deg': float(turn_drive.GetTargetPositionAttr().Get()) if turn_joint else None,
                           'rotary_actuator_cap_nm': float(turn_drive.GetMaxForceAttr().Get()) if turn_joint else 0.}
                    stream.write(json.dumps(row, separators=(',', ':'))+'\n'); rows.append(row)
                    if not np.isfinite(pos).all() or (speed is not None and max(speed)>5.) or row['lateral_m']>.002 or row['tilt_deg']>5.:
                        raise RuntimeError('Independent finite state / speed / displacement stop')
                    if time.monotonic()-started > 155.: raise RuntimeError('Internal wall-clock stop; reserve time for output')
                capture(phase)
                print(json.dumps({k: rows[-1][k] for k in ('phase', 'depth_m', 'lateral_m', 'tilt_deg', 'nut_relative_deg')}), flush=True)
        result = {'scope': 'SHORT_CONNECTOR_ENGAGEMENT_LOAD_TEST_NOT_ASSEMBLY_OR_HARDWARE', 'configuration': {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  'socket_origin_world_m': socket_origin.tolist(), 'mass_properties': mass_before, 'physics_hz': 960, 'position_iterations': 128,
                  'external_lateral_axial_or_tilt_constraint': False,
                  'finite_rotary_test_actuator': bool(turn_joint), 'post_start_pose_writes': False, 'prescribed_force_cap_n': 10., 'prescribed_bending_torque_cap_nm': .2,
                  'nut_torque_cap_nm': .05, 'wall_seconds': time.monotonic()-started, 'images': image_rows,
                  'maximum_lateral_m': max(r['lateral_m'] for r in rows), 'maximum_tilt_deg': max(r['tilt_deg'] for r in rows)}
        (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    except Exception:
        import traceback
        failed = True; error = traceback.format_exc()
        (args.output/'error.txt').write_text(error); print(error, flush=True)
    finally:
        app.close(exit_code=1 if failed else 0)
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
