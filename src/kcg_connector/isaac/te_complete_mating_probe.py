#!/usr/bin/env python3
"""Bounded contact-driven connector bench test, with explicit removable guide.

The guide represents the external assembly tool, not part of the connector.
Initial insertion and optional first-thread capture use a declared finite axial
push. Contact data are evaluation-only.
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
    parser.add_argument('--release', action='store_true')
    parser.add_argument('--round-trip', action='store_true',
                        help='One uninterrupted mating, unloaded hold, reversal and withdrawal; preserves contact preload state')
    parser.add_argument('--release-start-only', action='store_true')
    parser.add_argument('--release-direct-torque', action='store_true',
                        help='Bounded diagnostic: apply a known nut torque instead of the rotary drive')
    parser.add_argument('--release-viscous-brake', action='store_true',
                        help='Use a passive joint damper with the known reverse torque')
    parser.add_argument('--retention', action='store_true')
    parser.add_argument('--retention-set', choices=('all', 'torsion', 'side', 'side_pair', 'axial_pair', 'mid_all'), default='all')
    parser.add_argument('--release-duration-s', type=float, default=4.5)
    parser.add_argument('--release-command-deg', type=float, default=360.)
    parser.add_argument('--encoder-reverse-stroke', action='store_true',
                        help='Stop reversal using the explicit tool shaft encoder, quantized to0.01degree')
    parser.add_argument('--probe-angle-deg', type=float,
                        help='Short slow torque probe, from an observed intermediate pose; positive tightens, negative loosens')
    parser.add_argument('--probe-speed-deg-s', type=float, default=10.,
                        help='Short-probe speed, bounded by the existing full-stroke speed')
    parser.add_argument('--probe-drive-damping', type=float, choices=(10., 100.), default=100.,
                        help='Short-probe external-tool damping in Nm s/rad; 10 reproduces the fast bench drive')
    parser.add_argument('--velocity-iterations', type=int, choices=(1, 4, 16), default=1)
    parser.add_argument('--solver', choices=('TGS', 'PGS'), default='TGS')
    parser.add_argument('--rotary-tool-inertia-from-hand', action='store_true',
                        help='Explicit external spindle with the existing palm axial inertia; no connector mass changes')
    parser.add_argument('--torque-transducer', action='store_true',
                        help='Measure transmitted torque from the external frictionless coupling contact impulses')
    parser.add_argument('--diagnostic-contact-breakdown', action='store_true',
                        help='Read-only: save nut contact contributions and native spindle velocities at torque samples')
    parser.add_argument('--external-wrench-audit', action='store_true',
                        help='Read-only: register every active socket collider and record tensor contact wrenches about the socket axis')
    parser.add_argument('--solver-impulse-observer', '--native-report-fix', dest='native_report_fix', type=Path,
                        help='Explicit version-pinned read-only solver-impulse diagnostic directory')
    parser.add_argument('--segmented-mating', action='store_true',
                        help='Same uninterrupted simulation: brake, remove the tool near40/180degrees, then reconnect and finish')
    parser.add_argument('--free-guide', action='store_true',
                        help='Capture diagnostic: only finite axial/rotary tool actuation; no lateral or tilt guide constraints')
    parser.add_argument('--start-separated', action='store_true',
                        help='Use finite initial axial support while starting in front of the socket mouth')
    parser.add_argument('--key-blocking-probe', action='store_true',
                        help='Finite approach and withdrawal from a declared mis-keyed pose; no rotary stroke')
    parser.add_argument('--encoder-forward-completion', action='store_true',
                        help='Finish the forward bench stroke at near-one-turn encoder stall; actual stop contact is reviewed separately')
    parser.add_argument('--spec-speed-profile', action='store_true',
                        help='Mechanical torque reference check: full coupling/uncoupling takes at least5seconds; decelerate before seating')
    parser.add_argument('--spindle-encoder-drive', action='store_true',
                        help='Drive only the external spindle with bounded torque from its quantized shaft encoder')
    parser.add_argument('--engagement-axial-assist', action='store_true',
                        help='Declared finite3.04N axial push only through the first40degrees of forward encoder rotation')
    parser.add_argument('--tool-axis', choices=('Z', 'X'), default='Z')
    parser.add_argument('--torque-cap-nm', type=float, default=1.6)
    parser.add_argument('--position-iterations', type=int, choices=(64, 128), default=128)
    parser.add_argument('--couple-duration-s', type=float, default=2.7)
    parser.add_argument('--wall-limit-s', type=float, default=155.)
    parser.add_argument('--contact-report-hz', type=int, choices=(5, 20), default=20)
    args = parser.parse_args()
    if args.release and args.retention:
        parser.error('Choose one independent verification mode')
    if args.round_trip and (args.release or args.retention or args.segmented_mating or not args.spec_speed_profile):
        parser.error('Continuous round trip requires the specification full-stroke forward mode')
    if args.segmented_mating and (args.release or args.retention or args.probe_angle_deg is not None or not args.rotary_tool_inertia_from_hand):
        parser.error('Segmented mating requires its own forward mode and the explicit encoder spindle')
    if args.free_guide and (not args.rotary_tool_inertia_from_hand or args.torque_transducer or args.release or args.retention):
        parser.error('Free guidance uses the explicit torsion-only spindle, without lateral coupling contacts')
    if args.start_separated and not args.free_guide:
        parser.error('The separated start requires free lateral/tilt guidance')
    if args.key_blocking_probe and (not args.start_separated or args.release or args.retention or args.segmented_mating or args.engagement_axial_assist):
        parser.error('Key blocking is an independent separated approach without a rotary stroke')
    if args.encoder_forward_completion and (not args.rotary_tool_inertia_from_hand or args.release or args.retention or args.probe_angle_deg is not None):
        parser.error('Forward encoder completion belongs to the ordinary full mating bench')
    if args.spec_speed_profile and (not args.rotary_tool_inertia_from_hand or args.retention or args.probe_angle_deg is not None):
        parser.error('The specification speed profile needs an ordinary full stroke and its encoder spindle')
    if args.spindle_encoder_drive and not args.rotary_tool_inertia_from_hand:
        parser.error('Encoder torque drive requires the declared external spindle')
    if args.engagement_axial_assist and (args.release or args.retention or not args.rotary_tool_inertia_from_hand):
        parser.error('Engagement assistance is confined to forward spindle tests')
    if args.release_start_only and not args.release:
        parser.error('Breakaway diagnostic requires release mode')
    if args.release_direct_torque and not args.release:
        parser.error('Known reverse torque requires release mode')
    if args.release_viscous_brake and not args.release_direct_torque:
        parser.error('The passive brake option accompanies the known torque input')
    if args.rotary_tool_inertia_from_hand and (args.tool_axis != 'X' or args.release_direct_torque):
        parser.error('The explicit spindle uses the X twist coordinate and the ordinary rotary drive')
    if args.torque_transducer and not args.rotary_tool_inertia_from_hand:
        parser.error('The torque transducer belongs to the explicit external spindle')
    if args.diagnostic_contact_breakdown and not args.torque_transducer:
        parser.error('Contact breakdown requires the torque transducer')
    if args.release_direct_torque and not (args.release_start_only or args.release_viscous_brake):
        parser.error('Direct torque is restricted to the short breakaway diagnostic')
    if not 2.7 <= args.release_duration_s <= 6.:
        parser.error('Bounded reverse stroke must be between2.7and6seconds')
    if args.encoder_reverse_stroke and not ((args.release or args.round_trip) and args.rotary_tool_inertia_from_hand):
        parser.error('Encoder stroke requires reversal with the explicit external spindle')
    if args.probe_angle_deg is not None and (not args.rotary_tool_inertia_from_hand or args.retention or args.release or not 0 < abs(args.probe_angle_deg) <= 10.):
        parser.error('A short probe requires its own mode, spindle, and a finite angle up to10degrees')
    if not 0. < args.probe_speed_deg_s <= 400./2.6:
        parser.error('Probe speed must be positive and no faster than the existing full-stroke bench')
    if not 0. < args.release_command_deg <= 400.:
        parser.error('Reverse command must be positive and at most400degrees')
    if not 0. < args.torque_cap_nm <= 4.6:
        parser.error('Finite bench torque must remain within the shell25 TableVI4.6Nm maximum')
    if not 2.7 <= args.couple_duration_s <= (7. if args.spec_speed_profile else 4.5):
        parser.error('Coupling stroke exceeds the declared finite duration')
    if not 155. <= args.wall_limit_s <= (520. if args.round_trip else 360. if args.spec_speed_profile else 240.):
        parser.error('Wall-clock limit exceeds the bounded diagnostic allowance')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'driver_snapshot.py').write_bytes(Path(__file__).read_bytes())
    started = time.monotonic()
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'multi_gpu': False, 'fast_shutdown': True,
                         'shutdown_watchdog_timeout': 10.})
    failed = False
    try:
        import carb
        import cv2
        import numpy as np
        import omni.usd
        import omni.timeline
        import omni.replicator.core as rep
        from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, PhysicsSchemaTools, UsdUtils
        from scipy.spatial.transform import Rotation
        from isaacsim.core.api import World
        from isaacsim.core.experimental.prims import RigidPrim
        from isaacsim.core.simulation_manager import SimulationManager
        from omni.physx import get_physx_simulation_interface
        from omni.physx.bindings._physx import SETTING_DISABLE_CONTACT_PROCESSING
        from te_foundationpose_handoff_runtime import _author_camera, _camera_cv_pose_from_eye_target
        from te_fast_contact_reading import read_shape_contact_pairs_fast
        report_fix = None
        if args.native_report_fix is not None:
            import importlib.util
            spec = importlib.util.spec_from_file_location('verified_native_report_fix', args.native_report_fix.resolve()/'verified_report_fix.py')
            report_fix = importlib.util.module_from_spec(spec); spec.loader.exec_module(report_fix)
            (args.output/'native_report_fix_installation.json').write_text(json.dumps(report_fix.install(), indent=2)+'\n')

        dt = 1/960.
        SimulationManager.set_physics_sim_device('cpu')
        carb.settings.get_settings().set_bool(SETTING_DISABLE_CONTACT_PROCESSING, False)
        world = World(stage_units_in_meters=1., physics_dt=dt, rendering_dt=dt,
                      backend='numpy', device='cpu', sim_params={'use_gpu_pipeline': False})
        stage = omni.usd.get_context().get_stage()
        UsdGeom.Xform.Define(stage, '/World')
        source = Usd.Stage.Open(str(args.model.resolve()))
        layer = source.Flatten()
        for p in source.GetPrimAtPath('/World').GetChildren():
            if p.IsA(UsdPhysics.Scene) or p.GetName() in ('HandArm', 'FixtureMaterial'):
                continue
            if not Sdf.CopySpec(layer, p.GetPath(), stage.GetRootLayer(), p.GetPath()):
                raise RuntimeError('Model copy failed')
        scene = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath(world.get_physics_context().prim_path))
        scene.CreateEnableGPUDynamicsAttr(False)
        scene.CreateBroadphaseTypeAttr('MBP')
        scene.CreateSolverTypeAttr(args.solver)
        # This flag is specific to temporal substeps; PGS has no such mode.
        # PGS is used only for explicitly requested diagnostic comparisons.
        scene.CreateEnableExternalForcesEveryIterationAttr(args.solver == 'TGS')
        scene.CreateMinVelocityIterationCountAttr(args.velocity_iterations)
        scene.CreateMaxVelocityIterationCountAttr(args.velocity_iterations)
        body = '/World/TE_J35FreeSplitPlug/Body'
        nut = '/World/TE_J35FreeSplitPlug/CouplingNut'
        socket = '/World/TEVisualHandoff/FixedReceptaclePose/OfficialVisual/Geometry'
        pose = json.loads(args.initial_pose.read_text())
        UsdGeom.Xformable(stage.GetPrimAtPath(body).GetParent()).ClearXformOpOrder()
        mass = {}
        for path, xyz, q in zip((body, nut), pose['positions_world_m'], pose['quaternions_wxyz']):
            p = stage.GetPrimAtPath(path)
            mass[path] = {k: str(p.GetAttribute(k).Get()) for k in
                          ('physics:mass', 'physics:centerOfMass', 'physics:diagonalInertia', 'physics:principalAxes')}
            x = UsdGeom.Xformable(p); x.ClearXformOpOrder()
            x.AddTranslateOp().Set(Gf.Vec3d(*xyz)); x.AddOrientOp().Set(Gf.Quatf(q[0], Gf.Vec3f(*q[1:])))
            index = 0 if path == body else 1
            velocity = pose.get('native_linear_velocity_m_s', [[0., 0., 0.]]*2)[index]
            angular_velocity = pose.get('native_angular_velocity_rad_s', [[0., 0., 0.]]*2)[index]
            UsdPhysics.RigidBodyAPI(p).CreateVelocityAttr(Gf.Vec3f(*velocity))
            # USD authors angular velocity in degrees/s; tensor readback is rad/s.
            UsdPhysics.RigidBodyAPI(p).CreateAngularVelocityAttr(Gf.Vec3f(*np.rad2deg(angular_velocity)))
            for api in (UsdPhysics.ArticulationRootAPI, PhysxSchema.PhysxArticulationAPI):
                if p.HasAPI(api): p.RemoveAPI(api)
            rb = PhysxSchema.PhysxRigidBodyAPI.Apply(p)
            rb.CreateSolverPositionIterationCountAttr(args.position_iterations); rb.CreateSolverVelocityIterationCountAttr(args.velocity_iterations)
            rb.CreateSleepThresholdAttr(0.)
            PhysxSchema.PhysxContactReportAPI.Apply(p).CreateThresholdAttr(0.)
        socket_origin = np.asarray(UsdGeom.Xformable(stage.GetPrimAtPath(socket)).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))[3, :3]
        if args.start_separated and not -.004 <= socket_origin[2]-pose['positions_world_m'][0][2] < 0.:
            raise RuntimeError('Separated-start pose must put the plug front between0and4mm before the mouth')
        guide = UsdPhysics.Joint.Define(stage, '/World/DeclaredRemovableAssemblyTool')
        guide.CreateBody1Rel().SetTargets([Sdf.Path(nut)]); guide.CreateExcludeFromArticulationAttr(True)
        guide.CreateLocalPos0Attr(Gf.Vec3f(*pose['positions_world_m'][1]))
        q = pose['quaternions_wxyz'][1]
        basis = Gf.Quatf(float(np.sqrt(.5)), Gf.Vec3f(0., -float(np.sqrt(.5)), 0.)) if args.tool_axis == 'X' else Gf.Quatf(1.)
        guide.CreateLocalRot0Attr(Gf.Quatf(q[0], Gf.Vec3f(*q[1:]))*basis)
        guide.CreateLocalPos1Attr(Gf.Vec3f(0.)); guide.CreateLocalRot1Attr(basis)
        for axis in ([] if args.free_guide else ('transY', 'transZ', 'rotY', 'rotZ') if args.tool_axis == 'X' else ('transX', 'transY', 'rotX', 'rotY')):
            lim = UsdPhysics.LimitAPI.Apply(guide.GetPrim(), axis); lim.CreateLowAttr(1.); lim.CreateHighAttr(-1.)
        approach_drives = []
        if args.start_separated:
            # A hand supports the plug before its shells have engaged. This
            # declared compliant tool support is removed before any turning.
            for axis in ('transY', 'transZ', 'rotY', 'rotZ'):
                ad = UsdPhysics.DriveAPI.Apply(guide.GetPrim(), axis)
                angular = axis.startswith('rot'); conversion = np.pi/180. if angular else 1.
                kp, kd, cap = (.5, .005, .2) if angular else (1000., 15., 3.0400615)
                ad.CreateTypeAttr('force');ad.CreateStiffnessAttr(kp*conversion);ad.CreateDampingAttr(kd*conversion)
                ad.CreateMaxForceAttr(cap);ad.CreateTargetPositionAttr(0.);ad.CreateTargetVelocityAttr(0.)
                approach_drives.append((ad,kp*conversion,kd*conversion,cap))
        drive = UsdPhysics.DriveAPI.Apply(guide.GetPrim(), 'rot'+args.tool_axis)
        drive.CreateTypeAttr('force'); drive.CreateStiffnessAttr(30.*np.pi/180.)
        drive.CreateDampingAttr(.1*np.pi/180.); drive.CreateMaxForceAttr(args.torque_cap_nm)
        drive.CreateTargetPositionAttr(0.); drive.CreateTargetVelocityAttr(0.)
        axial = UsdPhysics.DriveAPI.Apply(guide.GetPrim(), 'trans'+args.tool_axis)
        axial.CreateTypeAttr('force'); axial.CreateStiffnessAttr(0.); axial.CreateDampingAttr(0.)
        axial.CreateMaxForceAttr(0.); axial.CreateTargetPositionAttr(0.); axial.CreateTargetVelocityAttr(0.)
        if args.retention or args.release:
            # The external tool never becomes a native constraint in this
            # independent holding test, including reset/warm-up steps.
            guide.CreateJointEnabledAttr(False)
            drive.GetMaxForceAttr().Set(0.); drive.GetStiffnessAttr().Set(0.); drive.GetDampingAttr().Set(0.)
        tool_coupling = None; rotor_view = None; rotor_report = None; transducer_paths = []
        from te_rotary_contact_transducer import author as author_transducer, set_enabled as enable_transducer, read_torque as read_transducer
        if args.rotary_tool_inertia_from_hand:
            import xml.etree.ElementTree as ET
            palm_source = Path(__file__).resolve().parents[2]/'iiwa_description/urdf/hand.xacro'
            palm = ET.parse(palm_source).find("link[@name='handbase_link']/inertial")
            palm_mass = float(palm.find('mass').attrib['value'])
            palm_com = np.fromstring(palm.find('origin').attrib['xyz'], sep=' ')
            palm_izz = float(palm.find('inertia').attrib['izz'])
            shaft_inertia = palm_izz+palm_mass*float(palm_com[:2]@palm_com[:2])
            # This is an external apparatus inertia, shown as an equivalent
            # spindle. Its world bearing supports its weight. Only a torsional
            # coupling transmits effort to the nut; axial motion stays free.
            drive.GetMaxForceAttr().Set(0.); drive.GetStiffnessAttr().Set(0.); drive.GetDampingAttr().Set(0.)
            rotor = UsdGeom.Cylinder.Define(stage, '/World/DeclaredRotaryToolRotor')
            rotor.CreateRadiusAttr(float(np.sqrt(2.*shaft_inertia/palm_mass)))
            rotor.CreateHeightAttr(.008); rotor.CreateAxisAttr('Z')
            rotor.CreateDisplayColorAttr([Gf.Vec3f(.15, .4, .75)])
            rotor_position = np.asarray(pose['positions_world_m'][1])+[0., 0., .09]
            rotor.AddTranslateOp().Set(Gf.Vec3d(*rotor_position))
            rotor.AddOrientOp().Set(Gf.Quatf(q[0], Gf.Vec3f(*q[1:])))
            UsdPhysics.RigidBodyAPI.Apply(rotor.GetPrim())
            rotor_mass = UsdPhysics.MassAPI.Apply(rotor.GetPrim()); rotor_mass.CreateMassAttr(palm_mass)
            rotor_mass.CreateCenterOfMassAttr(Gf.Vec3f(0.)); rotor_mass.CreatePrincipalAxesAttr(Gf.Quatf(1.))
            rotor_mass.CreateDiagonalInertiaAttr(Gf.Vec3f(shaft_inertia/2., shaft_inertia/2., shaft_inertia))
            rotor_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(rotor.GetPrim())
            rotor_rb.CreateSolverPositionIterationCountAttr(args.position_iterations)
            rotor_rb.CreateSolverVelocityIterationCountAttr(args.velocity_iterations)
            rotor_rb.CreateSleepThresholdAttr(0.)
            shaft = UsdPhysics.RevoluteJoint.Define(stage, '/World/DeclaredRotaryToolBearing')
            shaft.CreateBody1Rel().SetTargets([rotor.GetPath()]); shaft.CreateAxisAttr('Z')
            shaft.CreateLocalPos0Attr(Gf.Vec3f(*rotor_position))
            shaft.CreateLocalRot0Attr(Gf.Quatf(q[0], Gf.Vec3f(*q[1:])))
            shaft.CreateExcludeFromArticulationAttr(True)
            drive = UsdPhysics.DriveAPI.Apply(shaft.GetPrim(), 'angular')
            drive.CreateTypeAttr('force'); drive.CreateStiffnessAttr(0.); drive.CreateDampingAttr(0.)
            drive.CreateMaxForceAttr(0.); drive.CreateTargetPositionAttr(0.); drive.CreateTargetVelocityAttr(0.)
            if args.torque_transducer:
                transducer_paths = author_transducer(stage, nut, str(rotor.GetPath()))
                PhysxSchema.PhysxContactReportAPI.Apply(rotor.GetPrim()).CreateThresholdAttr(0.)
            else:
                tool_coupling = UsdPhysics.Joint.Define(stage, '/World/DeclaredRotaryToolTorsionalCoupling')
                tool_coupling.CreateBody0Rel().SetTargets([rotor.GetPath()])
                tool_coupling.CreateBody1Rel().SetTargets([Sdf.Path(nut)])
                tool_coupling.CreateLocalRot0Attr(basis); tool_coupling.CreateLocalRot1Attr(basis)
                tool_coupling.CreateExcludeFromArticulationAttr(True)
                torsion = UsdPhysics.LimitAPI.Apply(tool_coupling.GetPrim(), 'rotX')
                torsion.CreateLowAttr(1.); torsion.CreateHighAttr(-1.)
                tool_coupling.CreateJointEnabledAttr(not (args.release or args.retention))
            rotor_view = RigidPrim([str(rotor.GetPath())], resolve_paths=False)
            rotor_report = {'scope': 'EXPLICIT_EXTERNAL_TEST_SPINDLE_NOT_A_CONNECTOR_PART_OR_HAND_GRASP',
                            'palm_source': str(palm_source), 'palm_mass_kg': palm_mass,
                            'palm_center_inertia_zz_kg_m2': palm_izz, 'palm_com_m': palm_com.tolist(),
                            'spindle_axis_inertia_kg_m2': shaft_inertia,
                            'connection': 'torsion only; all translations free; disconnected during free hold',
                            'connector_mass_inertia_unchanged': True}
            rotor_report['torque_transducer'] = {'enabled': args.torque_transducer,
                'method': 'NATIVE_FRICTIONLESS_CONTACT_IMPULSE', 'readout_hz': 240,
                'paths': transducer_paths, 'axial_stroke_free': True,
                'scope': 'Measured normal-contact moment; no spring-deflection conversion'}
            (args.output/'external_spindle.json').write_text(json.dumps(rotor_report, indent=2)+'\n')
        socket_filters = [socket]
        if args.external_wrench_audit:
            socket_filters = [str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath('/World/TEVisualHandoff/FixedReceptaclePose'))
                              if p.HasAPI(UsdPhysics.CollisionAPI)
                              and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()]
        view = RigidPrim([body, nut], resolve_paths=False, contact_filter_paths=socket_filters, max_contact_count=32768)
        camera = '/World/CompleteMatingEvidenceCamera'
        _author_camera(stage, camera, _camera_cv_pose_from_eye_target(socket_origin+[.11, -.13, .085], socket_origin+[0, 0, .006]),
                       resolution=(800, 600), focal_length_mm=32., horizontal_aperture_mm=36., clipping_range_m=(.01, 5.), Gf=Gf, UsdGeom=UsdGeom)
        UsdLux.DomeLight.Define(stage, '/World/CompleteMatingEvidenceLight').CreateIntensityAttr(1000.)
        product = rep.create.render_product(camera, (800, 600))
        rgb = rep.AnnotatorRegistry.get_annotator('rgb'); rgb.attach([product.path])
        stage.Flatten().Export(str(args.output/'before_physics.usdc'))
        world.reset()
        rounded_pins = [p for p in Usd.PrimRange(stage.GetPrimAtPath(body)) if p.GetName().startswith('OfficialRoundedPin_')]
        quarter_pins = [p for p in Usd.PrimRange(stage.GetPrimAtPath(body))
                        if p.GetName().startswith(('OfficialPinNoseQuarter_', 'OfficialPinWholeQuarter_'))
                        and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()]
        expected_hulls = 512 if quarter_pins else 128
        rounded_pins = quarter_pins or rounded_pins
        if rounded_pins:
            from scipy.spatial import ConvexHull
            from omni.physx import get_physx_cooking_interface
            authored = np.asarray(UsdGeom.Mesh(rounded_pins[0]).GetPointsAttr().Get(), float)
            if len(rounded_pins) != expected_hulls or any(not np.array_equal(authored, np.asarray(UsdGeom.Mesh(p).GetPointsAttr().Get())) for p in rounded_pins):
                raise RuntimeError('Official pin geometry family differs')
            received = {}
            def cooked(status, convexes):
                received['status'] = str(status)
                received['vertices'] = [np.asarray([list(v) for v in c.vertices], float) for c in convexes]
            get_physx_cooking_interface().request_convex_collision_representation(
                UsdUtils.StageCache.Get().Insert(stage).ToLongInt(), PhysicsSchemaTools.sdfPathToInt(rounded_pins[0].GetPath()), False, cooked)
            if len(received.get('vertices', [])) != 1: raise RuntimeError('Pin did not cook to one convex')
            actual = received['vertices'][0]
            from scipy.spatial import cKDTree
            missing_vertex_mm = float(cKDTree(actual).query(authored)[0].max())
            planes = ConvexHull(authored).equations
            excess_mm = float((actual@planes[:, :3].T+planes[:, 3]).max())
            bounds_delta_mm = float(np.max(np.abs(np.array([actual.min(0), actual.max(0)])-np.array([authored.min(0), authored.max(0)]))))
            if excess_mm > .0001 or bounds_delta_mm > .0001 or missing_vertex_mm > .0001: raise RuntimeError('Native pin hull was inflated, eroded or changed')
            (args.output/'native_pin_geometry.json').write_text(json.dumps({'pin_count': 128, 'identical_local_shapes': True, 'cooked_vertices': len(actual),
                'hull_count': expected_hulls, 'outside_authored_hull_mm': excess_mm, 'bounds_change_mm': bounds_delta_mm,
                'max_authored_vertex_to_native_vertex_mm': missing_vertex_mm, 'scale_to_m': .001}, indent=2)+'\n')
        carb.logging.acquire_logging().set_level_threshold_for_source('omni.physx.plugin', carb.logging.LogSettingBehavior.OVERRIDE, carb.logging.LEVEL_ERROR)
        interface = get_physx_simulation_interface()
        from omni.physx.bindings._physx import acquire_physx_statistics_interface, PhysicsSceneStats
        statistics = acquire_physx_statistics_interface(); statistics.enable_carb_stats_upload(True)
        statistics_stage = UsdUtils.StageCache.Get().Insert(stage).ToLongInt()
        statistics_scene = PhysicsSchemaTools.sdfPathToInt(stage.GetPrimAtPath(world.get_physics_context().prim_path).GetPath())
        def native_statistics():
            value = PhysicsSceneStats()
            for sid in (statistics_stage, 0):
                if statistics.get_physx_scene_statistics(sid, statistics_scene, value):
                    return {'valid': True, 'active_constraints': value.nb_active_constraints,
                            'dynamic_bodies': value.nb_active_dynamic_rigids, 'articulations': value.nb_articulations}
            return {'valid': False}
        def host(x): return x.numpy() if hasattr(x, 'numpy') else np.asarray(x)
        def state(): return np.concatenate([host(x).ravel() for x in (*view.get_world_poses(), *view.get_velocities())])
        images = []
        video = cv2.VideoWriter(str(args.output/'mating_test.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 4., (800, 600))
        def capture(phase):
            before = state().copy(); now = float(world.current_time)
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
            if pixels.ndim != 3 or not np.array_equal(before, state()) or now != float(world.current_time):
                raise RuntimeError('Evidence rendering changed physical state')
            pixels = cv2.cvtColor(pixels[:, :, :3], cv2.COLOR_RGB2BGR)
            cv2.putText(pixels, f'{phase} | t={now:.3f}s', (15, 25), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
            video.write(pixels)
            path = args.output/f'{len(images):03d}_{phase}.png'; cv2.imwrite(str(path), pixels)
            images.append({'phase': phase, 'time_s': now, 'path': str(path.resolve()), 'state_delta': 0.})
        class Snapshot:
            data = None
            def get_full_contact_report(self): return self.data
        snapshot = Snapshot(); decoded = {}
        def decode(value):
            k = int(value)
            if k not in decoded: decoded[k] = str(PhysicsSchemaTools.intToSdfPath(k))
            return decoded[k]
        sequence = ([('release_free', .15), ('unscrew', args.release_duration_s), ('withdraw', .5), ('withdraw_hold', .15)] if args.release else
                    [('settle', .1), ('insert', .5), ('entry_hold', .1), ('couple', args.couple_duration_s), ('loaded_hold', .15), ('free_hold', .3)])
        if args.retention:
            sequence = [('retention_settle', .15), ('side_load', .66), ('side_unload', .12),
                        ('axial_pull', .66), ('axial_unload', .12), ('body_reverse_torque', .66),
                        ('twist_unload', .12), ('retention_free', .3)]
            if args.retention_set == 'torsion':
                sequence = [('retention_settle', .15), ('body_reverse_torque', .66),
                            ('twist_unload', .12), ('retention_free', .3)]
            elif args.retention_set == 'side':
                sequence = [('retention_settle', .15), ('side_load', .66),
                            ('side_unload', .12), ('retention_free', .3)]
            elif args.retention_set == 'side_pair':
                sequence = [('retention_settle', .15), ('side_load', .66), ('side_unload', .12),
                            ('side_pull', .66), ('side_unload', .12), ('retention_free', .15)]
            elif args.retention_set == 'axial_pair':
                sequence = [('retention_settle', .15), ('axial_push', .66), ('axial_unload', .12),
                            ('axial_pull', .66), ('axial_unload', .12), ('retention_free', .15)]
            elif args.retention_set == 'mid_all':
                sequence = [('retention_settle', .15), ('side_load', .66), ('side_unload', .12),
                            ('side_pull', .66), ('side_unload', .12), ('axial_push', .66), ('axial_unload', .12),
                            ('axial_pull', .66), ('axial_unload', .12), ('body_reverse_torque', .66),
                            ('twist_unload', .12), ('retention_free', .15)]
        if args.release_start_only:
            sequence = [('release_free', .15), ('unscrew', .6), ('loaded_hold', .1), ('free_hold', .15)]
        if args.probe_angle_deg is not None:
            sequence = [('settle', .15), ('couple' if args.probe_angle_deg > 0 else 'unscrew', 1.2),
                        ('loaded_hold', .15), ('free_hold', .15)]
        if args.segmented_mating:
            sequence = [('settle', .1), ('insert', .5), ('entry_hold', .1),
                        ('couple_to40', .7), ('brake40', .1), ('pause40', .15),
                        ('couple_to180', 1.5), ('brake180', .1), ('pause180', .15),
                        ('couple_to_stop', 2.), ('loaded_hold', .1), ('free_hold', .3)]
        if args.key_blocking_probe:
            sequence = [('settle', .1), ('insert', .5), ('entry_hold', .3),
                        ('withdraw', .5), ('withdraw_hold', .15)]
        if args.round_trip:
            sequence += [('release_free', .15), ('unscrew', args.release_duration_s),
                         ('withdraw', .5), ('withdraw_hold', .15)]
        initial_rotation = Rotation.from_quat(np.asarray(pose['quaternions_wxyz'][1])[[1, 2, 3, 0]]).as_matrix()
        keyed_body_rotation = Rotation.from_euler('y', 180., degrees=True)
        rows = []; wrapped_previous = 0.; angle = 0.; step = 0; previous_rotation = None
        tool_encoder_angle = 0.; tool_encoder_previous = 0.
        encoder_previous_measurement = 0.; encoder_velocity_rad_s = 0.; spindle_reference_rad = 0.
        timings = {'physics_s': 0., 'readback_s': 0., 'evidence_s': 0.}
        capture('initial')
        with (args.output/'samples.jsonl').open('x', buffering=1) as stream:
            for phase, duration in sequence:
                if args.round_trip and phase == 'unscrew':
                    # Reset only the external encoder's displayed origin;
                    # retain all physical poses, velocities and contact state.
                    tool_encoder_angle = 0.; tool_encoder_previous = tool_wrapped
                    encoder_previous_measurement = 0.; encoder_velocity_rad_s = 0.
                operation = 'couple' if phase.startswith('couple_to') else phase
                enabled = phase not in ('free_hold', 'release_free', 'pause40', 'pause180') and not args.retention
                velocity_stroke = operation in ('couple', 'unscrew')
                velocity_brake = phase in ('loaded_hold', 'withdraw', 'withdraw_hold', 'brake40', 'brake180')
                guide.CreateJointEnabledAttr(enabled)
                if tool_coupling: tool_coupling.CreateJointEnabledAttr(enabled)
                if transducer_paths: enable_transducer(stage, transducer_paths, enabled)
                supporting_approach = args.start_separated and (
                    phase in ('settle', 'insert', 'entry_hold') or
                    (args.key_blocking_probe and phase in ('withdraw', 'withdraw_hold')))
                for ad,kp,kd,cap in approach_drives:
                    ad.GetStiffnessAttr().Set(kp if supporting_approach else 0.)
                    ad.GetDampingAttr().Set(kd if supporting_approach else 0.)
                    ad.GetMaxForceAttr().Set(cap if supporting_approach else 0.)
                drive.GetMaxForceAttr().Set(args.torque_cap_nm if enabled else 0.)
                drive.GetStiffnessAttr().Set(30.*np.pi/180. if enabled and not (velocity_stroke or velocity_brake) else 0.)
                drive.GetDampingAttr().Set((10. if velocity_stroke or velocity_brake else .1)*np.pi/180. if enabled else 0.)
                if args.probe_angle_deg is not None and velocity_stroke:
                    drive.GetDampingAttr().Set(args.probe_drive_damping*np.pi/180.)
                if args.spec_speed_profile and velocity_stroke:
                    drive.GetDampingAttr().Set(100.*np.pi/180.)
                if args.spindle_encoder_drive:
                    drive.GetMaxForceAttr().Set(0.);drive.GetStiffnessAttr().Set(0.);drive.GetDampingAttr().Set(0.)
                    spindle_reference_rad = np.deg2rad(round(tool_encoder_angle,2))
                drive.GetTargetVelocityAttr().Set(0.)
                if args.release_direct_torque and phase == 'unscrew':
                    reverse_rate_rad_s = np.deg2rad(args.release_command_deg/(args.release_duration_s-.1))
                    brake_nm_s_rad = args.torque_cap_nm/reverse_rate_rad_s
                    drive.GetMaxForceAttr().Set(args.torque_cap_nm if args.release_viscous_brake else 0.)
                    drive.GetDampingAttr().Set(brake_nm_s_rad*np.pi/180. if args.release_viscous_brake else 0.)
                inserting = phase in ('insert', 'entry_hold', 'withdraw', 'withdraw_hold') or (args.start_separated and phase == 'settle')
                axial.GetMaxForceAttr().Set(3.0400615 if inserting else 0.)
                axial.GetStiffnessAttr().Set(10000. if inserting else 0.)
                axial.GetDampingAttr().Set(20. if inserting else 0.)
                interface.flush_changes()
                n = round(duration/dt)
                encoder_history = []
                for i in range(n):
                    encoder_target = {'couple_to40': 40., 'couple_to180': 180.}.get(phase)
                    if encoder_target is not None and round(tool_encoder_angle, 2) >= encoder_target:
                        drive.GetTargetVelocityAttr().Set(0.)
                        break
                    if phase == 'couple_to_stop' or (args.encoder_forward_completion and operation == 'couple'):
                        encoder_history.append(round(tool_encoder_angle, 2))
                        window = round(.12/dt)
                        if (tool_encoder_angle >= 360. or
                            (tool_encoder_angle > 350. and len(encoder_history) > window
                             and max(encoder_history[-window:])-min(encoder_history[-window:]) <= .02)):
                            drive.GetTargetVelocityAttr().Set(0.)
                            break
                    if args.probe_angle_deg is not None and velocity_stroke and np.sign(args.probe_angle_deg)*round(tool_encoder_angle, 2) >= abs(args.probe_angle_deg):
                        drive.GetTargetVelocityAttr().Set(0.)
                        break
                    if args.encoder_reverse_stroke and phase == 'unscrew' and round(tool_encoder_angle, 2) <= -args.release_command_deg:
                        drive.GetTargetVelocityAttr().Set(0.)
                        break
                    u = (i+1)/n; blend = 10*u**3-15*u**4+6*u**5
                    if phase == 'insert': axial.GetTargetPositionAttr().Set(.012*blend)
                    if phase == 'withdraw': axial.GetTargetPositionAttr().Set(-.018*blend)
                    if operation == 'couple':
                        # Preserve the original input prefix through2.6s,
                        # then continue the same finite velocity if requested.
                        elapsed = (i+1)*dt; original_u = min(elapsed/2.7, 1.)
                        original_blend = 10*original_u**3-15*original_u**4+6*original_u**5
                        drive.GetTargetPositionAttr().Set(400.*original_blend+max(0., elapsed-2.7)*400./2.6)
                    if phase == 'unscrew': drive.GetTargetPositionAttr().Set(-args.release_command_deg*(1. if args.encoder_reverse_stroke else blend))
                    if velocity_stroke:
                        edge = min(1., (i+1)*dt/.1, max(0., (duration-(i+1)*dt)/.1))
                        if args.encoder_reverse_stroke and phase == 'unscrew':
                            edge = min(1., (i+1)*dt/.1)
                        if args.probe_angle_deg is not None: edge = min(1., (i+1)*dt/.1)
                        if args.segmented_mating: edge = min(1., (i+1)*dt/.1)
                        smooth = edge*edge*(3.-2.*edge)
                        rate = 400./2.6 if operation == 'couple' else args.release_command_deg/(args.release_duration_s-.1)
                        if args.encoder_reverse_stroke and phase == 'unscrew': rate = 90.
                        if args.probe_angle_deg is not None: rate = args.probe_speed_deg_s
                        if args.spec_speed_profile:
                            rate = (17. if tool_encoder_angle < 40. else 120. if tool_encoder_angle < 350. else 20.) if operation == 'couple' else (90. if -tool_encoder_angle < 310. else 30.)
                        drive.GetTargetVelocityAttr().Set((1. if operation == 'couple' else -1.)*rate*smooth)
                        if args.release_direct_torque and phase == 'unscrew':
                            drive.GetTargetVelocityAttr().Set(0.)
                    forces = np.zeros((2, 3)); torques = np.zeros((2, 3))
                    if args.engagement_axial_assist and operation == 'couple' and tool_encoder_angle < 40.:
                        forces[1] = initial_rotation[:,2]*3.0400615*min(1.,(i+1)*dt/.05)
                        view.apply_forces_and_torques_at_pos(forces=forces)
                    spindle_torque_command = 0.
                    if args.spindle_encoder_drive:
                        measured_encoder = round(tool_encoder_angle,2)
                        measured_velocity = np.deg2rad(measured_encoder-encoder_previous_measurement)/dt
                        encoder_previous_measurement = measured_encoder
                        # Reverse breakaway suddenly unloads the thrust bearing.
                        # For I=0.00318kgm2, Ki=30Nm/rad, critical velocity
                        # feedback is0.62Nms/rad. Use overdamped2Nms/rad on
                        # reversal, with a0.5ms filter to avoid damping delay.
                        encoder_filter_s = .0005 if operation == 'unscrew' else .002
                        encoder_velocity_gain = 2. if operation == 'unscrew' else .62
                        alpha = 1.-np.exp(-dt/encoder_filter_s)
                        encoder_velocity_rad_s += alpha*(measured_velocity-encoder_velocity_rad_s)
                        reference_velocity = np.deg2rad(float(drive.GetTargetVelocityAttr().Get())) if enabled else 0.
                        if enabled:
                            spindle_reference_rad += reference_velocity*dt
                            measured_rad = np.deg2rad(measured_encoder)
                            spindle_reference_rad = float(np.clip(spindle_reference_rad,measured_rad-args.torque_cap_nm/30.,measured_rad+args.torque_cap_nm/30.))
                            spindle_torque_command = float(np.clip(30.*(spindle_reference_rad-measured_rad)+encoder_velocity_gain*(reference_velocity-encoder_velocity_rad_s),-args.torque_cap_nm,args.torque_cap_nm))
                        rotor_view.apply_forces_and_torques_at_pos(torques=np.asarray([initial_rotation[:,2]*spindle_torque_command]))
                    if args.release_direct_torque and phase == 'unscrew':
                        # World torque opposite the observed nut shaft. Geometry,
                        # joint limits and contact parameters remain unchanged.
                        torques[1] = -initial_rotation[:, 2]*args.torque_cap_nm*smooth
                        view.apply_forces_and_torques_at_pos(forces=forces, torques=torques)
                    if args.retention:
                        elapsed = (i+1)*dt
                        amplitude = min(1., elapsed/.03, max(0., (duration-elapsed)/.03))
                        if phase == 'side_load': forces[0, 0] = 10.*amplitude; torques[0, 1] = .2*amplitude
                        if phase == 'side_pull': forces[0, 0] = -10.*amplitude; torques[0, 1] = -.2*amplitude
                        if phase == 'axial_push': forces[0, 2] = -10.*amplitude
                        if phase == 'axial_pull': forces[0, 2] = 10.*amplitude
                        if phase == 'body_reverse_torque': torques[0, 2] = .2*amplitude
                        view.apply_forces_and_torques_at_pos(forces=forces, torques=torques)
                    applied_capture = report_fix is not None and args.external_wrench_audit and step % round(1./args.contact_report_hz/dt) == 0
                    if report_fix is not None:
                        read_contacts = ((args.torque_transducer or args.external_wrench_audit) and step % 4 == 0) or step % round(1./args.contact_report_hz/dt) == 0 or i == n-1
                        report_fix.begin_applied_capture(applied_capture, correct=read_contacts)
                    tick = time.monotonic(); world.step(render=False); timings['physics_s'] += time.monotonic()-tick; tick = time.monotonic()
                    pos, quat = (host(x).copy() for x in view.get_world_poses())
                    linear, angular = (host(x).copy() for x in view.get_velocities())
                    rotation = Rotation.from_quat(quat[:, [1, 2, 3, 0]])
                    pose_omega = None if previous_rotation is None else (rotation*previous_rotation.inv()).as_rotvec()/dt
                    previous_rotation = rotation
                    body_key_rotation = (rotation[0]*keyed_body_rotation.inv()).as_matrix()
                    body_yaw = float(np.degrees(np.arctan2(body_key_rotation[1, 0], body_key_rotation[0, 0])))
                    rel = initial_rotation.T@rotation[1].as_matrix()
                    wrapped = np.degrees(np.arctan2(rel[1, 0], rel[0, 0]))
                    angle += (wrapped-wrapped_previous+180.) % 360.-180.; wrapped_previous = wrapped
                    row = {'step': step, 'time_s': float(world.current_time), 'phase': phase,
                           'positions_world_m': pos.tolist(), 'quaternions_wxyz': quat.tolist(),
                           'native_linear_velocity_m_s': linear.tolist(), 'native_angular_velocity_rad_s': angular.tolist(),
                           'pose_increment_angular_velocity_rad_s': None if pose_omega is None else pose_omega.tolist(),
                           'applied_forces_world_n': forces.tolist(), 'applied_torques_world_nm': torques.tolist(),
                           'body_key_yaw_deg': body_yaw,
                           'body_centering_speed_over_5_rad_s': bool(pose_omega is not None and np.linalg.norm(pose_omega[0]) > 5.),
                           'nut_angle_deg': angle, 'body_depth_m': float(socket_origin[2]-pos[0, 2]),
                           'nut_depth_m': float(socket_origin[2]-pos[1, 2]),
                           'nut_axial_offset_in_body_m': float((rotation[0].as_matrix().T@(pos[1]-pos[0]))[2]),
                           'lateral_m': float(np.linalg.norm(pos[0, :2]-socket_origin[:2])),
                           'tilt_deg': float(np.degrees(np.arccos(np.clip(-rotation[0].as_matrix()[2, 2], -1, 1)))),
                           'external_tool_enabled': enabled, 'axial_drive_cap_n': float(axial.GetMaxForceAttr().Get()),
                           'rotary_drive_cap_nm': float(drive.GetMaxForceAttr().Get()), 'rotary_command_deg': float(drive.GetTargetPositionAttr().Get())}
                    row['rotary_target_velocity_deg_s'] = float(drive.GetTargetVelocityAttr().Get())
                    if args.spindle_encoder_drive:
                        row['spindle_encoder_torque_command_nm'] = spindle_torque_command
                        row['spindle_encoder_velocity_rad_s'] = encoder_velocity_rad_s
                        row['rotary_drive_cap_nm'] = args.torque_cap_nm if enabled else 0.
                    row['finite_lateral_tilt_approach_support_enabled'] = bool(supporting_approach)
                    if applied_capture:
                        row['solver_applied_impulse_records'] = report_fix.applied_capture()
                        if hasattr(report_fix, 'patch_capture'):
                            row['solver_patch_impulse_records'] = report_fix.patch_capture()
                    if args.external_wrench_audit and step % 4 == 0:
                        fn, cp, cn, _, cc, cs = (host(x) for x in view.get_contact_force_data())
                        ff, fp, fc, fs = (host(x) for x in view.get_friction_data())
                        cc = cc.reshape(2, -1); cs = cs.reshape(2, -1)
                        fc = fc.reshape(2, -1); fs = fs.reshape(2, -1)
                        fn = fn.ravel()
                        wrench_rows = []
                        for actor_index in range(2):
                            normal_wrench = np.zeros(6); friction_wrench = np.zeros(6)
                            for counts, starts, forces_data, points_data, target in (
                                (cc, cs, fn[:, None]*cn/dt, cp, normal_wrench),
                                (fc, fs, ff/dt, fp, friction_wrench)):
                                for count, start in zip(counts[actor_index], starts[actor_index]):
                                    if not count: continue
                                    section = slice(int(start), int(start+count))
                                    force = forces_data[section]
                                    target[:3] += force.sum(axis=0)
                                    target[3:] += np.cross(points_data[section]-socket_origin, force).sum(axis=0)
                            wrench_rows.append({'normal_world_n_nm': normal_wrench.tolist(),
                                                'friction_world_n_nm': friction_wrench.tolist(),
                                                'normal_point_count': int(cc[actor_index].sum()),
                                                'friction_anchor_count': int(fc[actor_index].sum())})
                        row['tensor_external_socket_wrenches'] = wrench_rows
                    if source.GetPrimAtPath('/World/TE_J35FreeSplitPlug/Joints/CouplingNutRevolute').GetAttribute('kcg:representativeThrustContact').Get():
                        margin = float(source.GetPrimAtPath('/World/TE_J35FreeSplitPlug/Joints/CouplingNutRevolute').GetAttribute('kcg:thrustBackupMarginM').Get())
                        row['thrust_backup_limit_approached'] = abs(row['nut_axial_offset_in_body_m']) >= .0005+margin-.0000001
                    if rotor_view is not None:
                        rp, rq = (host(x).copy() for x in rotor_view.get_world_poses())
                        row['external_spindle'] = {'position_world_m': rp[0].tolist(), 'quaternion_wxyz': rq[0].tolist(),
                                                   'torsional_coupling_enabled': enabled}
                        if args.diagnostic_contact_breakdown or args.spindle_encoder_drive:
                            rv, rw = (host(x).copy() for x in rotor_view.get_velocities())
                            row['external_spindle']['native_linear_velocity_m_s'] = rv[0].tolist()
                            row['external_spindle']['native_angular_velocity_rad_s'] = rw[0].tolist()
                        tool_rotation = Rotation.from_quat(rq[0, [1, 2, 3, 0]]).as_matrix()
                        tool_relative = initial_rotation.T@tool_rotation
                        tool_wrapped = float(np.rad2deg(np.arctan2(tool_relative[1, 0], tool_relative[0, 0])))
                        tool_encoder_angle += (tool_wrapped-tool_encoder_previous+180.)%360.-180.
                        tool_encoder_previous = tool_wrapped
                        row['external_spindle']['shaft_encoder_deg'] = round(tool_encoder_angle, 2)
                    row['tool_torque_sample_fresh'] = False
                    if args.torque_transducer and step % 4 == 0:
                        snapshot.data = interface.get_full_contact_report()
                        sensor_contacts = read_shape_contact_pairs_fast(snapshot, dt, nut, pos[1], decode_path=decode)
                        sensor_torque, sensor_count = read_transducer(sensor_contacts, rotation[1].as_matrix()[:, 2])
                        row['tool_transmitted_torque_nm'] = sensor_torque if enabled else 0.
                        row['tool_torque_contact_count'] = sensor_count
                        row['tool_torque_sample_fresh'] = True
                        row['measured_torque_over_4p6_nm'] = enabled and abs(sensor_torque) > 4.6
                        if args.diagnostic_contact_breakdown:
                            axis_world = rotation[1].as_matrix()[:, 2]
                            contributions = {}
                            for pair in sensor_contacts:
                                own = pair['own_collider']; other = pair['other_collider']
                                group = ('tool' if 'ExternalTorque' in own else
                                         'thrust' if 'Thrust' in own else
                                         'ratchet' if 'Ratchet' in own else
                                         'thread_socket' if 'FixedReceptacle' in other else own.rsplit('/', 1)[-1])
                                item = contributions.setdefault(group, {'normal_torque_nm': 0., 'friction_torque_nm': 0.,
                                                                        'normal_load_n': 0., 'maximum_penetration_m': 0.})
                                item['normal_torque_nm'] += float(np.asarray(pair['normal_wrench_n_nm'][3:])@axis_world)
                                item['friction_torque_nm'] += float(np.asarray(pair['friction_wrench_n_nm'][3:])@axis_world)
                                item['normal_load_n'] += pair['normal_load_n']
                                item['maximum_penetration_m'] = max(item['maximum_penetration_m'], pair['maximum_penetration_m'] or 0.)
                            row['diagnostic_nut_contacts'] = contributions
                            if abs(sensor_torque) >= 6.:
                                row['diagnostic_peak_nut_pairs'] = sensor_contacts
                            points = []
                            headers, contact_data, _ = snapshot.data
                            for header in headers:
                                actors = (decode(header.actor0), decode(header.actor1))
                                if nut not in actors: continue
                                own_index = actors.index(nut)
                                colliders = (decode(header.collider0), decode(header.collider1))
                                if 'FixedReceptacle' not in colliders[1-own_index]: continue
                                sign = 1. if own_index == 0 else -1.
                                for ci in range(int(header.num_contact_data)):
                                    record = contact_data[int(header.contact_data_offset)+ci]
                                    impulse = sign*np.asarray(record.impulse, dtype=float)
                                    if np.linalg.norm(impulse) < dt: continue
                                    points.append({'collider': colliders[own_index],
                                                   'position_world_m': list(record.position),
                                                   'impulse_world_ns': impulse.tolist(),
                                                   'contact_normal_world': (sign*np.asarray(record.normal, dtype=float)).tolist(),
                                                   'separation_m': float(record.separation)})
                            row['diagnostic_thread_contact_points'] = points
                    if step % round(1./args.contact_report_hz/dt) == 0 or i == n-1:
                        snapshot.data = interface.get_full_contact_report()
                        row['shape_contacts'] = [read_shape_contact_pairs_fast(snapshot, dt, path, pos[j], decode_path=decode) for j, path in enumerate((body, nut))]
                        row['native_scene_statistics'] = native_statistics()
                    stream.write(json.dumps(row, separators=(',', ':'))+'\n'); rows.append(row)
                    timings['readback_s'] += time.monotonic()-tick
                    # The driven nut retains the5rad/s motion limit. Body
                    # centering inside key clearance is recorded separately.
                    # Keep the legacy .60deg finite excursion stop unchanged.
                    # It is not a source tolerance or a penetration test: the
                    # socket uses triangles, so no socket-SDF allowance applies.
                    # Source-key/slot clearance is evaluated from recorded poses.
                    engaged_key_excursion = row['body_depth_m'] > .004 and abs(body_yaw) > .60
                    if not np.isfinite(pos).all() or (pose_omega is not None and np.linalg.norm(pose_omega[1]) > 5.) or engaged_key_excursion or row['lateral_m'] > .002 or row['tilt_deg'] > 5.:
                        capture('independent_state_stop'); raise RuntimeError('Independent finite state/speed/displacement stop')
                    if time.monotonic()-started > args.wall_limit_s: raise RuntimeError('Declared internal wall-clock limit')
                    if step % 240 == 0:
                        if report_fix is not None:
                            (args.output/'native_report_fix.json').write_text(json.dumps(report_fix.status(), indent=2)+'\n')
                            report_fix.check()
                        tick = time.monotonic(); capture(phase); timings['evidence_s'] += time.monotonic()-tick
                        (args.output/'progress.json').write_text(json.dumps({k: v for k, v in row.items() if k not in ('shape_contacts', 'solver_patch_impulse_records', 'solver_applied_impulse_records')}, indent=2)+'\n')
                    step += 1
                capture(phase+'_end')
                if phase in ('couple_to40', 'couple_to180') and round(tool_encoder_angle, 2) < {'couple_to40': 40., 'couple_to180': 180.}[phase]:
                    raise RuntimeError('Segmented encoder target not reached within the finite time/torque budget')
                if args.encoder_reverse_stroke and phase == 'unscrew' and round(tool_encoder_angle, 2) > -args.release_command_deg:
                    raise RuntimeError('Bounded tool encoder stroke did not finish; withdrawal not attempted')
                print(json.dumps({k: row[k] for k in ('phase', 'body_depth_m', 'nut_depth_m', 'nut_angle_deg', 'nut_axial_offset_in_body_m')}), flush=True)
        final = {k: rows[-1][k] for k in ('positions_world_m', 'quaternions_wxyz', 'native_linear_velocity_m_s', 'native_angular_velocity_rad_s')}
        final['scope'] = 'Observed end state, no state was overwritten during this run'
        (args.output/'final_pose.json').write_text(json.dumps(final, indent=2)+'\n')
        result = {'scope': 'CONTACT_DRIVEN_CONNECTOR_BENCH_TEST_REQUIRES_PHYSICAL_REVIEW',
                  'mode': 'torque_probe' if args.probe_angle_deg is not None else 'retention' if args.retention else 'release' if args.release else 'round_trip' if args.round_trip else 'key_blocking' if args.key_blocking_probe else 'mating',
                  'model': str(args.model.resolve()), 'configuration': {'rotation_cap_nm': args.torque_cap_nm, 'axial_insertion_cap_n': 3.0400615, 'physics_hz': 960, 'solver': args.solver, 'position_iterations': args.position_iterations, 'velocity_iterations': args.velocity_iterations, 'tool_joint_axis': args.tool_axis, 'couple_duration_s': args.couple_duration_s, 'release_duration_s': args.release_duration_s, 'release_command_deg': args.release_command_deg, 'encoder_reverse_stroke': args.encoder_reverse_stroke, 'probe_angle_deg': args.probe_angle_deg, 'release_direct_torque': args.release_direct_torque, 'release_viscous_brake': args.release_viscous_brake, 'full_contact_report_hz': args.contact_report_hz, 'wall_limit_s': args.wall_limit_s},
                  'post_start_pose_writes': False, 'screw_constraint': False, 'external_guide': 'Explicit assembly tool; disabled in free_hold',
                  'external_spindle': rotor_report,
                  'body_yaw_review_bound_deg': .60, 'body_bound_basis': 'Legacy finite excursion stop, not a manufacturer tolerance; source-key clearance is audited separately against the triangle-mesh socket',
                  'axial_drive_during_coupling': bool(args.engagement_axial_assist), 'mass_properties': mass, 'wall_seconds': time.monotonic()-started, 'timings': timings, 'images': images,
                  'last_state': {k: v for k, v in rows[-1].items() if k not in ('shape_contacts', 'solver_patch_impulse_records', 'solver_applied_impulse_records')}}
        result['configuration'].update(probe_speed_deg_s=args.probe_speed_deg_s,
                                       probe_drive_damping_nm_s_rad=args.probe_drive_damping,
                                       diagnostic_contact_breakdown=args.diagnostic_contact_breakdown,
                                       external_wrench_audit=args.external_wrench_audit,
                                       external_wrench_filter_paths=socket_filters if args.external_wrench_audit else [],
                                       segmented_mating=args.segmented_mating,
                                       external_forces_every_iteration=args.solver == 'TGS')
        result['configuration']['free_lateral_and_tilt_tool_coordinates'] = args.free_guide
        result['configuration']['start_separated_with_finite_axial_support'] = args.start_separated
        result['configuration']['approach_support'] = ({'linear_stiffness_n_m':1000., 'linear_damping_ns_m':15., 'linear_force_cap_n':3.0400615,
            'angular_stiffness_nm_rad':.5, 'angular_damping_nm_s_rad':.005, 'angular_torque_cap_nm':.2,
            'active_only_before_turning':True,'scope':'DECLARED_EXTERNAL_COMPLIANT_GRIPPER_NOT_CONNECTOR'} if args.start_separated else None)
        result['configuration']['encoder_forward_completion'] = args.encoder_forward_completion
        result['configuration']['specification_speed_profile'] = args.spec_speed_profile
        result['configuration']['engagement_axial_assist'] = args.engagement_axial_assist
        result['configuration']['key_blocking_probe'] = args.key_blocking_probe
        result['configuration']['uninterrupted_round_trip'] = args.round_trip
        result['configuration']['external_spindle_encoder_drive'] = ({'angle_resolution_deg':.01,'velocity_filter_s':.002,
            'position_gain_nm_rad':30.,'velocity_gain_nm_s_rad':.62,'torque_cap_nm':args.torque_cap_nm,
            'reverse_velocity_filter_s':.0005,'reverse_velocity_gain_nm_s_rad':2.,
            'connector_truth_used_for_control':False} if args.spindle_encoder_drive else None)
        if report_fix is not None:
            result['native_report_fix'] = report_fix.check()
        (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    except Exception:
        import traceback
        failed = True; error = traceback.format_exc(); (args.output/'error.txt').write_text(error); print(error, flush=True)
        (args.output/'failure_timing.json').write_text(json.dumps({'wall_s': time.monotonic()-started, 'timings': locals().get('timings'), 'steps': len(locals().get('rows', []))}, indent=2)+'\n')
    finally:
        if locals().get('report_fix') is not None:
            (args.output/'native_report_fix.json').write_text(json.dumps(report_fix.status(), indent=2)+'\n')
            report_fix.detach()
        if 'video' in locals(): video.release()
        # Images/video/results are synchronously saved above; there are no
        # Replicator writers to drain. Official fast exit avoids the observed
        # stage-teardown deadlock without altering or hiding the physical result.
        app.close(wait_for_replicator=False, skip_cleanup=True, exit_code=1 if failed else 0)
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
