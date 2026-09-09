#!/usr/bin/env python3
"""ARTICULATED_BODY_NUT_KNOWN_LOAD_COUPON_NOT_ASSEMBLY.

Copy of hard_stop_probe retaining the original floating Body+Nut articulation,
internal revolute joint, masses and inertias. There is no external/world joint.
A time-only native force is applied exclusively to Nut. Only Core, optional
original stop proxies and Socket can collide in this disposable laboratory.
No object pose is written after physics starts. Root alone schedules execution.
"""
import argparse
import json
from pathlib import Path
import sys
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--repository", type=Path, default=Path.cwd())
p.add_argument("--source", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--seal-stiffness", type=float, default=100.)
p.add_argument("--seal-damping", type=float, default=0.)
p.add_argument("--socket-friction", type=float, default=.45)
p.add_argument("--initial-depth-m", type=float, default=.01428)
p.add_argument("--stroke-duration-s", type=float, default=8.)
p.add_argument("--hold-duration-s", type=float, default=1.)
p.add_argument("--physics-hz", type=int, choices=(240,), default=240)
p.add_argument("--solver", choices=("TGS","PGS"), default="TGS")
p.add_argument("--convex-stop-proxy", action="store_true")
p.add_argument("--stop-boxes", action="store_true")
p.add_argument("--enable-seal", action="store_true", help="Diagnostic: add only the existing compliant seal to the same actor pair")
p.add_argument("--known-force-n", type=float, default=50.)
args = p.parse_args()
repo, out = args.repository.resolve(), args.output.resolve()
out.mkdir(parents=True, exist_ok=False)
(out/'driver_snapshot.py').write_bytes(Path(__file__).read_bytes())
sys.path.insert(0, str(repo / "src/kcg_connector/isaac"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "multi_gpu": False, "fast_shutdown": True})
failed, wall_start = False, time.perf_counter()
try:
    import carb
    import cv2
    import numpy as np
    import omni.usd
    import omni.replicator.core as rep
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade
    from isaacsim.core.api import World
    from isaacsim.core.experimental.prims import RigidPrim
    from isaacsim.core.simulation_manager import SimulationManager
    from omni.physx.bindings._physx import SETTING_DISABLE_CONTACT_PROCESSING
    from te_foundationpose_handoff_runtime import _author_camera, _camera_cv_pose_from_eye_target
    from seal_contact_model import install_front_seal

    dt = 1. / args.physics_hz
    if not (.014 <= args.initial_depth_m < .014605
            and 0 < args.seal_stiffness and 0 <= args.seal_damping
            and 0 < args.socket_friction <= 1.
            and 0 <= args.known_force_n <= 50.
            and args.stroke_duration_s >= 1.):
        raise ValueError("bounded declared seal coupon settings required")
    report = json.loads(args.source.with_name("assembly_scene.json").read_text())
    source = Usd.Stage.Open(str(args.source))
    SimulationManager.set_physics_sim_device("cuda:0")
    carb.settings.get_settings().set_bool(SETTING_DISABLE_CONTACT_PROCESSING, False)
    world = World(stage_units_in_meters=1., physics_dt=dt, rendering_dt=dt,
                  backend="numpy", device="cuda:0", sim_params={"use_gpu_pipeline": True})
    stage = omni.usd.get_context().get_stage()
    layer = stage.GetRootLayer()
    UsdGeom.Xform.Define(stage, "/World")
    for prim in source.GetPrimAtPath("/World").GetChildren():
        # The source background fixture has its own unrelated FixtureToWorld
        # joint. Do not copy that inactive-contact backdrop into this coupon:
        # the Socket itself is a static mesh and requires no world joint.
        if prim.GetName() in ("HandArm", "FixtureMaterial", "D38999TabletopV1"):
            continue
        if not Sdf.CopySpec(source.GetRootLayer(), prim.GetPath(), layer, prim.GetPath()):
            raise RuntimeError(f"source copy failed: {prim.GetPath()}")
    scene = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath(world.get_physics_context().prim_path))
    scene.CreateSolverTypeAttr(args.solver)
    scene.CreateEnableExternalForcesEveryIterationAttr(True)
    scene.CreateMinVelocityIterationCountAttr(1)
    scene.CreateMaxVelocityIterationCountAttr(1)
    seal = install_front_seal(repo, stage, report, stiffness_n_m=args.seal_stiffness,
                              damping_ns_m=args.seal_damping)
    body_path, seal_path, socket_path = seal["body_path"], seal["rigid_core_collision"], seal["socket_collision"]
    proxy_paths=[]; proxy_record=None
    if args.stop_boxes:
        from contact_stop_proxy import install_stop_boxes
        metadata=Path(__file__).resolve().parents[1]/'references/contact_stop_proxy_candidate.json'
        proxy_record=install_stop_boxes(stage,body_path,rigid_core_path=seal_path,record=json.loads(metadata.read_text()))
        proxy_paths=proxy_record['installed_paths']
        (out/'stop_proxy_authoring.json').write_text(json.dumps(proxy_record,indent=2)+'\n')
    if args.convex_stop_proxy:
        from contact_stop_proxy import install_stop_sectors
        metadata=Path(__file__).resolve().parents[1]/'references/contact_stop_proxy_candidate.json'
        proxy_metadata=json.loads(metadata.read_text()); proxy_data=np.load(proxy_metadata['geometry_npz'])
        proxy_record=install_stop_sectors(stage,body_path,rigid_core_path=seal_path,
             vertices=proxy_data['vertices_m'],faces=proxy_data['faces'],record=proxy_metadata)
        proxy_paths=proxy_record['installed_paths']
        (out/'stop_proxy_authoring.json').write_text(json.dumps(proxy_record,indent=2)+'\n')
    body = stage.GetPrimAtPath(body_path)
    nut_path = "/World/TE_J35FreeSplitPlug/CouplingNut"
    nut = stage.GetPrimAtPath(nut_path)
    joint_path = "/World/TE_J35FreeSplitPlug/Joints/CouplingNutRevolute"
    internal_joint = stage.GetPrimAtPath(joint_path)
    fields = ("physics:mass", "physics:centerOfMass", "physics:diagonalInertia", "physics:principalAxes")
    parts = (body, nut)
    mass = {str(part.GetPath()): {name: str(part.GetAttribute(name).Get()) for name in fields}
            for part in parts}
    source_masses = np.asarray([float(UsdPhysics.MassAPI(part).GetMassAttr().Get()) for part in parts])
    if not np.isclose(source_masses.sum(), .062459669349, atol=1e-8, rtol=0.):
        raise RuntimeError("source must retain the known Body+Nut combined mass")
    if (not body.HasAPI(UsdPhysics.ArticulationRootAPI)
            or not body.HasAPI(PhysxSchema.PhysxArticulationAPI)
            or not internal_joint.IsA(UsdPhysics.RevoluteJoint)
            or not all(part.IsActive() and part.HasAPI(UsdPhysics.RigidBodyAPI) for part in parts)):
        raise RuntimeError("source must retain the active original floating Body/Nut articulation")
    joint_api = UsdPhysics.Joint(internal_joint)
    if (joint_api.GetBody0Rel().GetTargets() != [body.GetPath()]
            or joint_api.GetBody1Rel().GetTargets() != [nut.GetPath()]):
        raise RuntimeError("source internal joint topology differs")
    joints = [x for x in stage.Traverse() if x.IsA(UsdPhysics.Joint)]
    if [x.GetPath() for x in joints] != [internal_joint.GetPath()]:
        raise RuntimeError("laboratory source must have only its original internal joint; no world guide allowed")
    art = PhysxSchema.PhysxArticulationAPI(body)
    art.CreateSolverPositionIterationCountAttr(32)
    art.CreateSolverVelocityIterationCountAttr(1)
    disabled = []
    enabled_paths = [seal_path, socket_path, *proxy_paths]
    if args.enable_seal:
        enabled_paths.append(seal['seal_collision'])
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI) and str(prim.GetPath()) not in enabled_paths:
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
            disabled.append(str(prim.GetPath()))
    mat = UsdShade.Material.Define(stage, "/World/DeclaredRigidStopCouponSocketMaterial")
    m = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    m.CreateStaticFrictionAttr(args.socket_friction)
    m.CreateDynamicFrictionAttr(args.socket_friction)
    m.CreateRestitutionAttr(0.)
    PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim()).CreateFrictionCombineModeAttr("max")
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(socket_path)).Bind(
        mat, UsdShade.Tokens.strongerThanDescendants, "physics")
    socket = np.asarray(report["socket_initial_position_world_m"], float)
    origin = socket + [0., 0., -args.initial_depth_m]
    # Ideal keyed Ry(180) is confirmed by the existing 128-contact centre map.
    # This deliberate pre-start coupon alignment is not an assembly operation.
    UsdGeom.Xformable(body.GetParent()).ClearXformOpOrder()
    for part in parts:
        xf = UsdGeom.Xformable(part)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(*origin))
        xf.AddOrientOp().Set(Gf.Quatf(0., Gf.Vec3f(0., 1., 0.)))
        rb = UsdPhysics.RigidBodyAPI(part)
        rb.CreateVelocityAttr(Gf.Vec3f(0.))
        rb.CreateAngularVelocityAttr(Gf.Vec3f(0.))
    joint_state = PhysxSchema.JointStateAPI.Apply(internal_joint, UsdPhysics.Tokens.angular)
    joint_state.CreatePositionAttr(0.)
    joint_state.CreateVelocityAttr(0.)
    if mass != {str(part.GetPath()): {name: str(part.GetAttribute(name).Get()) for name in fields}
                for part in parts}:
        raise RuntimeError("source Body/Nut mass/inertia changed")
    PhysxSchema.PhysxContactReportAPI.Apply(body).CreateThresholdAttr(0.)
    contact = RigidPrim([body_path], resolve_paths=False, contact_filter_paths=[socket_path], max_contact_count=32768)
    actors = RigidPrim([body_path, nut_path], resolve_paths=False)
    force_actor = RigidPrim([nut_path], resolve_paths=False)
    camera = "/World/DeclaredRigidStopCouponCamera"
    _author_camera(stage, camera, _camera_cv_pose_from_eye_target(socket+[.09, -.12, .06], socket+[0, 0, .005]),
                   resolution=(800, 600), focal_length_mm=50., horizontal_aperture_mm=36.,
                   clipping_range_m=(.01, 5.), Gf=Gf, UsdGeom=UsdGeom)
    UsdLux.DomeLight.Define(stage, "/World/DeclaredRigidStopCouponLight").CreateIntensityAttr(1200.)
    product = rep.create.render_product(camera, (800, 600))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach([product.path])
    layer.Export(str(out / "laboratory_coupon_before_physics.usdc"))
    from omni.physx import get_physx_interface
    native_step_events=[]
    native_subscription=get_physx_interface().subscribe_physics_step_events(lambda step_dt:native_step_events.append(float(step_dt)))
    world.reset()
    material_view=SimulationManager._physics_sim_view__warp.create_articulation_view(body_path)
    compliant,combine=material_view.get_compliant_material_properties()
    (out/'native_material_readback.json').write_text(json.dumps({
        'shape_count':material_view.max_shapes,'link_paths':material_view.link_paths,
        'friction_restitution':material_view.get_material_properties().numpy().tolist(),
        'compliant_stiffness_damping':compliant.numpy().tolist(),
        'compliant_combine_modes':combine.numpy().tolist()},indent=2)+'\n')
    native_masses = np.asarray(actors.get_masses().numpy(), float).ravel()
    if native_masses.shape != (2,) or not np.allclose(native_masses, source_masses, rtol=0., atol=1e-8):
        raise RuntimeError("native Body/Nut masses differ from unchanged source")
    gravity = float(UsdPhysics.Scene(stage.GetPrimAtPath(world.get_physics_context().prim_path)).GetGravityMagnitudeAttr().Get())
    if gravity == float('-inf'):
        gravity = float(np.float32(9.81))
    expected_weight = float(native_masses.sum()*gravity)
    (out/'mass_and_topology_readback.json').write_text(json.dumps({
        'actor_paths':[body_path,nut_path],'source_masses_kg':source_masses.tolist(),
        'native_masses_kg':native_masses.tolist(),'combined_mass_kg':float(native_masses.sum()),
        'expected_zero_load_support_n':expected_weight,
        'expected_loaded_support_n':expected_weight+args.known_force_n,
        'force_applied_only_to':nut_path,'world_joints':[], 'original_internal_joint':joint_path,
        'floating_articulation_preserved':True,'position_iterations':32,'velocity_iterations':1,
        'physics_dt_s':dt},indent=2)+'\n')
    (out/'reset_native_step_events.json').write_text(json.dumps(native_step_events)+'\n')
    native_step_events.clear()

    def host(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else x.numpy() if hasattr(x, "numpy") else np.asarray(x)

    def poses():
        return np.concatenate([host(x).ravel() for x in actors.get_world_poses()])

    images = []
    def capture(name):
        import omni.timeline
        from omni.physxfabric import get_physx_fabric_interface
        before, now, playing = poses().copy(), float(world.current_time), bool(world.is_playing())
        dt_before=float(world.get_physics_dt())
        if SimulationManager.is_fabric_enabled():
            get_physx_fabric_interface().force_update(dt, now)
        timeline = omni.timeline.get_timeline_interface()
        auto = timeline.is_auto_updating()
        settings = carb.settings.get_settings()
        old = settings.get("/app/player/playSimulations")
        try:
            timeline.set_auto_update(False)
            timeline.commit_silently()
            settings.set("/app/player/playSimulations", False)
            for _ in range(3):
                omni.kit.app.get_app().update()
            settings.set("/app/player/playSimulations", old)
            rep.orchestrator.step(rt_subframes=1, delta_time=0., pause_timeline=not playing)
        finally:
            settings.set("/app/player/playSimulations", old)
            timeline.set_auto_update(auto)
            timeline.commit_silently()
        rgba = np.asarray(rgb.get_data())
        dt_after=float(world.get_physics_dt())
        print('CAPTURE_TIMESTEP',json.dumps({'name':name,'before':dt_before,'after':dt_after,'playing':playing}),flush=True)
        delta = float(np.max(abs(poses()-before)))
        if rgba.ndim != 3 or not rgba.size or delta > 0 or float(world.current_time) != now:
            raise RuntimeError("coupon evidence render altered native state/time or failed")
        cv2.imwrite(str(out/(name+".png")), cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR))
        images.append({"image": name+".png", "time_s": now, "native_pose_delta": delta})
        if playing and not world.is_playing():
            world.play()

    rows, step = [], 0
    stream = (out / "samples.jsonl").open("x", buffering=1)
    capture("initial")
    applied_force_n = 0.
    import warp as wp
    loaded_indices = wp.array(np.asarray([0],np.int32),dtype=wp.int32,device='cuda:0')
    for phase, duration in (("initial_hold", .3), ("compression", args.stroke_duration_s),
                            ("compressed_hold", args.hold_duration_s),
                            ("withdrawal", args.stroke_duration_s), ("withdrawn_hold", .5)):
        count_steps = round(duration/dt)
        for i in range(count_steps):
            u = (i+1)/count_steps
            fraction = 10*u**3-15*u**4+6*u**5
            if phase == "compression":
                applied_force_n = args.known_force_n*fraction
            elif phase == "withdrawal":
                applied_force_n = args.known_force_n*(1-fraction)
            applied_world = np.array([0.,0.,-applied_force_n],np.float32)
            applied=wp.array(applied_world[None,:],dtype=wp.float32,device='cuda:0')
            # Only the native view containing Nut receives a force. No USD
            # force API is enabled, and no Body force or guide drive exists.
            force_actor._physics_rigid_body_view.apply_forces(applied,loaded_indices,is_global=True)
            actual_step_dt=float(world.get_physics_dt())
            world.step(render=False)
            positions, quats = (host(x) for x in actors.get_world_poses())
            linear, angular = (host(x) for x in actors.get_velocities())
            force, points, normals, separation, counts, starts, ids = contact.get_raw_contact_data()
            force, points, normals, separation, counts, starts = (host(x) for x in (force, points, normals, separation, counts, starts))
            count, start = int(counts.ravel()[0]), int(starts.ravel()[0])
            sl = slice(start, start+count)
            sep = separation.ravel()[sl]
            f = force.ravel()[sl, None]*normals[sl]/dt
            normal_wrench = np.r_[f.sum(0), np.cross(points[sl]-positions[0], f).sum(0)]
            fr, fp, fc, fs = (host(x) for x in contact.get_friction_data())
            friction_wrench = np.zeros(6)
            for n, first in zip(fc.ravel(), fs.ravel()):
                s = slice(int(first), int(first+n))
                ffr = fr[s]/dt
                friction_wrench += np.r_[ffr.sum(0), np.cross(fp[s]-positions[0], ffr).sum(0)]
            depth = float(socket[2]-positions[0, 2])
            row = {"scope": "ARTICULATED_BODY_NUT_KNOWN_LOAD_COUPON_NOT_ASSEMBLY", "step": step,
                   "declared_actuation": "TIME_ONLY_KNOWN_NATIVE_FORCE_ON_NUT_ONLY",
                   "known_external_force_world_n":applied_world.tolist(),
                   "force_actor_path":nut_path,
                   "native_masses_kg":host(actors.get_masses()).ravel().tolist(),
                   "native_combined_mass_kg":float(native_masses.sum()),
                   "expected_static_support_force_n":expected_weight+applied_force_n,
                   "positions_world_m":positions.tolist(),"quaternions_wxyz":quats.tolist(),
                   "native_linear_velocities_world_m_s":linear.tolist(),
                   "native_angular_velocities_world_rad_s":angular.tolist(),
                   "world_physics_dt_s":actual_step_dt,
                   "native_step_event_dt_s":native_step_events.copy(),
                   "net_contact_force_world_n":host(contact.get_net_contact_forces(dt=dt)).tolist(),
                   "time_s": float(world.current_time), "phase": phase,
                   "actual_depth_m": depth, "position_world_m": positions[0].tolist(),
                   "quaternion_wxyz": quats[0].tolist(), "native_velocity_world_m_s": linear[0].tolist(),
                   "native_angular_velocity_rad_s": angular[0].tolist(),
                   "normal_wrench_n_nm": normal_wrench.tolist(), "friction_wrench_n_nm": friction_wrench.tolist(),
                   "normal_axial_resistance_n": float(normal_wrench[2]),
                   "friction_axial_resistance_n": float(friction_wrench[2]),
                   "total_axial_resistance_n": float(normal_wrench[2]+friction_wrench[2]),
                   "normal_force_magnitude_sum_n": float(np.linalg.norm(f, axis=1).sum()),
                   "normal_point_count": count,
                   "penetration_sum_m": float(np.maximum(-sep, 0.).sum()),
                   "penetration_max_m": float(np.maximum(-sep, 0.).max()) if count else 0.,
                   "lateral_error_m": float(np.linalg.norm(positions[0, :2]-socket[:2]))}
            stream.write(json.dumps(row, separators=(",", ":"))+"\n")
            native_step_events.clear()
            rows.append(row)
            if step % round(1/dt) == 0:
                print(json.dumps({"phase": phase, "depth_mm": depth*1000, "force_n": row["total_axial_resistance_n"], "count": count}), flush=True)
                (out / "progress.json").write_text(json.dumps(row, indent=2)+"\n")
            if phase == "compressed_hold" and i == count_steps-1:
                np.savez_compressed(out/"compressed_raw.npz", force_world_n=f, points_world_m=points[sl],
                                    normals_world=normals[sl], separations_m=sep,
                                    friction_impulse_ns=fr, friction_points_world_m=fp,
                                    friction_counts=fc, friction_starts=fs)
            if not np.isfinite(positions).all() or abs(depth-args.initial_depth_m) > .001:
                raise RuntimeError("coupon pose invalid or outside declared stroke")
            step += 1
        if phase == "compressed_hold":
            capture("compressed")
    capture("withdrawn")
    stream.close()
    world.pause()
    hold_summaries = []
    for phase in ("initial_hold", "compressed_hold", "withdrawn_hold"):
        selected = [r for r in rows if r['phase'] == phase][-round(.25/dt):]
        hold_summaries.append({
            'phase':phase,'tail_samples':len(selected),
            'mean_total_contact_axial_n':float(np.mean([r['total_axial_resistance_n'] for r in selected])),
            'expected_static_support_n':float(np.mean([r['expected_static_support_force_n'] for r in selected])),
            'mean_force_minus_expected_n':float(np.mean([r['total_axial_resistance_n']-r['expected_static_support_force_n'] for r in selected])),
            'body_depth_range_m':[min(r['actual_depth_m'] for r in selected),max(r['actual_depth_m'] for r in selected)]})
    result = {"scope": "ARTICULATED_BODY_NUT_KNOWN_LOAD_COUPON_NOT_ASSEMBLY", "source_scene": str(args.source),
              "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "seal": seal, "source_body_and_nut_mass_properties": mass,
              "native_masses_kg":native_masses.tolist(),"native_combined_mass_kg":float(native_masses.sum()),
              "expected_zero_load_support_force_n":expected_weight,
              "expected_fully_loaded_support_force_n":expected_weight+args.known_force_n,
              "force_applied_only_to":nut_path,
              "hold_phase_comparisons":hold_summaries,
              "laboratory_collision_pair": [seal_path, socket_path],
              "collisions_disabled_only_in_disposable_coupon": disabled,
              "original_floating_articulation_and_internal_joint_preserved":True,
              "external_or_world_joint_present":False,
              "phase_names_withdrawal_means_force_unload_not_position_withdrawal":True,
              "post_start_object_pose_writes": False,
              "initial_depth_m": rows[0]["actual_depth_m"], "maximum_depth_m": max(r["actual_depth_m"] for r in rows),
              "final_depth_m": rows[-1]["actual_depth_m"],
              "maximum_total_axial_resistance_n": max(r["total_axial_resistance_n"] for r in rows),
              "maximum_normal_force_sum_n": max(r["normal_force_magnitude_sum_n"] for r in rows),
              "maximum_lateral_error_m": max(r["lateral_error_m"] for r in rows),
              "final_normal_point_count": rows[-1]["normal_point_count"],
              "final_total_axial_resistance_n": rows[-1]["total_axial_resistance_n"],
              "images": images, "wall_seconds": time.perf_counter()-wall_start,
              "constitutive_mapping_requires_postrun_evaluation": True}
    (out / "result.json").write_text(json.dumps(result, indent=2)+"\n")
except Exception:
    import traceback
    failed = True
    error = traceback.format_exc()
    (out / "error.txt").write_text(error)
    print(error, flush=True)
finally:
    app.close(exit_code=1 if failed else 0)
raise SystemExit(1 if failed else 0)
