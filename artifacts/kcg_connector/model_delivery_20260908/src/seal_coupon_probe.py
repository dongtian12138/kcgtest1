#!/usr/bin/env python3
"""SINGLE_SEAL_COUPON_NOT_ASSEMBLY.

Only the source seal envelope and original Socket collider interact. All other
collisions are explicitly disabled in this disposable laboratory coupon, not
in the deliverable connector. Source Body mass/inertia remain unchanged. A
finite native axial apparatus compresses and withdraws the seal; no object pose
is written after physics starts. The root agent alone schedules execution.
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
p.add_argument("--maximum-depth-m", type=float, default=.014605)
p.add_argument("--stroke-duration-s", type=float, default=8.)
p.add_argument("--hold-duration-s", type=float, default=1.)
p.add_argument("--drive-stiffness-n-m", type=float, default=1e6)
p.add_argument("--drive-damping-ns-m", type=float, default=100.)
p.add_argument("--drive-force-cap-n", type=float, default=250.)
p.add_argument("--physics-hz", type=int, default=240)
args = p.parse_args()
repo, out = args.repository.resolve(), args.output.resolve()
out.mkdir(parents=True, exist_ok=False)
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
    if not (.014 <= args.initial_depth_m < args.maximum_depth_m <= .01460501
            and 0 < args.seal_stiffness and 0 <= args.seal_damping
            and 0 < args.socket_friction <= 1.
            and 0 < args.drive_force_cap_n <= 500.
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
        if prim.GetName() in ("HandArm", "FixtureMaterial"):
            continue
        if not Sdf.CopySpec(source.GetRootLayer(), prim.GetPath(), layer, prim.GetPath()):
            raise RuntimeError(f"source copy failed: {prim.GetPath()}")
    scene = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath(world.get_physics_context().prim_path))
    scene.CreateSolverTypeAttr("TGS")
    scene.CreateEnableExternalForcesEveryIterationAttr(True)
    scene.CreateMinVelocityIterationCountAttr(1)
    scene.CreateMaxVelocityIterationCountAttr(1)
    seal = install_front_seal(repo, stage, report, stiffness_n_m=args.seal_stiffness,
                              damping_ns_m=args.seal_damping)
    body_path, seal_path, socket_path = seal["body_path"], seal["seal_collision"], seal["socket_collision"]
    body = stage.GetPrimAtPath(body_path)
    fields = ("physics:mass", "physics:centerOfMass", "physics:diagonalInertia", "physics:principalAxes")
    mass = {name: str(body.GetAttribute(name).Get()) for name in fields}
    # This is deliberately an isolated mechanical material coupon: disconnect
    # the nut, remove reduced-coordinate Body articulation, and retain exactly
    # the one collision pair whose constitutive response is being measured.
    stage.GetPrimAtPath("/World/TE_J35FreeSplitPlug/Joints").SetActive(False)
    stage.GetPrimAtPath("/World/TE_J35FreeSplitPlug/CouplingNut").SetActive(False)
    body.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    if body.HasAPI(PhysxSchema.PhysxArticulationAPI):
        body.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
    standalone=PhysxSchema.PhysxRigidBodyAPI.Apply(body)
    standalone.CreateSolverPositionIterationCountAttr(32)
    standalone.CreateSolverVelocityIterationCountAttr(1)
    disabled = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI) and str(prim.GetPath()) not in (seal_path, socket_path):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
            disabled.append(str(prim.GetPath()))
    mat = UsdShade.Material.Define(stage, "/World/DeclaredSealCouponSocketMaterial")
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
    xf = UsdGeom.Xformable(body)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*origin))
    xf.AddOrientOp().Set(Gf.Quatf(0., Gf.Vec3f(0., 1., 0.)))
    rb = UsdPhysics.RigidBodyAPI(body)
    rb.CreateVelocityAttr(Gf.Vec3f(0.))
    rb.CreateAngularVelocityAttr(Gf.Vec3f(0.))
    guide = UsdPhysics.Joint.Define(stage, "/World/DeclaredSealCouponAxialGuide")
    guide.CreateBody1Rel().SetTargets([body.GetPath()])
    guide.CreateExcludeFromArticulationAttr(True)
    guide.CreateLocalPos0Attr(Gf.Vec3f(*origin))
    guide.CreateLocalRot0Attr(Gf.Quatf(0., Gf.Vec3f(0., 1., 0.)))
    guide.CreateLocalPos1Attr(Gf.Vec3f(0.))
    guide.CreateLocalRot1Attr(Gf.Quatf(1.))
    for name in ("transX", "transY", "rotX", "rotY", "rotZ"):
        limit = UsdPhysics.LimitAPI.Apply(guide.GetPrim(), name)
        limit.CreateLowAttr(1.)
        limit.CreateHighAttr(-1.)
    drive = UsdPhysics.DriveAPI.Apply(guide.GetPrim(), "transZ")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(args.drive_stiffness_n_m)
    drive.CreateDampingAttr(args.drive_damping_ns_m)
    drive.CreateMaxForceAttr(args.drive_force_cap_n)
    drive.CreateTargetPositionAttr(0.)
    drive.CreateTargetVelocityAttr(0.)
    if mass != {name: str(body.GetAttribute(name).Get()) for name in fields}:
        raise RuntimeError("source Body mass/inertia changed")
    PhysxSchema.PhysxContactReportAPI.Apply(body).CreateThresholdAttr(0.)
    contact = RigidPrim([body_path], resolve_paths=False, contact_filter_paths=[socket_path], max_contact_count=32768)
    camera = "/World/DeclaredSealCouponCamera"
    _author_camera(stage, camera, _camera_cv_pose_from_eye_target(socket+[.09, -.12, .06], socket+[0, 0, .005]),
                   resolution=(800, 600), focal_length_mm=50., horizontal_aperture_mm=36.,
                   clipping_range_m=(.01, 5.), Gf=Gf, UsdGeom=UsdGeom)
    UsdLux.DomeLight.Define(stage, "/World/DeclaredSealCouponLight").CreateIntensityAttr(1200.)
    product = rep.create.render_product(camera, (800, 600))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach([product.path])
    layer.Export(str(out / "laboratory_coupon_before_physics.usdc"))
    world.reset()

    def host(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else x.numpy() if hasattr(x, "numpy") else np.asarray(x)

    def poses():
        return np.concatenate([host(x).ravel() for x in contact.get_world_poses()])

    images = []
    def capture(name):
        import omni.timeline
        from omni.physxfabric import get_physx_fabric_interface
        before, now, playing = poses().copy(), float(world.current_time), bool(world.is_playing())
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
    stroke = args.maximum_depth_m-args.initial_depth_m
    for phase, duration in (("initial_hold", .3), ("compression", args.stroke_duration_s),
                            ("compressed_hold", args.hold_duration_s),
                            ("withdrawal", args.stroke_duration_s), ("withdrawn_hold", .5)):
        count_steps = round(duration/dt)
        for i in range(count_steps):
            u = (i+1)/count_steps
            fraction = 10*u**3-15*u**4+6*u**5
            if phase == "compression":
                drive.GetTargetPositionAttr().Set(stroke*fraction)
            elif phase == "withdrawal":
                drive.GetTargetPositionAttr().Set(stroke*(1-fraction))
            world.step(render=False)
            positions, quats = (host(x) for x in contact.get_world_poses())
            linear, angular = (host(x) for x in contact.get_velocities())
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
            row = {"scope": "SINGLE_SEAL_COUPON_NOT_ASSEMBLY", "step": step,
                   "time_s": float(world.current_time), "phase": phase,
                   "command_depth_m": args.initial_depth_m+float(drive.GetTargetPositionAttr().Get()),
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
    result = {"scope": "SINGLE_SEAL_COUPON_NOT_ASSEMBLY", "source_scene": str(args.source),
              "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "seal": seal, "source_body_mass_properties": mass,
              "laboratory_collision_pair": [seal_path, socket_path],
              "collisions_disabled_only_in_disposable_coupon": disabled,
              "nut_and_body_nut_joint_inactive_only_in_coupon": True,
              "guide_is_declared_maximal_coordinate_native_joint": True,
              "post_start_object_pose_writes": False,
              "initial_depth_m": rows[0]["actual_depth_m"], "maximum_depth_m": max(r["actual_depth_m"] for r in rows),
              "final_depth_m": rows[-1]["actual_depth_m"],
              "maximum_total_axial_resistance_n": max(r["total_axial_resistance_n"] for r in rows),
              "maximum_normal_force_sum_n": max(r["normal_force_magnitude_sum_n"] for r in rows),
              "maximum_tracking_error_m": max(abs(r["actual_depth_m"]-r["command_depth_m"]) for r in rows),
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
