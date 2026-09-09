#!/usr/bin/env python3
"""Known-load active-joint coupon for TGS velocity-state fidelity.

No project robot or connector. Compare the same finite PD drives under one scene
flag. Poses, velocities and targets are never written after simulation starts.
"""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--external-forces-every-iteration", type=int, choices=(0, 1), required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "multi_gpu": False})
failed = False
try:
    import numpy as np
    import omni.usd
    from scipy.optimize import brentq
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleRigidPrim
    from isaacsim.core.simulation_manager import SimulationManager
    SimulationManager.set_physics_sim_device("cuda:0")
    dt, stiffness, damping, cap, lever = 1/240, 12., 2., 1., .02
    world = World(stage_units_in_meters=1., physics_dt=dt, rendering_dt=1/60,
                  backend="numpy", device="cuda:0", sim_params={"use_gpu_pipeline": True})
    stage = omni.usd.get_context().get_stage()
    scene = UsdPhysics.Scene.Get(stage, world.get_physics_context().prim_path)
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0., -1., 0.))
    scene.CreateGravityMagnitudeAttr(9.81)
    settings = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    settings.CreateSolverTypeAttr("TGS")
    settings.CreateMinVelocityIterationCountAttr(1); settings.CreateMaxVelocityIterationCountAttr(1)
    before_flag = settings.GetEnableExternalForcesEveryIterationAttr().Get()
    settings.CreateEnableExternalForcesEveryIterationAttr(bool(args.external_forces_every_iteration))
    views, cases = [], []
    for i, mass in enumerate((.035, .326260829362/(9.81*lever))):
        path = f"/World/Coupon{i}"
        moving = UsdGeom.Cube.Define(stage, path)
        moving.CreateSizeAttr(.01); moving.AddTranslateOp().Set(Gf.Vec3d(.2*i+lever, 0., 0.))
        UsdPhysics.RigidBodyAPI.Apply(moving.GetPrim())
        properties = UsdPhysics.MassAPI.Apply(moving.GetPrim())
        properties.CreateMassAttr(mass); properties.CreateDiagonalInertiaAttr(Gf.Vec3f(mass*.01**2/6))
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(moving.GetPrim())
        rb.CreateLinearDampingAttr(0.); rb.CreateAngularDampingAttr(0.); rb.CreateSleepThresholdAttr(0.)
        base_path = path+"Base"
        base = UsdGeom.Cube.Define(stage, base_path)
        base.CreateSizeAttr(.01); base.AddTranslateOp().Set(Gf.Vec3d(.2*i, 0., 0.))
        UsdPhysics.RigidBodyAPI.Apply(base.GetPrim())
        bp = UsdPhysics.MassAPI.Apply(base.GetPrim()); bp.CreateMassAttr(1.)
        bp.CreateDiagonalInertiaAttr(Gf.Vec3f(.001))
        clamp = UsdPhysics.FixedJoint.Define(stage, path+"ExplicitBaseClamp")
        clamp.CreateBody1Rel().SetTargets([base_path]); clamp.CreateLocalPos0Attr(Gf.Vec3f(.2*i, 0., 0.))
        UsdPhysics.ArticulationRootAPI.Apply(clamp.GetPrim())
        art = PhysxSchema.PhysxArticulationAPI.Apply(clamp.GetPrim())
        art.CreateSolverPositionIterationCountAttr(32); art.CreateSolverVelocityIterationCountAttr(1)
        art.CreateSleepThresholdAttr(0.)
        joint = UsdPhysics.RevoluteJoint.Define(stage, path+"Joint")
        joint.CreateBody0Rel().SetTargets([base_path]); joint.CreateBody1Rel().SetTargets([path])
        joint.CreateAxisAttr("Z"); joint.CreateLocalPos0Attr(Gf.Vec3f(0.))
        joint.CreateLocalPos1Attr(Gf.Vec3f(-lever, 0., 0.))
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(stiffness*np.pi/180.)
        drive.CreateDampingAttr(damping*np.pi/180.)
        drive.CreateMaxForceAttr(cap); drive.CreateTargetPositionAttr(0.); drive.CreateTargetVelocityAttr(0.)
        views.append(world.scene.add(SingleRigidPrim(path, name=f"coupon{i}", reset_xform_properties=False)))
        cases.append({"mass_kg": mass, "lever_m": lever, "initial_gravity_torque_nm": mass*9.81*lever,
                      "analytical_static_angle_rad": brentq(lambda q: stiffness*q+mass*9.81*lever*np.cos(q), -.1, 0.)})
    stage.GetRootLayer().Export(str(args.output / "coupon_before_reset.usda"))
    world.reset()
    rows = []
    for step in range(960):
        world.step(render=False)
        row = []
        for view in views:
            _, quaternion = view.get_world_pose(); omega = view.get_angular_velocity()
            q = quaternion.cpu().numpy() if hasattr(quaternion, "cpu") else np.asarray(quaternion)
            w = omega.cpu().numpy() if hasattr(omega, "cpu") else np.asarray(omega)
            row.append([2*np.arctan2(q[3], q[0]), w[2]])
        rows.append(row)
    values = np.asarray(rows)
    for i, case in enumerate(cases):
        q = np.unwrap(values[:, i, 0]); velocity = np.diff(q)/dt
        case.update(last_second_mean_angle_rad=float(q[-240:].mean()),
                    static_angle_error_rad=float(q[-240:].mean()-case["analytical_static_angle_rad"]),
                    last_second_mean_reported_velocity_rad_s=float(values[-240:, i, 1].mean()),
                    last_second_mean_pose_difference_velocity_rad_s=float(velocity[-240:].mean()),
                    last_second_max_abs_reported_velocity_rad_s=float(abs(values[-240:, i, 1]).max()))
    result = {"scope": "SYNTHETIC_PD_GRAVITY_COUPON_NOT_HAND_OR_ASSEMBLY",
              "physics_dt_s": dt, "solver": "TGS32_1", "drive_stiffness_nm_rad": stiffness,
              "drive_damping_nm_s_rad": damping, "drive_maximum_effort_nm": cap,
              "scene_flag_default_before_authoring": before_flag,
              "external_forces_every_iteration": settings.GetEnableExternalForcesEveryIterationAttr().Get(),
              "post_start_pose_velocity_or_target_writes": False,
              "source_assets_used_or_modified": False, "cases": cases}
    np.savez_compressed(args.output / "samples.npz", angle_and_reported_velocity=values)
    (args.output / "result.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2), flush=True)
except Exception:
    failed = True
    import traceback
    error = traceback.format_exc(); (args.output / "error.txt").write_text(error); print(error, flush=True)
finally:
    app.close(exit_code=1 if failed else 0)
