#!/usr/bin/env python3
"""A gravity-loaded synthetic pendulum checks the configured brake semantics.

No project connector, hand, robot or saved assembly state is loaded. Loads are
gravity on known masses; this is a physics coupon, not assembly evidence.
"""

import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--velocity-iterations", type=int, required=True)
parser.add_argument("--damping", type=float, default=1e10)
parser.add_argument("--steps", type=int, default=480)
parser.add_argument("--position-iterations", type=int, default=4)
parser.add_argument("--solver", choices=("TGS", "PGS"), default="TGS")
parser.add_argument("--coaxial-spin", action="store_true")
parser.add_argument("--articulated", action="store_true")
parser.add_argument("--native-friction", action="store_true",
                    help="Synthetic articulated coupon only: native 0.020 Nm static/dynamic friction instead of a velocity drive")
parser.add_argument("--external-forces-every-iteration", type=int, choices=(0, 1))
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.native_friction and not args.articulated:
    parser.error("native joint friction requires --articulated")
args.output.mkdir(parents=True, exist_ok=False)

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
try:
    import numpy as np
    import omni.usd
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleRigidPrim
    from isaacsim.core.simulation_manager import SimulationManager

    dt = 1/240
    SimulationManager.set_physics_sim_device("cuda:0")
    world = World(stage_units_in_meters=1., physics_dt=dt, rendering_dt=1/60,
                  backend="numpy", device="cuda:0", sim_params={"use_gpu_pipeline": True})
    stage = omni.usd.get_context().get_stage()
    physics = UsdPhysics.Scene.Get(stage, world.get_physics_context().prim_path)
    physics.CreateGravityDirectionAttr(Gf.Vec3f(0., -1., 0.))
    physics.CreateGravityMagnitudeAttr(0. if args.coaxial_spin else 9.81)
    settings = PhysxSchema.PhysxSceneAPI.Apply(physics.GetPrim())
    settings.CreateSolverTypeAttr(args.solver)
    settings.CreateMinVelocityIterationCountAttr(args.velocity_iterations)
    settings.CreateMaxVelocityIterationCountAttr(args.velocity_iterations)
    if args.external_forces_every_iteration is not None:
        settings.CreateEnableExternalForcesEveryIterationAttr(bool(args.external_forces_every_iteration))
    views, cases = [], []
    lever = 0. if args.coaxial_spin else .02
    for index, mass in enumerate((.035,) if args.coaxial_spin else (.035, .15)):
        path = f"/World/Coupon{index}"
        shape = UsdGeom.Cube.Define(stage, path)
        shape.CreateSizeAttr(.01)
        shape.AddTranslateOp().Set(Gf.Vec3d(.2*index+lever, 0., 0.))
        rigid = UsdPhysics.RigidBodyAPI.Apply(shape.GetPrim())
        if args.coaxial_spin:
            rigid.CreateAngularVelocityAttr(Gf.Vec3f(0., 0., float(np.rad2deg(100.))))
        mass_api = UsdPhysics.MassAPI.Apply(shape.GetPrim())
        mass_api.CreateMassAttr(mass)
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.))
        inertia = 1.14e-5 if args.coaxial_spin else mass*.01**2/6
        mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(inertia))
        body_api = PhysxSchema.PhysxRigidBodyAPI.Apply(shape.GetPrim())
        body_api.CreateSolverPositionIterationCountAttr(args.position_iterations)
        body_api.CreateSolverVelocityIterationCountAttr(args.velocity_iterations)
        body_api.CreateAngularDampingAttr(0.)
        body_api.CreateLinearDampingAttr(0.)
        joint = UsdPhysics.RevoluteJoint.Define(stage, path+"Joint")
        joint.CreateBody1Rel().SetTargets([path])
        joint.CreateAxisAttr("Z")
        joint.CreateLocalPos0Attr(Gf.Vec3f(.2*index, 0., 0.))
        joint.CreateLocalPos1Attr(Gf.Vec3f(-lever, 0., 0.))
        if args.articulated:
            base_path = path+"Base"
            base = UsdGeom.Cube.Define(stage, base_path)
            base.CreateSizeAttr(.01)
            base.AddTranslateOp().Set(Gf.Vec3d(.2*index, 0., 0.))
            UsdPhysics.RigidBodyAPI.Apply(base.GetPrim())
            base_mass = UsdPhysics.MassAPI.Apply(base.GetPrim())
            base_mass.CreateMassAttr(1.)
            base_mass.CreateDiagonalInertiaAttr(Gf.Vec3f(.001))
            fixed = UsdPhysics.FixedJoint.Define(stage, path+"BaseToWorld")
            fixed.CreateBody1Rel().SetTargets([base_path])
            fixed.CreateLocalPos0Attr(Gf.Vec3f(.2*index, 0., 0.))
            UsdPhysics.ArticulationRootAPI.Apply(fixed.GetPrim())
            art = PhysxSchema.PhysxArticulationAPI.Apply(fixed.GetPrim())
            art.CreateSolverPositionIterationCountAttr(args.position_iterations)
            art.CreateSolverVelocityIterationCountAttr(args.velocity_iterations)
            joint.CreateBody0Rel().SetTargets([base_path])
            joint.CreateLocalPos0Attr(Gf.Vec3f(0.))
        if args.native_friction:
            from omni.physx.bindings._physx import (
                JOINT_AXIS_API, JOINT_AXIS_ATTR_STATIC_FRICTION_EFFORT_ANGULAR,
                JOINT_AXIS_ATTR_DYNAMIC_FRICTION_EFFORT_ANGULAR,
                JOINT_AXIS_ATTR_VISCOUS_FRICTION_COEFFICIENT_ANGULAR)
            joint.GetPrim().ApplyAPI(JOINT_AXIS_API, UsdPhysics.Tokens.angular)
            for name, value in ((JOINT_AXIS_ATTR_STATIC_FRICTION_EFFORT_ANGULAR, .02),
                                (JOINT_AXIS_ATTR_DYNAMIC_FRICTION_EFFORT_ANGULAR, .02),
                                (JOINT_AXIS_ATTR_VISCOUS_FRICTION_COEFFICIENT_ANGULAR, 0.)):
                joint.GetPrim().CreateAttribute(name, Sdf.ValueTypeNames.Float).Set(value)
        else:
            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(0.)
            drive.CreateTargetVelocityAttr(0.)
            drive.CreateDampingAttr(args.damping)
            drive.CreateMaxForceAttr(.02)
        if args.coaxial_spin and args.articulated:
            initial_state = PhysxSchema.JointStateAPI.Apply(joint.GetPrim(), UsdPhysics.Tokens.angular)
            initial_state.CreateVelocityAttr(float(np.rad2deg(100.)))
        view = SingleRigidPrim(path, name=f"coupon{index}")
        world.scene.add(view)
        views.append(view)
        cases.append({"mass_kg": mass, "lever_m": lever,
                      "initial_gravity_torque_nm": mass*9.81*lever,
                      "center_inertia_kg_m2": inertia})
    stage.GetRootLayer().Export(str(args.output / "coupon_before_reset.usda"))
    world.reset()
    def array(value):
        return value.cpu().numpy() if hasattr(value, "cpu") else np.asarray(value)
    for view, case in zip(views, cases):
        case["runtime_mass_kg"] = float(array(view.get_mass()))
        case["runtime_inertia_kg_m2"] = array(view._rigid_prim_view.get_inertias()).tolist()
        case["angular_velocity_at_reset_rad_s"] = array(view.get_angular_velocity()).tolist()
    stage.GetRootLayer().Export(str(args.output / "coupon_after_reset.usda"))
    rows = []
    for step in range(args.steps):
        world.step(render=False)
        row = []
        for view in views:
            p, q = view.get_world_pose()
            omega = view.get_angular_velocity()
            if hasattr(p, "cpu"):
                p, q, omega = p.cpu().numpy(), q.cpu().numpy(), omega.cpu().numpy()
            row.append(np.r_[p, q, omega])
        rows.append(row)
    values = np.asarray(rows)
    for i, case in enumerate(cases):
        q = values[:, i, 3:7]
        angle = np.unwrap(2*np.arctan2(q[:, 3], q[:, 0]))
        case.update(maximum_absolute_angle_deg=float(np.rad2deg(abs(angle).max())),
                    final_angle_deg=float(np.rad2deg(angle[-1])),
                    final_angular_velocity_world_rad_s=values[-1, i, 7:].tolist())
    result = {"scope": "SYNTHETIC_COAXIAL_SPIN_BRAKING" if args.coaxial_spin else "SYNTHETIC_GRAVITY_COUPON_NOT_CONNECTOR_ASSEMBLY",
              "source_assets_used_or_modified": False, "solver": args.solver,
              "solver_after_reset": settings.GetSolverTypeAttr().Get(),
              "scene_velocity_iteration_bounds_after_reset": [settings.GetMinVelocityIterationCountAttr().Get(), settings.GetMaxVelocityIterationCountAttr().Get()],
              "gpu_dynamics": world.get_physics_context().is_gpu_dynamics_enabled(),
              "physics_dt_s": dt, "position_iterations": args.position_iterations,
              "velocity_iterations": args.velocity_iterations,
              "external_forces_every_iteration": settings.GetEnableExternalForcesEveryIterationAttr().Get(),
              "drive_damping_nm_s_per_deg": None if args.native_friction else args.damping,
              "drive_cap_nm": None if args.native_friction else .02,
              "native_joint_friction": args.native_friction,
              "native_static_and_dynamic_friction_effort_nm": .02 if args.native_friction else None,
              "coaxial_spin": args.coaxial_spin,
              "articulated_fixed_base_coupon": args.articulated,
              "ideal_coaxial_spin_deceleration_rad_s2": .02/1.14e-5 if args.coaxial_spin else None,
              "below_cap_viscous_creep_estimate_deg_s": None if args.native_friction else .006867/args.damping,
              "theoretical_expectation": "The 0.006867 Nm gravity load should be held by an ideal 0.020 Nm static brake; the 0.02943 Nm load exceeds its cap.",
              "cases": cases}
    np.savez_compressed(args.output / "samples.npz", state=values)
    (args.output / "result.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2), flush=True)
except Exception:
    import traceback
    failure = traceback.format_exc()
    (args.output / "error.txt").write_text(failure)
    print(failure, flush=True)
finally:
    app.close()
