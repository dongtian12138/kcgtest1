#!/usr/bin/env python3

"""Exercise the PhysX thread proxy on an isolated two-body unit rig."""

import argparse
import json
import math


def _wrapped_relative_z_angle(Gf, Usd, UsdGeom, body_prim, nut_prim):
    body_matrix = UsdGeom.Xformable(body_prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    nut_matrix = UsdGeom.Xformable(nut_prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    body_quaternion = Gf.Transform(body_matrix).GetRotation().GetQuat()
    nut_quaternion = Gf.Transform(nut_matrix).GetRotation().GetQuat()
    relative = body_quaternion.GetInverse() * nut_quaternion
    imaginary = relative.GetImaginary()
    angle = 2.0 * math.atan2(float(imaginary[2]), float(relative.GetReal()))
    return math.atan2(math.sin(angle), math.cos(angle))


def _unwrap(previous, wrapped):
    previous_wrapped = math.atan2(math.sin(previous), math.cos(previous))
    delta = math.atan2(
        math.sin(wrapped - previous_wrapped),
        math.cos(wrapped - previous_wrapped),
    )
    return previous + delta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lead", type=float, default=0.004)
    parser.add_argument("--direction", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--target-degrees", type=float, default=360.0)
    parser.add_argument("--target-speed-degrees", type=float, default=120.0)
    parser.add_argument("--physics-hz", type=float, default=240.0)
    arguments = parser.parse_args()
    if arguments.lead <= 0.0:
        raise ValueError("--lead must be positive")
    if arguments.target_degrees <= 0.0:
        raise ValueError("--target-degrees must be positive")
    if arguments.target_speed_degrees <= 0.0:
        raise ValueError("--target-speed-degrees must be positive")
    if arguments.physics_hz <= 0.0:
        raise ValueError("--physics-hz must be positive")

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": True,
            "multi_gpu": False,
            "active_gpu": 0,
            "physics_gpu": 0,
        }
    )

    try:
        import omni.usd
        from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

        from isaacsim.core.api import World

        from kcg_connector.thread_proxy import rack_and_pinion_ratio

        world = World(
            stage_units_in_meters=1.0,
            physics_dt=1.0 / arguments.physics_hz,
            rendering_dt=1.0 / 60.0,
        )
        stage = omni.usd.get_context().get_stage()
        world_prim = stage.GetPrimAtPath("/World")
        if not world_prim:
            UsdGeom.Xform.Define(stage, "/World")

        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Scene):
                scene = UsdPhysics.Scene(prim)
                scene.CreateGravityMagnitudeAttr(0.0)

        body = UsdGeom.Xform.Define(stage, "/World/ThreadRig/Body")
        body.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(0.06)
        body_shape = UsdGeom.Cube.Define(
            stage, "/World/ThreadRig/Body/Collision"
        )
        body_shape.CreateSizeAttr(0.020)
        UsdPhysics.CollisionAPI.Apply(body_shape.GetPrim())

        nut = UsdGeom.Xform.Define(stage, "/World/ThreadRig/Nut")
        nut.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
        UsdPhysics.RigidBodyAPI.Apply(nut.GetPrim())
        UsdPhysics.MassAPI.Apply(nut.GetPrim()).CreateMassAttr(0.04)
        nut_shape = UsdGeom.Cylinder.Define(
            stage, "/World/ThreadRig/Nut/Collision"
        )
        nut_shape.CreateAxisAttr(UsdGeom.Tokens.z)
        nut_shape.CreateRadiusAttr(0.015)
        nut_shape.CreateHeightAttr(0.010)
        UsdPhysics.CollisionAPI.Apply(nut_shape.GetPrim())

        hinge_path = "/World/ThreadRig/NutRevolute"
        hinge = UsdPhysics.RevoluteJoint.Define(stage, hinge_path)
        hinge.CreateAxisAttr("Z")
        hinge.CreateBody0Rel().SetTargets([body.GetPrim().GetPath()])
        hinge.CreateBody1Rel().SetTargets([nut.GetPrim().GetPath()])
        hinge.CreateLocalPos0Attr(Gf.Vec3f(0.0))
        hinge.CreateLocalPos1Attr(Gf.Vec3f(0.0))
        hinge.CreateLocalRot0Attr(Gf.Quatf(1.0))
        hinge.CreateLocalRot1Attr(Gf.Quatf(1.0))
        hinge.CreateCollisionEnabledAttr(False)

        prismatic_path = "/World/ThreadRig/InsertionPrismatic"
        prismatic = UsdPhysics.PrismaticJoint.Define(stage, prismatic_path)
        prismatic.CreateAxisAttr("Z")
        prismatic.CreateBody1Rel().SetTargets([body.GetPrim().GetPath()])
        prismatic.CreateLocalPos0Attr(Gf.Vec3f(0.0))
        prismatic.CreateLocalPos1Attr(Gf.Vec3f(0.0))
        prismatic.CreateLocalRot0Attr(Gf.Quatf(1.0))
        prismatic.CreateLocalRot1Attr(Gf.Quatf(1.0))
        maximum_travel = 1.5 * arguments.lead
        prismatic.CreateLowerLimitAttr(-maximum_travel)
        prismatic.CreateUpperLimitAttr(maximum_travel)

        rack = PhysxSchema.PhysxPhysicsRackAndPinionJoint.Define(
            stage, "/World/ThreadRig/ThreadCoupling"
        )
        rack.CreateBody0Rel().SetTargets([nut.GetPrim().GetPath()])
        rack.CreateBody1Rel().SetTargets([body.GetPrim().GetPath()])
        rack.CreateHingeRel().SetTargets([Sdf.Path(hinge_path)])
        rack.CreatePrismaticRel().SetTargets([Sdf.Path(prismatic_path)])
        ratio = rack_and_pinion_ratio(
            arguments.lead, 1.0, arguments.direction
        )
        rack.CreateRatioAttr(ratio)

        drive = UsdPhysics.DriveAPI.Apply(
            hinge.GetPrim(), UsdPhysics.Tokens.angular
        )
        drive.CreateTypeAttr(UsdPhysics.Tokens.force)
        drive.CreateStiffnessAttr(0.0)
        drive.CreateDampingAttr(0.01)
        drive.CreateTargetPositionAttr(0.0)
        drive.CreateTargetVelocityAttr(arguments.target_speed_degrees)
        drive.CreateMaxForceAttr(0.05)

        world.reset()
        initial_body_matrix = UsdGeom.Xformable(
            body.GetPrim()
        ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        initial_z = float(initial_body_matrix.ExtractTranslation()[2])
        unwrapped_angle = 0.0
        maximum_steps = math.ceil(
            2.0
            * arguments.physics_hz
            * arguments.target_degrees
            / arguments.target_speed_degrees
        )
        target_angle = math.radians(arguments.target_degrees)
        for step in range(maximum_steps):
            world.step(render=False)
            wrapped = _wrapped_relative_z_angle(
                Gf, Usd, UsdGeom, body.GetPrim(), nut.GetPrim()
            )
            unwrapped_angle = _unwrap(unwrapped_angle, wrapped)
            if abs(unwrapped_angle) >= target_angle:
                break
        else:
            raise RuntimeError("thread proxy did not reach its target rotation")

        drive.GetTargetVelocityAttr().Set(0.0)
        for _ in range(round(0.5 * arguments.physics_hz)):
            world.step(render=False)
            wrapped = _wrapped_relative_z_angle(
                Gf, Usd, UsdGeom, body.GetPrim(), nut.GetPrim()
            )
            unwrapped_angle = _unwrap(unwrapped_angle, wrapped)

        final_body_matrix = UsdGeom.Xformable(
            body.GetPrim()
        ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        axial_travel = float(final_body_matrix.ExtractTranslation()[2]) - initial_z
        expected_magnitude = (
            arguments.lead * abs(unwrapped_angle) / (2.0 * math.pi)
        )
        helical_error = abs(axial_travel) - expected_magnitude
        finite = all(
            math.isfinite(value)
            for value in (unwrapped_angle, axial_travel, helical_error)
        )
        passed = finite and abs(helical_error) <= 0.0002
        metrics = {
            "axial_travel": axial_travel,
            "coupling_angle_degrees": math.degrees(unwrapped_angle),
            "finite": finite,
            "helical_error": helical_error,
            "lead": arguments.lead,
            "passed": passed,
            "ratio_degrees_per_meter": ratio,
            "steps": step + 1,
        }
        print(json.dumps(metrics, sort_keys=True))
        if not passed:
            raise RuntimeError("PhysX thread proxy relation is outside tolerance")
        print("ISAAC THREAD PROXY PASSED")
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
