#!/usr/bin/env python3
"""Minimal Isaac Sim assembly smoke test for the TE/DEUTSCH J35 USD pair."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--physics-hz", type=float, default=240.0)
    return parser.parse_args()


def _world_pose(Usd, UsdGeom, prim):
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    translation = matrix.ExtractTranslation()
    rotation = transform.ExtractRotation().GetQuat()
    return translation, rotation


def _relative_z_angle(Usd, UsdGeom, body_prim, nut_prim) -> float:
    body_matrix = UsdGeom.Xformable(body_prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    nut_matrix = UsdGeom.Xformable(nut_prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    body_quaternion = body_matrix.ExtractRotation().GetQuat()
    nut_quaternion = nut_matrix.ExtractRotation().GetQuat()
    relative = body_quaternion.GetInverse() * nut_quaternion
    imaginary = relative.GetImaginary()
    angle = 2.0 * math.atan2(float(imaginary[2]), float(relative.GetReal()))
    return math.atan2(math.sin(angle), math.cos(angle))


def _unwrap(previous: float, wrapped: float) -> float:
    previous_wrapped = math.atan2(math.sin(previous), math.cos(previous))
    delta = math.atan2(
        math.sin(wrapped - previous_wrapped),
        math.cos(wrapped - previous_wrapped),
    )
    return previous + delta


def main() -> int:
    arguments = _arguments()
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
    metrics: dict[str, object] = {
        "stage": str(arguments.stage),
        "passed": False,
    }
    try:
        import omni.usd
        from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

        from isaacsim.core.api import World

        stage_path = str(arguments.stage.resolve())
        if not omni.usd.get_context().open_stage(stage_path):
            raise RuntimeError(f"cannot open stage: {stage_path}")
        stage = omni.usd.get_context().get_stage()
        world = World(
            stage_units_in_meters=1.0,
            physics_dt=1.0 / arguments.physics_hz,
            rendering_dt=1.0 / 60.0,
        )

        root_path = "/World/TE_J35Assembly"
        body_path = f"{root_path}/LoosePlug/BodyAssembly"
        nut_path = f"{root_path}/LoosePlug/CouplingNut"
        hinge_path = f"{root_path}/Joints/CouplingNutRevolute"
        prismatic_path = f"{root_path}/Joints/InsertionPrismatic"
        rack_path = f"{root_path}/Joints/ThreadCoupling"
        body_prim = stage.GetPrimAtPath(body_path)
        nut_prim = stage.GetPrimAtPath(nut_path)
        hinge_prim = stage.GetPrimAtPath(hinge_path)
        prismatic_prim = stage.GetPrimAtPath(prismatic_path)
        rack_prim = stage.GetPrimAtPath(rack_path)
        required = {
            "body": body_prim,
            "nut": nut_prim,
            "hinge": hinge_prim,
            "prismatic": prismatic_prim,
            "rack": rack_prim,
        }
        missing = [name for name, prim in required.items() if not prim]
        if missing:
            raise RuntimeError(f"required prims missing: {missing}")

        assembly_prim = stage.GetPrimAtPath(root_path)
        lead = float(
            assembly_prim.GetAttribute("kcg:threadLeadMPerRevolution").Get()
        )
        direction = int(
            assembly_prim.GetAttribute("kcg:threadDirection").Get()
        )
        ratio = direction * 360.0 / lead
        key_paths = [
            prim.GetPath()
            for prim in stage.Traverse()
            if str(prim.GetPath()).startswith(
                f"{body_path}/Collision/Key_"
            )
        ]
        if len(key_paths) != 5:
            raise RuntimeError(f"expected five key colliders, got {len(key_paths)}")

        world.reset()
        start_position, _ = _world_pose(Usd, UsdGeom, body_prim)
        world.pause()
        insertion_drive = UsdPhysics.DriveAPI.Apply(
            prismatic_prim, UsdPhysics.Tokens.linear
        )
        insertion_drive.CreateTypeAttr(UsdPhysics.Tokens.force)
        insertion_drive.CreateStiffnessAttr(1800.0)
        insertion_drive.CreateDampingAttr(85.0)
        insertion_drive.CreateTargetPositionAttr(-0.0050)
        insertion_drive.CreateTargetVelocityAttr(0.0)
        insertion_drive.CreateMaxForceAttr(8.0)
        world.play()
        simulation_app.update()
        for step in range(round(3.0 * arguments.physics_hz)):
            world.step(render=False)
            position, _ = _world_pose(Usd, UsdGeom, body_prim)
            if abs(float(position[2]) + 0.0010) <= 0.00012:
                break

        engaged_position, engaged_orientation = _world_pose(
            Usd, UsdGeom, body_prim
        )
        insertion_travel = float(engaged_position[2] - start_position[2])
        insertion_passed = bool(
            float(start_position[2]) >= 0.0030
            and insertion_travel <= -0.0040
            and -0.00125 <= float(engaged_position[2]) <= -0.00075
        )
        metrics["key_insertion"] = {
            "key_collider_count": len(key_paths),
            "start_z_m": float(start_position[2]),
            "engaged_z_m": float(engaged_position[2]),
            "travel_m": insertion_travel,
            "steps": step + 1,
            "passed": insertion_passed,
        }
        if not insertion_passed:
            raise RuntimeError("five-key insertion phase did not reach engagement")

        world.pause()
        insertion_drive.GetMaxForceAttr().Set(0.0)
        if not stage.RemovePrim(rack_path):
            raise RuntimeError("could not remove disabled thread coupling")
        if not stage.RemovePrim(prismatic_path):
            raise RuntimeError("could not replace insertion prismatic")

        prismatic = UsdPhysics.PrismaticJoint.Define(stage, prismatic_path)
        prismatic.CreateAxisAttr(UsdGeom.Tokens.z)
        prismatic.CreateBody1Rel().SetTargets([Sdf.Path(body_path)])
        prismatic.CreateLocalPos0Attr(
            Gf.Vec3f(*(float(value) for value in engaged_position))
        )
        prismatic.CreateLocalRot0Attr(
            Gf.Quatf(
                float(engaged_orientation.GetReal()),
                Gf.Vec3f(
                    *(float(value) for value in engaged_orientation.GetImaginary())
                ),
            )
        )
        prismatic.CreateLocalPos1Attr(Gf.Vec3f(0.0))
        prismatic.CreateLocalRot1Attr(Gf.Quatf(1.0))
        prismatic.CreateLowerLimitAttr(-0.0080)
        prismatic.CreateUpperLimitAttr(0.0010)
        prismatic.CreateCollisionEnabledAttr(False)

        rack = PhysxSchema.PhysxPhysicsRackAndPinionJoint.Define(
            stage, rack_path
        )
        rack.CreateBody0Rel().SetTargets([Sdf.Path(nut_path)])
        rack.CreateBody1Rel().SetTargets([Sdf.Path(body_path)])
        rack.CreateHingeRel().SetTargets([Sdf.Path(hinge_path)])
        rack.CreatePrismaticRel().SetTargets([Sdf.Path(prismatic_path)])
        rack.CreateRatioAttr(ratio)

        thread_drive = UsdPhysics.DriveAPI.Apply(
            hinge_prim, UsdPhysics.Tokens.angular
        )
        thread_drive.CreateTypeAttr(UsdPhysics.Tokens.force)
        thread_drive.CreateStiffnessAttr(0.0)
        thread_drive.CreateDampingAttr(0.015)
        thread_drive.CreateTargetPositionAttr(0.0)
        thread_drive.CreateTargetVelocityAttr(90.0)
        thread_drive.CreateMaxForceAttr(0.05)

        world.play()
        simulation_app.update()
        world.step(render=False)
        thread_start_position, _ = _world_pose(Usd, UsdGeom, body_prim)
        unwrapped_angle = 0.0
        for thread_step in range(round(5.0 * arguments.physics_hz)):
            world.step(render=False)
            wrapped = _relative_z_angle(Usd, UsdGeom, body_prim, nut_prim)
            unwrapped_angle = _unwrap(unwrapped_angle, wrapped)
            position, _ = _world_pose(Usd, UsdGeom, body_prim)
            if float(position[2]) <= -0.00750:
                break

        thread_drive.GetTargetVelocityAttr().Set(0.0)
        thread_drive.GetDampingAttr().Set(0.05)
        for _ in range(round(0.15 * arguments.physics_hz)):
            world.step(render=False)
            wrapped = _relative_z_angle(Usd, UsdGeom, body_prim, nut_prim)
            unwrapped_angle = _unwrap(unwrapped_angle, wrapped)

        final_position, _ = _world_pose(Usd, UsdGeom, body_prim)
        axial_travel = float(final_position[2] - thread_start_position[2])
        expected_travel = lead * abs(unwrapped_angle) / (2.0 * math.pi)
        helical_error = abs(abs(axial_travel) - expected_travel)
        thread_passed = bool(
            math.isfinite(unwrapped_angle)
            and axial_travel <= -0.0062
            and abs(unwrapped_angle) >= math.radians(285.0)
            and helical_error <= 0.00025
            and -0.00805 <= float(final_position[2]) <= -0.00745
        )
        metrics["thread_lock"] = {
            "lead_m_per_revolution": lead,
            "ratio_degrees_per_meter": ratio,
            "angle_degrees": math.degrees(unwrapped_angle),
            "start_z_m": float(thread_start_position[2]),
            "final_z_m": float(final_position[2]),
            "axial_travel_m": axial_travel,
            "expected_travel_m": expected_travel,
            "helical_error_m": helical_error,
            "steps": thread_step + 1,
            "passed": thread_passed,
        }
        metrics["passed"] = bool(insertion_passed and thread_passed)
        if not metrics["passed"]:
            raise RuntimeError("thread-lock phase did not meet the smoke criteria")
        return 0
    finally:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(metrics, ensure_ascii=False, sort_keys=True))
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
