#!/usr/bin/env python3
"""Build the compact PhysX collision and joint layer for the TE J35 pair."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Sequence


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _wrapped_error_deg(first: float, second: float) -> float:
    return (first - second + 180.0) % 360.0 - 180.0


def _relative_asset_path(workspace_path: str, output_path: Path) -> str:
    absolute = (Path.cwd() / workspace_path).resolve()
    return os.path.relpath(absolute, output_path.parent.resolve())


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _arguments(argv)

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {"headless": True, "multi_gpu": False, "active_gpu": 0, "physics_gpu": 0}
    )
    try:
        import yaml
        from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

        document = yaml.safe_load(arguments.config.read_text(encoding="utf-8"))
        keying = document["keying"]
        thread = document["thread"]
        collision = document["collision"]
        mass = document["mass"]
        initial_gap = float(document["assembly_frame"]["initial_face_gap_m"])
        engagement_length = float(keying["engagement_length_m"])

        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        stage = Usd.Stage.CreateNew(str(arguments.output))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())
        scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
        scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr(0.0)
        PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim()).CreateTimeStepsPerSecondAttr(240)

        assembly = UsdGeom.Xform.Define(stage, "/World/TE_J35Assembly")
        assembly_prim = assembly.GetPrim()
        assembly_prim.CreateAttribute(
            "kcg:plug", Sdf.ValueTypeNames.String, custom=True
        ).Set(document["identity"]["plug"])
        assembly_prim.CreateAttribute(
            "kcg:receptacle", Sdf.ValueTypeNames.String, custom=True
        ).Set(document["identity"]["receptacle"])
        assembly_prim.CreateAttribute(
            "kcg:contactCount", Sdf.ValueTypeNames.Int, custom=True
        ).Set(int(document["identity"]["contact_count"]))

        def add_reference(path: str, asset_key: str, *, flip_plug: bool) -> None:
            visual = UsdGeom.Xform.Define(stage, path)
            visual.GetPrim().GetReferences().AddReference(
                _relative_asset_path(document["visual_assets"][asset_key], arguments.output)
            )
            if flip_plug:
                visual.AddRotateXYZOp().Set(Gf.Vec3f(180.0, 0.0, 0.0))

        def add_cube(
            path: str,
            parent_position: tuple[float, float, float],
            dimensions: tuple[float, float, float],
            *,
            rotate_z_deg: float = 0.0,
        ):
            cube = UsdGeom.Cube.Define(stage, path)
            cube.CreateSizeAttr(1.0)
            xformable = UsdGeom.Xformable(cube.GetPrim())
            xformable.AddTranslateOp().Set(Gf.Vec3d(*parent_position))
            if rotate_z_deg:
                xformable.AddRotateZOp().Set(float(rotate_z_deg))
            xformable.AddScaleOp().Set(Gf.Vec3f(*dimensions))
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            return cube.GetPrim()

        def add_cylinder(
            path: str,
            center_z: float,
            radius: float,
            height: float,
        ):
            cylinder = UsdGeom.Cylinder.Define(stage, path)
            cylinder.CreateAxisAttr(UsdGeom.Tokens.z)
            cylinder.CreateRadiusAttr(radius)
            cylinder.CreateHeightAttr(height)
            UsdGeom.Xformable(cylinder.GetPrim()).AddTranslateOp().Set(
                Gf.Vec3d(0.0, 0.0, center_z)
            )
            UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim())
            return cylinder.GetPrim()

        fixed = UsdGeom.Xform.Define(stage, "/World/TE_J35Assembly/FixedReceptacle")
        fixed_rigid = UsdPhysics.RigidBodyAPI.Apply(fixed.GetPrim())
        fixed_rigid.CreateRigidBodyEnabledAttr(True)
        fixed_rigid.CreateKinematicEnabledAttr(True)
        add_reference(
            "/World/TE_J35Assembly/FixedReceptacle/Visual",
            "receptacle",
            flip_plug=False,
        )
        UsdGeom.Scope.Define(stage, "/World/TE_J35Assembly/FixedReceptacle/Collision")

        segment_count = int(collision["receptacle_wall_segment_count"])
        bore_radius = float(keying["receptacle_bore_radius_m"])
        wall_outer_radius = float(collision["receptacle_wall_outer_radius_m"])
        wall_center_radius = 0.5 * (bore_radius + wall_outer_radius)
        wall_radial_thickness = wall_outer_radius - bore_radius
        segment_angle_deg = 360.0 / segment_count
        segment_tangent = 0.92 * 2.0 * math.pi * wall_center_radius / segment_count
        assembly_key_angles = tuple((-float(value)) % 360.0 for value in keying["angles_deg"])
        extra = float(collision["key_clearance_extra_m"])
        keyway_widths = (
            float(keying["main_keyway_width_m"]),
            *([float(keying["minor_keyway_width_m"])] * 4),
        )
        authored_wall_segments = 0
        for index in range(segment_count):
            angle = index * segment_angle_deg
            blocked_by_keyway = any(
                abs(_wrapped_error_deg(angle, key_angle))
                <= math.degrees(math.atan2(0.5 * width + extra, bore_radius))
                + 0.5 * segment_angle_deg
                for key_angle, width in zip(assembly_key_angles, keyway_widths)
            )
            if blocked_by_keyway:
                continue
            theta = math.radians(angle)
            add_cube(
                f"/World/TE_J35Assembly/FixedReceptacle/Collision/Wall_{index:03d}",
                (
                    wall_center_radius * math.cos(theta),
                    wall_center_radius * math.sin(theta),
                    -0.5 * engagement_length,
                ),
                (wall_radial_thickness, segment_tangent, engagement_length),
                rotate_z_deg=angle,
            )
            authored_wall_segments += 1

        fixed_stop_z = -float(thread["axial_travel_m"]) - 0.00025
        add_cylinder(
            "/World/TE_J35Assembly/FixedReceptacle/Collision/ContactFaceStop",
            fixed_stop_z,
            float(collision["contact_face_stop_radius_m"]),
            float(collision["contact_face_stop_thickness_m"]),
        )

        loose = UsdGeom.Xform.Define(stage, "/World/TE_J35Assembly/LoosePlug")
        body = UsdGeom.Xform.Define(
            stage, "/World/TE_J35Assembly/LoosePlug/BodyAssembly"
        )
        body.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, initial_gap))
        body_rigid = UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        body_rigid.CreateRigidBodyEnabledAttr(True)
        body_mass = UsdPhysics.MassAPI.Apply(body.GetPrim())
        body_mass.CreateMassAttr(float(mass["plug_body_kg"]))
        PhysxSchema.PhysxRigidBodyAPI.Apply(body.GetPrim()).CreateEnableCCDAttr(True)
        add_reference(
            "/World/TE_J35Assembly/LoosePlug/BodyAssembly/Visual",
            "plug_body",
            flip_plug=True,
        )
        UsdGeom.Scope.Define(
            stage, "/World/TE_J35Assembly/LoosePlug/BodyAssembly/Collision"
        )
        add_cylinder(
            "/World/TE_J35Assembly/LoosePlug/BodyAssembly/Collision/GuideShell",
            0.5 * engagement_length,
            float(keying["plug_shell_radius_m"]),
            engagement_length,
        )
        add_cylinder(
            "/World/TE_J35Assembly/LoosePlug/BodyAssembly/Collision/ContactFaceStop",
            0.00025,
            float(collision["contact_face_stop_radius_m"]) - 0.0001,
            float(collision["contact_face_stop_thickness_m"]),
        )
        key_widths = (
            float(keying["main_key_width_m"]),
            *([float(keying["minor_key_width_m"])] * 4),
        )
        shell_radius = float(keying["plug_shell_radius_m"])
        key_outer_radius = float(keying["plug_key_outer_radius_m"])
        key_center_radius = 0.5 * (shell_radius + key_outer_radius)
        for index, (angle, width) in enumerate(zip(assembly_key_angles, key_widths)):
            theta = math.radians(angle)
            add_cube(
                f"/World/TE_J35Assembly/LoosePlug/BodyAssembly/Collision/Key_{index}",
                (
                    key_center_radius * math.cos(theta),
                    key_center_radius * math.sin(theta),
                    0.5 * engagement_length,
                ),
                (key_outer_radius - shell_radius, width, engagement_length),
                rotate_z_deg=angle,
            )

        nut = UsdGeom.Xform.Define(stage, "/World/TE_J35Assembly/LoosePlug/CouplingNut")
        nut.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, initial_gap))
        nut_rigid = UsdPhysics.RigidBodyAPI.Apply(nut.GetPrim())
        nut_rigid.CreateRigidBodyEnabledAttr(True)
        UsdPhysics.MassAPI.Apply(nut.GetPrim()).CreateMassAttr(
            float(mass["coupling_nut_kg"])
        )
        PhysxSchema.PhysxRigidBodyAPI.Apply(nut.GetPrim()).CreateEnableCCDAttr(True)
        add_reference(
            "/World/TE_J35Assembly/LoosePlug/CouplingNut/Visual",
            "coupling_nut",
            flip_plug=True,
        )
        UsdGeom.Scope.Define(
            stage, "/World/TE_J35Assembly/LoosePlug/CouplingNut/Collision"
        )
        nut_ring_radius = 0.0222
        nut_ring_radial_thickness = 0.0020
        nut_height = 0.0202
        nut_center_z = 0.01165
        for index in range(24):
            angle = index * 15.0
            theta = math.radians(angle)
            add_cube(
                f"/World/TE_J35Assembly/LoosePlug/CouplingNut/Collision/Outer_{index:02d}",
                (
                    nut_ring_radius * math.cos(theta),
                    nut_ring_radius * math.sin(theta),
                    nut_center_z,
                ),
                (nut_ring_radial_thickness, 0.0052, nut_height),
                rotate_z_deg=angle,
            )

        hinge_path = "/World/TE_J35Assembly/Joints/CouplingNutRevolute"
        hinge = UsdPhysics.RevoluteJoint.Define(stage, hinge_path)
        hinge.CreateAxisAttr(UsdGeom.Tokens.z)
        hinge.CreateBody0Rel().SetTargets([body.GetPrim().GetPath()])
        hinge.CreateBody1Rel().SetTargets([nut.GetPrim().GetPath()])
        hinge.CreateLocalPos0Attr(Gf.Vec3f(0.0))
        hinge.CreateLocalPos1Attr(Gf.Vec3f(0.0))
        hinge.CreateLocalRot0Attr(Gf.Quatf(1.0))
        hinge.CreateLocalRot1Attr(Gf.Quatf(1.0))
        hinge.CreateCollisionEnabledAttr(False)

        prismatic_path = "/World/TE_J35Assembly/Joints/InsertionPrismatic"
        prismatic = UsdPhysics.PrismaticJoint.Define(stage, prismatic_path)
        prismatic.CreateAxisAttr(UsdGeom.Tokens.z)
        prismatic.CreateBody0Rel().SetTargets([fixed.GetPrim().GetPath()])
        prismatic.CreateBody1Rel().SetTargets([body.GetPrim().GetPath()])
        prismatic.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, initial_gap))
        prismatic.CreateLocalPos1Attr(Gf.Vec3f(0.0))
        prismatic.CreateLocalRot0Attr(Gf.Quatf(1.0))
        prismatic.CreateLocalRot1Attr(Gf.Quatf(1.0))
        prismatic.CreateLowerLimitAttr(-0.013)
        prismatic.CreateUpperLimitAttr(0.001)
        prismatic.CreateCollisionEnabledAttr(False)

        rack = PhysxSchema.PhysxPhysicsRackAndPinionJoint.Define(
            stage, "/World/TE_J35Assembly/Joints/ThreadCoupling"
        )
        rack.CreateBody0Rel().SetTargets([nut.GetPrim().GetPath()])
        rack.CreateBody1Rel().SetTargets([body.GetPrim().GetPath()])
        rack.CreateHingeRel().SetTargets([hinge.GetPrim().GetPath()])
        rack.CreatePrismaticRel().SetTargets([prismatic.GetPrim().GetPath()])
        rack.CreateRatioAttr(
            int(thread["direction"])
            * 360.0
            / float(thread["lead_m_per_revolution"])
        )
        rack.CreateJointEnabledAttr(False)

        assembly_prim.CreateAttribute(
            "kcg:keyCount", Sdf.ValueTypeNames.Int, custom=True
        ).Set(5)
        assembly_prim.CreateAttribute(
            "kcg:threadStarts", Sdf.ValueTypeNames.Int, custom=True
        ).Set(int(thread["starts"]))
        assembly_prim.CreateAttribute(
            "kcg:threadLeadMPerRevolution", Sdf.ValueTypeNames.Double, custom=True
        ).Set(float(thread["lead_m_per_revolution"]))
        assembly_prim.CreateAttribute(
            "kcg:threadDirection", Sdf.ValueTypeNames.Int, custom=True
        ).Set(int(thread["direction"]))
        assembly_prim.CreateAttribute(
            "kcg:authoredReceptacleWallSegments", Sdf.ValueTypeNames.Int, custom=True
        ).Set(authored_wall_segments)
        assembly_prim.CreateAttribute(
            "kcg:threadRepresentation", Sdf.ValueTypeNames.String, custom=True
        ).Set("visual_supplier_thread_plus_physx_rotation_translation_joint")
        stage.GetRootLayer().Save()
        print(arguments.output)
        print(f"receptacle wall segments: {authored_wall_segments}")
        return 0
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
