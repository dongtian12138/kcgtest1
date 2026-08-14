#!/usr/bin/env python3

"""Create a simplified plug/receptacle USD asset from the task YAML."""

import argparse
import math
from pathlib import Path

import yaml


def _read_geometry(config_path):
    with config_path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    geometry = document["geometry"]
    required = {
        "coupling_nut_radius",
        "coupling_nut_length",
        "plug_nose_radius",
        "plug_nose_length",
        "receptacle_entry_radius",
        "receptacle_body_radius",
    }
    missing = sorted(required - geometry.keys())
    if missing:
        raise ValueError(f"connector geometry is missing: {', '.join(missing)}")
    values = {name: float(geometry[name]) for name in required}
    if not all(math.isfinite(value) and value > 0.0 for value in values.values()):
        raise ValueError("all connector dimensions must be finite and positive")
    if values["plug_nose_radius"] >= values["receptacle_entry_radius"]:
        raise ValueError("plug nose must be smaller than receptacle entry")
    if values["coupling_nut_radius"] <= values["plug_nose_radius"] + 0.002:
        raise ValueError("coupling nut must leave a physical bore around the plug")
    if values["receptacle_entry_radius"] >= values["receptacle_body_radius"]:
        raise ValueError("receptacle entry must be smaller than its body")
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()

    config_path = Path(arguments.config).expanduser().resolve()
    output_path = Path(arguments.output).expanduser().resolve()
    geometry = _read_geometry(config_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
        from pxr import Gf, UsdGeom, UsdPhysics

        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        world = UsdGeom.Xform.Define(stage, "/World")
        def cylinder(path, radius, height, translation, color, collision=True):
            shape = UsdGeom.Cylinder.Define(stage, path)
            shape.CreateAxisAttr(UsdGeom.Tokens.z)
            shape.CreateRadiusAttr(radius)
            shape.CreateHeightAttr(height)
            shape.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            UsdGeom.Xformable(shape).AddTranslateOp().Set(Gf.Vec3d(*translation))
            if collision:
                UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
            return shape

        UsdGeom.Xform.Define(stage, "/World/Receptacle")
        cylinder(
            "/World/Receptacle/FlangeVisual",
            geometry["receptacle_body_radius"] + 0.012,
            0.008,
            (0.0, 0.0, 0.0),
            (0.20, 0.25, 0.32),
            collision=False,
        )
        cylinder(
            "/World/Receptacle/BoreBottom",
            geometry["receptacle_entry_radius"],
            0.006,
            (0.0, 0.0, -0.033),
            (0.16, 0.18, 0.22),
        )

        segment_count = 12
        inner_radius = geometry["receptacle_entry_radius"]
        outer_radius = geometry["receptacle_body_radius"]
        segment_radius = 0.5 * (inner_radius + outer_radius)
        radial_size = outer_radius - inner_radius
        tangential_size = 0.90 * 2.0 * math.pi * segment_radius / segment_count
        for index in range(segment_count):
            angle = 2.0 * math.pi * index / segment_count
            path = f"/World/Receptacle/CollisionSegment_{index:02d}"
            segment = UsdGeom.Cube.Define(stage, path)
            segment.CreateSizeAttr(1.0)
            segment.CreateDisplayColorAttr([Gf.Vec3f(0.20, 0.25, 0.32)])
            transform = UsdGeom.Xformable(segment)
            transform.AddTranslateOp().Set(
                Gf.Vec3d(
                    segment_radius * math.cos(angle),
                    segment_radius * math.sin(angle),
                    -0.010,
                )
            )
            transform.AddRotateZOp().Set(math.degrees(angle))
            transform.AddScaleOp().Set(
                Gf.Vec3f(radial_size, tangential_size, 0.050)
            )
            UsdPhysics.CollisionAPI.Apply(segment.GetPrim())

        plug = UsdGeom.Xform.Define(stage, "/World/Plug")
        UsdGeom.Xformable(plug).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.095))

        body = UsdGeom.Xform.Define(stage, "/World/Plug/BodyAssembly")
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        body_mass = UsdPhysics.MassAPI.Apply(body.GetPrim())
        body_mass.CreateMassAttr(0.06)
        cylinder(
            "/World/Plug/BodyAssembly/Body",
            0.85 * geometry["plug_nose_radius"],
            geometry["coupling_nut_length"] + 0.020,
            (0.0, 0.0, -0.015),
            (0.72, 0.22, 0.06),
        )
        cylinder(
            "/World/Plug/BodyAssembly/MatingInsert",
            geometry["plug_nose_radius"],
            geometry["plug_nose_length"],
            (
                0.0,
                0.0,
                -0.5
                * (
                    geometry["coupling_nut_length"]
                    + geometry["plug_nose_length"]
                ),
            ),
            (0.82, 0.34, 0.10),
        )

        # The nut is an independent rigid body.  A ring of simple boxes gives
        # it a real bore, so it does not interpenetrate the plug body.
        nut = UsdGeom.Xform.Define(stage, "/World/Plug/CouplingNut")
        UsdPhysics.RigidBodyAPI.Apply(nut.GetPrim())
        nut_mass = UsdPhysics.MassAPI.Apply(nut.GetPrim())
        nut_mass.CreateMassAttr(0.04)
        nut_inner_radius = geometry["plug_nose_radius"] + 0.002
        nut_outer_radius = geometry["coupling_nut_radius"]
        nut_segment_count = 16
        nut_segment_radius = 0.5 * (nut_inner_radius + nut_outer_radius)
        nut_radial_size = nut_outer_radius - nut_inner_radius
        nut_tangential_size = (
            0.90
            * 2.0
            * math.pi
            * nut_segment_radius
            / nut_segment_count
        )
        for index in range(nut_segment_count):
            angle = 2.0 * math.pi * index / nut_segment_count
            path = f"/World/Plug/CouplingNut/Segment_{index:02d}"
            segment = UsdGeom.Cube.Define(stage, path)
            segment.CreateSizeAttr(1.0)
            segment.CreateDisplayColorAttr([Gf.Vec3f(0.78, 0.28, 0.08)])
            transform = UsdGeom.Xformable(segment)
            transform.AddTranslateOp().Set(
                Gf.Vec3d(
                    nut_segment_radius * math.cos(angle),
                    nut_segment_radius * math.sin(angle),
                    0.0,
                )
            )
            transform.AddRotateZOp().Set(math.degrees(angle))
            transform.AddScaleOp().Set(
                Gf.Vec3f(
                    nut_radial_size,
                    nut_tangential_size,
                    geometry["coupling_nut_length"],
                )
            )
            UsdPhysics.CollisionAPI.Apply(segment.GetPrim())

        joint = UsdPhysics.RevoluteJoint.Define(
            stage, "/World/Plug/CouplingNutJoint"
        )
        joint.CreateAxisAttr("Z")
        joint.CreateBody0Rel().SetTargets([body.GetPrim().GetPath()])
        joint.CreateBody1Rel().SetTargets([nut.GetPrim().GetPath()])
        joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot0Attr(Gf.Quatf(1.0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
        joint.CreateCollisionEnabledAttr(False)

        stage.SetDefaultPrim(world.GetPrim())
        stage.GetRootLayer().Export(str(output_path))
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise RuntimeError("connector USD export failed")
        print(f"ISAAC CONNECTOR USD EXPORTED: {output_path}")
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
