#!/usr/bin/env python3

"""Generate an independent shell-25/J D38999 proxy USD asset.

Only the command-line and pure configuration modules are imported at module
load time.  Isaac Sim and Pixar USD are imported after ``SimulationApp`` has
started, which keeps ordinary unit tests independent of the simulator.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import traceback

from kcg_connector.d38999_proxy import (
    DEFAULT_D38999_PROXY_CONFIG_PATH,
    RECOMMENDED_D38999_ASSET_NAME,
    load_d38999_shell25j_proxy,
    require_safe_d38999_output,
    verify_public_source_files,
)


def contact_positions(count=61):
    """Return a deterministic visual-only 1+6+12+18+24 contact layout."""
    if count != 61:
        raise ValueError("the v1 visual contact layout is defined for 61")
    positions = [(0.0, 0.0)]
    for ring_index, ring_count in enumerate((6, 12, 18, 24), start=1):
        radius_fraction = 0.20 * ring_index
        angular_offset = 0.5 * math.pi / ring_count
        for index in range(ring_count):
            angle = (
                2.0 * math.pi * index / ring_count + angular_offset
            )
            positions.append(
                (
                    radius_fraction * math.cos(angle),
                    radius_fraction * math.sin(angle),
                )
            )
    if len(positions) != count:
        raise RuntimeError("internal contact layout count mismatch")
    return tuple(positions)


def _arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a public-dimensional D38999 shell-25/J proxy USD"
        )
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_D38999_PROXY_CONFIG_PATH),
        help="versioned D38999 proxy YAML",
    )
    parser.add_argument(
        "--output",
        required=True,
        help=(
            "new .usd/.usda output; recommended basename: "
            + RECOMMENDED_D38999_ASSET_NAME
        ),
    )
    return parser.parse_args()


def main():
    arguments = _arguments()
    config_path = Path(arguments.config).expanduser().resolve()
    config = load_d38999_shell25j_proxy(config_path)
    package_root = Path(__file__).resolve().parents[1]
    verified_sources = verify_public_source_files(config, package_root)
    output_path = require_safe_d38999_output(arguments.output)
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

    passed = False
    try:
        import omni.usd
        from omni.physx.scripts import physicsUtils as physics_utils
        from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

        world = UsdGeom.Xform.Define(stage, "/World")
        root_path = "/World/D38999Shell25JProxy"
        root = UsdGeom.Xform.Define(stage, root_path)
        root_prim = root.GetPrim()
        root_prim.SetCustomDataByKey(
            "kcg:proxyId", config.identity.proxy_id
        )
        root_prim.SetCustomDataByKey(
            "kcg:fidelity", config.identity.fidelity
        )
        root_prim.SetCustomDataByKey(
            "kcg:loosePartNumber", config.identity.loose_part_number
        )
        root_prim.SetCustomDataByKey(
            "kcg:fixedPartNumber", config.identity.fixed_part_number
        )
        root_prim.SetCustomDataByKey(
            "kcg:certificationClaim", "none"
        )
        root_prim.SetCustomDataByKey(
            "kcg:threadCollisionMode", "none"
        )

        metal_material_path = root_path + "/Materials/MetalPhysics"
        metal_material = UsdShade.Material.Define(
            stage, metal_material_path
        )
        material_api = UsdPhysics.MaterialAPI.Apply(
            metal_material.GetPrim()
        )
        material_api.CreateStaticFrictionAttr(
            config.physics.static_friction
        )
        material_api.CreateDynamicFrictionAttr(
            config.physics.dynamic_friction
        )
        material_api.CreateRestitutionAttr(config.physics.restitution)

        collision_prims = []

        def cylinder(
            path,
            radius,
            height,
            translation,
            color,
            *,
            collision=True,
        ):
            shape = UsdGeom.Cylinder.Define(stage, path)
            shape.CreateAxisAttr(UsdGeom.Tokens.z)
            shape.CreateRadiusAttr(radius)
            shape.CreateHeightAttr(height)
            shape.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            UsdGeom.Xformable(shape).AddTranslateOp().Set(
                Gf.Vec3d(*translation)
            )
            if collision:
                UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
                collision_prims.append(shape.GetPrim())
            return shape

        def cube(
            path,
            size,
            translation,
            color,
            *,
            rotation_z=0.0,
            collision=True,
        ):
            shape = UsdGeom.Cube.Define(stage, path)
            shape.CreateSizeAttr(1.0)
            shape.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            transform = UsdGeom.Xformable(shape)
            transform.AddTranslateOp().Set(Gf.Vec3d(*translation))
            if abs(rotation_z) > 1.0e-12:
                transform.AddRotateZOp().Set(rotation_z)
            transform.AddScaleOp().Set(Gf.Vec3f(*size))
            if collision:
                UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
                collision_prims.append(shape.GetPrim())
            return shape

        def ring_segments(
            parent_path,
            inner_radius,
            outer_radius,
            height,
            center_z,
            count,
            color,
            *,
            collision=True,
        ):
            segment_radius = 0.5 * (inner_radius + outer_radius)
            radial_size = outer_radius - inner_radius
            tangential_size = (
                0.91 * 2.0 * math.pi * segment_radius / count
            )
            for index in range(count):
                angle = 2.0 * math.pi * index / count
                cube(
                    f"{parent_path}/Segment_{index:02d}",
                    (radial_size, tangential_size, height),
                    (
                        segment_radius * math.cos(angle),
                        segment_radius * math.sin(angle),
                        center_z,
                    ),
                    color,
                    rotation_z=math.degrees(angle),
                    collision=collision,
                )

        receptacle_path = root_path + "/FixedReceptacle"
        UsdGeom.Xform.Define(stage, receptacle_path)
        receptacle = config.receptacle_geometry_m
        flange_center_z = -0.5 * receptacle.flange_thickness
        cube(
            receptacle_path + "/FlangeVisual",
            (
                receptacle.flange_side,
                receptacle.flange_side,
                receptacle.flange_thickness,
            ),
            (0.0, 0.0, flange_center_z),
            (0.27, 0.30, 0.33),
            collision=False,
        )

        # Four small corner pads provide conservative static flange collision
        # without falsely filling the central mating bore.
        pad_size = 0.0070
        pad_offset = 0.5 * (receptacle.flange_side - pad_size)
        for index, (x_sign, y_sign) in enumerate(
            ((-1, -1), (-1, 1), (1, -1), (1, 1))
        ):
            cube(
                f"{receptacle_path}/FlangeCollision_{index}",
                (pad_size, pad_size, receptacle.flange_thickness),
                (
                    x_sign * pad_offset,
                    y_sign * pad_offset,
                    flange_center_z,
                ),
                (0.27, 0.30, 0.33),
            )

        ring_segments(
            receptacle_path + "/EntryShell",
            receptacle.entry_radius,
            receptacle.shell_outer_radius,
            receptacle.front_shell_length,
            0.5 * receptacle.front_shell_length,
            20,
            (0.24, 0.28, 0.32),
        )
        cylinder(
            receptacle_path + "/RearBody",
            receptacle.rear_body_radius,
            receptacle.rear_body_length,
            (0.0, 0.0, -0.5 * receptacle.rear_body_length),
            (0.20, 0.23, 0.27),
        )
        cylinder(
            receptacle_path + "/ContactFace",
            receptacle.contact_face_radius,
            0.0010,
            (0.0, 0.0, 0.0005),
            (0.84, 0.80, 0.69),
            collision=False,
        )

        for index, (x_sign, y_sign) in enumerate(
            ((-1, -1), (-1, 1), (1, -1), (1, 1))
        ):
            rotation = -45.0 if x_sign == y_sign else 45.0
            cube(
                f"{receptacle_path}/MountingSlotVisual_{index}",
                (
                    receptacle.mounting_slot_length,
                    receptacle.mounting_slot_width,
                    receptacle.flange_thickness + 0.0001,
                ),
                (
                    x_sign * receptacle.mounting_slot_center_offset,
                    y_sign * receptacle.mounting_slot_center_offset,
                    flange_center_z,
                ),
                (0.06, 0.07, 0.08),
                rotation_z=rotation,
                collision=False,
            )

        contact_layout = contact_positions(receptacle.contact_count)
        contact_scale = 0.90 * receptacle.contact_face_radius
        for index, (x_fraction, y_fraction) in enumerate(contact_layout):
            cylinder(
                f"{receptacle_path}/Pins/Pin_{index:02d}",
                receptacle.pin_visual_radius,
                receptacle.pin_visual_length,
                (
                    contact_scale * x_fraction,
                    contact_scale * y_fraction,
                    0.0010 + 0.5 * receptacle.pin_visual_length,
                ),
                (0.88, 0.66, 0.18),
                collision=False,
            )

        plug_path = root_path + "/LoosePlug"
        plug_root = UsdGeom.Xform.Define(stage, plug_path)
        UsdGeom.Xformable(plug_root).AddTranslateOp().Set(
            Gf.Vec3d(0.0, 0.0, config.plug_geometry_m.initial_pair_separation)
        )
        plug = config.plug_geometry_m

        body_path = plug_path + "/BodyAssembly"
        body = UsdGeom.Xform.Define(stage, body_path)
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(
            config.physics.plug_body_mass_kg
        )
        cylinder(
            body_path + "/RearBody",
            plug.rear_body_radius,
            plug.rear_body_length,
            (
                0.0,
                0.0,
                plug.overall_length - 0.5 * plug.rear_body_length,
            ),
            (0.30, 0.33, 0.36),
        )
        ring_segments(
            body_path + "/MatingShell",
            plug.mating_shell_inner_radius,
            plug.mating_shell_outer_radius,
            plug.mating_shell_length,
            0.5 * plug.mating_shell_length,
            20,
            (0.32, 0.35, 0.38),
        )
        cylinder(
            body_path + "/SocketFace",
            plug.contact_face_radius,
            0.0010,
            (0.0, 0.0, 0.0005),
            (0.77, 0.74, 0.65),
            collision=False,
        )
        plug_contact_scale = 0.90 * plug.contact_face_radius
        for index, (x_fraction, y_fraction) in enumerate(contact_layout):
            cylinder(
                f"{body_path}/Sockets/Socket_{index:02d}",
                plug.contact_visual_radius,
                plug.contact_visual_depth,
                (
                    plug_contact_scale * x_fraction,
                    plug_contact_scale * y_fraction,
                    0.0004,
                ),
                (0.07, 0.08, 0.09),
                collision=False,
            )

        nut_path = plug_path + "/CouplingNut"
        nut = UsdGeom.Xform.Define(stage, nut_path)
        UsdPhysics.RigidBodyAPI.Apply(nut.GetPrim())
        UsdPhysics.MassAPI.Apply(nut.GetPrim()).CreateMassAttr(
            config.physics.coupling_nut_mass_kg
        )
        nut_center_z = 0.5 * plug.overall_length
        ring_segments(
            nut_path,
            plug.coupling_nut_inner_radius,
            plug.coupling_nut_outer_radius,
            plug.coupling_nut_length,
            nut_center_z,
            plug.grip_segment_count,
            (0.54, 0.57, 0.60),
        )

        joint = UsdPhysics.RevoluteJoint.Define(
            stage, plug_path + "/CouplingNutJoint"
        )
        joint.CreateAxisAttr("Z")
        joint.CreateBody0Rel().SetTargets([body.GetPrim().GetPath()])
        joint.CreateBody1Rel().SetTargets([nut.GetPrim().GetPath()])
        joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, nut_center_z))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, nut_center_z))
        joint.CreateLocalRot0Attr(Gf.Quatf(1.0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
        joint.CreateCollisionEnabledAttr(False)

        for prim in collision_prims:
            physics_utils.add_physics_material_to_prim(
                stage, prim, Sdf.Path(metal_material_path)
            )

        stage.SetDefaultPrim(world.GetPrim())
        stage.GetRootLayer().Export(str(output_path))
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise RuntimeError("D38999 proxy USD export failed")
        print("D38999 PUBLIC-DIMENSIONAL PROXY EXPORTED")
        print(f"  output: {output_path}")
        print(f"  loose: {config.identity.loose_part_number}")
        print(f"  fixed: {config.identity.fixed_part_number}")
        print(
            "  verified public sources: "
            + ", ".join(path.name for path in verified_sources)
        )
        print("  qualification/certification claim: none")
        print("  thread collision: none")
        passed = True
    except BaseException:
        traceback.print_exc()
        print("D38999 PUBLIC-DIMENSIONAL PROXY EXPORT FAILED", flush=True)
    finally:
        simulation_app.close(exit_code=0 if passed else 1)


if __name__ == "__main__":
    main()
