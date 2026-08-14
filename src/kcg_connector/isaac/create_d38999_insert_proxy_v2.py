#!/usr/bin/env python3

"""Generate the independent D38999 insertion/contact proxy V2 USD asset."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import traceback

from kcg_connector.d38999_insert_proxy_v2 import (
    DEFAULT_CONFIG_PATH,
    RECOMMENDED_ASSET_NAME,
    load_insert_proxy_v2,
    safe_new_output,
)


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument(
        "--output",
        required=True,
        help=f"new asset path; recommended basename {RECOMMENDED_ASSET_NAME}",
    )
    return parser.parse_args()


def main():
    arguments = _arguments()
    config = load_insert_proxy_v2(arguments.config)
    output_path = safe_new_output(arguments.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {"headless": True, "multi_gpu": False, "active_gpu": 0, "physics_gpu": 0}
    )
    passed = False
    try:
        import omni.usd
        from omni.physx.scripts import physicsUtils as physics_utils
        from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade

        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        world = UsdGeom.Xform.Define(stage, "/World")
        root_path = "/World/D38999InsertProxyV2"
        root = UsdGeom.Xform.Define(stage, root_path)
        root_prim = root.GetPrim()
        for key, value in {
            "kcg:proxyId": "d38999_insert_proxy_v2",
            "kcg:sourceProxyId": "d38999_shell25j_61_pair_proxy_v1",
            "kcg:fidelity": "insertion_contact_and_c2_guidance_proxy",
            "kcg:assemblyPlusZ": "plug_insertion_direction_into_receptacle",
            "kcg:symmetry": "C2",
            "kcg:threadLabel": "PROXY THREAD",
            "kcg:lockLabel": "PROXY LOCK",
            "kcg:certificationClaim": "none",
            "kcg:configSha256": config.sha256,
        }.items():
            root_prim.SetCustomDataByKey(key, value)

        material_path = root_path + "/Materials/CompliantMetal"
        material = UsdShade.Material.Define(stage, material_path)
        material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        material_api.CreateStaticFrictionAttr(config.physics.static_friction)
        material_api.CreateDynamicFrictionAttr(config.physics.dynamic_friction)
        material_api.CreateRestitutionAttr(config.physics.restitution)
        physx_material = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
        physx_material.CreateCompliantContactStiffnessAttr().Set(
            config.physics.compliant_contact_stiffness_n_m
        )
        physx_material.CreateCompliantContactDampingAttr().Set(
            config.physics.compliant_contact_damping_n_s_m
        )

        collision_prims = []

        def _collision(prim):
            UsdPhysics.CollisionAPI.Apply(prim)
            physx = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            physx.CreateContactOffsetAttr().Set(config.physics.contact_offset_m)
            physx.CreateRestOffsetAttr().Set(config.physics.rest_offset_m)
            collision_prims.append(prim)

        def cylinder(path, radius, height, center_z, color, *, collision=True):
            shape = UsdGeom.Cylinder.Define(stage, path)
            shape.CreateAxisAttr(UsdGeom.Tokens.z)
            shape.CreateRadiusAttr(radius)
            shape.CreateHeightAttr(height)
            shape.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            UsdGeom.Xformable(shape).AddTranslateOp().Set(
                Gf.Vec3d(0.0, 0.0, center_z)
            )
            if collision:
                _collision(shape.GetPrim())
            return shape

        def cube(path, size, translation, color, *, rotation_z=0.0, collision=True):
            shape = UsdGeom.Cube.Define(stage, path)
            shape.CreateSizeAttr(1.0)
            shape.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            xform = UsdGeom.Xformable(shape)
            xform.AddTranslateOp().Set(Gf.Vec3d(*translation))
            if abs(rotation_z) > 1.0e-12:
                xform.AddRotateZOp().Set(rotation_z)
            xform.AddScaleOp().Set(Gf.Vec3f(*size))
            if collision:
                _collision(shape.GetPrim())
            return shape

        def ring_segments(
            parent,
            inner_radius,
            outer_radius,
            height,
            center_z,
            count,
            color,
            *,
            channels=False,
        ):
            radius = 0.5 * (inner_radius + outer_radius)
            radial = outer_radius - inner_radius
            tangential = 0.86 * 2.0 * math.pi * radius / count
            authored = 0
            for index in range(count):
                angle = 2.0 * math.pi * index / count
                wrapped = min(abs(angle), abs(angle - math.pi), abs(angle - 2.0 * math.pi))
                if channels and wrapped <= config.receptacle.c2_channel_half_width_rad:
                    continue
                cube(
                    f"{parent}/Segment_{index:02d}",
                    (radial, tangential, height),
                    (radius * math.cos(angle), radius * math.sin(angle), center_z),
                    color,
                    rotation_z=math.degrees(angle),
                )
                authored += 1
            return authored

        receptacle_path = root_path + "/Receptacle"
        receptacle = UsdGeom.Xform.Define(stage, receptacle_path)
        receptacle.GetPrim().SetCustomDataByKey("kcg:role", "fixed_receptacle")
        receptacle.GetPrim().SetCustomDataByKey("kcg:matingFrame", "mouth_center")
        r = config.receptacle
        # Four collision slices explicitly approximate the entry chamfer.
        chamfer_slices = 4
        for slice_index in range(chamfer_slices):
            fraction = (slice_index + 0.5) / chamfer_slices
            bore = r.mouth_bore_radius + fraction * (
                r.guide_bore_radius - r.mouth_bore_radius
            )
            height = r.entrance_chamfer_length / chamfer_slices
            ring_segments(
                f"{receptacle_path}/EntranceChamfer/Slice_{slice_index:02d}",
                bore,
                r.shell_outer_radius,
                height,
                (slice_index + 0.5) * height,
                r.collision_segment_count,
                (0.18, 0.42, 0.70),
                channels=True,
            )
        ring_segments(
            receptacle_path + "/GuideBore",
            r.guide_bore_radius,
            r.shell_outer_radius,
            r.guide_length - r.entrance_chamfer_length,
            0.5 * (r.guide_length + r.entrance_chamfer_length),
            r.collision_segment_count,
            (0.16, 0.34, 0.58),
            channels=True,
        )
        cylinder(
            receptacle_path + "/RearBody",
            r.rear_body_radius,
            r.rear_body_length,
            r.guide_length + 0.5 * r.rear_body_length,
            (0.14, 0.24, 0.38),
        )
        cube(
            receptacle_path + "/Flange",
            (r.flange_side, r.flange_side, r.flange_thickness),
            (0.0, 0.0, r.guide_length + 0.5 * r.flange_thickness),
            (0.20, 0.28, 0.38),
            collision=False,
        )

        plug_path = root_path + "/Plug"
        plug = UsdGeom.Xform.Define(stage, plug_path)
        plug.GetPrim().SetCustomDataByKey("kcg:role", "loose_plug")
        body_path = plug_path + "/Body"
        body = UsdGeom.Xform.Define(stage, body_path)
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        mass = UsdPhysics.MassAPI.Apply(body.GetPrim())
        mass.CreateMassAttr(config.physics.plug_body_mass_kg)
        mass.CreateDiagonalInertiaAttr(
            Gf.Vec3f(*config.physics.plug_body_diagonal_inertia_kg_m2)
        )
        physx_body = PhysxSchema.PhysxRigidBodyAPI.Apply(body.GetPrim())
        physx_body.CreateSolverPositionIterationCountAttr().Set(
            config.physics.solver_position_iterations
        )
        physx_body.CreateSolverVelocityIterationCountAttr().Set(
            config.physics.solver_velocity_iterations
        )
        p = config.plug
        # The plug mating-frame origin is the leading face at local z=0.
        # All plug geometry lies behind that face (negative z), while the
        # commanded insertion direction is +Z.
        # Plug leading chamfer: four stepped annular slices from guide to nose.
        for slice_index in range(chamfer_slices):
            fraction = (slice_index + 0.5) / chamfer_slices
            outer = p.guide_outer_radius - fraction * (
                p.guide_outer_radius - p.nose_radius
            )
            height = p.nose_chamfer_length / chamfer_slices
            ring_segments(
                f"{body_path}/NoseChamfer/Slice_{slice_index:02d}",
                p.guide_inner_radius,
                outer,
                height,
                -p.nose_chamfer_length + (slice_index + 0.5) * height,
                48,
                (0.68, 0.45, 0.16),
            )
        ring_segments(
            body_path + "/GuideShell",
            p.guide_inner_radius,
            p.guide_outer_radius,
            p.guide_length - p.nose_chamfer_length,
            -0.5 * (p.guide_length + p.nose_chamfer_length),
            48,
            (0.70, 0.50, 0.20),
        )
        key_radius = 0.5 * (p.guide_outer_radius + p.c2_key_outer_radius)
        key_radial = p.c2_key_outer_radius - p.guide_outer_radius
        key_center_z = -p.c2_key_start_from_tip - 0.5 * p.c2_key_length
        for index, angle in enumerate((0.0, math.pi)):
            cube(
                f"{body_path}/C2Keys/Key_{index}",
                (key_radial, p.c2_key_tangential_width, p.c2_key_length),
                (key_radius * math.cos(angle), key_radius * math.sin(angle), key_center_z),
                (0.92, 0.30, 0.12),
                rotation_z=math.degrees(angle),
            )
        cylinder(
            body_path + "/RearBody",
            p.body_radius,
            p.body_length,
            -p.guide_length - 0.5 * p.body_length,
            (0.38, 0.40, 0.43),
        )

        nut_path = plug_path + "/CouplingNut"
        nut = UsdGeom.Xform.Define(stage, nut_path)
        UsdPhysics.RigidBodyAPI.Apply(nut.GetPrim())
        nut_mass = UsdPhysics.MassAPI.Apply(nut.GetPrim())
        nut_mass.CreateMassAttr(config.physics.coupling_nut_mass_kg)
        nut_mass.CreateDiagonalInertiaAttr(
            Gf.Vec3f(*config.physics.coupling_nut_diagonal_inertia_kg_m2)
        )
        nut_center_z = -p.guide_length - 0.5 * p.coupling_nut_length
        ring_segments(
            nut_path,
            p.coupling_nut_inner_radius,
            p.coupling_nut_outer_radius,
            p.coupling_nut_length,
            nut_center_z,
            24,
            (0.56, 0.58, 0.62),
        )
        joint = UsdPhysics.RevoluteJoint.Define(stage, plug_path + "/CouplingNutJoint")
        joint.CreateAxisAttr("Z")
        joint.CreateBody0Rel().SetTargets([body.GetPrim().GetPath()])
        joint.CreateBody1Rel().SetTargets([nut.GetPrim().GetPath()])
        joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, nut_center_z))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, nut_center_z))
        joint.CreateLocalRot0Attr(Gf.Quatf(1.0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
        joint.CreateCollisionEnabledAttr(False)

        assembly = UsdGeom.Xform.Define(stage, root_path + "/connector_assembly_frame")
        assembly.GetPrim().SetCustomDataByKey("kcg:plusZ", "insertion")
        assembly.GetPrim().SetCustomDataByKey("kcg:plusX", "C2_key_channel")
        for prim in collision_prims:
            physics_utils.add_physics_material_to_prim(stage, prim, Sdf.Path(material_path))

        stage.SetDefaultPrim(world.GetPrim())
        stage.GetRootLayer().Export(str(output_path))
        if not output_path.is_file() or output_path.stat().st_size < 1000:
            raise RuntimeError("V2 asset export failed")
        print("D38999 INSERT CONTACT PROXY V2 EXPORTED")
        print(f"  output: {output_path}")
        print(f"  radial clearance: {config.radial_clearance * 1e3:.3f} mm")
        print("  symmetry: C2")
        print("  thread: PROXY THREAD")
        print("  lock: PROXY LOCK")
        passed = True
    except BaseException:
        traceback.print_exc()
        print("D38999 INSERT CONTACT PROXY V2 EXPORT FAILED", flush=True)
    finally:
        simulation_app.close(exit_code=0 if passed else 1)


if __name__ == "__main__":
    main()
