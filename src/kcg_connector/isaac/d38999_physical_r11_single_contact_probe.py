#!/usr/bin/env python3

"""Probe one realized r11 compliant contact without post-start pose writes.

This is an A3 modelling diagnostic, not an acceptance result.  It keeps the
real r11 colliders and their resolved PhysX material, disables unrelated
colliders, fixes the loose-plug bodies before reset, and reports raw contact
point impulses.  No file fingerprint is computed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import traceback
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "kcg_d38999_physical_r11_single_contact_probe_v1"
GENERATOR_ID = "kcg_d38999_physical_r11_realized_contact_probe_v1"

CASE_SPECS: Mapping[str, Mapping[str, Any]] = {
    "socket_petal_A0": {
        "separation_m": 0.01210,
        "left_family": "pins_61",
        "left_suffix": "/Contacts/Pin_A",
        "right_family": "socket_petals_366",
        "right_suffix": "/Contacts/Socket_A/Sleeve/Petal_0",
        "left_count": 1,
        "right_count": 1,
        "material_suffix": (
            "/Materials/pin_and_socket__compliant_socket_petal"
        ),
    },
    "spring_finger_00": {
        "separation_m": 0.01100,
        "left_family": "receptacle_bore_targets_12",
        "left_suffix": "/MatingShell/InnerBoreTarget/Target_00",
        "right_family": "spring_fingers_12",
        "right_suffix": "/SpringFingers/Finger_00",
        "left_count": 1,
        "right_count": 1,
        "material_suffix": (
            "/Materials/spring_finger__compliant_spring_finger"
        ),
    },
    "peripheral_seal_00": {
        "separation_m": 0.01480,
        "left_family": "seal_segments_24",
        "left_suffix": "/PeripheralSeal/Seg_00",
        "right_family": "seal_targets_24",
        "right_suffix": "/PeripheralSealTarget/Seg_00",
        "left_count": 1,
        "right_count": 1,
        "material_suffix": (
            "/Materials/peripheral_seal__compliant_peripheral_seal"
        ),
    },
    "synthetic_seal_sphere_00": {
        "separation_m": 0.01480,
        "left_family": "synthetic_seal_spheres",
        "left_suffix": "/PeripheralSealSphereProbe/Sphere_00",
        "right_family": "synthetic_seal_target",
        "right_suffix": "/PeripheralSealTargetProbe/ContinuousTarget",
        "left_count": 1,
        "right_count": 1,
        "material_suffix": (
            "/Materials/peripheral_seal__compliant_peripheral_seal"
        ),
        "synthetic_seal_sphere_count": 1,
        "hard_material_suffix": (
            "/Materials/plug_shell_and_keys__hard_seal_target"
        ),
    },
    "synthetic_seal_spheres_24": {
        "separation_m": 0.01505,
        "left_family": "synthetic_seal_spheres",
        "left_contains": "/PeripheralSealSphereProbe/Sphere_",
        "right_family": "synthetic_seal_target",
        "right_suffix": "/PeripheralSealTargetProbe/ContinuousTarget",
        "left_count": 24,
        "right_count": 1,
        "material_suffix": (
            "/Materials/peripheral_seal__compliant_peripheral_seal"
        ),
        "synthetic_seal_sphere_count": 24,
        "hard_material_suffix": (
            "/Materials/plug_shell_and_keys__hard_seal_target"
        ),
    },
    "synthetic_socket_spheres_A": {
        "separation_m": 0.01210,
        "left_family": "pins_61",
        "left_suffix": "/Contacts/Pin_A",
        "right_family": "synthetic_socket_spheres",
        "right_contains": "/Contacts/Socket_A/SyntheticSpherePetals/Petal_",
        "left_count": 1,
        "right_count": 6,
        "material_suffix": (
            "/Materials/pin_and_socket__compliant_socket_petal"
        ),
        "synthetic_socket_sphere_count": 6,
    },
    "synthetic_barrier_spheres_A": {
        "separation_m": 0.01505,
        "left_family": "synthetic_barrier_spheres",
        "left_contains": "/Contacts/Barrier_A/SyntheticSpheres/Sphere_",
        "right_family": "synthetic_barrier_target",
        "right_suffix": "/Contacts/Socket_A/SyntheticBarrierTarget",
        "left_count": 6,
        "right_count": 1,
        "material_suffix": (
            "/Materials/"
            "interfacial_pin_barrier__compliant_pin_barrier"
        ),
        "synthetic_barrier_sphere_count": 6,
        "hard_material_suffix": (
            "/Materials/pin_and_socket__hard_socket_entry"
        ),
    },
    "synthetic_spring_spheres_12": {
        "separation_m": 0.01505,
        "left_family": "synthetic_spring_target",
        "left_suffix": "/MatingShell/SyntheticSpringTarget",
        "right_family": "synthetic_spring_spheres",
        "right_contains": "/SpringFingers/SyntheticSpheres/Finger_",
        "left_count": 1,
        "right_count": 12,
        "material_suffix": (
            "/Materials/spring_finger__compliant_spring_finger"
        ),
        "synthetic_spring_sphere_count": 12,
        "hard_material_suffix": (
            "/Materials/fixture_and_receptacle__hard_receptacle_bore"
        ),
    },
    "metal_bottoming_existing_48x48": {
        "separation_m": 0.01505,
        "left_family": "fixed_metal_stop_48",
        "left_contains": "/MatingShell/MetalStop/Seg_",
        "right_family": "plug_metal_stop_48",
        "right_contains": "/InternalMatingShell/MetalStop/Seg_",
        "left_count": 48,
        "right_count": 48,
        "material_suffix": (
            "/Materials/peripheral_seal__compliant_peripheral_seal"
        ),
    },
    "synthetic_metal_bottoming_continuous": {
        "separation_m": 0.01505,
        "left_family": "synthetic_fixed_metal_stop",
        "left_suffix": "/MatingShell/SyntheticMetalStop",
        "right_family": "synthetic_plug_metal_stop_spheres",
        "right_contains": "/InternalMatingShell/SyntheticMetalStop/Sphere_",
        "left_count": 1,
        "right_count": 3,
        "material_suffix": (
            "/Materials/peripheral_seal__compliant_peripheral_seal"
        ),
        "synthetic_continuous_bottoming": True,
        "fixed_hard_material_suffix": (
            "/Materials/fixture_and_receptacle__hard_metal_bottoming"
        ),
        "plug_hard_material_suffix": (
            "/Materials/plug_shell_and_keys__hard_metal_bottoming"
        ),
    },
    "synthetic_shoulder_positive": {
        "separation_m": 0.01505,
        "nut_transz_m": 0.00005,
        "left_family": "synthetic_shoulder_positive_body0",
        "left_suffix": "/NutBearingShoulders/SyntheticPositiveStop/Cap",
        "right_family": "synthetic_shoulder_positive_body1_spheres",
        "right_contains": "/NutBearingShoulders/SyntheticPositiveStop/Sphere_",
        "left_count": 1,
        "right_count": 3,
        "material_suffix": (
            "/Materials/peripheral_seal__compliant_peripheral_seal"
        ),
        "synthetic_shoulder_polarity": "positive",
        "hard_material_suffix": (
            "/Materials/"
            "coupling_bearing_and_shoulder__hard_nut_body_shoulder"
        ),
    },
    "shoulder_positive_existing_48x48": {
        "separation_m": 0.01505,
        "nut_transz_m": 0.00005,
        "left_family": "shoulder_positive_body0_48",
        "left_contains": "/NutBearingShoulders/PositiveStop/Seg_",
        "right_family": "shoulder_positive_body1_48",
        "right_contains": "/NutBearingShoulders/PositiveStop/Seg_",
        "left_count": 48,
        "right_count": 48,
        "material_suffix": (
            "/Materials/peripheral_seal__compliant_peripheral_seal"
        ),
        "shoulder_probe_internal": True,
    },
    "synthetic_shoulder_negative": {
        "separation_m": 0.01505,
        "nut_transz_m": -0.00005,
        "left_family": "synthetic_shoulder_negative_body0",
        "left_suffix": "/NutBearingShoulders/SyntheticNegativeStop/Cap",
        "right_family": "synthetic_shoulder_negative_body1_spheres",
        "right_contains": "/NutBearingShoulders/SyntheticNegativeStop/Sphere_",
        "left_count": 1,
        "right_count": 3,
        "material_suffix": (
            "/Materials/peripheral_seal__compliant_peripheral_seal"
        ),
        "synthetic_shoulder_polarity": "negative",
        "hard_material_suffix": (
            "/Materials/"
            "coupling_bearing_and_shoulder__hard_nut_body_shoulder"
        ),
    },
    "pin_barrier_A": {
        "separation_m": 0.01450,
        "left_family": "pin_barriers_61",
        "left_contains": "/Contacts/Barrier_A/",
        "right_family": "hard_socket_entries_61",
        "right_contains": "/Contacts/Socket_A/HardEntry/",
        "left_count": 48,
        "right_count": 72,
        "material_suffix": (
            "/Materials/"
            "interfacial_pin_barrier__compliant_pin_barrier"
        ),
    },
}


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--case", choices=sorted(CASE_SPECS), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--scene-config",
        default=str(
            repository
            / "src/kcg_connector/config/"
            "d38999_keyed_v2_tabletop_scene_v1.yaml"
        ),
    )
    parser.add_argument("--separation-m", type=float)
    parser.add_argument("--nut-transz-m", type=float)
    parser.add_argument("--free-shoulder-nut", action="store_true")
    parser.add_argument("--use-original-shoulder-joint", action="store_true")
    parser.add_argument("--keep-shoulder-collision-groups", action="store_true")
    parser.add_argument("--shoulder-joint-limit-m", type=float)
    parser.add_argument("--stiffness-n-m", type=float)
    parser.add_argument("--damping-n-s-m", type=float)
    parser.add_argument("--settle-steps", type=int, default=240)
    result = parser.parse_args(argv)
    if not result.run:
        parser.error("the realized-contact probe requires --run")
    if result.settle_steps < 120 or result.settle_steps > 2000:
        parser.error("settle steps must be in [120, 2000]")
    for label, value in (
        ("separation", result.separation_m),
        ("nut transZ", result.nut_transz_m),
        ("stiffness", result.stiffness_n_m),
        ("damping", result.damping_n_s_m),
        ("shoulder joint limit", result.shoulder_joint_limit_m),
    ):
        if value is not None and not math.isfinite(value):
            parser.error(f"{label} override must be finite")
    if result.separation_m is not None and result.separation_m <= 0.0:
        parser.error("separation override must be positive")
    if result.stiffness_n_m is not None and result.stiffness_n_m <= 0.0:
        parser.error("stiffness override must be positive")
    if result.damping_n_s_m is not None and result.damping_n_s_m < 0.0:
        parser.error("damping override must be nonnegative")
    if (
        result.shoulder_joint_limit_m is not None
        and result.shoulder_joint_limit_m <= 0.0
    ):
        parser.error("shoulder joint limit must be positive")
    if (
        result.shoulder_joint_limit_m is not None
        and not result.use_original_shoulder_joint
    ):
        parser.error("shoulder joint limit requires the original shoulder joint")
    if result.free_shoulder_nut and result.use_original_shoulder_joint:
        parser.error(
            "free shoulder nut and original shoulder joint are mutually exclusive"
        )
    return result


def _emit(value: Any) -> None:
    os.write(1, (str(value) + "\n").encode("utf-8"))


def _family(prim: Any) -> str | None:
    attribute = prim.GetAttribute("kcg:primitiveFamily")
    value = attribute.Get() if attribute else None
    return None if value is None else str(value)


def _path_matches(path: str, spec: Mapping[str, Any], side: str) -> bool:
    suffix = spec.get(f"{side}_suffix")
    contains = spec.get(f"{side}_contains")
    return bool(
        (suffix is not None and path.endswith(str(suffix)))
        or (contains is not None and str(contains) in path)
    )


def _author_synthetic_seal_probe(
    *,
    stage: Any,
    connector_root: str,
    sphere_count: int,
    compliant_material: Any,
    hard_material: Any,
    Gf: Any,
    Sdf: Any,
    PhysxSchema: Any,
    UsdGeom: Any,
    UsdPhysics: Any,
    UsdShade: Any,
) -> None:
    """Author an in-memory, seam-free seal response candidate.

    The visual annulus is deliberately not reproduced here.  Each compliant
    sphere has one intended contact against one continuous hard target cap.
    The low sphere point remains at the r11 seal onset plane z=14.615 mm.
    """

    if sphere_count not in (1, 24):
        raise ValueError("synthetic seal probe supports exactly 1 or 24 spheres")

    def mark(
        prim: Any,
        *,
        family: str,
        material: Any,
        response_role: str,
    ) -> None:
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
        PhysxSchema.PhysxCollisionAPI.Apply(prim)
        prim.CreateAttribute(
            "physxCollision:contactOffset",
            Sdf.ValueTypeNames.Float,
            custom=False,
        ).Set(0.00001)
        prim.CreateAttribute(
            "physxCollision:restOffset",
            Sdf.ValueTypeNames.Float,
            custom=False,
        ).Set(0.0)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            UsdShade.Material(material), materialPurpose="physics"
        )
        prim.CreateAttribute(
            "kcg:primitiveFamily", Sdf.ValueTypeNames.String, custom=True
        ).Set(family)
        prim.CreateAttribute(
            "kcg:responseRole", Sdf.ValueTypeNames.String, custom=True
        ).Set(response_role)
        prim.CreateAttribute(
            "kcg:diagnosticOnly", Sdf.ValueTypeNames.Bool, custom=True
        ).Set(True)

    sphere_parent = (
        connector_root + "/FixedReceptacle/PeripheralSealSphereProbe"
    )
    UsdGeom.Xform.Define(stage, sphere_parent)
    sphere_radius = 0.00100
    sphere_ring_radius = 0.01575
    sphere_center_z = 0.014615 + sphere_radius
    for index in range(sphere_count):
        angle = math.radians(15.0 * index)
        sphere = UsdGeom.Sphere.Define(
            stage, sphere_parent + f"/Sphere_{index:02d}"
        )
        sphere.CreateRadiusAttr(sphere_radius)
        sphere.CreateExtentAttr(
            [
                Gf.Vec3f(-sphere_radius, -sphere_radius, -sphere_radius),
                Gf.Vec3f(sphere_radius, sphere_radius, sphere_radius),
            ]
        )
        UsdGeom.Xformable(sphere).AddTranslateOp().Set(
            Gf.Vec3d(
                sphere_ring_radius * math.cos(angle),
                sphere_ring_radius * math.sin(angle),
                sphere_center_z,
            )
        )
        mark(
            sphere.GetPrim(),
            family="synthetic_seal_spheres",
            material=compliant_material,
            response_role="compliant_peripheral_seal",
        )

    target_parent = connector_root + "/LoosePlug/BodyAssembly/PeripheralSealTargetProbe"
    UsdGeom.Xform.Define(stage, target_parent)
    target = UsdGeom.Cylinder.Define(
        stage, target_parent + "/ContinuousTarget"
    )
    target.CreateAxisAttr("Z")
    target.CreateRadiusAttr(0.01690)
    target.CreateHeightAttr(0.00100)
    target.CreateExtentAttr(
        [Gf.Vec3f(-0.01690, -0.01690, -0.00050),
         Gf.Vec3f(0.01690, 0.01690, 0.00050)]
    )
    UsdGeom.Xformable(target).AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, 0.00050)
    )
    mark(
        target.GetPrim(),
        family="synthetic_seal_target",
        material=hard_material,
        response_role="hard_seal_target",
    )


def _author_synthetic_socket_probe(
    *,
    stage: Any,
    connector_root: str,
    sphere_count: int,
    compliant_material: Any,
    Gf: Any,
    Sdf: Any,
    PhysxSchema: Any,
    UsdGeom: Any,
    UsdPhysics: Any,
    UsdShade: Any,
) -> None:
    """Author six one-point socket-petal candidates around realized Pin A."""

    if sphere_count != 6:
        raise ValueError("synthetic socket probe requires exactly six spheres")
    pin = stage.GetPrimAtPath(connector_root + "/FixedReceptacle/Contacts/Pin_A")
    if not pin:
        raise RuntimeError("missing realized Pin A")
    translate_ops = [
        operation
        for operation in UsdGeom.Xformable(pin).GetOrderedXformOps()
        if operation.GetOpType() == UsdGeom.XformOp.TypeTranslate
    ]
    if len(translate_ops) != 1:
        raise RuntimeError("Pin A must have exactly one translate operation")
    pin_center = translate_ops[0].Get()
    socket_center_x = float(pin_center[0])
    socket_center_y = -float(pin_center[1])

    sphere_radius = 0.000150
    declared_radial_overlap = 0.0000127
    center_ring_radius = 0.000508 + sphere_radius - declared_radial_overlap
    radial_gap_at_onset = sphere_radius - declared_radial_overlap
    axial_lead_in = math.sqrt(
        sphere_radius * sphere_radius
        - radial_gap_at_onset * radial_gap_at_onset
    )
    sphere_center_depth = 0.00200 + axial_lead_in
    parent = (
        connector_root
        + "/LoosePlug/BodyAssembly/Contacts/Socket_A/SyntheticSpherePetals"
    )
    UsdGeom.Xform.Define(stage, parent)
    for index in range(sphere_count):
        angle = math.radians(60.0 * index)
        sphere = UsdGeom.Sphere.Define(stage, parent + f"/Petal_{index}")
        sphere.CreateRadiusAttr(sphere_radius)
        sphere.CreateExtentAttr(
            [
                Gf.Vec3f(-sphere_radius, -sphere_radius, -sphere_radius),
                Gf.Vec3f(sphere_radius, sphere_radius, sphere_radius),
            ]
        )
        UsdGeom.Xformable(sphere).AddTranslateOp().Set(
            Gf.Vec3d(
                socket_center_x + center_ring_radius * math.cos(angle),
                socket_center_y + center_ring_radius * math.sin(angle),
                sphere_center_depth,
            )
        )
        prim = sphere.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
        collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        collision.CreateContactOffsetAttr(0.00001)
        collision.CreateRestOffsetAttr(0.0)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            UsdShade.Material(compliant_material), materialPurpose="physics"
        )
        prim.CreateAttribute(
            "kcg:primitiveFamily", Sdf.ValueTypeNames.String, custom=True
        ).Set("synthetic_socket_spheres")
        prim.CreateAttribute(
            "kcg:responseRole", Sdf.ValueTypeNames.String, custom=True
        ).Set("compliant_socket_petal")
        prim.CreateAttribute(
            "kcg:diagnosticOnly", Sdf.ValueTypeNames.Bool, custom=True
        ).Set(True)


def _author_synthetic_barrier_probe(
    *,
    stage: Any,
    connector_root: str,
    sphere_count: int,
    compliant_material: Any,
    hard_material: Any,
    Gf: Any,
    Sdf: Any,
    PhysxSchema: Any,
    UsdGeom: Any,
    UsdPhysics: Any,
    UsdShade: Any,
) -> None:
    """Author one six-point barrier response candidate around contact A."""

    if sphere_count != 6:
        raise ValueError("synthetic barrier probe requires exactly six spheres")
    pin = stage.GetPrimAtPath(connector_root + "/FixedReceptacle/Contacts/Pin_A")
    translate_ops = [
        operation
        for operation in UsdGeom.Xformable(pin).GetOrderedXformOps()
        if operation.GetOpType() == UsdGeom.XformOp.TypeTranslate
    ]
    if len(translate_ops) != 1:
        raise RuntimeError("Pin A must have exactly one translate operation")
    pin_center = translate_ops[0].Get()
    fixed_center_x = float(pin_center[0])
    fixed_center_y = float(pin_center[1])
    plug_center_x = fixed_center_x
    plug_center_y = -fixed_center_y

    sphere_radius = 0.000500
    full_radial_deflection = 0.000295
    target_radius = 0.000640
    sphere_ring_radius = target_radius + sphere_radius - full_radial_deflection
    target_front_depth = 0.000140
    first_touch_separation = 0.014305
    radial_gap_to_target_side = sphere_ring_radius - target_radius
    axial_lead_in = math.sqrt(
        sphere_radius * sphere_radius
        - radial_gap_to_target_side * radial_gap_to_target_side
    )
    sphere_center_z = (
        first_touch_separation - target_front_depth + axial_lead_in
    )

    sphere_parent = (
        connector_root
        + "/FixedReceptacle/Contacts/Barrier_A/SyntheticSpheres"
    )
    UsdGeom.Xform.Define(stage, sphere_parent)
    for index in range(sphere_count):
        angle = math.radians(60.0 * index)
        sphere = UsdGeom.Sphere.Define(
            stage, sphere_parent + f"/Sphere_{index}"
        )
        sphere.CreateRadiusAttr(sphere_radius)
        sphere.CreateExtentAttr(
            [
                Gf.Vec3f(-sphere_radius, -sphere_radius, -sphere_radius),
                Gf.Vec3f(sphere_radius, sphere_radius, sphere_radius),
            ]
        )
        UsdGeom.Xformable(sphere).AddTranslateOp().Set(
            Gf.Vec3d(
                fixed_center_x + sphere_ring_radius * math.cos(angle),
                fixed_center_y + sphere_ring_radius * math.sin(angle),
                sphere_center_z,
            )
        )
        prim = sphere.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
        collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        collision.CreateContactOffsetAttr(0.00001)
        collision.CreateRestOffsetAttr(0.0)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            UsdShade.Material(compliant_material), materialPurpose="physics"
        )
        prim.CreateAttribute(
            "kcg:primitiveFamily", Sdf.ValueTypeNames.String, custom=True
        ).Set("synthetic_barrier_spheres")
        prim.CreateAttribute(
            "kcg:responseRole", Sdf.ValueTypeNames.String, custom=True
        ).Set("compliant_pin_barrier")
        prim.CreateAttribute(
            "kcg:diagnosticOnly", Sdf.ValueTypeNames.Bool, custom=True
        ).Set(True)

    target_path = (
        connector_root
        + "/LoosePlug/BodyAssembly/Contacts/Socket_A/"
        "SyntheticBarrierTarget"
    )
    target_depth_end = 0.002000
    target = UsdGeom.Cylinder.Define(stage, target_path)
    target.CreateAxisAttr("Z")
    target.CreateRadiusAttr(target_radius)
    target.CreateHeightAttr(target_depth_end - target_front_depth)
    target.CreateExtentAttr(
        [
            Gf.Vec3f(
                -target_radius,
                -target_radius,
                -0.5 * (target_depth_end - target_front_depth),
            ),
            Gf.Vec3f(
                target_radius,
                target_radius,
                0.5 * (target_depth_end - target_front_depth),
            ),
        ]
    )
    UsdGeom.Xformable(target).AddTranslateOp().Set(
        Gf.Vec3d(
            plug_center_x,
            plug_center_y,
            0.5 * (target_front_depth + target_depth_end),
        )
    )
    prim = target.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
    collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    collision.CreateContactOffsetAttr(0.00001)
    collision.CreateRestOffsetAttr(0.0)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        UsdShade.Material(hard_material), materialPurpose="physics"
    )
    prim.CreateAttribute(
        "kcg:primitiveFamily", Sdf.ValueTypeNames.String, custom=True
    ).Set("synthetic_barrier_target")
    prim.CreateAttribute(
        "kcg:responseRole", Sdf.ValueTypeNames.String, custom=True
    ).Set("hard_socket_entry")
    prim.CreateAttribute(
        "kcg:diagnosticOnly", Sdf.ValueTypeNames.Bool, custom=True
    ).Set(True)


def _author_synthetic_spring_probe(
    *,
    stage: Any,
    connector_root: str,
    sphere_count: int,
    compliant_material: Any,
    hard_material: Any,
    Gf: Any,
    Sdf: Any,
    PhysxSchema: Any,
    UsdGeom: Any,
    UsdPhysics: Any,
    UsdShade: Any,
) -> None:
    """Author twelve one-point shell spring-finger response candidates."""

    if sphere_count != 12:
        raise ValueError("synthetic spring probe requires exactly twelve spheres")
    sphere_radius = 0.000500
    target_radius = 0.0179575
    full_radial_deflection = 0.000080
    sphere_ring_radius = target_radius + sphere_radius - full_radial_deflection
    radial_gap_to_target_side = sphere_ring_radius - target_radius
    axial_lead_in = math.sqrt(
        sphere_radius * sphere_radius
        - radial_gap_to_target_side * radial_gap_to_target_side
    )
    first_touch_separation = 0.01080
    sphere_center_depth = first_touch_separation + axial_lead_in

    sphere_parent = (
        connector_root
        + "/LoosePlug/BodyAssembly/SpringFingers/SyntheticSpheres"
    )
    UsdGeom.Xform.Define(stage, sphere_parent)
    for index in range(sphere_count):
        local_angle = math.radians(-8.0 - 30.0 * index)
        sphere = UsdGeom.Sphere.Define(
            stage, sphere_parent + f"/Finger_{index:02d}"
        )
        sphere.CreateRadiusAttr(sphere_radius)
        sphere.CreateExtentAttr(
            [
                Gf.Vec3f(-sphere_radius, -sphere_radius, -sphere_radius),
                Gf.Vec3f(sphere_radius, sphere_radius, sphere_radius),
            ]
        )
        UsdGeom.Xformable(sphere).AddTranslateOp().Set(
            Gf.Vec3d(
                sphere_ring_radius * math.cos(local_angle),
                sphere_ring_radius * math.sin(local_angle),
                sphere_center_depth,
            )
        )
        prim = sphere.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
        collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        collision.CreateContactOffsetAttr(0.00001)
        collision.CreateRestOffsetAttr(0.0)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            UsdShade.Material(compliant_material), materialPurpose="physics"
        )
        prim.CreateAttribute(
            "kcg:primitiveFamily", Sdf.ValueTypeNames.String, custom=True
        ).Set("synthetic_spring_spheres")
        prim.CreateAttribute(
            "kcg:responseRole", Sdf.ValueTypeNames.String, custom=True
        ).Set("compliant_spring_finger")
        prim.CreateAttribute(
            "kcg:diagnosticOnly", Sdf.ValueTypeNames.Bool, custom=True
        ).Set(True)

    target_path = (
        connector_root
        + "/FixedReceptacle/MatingShell/SyntheticSpringTarget"
    )
    target = UsdGeom.Cylinder.Define(stage, target_path)
    target.CreateAxisAttr("Z")
    target.CreateRadiusAttr(target_radius)
    target.CreateHeightAttr(0.0060)
    target.CreateExtentAttr(
        [
            Gf.Vec3f(-target_radius, -target_radius, -0.0030),
            Gf.Vec3f(target_radius, target_radius, 0.0030),
        ]
    )
    UsdGeom.Xformable(target).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0030))
    prim = target.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
    collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    collision.CreateContactOffsetAttr(0.00001)
    collision.CreateRestOffsetAttr(0.0)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        UsdShade.Material(hard_material), materialPurpose="physics"
    )
    prim.CreateAttribute(
        "kcg:primitiveFamily", Sdf.ValueTypeNames.String, custom=True
    ).Set("synthetic_spring_target")
    prim.CreateAttribute(
        "kcg:responseRole", Sdf.ValueTypeNames.String, custom=True
    ).Set("hard_receptacle_bore")
    prim.CreateAttribute(
        "kcg:diagnosticOnly", Sdf.ValueTypeNames.Bool, custom=True
    ).Set(True)


def _author_synthetic_continuous_bottoming(
    *,
    stage: Any,
    connector_root: str,
    fixed_material: Any,
    plug_material: Any,
    Gf: Any,
    Sdf: Any,
    PhysxSchema: Any,
    UsdGeom: Any,
    UsdPhysics: Any,
    UsdShade: Any,
) -> None:
    """Replace a 48x48 stop pair with one cap and three hard spheres."""

    def cylinder(
        *,
        path: str,
        center_z: float,
        family: str,
        material: Any,
    ) -> None:
        radius = 0.01695
        height = 0.00030
        geometry = UsdGeom.Cylinder.Define(stage, path)
        geometry.CreateAxisAttr("Z")
        geometry.CreateRadiusAttr(radius)
        geometry.CreateHeightAttr(height)
        geometry.CreateExtentAttr(
            [
                Gf.Vec3f(-radius, -radius, -0.5 * height),
                Gf.Vec3f(radius, radius, 0.5 * height),
            ]
        )
        UsdGeom.Xformable(geometry).AddTranslateOp().Set(
            Gf.Vec3d(0.0, 0.0, center_z)
        )
        prim = geometry.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
        collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        collision.CreateContactOffsetAttr(0.00001)
        collision.CreateRestOffsetAttr(0.0)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            UsdShade.Material(material), materialPurpose="physics"
        )
        prim.CreateAttribute(
            "kcg:primitiveFamily", Sdf.ValueTypeNames.String, custom=True
        ).Set(family)
        prim.CreateAttribute(
            "kcg:responseRole", Sdf.ValueTypeNames.String, custom=True
        ).Set("hard_metal_bottoming")
        prim.CreateAttribute(
            "kcg:diagnosticOnly", Sdf.ValueTypeNames.Bool, custom=True
        ).Set(True)

    cylinder(
        path=(
            connector_root
            + "/FixedReceptacle/MatingShell/SyntheticMetalStop"
        ),
        center_z=0.00015,
        family="synthetic_fixed_metal_stop",
        material=fixed_material,
    )
    sphere_parent = (
        connector_root
        + "/LoosePlug/BodyAssembly/InternalMatingShell/SyntheticMetalStop"
    )
    UsdGeom.Xform.Define(stage, sphere_parent)
    sphere_radius = 0.00050
    sphere_ring_radius = 0.01600
    sphere_center_z = 0.01505 + sphere_radius
    for index in range(3):
        angle = math.radians(120.0 * index)
        sphere = UsdGeom.Sphere.Define(
            stage, sphere_parent + f"/Sphere_{index}"
        )
        sphere.CreateRadiusAttr(sphere_radius)
        sphere.CreateExtentAttr(
            [
                Gf.Vec3f(-sphere_radius, -sphere_radius, -sphere_radius),
                Gf.Vec3f(sphere_radius, sphere_radius, sphere_radius),
            ]
        )
        UsdGeom.Xformable(sphere).AddTranslateOp().Set(
            Gf.Vec3d(
                sphere_ring_radius * math.cos(angle),
                sphere_ring_radius * math.sin(angle),
                sphere_center_z,
            )
        )
        prim = sphere.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
        collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        collision.CreateContactOffsetAttr(0.00001)
        collision.CreateRestOffsetAttr(0.0)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            UsdShade.Material(plug_material), materialPurpose="physics"
        )
        prim.CreateAttribute(
            "kcg:primitiveFamily", Sdf.ValueTypeNames.String, custom=True
        ).Set("synthetic_plug_metal_stop_spheres")
        prim.CreateAttribute(
            "kcg:responseRole", Sdf.ValueTypeNames.String, custom=True
        ).Set("hard_metal_bottoming")
        prim.CreateAttribute(
            "kcg:diagnosticOnly", Sdf.ValueTypeNames.Bool, custom=True
        ).Set(True)


def _author_synthetic_shoulder(
    *,
    stage: Any,
    connector_root: str,
    polarity: str,
    hard_material: Any,
    sphere_material: Any,
    Gf: Any,
    Sdf: Any,
    PhysxSchema: Any,
    UsdGeom: Any,
    UsdPhysics: Any,
    UsdShade: Any,
) -> None:
    """Author a seam-free three-point D6 shoulder stop candidate."""

    if polarity not in {"positive", "negative"}:
        raise ValueError("synthetic shoulder polarity must be positive or negative")

    positive = polarity == "positive"
    title = "Positive" if positive else "Negative"
    cap_center_z = 0.03015 if positive else 0.00985
    cap_family = f"synthetic_shoulder_{polarity}_body0"
    sphere_contact_face_z = 0.02995 if positive else 0.01005
    sphere_family = f"synthetic_shoulder_{polarity}_body1_spheres"
    cap_path = (
        connector_root
        + f"/LoosePlug/BodyAssembly/NutBearingShoulders/Synthetic{title}Stop/Cap"
    )
    radius = 0.01960
    height = 0.00030
    cap = UsdGeom.Cylinder.Define(stage, cap_path)
    cap.CreateAxisAttr("Z")
    cap.CreateRadiusAttr(radius)
    cap.CreateHeightAttr(height)
    cap.CreateExtentAttr(
        [
            Gf.Vec3f(-radius, -radius, -0.5 * height),
            Gf.Vec3f(radius, radius, 0.5 * height),
        ]
    )
    UsdGeom.Xformable(cap).AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, cap_center_z)
    )

    def mark(prim: Any, family: str, physics_material: Any) -> None:
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
        collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        collision.CreateContactOffsetAttr(0.00001)
        collision.CreateRestOffsetAttr(0.0)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            UsdShade.Material(physics_material), materialPurpose="physics"
        )
        prim.CreateAttribute(
            "kcg:primitiveFamily", Sdf.ValueTypeNames.String, custom=True
        ).Set(family)
        prim.CreateAttribute(
            "kcg:responseRole", Sdf.ValueTypeNames.String, custom=True
        ).Set("hard_nut_body_shoulder")
        prim.CreateAttribute(
            "kcg:diagnosticOnly", Sdf.ValueTypeNames.Bool, custom=True
        ).Set(True)

    mark(cap.GetPrim(), cap_family, hard_material)
    body0_group = stage.GetPrimAtPath(
        connector_root
        + f"/CollisionGroups/shoulder_{polarity}_body0_48"
    )
    body0_members = body0_group.GetRelationship(
        "collection:colliders:includes"
    ) if body0_group else None
    if not body0_members:
        raise RuntimeError("missing synthetic shoulder body0 collision group")
    body0_members.AddTarget(Sdf.Path(cap_path))
    sphere_parent = (
        connector_root
        + f"/LoosePlug/CouplingNut/NutBearingShoulders/Synthetic{title}Stop"
    )
    UsdGeom.Xform.Define(stage, sphere_parent)
    body1_group = stage.GetPrimAtPath(
        connector_root
        + f"/CollisionGroups/shoulder_{polarity}_body1_48"
    )
    body1_members = body1_group.GetRelationship(
        "collection:colliders:includes"
    ) if body1_group else None
    if not body1_members:
        raise RuntimeError("missing synthetic shoulder body1 collision group")
    sphere_radius = 0.00050
    sphere_ring_radius = 0.01850
    sphere_center_z = sphere_contact_face_z + (
        -sphere_radius if positive else sphere_radius
    )
    for index in range(3):
        angle = math.radians(120.0 * index)
        sphere_path = sphere_parent + f"/Sphere_{index}"
        sphere = UsdGeom.Sphere.Define(stage, sphere_path)
        sphere.CreateRadiusAttr(sphere_radius)
        sphere.CreateExtentAttr(
            [
                Gf.Vec3f(-sphere_radius, -sphere_radius, -sphere_radius),
                Gf.Vec3f(sphere_radius, sphere_radius, sphere_radius),
            ]
        )
        UsdGeom.Xformable(sphere).AddTranslateOp().Set(
            Gf.Vec3d(
                sphere_ring_radius * math.cos(angle),
                sphere_ring_radius * math.sin(angle),
                sphere_center_z,
            )
        )
        mark(sphere.GetPrim(), sphere_family, sphere_material)
        body1_members.AddTarget(Sdf.Path(sphere_path))

def _run(arguments: argparse.Namespace) -> Mapping[str, Any]:
    from isaacsim.core.api import World
    from isaacsim.core.prims import RigidPrim
    from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
    from omni.physx import get_physx_simulation_interface
    from omni.physx.scripts import physicsUtils
    import omni.usd
    from pxr import (
        Gf,
        PhysxSchema,
        PhysicsSchemaTools,
        Sdf,
        Usd,
        UsdGeom,
        UsdPhysics,
        UsdShade,
    )

    from kcg_connector.d38999_keyed_v2_physical_model_contract import (
        WORKSPACE_ROOT,
    )
    from kcg_connector.d38999_tabletop_scene import (
        author_d38999_tabletop_scene,
        load_d38999_tabletop_scene,
        verify_d38999_tabletop_asset,
    )
    from d38999_physical_r7_p1_nominal_bench import _set_existing_transform

    spec = CASE_SPECS[arguments.case]
    separation = float(
        spec["separation_m"]
        if arguments.separation_m is None
        else arguments.separation_m
    )
    nut_transz = float(
        spec.get("nut_transz_m", 0.0)
        if arguments.nut_transz_m is None
        else arguments.nut_transz_m
    )
    config = load_d38999_tabletop_scene(arguments.scene_config)
    asset_path = verify_d38999_tabletop_asset(config, WORKSPACE_ROOT)
    dt = 1.0 / float(config.physics.rate_hz)

    World.clear_instance()
    omni.usd.get_context().new_stage()
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=dt,
        rendering_dt=1.0 / 60.0,
        backend="numpy",
        device="cpu",
    )
    stage = get_current_stage()
    authored = author_d38999_tabletop_scene(
        stage,
        config,
        asset_path,
        add_reference_to_stage=add_reference_to_stage,
        Gf=Gf,
        Sdf=Sdf,
        UsdGeom=UsdGeom,
        UsdPhysics=UsdPhysics,
        UsdShade=UsdShade,
        physics_utils=physicsUtils,
    )
    if authored["object_pose_writes_after_start"] != 0:
        raise RuntimeError("scene reports a post-start object pose write")

    fixed_origin = np.asarray(
        config.fixed_endpoint.receptacle_origin_m, dtype=np.float64
    )
    _set_existing_transform(
        stage,
        config.asset.loose_plug_prim_path,
        fixed_origin + np.asarray((0.0, 0.0, -separation)),
        (0.0, 0.0, 0.0),
        UsdGeom,
        Gf,
    )
    connector_root = config.asset.body_prim_path.split("/LoosePlug/", 1)[0]
    joint = stage.GetPrimAtPath(connector_root + "/LoosePlug/CouplingNutJoint")
    if not joint:
        raise RuntimeError("missing coupling-nut joint")

    shoulder_probe = bool(spec.get("shoulder_probe_internal")) or (
        spec.get("synthetic_shoulder_polarity") is not None
    )
    use_original_shoulder_joint = bool(
        shoulder_probe and arguments.use_original_shoulder_joint
    )
    joint.GetAttribute("physics:jointEnabled").Set(
        use_original_shoulder_joint
    )
    if use_original_shoulder_joint:
        joint.GetAttribute("physics:collisionEnabled").Set(True)
        if arguments.shoulder_joint_limit_m is not None:
            joint_limit = float(arguments.shoulder_joint_limit_m)
            joint.GetAttribute("limit:transZ:physics:low").Set(-joint_limit)
            joint.GetAttribute("limit:transZ:physics:high").Set(joint_limit)
    for path in (config.asset.body_prim_path, config.asset.nut_prim_path):
        prim = stage.GetPrimAtPath(path)
        kinematic = not bool(
            shoulder_probe and path == config.asset.nut_prim_path
        )
        UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr().Set(kinematic)
        PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateEnableCCDAttr().Set(
            False
        )
    shoulder_holding_joint_authored = False
    if shoulder_probe:
        nut_prim = stage.GetPrimAtPath(config.asset.nut_prim_path)
        UsdGeom.Xformable(nut_prim).AddTranslateOp(
            opSuffix="diagnosticShoulderTransZ"
        ).Set(Gf.Vec3d(0.0, 0.0, nut_transz))
        if not arguments.free_shoulder_nut and not use_original_shoulder_joint:
            holding_joint = UsdPhysics.FixedJoint.Define(
                stage,
                connector_root + "/LoosePlug/DiagnosticShoulderHoldingJoint",
            )
            holding_joint.CreateJointEnabledAttr(True)
            holding_joint.CreateCollisionEnabledAttr(True)
            holding_joint.CreateBody0Rel().SetTargets(
                [Sdf.Path(config.asset.body_prim_path)]
            )
            holding_joint.CreateBody1Rel().SetTargets(
                [Sdf.Path(config.asset.nut_prim_path)]
            )
            holding_joint.CreateLocalPos0Attr(
                Gf.Vec3f(0.0, 0.0, 0.020 + nut_transz)
            )
            holding_joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.020))
            identity = Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0))
            holding_joint.CreateLocalRot0Attr(identity)
            holding_joint.CreateLocalRot1Attr(identity)
            shoulder_holding_joint_authored = True

    material_path = connector_root + str(spec["material_suffix"])
    material = stage.GetPrimAtPath(material_path)
    if not material:
        raise RuntimeError(f"missing compliant material: {material_path}")
    stiffness_attr = material.GetAttribute(
        "physxMaterial:compliantContactStiffness"
    )
    damping_attr = material.GetAttribute(
        "physxMaterial:compliantContactDamping"
    )
    if arguments.stiffness_n_m is not None:
        stiffness_attr.Set(float(arguments.stiffness_n_m))
    if arguments.damping_n_s_m is not None:
        damping_attr.Set(float(arguments.damping_n_s_m))

    synthetic_sphere_count = spec.get("synthetic_seal_sphere_count")
    if synthetic_sphere_count is not None:
        hard_material_path = connector_root + str(spec["hard_material_suffix"])
        hard_material = stage.GetPrimAtPath(hard_material_path)
        if not hard_material:
            raise RuntimeError(
                f"missing synthetic-probe hard material: {hard_material_path}"
            )
        _author_synthetic_seal_probe(
            stage=stage,
            connector_root=connector_root,
            sphere_count=int(synthetic_sphere_count),
            compliant_material=material,
            hard_material=hard_material,
            Gf=Gf,
            Sdf=Sdf,
            PhysxSchema=PhysxSchema,
            UsdGeom=UsdGeom,
            UsdPhysics=UsdPhysics,
            UsdShade=UsdShade,
        )
    synthetic_socket_sphere_count = spec.get("synthetic_socket_sphere_count")
    if synthetic_socket_sphere_count is not None:
        _author_synthetic_socket_probe(
            stage=stage,
            connector_root=connector_root,
            sphere_count=int(synthetic_socket_sphere_count),
            compliant_material=material,
            Gf=Gf,
            Sdf=Sdf,
            PhysxSchema=PhysxSchema,
            UsdGeom=UsdGeom,
            UsdPhysics=UsdPhysics,
            UsdShade=UsdShade,
        )
    synthetic_barrier_sphere_count = spec.get(
        "synthetic_barrier_sphere_count"
    )
    if synthetic_barrier_sphere_count is not None:
        hard_material_path = connector_root + str(spec["hard_material_suffix"])
        hard_material = stage.GetPrimAtPath(hard_material_path)
        if not hard_material:
            raise RuntimeError(
                f"missing barrier-probe hard material: {hard_material_path}"
            )
        _author_synthetic_barrier_probe(
            stage=stage,
            connector_root=connector_root,
            sphere_count=int(synthetic_barrier_sphere_count),
            compliant_material=material,
            hard_material=hard_material,
            Gf=Gf,
            Sdf=Sdf,
            PhysxSchema=PhysxSchema,
            UsdGeom=UsdGeom,
            UsdPhysics=UsdPhysics,
            UsdShade=UsdShade,
        )
    synthetic_spring_sphere_count = spec.get("synthetic_spring_sphere_count")
    if synthetic_spring_sphere_count is not None:
        hard_material_path = connector_root + str(spec["hard_material_suffix"])
        hard_material = stage.GetPrimAtPath(hard_material_path)
        if not hard_material:
            raise RuntimeError(
                f"missing spring-probe hard material: {hard_material_path}"
            )
        _author_synthetic_spring_probe(
            stage=stage,
            connector_root=connector_root,
            sphere_count=int(synthetic_spring_sphere_count),
            compliant_material=material,
            hard_material=hard_material,
            Gf=Gf,
            Sdf=Sdf,
            PhysxSchema=PhysxSchema,
            UsdGeom=UsdGeom,
            UsdPhysics=UsdPhysics,
            UsdShade=UsdShade,
        )
    if spec.get("synthetic_continuous_bottoming") is True:
        fixed_material_path = connector_root + str(
            spec["fixed_hard_material_suffix"]
        )
        plug_material_path = connector_root + str(
            spec["plug_hard_material_suffix"]
        )
        fixed_material = stage.GetPrimAtPath(fixed_material_path)
        plug_material = stage.GetPrimAtPath(plug_material_path)
        if not fixed_material or not plug_material:
            raise RuntimeError("missing continuous-bottoming hard material")
        _author_synthetic_continuous_bottoming(
            stage=stage,
            connector_root=connector_root,
            fixed_material=fixed_material,
            plug_material=plug_material,
            Gf=Gf,
            Sdf=Sdf,
            PhysxSchema=PhysxSchema,
            UsdGeom=UsdGeom,
            UsdPhysics=UsdPhysics,
            UsdShade=UsdShade,
        )
    shoulder_polarity = spec.get("synthetic_shoulder_polarity")
    if shoulder_polarity is not None:
        hard_material_path = connector_root + str(spec["hard_material_suffix"])
        hard_material = stage.GetPrimAtPath(hard_material_path)
        if not hard_material:
            raise RuntimeError("missing synthetic-shoulder hard material")
        _author_synthetic_shoulder(
            stage=stage,
            connector_root=connector_root,
            polarity=str(shoulder_polarity),
            hard_material=hard_material,
            sphere_material=material,
            Gf=Gf,
            Sdf=Sdf,
            PhysxSchema=PhysxSchema,
            UsdGeom=UsdGeom,
            UsdPhysics=UsdPhysics,
            UsdShade=UsdShade,
        )
    left_paths: list[str] = []
    right_paths: list[str] = []
    disabled_count = 0
    for prim in stage.Traverse():
        collision = prim.GetAttribute("physics:collisionEnabled")
        if not collision or collision.Get() is not True:
            continue
        path = str(prim.GetPath())
        family = _family(prim)
        left = bool(
            family == spec["left_family"]
            and _path_matches(path, spec, "left")
        )
        right = bool(
            family == spec["right_family"]
            and _path_matches(path, spec, "right")
        )
        keep = left or right
        if keep:
            (left_paths if left else right_paths).append(path)
        else:
            collision.Set(False)
            disabled_count += 1
    if (
        len(left_paths) != int(spec["left_count"])
        or len(right_paths) != int(spec["right_count"])
    ):
        raise RuntimeError(
            "probe collider inventory mismatch: "
            f"left={len(left_paths)} right={len(right_paths)}"
        )
    kept_paths = left_paths + right_paths
    left_path_set = set(left_paths)
    right_path_set = set(right_paths)
    collision_groups_removed_for_isolation = False
    if shoulder_probe and not arguments.keep_shoulder_collision_groups:
        collision_groups_path = connector_root + "/CollisionGroups"
        if not stage.GetPrimAtPath(collision_groups_path):
            raise RuntimeError("missing collision groups for shoulder isolation")
        if not stage.RemovePrim(collision_groups_path):
            raise RuntimeError("failed to remove collision groups for shoulder isolation")
        collision_groups_removed_for_isolation = True

    for owner_path in (
        config.asset.fixed_receptacle_prim_path,
        config.asset.body_prim_path,
        config.asset.nut_prim_path,
    ):
        PhysxSchema.PhysxContactReportAPI.Apply(
            stage.GetPrimAtPath(owner_path)
        ).CreateThresholdAttr().Set(0.0)

    body_view = RigidPrim(
        prim_paths_expr=config.asset.body_prim_path,
        name="single_contact_probe_body",
        reset_xform_properties=False,
    )
    nut_view = RigidPrim(
        prim_paths_expr=config.asset.nut_prim_path,
        name="single_contact_probe_nut",
        reset_xform_properties=False,
    )
    world.get_physics_context().set_gravity(0.0)
    world.reset()
    body_view.initialize()
    nut_view.initialize()
    initial_body_pose = body_view.get_world_poses()
    initial_nut_pose = nut_view.get_world_poses()

    def world_bounds(paths: Sequence[str]) -> Mapping[str, Any]:
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            useExtentsHint=True,
        )
        rows: list[Mapping[str, Any]] = []
        minimum = np.full(3, np.inf, dtype=np.float64)
        maximum = np.full(3, -np.inf, dtype=np.float64)
        for path in paths:
            world_range = cache.ComputeWorldBound(
                stage.GetPrimAtPath(path)
            ).ComputeAlignedRange()
            row_minimum = np.asarray(world_range.GetMin(), dtype=np.float64)
            row_maximum = np.asarray(world_range.GetMax(), dtype=np.float64)
            minimum = np.minimum(minimum, row_minimum)
            maximum = np.maximum(maximum, row_maximum)
            if len(rows) < 3:
                rows.append(
                    {
                        "path": path,
                        "minimum_m": row_minimum.tolist(),
                        "maximum_m": row_maximum.tolist(),
                    }
                )
        return {
            "union_minimum_m": minimum.tolist(),
            "union_maximum_m": maximum.tolist(),
            "first_rows": rows,
        }

    initial_left_world_bounds = world_bounds(left_paths)
    initial_right_world_bounds = world_bounds(right_paths)
    interface = get_physx_simulation_interface()
    samples: list[dict[str, Any]] = []
    for step in range(arguments.settle_steps):
        world.step(render=False)
        headers, contacts, _friction = interface.get_full_contact_report()
        points: list[dict[str, Any]] = []
        for header in headers:
            collider_paths = (
                str(PhysicsSchemaTools.intToSdfPath(header.collider0)),
                str(PhysicsSchemaTools.intToSdfPath(header.collider1)),
            )
            if not bool(
                (
                    collider_paths[0] in left_path_set
                    and collider_paths[1] in right_path_set
                )
                or (
                    collider_paths[1] in left_path_set
                    and collider_paths[0] in right_path_set
                )
            ):
                continue
            start = int(header.contact_data_offset)
            stop = start + int(header.num_contact_data)
            for index in range(start, stop):
                contact = contacts[index]
                impulse = [float(value) for value in contact.impulse]
                points.append(
                    {
                        "separation_m": float(contact.separation),
                        "impulse_ns": impulse,
                        "impulse_norm_ns": math.sqrt(
                            sum(value * value for value in impulse)
                        ),
                    }
                )
        if points:
            vector_sum = [
                sum(point["impulse_ns"][axis] for point in points)
                for axis in range(3)
            ]
            samples.append(
                {
                    "step": step + 1,
                    "point_count": len(points),
                    "minimum_separation_m": min(
                        point["separation_m"] for point in points
                    ),
                    "maximum_separation_m": max(
                        point["separation_m"] for point in points
                    ),
                    "sum_impulse_norm_force_n": sum(
                        point["impulse_norm_ns"] for point in points
                    )
                    / dt,
                    "vector_sum_force_n": [value / dt for value in vector_sum],
                    "points": points,
                }
            )

    nonzero = [
        sample
        for sample in samples
        if sample["sum_impulse_norm_force_n"] > 0.0
    ]
    final_body_pose = body_view.get_world_poses()
    final_nut_pose = nut_view.get_world_poses()
    tail = nonzero[-min(20, len(nonzero)) :]
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_id": GENERATOR_ID,
        "role": "a3_modelling_diagnostic_not_formal_acceptance",
        "case": arguments.case,
        "asset_path": str(asset_path),
        "scene_config": str(Path(arguments.scene_config).resolve()),
        "physics_rate_hz": config.physics.rate_hz,
        "separation_m": separation,
        "nut_transz_m": nut_transz,
        "free_shoulder_nut": bool(arguments.free_shoulder_nut),
        "use_original_shoulder_joint": use_original_shoulder_joint,
        "shoulder_joint_limit_m": arguments.shoulder_joint_limit_m,
        "shoulder_holding_joint_authored": shoulder_holding_joint_authored,
        "initial_body_position_m": np.asarray(initial_body_pose[0][0]).tolist(),
        "initial_body_orientation_wxyz": np.asarray(
            initial_body_pose[1][0]
        ).tolist(),
        "initial_nut_position_m": np.asarray(initial_nut_pose[0][0]).tolist(),
        "initial_nut_orientation_wxyz": np.asarray(
            initial_nut_pose[1][0]
        ).tolist(),
        "initial_left_world_bounds": initial_left_world_bounds,
        "initial_right_world_bounds": initial_right_world_bounds,
        "final_body_position_m": np.asarray(final_body_pose[0][0]).tolist(),
        "final_body_orientation_wxyz": np.asarray(
            final_body_pose[1][0]
        ).tolist(),
        "final_nut_position_m": np.asarray(final_nut_pose[0][0]).tolist(),
        "final_nut_orientation_wxyz": np.asarray(final_nut_pose[1][0]).tolist(),
        "kept_collider_paths": sorted(kept_paths),
        "left_collider_count": len(left_paths),
        "right_collider_count": len(right_paths),
        "disabled_collider_count": disabled_count,
        "collision_groups_removed_for_isolation": (
            collision_groups_removed_for_isolation
        ),
        "keep_shoulder_collision_groups": bool(
            arguments.keep_shoulder_collision_groups
        ),
        "material_path": material_path,
        "resolved_stiffness_n_m": float(stiffness_attr.Get()),
        "resolved_damping_n_s_m": float(damping_attr.Get()),
        "settle_steps": arguments.settle_steps,
        "contact_sample_count": len(samples),
        "nonzero_force_sample_count": len(nonzero),
        "first_contact_sample": samples[0] if samples else None,
        "last_contact_sample": samples[-1] if samples else None,
        "steady_tail_sample_count": len(tail),
        "steady_sum_impulse_norm_force_n": (
            None
            if not tail
            else float(
                np.median(
                    [sample["sum_impulse_norm_force_n"] for sample in tail]
                )
            )
        ),
        "steady_contact_point_count": (
            None
            if not tail
            else int(round(float(np.median([s["point_count"] for s in tail]))))
        ),
        "steady_minimum_separation_m": (
            None
            if not tail
            else float(
                np.median([sample["minimum_separation_m"] for sample in tail])
            )
        ),
        "object_pose_write_after_physics_start_count": 0,
        "file_fingerprints_computed": False,
        "formal_acceptance_evidence": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _arguments(argv)
    output = Path(arguments.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite probe output: {output}")
    output.mkdir(parents=True, exist_ok=False)

    from isaacsim import SimulationApp

    application = SimulationApp(
        {
            "headless": True,
            "hide_ui": True,
            "renderer": "Minimal",
            "multi_gpu": False,
            "fast_shutdown": True,
            "enable_crashreporter": False,
        }
    )
    status = 1
    try:
        report = _run(arguments)
        status = 0
    except BaseException as error:
        report = {
            "schema_version": SCHEMA_VERSION,
            "generator_id": GENERATOR_ID,
            "case": arguments.case,
            "status": "FAILED",
            "error_type": type(error).__name__,
            "error": str(error),
            "file_fingerprints_computed": False,
            "formal_acceptance_evidence": False,
        }
        traceback.print_exc()
    finally:
        (output / "report.json").write_text(
            json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        application.close()
    _emit(json.dumps(report, allow_nan=False, sort_keys=True))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
