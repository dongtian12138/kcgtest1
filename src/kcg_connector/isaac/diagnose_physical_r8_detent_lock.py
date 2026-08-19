#!/usr/bin/env python3

"""Isolate physical-r8 coupling-nut rotation with and without detent contact.

This is a read-only diagnostic, not an acceptance bench.  It starts the real
r8 connector at the P1 pre-entry separation, applies the same bounded nut-yaw
servo in two fresh CPU scenes, and reports whether disabling only the three
detent follower colliders changes the realized relative rotation.  It writes
no artifact and computes no file fingerprint.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import traceback
from typing import Any

import numpy as np


def _emit(value: Any) -> None:
    os.write(
        1,
        (json.dumps(value, allow_nan=False, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )


def _run_case(
    *,
    disable_detent_followers: bool,
    detent_follower_phase_offset_deg: float = 0.0,
    detent_stiffness_n_m: float | None = None,
    detent_damping_n_s_m: float | None = None,
    detent_follower_radial_outward_shift_m: float = 0.0,
    detent_cam_radial_rise_m: float | None = None,
    disable_detent_continuous_base: bool = False,
    disable_nut_body_shoulders: bool = False,
    replace_detent_followers_with_analytic_cylinders: bool = False,
    analytic_follower_shape: str = "cylinder",
    analytic_follower_radius_m: float = 0.00025,
    analytic_follower_center_radius_m: float = 0.022025,
    replace_segmented_cam_with_continuous_base: bool = False,
    add_triangular_teeth_to_continuous_base: bool = False,
    step_count: int = 240,
    physics_rate_hz: float | None = None,
    target_yaw_limit_rad: float = 0.40,
    target_yaw_rate_rad_s: float = 0.412335167120566,
    body_yaw_position_gain_nm_rad: float = 0.8,
    nut_yaw_position_gain_nm_rad: float = 0.8,
    angular_velocity_gain_nm_s_rad: float = 0.01,
    torque_component_limit_nm: float = 0.30,
    constant_nut_yaw_torque_nm: float | None = None,
    constant_nut_yaw_torque_start_step: int = 0,
) -> dict[str, Any]:
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.prims import RigidPrim
    from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
    from omni.physx import get_physx_simulation_interface
    from omni.physx.scripts import physicsUtils
    from pxr import Gf, PhysxSchema, PhysicsSchemaTools, Sdf, UsdGeom, UsdPhysics, UsdShade

    from d38999_physical_r7_p1_nominal_bench import (
        _clamp_vector,
        _contact_rows,
        _finite_vector,
        _quat_to_rpy_wxyz,
        _set_existing_transform,
    )
    from kcg_connector.d38999_keyed_v2_physical_model_contract import WORKSPACE_ROOT
    from kcg_connector.d38999_tabletop_scene import (
        author_d38999_tabletop_scene,
        load_d38999_tabletop_scene,
        verify_d38999_tabletop_asset,
    )

    repository = Path(__file__).resolve().parents[3]
    config = load_d38999_tabletop_scene(
        repository
        / "src/kcg_connector/config/d38999_keyed_v2_tabletop_scene_v1.yaml"
    )
    asset_path = verify_d38999_tabletop_asset(config, WORKSPACE_ROOT)
    resolved_physics_rate_hz = (
        float(config.physics.rate_hz)
        if physics_rate_hz is None
        else float(physics_rate_hz)
    )
    if (
        not math.isfinite(resolved_physics_rate_hz)
        or resolved_physics_rate_hz <= 0.0
    ):
        raise ValueError("physics_rate_hz must be finite and positive")
    for label, value in (
        ("body_yaw_position_gain_nm_rad", body_yaw_position_gain_nm_rad),
        ("nut_yaw_position_gain_nm_rad", nut_yaw_position_gain_nm_rad),
        ("angular_velocity_gain_nm_s_rad", angular_velocity_gain_nm_s_rad),
        ("torque_component_limit_nm", torque_component_limit_nm),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{label} must be finite and nonnegative")
    if torque_component_limit_nm <= 0.0:
        raise ValueError("torque_component_limit_nm must be positive")
    if constant_nut_yaw_torque_nm is not None and not math.isfinite(
        constant_nut_yaw_torque_nm
    ):
        raise ValueError("constant_nut_yaw_torque_nm must be finite")
    if (
        constant_nut_yaw_torque_start_step < 0
        or constant_nut_yaw_torque_start_step >= step_count
    ):
        raise ValueError(
            "constant_nut_yaw_torque_start_step must select a simulated step"
        )
    dt = 1.0 / resolved_physics_rate_hz

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
    author_d38999_tabletop_scene(
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
    fixed_origin = np.asarray(
        config.fixed_endpoint.receptacle_origin_m, dtype=np.float64
    )
    plug_origin = fixed_origin + np.asarray((0.0, 0.0, -0.00550))
    _set_existing_transform(
        stage,
        config.asset.loose_plug_prim_path,
        plug_origin,
        (0.0, 0.0, 0.0),
        UsdGeom,
        Gf,
    )

    disabled_paths: list[str] = []
    follower_paths: list[str] = []
    follower_prefix = config.asset.nut_prim_path + "/AntiDecoupling/Follower_"
    if (
        not math.isfinite(detent_follower_radial_outward_shift_m)
        or detent_follower_radial_outward_shift_m < 0.0
        or detent_follower_radial_outward_shift_m >= 0.00050
    ):
        raise ValueError(
            "detent_follower_radial_outward_shift_m must be in [0, 0.00050)"
        )
    for prim in stage.Traverse():
        family = prim.GetAttribute("kcg:primitiveFamily")
        if (
            str(prim.GetPath()).startswith(follower_prefix)
            and family
            and family.Get() == "detent_followers_3"
        ):
            follower_paths.append(str(prim.GetPath()))
            if detent_follower_radial_outward_shift_m:
                follower_index = int(str(prim.GetPath()).rsplit("_", 1)[1])
                phase_rad = math.radians(follower_index * 120.0)
                points_attr = UsdGeom.Mesh(prim).GetPointsAttr()
                points = points_attr.Get()
                if not points or len(points) != 8:
                    raise RuntimeError(
                        f"expected eight follower points at {prim.GetPath()}"
                    )
                shift_mm = detent_follower_radial_outward_shift_m * 1000.0
                delta = Gf.Vec3f(
                    shift_mm * math.cos(phase_rad),
                    shift_mm * math.sin(phase_rad),
                    0.0,
                )
                points_attr.Set([Gf.Vec3f(point) + delta for point in points])
            if (
                disable_detent_followers
                or replace_detent_followers_with_analytic_cylinders
            ):
                collision = prim.GetAttribute("physics:collisionEnabled")
                if not collision:
                    raise RuntimeError(f"missing collisionEnabled at {prim.GetPath()}")
                collision.Set(False)
                disabled_paths.append(str(prim.GetPath()))
            if (
                detent_follower_phase_offset_deg
                and not replace_detent_followers_with_analytic_cylinders
            ):
                UsdGeom.Xformable(prim).AddRotateZOp(
                    opSuffix="diagnosticDetentPhase"
                ).Set(float(detent_follower_phase_offset_deg))
    if len(follower_paths) != 3:
        raise RuntimeError(
            f"expected exactly three detent followers, got {follower_paths}"
        )

    if detent_cam_radial_rise_m is not None and (
        not math.isfinite(detent_cam_radial_rise_m)
        or detent_cam_radial_rise_m <= 0.0
        or detent_cam_radial_rise_m >= 0.001
    ):
        raise ValueError("detent_cam_radial_rise_m must be in (0, 0.001)")
    cam_tooth_paths: list[str] = []
    resolved_detent_cam_radial_rises_m: list[float] = []
    for prim in stage.Traverse():
        family = prim.GetAttribute("kcg:primitiveFamily")
        if (
            not family
            or family.Get() != "detent_cam_teeth_36"
            or not prim.IsA(UsdGeom.Mesh)
        ):
            continue
        mesh = UsdGeom.Mesh(prim)
        points_attr = mesh.GetPointsAttr()
        points = points_attr.Get()
        if not points or len(points) != 6:
            raise RuntimeError(
                f"expected six cam-tooth points at {prim.GetPath()}"
            )
        base_radius_mm = math.hypot(float(points[0][0]), float(points[0][1]))
        peak_radius_mm = math.hypot(float(points[1][0]), float(points[1][1]))
        if detent_cam_radial_rise_m is not None:
            target_peak_radius_mm = (
                base_radius_mm + detent_cam_radial_rise_m * 1000.0
            )
            for point_index in (1, 4):
                point = points[point_index]
                source_radius_mm = math.hypot(
                    float(point[0]), float(point[1])
                )
                if source_radius_mm <= 0.0:
                    raise RuntimeError(
                        f"invalid cam-tooth peak point at {prim.GetPath()}"
                    )
                scale = target_peak_radius_mm / source_radius_mm
                points[point_index] = Gf.Vec3f(
                    float(point[0]) * scale,
                    float(point[1]) * scale,
                    float(point[2]),
                )
            points_attr.Set(points)
            peak_radius_mm = target_peak_radius_mm
        cam_tooth_paths.append(str(prim.GetPath()))
        resolved_detent_cam_radial_rises_m.append(
            (peak_radius_mm - base_radius_mm) / 1000.0
        )
    if len(cam_tooth_paths) != 36:
        raise RuntimeError(
            f"expected exactly 36 detent cam teeth, got {cam_tooth_paths}"
        )
    resolved_detent_cam_radial_rise_m = float(
        sum(resolved_detent_cam_radial_rises_m)
        / len(resolved_detent_cam_radial_rises_m)
    )
    if max(resolved_detent_cam_radial_rises_m) - min(
        resolved_detent_cam_radial_rises_m
    ) > 1.0e-8:
        raise RuntimeError(
            "detent cam teeth do not share one radial rise: "
            + repr(
                (
                    min(resolved_detent_cam_radial_rises_m),
                    max(resolved_detent_cam_radial_rises_m),
                )
            )
        )

    material_root = config.asset.body_prim_path.split("/LoosePlug", 1)[0]
    compliant_material_path = (
        material_root
        + "/Materials/anti_decoupling_detent__compliant_detent_follower"
    )
    compliant_material_prim = stage.GetPrimAtPath(compliant_material_path)
    if not compliant_material_prim:
        raise RuntimeError(
            "missing compliant detent material at " + compliant_material_path
        )
    compliant_api = PhysxSchema.PhysxMaterialAPI(compliant_material_prim)
    stiffness_attr = compliant_api.GetCompliantContactStiffnessAttr()
    damping_attr = compliant_api.GetCompliantContactDampingAttr()
    if not stiffness_attr or not damping_attr:
        raise RuntimeError(
            "missing compliant detent stiffness/damping attributes at "
            + compliant_material_path
        )
    if detent_stiffness_n_m is not None:
        if not math.isfinite(detent_stiffness_n_m) or detent_stiffness_n_m <= 0.0:
            raise ValueError("detent_stiffness_n_m must be finite and positive")
        stiffness_attr.Set(float(detent_stiffness_n_m))
    if detent_damping_n_s_m is not None:
        if not math.isfinite(detent_damping_n_s_m) or detent_damping_n_s_m < 0.0:
            raise ValueError("detent_damping_n_s_m must be finite and nonnegative")
        damping_attr.Set(float(detent_damping_n_s_m))
    resolved_detent_stiffness_n_m = float(stiffness_attr.Get())
    resolved_detent_damping_n_s_m = float(damping_attr.Get())

    disabled_continuous_base_paths: list[str] = []
    if disable_detent_continuous_base:
        for prim in stage.Traverse():
            family = prim.GetAttribute("kcg:primitiveFamily")
            if (
                family
                and family.Get() == "detent_cam_continuous_base_1"
                and prim.HasAPI(UsdPhysics.CollisionAPI)
            ):
                prim.GetAttribute("physics:collisionEnabled").Set(False)
                disabled_continuous_base_paths.append(str(prim.GetPath()))
        if len(disabled_continuous_base_paths) != 1:
            raise RuntimeError(
                "expected one disabled continuous detent base, got "
                + str(disabled_continuous_base_paths)
            )

    if disable_detent_followers and replace_detent_followers_with_analytic_cylinders:
        raise ValueError(
            "cannot disable detent followers and add diagnostic replacements"
        )
    replacement_follower_paths: list[str] = []
    if replace_detent_followers_with_analytic_cylinders:
        if analytic_follower_shape not in {"cylinder", "sphere"}:
            raise ValueError(
                "analytic_follower_shape must be cylinder or sphere"
            )
        if (
            not math.isfinite(analytic_follower_radius_m)
            or analytic_follower_radius_m <= 0.0
            or not math.isfinite(analytic_follower_center_radius_m)
            or analytic_follower_center_radius_m <= analytic_follower_radius_m
        ):
            raise ValueError("invalid analytic detent follower dimensions")
        follower_group = stage.GetPrimAtPath(
            material_root + "/CollisionGroups/detent_followers_3"
        )
        if not follower_group:
            raise RuntimeError("missing detent follower collision group")
        includes = follower_group.GetRelationship("collection:colliders:includes")
        if not includes:
            raise RuntimeError("missing detent follower group membership relation")
        for follower_index in range(3):
            phase_rad = math.radians(
                follower_index * 120.0 + detent_follower_phase_offset_deg
            )
            path = (
                config.asset.nut_prim_path
                + "/AntiDecoupling/DiagnosticFollowerCylinder_"
                + str(follower_index)
            )
            if analytic_follower_shape == "sphere":
                geometry = UsdGeom.Sphere.Define(stage, path)
                geometry.CreateRadiusAttr(float(analytic_follower_radius_m))
                geometry.CreateExtentAttr(
                    [
                        Gf.Vec3f(
                            -analytic_follower_radius_m,
                            -analytic_follower_radius_m,
                            -analytic_follower_radius_m,
                        ),
                        Gf.Vec3f(
                            analytic_follower_radius_m,
                            analytic_follower_radius_m,
                            analytic_follower_radius_m,
                        ),
                    ]
                )
            else:
                geometry = UsdGeom.Cylinder.Define(stage, path)
                geometry.CreateAxisAttr(UsdGeom.Tokens.z)
                geometry.CreateRadiusAttr(float(analytic_follower_radius_m))
                geometry.CreateHeightAttr(0.00060)
            follower_prim = geometry.GetPrim()
            UsdGeom.Xformable(follower_prim).AddTranslateOp().Set(
                Gf.Vec3d(
                    analytic_follower_center_radius_m * math.cos(phase_rad),
                    analytic_follower_center_radius_m * math.sin(phase_rad),
                    0.0200,
                )
            )
            follower_prim.CreateAttribute(
                "kcg:primitiveFamily", Sdf.ValueTypeNames.String, custom=True
            ).Set("detent_followers_3")
            follower_prim.CreateAttribute(
                "kcg:materialRole", Sdf.ValueTypeNames.String, custom=True
            ).Set("anti_decoupling_detent")
            follower_prim.CreateAttribute(
                "kcg:responseRole", Sdf.ValueTypeNames.String, custom=True
            ).Set("compliant_detent_follower")
            UsdPhysics.CollisionAPI.Apply(
                follower_prim
            ).CreateCollisionEnabledAttr(True)
            follower_physx = PhysxSchema.PhysxCollisionAPI.Apply(follower_prim)
            follower_physx.CreateContactOffsetAttr(1.0e-5)
            follower_physx.CreateRestOffsetAttr(0.0)
            UsdShade.MaterialBindingAPI.Apply(follower_prim).Bind(
                UsdShade.Material(compliant_material_prim),
                materialPurpose="physics",
            )
            includes.AddTarget(Sdf.Path(path))
            replacement_follower_paths.append(path)

    disabled_shoulder_paths: list[str] = []
    shoulder_families = {
        "shoulder_positive_body0_48",
        "shoulder_positive_body1_48",
        "shoulder_negative_body0_48",
        "shoulder_negative_body1_48",
    }
    for prim in stage.Traverse():
        family = prim.GetAttribute("kcg:primitiveFamily")
        if not family or family.Get() not in shoulder_families:
            continue
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if disable_nut_body_shoulders:
            collision = prim.GetAttribute("physics:collisionEnabled")
            if not collision:
                raise RuntimeError(f"missing collisionEnabled at {prim.GetPath()}")
            collision.Set(False)
            disabled_shoulder_paths.append(str(prim.GetPath()))
    if disable_nut_body_shoulders and len(disabled_shoulder_paths) != 192:
        raise RuntimeError(
            "expected 192 disabled shoulder colliders, got "
            + str(len(disabled_shoulder_paths))
        )

    disabled_cam_paths: list[str] = []
    if replace_segmented_cam_with_continuous_base:
        cam_prefix = config.asset.body_prim_path + "/AntiDecoupling/Cam/"
        for prim in stage.Traverse():
            family = prim.GetAttribute("kcg:primitiveFamily")
            if (
                str(prim.GetPath()).startswith(cam_prefix)
                and family
                and family.Get() == "detent_cam_1368"
            ):
                collision = prim.GetAttribute("physics:collisionEnabled")
                if not collision:
                    raise RuntimeError(f"missing collisionEnabled at {prim.GetPath()}")
                collision.Set(False)
                disabled_cam_paths.append(str(prim.GetPath()))
        if len(disabled_cam_paths) != 1368:
            raise RuntimeError(
                "expected 1368 segmented cam colliders, got "
                + str(len(disabled_cam_paths))
            )
        cylinder = UsdGeom.Cylinder.Define(
            stage,
            config.asset.body_prim_path
            + "/AntiDecoupling/DiagnosticContinuousBase",
        )
        cylinder.CreateAxisAttr(UsdGeom.Tokens.z)
        cylinder.CreateRadiusAttr(0.021975)
        cylinder.CreateHeightAttr(0.00080)
        cylinder_prim = cylinder.GetPrim()
        UsdGeom.Xformable(cylinder_prim).AddTranslateOp().Set(
            Gf.Vec3d(0.0, 0.0, 0.0200)
        )
        cylinder_prim.CreateAttribute(
            "kcg:primitiveFamily", Sdf.ValueTypeNames.String, custom=True
        ).Set("diagnostic_detent_continuous_base_1")
        UsdPhysics.CollisionAPI.Apply(
            cylinder_prim
        ).CreateCollisionEnabledAttr(True)
        physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(cylinder_prim)
        physx_collision.CreateContactOffsetAttr(1.0e-5)
        physx_collision.CreateRestOffsetAttr(0.0)
        hard_cam = UsdShade.Material.Get(
            stage,
            material_root
            + "/Materials/anti_decoupling_detent__hard_detent_cam",
        )
        if not hard_cam:
            raise RuntimeError("missing hard detent cam material")
        UsdShade.MaterialBindingAPI.Apply(cylinder_prim).Bind(
            hard_cam, materialPurpose="physics"
        )
        if add_triangular_teeth_to_continuous_base:
            base_radius = 0.021975
            peak_radius = 0.022025
            z_low = 0.01960
            z_high = 0.02040
            for tooth_index in range(36):
                tooth_phase = tooth_index * 10.0
                polar = (
                    (tooth_phase - 10.0, base_radius),
                    (tooth_phase - 9.908821, peak_radius),
                    (tooth_phase - 8.982274, base_radius),
                )
                xy = [
                    (
                        radius * math.cos(math.radians(theta)),
                        radius * math.sin(math.radians(theta)),
                    )
                    for theta, radius in polar
                ]
                physical_points = [
                    (xy[0][0], xy[0][1], z_low),
                    (xy[1][0], xy[1][1], z_low),
                    (xy[2][0], xy[2][1], z_low),
                    (xy[0][0], xy[0][1], z_high),
                    (xy[1][0], xy[1][1], z_high),
                    (xy[2][0], xy[2][1], z_high),
                ]
                mesh = UsdGeom.Mesh.Define(
                    stage,
                    config.asset.body_prim_path
                    + f"/AntiDecoupling/DiagnosticTeeth/Tooth_{tooth_index:02d}",
                )
                mesh.CreatePointsAttr(
                    [
                        Gf.Vec3f(*(value * 1000.0 for value in point))
                        for point in physical_points
                    ]
                )
                mesh.CreateFaceVertexCountsAttr([3, 3, 4, 4, 4])
                mesh.CreateFaceVertexIndicesAttr(
                    [2, 1, 0, 3, 4, 5, 0, 1, 4, 3, 1, 2, 5, 4, 2, 0, 3, 5]
                )
                mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
                mesh.AddScaleOp().Set(Gf.Vec3f(0.001))
                tooth_prim = mesh.GetPrim()
                tooth_prim.CreateAttribute(
                    "kcg:primitiveFamily", Sdf.ValueTypeNames.String, custom=True
                ).Set("diagnostic_detent_triangular_teeth_36")
                UsdPhysics.CollisionAPI.Apply(
                    tooth_prim
                ).CreateCollisionEnabledAttr(True)
                UsdPhysics.MeshCollisionAPI.Apply(
                    tooth_prim
                ).CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
                PhysxSchema.PhysxConvexHullCollisionAPI.Apply(
                    tooth_prim
                ).CreateMinThicknessAttr(0.001)
                tooth_physx = PhysxSchema.PhysxCollisionAPI.Apply(tooth_prim)
                tooth_physx.CreateContactOffsetAttr(1.0e-5)
                tooth_physx.CreateRestOffsetAttr(0.0)
                UsdShade.MaterialBindingAPI.Apply(tooth_prim).Bind(
                    hard_cam, materialPurpose="physics"
                )

    for owner_path in (
        config.asset.body_prim_path,
        config.asset.nut_prim_path,
    ):
        PhysxSchema.PhysxContactReportAPI.Apply(
            stage.GetPrimAtPath(owner_path)
        ).CreateThresholdAttr().Set(0.0)

    body = RigidPrim(
        prim_paths_expr=config.asset.body_prim_path,
        name=("detent_disabled_body" if disable_detent_followers else "detent_body"),
        reset_xform_properties=False,
    )
    nut = RigidPrim(
        prim_paths_expr=config.asset.nut_prim_path,
        name=("detent_disabled_nut" if disable_detent_followers else "detent_nut"),
        reset_xform_properties=False,
    )
    world.get_physics_context().set_gravity(0.0)
    world.reset()
    body.initialize()
    nut.initialize()

    def state(view: Any, label: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        positions, orientations = view.get_world_poses()
        return (
            _finite_vector(positions[0], 3, label + " position"),
            _finite_vector(orientations[0], 4, label + " orientation"),
            _finite_vector(view.get_velocities()[0], 6, label + " velocity"),
        )

    def apply(view: Any, force: np.ndarray, torque: np.ndarray) -> None:
        view.apply_forces_and_torques_at_pos(
            forces=np.asarray([force], dtype=np.float32),
            torques=np.asarray([torque], dtype=np.float32),
            positions=None,
            is_global=True,
        )

    interface = get_physx_simulation_interface()
    contact_family_pairs: set[tuple[str | None, str | None]] = set()
    contact_data_attribute_names: list[str] = []
    detent_contact_samples: list[dict[str, Any]] = []
    detent_contact_torque_z_nm: list[float] = []
    detent_directional_resistance_nm: list[float] = []
    initial_body = state(body, "initial body")
    initial_nut = state(nut, "initial nut")
    body_yaw = _quat_to_rpy_wxyz(initial_body[1])[2]
    nut_yaw = _quat_to_rpy_wxyz(initial_nut[1])[2]
    previous_body_yaw = body_yaw
    previous_nut_yaw = nut_yaw
    unwrapped_body_yaw = body_yaw
    unwrapped_nut_yaw = nut_yaw
    maximum_abs_nut_torque = 0.0
    maximum_abs_relative_yaw = 0.0
    maximum_body_angular_speed_after_half_rad_s = 0.0
    maximum_nut_angular_speed_after_half_rad_s = 0.0
    maximum_abs_body_yaw_rate_after_half_rad_s = 0.0
    maximum_abs_nut_yaw_rate_after_half_rad_s = 0.0
    constant_torque_start_relative_yaw_rad: float | None = None
    maximum_abs_relative_yaw_change_after_constant_torque_rad = 0.0
    maximum_abs_relative_yaw_rate_after_constant_torque_rad_s = 0.0

    for step in range(step_count):
        body_position, body_quaternion, body_velocity = state(body, "body")
        nut_position, nut_quaternion, nut_velocity = state(nut, "nut")
        if step >= step_count // 2:
            maximum_body_angular_speed_after_half_rad_s = max(
                maximum_body_angular_speed_after_half_rad_s,
                float(np.linalg.norm(body_velocity[3:])),
            )
            maximum_nut_angular_speed_after_half_rad_s = max(
                maximum_nut_angular_speed_after_half_rad_s,
                float(np.linalg.norm(nut_velocity[3:])),
            )
            maximum_abs_body_yaw_rate_after_half_rad_s = max(
                maximum_abs_body_yaw_rate_after_half_rad_s,
                abs(float(body_velocity[5])),
            )
            maximum_abs_nut_yaw_rate_after_half_rad_s = max(
                maximum_abs_nut_yaw_rate_after_half_rad_s,
                abs(float(nut_velocity[5])),
            )
        body_rpy = _quat_to_rpy_wxyz(body_quaternion)
        nut_rpy = _quat_to_rpy_wxyz(nut_quaternion)
        if step:
            body_delta = (
                body_rpy[2] - previous_body_yaw + math.pi
            ) % (2.0 * math.pi) - math.pi
            nut_delta = (
                nut_rpy[2] - previous_nut_yaw + math.pi
            ) % (2.0 * math.pi) - math.pi
            unwrapped_body_yaw += body_delta
            unwrapped_nut_yaw += nut_delta
        previous_body_yaw = body_rpy[2]
        previous_nut_yaw = nut_rpy[2]
        relative_yaw = unwrapped_nut_yaw - unwrapped_body_yaw
        if (
            constant_nut_yaw_torque_nm is not None
            and step >= constant_nut_yaw_torque_start_step
        ):
            if constant_torque_start_relative_yaw_rad is None:
                constant_torque_start_relative_yaw_rad = relative_yaw
            maximum_abs_relative_yaw_change_after_constant_torque_rad = max(
                maximum_abs_relative_yaw_change_after_constant_torque_rad,
                abs(relative_yaw - constant_torque_start_relative_yaw_rad),
            )
            maximum_abs_relative_yaw_rate_after_constant_torque_rad_s = max(
                maximum_abs_relative_yaw_rate_after_constant_torque_rad_s,
                abs(float(nut_velocity[5] - body_velocity[5])),
            )

        elapsed = step * dt
        command_direction = (
            -math.copysign(1.0, target_yaw_rate_rad_s)
            if target_yaw_rate_rad_s != 0.0
            else 0.0
        )
        target_yaw_magnitude = min(
            target_yaw_limit_rad,
            elapsed * abs(target_yaw_rate_rad_s),
        )
        target_nut_yaw = command_direction * target_yaw_magnitude
        target_nut_omega = (
            command_direction * abs(target_yaw_rate_rad_s)
            if target_yaw_magnitude < target_yaw_limit_rad
            else 0.0
        )
        body_force = _clamp_vector(
            600.0 * (plug_origin - body_position) - 8.0 * body_velocity[:3],
            8.0,
        )
        nut_force = _clamp_vector(
            600.0 * (plug_origin - nut_position) - 8.0 * nut_velocity[:3],
            8.0,
        )
        body_torque = _clamp_vector(
            np.asarray(
                (
                    -1.2 * body_rpy[0] - 0.01 * body_velocity[3],
                    -1.2 * body_rpy[1] - 0.01 * body_velocity[4],
                    -body_yaw_position_gain_nm_rad * unwrapped_body_yaw
                    - angular_velocity_gain_nm_s_rad * body_velocity[5],
                )
            ),
            torque_component_limit_nm,
        )
        nut_yaw_torque = (
            nut_yaw_position_gain_nm_rad
            * (target_nut_yaw - unwrapped_nut_yaw)
            + angular_velocity_gain_nm_s_rad
            * (target_nut_omega - nut_velocity[5])
        )
        if (
            constant_nut_yaw_torque_nm is not None
            and step >= constant_nut_yaw_torque_start_step
        ):
            nut_yaw_torque = constant_nut_yaw_torque_nm
        nut_torque = _clamp_vector(
            np.asarray(
                (
                    -1.2 * nut_rpy[0] - 0.01 * nut_velocity[3],
                    -1.2 * nut_rpy[1] - 0.01 * nut_velocity[4],
                    nut_yaw_torque,
                )
            ),
            torque_component_limit_nm,
        )
        maximum_abs_nut_torque = max(
            maximum_abs_nut_torque, abs(float(nut_torque[2]))
        )
        apply(body, body_force, body_torque)
        apply(nut, nut_force, nut_torque)
        world.step(render=False)
        for row in _contact_rows(stage, interface, PhysicsSchemaTools):
            contact_family_pairs.add(tuple(row["families"]))
        headers, contacts, _friction = interface.get_full_contact_report()
        step_detent_torque_z_nm = 0.0
        for header in headers:
            actor_paths = (
                str(PhysicsSchemaTools.intToSdfPath(header.actor0)),
                str(PhysicsSchemaTools.intToSdfPath(header.actor1)),
            )
            collider_paths = (
                str(PhysicsSchemaTools.intToSdfPath(header.collider0)),
                str(PhysicsSchemaTools.intToSdfPath(header.collider1)),
            )
            families = tuple(
                (
                    stage.GetPrimAtPath(path)
                    .GetAttribute("kcg:primitiveFamily")
                    .Get()
                    if stage.GetPrimAtPath(path)
                    and stage.GetPrimAtPath(path).GetAttribute(
                        "kcg:primitiveFamily"
                    )
                    else None
                )
                for path in collider_paths
            )
            if "detent_followers_3" not in families or not any(
                family in {
                    "detent_cam_continuous_base_1",
                    "detent_cam_teeth_36",
                }
                for family in families
            ):
                continue
            if config.asset.nut_prim_path == actor_paths[0]:
                impulse_sign = 1.0
            elif config.asset.nut_prim_path == actor_paths[1]:
                impulse_sign = -1.0
            else:
                raise RuntimeError(
                    "detent contact does not include CouplingNut actor"
                )
            start = int(header.contact_data_offset)
            stop = start + int(header.num_contact_data)
            for contact in contacts[start:stop]:
                lever = np.asarray(contact.position, dtype=np.float64) - nut_position
                impulse = (
                    impulse_sign
                    * np.asarray(contact.impulse, dtype=np.float64)
                )
                step_detent_torque_z_nm += float(
                    np.cross(lever, impulse)[2] / dt
                )
        detent_contact_torque_z_nm.append(step_detent_torque_z_nm)
        if (
            command_direction != 0.0
            and 5 <= step
            and target_yaw_magnitude < target_yaw_limit_rad
        ):
            detent_directional_resistance_nm.append(
                -command_direction * step_detent_torque_z_nm
            )
        if step in {0, 1, step_count // 2, step_count - 1}:
            if contacts and not contact_data_attribute_names:
                contact_data_attribute_names = sorted(
                    name for name in dir(contacts[0]) if not name.startswith("_")
                )
            for header in headers:
                collider_paths = (
                    str(PhysicsSchemaTools.intToSdfPath(header.collider0)),
                    str(PhysicsSchemaTools.intToSdfPath(header.collider1)),
                )
                families = tuple(
                    (
                        stage.GetPrimAtPath(path)
                        .GetAttribute("kcg:primitiveFamily")
                        .Get()
                        if stage.GetPrimAtPath(path)
                        and stage.GetPrimAtPath(path).GetAttribute(
                            "kcg:primitiveFamily"
                        )
                        else None
                    )
                    for path in collider_paths
                )
                if set(families) != {
                    "detent_followers_3",
                    "detent_cam_continuous_base_1",
                }:
                    continue
                start = int(header.contact_data_offset)
                stop = start + int(header.num_contact_data)
                records: list[dict[str, Any]] = []
                for contact in contacts[start:stop]:
                    record: dict[str, Any] = {
                        "separation_m": float(contact.separation)
                    }
                    for name in ("position", "normal", "impulse"):
                        value = getattr(contact, name, None)
                        if value is not None:
                            record[name] = [float(component) for component in value]
                    records.append(record)
                detent_contact_samples.append(
                    {
                        "step": step + 1,
                        "families": list(families),
                        "collider_paths": list(collider_paths),
                        "records": records,
                    }
                )
        maximum_abs_relative_yaw = max(
            maximum_abs_relative_yaw,
            abs(unwrapped_nut_yaw - unwrapped_body_yaw),
        )

    final_body = state(body, "final body")
    final_nut = state(nut, "final nut")
    final_body_yaw = _quat_to_rpy_wxyz(final_body[1])[2]
    final_nut_yaw = _quat_to_rpy_wxyz(final_nut[1])[2]
    unwrapped_body_yaw += (
        final_body_yaw - previous_body_yaw + math.pi
    ) % (2.0 * math.pi) - math.pi
    unwrapped_nut_yaw += (
        final_nut_yaw - previous_nut_yaw + math.pi
    ) % (2.0 * math.pi) - math.pi
    if constant_torque_start_relative_yaw_rad is not None:
        maximum_abs_relative_yaw_change_after_constant_torque_rad = max(
            maximum_abs_relative_yaw_change_after_constant_torque_rad,
            abs(
                unwrapped_nut_yaw
                - unwrapped_body_yaw
                - constant_torque_start_relative_yaw_rad
            ),
        )
        maximum_abs_relative_yaw_rate_after_constant_torque_rad_s = max(
            maximum_abs_relative_yaw_rate_after_constant_torque_rad_s,
            abs(float(final_nut[2][5] - final_body[2][5])),
        )
    resistance = np.asarray(detent_directional_resistance_nm, dtype=np.float64)
    result = {
        "disable_detent_followers": disable_detent_followers,
        "detent_follower_phase_offset_deg": detent_follower_phase_offset_deg,
        "detent_stiffness_n_m": resolved_detent_stiffness_n_m,
        "detent_damping_n_s_m": resolved_detent_damping_n_s_m,
        "detent_follower_radial_outward_shift_m": (
            detent_follower_radial_outward_shift_m
        ),
        "detent_cam_radial_rise_m": resolved_detent_cam_radial_rise_m,
        "disable_detent_continuous_base": disable_detent_continuous_base,
        "disabled_continuous_base_paths": disabled_continuous_base_paths,
        "disable_nut_body_shoulders": disable_nut_body_shoulders,
        "disabled_shoulder_path_count": len(disabled_shoulder_paths),
        "replace_detent_followers_with_analytic_cylinders": (
            replace_detent_followers_with_analytic_cylinders
        ),
        "analytic_follower_shape": analytic_follower_shape,
        "analytic_follower_radius_m": analytic_follower_radius_m,
        "analytic_follower_center_radius_m": analytic_follower_center_radius_m,
        "replacement_follower_paths": replacement_follower_paths,
        "replace_segmented_cam_with_continuous_base": (
            replace_segmented_cam_with_continuous_base
        ),
        "add_triangular_teeth_to_continuous_base": (
            add_triangular_teeth_to_continuous_base
        ),
        "step_count": step_count,
        "physics_rate_hz": resolved_physics_rate_hz,
        "target_yaw_limit_rad": target_yaw_limit_rad,
        "target_yaw_rate_rad_s": target_yaw_rate_rad_s,
        "body_yaw_position_gain_nm_rad": body_yaw_position_gain_nm_rad,
        "nut_yaw_position_gain_nm_rad": nut_yaw_position_gain_nm_rad,
        "angular_velocity_gain_nm_s_rad": angular_velocity_gain_nm_s_rad,
        "torque_component_limit_nm": torque_component_limit_nm,
        "constant_nut_yaw_torque_nm": constant_nut_yaw_torque_nm,
        "constant_nut_yaw_torque_start_step": (
            constant_nut_yaw_torque_start_step
        ),
        "constant_torque_start_relative_yaw_rad": (
            constant_torque_start_relative_yaw_rad
        ),
        "maximum_abs_relative_yaw_change_after_constant_torque_rad": (
            maximum_abs_relative_yaw_change_after_constant_torque_rad
        ),
        "maximum_abs_relative_yaw_rate_after_constant_torque_rad_s": (
            maximum_abs_relative_yaw_rate_after_constant_torque_rad_s
        ),
        "follower_paths": sorted(follower_paths),
        "disabled_segmented_cam_path_count": len(disabled_cam_paths),
        "disabled_paths": sorted(disabled_paths),
        "initial_body_yaw_rad": body_yaw,
        "initial_nut_yaw_rad": nut_yaw,
        "final_body_wrapped_yaw_rad": final_body_yaw,
        "final_nut_wrapped_yaw_rad": final_nut_yaw,
        "final_unwrapped_body_yaw_rad": unwrapped_body_yaw,
        "final_unwrapped_nut_yaw_rad": unwrapped_nut_yaw,
        "final_relative_yaw_rad": unwrapped_nut_yaw - unwrapped_body_yaw,
        "maximum_abs_relative_yaw_rad": maximum_abs_relative_yaw,
        "maximum_abs_nut_torque_nm": maximum_abs_nut_torque,
        "maximum_body_angular_speed_after_half_rad_s": (
            maximum_body_angular_speed_after_half_rad_s
        ),
        "maximum_nut_angular_speed_after_half_rad_s": (
            maximum_nut_angular_speed_after_half_rad_s
        ),
        "maximum_abs_body_yaw_rate_after_half_rad_s": (
            maximum_abs_body_yaw_rate_after_half_rad_s
        ),
        "maximum_abs_nut_yaw_rate_after_half_rad_s": (
            maximum_abs_nut_yaw_rate_after_half_rad_s
        ),
        "final_body_velocity": final_body[2].tolist(),
        "final_nut_velocity": final_nut[2].tolist(),
        "contact_family_pairs": sorted(
            [list(pair) for pair in contact_family_pairs],
            key=lambda pair: str(pair),
        ),
        "contact_data_attribute_names": contact_data_attribute_names,
        "detent_contact_samples": detent_contact_samples,
        "detent_contact_torque_z_nm": detent_contact_torque_z_nm,
        "detent_directional_resistance_statistics_nm": {
            "sample_count": int(resistance.size),
            "mean": float(np.mean(resistance)) if resistance.size else 0.0,
            "p50": float(np.quantile(resistance, 0.50)) if resistance.size else 0.0,
            "p95": float(np.quantile(resistance, 0.95)) if resistance.size else 0.0,
            "maximum": float(np.max(resistance)) if resistance.size else 0.0,
            "minimum": float(np.min(resistance)) if resistance.size else 0.0,
        },
    }
    world.stop()
    World.clear_instance()
    return result


def main() -> int:
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
    try:
        enabled = _run_case(disable_detent_followers=False)
        centered = _run_case(
            disable_detent_followers=False,
            detent_follower_phase_offset_deg=-8.982274 / 2.0,
        )
        continuous_base = _run_case(
            disable_detent_followers=False,
            replace_segmented_cam_with_continuous_base=True,
        )
        continuous_base_and_teeth = _run_case(
            disable_detent_followers=False,
            replace_segmented_cam_with_continuous_base=True,
            add_triangular_teeth_to_continuous_base=True,
        )
        disabled = _run_case(disable_detent_followers=True)
        result = {
            "schema_version": "kcg_d38999_physical_r8_detent_lock_diagnostic_v1",
            "asset_revision": "keyed_v3_physical_r8",
            "role": "diagnostic_only_not_acceptance_evidence",
            "file_fingerprints_computed": False,
            "detent_enabled": enabled,
            "detent_enabled_centered_on_base_dwell": centered,
            "detent_continuous_base_only": continuous_base,
            "detent_continuous_base_and_triangular_teeth": (
                continuous_base_and_teeth
            ),
            "detent_disabled": disabled,
            "detent_isolated_as_lock_cause": bool(
                enabled["maximum_abs_relative_yaw_rad"] < 1.0e-3
                and disabled["maximum_abs_relative_yaw_rad"] > 5.0e-2
            ),
            "centered_base_dwell_phase_restores_rotation": bool(
                centered["maximum_abs_relative_yaw_rad"] > 5.0e-2
            ),
            "continuous_base_restores_rotation": bool(
                continuous_base["maximum_abs_relative_yaw_rad"] > 5.0e-2
            ),
            "continuous_base_and_teeth_preserve_rotation": bool(
                continuous_base_and_teeth["maximum_abs_relative_yaw_rad"]
                > 5.0e-2
            ),
        }
        _emit(result)
        return 0
    except Exception as error:
        _emit(
            {
                "schema_version": "kcg_d38999_physical_r8_detent_lock_diagnostic_v1",
                "status": "ERROR",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "file_fingerprints_computed": False,
            }
        )
        return 1
    finally:
        application.close()


if __name__ == "__main__":
    raise SystemExit(main())
