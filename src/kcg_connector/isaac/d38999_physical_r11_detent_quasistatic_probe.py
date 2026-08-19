#!/usr/bin/env python3

"""Quasistatically sample one realized r11 detent tooth with round followers.

Every angle is evaluated in a fresh stage.  BodyAssembly and CouplingNut are
made kinematic and positioned before reset, so the measured contact torque is
not contaminated by a yaw servo or by post-start pose writes.  The r11 asset
is never modified on disk and no file fingerprint is computed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Sequence

import numpy as np


SCHEMA_VERSION = "kcg_d38999_physical_r11_detent_quasistatic_probe_v1"
GENERATOR_ID = "kcg_d38999_physical_r11_detent_quasistatic_realized_probe_v1"
FOLLOWER_PHASE_OFFSET_DEG = -4.491137
CAM_BASE_RADIUS_M = 0.021975
CAM_PEAK_RADIUS_M = 0.022025
DEFAULT_ANGLES_DEG = (
    0.0,
    -4.491137,
    -4.70,
    -4.90,
    -5.10,
    -5.30,
    -5.40,
    -5.417684,
    4.491137,
    4.505,
    4.525,
    4.545,
    4.565,
    4.580,
    4.582316,
)


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--settle-steps", type=int, default=120)
    parser.add_argument("--stiffness-n-m", type=float, default=110000.0)
    parser.add_argument("--damping-n-s-m", type=float, default=2.0)
    parser.add_argument("--angles-deg", type=float, nargs="+")
    parser.add_argument("--tooth-only", action="store_true")
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument(
        "--follower-shape", choices=("cylinder", "sphere"), default="sphere"
    )
    parser.add_argument("--follower-radius-m", type=float, default=0.000075)
    parser.add_argument(
        "--kit-portable-root",
        required=True,
        help="Writable Kit portable root; this diagnostic only accepts /tmp paths.",
    )
    result = parser.parse_args(argv)
    if not result.run:
        parser.error("the quasistatic detent probe requires --run")
    if result.settle_steps < 60 or result.settle_steps > 500:
        parser.error("settle steps must be in [60, 500]")
    if not math.isfinite(result.stiffness_n_m) or result.stiffness_n_m <= 0.0:
        parser.error("stiffness must be finite and positive")
    if not math.isfinite(result.damping_n_s_m) or result.damping_n_s_m < 0.0:
        parser.error("damping must be finite and nonnegative")
    if (
        not math.isfinite(result.follower_radius_m)
        or result.follower_radius_m <= 0.000051
        or result.follower_radius_m > 0.00050
    ):
        parser.error("follower radius must be in (51 um, 0.50 mm]")
    if result.tooth_only and result.base_only:
        parser.error("tooth-only and base-only are mutually exclusive")
    portable_root = Path(result.kit_portable_root).expanduser().resolve()
    if not portable_root.is_relative_to(Path("/tmp")):
        parser.error("--kit-portable-root must resolve below /tmp")
    result.kit_portable_root = str(portable_root)
    return result


def _emit(value: Any) -> None:
    os.write(
        1,
        (json.dumps(value, allow_nan=False, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )


def _profile(progress_deg: float) -> tuple[str, float]:
    if progress_deg <= 8.982274:
        return "base_dwell", CAM_BASE_RADIUS_M
    if progress_deg <= 9.908821:
        fraction = (progress_deg - 8.982274) / 0.926547
        return (
            "shallow_positive_ascent",
            CAM_BASE_RADIUS_M
            + fraction * (CAM_PEAK_RADIUS_M - CAM_BASE_RADIUS_M),
        )
    fraction = (progress_deg - 9.908821) / 0.091179
    return (
        "steep_reverse_face",
        CAM_PEAK_RADIUS_M
        - fraction * (CAM_PEAK_RADIUS_M - CAM_BASE_RADIUS_M),
    )


def _run_angle(
    *,
    nut_local_yaw_deg: float,
    settle_steps: int,
    stiffness_n_m: float,
    damping_n_s_m: float,
    tooth_only: bool,
    base_only: bool,
    follower_shape: str,
    follower_radius_m: float,
) -> dict[str, Any]:
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.prims import RigidPrim
    from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
    from omni.physx import get_physx_simulation_interface
    from omni.physx.scripts import physicsUtils
    from pxr import (
        Gf,
        PhysxSchema,
        PhysicsSchemaTools,
        Sdf,
        UsdGeom,
        UsdPhysics,
        UsdShade,
    )

    from d38999_physical_r7_p1_nominal_bench import _set_existing_transform
    from kcg_connector.d38999_keyed_v2_physical_model_contract import (
        WORKSPACE_ROOT,
    )
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
    plug_origin = fixed_origin + np.asarray((0.0, 0.0, -0.00550))
    _set_existing_transform(
        stage,
        config.asset.loose_plug_prim_path,
        plug_origin,
        (0.0, 0.0, 0.0),
        UsdGeom,
        Gf,
    )
    connector_root = config.asset.body_prim_path.split("/LoosePlug/", 1)[0]
    joint = stage.GetPrimAtPath(connector_root + "/LoosePlug/CouplingNutJoint")
    joint.GetAttribute("physics:jointEnabled").Set(False)

    body_prim = stage.GetPrimAtPath(config.asset.body_prim_path)
    UsdPhysics.RigidBodyAPI(body_prim).CreateKinematicEnabledAttr().Set(True)
    PhysxSchema.PhysxRigidBodyAPI.Apply(body_prim).CreateEnableCCDAttr().Set(False)
    nut_prim = stage.GetPrimAtPath(config.asset.nut_prim_path)
    UsdPhysics.RigidBodyAPI(nut_prim).CreateKinematicEnabledAttr().Set(False)
    PhysxSchema.PhysxRigidBodyAPI.Apply(nut_prim).CreateEnableCCDAttr().Set(False)
    UsdGeom.Xformable(nut_prim).AddRotateZOp(
        opSuffix="diagnosticQuasistaticYaw"
    ).Set(float(nut_local_yaw_deg))
    angle_rad = math.radians(float(nut_local_yaw_deg))
    holding_joint = UsdPhysics.FixedJoint.Define(
        stage,
        connector_root + "/LoosePlug/DiagnosticQuasistaticHoldingJoint",
    )
    holding_joint.CreateJointEnabledAttr(True)
    holding_joint.CreateCollisionEnabledAttr(True)
    holding_joint.CreateBody0Rel().SetTargets(
        [Sdf.Path(config.asset.body_prim_path)]
    )
    holding_joint.CreateBody1Rel().SetTargets(
        [Sdf.Path(config.asset.nut_prim_path)]
    )
    holding_joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0200))
    holding_joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0200))
    holding_joint.CreateLocalRot0Attr(
        Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0))
    )
    holding_joint.CreateLocalRot1Attr(
        Gf.Quatf(
            math.cos(-0.5 * angle_rad),
            Gf.Vec3f(0.0, 0.0, math.sin(-0.5 * angle_rad)),
        )
    )

    compliant_material_path = (
        connector_root
        + "/Materials/anti_decoupling_detent__compliant_detent_follower"
    )
    compliant_material = stage.GetPrimAtPath(compliant_material_path)
    material_api = PhysxSchema.PhysxMaterialAPI(compliant_material)
    material_api.GetCompliantContactStiffnessAttr().Set(float(stiffness_n_m))
    material_api.GetCompliantContactDampingAttr().Set(float(damping_n_s_m))

    keep_cam_families = (
        {"detent_cam_continuous_base_1"}
        if base_only
        else {"detent_cam_teeth_36"}
    )
    if not tooth_only and not base_only:
        keep_cam_families.add("detent_cam_continuous_base_1")
    disabled_count = 0
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        collision = prim.GetAttribute("physics:collisionEnabled")
        if not collision or collision.Get() is not True:
            continue
        family_attr = prim.GetAttribute("kcg:primitiveFamily")
        family = family_attr.Get() if family_attr else None
        if family in keep_cam_families:
            continue
        collision.Set(False)
        disabled_count += 1

    follower_paths: list[str] = []
    follower_center_radius_m = (
        CAM_BASE_RADIUS_M + follower_radius_m - 0.000001
    )
    follower_group = stage.GetPrimAtPath(
        connector_root + "/CollisionGroups/detent_followers_3"
    )
    if not follower_group:
        raise RuntimeError("missing detent follower collision group")
    follower_group_members = follower_group.GetRelationship(
        "collection:colliders:includes"
    )
    if not follower_group_members:
        raise RuntimeError("missing detent follower collision-group membership")
    for index in range(3):
        phase_deg = FOLLOWER_PHASE_OFFSET_DEG + 120.0 * index
        phase_rad = math.radians(phase_deg)
        path = (
            config.asset.nut_prim_path
            + f"/AntiDecoupling/QuasistaticFollower_{index}"
        )
        if follower_shape == "sphere":
            geometry = UsdGeom.Sphere.Define(stage, path)
            geometry.CreateRadiusAttr(float(follower_radius_m))
            geometry.CreateExtentAttr(
                [
                    Gf.Vec3f(
                        -follower_radius_m,
                        -follower_radius_m,
                        -follower_radius_m,
                    ),
                    Gf.Vec3f(
                        follower_radius_m,
                        follower_radius_m,
                        follower_radius_m,
                    ),
                ]
            )
        else:
            geometry = UsdGeom.Cylinder.Define(stage, path)
            geometry.CreateAxisAttr(UsdGeom.Tokens.z)
            geometry.CreateRadiusAttr(float(follower_radius_m))
            geometry.CreateHeightAttr(0.00060)
        prim = geometry.GetPrim()
        UsdGeom.Xformable(prim).AddTranslateOp().Set(
            Gf.Vec3d(
                follower_center_radius_m * math.cos(phase_rad),
                follower_center_radius_m * math.sin(phase_rad),
                0.0200,
            )
        )
        prim.CreateAttribute(
            "kcg:primitiveFamily", Sdf.ValueTypeNames.String, custom=True
        ).Set("diagnostic_round_detent_followers_3")
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
        collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        collision_api.CreateContactOffsetAttr(1.0e-5)
        collision_api.CreateRestOffsetAttr(0.0)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            UsdShade.Material(compliant_material), materialPurpose="physics"
        )
        follower_group_members.AddTarget(Sdf.Path(path))
        follower_paths.append(path)

    for owner_path in (config.asset.body_prim_path, config.asset.nut_prim_path):
        PhysxSchema.PhysxContactReportAPI.Apply(
            stage.GetPrimAtPath(owner_path)
        ).CreateThresholdAttr().Set(0.0)

    body_view = RigidPrim(
        prim_paths_expr=config.asset.body_prim_path,
        name="quasistatic_detent_body",
        reset_xform_properties=False,
    )
    nut_view = RigidPrim(
        prim_paths_expr=config.asset.nut_prim_path,
        name="quasistatic_detent_nut",
        reset_xform_properties=False,
    )
    world.get_physics_context().set_gravity(0.0)
    world.reset()
    body_view.initialize()
    nut_view.initialize()
    initial_body_pose = body_view.get_world_poses()
    initial_nut_pose = nut_view.get_world_poses()
    interface = get_physx_simulation_interface()
    samples: list[dict[str, Any]] = []
    follower_set = set(follower_paths)
    for step in range(settle_steps):
        world.step(render=False)
        headers, contacts, _friction = interface.get_full_contact_report()
        torque_z_nm = 0.0
        point_count = 0
        minimum_separation_m: float | None = None
        collider_pairs: set[tuple[str, str]] = set()
        for header in headers:
            actor_paths = (
                str(PhysicsSchemaTools.intToSdfPath(header.actor0)),
                str(PhysicsSchemaTools.intToSdfPath(header.actor1)),
            )
            collider_paths = (
                str(PhysicsSchemaTools.intToSdfPath(header.collider0)),
                str(PhysicsSchemaTools.intToSdfPath(header.collider1)),
            )
            if not any(path in follower_set for path in collider_paths):
                continue
            if config.asset.nut_prim_path == actor_paths[0]:
                impulse_sign = 1.0
            elif config.asset.nut_prim_path == actor_paths[1]:
                impulse_sign = -1.0
            else:
                raise RuntimeError("detent contact is missing CouplingNut actor")
            collider_pairs.add(tuple(collider_paths))
            start = int(header.contact_data_offset)
            stop = start + int(header.num_contact_data)
            for contact in contacts[start:stop]:
                separation = float(contact.separation)
                minimum_separation_m = (
                    separation
                    if minimum_separation_m is None
                    else min(minimum_separation_m, separation)
                )
                lever = np.asarray(contact.position, dtype=np.float64) - plug_origin
                impulse = impulse_sign * np.asarray(
                    contact.impulse, dtype=np.float64
                )
                torque_z_nm += float(np.cross(lever, impulse)[2] / dt)
                point_count += 1
        samples.append(
            {
                "step": step + 1,
                "torque_z_nm": torque_z_nm,
                "point_count": point_count,
                "minimum_separation_m": minimum_separation_m,
                "collider_pair_count": len(collider_pairs),
            }
        )

    tail = samples[-20:]
    final_body_pose = body_view.get_world_poses()
    final_nut_pose = nut_view.get_world_poses()
    progress_deg = (-(FOLLOWER_PHASE_OFFSET_DEG + nut_local_yaw_deg)) % 10.0
    profile_region, expected_radius = _profile(progress_deg)
    result = {
        "nut_local_yaw_deg": nut_local_yaw_deg,
        "positive_coupling_is_negative_yaw": True,
        "positive_coupling_progress_deg": progress_deg,
        "profile_region": profile_region,
        "declared_profile_radius_m": expected_radius,
        "declared_radial_overlap_m": max(
            0.0,
            expected_radius + follower_radius_m - follower_center_radius_m,
        ),
        "steady_median_torque_z_nm": float(
            np.median([sample["torque_z_nm"] for sample in tail])
        ),
        "steady_minimum_torque_z_nm": min(
            sample["torque_z_nm"] for sample in tail
        ),
        "steady_maximum_torque_z_nm": max(
            sample["torque_z_nm"] for sample in tail
        ),
        "steady_median_contact_point_count": int(
            round(float(np.median([sample["point_count"] for sample in tail])))
        ),
        "steady_median_collider_pair_count": int(
            round(
                float(
                    np.median(
                        [sample["collider_pair_count"] for sample in tail]
                    )
                )
            )
        ),
        "steady_median_minimum_separation_m": (
            None
            if any(sample["minimum_separation_m"] is None for sample in tail)
            else float(
                np.median(
                    [sample["minimum_separation_m"] for sample in tail]
                )
            )
        ),
        "disabled_unrelated_collider_count": disabled_count,
        "tooth_only_response_track": tooth_only,
        "base_only_support_track": base_only,
        "follower_shape": follower_shape,
        "follower_radius_m": follower_radius_m,
        "follower_center_radius_m": follower_center_radius_m,
        "follower_paths": follower_paths,
        "initial_body_position_m": np.asarray(initial_body_pose[0][0]).tolist(),
        "initial_body_quaternion_wxyz": np.asarray(
            initial_body_pose[1][0]
        ).tolist(),
        "initial_nut_position_m": np.asarray(initial_nut_pose[0][0]).tolist(),
        "initial_nut_quaternion_wxyz": np.asarray(
            initial_nut_pose[1][0]
        ).tolist(),
        "final_body_position_m": np.asarray(final_body_pose[0][0]).tolist(),
        "final_body_quaternion_wxyz": np.asarray(
            final_body_pose[1][0]
        ).tolist(),
        "final_nut_position_m": np.asarray(final_nut_pose[0][0]).tolist(),
        "final_nut_quaternion_wxyz": np.asarray(
            final_nut_pose[1][0]
        ).tolist(),
        "final_body_velocity": np.asarray(body_view.get_velocities()[0]).tolist(),
        "final_nut_velocity": np.asarray(nut_view.get_velocities()[0]).tolist(),
        "object_pose_write_after_physics_start_count": 0,
    }
    world.stop()
    World.clear_instance()
    return result


def _run(arguments: argparse.Namespace) -> dict[str, Any]:
    samples = []
    angles = (
        DEFAULT_ANGLES_DEG
        if arguments.angles_deg is None
        else tuple(float(value) for value in arguments.angles_deg)
    )
    for angle in angles:
        _emit({"probe_angle_started_deg": angle})
        samples.append(
            _run_angle(
                nut_local_yaw_deg=angle,
                settle_steps=int(arguments.settle_steps),
                stiffness_n_m=float(arguments.stiffness_n_m),
                damping_n_s_m=float(arguments.damping_n_s_m),
                tooth_only=bool(arguments.tooth_only),
                base_only=bool(arguments.base_only),
                follower_shape=str(arguments.follower_shape),
                follower_radius_m=float(arguments.follower_radius_m),
            )
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_id": GENERATOR_ID,
        "role": "a3_modelling_diagnostic_not_formal_acceptance",
        "asset_revision_under_test": "keyed_v3_physical_r11",
        "candidate": {
            "shape": "analytic_round_" + str(arguments.follower_shape),
            "count": 3,
            "radius_m": float(arguments.follower_radius_m),
            "center_radius_m": (
                CAM_BASE_RADIUS_M
                + float(arguments.follower_radius_m)
                - 0.000001
            ),
            "base_preload_m": 0.000001,
            "stiffness_n_m": float(arguments.stiffness_n_m),
            "damping_n_s_m": float(arguments.damping_n_s_m),
            "tooth_only_response_track": bool(arguments.tooth_only),
            "base_only_support_track": bool(arguments.base_only),
        },
        "samples": samples,
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
    portable_root = Path(arguments.kit_portable_root)
    portable_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WARP_CACHE_PATH", str(portable_root / "warp-cache"))
    sys.argv.extend(["--portable-root", str(portable_root)])

    from isaacsim import SimulationApp

    application = SimulationApp(
        {
            "headless": True,
            "hide_ui": True,
            "renderer": "Minimal",
            "multi_gpu": False,
            "fast_shutdown": True,
            "enable_crashreporter": False,
        },
        experience=str(Path(__file__).with_name("d38999_cpu_physics_only.kit")),
    )
    status = 1
    try:
        report = _run(arguments)
        status = 0
    except BaseException as error:
        report = {
            "schema_version": SCHEMA_VERSION,
            "generator_id": GENERATOR_ID,
            "status": "FAILED",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
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
    _emit(report)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
