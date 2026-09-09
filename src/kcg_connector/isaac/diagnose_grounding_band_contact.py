#!/usr/bin/env python3
"""Clamped source-band geometry coupons for native contact-law sensitivity.

The explicit world clamps isolate material response at a prescribed compression;
they are laboratory fixtures, not an assembly controller or assembly evidence.
No robot is present and no source asset is modified. Stiffness is a diagnostic
input, not an identified manufacturer spring constant.
"""

import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--stiffness-n-m", type=float, required=True)
parser.add_argument("--depths-m", type=float, nargs="+", default=[.012])
parser.add_argument("--yaws-deg", type=float, nargs="+", default=[0., .4, 360/70/2])
parser.add_argument("--damping-ns-m", type=float, default=5.)
parser.add_argument("--include-rigid-core", action="store_true",
                    help="Test source Body partition/material authoring with unchanged source Body mass")
parser.add_argument("--collision-sectors", type=Path,
                    help="Source radial sectors as multiple compliant colliders on the SAME rigid body")
parser.add_argument("--external-forces-every-iteration", type=int, choices=(0, 1))
parser.add_argument("--preserve-non-socket-collision", action="store_true")
args = parser.parse_args()
if args.collision_sectors and args.include_rigid_core:
    parser.error("partitioned-collider diagnostic currently isolates only the band")
if args.preserve_non_socket_collision and not args.include_rigid_core:
    parser.error("pair-specific representation requires the complete rigid core")
args.output.mkdir(parents=True, exist_ok=False)

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "multi_gpu": False})
failed = False
try:
    import carb
    import numpy as np
    import omni.usd
    import trimesh
    from scipy.spatial.transform import Rotation
    from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade, Vt
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleRigidPrim
    from isaacsim.core.experimental.prims import RigidPrim as TensorRigidPrim
    from isaacsim.core.simulation_manager import SimulationManager
    from omni.physx.bindings._physx import SETTING_DISABLE_CONTACT_PROCESSING

    if not np.isfinite(args.stiffness_n_m) or args.stiffness_n_m <= 0:
        raise ValueError("diagnostic stiffness must be finite and positive")
    repo = Path(__file__).resolve().parents[3]
    band_source = repo / "artifacts/kcg_connector/isaac/te_full_assembly_20260905/grounding_band_geometry_03/circumferential_band.npz"
    socket_source = repo / "artifacts/kcg_connector/vision/sam6d_receptacle_current_camera_run21_v1/D38999_20FJ35SN_VISUAL.obj"
    band = np.load(band_source)
    sectors = json.loads(args.collision_sectors.read_text())["sectors"] if args.collision_sectors else None
    body_source = repo / "artifacts/carts_grasp/CARTS_GRASP_V1/object_models/te_j35/plug_body_visual_mesh.npz"
    source_body = np.load(body_source) if args.include_rigid_core else None
    mass_source = repo / "artifacts/kcg_connector/isaac/te_j35_free_split_tabletop_real_mass_resistance_0p020_v2/TE_J35_FREE_SPLIT_PLUG_V1.usdc"
    if args.include_rigid_core:
        mass_stage = Usd.Stage.Open(str(mass_source))
        source_mass = UsdPhysics.MassAPI(mass_stage.GetPrimAtPath("/TE_J35FreeSplitPlug/Body"))
    socket_mesh = trimesh.load(socket_source, force="mesh", process=False)
    dt, damping = 1/240, args.damping_ns_m
    cases = [(depth, yaw) for depth in args.depths_m for yaw in args.yaws_deg]
    if (not cases or len(cases) > 9 or not np.isfinite(cases).all()
            or not np.isfinite(damping) or damping < 0
            or any(not 0 < depth < .016 for depth, _ in cases)):
        raise ValueError("invalid finite laboratory coupon configuration")
    carb.settings.get_settings().set_bool(SETTING_DISABLE_CONTACT_PROCESSING, False)
    SimulationManager.set_physics_sim_device("cuda:0")
    world = World(stage_units_in_meters=1., physics_dt=dt, rendering_dt=1/60,
                  backend="numpy", device="cuda:0", sim_params={"use_gpu_pipeline": True})
    stage = omni.usd.get_context().get_stage()
    physics = UsdPhysics.Scene.Get(stage, world.get_physics_context().prim_path)
    physics.CreateGravityMagnitudeAttr(0.)
    scene_api = PhysxSchema.PhysxSceneAPI.Apply(physics.GetPrim())
    scene_api.CreateSolverTypeAttr("TGS")
    scene_api.CreateMinVelocityIterationCountAttr(1)
    scene_api.CreateMaxVelocityIterationCountAttr(1)
    if args.external_forces_every_iteration is not None:
        scene_api.CreateEnableExternalForcesEveryIterationAttr(bool(args.external_forces_every_iteration))

    def material(path, stiffness):
        value = UsdShade.Material.Define(stage, path)
        properties = UsdPhysics.MaterialAPI.Apply(value.GetPrim())
        properties.CreateStaticFrictionAttr(0.)
        properties.CreateDynamicFrictionAttr(0.)
        properties.CreateRestitutionAttr(0.)
        if stiffness:
            api = PhysxSchema.PhysxMaterialAPI.Apply(value.GetPrim())
            api.CreateCompliantContactStiffnessAttr(stiffness)
            api.CreateCompliantContactDampingAttr(damping)
            api.CreateCompliantContactAccelerationSpringAttr(False)
        return value

    rigid_material = material("/World/RigidMaterial", 0.)
    soft_material = material("/World/BandDiagnosticMaterial", args.stiffness_n_m)

    def mesh(path, vertices, faces, position, quaternion, mat, sdf=False, convex=False):
        shape = UsdGeom.Mesh.Define(stage, path)
        shape.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(vertices*(1000. if convex else 1.), dtype=np.float32)))
        shape.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, np.int32)))
        shape.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(faces, np.int32).ravel()))
        shape.CreateSubdivisionSchemeAttr("none")
        shape.AddTranslateOp().Set(Gf.Vec3d(*map(float, position)))
        shape.AddOrientOp().Set(Gf.Quatf(float(quaternion[3]), Gf.Vec3f(*map(float, quaternion[:3]))))
        if convex:
            shape.AddScaleOp().Set(Gf.Vec3f(.001))
        prim = shape.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull" if convex else "sdf" if sdf else "none")
        if convex:
            hull = PhysxSchema.PhysxConvexHullCollisionAPI.Apply(prim)
            hull.CreateHullVertexLimitAttr(255)
            hull.CreateMinThicknessAttr(.001)
        collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        collision.CreateRestOffsetAttr(0.)
        collision.CreateContactOffsetAttr(.0001)
        PhysxSchema.PhysxTriangleMeshCollisionAPI.Apply(prim).CreateWeldToleranceAttr(0.)
        if sdf:
            api = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim)
            api.CreateSdfResolutionAttr(1024)
            api.CreateSdfSubgridResolutionAttr(6)
            api.CreateSdfNarrowBandThicknessAttr(.002)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.strongerThanDescendants, "physics")
        return prim

    band_paths, socket_paths, views, initial_positions, model_reports = [], [], [], [], []
    for i, (depth, yaw) in enumerate(cases):
        x = .12*i
        socket_path, band_path = f"/World/Socket{i}", f"/World/Band{i}"
        mesh(socket_path, socket_mesh.vertices*.001, socket_mesh.faces, [x, 0., 0.], [0., 0., 0., 1.], rigid_material)
        quaternion = (Rotation.from_euler("z", yaw, degrees=True)*Rotation.from_euler("y", 180, degrees=True)).as_quat()
        position = [x, 0., -depth]
        if args.include_rigid_core:
            body_xform = UsdGeom.Xform.Define(stage, band_path)
            body_xform.AddTranslateOp().Set(Gf.Vec3d(*map(float, position)))
            body_xform.AddOrientOp().Set(Gf.Quatf(float(quaternion[3]), Gf.Vec3f(*map(float, quaternion[:3]))))
            prim = body_xform.GetPrim()
            mesh(band_path+"/SourceCadCollision", source_body["vertices_m"], source_body["faces"],
                 [0., 0., 0.], [0., 0., 0., 1.], rigid_material, sdf=True)
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                rigid_material, UsdShade.Tokens.strongerThanDescendants, "physics")
        elif sectors:
            body_xform = UsdGeom.Xform.Define(stage, band_path)
            body_xform.AddTranslateOp().Set(Gf.Vec3d(*map(float, position)))
            body_xform.AddOrientOp().Set(Gf.Quatf(float(quaternion[3]), Gf.Vec3f(*map(float, quaternion[:3]))))
            prim = body_xform.GetPrim()
            for sector in sectors:
                with np.load(sector["path"]) as data:
                    mesh(band_path+f"/Sector{sector['index']:03d}", data["vertices_m"], data["faces"],
                         [0., 0., 0.], [0., 0., 0., 1.], soft_material, convex=True)
        else:
            prim = mesh(band_path, band["vertices_m"], band["faces"], position, quaternion, soft_material, sdf=True)
        UsdPhysics.RigidBodyAPI.Apply(prim)
        properties = UsdPhysics.MassAPI.Apply(prim)
        properties.CreateMassAttr(source_mass.GetMassAttr().Get() if args.include_rigid_core else .1)
        properties.CreateCenterOfMassAttr(source_mass.GetCenterOfMassAttr().Get()
                                         if args.include_rigid_core else Gf.Vec3f(0., 0., -.0108))
        properties.CreateDiagonalInertiaAttr(source_mass.GetDiagonalInertiaAttr().Get()
                                            if args.include_rigid_core else Gf.Vec3f(.0001))
        if args.include_rigid_core:
            properties.CreatePrincipalAxesAttr(source_mass.GetPrincipalAxesAttr().Get())
        if args.include_rigid_core:
            from te_grounding_band_scene import install_grounding_band_contact_model
            model_reports.append(install_grounding_band_contact_model(
                stage, band_path, band_source.parent / "geometry_manifest.json",
                stiffness_n_m=args.stiffness_n_m, damping_ns_m=damping,
                socket_collision_path=socket_path if args.preserve_non_socket_collision else None))
        api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        api.CreateSolverPositionIterationCountAttr(32)
        api.CreateSolverVelocityIterationCountAttr(1)
        api.CreateSleepThresholdAttr(0.)
        PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
        joint = UsdPhysics.FixedJoint.Define(stage, band_path+"ExplicitLaboratoryClamp")
        joint.CreateBody1Rel().SetTargets([band_path])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*map(float, position)))
        joint.CreateLocalRot0Attr(Gf.Quatf(float(quaternion[3]), Gf.Vec3f(*map(float, quaternion[:3]))))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.))
        joint.CreateLocalRot1Attr(Gf.Quatf(1.))
        band_paths.append(band_path)
        socket_paths.append(socket_path)
        views.append(world.scene.add(SingleRigidPrim(prim_path=band_path, name=f"band{i}", reset_xform_properties=False)))
        initial_positions.append(position)
    maximum_contact_records = 16384 if sectors else 4096
    contact_view = TensorRigidPrim(band_paths, resolve_paths=False,
                                  contact_filter_paths=socket_paths, max_contact_count=maximum_contact_records)
    stage.GetRootLayer().Export(str(args.output / "clamped_coupons_before_reset.usda"))
    world.reset()
    if not contact_view.is_physics_tensor_entity_valid():
        raise RuntimeError("native coupon contact view is invalid")
    rows = []
    for step in range(240):
        world.step(render=False)
        impulse, point, normal, separation, counts, starts, _ = contact_view.get_raw_contact_data(dt=1.)
        impulse, point, normal, separation = (x.numpy() for x in (impulse, point, normal, separation))
        impulse, separation = impulse.ravel(), separation.ravel()
        per_case = []
        for i, (count, start) in enumerate(zip(counts.numpy().ravel(), starts.numpy().ravel())):
            start, count = int(start), int(count)
            sl = slice(start, start+count)
            forces = impulse[sl, None]*normal[sl]/dt
            radial = point[sl, :2]-np.array([initial_positions[i][0], 0.])
            radial /= np.maximum(np.linalg.norm(radial, axis=1)[:, None], 1e-12)
            position, _ = views[i].get_world_pose()
            position = position.cpu().numpy() if hasattr(position, "cpu") else np.asarray(position)
            torque = np.cross(point[sl]-np.asarray(initial_positions[i]), forces).sum(0)
            per_case.append([count, float(np.linalg.norm(forces, axis=1).sum()),
                float(-np.sum(forces[:, :2]*radial)),
                float(np.maximum(-separation[sl], 0.).sum()),
                float(np.linalg.norm(position-initial_positions[i])),
                *forces.sum(0).tolist(), *torque.tolist()])
        rows.append(per_case)
    values = np.asarray(rows)
    result = {"scope": "EXPLICITLY_CLAMPED_SOURCE_GEOMETRY_COUPONS_NOT_ASSEMBLY",
              "band_geometry": str(band_source), "socket_geometry": str(socket_source),
              "source_assets_modified": False, "robot_present": False,
              "source_rigid_core_included": args.include_rigid_core,
              "source_collision_sector_manifest": str(args.collision_sectors) if sectors else None,
              "collision_sector_count_per_rigid_body": len(sectors) if sectors else 1,
              "additional_rigid_bodies_or_radial_joints": False,
              "compliance_model": "NATIVE_PER_CONTACT_MATERIAL_PENALTY_NOT_WHOLE_RING_STIFFNESS",
              "maximum_contact_records": maximum_contact_records,
              "external_forces_every_iteration": scene_api.GetEnableExternalForcesEveryIterationAttr().Get(),
              "source_body": str(body_source) if args.include_rigid_core else None,
              "source_body_mass_properties_asset": str(mass_source) if args.include_rigid_core else None,
              "local_model_authoring_reports": model_reports,
              "world_clamps": "EXPLICIT_LABORATORY_FIXED_JOINTS_ISOLATE_CONTACT_RESPONSE",
              "post_start_object_pose_writes": False, "manufacturer_stiffness_identified": False,
              "material_stiffness_n_m": args.stiffness_n_m, "material_damping_ns_m": damping,
              "diagnostic_friction": 0., "gravity": 0.,
              "physics_dt_s": dt, "source_mesh_facet_angle_test_not_physical_finger_count": True,
              "columns": ["native_contact_count", "normal_force_magnitudes_sum_n", "inward_radial_force_sum_n",
                          "negative_contact_separation_magnitudes_sum_m", "clamp_position_error_m",
                          "world_fx_n", "world_fy_n", "world_fz_n",
                          "world_tx_nm", "world_ty_nm", "world_tz_nm"],
              "cases": [{"body_depth_m": depth, "body_yaw_deg": yaw,
                         "final_half_second_mean": values[-120:, i].mean(0).tolist(),
                         "final_half_second_minimum": values[-120:, i].min(0).tolist(),
                         "final_half_second_maximum": values[-120:, i].max(0).tolist()}
                        for i, (depth, yaw) in enumerate(cases)]}
    np.savez_compressed(args.output / "samples.npz", values=values)
    (args.output / "result.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2), flush=True)
except Exception:
    failed = True
    import traceback
    failure = traceback.format_exc()
    (args.output / "error.txt").write_text(failure)
    print(failure, flush=True)
finally:
    app.close(exit_code=1 if failed else 0)
