"""Prepare the current body grasp and source-CAD socket before physics reset.

This module prepares the scene before reset. It neither starts physics nor supplies a
socket pose to the controller.  The returned ``scene`` must replace the caller's
scene before its ordinary evidence binding, contact setup, and preflight.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml


RECEPTACLE_ROOT = "/World/TEVisualHandoff/FixedReceptaclePose"
RECEPTACLE_VISUAL = (
    "artifacts/kcg_connector/isaac/te_j35_engineering_v1/visual/"
    "D38999_20FJ35SN_VISUAL.usdc"
)
RECEPTACLE_MESH = (
    "artifacts/kcg_connector/vision/sam6d_receptacle_current_camera_run21_v1/"
    "D38999_20FJ35SN_VISUAL.obj"
)


def prepare_body_assembly_scene(repository, stage, scene, collision_config_path):
    """Add the fixed socket and replace Body convex hulls with source-CAD SDF.

    Call only before the first ``world.reset()``, immediately after
    ``prepare_dynamic_scene`` and before the existing material/mass readbacks.
    ``collision_config_path`` is the current light-contact YAML; its collision
    settings are reused without changing force limits or restoring its old
    grasp/search plan.  The fixture already in ``scene`` is reused unchanged.
    """
    import omni.timeline
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    from te_foundationpose_handoff_runtime import (
        _author_key_entry_collisions,
        _validate_receptacle_visual_binding,
    )

    timeline = omni.timeline.get_timeline_interface()
    if timeline.is_playing() or timeline.get_current_time() != 0.0:
        raise RuntimeError("assembly geometry must be prepared before physics starts")
    repository = Path(repository).resolve()
    config_path = Path(collision_config_path)
    if not config_path.is_absolute():
        config_path = repository / config_path
    config_path = config_path.resolve()
    config_path.relative_to(repository)
    probe = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    authorization = probe.get("authorization", {})
    if not (
        probe.get("schema_version") == "te_light_contact_key_entry_v1"
        and authorization.get("simulation_only") is True
        and authorization.get("hardware_authorized") is False
        and probe.get("boundaries", {}).get("online_object_or_contact_truth_allowed") is False
        and probe.get("boundaries", {}).get("post_start_object_pose_write_allowed") is False
        and probe["collision"]["body_representation"] == "source_cad_sparse_sdf"
        and probe["collision"]["receptacle_representation"]
        == "stationary_source_cad_triangle_mesh"
    ):
        raise ValueError("assembly source geometry or simulation boundary differs")
    body_path, nut_path = tuple(scene["part_prim_paths"])
    table_collision_audit = None
    if probe.get("collision", {}).get("table_representation") == "source_equivalent_triangle_box":
        table_prim = stage.GetPrimAtPath(str(scene["roots"]["table"]))
        if not table_prim.IsA(UsdGeom.Cube) or table_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError("equivalent table contact test requires the original static Cube")
        cube_size = float(UsdGeom.Cube(table_prim).GetSizeAttr().Get())
        bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        before = bounds.ComputeWorldBound(table_prim).ComputeAlignedBox()
        half = cube_size / 2.
        points = [Gf.Vec3f(x*half, y*half, z*half) for x, y, z in
                  [(-1,-1,-1), (1,-1,-1), (1,1,-1), (-1,1,-1),
                   (-1,-1,1), (1,-1,1), (1,1,1), (-1,1,1)]]
        faces = [0,2,1, 0,3,2, 4,5,6, 4,6,7, 0,1,5, 0,5,4,
                 1,2,6, 1,6,5, 2,3,7, 2,7,6, 3,0,4, 3,4,7]
        table_prim.SetTypeName("Mesh")
        table_mesh = UsdGeom.Mesh(table_prim)
        table_mesh.CreatePointsAttr(points)
        table_mesh.CreateFaceVertexCountsAttr([3]*12)
        table_mesh.CreateFaceVertexIndicesAttr(faces)
        table_mesh.CreateSubdivisionSchemeAttr("none")
        table_mesh.CreateExtentAttr([Gf.Vec3f(-half), Gf.Vec3f(half)])
        UsdPhysics.MeshCollisionAPI.Apply(table_prim).CreateApproximationAttr("none")
        PhysxSchema.PhysxTriangleMeshCollisionAPI.Apply(table_prim).CreateWeldToleranceAttr(0.)
        bounds.Clear()
        after = bounds.ComputeWorldBound(table_prim).ComputeAlignedBox()
        if before.GetMin() != after.GetMin() or before.GetMax() != after.GetMax():
            raise RuntimeError("equivalent table mesh changed the original bounds")
        table_collision_audit = {
            "scope": "CONTACT_DISPATCH_TEST_IDENTICAL_BOX_SURFACE",
            "prim_path": str(table_prim.GetPath()), "before": "STATIC_CUBE",
            "after": "STATIC_TWELVE_TRIANGLE_BOX", "local_cube_size": cube_size,
            "world_bounds_min_m": list(after.GetMin()), "world_bounds_max_m": list(after.GetMax()),
            "transform_material_static_role_and_box_surface_preserved": True,
            "hypothesis": "Mesh-versus-SDF contact dispatch may avoid the observed CPU box contact loss and penetration",
            "physical_improvement_validated": False,
        }
    initial_part = probe.get("initial_grasp_part", "Body")
    if initial_part not in ("Body", "CouplingNut"):
        raise ValueError("the declared initial grasp part is unsupported")
    initial_path = body_path if initial_part == "Body" else nut_path
    if tuple(scene["legal_grasp_contact_paths"]) != (initial_path,):
        raise ValueError("the selected scene contact policy differs from the declared initial grasp part")
    if stage.GetPrimAtPath(RECEPTACLE_ROOT).IsValid() or stage.GetPrimAtPath(
        body_path + "/SourceCadCollision"
    ).IsValid():
        raise ValueError("assembly geometry is already authored in this stage")
    fixture_path = str(scene["roots"]["fixture"])
    fixture_prim = stage.GetPrimAtPath(fixture_path)
    if not fixture_prim.IsValid():
        raise ValueError("the existing fixed socket fixture is missing")

    # These are authoring-time fixture/asset dimensions, never an online pose
    # observation.  Seat the same source socket on the unchanged fixture top.
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
    )
    fixture_box = bbox_cache.ComputeWorldBound(fixture_prim).ComputeAlignedBox()
    fixture_min = np.asarray(fixture_box.GetMin(), dtype=np.float64)
    fixture_max = np.asarray(fixture_box.GetMax(), dtype=np.float64)
    fixture = {
        "center_world_m": ((fixture_min + fixture_max) / 2.0).tolist(),
        "size_m": (fixture_max - fixture_min).tolist(),
    }
    visual_path, mesh_path = repository / RECEPTACLE_VISUAL, repository / RECEPTACLE_MESH
    source_stage = Usd.Stage.Open(str(visual_path))
    if source_stage is None:
        raise ValueError("source receptacle USD is unavailable")
    source_box = bbox_cache.ComputeWorldBound(source_stage.GetPseudoRoot()).ComputeAlignedBox()
    socket_position = np.asarray(fixture["center_world_m"], dtype=np.float64)
    socket_position[2] = fixture_max[2] - float(source_box.GetMin()[2])
    geometry_binding = _validate_receptacle_visual_binding(
        visual_path=visual_path, collision_mesh_path=mesh_path,
        position_world_m=socket_position, fixture=fixture, Usd=Usd, UsdGeom=UsdGeom,
    )
    socket = UsdGeom.Xform.Define(stage, RECEPTACLE_ROOT)
    socket.AddTranslateOp().Set(Gf.Vec3d(*map(float, socket_position)))
    visual = UsdGeom.Xform.Define(stage, RECEPTACLE_ROOT + "/OfficialVisual")
    visual.GetPrim().GetReferences().AddReference(str(visual_path))
    collision_report = _author_key_entry_collisions(
        stage, repository, scene, RECEPTACLE_ROOT, probe,
        Gf=Gf, Usd=Usd, UsdGeom=UsdGeom, UsdPhysics=UsdPhysics,
        PhysxSchema=PhysxSchema, UsdShade=UsdShade,
    )
    if any(prim.HasAPI(UsdPhysics.RigidBodyAPI) for prim in Usd.PrimRange(socket.GetPrim())):
        raise ValueError("source receptacle must remain the existing fixed-fixture model")

    # Include the geometry sources and the authoring implementation in the
    # normal preflight binding.  Old no-socket preflights must therefore fail
    # the runner's existing comparison, without changing any stored result.
    evidence_paths = tuple(dict.fromkeys((
        *scene["evidence_paths"], config_path, Path(__file__).resolve(),
        Path(__file__).with_name("te_foundationpose_handoff_runtime.py").resolve(),
        Path(__file__).with_name("build_te_free_split_plug.py").resolve(),
        Path(collision_report["body_source"]), visual_path, mesh_path,
        *((Path(collision_report["nut_source"]),) if "nut_source" in collision_report else ()),
    )))
    prepared_scene = dict(scene)
    prepared_scene.update({
        "roots": {**scene["roots"], "receptacle": RECEPTACLE_ROOT},
        "evidence_paths": evidence_paths,
        "environment_scope": "SHARED_FINITE_TABLE_FIXED_SOURCE_CAD_SOCKET_BODY_SDF",
    })
    result = {
        "scene": prepared_scene,
        "receptacle_root": RECEPTACLE_ROOT,
        "receptacle_collision_path": collision_report["receptacle_collision"],
        "fixture": {"prim_path": fixture_path, **fixture},
        "contact_recording": {
            "additional_required_contact_filter_paths": (
                [collision_report["receptacle_collision"]]
                if probe.get("evaluation_recording", {}).get("socket_friction", False) else []),
            "additional_rigid_sensor_paths": [],
            "normal_contacts": "EXISTING_ROBOT_BODY_NUT_SENSORS_REPORT_ALL_OTHER_ACTOR_PATHS",
            "optional_socket_friction_filter_paths": [RECEPTACLE_ROOT],
            "legal_grasp_contact_paths": [initial_path],
            "body_socket_pair": [body_path, RECEPTACLE_ROOT],
            "nut_socket_pair": [nut_path, RECEPTACLE_ROOT],
            "robot_socket_contact_is_forbidden": True,
        },
        "report": {
            "simulation_only": True,
            "hardware_authorized": False,
            "initial_grasp_part": initial_part,
            "geometry": geometry_binding,
            "collision": collision_report,
            "nut_source_cad_sdf_enabled": "nut_source" in collision_report,
            "source_mass_material_and_joint_drive_parameters_changed": False,
            "controller_contact_force_override_configured": bool(
                probe.get("wrist_contact_force_probe", {}).get("phase_limits_n")),
            "socket_support": "EXISTING_FIXED_FIXTURE_NOT_A_RELEASE_LOCKING_PROOF",
            "socket_initial_position_world_m": socket_position.tolist(),
            "socket_position_role": "SCENE_AUTHORING_AND_POSTRUN_EVALUATION_ONLY",
            "online_socket_pose_must_come_from_current_rgbd": True,
            "fresh_same_scene_preflight_required": True,
            "physical_grasp_or_key_entry_verified": False,
            "thread_rotation_to_axial_progress_represented": False,
        },
    }
    if probe["collision"].get("representative_inner_thread_manifest"):
        from te_representative_thread_scene import install_representative_inner_thread
        thread = install_representative_inner_thread(
            repository, stage, result["report"],
            probe["collision"]["representative_inner_thread_manifest"])
        result["scene"]["evidence_paths"] = tuple(dict.fromkeys((
            *result["scene"]["evidence_paths"],
            Path(thread["manifest_path"]), Path(thread["mesh_path"]),
            Path(__file__).with_name("te_representative_thread_scene.py").resolve(),
        )))
        result["report"]["thread_rotation_to_axial_progress_represented"] = True
        result["report"]["thread_representation_scope"] = (
            "GEOMETRIC_CONTACT_ONLY_DYNAMIC_ADVANCE_NOT_YET_VERIFIED")
    if table_collision_audit is not None:
        result["report"]["table_collision_representation"] = table_collision_audit
    solver = probe.get("passive_joint_solver", {})
    if solver.get("representation") == "floating_reduced_coordinate_articulation":
        body, nut = stage.GetPrimAtPath(body_path), stage.GetPrimAtPath(nut_path)
        plug_root = body.GetParent()
        joints = [p for p in Usd.PrimRange(plug_root) if p.IsA(UsdPhysics.Joint)]
        if len(joints) != 1 or not joints[0].IsA(UsdPhysics.RevoluteJoint):
            raise ValueError("the floating plug requires exactly its original revolute joint")
        joint = UsdPhysics.Joint(joints[0])
        if (joint.GetBody0Rel().GetTargets() != [body.GetPath()]
                or joint.GetBody1Rel().GetTargets() != [nut.GetPath()]):
            raise ValueError("the original body-to-nut topology differs; no world attachment is allowed")
        if any(p.HasAPI(UsdPhysics.ArticulationRootAPI) for p in Usd.PrimRange(plug_root)):
            raise ValueError("the plug already has an articulation root")
        if any(UsdPhysics.RigidBodyAPI(p).GetKinematicEnabledAttr().Get() for p in (body, nut)):
            raise ValueError("both plug parts must retain dynamic rigid-body motion")
        mass_before = {str(p.GetPath()): {
            a.GetName(): str(a.Get()) for a in p.GetAttributes()
            if a.GetName() in ("physics:mass", "physics:centerOfMass", "physics:diagonalInertia", "physics:principalAxes")
        } for p in (body, nut)}
        joint_before = {a.GetName(): str(a.Get()) for a in joints[0].GetAttributes()}
        # A floating articulation solves the existing internal joint in reduced
        # coordinates. No joint to the world or another object is authored.
        UsdPhysics.ArticulationRootAPI.Apply(body)
        articulation = PhysxSchema.PhysxArticulationAPI.Apply(body)
        articulation.CreateSolverPositionIterationCountAttr(int(solver["position_iterations"]))
        articulation.CreateSolverVelocityIterationCountAttr(int(solver["velocity_iterations"]))
        mass_after = {str(p.GetPath()): {
            a.GetName(): str(a.Get()) for a in p.GetAttributes()
            if a.GetName() in ("physics:mass", "physics:centerOfMass", "physics:diagonalInertia", "physics:principalAxes")
        } for p in (body, nut)}
        if mass_before != mass_after or joint_before != {a.GetName(): str(a.Get()) for a in joints[0].GetAttributes()}:
            raise RuntimeError("solver preparation altered source mass, inertia or joint drive parameters")
        result["report"]["passive_joint_solver"] = {
            **solver, "root_path": body_path, "internal_joint_path": str(joints[0].GetPath()),
            "floating_base": True, "world_joint_added": False,
            "joint_degrees_of_freedom_or_drive_changed": False,
            "mass_and_inertia_unchanged": True, "source_assets_modified": False,
            "physical_resistance_under_robot_load_verified": False,
        }
        result["report"]["joint_solver_representation_changed"] = True
        if solver.get("brake_model") == "native_static_dynamic_friction":
            from omni.physx.bindings._physx import (
                JOINT_AXIS_API, JOINT_AXIS_ATTR_STATIC_FRICTION_EFFORT_ANGULAR,
                JOINT_AXIS_ATTR_DYNAMIC_FRICTION_EFFORT_ANGULAR,
                JOINT_AXIS_ATTR_VISCOUS_FRICTION_COEFFICIENT_ANGULAR)
            drive = UsdPhysics.DriveAPI.Get(joints[0], "angular")
            if not drive or float(drive.GetStiffnessAttr().Get()) != 0.:
                raise ValueError("the source must have its original zero-stiffness brake drive")
            resistance = float(scene["joint_rotational_resistance"]["assumed_resisting_torque_nm"])
            if (resistance <= 0 or not np.isclose(float(drive.GetMaxForceAttr().Get()), resistance,
                                                rtol=1e-6, atol=1e-9)
                    or float(drive.GetTargetVelocityAttr().Get()) != 0.):
                raise ValueError("native friction must preserve the source assumed resistance")
            # Explicit constitutive variant, authored before reset. Disable the
            # old saturated velocity drive so its torque is not added a second
            # time. Preserve the source asset, topology, mass and inertia.
            drive.CreateDampingAttr(0.)
            drive.CreateMaxForceAttr(0.)
            joints[0].ApplyAPI(JOINT_AXIS_API, UsdPhysics.Tokens.angular)
            friction_values = {
                JOINT_AXIS_ATTR_STATIC_FRICTION_EFFORT_ANGULAR: resistance,
                JOINT_AXIS_ATTR_DYNAMIC_FRICTION_EFFORT_ANGULAR: resistance,
                JOINT_AXIS_ATTR_VISCOUS_FRICTION_COEFFICIENT_ANGULAR: 0.,
            }
            for name, value in friction_values.items():
                joints[0].CreateAttribute(name, Sdf.ValueTypeNames.Float).Set(value)
            actual = {name: float(joints[0].GetAttribute(name).Get()) for name in friction_values}
            if any(not np.isclose(actual[name], value, rtol=1e-6, atol=1e-9)
                   for name, value in friction_values.items()):
                raise RuntimeError("native joint friction authoring readback differs")
            result["scene"]["joint_rotational_resistance"] = {
                "model": "NATIVE_STATIC_DYNAMIC_COULOMB_FRICTION",
                "assumed_resisting_torque_nm": resistance,
                "static_friction_effort_nm": resistance,
                "dynamic_friction_effort_nm": resistance,
                "viscous_friction_coefficient_nm_s_per_deg": 0.,
                "disabled_velocity_drive_maximum_torque_nm": float(drive.GetMaxForceAttr().Get()),
                "disabled_velocity_drive_damping_nm_s_per_deg": float(drive.GetDampingAttr().Get()),
                "original_asset_drive_readback": scene["joint_rotational_resistance"],
                "source_asset_modified": False,
                "hardware_resistance_calibrated": False,
            }
            result["report"].update(
                source_mass_material_and_joint_drive_parameters_changed=True,
                source_mass_material_parameters_changed=False,
                joint_resistance_constitutive_representation_changed=True,
                native_joint_friction=result["scene"]["joint_rotational_resistance"])
            result["report"]["passive_joint_solver"]["joint_degrees_of_freedom_or_drive_changed"] = True
            result["report"]["passive_joint_solver"]["joint_degrees_of_freedom_changed"] = False
    if probe.get("physics_numerics", {}).get("disable_plug_sleep") is True:
        sleep_audit = []
        for path in (body_path, nut_path):
            prim = stage.GetPrimAtPath(path)
            rigid = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            attr = rigid.GetSleepThresholdAttr()
            row = {"prim_path": path, "rigid_sleep_threshold_before_m2_s2": attr.Get(),
                   "rigid_sleep_threshold_was_authored": attr.HasAuthoredValueOpinion()}
            rigid.CreateSleepThresholdAttr(0.)
            row["rigid_sleep_threshold_after_m2_s2"] = rigid.GetSleepThresholdAttr().Get()
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                api = PhysxSchema.PhysxArticulationAPI.Apply(prim)
                row["articulation_sleep_threshold_before_m2_s2"] = api.GetSleepThresholdAttr().Get()
                api.CreateSleepThresholdAttr(0.)
                row["articulation_sleep_threshold_after_m2_s2"] = api.GetSleepThresholdAttr().Get()
            if row["rigid_sleep_threshold_after_m2_s2"] != 0.:
                raise RuntimeError("plug sleep authoring readback differs")
            sleep_audit.append(row)
        result["report"]["plug_sleep_audit"] = {
            "purpose": "Prevent automatic sleep from masking post-release dynamics",
            "before_physics_start_only": True, "source_assets_modified": False,
            "per_step_wake_or_pose_commands_used": False, "parts": sleep_audit}
    return result
