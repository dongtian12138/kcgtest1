#!/usr/bin/env python3
"""Explicitly clamped radial-spring coupon; no robot, Nut or assembly claim.

Only the source band envelope collides. An explicit laboratory fixture carries
passive radial elements. Optional axial loading commands only the laboratory
joint; source object poses and passive radial targets are never rewritten after
the simulation starts. Element count is numerical, not a TE finger count.
"""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--geometry", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--depth-m", type=float, default=.012)
parser.add_argument("--collider-mode", choices=("sdf", "convex_decomposition", "prepared_convex", "convex_hull", "peak_sphere"), default="sdf")
parser.add_argument("--sector-index", type=int, help="Isolate one original sector with its unchanged local K/D and mass")
parser.add_argument("--collision-sector-index", type=int, nargs="+", help="Keep the full joint tree but enable only these sector colliders")
parser.add_argument("--external-forces-every-iteration", type=int, choices=(0, 1))
parser.add_argument("--approach-from-depth-m", type=float,
                    help="Declared axial laboratory drive starts shallower and advances at at most 1 mm/s")
parser.add_argument("--velocity-iterations", type=int, choices=(0, 1), default=1)
parser.add_argument("--solver", choices=("TGS", "PGS"), default="TGS")
parser.add_argument("--ramped-approach", action="store_true")
parser.add_argument("--mesh-local-scale", type=float, choices=(1., 1000.), default=1.,
                    help="Cook sector mesh coordinates at a larger local scale with inverse mesh transform; world geometry unchanged")
parser.add_argument("--physics-dt", type=float, default=1/240)
parser.add_argument("--contact-last", type=int, choices=(0, 1))
parser.add_argument("--joint-model", choices=("articulation", "regular_d6"), default="articulation")
parser.add_argument("--position-iterations", type=int, default=32)
parser.add_argument("--compensate-drive-discretization", action="store_true")
parser.add_argument("--sphere-outer-offset-m", type=float, default=0.)
parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
parser.add_argument("--cfm-scale", type=float, choices=(0., .025))
parser.add_argument("--gpu-partitions", type=int, choices=(8, 32))
parser.add_argument("--omit-lab-velocity-constraint", action="store_true")
parser.add_argument("--sdf-resolution", type=int, choices=(512, 1024), default=512)
parser.add_argument("--hull-vertex-limit", type=int, choices=(64, 255), default=64)
parser.add_argument("--center-sector-frames", action="store_true",
                    help="Equivalent coordinates: sector origins at their COM; retain world shapes, masses and joint anchors")
args = parser.parse_args()
if args.solver == "PGS" and args.external_forces_every_iteration == 1:
    parser.error("PGS does not support external forces on every iteration")
if args.ramped_approach and args.approach_from_depth_m is None:
    parser.error("ramped approach requires a shallower start depth")
if args.joint_model == "regular_d6" and args.approach_from_depth_m is not None:
    parser.error("regular-joint diagnostic currently uses only a fixed-core static load")
if not 1 <= args.position_iterations <= 255:
    parser.error("position iteration count must be in the native supported range")
if args.compensate_drive_discretization and args.solver != "TGS":
    parser.error("inverse implicit-drive coefficient diagnostic requires TGS")
if abs(args.sphere_outer_offset_m) > .0002 or (args.sphere_outer_offset_m and args.collider_mode != "peak_sphere"):
    parser.error("sphere offset is a bounded analytic-contact diagnostic only")
if args.omit_lab_velocity_constraint and not args.ramped_approach:
    parser.error("the lab velocity constraint can only be omitted with a bounded time-ramped target")
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
    from pxr import Gf, PhysxSchema, PhysicsSchemaTools, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils, Vt
    from scipy.spatial import ConvexHull
    from omni.physx import get_physx_cooking_interface, get_physx_simulation_interface
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
    from isaacsim.core.utils.types import ArticulationAction
    import warp as wp
    from isaacsim.core.experimental.prims import RigidPrim as TensorRigidPrim
    from isaacsim.core.simulation_manager import SimulationManager
    from omni.physx.bindings._physx import SETTING_DISABLE_CONTACT_PROCESSING

    repo = Path(__file__).resolve().parents[3]
    root = repo / "artifacts/kcg_connector/isaac/te_full_assembly_20260905"
    geometry = json.loads(args.geometry.read_text())
    recipe = json.loads((root / "grounding_band_representative_law_proposal_v1.json").read_text())
    if not (0 < args.depth_m < .016) or geometry["element_count"] != len(geometry["sectors"]):
        raise ValueError("invalid bounded coupon geometry/depth")
    n = geometry["element_count"]
    elements = geometry["sectors"]
    if args.sector_index is not None:
        elements = [element for element in elements if element["index"] == args.sector_index]
        if len(elements) != 1:
            raise ValueError("requested sector does not exist")
    if args.collision_sector_index is not None and not set(args.collision_sector_index) <= {e["index"] for e in elements}:
        raise ValueError("requested collision sector does not exist")
    dt, travel = args.physics_dt, .0004
    if not 0 < dt <= 1/240+1e-12:
        raise ValueError("diagnostic time step must be positive and no larger than the 240 Hz baseline")
    tail_count = int(round(.5/dt))
    start_depth = args.depth_m if args.approach_from_depth_m is None else args.approach_from_depth_m
    if not 0 < start_depth <= args.depth_m:
        raise ValueError("approach must begin shallower than the final depth")
    axial_speed = .001
    steps = int(np.ceil(((args.depth_m-start_depth)/axial_speed+2.)/dt))
    k = recipe["mapped_circumferential_stiffness_n_m"]/n
    SimulationManager.set_physics_sim_device(args.device)
    carb.settings.get_settings().set_bool(SETTING_DISABLE_CONTACT_PROCESSING, False)
    world = World(stage_units_in_meters=1., physics_dt=dt, rendering_dt=1/60,
                  backend="numpy", device=args.device, sim_params={"use_gpu_pipeline": args.device != "cpu"})
    stage = omni.usd.get_context().get_stage()
    physics = UsdPhysics.Scene.Get(stage, world.get_physics_context().prim_path)
    physics.CreateGravityMagnitudeAttr(0.)
    settings = PhysxSchema.PhysxSceneAPI.Apply(physics.GetPrim())
    settings.CreateSolverTypeAttr(args.solver)
    settings.CreateMinVelocityIterationCountAttr(args.velocity_iterations)
    settings.CreateMaxVelocityIterationCountAttr(args.velocity_iterations)
    if args.external_forces_every_iteration is not None:
        settings.CreateEnableExternalForcesEveryIterationAttr(bool(args.external_forces_every_iteration))
    if args.contact_last is not None:
        settings.CreateSolveArticulationContactLastAttr(bool(args.contact_last))
    if args.gpu_partitions is not None:
        settings.CreateGpuMaxNumPartitionsAttr(args.gpu_partitions)

    material = UsdShade.Material.Define(stage, "/World/DiagnosticFrictionlessMaterial")
    mat = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    mat.CreateStaticFrictionAttr(0.); mat.CreateDynamicFrictionAttr(0.)
    mat.CreateRestitutionAttr(0.)

    def mesh(path, vertices, faces, sdf):
        shape = UsdGeom.Mesh.Define(stage, path)
        local_scale = args.mesh_local_scale if sdf else 1.
        shape.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(vertices*local_scale, np.float32)))
        if local_scale != 1.:
            UsdGeom.Xformable(shape.GetPrim()).AddScaleOp().Set(Gf.Vec3f(1/local_scale))
        shape.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, np.int32)))
        shape.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(faces, np.int32).ravel()))
        shape.CreateSubdivisionSchemeAttr("none")
        prim = shape.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        approximation = {"sdf": "sdf", "convex_decomposition": "convexDecomposition", "peak_sphere": "none",
                         "prepared_convex": "convexHull", "convex_hull": "convexHull"}[args.collider_mode] if sdf else "none"
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(approximation)
        collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        collision.CreateRestOffsetAttr(0.); collision.CreateContactOffsetAttr(.00005)
        PhysxSchema.PhysxTriangleMeshCollisionAPI.Apply(prim).CreateWeldToleranceAttr(0.)
        if sdf and args.collider_mode == "sdf":
            api = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim)
            api.CreateSdfResolutionAttr(args.sdf_resolution)
            api.CreateSdfSubgridResolutionAttr(6)
            api.CreateSdfNarrowBandThicknessAttr(.002)
        elif sdf and args.collider_mode == "convex_decomposition":
            api = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
            api.CreateErrorPercentageAttr(.1)
            api.CreateHullVertexLimitAttr(args.hull_vertex_limit)
            api.CreateMaxConvexHullsAttr(32)
            api.CreateMinThicknessAttr(.000001*local_scale)
            api.CreateShrinkWrapAttr(True)
            api.CreateVoxelResolutionAttr(500000)
        elif sdf:
            api = PhysxSchema.PhysxConvexHullCollisionAPI.Apply(prim)
            api.CreateHullVertexLimitAttr(args.hull_vertex_limit)
            api.CreateMinThicknessAttr(.000001*local_scale)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, UsdShade.Tokens.strongerThanDescendants, "physics")
        return prim

    def rigid_mass(prim, mass, center, inertia):
        UsdPhysics.RigidBodyAPI.Apply(prim)
        properties = UsdPhysics.MassAPI.Apply(prim)
        values, axes = np.linalg.eigh(np.asarray(inertia))
        if values.min() <= 0:
            raise ValueError("nonpositive element inertia")
        if np.linalg.det(axes) < 0: axes[:, 0] *= -1
        quat = Rotation.from_matrix(axes).as_quat()
        properties.CreateMassAttr(float(mass))
        properties.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, center)))
        properties.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, values)))
        properties.CreatePrincipalAxesAttr(Gf.Quatf(float(quat[3]), Gf.Vec3f(*map(float, quat[:3]))))
        api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        api.CreateLinearDampingAttr(0.); api.CreateAngularDampingAttr(0.)
        api.CreateSleepThresholdAttr(0.)
        if args.cfm_scale is not None:
            api.CreateCfmScaleAttr(args.cfm_scale)
        api.CreateSolverPositionIterationCountAttr(args.position_iterations)
        api.CreateSolverVelocityIterationCountAttr(args.velocity_iterations)
        PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)

    socket = trimesh.load(repo / "artifacts/kcg_connector/vision/sam6d_receptacle_current_camera_run21_v1/D38999_20FJ35SN_VISUAL.obj", force="mesh", process=False)
    mesh("/World/Socket", socket.vertices*.001, socket.faces, False)
    frame = UsdGeom.Xform.Define(stage, "/World/SectorCoupon")
    frame.AddTranslateOp().Set(Gf.Vec3d(0., 0., -start_depth))
    frame.AddOrientOp().Set(Gf.Quatf(0., Gf.Vec3f(0., 1., 0.)))
    if args.joint_model == "regular_d6":
        # These numerical elements are pieces of one deforming band. Suppress
        # internal element contacts just as in the articulation counterpart.
        group = UsdPhysics.CollisionGroup.Define(stage, "/World/BandInternalCollisionGroup")
        Usd.CollectionAPI.Apply(group.GetPrim(), "colliders").CreateIncludesRel().SetTargets([frame.GetPath()])
        group.CreateFilteredGroupsRel().SetTargets([group.GetPath()])
    core_path = "/World/SectorCoupon/Core"
    core = UsdGeom.Xform.Define(stage, core_path).GetPrim()
    rigid_mass(core, geometry["rigid_core_mass_kg"], geometry["rigid_core_center_of_mass_m"],
               geometry["rigid_core_inertia_about_com_kg_m2"])
    clamp = UsdPhysics.FixedJoint.Define(stage, "/World/SectorCoupon/ExplicitLaboratoryClamp")
    clamp_body = core_path
    if args.approach_from_depth_m is not None:
        clamp_body = "/World/SectorCoupon/ExplicitLaboratoryAnchor"
        anchor = UsdGeom.Xform.Define(stage, clamp_body).GetPrim()
        rigid_mass(anchor, 1., [0., 0., 0.], np.eye(3)*.001)
        axial = UsdPhysics.PrismaticJoint.Define(stage, "/World/SectorCoupon/ExplicitLaboratoryAxialDrive")
        axial.CreateBody0Rel().SetTargets([clamp_body]); axial.CreateBody1Rel().SetTargets([core_path])
        axial.CreateAxisAttr("Z")
        axial.CreateLowerLimitAttr(0.); axial.CreateUpperLimitAttr(args.depth_m-start_depth)
        if not args.omit_lab_velocity_constraint:
            PhysxSchema.PhysxJointAPI.Apply(axial.GetPrim()).CreateMaxJointVelocityAttr(axial_speed)
        drive = UsdPhysics.DriveAPI.Apply(axial.GetPrim(), "linear")
        drive.CreateTypeAttr("force"); drive.CreateStiffnessAttr(100000.)
        drive.CreateDampingAttr(100.); drive.CreateMaxForceAttr(20.)
        drive.CreateTargetPositionAttr(0. if args.ramped_approach else args.depth_m-start_depth)
        drive.CreateTargetVelocityAttr(0.)
    clamp.CreateBody1Rel().SetTargets([clamp_body])
    clamp.CreateLocalPos0Attr(Gf.Vec3f(0., 0., -start_depth))
    clamp.CreateLocalRot0Attr(Gf.Quatf(0., Gf.Vec3f(0., 1., 0.)))
    if args.joint_model == "articulation":
        UsdPhysics.ArticulationRootAPI.Apply(clamp.GetPrim())
        articulation = PhysxSchema.PhysxArticulationAPI.Apply(clamp.GetPrim())
        articulation.CreateSolverPositionIterationCountAttr(args.position_iterations)
        articulation.CreateSolverVelocityIterationCountAttr(args.velocity_iterations)
        articulation.CreateEnabledSelfCollisionsAttr(False)
    paths, directions, dampings, collision_entries, drive_mapping = [], [], [], [], []
    for element in elements:
        path = f"/World/SectorCoupon/Sector{element['index']:03d}"
        prim = UsdGeom.Xform.Define(stage, path).GetPrim()
        frame_offset = (np.asarray(element["center_of_mass_body_m"])
                        if args.center_sector_frames else np.zeros(3))
        if args.center_sector_frames:
            UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(*map(float, frame_offset)))
        if args.collider_mode == "peak_sphere":
            # Analytic contact diagnostic, not a source-envelope replacement.
            from diagnose_grounding_band_radial_quadrature import radial_intersections
            with np.load(element["path"]) as data:
                source_sector = trimesh.Trimesh(data["vertices_m"], data["faces"], process=False)
            peak_z = -.010105022229
            theta = element["angle_center_rad"]
            section = trimesh.intersections.mesh_plane(source_sector, [0., 0., 1.], [0., 0., peak_z+1e-9])
            outer_radius = radial_intersections(section, np.array([theta]), nearest=False)[0]
            outer_radius += args.sphere_outer_offset_m
            radius = .0005
            sphere = UsdGeom.Sphere.Define(stage, path+"/ExplicitAnalyticContactProbe")
            sphere.CreateRadiusAttr(radius)
            sphere_center = np.array([(outer_radius-radius)*np.cos(theta),
                                      (outer_radius-radius)*np.sin(theta), peak_z]) - frame_offset
            sphere.AddTranslateOp().Set(Gf.Vec3d(*map(float, sphere_center)))
            UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
            api = PhysxSchema.PhysxCollisionAPI.Apply(sphere.GetPrim())
            api.CreateContactOffsetAttr(.00005); api.CreateRestOffsetAttr(0.)
            UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(material, UsdShade.Tokens.strongerThanDescendants, "physics")
        elif args.collision_sector_index is None or element["index"] in args.collision_sector_index:
            source_shapes = (element["prepared_convex_collision_slabs"] if args.collider_mode == "prepared_convex"
                             else [{"path": element["path"]}])
            for shape_index, source_shape in enumerate(source_shapes):
                shape_path = path+f"/SourceEnvelope{shape_index:03d}"
                with np.load(source_shape["path"]) as data:
                    mesh(shape_path, data["vertices_m"] - frame_offset, data["faces"], True)
                collision_entries.append((element["index"], shape_path))
        rigid_mass(prim, element["mass_kg"], np.asarray(element["center_of_mass_body_m"]) - frame_offset,
                   element["inertia_about_sector_com_in_body_axes_kg_m2"])
        theta = element["angle_center_rad"]
        quat = Rotation.from_euler("z", theta).as_quat()
        frame_quat = Gf.Quatf(float(quat[3]), Gf.Vec3f(*map(float, quat[:3])))
        joint = (UsdPhysics.PrismaticJoint.Define(stage, path+"RadialSpring") if args.joint_model == "articulation"
                 else UsdPhysics.Joint.Define(stage, path+"RadialSpring"))
        joint.CreateBody0Rel().SetTargets([core_path]); joint.CreateBody1Rel().SetTargets([path])
        if args.joint_model == "articulation":
            joint.CreateAxisAttr("X")
        center = Gf.Vec3f(*map(float, element["center_of_mass_body_m"]))
        joint.CreateLocalPos0Attr(center)
        joint.CreateLocalPos1Attr(Gf.Vec3f(*map(float, np.asarray(center) - frame_offset)))
        joint.CreateLocalRot0Attr(frame_quat); joint.CreateLocalRot1Attr(frame_quat)
        if args.joint_model == "articulation":
            joint.CreateLowerLimitAttr(-travel); joint.CreateUpperLimitAttr(0.)
        else:
            for axis in ("transX", "transY", "transZ", "rotX", "rotY", "rotZ"):
                limit = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), axis)
                limit.CreateLowAttr(-travel if axis == "transX" else 1.)
                limit.CreateHighAttr(0. if axis == "transX" else -1.)
        damping = 2*np.sqrt(k*element["mass_kg"])
        native_stiffness, native_damping = k, damping
        mapping_denominator = 1.
        if args.compensate_drive_discretization:
            h = dt/args.position_iterations
            mapping_denominator = 1-(h*damping+h*h*k)/element["mass_kg"]
            if mapping_denominator <= 0:
                raise ValueError("inverse implicit coefficient requires a smaller effective solver step")
            native_stiffness, native_damping = k/mapping_denominator, damping/mapping_denominator
        drive_mapping.append({"sector_index": element["index"], "physical_target_k_n_m": k,
                              "physical_target_d_ns_m": float(damping),
                              "native_drive_k_n_m": float(native_stiffness),
                              "native_drive_d_ns_m": float(native_damping),
                              "inverse_coefficient_denominator": float(mapping_denominator)})
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear" if args.joint_model == "articulation" else "transX")
        drive.CreateTypeAttr("force"); drive.CreateStiffnessAttr(float(native_stiffness))
        drive.CreateDampingAttr(float(native_damping)); drive.CreateTargetPositionAttr(0.)
        drive.CreateTargetVelocityAttr(0.)
        drive.CreateMaxForceAttr(float(k*travel+damping*.003))
        paths.append(path); directions.append([-np.cos(theta), np.sin(theta), 0.]); dampings.append(float(damping))
    # Save the actual native cooked convexes before motion, so geometric
    # approximation can be evaluated independently of the force result.
    cooked_records = []
    if args.collider_mode in ("convex_decomposition", "prepared_convex", "convex_hull"):
        cooking = get_physx_cooking_interface()
        stage_id = UsdUtils.StageCache.Get().Insert(stage).ToLongInt()
        cooked_dir = args.output / "cooked_sectors"
        cooked_dir.mkdir()
        for index, (sector_index, path) in enumerate(collision_entries):
            received = {}
            def on_result(result, convexes):
                received["status"] = str(result)
                received["hulls"] = []
                for hull_index, convex in enumerate(convexes):
                    vertices = np.asarray([list(p) for p in convex.vertices], dtype=float)
                    hull = ConvexHull(vertices)
                    destination = cooked_dir / f"sector_{index:03d}_hull_{hull_index:02d}.npz"
                    frame_offset = (np.asarray(next(e for e in elements if e["index"] == index)["center_of_mass_body_m"])
                                    if args.center_sector_frames else np.zeros(3))
                    np.savez_compressed(destination, vertices_body_m=vertices/args.mesh_local_scale + frame_offset,
                                        faces=hull.simplices)
                    received["hulls"].append({"path": str(destination.resolve()),
                                              "vertex_count": len(vertices), "face_count": len(hull.simplices)})
            cooking.request_convex_collision_representation(
                stage_id, PhysicsSchemaTools.sdfPathToInt(stage.GetPrimAtPath(path).GetPath()),
                False, on_result)
            if not received.get("hulls"):
                raise RuntimeError(f"native convex cooking unavailable: {index}, {received}")
            cooked_records.append({"sector_index": sector_index, "source_collider_path": path, **received})
        (args.output / "cooked_collision_manifest.json").write_text(json.dumps(cooked_records, indent=2)+"\n")
    core_view = SingleRigidPrim(core_path, name="clamped_core", reset_xform_properties=False)
    contacts = TensorRigidPrim(paths, resolve_paths=False,
                              contact_filter_paths=["/World/Socket"], max_contact_count=16384)
    event_contacts = []
    def on_contact_report(headers, data):
        for header in headers:
            actors = [str(PhysicsSchemaTools.intToSdfPath(value)) for value in (header.actor0, header.actor1)]
            if "/World/Socket" not in actors:
                continue
            sector_side = next((i for i, actor in enumerate(actors) if actor in paths), None)
            if sector_side is None:
                continue
            sign = 1. if sector_side == 0 else -1.
            for j in range(int(header.contact_data_offset), int(header.contact_data_offset+header.num_contact_data)):
                event_contacts.append({"sector_actor": actors[sector_side], "actors": actors,
                                       "position_world_m": list(data[j].position),
                                       "normal_world": list(data[j].normal),
                                       "impulse_on_sector_world_n_s": (sign*np.asarray(data[j].impulse)).tolist(),
                                       "separation_m": float(data[j].separation)})
    contact_subscription = get_physx_simulation_interface().subscribe_contact_report_events(on_contact_report)
    stage.GetRootLayer().Export(str(args.output / "coupon_before_reset.usda"))
    world.reset()
    core_view.initialize()
    # Initialize handles after the ordinary reset; do not register another
    # scene object whose post-reset hook could reset joint states or targets.
    tree_view = None
    if args.joint_model == "articulation":
        tree_view = SingleArticulation(
            "/World/SectorCoupon", name="radial_spring_tree", reset_xform_properties=False)
        tree_view.initialize()
    if not contacts.is_physics_tensor_entity_valid():
        raise RuntimeError("radial element tensor view is unavailable")
    native_names = list(tree_view.dof_names) if tree_view else []
    axial_index = native_names.index("ExplicitLaboratoryAxialDrive") if args.ramped_approach else None
    native_body_paths = list(contacts._physics_rigid_body_view.prim_paths)
    identity = {
        "requested_rigid_sensor_paths": paths,
        "native_rigid_view_paths": native_body_paths,
        "native_path_order_matches_request": native_body_paths == paths,
        "native_dof_names": native_names,
        "native_dof_count": len(native_names),
        "articulation_reader_initialized_after_reset_without_post_reset": True,
        "native_self_collisions_enabled": int(tree_view.get_enabled_self_collisions()) if tree_view else None,
        "native_solver_position_iterations": int(tree_view.get_solver_position_iteration_count()) if tree_view else None,
        "native_solver_velocity_iterations": int(tree_view.get_solver_velocity_iteration_count()) if tree_view else None,
    }
    def native_array(value):
        return value.cpu().numpy() if hasattr(value, "cpu") else np.asarray(value)
    if tree_view:
        native_k, native_d = tree_view._articulation_view.get_gains()
        identity["native_joint_stiffnesses"] = native_array(native_k).tolist()
        identity["native_joint_dampings"] = native_array(native_d).tolist()
        identity["native_joint_max_efforts"] = native_array(tree_view._articulation_view.get_max_efforts()).tolist()
        initial_actions = tree_view._articulation_view.get_applied_actions()
        identity["native_initial_joint_position_targets"] = native_array(initial_actions.joint_positions).tolist()
        identity["native_initial_joint_velocity_targets"] = native_array(initial_actions.joint_velocities).tolist()
    (args.output / "native_identity_after_reset.json").write_text(json.dumps(identity, indent=2)+"\n")
    direction = np.asarray(directions)
    rows, q_rows, native_q_rows, native_v_rows, position_rows, quaternion_rows = [], [], [], [], [], []
    origin_projection_rows, core_position_rows, core_orientation_rows = [], [], []
    local_anchors = np.asarray([e["center_of_mass_body_m"] for e in elements])
    leaf_anchors = np.zeros_like(local_anchors) if args.center_sector_frames else local_anchors
    local_axes = np.asarray([[np.cos(e["angle_center_rad"]), np.sin(e["angle_center_rad"]), 0.] for e in elements])
    native_effort_rows, native_joint_force_rows = [], []
    event_contact_rows = []
    contact_partners = {}
    for step in range(steps):
        event_contacts.clear()
        if args.ramped_approach:
            commanded_position = min((step+1)*dt*axial_speed, args.depth_m-start_depth)
            commanded_velocity = axial_speed if commanded_position < args.depth_m-start_depth else 0.
            tree_view.apply_action(ArticulationAction(
                joint_positions=np.array([commanded_position]),
                joint_velocities=np.array([commanded_velocity]), joint_indices=[axial_index]))
        world.step(render=False)
        event_force = np.zeros(3)
        event_radial, event_nonzero = 0., 0
        for contact in event_contacts:
            force = np.asarray(contact["impulse_on_sector_world_n_s"])/dt
            point = np.asarray(contact["position_world_m"])
            event_force += force
            event_radial -= float(np.dot(force[:2], point[:2]/max(np.linalg.norm(point[:2]), 1e-12)))
            event_nonzero += int(np.linalg.norm(force) > 1e-8)
        event_contact_rows.append([event_radial, *event_force, event_nonzero, len(event_contacts),
                                   min((c["separation_m"] for c in event_contacts), default=0.)])
        positions, quaternions = contacts.get_world_poses()
        core_position, core_orientation = core_view.get_world_pose()
        core_position = core_position.cpu().numpy() if hasattr(core_position, "cpu") else np.asarray(core_position)
        core_orientation = native_array(core_orientation)
        core_rotation = Rotation.from_quat(core_orientation[[1,2,3,0]]).as_matrix()
        leaf_rotations = Rotation.from_quat(quaternions.numpy()[:, [1,2,3,0]]).as_matrix()
        anchor_delta = (positions.numpy()+np.einsum("nij,nj->ni", leaf_rotations, leaf_anchors)
                        -core_position-local_anchors@core_rotation.T)
        q = np.sum(anchor_delta*(local_axes@core_rotation.T), axis=1)
        origin_delta = positions.numpy()-core_position
        if args.center_sector_frames:
            origin_delta = origin_delta - local_anchors @ core_rotation.T
        origin_projection_rows.append(np.sum(origin_delta*direction, axis=1))
        core_position_rows.append(core_position.copy()); core_orientation_rows.append(core_orientation.copy())
        impulse, point, normal, separation, counts, starts, other_ids = contacts.get_raw_contact_data(dt=1.)
        impulse, point, normal = impulse.numpy().ravel(), point.numpy(), normal.numpy()
        all_forces, total_radial, active = np.zeros(3), 0., 0
        for count, start in zip(counts.numpy().ravel(), starts.numpy().ravel()):
            start, count = int(start), int(count)
            sl = slice(start, start+count)
            force = impulse[sl, None]*normal[sl]/dt
            radial = point[sl, :2]/np.maximum(np.linalg.norm(point[sl, :2], axis=1)[:, None], 1e-12)
            total_radial -= float(np.sum(force[:, :2]*radial))
            active += int(np.count_nonzero(np.linalg.norm(force, axis=1) > 1e-8))
            all_forces += force.sum(axis=0)
        rows.append([total_radial, *all_forces, active, float(counts.numpy().sum()),
                     float(np.linalg.norm(core_position-np.array([0., 0., -args.depth_m]))),
                     float(-core_position[2])])
        q_rows.append(q)
        if tree_view:
            native_q_rows.append(native_array(tree_view.get_joint_positions()))
            native_v_rows.append(native_array(tree_view.get_joint_velocities()))
            native_effort_rows.append(native_array(tree_view.get_measured_joint_efforts()))
            native_joint_force_rows.append(native_array(tree_view.get_measured_joint_forces()))
        position_rows.append(positions.numpy().copy()); quaternion_rows.append(quaternions.numpy().copy())
        if step == steps-1:
            (args.output / "final_contact_events.json").write_text(json.dumps(event_contacts, indent=2)+"\n")
            np.savez_compressed(args.output / "final_raw_contacts.npz", impulses_n_s=impulse,
                                points_world_m=point, normals_world=normal, separations_m=separation.numpy(),
                                counts=counts.numpy(), starts=starts.numpy(), other_actor_ids=other_ids.numpy())
            for count, start in zip(counts.numpy().ravel(), starts.numpy().ravel()):
                start, count = int(start), int(count)
                ids = np.ascontiguousarray(other_ids.numpy().ravel()[start:start+count], dtype=np.uint64)
                partners = contacts.get_actor_paths_from_ids(wp.array(ids, dtype=wp.uint64, device="cpu"))
                for j, partner in enumerate(partners):
                    record = contact_partners.setdefault(str(partner), {"records": 0, "nonzero_impulse_records": 0, "sum_absolute_impulse_n_s": 0.})
                    magnitude = abs(float(impulse[start+j]))
                    record["records"] += 1
                    record["nonzero_impulse_records"] += int(magnitude > 1e-10)
                    record["sum_absolute_impulse_n_s"] += magnitude
    values, deflections = np.asarray(rows), np.asarray(q_rows)
    result = {"scope": "EXPLICITLY_CLAMPED_PASSIVE_RADIAL_ELEMENT_COUPON_NOT_ASSEMBLY",
              "sector_frame_origins": "SECTOR_COM" if args.center_sector_frames else "BODY_ORIGIN",
              "equivalent_world_geometry_mass_and_joint_anchors": True,
              "geometry_manifest": str(args.geometry.resolve()), "source_assets_modified": False,
              "robot_or_nut_present": False, "rigid_core_collision_present": False,
              "fixed_core_role": "DECLARED_LABORATORY_FIXTURE_ONLY",
              "axial_fixture": "FIXED" if args.approach_from_depth_m is None else "EXPLICIT_BOUNDED_PRISMATIC_LABORATORY_DRIVE",
              "approach_start_depth_m": start_depth,
              "axial_drive_if_used": {"max_joint_velocity_constraint_m_s": None if args.omit_lab_velocity_constraint else axial_speed,
                                      "commanded_path_velocity_m_s": axial_speed, "stiffness_n_m": 100000.,
                                      "damping_ns_m": 100., "maximum_force_n": 20.,
                                      "target_set_only_before_start": not args.ramped_approach,
                                      "ramped_target_depends_only_on_time": args.ramped_approach},
              "post_start_object_pose_writes": False,
              "post_start_passive_radial_joint_target_writes": False,
              "post_start_laboratory_axial_drive_target_writes": args.ramped_approach,
              "numerical_element_count": len(elements), "original_full_ring_element_count": n,
              "radial_joint_representation": args.joint_model,
              "radial_displacement_measurement": "RELATIVE_JOINT_ANCHORS_PROJECTED_ON_CURRENT_CORE_JOINT_AXIS",
              "isolated_sector_index": args.sector_index, "physical_TE_finger_count_claimed": False,
              "only_enabled_collision_sector_index": args.collision_sector_index,
              "whole_body_rest_mass_properties_preserved_by_allocation": args.sector_index is None,
              "body_depth_m": args.depth_m, "physics_dt_s": dt, "solver": f"{args.solver}{args.position_iterations}_{args.velocity_iterations}",
              "requested_physics_device": args.device,
              "explicit_constraint_force_mixing_scale": args.cfm_scale,
              "gpu_max_num_partitions": settings.GetGpuMaxNumPartitionsAttr().Get(),
              "inverse_implicit_coefficient_diagnostic": args.compensate_drive_discretization,
              "drive_parameter_mapping": drive_mapping,
              "external_forces_every_iteration": settings.GetEnableExternalForcesEveryIterationAttr().Get(),
              "solve_articulation_contact_last": settings.GetSolveArticulationContactLastAttr().Get(),
              "radial_stiffness_per_element_n_m": k,
              "damping_rule": "Critical damping of each allocated element mass; numerical modeling choice",
              "damping_per_element_ns_m": dampings,
              "maximum_inward_joint_travel_m": travel,
              "drive_force_cap_rule": "K*travel + D*0.003 m/s; finite passive numerical law",
              "element_self_collisions": "DISABLED_WITHIN_THE_EQUIVALENT_DEFORMING_BODY",
              "collider_representation": args.collider_mode,
              "sdf_resolution_if_used": args.sdf_resolution if args.collider_mode == "sdf" else None,
              "source_collision_envelope_preserved": args.collider_mode in ("sdf", "prepared_convex"),
              "hull_vertex_limit_if_used": args.hull_vertex_limit,
              "peak_sphere_role": "ANALYTIC_CONTACT_DIAGNOSTIC_ONLY" if args.collider_mode == "peak_sphere" else None,
              "peak_sphere_outer_radius_offset_m": args.sphere_outer_offset_m,
              "mesh_local_coordinate_scale_with_inverse_transform": args.mesh_local_scale,
              "cooked_convex_manifest": "cooked_collision_manifest.json" if cooked_records else None,
              "diagnostic_friction": 0., "gravity": 0.,
              "columns": ["inward_radial_contact_force_sum_n", "world_fx_n", "world_fy_n", "world_fz_n",
                          "positive_force_record_count", "all_native_contact_record_count", "core_position_error_from_final_target_m",
                          "actual_body_depth_m"],
              "last_half_second_sample_count": tail_count,
              "last_half_second_mean": values[-tail_count:].mean(0).tolist(),
              "last_half_second_minimum": values[-tail_count:].min(0).tolist(),
              "last_half_second_maximum": values[-tail_count:].max(0).tolist(),
              "last_half_second_deflection_min_max_m": [float(deflections[-tail_count:].min()), float(deflections[-tail_count:].max())],
              "last_half_second_elastic_force_sum_mean_n": float((-k*deflections[-tail_count:]).sum(axis=1).mean()),
              "manufacturer_stiffness_or_damping_identified": False}
    result.update(native_identity=identity, final_contact_partners=contact_partners,
                  native_joint_position_range_m=[float(np.min(native_q_rows)), float(np.max(native_q_rows))] if native_q_rows else None)
    result["contact_event_columns"] = ["inward_radial_force_sum_n", "world_fx_n", "world_fy_n", "world_fz_n",
                                       "nonzero_impulse_contact_count", "all_contact_count", "minimum_separation_m"]
    result["last_half_second_contact_event_mean"] = np.asarray(event_contact_rows)[-tail_count:].mean(0).tolist()
    result["raw_contact_tensor_has_records"] = bool(np.any(values[:, 5] > 0))
    np.savez_compressed(args.output / "samples.npz", result_columns=values, radial_joint_displacements_m=deflections,
                        legacy_origin_projection_m=np.asarray(origin_projection_rows),
                        core_positions_world_m=np.asarray(core_position_rows),
                        core_orientations_world_wxyz=np.asarray(core_orientation_rows),
                        native_joint_positions_m=np.asarray(native_q_rows),
                        native_joint_velocities_m_s=np.asarray(native_v_rows),
                        native_projected_joint_efforts_n=np.asarray(native_effort_rows),
                        native_incoming_joint_wrenches=np.asarray(native_joint_force_rows),
                        contact_event_data=np.asarray(event_contact_rows),
                        native_rigid_positions_world_m=np.asarray(position_rows),
                        native_rigid_orientations_world_wxyz=np.asarray(quaternion_rows))
    (args.output / "result.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2), flush=True)
except Exception:
    failed = True
    import traceback
    error = traceback.format_exc()
    (args.output / "error.txt").write_text(error); print(error, flush=True)
finally:
    app.close(exit_code=1 if failed else 0)
