"""Passive CPU contact reproduction from completed-episode poses.

Archived poses define initial diagnostic conditions only. No robot, Nut,
controller, gravity, world attachment or post-start pose command is present.
This is not an assembly episode.
"""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
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
    from isaacsim.core.experimental.prims import RigidPrim
    from isaacsim.core.simulation_manager import SimulationManager
    from isaacsim.core.experimental.utils.backend import use_backend
    from omni.physx.bindings._physx import SETTING_DISABLE_CONTACT_PROCESSING

    repo = Path(__file__).resolve().parents[3]
    selected = {}
    with (args.run/"truth_samples.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["step"] in (41747, 42410):
                selected[row["step"]] = row
    if len(selected) != 2:
        raise ValueError("expected two completed CPU failure endpoints")
    socket_origin = np.asarray(json.loads((args.run/"assembly_scene.json").read_text())["socket_initial_position_world_m"])
    source = np.load(repo/"artifacts/carts_grasp/CARTS_GRASP_V1/object_models/te_j35/plug_body_visual_mesh.npz")
    socket = trimesh.load(repo/"artifacts/kcg_connector/vision/sam6d_receptacle_current_camera_run21_v1/D38999_20FJ35SN_VISUAL.obj", process=False)
    source_stage = Usd.Stage.Open(str(repo/"artifacts/kcg_connector/isaac/te_j35_free_split_tabletop_real_mass_resistance_0p020_v2/TE_J35_FREE_SPLIT_PLUG_V1.usdc"))
    mass_source = UsdPhysics.MassAPI(source_stage.GetPrimAtPath("/TE_J35FreeSplitPlug/Body"))
    SimulationManager.set_physics_sim_device("cpu")
    carb.settings.get_settings().set_bool(SETTING_DISABLE_CONTACT_PROCESSING, False)
    dt = 1/240
    world = World(stage_units_in_meters=1., physics_dt=dt, rendering_dt=1/60,
                  backend="numpy", device="cpu", sim_params={"use_gpu_pipeline": False})
    stage = omni.usd.get_context().get_stage()
    physics = UsdPhysics.Scene.Get(stage, world.get_physics_context().prim_path)
    physics.CreateGravityMagnitudeAttr(0.)
    api = PhysxSchema.PhysxSceneAPI.Apply(physics.GetPrim())
    api.CreateSolverTypeAttr("TGS")
    api.CreateMinPositionIterationCountAttr(128); api.CreateMaxPositionIterationCountAttr(128)
    api.CreateMinVelocityIterationCountAttr(1); api.CreateMaxVelocityIterationCountAttr(1)
    api.CreateEnableExternalForcesEveryIterationAttr(True)
    material = UsdShade.Material.Define(stage, "/World/OriginalBodyFriction")
    mat = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    mat.CreateStaticFrictionAttr(.45); mat.CreateDynamicFrictionAttr(.45); mat.CreateRestitutionAttr(0.)
    PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim()).CreateFrictionCombineModeAttr("max")

    def mesh(path, vertices, faces, position, orientation):
        m = UsdGeom.Mesh.Define(stage, path)
        m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(vertices, np.float32)))
        m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, np.int32)))
        m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(faces, np.int32).ravel()))
        m.CreateSubdivisionSchemeAttr("none")
        m.AddTranslateOp().Set(Gf.Vec3d(*map(float, position)))
        m.AddOrientOp().Set(Gf.Quatf(float(orientation[0]), Gf.Vec3f(*map(float, orientation[1:]))))
        prim = m.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        c = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        c.CreateContactOffsetAttr(.00005); c.CreateRestOffsetAttr(0.)
        PhysxSchema.PhysxTriangleMeshCollisionAPI.Apply(prim).CreateWeldToleranceAttr(0.)
        return prim

    cases, paths, sockets = [], [], []
    for step, row in selected.items():
        for narrow in (.002, .01):
            i = len(cases)
            offset = np.array([.15*i, 0., 0.])
            socket_path, body_path = f"/World/Socket{i}", f"/World/Body{i}"
            mesh(socket_path, socket.vertices*.001, socket.faces, offset, [1.,0.,0.,0.])
            pose = np.asarray(row["object_part_positions_m"][0])-socket_origin+offset
            quat = np.asarray(row["object_part_orientations_wxyz"][0])
            prim = mesh(body_path, source["vertices_m"], source["faces"], pose, quat)
            UsdPhysics.RigidBodyAPI.Apply(prim)
            mass = UsdPhysics.MassAPI.Apply(prim)
            mass.CreateMassAttr(mass_source.GetMassAttr().Get())
            mass.CreateCenterOfMassAttr(mass_source.GetCenterOfMassAttr().Get())
            mass.CreateDiagonalInertiaAttr(mass_source.GetDiagonalInertiaAttr().Get())
            mass.CreatePrincipalAxesAttr(mass_source.GetPrincipalAxesAttr().Get())
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, UsdShade.Tokens.strongerThanDescendants, "physics")
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("sdf")
            sdf = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim)
            sdf.CreateSdfResolutionAttr(1024); sdf.CreateSdfSubgridResolutionAttr(6)
            sdf.CreateSdfNarrowBandThicknessAttr(narrow)
            rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            rb.CreateSolverPositionIterationCountAttr(128); rb.CreateSolverVelocityIterationCountAttr(1)
            rb.CreateSleepThresholdAttr(0.)
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
            paths.append(body_path); sockets.append(socket_path)
            cases.append({"source_step": step, "sdf_narrow_band_ratio": narrow,
                          "initial_position_world_m": pose.tolist(), "initial_quaternion_wxyz": quat.tolist()})
    contacts = RigidPrim(paths, resolve_paths=False, contact_filter_paths=sockets, max_contact_count=4096)
    stage.GetRootLayer().Export(str(args.output/"initial_scene.usda"))
    world.reset()
    if not contacts.is_physics_tensor_entity_valid():
        raise RuntimeError("native body contact view was not initialized")
    poses, quaternions, forces, penetrations = [], [], [], []
    for _ in range(120):
        world.step(render=False)
        with use_backend("tensor", raise_on_fallback=True, raise_on_unsupported=True):
            p, q = contacts.get_world_poses()
        impulse, point, normal, separation, counts, starts, _ = contacts.get_raw_contact_data(dt=1.)
        impulse, normal, separation = impulse.numpy().ravel(), normal.numpy(), separation.numpy().ravel()
        fs, ps = [], []
        for start, count in zip(starts.numpy(), counts.numpy()):
            sl = slice(int(start), int(start+count))
            fs.append((impulse[sl,None]*normal[sl]/dt).sum(0))
            ps.append(float(max(0., -separation[sl].min())) if count else 0.)
        poses.append(p.numpy()); quaternions.append(q.numpy()); forces.append(fs); penetrations.append(ps)
    poses, quaternions, forces, penetrations = map(np.asarray, (poses, quaternions, forces, penetrations))
    for i, case in enumerate(cases):
        initial = np.asarray(case["initial_position_world_m"])
        Ri = Rotation.from_quat(np.asarray(case["initial_quaternion_wxyz"])[[1,2,3,0]])
        Rf = Rotation.from_quat(quaternions[-1,i,[1,2,3,0]])
        case.update(maximum_translation_m=float(np.linalg.norm(poses[:,i]-initial, axis=1).max()),
                    final_rotation_change_deg=float(np.degrees((Rf*Ri.inv()).magnitude())),
                    maximum_contact_force_n=float(np.linalg.norm(forces[:,i],axis=1).max()),
                    maximum_reported_penetration_m=float(penetrations[:,i].max()),
                    nonzero_force_samples=int(np.sum(np.linalg.norm(forces[:,i],axis=1)>1e-8)))
    np.savez_compressed(args.output/"samples.npz", positions=poses, orientations_wxyz=quaternions,
                        forces_world_n=forces, maximum_penetration_m=penetrations)
    (args.output/"result.json").write_text(json.dumps({
        "scope":"PASSIVE_CPU_SOURCE_CONTACT_REPRODUCTION_NOT_ASSEMBLY",
        "archived_pose_used_only_for_initial_diagnostic_conditions":True,
        "post_start_pose_writes":False,"robot_or_nut_or_controller_present":False,
        "gravity":0.,"source_geometry_and_mass_unchanged":True,"cases":cases},indent=2)+"\n")
    print(json.dumps(cases,indent=2))
except Exception:
    failed = True
    import traceback
    (args.output/"error.txt").write_text(traceback.format_exc())
    print(traceback.format_exc(), flush=True)
finally:
    app.close(exit_code=1 if failed else 0)
