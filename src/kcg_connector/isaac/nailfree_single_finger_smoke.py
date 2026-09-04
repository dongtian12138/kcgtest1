#!/usr/bin/env python3

"""Directly test the user-provided nail-free STL on the real first finger."""

from __future__ import annotations

import argparse
import json
import math
import traceback
from pathlib import Path


ROBOT_ROOT = "/World/HandArm"
ARTICULATION_PATH = ROBOT_ROOT + "/Geometry/world"
HAND_BASE_PATH = ARTICULATION_PATH + (
    "/iiwa_link_0/iiwa_link_1/iiwa_link_2/iiwa_link_3/iiwa_link_4/"
    "iiwa_link_5/iiwa_link_6/iiwa_link_7/iiwa_link_ee/handbase_link"
)
F1_LINK_PATH = HAND_BASE_PATH + "/f1Link1/f1Link2/f1Link3"
OLD_F1_VISUAL_PATH = F1_LINK_PATH + "/f1Link3"
OLD_F1_COLLISION_PATH = F1_LINK_PATH + "/f1Link3_convex"
NEW_F1_VISUAL_PATH = F1_LINK_PATH + "/nailfree_visual"
NEW_F1_COLLISION_PATH = F1_LINK_PATH + "/nailfree_collision"
PROBE_PATH = "/World/NailfreeF1Probe"

ARM_JOINTS = tuple(f"iiwa_joint_{index}" for index in range(1, 8))
ACTIVE_HAND_JOINTS = ("f1j1", "f1j2", "f2j1", "f3j2")
EXPECTED_DOF_NAMES = ARM_JOINTS + (
    "f1j1", "f1j2", "f1j3", "f2j1", "f2j2", "f3j1", "f3j2", "f3j3",
)

# Robust alignment of the supplied millimetre STL to the f1Link3 frame.
STL_SCALE_M_PER_UNIT = 0.001
STL_TO_F1_YAW_RAD = 2.34686878808
STL_TO_F1_TRANSLATION_M = (0.02049125144, -0.00300734189, 0.0)

# Predeclared before the normal-probe experiment: this large, interior face is
# in the user-confirmed blue pad body, away from the upper bevel contacted in
# run04.  Its surrounding authored pad faces have nearly parallel normals.
TARGET_PAD_FACE_ID = 10154
TARGET_PAD_CENTER_F1_M = (
    -0.03623537776849429,
    -0.05477317655723198,
    0.011083333333333334,
)
TARGET_PAD_NORMAL_F1 = (
    -0.09216002431363503,
    -0.9957442090810823,
    0.0,
)
TOTAL_CONTACT_OFFSET_M = 0.00010
FIRST_CONTACT_RAW_GAP_MIN_M = -0.00025
FIRST_CONTACT_RAW_GAP_MAX_M = 0.00015


def _arguments() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asset",
        default=str(
            repository
            / "artifacts/kcg_connector/isaac/robot/"
            "handarm_keyed_v3_physical_r7/handarm.usda"
        ),
    )
    parser.add_argument(
        "--stl", default="/home/noob/Downloads/指尖无指甲.STL"
    )
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--warmup-steps", type=int, default=60)
    parser.add_argument("--sweep-steps", type=int, default=180)
    parser.add_argument("--baseline-f1j2-rad", type=float, default=0.656261)
    parser.add_argument("--target-f1j2-rad", type=float, default=0.9)
    parser.add_argument("--probe-radius-m", type=float, default=0.001)
    parser.add_argument("--probe-final-overlap-m", type=float, default=0.0002)
    return parser.parse_args()


def _write_result(output: Path, result: dict[str, object]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _load_stl_in_f1(stl_path: Path):
    import numpy as np
    import trimesh

    raw = trimesh.load_mesh(stl_path, force="mesh", process=False)
    if len(raw.faces) == 0 or len(raw.vertices) == 0:
        raise ValueError("supplied STL contains no triangles")
    points = np.asarray(raw.vertices, dtype=np.float64) * STL_SCALE_M_PER_UNIT
    yaw = STL_TO_F1_YAW_RAD
    rotation = np.asarray(
        (
            (math.cos(yaw), -math.sin(yaw), 0.0),
            (math.sin(yaw), math.cos(yaw), 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    points = points @ rotation.T + np.asarray(
        STL_TO_F1_TRANSLATION_M, dtype=np.float64
    )
    faces = np.asarray(raw.faces, dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=points, faces=faces, process=False)

    merged = mesh.copy()
    merged.merge_vertices()
    edges = np.sort(np.asarray(merged.edges, dtype=np.int64), axis=1)
    _, incidence = np.unique(edges, axis=0, return_counts=True)
    topology = {
        "raw_vertex_count": int(len(points)),
        "merged_vertex_count": int(len(merged.vertices)),
        "face_count": int(len(faces)),
        "boundary_edge_count": int(np.count_nonzero(incidence == 1)),
        "nonmanifold_edge_count": int(np.count_nonzero(incidence > 2)),
        "watertight": bool(merged.is_watertight),
        "bounds_f1_m": np.asarray(mesh.bounds, dtype=np.float64).tolist(),
    }
    return points, faces, mesh, topology


def _author_nailfree_f1(stage, points, faces) -> dict[str, object]:
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics

    old = {}
    for path in (OLD_F1_VISUAL_PATH, OLD_F1_COLLISION_PATH):
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"required old fingertip prim is missing: {path}")
        old[path] = {
            "active_before": bool(prim.IsActive()),
            "type": prim.GetTypeName(),
        }
        prim.SetActive(False)

    point_values = [Gf.Vec3f(*map(float, row)) for row in points]
    face_counts = [3] * len(faces)
    face_indices = faces.reshape(-1).tolist()

    visual = UsdGeom.Mesh.Define(stage, NEW_F1_VISUAL_PATH)
    visual.CreatePointsAttr(point_values)
    visual.CreateFaceVertexCountsAttr(face_counts)
    visual.CreateFaceVertexIndicesAttr(face_indices)
    visual.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    visual.CreateDoubleSidedAttr(True)
    visual.CreateDisplayColorAttr([Gf.Vec3f(0.18, 0.62, 0.92)])
    visual.GetPrim().CreateAttribute(
        "kcg:sourceMeshUri", Sdf.ValueTypeNames.String, custom=True
    ).Set("/home/noob/Downloads/指尖无指甲.STL")
    visual.GetPrim().CreateAttribute(
        "kcg:geometryRole", Sdf.ValueTypeNames.String, custom=True
    ).Set("nailfree_fingertip_visual")

    collision = UsdGeom.Mesh.Define(stage, NEW_F1_COLLISION_PATH)
    collision.CreatePointsAttr(point_values)
    collision.CreateFaceVertexCountsAttr(face_counts)
    collision.CreateFaceVertexIndicesAttr(face_indices)
    collision.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    collision.CreateDoubleSidedAttr(True)
    collision.CreatePurposeAttr(UsdGeom.Tokens.guide)
    collision.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    collision_api = UsdPhysics.CollisionAPI.Apply(collision.GetPrim())
    collision_api.CreateCollisionEnabledAttr(True)
    mesh_api = UsdPhysics.MeshCollisionAPI.Apply(collision.GetPrim())
    mesh_api.CreateApproximationAttr().Set("convexDecomposition")
    physx_api = PhysxSchema.PhysxCollisionAPI.Apply(collision.GetPrim())
    physx_api.CreateContactOffsetAttr(0.00005)
    physx_api.CreateRestOffsetAttr(0.0)
    collision.GetPrim().CreateRelationship("material:binding:physics").SetTargets(
        [Sdf.Path(ROBOT_ROOT + "/PhysicsMaterials/fingertip_pad")]
    )
    for name, value in (
        ("kcg:sourceMeshUri", "/home/noob/Downloads/指尖无指甲.STL"),
        ("kcg:materialRole", "fingertip_pad"),
        ("kcg:geometryRole", "nailfree_fingertip_collision"),
    ):
        collision.GetPrim().CreateAttribute(
            name, Sdf.ValueTypeNames.String, custom=True
        ).Set(value)

    f1_prim = stage.GetPrimAtPath(F1_LINK_PATH)
    if not f1_prim.IsValid() or not f1_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError("f1Link3 is not the expected rigid body")
    PhysxSchema.PhysxContactReportAPI.Apply(f1_prim).CreateThresholdAttr().Set(0.0)

    active_colliders = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if (
            (path == F1_LINK_PATH or path.startswith(F1_LINK_PATH + "/"))
            and prim.IsActive()
            and prim.HasAPI(UsdPhysics.CollisionAPI)
        ):
            active_colliders.append(path)
    return {
        "old_prims": old,
        "old_visual_active_after": bool(
            stage.GetPrimAtPath(OLD_F1_VISUAL_PATH).IsActive()
        ),
        "old_collision_active_after": bool(
            stage.GetPrimAtPath(OLD_F1_COLLISION_PATH).IsActive()
        ),
        "new_visual_path": NEW_F1_VISUAL_PATH,
        "new_collision_path": NEW_F1_COLLISION_PATH,
        "new_collision_approximation": mesh_api.GetApproximationAttr().Get(),
        "active_f1_collision_paths": active_colliders,
    }


def _transform_point(stage, prim_path: str, point_local):
    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    value = matrix.Transform(Gf.Vec3d(*map(float, point_local)))
    return [float(value[index]) for index in range(3)]


def _world_to_local(stage, prim_path: str, point_world):
    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim).GetInverse()
    value = matrix.Transform(Gf.Vec3d(*map(float, point_world)))
    return [float(value[index]) for index in range(3)]


def _transform_direction(stage, prim_path: str, direction_local):
    import numpy as np
    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    value = matrix.TransformDir(Gf.Vec3d(*map(float, direction_local)))
    vector = np.asarray([float(value[index]) for index in range(3)])
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise RuntimeError("target pad normal did not transform to a finite direction")
    return (vector / norm).tolist()


def _source_gap_to_probe(stage, mesh, probe_center_world, probe_radius):
    import numpy as np
    import trimesh

    center_f1 = _world_to_local(stage, F1_LINK_PATH, probe_center_world)
    closest, distances, triangle_ids = trimesh.proximity.closest_point_naive(
        mesh, np.asarray(center_f1, dtype=np.float64).reshape(1, 3)
    )
    return {
        "probe_center_f1_m": center_f1,
        "nearest_source_point_f1_m": closest[0].tolist(),
        "nearest_source_triangle_to_probe_center": int(triangle_ids[0]),
        "source_surface_to_probe_center_distance_m": float(distances[0]),
        "raw_mesh_to_probe_surface_gap_m": float(
            distances[0] - float(probe_radius)
        ),
    }


def _author_probe(stage, center, radius: float) -> None:
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    sphere = UsdGeom.Sphere.Define(stage, PROBE_PATH)
    sphere.CreateRadiusAttr(float(radius))
    sphere.AddTranslateOp().Set(Gf.Vec3d(*map(float, center)))
    sphere.CreateDisplayColorAttr([Gf.Vec3f(0.92, 0.18, 0.12)])
    UsdPhysics.CollisionAPI.Apply(sphere.GetPrim()).CreateCollisionEnabledAttr(True)
    rigid_api = UsdPhysics.RigidBodyAPI.Apply(sphere.GetPrim())
    rigid_api.CreateRigidBodyEnabledAttr(True)
    rigid_api.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(sphere.GetPrim()).CreateMassAttr(0.01)
    physx_api = PhysxSchema.PhysxCollisionAPI.Apply(sphere.GetPrim())
    physx_api.CreateContactOffsetAttr(0.00005)
    physx_api.CreateRestOffsetAttr(0.0)
    PhysxSchema.PhysxContactReportAPI.Apply(
        sphere.GetPrim()
    ).CreateThresholdAttr().Set(0.0)


def _set_known_pose(robot, name_to_index, arguments) -> None:
    import numpy as np

    positions = np.zeros(robot.num_dof, dtype=np.float32)
    prescribed = {
        "f1j1": 0.7,
        "f1j2": arguments.baseline_f1j2_rad,
        "f1j3": arguments.baseline_f1j2_rad,
        "f2j1": 0.447734,
        "f2j2": 0.447734,
        "f3j1": 0.7,
        "f3j2": 0.656261,
        "f3j3": 0.656261,
    }
    for name, value in prescribed.items():
        positions[name_to_index[name]] = float(value)
    robot.set_joint_positions(positions)
    robot.set_joint_velocities(np.zeros(robot.num_dof, dtype=np.float32))


def _configure_drives(robot, name_to_index):
    import numpy as np

    controller = robot.get_articulation_controller()
    kps = np.zeros(robot.num_dof, dtype=np.float32)
    kds = np.zeros(robot.num_dof, dtype=np.float32)
    for name in ARM_JOINTS:
        kps[name_to_index[name]] = 400.0
        kds[name_to_index[name]] = 40.0
    for name in ACTIVE_HAND_JOINTS:
        kps[name_to_index[name]] = 12.0
        kds[name_to_index[name]] = 2.0
    controller.set_gains(kps=kps, kds=kds, save_to_usd=False)
    controlled = ARM_JOINTS + ACTIVE_HAND_JOINTS
    indices = [name_to_index[name] for name in controlled]
    maximum_efforts = np.asarray(
        [100.0] * len(ARM_JOINTS) + [1.0] * len(ACTIVE_HAND_JOINTS),
        dtype=np.float32,
    )
    controller.set_max_efforts(maximum_efforts, joint_indices=indices)
    observed = controller.get_max_efforts()
    return controller, indices, {
        name: float(observed[name_to_index[name]]) for name in controlled
    }


def _ramp_f1(
    world, robot, controller, controlled_indices, name_to_index, target_q2,
    steps, *, collect_contacts=False, mesh=None, probe_center_world=None,
    probe_radius=None,
):
    import numpy as np
    from omni.physx import get_physx_simulation_interface
    from pxr import PhysicsSchemaTools

    if collect_contacts and (
        mesh is None or probe_center_world is None or probe_radius is None
    ):
        raise ValueError("contact localization requires mesh and probe geometry")

    baseline = np.asarray(robot.get_joint_positions(), dtype=np.float64)
    targets = baseline[controlled_indices].copy()
    f1_slot = tuple(ARM_JOINTS + ACTIVE_HAND_JOINTS).index("f1j2")
    targets[f1_slot] = float(target_q2)
    start_q2 = float(baseline[name_to_index["f1j2"]])
    records = []
    contact_steps = 0
    first_contact_step = None
    maximum_effort = 0.0
    aborted = False
    for step_index in range(steps):
        blend = float(step_index + 1) / float(steps)
        step_targets = baseline[controlled_indices] + blend * (
            targets - baseline[controlled_indices]
        )
        from isaacsim.core.utils.types import ArticulationAction

        controller.apply_action(
            ArticulationAction(
                joint_positions=step_targets.astype(np.float32),
                joint_indices=np.asarray(controlled_indices, dtype=np.int32),
            )
        )
        world.step(render=False)
        efforts = robot.get_measured_joint_efforts()
        if efforts is not None:
            effort = abs(float(efforts[name_to_index["f1j2"]]))
            maximum_effort = max(maximum_effort, effort)
            if effort > 0.90:
                aborted = True
                break
        if not collect_contacts:
            continue
        headers, contact_data, _ = (
            get_physx_simulation_interface().get_full_contact_report()
        )
        found_this_step = False
        for header in headers:
            paths = tuple(
                str(PhysicsSchemaTools.intToSdfPath(value))
                for value in (
                    header.actor0, header.actor1, header.collider0, header.collider1,
                )
            )
            if PROBE_PATH not in paths or NEW_F1_COLLISION_PATH not in paths:
                continue
            probe_localization = _source_gap_to_probe(
                world.stage, mesh, probe_center_world, probe_radius
            )
            offset = int(header.contact_data_offset)
            count = int(header.num_contact_data)
            for index in range(offset, offset + count):
                row = contact_data[index]
                world_point = [float(row.position[axis]) for axis in range(3)]
                local_point = _world_to_local(stage=world.stage, prim_path=F1_LINK_PATH,
                                              point_world=world_point)
                records.append({
                    "step": step_index,
                    "paths": paths,
                    "position_world_m": world_point,
                    "position_f1_m": local_point,
                    "normal": [float(row.normal[axis]) for axis in range(3)],
                    "impulse_n_s": [float(row.impulse[axis]) for axis in range(3)],
                    "separation_m": float(row.separation),
                    **probe_localization,
                })
            found_this_step = found_this_step or count > 0
        if found_this_step:
            contact_steps += 1
            if first_contact_step is None:
                first_contact_step = step_index
            if contact_steps >= 8:
                break

    final = np.asarray(robot.get_joint_positions(), dtype=np.float64)
    if mesh is not None and records:
        import trimesh

        query = np.asarray([row["position_f1_m"] for row in records], dtype=np.float64)
        closest, distances, triangle_ids = trimesh.proximity.closest_point_naive(
            mesh, query
        )
        target = np.asarray(TARGET_PAD_CENTER_F1_M, dtype=np.float64)
        for row, nearest, distance, triangle_id in zip(
            records, closest, distances, triangle_ids
        ):
            row["nearest_source_surface_f1_m"] = nearest.tolist()
            row["source_surface_distance_m"] = float(distance)
            row["target_pad_center_distance_m"] = float(
                np.linalg.norm(np.asarray(row["position_f1_m"]) - target)
            )
            row["nearest_source_triangle"] = int(triangle_id)
    return {
        "start_f1j2_rad": start_q2,
        "final_f1j2_rad": float(final[name_to_index["f1j2"]]),
        "measured_delta_f1j2_rad": float(
            final[name_to_index["f1j2"]] - start_q2
        ),
        "maximum_abs_measured_f1j2_effort_nm": maximum_effort,
        "effort_abort": aborted,
        "first_contact_step": first_contact_step,
        "contact_steps": contact_steps,
        "contact_record_count": len(records),
        "contact_records": records[:128],
    }


def _capture(world, output: Path, name: str, target, *, collision_debug: bool) -> str:
    import omni.kit.renderer_capture
    from isaacsim.core.utils.viewports import set_camera_view
    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("active viewport is unavailable")
    viewport.resolution = (1600, 900)
    target = [float(value) for value in target]
    eye = [target[0] + 0.14, target[1] + 0.12, target[2] + 0.10]
    set_camera_view(eye=eye, target=target, viewport_api=viewport)

    visualization = None
    physx_ui = None
    hidden_imageables = []
    if collision_debug:
        import omni.kit.app
        from pxr import UsdGeom

        extension_manager = omni.kit.app.get_app().get_extension_manager()
        if not extension_manager.set_extension_enabled_immediate(
            "omni.physx.ui", True
        ):
            raise RuntimeError("failed to enable omni.physx.ui")
        for _ in range(2):
            world.render()
        from omni.physx import get_physx_visualization_interface
        from omni.physxui import get_physxui_interface

        physx_ui = get_physxui_interface()
        visualization = get_physx_visualization_interface()
        for path in (NEW_F1_VISUAL_PATH, PROBE_PATH):
            imageable = UsdGeom.Imageable(world.stage.GetPrimAtPath(path))
            imageable.MakeInvisible()
            hidden_imageables.append(imageable)
        physx_ui.enable_redraw_optimizations(False)
        physx_ui.enable_debug_visualization(True)
        visualization.enable_visualization(True)
        visualization.set_visualization_parameter("CollisionShapes", True)
        world.step(render=True)

    try:
        for _ in range(12):
            world.render()
        image_path = output / f"{name}.png"
        capture_viewport_to_file(viewport, file_path=str(image_path))
        for _ in range(24):
            world.render()
        omni.kit.renderer_capture.acquire_renderer_capture_interface().wait_async_capture()
        if not image_path.is_file() or image_path.stat().st_size <= 0:
            raise RuntimeError(f"viewport capture did not produce {image_path}")
        return str(image_path)
    finally:
        if visualization is not None:
            visualization.set_visualization_parameter("CollisionShapes", False)
            visualization.enable_visualization(False)
        if physx_ui is not None:
            physx_ui.enable_debug_visualization(False)
            physx_ui.enable_redraw_optimizations(True)
        for imageable in hidden_imageables:
            imageable.MakeVisible()


def main() -> None:
    arguments = _arguments()
    output = Path(arguments.output_directory).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output}")
    output.mkdir(parents=True, exist_ok=False)

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {"headless": True, "multi_gpu": False, "active_gpu": 0, "physics_gpu": 0}
    )
    passed = False
    result: dict[str, object] = {
        "schema_version": "nailfree_single_finger_physx_v2",
        "formal_dynamic_pass": False,
        "hardware_authorized": False,
        "simulation_only": True,
        "asset": str(Path(arguments.asset).expanduser().resolve()),
        "source_stl": str(Path(arguments.stl).expanduser().resolve()),
        "output_directory": str(output),
    }
    try:
        import numpy as np
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
        from pxr import Gf, UsdLux

        asset = Path(arguments.asset).expanduser().resolve()
        stl = Path(arguments.stl).expanduser().resolve()
        if not asset.is_file():
            raise FileNotFoundError(asset)
        if not stl.is_file():
            raise FileNotFoundError(stl)
        if arguments.warmup_steps < 1 or arguments.sweep_steps < 60:
            raise ValueError("warmup must be positive and sweep must be at least 60 steps")
        if not 0.0 < arguments.probe_final_overlap_m < arguments.probe_radius_m:
            raise ValueError("probe overlap must be positive and smaller than its radius")

        points, faces, source_mesh, topology = _load_stl_in_f1(stl)
        target_center = np.asarray(TARGET_PAD_CENTER_F1_M, dtype=np.float64)
        target_normal = np.asarray(TARGET_PAD_NORMAL_F1, dtype=np.float64)
        observed_target_center = source_mesh.triangles_center[TARGET_PAD_FACE_ID]
        observed_target_normal = source_mesh.face_normals[TARGET_PAD_FACE_ID]
        if (
            float(np.linalg.norm(observed_target_center - target_center)) > 1.0e-9
            or float(np.dot(observed_target_normal, target_normal)) < 0.999999
        ):
            raise RuntimeError("predeclared interior pad face differs from supplied STL")
        result["stl"] = {
            "scale_m_per_stl_unit": STL_SCALE_M_PER_UNIT,
            "yaw_rad": STL_TO_F1_YAW_RAD,
            "translation_m": list(STL_TO_F1_TRANSLATION_M),
            "transform_formula": "p_f1 = Rz(yaw) * (0.001 * p_stl) + t",
            "topology_observed_without_repair": topology,
        }
        result["target_pad_patch"] = {
            "selection_timing": "PREDECLARED_BEFORE_NORMAL_PROBE_RUN",
            "user_stl_face_id": TARGET_PAD_FACE_ID,
            "center_f1_m": target_center.tolist(),
            "outward_normal_f1": target_normal.tolist(),
            "face_area_m2": float(source_mesh.area_faces[TARGET_PAD_FACE_ID]),
            "identity_basis": (
                "INTERIOR_LARGE_FACE_IN_USER_CONFIRMED_BLUE_PAD_BODY; "
                "NOT_THE_RUN04_UPPER_BEVEL"
            ),
        }

        World.clear_instance()
        world = World(
            stage_units_in_meters=1.0,
            physics_dt=1.0 / 120.0,
            rendering_dt=1.0 / 60.0,
            backend="numpy",
            device="cpu",
        )
        add_reference_to_stage(str(asset), ROBOT_ROOT)
        stage = get_current_stage()
        result["asset_override"] = _author_nailfree_f1(stage, points, faces)

        dome = UsdLux.DomeLight.Define(stage, "/World/NailfreeLighting/Dome")
        dome.CreateIntensityAttr(900.0)
        key = UsdLux.DistantLight.Define(stage, "/World/NailfreeLighting/Key")
        key.CreateIntensityAttr(1800.0)
        key.AddRotateXYZOp().Set(Gf.Vec3f(-35.0, 25.0, 25.0))

        robot = world.scene.add(
            SingleArticulation(
                prim_path=ARTICULATION_PATH, name="nailfree_single_finger_robot"
            )
        )
        world.reset()
        if not robot.handles_initialized:
            raise RuntimeError("articulation handles were not initialized")
        dof_names = tuple(robot.dof_names)
        if set(dof_names) != set(EXPECTED_DOF_NAMES) or len(dof_names) != 15:
            raise RuntimeError(f"unexpected DOF layout: {dof_names}")
        name_to_index = {name: index for index, name in enumerate(dof_names)}
        _set_known_pose(robot, name_to_index, arguments)
        controller, controlled_indices, effort_caps = _configure_drives(
            robot, name_to_index
        )
        result["drive_maximum_effort_nm"] = effort_caps

        hold = np.asarray(robot.get_joint_positions(), dtype=np.float64)[
            controlled_indices
        ]
        from isaacsim.core.utils.types import ArticulationAction

        for _ in range(arguments.warmup_steps):
            controller.apply_action(
                ArticulationAction(
                    joint_positions=hold.astype(np.float32),
                    joint_indices=np.asarray(controlled_indices, dtype=np.int32),
                )
            )
            world.step(render=False)

        target_start = _transform_point(
            stage, F1_LINK_PATH, TARGET_PAD_CENTER_F1_M
        )
        dry_sweep = _ramp_f1(
            world, robot, controller, controlled_indices, name_to_index,
            arguments.target_f1j2_rad, arguments.sweep_steps,
        )
        target_end = _transform_point(
            stage, F1_LINK_PATH, TARGET_PAD_CENTER_F1_M
        )
        target_normal_end = np.asarray(
            _transform_direction(stage, F1_LINK_PATH, TARGET_PAD_NORMAL_F1),
            dtype=np.float64,
        )
        displacement = np.asarray(target_end) - np.asarray(target_start)
        displacement_norm = float(np.linalg.norm(displacement))
        if displacement_norm <= 0.005:
            raise RuntimeError("f1 target pad did not move enough to define closure")
        motion_direction = displacement / displacement_norm
        approach_along_target_normal = float(
            np.dot(displacement, target_normal_end)
        )
        if approach_along_target_normal <= 0.005:
            raise RuntimeError("target pad does not approach along its outward normal")
        probe_center = np.asarray(target_end) + target_normal_end * (
            arguments.probe_radius_m - arguments.probe_final_overlap_m
        )
        result["dry_closure_motion"] = {
            **dry_sweep,
            "target_pad_center_start_world_m": target_start,
            "target_pad_center_end_world_m": target_end,
            "target_pad_normal_end_world": target_normal_end.tolist(),
            "target_pad_displacement_world_m": displacement.tolist(),
            "target_pad_displacement_m": displacement_norm,
            "approach_along_target_normal_m": approach_along_target_normal,
            "measured_closure_direction_world": motion_direction.tolist(),
        }

        world.stop()
        _author_probe(stage, probe_center, arguments.probe_radius_m)
        stage.GetRootLayer().Export(str(output / "runtime_stage.usda"))
        world.reset()
        _set_known_pose(robot, name_to_index, arguments)
        controller, controlled_indices, effort_caps_after_reset = _configure_drives(
            robot, name_to_index
        )
        if effort_caps_after_reset != effort_caps:
            raise RuntimeError("drive effort caps changed across reset")
        hold = np.asarray(robot.get_joint_positions(), dtype=np.float64)[
            controlled_indices
        ]
        for _ in range(arguments.warmup_steps):
            controller.apply_action(
                ArticulationAction(
                    joint_positions=hold.astype(np.float32),
                    joint_indices=np.asarray(controlled_indices, dtype=np.int32),
                )
            )
            world.step(render=False)

        initial_probe_localization = _source_gap_to_probe(
            stage, source_mesh, probe_center, arguments.probe_radius_m
        )
        result["probe"] = {
            "path": PROBE_PATH,
            "center_world_m": probe_center.tolist(),
            "radius_m": float(arguments.probe_radius_m),
            "kinematic_rigid_body": True,
            "placement_formula": (
                "target_center_at_final_pose + outward_raw_normal * "
                "(probe_radius - final_overlap)"
            ),
            "final_target_overlap_m": float(arguments.probe_final_overlap_m),
            "initial_localization": initial_probe_localization,
        }
        before_contact = np.asarray(robot.get_joint_positions(), dtype=np.float64)
        contact_sweep = _ramp_f1(
            world, robot, controller, controlled_indices, name_to_index,
            arguments.target_f1j2_rad, arguments.sweep_steps,
            collect_contacts=True, mesh=source_mesh,
            probe_center_world=probe_center,
            probe_radius=arguments.probe_radius_m,
        )
        after_contact = np.asarray(robot.get_joint_positions(), dtype=np.float64)
        result["contact_sweep"] = contact_sweep
        result["other_active_joint_deltas_rad"] = {
            name: float(after_contact[name_to_index[name]] - before_contact[name_to_index[name]])
            for name in ("f1j1", "f2j1", "f3j2")
        }

        records = contact_sweep["contact_records"]
        maximum_surface_distance = (
            max(float(row["source_surface_distance_m"]) for row in records)
            if records else None
        )
        maximum_target_distance = (
            max(float(row["target_pad_center_distance_m"]) for row in records)
            if records else None
        )
        first_record = records[0] if records else None
        first_raw_gap = (
            float(first_record["raw_mesh_to_probe_surface_gap_m"])
            if first_record else None
        )
        first_probe_triangle = (
            int(first_record["nearest_source_triangle_to_probe_center"])
            if first_record else None
        )
        first_contact_on_target_patch = bool(
            first_record
            and first_probe_triangle == TARGET_PAD_FACE_ID
            and first_raw_gap is not None
            and FIRST_CONTACT_RAW_GAP_MIN_M
            <= first_raw_gap
            <= FIRST_CONTACT_RAW_GAP_MAX_M
        )
        result["contact_localization"] = {
            "maximum_source_surface_distance_m": maximum_surface_distance,
            "maximum_target_pad_center_distance_m": maximum_target_distance,
            "first_contact_raw_mesh_to_probe_surface_gap_m": first_raw_gap,
            "first_contact_nearest_source_triangle_to_probe_center": (
                first_probe_triangle
            ),
            "required_target_face_id": TARGET_PAD_FACE_ID,
            "total_contact_offset_m": TOTAL_CONTACT_OFFSET_M,
            "accepted_first_contact_raw_gap_range_m": [
                FIRST_CONTACT_RAW_GAP_MIN_M,
                FIRST_CONTACT_RAW_GAP_MAX_M,
            ],
            "first_contact_is_on_predeclared_pad_patch": (
                first_contact_on_target_patch
            ),
        }

        result["images"] = {
            "visible_geometry": _capture(
                world, output, "01_nailfree_f1_visible_contact", probe_center,
                collision_debug=False,
            ),
            "collision_debug": _capture(
                world, output, "02_nailfree_f1_collision_debug", probe_center,
                collision_debug=True,
            ),
        }

        asset_override = result["asset_override"]
        other_joint_motion = max(
            abs(float(value)) for value in result["other_active_joint_deltas_rad"].values()
        )
        passed = bool(
            asset_override["old_visual_active_after"] is False
            and asset_override["old_collision_active_after"] is False
            and asset_override["new_collision_approximation"] == "convexDecomposition"
            and asset_override["active_f1_collision_paths"] == [NEW_F1_COLLISION_PATH]
            and dry_sweep["measured_delta_f1j2_rad"] > 0.1
            and dry_sweep["effort_abort"] is False
            and contact_sweep["contact_steps"] >= 3
            and contact_sweep["contact_record_count"] > 0
            and contact_sweep["effort_abort"] is False
            and initial_probe_localization[
                "raw_mesh_to_probe_surface_gap_m"
            ] > TOTAL_CONTACT_OFFSET_M
            and result["contact_localization"][
                "first_contact_is_on_predeclared_pad_patch"
            ] is True
            and other_joint_motion <= 0.02
        )
        result["single_finger_dynamic_pass"] = passed
        result["evidence_limit"] = (
            "Only f1 nail-free visual/collision loading, PhysX contact with a simple "
            "kinematic rigid body, and positive closure motion were tested. No three-"
            "finger grasp, table release, lift, robustness, formal dynamic, or hardware "
            "claim is supported."
        )
        _write_result(output, result)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
        print(
            "ISAAC NAILFREE SINGLE FINGER " + ("PASSED" if passed else "FAILED"),
            flush=True,
        )
    except BaseException as exception:
        result.update(
            {
                "single_finger_dynamic_pass": False,
                "error": f"{type(exception).__name__}: {exception}",
                "traceback": traceback.format_exc(),
            }
        )
        _write_result(output, result)
        traceback.print_exc()
        print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
        print("ISAAC NAILFREE SINGLE FINGER FAILED", flush=True)
    finally:
        simulation_app.close(exit_code=0 if passed else 1)


if __name__ == "__main__":
    main()
