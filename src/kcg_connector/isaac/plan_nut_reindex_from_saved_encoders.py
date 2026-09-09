#!/usr/bin/env python3
"""Offline open-hand reindex candidate from saved encoders and visual geometry.

This does not release a grip, run Isaac, or establish post-release support.
Fresh perception and per-step checks would be required before actual execution.
"""

import argparse
import json
from pathlib import Path
import sys

import fcl
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
import yaml

repository = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repository / "src/kcg_connector"))
from kcg_connector.grasp.carts_v2.models import load_v2_inputs
from kcg_connector.grasp.robust.bounded_hand_base_ik import solve_bounded_hand_base_ik
from te_body_nut_regrasp import body_yaw_swept_bound
from te_foundationpose_handoff_plan import FullRobotCollisionScene
from te_foundationpose_handoff_runtime import _first_discrete_collision, MOVEIT_SOFT_ARM_BOUNDS_RAD, control


def bvh(vertices, faces, rotation=np.eye(3), position=np.zeros(3)):
    model = fcl.BVHModel()
    model.beginModel(len(faces), len(vertices))
    model.addSubModel(np.asarray(vertices), np.asarray(faces, dtype=np.int32))
    model.endModel()
    return fcl.CollisionObject(model, fcl.Transform(rotation, position))


def plan(directory, output):
    if output.exists():
        raise FileExistsError(output)
    launch = json.loads(directory.with_suffix(".launch.json").read_text())
    argv = launch["argv"]
    grasp_config = repository / argv[argv.index("--config")+1]
    assembly_config = repository / argv[argv.index("--body-assembly-collision-config")+1]
    config = yaml.safe_load(assembly_config.read_text())
    inputs = load_v2_inputs(repository, config_path=grasp_config,
                           object_id="te_deutsch_d38999_26fj35pn_step")
    sample = json.loads((directory / "last_rotation_encoder_for_reindex_planning_v1.json").read_text())["sample"]
    q = np.asarray(sample["active_positions_rad"])
    hand = np.asarray(inputs.robot_model.forward_kinematics(tuple(q), enforce_limits=False)["handbase_link"])
    record = json.loads((directory / "socket_transport/nut_rotation/nut_rotation_controller_result.json").read_text())
    observation = record["postgrip_palm_observation"]
    old_hand = np.asarray(observation["world_from_hand_encoder"])
    old_body = np.asarray(observation["world_from_plug_five_dof"])
    body = hand @ np.linalg.inv(old_hand) @ old_body
    geometry = json.loads((repository / config["nut_regrasp"]["geometry_plan"]).read_text())
    open_hand = np.asarray(geometry["open_hand_positions_rad"])
    socket_record = json.loads((directory / "socket_transport/wrist_socket/camera_and_estimate.json").read_text())
    socket = np.asarray(socket_record["measurement"]["world_from_receptacle_row_major"]).reshape(4, 4)
    scene = FullRobotCollisionScene(inputs)
    cooked = json.loads((repository / config["nut_regrasp"]["cooked_hand_geometry"]).read_text())
    meshes = np.load(cooked["mesh_data"])
    for link in ("f1Link3", "f2Link2", "f3Link3"):
        triangles = meshes[link]
        scene.objects[link] = bvh(triangles.reshape(-1, 3), np.arange(triangles.size//3).reshape(-1, 3))
    table = np.asarray(inputs.table_xy_bounds_m)
    table_size = np.r_[table[:, 1]-table[:, 0], 1.]
    table_center = np.r_[table.mean(1), inputs.table_top_z_m-.5]
    authored = json.loads((directory / "socket_transport/tesseract_plans/body_full_pose_to_socket_50mm/request.json").read_text())
    fixture = next(obj for obj in authored["collision_objects"] if obj["id"] == "fixture")
    fixture_pose = np.asarray(fixture["world_from_primitive_row_major"]).reshape(4, 4)
    environment = {
        "table": fcl.CollisionObject(fcl.Box(*table_size), fcl.Transform(table_center)),
        "fixture": fcl.CollisionObject(fcl.Box(*fixture["dimensions_m"]), fcl.Transform(fixture_pose[:3, :3], fixture_pose[:3, 3])),
    }
    socket_mesh = trimesh.load(repository / "artifacts/kcg_connector/vision/sam6d_receptacle_current_camera_run21_v1/D38999_20FJ35SN_VISUAL.obj", process=False, force="mesh")
    environment["socket"] = bvh(socket_mesh.vertices*.001, socket_mesh.faces, socket[:3, :3], socket[:3, 3])
    part_root = repository / "artifacts/carts_grasp/CARTS_GRASP_V1/object_models/te_j35"
    raw = np.load(part_root / "plug_body_visual_mesh.npz")
    envelope, envelope_report = body_yaw_swept_bound(raw["vertices_m"], raw["faces"],
        margin_m=config["nut_regrasp"].get("body_pose_reserve_m", .0001))
    environment["body_all_yaw_bound"] = bvh(envelope.vertices, envelope.faces, body[:3, :3], body[:3, 3])
    nv = np.load(part_root / "coupling_nut_visual_mesh.npz")["vertices_m"]
    low, high = nv[:, 2].min(), nv[:, 2].max()
    center = body[:3, 3]+body[:3, 2]*(low+high)/2
    environment["nut_outer_cylinder_bound"] = fcl.CollisionObject(
        fcl.Cylinder(float(np.linalg.norm(nv[:, :2], axis=1).max()), float(high-low)),
        fcl.Transform(body[:3, :3], center))
    rows, first_stop = [], None
    bounds = np.asarray([MOVEIT_SOFT_ARM_BOUNDS_RAD[name] for name in control.ARM_JOINT_NAMES])
    seed = q[:7].copy()
    for angle in np.arange(0., 120.0001, 1.):
        R = Rotation.from_rotvec(socket[:3, 2]*np.deg2rad(angle)).as_matrix()
        target = hand.copy()
        target[:3, :3] = R @ hand[:3, :3]
        target[:3, 3] = body[:3, 3] + R @ (hand[:3, 3]-body[:3, 3])
        arm, pe, ae, _ = solve_bounded_hand_base_ik(inputs.config.section("ik")["solver"],
            model=inputs.robot_model, hand_positions=open_hand, target_world_from_hand_base=target,
            seed_arm_positions=(seed,), label="OFFLINE_OPEN_HAND_REINDEX")
        arm = np.asarray(arm)
        margin = float(np.min(np.minimum(arm-bounds[:, 0], bounds[:, 1]-arm)))
        collision = _first_discrete_collision(scene, arm, open_hand, environment)
        row = {"angle_deg": float(angle), "arm_rad": arm.tolist(), "position_error_m": pe,
               "orientation_error_rad": ae, "minimum_soft_joint_margin_rad": margin,
               "collision": collision}
        rows.append(row)
        if collision is not None or margin < 0 or pe > .0001 or ae > .001:
            first_stop = row
            break
        seed = arm
    result = {"scope": "OFFLINE_120_DEG_OPEN_HAND_REINDEX_DISCRETE_GEOMETRY_AND_IK_ONLY",
              "source_run": str(directory), "encoder_step": sample["step"],
              "initial_body_origin_from": "SAVED_PALM_POSE_PROPAGATED_THROUGH_EXECUTED_ENCODERS_WHILE_NUT_WAS_HELD",
              "body_yaw_assumed_known": False, "body_all_yaw_envelope": envelope_report,
              "open_hand_positions_rad": open_hand.tolist(), "sample_spacing_deg": 1.,
              "first_stop": first_stop, "all_sampled_states_clear_and_within_bounds": first_stop is None,
              "opening_motion_and_post_release_support_checked": False,
              "between_sample_continuous_collision_checked": False,
              "fresh_post_release_vision_required_before_actual_motion": True,
              "object_or_contact_truth_used": False, "robot_motion_commanded": False,
              "steps": rows}
    output.open("x").write(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"sample_count": len(rows), "first_stop": first_stop,
                      "minimum_soft_margin_rad": min(r["minimum_soft_joint_margin_rad"] for r in rows)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan(args.directory.resolve(), args.output.resolve())
