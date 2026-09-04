#!/usr/bin/env python3
"""Build and collision-check one visual handoff plan without Isaac dynamics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import fcl
import matplotlib.pyplot as plt
import numpy as np
import trimesh
import yaml

sys.path.insert(0, str(Path(__file__).resolve().with_name("carts_v2")))
import controller as control  # type: ignore  # noqa: E402
from finite_cad_search import _fcl_model  # type: ignore  # noqa: E402
from kcg_connector.grasp.carts_v2.models import load_v2_inputs
from kcg_connector.grasp.robust.bounded_hand_base_ik import (
    solve_bounded_hand_base_ik,
)
from kcg_connector.grasp.robust.collision_roster import (
    load_authoritative_collision_link_roster,
)
from kcg_connector.grasp.robust.hand_model import rpy_rotation
from kcg_connector.grasp.robust.object_model import load_stl_mesh


def _matrix4(values: object, label: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.size != 16:
        raise ValueError(f"{label} must contain 16 values")
    matrix = matrix.reshape(4, 4)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{label} is not finite")
    return matrix


def _yaw_free_object_frame(world_from_object: np.ndarray) -> np.ndarray:
    """Keep observed position/+Z axis while fixing unobservable axial yaw."""
    z_axis = np.asarray(world_from_object[:3, 2], dtype=np.float64)
    z_axis /= np.linalg.norm(z_axis)
    for reference in (np.array((1.0, 0.0, 0.0)), np.array((0.0, 1.0, 0.0))):
        x_axis = reference - float(reference @ z_axis) * z_axis
        norm = float(np.linalg.norm(x_axis))
        if norm > 1.0e-8:
            x_axis /= norm
            break
    else:  # pragma: no cover - impossible for the two orthogonal references
        raise ValueError("cannot construct yaw-free object frame")
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    frame = np.eye(4, dtype=np.float64)
    frame[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
    frame[:3, 3] = world_from_object[:3, 3]
    return frame


def _fcl_distance(first: fcl.CollisionObject, second: fcl.CollisionObject) -> float:
    collision = fcl.collide(
        first,
        second,
        fcl.CollisionRequest(num_max_contacts=1, enable_contact=False),
        fcl.CollisionResult(),
    )
    if collision:
        return 0.0
    distance = float(
        fcl.distance(
            first,
            second,
            fcl.DistanceRequest(enable_nearest_points=False),
            fcl.DistanceResult(),
        )
    )
    if not math.isfinite(distance) or distance < 0.0:
        raise RuntimeError("FCL returned an invalid distance")
    return distance


class FullRobotCollisionScene:
    """All 17 collision links from the active robot/hand contract."""

    def __init__(self, inputs: object) -> None:
        roster = load_authoritative_collision_link_roster(
            inputs.config.section("inputs")["collision_roster"],
            repository_root=inputs.repository_root,
        )
        hand_triangles = inputs.hand_collision_triangles_by_link
        self.objects: dict[str, fcl.CollisionObject] = {}
        for link in roster.links:
            if link.link_name in hand_triangles:
                triangles = np.asarray(hand_triangles[link.link_name], dtype=np.float64)
            else:
                mesh, provenance = load_stl_mesh(
                    link.absolute_path,
                    unit=link.unit,
                    orient_outward=False,
                )
                if provenance.source_sha256 != link.sha256:
                    raise ValueError(f"collision mesh hash changed: {link.link_name}")
                triangles = np.asarray(mesh.face_vertices_m, dtype=np.float64)
                triangles = triangles * np.asarray(link.scale, dtype=np.float64)
                triangles = (
                    triangles @ rpy_rotation(link.origin_rpy_rad).T
                    + np.asarray(link.origin_xyz_m, dtype=np.float64)
                )
            self.objects[link.link_name] = fcl.CollisionObject(_fcl_model(triangles))
        expected = {link.link_name for link in roster.links}
        if set(self.objects) != expected or len(self.objects) != 17:
            raise ValueError("complete 17-link collision roster is unavailable")
        adjacent = {
            tuple(sorted((joint.parent_link, joint.child_link)))
            for joint in inputs.robot_model.joints.values()
            if joint.parent_link in self.objects and joint.child_link in self.objects
        }
        names = tuple(sorted(self.objects))
        self.pairs = tuple(
            (first, second)
            for index, first in enumerate(names)
            for second in names[index + 1 :]
            if tuple(sorted((first, second))) not in adjacent
        )
        self.inputs = inputs

    def state_clearance(
        self,
        arm_positions: np.ndarray,
        hand_positions: np.ndarray,
    ) -> tuple[float, tuple[str, str] | None]:
        transforms = self.inputs.robot_model.forward_kinematics(
            tuple(np.concatenate((arm_positions, hand_positions))),
            enforce_limits=False,
        )
        for name, collision_object in self.objects.items():
            transform = np.asarray(transforms[name], dtype=np.float64)
            collision_object.setTransform(
                fcl.Transform(transform[:3, :3], transform[:3, 3])
            )
        minimum = math.inf
        limiting_pair = None
        for first, second in self.pairs:
            distance = _fcl_distance(self.objects[first], self.objects[second])
            if distance == 0.0:
                return 0.0, (first, second)
            if distance < minimum:
                minimum = distance
                limiting_pair = (first, second)
        return minimum, limiting_pair


def _cylinder_from_mesh(
    mesh_path: Path,
    scale_to_m: float,
    world_from_object: np.ndarray,
) -> tuple[fcl.CollisionObject, dict[str, float]]:
    mesh = trimesh.load(mesh_path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"expected one mesh: {mesh_path}")
    vertices = np.asarray(mesh.vertices, dtype=np.float64) * float(scale_to_m)
    radius = float(np.max(np.linalg.norm(vertices[:, :2], axis=1)))
    z_min = float(np.min(vertices[:, 2]))
    z_max = float(np.max(vertices[:, 2]))
    length = z_max - z_min
    center_local = np.asarray((0.0, 0.0, 0.5 * (z_min + z_max), 1.0))
    center_world = (world_from_object @ center_local)[:3]
    return (
        fcl.CollisionObject(
            fcl.Cylinder(radius, length),
            fcl.Transform(world_from_object[:3, :3], center_world),
        ),
        {"radius_m": radius, "z_min_m": z_min, "z_max_m": z_max},
    )


def _save_figure(
    output_stem: Path,
    times: np.ndarray,
    joints: np.ndarray,
    hand_joints: np.ndarray,
    clearances: dict[str, np.ndarray],
) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(9.0, 7.2), sharex=True, constrained_layout=True)
    for index in range(joints.shape[1]):
        axes[0].plot(times, joints[:, index], linewidth=1.2, label=f"q{index + 1}")
    for index in range(hand_joints.shape[1]):
        axes[0].plot(
            times,
            hand_joints[:, index],
            linestyle="--",
            linewidth=0.9,
            label=f"h{index + 1}",
        )
    axes[0].set_ylabel("Joint position [rad]")
    axes[0].set_title("(a) Minimum-jerk handoff trajectory")
    axes[0].grid(alpha=0.25)
    axes[0].legend(ncol=4, fontsize=8)
    for name, values in clearances.items():
        axes[1].plot(times, 1000.0 * values, linewidth=1.3, label=name)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Minimum clearance [mm]")
    axes[1].set_title("(b) Discrete FCL clearance at every 240 Hz command state")
    axes[1].grid(alpha=0.25)
    axes[1].legend(ncol=3, fontsize=8)
    figure.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-result", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--grasp-relation", type=Path, required=True)
    parser.add_argument("--scene-provider-json", type=Path, required=True)
    parser.add_argument("--plug-mesh", type=Path, required=True)
    parser.add_argument("--receptacle-mesh", type=Path, required=True)
    parser.add_argument("--mesh-scale-to-m", type=float, default=0.001)
    parser.add_argument("--receptacle-position-m", nargs=3, type=float, required=True)
    parser.add_argument("--handoff-clearance-m", type=float, default=0.05)
    parser.add_argument("--observation-tilt-deg", type=float, default=0.0)
    parser.add_argument("--hand-camera-config", type=Path)
    parser.add_argument("--motion-duration-s", type=float, default=6.5)
    parser.add_argument("--preshape-duration-s", type=float, default=3.0)
    parser.add_argument("--rate-hz", type=int, default=240)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    repository = Path(__file__).resolve().parents[3]
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    pose_document = json.loads(args.pose_result.read_text(encoding="utf-8"))
    observed_world_from_object = _matrix4(
        pose_document["world_from_object_row_major"], "observed world pose"
    )
    world_from_object = _yaw_free_object_frame(observed_world_from_object)

    relation = yaml.safe_load(args.grasp_relation.read_text(encoding="utf-8"))
    object_from_hand = _matrix4(
        relation["transform"]["object_from_hand_base_row_major"],
        "object_from_hand_base",
    )
    hand_contract = relation["hand_contract"]
    control_plan = {
        "object_from_hand_row_major": object_from_hand.ravel().tolist(),
        "approach_direction_object": relation["transform"]["approach_direction_object"],
        "pregrasp_joint_positions_rad": hand_contract["pregrasp_joint_positions_rad"],
        "final_joint_positions_rad": hand_contract["final_joint_positions_rad"],
        "approach_high_seed_arm_positions_rad": hand_contract[
            "approach_high_seed_arm_positions_rad"
        ],
        "closing_order": hand_contract["closing_order"],
    }
    inputs = load_v2_inputs(
        repository,
        config_path=args.config.resolve(),
        object_id=args.object_id,
    )
    motion_plan = control.build_joint_motion_plan(
        repository,
        inputs,
        control_plan,
        world_from_object,
        include_lift=False,
    )
    approach_direction = np.asarray(
        motion_plan["approach_direction_world"], dtype=np.float64
    )
    pregrasp_world_from_hand = _matrix4(
        motion_plan["world_from_hand_base_target"], "planned pregrasp"
    )
    handoff_world_from_hand = pregrasp_world_from_hand.copy()
    handoff_world_from_hand[:3, 3] -= approach_direction * float(
        args.handoff_clearance_m
    )
    handoff_arm = np.asarray(
        motion_plan["approach_arm_waypoints_rad"][0], dtype=np.float64
    )
    hand_positions = np.asarray(
        motion_plan["pregrasp_hand_positions_rad"], dtype=np.float64
    )
    tilted_camera_pose = None
    tilted_ik_position_error = None
    tilted_ik_rotation_error = None
    tilt_degrees = float(args.observation_tilt_deg)
    if not math.isfinite(tilt_degrees):
        raise ValueError("observation tilt must be finite")
    if abs(tilt_degrees) > 0.0:
        if args.hand_camera_config is None:
            raise ValueError("observation tilt requires hand-camera config")
        rig = yaml.safe_load(
            args.hand_camera_config.read_text(encoding="utf-8")
        )["camera_rig"]
        hand_from_camera = np.asarray(
            rig["palm"]["T_HC_cv"], dtype=np.float64
        )
        if hand_from_camera.shape != (4, 4):
            raise ValueError("palm hand-eye transform is invalid")
        original_camera = handoff_world_from_hand @ hand_from_camera
        plug_mesh = trimesh.load(args.plug_mesh, process=False)
        if not isinstance(plug_mesh, trimesh.Trimesh):
            raise ValueError("plug mesh did not load as one Trimesh")
        vertices_m = (
            np.asarray(plug_mesh.vertices, dtype=np.float64)
            * float(args.mesh_scale_to_m)
        )
        center_local = 0.5 * (vertices_m.min(axis=0) + vertices_m.max(axis=0))
        center_world = (
            world_from_object
            @ np.concatenate((center_local, np.asarray((1.0,))))
        )[:3]
        viewing_distance = float(
            np.linalg.norm(center_world - original_camera[:3, 3])
        )
        angle = math.radians(tilt_degrees)
        rotation_about_camera_y = np.asarray(
            (
                (math.cos(angle), 0.0, math.sin(angle)),
                (0.0, 1.0, 0.0),
                (-math.sin(angle), 0.0, math.cos(angle)),
            ),
            dtype=np.float64,
        )
        tilted_camera_pose = original_camera.copy()
        tilted_camera_pose[:3, :3] = (
            original_camera[:3, :3] @ rotation_about_camera_y
        )
        tilted_camera_pose[:3, 3] = (
            center_world
            - tilted_camera_pose[:3, 2] * viewing_distance
        )
        handoff_world_from_hand = (
            tilted_camera_pose @ np.linalg.inv(hand_from_camera)
        )
        solved, tilted_ik_position_error, tilted_ik_rotation_error, _ = (
            solve_bounded_hand_base_ik(
                inputs.config.section("ik")["solver"],
                model=inputs.robot_model,
                hand_positions=hand_positions,
                target_world_from_hand_base=handoff_world_from_hand,
                seed_arm_positions=(handoff_arm,),
                label="VISUAL_OBLIQUE_HANDOFF",
            )
        )
        handoff_arm = np.asarray(solved, dtype=np.float64)
    handoff_fk = np.asarray(
        inputs.robot_model.forward_kinematics(
            tuple(np.concatenate((handoff_arm, hand_positions))),
            enforce_limits=False,
        )["handbase_link"],
        dtype=np.float64,
    )
    handoff_position_error = float(
        np.linalg.norm(handoff_fk[:3, 3] - handoff_world_from_hand[:3, 3])
    )
    rotation_relative = handoff_fk[:3, :3].T @ handoff_world_from_hand[:3, :3]
    handoff_rotation_error = float(
        math.acos(np.clip(0.5 * (np.trace(rotation_relative) - 1.0), -1.0, 1.0))
    )

    if (
        args.rate_hz <= 0
        or args.motion_duration_s <= 0.0
        or args.preshape_duration_s <= 0.0
    ):
        raise ValueError("trajectory rate and durations must be positive")
    preshape_count = round(float(args.preshape_duration_s) * int(args.rate_hz))
    approach_count = round(float(args.motion_duration_s) * int(args.rate_hz))
    preshape_blends = np.asarray(
        [
            control.minimum_jerk_blend(index / preshape_count)
            for index in range(preshape_count + 1)
        ]
    )
    approach_blends = np.asarray(
        [
            control.minimum_jerk_blend(index / approach_count)
            for index in range(approach_count + 1)
        ]
    )
    preshape_arms = np.zeros((preshape_count + 1, 7), dtype=np.float64)
    preshape_hands = preshape_blends[:, None] * hand_positions[None, :]
    approach_arms = approach_blends[:, None] * handoff_arm[None, :]
    approach_hands = np.repeat(
        hand_positions[None, :], approach_count + 1, axis=0
    )
    arm_states = np.concatenate((preshape_arms, approach_arms[1:]), axis=0)
    hand_states = np.concatenate((preshape_hands, approach_hands[1:]), axis=0)
    total_duration = float(args.preshape_duration_s + args.motion_duration_s)
    times = np.arange(len(arm_states), dtype=np.float64) / float(args.rate_hz)
    if not math.isclose(times[-1], total_duration, rel_tol=0.0, abs_tol=1.0e-12):
        raise RuntimeError("trajectory duration and state count differ")
    active_states = np.concatenate((arm_states, hand_states), axis=1)
    maximum_joint_step = float(np.max(np.abs(np.diff(active_states, axis=0))))

    self_scene = FullRobotCollisionScene(inputs)
    provider = json.loads(args.scene_provider_json.read_text(encoding="utf-8"))
    geometry = provider["known_static_scene_geometry"]
    table_bounds = np.asarray(inputs.table_xy_bounds_m, dtype=np.float64)
    table_size = np.asarray(
        (table_bounds[0, 1] - table_bounds[0, 0], table_bounds[1, 1] - table_bounds[1, 0], 1.0)
    )
    table_center = np.asarray(
        (np.mean(table_bounds[0]), np.mean(table_bounds[1]), inputs.table_top_z_m - 0.5)
    )
    obstacles: dict[str, fcl.CollisionObject] = {
        "table": fcl.CollisionObject(fcl.Box(*table_size), fcl.Transform(table_center))
    }
    fixture = geometry["fixture"]
    obstacles["fixture"] = fcl.CollisionObject(
        fcl.Box(*np.asarray(fixture["size_m"], dtype=np.float64)),
        fcl.Transform(np.asarray(fixture["center_world_m"], dtype=np.float64)),
    )
    plug_obstacle, plug_cylinder = _cylinder_from_mesh(
        args.plug_mesh.resolve(), args.mesh_scale_to_m, world_from_object
    )
    obstacles["plug"] = plug_obstacle
    receptacle_world = np.eye(4, dtype=np.float64)
    receptacle_world[:3, 3] = np.asarray(args.receptacle_position_m, dtype=np.float64)
    receptacle_obstacle, receptacle_cylinder = _cylinder_from_mesh(
        args.receptacle_mesh.resolve(), args.mesh_scale_to_m, receptacle_world
    )
    obstacles["receptacle"] = receptacle_obstacle

    clearance_rows = {
        "self": np.empty(len(arm_states), dtype=np.float64),
        **{
            name: np.empty(len(arm_states), dtype=np.float64)
            for name in obstacles
        },
    }
    first_collision: dict[str, object] | None = None
    limiting_self_pair = None
    minimum_self_seen = math.inf
    limiting_environment_links: dict[str, str | None] = {
        name: None for name in obstacles
    }
    minimum_environment_seen = {name: math.inf for name in obstacles}
    for state_index, (arm, hand_state) in enumerate(zip(arm_states, hand_states)):
        self_clearance, self_pair = self_scene.state_clearance(arm, hand_state)
        clearance_rows["self"][state_index] = self_clearance
        if self_clearance < minimum_self_seen:
            minimum_self_seen = self_clearance
            limiting_self_pair = self_pair
        if self_clearance == 0.0 and self_pair is not None and first_collision is None:
            first_collision = {
                "state_index": state_index,
                "kind": "self",
                "pair": list(self_pair),
            }
        for obstacle_name, obstacle in obstacles.items():
            minimum = math.inf
            limiting_link = None
            for link_name, link_object in self_scene.objects.items():
                distance = _fcl_distance(link_object, obstacle)
                if distance < minimum:
                    minimum = distance
                    limiting_link = link_name
            clearance_rows[obstacle_name][state_index] = minimum
            if minimum < minimum_environment_seen[obstacle_name]:
                minimum_environment_seen[obstacle_name] = minimum
                limiting_environment_links[obstacle_name] = limiting_link
            if minimum == 0.0 and first_collision is None:
                first_collision = {
                    "state_index": state_index,
                    "kind": "environment",
                    "link": limiting_link,
                    "obstacle": obstacle_name,
                }

    minimum_clearances = {
        name: float(np.min(values)) for name, values in clearance_rows.items()
    }
    minimum_clearance_indices = {
        name: int(np.argmin(values)) for name, values in clearance_rows.items()
    }
    collision_free = first_collision is None and all(
        value > 0.0 for value in minimum_clearances.values()
    )
    result = {
        "schema_version": "kcg_visual_handoff_plan_v1",
        "method": "YAW_FREE_BOUNDED_IK_MINIMUM_JERK_WITH_FCL_240HZ",
        "pose_provider": pose_document.get("pose_provider", "UNSPECIFIED"),
        "pose_result": str(args.pose_result.resolve()),
        "full_rotation_consumed": False,
        "position_and_outward_axis_consumed": True,
        "inputs": {
            "config": str(args.config.resolve()),
            "object_id": args.object_id,
            "grasp_relation": str(args.grasp_relation.resolve()),
            "scene_provider_json": str(args.scene_provider_json.resolve()),
            "plug_mesh": str(args.plug_mesh.resolve()),
            "receptacle_mesh": str(args.receptacle_mesh.resolve()),
            "hand_camera_config": (
                None
                if args.hand_camera_config is None
                else str(args.hand_camera_config.resolve())
            ),
        },
        "yaw_policy": "PROJECT_WORLD_X_ONTO_AXIS_NORMAL_PLANE",
        "observed_world_from_object_row_major": observed_world_from_object.ravel().tolist(),
        "planned_world_from_object_row_major": world_from_object.ravel().tolist(),
        "world_from_hand_pregrasp_row_major": pregrasp_world_from_hand.ravel().tolist(),
        "world_from_hand_handoff_row_major": handoff_world_from_hand.ravel().tolist(),
        "handoff_clearance_m": float(args.handoff_clearance_m),
        "observation_tilt_deg": tilt_degrees,
        "world_from_palm_camera_handoff_row_major": (
            None
            if tilted_camera_pose is None
            else tilted_camera_pose.ravel().tolist()
        ),
        "tilted_ik_position_error_m": tilted_ik_position_error,
        "tilted_ik_rotation_error_rad": tilted_ik_rotation_error,
        "handoff_arm_joint_target_rad": handoff_arm.tolist(),
        "handoff_hand_joint_target_rad": hand_positions.tolist(),
        "handoff_fk_position_error_m": handoff_position_error,
        "handoff_fk_rotation_error_rad": handoff_rotation_error,
        "trajectory": {
            "total_duration_s": total_duration,
            "preshape_duration_s": float(args.preshape_duration_s),
            "approach_duration_s": float(args.motion_duration_s),
            "rate_hz": int(args.rate_hz),
            "state_count": len(arm_states),
            "preshape_state_count": preshape_count + 1,
            "approach_state_count_including_shared_start": approach_count + 1,
            "maximum_active_joint_step_rad": maximum_joint_step,
            "continuous_collision_proof_claimed": False,
        },
        "collision": {
            "collision_free_at_all_discrete_states": collision_free,
            "first_collision": first_collision,
            "minimum_clearance_m": minimum_clearances,
            "minimum_clearance_state_index": minimum_clearance_indices,
            "limiting_self_pair": (
                None if limiting_self_pair is None else list(limiting_self_pair)
            ),
            "limiting_environment_link": limiting_environment_links,
            "robot_collision_link_count": len(self_scene.objects),
            "nonadjacent_self_pair_count": len(self_scene.pairs),
            "obstacles": ["table", "fixture", "plug", "receptacle"],
            "plug_yaw_swept_cylinder": plug_cylinder,
            "receptacle_cylinder": receptacle_cylinder,
            "evidence_limit": "DISCRETE_RUNTIME_COMMAND_STATES_NOT_CONTINUOUS_PROOF",
        },
        "motion_authorized_by_this_file": False,
        "hardware_authorized": False,
    }
    (output / "handoff_plan.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        output / "handoff_trajectory.npz",
        time_s=times,
        arm_joint_positions_rad=arm_states,
        hand_joint_positions_rad=hand_states,
        **{f"clearance_{name}_m": values for name, values in clearance_rows.items()},
    )
    _save_figure(
        output / "figure_handoff_trajectory_and_clearance",
        times,
        arm_states,
        hand_states,
        clearance_rows,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if collision_free else 2


if __name__ == "__main__":
    raise SystemExit(main())
