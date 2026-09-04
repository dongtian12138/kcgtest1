#!/usr/bin/env python3
"""Plan one visual hand-base pose with pick_ik and MoveIt RRTConnect.

The process is deliberately plan-only.  Isaac remains the sole trajectory
executor and performs the final task-specific collision check before motion.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

from geometry_msgs.msg import Pose
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    CollisionObject,
    Constraints,
    JointConstraint,
    MoveItErrorCodes,
    OrientationConstraint,
    PositionConstraint,
)
from moveit_msgs.srv import ApplyPlanningScene, GetPositionIK
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from shape_msgs.msg import SolidPrimitive


def _finite_vector(value, length: int, label: str) -> list[float]:
    result = [float(item) for item in value]
    if len(result) != length or not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} must contain {length} finite values")
    return result


def _pose_from_matrix(values, label: str) -> Pose:
    flat = _finite_vector(values, 16, label)
    matrix = [flat[index : index + 4] for index in range(0, 16, 4)]
    if any(abs(matrix[3][index] - expected) > 1.0e-12 for index, expected in enumerate((0.0, 0.0, 0.0, 1.0))):
        raise ValueError(f"{label} has an invalid homogeneous row")
    quaternion = Rotation.from_matrix(
        [row[:3] for row in matrix[:3]]
    ).as_quat()
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = (
        matrix[0][3], matrix[1][3], matrix[2][3]
    )
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = (
        float(quaternion[0]),
        float(quaternion[1]),
        float(quaternion[2]),
        float(quaternion[3]),
    )
    return pose


def _collision_object(document: dict, frame_id: str) -> CollisionObject:
    result = CollisionObject()
    result.header.frame_id = frame_id
    result.id = str(document["id"])
    result.operation = CollisionObject.ADD
    primitive = SolidPrimitive()
    kind = str(document["type"]).lower()
    dimensions = document["dimensions_m"]
    if kind == "box":
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = _finite_vector(dimensions, 3, "box dimensions")
    elif kind == "cylinder":
        primitive.type = SolidPrimitive.CYLINDER
        primitive.dimensions = _finite_vector(
            dimensions, 2, "cylinder [height, radius]"
        )
    else:
        raise ValueError(f"unsupported collision primitive: {kind}")
    if any(value <= 0.0 for value in primitive.dimensions):
        raise ValueError(f"collision primitive {result.id} is non-positive")
    result.primitives = [primitive]
    result.primitive_poses = [
        _pose_from_matrix(document["world_from_primitive_row_major"], result.id)
    ]
    return result


def _json_ready_trajectory(trajectory) -> dict:
    joint = trajectory.joint_trajectory
    return {
        "joint_names": list(joint.joint_names),
        "points": [
            {
                "time_from_start_s": (
                    float(point.time_from_start.sec)
                    + 1.0e-9 * float(point.time_from_start.nanosec)
                ),
                "positions_rad": list(map(float, point.positions)),
                "velocities_rad_s": list(map(float, point.velocities)),
                "accelerations_rad_s2": list(map(float, point.accelerations)),
            }
            for point in joint.points
        ],
    }


class VisualPosePlanner(Node):
    def __init__(self) -> None:
        super().__init__("kcg_visual_pose_planner")
        self._client = ActionClient(self, MoveGroup, "/move_action")
        self._apply_scene = self.create_client(
            ApplyPlanningScene, "/apply_planning_scene"
        )
        self._compute_ik = self.create_client(GetPositionIK, "/compute_ik")

    def plan(self, request: dict) -> dict:
        server_timeout = float(request.get("server_timeout_s", 20.0))
        if not self._client.wait_for_server(timeout_sec=server_timeout):
            raise RuntimeError("MoveIt /move_action is unavailable")
        if not self._apply_scene.wait_for_service(timeout_sec=server_timeout):
            raise RuntimeError("MoveIt /apply_planning_scene is unavailable")
        if not self._compute_ik.wait_for_service(timeout_sec=server_timeout):
            raise RuntimeError("MoveIt /compute_ik is unavailable")

        frame_id = str(request.get("frame_id", "world"))
        target_link = str(request.get("target_link", "handbase_link"))
        start_names = [str(name) for name in request["start_joint_names"]]
        start_positions = _finite_vector(
            request["start_joint_positions_rad"], len(start_names), "start joints"
        )
        if len(start_names) != len(set(start_names)):
            raise ValueError("start joint names are not unique")
        target_pose = _pose_from_matrix(
            request["world_from_target_link_row_major"], "target pose"
        )
        position_tolerance = float(request.get("position_tolerance_m", 1.0e-4))
        orientation_tolerance = float(
            request.get("orientation_tolerance_rad", 5.0e-4)
        )
        if (
            not math.isfinite(position_tolerance)
            or position_tolerance <= 0.0
            or not math.isfinite(orientation_tolerance)
            or orientation_tolerance <= 0.0
        ):
            raise ValueError("pose tolerances must be positive and finite")

        position = PositionConstraint()
        position.header.frame_id = frame_id
        position.link_name = target_link
        position.weight = 1.0
        position_region = SolidPrimitive()
        position_region.type = SolidPrimitive.SPHERE
        position_region.dimensions = [position_tolerance]
        region_pose = Pose()
        region_pose.position = target_pose.position
        region_pose.orientation.w = 1.0
        position.constraint_region.primitives = [position_region]
        position.constraint_region.primitive_poses = [region_pose]

        orientation = OrientationConstraint()
        orientation.header.frame_id = frame_id
        orientation.link_name = target_link
        orientation.orientation = target_pose.orientation
        orientation.absolute_x_axis_tolerance = orientation_tolerance
        orientation.absolute_y_axis_tolerance = orientation_tolerance
        orientation.absolute_z_axis_tolerance = orientation_tolerance
        orientation.weight = 1.0

        path_constraints = Constraints()
        path_constraints.name = "positive_joint_limit_margin"
        for name, bounds in request.get("path_joint_bounds_rad", {}).items():
            lower, upper = _finite_vector(bounds, 2, f"bounds for {name}")
            if lower >= upper:
                raise ValueError(f"reversed bounds for {name}")
            path_constraints.joint_constraints.append(
                JointConstraint(
                    joint_name=str(name),
                    position=0.5 * (lower + upper),
                    tolerance_below=0.5 * (upper - lower),
                    tolerance_above=0.5 * (upper - lower),
                    weight=1.0,
                )
            )

        collision_objects = [
            _collision_object(item, frame_id)
            for item in request.get("collision_objects", [])
        ]
        scene_request = ApplyPlanningScene.Request()
        scene_request.scene.is_diff = True
        scene_request.scene.robot_state.is_diff = True
        scene_request.scene.world.collision_objects = collision_objects
        scene_future = self._apply_scene.call_async(scene_request)
        rclpy.spin_until_future_complete(
            self, scene_future, timeout_sec=server_timeout
        )
        if (
            not scene_future.done()
            or scene_future.result() is None
            or not scene_future.result().success
        ):
            raise RuntimeError("MoveIt rejected the planning-scene collision objects")

        ik_request = GetPositionIK.Request()
        ik_request.ik_request.group_name = str(
            request.get("group_name", "kuka")
        )
        ik_request.ik_request.robot_state.joint_state.header.frame_id = frame_id
        ik_request.ik_request.robot_state.joint_state.name = start_names
        ik_request.ik_request.robot_state.joint_state.position = start_positions
        ik_request.ik_request.robot_state.is_diff = False
        ik_request.ik_request.constraints = path_constraints
        ik_request.ik_request.avoid_collisions = True
        ik_request.ik_request.ik_link_name = target_link
        ik_request.ik_request.pose_stamped.header.frame_id = frame_id
        ik_request.ik_request.pose_stamped.pose = target_pose
        ik_timeout = float(request.get("ik_timeout_s", 0.2))
        if not math.isfinite(ik_timeout) or ik_timeout <= 0.0:
            raise ValueError("IK timeout must be positive and finite")
        ik_request.ik_request.timeout.sec = int(ik_timeout)
        ik_request.ik_request.timeout.nanosec = int(
            round(1.0e9 * (ik_timeout - int(ik_timeout)))
        )
        ik_started = time.perf_counter()
        ik_future = self._compute_ik.call_async(ik_request)
        rclpy.spin_until_future_complete(
            self, ik_future, timeout_sec=server_timeout + ik_timeout
        )
        ik_elapsed = time.perf_counter() - ik_started
        if not ik_future.done() or ik_future.result() is None:
            raise RuntimeError("timed out waiting for pick_ik")
        ik_response = ik_future.result()
        if ik_response.error_code.val != MoveItErrorCodes.SUCCESS:
            return {
                "schema_version": "kcg_moveit_pick_ik_rrtconnect_plan_v1",
                "success": False,
                "error_code": int(ik_response.error_code.val),
                "failure_stage": "pick_ik",
                "ik_elapsed_s": ik_elapsed,
                "planning_time_s": 0.0,
                "group_name": ik_request.ik_request.group_name,
                "target_link": target_link,
                "pipeline_id": "ompl",
                "planner_id": str(
                    request.get("planner_id", "RRTConnect")
                ),
                "ik_solver": "pick_ik/PickIkPlugin",
                "ik_mode": "global",
                "plan_only": True,
                "collision_object_ids": [item.id for item in collision_objects],
                "trajectory": None,
            }
        solution_by_name = dict(
            zip(
                ik_response.solution.joint_state.name,
                ik_response.solution.joint_state.position,
            )
        )
        group_joint_names = [
            name for name in start_names if name.startswith("iiwa_joint_")
        ]
        if len(group_joint_names) != 7 or any(
            name not in solution_by_name for name in group_joint_names
        ):
            raise RuntimeError("pick_ik returned an incomplete arm solution")
        ik_solution = [float(solution_by_name[name]) for name in group_joint_names]
        for name, value in zip(group_joint_names, ik_solution):
            if name in request.get("path_joint_bounds_rad", {}):
                lower, upper = request["path_joint_bounds_rad"][name]
                if not float(lower) <= value <= float(upper):
                    raise RuntimeError(
                        f"pick_ik solution violates the requested bound for {name}"
                    )

        goal_constraints = Constraints()
        goal_constraints.name = "pick_ik_joint_solution"
        joint_goal_tolerance = float(
            request.get("joint_goal_tolerance_rad", 1.0e-4)
        )
        goal_constraints.joint_constraints = [
            JointConstraint(
                joint_name=name,
                position=value,
                tolerance_below=joint_goal_tolerance,
                tolerance_above=joint_goal_tolerance,
                weight=1.0,
            )
            for name, value in zip(group_joint_names, ik_solution)
        ]

        goal = MoveGroup.Goal()
        goal.request.group_name = str(request.get("group_name", "kuka"))
        goal.request.pipeline_id = "ompl"
        goal.request.planner_id = str(
            request.get("planner_id", "RRTConnect")
        )
        goal.request.num_planning_attempts = int(
            request.get("num_planning_attempts", 4)
        )
        goal.request.allowed_planning_time = float(
            request.get("allowed_planning_time_s", 5.0)
        )
        goal.request.max_velocity_scaling_factor = float(
            request.get("maximum_velocity_scaling_factor", 0.1)
        )
        goal.request.max_acceleration_scaling_factor = float(
            request.get("maximum_acceleration_scaling_factor", 0.1)
        )
        goal.request.workspace_parameters.header.frame_id = frame_id
        workspace_min = _finite_vector(
            request.get("workspace_min_m", [-1.0, -1.0, -0.1]),
            3,
            "workspace minimum",
        )
        workspace_max = _finite_vector(
            request.get("workspace_max_m", [1.5, 1.5, 2.0]),
            3,
            "workspace maximum",
        )
        (
            goal.request.workspace_parameters.min_corner.x,
            goal.request.workspace_parameters.min_corner.y,
            goal.request.workspace_parameters.min_corner.z,
        ) = workspace_min
        (
            goal.request.workspace_parameters.max_corner.x,
            goal.request.workspace_parameters.max_corner.y,
            goal.request.workspace_parameters.max_corner.z,
        ) = workspace_max
        goal.request.start_state.joint_state.header.frame_id = frame_id
        goal.request.start_state.joint_state.name = start_names
        goal.request.start_state.joint_state.position = start_positions
        goal.request.start_state.is_diff = False
        goal.request.goal_constraints = [goal_constraints]
        goal.request.path_constraints = path_constraints
        goal.planning_options.plan_only = True
        goal.planning_options.look_around = False
        goal.planning_options.replan = False
        scene = goal.planning_options.planning_scene_diff
        scene.is_diff = True
        scene.robot_state.is_diff = True
        scene.world.collision_objects = collision_objects

        send_future = self._client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=server_timeout)
        if not send_future.done() or send_future.result() is None:
            raise RuntimeError("timed out sending MoveIt planning goal")
        goal_handle = send_future.result()
        if not goal_handle.accepted:
            raise RuntimeError("MoveIt rejected planning goal")
        wait_timeout = goal.request.allowed_planning_time + server_timeout
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=wait_timeout)
        if not result_future.done() or result_future.result() is None:
            raise RuntimeError("timed out waiting for MoveIt planning result")
        result = result_future.result().result
        success = result.error_code.val == MoveItErrorCodes.SUCCESS
        returned_points = result.planned_trajectory.joint_trajectory.points
        return {
            "schema_version": "kcg_moveit_pick_ik_rrtconnect_plan_v1",
            "success": success,
            "error_code": int(result.error_code.val),
            "failure_stage": None if success else "rrtconnect",
            "ik_elapsed_s": ik_elapsed,
            "ik_solution_rad": dict(zip(group_joint_names, ik_solution)),
            "planning_time_s": float(result.planning_time),
            "group_name": goal.request.group_name,
            "target_link": target_link,
            "pipeline_id": goal.request.pipeline_id,
            "planner_id": goal.request.planner_id,
            "ik_solver": "pick_ik/PickIkPlugin",
            "ik_mode": "global",
            "plan_only": True,
            "collision_object_ids": [
                item.id for item in scene.world.collision_objects
            ],
            "trajectory": (
                _json_ready_trajectory(result.planned_trajectory)
                if returned_points
                else None
            ),
            "trajectory_validated_by_moveit": success,
        }


def _read_request(path: str) -> dict:
    text = sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("planning request must be one JSON object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    request = _read_request(args.request)
    rclpy.init()
    node = VisualPosePlanner()
    try:
        result = node.plan(request)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        sys.stdout.write(output)
    else:
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    return 0 if result["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
