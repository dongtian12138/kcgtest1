#!/usr/bin/env python3

import sys

import rclpy
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes
from rclpy.action import ActionClient
from rclpy.node import Node


SAFE_GOALS = (
    (
        "kuka",
        {
            "iiwa_joint_1": 0.08,
            "iiwa_joint_2": -0.12,
            "iiwa_joint_3": 0.08,
            "iiwa_joint_4": -0.16,
            "iiwa_joint_5": 0.04,
            "iiwa_joint_6": 0.10,
            "iiwa_joint_7": -0.04,
        },
    ),
    (
        "hand",
        {
            "f1j1": 0.15,
            "f1j2": 0.12,
            "f2j1": 0.15,
            "f3j2": 0.12,
        },
    ),
)


class MoveItSmokeTest(Node):
    def __init__(self):
        super().__init__("kcg_moveit_smoke_test")
        self.client = ActionClient(self, MoveGroup, "/move_action")

    def execute_joint_goal(self, group_name, targets):
        if not self.client.wait_for_server(timeout_sec=15.0):
            self.get_logger().error("MoveIt action /move_action is unavailable")
            return False

        constraints = Constraints(name=f"{group_name}_smoke_goal")
        constraints.joint_constraints = [
            JointConstraint(
                joint_name=name,
                position=value,
                tolerance_above=0.01,
                tolerance_below=0.01,
                weight=1.0,
            )
            for name, value in targets.items()
        ]

        goal = MoveGroup.Goal()
        goal.request.group_name = group_name
        goal.request.pipeline_id = "ompl"
        goal.request.num_planning_attempts = 2
        goal.request.allowed_planning_time = 5.0
        goal.request.max_velocity_scaling_factor = 0.1
        goal.request.max_acceleration_scaling_factor = 0.1
        goal.request.start_state.is_diff = True
        goal.request.goal_constraints = [constraints]
        goal.planning_options.plan_only = False
        goal.planning_options.look_around = False
        goal.planning_options.replan = False
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True

        self.get_logger().info(f"Planning and executing group '{group_name}'")
        send_future = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=15.0)
        if not send_future.done() or send_future.result() is None:
            self.get_logger().error(f"Timed out sending goal for '{group_name}'")
            return False

        goal_handle = send_future.result()
        if not goal_handle.accepted:
            self.get_logger().error(f"MoveIt rejected goal for '{group_name}'")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=45.0)
        if not result_future.done() or result_future.result() is None:
            self.get_logger().error(f"Timed out executing goal for '{group_name}'")
            return False

        result = result_future.result().result
        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(
                f"MoveIt goal for '{group_name}' failed with code "
                f"{result.error_code.val}"
            )
            return False

        self.get_logger().info(
            f"MoveIt goal for '{group_name}' succeeded "
            f"(planning_time={result.planning_time:.3f} s)"
        )
        return True


def main():
    rclpy.init()
    node = MoveItSmokeTest()
    passed = all(
        node.execute_joint_goal(group_name, targets)
        for group_name, targets in SAFE_GOALS
    )
    if passed:
        node.get_logger().info("MoveIt smoke test PASSED")
    else:
        node.get_logger().error("MoveIt smoke test FAILED")
    node.destroy_node()
    rclpy.shutdown()
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
