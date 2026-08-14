#!/usr/bin/env python3

import json
import sys
import time

import rclpy
from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.srv import SwitchController
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectoryPoint


class ScriptedCylinderGrasp(Node):
    def __init__(self):
        super().__init__("scripted_cylinder_grasp")
        self.arm_joint_names = list(
            self.declare_parameter(
                "arm_joint_names",
                [f"iiwa_joint_{index}" for index in range(1, 8)],
            ).value
        )
        self.hand_joint_names = list(
            self.declare_parameter(
                "hand_joint_names", ["f1j1", "f1j2", "f2j1", "f3j2"]
            ).value
        )
        self.hand_controller_joint_names = list(
            self.declare_parameter(
                "hand_controller_joint_names",
                [
                    "f1j1",
                    "f1j2",
                    "f2j1",
                    "f3j2",
                    "f3j1",
                    "f1j3",
                    "f2j2",
                    "f3j3",
                ],
            ).value
        )
        self.approach_positions = list(
            self.declare_parameter(
                "approach_joint_positions", [0.0] * 7
            ).value
        )
        self.pre_approach_positions = list(
            self.declare_parameter(
                "pre_approach_joint_positions", [0.0] * 7
            ).value
        )
        self.lift_positions = list(
            self.declare_parameter("lift_joint_positions", [0.0] * 7).value
        )
        self.open_positions = list(
            self.declare_parameter("open_hand_positions", [0.0] * 4).value
        )
        self.closed_positions = list(
            self.declare_parameter("closed_hand_positions", [0.5] * 4).value
        )
        self.approach_velocity_scale = float(
            self.declare_parameter("approach_velocity_scale", 0.08).value
        )
        self.approach_acceleration_scale = float(
            self.declare_parameter("approach_acceleration_scale", 0.08).value
        )
        self.approach_motion_duration = float(
            self.declare_parameter("approach_motion_duration", 15.0).value
        )
        self.pre_approach_motion_duration = float(
            self.declare_parameter(
                "pre_approach_motion_duration", 15.0
            ).value
        )
        self.use_moveit_for_approach = bool(
            self.declare_parameter("use_moveit_for_approach", False).value
        )
        self.hand_motion_duration = float(
            self.declare_parameter("hand_motion_duration", 3.0).value
        )
        self.lift_motion_duration = float(
            self.declare_parameter("lift_motion_duration", 5.0).value
        )
        self.settle_duration = float(
            self.declare_parameter("settle_duration", 1.0).value
        )
        self.hold_duration = float(
            self.declare_parameter("hold_duration", 4.0).value
        )
        self.perform_approach = bool(
            self.declare_parameter("perform_approach", True).value
        )
        self.activate_hand_controller_at_runtime = bool(
            self.declare_parameter(
                "activate_hand_controller_at_runtime", False
            ).value
        )

        self.move_group_client = ActionClient(self, MoveGroup, "/move_action")
        self.arm_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/controller_gazebo_kuka/follow_joint_trajectory",
        )
        self.hand_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/controller_gazebo_hand/follow_joint_trajectory",
        )
        self.reset_client = self.create_client(Trigger, "/kcg_grasp/reset")
        self.evaluate_client = self.create_client(Trigger, "/kcg_grasp/evaluate")
        self.switch_controller_client = self.create_client(
            SwitchController, "/controller_manager/switch_controller"
        )
        self.phase_publisher = self.create_publisher(
            String, "/kcg_grasp/phase", 10
        )
        self.latest_state = None
        self.seen_joint_names = set()
        self.create_subscription(
            String, "/kcg_grasp/task_state", self._state_callback, 10
        )
        self.create_subscription(
            JointState, "/joint_states", self._joint_state_callback, 10
        )

    def _state_callback(self, message):
        try:
            self.latest_state = json.loads(message.data)
        except json.JSONDecodeError:
            self.latest_state = {"raw": message.data}

    def _joint_state_callback(self, message):
        self.seen_joint_names.update(message.name)

    def _wait_for_joint_states(self, timeout=30.0):
        required = set(self.arm_joint_names + self.hand_joint_names)
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if required.issubset(self.seen_joint_names):
                return True
        missing = sorted(required - self.seen_joint_names)
        self.get_logger().error(
            f"Timed out waiting for /joint_states; missing: {missing}"
        )
        return False

    def _phase(self, phase):
        self.get_logger().info(f"PHASE {phase}")
        self.phase_publisher.publish(String(data=phase))

    def _wait_sim(self, seconds):
        start = self.get_clock().now()
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            if (self.get_clock().now() - start).nanoseconds >= seconds * 1e9:
                return True
        return False

    def _trigger(self, client, name, timeout=10.0):
        if not client.wait_for_service(timeout_sec=timeout):
            self.get_logger().error(f"Service {name} is unavailable")
            return None
        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        if not future.done() or future.result() is None:
            self.get_logger().error(f"Service {name} timed out")
            return None
        return future.result()

    def _log_task_snapshot(self, label):
        result = self._trigger(self.evaluate_client, "/kcg_grasp/evaluate")
        if result is None:
            self.get_logger().warning(f"TASK SNAPSHOT {label}: unavailable")
            return
        self.get_logger().info(f"TASK SNAPSHOT {label}: {result.message}")

    def _activate_hand_controller(self):
        if not self.switch_controller_client.wait_for_service(timeout_sec=15.0):
            self.get_logger().error("Controller switch service is unavailable")
            return False
        request = SwitchController.Request()
        request.activate_controllers = ["controller_gazebo_hand"]
        request.strictness = SwitchController.Request.STRICT
        request.activate_asap = True
        request.timeout.sec = 10
        future = self.switch_controller_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)
        if not future.done() or future.result() is None or not future.result().ok:
            self.get_logger().error("Failed to activate physical hand controller")
            return False
        return True

    def _follow_trajectory(self, client, names, positions, duration, label):
        if len(names) != len(positions):
            self.get_logger().error(f"Invalid {label} trajectory dimensions")
            return False
        if not client.wait_for_server(timeout_sec=15.0):
            self.get_logger().error(f"Action server for {label} is unavailable")
            return False

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = names
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start = Duration(seconds=duration).to_msg()
        goal.trajectory.points = [point]

        send_future = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=15.0)
        if not send_future.done() or send_future.result() is None:
            self.get_logger().error(f"Timed out sending {label} trajectory")
            return False
        goal_handle = send_future.result()
        if not goal_handle.accepted:
            self.get_logger().error(f"{label} trajectory was rejected")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(
            self, result_future, timeout_sec=duration + 15.0
        )
        if not result_future.done() or result_future.result() is None:
            self.get_logger().error(f"{label} trajectory timed out")
            return False
        result = result_future.result().result
        if result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            self.get_logger().error(
                f"{label} trajectory failed: {result.error_string}"
            )
            return False
        return True

    def _expanded_hand_positions(self, active_positions):
        if len(self.hand_joint_names) != len(active_positions):
            raise ValueError("Invalid active hand position dimensions")
        active = dict(zip(self.hand_joint_names, active_positions))
        mapped = {
            **active,
            "f3j1": active["f1j1"],
            "f1j3": active["f1j2"],
            "f2j2": active["f2j1"],
            "f3j3": active["f3j2"],
        }
        return [mapped[name] for name in self.hand_controller_joint_names]

    def _moveit_approach(self):
        if len(self.arm_joint_names) != len(self.approach_positions):
            self.get_logger().error("Invalid MoveIt approach dimensions")
            return False
        if not self.move_group_client.wait_for_server(timeout_sec=20.0):
            self.get_logger().error("MoveIt /move_action is unavailable")
            return False

        constraints = Constraints(name="cylinder_pregrasp")
        constraints.joint_constraints = [
            JointConstraint(
                joint_name=name,
                position=position,
                tolerance_above=0.005,
                tolerance_below=0.005,
                weight=1.0,
            )
            for name, position in zip(
                self.arm_joint_names, self.approach_positions
            )
        ]

        goal = MoveGroup.Goal()
        goal.request.group_name = "kuka"
        goal.request.pipeline_id = "ompl"
        goal.request.num_planning_attempts = 5
        goal.request.allowed_planning_time = 8.0
        goal.request.max_velocity_scaling_factor = self.approach_velocity_scale
        goal.request.max_acceleration_scaling_factor = (
            self.approach_acceleration_scale
        )
        goal.request.start_state.is_diff = True
        goal.request.goal_constraints = [constraints]
        goal.planning_options.plan_only = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 2
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True

        send_future = self.move_group_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=20.0)
        if not send_future.done() or send_future.result() is None:
            self.get_logger().error("Timed out sending MoveIt approach")
            return False
        goal_handle = send_future.result()
        if not goal_handle.accepted:
            self.get_logger().error("MoveIt rejected the approach")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=90.0)
        if not result_future.done() or result_future.result() is None:
            self.get_logger().error("MoveIt approach timed out")
            return False
        result = result_future.result().result
        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(
                f"MoveIt approach failed with code {result.error_code.val}"
            )
            return False
        return True

    def run(self):
        if not self._wait_for_joint_states():
            return False

        if self.activate_hand_controller_at_runtime:
            self._phase("ACTIVATE_HAND")
            if not self._activate_hand_controller():
                return False

        self._phase("OPEN")
        if not self._follow_trajectory(
            self.hand_client,
            self.hand_controller_joint_names,
            self._expanded_hand_positions(self.open_positions),
            self.hand_motion_duration,
            "hand open",
        ):
            return False

        if self.perform_approach:
            if self.use_moveit_for_approach:
                self._phase("APPROACH")
                approached = self._moveit_approach()
            else:
                self._phase("PRE_APPROACH")
                approached = self._follow_trajectory(
                    self.arm_client,
                    self.arm_joint_names,
                    self.pre_approach_positions,
                    self.pre_approach_motion_duration,
                    "arm pre-approach",
                )
                if approached:
                    self._phase("APPROACH")
                    approached = self._follow_trajectory(
                        self.arm_client,
                        self.arm_joint_names,
                        self.approach_positions,
                        self.approach_motion_duration,
                        "arm approach",
                    )
            if not approached:
                return False
        else:
            self.get_logger().info(
                "Robot was initialized at the deterministic pregrasp"
            )

        # Start task accounting only after the robot reaches its pregrasp.
        # Natural demonstrations leave the visible cylinder untouched here;
        # the fast RL mode teleports it to its deterministic episode pose.
        self._phase("RESET")
        reset = self._trigger(self.reset_client, "/kcg_grasp/reset")
        if reset is None or not reset.success:
            self.get_logger().error(
                reset.message if reset is not None else "Cylinder reset failed"
            )
            return False
        self.get_logger().info(f"RESET RESULT: {reset.message}")
        self._wait_sim(self.settle_duration)
        self._log_task_snapshot("AFTER_RESET")

        self._phase("GRASP")
        if not self._follow_trajectory(
            self.hand_client,
            self.hand_controller_joint_names,
            self._expanded_hand_positions(self.closed_positions),
            self.hand_motion_duration,
            "hand close",
        ):
            return False
        self._wait_sim(self.settle_duration)
        self._log_task_snapshot("AFTER_GRASP")

        self._phase("LIFT")
        if not self._follow_trajectory(
            self.arm_client,
            self.arm_joint_names,
            self.lift_positions,
            self.lift_motion_duration,
            "arm lift",
        ):
            return False
        self._log_task_snapshot("AFTER_LIFT")

        self._phase("HOLD")
        self._wait_sim(self.hold_duration)
        self._log_task_snapshot("AFTER_HOLD")

        result = self._trigger(self.evaluate_client, "/kcg_grasp/evaluate")
        if result is None:
            return False
        if result.success:
            self._phase("PASSED")
            self.get_logger().info(f"SCRIPTED GRASP PASSED: {result.message}")
            return True

        self._phase("FAILED")
        self.get_logger().error(f"SCRIPTED GRASP FAILED: {result.message}")
        return False


def main():
    rclpy.init()
    node = ScriptedCylinderGrasp()
    try:
        passed = node.run()
    except KeyboardInterrupt:
        passed = False
    try:
        node.destroy_node()
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
