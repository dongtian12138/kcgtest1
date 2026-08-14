#!/usr/bin/env python3

import math
import sys

import rclpy
from gazebo_msgs.msg import LinkStates
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState


EXPECTED_JOINTS = {
    "iiwa_joint_1",
    "iiwa_joint_2",
    "iiwa_joint_3",
    "iiwa_joint_4",
    "iiwa_joint_5",
    "iiwa_joint_6",
    "iiwa_joint_7",
    "f1j1",
    "f1j2",
    "f1j3",
    "f2j1",
    "f2j2",
    "f3j1",
    "f3j2",
    "f3j3",
}

HAND_LINK_SUFFIXES = {
    "handbase_link",
    "f1Link1",
    "f1Link2",
    "f1Link3",
    "f2Link1",
    "f2Link2",
    "f3Link1",
    "f3Link2",
    "f3Link3",
}

MIMIC_RELATIONSHIPS = {
    "f1j3": "f1j2",
    "f2j2": "f2j1",
    "f3j1": "f1j1",
    "f3j3": "f3j2",
}


def normalize_joint_name(name):
    # gazebo_ros2_control in ROS 2 Humble exports mimic state interfaces
    # with this suffix; GenericSystem and URDF use the physical joint name.
    return name.removesuffix("_mimic")


class PhysicsStabilityCheck(Node):
    def __init__(self):
        super().__init__("kcg_physics_stability_check")
        self.duration = float(self.declare_parameter("duration", 30.0).value)
        self.joint_state_topic = str(
            self.declare_parameter(
                "joint_state_topic",
                "/physics_joint_state_broadcaster/joint_states",
            ).value
        )
        self.max_abs_velocity = float(
            self.declare_parameter("max_abs_velocity", 20.0).value
        )
        self.max_hand_distance = float(
            self.declare_parameter("max_hand_distance", 2.0).value
        )
        self.max_mimic_error = float(
            self.declare_parameter("max_mimic_error", 0.02).value
        )
        # With use_sim_time the node can be constructed before its first
        # /clock sample.  Starting at that zero timestamp made a delayed check
        # finish immediately instead of observing the requested duration.
        self.started_at = None
        self.joint_samples = 0
        self.link_samples = 0
        self.seen_joints = set()
        self.max_seen_abs_velocity = 0.0
        self.max_seen_hand_distance = 0.0
        self.max_seen_mimic_error = 0.0
        self.failures = []
        self.finished = False
        self.result = 1

        self.create_subscription(
            JointState,
            self.joint_state_topic,
            self.on_joint_state,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            LinkStates, "/gazebo/link_states", self.on_link_states, 10
        )
        self.create_timer(0.1, self.on_timer)
        self.get_logger().info(
            f"Monitoring simulated mechanism physics for {self.duration:.1f} s "
            f"on {self.joint_state_topic}"
        )

    def elapsed(self):
        now = self.get_clock().now()
        if now.nanoseconds <= 0:
            return 0.0
        if self.started_at is None:
            self.started_at = now
            return 0.0
        return (now - self.started_at).nanoseconds / 1.0e9

    def fail(self, message):
        if message not in self.failures:
            self.failures.append(message)
            self.get_logger().error(message)

    def on_joint_state(self, msg):
        self.joint_samples += 1
        names = [normalize_joint_name(name) for name in msg.name]
        self.seen_joints.update(names)
        positions = dict(zip(names, msg.position))
        for name, position in zip(msg.name, msg.position):
            if not math.isfinite(position):
                self.fail(f"Non-finite joint position: {name}={position}")
        for name, velocity in zip(msg.name, msg.velocity):
            if not math.isfinite(velocity):
                self.fail(f"Non-finite joint velocity: {name}={velocity}")
            else:
                self.max_seen_abs_velocity = max(
                    self.max_seen_abs_velocity, abs(velocity)
                )
            if math.isfinite(velocity) and abs(velocity) > self.max_abs_velocity:
                self.fail(
                    f"Excessive joint velocity: {name}={velocity:.3f} rad/s"
                )

        for mimic, source in MIMIC_RELATIONSHIPS.items():
            if mimic not in positions or source not in positions:
                continue
            error = abs(positions[mimic] - positions[source])
            self.max_seen_mimic_error = max(self.max_seen_mimic_error, error)
            if error > self.max_mimic_error:
                self.fail(
                    f"Mimic tracking error: {mimic} vs {source} = {error:.4f} rad"
                )

    def on_link_states(self, msg):
        positions = {}
        for name, pose in zip(msg.name, msg.pose):
            suffix = name.rsplit("::", 1)[-1]
            positions[suffix] = pose.position
            values = (pose.position.x, pose.position.y, pose.position.z)
            if not all(math.isfinite(value) for value in values):
                self.fail(f"Non-finite link position: {name}")

        base = positions.get("iiwa_link_0")
        hand_positions = {
            name: positions[name] for name in HAND_LINK_SUFFIXES if name in positions
        }
        if base is None or not hand_positions:
            return

        self.link_samples += 1
        for name, position in hand_positions.items():
            distance = math.sqrt(
                (position.x - base.x) ** 2
                + (position.y - base.y) ** 2
                + (position.z - base.z) ** 2
            )
            self.max_seen_hand_distance = max(
                self.max_seen_hand_distance, distance
            )
            if distance > self.max_hand_distance:
                self.fail(
                    f"Hand link escaped the mechanism: {name}, distance={distance:.3f} m"
                )

    def on_timer(self):
        if self.finished or self.elapsed() < self.duration:
            return
        missing = EXPECTED_JOINTS - self.seen_joints
        if self.joint_samples == 0:
            self.fail(f"No {self.joint_state_topic} samples received")
        elif missing:
            self.fail("Missing joints: " + ", ".join(sorted(missing)))
        if self.link_samples == 0:
            self.fail("No usable /gazebo/link_states samples received")

        self.finished = True
        if self.failures:
            self.result = 1
            self.get_logger().error(
                f"Physics stability check FAILED with {len(self.failures)} issue(s)"
            )
        else:
            self.result = 0
            self.get_logger().info(
                "Physics stability check PASSED: "
                f"joint_samples={self.joint_samples}, "
                f"link_samples={self.link_samples}, "
                f"max_abs_velocity={self.max_seen_abs_velocity:.3f} rad/s, "
                f"max_hand_distance={self.max_seen_hand_distance:.3f} m, "
                f"max_mimic_error={self.max_seen_mimic_error:.6f} rad"
            )


def main():
    rclpy.init()
    node = PhysicsStabilityCheck()
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        node.fail("Stability check interrupted")
    result = node.result
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    return result


if __name__ == "__main__":
    sys.exit(main())
