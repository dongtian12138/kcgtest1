#!/usr/bin/env python3

import json
import math
import threading

import rclpy
from gazebo_msgs.msg import EntityState, ModelStates
from gazebo_msgs.srv import SetEntityState
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, String
from std_srvs.srv import Empty, Trigger
from tf2_ros import Buffer, TransformException, TransformListener


HAND_JOINTS = ("f1j1", "f1j2", "f2j1", "f3j2")
# The real hand has one Wheatstone-full-bridge strain channel per finger.  Each
# channel is a scalar measurement of the corresponding joint-axis torque.
DEFAULT_TORQUE_JOINTS = ("f1j2", "f2j1", "f3j2")


def quaternion_conjugate(quaternion):
    return (-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3])


def quaternion_multiply(first, second):
    ax, ay, az, aw = first
    bx, by, bz, bw = second
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def rotate_vector_inverse(quaternion, vector):
    vector_quaternion = (vector[0], vector[1], vector[2], 0.0)
    inverse = quaternion_conjugate(quaternion)
    rotated = quaternion_multiply(
        quaternion_multiply(inverse, vector_quaternion), quaternion
    )
    return rotated[:3]


class CylinderTaskManager(Node):
    def __init__(self):
        super().__init__("cylinder_task_manager")

        self.object_name = self.declare_parameter(
            "object_name", "grasp_cylinder"
        ).value
        self.world_frame = self.declare_parameter("world_frame", "world").value
        self.grasp_frame = self.declare_parameter(
            "grasp_frame", "grasp_tcp"
        ).value
        self.reset_position = tuple(
            float(value)
            for value in self.declare_parameter(
                "reset_position", [0.5, 0.0, 0.300]
            ).value
        )
        self.reset_orientation = tuple(
            float(value)
            for value in self.declare_parameter(
                "reset_orientation", [0.0, 0.0, 0.0, 1.0]
            ).value
        )
        self.align_reset_xy_with_grasp = bool(
            self.declare_parameter("align_reset_xy_with_grasp", True).value
        )
        self.teleport_on_reset = bool(
            self.declare_parameter("teleport_on_reset", True).value
        )
        observation_rate = float(
            self.declare_parameter("observation_rate", 20.0).value
        )
        self.finger_torque_topic = str(
            self.declare_parameter(
                "finger_torque_topic",
                "/finger_torque_broadcaster/joint_states",
            ).value
        )
        self.finger_torque_joint_names = tuple(
            self.declare_parameter(
                "finger_torque_joint_names", list(DEFAULT_TORQUE_JOINTS)
            ).value
        )
        if len(self.finger_torque_joint_names) != 3:
            raise ValueError("finger_torque_joint_names must contain 3 joints")
        self.torque_tare_duration = float(
            self.declare_parameter("torque_tare_duration", 0.5).value
        )
        self.torque_delta_threshold = float(
            self.declare_parameter("torque_delta_threshold", 0.03).value
        )
        self.minimum_loaded_torque_channels = int(
            self.declare_parameter("minimum_loaded_torque_channels", 2).value
        )
        self.require_torque_for_success = bool(
            self.declare_parameter("require_torque_for_success", False).value
        )
        self.lift_height = float(
            self.declare_parameter("lift_height", 0.08).value
        )
        self.hold_height_tolerance = float(
            self.declare_parameter("hold_height_tolerance", 0.01).value
        )
        self.hold_duration = float(
            self.declare_parameter("hold_duration", 3.0).value
        )
        self.maximum_grasp_distance = float(
            self.declare_parameter("maximum_grasp_distance", 0.075).value
        )

        self._lock = threading.Lock()
        self._object_pose = None
        self._object_twist = None
        self._joint_position = {}
        self._joint_velocity = {}
        self._finger_torque_raw = {}
        self._finger_torque_bias = {
            name: 0.0 for name in self.finger_torque_joint_names
        }
        self._torque_tare_end_ns = None
        self._torque_tare_sum = {
            name: 0.0 for name in self.finger_torque_joint_names
        }
        self._torque_tare_samples = 0
        self._hold_start_ns = None
        self._success = False
        self._latest_metrics = {}
        self._episode_start_position = tuple(self.reset_position)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.observation_publisher = self.create_publisher(
            Float64MultiArray, "/kcg_grasp/observation", 10
        )
        self.state_publisher = self.create_publisher(
            String, "/kcg_grasp/task_state", 10
        )
        self.create_subscription(
            ModelStates,
            "/gazebo/model_states",
            self._model_states_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            JointState,
            "/joint_states",
            self._joint_state_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            JointState,
            self.finger_torque_topic,
            self._finger_torque_callback,
            qos_profile_sensor_data,
        )

        service_group = ReentrantCallbackGroup()
        self.pause_client = self.create_client(
            Empty, "/pause_physics", callback_group=service_group
        )
        self.set_state_client = self.create_client(
            SetEntityState,
            "/gazebo/set_entity_state",
            callback_group=service_group,
        )
        self.unpause_client = self.create_client(
            Empty, "/unpause_physics", callback_group=service_group
        )
        self.create_service(
            Trigger,
            "/kcg_grasp/reset",
            self._reset_callback,
            callback_group=service_group,
        )
        self.create_service(
            Trigger,
            "/kcg_grasp/evaluate",
            self._evaluate_callback,
            callback_group=service_group,
        )
        self.create_timer(1.0 / observation_rate, self._publish_observation)

        self.get_logger().info(
            "Cylinder task manager ready: reset=/kcg_grasp/reset, "
            "observation=/kcg_grasp/observation, reset_mode="
            + ("teleport" if self.teleport_on_reset else "stationary")
        )

    def _model_states_callback(self, message):
        try:
            index = message.name.index(self.object_name)
        except ValueError:
            return
        with self._lock:
            self._object_pose = message.pose[index]
            self._object_twist = message.twist[index]

    def _joint_state_callback(self, message):
        with self._lock:
            for index, name in enumerate(message.name):
                if name not in HAND_JOINTS:
                    continue
                if index < len(message.position):
                    self._joint_position[name] = message.position[index]
                if index < len(message.velocity):
                    self._joint_velocity[name] = message.velocity[index]

    def _finger_torque_callback(self, message):
        values = {}
        for index, name in enumerate(message.name):
            if (
                name in self.finger_torque_joint_names
                and index < len(message.effort)
                and math.isfinite(message.effort[index])
            ):
                values[name] = message.effort[index]

        with self._lock:
            self._finger_torque_raw.update(values)
            if (
                self._torque_tare_end_ns is not None
                and len(values) == len(self.finger_torque_joint_names)
            ):
                now_ns = self.get_clock().now().nanoseconds
                if now_ns <= self._torque_tare_end_ns:
                    for name in self.finger_torque_joint_names:
                        self._torque_tare_sum[name] += values[name]
                    self._torque_tare_samples += 1
                else:
                    if self._torque_tare_samples > 0:
                        self._finger_torque_bias = {
                            name: self._torque_tare_sum[name]
                            / self._torque_tare_samples
                            for name in self.finger_torque_joint_names
                        }
                    self._torque_tare_end_ns = None

    def _wait_for_future(self, future, timeout):
        completed = threading.Event()
        future.add_done_callback(lambda unused: completed.set())
        if not completed.wait(timeout):
            return None
        try:
            return future.result()
        except Exception as error:  # pragma: no cover - ROS transport failure
            self.get_logger().error(f"Service call failed: {error}")
            return None

    def _call(self, client, request, timeout=5.0):
        if not client.wait_for_service(timeout_sec=timeout):
            return None
        return self._wait_for_future(client.call_async(request), timeout)

    def _reset_callback(self, request, response):
        del request
        with self._lock:
            missing_torques = [
                name
                for name in self.finger_torque_joint_names
                if name not in self._finger_torque_raw
            ]
        if missing_torques:
            response.success = False
            response.message = (
                "Finger torque state unavailable for: "
                + ", ".join(missing_torques)
            )
            return response

        target_position = list(self.reset_position)
        reset_error = None
        if self.teleport_on_reset:
            if self.align_reset_xy_with_grasp:
                try:
                    transform = self.tf_buffer.lookup_transform(
                        self.world_frame, self.grasp_frame, Time()
                    )
                except TransformException as error:
                    response.success = False
                    response.message = (
                        f"Grasp-frame transform unavailable: {error}"
                    )
                    return response
                target_position[0] = transform.transform.translation.x
                target_position[1] = transform.transform.translation.y

            paused = self._call(self.pause_client, Empty.Request()) is not None
            if not paused:
                response.success = False
                response.message = "Gazebo pause service unavailable"
                return response

            state = EntityState()
            state.name = self.object_name
            state.reference_frame = self.world_frame
            state.pose.position.x = target_position[0]
            state.pose.position.y = target_position[1]
            state.pose.position.z = target_position[2]
            state.pose.orientation.x = self.reset_orientation[0]
            state.pose.orientation.y = self.reset_orientation[1]
            state.pose.orientation.z = self.reset_orientation[2]
            state.pose.orientation.w = self.reset_orientation[3]

            set_request = SetEntityState.Request()
            set_request.state = state
            set_result = self._call(self.set_state_client, set_request)
            unpaused = self._call(self.unpause_client, Empty.Request()) is not None
            reset_succeeded = bool(
                set_result is not None and set_result.success and unpaused
            )
            if not reset_succeeded:
                reset_error = (
                    set_result.status_message
                    if set_result is not None
                    else "no reply"
                )
        else:
            with self._lock:
                if self._object_pose is None:
                    response.success = False
                    response.message = "Cylinder state unavailable"
                    return response
                target_position = [
                    self._object_pose.position.x,
                    self._object_pose.position.y,
                    self._object_pose.position.z,
                ]
            reset_succeeded = True

        with self._lock:
            self._finger_torque_bias = {
                name: self._finger_torque_raw[name]
                for name in self.finger_torque_joint_names
            }
            self._torque_tare_end_ns = (
                self.get_clock().now().nanoseconds
                + int(max(0.0, self.torque_tare_duration) * 1e9)
            )
            self._torque_tare_sum = {
                name: 0.0 for name in self.finger_torque_joint_names
            }
            self._torque_tare_samples = 0
            self._hold_start_ns = None
            self._success = False
            self._latest_metrics = {}
            self._episode_start_position = tuple(target_position)

        response.success = reset_succeeded
        if response.success:
            action = (
                "Cylinder pose and twist reset at"
                if self.teleport_on_reset
                else "Episode initialized without moving cylinder at"
            )
            response.message = (
                f"{action} [{target_position[0]:.6f}, "
                f"{target_position[1]:.6f}, {target_position[2]:.6f}]"
            )
        else:
            response.message = f"Failed to reset cylinder: {reset_error}"
        return response

    def _evaluate_callback(self, request, response):
        del request
        with self._lock:
            metrics = dict(self._latest_metrics)
            response.success = self._success
        metrics["success"] = response.success
        response.message = json.dumps(metrics, sort_keys=True)
        return response

    def _publish_observation(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.world_frame, self.grasp_frame, Time()
            )
        except TransformException:
            return

        with self._lock:
            if self._object_pose is None or self._object_twist is None:
                return
            object_pose = self._object_pose
            object_twist = self._object_twist
            joint_position = dict(self._joint_position)
            joint_velocity = dict(self._joint_velocity)
            finger_torque_raw = dict(self._finger_torque_raw)
            finger_torque_bias = dict(self._finger_torque_bias)
            torque_tare_active = self._torque_tare_end_ns is not None
            episode_start_position = self._episode_start_position

        grasp_position = transform.transform.translation
        grasp_rotation = transform.transform.rotation
        grasp_quaternion = (
            grasp_rotation.x,
            grasp_rotation.y,
            grasp_rotation.z,
            grasp_rotation.w,
        )
        world_delta = (
            object_pose.position.x - grasp_position.x,
            object_pose.position.y - grasp_position.y,
            object_pose.position.z - grasp_position.z,
        )
        relative_position = rotate_vector_inverse(grasp_quaternion, world_delta)
        object_quaternion = (
            object_pose.orientation.x,
            object_pose.orientation.y,
            object_pose.orientation.z,
            object_pose.orientation.w,
        )
        relative_orientation = quaternion_multiply(
            quaternion_conjugate(grasp_quaternion), object_quaternion
        )

        now_ns = self.get_clock().now().nanoseconds
        finger_torque_deltas = [
            finger_torque_raw.get(name, 0.0)
            - finger_torque_bias.get(name, 0.0)
            for name in self.finger_torque_joint_names
        ]
        loaded_torque_channel_count = sum(
            abs(value) >= self.torque_delta_threshold
            for value in finger_torque_deltas
        )
        height_gain = object_pose.position.z - episode_start_position[2]
        grasp_distance = math.sqrt(sum(value * value for value in world_delta))
        height_valid = height_gain >= (
            self.lift_height - self.hold_height_tolerance
        )
        torque_valid = (
            loaded_torque_channel_count
            >= self.minimum_loaded_torque_channels
        )
        hold_valid = (
            height_valid
            and grasp_distance <= self.maximum_grasp_distance
            and (not self.require_torque_for_success or torque_valid)
        )

        with self._lock:
            if hold_valid:
                if self._hold_start_ns is None:
                    self._hold_start_ns = now_ns
                held_for = max(0.0, (now_ns - self._hold_start_ns) / 1e9)
                self._success = held_for >= self.hold_duration
            else:
                self._hold_start_ns = None
                held_for = 0.0
                self._success = False

            metrics = {
                "loaded_torque_channels": loaded_torque_channel_count,
                "finger_torque_deltas": [
                    round(value, 6) for value in finger_torque_deltas
                ],
                "finger_torques_raw": [
                    round(finger_torque_raw.get(name, 0.0), 6)
                    for name in self.finger_torque_joint_names
                ],
                "torque_tare_active": torque_tare_active,
                "grasp_distance": round(grasp_distance, 6),
                "height_gain": round(height_gain, 6),
                "reset_mode": (
                    "teleport" if self.teleport_on_reset else "stationary"
                ),
                "episode_start_position": [
                    round(value, 6) for value in episode_start_position
                ],
                "hold_seconds": round(held_for, 3),
                "object_position": [
                    round(object_pose.position.x, 6),
                    round(object_pose.position.y, 6),
                    round(object_pose.position.z, 6),
                ],
                "grasp_position": [
                    round(grasp_position.x, 6),
                    round(grasp_position.y, 6),
                    round(grasp_position.z, 6),
                ],
                "relative_position": [round(value, 6) for value in relative_position],
                "hand_positions": {
                    name: round(joint_position.get(name, 0.0), 6)
                    for name in HAND_JOINTS
                },
                "success": self._success,
            }
            self._latest_metrics = metrics

        observation = Float64MultiArray()
        observation.layout.dim = [
            MultiArrayDimension(
                label="kcg_cylinder_observation_v1", size=27, stride=27
            )
        ]
        observation.data = [
            *relative_position,
            *relative_orientation,
            object_twist.linear.x,
            object_twist.linear.y,
            object_twist.linear.z,
            object_twist.angular.x,
            object_twist.angular.y,
            object_twist.angular.z,
            *(joint_position.get(name, 0.0) for name in HAND_JOINTS),
            *(joint_velocity.get(name, 0.0) for name in HAND_JOINTS),
            *finger_torque_deltas,
            height_gain,
            grasp_distance,
            held_for,
        ]
        self.observation_publisher.publish(observation)
        self.state_publisher.publish(String(data=json.dumps(metrics, sort_keys=True)))


def main():
    rclpy.init()
    node = CylinderTaskManager()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            executor.shutdown()
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        finally:
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    main()
