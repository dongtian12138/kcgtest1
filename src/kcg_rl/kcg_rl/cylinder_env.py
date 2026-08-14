"""Gym-style, fixed-step ROS 2 environment for the cylinder curriculum.

The core environment intentionally depends only on ROS 2 and NumPy.  A thin
Gymnasium wrapper is exposed when Gymnasium is installed in the training
virtual environment.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
import threading
import time
from typing import Any, Dict, Optional, Sequence, Tuple

from builtin_interfaces.msg import Duration as DurationMsg
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import SetEntityState
import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String
from std_srvs.srv import Empty, Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


ARM_JOINTS = tuple(f"iiwa_joint_{index}" for index in range(1, 8))
ACTIVE_HAND_JOINTS = ("f1j1", "f1j2", "f2j1", "f3j2")
HAND_CONTROLLER_JOINTS = (
    "f1j1",
    "f1j2",
    "f2j1",
    "f3j2",
    "f3j1",
    "f1j3",
    "f2j2",
    "f3j3",
)

APPROACH_POSITIONS = np.asarray(
    [0.173683, 0.256410, -0.236703, -1.171082, 0.060182, 1.720685, -0.046556],
    dtype=np.float64,
)
LIFT_POSITIONS = np.asarray(
    [0.158382, 0.340953, -0.278255, -0.800570, 0.101666, 2.010246, -0.061229],
    dtype=np.float64,
)
OPEN_HAND_POSITIONS = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
VALIDATED_CLOSED_HAND_POSITIONS = np.asarray(
    [1.0, 0.75, 0.50, 0.75], dtype=np.float64
)
# Curriculum stage 1 learns a residual around the independently validated
# grasp.  Sampling the full mechanical joint range can command a severe
# asymmetric collision before the policy has learned anything and is not a
# meaningful exploration distribution for this first task.
HAND_ACTION_HALF_RANGES = np.asarray(
    [0.04, 0.05, 0.04, 0.05], dtype=np.float64
)
HAND_LOWER_BOUNDS = (
    VALIDATED_CLOSED_HAND_POSITIONS - HAND_ACTION_HALF_RANGES
)
HAND_UPPER_BOUNDS = (
    VALIDATED_CLOSED_HAND_POSITIONS + HAND_ACTION_HALF_RANGES
)

TASK_OBSERVATION_SIZE = 27
RL_OBSERVATION_SIZE = TASK_OBSERVATION_SIZE + 2 * len(ARM_JOINTS)
ACTION_SIZE = 5


def duration_message(seconds: float) -> DurationMsg:
    nanoseconds = int(round(max(0.001, seconds) * 1.0e9))
    return DurationMsg(
        sec=nanoseconds // 1_000_000_000,
        nanosec=nanoseconds % 1_000_000_000,
    )


def expand_hand_positions(active_positions: Sequence[float]) -> np.ndarray:
    values = np.asarray(active_positions, dtype=np.float64)
    if values.shape != (4,):
        raise ValueError("active hand target must contain four positions")
    return np.asarray(
        [
            values[0],
            values[1],
            values[2],
            values[3],
            values[0],
            values[1],
            values[2],
            values[3],
        ],
        dtype=np.float64,
    )


def interpolate_arm_positions(lift_progress: float) -> np.ndarray:
    progress = float(np.clip(lift_progress, 0.0, 1.0))
    return APPROACH_POSITIONS + progress * (LIFT_POSITIONS - APPROACH_POSITIONS)


def decode_macro_action(
    action: Sequence[float], lift_trigger_threshold: float
) -> Tuple[np.ndarray, bool, np.ndarray]:
    """Map a normalized policy action to a physical grasp and lift request.

    The hand portion is absolute rather than incremental.  Each selected pose
    is executed as one complete, validated trajectory instead of streaming
    short effort-controller splines, which are numerically unstable for this
    legacy hand model on ROS 2 Humble.
    """

    clipped_action = np.asarray(action, dtype=np.float64)
    if clipped_action.shape != (ACTION_SIZE,):
        raise ValueError(f"action must have shape ({ACTION_SIZE},)")
    if not np.all(np.isfinite(clipped_action)):
        raise ValueError("action contains a non-finite value")
    clipped_action = np.clip(clipped_action, -1.0, 1.0)

    hand_target = HAND_LOWER_BOUNDS + 0.5 * (
        clipped_action[:4] + 1.0
    ) * (
        HAND_UPPER_BOUNDS - HAND_LOWER_BOUNDS
    )
    lift_requested = bool(
        clipped_action[4] >= float(lift_trigger_threshold)
    )
    return hand_target, lift_requested, clipped_action


def calculate_reward(
    previous_metrics: Dict[str, Any],
    metrics: Dict[str, Any],
    action: np.ndarray,
    termination_reason: str,
) -> Tuple[float, Dict[str, float]]:
    previous_height = float(previous_metrics.get("height_gain", 0.0))
    height = float(metrics.get("height_gain", previous_height))
    previous_distance = float(previous_metrics.get("grasp_distance", 0.1))
    distance = float(metrics.get("grasp_distance", previous_distance))
    loaded_channels = int(metrics.get("loaded_torque_channels", 0))

    height_delta = float(np.clip(height - previous_height, -0.03, 0.03))
    distance_delta = float(
        np.clip(previous_distance - distance, -0.03, 0.03)
    )
    holding = height >= 0.07 and distance <= 0.075

    terms = {
        "height_progress": 40.0 * height_delta,
        "distance_progress": 2.0 * distance_delta,
        "loaded_channels": 0.003 * loaded_channels,
        "holding": 0.20 if holding else 0.0,
        "control": -0.002 * float(np.dot(action, action)),
        "living": -0.01,
        "terminal": 0.0,
    }
    if termination_reason == "success":
        terms["terminal"] = 25.0
    elif termination_reason in {"dropped", "lost_grasp"}:
        terms["terminal"] = -10.0
    elif termination_reason == "invalid_physics":
        terms["terminal"] = -25.0
    return float(sum(terms.values())), terms


@dataclass(frozen=True)
class CylinderEnvConfig:
    reset_motion_duration: float = 8.0
    reset_settle_duration: float = 1.0
    hand_motion_duration: float = 5.0
    preload_duration: float = 1.0
    lift_motion_duration: float = 8.0
    idle_step_duration: float = 1.0
    lift_trigger_threshold: float = 0.5
    target_change_tolerance: float = 1.0e-4
    service_timeout: float = 10.0
    data_timeout: float = 10.0
    max_episode_steps: int = 12
    park_position: Tuple[float, float, float] = (2.0, 0.0, 0.30)
    object_name: str = "grasp_cylinder"
    world_frame: str = "world"

    def validated(self) -> "CylinderEnvConfig":
        if self.reset_motion_duration <= 0.0:
            raise ValueError("reset_motion_duration must be positive")
        if self.reset_settle_duration <= 0.0:
            raise ValueError("reset_settle_duration must be positive")
        if self.hand_motion_duration <= 0.0:
            raise ValueError("hand_motion_duration must be positive")
        if self.preload_duration <= 0.0:
            raise ValueError("preload_duration must be positive")
        if self.lift_motion_duration <= 0.0:
            raise ValueError("lift_motion_duration must be positive")
        if self.idle_step_duration <= 0.0:
            raise ValueError("idle_step_duration must be positive")
        if not -1.0 <= self.lift_trigger_threshold <= 1.0:
            raise ValueError("lift_trigger_threshold must be in [-1, 1]")
        if self.target_change_tolerance < 0.0:
            raise ValueError("target_change_tolerance cannot be negative")
        if self.service_timeout <= 0.0 or self.data_timeout <= 0.0:
            raise ValueError("timeouts must be positive")
        if self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        if len(self.park_position) != 3:
            raise ValueError("park_position must contain x, y, z")
        return self


class _CylinderROSBridge(Node):
    def __init__(self, defaults: CylinderEnvConfig):
        super().__init__("cylinder_rl_env")
        defaults = defaults.validated()
        self.config = replace(
            defaults,
            reset_motion_duration=float(
                self.declare_parameter(
                    "reset_motion_duration", defaults.reset_motion_duration
                ).value
            ),
            reset_settle_duration=float(
                self.declare_parameter(
                    "reset_settle_duration", defaults.reset_settle_duration
                ).value
            ),
            hand_motion_duration=float(
                self.declare_parameter(
                    "hand_motion_duration", defaults.hand_motion_duration
                ).value
            ),
            preload_duration=float(
                self.declare_parameter(
                    "preload_duration", defaults.preload_duration
                ).value
            ),
            lift_motion_duration=float(
                self.declare_parameter(
                    "lift_motion_duration", defaults.lift_motion_duration
                ).value
            ),
            idle_step_duration=float(
                self.declare_parameter(
                    "idle_step_duration", defaults.idle_step_duration
                ).value
            ),
            lift_trigger_threshold=float(
                self.declare_parameter(
                    "lift_trigger_threshold", defaults.lift_trigger_threshold
                ).value
            ),
            target_change_tolerance=float(
                self.declare_parameter(
                    "target_change_tolerance",
                    defaults.target_change_tolerance,
                ).value
            ),
            service_timeout=float(
                self.declare_parameter(
                    "service_timeout", defaults.service_timeout
                ).value
            ),
            data_timeout=float(
                self.declare_parameter("data_timeout", defaults.data_timeout).value
            ),
            max_episode_steps=int(
                self.declare_parameter(
                    "max_episode_steps", defaults.max_episode_steps
                ).value
            ),
            park_position=tuple(
                float(value)
                for value in self.declare_parameter(
                    "park_position", list(defaults.park_position)
                ).value
            ),
            object_name=str(
                self.declare_parameter("object_name", defaults.object_name).value
            ),
            world_frame=str(
                self.declare_parameter("world_frame", defaults.world_frame).value
            ),
        ).validated()

        self._condition = threading.Condition()
        self._clock_ns: Optional[int] = None
        self._task_observation: Optional[np.ndarray] = None
        self._observation_sequence = 0
        self._metrics: Dict[str, Any] = {}
        self._arm_position: Dict[str, float] = {}
        self._arm_velocity: Dict[str, float] = {}

        self.arm_publisher = self.create_publisher(
            JointTrajectory,
            "/controller_gazebo_kuka/joint_trajectory",
            10,
        )
        self.hand_publisher = self.create_publisher(
            JointTrajectory,
            "/controller_gazebo_hand/joint_trajectory",
            10,
        )
        self.create_subscription(
            Clock, "/clock", self._clock_callback, qos_profile_sensor_data
        )
        self.create_subscription(
            Float64MultiArray,
            "/kcg_grasp/observation",
            self._observation_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            String, "/kcg_grasp/task_state", self._state_callback, 10
        )
        self.create_subscription(
            JointState,
            "/joint_states",
            self._joint_state_callback,
            qos_profile_sensor_data,
        )

        callback_group = ReentrantCallbackGroup()
        self.pause_client = self.create_client(
            Empty, "/pause_physics", callback_group=callback_group
        )
        self.unpause_client = self.create_client(
            Empty, "/unpause_physics", callback_group=callback_group
        )
        self.set_state_client = self.create_client(
            SetEntityState,
            "/gazebo/set_entity_state",
            callback_group=callback_group,
        )
        self.reset_client = self.create_client(
            Trigger, "/kcg_grasp/reset", callback_group=callback_group
        )

    def _clock_callback(self, message: Clock) -> None:
        clock_ns = message.clock.sec * 1_000_000_000 + message.clock.nanosec
        with self._condition:
            self._clock_ns = clock_ns
            self._condition.notify_all()

    def _observation_callback(self, message: Float64MultiArray) -> None:
        if len(message.data) != TASK_OBSERVATION_SIZE:
            return
        with self._condition:
            self._task_observation = np.asarray(
                message.data, dtype=np.float64
            )
            self._observation_sequence += 1
            self._condition.notify_all()

    def _state_callback(self, message: String) -> None:
        try:
            metrics = json.loads(message.data)
        except json.JSONDecodeError:
            return
        with self._condition:
            self._metrics = metrics
            self._condition.notify_all()

    def _joint_state_callback(self, message: JointState) -> None:
        with self._condition:
            for index, name in enumerate(message.name):
                if name not in ARM_JOINTS:
                    continue
                if index < len(message.position):
                    self._arm_position[name] = float(message.position[index])
                if index < len(message.velocity):
                    self._arm_velocity[name] = float(message.velocity[index])
            self._condition.notify_all()

    def _call(self, client, request, timeout: Optional[float] = None):
        timeout = self.config.service_timeout if timeout is None else timeout
        if not client.wait_for_service(timeout_sec=timeout):
            return None
        completed = threading.Event()
        future = client.call_async(request)
        future.add_done_callback(lambda unused: completed.set())
        if not completed.wait(timeout):
            return None
        try:
            return future.result()
        except Exception as error:  # pragma: no cover - ROS transport failure
            self.get_logger().error(f"ROS service call failed: {error}")
            return None

    def wait_until_ready(self) -> None:
        deadline = time.monotonic() + self.config.data_timeout
        clients = (
            self.pause_client,
            self.unpause_client,
            self.set_state_client,
            self.reset_client,
        )
        while time.monotonic() < deadline:
            services_ready = all(
                client.wait_for_service(timeout_sec=0.05) for client in clients
            )
            with self._condition:
                data_ready = (
                    self._clock_ns is not None
                    and self._task_observation is not None
                    and all(name in self._arm_position for name in ARM_JOINTS)
                    and all(name in self._arm_velocity for name in ARM_JOINTS)
                )
            controllers_ready = (
                self.arm_publisher.get_subscription_count() > 0
                and self.hand_publisher.get_subscription_count() > 0
            )
            if services_ready and data_ready and controllers_ready:
                return
            time.sleep(0.05)
        raise RuntimeError(
            "Timed out waiting for Gazebo, task observations, and controllers"
        )

    def pause(self) -> None:
        if self._call(self.pause_client, Empty.Request()) is None:
            raise RuntimeError("Gazebo pause service is unavailable")

    def unpause(self) -> None:
        if self._call(self.unpause_client, Empty.Request()) is None:
            raise RuntimeError("Gazebo unpause service is unavailable")

    def park_object(self) -> None:
        state = EntityState()
        state.name = self.config.object_name
        state.reference_frame = self.config.world_frame
        state.pose.position.x = self.config.park_position[0]
        state.pose.position.y = self.config.park_position[1]
        state.pose.position.z = self.config.park_position[2]
        state.pose.orientation.w = 1.0
        request = SetEntityState.Request()
        request.state = state
        response = self._call(self.set_state_client, request)
        if response is None or not response.success:
            raise RuntimeError("Failed to park the cylinder before episode reset")

    def reset_task(self) -> str:
        response = self._call(self.reset_client, Trigger.Request())
        if response is None or not response.success:
            detail = response.message if response is not None else "no response"
            raise RuntimeError(f"Cylinder task reset failed: {detail}")
        return response.message

    @staticmethod
    def _trajectory(
        names: Sequence[str], positions: Sequence[float], duration: float
    ) -> JointTrajectory:
        trajectory = JointTrajectory()
        trajectory.joint_names = list(names)
        point = JointTrajectoryPoint()
        point.positions = [float(value) for value in positions]
        point.time_from_start = duration_message(duration)
        trajectory.points = [point]
        return trajectory

    def publish_targets(
        self,
        hand_positions: Sequence[float],
        arm_positions: Sequence[float],
        duration: float,
    ) -> None:
        self.publish_hand_target(hand_positions, duration)
        self.publish_arm_target(arm_positions, duration)

    def publish_hand_target(
        self, hand_positions: Sequence[float], duration: float
    ) -> None:
        self.hand_publisher.publish(
            self._trajectory(
                HAND_CONTROLLER_JOINTS,
                expand_hand_positions(hand_positions),
                duration,
            )
        )

    def publish_arm_target(
        self, arm_positions: Sequence[float], duration: float
    ) -> None:
        self.arm_publisher.publish(
            self._trajectory(ARM_JOINTS, arm_positions, duration)
        )

    def advance(self, duration: float, *, pause_after: bool = True) -> None:
        with self._condition:
            if self._clock_ns is None:
                raise RuntimeError("No Gazebo clock is available")
            target_clock_ns = self._clock_ns + int(duration * 1.0e9)
            observation_sequence = self._observation_sequence

        self.unpause()
        deadline = time.monotonic() + max(
            self.config.data_timeout, 4.0 * duration
        )
        with self._condition:
            while (
                self._clock_ns is None or self._clock_ns < target_clock_ns
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                self._condition.wait(timeout=min(remaining, 0.1))
            reached_target = (
                self._clock_ns is not None and self._clock_ns >= target_clock_ns
            )
        if pause_after:
            self.pause()
        if not reached_target:
            raise RuntimeError("Gazebo clock did not advance by one environment step")

        # Make sure the returned state was sampled after this physics interval.
        observation_deadline = time.monotonic() + self.config.data_timeout
        with self._condition:
            while self._observation_sequence <= observation_sequence:
                remaining = observation_deadline - time.monotonic()
                if remaining <= 0.0:
                    raise RuntimeError("No post-step task observation was received")
                self._condition.wait(timeout=min(remaining, 0.1))

    def snapshot(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        with self._condition:
            if self._task_observation is None:
                raise RuntimeError("Task observation is unavailable")
            arm_positions = [self._arm_position[name] for name in ARM_JOINTS]
            arm_velocities = [self._arm_velocity[name] for name in ARM_JOINTS]
            observation = np.concatenate(
                (
                    self._task_observation.copy(),
                    np.asarray(arm_positions, dtype=np.float64),
                    np.asarray(arm_velocities, dtype=np.float64),
                )
            ).astype(np.float32)
            metrics = dict(self._metrics)
        if observation.shape != (RL_OBSERVATION_SIZE,):
            raise RuntimeError("Internal RL observation dimension mismatch")
        return observation, metrics


class KcgCylinderEnv:
    """Fixed-step environment with the Gymnasium reset/step return contract."""

    observation_size = RL_OBSERVATION_SIZE
    action_size = ACTION_SIZE

    def __init__(
        self,
        config: Optional[CylinderEnvConfig] = None,
        ros_args: Optional[Sequence[str]] = None,
    ):
        self._owns_rclpy = False
        if not rclpy.ok():
            rclpy.init(args=ros_args)
            self._owns_rclpy = True
        self._bridge = _CylinderROSBridge(config or CylinderEnvConfig())
        self.config = self._bridge.config
        self._executor = MultiThreadedExecutor(num_threads=4)
        self._executor.add_node(self._bridge)
        self._executor_thread = threading.Thread(
            target=self._executor.spin,
            name="kcg-cylinder-env-executor",
            daemon=True,
        )
        self._executor_thread.start()

        self._hand_target = OPEN_HAND_POSITIONS.copy()
        self._lift_progress = 0.0
        self._lift_started = False
        self._episode_steps = 0
        self._previous_metrics: Dict[str, Any] = {}
        self._closed = False

    @property
    def hand_target(self) -> np.ndarray:
        return self._hand_target.copy()

    @property
    def lift_progress(self) -> float:
        return self._lift_progress

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        del seed, options
        if self._closed:
            raise RuntimeError("Environment is closed")
        self._bridge.wait_until_ready()
        self._bridge.pause()
        self._bridge.park_object()

        self._hand_target = OPEN_HAND_POSITIONS.copy()
        self._lift_progress = 0.0
        self._lift_started = False
        self._bridge.publish_hand_target(
            self._hand_target,
            min(
                self.config.hand_motion_duration,
                self.config.reset_motion_duration,
            ),
        )
        self._bridge.publish_arm_target(
            APPROACH_POSITIONS,
            self.config.reset_motion_duration,
        )
        self._bridge.advance(
            self.config.reset_motion_duration, pause_after=False
        )
        reset_message = self._bridge.reset_task()
        self._bridge.advance(
            self.config.reset_settle_duration, pause_after=False
        )

        observation, metrics = self._bridge.snapshot()
        if not np.all(np.isfinite(observation)):
            raise RuntimeError("Episode reset produced a non-finite observation")
        self._episode_steps = 0
        self._previous_metrics = dict(metrics)
        info = {
            "metrics": metrics,
            "reset_message": reset_message,
            "hand_target": self._hand_target.tolist(),
            "lift_progress": self._lift_progress,
            "action_mode": "absolute_hand_pose_with_lift_trigger",
            "observation_layout": "kcg_cylinder_rl_observation_v1",
        }
        return observation, info

    def step(
        self, action: Sequence[float]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self._closed:
            raise RuntimeError("Environment is closed")
        hand_target, lift_requested, clipped_action = decode_macro_action(
            action, self.config.lift_trigger_threshold
        )
        hand_changed = bool(
            np.max(np.abs(hand_target - self._hand_target))
            > self.config.target_change_tolerance
        )
        sim_time_advanced = 0.0
        if hand_changed:
            self._hand_target = hand_target
            self._bridge.publish_hand_target(
                self._hand_target, self.config.hand_motion_duration
            )
            self._bridge.advance(
                self.config.hand_motion_duration, pause_after=False
            )
            sim_time_advanced += self.config.hand_motion_duration

        lift_triggered = bool(lift_requested and not self._lift_started)
        if lift_triggered:
            # Never start arm motion concurrently with a new grasp command.
            # The validated physical sequence finishes the hand trajectory,
            # allows contact preload to settle, and only then lifts.
            if hand_changed:
                self._bridge.advance(
                    self.config.preload_duration, pause_after=False
                )
                sim_time_advanced += self.config.preload_duration
            self._lift_started = True
            self._lift_progress = 1.0
            self._bridge.publish_arm_target(
                LIFT_POSITIONS, self.config.lift_motion_duration
            )
            self._bridge.advance(
                self.config.lift_motion_duration, pause_after=False
            )
            sim_time_advanced += self.config.lift_motion_duration

        # Gazebo Classic can become numerically unstable when stiff contact is
        # repeatedly paused and resumed.  Synchronize on /clock while leaving
        # physics continuous between macro decisions, as in the validated
        # scripted baseline.
        if sim_time_advanced == 0.0:
            sim_time_advanced = self.config.idle_step_duration
            self._bridge.advance(sim_time_advanced, pause_after=False)
        observation, metrics = self._bridge.snapshot()
        self._episode_steps += 1

        termination_reason = ""
        finite_observation = bool(np.all(np.isfinite(observation)))
        object_position = metrics.get("object_position", [0.0, 0.0, 0.0])
        object_height = (
            float(object_position[2]) if len(object_position) >= 3 else -math.inf
        )
        grasp_distance = float(metrics.get("grasp_distance", math.inf))
        if not finite_observation:
            termination_reason = "invalid_physics"
        elif bool(metrics.get("success", False)):
            termination_reason = "success"
        elif object_height < 0.18:
            termination_reason = "dropped"
        elif self._lift_progress > 0.15 and grasp_distance > 0.12:
            termination_reason = "lost_grasp"

        terminated = bool(termination_reason)
        truncated = (
            not terminated and self._episode_steps >= self.config.max_episode_steps
        )
        if truncated:
            termination_reason = "time_limit"

        reward, reward_terms = calculate_reward(
            self._previous_metrics,
            metrics,
            clipped_action,
            termination_reason,
        )
        self._previous_metrics = dict(metrics)
        if not finite_observation:
            observation = np.nan_to_num(
                observation, nan=0.0, posinf=1.0e6, neginf=-1.0e6
            )

        info = {
            "metrics": metrics,
            "reward_terms": reward_terms,
            "termination_reason": termination_reason,
            "episode_steps": self._episode_steps,
            "hand_target": self._hand_target.tolist(),
            "lift_progress": self._lift_progress,
            "hand_motion_commanded": hand_changed,
            "lift_triggered": lift_triggered,
            "sim_time_advanced": sim_time_advanced,
        }
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._bridge.pause()
        except RuntimeError:
            pass
        self._executor.shutdown(timeout_sec=2.0)
        self._executor_thread.join(timeout=2.0)
        self._bridge.destroy_node()
        if self._owns_rclpy and rclpy.ok():
            rclpy.shutdown()

    def __enter__(self) -> "KcgCylinderEnv":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        self.close()


try:  # Optional dependency, installed only in the training virtual environment.
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - exercised on the base ROS workstation
    gym = None
    spaces = None


if gym is not None:

    class GymnasiumKcgCylinderEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self, **kwargs):
            super().__init__()
            self.core = KcgCylinderEnv(**kwargs)
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(ACTION_SIZE,), dtype=np.float32
            )
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(RL_OBSERVATION_SIZE,),
                dtype=np.float32,
            )

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return self.core.reset(seed=seed, options=options)

        def step(self, action):
            return self.core.step(action)

        def close(self):
            self.core.close()

else:

    class GymnasiumKcgCylinderEnv:  # pragma: no cover - clear optional error
        def __init__(self, *args, **kwargs):
            del args, kwargs
            raise ImportError(
                "Gymnasium is not installed.  Use KcgCylinderEnv for the ROS "
                "smoke test, or install Gymnasium in the training virtual environment."
            )
