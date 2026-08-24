"""Prepare bounded V2 joint plans and implement joint-signal-only control laws."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from kcg_connector.grasp.carts_v2.models import V2Inputs
from kcg_connector.grasp.robust.bounded_hand_base_ik import (
    solve_bounded_hand_base_ik,
)


ARM_JOINT_NAMES = tuple(f"iiwa_joint_{index}" for index in range(1, 8))
ACTIVE_HAND_JOINT_NAMES = ("f1j1", "f1j2", "f2j1", "f3j2")


def minimum_jerk_blend(unit_time: float) -> float:
    value = float(np.clip(unit_time, 0.0, 1.0))
    return 10.0 * value**3 - 15.0 * value**4 + 6.0 * value**5


def piecewise_waypoint(waypoints: np.ndarray, fraction: float) -> np.ndarray:
    coordinate = float(np.clip(fraction, 0.0, 1.0)) * (len(waypoints) - 1)
    left = min(int(coordinate), len(waypoints) - 2)
    local = coordinate - left
    return (1.0 - local) * waypoints[left] + local * waypoints[left + 1]


def create_native_gravity_compensated_robot(
    articulation_path: str,
    expected_dof_names: Sequence[str],
    settings: Mapping[str, object],
):
    """Create the only robot view and configure its bounded native drives."""

    from isaacsim.core.experimental.prims import Articulation

    expected_law = "NATIVE_FORCE_DRIVE_GRAVITY_EQUIVALENT_POSITION_BIAS_V1"
    if settings.get("arm_control_law") != expected_law:
        raise RuntimeError("dynamic arm control law differs from the implemented law")
    robot = Articulation(articulation_path)
    dof_names = tuple(robot.dof_names)
    if len(dof_names) != 15 or set(dof_names) != set(expected_dof_names):
        raise RuntimeError(f"unexpected robot DOF identity: {dof_names}")
    arm_indices = np.asarray(
        [dof_names.index(name) for name in ARM_JOINT_NAMES], dtype=np.int32
    )
    hand_indices = np.asarray(
        [dof_names.index(name) for name in ACTIVE_HAND_JOINT_NAMES], dtype=np.int32
    )
    active_indices = np.concatenate((arm_indices, hand_indices))
    robot.set_dof_positions(np.zeros((1, robot.num_dofs)))
    robot.set_dof_velocities(np.zeros((1, robot.num_dofs)))
    robot.set_dof_velocity_targets(np.zeros((1, robot.num_dofs)))
    stiffnesses = np.asarray(
        [float(settings["arm_stiffness"])] * 7
        + [float(settings["hand_stiffness"])] * 4,
        dtype=np.float32,
    )
    dampings = np.asarray(
        [float(settings["arm_damping"])] * 7
        + [float(settings["hand_damping"])] * 4,
        dtype=np.float32,
    )
    caps = np.asarray(
        [float(settings["arm_drive_maximum_effort_nm"])] * 7
        + [float(settings["hand_drive_maximum_effort_nm"])] * 4,
        dtype=np.float32,
    )
    if (
        not all(np.isfinite(row).all() for row in (stiffnesses, dampings, caps))
        or np.any(stiffnesses <= 0.0)
        or np.any(dampings < 0.0)
        or np.any(caps <= 0.0)
    ):
        raise RuntimeError("native drive settings must be finite and physically bounded")
    # Mimic followers keep their kinematic constraint but no independent drive.
    zero_gains = np.zeros(robot.num_dofs, dtype=np.float32)
    robot.set_dof_gains(zero_gains, zero_gains, indices=0)
    robot.set_dof_gains(stiffnesses, dampings, indices=0, dof_indices=active_indices)
    robot.set_dof_max_efforts(caps, indices=0, dof_indices=active_indices)
    observed_kp, observed_kd = robot.get_dof_gains(indices=0, dof_indices=active_indices)
    observed_caps = robot.get_dof_max_efforts(indices=0, dof_indices=active_indices)
    drive_types = robot.get_dof_drive_types(indices=0, dof_indices=active_indices)[0]
    if drive_types != ["force"] * len(active_indices) or not all(
        np.allclose(observed.numpy()[0], expected)
        for observed, expected in (
            (observed_kp, stiffnesses),
            (observed_kd, dampings),
            (observed_caps, caps),
        )
    ):
        raise RuntimeError("native robot drive audit differs from V2 configuration")
    lower, upper = robot.get_dof_limits(indices=0, dof_indices=arm_indices)
    lower_values, upper_values = lower.numpy()[0], upper.numpy()[0]
    if not (
        np.isfinite(lower_values).all()
        and np.isfinite(upper_values).all()
        and np.all(lower_values < upper_values)
    ):
        raise RuntimeError("arm joint limits are nonfinite or reversed")
    audit = {
        "dof_names": list(dof_names),
        "active_drive_types": drive_types,
        "active_stiffnesses": observed_kp.numpy()[0].tolist(),
        "active_dampings": observed_kd.numpy()[0].tolist(),
        "active_maximum_efforts": observed_caps.numpy()[0].tolist(),
    }
    return robot, active_indices, arm_indices, lower_values, upper_values, audit


def gravity_biased_arm_target(
    robot,
    arm_indices: np.ndarray,
    nominal_target: Sequence[float],
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
    settings: Mapping[str, object],
) -> tuple[np.ndarray, dict[str, object]]:
    """Express model gravity compensation as a native-drive target offset."""

    position = robot.get_dof_positions(indices=0, dof_indices=arm_indices).numpy()
    velocity = robot.get_dof_velocities(indices=0, dof_indices=arm_indices).numpy()
    gravity = robot.get_dof_gravity_compensation_forces(
        indices=0, dof_indices=arm_indices
    ).numpy()
    target = np.asarray(nominal_target, dtype=np.float64).reshape(1, 7)
    if any(row.shape != (1, 7) for row in (position, velocity, gravity)):
        raise RuntimeError("arm gravity control returned an unexpected shape")
    if not all(np.isfinite(row).all() for row in (position, velocity, gravity)):
        raise RuntimeError("arm gravity control returned a nonfinite signal")
    kp = float(settings["arm_stiffness"])
    kd = float(settings["arm_damping"])
    cap = float(settings["arm_drive_maximum_effort_nm"])
    pd_effort = kp * (target - position) - kd * velocity
    raw_effort = pd_effort + gravity
    drive_target = target + gravity / kp
    limit_margin = np.minimum(
        drive_target[0] - lower_limits, upper_limits - drive_target[0]
    )
    if not np.isfinite(drive_target).all() or float(np.min(limit_margin)) <= 0.0:
        raise RuntimeError("gravity-biased arm target is nonfinite or outside limits")
    return drive_target[0], {
        "drive_target_rad": drive_target[0].tolist(),
        "target_bias_rad": (gravity / kp)[0].tolist(),
        "minimum_drive_target_limit_margin_rad": float(np.min(limit_margin)),
        "pd_effort_nm": pd_effort[0].tolist(),
        "gravity_compensation_nm": gravity[0].tolist(),
        "raw_total_effort_nm": raw_effort[0].tolist(),
        "clipped_total_effort_nm": np.clip(raw_effort, -cap, cap)[0].tolist(),
        "saturated": bool(np.any(np.abs(raw_effort) >= cap)),
    }


def _solve_approach_waypoints(inputs, model, settings, hand, target):
    clearance = float(
        inputs.config.section("dynamic")["approach_clearance_height_m"]
    )
    count = int(inputs.config.section("fast_filter")["approach_path_sample_count"])
    rows = []
    errors = []
    previous = None
    first_seed = None
    last_seed = None
    for index, fraction in enumerate(np.linspace(1.0, 0.0, count)):
        path_target = np.array(target, copy=True)
        path_target[2, 3] += clearance * float(fraction)
        keyword = {} if previous is None else {"seed_arm_positions": (previous,)}
        previous, position_error, orientation_error, seed_index = (
            solve_bounded_hand_base_ik(
                settings,
                model=model,
                hand_positions=hand,
                target_world_from_hand_base=path_target,
                label=("V2_APPROACH_ABOVE" if index == 0 else f"V2_APPROACH_DESCENT_{index}"),
                **keyword,
            )
        )
        first_seed = seed_index if first_seed is None else first_seed
        last_seed = seed_index
        rows.append(previous)
        errors.append((position_error, orientation_error))
    return rows, errors, first_seed, last_seed


def _solve_lift_waypoints(inputs, model, settings, hand, target, start):
    distance = float(inputs.config.section("dynamic")["lift_distance_m"])
    count = int(inputs.config.section("ik")["lift_waypoint_count"])
    rows = [start]
    errors = []
    previous = start
    for index, fraction in enumerate(np.linspace(0.0, 1.0, count)[1:], 1):
        lift_target = np.array(target, copy=True)
        lift_target[2, 3] += distance * float(fraction)
        previous, position_error, orientation_error, _ = solve_bounded_hand_base_ik(
            settings,
            model=model,
            hand_positions=hand,
            target_world_from_hand_base=lift_target,
            seed_arm_positions=(previous,),
            label=f"V2_LIFT_{index}",
        )
        rows.append(previous)
        errors.append((position_error, orientation_error))
    return rows, errors


def build_joint_motion_plan(
    repository_root: Path | str,
    inputs: V2Inputs,
    control_plan: Mapping[str, object],
    world_from_object: Sequence[Sequence[float]],
) -> dict[str, object]:
    """Solve pregrasp and +50 mm hand-base targets without simulator truth."""

    root = Path(repository_root).resolve()
    if root != inputs.repository_root:
        raise ValueError("motion-plan repository root differs from loaded V2 inputs")
    object_from_hand = np.asarray(
        control_plan["object_from_hand_row_major"], dtype=np.float64
    ).reshape(4, 4)
    world_from_object_matrix = np.asarray(world_from_object, dtype=np.float64)
    if world_from_object_matrix.shape != (4, 4):
        raise ValueError("world_from_object must be 4x4")
    target = world_from_object_matrix @ object_from_hand
    pregrasp_hand = tuple(
        float(value) for value in control_plan["pregrasp_joint_positions_rad"]
    )
    final_hand = tuple(
        float(value) for value in control_plan["final_joint_positions_rad"]
    )
    model = inputs.robot_model
    solver_settings = inputs.config.section("ik")["solver"]
    approach_rows, approach_errors, approach_seed, seed_index = (
        _solve_approach_waypoints(
            inputs, model, solver_settings, pregrasp_hand, target
        )
    )
    arm = approach_rows[-1]
    lift_rows, lift_only_errors = _solve_lift_waypoints(
        inputs, model, solver_settings, final_hand, target, arm
    )
    lift_errors = list(approach_errors) + lift_only_errors
    return {
        "arm_joint_names": ARM_JOINT_NAMES,
        "active_hand_joint_names": ACTIVE_HAND_JOINT_NAMES,
        "approach_arm_waypoints_rad": tuple(approach_rows),
        "pregrasp_arm_positions_rad": arm,
        "pregrasp_hand_positions_rad": pregrasp_hand,
        "final_hand_positions_rad": final_hand,
        "lift_arm_waypoints_rad": tuple(lift_rows),
        "world_from_hand_base_target": tuple(float(v) for v in target.ravel()),
        "approach_seed_index": approach_seed,
        "pregrasp_seed_index": seed_index,
        "maximum_ik_position_error_m": max(row[0] for row in lift_errors),
        "maximum_ik_orientation_error_rad": max(row[1] for row in lift_errors),
        "maximum_lift_joint_step_rad": max(
            float(np.max(np.abs(np.asarray(right) - np.asarray(left))))
            for left, right in zip(lift_rows, lift_rows[1:])
        ),
        "maximum_approach_joint_step_rad": max(
            float(np.max(np.abs(np.asarray(right) - np.asarray(left))))
            for left, right in zip(approach_rows, approach_rows[1:])
        ),
        "online_signals": (
            "joint_position",
            "joint_velocity",
            "joint_target_error",
            "tare_subtracted_measured_joint_effort",
            "robot_model_gravity_compensation_from_joint_state",
        ),
        "online_object_or_contact_truth_used": False,
    }


class SequentialEffortContactController:
    """Advance one finger until its joint-side stall evidence persists."""

    def __init__(
        self,
        start: Sequence[float],
        goal: Sequence[float],
        *,
        effort_rise_nm: float,
        position_error_rad: float,
        consecutive_samples: int,
        endpoint_timeout_samples: int,
    ) -> None:
        self.target = np.asarray(start, dtype=np.float64)
        self.goal = np.asarray(goal, dtype=np.float64)
        self.effort_rise_nm = float(effort_rise_nm)
        self.position_error_rad = float(position_error_rad)
        self.consecutive_samples = int(consecutive_samples)
        self.endpoint_timeout_samples = int(endpoint_timeout_samples)
        self.finger_order = (1, 2, 3)
        self.active_finger = self._evidence_count = self._endpoint_count = 0
        self._contact_targets: list[float] = []
        self.failure_reason: str | None = None

    @property
    def complete(self) -> bool:
        return self.active_finger == len(self.finger_order) and not self.failed

    @property
    def failed(self) -> bool:
        return self.failure_reason is not None

    @property
    def contact_targets_rad(self) -> tuple[float, ...]:
        return tuple(self._contact_targets)

    def step(
        self,
        measured_position: Sequence[float],
        measured_effort_delta: Sequence[float],
        maximum_increment_rad: float,
    ) -> np.ndarray:
        if self.complete or self.failed:
            return self.target.copy()
        for completed in self.finger_order[:self.active_finger]:
            self.target[completed] = float(measured_position[completed])
        index = self.finger_order[self.active_finger]
        remaining = self.goal[index] - self.target[index]
        increment = math.copysign(
            min(abs(remaining), float(maximum_increment_rad)), remaining
        )
        self.target[index] += increment
        error = abs(self.target[index] - float(measured_position[index]))
        loaded = abs(float(measured_effort_delta[index])) >= self.effort_rise_nm
        contact_evidence = loaded and error >= self.position_error_rad
        self._evidence_count = self._evidence_count + 1 if contact_evidence else 0
        at_endpoint = abs(self.goal[index] - self.target[index]) <= 1.0e-12
        self._endpoint_count = self._endpoint_count + 1 if at_endpoint else 0
        if self._evidence_count >= self.consecutive_samples:
            self.target[index] = float(measured_position[index])
            self._contact_targets.append(float(self.target[index]))
            self.active_finger += 1
            self._evidence_count = self._endpoint_count = 0
        elif self._endpoint_count >= self.endpoint_timeout_samples:
            self.failure_reason = f"FINGER_{self.active_finger + 1}_NO_CONTACT_SIGNAL"
        return self.target.copy()


class JointSignalStepper:
    def __init__(
        self,
        *,
        robot,
        world,
        auditor,
        active_indices: np.ndarray,
        arm_indices: np.ndarray,
        arm_lower_limits: np.ndarray,
        arm_upper_limits: np.ndarray,
        settings: Mapping[str, object],
        render: bool,
    ) -> None:
        self.robot, self.world, self.auditor = robot, world, auditor
        self.active_indices, self.arm_indices = active_indices, arm_indices
        self.arm_lower_limits, self.arm_upper_limits = arm_lower_limits, arm_upper_limits
        self.settings = settings
        self.render = bool(render)
        self.step_index = 0
        self.maximum_speed = self.maximum_arm_error = 0.0
        self.maximum_speed_joint: str | None = None
        self.maximum_hand_effort = self.maximum_gravity_effort = 0.0
        self.maximum_projected_arm_force = self.maximum_target_bias = 0.0
        self.minimum_drive_target_limit_margin = float("inf")
        self.abort_reason: str | None = None
        self.latest: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def advance(
        self, phase: str, arm_target: np.ndarray, hand_target: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        if self.abort_reason is not None:
            return self.latest
        active_target = np.concatenate((arm_target, hand_target))
        drive_arm_target, arm_control = gravity_biased_arm_target(
            self.robot,
            self.arm_indices,
            arm_target,
            self.arm_lower_limits,
            self.arm_upper_limits,
            self.settings,
        )
        if arm_control["saturated"]:
            self.abort_reason = "ARM_COMMAND_EFFORT_SATURATION_ABORT"
            return self.latest
        drive_target = np.concatenate((drive_arm_target, hand_target))
        self.robot.set_dof_position_targets(
            drive_target.reshape(1, -1),
            indices=0,
            dof_indices=self.active_indices,
        )
        self.world.step(render=self.render)
        positions = self.robot.get_dof_positions(
            indices=0, dof_indices=self.active_indices
        ).numpy()[0]
        velocities = self.robot.get_dof_velocities(
            indices=0, dof_indices=self.active_indices).numpy()[0]
        all_velocities = self.robot.get_dof_velocities(indices=0).numpy()[0]
        efforts = self.robot.get_dof_projected_joint_forces(
            indices=0, dof_indices=self.active_indices).numpy()[0]
        arm_control["projected_joint_force_nm"] = efforts[:7].tolist()
        self._update_metrics(positions, all_velocities, efforts, arm_target, arm_control)
        self.auditor.capture(
            step=self.step_index,
            phase=phase,
            active_positions=positions,
            active_velocities=velocities,
            active_efforts=efforts,
            active_targets=active_target,
            arm_control=arm_control,
        )
        self.step_index += 1
        self.latest = (positions, velocities, efforts)
        self._apply_signal_aborts(arm_target, all_velocities)
        return self.latest

    def _update_metrics(self, positions, all_velocities, efforts, arm_target, arm_control):
        peak_index = int(np.argmax(np.abs(all_velocities)))
        peak_speed = float(abs(all_velocities[peak_index]))
        if peak_speed > self.maximum_speed:
            self.maximum_speed = peak_speed
            self.maximum_speed_joint = self.robot.dof_names[peak_index]
        self.maximum_arm_error = max(
            self.maximum_arm_error,
            float(np.max(np.abs(positions[:7] - arm_target))),
        )
        self.maximum_hand_effort = max(
            self.maximum_hand_effort, float(np.max(np.abs(efforts[7:])))
        )
        self.maximum_gravity_effort = max(
            self.maximum_gravity_effort,
            float(np.max(np.abs(arm_control["gravity_compensation_nm"]))),
        )
        self.maximum_projected_arm_force = max(
            self.maximum_projected_arm_force, float(np.max(np.abs(efforts[:7])))
        )
        self.maximum_target_bias = max(
            self.maximum_target_bias,
            float(np.max(np.abs(arm_control["target_bias_rad"]))),
        )
        self.minimum_drive_target_limit_margin = min(
            self.minimum_drive_target_limit_margin,
            float(arm_control["minimum_drive_target_limit_margin_rad"]),
        )

    def _apply_signal_aborts(self, arm_target, all_velocities) -> None:
        assert self.latest is not None
        positions, _, efforts = self.latest
        if not all(np.all(np.isfinite(row)) for row in (positions, all_velocities, efforts)):
            self.abort_reason = "NONFINITE_JOINT_SIGNAL_ABORT"
        elif float(np.max(np.abs(all_velocities))) > float(
            self.settings["maximum_joint_speed_rad_s"]
        ):
            self.abort_reason = "JOINT_SPEED_ABORT"
        elif float(np.max(np.abs(positions[:7] - arm_target))) > float(
            self.settings["maximum_arm_tracking_error_rad"]
        ):
            self.abort_reason = "ARM_TRACKING_ERROR_ABORT"

    def active_steps(self, count: int):
        for step in range(count):
            if self.abort_reason is not None:
                break
            yield step


def run_pregrasp_sequence(
    stepper: JointSignalStepper,
    motion_plan: Mapping[str, object],
    settings: Mapping[str, object],
) -> dict[str, object]:
    dt = float(settings["physics_dt_s"])
    home_arm = np.zeros(7, dtype=np.float64)
    home_hand = np.zeros(4, dtype=np.float64)
    for _ in stepper.active_steps(round(float(settings["settle_duration_s"]) / dt)):
        stepper.advance("settle", home_arm, home_hand)
    pregrasp_arm = np.asarray(motion_plan["pregrasp_arm_positions_rad"])
    pregrasp_hand = np.asarray(motion_plan["pregrasp_hand_positions_rad"])
    approach = np.asarray(motion_plan["approach_arm_waypoints_rad"])
    above_steps = round(float(settings["approach_above_duration_s"]) / dt)
    for index in stepper.active_steps(above_steps):
        blend = minimum_jerk_blend((index + 1) / above_steps)
        stepper.advance("approach_above", blend * approach[0], blend * pregrasp_hand)

    settled_count = 0
    settled = False
    final_error = None
    settle_limit = round(float(settings["approach_above_settle_timeout_s"]) / dt)
    for _ in stepper.active_steps(settle_limit):
        latest = stepper.advance("wait_above_settled", approach[0], pregrasp_hand)
        if stepper.abort_reason is not None:
            break
        final_error = float(np.max(np.abs(latest[0][:7] - approach[0])))
        settled_count = (
            settled_count + 1
            if final_error <= float(settings["approach_above_settle_error_rad"])
            else 0
        )
        if settled_count >= int(settings["approach_above_settle_consecutive_samples"]):
            settled = True
            break
    if stepper.abort_reason is None and not settled:
        stepper.abort_reason = "PREGRASP_ABOVE_NOT_SETTLED"
    descent_steps = round(float(settings["approach_descent_duration_s"]) / dt)
    for index in stepper.active_steps(descent_steps):
        blend = minimum_jerk_blend((index + 1) / descent_steps)
        stepper.advance(
            "approach_descent", piecewise_waypoint(approach, blend), pregrasp_hand
        )
    hold_steps = round(float(settings["pregrasp_hold_duration_s"]) / dt)
    for _ in stepper.active_steps(hold_steps):
        stepper.advance("pregrasp_hold", pregrasp_arm, pregrasp_hand)
    return {
        "arm": pregrasp_arm,
        "hand": pregrasp_hand,
        "above_settled": settled,
        "above_final_error_rad": final_error,
    }


def _tare_and_close(stepper, motion_plan, settings, pregrasp):
    dt = float(settings["physics_dt_s"])
    tare_rows = []
    tare_steps = round(float(settings["effort_tare_duration_s"]) / dt)
    for _ in stepper.active_steps(tare_steps):
        latest = stepper.advance("tare", pregrasp["arm"], pregrasp["hand"])
        if stepper.abort_reason is None:
            tare_rows.append(latest[2][7:].copy())
    tare = np.mean(np.stack(tare_rows), axis=0) if tare_rows else np.zeros(4)
    final_hand = np.asarray(motion_plan["final_hand_positions_rad"])
    contact = SequentialEffortContactController(
        pregrasp["hand"],
        final_hand,
        effort_rise_nm=float(settings["contact_effort_rise_nm"]),
        position_error_rad=float(settings["contact_position_error_rad"]),
        consecutive_samples=int(settings["contact_consecutive_samples"]),
        endpoint_timeout_samples=round(float(settings["contact_endpoint_timeout_s"]) / dt),
    )
    maximum_increment = float(settings["finger_maximum_speed_rad_s"]) * dt
    closure_steps = sum(
        int(abs(final_hand[index] - pregrasp["hand"][index]) / maximum_increment)
        + contact.endpoint_timeout_samples
        + 2
        for index in (1, 2, 3)
    )
    for _ in stepper.active_steps(closure_steps):
        measured_hand = stepper.latest[0][7:]
        effort_delta = stepper.latest[2][7:] - tare
        hand_target = contact.step(measured_hand, effort_delta, maximum_increment)
        phase = f"finger_{min(contact.active_finger + 1, 3)}"
        stepper.advance(phase, pregrasp["arm"], hand_target)
        if stepper.abort_reason or contact.complete or contact.failed:
            break
    return contact, tare, final_hand


def _run_preload_lift_hold(stepper, motion_plan, settings, pregrasp, contact, tare, final_hand):
    dt = float(settings["physics_dt_s"])
    closure_target = contact.target.copy()
    direction = np.sign(final_hand - pregrasp["hand"])
    preload_goal = closure_target + float(settings["preload_increment_rad"]) * direction
    preload_steps = round(float(settings["preload_duration_s"]) / dt)
    for index in stepper.active_steps(preload_steps):
        blend = minimum_jerk_blend((index + 1) / preload_steps)
        hand_target = closure_target + blend * (preload_goal - closure_target)
        latest = stepper.advance("preload", pregrasp["arm"], hand_target)
        if stepper.abort_reason is not None:
            return stepper.abort_reason
        if np.max(np.abs(latest[2][8:] - tare[1:])) > float(
            settings["measured_effort_abort_nm"]
        ):
            return "HAND_MEASURED_EFFORT_ABORT"
    waypoints = np.asarray(motion_plan["lift_arm_waypoints_rad"])
    lift_steps = round(float(settings["lift_duration_s"]) / dt)
    for index in stepper.active_steps(lift_steps):
        blend = minimum_jerk_blend((index + 1) / lift_steps)
        stepper.advance("lift", piecewise_waypoint(waypoints, blend), preload_goal)
    final_arm = waypoints[-1]
    for _ in stepper.active_steps(round(float(settings["hold_duration_s"]) / dt)):
        stepper.advance("hold", final_arm, preload_goal)
    return stepper.abort_reason


def run_grasp_lift_sequence(stepper, motion_plan, settings, pregrasp):
    if stepper.abort_reason is not None:
        return {"contact_controller": None, "failure_reason": stepper.abort_reason}
    contact, tare, final_hand = _tare_and_close(
        stepper, motion_plan, settings, pregrasp
    )
    failure = contact.failure_reason or stepper.abort_reason
    if contact.complete and failure is None:
        failure = _run_preload_lift_hold(
            stepper, motion_plan, settings, pregrasp, contact, tare, final_hand
        )
    return {"contact_controller": contact, "failure_reason": failure}


def controller_outcome(
    stepper: JointSignalStepper,
    *,
    mode: str,
    native_drive_audit: Mapping[str, object],
    pregrasp: Mapping[str, object],
    grasp: Mapping[str, object],
) -> dict[str, object]:
    contact = grasp["contact_controller"]
    failure = grasp["failure_reason"] or stepper.abort_reason
    limits_ok = bool(
        stepper.maximum_speed <= float(stepper.settings["maximum_joint_speed_rad_s"])
        and stepper.maximum_arm_error
        <= float(stepper.settings["maximum_arm_tracking_error_rad"])
    )
    completed = bool(
        limits_ok
        and failure is None
        and (mode == "preflight" or contact.complete)
    )
    if not limits_ok and failure is None:
        failure = "JOINT_SPEED_OR_ARM_TRACKING_LIMIT"
    return {
        "completed": completed,
        "failure_reason": failure,
        "maximum_joint_speed_rad_s": stepper.maximum_speed,
        "maximum_joint_speed_joint": stepper.maximum_speed_joint,
        "maximum_arm_tracking_error_rad": stepper.maximum_arm_error,
        "maximum_absolute_hand_effort_nm": stepper.maximum_hand_effort,
        "maximum_gravity_compensation_nm": stepper.maximum_gravity_effort,
        "maximum_projected_arm_force_nm": stepper.maximum_projected_arm_force,
        "maximum_gravity_target_bias_rad": stepper.maximum_target_bias,
        "minimum_drive_target_limit_margin_rad": stepper.minimum_drive_target_limit_margin,
        "native_drive_audit": dict(native_drive_audit),
        "approach_above_settled": pregrasp["above_settled"],
        "approach_above_final_tracking_error_rad": pregrasp["above_final_error_rad"],
        "contact_targets_rad": [] if contact is None else list(contact.contact_targets_rad),
    }


__all__ = [
    "ACTIVE_HAND_JOINT_NAMES", "ARM_JOINT_NAMES", "JointSignalStepper",
    "SequentialEffortContactController", "build_joint_motion_plan",
    "controller_outcome", "create_native_gravity_compensated_robot",
    "gravity_biased_arm_target", "minimum_jerk_blend", "piecewise_waypoint",
    "run_grasp_lift_sequence", "run_pregrasp_sequence",
]
