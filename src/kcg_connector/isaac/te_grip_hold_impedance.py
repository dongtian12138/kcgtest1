"""Finite hand holding impedance with a torque-continuous mode transition."""
import numpy as np


def _host(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return value.numpy() if hasattr(value, "numpy") else np.asarray(value)


def activate_grip_hold(runtime, stepper, dynamic, hand_target, bounds, scale):
    world = runtime["world"]
    if world.is_playing() or not 1. < scale <= 4.:
        raise ValueError("holding impedance requires paused physics and a finite scale up to four")
    if "grip_hold_original_gains" in runtime:
        raise RuntimeError("the previous holding mode must be restored before a new grasp")
    robot, active, *_ = runtime["robot_data"]
    indices = active[7:]
    q = _host(robot.get_dof_positions(indices=0, dof_indices=indices))[0].astype(float)
    v = _host(robot.get_dof_velocities(indices=0, dof_indices=indices))[0].astype(float)
    targets = _host(robot.get_dof_position_targets(indices=0, dof_indices=indices))[0].astype(float)
    velocity_targets = _host(robot.get_dof_velocity_targets(indices=0, dof_indices=indices))[0].astype(float)
    kp, kd = (_host(x)[0].astype(float) for x in robot.get_dof_gains(indices=0, dof_indices=indices))
    caps = _host(robot.get_dof_max_efforts(indices=0, dof_indices=indices))[0].astype(float)
    if not np.allclose(targets, hand_target, atol=2e-7, rtol=0):
        raise ValueError("holding transition must start from the targets actually applied")
    if not np.all(kp > 0) or not np.all(kd >= 0) or not np.all(caps > 0):
        raise ValueError("positive finite native gains and force caps are required")
    new_kp, new_kd = kp*scale, kd*np.sqrt(scale)
    before_pd = kp*(targets-q)+kd*(velocity_targets-v)
    new_target = q+(before_pd-new_kd*(velocity_targets-v))/new_kp
    lower, upper = (_host(x)[0] for x in robot.get_dof_limits(indices=0, dof_indices=indices))
    preload_lower, preload_upper = np.asarray(bounds)
    if (not np.isfinite(new_target).all() or np.any(new_target < lower) or np.any(new_target > upper)
            or np.any(new_target[1:] < preload_lower[1:]) or np.any(new_target[1:] > preload_upper[1:])):
        raise ValueError("torque-continuous target is outside original physical/closing preload bounds")
    before_time = float(world.current_time)
    robot.set_dof_gains(new_kp, new_kd, indices=0, dof_indices=indices)
    robot.set_dof_position_targets(new_target[None, :], indices=0, dof_indices=indices)
    observed_kp, observed_kd = (_host(x)[0].astype(float) for x in robot.get_dof_gains(indices=0, dof_indices=indices))
    observed_target = _host(robot.get_dof_position_targets(indices=0, dof_indices=indices))[0].astype(float)
    observed_caps = _host(robot.get_dof_max_efforts(indices=0, dof_indices=indices))[0].astype(float)
    after_q = _host(robot.get_dof_positions(indices=0, dof_indices=indices))[0].astype(float)
    after_pd = observed_kp*(observed_target-q)+observed_kd*(velocity_targets-v)
    error = float(np.max(np.abs(after_pd-before_pd)))
    if (not np.array_equal(observed_caps, caps) or not np.array_equal(after_q, q)
            or float(world.current_time) != before_time or error > 1e-4
            or not np.allclose(observed_kp, new_kp) or not np.allclose(observed_kd, new_kd)):
        raise RuntimeError("native holding-gain transition failed torque/state/cap invariance")
    runtime["grip_hold_original_gains"] = {"kp": kp.copy(), "kd": kd.copy(),
        "dynamic_kp": dynamic["hand_stiffness"], "dynamic_kd": dynamic["hand_damping"]}
    if not np.allclose(observed_kp, observed_kp[0]) or not np.allclose(observed_kd, observed_kd[0]):
        raise ValueError("the current controller expects uniform gains across active hand axes")
    dynamic["hand_stiffness"] = float(observed_kp[0])
    dynamic["hand_damping"] = float(observed_kd[0])
    stepper.settings.update(hand_stiffness=dynamic["hand_stiffness"], hand_damping=dynamic["hand_damping"])
    record = {"scope": "FINITE_ACTIVE_HAND_HOLDING_IMPEDANCE_NOT_OBJECT_FIXATION",
        "step": int(stepper.step_index), "kp_before_nm_rad": kp.tolist(),
        "kp_after_nm_rad": observed_kp.tolist(), "kd_before_nm_s_rad": kd.tolist(),
        "kd_after_nm_s_rad": observed_kd.tolist(), "unchanged_motor_caps_nm": caps.tolist(),
        "targets_before_rad": targets.tolist(), "targets_after_rad": observed_target.tolist(),
        "raw_pd_effort_before_nm": before_pd.tolist(), "raw_pd_effort_after_nm": after_pd.tolist(),
        "maximum_pd_effort_change_nm": error, "physics_time_unchanged": True,
        "actual_joint_positions_unchanged_at_switch": True,
        "object_pose_or_force_command_used": False, "loaded_sensor_rezeroed": False,
        "later_contact_force_peaks_still_require_evaluation": True}
    runtime["active_grip_hold_impedance"] = record
    return observed_target, record


def restore_open_hand_gains(runtime, stepper, dynamic):
    original = runtime.pop("grip_hold_original_gains", None)
    if original is None:
        return None
    world = runtime["world"]
    if world.is_playing():
        raise RuntimeError("restore approach gains after the paused open-hand observation")
    robot, active, *_ = runtime["robot_data"]
    robot.set_dof_gains(original["kp"], original["kd"], indices=0, dof_indices=active[7:])
    kp, kd = (_host(x)[0] for x in robot.get_dof_gains(indices=0, dof_indices=active[7:]))
    if not np.allclose(kp, original["kp"]) or not np.allclose(kd, original["kd"]):
        raise RuntimeError("original approach gains did not read back")
    dynamic["hand_stiffness"], dynamic["hand_damping"] = original["dynamic_kp"], original["dynamic_kd"]
    stepper.settings.update(hand_stiffness=dynamic["hand_stiffness"], hand_damping=dynamic["hand_damping"])
    runtime.pop("active_grip_hold_impedance", None)
    return {"step": int(stepper.step_index), "original_approach_gains_restored": True,
        "open_position_targets_retained": True, "object_pose_written": False}
