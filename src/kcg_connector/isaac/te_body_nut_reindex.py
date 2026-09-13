"""Release a seated nut and reindex the open hand using current RGB-D.

No object/contact truth, object pose command or world attachment is used.
The separate post-run audit must establish actual retention and contact loss.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np


def can_unload_after_lateral_hold_stop(rotation_record, latest_sensor_phase,
                                      step_index, outer_abort_reason):
    """Recognize a stopped post-turn hold, never waive a motion/safety abort."""
    return bool(
        not rotation_record.get("completed")
        and rotation_record.get("failure_reason") == "bounded thread pilot stop: LATERAL_CONTACT_FORCE"
        and latest_sensor_phase == "key_probe_nut_rotation_hold"
        and rotation_record.get("last_step") == step_index
        and rotation_record.get("outer_abort_reason") is None
        and outer_abort_reason is None)


def can_unload_after_torsional_pilot_stop(rotation_record, latest_sensor_phase,
                                         step_index, outer_abort_reason):
    """Allow a visual guide check and release after this pilot's torque stop.

    This does not waive a joint/wrist/collision abort or assert final seating.
    The original failed rotation record is preserved for physical evaluation.
    """
    return bool(
        not rotation_record.get("completed")
        and rotation_record.get("failure_reason") == "bounded thread pilot stop: TORSIONAL_CONTACT_MOMENT"
        and latest_sensor_phase in ("key_probe_nut_rotation_turn", "key_probe_nut_rotation_hold")
        and rotation_record.get("last_step") == step_index
        and rotation_record.get("outer_abort_reason") is None
        and outer_abort_reason is None)


def can_unload_after_transmission_reserve_stop(rotation_record, latest_sensor_phase,
                                              step_index, outer_abort_reason):
    """A pre-boundary controller stop permits observation and monotonic unload."""
    event=rotation_record.get("transmission_reserve_stop",{})
    return bool(not rotation_record.get("completed")
        and rotation_record.get("failure_reason")=="bounded thread pilot stop: TRANSMISSION_RESERVE"
        and event.get("hard_boundary_violated") is False
        and event.get("effort_margins_nm") and min(event["effort_margins_nm"])>=0.
        and latest_sensor_phase in ("key_probe_nut_rotation_turn","key_probe_nut_rotation_hold")
        and rotation_record.get("last_step")==step_index
        and rotation_record.get("outer_abort_reason") is None and outer_abort_reason is None)


def run_nut_release_and_reindex(repository, runtime, stepper, dynamic, grip,
                               rotation_record, world_from_socket, settings, output):
    import omni.replicator.core as rep
    import omni.usd
    from scipy.spatial.transform import Rotation

    from kcg_connector.grasp.robust.bounded_hand_base_ik import solve_bounded_hand_base_ik
    from te_body_socket_observation import observe_released_plug_from_rgbd
    from te_foundationpose_handoff_runtime import _json_ready, MOVEIT_SOFT_ARM_BOUNDS_RAD, control

    repository, output = Path(repository).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    world, ft = runtime["world"], runtime["nail_body_ft_auditor"]
    inputs, stage = runtime["inputs"], omni.usd.get_context().get_stage()
    socket = np.asarray(world_from_socket).reshape(4, 4)
    dt = float(dynamic["physics_dt_s"])
    check = runtime["nut_regrasp_geometry_check"]
    locate_bounds = runtime["nut_regrasp_locate_visual_bounds"]
    first_ft = len(ft.samples)
    commands = []
    record = {"completed": False, "stage": "BEFORE_NUT_RELEASE", "first_step": int(stepper.step_index),
              "simulation_only": True, "hardware_authorized": False,
              "online_object_or_contact_truth_used": False,
              "object_pose_or_force_command_used": False,
              "physical_support_verified": False,
              "scope": "CONTROLLED_RELEASE_AND_OPEN_HAND_REINDEX_REQUIRES_POSTRUN_EVALUATION",
              "settings": settings}

    def save():
        (output / "nut_reindex_controller_result.json").write_text(
            json.dumps(_json_ready(record), ensure_ascii=False, indent=2)+"\n")

    def measured():
        q = np.asarray(stepper.latest[0])
        hand = np.asarray(inputs.robot_model.forward_kinematics(tuple(q), enforce_limits=False)["handbase_link"])
        return q, hand

    def observe(label):
        world.pause()
        _, hand = measured()
        frame = output / label
        fresh = observe_released_plug_from_rgbd(repository, stage, world, rep, hand, frame)
        if not fresh.get("position_and_axis_measured"):
            raise RuntimeError("current palm observation did not measure the released Body")
        body = np.asarray(fresh["world_from_plug_five_dof"])
        record[label] = {
            "five_dof_observation": fresh, "body_key_yaw_measured_independently": False,
            "body_key_yaw_required": False, "rear_array_observation_executed": False,
            "control_pose_components": "POSITION_AND_DIRECTED_AXIS_ONLY",
            "avoidance_envelope_covers_all_body_yaw": True,
            "wired_connector_visibility_validated": False}
        cosine = abs(float(socket[:3, 2] @ body[:3, 2]))
        rear_overlap = (-float(socket[:3, 2] @ (body[:3, 3]-socket[:3, 3]))
                        - .007645401*cosine
                        - .018821401*np.sqrt(max(0., 1.-cosine**2)))
        record[label]["minimum_source_key_rear_overlap_estimate_m"] = rear_overlap
        guide_overlap = (-float(socket[:3, 2] @ (body[:3, 3]-socket[:3, 3]))
                         - .000762*cosine
                         - .018821401*np.sqrt(max(0., 1.-cosine**2)))
        record[label]["minimum_source_key_guide_overlap_estimate_m"] = guide_overlap
        if guide_overlap < float(settings.get("minimum_observed_key_guide_overlap_m", .0001)):
            raise RuntimeError("current vision no longer retains guide engagement; reinsertion is required")
        locate_bounds(body)
        save()
        return fresh, body

    def advance(phase, arm, hand, *, nut_contact=False, unloading=False):
        if stepper.abort_reason is not None:
            raise RuntimeError(f"existing protection: {stepper.abort_reason}")
        q, _ = measured()
        hit = check(q[:7], q[7:], nut_contact=nut_contact)
        if hit is None and not unloading:
            hit = check(arm, hand, nut_contact=nut_contact)
        if hit is not None:
            record["geometry_stop"] = hit
            raise RuntimeError(f"nut reindex geometry: {hit}")
        before = int(stepper.step_index)
        stepper.advance(phase, np.asarray(arm).copy(), np.asarray(hand).copy())
        commands.append({"step": before, "phase": phase, "arm_target_rad": np.asarray(arm).tolist(),
                         "hand_target_rad": np.asarray(hand).tolist()})
        q, _ = measured()
        hit = check(q[:7], q[7:], nut_contact=nut_contact)
        if hit is not None or stepper.abort_reason is not None:
            record["geometry_stop"] = hit
            raise RuntimeError(f"nut reindex stopped: {hit or stepper.abort_reason}")

    try:
        stopped_hold_unload = bool(settings.get("allow_unload_after_lateral_hold_stop", False)
            and can_unload_after_lateral_hold_stop(rotation_record, ft.samples[-1]["phase"],
                                                  int(stepper.step_index), stepper.abort_reason))
        stopped_torsion_unload = bool(settings.get("release_only", False)
            and settings.get("allow_release_after_torsional_pilot_stop", False)
            and can_unload_after_torsional_pilot_stop(rotation_record, ft.samples[-1]["phase"],
                int(stepper.step_index), stepper.abort_reason))
        stopped_reserve_unload=bool(settings.get("release_only",False)
            and settings.get("allow_release_after_transmission_reserve_stop",False)
            and can_unload_after_transmission_reserve_stop(rotation_record,ft.samples[-1]["phase"],
                int(stepper.step_index),stepper.abort_reason))
        stopped_unload = stopped_hold_unload or stopped_torsion_unload or stopped_reserve_unload
        if ((not rotation_record.get("completed") and not stopped_unload)
                or stepper.abort_reason is not None):
            raise RuntimeError("the current bounded rotation did not finish")
        record["preceding_rotation_controller_completed"] = bool(rotation_record.get("completed"))
        record["unload_after_additional_lateral_hold_stop"] = stopped_hold_unload
        record["release_after_additional_torsional_pilot_stop"] = stopped_torsion_unload
        record["release_after_transmission_reserve_stop"] = stopped_reserve_unload
        if stopped_unload:
            record["preceding_stop_preserved"] = {
                "reason": rotation_record["failure_reason"],
                "last_step": rotation_record["last_step"],
                "action": "CURRENT_VISUAL_GUIDE_CHECK_THEN_MONOTONIC_FINGER_UNLOAD",
                "independent_joint_wrist_and_collision_protection_waived": False,
                "rotation_record_relabelled_completed": False}
        _, _ = observe("before_nut_release")
        if stopped_unload:
            # The failed controller may have computed an unapplied next target.
            # Start unloading from the last target actually sent to the robot.
            last_applied = np.asarray(ft.samples[-1]["active_targets_rad"])
            arm, hand = last_applied[:7].copy(), last_applied[7:].copy()
            record["release_target_source"] = "LAST_APPLIED_ROBOT_TARGETS_AT_THE_STOP"
        else:
            arm = np.asarray(rotation_record["final_arm_target_rad"]).copy()
            hand = np.asarray(rotation_record["final_hand_target_rad"]).copy()
        open_hand = np.asarray(settings["open_hand_positions_rad"])
        lower, upper = np.asarray(grip["finite_preload_bounds_rad"])
        tare = np.asarray(grip["new_grasp_effort_tare_nm"])
        effort = np.asarray(grip["effort_reference_nm"])
        if not np.all(hand[1:] > open_hand[1:]):
            raise ValueError("this nut release requires the established positive closing directions")
        increment = float(dynamic["finger_maximum_speed_rad_s"])*dt
        stiffness = float(dynamic["hand_stiffness"])
        count = max(1, round(float(dynamic["hold_duration_s"])/dt))
        record.update(initial_arm_target_rad=arm.tolist(), initial_hand_target_rad=hand.tolist(),
                      open_hand_target_rad=open_hand.tolist(), unload_duration_s=count*dt,
                      stage="UNLOADING_NUT_PAD_EFFORT")
        save()
        world.play()
        for index in range(count):
            reference = effort*(1.-(index+1)/count)
            loaded = np.asarray(stepper.latest[2])[8:]-tare[1:]
            hand[1:] = np.clip(hand[1:]+np.clip((reference-loaded)/stiffness, -increment, 0.),
                               lower[1:], upper[1:])
            advance("key_probe_nut_index_unload", arm, hand, nut_contact=True, unloading=True)
        record["unload_last_step"] = int(stepper.step_index)
        record["stage"] = "OPENING_NUT_PADS"
        start = hand.copy()
        count = max(1, int(np.ceil(np.max(np.abs(open_hand-start))/increment)))
        for index in range(count):
            hand = start+(index+1)/count*(open_hand-start)
            advance("key_probe_nut_index_open", arm, hand, nut_contact=True)
        record["opening_last_step"] = int(stepper.step_index)
        stepper.payload_compensation_fraction = 0.0
        # These phases deliberately use the existing free-space FT mode.
        for _ in range(round(float(settings["open_hold_duration_s"])/dt)):
            advance("nut_index_free_open_hold", arm, open_hand)
        record["support_hold_last_step"] = int(stepper.step_index)
        fresh, body = observe("after_nut_release")
        if "grip_hold_original_gains" in runtime:
            from te_grip_hold_impedance import restore_open_hand_gains
            record["holding_impedance_restoration"] = restore_open_hand_gains(runtime, stepper, dynamic)
        if settings.get("release_only", False):
            record.update(completed=True, stage="NUT_RELEASE_FINISHED_REQUIRES_RETENTION_EVALUATION",
                scope="CONTROLLED_FINAL_RELEASE_NOT_AN_ASSEMBLY_SUCCESS_CLAIM",
                final_observation=fresh, final_arm_target_rad=arm.tolist(),
                final_hand_target_rad=open_hand.tolist(), open_hand_reindex_executed=False)
        else:
            record["stage"] = "PLANNING_OPEN_HAND_REINDEX_FROM_CURRENT_VISION"
            q, initial_hand = measured()
            angle = np.deg2rad(float(settings["rotation_about_socket_plus_z_deg"]))
            bounds = np.asarray([MOVEIT_SOFT_ARM_BOUNDS_RAD[name] for name in control.ARM_JOINT_NAMES])
            waypoints = [q[:7].copy()]
            count = max(1, int(np.ceil(abs(np.rad2deg(angle)))))
            for fraction in np.linspace(0., 1., count+1)[1:]:
                R = Rotation.from_rotvec(socket[:3, 2]*angle*fraction).as_matrix()
                target = initial_hand.copy()
                target[:3, :3] = R @ initial_hand[:3, :3]
                target[:3, 3] = body[:3, 3]+R @ (initial_hand[:3, 3]-body[:3, 3])
                solved, pe, ae, _ = solve_bounded_hand_base_ik(inputs.config.section("ik")["solver"],
                    model=inputs.robot_model, hand_positions=open_hand, target_world_from_hand_base=target,
                    seed_arm_positions=(waypoints[-1],), label="CURRENT_VISION_OPEN_HAND_NUT_REINDEX")
                if pe > .0001 or ae > .001:
                    raise RuntimeError("open-hand reindex IK did not reach its target")
                waypoints.append(np.asarray(solved))
            waypoints = np.asarray(waypoints)
            offset = np.asarray(ft.samples[-1]["active_targets_rad"][:7])-q[:7]
            waypoints += offset
            maximum_speed = float(settings["maximum_arm_speed_rad_s"])
            duration = max(1., 1.875*len(waypoints[:-1])*float(np.max(np.abs(np.diff(waypoints, axis=0))))/maximum_speed)
            count = int(np.ceil(duration/dt))
            states = np.asarray([control.piecewise_waypoint(waypoints, control.minimum_jerk_blend(i/count))
                                 for i in range(count+1)])
            peak = float(np.max(np.abs(np.diff(states, axis=0)))/dt)
            if peak > maximum_speed*(1.+1e-6) or np.any(states < bounds[:, 0]) or np.any(states > bounds[:, 1]):
                raise RuntimeError("open-hand reindex exceeds its joint speed or position bounds")
            for candidate in states:
                hit = check(candidate, open_hand)
                if hit is not None:
                    record["planned_geometry_stop"] = hit
                    raise RuntimeError(f"planned open-hand reindex collision: {hit}")
            np.save(output / "open_hand_arm_path_rad.npy", states)
            record.update(stage="REINDEXING_WITH_OPEN_HAND", reindex_first_step=int(stepper.step_index),
                          path_duration_s=count*dt, maximum_planned_arm_speed_rad_s=peak,
                          arm_target_offset_rad=offset.tolist(), body_yaw_inferred_from_hand_or_nut=False)
            save()
            world.play()
            for candidate in states:
                advance("nut_index_free_rotate", candidate, open_hand)
            for _ in range(round(float(settings["open_hold_duration_s"])/dt)):
                advance("nut_index_free_final_hold", states[-1], open_hand)
            record["reindex_last_step"] = int(stepper.step_index)
            fresh, _ = observe("after_open_hand_reindex")
            record.update(completed=True, stage="OPEN_HAND_REINDEX_FINISHED_REQUIRES_RETENTION_EVALUATION",
                          final_observation=fresh, final_arm_target_rad=states[-1].tolist(),
                          final_hand_target_rad=open_hand.tolist())
    except Exception as error:
        record.update(failure_stage=record["stage"], stage="STOPPED", failure_reason=str(error))
    finally:
        world.pause()
        record.update(last_step=int(stepper.step_index), outer_abort_reason=stepper.abort_reason)
        with gzip.open(output / "joint_ft_samples.json.gz", "wt", encoding="utf-8") as stream:
            json.dump(_json_ready(ft.samples[first_ft:]), stream, separators=(",", ":"))
        with gzip.open(output / "commands.json.gz", "wt", encoding="utf-8") as stream:
            json.dump(commands, stream, separators=(",", ":"))
        save()
    return _json_ready(record)
