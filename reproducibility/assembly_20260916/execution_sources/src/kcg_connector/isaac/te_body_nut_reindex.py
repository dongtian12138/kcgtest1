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


def can_recover_rotation_stop(rotation_record,latest_sensor_phase,step_index,outer_abort_reason):
    reasons={'recoverable nut turn stop: LOADED_POSE_ERROR',
        'recoverable nut turn stop: GRASP_RELATION_CHANGED',
        'recoverable nut turn stop: NO_OBSERVED_AXIAL_PROGRESS',
        'recoverable nut turn stop: OBSERVED_GRIP_SLIP',
        'recoverable nut turn stop: OBSERVED_CONTACT_STALL',
        'recoverable nut turn stop: PLANAR_COMPLIANCE_TRAVEL',
        'recoverable nut turn stop: PREPARATION_NOT_READY',
        'recoverable nut turn stop: COORDINATED_PROGRESS_DID_NOT_RECOVER',
        'recoverable nut turn stop: VISUAL_TRACKING_LOST'}
    return bool(not rotation_record.get('completed') and rotation_record.get('failure_reason') in reasons
        and latest_sensor_phase.startswith('key_probe_nut_rotation_')
        and rotation_record.get('last_step')==step_index
        and rotation_record.get('outer_abort_reason') is None and outer_abort_reason is None)


def run_nut_release_and_reindex(repository, runtime, stepper, dynamic, grip,
                               rotation_record, world_from_socket, settings, output):
    import omni.replicator.core as rep
    import omni.usd
    from scipy.spatial.transform import Rotation

    from kcg_connector.grasp.robust.bounded_hand_base_ik import solve_bounded_hand_base_ik
    from te_body_socket_observation import observe_current_plug_from_rgbd
    def observe_released_plug_from_rgbd(*args):
        return observe_current_plug_from_rgbd(*args,runtime)
    from te_foundationpose_handoff_runtime import _json_ready, MOVEIT_SOFT_ARM_BOUNDS_RAD, control
    from carts_v2.fast_json import dump_array
    grip=rotation_record.get('continuation_grip',grip)

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
        if settings.get('coaxial_measured_release') and phase in ('nut_index_free_rotate','nut_index_free_final_hold'):
            from te_worm_drive import finger_output_posture_targets
            hand,_=finger_output_posture_targets(world.hand_mechanism,q[8:],
                world.hand_mechanism.reference,settings['open_hand_positions_rad'],dt)
            unloading=True
        hit = check(q[:7], q[7:], nut_contact=nut_contact)
        if hit is None and not unloading:
            hit = check(arm, hand, nut_contact=nut_contact)
        if hit is not None:
            record["geometry_stop"] = hit
            raise RuntimeError(f"nut reindex geometry: {hit}")
        before = int(stepper.step_index)
        velocity_reference=None
        if (settings.get('bounded_release_braking',False) and settings.get('coaxial_measured_release')
                and phase=='key_probe_nut_index_unload'):
            from te_nut_motion import bounded_release_hold_velocity
            gravity=stepper.robot.get_dof_gravity_compensation_forces(
                indices=0,dof_indices=stepper.arm_indices).numpy().reshape(7)
            velocity_reference,braking=bounded_release_hold_velocity(arm,q[:7],
                np.asarray(stepper.latest[1])[:7],gravity,
                stiffness=float(stepper.settings['arm_stiffness']),
                damping=float(stepper.effective_arm_damping_nm_s_rad),
                effort_limit=float(stepper.settings['arm_drive_maximum_effort_nm']))
            if np.any(velocity_reference):
                braking.update(step=before)
                record.setdefault('bounded_release_braking',[]).append(braking)
        stepper.advance(phase, np.asarray(arm).copy(), np.asarray(hand).copy(),
            arm_velocity_target=velocity_reference)
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
        recovery_unload=bool(settings.get('allow_recovery_reindex',False) and (
            can_recover_rotation_stop(rotation_record,ft.samples[-1]['phase'],int(stepper.step_index),stepper.abort_reason)
            or can_unload_after_transmission_reserve_stop(rotation_record,ft.samples[-1]['phase'],int(stepper.step_index),stepper.abort_reason)))
        stopped_unload = stopped_hold_unload or stopped_torsion_unload or stopped_reserve_unload or recovery_unload
        if ((not rotation_record.get("completed") and not stopped_unload)
                or stepper.abort_reason is not None):
            raise RuntimeError("the current bounded rotation did not finish")
        record["preceding_rotation_controller_completed"] = bool(rotation_record.get("completed"))
        record["unload_after_additional_lateral_hold_stop"] = stopped_hold_unload
        record["release_after_additional_torsional_pilot_stop"] = stopped_torsion_unload
        record["release_after_transmission_reserve_stop"] = stopped_reserve_unload
        record['recovery_reindex_after_soft_stop']=recovery_unload
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
        if rotation_record.get('load_compensated_hold_arm_target_rad') is not None:
            arm=np.asarray(rotation_record['load_compensated_hold_arm_target_rad']).copy()
            record['release_target_source']='LAST_APPLIED_NATIVE_HOLD_EQUIVALENT_WITH_LOAD_FEEDFORWARD_TRANSFERRED'
        open_hand = np.asarray(settings["open_hand_positions_rad"])
        lower, upper = np.asarray(grip["finite_preload_bounds_rad"])
        tare = np.asarray(grip["new_grasp_effort_tare_nm"])
        effort = np.asarray(grip["effort_reference_nm"])
        if not np.all(hand[1:] > open_hand[1:]):
            raise ValueError("this nut release requires the established positive closing directions")
        increment = float(dynamic["finger_maximum_speed_rad_s"])*dt
        stiffness = float(dynamic["hand_stiffness"])
        root_observer=runtime.get('nut_root_moment_observer')
        shared_motor=world.hand_mechanism if root_observer is not None else None
        coaxial_release=bool(settings.get('coaxial_measured_release',False))
        if coaxial_release:
            if shared_motor is None:raise ValueError('Measured release requires the current shared hand')
            arm=np.asarray(stepper.latest[0])[:7].copy()
            hand[1:]=[shared_motor.drives[name].input_angle for name in ('f1j2','f2j1','f3j2')]
            stepper.payload_compensation_fraction=0.
            record['release_target_source']='CURRENT_ARM_ENCODERS_WITH_GRAVITY_ONLY_AND_NO_CONTACT_FEEDFORWARD'
            from te_worm_drive import finger_output_posture_targets
        if settings.get('require_release_load_observation',False) and root_observer is None:
            raise RuntimeError('current release requires the calibrated robot-side root observer')
        if root_observer is not None:
            from te_nut_motion import released_grip_readiness
            stiffness=float(grip.get('root_moment_preload',{}).get('position_stiffness_reference_nm_rad',120.))
        count = max(1, round((.2 if coaxial_release else float(dynamic["hold_duration_s"]))/dt))
        record.update(initial_arm_target_rad=arm.tolist(), initial_hand_target_rad=hand.tolist(),
                      open_hand_target_rad=open_hand.tolist(), unload_duration_s=count*dt,
                      stage="UNLOADING_NUT_PAD_EFFORT")
        save()
        world.play()
        initial_unload_moments=None
        for index in range(count):
            if coaxial_release:
                hand[1:]=[shared_motor.drives[name].input_angle for name in ('f1j2','f2j1','f3j2')]
                advance('key_probe_nut_index_unload',arm,hand,nut_contact=True,unloading=True)
                continue
            loaded = np.asarray(stepper.latest[2])[8:]-tare[1:]
            if root_observer is not None:
                loaded=np.asarray(stepper.latest[2])[8:]-root_observer.tare_reaction-(
                    root_observer._system(np.asarray(stepper.latest[0]))[2]-root_observer.tare_gravity)
            if initial_unload_moments is None:
                initial_unload_moments=loaded.copy()
            reference=initial_unload_moments*(1.-control.minimum_jerk_blend((index+1)/count))
            if shared_motor is not None:
                from te_worm_drive import finger_force_motor_targets
                filtered_unload_moments=root_observer.filter_moments(loaded,stepper.step_index,dt,.05)
                hand,motor_rows=finger_force_motor_targets(shared_motor,np.asarray(stepper.latest[0])[8:],hand,
                    reference,filtered_unload_moments,dt,stiffness,1/6,float(dynamic['finger_maximum_speed_rad_s']),
                    .02,np.minimum(lower,open_hand),upper,allow_closing=False)
                record['last_unload_motor_command']=motor_rows
                record['unload_motor_interface']='SHARED_FINITE_INPUT_VELOCITY_WITH_NO_ACTIVE_CLOSING'
            else:
                hand[1:] = np.clip(hand[1:]+np.clip((reference-loaded)/stiffness, -increment, 0.),
                                   lower[1:], upper[1:])
            advance("key_probe_nut_index_unload", arm, hand, nut_contact=True, unloading=True)
        record["unload_last_step"] = int(stepper.step_index)
        record["stage"] = "OPENING_NUT_PADS"
        start = hand.copy()
        count = round(.8/dt) if coaxial_release else max(1, int(np.ceil(np.max(np.abs(open_hand-start))/increment)))
        for index in range(count):
            if coaxial_release:
                hand,motor_rows=finger_output_posture_targets(shared_motor,np.asarray(stepper.latest[0])[8:],hand,open_hand,dt)
                record['last_opening_motor_command']=motor_rows
            else:hand = start+(index+1)/count*(open_hand-start)
            advance("key_probe_nut_index_open", arm, hand, nut_contact=True,unloading=coaxial_release)
        record["opening_last_step"] = int(stepper.step_index)
        # Keep contact-aware protection until actual encoder geometry and
        # measured unloading agree. Merely finishing the open target is not release.
        ready=False
        for _ in range(round(float(settings["open_hold_duration_s"])/dt)):
            q,_=measured()
            if coaxial_release:
                hand,motor_rows=finger_output_posture_targets(shared_motor,q[8:],hand,open_hand,dt)
            clear=check(q[:7],q[7:],nut_contact=False) is None
            if root_observer is not None:
                moments=np.asarray(stepper.latest[2])[8:]-root_observer.tare_reaction-(
                    root_observer._system(q)[2]-root_observer.tare_gravity)
                observation=released_grip_readiness(clear,moments,
                    float(settings.get('release_root_moment_tolerance_nm',dynamic['contact_effort_rise_nm'])))
                ready=observation['ready'];record['release_readiness']=observation
            else:ready=clear
            if ready:
                stepper.payload_compensation_fraction=0.
                advance('nut_index_free_open_hold',arm,hand if coaxial_release else open_hand,unloading=coaxial_release)
            else:advance('key_probe_nut_index_open',arm,hand if coaxial_release else open_hand,nut_contact=True,unloading=coaxial_release)
        q,_=measured()
        clear=check(q[:7],q[7:],nut_contact=False) is None
        if root_observer is not None:
            moments=np.asarray(stepper.latest[2])[8:]-root_observer.tare_reaction-(
                root_observer._system(q)[2]-root_observer.tare_gravity)
            observation=released_grip_readiness(clear,moments,
                float(settings.get('release_root_moment_tolerance_nm',dynamic['contact_effort_rise_nm'])))
            ready=observation['ready'];record['release_readiness']=observation
        else:ready=clear
        record['actual_open_hand_positions_rad']=q[7:].tolist()
        if not ready:raise RuntimeError('nut release stopped: OUTPUT_OPENING_OR_UNLOADING_NOT_CONFIRMED')
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
            if settings.get('joint7_only',False):
                from te_nut_motion import joint7_return_path
                held=np.asarray(ft.samples[-1]['active_targets_rad'][:7]).copy()
                initial_encoder=rotation_record.get('initial_arm_encoder_rad')
                if initial_encoder is None:
                    initial_encoder=grip['fixed_arm_target_rad']
                target_joint7=float(initial_encoder[6])
                source_speed=float(inputs.robot_model.joints[control.ARM_JOINT_NAMES[6]].limit.velocity)
                speed=min(source_speed,float(settings.get('maximum_joint7_speed_rad_s',1.)))
                states,details=joint7_return_path(held,target_joint7,bounds[:,0],bounds[:,1],dt,
                    maximum_speed=speed,maximum_acceleration=float(settings.get('maximum_joint7_acceleration_rad_s2',2.)))
                for candidate in states:
                    hit=check(candidate,open_hand)
                    if hit is not None:raise RuntimeError(f'joint7-only return path is not clear: {hit}')
                record.update(stage='REINDEXING_JOINT7_WITH_OPEN_HAND',reindex_first_step=int(stepper.step_index),
                    path_duration_s=details['duration_s'],maximum_planned_arm_speed_rad_s=details['maximum_planned_speed_rad_s'],
                    arm_target_offset_rad=(held-q[:7]).tolist(),joint7_return=details)
                np.save(output/'open_hand_arm_path_rad.npy',states);save();world.play()
                for candidate in states[1:]:advance('nut_index_free_rotate',candidate,open_hand)
                settle_limit=float(settings.get('joint7_settle_timeout_s',.5))
                tolerance=float(settings.get('joint7_position_tolerance_rad',.003))
                for _ in range(max(1,round(settle_limit/dt))):
                    advance('nut_index_free_final_hold',states[-1],open_hand)
                    actual=np.asarray(stepper.latest[0])
                    if abs(actual[6]-target_joint7)<=tolerance and abs(stepper.latest[1][6])<=.02:break
                if abs(stepper.latest[0][6]-target_joint7)>tolerance:
                    raise RuntimeError('joint7 return did not reach its measured endpoint within the settle budget')
                record['joint7_return']['actual_final_error_rad']=float(stepper.latest[0][6]-target_joint7)
                record['reindex_last_step']=int(stepper.step_index)
                fresh,_=observe('after_open_hand_reindex')
                record.update(completed=True,stage='OPEN_HAND_REINDEX_FINISHED_REQUIRES_RETENTION_EVALUATION',
                    final_observation=fresh,final_arm_target_rad=states[-1].tolist(),final_hand_target_rad=open_hand.tolist())
            else:
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
        with gzip.open(output / "joint_ft_samples.json.gz", "wt", encoding="utf-8",compresslevel=1) as stream:
            dump_array(stream,ft.samples[first_ft:])
        with gzip.open(output / "commands.json.gz", "wt", encoding="utf-8",compresslevel=1) as stream:
            dump_array(stream,commands)
        save()
    return _json_ready(record)
