"""Bounded robot-driven nut rotation with independent axial force admittance.

The thread lead is deliberately absent from the online position command. Any
axial progress must arise from contact and the measured wrist force response.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np


def observed_guided_feature_fit(body, socket, configuration):
    """Nominal feature-displacement budget from current measured centre/axis.

    Body key yaw is retained by the preceding guided state, not inferred here.
    This is an admission check, not a substitute for physical contact audits.
    """
    body, socket = np.asarray(body).reshape(4, 4), np.asarray(socket).reshape(4, 4)
    axis = socket[:3, 2]
    cosine = float(np.clip(-axis @ body[:3, 2], -1., 1.))
    tilt = float(np.arccos(cosine))
    delta = body[:3, 3]-socket[:3, 3]
    lateral = float(np.linalg.norm(delta-axis*(axis@delta)))
    pin = (lateral+abs(float(configuration["source_pin_tip_z_body_m"]))*np.sin(tilt)
        +float(configuration["pin_pattern_radius_upper_m"])*(1.-np.cos(tilt)))
    key = lateral+.007645401*np.sin(tilt)+.018821401*(1.-np.cos(tilt))
    overlap = -float(axis@delta)-.000762*abs(cosine)-.0188214*np.sin(tilt)
    accepted = (cosine > 0 and overlap > .0001
        and pin <= float(configuration["pin_lateral_displacement_budget_m"])
        and key <= float(configuration["key_lateral_displacement_budget_m"]))
    return {**configuration, "centre_error_m": lateral, "axis_error_deg": float(np.degrees(tilt)),
        "pin_lateral_displacement_bound_m": float(pin), "key_lateral_displacement_bound_m": float(key),
        "minimum_key_guide_overlap_m": overlap, "accepted": bool(accepted),
        "body_key_yaw_assumed_retained_by_source_guide": True,
        "online_object_or_contact_truth_used": False}


def interface_wrench_from_wrist(wrist_wrench_world, wrist_position_world,
                                plug_origin_world, payload_com_world,
                                payload_mass_kg, scene_gravity_m_s2,
                                socket_rotation_world):
    """Remove downward payload gravity and shift the remaining wrench origin.

    The scene stores signed world-z gravity; other existing contracts store its
    positive magnitude. Both describe the same downward acceleration here.
    Hand gravity/inertia compensation must already have been applied.
    """
    measured = np.asarray(wrist_wrench_world, dtype=np.float64).reshape(6)
    wrist = np.asarray(wrist_position_world, dtype=np.float64).reshape(3)
    origin = np.asarray(plug_origin_world, dtype=np.float64).reshape(3)
    com = np.asarray(payload_com_world, dtype=np.float64).reshape(3)
    rotation = np.asarray(socket_rotation_world, dtype=np.float64).reshape(3, 3)
    gravity_force = np.asarray([0., 0., -abs(float(scene_gravity_m_s2)) * float(payload_mass_kg)])
    force = measured[:3] - gravity_force
    moment = (measured[3:] - np.cross(com-wrist, gravity_force)
              + np.cross(wrist-origin, force))
    return np.r_[rotation.T @ force, rotation.T @ moment]


def interface_wrench_from_sensor_sample(sample, task_axes_world, hand_world,
                                        interface_origin_world, payload_com_world,
                                        payload_mass_kg, scene_gravity_m_s2,
                                        output_axes_world):
    """Use the same causal compensation for current and saved sensor samples."""
    task_axes, hand = np.asarray(task_axes_world), np.asarray(hand_world)
    measured = np.asarray(sample["gravity_and_dynamic_compensated_task_wrench"])
    force, moment = task_axes @ measured[:3], task_axes @ measured[3:]
    prediction = sample["dynamic_inertia_prediction"]
    if prediction is None or not prediction["ready"]:
        raise RuntimeError("the causal hand-inertia prediction is unavailable")
    if not prediction["applied_to_safety_residual"]:
        inertia = np.asarray(prediction["predicted_positive_inertia_wrench_sensor"])
        force += hand[:3, :3] @ inertia[:3]
        moment += hand[:3, :3] @ inertia[3:]
    return interface_wrench_from_wrist(
        np.r_[force, moment], hand[:3, 3], interface_origin_world,
        payload_com_world, payload_mass_kg, scene_gravity_m_s2, output_axes_world)


def run_body_nut_rotation(repository, runtime, stepper, dynamic, grip,
                          world_from_socket, settings, output, *, initial_position_axis_observation=None):
    if settings.get('recovery',{}).get('enabled',False):
        from te_body_nut_continuation import run_nut_rotation_with_recovery
        return run_nut_rotation_with_recovery(repository,runtime,stepper,dynamic,grip,
            world_from_socket,settings,output,initial_position_axis_observation=initial_position_axis_observation)
    if settings.get('in_turn_feedback', {}).get('enabled', False):
        from te_observed_nut_rotation import run_observed_nut_rotation
        return run_observed_nut_rotation(repository,runtime,stepper,dynamic,grip,
            world_from_socket,settings,output)
    return _run_body_nut_rotation_interval(repository,runtime,stepper,dynamic,grip,
        world_from_socket,settings,output,initial_position_axis_observation=initial_position_axis_observation)


def _run_body_nut_rotation_interval(repository, runtime, stepper, dynamic, grip,
                          world_from_socket, settings, output, *, initial_position_axis_observation=None,
                          observation_session=None):
    import omni.replicator.core as rep
    import omni.usd
    from scipy.spatial.transform import Rotation
    from te_nut_motion import ScalarMotion,limit_joint_velocity_without_changing_direction,classify_observed_progress
    from carts_v2.fast_json import dumps as encode_row,dump_array

    from te_body_socket_observation import observe_released_plug_from_rgbd as full_observe_released_plug
    observe_released_plug_from_rgbd=full_observe_released_plug
    if settings.get('visual_progress',{}).get('enabled',False):
        from te_body_socket_observation import observe_current_plug_from_rgbd
        def observe_released_plug_from_rgbd(*args):
            return observe_current_plug_from_rgbd(*args,runtime)
    from te_foundationpose_handoff_runtime import (
        MOVEIT_SOFT_ARM_BOUNDS_RAD, _json_ready, control,
    )

    repository, output = Path(repository).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    world, inputs, ft = runtime["world"], runtime["inputs"], runtime["nail_body_ft_auditor"]
    first_ft = len(ft.samples)
    dt = float(dynamic["physics_dt_s"])
    socket = np.asarray(world_from_socket, dtype=np.float64).reshape(4, 4)
    axis = socket[:3, 2]
    angle = np.deg2rad(float(settings["rotation_about_socket_plus_z_deg"]))
    maximum_speed = np.deg2rad(float(settings["maximum_rotation_speed_deg_s"]))
    duration = max(dt, 1.875 * abs(angle) / maximum_speed)
    maximum_acceleration = settings.get("maximum_rotation_acceleration_deg_s2")
    if maximum_acceleration is not None:
        maximum_acceleration = np.deg2rad(float(maximum_acceleration))
        if not np.isfinite(maximum_acceleration) or maximum_acceleration <= 0:
            raise ValueError("rotation acceleration reference must be finite and positive")
        duration = max(duration, np.sqrt((10. / np.sqrt(3.)) * abs(angle) / maximum_acceleration))
    scalar_profile=None
    if settings.get('rotation_profile')=='trapezoid':
        if maximum_acceleration is None:raise ValueError('the cruise profile needs a declared acceleration limit')
        scalar_profile=ScalarMotion(angle,maximum_speed,maximum_acceleration)
        duration=max(dt,scalar_profile.duration)
    settle_s = float(settings["axial_settle_duration_s"])
    hold_s = float(settings["post_rotation_hold_s"])
    filter_tau = float(settings.get("contact_estimate_filter_time_constant_s", 0.0))
    if not np.isfinite(filter_tau) or filter_tau < 0:
        raise ValueError("contact-estimate time constant must be finite and nonnegative")
    preparation_force_reference = float(settings["axial_force_reference_n"])
    turn_force_reference = float(settings.get("turn_axial_force_reference_n", preparation_force_reference))
    force_reference_ramp_s = float(settings.get("turn_force_reference_ramp_duration_s", 0.5))
    engagement = settings.get("engagement_axial_assist", {})
    engagement_enabled = bool(engagement.get("enabled", False))
    engagement_start_deg = 0.
    engagement_applied_deg = 0.
    if engagement_enabled:
        engagement_force = float(engagement["additional_downward_force_n"])
        engagement_end_deg = float(engagement["ending_cumulative_command_deg"])
        engagement_fade_deg = float(engagement["ramp_out_angle_deg"])
        engagement_start_deg = float(runtime.get("engagement_loaded_turn_command_deg",
            engagement.get("initial_cumulative_command_deg", 0.)))
        if not (0 < engagement_force <= 3.0400615 + 1e-9
                and 0 < engagement_fade_deg <= engagement_end_deg <= 40.
                and np.isfinite(engagement_start_deg) and engagement_start_deg >= 0.):
            raise ValueError("engagement assistance exceeds the declared finite source-bench input")
    effort_tau = float(settings.get("finger_effort_regulation_time_constant_s", 0.0))
    if not np.isfinite(effort_tau) or effort_tau < 0:
        raise ValueError("finger effort regulation time constant must be finite and nonnegative")
    effort_gain = dt/(effort_tau+dt) if effort_tau > 0 else 1.
    regulate_finger_effort = bool(settings.get("regulate_finger_effort_during_rotation", True))
    regulate_preparation_effort = bool(settings.get("regulate_finger_effort_during_preparation", True))
    rate_guard = settings.get("finger_effort_limited_turn", {})
    rate_guard_enabled = bool(rate_guard.get("enabled", False))
    rate_extra_time = float(rate_guard.get("maximum_extra_turn_time_s", 20.)) if rate_guard_enabled else 0.
    rate_slow_margin = float(rate_guard.get("begin_slowing_reserve_nm", .5))
    rate_stop_margin = float(rate_guard.get("hold_reserve_nm", .1))
    rate_hold_limit = float(rate_guard.get("maximum_near_hold_s", 2.))
    rate_lost_time = 0.
    rate_near_hold_time = 0.
    rate_turn_finished_s = None
    rate_cycle_finished = False
    turn_rate_scale = 1.
    model_elastic_efforts = None
    model_effort_margin = None
    release_reserve = settings.get("controlled_release_effort_reserve_nm")
    measured_tracking=settings.get('measured_tracking',{})
    tracking_enabled=bool(measured_tracking.get('enabled',False))
    progress_settings=settings.get('visual_progress',{})
    progress_enabled=bool(progress_settings.get('enabled',False))
    tracking_context=runtime.setdefault('plug_visual_tracking_context',{}) if progress_enabled else {}
    last_observed_body=None
    next_observation_s=0.
    seating_count=0
    visual_follow=progress_settings.get('reference_following',{})
    visual_follow_enabled=progress_enabled and bool(visual_follow.get('enabled',False))
    visual_follow_translation=np.zeros(3);visual_follow_rotvec=np.zeros(3)
    motor_force_control=settings.get('finger_motor_force_control',{})
    root_moment_filtered=None
    axial_lead=settings.get('axial_lead_following',{})
    axial_lead_enabled=bool(axial_lead.get('enabled',False))
    turn_axial_origin=None
    axial_correction=0.
    if release_reserve is not None and not 0 < float(release_reserve) <= .1:
        raise ValueError("controlled release reserve must be positive and no greater than0.1Nm")
    if (not all(np.isfinite(v) and v >= 0 for v in
                (preparation_force_reference, turn_force_reference))
            or not np.isfinite(force_reference_ramp_s) or force_reference_ramp_s <= 0):
        raise ValueError("axial force references must be finite and nonnegative with a positive ramp duration")
    arm = np.asarray(grip["fixed_arm_target_rad"], dtype=np.float64)
    hand_target = np.asarray(grip["final_hand_target_rad"], dtype=np.float64)
    tare = np.asarray(grip["new_grasp_effort_tare_nm"], dtype=np.float64)
    lower_hand, upper_hand = np.asarray(grip["finite_preload_bounds_rad"], dtype=np.float64)
    desired_effort = np.asarray(grip["effort_reference_nm"], dtype=np.float64)
    increment = float(dynamic["finger_maximum_speed_rad_s"]) * dt
    bounds = np.asarray([MOVEIT_SOFT_ARM_BOUNDS_RAD[name] for name in control.ARM_JOINT_NAMES])
    check = runtime["nut_regrasp_geometry_check"]
    locate_bounds = runtime["nut_regrasp_locate_visual_bounds"]
    record = {"completed": False, "stage": "BEFORE_POSTGRIP_PALM_OBSERVATION",
        "simulation_only": True, "hardware_authorized": False,
        "online_object_or_contact_truth_used": False,
        "direct_object_force_or_pose_command_used": False,
        "thread_lead_used_for_axial_commands": axial_lead_enabled,
        "physical_thread_progress_verified": False,
        "physical_result": "REQUIRES_POSTRUN_NUT_ROTATION_KEY_REACTION_AND_THREAD_ADVANCE",
        "first_step": int(stepper.step_index), "settings": settings,
        "control_samples_file": str(output / "nut_rotation_control_samples.jsonl"),
        "sample_count": 0, "sample_storage": "ONE_APPEND_ONLY_JSONL_FILE"}
    sample_stream = Path(record["control_samples_file"]).open("x", encoding="utf-8", buffering=1)
    record["finger_effort_regulation"] = {
        "enabled": regulate_finger_effort,
        "disabled_mode": (None if regulate_finger_effort else "HOLD_REACHED_PREPARATION_END_FINITE_NATIVE_POSITION_PRELOAD"),
        "preparation_effort_regulation_enabled": regulate_preparation_effort,
        "time_constant_s": effort_tau, "per_step_error_relaxation_gain": effort_gain,
        "measured_signal": "NATIVE_PROJECTED_JOINT_FORCE_MINUS_CURRENT_GRIP_TARE",
        "measurement_filtered_or_rezeroed": False,
        "force_references_drive_gains_effort_caps_and_position_bounds_changed": False,
        "not_an_identified_optimal_closed_loop_bandwidth": True,
    }

    def save():
        (output / "nut_rotation_controller_result.json").write_text(
            json.dumps(_json_ready(record), ensure_ascii=False, indent=2) + "\n")

    def measured_hand():
        q = np.asarray(stepper.latest[0], dtype=np.float64)
        return q, np.asarray(inputs.robot_model.forward_kinematics(tuple(q), enforce_limits=False)["handbase_link"])

    try:
        initial_q,_=measured_hand()
        record['initial_arm_encoder_rad']=initial_q[:7].tolist()
        if not grip["completed"] or stepper.abort_reason is not None:
            raise RuntimeError("the current nut grip controller is not ready")
        if rate_guard_enabled:
            rate_mechanism = getattr(world, "hand_mechanism", None)
            if (rate_mechanism is None or not 0 < rate_extra_time <= 20.
                    or not 0 < rate_stop_margin < rate_slow_margin <= .5
                    or not 0 < rate_hold_limit <= 2.):
                raise ValueError("effort-limited turning requires the shared mechanism and finite timing/reserve bounds")
            rate_names = ("f1j2", "f2j1", "f3j2")
            rate_caps = np.array([rate_mechanism.drives[n].reference.transmission_effort_boundary for n in rate_names])
            record["finger_effort_limited_turn"] = {
                **rate_guard, "finite_transmission_caps_nm": rate_caps.tolist(),
                "effort_source": "ELASTIC_TRANSMISSION_MODEL_STATE_AND_MEASURED_OUTPUT_ENCODERS",
                "object_or_contact_truth_used": False,
                "native_force_caps_changed": False,
                "reported_nominal_profile_acceleration_is_not_an_adaptive_acceleration_bound": True}
        if not runtime["body_assembly_scene"]["report"].get("representative_inner_thread"):
            raise ValueError("this thread-contact pilot requires the authored inner thread")
        root_observer=runtime.get('nut_root_moment_observer')
        if (regulate_finger_effort or regulate_preparation_effort) and getattr(world,'hand_mechanism',None) is not None:
            if root_observer is None:
                from te_three_finger_wrench_observer import ThreeFingerWrenchObserver
                calibration=grip.get('root_moment_preload',{})
                if not calibration.get('gravity_compensated') or 'tare_gravity_nm' not in calibration:
                    raise ValueError('the current grip lacks its original free-space base-moment calibration')
                root_observer=ThreeFingerWrenchObserver(repository,inputs.robot_model,grip['geometry_plan'],
                    sensor_semantics='BASE_BRIDGE_EXTERNAL_MOMENT_ABOUT_O')
                root_observer.tare_reaction=np.asarray(grip['new_grasp_effort_tare_nm'][1:],dtype=float)
                root_observer.tare_gravity=np.asarray(calibration['tare_gravity_nm'],dtype=float)
                runtime['nut_root_moment_observer']=root_observer
            record['finger_force_feedback_source']='CURRENT_BASE_BRIDGE_WITH_POSE_GRAVITY_AND_ORIGINAL_FREE_SPACE_TARE'
            record['active_finger_reference_can_open_and_close']=True
        world.pause()
        if settings.get("grip_hold_impedance"):
            if regulate_finger_effort or regulate_preparation_effort or settings.get("grip_lateral_balance", {}).get("enabled"):
                raise ValueError("holding impedance requires the fixed finite preload mode")
            from te_grip_hold_impedance import activate_grip_hold
            hand_target, transition = activate_grip_hold(runtime, stepper, dynamic, hand_target,
                grip["finite_preload_bounds_rad"], float(settings["grip_hold_impedance"]["stiffness_scale"]))
            record["grip_hold_impedance_transition"] = transition
            record["finger_effort_regulation"]["drive_gains_changed_for_holding"] = True
            record["finger_effort_regulation"].pop("force_references_drive_gains_effort_caps_and_position_bounds_changed", None)
            save()
        q, hand = measured_hand()
        if initial_position_axis_observation is None:
            fresh = observe_released_plug_from_rgbd(
                repository, omni.usd.get_context().get_stage(), world, rep, hand, output / "postgrip_palm")
        else:
            fresh = initial_position_axis_observation
            source=fresh.get('source')
            if (source not in ('SEALED_RGBD_AND_CURRENT_ENCODERS_SAME_GRIP_LOCAL_DIAGNOSTIC',
                               'CURRENT_SEGMENT_RGBD_AND_ENCODERS')
                    or fresh.get('online_object_or_contact_truth_used') is not False):
                raise ValueError('supplied observation must come from actual RGBD and encoders')
            if source=='CURRENT_SEGMENT_RGBD_AND_ENCODERS':
                if (fresh.get('encoder_step')!=int(stepper.step_index)
                        or abs(float(fresh['physics_time_s'])-float(world.current_time))>1e-10):
                    raise ValueError('segment observation is stale')
            else:
                record["local_diagnostic_scope"] = "INITIAL_CONDITION_FROM_COMPLETED_EPISODE_NOT_FULL_ASSEMBLY"
        if not fresh.get("position_and_axis_measured"):
            raise RuntimeError("the post-grip palm image did not measure position and directed axis")
        if progress_enabled:
            tracking_context['last_observation']=fresh
            if fresh.get('tracking_seed_mask'):tracking_context['mask_path']=fresh['tracking_seed_mask']
            last_observed_body=np.asarray(fresh['world_from_plug_five_dof']).copy()
            next_observation_s=float(progress_settings['observation_period_s'])
            record['current_visual_progress']=[]
            record['desired_grasp_relation_held_between_regrasps']=True
        body = np.asarray(fresh["world_from_plug_five_dof"]).reshape(4, 4)
        if progress_enabled:
            runtime.pop('last_nut_progress_observation',None)
            record['visual_progress_anchor']={'depth_m':float(-axis@(body[:3,3]-socket[:3,3])),
                'command_deg':engagement_start_deg,'time_s':float(world.current_time)}
        record['initial_hand_world_from_encoders']=hand.tolist()
        if observation_session is not None:
            observation_session.setdefault('original_body_origin_world_m',body[:3,3].tolist())
            observation_session.setdefault('original_hand_rotation_world',hand[:3,:3].tolist())
            if observation_session.get('locked_hand_from_body') is not None:
                body=hand@np.asarray(observation_session['locked_hand_from_body'])
                record['control_body_pose_source']='FROZEN_POSTPREPARATION_GRIP_RELATION_ABLATION'
            else:
                record['control_body_pose_source']='CURRENT_RGBD_POSITION_AND_AXIS'
        feature_fit_settings = settings.get("guided_feature_fit")
        initial_fit_accepted = False
        if feature_fit_settings is not None:
            initial_fit = observed_guided_feature_fit(body, socket, feature_fit_settings)
            record["initial_current_guided_feature_fit"] = initial_fit
            initial_fit_accepted = initial_fit["accepted"]
            if initial_fit_accepted:
                settle_s = 0.
                record["unnecessary_preparation_motion_skipped"] = True
        if settings.get('force_pose_margin_refinement',False):
            initial_fit_accepted=False
            record['additional_margin_refinement_requested']=True
        if settings.get("guided_fit_continuation"):
            fit = settings["guided_fit_continuation"]
            directed_cosine = float(np.clip(-axis @ body[:3, 2], -1., 1.))
            tilt = float(np.arccos(directed_cosine))
            delta = body[:3, 3]-socket[:3, 3]
            lateral = float(np.linalg.norm(delta-axis*(axis@delta)))
            pin_bound = (lateral+abs(float(fit["source_pin_tip_z_body_m"]))*np.sin(tilt)
                +float(fit["pin_pattern_radius_upper_m"])*(1.-np.cos(tilt)))
            key_bound = lateral+.007645401*np.sin(tilt)+.018821401*(1.-np.cos(tilt))
            guide_overlap = -float(axis@delta)-.000762*abs(directed_cosine)-.0188214*np.sin(tilt)
            accepted = (directed_cosine > 0 and guide_overlap > .0001
                and pin_bound <= float(fit["pin_lateral_displacement_budget_m"])
                and key_bound <= float(fit["key_lateral_displacement_budget_m"]))
            record["current_measured_guided_fit"] = {
                **fit, "centre_error_m": lateral, "axis_error_deg": float(np.degrees(tilt)),
                "pin_lateral_displacement_bound_m": float(pin_bound),
                "key_lateral_displacement_bound_m": float(key_bound),
                "minimum_key_guide_overlap_m": guide_overlap, "accepted": bool(accepted),
                "body_key_yaw_assumed_retained_by_source_guide": True,
                "online_object_or_contact_truth_used": False}
            record["postgrip_palm_observation"] = fresh
            save()
            if not accepted:
                raise RuntimeError("current RGBD pose exceeds nominal guided-feature fit budget; no turn commanded")
        record.update(
            body_key_yaw_measured_independently=False,
            body_key_yaw_required=False,
            rear_array_observation_executed=False,
            control_pose_components="POSITION_AND_DIRECTED_AXIS_ONLY",
            guide_entry_precondition="PRECEDING_BODY_KEY_ENTRY_AND_SUPPORTED_NUT_REGRASP_SEQUENCE",
            wired_connector_visibility_validated=False)
        if (initial_position_axis_observation is not None
                and initial_position_axis_observation.get('source')!='CURRENT_SEGMENT_RGBD_AND_ENCODERS'):
            record["guide_entry_precondition"] = "SOURCE_EPISODE_GUIDE_HISTORY; NOT_REEXECUTED_IN_LOCAL_DIAGNOSTIC"
        # This virtual frame follows the nut grip. Its yaw gauge is the hand's
        # current x axis, not an estimate of Body yaw. Key angle belongs to the
        # preceding insertion task; threading uses the seated guide, axis and FT.
        pivot = hand.copy()
        pivot[:3, 3] = body[:3, 3]
        hand_from_pivot = np.linalg.inv(hand) @ pivot
        original_pivot = pivot.copy()
        fixed_wrench_origin_world = original_pivot[:3, 3].copy()
        if observation_session is not None:
            observation_session.setdefault('fixed_wrench_origin_world_m',fixed_wrench_origin_world.tolist())
            fixed_wrench_origin_world=np.asarray(observation_session['fixed_wrench_origin_world_m']).copy()
        desired_pivot = pivot.copy()
        body_axis_rotation = body[:3, :3].copy()
        alignment = settings.get("pre_turn_visual_alignment", {})
        lateral_delta, alignment_rotvec = np.zeros(3), np.zeros(3)
        alignment_duration = 0.0
        if alignment.get("enabled", False) and not initial_fit_accepted:
            observed_axis = body[:3, 2] / np.linalg.norm(body[:3, 2])
            target_axis = -axis
            cross = np.cross(observed_axis, target_axis)
            sine = float(np.linalg.norm(cross))
            alignment_angle = float(np.arctan2(sine, observed_axis @ target_axis))
            lateral_delta = (np.eye(3) - np.outer(axis, axis)) @ (socket[:3, 3] - body[:3, 3])
            if (np.linalg.norm(lateral_delta) > float(alignment["maximum_lateral_correction_m"])
                    or alignment_angle > np.deg2rad(float(alignment["maximum_axis_correction_deg"]))):
                raise RuntimeError("current visual alignment exceeds the bounded correction region")
            if sine > 1e-12:
                alignment_rotvec = cross / sine * alignment_angle
            alignment_duration = max(1.0,
                1.875 * np.linalg.norm(lateral_delta) / float(alignment["maximum_lateral_speed_m_s"]),
                1.875 * alignment_angle / float(alignment["maximum_axis_speed_rad_s"]))
            record["pre_turn_visual_alignment"] = {
                "source": "CURRENT_POSTGRIP_PALM_AXIS_AND_POSITION_WITH_WRIST_SOCKET_VISION",
                "lateral_correction_world_m": lateral_delta.tolist(),
                "axis_correction_world_rotvec_rad": alignment_rotvec.tolist(),
                "axis_correction_deg": float(np.rad2deg(alignment_angle)),
                "duration_s": float(alignment_duration),
                "axial_command_remains_force_admittance": True,
                "object_pose_or_contact_truth_used": False,
            }
        torsion = settings.get("pre_turn_torsional_compliance", {})
        torsion_enabled = bool(torsion.get("enabled", False)) and not initial_fit_accepted
        yaw_offset = 0.0
        planar = settings.get("planar_force_admittance", {})
        planar_enabled = bool(planar.get("enabled", False)) and not initial_fit_accepted
        planar_freeze_after_preparation = bool(planar.get("freeze_after_preparation", False))
        planar_during_preparation = bool(planar.get("activate_during_axial_preparation", False))
        planar_during_alignment = bool(planar.get("activate_during_visual_alignment", False))
        planar_start_s = (0. if planar_during_alignment else
                          alignment_duration + (0. if planar_during_preparation else settle_s))
        planar_offset = np.zeros(2)
        planar_restoring_stiffness = float(planar.get("virtual_restoring_stiffness_n_m", 0.))
        if not np.isfinite(planar_restoring_stiffness) or planar_restoring_stiffness < 0:
            raise ValueError("virtual planar restoring stiffness must be finite and nonnegative")
        if planar_enabled:
            planar_gain = float(planar["admittance_m_per_n_s"])
            maximum_planar_speed = float(planar["maximum_speed_m_s"])
            maximum_planar_offset = float(planar["maximum_offset_m"])
            range_basis=planar.get('force_consistent_range')
            if range_basis is not None:
                old_offset=maximum_planar_offset
                force_limit=float(range_basis['unchanged_wrist_force_limit_n'])
                visual_budget=min(float(range_basis['visual_geometry_allowance_m']),
                                  float(alignment.get('maximum_lateral_correction_m',float('inf'))))
                remaining_visual=visual_budget-float(np.linalg.norm(lateral_delta))
                if not (np.isfinite(force_limit) and force_limit>0 and planar_restoring_stiffness>0
                        and np.isfinite(remaining_visual) and remaining_visual>0):
                    raise ValueError('No finite force/geometry-consistent planar correction range')
                maximum_planar_offset=min(force_limit/planar_restoring_stiffness,remaining_visual)
                record['planar_range_derivation']={
                    'previous_reference_m':old_offset,'derived_maximum_offset_m':maximum_planar_offset,
                    'force_limit_n':force_limit,'restoring_stiffness_n_m':planar_restoring_stiffness,
                    'visual_budget_m':visual_budget,'planned_visual_lateral_correction_m':float(np.linalg.norm(lateral_delta)),
                    'remaining_visual_budget_m':remaining_visual,'force_speed_joint_limits_changed':False}
            if not all(np.isfinite(v) and v > 0 for v in
                       (planar_gain, maximum_planar_speed, maximum_planar_offset)):
                raise ValueError("planar admittance requires finite positive gain and bounds")
            record["planar_force_admittance"] = {
                **planar, "force_reference_socket_xy_n": [0., 0.],
                "maximum_offset_m":maximum_planar_offset,
                "activation": ("FROM_VISUAL_ALIGNMENT_START"
                               if planar_during_alignment else
                               "AFTER_VISUAL_ALIGNMENT_INCLUDING_AXIAL_PREPARATION"
                               if planar_during_preparation else
                               "AFTER_VISUAL_ALIGNMENT_AND_AXIAL_PREPARATION"),
                "activation_elapsed_s": planar_start_s,
                "source": "CAUSAL_FILTERED_WRIST_INTERFACE_FORCE",
                "body_key_yaw_used": False,
                "virtual_restoring_stiffness_n_m": planar_restoring_stiffness,
                "restoring_reference": "ZERO_OFFSET_FROM_CURRENT_VISUAL_ALIGNMENT_REFERENCE",
                "controller_stiffness_not_contact_material_property": True,
            }
        if torsion_enabled:
            if alignment_duration <= 0:
                raise ValueError("pre-turn torsional compliance requires visual alignment")
            yaw_admittance = float(torsion["angular_admittance_rad_per_nm_s"])
            maximum_yaw_speed = np.deg2rad(float(torsion["maximum_speed_deg_s"]))
            maximum_yaw_offset = np.deg2rad(float(torsion["maximum_total_correction_deg"]))
            if not all(np.isfinite(v) and v > 0 for v in (yaw_admittance, maximum_yaw_speed, maximum_yaw_offset)):
                raise ValueError("pre-turn torsional compliance needs finite positive bounds")
            record["pre_turn_torsional_compliance"] = {
                **torsion, "torque_reference_nm": 0.,
                "purpose": "RELIEVE_UNOBSERVED_YAW_PRELOAD_DURING_VISUAL_CENTRE_AXIS_ALIGNMENT",
                "body_key_yaw_inferred_from_nut_or_hand": False,
                "source": "CAUSAL_FILTERED_WRIST_TORQUE_AT_CURRENT_VISUAL_ORIGIN",
            }
        balance = settings.get("grip_lateral_balance", {})
        balance_enabled = bool(balance.get("enabled", False))
        balance_reference = None
        balance_offset_n = np.zeros(3)
        if balance_enabled:
            if regulate_finger_effort or not planar_freeze_after_preparation:
                raise ValueError("grip redistribution requires the retained preparation pose and its own finger-position feedback")
            balance_rate = float(balance["response_rate_s_inv"])
            balance_bound = float(balance["maximum_model_normal_redistribution_n"])
            mechanism = getattr(world, "hand_mechanism", None)
            balance_bound_limit = 8. if mechanism is not None else 1.
            if not (0 < balance_rate <= 1. and 0 < balance_bound <= balance_bound_limit):
                raise ValueError("load redistribution exceeds its declared response or per-finger range")
            if mechanism is not None:
                # Closing-side static series compliance of the existing motor
                # PD and worm output spring. This is a model conversion, not a
                # claim that distributed contact-normal forces are measured.
                ms = mechanism.settings
                balance_stiffness = 1. / (1. / float(ms["transmission_stiffness_nm_rad"])
                    + (1. + float(ms["load_friction_ratio"])) / float(ms["motor_position_kp"]))
            else:
                balance_stiffness = float(dynamic["hand_stiffness"])
            balance_geometry = json.loads((repository / balance["geometry_plan"]).read_text())
            if "first_contacts" in balance_geometry:
                from te_three_finger_wrench_observer import ThreeFingerWrenchObserver
                cad_points = ThreeFingerWrenchObserver(repository, inputs.robot_model,
                    repository / balance["geometry_plan"],
                    sensor_semantics="BASE_BRIDGE_EXTERNAL_MOMENT_ABOUT_O")
                balance_rows = balance_geometry["first_contacts"]
                balance_points_local = cad_points.points_local
                balance_points_body = [row["nearest_nut_point_before_contact_body_frame_m"] for row in balance_rows]
            else:
                balance_rows = balance_geometry["rows"]
                balance_points_local = {row["link"]: row["first_contacts"]["full_cooked_fingertip"]["point_in_link_m"] for row in balance_rows}
                balance_points_body = [row["first_contacts"]["full_cooked_fingertip"]["approach_side_nearest_point_object_m"] for row in balance_rows]
            if tuple(row["link"] for row in balance_rows) != ("f1Link3", "f2Link2", "f3Link3"):
                raise ValueError("expected the existing three-finger CAD contact plan")
            body_from_hand = np.asarray(balance_geometry["canonical_body_from_hand_for_nut_grasp"])
            normals_hand = []
            for point in balance_points_body:
                point = np.asarray(point)
                radial = np.r_[point[:2]/np.linalg.norm(point[:2]), 0.]
                normals_hand.append(body_from_hand[:3, :3].T @ radial)
            normals_hand = np.asarray(normals_hand)
            record["finger_effort_regulation"]["disabled_mode"] = "BOUNDED_WRIST_LATERAL_LOAD_DISTRIBUTION_AFTER_PREPARATION"
            record["grip_lateral_balance"] = {
                **balance, "source": "WRIST_FT_CURRENT_ENCODERS_AND_EXISTING_OFFLINE_CAD",
                "normal_directions_hand": normals_hand.tolist(),
                "sum_model_normal_corrections_reference_n": 0.,
                "closing_side_model_position_stiffness_nm_rad": balance_stiffness,
                "actual_total_normal_force_preservation_claimed": False,
                "model_normal_force_mapping_calibrated": False,
                "object_or_contact_truth_used": False,
                "original_motor_wrench_and_position_limits_retained": True}
        mass = float(settings["payload_mass_kg"])
        com_from_pivot = pivot[:3, :3].T @ (body[:3, :3] @ np.asarray(settings["payload_com_body_m"]))
        record.update(postgrip_palm_observation=fresh,
            hand_from_virtual_nut_axis_frame=hand_from_pivot.tolist(),
            commanded_rotation_duration_s=duration,
            planned_peak_rotation_speed_deg_s=float(np.rad2deg(
                scalar_profile.maximum_acceleration*scalar_profile.ramp_time if scalar_profile else 1.875*abs(angle)/duration)),
            planned_peak_rotation_acceleration_deg_s2=float(np.rad2deg(
                scalar_profile.maximum_acceleration if scalar_profile else (10./np.sqrt(3.))*abs(angle)/duration**2)),
            modeled_gravity_acceleration_world_m_s2=[0., 0., -abs(float(runtime["scene"]["gravity_m_s2"]))],
            grip_effort_reference_nm=desired_effort.tolist(),
            new_pose_relation_scope="CURRENT_PALM_CENTER_AND_CURRENT_ENCODER_GRIP_YAW_GAUGE",
            wrench_semantics="HAND_GRAVITY_COMPENSATED_WRIST_MINUS_KNOWN_PAYLOAD_GRAVITY_SHIFTED_TO_VISUAL_PLUG_ORIGIN",
            contact_estimate_filter={
                "time_constant_s": filter_tau,
                "scheme": "CAUSAL_BACKWARD_EULER_FIRST_ORDER_AT_FIXED_INITIAL_VISUAL_ORIGIN",
                "applies_to": "ADMITTANCE_AND_ADDITIONAL_PILOT_STOPS",
                "raw_estimates_retained": True,
                "existing_joint_and_wrist_protections_remain_active": True,
                "independent_planned_contact_wrist_force_time_constant_s": ft.planned_contact_force_time_constant_s,
            },
            loaded_wrench_rezeroed=False)

        filtered_at_fixed_origin = None
        filtered_last_step = None
        if observation_session is not None and observation_session.get('filtered_wrench_at_fixed_origin') is not None:
            filtered_at_fixed_origin=np.asarray(observation_session['filtered_wrench_at_fixed_origin']).copy()
            filtered_last_step=int(observation_session['filtered_last_step'])

        def sample_at_fixed_origin(sample, hand, current):
            com = current[:3, 3] + current[:3, :3] @ com_from_pivot
            return interface_wrench_from_sensor_sample(
                sample, ft.task_rotation_world, hand, fixed_wrench_origin_world,
                com, mass, runtime["scene"]["gravity_m_s2"], socket[:3, :3])

        if filter_tau > 0 and filtered_at_fixed_origin is None:
            # The completed grip already supplies two seconds of loaded sensor
            # history. Initialize from it instead of an isolated noisy sample.
            # Current vision sets the virtual origin; past encoder poses carry
            # that fixed-grip relation through the recent hold interval.
            history_s = float(settings.get("contact_filter_initialization_history_s", 10 * filter_tau))
            if not np.isfinite(history_s) or history_s <= 0:
                raise ValueError("filter initialization history must be positive and finite")
            count_history = max(1, round(history_s / dt))
            history = ft.samples[-count_history:]
            if len(history) != count_history or any(s["phase"] != "key_probe_nut_grip_hold" for s in history):
                raise RuntimeError("the completed grip does not supply the required sensor history")
            for sample in history:
                historical_hand = np.eye(4)
                historical_hand[:3, :3] = sample["handbase_rotation_world_row_major"]
                historical_hand[:3, 3] = sample["handbase_position_world_m"]
                value = sample_at_fixed_origin(sample, historical_hand, historical_hand @ hand_from_pivot)
                if filtered_at_fixed_origin is None:
                    filtered_at_fixed_origin = value.copy()
                else:
                    filtered_at_fixed_origin += dt / (filter_tau + dt) * (value-filtered_at_fixed_origin)
                filtered_last_step = int(sample["step"])
            record["contact_estimate_filter"].update(
                initialization_source="COMPLETED_GRIP_SENSOR_HISTORY_ENCODERS_AND_CURRENT_VISION",
                history_sample_count=len(history), history_first_step=int(history[0]["step"]),
                history_last_step=filtered_last_step,
                initial_filtered_wrench_at_fixed_origin=filtered_at_fixed_origin.tolist(),
                loaded_wrench_rezeroed=False)
        elif filter_tau > 0:
            record['contact_estimate_filter'].update(initialization_source='CONTINUED_SAME_EPISODE_FILTER_STATE',
                fixed_origin_world_m=fixed_wrench_origin_world.tolist(),history_last_step=filtered_last_step,
                initial_filtered_wrench_at_fixed_origin=filtered_at_fixed_origin.tolist(),loaded_wrench_rezeroed=False)

        def observe():
            nonlocal filtered_at_fixed_origin, filtered_last_step
            q, hand = measured_hand()
            current = hand @ hand_from_pivot
            sample = ft.samples[-1]
            at_fixed = sample_at_fixed_origin(sample, hand, current)
            offset = socket[:3, :3].T @ (current[:3, 3] - fixed_wrench_origin_world)
            raw = at_fixed.copy()
            raw[3:] -= np.cross(offset, raw[:3])
            corrected = raw.copy()
            if filter_tau > 0:
                if int(sample["step"]) < filtered_last_step:
                    raise RuntimeError("contact sensor time moved backwards")
                if int(sample["step"]) > filtered_last_step:
                    filtered_at_fixed_origin += dt / (filter_tau + dt) * (at_fixed - filtered_at_fixed_origin)
                    filtered_last_step = int(sample["step"])
                corrected = filtered_at_fixed_origin.copy()
                corrected[3:] -= np.cross(offset, corrected[:3])
            return q, hand, current, corrected, raw

        world.play()
        record["stage"] = "VISUAL_AXIS_ALIGNMENT_FORCE_SETTLE_AND_ONE_SMOOTH_ROTATION"
        save()
        count = round((alignment_duration + settle_s + duration + hold_s + rate_extra_time) / dt)
        preparation_end = alignment_duration + settle_s
        visual_feedback = None if initial_fit_accepted else settings.get("pre_turn_visual_feedback")
        feedback_verified = visual_feedback is None
        feedback_segment = None
        feedback_corrections = 0
        position_feedback = bool((visual_feedback or {}).get("position_feedback_enabled", False))
        axial_refinement_mode = (visual_feedback or {}).get("axial_refinement_mode", "hold")
        axis_budget_mode = (visual_feedback or {}).get("axis_budget_mode", "path_sum")
        pose_correction_gain = float((visual_feedback or {}).get("pose_correction_gain", 1.))
        axis_budget_deg = float((visual_feedback or {}).get("maximum_reference_rotation_deg",
            (visual_feedback or {}).get("maximum_cumulative_axis_command_deg", 0.)))
        feedback_translation_total = np.zeros(3)
        reference_rebase = np.zeros(3)
        cumulative_axis_command = float(np.linalg.norm(alignment_rotvec))
        if visual_feedback is not None:
            if (alignment_duration <= 0 or not planar_freeze_after_preparation
                    or not 1 <= int(visual_feedback["maximum_corrections"]) <= (8 if position_feedback else 3)
                    or not 0 < pose_correction_gain <= 1.
                    or not 0 < float(visual_feedback["axis_tolerance_deg"]) <= .1
                    or axis_budget_mode not in ("path_sum", "net_reference_rotation")
                    or not 0 < axis_budget_deg <= 1.):
                raise ValueError("visual feedback requires finite damped corrections and at most a one-degree orientation region")
            record["pre_turn_visual_feedback"] = {
                **visual_feedback, "observations": [], "verified": False,
                "wrist_filter_or_loaded_tare_reset": False,
                "position_travel_origin_reset": False,
                "online_object_or_contact_truth_used": False,
            }
            if position_feedback:
                if settings.get("guided_socket_rotation_pivot", False):
                    raise ValueError("measured centre feedback already defines the rotation pivot")
                lateral_feedback_budget = min(
                    float(visual_feedback["maximum_cumulative_lateral_correction_m"]),
                    float(grip["visual_alignment_allowance_from_quarter_body_clearance_m"]))
                if (not 0 < lateral_feedback_budget <= .001
                        or not 0 < float(visual_feedback["position_tolerance_m"]) <= .00005
                        or axial_refinement_mode not in ("hold", "retract_only", "force_admittance")):
                    raise ValueError("centre feedback requires a current finite geometry allowance and a bounded axial mode")
                record["pre_turn_visual_feedback"].update(
                    lateral_correction_budget_m=lateral_feedback_budget,
                    lateral_budget_source="CURRENT_GRIP_QUARTER_BODY_CLEARANCE_CAPPED_AT_ONE_MM",
                    axial_refinement_mode=axial_refinement_mode,
                    planar_force_offset_held_during_visual_refinement=True,
                    forward_axial_motion_inhibited_during_visual_refinement=axial_refinement_mode != "force_admittance")
        axial_offset = 0.0
        previous_align_fraction = 0.0
        index = 0
        interval_paused=False
        maximum_execution_s=settings.get('maximum_unobserved_execution_s')
        monitor_period=settings.get('preparation_observation_period_s')
        next_monitor_s=float(monitor_period) if monitor_period else float('inf')
        while index < count:
            if stepper.abort_reason is not None:
                raise RuntimeError(f"existing joint/FT protection: {stepper.abort_reason}")
            stop_request = runtime.get("simulation_stop_request_path")
            if (stop_request and index % max(1, round(.1 / dt)) == 0
                    and Path(stop_request).exists()):
                record["explicit_pause_request"] = str(stop_request)
                raise RuntimeError("explicit simulation pause requested; preserve the incomplete episode")
            q, hand, current, wrench, raw_wrench = observe()
            root_moments=None
            if root_observer is not None and (regulate_finger_effort or regulate_preparation_effort):
                root_moments=np.asarray(stepper.latest[2])[8:]-root_observer.tare_reaction-(
                    root_observer._system(q)[2]-root_observer.tare_gravity)
                if motor_force_control.get('enabled',False):
                    if root_moment_filtered is None:root_moment_filtered=root_moments.copy()
                    else:root_moment_filtered+=dt/(float(motor_force_control['signal_filter_time_s'])+dt)*(root_moments-root_moment_filtered)
            elapsed = index * dt
            if progress_enabled and elapsed>=next_observation_s:
                from te_body_socket_observation import observe_tracked_plug_from_rgbd
                world.pause()
                folder=output/f'turn_observation_{int(stepper.step_index):08d}'
                try:
                    observed=observe_tracked_plug_from_rgbd(repository,omni.usd.get_context().get_stage(),
                        world,rep,hand,folder,tracking_context)
                except (RuntimeError,ValueError) as tracking_error:
                    observed=full_observe_released_plug(repository,omni.usd.get_context().get_stage(),
                        world,rep,hand,output/f'turn_reacquire_{int(stepper.step_index):08d}')
                    if not observed.get('position_and_axis_measured'):
                        raise RuntimeError('recoverable nut turn stop: VISUAL_TRACKING_LOST') from tracking_error
                    tracking_context.update(last_observation=observed)
                    tracking_context.pop('mask_path',None)
                observed_body=np.asarray(observed['world_from_plug_five_dof'])
                if -axis@observed_body[:3,2]<=0:
                    raise RuntimeError('recoverable nut turn stop: VISUAL_TRACKING_LOST')
                from te_nut_motion import grasp_relation_residual
                relation_error=observed_body[:3,3]-current[:3,3]
                relation_components=grasp_relation_residual(relation_error,axis,
                    progress_settings.get('captive_nut_axial_travel_m',0.))
                if relation_components['unexplained_norm_m']>float(progress_settings['maximum_grasp_relation_error_m']):
                    record['grasp_relation_stop']={'step':int(stepper.step_index),'error_world_m':relation_error.tolist(),
                        'relation_components':relation_components,'current_observation':observed}
                    raise RuntimeError('recoverable nut turn stop: GRASP_RELATION_CHANGED')
                # Observe the changed relation without redefining the desired
                # grasp to that error. Otherwise a fresh image would cancel
                # the measured tracking correction and accept a displaced hand.
                last_observed_body=observed_body.copy()
                record['observed_hand_from_body']=(np.linalg.inv(hand)@observed_body).tolist()
                current_command=engagement_start_deg+abs(float(record.get('last_loaded_command_deg',0.)))
                observation_row={'step':int(stepper.step_index),'time_s':float(world.current_time),
                    'depth_m':float(-axis@(observed_body[:3,3]-socket[:3,3])),
                    'command_deg':current_command,'torsion_nm':float(wrench[5]),
                    'grasp_relation_error_m':relation_components['unexplained_norm_m'],
                    'grasp_relation_components':relation_components,
                    'relation_update_world_m':relation_error.tolist(),'observation':observed}
                previous=runtime.get('last_nut_progress_observation')
                observation_row['state']=classify_observed_progress(previous,observation_row,progress_settings,
                    progress_anchor=record['visual_progress_anchor'])
                runtime['last_nut_progress_observation']=observation_row
                record['current_visual_progress'].append(observation_row)
                seating_count=seating_count+1 if observation_row['state']=='VISUAL_SEATING_CANDIDATE' else 0
                next_observation_s=elapsed+float(progress_settings['observation_period_s'])
                if seating_count>=int(progress_settings.get('seating_confirmations',2)):
                    record['seating_candidate']=True
                    record['normal_stop_reason']='CURRENT_VISUAL_DEPTH_STABILITY_AND_TORQUE'
                    break
                if observation_row['state'] in ('RECOVERABLE_NO_PROGRESS','RECOVERABLE_GRIP_SLIP','RECOVERABLE_CONTACT_STALL'):
                    reason={'RECOVERABLE_NO_PROGRESS':'NO_OBSERVED_AXIAL_PROGRESS',
                        'RECOVERABLE_GRIP_SLIP':'OBSERVED_GRIP_SLIP','RECOVERABLE_CONTACT_STALL':'OBSERVED_CONTACT_STALL'}[observation_row['state']]
                    raise RuntimeError('recoverable nut turn stop: '+reason)
                if observation_row['state']=='OBSERVED_DEPTH_OVERRUN':
                    raise RuntimeError('current visual depth exceeds the bounded source seating range')
                world.play()
            if release_reserve is not None and elapsed >= preparation_end:
                release_mechanism=getattr(world,"hand_mechanism",None)
                if release_mechanism is None:
                    raise ValueError("controlled transmission-reserve stop requires the shared hand")
                release_efforts=[];release_margins=[]
                for name,position in zip(("f1j2","f2j1","f3j2"),q[8:]):
                    drive=release_mechanism.drives[name]
                    value=drive.reference.transmission_stiffness*(drive.input_angle-float(position))
                    release_efforts.append(value)
                    release_margins.append(drive.reference.transmission_effort_boundary-abs(value))
                if min(release_margins) <= float(release_reserve):
                    record["transmission_reserve_stop"]={"step":int(stepper.step_index),
                        "modeled_elastic_efforts_nm":release_efforts,"effort_margins_nm":release_margins,
                        "reserve_nm":float(release_reserve),"hard_boundary_violated":min(release_margins)<0.,
                        "action":"CURRENT_VISUAL_GUIDE_CHECK_THEN_CONTROLLED_RELEASE; NOT_A_SEATING_CLAIM"}
                    raise RuntimeError("bounded thread pilot stop: TRANSMISSION_RESERVE")
            if (rate_guard_enabled and rate_turn_finished_s is not None
                    and elapsed-rate_turn_finished_s >= hold_s):
                rate_cycle_finished = True
                break
            if maximum_execution_s is not None and elapsed>=float(maximum_execution_s)-dt/10:
                interval_paused=True
                break
            if elapsed>=next_monitor_s and elapsed<preparation_end:
                world.pause();before_time=float(world.current_time)
                monitored=observe_released_plug_from_rgbd(repository,omni.usd.get_context().get_stage(),world,rep,hand,
                    output/f'preparation_observation_{int(round(elapsed/dt)):06d}')
                if float(world.current_time)!=before_time or not monitored.get('position_and_axis_measured'):
                    raise RuntimeError('preparation observation invalid or advanced physics')
                record.setdefault('periodic_preparation_observations',[]).append({'step':int(stepper.step_index),'observation':monitored})
                next_monitor_s=elapsed+float(monitor_period);save();world.play()
            if visual_feedback is not None and not feedback_verified and elapsed >= preparation_end:
                world.pause()
                before_time = float(world.current_time)
                observed = observe_released_plug_from_rgbd(
                    repository, omni.usd.get_context().get_stage(), world, rep, hand,
                    output / f"alignment_check_{feedback_corrections:02d}")
                if float(world.current_time) != before_time:
                    raise RuntimeError("alignment RGBD observation advanced physics")
                if not observed.get("position_and_axis_measured"):
                    raise RuntimeError("alignment feedback has no measured connector axis")
                measured_body = np.asarray(observed["world_from_plug_five_dof"]).reshape(4, 4)
                measured_axis = measured_body[:3, 2]
                measured_axis /= np.linalg.norm(measured_axis)
                cross = np.cross(measured_axis, -axis)
                sine = float(np.linalg.norm(cross))
                residual = float(np.arctan2(sine, measured_axis @ -axis))
                residual_vector = cross / sine * residual if sine > 1e-12 else np.zeros(3)
                lateral_error = float(np.linalg.norm(
                    (np.eye(3) - np.outer(axis, axis)) @ (measured_body[:3, 3] - socket[:3, 3])))
                observation_record = {
                    "step": int(stepper.step_index), "elapsed_s": elapsed,
                    "axis_error_deg": float(np.degrees(residual)),
                    "lateral_error_m": lateral_error,
                    "observation": observed,
                    "cumulative_axis_command_before_deg": float(np.degrees(cumulative_axis_command)),
                }
                record["pre_turn_visual_feedback"]["observations"].append(observation_record)
                feature_fit = (observed_guided_feature_fit(measured_body, socket, feature_fit_settings)
                               if feature_fit_settings is not None else None)
                if feature_fit is not None:
                    observation_record["guided_feature_fit"] = feature_fit
                if position_feedback:
                    # A measured Body centre replaces the old rigid-grip
                    # estimate. Reparameterize the virtual pivot while leaving
                    # the desired physical hand pose exactly unchanged.
                    previous_hand_from_pivot = hand_from_pivot.copy()
                    desired_hand_before = desired_pivot @ np.linalg.inv(previous_hand_from_pivot)
                    delta = measured_body[:3, 3] - current[:3, 3]
                    if np.linalg.norm(delta) > lateral_feedback_budget:
                        save()
                        raise RuntimeError("observed grip-frame shift exceeds the current grasp geometry allowance")
                    hand_from_pivot[:3, 3] += hand[:3, :3].T @ delta
                    updated_desired = desired_hand_before @ hand_from_pivot
                    reference_rebase += updated_desired[:3, 3] - desired_pivot[:3, 3]
                    desired_pivot = updated_desired
                    invariant_error = float(np.max(np.abs(
                        desired_pivot @ np.linalg.inv(hand_from_pivot) - desired_hand_before)))
                    if invariant_error > 1e-10:
                        raise RuntimeError("visual grip-frame update changed the physical hand target")
                    current = hand @ hand_from_pivot
                    com_from_pivot = current[:3, :3].T @ (
                        measured_body[:3, :3] @ np.asarray(settings["payload_com_body_m"]))
                    q, hand, current, wrench, raw_wrench = observe()
                    observation_record.update(
                        measured_grip_frame_translation_update_world_m=delta.tolist(),
                        updated_hand_from_virtual_pivot=hand_from_pivot.tolist(),
                        physical_hand_target_invariance_error=invariant_error,
                        original_wrench_filter_origin_world_m=fixed_wrench_origin_world.tolist(),
                        loaded_wrench_rezeroed=False,scene_pose_written=False)
                # Recalibrate only the observed Body-axis relation for geometry
                # evaluation. The force filter, travel origin and physical
                # object poses are never reset by a new camera observation.
                body_axis_rotation = original_pivot[:3, :3] @ current[:3, :3].T @ measured_body[:3, :3]
                if lateral_error > (lateral_feedback_budget if position_feedback else
                                    float(alignment["maximum_lateral_correction_m"])):
                    save()
                    raise RuntimeError("measured connector centre exceeds the existing visual correction region")
                pose_accepted = (feature_fit["accepted"] if feature_fit is not None else
                    residual <= np.deg2rad(float(visual_feedback["axis_tolerance_deg"]))
                    and (not position_feedback or lateral_error <= float(visual_feedback["position_tolerance_m"])))
                if pose_accepted:
                    feedback_verified = True
                    record["pre_turn_visual_feedback"]["verified"] = True
                else:
                    if feedback_corrections >= int(visual_feedback["maximum_corrections"]):
                        save()
                        raise RuntimeError("measured connector pose did not converge within the bounded visual corrections")
                    correction_rotvec = pose_correction_gain * residual_vector
                    cumulative_axis_command += pose_correction_gain * residual
                    proposed_reference_rotation = Rotation.from_rotvec(correction_rotvec).as_matrix() @ desired_pivot[:3, :3]
                    net_reference_angle = float(Rotation.from_matrix(
                        proposed_reference_rotation @ original_pivot[:3, :3].T).magnitude())
                    budget_used = net_reference_angle if axis_budget_mode == "net_reference_rotation" else cumulative_axis_command
                    observation_record.update(axis_budget_mode=axis_budget_mode,
                        proposed_net_reference_rotation_deg=float(np.degrees(net_reference_angle)),
                        axis_budget_used_deg=float(np.degrees(budget_used)))
                    if budget_used > np.deg2rad(axis_budget_deg):
                        save()
                        raise RuntimeError("measured residual requires more than the bounded total axis correction")
                    correction_s = max(1., 1.875 * residual / float(alignment["maximum_axis_speed_rad_s"]))
                    translation_delta = np.zeros(3)
                    translation_start = feedback_translation_total.copy()
                    if position_feedback:
                        full_translation_delta = (np.eye(3)-np.outer(axis, axis)) @ (
                            socket[:3, 3]-measured_body[:3, 3])
                        translation_delta = pose_correction_gain * full_translation_delta
                        translation_delta *= min(1., float(alignment["maximum_lateral_correction_m"])
                            / max(float(np.linalg.norm(translation_delta)), 1e-15))
                        proposed = (lateral_delta + feedback_translation_total + translation_delta
                                    + socket[:3, :2] @ planar_offset)
                        if np.linalg.norm(proposed) > lateral_feedback_budget:
                            save()
                            raise RuntimeError("current geometry allowance bounds the total lateral correction")
                        # Retain the duration for the full residual while
                        # applying only the damped correction. Its peak speed
                        # and acceleration decrease rather than increase.
                        correction_s = max(correction_s, 1.875*np.linalg.norm(full_translation_delta)
                            / float(alignment["maximum_lateral_speed_m_s"]))
                        feedback_translation_total += translation_delta
                    feedback_segment = {
                        "start_s": elapsed, "duration_s": correction_s,
                        "start_rotation": desired_pivot[:3, :3].copy(),
                        "rotvec": correction_rotvec, "initial_yaw_offset": yaw_offset,
                        "translation_start": translation_start,
                        "translation_delta": translation_delta,
                    }
                    preparation_end = elapsed + correction_s + settle_s
                    count = round((preparation_end + duration + hold_s + rate_extra_time) / dt)
                    feedback_corrections += 1
                    observation_record.update(
                        correction_world_rotvec_rad=correction_rotvec.tolist(),
                        pose_correction_gain=pose_correction_gain,
                        correction_translation_world_m=translation_delta.tolist(),
                        correction_duration_s=correction_s,
                        cumulative_axis_command_after_deg=float(np.degrees(cumulative_axis_command)))
                print("NUT_ALIGNMENT_FEEDBACK", json.dumps({k: v for k, v in observation_record.items() if k != "observation"}), flush=True)
                save()
                world.play()
            if (settings.get("guided_socket_rotation_pivot", False)
                    and elapsed >= preparation_end
                    and "guided_socket_rotation_pivot" not in record):
                if not planar_freeze_after_preparation or alignment_duration <= 0:
                    raise ValueError("guided pivot requires completed visual/force preparation and a retained lateral reference")
                delta = (np.eye(3)-np.outer(axis, axis)) @ (socket[:3, 3]-current[:3, 3])
                if np.linalg.norm(delta) > float(alignment["maximum_lateral_correction_m"]):
                    raise RuntimeError("guided pivot shift exceeds the existing visual correction region")
                before_com = current[:3, 3] + current[:3, :3] @ com_from_pivot
                previous = current.copy()
                # Reparameterize the robot's rotation reference, never a scene
                # pose. A seated guide supplies the already observed socket axis.
                # Shift the payload offset inversely, preserving modeled gravity
                # and the causal wrench filter's fixed origin without rezeroing.
                hand_from_pivot[:3, 3] += hand[:3, :3].T @ delta
                com_from_pivot -= current[:3, :3].T @ delta
                original_pivot[:3, 3] += delta
                current = hand @ hand_from_pivot
                shift_socket = socket[:3, :3].T @ delta
                wrench[3:] -= np.cross(shift_socket, wrench[:3])
                raw_wrench[3:] -= np.cross(shift_socket, raw_wrench[:3])
                record["guided_socket_rotation_pivot"] = {
                    "step": int(stepper.step_index), "elapsed_s": elapsed,
                    "source": "EXISTING_WRIST_RGBD_SOCKET_AXIS_AFTER_GUIDED_FORCE_PREPARATION_AND_CURRENT_ENCODERS",
                    "previous_virtual_pivot_world_m": previous[:3, 3].tolist(),
                    "current_virtual_pivot_world_m": current[:3, 3].tolist(),
                    "reference_shift_world_m": delta.tolist(),
                    "updated_hand_from_virtual_axis": hand_from_pivot.tolist(),
                    "modeled_payload_com_change_m": float(np.linalg.norm(
                        current[:3, 3]+current[:3, :3]@com_from_pivot-before_com)),
                    "wrench_filter_origin_world_m": fixed_wrench_origin_world.tolist(),
                    "loaded_wrench_rezeroed": False, "body_key_yaw_used": False,
                    "object_or_contact_truth_used": False, "scene_pose_write": False}
            align_u = np.clip(elapsed / alignment_duration, 0., 1.) if alignment_duration > 0 else 1.0
            align_fraction = 10*align_u**3 - 15*align_u**4 + 6*align_u**5
            align_rate = 30*align_u**2*(1-align_u)**2/alignment_duration if alignment_duration > 0 else 0.0
            effective_turn_time = max(0., elapsed-preparation_end-rate_lost_time)
            if rate_guard_enabled and elapsed >= preparation_end:
                model_elastic_efforts = np.array([
                    rate_mechanism.drives[n].reference.transmission_stiffness
                    * (rate_mechanism.drives[n].input_angle-float(position))
                    for n, position in zip(rate_names, q[8:])])
                model_effort_margin = float(np.min(rate_caps-np.abs(model_elastic_efforts)))
                turn_rate_scale = float(np.clip((model_effort_margin-rate_stop_margin)
                    / (rate_slow_margin-rate_stop_margin), 0., 1.))
                if turn_rate_scale < .05:
                    rate_near_hold_time += dt
                elif turn_rate_scale > .1:
                    rate_near_hold_time = 0.
                if rate_near_hold_time > rate_hold_limit:
                    record["effort_margin_hold_stop"] = {
                        "step": int(stepper.step_index), "elapsed_s": elapsed,
                        "modeled_elastic_efforts_nm": model_elastic_efforts.tolist(),
                        "minimum_effort_margin_nm": model_effort_margin,
                        "near_hold_duration_s": rate_near_hold_time}
                    raise RuntimeError("finger effort margin did not recover within the bounded rotation hold")
            u = np.clip(effective_turn_time / duration, 0.0, 1.0)
            fraction = 10*u**3 - 15*u**4 + 6*u**5
            rotation_velocity = angle * 30*u**2*(1-u)**2 / duration * turn_rate_scale
            if scalar_profile is not None:
                displacement,profile_speed=scalar_profile.at(effective_turn_time)
                fraction=displacement/angle if abs(angle)>1e-15 else 1.
                rotation_velocity=profile_speed*turn_rate_scale
            record['last_loaded_command_deg']=float(np.rad2deg(angle*fraction))
            if rate_guard_enabled and u >= 1. and rate_turn_finished_s is None:
                rate_turn_finished_s = elapsed
            yaw_compliance_velocity = 0.0
            if torsion_enabled and elapsed < preparation_end:
                # The online circle estimator measures centre and axis only.
                # Yield the otherwise frozen hand/nut yaw toward the measured
                # reaction torque, then retain this bounded offset for turning.
                yaw_compliance_velocity = float(np.clip(
                    yaw_admittance*wrench[5], -maximum_yaw_speed, maximum_yaw_speed))
                yaw_offset += yaw_compliance_velocity*dt
                if abs(yaw_offset) > maximum_yaw_offset:
                    raise RuntimeError("bounded pre-turn yaw-compliance travel exhausted")
            force_u = float(np.clip((elapsed-preparation_end)/force_reference_ramp_s, 0., 1.))
            if observation_session is not None and observation_session.get('turn_start_step') is not None:
                force_u=float(np.clip((int(stepper.step_index)-observation_session['turn_start_step'])*dt/force_reference_ramp_s,0.,1.))
                preparation_force_reference=float(observation_session['initial_turn_force_reference_n'])
            force_fraction = 10*force_u**3 - 15*force_u**4 + 6*force_u**5
            force_reference = preparation_force_reference + force_fraction*(
                turn_force_reference-preparation_force_reference)
            engagement_reference = 0.
            engagement_progress_deg = engagement_start_deg + abs(float(np.degrees(angle*fraction)))
            if engagement_enabled:
                fade = float(np.clip((engagement_progress_deg
                    - (engagement_end_deg-engagement_fade_deg))/engagement_fade_deg, 0., 1.))
                engagement_reference = engagement_force*force_fraction*(
                    1. - (10*fade**3-15*fade**4+6*fade**5))
                force_reference += engagement_reference
            if observation_session is not None:observation_session['last_force_reference_n']=float(force_reference)
            velocity_z = np.clip(
                (wrench[2] - force_reference)
                * float(settings["axial_admittance_m_per_n_s"]),
                -float(settings["maximum_axial_speed_m_s"]), float(settings["maximum_axial_speed_m_s"]))
            if axial_lead_enabled and elapsed>=preparation_end:
                if turn_axial_origin is None:turn_axial_origin=axial_offset
                lead=float(axial_lead['lead_m'])
                correction_velocity=(wrench[2]-force_reference)*float(settings['axial_admittance_m_per_n_s'])
                correction_velocity-=float(axial_lead['restoring_rate_s_inv'])*axial_correction
                bound=float(axial_lead['maximum_correction_m'])
                next_correction=float(np.clip(axial_correction+correction_velocity*dt,-bound,bound))
                nominal_axial=turn_axial_origin+lead*angle*fraction/(2*np.pi)
                velocity_z=float(np.clip((nominal_axial+next_correction-axial_offset)/dt,
                    -float(settings['maximum_axial_speed_m_s']),float(settings['maximum_axial_speed_m_s'])))
            visual_refinement_active = position_feedback and feedback_segment is not None and not feedback_verified
            unconstrained_axial_velocity_z = float(velocity_z)
            if visual_refinement_active:
                # Keep the failed hold/one-way cases reproducible. The current
                # centring mode retains the original bilateral force response
                # so both compressive and tensile preload can be relieved.
                if axial_refinement_mode == "hold":
                    velocity_z = 0.
                elif axial_refinement_mode == "retract_only":
                    velocity_z = max(0., velocity_z)
            axial_reference_held = bool(visual_refinement_active and velocity_z == 0.)
            axial_offset += velocity_z * dt
            if axial_lead_enabled and turn_axial_origin is not None:
                axial_correction=axial_offset-nominal_axial
            planar_velocity = np.zeros(2)
            planar_restoring_force = np.zeros(2)
            combined_alignment_velocity_world = None
            planar_frozen = planar_freeze_after_preparation and (
                elapsed >= preparation_end or (position_feedback and feedback_segment is not None))
            if planar_enabled and planar_frozen and "held_preparation_reference" not in record["planar_force_admittance"]:
                record["planar_force_admittance"]["held_preparation_reference"] = {
                    "elapsed_s": elapsed, "step": int(stepper.step_index),
                    "offset_socket_xy_m": planar_offset.tolist(),
                    "source": "COMPLETED_VISUAL_ALIGNMENT_AND_BOUNDED_WRIST_FORCE_PREPARATION",
                    "force_stops_remain_active": True,
                    "object_or_contact_truth_used": False}
            if planar_enabled and elapsed >= planar_start_s and not planar_frozen:
                planar_restoring_force = -planar_restoring_stiffness*planar_offset
                planar_velocity = planar_gain*(wrench[:2] + planar_restoring_force)
                planar_velocity *= min(1., maximum_planar_speed/max(
                    float(np.linalg.norm(planar_velocity)), 1e-15))
                if planar_during_alignment and elapsed < alignment_duration:
                    # Both terms now act during alignment. Limit their sum,
                    # including the actual discrete visual-target increment,
                    # so early force relief does not increase reference speed.
                    visual_step_velocity = lateral_delta*(align_fraction-previous_align_fraction)/dt
                    combined_alignment_velocity_world = (
                        visual_step_velocity + socket[:3, :2] @ planar_velocity)
                    combined_limit = min(maximum_planar_speed,
                        float(alignment["maximum_lateral_speed_m_s"]))
                    combined_alignment_velocity_world *= min(1., combined_limit/max(
                        float(np.linalg.norm(combined_alignment_velocity_world)), 1e-15))
                    planar_velocity = socket[:3, :2].T @ (
                        combined_alignment_velocity_world - visual_step_velocity)
                    record["planar_force_admittance"]["combined_reference_speed_cap_m_s"] = combined_limit
                planar_offset += planar_velocity*dt
                if np.linalg.norm(planar_offset) > maximum_planar_offset:
                    record["planar_force_travel_stop"] = {
                        "step": int(stepper.step_index), "elapsed_s": elapsed,
                        "attempted_offset_socket_m": planar_offset.tolist(),
                        "maximum_offset_m": maximum_planar_offset,
                        "filtered_force_socket_xy_n": wrench[:2].tolist(),
                    }
                    raise RuntimeError("bounded planar force-admittance travel exhausted")
            previous_align_fraction = align_fraction
            desired_pivot[:3, 3] = (original_pivot[:3, 3] + align_fraction*lateral_delta
                + axis*axial_offset + socket[:3, :2] @ planar_offset + reference_rebase)
            yaw_rotation = Rotation.from_rotvec(axis * (angle*fraction + yaw_offset)).as_matrix()
            desired_pivot[:3, :3] = (yaw_rotation
                @ Rotation.from_rotvec(alignment_rotvec * align_fraction).as_matrix()
                @ original_pivot[:3, :3])
            alignment_angular_velocity_world = yaw_rotation @ (alignment_rotvec * align_rate)
            feedback_translation_velocity_world = np.zeros(3)
            if feedback_segment is not None:
                f = feedback_segment
                fu = float(np.clip((elapsed-f["start_s"])/f["duration_s"], 0., 1.))
                blend = 10*fu**3 - 15*fu**4 + 6*fu**5
                rate = 30*fu**2*(1-fu)**2/f["duration_s"]
                yaw_after_observation = Rotation.from_rotvec(
                    axis * (angle*fraction + yaw_offset-f["initial_yaw_offset"])).as_matrix()
                desired_pivot[:3, :3] = (yaw_after_observation
                    @ Rotation.from_rotvec(f["rotvec"]*blend).as_matrix() @ f["start_rotation"])
                alignment_angular_velocity_world = yaw_after_observation @ (f["rotvec"]*rate)
                if position_feedback:
                    desired_pivot[:3, 3] += f["translation_start"] + f["translation_delta"]*blend
                    feedback_translation_velocity_world = f["translation_delta"]*rate
            if axis_budget_mode == "net_reference_rotation" and elapsed < preparation_end:
                actual_reference_angle = float(Rotation.from_matrix(
                    desired_pivot[:3, :3] @ original_pivot[:3, :3].T).magnitude())
                record["pre_turn_visual_feedback"]["maximum_net_reference_rotation_deg"] = max(
                    record["pre_turn_visual_feedback"].get("maximum_net_reference_rotation_deg", 0.),
                    float(np.degrees(actual_reference_angle)))
                if actual_reference_angle > np.deg2rad(axis_budget_deg) + 1e-10:
                    raise RuntimeError("actual preparation reference exceeds the bounded orientation region")
            actual_depth_change = -float(axis @ (current[:3, 3] - original_pivot[:3, 3]))
            follow_velocity=np.zeros(3);follow_angular_velocity=np.zeros(3);follow_rotation=np.eye(3)
            if visual_follow_enabled and elapsed>=preparation_end:
                # Move the world reference using independently observed Body
                # motion. Keep hand_from_pivot fixed: measured hand errors must
                # not redefine the desired grip or disappear on observation.
                target=last_observed_body[:3,3]-desired_pivot[:3,3]
                target-=axis*(axis@target)
                expected_axis=desired_pivot[:3,:3]@original_pivot[:3,:3].T@body_axis_rotation[:,2]
                observed_axis=last_observed_body[:3,2]
                cross=np.cross(expected_axis,observed_axis);sine=float(np.linalg.norm(cross))
                theta=float(np.arctan2(sine,expected_axis@observed_axis))
                rotation_target=cross*(theta/sine) if sine>1e-12 else np.zeros(3)
                if (np.linalg.norm(target)>float(visual_follow['maximum_lateral_motion_m'])
                        or theta>np.deg2rad(float(visual_follow['maximum_axis_motion_deg']))):
                    raise RuntimeError('recoverable nut turn stop: VISUAL_TRACKING_LOST')
                tau=float(visual_follow['smoothing_time_s'])
                follow_velocity=(target-visual_follow_translation)/(tau+dt)
                follow_velocity*=min(1.,float(visual_follow['maximum_lateral_speed_m_s'])/max(float(np.linalg.norm(follow_velocity)),1e-15))
                rotation_velocity_follow=(rotation_target-visual_follow_rotvec)/(tau+dt)
                rotation_velocity_follow*=min(1.,float(visual_follow['maximum_axis_speed_rad_s'])/max(float(np.linalg.norm(rotation_velocity_follow)),1e-15))
                old_rotation=Rotation.from_rotvec(visual_follow_rotvec).as_matrix()
                visual_follow_translation+=follow_velocity*dt
                visual_follow_rotvec+=rotation_velocity_follow*dt
                follow_rotation=Rotation.from_rotvec(visual_follow_rotvec).as_matrix()
                follow_angular_velocity=Rotation.from_matrix(follow_rotation@old_rotation.T).as_rotvec()/dt
                desired_pivot[:3,3]+=visual_follow_translation
                desired_pivot[:3,:3]=follow_rotation@desired_pivot[:3,:3]
            commanded_depth_change = -float(axis @ (desired_pivot[:3, 3] - original_pivot[:3, 3]))
            if observation_session is not None:
                origin=np.asarray(observation_session['original_body_origin_world_m'])
                lateral_budget=float(observation_session['lateral_budget_m'])
                for point in (current[:3,3],desired_pivot[:3,3]):
                    delta=point-origin
                    if abs(axis@delta)>float(settings['maximum_axial_travel_m']):
                        raise RuntimeError('bounded observed-turn global axial travel exhausted')
                    if np.linalg.norm(delta-axis*(axis@delta))>lateral_budget:
                        raise RuntimeError('bounded observed-turn global lateral region exhausted')
                nominal_angle=float(observation_session.get('budget_nominal_twist_rad',0.))+angle*fraction+yaw_offset
                relative=Rotation.from_rotvec(-axis*nominal_angle).as_matrix()@desired_pivot[:3,:3]@np.asarray(observation_session['original_hand_rotation_world']).T
                if Rotation.from_matrix(relative).magnitude()>np.deg2rad(1.)+1e-8:
                    raise RuntimeError('bounded observed-turn global orientation correction exhausted')
            # All-yaw front-edge depth bound from the source key radius/length,
            # the current palm axis and encoder translation. Preserve a small
            # positive guide overlap while allowing force-controlled retreat.
            predicted_body_rotation = current[:3, :3] @ original_pivot[:3, :3].T @ body_axis_rotation
            geometry_origin=current[:3,3]
            if progress_enabled and last_observed_body is not None:
                predicted_body_rotation=last_observed_body[:3,:3]
                geometry_origin=last_observed_body[:3,3]
            axis_cosine = abs(float(axis @ predicted_body_rotation[:, 2]))
            estimated_key_depth = (-float(axis @ (geometry_origin - socket[:3, 3]))
                - .000762 * axis_cosine - .0188214 * np.sqrt(max(0., 1.-axis_cosine**2)))
            estimated_key_rear_depth = (-float(axis @ (geometry_origin - socket[:3, 3]))
                - .007645401*axis_cosine - .018821401*np.sqrt(max(0., 1.-axis_cosine**2)))
            balance_target = None
            if balance_enabled and elapsed >= preparation_end:
                if balance_reference is None:
                    balance_reference = hand_target.copy()
                    levers = []
                    for row, normal, joint in zip(balance_rows, normals_hand, ("f1j2", "f2j1", "f3j2")):
                        point = balance_points_local[row["link"]]
                        J = inputs.robot_model.geometric_jacobian(row["link"], tuple(q),
                            point_local_m=point, enforce_limits=False)
                        col = inputs.robot_model.independent_joint_names.index(joint)
                        levers.append(-float((hand[:3, :3] @ normal) @ J[:3, col]))
                    balance_levers = np.asarray(levers)
                    if np.any(balance_levers <= .01) or np.any(balance_levers >= .5):
                        raise RuntimeError("CAD closing-to-normal Jacobian has an unsupported sign or scale")
                    record["grip_lateral_balance"].update(
                        first_step=int(stepper.step_index),
                        prepared_hand_target_rad=balance_reference.tolist(),
                        closing_normal_levers_m=balance_levers.tolist())
                normal_axes = socket[:3, :3].T @ hand[:3, :3] @ normals_hand.T
                allocation = np.vstack((normal_axes[:2], np.ones(3)))
                if np.linalg.cond(allocation) > 100.:
                    raise RuntimeError("current CAD normal directions do not support bounded transverse redistribution")
                change = np.linalg.solve(allocation, np.r_[-wrench[:2], 0.])
                balance_offset_n += balance_rate * change * dt
                balance_offset_n -= balance_offset_n.mean()
                balance_offset_n *= min(1., balance_bound/max(np.max(np.abs(balance_offset_n)), 1e-15))
                # Positive closing motion raises outward pad normal reaction.
                # This uses the existing finite native spring, not an object force.
                balance_target = balance_reference.copy()
                balance_target[1:] += balance_levers/balance_stiffness*balance_offset_n
                balance_target = np.clip(balance_target, lower_hand, upper_hand)
            control_sample = _json_ready({"step": int(stepper.step_index), "elapsed_s": elapsed,
                "commanded_rotation_deg": float(np.rad2deg(angle * fraction)),
                "turn_reference_rate_scale": turn_rate_scale,
                "turn_reference_time_lost_s": rate_lost_time,
                "modeled_elastic_efforts_for_turn_rate_nm": model_elastic_efforts.tolist() if model_elastic_efforts is not None else None,
                "minimum_modeled_effort_margin_nm": model_effort_margin,
                "axial_force_reference_n": float(force_reference),
                "engagement_additional_downward_reference_n": float(engagement_reference),
                "engagement_cumulative_loaded_turn_command_deg": engagement_progress_deg if engagement_enabled else None,
                "pre_turn_yaw_correction_deg": float(np.rad2deg(yaw_offset)),
                "pre_turn_yaw_velocity_deg_s": float(np.rad2deg(yaw_compliance_velocity)),
                "planar_force_offset_socket_m": planar_offset.tolist(),
                "planar_force_velocity_socket_m_s": planar_velocity.tolist(),
                "virtual_planar_restoring_force_socket_n": planar_restoring_force.tolist(),
                "grip_model_normal_redistribution_n": balance_offset_n.tolist() if balance_enabled else None,
                "grip_balance_target_rad": balance_target.tolist() if balance_target is not None else None,
                "combined_alignment_reference_velocity_world_m_s": (
                    combined_alignment_velocity_world.tolist()
                    if combined_alignment_velocity_world is not None else None),
                "contact_wrench_at_virtual_plug_origin": wrench.tolist(),
                "raw_contact_wrench_at_virtual_plug_origin": raw_wrench.tolist(),
                "alignment_fraction": float(align_fraction),
                "preparation_end_elapsed_s": preparation_end,
                "visual_feedback_corrections": feedback_corrections,
                "axial_reference_held_for_visual_centring": axial_reference_held,
                "forward_axial_motion_inhibited_for_visual_centring": visual_refinement_active and axial_refinement_mode != "force_admittance",
                "unconstrained_axial_velocity_command_m_s": unconstrained_axial_velocity_z,
                "visual_feedback_translation_velocity_world_m_s": feedback_translation_velocity_world.tolist(),
                "visual_encoder_lateral_error_m": float(np.linalg.norm(
                    (np.eye(3)-np.outer(axis, axis)) @ (current[:3, 3]-socket[:3, 3]))),
                "visual_encoder_axis_error_deg": float(np.rad2deg(np.arccos(np.clip(axis_cosine, -1, 1)))),
                "encoder_axial_progress_m": actual_depth_change,
                "commanded_axial_progress_m": commanded_depth_change,
                "estimated_minimum_key_front_depth_m": estimated_key_depth,
                "estimated_minimum_key_rear_depth_m": estimated_key_rear_depth,
                "axial_velocity_command_m_s": float(velocity_z),
                "encoder_pivot_world_m": current[:3, 3].tolist()})
            control_sample.update(axial_lead_following_enabled=axial_lead_enabled,
                bounded_axial_correction_m=axial_correction if axial_lead_enabled else None,
                visual_reference_translation_m=visual_follow_translation.tolist(),
                visual_reference_rotation_rad=visual_follow_rotvec.tolist(),
                measured_base_moments_nm=None if root_moments is None else root_moments.tolist(),
                desired_pivot_world_m=desired_pivot[:3,3].tolist())
            sample_stream.write(encode_row(control_sample) + "\n")
            record["sample_count"] += 1
            limits = settings["stops"]
            # A null axial entry retires only the additional probe-style stop.
            # The original stepper wrist/finger protection still runs before
            # and after every command; force reference and travel stay bounded.
            for hit, reason in (
                (limits.get("axial_force_n") is not None
                 and abs(wrench[2]) > limits["axial_force_n"], "AXIAL_CONTACT_FORCE"),
                (np.linalg.norm(wrench[:2]) > limits["lateral_force_n"], "LATERAL_CONTACT_FORCE"),
                (np.linalg.norm(wrench[3:5]) > limits["bending_moment_nm"], "BENDING_CONTACT_MOMENT"),
                (abs(wrench[5]) > limits["torsional_moment_nm"], "TORSIONAL_CONTACT_MOMENT"),
                (max(abs(actual_depth_change), abs(commanded_depth_change)) > settings["maximum_axial_travel_m"], "AXIAL_TRAVEL"),
                (estimated_key_depth < .0001, "VISUAL_ENCODER_KEY_GUIDE_OVERLAP"),
                (settings.get("minimum_key_rear_overlap_m") is not None
                 and estimated_key_rear_depth < float(settings["minimum_key_rear_overlap_m"]),
                 "VISUAL_ENCODER_COMPLETE_KEY_OVERLAP"),
            ):
                if hit:
                    raise RuntimeError(f"bounded thread pilot stop: {reason}")
            if index % round(2.0 / dt) == 0:
                print("NUT_ROTATION", json.dumps(control_sample), flush=True)
                (output / "nut_rotation_progress.json").write_text(json.dumps({
                    "stage": record["stage"], "sample_count": record["sample_count"],
                    "latest_control_sample": control_sample}, ensure_ascii=False) + "\n")
            ik_reference = settings.get("arm_kinematic_reference", "measured_pose")
            if ik_reference not in ("measured_pose", "commanded_pose"):
                raise ValueError("unsupported arm kinematic integration reference")
            if ik_reference == "commanded_pose":
                # Integrate the kinematic trajectory from its own nominal
                # configuration. Feeding the loaded encoder deflection back
                # into this integrator would continuously wind up the native
                # PD position target against a constrained connector. Actual
                # encoder motion still drives force/travel/geometry checks.
                ik_q = np.r_[arm, q[7:]]
                ik_hand = np.asarray(inputs.robot_model.forward_kinematics(
                    tuple(ik_q), enforce_limits=False)["handbase_link"])
                ik_pivot = ik_hand @ hand_from_pivot
            else:
                ik_q, ik_hand, ik_pivot = q, hand, current
            jacobian = np.asarray(inputs.robot_model.geometric_jacobian("handbase_link", tuple(ik_q)))[:, :7]
            radius = ik_pivot[:3, 3] - ik_hand[:3, 3]
            skew = np.asarray([[0., -radius[2], radius[1]], [radius[2], 0., -radius[0]], [-radius[1], radius[0], 0.]])
            point_jacobian = np.vstack((jacobian[:3] - skew @ jacobian[3:], .05 * jacobian[3:]))
            error = np.r_[desired_pivot[:3, 3] - ik_pivot[:3, 3],
                .05 * Rotation.from_matrix(desired_pivot[:3, :3] @ ik_pivot[:3, :3].T).as_rotvec()]
            if tracking_enabled:
                # Keep the nominal joint integrator; feedback uses the real
                # encoder pose, without resetting that integrator to loaded q.
                physical_error=desired_pivot[:3,3]-current[:3,3]
                lateral_error=physical_error-axis*(axis@physical_error)
                if elapsed>=preparation_end and np.linalg.norm(lateral_error)>float(measured_tracking['maximum_lateral_error_m']):
                    record['tracking_stop']={'step':int(stepper.step_index),'error_world_m':physical_error.tolist()}
                    raise RuntimeError('recoverable nut turn stop: LOADED_POSE_ERROR')
                error=np.r_[physical_error,.05*Rotation.from_matrix(desired_pivot[:3,:3]@current[:3,:3].T).as_rotvec()]
            lateral_feedforward = (combined_alignment_velocity_world
                if combined_alignment_velocity_world is not None else
                lateral_delta*align_rate + socket[:3, :2] @ planar_velocity)
            lateral_feedforward += feedback_translation_velocity_world+follow_velocity
            twist = np.r_[axis * velocity_z + lateral_feedforward,
                .05 * (follow_rotation@(axis * (rotation_velocity+yaw_compliance_velocity)
                       + alignment_angular_velocity_world)+follow_angular_velocity)]
            joint_velocity = point_jacobian.T @ np.linalg.solve(
                point_jacobian @ point_jacobian.T + .0005**2 * np.eye(6), twist + 3.0 * error)
            if tracking_enabled:
                joint_velocity,velocity_scale=limit_joint_velocity_without_changing_direction(joint_velocity,settings['maximum_arm_speed_rad_s'])
                record['minimum_joint_velocity_scale']=min(record.get('minimum_joint_velocity_scale',1.),velocity_scale)
                arm+=joint_velocity*dt
            else:
                arm += np.clip(joint_velocity, -settings["maximum_arm_speed_rad_s"], settings["maximum_arm_speed_rad_s"]) * dt
            if np.any(arm < bounds[:, 0]) or np.any(arm > bounds[:, 1]):
                raise RuntimeError("the original arm soft limit bounds the nut rotation")
            body_bound = current.copy()
            if progress_enabled and last_observed_body is not None:
                body_bound[:3,3]=last_observed_body[:3,3]
            # The coaxial body/nut axis follows the measured grip orientation;
            # the body avoidance mesh already covers every unobserved body yaw.
            body_bound[:3, :3] = predicted_body_rotation
            locate_bounds(body_bound)
            hit = check(q[:7], q[7:], nut_contact=True)
            checked_pose = "measured_pose"
            if hit is None:
                hit = check(arm, q[7:], nut_contact=True)
                checked_pose = "commanded_arm_with_measured_hand"
            if hit is not None:
                record["geometry_stop"] = {**hit, "checked_pose": checked_pose,
                    "step": int(stepper.step_index), "measured_joints_rad": q.tolist(),
                    "commanded_arm_rad": arm.tolist(), "visual_body_bound": body_bound.tolist()}
                raise RuntimeError(f"current robot or visual Body avoidance check: {hit}")
            # Retain the grip reference and finite preload interval. Optional
            # relaxation slows target corrections driven by raw projected
            # effort; it does not replace the measurement with a drive-torque
            # estimate or alter any measured-effort protection.
            effort = np.asarray(stepper.latest[2])[8:] - tare[1:]
            if root_moments is not None and (regulate_preparation_effort if elapsed<preparation_end else regulate_finger_effort):
                effort=root_moments
                if np.any(effort<.1*desired_effort):
                    raise RuntimeError('recoverable nut turn stop: OBSERVED_GRIP_SLIP')
            if balance_target is not None:
                hand_target[1:] += np.clip(balance_target[1:]-hand_target[1:], -increment, increment)
            elif regulate_preparation_effort if elapsed < preparation_end else regulate_finger_effort:
                if motor_force_control.get('enabled',False):
                    from te_worm_drive import position_reference_for_input_velocity
                    mechanism=world.hand_mechanism
                    velocity=(desired_effort-root_moment_filtered)/(
                        float(settings['finger_position_stiffness_reference_nm_rad'])*max(effort_tau,dt))
                    velocity[np.abs(desired_effort-root_moment_filtered)<=float(motor_force_control['moment_deadband_nm'])]=0.
                    velocity=np.clip(velocity,-float(dynamic['finger_maximum_speed_rad_s']),float(dynamic['finger_maximum_speed_rad_s']))
                    motor_rows=[]
                    for i,name in enumerate(('f1j2','f2j1','f3j2')):
                        hand_target[i+1],motor_row=position_reference_for_input_velocity(mechanism.drives[name],q[i+8],
                            velocity[i],dt,mechanism.settings['motor_position_kp'],mechanism.settings['motor_position_kd'],
                            lower_hand[i+1],upper_hand[i+1])
                        motor_rows.append(motor_row)
                    record['last_friction_aware_finger_command']=motor_rows
                    record['finger_command_semantics']='TORQUE_EQUIVALENT_MOTOR_REFERENCE_WITH_BOUNDED_INPUT_SPEED_AND_UNCHANGED_EFFORT_CAPS'
                else:
                    hand_target[1:] = np.clip(hand_target[1:] + np.clip(
                        effort_gain*(desired_effort - effort) / float(settings.get('finger_position_stiffness_reference_nm_rad',dynamic["hand_stiffness"])), -increment, increment),
                        lower_hand[1:], upper_hand[1:])
            else:
                key = ("held_initial_grip_target_rad" if elapsed < preparation_end
                       else "held_hand_target_after_preparation_rad")
                record.setdefault(key, hand_target.tolist())
            phase = ("visual_align" if elapsed < alignment_duration else
                     "visual_refine" if visual_refinement_active else
                     "axial_settle" if elapsed < preparation_end else
                     "turn" if effective_turn_time < duration else "hold")
            if settings.get('observed_hold_only',False) and initial_fit_accepted:phase='hold'
            before_step = int(stepper.step_index)
            advance_options={}
            if settings.get('arm_trajectory_velocity_feedforward',False):
                advance_options['arm_velocity_target']=joint_velocity.copy()
            stepper.advance("key_probe_nut_rotation_" + phase, arm.copy(), hand_target.copy(),**advance_options)
            if int(stepper.step_index)>before_step:
                record['last_applied_rotation_command_deg']=float(np.rad2deg(angle*fraction))
                record['last_applied_control_step']=before_step
            if engagement_enabled and int(stepper.step_index) > before_step:
                engagement_applied_deg = abs(float(np.degrees(angle*fraction)))
            if (rate_guard_enabled and int(stepper.step_index) > before_step
                    and elapsed >= preparation_end and u < 1.):
                rate_lost_time += dt*(1.-turn_rate_scale)
            index += 1
        if stepper.abort_reason is not None:
            raise RuntimeError(stepper.abort_reason)
        if rate_guard_enabled and not interval_paused and not rate_cycle_finished:
            raise RuntimeError("bounded effort-limited turn duration exhausted before completing the turn and hold")
        record.update(completed=not interval_paused, needs_fresh_observation=interval_paused,
            stage="PAUSED_FOR_FRESH_OBSERVATION" if interval_paused else "ROTATION_COMMAND_FINISHED_REQUIRES_CONTACT_EVALUATION",
            final_arm_target_rad=arm.tolist(), final_hand_target_rad=hand_target.tolist())
    except Exception as error:
        record.update(failure_stage=record["stage"], stage="STOPPED", failure_reason=str(error))
    finally:
        world.pause()
        if tracking_context.get('resources'):
            from te_foundationpose_handoff_runtime import _close_rgbd_resources
            _close_rgbd_resources(tracking_context['resources'])
        if rate_guard_enabled:
            record["effort_limited_turn_time_lost_s"] = rate_lost_time
        if engagement_enabled:
            runtime["engagement_loaded_turn_command_deg"] = engagement_start_deg + engagement_applied_deg
            record["engagement_assistance"] = {
                **engagement, "starting_cumulative_command_deg": engagement_start_deg,
                "final_cumulative_command_deg": runtime["engagement_loaded_turn_command_deg"],
                "angle_is_command_budget_not_actual_nut_rotation": True,
                "applied_through_wrist_force_control_and_original_robot": True,
                "direct_object_force_used": False,
                "source_bench_input_is_not_a_hardware_rating": True}
        if observation_session is not None and 'filtered_at_fixed_origin' in locals():
            observation_session['filtered_wrench_at_fixed_origin']=(filtered_at_fixed_origin.tolist() if filtered_at_fixed_origin is not None else None)
            observation_session['filtered_last_step']=filtered_last_step
        sample_stream.close()
        record.update(last_step=int(stepper.step_index), outer_abort_reason=stepper.abort_reason)
        with gzip.open(output / "joint_ft_samples.json.gz", "wt", encoding="utf-8",compresslevel=1) as stream:
            dump_array(stream,ft.samples[first_ft:])
        save()
    return _json_ready(record)
