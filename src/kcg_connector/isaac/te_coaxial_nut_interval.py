"""Run the physically tested coaxial controller inside the live visual task.

The existing visual regrasp and release stages remain responsible for hand
transfer, current RGB-D observations and open-hand collision checks. This
adapter changes only the loaded turn, using the same finite hand mechanism.
"""
import copy
import gzip
import json
from pathlib import Path

import numpy as np


def run_coaxial_nut_interval(repository,runtime,stepper,dynamic,grip,socket,settings,output):
    from te_local_interface_following import LocalInterfaceFollowing
    from carts_v2.fast_json import dumps,dump_array
    root=Path(repository);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    world=runtime['world'];ft=runtime['nail_body_ft_auditor'];dt=float(world.get_physics_dt())
    mechanism=world.hand_mechanism
    recipe=copy.deepcopy(json.loads((root/settings['coaxial_interface_recipe']).read_text()))
    # A staged grasp may deliberately establish a different finite preload
    # before capture. Keep that measured, recorded reference instead of
    # silently restoring a stronger reference at the turn boundary.
    recipe['root_moment_targets_nm']=list(grip['effort_reference_nm'])
    geometry=json.loads(Path(recipe['source_geometry_plan']).read_text())
    if geometry['finger_mechanism_id']!=mechanism.setup['mechanism_id']:
        raise ValueError('The live grasp and coaxial controller must use the same fourbar hand')
    degrees=-float(settings['rotation_about_socket_plus_z_deg'])
    speed=float(settings['maximum_rotation_speed_deg_s'])
    if not (0<degrees<=120 and 0<speed<=50):raise ValueError('Finite verified-direction wrist stroke required')
    diagnostic=settings.get('diagnostic_hold_and_prefix')
    profile_degrees=degrees;initial_hold_s=0.
    if diagnostic:
        if not runtime.get('declared_source_stage_diagnostic'):
            raise ValueError('A cut trajectory prefix is restricted to the declared local diagnosis')
        profile_degrees=float(diagnostic['reference_stroke_deg'])
        initial_hold_s=float(diagnostic['initial_hold_s'])
        if not (profile_degrees==90. and 0<degrees<=6. and 0<=initial_hold_s<=1.):
            raise ValueError('Local prefix diagnosis is bounded to90degree reference,6degree prefix and1s hold')
    duration=1.875*profile_degrees/speed
    stroke_acceleration=settings.get('coaxial_maximum_profile_acceleration_deg_s2')
    if stroke_acceleration is not None:
        from te_visual_seating_axis import seating_segment_duration
        acceleration=float(stroke_acceleration)
        if not 0<acceleration<=50.:raise ValueError('Finite transferred stroke acceleration must be within0..50deg/s²')
        duration=seating_segment_duration(profile_degrees,speed,acceleration)
    before=float(runtime.get('coaxial_nut_commanded_degrees',0.))
    recipe['seating_torque_detection']['minimum_command_deg']=max(0.,350.-before)
    history_file=Path(grip['robot_sensor_history_file'])
    if not history_file.resolve().is_relative_to(out.resolve().parent):
        raise ValueError('Controller calibration must come from the current physical episode')
    with gzip.open(history_file,'rt') as stream:history=json.load(stream)
    opened=[row for row in history if row['phase']=='key_probe_nut_tare']
    loaded=[row for row in history if row['phase']=='key_probe_nut_grip_hold'][-max(20,round(.5/dt)):]
    # The declared old zero belongs only to the initial cold grip. A later
    # actual regrasp has its own newly measured open-hand calibration.
    diagnostic_calibration=runtime.pop('coaxial_declared_cold_grip_calibration',None)
    if diagnostic_calibration:
        if opened or not runtime.get('declared_source_stage_diagnostic'):
            raise ValueError('Earlier sensor zero is only allowed in the explicit cold loaded-grip diagnostic')
        opened=diagnostic_calibration['open_history']
    q=np.asarray(stepper.latest[0],float).copy()
    model=runtime['inputs'].robot_model
    hand=np.asarray(model.forward_kinematics(q,enforce_limits=False)['handbase_link'])
    body=np.asarray(grip['world_from_body_palm_five_dof'],float)
    fixture=np.asarray(socket,float);axis=fixture[:3,2]
    pivot=hand@np.linalg.inv(np.asarray(geometry['canonical_body_from_hand_for_nut_grasp']))
    # The fixed socket axis comes from the current visual assembly scene.
    # Hand encoders and the source grasp supply its axial coordinate. This is
    # an estimated grasp frame, never a simulator Nut pose measurement.
    pivot[:3,3]=fixture[:3,3]+axis*float(axis@(pivot[:3,3]-fixture[:3,3]))
    controller=LocalInterfaceFollowing(root,q,pivot,body,fixture,dt,grip_recipe=recipe,
        finger_mechanism_path=mechanism.setup['contract_path'],hand_mechanism=mechanism,
        arm_position_stiffness=float(stepper.settings['arm_stiffness']))
    controller.lower=np.asarray(stepper.arm_lower_limits).copy()
    controller.upper=np.asarray(stepper.arm_upper_limits).copy()
    controller.adopt_existing_grip(opened,loaded,
        diagnostic_open_calibration_source=diagnostic_calibration['source_file'] if diagnostic_calibration else None)
    stepper.payload_compensation_fraction=0.
    stepper.set_lift_arm_damping(160.)
    runtime['nut_root_moment_observer']=controller.grip_observer
    first_ft=len(ft.samples);first_step=int(stepper.step_index)
    record={'completed':False,'stage':'CURRENT_VISION_COAXIAL_NUT_TURN','settings':settings,
        'first_step':first_step,'coaxial_interface_recipe':settings['coaxial_interface_recipe'],
        'robot_sensor_history_file':str(history_file),'commanded_before_interval_deg':before,
        'control_samples_file':str(out/'nut_rotation_control_samples.jsonl'),
        'initial_arm_encoder_rad':q[:7].tolist(),'hand_from_virtual_nut_axis_frame':controller.hand_from_pivot.tolist(),
        'initial_virtual_nut_frame':pivot.tolist(),'postgrip_palm_observation':grip.get('preclose_palm_observation'),
        'online_object_or_contact_truth_used':False,'source_of_axis':'CURRENT_VISUAL_SOCKET',
        'source_of_grasp_axial_position':'ROBOT_ENCODERS_AND_SOURCE_CAD_GRASP',
        'physical_thread_progress_verified':False,'sample_count':0}
    if diagnostic_calibration:
        record['declared_cold_sensor_calibration']={
            'source_file':diagnostic_calibration['source_file'],
            'new_loaded_sensor_history':str(history_file),
            'complete_visual_assembly_claimed':False,'loaded_wrench_rezeroed':False}
    (out/'coaxial_recipe.json').write_text(json.dumps(recipe,indent=2)+'\n')
    check=runtime['nut_regrasp_geometry_check'];locate=runtime['nut_regrasp_locate_visual_bounds']
    geometry_reference={'body':body.copy(),'pivot_position':pivot[:3,3].copy()}
    axis_control=settings.get('visual_seating_axis_control')
    use_seating_schedule=bool(settings.get('use_verified_seating_schedule'))
    if axis_control and use_seating_schedule:
        raise ValueError('Select one seating orientation controller')
    if diagnostic and (axis_control or use_seating_schedule):
        raise ValueError('Local hold/prefix uses the unchanged ordinary coaxial controller')
    if diagnostic:
        record['diagnostic_hold_and_prefix']={
            'initial_hold_s':initial_hold_s,'requested_prefix_deg':degrees,
            'reference_stroke_deg':profile_degrees,'reference_duration_s':duration,
            'controller_clock_continues_during_hold':True,
            'zero_tightening_command_still_allows_existing_pose_feedback':True,
            'prefix_ends_by_pausing_the_diagnostic_not_a_physical_braking_or_release_test':True,
            'complete_ninety_degree_stroke_claimed':False}
    schedule=None
    if use_seating_schedule:
        from te_interface_stroke_schedule import InterfaceStrokeSchedule
        schedule=InterfaceStrokeSchedule(degrees,degrees,duration,release=False,
            grip_duration_s=3.,preload_duration_s=.5,
            seating_alignment=recipe['seating_alignment_stage'],post_turn_hold_s=0.)
        record['verified_seating_schedule']={**schedule.as_dict(),
            'initial_schedule_grip_clock_skipped':True,'actual_grip_already_acquired':True}
    def commands():
        if axis_control:
            from te_visual_seating_axis import visual_seating_commands
            yield from visual_seating_commands(root,runtime,stepper,controller,fixture,grip,out,
                record,geometry_reference,degrees,speed,axis_control)
        elif schedule is not None:
            start=schedule.phases[1].start
            for index in range(int(np.ceil((schedule.duration-start)/dt))+1):
                elapsed=index*dt;part,angle,rate,_=schedule.sample(start+elapsed)
                phase=('key_probe_nut_rotation_visual_refine' if part.name=='alignment_hold'
                    else 'key_probe_nut_rotation_turn')
                yield index,elapsed,angle,rate,phase
        else:
            hold_steps=round(initial_hold_s/dt)
            for index in range(hold_steps):
                yield index,index*dt,0.,0.,'key_probe_nut_rotation_hold'
            for index in range(round(duration/dt)):
                elapsed=index*dt;u=float(np.clip(elapsed/duration,0.,1.))
                blend=10*u**3-15*u**4+6*u**5
                angle=min(np.radians(profile_degrees)*blend,np.radians(degrees))
                yield hold_steps+index,initial_hold_s+elapsed,angle,np.radians(profile_degrees)*30*u*u*(1-u)**2/duration,'key_probe_nut_rotation_turn'
                if diagnostic and angle>=np.radians(degrees):break
    angle=0.;last_applied_angle=0.;failure=None;elapsed=0.;axis_output_goal=None
    try:
        world.play()
        with (out/'nut_rotation_control_samples.jsonl').open('x') as stream:
            for index,elapsed,angle,rate,phase in commands():
                if stepper.abort_reason:raise RuntimeError(stepper.abort_reason)
                q=np.asarray(stepper.latest[0],float)
                sample=ft.samples[-1]
                nominal=controller.update(q,sample['hand2arm_raw_wrench'],controller.turn_start+elapsed,
                    angle,rate,projected_reactions=np.asarray(stepper.latest[2])[8:],
                    grip_elapsed=controller.turn_start,
                    regulate_static_grip=bool(use_seating_schedule
                        and phase=='key_probe_nut_rotation_visual_refine'
                        and recipe.get('regulate_grip_while_aligning')))
                active_axis_grip=bool(axis_control and axis_control.get('hold_finger_output_posture')
                    and phase=='key_probe_nut_rotation_visual_refine')
                if active_axis_grip:
                    from te_worm_drive import finger_output_posture_targets
                    if axis_output_goal is None:
                        axis_output_goal=controller.hand_goal.copy();axis_output_goal[1:]=q[8:]
                        record.setdefault('axis_finger_output_holds',[]).append({
                            'first_step':int(stepper.step_index),'physical_output_reference_rad':axis_output_goal.tolist(),
                            'source':'CURRENT_ROBOT_OUTPUT_ENCODERS','physical_outputs_not_locked':True})
                    controller.hand_goal,motor_rows=finger_output_posture_targets(mechanism,q[8:],
                        controller.hand_goal,axis_output_goal,dt,allow_closing=True)
                    controller.last_grip.update(existing_transmission_input_self_lock_hold=False,
                        finite_input_motor_output_posture_hold=True,
                        hand_goal_rad=controller.hand_goal.tolist(),
                        physical_output_reference_rad=axis_output_goal.tolist(),
                        motor_command_summary=motor_rows)
                else:axis_output_goal=None
                current=np.asarray(model.forward_kinematics(q,enforce_limits=False)['handbase_link'])@controller.hand_from_pivot
                # The keyed Body does not follow lateral hand deflection.
                # Only axial progress is inferred from the held grasp; the
                # configured conservative envelope covers captive axial play.
                delta=current[:3,3]-geometry_reference['pivot_position']
                bound=geometry_reference['body'].copy();bound[:3,3]+=axis*float(axis@delta)
                locate(bound)
                hit=check(q[:7],q[7:],nut_contact=True)
                checked_pose='MEASURED_ROBOT_POSE'
                if hit is None:
                    if recipe.get('encoder_step_geometry_prediction',False):
                        # The nominal PD reference contains load-tracking
                        # correction; it is not the next physical pose.
                        # Check both measured and requested one-tick motion
                        # with the same conservative visual obstacle.
                        measured_velocity=np.asarray(stepper.latest[1],float)
                        predictions=(
                            ('ENCODER_PLUS_MEASURED_VELOCITY_STEP',q+dt*measured_velocity),
                            ('ENCODER_PLUS_REQUESTED_BOUNDED_VELOCITY_STEP',
                             q+dt*np.r_[controller.last_joint_velocity,measured_velocity[7:]]))
                        for checked_pose,predicted in predictions:
                            hit=check(predicted[:7],predicted[7:],nut_contact=True)
                            if hit is not None:break
                    else:
                        hit=check(nominal,q[7:],nut_contact=True)
                        checked_pose='NOMINAL_ARM_REFERENCE_WITH_MEASURED_HAND_OUTPUTS'
                if hit is not None:
                    record['geometry_stop']={'step':int(stepper.step_index),**hit,
                        'checked_pose':checked_pose,'body_bound_world':bound.tolist(),
                        'measured_joints_rad':q.tolist(),'nominal_arm_reference_rad':nominal.tolist()}
                    raise RuntimeError('Current robot/visual Body avoidance: '+str(hit))
                if controller.seating_torque_candidate is not None:
                    record['seating_candidate']={**controller.seating_torque_candidate,
                        'cumulative_commanded_degrees':before+float(np.degrees(angle)),
                        'source':'CURRENT_ROBOT_WRIST_TORQUE_NOT_OBJECT_CONTACT_TRUTH'}
                    break
                wrench_reserve=recipe.get('regrasp_bending_reserve')
                if wrench_reserve:
                    threshold=float(wrench_reserve['bending_nm'])
                    if not 0<threshold<float(recipe['planned_wrist_bending_observation_nm']):
                        raise ValueError('Normal regrasp must precede the unchanged bending stop')
                    observed_wrench=np.asarray(controller.records[-1]['interface_wrench'])
                    bending=float(np.linalg.norm(observed_wrench[3:5]))
                    if (float(np.degrees(angle))>=float(wrench_reserve['minimum_stroke_deg'])
                            and bending>=threshold):
                        record['normal_stop_reason']='WRIST_BENDING_RESERVE_EARLY_REGRASP'
                        record['early_regrasp']={'step':int(stepper.step_index),
                            'filtered_interface_wrench_n_nm':observed_wrench.tolist(),
                            'wrench_origin':'CURRENT_ENCODER_PIVOT_SAME_AS_HARD_WRIST_BOUND',
                            'bending_nm':bending,'threshold_nm':threshold,
                            'last_applied_rotation_deg':float(np.degrees(last_applied_angle)),
                            'all_original_force_and_transmission_limits_retained':True}
                        break
                force_reserve=recipe.get('regrasp_force_reserve')
                if force_reserve:
                    threshold=float(force_reserve['force_n'])
                    if not 0<threshold<float(recipe['planned_wrist_force_limit_n']):
                        raise ValueError('Normal regrasp must precede the unchanged resultant-force stop')
                    observed_wrench=np.asarray(controller.records[-1]['interface_wrench'])
                    force=float(np.linalg.norm(observed_wrench[:3]))
                    if force>=threshold:
                        record['normal_stop_reason']='WRIST_FORCE_RESERVE_EARLY_REGRASP'
                        record['early_regrasp']={'step':int(stepper.step_index),
                            'filtered_interface_wrench_n_nm':observed_wrench.tolist(),
                            'wrench_origin':'CURRENT_ENCODER_PIVOT_SAME_AS_HARD_WRIST_BOUND',
                            'force_n':force,'threshold_n':threshold,
                            'last_applied_rotation_deg':float(np.degrees(last_applied_angle)),
                            'all_original_force_and_transmission_limits_retained':True}
                        break
                reserve=recipe.get('regrasp_elastic_reserve')
                if reserve and float(np.degrees(angle))>=float(reserve['minimum_stroke_deg']):
                    requested_margin=float(reserve['minimum_margin_nm'])
                    if not 0<requested_margin<=.5:raise ValueError('Regrasp reserve must remain within0..0.5Nm')
                    margins={name:float(drive.reference.transmission_effort_boundary
                        -abs(drive.reference.transmission_stiffness*(drive.input_angle-q[8+i])))
                        for i,name in enumerate(('f1j2','f2j1','f3j2'))
                        for drive in (mechanism.drives[name],)}
                    if min(margins.values())<=requested_margin:
                        record['normal_stop_reason']='ELASTIC_RESERVE_EARLY_REGRASP'
                        record['early_regrasp']={'step':int(stepper.step_index),
                            'elastic_margins_nm':margins,'threshold_nm':requested_margin,
                            'last_applied_rotation_deg':float(np.degrees(last_applied_angle)),
                            'all_original_force_and_transmission_limits_retained':True}
                        break
                step=int(stepper.step_index)
                stepper.advance(phase,nominal,controller.hand_goal,
                    arm_velocity_target=controller.last_joint_velocity,
                    arm_load_compensation_nm=controller.last_contact_compensation)
                if stepper.abort_reason:raise RuntimeError(stepper.abort_reason)
                last_applied_angle=angle
                if recipe.get('encoder_step_geometry_prediction',False):
                    actual=np.asarray(stepper.latest[0],float)
                    actual_pivot=np.asarray(model.forward_kinematics(actual,enforce_limits=False)['handbase_link'])@controller.hand_from_pivot
                    actual_bound=geometry_reference['body'].copy()
                    actual_bound[:3,3]+=axis*float(axis@(actual_pivot[:3,3]-geometry_reference['pivot_position']))
                    locate(actual_bound)
                    hit=check(actual[:7],actual[7:],nut_contact=True)
                    if hit is not None:
                        record['geometry_stop']={'step':int(stepper.step_index),**hit,
                            'checked_pose':'MEASURED_ROBOT_POSE_AFTER_STEP',
                            'body_bound_world':actual_bound.tolist(),'measured_joints_rad':actual.tolist()}
                        raise RuntimeError('Current robot/visual Body avoidance after step: '+str(hit))
                row={**controller.records[-1],'step':step,'elapsed_s':elapsed,
                    'control_phase':phase,
                    'commanded_rotation_deg':-float(np.degrees(angle)),
                    'cumulative_commanded_degrees':before+float(np.degrees(angle))}
                stream.write(dumps(row)+'\n');record['sample_count']+=1
                if index%240==0:
                    stream.flush();(out/'progress.json').write_text(dumps(row)+'\n')
        record.update(completed=True,stage=('EARLY_REGRASP_STOP_REQUIRES_PHYSICAL_AUDIT'
            if record.get('early_regrasp') else 'ROTATION_FINISHED_REQUIRES_PHYSICAL_AUDIT'))
        if diagnostic:
            record.update(stage='DIAGNOSTIC_HOLD_AND_PREFIX_FINISHED_NOT_A_COMPLETE_STROKE',
                diagnostic_window_completed=True,physical_mating_or_release_claimed=False)
    except (RuntimeError,ValueError) as error:
        failure=str(error);record.update(stage='STOPPED',failure_reason=failure)
    finally:
        world.pause()
        record.update(last_step=int(stepper.step_index),outer_abort_reason=stepper.abort_reason,
            geometry_prediction_mode=('CURRENT_ENCODERS_MEASURED_AND_REQUESTED_ONE_STEP_PLUS_POSTSTEP'
                if recipe.get('encoder_step_geometry_prediction',False) else 'LEGACY_NOMINAL_REFERENCE'),
            final_arm_target_rad=np.asarray(stepper.latest[0])[:7].tolist(),
            final_hand_target_rad=controller.hand_goal.tolist(),
            last_applied_rotation_command_deg=-float(np.degrees(last_applied_angle)),
            commanded_rotation_duration_s=elapsed+dt if axis_control or use_seating_schedule or diagnostic else duration,
            planned_peak_rotation_speed_deg_s=(max((r['peak_speed_deg_s'] for r in record.get('visual_seating_axis_control',{}).get('turn_profiles',[])),default=0.) if axis_control else
                max(1.875*abs(p.angle1-p.angle0)/p.duration for p in schedule.phases) if schedule else 1.875*profile_degrees/duration),
            planned_peak_rotation_acceleration_deg_s2=(10./np.sqrt(3.))*profile_degrees/duration**2 if not axis_control and not schedule else None,
            coaxial_controller_report=controller.report(),release_from_actual_encoder_pose=True,
            release_with_contact_feedforward_removed=True)
        runtime['coaxial_nut_commanded_degrees']=before+float(np.degrees(last_applied_angle))
        with gzip.open(out/'joint_ft_samples.json.gz','wt',compresslevel=1) as stream:dump_array(stream,ft.samples[first_ft:])
        (out/'nut_rotation_controller_result.json').write_text(dumps(record)+'\n')
    return record
