"""Same-episode bounded nut-turn intervals using current RGBD and encoders.

The nut's actual yaw and all contact truth remain postrun measurements. The
online angle feedback here is explicitly the hand's encoder-derived rotation.
"""
import copy
import gzip
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def encoder_twist_rad(rotation,reference,axis,previous=0.):
    q=Rotation.from_matrix(np.asarray(rotation)@np.asarray(reference).T).as_quat()
    angle=2*np.arctan2(float(q[:3]@axis),float(q[3]))
    return float(angle+2*np.pi*round((previous-angle)/(2*np.pi)))


def bounded_segment_angle_deg(remaining,speed,horizon,acceleration,maximum=.5):
    """Minimum-jerk segment satisfying angle, time, speed and acceleration caps."""
    if not all(np.isfinite(x) and x>0 for x in (speed,horizon,acceleration,maximum)):
        raise ValueError('finite positive interval bounds required')
    magnitude=min(abs(remaining),maximum,speed*horizon/1.875,acceleration*horizon*horizon/(10/np.sqrt(3)))
    return float(np.copysign(magnitude,remaining))


def run_observed_nut_rotation(repository,runtime,stepper,dynamic,grip,socket,settings,output):
    import omni.usd
    import omni.replicator.core as rep
    from te_body_socket_observation import observe_released_plug_from_rgbd
    from te_body_nut_rotation import _run_body_nut_rotation_interval,observed_guided_feature_fit
    from te_foundationpose_handoff_runtime import _json_ready
    repository=Path(repository).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    world=runtime['world'];inputs=runtime['inputs'];ft=runtime['nail_body_ft_auditor'];first_ft=len(ft.samples)
    dt=float(dynamic['physics_dt_s']);socket=np.asarray(socket);axis=socket[:3,2]
    cfg=settings['in_turn_feedback'];maximum=float(cfg.get('maximum_segment_angle_deg',.5))
    period=float(cfg.get('maximum_unobserved_time_s',.5));acceleration=float(cfg.get('maximum_segment_acceleration_deg_s2',13.2))
    if settings.get('maximum_rotation_acceleration_deg_s2') is not None:
        acceleration=min(acceleration,float(settings['maximum_rotation_acceleration_deg_s2']))
    corrections=bool(cfg.get('corrections_enabled',True));tolerance=float(cfg.get('hand_angle_tolerance_deg',.02))
    minimum=float(cfg.get('minimum_segment_angle_deg',.01));requested=float(settings['rotation_about_socket_plus_z_deg'])
    if not (0<maximum<=.5 and 0<period<=.5 and 0<acceleration<=13.2 and .001<=tolerance<=.02
            and 0<minimum<=.01 and 0<abs(requested)<=120 and 'guided_feature_fit' in settings):
        raise ValueError('observed-turn configuration exceeds the declared bounded pilot')
    if settings.get('grip_hold_impedance'):
        raise ValueError('observed-turn pilot retains original hand gains')
    speed=float(settings['maximum_rotation_speed_deg_s']);direction=float(np.sign(requested))
    minimum=min(minimum,abs(bounded_segment_angle_deg(1.,speed,period,acceleration,maximum)))
    lateral_budget=min(.001,float(grip['visual_alignment_allowance_from_quarter_body_clearance_m']))
    session={'lateral_budget_m':lateral_budget};working_grip=copy.deepcopy(grip)
    base_settings=copy.deepcopy(settings);base_settings.pop('in_turn_feedback',None)
    record=dict(completed=False,stage='OBSERVED_TURN_PREPARATION',settings=settings,
        first_step=int(stepper.step_index),simulation_only=True,hardware_authorized=False,
        online_object_or_contact_truth_used=False,loaded_wrench_rezeroed=False,
        physical_thread_progress_verified=False,
        physical_result='REQUIRES_POSTRUN_ACTUAL_NUT_ROTATION_ADVANCE_SLIP_LOAD_AND_HOLD',
        control_samples_file=str(output/'nut_rotation_control_samples.jsonl'),sample_count=0,
        intervals=[],observations=[],hand_encoder_angle_is_not_nut_angle=True)
    stream=(output/'nut_rotation_control_samples.jsonl').open('x',buffering=1)
    first_step=int(stepper.step_index);turn_reference=None;previous_twist=0.;prepared=False;hold_time=0.;bad_count=0
    previous_bound=None;previous_observation_step=None;previous_interval_mode=None;drift_rate=0.
    maximum_steps=round((120+4*1.875*abs(requested)/speed)/dt)

    def save():
        (output/'nut_rotation_controller_result.json').write_text(json.dumps(_json_ready(record),ensure_ascii=False,indent=2)+'\n')

    def hand_pose():
        q=np.asarray(stepper.latest[0],float)
        return np.asarray(inputs.robot_model.forward_kinematics(tuple(q),enforce_limits=False)['handbase_link'])

    try:
        if not grip.get('completed') or stepper.abort_reason is not None:raise RuntimeError('current grip is not ready')
        for interval_index in range(2000):
            if int(stepper.step_index)-first_step>maximum_steps:raise RuntimeError('bounded observed-turn duration exhausted')
            if stepper.abort_reason is not None:raise RuntimeError(stepper.abort_reason)
            world.pause();H=hand_pose();now=float(world.current_time);before_step=int(stepper.step_index)
            fresh=observe_released_plug_from_rgbd(repository,omni.usd.get_context().get_stage(),world,rep,H,
                output/f'observation_{interval_index:04d}')
            if (not fresh.get('position_and_axis_measured') or float(world.current_time)!=now
                    or int(stepper.step_index)!=before_step):raise RuntimeError('current interval RGBD is invalid or advanced physics')
            fresh.update(source='CURRENT_SEGMENT_RGBD_AND_ENCODERS',encoder_step=before_step,physics_time_s=now)
            body=np.asarray(fresh['world_from_plug_five_dof'])
            fit=observed_guided_feature_fit(body,socket,settings['guided_feature_fit'])
            record['observations'].append({'step':before_step,'fit':fit,'observation':fresh})
            record.setdefault('postgrip_palm_observation',fresh)
            session.setdefault('original_hand_rotation_world',H[:3,:3].tolist())
            session['budget_nominal_twist_rad']=encoder_twist_rad(H[:3,:3],session['original_hand_rotation_world'],axis,
                float(session.get('budget_nominal_twist_rad',0.)))
            bound=float(fit['pin_lateral_displacement_bound_m'])
            if (previous_bound is not None and before_step>previous_observation_step
                    and previous_interval_mode in ('TURN','HOLD')):
                rate=max(0.,bound-previous_bound)/((before_step-previous_observation_step)*dt)
                drift_rate=max(drift_rate,rate)
            previous_bound=bound;previous_observation_step=before_step
            child=copy.deepcopy(base_settings)
            child['regulate_finger_effort_during_preparation']=False
            child['regulate_finger_effort_during_rotation']=False
            child['preparation_observation_period_s']=period
            mode='INITIAL_PREPARATION';horizon=period;hand_angle=0.;remaining=requested
            if not prepared:
                # Preserve the existing initial alignment/force preparation.
                # Periodic images observe this process without resetting it.
                child['rotation_about_socket_plus_z_deg']=0.
                child['post_rotation_hold_s']=0.
            else:
                if turn_reference is None:
                    turn_reference=H[:3,:3].copy()
                    session['turn_start_step']=before_step
                    session['initial_turn_force_reference_n']=float(session.get('last_force_reference_n',settings['axial_force_reference_n']))
                    if not corrections:session['locked_hand_from_body']=(np.linalg.inv(H)@body).tolist()
                previous_twist=encoder_twist_rad(H[:3,:3],turn_reference,axis,previous_twist)
                hand_angle=float(np.degrees(previous_twist));remaining=requested-hand_angle
                if direction*hand_angle>abs(requested)+maximum:raise RuntimeError('encoder hand turn exceeded the bounded target region')
                angle_reached=direction*hand_angle>=abs(requested)-tolerance
                need_correction=corrections and not fit['accepted']
                if angle_reached and not need_correction and hold_time>=float(settings['post_rotation_hold_s']):
                    record.update(completed=True,stage='OBSERVED_HAND_TURN_AND_HOLD_FINISHED_REQUIRES_PHYSICAL_EVALUATION',
                        final_observation=fresh,encoder_hand_rotation_deg=hand_angle,
                        final_arm_target_rad=working_grip['fixed_arm_target_rad'],final_hand_target_rad=working_grip['final_hand_target_rad'])
                    break
                if corrections and fit['accepted'] and not angle_reached and drift_rate>0:
                    margin=float(settings['guided_feature_fit']['pin_lateral_displacement_budget_m'])-bound
                    horizon=min(period,.5*max(margin,0.)/drift_rate)
                    possible=bounded_segment_angle_deg(remaining,speed,max(horizon,dt),acceleration,maximum)
                    if abs(possible)<min(minimum,abs(remaining)):
                        need_correction=True;horizon=period;child['force_pose_margin_refinement']=True
                child['maximum_unobserved_execution_s']=period if need_correction else max(dt,np.floor(horizon/dt)*dt)
                child['planar_force_admittance']['enabled']=False
                child['pre_turn_torsional_compliance']['enabled']=False
                child['maximum_rotation_acceleration_deg_s2']=acceleration
                child['axial_settle_duration_s']=0.
                if need_correction:
                    mode='POSE_CORRECTION';bad_count+=1;hold_time=0.
                    if bad_count>int(cfg.get('maximum_consecutive_corrections',8)):
                        raise RuntimeError('current pose did not recover within eight observed corrections')
                    child['rotation_about_socket_plus_z_deg']=0.;child['post_rotation_hold_s']=0.
                elif angle_reached:
                    mode='HOLD';bad_count=0
                    child['rotation_about_socket_plus_z_deg']=0.;child['post_rotation_hold_s']=max(0.,period-dt)
                    child['observed_hold_only']=True
                else:
                    mode='TURN';bad_count=0;hold_time=0.
                    child['rotation_about_socket_plus_z_deg']=bounded_segment_angle_deg(remaining,speed,
                        child['maximum_unobserved_execution_s'],acceleration,maximum)
                    child['post_rotation_hold_s']=0.
                if not corrections:
                    child.pop('guided_feature_fit',None);child.pop('pre_turn_visual_feedback',None)
                    child['pre_turn_visual_alignment']['enabled']=False
            target_before=np.r_[working_grip['fixed_arm_target_rad'],working_grip['final_hand_target_rad']]
            entry={'index':interval_index,'mode':mode,'first_step':before_step,'hand_encoder_rotation_before_deg':hand_angle,
                'measured_feature_drift_rate_m_s':drift_rate,'maximum_unobserved_time_s':child.get('maximum_unobserved_execution_s'),
                'requested_interval_rotation_deg':child['rotation_about_socket_plus_z_deg']}
            record['intervals'].append(entry);record['stage']=mode;save()
            result=_run_body_nut_rotation_interval(repository,runtime,stepper,dynamic,working_grip,socket,child,
                output/f'interval_{interval_index:04d}',initial_position_axis_observation=fresh,observation_session=session)
            entry['result']=result;entry['last_step']=int(stepper.step_index)
            if result.get('control_samples_file'):
                with Path(result['control_samples_file']).open() as f:
                    for line in f:
                        sample=json.loads(line);sample['interval_elapsed_s']=sample['elapsed_s']
                        sample['elapsed_s']=(sample['step']-first_step)*dt
                        sample['interval_commanded_rotation_deg']=sample['commanded_rotation_deg']
                        sample['commanded_rotation_deg']=hand_angle+sample['commanded_rotation_deg']
                        sample['observed_interval_index']=interval_index;sample['observed_interval_mode']=mode
                        stream.write(json.dumps(sample,separators=(',',':'))+'\n');record['sample_count']+=1
            if result.get('failure_reason') or stepper.abort_reason is not None:
                raise RuntimeError(result.get('failure_reason') or stepper.abort_reason)
            if not result.get('completed') and not result.get('needs_fresh_observation'):
                raise RuntimeError('interval stopped without a declared observation boundary')
            working_grip['fixed_arm_target_rad']=result['final_arm_target_rad']
            working_grip['final_hand_target_rad']=result['final_hand_target_rad']
            if not np.array_equal(np.asarray(working_grip['final_hand_target_rad']),target_before[7:]):
                raise RuntimeError('fixed-grip interval changed finger targets')
            elapsed_steps=int(stepper.step_index)-before_step
            if mode!='INITIAL_PREPARATION' and elapsed_steps*dt>period+dt/10:
                raise RuntimeError('interval exceeded the maximum unobserved motion time')
            if mode=='INITIAL_PREPARATION':prepared=True
            elif mode=='HOLD':hold_time+=elapsed_steps*dt
            previous_interval_mode=mode
            print('OBSERVED_NUT_INTERVAL',json.dumps({k:v for k,v in entry.items() if k!='result'}),flush=True)
            record['continued_filter_state']={k:v for k,v in session.items() if k not in ('locked_hand_from_body',)}
            save()
        else:
            raise RuntimeError('bounded observation count exhausted')
    except Exception as error:
        record.update(completed=False,stage='STOPPED',failure_reason=str(error))
    finally:
        world.pause();stream.close();record.update(last_step=int(stepper.step_index),outer_abort_reason=stepper.abort_reason)
        with gzip.open(output/'joint_ft_samples.json.gz','wt') as f:json.dump(_json_ready(ft.samples[first_ft:]),f,separators=(',',':'))
        save()
    return _json_ready(record)
