"""Completed fourth-interval controls and online visual axes only; no truth reads."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
names=('contact_last_gc128_repeat01_restart01','contact_last_gc128_repeat02')
cases=[]
prefix_arrays=[]
settings_keys=('planar_hold','sidewall_sequence','planar_compliance_m_per_n',
    'planar_compliance_time_constant_s','maximum_planar_offset_m','maximum_planar_speed_m_s',
    'pivot_position_error_gain_s_inv','bound_total_interface_velocity',
    'compensate_measured_contact_load','contact_load_compensation_filter_time_constant_s',
    'angular_admittance_rad_per_nm_s','validated_grip_helical_following')
for name in names:
    run=ROOT/'artifacts/full_validation'/name/'run'
    directory=run/'socket_transport/nut_rotation_continued_02'
    result=json.loads((directory/'nut_rotation_controller_result.json').read_text())
    report=result['coaxial_controller_report'];recipe=report['grip_recipe']
    assert result['completed'] and result['outer_abort_reason'] is None
    rows=[json.loads(line) for line in (directory/'nut_rotation_control_samples.jsonl').read_text().splitlines()]
    assert len(rows)==result['sample_count']==result['last_step']-result['first_step']
    assert [r['step'] for r in rows]==list(range(result['first_step'],result['last_step']))
    visual_path=run/'socket_transport/wrist_socket/camera_and_estimate.json'
    visual=json.loads(visual_path.read_text())
    assert visual['online_object_or_contact_truth_used'] is False
    axes=np.asarray(visual['measurement']['world_from_receptacle_row_major']).reshape(4,4)[:3,:3]
    assert np.max(np.abs(axes.T@axes-np.eye(3)))<1e-10
    dt=report['controller_period_s'];gain=report['pivot_position_error_gain_s_inv']
    offsets=np.asarray([r['offset_socket_m'] for r in rows])
    error=np.asarray([r['pivot_position_tracking_error_m'] for r in rows])
    pivot=np.asarray([r['encoder_pivot_world_m'] for r in rows])
    initial=np.asarray(result['initial_virtual_nut_frame'])[:3,3]
    unlimited=np.asarray([r['unlimited_interface_velocity_world_m_rad_s'] for r in rows])
    requested=np.asarray([r['requested_interface_velocity_world_m_rad_s'] for r in rows])
    residual=np.asarray([r['bounded_velocity_task_residual'] for r in rows])
    assert all(r['control_phase']=='key_probe_nut_rotation_turn' and r['stroke']==0 for r in rows)
    assert recipe['planar_hold'] is True and not recipe.get('sidewall_sequence',False)
    assert np.array_equal(offsets[:,:2],np.zeros_like(offsets[:,:2]))
    assert all(r['angular_follow_offset_deg']==[0.,0.,0.] for r in rows)
    target=initial+offsets@axes.T
    target_residual=float(np.max(np.abs(target-(pivot+error))))
    assert target_residual<1e-12
    offset_velocity=np.diff(np.vstack([np.zeros(3),offsets]),axis=0)/dt
    unlimited_reconstruction=offset_velocity@axes.T+gain*error
    unlimited_residual=float(np.max(np.abs(unlimited[:,:3]-unlimited_reconstruction)))
    assert unlimited_residual<1e-10
    ulocal=unlimited[:,:3]@axes;qlocal=requested[:,:3]@axes;elocal=error@axes
    plimit=report['effective_motion_settings']['planar_force_admittance']['maximum_speed_m_s']
    zlimit=report['effective_motion_settings']['maximum_axial_speed_m_s']
    before=np.linalg.norm(ulocal[:,:2],axis=1);after=np.linalg.norm(qlocal[:,:2],axis=1)
    expected=ulocal.copy();expected[:,:2]*=np.minimum(1.,plimit/np.maximum(before,1e-15))[:,None]
    expected[:,2]=np.clip(expected[:,2],-zlimit,zlimit)
    limiter_residual=float(np.max(np.abs(expected-qlocal)))
    assert limiter_residual<1e-12
    assert np.max(np.abs(ulocal[:,:2]-gain*elocal[:,:2]))<1e-10
    clipped=before>plimit+1e-12
    angles=np.abs([r['commanded_rotation_deg'] for r in rows])
    prefix_arrays.append((angles,before,rows))
    picks=[int(np.argmin(np.abs(angles-a))) for a in (0.,2.,5.,10.,13.4)]
    picks+= [len(rows)-1,int(np.argmax(before))]
    selected=[]
    for i in dict.fromkeys(picks):
        r=rows[i];w=np.asarray(r['interface_wrench'])
        selected.append({'step':r['step'],'observed_sensor_step':r['step']-1,'issued_angle_deg':float(angles[i]),
            'interface_lateral_force_n':float(np.linalg.norm(w[:2])),
            'interface_bending_nm':float(np.linalg.norm(w[3:5])),
            'interface_offset_xy_m':r['offset_socket_m'][:2],
            'interface_position_error_xy_mm':float(np.linalg.norm(elocal[i,:2]))*1000,
            'world_position_error_xy_mm':float(np.linalg.norm(error[i,:2]))*1000,
            'unlimited_interface_xy_mm_s':float(before[i])*1000,
            'requested_interface_xy_mm_s':float(after[i])*1000,
            'lateral_request_scale':float(min(1.,plimit/max(before[i],1e-15))),
            'requested_world_xy_mm_s':float(np.linalg.norm(requested[i,:2]))*1000,
            'joint_velocity_solver_linear_task_residual_norm_m_s':float(np.linalg.norm(residual[i,:3])),
            'joint_velocity_solver_linear_task_residual_interface_xy_m_s':float(np.linalg.norm((residual[i,:3]@axes)[:2]))})
    clip_indices=np.flatnonzero(clipped)
    cases.append({'run_name':name,'scope':'COMPLETED_FOURTH_CONTROL_INTERVAL_ROBOT_AND_ONLINE_VISION_ONLY',
        'first_control_step':result['first_step'],'last_exclusive_control_step':result['last_step'],
        'saved_control_row_count':len(rows),'normal_stop_reason':result['normal_stop_reason'],
        'last_issued_interval_angle_deg':abs(result['last_applied_rotation_command_deg']),
        'early_regrasp':result['early_regrasp'],
        'settings':{k:recipe.get(k,False if k=='sidewall_sequence' else None) for k in settings_keys},
        'online_visual_axes_source':str(visual_path),'interface_axes_in_world':axes.tolist(),
        'all_recorded_interface_offset_xy_exactly_zero':True,
        'maximum_interface_offset_xy_m':float(np.max(np.linalg.norm(offsets[:,:2],axis=1))),
        'target_reconstruction_max_abs_m':target_residual,
        'unlimited_velocity_reconstruction_max_abs_m_s':unlimited_residual,
        'limiter_reconstruction_max_abs_m_s':limiter_residual,
        'maximum_world_xy_reference_shift_from_axial_following_m':float(np.max(np.linalg.norm((target-initial)[:,:2],axis=1))),
        'interface_planar_speed_limit_mm_s':plimit*1000,
        'clipped_saved_row_count':int(clipped.sum()),'clipped_fraction':float(clipped.mean()),
        'clipping_count_numeric_slack_m_s':1e-12,
        'first_clipped_step':None if not len(clip_indices) else rows[int(clip_indices[0])]['step'],
        'first_clipped_angle_deg':None if not len(clip_indices) else float(angles[int(clip_indices[0])]),
        'maximum_unlimited_interface_xy_mm_s':float(before.max())*1000,
        'maximum_requested_interface_xy_mm_s':float(after.max())*1000,
        'minimum_lateral_request_scale':float(np.min(np.minimum(1.,plimit/np.maximum(before,1e-15)))),
        'maximum_joint_velocity_solver_linear_task_residual_norm_m_s':float(np.max(np.linalg.norm(residual[:,:3],axis=1))),
        'maximum_joint_velocity_solver_angular_task_residual_norm_rad_s':float(np.max(np.linalg.norm(residual[:,3:]/.05,axis=1))),
        'selected_saved_controls':selected})

common_count=min(len(x[0]) for x in prefix_arrays)
assert np.array_equal(prefix_arrays[0][0][:common_count],prefix_arrays[1][0][:common_count])
common={'saved_rows_each':common_count,'issued_angle_sequence_exactly_equal':True,
    'last_issued_angle_deg':float(prefix_arrays[0][0][common_count-1]),'cases':[]}
for case,(angles,speed,rows) in zip(cases,prefix_arrays):
    common['cases'].append({'run_name':case['run_name'],
        'clipped_rows':int(np.count_nonzero(speed[:common_count]>.002+1e-12)),
        'clipped_fraction':float(np.mean(speed[:common_count]>.002+1e-12)),
        'maximum_unlimited_interface_xy_mm_s':float(speed[:common_count].max())*1000,
        'last_prefix_bending_nm':float(np.linalg.norm(rows[common_count-1]['interface_wrench'][3:5]))})
result={'scope':'READ_ONLY_PLANAR_HOLD_CONTROL_FACT_REVIEW','cases':cases,
    'matched_issued_command_prefix':common,
    'same_relevant_planar_and_load_settings':cases[0]['settings']==cases[1]['settings'],
    'object_or_contact_truth_read':False,'physical_experiment_executed':False,'production_modified':False,
    'limits':['Recorded control rows use preceding robot observations. The normal-stop event is the next decision, not an additional applied row.',
        'The quoted joint-velocity residual is the Jacobian task-fit residual, not measured actuator tracking.',
        'Interface vectors are projected using the online wrist-vision axes; worldXY is not used as a substitute.',
        'These runs have different prior motion/grip histories. Neither a planar_hold change nor a speed-limit change has been tested here.']}
with (OUT/'robot_only_review.json').open('x') as stream:
    json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
