"""Bounded ended-run archive/finite-mechanism window; no SimulationApp or physics."""
import gzip
import io
import json
import math
from pathlib import Path
import re
import msgpack
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
FIRST,LAST=14700,14719
NAMES=('contact_last_gc128_pose_x1mm_yaw1deg','contact_last_gc128_repeat02')

def window(run):
    path=run/'truth_samples.msgpack.gz'
    index=json.loads(Path(str(path)+'.index.json').read_text())
    assert index['blocks'][-1]['end']==path.stat().st_size
    assert (run/'run_timing.json').exists() and (run/'pickup_controller_outcome.json').exists()
    result=[]
    with path.open('rb') as stream:
        for block in index['blocks']:
            if block['first']>LAST or block['first']+block['count']<=FIRST:continue
            stream.seek(block['offset'])
            reader=msgpack.Unpacker(io.BytesIO(gzip.decompress(stream.read(block['end']-block['offset']))),raw=False,max_buffer_size=0)
            for i in range(block['count']):
                step=block['first']+i
                if step<FIRST or step>LAST:reader.skip();continue
                row=reader.unpack();assert row['step']==step
                result.append(row)
    assert [r['step'] for r in result]==list(range(FIRST,LAST+1))
    return result,index

cases=[]
for name in NAMES:
    run=ROOT/'artifacts/full_validation'/name/'run'
    raw,index=window(run)
    metadata=json.loads((run/'trace_metadata.json').read_text())
    dt=metadata['physics_dt_s'];limit=metadata['maximum_joint_speed_limit_rad_s']
    outcome=json.loads((run/'pickup_controller_outcome.json').read_text())
    rows=[]
    for value in raw:
        diag=value['arm_control']['hand_joint_diagnostic'];pairs={};contacts=[]
        for h in value['contacts']['poll_headers']:
            side=next((i for i,path in enumerate(h['paths'][:2]) if '/handbase_link' in path),None)
            if side is None:continue
            points=[c for c in h['contacts'] if math.hypot(*c['impulse_n_s'])>0.]
            if not points:continue
            key=h['paths'][side].split('/')[-1]+'->'+h['paths'][1-side].split('/')[-1]
            entry=pairs.setdefault(key,{'point_count':0,'impulse_norm_sum_n_s':0.,'minimum_separation_m':float('inf')})
            entry['point_count']+=len(points)
            entry['impulse_norm_sum_n_s']+=sum(math.hypot(*c['impulse_n_s']) for c in points)
            entry['minimum_separation_m']=min(entry['minimum_separation_m'],*(c['separation_m'] for c in points))
            contacts.append({'paths':h['paths'],'positive_points':points})
        velocities={k:v['velocity_rad_s'] for k,v in diag['joints'].items()}
        velocities.update({f'iiwa_joint_{i+1}':v for i,v in enumerate(value['active_velocities_rad_s'][:7])})
        rows.append({'step':value['step'],'phase':value['phase'],
            'all_joint_velocities_rad_s':velocities,
            'hand_joint_positions_rad':{k:v['position_rad'] for k,v in diag['joints'].items()},
            'f1_mimic':diag['mimic_errors']['f1j3'],
            'active_targets_rad':value['active_targets_rad'],
            'positive_hand_contact_pairs':pairs,'positive_native_hand_contact_records':contacts,
            'contact_coverage':{k:value['contacts'].get(k) for k in ('physics_step_callback_count','all_provided_native_report_points_retained','contact_report_channels_agree')}})
    mechanism=[];step_pattern=re.compile(rb'"step"\s*:\s*(\d+)')
    with gzip.open(run/'hand_mechanism_samples.jsonl.gz','rb') as stream:
        for line in stream:
            match=step_pattern.search(line)
            if not match:continue
            step=int(match[1])
            if step>LAST:break
            if step<14714:continue
            r=json.loads(line)
            if r['joint']!='f1j2':continue
            keys=('step','joint','phase','input_angle','input_velocity','output_angle','output_velocity',
                'input_effort','transmission_effort','applied_constraint_slope',
                'rod_length_error_from_actual_body_poses_m','elastic_effort_boundary_exceeded',
                'native_drive_clipping_predicted','drive_saturation','spring_law_residual_nm','output_integration_residual_rad')
            mechanism.append({k:r.get(k) for k in keys})
    targets=np.asarray([r['active_targets_rad'] for r in rows])
    differences=np.diff(targets,axis=0)
    first_positive=next((r['step'] for r in rows if r['positive_hand_contact_pairs']),None)
    overspeed=[{'step':r['step'],'joint':j,'velocity_rad_s':v} for r in rows for j,v in r['all_joint_velocities_rad_s'].items() if abs(v)>limit]
    before,after=rows[-2],rows[-1]
    q0=before['hand_joint_positions_rad']['f1j2'];q1=after['hand_joint_positions_rad']['f1j2']
    m=mechanism[-1]['applied_constraint_slope']
    affine_prediction=before['f1_mimic']['expected_position_rad']+m*(q1-q0)
    linearization_error=after['f1_mimic']['expected_position_rad']-affine_prediction
    assert m==before['f1_mimic']['local_derivative']
    cases.append({'case':name,'run':str(run),'sealed_sample_count':index['sample_count'],
        'archive_bytes':index['blocks'][-1]['end'],'window_inclusive':[FIRST,LAST],
        'motion_timing_present':(run/'motion_timing.json').exists(),
        'early_end_evidence':['run_timing.json','pickup_controller_outcome.json','matching_archive_index'],
        'physics_dt_s':dt,'joint_speed_limit_rad_s':limit,
        'outcome':{k:outcome.get(k) for k in ('completed','failure_reason','maximum_joint_speed_rad_s','maximum_joint_speed_joint')},
        'first_positive_hand_contact_in_window':first_positive,
        'overspeed_observations_in_window':overspeed,
        'any_positive_hand_nut_contact_in_window':any(any('CouplingNut' in key for key in r['positive_hand_contact_pairs']) for r in rows),
        'maximum_arm_target_step_change_rad':float(np.max(np.abs(differences[:,:7]))),
        'finger_target_step_change_min_rad':differences[:,8:].min(axis=0).tolist(),
        'finger_target_step_change_max_rad':differences[:,8:].max(axis=0).tolist(),
        'finger_command_rate_rad_s':(differences[:,8:]/dt).max(axis=0).tolist(),
        'final_f1_pair_position_difference_velocity_rad_s':{j:(after['hand_joint_positions_rad'][j]-before['hand_joint_positions_rad'][j])/dt for j in ('f1j2','f1j3')},
        'final_f1_tangent_geometric_linearization_error_rad':linearization_error,
        'rows':rows,'f1_finite_mechanism_14714_14719':mechanism})

assert cases[0]['overspeed_observations_in_window']==[{'step':14719,'joint':'f1j3','velocity_rad_s':-4.000000476837158}]
assert np.array_equal(np.asarray([r['active_targets_rad'][7:] for r in cases[0]['rows']]),np.asarray([r['active_targets_rad'][7:] for r in cases[1]['rows']]))
result={'scope':'ENDED_POSE_VARIATION_FIRST_CONTACT_SPEED_ABORT_BOUNDED_REVIEW',
    'cases':cases,'hand_targets_identical_between_cases_in_window':True,
    'record_steps_first_contact_to_stop':3,'index_time_first_contact_to_stop_s':3*cases[0]['physics_dt_s'],
    'physics_executed':False,'production_or_original_records_modified':False,
    'timing_boundary':'Target is sent before world.step. Native contact reports are captured during that step; joint/native pose readback and archive row follow world.step; then step_index increments and the speed abort is applied. Contact manifold positions/separations need not be at the exact poststep pose time.',
    'limitations':['First positive contact means first in the selected14700..14719 window; no whole-episode raw rescan is claimed.',
        'Impulse norm sums identify loaded actor pairs, not a motor-torque measurement or contact-source mesh attribution.',
        'A smooth tangent slope and small geometric linearization error do not identify which native constraint/contact point caused the transient.',
        'Position-difference velocities are not substitutes for the instantaneous native velocity used by the unchanged3rad/s guard.']}
with (OUT/'independent_contact_speed_window.json').open('x') as stream:
    json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
print(json.dumps({'cases':[{'case':c['case'],'first_contact':c['first_positive_hand_contact_in_window'],'overspeed':c['overspeed_observations_in_window'],'target_rate':c['finger_command_rate_rad_s'],'position_difference_velocity':c['final_f1_pair_position_difference_velocity_rad_s'],'tangent_linearization_error_rad':c['final_f1_tangent_geometric_linearization_error_rad']} for c in cases]},indent=2))
