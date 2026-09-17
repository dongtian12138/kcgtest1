"""Six sealed pose frames: base/finger center-prediction decomposition, no physics."""
import ast
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
RUN=ROOT/'artifacts/full_validation/contact_last_gc128_repeat02/run'
sys.path.insert(0,str(ROOT/'src/kcg_connector/isaac'))
from trace_metadata import iter_truth_fields

def load(path):return json.loads(path.read_text())
def pose(row,name):
    d=row['native_robot_link_pose_audit']['poses'][name]
    p=np.asarray(d['position_world_m'],float);q=np.asarray(d['orientation_world_wxyz'],float)
    assert p.shape==(3,) and q.shape==(4,) and np.isfinite(np.r_[p,q]).all()
    H=np.eye(4);H[:3,3]=p;H[:3,:3]=Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
    assert np.max(np.abs(H[:3,:3].T@H[:3,:3]-np.eye(3)))<1e-12
    return H

index=load(RUN/'truth_samples.msgpack.gz.index.json')
assert (RUN/'motion_timing.json').exists()
assert index['blocks'][-1]['end']==(RUN/'truth_samples.msgpack.gz').stat().st_size
install=load(RUN/'frozen_model_installation.json')
socket=np.asarray(ast.literal_eval(install['pose_mass_velocity_after'][
    '/World/TEVisualHandoff/FixedReceptaclePose']['world_transform'])).T
assert np.array_equal(socket[:3,:3],np.eye(3))
fields={'step','phase','object_part_positions_m','native_robot_link_pose_audit','arm_control'}
cache={}
def sample(step):
    if step not in cache:
        values=list(iter_truth_fields(RUN,fields,first_step=step,last_step=step))
        assert len(values)==1 and values[0]['step']==step
        cache[step]=values[0]
    return cache[step]

results=[]
for existing in load(HERE/'comparison.json'):
    assert Path(existing['run'])==RUN
    start=existing['rows'][0];maximum=max(existing['rows'],key=lambda r:r['reported_xy_error_mm'])
    directory=RUN/'socket_transport'/existing['stage']
    controls=[json.loads(s) for s in (directory/'nut_rotation_control_samples.jsonl').read_text().splitlines()]
    assert controls[0]['step']==start['control_step']
    actual_max=max(controls,key=lambda r:np.linalg.norm(r['pivot_position_tracking_error_m'][:2]))
    assert actual_max['step']==maximum['control_step']
    assert start['physical_observation_step']==start['control_step']-1
    assert maximum['physical_observation_step']==maximum['control_step']-1
    row0=sample(start['physical_observation_step']);row=sample(maximum['physical_observation_step'])
    H0,H=pose(row0,'handbase_link'),pose(row,'handbase_link')
    N0=np.r_[row0['object_part_positions_m'][1],1.]
    N=np.asarray(row['object_part_positions_m'][1],float)
    n0_hand=np.linalg.solve(H0,N0)
    P_H=(H@n0_hand)[:3]
    D=N-P_H
    expected=np.asarray(maximum['error_decomposition_world_mm']['actual_nut_minus_fixed_grasp_prediction'])
    assert np.max(np.abs(D*1000-expected))<1e-9
    joints=('f1j2','f2j1','f3j2')
    changed=[]
    for j in joints:
        q0=row0['arm_control']['hand_joint_diagnostic']['joints'][j]['position_rad']
        q=row['arm_control']['hand_joint_diagnostic']['joints'][j]['position_rad']
        changed.append(float(np.degrees(q-q0)))
    assert np.max(np.abs(np.asarray(changed)-maximum['finger_output_angle_change_deg']))<1e-10
    finger_rows=[]
    for name in ('f1Link3','f2Link2','f3Link3'):
        F0,F=pose(row0,name),pose(row,name)
        P_F=(F@np.linalg.solve(F0,N0))[:3]
        A=P_F-P_H;B=N-P_F
        C0=np.linalg.solve(H0,F0);C=np.linalg.solve(H,F)
        # Independent factorization in the hand frame confirms transform order.
        A_hand=(C@np.linalg.solve(C0,n0_hand))[:3]-n0_hand[:3]
        A_alt=H[:3,:3]@A_hand
        closure=float(np.max(np.abs(A+B-D)))
        relative_factorization=float(np.max(np.abs(A-A_alt)))
        assert closure<1e-12 and relative_factorization<1e-12
        xy_norm=float(np.linalg.norm(D[:2]));direction=D[:2]/xy_norm
        finger_rows.append({'terminal_link':name,
            'P_finger_world_m':P_F.tolist(),
            'finger_relative_motion_term_world_mm':(A*1000).tolist(),
            'remaining_finger_to_nut_center_term_world_mm':(B*1000).tolist(),
            'finger_relative_motion_xy_norm_mm':float(np.linalg.norm(A[:2]))*1000,
            'remaining_finger_to_nut_center_xy_norm_mm':float(np.linalg.norm(B[:2]))*1000,
            'finger_relative_motion_xyz_norm_mm':float(np.linalg.norm(A))*1000,
            'remaining_finger_to_nut_center_xyz_norm_mm':float(np.linalg.norm(B))*1000,
            'signed_projection_on_total_xy_direction_mm':{'finger_motion':float(A[:2]@direction)*1000,'remaining_relation':float(B[:2]@direction)*1000},
            'relative_terminal_origin_change_in_hand_frame_mm':((C[:3,3]-C0[:3,3])*1000).tolist(),
            'relative_terminal_rotation_change_deg':float(np.degrees(Rotation.from_matrix(C[:3,:3]@C0[:3,:3].T).magnitude())),
            'vector_closure_max_abs_m':closure,
            'hand_frame_factorization_max_abs_m':relative_factorization,
            'initial_native_finger_world_transform':F0.tolist(),'maximum_native_finger_world_transform':F.tolist()})
    results.append({'stage':existing['stage'],'start_control_step':start['control_step'],
        'start_physical_observation_step':row0['step'],'maximum_control_step':maximum['control_step'],
        'maximum_physical_observation_step':row['step'],'maximum_command_deg':maximum['command_deg'],
        'start_phase':row0['phase'],'maximum_phase':row['phase'],
        'reported_virtual_xy_error_mm':maximum['reported_xy_error_mm'],
        'actual_nut_lateral_to_socket_mm':float(np.linalg.norm((N-socket[:3,3])[:2]))*1000,
        'N0_world_m':N0[:3].tolist(),'N_world_m':N.tolist(),'P_hand_world_m':P_H.tolist(),
        'base_relative_center_change_world_mm':(D*1000).tolist(),
        'base_relative_center_change_xy_norm_mm':float(np.linalg.norm(D[:2]))*1000,
        'initial_native_hand_world_transform':H0.tolist(),'maximum_native_hand_world_transform':H.tolist(),
        'finger_output_angle_change_deg':dict(zip(joints,changed)),'fingers':finger_rows})

result={'scope':'SEALED_REPEAT02_SIX_POSE_FRAME_FINGER_CENTER_HYPOTHESIS_DECOMPOSITION',
    'run':str(RUN),'sealed_sample_count':index['sample_count'],'sealed_archive_bytes':index['blocks'][-1]['end'],
    'unique_truth_steps_read':sorted(cache),'world_xy_is_physical_socket_lateral_plane':True,
    'formula':{'P_hand':'H(t) H(0)^-1 [N(0),1]', 'P_finger_i':'Fi(t) Fi(0)^-1 [N(0),1]',
               'finger_relative_motion_term':'P_finger_i - P_hand',
               'remaining_finger_center_relation_term':'N(t) - P_finger_i',
               'closure':'N(t)-P_hand = (P_finger_i-P_hand) + (N(t)-P_finger_i)'},
    'results':results,'physics_executed':False,'production_or_source_reports_modified':False,
    'limitations':['Each finger is a separate hypothetical rigid attachment of the initial Nut reference center, not a measured no-slip contact.',
        'The finger-relative term includes all observed terminal-link motion relative to the base, including joint/palm motion and any constraint/pose residual; it is not a uniquely identified single-joint causal contribution.',
        'The remaining center relation can change through relative rotation, rolling, patch migration, sliding or compliance; it is not pure tangential material slip.',
        'Per-finger decompositions must not be summed or averaged as physical contributions. Only vectors within each identity add; norms and projections are not causal percentages.',
        'Only stage start and recorded maximum-error observations are sampled; no history/contact-material correspondence or full-episode acceptance is inferred.']}
with (HERE/'independent_finger_motion_decomposition.json').open('x') as stream:
    json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
print(json.dumps({'steps':result['unique_truth_steps_read'],'results':[{'stage':r['stage'],'D_world_mm':r['base_relative_center_change_world_mm'],'D_xy_mm':r['base_relative_center_change_xy_norm_mm'],'finger_output_angle_change_deg':r['finger_output_angle_change_deg'],'fingers':[{k:v for k,v in f.items() if k in ('terminal_link','finger_relative_motion_term_world_mm','remaining_finger_to_nut_center_term_world_mm','finger_relative_motion_xy_norm_mm','remaining_finger_to_nut_center_xy_norm_mm','signed_projection_on_total_xy_direction_mm','relative_terminal_rotation_change_deg')} for f in r['fingers']]} for r in results]},ensure_ascii=False,indent=2))
