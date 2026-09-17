"""Bounded check of existing sealed-run pose decomposition; no simulator imports."""
import ast
import gzip
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src/kcg_connector/isaac'))
from trace_metadata import iter_truth_fields

def load(path):return json.loads(path.read_text())
def transform(position,orientation_wxyz):
    H=np.eye(4);H[:3,3]=position
    H[:3,:3]=Rotation.from_quat(np.asarray(orientation_wxyz)[[1,2,3,0]]).as_matrix()
    return H
def ft_transform(row):
    H=np.eye(4);H[:3,3]=row['handbase_position_world_m']
    H[:3,:3]=row['handbase_rotation_world_row_major']
    return H

comparison=load(Path(__file__).with_name('comparison.json'))
results=[]
for existing in comparison:
    run=Path(existing['run'])
    # Explicitly restrict the audit to the two already sealed historical inputs.
    assert run.name in ('visual_complete_all_reserves14','run')
    assert ('visual_complete_all_reserves14' in str(run) or '/load10ms_repeat01/run' in str(run))
    index=load(run/'truth_samples.msgpack.gz.index.json')
    assert (run/'motion_timing.json').exists()
    assert index['blocks'][-1]['end']==(run/'truth_samples.msgpack.gz').stat().st_size
    folder=run/'socket_transport/nut_rotation_continued_02'
    record=load(folder/'nut_rotation_controller_result.json')
    controls=[json.loads(line) for line in (folder/'nut_rotation_control_samples.jsonl').read_text().splitlines()]
    control_map={row['step']:row for row in controls}
    maximum=max(controls,key=lambda row:np.linalg.norm(row['pivot_position_tracking_error_m'][:2]))
    with gzip.open(folder/'joint_ft_samples.json.gz','rt') as stream:ft_rows=json.load(stream)
    ft={row['step']:row for row in ft_rows}
    with gzip.open(Path(record['robot_sensor_history_file']),'rt') as stream:
        for row in json.load(stream):
            if row['step']==controls[0]['step']-1:ft[row['step']]=row
    fields={'step','phase','object_part_positions_m','object_part_orientations_wxyz',
            'native_robot_link_pose_audit','arm_control'}
    cache={}
    def sensed(step):
        if step not in cache:
            rows=list(iter_truth_fields(run,fields,first_step=step,last_step=step))
            assert len(rows)==1 and rows[0]['step']==step
            row=rows[0];p=row['native_robot_link_pose_audit']['poses']['handbase_link']
            cache[step]=(row,transform(p['position_world_m'],p['orientation_world_wxyz']))
        return cache[step]
    first,H0=sensed(controls[0]['step']-1)
    N0=np.asarray(first['object_part_positions_m'][1])
    hand_to_nut=np.linalg.solve(H0,np.r_[N0,1.])
    F=np.asarray(record['hand_from_virtual_nut_axis_frame'])
    install=load(run/'frozen_model_installation.json')
    S=np.asarray(ast.literal_eval(install['pose_mass_velocity_after'][
        '/World/TEVisualHandoff/FixedReceptaclePose']['world_transform']),float).T
    rows=[]
    for expected in existing['rows']:
        c=control_map[expected['control_step']]
        step=c['step']-1;row,H=sensed(step)
        N=np.asarray(row['object_part_positions_m'][1]);P=(H@hand_to_nut)[:3]
        E=np.asarray(c['encoder_pivot_world_m']);error=np.asarray(c['pivot_position_tracking_error_m'])
        T=E+error;native=(H@F)[:3,3]
        A,B,C=T-N,N-P,P-E
        parts={'target_minus_actual_nut':A,'actual_nut_minus_fixed_grasp_prediction':B,
               'initial_calibration_and_native_fk_difference':C}
        computed={
            'reported_xy_error_mm':float(np.linalg.norm(error[:2]))*1000,
            'actual_nut_lateral_to_socket_mm':float(np.linalg.norm((S[:3,:3].T@(N-S[:3,3]))[:2]))*1000,
            'actual_nut_minus_fixed_grasp_prediction_xy_mm':float(np.linalg.norm(B[:2]))*1000,
            'native_vs_encoder_virtual_point_xy_mm':float(np.linalg.norm((native-E)[:2]))*1000,
        }
        errors={name:computed[name]-expected[name] for name in computed}
        assert max(abs(v) for v in errors.values())<1e-9
        for name,value in parts.items():
            assert np.max(np.abs(value*1000-expected['error_decomposition_world_mm'][name]))<1e-9
        # Independent archived encoder-FK reconstruction confirms the preceding-frame pairing.
        prior=ft_transform(ft[step]);after=ft_transform(ft[c['step']])
        fk_prior=(prior@F)[:3,3];fk_after=(after@F)[:3,3]
        residual=np.max(np.abs(fk_prior-E))
        assert residual<1e-10
        assert np.array_equal(np.asarray(ft[c['step']]['active_targets_rad'])[:7],np.asarray(c['nominal_arm_target_rad']))
        delta_local=(np.linalg.solve(H,np.r_[N,1.])-hand_to_nut)[:3]
        assert np.max(np.abs(H[:3,:3]@delta_local-B))<1e-12
        calibration=P-native;native_fk=native-E
        assert np.max(np.abs(calibration+native_fk-C))<1e-12
        rows.append({'control_step':c['step'],'observed_step':step,
            'is_interval_maximum':c['step']==maximum['step'],**computed,
            'difference_from_existing_scalar_table_mm':errors,
            'saved_control_error_world_mm':(error*1000).tolist(),
            'sum_of_three_vectors_world_mm':((A+B+C)*1000).tolist(),
            'vector_identity_max_abs_residual_m':float(np.max(np.abs(A+B+C-error))),
            'preceding_FT_FK_reconstructs_saved_encoder_pivot_max_abs_m':float(residual),
            'same_numbered_poststep_FT_is_not_control_observation_xy_delta_mm':float(np.linalg.norm((fk_after-E)[:2]))*1000,
            'same_step_command_matches_FT_active_target':True,
            'target_minus_nut_xy_mm':float(np.linalg.norm(A[:2]))*1000,
            'fixed_calibration_plus_native_fk_xy_mm':float(np.linalg.norm(C[:2]))*1000,
            'native_initial_center_calibration_xy_mm':float(np.linalg.norm(calibration[:2]))*1000,
            'relative_center_change_is_not_full_relative_orientation_measurement':True,
            'actual_nut_drift_from_initial_socket_xy_mm':float(np.linalg.norm((S[:3,:3].T@(N-N0))[:2]))*1000})
    results.append({'run':str(run),'sealed_sample_count':index['sample_count'],
        'unique_truth_frames_read':sorted(cache),'socket_rotation_equals_world_axes':bool(np.array_equal(S[:3,:3],np.eye(3))),
        'max_control_error_step':maximum['step'],'all_existing_scalar_and_vector_rows_match':True,'rows':rows})
result={'scope':'BOUNDED_POSE_AND_CONTROL_ALIGNMENT_RECHECK_NO_PHYSICS_NO_FULL_SCAN',
        'results':results,'limitations':[
            'Target is reconstructed from saved encoder pivot plus saved error; vector closure alone is algebraic, not independent validation of timing.',
            'All quoted small-distance comparisons are lateral XY metrics. The fixed relation is the Nut center in the initial hand frame, not a separated material-point slip measurement.',
            'Two full episodes have different entry/grip histories; this comparison is not an isolated50ms-vs10ms causal experiment.',
            'Only the existing four selected rows per run (three unique historical frames in full14) are independently recomputed.'
        ]}
with Path(__file__).with_suffix('.json').open('x') as stream:
    json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
