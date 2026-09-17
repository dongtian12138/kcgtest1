"""Ended-run decomposition of fixed grasp-reference error, without slip claims."""
import json,ast,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path('/home/noob/WorkPlace/kcgtest1-improvements-20260916')
sys.path[:0]=[str(ROOT/'src/kcg_connector'),str(ROOT/'src/kcg_connector/isaac')]
from trace_metadata import read_truth_sample,without_cyclic_gc

def review(run):
    run=Path(run);idx=json.loads((run/'truth_samples.msgpack.gz.index.json').read_text());assert (run/'motion_timing.json').exists() and idx['blocks'][-1]['end']==(run/'truth_samples.msgpack.gz').stat().st_size
    folder=run/'socket_transport/nut_rotation_continued_02';record=json.loads((folder/'nut_rotation_controller_result.json').read_text());controls=[json.loads(s) for s in (folder/'nut_rotation_control_samples.jsonl').read_text().splitlines()]
    def sensed(step):
        row=without_cyclic_gc(read_truth_sample,run,step);hand=row['native_robot_link_pose_audit']['poses']['handbase_link'];H=np.eye(4);H[:3,3]=hand['position_world_m'];H[:3,:3]=Rotation.from_quat(np.asarray(hand['orientation_world_wxyz'])[[1,2,3,0]]).as_matrix();return row,H
    first,H0=sensed(controls[0]['step']-1);N0=np.asarray(first['object_part_positions_m'][1]);hand_to_initial_nut=np.linalg.inv(H0)@np.r_[N0,1.]
    installation=json.loads((run/'frozen_model_installation.json').read_text());S=np.asarray(ast.literal_eval(installation['pose_mass_velocity_after']['/World/TEVisualHandoff/FixedReceptaclePose']['world_transform']),float).T
    hand_to_virtual=np.asarray(record['hand_from_virtual_nut_axis_frame']);maximum=max(controls,key=lambda r:np.linalg.norm(r['pivot_position_tracking_error_m'][:2]));selected=[controls[0],next(r for r in controls if abs(r['commanded_rotation_deg'])>=20),maximum,controls[-1]];out=[]
    initial_angles=np.asarray([first['arm_control']['hand_joint_diagnostic']['joints'][j]['position_rad'] for j in ['f1j2','f2j1','f3j2']])
    for control in selected:
        row,H=sensed(control['step']-1);N=np.asarray(row['object_part_positions_m'][1]);prediction=(H@hand_to_initial_nut)[:3];encoder=np.asarray(control['encoder_pivot_world_m']);error=np.asarray(control['pivot_position_tracking_error_m']);target=encoder+error;native_virtual=(H@hand_to_virtual)[:3,3]
        vectors=[target-N,N-prediction,prediction-encoder];assert np.max(np.abs(sum(vectors)-error))<1e-12
        angles=np.asarray([row['arm_control']['hand_joint_diagnostic']['joints'][j]['position_rad'] for j in ['f1j2','f2j1','f3j2']])
        out.append({'control_step':control['step'],'physical_observation_step':row['step'],'command_deg':abs(control['commanded_rotation_deg']),'reported_xy_error_mm':float(np.linalg.norm(error[:2]))*1000,'actual_nut_lateral_to_socket_mm':float(np.linalg.norm((S[:3,:3].T@(N-S[:3,3]))[:2]))*1000,'actual_nut_minus_fixed_grasp_prediction_xy_mm':float(np.linalg.norm((N-prediction)[:2]))*1000,'native_vs_encoder_virtual_point_xy_mm':float(np.linalg.norm((native_virtual-encoder)[:2]))*1000,'error_decomposition_world_mm':{'target_minus_actual_nut':(vectors[0]*1000).tolist(),'actual_nut_minus_fixed_grasp_prediction':(vectors[1]*1000).tolist(),'initial_calibration_and_native_fk_difference':(vectors[2]*1000).tolist()},'finger_output_angle_change_deg':np.degrees(angles-initial_angles).tolist()})
    return {'run':str(run),'scope':'FOURTH_STROKE_ENDED_POSE_DECOMPOSITION','control_rows_match_preceding_observed_physics_step':True,'initial_actual_hand_nut_relation_used_for_offline_comparison_only':True,'rows':out,'limitations':['The changing hand/Nut relation combines finger joint motion, structural/solver compliance, rolling and sliding; these components are not separated here.','Native-vs-FK is a positional comparison, not a manufacturer sensor accuracy claim.','This is not an online controller or a proof that increasing grip force resolves assembly.']}
results=[review('/home/noob/WorkPlace/kcgtest1/artifacts/full_rotation_20260914/visual_integration/visual_complete_all_reserves14'),review(ROOT/'artifacts/full_validation/load10ms_repeat01/run')]
output=Path(__file__).with_name('comparison.json');output.write_text(json.dumps(results,indent=2)+'\n')
for result in results:
 print(result['run'])
 print(json.dumps([{k:v for k,v in r.items() if k!='error_decomposition_world_mm'} for r in result['rows']],indent=2))
