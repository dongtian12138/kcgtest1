"""Ended high-Global1 coarse localization / grasp-prefix review, not assembly."""
import argparse,ast,json,subprocess
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]


def load(path):return json.loads(Path(path).read_text())


def review(run):
    import yaml
    run=Path(run).resolve();process=load(run.parent/'process.json')
    if 'exit_code' not in process:raise ValueError('Ended initial-grasp episode required')
    initial=load(run/'initial_rgbd/consumed_grasp_plan.json')
    provider=load(run/'initial_rgbd/provider_input.json')
    observation=load(run/'initial_rgbd/pose_provider_result.json')
    socket=load(run/'global1_socket/camera_and_estimate.json')
    video=load(run/'video/assembly_five_view_video.json')
    grasp=load(run/'source_nail_body_review.json')
    meta=load(run/'trace_metadata.json');evaluation=load(run/'evaluation.json')
    commit=process['source_commit']
    def source(path):return yaml.safe_load(subprocess.check_output(['git','show',f'{commit}:{path}'],cwd=ROOT,text=True))
    argv=process['argv'];config=source(argv[argv.index('--body-assembly-collision-config')+1])
    rig=source(config['perception']['four_camera_rig']);camera=source(rig['global_1']['source_config'])['camera']
    pose=np.array(provider['camera_calibration']['world_from_camera_cv_row_major']).reshape(4,4)
    checks={
        'declared_high_camera_used_for_initial_image':initial['capture']['camera_path']==camera['prim_path']
            and rig['global_1']['source_config']=='src/kcg_connector/config/te_rgbd_camera_global_e50_v1.yaml',
        'recorded_camera_at_high_installation':bool(np.allclose(pose[:3,3],camera['eye_world_m'],atol=1e-12,rtol=0)),
        'video_global1_is_the_same_camera':video['camera_paths']['global_1']==initial['capture']['camera_path'],
        'same_image_used_for_both_parts':socket['same_frame_as_initial_plug_localization'] is True
            and Path(socket['rgbd_directory'])==run/'initial_rgbd/observation',
        'visible_rear_face_method_used':initial['initial_plug_pose_method']=='VISIBLE_BODY_REAR_FACE'
            and observation['status']=='OBSERVED_AXIS_POSITION_YAW_FREE',
        'unobserved_key_yaw_not_claimed':initial['key_angle_measured'] is False
            and socket['socket_key_yaw_measured'] is False,
        'current_image_plan_consumed_causally':initial['availability']['measurement_not_used_for_motion_during_delay'] is True
            and initial['controller_start_step']>initial['robot_sample_step'],
        'actual_original_nail_grasp_lift_hold_verified':grasp['accepted'] is True,
        'controller_completed':meta['controller_outcome']['completed'] is True
            and meta['controller_outcome']['failure_reason'] is None,
        'engine_identity_and_finiteness':all(evaluation[k] for k in ('engine_health_pass','identity_hash_check_pass','finite_throughout')),
        'no_object_pose_writes_or_online_truth':meta['object_pose_writes_after_start']==0
            and meta['online_object_or_contact_truth_used'] is False and meta['truth_audit_data_returned_to_controller'] is False,
        'video_did_not_change_native_state':all(v==0 for v in video['maximum_render_native_state_deltas'].values()),
    }
    installed=load(run/'frozen_model_installation.json')
    truth=np.asarray(ast.literal_eval(installed['pose_mass_velocity_after'][
        '/World/TEVisualHandoff/FixedReceptaclePose']['world_transform'])).T
    measured=np.asarray(socket['world_from_socket_coarse'])
    body_truth_review=load(run/'initial_rgbd/posthoc_truth_comparison.json')
    result={'scope':'HIGH_GLOBAL1_CURRENT_IMAGE_TO_ACTUAL_INITIAL_GRASP_LIFT_HOLD_ONLY',
        'accepted':all(checks.values()),'checks':checks,'source_commit':commit,
        'camera_path':camera['prim_path'],'camera_eye_world_m':camera['eye_world_m'],
        'postrun_only_pose_errors':{'body_position_m':body_truth_review['center_error_m'],
            'body_axis_deg':body_truth_review['axis_error_deg'],
            'socket_position_m':float(np.linalg.norm(measured[:3,3]-truth[:3,3])),
            'socket_axis_deg':float(np.degrees(np.arccos(np.clip(measured[:3,2]@truth[:3,2],-1,1))))},
        'minimum_actual_lift_during_hold_m':grasp['minimum_body_lift_during_hold_m'],
        'hold_duration_s':grasp['hold_duration_s'],
        'three_original_nails_body_contact_fraction':grasp['all_three_nails_body_contact_fraction_in_hold'],
        'video_path':video['path'],'video_duration_s':video['duration_s'],'wall_s':process['wall_s'],
        'high_global1_full_assembly_verified':False,'postrun_truth_returned_to_control':False}
    (run/'high_global1_initial_review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run');review(p.parse_args().run)
