"""Audit an ENDED two-key episode; all physical truth stays in this report."""
import argparse,ast,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src/kcg_connector/isaac'),str(ROOT/'src/kcg_connector/isaac/carts_v2')]
from trace_metadata import read_truth_sample
from two_stage_key_alignment import axial_target


def load(path):return json.loads(Path(path).read_text())


def body_pose(row):
    pose=np.eye(4);pose[:3,3]=row['object_part_positions_m'][0]
    pose[:3,:3]=Rotation.from_quat(np.asarray(row['object_part_orientations_wxyz'][0])[[1,2,3,0]]).as_matrix()
    return pose


def review(run):
    run=Path(run).resolve()
    if 'exit_code' not in load(run.parent/'process.json'):
        raise ValueError('Do not read physical truth before the episode has ended')
    transport=load(run/'socket_transport/transport_and_observation.json')
    anchors=[load(run/name/'camera_and_estimate.json') for name in ('postgrasp_key','postgrasp_key_refined')]
    wrist=load(run/'socket_transport/wrist_socket/camera_and_estimate.json')
    socket_visual=np.array(wrist['measurement']['world_from_receptacle_row_major']).reshape(4,4)
    coarse=next(m for m in transport['motions'] if m['phase']=='key_probe_body_coarse_axial_alignment')
    start=np.array(coarse['plan']['start_world_from_body']);target=np.array(coarse['plan']['target_world_from_body'])
    recomputed,details=axial_target(start,socket_visual)
    declared=transport['coarse_axial_alignment']['signed_rotation_about_body_axis_deg']
    events=[json.loads(line) for line in (run/'four_camera_perception/events.jsonl').read_text().splitlines()]
    refinements=[e for e in events if e['event']=='KEY_ANCHOR_REOBSERVED_AFTER_COARSE_TURN']
    # Independent physical data is read ONLY here, after motion has stopped.
    installation=load(run/'frozen_model_installation.json')
    socket=np.asarray(ast.literal_eval(installation['pose_mass_velocity_after'][
        '/World/TEVisualHandoff/FixedReceptaclePose']['world_transform'])).T
    before=read_truth_sample(run,coarse['execution']['first_step']-1)
    after=read_truth_sample(run,coarse['execution']['last_step']-1)
    actual_before,actual_after=body_pose(before),body_pose(after)
    physical_turn=float(np.degrees(Rotation.from_matrix(actual_after[:3,:3]@actual_before[:3,:3].T).magnitude()))
    observations=[]
    for anchor in anchors:
        row=read_truth_sample(run,anchor['robot_sample_step']);actual=body_pose(row)
        measured=np.array(anchor['key_measurement']['world_from_plug_row_major']).reshape(4,4)
        _,actual_error=axial_target(actual,socket)
        _,vision_error=axial_target(measured,socket_visual)
        observations.append({'index':anchor['key_observation_event_count'],
            'sample_time_s':anchor['physics_time_s'],
            'actual_remaining_axial_correction_deg_POSTRUN_ONLY':actual_error['signed_rotation_about_body_axis_deg'],
            'image_based_remaining_axial_correction_deg':vision_error['signed_rotation_about_body_axis_deg'],
            'measured_key_direction_error_deg_POSTRUN_ONLY':float(np.degrees(np.arccos(np.clip(
                measured[:3,1]@actual[:3,1],-1,1))))})
    frames=[json.loads(line) for line in (run/'video/assembly_five_view_frames.jsonl').read_text().splitlines()]
    video=load(run/'video/assembly_five_view_video.json')
    residual=transport['after_second_observation']['signed_rotation_about_body_axis_deg']
    checks={
        'exactly_two_resolved_key_samples':[a['key_observation_event_count'] for a in anchors]==[1,2]
            and all(a['key_measurement']['key_direction_measured'] for a in anchors),
        'same_fixed_global2_pose':np.allclose(anchors[0]['world_from_camera_cv'],anchors[1]['world_from_camera_cv'],rtol=0,atol=1e-12)
            and anchors[0]['capture']['camera_path']==anchors[1]['capture']['camera_path'],
        'coarse_turn_after_both_measurements':coarse['execution']['first_step']>wrist['sample_step']
            and coarse['execution']['first_step']>=anchors[0]['observation_latency']['availability_step'],
        'second_key_sample_after_physical_coarse_turn':anchors[1]['robot_sample_step']>=coarse['execution']['last_step']-1,
        'positive_physical_delay_for_each_key_result':all(a['observation_latency']['physics_hold_steps']>0
            and a['observation_latency']['availability_step']>a['robot_sample_step'] for a in anchors),
        'coarse_target_recomputed_from_visual_key_and_slot':bool(np.allclose(recomputed,target,atol=1e-9,rtol=0))
            and abs(details['signed_rotation_about_body_axis_deg']-declared)<1e-9,
        'coarse_target_preserves_body_center_and_axis':bool(np.allclose(start[:3,3],target[:3,3],rtol=0,atol=1e-12)
            and np.allclose(start[:3,2],target[:3,2],rtol=0,atol=1e-12)),
        'perturbed_case_requires_at_least_ten_degree_correction':abs(declared)>=10.,
        'physical_body_actually_turns_at_least_ten_degrees':physical_turn>=10.,
        'fresh_second_key_anchor_consumed':len(refinements)==1 and refinements[0]['sample_time_s']==anchors[1]['physics_time_s']
            and refinements[0]['consumed_time_s']>=anchors[1]['observation_latency']['availability_physics_time_s'],
        'remaining_small_motion_admission':abs(residual)<=1.,
        'five_recorded_views':video['view_count']==5 and set(video['camera_paths'])=={'main','global_1','global_2','palm','wrist'},
        'video_render_does_not_change_native_state':all(v==0 for v in video['maximum_render_native_state_deltas'].values()),
        'recorded_overlay_includes_both_measurement_stages':{1,2}.issubset({f.get('online_visual_and_encoder_status',{}).get('observation_count') for f in frames}),
    }
    entry=None
    entry_path=run/'socket_transport/key_entry/key_entry_controller_result.json'
    if entry_path.exists():
        step=load(entry_path)['contact_first_step'];actual=body_pose(read_truth_sample(run,step))
        _,entry=axial_target(actual,socket)
        entry={'step':step,'remaining_axial_correction_deg_POSTRUN_ONLY':entry['signed_rotation_about_body_axis_deg']}
    result={'scope':'ENDED_TWO_VISUAL_KEY_OBSERVATIONS_AND_PHYSICAL_COARSE_TURN',
        'passed':all(checks.values()),'checks':checks,
        'source_commit':load(run.parent/'process.json')['source_commit'],
        'online_observed_coarse_rotation_deg':declared,'online_remaining_refinement_deg':residual,
        'camera_paths':video['camera_paths'],'observations':observations,
        'second_anchor_update':refinements,
        'truth_only_postrun':{'socket_world_yaw_deg':float(np.degrees(np.arctan2(socket[1,0],socket[0,0]))),
            'actual_body_rotation_during_coarse_stage_deg':physical_turn,
            'actual_joint7_delta_deg':float(np.degrees(after['active_positions_rad'][6]-before['active_positions_rad'][6])),
            'coarse_stage_body_center_displacement_m':float(np.linalg.norm(actual_after[:3,3]-actual_before[:3,3])),
            'entry':entry},
        'postrun_truth_returned_to_control':False,'physical_full_assembly_requires_separate_original_acceptance':True}
    (run/'two_key_alignment_review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run');review(p.parse_args().run)
