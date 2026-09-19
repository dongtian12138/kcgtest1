"""Independent whole-episode mechanical review, never an online controller."""
from __future__ import annotations

import argparse
import ast
from collections import Counter,deque
import json
from math import hypot
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from trace_metadata import iter_truth_samples


def terminal_sensor_decision_review(directory, transport):
    from evaluate_terminal_completion_sensors import review as review_completion
    completion=review_completion(directory)
    return {
        'path':str(Path(directory)/'online_completion_sensor_replay.json'),
        'accepted':completion['accepted'],
        'historical_runtime_online_seating_confirmed':completion['historical_runtime_online_seating_confirmed'],
        'validation_mode':'REPLAY_OF_ALREADY_CONSUMED_ONLINE_SENSORS_AFTER_ALL_MOTION',
        'new_terminal_logic_executed_in_recorded_physics_run':
            'online_completion_sampling_contract' in transport}


def finalize_conditions(result):
    result['complete_visual_assembly_verified']=bool(result['continuous_step_sequence']
        and all(result['mechanical_conditions'].values())
        and all(result['visual_stage_conditions'].values())
        and all(result['additional_review_conditions'].values()))
    result['status']='VERIFIED' if result['complete_visual_assembly_verified'] else 'REVIEW_REQUIRED'


def refresh_terminal_decision(directory, original_review):
    """Reuse this run's completed raw audit when only the terminal predicate changed.

    The original audit remains separate and is bound by its digest. No contact,
    geometry, sequence, or image-inspection result is recomputed or replaced.
    """
    import hashlib
    directory=Path(directory).resolve();original_review=Path(original_review).resolve()
    if original_review.parent!=directory or original_review.name=='whole_assembly_review.json':
        raise ValueError('A separately preserved audit from this exact run is required')
    raw=original_review.read_bytes();result=json.loads(raw)
    terminal=json.loads((directory/'socket_transport/nut_terminal_release/nut_reindex_controller_result.json').read_text())
    if (result['scope']!='POSTRUN_SAME_EPISODE_VISUAL_ASSEMBLY_REVIEW'
            or result['status'] not in ('VERIFIED','REVIEW_REQUIRED')
            or result['sample_count']!=terminal['support_hold_last_step']
            or result['final']['step']!=terminal['support_hold_last_step']-1):
        raise ValueError('The preserved full audit does not cover this completed episode')
    transport=json.loads((directory/'socket_transport/transport_and_observation.json').read_text())
    completion=terminal_sensor_decision_review(directory,transport)
    visual=result['visual_stage_conditions']
    historical=visual.pop('online_seating_confirmed')
    if historical is not completion['historical_runtime_online_seating_confirmed']:
        raise ValueError('Historical controller outcome and preserved review differ')
    visual['released_window_sensor_seating_confirmed']=completion['accepted'] is True
    result['terminal_sensor_decision_review']=completion
    result['unchanged_physical_audit_basis']={
        'path':str(original_review),'sha256':hashlib.sha256(raw).hexdigest(),
        'all_mechanical_geometry_sequence_and_visibility_conditions_preserved':True,
        'only_revised_condition':'FINAL_SENSOR_STABILITY_USES_THE_RELEASED_WINDOW'}
    finalize_conditions(result)
    (directory/'whole_assembly_review.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def review(directory):
    directory=Path(directory)
    terminal_path=directory/'socket_transport/nut_terminal_release/nut_reindex_controller_result.json'
    result={'scope':'POSTRUN_SAME_EPISODE_VISUAL_ASSEMBLY_REVIEW','online_control_used':False,
        'complete_visual_assembly_verified':False,'required_final_observation_s':2.,
        'nominal_source_stop_depth_m':.014605,'source_stop_depth_review_tolerance_m':.00001,
        'thrust_backup_limit_m':.0005999}
    if not terminal_path.exists():
        result['status']='INCOMPLETE_NO_TERMINAL_RELEASE_RECORD'
        (directory/'whole_assembly_review.json').write_text(json.dumps(result,indent=2)+'\n')
        return result
    terminal=json.loads(terminal_path.read_text())
    if not terminal.get('completed'):
        result.update(status='INCOMPLETE_TERMINAL_RELEASE_NOT_COMPLETED',terminal_release=terminal)
        (directory/'whole_assembly_review.json').write_text(json.dumps(result,indent=2)+'\n')
        return result
    metadata=json.loads((directory/'trace_metadata.json').read_text())
    installation=json.loads((directory/'frozen_model_installation.json').read_text())
    socket_path='/World/TEVisualHandoff/FixedReceptaclePose'
    socket=np.asarray(ast.literal_eval(installation['pose_mass_velocity_after'][socket_path]['world_transform']),float).T
    if socket.shape!=(4,4) or not np.allclose(socket[3],[0,0,0,1]):
        raise ValueError('The preserved static socket transform is malformed')
    dt=float(metadata['physics_dt_s']);count=round(2./dt)
    end_step=int(terminal['support_hold_last_step'])-1;start_step=end_step-count+1
    window=[];phase_counts=Counter();all_clips=set();stop_steps=[];maximum_backup=0.
    body_first=None;last=None;expected_step=0;sequence_complete=True
    angle_rows=[];stage_order=[];previous_phase=None;support_rows=[];nut_pairs=set()
    for row in iter_truth_samples(directory):
        step=int(row['step']);sequence_complete&=step==expected_step;expected_step=step+1
        phase=row['phase'];phase_counts[phase]+=1
        if phase!=previous_phase:stage_order.append({'phase':phase,'step':step});previous_phase=phase
        pos=np.asarray(row['object_part_positions_m']);quats=np.asarray(row['object_part_orientations_wxyz'])
        rotations=Rotation.from_quat(quats[:,[1,2,3,0]]).as_matrix()
        local=(pos-socket[:3,3])@socket[:3,:3]
        body_rotation=socket[:3,:3].T@rotations[0]
        axial_offset=float((rotations[0].T@(pos[1]-pos[0]))[2])
        maximum_backup=max(maximum_backup,abs(axial_offset))
        relative_rotation=rotations[0].T@rotations[1]
        if 'nut_rotation_turn' in phase:
            angle_rows.append((step,float(np.arctan2(relative_rotation[1,0],relative_rotation[0,0])),float(-local[0,2])))
        hand_impulse=0.;loaded_clips=set();stop=False;bad_sector=False
        for header in row['contacts']['poll_headers']:
            paths=header['paths'];impulse=sum(hypot(*c['impulse_n_s']) for c in header['contacts'])
            hand=any('/handbase_link' in p for p in paths[:2])
            # Release means no recorded hand impulse, including values below
            # the threshold used to identify positively loaded internal parts.
            if hand:hand_impulse+=impulse
            if impulse<=1e-10:continue
            fixed=any('FixedReceptaclePose' in p for p in paths[:2])
            plug=any('TE_J35FreeSplitPlug' in p for p in paths[:2])
            if hand and 'key_probe_nut' in phase:
                nut_pairs.add(tuple(paths[:2]))
            if plug and fixed:
                stop|=sum('SourceMetalStopBox' in path for path in paths[2:])==2
                for path in paths[2:]:
                    if '/Leaf_' in path:loaded_clips.add(path.split('/Leaf_')[0])
                    if 'SocketSector_' in path:bad_sector=True
        all_clips.update(loaded_clips)
        if stop:stop_steps.append(step)
        entry={'step':step,'phase':phase,'body_depth_m':float(-local[0,2]),
            'nut_depth_m':float(-local[1,2]),'body_lateral_m':float(np.linalg.norm(local[0,:2])),
            'body_axis_tilt_deg':float(np.degrees(np.arccos(np.clip(-body_rotation[2,2],-1,1)))),
            'thrust_coordinate_m':axial_offset,'hand_positive_normal_impulse_n_s':hand_impulse,
            'loaded_clip_count':len(loaded_clips),'stop_contact':stop,'artificial_sector_contact':bad_sector}
        if body_first is None:body_first=entry
        if phase=='key_probe_body_support_hold':support_rows.append(entry)
        if start_step<=step<=end_step:window.append(entry)
        last=entry
    result.update(sample_count=expected_step,continuous_step_sequence=bool(sequence_complete),
        phase_counts=dict(phase_counts),stage_order=stage_order,first=body_first,final=last,
        final_release_window_sample_count=len(window),source_stop_positive_contact_samples=len(stop_steps),
        maximum_absolute_thrust_coordinate_m=maximum_backup,loaded_clip_union_count=len(all_clips),
        nut_stage_hand_contact_actor_pairs=[list(x) for x in sorted(nut_pairs)])
    if len(window)!=count:
        result['status']='INCOMPLETE_FINAL_RELEASE_WINDOW'
    else:
        depths=np.array([r['body_depth_m'] for r in window]);stop_gaps=.014605-depths
        mechanics={'final_hand_detached':all(r['hand_positive_normal_impulse_n_s']==0 for r in window),
            'full_two_second_window':len(window)*dt>=2.-dt/2,
            'source_stop_contact_in_final_window':any(r['stop_contact'] for r in window),
            'body_at_source_stop':bool(np.max(np.abs(stop_gaps))<=.00001),
            'all128_sockets_simultaneously_loaded':max(r['loaded_clip_count'] for r in window)==128,
            'no_artificial_sector_contact_in_final_window':not any(r['artificial_sector_contact'] for r in window),
            'thrust_backup_limit_not_reached':maximum_backup<.0005999,
            'support_had_actual_hand_separation':bool(support_rows and all(r['hand_positive_normal_impulse_n_s']==0 for r in support_rows))}
        result.update(mechanical_conditions=mechanics,final_body_depth_range_m=[float(depths.min()),float(depths.max())],
            final_body_axial_range_m=float(np.ptp(depths)),final_source_stop_gap_range_m=[float(stop_gaps.min()),float(stop_gaps.max())],
            final_window_maximum_simultaneous_loaded_clips=max(r['loaded_clip_count'] for r in window))
        initial=directory/'initial_rgbd/consumed_grasp_plan.json';held=directory/'postgrasp_key/camera_and_estimate.json'
        transport=json.loads((directory/'socket_transport/transport_and_observation.json').read_text())
        visual={'fresh_image_plan_consumed':initial.exists(),
            'current_held_body_key_observed':held.exists() and json.loads(held.read_text())['key_measurement']['key_direction_measured'],
            'current_wrist_socket_observed':bool(transport.get('wrist_socket_observation_executed'))}
        if 'key_observation_event_count' in transport:
            camera_review=directory/'four_camera_contract_review.json'
            session_path=directory/'four_camera_perception/summary.json'
            session=json.loads(session_path.read_text()) if session_path.exists() else {}
            memory=session.get('key_memory',{})
            # Re-evaluate the final sensor predicate separately from physical
            # acceptance. Preserve and expose a historical false negative;
            # never rewrite the recorded controller outcome to claim a rerun.
            completion=terminal_sensor_decision_review(directory,transport)
            result['terminal_sensor_decision_review']=completion
            visual.update(
                fixed_four_camera_records_verified=(camera_review.exists()
                    and json.loads(camera_review.read_text()).get('passed') is True),
                palm_position_and_axis_updates=int(memory.get('palm_update_count',0))>=2,
                body_grasp_reference_retired=(memory.get('active_body_grasp') is False
                    and memory.get('retirement_reason')=='BODY_RELEASED_AFTER_GUIDED_ENTRY'),
                released_window_sensor_seating_confirmed=completion['accepted'] is True)
            if transport['key_observation_event_count']==2:
                two_key_review=directory/'two_key_alignment_review.json'
                visual['two_stage_key_observations']=(transport.get('body_key_reobservations_after_memory')==1
                    and memory.get('anchor_count')==2 and two_key_review.exists()
                    and json.loads(two_key_review.read_text()).get('passed') is True)
            else:
                visual['single_key_observation']=(transport['key_observation_event_count']==1
                    and transport.get('body_key_reobservations_after_memory')==0)
        else:
            visual['body_key_reobserved_after_carry']=transport.get('body_key_reobservations_after_memory',0)>0
        result['visual_stage_conditions']=visual
        # Source face identification and camera visibility remain separately
        # auditable requirements. A generic contact or angle is insufficient.
        required={'nail_body_source_faces':directory/'source_nail_body_review.json',
                  'nut_pad_source_faces':directory/'source_nut_pad_review.json',
                  'final_red_band_visibility':directory/'final_mating_visibility_review.json',
                  'source_key_containment':directory/'source_key_containment_review.json'}
        result['required_additional_reviews']={k:str(v) for k,v in required.items()}
        external={k:(json.loads(v.read_text()).get('accepted') is True if v.exists() else False) for k,v in required.items()}
        result['additional_review_conditions']=external
        finalize_conditions(result)
    if angle_rows:
        angles=np.degrees(np.unwrap(np.array(angle_rows)[:,1]));result['recorded_turn_interval_relative_nut_rotation_range_deg']=float(np.ptp(angles))
        result['turn_angles_are_measured_pose_results_not_controller_commands']=True
    (directory/'whole_assembly_review.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path);args=parser.parse_args()
    print(json.dumps(review(args.directory),indent=2))
