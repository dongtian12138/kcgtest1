"""Read every original raw frame of the final three-second released hold."""
import argparse,ast,json,sys
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
root=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(root/'src/kcg_connector/isaac'),str(root/'src/kcg_connector/isaac/carts_v2')]
from trace_metadata import iter_truth_fields

def review(run,output=None):
 run=Path(run).resolve()
 terminal=json.loads((run/'socket_transport/nut_terminal_release/nut_reindex_controller_result.json').read_text())
 meta=json.loads((run/'trace_metadata.json').read_text());dt=float(meta['physics_dt_s'])
 first=int(terminal['opening_last_step']);end=int(terminal['support_hold_last_step'])
 install=json.loads((run/'frozen_model_installation.json').read_text())
 socket=np.array(ast.literal_eval(install['pose_mass_velocity_after']['/World/TEVisualHandoff/FixedReceptaclePose']['world_transform'])).T
 rows=[];clip_sets=set();coverage=Counter();sleeping=Counter();phases=Counter();max_backup=0.
 fields=('phase','contacts','object_part_positions_m','object_part_orientations_wxyz','object_part_sleeping')
 for row in iter_truth_fields(run,fields,first_step=first,last_step=end-1):
  contact=row['contacts'];phases[row['phase']]+=1
  coverage['all_provided_native_points_retained']+=int(contact['all_provided_native_report_points_retained'] is True)
  coverage['channels_agree']+=int(contact['contact_report_channels_agree'] is True)
  coverage['one_physics_callback']+=int(contact['physics_step_callback_count']==1)
  body_rotation=Rotation.from_quat(np.roll(row['object_part_orientations_wxyz'][0],-1)).as_matrix()
  positions=np.array(row['object_part_positions_m']);depth=-float(socket[:3,2]@(positions[0]-socket[:3,3]))
  backup=abs(float((body_rotation.T@(positions[1]-positions[0]))[2]));max_backup=max(max_backup,backup)
  sleepers=tuple(map(bool,row['object_part_sleeping']));sleeping[str(sleepers)]+=1
  hand_impulse=0.;clips=set();stops=0
  for header in contact['poll_headers']:
   paths=header['paths'];impulse=sum(float(np.linalg.norm(point['impulse_n_s'])) for point in header['contacts'])
   if any('/handbase_link' in path for path in paths[:2]):hand_impulse+=impulse
   if impulse<=1e-10:continue
   if any('FixedReceptaclePose' in path for path in paths[:2]) and any('TE_J35FreeSplitPlug' in path for path in paths[:2]):
    stops+=int(sum('SourceMetalStopBox' in path for path in paths[2:])==2)
    for path in paths[2:]:
     if '/Leaf_' in path:clips.add(path.split('/Leaf_')[0])
  clip_sets.add(tuple(sorted(clips)))
  rows.append({'step':row['step'],'body_depth_m':depth,'hand_raw_impulse_norm_sum_n_s':hand_impulse,
               'source_stop_positive_reports':stops,'loaded_clip_count':len(clips),'body_and_nut_awake':not any(sleepers)})
 count=len(rows);depth=np.array([row['body_depth_m'] for row in rows])
 checks={'completed_terminal_release':terminal.get('completed') is True,
         'actual_full_three_second_window':count==end-first and count*dt>=3.-dt/2,
         'all_frames_are_open_hold':set(phases)=={'nut_index_free_open_hold'},
         'raw_hand_impulse_exactly_zero_every_frame':all(row['hand_raw_impulse_norm_sum_n_s']==0 for row in rows),
         'source_stop_positive_every_frame':all(row['source_stop_positive_reports']>0 for row in rows),
         'same128_clips_loaded_every_frame':len(clip_sets)==1 and all(row['loaded_clip_count']==128 for row in rows),
         'body_and_nut_awake_every_frame':all(row['body_and_nut_awake'] for row in rows),
         'original_source_stop_depth_tolerance':bool(np.max(abs(depth-.014605))<=.00001),
         'backup_limit_not_reached_in_this_window':max_backup<.0005999,
         'raw_report_coverage_complete':all(value==count for value in coverage.values())}
 result={'scope':'POSTRUN_ORIGINAL_THREE_SECOND_RELEASE_CONDITIONS','accepted':all(checks.values()),
         'run':str(run),'reviewed_raw_range_half_open':[first,end],'sample_count':count,
         'duration_s':count*dt,'conditions':checks,'phase_counts':dict(phases),
         'raw_report_coverage':dict(coverage),'sleeping_state_counts':dict(sleeping),
         'depth_range_m':[float(depth.min()),float(depth.max())],
         'maximum_backup_coordinate_m_in_window':max_backup,'distinct_loaded_clip_sets':len(clip_sets),
         'online_control_used':False,'acceptance_tolerances_changed':False,
         'full_assembly_requires_other_source_geometry_contact_and_vision_reviews':True}
 destination=Path(output) if output else run/'three_second_release_review.json'
 destination.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--output');a=p.parse_args();review(a.run,a.output)
