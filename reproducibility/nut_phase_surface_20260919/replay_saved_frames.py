"""Compare ended episodes' unchanged images; never supply truth to the estimator."""
import argparse,json,time,sys
from pathlib import Path
import numpy as np
import yaml
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src/kcg_connector/isaac'),str(ROOT/'src/kcg_connector/isaac/carts_v2')]
from te_nut_phase_vision import _source_template,extract_points,fit_phase,grasp_phase_error_deg
from te_body_socket_observation import hand_camera_mount

def replay(runs,output):
 cfg=yaml.safe_load((ROOT/'reproducibility/two_key_20260919/assembly_high_global1_plus20.yaml').read_text())
 settings=cfg['nut_regrasp']['visual_nut_phase'];cache={};surface=_source_template(ROOT,settings,cache)
 surface.query([[.022,0.,-.02]])  # Separate first kernel setup from per-image timing.
 K=np.array(hand_camera_mount(ROOT,'palm')['intrinsics_3x3']);rows=[]
 for run in runs:
  run=Path(run).resolve();assert 'exit_code' in json.loads((run.parent/'process.json').read_text())
  socket=np.array(json.loads((run/'socket_transport/wrist_socket/camera_and_estimate.json').read_text())['measurement']['world_from_receptacle_row_major']).reshape(4,4)
  for record in sorted((run/'socket_transport').glob('nut_regrasp*/*controller_result.json')):
   d=json.loads(record.read_text())
   for observation_key,phase_key in [('pre_phase_alignment_observation','visual_nut_phase_before_alignment'),('phase_aligned_palm_observation','visual_nut_phase_after_alignment')]:
    if phase_key not in d:continue
    obs=d[observation_key];old=d[phase_key];frame=Path(obs['rgbd_directory']) if obs.get('rgbd_directory') else Path(obs['tracking_seed_mask']).parent/'rgbd'
    recorded=json.loads((Path(obs['tracking_seed_mask']).parent/'observation.json').read_text())
    started=time.perf_counter();points,R,t=extract_points(obs,np.load(frame/'depth_m.npy'),K,socket,offset_m=cfg['nut_regrasp']['captured_nut_offset_m']);fit=fit_phase(surface,points,old['phase_fit']['prior_deg'],settings['search_half_width_deg']);elapsed=time.perf_counter()-started
    ratio=min(x['clipped_rms_m'] for x in fit['three_degree_alternatives'])/max(fit['best']['clipped_rms_m'],1e-12)
    quality=len(points)>=500 and abs(fit['best']['yaw_deg']-fit['prior_deg'])<settings['search_half_width_deg']-.4 and fit['best']['clipped_rms_m']<=settings['maximum_fit_rms_m'] and ratio>=settings['minimum_three_degree_score_ratio']
    phase={**old,'phase_fit':fit,'estimated_world_from_nut_rotation':(R@Rotation.from_euler('z',fit['best']['yaw_deg'],degrees=True).as_matrix()).tolist(),'quality_passed':bool(quality)}
    rows.append({'run':str(run),'stage':record.parent.name,'observation':observation_key,'sample_step':recorded['sample_step'],'frame':str(frame),'old_best':old['phase_fit']['best'],'surface_best':fit['best'],'surface_quality_passed':bool(quality),'surface_three_degree_score_ratio':ratio,'old_hand_error_at_same_sample_deg':grasp_phase_error_deg(recorded['hand'],old),'surface_hand_error_at_same_sample_deg':grasp_phase_error_deg(recorded['hand'],phase),'surface_depth_and_fit_wall_s':elapsed,'distance_metric':surface.method})
 result={'scope':'ENDED_FIXED_IMAGE_REPLAY_NOT_NEW_PHYSICAL_ASSEMBLY','estimator_input_sources':['RECORDED_DEPTH','RECORDED_CAMERA_CALIBRATION_AND_ENCODERS','IMAGE_BASED_BODY_POSITION_AND_AXIS','ORIGINAL_SOURCE_CAD'],'truth_or_contacts_input_to_estimator':False,'thresholds_changed':False,'rows':rows,'all_original_quality_gates_passed':all(r['surface_quality_passed'] for r in rows),'frame_count':len(rows),'median_per_image_wall_s':float(np.median([r['surface_depth_and_fit_wall_s'] for r in rows]))}
 Path(output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
 for row in rows:print(Path(row['run']).parent.name,row['stage'],row['observation'],row['old_best']['yaw_deg'],row['surface_best']['yaw_deg'],row['surface_hand_error_at_same_sample_deg'],row['surface_quality_passed'])
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('runs',nargs='+',type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();replay(a.runs,a.output)
