"""Audit recorded perception paths, fixed mounts and causal image consumption."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
import yaml
root=Path(__file__).resolve().parents[2]

def installed_pose(spec):
 e=np.array(spec['eye_world_m'],float);z=np.array(spec['target_world_m'],float)-e;z/=np.linalg.norm(z)
 x=np.cross(z,[0.,0.,1.]);x/=np.linalg.norm(x);y=np.cross(z,x)
 p=np.eye(4);p[:3,:3]=np.column_stack((x,y,z));p[:3,3]=e;return p

def main(run):
 run=Path(run).resolve();process=json.loads((run.parent/'process.json').read_text())
 if 'exit_code' not in process:raise ValueError('Audit only an ended episode')
 commit=process['source_commit']
 def source(path):return yaml.safe_load(subprocess.check_output(['git','show',f'{commit}:{path}'],cwd=root,text=True))
 rig=source('src/kcg_connector/config/four_camera_assembly_20260918.yaml')
 g1=source(rig['global_1']['source_config'])['camera'];mounts=source(rig['mount_source_config'])['camera_rig']
 expected={'global_1':g1['prim_path'],**{role:rig[role]['prim_path'] for role in ('global_2','palm','wrist')}}
 actual={};violations=[];mount_error={'palm':0.,'wrist':0.}
 a=json.loads((run/'postgrasp_key/camera_and_estimate.json').read_text())
 initial=json.loads((run/'initial_rgbd/body_localization.json').read_text())
 actual['global_1']={initial['capture']['camera_path']};actual['global_2']={a['capture']['camera_path']}
 if not np.allclose(np.asarray(a['world_from_camera_cv']),installed_pose(rig['global_2']),atol=1e-12,rtol=0):violations.append('Global2 extrinsics differ from episode installation')
 provider=json.loads((run/'initial_rgbd/provider_input.json').read_text())
 if not np.allclose(np.array(provider['camera_calibration']['world_from_camera_cv_row_major']).reshape(4,4),installed_pose(g1),atol=1e-12,rtol=0):violations.append('Global1 extrinsics differ from episode installation')
 socket=json.loads((run/'global1_socket/camera_and_estimate.json').read_text())
 if Path(socket['rgbd_directory'])!=run/'initial_rgbd/observation':violations.append('Socket coarse localization did not reuse the initial Global1 frame')
 records=[('palm',a['palm_observation']['capture'],a['world_from_hand_encoder'],a['palm_observation']['world_from_camera_cv'])]
 for p in sorted((run/'four_camera_perception').glob('palm_*/observation.json')):
  d=json.loads(p.read_text());records.append(('palm',d['capture'],d['hand'],d['camera']))
 p=run/'socket_transport/wrist_socket/camera_and_estimate.json'
 if p.exists():
  d=json.loads(p.read_text());records.append(('wrist',d['capture'],d['hand'],d['camera']))
 for role,capture,hand,camera in records:
  actual.setdefault(role,set()).add(capture['camera_path'])
  error=float(np.max(np.abs(np.linalg.inv(np.array(hand).reshape(4,4))@np.array(camera).reshape(4,4)-np.array(mounts[role]['T_HC_cv']))))
  mount_error[role]=max(mount_error[role],error)
  if error>1e-10:violations.append(f'{role} mount changed')
 for role,path in expected.items():
  if actual.get(role)!={path}:violations.append(f'{role} camera path differs from its single declared camera')
 events=[json.loads(line) for line in (run/'four_camera_perception/events.jsonl').read_text().splitlines()]
 consumed=[e for e in events if e['event']=='PALM_MEASUREMENT_CONSUMED']
 for event in consumed:
  if event['consumed_time_s']<event['available_time_s'] or event['intervening_physics_steps']<=0:violations.append('A palm result bypassed its physical latency')
 initialized=[e for e in events if e['event']=='SINGLE_KEY_ANCHOR_INITIALIZED']
 expected_events=1 if rig.get('key_observation_after_major_transport') else 0
 if a['key_observation_event_count']!=1 or len(initialized)!=expected_events:violations.append('Single key initialization count differs')
 out={'scope':'ENDED_EPISODE_RECORDED_PERCEPTION_CONTRACT','passed':not violations,'source_commit':commit,
      'declared_camera_paths':expected,'actual_perception_camera_paths':{k:sorted(v) for k,v in actual.items()},
      'unique_functional_camera_paths':len(set.union(*actual.values())),
      'key_observation_events':a['key_observation_event_count'],'palm_measurements_consumed':len(consumed),
      'maximum_hand_mount_matrix_difference':mount_error,'violations':violations,
      'evidence_video_cameras_are_not_perception_inputs':True,'physical_assembly_verified_by_this_audit':False}
 (run/'four_camera_contract_review.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('run');main(p.parse_args().run)
