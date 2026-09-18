"""Ended native-contact support review; exact source-key review stays separate."""
from pathlib import Path
from collections import Counter
import json,sys
import numpy as np
from scipy.spatial.transform import Rotation
root=Path(__file__).resolve().parents[2];sys.path[:0]=[str(root/'src/kcg_connector/isaac'),str(root/'src/kcg_connector/isaac/carts_v2')]
from trace_metadata import iter_truth_fields
r=Path(sys.argv[1]);d=json.loads((r/'socket_transport/body_support/support_release_controller_result.json').read_text())
if 'exit_code' not in json.loads((r.parent/'process.json').read_text()):raise ValueError('Ended episode required')
dt=float(json.loads((r/'trace_metadata.json').read_text())['physics_dt_s'])
rows=[];pairs=Counter();coverage=Counter();contact_residual=0.;maximum_backup=0.
for row in iter_truth_fields(r,('phase','simulation_time_s','contacts','object_part_positions_m','object_part_orientations_wxyz','object_part_sleeping'),first_step=d['first_step']-1,last_step=d['last_step']-1):
 positions=np.array(row['object_part_positions_m']);rotation=Rotation.from_quat(np.roll(row['object_part_orientations_wxyz'][0],-1)).as_matrix()
 maximum_backup=max(maximum_backup,abs(float((rotation.T@(positions[1]-positions[0]))[2])))
 c=row['contacts'];hand=0.;support=0.
 for h in c['poll_headers']:
  paths=h['paths'];impulse=sum(float(np.linalg.norm(p['impulse_n_s'])) for p in h['contacts'])
  if any('/handbase_link' in p for p in paths[:2]):hand+=impulse
  if any('FixedReceptaclePose' in p for p in paths[:2]) and any('TE_J35FreeSplitPlug' in p for p in paths[:2]):
   support+=impulse
   if impulse>0:
    pairs[tuple(paths[2:])]+=1
    for point in h['contacts']:contact_residual=max(contact_residual,-float(point.get('separation_m',0)))
 coverage['one_callback']+=int(c['physics_step_callback_count']==1)
 coverage['all_points_retained']+=int(c['all_provided_native_report_points_retained'])
 rows.append({'step':row['step'],'phase':row['phase'],'body_z':row['object_part_positions_m'][0][2], 'hand_raw_impulse':hand,'socket_raw_impulse':support,'awake':not any(row['object_part_sleeping'])})
held=[x for x in rows if x['phase']=='key_probe_body_support_hold'];n=len(held)
checks={'controller_support_completed':d['completed'], 'hold_exists':bool(held),'strictly_zero_hand_impulse_every_hold_frame':bool(held) and all(x['hand_raw_impulse']==0 for x in held),
 'positive_socket_support_every_hold_frame':bool(held) and all(x['socket_raw_impulse']>0 for x in held),'body_and_nut_awake_every_hold_frame':bool(held) and all(x['awake'] for x in held),'native_coverage':all(v==len(rows) for v in coverage.values()),'complete_recorded_interval':len(rows)==d['last_step']-d['first_step']+1,'backup_limit_not_reached':maximum_backup<.0005999}
out={'scope':'ENDED_NATIVE_CONTACT_BODY_SUPPORT_ONLY','accepted':all(checks.values()),'checks':checks,'hold_sample_count':n,'hold_duration_s':n*dt,'maximum_backup_coordinate_m':maximum_backup,'hold_body_z_range_m':float(np.ptp([x['body_z'] for x in held])) if held else None,'hold_socket_impulse_range_n_s':[min(x['socket_raw_impulse'] for x in held),max(x['socket_raw_impulse'] for x in held)] if held else None,'maximum_native_socket_contact_penetration_m':contact_residual,'positive_source_pairs':[{'paths':list(k),'reports':v} for k,v in pairs.items()],'full_assembly_claimed':False}
(r/'native_body_support_review.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
