"""Post-run source-nail contact and lift review; never imported by control."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


def review(run, repository):
    run,repository=Path(run),Path(repository)
    contract_path=repository/'src/kcg_connector/config/visual_assembly_v1_contact_regions.json'
    contract=json.loads(contract_path.read_text())
    names=list(contract['bindings']);lo,hi=contract['nail_shell_face_range_zero_based_half_open']
    points={name:[] for name in names};metadata={name:[] for name in names};states=[]
    stream=run/'truth_samples.jsonl.gz';body_grasp_ended=False
    with gzip.open(stream,'rt') as handle:
        for line in handle:
            row=json.loads(line)
            if row['phase'].startswith('key_probe_body_support_'):
                body_grasp_ended=True
            if body_grasp_ended:break
            states.append({'step':row['step'],'t':row['simulation_time_s'],'phase':row['phase'],
                'body_z':row['object_part_positions_m'][0][2],
                'table_impulse':row['contacts']['object_table_positive_normal_impulse_n_s']})
            for header in row['contacts'].get('poll_headers',[]):
                paths=header.get('paths',[])
                if len(paths)!=4:continue
                for i,name in enumerate(names):
                    side=0 if paths[0].endswith('/'+name) else 1 if paths[1].endswith('/'+name) else None
                    if side is None:continue
                    other=paths[1-side];body=other.endswith('/TE_J35FreeSplitPlug/Body')
                    pose=row['native_robot_link_pose_audit']['poses'][name]
                    R=Rotation.from_quat(np.asarray(pose['orientation_world_wxyz'])[[1,2,3,0]]).as_matrix()
                    for contact in header['contacts']:
                        magnitude=float(np.linalg.norm(contact['impulse_n_s']))
                        if magnitude<=0:continue
                        p=(np.asarray(contact['position_m'])-pose['position_world_m'])@R
                        points[name].append(p)
                        metadata[name].append((row['step'],magnitude,body,other))
    result={'scope':'POSTRUN_ONLY_SOURCE_NAIL_CONTACT_AND_LIFT','online_control_used':False,
        'classification':'nearest original STL face at native rigid-link contact position',
        'maximum_source_projection_residual_m':.0005,'source_face_range':[lo,hi],
        'sensor_truth_stream':str(stream),'fingers':{},'physical_assembly_complete_claimed':False}
    nail_steps={name:set() for name in names}
    for name in names:
        source=repository/'src/iiwa_description/meshes/hand'/f'{name}.STL'
        if hashlib.sha256(source.read_bytes()).hexdigest()!=contract['bindings'][name]:
            raise ValueError('Source nail identity changed: '+name)
        mesh=trimesh.load(source,process=False)
        if len(mesh.faces)!=contract['source_face_count']:raise ValueError('Source face order changed')
        pts=np.asarray(points[name]);info=metadata[name]
        categories={'nail_body':0,'other_surface_body':0,'uncertain_projection':0,'other_object':0}
        impulses={key:0. for key in categories};max_residual=0.;examples=[]
        proximity=trimesh.proximity.ProximityQuery(mesh)
        for offset in range(0,len(pts),2048):
            _,distance,faces=proximity.on_surface(pts[offset:offset+2048])
            for j,(d,face) in enumerate(zip(distance,faces)):
                step,magnitude,body,other=info[offset+j]
                category=('other_object' if not body else 'uncertain_projection' if d>.0005
                          else 'nail_body' if lo<=face<hi else 'other_surface_body')
                categories[category]+=1;impulses[category]+=magnitude;max_residual=max(max_residual,float(d))
                if category=='nail_body':nail_steps[name].add(step)
                elif len(examples)<5:examples.append({'step':step,'category':category,'face':int(face),
                    'distance_m':float(d),'other':other,'point_local_m':pts[offset+j].tolist()})
        result['fingers'][name]={'positive_contact_points':len(pts),'counts':categories,
            'normal_impulse_sum_n_s':impulses,'maximum_projection_residual_m':max_residual,
            'exceptions':examples}
    hold=[s for s in states if s['phase']=='hold'];initial_z=states[0]['body_z']
    interval=states[1]['t']-states[0]['t'] if len(states)>1 else 0.
    common=set.intersection(*(nail_steps[name] for name in names))
    result.update(maximum_body_lift_m=max(s['body_z'] for s in states)-initial_z,
        hold_duration_s=len(hold)*interval,
        minimum_body_lift_during_hold_m=min((s['body_z']-initial_z for s in hold),default=None),
        table_contact_samples_in_hold=sum(s['table_impulse']>0 for s in hold),
        all_three_nails_body_contact_fraction_in_hold=(sum(s['step'] in common for s in hold)/len(hold) if hold else None),
        each_nail_body_contact_fraction_in_hold={name:(sum(s['step'] in nail_steps[name] for s in hold)/len(hold) if hold else None) for name in names})
    result['fresh_image_motion_consumption_record_present']=(run/'initial_rgbd/consumed_grasp_plan.json').is_file()
    result['contact_scope']='INITIAL_BODY_GRASP_AND_CARRY_BEFORE_INTENTIONAL_BODY_UNLOAD'
    result['accepted']=bool(result['fresh_image_motion_consumption_record_present']
        and result['hold_duration_s']>=2.-interval/2
        and result['minimum_body_lift_during_hold_m'] is not None
        and result['minimum_body_lift_during_hold_m']>=.05
        and result['table_contact_samples_in_hold']==0
        and result['all_three_nails_body_contact_fraction_in_hold']==1.
        and all(v['counts']['nail_body']>0 and sum(c for k,c in v['counts'].items() if k!='nail_body')==0
                for v in result['fingers'].values()))
    (run/'source_nail_body_review.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path)
    args=parser.parse_args()
    print(json.dumps(review(args.run,Path(__file__).resolve().parents[3]),indent=2))
