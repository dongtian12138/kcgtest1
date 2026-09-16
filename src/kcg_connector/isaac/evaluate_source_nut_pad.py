"""Source PAD identity for the completed nut-grip/turn stages only."""
from __future__ import annotations
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from trace_metadata import iter_truth_samples


def review(directory, *, geometry_plan=None):
    directory=Path(directory);root=Path(__file__).resolve().parents[3]
    names=('f1Link3','f2Link2','f3Link3')
    contract=json.loads((root/'src/kcg_connector/config/visual_assembly_v1_contact_regions.json').read_text())
    nail_first,nail_end=contract['nail_shell_face_range_zero_based_half_open']
    allowed={name:{'PAD'} for name in names}
    if geometry_plan is not None:
        geometry=json.loads(Path(geometry_plan).read_text())
        if not geometry.get('terminal_original_surface_geometry'):
            raise ValueError('Declared Nut regions require the original source finger meshes')
        declared=geometry.get('allowed_contact_regions',{})
        allowed={name:set(declared.get(name,['PAD'])) for name in names}
        if any(not regions or regions-{'PAD','NAIL'} for regions in allowed.values()):
            raise ValueError('Only original source PAD or NAIL surfaces can be admitted')
    groups={name:[] for name in names};other=[];other_count=0;stage_samples=0
    for row in iter_truth_samples(directory):
        phase=row['phase']
        if not (phase in ('key_probe_nut_contact','key_probe_nut_grip_hold') or phase.startswith('key_probe_nut_rotation_')):
            continue
        stage_samples+=1
        for header in row['contacts']['poll_headers']:
            paths=header['paths']
            hand_side=next((i for i,path in enumerate(paths[:2]) if '/handbase_link' in path),None)
            # This review classifies hand contacts. Internal pin/bore points
            # remain in the raw archive and in the assembly mechanics review.
            if hand_side is not None:
                loaded=sum(np.linalg.norm(point['impulse_n_s'])>1e-10 for point in header['contacts'])
                hand_name=paths[hand_side].split('/')[-1];partner=paths[1-hand_side]
                if loaded and (hand_name not in names or not partner.endswith('/TE_J35FreeSplitPlug/CouplingNut')):
                    other_count+=loaded
                    if len(other)<12:other.append({'step':row['step'],'link':hand_name,'partner':partner})
            for name in names:
                side=0 if paths[0].endswith('/'+name) else 1 if paths[1].endswith('/'+name) else None
                if side is None:continue
                pose=row['native_robot_link_pose_audit']['poses'][name]
                R=Rotation.from_quat(np.asarray(pose['orientation_world_wxyz'])[[1,2,3,0]]).as_matrix()
                for point in header['contacts']:
                    if np.linalg.norm(point['impulse_n_s'])<=1e-10:continue
                    partner=paths[1-side]
                    local=(np.asarray(point['position_m'])-pose['position_world_m'])@R
                    groups[name].append(local)
    result={'scope':'POSTRUN_ORIGINAL_SOURCE_PAD_DURING_NUT_CONTACT_GRIP_AND_ROTATION',
        'online_control_used':False,'stages_sampled':stage_samples,'other_actor_positive_contacts':other_count,
        'first_other_contacts':other[:12],'per_link':{},'projection_residual_limit_m':.0005}
    result.update(expected_original_regions={name:sorted(value) for name,value in allowed.items()},
        declared_geometry_plan=None if geometry_plan is None else str(Path(geometry_plan).resolve()))
    if geometry_plan is not None:result['scope']='POSTRUN_DECLARED_ORIGINAL_NUT_GRASP_CONTACT_REGIONS'
    for name,points in groups.items():
        source=root/f'src/iiwa_description/meshes/hand/{name}.STL'
        binding=contract['bindings'][name]
        if hashlib.sha256(source.read_bytes()).hexdigest()!=binding:raise ValueError('Original source surface changed: '+name)
        mesh=trimesh.load(source,process=False)
        pad=np.load(root/f'artifacts/agent_control/tasks/CARTS-GRASP-CROSS-OBJECT-V1/TERMINAL_PAD_EXACT_SOURCE_V2/{name}_PAD_BODY_raw_source_local_m.npz')['source_face_indices']
        count=0;maximum=0.;nonpad=0;uncertain=0;disallowed=0;nail_count=0
        points=np.asarray(points)
        for offset in range(0,len(points),2048):
            _,distance,face=trimesh.proximity.closest_point(mesh,points[offset:offset+2048])
            count+=len(face);maximum=max(maximum,float(distance.max()))
            nonpad+=int((~np.isin(face,pad)).sum());uncertain+=int((distance>.0005).sum())
            nail=(face>=nail_first)&(face<nail_end);nail_count+=int(nail.sum())
            admitted=(np.isin(face,pad) if 'PAD' in allowed[name] else np.zeros(len(face),dtype=bool))
            if 'NAIL' in allowed[name]:admitted|=nail
            disallowed+=int((~admitted).sum())
        result['per_link'][name]={'positive_contact_points':count,'nonpad_points':nonpad,
            'nail_points':nail_count,'disallowed_original_region_points':disallowed,
            'uncertain_projection_points':uncertain,'maximum_projection_residual_m':maximum}
    result['accepted']=bool(stage_samples and not other_count and all(
        row['positive_contact_points']>0 and row['disallowed_original_region_points']==0 and row['uncertain_projection_points']==0
        for row in result['per_link'].values()))
    (directory/'source_nut_pad_review.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    parser.add_argument('--geometry-plan',type=Path);args=parser.parse_args()
    print(json.dumps(review(args.directory,geometry_plan=args.geometry_plan),indent=2))
