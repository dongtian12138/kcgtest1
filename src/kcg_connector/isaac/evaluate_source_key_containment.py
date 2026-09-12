"""Read-only slot-sidewall review using the accepted model's source keys."""
from pathlib import Path
import argparse
import ast
import json
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Usd,UsdGeom
from trace_metadata import iter_truth_samples


def review(directory):
    directory=Path(directory);root=Path(__file__).resolve().parents[3]
    source=root/'artifacts/kcg_connector/te_connector_contact_repaired_20260911/connector_model.usdc'
    stage=Usd.Stage.Open(str(source))
    dimensions=json.loads((root/'artifacts/kcg_connector/key_antirotation_20260910/key_backlash_dimensions.json').read_text())
    entry=json.loads((directory/'socket_transport/key_entry/key_entry_controller_result.json').read_text())
    first_contact_step=int(entry['contact_first_step'])
    installation=json.loads((directory/'frozen_model_installation.json').read_text())
    socket_path='/World/TEVisualHandoff/FixedReceptaclePose'
    socket=np.asarray(ast.literal_eval(installation['pose_mass_velocity_after'][socket_path]['world_transform']),float).T
    vertices=[]
    for index in range(5):
        mesh=UsdGeom.Mesh(stage.GetPrimAtPath(f'/World/TE_J35FreeSplitPlug/Body/SourceGuideKey_{index:03d}'))
        local=np.asarray(mesh.GetPointsAttr().Get());transform=np.asarray(UsdGeom.Xformable(mesh).GetLocalTransformation())
        vertices.append((np.c_[local,np.ones(len(local))]@transform)[:,:3])
    minimum=np.full(5,np.inf);counts=np.zeros(5,dtype=int);witness=[None]*5;full_extent_steps=0;last=None
    for row in iter_truth_samples(directory):
        if int(row['step'])<first_contact_step:continue
        if not row['phase'].startswith(('key_probe_','nut_index_')):continue
        R=socket[:3,:3].T@Rotation.from_quat(np.roll(row['object_part_orientations_wxyz'][0],-1)).as_matrix()
        p=socket[:3,:3].T@(np.asarray(row['object_part_positions_m'][0])-socket[:3,3])
        complete=True
        for index,(v,key) in enumerate(zip(vertices,dimensions['keys'])):
            points=v@R.T+p
            complete&=bool(np.max(points[:,2])<0)
            inside=points[points[:,2]<0]
            if not len(inside):continue
            gaps=np.asarray([inside*1000@np.asarray(plane['normal'])-plane['d_mm'] for plane in key['planes']])*.001
            gap=float(gaps.min());counts[index]+=1
            if gap<minimum[index]:minimum[index]=gap;witness[index]={'step':row['step'],'phase':row['phase'],'gap_m':gap}
        if complete:full_extent_steps+=1
        last={'step':row['step'],'all_keys_behind_mouth':complete,'body_depth_m':float(-p[2])}
    # The delivered reference records up to0.61um signed residual in the hand
    # interface test. Keep a1um numerical review band, with all raw minima.
    tolerance=.000001
    result={'scope':'POSTRUN_ACCEPTED_SOURCE_KEY_VERTICES_VS_SOURCE_SLOT_SIDEWALL_PLANES',
        'online_control_used':False,'source_model':str(source),'minimum_sidewall_gaps_m':minimum.tolist(),
        'evaluated_sample_count_by_key':counts.tolist(),'minimum_gap_witnesses':witness,
        'full_key_axial_extent_sample_count':full_extent_steps,'final':last,
        'numerical_sidewall_review_tolerance_m':tolerance,
        'reference':'artifacts/kcg_connector/te_connector_contact_repaired_20260911/final_key_clearance_review.json'}
    result['accepted']=bool(np.all(counts>0) and np.all(minimum>=-tolerance)
        and last and last['all_keys_behind_mouth'])
    (directory/'source_key_containment_review.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path);args=parser.parse_args()
    print(json.dumps(review(args.directory),indent=2))
