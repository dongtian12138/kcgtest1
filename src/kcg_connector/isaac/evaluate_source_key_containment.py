"""Read-only slot-sidewall review using the accepted model's source keys."""
from pathlib import Path
import argparse
import ast
import json
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Usd,UsdGeom
from trace_metadata import iter_truth_samples


def review(directory, *, first_step=None):
    directory=Path(directory);root=Path(__file__).resolve().parents[3]
    source=root/'artifacts/kcg_connector/te_connector_contact_repaired_20260911/connector_model.usdc'
    stage=Usd.Stage.Open(str(source))
    dimensions=json.loads((root/'artifacts/kcg_connector/key_antirotation_20260910/key_backlash_dimensions.json').read_text())
    if first_step is None:
        entry=json.loads((directory/'socket_transport/key_entry/key_entry_controller_result.json').read_text())
        first_contact_step=int(entry['contact_first_step'])
        step_source='SAME_EPISODE_KEY_ENTRY_CONTROLLER'
    else:
        local=json.loads((directory/'source_stage_probe_result.json').read_text())
        if local.get('scope')!='LOCAL_SOURCE_STAGE_DIAGNOSIS_NOT_VISUAL_ASSEMBLY' or first_step<0:
            raise ValueError('An explicit first step is restricted to the recorded local diagnosis')
        first_contact_step=int(first_step)
        step_source='EXPLICIT_LOCAL_DIAGNOSTIC_START_NOT_A_KEY_ENTRY_REPLAY'
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
    # The low-load hand check reached0.61um, but the already accepted complete
    # mating reference reaches1.706um with this same vertex/plane metric.
    # Use its full loading envelope, rounded outward to2um; retain every raw
    # minimum. This is a postrun numerical band, not a changed physical gap.
    tolerance=.000002
    result={'scope':'POSTRUN_ACCEPTED_SOURCE_KEY_VERTICES_VS_SOURCE_SLOT_SIDEWALL_PLANES',
        'online_control_used':False,'source_model':str(source),'minimum_sidewall_gaps_m':minimum.tolist(),
        'first_reviewed_step':first_contact_step,'first_step_source':step_source,
        'evaluated_sample_count_by_key':counts.tolist(),'minimum_gap_witnesses':witness,
        'full_key_axial_extent_sample_count':full_extent_steps,'final':last,
        'numerical_sidewall_review_tolerance_m':tolerance,
        'reference':'artifacts/visual_assembly_v1/reference_full_mating_key_gap_review.json',
        'accepted_full_mating_reference_maximum_residual_m':1.7060737748285649e-6,
        'physical_geometry_or_contact_parameters_changed':False}
    result['accepted']=bool(np.all(counts>0) and np.all(minimum>=-tolerance)
        and last and last['all_keys_behind_mouth'])
    (directory/'source_key_containment_review.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    parser.add_argument('--first-step',type=int,help='Explicit start for local diagnoses only; never fabricates key-entry evidence')
    args=parser.parse_args()
    print(json.dumps(review(args.directory,first_step=args.first_step),indent=2))
