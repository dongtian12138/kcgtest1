"""Select image proposals using a declared installation workspace and depth.

No live scene transform, semantic label or object pose is an input. Original
proposal masks/scores remain unchanged; only proposals outside the workspace
are excluded before the existing depth filter and pose estimator.
"""
from pathlib import Path
import json
import numpy as np


def select_detections(detections_json,detections_npz,depth_m,intrinsics,
                      world_from_camera,workspace,output_json,summary_json):
    records=json.loads(Path(detections_json).read_text())
    with np.load(detections_npz,allow_pickle=False) as archive:
        masks=archive['segmentation'];scores=archive['score'];boxes=archive['bbox']
    depth=np.asarray(depth_m,dtype=float);K=np.asarray(intrinsics,dtype=float).reshape(3,3)
    camera=np.asarray(world_from_camera,dtype=float).reshape(4,4)
    lo,hi=np.asarray(workspace['minimum'],float),np.asarray(workspace['maximum'],float)
    if (lo.shape!=(3,) or hi.shape!=(3,) or np.any(hi<=lo)
            or not np.isfinite(np.r_[lo,hi,camera.ravel(),K.ravel()]).all()):
        raise ValueError('Finite calibrated camera and declared workspace required')
    if masks.shape!=(len(records),*depth.shape):raise ValueError('SAM masks and current image differ')
    y,x=np.indices(depth.shape)
    with np.errstate(invalid='ignore'):
        points=np.stack(((x+.5-K[0,2])*depth/K[0,0],(y+.5-K[1,2])*depth/K[1,1],depth),axis=-1)
        world=points@camera[:3,:3].T+camera[:3,3]
    valid=np.isfinite(depth)&(depth>0)
    inside=valid&np.all((world>=lo)&(world<=hi),axis=2)
    kept=[];audit=[]
    for index,(row,mask) in enumerate(zip(records,masks)):
        if (float(row['score'])!=float(scores[index])
                or not np.array_equal(row['bbox'],boxes[index])):
            raise ValueError('Proposal JSON and NPZ ordering differs')
        support=(mask>.5)&valid;total=int(support.sum());count=int((support&inside).sum())
        fraction=count/max(1,total)
        accepted=count>=150 and fraction>=.5
        audit.append({'source_index':index,'score':float(row['score']),
                      'valid_depth_pixels':total,'pixels_in_declared_workspace':count,
                      'workspace_fraction':fraction,'accepted':accepted})
        if accepted:kept.append(row)
    report={'scope':'CURRENT_IMAGE_PROPOSALS_INSIDE_DECLARED_INSTALLATION_WORKSPACE',
            'workspace_world_aabb_m':workspace,'minimum_inside_pixels':150,
            'minimum_inside_fraction':.5,'proposals':audit,'kept_count':len(kept),
            'source_scores_and_masks_modified':False,'object_or_contact_truth_used':False}
    Path(summary_json).write_text(json.dumps(report,indent=2)+'\n')
    if not kept:raise RuntimeError('No current-image proposal has sufficient support in the declared workspace')
    Path(output_json).write_text(json.dumps(kept,indent=2)+'\n')
    return report
