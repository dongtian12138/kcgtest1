"""Coarse socket center/axis from its current segmented depth and fixed workspace.

The upright fixture prior selects the upper lip. Position and the normal are
then fitted from depth. Transverse yaw is unobserved and is not key alignment.
"""
import numpy as np
from scipy.optimize import least_squares


def depth_component_from_image_seed(depth, static_depth, seed_mask, intrinsics,
                                    world_from_camera, workspace, *, foreground_delta_m=.00025):
    """Recover an object's current depth silhouette when SAM selects an inner face.

    No pose is supplied: the seed must substantially overlap one connected
    foreground component inside the same declared socket search workspace.
    Original lip geometry checks still decide whether this region is usable.
    """
    import cv2
    depth=np.asarray(depth,float);background=np.asarray(static_depth,float)
    seed=np.asarray(seed_mask,bool);k=np.asarray(intrinsics,float);camera=np.asarray(world_from_camera,float)
    if depth.shape!=background.shape or seed.shape!=depth.shape:
        raise ValueError('Seed, current depth and calibrated background must have identical dimensions')
    y,x=np.indices(depth.shape)
    with np.errstate(invalid='ignore'):
        points=np.stack(((x+.5-k[0,2])*depth/k[0,0],(y+.5-k[1,2])*depth/k[1,1],depth),-1)
        world=points@camera[:3,:3].T+camera[:3,3]
        foreground=(np.isfinite(depth)&(depth>0)&
            ((~np.isfinite(background))|(background<=0)|(background-depth>=foreground_delta_m)))
        foreground&=np.all((world>=workspace['minimum'])&(world<=workspace['maximum']),axis=-1)
    count,labels=cv2.connectedComponents(foreground.astype(np.uint8),connectivity=8)
    overlap=np.bincount(labels[seed&foreground],minlength=count);overlap[0]=0
    candidates=np.flatnonzero((overlap>=150)&(overlap/max(1,int(seed.sum()))>=.5))
    if len(candidates)!=1:
        raise ValueError('Image seed does not uniquely overlap one supported foreground depth component')
    chosen=int(candidates[0]);mask=labels==chosen
    return mask,{'method':'CURRENT_DEPTH_COMPONENT_OVERLAPPING_SAM_IMAGE_SEED',
        'seed_pixels':int(seed.sum()),'component_pixels':int(mask.sum()),
        'overlap_pixels':int(overlap[chosen]),'foreground_delta_m':float(foreground_delta_m),
        'minimum_overlap_pixels':150,'minimum_seed_overlap_fraction':.5,
        'source_mask_modified':False,'object_or_contact_truth_used':False}


def estimate(depth, mask, intrinsics, world_from_camera, workspace):
    depth=np.asarray(depth,float);k=np.asarray(intrinsics,float);camera=np.asarray(world_from_camera,float)
    y,x=np.nonzero(np.asarray(mask,bool)&np.isfinite(depth)&(depth>0))
    z=depth[y,x]
    points=np.column_stack(((x+.5-k[0,2])*z/k[0,0],(y+.5-k[1,2])*z/k[1,1],z))
    world=points@camera[:3,:3].T+camera[:3,3]
    lo,hi=np.asarray(workspace['minimum']),np.asarray(workspace['maximum'])
    world=world[np.all((world>=lo)&(world<=hi),axis=1)]
    if len(world)<150:raise ValueError('Socket coarse depth has insufficient workspace support')
    # Same40um axial sampling scale as the existing lip observer. Require
    # an actual populated plane, rather than treating one high pixel as a lip.
    bins=np.floor((world[:,2]-lo[2])/.00004).astype(int)
    labels,counts=np.unique(bins,return_counts=True)
    supported=labels[counts>=20]
    if not len(supported):raise ValueError('No upper socket lip depth cluster')
    highest=lo[2]+(int(supported[-1])+1)*.00004
    upper=world[(world[:,2]>=highest-.00008)&(world[:,2]<=highest+.00004)]
    median=float(np.median(upper[:,2]));mad=float(np.median(abs(upper[:,2]-median)))
    flat=upper[abs(upper[:,2]-median)<=max(10.*mad,.000001)]
    if len(flat)<20:raise ValueError('Upper lip plane is not supported by current depth')
    centroid=flat.mean(0);_,singular,vectors=np.linalg.svd(flat-centroid,full_matrices=False)
    axis=vectors[-1];axis*=np.sign(axis[2])
    plane_rms=float(np.sqrt(np.mean(((flat-centroid)@axis)**2)))
    if plane_rms>.00004 or axis[2]<np.cos(np.deg2rad(5.)) or singular[1]<.001:
        raise ValueError('Socket lip fit contradicts the declared upright fixture/plane support')
    first=np.array([1.,0.,0.]);first-=axis*(first@axis);first/=np.linalg.norm(first)
    second=np.cross(axis,first);basis=np.column_stack((first,second))
    xy=(upper-centroid)@basis
    start=(xy.min(0)+xy.max(0))/2
    fitted=least_squares(lambda v:np.linalg.norm(xy-v[:2],axis=1)-v[2],np.r_[start,.0188],
                         loss='soft_l1',f_scale=.0002)
    radius=float(fitted.x[2]);residual=float(np.sqrt(np.mean(fitted.fun**2)))
    # Points lie in an annulus, not on a single contour. This radius check
    # uses the source inner/outer lip radii; it is a coarse-only estimate.
    if not .0179705<=radius<=.0195707 or residual>.0008:
        raise ValueError('Coarse annular fit is inconsistent with the original socket lip')
    pose=np.eye(4);pose[:3,:3]=np.column_stack((first,second,axis));pose[:3,3]=centroid+basis@fitted.x[:2]
    if not np.all((pose[:3,3]>=lo)&(pose[:3,3]<=hi)):
        raise ValueError('Measured socket center is outside the declared workspace')
    return {'world_from_socket_coarse':pose.tolist(),'socket_key_yaw_measured':False,
        'source':'CURRENT_GLOBAL1_SEGMENTED_DEPTH_UPPER_LIP_PLANE_AND_ANNULUS',
        'workspace_points':len(world),'upper_lip_points':len(upper),'plane_points':len(flat),
        'plane_rms_m':plane_rms,'annulus_fit_radius_m':radius,'annulus_fit_rms_m':residual,
        'fixture_prior':'UPRIGHT_SUPPORT_NORMAL_PLUS_WORLD_Z_SELECTS_UPPER_LIP_ONLY',
        'transverse_frame_chosen_not_measured':True,'fine_contact_pose_requires_wrist':True,
        'online_object_or_contact_truth_used':False}
