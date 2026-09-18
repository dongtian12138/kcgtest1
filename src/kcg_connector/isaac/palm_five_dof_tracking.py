"""Current palm depth measurement around a previous visual/encoder prediction."""
from pathlib import Path
import numpy as np
from te_plug_five_dof_geometry import estimate_plug_rear_circle_from_float_depth,_visible_face_geometry


def measure(depth_m,intrinsics,world_from_camera,predicted_world_from_body,cad_path):
    depth=np.asarray(depth_m,float);K=np.asarray(intrinsics,float).reshape(3,3)
    camera=np.asarray(world_from_camera,float).reshape(4,4)
    seed=np.asarray(predicted_world_from_body,float).reshape(4,4)
    if not np.isfinite(np.r_[K.ravel(),camera.ravel(),seed.ravel()]).all():
        raise ValueError('Finite visual prediction and calibrated camera required')
    radius,offset=_visible_face_geometry(Path(cad_path))
    body_from_camera=np.linalg.inv(seed)@camera
    y,x=np.indices(depth.shape)
    with np.errstate(invalid='ignore'):
        points=np.stack(((x+.5-K[0,2])*depth/K[0,0],(y+.5-K[1,2])*depth/K[1,1],depth),axis=-1)
        local=points@body_from_camera[:3,:3].T+body_from_camera[:3,3]
        radial=np.linalg.norm(local[:,:,:2],axis=2)
    # The ROI admits4mm around the previous visible face; it is a search
    # region, not a changed pose or insertion acceptance tolerance.
    mask=(np.isfinite(local).all(2)&(depth>0)&(radial<radius+.004)
          &(np.abs(local[:,:,2]+offset)<.004))
    geometry=estimate_plug_rear_circle_from_float_depth(depth_m=depth,mask=mask,intrinsics=K,
        mesh_path=Path(cad_path),pixel_center_offset_px=.5,plane_iterations=128)
    geometry['metrics']['mask_source']='PREVIOUS_VISUAL_IDENTITY_ENCODER_PREDICTION_AND_CURRENT_DEPTH'
    geometry['metrics']['sam_used_for_this_observation']=False
    geometry['metrics']['roi_is_not_an_observed_pose']=True
    if geometry['metrics']['plane_ransac_sampled_inlier_fraction']<.45:
        raise RuntimeError('Current palm face has insufficient plane support')
    world=camera@np.asarray(geometry['camera_from_object'])
    return {'world_from_plug_five_dof':world,'camera_from_plug_five_dof':geometry['camera_from_object'],
            'metrics':geometry['metrics'],'axial_yaw_measured':False,
            'online_object_or_contact_truth_used':False},mask
