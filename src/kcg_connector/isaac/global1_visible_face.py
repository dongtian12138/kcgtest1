"""Initial coarse Body position/axis from the high Global1's visible rear face.

Uses the existing float-depth face estimator and its unchanged image-quality
gates. This is a perception function: no runtime, scene poses or robot API.
"""
import copy
import math
import numpy as np
from kcg_connector.te_rgbd_pose_provider import (
    _foreground_points, _RX180_AXIS_TILT_RESEARCH_BOUND_RAD,
    _RX180_SUPPORT_Z_RESEARCH_BOUND_M)
from te_plug_five_dof_geometry import estimate_plug_rear_circle_from_float_depth


def estimate(inputs, legacy_result, mesh_path):
    old=legacy_result['transport_grasp_pose']
    sign=old['derivation']['axis_sign']
    if not sign.get('sign_unique') or sign.get('supplier_plus_z_world_sign')!=-1:
        raise ValueError('Current Global1 depth must resolve the supported plug axis sign')
    points=_foreground_points(inputs)['plug']
    mask=np.zeros(inputs.depth_m.shape,bool)
    mask[points['pixel_v'],points['pixel_u']]=True
    measured=estimate_plug_rear_circle_from_float_depth(
        depth_m=inputs.depth_m,mask=mask,intrinsics=inputs.intrinsics,
        mesh_path=mesh_path,pixel_center_offset_px=.5,plane_iterations=128)
    measured['metrics'].update(mask_source='ORDINARY_BACKGROUND_DEPTH_DIFFERENCE_AND_DECLARED_WORKSPACE',
                               sam_used_for_this_observation=False)
    world=inputs.world_from_camera@np.asarray(measured['camera_from_object'])
    table=inputs.manifest['known_static_scene_geometry']['table']
    support=float(table['center_world_m'][2])+.5*float(table['size_m'][2])
    tilt=math.acos(float(np.clip(-world[2,2],-1,1)))
    if (abs(float(world[2,3])-support)>_RX180_SUPPORT_Z_RESEARCH_BOUND_M
            or tilt>_RX180_AXIS_TILT_RESEARCH_BOUND_RAD):
        raise ValueError('Measured rear face leaves the original initial support/tilt envelope')
    bounds=inputs.manifest['frozen_endpoint_workspaces_world_aabb_m']['plug']
    if not np.all((world[:3,3]>=bounds['minimum'])&(world[:3,3]<=bounds['maximum'])):
        raise ValueError('Measured plug origin is outside its declared search workspace')
    pose=copy.deepcopy(old)
    pose.update(status='OBSERVED_AXIS_POSITION_YAW_FREE',target_part='Body',
        position_xyz_m=world[:3,3].tolist(),outward_axis_world=world[:3,2].tolist(),
        yaw_status='UNOBSERVED_FREE_TRANSVERSE_FRAME_FOR_INITIAL_GRASP_ONLY',
        transport_grasp_planning_input_available=True,confidence=None,
        confidence_scope='NO_NEW_CALIBRATED_CONFIDENCE_CLAIMED',
        derivation={'method':'CURRENT_GLOBAL1_VISIBLE_BODY_REAR_FACE_DEPTH_CIRCLE',
            'axis_sign_profile_check':sign,'metrics':measured['metrics'],
            'support_height_used_as_acceptance_check_not_substituted_for_measured_z':True,
            'original_support_z_bound_m':_RX180_SUPPORT_Z_RESEARCH_BOUND_M,
            'original_axis_tilt_bound_rad':_RX180_AXIS_TILT_RESEARCH_BOUND_RAD})
    return {'schema_version':'kcg_global1_visible_face_coarse_pose_v1',
        'status':pose['status'],'transport_grasp_pose':pose,
        'world_from_plug_five_dof_with_arbitrary_transverse_basis':world.tolist(),
        'legacy_lower_ring_observation':old,'key_yaw_measured':False,
        'object_pose_or_contact_truth_used':False,'robot_command_count':0,
        'input_scope':'CURRENT_ORDINARY_RGBD_CALIBRATION_BACKGROUND_CAD_AND_DECLARED_WORKSPACE'}
