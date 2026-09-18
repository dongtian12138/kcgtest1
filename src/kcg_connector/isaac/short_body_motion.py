"""Bounded source-model IK along an explicitly requested carried-body path."""
import math
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from kcg_connector.grasp.robust.bounded_hand_base_ik import bounded_ik_settings,solve_bounded_target


def plan_body_path(inputs,start_positions,hand_from_body,target_world_from_body,dt,speed_limit):
    from te_foundationpose_handoff_runtime import MOVEIT_SOFT_ARM_BOUNDS_RAD,control
    start=np.asarray(start_positions,float);memory=np.asarray(hand_from_body,float).reshape(4,4)
    target=np.asarray(target_world_from_body,float).reshape(4,4)
    if start.shape!=(11,) or not np.isfinite(np.r_[start,memory.ravel(),target.ravel()]).all():
        raise ValueError('Finite current encoders and visual body poses required')
    hand=np.asarray(inputs.robot_model.forward_kinematics(tuple(start),enforce_limits=False)['handbase_link'])
    body=hand@memory
    distance=float(np.linalg.norm(target[:3,3]-body[:3,3]))
    angle=float(Rotation.from_matrix(target[:3,:3]@body[:3,:3].T).magnitude())
    intervals=max(2,math.ceil(distance/.005),math.ceil(angle/.04))
    fractions=np.linspace(0.,1.,intervals+1)
    rotations=Slerp([0,1],Rotation.from_matrix([body[:3,:3],target[:3,:3]]))
    bounds=np.asarray([MOVEIT_SOFT_ARM_BOUNDS_RAD[n] for n in control.ARM_JOINT_NAMES])
    settings=bounded_ik_settings(inputs.config.section('ik')['solver'])
    qs=[start[:7].copy()];errors=[]
    for index,fraction in enumerate(fractions[1:],1):
        pose=np.eye(4);pose[:3,3]=(1-fraction)*body[:3,3]+fraction*target[:3,3]
        pose[:3,:3]=rotations(fraction).as_matrix();goal=pose@np.linalg.inv(memory)
        q,position_error,orientation_error,_=solve_bounded_target(
            model=inputs.robot_model,hand_positions=start[7:],target=goal,seeds=(qs[-1],),
            lower=bounds[:,0],upper=bounds[:,1],settings=settings,label=f'FIXED_CAMERA_BODY_PATH_{index}')
        qs.append(np.asarray(q));errors.append([position_error,orientation_error])
    qs=np.asarray(qs)
    derivative=float(np.max(np.abs(np.diff(qs,axis=0))))*intervals
    duration=max(1.,1.875*derivative/(.8*float(speed_limit)))
    count=math.ceil(duration/float(dt));t=np.arange(count+1)/count
    progress=10*t**3-15*t**4+6*t**5
    states=np.column_stack([np.interp(progress,fractions,qs[:,j]) for j in range(7)])
    peak=float(np.max(abs(np.diff(states,axis=0)))/dt)
    if peak>.8*speed_limit+1e-8 or np.any(states<bounds[:,0]) or np.any(states>bounds[:,1]):
        raise RuntimeError('Cartesian body timing left the original joint-speed/position envelope')
    return states,{'scope':'PLANNED_BODY_CARTESIAN_PATH_REQUIRES_FULL_HAND_AND_CARRIED_BODY_COLLISION_CHECK',
        'start_world_from_body':body.tolist(),'target_world_from_body':target.tolist(),
        'body_translation_distance_m':distance,'body_rotation_rad':angle,'ik_waypoint_count':len(qs),
        'duration_s':count*dt,'maximum_commanded_joint_speed_rad_s':peak,
        'maximum_ik_position_error_m':max(e[0] for e in errors),
        'maximum_ik_orientation_error_rad':max(e[1] for e in errors),
        'same_original_soft_joint_bounds':True,'no_object_or_contact_truth_input':True}
