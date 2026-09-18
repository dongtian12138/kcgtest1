"""Shorten a straight free-space joint path within its existing speed limits."""
import math
import numpy as np
from te_nut_motion import ScalarMotion


def retime_straight_transport(states, dt, speed_limit):
    source=np.asarray(states,dtype=float)
    if source.ndim!=2 or source.shape[1]!=7 or len(source)<3 or not np.isfinite(source).all():
        raise ValueError('Seven-joint sampled path required')
    delta=source[-1]-source[0];distance=float(np.max(np.abs(delta)))
    if distance<1e-10:return source,{'applied':False,'reason':'stationary path'}
    progress=(source-source[0])@delta/(delta@delta)
    residual=float(np.max(np.abs(source-(source[0]+progress[:,None]*delta))))
    if residual>1e-10 or np.min(np.diff(progress)) < -1e-12:
        return source,{'applied':False,'reason':'path is not one monotone straight joint segment',
                       'maximum_line_residual_rad':residual}
    original_v=np.max(np.abs(np.diff(source,axis=0)/dt),axis=0)
    original_a=np.max(np.abs(np.diff(source,n=2,axis=0)/dt**2),axis=0)
    direction=np.abs(delta)/distance
    active=direction>1e-10
    # Preserve the earlier0.8speed headroom and do not exceed any original
    # joint's commanded peak velocity or acceleration. A one-second ramp
    # supplies a finite acceleration ceiling below the old waypoint corners.
    speed=min(.8*float(speed_limit),float(np.min(original_v[active]/direction[active])))
    acceleration=min(speed,float(np.min(original_a[active]/direction[active])))
    if min(speed,acceleration)<=0:return source,{'applied':False,'reason':'no positive original timing envelope'}
    profile=ScalarMotion(distance,speed,acceleration)
    count=math.ceil(profile.duration/dt)
    if count>=len(source)-1:return source,{'applied':False,'reason':'no reduction within original envelope'}
    fraction=np.array([profile.at(i*dt)[0]/distance for i in range(count+1)])
    output=source[0]+fraction[:,None]*delta
    output[0]=source[0];output[-1]=source[-1]
    peak_v=np.max(np.abs(np.diff(output,axis=0)/dt),axis=0)
    peak_a=np.max(np.abs(np.diff(output,n=2,axis=0)/dt**2),axis=0)
    if np.any(peak_v>original_v+1e-9) or np.any(peak_a>original_a+1e-7):
        raise RuntimeError('Retimed transport exceeds its original commanded motion envelope')
    return output,{'applied':True,'source_duration_s':(len(source)-1)*dt,'duration_s':count*dt,
                   'maximum_line_residual_rad':residual,'original_peak_joint_speed_rad_s':original_v.tolist(),
                   'new_peak_joint_speed_rad_s':peak_v.tolist(),
                   'original_peak_joint_acceleration_rad_s2':original_a.tolist(),
                   'new_peak_joint_acceleration_rad_s2':peak_a.tolist(),
                   'same_start_end_and_joint_line':True,'requires_current_full_hand_and_plug_collision_check':True}
