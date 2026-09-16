"""Bounded current-image updates of the held connector estimate near seating."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def visual_grasp_axis_update(hand, hand_from_pivot, body_observation, socket_axis,
        *, maximum_lateral_change_m, maximum_axis_change_deg, captured_nut_offset_m=.0005):
    """Update a robot-side estimate; preserve axial progress and unobserved yaw.

    Body and captured Nut share their guide axis. Only camera position/axis
    and robot FK enter this calculation. No physical pose is written.
    """
    hand=np.asarray(hand,float);relation=np.asarray(hand_from_pivot,float)
    body=np.asarray(body_observation,float);axis=np.asarray(socket_axis,float)
    if any(x.shape!=(4,4) or not np.isfinite(x).all() for x in (hand,relation,body)):
        raise ValueError('Finite hand and current-image transforms are required')
    if axis.shape!=(3,) or not np.isfinite(axis).all() or np.linalg.norm(axis)<.5:
        raise ValueError('A finite visual Socket axis is required')
    limits=np.asarray([maximum_lateral_change_m,maximum_axis_change_deg,captured_nut_offset_m],float)
    if (not np.isfinite(limits).all() or not 0<limits[0]<=.0015
            or not 0<limits[1]<=.75 or not 0<=limits[2]<=.0005000001
            or np.linalg.norm(body[:3,2])<.5):
        raise ValueError('The visual grasp update needs finite local correction bounds')
    axis=axis/np.linalg.norm(axis);seen=body[:3,2]/np.linalg.norm(body[:3,2])
    current=hand@relation;old_axis=current[:3,2]/np.linalg.norm(current[:3,2])
    cross=np.cross(old_axis,seen);sine=np.linalg.norm(cross)
    angle=float(np.arctan2(sine,old_axis@seen))
    if angle>np.deg2rad(maximum_axis_change_deg):
        raise ValueError('Current image exceeds the bounded grasp-axis update')
    rotation=np.eye(3) if sine<1e-12 else Rotation.from_rotvec(cross*angle/sine).as_matrix()
    observed_nut=body[:3,3]+seen*captured_nut_offset_m
    lateral=(np.eye(3)-np.outer(axis,axis))@(observed_nut-current[:3,3])
    if np.linalg.norm(lateral)>maximum_lateral_change_m:
        raise ValueError('Current image exceeds the bounded lateral grasp update')
    updated=current.copy();updated[:3,:3]=rotation@current[:3,:3];updated[:3,3]+=lateral
    result=np.linalg.inv(hand)@updated
    return result,{'camera_body_axis_error_deg':float(np.degrees(np.arccos(np.clip(-axis@seen,-1,1)))),
        'grasp_axis_update_deg':float(np.degrees(angle)),
        'lateral_grasp_update_world_m':lateral.tolist(),
        'estimate_update_is_not_a_physical_displacement_command':True,
        'axial_estimate_change_m':float(axis@(updated[:3,3]-current[:3,3])),
        'camera_yaw_used':False,'physical_pose_written':False}


def seating_segment_duration(degrees, speed_deg_s, acceleration_deg_s2):
    values=np.asarray([degrees,speed_deg_s,acceleration_deg_s2],float)
    if not np.isfinite(values).all() or np.min(values)<=0:
        raise ValueError('Positive finite seating motion parameters are required')
    return max(1.875*degrees/speed_deg_s,
        np.sqrt((10./np.sqrt(3.))*degrees/acceleration_deg_s2))


def visual_seating_commands(repository,runtime,stepper,controller,socket,grip,output,
        record,geometry_reference,degrees,speed,config):
    """Yield turn/axis-hold commands using the existing shared controller."""
    import omni.usd
    import omni.replicator.core as rep
    from te_body_socket_observation import observe_current_plug_from_rgbd
    world=runtime['world'];dt=float(world.get_physics_dt());out=Path(output)
    axis=np.asarray(socket)[:3,2];segment=float(config['maximum_segment_deg'])
    acceleration=float(config['maximum_profile_acceleration_deg_s2'])
    settle=float(config.get('axis_settle_s',1.));attempts=int(config.get('maximum_axis_updates',2))
    tolerance=float(config.get('axis_tolerance_deg',.02));maximum_axis=float(config.get('maximum_axis_update_deg',.75))
    estimate_limit=float(config.get('maximum_lateral_estimate_update_m',.0006))
    if not (0<segment<=12. and 0<acceleration<=50. and .5<=settle<=1.
            and 1<=attempts<=2 and 0<tolerance<=.03 and 0<maximum_axis<=.75
            and 0<estimate_limit<=.0015):
        raise ValueError('Visual seating exceeds its finite local motion bounds')
    if controller.grip_recipe.get('angular_admittance_rad_per_nm_s'):
        raise ValueError('Current-image axis following has one orientation controller')
    initial_relation=controller.hand_from_pivot.copy()
    initial_wrench_origin=controller.pivot0.copy()
    elapsed=0.;index=0;done=0.;checkpoint=0
    phases=[];observations=[]
    record['visual_seating_axis_control']={'source':'CURRENT_RGBD_BODY_POSITION_AXIS_AND_ROBOT_ENCODERS',
        'observations':observations,'turn_profiles':phases,'object_truth_used':False,
        'source_wrench_filter_origin_and_two_stage_transform_unchanged':True}
    while done<degrees-1e-8:
        amount=min(segment,degrees-done)
        duration=seating_segment_duration(amount,speed,acceleration)
        count=max(1,int(np.ceil(duration/dt)));duration=count*dt
        phases.append({'start_deg':done,'end_deg':done+amount,'duration_s':duration,
            'peak_speed_deg_s':1.875*amount/duration,
            'peak_acceleration_deg_s2':(10./np.sqrt(3.))*amount/duration**2})
        for local_index in range(count+1):
            u=local_index/count;blend=10*u**3-15*u**4+6*u**5
            angle=np.radians(done+amount*blend)
            rate=np.radians(amount)*30*u*u*(1-u)**2/duration
            yield index,elapsed,angle,rate,'key_probe_nut_rotation_turn'
            index+=1;elapsed+=dt
        done+=amount
        if done>=degrees-1e-8:break
        for attempt in range(attempts+1):
            world.pause()
            q=np.asarray(stepper.latest[0]);hand=controller.model.forward_kinematics(q,enforce_limits=False)['handbase_link']
            path=out/f'body_axis_{checkpoint:02d}_{attempt:02d}'
            observation=observe_current_plug_from_rgbd(repository,omni.usd.get_context().get_stage(),
                world,rep,hand,path,runtime)
            if not observation.get('position_and_axis_measured'):
                raise RuntimeError('The current image did not resolve the seating axis')
            # A stationary scene viewed from an unchanged camera must agree
            # across completed captures. This catches delayed pose/image pairs
            # without consulting an object's simulator pose.
            consistency=[];consistent=False
            for repeat in range(2):
                previous=np.asarray(observation['world_from_plug_five_dof'],float).reshape(4,4)
                next_path=path.with_name(path.name+f'_confirm_{repeat:02d}')
                fresh=observe_current_plug_from_rgbd(repository,omni.usd.get_context().get_stage(),
                    world,rep,hand,next_path,runtime)
                if not fresh.get('position_and_axis_measured'):
                    raise RuntimeError('Repeated current image did not resolve the seating axis')
                latest=np.asarray(fresh['world_from_plug_five_dof'],float).reshape(4,4)
                position_delta=float(np.linalg.norm(latest[:3,3]-previous[:3,3]))
                axis_delta=float(np.degrees(np.arccos(np.clip(latest[:3,2]@previous[:3,2],-1,1))))
                consistency.append({'image_directory':str(next_path),
                    'position_difference_m':position_delta,'axis_difference_deg':axis_delta})
                observation=fresh
                if position_delta<=.00001 and axis_delta<=.005:
                    consistent=True;break
            (path/'capture_consistency.json').write_text(json.dumps({'consistent':consistent,
                'checks':consistency,'scene_physics_paused':True,'object_truth_used':False},indent=2)+'\n')
            if not consistent:
                raise RuntimeError('Repeated stationary-camera images did not agree before axis control')
            body=np.asarray(observation['world_from_plug_five_dof'],float).reshape(4,4)
            body_lateral=np.linalg.norm((np.eye(3)-np.outer(axis,axis))@(body[:3,3]-np.asarray(socket)[:3,3]))
            if body_lateral>float(grip['visual_alignment_allowance_from_quarter_body_clearance_m']):
                raise RuntimeError('Current image places the Body outside its existing guided region')
            relation,details=visual_grasp_axis_update(hand,controller.hand_from_pivot,body,axis,
                maximum_lateral_change_m=estimate_limit,
                maximum_axis_change_deg=maximum_axis)
            total=Rotation.from_matrix(relation[:3,:3]@initial_relation[:3,:3].T).magnitude()
            if total>np.deg2rad(maximum_axis):raise RuntimeError('Total visual grasp-axis update exceeds its bound')
            controller.hand_from_pivot=relation
            # The fixed wrench-filter point and its later shift to the current
            # estimated pivot are kept intact. Only the grasp estimate changes.
            if not np.array_equal(controller.pivot0,initial_wrench_origin):
                raise RuntimeError('Visual seating changed the fixed wrench-filter frame')
            geometry_reference.update(body=body.copy(),pivot_position=(hand@relation)[:3,3].copy())
            entry={'step':int(stepper.step_index),'commanded_angle_deg':done,'image_directory':str(path),
                **details,'observed_body_lateral_error_m':float(body_lateral),
                'accepted_image_directory':str(next_path),'capture_consistency':consistency,
                'new_hand_from_pivot':relation.tolist()}
            observations.append(entry)
            (out/'visual_axis_progress.json').write_text(json.dumps(record['visual_seating_axis_control'],indent=2)+'\n')
            world.play()
            if details['camera_body_axis_error_deg']<=tolerance:break
            if attempt==attempts:
                raise RuntimeError('Current camera axis did not align within the bounded seating corrections')
            for _ in range(round(settle/dt)):
                yield index,elapsed,np.radians(done),0.,'key_probe_nut_rotation_visual_refine'
                index+=1;elapsed+=dt
        checkpoint+=1
