"""Ended-episode comparison of measured Body motion and the single-key tracker."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

root=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(root/'src/kcg_connector/isaac'),str(root/'src/kcg_connector/isaac/carts_v2')]
from trace_metadata import iter_truth_fields
from key_direction_memory import KeyDirectionMemory

def pose(position,quaternion):
    result=np.eye(4);result[:3,3]=position
    w,x,y,z=quaternion;result[:3,:3]=Rotation.from_quat([x,y,z,w]).as_matrix()
    return result

def errors(predicted,actual):
    axis=actual[:3,2];key=predicted[:3,1]-axis*float(axis@predicted[:3,1]);key/=np.linalg.norm(key)
    return {'center_error_m':float(np.linalg.norm(predicted[:3,3]-actual[:3,3])),
        'axis_error_deg':float(np.degrees(np.arccos(np.clip(predicted[:3,2]@axis,-1,1)))),
        'key_direction_error_deg':float(np.degrees(np.arccos(np.clip(predicted[:3,1]@actual[:3,1],-1,1)))),
        'axial_key_error_deg':float(np.degrees(np.arctan2(axis@np.cross(key,actual[:3,1]),key@actual[:3,1])))}

def main(run):
    run=Path(run).resolve();transport=json.loads((run/'socket_transport/transport_and_observation.json').read_text())
    anchor=json.loads((run/'postgrasp_key/camera_and_estimate.json').read_text())
    memory=KeyDirectionMemory();memory.initialize(anchor['world_from_hand_encoder'],np.array(anchor['key_measurement']['world_from_plug_row_major']).reshape(4,4),anchor['physics_time_s'])
    palm=anchor['palm_observation'];camera=np.array(palm['world_from_camera_cv']);H=np.array(anchor['world_from_hand_encoder'])
    mount=np.linalg.inv(H)@camera
    memory.update_palm(mount,np.linalg.inv(camera)@np.array(palm['world_from_plug_five_dof']),anchor['physics_time_s'])
    frozen=np.array(anchor['hand_from_body_visual_memory']);events=[];last_held_step=None
    key_availability_step=int(anchor['observation_latency']['availability_step'])
    for line in (run/'four_camera_perception/events.jsonl').read_text().splitlines():
        event=json.loads(line)
        if event['event']=='BODY_GRASP_REFERENCE_RETIRED':last_held_step=int(event['step'])-1
        if (event['event']=='PALM_MEASUREMENT_CONSUMED' and event.get('key_update') is not None
                and event['sample_time_s']>=anchor['physics_time_s']
                and event['consumed_step']>=key_availability_step):
            observed=json.loads((Path(event['observation_directory'])/'observation.json').read_text())
            events.append((event,observed))
    phases={};metrics={};old_metrics={};last=None;index=0;first=key_availability_step;trajectory=[]
    fields=('phase','simulation_time_s','hand_base_position_m','hand_base_orientation_wxyz',
            'object_part_positions_m','object_part_orientations_wxyz','active_velocities_rad_s')
    for row in iter_truth_fields(run,fields,first_step=first,last_step=last_held_step):
        while index<len(events) and events[index][0]['consumed_step']<=row['step']:
            event,observed=events[index];memory.update_palm(mount,observed['measurement']['camera_from_plug_five_dof'],event['sample_time_s']);index+=1
        H=pose(row['hand_base_position_m'],row['hand_base_orientation_wxyz'])
        actual=pose(row['object_part_positions_m'][0],row['object_part_orientations_wxyz'][0]);predicted=memory.predict(H)
        error=errors(predicted,actual);old_error=errors(H@frozen,actual)
        for key,value in error.items():metrics[key]=max(metrics.get(key,0),abs(value))
        for key,value in old_error.items():old_metrics[key]=max(old_metrics.get(key,0),abs(value))
        phase=phases.setdefault(row['phase'],{'first_step':row['step'],'last_step':row['step'],'sampled_path_m':0.,'first_position_m':actual[:3,3].tolist(),'peak_arm_speed_rad_s':0.})
        phase['last_step']=row['step'];phase['last_position_m']=actual[:3,3].tolist()
        if last is not None:phase['sampled_path_m']+=float(np.linalg.norm(actual[:3,3]-last))
        phase['peak_arm_speed_rad_s']=max(phase['peak_arm_speed_rad_s'],float(np.max(abs(np.asarray(row['active_velocities_rad_s'])[:7]))))
        last=actual[:3,3];final={'step':row['step'],'updated':error,'frozen':old_error,'world_from_body_truth':actual.tolist(),'world_from_body_tracker':predicted.tolist()}
        if row['step'] % 32 == 0:
            trajectory.append({'step':row['step'],'phase':row['phase'],'position_world_m':actual[:3,3].tolist(),
                               'key_direction_world':actual[:3,1].tolist(),'tracker_key_world':predicted[:3,1].tolist()})
    palm_accuracy=[]
    from trace_metadata import read_truth_sample
    for event,observation in events:
        raw=read_truth_sample(run,observation['sample_step']);actual=pose(raw['object_part_positions_m'][0],raw['object_part_orientations_wxyz'][0])
        measured=np.array(observation['measurement']['world_from_plug_five_dof']);e=errors(measured,actual)
        palm_accuracy.append({'step':raw['step'],'center_error_m':e['center_error_m'],'axis_error_deg':e['axis_error_deg']})
    wrist_path=run/'socket_transport/wrist_socket/camera_and_estimate.json'
    wrist=None
    if wrist_path.exists():
        record=json.loads(wrist_path.read_text());wrist={'key_direction_measured':record['measurement']['key_direction_measured'],'measurement':record['measurement']}
    result={'scope':'ENDED_HELD_BODY_SINGLE_KEY_LIFETIME_REVIEW','truth_used_only_after_motion':True,
        'body_grasp_reference_retired_after_step':last_held_step,
        'key_observation_sample_step':int(anchor['robot_sample_step']),
        'key_first_available_step':key_availability_step,
        'controller_completed':transport['completed'],'controller_stage':transport['stage'],
        'key_observation_events':anchor['key_observation_event_count'],'palm_consumed':index,
        'maximum_tracker_errors':metrics,'maximum_initial_frozen_memory_errors':old_metrics,
        'final':final,'phases':phases,'palm_current_frame_accuracy':palm_accuracy,'wrist':wrist,
        'full_assembly_claimed':False,'extra_axial_slip_is_not_observed_by_online_palm':True,
        'pose_error_hand_transform_source':'RECORDED_ENCODER_FK_SAME_AS_ONLINE_CONTROL; BODY_POSE_IS_POSTHOC_TRUTH'}
    (run/'four_camera_transport_posthoc.json').write_text(json.dumps(result,indent=2)+'\n')
    (run/'four_camera_transport_trajectory_30hz.json').write_text(json.dumps(trajectory,separators=(',',':'))+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('phases','palm_current_frame_accuracy','wrist')},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run');main(parser.parse_args().run)
