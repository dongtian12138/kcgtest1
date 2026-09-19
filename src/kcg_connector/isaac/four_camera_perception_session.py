"""Four-camera observations with one key anchor and delayed palm feedback."""
from pathlib import Path
from time import perf_counter
import json
import numpy as np
from four_camera_rig import configuration,camera_spec,intrinsics
from key_direction_memory import KeyDirectionMemory
from key_direction_memory import minimum_axis_rotation
from palm_five_dof_tracking import measure
from perception_latency import DelayedObservation


class FourCameraPerceptionSession:
    def __init__(self,repository,runtime,stepper,anchor=None,*,prekey_world_from_body=None):
        import yaml
        self.root=Path(repository);self.runtime=runtime;self.stepper=stepper
        self.world=runtime['world'];self.rig=configuration(self.root,runtime)
        if self.rig is None:raise ValueError('Declared four-camera rig required')
        if (anchor is None)==(prekey_world_from_body is None):
            raise ValueError('Provide either the single key anchor or an explicitly unkeyed transport frame')
        self.output=Path(runtime['output_directory'])/'four_camera_perception'
        self.output.mkdir(exist_ok=False)
        settings=yaml.safe_load((self.root/runtime['body_assembly_control_config']).read_text())['perception']
        self.period=float(settings.get('palm_update_period_s',.2))
        self.sensor_delay=float(settings.get('nominal_sensor_frame_period_s',.05))
        self.maximum_observation_age_s=.5
        if self.period<=0 or self.sensor_delay<=0:raise ValueError('Positive declared camera periods required')
        self.resources={};self.pending=None;self.next_request=float(self.world.current_time)
        self.last_observation=None
        self.camera_counts={};self.palm_count=0;self.consumed_count=0;self.closed=False
        self.memory=KeyDirectionMemory()
        self.key_anchor_pending=False
        self.prekey_relation=None
        spec,_=camera_spec(self.root,self.rig,'palm',self.hand())
        self.hand_from_palm=np.asarray(spec['mount']['hand_from_camera_cv'])
        self.cad=self.root/'artifacts/kcg_connector/vision/sam6d_segmentation_run19_observation_v1/D38999_26FJ35PN_VISUAL.obj'
        self.events=(self.output/'events.jsonl').open('x',buffering=1)
        if anchor is not None:
            self.adopt_key_anchor(anchor)
        else:
            self.last_body=np.asarray(prekey_world_from_body,float).reshape(4,4)
            self.prekey_relation=np.linalg.inv(self.hand())@self.last_body
            self.last_sample_time=float(self.world.current_time)
            self._write({'event':'UNKEYED_TRANSPORT_FRAME_STARTED',
                'physics_time_s':self.last_sample_time,'key_yaw_measured':False,
                'transverse_frame_is_chosen_for_motion_only':True})
        runtime['four_camera_perception_session']=self

    def hand(self):
        return np.asarray(self.runtime['inputs'].robot_model.forward_kinematics(
            tuple(self.stepper.latest[0]),enforce_limits=False)['handbase_link'])

    def body(self):
        if self.memory.active:return self.memory.predict(self.hand())
        if self.prekey_relation is not None:return self.hand()@self.prekey_relation
        return self.last_body.copy()

    def relation_for_transport(self):
        if self.memory.active:return self.memory.hand_from_body()
        if self.prekey_relation is not None:return self.prekey_relation.copy()
        raise RuntimeError('A retired Body grasp cannot supply a held transport frame')

    def adopt_key_anchor(self,anchor):
        count=anchor.get('key_observation_event_count')
        refinement=(count==2 and self.rig.get('maximum_key_observation_events')==2
                    and self.runtime.get('coarse_key_alignment_completed') is True
                    and self.key_anchor_pending)
        if not refinement and (count!=1 or self.memory.initialized):
            raise ValueError('A second key anchor requires the completed coarse turn and a fresh image')
        availability=float(anchor['observation_latency']['availability_physics_time_s'])
        if availability>float(self.world.current_time) or float(anchor['physics_time_s'])>availability:
            raise ValueError('The single key observation is not causally available yet')
        if self.pending is not None:
            self._write({'event':'OLDER_PREKEY_PENDING_FRAME_REPLACED_BY_SYNCHRONIZED_KEY_ANCHOR',
                         'sample_time_s':self.pending.sample_time_s})
            self.pending=None
        args=(anchor['world_from_hand_encoder'],
            np.asarray(anchor['key_measurement']['world_from_plug_row_major']).reshape(4,4),anchor['physics_time_s'])
        correction=self.memory.reobserve_after_coarse_turn(*args) if refinement else self.memory.initialize(*args)
        palm=anchor['palm_observation']
        camera_from_body=np.linalg.inv(np.asarray(palm['world_from_camera_cv']))@np.asarray(palm['world_from_plug_five_dof'])
        self.memory.update_palm(self.hand_from_palm,camera_from_body,anchor['physics_time_s'])
        self.prekey_relation=None
        self.last_body=np.asarray(palm['world_from_plug_five_dof'])
        self.last_sample_time=float(anchor['physics_time_s'])
        self.key_anchor_pending=False
        self._write({'event':'KEY_ANCHOR_REOBSERVED_AFTER_COARSE_TURN' if refinement else 'SINGLE_KEY_ANCHOR_INITIALIZED',
                     'sample_time_s':self.last_sample_time,'key_observation_event_count':count,
                     'consumed_time_s':float(self.world.current_time),'refinement':correction})

    def begin_key_refinement(self):
        if (self.rig.get('maximum_key_observation_events')!=2 or self.memory.anchor_count!=1
                or self.runtime.get('coarse_key_alignment_completed') is not True):
            raise RuntimeError('The declared coarse turn must finish before the second key sample')
        self.pending=None
        self.key_anchor_pending=True
        self._write({'event':'PALM_UPDATES_HELD_FOR_SYNCHRONIZED_KEY_REFINEMENT',
                     'physics_time_s':float(self.world.current_time)})

    def _write(self,event):
        from te_foundationpose_handoff_runtime import _json_ready
        self.events.write(json.dumps(_json_ready(event),separators=(',',':'))+'\n')

    def capture(self,role,folder):
        if role not in ('palm','wrist'):
            raise ValueError('After key initialization, this session only acquires mounted palm/wrist images')
        import omni.replicator.core as rep
        import omni.usd
        from pxr import Gf,UsdGeom
        from te_foundationpose_handoff_runtime import _author_camera,_capture_rgbd
        hand=self.hand();spec,camera=camera_spec(self.root,self.rig,role,hand)
        timestamp=float(self.world.current_time);step=int(self.stepper.step_index)-1
        playing=self.world.is_playing();self.world.pause();started=perf_counter()
        try:
            _author_camera(omni.usd.get_context().get_stage(),spec['prim_path'],camera,
                resolution=tuple(spec['resolution_px']),focal_length_mm=spec['focal_length_mm'],
                horizontal_aperture_mm=spec['horizontal_aperture_mm'],
                clipping_range_m=tuple(spec['clipping_range_m']),Gf=Gf,UsdGeom=UsdGeom)
            self.world.render()
            capture=_capture_rgbd(rep=rep,resources=self.resources,camera_path=spec['prim_path'],
                resolution=tuple(spec['resolution_px']),output_dir=folder,warmup_frames=3,rt_subframes=4)
        finally:
            if playing:self.world.play()
        if float(self.world.current_time)!=timestamp:
            raise RuntimeError('Sensor sample and encoder timestamp were not synchronized')
        self.camera_counts[role]=self.camera_counts.get(role,0)+1
        return {'capture':capture,'sample_time_s':timestamp,'sample_step':step,
                'hand':hand,'camera':camera,'intrinsics':intrinsics(spec),
                'depth':np.load(folder/'depth_m.npy'),'render_and_file_io_wall_s':perf_counter()-started}

    def service(self,*,request_new=True):
        """Called between ordinary controller steps; never advances physics itself."""
        if self.key_anchor_pending:return
        now=float(self.world.current_time)
        if self.pending is not None and self.pending.ready(now):
            payload=self.pending.payload;measurement=payload['measurement']
            self.last_body=np.asarray(measurement['world_from_plug_five_dof'])
            self.last_sample_time=self.pending.sample_time_s
            update=None
            if self.memory.active:
                update=self.memory.update_palm(self.hand_from_palm,
                    measurement['camera_from_plug_five_dof'],self.pending.sample_time_s)
            elif self.prekey_relation is not None:
                measured=self.hand_from_palm@np.asarray(measurement['camera_from_plug_five_dof'])
                rotation=minimum_axis_rotation(self.prekey_relation[:3,2],measured[:3,2])
                self.prekey_relation[:3,:3]=rotation@self.prekey_relation[:3,:3]
                self.prekey_relation[:3,3]=measured[:3,3]
            observed=payload['observation']
            self.last_observation={
                'status':'DELAYED_CURRENT_PALM_POSITION_AND_AXIS',
                'position_and_axis_measured':True,
                'physics_time_s':self.pending.sample_time_s,
                'capture_physics_time_s':self.pending.sample_time_s,
                'available_physics_time_s':self.pending.available_time_s,
                'consumed_physics_time_s':now,
                'capture':observed['capture'],
                'rgbd_directory':payload['directory'],
                'world_from_hand_encoder':observed['hand'].tolist(),
                'world_from_camera_cv':observed['camera'].tolist(),
                'intrinsics_3x3':observed['intrinsics'].tolist(),
                'world_from_plug_five_dof':self.last_body.tolist(),
                'metrics':measurement['metrics'],
                'measurement':{'geometry':{'metrics':measurement['metrics'],
                    'camera_from_object':measurement['camera_from_plug_five_dof']}},
                'tracking_seed_mask':str(Path(payload['directory'])/'roi.png'),
                'axial_yaw_measured':False,
                'previous_hand_body_relation_used_for_measurement':False,
                'previous_pose_used_only_for_search_region':True,
                'online_object_or_contact_truth_used':False}
            self.consumed_count+=1
            self._write({'event':'PALM_MEASUREMENT_CONSUMED','sample_step':payload['sample_step'],
                'sample_time_s':self.pending.sample_time_s,'available_time_s':self.pending.available_time_s,
                'consumed_time_s':now,'consumed_step':int(self.stepper.step_index),
                'intervening_physics_steps':int(self.stepper.step_index)-payload['sample_step']-1,
                'key_update':update,'observation_directory':payload['directory']})
            self.pending=None
        if request_new and self.pending is None and now>=self.next_request:
            predicted=self.body();folder=self.output/f'palm_{self.palm_count:05d}'
            frame=self.capture('palm',folder)
            from te_foundationpose_handoff_runtime import _json_ready
            (folder/'measurement_input.json').write_text(json.dumps(_json_ready({
                **{k:v for k,v in frame.items() if k!='depth'},
                'predicted_world_from_body_for_roi_only':predicted,
                'cad_path':str(self.cad),'depth_file':str(folder/'depth_m.npy'),
                'online_object_or_contact_truth_used':False}),indent=2)+'\n')
            started=perf_counter()
            measurement,mask=measure(frame['depth'],frame['intrinsics'],frame['camera'],predicted,self.cad)
            compute=perf_counter()-started
            import cv2
            cv2.imwrite(str(folder/'roi.png'),mask.astype(np.uint8)*255)
            delta=float(np.linalg.norm(np.asarray(measurement['world_from_plug_five_dof'])[:3,3]-predicted[:3,3]))
            if delta>.002:raise RuntimeError('Palm position change exceeds the existing2mm bounded observation region')
            self.pending=DelayedObservation(frame['sample_time_s'],self.sensor_delay,compute,
                {'measurement':measurement,'sample_step':frame['sample_step'],'directory':str(folder)})
            from te_foundationpose_handoff_runtime import _json_ready
            record={**{k:v for k,v in frame.items() if k!='depth'},'measurement':measurement,
                    'estimation_wall_s':compute,'nominal_sensor_delay_s':self.sensor_delay,
                    'available_time_s':self.pending.available_time_s,
                    'delay_model':'NOMINAL_SENSOR_FRAME_PERIOD_PLUS_MEASURED_GEOMETRY_ESTIMATION',
                    'render_and_file_io_are_simulation_overhead_not_hardware_sensor_latency':True,
                    'hardware_latency_calibrated':False,'online_object_or_contact_truth_used':False}
            self.pending.payload['observation']=record
            (folder/'observation.json').write_text(json.dumps(_json_ready(record),indent=2)+'\n')
            self._write({'event':'PALM_SAMPLE_PENDING','sample_time_s':frame['sample_time_s'],
                'sample_step':frame['sample_step'],'available_time_s':self.pending.available_time_s,
                'observation_directory':str(folder)})
            self.palm_count+=1;self.next_request=frame['sample_time_s']+self.period

    def available_observation(self, output):
        """Return the latest causally available image, without pausing control for a new result."""
        import copy
        from te_foundationpose_handoff_runtime import _json_ready
        self.service(request_new=False)
        if self.last_observation is None:
            raise RuntimeError('No consumed palm observation is available')
        age=float(self.world.current_time)-self.last_observation['capture_physics_time_s']
        if age > self.maximum_observation_age_s:
            raise RuntimeError('Palm observation is older than the declared0.5s maximum age')
        result=_json_ready(copy.deepcopy(self.last_observation))
        result.update(consumed_by_stage_at_physics_time_s=float(self.world.current_time),
                      sample_age_at_stage_consumption_s=age,
                      new_image_acquired_by_this_function=False)
        output=Path(output);output.mkdir(parents=True,exist_ok=False)
        (output/'camera_and_estimate.json').write_text(json.dumps(result,indent=2)+'\n')
        return result

    def retire_body_grasp(self):
        self.last_body=self.body();self.memory.retire('BODY_RELEASED_AFTER_GUIDED_ENTRY')
        self.prekey_relation=None
        self._write({'event':'BODY_GRASP_REFERENCE_RETIRED','time_s':float(self.world.current_time),
                     'step':int(self.stepper.step_index)})

    def close(self):
        if self.closed:return
        from te_foundationpose_handoff_runtime import _close_rgbd_resources
        _close_rgbd_resources(self.resources)
        (self.output/'summary.json').write_text(json.dumps({'camera_counts':self.camera_counts,
            'key_observation_event_count':self.memory.anchor_count,'palm_samples':self.palm_count,
            'palm_measurements_consumed':self.consumed_count,'key_memory':self.memory.report(),
            'hardware_latency_calibrated':False,'online_object_or_contact_truth_used':False},indent=2)+'\n')
        self.events.close();self.closed=True
