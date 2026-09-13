"""Replay one current nut controller from a declared local source-stage state.

This adapter shares the production controller, hand runtime, images and robot
sensors. The cold scene comes from a stopped run for diagnosis only; it is not
a visual assembly acceptance episode.
"""
import copy
import json
from time import perf_counter
from pathlib import Path
import numpy as np
import yaml


def run_source_stage_probe(*,repository,args,world,robot_data,ft_tree,contact_view,
                          contact_paths,prepared,metadata,sensor_sample,source_rotation,recipe):
    import omni.usd
    import omni.replicator.core as rep
    from pxr import Gf,Usd,UsdGeom,UsdLux,PhysicsSchemaTools
    from omni.physx import get_physx_interface,get_physx_simulation_interface
    from isaacsim.core.prims import SingleRigidPrim
    from kcg_connector.grasp.carts_v2.models import load_v2_inputs
    from carts_v2 import controller
    from carts_v2.sample_store import GzipSampleStore
    from carts_v2.engine_health import PhysxStatsMonitor
    from carts_v2.evaluate_run import TruthAuditRecorder
    from carts_v2.run_grasp_lift import _HighObservationWristFtAuditor,_load_frozen_hand_inertials
    from te_foundationpose_handoff_runtime import _json_ready,_install_rgbd_resume_sync
    from te_foundationpose_handoff_plan import FullRobotCollisionScene
    from te_body_nut_regrasp import run_body_nut_regrasp
    from te_body_nut_rotation import run_body_nut_rotation
    from te_body_socket_observation import observe_released_plug_from_rgbd
    from te_body_assembly_video import BodyAssemblyVideo
    from trace_metadata import write_gzip_array
    import fcl

    repository=Path(repository);output=args.output;stage=omni.usd.get_context().get_stage()
    inputs=load_v2_inputs(repository,config_path=repository/recipe['base_config'],
        object_id=metadata['object_id'],finger_mechanism_path=args.finger_mechanism)
    dynamic=copy.deepcopy(yaml.safe_load((repository/recipe['base_config']).read_text())['dynamic'])
    dynamic.update(physics_dt_s=float(world.get_physics_dt()),closing_drive_maximum_effort_nm=3.5,
                   measured_effort_abort_action='record_only',arm_damping=float(metadata['effective_lift_arm_damping_nm_s_rad']))
    assembly_path=repository/recipe['assembly_config'];assembly=yaml.safe_load(assembly_path.read_text())
    scene=prepared['scene'];parts=[]
    for i,path in enumerate(scene['part_prim_paths']):
        part=SingleRigidPrim(path,name=f'source_probe_part_{i}',reset_xform_properties=False);part.initialize();parts.append(part)
    hand_path=next(p for p in contact_paths if p.endswith('/handbase_link'))
    recorder=TruthAuditRecorder(object_parts=parts,object_articulation=None,
        hand_base_prim=stage.GetPrimAtPath(hand_path),robot_model=inputs.robot_model,
        stage_modules=(Gf,Usd,UsdGeom),contact_interface=get_physx_simulation_interface(),
        path_decoder=PhysicsSchemaTools.intToSdfPath,roots={'robot':'/World/HandArm',**scene['roots']},
        expected_total_mass_kg=inputs.object_contract.model.mass_kg,
        part_bottom_offsets_m=scene['part_bottom_offsets_m'],table_top_z_m=scene['table_top_z_m'],
        physics_dt_s=dynamic['physics_dt_s'],engine_monitor=PhysxStatsMonitor(world.get_physics_context()),
        physics_step_interface=get_physx_interface(),tensor_contact_prim=contact_view,
        tensor_contact_sensor_paths=contact_paths,tensor_contact_max_count=32768)
    recorder.samples=GzipSampleStore(output/'truth_samples.jsonl.gz')
    ft_doc=json.loads((repository/'src/kcg_connector/config/te_visual_high_reobserve_v1.json').read_text())
    safety=ft_doc['wrist_ft_safety'];monitor=assembly['wrist_planned_contact_torque_monitor']
    phases=('visual_align','visual_refine','axial_settle','turn','hold')
    limits={f'key_probe_nut_rotation_{phase}':20. for phase in phases};limits['key_probe_nut_grip_hold']=20.
    ft=_HighObservationWristFtAuditor(ft_articulation=ft_tree,
        reaction_row=ft_tree._articulation_view._metadata.joint_indices['hand2arm']+1,
        robot_model=inputs.robot_model,
        hand_inertials=_load_frozen_hand_inertials(repository/ft_doc['source_evidence']['hand_inertial_source']['path']),
        gravity_m_s2=scene['gravity_m_s2'],physics_dt_s=dynamic['physics_dt_s'],task_rotation_world=np.eye(3),
        force_limit_n=safety['maximum_resultant_force_n'],torque_limit_nm=safety['maximum_resultant_torque_nm'],
        planned_contact_torque_limit_nm=monitor['legacy_reference_nm'],planned_contact_torque_action=monitor['action'],
        contact_force_limit_overrides_n=limits,
        planned_contact_force_time_constant_s=assembly['wrist_planned_contact_force_monitor']['time_constant_s'],
        dynamic_inertia_compensation_enabled=True,frozen_hand_positions=sensor_sample['active_positions_rad'][7:],
        dynamic_inertia_enabled_phases=('approach_above','wait_above_settled','approach_descent','settle','pregrasp_hold','tare'))
    ft.tare_canonical_sensor=np.asarray(sensor_sample['run_specific_tare_canonical_sensor_wrench'])
    ft.tare_gravity_sensor=np.asarray(sensor_sample['run_specific_tare_modeled_gravity_sensor_wrench'])
    ft.tare_model_residual_sensor=ft.tare_canonical_sensor-ft.tare_gravity_sensor
    ft._PLANNED_CONTACT_PHASE_PREFIXES=(*ft._PLANNED_CONTACT_PHASE_PREFIXES,'key_probe_')
    original_capture=recorder.capture
    def capture(**kwargs):original_capture(**kwargs);ft.capture(**kwargs)
    recorder.capture=capture
    robot,active,arm_indices,lower,upper,drive_audit=robot_data
    stepper=controller.JointSignalStepper(robot=robot,world=world,auditor=recorder,
        active_indices=active,arm_indices=arm_indices,arm_lower_limits=lower,arm_upper_limits=upper,
        settings=dynamic,render=False,robot_model=inputs.robot_model,
        payload_model=metadata['controller_outcome']['payload_compensation_model'])
    ft.stepper=stepper
    runtime={'world':world,'inputs':inputs,'scene':scene,'auditor':recorder,'robot_data':robot_data,
        'object_parts':parts,'nail_body_ft_auditor':ft,'body_assembly_control_config':str(assembly_path),
        'body_assembly_scene':prepared,'robot_asset':metadata['robot_asset']}
    runtime['simulation_stop_request_path']=str(output/'STOP_REQUEST')
    if 'initial_loaded_command_deg' in recipe:
        runtime['engagement_loaded_turn_command_deg']=float(recipe['initial_loaded_command_deg'])
    _install_rgbd_resume_sync(world,stage)
    light=UsdLux.DomeLight.Define(stage,'/World/SourceStageDiagnosticLighting')
    light.CreateIntensityAttr(float(scene['render'].dome_light_intensity))
    video=BodyAssemblyVideo(repository,runtime,output/'video',dynamic['physics_dt_s'],fps=5)
    source_transport=json.loads((args.run/'socket_transport/transport_and_observation.json').read_text())
    socket=np.asarray(source_transport['world_from_socket_wrist_visual']).reshape(4,4)
    video.freeze_main_target(socket[:3,3])
    grip=json.loads((args.run/'socket_transport'/args.source_grip_stage/'nut_regrasp_controller_result.json').read_text())
    arm=np.asarray(sensor_sample['active_targets_rad'][:7]);hand=np.asarray(sensor_sample['active_targets_rad'][7:])
    grip.update(fixed_arm_target_rad=arm.tolist(),final_hand_target_rad=hand.tolist())
    result={'scope':'LOCAL_SOURCE_STAGE_DIAGNOSIS_NOT_VISUAL_ASSEMBLY','source_run':str(args.run),
        'source_step':args.source_step,'source_setup_uses_recorded_pose':True,
        'post_start_object_pose_or_contact_truth_used_for_control':False,'source_free_space_tare_preserved':True,
        'hardware_authorized':False,'full_visual_assembly_success':False,'recipe':recipe}
    started=perf_counter()
    try:
        world.play()
        warmup=float(recipe.get('warmup_s',2.))
        if not .5<=warmup<=2.:raise ValueError('local warmup must be between0.5and2seconds')
        for _ in range(round(warmup/dynamic['physics_dt_s'])):
            stepper.advance('nut_index_free_open_hold' if 'free_joint7_delta_rad' in recipe else 'key_probe_nut_grip_hold',arm,hand)
            if stepper.abort_reason:raise RuntimeError(stepper.abort_reason)
        world.pause()
        q=stepper.latest[0];H=inputs.robot_model.forward_kinematics(tuple(q),enforce_limits=False)['handbase_link']
        observation=observe_released_plug_from_rgbd(repository,stage,world,rep,H,output/'current_palm')
        if not observation.get('position_and_axis_measured'):raise RuntimeError('Current diagnostic image did not resolve the connector axis')
        observation.update(source='CURRENT_SEGMENT_RGBD_AND_ENCODERS',encoder_step=int(stepper.step_index))
        result['initial_reference_check']={'nominal_arm_target_rad':arm.tolist(),
            'source_encoder_arm_rad':sensor_sample['active_positions_rad'][:7],
            'after_warmup_encoder_arm_rad':np.asarray(q[:7]).tolist(),
            'maximum_encoder_change_rad':float(np.max(np.abs(np.asarray(q[:7])-sensor_sample['active_positions_rad'][:7]))),
            'native_gravity_bias_reused_as_nominal_reference':False}
        collision=FullRobotCollisionScene(inputs);box=np.asarray(inputs.table_xy_bounds_m)
        fixture=prepared['fixture']
        obstacles={'table':fcl.CollisionObject(fcl.Box(*(np.r_[box[:,1]-box[:,0],1.])),fcl.Transform(np.r_[box.mean(axis=1),inputs.table_top_z_m-.5])),
            'fixture':fcl.CollisionObject(fcl.Box(*fixture['size_m']),fcl.Transform(fixture['center_world_m']))}
        geometry=run_body_nut_regrasp(repository,runtime,stepper,dynamic,observation,socket,
            collision,obstacles,output/'geometry',prepare_geometry_only=True)
        if not geometry.get('geometry_only'):raise RuntimeError(geometry.get('failure_reason','Geometry preparation failed'))
        if 'free_joint7_delta_rad' in recipe:
            from te_nut_motion import joint7_return_path
            delta=float(recipe['free_joint7_delta_rad'])
            if abs(delta)>np.pi/2+1e-12:raise ValueError('this free-return probe is bounded to90degrees')
            states,details=joint7_return_path(arm,arm[6]+delta,lower,upper,dynamic['physics_dt_s'])
            check=runtime['nut_regrasp_geometry_check']
            for state in states:
                hit=check(state,hand,nut_contact=False)
                if hit:raise RuntimeError('free joint7 path is obstructed: '+str(hit))
            np.save(output/'joint7_only_path_rad.npy',states)
            world.play()
            initial_encoder=float(stepper.latest[0][6])
            for state in states[1:]:
                stepper.advance('nut_index_free_rotate',state,hand)
                if stepper.abort_reason:raise RuntimeError(stepper.abort_reason)
            for _ in range(round(.5/dynamic['physics_dt_s'])):
                stepper.advance('nut_index_free_final_hold',states[-1],hand)
                if stepper.abort_reason:raise RuntimeError(stepper.abort_reason)
            details.update(actual_joint7_motion_rad=float(stepper.latest[0][6]-initial_encoder),
                final_tracking_error_rad=float(stepper.latest[0][6]-states[-1,6]),
                original_other_joint_targets_unchanged=True)
            result['free_joint7_return']=details
            result['free_return_completed']=abs(details['final_tracking_error_rad'])<.003
            return result
        if recipe.get('task_root_preload'):
            from te_three_finger_wrench_observer import ThreeFingerWrenchObserver
            preload=recipe['task_root_preload'];targets=np.asarray(preload['targets_nm'],float)
            if targets.shape!=(3,) or not np.isfinite(targets).all() or np.any(targets<=0):
                raise ValueError('Three finite positive base-moment references are required')
            root_sensor=ThreeFingerWrenchObserver(repository,inputs.robot_model,
                assembly['nut_regrasp']['geometry_plan'],sensor_semantics='BASE_BRIDGE_EXTERNAL_MOMENT_ABOUT_O')
            root_sensor.tare_reaction=np.asarray(grip['new_grasp_effort_tare_nm'][1:])
            root_sensor.tare_gravity=np.asarray(grip['root_moment_preload']['tare_gravity_nm'])
            lower_hand,upper_hand=np.asarray(grip['finite_preload_bounds_rad'])
            ramp=float(preload['ramp_duration_s']);period=float(preload['total_duration_s'])
            if not 0<ramp<=period<=3.:raise ValueError('This local preload is bounded to three seconds')
            h=float(dynamic['physics_dt_s']);gain=h/(1./6.+h)
            increment=min(float(dynamic['finger_maximum_speed_rad_s']),.15)*h
            def moment():
                encoder=np.asarray(stepper.latest[0]);gravity=root_sensor._system(encoder)[2]
                return np.asarray(stepper.latest[2])[8:]-root_sensor.tare_reaction-(gravity-root_sensor.tare_gravity)
            initial_moment=moment();check=runtime['nut_regrasp_geometry_check']
            with (output/'task_root_preload_samples.jsonl').open('x',buffering=1) as stream:
                world.play()
                for index in range(round(period/h)):
                    if (output/'STOP_REQUEST').exists():raise RuntimeError('explicit simulation pause requested')
                    measured=moment();u=min(1.,(index+1)*h/ramp);blend=10*u**3-15*u**4+6*u**5
                    reference=initial_moment+(targets-initial_moment)*blend
                    hand[1:]=np.clip(hand[1:]+np.clip(gain*(reference-measured)/120.,-increment,increment),lower_hand[1:],upper_hand[1:])
                    encoder=np.asarray(stepper.latest[0]);hit=check(encoder[:7],encoder[7:],nut_contact=True)
                    if hit:raise RuntimeError('Task preload geometry: '+str(hit))
                    stream.write(json.dumps({'step':int(stepper.step_index),'measured_base_moment_nm':measured.tolist(),
                        'reference_base_moment_nm':reference.tolist(),'hand_target_rad':hand.tolist()},separators=(',',':'))+'\n')
                    stepper.advance('key_probe_nut_grip_hold',arm,hand)
                    if stepper.abort_reason:raise RuntimeError(stepper.abort_reason)
                world.pause()
            grip.update(final_hand_target_rad=hand.tolist(),effort_reference_nm=targets.tolist())
            result['task_root_preload']={**preload,'initial_measured_nm':initial_moment.tolist(),
                'final_measured_nm':moment().tolist(),'final_hand_target_rad':hand.tolist(),
                'force_observation_limits_or_motor_caps_changed':False,'exact_normal_force_achievement_claimed':False}
            H=inputs.robot_model.forward_kinematics(tuple(stepper.latest[0]),enforce_limits=False)['handbase_link']
            observation=observe_released_plug_from_rgbd(repository,stage,world,rep,H,output/'after_task_preload_palm')
            if not observation.get('position_and_axis_measured'):raise RuntimeError('Fresh axis observation failed after task preload')
            observation.update(source='CURRENT_SEGMENT_RGBD_AND_ENCODERS',encoder_step=int(stepper.step_index))
            runtime['nut_regrasp_locate_visual_bounds'](np.asarray(observation['world_from_plug_five_dof']))
        settings=copy.deepcopy(assembly['nut_rotation_after_index'] if recipe.get('use_current_rotation_config',False) else source_rotation['settings'])
        if 'rotation_degrees' in recipe:
            degrees=float(recipe['rotation_degrees'])
            if not 0<degrees<=5.:raise ValueError('a short local control check is bounded to five degrees')
            settings['rotation_about_socket_plus_z_deg']=-degrees
        if recipe.get('single_attempt_diagnostic',False):settings['recovery']={'enabled':False}
        settings['planar_force_admittance']['virtual_restoring_stiffness_n_m']=float(recipe['virtual_restoring_stiffness_n_m'])
        if 'freeze_planar_after_preparation' in recipe:
            settings['planar_force_admittance']['freeze_after_preparation']=bool(recipe['freeze_planar_after_preparation'])
        if 'arm_kinematic_reference' in recipe:
            settings['arm_kinematic_reference']=recipe['arm_kinematic_reference']
        if 'grip_lateral_balance' in recipe:
            settings['grip_lateral_balance']=copy.deepcopy(recipe['grip_lateral_balance'])
        if 'engagement_axial_assist' in recipe:
            settings['engagement_axial_assist']=copy.deepcopy(recipe['engagement_axial_assist'])
        if 'finger_effort_limited_turn' in recipe:
            settings['finger_effort_limited_turn']=copy.deepcopy(recipe['finger_effort_limited_turn'])
        result['rotation']=run_body_nut_rotation(repository,runtime,stepper,dynamic,grip,socket,
            settings,output/'rotation',initial_position_axis_observation=observation)
    except Exception as error:
        result['error']=str(error)
        import traceback
        result['traceback']=traceback.format_exc()
    finally:
        world.pause();recorder.samples.close();video.close()
        write_gzip_array(output/'joint_ft_samples.json.gz',ft.samples,prepare=_json_ready)
        result.update(physical_steps=stepper.step_index,outer_abort=stepper.abort_reason,
            wrist_summary=ft.summary())
        result['wall_timing']={'local_run_and_closeout_s':perf_counter()-started,'stepper':stepper.wall_times,
            'hand_mechanism':dict(getattr(world.hand_mechanism,'wall_times',{})),
            'truth_capture':dict(recorder.capture_wall_times)}
        result['scene_output_backend']=getattr(world,'_kcg_rgbd_resume_sync_backend',None)
        (output/'source_stage_probe_result.json').write_text(json.dumps(_json_ready(result),indent=2)+'\n')
    return _json_ready(result)
