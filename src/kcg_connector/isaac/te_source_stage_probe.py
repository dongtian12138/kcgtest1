"""Replay one current nut controller from a declared local source-stage state.

This adapter shares the production controller, hand runtime, images and robot
sensors. The cold scene comes from a stopped run for diagnosis only; it is not
a visual assembly acceptance episode.
"""
import copy
import gzip
import json
from time import perf_counter
from pathlib import Path
import numpy as np
import yaml


def run_source_stage_probe(*,repository,args,world,robot_data,ft_tree,contact_view,
                          contact_paths,prepared,metadata,sensor_sample,source_rotation,recipe,
                          source_visual_context_run=None):
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
    from te_nut_motion import source_probe_rotation_degrees
    from trace_metadata import write_gzip_array
    import fcl

    degrees=source_probe_rotation_degrees(recipe)
    repository=Path(repository);output=args.output;stage=omni.usd.get_context().get_stage()
    inputs=load_v2_inputs(repository,config_path=repository/recipe['base_config'],
        object_id=metadata['object_id'],finger_mechanism_path=args.finger_mechanism)
    dynamic=copy.deepcopy(yaml.safe_load((repository/recipe['base_config']).read_text())['dynamic'])
    dynamic.update(physics_dt_s=float(world.get_physics_dt()),
                   closing_drive_maximum_effort_nm=float(world.hand_mechanism.settings['finger_transmission_boundary_nm']),
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
        tensor_contact_sensor_paths=contact_paths,tensor_contact_max_count=32768,
        contact_audit_mode=args.contact_audit_mode)
    codec=args.truth_archive_codec
    recorder.samples=GzipSampleStore(output/f'truth_samples.{codec}.gz',block_size=64,cache_blocks=1,codec=codec)
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
    runtime['declared_source_stage_diagnostic']=True
    runtime['inspection_ui_enabled']=bool(recipe.get('inspection_ui_enabled',False))
    runtime['simulation_stop_request_path']=str(output/'STOP_REQUEST')
    stepper.diagnostic_stop_request_path=runtime['simulation_stop_request_path']
    if 'maximum_physical_steps' in recipe:
        count=recipe['maximum_physical_steps']
        if isinstance(count,bool) or not isinstance(count,int) or not 1<=count<=200000:
            raise ValueError('diagnostic physical step limit must be a positive bounded integer')
        stepper.diagnostic_step_limit=count
    if 'initial_loaded_command_deg' in recipe:
        runtime['engagement_loaded_turn_command_deg']=float(recipe['initial_loaded_command_deg'])
        runtime['coaxial_nut_commanded_degrees']=float(recipe['initial_loaded_command_deg'])
    _install_rgbd_resume_sync(world,stage)
    light=UsdLux.DomeLight.Define(stage,'/World/SourceStageDiagnosticLighting')
    light.CreateIntensityAttr(float(scene['render'].dome_light_intensity))
    video=BodyAssemblyVideo(repository,runtime,output/'video',dynamic['physics_dt_s'],fps=5)
    source_transport=json.loads((Path(source_visual_context_run or args.run)/'socket_transport/transport_and_observation.json').read_text())
    socket=np.asarray(source_transport['world_from_socket_wrist_visual']).reshape(4,4)
    video.freeze_main_target(socket[:3,3])
    grip=json.loads((args.run/args.source_stage_root/args.source_grip_stage/'nut_regrasp_controller_result.json').read_text())
    runtime['nut_grasp_yaw_bias_deg']=float(grip.get('grasp_yaw_bias',{}).get('desired_total_bias_deg',0.))
    arm=np.asarray(sensor_sample['active_targets_rad'][:7]);hand=np.asarray(sensor_sample['active_targets_rad'][7:])
    grip.update(fixed_arm_target_rad=arm.tolist(),final_hand_target_rad=hand.tolist())
    result={'scope':'LOCAL_SOURCE_STAGE_DIAGNOSIS_NOT_VISUAL_ASSEMBLY','source_run':str(args.run),
        'source_step':args.source_step,'source_setup_uses_recorded_pose':True,
        'post_start_object_pose_or_contact_truth_used_for_control':False,'source_free_space_tare_preserved':True,
        'hardware_authorized':False,'full_visual_assembly_success':False,'recipe':recipe}
    result['time_resolution']={'physics_hz':1./dynamic['physics_dt_s'],
        'experimental_comparison':bool(args.experimental_connector_time_resolution),
        'validated_960hz_connector_results_transfer_automatically':False}
    started=perf_counter()
    profiler=None
    if recipe.get('profile_controller',False):
        import cProfile
        profiler=cProfile.Profile();profiler.enable()
    if recipe.get('adaptive_fourbar_updates',False):
        world.hand_mechanism.adaptive_tangent_settings={'closure_tolerance_m':1e-8,'maximum_slope_error':1e-4}
    try:
        world.play()
        warmup=float(recipe.get('warmup_s',2.))
        if not .5<=warmup<=2.:raise ValueError('local warmup must be between0.5and2seconds')
        warmup_steps=round(warmup/dynamic['physics_dt_s'])
        if 'free_joint7_delta_rad' not in recipe:
            history_s=float(assembly['nut_rotation_after_index'].get('contact_filter_initialization_history_s',.5))
            # The first sample has no preceding state for causal acceleration.
            warmup_steps=max(warmup_steps,1+round(history_s/dynamic['physics_dt_s']))
        result['warmup_physical_steps']=warmup_steps
        for _ in range(warmup_steps):
            warm_phase=('nut_index_free_open_hold' if 'free_joint7_delta_rad' in recipe
                else 'key_probe_nut_open_hold' if recipe.get('perform_current_regrasp') else 'key_probe_nut_grip_hold')
            stepper.advance(warm_phase,arm,hand)
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
        perform_grip=bool(recipe.get('perform_current_regrasp',False))
        geometry=run_body_nut_regrasp(repository,runtime,stepper,dynamic,observation,socket,
            collision,obstacles,output/('current_grip' if perform_grip else 'geometry'),
            prepare_geometry_only=not perform_grip)
        if perform_grip:
            result['current_grip']=geometry
            if not geometry.get('completed'):raise RuntimeError(geometry.get('failure_reason','Current regrasp failed'))
            grip=geometry
        elif not geometry.get('geometry_only'):raise RuntimeError(geometry.get('failure_reason','Geometry preparation failed'))
        if recipe.get('preserve_source_grip_state'):
            if perform_grip or not recipe.get('motor_input_state'):
                raise ValueError('Preserving a cold source grip requires its recorded finite motor state and no regrasp')
            source_file=args.run/args.source_stage_root/args.source_grip_stage/'joint_ft_samples.json.gz'
            with gzip.open(source_file,'rt') as stream:source_rows=json.load(stream)
            opened=[r for r in source_rows if r['phase']=='key_probe_nut_tare']
            if len(opened)<20:raise ValueError('Declared source grip has no open-hand calibration')
            runtime['coaxial_declared_cold_grip_calibration']={
                'source_file':str(source_file.resolve()),'open_history':opened}
            history_path=output/'diagnostic_loaded_robot_history.json.gz'
            write_gzip_array(history_path,ft.samples,prepare=_json_ready)
            grip.update(robot_sensor_history_file=str(history_path.resolve()),
                world_from_body_palm_five_dof=observation['world_from_plug_five_dof'])
            result['preserved_source_grip_diagnostic']={
                'motor_state_source_step':args.source_step,'earlier_robot_sensor_zero':str(source_file),
                'new_loaded_sensor_history':str(history_path),'new_regrasp_performed':False,
                'native_contact_warm_start_restored':False,'same_episode_visual_success_claimed':False}
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
        if recipe.get('loaded_motion_profile'):
            profile=recipe['loaded_motion_profile']
            speed=float(profile['maximum_rotation_speed_deg_s'])
            acceleration=float(profile['maximum_rotation_acceleration_deg_s2'])
            arm_speed=float(profile['maximum_arm_speed_rad_s'])
            axial_speed=float(profile.get('maximum_axial_speed_m_s',settings['maximum_axial_speed_m_s']))
            axial_headroom=axial_speed-.00762*speed/360.
            if not (0<speed<=30. and 0<acceleration<=90. and 0<arm_speed<=.65 and 0<axial_speed<=.002 and axial_headroom>=.00008):
                raise ValueError('local loaded profile exceeds arm/axial speed reserve or acceleration range')
            settings.update(maximum_rotation_speed_deg_s=speed,
                maximum_rotation_acceleration_deg_s2=acceleration,maximum_arm_speed_rad_s=arm_speed,
                maximum_axial_speed_m_s=axial_speed)
            result['loaded_motion_profile']={**profile,'axial_speed_headroom_m_s':axial_headroom,
                'physical_motor_wrench_geometry_boundaries_changed':False}
        if degrees is not None:
            settings['rotation_about_socket_plus_z_deg']=-degrees
        if recipe.get('single_attempt_diagnostic',False):settings['recovery']={'enabled':False}
        if recipe.get('already_loaded_short_window',False):
            # This diagnostic starts inside an established guided, loaded turn.
            # Repeating acquisition/alignment would test a different transition.
            from te_nut_motion import ScalarMotion
            motion=ScalarMotion(float(recipe.get('rotation_degrees',0)),
                float(settings['maximum_rotation_speed_deg_s']),float(settings['maximum_rotation_acceleration_deg_s2']))
            if not recipe.get('single_attempt_diagnostic') or not 0<motion.duration<=12.:
                raise ValueError('the already-loaded window requires one turn lasting at most12simulation seconds')
            for name in ('pre_turn_visual_alignment','pre_turn_torsional_compliance','planar_force_admittance'):
                settings.setdefault(name,{})['enabled']=False
            settings.pop('pre_turn_visual_feedback',None)
            settings['axial_settle_duration_s']=0.
            # A restored local window does not create a fresh preparation history.
            # Explicit loaded_preparation_s below can request and validate one.
            settings['require_preparation_ready']=False
            settings['post_rotation_hold_s']=.25
            result['local_preparation_scope']='CURRENT_RGBD_GUIDE_CHECK; NO_NEW_GRASP_OR_ALIGNMENT'
        settings['planar_force_admittance']['virtual_restoring_stiffness_n_m']=float(recipe['virtual_restoring_stiffness_n_m'])
        if 'freeze_planar_after_preparation' in recipe:
            settings['planar_force_admittance']['freeze_after_preparation']=bool(recipe['freeze_planar_after_preparation'])
        if recipe.get('continuous_planar_force_admittance',False):
            settings['planar_force_admittance'].update(enabled=True,freeze_after_preparation=False)
        if 'arm_kinematic_reference' in recipe:
            settings['arm_kinematic_reference']=recipe['arm_kinematic_reference']
        if 'grip_lateral_balance' in recipe:
            settings['grip_lateral_balance']=copy.deepcopy(recipe['grip_lateral_balance'])
        if 'engagement_axial_assist' in recipe:
            settings['engagement_axial_assist']=copy.deepcopy(recipe['engagement_axial_assist'])
        if 'finger_effort_limited_turn' in recipe:
            settings['finger_effort_limited_turn']=copy.deepcopy(recipe['finger_effort_limited_turn'])
        if 'regulate_finger_effort_during_rotation' in recipe:
            settings['regulate_finger_effort_during_rotation']=bool(recipe['regulate_finger_effort_during_rotation'])
        if 'finger_motor_force_control' in recipe:
            settings['finger_motor_force_control']=copy.deepcopy(recipe['finger_motor_force_control'])
        if 'loaded_coordination' in recipe:
            settings['loaded_coordination']=copy.deepcopy(recipe['loaded_coordination'])
            settings['continue_established_axial_force_reference']=bool(recipe.get('continue_established_axial_force_reference',False))
        if 'loaded_preparation_s' in recipe:
            period=float(recipe['loaded_preparation_s'])
            if not .5<=period<=2.:raise ValueError('local force preparation is bounded to0.5..2seconds')
            settings.update(axial_settle_duration_s=period,regulate_finger_effort_during_preparation=True,
                require_preparation_ready=True)
        if 'visual_progress_period_s' in recipe:
            period=float(recipe['visual_progress_period_s'])
            if not .1<=period<=1.:raise ValueError('local current-image period must be0.1..1second')
            settings['visual_progress']['observation_period_s']=period
        if 'arm_transverse_load_compensation' in recipe:
            settings['arm_transverse_load_compensation']=copy.deepcopy(recipe['arm_transverse_load_compensation'])
        if recipe.get('successful_stroke_motion',{}).get('enabled',False):
            # Transfer the proved arm-motion policy, while the current hand's
            # observer and finite self-lock motor inverse remain authoritative.
            settings['successful_stroke_motion']=copy.deepcopy(recipe['successful_stroke_motion'])
            settings['loaded_coordination']={'enabled':False}
            settings['axial_lead_following']={'enabled':False}
            settings['arm_kinematic_reference']='measured_pose'
            settings['visual_progress']['reference_following']={'enabled':False}
            settings['planar_force_admittance']['enabled']=False
            settings['arm_transverse_load_compensation']={'enabled':False}
            settings['rotation_profile']='minimum_jerk'
            if recipe.get('already_loaded_short_window',False):
                settings['axial_force_reference_n']=settings['turn_axial_force_reference_n']
        result['rotation']=run_body_nut_rotation(repository,runtime,stepper,dynamic,grip,socket,
            settings,output/'rotation',initial_position_axis_observation=observation)
        if (recipe.get('additional_strokes',0) or recipe.get('release_after_rotation')) and result['rotation'].get('completed'):
            from te_body_nut_continuation import continue_nut_strokes_and_release
            additional=int(recipe['additional_strokes'])
            if not 0<=additional<=3:raise ValueError('Local continuation is bounded to three additional strokes')
            series_config=copy.deepcopy(assembly)
            origin=float(recipe.get('initial_loaded_command_deg',0.))
            command_budget=float(recipe.get('maximum_total_command_deg',
                origin+abs(float(settings['rotation_about_socket_plus_z_deg']))+90.*additional
                if series_config['continued_nut_strokes'].get('complete_remaining_command_on_last_stroke')
                else 90.*(1+additional)))
            if not np.isfinite(command_budget) or not origin<command_budget<=380.:
                raise ValueError('Local continuation needs a finite cumulative command budget within380degrees')
            series_config['continued_nut_strokes'].update(maximum_additional_strokes=additional,
                maximum_total_command_deg=command_budget,release_at_end=True)
            sequence=output/'sequence';sequence.mkdir(exist_ok=False)
            record={'completed':True,'nut_regrasp':grip,'nut_rotation':result['rotation']}
            def save_sequence():(sequence/'sequence_controller_result.json').write_text(json.dumps(_json_ready(record),indent=2)+'\n')
            continue_nut_strokes_and_release(repository,runtime,stepper,dynamic,record,socket,
                series_config,collision,obstacles,sequence,save_sequence)
            result['sequence']=record
    except Exception as error:
        result['error']=str(error)
        import traceback
        result['traceback']=traceback.format_exc()
    finally:
        if profiler is not None:
            profiler.disable();profiler.dump_stats(str(output/'python_controller.prof'))
        world.pause();recorder.samples.close()
        # Preserve the first failure even when a zero-frame video cannot be
        # finalized. Encoding must not replace the physical/adapter error.
        (output/'source_stage_probe_result.json').write_text(json.dumps(_json_ready(result),indent=2)+'\n')
        try:video.close()
        except Exception as error:result['video_close_error']=str(error)
        write_gzip_array(output/'joint_ft_samples.json.gz',ft.samples,prepare=_json_ready)
        result.update(physical_steps=stepper.step_index,outer_abort=stepper.abort_reason,
            wrist_summary=ft.summary() if len(ft.samples) else {'status':'NO_RECORDED_SAMPLES'})
        result['wall_timing']={'local_run_and_closeout_s':perf_counter()-started,'stepper':stepper.wall_times,
            'hand_mechanism':dict(getattr(world.hand_mechanism,'wall_times',{})),
            'truth_capture':dict(recorder.capture_wall_times)}
        result['scene_output_backend']=getattr(world,'_kcg_rgbd_resume_sync_backend',None)
        result['fourbar_tangent_updates']=getattr(world.hand_mechanism,'tangent_update_stats',None)
        (output/'source_stage_probe_result.json').write_text(json.dumps(_json_ready(result),indent=2)+'\n')
    return _json_ready(result)
