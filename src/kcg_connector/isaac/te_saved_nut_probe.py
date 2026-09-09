"""Run the existing bounded nut controller from a declared local initial state.

The scene is authored by diagnose_saved_hand_wrench before measurement. Online
reference geometry comes from the source episode's ordinary RGB-D and current
encoders. Native object/contact data only go to the offline recorder.
"""
import copy
import gzip
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import yaml


def run_probe(*, repository, args, world, robot_data, ft_tree, plug_tree,
              contact_view, contact_paths, prepared_scene, source_metadata,
              source_sensor, source_rotation, passive_initialization,
              base_run, assembly_config_path):
    import fcl
    import omni.usd
    from pxr import Gf, Usd, UsdGeom, UsdLux, PhysicsSchemaTools
    from omni.physx import get_physx_interface, get_physx_simulation_interface
    from isaacsim.core.prims import SingleRigidPrim
    from kcg_connector.grasp.carts_v2.models import load_v2_inputs
    from carts_v2 import controller
    from carts_v2.engine_health import PhysxStatsMonitor
    from carts_v2.evaluate_run import TruthAuditRecorder
    from carts_v2.run_grasp_lift import _HighObservationWristFtAuditor, _load_frozen_hand_inertials
    from te_foundationpose_handoff_plan import FullRobotCollisionScene
    from te_body_nut_regrasp import run_body_nut_regrasp
    from te_body_nut_rotation import run_body_nut_rotation
    from te_body_assembly_video import BodyAssemblyVideo
    from te_foundationpose_handoff_runtime import _install_rgbd_resume_sync

    started=perf_counter();output=args.output
    angle=float(args.probe_additional_turn_deg);speed=float(args.probe_speed_deg_s)
    if not (0 < angle <= 120. or (args.probe_regrasp_only and angle == 0.)) or not 0 < speed <= 10.:
        raise ValueError("this local diagnostic retains the full-entry 120-degree stroke and 10-degree/s development bounds")
    if args.probe_repeat_after_reindex and angle > 60.:
        raise ValueError("the two-stroke local diagnostic is bounded to 120 total commanded degrees")
    if args.probe_unload_after_lateral_hold_stop and not args.probe_repeat_after_reindex:
        raise ValueError("stopped-hold unloading is part of the physical two-stroke sequence")
    if args.probe_second_grasp_axis_shift_m is not None and (
            not args.probe_repeat_after_reindex or not np.isfinite(args.probe_second_grasp_axis_shift_m)
            or abs(args.probe_second_grasp_axis_shift_m) > .003):
        raise ValueError("a second-grip shift requires the two-stroke sequence and the existing 3 mm bound")
    if args.probe_second_regrasp_effort_time_constant_s is not None and (
            not args.probe_repeat_after_reindex or not 0 <= args.probe_second_regrasp_effort_time_constant_s <= .5):
        raise ValueError("the second-grip target-relaxation diagnostic is bounded to 0.5 s")
    if args.probe_start_open and not args.probe_grasp_axis_shift_m:
        raise ValueError("an open-hand diagnostic must execute a declared actual regrasp")
    if args.probe_regrasp_only and (not args.probe_start_open or args.probe_repeat_after_reindex):
        raise ValueError("regrasp-only diagnosis starts with an open hand and has no later stroke sequence")
    launch=json.loads(base_run.with_suffix(".launch.json").read_text())["argv"]
    base_config=repository/launch[launch.index("--config")+1]
    assembly_config=Path(assembly_config_path)
    config=yaml.safe_load(base_config.read_text());assembly=yaml.safe_load(assembly_config.read_text())
    grip_scale=float(args.probe_grip_effort_scale)
    reaction_stop=float(args.closing_reaction_stop_nm)
    higher_load_envelope=(args.cpu_wrist_reference is not None and args.closing_drive_cap_nm>=2.
                         and reaction_stop>=1.2 and args.probe_visual_alignment_feedback)
    if not .9 <= reaction_stop <= 2.0 or (reaction_stop>.9 and not higher_load_envelope):
        raise ValueError("higher projected-reaction reference requires the explicit bounded CPU load envelope")
    if not 1. <= grip_scale <= (1.8 if higher_load_envelope else 1.5):
        raise ValueError("grip reference exceeds the bounded CPU load envelope")
    if grip_scale != 1. and not args.probe_grasp_axis_shift_m:
        raise ValueError("a changed grip reference requires the actual opening/regrasp sequence")
    grip_weights=np.ones(3) if args.probe_grip_effort_weights is None else np.asarray(args.probe_grip_effort_weights)
    if (not np.all(np.isfinite(grip_weights)) or np.any(grip_weights<.7)
            or np.any(grip_weights>1.3) or not np.isclose(grip_weights.sum(),3.,atol=1e-10,rtol=0)):
        raise ValueError("relative grip-weight diagnostic requires three bounded weights with unchanged unit mean")
    if args.probe_grip_effort_weights is not None and not args.probe_grasp_axis_shift_m:
        raise ValueError("a changed grip distribution requires the actual opening/regrasp sequence")
    if args.probe_grasp_axis_shift_m:
        if abs(args.probe_grasp_axis_shift_m) > .003:
            raise ValueError("local grip-location diagnostic is bounded to 3 mm")
        assembly["nut_regrasp"]["canonical_axial_shift_m"]=float(args.probe_grasp_axis_shift_m)
        assembly["nut_regrasp"]["required_closing_joint_effort_nm"]=(
            np.asarray(assembly["nut_regrasp"]["required_closing_joint_effort_nm"])*grip_scale*grip_weights).tolist()
        assembly_config=output/"local_assembly_control.yaml"
        assembly_config.write_text(yaml.safe_dump(assembly,allow_unicode=True,sort_keys=False))
    inputs=load_v2_inputs(repository,config_path=base_config,object_id=source_metadata["object_id"])
    dynamic=copy.deepcopy(config["dynamic"])
    dynamic['physics_dt_s']=float(world.get_physics_dt())
    dynamic['closing_drive_maximum_effort_nm']=float(args.closing_drive_cap_nm)
    dynamic['measured_effort_abort_nm']=reaction_stop
    dynamic["arm_damping"]=float(source_metadata["effective_lift_arm_damping_nm_s_rad"])
    robot,active,arm_indices,lower,upper,drive_audit=robot_data
    scene=prepared_scene["scene"];stage=omni.usd.get_context().get_stage()
    parts=[]
    for i,path in enumerate(scene["part_prim_paths"]):
        part=SingleRigidPrim(path,name=f"local_probe_part_{i}",reset_xform_properties=False)
        part.initialize();parts.append(part)
    hand_path=next(p for p in contact_paths if p.endswith("/handbase_link"))
    recorder=TruthAuditRecorder(object_parts=parts,object_articulation=plug_tree,
        hand_base_prim=stage.GetPrimAtPath(hand_path),robot_model=inputs.robot_model,
        stage_modules=(Gf,Usd,UsdGeom),contact_interface=get_physx_simulation_interface(),
        path_decoder=PhysicsSchemaTools.intToSdfPath,roots={"robot":"/World/HandArm",**scene["roots"]},
        expected_total_mass_kg=inputs.object_contract.model.mass_kg,
        part_bottom_offsets_m=scene["part_bottom_offsets_m"],table_top_z_m=scene["table_top_z_m"],
        physics_dt_s=dynamic["physics_dt_s"],engine_monitor=PhysxStatsMonitor(world.get_physics_context()),
        physics_step_interface=get_physx_interface(),tensor_contact_prim=contact_view,
        tensor_contact_sensor_paths=contact_paths,tensor_contact_max_count=max(
            4096,int(prepared_scene["contact_recording"].get("minimum_contact_records",4096))))
    ft_document=json.loads((repository/"src/kcg_connector/config/te_visual_high_reobserve_v1.json").read_text())
    w=source_metadata["wrist_ft"]
    rotation_force_reference=(float(w["force_limit_n"]) if args.probe_planned_contact_force_stop_n is None
                              else float(args.probe_planned_contact_force_stop_n))
    rotation_force_overrides=({} if args.probe_planned_contact_force_stop_n is None else
        {f"key_probe_nut_rotation_{phase}":rotation_force_reference
         for phase in ('visual_align','axial_settle','visual_refine','turn','hold')})
    if args.probe_guided_fit_continuation and rotation_force_overrides:
        rotation_force_overrides['key_probe_nut_grip_hold']=rotation_force_reference
    ft=_HighObservationWristFtAuditor(ft_articulation=ft_tree,
        reaction_row=ft_tree._articulation_view._metadata.joint_indices["hand2arm"]+1,
        robot_model=inputs.robot_model,
        hand_inertials=_load_frozen_hand_inertials(repository/ft_document["source_evidence"]["hand_inertial_source"]["path"]),
        gravity_m_s2=scene["gravity_m_s2"],physics_dt_s=dynamic["physics_dt_s"],task_rotation_world=np.eye(3),
        force_limit_n=w["force_limit_n"],torque_limit_nm=w["torque_limit_nm"],
        contact_force_limit_overrides_n=rotation_force_overrides,
        planned_contact_torque_limit_nm=w["planned_contact_torque_limit_nm"],
        planned_contact_torque_action=w["planned_contact_torque_action"],
        planned_contact_force_time_constant_s=w["planned_contact_force_monitoring"]["time_constant_s"],
        dynamic_inertia_compensation_enabled=True,
        frozen_hand_positions=source_metadata["motion_plan"]["pregrasp_hand_positions_rad"],
        dynamic_inertia_enabled_phases=("approach_above","wait_above_settled","approach_descent","settle","pregrasp_hold","tare"))
    ft._PLANNED_CONTACT_PHASE_PREFIXES=(*ft._PLANNED_CONTACT_PHASE_PREFIXES,"key_probe_")
    # Reuse the identical model's completed free-space tare, never a loaded zero.
    ft.tare_canonical_sensor=np.asarray(w["tare_canonical_sensor_wrench"])
    ft.tare_gravity_sensor=np.asarray(w["tare_modeled_gravity_sensor_wrench"])
    ft.tare_model_residual_sensor=np.asarray(w["tare_minus_model_sensor_wrench"])
    if args.cpu_wrist_reference is not None:
        reference_path=args.cpu_wrist_reference.resolve()
        reference_result=json.loads((reference_path/'reference_result.json').read_text())
        if reference_result['maximum_recorded_hand_normal_load_n']!=0.:
            raise ValueError('The CPU wrist zero must come from an unloaded hand')
        reference_rows=[]
        with (reference_path/'reference_samples.jsonl').open() as stream:
            for line in stream:
                row=json.loads(line)
                if row['phase']=='unloaded_initial':reference_rows.append(row)
        reference_rows=reference_rows[-round(.25/dynamic['physics_dt_s']):]
        canonical=[];gravity=[]
        for row in reference_rows:
            links,sensor=ft._kinematics(np.asarray(row['active_q_rad']))
            canonical.append(-np.asarray(row['raw_sensor_wrench']))
            gravity.append(ft._gravity_wrench_sensor(links,sensor))
        ft.tare_canonical_sensor=np.mean(canonical,axis=0)
        ft.tare_gravity_sensor=np.mean(gravity,axis=0)
        ft.tare_model_residual_sensor=ft.tare_canonical_sensor-ft.tare_gravity_sensor
        (output/'cpu_wrist_tare.json').write_text(json.dumps({
            'source':str(reference_path),'loaded_rezeroing':False,
            'tare_canonical_sensor':ft.tare_canonical_sensor.tolist(),
            'tare_gravity_sensor':ft.tare_gravity_sensor.tolist(),
            'tare_model_residual_sensor':ft.tare_model_residual_sensor.tolist()},indent=2)+'\n')
    # Like the main entry's initial settling, populate the causal sensor state
    # before enabling a new free-space test. Grasp/contact motions are not run
    # in this open-hand warmup; all raw FT and existing joint limits remain.
    ft.gate_enabled=not args.probe_start_open
    truth_stream=(output/"truth_samples.jsonl").open("x",buffering=1)
    original_capture=recorder.capture
    def capture(**kwargs):
        original_capture(**kwargs)
        truth_stream.write(json.dumps(recorder.samples[-1],separators=(",",":"))+"\n")
        ft.capture(**kwargs)
    recorder.capture=capture
    stepper=controller.JointSignalStepper(robot=robot,world=world,auditor=recorder,
        active_indices=active,arm_indices=arm_indices,arm_lower_limits=lower,arm_upper_limits=upper,
        settings=dynamic,render=False,robot_model=inputs.robot_model,
        payload_model=source_metadata["controller_outcome"]["payload_compensation_model"])
    ft.stepper=stepper
    runtime={"world":world,"inputs":inputs,"scene":scene,"auditor":recorder,
             "robot_data":robot_data,"object_parts":parts,"nail_body_ft_auditor":ft,
             "body_assembly_control_config":str(assembly_config),"body_assembly_scene":prepared_scene,
             "robot_asset":source_metadata["robot_asset"]}
    _install_rgbd_resume_sync(world, stage)
    light=UsdLux.DomeLight.Define(stage,"/World/LocalProbeLighting")
    light.CreateIntensityAttr(float(scene["render"].dome_light_intensity))
    video=BodyAssemblyVideo(repository,runtime,output/"video",dynamic["physics_dt_s"],fps=5)
    transport=json.loads((base_run/"socket_transport/transport_and_observation.json").read_text())
    socket=np.asarray(transport["world_from_socket_wrist_visual"]).reshape(4,4)
    video.freeze_main_target(socket[:3,3])
    grip=json.loads((args.run/"socket_transport"/args.source_grip_stage/"nut_regrasp_controller_result.json").read_text())
    arm=np.asarray(source_sensor["active_targets_rad"][:7]);hand=np.asarray(source_sensor["active_targets_rad"][7:])
    tare=np.asarray(grip["new_grasp_effort_tare_nm"]);hand_lo,hand_hi=np.asarray(grip["finite_preload_bounds_rad"])
    desired=np.asarray(grip["effort_reference_nm"]);dt=dynamic["physics_dt_s"]
    result={"scope":"LOCAL_CONTINUED_NUT_TURN_NOT_FULL_ASSEMBLY","source_run":str(args.run),
            "source_step":args.source_step,"passive_joint_initialization":passive_initialization,
            "physics_solver":args.solver_type,
            "velocity_iterations":args.velocity_iterations,
            "solver_configuration_scope":("CONTACT_CONVERGENCE_DIAGNOSTIC_NOT_YET_VALIDATED_FOR_DELIVERY"
                if args.contact_convergence_check else "RECORDED_BASELINE_CONFIGURATION"),
            "robot_source_state_authored_before_reset":args.robot_state_before_reset,
            "external_forces_every_iteration":bool(args.external_forces_every_iteration),
            "hand_friction_effort_from_urdf":args.hand_friction_effort_from_urdf,
            "additional_rotation_deg":angle,"maximum_rotation_speed_deg_s":speed,
            "nut_grasp_axial_shift_m":args.probe_grasp_axis_shift_m,
            "grip_effort_reference_multiplier":grip_scale,
            "grip_effort_relative_weights":grip_weights.tolist(),
            "hardware_authorized":False,"source_tare_reused_without_loaded_rezeroing":True}
    if rotation_force_overrides:
        result['planned_contact_force_window']={
            'original_force_reference_n':float(w['force_limit_n']),
            'current_nut_rotation_contact_reference_n':rotation_force_reference,
            'phase_overrides':rotation_force_overrides,
            'free_space_and_regrasp_references_changed':False,
            'source_existing_contact_window':'src/kcg_connector/config/te_body_gpu_wrist_force_consistent_v1.yaml:wrist_contact_force_probe',
            'basis':'Bilateral centring kept axial load near .2 N and normal contacts finite, but about3 N transverse correction reached the old research stop. Observe the same bounded motion under the existing5 N contact window; this is not a manufacturer rating.',
            'drive_targets_force_reference_gains_and_speed_limits_changed':False,
            'actual_per_finger_normal_acceptance_n':15.}
    if args.cpu_wrist_reference is not None:
        result['wrist_tare_source']=str(args.cpu_wrist_reference.resolve())
        result['source_tare_reused_without_loaded_rezeroing']=False
        result['separate_cpu_free_space_tare_used_without_loaded_rezeroing']=True
    if args.probe_repeat_after_reindex:
        result["scope"]="LOCAL_TWO_STROKES_WITH_PHYSICAL_RELEASE_REINDEX_REGRASP_NOT_FULL_ASSEMBLY"
    if args.probe_start_open:
        result["initial_grip_state"]="OPEN_HAND_AT_DECLARED_SUPPORTED_OBJECT_INITIAL_CONDITION"
        result["contact_warm_start_restored"]=False
    elif args.probe_guided_fit_continuation:
        result['initial_grip_state']='DECLARED_COLD_SOURCE_GRIP_CURRENT_CONTACT_AND_MOTION_REQUIRE_EVALUATION'
        result['contact_warm_start_restored']=False
        result['new_grasp_motion_executed']=False
    if args.probe_regrasp_only:
        result["scope"]="LOCAL_SUPPORTED_OBJECT_REGRASP_ONLY_NOT_FULL_ASSEMBLY"
    try:
        world.play()
        for _ in range(round(2./dt)):
            if stepper.latest is not None and not args.probe_start_open and not args.probe_guided_fit_continuation:
                effort=stepper.latest[2][8:]-tare[1:]
                change=(desired-effort)/dynamic["hand_stiffness"]
                hand[1:]=np.clip(hand[1:]+np.clip(change,-dynamic["finger_maximum_speed_rad_s"]*dt,
                    dynamic["finger_maximum_speed_rad_s"]*dt),hand_lo[1:],hand_hi[1:])
            stepper.advance("nut_index_free_open_hold" if args.probe_start_open else "key_probe_nut_grip_hold",arm,hand)
            if stepper.abort_reason:raise RuntimeError(stepper.abort_reason)
        mean_effort=np.mean([np.asarray(s["active_efforts_nm"])[8:]-tare[1:] for s in ft.samples[-120:]],axis=0)
        result["warm_hold_mean_finger_effort_nm"]=mean_effort.tolist()
        if not args.probe_start_open and not args.probe_guided_fit_continuation and np.any(np.abs(mean_effort-desired)>.03):
            raise RuntimeError("local grip effort did not recover the source grip reference")
        if args.probe_start_open:
            settled=ft.samples[-round(.5/dt):]
            force_peak=max(float(s["resultant_force_used_for_protection_n"]) for s in settled)
            torque_peak=max(float(s["resultant_torque_nm"]) for s in settled)
            ready=all((s.get("dynamic_inertia_prediction") or {}).get("ready",False) for s in settled)
            result["initial_open_hand_settling"]={
                "duration_s":2.,"ft_gate_enabled_during_initialization":False,
                "nominal_joint_targets_fixed":True,"grasp_pressure_commanded":False,
                "joint_drive_and_effort_bounds_retained":True,"raw_samples_retained":True,
                "final_half_second_force_peak_n":force_peak,
                "final_half_second_torque_peak_nm":torque_peak,
                "causal_inertia_history_ready":ready}
            if not ready or force_peak>ft.force_limit_n or torque_peak>ft.torque_limit_nm:
                raise RuntimeError("open-hand initialization did not settle inside the original free-space FT limits")
            ft.gate_enabled=True
            result["initial_open_hand_settling"]["ft_gate_reenabled_before_observation_and_grasp"]=True
        world.pause()
        H=np.asarray(inputs.robot_model.forward_kinematics(tuple(stepper.latest[0]),enforce_limits=False)["handbase_link"])
        if args.probe_start_open or args.probe_reobserve_axis:
            import omni.replicator.core as rep
            from te_body_socket_observation import observe_released_plug_from_rgbd
            observation_label=("open_hand_current_observation" if args.probe_start_open else
                "current_observation_before_local_regrasp" if args.probe_grasp_axis_shift_m else
                "current_loaded_grip_observation")
            observation=observe_released_plug_from_rgbd(
                repository,stage,world,rep,H,output/observation_label)
            if not observation.get("position_and_axis_measured"):
                raise RuntimeError("the supported-object diagnostic requires a current ordinary RGB-D observation")
            result[observation_label]=observation
        else:
            old=source_rotation["postgrip_palm_observation"]
            relative=np.linalg.inv(np.asarray(old["world_from_hand_encoder"]).reshape(4,4))@np.asarray(old["world_from_plug_five_dof"]).reshape(4,4)
            B=H@relative
            # Use the current directed axis and a free yaw gauge, not propagated Body key yaw.
            z=B[:3,2]/np.linalg.norm(B[:3,2]);x=np.array([1.,0.,0.])-z*z[0];x/=np.linalg.norm(x)
            B[:3,:3]=np.column_stack((x,np.cross(z,x),z))
            observation={"position_and_axis_measured":True,"world_from_plug_five_dof":B.tolist(),
                "source":"SEALED_RGBD_AND_CURRENT_ENCODERS_SAME_GRIP_LOCAL_DIAGNOSTIC",
                "online_object_or_contact_truth_used":False,"body_key_yaw_measured":False,
                "original_rgbd_observation":str(args.run/"socket_transport"/args.source_rotation_stage/"postgrip_palm/camera_and_estimate.json")}
        table=np.asarray(inputs.table_xy_bounds_m);size=np.r_[table[:,1]-table[:,0],1.]
        center=np.r_[table.mean(axis=1),inputs.table_top_z_m-.5];fixture=prepared_scene["fixture"]
        obstacles={"table":fcl.CollisionObject(fcl.Box(*size),fcl.Transform(center)),
            "fixture":fcl.CollisionObject(fcl.Box(*fixture["size_m"]),fcl.Transform(np.asarray(fixture["center_world_m"])))}
        if args.probe_grasp_axis_shift_m:
            replay_effort_tau=(grip.get("grip_effort_target_update",{}).get("time_constant_s")
                               if args.probe_regrasp_only else None)
            if args.probe_regrasp_only:
                result["replayed_source_grip_effort_time_constant_s"]=replay_effort_tau
            result["new_grip"]=run_body_nut_regrasp(repository,runtime,stepper,dynamic,observation,socket,
                FullRobotCollisionScene(inputs),obstacles,output/"socket_transport/nut_regrasp",
                start_from_nut_grip=not args.probe_start_open,
                effort_regulation_time_constant_s_override=replay_effort_tau)
            if not result["new_grip"].get("completed"):
                raise RuntimeError(result["new_grip"].get("failure_reason","local relocated grip failed"))
            grip=result["new_grip"]
            observation=None  # The new grip requires its own actual post-grip RGB-D.
            if args.probe_regrasp_only:
                result["regrasp_only_completed"]=True
                return result
        else:
            run_body_nut_regrasp(repository,runtime,stepper,dynamic,observation,socket,
                FullRobotCollisionScene(inputs),obstacles,output/"geometry_setup",prepare_geometry_only=True)
            grip.update(completed=True,fixed_arm_target_rad=arm.tolist(),final_hand_target_rad=hand.tolist())
            if args.probe_reobserve_axis:
                observation=None
        settings=copy.deepcopy(assembly["nut_rotation_after_index"])
        # Local assembly YAML stores regrasp edits, but earlier CLI control
        # overrides live in the completed rotation record. Preserve its actual
        # planar law instead of silently reverting to the un-restored integrator.
        source_planar=source_rotation.get('settings',{}).get('planar_force_admittance')
        if source_planar is not None:
            settings['planar_force_admittance']=copy.deepcopy(source_planar)
        settings.update(rotation_about_socket_plus_z_deg=-angle,maximum_rotation_speed_deg_s=speed,
            maximum_arm_speed_rad_s=.075*speed/2.,post_rotation_hold_s=.5)
        if args.probe_visual_alignment_feedback:
            settings['pre_turn_visual_feedback']={
                'maximum_corrections':3,'axis_tolerance_deg':.1,
                'maximum_cumulative_axis_command_deg':1.,
                'rationale':'Postrun CPU observation showed 0.349 deg actual residual after encoder-predicted alignment. Reobserve the real RGBD axis before turning and correct within the original 1 deg total command and speed bounds. The 0.1 deg target leaves tilt-induced displacement about 58 um over 33 mm, below the delivered source pin radial clearance of about 82 um; it is a simulation alignment target, not a hardware guarantee.'}
            if args.probe_visual_position_feedback:
                settings['pre_turn_visual_feedback'].update(
                    position_feedback_enabled=True,position_tolerance_m=.000025,
                    axis_tolerance_deg=.05,maximum_cumulative_lateral_correction_m=.001,
                    axis_budget_mode='net_reference_rotation',maximum_reference_rotation_deg=1.,
                    maximum_corrections=8,pose_correction_gain=.5,
                    axial_refinement_mode='force_admittance',
                    rationale='Same recorded motion was unstable with about 215 um eccentricity and stable in the explicitly centred-fixture counterfactual. Correct actual RGBD centre and axis with the robot; re-estimate its current grip relation while preserving the physical hand target. Use the current grasp geometry allowance (at most 1 mm), original per-correction speeds and all force/drive/travel stops. No fixture or object pose is changed online.')
                settings['pre_turn_visual_feedback'].pop('maximum_cumulative_axis_command_deg')
                settings['pre_turn_visual_feedback']['pose_target_geometry_basis']={
                    'body_source_geometry_z_max_m':0.,'maximum_inserted_axial_span_m':.014605,
                    'nominal_minimum_pin_radial_clearance_m':.000081843,
                    'controlled_position_plus_tilt_deviation_m':.000025+.014605*np.sin(np.deg2rad(.05)),
                    'source':'artifacts/kcg_connector/model_delivery_20260908/package/README_CN.md',
                    'scope':'Nominal simulation pose target below half the model radial clearance; remaining clearance is a reserve, not a certified perception error bound. Actual rigid contacts and final assembly still require postrun evaluation.'}
                settings['post_rotation_hold_s']=2.
        if args.probe_arm_speed_limit_rad_s is not None:
            arm_limit=float(args.probe_arm_speed_limit_rad_s)
            if not 0 < arm_limit <= .375:
                raise ValueError("local arm speed must stay inside the existing development bound")
            settings["maximum_arm_speed_rad_s"]=arm_limit
        if not args.probe_grasp_axis_shift_m and not args.probe_reobserve_axis:
            # Continuation from an already aligned grip keeps its turn load.
            # A new measurement after static reinitialization can reveal actual
            # settling drift. It needs the original measured alignment/preparation
            # just as a real new grip does; source alignment is not a current fact.
            settings.update(axial_settle_duration_s=.25,
                axial_force_reference_n=settings["turn_axial_force_reference_n"])
            settings["pre_turn_visual_alignment"]["enabled"]=False
            settings["pre_turn_torsional_compliance"]["enabled"]=False
        if args.probe_accel_deg_s2 is not None:
            settings["maximum_rotation_acceleration_deg_s2"]=float(args.probe_accel_deg_s2)
        if args.probe_torsional_stop_nm is not None:
            bound=float(args.probe_torsional_stop_nm)
            reference_ceiling=(.25 if higher_load_envelope else
                .1*grip_scale if args.cpu_wrist_reference is not None and args.closing_drive_cap_nm>1. else .1)
            if not 0 < bound <= reference_ceiling+1e-12:
                raise ValueError("local torsional reference exceeds the bounded grip-scaled diagnostic range")
            if bound>.1:
                result['torsional_reference_scope']={
                    'previous_trial_reference_nm':.1,'current_reference_nm':bound,
                    'planned_grip_scale':grip_scale,'closing_motor_cap_nm':args.closing_drive_cap_nm,
                    'reason':('Bounded higher-load CPU trial after measured visual alignment still reached the 0.14 Nm pilot stop. Prior same-hand lab transferred 0.269 Nm; the present torque ceiling is a diagnostic reference, not a claimed free-assembly capacity.' if higher_load_envelope else
                        'Limited extension proportional to planned normal-load increase after observed material slip; not a hardware rating or guaranteed grasp capacity'),
                    'projected_reaction_stop_nm':reaction_stop,
                    'previous_projected_reaction_reference_nm':.9,
                    'source_geometry_reaction_envelope_nm':2. if higher_load_envelope else None,
                    'projected_reaction_budget':'artifacts/kcg_connector/isaac/te_full_assembly_20260908/source70_load_sizing_geometry_01/physical_joint_reaction_sizing.json' if higher_load_envelope else None,
                    'source_geometry_budget':'artifacts/kcg_connector/isaac/te_full_assembly_20260908/source70_load_sizing_geometry_01/normal_feedback_full_wrench_budget.json' if higher_load_envelope else None,
                    'actual_per_finger_normal_acceptance_n':15.,
                    'independent_joint_speed_stops_changed':False,
                    'wrist_force_reference_changed_by_explicit_contact_window':bool(rotation_force_overrides)}
            result["previous_torsional_stop_nm"]=settings["stops"]["torsional_moment_nm"]
            settings["stops"]["torsional_moment_nm"]=bound
        if args.probe_lateral_stop_n is not None:
            bound=float(args.probe_lateral_stop_n)
            if not 0 < bound <= rotation_force_reference:
                raise ValueError("the lateral reference cannot exceed the declared independent contact wrist reference")
            result["previous_lateral_stop_n"]=settings["stops"]["lateral_force_n"]
            settings["stops"]["lateral_force_n"]=bound
            if bound > 1.5:
                result["additional_lateral_reference_role"]={
                    "retained_reference_n":1.5,
                    "new_additional_stop_n":bound,
                    "independent_contact_wrist_resultant_stop_n":rotation_force_reference,
                    "free_space_wrist_force_reference_n":ft.force_limit_n,
                    "reason":"Retired pilot references are preserved as crossings. The explicit planned-contact window, when selected, bounds both the wrist resultant and its lateral component; all other limits remain independently recorded.",
                    "force_target_or_drive_cap_increased":False,
                    "source_key_intersections_and_physical_slip_still_require_postrun_evaluation":True}
        if args.probe_hold_finger_targets:
            settings["regulate_finger_effort_during_rotation"]=False
        if args.probe_hold_grip_through_preparation:
            settings["regulate_finger_effort_during_preparation"]=False
            settings["regulate_finger_effort_during_rotation"]=False
            settings["grip_hold_mode_rationale"]=(
                "Retain the actually reached finite native-PD finger targets during arm pose correction and turning. "
                "Projected proximal reaction includes manipulation moments and is not a pure normal-force signal. "
                "No object fixation, drive-gain increase, torque-cap increase or loaded sensor rezero is used.")
        if args.probe_grip_hold_stiffness_scale is not None:
            settings["grip_hold_impedance"]={"stiffness_scale":float(args.probe_grip_hold_stiffness_scale),
                "scope":"Increase finite holding stiffness with torque-continuous retargeting, unchanged caps and no object attachment"}
            settings["grip_hold_mode_rationale"]=(
                "Retain finite native-PD finger targets after a declared holding-gain transition that preserves the "
                "instantaneous raw PD effort at the measured state. Motor caps, object freedom and sensor tare stay unchanged; "
                "later measured contact forces must still be evaluated.")
        if args.probe_planar_stiffness_n_m is not None:
            stiffness=float(args.probe_planar_stiffness_n_m)
            if not 0 <= stiffness <= 5000.:
                raise ValueError("this local virtual-restoring-stiffness probe is bounded to 5000 N/m")
            settings["planar_force_admittance"]["virtual_restoring_stiffness_n_m"]=stiffness
        if args.probe_hold_planar_position:
            settings["planar_force_admittance"]["enabled"]=False
            settings["local_planar_position_hold_rationale"]=(
                "Retain the current visual center/axis reference and original finite native drives. "
                "The preceding guided Body remained centered while force admittance moved the hand; "
                "all six-axis wrench stops and axial force admittance remain active.")
        if args.probe_freeze_planar_after_preparation:
            if args.probe_hold_planar_position:
                raise ValueError("preparation admittance cannot both be disabled and retained")
            settings["planar_force_admittance"]["freeze_after_preparation"]=True
        if args.probe_force_consistent_planar_range:
            allowance=float(grip['visual_alignment_allowance_from_quarter_body_clearance_m'])
            settings['planar_force_admittance']['force_consistent_range']={
                'unchanged_wrist_force_limit_n':float(ft.force_limit_n),
                'visual_geometry_allowance_m':allowance,
                'scope':'Reference-offset bound derived from existing force limit/restoring stiffness and remaining visual correction allowance; not a larger force or speed limit'}
        if args.probe_guided_socket_pivot:
            if not args.probe_freeze_planar_after_preparation:
                raise ValueError("the guided-axis diagnostic requires the reached preparation position to be held")
            settings["guided_socket_rotation_pivot"]=True
        if args.probe_grip_lateral_balance:
            if not args.probe_hold_finger_targets or not args.probe_freeze_planar_after_preparation:
                raise ValueError("local grip load distribution requires the prepared hand and lateral position references")
            settings["grip_lateral_balance"]={
                "enabled":True,"geometry_plan":assembly["nut_regrasp"]["geometry_plan"],
                "response_rate_s_inv":1.,"maximum_model_normal_redistribution_n":1.}
        if args.probe_geometric_fit_feedback:
            settings['guided_feature_fit']={
                'source_pin_tip_z_body_m':-.0094615,'pin_pattern_radius_upper_m':.02,
                'pin_lateral_displacement_budget_m':float(args.probe_pin_fit_budget_m),
                'key_lateral_displacement_budget_m':.0001,
                'source_geometry':'artifacts/kcg_connector/isaac/te_full_assembly_20260905/representative_socket_cavities_01/geometry_manifest.json',
                'scope':'Coupled current-RGBD position/axis admission with nominal guided-key yaw; measured feature displacement is bounded rather than demanding each pose coordinate approach zero. Not a global perception or hardware guarantee.'}
            settings['pre_turn_visual_feedback']['pose_correction_gain']=1.
        if args.probe_guided_fit_continuation:
            settings.pop('pre_turn_visual_feedback',None)
            settings['pre_turn_visual_alignment']['enabled']=False
            settings['pre_turn_torsional_compliance']['enabled']=False
            settings['planar_force_admittance']['enabled']=False
            settings['axial_settle_duration_s']=0.
            settings['post_rotation_hold_s']=2.
            settings['guided_fit_continuation']={
                'source_pin_tip_z_body_m':-.0094615,'pin_pattern_radius_upper_m':.02,
                'pin_lateral_displacement_budget_m':.00004,
                'key_lateral_displacement_budget_m':.0001,
                'source_geometry':'artifacts/kcg_connector/isaac/te_full_assembly_20260905/representative_socket_cavities_01/geometry_manifest.json',
                'scope':'Current RGBD must establish nominal guided feature fit before the bounded turn. Budgets are not global perception/geometry guarantees; no new physical alignment or source contact history is claimed.'}
        if args.probe_in_turn_feedback:
            settings['in_turn_feedback']={
                'enabled':True,'corrections_enabled':not args.probe_segmented_observation_only,
                'maximum_segment_angle_deg':.5,'maximum_unobserved_time_s':.5,
                'maximum_segment_acceleration_deg_s2':13.2,
                'acceleration_reference_scope':'Finite interval acceleration reference derived from a 0.5 degree minimum-jerk segment at 2 deg/s; not a manufacturer rating. Actual timing/acceleration must be compared across methods.',
                'hand_angle_tolerance_deg':.02,'minimum_segment_angle_deg':.01,
                'maximum_consecutive_corrections':8}
        result["controller"]=run_body_nut_rotation(repository,runtime,stepper,dynamic,grip,socket,settings,
            output/"socket_transport/nut_rotation",initial_position_axis_observation=observation)
        if args.probe_lateral_stop_n is not None:
            old_bound=result["previous_lateral_stop_n"]
            count_crossings=0;first_crossing=None;peak=0.
            additional_count=0;additional_first=None
            with (output/"socket_transport/nut_rotation/nut_rotation_control_samples.jsonl").open() as stream:
                for line in stream:
                    sample=json.loads(line)
                    value=float(np.linalg.norm(sample["contact_wrench_at_virtual_plug_origin"][:2]))
                    peak=max(peak,value)
                    if value>old_bound:
                        count_crossings+=1
                        if first_crossing is None:first_crossing={"step":sample["step"],"elapsed_s":sample["elapsed_s"],"lateral_force_n":value}
                    if value>1.5:
                        additional_count+=1
                        if additional_first is None:additional_first={"step":sample["step"],"elapsed_s":sample["elapsed_s"],"lateral_force_n":value}
            result["previous_lateral_reference_postrun_audit"]={
                "stage_name":"nut_rotation",
                "previous_reference_n":old_bound,"exceeding_sample_count":count_crossings,
                "first_exceedance":first_crossing,"peak_recorded_filtered_lateral_force_n":peak,
                "original_failure_and_all_raw_samples_retained":True,
                "not_a_manufacturer_rating_or_physical_success_criterion":True}
            result["previous_1p5_n_lateral_reference_postrun_audit"]={
                "stage_name":"nut_rotation","reference_n":1.5,
                "exceeding_sample_count":additional_count,"first_exceedance":additional_first,
                "all_raw_measurements_and_original_failed_runs_retained":True}
        if args.probe_repeat_after_reindex:
            from te_body_nut_reindex import run_nut_release_and_reindex, can_unload_after_lateral_hold_stop
            recoverable_hold = bool(args.probe_unload_after_lateral_hold_stop
                and can_unload_after_lateral_hold_stop(result["controller"], ft.samples[-1]["phase"],
                                                       int(stepper.step_index), stepper.abort_reason))
            if not result["controller"].get("completed") and not recoverable_hold:
                raise RuntimeError("first local stroke did not finish; physical reindex was not attempted")
            index_settings=copy.deepcopy(assembly["nut_reindex"])
            index_settings.update(rotation_about_socket_plus_z_deg=angle,
                                  maximum_arm_speed_rad_s=.375,
                                  allow_unload_after_lateral_hold_stop=args.probe_unload_after_lateral_hold_stop)
            result["first_stroke_controller_stop_recovery_requested"] = recoverable_hold
            result["reindex"]=run_nut_release_and_reindex(
                repository,runtime,stepper,dynamic,grip,result["controller"],socket,
                index_settings,output/"socket_transport/nut_reindex")
            if not result["reindex"].get("completed"):
                raise RuntimeError(result["reindex"].get("failure_reason","local physical reindex failed"))
            result["second_grip"]=run_body_nut_regrasp(
                repository,runtime,stepper,dynamic,result["reindex"]["final_observation"],
                socket,FullRobotCollisionScene(inputs),obstacles,
                output/"socket_transport/nut_regrasp_after_index",
                grasp_axis_shift_override_m=args.probe_second_grasp_axis_shift_m,
                effort_regulation_time_constant_s_override=args.probe_second_regrasp_effort_time_constant_s)
            if not result["second_grip"].get("completed"):
                raise RuntimeError(result["second_grip"].get("failure_reason","second local grip failed"))
            result["controller_after_index"]=run_body_nut_rotation(
                repository,runtime,stepper,dynamic,result["second_grip"],socket,
                copy.deepcopy(settings),output/"socket_transport/nut_rotation_after_index")
    except Exception as error:
        result["failure_reason"]=str(error)
    finally:
        world.pause();truth_stream.close()
        result["video"]=video.close()
        result["stepper_wall_s"]=stepper.wall_times
        result["execution_wall_s"]=perf_counter()-started
        result["physical_step_count"]=stepper.step_index
        result["controller_completed"]=bool(result.get("controller",{}).get("completed"))
        if args.probe_repeat_after_reindex:
            result["controller_completed"] &= bool(result.get("controller_after_index",{}).get("completed"))
            result["release_reindex_regrasp_and_second_controller_completed"] = all(
                result.get(name,{}).get("completed",False)
                for name in ("reindex","second_grip","controller_after_index"))
        with gzip.open(output/"wrist_ft_samples.json.gz","wt") as f:json.dump(ft.samples,f,separators=(",",":"))
        meta={"scope":result["scope"],"physics_dt_s":dt,"robot_asset":source_metadata["robot_asset"],
              "physics_solver":args.solver_type,
              "robot_source_state_authored_before_reset":args.robot_state_before_reset,
              "external_forces_every_iteration":bool(args.external_forces_every_iteration),
              "hand_friction_effort_from_urdf":args.hand_friction_effort_from_urdf,
              "base_run":str(base_run),"body_assembly_control_config":str(assembly_config),
              "source_run":str(args.run),"source_step":args.source_step,"hardware_authorized":False,
              "online_object_or_contact_truth_used":False,"object_pose_writes_after_start":False,
              "tensor_contact_view_audit":{"valid_after_reset":True,"sensor_paths":contact_paths,
                  "contact_filter_paths":list(contact_view._contact_filter_paths)},
              "controller_outcome":{"native_drive_audit":drive_audit},"wrist_ft":{"source_free_space_tare":w}}
        (output/"trace_metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
        (output/"local_probe_result.json").write_text(json.dumps(result,indent=2)+"\n")
    return result
