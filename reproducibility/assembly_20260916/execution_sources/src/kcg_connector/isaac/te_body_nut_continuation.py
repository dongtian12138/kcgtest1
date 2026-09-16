"""Bounded reuse of the existing release, regrasp and nut-turn controllers.

No measured object/contact truth enters this sequence. A terminal torque stop
may be followed by a current-vision guide check and controlled release, while
the original stop remains recorded. Final seating is a separate physical audit.
"""
import copy


def run_nut_rotation_with_recovery(repository,runtime,stepper,dynamic,grip,socket,settings,output,
                                   *,initial_position_axis_observation=None):
    """Continue the same physical episode after a bounded, recoverable stop."""
    import gzip,json
    from pathlib import Path
    import yaml
    from carts_v2.fast_json import dumps,dump_array
    from te_body_nut_rotation import _run_body_nut_rotation_interval
    from te_body_nut_reindex import run_nut_release_and_reindex,can_recover_rotation_stop,can_unload_after_transmission_reserve_stop
    from te_body_nut_regrasp import run_body_nut_regrasp
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    assembly=yaml.safe_load((Path(repository)/runtime['body_assembly_control_config']).read_text())
    ft=runtime['nail_body_ft_auditor'];first_ft=len(ft.samples)
    requested=float(settings['rotation_about_socket_plus_z_deg']);remaining=requested;sent=0.
    limit=int(settings['recovery'].get('maximum_regrasps',1))
    if not 0<=limit<=2:raise ValueError('at most two current-state regrasp recoveries are allowed')
    record={'completed':False,'stage':'BOUNDED_ROTATION_WITH_RECOVERY','settings':settings,
        'first_step':int(stepper.step_index),'attempts':[],'control_samples_file':str(output/'nut_rotation_control_samples.jsonl'),
        'sample_count':0,'online_object_or_contact_truth_used':False,'physical_thread_progress_verified':False,
        'hardware_authorized':False,'simulation_only':True,'requested_rotation_is_not_actual_nut_rotation':True}
    current_grip=grip;last=None
    with (output/'nut_rotation_control_samples.jsonl').open('x') as stream:
        try:
            for attempt in range(limit+1):
                child=copy.deepcopy(settings);child['recovery']['enabled']=False
                child['rotation_about_socket_plus_z_deg']=remaining
                last=_run_body_nut_rotation_interval(repository,runtime,stepper,dynamic,current_grip,socket,child,
                    output/f'attempt_{attempt:02d}',initial_position_axis_observation=initial_position_axis_observation if attempt==0 else None)
                record['attempts'].append({'rotation':last})
                last_command=0.
                with Path(last['control_samples_file']).open() as source:
                    for line in source:
                        sample=json.loads(line);last_command=float(sample['commanded_rotation_deg'])
                        sample.update(attempt_index=attempt,attempt_elapsed_s=sample['elapsed_s'],
                            elapsed_s=(sample['step']-record['first_step'])*float(dynamic['physics_dt_s']),
                            attempt_commanded_rotation_deg=last_command,commanded_rotation_deg=sent+last_command)
                        stream.write(dumps(sample)+'\n');record['sample_count']+=1
                sent+=float(last.get('last_applied_rotation_command_deg',last_command));remaining=requested-sent
                record.pop('transmission_reserve_stop',None)
                for key in ('initial_arm_encoder_rad','final_arm_target_rad','final_hand_target_rad',
                            'held_initial_grip_target_rad','held_hand_target_after_preparation_rad',
                            'transmission_reserve_stop','finger_effort_regulation','postgrip_palm_observation',
                            'hand_from_virtual_nut_axis_frame','seating_candidate','normal_stop_reason',
                            'commanded_rotation_duration_s','planned_peak_rotation_speed_deg_s','planned_peak_rotation_acceleration_deg_s2',
                            'load_compensated_hold_arm_target_rad','last_applied_load_compensation_nm'):
                    if key in last:record[key]=last[key]
                if last.get('completed'):
                    record.update(completed=True,stage='ROTATION_COMMAND_FINISHED_REQUIRES_CONTACT_EVALUATION')
                    break
                phase=ft.samples[-1]['phase']
                recoverable=(can_recover_rotation_stop(last,phase,int(stepper.step_index),stepper.abort_reason)
                    or can_unload_after_transmission_reserve_stop(last,phase,int(stepper.step_index),stepper.abort_reason))
                if not recoverable or attempt==limit or abs(remaining)<.01:
                    record.update(stage='STOPPED',failure_reason=last.get('failure_reason'))
                    break
                release=copy.deepcopy(assembly['nut_reindex'])
                release.update(release_only=False,allow_recovery_reindex=True)
                reindex=run_nut_release_and_reindex(repository,runtime,stepper,dynamic,current_grip,last,socket,release,
                    output/f'recovery_reindex_{attempt:02d}')
                record['attempts'][-1]['reindex']=reindex
                if not reindex.get('completed'):raise RuntimeError(reindex.get('failure_reason'))
                collision_scene,obstacles=runtime['nut_regrasp_collision_context']
                current_grip=run_body_nut_regrasp(repository,runtime,stepper,dynamic,reindex['final_observation'],socket,
                    collision_scene,obstacles,output/f'recovery_grip_{attempt:02d}')
                record['attempts'][-1]['regrasp']=current_grip
                if not current_grip.get('completed'):raise RuntimeError(current_grip.get('failure_reason'))
        except Exception as error:
            record.update(completed=False,stage='STOPPED',failure_reason=str(error))
        finally:
            record.update(last_step=int(stepper.step_index),outer_abort_reason=stepper.abort_reason,
                continuation_grip=current_grip,executed_loaded_command_deg=sent,
                recovery_count=sum('regrasp' in a for a in record['attempts']),
                control_samples_exclude_separately_recorded_recovery_motion=True)
            with gzip.open(output/'joint_ft_samples.json.gz','wt',compresslevel=1) as f:dump_array(f,ft.samples[first_ft:])
            (output/'nut_rotation_controller_result.json').write_text(dumps(record)+'\n')
    return record


def continue_nut_strokes_and_release(repository, runtime, stepper, dynamic, record,
        socket, assembly, collision_scene, obstacles, output, save_record):
    from te_body_nut_regrasp import run_body_nut_regrasp
    from te_body_nut_rotation import run_body_nut_rotation
    from te_body_nut_reindex import (
        run_nut_release_and_reindex, can_unload_after_torsional_pilot_stop,
        can_unload_after_transmission_reserve_stop,can_recover_rotation_stop)

    settings = assembly["continued_nut_strokes"]
    maximum = settings["maximum_additional_strokes"]
    if isinstance(maximum, bool) or not isinstance(maximum, int) or not 0 <= maximum <= 6:
        raise ValueError("continued nut assembly is bounded to at most six additional strokes")
    turn_settings = copy.deepcopy(assembly["nut_rotation_after_index"])
    if settings.get('coaxial_interface_recipe'):
        turn_settings['coaxial_interface_recipe']=settings['coaxial_interface_recipe']
    angle = float(turn_settings["rotation_about_socket_plus_z_deg"])
    if not -120. <= angle < 0.:
        raise ValueError("continued strokes retain the bounded tightening direction")
    last_rotation = record.get("nut_rotation_after_index", record["nut_rotation"])
    last_grip = (record["nut_regrasp_after_index"] if "nut_rotation_after_index" in record
                 else record["nut_regrasp"])
    total_requested = sum(abs(float(record[name]["settings"]["rotation_about_socket_plus_z_deg"]))
        for name in ("nut_rotation", "nut_rotation_after_index") if name in record)
    series = record["continued_nut_strokes"] = {
        "maximum_additional_strokes": maximum, "attempts": [],
        "sum_requested_stroke_angles_deg": total_requested,
        "requested_angles_are_not_actual_rotation": True,
        "full_assembly_requires_postrun_physical_evaluation": True,
        "online_object_or_contact_truth_used": False,
    }
    for index in range(maximum):
        if last_rotation.get('seating_candidate'):
            series['termination']='CURRENT_VISUAL_SEATING_CANDIDATE'
            break
        if not record.get("completed") or not last_rotation.get("completed"):
            break
        if settings.get('complete_remaining_command_on_last_stroke',False):
            from te_nut_motion import remaining_command_stroke
            total_requested=float(runtime.get('coaxial_nut_commanded_degrees',total_requested))
            amount=remaining_command_stroke(abs(float(turn_settings['rotation_about_socket_plus_z_deg'])),
                total_requested,float(settings.get('maximum_total_command_deg',360.)),
                final_interval=index==maximum-1)
            if amount==0.:
                series['termination']='ACTUAL_COMMAND_BUDGET_REACHED'
                break
            angle=-amount
            turn_settings['rotation_about_socket_plus_z_deg']=angle
            series['requested_angles_account_for_recorded_early_regrasp']=True
        if total_requested + abs(angle) > float(settings.get('maximum_total_command_deg',380.)):
            series["termination"] = "REQUESTED_ROTATION_BUDGET_REACHED"
            break
        entry = {"index": index + 1, "requested_rotation_deg": angle}
        series["attempts"].append(entry)
        reindex_settings = copy.deepcopy(assembly["nut_reindex"])
        reindex_settings.update(release_only=False,
            rotation_about_socket_plus_z_deg=(
                float(assembly['nut_reindex']['rotation_about_socket_plus_z_deg'])
                if last_rotation.get('early_regrasp')
                else -float(last_rotation["settings"]["rotation_about_socket_plus_z_deg"])))
        # A compensated 94-degree request may stop early. The existing
        # quarter-turn open return preserves the tested flute grasp phase;
        # the unused requested remainder must not rotate that phase again.
        entry['early_regrasp_retains_configured_open_return']=bool(last_rotation.get('early_regrasp'))
        record["stage"] = "CONTINUED_NUT_RELEASE_AND_OPEN_HAND_REINDEX"
        save_record()
        entry["reindex"] = run_nut_release_and_reindex(
            repository, runtime, stepper, dynamic, last_grip, last_rotation,
            socket, reindex_settings, output / f"nut_reindex_continued_{index+1:02d}")
        record["completed"] = bool(entry["reindex"].get("completed"))
        if not record["completed"]:
            record["failure_reason"] = entry["reindex"].get("failure_reason")
            save_record()
            break
        record["stage"] = "CONTINUED_CURRENT_VISION_NUT_REGRASP"
        save_record()
        entry["grip"] = run_body_nut_regrasp(
            repository, runtime, stepper, dynamic, entry["reindex"]["final_observation"],
            socket, collision_scene, obstacles, output / f"nut_regrasp_continued_{index+1:02d}")
        record["completed"] = bool(entry["grip"].get("completed"))
        if not record["completed"]:
            record["failure_reason"] = entry["grip"].get("failure_reason")
            save_record()
            break
        last_grip = entry["grip"]
        record["stage"] = "CONTINUED_BOUNDED_NUT_ROTATION"
        total_requested += abs(angle)
        if settings.get('complete_remaining_command_on_last_stroke',False):
            series['sum_requested_stroke_angles_deg']+=abs(angle)
            series['planned_cumulative_command_budget_deg']=total_requested
        else:
            series["sum_requested_stroke_angles_deg"] = total_requested
        save_record()
        entry["rotation"] = run_body_nut_rotation(
            repository, runtime, stepper, dynamic, last_grip, socket,
            copy.deepcopy(turn_settings), output / f"nut_rotation_continued_{index+1:02d}")
        last_rotation = entry["rotation"]
        record["completed"] = bool(last_rotation.get("completed"))
        if not record["completed"]:
            record["failure_reason"] = last_rotation.get("failure_reason")
        save_record()

    latest_phase = runtime["nail_body_ft_auditor"].samples[-1]["phase"]
    pilot_stop = can_unload_after_torsional_pilot_stop(
        last_rotation, latest_phase, int(stepper.step_index), stepper.abort_reason)
    pilot_stop = pilot_stop or can_unload_after_transmission_reserve_stop(
        last_rotation, latest_phase, int(stepper.step_index), stepper.abort_reason)
    pilot_stop = pilot_stop or can_recover_rotation_stop(
        last_rotation,latest_phase,int(stepper.step_index),stepper.abort_reason)
    release_eligible = (stepper.abort_reason is None
        and last_rotation.get("last_step") == int(stepper.step_index)
        and ((record.get("completed") and last_rotation.get("completed")) or pilot_stop))
    series["terminal_release_eligible"] = bool(release_eligible)
    if settings.get("release_at_end", True) and release_eligible:
        release_settings = copy.deepcopy(assembly["nut_reindex"])
        release_settings.update(release_only=True, open_hold_duration_s=3.,
            rotation_about_socket_plus_z_deg=0., allow_release_after_torsional_pilot_stop=True,
            allow_release_after_transmission_reserve_stop=True,allow_recovery_reindex=True)
        record["stage"] = "TERMINAL_CURRENT_VISION_CHECK_AND_NUT_RELEASE"
        save_record()
        series["terminal_release"] = run_nut_release_and_reindex(
            repository, runtime, stepper, dynamic, last_grip, last_rotation,
            socket, release_settings, output / "nut_terminal_release")
        # The additional torque stop is retained, even if unloading succeeds.
        # It can coincide with seating, but only the physical audit decides.
        record["completed"] = bool(last_rotation.get("completed")
            and series["terminal_release"].get("completed"))
        if not series["terminal_release"].get("completed"):
            series["terminal_release_failure_reason"] = series["terminal_release"].get("failure_reason")
            if not record.get("failure_reason"):
                record["failure_reason"] = series["terminal_release_failure_reason"]
    record["completed_scope"] = "BOUNDED_TURN_SERIES_AND_RELEASE_REQUIRE_FULL_ASSEMBLY_PHYSICAL_EVALUATION"
    save_record()
