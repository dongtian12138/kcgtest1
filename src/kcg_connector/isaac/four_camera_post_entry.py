"""Existing bounded support/regrasp/turn sequence after the new visual entry."""
from pathlib import Path
import numpy as np

def continue_after_entry(repository,runtime,stepper,grasp_result,dynamic,record,
        wrist_socket,collision_scene,obstacles,output,save_record):
    import omni.usd
    import omni.replicator.core as rep
    import yaml
    from te_body_socket_observation import observe_current_plug_from_rgbd
    def observe_released_plug_from_rgbd(*args):
        return observe_current_plug_from_rgbd(*args,runtime)
    repository=Path(repository);output=Path(output)
    world=runtime['world'];inputs=runtime['inputs'];stage=omni.usd.get_context().get_stage()
    probe_record=record['key_entry'].get('probe_controller') or {}
    if (runtime.get("body_support_test_requested")
            and record["key_entry"].get("controller_depth_reached")):
        from te_body_support_release import run_body_support_release
        session = runtime.get('four_camera_perception_session')
        if session is not None:
            session.retire_body_grasp()
        record["stage"] = "TESTING_SOCKET_SUPPORT_BY_UNLOADING_AND_OPENING"
        save_record()
        record["body_support_release"] = run_body_support_release(
            repository, runtime, stepper, grasp_result, dynamic, probe_record,
            runtime["body_pregrasp_hand_positions_rad"], collision_scene,
            wrist_socket, obstacles, output / "body_support")
        record["completed"] = bool(record["body_support_release"].get("completed"))
        record["completed_scope"] = "CONTROLLER_ONLY_BODY_SUPPORT_REQUIRES_POSTRUN_EVALUATION"
        if record["completed"]:
            record["stage"] = "OBSERVING_RELEASED_PLUG_POSITION_AND_AXIS"
            save_record()
            released_hand = np.asarray(inputs.robot_model.forward_kinematics(
                tuple(stepper.latest[0]), enforce_limits=False)["handbase_link"])
            record["released_plug_observation"] = observe_released_plug_from_rgbd(
                repository, stage, world, rep, released_hand, output / "released_plug_observation")
            if runtime.get("body_nut_regrasp_requested"):
                from te_body_nut_regrasp import run_body_nut_regrasp
                record["stage"] = "PALM_GUIDED_FINITE_NUT_REGRASP"
                save_record()
                record["nut_regrasp"] = run_body_nut_regrasp(
                    repository, runtime, stepper, dynamic, record["released_plug_observation"],
                    wrist_socket, collision_scene, obstacles, output / "nut_regrasp")
                record["completed"] = bool(record["nut_regrasp"].get("completed"))
                record["completed_scope"] = "CONTROLLER_ONLY_NUT_REGRASP_REQUIRES_POSTRUN_EVALUATION"
                if not record["completed"]:
                    record["failure_reason"] = record["nut_regrasp"].get("failure_reason")
                else:
                    import yaml
                    assembly_config = yaml.safe_load((repository / runtime["body_assembly_control_config"]).read_text())
                    if assembly_config.get("nut_rotation", {}).get("enabled"):
                        from te_body_nut_rotation import run_body_nut_rotation
                        record["stage"] = "ROTATING_NUT_WITH_AXIAL_FORCE_ADMITTANCE"
                        save_record()
                        record["nut_rotation"] = run_body_nut_rotation(
                            repository, runtime, stepper, dynamic, record["nut_regrasp"],
                            wrist_socket, assembly_config["nut_rotation"], output / "nut_rotation")
                        record["completed"] = bool(record["nut_rotation"].get("completed"))
                        record["completed_scope"] = "CONTROLLER_ONLY_THREAD_PILOT_REQUIRES_POSTRUN_EVALUATION"
                        if not record["completed"]:
                            record["failure_reason"] = record["nut_rotation"].get("failure_reason")
                        elif (assembly_config.get("nut_reindex", {}).get("enabled")
                              and not record['nut_rotation'].get('seating_candidate')):
                            from te_body_nut_reindex import run_nut_release_and_reindex
                            record["stage"] = "RELEASING_NUT_AND_REINDEXING_OPEN_HAND"
                            save_record()
                            record["nut_reindex"] = run_nut_release_and_reindex(
                                repository, runtime, stepper, dynamic, record["nut_regrasp"],
                                record["nut_rotation"], wrist_socket, assembly_config["nut_reindex"],
                                output / "nut_reindex")
                            record["completed"] = bool(record["nut_reindex"].get("completed"))
                            record["completed_scope"] = "CONTROLLER_ONLY_NUT_REINDEX_REQUIRES_RETENTION_EVALUATION"
                            if not record["completed"]:
                                record["failure_reason"] = record["nut_reindex"].get("failure_reason")
                            elif assembly_config["nut_reindex"].get("regrasp_after_reindex", False):
                                record["stage"] = "REGRASPING_NUT_AFTER_OPEN_HAND_REINDEX"
                                save_record()
                                record["nut_regrasp_after_index"] = run_body_nut_regrasp(
                                    repository, runtime, stepper, dynamic,
                                    record["nut_reindex"]["final_observation"], wrist_socket,
                                    collision_scene, obstacles, output / "nut_regrasp_after_index")
                                record["completed"] = bool(record["nut_regrasp_after_index"].get("completed"))
                                record["completed_scope"] = "CONTROLLER_ONLY_SECOND_NUT_GRIP_REQUIRES_POSTRUN_EVALUATION"
                                if not record["completed"]:
                                    record["failure_reason"] = record["nut_regrasp_after_index"].get("failure_reason")
                                elif assembly_config.get("nut_rotation_after_index", {}).get("enabled"):
                                    record["stage"] = "ROTATING_NUT_AFTER_OPEN_HAND_REINDEX"
                                    save_record()
                                    record["nut_rotation_after_index"] = run_body_nut_rotation(
                                        repository, runtime, stepper, dynamic, record["nut_regrasp_after_index"],
                                        wrist_socket, assembly_config["nut_rotation_after_index"],
                                        output / "nut_rotation_after_index")
                                    record["completed"] = bool(record["nut_rotation_after_index"].get("completed"))
                                    record["completed_scope"] = "CONTROLLER_ONLY_SECOND_THREAD_STROKE_REQUIRES_POSTRUN_EVALUATION"
                                    if not record["completed"]:
                                        record["failure_reason"] = record["nut_rotation_after_index"].get("failure_reason")
        if ("nut_rotation" in record
                and assembly_config.get("continued_nut_strokes", {}).get("enabled", False)):
            from te_body_nut_continuation import continue_nut_strokes_and_release
            continue_nut_strokes_and_release(repository, runtime, stepper, dynamic,
                record, wrist_socket, assembly_config, collision_scene, obstacles,
                output, save_record)
        record["stage"] = ("NUT_SERIES_RETURNED_FOR_POSTRUN_PHYSICAL_EVALUATION"
                           if "continued_nut_strokes" in record else
                           "SECOND_NUT_ROTATION_RETURNED_FOR_POSTRUN_PHYSICAL_EVALUATION"
                           if "nut_rotation_after_index" in record else
                           "NUT_REINDEX_RETURNED_FOR_POSTRUN_PHYSICAL_EVALUATION"
                           if "nut_reindex" in record else
                           "NUT_ROTATION_RETURNED_FOR_POSTRUN_PHYSICAL_EVALUATION"
                           if "nut_rotation" in record else
                           "NUT_REGRASP_RETURNED_FOR_POSTRUN_PHYSICAL_EVALUATION"
                           if runtime.get("body_nut_regrasp_requested")
                           else "BODY_SUPPORT_TEST_RETURNED_FOR_POSTRUN_PHYSICAL_EVALUATION")

    series=record.get('continued_nut_strokes',{})
    release=series.get('terminal_release')
    if release is not None:
        from four_camera_online_completion import assess
        rotations=[record.get('nut_rotation'),record.get('nut_rotation_after_index')]
        rotations.extend(item.get('rotation') for item in series.get('attempts',[]))
        last=next(item for item in reversed(rotations) if item is not None)
        observations=[release[label]['five_dof_observation'] for label in ('before_nut_release','after_nut_release')
                      if label in release]
        record['controller_sequence_completed']=record['completed']
        record['online_completion']=assess(wrist_socket,observations,release,
            last['last_loaded_interface_wrench_n_nm'],last['settings']['visual_progress'],
            physics_dt_s=float(dynamic['physics_dt_s']),
            maximum_lateral_error_m=float(last['settings']['measured_tracking']['maximum_lateral_error_m']),
            maximum_axis_error_deg=.10)
        record['completed']=bool(record['online_completion']['online_seating_confirmed'])
        record['completed_scope']='ONLINE_SENSOR_SEATING_DECISION_REQUIRES_ORIGINAL_POSTRUN_PHYSICAL_ACCEPTANCE'
        if not record['completed']:
            record['failure_reason']='Online seating was not confirmed by depth, alignment, load and released hold'
        save_record()
