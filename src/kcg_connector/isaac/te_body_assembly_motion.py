"""One visually located, contact-free carry followed by a wrist socket view."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np


def run_to_socket_observation(
    repository, runtime, stepper, grasp_result, dynamic, body_observation, output,
):
    """Carry to a 50 mm gap, refresh body pose, then view the socket from the side.

    Required runtime additions are ``inputs`` and ``body_assembly_scene`` (the
    prepared scene result).  This function never accesses object_parts, truth
    samples, or a socket prim transform.  Its only robot commands use the
    existing finite finger controller, joint protections, and wrist FT auditor.
    A failed existing-planner or current-hand collision check stops this attempt.
    """
    import fcl
    import yaml
    import omni.replicator.core as rep
    import omni.usd
    from pxr import Gf, UsdGeom
    from scipy.spatial.transform import Rotation

    from kcg_connector.te_rgbd_pose_provider import estimate_receptacle_key_from_depth
    from te_body_socket_observation import (
        observe_socket_from_global_rgbd, observe_body_after_transport, wrist_camera_mount,
        observe_released_plug_from_rgbd,
    )
    from te_foundationpose_handoff_plan import FullRobotCollisionScene, _cylinder_from_mesh
    from te_foundationpose_handoff_runtime import (
        _author_camera, _capture_rgbd, _check_held_plug_path, _close_rgbd_resources,
        _execute_held_plug_path, _install_rgbd_resume_sync, _json_ready,
        _run_external_pose_plan, MOVEIT_SOFT_ARM_BOUNDS_RAD, control,
        _plan_key_probe_descent,
    )

    repository, output = Path(repository).resolve(), Path(output).resolve()
    speed_config = yaml.safe_load((repository / runtime["body_assembly_control_config"]).read_text())
    transport_speed = float(speed_config.get("motion", {}).get("maximum_transport_joint_speed_rad_s", .15))
    retime_free = bool(speed_config.get("development_feedback", {}).get("retime_free_space_paths", False))
    retime_straight=bool(speed_config.get('computation',{}).get('retime_straight_free_transport',False))
    if not np.isfinite(transport_speed) or transport_speed <= 0:
        raise ValueError("transport speed must be finite and positive")
    output.mkdir(parents=True, exist_ok=False)
    world, inputs = runtime["world"], runtime["inputs"]
    stage = omni.usd.get_context().get_stage()
    ft = runtime["nail_body_ft_auditor"]
    first_ft_sample = len(ft.samples)
    resources = {}
    record = {
        "completed": False, "stage": "BEFORE_GLOBAL_SOCKET_OBSERVATION",
        "simulation_only": True, "hardware_authorized": False,
        "online_object_or_contact_truth_used": False,
        "body_key_reobservations_after_memory": 0,
        "contact_motion_commanded": False,
        "wrist_socket_observation_executed": False,
        "physical_carry_and_memory_validity": "REQUIRES_POSTRUN_EVALUATION",
        "maximum_transport_joint_speed_rad_s": transport_speed,
        "target_face_gap_m": 0.050,
    }

    def save_record():
        (output / "transport_and_observation.json").write_text(
            json.dumps(_json_ready(record), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    try:
        if stepper.abort_reason is not None or grasp_result.get("failure_reason"):
            raise RuntimeError("current joint/force grasp controller did not finish safely")
        if not body_observation["key_measurement"]["key_direction_measured"]:
            raise RuntimeError("the current grasp has no visual body-key memory")
        memory = np.asarray(body_observation["hand_from_body_visual_memory"], dtype=np.float64).reshape(4, 4)
        initial_memory = memory.copy()
        refreshed_memory_start_step = None
        if not np.isfinite(memory).all():
            raise ValueError("current body-key memory is nonfinite")
        record["hand_from_body_visual_memory"] = memory.tolist()
        record["memory_observation_physics_time_s"] = body_observation["physics_time_s"]
        record["memory_observation_hand_encoder"] = body_observation["world_from_hand_encoder"]
        record["wrist_force_limit_n"] = ft.force_limit_n
        record["wrist_free_space_torque_limit_nm"] = ft.torque_limit_nm
        record["wrist_held_payload_torque_limit_nm"] = ft.planned_contact_torque_limit_nm
        record["finger_effort_references_nm"] = list(dynamic["required_closing_joint_effort_nm"])
        world.pause()
        global_observation = observe_socket_from_global_rgbd(
            repository, stage, world, rep, output / "global_socket",
        )
        refined_global = bool(global_observation["global_slot_refinement_succeeded"])
        socket = np.asarray(global_observation[
            "world_from_receptacle_row_major" if refined_global
            else "coarse_world_from_receptacle_row_major"
        ], dtype=np.float64).reshape(4, 4)
        record["global_socket_result"] = str(output / "global_socket/camera_and_estimate.json")
        record["global_socket_pose_source"] = "LIP_AND_SLOT_FIT" if refined_global else "SAM6D_COARSE_ONLY"
        record["world_from_socket_visual"] = socket.tolist()
        target_body = socket.copy()
        target_body[:3, :3] = socket[:3, :3] @ Rotation.from_euler("y", 180.0, degrees=True).as_matrix()
        target_body[:3, 3] = socket[:3, 3] + 0.050 * socket[:3, 2]
        target_hand = target_body @ np.linalg.inv(memory)
        record["target_world_from_body"] = target_body.tolist()
        record["target_world_from_hand"] = target_hand.tolist()
        record["stage"] = "PLANNING_WITH_EXISTING_TESSERACT"
        save_record()

        active = np.asarray(stepper.latest[0], dtype=np.float64)
        dt = float(dynamic["physics_dt_s"])
        collision_scene = FullRobotCollisionScene(inputs)
        table_bounds = np.asarray(inputs.table_xy_bounds_m, dtype=np.float64)
        table_size = np.r_[table_bounds[:, 1] - table_bounds[:, 0], 1.0]
        table_center = np.r_[np.mean(table_bounds, axis=1), inputs.table_top_z_m - 0.5]
        fixture = runtime["body_assembly_scene"]["fixture"]
        obstacles = {
            "table": fcl.CollisionObject(fcl.Box(*table_size), fcl.Transform(table_center)),
            "fixture": fcl.CollisionObject(
                fcl.Box(*fixture["size_m"]), fcl.Transform(np.asarray(fixture["center_world_m"])),
            ),
        }
        obstacles["receptacle"], _ = _cylinder_from_mesh(
            Path(global_observation["receptacle_obstacle"]["mesh"]), 0.001, socket,
        )
        plug_vertices = np.asarray(inputs.object_contract.model.mesh.vertices_m)
        plug_bounds = {
            "radius_m": float(np.max(np.linalg.norm(plug_vertices[:, :2], axis=1))),
            "z_min_m": float(np.min(plug_vertices[:, 2])),
            "z_max_m": float(np.max(plug_vertices[:, 2])),
        }
        planner_obstacles = []
        for name, size, center in (
            ("table", table_size, table_center),
            ("fixture", fixture["size_m"], fixture["center_world_m"]),
        ):
            pose = np.eye(4)
            pose[:3, 3] = center
            planner_obstacles.append({
                "id": name, "type": "box", "dimensions_m": list(map(float, size)),
                "world_from_primitive_row_major": pose.ravel().tolist(),
            })
        socket_bounds = global_observation["receptacle_obstacle"]["conservative_cylinder"]
        socket_centered = socket.copy()
        socket_centered[:3, 3] += socket[:3, 2] * 0.5 * (
            socket_bounds["z_min_m"] + socket_bounds["z_max_m"])
        planner_obstacles.append({
            "id": "receptacle", "type": "cylinder",
            "dimensions_m": [socket_bounds["z_max_m"] - socket_bounds["z_min_m"],
                             socket_bounds["radius_m"]],
            "world_from_primitive_row_major": socket_centered.ravel().tolist(),
        })
        arm_states, planner_report = _run_external_pose_plan(
            backend="tesseract", repository=repository, ros_domain_id=73,
            stage_name="body_full_pose_to_socket_50mm", output_dir=output,
            start_arm=active[:7], hand_positions=active[7:],
            target_world_from_hand=target_hand, collision_objects=planner_obstacles,
            physics_dt_s=dt, ik_timeout_s=2.0, planning_timeout_s=8.0,
        )
        # The existing planner still proposes paths with its old nail-free hand
        # meshes. Only the current full-hand and carried-body check below can
        # accept the path for this original-nail grasp.
        record["planner_hand_geometry"] = "LEGACY_NAILFREE_PROPOSAL_REQUIRES_CURRENT_HAND_RECHECK"
        record["stage"] = "CHECKING_CURRENT_HAND_AND_CARRIED_BODY"
        target_offset = np.asarray(ft.samples[-1]["active_targets_rad"][:7]) - active[:7]
        arm_states = np.asarray(arm_states) + target_offset
        record["initial_loaded_target_offset_rad"] = target_offset.tolist()
        soft_bounds = np.asarray([MOVEIT_SOFT_ARM_BOUNDS_RAD[name] for name in control.ARM_JOINT_NAMES])
        soft_margin = float(np.min(np.minimum(
            arm_states - soft_bounds[:, 0], soft_bounds[:, 1] - arm_states)))
        record["minimum_soft_limit_margin_after_loaded_offset_rad"] = soft_margin
        if soft_margin < -1e-9:
            raise RuntimeError("loaded nominal target offset crosses an original arm soft limit")
        arm_states, collision_report = _check_held_plug_path(
            collision_scene, arm_states, active[7:], obstacles, memory, plug_bounds, dt, transport_speed,
            allow_speedup=retime_free,retime_straight=retime_straight,
        )
        record["path"] = {"external_planner": planner_report, "collision": collision_report,
                          "plug_bounds": plug_bounds, "checked_hand_variant": inputs.hand_variant}
        save_record()
        if collision_report["first_collision"] is not None:
            raise RuntimeError(f"current-hand carried-body path collides: {collision_report['first_collision']}")
        arm_states = np.vstack((arm_states, np.repeat(
            arm_states[-1:], round(float(dynamic["hold_duration_s"]) / dt), axis=0,
        )))
        np.save(output / "commanded_arm_path_rad.npy", arm_states)
        probe = {
            "authorization": {"simulation_only": True, "hardware_authorized": False},
            "motion": {"maximum_transport_joint_speed_rad_s": transport_speed},
        }
        record["stage"] = "CARRYING_WITH_SINGLE_BODY_KEY_MEMORY"
        save_record()
        _install_rgbd_resume_sync(world, stage)
        execution = _execute_held_plug_path(
            world, stepper, ft, grasp_result, dynamic, arm_states, probe,
            phase="key_probe_body_carry_observation",
        )
        record["execution"] = execution
        if not execution["completed"]:
            raise RuntimeError(f"body carry stopped by joint/force control: {execution['abort_reason']}")

        active = np.asarray(stepper.latest[0], dtype=np.float64)
        hand = np.asarray(inputs.robot_model.forward_kinematics(
            tuple(active), enforce_limits=False)["handbase_link"], dtype=np.float64)
        predicted_body = hand @ memory
        record["arrival_world_from_hand_encoder"] = hand.tolist()
        record["arrival_world_from_body_from_memory"] = predicted_body.tolist()
        record["arrival_encoder_predicted_face_gap_m"] = float((predicted_body[:3, 3] - socket[:3, 3]) @ socket[:3, 2])
        record["stage"] = "REOBSERVING_BODY_AFTER_CARRY"
        save_record()
        refreshed = observe_body_after_transport(
            repository, stage, world, rep, hand, predicted_body, output / "postcarry_body")
        record["refreshed_body_observation"] = refreshed
        record["body_key_reobservations_after_memory"] = 1
        if not refreshed["key_measurement"]["key_direction_measured"]:
            raise RuntimeError("postcarry body key was not observed; no closer motion is allowed")
        memory = np.asarray(refreshed["hand_from_body_visual_memory"])
        refreshed_memory_start_step = int(stepper.step_index)
        record["refreshed_memory_start_step"] = refreshed_memory_start_step
        record["before_carry_memory_retired"] = True
        record["refreshed_hand_from_body_visual_memory"] = memory.tolist()

        # The prior wrist image showed four slots hidden behind the held body.
        # Translate without changing the hand's gravity orientation to expose
        # the socket, using the newly observed body relation for clearance.
        side_target = hand.copy()
        side_target[:3, 3] += np.array([-0.035, 0.0, 0.0])
        side_states, side_ik = _plan_key_probe_descent(
            inputs, active, side_target, dt, 0.035)
        side_states = np.asarray(side_states) + (
            np.asarray(ft.samples[-1]["active_targets_rad"][:7]) - active[:7])
        side_margin = float(np.min(np.minimum(side_states - soft_bounds[:, 0],
                                              soft_bounds[:, 1] - side_states)))
        if side_margin < -1e-9:
            raise RuntimeError("side observation path crosses an original arm soft limit")
        side_states, side_check = _check_held_plug_path(
            collision_scene, side_states, active[7:], obstacles, memory, plug_bounds, dt, transport_speed,
            allow_speedup=retime_free,retime_straight=retime_straight)
        record["side_observation_path"] = {"ik": side_ik, "collision": side_check,
                                           "minimum_soft_limit_margin_rad": side_margin,
                                           "translation_world_m": [-0.035, 0.0, 0.0]}
        if side_check["first_collision"] is not None:
            raise RuntimeError(f"side observation path collides: {side_check['first_collision']}")
        side_states = np.vstack((side_states, np.repeat(
            side_states[-1:], round(float(dynamic["hold_duration_s"]) / dt), axis=0)))
        record["stage"] = "MOVING_TO_SIDE_OBSERVATION"
        save_record()
        side_execution = _execute_held_plug_path(
            world, stepper, ft, grasp_result, dynamic, side_states, probe,
            phase="key_probe_body_side_observation")
        record["side_observation_execution"] = side_execution
        if not side_execution["completed"]:
            raise RuntimeError(f"side observation stopped: {side_execution['abort_reason']}")
        active = np.asarray(stepper.latest[0], dtype=np.float64)
        hand = np.asarray(inputs.robot_model.forward_kinematics(
            tuple(active), enforce_limits=False)["handbase_link"], dtype=np.float64)
        record["stage"] = "CAPTURING_WRIST_SOCKET_RGBD"
        save_record()
        mount = wrist_camera_mount(repository)
        wrist_camera = hand @ np.asarray(mount["hand_from_camera_cv"], dtype=np.float64)
        camera_path = "/World/BodyAssemblyWristCaptureCamera"
        _author_camera(
            stage, camera_path, wrist_camera, resolution=tuple(mount["resolution_px"]),
            focal_length_mm=float(mount["focal_length_mm"]),
            horizontal_aperture_mm=float(mount["horizontal_aperture_mm"]),
            clipping_range_m=tuple(mount["clipping_range_m"]), Gf=Gf, UsdGeom=UsdGeom,
        )
        world.render()
        capture = _capture_rgbd(
            rep=rep, resources=resources, camera_path=camera_path,
            resolution=tuple(mount["resolution_px"]), output_dir=output / "wrist_socket",
            warmup_frames=3, rt_subframes=4,
        )
        depth = np.load(output / "wrist_socket/depth_m.npy")
        camera_from_socket_seed = np.linalg.inv(wrist_camera) @ socket
        # The existing estimator itself selects a thin lip ROI around this
        # global-image seed. No semantic mask or scene transform is supplied.
        measurement = estimate_receptacle_key_from_depth(
            depth, np.isfinite(depth) & (depth > 0.0), np.asarray(mount["intrinsics_3x3"]),
            wrist_camera, camera_from_socket_seed,
        )
        mount.update({
            "wrist_capture_executed": True,
            "wrist_refinement_executed": True,
            "current_socket_visibility_verified": bool(measurement["key_direction_measured"]),
        })
        wrist_record = {
            "capture": capture, "mount": mount,
            "physics_time_s": float(world.current_time),
            "world_from_hand_encoder": hand.tolist(),
            "world_from_camera_cv": wrist_camera.tolist(),
            "camera_from_socket_seed_from_global_rgbd": camera_from_socket_seed.tolist(),
            "mask_source": "VALID_DEPTH_THEN_EXISTING_GLOBAL_SEED_LIP_ROI",
            "measurement": measurement, "online_object_or_contact_truth_used": False,
        }
        (output / "wrist_socket/camera_and_estimate.json").write_text(
            json.dumps(_json_ready(wrist_record), ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        record["wrist_socket_observation_executed"] = True
        record["wrist_socket_result"] = str(output / "wrist_socket/camera_and_estimate.json")
        record["wrist_socket_key_measured"] = bool(measurement["key_direction_measured"])
        record["world_from_socket_wrist_visual"] = measurement.get("world_from_receptacle_row_major")
        record["completed"] = bool(measurement["key_direction_measured"])
        record["stage"] = "WRIST_SOCKET_MEASURED" if record["completed"] else "WRIST_SOCKET_UNRESOLVED"
        if not record["completed"]:
            record["failure_reason"] = measurement.get("reason", "WRIST_SOCKET_UNRESOLVED")
        elif runtime.get("body_key_entry_requested"):
            from te_body_key_entry import run_body_key_entry
            wrist_socket = np.asarray(measurement["world_from_receptacle_row_major"]).reshape(4, 4)
            obstacles["receptacle"], _ = _cylinder_from_mesh(
                Path(global_observation["receptacle_obstacle"]["mesh"]), 0.001, wrist_socket)
            record["stage"] = "VISUAL_ALIGNMENT_AND_ONE_LIGHT_KEY_PROBE"
            record["key_entry_first_step"] = int(stepper.step_index)
            save_record()
            record["key_entry"] = run_body_key_entry(
                repository, runtime, stepper, grasp_result, dynamic, memory,
                wrist_socket, collision_scene, obstacles, plug_bounds, output / "key_entry")
            probe_record = record["key_entry"].get("probe_controller") or {}
            record["contact_motion_commanded"] = bool(probe_record.get("probe_motion_executed"))
            record["completed"] = bool(record["key_entry"].get("completed"))
            record["completed_scope"] = "CONTROLLER_ONLY_PHYSICAL_ENTRY_REQUIRES_POSTRUN_EVALUATION"
            if not record["completed"]:
                record["failure_reason"] = (record["key_entry"].get("failure_reason")
                                             or record["key_entry"].get("original_contact_termination"))
            record["stage"] = "KEY_PROBE_RETURNED_FOR_POSTRUN_PHYSICAL_EVALUATION"
            record["completed"]=bool(record["key_entry"].get("controller_depth_reached"))
            record["completed_scope"]="REQUESTED_KEY_ENTRY_CONTROLLER_REQUIRES_POSTRUN_PHYSICAL_EVALUATION"
            if not record["completed"]:
                record["failure_reason"]=(record["key_entry"].get("failure_reason")
                    or (probe_record or {}).get("termination") or "KEY_ENTRY_NOT_REACHED")
            if (runtime.get("body_support_test_requested")
                    and record["key_entry"].get("controller_depth_reached")):
                from te_body_support_release import run_body_support_release
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
    except Exception as error:
        record["failure_stage"] = record["stage"]
        record.update({"completed": False, "stage": "STOPPED", "failure_reason": str(error)})
        raise
    finally:
        world.pause()
        _close_rgbd_resources(resources)
        samples = ft.samples[first_ft_sample:]
        from trace_metadata import write_gzip_array
        write_gzip_array(output/"transport_joint_ft_samples.json.gz",samples,prepare=_json_ready)
        predictions = []
        if "hand_from_body_visual_memory" in record:
            for sample in samples:
                if sample["step"] >= record.get("key_entry_first_step", float("inf")):
                    # Alignment and entry use the separately saved camera
                    # epochs in key_entry; do not label them with this memory.
                    continue
                hand = np.eye(4)
                hand[:3, :3] = np.asarray(sample["handbase_rotation_world_row_major"])
                hand[:3, 3] = sample["handbase_position_world_m"]
                predictions.append({
                    "step": sample["step"], "phase": sample["phase"],
                    "memory_source": ("AFTER_CARRY_RGBD" if refreshed_memory_start_step is not None
                                      and sample["step"] >= refreshed_memory_start_step else "BEFORE_CARRY_RGBD"),
                    "world_from_body_from_visual_memory": (hand @ (
                        memory if refreshed_memory_start_step is not None
                        and sample["step"] >= refreshed_memory_start_step else initial_memory)).tolist(),
                })
        with gzip.open(output / "body_memory_encoder_predictions.json.gz", "wt", encoding="utf-8") as stream:
            json.dump(predictions, stream, ensure_ascii=False, separators=(",", ":"))
        save_record()
    return _json_ready(record)
