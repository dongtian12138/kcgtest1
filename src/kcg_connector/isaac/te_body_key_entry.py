"""Bounded visual alignment and one low-force body-held key-entry attempt."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import yaml


def run_body_key_entry(
    repository, runtime, stepper, grasp_result, dynamic, hand_from_body,
    world_from_socket, collision_scene, obstacles, plug_bounds, output,
):
    """Align from current RGB-D, probe once, and unload a failed low-force probe.

    Object/contact truth is neither read nor evaluated here. All body poses are
    current visual observations or predictions from joint FK and visual memory.
    The returned controller depth is not evidence of physical key engagement.
    """
    import fcl
    import omni.replicator.core as rep
    import omni.usd
    import trimesh
    from scipy.spatial.transform import Rotation

    from te_body_socket_observation import SOCKET_CAD_MM, observe_body_after_transport
    from te_foundationpose_handoff_runtime import (
        MOVEIT_SOFT_ARM_BOUNDS_RAD, control, _check_held_plug_path,
        _execute_held_plug_path, _first_discrete_collision, _json_ready,
        _plan_key_probe_descent, _run_light_contact_key_search,
    )

    repository, output = Path(repository).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    world, inputs, ft = runtime["world"], runtime["inputs"], runtime["nail_body_ft_auditor"]
    stage = omni.usd.get_context().get_stage()
    memory = np.asarray(hand_from_body, dtype=np.float64).reshape(4, 4).copy()
    socket = np.asarray(world_from_socket, dtype=np.float64).reshape(4, 4).copy()
    dt = float(dynamic["physics_dt_s"])
    config_path = repository / runtime["body_assembly_control_config"]
    probe = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not (probe["authorization"]["simulation_only"] is True
            and probe["authorization"]["hardware_authorized"] is False
            and probe["authorization"]["light_contact_key_entry_authorized"] is True):
        raise ValueError("the existing light-contact simulation authorization is unavailable")
    probe["motion"]["trial_wrist_offsets_deg"] = [0.0]
    probe["motion"]["search_mode"] = "retract_rotate_probe"
    probe["motion"]["maximum_total_search_duration_s"] = min(
        probe["motion"]["maximum_total_search_duration_s"],
        probe["motion"]["maximum_contact_duration_s"],
    )
    motion = probe["motion"]
    soft = np.asarray([MOVEIT_SOFT_ARM_BOUNDS_RAD[name] for name in control.ARM_JOINT_NAMES])
    first_ft_sample = len(ft.samples)
    contact_record = None
    record = {
        "stage": "BEFORE_VISUAL_ALIGNMENT", "completed": False,
        "simulation_only": True, "hardware_authorized": False,
        "physical_key_entry_verified": False, "postrun_geometry_evaluation_required": True,
        "online_object_or_contact_truth_used": False,
        "software_alignment_goal": {"center_error_m": 0.00005,
                                    "full_rotation_error_deg": 0.10,
                                    "maximum_alignment_moves_and_observations": 3},
        "software_goal_is_not_physical_contact_evidence": True,
        "world_from_socket_visual": socket.tolist(),
        "initial_hand_from_body_visual_memory": memory.tolist(),
        "probe_source_config": str(config_path),
        "probe_config": probe, "probe_controller": None,
        "alignment": [], "motions": [], "contact_attempted": False,
        "unload": {"requested": False, "executed": False},
    }

    def save():
        (output / "key_entry_controller_result.json").write_text(
            json.dumps(_json_ready(record), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def hand_pose():
        active = np.asarray(stepper.latest[0], dtype=np.float64)
        hand = np.asarray(inputs.robot_model.forward_kinematics(
            tuple(active), enforce_limits=False)["handbase_link"], dtype=np.float64)
        return active, hand

    def desired_body(gap):
        pose = socket.copy()
        pose[:3, :3] = socket[:3, :3] @ Rotation.from_euler("y", 180.0, degrees=True).as_matrix()
        pose[:3, 3] += float(gap) * socket[:3, 2]
        return pose

    # Keep the true source socket shape for robot avoidance during contact and
    # unloading. Its placement comes exclusively from the current wrist image.
    source_mesh = trimesh.load(repository / SOCKET_CAD_MM, force="mesh", process=False)
    socket_bvh = fcl.BVHModel()
    socket_bvh.beginModel(len(source_mesh.faces), len(source_mesh.vertices))
    socket_bvh.addSubModel(np.asarray(source_mesh.vertices) * 0.001,
                          np.asarray(source_mesh.faces, dtype=np.int32))
    socket_bvh.endModel()
    probe_obstacles = dict(obstacles)
    probe_obstacles["receptacle"] = fcl.CollisionObject(
        socket_bvh, fcl.Transform(socket[:3, :3], socket[:3, 3]))

    def move_checked(target_hand, phase, speed, *, unloading=False, settle=True):
        if stepper.abort_reason is not None:
            raise RuntimeError(f"outer joint/FT abort remains active: {stepper.abort_reason}")
        active, _ = hand_pose()
        arm_states, ik = _plan_key_probe_descent(inputs, active, target_hand, dt, float(speed))
        offset = np.asarray(ft.samples[-1]["active_targets_rad"][:7]) - active[:7]
        arm_states = np.asarray(arm_states) + offset
        margin = float(np.min(np.minimum(arm_states - soft[:, 0], soft[:, 1] - arm_states)))
        item = {"phase": phase, "ik": ik, "loaded_target_offset_rad": offset.tolist(),
                "target_world_from_hand": np.asarray(target_hand).tolist(),
                "minimum_soft_joint_margin_rad": margin,
                "first_step": int(stepper.step_index), "unloading": unloading}
        record["motions"].append(item)
        if margin < -1e-9:
            save()
            raise RuntimeError(f"{phase}: original soft joint limit would be crossed")
        checked_obstacles = ({name: value for name, value in probe_obstacles.items()
                              if name != "receptacle"} if unloading else obstacles)
        arm_states, check = _check_held_plug_path(
            collision_scene, arm_states, active[7:], checked_obstacles, memory,
            plug_bounds, dt, float(motion["maximum_transport_joint_speed_rad_s"]),
        )
        item["collision"] = check
        if check["first_collision"] is not None:
            save()
            raise RuntimeError(f"{phase}: {check['first_collision']}")
        if unloading:
            # Existing body/socket contact must be allowed to disappear along
            # the outward path. Robot/socket contacts remain forbidden.
            for index, arm in enumerate(arm_states):
                collision = _first_discrete_collision(
                    collision_scene, arm, active[7:], {"receptacle": probe_obstacles["receptacle"]})
                if collision is not None:
                    item["robot_socket_collision"] = {"sample": index, **collision}
                    save()
                    raise RuntimeError(f"unload robot path collides: {collision}")
        if settle:
            arm_states = np.vstack((arm_states, np.repeat(
                arm_states[-1:], round(float(dynamic["hold_duration_s"]) / dt), axis=0)))
        np.save(output / f"{phase}_arm_path_rad.npy", arm_states)
        save()
        execution = _execute_held_plug_path(
            world, stepper, ft, grasp_result, dynamic, arm_states, probe, phase=phase)
        item["execution"] = execution
        item["last_step"] = int(stepper.step_index)
        save()
        if not execution["completed"]:
            raise RuntimeError(f"{phase} stopped: {execution['abort_reason']}")
        return execution

    def unload_failed_probe():
        record["unload"]["requested"] = True
        if stepper.abort_reason is not None:
            record["unload"].update(reason="OUTER_ABORT_NOT_CLEARED", abort_reason=stepper.abort_reason)
            return
        _, hand = hand_pose()
        predicted = hand @ memory
        gap = float((predicted[:3, 3] - socket[:3, 3]) @ socket[:3, 2])
        minimum_gap = float(motion["precontact_face_gap_m"])
        record["unload"]["initial_encoder_predicted_gap_m"] = gap
        if gap >= minimum_gap:
            record["unload"].update(reason="ALREADY_AT_OR_OUTSIDE_PRECONTACT_GAP", gap_reached=True)
            return
        # A 0.1 mm outward reserve leaves room for ordinary arm tracking error;
        # it does not change contact forces or any inward travel allowance.
        retreat_hand = hand.copy()
        retreat_hand[:3, 3] += (minimum_gap + 0.0001 - gap) * socket[:3, 2]
        first_step = int(stepper.step_index)
        record["unload"]["first_step"] = first_step
        try:
            execution = move_checked(
                retreat_hand, "key_probe_body_unload", motion["maximum_precontact_speed_m_s"],
                unloading=True, settle=False)
            _, final_hand = hand_pose()
            final_gap = float(((final_hand @ memory)[:3, 3] - socket[:3, 3]) @ socket[:3, 2])
            record["unload"].update(execution=execution, final_encoder_predicted_gap_m=final_gap,
                                    gap_reached=final_gap >= minimum_gap)
        except Exception as error:
            record["unload"].update(error=str(error), gap_reached=False)
        finally:
            record["unload"]["executed"] = int(stepper.step_index) > first_step
            record["unload"]["last_step"] = int(stepper.step_index)

    try:
        if not np.isfinite(memory).all() or not np.isfinite(socket).all():
            raise ValueError("current visual transforms are nonfinite")
        if stepper.abort_reason is not None or grasp_result.get("failure_reason"):
            raise RuntimeError("current joint/FT controller is not clear for alignment")
        world.pause()
        target = desired_body(motion["transport_face_gap_m"])
        aligned = False
        for attempt in range(3):
            record["stage"] = f"VISUAL_ALIGNMENT_{attempt + 1}"
            move_checked(target @ np.linalg.inv(memory), f"key_probe_body_align_{attempt + 1}", 0.02)
            _, hand = hand_pose()
            observation = observe_body_after_transport(
                repository, stage, world, rep, hand, hand @ memory, output / f"alignment_rgbd_{attempt + 1}")
            item = {"attempt": attempt + 1, "step": int(stepper.step_index), "observation": observation}
            record["alignment"].append(item)
            if not observation["key_measurement"]["key_direction_measured"]:
                record.update(stage="ALIGNMENT_KEY_UNOBSERVED", failure_reason="No new visual memory; contact was not started")
                return _json_ready(record)
            memory = np.asarray(observation["hand_from_body_visual_memory"])
            observed = np.asarray(observation["key_measurement"]["world_from_plug_row_major"]).reshape(4, 4)
            position_error = float(np.linalg.norm(observed[:3, 3] - target[:3, 3]))
            rotation_error = float(Rotation.from_matrix(target[:3, :3].T @ observed[:3, :3]).magnitude())
            item["visual_error"] = {
                "center_error_m": position_error,
                "full_rotation_error_deg": float(np.degrees(rotation_error)),
                "axis_error_deg": float(np.degrees(np.arccos(np.clip(observed[:3, 2] @ target[:3, 2], -1, 1)))),
                "main_key_direction_error_deg": float(np.degrees(np.arccos(np.clip(observed[:3, 1] @ target[:3, 1], -1, 1)))),
            }
            record["current_hand_from_body_visual_memory"] = memory.tolist()
            aligned = position_error <= 0.00005 and rotation_error <= np.deg2rad(0.10)
            item["software_alignment_goal_reached"] = aligned
            save()
            if aligned:
                break
            if (position_error > float(motion["maximum_xy_correction_m"])
                    or rotation_error > np.deg2rad(float(motion["maximum_tilt_correction_deg"]))):
                record.update(stage="VISUAL_CORRECTION_OUTSIDE_BOUND", failure_reason="Observed error exceeds the original 2 mm / 1 degree correction range")
                return _json_ready(record)
        if not aligned:
            record.update(stage="VISUAL_ALIGNMENT_GOAL_NOT_REACHED", failure_reason="Three visual alignment attempts exhausted; contact was not started")
            return _json_ready(record)
        record["stage"] = "DESCENDING_TO_PRECONTACT_GAP"
        precontact_body_goal = desired_body(motion["precontact_face_gap_m"])
        precontact_hand_goal = precontact_body_goal @ np.linalg.inv(memory)
        move_checked(precontact_hand_goal,
                     "key_probe_body_precontact", motion["maximum_precontact_speed_m_s"])
        # The loaded joint-target bias changes over the long descent. Close
        # this remaining task-space error while the body is still outside the
        # socket, using the current encoders and the current visual relation.
        # No contact/object truth or post-entry yaw observation is involved.
        record["precontact_tracking_corrections"] = []
        for correction in range(4):
            _, hand = hand_pose()
            predicted_body = hand @ memory
            position_error = float(np.linalg.norm(predicted_body[:3, 3]-precontact_body_goal[:3, 3]))
            rotation_error = float(Rotation.from_matrix(
                precontact_body_goal[:3, :3].T @ predicted_body[:3, :3]).magnitude())
            record["precontact_tracking_corrections"].append({
                "step": int(stepper.step_index), "correction_count": correction,
                "position_error_m": position_error,
                "rotation_error_deg": float(np.degrees(rotation_error)),
                "body_origin_in_observed_socket_m": (socket[:3, :3].T @ (
                    predicted_body[:3, 3]-socket[:3, 3])).tolist(),
                "source": "CURRENT_ENCODER_FK_AND_CURRENT_VISUAL_BODY_RELATION",
                "physical_contact_truth_used": False})
            save()
            if position_error <= .00005 and rotation_error <= np.deg2rad(.10):
                break
            if correction == 3:
                raise RuntimeError("precontact task-space correction did not reach the existing alignment goal")
            if (position_error > float(motion["maximum_xy_correction_m"])
                    or rotation_error > np.deg2rad(float(motion["maximum_tilt_correction_deg"]))):
                raise RuntimeError("precontact task-space error exceeds the existing bounded correction region")
            record["stage"] = "CORRECTING_PRECONTACT_TRACKING_ERROR"
            move_checked(precontact_hand_goal, f"key_probe_body_precontact_correct_{correction+1}",
                         min(float(motion["maximum_precontact_speed_m_s"]), .001))
        payload = dict(runtime["payload_model"])
        payload["center_of_mass_from_hand_m"] = (
            memory @ np.r_[np.asarray(payload["center_of_mass_object_m"]), 1.0])[:3].tolist()
        record["contact_payload_model_from_visual_memory"] = payload
        record.update(stage="SINGLE_LOW_FORCE_KEY_PROBE", contact_attempted=True,
                      contact_first_step=int(stepper.step_index))
        save()
        try:
            contact_record = _run_light_contact_key_search(
                world, stepper, ft, grasp_result, dynamic, inputs, probe,
                memory, socket, payload, collision_scene, probe_obstacles)
        except Exception as error:
            world.pause()
            contact_record = {"termination": "CONTACT_CONTROLLER_EXCEPTION", "error": str(error),
                              "controller_entry_depth_reached": False, "samples": []}
        record["probe_controller"] = contact_record
        (output / "contact_controller_record.json").write_text(
            json.dumps(_json_ready(contact_record), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        controller_depth_reached = bool(
            contact_record.get("controller_entry_depth_reached")
            and contact_record.get("termination") in (
                "ENCODER_ENTRY_DEPTH_REACHED_REQUIRES_PHYSICAL_EVALUATION",
                "STABLE_AXIAL_CONTACT_REQUIRES_POSTRUN_EVALUATION")
            and stepper.abort_reason is None)
        record["stable_axial_contact_detected"] = bool(contact_record.get("stable_axial_contact_detected"))
        record["controller_depth_reached"] = controller_depth_reached
        record["original_contact_termination"] = contact_record.get("termination")
        if not controller_depth_reached:
            unload_failed_probe()
        record["stage"] = "PROBE_FINISHED_REQUIRES_POSTRUN_EVALUATION"
        record["completed"] = controller_depth_reached
    except Exception as error:
        record.update(failure_stage=record["stage"], stage="STOPPED", failure_reason=str(error))
        raise
    finally:
        world.pause()
        record["outer_abort_reason"] = stepper.abort_reason
        record["last_step"] = int(stepper.step_index)
        with gzip.open(output / "joint_ft_samples.json.gz", "wt", encoding="utf-8") as stream:
            json.dump(_json_ready(ft.samples[first_ft_sample:]), stream, ensure_ascii=False, separators=(",", ":"))
        save()
    return _json_ready(record)
