"""Release the body grip at a fixed arm target to test socket support afterward."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np


def run_body_support_release(
    repository, runtime, stepper, grasp_result, dynamic, entry_controller,
    open_hand_target, collision_scene, world_from_socket, obstacles, output,
):
    """Unload the three fingers, open to the current pregrasp, and hold 0.5 s.

    Entry depth and joint/FT signals permit this bounded control experiment;
    they do not prove that the socket supports the body. No object_parts or
    truth-auditor data is read. The arm's last nominal target remains fixed.
    """
    import fcl
    import trimesh

    from te_body_socket_observation import SOCKET_CAD_MM
    from te_foundationpose_handoff_runtime import _first_discrete_collision, _json_ready

    repository, output = Path(repository).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    world, ft = runtime["world"], runtime["nail_body_ft_auditor"]
    first_ft_sample = len(ft.samples)
    command_rows = []
    record = {
        "completed": False, "stage": "BEFORE_GRIP_UNLOAD",
        "simulation_only": True, "hardware_authorized": False,
        "online_object_or_contact_truth_used": False,
        "socket_body_support_evaluated": False,
        "support_result": "REQUIRES_POSTRUN_BODY_EXIT_TILT_AND_KEY_ENGAGEMENT_EVALUATION",
        "arm_nominal_target_changed": False,
        "nut_regrasp_or_rotation_executed": False,
        "prior_hand_body_memory_must_not_drive_followup_as_a_fixed_relation": True,
        "first_step": int(stepper.step_index),
        "open_hand_goal_reached": False,
    }

    def save():
        (output / "support_release_controller_result.json").write_text(
            json.dumps(_json_ready(record), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    try:
        if not (entry_controller.get("controller_entry_depth_reached") is True
                and entry_controller.get("termination")
                in ("ENCODER_ENTRY_DEPTH_REACHED_REQUIRES_PHYSICAL_EVALUATION",
                    "STABLE_AXIAL_CONTACT_REQUIRES_POSTRUN_EVALUATION")):
            raise ValueError("the current entry controller did not reach its bounded endpoint")
        record["release_trigger"] = entry_controller["termination"]
        record["stable_axial_contact_detected"] = bool(entry_controller.get("stable_axial_contact_detected"))
        if stepper.abort_reason is not None:
            raise RuntimeError(f"outer abort remains active: {stepper.abort_reason}")
        open_target = np.asarray(open_hand_target, dtype=np.float64).reshape(4)
        held_arm = np.asarray(ft.samples[-1]["active_targets_rad"][:7], dtype=np.float64).copy()
        hand_target = np.asarray(ft.samples[-1]["active_targets_rad"][7:], dtype=np.float64).copy()
        if not np.isfinite(open_target).all():
            raise ValueError("the current body-pregrasp hand target is nonfinite")
        lower_joints, upper_joints = runtime["robot_model"].joint_limit_vectors()
        if np.any(open_target < np.asarray(lower_joints)[7:]) or np.any(open_target > np.asarray(upper_joints)[7:]):
            raise ValueError("body-pregrasp hand target is outside the original hand limits")
        contact = grasp_result["contact_controller"]
        direction = np.sign(np.asarray(contact.goal) - np.asarray(contact.start))
        if np.any(direction[1:] == 0.0):
            raise ValueError("the three body-grasp closing directions are unavailable")
        tare_rows = [row["active_efforts_nm"][7:] for row in ft.samples if row["phase"] == "tare"]
        tare = np.mean(tare_rows, axis=0)
        if tare.shape != (4,) or not np.isfinite(tare).all():
            raise ValueError("the current finger effort tare is unavailable")
        reference = np.asarray(dynamic["required_closing_joint_effort_nm"], dtype=np.float64)
        dt = float(dynamic["physics_dt_s"])
        maximum_increment = float(dynamic["finger_maximum_speed_rad_s"]) * dt
        stiffness = float(dynamic["hand_stiffness"])
        unload_steps = round(float(dynamic["hold_duration_s"]) / dt)
        if unload_steps < 1 or maximum_increment <= 0.0 or stiffness <= 0.0:
            raise ValueError("the existing finite finger timing/gain is invalid")
        scales = np.r_[0.0, np.asarray(dynamic.get("finger_preload_scales", [1.0, 1.0, 1.0]))]
        finite_limit = np.asarray(contact.target) + float(dynamic["preload_increment_rad"]) * direction * scales
        finite_lower = np.minimum(contact.start, finite_limit)
        finite_upper = np.maximum(contact.start, finite_limit)

        socket = np.asarray(world_from_socket, dtype=np.float64).reshape(4, 4)
        source = trimesh.load(repository / SOCKET_CAD_MM, force="mesh", process=False)
        bvh = fcl.BVHModel()
        bvh.beginModel(len(source.faces), len(source.vertices))
        bvh.addSubModel(np.asarray(source.vertices) * 0.001, np.asarray(source.faces, dtype=np.int32))
        bvh.endModel()
        environment = dict(obstacles)
        environment["receptacle"] = fcl.CollisionObject(bvh, fcl.Transform(socket[:3, :3], socket[:3, 3]))
        record.update({
            "fixed_arm_nominal_target_rad": held_arm.tolist(),
            "initial_hand_target_rad": hand_target.tolist(),
            "open_hand_target_rad": open_target.tolist(),
            "initial_finger_effort_references_nm": reference.tolist(),
            "unload_duration_s": unload_steps * dt,
            "maximum_commanded_finger_speed_rad_s": float(dynamic["finger_maximum_speed_rad_s"]),
            "post_open_hold_duration_s": 0.5,
            "socket_pose_source": "CURRENT_VISUAL_SOCKET_POSE",
            "robot_socket_collision_shape": "SOURCE_CAD_TRIANGLE_MESH",
            "body_grasp_contact_is_allowed_to_disappear": True,
            "checked_hand_variant": runtime["inputs"].hand_variant,
        })
        maximum_actual_finger_speed = 0.0

        def advance(phase, target, effort_reference, *, unloading_effort=False):
            nonlocal maximum_actual_finger_speed
            if stepper.abort_reason is not None:
                raise RuntimeError(f"outer abort remains active: {stepper.abort_reason}")
            # A loaded position-drive target includes the spring deflection
            # that generates grip force. It is not a reachable finger pose
            # while the body is between the fingers. During monotone effort
            # unloading, check the encoder geometry before and after each
            # step; free opening still checks the next position target too.
            measured_before = np.asarray(stepper.latest[0])
            geometry_arm = measured_before[:7] if unloading_effort else held_arm
            geometry_hand = measured_before[7:] if unloading_effort else target
            collision = _first_discrete_collision(
                collision_scene, geometry_arm, geometry_hand, environment)
            if collision is not None:
                record["first_geometry_stop"] = {"step": int(stepper.step_index),
                    "source": ("CURRENT_ENCODER_FK_BEFORE_EFFORT_UNLOAD" if unloading_effort
                               else "NEXT_COMMAND_FK"), **collision}
                raise RuntimeError(f"opening path collides with source geometry: {collision}")
            current_step = int(stepper.step_index)
            stepper.advance(phase, held_arm, target)
            if int(stepper.step_index) > current_step:
                command_rows.append({"step": current_step, "phase": phase,
                                     "finger_effort_reference_nm": np.asarray(effort_reference).tolist(),
                                     "arm_nominal_target_rad": held_arm.tolist(),
                                     "hand_target_rad": np.asarray(target).tolist()})
                maximum_actual_finger_speed = max(maximum_actual_finger_speed,
                    float(np.max(np.abs(np.asarray(stepper.latest[1])[8:]))))
                measured = np.asarray(stepper.latest[0])
                collision = _first_discrete_collision(
                    collision_scene, measured[:7], measured[7:], environment)
                if collision is not None:
                    record["first_geometry_stop"] = {"step": current_step,
                                                      "source": "CURRENT_ENCODER_FK", **collision}
                    raise RuntimeError(f"measured open-hand geometry collides: {collision}")
            if stepper.abort_reason is not None:
                raise RuntimeError(f"joint/FT protection stopped grip release: {stepper.abort_reason}")

        record["stage"] = "LINEAR_EFFORT_UNLOAD"
        record["effort_unload_geometry_input"] = "ENCODERS_BEFORE_AND_AFTER_EACH_STEP"
        record["free_open_geometry_input"] = "NEXT_POSITION_TARGET_AND_ENCODERS"
        save()
        world.play()
        for index in range(unload_steps):
            desired = reference * (1.0 - (index + 1) / unload_steps)
            measured = direction[1:] * (np.asarray(stepper.latest[2])[8:] - tare[1:])
            # Unload only: a brief low effort reading must not reclose a finger.
            opening_increment = np.clip((desired - measured) / stiffness, -maximum_increment, 0.0)
            hand_target[1:] = np.clip(hand_target[1:] + direction[1:] * opening_increment,
                                     finite_lower[1:], finite_upper[1:])
            advance("key_probe_body_support_unload", hand_target, desired, unloading_effort=True)
        record["unload_last_step"] = int(stepper.step_index)
        record["stage"] = "OPENING_TO_BODY_PREGRASP"
        save()
        open_start = hand_target.copy()
        open_steps = max(1, int(np.ceil(np.max(np.abs(open_target - open_start)) / maximum_increment)))
        for index in range(open_steps):
            fraction = (index + 1) / open_steps
            hand_target = (1.0 - fraction) * open_start + fraction * open_target
            advance("key_probe_body_support_open", hand_target, np.zeros(3))
        record["opening_last_step"] = int(stepper.step_index)
        stepper.payload_compensation_fraction = 0.0
        record["payload_feedforward_fraction_after_open_command"] = 0.0
        record["body_pose_relation_retired_after_open_command"] = True
        record["stage"] = "OPEN_HAND_SUPPORT_OBSERVATION_HOLD"
        for _ in range(round(0.5 / dt)):
            advance("key_probe_body_support_hold", open_target, np.zeros(3))
        actual_open = np.asarray(stepper.latest[0])[7:]
        error = float(np.max(np.abs(actual_open - open_target)))
        tolerance = float(dynamic["contact_position_error_rad"])
        record.update({
            "stage": "OPEN_HAND_HOLD_FINISHED_REQUIRES_POSTRUN_SUPPORT_EVALUATION",
            "actual_hand_positions_rad": actual_open.tolist(),
            "maximum_open_hand_position_error_rad": error,
            "open_hand_position_comparison_tolerance_rad": tolerance,
            "position_tolerance_source": "EXISTING_DYNAMIC_CONTACT_POSITION_ERROR_SOFTWARE_VALUE",
            "open_hand_goal_reached": error <= tolerance,
            "maximum_actual_finger_speed_rad_s": maximum_actual_finger_speed,
            "completed": True,
        })
    except Exception as error:
        record.update(failure_stage=record["stage"], stage="STOPPED", failure_reason=str(error))
        raise
    finally:
        world.pause()
        record["last_step"] = int(stepper.step_index)
        record["executed_command_steps"] = len(command_rows)
        record["outer_abort_reason"] = stepper.abort_reason
        with gzip.open(output / "release_commands.json.gz", "wt", encoding="utf-8") as stream:
            json.dump(command_rows, stream, ensure_ascii=False, separators=(",", ":"))
        with gzip.open(output / "release_joint_ft_samples.json.gz", "wt", encoding="utf-8") as stream:
            json.dump(_json_ready(ft.samples[first_ft_sample:]), stream, ensure_ascii=False, separators=(",", ":"))
        save()
    return _json_ready(record)
