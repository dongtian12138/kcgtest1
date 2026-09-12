"""One current-palm-image-guided, finite regrasp of the coupling nut."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


def body_yaw_swept_bound(vertices, faces, *, band_height_m=.00025, margin_m=.0001):
    """Conservative axial CAD envelope without filling the body's large gaps.

    Each occupied axial band bounds all source-triangle radii and all yaw
    angles. A circumscribed polygon and 0.1 mm pose reserve preserve clearance
    conservatism; this is only an avoidance mesh, never a physical collider.
    """
    import trimesh

    vertices, faces = np.asarray(vertices), np.asarray(faces)
    triangle_z = vertices[faces, 2]
    low, high = triangle_z.min(axis=1), triangle_z.max(axis=1)
    triangle_radius = np.linalg.norm(vertices[faces, :2], axis=2).max(axis=1)
    cuts = np.r_[np.arange(vertices[:, 2].min(), vertices[:, 2].max(), band_height_m), vertices[:, 2].max()]
    radii = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        hit = (low <= b + 1e-10) & (high >= a - 1e-10)
        radii.append(float(triangle_radius[hit].max()) if np.any(hit) else 0.0)
    components, bands = [], []
    index = 0
    sections = 128
    while index < len(radii):
        if radii[index] == 0:
            index += 1
            continue
        end = index + 1
        while end < len(radii) and radii[end] > 0:
            end += 1
        profile = [[0.0, cuts[index] - margin_m]]
        previous_radius = None
        for band in range(index, end):
            # Supplier float vertices give nanometre-scale radius differences.
            # Round outward to avoid degenerate annuli after mesh welding.
            radius = (np.ceil((radii[band] + margin_m) / 1e-6) * 1e-6) / np.cos(np.pi / sections)
            z0 = cuts[band] - margin_m if band == index else cuts[band]
            z1 = cuts[band + 1] + margin_m if band == end - 1 else cuts[band + 1]
            if radius != previous_radius:
                profile.append([radius, z0])
            profile.append([radius, z1])
            previous_radius = radius
            bands.append({"z_min_m": float(z0), "z_max_m": float(z1), "source_radius_bound_m": radii[band]})
        profile.append([0.0, cuts[end] + margin_m])
        # Build in millimetres so the mesher's absolute small-face threshold
        # does not discard micrometre-scale radial step annuli.
        component = trimesh.creation.revolve(np.asarray(profile) * 1000.0, sections=sections)
        component.apply_scale(.001)
        if not component.is_volume:
            raise ValueError("the source-body avoidance envelope is not a closed positive volume")
        components.append(component)
        index = end
    if not components:
        raise ValueError("the source-body envelope is empty")
    mesh = trimesh.util.concatenate(components)
    return mesh, {"source": "AXIAL_SOURCE_CAD_TRIANGLE_RADIUS_BOUNDS_OVER_ALL_YAW",
                  "band_height_m": band_height_m, "pose_reserve_m": margin_m,
                  "radius_rounding_up_m": 1e-6,
                  "angular_sections": sections, "component_count": len(components), "bands": bands,
                  "used_as_physical_collision_geometry": False}


def run_body_nut_regrasp(repository, runtime, stepper, dynamic, observation,
                         world_from_socket, collision_scene, obstacles, output,
                         *, prepare_geometry_only=False, start_from_nut_grip=False,
                         grasp_axis_shift_override_m=None,
                         effort_regulation_time_constant_s_override=None):
    """Open, move from fresh five-DOF vision, reobserve, and grip the nut.

    No previous hand/body rigid relation or simulator object/contact state is
    consumed. Physical nut-only grasp success is evaluated after execution.
    """
    import fcl
    import omni.replicator.core as rep
    import omni.usd
    import trimesh
    from scipy.spatial.transform import Rotation

    from build_te_free_split_plug import BODY_VISUAL, NUT_VISUAL, _load_single_usd_mesh
    from te_body_socket_observation import SOCKET_CAD_MM, observe_released_plug_from_rgbd
    from te_foundationpose_handoff_runtime import (
        _first_discrete_collision, _json_ready, _plan_key_probe_descent,
        MOVEIT_SOFT_ARM_BOUNDS_RAD, control,
    )

    repository, output = Path(repository).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load((repository / runtime["body_assembly_control_config"]).read_text())
    geometry_path = repository / config["nut_regrasp"]["geometry_plan"]
    geometry = json.loads(geometry_path.read_text())
    canonical = np.asarray(geometry["canonical_body_from_hand_for_nut_grasp"]).copy()
    grasp_axis_shift = float(config["nut_regrasp"].get("canonical_axial_shift_m", 0.))
    if grasp_axis_shift_override_m is not None:
        grasp_axis_shift = float(grasp_axis_shift_override_m)
    if not np.isfinite(grasp_axis_shift) or abs(grasp_axis_shift) > .003:
        raise ValueError("the declared nut-grasp axial shift must be within 3 mm")
    canonical[2, 3] += grasp_axis_shift
    open_goal = np.asarray(geometry["open_hand_positions_rad"])
    closing=geometry.get("finite_closing_goal_rad")
    if closing is None:
        source_config=yaml.safe_load((repository/"src/kcg_connector/config/te_nail_tip_body_grasp_v1.yaml").read_text())
        closing=source_config["dynamic"]["nail_body_grasp_control_plan"]["final_joint_positions_rad"]
    close_goal=np.asarray(closing).copy()
    # Palm layout belongs to this nut grasp. Finger closure must not send
    # the palm back to the legacy Body layout.
    close_goal[0]=open_goal[0]
    world, inputs, ft = runtime["world"], runtime["inputs"], runtime["nail_body_ft_auditor"]
    stage = omni.usd.get_context().get_stage()
    dt = float(dynamic["physics_dt_s"])
    effort_tau = float(config["nut_regrasp"].get("effort_regulation_time_constant_s", 0.)
        if effort_regulation_time_constant_s_override is None else effort_regulation_time_constant_s_override)
    if not np.isfinite(effort_tau) or effort_tau < 0:
        raise ValueError("nut grip target relaxation time must be finite and nonnegative")
    effort_gain = dt/(effort_tau+dt) if effort_tau > 0 else 1.
    root_preload=config["nut_regrasp"].get("root_moment_preload")
    first_ft = len(ft.samples)
    command_rows = []
    record = {"completed": False, "stage": "BEFORE_NUT_REGRASP", "first_step": int(stepper.step_index),
        "simulation_only": True, "hardware_authorized": False,
        "online_object_or_contact_truth_used": False, "old_hand_body_memory_used": False,
        "physical_nut_grasp_verified": False, "physical_result": "REQUIRES_POSTRUN_CONTACT_AND_KEY_GEOMETRY",
        "geometry_plan": str(geometry_path), "motions": [], "nut_rotation_commanded": False}
    record["nailfree_grasp_reuse"] = config["nut_regrasp"].get("nailfree_grasp_reuse")
    record["canonical_grasp_axial_shift_m"] = grasp_axis_shift
    record["grasp_axis_shift_override_used"] = grasp_axis_shift_override_m is not None
    record["grip_effort_target_update"] = {
        "time_constant_s": effort_tau, "per_step_error_relaxation_gain": effort_gain,
        "signal": "UNFILTERED_NATIVE_PROJECTED_JOINT_FORCE_MINUS_OPEN_HAND_TARE",
        "raw_measurement_or_protection_changed": False,
        "independent_time_scale_reference": "EXISTING_ROTATION_TARGET_RELAXATION_AND_HAND_DAMPING_OVER_STIFFNESS"}
    record["opening_from_existing_nut_grip"] = bool(start_from_nut_grip)

    # The legacy roster uses one convex envelope per fingertip; physics uses
    # the original source mesh's convex decomposition. Load the same cooked
    # geometry for this close-range check, preserving the complete link roster.
    if config["nut_regrasp"].get("cooked_hand_geometry"):
        manifest_path = repository / config["nut_regrasp"]["cooked_hand_geometry"]
        manifest = json.loads(manifest_path.read_text())
        mesh_path = Path(manifest["mesh_data"])
        if (manifest["contains_object_pose_contact_or_witness_data"] is not False
                or hashlib.sha256(Path(runtime["robot_asset"]).read_bytes()).hexdigest() != manifest["source_robot_asset_sha256"]
                or hashlib.sha256(mesh_path.read_bytes()).hexdigest() != manifest["mesh_data_sha256"]):
            raise ValueError("cooked fingertip geometry does not match the unmodified robot asset")
        meshes = np.load(mesh_path)
        for link in ("f1Link3", "f2Link2", "f3Link3"):
            triangles = np.asarray(meshes[link])
            model = fcl.BVHModel()
            vertices = triangles.reshape(-1, 3)
            faces = np.arange(len(vertices), dtype=np.int32).reshape(-1, 3)
            model.beginModel(len(faces), len(vertices))
            model.addSubModel(vertices, faces)
            model.endModel()
            collision_scene.objects[link] = fcl.CollisionObject(model)
        record["fingertip_planning_geometry"] = {
            "manifest": str(manifest_path), "source": manifest["source"],
            "physical_robot_geometry_changed": False,
            "object_pose_or_contact_truth_consumed": False,
            "full_robot_collision_link_count": len(collision_scene.objects)}

    def save():
        (output / "nut_regrasp_controller_result.json").write_text(
            json.dumps(_json_ready(record), ensure_ascii=False, indent=2) + "\n")

    def hand_pose():
        joints = np.asarray(stepper.latest[0])
        hand = np.asarray(inputs.robot_model.forward_kinematics(tuple(joints), enforce_limits=False)["handbase_link"])
        return joints, hand

    def target_from_palm(measurement):
        if not measurement.get("position_and_axis_measured"):
            raise ValueError("the current palm frame has no position and directed-axis measurement")
        body = np.asarray(measurement["world_from_plug_five_dof"], dtype=np.float64).reshape(4, 4)
        _, hand = hand_pose()
        axis = body[:3, 2] / np.linalg.norm(body[:3, 2])
        x = hand[:3, 0] - axis * float(axis @ hand[:3, 0])
        if np.linalg.norm(x) < .5:
            raise ValueError("the current hand transverse basis is inconsistent with the observed plug axis")
        x /= np.linalg.norm(x)
        target = np.eye(4)
        target[:3, :3] = np.column_stack((x, np.cross(axis, x), axis))
        # Canonical translation is expressed in the source Body frame; the
        # observed cylinder leaves yaw free. Express that offset in the hand
        # frame before using the currently selected transverse hand basis.
        target[:3, 3] = body[:3, 3] + target[:3, :3] @ canonical[:3, :3].T @ canonical[:3, 3]
        return body, target

    environment = dict(obstacles)
    mesh = trimesh.load(repository / SOCKET_CAD_MM, force="mesh", process=False)
    bvh = fcl.BVHModel()
    bvh.beginModel(len(mesh.faces), len(mesh.vertices))
    bvh.addSubModel(np.asarray(mesh.vertices) * .001, np.asarray(mesh.faces, dtype=np.int32))
    bvh.endModel()
    socket = np.asarray(world_from_socket)
    environment["receptacle"] = fcl.CollisionObject(bvh, fcl.Transform(socket[:3, :3], socket[:3, 3]))
    part_bounds, part_obstacles = {}, {}
    body_envelope_report = None
    for name, path in (("body", BODY_VISUAL), ("nut", NUT_VISUAL)):
        vertices, _ = _load_single_usd_mesh(path)
        lower, upper = float(vertices[:, 2].min()), float(vertices[:, 2].max())
        radius = float(np.linalg.norm(vertices[:, :2], axis=1).max())
        part_bounds[name] = (lower, upper, radius)
        if name == "body":
            source_vertices, source_faces = _load_single_usd_mesh(path)
            envelope, body_envelope_report = body_yaw_swept_bound(
                source_vertices, source_faces,
                margin_m=float(config["nut_regrasp"].get("body_pose_reserve_m", .0001)))
            model = fcl.BVHModel()
            model.beginModel(len(envelope.faces), len(envelope.vertices))
            model.addSubModel(np.asarray(envelope.vertices), np.asarray(envelope.faces, dtype=np.int32))
            model.endModel()
            part_obstacles[name] = fcl.CollisionObject(model)
        else:
            part_obstacles[name] = fcl.CollisionObject(fcl.Cylinder(radius, upper - lower))
    record["body_avoidance_envelope"] = body_envelope_report

    def locate_part_bounds(body):
        for name, obj in part_obstacles.items():
            lower, upper, _ = part_bounds[name]
            center = body[:3, 3] if name == "body" else body[:3, 3] + .5 * (lower + upper) * body[:3, 2]
            obj.setTransform(fcl.Transform(body[:3, :3], center))

    terminals = {"f1Link3", "f2Link2", "f3Link3"}
    request = fcl.CollisionRequest(num_max_contacts=1, enable_contact=False)

    def check(arm, hand, *, nut_contact=False):
        collision = _first_discrete_collision(collision_scene, arm, hand, environment)
        if collision is not None:
            return collision
        for name, obj in part_obstacles.items():
            for link, robot in collision_scene.objects.items():
                if name == "nut" and nut_contact and link in terminals:
                    continue
                if fcl.collide(robot, obj, request, fcl.CollisionResult()):
                    return {"kind": "visual_plug_bound", "part": name, "link": link}
        return None

    def advance(phase, arm, hand, *, nut_contact=False):
        if stepper.abort_reason is not None:
            raise RuntimeError(f"outer joint/FT stop remains active: {stepper.abort_reason}")
        measured, _ = hand_pose()
        collision = check(measured[:7], measured[7:], nut_contact=nut_contact)
        if collision is None and not nut_contact:
            collision = check(arm, hand)
        if collision is not None:
            record["geometry_stop"] = {"step": int(stepper.step_index), **collision}
            raise RuntimeError(f"nut regrasp path collision: {collision}")
        command_rows.append({"step": int(stepper.step_index), "phase": phase,
                             "arm_target_rad": np.asarray(arm).tolist(), "hand_target_rad": np.asarray(hand).tolist()})
        stepper.advance(phase, arm, hand)
        measured, _ = hand_pose()
        collision = check(measured[:7], measured[7:], nut_contact=nut_contact)
        if collision is not None or stepper.abort_reason is not None:
            record["geometry_stop"] = collision
            raise RuntimeError(f"nut regrasp stopped: {collision or stepper.abort_reason}")

    try:
        if stepper.abort_reason is not None or not config["authorization"]["simulation_only"]:
            raise ValueError("the existing controller is not clear for simulation regrasp")
        if (not np.allclose(canonical[:3,:3].T@canonical[:3,:3],np.eye(3),atol=1e-8)
                or np.linalg.norm(canonical[:3,2]-np.array([0.,0.,1.]))>1e-4):
            raise ValueError("this nut-grasp adapter requires a coaxial canonical hand pose")
        mechanism=getattr(world,"hand_mechanism",None)
        if mechanism is not None and geometry.get("finger_mechanism_id")!=mechanism.setup["mechanism_id"]:
            raise ValueError("nut-grasp geometry differs from the running hand mechanism")
        if root_preload:
            if mechanism is None:raise ValueError("root-moment preload requires the shared source transmission")
            caps=[float(mechanism.settings["palm_transmission_boundary_nm"]),
                  *[float(root_preload["finite_motor_cap_nm"])]*3]
            mechanism.set_caps(caps)
            record["motor_cap_transition"]={"active_caps_nm":caps,"motor_input_state_reset":False,
                "source":root_preload["source_evidence"],"hardware_rating_claimed":False}
            # The old 0.9 Nm value is an observation reference from the old
            # grasp, not a rating for the requested finite 3.5 Nm motor path.
            stepper.settings={**stepper.settings,"measured_effort_abort_action":root_preload["legacy_effort_monitor_action"]}
            reference_force=float(root_preload["wrist_force_observation_n"])
            if not np.isfinite(reference_force) or not 0<reference_force<=20.:
                raise ValueError("nut grip retains the declared finite wrist observation range")
            loaded_phases=("key_probe_nut_contact","key_probe_nut_grip_hold",
                "key_probe_nut_index_unload","key_probe_nut_index_open",
                "key_probe_nut_rotation_visual_align","key_probe_nut_rotation_visual_refine",
                "key_probe_nut_rotation_axial_settle","key_probe_nut_rotation_turn","key_probe_nut_rotation_hold")
            ft.contact_force_limit_overrides_n.update({phase:reference_force for phase in loaded_phases})
            record["loaded_nut_force_observation"]={"limit_n":reference_force,"phases":list(loaded_phases),
                "source":root_preload["source_evidence"],"source_local_grip_peak_n":7.423843484462589,
                "commanded_force_reference_changed":False,"hardware_rating_claimed":False}
        body, target = target_from_palm(observation)
        locate_part_bounds(body)
        record.update(world_from_body_palm_five_dof=body.tolist(), target_world_from_hand=target.tolist(),
                      open_hand_goal_rad=open_goal.tolist(), finite_closing_goal_rad=close_goal.tolist())
        if prepare_geometry_only:
            runtime["nut_regrasp_geometry_check"] = check
            runtime["nut_regrasp_locate_visual_bounds"] = locate_part_bounds
            record.update(stage="LOCAL_DIAGNOSTIC_AVOIDANCE_GEOMETRY_PREPARED", geometry_only=True)
            return _json_ready(record)
        held_arm = np.asarray(ft.samples[-1]["active_targets_rad"][:7]).copy()
        hand_start = np.asarray(ft.samples[-1]["active_targets_rad"][7:]).copy()
        increment = float(dynamic["finger_maximum_speed_rad_s"]) * dt
        count = max(1, int(np.ceil(np.max(np.abs(open_goal - hand_start)) / increment)))
        record["stage"] = "EXPANDING_FINGERS_BEFORE_NUT_TRANSFER"
        save()
        world.play()
        for index in range(count):
            q = hand_start + (index + 1) / count * (open_goal - hand_start)
            advance("key_probe_nut_open", held_arm, q, nut_contact=start_from_nut_grip)
        for _ in range(round(.5 / dt)):
            advance("key_probe_nut_open_hold", held_arm, open_goal, nut_contact=start_from_nut_grip)
        world.pause()
        active, released_hand = hand_pose()
        if start_from_nut_grip:
            # The plug can settle when the supporting grip is released. The
            # held observation remains valid for opening, not for the following
            # transfer. Reuse the ordinary position/axis observer after release.
            record["pre_open_world_from_body_five_dof"]=body.tolist()
            record["stage"]="OBSERVING_POSITION_AND_AXIS_AFTER_ACTUAL_NUT_RELEASE"
            save()
            observation=observe_released_plug_from_rgbd(
                repository,stage,world,rep,released_hand,output/"after_open_palm")
            if not observation.get("position_and_axis_measured"):
                raise RuntimeError("position and axis must be observed after releasing the existing nut grip")
            record["post_open_position_axis_observation"]=observation
        body, target = target_from_palm(observation)
        locate_part_bounds(body)
        record.update(world_from_body_palm_five_dof=body.tolist(),target_world_from_hand=target.tolist())
        transfer_speed = float(config["nut_regrasp"].get("maximum_transfer_speed_m_s", .003))
        transfer_arm_speed = float(config["nut_regrasp"].get("maximum_transfer_arm_speed_rad_s", .15))
        if not all(np.isfinite(v) and v > 0 for v in (transfer_speed, transfer_arm_speed)):
            raise ValueError("nut transfer speed references must be finite and positive")
        states, ik = _plan_key_probe_descent(inputs, active, target, dt, transfer_speed)
        offset = np.asarray(ft.samples[-1]["active_targets_rad"][:7]) - active[:7]
        states += offset
        soft = np.asarray([MOVEIT_SOFT_ARM_BOUNDS_RAD[name] for name in control.ARM_JOINT_NAMES])
        peak = float(np.max(np.abs(np.diff(states, axis=0))) / dt)
        if peak > transfer_arm_speed or np.any(states < soft[:, 0]) or np.any(states > soft[:, 1]):
            raise RuntimeError("the nut transfer exceeds the existing joint speed or position bounds")
        for index, arm in enumerate(states):
            hit = check(arm, open_goal)
            if hit is not None:
                record["planned_collision"] = {"sample": index, **hit}
                raise RuntimeError(f"planned nut transfer collides: {hit}")
        record["motions"].append({"ik": ik, "arm_target_offset_rad": offset.tolist(),
                                  "maximum_arm_speed_rad_s": peak, "path_steps": len(states)})
        np.save(output / "open_hand_arm_path_rad.npy", states)
        record["stage"] = "MOVING_OPEN_HAND_TO_NUT"
        save()
        world.play()
        for arm in states:
            advance("key_probe_nut_transfer", arm, open_goal)
        held_arm = states[-1].copy()
        for _ in range(round(.5 / dt)):
            advance("key_probe_nut_transfer_hold", held_arm, open_goal)
        world.pause()
        _, hand = hand_pose()
        record["stage"] = "RECHECKING_PALM_BEFORE_NUT_CLOSURE"
        save()
        fresh = observe_released_plug_from_rgbd(repository, stage, world, rep, hand, output / "preclose_palm")
        record["preclose_palm_observation"] = fresh
        body, new_target = target_from_palm(fresh)
        locate_part_bounds(body)
        error = float(np.linalg.norm(new_target[:3, 3] - hand[:3, 3]))
        tolerance = .25 * min(row["body_clearance_m_at_first_nut_contact"] for row in geometry["first_contacts"])
        record.update(current_visual_hand_position_error_m=error,
                      visual_alignment_allowance_from_quarter_body_clearance_m=tolerance)
        if error > tolerance:
            raise RuntimeError("the fresh palm pose leaves insufficient clearance for the planned nut grip")
        record["stage"] = "CURRENT_OPEN_HAND_EFFORT_TARE"
        save()
        world.play()
        tare_rows = [];tare_encoder_rows=[]
        for _ in range(round(float(dynamic["effort_tare_duration_s"]) / dt)):
            advance("key_probe_nut_tare", held_arm, open_goal)
            tare_rows.append(np.asarray(stepper.latest[2])[7:].copy())
            tare_encoder_rows.append(np.asarray(stepper.latest[0]).copy())
        tare = np.mean(tare_rows, axis=0)
        root_observer=None
        if root_preload:
            from te_three_finger_wrench_observer import ThreeFingerWrenchObserver
            root_observer=ThreeFingerWrenchObserver(repository,inputs.robot_model,geometry_path,
                sensor_semantics="BASE_BRIDGE_EXTERNAL_MOMENT_ABOUT_O")
            root_observer.calibrate_free_space(tare_encoder_rows,np.asarray(tare_rows)[:,1:])
        contact = control.ParallelEffortContactController(
            open_goal, close_goal, effort_rise_nm=float(dynamic["contact_effort_rise_nm"]),
            position_error_rad=float(dynamic["contact_position_error_rad"]),
            velocity_absolute_max_rad_s=dynamic.get("contact_velocity_absolute_max_rad_s"),
            consecutive_samples=int(dynamic["contact_consecutive_samples"]),
            endpoint_timeout_samples=round(float(dynamic["contact_endpoint_timeout_s"]) / dt),
            hand_stiffness=float(dynamic["hand_stiffness"]), finger_order=(1, 2, 3))
        record["stage"] = "FINITE_PARALLEL_NUT_CONTACT"
        save()
        count = int(np.ceil(np.max(np.abs(close_goal - open_goal)) / increment)) + 3 * contact.endpoint_timeout_samples
        for _ in range(count):
            q = contact.step(stepper.latest[0][7:], stepper.latest[2][7:] - tare, increment,
                             measured_velocity=stepper.latest[1][7:])
            if contact.failed:
                raise RuntimeError(contact.failure_reason)
            advance("key_probe_nut_contact", held_arm, q, nut_contact=True)
            if contact.complete:
                break
        if not contact.complete:
            raise RuntimeError("finite nut closure did not confirm all three fingers")
        direction = np.sign(close_goal - open_goal)
        finite = contact.target + float(dynamic["preload_increment_rad"]) * direction
        lower, upper = np.minimum(open_goal, finite), np.maximum(open_goal, finite)
        desired = np.asarray(config["nut_regrasp"]["required_closing_joint_effort_nm"]
            if "required_closing_joint_effort_nm" in config["nut_regrasp"]
            else dynamic["required_closing_joint_effort_nm"])
        q = contact.target.copy()
        record["stage"] = "ESTABLISHING_CONFIGURED_THREE_FINGER_EFFORT_ON_NUT"
        record.update(new_grasp_effort_tare_nm=tare.tolist(), first_contact_hand_targets_rad=contact.target.tolist(),
                      finite_preload_bounds_rad=[lower.tolist(), upper.tolist()], effort_reference_nm=desired.tolist())
        save()
        if root_preload:
            desired=np.asarray(root_preload["targets_nm"],float)
            source_lower,source_upper=inputs.robot_model.joint_limit_vectors()
            lower[1:]=np.maximum(open_goal[1:],contact.target[1:]-float(root_preload["target_open_allowance_rad"]))
            upper[1:]=np.minimum(np.asarray(source_upper)[8:],contact.target[1:]+float(root_preload["target_close_allowance_rad"]))
            ramp=float(root_preload["ramp_duration_s"]);duration=float(root_preload["total_duration_s"])
            relaxation=dt/(float(root_preload["target_relaxation_time_constant_s"])+dt)
            stiffness=float(root_preload["position_stiffness_reference_nm_rad"])
            limit=min(increment,float(root_preload["maximum_target_speed_rad_s"])*dt)
            if desired.shape!=(3,) or not np.isfinite(desired).all() or np.any(desired<=0) or not 0<ramp<=duration:
                raise ValueError("finite root-moment preload references are required")
            for index in range(round(duration/dt)):
                encoder=np.asarray(stepper.latest[0]);raw=np.asarray(stepper.latest[2])[8:]
                gravity=root_observer._system(encoder)[2]
                measured=raw-root_observer.tare_reaction-(gravity-root_observer.tare_gravity)
                reference=control.minimum_jerk_blend(min(1.,(index+1)*dt/ramp))*desired
                q[1:]=np.clip(q[1:]+np.clip(relaxation*(reference-measured)/stiffness,-limit,limit),lower[1:],upper[1:])
                advance("key_probe_nut_grip_hold",held_arm,q,nut_contact=True)
                command_rows[-1]["base_bridge_external_moment_nm"]=measured.tolist()
                command_rows[-1]["base_bridge_reference_nm"]=reference.tolist()
            if np.any(measured<=float(dynamic["contact_effort_rise_nm"])):
                raise RuntimeError("a finger lost its load evidence during root-moment preload")
            record["root_moment_preload"]={**root_preload,"gravity_compensated":True,
                "tare_gravity_nm":root_observer.tare_gravity.tolist(),
                "final_measured_external_moment_nm":measured.tolist(),
                "relative_reference_error":((measured-desired)/desired).tolist(),
                "exact_target_reached_claimed":False,"motor_targets_frozen_for_next_rotation":True}
            record.update(finite_preload_bounds_rad=[lower.tolist(),upper.tolist()],effort_reference_nm=desired.tolist())
        else:
            for _ in range(round((float(dynamic["preload_duration_s"]) + float(dynamic["hold_duration_s"])) / dt)):
                measured = direction[1:] * (stepper.latest[2][8:] - tare[1:])
                q[1:] = np.clip(q[1:] + direction[1:] * np.clip(
                    effort_gain*(desired - measured) / float(dynamic["hand_stiffness"]), -increment, increment), lower[1:], upper[1:])
                advance("key_probe_nut_grip_hold", held_arm, q, nut_contact=True)
        record.update(completed=True, stage="NUT_GRIP_CONTROLLER_FINISHED_REQUIRES_PHYSICAL_EVALUATION",
                      final_hand_target_rad=q.tolist(), fixed_arm_target_rad=held_arm.tolist())
        # Carry only robot-model geometry checks into the immediately following
        # rotation stage. These closures have never read simulator object state.
        runtime["nut_regrasp_geometry_check"] = check
        runtime["nut_regrasp_locate_visual_bounds"] = locate_part_bounds
    except Exception as error:
        record.update(failure_stage=record["stage"], stage="STOPPED", failure_reason=str(error))
    finally:
        world.pause()
        record.update(last_step=int(stepper.step_index), executed_command_steps=len(command_rows),
                      outer_abort_reason=stepper.abort_reason)
        with gzip.open(output / "joint_ft_samples.json.gz", "wt", encoding="utf-8") as stream:
            json.dump(_json_ready(ft.samples[first_ft:]), stream, ensure_ascii=False, separators=(",", ":"))
        with gzip.open(output / "commands.json.gz", "wt", encoding="utf-8") as stream:
            json.dump(command_rows, stream, separators=(",", ":"))
        save()
    return _json_ready(record)
