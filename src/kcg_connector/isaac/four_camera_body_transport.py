"""Short Body carry with one key anchor and delayed mounted-camera feedback."""
from pathlib import Path
from time import perf_counter
import json
import math
import numpy as np


def run(repository, runtime, stepper, grasp_result, dynamic, anchor, output):
    import fcl
    import yaml
    from scipy.spatial.transform import Rotation
    from four_camera_perception_session import FourCameraPerceptionSession
    from short_body_motion import plan_body_path
    from kcg_connector.te_rgbd_pose_provider import estimate_receptacle_key_from_depth
    from te_foundationpose_handoff_plan import FullRobotCollisionScene, _cylinder_from_mesh
    from te_foundationpose_handoff_runtime import (
        _check_held_plug_path, _execute_held_plug_path, _json_ready,
        MOVEIT_SOFT_ARM_BOUNDS_RAD, control)

    root, output = Path(repository), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    world, inputs, ft = runtime['world'], runtime['inputs'], runtime['nail_body_ft_auditor']
    config = yaml.safe_load((root / runtime['body_assembly_control_config']).read_text())
    dt = float(dynamic['physics_dt_s'])
    speed = float(config['motion']['maximum_transport_joint_speed_rad_s'])
    probe = {'authorization': {'simulation_only': True, 'hardware_authorized': False},
             'motion': {'maximum_transport_joint_speed_rad_s': speed}}
    session = FourCameraPerceptionSession(root, runtime, stepper, anchor)
    record = {'stage': 'WAITING_FOR_GLOBAL1_SOCKET', 'completed': False,
        'simulation_only': True, 'hardware_authorized': False,
        'online_object_or_contact_truth_used': False,
        'key_observation_event_count': 1, 'body_key_reobservations_after_memory': 0,
        'hand_from_body_visual_memory': anchor['hand_from_body_visual_memory'],
        'contact_motion_commanded': False, 'motions': [],
        'target_face_gap_m': .050, 'wrist_socket_observation_executed': False,
        'transport_macro_legs': ['PICKUP_TO_FIXED_GLOBAL2', 'GLOBAL2_TO_SOCKET_WITH_LOCAL_WRIST_VIEW'],
        'physical_carry_and_memory_validity': 'REQUIRES_POSTRUN_EVALUATION'}

    def save():
        (output / 'transport_and_observation.json').write_text(json.dumps(_json_ready(record), indent=2) + '\n')

    def tracked(states, request_new=True):
        for arm in states:
            session.service(request_new=request_new)
            yield arm

    def hold(seconds, phase, *, request_new=False):
        steps = max(1, math.ceil(max(0., seconds) / dt))
        arm = np.asarray(ft.samples[-1]['active_targets_rad'][:7])
        result = _execute_held_plug_path(world, stepper, ft, grasp_result, dynamic,
            tracked((arm for _ in range(steps)), request_new), probe, phase=phase)
        if not result['completed']:
            raise RuntimeError(f'{phase}: {result["abort_reason"]}')
        session.service(request_new=False)
        return result

    def refresh():
        if session.pending is not None:
            hold(session.pending.available_time_s - float(world.current_time) + dt,
                 'key_probe_pending_palm_hold')
        session.next_request = float(world.current_time)
        session.service()
        hold(session.pending.available_time_s - float(world.current_time) + dt,
             'key_probe_fresh_palm_hold')

    try:
        if stepper.abort_reason is not None or grasp_result.get('failure_reason'):
            raise RuntimeError('The original grasp controller did not complete safely')
        job = runtime.get('global1_socket_job')
        if job is None:
            raise RuntimeError('No socket localization from the current initial Global1 image')
        global_observation = job.result()
        if float(world.current_time) < global_observation['available_time_s']:
            record['global1_delay_hold'] = hold(global_observation['available_time_s'] - float(world.current_time),
                'key_probe_global1_latency_hold')
        record['global_socket_result'] = str(Path(runtime['output_directory']) / 'global1_socket/camera_and_estimate.json')
        record['global1_consumption_time_s'] = float(world.current_time)
        socket = np.asarray(global_observation['world_from_socket_coarse'])
        record['world_from_socket_visual'] = socket.tolist()
        refresh()

        geometry_started = perf_counter()
        scene = FullRobotCollisionScene(inputs)
        xy = np.asarray(inputs.table_xy_bounds_m)
        table_size = np.r_[xy[:, 1] - xy[:, 0], 1.]
        table_center = np.r_[np.mean(xy, axis=1), inputs.table_top_z_m - .5]
        fixture = runtime['body_assembly_scene']['fixture']
        obstacles = {
            'table': fcl.CollisionObject(fcl.Box(*table_size), fcl.Transform(table_center)),
            'fixture': fcl.CollisionObject(fcl.Box(*fixture['size_m']),
                                         fcl.Transform(np.asarray(fixture['center_world_m'])))}
        obstacles['receptacle'], _ = _cylinder_from_mesh(Path(global_observation['receptacle_cad_mm']), .001, socket)
        vertices = np.asarray(inputs.object_contract.model.mesh.vertices_m)
        bounds = {'radius_m': float(np.max(np.linalg.norm(vertices[:, :2], axis=1))),
                  'z_min_m': float(np.min(vertices[:, 2])), 'z_max_m': float(np.max(vertices[:, 2]))}
        record['geometry_preparation_wall_s'] = perf_counter() - geometry_started
        hold(record['geometry_preparation_wall_s'], 'key_probe_geometry_preparation_hold')

        def move(target_body, phase):
            refresh()
            record['stage'] = phase
            save()
            planning_started = perf_counter()
            active = np.asarray(stepper.latest[0])
            memory = session.memory.hand_from_body()
            states, plan = plan_body_path(inputs, active, memory, target_body, dt, speed)
            offset = np.asarray(ft.samples[-1]['active_targets_rad'][:7]) - active[:7]
            states += offset
            soft = np.asarray([MOVEIT_SOFT_ARM_BOUNDS_RAD[n] for n in control.ARM_JOINT_NAMES])
            if np.any(states < soft[:, 0]) or np.any(states > soft[:, 1]):
                raise RuntimeError('Loaded command offset crossed an original soft joint limit')
            # Discrete source-geometry check at <= .01 rad in every arm joint.
            # The 960Hz physical controller still checks its original force,
            # speed and position limits at every step; no continuous-proof
            # claim is made for this finite geometric sampling.
            selected = [0]
            for index in range(1, len(states)):
                if np.max(np.abs(states[index] - states[selected[-1]])) >= .009:
                    selected.append(index)
            if selected[-1] != len(states) - 1:
                selected.append(len(states) - 1)
            spatial = states[selected]
            spacing = float(np.max(np.abs(np.diff(spatial, axis=0))))
            if spacing > .01:
                raise RuntimeError('Source-geometry sampling exceeded its declared joint spacing')
            _, check = _check_held_plug_path(scene, spatial, active[7:], obstacles, memory, bounds, 1., speed)
            check.update(checked_states=len(spatial), maximum_joint_spacing_rad=spacing,
                         continuous_collision_guarantee=False,
                         sampled_source_duration_is_not_execution_duration=True)
            if check['first_collision'] is not None:
                raise RuntimeError(f'Full hand or carried Body path collision: {check["first_collision"]}')
            latency = perf_counter() - planning_started
            item = {'phase': phase, 'plan': plan, 'collision': check,
                    'loaded_target_offset_rad': offset.tolist(), 'planning_wall_s': latency}
            record['motions'].append(item)
            np.save(output / f'{phase}_arm_path_rad.npy', states)
            save()
            item['planning_delay_hold'] = hold(latency, 'key_probe_planning_latency_hold', request_new=True)
            states = np.vstack((states, np.repeat(states[-1:], round(dynamic['hold_duration_s'] / dt), axis=0)))
            item['first_step'] = int(stepper.step_index)
            item['execution'] = _execute_held_plug_path(world, stepper, ft, grasp_result, dynamic,
                tracked(states), probe, phase=phase)
            item['last_step'] = int(stepper.step_index)
            if not item['execution']['completed']:
                raise RuntimeError(f'Carry stopped by original limits: {item["execution"]["abort_reason"]}')
            refresh()
            item['arrival_body_estimate'] = session.body().tolist()
            save()

        # Global1 supplies center/axis only. Choose a convenient free-space
        # yaw for the wrist view; it is explicitly not a measured key alignment.
        z = -socket[:3, 2]
        y = np.array([0., 1., 0.]); y -= z * float(z @ y); y /= np.linalg.norm(y)
        side = np.eye(4); side[:3, :3] = np.column_stack((np.cross(y, z), y, z))
        side[:3, 3] = socket[:3, 3] + .05 * socket[:3, 2] + np.array([-.035, 0., 0.])
        record['preobservation_axial_yaw_source'] = 'CHOSEN_FREE_SPACE_POSTURE_NOT_KEYWAY_MEASUREMENT'
        move(side, 'key_probe_body_short_carry')

        record['stage'] = 'FIXED_MOUNT_WRIST_SLOT_OBSERVATION'; save()
        frame = session.capture('wrist', output / 'wrist_socket')
        started = perf_counter()
        measurement = estimate_receptacle_key_from_depth(frame['depth'],
            np.isfinite(frame['depth']) & (frame['depth'] > 0), frame['intrinsics'],
            frame['camera'], np.linalg.inv(frame['camera']) @ socket)
        latency = session.sensor_delay + perf_counter() - started
        wrist = {k: v for k, v in frame.items() if k != 'depth'}
        wrist.update(measurement=measurement, available_time_s=frame['sample_time_s'] + latency,
                     online_object_or_contact_truth_used=False)
        (output / 'wrist_socket/camera_and_estimate.json').write_text(json.dumps(_json_ready(wrist), indent=2) + '\n')
        record['wrist_observation_delay_hold'] = hold(latency, 'key_probe_wrist_latency_hold', request_new=True)
        record['wrist_socket_observation_executed'] = True
        record['wrist_socket_result'] = str(output / 'wrist_socket/camera_and_estimate.json')
        record['wrist_socket_key_measured'] = bool(measurement['key_direction_measured'])
        if not measurement['key_direction_measured']:
            raise RuntimeError(measurement.get('reason', 'Fixed wrist view did not resolve the slot pattern'))
        socket = np.asarray(measurement['world_from_receptacle_row_major']).reshape(4, 4)
        record['world_from_socket_wrist_visual'] = socket.tolist()
        obstacles['receptacle'], _ = _cylinder_from_mesh(Path(global_observation['receptacle_cad_mm']), .001, socket)
        above = socket.copy()
        above[:3, :3] = socket[:3, :3] @ Rotation.from_euler('y', 180, degrees=True).as_matrix()
        above[:3, 3] += .05 * socket[:3, 2]
        move(above, 'key_probe_body_socket_centering')
        record.update(stage='ABOVE_SOCKET_FROM_SINGLE_KEY_AND_DELAYED_PALM', completed=True,
            completed_scope='TRANSPORT_AND_WRIST_OBSERVATION_ONLY_NOT_ASSEMBLY',
            arrival_world_from_body_from_memory=session.body().tolist(),
            current_hand_from_body_visual_memory=session.memory.hand_from_body().tolist(),
            key_memory=session.memory.report())
        if runtime.get('body_key_entry_requested'):
            raise RuntimeError('Four-camera contact continuation is not connected in this transport-only revision')
    except Exception as error:
        record.update(failure_stage=record['stage'], stage='STOPPED', completed=False, failure_reason=str(error))
        raise
    finally:
        world.pause()
        record['last_step'] = int(stepper.step_index)
        record['key_memory'] = session.memory.report()
        save()
    return _json_ready(record)
