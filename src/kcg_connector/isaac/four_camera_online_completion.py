"""Online seating evidence is distinct from a protection or regrasp stop."""
import numpy as np


def released_observations(session_directory, release, *, decision_step, decision_time_s):
    """Read every palm frame already consumed in the actual released hold.

    The session flushes these ordinary image observations as they arrive. This
    function runs after motion, reads no simulator state, and advances no time.
    The same inputs can therefore replay the terminal decision after a run.
    """
    import json
    from pathlib import Path

    directory = Path(session_directory).resolve()
    first = int(release['opening_last_step']) - 1
    last = int(release['support_hold_last_step']) - 1
    observations = []
    for line in (directory / 'events.jsonl').read_text().splitlines():
        event = json.loads(line)
        if event['event'] != 'PALM_MEASUREMENT_CONSUMED':
            continue
        if not first <= int(event['sample_step']) <= last:
            continue
        if (int(event['consumed_step']) > decision_step
                or float(event['consumed_time_s']) > decision_time_s):
            continue
        folder = Path(event['observation_directory']).resolve()
        folder.relative_to(directory)
        frame = json.loads((folder / 'observation.json').read_text())
        if (frame['sample_step'] != event['sample_step']
                or frame['sample_time_s'] != event['sample_time_s']
                or frame['available_time_s'] != event['available_time_s']
                or frame['online_object_or_contact_truth_used'] is not False):
            raise ValueError('Consumed palm event and saved image observation differ')
        observations.append({
            'position_and_axis_measured': True,
            'capture_physics_time_s': frame['sample_time_s'],
            'sample_step': event['sample_step'],
            'available_physics_time_s': event['available_time_s'],
            'consumed_physics_time_s': event['consumed_time_s'],
            'consumed_step': event['consumed_step'],
            'world_from_plug_five_dof': frame['measurement']['world_from_plug_five_dof'],
            'rgbd_directory': str(folder),
            'online_object_or_contact_truth_used': False,
        })
    return observations


def assess(socket, observations, release, last_loaded_wrench, settings, *, physics_dt_s,
           maximum_lateral_error_m, maximum_axis_error_deg, palm_period_s,
           maximum_observation_age_s, decision_time_s):
    """Use ordinary visual poses and robot-side release/wrench observations.

    This is a controller judgment. Original CAD/contact/depth tolerances and
    the independent three-second zero-contact physical review remain separate.
    """
    socket = np.asarray(socket, float).reshape(4, 4)
    limits = np.asarray([physics_dt_s, maximum_lateral_error_m, maximum_axis_error_deg,
                         palm_period_s, maximum_observation_age_s], float)
    if not np.isfinite(socket).all() or not np.isfinite(limits).all() or np.any(limits <= 0):
        raise ValueError('Finite positive declared sampling and visual alignment limits required')
    rows = []
    seen = set()
    first = int(release.get('opening_last_step', 0)) - 1
    last = int(release.get('support_hold_last_step', 0)) - 1
    if not np.isfinite(decision_time_s):
        raise ValueError('Finite current decision time is required')
    for observation in observations:
        if not observation.get('position_and_axis_measured'):
            continue
        timestamp = float(observation['capture_physics_time_s'])
        sample_step = int(observation['sample_step'])
        if not first <= sample_step <= last:
            continue
        available = float(observation['available_physics_time_s'])
        consumed = float(observation['consumed_physics_time_s'])
        consumed_step = int(observation['consumed_step'])
        if not (timestamp <= available <= consumed <= decision_time_s
                and consumed - timestamp <= maximum_observation_age_s
                and sample_step < consumed_step <= last + 1):
            raise ValueError('Released visual evidence was not causally available')
        if timestamp in seen:
            continue
        seen.add(timestamp)
        body = np.asarray(observation['world_from_plug_five_dof'], float).reshape(4, 4)
        if not np.isfinite(np.r_[body.ravel(), timestamp]).all():
            raise ValueError('Finite timestamped visual poses are required')
        depth = -float(socket[:3, 2] @ (body[:3, 3] - socket[:3, 3]))
        rows.append({'sample_time_s': timestamp, 'sample_step': sample_step,
            'available_time_s': available, 'consumed_time_s': consumed,
            'rgbd_directory': observation.get('rgbd_directory'), 'observed_depth_m': depth,
            'lateral_error_m': float(np.linalg.norm((np.eye(3) - np.outer(socket[:3, 2], socket[:3, 2]))
                                                  @ (body[:3, 3] - socket[:3, 3]))),
            'axis_error_deg': float(np.degrees(np.arccos(np.clip(-socket[:3, 2] @ body[:3, 2], -1., 1.))))})
    rows.sort(key=lambda row: row['sample_time_s'])
    depths = np.asarray([row['observed_depth_m'] for row in rows])
    wrench = np.asarray(last_loaded_wrench, float)
    if wrench.shape != (6,) or not np.isfinite(wrench).all():
        raise ValueError('Finite last loaded robot wrist wrench required')
    minimum_confirmations = int(settings['seating_confirmations'])
    if minimum_confirmations < 2:
        raise ValueError('At least two distinct visual confirmations are required')
    enough = len(rows) >= minimum_confirmations
    visual = bool(enough and np.all(abs(depths - float(settings['nominal_seated_depth_m']))
                                    <= float(settings['visual_seating_tolerance_m'])))
    stable = bool(enough and np.ptp(depths) <= float(settings['stable_depth_tolerance_m']))
    # Image sampling and inference delay mean that captures cannot span an
    # entire exactly-three-second hold. Check coverage using the existing
    # camera period and maximum observation age, and test ALL released frames.
    # A loaded frame before opening must not masquerade as released evidence.
    steps = np.asarray([row['sample_step'] for row in rows], dtype=np.int64)
    first_gap = (int(steps[0]) - first) * physics_dt_s if enough else None
    last_gap = (last - int(steps[-1])) * physics_dt_s if enough else None
    maximum_gap = float(np.max(np.diff(steps))) * physics_dt_s if enough else None
    # The existing session has one frame in flight: it requests again after
    # both the nominal period and the preceding result's measured latency.
    expected_gaps = np.asarray([max(palm_period_s, row['available_time_s'] - row['sample_time_s'])
                                + 2 * physics_dt_s for row in rows[:-1]])
    coverage = bool(enough and np.all(np.diff(steps) > 0)
        and first_gap <= maximum_observation_age_s
        and last_gap <= maximum_observation_age_s
        and np.all(np.diff(steps) * physics_dt_s <= expected_gaps))
    actual_hold_s = (int(release.get('support_hold_last_step', 0))
                     - int(release.get('opening_last_step', 0))) * float(physics_dt_s)
    released = bool(release.get('completed') and release.get('release_readiness', {}).get('ready')
        and release.get('release_readiness', {}).get('unloaded')
        and actual_hold_s >= 3. - physics_dt_s / 2
        and coverage and release.get('outer_abort_reason') is None)
    loaded = abs(float(wrench[5])) >= float(settings['seating_minimum_torque_nm'])
    aligned = bool(enough and all(row['lateral_error_m'] <= maximum_lateral_error_m
        and row['axis_error_deg'] <= maximum_axis_error_deg for row in rows))
    confirmed = visual and stable and aligned and released and loaded
    return {'status': 'ONLINE_SEATING_CONFIRMED' if confirmed else 'ONLINE_SEATING_NOT_CONFIRMED',
        'online_seating_confirmed': confirmed, 'observations': rows,
        'within_original_visual_depth_tolerance': visual,
        'within_original_visual_depth_stability_tolerance': stable,
        'within_declared_visual_axis_and_lateral_tolerances': aligned,
        'actual_robot_open_hold_s': actual_hold_s,
        'visual_stability_scope': 'EVERY_CONSUMED_PALM_FRAME_DURING_COMPLETED_OPEN_HOLD',
        'released_window_coverage': {'complete': coverage,
            'pose_step_interval_inclusive': [first, last],
            'sample_count': len(rows), 'first_sample_gap_s': first_gap,
            'last_sample_age_at_hold_end_s': last_gap, 'maximum_sample_gap_s': maximum_gap,
            'gap_admission': 'DECLARED_PERIOD_OR_PRECEDING_RECORDED_PIPELINE_LATENCY_PLUS_TWO_PHYSICS_STEPS',
            'maximum_admitted_sample_gap_s': float(expected_gaps.max()) if len(expected_gaps) else None,
            'declared_palm_period_s': float(palm_period_s),
            'existing_maximum_observation_age_s': float(maximum_observation_age_s)},
        'observed_released_depth_range_m': float(np.ptp(depths)) if len(depths) else None,
        'robot_observed_open_unloaded_hold_completed': released,
        'preceding_loaded_torque_observed': loaded,
        'last_loaded_robot_wrist_wrench_n_nm': wrench.tolist(),
        'online_object_or_contact_truth_used': False,
        'protection_or_regrasp_stop_alone_is_completion': False,
        'physical_assembly_acceptance': 'REQUIRES_ORIGINAL_INDEPENDENT_POSTRUN_REVIEW'}
