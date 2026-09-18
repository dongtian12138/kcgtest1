"""Online seating evidence is distinct from a protection or regrasp stop."""
import numpy as np


def assess(socket, observations, release, last_loaded_wrench, settings, *, physics_dt_s,
           maximum_lateral_error_m, maximum_axis_error_deg):
    """Use ordinary visual poses and robot-side release/wrench observations.

    This is a controller judgment. Original CAD/contact/depth tolerances and
    the independent three-second zero-contact physical review remain separate.
    """
    socket = np.asarray(socket, float).reshape(4, 4)
    limits = np.asarray([physics_dt_s, maximum_lateral_error_m, maximum_axis_error_deg], float)
    if not np.isfinite(socket).all() or not np.isfinite(limits).all() or np.any(limits <= 0):
        raise ValueError('Finite positive declared sampling and visual alignment limits required')
    rows = []
    seen = set()
    for observation in observations:
        if not observation.get('position_and_axis_measured'):
            continue
        timestamp = float(observation['capture_physics_time_s'])
        if timestamp in seen:
            continue
        seen.add(timestamp)
        body = np.asarray(observation['world_from_plug_five_dof'], float).reshape(4, 4)
        if not np.isfinite(np.r_[body.ravel(), timestamp]).all():
            raise ValueError('Finite timestamped visual poses are required')
        depth = -float(socket[:3, 2] @ (body[:3, 3] - socket[:3, 3]))
        rows.append({'sample_time_s': timestamp, 'observed_depth_m': depth,
            'lateral_error_m': float(np.linalg.norm((np.eye(3) - np.outer(socket[:3, 2], socket[:3, 2]))
                                                  @ (body[:3, 3] - socket[:3, 3]))),
            'axis_error_deg': float(np.degrees(np.arccos(np.clip(-socket[:3, 2] @ body[:3, 2], -1., 1.))))})
    rows.sort(key=lambda row: row['sample_time_s'])
    depths = np.asarray([row['observed_depth_m'] for row in rows])
    wrench = np.asarray(last_loaded_wrench, float)
    if wrench.shape != (6,) or not np.isfinite(wrench).all():
        raise ValueError('Finite last loaded robot wrist wrench required')
    enough = len(rows) >= int(settings['seating_confirmations'])
    visual = bool(enough and np.all(abs(depths - float(settings['nominal_seated_depth_m']))
                                    <= float(settings['visual_seating_tolerance_m'])))
    stable = bool(enough and np.ptp(depths) <= float(settings['stable_depth_tolerance_m']))
    observed_interval = rows[-1]['sample_time_s'] - rows[0]['sample_time_s'] if enough else 0.
    actual_hold_s = (int(release.get('support_hold_last_step', 0))
                     - int(release.get('opening_last_step', 0))) * float(physics_dt_s)
    released = bool(release.get('completed') and release.get('release_readiness', {}).get('ready')
        and release.get('release_readiness', {}).get('unloaded')
        and actual_hold_s >= 3. - physics_dt_s / 2
        and observed_interval >= 3. and release.get('outer_abort_reason') is None)
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
        'robot_observed_open_unloaded_hold_completed': released,
        'preceding_loaded_torque_observed': loaded,
        'last_loaded_robot_wrist_wrench_n_nm': wrench.tolist(),
        'online_object_or_contact_truth_used': False,
        'protection_or_regrasp_stop_alone_is_completion': False,
        'physical_assembly_acceptance': 'REQUIRES_ORIGINAL_INDEPENDENT_POSTRUN_REVIEW'}
