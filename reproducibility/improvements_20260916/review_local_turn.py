"""Review an ended local turn without confusing it with full assembly success."""
import argparse
import ast
import gzip
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from trace_metadata import read_truth_sample, without_cyclic_gc


def review(directory):
    directory = Path(directory).resolve()
    local = json.loads((directory / 'source_stage_probe_result.json').read_text())
    archive = directory / 'truth_samples.msgpack.gz'
    index = json.loads(Path(str(archive) + '.index.json').read_text())
    assert index['blocks'][-1]['end'] == archive.stat().st_size
    rotation = json.loads((directory / 'rotation/nut_rotation_controller_result.json').read_text())
    controls = [json.loads(line) for line in (directory / 'rotation/nut_rotation_control_samples.jsonl').read_text().splitlines()]
    install = json.loads((directory / 'frozen_model_installation.json').read_text())
    socket = np.asarray(ast.literal_eval(install['pose_mass_velocity_after'][
        '/World/TEVisualHandoff/FixedReceptaclePose']['world_transform']), float).T
    endpoints = []
    for step in (rotation['first_step']-1, index['sample_count']-1):
        row = without_cyclic_gc(read_truth_sample, directory, step)
        body_p, nut_p = np.asarray(row['object_part_positions_m'])
        body_R, nut_R = Rotation.from_quat(np.asarray(row['object_part_orientations_wxyz'])[:, [1, 2, 3, 0]]).as_matrix()
        relative = body_R.T @ nut_R
        endpoints.append({'step': step, 'body_depth_mm': -float(socket[:3, 2] @ (body_p-socket[:3, 3]))*1000,
                          'body_lateral_mm': float(np.linalg.norm((socket[:3, :3].T @ (body_p-socket[:3, 3]))[:2]))*1000,
                          'nut_lateral_mm': float(np.linalg.norm((socket[:3, :3].T @ (nut_p-socket[:3, 3]))[:2]))*1000,
                          'nut_body_yaw_deg': math.degrees(math.atan2(relative[1, 0], relative[0, 0])),
                          'native_points_retained_at_endpoint': row['contacts']['all_provided_native_report_points_retained'],
                          'native_callback_count_at_endpoint': row['contacts']['physics_step_callback_count']})
    motors = {}
    with gzip.open(directory / 'hand_mechanism_samples.jsonl.gz', 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            m = motors.setdefault(row['joint'], {'samples': 0, 'max_abs_transmission_effort_nm': 0.,
                                                 'any_elastic_boundary_exceeded': False, 'any_drive_saturation': False})
            m['samples'] += 1
            m['max_abs_transmission_effort_nm'] = max(m['max_abs_transmission_effort_nm'], abs(row['transmission_effort']))
            m['any_elastic_boundary_exceeded'] |= bool(row['elastic_effort_boundary_exceeded'])
            m['any_drive_saturation'] |= bool(row['drive_saturation'])
    recipe = json.loads((directory / 'rotation/coaxial_recipe.json').read_text())
    checkpoints = []
    for angle in (20, 24, 25, 26, 27, 28, 29, 30):
        c = next((row for row in controls if abs(row['commanded_rotation_deg']) >= angle), None)
        if c is not None:
            checkpoints.append({'angle_deg': abs(c['commanded_rotation_deg']),
                                'xy_error_mm': math.hypot(*c['pivot_position_tracking_error_m'][:2])*1000})
    result = {'scope': 'ENDED_LOCAL_TURN_CONTROL_AND_PHYSICAL_ENDPOINT_REVIEW',
              'full_assembly_or_terminal_release_claimed': False, 'online_truth_used': False,
              'run': str(directory), 'physical_samples': index['sample_count'],
              'selected_planar_speed_mm_s': 1000*recipe['maximum_planar_speed_m_s'],
              'selected_axial_speed_mm_s': 1000*recipe['maximum_axial_speed_m_s'],
              'load_compensation_filter_s': recipe.get('contact_load_compensation_filter_time_constant_s', .05),
              'completed_controller_interval': rotation['completed'],
              'outer_abort': local.get('outer_abort'), 'error': local.get('error'),
              'normal_stop': rotation.get('early_regrasp'),
              'commanded_rotation_deg': abs(rotation['last_applied_rotation_command_deg']),
              'actual_nut_relative_rotation_deg': (endpoints[-1]['nut_body_yaw_deg']-endpoints[0]['nut_body_yaw_deg']+180)%360-180,
              'body_advance_mm': endpoints[-1]['body_depth_mm']-endpoints[0]['body_depth_mm'],
              'maximum_virtual_pivot_xy_error_mm': max(math.hypot(*row['pivot_position_tracking_error_m'][:2])*1000 for row in controls),
              'checkpoints': checkpoints, 'physical_endpoints': endpoints, 'motor_records': motors,
              'wall_timing': local['wall_timing'],
              'limitations': ['This cold-initialized local turn is not a continuous visual assembly episode.',
                              'Physical endpoint checks do not certify every intermediate source contact or key gap.',
                              'The full raw archive, video and controller records are retained for further review.']}
    (directory / 'local_turn_physical_review.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    print(json.dumps(review(args.directory), indent=2))
