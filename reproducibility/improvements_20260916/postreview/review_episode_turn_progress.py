"""Postrun measured turn/depth checkpoints; never an online control input."""
import argparse
import ast
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from trace_metadata import read_truth_sample, without_cyclic_gc


def yaw(matrix):
    return float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))


def difference(end, start):
    return float((end - start + 180.0) % 360.0 - 180.0)


def review(directory):
    directory = Path(directory).resolve()
    archive = directory / 'truth_samples.msgpack.gz'
    index = json.loads(Path(str(archive) + '.index.json').read_text())
    if index['blocks'][-1]['end'] != archive.stat().st_size:
        raise ValueError('Only a sealed and unchanged archive may be reviewed')
    installation = json.loads((directory / 'frozen_model_installation.json').read_text())
    socket = np.asarray(ast.literal_eval(installation['pose_mass_velocity_after'][
        '/World/TEVisualHandoff/FixedReceptaclePose']['world_transform']), float).T

    def state(step):
        row = without_cyclic_gc(read_truth_sample, directory, int(step))
        position = np.asarray(row['object_part_positions_m'])
        body, nut = Rotation.from_quat(np.asarray(
            row['object_part_orientations_wxyz'])[:, [1, 2, 3, 0]]).as_matrix()
        pose = row['native_robot_link_pose_audit']['poses']['handbase_link']
        hand = Rotation.from_quat(np.asarray(
            pose['orientation_world_wxyz'])[[1, 2, 3, 0]]).as_matrix()
        return {
            'step': int(step), 'phase': row['phase'],
            'body_depth_mm': -float(socket[:3, 2] @ (position[0] - socket[:3, 3])) * 1000,
            'body_tilt_deg': float(np.degrees(np.arccos(np.clip(
                -socket[:3, 2] @ body[:, 2], -1.0, 1.0)))),
            'nut_body_yaw_deg': yaw(body.T @ nut),
            'hand_nut_yaw_deg': yaw(nut.T @ hand),
            'nut_body_axial_offset_mm': float((body.T @ (position[1] - position[0]))[2] * 1000),
        }

    records = []
    for path in (directory / 'socket_transport').glob('nut_rotation*/nut_rotation_controller_result.json'):
        controller = json.loads(path.read_text())
        if 'last_step' not in controller:
            continue
        records.append((int(controller['first_step']), path, controller))
    intervals = []
    for first_step, path, controller in sorted(records):
        before = state(first_step - 1)
        end = state(min(int(controller['last_step']) - 1, index['sample_count'] - 1))
        command = controller.get('last_applied_rotation_command_deg')
        interval = {
            'stage_directory': str(path.parent.relative_to(directory)),
            'controller_completed': controller.get('completed'),
            'normal_stop_reason': controller.get('normal_stop_reason'),
            'failure_reason': controller.get('failure_reason'),
            'last_applied_wrist_command_deg': abs(float(command)) if command is not None else None,
            'before': before, 'end': end,
            'actual_nut_body_rotation_deg': difference(end['nut_body_yaw_deg'], before['nut_body_yaw_deg']),
            'hand_nut_yaw_change_deg': difference(end['hand_nut_yaw_deg'], before['hand_nut_yaw_deg']),
            'body_depth_increase_mm': end['body_depth_mm'] - before['body_depth_mm'],
        }
        intervals.append(interval)
    result = {
        'scope': 'POSTRUN_SAME_EPISODE_TURN_ENDPOINT_CHECKPOINTS',
        'online_control_used': False, 'source_archive_sample_count': index['sample_count'],
        'endpoint_yaw_differences_use_nearest_signed_angle': True,
        'full_stream_contact_and_key_reviews_still_required': True,
        'complete_assembly_success_claimed': False,
        'turns': intervals, 'final_archived_state': state(index['sample_count'] - 1),
    }
    (directory / 'turn_progress_physical_review.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    print(json.dumps(review(args.directory), indent=2))
