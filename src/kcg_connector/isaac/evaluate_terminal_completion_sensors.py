"""Replay the after-motion decision using only recorded, then-available sensors.

This module is never imported by online control. It preserves the historical
runtime result and does not read simulator object/contact ground truth.
"""
import argparse
import json
from pathlib import Path
import subprocess

import yaml

from four_camera_online_completion import assess, released_observations


def review(directory, output_path=None):
    directory = Path(directory).resolve()
    repository = Path(__file__).resolve().parents[3]
    process = json.loads((directory.parent / 'process.json').read_text())
    if 'exit_code' not in process:
        raise ValueError('A finished run is required for an explicitly offline replay')
    transport = json.loads((directory / 'socket_transport/transport_and_observation.json').read_text())
    series = transport['continued_nut_strokes']
    release = series['terminal_release']
    rotations = [transport.get('nut_rotation'), transport.get('nut_rotation_after_index')]
    rotations.extend(item.get('rotation') for item in series.get('attempts', []))
    last = next(item for item in reversed(rotations) if item is not None)
    metadata = json.loads((directory / 'trace_metadata.json').read_text())
    contract = transport.get('online_completion_sampling_contract')
    if contract is None:
        # Compatibility with the preserved pre-fix run: recover the actual
        # declared camera period from its committed configuration, not from
        # observed gaps, which could hide missing measurements.
        argv = process['argv']
        config_path = Path(argv[argv.index('--body-assembly-collision-config') + 1])
        if config_path.is_absolute():
            config_path = config_path.relative_to(repository)
        def frozen(path):
            return subprocess.check_output(['git', 'show', f"{process['source_commit']}:{path}"],
                cwd=repository, text=True)
        config = yaml.safe_load(frozen(config_path.as_posix()))
        source = frozen('src/kcg_connector/isaac/four_camera_perception_session.py')
        if 'if age > .5:' not in source:
            raise ValueError('The historical maximum observation age must be explicitly known')
        contract = {
            'palm_period_s': float(config['perception']['palm_update_period_s']),
            'maximum_observation_age_s': .5,
            'decision_step': int(transport['last_step']),
            'decision_time_s': float(release['after_nut_release']['five_dof_observation']
                                     ['consumed_by_stage_at_physics_time_s']),
            'source': 'ORIGINAL_COMMITTED_CAMERA_SETTINGS_AND_RECORDED_ROBOT_CLOCK',
        }
    observations = released_observations(directory / 'four_camera_perception', release,
        decision_step=int(contract['decision_step']), decision_time_s=float(contract['decision_time_s']))
    decision = assess(transport['world_from_socket_wrist_visual'], observations, release,
        last['last_loaded_interface_wrench_n_nm'], last['settings']['visual_progress'],
        physics_dt_s=float(metadata['physics_dt_s']),
        maximum_lateral_error_m=float(last['settings']['measured_tracking']['maximum_lateral_error_m']),
        maximum_axis_error_deg=.10, palm_period_s=float(contract['palm_period_s']),
        maximum_observation_age_s=float(contract['maximum_observation_age_s']),
        decision_time_s=float(contract['decision_time_s']))
    result = {
        'scope': 'POSTRUN_REPLAY_OF_TERMINAL_DECISION_FROM_ALREADY_CONSUMED_ONLINE_SENSORS',
        'accepted': decision['online_seating_confirmed'],
        'historical_runtime_online_seating_confirmed': transport['online_completion']['online_seating_confirmed'],
        'historical_runtime_completion_record_modified': False,
        'motion_source_commit': process['source_commit'],
        'decision_code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository, text=True).strip(),
        'sampling_contract': contract, 'decision': decision,
        'object_or_contact_truth_used_for_replayed_decision': False,
        'physical_motion_reexecuted': False, 'new_images_synthesized_or_captured': False,
        'independent_physical_geometry_and_three_second_contact_reviews_still_required': True,
    }
    destination = Path(output_path) if output_path is not None else directory / 'online_completion_sensor_replay.json'
    destination.write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    print(json.dumps(review(parser.parse_args().directory), indent=2))
