"""Postrun decomposition of commanded and actual fourth-stroke tracking."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from kcg_connector.grasp.carts_v2.models import load_v2_inputs

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--window', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
window = json.loads(args.window.read_text())
source = Path(window['source_run'])
index = json.loads((source / 'truth_samples.msgpack.gz.index.json').read_text())
assert (source / 'motion_timing.json').is_file()
assert index['blocks'][-1]['end'] == (source / 'truth_samples.msgpack.gz').stat().st_size
stage = source / 'socket_transport/nut_rotation_continued_02'
record = json.loads((stage / 'nut_rotation_controller_result.json').read_text())
control = [json.loads(line) for line in (stage / 'nut_rotation_control_samples.jsonl').read_text().splitlines()]
sensors = {row['step']: row for row in window['samples']}
inputs = load_v2_inputs(root, config_path=root / 'src/kcg_connector/config/visual_assembly_v1_body.yaml',
                        object_id='te_deutsch_d38999_26fj35pn_step')
model = inputs.robot_model
hand_from_pivot = np.asarray(record['hand_from_virtual_nut_axis_frame'])
initial = np.asarray(record['initial_virtual_nut_frame'])[:3, 3]
dt = 1 / 960
kp, kd = 2500., 160.
rows = []
previous_target = None
for c in control:
    step = c['step']
    before, after = sensors[step - 1], sensors[step]
    q = np.asarray(before['active_positions_rad'])
    qdot = np.asarray(before['active_velocities_rad_s'])[:7]
    target = np.asarray(c['nominal_arm_target_rad'])
    assert np.array_equal(target, np.asarray(after['active_targets_rad'])[:7])
    H = np.asarray(model.forward_kinematics(q, enforce_limits=False)['handbase_link'])
    pivot = H @ hand_from_pivot
    assert np.max(np.abs(pivot[:3, 3] - c['encoder_pivot_world_m'])) < 1e-10
    nominal_q = q.copy()
    nominal_q[:7] = target
    nominal = np.asarray(model.forward_kinematics(nominal_q, enforce_limits=False)['handbase_link']) @ hand_from_pivot
    J = np.asarray(model.geometric_jacobian('handbase_link', tuple(q)))[:, :7]
    x, y, z = pivot[:3, 3] - H[:3, 3]
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    P = J[:3] - skew @ J[3:]
    actual_velocity = P @ qdot
    H_after = np.eye(4)
    H_after[:3, :3] = np.asarray(after['handbase_rotation_world_row_major']).reshape(3, 3)
    H_after[:3, 3] = after['handbase_position_world_m']
    displacement_velocity = ((H_after @ hand_from_pivot)[:3, 3] - pivot[:3, 3]) / dt
    expected_velocity = np.asarray(c['requested_interface_velocity_world_m_rad_s'])[:3]
    drive = np.asarray(after['arm_control']['drive_target_rad'])
    gravity = np.asarray(after['arm_control']['gravity_compensation_nm'])
    # Payload fraction is explicitly zero in run_coaxial_nut_interval.
    load = kp * (drive - target) - gravity
    velocity_reference = np.zeros(7) if previous_target is None else (target - previous_target) / dt
    error = np.asarray(c['pivot_position_tracking_error_m'])
    r = {'step': step, 'time_s': c['elapsed_s'], 'angle_deg': abs(c['commanded_rotation_deg']),
         'actual_pivot_error_xy_mm': 1000 * float(np.linalg.norm(error[:2])),
         'nominal_pivot_error_xy_mm': 1000 * float(np.linalg.norm((nominal[:3, 3] - initial)[:2])),
         'nominal_minus_actual_xy_mm': (1000 * (nominal[:3, 3] - pivot[:3, 3])[:2]).tolist(),
         'requested_xy_velocity_mm_s': (1000 * expected_velocity[:2]).tolist(),
         'actual_xy_velocity_mm_s': (1000 * actual_velocity[:2]).tolist(),
         'actual_pose_difference_xy_velocity_mm_s': (1000 * displacement_velocity[:2]).tolist(),
         'joint_velocity_tracking_error_rad_s': (velocity_reference - qdot).tolist(),
         'joint_position_tracking_error_rad': (target - q[:7]).tolist(),
         'measured_load_compensation_nm': load.tolist(),
         'position_pd_nm': (kp * (target - q[:7])).tolist(),
         'velocity_pd_nm': (kd * (velocity_reference - qdot)).tolist(),
         'filtered_interface_wrench_n_nm': c['interface_wrench'],
         'command_solver_residual_m_s': c['bounded_velocity_task_residual'][:3],
         'saturated': after['arm_control']['saturated']}
    if rows:
        filtered = np.asarray(c['interface_wrench'])
        prior = np.asarray(rows[-1]['filtered_interface_wrench_n_nm'])
        r['pre_filter_interface_wrench_inferred_n_nm'] = (filtered + .05 / dt * (filtered - prior)).tolist()
    rows.append(r)
    previous_target = target

selected = []
for angle in (0, 10, 20, 22, 24, 25, 25.6, 26, 27, 28, 29.6):
    selected.append(next(row for row in rows if row['angle_deg'] >= angle))
result = {'scope': 'POSTRUN_ROBOT_ENCODER_COMMAND_AND_FORCE_DECOMPOSITION',
          'source_run': str(source), 'physics_executed': False, 'online_truth_used': False,
          'sample_count': len(rows), 'control_rows_pair_with_preceding_sensed_step': True,
          'all_encoder_pivots_reconstructed_within_1e_minus_10_m': True,
          'all_arm_targets_equal_recorded_commands': True,
          'any_arm_saturated': any(row['saturated'] for row in rows),
          'maximum_solver_residual_mm_s': max(1000 * np.linalg.norm(row['command_solver_residual_m_s']) for row in rows),
          'checkpoints': selected,
          'limitations': ['The inferred unfiltered wrench algebraically inverts the recorded fixed 50 ms first-order filter; it is not an independently measured contact wrench.',
                          'PhysX constraint solving can make reported velocities differ from position differences; both quantities are retained and actual movement conclusions use the latter.',
                          'Joint effort readback is projected joint reaction, not motor torque, and is not used as an actuation measurement.',
                          'The first velocity reference is excluded from transient conclusions because its previous nominal target is outside the control file.',
                          'This separates command and execution tracking; a changed-control physical comparison is still required for causal validation.']}
args.output.mkdir(parents=True, exist_ok=True)
(args.output / 'fourth_robot_tracking_decomposition.json').write_text(json.dumps(result, indent=2) + '\n')
(args.output / 'fourth_robot_tracking_series.json').write_text(json.dumps(rows, separators=(',', ':')) + '\n')
print(json.dumps(result, indent=2))
