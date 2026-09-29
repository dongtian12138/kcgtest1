"""Audit recorded robot return decisions after an ended full assembly episode."""
import ast
import gzip
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text())


def review(run):
    run = Path(run).resolve()
    process = read(run.parent / 'process.json')
    if 'exit_code' not in process:
        raise ValueError('Only an ended full episode may be audited')
    tree = ast.parse((ROOT / 'src/kcg_connector/isaac/te_foundationpose_handoff_runtime.py').read_text())
    bounds = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'MOVEIT_SOFT_ARM_BOUNDS_RAD' for t in n.targets))
    limits = np.asarray([bounds[f'iiwa_joint_{i}'] for i in range(1, 8)])
    install = read(run / 'frozen_model_installation.json')
    socket = np.asarray(ast.literal_eval(install['pose_mass_velocity_after'][
        '/World/TEVisualHandoff/FixedReceptaclePose']['world_transform']), float).T
    stages = run / 'socket_transport'
    files = [p for p in stages.glob('nut_reindex*/nut_reindex_controller_result.json')]
    records = sorted(((p, read(p)) for p in files), key=lambda x: x[1]['first_step'])
    rows = []
    for path, record in records:
        selection = record.get('phase_equivalent_return_selection', {})
        angle = record.get('selected_return_angle_deg')
        selected = [c for c in selection.get('candidates', [])
                    if c.get('feasible') is True and c.get('return_angle_deg') == angle]
        with gzip.open(path.parent / 'joint_ft_samples.json.gz', 'rt') as f:
            samples = json.load(f)
        q = np.asarray([s['active_positions_rad'][:7] for s in samples])
        margin = np.minimum(q - limits[:, 0], limits[:, 1] - q)
        free = [s for s in samples if s['phase'].startswith('nut_index_free_')]
        if not free:
            raise ValueError('Missing actual robot samples for free return')
        first = np.asarray(free[0]['handbase_rotation_world_row_major'])
        last = np.asarray(free[-1]['handbase_rotation_world_row_major'])
        measured = float(np.degrees(socket[:3, 2] @ Rotation.from_matrix(last @ first.T).as_rotvec()))
        grasp_name = ('nut_regrasp_after_index' if path.parent.name == 'nut_reindex'
                      else path.parent.name.replace('nut_reindex_continued_', 'nut_regrasp_continued_'))
        grasp = read(stages / grasp_name / 'nut_regrasp_controller_result.json')
        checks = {
            'return_completed': record.get('completed') is True,
            'phase_selection_enabled': record['settings'].get('phase_equivalent_return') is True,
            'selected_original_phase_candidate': angle in (0., 90., -90.) and len(selected) == 1,
            'next_stroke_joint_preview_present': bool(selected) and selected[0].get('lookahead_samples', 0) > 0,
            'actual_return_arm_within_original_limits': bool(np.all(margin >= 0)),
            'next_grasp_completed': grasp.get('completed') is True,
            'next_grasp_preserved_selected_phase': grasp.get('phase_equivalent_reindex_yaw_retained') is True,
        }
        rows.append({
            'stage': path.parent.name, 'first_step': record['first_step'], 'last_step': record['last_step'],
            'selected_return_deg': angle, 'measured_free_hand_axial_rotation_deg': measured,
            'actual_j7_before_after_deg': [float(np.degrees(s['active_positions_rad'][6])) for s in (free[0], free[-1])],
            'actual_return_minimum_arm_margin_deg': float(np.degrees(margin.min())),
            'free_rotate_phase_samples': sum(s['phase'] == 'nut_index_free_rotate' for s in samples),
            'selection': selection, 'checks': checks,
        })
    result = {
        'scope': 'POSTRUN_RECORDED_RETURN_POLICY_AND_ROBOT_JOINT_AUDIT',
        'run': str(run), 'source_commit': process['source_commit'],
        'online_truth_used': False, 'rows': rows,
        'accepted': bool(rows) and all(all(r['checks'].values()) for r in rows),
        'selected_return_angles_deg': [r['selected_return_deg'] for r in rows],
        'zero_return_executed': any(r['selected_return_deg'] == 0. for r in rows),
        'limitations': [
            'This audit checks robot measurements during recorded returns and the subsequent grasp handoffs.',
            'Whole-episode physical contacts, seating, release and source geometry require the independent original full-assembly reviews.',
        ],
    }
    (run / 'adaptive_return_policy_review.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if not result['accepted']:
        raise SystemExit(2)


if __name__ == '__main__':
    review(sys.argv[1])
