"""Exercise the real interval's handoff decisions without running physics."""
import gzip
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'isaac'))
import te_local_interface_following
from te_coaxial_nut_interval import run_coaxial_nut_interval


def run_interval(tmp_path, monkeypatch, *, forces, threshold=90., seating=False):
    geometry = tmp_path / 'geometry.json'
    geometry.write_text(json.dumps({'finger_mechanism_id': 'test',
        'canonical_body_from_hand_for_nut_grasp': np.eye(4).tolist()}))
    recipe = {'source_geometry_plan': str(geometry),
        'planned_wrist_force_limit_n': 110.,
        'seating_torque_detection': {'minimum_command_deg': 350.}}
    if threshold is not None:
        recipe['regrasp_force_reserve'] = {'force_n': threshold}
    recipe_path = tmp_path / 'recipe.json'
    recipe_path.write_text(json.dumps(recipe))
    history = tmp_path / 'history.json.gz'
    with gzip.open(history, 'wt') as stream:
        json.dump([{'phase': 'key_probe_nut_tare'}, {'phase': 'key_probe_nut_grip_hold'}], stream)
    controllers = []

    class Controller:
        def __init__(self, *args, **kwargs):
            controllers.append(self)
            self.hand_from_pivot = np.eye(4)
            self.hand_goal = np.zeros(4)
            self.grip_observer = None
            self.turn_start = 3.5
            self.records = []
            self.calls = []
            self.seating_torque_candidate = None
            self.last_joint_velocity = np.zeros(7)
            self.last_contact_compensation = None

        def adopt_existing_grip(self, *args, **kwargs):
            pass

        def update(self, q, raw, elapsed, angle, rate, **kwargs):
            index = len(self.calls)
            self.calls.append(float(angle))
            force = forces[min(index, len(forces) - 1)]
            if np.linalg.norm(force) > 110.:
                raise RuntimeError('WRIST_OBSERVATION_BOUND: force_n')
            self.records.append({'interface_wrench': [*force, 0., 0., 0.]})
            if seating and index == 2:
                self.seating_torque_candidate = {'wrist_torque_nm': 2.1}
            return q[:7].copy()

        def report(self):
            return {'test_double': True}

    monkeypatch.setattr(te_local_interface_following, 'LocalInterfaceFollowing', Controller)
    ft = SimpleNamespace(samples=[{'hand2arm_raw_wrench': [0.] * 6}])
    mechanism = SimpleNamespace(setup={'mechanism_id': 'test', 'contract_path': str(tmp_path / 'unused')})
    world = SimpleNamespace(hand_mechanism=mechanism, get_physics_dt=lambda: .01,
        play=lambda: None, pause=lambda: None)
    stepper = SimpleNamespace(latest=(np.zeros(11), np.zeros(11), np.zeros(11)),
        settings={'arm_stiffness': 2500.}, arm_lower_limits=np.full(7, -3.),
        arm_upper_limits=np.full(7, 3.), step_index=100, abort_reason=None,
        set_lift_arm_damping=lambda value: None)

    def advance(*args, **kwargs):
        ft.samples.append({'step': stepper.step_index, 'hand2arm_raw_wrench': [0.] * 6})
        stepper.step_index += 1

    stepper.advance = advance
    model = SimpleNamespace(forward_kinematics=lambda *args, **kwargs: {'handbase_link': np.eye(4)})
    runtime = {'world': world, 'nail_body_ft_auditor': ft,
        'inputs': SimpleNamespace(robot_model=model), 'coaxial_nut_commanded_degrees': 245.,
        'nut_regrasp_geometry_check': lambda *args, **kwargs: None,
        'nut_regrasp_locate_visual_bounds': lambda *args, **kwargs: None}
    grip = {'effort_reference_nm': [2.] * 3, 'robot_sensor_history_file': str(history),
        'world_from_body_palm_five_dof': np.eye(4).tolist()}
    settings = {'coaxial_interface_recipe': str(recipe_path),
        'rotation_about_socket_plus_z_deg': -90., 'maximum_rotation_speed_deg_s': 50.}
    result = run_coaxial_nut_interval(tmp_path, runtime, stepper, {}, grip,
        np.eye(4), settings, tmp_path / 'interval')
    return result, runtime, controllers[0]


def test_resultant_force_stops_before_next_command_and_preserves_actual_budget(tmp_path, monkeypatch):
    result, runtime, controller = run_interval(tmp_path, monkeypatch,
        forces=[(10., 0., 0.), (50., 0., 0.), (70., 60., 0.)])
    assert result['completed'] and result['normal_stop_reason'] == 'WRIST_FORCE_RESERVE_EARLY_REGRASP'
    assert result['last_step'] == 102 and result['sample_count'] == 2
    assert result['early_regrasp']['force_n'] == pytest.approx(np.hypot(70., 60.))
    applied = np.degrees(controller.calls[1])
    assert applied < np.degrees(controller.calls[2])
    assert result['last_applied_rotation_command_deg'] == pytest.approx(-applied)
    assert runtime['coaxial_nut_commanded_degrees'] == pytest.approx(245. + applied)
    assert result['release_from_actual_encoder_pose'] and result['release_with_contact_feedforward_removed']
    assert result['physical_thread_progress_verified'] is False


def test_hard_controller_stop_is_not_relabelled_as_normal_regrasp(tmp_path, monkeypatch):
    result, runtime, controller = run_interval(tmp_path, monkeypatch,
        forces=[(10., 0., 0.), (50., 0., 0.), (111., 0., 0.)])
    assert not result['completed']
    assert result['failure_reason'] == 'WRIST_OBSERVATION_BOUND: force_n'
    assert 'early_regrasp' not in result and 'normal_stop_reason' not in result
    assert result['last_step'] == 102


def test_existing_seating_candidate_keeps_priority(tmp_path, monkeypatch):
    result, _, _ = run_interval(tmp_path, monkeypatch,
        forces=[(10., 0., 0.), (50., 0., 0.), (95., 0., 0.)], seating=True)
    assert result['completed'] and 'seating_candidate' in result
    assert 'early_regrasp' not in result and result['last_step'] == 102


def test_omitting_force_reserve_preserves_old_finite_interval(tmp_path, monkeypatch):
    result, _, controller = run_interval(tmp_path, monkeypatch,
        forces=[(95., 0., 0.)], threshold=None)
    assert result['completed'] and 'early_regrasp' not in result
    assert result['sample_count'] == len(controller.calls) == round(3.375 / .01)


@pytest.mark.parametrize('threshold', [0., 110., float('nan')])
def test_reserve_requires_positive_margin_below_hard_stop(tmp_path, monkeypatch, threshold):
    result, _, _ = run_interval(tmp_path, monkeypatch, forces=[(10., 0., 0.)], threshold=threshold)
    assert not result['completed'] and result['last_step'] == 100
    assert 'must precede the unchanged resultant-force stop' in result['failure_reason']
