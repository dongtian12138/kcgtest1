import copy
import numpy as np
from four_camera_online_completion import assess


def inputs():
    body = np.diag([1., -1., -1., 1.])
    body[2, 3] = -.014605
    observations = [{'position_and_axis_measured': True, 'capture_physics_time_s': t,
                     'world_from_plug_five_dof': body.tolist()} for t in (1., 5.)]
    release = {'completed': True, 'release_readiness': {'ready': True, 'unloaded': True},
               'opening_last_step': 960, 'support_hold_last_step': 3840, 'outer_abort_reason': None}
    settings = {'seating_confirmations': 2, 'nominal_seated_depth_m': .014605,
                'visual_seating_tolerance_m': .00002, 'stable_depth_tolerance_m': .000005,
                'seating_minimum_torque_nm': .4}
    return observations, release, settings


def judge(observations, release, settings):
    return assess(np.eye(4), observations, release, [0., 0., 90., 0., 0., 1.], settings,
        physics_dt_s=1/960, maximum_lateral_error_m=.00025, maximum_axis_error_deg=.1)


def test_force_stop_alone_does_not_mean_seated():
    observations, release, settings = inputs()
    for row in observations:
        row['world_from_plug_five_dof'][2][3] += .001
    assert not judge(observations, release, settings)['online_seating_confirmed']


def test_same_image_cannot_be_two_confirmations():
    observations, release, settings = inputs()
    assert not judge([observations[0], copy.deepcopy(observations[0])], release, settings)['online_seating_confirmed']


def test_declared_three_second_hold_does_not_replace_executed_hold():
    observations, release, settings = inputs()
    release.update(settings={'open_hold_duration_s': 3.}, support_hold_last_step=2880)
    assert not judge(observations, release, settings)['online_seating_confirmed']


def test_lateral_misalignment_rejects_matching_depth():
    observations, release, settings = inputs()
    for row in observations:
        row['world_from_plug_five_dof'][0][3] = .001
    assert not judge(observations, release, settings)['online_seating_confirmed']


def test_distinct_stable_visual_frames_and_unloaded_hold_confirm_online_only():
    observations, release, settings = inputs()
    result = judge(observations, release, settings)
    assert result['online_seating_confirmed']
    assert result['physical_assembly_acceptance'] == 'REQUIRES_ORIGINAL_INDEPENDENT_POSTRUN_REVIEW'
