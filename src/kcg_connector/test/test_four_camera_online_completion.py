import copy
import numpy as np
import pytest
from four_camera_online_completion import assess


def inputs():
    body = np.diag([1., -1., -1., 1.])
    body[2, 3] = -.014605
    observations = [{'position_and_axis_measured': True,
                     'capture_physics_time_s': step/960, 'sample_step': step,
                     'available_physics_time_s': step/960+.08,
                     'consumed_physics_time_s': step/960+.1,
                     'consumed_step': step+96,
                     'world_from_plug_five_dof': body.tolist()}
                    for step in range(979, 3840, 192)]
    release = {'completed': True, 'release_readiness': {'ready': True, 'unloaded': True},
               'opening_last_step': 960, 'support_hold_last_step': 3840, 'outer_abort_reason': None}
    settings = {'seating_confirmations': 2, 'nominal_seated_depth_m': .014605,
                'visual_seating_tolerance_m': .00002, 'stable_depth_tolerance_m': .000005,
                'seating_minimum_torque_nm': .4}
    return observations, release, settings


def judge(observations, release, settings):
    return assess(np.eye(4), observations, release, [0., 0., 90., 0., 0., 1.], settings,
        physics_dt_s=1/960, maximum_lateral_error_m=.00025, maximum_axis_error_deg=.1,
        palm_period_s=.2, maximum_observation_age_s=.5, decision_time_s=4.)


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


def test_loaded_to_released_settling_is_not_released_instability():
    observations, release, settings = inputs()
    loaded = copy.deepcopy(observations[0])
    loaded.update(sample_step=800, capture_physics_time_s=800/960)
    loaded['world_from_plug_five_dof'][2][3] += 7.4e-6
    result = judge([loaded, *observations], release, settings)
    assert result['online_seating_confirmed']
    assert len(result['observations']) == len(observations)


def test_released_instability_is_rejected_even_when_endpoints_agree():
    observations, release, settings = inputs()
    observations[7]['world_from_plug_five_dof'][2][3] += 6e-6
    result = judge(observations, release, settings)
    assert result['within_original_visual_depth_tolerance']
    assert not result['within_original_visual_depth_stability_tolerance']
    assert not result['online_seating_confirmed']


def test_endpoints_do_not_replace_continuous_released_sensor_coverage():
    observations, release, settings = inputs()
    assert not judge([observations[0], observations[-1]], release, settings)['online_seating_confirmed']


def test_missing_interior_frame_rejects_coverage():
    observations, release, settings = inputs()
    del observations[7]
    assert not judge(observations, release, settings)['released_window_coverage']['complete']


def test_recorded_processing_delay_is_not_a_missing_frame():
    observations, release, settings = inputs()
    previous = observations[6]
    previous['available_physics_time_s'] = previous['capture_physics_time_s'] + .399
    previous['consumed_physics_time_s'] = previous['capture_physics_time_s'] + .4
    previous['consumed_step'] = previous['sample_step'] + 384
    del observations[7]
    assert judge(observations, release, settings)['online_seating_confirmed']


def test_future_measurement_cannot_confirm_completion():
    observations, release, settings = inputs()
    observations[-1]['consumed_physics_time_s'] = 4.1
    with pytest.raises(ValueError, match='causally available'):
        judge(observations, release, settings)


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
