import sys
import json
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import four_camera_perception_session as module
from perception_latency import DelayedObservation


@pytest.fixture
def session(tmp_path, monkeypatch):
    (tmp_path / 'assembly.yaml').write_text('perception: {}\n')
    monkeypatch.setattr(module, 'configuration', lambda *_: {})
    monkeypatch.setattr(module, 'camera_spec', lambda *_:
        ({'mount': {'hand_from_camera_cv': np.eye(4)}}, np.eye(4)))
    monkeypatch.setitem(sys.modules, 'te_foundationpose_handoff_runtime',
        SimpleNamespace(_json_ready=lambda value: json.loads(json.dumps(value,default=lambda item:item.tolist()))))
    world = SimpleNamespace(current_time=0.)
    model = SimpleNamespace(forward_kinematics=lambda *_, **__: {'handbase_link': np.eye(4)})
    runtime = {'world': world, 'output_directory': tmp_path, 'inputs': SimpleNamespace(robot_model=model),
               'body_assembly_control_config': 'assembly.yaml'}
    stepper = SimpleNamespace(latest=(np.zeros(11),), step_index=1)
    value = module.FourCameraPerceptionSession(tmp_path, runtime, stepper,
                                             prekey_world_from_body=np.eye(4))
    yield value
    value.events.close()


def test_prekey_palm_update_waits_and_does_not_claim_a_key(session):
    measured = np.eye(4)
    measured[0, 3] = .001
    measured[:3, :3] = Rotation.from_euler('z', 1.).as_matrix()
    session.pending = DelayedObservation(0., .1, 0., {
        'observation': {'hand': np.eye(4), 'camera': np.eye(4), 'intrinsics': np.eye(3), 'capture': {}},
        'sample_step': 0, 'directory': 'image-only', 'measurement': {
            'world_from_plug_five_dof': measured, 'camera_from_plug_five_dof': measured, 'metrics': {}}})
    session.world.current_time = .05
    session.service(request_new=False)
    assert session.relation_for_transport()[0, 3] == 0.
    session.world.current_time = .1
    session.service(request_new=False)
    np.testing.assert_allclose(session.relation_for_transport()[:3, :3], np.eye(3))
    assert session.relation_for_transport()[0, 3] == .001
    assert not session.memory.initialized
    result=session.available_observation(session.root/'consumer')
    assert result['capture_physics_time_s']==0.
    assert result['sample_age_at_stage_consumption_s']==.1
    session.world.current_time=.6
    with pytest.raises(RuntimeError):
        session.available_observation(session.root/'stale_consumer')


def test_key_is_initialized_once_at_later_station_and_palm_does_not_replace_yaw(session):
    keyed = np.eye(4)
    keyed[:3, :3] = Rotation.from_euler('z', .2).as_matrix()
    palm = keyed.copy()
    palm[:3, :3] = Rotation.from_euler('z', 1.7).as_matrix()
    anchor = {'key_observation_event_count': 1, 'physics_time_s': 1.,
        'observation_latency': {'availability_physics_time_s': 1.5},
        'world_from_hand_encoder': np.eye(4),
        'key_measurement': {'world_from_plug_row_major': keyed.ravel()},
        'palm_observation': {'world_from_camera_cv': np.eye(4), 'world_from_plug_five_dof': palm}}
    session.world.current_time = 1.1
    with pytest.raises(ValueError):
        session.adopt_key_anchor(anchor)
    assert not session.memory.initialized
    session.world.current_time = 2.
    session.adopt_key_anchor(anchor)
    np.testing.assert_allclose(session.body(), keyed)
    assert session.prekey_relation is None
    with pytest.raises(ValueError):
        session.adopt_key_anchor(anchor)


def test_released_body_does_not_follow_subsequent_hand_rotation(session):
    session.retire_body_grasp()
    hand=np.eye(4)
    hand[:3,:3]=Rotation.from_euler('z',1.).as_matrix()
    hand[0,3]=.1
    session.hand=lambda:hand
    np.testing.assert_allclose(session.body(),np.eye(4))


def test_two_stage_session_requires_completed_turn_and_causal_second_image(session):
    session.rig.update(maximum_key_observation_events=2)
    keyed=np.eye(4)
    def anchor(count,time,yaw):
        measured=keyed.copy();measured[:3,:3]=Rotation.from_euler('z',yaw).as_matrix()
        return {'key_observation_event_count':count,'physics_time_s':time,
            'observation_latency':{'availability_physics_time_s':time+.5},
            'world_from_hand_encoder':np.eye(4),
            'key_measurement':{'world_from_plug_row_major':measured.ravel()},
            'palm_observation':{'world_from_camera_cv':np.eye(4),'world_from_plug_five_dof':keyed}}
    session.world.current_time=2.;session.adopt_key_anchor(anchor(1,1.,.2))
    with pytest.raises(RuntimeError):session.begin_key_refinement()
    session.runtime['coarse_key_alignment_completed']=True
    session.world.current_time=3.;session.begin_key_refinement()
    second=anchor(2,3.,.3)
    with pytest.raises(ValueError):session.adopt_key_anchor(second)
    assert session.key_anchor_pending and session.memory.anchor_count==1
    session.service()  # Suspended while holding physically for the second result.
    session.world.current_time=3.5;session.adopt_key_anchor(second)
    np.testing.assert_allclose(session.body()[:3,:3],Rotation.from_euler('z',.3).as_matrix())
    assert session.memory.anchor_count==2 and not session.key_anchor_pending
    with pytest.raises(ValueError):session.adopt_key_anchor(second)
