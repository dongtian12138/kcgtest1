import sys
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
        SimpleNamespace(_json_ready=lambda value: value))
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
        'sample_step': 0, 'directory': 'image-only', 'measurement': {
            'world_from_plug_five_dof': measured, 'camera_from_plug_five_dof': measured}})
    session.world.current_time = .05
    session.service(request_new=False)
    assert session.relation_for_transport()[0, 3] == 0.
    session.world.current_time = .1
    session.service(request_new=False)
    np.testing.assert_allclose(session.relation_for_transport()[:3, :3], np.eye(3))
    assert session.relation_for_transport()[0, 3] == .001
    assert not session.memory.initialized


def test_key_is_initialized_once_at_later_station_and_palm_does_not_replace_yaw(session):
    keyed = np.eye(4)
    keyed[:3, :3] = Rotation.from_euler('z', .2).as_matrix()
    palm = keyed.copy()
    palm[:3, :3] = Rotation.from_euler('z', 1.7).as_matrix()
    anchor = {'key_observation_event_count': 1, 'physics_time_s': 1.,
        'world_from_hand_encoder': np.eye(4),
        'key_measurement': {'world_from_plug_row_major': keyed.ravel()},
        'palm_observation': {'world_from_camera_cv': np.eye(4), 'world_from_plug_five_dof': palm}}
    session.world.current_time = 2.
    session.adopt_key_anchor(anchor)
    np.testing.assert_allclose(session.body(), keyed)
    assert session.prekey_relation is None
    with pytest.raises(ValueError):
        session.adopt_key_anchor(anchor)
