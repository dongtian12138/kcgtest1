import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from two_stage_key_alignment import axial_target
from key_direction_memory import KeyDirectionMemory


def body_at(angle):
    p=np.eye(4);p[:3,:3]=Rotation.from_euler('z',angle,degrees=True).as_matrix()@np.diag([-1.,1.,-1.])
    p[:3,3]=[.49,.185,.275]
    return p


@pytest.mark.parametrize('slot_angle',[-25.,20.])
def test_same_body_and_transport_require_opposite_turns_from_changed_visual_slot(slot_angle):
    body=body_at(0);socket=np.eye(4)
    socket[:3,:3]=Rotation.from_euler('z',slot_angle,degrees=True).as_matrix()
    target,record=axial_target(body,socket)
    assert record['signed_rotation_about_body_axis_deg']==pytest.approx(-slot_angle)
    np.testing.assert_allclose(target[:3,3],body[:3,3])
    np.testing.assert_allclose(target[:3,2],body[:3,2])
    np.testing.assert_allclose(target[:3,1],socket[:3,1],atol=1e-12)


def test_target_changes_with_visual_measurement_in_rotated_world_frame():
    body=body_at(-30);socket=np.eye(4)
    world=np.eye(4);world[:3,:3]=Rotation.from_euler('xyz',[.2,-.3,.7]).as_matrix();world[:3,3]=[1.,2.,3.]
    ordinary,a=axial_target(body,socket)
    rotated,b=axial_target(world@body,world@socket)
    np.testing.assert_allclose(rotated,world@ordinary,atol=1e-12)
    assert a['signed_rotation_about_body_axis_deg']==pytest.approx(b['signed_rotation_about_body_axis_deg'])


def test_second_key_image_corrects_twist_that_palm_cannot_observe():
    memory=KeyDirectionMemory();hand=np.eye(4);body=body_at(0)
    memory.initialize(hand,body,0.)
    slipped=body_at(.5)
    memory.update_palm(np.eye(4),slipped,1.)
    np.testing.assert_allclose(memory.predict(hand),body)
    report=memory.reobserve_after_coarse_turn(hand,slipped,2.)
    np.testing.assert_allclose(memory.predict(hand),slipped)
    assert abs(report['previous_prediction_to_new_measurement_axial_deg'])==pytest.approx(.5)
    assert memory.report()['anchor_count']==2
    with pytest.raises(RuntimeError):memory.reobserve_after_coarse_turn(hand,slipped,3.)


def test_stale_refinement_and_refinement_after_release_are_rejected():
    memory=KeyDirectionMemory();p=body_at(0);memory.initialize(np.eye(4),p,2.)
    with pytest.raises(ValueError):memory.reobserve_after_coarse_turn(np.eye(4),p,1.)
    memory.retire('Body released')
    with pytest.raises(RuntimeError):memory.reobserve_after_coarse_turn(np.eye(4),p,3.)
