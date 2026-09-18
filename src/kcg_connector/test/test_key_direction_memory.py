"""Coordinate tests for the one-key-observation/5DOF reconstruction contract."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from key_direction_memory import KeyDirectionMemory,minimum_axis_rotation


def pose(angles=(0,0,0),position=(0,0,0)):
    out=np.eye(4);out[:3,:3]=Rotation.from_euler('xyz',angles).as_matrix();out[:3,3]=position;return out


class KeyDirectionMemoryTests(unittest.TestCase):
    def test_encoder_motion_carries_initial_key_and_all_rigid_offsets(self):
        h0=pose((.2,-.1,.8),(.4,.1,.7));hb=pose((.03,.01,1.2),(.01,-.005,.45))
        tracker=KeyDirectionMemory();tracker.initialize(h0,h0@hb,1.)
        h1=pose((-.5,.3,-1.7),(.6,-.2,.8))
        np.testing.assert_allclose(tracker.predict(h1),h1@hb,atol=1e-12)

    def test_palm_corrects_translation_and_tilt_without_using_its_arbitrary_yaw(self):
        hb=pose((.05,-.02,.8),(.01,-.005,.45));hc=pose((.1,.2,-.4),(.02,0,.315))
        tracker=KeyDirectionMemory();tracker.initialize(np.eye(4),hb,0.)
        axis=Rotation.from_rotvec([.002,-.003,0]).apply(hb[:3,2])
        expected=hb.copy();expected[:3,:3]=minimum_axis_rotation(hb[:3,2],axis)@hb[:3,:3]
        expected[:3,3]+=[.0003,-.0002,.0001]
        camera=np.linalg.inv(hc)@expected
        # The image estimator is allowed an arbitrary transverse basis.
        camera[:3,:3]=camera[:3,:3]@Rotation.from_euler('z',2.4).as_matrix()
        result=tracker.update_palm(hc,camera,.1)
        self.assertFalse(result['palm_axial_yaw_used'])
        np.testing.assert_allclose(tracker.hand_from_body(),expected,atol=1e-12)

    def test_unobserved_twist_does_not_get_invented_from_circular_face_frame(self):
        tracker=KeyDirectionMemory();body=pose((0,0,.7),(0,0,.45));tracker.initialize(np.eye(4),body,0.)
        image=body.copy();image[:3,:3]=image[:3,:3]@Rotation.from_euler('z',.2).as_matrix()
        tracker.update_palm(np.eye(4),image,1.)
        np.testing.assert_allclose(tracker.hand_from_body(),body,atol=1e-12)
        self.assertFalse(tracker.report()['extra_axial_rotation_is_measured'])

    def test_body_release_retires_grasp_relation_and_reobservation_is_not_silent(self):
        tracker=KeyDirectionMemory();tracker.initialize(np.eye(4),np.eye(4),0.)
        with self.assertRaises(RuntimeError):tracker.initialize(np.eye(4),np.eye(4),1.)
        tracker.retire('BODY_RELEASED_AFTER_KEY_ENTRY')
        with self.assertRaises(RuntimeError):tracker.predict(np.eye(4))

    def test_out_of_order_sample_and_axis_reversal_fail(self):
        tracker=KeyDirectionMemory();tracker.initialize(np.eye(4),np.eye(4),2.)
        with self.assertRaises(ValueError):tracker.update_palm(np.eye(4),np.eye(4),1.)
        with self.assertRaises(ValueError):tracker.update_palm(np.eye(4),pose((np.pi,0,0)),3.)


if __name__=='__main__':unittest.main()
