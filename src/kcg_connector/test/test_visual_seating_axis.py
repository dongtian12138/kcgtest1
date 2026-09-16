import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from te_visual_seating_axis import visual_grasp_axis_update,seating_segment_duration


class VisualSeatingAxisTests(unittest.TestCase):
    def test_camera_updates_axis_and_lateral_estimate_without_axial_or_pose_write(self):
        hand=np.eye(4);hand[:3,3]=[.55,.185,.7]
        relation=np.eye(4);relation[:3,3]=[.005,.001,-.441]
        relation[:3,:3]=Rotation.from_rotvec([np.pi,0.,0.]).as_matrix()
        current=hand@relation
        body=current.copy();body[:3,:3]=Rotation.from_rotvec([.002,-.001,0.]).as_matrix()@current[:3,:3]
        body[:3,3]=current[:3,3]+[.0002,-.0001,.0003]-body[:3,2]*.0005
        saved=[x.copy() for x in (hand,relation,body)]
        new,report=visual_grasp_axis_update(hand,relation,body,[0.,0.,1.],
            maximum_lateral_change_m=.0006,maximum_axis_change_deg=.75)
        updated=hand@new
        np.testing.assert_allclose(updated[:3,2],body[:3,2],atol=1e-12)
        np.testing.assert_allclose(updated[:2,3],current[:2,3]+[.0002,-.0001],atol=1e-12)
        self.assertAlmostEqual(updated[2,3],current[2,3])
        for actual,original in zip((hand,relation,body),saved):np.testing.assert_array_equal(actual,original)
        self.assertFalse(report['physical_pose_written'])
        # Body yaw is unobserved and must not change the grasp yaw estimate.
        body[:3,:3]=body[:3,:3]@Rotation.from_rotvec([0.,0.,1.2]).as_matrix()
        other,_=visual_grasp_axis_update(hand,relation,body,[0.,0.,1.],
            maximum_lateral_change_m=.0006,maximum_axis_change_deg=.75)
        np.testing.assert_allclose(other,new,atol=1e-12)

    def test_large_camera_correction_is_rejected(self):
        h=np.eye(4);r=np.eye(4);b=np.eye(4);b[:3,3]=[.002,0.,-.0005]
        with self.assertRaises(ValueError):
            visual_grasp_axis_update(h,r,b,[0.,0.,1.],maximum_lateral_change_m=.0006,maximum_axis_change_deg=.75)

    def test_stale_grasp_offset_can_be_reestimated_without_moving_the_body(self):
        hand=np.eye(4);relation=np.eye(4);relation[:3,3]=[-.000783,.000385,0.]
        body=np.eye(4);body[:3,3]=[.0000036,-.0000063,-.0005]
        saved=body.copy()
        new,report=visual_grasp_axis_update(hand,relation,body,[0.,0.,1.],
            maximum_lateral_change_m=.0012,maximum_axis_change_deg=.75)
        np.testing.assert_array_equal(body,saved)
        np.testing.assert_allclose(new[:2,3],body[:2,3],atol=1e-12)
        self.assertTrue(report['estimate_update_is_not_a_physical_displacement_command'])

    def test_short_stroke_keeps_the_successful_quarter_turn_acceleration_envelope(self):
        acceleration=(10/np.sqrt(3))*90/3.375**2
        self.assertAlmostEqual(seating_segment_duration(90.,50.,acceleration),3.375)
        for degrees in (3.43140506623604,8.,27.43140506623604):
            duration=seating_segment_duration(degrees,50.,acceleration)
            self.assertLessEqual(1.875*degrees/duration,50.+1e-10)
            self.assertLessEqual((10/np.sqrt(3))*degrees/duration**2,acceleration+1e-10)


if __name__=='__main__':unittest.main()
