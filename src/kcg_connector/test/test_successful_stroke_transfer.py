"""Task-space and joint-limit regressions for the successful policy transfer."""
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'isaac'))
from te_nut_motion import successful_stroke_tilt,successful_stroke_joint_velocity


def test_tilt_uses_socket_axes_and_never_commands_thread_axis_rotation():
    axes=Rotation.from_rotvec([.4,-.3,.2]).as_matrix()
    rotation,velocity=successful_stroke_tilt(np.eye(3),axes,[.2,-.1],1/240)
    assert abs(axes[:,2]@velocity)<1e-12
    assert np.allclose(axes.T@velocity,[.02,-.01,0.])
    assert np.allclose(Rotation.from_matrix(rotation).as_rotvec(),velocity/240)


def test_tilt_stops_at_bound_and_relaxes_after_load_removal():
    rotation=np.eye(3)
    for _ in range(1000):
        rotation,velocity=successful_stroke_tilt(rotation,np.eye(3),[2.,0.],1/240)
        assert Rotation.from_matrix(rotation).magnitude()<=np.deg2rad(.75)+1e-12
        assert np.linalg.norm(velocity)<=np.deg2rad(2.)+1e-12
    before=Rotation.from_matrix(rotation).magnitude()
    rotation,velocity=successful_stroke_tilt(rotation,np.eye(3),[0.,0.],1/240)
    assert Rotation.from_matrix(rotation).magnitude()<before


def test_redundancy_preserves_task_motion_at_joint_limit():
    # Two joints can make the same x translation. Joint 0 is at its upper
    # braking margin; joint 6 must supply the requested movement instead.
    J=np.column_stack((np.eye(6),np.array([1.,0.,0.,0.,0.,0.])))
    bounds=np.tile([-1.,1.],(7,1));arm=np.zeros(7);arm[0]=.995
    velocity,residual=successful_stroke_joint_velocity(J,[.1,0,0,0,0,0],np.zeros(6),arm,bounds,.2)
    assert velocity[0]<=1e-12 and velocity[6]>.0999
    assert np.linalg.norm(residual)<1e-6


def test_translation_feedback_corrects_measured_displacement():
    J=np.column_stack((np.eye(6),np.zeros(6)))
    error=np.array([.0001,0,0,0,0,0])
    velocity,_=successful_stroke_joint_velocity(J,np.zeros(6),error,np.zeros(7),np.tile([-1.,1.],(7,1)),.2)
    assert abs(velocity[0]-.002)<1e-8


def test_contact_load_compensation_restores_payload_and_wrench_origin():
    from te_body_nut_rotation import interface_wrench_from_wrist
    axes=Rotation.from_rotvec([.1,-.2,.3]).as_matrix()
    hand=np.array([.55,.18,.69]);pivot=np.array([.55,.185,.25]);com=pivot+np.array([.001,-.002,.01])
    measured=np.array([20.,-12.,8.,-6.,9.,.7]);mass=.062459669349
    fixed=interface_wrench_from_wrist(measured,hand,pivot,com,mass,9.81,axes)
    gravity=np.array([0.,0.,-mass*9.81])
    force=axes@fixed[:3]+gravity
    moment=axes@fixed[3:]+np.cross(com-pivot,gravity)+np.cross(pivot-hand,force)
    assert np.allclose(np.r_[force,moment],measured,atol=1e-12)


def test_two_stage_wrench_transform_preserves_the_final_reference():
    from te_body_nut_rotation import interface_wrench_from_wrist
    # Regression for an incomplete audit that inspected only the first shift.
    # Intermediate references differ, but both full transform chains must
    # produce the same wrench at the current pivot.
    force=np.array([80.,0.,0.]);couple=np.array([0.,1.,1.1])
    sensor=np.array([0.,0.,.45]);origin=np.zeros(3)
    measured=np.r_[force,couple+np.cross(origin-sensor,force)]
    original=interface_wrench_from_wrist(measured,sensor,origin,origin,0.,9.81,np.eye(3))
    deeper=interface_wrench_from_wrist(measured,sensor,[0.,0.,-.006],origin,0.,9.81,np.eye(3))
    assert np.isclose(original[4],1.) and np.isclose(deeper[4],1.48)
    assert np.linalg.norm(original[3:5])<1.2<np.linalg.norm(deeper[3:5])
    assert np.array_equal(original[:3],deeper[:3])
    assert original[5]==deeper[5]
    current=np.array([0.,0.,-.009])
    final_from_original=original[3:]+np.cross(origin-current,original[:3])
    final_from_deeper=deeper[3:]+np.cross(np.array([0.,0.,-.006])-current,deeper[:3])
    assert np.allclose(final_from_original,final_from_deeper,atol=1e-12)
