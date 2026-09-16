"""Bounded reference progression, pause/resume and braking margin regression."""
from pathlib import Path
import sys
import math
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'isaac'))
from te_nut_motion import LoadedTurnProgress,coordinated_turn_scale


def test_paused_reference_keeps_state_until_step_is_committed_and_can_resume():
    p=LoadedTurnProgress(-math.radians(12),math.radians(5),math.radians(8.75))
    dt=1/960;history=[]
    for n in range(4800):
        scale=0. if 960<=n<1920 else 1.
        old=p.position;c=p.candidate(dt,scale)
        assert p.position==old
        p.commit(c);history.append((p.position,p.speed))
    h=np.asarray(history)
    assert np.max(h[:,1])<=p.maximum_speed+1e-12
    assert np.max(np.abs(np.diff(h[:,1])))<=p.maximum_acceleration*dt+1e-9
    assert np.max(np.diff(h[:,0]))<=1e-12
    assert abs(h[1800,0]-h[1919,0])<1e-12
    assert h[2300,0]<h[1919,0]
    assert abs(p.position-p.distance)<1e-10 and p.speed==0.


def test_tracking_growth_brakes_before_the_hard_stop_and_unload_stops_phase():
    args=(.00018,.00017,0.,.05,.1,.000125,.00025,.01)
    scale,growth,predicted=coordinated_turn_scale(*args,False)
    assert growth>0 and predicted>.00018 and scale<.56
    assert coordinated_turn_scale(*args,True)[0]==0.
    assert coordinated_turn_scale(.00005,.00006,0.,0.,.1,.000125,.00025,.01,False)[0]==1.


def test_positive_force_correction_does_not_pause_net_inward_screw_motion():
    from te_nut_motion import requires_axial_unload
    speed=-math.radians(5)
    assert not requires_axial_unload(.00005,speed,.00762,.00003)
    assert requires_axial_unload(.0002,speed,.00762,.00003)
    assert requires_axial_unload(.00005,0.,.00762,.00003)


def test_shared_preload_and_turn_interface_causes_real_input_reversal_under_load():
    from types import SimpleNamespace
    from te_worm_drive import WormDrive,WormReference,finger_force_motor_targets
    drives={n:WormDrive(.67,reference=WormReference(transmission_damping=0.,output_viscosity=2.),integration='passive_split')
            for n in ('f1j2','f2j1','f3j2')}
    for drive in drives.values():drive.input_angle=.695
    mechanism=SimpleNamespace(drives=drives,settings={'motor_position_kp':264.,'motor_position_kd':4.4})
    targets,rows=finger_force_motor_targets(mechanism,[.67]*3,[1.047,.70,.70,.70],
        [2.1]*3,[2.4]*3,1/960,120.,1/6,.18,.02,[1.,.5,.5,.5],[1.1,.85,.85,.85])
    for i,drive in enumerate(drives.values()):
        drive.prepare_position(.67,0.,targets[i+1],1/960,stiffness=264.,damping=4.4)
        assert drive.pending['v']<0
        assert abs(drive.pending['v']-rows[i]['desired_input_velocity_rad_s'])<1e-10
        assert abs(rows[i]['requested_motor_effort_nm'])<=7.7


def test_release_does_not_reclose_when_measured_load_falls_faster_than_reference():
    from types import SimpleNamespace
    from te_worm_drive import WormDrive,WormReference,finger_force_motor_targets
    drives={n:WormDrive(.67,reference=WormReference(transmission_damping=0.,output_viscosity=2.),integration='passive_split')
            for n in ('f1j2','f2j1','f3j2')}
    for d in drives.values():d.input_angle=.695
    mechanism=SimpleNamespace(drives=drives,settings={'motor_position_kp':264.,'motor_position_kd':4.4})
    targets,rows=finger_force_motor_targets(mechanism,[.67]*3,[1.047,.7,.7,.7],
        [.4]*3,[.2]*3,1/960,120.,1/6,.18,.02,[1.,.5,.5,.5],[1.1,.85,.85,.85],allow_closing=False)
    for i,d in enumerate(drives.values()):
        d.prepare_position(.67,0.,targets[i+1],1/960,stiffness=264.,damping=4.4)
        assert d.pending['v']==0.
        assert rows[i]['requested_motor_effort_nm']==0.


def test_release_requires_actual_geometry_and_unloaded_measurement_together():
    from te_nut_motion import released_grip_readiness
    assert not released_grip_readiness(False,[0.,0.,0.],.02)['ready']
    assert not released_grip_readiness(True,[0.,.3,0.],.02)['ready']
    assert not released_grip_readiness(True,[float('nan'),0.,0.],.02)['ready']
    assert released_grip_readiness(True,[.001,-.002,.001],.02)['ready']


def test_total_interface_speed_limits_include_large_pose_error_correction():
    from scipy.spatial.transform import Rotation
    from te_nut_motion import bound_interface_velocity
    axes=Rotation.from_rotvec([.3,-.2,.5]).as_matrix();axis=-axes[:,2]
    # Captured seating error was about0.7mm;20/s feedback can request14mm/s.
    original=np.r_[axes@np.array([.014,-.006,.012]),axes@np.array([.08,-.05,.12])]
    bounded=bound_interface_velocity(original,axes,axis,.001,.0015,np.deg2rad(2),np.deg2rad(2))
    local=axes.T@bounded[:3]
    assert np.linalg.norm(local[:2])<=.001+1e-12
    assert abs(local[2]-.0015)<1e-12
    assert np.allclose(local[:2]/np.linalg.norm(local[:2]),np.array([.014,-.006])/np.hypot(.014,.006))
    spin=bounded[3:]@axis;tilt=bounded[3:]-spin*axis
    assert abs(spin)<=np.deg2rad(2)+1e-12
    assert np.linalg.norm(tilt)<=np.deg2rad(2)+1e-12


def test_interface_bound_preserves_an_admissible_screw_motion():
    from te_nut_motion import bound_interface_velocity
    request=np.array([0.,0.,-.00091,.002,0.,-.75])
    assert np.allclose(bound_interface_velocity(request,np.eye(3),[0,0,-1],
        .001,.0015,np.deg2rad(2),np.deg2rad(50)),request)
def test_axial_unloading_is_not_overridden_by_accumulated_position_error():
    from te_nut_motion import combine_with_axial_force_control,bound_interface_velocity
    from scipy.spatial.transform import Rotation
    axes=Rotation.from_rotvec([.2,-.4,.1]).as_matrix();axis=axes[:,2]
    # The measured force requests1.5mm/s outward, but the old position loop
    # would demand15.8mm/s inward even though the nut turn has paused.
    motion=np.r_[axis*.0015,[0.,0.,0.]]
    feedback=np.r_[axes@np.array([.0002,-.0003,-.0158]),[.01,.02,0.]]
    command=combine_with_axial_force_control(motion,feedback,axis)
    bounded=bound_interface_velocity(command,axes,-axis,.001,.0015,.035,.035)
    assert np.isclose(axis@bounded[:3],.0015)
    assert np.allclose(axes[:,:2].T@bounded[:3],[.0002,-.0003])
    assert np.allclose(command[3:],feedback[3:])
