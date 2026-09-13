"""Regression of bounded trajectories, readiness and observed progress."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'isaac'))
from te_nut_motion import ScalarMotion,joint7_return_path,preload_readiness,classify_observed_progress,limit_joint_velocity_without_changing_direction


@pytest.mark.parametrize('distance',[0.,.02,-.02,np.pi/2,-np.pi/2])
def test_profile_respects_speed_acceleration_and_endpoint(distance):
    p=ScalarMotion(distance,1.,2.)
    times=np.linspace(0,max(p.duration,.01),2001)
    data=np.array([p.at(t) for t in times]);dt=times[1]-times[0]
    assert data[0,0]==0 and abs(data[-1,0]-distance)<1e-12
    assert data[0,1]==0 and abs(data[-1,1])<1e-12
    assert max(abs(data[:,1]))<=1.+1e-12
    assert max(abs(np.diff(data[:,1])/dt))<=2.+1e-9


def test_empty_hand_return_changes_only_joint7_in_a_few_seconds():
    start=np.array([.2,.4,.3,-.7,-.2,1.7,2.])
    path,r=joint7_return_path(start,2.-np.pi/2,np.full(7,-3.),np.full(7,3.),1/960)
    assert r['duration_s']<2.2
    assert np.array_equal(path[:,:6],np.repeat(start[None,:6],len(path),axis=0))
    assert path[-1,6]==2.-np.pi/2
    assert r['maximum_planned_speed_rad_s']<=1.+1e-10
    assert np.array_equal(start,np.array([.2,.4,.3,-.7,-.2,1.7,2.]))
    with pytest.raises(ValueError):joint7_return_path(start,4.,np.full(7,-3.),np.full(7,3.),1/960)


def test_velocity_limit_preserves_cartesian_command_direction():
    v=np.array([.1,.05,-.2,.01,0,0,.02]);limited,scale=limit_joint_velocity_without_changing_direction(v,.075)
    assert np.max(abs(limited))<=.075
    assert np.allclose(limited,v*scale)


def test_preload_time_elapsed_is_not_enough_to_accept_grip():
    targets=np.array([2.1,2.25,2.14]);healthy=np.repeat(targets[None,:],240,axis=0)
    assert preload_readiness(healthy,targets,[.5,.3,.5])['ready']
    assert 'PRELOAD_TARGET_NOT_REACHED' in preload_readiness(healthy*.5,targets,[.5,.3,.5])['reasons']
    assert 'INSUFFICIENT_TRANSMISSION_RESERVE' in preload_readiness(healthy,targets,[.5,.01,.5])['reasons']
    healthy[0,2]*=1.4
    assert 'PRELOAD_NOT_STABLE' in preload_readiness(healthy,targets,[.5,.3,.5])['reasons']


def progress_settings():
    return dict(nominal_seated_depth_m=.014605,visual_seating_tolerance_m=.00002,
        stable_depth_tolerance_m=.000005,seating_minimum_torque_nm=.4,
        minimum_loaded_command_deg=40.,minimum_progress_check_angle_deg=.75,minimum_observed_progress_m=.00001)


def test_observed_progress_distinguishes_seating_stall_and_slip():
    s=progress_settings();p=dict(depth_m=.010,time_s=0,command_deg=100.,torsion_nm=.3)
    now=dict(p,time_s=1,command_deg=102.)
    assert classify_observed_progress(p,dict(now,depth_m=.01004),s)=='ADVANCING'
    assert classify_observed_progress(p,now,s)=='RECOVERABLE_NO_PROGRESS'
    assert classify_observed_progress(p,dict(now,grasp_relation_error_m=.00015),s)=='RECOVERABLE_GRIP_SLIP'
    assert classify_observed_progress(p,dict(now,torsion_nm=1.),s)=='RECOVERABLE_CONTACT_STALL'
    near=dict(p,depth_m=.01460)
    assert classify_observed_progress(near,dict(now,depth_m=.014602,torsion_nm=.7),s)=='VISUAL_SEATING_CANDIDATE'
    assert classify_observed_progress(near,dict(now,depth_m=.0147,torsion_nm=.7),s)=='OBSERVED_DEPTH_OVERRUN'
    assert classify_observed_progress(dict(p,command_deg=0),dict(now,command_deg=2),s)=='OBSERVING'
