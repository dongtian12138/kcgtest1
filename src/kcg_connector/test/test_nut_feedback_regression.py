"""Exercise the real turn loop with an affine loaded test plant, without Isaac."""
from pathlib import Path
import sys,types,json
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'isaac'))


def setup_plant(monkeypatch,*,seated=False,moving_body=False):
    def module(name,**items):
        value=types.ModuleType(name);value.__dict__.update(items);monkeypatch.setitem(sys.modules,name,value);return value
    usd=module('omni.usd',get_context=lambda:types.SimpleNamespace(get_stage=lambda:None))
    rep=module('omni.replicator.core');replicator=module('omni.replicator',core=rep);module('omni',usd=usd,replicator=replicator)
    def ready(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        if isinstance(v,dict):return {k:ready(x) for k,x in v.items()}
        if isinstance(v,(tuple,list)):return [ready(x) for x in v]
        return v
    names=[f'j{i}' for i in range(7)]
    module('te_foundationpose_handoff_runtime',_json_ready=ready,
        MOVEIT_SOFT_ARM_BOUNDS_RAD={n:(-5.,5.) for n in names},
        control=types.SimpleNamespace(ARM_JOINT_NAMES=names),_close_rgbd_resources=lambda r:r.clear())
    class World:
        current_time=0.;playing=False
        def play(self):self.playing=True
        def pause(self):self.playing=False
        def is_playing(self):return self.playing
    world=World();dt=1/240
    class Model:
        def forward_kinematics(self,q,**kwargs):
            H=np.eye(4);H[:3,3]=np.asarray(q[:3])+[0,0,.42]
            H[:3,:3]=Rotation.from_rotvec(q[3:6]).as_matrix()@np.diag([1.,-1.,-1.]);return {'handbase_link':H}
        def geometric_jacobian(self,*a,**k):
            J=np.zeros((6,7));J[:6,:6]=np.eye(6);return J
    model=Model();ft=types.SimpleNamespace(samples=[],task_rotation_world=np.eye(3),planned_contact_force_time_constant_s=.05)
    def sensor(step,phase,q):
        H=model.forward_kinematics(q)['handbase_link']
        return dict(step=step,phase=phase,handbase_rotation_world_row_major=H[:3,:3].tolist(),
            handbase_position_world_m=H[:3,3].tolist(),gravity_and_dynamic_compensated_task_wrench=[0,0,0,0,0,.7 if seated else 0],
            dynamic_inertia_prediction={'ready':True,'applied_to_safety_residual':True},active_targets_rad=list(q))
    q=np.zeros(11);ft.samples=[sensor(i,'key_probe_nut_grip_hold',q) for i in range(-120,0)]
    class Stepper:
        step_index=0;abort_reason=None;latest=(q.copy(),q.copy(),q.copy());targets=[]
        def advance(self,phase,arm,hand,**kwargs):
            actual=np.r_[arm,hand].copy();actual[0]-=.0001
            self.latest=(actual,np.zeros(11),np.zeros(11));self.targets.append(np.r_[arm,hand].copy())
            ft.samples.append(sensor(self.step_index,phase,actual));self.step_index+=1;world.current_time=self.step_index*dt
    stepper=Stepper()
    def observe(*args):
        B=np.diag([1.,-1.,-1.,1.]);B[:3,3]=[0,0,-.01460 if seated else -.01]
        if moving_body:B[0,3]=min(.0001,.00005*world.current_time)
        return dict(position_and_axis_measured=True,world_from_plug_five_dof=B.tolist(),physics_time_s=world.current_time,
            online_object_or_contact_truth_used=False)
    module('te_body_socket_observation',observe_released_plug_from_rgbd=observe,observe_tracked_plug_from_rgbd=observe,
        observe_current_plug_from_rgbd=observe)
    runtime=dict(world=world,inputs=types.SimpleNamespace(robot_model=model),nail_body_ft_auditor=ft,
        scene={'gravity_m_s2':-9.81},body_assembly_scene={'report':{'representative_inner_thread':True}},
        nut_regrasp_geometry_check=lambda *a,**k:None,nut_regrasp_locate_visual_bounds=lambda *a:None)
    settings=dict(rotation_about_socket_plus_z_deg=-2.,maximum_rotation_speed_deg_s=2.,maximum_rotation_acceleration_deg_s2=.35,
        rotation_profile='trapezoid',axial_settle_duration_s=0.,post_rotation_hold_s=1.,
        axial_force_reference_n=.0625*9.81,turn_axial_force_reference_n=.0625*9.81,turn_force_reference_ramp_duration_s=.5,
        contact_estimate_filter_time_constant_s=.05,payload_mass_kg=.0625,payload_com_body_m=[0,0,0],
        axial_admittance_m_per_n_s=.0015,maximum_axial_speed_m_s=.0003,maximum_axial_travel_m=.004,maximum_arm_speed_rad_s=.075,
        arm_kinematic_reference='commanded_pose',measured_tracking={'enabled':True,'maximum_lateral_error_m':.00025},
        regulate_finger_effort_during_rotation=False,regulate_finger_effort_during_preparation=False,
        planar_force_admittance={'enabled':False,'freeze_after_preparation':True},pre_turn_torsional_compliance={'enabled':False},
        pre_turn_visual_alignment={'enabled':False},stops={'axial_force_n':None,'lateral_force_n':5.,'bending_moment_nm':.2,'torsional_moment_nm':4.6})
    grip=dict(completed=True,fixed_arm_target_rad=[0.]*7,final_hand_target_rad=[0.]*4,new_grasp_effort_tare_nm=[0.]*4,
        finite_preload_bounds_rad=[[-1.]*4,[1.]*4],effort_reference_nm=[.1]*3,visual_alignment_allowance_from_quarter_body_clearance_m=.001)
    return runtime,stepper,settings,grip,dt


def test_visual_motion_moves_world_reference_without_redefining_desired_grip(monkeypatch,tmp_path):
    from te_body_nut_rotation import _run_body_nut_rotation_interval
    runtime,stepper,settings,grip,dt=setup_plant(monkeypatch,moving_body=True)
    settings['visual_progress']=dict(enabled=True,observation_period_s=.25,maximum_grasp_relation_error_m=.0002,
        nominal_seated_depth_m=.014605,visual_seating_tolerance_m=.00002,stable_depth_tolerance_m=.000005,
        seating_minimum_torque_nm=.4,seating_confirmations=2,minimum_loaded_command_deg=40.,
        minimum_progress_check_angle_deg=.75,minimum_observed_progress_m=.00001,
        reference_following=dict(enabled=True,maximum_lateral_motion_m=.0005,maximum_axis_motion_deg=1.,
            maximum_lateral_speed_m_s=.00015,maximum_axis_speed_rad_s=.002,smoothing_time_s=.2))
    result=_run_body_nut_rotation_interval(tmp_path,runtime,stepper,{'physics_dt_s':dt,'finger_maximum_speed_rad_s':.18,'hand_stiffness':12.},
        grip,np.eye(4),settings,tmp_path/'follow')
    assert result['completed'],result.get('failure_reason')
    assert abs(stepper.latest[0][0]-.0001)<2e-6
    assert abs(stepper.targets[-1][0]-.0002)<2e-6
    assert result['hand_from_virtual_nut_axis_frame'][0][3]==0.


def test_turn_base_moment_feedback_can_relax_overloaded_fingers_and_close_underloaded_one(monkeypatch,tmp_path):
    from te_body_nut_rotation import _run_body_nut_rotation_interval
    runtime,stepper,settings,grip,dt=setup_plant(monkeypatch)
    targets=np.array([2.1,2.25,2.14]);disturbance=np.array([-.1,.05,.1])
    grip['effort_reference_nm']=targets.tolist()
    settings.update(regulate_finger_effort_during_rotation=True,finger_effort_regulation_time_constant_s=1/6,
                    finger_position_stiffness_reference_nm_rad=120.)
    runtime['nut_root_moment_observer']=types.SimpleNamespace(tare_reaction=np.zeros(3),tare_gravity=np.zeros(3),
        _system=lambda q:(None,None,np.zeros(3),None))
    stepper.latest[2][8:]=targets+disturbance
    advance=stepper.advance
    def plant(phase,arm,hand,**kwargs):
        advance(phase,arm,hand,**kwargs)
        stepper.latest[2][8:]=targets+disturbance+60*np.asarray(hand[1:])
    stepper.advance=plant
    result=_run_body_nut_rotation_interval(tmp_path,runtime,stepper,{'physics_dt_s':dt,'finger_maximum_speed_rad_s':.18,'hand_stiffness':12.},
        grip,np.eye(4),settings,tmp_path/'moments')
    assert result['completed'],result.get('failure_reason')
    assert np.max(np.abs(stepper.latest[2][8:]-targets))<1e-6
    final=np.asarray(result['final_hand_target_rad'])[1:]
    assert final[0]>0 and final[1]<0 and final[2]<0


@pytest.mark.parametrize('with_vision',[False,True])
def test_real_loop_corrects_loaded_pose_without_resetting_nominal_integrator(monkeypatch,tmp_path,with_vision):
    from te_body_nut_rotation import _run_body_nut_rotation_interval
    runtime,stepper,settings,grip,dt=setup_plant(monkeypatch)
    if with_vision:
        settings['visual_progress']=dict(enabled=True,observation_period_s=.5,maximum_grasp_relation_error_m=.0002,
            nominal_seated_depth_m=.014605,visual_seating_tolerance_m=.00002,stable_depth_tolerance_m=.000005,
            seating_minimum_torque_nm=.4,seating_confirmations=2,minimum_loaded_command_deg=40.,
            minimum_progress_check_angle_deg=.75,minimum_observed_progress_m=.00001)
    result=_run_body_nut_rotation_interval(tmp_path,runtime,stepper,{'physics_dt_s':dt,'finger_maximum_speed_rad_s':.18,'hand_stiffness':12.},
        grip,np.eye(4),settings,tmp_path/'feedback')
    assert result['completed'],result.get('failure_reason')
    assert abs(stepper.latest[0][0])<1e-7
    assert abs(stepper.targets[-1][0]-.0001)<1e-7
    assert np.max(abs(np.asarray(stepper.targets)[:,7:]))==0


def test_current_visual_seating_stops_before_the_remaining_angle_budget(monkeypatch,tmp_path):
    from te_body_nut_rotation import _run_body_nut_rotation_interval
    runtime,stepper,settings,grip,dt=setup_plant(monkeypatch,seated=True)
    settings['rotation_about_socket_plus_z_deg']=-90.
    settings['visual_progress']=dict(enabled=True,observation_period_s=.5,maximum_grasp_relation_error_m=.0002,
        nominal_seated_depth_m=.014605,visual_seating_tolerance_m=.00002,stable_depth_tolerance_m=.000005,
        seating_minimum_torque_nm=.4,seating_confirmations=2,minimum_loaded_command_deg=40.,
        minimum_progress_check_angle_deg=.75,minimum_observed_progress_m=.00001)
    result=_run_body_nut_rotation_interval(tmp_path,runtime,stepper,{'physics_dt_s':dt,'finger_maximum_speed_rad_s':.18,'hand_stiffness':12.},
        grip,np.eye(4),settings,tmp_path/'seat')
    assert result['completed'],result.get('failure_reason')
    assert result['seating_candidate']
    assert runtime['world'].current_time<2.
    assert abs(result['last_loaded_command_deg'])<1.
    assert result['physical_thread_progress_verified'] is False
