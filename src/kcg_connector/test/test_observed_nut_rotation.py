"""Pure checks for the new bounded interval and encoder-angle calculations."""
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'isaac'))
from te_observed_nut_rotation import bounded_segment_angle_deg,encoder_twist_rad


def test_interval_respects_angle_speed_acceleration_and_sampling_time():
    for remaining in (-20.,-.03,20.,.03):
        for horizon in (.5,.2,.04):
            angle=bounded_segment_angle_deg(remaining,2.,horizon,13.2)
            duration=max(1.875*abs(angle)/2.,np.sqrt((10/np.sqrt(3))*abs(angle)/13.2))
            assert np.sign(angle)==np.sign(remaining)
            assert abs(angle)<=min(.5,abs(remaining))+1e-12
            assert duration<=horizon+1e-12
            assert 1.875*abs(angle)/duration<=2.+1e-12
            assert (10/np.sqrt(3))*abs(angle)/duration**2<=13.2+1e-10


def test_encoder_twist_unwraps_without_reverse_at_pi_boundary():
    axis=np.array([0.,0.,1.]);previous=0.
    reference=Rotation.from_euler('xy',[179.,.4],degrees=True).as_matrix()
    for degrees in np.linspace(0.,-360.,721):
        rotation=Rotation.from_rotvec(axis*np.deg2rad(degrees)).as_matrix()@reference
        actual=encoder_twist_rad(rotation,reference,axis,previous)
        assert abs(actual-np.deg2rad(degrees))<1e-10
        assert actual<=previous+1e-10
        previous=actual


def test_pure_alignment_swing_is_not_reported_as_screw_rotation():
    reference=Rotation.from_euler('xyz',[15.,20.,30.],degrees=True).as_matrix()
    swing=Rotation.from_rotvec([.003,-.002,0.]).as_matrix()
    assert abs(encoder_twist_rad(swing@reference,reference,np.array([0.,0.,1.])))<1e-12


def test_observation_boundaries_preserve_wrench_state_and_global_force_ramp(monkeypatch,tmp_path):
    import types,json
    from te_observed_nut_rotation import run_observed_nut_rotation
    def module(name,**values):
        m=types.ModuleType(name);m.__dict__.update(values);monkeypatch.setitem(sys.modules,name,m);return m
    usd=module('omni.usd',get_context=lambda:types.SimpleNamespace(get_stage=lambda:None))
    core=module('omni.replicator.core');replicator=module('omni.replicator',core=core)
    module('omni',usd=usd,replicator=replicator)
    def ready(value):
        if isinstance(value,np.ndarray):return value.tolist()
        if isinstance(value,np.generic):return value.item()
        if isinstance(value,dict):return {k:ready(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return [ready(v) for v in value]
        return value
    names=[f'j{i}' for i in range(7)]
    module('te_foundationpose_handoff_runtime',_json_ready=ready,
        MOVEIT_SOFT_ARM_BOUNDS_RAD={n:(-5.,5.) for n in names},control=types.SimpleNamespace(ARM_JOINT_NAMES=names))
    class World:
        current_time=0.;playing=False
        def pause(self):self.playing=False
        def play(self):self.playing=True
        def is_playing(self):return self.playing
    world=World();dt=1/240;mass=.0625
    class Model:
        def forward_kinematics(self,q,**kwargs):
            H=np.eye(4);H[:3,:3]=Rotation.from_rotvec(q[3:6]).as_matrix()@np.diag([1.,-1.,-1.]);H[:3,3]=np.asarray(q[:3])+[0,0,.42]
            return {'handbase_link':H}
        def geometric_jacobian(self,*args,**kwargs):
            J=np.zeros((6,7));J[:6,:6]=np.eye(6);return J
    model=Model();ft=types.SimpleNamespace(samples=[],task_rotation_world=np.eye(3),planned_contact_force_time_constant_s=.05)
    def sample(step,phase,q):
        H=model.forward_kinematics(q)['handbase_link']
        return dict(step=step,phase=phase,handbase_rotation_world_row_major=H[:3,:3].tolist(),handbase_position_world_m=H[:3,3].tolist(),
            gravity_and_dynamic_compensated_task_wrench=[0,0,.2-mass*9.81,0,0,0],
            dynamic_inertia_prediction={'ready':True,'applied_to_safety_residual':True})
    q=np.zeros(11)
    ft.samples=[sample(i,'key_probe_nut_grip_hold',q) for i in range(-120,0)]
    class Stepper:
        step_index=0;abort_reason=None;latest=(q.copy(),q.copy(),q.copy());targets=[]
        def advance(self,phase,arm,hand):
            self.latest=(np.r_[arm,hand],np.zeros(11),np.zeros(11));self.targets.append(self.latest[0].copy())
            ft.samples.append(sample(self.step_index,phase,self.latest[0]));self.step_index+=1;world.current_time=self.step_index*dt
    stepper=Stepper()
    def observe(*args):
        B=np.diag([1.,-1.,-1.,1.]);B[:3,3]=[0,0,-.01]
        return dict(position_and_axis_measured=True,world_from_plug_five_dof=B.tolist(),online_object_or_contact_truth_used=False,physics_time_s=world.current_time)
    module('te_body_socket_observation',observe_released_plug_from_rgbd=observe)
    runtime=dict(world=world,inputs=types.SimpleNamespace(robot_model=model),nail_body_ft_auditor=ft,
        scene={'gravity_m_s2':-9.81},body_assembly_scene={'report':{'representative_inner_thread':True}},
        nut_regrasp_geometry_check=lambda *a,**k:None,nut_regrasp_locate_visual_bounds=lambda *a:None)
    settings=dict(rotation_about_socket_plus_z_deg=-1.,maximum_rotation_speed_deg_s=2.,axial_settle_duration_s=0.,post_rotation_hold_s=2.,
        axial_force_reference_n=.2,turn_axial_force_reference_n=mass*9.81,turn_force_reference_ramp_duration_s=.5,
        contact_estimate_filter_time_constant_s=.05,payload_mass_kg=mass,payload_com_body_m=[0,0,0],
        axial_admittance_m_per_n_s=.0015,maximum_axial_speed_m_s=.0003,maximum_axial_travel_m=.004,maximum_arm_speed_rad_s=.075,
        planar_force_admittance={'enabled':False,'freeze_after_preparation':True},pre_turn_torsional_compliance={'enabled':False},
        pre_turn_visual_alignment={'enabled':False},stops={'axial_force_n':None,'lateral_force_n':5.,'bending_moment_nm':.2,'torsional_moment_nm':.25},
        guided_feature_fit={'source_pin_tip_z_body_m':-.0094615,'pin_pattern_radius_upper_m':.02,'pin_lateral_displacement_budget_m':.000065,'key_lateral_displacement_budget_m':.0001},
        in_turn_feedback={'enabled':True})
    grip=dict(completed=True,fixed_arm_target_rad=[0.]*7,final_hand_target_rad=[0.]*4,new_grasp_effort_tare_nm=[0.]*4,
        finite_preload_bounds_rad=[[-1.]*4,[1.]*4],effort_reference_nm=[.1]*3,visual_alignment_allowance_from_quarter_body_clearance_m=.001)
    result=run_observed_nut_rotation(tmp_path,runtime,stepper,{'physics_dt_s':dt,'finger_maximum_speed_rad_s':.18,'hand_stiffness':12.},grip,np.eye(4),settings,tmp_path/'run')
    assert result['completed'],result.get('failure_reason')
    assert result['physical_thread_progress_verified'] is False
    continued=[x['result']['contact_estimate_filter'] for x in result['intervals'][1:]]
    assert all(x['initialization_source']=='CONTINUED_SAME_EPISODE_FILTER_STATE' for x in continued)
    assert all(x['loaded_wrench_rezeroed'] is False for x in continued)
    controls=[json.loads(line) for line in Path(result['control_samples_file']).read_text().splitlines()]
    reference=np.array([r['axial_force_reference_n'] for r in controls])
    assert np.diff(reference).min()>=-1e-12
    assert abs(reference[-1]-mass*9.81)<1e-9
    assert all(i['requested_interval_rotation_deg']<=0 for i in result['intervals'])
    assert np.max(abs(np.asarray(stepper.targets)[:,7:]))==0
    assert sum((i['last_step']-i['first_step'])*dt for i in result['intervals'] if i['mode']=='HOLD')>=2.
