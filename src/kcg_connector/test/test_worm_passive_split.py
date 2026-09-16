"""Work, lock and driven release of the finite elastic transmission substeps."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"isaac"))
from te_worm_drive import WormDrive,WormReference


@pytest.mark.parametrize('load',[3.,-1.])
@pytest.mark.parametrize('velocity',[-.005,0.,.005])
def test_motor_reference_inverse_reaches_both_directions_through_existing_self_lock(load,velocity):
    from te_worm_drive import position_reference_for_input_velocity
    motor=drive(.7);q=.7-load/120.;h=1/960
    target,request=position_reference_for_input_velocity(motor,q,velocity,h,264.,4.4,.6,.9)
    motor.prepare_position(q,0.,target,h,stiffness=264.,damping=4.4)
    result=motor.complete(q,0.)
    assert result['input_velocity']==pytest.approx(velocity,abs=1e-11)
    assert abs(result['input_effort'])<=motor.reference.input_effort_boundary
    assert abs(result['motor_substep_energy_residual_j'])<1e-12
    assert request['output_position_or_velocity_written'] is False


def test_smaller_motor_position_reference_can_still_be_inside_worm_stiction_band():
    from te_worm_drive import position_reference_for_input_velocity
    motor=drive(.7);q=.675;h=1/960
    motor.prepare_position(q,0.,.716,h,stiffness=264.,damping=4.4)
    held=motor.complete(q,0.)
    assert held['input_velocity']==0.
    target,_=position_reference_for_input_velocity(motor,q,-.005,h,264.,4.4,.637,.917)
    motor.prepare_position(q,0.,target,h,stiffness=264.,damping=4.4)
    released=motor.complete(q,0.)
    assert released['input_velocity']<0.
    assert released['input_substep_spring_effort_nm']<held['input_substep_spring_effort_nm']


def test_inverse_command_regulates_contact_load_after_both_load_changes():
    from te_worm_drive import position_reference_for_input_velocity
    target_moment=2.14;ratio=.7;contact_q=.65;h=1/960;K=120.;I=.0002;D=2.
    q=contact_q+target_moment/(ratio*K);motor=drive(q+target_moment/(ratio*K));w=0.
    filtered=target_moment;input_velocities=[];errors=[]
    for step in range(1800):
        if step==300:contact_q-=.001
        if step==1000:contact_q+=.002
        measured=ratio*K*(q-contact_q)
        filtered+=h/(.05+h)*(measured-filtered)
        error=target_moment-filtered
        desired=0. if abs(error)<=.02 else np.clip(error/(120/6),-.18,.18)
        ref,_=position_reference_for_input_velocity(motor,q,desired,h,264.,4.4,.6,.9)
        law=motor.prepare_position(q,w,ref,h,stiffness=264.,damping=4.4)
        next_q=(I*(q+h*w)/h**2+D*q/h+K*law['position_target']+K*contact_q)/(I/h**2+D/h+2*K)
        w=(next_q-q)/h;q=next_q
        row=motor.complete(q,w);input_velocities.append(row['input_velocity']);errors.append(measured-target_moment)
        assert abs(row['input_effort'])<=motor.reference.input_effort_boundary
        assert abs(120*(motor.input_angle-q))<3.5
    assert max(abs(np.asarray(errors)[850:1000]))<.022
    assert max(abs(np.asarray(errors)[1650:]))<.022
    assert min(input_velocities)<0<max(input_velocities)


def drive(z):
    return WormDrive(z,reference=WormReference(transmission_damping=0.,output_viscosity=2.),
                     integration="passive_split")


@pytest.mark.parametrize("load",[-.9,-.3,0.,.3,.9])
def test_output_load_cannot_turn_unpowered_input(load):
    motor=drive(.5)
    q=.5-load/120.
    law=motor.prepare(q,0.,0.,1/960)
    r=motor.complete(q,0.)
    assert law["position_target"]==.5
    assert r["input_angle"]==.5
    assert r["motor_work_j"]==0.
    assert r["friction_heat_j"]==0.


@pytest.mark.parametrize("initial_load",[-.3,0.,.3])
@pytest.mark.parametrize("effort",[-1.2,-.2,0.,.2,1.2])
def test_motor_step_obeys_exact_friction_and_does_not_create_energy(initial_load,effort):
    motor=drive(.5+initial_load/120.)
    motor.prepare(.5,0.,effort,1/960)
    r=motor.complete(.5,0.)
    v=r["input_velocity"];force=r["friction_effort"];tau=r["input_substep_spring_effort_nm"]
    assert effort-v-tau-force==pytest.approx(0.,abs=2e-13)
    assert force*v>=-1e-14
    if v:
        assert force==pytest.approx(np.sign(v)*1.2*abs(tau),abs=2e-13)
    else:
        assert abs(force)<=1.2*abs(tau)+1e-13
    assert abs(r["motor_substep_energy_residual_j"])<2e-14


def test_complete_elastic_robot_system_is_passive_and_can_actively_release():
    motor=drive(.5);q=.5;w=0.;I=.0002;h=1/960;K=120.;D=2.
    energized_motion=[]
    for step in range(1920):
        u=0. if step<480 or step>=1440 else (1.2 if step<960 else -1.2)
        external=.3 if step<960 else -.3
        z0=motor.input_angle;q0=q
        old_energy=.5*I*w*w+.5*K*(z0-q)**2
        law=motor.prepare(q,w,u,h)
        w=(I*w+h*(K*(law["position_target"]-q)+external))/(I+h*D+h*h*K)
        q+=h*w
        r=motor.complete(q,w)
        energy=.5*I*w*w+.5*K*(motor.input_angle-q)**2
        assert energy-old_energy-r["motor_work_j"]-external*(q-q0)<1e-12
        if u==0.:
            assert motor.input_angle==z0
        elif step in (959,1439):
            energized_motion.append(motor.input_angle-z0)
    assert energized_motion[0]>0.
    assert energized_motion[1]<0.


@pytest.mark.parametrize("target",[.2,.49,.5,.51,.8])
def test_implicit_motor_pd_is_bounded_and_closes_its_actual_work_balance(target):
    motor=drive(.5);q=.499;h=1/960
    motor.prepare_position(q,0.,target,h,stiffness=264.,damping=4.4)
    r=motor.complete(q,0.)
    raw=264.*(target-r["input_angle"])-4.4*r["input_velocity"]
    assert r["input_effort"]==pytest.approx(np.clip(raw,-7.7,7.7),abs=1e-12)
    assert abs(r["motor_substep_energy_residual_j"])<1e-13
    assert r["friction_heat_j"]>=0.
    assert r["motor_position_command_rad"]==target


@pytest.mark.parametrize("load",[-.9,.9])
def test_motor_pd_holds_its_set_position_against_backdrive(load):
    motor=drive(.5);q=.5-load/120.
    motor.prepare_position(q,0.,.5,1/960,stiffness=264.,damping=4.4)
    r=motor.complete(q,0.)
    assert r["input_angle"]==.5
    assert r["input_effort"]==0.


def test_motor_position_reversal_can_release_a_loaded_output():
    motor=drive(.5025)
    law=motor.prepare_position(.5,0.,.49,1/960,stiffness=264.,damping=4.4)
    r=motor.complete(.5,0.)
    assert law["position_target"]<.5025
    assert r["input_velocity"]<0.
    assert r["input_effort"]<0.


@pytest.mark.parametrize("readback,output_delta,effort_violation,elastic_violation",[
    (3.500000476837158,0.,False,False),
    (3.5001,0.,True,False),
    (3.5,-.001,False,True),
])
def test_native_cap_roundoff_preserves_both_real_effort_and_elastic_guards(
        readback,output_delta,effort_violation,elastic_violation):
    # Captured adjacent F3 samples34994/34995 from the actual assisted turn17.
    # The first case previously stopped with only two float32 ULPs of excess.
    motor=drive(.6971159962427167)
    law=motor.prepare_position(.668010950088501,.015775898471474648,
        .7253195615420972,1/960,stiffness=264.,damping=4.4)
    result=motor.complete(.6680055260658264+output_delta,.03799189627170563,
        observed_drive_effort=readback)
    assert law["max_effort"]==3.5
    assert result["transmission_effort"]==readback
    assert result["drive_saturation"] is effort_violation
    assert result["elastic_effort_boundary_exceeded"] is elastic_violation


@pytest.mark.parametrize('output_angle,output_velocity,elastic_excess',[
    (.7270070910453796,-.18478606641292572,False),
    (.7200,0.,True),
])
def test_native_capped_pd_estimate_preserves_independent_elastic_guard(output_angle,output_velocity,elastic_excess):
    # Joint-reaction projections are not supplied as an isolated drive sensor.
    motor=WormDrive(.7578673218175987,reference=WormReference(transmission_damping=0.,
        output_viscosity=2.,output_active_effort_reference=4.,transmission_effort_boundary=4.),
        integration='passive_split')
    law=motor.prepare_position(.7271814942359924,-.17669999599456787,
        motor.input_angle,1/960,stiffness=264.,damping=4.4)
    result=motor.complete(output_angle,output_velocity,native_capped_drive=True)
    assert law['max_effort']==4.
    assert abs(result['transmission_effort'])<=4.
    assert result['observed_drive_effort_nm'] is None
    assert result['drive_effort_source']=='CAPPED_ENDPOINT_PD_ESTIMATE_NOT_SENSOR'
    assert result['output_drive_work_is_estimate'] is True
    assert result['elastic_effort_boundary_exceeded'] is elastic_excess


def test_captured_constraint_reaction_must_not_be_a_motor_drive_measurement():
    motor=WormDrive(.7552633369376072,reference=WormReference(transmission_damping=0.,
        output_viscosity=2.,output_active_effort_reference=4.,transmission_effort_boundary=4.),
        integration='passive_split')
    before=motor.input_angle-3.0205832053400083/120
    motor.prepare_position(before,0.,motor.input_angle,1/960,stiffness=264.,damping=4.4)
    result=motor.complete(.7291678786277771,.16318921744823456,native_capped_drive=True)
    assert result['transmission_effort']==pytest.approx(2.8050765622831388)
    assert result['observed_drive_effort_nm'] is None
    assert not result['elastic_effort_boundary_exceeded']


def test_captured_open_target_sticks_but_friction_aware_release_moves():
    from te_worm_drive import position_reference_for_input_velocity
    reference=WormReference(transmission_damping=0.,output_viscosity=2.,
        output_active_effort_reference=4.,transmission_effort_boundary=4.)
    z=.6653666302891776;q=.6970000267028809;h=1/960
    stuck=WormDrive(z,reference=reference,integration='passive_split')
    stuck.prepare_position(q,0.,.6345330734252933,h,stiffness=264.,damping=4.4)
    assert stuck.complete(q,0.)['input_velocity']==0.
    moving=WormDrive(z,reference=reference,integration='passive_split')
    target,request=position_reference_for_input_velocity(moving,q,-.15,h,264.,4.4,0.,1.3963,clip_to_feasible=True)
    moving.prepare_position(q,0.,target,h,stiffness=264.,damping=4.4)
    row=moving.complete(q,0.)
    assert row['input_velocity']==pytest.approx(-.15,abs=1e-11)
    assert abs(request['requested_motor_effort_nm'])<=8.8


def test_speed_request_saturates_without_raising_either_effort_limit():
    from te_worm_drive import position_reference_for_input_velocity
    motor=drive(.7361341063890952);q=.7070005536079407;h=1/960
    target,request=position_reference_for_input_velocity(motor,q,.0088024469,h,264.,4.4,.6,.9,clip_to_feasible=True)
    assert request['input_velocity_limited']
    assert 0<request['desired_input_velocity_rad_s']<.0088024469
    assert abs(request['requested_motor_effort_nm'])<=7.7
    motor.prepare_position(q,0.,target,h,stiffness=264.,damping=4.4)
    row=motor.complete(q,0.)
    assert row['input_velocity']==pytest.approx(request['desired_input_velocity_rad_s'],abs=1e-11)
