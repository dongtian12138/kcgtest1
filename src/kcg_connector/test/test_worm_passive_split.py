"""Work, lock and driven release of the finite elastic transmission substeps."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"isaac"))
from te_worm_drive import WormDrive,WormReference


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
