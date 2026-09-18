"""Causal observation availability in simulation time, independent of wall speed."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class DelayedObservation:
    sample_time_s: float
    sensor_delay_s: float
    estimation_delay_s: float
    payload: object

    def __post_init__(self):
        values=(self.sample_time_s,self.sensor_delay_s,self.estimation_delay_s)
        if not all(map(math.isfinite,values)) or min(values)<0:
            raise ValueError('Finite nonnegative observation times required')

    @property
    def available_time_s(self):
        return self.sample_time_s+self.sensor_delay_s+self.estimation_delay_s

    def ready(self,simulation_time_s):
        if not math.isfinite(simulation_time_s):raise ValueError('Finite simulation time required')
        return simulation_time_s>=self.available_time_s


def execute_computation_delay(world, stepper, physics_dt_s, calculation_wall_s, advance):
    """Pay a completed planning/estimation delay with the caller's protected hold."""
    dt,delay=float(physics_dt_s),float(calculation_wall_s)
    if not math.isfinite(dt) or dt<=0 or not math.isfinite(delay) or delay<0:
        raise ValueError('A positive physics interval and finite nonnegative calculation delay are required')
    sample_time=float(world.current_time);first=int(stepper.step_index)
    count=math.ceil(delay/dt)
    world.play()
    for _ in range(count):
        previous_time=float(world.current_time);previous_step=int(stepper.step_index)
        advance()
        if float(world.current_time)<=previous_time or int(stepper.step_index)<=previous_step:
            raise RuntimeError('The protected computation hold did not advance physical time and steps')
    consumed=float(world.current_time)
    if consumed+dt*1e-6<sample_time+delay:
        raise RuntimeError('The calculation result became available before its physical delay elapsed')
    return {'calculation_wall_s':delay,'calculation_finished_at_sim_time_s':sample_time,
        'available_time_s':sample_time+delay,'consumed_time_s':consumed,
        'first_step':first,'last_step':int(stepper.step_index),
        'intervening_physics_steps':int(stepper.step_index)-first,
        'existing_force_speed_and_geometry_checks_retained':True,
        'object_or_contact_truth_used':False}
