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
