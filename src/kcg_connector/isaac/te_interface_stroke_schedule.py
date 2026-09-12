"""Finite wrist strokes with real finger opening and open-hand reindexing.

This schedule contains no object pose writes or contact-truth feedback. Angles
are robot targets, never evidence of actual connector rotation.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class StrokePhase:
    start: float
    duration: float
    name: str
    angle0: float
    angle1: float
    stroke: int


class InterfaceStrokeSchedule:
    def __init__(self, total_deg, stroke_deg, turn_duration_s, *, release=True,
                 reindex_speed_rad_s=.8, preload_duration_s=2.):
        if not (0 < total_deg <= 360 and 0 < stroke_deg <= 120 and 0 < turn_duration_s <= 20):
            raise ValueError('Finite local stroke range exceeded')
        if not (0<reindex_speed_rad_s<=1.6 and .5<=preload_duration_s<=2.):
            raise ValueError('Finite regrasp timing references exceeded')
        self.phases = []
        time = 0.

        def add(name, duration, a0, a1, stroke):
            nonlocal time
            self.phases.append(StrokePhase(time, duration, name, a0, a1, stroke))
            time += duration

        add('grip', 3., 0., 0., 0)
        left = float(total_deg)
        index = 0
        while left > 1e-8:
            amount = min(left, stroke_deg)
            add('turn', turn_duration_s * amount / total_deg, 0., amount, index)
            add('loaded_hold', .2, amount, amount, index)
            left -= amount
            if left > 1e-8:
                # Open fully before moving the wrist back. A multiple of the
                # outer nut's flute period is chosen by the experiment caller.
                add('regrasp_open', .6, amount, amount, index)
                add('reindex', max(1., 1.875 * math.radians(amount) / reindex_speed_rad_s), amount, 0., index)
                add('regrasp_close', .8, 0., 0., index + 1)
                # Match the initial grip's1.5s ramp plus0.5s settling.
                # Freezing the drive immediately at the end of the ramp
                # otherwise leaves about8-10% of the load target unmet.
                add('regrasp_preload', preload_duration_s, 0., 0., index + 1)
            elif release:
                add('opening', .8, amount, amount, index)
                add('free_hold', .5, amount, amount, index)
            index += 1
        self.duration = time

    def sample(self, elapsed):
        phase = next((p for p in self.phases if elapsed < p.start + p.duration), self.phases[-1])
        x = min(1., max(0., (elapsed - phase.start) / phase.duration))
        smooth = 10*x**3 - 15*x**4 + 6*x**5
        angle = phase.angle0 + (phase.angle1-phase.angle0)*smooth
        rate = (phase.angle1-phase.angle0)*30*x*x*(1-x)**2/phase.duration
        return phase, math.radians(angle), math.radians(rate), smooth

    def as_dict(self):
        return {'duration_s': self.duration, 'phases': [vars(p) for p in self.phases],
                'object_pose_writes': False, 'commanded_angles_are_not_actual_nut_rotation': True}
