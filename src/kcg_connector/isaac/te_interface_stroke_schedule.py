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
    progress0: float = 0.
    progress1: float = 0.


class InterfaceStrokeSchedule:
    def __init__(self, total_deg, stroke_deg, turn_duration_s, *, release=True,
                 reindex_speed_rad_s=.8, preload_duration_s=2., grip_duration_s=3.,
                 relax_duration_s=0., opening_duration_s=.6, open_hold_duration_s=0.,
                 seating_alignment=None,post_turn_hold_s=.2):
        if not (0 < total_deg <= 360 and 0 < stroke_deg <= 120 and 0 < turn_duration_s <= 20):
            raise ValueError('Finite local stroke range exceeded')
        if not (0<reindex_speed_rad_s<=1.6 and .5<=preload_duration_s<=2.):
            raise ValueError('Finite regrasp timing references exceeded')
        if not (3.<=grip_duration_s<=4. and 0<=relax_duration_s<=.5
                and .6<=opening_duration_s<=1.5 and 0<=open_hold_duration_s<=.5
                and 0<=post_turn_hold_s<=.5):
            raise ValueError('Finite grip and release timing references exceeded')
        self.phases = []
        time = 0.;progress=0.

        def add(name, duration, a0, a1, stroke, p0=None, p1=None):
            nonlocal time
            self.phases.append(StrokePhase(time, duration, name, a0, a1, stroke,
                progress if p0 is None else p0,progress if p1 is None else p1))
            time += duration

        add('grip', grip_duration_s, 0., 0., 0)
        left = float(total_deg)
        index = 0
        while left > 1e-8:
            amount = min(left, stroke_deg)
            if seating_alignment and (seating_alignment.get('every_stroke',False) or left<=stroke_deg+1e-8):
                first=float(seating_alignment['first_turn_deg'])
                durations=[float(seating_alignment[k]) for k in
                    ('first_turn_duration_s','alignment_hold_s','final_turn_duration_s')]
                if not (0<first<amount and all(math.isfinite(t) and t>0 for t in durations)):
                    raise ValueError('Final stroke needs finite two-part seating motion')
                add('turn',durations[0],0.,first,index,progress,progress+first)
                add('alignment_hold',durations[1],first,first,index,progress+first,progress+first)
                add('turn',durations[2],first,amount,index,progress+first,progress+amount)
            else:
                add('turn', turn_duration_s * amount / total_deg, 0., amount, index,progress,progress+amount)
            progress+=amount
            if post_turn_hold_s:add('loaded_hold', post_turn_hold_s, amount, amount, index)
            left -= amount
            if left > 1e-8:
                # Open fully before moving the wrist back. A multiple of the
                # outer nut's flute period is chosen by the experiment caller.
                if relax_duration_s:add('regrasp_relax',relax_duration_s,amount,amount,index)
                add('regrasp_open', opening_duration_s, amount, amount, index)
                if open_hold_duration_s:add('regrasp_open_hold',open_hold_duration_s,amount,amount,index)
                add('reindex', max(1., 1.875 * math.radians(amount) / reindex_speed_rad_s), amount, 0., index)
                add('regrasp_close', .8, 0., 0., index + 1)
                # Match the initial grip's1.5s ramp plus0.5s settling.
                # Freezing the drive immediately at the end of the ramp
                # otherwise leaves about8-10% of the load target unmet.
                add('regrasp_preload', preload_duration_s, 0., 0., index + 1)
            elif release:
                if relax_duration_s:add('torque_relaxation',relax_duration_s,amount,amount,index)
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

    @staticmethod
    def commanded_progress(phase, blend):
        """Insertion command accumulates only loaded turns, never wrist return."""
        return math.radians(phase.progress0+(phase.progress1-phase.progress0)*blend)
