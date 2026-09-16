"""The wrist may return to zero without withdrawing accumulated screw travel."""
import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'isaac'))
from te_interface_stroke_schedule import InterfaceStrokeSchedule
from te_nut_motion import released_grip_readiness


def schedule():
    return InterfaceStrokeSchedule(360.,90.,13.5,grip_duration_s=3.5,
        relax_duration_s=.2,opening_duration_s=.8,open_hold_duration_s=.5,
        preload_duration_s=2.,reindex_speed_rad_s=1.6,
        seating_alignment={'first_turn_deg':65.,'first_turn_duration_s':2.4375,
                           'alignment_hold_s':1.2,'final_turn_duration_s':1.5})


def test_wrist_return_preserves_all_previous_turn_progress():
    s=schedule()
    for p in s.phases:
        if p.name=='reindex':
            assert p.angle0==90. and p.angle1==0.
            for blend in (0.,.5,1.):
                assert s.commanded_progress(p,blend)==pytest.approx(math.radians(90.*(p.stroke+1)))


def test_progress_is_continuous_including_regrasp_and_alignment():
    s=schedule()
    for before,after in zip(s.phases,s.phases[1:]):
        assert before.progress1==after.progress0
        assert before.start+before.duration==pytest.approx(after.start)
    assert s.phases[-1].progress1==360.
    assert s.phases[0].duration==3.5


def test_alignment_only_occurs_in_last_stroke_and_retains_progress():
    s=schedule();holds=[p for p in s.phases if p.name=='alignment_hold']
    assert len(holds)==1
    assert holds[0].stroke==3 and holds[0].progress0==holds[0].progress1==335.


def test_wrist_return_requires_completed_relax_open_and_observation_phases():
    phases=schedule().phases
    for i,p in enumerate(phases):
        if p.name=='reindex':
            assert [a.name for a in phases[i-3:i]]==['regrasp_relax','regrasp_open','regrasp_open_hold']
            assert phases[i+1].name=='regrasp_close'
            assert phases[i+2].name=='regrasp_preload'


def test_each_stroke_can_restore_alignment_before_releasing_the_grip():
    s=InterfaceStrokeSchedule(360.,90.,13.5,seating_alignment={
        'every_stroke':True,'first_turn_deg':65.,'first_turn_duration_s':2.4375,
        'alignment_hold_s':1.2,'final_turn_duration_s':1.5})
    holds=[p for p in s.phases if p.name=='alignment_hold']
    assert [p.stroke for p in holds]==[0,1,2,3]
    assert [p.angle0 for p in holds]==[65.]*4
    assert [p.progress0 for p in holds]==[65.,155.,245.,335.]
    for before,after in zip(s.phases,s.phases[1:]):assert before.progress1==after.progress0


def test_completed_turn_can_unload_immediately_without_continued_loaded_tracking():
    s=InterfaceStrokeSchedule(360.,90.,13.5,relax_duration_s=.2,post_turn_hold_s=0.)
    assert all(p.name!='loaded_hold' for p in s.phases)
    for i,p in enumerate(s.phases):
        if p.name=='turn':
            assert s.phases[i+1].name in ('regrasp_relax','torque_relaxation')
            assert s.phases[i+1].start==pytest.approx(p.start+p.duration)


@pytest.mark.parametrize('open_reached,moments',[(False,[0.,0.,0.]),(True,[0.,.03,0.]),(True,[0.,float('nan'),0.])])
def test_an_open_command_alone_or_residual_grip_load_cannot_start_reindex(open_reached,moments):
    assert not released_grip_readiness(open_reached,moments,.02)['ready']
