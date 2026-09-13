"""Bounded relinearization keeps the measured CAD closure and native force ratio."""
from pathlib import Path
import sys,types
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'isaac'))
from te_hand_fourbar import update_fourbar_tangents
from kcg_connector.grasp.robust.finger_fourbar import load_finger_fourbars


def test_adaptive_updates_preserve_closure_across_motion_and_hold(monkeypatch):
    class Attr:
        def __init__(self):self.value=None;self.writes=0
        def Set(self,v):self.value=v;self.writes+=1
    class Api:
        def __init__(self):self.g,self.b=Attr(),Attr()
        def GetGearingAttr(self):return self.g
        def GetOffsetAttr(self):return self.b
    apis={};pxr=types.ModuleType('pxr')
    pxr.PhysxSchema=types.SimpleNamespace(PhysxMimicJointAPI=lambda p,a:apis.setdefault(p,Api()))
    monkeypatch.setitem(sys.modules,'pxr',pxr)
    stage=types.SimpleNamespace(GetPrimAtPath=lambda p:p)
    root=Path(__file__).resolve().parents[3]
    _,cs=load_finger_fourbars(root/'src/kcg_connector/config/hand_fourbar_20260912.json')
    cache={};updates=0;largest=0.
    for q in np.r_[np.linspace(.65,.75,1200),np.full(1200,.75)]:
        rows=update_fourbar_tangents(stage,'/robot',cs,{c.source_joint:q for c in cs.values()},
            cache=cache,closure_tolerance_m=1e-8,maximum_slope_error=1e-4,lookahead_source_angle_rad=3/960)
        for row in rows:
            updates+=row['relinearized'];largest=max(largest,row['predicted_rod_error_m'])
            follower=next(k for k,c in cs.items() if c.source_joint==row['source'])
            assert row['slope']==-apis['/robot/'+follower].g.value
            assert abs(row['slope']-row['exact_slope'])<=1e-4+1e-7
    assert largest<1e-8
    assert updates<2400*3/20
