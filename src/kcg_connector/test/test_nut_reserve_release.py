"""Controlled unloading may follow a soft reserve stop, never a hard abort."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"isaac"))
from te_body_nut_reindex import can_unload_after_transmission_reserve_stop


def event():
    return {"completed":False,"failure_reason":"bounded thread pilot stop: TRANSMISSION_RESERVE",
        "last_step":100,"outer_abort_reason":None,
        "transmission_reserve_stop":{"hard_boundary_violated":False,"effort_margins_nm":[.4,.3,.04]}}


def test_soft_reserve_permits_observation_and_unload():
    assert can_unload_after_transmission_reserve_stop(event(),"key_probe_nut_rotation_turn",100,None)


@pytest.mark.parametrize("fault",["hard_transmission","wrist","past_step","other_phase","elastic_overload"])
def test_hard_or_stale_stops_cannot_enter_the_soft_release_path(fault):
    r=event();phase="key_probe_nut_rotation_turn";step=100;abort=None
    if fault=="hard_transmission":r["failure_reason"]="Finite hand transmission boundary: f3j2"
    if fault=="wrist":abort="WRIST_FT_RESULTANT_FORCE_ABORT"
    if fault=="past_step":step=101
    if fault=="other_phase":phase="key_probe_nut_contact"
    if fault=="elastic_overload":
        r["transmission_reserve_stop"].update(hard_boundary_violated=True,effort_margins_nm=[.4,.3,-.00024])
    assert not can_unload_after_transmission_reserve_stop(r,phase,step,abort)
