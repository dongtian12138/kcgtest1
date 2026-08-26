"""Regression checks for the corrected B0 object-side semantics."""

from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace

from kcg_connector.grasp.carts_v2.b0_surface_semantics import (
    _b0_role_arrays, b0_nominal_pickup_task_pass,
    b0_sampled_table_clearance_pass,
)
from kcg_connector.grasp.carts_v2.models import (
    HARD_FORBIDDEN, PRIMARY_GRIP, SECONDARY_GRIP,
)


def test_old_geometric_hard_faces_become_b0_load_bearing_preferences() -> None:
    semantic = "EXTERNALLY_FIRST_VISIBLE_PAD_REACHABLE_SURFACE"
    roles, allowed, reasons = _b0_role_arrays(
        (semantic, semantic, semantic), (semantic,), (),
        np.asarray((PRIMARY_GRIP, SECONDARY_GRIP, HARD_FORBIDDEN)),
    )
    assert roles.tolist() == [PRIMARY_GRIP, SECONDARY_GRIP, SECONDARY_GRIP]
    assert allowed.tolist() == [True, True, True]
    assert reasons.tolist() == [0, 0, 0]


def test_b0_preserves_explicit_protection_and_rejects_unknown_semantics() -> None:
    roles, allowed, reasons = _b0_role_arrays(
        ("EXTERNAL", "PIN"), ("EXTERNAL",), ("PIN",),
        np.asarray((PRIMARY_GRIP, PRIMARY_GRIP)),
    )
    assert roles.tolist() == [PRIMARY_GRIP, HARD_FORBIDDEN]
    assert allowed.tolist() == [True, False]
    assert reasons.tolist() == [0, 1]
    with pytest.raises(ValueError, match="B0_OBJECT_SURFACE_SEMANTICS_UNRESOLVED"):
        _b0_role_arrays(("UNKNOWN",), ("EXTERNAL",), (), np.asarray((PRIMARY_GRIP,)))


def test_b0_nominal_pickup_does_not_require_extra_disturbance_margin() -> None:
    quality = SimpleNamespace(
        nominal_gravity_lift_balance_pass=True,
        nominal_operation_force_cap_n=12.0,
        nominal_parameter_task_margin=0.287,
        status="TASK_REJECT",
    )
    assert b0_nominal_pickup_task_pass(quality, 12.0)
    assert not b0_nominal_pickup_task_pass(quality, 8.0)


def test_b0_sampled_table_clearance_uses_registered_numerical_tolerance() -> None:
    assert b0_sampled_table_clearance_pass(0.010, 0.000991278, 0.001, 1.0e-5)
    assert not b0_sampled_table_clearance_pass(0.010, 0.000989999, 0.001, 1.0e-5)
    assert not b0_sampled_table_clearance_pass(None, 0.010, 0.001, 1.0e-5)
