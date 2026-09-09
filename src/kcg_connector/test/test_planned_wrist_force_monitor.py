"""Contact load bandwidth must not weaken free-space checks or erase preload."""

from pathlib import Path
import sys

import numpy as np
import pytest

package = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(package), str(package / "isaac/carts_v2")]
from run_grasp_lift import _HighObservationWristFtAuditor


def monitor(dt=1/240, tau=.05):
    value = _HighObservationWristFtAuditor.__new__(_HighObservationWristFtAuditor)
    value.physics_dt_s = dt
    value.planned_contact_force_time_constant_s = tau
    value._continuous_filtered_force_task = None
    return value


@pytest.mark.parametrize("dt", [1/240, 1/480])
def test_isolated_contact_pulse_and_sustained_load_are_distinguished(dt):
    reader = monitor(dt)
    reader._force_for_protection([0., 0., 0.], False)
    short, _ = reader._force_for_protection([0., 0., 6.], True)
    assert np.linalg.norm(short) < 3.
    for _ in range(round(.2/dt)):
        sustained, _ = reader._force_for_protection([0., 0., 6.], True)
    assert np.linalg.norm(sustained) > 3.


def test_free_space_force_remains_instantaneous_after_contact_history():
    reader = monitor()
    reader._force_for_protection([0., 0., 0.], True)
    force, signal = reader._force_for_protection([4., -3., 2.], False)
    np.testing.assert_array_equal(force, [4., -3., 2.])
    assert signal == "RAW_CURRENT_WRENCH_FORCE"


def test_first_loaded_sample_is_not_initialized_as_zero():
    force, _ = monitor()._force_for_protection([0., 0., 6.], True)
    np.testing.assert_array_equal(force, [0., 0., 6.])


def test_switch_to_contact_keeps_accumulated_force_history():
    reader = monitor()
    for _ in range(60):
        reader._force_for_protection([0., 0., 6.], False)
    force, _ = reader._force_for_protection([0., 0., 6.], True)
    assert np.linalg.norm(force) > 3.


def test_zero_time_constant_preserves_legacy_contact_force_gate():
    reader = monitor(tau=0.)
    reader._force_for_protection([0., 0., 0.], False)
    force, signal = reader._force_for_protection([4., -3., 2.], True)
    np.testing.assert_array_equal(force, [4., -3., 2.])
    assert signal == "RAW_CURRENT_WRENCH_FORCE"
