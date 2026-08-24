"""Regression for stopping a finger at measured contact position."""

from pathlib import Path
import sys


ISAAC_V2 = Path(__file__).resolve().parents[2] / "isaac/carts_v2"
sys.path.insert(0, str(ISAAC_V2))

from controller import SequentialEffortContactController  # noqa: E402


def test_confirmed_contact_removes_residual_position_push() -> None:
    controller = SequentialEffortContactController(
        [0.0, 0.0, 0.0, 0.0], [0.0, 0.5, 0.5, 0.5],
        effort_rise_nm=0.02, position_error_rad=0.004,
        consecutive_samples=1, endpoint_timeout_samples=10,
    )
    target = controller.step(
        measured_position=[0.0, 0.03, 0.0, 0.0],
        measured_effort_delta=[0.0, 0.1, 0.0, 0.0],
        maximum_increment_rad=0.1,
    )
    assert target[1] == 0.03
    assert controller.contact_targets_rad == (0.03,)
    next_target = controller.step(
        measured_position=target, measured_effort_delta=[0.0] * 4,
        maximum_increment_rad=0.1,
    )
    assert next_target[1] == 0.03
    assert next_target[2] == 0.1
