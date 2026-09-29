"""Counterexamples to a fixed quarter-turn after a short loaded stroke."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'isaac'))
from te_phase_equivalent_reindex import select_phase_equivalent_return, retain_returned_grasp_yaw
from kcg_connector.grasp.robust.bounded_hand_base_ik import CandidateJointRouteError


def wrist_fixture(start_deg, *, collision_angles=()):
    # This simple wrist fixture isolates cumulative travel and future bounds.
    start = np.zeros(7)
    start[6] = np.deg2rad(start_deg)

    def returned(angle):
        if angle in collision_angles:
            raise RuntimeError('Source hand collides on this return')
        end = start.copy()
        end[6] -= np.deg2rad(angle)
        return {'states': np.linspace(start, end, 91), 'angle': angle}

    def preview(q, turn):
        end = q.copy()
        end[6] -= np.deg2rad(turn)
        return np.linspace(q, end, 91)

    return returned, preview


BOUNDS = np.deg2rad(np.full(7, 173.0))


def choose(start, next_turn=-90.0, **kwargs):
    back, forward = wrist_fixture(start, **kwargs)
    return select_phase_equivalent_return(back, forward, -BOUNDS, BOUNDS,
                                          next_turn_deg=next_turn)


def test_short_stroke_does_not_require_a_quarter_turn():
    _, record = choose(32.6)
    assert record['selected_return_deg'] == 0.0


def test_next_loaded_stroke_is_checked_before_omitting_return():
    plan, record = choose(95.0)
    assert record['selected_return_deg'] == 90.0
    assert np.rad2deg(plan['states'][-1, 6]) == pytest.approx(5.0)
    assert not record['candidates'][0]['feasible']


def test_final_small_stroke_does_not_inherit_a_ninety_degree_return():
    _, record = choose(95.0, next_turn=-5.5)
    assert record['selected_return_deg'] == 0.0


def test_repeated_five_degree_strokes_do_not_accumulate_return_drift():
    wrist = 9.0
    choices = []
    for _ in range(40):
        wrist += 5.0
        plan, record = choose(wrist)
        wrist = float(np.rad2deg(plan['states'][-1, 6]))
        choices.append(record['selected_return_deg'])
        assert -173.0 <= wrist <= 173.0
        assert wrist + 90.0 <= 173.0 + 1e-9
    assert choices.count(0.0) > choices.count(90.0)


def test_collision_rejection_is_not_overridden_by_joint_margin():
    with pytest.raises(RuntimeError, match='No phase-equivalent return'):
        choose(95.0, collision_angles=(90.0,))


def test_unreachable_ik_alternative_does_not_discard_feasible_return():
    returned, preview = wrist_fixture(95.)
    def candidate(angle):
        if angle == -90.:
            raise CandidateJointRouteError('IK_TARGET_UNREACHABLE', 'negative return cannot be reached')
        return returned(angle)
    _, record = select_phase_equivalent_return(candidate, preview, -BOUNDS, BOUNDS, next_turn_deg=-90.)
    assert record['selected_return_deg'] == 90.
    assert not record['candidates'][-1]['feasible']


def test_invalid_ik_input_is_not_treated_as_an_unreachable_candidate():
    returned, preview = wrist_fixture(95.)
    def candidate(angle):
        raise CandidateJointRouteError('IK_TARGET_INVALID', 'malformed target')
    with pytest.raises(CandidateJointRouteError, match='IK_TARGET_INVALID'):
        select_phase_equivalent_return(candidate, preview, -BOUNDS, BOUNDS, next_turn_deg=-90.)


@pytest.mark.parametrize('turn', [0.0, 10.0, -120.1, float('nan')])
def test_unknown_next_motion_is_rejected(turn):
    with pytest.raises(ValueError):
        choose(10.0, next_turn=turn)


def test_completed_quarter_turn_does_not_erase_a_selected_alternative_phase():
    assert retain_returned_grasp_yaw(180., preserve_partial=True, equivalent_return_completed=True)
    assert not retain_returned_grasp_yaw(180., preserve_partial=True, equivalent_return_completed=False)


def test_legacy_partial_return_rule_remains_available():
    assert retain_returned_grasp_yaw(204.8, preserve_partial=True, equivalent_return_completed=False)
    assert not retain_returned_grasp_yaw(0., preserve_partial=True, equivalent_return_completed=False)
