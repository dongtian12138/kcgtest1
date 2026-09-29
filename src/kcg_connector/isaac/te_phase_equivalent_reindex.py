"""Choose a short, phase-equivalent free return with room for the next turn.

The caller supplies actual robot IK and collision-checked return paths. The
lookahead checks joint feasibility only; the next measured regrasp and loaded
controller must still perform their normal geometry and force checks.
"""
import numpy as np
from kcg_connector.grasp.robust.bounded_hand_base_ik import CandidateJointRouteError


def retain_returned_grasp_yaw(commanded_deg, *, preserve_partial, equivalent_return_completed):
    """Keep a chosen equivalent phase from being undone by the next grasp."""
    angle = float(commanded_deg)
    if not np.isfinite(angle) or angle < 0:
        raise ValueError('Cumulative commanded angle must be finite and nonnegative')
    if type(preserve_partial) is not bool or type(equivalent_return_completed) is not bool:
        raise ValueError('Return phase policy flags must be boolean')
    partial_quarter = abs((angle+45.) % 90.-45.) > .01
    return equivalent_return_completed or (preserve_partial and angle >= 80. and partial_quarter)


def select_phase_equivalent_return(build_return, build_lookahead, lower, upper,
                                   *, next_turn_deg, phase_period_deg=90.0):
    """Select among zero and one positive/negative exterior phase period.

Minimize free angular travel first. Among equally short feasible returns,
prefer the larger joint margin over the return and next-turn preview. These
are finite geometric alternatives, not physics trials or parameter sweeps.
"""
    lo, hi = np.asarray(lower, float), np.asarray(upper, float)
    turn, period = float(next_turn_deg), float(phase_period_deg)
    if (lo.shape != (7,) or hi.shape != (7,) or
            not np.isfinite(np.r_[lo, hi, turn, period]).all() or
            np.any(lo >= hi) or not -120.0 <= turn < 0.0 or period != 90.0):
        raise ValueError('Expected original seven-joint bounds, a finite tightening turn, and the source 90 degree phase')

    def checked_path(values, label):
        path = np.asarray(values, float)
        if path.ndim != 2 or path.shape[1] != 7 or not len(path) or not np.isfinite(path).all():
            raise RuntimeError(label + ' is not a finite seven-joint path')
        if np.any(path < lo) or np.any(path > hi):
            raise RuntimeError(label + ' crosses an original soft joint limit')
        return path

    attempts, feasible = [], []
    for angle in (0.0, period, -period):
        try:
            plan = build_return(angle)
            path = checked_path(plan['states'], 'Open return')
            preview = checked_path(build_lookahead(path[-1].copy(), turn), 'Next-turn preview')
            if not np.allclose(preview[0], path[-1], rtol=0, atol=1e-9):
                raise RuntimeError('Next-turn preview does not start at the return endpoint')
            joined = np.vstack((path, preview))
            margin = float(np.min(np.minimum(joined - lo, hi - joined)))
            travel = float(np.sum(np.abs(np.diff(path, axis=0))))
            row = {'return_angle_deg': angle, 'feasible': True,
                   'minimum_joint_margin_rad': margin,
                   'return_joint_travel_rad': travel,
                   'return_samples': len(path), 'lookahead_samples': len(preview)}
            attempts.append(row)
            feasible.append(((abs(angle), -margin, travel), plan, row))
            if angle == 0.0:
                # No other candidate can improve the primary travel objective.
                break
        except (RuntimeError, CandidateJointRouteError) as error:
            # A geometrically unreachable alternative is a candidate rejection,
            # not a reason to discard another feasible return. Invalid solver
            # inputs/configuration remain errors and must not be hidden.
            if isinstance(error, CandidateJointRouteError) and error.code != 'IK_TARGET_UNREACHABLE':
                raise
            attempts.append({'return_angle_deg': angle, 'feasible': False, 'reason': str(error)})

    if not feasible:
        details = '; '.join(f"{r['return_angle_deg']:g}deg: {r.get('reason', '')}" for r in attempts)
        raise RuntimeError('No phase-equivalent return leaves room for the next turn: ' + details)
    _, plan, chosen = min(feasible, key=lambda entry: entry[0])
    record = {'policy': 'MINIMUM_FREE_TURN_WITH_NEXT_STROKE_JOINT_FEASIBILITY',
              'selected_return_deg': chosen['return_angle_deg'],
              'next_loaded_turn_deg': turn, 'phase_period_deg': period,
              'candidates': attempts,
              'zero_return_short_circuit': chosen['return_angle_deg'] == 0.0,
              'future_collision_or_grasp_success_certified': False,
              'online_object_or_contact_truth_used': False}
    return plan, record
