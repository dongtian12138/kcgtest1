from dataclasses import replace

import numpy as np
import pytest

from kcg_connector.grasp.carts_v2.full_palm_search import (
    fixed_pregrasp_phase_combinations,
    group_candidates_by_palm,
    run_full_palm_cascade,
    select_pregrasp_combination,
)
from kcg_connector.grasp.carts_v2.models import CandidateSeed


GRID = tuple(float(value) for value in np.linspace(0.0, 1.57, 91))


def _seed(identifier: str, palm_index: int, source_index: int = 0) -> CandidateSeed:
    palm = GRID[palm_index]
    return CandidateSeed(
        candidate_id=identifier,
        object_id="object_a",
        anchor_face_index=0,
        anchor_position_object_m=(0.0, 0.0, 0.0),
        object_from_hand=tuple(float(value) for value in np.eye(4).ravel()),
        pregrasp_joint_positions_rad=(palm, 0.0, 0.0, 0.0),
        pregrasp_closure_phases=(0.0, 0.0, 0.0),
        source_sample_index=source_index,
        descriptor_id=f"palm_{palm_index:03d}",
        palm_configuration_rad=palm,
    )


def _result(
    *, accepted=True, open_collision=False, finger_collision=False,
    key=(0.0,), count=1,
):
    return {
        "accepted": accepted,
        "open_palm_base_table_collision": open_collision,
        "pregrasp_finger_chain_table_collision": finger_collision,
        "selection_key": key,
        "evaluation_count": count,
    }


def test_fixed_pregrasp_design_has_27_deterministic_combinations() -> None:
    combinations = fixed_pregrasp_phase_combinations()
    assert len(combinations) == len(set(combinations)) == 27
    assert combinations[0] == (0.0, 0.0, 0.0)
    assert combinations[13] == (0.1, 0.1, 0.1)
    assert combinations[-1] == (0.2, 0.2, 0.2)


def test_grouping_returns_91_buckets_and_fails_closed_on_identity_or_cap() -> None:
    groups = group_candidates_by_palm(
        (_seed("z", 0), _seed("a", 0), _seed("end", 90)), GRID
    )
    assert len(groups) == 91
    assert [seed.candidate_id for seed in groups[0][1]] == ["a", "z"]
    assert groups[-1][0] == GRID[-1]
    assert groups[-1][1][0].candidate_id == "end"
    with pytest.raises(ValueError, match="64-candidate"):
        group_candidates_by_palm(
            tuple(_seed(f"c{index:02d}", 1, index) for index in range(65)), GRID
        )
    with pytest.raises(ValueError, match="no finite palm"):
        group_candidates_by_palm((replace(_seed("bad", 0), palm_configuration_rad=None),), GRID)
    with pytest.raises(ValueError, match="91 finite"):
        group_candidates_by_palm((), (*GRID[:-1], GRID[-1] + 0.1))


def test_open_palm_or_base_table_collision_rejects_immediately() -> None:
    calls = []

    def cheap(seed, phases, remaining):
        calls.append((seed.candidate_id, phases, remaining))
        return _result(accepted=False, open_collision=True, count=3)

    result = select_pregrasp_combination(_seed("open_collision", 0), cheap)
    assert result["reason"] == "OPEN_PALM_BASE_TABLE_COLLISION"
    assert result["stage_a_evaluation_count"] == 3
    assert len(calls) == 1


def test_one_finger_chain_collision_only_removes_that_pregrasp() -> None:
    def cheap(_seed_value, phases, _remaining):
        if phases == (0.0, 0.0, 0.0):
            return _result(accepted=False, finger_collision=True)
        distance = sum((value - 0.1) ** 2 for value in phases)
        return _result(key=(distance,))

    result = select_pregrasp_combination(_seed("survives", 45), cheap)
    assert result["status"] == "SURVIVE"
    assert result["pregrasp_closure_phases"] == (0.1, 0.1, 0.1)
    assert result["stage_a_evaluation_count"] == 27

    all_collide = select_pregrasp_combination(
        _seed("all_collide", 45),
        lambda *_args: _result(accepted=False, finger_collision=True),
    )
    assert all_collide["reason"] == "ALL_27_PREGRASP_FINGER_CHAINS_COLLIDE_TABLE"


def test_cascade_sends_only_deterministic_per_angle_top8_to_precise() -> None:
    candidates = tuple(_seed(f"c{index:02d}", 30, index) for index in range(10))
    precise_calls = []

    def cheap(seed, phases, _remaining):
        return _result(key=(float(seed.source_sample_index), sum(phases)))

    def precise(seed, phases, remaining):
        precise_calls.append((seed.source_sample_index, phases, remaining))
        return {"accepted": True, "selection_key": (-seed.source_sample_index,), "evaluation_count": 5}

    result = run_full_palm_cascade(
        candidates, GRID, cheap_evaluator=cheap, precise_evaluator=precise
    )
    assert [row[0] for row in precise_calls] == list(range(8))
    assert len(result["selected"]) == 8
    assert len(result["deferred"]) == 2
    assert [row["seed"].source_sample_index for row in result["selected"]] == list(reversed(range(8)))
    assert result["evaluation_count_by_candidate"]["c00"] == {
        "stage_a": 27, "stage_b": 5, "total": 32,
    }


def test_stage_a_and_b_share_one_fail_closed_300_evaluation_budget() -> None:
    with pytest.raises(ValueError, match="budget"):
        select_pregrasp_combination(
            _seed("stage_a_over", 0),
            lambda *_args: _result(count=12),
        )
    with pytest.raises(ValueError, match=r"\[1, 300\]"):
        select_pregrasp_combination(
            _seed("invalid_budget", 0),
            lambda *_args: _result(),
            global_budget_per_seed=301,
        )
    with pytest.raises(ValueError, match=r"\[1, 300\]"):
        run_full_palm_cascade(
            (),
            GRID,
            cheap_evaluator=lambda *_args: _result(),
            precise_evaluator=lambda *_args: _result(),
            global_budget_per_seed=301,
        )

    def precise_over(_seed_value, _phases, remaining):
        return {"accepted": True, "selection_key": (0.0,), "evaluation_count": remaining + 1}

    with pytest.raises(ValueError, match="budget exceeded"):
        run_full_palm_cascade(
            (_seed("stage_b_over", 0),),
            GRID,
            cheap_evaluator=lambda *_args: _result(),
            precise_evaluator=precise_over,
        )
