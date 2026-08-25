import json
from dataclasses import replace
from types import SimpleNamespace

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


class _Hand:
    independent_joint_names = ("f1j1", "f1j2", "f2j1", "f3j2")
    joints = {
        name: SimpleNamespace(limit=SimpleNamespace(lower=0.0, upper=limit))
        for name, limit in zip(independent_joint_names, (1.57, 1.0, 1.0, 1.0))
    }

    def resolve_joint_positions(self, positions, *, enforce_limits):
        assert enforce_limits
        assert len(positions) == 4
        return dict(zip(self.independent_joint_names, positions))


class _Config:
    def section(self, name):
        assert name == "candidate_generation"
        return {"backend": "GRASPGENX_FULL_PALM"}


INPUTS = SimpleNamespace(
    hand_model=_Hand(), config=_Config(),
    closing_directions=np.asarray(((0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))),
)


def _seed(identifier: str, palm_index: int, source_index: int = 0) -> CandidateSeed:
    palm = GRID[palm_index]
    return CandidateSeed(
        candidate_id=identifier, object_id="object_a", anchor_face_index=0,
        anchor_position_object_m=(0.0, 0.0, 0.0),
        object_from_hand=tuple(float(value) for value in np.eye(4).ravel()),
        pregrasp_joint_positions_rad=(palm, 0.0, 0.0, 0.0),
        pregrasp_closure_phases=(0.0, 0.0, 0.0), source_sample_index=source_index,
        generator_score=float(source_index), descriptor_id=f"palm_{palm_index:03d}",
        palm_configuration_rad=palm,
    )


def _pregrasp_result(seed, phases):
    distance = sum(abs(value - 0.1) for value in phases)
    return {
        "candidate_id": seed.candidate_id,
        "pregrasp_closure_phases": phases,
        "accepted": True, "reasons": (),
        "minimum_table_clearance_m": 0.02 - distance,
        "pregrasp_pad_clearance_by_name_m": {
            "finger_1_pad": 0.001 + distance,
            "finger_2_pad": 0.002 + distance,
            "finger_3_pad": 0.003 + distance,
        },
    }


def test_fixed_design_and_group_identity_are_fail_closed() -> None:
    combinations = fixed_pregrasp_phase_combinations()
    assert len(combinations) == len(set(combinations)) == 27
    assert combinations[13] == (0.1, 0.1, 0.1)
    groups = group_candidates_by_palm((_seed("z", 0), _seed("a", 0)), GRID)
    assert len(groups) == 91
    assert [seed.candidate_id for seed in groups[0][1]] == ["a", "z"]
    with pytest.raises(ValueError, match="64-candidate"):
        group_candidates_by_palm(
            tuple(_seed(f"c{index:02d}", 1, index) for index in range(65)), GRID
        )
    with pytest.raises(ValueError, match="no finite palm"):
        group_candidates_by_palm(
            (replace(_seed("bad", 0), palm_configuration_rad=None),), GRID
        )


def test_pregrasp_choice_requires_all_27_bound_phase_results() -> None:
    seed = _seed("candidate", 45)
    rows = tuple(
        (phases, _pregrasp_result(seed, phases))
        for phases in fixed_pregrasp_phase_combinations()
    )
    selected = select_pregrasp_combination(seed, rows)
    assert selected["pregrasp_closure_phases"] == (0.1, 0.1, 0.1)
    outside_table = dict(rows[0][1], minimum_table_clearance_m=None)
    outside_selected = select_pregrasp_combination(
        seed, ((rows[0][0], outside_table), *rows[1:])
    )
    assert outside_selected["pregrasp_closure_phases"] == (0.1, 0.1, 0.1)
    assert outside_selected["worst_pregrasp_pad_clearance_m"] == pytest.approx(0.003)
    misleading = dict(rows[0][1], pregrasp_pad_clearance_by_name_m={
        "finger_1_pad": 0.0001, "finger_2_pad": 0.05, "finger_3_pad": 0.05})
    worst_selected = select_pregrasp_combination(
        seed, ((rows[0][0], misleading), *rows[1:]))
    assert worst_selected["pregrasp_closure_phases"] == (0.1, 0.1, 0.1)
    changed = dict(rows[0][1], pregrasp_closure_phases=(0.2, 0.2, 0.2))
    with pytest.raises(ValueError, match="phase identity"):
        select_pregrasp_combination(seed, ((rows[0][0], changed), *rows[1:]))


def test_cascade_reserves_each_branch_and_binds_selected_preshape() -> None:
    candidates = tuple(_seed(f"diff_{index}", 30, index) for index in range(9))
    candidates += tuple(_seed(f"obb_{index}", 30, 100 + index) for index in range(9))
    diagnostics = {
        seed.candidate_id: {
            "branch": "obb" if seed.candidate_id.startswith("obb") else "diff",
            "generator_score": 1000.0 if seed.candidate_id.startswith("diff") else -1000.0,
            "physical_selection_key": [float(seed.source_sample_index)],
            "aabb_selection_role": "DIAGNOSTIC_ONLY_NOT_HARD_REJECT",
        }
        for seed in candidates
    }
    batch_sizes, precise = [], []

    def pregrasp(variants):
        batch_sizes.append(len(variants))
        return tuple(_pregrasp_result(seed, phases) for seed, phases in variants)

    def exact(seed):
        precise.append(seed)
        return {
            "accepted": True, "closure_pass": True, "fast_filter_pass": True,
            "selection_key": [float(seed.source_sample_index)],
        }

    selected, audit = run_full_palm_cascade(
        INPUTS, candidates, GRID, budget_diagnostics=diagnostics,
        pregrasp_evaluator=pregrasp, precise_evaluator=exact,
    )
    assert batch_sizes == [8 * 27]
    assert sum(seed.candidate_id.startswith("diff") for seed in precise) == 7
    assert sum(seed.candidate_id.startswith("obb") for seed in precise) == 1
    assert len(selected) == 8
    assert selected[0].candidate_id == "diff_0"
    assert selected[0].pregrasp_closure_phases == (0.1, 0.1, 0.1)
    assert selected[0].pregrasp_joint_positions_rad == pytest.approx(
        (GRID[30], 0.1, 0.1, 0.1)
    )
    assert selected[0].palm_configuration_rad == GRID[30]
    assert {row["reason"] for row in audit["deferred"]} == {
        "DEFERRED_BY_PER_ANGLE_PRECISE_BUDGET"
    }
    assert max(audit["callback_evaluation_count_by_candidate"].values()) == 28
    assert audit["pregrasp_physical_query_state_count"] == 0
    assert audit["pregrasp_reused_identical_state_count"] == 0
    json.dumps(audit, allow_nan=False)


def test_precise_rejection_is_diagnostic_and_selects_nothing() -> None:
    seed = _seed("rejected", 0)
    diagnostic = {
        seed.candidate_id: {"branch": "diff", "generator_score": 1.0,
                            "physical_selection_key": [0.0]}
    }
    selected, audit = run_full_palm_cascade(
        INPUTS, (seed,), GRID, budget_diagnostics=diagnostic,
        pregrasp_evaluator=lambda rows: tuple(
            _pregrasp_result(candidate, phases) for candidate, phases in rows
        ),
        precise_evaluator=lambda _seed_value: {
            "accepted": False, "closure_pass": False, "fast_filter_pass": False,
            "reason": "NO_THREE_PAD_CONTACT", "selection_key": [0.0],
        },
    )
    assert selected == ()
    assert audit["rejected"][0]["reason"] == (
        "SELECTED_PREGRASP_PRECISE_REJECT:NO_THREE_PAD_CONTACT")
    assert audit["rejected"][0]["alternative_pregrasp_status"] == (
        "BUDGET_DEFERRED_NOT_PHYSICAL_FAILURE")
    assert audit["claim_scope"].startswith("OFFLINE_")
