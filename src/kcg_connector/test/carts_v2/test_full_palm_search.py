"""Regressions for height feasibility before the per-angle Top-8 budget."""

from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from kcg_connector.grasp.carts_v2.full_palm_search import (
    bind_pregrasp,
    fixed_pregrasp_phase_combinations,
    group_candidates_by_palm,
    run_full_palm_cascade,
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


def _height_evaluator(calls, rejected=()):
    rejected = set(rejected)

    def evaluate(seed):
        calls.append(seed.candidate_id)
        phases = (0.1, 0.1, 0.1)
        if seed.candidate_id in rejected:
            return (), {
                "candidate_id": seed.candidate_id,
                "exact_variant_evaluated_count": 1,
                "evaluated": [{"pregrasp_closure_phases": list(phases),
                               "status": "HARD_REJECT",
                               "reason": "EMPTY_TABLE_AND_CONTACT_HEIGHT_INTERSECTION"}],
            }
        projected = bind_pregrasp(INPUTS, seed, phases)
        pose = projected.object_from_hand_matrix().copy()
        pose[2, 3] = 0.1 + 0.001 * seed.source_sample_index
        projected = replace(projected, object_from_hand=tuple(pose.ravel()))
        return (projected,), {
            "candidate_id": seed.candidate_id,
            "exact_variant_evaluated_count": 1,
            "evaluated": [{
                "pregrasp_closure_phases": list(phases),
                "status": "OFFLINE_SAMPLED_HAND_HEIGHT_FEASIBLE_AT_PROJECTED_Z",
                "selection_key": [float(seed.source_sample_index)],
            }],
        }

    return evaluate


def test_fixed_design_and_group_identity_are_fail_closed() -> None:
    combinations = fixed_pregrasp_phase_combinations()
    assert len(combinations) == len(set(combinations)) == 27
    assert combinations[13] == (0.1, 0.1, 0.1)
    groups = group_candidates_by_palm((_seed("z", 0), _seed("a", 0)), GRID)
    assert len(groups) == 91
    assert [seed.candidate_id for seed in groups[0][1]] == ["a", "z"]
    with pytest.raises(ValueError, match="64-candidate"):
        group_candidates_by_palm(
            tuple(_seed(f"c{index:02d}", 1, index) for index in range(65)), GRID)


def test_every_seed_is_height_evaluated_before_per_angle_top8() -> None:
    candidates = tuple(_seed(f"seed_{index:02d}", 30, index)
                       for index in range(12))
    calls, checkpoints = [], []
    selected, audit = run_full_palm_cascade(
        INPUTS, candidates, GRID, height_evaluator=_height_evaluator(calls),
        progress_callback=lambda value: checkpoints.append(
            json.loads(json.dumps(value))),
    )
    assert calls == sorted(seed.candidate_id for seed in candidates)
    assert len(selected) == 8
    assert [seed.source_sample_index for seed in selected] == list(range(8))
    angle = audit["per_angle"][30]
    assert angle["height_evaluated_seed_count"] == 12
    assert angle["budget_retained_count"] == 8
    assert angle["budget_deferred_count"] == 4
    assert {row["reason"] for row in audit["deferred"]} == {
        "DEFERRED_BY_POST_HEIGHT_PER_ANGLE_TOP8"
    }
    assert len(checkpoints) == 91
    resumed, _ = run_full_palm_cascade(
        INPUTS, candidates, GRID,
        height_evaluator=lambda _seed_value: pytest.fail("complete checkpoint reran"),
        resume_audit=checkpoints[-1],
    )
    assert resumed == selected


def test_partial_checkpoint_resumes_at_next_angle_and_preserves_projection() -> None:
    candidates = tuple(_seed(f"palm_{index}", index, index) for index in range(3))
    calls, checkpoints = [], []
    full, _ = run_full_palm_cascade(
        INPUTS, candidates, GRID, height_evaluator=_height_evaluator(calls),
        progress_callback=lambda value: checkpoints.append(
            json.loads(json.dumps(value))),
    )
    resumed_calls = []
    resumed, _ = run_full_palm_cascade(
        INPUTS, candidates, GRID,
        height_evaluator=_height_evaluator(resumed_calls),
        resume_audit=checkpoints[1],
    )
    assert resumed_calls == ["palm_2"]
    assert resumed == full
    assert [seed.object_from_hand_matrix()[2, 3] for seed in resumed] == pytest.approx(
        [0.1, 0.101, 0.102])


def test_height_rejection_is_not_promoted_and_old_checkpoint_fails_closed() -> None:
    seed = _seed("rejected", 0)
    selected, audit = run_full_palm_cascade(
        INPUTS, (seed,), GRID,
        height_evaluator=_height_evaluator([], rejected={seed.candidate_id}),
    )
    assert selected == ()
    assert audit["rejected"][0]["reason"] == (
        "EMPTY_TABLE_AND_CONTACT_HEIGHT_INTERSECTION")
    old = dict(audit, schema_version="carts_full_palm_cascade_v1")
    with pytest.raises(ValueError, match="checkpoint identity"):
        run_full_palm_cascade(
            INPUTS, (seed,), GRID, height_evaluator=_height_evaluator([]),
            resume_audit=old,
        )
