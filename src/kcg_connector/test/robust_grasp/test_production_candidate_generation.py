from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import kcg_connector.grasp.robust.production_candidate_generation as production
from kcg_connector.grasp.robust.generation_checkpoint import CheckpointLifecycle
from kcg_connector.grasp.robust.production_candidate_generation import (
    ALLOWED_OBJECT_IDS,
    ProductionCandidateGenerationError,
    advance_checkpoint_incrementally,
    build_production_candidate_generation_runtime,
    generation_status_document,
    initialize_or_load_checkpoint,
)
from kcg_connector.grasp.robust.ray_closure import RayClosureSurfaceModel
from kcg_connector.grasp.robust.surface_anchored_closure import (
    SurfaceAnchoredRayClosureModel,
)
from kcg_connector.grasp.robust.top_level_candidate_generator import (
    MAIN_TOTAL_ATTEMPT_BUDGET,
    TopLevelCandidateGenerator,
)


REPOSITORY = Path(__file__).resolve().parents[4]


@pytest.fixture(scope="module")
def production_runtimes():
    return {
        object_id: build_production_candidate_generation_runtime(
            repository_root=REPOSITORY,
            object_id=object_id,
        )
        for object_id in ALLOWED_OBJECT_IDS
    }


def test_both_objects_use_real_contracts_and_the_same_shared_method(
    production_runtimes,
) -> None:
    method_settings = set()
    for object_id, runtime in production_runtimes.items():
        assert runtime.object_id == object_id
        assert isinstance(runtime.closure_model, RayClosureSurfaceModel)
        assert isinstance(runtime.surface_proposer, SurfaceAnchoredRayClosureModel)
        assert isinstance(runtime.generator, TopLevelCandidateGenerator)
        assert runtime.generator.v9_evaluator is runtime.closure_model
        assert runtime.generator.surface_proposer is runtime.surface_proposer
        assert runtime.surface_proposer.closure_model is runtime.closure_model
        assert runtime.generator.hand_model is runtime.closure_model.hand_model
        assert runtime.generator.anchor_pad_names == (
            "finger_1_pad",
            "finger_2_pad",
            "finger_3_pad",
        )
        expected_axis = runtime.object_contract.task_frame_rotation_object[:, 0]
        assert np.array_equal(
            runtime.closure_model.task_frame.transverse_axis_object,
            expected_axis,
        )
        expected_directions = runtime.hand_contract.closing_actuation_directions_unit(
            runtime.closure_model.hand_model
        )
        assert np.array_equal(
            runtime.closure_model.closing_directions_unit,
            expected_directions,
        )
        method_settings.add(
                (
                    runtime.closure_model.maximum_subdivision_intervals,
                    runtime.closure_model.interval_arithmetic_options.decimal_precision,
                    runtime.closure_model.interval_arithmetic_options.maximum_root_bisection_iterations,
                )
            )
        assert runtime.hand_contract.hardware_authorized is False
        assert runtime.object_contract.dynamic_eligible is False
    assert method_settings == {(4096, 80, 256)}


def test_checkpoint_initialization_is_empty_and_fail_closed(
    production_runtimes,
    tmp_path: Path,
) -> None:
    runtime = production_runtimes[ALLOWED_OBJECT_IDS[0]]
    store, stored = initialize_or_load_checkpoint(
        runtime,
        checkpoint_root=tmp_path / "checkpoint",
    )
    assert stored.manifest.lifecycle is CheckpointLifecycle.READY
    assert stored.state.target_total_attempt_budget == MAIN_TOTAL_ATTEMPT_BUDGET
    assert stored.state.completed_attempt_count == 0
    restored = store.load_latest(
        generator=runtime.generator,
        run_id=runtime.run_id,
        execution_environment_sha256=runtime.execution_environment_sha256,
    )
    assert restored == stored
    status = generation_status_document(
        runtime,
        restored,
        checkpoint_root=store.root,
    )
    assert status["completed_attempt_count"] == 0
    assert status["production_joint_route_count"] == 0
    assert status["formal_selected_candidate"] is None
    assert status["formal_selected_contact_range_policy"] is None
    assert status["full_hand_collision_state"] == "NOT_CERTIFIABLE"
    assert status["dynamic_launch_allowed"] is False
    assert status["hardware_authorized"] is False
    assert status["legacy_candidate_imported"] is False
    assert status["display_only_proposal_used_as_formal_evidence"] is False
    assert status["online_object_or_contact_truth_used"] is False


def test_incremental_runner_reuses_runtime_but_commits_each_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    runtime = object()
    store = object()
    stored = SimpleNamespace(
        state=SimpleNamespace(completed_attempt_count=2)
    )

    def fake_advance(
        actual_runtime,
        actual_store,
        actual_stored,
        *,
        stop_attempt_index_exclusive: int,
    ):
        assert actual_runtime is runtime
        assert actual_store is store
        assert stop_attempt_index_exclusive == (
            actual_stored.state.completed_attempt_count + 1
        )
        calls.append(stop_attempt_index_exclusive)
        return SimpleNamespace(
            state=SimpleNamespace(
                completed_attempt_count=stop_attempt_index_exclusive
            )
        )

    monkeypatch.setattr(production, "advance_checkpoint", fake_advance)
    completed = advance_checkpoint_incrementally(
        runtime,  # type: ignore[arg-type]
        store,  # type: ignore[arg-type]
        stored,  # type: ignore[arg-type]
        stop_attempt_index_exclusive=5,
    )

    assert calls == [3, 4, 5]
    assert completed.state.completed_attempt_count == 5
    with pytest.raises(
        ProductionCandidateGenerationError,
        match="cannot precede",
    ):
        advance_checkpoint_incrementally(
            runtime,  # type: ignore[arg-type]
            store,  # type: ignore[arg-type]
            completed,  # type: ignore[arg-type]
            stop_attempt_index_exclusive=4,
        )
