"""Direct regressions for offline replay of observed nail-free hand states."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from kcg_connector.grasp.carts_v2.models import load_v2_inputs
from kcg_connector.grasp.carts_v2.observed_state_replay import (
    ObservedHandStateEvaluator,
)
from kcg_connector.robot_model import expand_active_hand_positions


ROOT = Path(__file__).resolve().parents[4]
CONFIG = ROOT / "src/kcg_connector/config/carts_nailfree_height_projected.yaml"
OBJECT_B = "te_deutsch_d38999_26fj35pn_step"
ANCHOR2 = ROOT / (
    "artifacts/carts_v2/opposition60_isaac/qp60_anchor_a02_task_ik/"
    "opposition60_anchor_a02_exact_offset_00_count_01_static_control.json"
)


@pytest.fixture(scope="module")
def replay_context():
    pytest.importorskip("fcl")
    inputs = load_v2_inputs(ROOT, config_path=CONFIG, object_id=OBJECT_B)
    document = json.loads(ANCHOR2.read_text(encoding="utf-8"))
    survivor = document["survivor_candidates"][0]
    target = np.asarray(
        document["task_and_bounded_ik"][0]["target_world_from_handbase_row_major"],
        dtype=np.float64,
    ).reshape(4, 4)
    joints = expand_active_hand_positions(survivor["pregrasp_joint_positions_rad"])
    return inputs, ObservedHandStateEvaluator(inputs), target, joints


def test_observed_replay_preserves_measured_mimic_error(replay_context) -> None:
    inputs, evaluator, target, joints = replay_context
    inconsistent = dict(joints)
    inconsistent["f1j3"] += 0.01
    result = evaluator.evaluate(
        target, inconsistent, inputs.frozen_world_from_object)
    assert result["mimic_error_rad_by_joint"]["f1j3"] == pytest.approx(0.01)


def test_anchor2_pregrasp_replays_original_mesh_clearance(replay_context) -> None:
    inputs, evaluator, target, joints = replay_context
    result = evaluator.evaluate(target, joints, inputs.frozen_world_from_object)
    safety = evaluator.evaluate_safety(
        target, joints, inputs.frozen_world_from_object)
    assert result["table_top"]["minimum_clearance_m"] == pytest.approx(
        0.014650263, abs=5.0e-7)
    assert result["table_top"]["minimum_clearance_m"] > 0.010
    assert result["table_top"]["evidence_scope"].endswith("NOT_SIDE_OR_BOTTOM")
    assert result["self_collision"]["intersecting_pairs"] == []
    assert result["non_task_hand_object"]["intersecting_links"] == []
    assert all(not row["full_object_intersecting"]
               for row in result["task_grip_surface_by_finger"].values())
    assert result["fail_closed"] is False
    assert safety["task_grip_surface_by_finger"] is None
    assert safety["table_top"] == result["table_top"]
    assert safety["self_collision"] == result["self_collision"]
    assert safety["non_task_hand_object"] == result["non_task_hand_object"]
    assert not any(safety["task_surface_intersecting_by_finger"].values())
    assert safety["fail_closed"] is False


def test_safety_replay_rejects_task_surface_penetration(
    replay_context, monkeypatch,
) -> None:
    inputs, evaluator, target, joints = replay_context
    monkeypatch.setattr(
        evaluator._surface_query,
        "query_pad",
        lambda _name, _transform: (SimpleNamespace(intersecting=True), None, None),
    )
    result = evaluator.evaluate_safety(
        target, joints, inputs.frozen_world_from_object)
    assert result["non_task_hand_object"]["intersecting_links"] == []
    assert all(result["task_surface_intersecting_by_finger"].values())
    assert result["fail_closed"] is True
    assert result["status"] == "OBSERVED_STATE_GEOMETRY_REJECT"


def test_observed_object_pose_changes_exact_mesh_distances(replay_context) -> None:
    inputs, evaluator, target, joints = replay_context
    baseline = evaluator.evaluate(target, joints, inputs.frozen_world_from_object)
    moved_object = np.array(inputs.frozen_world_from_object, copy=True)
    moved_object[0, 3] += 0.05
    moved = evaluator.evaluate(target, joints, moved_object)
    before = baseline["non_task_hand_object"]["minimum_clearance_m"]
    after = moved["non_task_hand_object"]["minimum_clearance_m"]
    assert after != pytest.approx(before, abs=1.0e-6)
    before_task = min(row["allowed_distance_m"]
                      for row in baseline["task_grip_surface_by_finger"].values())
    after_task = min(row["allowed_distance_m"]
                     for row in moved["task_grip_surface_by_finger"].values())
    assert after_task != pytest.approx(before_task, abs=1.0e-6)


def test_observed_replay_fails_closed_on_table_intersection(replay_context) -> None:
    inputs, evaluator, target, joints = replay_context
    lowered = np.array(target, copy=True)
    lowered[2, 3] -= 0.020
    result = evaluator.evaluate(lowered, joints, inputs.frozen_world_from_object)
    assert result["table_top"]["top_intersection_beyond_numerical_tolerance"] is True
    assert result["fail_closed"] is True
    assert result["status"] == "OBSERVED_STATE_GEOMETRY_REJECT"
