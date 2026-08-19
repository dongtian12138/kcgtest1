from __future__ import annotations

import copy
import inspect
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from kcg_connector.d38999_multilayer_multiview_fusion import (
    FROZEN_SOURCES,
    MAXIMUM_ROTATION_SPREAD_RAD,
    MAXIMUM_TRANSLATION_SPREAD_M,
    MultiviewC2Observation,
    MultiviewFusionError,
    build_multilayer_multiview_fusion_contract,
    fuse_c2_multiview,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
BRANCH_IDS = ("C2_LINKED_BRANCH_0", "C2_LINKED_BRANCH_PI")


def _transform(rotation, translation):
    value = np.eye(4); value[:3, :3] = rotation; value[:3, 3] = translation
    return value


def _view(index, perturbation_rotation_rad, perturbation_translation_m, *, independence_id=None):
    base_rotation = Rotation.from_euler("xyz", (0.10, -0.14, 0.31)).as_matrix()
    base_translation = np.asarray((0.08, -0.03, 0.70))
    perturbation = Rotation.from_euler("x", perturbation_rotation_rad).as_matrix()
    zero = _transform(base_rotation @ perturbation, base_translation + np.asarray(perturbation_translation_m))
    pi = zero.copy(); pi[:3, :3] = zero[:3, :3] @ np.diag((-1.0, -1.0, 1.0))
    candidates = tuple(
        {"branch_id": branch_id, "T_camera_model": transform.tolist(), "selected_for_control": False}
        for branch_id, transform in zip(BRANCH_IDS, (zero, pi))
    )
    return MultiviewC2Observation(
        view_id=f"V{index}",
        independence_id=independence_id or f"CAMERA_EXTRINSIC_{index}",
        capture_batch_id="BATCH_001",
        timestamp_utc=f"2026-08-17T00:00:0{index}Z",
        frame_id="receptacle_camera_optical",
        candidates=candidates,
    )


def _fixture_views():
    return [
        _view(0, -0.01, (-0.0004, 0.0002, 0.0)),
        _view(1, 0.0, (0.0, 0.0, 0.0)),
        _view(2, 0.01, (0.0004, -0.0002, 0.0)),
    ]


def _copy_sources(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    for relative in FROZEN_SOURCES:
        source = REPOSITORY_ROOT / relative
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    return root


def test_current_contract_reuses_existing_candidate_envelopes():
    contract = build_multilayer_multiview_fusion_contract(REPOSITORY_ROOT)
    assert contract["status"] == "OFFLINE_PASS"
    assert contract["minimum_independent_views"] == 2
    assert contract["maximum_translation_spread_m"] == 0.002
    assert contract["maximum_rotation_spread_rad"] == pytest.approx(math.radians(6.0))
    assert contract["threshold_label"] == "SIM_TUNING_ONLY_CANDIDATE"
    assert contract["cross_branch_fusion_allowed"] is False
    assert contract["automatic_outlier_removal_allowed"] is False
    assert contract["current_readiness"]["dynamic_independent_views_proven"] == 0
    assert contract["dynamic_multiview_fusion_pass_claimed"] is False


def test_three_independent_views_fuse_each_branch_and_reduce_error():
    views = _fixture_views()
    result = fuse_c2_multiview(views)
    expected_rotation = Rotation.from_euler("xyz", (0.10, -0.14, 0.31)).as_matrix()
    expected_translation = np.asarray((0.08, -0.03, 0.70))
    first = np.asarray(result.candidates[0]["T_camera_model"])
    assert np.linalg.norm(first[:3, 3] - expected_translation) < 1.0e-12
    relative = first[:3, :3].T @ expected_rotation
    assert math.acos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)) < 1.0e-7
    assert result.summary["view_count"] == 3
    assert result.summary["cross_branch_fusion_used"] is False


def test_two_fused_branches_remain_pi_linked_and_unselected():
    result = fuse_c2_multiview(_fixture_views())
    first = np.asarray(result.candidates[0]["T_camera_model"])
    second = np.asarray(result.candidates[1]["T_camera_model"])
    assert np.allclose(second[:3, :3], first[:3, :3] @ np.diag((-1.0, -1.0, 1.0)), atol=1.0e-9)
    assert np.allclose(second[:3, 3], first[:3, 3], atol=1.0e-12)
    assert all(item["selected_for_control"] is False for item in result.candidates)
    assert result.summary["selected_for_control"] is None
    assert result.summary["control_authorized"] is False


def test_input_order_is_normalized_by_timestamp():
    views = _fixture_views()
    first = fuse_c2_multiview(views)
    second = fuse_c2_multiview(list(reversed(views)))
    assert first.summary["view_ids"] == ["V0", "V1", "V2"]
    assert json.dumps(first.summary, sort_keys=True) == json.dumps(second.summary, sort_keys=True)
    for before, after in zip(first.candidates, second.candidates):
        assert np.allclose(before["T_camera_model"], after["T_camera_model"], atol=1.0e-12)


def test_repeated_comoving_independence_id_is_rejected():
    views = [_view(0, 0.0, (0, 0, 0), independence_id="WRIST_CAMERA"), _view(1, 0.0, (0, 0, 0), independence_id="WRIST_CAMERA")]
    with pytest.raises(MultiviewFusionError) as caught:
        fuse_c2_multiview(views)
    assert caught.value.code == "NONINDEPENDENT_VIEW_REUSE"


@pytest.mark.parametrize(
    "field,value,code",
    [
        ("frame_id", "other_frame", "FRAME_MISMATCH"),
        ("capture_batch_id", "OTHER_BATCH", "CAPTURE_BATCH_MISMATCH"),
        ("timestamp_utc", "2026-08-17T00:00:00Z", "DUPLICATE_TIMESTAMP"),
        ("view_id", "V0", "DUPLICATE_VIEW"),
    ],
)
def test_frame_batch_time_and_view_identity_mismatch_fail_closed(field, value, code):
    views = _fixture_views()[:2]
    values = dict(views[1].__dict__); values[field] = value
    views[1] = MultiviewC2Observation(**values)
    with pytest.raises(MultiviewFusionError) as caught:
        fuse_c2_multiview(views)
    assert caught.value.code == code


def test_translation_and_rotation_outlier_sets_are_rejected_without_removal():
    views = _fixture_views()
    far = _view(3, 0.0, (MAXIMUM_TRANSLATION_SPREAD_M * 2.0, 0.0, 0.0))
    with pytest.raises(MultiviewFusionError) as caught:
        fuse_c2_multiview([views[1], far])
    assert caught.value.code == "TRANSLATION_OUTLIER_SET_REJECTED"
    rotated = _view(3, MAXIMUM_ROTATION_SPREAD_RAD * 2.0, (0.0, 0.0, 0.0))
    with pytest.raises(MultiviewFusionError) as caught:
        fuse_c2_multiview([views[1], rotated])
    assert caught.value.code == "ROTATION_OUTLIER_SET_REJECTED"


def test_missing_branch_selection_and_truth_fields_are_rejected():
    views = _fixture_views()[:2]
    bad_values = dict(views[1].__dict__); bad_values["candidates"] = (views[1].candidates[0],)
    with pytest.raises(MultiviewFusionError) as caught:
        fuse_c2_multiview([views[0], MultiviewC2Observation(**bad_values)])
    assert caught.value.code == "INVALID_C2_BRANCHES"
    candidates = copy.deepcopy(list(views[1].candidates)); candidates[0]["selected_for_control"] = True
    bad_values = dict(views[1].__dict__); bad_values["candidates"] = tuple(candidates)
    with pytest.raises(MultiviewFusionError) as caught:
        fuse_c2_multiview([views[0], MultiviewC2Observation(**bad_values)])
    assert caught.value.code == "CONTROL_SELECTION_REJECTED"
    candidates = copy.deepcopy(list(views[1].candidates)); candidates[0]["ground_truth_pose"] = [0] * 7
    bad_values = dict(views[1].__dict__); bad_values["candidates"] = tuple(candidates)
    with pytest.raises(MultiviewFusionError) as caught:
        fuse_c2_multiview([views[0], MultiviewC2Observation(**bad_values)])
    assert caught.value.code == "TRUTH_FIELD_REJECTED"


def test_one_view_is_not_multiview():
    with pytest.raises(MultiviewFusionError) as caught:
        fuse_c2_multiview(_fixture_views()[:1])
    assert caught.value.code == "INSUFFICIENT_INDEPENDENT_VIEWS"


def test_summary_is_finite_and_never_claims_dynamic_pass():
    result = fuse_c2_multiview(_fixture_views())
    json.dumps(result.summary, allow_nan=False)
    assert result.summary["ground_truth_object_pose_used"] is False
    assert result.summary["semantic_truth_used"] is False
    assert result.summary["contact_truth_used"] is False
    assert result.summary["event_truth_used"] is False
    assert result.summary["dynamic_multiview_fusion_pass_claimed"] is False


def test_public_api_has_no_truth_contact_or_branch_selection_inputs():
    names = set(inspect.signature(fuse_c2_multiview).parameters)
    assert names == {"observations"}
    assert names.isdisjoint({"ground_truth_pose", "object_pose", "contact_name", "contact_normal", "event_truth", "selected_branch"})


def test_frozen_source_tamper_is_rejected(tmp_path):
    root = _copy_sources(tmp_path)
    source = root / "artifacts/agent_control/tasks/EIGHT-HOUR-C3-POSTGRASP-VIEW-PLAN/TASK_RESULT.json"
    source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        build_multilayer_multiview_fusion_contract(root)
