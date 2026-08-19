from __future__ import annotations

import copy
import inspect
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from kcg_connector.d38999_multilayer_coarse_registration import (
    CoarseRegistrationResult,
    SCHEMA_VERSION as COARSE_SCHEMA_VERSION,
)
from kcg_connector.d38999_multilayer_icp_refinement import (
    FROZEN_SOURCES,
    IcpRefinementError,
    MAXIMUM_ITERATIONS,
    build_multilayer_icp_refinement_contract,
    refine_c2_candidates_icp,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
BRANCH_IDS = ("C2_LINKED_BRANCH_0", "C2_LINKED_BRANCH_PI")


def _c2_points() -> np.ndarray:
    rng = np.random.default_rng(7)
    half = np.column_stack(
        (
            rng.uniform(0.006, 0.025, 80),
            rng.uniform(-0.018, 0.018, 80),
            rng.uniform(-0.012, 0.016, 80),
        )
    )
    partner = half.copy()
    partner[:, :2] *= -1.0
    return np.vstack((half, partner))


def _transform(rotation, translation):
    value = np.eye(4)
    value[:3, :3] = rotation
    value[:3, 3] = translation
    return value


def _fixture():
    model = _c2_points()
    true_rotation = Rotation.from_euler("xyz", (0.12, -0.16, 0.37)).as_matrix()
    true_translation = np.asarray((0.08, -0.03, 0.70))
    truth = _transform(true_rotation, true_translation)
    observed = model @ true_rotation.T + true_translation
    perturbation = _transform(
        Rotation.from_euler("xyz", (math.radians(0.8), -math.radians(0.5), math.radians(0.6))).as_matrix(),
        np.asarray((0.0010, -0.0007, 0.0005)),
    )
    initial_zero = truth @ perturbation
    rz_pi = np.diag((-1.0, -1.0, 1.0))
    initial_pi = initial_zero.copy()
    initial_pi[:3, :3] = initial_zero[:3, :3] @ rz_pi
    candidates = tuple(
        {
            "branch_id": branch_id,
            "T_camera_model": transform.tolist(),
            "selected_for_control": False,
        }
        for branch_id, transform in zip(BRANCH_IDS, (initial_zero, initial_pi))
    )
    summary = {
        "schema_version": COARSE_SCHEMA_VERSION,
        "candidate_count": 2,
        "candidate_branch_ids": list(BRANCH_IDS),
        "selected_for_control": None,
        "dynamic_registration_pass_claimed": False,
        "object_pose_truth_used": False,
        "contact_truth_used": False,
        "event_truth_used": False,
        "semantic_mask_used": False,
        "frame_id": "PalmCamera_optical",
    }
    return model, observed, truth, CoarseRegistrationResult(candidates, summary)


def _rotation_error(first, second):
    relative = np.asarray(first)[:3, :3].T @ np.asarray(second)[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(math.acos(float(cosine)))


def _copy_sources(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    for relative in FROZEN_SOURCES:
        source = REPOSITORY_ROOT / relative
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    return root


def test_current_contract_is_bounded_truth_free_and_offline_only():
    contract = build_multilayer_icp_refinement_contract(REPOSITORY_ROOT)
    assert contract["status"] == "OFFLINE_PASS"
    assert contract["maximum_iterations"] == 20
    assert contract["maximum_correspondence_distance_m"] == 0.020
    assert contract["output_candidate_count"] == 2
    assert contract["branch_selection_allowed"] is False
    assert contract["parameter_search_allowed"] is False
    assert contract["current_readiness"]["dynamic_pointclouds_available"] == 0
    assert contract["dynamic_icp_pass_claimed"] is False
    assert all(value is False for value in contract["truth_firewall"].values())


def test_both_c2_branches_refine_and_residual_decreases():
    model, observed, truth, coarse = _fixture()
    result = refine_c2_candidates_icp(
        model, observed, coarse, maximum_correspondence_distance_m=0.010
    )
    assert [item["branch_id"] for item in result.candidates] == list(BRANCH_IDS)
    assert all(item["final_rmse_m"] < item["initial_rmse_m"] for item in result.candidates)
    assert all(item["iterations"] <= MAXIMUM_ITERATIONS for item in result.candidates)
    assert all(item["converged"] is True for item in result.candidates)
    first = np.asarray(result.candidates[0]["T_camera_model"])
    assert _rotation_error(first, truth) < 1.0e-7
    assert np.linalg.norm(first[:3, 3] - truth[:3, 3]) < 1.0e-9


def test_refined_branches_remain_pi_linked_and_unselected():
    model, observed, _, coarse = _fixture()
    result = refine_c2_candidates_icp(
        model, observed, coarse, maximum_correspondence_distance_m=0.010
    )
    first = np.asarray(result.candidates[0]["T_camera_model"])
    second = np.asarray(result.candidates[1]["T_camera_model"])
    assert np.allclose(second[:3, :3], first[:3, :3] @ np.diag((-1.0, -1.0, 1.0)), atol=1.0e-9)
    assert np.allclose(second[:3, 3], first[:3, 3], atol=1.0e-9)
    assert all(item["selected_for_control"] is False for item in result.candidates)
    assert result.summary["selected_for_control"] is None
    assert result.summary["control_authorized"] is False


def test_point_order_is_deterministic():
    model, observed, _, coarse = _fixture()
    first = refine_c2_candidates_icp(
        model, observed, coarse, maximum_correspondence_distance_m=0.010
    )
    rng = np.random.default_rng(31)
    second = refine_c2_candidates_icp(
        model[rng.permutation(len(model))],
        observed[rng.permutation(len(observed))],
        coarse,
        maximum_correspondence_distance_m=0.010,
    )
    assert json.dumps(first.summary, sort_keys=True) == json.dumps(second.summary, sort_keys=True)
    for before, after in zip(first.candidates, second.candidates):
        assert np.allclose(before["T_camera_model"], after["T_camera_model"], atol=1.0e-12)


@pytest.mark.parametrize("value", [0.0, -0.001, 0.021, np.inf, np.nan])
def test_correspondence_distance_is_explicit_and_bounded(value):
    model, observed, _, coarse = _fixture()
    with pytest.raises(IcpRefinementError) as caught:
        refine_c2_candidates_icp(
            model, observed, coarse, maximum_correspondence_distance_m=value
        )
    assert caught.value.code == "INVALID_CORRESPONDENCE_BOUND"


def test_far_initial_pose_fails_closed_without_search():
    model, observed, _, coarse = _fixture()
    candidates = copy.deepcopy(list(coarse.candidates))
    for candidate in candidates:
        transform = np.asarray(candidate["T_camera_model"])
        transform[:3, 3] += 0.10
        candidate["T_camera_model"] = transform.tolist()
    far = CoarseRegistrationResult(tuple(candidates), dict(coarse.summary))
    with pytest.raises(IcpRefinementError) as caught:
        refine_c2_candidates_icp(
            model, observed, far, maximum_correspondence_distance_m=0.010
        )
    assert caught.value.code == "INSUFFICIENT_CORRESPONDENCES"


def test_selected_or_truth_augmented_candidate_is_rejected():
    model, observed, _, coarse = _fixture()
    candidates = copy.deepcopy(list(coarse.candidates))
    candidates[0]["selected_for_control"] = True
    selected = CoarseRegistrationResult(tuple(candidates), dict(coarse.summary))
    with pytest.raises(IcpRefinementError) as caught:
        refine_c2_candidates_icp(
            model, observed, selected, maximum_correspondence_distance_m=0.010
        )
    assert caught.value.code == "CONTROL_SELECTION_REJECTED"
    candidates = copy.deepcopy(list(coarse.candidates))
    candidates[0]["ground_truth_pose"] = [0.0] * 7
    truth_augmented = CoarseRegistrationResult(tuple(candidates), dict(coarse.summary))
    with pytest.raises(IcpRefinementError) as caught:
        refine_c2_candidates_icp(
            model, observed, truth_augmented, maximum_correspondence_distance_m=0.010
        )
    assert caught.value.code == "TRUTH_FIELD_REJECTED"


def test_bad_c2_relation_fails_closed():
    model, observed, _, coarse = _fixture()
    candidates = copy.deepcopy(list(coarse.candidates))
    transform = np.asarray(candidates[1]["T_camera_model"])
    transform[0, 3] += 0.001
    candidates[1]["T_camera_model"] = transform.tolist()
    bad = CoarseRegistrationResult(tuple(candidates), dict(coarse.summary))
    with pytest.raises(IcpRefinementError) as caught:
        refine_c2_candidates_icp(
            model, observed, bad, maximum_correspondence_distance_m=0.010
        )
    assert caught.value.code == "INVALID_C2_RELATION"


def test_rank_degenerate_observation_fails_closed():
    model, _, _, coarse = _fixture()
    line = np.column_stack((np.arange(20), np.zeros(20), np.zeros(20)))
    with pytest.raises(IcpRefinementError) as caught:
        refine_c2_candidates_icp(
            model, line, coarse, maximum_correspondence_distance_m=0.010
        )
    assert caught.value.code == "RANK_DEGENERATE"


def test_summary_is_finite_and_never_claims_dynamic_pass():
    model, observed, _, coarse = _fixture()
    result = refine_c2_candidates_icp(
        model, observed, coarse, maximum_correspondence_distance_m=0.010
    )
    json.dumps(result.summary, allow_nan=False)
    assert result.summary["semantic_mask_used"] is False
    assert result.summary["object_pose_truth_used"] is False
    assert result.summary["contact_truth_used"] is False
    assert result.summary["event_truth_used"] is False
    assert result.summary["dynamic_icp_pass_claimed"] is False


def test_public_api_has_no_truth_mask_contact_or_search_inputs():
    names = set(inspect.signature(refine_c2_candidates_icp).parameters)
    assert names == {
        "cad_points_model_m",
        "observed_points_camera_m",
        "coarse_result",
        "maximum_correspondence_distance_m",
    }
    assert names.isdisjoint(
        {
            "mask",
            "semantic_mask",
            "object_pose",
            "ground_truth_pose",
            "contact_report",
            "contact_name",
            "contact_normal",
            "event_truth",
            "search_parameters",
        }
    )


def test_frozen_source_tamper_is_rejected(tmp_path):
    root = _copy_sources(tmp_path)
    source = root / (
        "artifacts/agent_control/tasks/EIGHT-HOUR-C6-COARSE-REGISTRATION/"
        "COARSE_REGISTRATION_CONTRACT_MANIFEST.json"
    )
    source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        build_multilayer_icp_refinement_contract(root)
