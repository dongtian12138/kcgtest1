from __future__ import annotations

import inspect
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from kcg_connector.d38999_cad_registration import shell25j_plug_cad_profile
from kcg_connector.d38999_multilayer_coarse_registration import (
    BEZOUT_COEFFICIENTS,
    CoarseRegistrationError,
    FROZEN_SOURCES,
    HARMONIC_ORDERS,
    build_multilayer_coarse_registration_contract,
    coarse_register_c2,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _authoritative_points() -> np.ndarray:
    return shell25j_plug_cad_profile("shell_plus_socket").plug_mating.xyz


def _transform(points: np.ndarray):
    rotation = Rotation.from_euler("xyz", (0.18, -0.21, 0.43)).as_matrix()
    translation = np.asarray((0.12, -0.04, 0.72))
    assert (rotation @ np.asarray((0.0, 0.0, 1.0)))[2] > 0.0
    return points @ rotation.T + translation, rotation, translation


def _rotation_distance(first: np.ndarray, second: np.ndarray) -> float:
    relative = first.T @ second
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(math.acos(cosine))


def _copy_sources(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    for relative in FROZEN_SOURCES:
        source = REPOSITORY_ROOT / relative
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    return root


def test_current_contract_is_c2_truth_free_and_offline_only():
    contract = build_multilayer_coarse_registration_contract(REPOSITORY_ROOT)
    assert contract["status"] == "OFFLINE_PASS"
    assert contract["cad_symmetry_order"] == 2
    assert contract["harmonic_orders"] == [6, 20]
    assert contract["bezout_coefficients"] == [7, -2]
    assert contract["candidate_branch_ids"] == [
        "C2_LINKED_BRANCH_0",
        "C2_LINKED_BRANCH_PI",
    ]
    assert contract["candidate_count"] == 2
    assert contract["parameter_search_allowed"] is False
    assert contract["current_readiness"]["dynamic_pointclouds_available"] == 0
    assert contract["dynamic_registration_pass_claimed"] is False
    assert all(value is False for value in contract["truth_firewall"].values())
    assert len(contract["sources"]) == len(FROZEN_SOURCES)


def test_authoritative_cad_transform_is_present_in_two_c2_candidates():
    model = _authoritative_points()
    observed, expected_rotation, expected_translation = _transform(model)
    result = coarse_register_c2(model, observed, frame_id="PalmCamera_optical")
    rotations = [np.asarray(item["rotation_model_to_camera"]) for item in result.candidates]
    closest = int(np.argmin([_rotation_distance(value, expected_rotation) for value in rotations]))
    assert _rotation_distance(rotations[closest], expected_rotation) < 1.0e-7
    assert np.allclose(
        result.candidates[closest]["translation_camera_m"],
        expected_translation,
        atol=1.0e-10,
    )
    assert result.summary["candidate_count"] == 2
    assert result.summary["selected_for_control"] is None
    assert result.summary["dynamic_registration_pass_claimed"] is False


def test_two_candidates_are_exactly_related_by_model_z_pi_action():
    model = _authoritative_points()
    observed, _, _ = _transform(model)
    result = coarse_register_c2(model, observed, frame_id="camera")
    first = np.asarray(result.candidates[0]["rotation_model_to_camera"])
    second = np.asarray(result.candidates[1]["rotation_model_to_camera"])
    rz_pi = np.diag((-1.0, -1.0, 1.0))
    assert np.allclose(second, first @ rz_pi, atol=1.0e-12)
    assert [item["branch_id"] for item in result.candidates] == [
        "C2_LINKED_BRANCH_0",
        "C2_LINKED_BRANCH_PI",
    ]
    assert all(item["selected_for_control"] is False for item in result.candidates)


def test_unpaired_point_order_does_not_change_candidate_set():
    model = _authoritative_points()
    observed, _, _ = _transform(model)
    first = coarse_register_c2(model, observed, frame_id="camera")
    permutation = np.random.default_rng(17).permutation(len(observed))
    second = coarse_register_c2(model, observed[permutation], frame_id="camera")
    for before, after in zip(first.candidates, second.candidates):
        assert np.allclose(before["T_camera_model"], after["T_camera_model"], atol=1.0e-12)


def test_bezout_identity_has_exactly_c2_order():
    assert math.gcd(*HARMONIC_ORDERS) == 2
    assert sum(
        coefficient * order
        for coefficient, order in zip(BEZOUT_COEFFICIENTS, HARMONIC_ORDERS)
    ) == 2


@pytest.mark.parametrize(
    "points,code",
    [
        (np.zeros((7, 3)), "INSUFFICIENT_POINTS"),
        (np.column_stack((np.arange(8), np.zeros(8), np.zeros(8))), "RANK_DEGENERATE"),
        (np.full((8, 3), np.nan), "NONFINITE_POINTS"),
    ],
)
def test_bad_observations_fail_closed(points, code):
    with pytest.raises(CoarseRegistrationError) as caught:
        coarse_register_c2(_authoritative_points(), points, frame_id="camera")
    assert caught.value.code == code


def test_axisymmetric_points_have_no_c2_phase_and_fail_closed():
    angles = np.linspace(0.0, 2.0 * math.pi, 32, endpoint=False)
    model = np.vstack(
        [
            np.column_stack(
                (np.cos(angles), np.sin(angles), np.full(len(angles), z))
            )
            for z in (-0.5, 0.5)
        ]
    )
    observed = model + np.asarray((0.0, 0.0, 2.0))
    with pytest.raises(CoarseRegistrationError) as caught:
        coarse_register_c2(model, observed, frame_id="camera")
    assert caught.value.code == "C2_HARMONIC_DEGENERATE"


def test_invalid_frame_fails_closed():
    model = _authoritative_points()
    with pytest.raises(CoarseRegistrationError) as caught:
        coarse_register_c2(model, model, frame_id="")
    assert caught.value.code == "INVALID_FRAME_ID"


def test_summary_is_finite_json_and_preserves_truth_firewall():
    model = _authoritative_points()
    observed, _, _ = _transform(model)
    result = coarse_register_c2(model, observed, frame_id="camera")
    json.dumps(result.summary, allow_nan=False)
    assert result.summary["semantic_mask_used"] is False
    assert result.summary["object_pose_truth_used"] is False
    assert result.summary["contact_truth_used"] is False
    assert result.summary["event_truth_used"] is False
    assert result.summary["postrun_object_pose_write_count"] == 0
    assert result.summary["control_authorized"] is False


def test_public_api_has_no_truth_mask_contact_or_search_inputs():
    names = set(inspect.signature(coarse_register_c2).parameters)
    assert names == {
        "cad_points_model_m",
        "observed_points_camera_m",
        "frame_id",
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
        "artifacts/agent_control/tasks/EIGHT-HOUR-C5-OPEN3D-PREPROCESS/"
        "PREPROCESS_CONTRACT_MANIFEST.json"
    )
    source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        build_multilayer_coarse_registration_contract(root)
