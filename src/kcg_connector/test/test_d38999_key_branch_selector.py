import math

import numpy as np
import pytest

from kcg_connector.d38999_key_branch_selector import (
    BRANCH_IDS,
    SHADOW_HYPOTHESIS_IDS,
    SUPPORTED_KEYED_PLUG_MODEL_IDS,
    THRESHOLD_LABEL,
    blocked_key_branch_selection,
    select_key_branch_from_rgbd,
)


KEYED_V2_ID = "d38999_26kj61sn_keyed_proxy_v2"


def _observation(side="right"):
    probability = np.zeros((41, 41), dtype=np.float64)
    face = np.zeros((41, 41), dtype=np.bool_)
    face[4:37, 4:37] = True
    depth = np.full((41, 41), 0.42, dtype=np.float64)
    if side == "right":
        probability[18:23, 27:35] = 0.95
    elif side == "left":
        probability[18:23, 6:14] = 0.95
    else:
        raise ValueError(side)
    return probability, face, depth


def _select(probability, face, depth, **overrides):
    arguments = {
        "key_probability": probability,
        "face_mask": face,
        "depth_m": depth,
        "face_center_uv": (20.0, 20.0),
        "branch_directions_uv": ((1.0, 0.0), (-1.0, 0.0)),
        "keyed_model_id": KEYED_V2_ID,
        "occlusion_mask": np.zeros_like(face, dtype=np.bool_),
    }
    arguments.update(overrides)
    return select_key_branch_from_rgbd(**arguments)


def _assert_never_authorizes_control(result):
    assert result["control_authorized"] is False
    assert result["shadow_only"] is True
    assert result["confidence_calibrated"] is False
    assert result["threshold_label"] == THRESHOLD_LABEL
    assert "selected_for_control" not in result


@pytest.mark.parametrize(
    "status,model_id",
    (
        ("KEYED_MODEL_ID_UNAVAILABLE", None),
        ("KEYED_GEOMETRY_UNAVAILABLE", KEYED_V2_ID),
    ),
)
def test_explicit_preobservation_blockers_need_no_fake_probability_map(
    status, model_id
):
    result = blocked_key_branch_selection(
        status,
        "manufacturer-controlled keyed geometry has not been obtained",
        keyed_model_id=model_id,
    )

    assert result["status"] == status
    assert result["passed"] is False
    assert result["blocked_before_observation"] is True
    assert result["selected_for_shadow"] is None
    _assert_never_authorizes_control(result)


@pytest.mark.parametrize(
    "status,reason",
    (
        ("OUT_OF_FRAME", "not a pre-observation blocker"),
        ("KEYED_MODEL_ID_UNAVAILABLE", ""),
    ),
)
def test_blocked_helper_rejects_invalid_contract_values(status, reason):
    with pytest.raises(ValueError):
        blocked_key_branch_selection(status, reason)


@pytest.mark.parametrize(
    "side,expected_index",
    (("right", 0), ("left", 1)),
)
def test_success_selects_each_c2_branch_for_shadow_only(side, expected_index):
    probability, face, depth = _observation(side)
    result = _select(probability, face, depth)

    assert result["status"] == "SHADOW_BRANCH_SELECTED"
    assert result["passed"] is True
    assert result["selected_branch_index"] == expected_index
    assert result["selected_for_shadow"] == BRANCH_IDS[expected_index]
    assert result["shadow_selected_hypothesis_id"] == (
        SHADOW_HYPOTHESIS_IDS[expected_index]
    )
    assert np.allclose(
        result["key_direction_uv"],
        (1.0, 0.0) if expected_index == 0 else (-1.0, 0.0),
    )
    assert np.dot(
        result["pca_principal_axis_uv"], result["radial_direction_uv"]
    ) > 0.0
    assert result["key_region"]["support_pixels"] == 40
    assert math.isclose(result["key_region"]["mean_depth_m"], 0.42)
    _assert_never_authorizes_control(result)


def test_only_the_registered_plug_keyed_v2_identity_is_accepted():
    assert SUPPORTED_KEYED_PLUG_MODEL_IDS == {KEYED_V2_ID}
    probability, face, depth = _observation()
    result = _select(
        probability,
        face,
        depth,
        keyed_model_id="fake_keyed_v2",
    )

    assert result["status"] == "MODEL_NOT_KEYED_V2"
    assert result["reason"] == "UNREGISTERED_KEYED_V2_MODEL_ID"
    assert result["rejection_code"] == "KEYED_MODEL_ID_UNAVAILABLE"
    _assert_never_authorizes_control(result)


def test_missing_occlusion_evidence_fails_closed():
    probability, face, depth = _observation()
    result = _select(
        probability,
        face,
        depth,
        occlusion_mask=None,
    )

    assert result["status"] == "OCCLUSION_UNKNOWN"
    assert result["rejection_code"] == "KEY_REGION_OCCLUSION_UNKNOWN"
    _assert_never_authorizes_control(result)


def test_face_occlusion_above_limit_is_rejected():
    probability, face, depth = _observation()
    occlusion = np.zeros_like(face)
    occlusion[4:16, 4:37] = True

    result = _select(
        probability, face, depth, occlusion_mask=occlusion
    )

    assert result["status"] == "OCCLUDED"
    assert result["reason"] == "FACE_OCCLUSION_ABOVE_LIMIT"
    assert result["selected_for_shadow"] is None
    _assert_never_authorizes_control(result)


def test_key_occlusion_above_limit_is_rejected_even_if_face_fraction_is_small():
    probability, face, depth = _observation()
    occlusion = np.zeros_like(face)
    occlusion[18:23, 27:31] = True

    result = _select(
        probability, face, depth, occlusion_mask=occlusion
    )

    assert result["status"] == "OCCLUDED"
    assert result["reason"] == "KEY_OCCLUSION_ABOVE_LIMIT"
    _assert_never_authorizes_control(result)


def test_face_touching_image_border_is_out_of_frame():
    probability, face, depth = _observation()
    face[0:5, 10:15] = True

    result = _select(probability, face, depth)

    assert result["status"] == "OUT_OF_FRAME"
    assert result["reason"] == "FACE_TOUCHES_IMAGE_BORDER"
    _assert_never_authorizes_control(result)


def test_key_touching_image_border_is_out_of_frame():
    probability, face, depth = _observation()
    face[0:5, 27:35] = True
    probability[0:5, 27:35] = 0.99

    result = _select(probability, face, depth)

    assert result["status"] == "OUT_OF_FRAME"
    assert result["reason"] == "KEY_TOUCHES_IMAGE_BORDER"
    _assert_never_authorizes_control(result)


def test_two_pixel_border_margin_is_enforced_before_actual_edge_contact():
    probability, face, depth = _observation()
    face[:] = False
    face[1:37, 4:37] = True
    probability[:] = 0.0
    probability[18:23, 27:35] = 0.95

    result = _select(probability, face, depth)

    assert result["status"] == "OUT_OF_FRAME"
    assert result["reason"] == "FACE_TOUCHES_IMAGE_BORDER"
    assert result["rejection_code"] == "CONNECTOR_FACE_OUT_OF_FRAME"
    _assert_never_authorizes_control(result)


def test_face_center_outside_image_is_out_of_frame_observation_failure():
    probability, face, depth = _observation()
    result = _select(
        probability, face, depth, face_center_uv=(99.0, 20.0)
    )

    assert result["status"] == "OUT_OF_FRAME"
    assert result["reason"] == "FACE_CENTER_OUTSIDE_IMAGE"
    _assert_never_authorizes_control(result)


def test_face_center_must_lie_inside_the_observed_face_mask():
    probability, face, depth = _observation()
    face[19:22, 19:22] = False

    result = _select(probability, face, depth)

    assert result["status"] == "LOW_CONFIDENCE"
    assert result["reason"] == "FACE_CENTER_OUTSIDE_FACE_MASK"
    _assert_never_authorizes_control(result)


def test_missing_key_depth_is_rejected():
    probability, face, depth = _observation()
    depth[18:23, 27:35] = np.nan

    result = _select(probability, face, depth)

    assert result["status"] == "DEPTH_MISSING"
    assert result["reason"] == "VALID_KEY_DEPTH_BELOW_LIMIT"
    assert result["valid_key_depth_fraction"] == 0.0
    _assert_never_authorizes_control(result)


@pytest.mark.parametrize(
    "setup,reason",
    (
        ("low_probability", "KEY_SUPPORT_TOO_SMALL"),
        ("few_pixels", "KEY_SUPPORT_TOO_SMALL"),
        ("low_mean", "KEY_MEAN_PROBABILITY_TOO_LOW"),
    ),
)
def test_low_probability_or_support_is_low_confidence(setup, reason):
    probability, face, depth = _observation()
    if setup == "low_probability":
        probability[:] = 0.0
    elif setup == "few_pixels":
        probability[:] = 0.0
        probability[20, 29:32] = 0.95
    elif setup == "low_mean":
        probability[18:23, 27:35] = 0.65

    result = _select(probability, face, depth)

    assert result["status"] == "LOW_CONFIDENCE"
    assert result["reason"] == reason
    _assert_never_authorizes_control(result)


def test_key_region_that_is_too_large_is_low_confidence():
    probability, face, depth = _observation()
    probability[face] = 0.95

    result = _select(probability, face, depth)

    assert result["status"] == "LOW_CONFIDENCE"
    assert result["reason"] == "KEY_AREA_FRACTION_TOO_LARGE"
    _assert_never_authorizes_control(result)


def test_two_comparable_key_components_are_not_silently_discarded():
    probability, face, depth = _observation()
    probability[18:23, 7:15] = 0.90

    result = _select(probability, face, depth)

    assert result["status"] == "LOW_CONFIDENCE"
    assert result["reason"] == "MULTIPLE_COMPARABLE_KEY_COMPONENTS"
    _assert_never_authorizes_control(result)


def test_radial_center_degeneracy_is_ambiguous():
    probability, face, depth = _observation()
    result = _select(
        probability,
        face,
        depth,
        face_center_uv=(30.5, 20.0),
    )

    assert result["status"] == "AMBIGUOUS"
    assert result["reason"] == "RADIAL_LENGTH_DEGENERATE"
    _assert_never_authorizes_control(result)


def test_branch_margin_below_limit_is_ambiguous():
    probability, face, depth = _observation()
    diagonal = math.sqrt(0.5)
    result = _select(
        probability,
        face,
        depth,
        branch_directions_uv=((diagonal, diagonal), (diagonal, -diagonal)),
        thresholds={"maximum_branch_angle_error_deg": 60.0},
    )

    assert result["status"] == "AMBIGUOUS"
    assert result["reason"] == "BRANCH_MARGIN_BELOW_LIMIT"
    assert math.isclose(result["branch_margin_deg"], 0.0, abs_tol=1.0e-9)
    _assert_never_authorizes_control(result)


def test_best_branch_angle_error_above_limit_is_ambiguous():
    probability, face, depth = _observation()
    result = _select(
        probability,
        face,
        depth,
        branch_directions_uv=((0.0, 1.0), (0.0, -1.0)),
    )

    assert result["status"] == "AMBIGUOUS"
    assert result["reason"] == "BEST_BRANCH_ANGLE_ERROR_ABOVE_LIMIT"
    _assert_never_authorizes_control(result)


def test_branch_choice_uses_radial_key_location_not_tangential_pca_axis():
    probability, face, depth = _observation()
    probability[:] = 0.0
    probability[11:30, 29:32] = 0.95

    result = _select(probability, face, depth)

    assert result["status"] == "SHADOW_BRANCH_SELECTED"
    assert result["selected_branch_index"] == 0
    assert result["key_direction_method"] == (
        "FACE_CENTER_TO_WEIGHTED_KEY_CENTROID"
    )
    assert abs(result["pca_principal_axis_uv"][1]) > 0.9
    assert result["key_direction_uv"][0] > 0.9
    _assert_never_authorizes_control(result)


@pytest.mark.parametrize(
    "model_id,reason",
    (
        (None, "KEYED_MODEL_ID_MISSING"),
        ("", "KEYED_MODEL_ID_MISSING"),
        ("d38999_shell25j_proxy_v1", "OLD_OR_UNKEYED_MODEL_ID"),
        ("C2_LINKED_BRANCH_0", "OLD_OR_UNKEYED_MODEL_ID"),
    ),
)
def test_missing_old_v1_or_c2_model_identity_is_rejected(model_id, reason):
    probability, face, depth = _observation()
    result = _select(
        probability, face, depth, keyed_model_id=model_id
    )

    assert result["status"] == "MODEL_NOT_KEYED_V2"
    assert result["reason"] == reason
    assert result["selected_for_shadow"] is None
    _assert_never_authorizes_control(result)


def test_observation_value_failures_return_result_instead_of_raising():
    probability, face, depth = _observation()
    probability[20, 30] = np.nan

    result = _select(probability, face, depth)

    assert result["status"] == "LOW_CONFIDENCE"
    assert result["reason"] == "KEY_PROBABILITY_INVALID"
    _assert_never_authorizes_control(result)


@pytest.mark.parametrize(
    "overrides",
    (
        {"key_probability": np.zeros((3, 3, 1))},
        {"depth_m": np.zeros((40, 41))},
        {"face_mask": np.zeros((41, 41), dtype=np.uint8)},
        {"branch_directions_uv": ((1.0, 0.0),)},
        {"branch_directions_uv": ((0.0, 0.0), (-1.0, 0.0))},
        {"occlusion_mask": np.zeros((40, 41), dtype=np.bool_)},
        {"thresholds": {"minimum_key_support_pixels": 1}},
        {"thresholds": {"image_border_margin_px": 1.5}},
        {"thresholds": {"unknown_gate": 1.0}},
        {"thresholds": {"threshold_label": "REAL_CALIBRATED"}},
    ),
)
def test_invalid_shapes_or_threshold_configuration_raise_value_error(overrides):
    probability, face, depth = _observation()
    with pytest.raises(ValueError):
        _select(probability, face, depth, **overrides)
