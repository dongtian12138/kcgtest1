import inspect
import math

import numpy as np

from kcg_connector.d38999_key_region_detector import SUPPORTED_KEYED_MODEL_ID
from kcg_connector.d38999_key_shadow_pipeline import (
    run_palm_key_shadow_pipeline,
)


KEY_ANGLES_DEG = (0.0, 80.0, 142.0, 196.0, 293.0)


def _observation(rotation_deg=0.0):
    shape = (181, 181)
    center = (90.0, 90.0)
    rows, columns = np.indices(shape, dtype=np.float64)
    du = columns - center[0]
    dv = rows - center[1]
    radii = np.hypot(du, dv)
    angles = np.mod(np.arctan2(dv, du), 2.0 * math.pi)
    face = radii <= 48.0
    for index, base_angle in enumerate(KEY_ANGLES_DEG):
        angle = math.radians((base_angle + rotation_deg) % 360.0)
        width = math.radians(16.0 if index == 0 else 8.0)
        delta = np.abs(np.angle(np.exp(1j * (angles - angle))))
        face |= (delta <= 0.5 * width) & (radii <= 58.0)
    depth = np.full(shape, np.nan, dtype=np.float64)
    depth[face] = 0.42
    occlusion = np.zeros(shape, dtype=np.bool_)
    return face, depth, occlusion, center


def _run(rotation_deg=0.0, **overrides):
    face, depth, occlusion, center = _observation(rotation_deg)
    arguments = {
        "connector_face_mask": face,
        "depth_m": depth,
        "face_center_uv": center,
        "branch_directions_uv": ((1.0, 0.0), (-1.0, 0.0)),
        "keyed_model_id": SUPPORTED_KEYED_MODEL_ID,
        "occlusion_mask": occlusion,
    }
    arguments.update(overrides)
    return run_palm_key_shadow_pipeline(**arguments)


def _assert_no_control(result):
    assert result["shadow_only"] is True
    assert result["control_authorized"] is False
    assert result["selected_for_control_allowed"] is False
    assert "selected_for_control" not in result


def test_clear_n_face_selects_yaw_zero_for_shadow_only():
    result = _run()
    assert result["status"] == "SHADOW_C2_BRANCH_SELECTED"
    assert result["passed"] is True
    assert result["selected_for_shadow"] == "C2_LINKED_BRANCH_0"
    assert result["shadow_selected_hypothesis_id"] == "YAW_0"
    assert result["key_region_detection"]["passed"] is True
    assert result["key_branch_selection"]["passed"] is True
    _assert_no_control(result)


def test_pi_rotated_n_face_selects_other_shadow_branch():
    result = _run(180.0)
    assert result["status"] == "SHADOW_C2_BRANCH_SELECTED"
    assert result["selected_for_shadow"] == "C2_LINKED_BRANCH_PI"
    assert result["shadow_selected_hypothesis_id"] == "YAW_PI"
    _assert_no_control(result)


def test_occlusion_and_missing_depth_stop_before_branch_selection():
    face, depth, occlusion, _ = _observation()
    occlusion[85:96, 136:151] = True
    result = _run(occlusion_mask=occlusion)
    assert result["status"] == "REJECTED_KEY_REGION_DETECTION"
    assert result["rejection_code"] == "KEY_REGION_OCCLUDED"
    assert result["key_branch_selection"] is None
    _assert_no_control(result)

    depth[90, 140] = np.nan
    result = _run(depth_m=depth)
    assert result["status"] == "REJECTED_KEY_REGION_DETECTION"
    assert result["rejection_code"] == "KEY_REGION_DEPTH_MISSING"
    assert result["key_branch_selection"] is None
    _assert_no_control(result)


def test_old_model_and_ambiguous_branches_fail_closed():
    result = _run(keyed_model_id="d38999_shell25j_proxy_v1")
    assert result["status"] == "REJECTED_KEY_REGION_DETECTION"
    assert result["rejection_code"] == "KEYED_MODEL_ID_UNAVAILABLE"
    _assert_no_control(result)

    diagonal = math.sqrt(0.5)
    result = _run(
        branch_directions_uv=((diagonal, diagonal), (diagonal, -diagonal))
    )
    assert result["status"] == "REJECTED_KEY_BRANCH_SELECTION"
    assert result["rejection_code"] == "KEY_BRANCH_AMBIGUOUS"
    _assert_no_control(result)


def test_public_pipeline_signature_has_no_truth_or_control_inputs():
    parameters = tuple(inspect.signature(run_palm_key_shadow_pipeline).parameters)
    assert parameters == (
        "connector_face_mask",
        "depth_m",
        "face_center_uv",
        "branch_directions_uv",
        "keyed_model_id",
        "occlusion_mask",
    )
    forbidden = {
        "semantic_segmentation_truth",
        "object_pose_truth",
        "key_angle_truth",
        "collider_identity",
        "contact_report",
        "selected_for_control",
    }
    assert forbidden.isdisjoint(parameters)

