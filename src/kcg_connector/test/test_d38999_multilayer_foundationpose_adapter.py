from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from kcg_connector.d38999_multilayer_foundationpose_adapter import (
    FROZEN_SOURCES,
    FoundationPoseAdapterError,
    MASK_PROVENANCE,
    adapt_foundationpose_estimate_to_c2,
    build_multilayer_foundationpose_adapter_contract,
    validate_foundationpose_input_envelope,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = Path(__file__).parents[1] / "kcg_connector" / "d38999_multilayer_foundationpose_adapter.py"


def _intrinsics():
    return {"width": 3, "height": 2, "fx": 100.0, "fy": 100.0, "cx": 1.0, "cy": 0.5}


def _copy_sources(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    for relative in FROZEN_SOURCES:
        source = REPOSITORY_ROOT / relative
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    return root


def test_current_contract_is_static_ready_but_runtime_blocked():
    contract = build_multilayer_foundationpose_adapter_contract(REPOSITORY_ROOT)
    assert contract["status"] == "STATIC_PASS_RUNTIME_BLOCKED_EXTERNAL"
    assert contract["visual_representation"] == "D38999_VISUAL_COMPLETE_V1"
    assert contract["visible_geometry_preserved"] is True
    assert contract["legacy_bootstrap"]["onnx_models_verified"] == 2
    assert contract["legacy_bootstrap"]["legacy_proxy_meshes_verified"] == 3
    assert contract["legacy_bootstrap"]["runtime_environment_ready"] is False
    assert contract["current_multilayer_obj_mesh_verified"] is False
    assert "current_multilayer_visual_obj_export_not_verified" in contract["blockers"]
    assert contract["inference_execution_authorized"] is False
    assert contract["dynamic_foundationpose_pass_claimed"] is False


def test_contract_exposes_exact_truth_and_c2_boundaries():
    contract = build_multilayer_foundationpose_adapter_contract(REPOSITORY_ROOT)
    assert contract["allowed_mask_provenance"] == MASK_PROVENANCE
    assert contract["simulator_semantic_mask_allowed"] is False
    assert contract["output_adapter"]["candidate_count"] == 2
    assert contract["output_adapter"]["branch_ids"] == [
        "C2_LINKED_BRANCH_0",
        "C2_LINKED_BRANCH_PI",
    ]
    assert contract["output_adapter"]["selected_for_control"] is None
    assert all(value is False for value in contract["truth_firewall"].values())


def test_input_envelope_accepts_only_image_derived_mask():
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    depth = np.ones((2, 3), dtype=np.float32)
    mask = np.asarray([[True, False, False], [False, True, False]])
    result = validate_foundationpose_input_envelope(
        rgb,
        depth,
        _intrinsics(),
        mask,
        frame_id="PalmCamera_optical",
        mask_provenance=MASK_PROVENANCE,
    )
    assert result.summary["valid_masked_depth_count"] == 2
    assert result.summary["simulator_semantic_truth_used"] is False
    assert result.summary["inference_performed"] is False
    assert result.summary["control_authorized"] is False


@pytest.mark.parametrize(
    "provenance",
    ["SIMULATOR_SEMANTIC_TRUTH", "USD_PRIM_LABEL", "CONTACT_NAME_MASK", ""],
)
def test_truth_or_unknown_mask_provenance_is_rejected(provenance):
    with pytest.raises(FoundationPoseAdapterError) as caught:
        validate_foundationpose_input_envelope(
            np.zeros((2, 3, 3), dtype=np.uint8),
            np.ones((2, 3)),
            _intrinsics(),
            np.ones((2, 3), dtype=bool),
            frame_id="camera",
            mask_provenance=provenance,
        )
    assert caught.value.code == "MASK_PROVENANCE_REJECTED"


def test_input_shape_dtype_intrinsics_and_valid_depth_fail_closed():
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    depth = np.ones((2, 3))
    mask = np.ones((2, 3), dtype=bool)
    with pytest.raises(FoundationPoseAdapterError, match="SHAPE_MISMATCH"):
        validate_foundationpose_input_envelope(
            rgb[:, :2], depth, _intrinsics(), mask, frame_id="camera", mask_provenance=MASK_PROVENANCE
        )
    with pytest.raises(FoundationPoseAdapterError, match="INVALID_DTYPE"):
        validate_foundationpose_input_envelope(
            rgb.astype(float), depth, _intrinsics(), mask, frame_id="camera", mask_provenance=MASK_PROVENANCE
        )
    bad = _intrinsics(); bad["width"] = 4
    with pytest.raises(FoundationPoseAdapterError, match="INVALID_INTRINSICS"):
        validate_foundationpose_input_envelope(
            rgb, depth, bad, mask, frame_id="camera", mask_provenance=MASK_PROVENANCE
        )
    with pytest.raises(FoundationPoseAdapterError, match="NO_VALID_MASKED_DEPTH"):
        validate_foundationpose_input_envelope(
            rgb, np.full_like(depth, np.nan), _intrinsics(), mask, frame_id="camera", mask_provenance=MASK_PROVENANCE
        )


def test_external_estimate_expands_to_two_unselected_c2_branches():
    estimate = np.eye(4)
    estimate[:3, 3] = (0.1, -0.2, 0.7)
    result = adapt_foundationpose_estimate_to_c2(
        estimate,
        frame_id="PalmCamera_optical",
        inference_provenance="OFFLINE_FIXTURE_ONLY",
    )
    first = np.asarray(result.candidates[0]["T_camera_model"])
    second = np.asarray(result.candidates[1]["T_camera_model"])
    assert np.allclose(second[:3, :3], first[:3, :3] @ np.diag((-1.0, -1.0, 1.0)))
    assert np.allclose(second[:3, 3], first[:3, 3])
    assert [item["branch_id"] for item in result.candidates] == [
        "C2_LINKED_BRANCH_0",
        "C2_LINKED_BRANCH_PI",
    ]
    assert all(item["selected_for_control"] is False for item in result.candidates)
    assert result.summary["selected_for_control"] is None
    assert result.summary["dynamic_foundationpose_pass_claimed"] is False
    assert result.summary["control_authorized"] is False


def test_bad_estimate_and_provenance_fail_closed():
    bad = np.eye(4); bad[0, 0] = 2.0
    with pytest.raises(FoundationPoseAdapterError) as caught:
        adapt_foundationpose_estimate_to_c2(
            bad, frame_id="camera", inference_provenance="OFFLINE_FIXTURE_ONLY"
        )
    assert caught.value.code == "INVALID_ESTIMATE"
    with pytest.raises(FoundationPoseAdapterError) as caught:
        adapt_foundationpose_estimate_to_c2(
            np.eye(4), frame_id="camera", inference_provenance="SIMULATOR_TRUTH"
        )
    assert caught.value.code == "INFERENCE_PROVENANCE_REJECTED"


def test_module_has_no_foundationpose_gpu_ros_or_isaac_imports():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    assert roots.isdisjoint(
        {"torch", "tensorrt", "onnx", "onnxruntime", "rclpy", "isaacsim", "omni", "pxr", "FoundationPose", "foundationpose"}
    )


def test_public_apis_have_no_ground_truth_contact_or_control_switch():
    names = set(inspect.signature(validate_foundationpose_input_envelope).parameters)
    names |= set(inspect.signature(adapt_foundationpose_estimate_to_c2).parameters)
    assert names.isdisjoint(
        {"ground_truth_pose", "object_pose", "contact_name", "contact_normal", "collider", "event_truth", "select_for_control"}
    )


def test_summaries_are_finite_json():
    envelope = validate_foundationpose_input_envelope(
        np.zeros((2, 3, 3), dtype=np.uint8),
        np.ones((2, 3)),
        _intrinsics(),
        np.ones((2, 3), dtype=bool),
        frame_id="camera",
        mask_provenance=MASK_PROVENANCE,
    )
    adapted = adapt_foundationpose_estimate_to_c2(
        np.eye(4), frame_id="camera", inference_provenance="EXTERNAL_UNVERIFIED"
    )
    json.dumps(envelope.summary, allow_nan=False)
    json.dumps(adapted.summary, allow_nan=False)


def test_frozen_source_tamper_is_rejected(tmp_path):
    root = _copy_sources(tmp_path)
    source = root / (
        "artifacts/agent_control/tasks/EIGHT-HOUR-C7-ICP-REFINEMENT/"
        "ICP_REFINEMENT_CONTRACT_MANIFEST.json"
    )
    source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        build_multilayer_foundationpose_adapter_contract(root)
