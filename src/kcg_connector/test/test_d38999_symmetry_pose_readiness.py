"""Pure tests for symmetry-aware pose metrics and provider adaptation."""

import ast
from copy import deepcopy
import json
import math
from pathlib import Path

import pytest
import yaml

from kcg_connector.connector_pose import load_connector_pose_contract
from kcg_connector.d38999_symmetry_pose_readiness import (
    CANDIDATE_SCHEMA_VERSION,
    DEFAULT_CONFIG_PATH,
    SCHEMA_VERSION,
    adapt_foundationpose_candidate_to_pose_provider,
    evaluate_pose_modulo_symmetry,
    evaluate_symmetry_pose_readiness,
    load_symmetry_pose_readiness_contract,
    parse_foundationpose_candidate_pair,
)
from kcg_connector.pose_provider import PoseProviderPurpose


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    Path(__file__).parents[1]
    / "kcg_connector"
    / "d38999_symmetry_pose_readiness.py"
)
E2E_RUNNER_PATH = (
    Path(__file__).parents[1]
    / "isaac"
    / "d38999_tabletop_pick_smoke.py"
)


def _contract():
    return load_symmetry_pose_readiness_contract()


def _candidate(timestamp_s=10.0):
    contract = _contract()
    covariance = [
        [1.0e-6 if row == column else 0.0 for column in range(6)]
        for row in range(6)
    ]
    endpoints = {}
    for role, position in (
        ("loose_plug", [0.52, -0.21, 0.215]),
        ("fixed_receptacle", [0.55, 0.185, 0.2615]),
    ):
        spec = contract.symmetry[role]
        endpoints[role] = {
            "model_id": spec.model_id,
            "mesh_id": spec.mesh_input,
            "mesh_sha256": contract.inputs[spec.mesh_input].sha256,
            "position_xyz_m": position,
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            "covariance_6x6": covariance,
            "confidence": 0.95,
        }
    return {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "capture_id": "capture-0001",
        "inference_id": "foundationpose-0001",
        "timestamp_s": timestamp_s,
        "clock_domain": "simulation_time",
        "control_frame": "world",
        "calibration_sha256": contract.adapter.required_calibration_sha256,
        "model_version": "1.0.1_onnx",
        "refine_model_sha256": (
            "dcc695a19c4bcfe5e1d909a22d8f652d8ec8bab1e19bd1544c6b45f2d3595cf7"
        ),
        "score_model_sha256": (
            "0bf1026c0db7320ebf9a548ecf0d3c810c8dbd377948630bd3e5af1d49440503"
        ),
        "endpoints": endpoints,
    }


def _mutated(tmp_path, mutate):
    document = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    mutate(document)
    output = tmp_path / "mutated.yaml"
    output.write_text(yaml.safe_dump(document), encoding="utf-8")
    return output


def test_contract_is_pure_disabled_and_not_wired_into_e2e():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    assert roots.isdisjoint(
        {
            "isaacsim",
            "omni",
            "pxr",
            "rclpy",
            "torch",
            "tensorrt",
            "onnx",
            "cv2",
            "open3d",
            "numpy",
        }
    )
    contract = _contract()
    assert contract.schema_version == SCHEMA_VERSION
    assert contract.enabled is False
    assert all(value is False for value in contract.boundaries.values())
    source = E2E_RUNNER_PATH.read_text(encoding="utf-8")
    assert "d38999_symmetry_pose_readiness" not in source


def test_pose_modulo_order_two_accepts_pi_equivalent_yaw():
    contract = _contract()
    result = evaluate_pose_modulo_symmetry(
        estimated_position_xyz_m=(0.501, 0.199, 0.3005),
        estimated_quaternion_xyzw=(0.0, 0.0, 1.0, 0.0),
        truth_position_xyz_m=(0.5, 0.2, 0.3),
        truth_quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
        symmetry_order=2,
        unique_key_observed=False,
        thresholds=contract.accuracy,
    )
    assert result.selected_symmetry_index == 1
    assert result.orientation_error_modulo_symmetry_rad == pytest.approx(0.0)
    assert result.symmetry_aware_pose_passed is True
    assert result.unique_key_yaw_observable is False
    assert result.unique_key_yaw_error_rad is None
    assert result.unique_key_yaw_rejection_reason == "unique_key_not_observed"
    assert result.vision_control_qualified is False


def test_modulo_metric_still_rejects_axis_translation_and_residual_errors():
    contract = _contract()
    tilted = (
        math.sin(math.radians(5.0)),
        0.0,
        0.0,
        math.cos(math.radians(5.0)),
    )
    result = evaluate_pose_modulo_symmetry(
        estimated_position_xyz_m=(0.52, 0.2, 0.3),
        estimated_quaternion_xyzw=tilted,
        truth_position_xyz_m=(0.5, 0.2, 0.3),
        truth_quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
        symmetry_order=2,
        unique_key_observed=False,
        thresholds=contract.accuracy,
    )
    assert result.translation_passed is False
    assert result.axis_passed is False
    assert result.orientation_modulo_symmetry_passed is False
    assert result.symmetry_aware_pose_passed is False


def test_unique_key_claim_contradicting_order_two_is_rejected():
    with pytest.raises(ValueError, match="contradicts"):
        evaluate_pose_modulo_symmetry(
            estimated_position_xyz_m=(0.0, 0.0, 0.0),
            estimated_quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
            truth_position_xyz_m=(0.0, 0.0, 0.0),
            truth_quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
            symmetry_order=2,
            unique_key_observed=True,
            thresholds=_contract().accuracy,
        )


def test_five_anchor_and_three_mesh_evidence_is_bound_but_partial():
    report = evaluate_symmetry_pose_readiness(_contract(), PROJECT_ROOT)
    json.dumps(report, allow_nan=False)
    assert report["status"] == "METRIC_AND_ADAPTER_STATIC_READY_PAIR_BLOCKED"
    assert report["gates"]["content_addressed_inputs_verified"] is True
    assert report["gates"]["three_obj_mesh_bundle_verified"] is True
    assert report["gates"]["five_anchor_xy_accuracy_passed"] is True
    assert report["gates"]["xyz_translation_accuracy_qualified"] is False
    assert report["gates"]["axis_accuracy_qualified"] is False
    assert (
        report["gates"]["orientation_modulo_symmetry_accuracy_qualified"]
        is False
    )
    assert report["gates"]["unique_key_yaw_qualified"] is False
    assert report["gates"]["pose_provider_pair_ready"] is False
    assert report["gates"]["vision_control_authorized"] is False
    assert report["five_anchor_rgbd"]["endpoints"]["loose_plug"][
        "maximum_xy_error_m"
    ] == pytest.approx(0.00250333936022795)
    assert report["five_anchor_rgbd"]["endpoints"]["fixed_receptacle"][
        "maximum_xy_error_m"
    ] == pytest.approx(0.006183311038175034)


def test_candidate_parser_binds_models_meshes_calibration_and_numeric_pose():
    contract = _contract()
    candidate = parse_foundationpose_candidate_pair(
        _candidate(),
        contract,
        load_connector_pose_contract(),
        now_s=10.1,
    )
    assert candidate.model_version == "1.0.1_onnx"
    assert set(candidate.endpoints) == {"loose_plug", "fixed_receptacle"}
    assert candidate.endpoints["loose_plug"].confidence == 0.95

    bad = deepcopy(_candidate())
    bad["endpoints"]["loose_plug"]["mesh_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="mesh hash differs"):
        parse_foundationpose_candidate_pair(
            bad, contract, load_connector_pose_contract(), now_s=10.1
        )


def test_adapter_returns_strict_diagnostic_only_pose_provider_sample():
    sample = adapt_foundationpose_candidate_to_pose_provider(
        _candidate(),
        _contract(),
        load_connector_pose_contract(),
        purpose=PoseProviderPurpose.EVALUATION,
        now_s=10.1,
    )
    assert sample.purpose is PoseProviderPurpose.EVALUATION
    assert sample.pair is None
    assert sample.reference_truth_pair is None
    assert sample.full_6d is False
    assert sample.keyed_orientation_observed is False
    assert sample.uses_truth_position is False
    assert sample.uses_truth_orientation is False
    assert sample.preflight_passed is False
    assert sample.control_authorized is False
    assert sample.diagnostics["registry_symmetry_compatible"] is False
    assert (
        "pose_registry_claims_keyed_order_1_but_proxy_is_order_2"
        in sample.diagnostics["pair_publication_blockers"]
    )


def test_adapter_rejects_control_purpose_before_parsing_candidate():
    with pytest.raises(ValueError, match="rejects control purpose"):
        adapt_foundationpose_candidate_to_pose_provider(
            _candidate(),
            _contract(),
            load_connector_pose_contract(),
            purpose=PoseProviderPurpose.CONTROL,
            now_s=10.1,
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (lambda doc: doc.update({"enabled": True}), "must remain disabled"),
        (
            lambda doc: doc["symmetry"]["objects"]["loose_plug"].update(
                {"rotational_symmetry_order": 1}
            ),
            "symmetry order changed",
        ),
        (
            lambda doc: doc["symmetry"]["objects"]["fixed_receptacle"].update(
                {"keyed_yaw_observable": True}
            ),
            "cannot claim a unique keyed yaw",
        ),
        (
            lambda doc: doc["pose_provider_adapter"].update(
                {"pair_publication_enabled": True}
            ),
            "publication/control gates",
        ),
        (
            lambda doc: doc["pose_provider_adapter"].update(
                {"control_purpose_allowed": True}
            ),
            "publication/control gates",
        ),
        (
            lambda doc: doc["boundaries"].update(
                {"vision_control_authorized": True}
            ),
            "must be false",
        ),
    ),
)
def test_contract_fails_closed_when_symmetry_or_claims_are_weakened(
    tmp_path, mutate, message
):
    with pytest.raises(ValueError, match=message):
        load_symmetry_pose_readiness_contract(_mutated(tmp_path, mutate))
