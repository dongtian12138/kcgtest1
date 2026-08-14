"""Pure tests for the disabled D38999 multi-position vision contract."""

import ast
from copy import deepcopy
import json
import math
from pathlib import Path

import pytest
import yaml

from kcg_connector.d38999_multisite_vision6d import (
    DEFAULT_CONFIG_PATH,
    SCHEMA_VERSION,
    evaluate_pose_control_gate,
    load_d38999_multisite_vision6d_contract,
    resolve_target_candidate,
    sample_multisite_placement,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    Path(__file__).parents[1]
    / "kcg_connector"
    / "d38999_multisite_vision6d.py"
)
E2E_RUNNER_PATH = (
    Path(__file__).parents[1]
    / "isaac"
    / "d38999_tabletop_pick_smoke.py"
)


def _contract():
    return load_d38999_multisite_vision6d_contract(
        repository=PROJECT_ROOT
    )


def _excellent_evidence():
    endpoint = {
        "mask_pixels": 1500,
        "visible_fraction": 0.08,
        "valid_depth_fraction_in_mask": 0.99,
        "center_margin_px": 80.0,
        "occlusion_fraction": 0.05,
        "calibrated_view_count": 2,
        "translation_std_m": 0.001,
        "cad_fit_rmse_m": 0.0015,
        "axis_error_rad": math.radians(0.8),
        "axis_inlier_fraction": 0.91,
        "key_feature_observed": True,
        "yaw_hypothesis_count": 1,
        "unique_yaw_score_margin": 0.40,
    }
    return {
        "loose_plug": deepcopy(endpoint),
        "fixed_receptacle": deepcopy(endpoint),
    }


def _write_mutated(tmp_path, mutate):
    document = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    mutate(document)
    path = tmp_path / "mutated.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def test_module_is_pure_and_contract_is_disabled_and_content_addressed():
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
            "cv2",
            "open3d",
        }
    )

    contract = _contract()
    assert contract.schema_version == SCHEMA_VERSION
    assert contract.enabled is False
    assert contract.delivery_window_hours == 8.0
    assert math.gcd(*contract.periodic_component_orders["loose_body"]) == 2
    assert (
        math.gcd(*contract.periodic_component_orders["fixed_receptacle"])
        == 2
    )
    assert set(contract.input_paths) == {
        "pose_contract",
        "proxy_asset",
        "proxy_config",
        "tabletop_scene",
    }
    assert all(path.is_file() for path in contract.input_paths.values())
    json.dumps(contract.boundaries, allow_nan=False)


def test_preparation_contract_is_not_wired_into_the_e2e_runner():
    source = E2E_RUNNER_PATH.read_text(encoding="utf-8")
    assert "d38999_multisite_vision6d" not in source
    assert "d38999_multisite_vision6d_v1.yaml" not in source


def test_multisite_bounds_include_five_safe_anchors_and_seeded_samples():
    contract = _contract()
    assert len(contract.required_anchor_pairs) == 5
    assert {
        item["id"] for item in contract.required_anchor_pairs
    } == {
        "nominal",
        "loose_left_fixed_right",
        "loose_right_fixed_left",
        "cross_corner_a",
        "cross_corner_b",
    }

    first = sample_multisite_placement(contract, seed=42)
    assert sample_multisite_placement(contract, seed=42) == first
    assert sample_multisite_placement(contract, seed=43) != first
    for seed in range(100):
        sample = sample_multisite_placement(contract, seed=seed)
        assert contract.loose_plug.x_m.contains(
            sample.loose_position_xyz_m[0]
        )
        assert contract.loose_plug.y_m.contains(
            sample.loose_position_xyz_m[1]
        )
        assert contract.fixed_receptacle.x_m.contains(
            sample.fixed_position_xyz_m[0]
        )
        assert contract.fixed_receptacle.y_m.contains(
            sample.fixed_position_xyz_m[1]
        )
        assert (
            sample.center_separation_m
            >= contract.minimum_center_separation_m
        )


def test_strong_geometry_evidence_still_cannot_invent_keyed_yaw():
    contract = _contract()
    result = evaluate_pose_control_gate(contract, _excellent_evidence())

    assert result.loose_plug.visibility_passed is True
    assert result.loose_plug.translation_observed is True
    assert result.loose_plug.axis_observed is True
    assert result.loose_plug.keyed_yaw_observed is False
    assert result.fixed_receptacle.axis_observed is True
    assert result.full_6d is False
    assert result.keyed_orientation_observed is False
    assert result.control_authorized is False
    assert "proxy_unique_key_geometry_absent" in result.reasons
    assert "target_transforms_unqualified" in result.reasons
    assert "contract_disabled" in result.reasons


def test_visibility_translation_and_axis_failures_are_separate():
    contract = _contract()
    evidence = _excellent_evidence()
    evidence["loose_plug"]["center_margin_px"] = 2.0
    evidence["fixed_receptacle"]["axis_inlier_fraction"] = 0.2
    result = evaluate_pose_control_gate(contract, evidence)

    assert result.loose_plug.visibility_passed is False
    assert result.loose_plug.translation_observed is False
    assert result.fixed_receptacle.translation_observed is True
    assert result.fixed_receptacle.axis_observed is False
    assert "loose_plug:visibility_gate_failed" in result.reasons
    assert "fixed_receptacle:axis_gate_failed" in result.reasons


def test_object_target_composition_preserves_unqualified_status():
    contract = _contract()
    target = resolve_target_candidate(
        contract,
        transform_id="d38999_loose_object_T_grasp_tcp_candidate_v1",
        object_position_xyz_m=(0.52, -0.21, 0.20),
        object_quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
    )
    assert target.position_xyz_m == pytest.approx(
        (0.52, -0.21, 0.24848)
    )
    assert target.quaternion_xyzw == pytest.approx(
        (
            0.19073322597108375,
            0.9816419084934265,
            -1.845583392825855e-9,
            2.1627256602512928e-7,
        )
    )
    assert target.qualified is False

    rotated = resolve_target_candidate(
        contract,
        transform_id="d38999_loose_object_T_assembly_candidate_v1",
        object_position_xyz_m=(0.5, 0.1, 0.2),
        object_quaternion_xyzw=(
            0.0,
            0.0,
            math.sin(math.pi / 4.0),
            math.cos(math.pi / 4.0),
        ),
    )
    assert rotated.position_xyz_m == pytest.approx((0.5, 0.1, 0.2))
    assert rotated.quaternion_xyzw[2:] == pytest.approx(
        (math.sqrt(0.5), math.sqrt(0.5))
    )
    assert rotated.qualified is False


def test_foundationpose_plan_uses_official_isolated_assets_and_blockers():
    contract = _contract()
    assert contract.foundationpose_model_version == "1.0.1_onnx"
    assert contract.foundationpose_model_license == "NVIDIA_Model_EULA"
    assert contract.foundationpose_ros_wrapper_license == "Apache-2.0"
    assert (
        contract.foundationpose_reference_code_license
        == "NVIDIA_Source_Code_License"
    )
    assert all(
        value.startswith("https://")
        for value in contract.foundationpose_official_sources.values()
    )
    mesh_path = contract.foundationpose_required_asset_paths["object_mesh"]
    assert mesh_path.endswith(".obj")
    assert {
        "isaac_ros_environment_not_installed",
        "tensorrt_and_trtexec_not_available_on_host_path",
        "simplified_obj_mesh_not_exported_and_validated",
        "current_proxy_has_no_unique_polarization_key",
    }.issubset(contract.foundationpose_blockers)


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda document: document.update({"enabled": True}),
            "must remain disabled",
        ),
        (
            lambda document: document["symmetry_and_keying"].update(
                {"current_proxy_has_unique_polarization_key": True}
            ),
            "symmetry declaration",
        ),
        (
            lambda document: document["object_target_transforms"][
                "candidates"
            ][0].update({"qualified": True}),
            "not calibrated",
        ),
        (
            lambda document: document["foundationpose"].update(
                {"blockers": ["current_proxy_has_no_unique_polarization_key"]}
            ),
            "blocker list is incomplete",
        ),
        (
            lambda document: document["inputs"]["proxy_config"].update(
                {"sha256": "0" * 64}
            ),
            "SHA-256 mismatch",
        ),
    ),
)
def test_contract_fails_closed_on_unsafe_claims_or_input_drift(
    tmp_path, mutate, message
):
    path = _write_mutated(tmp_path, mutate)
    with pytest.raises(ValueError, match=message):
        load_d38999_multisite_vision6d_contract(
            path, repository=PROJECT_ROOT
        )


def test_pose_evidence_requires_exact_finite_schema():
    contract = _contract()
    extra = _excellent_evidence()
    extra["loose_plug"]["truth_yaw"] = 0.0
    with pytest.raises(ValueError, match="keys differ"):
        evaluate_pose_control_gate(contract, extra)

    invalid = _excellent_evidence()
    invalid["loose_plug"]["cad_fit_rmse_m"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        evaluate_pose_control_gate(contract, invalid)

    with pytest.raises(ValueError, match="seed must be an integer"):
        sample_multisite_placement(contract, seed=True)
