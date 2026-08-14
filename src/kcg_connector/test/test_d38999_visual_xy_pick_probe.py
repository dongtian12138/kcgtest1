"""Pure tests for the independent non-nominal visual-XY pick probe."""

import ast
from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path

import pytest
import yaml

from kcg_connector.d38999_visual_xy_pick_probe import (
    APPROVED_PROBE_VARIANTS,
    DEFAULT_CONFIG_PATH,
    SCHEMA_VERSION,
    build_visual_xy_pick_plan,
    evaluate_visual_xy_truth_only,
    load_visual_xy_pick_probe_contract,
    pose_provider_sample_from_rgbd_metrics,
)
import kcg_connector.d38999_visual_xy_pick_probe as probe_module


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "kcg_connector/d38999_visual_xy_pick_probe.py"
)
E2E_PATH = (
    Path(__file__).resolve().parents[1]
    / "isaac/d38999_tabletop_pick_smoke.py"
)
CPU_PLAN_PATH = (
    PROJECT_ROOT
    / "artifacts/kcg_connector/"
    "d38999_visual_xy_pick_probe_v1/cpu_plan.json"
)
VARIANTS = (
    (
        "d38999_visual_xy_pick_px20_y0_v1.yaml",
        "loose_plus_20mm_x_fixed_nominal",
        (0.540, -0.210),
        "loose_px20_y0.json",
        0.10314326135999397,
    ),
    (
        "d38999_visual_xy_pick_mx20_y0_v1.yaml",
        "loose_minus_20mm_x_fixed_nominal",
        (0.500, -0.210),
        "loose_mx20_y0.json",
        0.08867323870159804,
    ),
    (
        "d38999_visual_xy_pick_x0_my20_v1.yaml",
        "loose_minus_20mm_y_fixed_nominal",
        (0.520, -0.230),
        "loose_x0_my20.json",
        0.13235988838016738,
    ),
)


def _contract():
    return load_visual_xy_pick_probe_contract(repository=PROJECT_ROOT)


def _endpoint(x, y, *, valid_count=1000, in_frame=True):
    return {
        "ray_plane_registered_model_height_world_xyz_m": [x, y, 0.215],
        "mask_depth": {
            "pixel_count": 1000,
            "valid_depth_count": valid_count,
            "visible_fraction": 0.005,
            "minimum_depth_m": 0.5,
            "maximum_depth_m": 1.2,
        },
        "semantic_mask_center": {"in_frame": in_frame},
    }


def _capture(loose=(0.530, -0.200), fixed=(0.550, 0.185)):
    return {
        "endpoint_semantic_ids": {
            "loose_plug": [3],
            "fixed_receptacle": [2],
        },
        "loose_plug": _endpoint(*loose),
        "fixed_receptacle": _endpoint(*fixed),
        # Runtime metrics may contain truth evaluation fields.  The provider
        # adapter must ignore them rather than copy them into its diagnostics.
        "unrelated_truth_evaluation": {"do_not_consume": [99.0, 99.0]},
    }


def _sample_for_contract(contract, loose):
    return pose_provider_sample_from_rgbd_metrics(
        contract,
        _capture(loose=loose, fixed=contract.authored_fixed_xy_m),
        timestamp_s=1.0,
        capture_id=contract.trial_id,
    )


def _sample(capture=None):
    return pose_provider_sample_from_rgbd_metrics(
        _contract(),
        _capture() if capture is None else capture,
        timestamp_s=1.0,
        capture_id="capture-001",
    )


def _plan(sample=None):
    return build_visual_xy_pick_plan(
        _contract(),
        _sample() if sample is None else sample,
        now_s=1.0,
        explicit_probe_opt_in=True,
    )


def test_contract_is_pure_disabled_non_nominal_and_not_in_e2e():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    assert roots.isdisjoint(
        {"isaacsim", "omni", "pxr", "rclpy", "torch", "cv2", "open3d"}
    )

    contract = _contract()
    assert contract.schema_version == SCHEMA_VERSION
    assert contract.enabled_by_default is False
    assert contract.authored_loose_xy_m == pytest.approx((0.530, -0.200))
    assert contract.authored_fixed_xy_m == pytest.approx((0.550, 0.185))
    assert contract.orientation_source == "registered_nominal"
    assert contract.boundaries["truth_xy_used_for_target"] is False
    assert contract.boundaries["truth_xy_evaluation_only"] is True
    assert all(path.is_file() for path in contract.input_paths.values())
    assert "d38999_visual_xy_pick" not in E2E_PATH.read_text(
        encoding="utf-8"
    )


def test_capture_adapter_source_cannot_read_truth_xy_or_error_fields():
    source = inspect.getsource(pose_provider_sample_from_rgbd_metrics)
    forbidden_tokens = (
        "registered_truth" + "_xy_m",
        "xy_" + "error_m",
        "camera_observation_" + "present",
    )
    assert all(token not in source for token in forbidden_tokens)

    sample = _sample()
    assert sample.pair is None
    assert sample.uses_truth_position is False
    assert sample.uses_truth_orientation is False
    assert sample.full_6d is False
    assert sample.control_authorized is False
    assert sample.diagnostics["endpoints"]["loose_plug"][
        "estimated_world_xy_m"
    ] == pytest.approx((0.530, -0.200))


def test_capture_confidence_is_a_strict_visibility_depth_gate_score():
    capture = _capture()
    capture["loose_plug"] = _endpoint(
        0.530, -0.200, valid_count=700
    )
    sample = _sample(capture)
    confidence = sample.diagnostics["endpoints"]["loose_plug"][
        "confidence"
    ]
    assert confidence == 0.0
    with pytest.raises(ValueError, match="confidence_below_gate"):
        _plan(sample)

    capture = _capture()
    capture["endpoint_semantic_ids"]["fixed_receptacle"] = [1]
    assert _sample(capture).diagnostics["endpoints"]["fixed_receptacle"][
        "confidence"
    ] == 0.0


def test_plus_10mm_visual_xy_builds_bounded_fixed_q7_plan():
    contract = _contract()
    plan = _plan()
    assert plan.adapter_result.eligible_for_independent_probe is True
    assert plan.tcp_targets_world_m["pregrasp_tcp"] == pytest.approx(
        (0.530, -0.200, 0.360)
    )
    assert plan.tcp_targets_world_m["grasp_tcp"] == pytest.approx(
        (0.530, -0.200, 0.24848)
    )
    assert plan.maximum_abs_joint_delta_from_nominal_rad == pytest.approx(
        0.09548185977999885
    )
    assert plan.maximum_abs_joint_delta_from_nominal_rad < (
        contract.local_ik.maximum_abs_joint_delta_from_nominal_rad
    )
    assert plan.planned_peak_joint_speed_rad_s == pytest.approx(
        0.30616140758952876
    )
    assert max(plan.fk_position_errors_m.values()) < 2.0e-10
    assert max(plan.fk_orientation_errors_rad.values()) == 0.0
    nominal = {
        "pregrasp": contract.pick.motion.approach_segments[-1].target_arm_rad,
        "closure_clearance": contract.pick.motion.closure_clearance_arm_rad,
        "grasp": contract.pick.motion.grasp_arm_rad,
    }
    for name, arm in plan.arm_targets_rad.items():
        assert arm[6] == nominal[name][6]
    report = plan.to_mapping()
    assert report["uses_truth_xy_for_target"] is False
    assert report["full_6d"] is False
    assert report["production_control_authorized"] is False
    assert report["gpu_or_physx_validated"] is False
    json.dumps(report, allow_nan=False)


def test_checked_in_cpu_plan_matches_current_contract_and_solver():
    artifact = json.loads(CPU_PLAN_PATH.read_text(encoding="utf-8"))
    generated = _plan().to_mapping()
    assert artifact["probe_config_sha256"] == hashlib.sha256(
        DEFAULT_CONFIG_PATH.read_bytes()
    ).hexdigest()
    assert artifact["input_kind"] == (
        "reference_visual_estimate_for_cpu_reachability_only"
    )
    assert artifact["tcp_targets_world_m"] == pytest.approx(
        generated["tcp_targets_world_m"]
    )
    for name, values in artifact["arm_targets_rad"].items():
        assert values == pytest.approx(generated["arm_targets_rad"][name])
    assert artifact["truth_xy_used_for_target"] is False
    assert artifact["gpu_or_physx_validated"] is False


@pytest.mark.parametrize(
    ("filename", "trial_id", "loose_xy", "artifact_name", "maximum_delta"),
    VARIANTS,
)
def test_approved_variant_is_hash_bound_and_matches_cpu_plan(
    filename, trial_id, loose_xy, artifact_name, maximum_delta
):
    config_path = PROJECT_ROOT / "src/kcg_connector/config" / filename
    approved = APPROVED_PROBE_VARIANTS[filename]
    assert approved["sha256"] == hashlib.sha256(
        config_path.read_bytes()
    ).hexdigest()
    assert approved["trial_id"] == trial_id
    assert approved["loose_xy_m"] == loose_xy

    contract = load_visual_xy_pick_probe_contract(
        config_path, repository=PROJECT_ROOT
    )
    sample = _sample_for_contract(contract, loose_xy)
    plan = build_visual_xy_pick_plan(
        contract,
        sample,
        now_s=1.0,
        explicit_probe_opt_in=True,
    )
    artifact_path = CPU_PLAN_PATH.parent / "cpu_plans" / artifact_name
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["probe_config_sha256"] == approved["sha256"]
    assert artifact["trial_id"] == trial_id
    assert artifact["reference_visual_estimate_world_xy_m"][
        "loose_plug"
    ] == pytest.approx(loose_xy)
    assert plan.maximum_abs_joint_delta_from_nominal_rad == pytest.approx(
        maximum_delta
    )
    assert plan.maximum_abs_joint_delta_from_nominal_rad < 0.150
    assert plan.planned_peak_joint_speed_rad_s < (
        contract.pick.acceptance.maximum_observed_joint_speed_rad_s
    )
    assert max(plan.fk_position_errors_m.values()) < 3.0e-10
    assert max(plan.fk_orientation_errors_rad.values()) == 0.0
    assert artifact["tcp_targets_world_m"] == pytest.approx(
        plan.to_mapping()["tcp_targets_world_m"]
    )
    for name, values in artifact["arm_targets_rad"].items():
        assert values == pytest.approx(plan.arm_targets_rad[name])
    assert artifact["gpu_or_physx_validated"] is False


def test_unapproved_config_path_and_variant_hash_drift_fail_closed(tmp_path):
    source = DEFAULT_CONFIG_PATH.read_bytes()
    unapproved = tmp_path / DEFAULT_CONFIG_PATH.name
    unapproved.write_bytes(source)
    with pytest.raises(ValueError, match="not an approved repository variant"):
        load_visual_xy_pick_probe_contract(
            unapproved, repository=PROJECT_ROOT
        )

    approved = APPROVED_PROBE_VARIANTS[DEFAULT_CONFIG_PATH.name]
    assert approved["sha256"] == hashlib.sha256(source).hexdigest()


def test_out_of_domain_visual_estimate_fails_before_ik():
    sample = _sample(_capture(loose=(0.570, -0.200)))
    with pytest.raises(ValueError, match="outside_bounded"):
        _plan(sample)


def test_truth_evaluation_is_separate_and_cannot_change_plan():
    plan = _plan()
    before = deepcopy(plan.to_mapping())
    evaluation = evaluate_visual_xy_truth_only(
        plan,
        loose_truth_xy_m=(0.531, -0.202),
        fixed_truth_xy_m=(0.550, 0.185),
    )
    assert evaluation["scope"] == (
        "post_hoc_truth_evaluation_not_target_input"
    )
    assert evaluation["loose_xy_error_m"] == pytest.approx(
        (5.0**0.5) * 0.001
    )
    assert evaluation["fixed_xy_error_m"] == 0.0
    assert evaluation["truth_xy_used_for_target"] is False
    assert plan.to_mapping() == before


def _mutated_config(tmp_path, mutate):
    document = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    mutate(document)
    path = tmp_path / "visual_xy_pick.yaml"
    path.write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )
    return path


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda doc: doc.update(enabled_by_default=True),
            "disabled by default",
        ),
        (
            lambda doc: doc["trial"].update(loose_plug_xy_m=[0.540, -0.200]),
            "approved enumerated variant",
        ),
        (
            lambda doc: doc["rgbd_observation"].update(
                per_capture_xy_error_bound_m=0.011
            ),
            "observation scope",
        ),
        (
            lambda doc: doc["local_fixed_q7_ik"].update(
                maximum_abs_joint_delta_from_nominal_rad=0.5
            ),
            "local IK policy",
        ),
        (
            lambda doc: doc["boundaries"].update(
                truth_xy_used_for_target=True
            ),
            "boundaries changed",
        ),
        (
            lambda doc: doc["inputs"]["visual_xy_adapter"].update(
                sha256="0" * 64
            ),
            "SHA-256 mismatch",
        ),
    ),
)
def test_probe_rejects_scope_gate_or_hash_drift(
    tmp_path, monkeypatch, mutate, message
):
    path = _mutated_config(tmp_path, mutate)
    # Production calls still pass through the repository path and SHA lock.
    # Replacing only that lookup lets this pure test retain coverage of every
    # deeper semantic gate on a deliberately mutated document.
    monkeypatch.setattr(
        probe_module,
        "_approved_probe_variant",
        lambda config_path, root: APPROVED_PROBE_VARIANTS[
            DEFAULT_CONFIG_PATH.name
        ],
    )
    with pytest.raises(ValueError, match=message):
        load_visual_xy_pick_probe_contract(path, repository=PROJECT_ROOT)
