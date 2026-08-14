"""Pure tests for the visual-XY pick-to-preinsert continuation contract."""

import ast
from dataclasses import replace
import inspect
import json
from pathlib import Path

import pytest
import yaml

from kcg_connector.d38999_visual_xy_pick_probe import (
    APPROVED_PROBE_VARIANTS,
    build_visual_xy_pick_plan,
    load_visual_xy_pick_probe_contract,
    pose_provider_sample_from_rgbd_metrics,
)
from kcg_connector.d38999_visual_xy_preinsert_probe import (
    DEFAULT_CONFIG_PATH,
    SCHEMA_VERSION,
    TARGET_ORDER,
    build_visual_xy_preinsert_plan,
    load_visual_xy_preinsert_probe_contract,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "kcg_connector/d38999_visual_xy_preinsert_probe.py"
)


def _endpoint(x, y):
    return {
        "ray_plane_registered_model_height_world_xyz_m": [x, y, 0.215],
        "mask_depth": {
            "pixel_count": 1000,
            "valid_depth_count": 1000,
            "visible_fraction": 0.005,
            "minimum_depth_m": 0.5,
            "maximum_depth_m": 1.2,
        },
        "semantic_mask_center": {"in_frame": True},
    }


def _pick_plan(pick_contract, *, fixed=(0.5499213231, 0.1868220323)):
    capture = {
        "endpoint_semantic_ids": {
            "loose_plug": [3],
            "fixed_receptacle": [2],
        },
        "loose_plug": _endpoint(*pick_contract.authored_loose_xy_m),
        "fixed_receptacle": _endpoint(*fixed),
        # Deliberate bait: neither provider conversion nor this continuation
        # planner may consume truth-like fields from the capture document.
        "registered_truth_xy_m": [99.0, 99.0],
    }
    sample = pose_provider_sample_from_rgbd_metrics(
        pick_contract,
        capture,
        timestamp_s=1.0,
        capture_id=f"{pick_contract.trial_id}-capture",
    )
    return build_visual_xy_pick_plan(
        pick_contract,
        sample,
        now_s=1.0,
        explicit_probe_opt_in=True,
    )


def _contracts(pick_path=None):
    continuation = load_visual_xy_preinsert_probe_contract(
        repository=PROJECT_ROOT
    )
    pick = load_visual_xy_pick_probe_contract(
        pick_path or None, repository=PROJECT_ROOT
    ) if pick_path is not None else load_visual_xy_pick_probe_contract(
        repository=PROJECT_ROOT
    )
    return continuation, pick


def test_contract_is_pure_disabled_and_does_not_modify_e2e():
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

    contract, _ = _contracts()
    assert contract.schema_version == SCHEMA_VERSION
    assert contract.enabled_by_default is False
    assert contract.status == "prepared_cpu_plan_not_physx_executed"
    assert contract.preinsert_gap_m == pytest.approx(0.012)
    assert contract.entry_gap_m == pytest.approx(0.010)
    assert contract.registered_margin_before_entry_m == pytest.approx(0.002)
    assert contract.cpu_plan_filename == "preinsert_cpu_plan.json"
    assert contract.boundaries["existing_e2e_modified"] is False
    assert contract.boundaries["gpu_or_physx_validated"] is False


def test_runtime_builder_has_no_truth_pose_input_seam():
    signature = inspect.signature(build_visual_xy_preinsert_plan)
    assert tuple(signature.parameters) == (
        "contract",
        "pick_contract",
        "pick_plan",
        "explicit_probe_opt_in",
    )
    source = inspect.getsource(build_visual_xy_preinsert_plan)
    assert "get_world_pose" not in source
    assert "registered_truth" not in source
    assert "truth_xy" not in source


def test_visual_fixed_xy_builds_transport_axis_high_preinsert_plan():
    contract, pick = _contracts()
    pick_plan = _pick_plan(pick)
    plan = build_visual_xy_preinsert_plan(
        contract, pick, pick_plan, explicit_probe_opt_in=True
    )

    assert tuple(plan.arm_targets_rad) == TARGET_ORDER
    assert tuple(plan.tcp_targets_world_m) == TARGET_ORDER
    assert plan.fixed_translation_xy_m == pytest.approx(
        (-0.0000786769, 0.0018220323)
    )
    assert plan.tcp_targets_world_m["preinsert"] == pytest.approx(
        (0.5499213231, 0.1868220323, 0.32198)
    )
    assert plan.maximum_abs_joint_delta_from_nominal_rad < 0.013
    assert plan.planned_peak_joint_speed_rad_s < 0.30
    assert max(plan.fk_position_errors_m.values()) < 1.0e-9
    assert max(plan.fk_orientation_errors_rad.values()) == 0.0
    for name, arm in plan.arm_targets_rad.items():
        nominal = getattr(contract.insertion.motion, f"{name}_arm_rad")
        assert arm[6] == pytest.approx(nominal[6], abs=1.0e-12)

    report = plan.to_mapping()
    assert report["stop_stage"] == "PREINSERT"
    assert report["truth_xy_used_for_target"] is False
    assert report["truth_pose_feedback_used_for_target"] is False
    assert report["engage_executed"] is False
    assert report["gpu_or_physx_validated"] is False
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("filename", tuple(APPROVED_PROBE_VARIANTS))
def test_all_approved_visual_pick_variants_share_the_continuation_seam(filename):
    pick_path = PROJECT_ROOT / "src/kcg_connector/config" / filename
    contract, pick = _contracts(pick_path)
    pick_plan = _pick_plan(pick)
    plan = build_visual_xy_preinsert_plan(
        contract, pick, pick_plan, explicit_probe_opt_in=True
    )
    assert plan.trial_id == pick.trial_id
    assert plan.planned_peak_joint_speed_rad_s < 1.0
    assert max(plan.fk_position_errors_m.values()) < 1.0e-9


def test_missing_opt_in_and_truth_orientation_fail_before_plan_use():
    contract, pick = _contracts()
    plan = _pick_plan(pick)
    with pytest.raises(ValueError, match="opt-in"):
        build_visual_xy_preinsert_plan(
            contract, pick, plan, explicit_probe_opt_in=False
        )

    contaminated_result = replace(
        plan.adapter_result, uses_truth_orientation=True
    )
    contaminated_plan = replace(plan, adapter_result=contaminated_result)
    with pytest.raises(ValueError, match="not eligible"):
        build_visual_xy_preinsert_plan(
            contract, pick, contaminated_plan, explicit_probe_opt_in=True
        )


def test_input_hash_drift_and_scope_upgrade_fail_closed(tmp_path):
    document = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    document["inputs"]["nominal_insertion"]["sha256"] = "0" * 64
    drifted = tmp_path / "drifted.yaml"
    drifted.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_visual_xy_preinsert_probe_contract(
            drifted, repository=PROJECT_ROOT
        )

    document = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    document["axial_scope"]["engage_target_planned"] = True
    upgraded = tmp_path / "upgraded.yaml"
    upgraded.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="axial scope changed"):
        load_visual_xy_preinsert_probe_contract(
            upgraded, repository=PROJECT_ROOT
        )
