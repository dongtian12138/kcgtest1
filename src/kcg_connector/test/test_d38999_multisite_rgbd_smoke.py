"""Pure contracts for the opt-in five-anchor Isaac RGB-D sweep."""

import ast
from pathlib import Path
import runpy
from types import SimpleNamespace
import sys

import pytest

from kcg_connector.d38999_multisite_vision6d import (
    load_d38999_multisite_vision6d_contract,
)
from kcg_connector.d38999_tabletop_scene import (
    load_d38999_tabletop_scene,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "isaac"
    / "d38999_multisite_rgbd_smoke.py"
)
E2E_PATH = (
    Path(__file__).parents[1]
    / "isaac"
    / "d38999_tabletop_pick_smoke.py"
)


def _source():
    return SCRIPT_PATH.read_text(encoding="utf-8")


def _module():
    return runpy.run_path(str(SCRIPT_PATH), run_name="multisite_rgbd_test")


def _passing_trial():
    endpoint = {
        "passed": True,
        "ray_plane_xy_error_m": 0.003,
        "semantic_ids": [2],
    }
    return {
        "capture_passed": True,
        "endpoints": {
            "loose_plug": dict(endpoint),
            "fixed_receptacle": dict(endpoint, semantic_ids=[3]),
        },
        "resource_cleanup": {"resources_released": True},
        "timeline_state": {"restored": True},
        "object_pose_writes_after_start": 0,
        "pose_scope": {
            "yaw_observed": False,
            "full_6d": False,
            "control_authorized": False,
        },
    }


def test_script_import_is_lazy_and_does_not_load_isaac_or_numpy():
    before = set(sys.modules)
    _module()
    imported = set(sys.modules) - before
    assert not any(
        name == "isaacsim" or name.startswith(("isaacsim.", "omni.", "pxr"))
        for name in imported
    )
    tree = ast.parse(_source())
    roots = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    assert roots.isdisjoint(
        {"isaacsim", "omni", "pxr", "numpy", "PIL", "torch", "rclpy"}
    )


def test_script_uses_one_app_but_fresh_stage_and_world_per_anchor():
    source = _source()
    assert source.count("SimulationApp(") == 1
    trial_loop = source.split(
        "for anchor_index, anchor in enumerate(", 1
    )[1].split("passed = bool(", 1)[0]
    assert "World.clear_instance()" in trial_loop
    assert "create_new_stage()" in trial_loop
    assert "world = World(" in trial_loop
    assert "world.stop()" in trial_loop
    assert "capture_d38999_rgbd_runtime(" in trial_loop
    assert "contract.required_anchor_pairs" in source
    assert '"required_trial_count": 5' in source


def test_endpoint_pose_authoring_precedes_physics_and_never_repeats():
    source = _source()
    main_source = source.split("def main():", 1)[1]
    author_index = main_source.index(
        "_author_endpoint_yaws_before_physics("
    )
    reset_index = main_source.index("world.reset()")
    capture_index = main_source.index("capture_d38999_rgbd_runtime(")
    assert author_index < reset_index < capture_index
    after_reset = main_source[reset_index:]
    assert "_author_endpoint_yaws_before_physics(" not in after_reset
    assert ".set_world_pose(" not in source
    assert ".set_local_pose(" not in source
    assert '"object_pose_writes_after_start": 0' in source


def test_scene_copy_moves_fixed_fixture_with_receptacle_and_keeps_z():
    module = _module()
    contract = load_d38999_multisite_vision6d_contract(
        repository=PROJECT_ROOT
    )
    tabletop = load_d38999_tabletop_scene(
        contract.input_paths["tabletop_scene"]
    )
    anchor = contract.required_anchor_pairs[1]
    moved = module["_scene_for_anchor"](tabletop, anchor)

    assert moved.loose_endpoint.initial_origin_m[:2] == anchor["loose_xy_m"]
    assert (
        moved.fixed_endpoint.receptacle_origin_m[:2]
        == anchor["fixed_xy_m"]
    )
    delta = (
        anchor["fixed_xy_m"][0]
        - tabletop.fixed_endpoint.receptacle_origin_m[0],
        anchor["fixed_xy_m"][1]
        - tabletop.fixed_endpoint.receptacle_origin_m[1],
    )
    assert moved.fixed_endpoint.fixture_center_m[:2] == pytest.approx(
        (
            tabletop.fixed_endpoint.fixture_center_m[0] + delta[0],
            tabletop.fixed_endpoint.fixture_center_m[1] + delta[1],
        )
    )
    assert (
        moved.loose_endpoint.initial_origin_m[2]
        == tabletop.loose_endpoint.initial_origin_m[2]
    )
    assert (
        moved.fixed_endpoint.receptacle_origin_m[2]
        == tabletop.fixed_endpoint.receptacle_origin_m[2]
    )
    assert tabletop.fixed_endpoint.receptacle_origin_m == (
        0.550,
        0.185,
        0.2615,
    )


def test_yaw_schedule_covers_each_anchor_and_stays_in_contract():
    module = _module()
    contract = load_d38999_multisite_vision6d_contract(
        repository=PROJECT_ROOT
    )
    loose = [
        module["_yaw_for_anchor"](index, "loose_plug")
        for index in range(5)
    ]
    fixed = [
        module["_yaw_for_anchor"](index, "fixed_receptacle")
        for index in range(5)
    ]
    assert len(set(loose)) == 5
    assert len(set(fixed)) == 5
    assert all(contract.loose_plug.yaw_rad.contains(value) for value in loose)
    assert all(
        contract.fixed_receptacle.yaw_rad.contains(value) for value in fixed
    )
    with pytest.raises(ValueError, match="unknown endpoint"):
        module["_yaw_for_anchor"](0, "nut")


def test_report_records_real_semantics_pixels_xy_cleanup_and_timeline():
    source = _source()
    for token in (
        '"semantic_ids"',
        '"mask_pixel_count"',
        '"ray_plane_xy_error_m"',
        '"visible_fraction"',
        '"resource_cleanup"',
        '"timeline_state"',
        '"timeline_pause"',
        '"observed_semantic_ids"',
    ):
        assert token in source
    assert "STRICT_MAXIMUM_XY_ERROR_M = 0.010" in source
    assert "value not in (0, 1)" in source
    assert 'output_dir / "report.json"' in source
    assert "allow_nan=False" in source
    assert "return 0 if passed else 1" in source


def test_strict_trial_gate_rejects_error_sentinel_or_scope_upgrade():
    trial_passed = _module()["_trial_passed"]
    trial = _passing_trial()
    assert trial_passed(trial) is True

    xy_bad = _passing_trial()
    xy_bad["endpoints"]["loose_plug"]["ray_plane_xy_error_m"] = 0.0101
    assert trial_passed(xy_bad) is False

    sentinel = _passing_trial()
    sentinel["endpoints"]["fixed_receptacle"]["semantic_ids"] = [1]
    assert trial_passed(sentinel) is False

    upgraded = _passing_trial()
    upgraded["pose_scope"]["full_6d"] = True
    assert trial_passed(upgraded) is False


def test_non_authorization_record_is_fail_closed():
    record = _module()["_non_authorization_record"]
    contract = load_d38999_multisite_vision6d_contract(
        repository=PROJECT_ROOT
    )
    result = record(contract)
    assert result["yaw_observed"] is False
    assert result["full_6d"] is False
    assert result["control_authorized"] is False
    assert "yaw_symmetry_has_multiple_equivalent_hypotheses" in (
        result["rejection_reasons"]
    )

    dishonest = SimpleNamespace(
        current_proxy_has_unique_polarization_key=True,
        pose_control_current_authorized=False,
    )
    with pytest.raises(RuntimeError, match="claims a key"):
        record(dishonest)


def test_output_directory_rejects_absolute_and_parent_escape():
    resolve = _module()["_repository_output_path"]
    with pytest.raises(ValueError, match="repository-relative"):
        resolve(PROJECT_ROOT, "/tmp/out")
    with pytest.raises(ValueError, match="repository-relative"):
        resolve(PROJECT_ROOT, "../out")
    assert resolve(PROJECT_ROOT, "artifacts/safe") == (
        PROJECT_ROOT / "artifacts/safe"
    )


def test_new_smoke_is_not_connected_to_the_existing_e2e():
    e2e_source = E2E_PATH.read_text(encoding="utf-8")
    assert "d38999_multisite_rgbd_smoke" not in e2e_source
    assert "d38999_multisite_vision6d_v1" not in e2e_source
