"""Pure contract tests for the D38999 20-degree q7 probe."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys

import pytest
import yaml

from kcg_connector.d38999_twist_probe import (
    DEFAULT_CONFIG_PATH,
    load_d38999_twist_probe_contract,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
SMOKE_PATH = PACKAGE_ROOT / "isaac/d38999_nut_regrasp_smoke.py"
STAGE120_CONFIG_PATH = (
    PACKAGE_ROOT / "config/d38999_q7_twist_probe_stage120_v1.yaml"
)


def test_contract_loads_and_proves_twenty_degree_arithmetic():
    document, resolved = load_d38999_twist_probe_contract(
        repository=PROJECT_ROOT
    )
    assert set(resolved) == {
        "assembly_baseline",
        "nut_regrasp_physx",
        "passed_regrasp_run_1",
        "passed_regrasp_run_2",
        "runner_source",
        "twist_contract_source",
    }
    probe = document["probe"]
    assert document["probe_id"] == "stage20"
    assert math.degrees(probe["q7_delta_rad"]) == pytest.approx(-20.0)
    assert math.degrees(probe["expected_nut_delta_rad"]) == pytest.approx(20.0)
    assert probe["expected_axial_travel_m"] == pytest.approx(-1.0 / 6000.0)
    assert probe["motion_duration_s"] == 4.0
    assert probe["hold_settle_duration_s"] == 0.25
    assert probe["hold_evaluation_duration_s"] == 0.5
    assert probe["total_hold_duration_s"] == 0.75


def test_runtime_thread_is_one_way_and_filters_exact_proxy_pair_set():
    document, _ = load_d38999_twist_probe_contract(
        repository=PROJECT_ROOT
    )
    thread = document["runtime_thread"]
    assert thread["lower_limit_m"] == -0.0031
    assert thread["upper_limit_m"] == 0.0001
    assert thread["rack_ratio_degrees_per_meter"] == 120000.0
    assert thread["expected_nut_segment_count"] == 24
    assert thread["expected_body_mating_segment_count"] == 20
    assert thread["expected_fixed_entry_segment_count"] == 20
    assert thread["expected_filtered_pair_count"] == 500


def test_stage120_contract_is_explicit_and_has_q7_headroom():
    document, _ = load_d38999_twist_probe_contract(
        STAGE120_CONFIG_PATH, repository=PROJECT_ROOT
    )
    probe = document["probe"]
    assert document["probe_id"] == "stage120"
    assert math.degrees(probe["q7_delta_rad"]) == pytest.approx(-120.0)
    assert math.degrees(probe["expected_nut_delta_rad"]) == pytest.approx(
        120.0
    )
    assert probe["expected_axial_travel_m"] == -0.001
    assert probe["motion_duration_s"] == 24.0
    assert probe["hold_evaluation_duration_s"] == 2.0
    q7_start = 0.650482794
    q7_target = q7_start + probe["q7_delta_rad"]
    assert -2.5 < q7_target < 2.5
    assert math.degrees(q7_target - (-2.5)) > 10.0


def test_user_torque_limits_and_non_claim_boundaries_are_locked():
    document, _ = load_d38999_twist_probe_contract(
        repository=PROJECT_ROOT
    )
    assert document["sensing"]["operational_torque_target_nm"] == 1.8
    assert document["sensing"]["hard_stop_nm"] == 2.0
    boundaries = document["boundaries"]
    assert boundaries["thread_teeth_collision_modeled"] is False
    assert boundaries["real_thread_pitch_claimed"] is False
    assert boundaries["assembly_success_claimed"] is False
    acceptance = document["acceptance"]
    assert acceptance["hold_axial_observable_window_steps"] == 5
    assert acceptance["maximum_hold_axial_speed_m_s"] == 0.00005


def test_loader_fails_closed_on_changed_ratio(tmp_path):
    document = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    document["runtime_thread"]["rack_ratio_degrees_per_meter"] = -120000.0
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="rack/filter geometry changed"):
        load_d38999_twist_probe_contract(path, repository=PROJECT_ROOT)


def test_smoke_remains_lazy_and_exposes_explicit_twist_flag():
    source = SMOKE_PATH.read_text(encoding="utf-8")
    before_main = source.split("def main():", 1)[0]
    assert "isaacsim" not in before_main
    assert "from pxr" not in before_main
    assert '"--twist-probe"' in source
    assert "ISAAC D38999 Q7 TWIST PROBE V1" in source
    assert "load_d38999_twist_probe_contract" in source
    assert "Gf.Quatd(" in source
    assert "set_world_pose(" not in source
    assert source.index("if not regrasp_passed") < source.index(
        "UsdPhysics.FilteredPairsAPI.Apply"
    )


def test_pure_module_import_does_not_load_isaac():
    before = set(sys.modules)
    spec = importlib.util.spec_from_file_location(
        "d38999_twist_probe_isolated",
        PACKAGE_ROOT / "kcg_connector/d38999_twist_probe.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loaded = set(sys.modules) - before
    assert not any(
        name.startswith(("isaacsim", "omni", "pxr")) for name in loaded
    )
