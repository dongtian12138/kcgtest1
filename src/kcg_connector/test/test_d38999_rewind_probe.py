"""Pure tests for the D38999 release/rewind/regrasp contract."""

from __future__ import annotations

import hashlib
import importlib.util
import math
from pathlib import Path
import sys

import pytest
import yaml

from kcg_connector.d38999_rewind_probe import (
    DEFAULT_CONFIG_PATH,
    load_d38999_rewind_probe_contract,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
SMOKE_PATH = PACKAGE_ROOT / "isaac/d38999_nut_regrasp_smoke.py"
STAGE120_CONFIG_PATH = (
    PACKAGE_ROOT / "config/d38999_q7_twist_probe_stage120_v1.yaml"
)


def test_contract_is_stage120_cycle_and_preserves_user_torque_limits():
    document, resolved = load_d38999_rewind_probe_contract(
        repository=PROJECT_ROOT
    )
    assert set(resolved) == {
        "stage120_twist_contract",
        "nut_regrasp_physx",
        "runner_source",
        "rewind_contract_source",
    }
    control = document["control"]
    assert math.degrees(control["rewind_delta_rad"]) == pytest.approx(120.0)
    assert control["rewind_duration_s"] == 24.0
    assert document["sensing"]["operational_torque_target_nm"] == 1.8
    assert document["sensing"]["hard_stop_nm"] == 2.0


def test_runner_and_nested_stage120_provenance_chain_is_current():
    """Lock every hash edge updated when the shared runner changes."""

    rewind = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    stage120 = yaml.safe_load(
        STAGE120_CONFIG_PATH.read_text(encoding="utf-8")
    )
    runner_digest = hashlib.sha256(SMOKE_PATH.read_bytes()).hexdigest()
    stage120_digest = hashlib.sha256(
        STAGE120_CONFIG_PATH.read_bytes()
    ).hexdigest()
    assert stage120["inputs"]["runner_source"] == {
        "path": "src/kcg_connector/isaac/d38999_nut_regrasp_smoke.py",
        "sha256": runner_digest,
    }
    assert rewind["inputs"]["runner_source"] == {
        "path": "src/kcg_connector/isaac/d38999_nut_regrasp_smoke.py",
        "sha256": runner_digest,
    }
    assert rewind["inputs"]["stage120_twist_contract"] == {
        "path": (
            "src/kcg_connector/config/"
            "d38999_q7_twist_probe_stage120_v1.yaml"
        ),
        "sha256": stage120_digest,
    }


def test_release_stability_and_non_claim_boundaries_are_explicit():
    document, _ = load_d38999_rewind_probe_contract(
        repository=PROJECT_ROOT
    )
    acceptance = document["acceptance"]
    assert math.degrees(
        acceptance["maximum_released_nut_drift_rad"]
    ) == pytest.approx(0.5)
    assert acceptance["maximum_released_body_axial_drift_m"] == 0.00005
    brake = document["interstroke_self_lock_brake_proxy"]
    assert brake["stiffness"] == 1.0
    assert brake["damping"] == 0.01
    assert brake["maximum_force_nm"] == 0.05
    assert brake["applied_after_twist"] is True
    assert brake["removed_after_regrasp_preload"] is True
    boundaries = document["boundaries"]
    assert boundaries["interstroke_brake_proxy_used"] is True
    assert boundaries["real_thread_self_lock_verified"] is False
    assert boundaries["assembly_success_claimed"] is False


def test_loader_fails_closed_on_looser_drift_gate(tmp_path):
    document = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    document["acceptance"]["maximum_released_nut_drift_rad"] = 1.0
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="rewind acceptance changed"):
        load_d38999_rewind_probe_contract(path, repository=PROJECT_ROOT)


def test_smoke_remains_lazy_and_exposes_rewind_as_explicit_opt_in():
    source = SMOKE_PATH.read_text(encoding="utf-8")
    before_main = source.split("def main():", 1)[0]
    assert "isaacsim" not in before_main
    assert "from pxr" not in before_main
    assert '"--rewind-probe"' in source
    assert "ISAAC D38999 Q7 REWIND PROBE V1" in source
    assert "load_d38999_rewind_probe_contract" in source
    assert "set_world_pose(" not in source


def test_pure_module_import_does_not_load_isaac():
    before = set(sys.modules)
    spec = importlib.util.spec_from_file_location(
        "d38999_rewind_probe_isolated",
        PACKAGE_ROOT / "kcg_connector/d38999_rewind_probe.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loaded = set(sys.modules) - before
    assert not any(
        name.startswith(("isaacsim", "omni", "pxr")) for name in loaded
    )
