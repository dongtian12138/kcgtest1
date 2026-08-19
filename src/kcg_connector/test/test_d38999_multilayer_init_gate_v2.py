from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
RUNNER_PATH = (
    WORKSPACE_ROOT
    / "src/kcg_connector/isaac/d38999_multilayer_init_gate_v2.py"
)


def _load_runner():
    spec = importlib.util.spec_from_file_location("d38999_init_gate_v2_test", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v2_initialization_gate_is_frozen_to_its_historical_a1_asset() -> None:
    runner = _load_runner()
    assert runner.EXPECTED_ACTIVE_COLLIDER_COUNT == 270
    assert runner.EXPECTED_FIXED_COLLIDER_COUNT == 199
    assert runner.EXPECTED_BODY_COLLIDER_COUNT == 70
    assert runner.EXPECTED_NUT_COLLIDER_COUNT == 1
    assert runner.START_SEPARATION_M == 0.0055
    assert runner.INTERNAL_DATUM_TARGET_M == 1.0e-05
    assert runner.EXPECTED_SHA256["model"] == (
        "d5bcc5e8b28e31912f65cd87a0bbe5d7a035744f7f7d8c7b785e17cdad382a6e"
    )
    current_model_sha256 = runner._sha256(
        WORKSPACE_ROOT / runner.MODEL_RELATIVE
    )
    assert current_model_sha256 == (
        "a3e43d53150dc94f1c703e41bcc6facd7df0f55ea7e083f8debf600349e8cc3d"
    )
    assert current_model_sha256 != runner.EXPECTED_SHA256["model"]
    assert runner._sha256(WORKSPACE_ROOT / runner.MAPPING_RELATIVE) == runner.EXPECTED_SHA256["mapping"]
    assert runner._sha256(WORKSPACE_ROOT / runner.FINE_OFFSET_RESULT_RELATIVE) == runner.EXPECTED_SHA256["fine_offset_result"]

    master = yaml.safe_load(
        (WORKSPACE_ROOT / runner.MASTER_RELATIVE).read_text(encoding="utf-8")
    )
    assert master["acceptance_limits"]["maximum_fixed_receptacle_translation_drift_m"] == 5.0e-06
    assert master["acceptance_limits"]["maximum_noncompliant_hard_penetration_m"] == 5.0e-05

    source = RUNNER_PATH.read_text(encoding="utf-8")
    forbidden_call = "configure_" + "continuous_plug_guide_runtime_collision"
    assert forbidden_call not in source
    assert "runtime_geometry_created_count\": 0" in source
    assert "old_runtime_proxy_builder_called\": False" in source
    assert "EXPECTED_ACTIVE_COLLIDER_COUNT = 270" in source


def test_pair_class_is_posthoc_and_deterministic() -> None:
    runner = _load_runner()
    fixed = runner.FIXED_PATH + "/MatingShell/Any"
    body = runner.BODY_PATH + "/MatingShell/Any"
    nut = runner.NUT_PATH + "/CouplingNutGraspCollision"
    assert runner._pair_class({"collider_paths": [fixed, body]}) == "fixed_receptacle_vs_loose_plug"
    assert runner._pair_class({"collider_paths": [body, nut]}) == "loose_plug_body_vs_coupling_nut"
    assert runner._pair_class({"collider_paths": [fixed, fixed]}) == "other_connector_contact"
