"""Pure gates for the independent D38999 tabletop physical smoke."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from kcg_connector.d38999_tabletop_scene import (
    DEFAULT_D38999_TABLETOP_CONFIG_PATH,
    D38999_TABLETOP_SCHEMA_VERSION,
    load_d38999_tabletop_scene,
    verify_d38999_tabletop_asset,
)


REPOSITORY = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SMOKE_PATH = PACKAGE_ROOT / "isaac/d38999_tabletop_smoke.py"
MULTILAYER_GRASP_SCENE = (
    REPOSITORY
    / "src/kcg_connector/config/d38999_multilayer_tabletop_scene_grasp_v1.yaml"
)
MULTILAYER_BUILD_RESULT = (
    REPOSITORY
    / "artifacts/agent_control/tasks/"
    "D38999-AUTONOMOUS-DYNAMIC-CLOSEOUT-V2/"
    "DYN-A2-NOMINAL-INSERTION-V2/"
    "A2_RUN05_NUT_BODY_SHOULDER_TARGETED_FIX_RESULT.json"
)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _document():
    return yaml.safe_load(
        DEFAULT_D38999_TABLETOP_CONFIG_PATH.read_text(encoding="utf-8")
    )


def _write(tmp_path, document):
    path = tmp_path / "d38999_tabletop.yaml"
    path.write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )
    return path


def test_shipped_scene_is_independent_and_exactly_15_mm_above_table():
    config = load_d38999_tabletop_scene()
    assert config.schema_version == D38999_TABLETOP_SCHEMA_VERSION
    assert config.table.top_z_m == pytest.approx(0.200)
    assert config.loose_endpoint.initial_bottom_z_m == pytest.approx(0.215)
    assert config.loose_endpoint.initial_clearance_above_table_m == (
        pytest.approx(0.015)
    )
    assert config.fixed_endpoint.fixture_top_z_m == pytest.approx(0.240)
    assert (
        config.fixed_endpoint.receptacle_origin_m[2]
        - config.fixed_endpoint.receptacle_bottom_offset_m
    ) == pytest.approx(0.240)
    assert config.physics.rate_hz == 240
    assert config.physics.settle_steps == 480
    assert config.physics.tail_steps == 120
    assert "connector_pair.usda" not in config.asset.local_path
    assert config.asset.local_path.endswith(
        "d38999_shell25j_61_pair_proxy_v1.usda"
    )
    assert "/Plug/" not in config.asset.body_prim_path
    assert "/LoosePlug/" in config.asset.body_prim_path
    json.dumps(config.as_dict(), allow_nan=False, sort_keys=True)


def test_shipped_asset_is_allowlisted_and_not_the_synthetic_asset():
    path = verify_d38999_tabletop_asset(
        load_d38999_tabletop_scene(), REPOSITORY
    )
    assert path.name == "d38999_shell25j_61_pair_proxy_v1.usda"
    assert path.stat().st_size > 100000


def test_current_multilayer_grasp_asset_uses_guarded_a2_shoulder_lineage():
    config = load_d38999_tabletop_scene(MULTILAYER_GRASP_SCENE)
    assert config.asset_profile.expected_body_collider_count == 72
    assert config.asset_profile.expected_nut_collider_count == 7
    asset = verify_d38999_tabletop_asset(config, REPOSITORY)
    build_result = json.loads(MULTILAYER_BUILD_RESULT.read_text(encoding="utf-8"))
    assert _sha256(MULTILAYER_BUILD_RESULT) == (
        "a8d144799c3d5e38ff04a875f39180c9a92c074e0fc2d7b34ac59f82b5918718"
    )
    assert _sha256(asset) == build_result["assembly_control"]["sha256_after"]
    assert build_result["determinism"]["identical"] is True


@pytest.mark.parametrize(
    ("mutator", "message"),
    (
        (lambda doc: doc.update(extra=True), "keys are invalid"),
        (
            lambda doc: doc["asset"].update(
                local_path="artifacts/kcg_connector/isaac/connector_pair.usda"
            ),
            "independent D38999",
        ),
        (lambda doc: doc["asset"].update(fingerprint="forbidden"), "keys are invalid"),
        (
            lambda doc: doc["asset"].update(
                body_prim_path=(
                    "/World/D38999TabletopV1/D38999Pair/"
                    "D38999Shell25JProxy/Plug/BodyAssembly"
                )
            ),
            "does not match D38999 USD",
        ),
        (
            lambda doc: doc["loose_endpoint"].update(
                initial_origin_m=[0.520, -0.210, 0.200]
            ),
            "declared clearance",
        ),
        (
            lambda doc: doc["loose_endpoint"].update(
                initial_clearance_above_table_m=0.030
            ),
            "declared clearance",
        ),
        (
            lambda doc: doc["fixed_endpoint"].update(
                receptacle_origin_m=[0.550, 0.185, 0.250]
            ),
            "sit exactly on fixture",
        ),
        (
            lambda doc: doc["physics"].update(rate_hz=120),
            "exactly 240 Hz",
        ),
        (
            lambda doc: doc["physics"].update(
                maximum_transient_table_penetration_m=0.010
            ),
            "safety bound",
        ),
        (
            lambda doc: doc["physics"].update(
                maximum_upright_axis_tilt_rad=0.5
            ),
            "10 degrees",
        ),
        (
            lambda doc: doc["table"].update(
                static_friction=0.5, dynamic_friction=0.6
            ),
            "static friction",
        ),
        (
            lambda doc: doc["loose_endpoint"].update(
                initial_origin_m=[0.550, 0.100, 0.215]
            ),
            "not physically separated",
        ),
    ),
)
def test_loader_rejects_wrong_asset_or_unsafe_physics(
    tmp_path, mutator, message
):
    document = deepcopy(_document())
    mutator(document)
    with pytest.raises(ValueError, match=message):
        load_d38999_tabletop_scene(_write(tmp_path, document))


def test_loader_import_does_not_load_isaac_omni_or_pxr():
    script = r'''
import json
import sys
from kcg_connector.d38999_tabletop_scene import load_d38999_tabletop_scene
for name in ("isaacsim", "omni", "pxr"):
    assert name not in sys.modules, name
print(json.dumps({"pure_import": True}))
'''
    environment = dict(__import__("os").environ)
    python_path = str(PACKAGE_ROOT)
    if environment.get("PYTHONPATH"):
        python_path += ":" + environment["PYTHONPATH"]
    environment["PYTHONPATH"] = python_path
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert json.loads(result.stdout) == {"pure_import": True}


def test_smoke_import_is_lazy_and_contains_no_pose_setter():
    script = f'''
import importlib.util
import json
import sys
spec = importlib.util.spec_from_file_location(
    "d38999_tabletop", {str(SMOKE_PATH)!r}
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
for name in ("isaacsim", "omni", "pxr"):
    assert name not in sys.modules, name
print(json.dumps({{"lazy_import": True}}))
'''
    environment = dict(__import__("os").environ)
    python_path = str(PACKAGE_ROOT)
    if environment.get("PYTHONPATH"):
        python_path += ":" + environment["PYTHONPATH"]
    environment["PYTHONPATH"] = python_path
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert json.loads(result.stdout) == {"lazy_import": True}
    source = SMOKE_PATH.read_text(encoding="utf-8")
    assert ".set_world_pose(" not in source
    assert "object_pose_writes_after_start" in source
    assert "exit_code=0 if passed else 1" in source
