"""Pure contract tests for the standalone physical tabletop-v1 scene."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from kcg_connector.isaac_tabletop_scene import (
    TABLETOP_SCHEMA_VERSION,
    load_connector_tabletop_scene,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PACKAGE_ROOT / "config/connector_tabletop_scene_v1.yaml"


def _config():
    return load_connector_tabletop_scene(CONFIG_PATH)


def _invalid_document(tmp_path, mutator):
    document = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    mutator(document)
    path = tmp_path / "invalid_tabletop.yaml"
    path.write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )
    return path


def test_tabletop_v1_layout_is_exact_separated_and_json_safe():
    config = _config()
    assert config.schema_version == TABLETOP_SCHEMA_VERSION
    assert config.table.top_z_m == pytest.approx(0.200)
    assert config.fixed_endpoint.fixture_top_z_m == pytest.approx(0.240)
    assert config.fixed_endpoint.receptacle_origin_m == pytest.approx(
        (0.550, 0.185, 0.276)
    )
    assert config.loose_endpoint.initial_bottom_z_m == pytest.approx(
        0.215
    )
    assert config.physics.settle_steps == 480
    assert config.physics.tail_steps == 60
    assert config.loose_endpoint.minimum_endpoint_separation_m == (
        pytest.approx(0.300)
    )
    assert config.world.root_prim_path == "/World/ConnectorTabletopV1"
    json.dumps(config.as_dict(), allow_nan=False, sort_keys=True)


def test_tabletop_starts_clear_of_robot_base_and_keeps_pickup_reachable():
    config = _config()
    table_min_x = config.table.center_m[0] - 0.5 * config.table.size_m[0]
    assert table_min_x == pytest.approx(0.150)
    assert config.loose_endpoint.initial_center_m[0] == pytest.approx(0.520)
    assert config.fixed_endpoint.receptacle_origin_m[0] == pytest.approx(
        0.550
    )
    assert config.loose_endpoint.initial_center_m[1] < 0.0
    assert config.fixed_endpoint.receptacle_origin_m[1] > 0.0


@pytest.mark.parametrize(
    "mutator,message",
    (
        (
            lambda document: document.update(unexpected=True),
            "keys are invalid",
        ),
        (
            lambda document: document.update(schema_version="wrong"),
            "unsupported",
        ),
        (
            lambda document: document["table"].update(size_m=[0.8, 0, 0.08]),
            "positive",
        ),
        (
            lambda document: document["table"].update(
                center_m=[0.55, 0.0, float("nan")]
            ),
            "finite",
        ),
        (
            lambda document: document["table"].update(
                color_rgb=[0.2, 1.1, 0.3]
            ),
            r"\[0, 1\]",
        ),
        (
            lambda document: document["world"].update(
                root_prim_path="World/Relative"
            ),
            "absolute /World",
        ),
        (
            lambda document: document["table"].update(
                static_friction=0.5, dynamic_friction=0.6
            ),
            "static friction",
        ),
        (
            lambda document: document["physics"].update(
                gravity_m_s2=9.81
            ),
            "point down",
        ),
        (
            lambda document: document["physics"].update(
                settle_duration_s=1.99
            ),
            "at least 2 seconds",
        ),
        (
            lambda document: document["loose_endpoint"].update(
                initial_center_m=[0.52, -0.21, 0.200]
            ),
            "start above",
        ),
        (
            lambda document: document["loose_endpoint"].update(
                initial_center_m=[1.2, -0.21, 0.275]
            ),
            "outside the table",
        ),
        (
            lambda document: document["loose_endpoint"].update(
                initial_center_m=[0.55, 0.10, 0.275]
            ),
            "not separated",
        ),
        (
            lambda document: document["fixed_endpoint"].update(
                fixture_center_m=[0.55, 0.185, 0.221]
            ),
            "sit exactly on the table",
        ),
        (
            lambda document: document["fixed_endpoint"].update(
                receptacle_origin_m=[0.55, 0.184, 0.276]
            ),
            "centered",
        ),
        (
            lambda document: document["world"].update(
                physics_material_prim_path=(
                    "/World/ConnectorTabletopV1/ConnectorPair/Material"
                )
            ),
            "not canonical",
        ),
        (
            lambda document: document["loose_endpoint"].update(
                minimum_endpoint_separation_m=0.001
            ),
            "at least 0.3 m",
        ),
        (
            lambda document: document["physics"].update(rate_hz=120),
            "exactly 240 Hz",
        ),
        (
            lambda document: document["physics"].update(rate_hz=True),
            "positive integer",
        ),
        (
            lambda document: document["fixed_endpoint"].update(
                unexpected=True
            ),
            "keys are invalid",
        ),
        (
            lambda document: document["physics"].update(
                tail_duration_s=0.251
            ),
            "whole steps",
        ),
        (
            lambda document: document["physics"].update(
                maximum_table_penetration_m=0.1
            ),
            "safety bound",
        ),
    ),
)
def test_tabletop_v1_rejects_malformed_or_unsafe_layout(
    tmp_path, mutator, message
):
    path = _invalid_document(tmp_path, mutator)
    with pytest.raises(ValueError, match=message):
        load_connector_tabletop_scene(path)


def test_geometry_validator_rejects_wrong_asset_child_paths(tmp_path):
    path = _invalid_document(
        tmp_path,
        lambda document: document["loose_endpoint"].update(
            body_prim_path=(
                "/World/ConnectorTabletopV1/ConnectorPair/Plug/WrongBody"
            )
        ),
    )
    with pytest.raises(ValueError, match="do not match the asset"):
        load_connector_tabletop_scene(path)


def test_tabletop_dataclasses_are_immutable():
    config = _config()
    with pytest.raises(Exception):
        config.table.center_m = (0.0, 0.0, 0.0)
    changed = replace(config.table, center_m=(0.0, 0.0, 0.0))
    assert changed.center_m != config.table.center_m


def test_import_does_not_load_isaac_omni_or_pxr():
    script = r'''
import json
import sys
from kcg_connector.isaac_tabletop_scene import (
    load_connector_tabletop_scene,
)
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


def test_standalone_smoke_import_is_also_runtime_lazy():
    smoke_path = PACKAGE_ROOT / "isaac/connector_tabletop_smoke.py"
    script = f'''
import importlib.util
import json
import sys
smoke_path = {str(smoke_path)!r}
spec = importlib.util.spec_from_file_location("tabletop_smoke", smoke_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
for name in ("isaacsim", "omni", "pxr"):
    assert name not in sys.modules, name
print(json.dumps({{"lazy_smoke_import": True}}))
'''
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == {"lazy_smoke_import": True}
