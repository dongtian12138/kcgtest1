"""Pure contracts for the D38999 tabletop plus Home robot smoke."""

import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from kcg_connector.d38999_tabletop_scene import (
    load_d38999_tabletop_scene,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = PACKAGE_ROOT.parents[1]
SMOKE_PATH = PACKAGE_ROOT / "isaac/d38999_tabletop_robot_smoke.py"
ROBOT_ASSET = (
    REPOSITORY
    / "artifacts/kcg_connector/isaac/robot/handarm/handarm.usda"
)


def _module():
    spec = importlib.util.spec_from_file_location(
        "d38999_tabletop_robot_smoke", SMOKE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_entrypoint_import_is_runtime_lazy():
    script = f'''
import importlib.util
import json
import sys
spec = importlib.util.spec_from_file_location(
    "d38999_robot", {str(SMOKE_PATH)!r}
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
for name in ("isaacsim", "omni", "pxr"):
    assert name not in sys.modules, name
print(json.dumps({{"lazy_import": True}}))
'''
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == {"lazy_import": True}


def test_robot_asset_and_every_payload_are_hash_pinned():
    module = _module()
    verified = module._verify_robot_asset_bundle(ROBOT_ASSET)
    assert verified["handarm.usda"].name == "handarm.usda"
    assert len(verified) == 9
    assert set(verified) == set(module.ROBOT_ASSET_SHA256)


def test_home_contract_is_exact_seven_arm_plus_four_active_hand_zeros():
    module = _module()
    shuffled_names = (
        "f1j3",
        "iiwa_joint_7",
        "f2j2",
        "iiwa_joint_1",
        "f3j1",
        "iiwa_joint_2",
        "f3j3",
        "iiwa_joint_3",
        "f1j1",
        "iiwa_joint_4",
        "f1j2",
        "iiwa_joint_5",
        "f2j1",
        "iiwa_joint_6",
        "f3j2",
    )
    names, indices, targets = module._home_control_spec(shuffled_names)
    assert names == (
        *(f"iiwa_joint_{index}" for index in range(1, 8)),
        "f1j1",
        "f1j2",
        "f2j1",
        "f3j2",
    )
    assert tuple(shuffled_names[index] for index in indices) == names
    assert targets == pytest.approx((0.0,) * 11)
    assert set(module.MIMIC_HAND_JOINT_NAMES).isdisjoint(names)


def test_home_contract_rejects_missing_and_duplicate_dofs():
    module = _module()
    with pytest.raises(ValueError, match="unexpected articulation"):
        module._home_control_spec(module.EXPECTED_DOF_NAMES[:-1])
    duplicate = (
        module.EXPECTED_DOF_NAMES[:-1]
        + (module.EXPECTED_DOF_NAMES[0],)
    )
    with pytest.raises(ValueError, match="unique"):
        module._home_control_spec(duplicate)


def test_contact_classifier_separates_table_fixture_and_d38999():
    module = _module()
    config = load_d38999_tabletop_scene()
    robot = module.ROBOT_ROOT_PATH
    table = config.table.prim_path
    fixture = config.fixed_endpoint.fixture_prim_path
    connector = config.asset.model_root_prim_path
    robot_link = robot + "/Geometry/world/iiwa_link_2"
    assert module._classify_robot_external_contact(
        (robot_link, table), robot, table, fixture, connector
    ) == "table"
    assert module._classify_robot_external_contact(
        (robot_link, fixture + "/Child"),
        robot,
        table,
        fixture,
        connector,
    ) == "fixture"
    assert module._classify_robot_external_contact(
        (robot_link, connector + "/LoosePlug/BodyAssembly"),
        robot,
        table,
        fixture,
        connector,
    ) == "d38999"
    assert module._classify_robot_external_contact(
        (robot + "ish/Link", table), robot, table, fixture, connector
    ) is None
    assert module._classify_robot_external_contact(
        (robot_link, table + "Material"),
        robot,
        table,
        fixture,
        connector,
    ) is None


def test_script_is_home_only_and_has_no_object_pose_setter():
    source = SMOKE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    attribute_calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
    }
    assert "connector_pair.usda" not in source
    assert "d38999_tabletop_scene_v1.yaml" in source
    assert '"--gui"' in source
    assert '"--keep-open"' in source
    assert '"object_pose_writes_after_start": 0' in source
    assert '"task_scope": "Home hold only"' in source
    assert "set_world_pose" not in attribute_calls
    assert "set_local_pose" not in attribute_calls
    assert "set_default_state" not in attribute_calls
    assert "while simulation_app.is_running()" in source
    assert "exit_code=0 if passed else 1" in source


def test_scene_contract_retains_240_hz_and_separated_d38999_endpoints():
    config = load_d38999_tabletop_scene()
    assert config.physics.rate_hz == 240
    assert config.physics.settle_steps == 480
    assert config.loose_endpoint.initial_clearance_above_table_m == (
        pytest.approx(0.015)
    )
    assert config.asset.fixed_receptacle_prim_path.endswith(
        "/D38999Shell25JProxy/FixedReceptacle"
    )
