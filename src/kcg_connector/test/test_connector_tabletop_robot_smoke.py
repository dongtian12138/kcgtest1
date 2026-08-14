"""Pure contracts for the robot-inclusive tabletop Isaac entry point."""

import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from kcg_connector.isaac_tabletop_scene import (
    load_connector_tabletop_scene,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SMOKE_PATH = PACKAGE_ROOT / "isaac/connector_tabletop_robot_smoke.py"
CONFIG_PATH = PACKAGE_ROOT / "config/connector_tabletop_scene_v1.yaml"
REPOSITORY_ROOT = PACKAGE_ROOT.parents[1]


def _load_smoke_module():
    spec = importlib.util.spec_from_file_location(
        "connector_tabletop_robot_smoke", SMOKE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_robot_tabletop_entrypoint_import_is_runtime_lazy():
    script = f'''
import importlib.util
import json
import sys
path = {str(SMOKE_PATH)!r}
spec = importlib.util.spec_from_file_location("robot_tabletop_smoke", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
for name in ("isaacsim", "omni", "pxr"):
    assert name not in sys.modules, name
print(json.dumps({{"lazy_robot_tabletop_import": True}}))
'''
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == {
        "lazy_robot_tabletop_import": True
    }


def test_home_is_exactly_seven_arm_plus_four_active_hand_zeros():
    module = _load_smoke_module()
    dof_names = (
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
    active_names, indices, targets = module._home_control_spec(dof_names)
    assert active_names == (
        *(f"iiwa_joint_{index}" for index in range(1, 8)),
        "f1j1",
        "f1j2",
        "f2j1",
        "f3j2",
    )
    assert tuple(dof_names[index] for index in indices) == active_names
    assert targets == pytest.approx((0.0,) * 11)
    assert set(module.MIMIC_HAND_JOINT_NAMES).isdisjoint(active_names)


def test_home_spec_rejects_missing_unexpected_and_duplicate_dofs():
    module = _load_smoke_module()
    with pytest.raises(ValueError, match="unexpected articulation"):
        module._home_control_spec(module.EXPECTED_DOF_NAMES[:-1])
    duplicate = (
        module.EXPECTED_DOF_NAMES[:-1]
        + (module.EXPECTED_DOF_NAMES[0],)
    )
    with pytest.raises(ValueError, match="unique"):
        module._home_control_spec(duplicate)


def test_contact_classifier_requires_both_robot_and_exact_table_subtree():
    module = _load_smoke_module()
    robot = "/World/HandArm"
    table = "/World/ConnectorTabletopV1/Table"
    assert module._is_robot_table_contact(
        (
            "/World/HandArm/Geometry/world/iiwa_link_0",
            "/World/ConnectorTabletopV1/Table",
        ),
        robot,
        table,
    )
    assert not module._is_robot_table_contact(
        (
            "/World/HandArmish/Geometry",
            "/World/ConnectorTabletopV1/Table",
        ),
        robot,
        table,
    )
    assert not module._is_robot_table_contact(
        (
            "/World/HandArm/Geometry/world/iiwa_link_0",
            "/World/ConnectorTabletopV1/TableMaterial",
        ),
        robot,
        table,
    )


def test_table_front_and_static_robot_asset_preserve_clearance_contract():
    module = _load_smoke_module()
    config = load_connector_tabletop_scene(CONFIG_PATH)
    table_front_x = (
        config.table.center_m[0] - 0.5 * config.table.size_m[0]
    )
    robot_asset = (
        REPOSITORY_ROOT
        / "artifacts/kcg_connector/isaac/robot/handarm/handarm.usda"
    )
    connector_asset = (
        REPOSITORY_ROOT
        / "artifacts/kcg_connector/isaac/connector_pair.usda"
    )
    assert table_front_x == pytest.approx(0.150)
    assert module.ROBOT_TABLE_MINIMUM_CLEARANCE_M >= 0.005
    assert robot_asset.is_file()
    assert connector_asset.is_file()


def test_entrypoint_has_gui_keep_open_and_no_runtime_pose_write_calls():
    source = SMOKE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    attribute_calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
    }
    assert '"--gui"' in source
    assert '"--keep-open"' in source
    assert '"object_pose_writes_after_start": 0' in source
    assert "set_world_pose" not in attribute_calls
    assert "set_local_pose" not in attribute_calls
    assert "set_default_state" not in attribute_calls
    assert "while simulation_app.is_running()" in source
