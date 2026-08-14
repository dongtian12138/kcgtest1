"""Pure contracts for the Isaac Home-to-pregrasp entry point."""

import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SMOKE_PATH = (
    PACKAGE_ROOT
    / "isaac/connector_tabletop_home_to_pregrasp_smoke.py"
)


def _module():
    spec = importlib.util.spec_from_file_location(
        "tabletop_home_to_pregrasp_smoke", SMOKE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_entrypoint_import_is_runtime_lazy():
    script = f'''
import importlib.util
import json
import sys
path = {str(SMOKE_PATH)!r}
spec = importlib.util.spec_from_file_location("pregrasp_smoke", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
for name in ("isaacsim", "omni", "pxr"):
    assert name not in sys.modules, name
print(json.dumps({{"lazy_pregrasp_import": True}}))
'''
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == {"lazy_pregrasp_import": True}


def test_external_contact_classifier_is_exact_and_separates_categories():
    module = _module()
    roots = (
        "/World/HandArm",
        "/World/ConnectorTabletopV1/Table",
        "/World/ConnectorTabletopV1/FixedFixture",
        "/World/ConnectorTabletopV1/ConnectorPair",
    )
    robot_link = "/World/HandArm/Geometry/world/iiwa_link_0/iiwa_link_1"
    assert module._classify_external_contact(
        (robot_link, roots[1]), *roots
    ) == "table"
    assert module._classify_external_contact(
        (robot_link, roots[2]), *roots
    ) == "fixture"
    assert module._classify_external_contact(
        (robot_link, roots[3] + "/Plug"), *roots
    ) == "connector"
    assert module._classify_external_contact(
        ("/World/HandArmish", roots[1]), *roots
    ) is None
    assert module._classify_external_contact(
        (robot_link, roots[1] + "Material"), *roots
    ) is None


def test_entrypoint_declares_screening_limits_and_no_pose_writes():
    source = SMOKE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    attribute_calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
    }
    assert "joint_interpolation_screening_not_collision_planned" in source
    assert '"self_collision_verified": False' in source
    assert '"object_pose_writes_after_start": 0' in source
    assert "get_full_contact_report" in source
    assert "set_world_pose" not in attribute_calls
    assert "set_local_pose" not in attribute_calls
    assert "set_default_state" not in attribute_calls


def test_isaac_numpy_targets_are_converted_at_the_pure_contract_boundary():
    source = SMOKE_PATH.read_text(encoding="utf-8")
    assert "tuple(float(value) for value in start_arm)" in source
    assert "tuple(float(value) for value in final_arm)" in source


def test_gui_keep_open_is_supported():
    source = SMOKE_PATH.read_text(encoding="utf-8")
    assert '"--gui"' in source
    assert '"--keep-open"' in source
    assert "while simulation_app.is_running()" in source
