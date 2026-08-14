"""Pure static contracts for the Isaac tabletop pick entry point."""

import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SMOKE_PATH = PACKAGE_ROOT / "isaac/connector_tabletop_pick_smoke.py"


def _module():
    spec = importlib.util.spec_from_file_location(
        "connector_tabletop_pick_smoke", SMOKE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pick_entrypoint_import_is_runtime_lazy():
    script = f'''
import importlib.util
import json
import sys
path = {str(SMOKE_PATH)!r}
spec = importlib.util.spec_from_file_location("pick_smoke", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
for name in ("isaacsim", "omni", "pxr"):
    assert name not in sys.modules, name
print(json.dumps({{"lazy_pick_import": True}}))
'''
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == {"lazy_pick_import": True}


def test_contact_classifier_is_exact_and_phase_compatible():
    module = _module()
    roots = (
        "/World/HandArm",
        "/World/ConnectorTabletopV1/Table",
        "/World/ConnectorTabletopV1/FixedFixture",
        "/World/ConnectorTabletopV1/ConnectorPair/Receptacle",
        "/World/ConnectorTabletopV1/ConnectorPair/Plug",
    )
    robot_link = "/World/HandArm/Geometry/world/iiwa_link_0/iiwa_link_1"
    assert module._classify_robot_external_contact(
        (robot_link, roots[1]), *roots
    ) == "table"
    assert module._classify_robot_external_contact(
        (robot_link, roots[2]), *roots
    ) == "fixture"
    assert module._classify_robot_external_contact(
        (robot_link, roots[3] + "/Shell"), *roots
    ) == "fixed_endpoint"
    assert module._classify_robot_external_contact(
        (robot_link, roots[4] + "/CouplingNut"), *roots
    ) == "loose_plug"
    assert module._classify_robot_external_contact(
        ("/World/HandArmish", roots[1]), *roots
    ) is None


def test_plug_table_classifier_requires_both_exact_subtrees():
    module = _module()
    plug = "/World/ConnectorTabletopV1/ConnectorPair/Plug"
    table = "/World/ConnectorTabletopV1/Table"
    assert module._is_plug_table_contact(
        (plug + "/BodyAssembly", table), plug, table
    )
    assert not module._is_plug_table_contact(
        (plug + "ish", table), plug, table
    )
    assert not module._is_plug_table_contact(
        (plug + "/BodyAssembly", table + "Material"), plug, table
    )


def test_allowed_plug_contact_requires_an_actual_finger_link():
    module = _module()
    robot = "/World/HandArm"
    plug = "/World/ConnectorTabletopV1/ConnectorPair/Plug"
    assert module._is_finger_plug_contact(
        (
            robot + "/Geometry/world/iiwa_link_0/f1Link1",
            plug + "/CouplingNut",
        ),
        robot,
        plug,
    )
    assert not module._is_finger_plug_contact(
        (
            robot + "/Geometry/world/iiwa_link_0/iiwa_link_1",
            plug + "/BodyAssembly",
        ),
        robot,
        plug,
    )


def test_scalar_first_quaternion_difference_is_shortest_angle():
    module = _module()
    half_angle = 0.25
    rotated = (
        __import__("math").cos(half_angle),
        0.0,
        0.0,
        __import__("math").sin(half_angle),
    )
    assert module._array_quaternion_error_radians(
        (1.0, 0.0, 0.0, 0.0), rotated
    ) == __import__("pytest").approx(0.5)
    assert module._array_quaternion_error_radians(
        (1.0, 0.0, 0.0, 0.0), tuple(-value for value in rotated)
    ) == __import__("pytest").approx(0.5)


def test_source_forbids_object_pose_drive_and_attachment():
    source = SMOKE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    attribute_calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
    }
    for forbidden in (
        "set_world_pose",
        "set_local_pose",
        "set_default_state",
        "set_linear_velocity",
        "set_angular_velocity",
    ):
        assert forbidden not in attribute_calls
    assert "FixedJoint" not in source
    assert "CreateJoint" not in source
    assert "RemovePrim" not in source
    assert '"attachment": "none"' in source
    assert '"object_drive": "none"' in source
    assert '"object_pose_writes_after_start": 0' in source


def test_source_uses_real_effort_contact_and_material_evidence():
    source = SMOKE_PATH.read_text(encoding="utf-8")
    assert "get_measured_joint_efforts" in source
    assert '"torque_channels": ["f1j2", "f2j1", "f3j2"]' in source
    assert "get_full_contact_report" in source
    assert "TraverseInstanceProxies" in source
    assert "ComputeBoundMaterial" in source
    assert "grip_material_contact_records" in source
    assert "plug_table_records" in source
    assert "loaded_channels" in source
    assert "loose_plug_unexpected_robot_link" in source
    assert "maximum_post_tare_absolute_delta_by_channel" in source
    assert "final_maximum_absolute_torque_delta" in source
    assert "final_loaded_channels" in source
    assert "maximum_final_observable_joint_speed" in source
    assert "maximum_final_post_solver_joint_speed" in source
    assert "tail_diagnostics_finite" in source
    assert "final_tail_window_steps" in source


def test_source_declares_screening_boundaries_and_all_motion_phases():
    source = SMOKE_PATH.read_text(encoding="utf-8")
    assert "joint_interpolation_screening_not_collision_planned" in source
    assert '"self_collision_verified": False' in source
    for phase in (
        "initial_settle",
        "home_hand_open",
        "open_hand_descent",
        "open_grasp_tare",
        "physical_hand_closure",
        "physical_grip_preload",
        "physical_grip_lift",
        "unsupported_final_hold",
    ):
        assert f'phase = "{phase}"' in source


def test_gui_keep_open_is_supported():
    source = SMOKE_PATH.read_text(encoding="utf-8")
    assert '"--gui"' in source
    assert '"--keep-open"' in source
    assert "while simulation_app.is_running()" in source
