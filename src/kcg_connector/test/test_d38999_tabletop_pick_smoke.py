"""Static contracts for the independent D38999 Isaac pick entry point."""

import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SMOKE_PATH = PACKAGE_ROOT / "isaac/d38999_tabletop_pick_smoke.py"


def _module():
    spec = importlib.util.spec_from_file_location(
        "d38999_tabletop_pick_smoke", SMOKE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_entrypoint_import_is_runtime_lazy():
    script = f"""
import importlib.util
import json
import sys
path = {str(SMOKE_PATH)!r}
spec = importlib.util.spec_from_file_location("d38999_pick_smoke", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
for name in ("isaacsim", "omni", "pxr"):
    assert name not in sys.modules, name
print(json.dumps({{"lazy_d38999_pick_import": True}}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == {"lazy_d38999_pick_import": True}


def test_contact_classifier_uses_exact_d38999_subtrees():
    module = _module()
    roots = (
        "/World/HandArm",
        "/World/D38999TabletopV1/Table",
        "/World/D38999TabletopV1/FixedFixture",
        (
            "/World/D38999TabletopV1/D38999Pair/"
            "D38999Shell25JProxy/FixedReceptacle"
        ),
        (
            "/World/D38999TabletopV1/D38999Pair/"
            "D38999Shell25JProxy/LoosePlug"
        ),
    )
    finger = "/World/HandArm/Geometry/world/f1Link1"
    assert (
        module._classify_robot_external_contact((finger, roots[1]), *roots)
        == "table"
    )
    assert (
        module._classify_robot_external_contact((finger, roots[2]), *roots)
        == "fixture"
    )
    assert (
        module._classify_robot_external_contact(
            (finger, roots[3] + "/EntryShell"), *roots
        )
        == "fixed_endpoint"
    )
    assert (
        module._classify_robot_external_contact(
            (finger, roots[4] + "/CouplingNut/Segment_00"), *roots
        )
        == "loose_plug"
    )
    assert (
        module._classify_robot_external_contact(
            ("/World/HandArmish", roots[1]), *roots
        )
        is None
    )


def test_only_actual_finger_to_loose_d38999_is_allowed():
    module = _module()
    robot = "/World/HandArm"
    plug = (
        "/World/D38999TabletopV1/D38999Pair/" "D38999Shell25JProxy/LoosePlug"
    )
    assert module._is_finger_plug_contact(
        (
            robot + "/Geometry/world/iiwa_link_0/f3Link3",
            plug + "/BodyAssembly/RearBody",
        ),
        robot,
        plug,
    )
    assert not module._is_finger_plug_contact(
        (
            robot + "/Geometry/world/iiwa_link_0/handbase_link",
            plug + "/CouplingNut/Segment_00",
        ),
        robot,
        plug,
    )
    assert module._is_plug_table_contact(
        (plug + "/BodyAssembly/RearBody", "/World/Table"),
        plug,
        "/World/Table",
    )


def test_loose_colliders_are_grouped_by_the_two_rigid_body_subtrees():
    module = _module()
    loose = "/World/D38999/LoosePlug"
    body = loose + "/BodyAssembly"
    nut = loose + "/CouplingNut"
    assert (
        module._d38999_loose_collider_group(body + "/RearBody", body, nut)
        == "body"
    )
    assert (
        module._d38999_loose_collider_group(
            body + "/MatingShell/Segment_19", body, nut
        )
        == "body"
    )
    assert (
        module._d38999_loose_collider_group(nut + "/Segment_23", body, nut)
        == "nut"
    )
    assert (
        module._d38999_loose_collider_group(
            loose + "/CouplingNutJoint", body, nut
        )
        is None
    )


def test_finger_contacts_are_grouped_by_finger_and_loose_rigid_body():
    module = _module()
    robot = "/World/HandArm"
    loose = "/World/D38999/LoosePlug"
    body = loose + "/BodyAssembly"
    nut = loose + "/CouplingNut"
    assert module._finger_loose_contact_group(
        (robot + "/Geometry/world/f2Link2", body + "/RearBody"),
        robot,
        body,
        nut,
    ) == ("f2", "body")
    assert module._finger_loose_contact_group(
        (robot + "/Geometry/world/f3Link3", nut + "/Segment_04"),
        robot,
        body,
        nut,
    ) == ("f3", "nut")
    assert (
        module._finger_loose_contact_group(
            (robot + "/Geometry/world/handbase_link", nut + "/Segment_04"),
            robot,
            body,
            nut,
        )
        is None
    )


def test_body_contact_gate_requires_every_finger_but_allows_nut_contacts():
    module = _module()
    mixed_contacts = {
        "finger_body_group_records": {
            "f1": {"body": 1, "nut": 4},
            "f2": {"body": 2, "nut": 2},
            "f3": {"body": 1, "nut": 5},
        }
    }
    assert module._all_fingers_have_body_contact(mixed_contacts)

    nut_only_f2 = {
        "finger_body_group_records": {
            "f1": {"body": 1, "nut": 0},
            "f2": {"body": 0, "nut": 20},
            "f3": {"body": 1, "nut": 0},
        }
    }
    assert not module._all_fingers_have_body_contact(nut_only_f2)
    assert not module._all_fingers_have_body_contact({})


def test_nut_only_and_clear_contact_gates_are_strict():
    module = _module()
    nut_only = {
        "finger_body_group_records": {
            "f1": {"body": 0, "nut": 1},
            "f2": {"body": 0, "nut": 2},
            "f3": {"body": 0, "nut": 1},
        }
    }
    assert module._all_fingers_have_nut_contact(nut_only)
    assert module._zero_finger_body_contact(nut_only)
    assert not module._zero_finger_endpoint_contact(nut_only)
    clear = {
        "finger_body_group_records": {
            finger: {"body": 0, "nut": 0}
            for finger in ("f1", "f2", "f3")
        }
    }
    assert module._zero_finger_endpoint_contact(clear)


def test_quaternion_tail_diagnostics_are_sign_invariant_and_finite():
    module = _module()
    identity = (1.0, 0.0, 0.0, 0.0)
    same_rotation_opposite_sign = (-1.0, 0.0, 0.0, 0.0)
    relative = module._relative_array_quaternion(
        identity, same_rotation_opposite_sign
    )
    assert module._array_quaternion_error_radians(identity, relative) == 0.0
    statistics = module._speed_statistics((1.0, 2.0, 3.0))
    assert statistics["maximum"] == 3.0
    assert statistics["mean"] == 2.0
    assert statistics["median"] == 2.0
    assert statistics["rms"] > statistics["mean"]


def test_source_pins_exact_21_body_plus_24_nut_collider_topology():
    source = SMOKE_PATH.read_text(encoding="utf-8")
    assert '"body": 21' in source
    assert '"nut": 24' in source
    assert "plug_collision_counts != expected_plug_collision_counts" in source


def test_source_forbids_attachment_object_drive_and_pose_writes():
    source = SMOKE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    for forbidden in (
        "set_world_pose",
        "set_local_pose",
        "set_default_state",
        "set_linear_velocity",
        "set_angular_velocity",
    ):
        assert forbidden not in calls
    # The optional end-to-end continuation uses one explicit world-to-body
    # keying proxy only after measured physical insertion.  It is not an
    # attachment and its activation jump is gated before regrasp/twist.
    assert source.count("UsdPhysics.FixedJoint.Define(") == 1
    assert "EngagedKeyingProxy" in source
    assert '"constraint_is_real_keying_claim": False' in source
    assert "CreateJoint" not in source
    # The measured-pose keying proxy is removed exactly once when the same
    # body is handed to the rack/prismatic thread proxy.  No object prim or
    # physical connector geometry may be removed.
    assert source.count("stage.RemovePrim(") == 1
    assert "stage.RemovePrim(constraint_path)" in source
    assert '"attachment": "none"' in source
    assert '"object_drive": "none"' in source
    assert '"object_pose_writes_after_start": 0' in source


def test_source_uses_d38999_scene_real_effort_and_dual_velocity_gates():
    source = SMOKE_PATH.read_text(encoding="utf-8")
    assert "verify_d38999_pick_dependencies" in source
    assert "author_d38999_tabletop_scene" in source
    assert "connector_pair.usda" not in source
    assert "get_measured_joint_efforts" in source
    assert '"torque_channels": ["f1j2", "f2j1", "f3j2"]' in source
    assert '"operational_torque_target_nm": 1.8' in source
    assert "get_full_contact_report" in source
    assert "loose_plug_unexpected_robot_link" in source
    assert "maximum_post_tare_absolute_delta_by_channel" in source
    assert "maximum_final_observable_joint_speed" in source
    assert "maximum_final_post_solver_joint_speed" in source
    assert "final_body_observable_linear_speed" in source
    assert "final_body_post_solver_linear_speed" in source
    assert "final_body_observable_angular_speed" in source
    assert "final_body_post_solver_angular_speed" in source
    assert "finger_body_group_records" in source
    assert "postclosure_all_fingers_body_contact" in source
    assert "final_all_fingers_body_contact" in source
    assert "body_contact_gate" in source
    assert "and body_contact_gate" in source
    assert '"nut_contacts_allowed": True' in source
    assert "final_tail_net_rotation_rad" in source
    assert "nut_relative_to_body" in source


def test_source_declares_screening_and_self_collision_boundaries():
    source = SMOKE_PATH.read_text(encoding="utf-8")
    assert "joint_interpolation_screening_not_collision_planned" in source
    assert '"self_collision_verified": False' in source
    assert (
        '"candidate_kind": "geometry_screened_not_dynamics_validated"'
        in source
    )
    for phase in (
        "initial_settle",
        "home_hand_open",
        "open_hand_descent",
        "open_grasp_tare",
        "physical_hand_closure",
        "closed_hand_seating",
        "physical_grip_preload",
        "physical_grip_lift",
        "unsupported_final_hold",
    ):
        assert f'phase = "{phase}"' in source


def test_optional_insertion_reuses_same_world_and_measured_grasp_transform():
    source = SMOKE_PATH.read_text(encoding="utf-8")
    assert '"--insertion-probe"' in source
    assert source.count("World(") == 1
    assert "compensated_tcp_transform" in source
    assert "measured_tcp_body_transforms" in source
    assert "solve_fixed_q7_tcp_pose" in source
    assert "tuple(float(value) for value in target)" in source
    assert 'phase = "mixed_grip_transport_to_fixed_safe"' in source
    assert '"mixed_grip_preinsert"' in source
    assert '"mixed_grip_physical_insert"' in source
    assert '"physical_insertion_included": True' in source
    assert '"pose_source": insertion.boundaries.pose_source' in source
    assert '"vision_included": False' in source
    assert '"assembly_success_claimed": False' in source
    assert "build_proxy_collision_filter_plan" in source
    assert "apply_proxy_collision_filter" in source
    assert source.index("apply_proxy_collision_filter(") < source.index(
        "\n        world.reset()"
    )


def test_insertion_fail_fast_forbids_loose_world_contacts_and_keeps_2nm():
    source = SMOKE_PATH.read_text(encoding="utf-8")
    for category in ("loose_fixed", "loose_fixture", "loose_table"):
        assert f'"{category}"' in source
    assert 'metrics["first_insertion_forbidden_contact"]' in source
    assert "sample_post_tare_efforts()" in source
    assert "maximum_absolute_torque_delta_nm" in source
    assert '"attachment": "none"' in source
    assert '"object_drive": "none"' in source
    assert '"object_pose_writes_after_start": 0' in source


def test_end_to_end_mode_reuses_rotation_and_returns_home():
    source = SMOKE_PATH.read_text(encoding="utf-8")
    assert '"--end-to-end-probe"' in source
    assert '"--smooth-demo"' in source
    assert (
        'parser.error("--smooth-demo requires --end-to-end-probe")'
        in source
    )
    assert '"smooth_demo_v1"' in source
    assert "home_hand_open_speedup = 4.0" in source
    assert "q7_motion_speedup = 1.4" in source
    assert "q7_motion_speedup = 1.6" not in source
    assert source.count("/ home_hand_open_speedup") == 1
    assert source.count("/ q7_motion_speedup") == 3
    assert round(16.0 * 240 / 4.0) == 960
    assert round(24.0 * 240 / 1.4) == 4114
    assert '"q7_twist_rewind_and_return": q7_motion_speedup' in source
    assert "ISAAC D38999 END TO END V1" in source
    assert "evaluate_d38999_full_rotation" in source
    assert "validate_final_seating_contact_pairs" in source
    assert "for stroke_index in (1, 2, 3):" in source
    assert "post-regrasp progress recovery" in source
    assert 'phase = "end_to_end_final_release"' in source
    assert '"end_to_end_retreat_above_fixed"' in source
    assert '"end_to_end_reverse_transport"' in source
    assert '"end_to_end_reverse_mid_to_home"' in source
    assert 'phase = "end_to_end_home_hold"' in source
    assert 'metrics["proxy_assembly_verification"]' in source
    assert '"assembly_success_claimed": False' in source
    assert '"real_vision_included": False' in source


def test_smooth_demo_requires_end_to_end_without_starting_isaac():
    result = subprocess.run(
        [sys.executable, str(SMOKE_PATH), "--smooth-demo"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "--smooth-demo requires --end-to-end-probe" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr


def test_masked_rgbd_preflight_requires_end_to_end_before_isaac():
    result = subprocess.run(
        [
            sys.executable,
            str(SMOKE_PATH),
            "--pose-preflight",
            "masked-rgbd",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert (
        "--pose-preflight masked-rgbd requires --end-to-end-probe"
        in result.stderr
    )
    assert "ModuleNotFoundError" not in result.stderr


def test_masked_rgbd_preflight_is_same_world_opt_in_and_fail_closed():
    source = SMOKE_PATH.read_text(encoding="utf-8")
    assert 'choices=("none", "masked-rgbd")' in source
    assert 'default="none"' in source
    assert source.count("World(") == 1
    assert source.count("\n        world.reset()") == 1
    assert "capture_d38999_rgbd_runtime(" in source
    capture_index = source.index("capture_d38999_rgbd_runtime(")
    settled_index = source.index("settled_on_table = bool(")
    intentional_motion_index = source.index('phase = "home_hand_open"')
    assert settled_index < capture_index < intentional_motion_index
    assert "preflight.passed is not True" in source
    assert "masked RGB-D pose preflight failed before intentional " in source
    assert '"robot motion"' in source
    assert "and pose_preflight_gate" in source
    assert '"pose_preflight_passed"' in source
    assert "report.json" not in source


def test_masked_rgbd_preflight_keeps_control_truth_boundary_explicit():
    source = SMOKE_PATH.read_text(encoding="utf-8")
    assert '"control_pose_provider": "sim_ground_truth"' in source
    assert '"masked_rgbd_xy_used_for_control": False' in source
    assert '"truth_orientation_used": True' in source
    assert '"foundation_pose": False' in source
    assert '"real_vision_included": False' in source
    assert '"masked_rgbd_preflight_included"' in source
    assert "if arguments.pose_preflight == \"masked-rgbd\":" in source
    assert source.index(
        "from isaacsim.sensors.camera import Camera"
    ) > source.index('if arguments.pose_preflight == "masked-rgbd":')


def test_rgbd_usd_lighting_binding_is_defined_for_headless_preflight():
    """Headless capture must not depend on the GUI-only lighting branch."""

    source = SMOKE_PATH.read_text(encoding="utf-8")
    unified_pxr_import = source.split("from pxr import (", 1)[1].split(
        ")", 1
    )[0]
    assert "UsdLux," in unified_pxr_import
    assert "from pxr import UsdLux" not in source


def test_end_to_end_brake_is_applied_before_each_hold_window():
    """The proxy thread must be held before measuring post-twist stability."""

    source = SMOKE_PATH.read_text(encoding="utf-8")
    function = source.split("def run_rotation_stroke", 1)[1]
    function = function.split("stroke_reports = []", 1)[0]
    brake_apply = function.index("brake_drive = UsdPhysics.DriveAPI.Apply")
    hold_phase = function.index("end_to_end_rotation_{stroke_index}_hold")
    hold_loop = function.index("for _ in range(hold_steps)")
    assert brake_apply < hold_phase < hold_loop
    assert "maximum_force_nm" in function[brake_apply:hold_phase]


def test_inserted_regrasp_has_measured_temporary_anti_spin_proxy():
    """Free Nut motion is bounded during release, then the proxy is removed."""

    source = SMOKE_PATH.read_text(encoding="utf-8")
    assert 'pre_twist_stability = {' in source
    assert 'phase = "end_to_end_release_mixed_grip"' in source
    assert "pre_twist_drive = UsdPhysics.DriveAPI.Apply" in source
    assert '"maximum_drift_rad"' in source
    assert '"maximum_observable_speed_rad_s"' in source
    assert '"maximum_post_solver_speed_rad_s"' in source
    assert "pre_twist_hinge_prim.RemoveAPI" in source
    assert '"removed_before_thread_activation": True' in source
    assert "and pre_twist_drift_gate" in source


def test_final_seating_has_low_force_moving_stabilizer_proxy():
    """Only stroke three gets an explicit, reported 0.05 Nm stabilizer."""

    source = SMOKE_PATH.read_text(encoding="utf-8")
    function = source.split("def run_rotation_stroke", 1)[1]
    function = function.split("stroke_reports = []", 1)[0]
    assert "if stroke_index == 3:" in function
    assert "final_seating_stabilizer =" in function
    assert "expected_nut_angle =" in function
    assert "GetTargetPositionAttr()" in function
    assert "target_position_attr.Set" in function
    assert '"final_seating_stabilizer_proxy_used"' in function
    assert '"final_seating_stabilizer_maximum_force_nm"' in function


def test_gui_keep_open_is_supported():
    source = SMOKE_PATH.read_text(encoding="utf-8")
    assert '"--gui"' in source
    assert '"--keep-open"' in source
    assert "while simulation_app.is_running()" in source


def test_virtual_wrist_ft_is_opt_in_monitor_only_and_keeps_default_unchanged():
    """The 6D reaction wrench must not silently become an E2E success gate."""

    source = SMOKE_PATH.read_text(encoding="utf-8")
    assert '"--wrist-ft-monitor"' in source
    assert '"--wrist-ft-config"' in source
    error_text = (
        'parser.error("--wrist-ft-monitor requires --end-to-end-probe")'
    )
    assert error_text in source
    assert "if arguments.wrist_ft_monitor:" in source
    assert "reaction_row_index(" in source
    assert "robot.get_measured_joint_forces" in source
    assert "Gf.Matrix3d(sensor_transform.GetRotation())" in source
    assert "column_rotation_from_gf_matrix3d(" in source
    assert "GetRotation().GetMatrix()" not in source
    assert "wrist_ft_monitor.capture_home_tare()" in source
    assert "wrist_ft_monitor.capture_payload_baseline()" in source
    assert '"modifies_e2e_pass_gate": False' in source
    assert '"residual_v1_enabled": False' in source
    assert '"safety_gate_claimed": False' in source


def test_fail_fast_records_phase_step_paths_and_nonfinite_indices():
    source = SMOKE_PATH.read_text(encoding="utf-8")
    assert 'metrics["first_forbidden_contact"]' in source
    assert 'metrics["first_nonfinite_state"]' in source
    assert 'metrics["first_nonfinite_effort"]' in source
    assert 'metrics["first_torque_safety_violation"]' in source
    assert '"global_step": global_step' in source
    assert '"phase_step": phase_step' in source
    assert '"paths": list(paths)' in source
    assert "maximum_absolute_torque_delta_nm" in source
    assert '"operational_torque_target_exceeded"' in source
    assert "raise RuntimeError" in source


def test_metrics_json_replaces_nonfinite_values_with_null():
    module = _module()
    encoded = module._metrics_json(
        {
            "finite": 1.25,
            "nested": [float("nan"), float("inf"), -float("inf")],
        }
    )
    assert "NaN" not in encoded
    assert "Infinity" not in encoded
    assert json.loads(encoded) == {
        "finite": 1.25,
        "nested": [None, None, None],
    }


def test_process_exit_is_pinned_to_isaac_fast_shutdown_close_argument():
    source = SMOKE_PATH.read_text(encoding="utf-8")
    expected = "simulation_app.close(exit_code=0 if passed else 1)"
    assert expected in source
    assert source.count("simulation_app.close(") == 1
    assert "return 0 if passed else 1" in source
    assert "raise SystemExit(main())" in source
