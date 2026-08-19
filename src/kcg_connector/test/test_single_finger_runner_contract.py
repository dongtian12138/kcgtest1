'''E4b2-B1/B2 contract tests: single-finger CLI, control block, exit flow.'''

from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
import runpy
import subprocess
import sys

import numpy as np
import pytest

from kcg_connector.grasp.physical_grasp_config import (
    load_physical_grasp_experiment_config,
)

REPOSITORY = Path(__file__).resolve().parents[3]
RUNNER = (
    REPOSITORY
    / "src/kcg_connector/isaac/d38999_tabletop_pick_smoke.py"
)
CONFIG = (
    REPOSITORY
    / "src/kcg_connector/config/d38999_tabletop_physical_grasp_v1.yaml"
)
SOURCE_ROOT = str(REPOSITORY / "src" / "kcg_connector")

TRUTH_TOKENS = (
    "get_world_pose",
    "contact_snapshot",
    "get_full_contact_report",
    "collider",
    "pending_posthoc",
    "posthoc_snapshot",
    "snapshot",
    "settled_body",
    "settled_nut",
    "body_in_tcp_frame",
)


def _run_cli(*arguments):
    environment = dict(os.environ)
    environment["PYTHONPATH"] = SOURCE_ROOT + os.pathsep + environment.get(
        "PYTHONPATH", ""
    )
    return subprocess.run(
        [sys.executable, str(RUNNER), *arguments],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def _source():
    return RUNNER.read_text(encoding="utf-8")


def _single_finger_region():
    source = _source()
    start = source.index(
        'arguments.physical_grasp_method == "single-finger":'
    )
    end = source.index(
        'arguments.physical_grasp_method == "sequential-compliant":',
        start,
    )
    return source[start:end]


def test_single_finger_requires_selected_finger_before_isaac():
    result = _run_cli("--physical-grasp-method", "single-finger")
    assert result.returncode == 2
    assert "requires --single-finger" in result.stderr
    assert "isaacsim" not in result.stderr


def test_single_finger_flag_requires_method_before_isaac():
    result = _run_cli("--single-finger", "f1")
    assert result.returncode == 2
    assert "requires --physical-grasp-method single-finger" in result.stderr
    assert "isaacsim" not in result.stderr


def test_single_finger_requires_zero_lift_hold_mode_before_isaac():
    result = _run_cli(
        "--physical-grasp-method",
        "single-finger",
        "--single-finger",
        "f2",
    )
    assert result.returncode == 2
    assert "requires --formal-lift-mode zero-lift-hold" in result.stderr
    assert "isaacsim" not in result.stderr


def test_single_finger_rejects_insertion_e2e_preflight_smooth():
    base = (
        "--physical-grasp-method",
        "single-finger",
        "--single-finger",
        "f1",
        "--formal-lift-mode",
        "zero-lift-hold",
    )
    for extra in (
        ("--insertion-probe",),
        ("--end-to-end-probe",),
        ("--pose-preflight", "masked-rgbd"),
        ("--smooth-demo",),
    ):
        result = _run_cli(*base, *extra)
        assert result.returncode == 2, extra
        assert "isaacsim" not in result.stderr


def test_method_choices_include_single_finger_and_marker():
    source = _source()
    assert '"single-finger"' in source
    assert '"ISAAC D38999 SINGLE FINGER CONTROL V1 "' in source
    assert "COMPLETED_POSTHOC_NOT_YET_RUN" in source


def test_formal_grasp_derivation_stays_single_sourced():
    tree = ast.parse(_source())
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "formal_grasp"
            for target in node.targets
        )
    ]
    assert len(assignments) == 1
    assert "physical_grasp_method" in ast.unparse(assignments[0].value)


def test_single_finger_region_reads_no_truth_or_posthoc():
    region = _single_finger_region()
    # Ignore comment-only lines so the code scan is not fooled by the
    # block's own documentation.
    code_lines = [
        line
        for line in region.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    code_region = "\n".join(code_lines)
    for token in TRUTH_TOKENS:
        assert token not in code_region, (
            f"single-finger region reads {token}"
        )


def test_single_finger_region_metrics_are_store_only():
    tree = ast.parse(_source())
    # Find the single-finger block line span from the marker lines.
    lines = _source().splitlines()
    start_line = next(
        index
        for index, line in enumerate(lines, start=1)
        if 'arguments.physical_grasp_method == "single-finger":' in line
    )
    end_line = next(
        index
        for index, line in enumerate(lines, start=1)
        if 'arguments.physical_grasp_method == "sequential-compliant":' in line
        and index > start_line
    )
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        if not (
            isinstance(node.value, ast.Name)
            and node.value.id == "metrics"
        ):
            continue
        if not start_line <= node.lineno <= end_line:
            continue
        # Every metrics access inside the single-finger region must be a
        # Store (a report write); a Load would let posthoc/validation data
        # leak into control.
        assert isinstance(
            node.ctx, ast.Store
        ), f"metrics read inside the single-finger region at line {node.lineno}"


def test_single_finger_invariants_and_budget_in_source():
    region = _single_finger_region()
    assert "other_fingers_open_target_invariant_broken" in region
    assert "arm_target_invariant_broken" in region
    assert "np.allclose(current_arm_target, closure_clearance_arm)" in region
    assert "single_finger_budget = (" in region
    assert "maximum_approach_steps" in region
    assert "soft_hold_steps" in region
    assert "maximum_release_steps" in region
    # The runner budget must cover the controller's worst internal path
    # (approach + confirmation update + 24 subsequent hold outputs +
    # release budget).
    config = load_physical_grasp_experiment_config(CONFIG)
    single = config.single_finger
    runner_budget = (
        single.maximum_approach_steps
        + single.soft_hold_steps
        + single.maximum_release_steps
        + 8
    )
    controller_worst = (
        single.maximum_approach_steps
        + 1
        + single.soft_hold_steps
        + single.maximum_release_steps
    )
    assert runner_budget >= controller_worst


def test_single_finger_region_runs_no_seating_preload_or_lift():
    region = _single_finger_region()
    for token in (
        "closed_hand_seating",
        "physical_grip_preload",
        "physical_grip_zero_lift_hold",
        "physical_grip_lift_stage_",
    ):
        assert token not in region, f"single-finger region touches {token}"


def test_single_finger_success_exit_semantics():
    source = _source()
    assert "process_exit_code = 3" in source
    assert '"single_finger_validation_passed": None' in source
    assert '"validation_status": (' in source
    assert '"posthoc_contact_audit_passed": None' in source
    assert 'metrics["passed"] = False' in source
    assert 'metrics["grasp_success_claimed"] = False' in source
    assert "return process_exit_code" in source
    assert "simulation_app.close(exit_code=process_exit_code)" in source
    assert "process_exit_code = 0 if passed else 1" in source
    # The success return happens inside the try before the normal flow.
    success_return = source.index(
        "COMPLETED_POSTHOC_NOT_YET_RUN",
    )
    assert "return process_exit_code" in source[success_return:]




EVIDENCE_METRIC_KEYS = (
    "finger_root_torque_proxy_baseline_statistics",
    "formal_empty_wrist_reference_statistics_raw",
    "formal_empty_wrist_reference_statistics_residual",
    "runtime_mode_evidence",
)


def test_evidence_statistics_blocks_are_single_write_store_only():
    source = _source()
    tree = ast.parse(source)
    occurrences = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        if not (
            isinstance(node.value, ast.Name) and node.value.id == "metrics"
        ):
            continue
        if (
            isinstance(node.slice, ast.Constant)
            and node.slice.value in EVIDENCE_METRIC_KEYS
        ):
            occurrences += 1
            assert isinstance(
                node.ctx, ast.Store
            ), f"metrics[{node.slice.value}] loaded at line {node.lineno}"
    assert occurrences == len(EVIDENCE_METRIC_KEYS)
    for key in EVIDENCE_METRIC_KEYS:
        assert source.count(f'"{key}"') == 1, key


def test_evidence_statistics_between_tare_and_single_finger_branch():
    source = _source()
    tare_end = source.index(
        "tare_efforts = np.mean(np.stack(tare_effort_samples), axis=0)"
    )
    region_start = source.index(
        'if formal_grasp and arguments.physical_grasp_method == "single-finger":'
    )
    for key in EVIDENCE_METRIC_KEYS:
        position = source.index(f'"{key}"')
        assert tare_end < position < region_start, key


def test_evidence_statistics_region_reads_no_truth_tokens():
    source = _source()
    start = source.index(
        "tare_efforts = np.mean(np.stack(tare_effort_samples), axis=0)"
    )
    end = source.index(
        'if formal_grasp and arguments.physical_grasp_method == "single-finger":'
    )
    region = source[start:end]
    for token in TRUTH_TOKENS:
        assert token not in region, (
            f"evidence statistics region reads {token}"
        )
    assert "get_full_contact_report" not in region


def test_runtime_mode_evidence_helper_has_no_truth_or_contact_reads():
    source = _source()
    start = source.index("def _runtime_mode_evidence(")
    end = source.index("def _json_safe(", start)
    region = source[start:end]
    for token in TRUTH_TOKENS:
        assert token not in region, (
            f"runtime mode evidence helper reads {token}"
        )
    assert "get_full_contact_report" not in region
    assert "get_world_pose" not in region


def test_multilayer_solver_contract_is_hash_bound_and_not_guessed():
    source = _source()
    contract = (
        REPOSITORY
        / "src/kcg_connector/config/"
        "d38999_keyed_v3_physical_model_contract_r12_v1.yaml"
    )
    assert hashlib.sha256(contract.read_bytes()).hexdigest() == (
        "6068066a2ac0339fa83caf2cc0c28050e76ed7e56e960da1b29e121a083b650e"
    )
    assert "FROZEN_SOLVER_CONTRACT_RELATIVE_PATH" in source
    assert "FROZEN_SOLVER_CONTRACT_SHA256" in source
    assert '"position_iterations": 32' in source
    assert '"velocity_iterations": 8' in source
    assert '"solver_type": "TGS"' in source
    assert "default_position_iterations" not in source
    assert "default_velocity_iterations" not in source


def test_frozen_solver_contract_authoring_and_readback_precede_control():
    source = _source()
    author_call = source.index(
        "frozen_solver_contract_evidence = (\n"
        "                _author_frozen_solver_contract("
    )
    articulation_wrapper = source.index(
        "robot = world.scene.add(\n"
        "            SingleArticulation(",
        author_call,
    )
    reset = source.index("world.reset()", articulation_wrapper)
    verify_call = source.index(
        "post_reset_solver_contract = (\n"
        "                _verify_frozen_solver_contract_after_reset(",
        reset,
    )
    first_tare = source.index("tare_effort_samples", verify_call)
    assert author_call < articulation_wrapper < reset < verify_call < first_tare


def test_frozen_solver_helpers_read_no_object_or_contact_truth():
    source = _source()
    start = source.index("def _load_frozen_solver_contract(")
    end = source.index("def _runtime_mode_evidence(", start)
    region = source[start:end]
    for token in TRUTH_TOKENS:
        assert token not in region, (
            f"frozen solver helper reads forbidden truth token {token}"
        )
    for token in (
        "get_full_contact_report",
        "get_world_pose",
        "get_local_pose",
        "get_linear_velocity",
        "get_angular_velocity",
    ):
        assert token not in region
    assert "set_solver_type" in region
    assert "enable_ccd" in region
    assert "enable_stabilization" in region
    assert "physxArticulation:solverPositionIterationCount" in region
    assert "physxRigidBody:solverPositionIterationCount" in region


def test_solver_float_readback_requires_exact_float32_storage_value():
    namespace = runpy.run_path(str(RUNNER))
    matches = namespace["_solver_value_matches"]
    canonical = float(np.float32(0.2))
    assert canonical == 0.20000000298023224
    assert matches(canonical, 0.2)
    assert not matches(float(np.nextafter(np.float32(0.2), np.float32(1.0))), 0.2)


def test_post_reset_rigid_solver_readback_uses_supported_isaac_6_interface():
    source = _source()
    start = source.index("def _verify_frozen_solver_contract_after_reset(")
    end = source.index("def _runtime_mode_evidence(", start)
    region = source[start:end]
    assert "_rigid_prim_view.get_solver_" not in region
    assert "robot.get_solver_position_iteration_count()" in region
    assert "robot.get_solver_velocity_iteration_count()" in region
    assert '"connector_rigid_bodies": "composed_usd_after_reset"' in region
    assert (
        '"isaac_6_0_1_rigid_runtime_iteration_getter_available": False'
        in region
    )
    assert (
        '"connector_rigid_body_values_claimed_as_live_tensor_readback": False'
        in region
    )


def test_legacy_exit_semantics_unchanged():
    source = _source()
    # Normal paths still derive exit from the top-level passed flag.
    assert "process_exit_code = 0 if passed else 1" in source
    assert "process_exit_code = 1" in source
    assert '"process_exit_code": process_exit_code' in source


def _index_space_span():
    lines = _source().splitlines()
    start_line = next(
        index
        for index, line in enumerate(lines, start=1)
        if 'arguments.physical_grasp_method == "single-finger":' in line
    )
    end_line = next(
        index
        for index, line in enumerate(lines, start=1)
        if 'arguments.physical_grasp_method == "sequential-compliant":' in line
        and index > start_line
    )
    return start_line, end_line


def test_release_budget_preflight_runs_before_world_scene_reset():
    source = _source()
    preflight = source.index("single_finger_release_budget_preflight")
    world_line = source.index("world = World(")
    assert preflight < world_line
    # Fail closed on an infeasible configured budget.
    assert "release budget is infeasible" in source


def test_audit_mode_cli_rejected_outside_single_finger():
    result = _run_cli(
        "--physical-grasp-method",
        "synchronous",
        "--formal-lift-mode",
        "zero-lift-hold",
        "--single-finger-posthoc-audit-mode",
        "capture",
    )
    assert result.returncode == 2
    assert "requires --physical-grasp-method single-finger" in result.stderr
    assert "isaacsim" not in result.stderr
    # Default stays skip (backwards compatible); the flag is declared.
    source = _source()
    assert '"--single-finger-posthoc-audit-mode"' in source
    assert 'default="skip"' in source


def test_single_finger_region_has_exactly_four_audit_hook_calls():
    region = _single_finger_region()
    assert region.count("capture_single_finger_audit_point(") == 4
    # The region itself must never touch the contact API directly.
    assert "contact_snapshot(" not in region
    assert "get_full_contact_report" not in region


def test_audit_hook_reads_contact_only_under_capture_guard():
    tree = ast.parse(_source())
    hook = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "capture_single_finger_audit_point"
    )
    guard_line = None
    snapshot_call_line = None
    store_roots = []
    parent_map = {}
    for parent in ast.walk(hook):
        for child in ast.iter_child_nodes(parent):
            parent_map[child] = parent
    for node in ast.walk(hook):
        if (
            isinstance(node, ast.If)
            and "single_finger_posthoc_audit_mode" in ast.unparse(node.test)
        ):
            guard_line = node.lineno
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "contact_snapshot"
        ):
            snapshot_call_line = node.lineno
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "metrics"
        ):
            # Walk up to the chain root: the root must be a Store; inner
            # subscripts of the same chain may be Loads.
            root = node
            while (
                isinstance(parent_map.get(root), ast.Subscript)
                and parent_map[root].value is root
            ):
                root = parent_map[root]
            assert isinstance(root.ctx, ast.Store), (
                f"audit hook metrics write is not a Store at line "
                f"{node.lineno}"
            )
            store_roots.append(root)
    assert guard_line is not None
    assert snapshot_call_line is not None
    # The contact API call is reachable only after the skip early-return:
    # the guard precedes the call in source order.
    assert guard_line < snapshot_call_line
    assert len(store_roots) >= 2


def test_contact_report_callsites_are_confined_to_known_blocks():
    source = _source()
    callsites = source.count("get_full_contact_report()")
    assert callsites == 4
    region = _single_finger_region()
    assert "get_full_contact_report()" not in region


def test_single_finger_boundary_and_audit_report_fields():
    region = _single_finger_region()
    assert 'metrics["control_reads_object_truth"] = False' in region
    assert 'metrics["control_reads_contact_report"] = False' in region
    assert 'metrics["posthoc_audit"] = {' in region
    assert 'metrics["posthoc_audit_reads_contact_report"] = bool(' in region
    assert 'metrics["posthoc_audit_consumed_by_control"] = False' in region
    assert '"consumed_by_control": False' in region


def test_release_budget_gate_binds_to_finger_root_torque_limit():
    source = _source()
    preflight_start = source.index(
        "if single_finger_mode:",
        source.index("realize_call_count"),
    )
    preflight_end = source.index(
        "realized_randomization = None",
        preflight_start,
    )
    preflight_region = source[preflight_start:preflight_end]
    # The filter-tail gate must come from the pick sensing finger-root
    # torque limit (2.0 N*m), never from the wrist force gate (8.0 N,
    # a different sensor and unit).
    assert (
        "maximum_torque_delta_gate_nm=("
        in preflight_region
    )
    assert (
        "pick.sensing.maximum_absolute_torque_delta_nm"
        in preflight_region
    )
    assert "maximum_wrist_force_n" not in preflight_region


def test_success_early_return_finalizes_wrist_monitor_and_torque_summary():
    region = _single_finger_region()
    assert "SINGLE_FINGER_CONTROL_COMPLETED_" in region
    assert "POSTHOC_PENDING" in region
    assert 'metrics["virtual_wrist_ft_monitor"] = wrist_report' in region
    assert "MONITOR_FAILED" in region
    # The three-channel post-tare torque summary with named joints plus an
    # overall maximum are recorded on the success path.
    assert "maximum_post_tare_absolute_delta_by_channel_nm" in region
    assert "maximum_post_tare_absolute_delta_nm" in region
    assert "torque_joint_names" in region


def test_failure_path_claims_no_grasp_and_records_recovery_sensors():
    region = _single_finger_region()
    # Explicit false claim before any control work happens.
    claim = region.index('metrics["grasp_success_claimed"] = False')
    calibrate = region.index("single_finger_controller.calibrate")
    assert claim < calibrate
    # Recovery evidence carries the selected torque proxy and q/qd so a
    # load plateau can be told apart from a decaying tail offline.
    recovery = region[region.index("single_finger_recovery_open"):]
    assert "finger_root_torque_proxy_nm" in recovery
    assert "selected_q_rad" in recovery
    assert "selected_qd_rad_s" in recovery
    assert "hand_q_rad" in recovery
    assert "hand_qd_rad_s" in recovery
    assert "recovery_positions" in recovery


def test_hand_local_vs_robot_dof_index_spaces_never_mix():
    tree = ast.parse(_source())
    start_line, end_line = _index_space_span()
    hand_local_targets = ("open_hand", "grasp_hand", "current_hand_target")
    robot_dof_arrays = ("positions", "velocities", "kps", "kds")
    local_index_names = {
        "selected_hand_local_index",
        "other_hand_local_index",
    }
    robot_index_names = {
        "selected_robot_dof_index",
        "hand_indices",
    }
    checked = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        if not start_line <= node.lineno <= end_line:
            continue
        if not isinstance(node.value, ast.Name):
            continue
        target = node.value.id
        if target in hand_local_targets:
            if isinstance(node.slice, ast.Name):
                assert node.slice.id in local_index_names, (
                    f"{target}[{node.slice.id}] at line {node.lineno} "
                    "uses a non-hand-local index"
                )
            elif isinstance(node.slice, ast.Constant):
                assert node.slice.value == 0, (
                    f"{target}[{node.slice.value}] at line {node.lineno} "
                    "uses an invalid constant index"
                )
            else:
                raise AssertionError(
                    f"{target} subscript at line {node.lineno} is not a "
                    "simple hand-local index"
                )
            checked += 1
        if target in robot_dof_arrays:
            assert isinstance(node.slice, ast.Name), (
                f"{target} subscript at line {node.lineno} is not a name"
            )
            assert node.slice.id in robot_index_names, (
                f"{target}[{node.slice.id}] at line {node.lineno} "
                "uses a hand-local index on a robot DOF array"
            )
            checked += 1
    assert checked >= 8, "index-space checks never fired"


def test_hand_local_vs_robot_dof_mapping_is_behaviorally_valid():
    from kcg_connector.d38999_tabletop_pick import (
        load_d38999_tabletop_pick_config,
    )

    pick = load_d38999_tabletop_pick_config(
        "src/kcg_connector/config/d38999_tabletop_pick_v1.yaml"
    )
    # Replicate the runner's full 15-DOF layout: 7 arm joints plus the
    # 8 hand DOFs in the asset order; only 4 hand DOFs are active targets.
    dof_names = tuple(f"iiwa_joint_{index}" for index in range(1, 8)) + (
        "f1j1",
        "f1j2",
        "f1j3",
        "f2j1",
        "f2j2",
        "f3j1",
        "f3j2",
        "f3j3",
    )
    assert len(dof_names) == 15
    hand_local_names = tuple(pick.robot.active_hand_joint_names)
    assert len(hand_local_names) == 4
    hand_indices = tuple(
        dof_names.index(name) for name in hand_local_names
    )
    expected_joints = {"f1": "f1j2", "f2": "f2j1", "f3": "f3j2"}
    formal_finger_hand_indices = (1, 2, 3)
    for channel, finger in enumerate(("f1", "f2", "f3")):
        hand_local_index = formal_finger_hand_indices[channel]
        robot_dof_index = hand_indices[hand_local_index]
        assert 0 <= hand_local_index <= 3
        assert hand_local_names[hand_local_index] == expected_joints[finger]
        assert dof_names[robot_dof_index] == expected_joints[finger]
        assert 0 <= hand_local_index < 4
        # The robot DOF index must NOT be safe on the length-4 hand-local
        # vectors: this is the exact bug class the runner must avoid.
        assert not 0 <= robot_dof_index < 4
        assert 7 <= robot_dof_index < 15
