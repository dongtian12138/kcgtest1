'''Static firewall tests for the formal grasp control path in the runner.

The formal control path must never read object pose, object velocity,
contact reports, collider identity, or contact normals.  Truth reads are
allowed only for post-motion evaluation, and inside the per-step loop they
must sit behind the ``formal_grasp`` flag.

Terminal evaluator exception (frozen spec): after a formal grasp fail-closed
exception or formal lift gate makes the controller terminal, a log-only truth
snapshot is allowed inside ``capture_terminal_snapshot``.  Its result may only
be assigned into the report ``metrics`` mapping and must never feed commands,
recovery planning or PASS.  The tests below enforce both the confinement of
the truth reads and the confinement of the snapshot sinks.
'''

import ast
from pathlib import Path

RUNNER = (
    Path(__file__).resolve().parents[1]
    / "isaac"
    / "d38999_tabletop_pick_smoke.py"
)

TRUTH_CALL_NAMES = {
    "get_full_contact_report",
    "get_world_pose",
    "get_linear_velocity",
    "get_angular_velocity",
    "contact_snapshot",
    "intToSdfPath",
}

TRUTH_TOKENS = TRUTH_CALL_NAMES | {"collider", "contact_normal"}

SNAPSHOT_KEYS = (
    "formal_grasp_failure_evaluator_snapshot",
    "formal_terminal_evaluator_snapshot",
    "formal_recovery_end_evaluator_snapshot",
    "empty_hand_diagnostic_stage_terminal_snapshot",
    "empty_hand_diagnostic_endpoint_snapshot",
)


def _source_lines():
    return RUNNER.read_text(encoding="utf-8").splitlines()


def _function(tree, name):
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1, f"expected exactly one {name} definition"
    return matches[0]


def _call_name(call):
    function = call.func
    if isinstance(function, ast.Attribute):
        return function.attr
    if isinstance(function, ast.Name):
        return function.id
    return None


def _parents(tree):
    mapping = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            mapping[child] = parent
    return mapping


def _guarded_by_formal_flag(call, parents):
    current = parents.get(call)
    while current is not None:
        if isinstance(current, ast.FunctionDef):
            return False
        if isinstance(current, ast.If) and "formal_grasp" in ast.unparse(
            current.test
        ):
            return True
        current = parents.get(current)
    return False


def test_per_step_loop_reads_truth_only_behind_formal_flag():
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    parents = _parents(tree)
    function = _function(tree, "observe_and_step")
    for call in ast.walk(function):
        if not isinstance(call, ast.Call):
            continue
        name = _call_name(call)
        if name in TRUTH_CALL_NAMES:
            assert _guarded_by_formal_flag(call, parents), (
                f"unguarded truth read {name!r} inside observe_and_step"
            )


def test_effort_sampler_never_reads_truth():
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    function = _function(tree, "sample_post_tare_efforts")
    for call in ast.walk(function):
        if isinstance(call, ast.Call):
            assert _call_name(call) not in TRUTH_CALL_NAMES


def _region(lines, start_marker, end_marker):
    start = next(
        index for index, line in enumerate(lines) if start_marker in line
    )
    end = next(
        index for index, line in enumerate(lines) if end_marker in line
    )
    assert start < end, f"{start_marker} must precede {end_marker}"
    return "\n".join(lines[start:end])


def test_preload_reference_window_reads_no_truth():
    # The preload window produces the grasped-payload wrist reference, so it
    # is control code.  The postclosure pose/contact reads that follow it are
    # evaluation-only and sit after this region.
    region = _region(
        _source_lines(),
        'phase = "physical_grip_preload"',
        "postclosure_body_position, postclosure_body_orientation = (",
    )
    for token in TRUTH_TOKENS:
        assert token not in region, f"preload control window reads {token}"


def test_lift_and_recovery_region_reads_no_truth_outside_snapshot_helper():
    # Truth reads inside the lift/recovery window are permitted ONLY inside
    # the log-only terminal snapshot helper.  Everything else in this region
    # (staged loop, zero-lift hold loop, recovery closure body) must stay
    # sensor/robot-state only.
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    lines = _source_lines()
    start_index = next(
        i
        for i, line in enumerate(lines)
        if "physical_grip_lift" in line and "phase = " in line
    )
    end_index = next(
        i
        for i, line in enumerate(lines)
        if 'phase = "unsupported_final_hold"' in line
    )
    region = "\n".join(lines[start_index:end_index])
    for token in TRUTH_TOKENS:
        assert token not in region, (
            f"lift/recovery control window reads {token} outside the "
            "terminal snapshot helper"
        )
    assert "plan_recovery_return" in region
    assert "plan_recovery_open" in region


def test_terminal_snapshot_helper_truth_reads_are_log_only_confined():
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    snapshot_function = _function(tree, "capture_terminal_snapshot")
    calls = {
        _call_name(call)
        for call in ast.walk(snapshot_function)
        if isinstance(call, ast.Call)
    }
    allowed = {"get_world_pose", "_world_pose", "contact_snapshot",
               "_gf_quaternion_tuple", "build_terminal_snapshot", "np",
               "float"}
    unexpected = (calls & TRUTH_CALL_NAMES) - allowed
    assert not unexpected, f"unexpected truth call {unexpected} in snapshot"


def test_terminal_snapshot_results_sink_only_into_metrics():
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    found = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        value = node.value
        if not isinstance(value, ast.Call):
            continue
        if _call_name(value) != "capture_terminal_snapshot":
            continue
        for target in node.targets:
            found += 1
            assert isinstance(target, ast.Subscript), (
                "capture_terminal_snapshot result must only be assigned "
                "into the report metrics mapping"
            )
            assert isinstance(target.value, ast.Name)
            assert target.value.id == "metrics"
            assert isinstance(target.slice, ast.Constant)
            assert target.slice.value in SNAPSHOT_KEYS
    assert found == 5, "all terminal snapshots must be assigned into metrics"


def test_formal_grasp_failure_snapshot_is_exception_only_and_log_only():
    source = RUNNER.read_text(encoding="utf-8")
    except_index = source.index("except BaseException as exception:")
    finally_index = source.index("    finally:", except_index)
    failure_region = source[except_index:finally_index]
    assert 'metrics["formal_grasp_failure_evaluator_snapshot"]' in failure_region
    assert '"formal_grasp_fail_closed_exception"' in failure_region
    assert "robot.apply_action" not in failure_region
    assert "world.step" not in failure_region
    assert "passed = True" not in failure_region


def test_snapshot_evidence_precedes_recovery_planning_and_fail_closed_raise():
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    recovery = _function(tree, "run_formal_failure_recovery")
    snapshot_a = None
    snapshot_b = None
    recovery_plan = None
    raise_line = None
    for node in ast.walk(recovery):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "metrics"
                    and isinstance(target.slice, ast.Constant)
                ):
                    if target.slice.value == "formal_terminal_evaluator_snapshot":
                        snapshot_a = node.lineno
                    if target.slice.value == "formal_recovery_end_evaluator_snapshot":
                        snapshot_b = node.lineno
        if isinstance(node, ast.Call) and _call_name(node) == "plan_recovery_return":
            if recovery_plan is None:
                recovery_plan = node.lineno
        if isinstance(node, ast.Raise):
            raise_line = node.lineno
    assert snapshot_a is not None
    assert snapshot_b is not None
    assert recovery_plan is not None
    assert raise_line is not None
    assert snapshot_a < recovery_plan, (
        "terminal snapshot must precede recovery planning"
    )
    assert recovery_plan < snapshot_b, (
        "recovery-end snapshot must follow recovery planning"
    )
    assert snapshot_b < raise_line, (
        "recovery-end snapshot must precede the fail-closed raise"
    )


def test_pre_lift_evidence_is_recorded_before_lift_starts():
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    prelift = None
    lift_line = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "metrics"
                    and isinstance(target.slice, ast.Constant)
                    and target.slice.value == "pre_lift_grasp_controller_evidence"
                ):
                    prelift = node.lineno
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Constant)
            and node.value.value == "physical_grip_lift"
        ):
            lift_line = node.lineno
    assert prelift is not None
    assert lift_line is not None
    assert prelift < lift_line, (
        "pre-lift grasp controller evidence must be written before lift"
    )


def test_h4_pre_lift_centering_is_sensor_robot_only_and_xy_bounded():
    source = RUNNER.read_text(encoding="utf-8")
    start = source.index(
        "centering = physical_grasp.pre_lift_centering"
    )
    end = source.index(
        "if arguments.empty_hand_first_stage_diagnostic:", start
    )
    region = source[start:end]
    for token in TRUTH_TOKENS:
        assert token not in region, f"H4 centering reads {token}"
    assert "formal_lift_monitor.update(" in region
    assert "solve_bounded_xy_centering(" in region
    assert "corrected_position[:2] += correction_xy" in region
    assert "corrected_position[2]" not in region
    assert '"z_target_unchanged": True' in region
    assert '"sensor_origin_hard_gate_unchanged": True' in region
    assert '"object_truth_used": False' in region
    assert '"contact_truth_used": False' in region
    assert "maximum_entry_moment_score_nm" in region
    assert "maximum_entry_load_imbalance" in region
    assert "transform_wrench_to_task(" in source
    assert "inverse_wrench_transform(" in source


def test_h5_realized_state_rebase_is_robot_sensor_only_and_hard_gated():
    source = RUNNER.read_text(encoding="utf-8")
    start = source.index(
        "rebase = physical_grasp.pre_lift_realized_state_rebase"
    )
    end = source.index(
        "physical_grasp.pre_lift_centering.enabled", start
    )
    region = source[start:end]
    for token in TRUTH_TOKENS:
        assert token not in region, f"H5 realized-state rebase reads {token}"
    assert "positions[arm_indices]" in region
    assert "validate_realized_state_rebase(" in region
    assert "current_arm_target = rebase_realized_arm.copy()" in region
    assert "formal_lift_monitor.update(" in region
    assert "formal_latest_wrist_canonical" in region
    assert '"sensor_origin_hard_gate_unchanged": True' in region
    assert '"robot_joint_state_only": True' in region
    assert '"object_truth_used": False' in region
    assert '"contact_truth_used": False' in region
    assert "set_world_pose" not in region


def test_h6_arm_drive_compliance_is_robot_sensor_only_and_bumpless():
    source = RUNNER.read_text(encoding="utf-8")
    start = source.index(
        "compliance = (\n                    physical_grasp.pre_lift_arm_drive_compliance"
    )
    end = source.index(
        "physical_grasp.pre_lift_centering.enabled", start
    )
    region = source[start:end]
    for token in TRUTH_TOKENS:
        assert token not in region, f"H6 arm-drive compliance reads {token}"
    assert "positions[arm_indices]" in region
    assert "capture_position_preload_nm(" in region
    assert "derive_bumpless_drive_step(" in region
    assert "compliant_path_drive_target(" in region
    assert "controller.set_gains(" in region
    assert "controller.get_gains()" in region
    assert "formal_lift_monitor.update(" in region
    assert "formal_latest_wrist_canonical" in region
    assert '"payload_reference_rebased": False' in region
    assert '"sensor_origin_hard_gate_unchanged": True' in region
    assert '"robot_joint_state_only": True' in region
    assert '"object_truth_used": False' in region
    assert '"contact_truth_used": False' in region
    assert "set_world_pose" not in region


def test_h6_window_summary_adapts_numpy_vectors_to_frozen_moment_interface():
    source = RUNNER.read_text(encoding="utf-8")
    start = source.index("compliance_entry_moment_scores = [")
    end = source.index(
        "compliance_entry_moment_score = max(", start
    )
    region = source[start:end]
    assert "float(value) for value in sample[3:]" in region
    assert (
        "for value in formal_wrist_payload_reference[3:]" in region
    )
    assert "evaluate_wrist_moment_safety(" in region


def test_task_wrench_transform_cannot_replace_sensor_origin_hard_gate():
    source = RUNNER.read_text(encoding="utf-8")
    helper = _region(
        _source_lines(),
        "def formal_wrist_step_evidence(positions):",
        "def run_formal_failure_recovery():",
    )
    assert "wrist_wrench_grasp_tcp_frame" in helper
    assert "wrist_wrench_connector_task_frame" in helper
    assert "wrench_transform_roundtrip_max_abs_error" in helper
    assert '"task_wrench_control_use": (' in helper
    staged_start = source.index(
        "# Staged mode below keeps the validated lift"
    )
    staged_end = source.index(
        'metrics["formal_lift_stages"] = formal_lift_stage_records',
        staged_start,
    )
    staged = source[staged_start:staged_end]
    assert "formal_lift_monitor.update(" in staged
    assert "formal_latest_wrist_canonical" in staged
    assert "wrist_wrench_connector_task_frame" not in staged


def test_formal_postclosure_contact_report_is_deferred_until_motion_ends():
    source = RUNNER.read_text(encoding="utf-8")
    assert "None if formal_grasp else contact_snapshot()" in source


def test_report_keys_for_baselines_payload_increment_and_recovery_exist():
    source = RUNNER.read_text(encoding="utf-8")
    for key in (
        "formal_home_tare_wrist_baseline",
        "formal_pregrasp_empty_wrist_baseline",
        "formal_payload_wrist_reference",
        "formal_payload_wrist_reference_statistics",
        "pre_lift_grasp_controller_evidence",
        "formal_wrist_wrench_frame_audit",
        "formal_lift_stages",
        "formal_lift_monitor",
        "formal_recovery",
        "formal_terminal_evaluator_snapshot",
        "formal_recovery_end_evaluator_snapshot",
        "wrist_wrench_payload_reference_increment",
        "wrist_wrench_empty_baseline_compensated",
    ):
        assert key in source, f"report key {key!r} is missing from the runner"


def test_original_gate_values_are_unchanged_in_source_and_config():
    # The runner constructs the monitor from the strict config loader and
    # never hard-codes the gate numbers.
    source = RUNNER.read_text(encoding="utf-8")
    assert "GraspStabilityMonitor(" in source
    assert "physical_grasp.stability" in source
    config_path = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "d38999_tabletop_physical_grasp_v1.yaml"
    )
    document = config_path.read_text(encoding="utf-8")
    assert "maximum_wrist_force_n: 8.0" in document
    assert "maximum_wrist_moment_nm: 0.30" in document


def _call_count(tree, name):
    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _call_name(node) == name
    )


def test_realize_randomization_is_called_exactly_once():
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    assert _call_count(tree, "realize_randomization") == 1


def test_runner_consumes_no_global_rng():
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                assert alias.name.split(".")[0] != "random"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id == "np"
                and node.func.attr == "random"
            ):
                raise AssertionError("runner consumes np.random")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in ("random", "randint", "seed")


def _reset_index(source):
    # Match the actual reset statement, not the comments that mention it.
    return source.index("\n        world.reset()")


def test_realized_usd_authoring_precedes_first_world_reset():
    source = RUNNER.read_text(encoding="utf-8")
    authoring_marker = source.index(
        "realized USD authoring block (pre-reset"
    )
    assert authoring_marker < _reset_index(source)


def test_no_usd_authoring_writes_after_first_world_reset():
    source = RUNNER.read_text(encoding="utf-8")
    reset_index = _reset_index(source)
    after_reset = source[reset_index:]
    for token in (
        "AddTranslateOp",
        "AddTransformOp",
        "ClearXformOpOrder",
        "CreateStaticFrictionAttr",
        "CreateDynamicFrictionAttr",
        "GetCenterOfMassAttr().Set",
        "GetMassAttr().Set",
    ):
        assert token not in after_reset, (
            f"post-reset USD authoring write {token!r} found"
        )


def test_zero_lift_mode_never_consumes_lift_speed_scale():
    source = RUNNER.read_text(encoding="utf-8")
    start = source.index("if zero_lift_hold_mode:")
    end = source.index("# Staged mode below keeps the validated lift")
    zero_lift_region = source[start:end]
    assert "lift_speed_scale" not in zero_lift_region


def test_synchronous_stability_uses_final_detector_states():
    source = RUNNER.read_text(encoding="utf-8")
    assert "synchronous_contact_stability(" in source
    # The final states dict is built from the live detectors, not from the
    # historical contact order records.
    assert "for name, detector in synchronous_detectors.items()" in source
    assert "grasp_controller_final_states = {" in source


def test_sequential_step_evidence_records_stiffness_and_transitions():
    source = RUNNER.read_text(encoding="utf-8")
    assert '"finger_stiffness_scale": list(' in source
    assert 'sequential_command.finger_stiffness_scale' in source
    assert '"controller_evidence": dict(' in source
    assert 'sequential_command.evidence' in source


def test_nominal_vs_realized_evaluator_boundary():
    source = RUNNER.read_text(encoding="utf-8")
    # Tracking evaluators compare against the realized target FK.
    assert "closure_tcp_reference" in source
    assert "grasp_tcp_reference" in source
    # The posthoc nominal T_hand_plug must keep the nominal grasp target.
    assert "for value in nominal_grasp_arm" in source
    assert "nominal_world_plug" in source


def test_binding_traversal_reuses_validated_ordinary_prim_identity():
    source = RUNNER.read_text(encoding="utf-8")
    # The current multilayer composition exposes ordinary prims.  The first
    # traversal is count-gated, and binding readback must reuse exactly those
    # validated paths instead of silently running an instance-only pass.
    assert "Usd.TraverseInstanceProxies()" not in source
    assert "expected_plug_collision_counts" in source
    assert 'for prim_path in plug_collision_prims[group]:' in source
    assert "plug_binding_identity" in source
    assert "material_binding_identity" in source
    assert '"plug_collider_count": len(plug_binding_identity)' in source


def test_multilayer_grasp_filter_preserves_physical_nut_shoulders():
    source = RUNNER.read_text(encoding="utf-8")
    assert 'if len(non_grip_nut_paths) != 6:' in source
    assert '"BodyShoulderPositive"' in source
    assert '"BodyShoulderNegative"' in source
    assert '"NutShoulderPositive"' in source
    assert '"NutShoulderNegative"' in source
    assert '"BodyShoulderPositive<->NutShoulderPositive"' in source
    assert '"BodyShoulderNegative<->NutShoulderNegative"' in source
    assert '"default_cross_role_pairs_filtered": True' in source
    assert '"body_non_shoulder_collider_count"' in source
    assert '"body_shoulder_collider_count"' in source
    assert '"non_grip_nut_shoulders<->non_grip_loose_plug"' not in source


def test_formal_step_evidence_has_explicit_same_physics_step_audit():
    source = RUNNER.read_text(encoding="utf-8")
    assert "formal_latest_wrist_global_step = None" in source
    assert "formal_latest_wrist_global_step = int(global_step)" in source
    assert '"sample_synchronization": {' in source
    assert '"finger_root_effort_step": int(global_step)' in source
    assert '"wrist_wrench_step": (' in source
    assert 'formal_latest_wrist_global_step == global_step' in source
    assert '"detector_input_targets_rad": list(' in source
    assert '"finger_q_actual_rad": [' in source
    assert '"finger_qd_actual_rad_s": [' in source
    assert '"detector_observations": {' in source


def test_preexisting_api_checks_precede_schema_construction():
    source = RUNNER.read_text(encoding="utf-8")
    # The table material and body/nut MassAPI are asset contracts: HasAPI
    # checks must precede any Apply/schema-wrapper construction.
    assert (
        "table_material_prim.HasAPI(UsdPhysics.MaterialAPI)"
        in source
    )
    assert "part_prim.HasAPI(UsdPhysics.MassAPI)" in source
    assert "mass_attr.HasAuthoredValueOpinion()" in source
    # No Apply(...) that could mask a missing schema contract.
    assert "UsdPhysics.MaterialAPI.Apply(table_material_prim)" not in source
    assert "UsdPhysics.MassAPI.Apply(part_prim)" not in source


def test_table_and_grip_use_float32_readback_evidence():
    source = RUNNER.read_text(encoding="utf-8")
    assert "float32_readback_evidence(" in source
    for label in (
        "table_static_friction",
        "table_dynamic_friction",
        "grip_static_friction",
        "grip_dynamic_friction",
    ):
        assert label in source
    assert 'record_float32_evidence(' in source
    assert '"table_static_friction"' in source
    assert '"grip_static_friction"' in source
    assert '"grip_dynamic_friction"' in source
    # Attribute types must be checked: scalar Float and the MassAPI
    # centerOfMass schema type point3f (never a loose vec3 type).
    assert '.lower() != "float"' in source
    assert '.lower() != "point3f"' in source
    assert '.lower() != "float3"' not in source


def test_keyed_zero_com_offset_preserves_checkpoint_a_asset_baseline():
    source = RUNNER.read_text(encoding="utf-8")
    assert "per_part_asset_baseline_plus_local_offset" in source
    assert 'f"{name}_asset_baseline_com"' in source
    assert "baseline + offset" in source
    assert '"center_of_mass_authored_in_asset": com_authored' in source
    assert 'not baseline_com_evidence["verified"]' in source


def test_authoring_evidence_is_mounted_before_any_check():
    source = RUNNER.read_text(encoding="utf-8")
    mount_index = source.index(
        'metrics["realized_usd_authoring"] = realized_usd_authoring'
    )
    authoring_branch = source.index(
        "if formal_grasp:",
        source.index("realized USD authoring block"),
    )
    # The formal branch containing the checks starts after the mount; only
    # one mount assignment exists.
    assert mount_index < authoring_branch
    assert source.count(
        'metrics["realized_usd_authoring"] = realized_usd_authoring'
    ) == 1


def test_final_usd_authoring_verified_is_single_and_after_bindings():
    source = RUNNER.read_text(encoding="utf-8")
    assignments = [
        index
        for index in range(len(source))
        if source.startswith(
            'realized_usd_authoring["usd_authoring_verified"] =',
            index,
        )
    ]
    assert len(assignments) == 1
    final_flag_index = assignments[0]
    binding_marker = source.index('"all_bindings_ok": binding_identity_ok')
    assert final_flag_index > binding_marker
    # The final true also requires the attribute readbacks to be verified.
    assert "attribute_readbacks_verified" in source[
        final_flag_index:final_flag_index + 400
    ]


def test_final_verified_fails_closed_before_world_reset():
    source = RUNNER.read_text(encoding="utf-8")
    final_flag = source.index(
        'realized_usd_authoring["usd_authoring_verified"] = bool('
    )
    fail_closed = source.index(
        "realized USD authoring did not fully verify before reset",
        final_flag,
    )
    reset_index = _reset_index(source)
    assert final_flag < fail_closed < reset_index


def test_no_duplicate_dict_literal_keys_in_runner():
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        seen = set()
        for key in node.keys:
            if key is None:
                continue
            rendered = ast.unparse(key)
            assert rendered not in seen, (
                f"duplicate dict literal key {rendered!r} at line "
                f"{node.lineno}"
            )
            seen.add(rendered)


def test_diagnostic_stage_snapshot_precedes_return_and_has_no_loads():
    source = RUNNER.read_text(encoding="utf-8")
    stage_index = source.index(
        "empty_hand_diagnostic_stage_terminal_snapshot",
    )
    return_index = source.index(
        'phase = "empty_hand_diagnostic_return"',
    )
    assert stage_index < return_index, (
        "stage-terminal snapshot must be captured before any return step"
    )
    endpoint_index = source.index(
        "empty_hand_diagnostic_endpoint_snapshot",
    )
    assert endpoint_index > return_index, (
        "endpoint snapshot must be captured after the return"
    )
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        if not (
            isinstance(node.value, ast.Name) and node.value.id == "metrics"
        ):
            continue
        if (
            isinstance(node.slice, ast.Constant)
            and node.slice.value
            in (
                "empty_hand_diagnostic_stage_terminal_snapshot",
                "empty_hand_diagnostic_endpoint_snapshot",
            )
        ):
            assert isinstance(node.ctx, ast.Store), (
                "diagnostic snapshot evidence must never be loaded back "
                "into control"
            )
