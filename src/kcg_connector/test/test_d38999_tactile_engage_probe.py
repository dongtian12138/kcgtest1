"""CPU-only tests for the disabled tactile engage experiment contract."""

import ast
from dataclasses import replace
import inspect
import math
from pathlib import Path

import pytest
import yaml

from kcg_connector.d38999_tactile_engage_probe import (
    CENTERED_NO_LIP_ENTRY_MODE,
    CPU_VALIDATION_NODES,
    DEFAULT_CONFIG_PATH,
    EngageObservation,
    EngageState,
    EntryConfirmationEvidence,
    GPU_VALIDATION_NODES,
    LEGACY_CONFIG_PATH,
    LEGACY_LIP_BREAKTHROUGH_MODE,
    LEGACY_SCHEMA_VERSION,
    SCHEMA_VERSION,
    contact_candidate,
    contact_release_candidate,
    decide_engage_transition,
    entry_confirmation_candidate,
    lip_contact_entry_confirmation_candidate,
    load_tactile_engage_contract,
    moment_guided_center_step,
    spiral_offset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "kcg_connector/d38999_tactile_engage_probe.py"
)


def _observation(**updates):
    values = {
        "sample_age_s": 1.0 / 240.0,
        "axial_force_n": 0.05,
        "lateral_force_xy_n": (0.01, -0.01),
        "bending_torque_xy_nm": (0.0, -0.010),
        "tightening_torque_nm": 0.001,
        "finger_base_torques_nm": (0.3, 0.3, 0.3),
        "estimated_gap_m": 0.011,
        "search_offset_xy_m": (0.0, 0.0),
        "contact_attempts": 0,
        "elapsed_search_s": 0.0,
    }
    values.update(updates)
    return EngageObservation(**values)


def _contract():
    return load_tactile_engage_contract(repository=PROJECT_ROOT)


def _entry_evidence(contract, **updates):
    count = contract.entry_confirmation.required_consecutive_samples
    command_gaps = tuple(
        0.00949 - index * (0.00004 / (count - 1))
        for index in range(count)
    )
    measured_gaps = tuple(
        value + 0.000001 for value in command_gaps
    )
    observations = tuple(
        _observation(
            axial_force_n=0.05,
            lateral_force_xy_n=(0.01, -0.01),
            bending_torque_xy_nm=(0.003, 0.0),
            tightening_torque_nm=0.001,
            estimated_gap_m=gap,
        )
        for gap in measured_gaps
    )
    values = {
        "mode": CENTERED_NO_LIP_ENTRY_MODE,
        "registered_preentry_command_fk_gap_m": 0.0100,
        "registered_preentry_measured_gap_m": 0.0100,
        "tick_indices": tuple(range(100, 100 + count)),
        "command_fk_gap_samples_m": command_gaps,
        "measured_gap_samples_m": measured_gaps,
        "loose_fixed_contact_records": (0,) * count,
        "observations": observations,
        "preinsert_capture_id": "capture-001",
        "current_capture_id": "capture-001",
        "upstream_trial_id": "trial-001",
        "current_trial_id": "trial-001",
        "task_frame_id": "connector_task_frame",
        "task_rotation_world": (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        "command_gap_source": (
            "commanded_fixed_q7_fk_against_registered_fixed_z"
        ),
        "measured_gap_source": (
            "measured_tcp_prim_against_registered_fixed_z"
        ),
        "runner_source_sha256": "a" * 64,
        "engage_config_sha256": contract.config_sha256,
        "preinsert_plan_sha256": "b" * 64,
        "registered_pose_sha256": "c" * 64,
        "same_world_and_capture_id": True,
        "object_pose_writes_after_start": 0,
        "truth_used_for_entry_control": False,
    }
    values.update(updates)
    return EntryConfirmationEvidence(**values)


def _entry_candidate(evidence, contract, **expected_updates):
    expected = {
        "expected_runner_source_sha256": "a" * 64,
        "expected_preinsert_plan_sha256": "b" * 64,
        "expected_registered_pose_sha256": "c" * 64,
    }
    expected.update(expected_updates)
    return entry_confirmation_candidate(evidence, contract, **expected)


def test_contract_is_pure_disabled_hash_bound_and_non_safety():
    module_source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(module_source)
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    assert roots.isdisjoint(
        {"isaacsim", "omni", "pxr", "rclpy", "torch", "cv2", "open3d"}
    )

    contract = _contract()
    assert contract.schema_version == SCHEMA_VERSION
    assert contract.enabled_by_default is False
    assert contract.status.endswith("gpu_contact_characterization_required")
    assert contract.entry_confirmation.default_mode == (
        CENTERED_NO_LIP_ENTRY_MODE
    )
    assert contract.entry_confirmation.first_contact_gap_required is False
    assert contract.entry_confirmation.required_consecutive_samples == 24
    assert contract.boundaries[
        "entry_confirmation_runtime_integrated"
    ] is False
    assert contract.boundaries[
        "bare_entry_confirmed_proves_entry_evidence"
    ] is False
    assert "future runtime must construct fresh" in module_source
    assert "literal ``True`` is not evidence" in module_source
    assert "caller-supplied literal ``True``" in module_source
    assert set(contract.states) == set(EngageState)
    assert contract.cpu_nodes == CPU_VALIDATION_NODES
    assert contract.gpu_nodes == GPU_VALIDATION_NODES
    assert contract.abort.calibrated_hardware_safety_limit is False
    assert contract.boundaries["virtual_ft_is_calibrated_safety_gate"] is False
    assert contract.boundaries["gpu_or_physx_validated"] is False
    assert contract.boundaries["production_control_authorized"] is False
    assert contract.boundaries["real_connector_assembly_claimed"] is False
    assert contract.boundaries[
        "foundationpose_required_for_this_proxy_probe"
    ] is False
    assert contract.proxy_boundaries["entry_chamfer_modeled"] is False
    assert contract.proxy_boundaries[
        "filtered_proxy_collision_pair_count"
    ] == 500
    assert all(path.is_file() for path in contract.input_paths.values())


def test_visual_error_is_bounded_only_for_observed_trial_not_provider_bound():
    contract = _contract()
    assert (
        contract.eligibility.observed_fixed_xy_error_m
        < contract.eligibility.search_radius_m
    )
    assert (
        contract.eligibility.search_radius_m
        < contract.eligibility.provider_xy_error_bound_m
    )
    assert contract.eligibility.provider_error_bound_fully_covered is False
    assert contract.eligibility.axial_progress_source == (
        "measured_robot_fk_against_registered_fixed_z"
    )


def test_moment_contact_math_proposes_opposite_radial_correction():
    contract = _contract()
    # F=(0,0,+1) at r=(+10 mm,0,0) gives M=(0,-0.01,0).
    step = moment_guided_center_step((0.0, -0.010), 1.0, contract.motion)
    assert step == pytest.approx((-0.0002, 0.0), abs=1.0e-12)
    assert math.hypot(*step) == pytest.approx(
        contract.motion.moment_guided_xy_step_m
    )
    with pytest.raises(ValueError, match="too small"):
        moment_guided_center_step((0.0, -0.010), 0.0, contract.motion)
    with pytest.raises(ValueError, match="implausible"):
        moment_guided_center_step((0.0, -0.100), 1.0, contract.motion)


def test_contact_hysteresis_uses_local_wrench():
    contract = _contract()
    contact = _observation(
        axial_force_n=0.26,
        bending_torque_xy_nm=(0.0, 0.0),
    )
    assert contact_candidate(contact, contract) is True
    assert contact_release_candidate(contact, contract) is False

    bending_contact = _observation(
        axial_force_n=0.0,
        bending_torque_xy_nm=(0.0081, 0.0),
    )
    assert contact_candidate(bending_contact, contract) is True

    released = _observation(
        axial_force_n=0.05,
        bending_torque_xy_nm=(0.003, 0.0),
        estimated_gap_m=0.0094,
    )
    assert contact_release_candidate(released, contract) is True


def test_centered_no_lip_entry_requires_exact_contact_free_progress_window():
    contract = _contract()
    evidence = _entry_evidence(contract)
    assert _entry_candidate(evidence, contract) is True
    signature = inspect.signature(entry_confirmation_candidate)
    assert "first_contact_gap_m" not in signature.parameters
    assert tuple(signature.parameters) == (
        "evidence",
        "contract",
        "expected_runner_source_sha256",
        "expected_preinsert_plan_sha256",
        "expected_registered_pose_sha256",
    )
    for name in tuple(signature.parameters)[2:]:
        assert signature.parameters[name].kind.name == "KEYWORD_ONLY"


@pytest.mark.parametrize(
    "mutation",
    (
        {"registered_preentry_command_fk_gap_m": 0.00999},
        {"registered_preentry_measured_gap_m": 0.00999},
        pytest.param(
            {"registered_preentry_command_fk_gap_m": 10**10000},
            id="huge-preentry-gap",
        ),
        {"command_fk_gap_samples_m": (0.00951,) + (0.00949,) * 23},
        pytest.param(
            {
                "measured_gap_samples_m":
                (10**10000,) + (0.00949,) * 23
            },
            id="huge-window-gap",
        ),
        {"tick_indices": tuple(range(100, 123)) + (124,)},
        {"tick_indices": (-1,) + tuple(range(23))},
        {"loose_fixed_contact_records": (0,) * 12 + (1,) + (0,) * 11},
        {"preinsert_capture_id": "capture-upstream"},
        {"current_capture_id": "capture-current"},
        {"upstream_trial_id": "trial-upstream"},
        {"current_trial_id": "trial-current"},
        {"task_frame_id": "world"},
        {
            "task_rotation_world": (
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, -1.0),
            )
        },
        {"task_rotation_world": None},
        {"task_rotation_world": ((1.0, 0.0), (0.0, 1.0))},
        {"task_rotation_world": "not-a-rotation"},
        {
            "task_rotation_world": (
                (1.0, 0.0, 0.0),
                (0.0, float("nan"), 0.0),
                (0.0, 0.0, 1.0),
            )
        },
        {"engage_config_sha256": "d" * 64},
        {"preinsert_plan_sha256": "0" * 64},
        pytest.param(
            {"runner_source_sha256": 10**10000},
            id="huge-evidence-hash",
        ),
        {"same_world_and_capture_id": False},
        {"same_world_and_capture_id": 1},
        {"object_pose_writes_after_start": 1},
        {"truth_used_for_entry_control": True},
        {"truth_used_for_entry_control": 0},
        {"observations": [object()] * 24},
        {"observations": (object(),) * 24},
    ),
)
def test_centered_no_lip_entry_fails_closed_on_gate_mutation(mutation):
    contract = _contract()
    assert _entry_candidate(
        _entry_evidence(contract, **mutation), contract
    ) is False


@pytest.mark.parametrize(
    ("expected_name", "value"),
    (
        ("expected_runner_source_sha256", "d" * 64),
        ("expected_preinsert_plan_sha256", "d" * 64),
        ("expected_registered_pose_sha256", "d" * 64),
        ("expected_runner_source_sha256", "not-a-sha"),
        ("expected_preinsert_plan_sha256", None),
        pytest.param(
            "expected_registered_pose_sha256",
            10**10000,
            id="huge-expected-hash",
        ),
    ),
)
def test_centered_entry_requires_exact_expected_provenance(
    expected_name, value
):
    contract = _contract()
    assert _entry_candidate(
        _entry_evidence(contract),
        contract,
        **{expected_name: value},
    ) is False


def test_centered_entry_rejects_non_evidence_without_exception():
    contract = _contract()
    assert _entry_candidate(None, contract) is False


def test_bare_entry_true_is_only_a_state_machine_test_seam():
    contract = _contract()
    entered = _observation(estimated_gap_m=0.0094)
    decision = decide_engage_transition(
        EngageState.ENTRY_CONFIRM,
        entered,
        contract,
        entry_confirmed=True,
    )
    assert decision.next_state is EngageState.COMPLIANT_INSERT
    assert contract.boundaries[
        "entry_confirmation_runtime_integrated"
    ] is False
    assert contract.boundaries[
        "bare_entry_confirmed_proves_entry_evidence"
    ] is False
    assert "literal ``True``" in inspect.getdoc(decide_engage_transition)


def test_centered_no_lip_entry_rejects_loaded_or_mismatched_tick():
    contract = _contract()
    evidence = _entry_evidence(contract)
    observations = list(evidence.observations)
    observations[12] = replace(observations[12], axial_force_n=0.11)
    assert _entry_candidate(
        replace(evidence, observations=tuple(observations)), contract
    ) is False

    observations = list(evidence.observations)
    observations[12] = replace(
        observations[12], estimated_gap_m=0.0090
    )
    assert _entry_candidate(
        replace(evidence, observations=tuple(observations)), contract
    ) is False


@pytest.mark.parametrize(
    ("field", "malformed"),
    (
        ("finite", 1),
        ("three_finger_body_contact", 1),
        ("forbidden_contact", 0),
        ("lateral_force_xy_n", None),
        pytest.param(
            "lateral_force_xy_n",
            (10**10000, 0.0),
            id="huge-ft-value",
        ),
        ("bending_torque_xy_nm", None),
        ("finger_base_torques_nm", None),
        ("search_offset_xy_m", None),
        ("axial_force_n", "bad"),
        ("estimated_gap_m", object()),
    ),
)
def test_centered_entry_malformed_observation_fails_false(field, malformed):
    contract = _contract()
    evidence = _entry_evidence(contract)
    observations = list(evidence.observations)
    observations[7] = replace(observations[7], **{field: malformed})
    assert _entry_candidate(
        replace(evidence, observations=tuple(observations)), contract
    ) is False

    # The public state seam must also convert malformed data into an abort
    # decision rather than leak a TypeError from _abort_reason.
    decision = decide_engage_transition(
        EngageState.GUARDED_APPROACH,
        observations[7],
        contract,
    )
    assert decision.next_state is EngageState.ABORT_RETRACT
    assert decision.reason in {
        "malformed_observation",
        "nonfinite_observation",
    }


def test_legacy_first_contact_path_is_explicit_and_v1_remains_loadable():
    contract = _contract()
    released = _observation(
        axial_force_n=0.05,
        bending_torque_xy_nm=(0.003, 0.0),
        estimated_gap_m=0.0094,
    )
    assert lip_contact_entry_confirmation_candidate(
        0.0100,
        released,
        contract,
        mode=LEGACY_LIP_BREAKTHROUGH_MODE,
    ) is True
    with pytest.raises(ValueError, match="must be explicit"):
        lip_contact_entry_confirmation_candidate(
            0.0100,
            released,
            contract,
            mode=CENTERED_NO_LIP_ENTRY_MODE,
        )

    legacy = load_tactile_engage_contract(
        LEGACY_CONFIG_PATH, repository=PROJECT_ROOT
    )
    assert legacy.schema_version == LEGACY_SCHEMA_VERSION
    assert legacy.entry_confirmation.default_mode == (
        LEGACY_LIP_BREAKTHROUGH_MODE
    )
    assert _entry_candidate(_entry_evidence(contract), legacy) is False


def test_spiral_is_endpoint_bounded_and_approximately_arc_spaced():
    contract = _contract()
    points = [
        spiral_offset(index, contract.motion)
        for index in range(contract.motion.maximum_contact_attempts + 1)
    ]
    radii = [math.hypot(*point) for point in points]
    distances = [
        math.dist(first, second)
        for first, second in zip(points, points[1:])
    ]
    assert points[0] == (0.0, 0.0)
    assert max(radii) <= contract.motion.maximum_search_radius_m + 1e-12
    assert radii[-1] == pytest.approx(
        contract.motion.maximum_search_radius_m, abs=1e-12
    )
    assert max(distances) <= contract.motion.spiral_arc_step_m * 1.001
    assert min(distances[:-1]) > contract.motion.spiral_arc_step_m * 0.49
    with pytest.raises(ValueError, match="non-negative"):
        spiral_offset(-1, contract.motion)


def test_state_machine_happy_path_requires_contact_calibration_and_truth_audit():
    contract = _contract()
    observation = _observation()

    decision = decide_engage_transition(
        EngageState.WAIT_PREINSERT_PASS,
        observation,
        contract,
        prerequisite_passed=False,
    )
    assert decision.next_state is EngageState.WAIT_PREINSERT_PASS
    decision = decide_engage_transition(
        EngageState.WAIT_PREINSERT_PASS, observation, contract
    )
    assert decision.next_state is EngageState.LOCAL_REFERENCE
    decision = decide_engage_transition(
        EngageState.LOCAL_REFERENCE, observation, contract
    )
    assert decision.next_state is EngageState.GUARDED_APPROACH

    decision = decide_engage_transition(
        EngageState.GUARDED_APPROACH,
        observation,
        contract,
        contact_debounced=True,
    )
    assert decision.next_state is EngageState.RETRACT_UNLOAD
    decision = decide_engage_transition(
        EngageState.RETRACT_UNLOAD,
        observation,
        contract,
        contact_released=True,
    )
    assert decision.next_state is EngageState.CENTER_CORRECTION

    decision = decide_engage_transition(
        EngageState.CENTER_CORRECTION,
        observation,
        contract,
        moment_direction_calibrated=False,
    )
    assert decision.next_state is EngageState.SPIRAL_FALLBACK
    decision = decide_engage_transition(
        EngageState.CENTER_CORRECTION,
        replace(observation, axial_force_n=1.0),
        contract,
        moment_direction_calibrated=True,
    )
    assert decision.next_state is EngageState.GUARDED_APPROACH
    assert decision.command_delta_xy_m == pytest.approx((-0.0002, 0.0))

    entered = replace(observation, estimated_gap_m=0.0094)
    decision = decide_engage_transition(
        EngageState.GUARDED_APPROACH, entered, contract
    )
    assert decision.next_state is EngageState.ENTRY_CONFIRM
    decision = decide_engage_transition(
        EngageState.ENTRY_CONFIRM,
        entered,
        contract,
        entry_confirmed=True,
    )
    assert decision.next_state is EngageState.COMPLIANT_INSERT
    engaged = replace(observation, estimated_gap_m=0.003)
    decision = decide_engage_transition(
        EngageState.COMPLIANT_INSERT, engaged, contract
    )
    assert decision.next_state is EngageState.ENGAGE_HOLD
    decision = decide_engage_transition(
        EngageState.ENGAGE_HOLD,
        engaged,
        contract,
        engage_hold_complete=True,
    )
    assert decision.next_state is EngageState.SIM_TRUTH_AUDIT
    decision = decide_engage_transition(
        EngageState.SIM_TRUTH_AUDIT,
        engaged,
        contract,
        sim_truth_audit_passed=False,
    )
    assert decision.next_state is EngageState.ABORT_RETRACT
    decision = decide_engage_transition(
        EngageState.SIM_TRUTH_AUDIT,
        engaged,
        contract,
        sim_truth_audit_passed=True,
    )
    assert decision.next_state is EngageState.READY_FOR_EXISTING_PROXY_TWIST


@pytest.mark.parametrize(
    ("update", "reason"),
    (
        ({"sample_age_s": 0.1}, "stale_wrench"),
        ({"axial_force_n": 5.1}, "experimental_axial_force_ceiling"),
        ({"lateral_force_xy_n": (2.1, 0.0)}, "experimental_lateral_force_ceiling"),
        ({"bending_torque_xy_nm": (0.181, 0.0)}, "experimental_bending_torque_ceiling"),
        ({"tightening_torque_nm": 0.051}, "experimental_tightening_torque_ceiling"),
        ({"finger_base_torques_nm": (2.01, 0.2, 0.2)}, "finger_base_torque_hard_stop"),
        ({"three_finger_body_contact": False}, "grasp_contact_lost"),
        ({"forbidden_contact": True}, "forbidden_contact"),
        ({"estimated_gap_m": -0.001}, "invalid_observation_range"),
        ({"contact_attempts": -1}, "malformed_observation"),
    ),
)
def test_every_experimental_ceiling_aborts_before_more_search(update, reason):
    contract = _contract()
    decision = decide_engage_transition(
        EngageState.GUARDED_APPROACH,
        _observation(**update),
        contract,
    )
    assert decision.next_state is EngageState.ABORT_RETRACT
    assert decision.reason == reason
    assert decision.requires_abort_retract is True


def test_abort_retract_is_bounded_and_latches_terminal_abort():
    contract = _contract()
    observation = _observation()
    decision = decide_engage_transition(
        EngageState.ABORT_RETRACT, observation, contract
    )
    assert decision.next_state is EngageState.TERMINAL_ABORT
    assert decision.command_delta_z_m == pytest.approx(
        contract.motion.unload_retract_distance_m
    )
    terminal = decide_engage_transition(
        EngageState.TERMINAL_ABORT, observation, contract
    )
    assert terminal.next_state is EngageState.TERMINAL_ABORT


def test_public_controller_seam_has_no_object_or_truth_pose_input():
    signature = inspect.signature(decide_engage_transition)
    forbidden = {
        "body_world_pose",
        "fixed_world_pose",
        "truth_xy",
        "truth_alignment",
    }
    assert forbidden.isdisjoint(signature.parameters)
    assert "sim_truth_audit_passed" in signature.parameters
    assert signature.parameters["sim_truth_audit_passed"].kind.name == (
        "KEYWORD_ONLY"
    )


def test_scope_and_hash_mutations_fail_closed(tmp_path):
    base = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))

    mutated = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    mutated["enabled_by_default"] = True
    path = tmp_path / "enabled.yaml"
    path.write_text(yaml.safe_dump(mutated), encoding="utf-8")
    with pytest.raises(ValueError, match="disabled"):
        load_tactile_engage_contract(path, repository=PROJECT_ROOT)

    mutated = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    mutated["boundaries"]["virtual_ft_is_calibrated_safety_gate"] = True
    path = tmp_path / "safety.yaml"
    path.write_text(yaml.safe_dump(mutated), encoding="utf-8")
    with pytest.raises(ValueError, match="overclaims"):
        load_tactile_engage_contract(path, repository=PROJECT_ROOT)

    mutated = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    mutated["eligibility"]["provider_error_bound_fully_covered"] = True
    path = tmp_path / "coverage.yaml"
    path.write_text(yaml.safe_dump(mutated), encoding="utf-8")
    with pytest.raises(ValueError, match="overclaims coverage"):
        load_tactile_engage_contract(path, repository=PROJECT_ROOT)

    mutated = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    mutated["entry_confirmation_policy"]["default_mode"] = (
        LEGACY_LIP_BREAKTHROUGH_MODE
    )
    path = tmp_path / "legacy_default.yaml"
    path.write_text(yaml.safe_dump(mutated), encoding="utf-8")
    with pytest.raises(ValueError, match="entry confirmation policy"):
        load_tactile_engage_contract(path, repository=PROJECT_ROOT)

    mutated = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    mutated["entry_confirmation_policy"]["first_contact_gap_required"] = True
    path = tmp_path / "first_contact_required.yaml"
    path.write_text(yaml.safe_dump(mutated), encoding="utf-8")
    with pytest.raises(ValueError, match="entry confirmation policy"):
        load_tactile_engage_contract(path, repository=PROJECT_ROOT)

    mutated = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    mutated["boundaries"]["entry_confirmation_runtime_integrated"] = True
    path = tmp_path / "runtime_integrated.yaml"
    path.write_text(yaml.safe_dump(mutated), encoding="utf-8")
    with pytest.raises(ValueError, match="integration boundary|overclaims"):
        load_tactile_engage_contract(path, repository=PROJECT_ROOT)

    mutated = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    mutated["entry_confirmation_policy"]["required_task_rotation_world"] = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
    ]
    path = tmp_path / "reflected_task_frame.yaml"
    path.write_text(yaml.safe_dump(mutated), encoding="utf-8")
    with pytest.raises(ValueError, match="right-handed"):
        load_tactile_engage_contract(path, repository=PROJECT_ROOT)

    base["inputs"]["virtual_wrist_ft_monitor"]["sha256"] = "0" * 64
    path = tmp_path / "hash.yaml"
    path.write_text(yaml.safe_dump(base), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_tactile_engage_contract(path, repository=PROJECT_ROOT)
