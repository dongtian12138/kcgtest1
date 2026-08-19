'''Runner contract tests for the sequential consolidation phase (030/030a).

Static source-order and token checks: the consolidation phase must sit
between the recovery helper and the lift branches, failures must route
through the shared snapshot-first recovery, references may only rebase
after a full clean window with explicit pre-consolidation preservation,
non-sequential modes never enter the phase, the seating/preload spans
keep detector timestamps continuous, and the lift branches stay
unreachable without LIFT_READY.
'''

from __future__ import annotations

from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[3]
RUNNER = (
    REPOSITORY
    / "src/kcg_connector/isaac/d38999_tabletop_pick_smoke.py"
)


def _source() -> str:
    return RUNNER.read_text(encoding="utf-8")


def test_consolidation_phase_precedes_zero_lift_and_staged_branches():
    source = _source()
    consolidation = source.index("physical_grip_consolidation")
    recovery_def = source.index("def run_formal_failure_recovery():")
    zero_lift = source.index("if zero_lift_hold_mode:")
    assert recovery_def < consolidation < zero_lift


def test_consolidation_failure_calls_shared_recovery():
    source = _source()
    consolidation = source.index("physical_grip_consolidation")
    zero_lift = source.index("if zero_lift_hold_mode:")
    region = source[consolidation:zero_lift]
    assert region.count("run_formal_failure_recovery()") >= 2
    assert "consolidation_budget_exhausted" in region
    assert "begin_consolidation_rejected" in region
    assert "consolidation_window_sample_count_mismatch" in region
    assert "consolidation_window_applied_scale_mismatch" in region


def test_rebase_only_after_clean_window_with_full_evidence():
    source = _source()
    consolidation = source.index("physical_grip_consolidation")
    zero_lift = source.index("if zero_lift_hold_mode:")
    region = source[consolidation:zero_lift]
    for token in (
        "pre_consolidation_root_reference_nm",
        "final_root_reference_nm",
        "root_reference_delta_nm",
        "pre_consolidation_wrist_reference",
        "final_wrist_reference",
        "wrist_reference_delta",
        "consolidation_peak_moment_safety_score_nm",
        "final_window_wrist_statistics",
        "targets_frozen_exact",
        "commanded_scale_monotonic",
        "lift_ready_global_step",
        "stable_implies_lift_ready",
    ):
        assert token in region, token
    rebase = region.index("GraspStabilityMonitor(")
    first_recovery_call = region.index("run_formal_failure_recovery()")
    assert first_recovery_call < rebase


def test_consolidation_uses_preconsolidation_monitor_before_rebase():
    source = _source()
    consolidation = source.index("physical_grip_consolidation")
    zero_lift = source.index("if zero_lift_hold_mode:")
    region = source[consolidation:zero_lift]
    assert "consolidation_monitor = formal_lift_monitor" in region
    assert "consolidation_monitor.update(" in region
    assert "wrist_reference=final_wrist_reference" in region


def test_non_sequential_modes_never_enter_consolidation():
    source = _source()
    consolidation = source.index("physical_grip_consolidation")
    guard = source.index(
        'arguments.physical_grasp_method == "sequential-compliant"',
        source.index("def run_formal_failure_recovery():"),
    )
    assert guard < consolidation
    assert "and not arguments.empty_hand_first_stage_diagnostic" in (
        source[guard:consolidation]
    )


def test_failure_flag_initialized_before_seating_and_arm_advance_guarded():
    source = _source()
    failure_decl = source.index("sequential_preconsolidation_failure = None")
    seating = source.index('phase = "closed_hand_seating"')
    preload = source.index('phase = "physical_grip_preload"')
    grasp_copy = source.index("current_arm_target = grasp_arm.copy()")
    assert failure_decl < seating < preload
    assert seating < grasp_copy
    # The final arm advance is guarded by the failure flag.
    guard = source.index(
        "if sequential_preconsolidation_failure is None:",
        seating,
    )
    assert guard < grasp_copy


def test_seating_and_preload_keep_detectors_continuous():
    source = _source()
    start = source.index("sequential_preconsolidation_failure = None")
    contact_efforts = source.index("contact_efforts = np.mean(")
    region = source[start:contact_efforts]
    assert "seating_stable_hold" in region
    assert "preload_stable_hold" in region
    assert region.count("sequential_stable_hold_step(") >= 3
    assert "except (ValueError, RuntimeError)" in region
    # Both spans must apply targets/gains through the shared helper.
    assert "controller.set_gains(kps=kps, kds=kds, save_to_usd=False)" in (
        region
    )


def test_seating_failure_freezes_targets_without_grasp_advance():
    source = _source()
    seating = source.index('phase = "closed_hand_seating"')
    grasp_copy = source.index("current_arm_target = grasp_arm.copy()")
    region = source[seating:grasp_copy]
    assert "break" in region
    assert "failed" in region
    assert "failure_reason" in region


def test_consolidation_records_applied_and_next_stiffness():
    source = _source()
    consolidation = source.index("physical_grip_consolidation")
    zero_lift = source.index("if zero_lift_hold_mode:")
    region = source[consolidation:zero_lift]
    assert "applied_finger_stiffness_scale" in region
    assert "next_command_finger_stiffness_scale" in region
    assert "step_applied_scale = applied_scale" in region
    assert "first_window_sample_applied_scale" in region
    assert "applied_scale_semantics" in region


def test_reference_single_meaning_after_rebase():
    source = _source()
    consolidation = source.index("physical_grip_consolidation")
    zero_lift = source.index("if zero_lift_hold_mode:")
    region = source[consolidation:zero_lift]
    # Old reference copied verbatim before the top-level rebase.
    preserve = region.index(
        'metrics["formal_preconsolidation_wrist_reference"] = list('
    )
    top_level = region.index(
        'metrics["formal_payload_wrist_reference"] = ['
    )
    assert preserve < top_level
    assert (
        '"sequential_consolidated_quasistatic"' in region
    )
    assert (
        'metrics["formal_payload_wrist_reference_sample_count"] = ('
        in region
    )
    # The lift monitor and the per-step increment share the final array.
    assert "wrist_reference=final_wrist_reference" in region
    assert (
        "formal_wrist_payload_reference = final_wrist_reference.copy()"
        in region
    )
    # Pre-lift evidence points at the rebased top-level fields and keeps
    # the pre-consolidation copies.
    assert '"preconsolidation_payload_reference": metrics[' in region
    assert '"payload_reference": metrics[' in region
    assert "pre_consolidation_root_reference_signed_nm" in region
    assert "final_window_root_mean_signed_nm" in region


def test_zero_lift_remains_characterization_only():
    source = _source()
    zero_lift = source.index("if zero_lift_hold_mode:")
    region = source[zero_lift:source.index("# Staged mode below")]
    assert "CHARACTERIZATION_ONLY" in region
    assert '"passed": False' in region
    assert "characterization completed" in region


def test_frozen_gates_unchanged_in_config():
    config = (
        REPOSITORY
        / "src/kcg_connector/config/d38999_tabletop_physical_grasp_v1.yaml"
    ).read_text(encoding="utf-8")
    assert "maximum_wrist_force_n: 8.0" in config
    assert "maximum_wrist_moment_nm: 0.30" in config
    assert "maximum_root_torque_delta_nm: 2.0" in config


def test_applied_scale_summary_tracks_observed_step():
    source = _source()
    consolidation = source.index("physical_grip_consolidation")
    zero_lift = source.index("if zero_lift_hold_mode:")
    region = source[consolidation:zero_lift]
    assert "min(applied_scale_min, step_applied_scale)" in region
    assert "max(applied_scale_max, step_applied_scale)" in region
    # The observed-step capture happens before the next-command advance
    # inside the loop (the first "applied_scale = float(" is the pre-loop
    # soft-scale initialization).
    capture = region.index("step_applied_scale = applied_scale")
    advance = region.index("applied_scale = float(", capture)
    assert capture < advance


def test_contact_torque_deltas_marked_pre_consolidation():
    source = _source()
    consolidation = source.index("physical_grip_consolidation")
    zero_lift = source.index("if zero_lift_hold_mode:")
    region = source[consolidation:zero_lift]
    assert "contact_torque_deltas_nm_kind" in region
    assert "pre_consolidation_signed_root_delta" in region


def test_evidence_construction_precedes_rebase_and_fails_closed():
    source = _source()
    consolidation = source.index("physical_grip_consolidation")
    zero_lift = source.index("if zero_lift_hold_mode:")
    region = source[consolidation:zero_lift]
    assert "channel_window_statistics(" in region
    assert "consolidation_evidence_construction_failed" in region
    # Local construction/validation happens before any top-level mutation.
    try_block = region.index("final_window_root_stack = np.stack(")
    mutation = region.index(
        'metrics["formal_preconsolidation_wrist_reference"] = list('
    )
    assert try_block < mutation
    # The evidence failure path calls the shared recovery afterwards.
    failure = region.index("consolidation_evidence_construction_failed")
    recovery = region.index("run_formal_failure_recovery()", failure)
    assert recovery > failure
