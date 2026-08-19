from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from kcg_connector.d38999_keyed_v2_physical_acceptance import (
    ALLOWED_CONTROLLER_INPUTS,
    EXPECTED_BENCH_NAMES,
    NOMINAL_R7_EVENT_ORDER,
    load_physical_acceptance_matrix,
)
from kcg_connector.d38999_keyed_v2_physical_model_contract import (
    REQUIRED_BENCH_IDS,
    REQUIRED_SEQUENCE_PRECEDENCE,
)


def _write_mutated(tmp_path: Path, document, name: str = "mutated.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def test_acceptance_matrix_freezes_all_benches_without_claiming_results():
    matrix = load_physical_acceptance_matrix()

    assert matrix.document["status"] == "THRESHOLDS_FROZEN_BENCHES_NOT_RUN"
    assert matrix.benches_have_run is False
    assert matrix.bench_ids == REQUIRED_BENCH_IDS
    assert {
        bench_id: matrix.document["benches"][bench_id]["name"]
        for bench_id in matrix.bench_ids
    } == EXPECTED_BENCH_NAMES
    assert matrix.document["phase_release"]["downstream_release_before_A5"] is False


def test_event_sequence_contacts_thread_and_bottoming_are_hard_gates():
    document = load_physical_acceptance_matrix().document

    assert tuple(
        tuple(edge)
        for edge in document["benches"]["P1"]["pass"][
            "required_precedence_edges"
        ]
    ) == (
        REQUIRED_SEQUENCE_PRECEDENCE
    )
    assert tuple(
        document["benches"]["P1"]["pass"]["nominal_r7_expected_order"]
    ) == (
        NOMINAL_R7_EVENT_ORDER
    )
    assert document["benches"]["P2"]["pass"][
        "block_before_first_electrical_contact"
    ] is True
    assert document["benches"]["P5"]["pass"][
        "nominal_same_label_contact_count"
    ] == 61
    assert document["benches"]["P6"]["inputs"][
        "expected_lead_m_per_revolution"
    ] == pytest.approx(0.00762)
    assert document["benches"]["P7"]["pass"]["red_band_is_not_a_pass_signal"] is True


def test_passive_capture_envelope_is_fixed_before_authoring():
    p3 = load_physical_acceptance_matrix().document["benches"]["P3"]

    assert p3["pass"]["centered_case_must_bottom"] is True
    assert p3["pass"]["minimum_passive_capture_abs_xy_m"] == pytest.approx(
        0.00005
    )
    assert p3["pass"]["minimum_passive_capture_abs_tilt_rad"] == pytest.approx(
        0.002
    )
    assert p3["pass"][
        "larger_cases_may_guide_or_block_but_not_false_bottom"
    ] is True


def test_component_torque_range_never_replaces_robot_gate():
    document = load_physical_acceptance_matrix().document
    p8 = document["benches"]["P8"]["pass"]
    p9 = document["benches"]["P9"]["pass"]
    p14 = document["benches"]["P14"]["pass"]

    assert p8["robot_formal_perpendicular_moment_nm_max"] == pytest.approx(0.30)
    assert p8["complete_pair_coupling_torque_nm_max"] == pytest.approx(4.6)
    assert p9["complete_pair_disengagement_torque_nm_min"] == pytest.approx(0.6)
    assert p9["complete_pair_disengagement_torque_nm_max"] == pytest.approx(4.6)
    assert p9["complete_pair_limits_do_not_apply_to_detent_alone"] is True
    assert p9["component_limit_cannot_relax_robot_limit"] is True
    assert p14["robot_formal_perpendicular_moment_nm_max"] == pytest.approx(0.30)
    assert p14["first_formal_exceedance_fails_episode"] is True


def test_truth_firewall_and_missing_tactile_remain_explicit():
    document = load_physical_acceptance_matrix().document
    p12 = document["benches"]["P12"]["pass"]
    p14 = document["benches"]["P14"]

    assert tuple(p14["inputs"]["controller_inputs"]) == ALLOWED_CONTROLLER_INPUTS
    assert p14["pass"]["forbidden_controller_input_count"] == 0
    assert p14["pass"]["forbidden_mechanism_count"] == 0
    assert p14["pass"]["posthoc_truth_cannot_restore_failure"] is True
    assert p12["fingertip_tactile_input_count"] == 0


def test_compliant_deflection_is_not_misread_as_hard_solver_penetration():
    document = load_physical_acceptance_matrix().document
    shared = document["shared_numeric_profile"]
    p4 = document["benches"]["P4"]
    p5 = document["benches"]["P5"]["pass"]

    assert shared[
        "hard_penetration_excludes_declared_compliant_physical_deflection"
    ] is True
    assert p4["inputs"]["seal_physical_deflection_probe_m"] == pytest.approx(
        [0.000280, 0.000435, 0.000590]
    )
    assert p4["pass"]["external_work_accounted_in_energy_balance"] is True
    assert p5["max_hard_annulus_or_wrong_surface_penetration_m"] == pytest.approx(
        0.00005
    )
    assert p5["compliant_sleeve_physical_deflection_m_max"] == pytest.approx(
        0.000075
    )


def test_full_yaw_sweep_cannot_be_narrowed(tmp_path):
    source = load_physical_acceptance_matrix()
    document = deepcopy(source.document)
    document["benches"]["P2"]["inputs"]["yaw_step_deg"] = 5

    with pytest.raises(ValueError, match="one-degree yaw sweep"):
        load_physical_acceptance_matrix(_write_mutated(tmp_path, document))


def test_bad_contact_positive_control_cannot_be_removed(tmp_path):
    source = load_physical_acceptance_matrix()
    document = deepcopy(source.document)
    document["benches"]["P5"]["pass"]["bad_control_complete_contact_count_max"] = 61

    with pytest.raises(ValueError, match="failing control"):
        load_physical_acceptance_matrix(_write_mutated(tmp_path, document))


def test_robot_moment_gate_cannot_be_relaxed(tmp_path):
    source = load_physical_acceptance_matrix()
    document = deepcopy(source.document)
    document["benches"]["P14"]["pass"][
        "robot_formal_perpendicular_moment_nm_max"
    ] = 0.31

    with pytest.raises(ValueError, match="changed from the frozen threshold"):
        load_physical_acceptance_matrix(_write_mutated(tmp_path, document))


def test_fingerprint_metadata_is_rejected_without_computing_it(tmp_path):
    source = load_physical_acceptance_matrix()
    document = deepcopy(source.document)
    document["evidence_policy"]["checksum"] = "not-computed"

    with pytest.raises(ValueError, match="fingerprint metadata"):
        load_physical_acceptance_matrix(_write_mutated(tmp_path, document))
