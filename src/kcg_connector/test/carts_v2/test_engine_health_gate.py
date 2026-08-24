"""Regressions for the PhysX-backed preflight acceptance boundary."""

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[4]
ISAAC_V2 = ROOT / "src/kcg_connector/isaac/carts_v2"
sys.path.insert(0, str(ISAAC_V2))

from engine_health import (  # noqa: E402
    ENGINE_EVIDENCE_FIELDS, pending_engine_fields, preflight_is_accepted,
)
from evaluate_run import evaluate_trace  # noqa: E402


def _accepted_document() -> dict[str, object]:
    return {
        "controller_preflight_pass": True,
        "engine_health_pass": True,
        "accepted_preflight_pass": True,
        "physx_capacity_warning_count": 0,
        "configured_gpu_found_lost_aggregate_pairs_capacity": 16384,
        "configured_gpu_total_aggregate_pairs_capacity": 16384,
        "observed_gpu_found_lost_aggregate_pairs_peak": 2115,
        "observed_gpu_total_aggregate_pairs_peak": 5855,
        "engine_log_sha256": "a" * 64,
        "identity_hash_check_pass": True,
    }


def _preflight_trace() -> dict[str, object]:
    contacts = {
        "terminal_link_object": [0, 0, 0], "robot_object_unauthorized": 0,
        "robot_table": 0, "robot_fixture": 0, "object_table": 1, "examples": {},
    }
    return {
        "mode": "preflight", "object_id": "A", "candidate_id": "candidate_11",
        "physics_dt_s": 1.0 / 120.0, "offline_task_gate_passed": False,
        "identity_hash_check_pass": True, "pad_surface_identity_verified": False,
        "controller_outcome": {"completed": True, "failure_reason": None},
        "criteria": {
            "maximum_table_penetration_m": 5.0e-5, "lift_distance_m": 0.05,
            "lift_tolerance_m": 5.0e-5, "hold_duration_s": 2.0,
            "table_release_clearance_m": 5.0e-5,
            "sustained_three_contact_samples": 3,
            "lift_acceleration_difference_window_samples": 2,
            "registered_lift_peak_acceleration_m_s2": 1.0,
            "lift_acceleration_tolerance_m_s2": 0.1,
        },
        "samples": [{
            "phase": "settle", "active_positions_rad": [0.0],
            "active_velocities_rad_s": [0.0], "active_efforts_nm": [0.0],
            "object_center_m": [0.0, 0.0, 0.1],
            "object_bottom_clearance_m": 0.0,
            "object_center_in_hand_base_m": [0.0, 0.0, 0.1],
            "reference_part_orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
            "contacts": contacts,
        }],
    }


def test_complete_healthy_engine_evidence_is_accepted() -> None:
    assert preflight_is_accepted(_accepted_document())


@pytest.mark.parametrize("field", ENGINE_EVIDENCE_FIELDS)
def test_missing_engine_field_fails_closed(field: str) -> None:
    document = _accepted_document()
    del document[field]
    assert not preflight_is_accepted(document)


def test_warning_or_inconsistent_accepted_flag_fails_closed() -> None:
    warned = _accepted_document()
    warned["physx_capacity_warning_count"] = 1
    assert not preflight_is_accepted(warned)
    inconsistent = _accepted_document()
    inconsistent["engine_health_pass"] = False
    assert not preflight_is_accepted(inconsistent)


def test_trace_evaluation_cannot_pass_before_engine_audit() -> None:
    evaluation = evaluate_trace(_preflight_trace())
    assert evaluation["controller_preflight_pass"] is True
    assert evaluation["engine_health_pass"] is False
    assert evaluation["accepted_preflight_pass"] is False
    assert evaluation["preflight_pass"] is False
    assert pending_engine_fields(True, True)["engine_log_sha256"] is None
    assert not preflight_is_accepted(evaluation)


def test_runner_no_longer_accepts_legacy_preflight_pass() -> None:
    source = (ISAAC_V2 / "run_grasp_lift.py").read_text(encoding="utf-8")
    assert "preflight_is_accepted(preflight)" in source
    assert 'preflight.get("preflight_pass")' not in source
