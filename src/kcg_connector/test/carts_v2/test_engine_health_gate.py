"""Regressions for the PhysX-backed preflight acceptance boundary."""

import hashlib
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[4]
ISAAC_V2 = ROOT / "src/kcg_connector/isaac/carts_v2"
sys.path.insert(0, str(ISAAC_V2))

from engine_health import (  # noqa: E402
    ENGINE_EVIDENCE_FIELDS, audit_physx_log, finalize_engine_evaluation,
    load_runtime_resources, pending_engine_fields, preflight_is_accepted,
)
from evaluate_run import evaluate_trace  # noqa: E402


def _accepted_document() -> dict[str, object]:
    sha = "a" * 64
    return {
        "schema_version": "carts_grasp_v2_dynamic_evaluation_v2",
        "mode": "preflight", "hardware_authorized": False,
        "formal_dynamic_pass": False, "research_dynamic_pass": False,
        "controller_preflight_pass": True,
        "engine_health_pass": True,
        "accepted_preflight_pass": True,
        "physx_capacity_warning_count": 0,
        "configured_gpu_found_lost_aggregate_pairs_capacity": 8192,
        "configured_gpu_total_aggregate_pairs_capacity": 16384,
        "observed_gpu_found_lost_aggregate_pairs_peak": 2115,
        "observed_gpu_total_aggregate_pairs_peak": 5855,
        "engine_log_sha256": sha,
        "identity_hash_check_pass": True,
        "evidence_binding": {
            "config_sha256": sha, "offline_result_sha256": sha,
            "control_plan_sha256": sha, "runtime_resources_sha256": sha,
            "capacity_audit_sha256": sha, "scene_evidence_sha256": {"scene": sha},
            "object_asset_sha256": sha, "robot_asset_sha256": sha,
            "controller_source_sha256": sha, "runner_source_sha256": sha,
            "evaluator_source_sha256": sha, "engine_health_source_sha256": sha,
        },
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


def test_incomplete_identity_binding_fails_closed() -> None:
    document = _accepted_document()
    del document["evidence_binding"]["capacity_audit_sha256"]
    assert not preflight_is_accepted(document)


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
    engine_source = (ISAAC_V2 / "engine_health.py").read_text(encoding="utf-8")
    assert "preflight_is_accepted(preflight)" in source
    assert 'preflight.get("preflight_pass")' not in source
    assert '"fast_shutdown": True' in source
    assert "sdfPathToInt" in engine_source and "encodeSdfPath" not in engine_source


def test_runtime_capacity_uses_frozen_measured_rule() -> None:
    resources = load_runtime_resources(
        ROOT / "src/kcg_connector/config/carts_v2_isaac_runtime.json"
    )
    assert resources["gpu_found_lost_aggregate_pairs_capacity"] == 8192
    assert resources["gpu_total_aggregate_pairs_capacity"] == 16384
    audit = ROOT / "artifacts/carts_v2/recovery2_capacity_audit.json"
    assert resources["capacity_audit_sha256"] == hashlib.sha256(audit.read_bytes()).hexdigest()


def test_real_capacity_warning_syntax_is_detected(tmp_path: Path) -> None:
    log = tmp_path / "kit.log"
    log.write_text(
        "[Error] [omni.physx.plugin] PhysX error: The application needs to increase "
        "PxGpuDynamicsMemoryConfig::totalAggregatePairsCapacity to 5855, otherwise, "
        "the simulation will miss interactions\n",
        encoding="utf-8",
    )
    audit = audit_physx_log(log)
    assert audit["capacity_warning_count"] == 1
    assert audit["requested_total_peak"] == 5855


def test_post_shutdown_clean_log_accepts_preflight(tmp_path: Path) -> None:
    log = tmp_path / "kit.log"
    marker = "CARTS_V2_ENGINE_LOG_SYNC_1"
    payload = f"[Info] [omni.physx.plugin] clean\n[Info] {marker}\n".encode()
    log.write_bytes(payload + b"[Error] [omni.physx.plugin] post-marker teardown\n")
    engine = {
        "gpu_backend_pass": True, "physx_statistics_sample_count": 10,
        "physx_statistics_read_failures": 0,
        "engine_log_sync": {
            "marker": marker, "marker_seen": True,
            "audit_byte_count": len(payload),
            "audit_boundary": "PROCESS_START_THROUGH_SYNC_MARKER",
        },
        "configured_gpu_found_lost_aggregate_pairs_capacity": 8192,
        "configured_gpu_total_aggregate_pairs_capacity": 16384,
        "observed_gpu_found_lost_aggregate_pairs_peak": 2115,
        "observed_gpu_total_aggregate_pairs_peak": 5855,
    }
    evaluation = _accepted_document() | {"mode": "preflight"}
    accepted = finalize_engine_evaluation(evaluation, engine, log)
    assert accepted["accepted_preflight_pass"] is True
    assert accepted["engine_log_marker_seen"] is True
    assert accepted["engine_log_audit_byte_count"] == len(payload)
    assert accepted["engine_log_sha256"] == hashlib.sha256(payload).hexdigest()
    assert preflight_is_accepted(accepted)


def test_post_shutdown_physx_error_overrides_controller_pass(tmp_path: Path) -> None:
    log = tmp_path / "kit.log"
    marker = "CARTS_V2_ENGINE_LOG_SYNC_2"
    payload = ("[Error] [omni.physx.plugin] invalid collision data\n"
               f"[Info] {marker}\n").encode()
    log.write_bytes(payload)
    engine = {
        "gpu_backend_pass": True, "physx_statistics_sample_count": 1,
        "physx_statistics_read_failures": 0,
        "engine_log_sync": {
            "marker": marker, "marker_seen": True,
            "audit_byte_count": len(payload),
            "audit_boundary": "PROCESS_START_THROUGH_SYNC_MARKER",
        },
        "configured_gpu_found_lost_aggregate_pairs_capacity": 8192,
        "configured_gpu_total_aggregate_pairs_capacity": 16384,
        "observed_gpu_found_lost_aggregate_pairs_peak": 1,
        "observed_gpu_total_aggregate_pairs_peak": 1,
    }
    rejected = finalize_engine_evaluation(
        _accepted_document() | {"mode": "preflight"}, engine, log
    )
    assert rejected["physx_capacity_warning_count"] == 0
    assert rejected["engine_health_pass"] is False
    assert rejected["accepted_preflight_pass"] is False
