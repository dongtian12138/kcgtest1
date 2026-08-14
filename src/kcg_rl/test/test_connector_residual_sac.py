"""Pure contract tests for formal connector residual SAC orchestration."""

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import hashlib
import json
import math
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from kcg_rl.connector_residual_sac import (
    ACTION_SIZE,
    EVALUATION_RUNTIME_COMPATIBILITY_FIELDS,
    INTERFACE_VERSION,
    OBSERVATION_SIZE,
    SCHEMA_VERSION,
    TrainingRawSafetyAudit,
    aggregate_evaluation_reports,
    aggregate_paired_evaluation_reports,
    backend_randomization_metadata,
    capture_provenance_snapshot,
    clopper_pearson_lower_bound,
    compare_reset_initial_signatures,
    comparable_episode_randomization,
    exact_mcnemar_one_sided_p_value,
    file_sha256,
    load_connector_residual_sac_config,
    load_training_metadata_for_model,
    normalized_reset_initial_signature,
    paired_execution_order,
    provenance_metadata,
    physical_episode_report,
    positive_claim_training_evidence,
    resolved_backend_curriculum_document,
    resolved_backend_randomization_document,
    resolved_config_document,
    state_mapping_sha256,
    training_randomization_phase_verified,
    resolve_training_timesteps,
    validate_evaluation_provenance,
    validate_evaluation_runtime,
    validate_loaded_actor_training_binding,
    validate_raw_safety_report,
    validate_training_raw_safety_report,
    validate_curriculum_provenance,
    validate_randomization_provenance,
    validate_resolved_curriculum_snapshot,
    validate_resolved_randomization_snapshot,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PACKAGE_ROOT / "config/connector_residual_sac.yaml"


def test_formal_config_is_explicit_and_targets_residual_v0():
    config = load_connector_residual_sac_config(CONFIG_PATH)
    assert config.schema_version == SCHEMA_VERSION
    assert config.interface_version == INTERFACE_VERSION
    assert config.action_size == ACTION_SIZE
    assert config.observation_size == OBSERVATION_SIZE
    assert config.device == "cuda:0"
    assert config.required_torch_version == "2.11.0+cu128"
    assert config.required_cuda_build == "12.8"
    assert config.maximum_unconfirmed_timesteps == 32
    assert config.evaluation_deterministic is True
    assert config.minimum_success_rate == pytest.approx(0.95)
    assert config.maximum_safety_failures == 0
    assert config.use_vecnormalize is False
    assert config.save_replay_buffer is True


def test_formal_training_requires_explicit_steps_and_long_run_opt_in():
    config = load_connector_residual_sac_config(CONFIG_PATH)
    with pytest.raises(ValueError, match="explicit"):
        resolve_training_timesteps(
            None, config, allow_long_training=False
        )
    assert (
        resolve_training_timesteps(
            32, config, allow_long_training=False
        )
        == 32
    )
    with pytest.raises(ValueError, match="allow-long-training"):
        resolve_training_timesteps(
            33, config, allow_long_training=False
        )
    assert (
        resolve_training_timesteps(
            33, config, allow_long_training=True
        )
        == 33
    )


@pytest.mark.parametrize(
    "field,value,message",
    (
        ("action_size", 3, "action size"),
        ("observation_size", 23, "observation size"),
    ),
)
def test_formal_config_rejects_changed_contract(
    tmp_path, field, value, message
):
    document = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    document["contract"][field] = value
    changed = tmp_path / "changed.yaml"
    changed.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_connector_residual_sac_config(changed)


def test_provenance_hashes_paths_with_smoke_compatible_keys(tmp_path):
    source = tmp_path / "source.py"
    source.write_bytes(b"formal source\n")
    metadata = provenance_metadata({"runner": source})
    assert metadata["source_runner_path"] == str(source.resolve())
    assert metadata["source_runner_sha256"] == file_sha256(source)


def test_provenance_snapshot_keeps_exact_preflight_bytes(tmp_path):
    source = tmp_path / "source.py"
    source.write_bytes(b"before preflight\n")
    metadata, contents = capture_provenance_snapshot({"runner": source})
    source.write_bytes(b"changed during training\n")
    assert contents["runner"] == b"before preflight\n"
    assert metadata["source_runner_sha256"] == hashlib.sha256(
        contents["runner"]
    ).hexdigest()
    assert metadata["source_runner_sha256"] != file_sha256(source)


def test_loaded_randomization_snapshot_is_resolved_and_fail_closed():
    @dataclass(frozen=True)
    class Distribution:
        enabled: bool
        offsets: tuple[float, ...]

    backend = SimpleNamespace(
        scene=SimpleNamespace(
            randomization_config=Distribution(
                enabled=True, offsets=(-0.01, 0.01)
            )
        )
    )
    document = resolved_backend_randomization_document(backend)
    payload = yaml.safe_dump(
        document, sort_keys=True, allow_unicode=True
    ).encode("utf-8")
    metadata = {
        "resolved_randomization_config": document,
        "resolved_randomization_config_sha256": hashlib.sha256(
            payload
        ).hexdigest(),
    }
    validate_resolved_randomization_snapshot(metadata, document)
    with pytest.raises(ValueError, match="differs"):
        validate_resolved_randomization_snapshot(
            metadata, {"enabled": True, "offsets": [-0.02, 0.02]}
        )
    with pytest.raises(ValueError, match="lacks"):
        validate_resolved_randomization_snapshot({}, document)


def test_loaded_curriculum_snapshot_matches_backend_and_fails_closed():
    @dataclass(frozen=True)
    class ResidualConfig:
        interface_version: str
        minimum_axial_progress_fraction: float
        tightening_direction: int
        target_angle_rad: float

    config = ResidualConfig(
        INTERFACE_VERSION, 0.90, -1, 1.0
    )
    document = {
        "interface_version": INTERFACE_VERSION,
        "stage_name": "stage60",
        "maximum_episode_steps": 100,
        "minimum_axial_progress_fraction": 0.90,
        "initial_q7_rad": 0.0,
        "planned_final_q7_rad": -1.0,
        "q7_safe_lower_rad": -2.5,
        "q7_safe_upper_rad": 2.5,
        "q7_command_reserve_rad": math.radians(10.0),
        "resolved_residual_config": asdict(config),
    }
    backend = SimpleNamespace(
        scene=SimpleNamespace(
            residual_config=config,
            resolved_curriculum_stage=document,
            maximum_episode_steps=100,
            checkpoint_positions=(0.0,),
            q7_index=0,
        )
    )
    normalized = resolved_backend_curriculum_document(backend)
    payload = yaml.safe_dump(
        normalized, sort_keys=True, allow_unicode=True
    ).encode("utf-8")
    metadata = {
        "resolved_curriculum_stage": normalized,
        "resolved_curriculum_stage_sha256": hashlib.sha256(
            payload
        ).hexdigest(),
    }
    validate_resolved_curriculum_snapshot(metadata, normalized)
    with pytest.raises(ValueError, match="differs"):
        validate_resolved_curriculum_snapshot(
            metadata, dict(normalized, stage_name="stage120")
        )
    backend.scene.maximum_episode_steps = 180
    with pytest.raises(ValueError, match="episode limit"):
        resolved_backend_curriculum_document(backend)


def test_curriculum_provenance_requires_yaml_and_resolver_hashes():
    complete = {
        "source_curriculum_config_path": "/tmp/curriculum.yaml",
        "source_curriculum_config_sha256": "config-hash",
        "source_curriculum_contract_path": "/tmp/curriculum.py",
        "source_curriculum_contract_sha256": "contract-hash",
    }
    validate_curriculum_provenance(complete)
    incomplete = dict(complete)
    del incomplete["source_curriculum_config_sha256"]
    with pytest.raises(ValueError, match="config_sha256"):
        validate_curriculum_provenance(incomplete)


def test_backend_randomization_metadata_is_bounded_json_evidence():
    backend = SimpleNamespace(
        randomization_enabled=True,
        randomization_schema_version=(
            "kcg_connector_residual_randomization_v1"
        ),
        physics_randomization_applied=False,
        safety_signal_source="raw_physics",
        episode_randomization_history=[
            {"episode": 1, "seed": 1201, "action_delay_policy_steps": 0},
            {"episode": 2, "seed": 1204, "action_delay_policy_steps": 1},
        ],
    )
    metadata = backend_randomization_metadata(backend)
    assert metadata["control_observation_randomization_applied"] is True
    assert metadata["episode_randomization_count"] == 2
    assert metadata["episode_randomization_first"]["seed"] == 1201
    assert metadata["episode_randomization_last"]["seed"] == 1204
    assert metadata["physics_parameter_randomization"] == {
        "friction": False,
        "mass": False,
        "thread_lead": False,
    }
    assert metadata["physics_randomization_applied"] is False
    assert metadata["safety_signal_source"] == "raw_physics"
    json.dumps(metadata, allow_nan=False)
    training_only = backend_randomization_metadata(
        backend, history_start=1
    )
    assert training_only["episode_randomization_count"] == 1
    assert training_only["episode_randomization_first"]["seed"] == 1204
    assert training_only["episode_randomization_last"]["seed"] == 1204


def test_training_randomization_phase_excludes_preflight_and_checks_seed():
    phase = {
        "randomization_enabled": True,
        "episode_randomization_count": 2,
        "episode_randomization_first": {"seed": 42},
    }
    assert training_randomization_phase_verified(
        phase, training_reset_count=2, expected_seed=42
    )
    assert not training_randomization_phase_verified(
        phase, training_reset_count=3, expected_seed=42
    )
    assert not training_randomization_phase_verified(
        phase, training_reset_count=2, expected_seed=0
    )
    assert training_randomization_phase_verified(
        {
            "randomization_enabled": False,
            "episode_randomization_count": 0,
        },
        training_reset_count=1,
        expected_seed=42,
    )


@pytest.mark.parametrize(
    "change,message",
    (
        ({"physics_randomization_applied": True}, "forbids"),
        ({"safety_signal_source": "noisy_observation"}, "raw_physics"),
        ({"randomization_schema_version": None}, "schema"),
    ),
)
def test_backend_randomization_metadata_rejects_unsafe_contract(
    change, message
):
    values = {
        "randomization_enabled": True,
        "randomization_schema_version": (
            "kcg_connector_residual_randomization_v1"
        ),
        "physics_randomization_applied": False,
        "safety_signal_source": "raw_physics",
        "episode_randomization_history": [],
    }
    values.update(change)
    with pytest.raises((TypeError, ValueError), match=message):
        backend_randomization_metadata(SimpleNamespace(**values))


def test_enabled_randomization_requires_yaml_and_sampler_hashes():
    randomization = {
        "control_observation_randomization_applied": True
    }
    complete = {
        "source_randomization_config_path": "/tmp/randomization.yaml",
        "source_randomization_config_sha256": "config-hash",
        "source_randomization_contract_path": "/tmp/randomization.py",
        "source_randomization_contract_sha256": "contract-hash",
    }
    validate_randomization_provenance(randomization, complete)
    incomplete = dict(complete)
    del incomplete["source_randomization_config_sha256"]
    with pytest.raises(ValueError, match="config_sha256"):
        validate_randomization_provenance(randomization, incomplete)


def test_training_metadata_binds_the_exact_model_file(tmp_path):
    model = tmp_path / "final_model.zip"
    model.write_bytes(b"model-v0")
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "action_size": ACTION_SIZE,
        "observation_size": OBSERVATION_SIZE,
        "vecnormalize_used": False,
        "model_sha256": file_sha256(model),
    }
    metadata_path = tmp_path / "training_metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    loaded_path, loaded = load_training_metadata_for_model(model)
    assert loaded_path == metadata_path
    assert loaded == metadata

    model.write_bytes(b"changed")
    with pytest.raises(ValueError, match="model hash"):
        load_training_metadata_for_model(model)


def test_actor_state_hash_is_order_independent_and_fail_closed():
    first = {
        "layer.weight": np.asarray([[1.0, 2.0]], dtype=np.float32),
        "layer.bias": np.asarray([0.5], dtype=np.float32),
    }
    reordered = {
        "layer.bias": first["layer.bias"].copy(),
        "layer.weight": first["layer.weight"].copy(),
    }
    changed = dict(reordered)
    changed["layer.bias"] = np.asarray([0.6], dtype=np.float32)
    assert state_mapping_sha256(first) == state_mapping_sha256(reordered)
    assert state_mapping_sha256(first) != state_mapping_sha256(changed)
    assert len(state_mapping_sha256(first)) == 64

    with pytest.raises(ValueError, match="nonempty mapping"):
        state_mapping_sha256({})
    with pytest.raises(ValueError, match="finite"):
        state_mapping_sha256(
            {"bad": np.asarray([np.nan], dtype=np.float32)}
        )
    with pytest.raises(ValueError, match="numeric"):
        state_mapping_sha256({"bad": np.asarray(["value"])})


def _positive_training_raw_safety_report(policy_steps=2000):
    peaks = {
        "physics_substep_max_abs_finger_base_torque_nm": 0.6,
        "physics_substep_max_abs_joint_velocity_rad_s": 0.8,
        "physics_substep_max_abs_q7_velocity_rad_s": 0.2,
        "physics_substep_max_joint_limit_violation_rad": 0.0,
        "policy_boundary_max_abs_nut_angular_velocity_rad_s": 0.2,
        "policy_boundary_max_abs_q7_tracking_error_rad": 0.01,
        "policy_boundary_max_grasp_rotation_error_rad": 0.02,
        "policy_boundary_max_grasp_translation_error_m": 0.001,
    }
    limits = {
        "physics_substep_max_abs_finger_base_torque_nm": 1.0,
        "physics_substep_max_abs_q7_velocity_rad_s": 0.4,
        "physics_substep_max_joint_limit_violation_rad": 0.02,
        "policy_boundary_max_abs_nut_angular_velocity_rad_s": 0.45,
        "policy_boundary_max_abs_q7_tracking_error_rad": 0.04,
        "policy_boundary_max_grasp_rotation_error_rad": 0.09,
        "policy_boundary_max_grasp_translation_error_m": 0.005,
    }
    episode = {
        "complete": False,
        "evidence_valid_throughout": True,
        "failure_reasons": [],
        "finite_throughout": True,
        "last_sampling": {
            "physics_substep": {
                "includes_episode_initial_snapshot": True,
                "rate_hz": 240.0,
                "samples": 48001,
            },
            "policy_boundary": {
                "includes_episode_initial_snapshot": True,
                "rate_hz": 10.0,
                "samples": 2001,
            },
        },
        "limits": limits,
        "passed": True,
        "peaks": peaks,
        "policy_steps": policy_steps,
        "signal_source": "raw_physics",
    }
    return {
        "complete_episode_count": 0,
        "episode_reports": [episode],
        "evidence_valid_throughout": True,
        "failure_reasons": [],
        "finite_throughout": True,
        "limits": limits,
        "partial_episode_count": 1,
        "passed": True,
        "peaks": peaks,
        "policy_steps_audited": policy_steps,
        "schema_version": "kcg_training_raw_safety_audit_v1",
        "signal_source": "raw_physics",
    }


def _positive_training_metadata(**changes):
    raw_safety = _positive_training_raw_safety_report()
    metadata = {
        "actor_reload_verified": True,
        "actor_final_state_sha256": "1" * 64,
        "actor_initial_state_sha256": "0" * 64,
        "actor_parameter_max_delta": 1.0e-4,
        "learning_starts": 1000,
        "model_timesteps": 2000,
        "optimization_expected": True,
        "optimization_verified": True,
        "optimizer_updates": 1000,
        "passed": True,
        "training_completed": True,
        "training_raw_safety_complete_episode_count": 0,
        "training_raw_safety_failure_reasons": [],
        "training_raw_safety_partial_episode_count": 1,
        "training_raw_safety_passed": True,
        "training_raw_safety_peaks": raw_safety["peaks"],
        "training_raw_safety_policy_steps_audited": 2000,
        "training_raw_safety_report": raw_safety,
        "reloaded_actor_state_sha256": "1" * 64,
        "training_config": {
            "gradient_steps": 1,
            "learning_starts": 1000,
            "train_freq_steps": 1,
        },
    }
    metadata.update(changes)
    return metadata


def test_positive_claim_training_evidence_requires_real_optimization():
    verified = positive_claim_training_evidence(
        _positive_training_metadata()
    )
    assert verified["policy_improvement_training_evidence_verified"] is True
    assert verified["minimum_required_optimizer_updates"] == 999
    assert verified["policy_improvement_training_evidence_failures"] == []

    multi_gradient = positive_claim_training_evidence(
        _positive_training_metadata(
            optimizer_updates=1999,
            training_config={
                "gradient_steps": 4,
                "learning_starts": 1000,
                "train_freq_steps": 2,
            },
        )
    )
    assert multi_gradient["minimum_required_optimizer_updates"] == 1999
    assert multi_gradient[
        "policy_improvement_training_evidence_verified"
    ] is True
    multi_gradient_below = positive_claim_training_evidence(
        _positive_training_metadata(
            optimizer_updates=1998,
            training_config={
                "gradient_steps": 4,
                "learning_starts": 1000,
                "train_freq_steps": 2,
            },
        )
    )
    assert multi_gradient_below[
        "policy_improvement_training_evidence_verified"
    ] is False

    dry_run = positive_claim_training_evidence(
        _positive_training_metadata(
            actor_final_state_sha256="0" * 64,
            actor_parameter_max_delta=0.0,
            model_timesteps=32,
            optimization_expected=False,
            optimizer_updates=0,
        )
    )
    assert dry_run["policy_improvement_training_evidence_verified"] is False
    assert "optimization_expected_not_true" in dry_run[
        "policy_improvement_training_evidence_failures"
    ]
    assert "model_timesteps_not_beyond_learning_starts" in dry_run[
        "policy_improvement_training_evidence_failures"
    ]
    assert "actor_state_sha256_unchanged" in dry_run[
        "policy_improvement_training_evidence_failures"
    ]


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        {},
        _positive_training_metadata(training_completed=False),
        _positive_training_metadata(passed=False),
        _positive_training_metadata(optimization_verified=False),
        _positive_training_metadata(actor_reload_verified=False),
        _positive_training_metadata(optimizer_updates=998),
        _positive_training_metadata(actor_parameter_max_delta=1.0e-8),
        _positive_training_metadata(actor_parameter_max_delta=float("nan")),
        _positive_training_metadata(actor_initial_state_sha256=None),
        _positive_training_metadata(actor_final_state_sha256="0" * 64),
        _positive_training_metadata(
            reloaded_actor_state_sha256="2" * 64
        ),
        _positive_training_metadata(training_config=None),
        _positive_training_metadata(
            training_config={
                "gradient_steps": 1,
                "learning_starts": 999,
                "train_freq_steps": 1,
            }
        ),
    ],
)
def test_positive_claim_training_evidence_rejects_missing_or_bad_values(
    metadata,
):
    evidence = positive_claim_training_evidence(metadata)
    assert evidence["policy_improvement_training_evidence_verified"] is False
    assert evidence["policy_improvement_training_evidence_failures"]


def test_evaluation_rejects_changed_source_or_asset_hash():
    current = {
        "source_backend_path": "/tmp/backend.py",
        "source_backend_sha256": "new",
    }
    with pytest.raises(ValueError, match="provenance mismatch"):
        validate_evaluation_provenance(
            {"source_backend_sha256": "trained"}, current
        )


def _runtime_metadata_fixture():
    return {
        "gpu": "Test GPU",
        "gymnasium": "1.2.3",
        "isaacsim": "6.0.1.0",
        "numpy": "2.4.0",
        "python": "3.11.9",
        "stable_baselines3": "2.7.1",
        "torch": "2.11.0+cu128",
        "torch_cuda_build": "12.8",
    }


def _raw_safety_report_fixture():
    return {
        "failure_reasons": [],
        "finite_throughout": True,
        "limits": {
            "physics_substep_max_abs_finger_base_torque_nm": 1.0,
            "physics_substep_max_abs_q7_velocity_rad_s": 0.40,
            "physics_substep_max_joint_limit_violation_rad": 0.02,
            "policy_boundary_max_abs_nut_angular_velocity_rad_s": 0.45,
            "policy_boundary_max_abs_q7_tracking_error_rad": 0.04,
            "policy_boundary_max_grasp_rotation_error_rad": 0.09,
            "policy_boundary_max_grasp_translation_error_m": 0.005,
        },
        "metrics": {
            "physics_substep_max_abs_finger_base_torque_nm": 0.6,
            "physics_substep_max_abs_joint_velocity_rad_s": 0.8,
            "physics_substep_max_abs_q7_velocity_rad_s": 0.2,
            "physics_substep_max_joint_limit_violation_rad": 0.0,
            "policy_boundary_max_abs_nut_angular_velocity_rad_s": 0.2,
            "policy_boundary_max_abs_q7_tracking_error_rad": 0.01,
            "policy_boundary_max_grasp_rotation_error_rad": 0.02,
            "policy_boundary_max_grasp_translation_error_m": 0.001,
        },
        "passed": True,
        "sampling": {
            "physics_substep": {
                "includes_episode_initial_snapshot": True,
                "rate_hz": 240.0,
                "samples": 241,
            },
            "policy_boundary": {
                "includes_episode_initial_snapshot": True,
                "rate_hz": 10.0,
                "samples": 11,
            },
        },
        "signal_source": "raw_physics",
    }


def test_evaluation_runtime_requires_exact_recorded_match():
    runtime = _runtime_metadata_fixture()
    validate_evaluation_runtime(dict(runtime), dict(runtime))


@pytest.mark.parametrize(
    "field", EVALUATION_RUNTIME_COMPATIBILITY_FIELDS
)
def test_evaluation_runtime_rejects_missing_changed_or_boolean_fields(field):
    training = _runtime_metadata_fixture()
    current = _runtime_metadata_fixture()

    missing_training = dict(training)
    missing_training.pop(field)
    with pytest.raises(ValueError, match="missing or invalid"):
        validate_evaluation_runtime(missing_training, current)

    missing_current = dict(current)
    missing_current.pop(field)
    with pytest.raises(ValueError, match="missing or invalid"):
        validate_evaluation_runtime(training, missing_current)

    changed_current = dict(current)
    changed_current[field] = current[field] + "-different"
    with pytest.raises(ValueError, match="runtime mismatch"):
        validate_evaluation_runtime(training, changed_current)

    boolean_training = dict(training)
    boolean_training[field] = True
    with pytest.raises(ValueError, match="missing or invalid"):
        validate_evaluation_runtime(boolean_training, current)

    boolean_current = dict(current)
    boolean_current[field] = False
    with pytest.raises(ValueError, match="missing or invalid"):
        validate_evaluation_runtime(training, boolean_current)


@pytest.mark.parametrize("invalid", [None, [], "runtime"])
def test_evaluation_runtime_rejects_non_mapping_inputs(invalid):
    runtime = _runtime_metadata_fixture()
    with pytest.raises(TypeError, match="training metadata"):
        validate_evaluation_runtime(invalid, runtime)
    with pytest.raises(TypeError, match="current runtime metadata"):
        validate_evaluation_runtime(runtime, invalid)


def test_raw_safety_report_strictly_validates_schema_and_evidence():
    expected = _raw_safety_report_fixture()
    assert validate_raw_safety_report(expected) == expected

    unsafe = _raw_safety_report_fixture()
    unsafe["metrics"] = dict(unsafe["metrics"])
    unsafe["metrics"][
        "physics_substep_max_abs_finger_base_torque_nm"
    ] = 1.01
    unsafe["passed"] = False
    unsafe["failure_reasons"] = ["finger_overload"]
    assert validate_raw_safety_report(unsafe)["passed"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        lambda report: report.update(passed="false"),
        lambda report: report.update(finite_throughout=np.bool_(True)),
        lambda report: report.update(extra=True),
        lambda report: report["metrics"].update(
            physics_substep_max_abs_joint_velocity_rad_s=float("nan")
        ),
        lambda report: report["sampling"]["physics_substep"].update(
            samples=True
        ),
    ],
)
def test_raw_safety_report_rejects_invalid_or_nonfinite_data(mutation):
    report = _raw_safety_report_fixture()
    mutation(report)
    with pytest.raises(ValueError, match="raw safety"):
        validate_raw_safety_report(report)


def _raw_safety_info(report):
    return {
        "raw_safety_failure_reasons": report["failure_reasons"],
        "raw_safety_passed": report["passed"],
        "raw_safety_peaks": report["metrics"],
        "safety_signal_source": "raw_physics",
    }


def test_training_raw_safety_audit_covers_complete_and_partial_episodes():
    audit = TrainingRawSafetyAudit()
    first = _raw_safety_report_fixture()
    audit.record_step(first, _raw_safety_info(first), episode_done=False)
    terminal = _raw_safety_report_fixture()
    terminal["metrics"] = dict(terminal["metrics"])
    terminal["metrics"][
        "physics_substep_max_abs_joint_velocity_rad_s"
    ] = 0.9
    terminal["sampling"] = {
        name: dict(entry) for name, entry in terminal["sampling"].items()
    }
    terminal["sampling"]["physics_substep"]["samples"] = 481
    terminal["sampling"]["policy_boundary"]["samples"] = 21
    audit.record_step(
        terminal, _raw_safety_info(terminal), episode_done=True
    )
    partial = _raw_safety_report_fixture()
    audit.record_step(
        partial, _raw_safety_info(partial), episode_done=False
    )

    report = audit.finalize()
    assert validate_training_raw_safety_report(report) == report
    assert report["passed"] is True
    assert report["policy_steps_audited"] == 3
    assert report["complete_episode_count"] == 1
    assert report["partial_episode_count"] == 1
    assert report["peaks"][
        "physics_substep_max_abs_joint_velocity_rad_s"
    ] == pytest.approx(0.9)


@pytest.mark.parametrize("failure", ["schema", "nonfinite", "projection"])
def test_training_raw_safety_audit_fails_closed_on_bad_step_evidence(
    failure,
):
    audit = TrainingRawSafetyAudit()
    report = _raw_safety_report_fixture()
    info = _raw_safety_info(report)
    if failure == "schema":
        report.pop("sampling")
    elif failure == "nonfinite":
        report["metrics"] = dict(report["metrics"])
        report["metrics"][
            "physics_substep_max_abs_joint_velocity_rad_s"
        ] = float("nan")
    else:
        info = dict(info, raw_safety_passed=False)
    audit.record_step(report, info, episode_done=False)
    result = audit.finalize()
    assert result["passed"] is False
    assert result["policy_steps_audited"] == 1
    assert result["failure_reasons"]
    json.dumps(result, allow_nan=False)


def test_training_raw_safety_audit_latches_transient_violation():
    audit = TrainingRawSafetyAudit()
    unsafe = _raw_safety_report_fixture()
    unsafe["metrics"] = dict(unsafe["metrics"])
    unsafe["metrics"][
        "physics_substep_max_abs_finger_base_torque_nm"
    ] = 1.01
    unsafe["failure_reasons"] = ["finger_overload"]
    unsafe["passed"] = False
    audit.record_step(
        unsafe, _raw_safety_info(unsafe), episode_done=False
    )
    recovered = _raw_safety_report_fixture()
    audit.record_step(
        recovered, _raw_safety_info(recovered), episode_done=True
    )
    result = audit.finalize()
    assert result["passed"] is False
    assert "finger_overload" in result["failure_reasons"]
    assert "backend_raw_safety_passed_recovered" in result[
        "failure_reasons"
    ]


def test_positive_claim_requires_exact_training_raw_safety_evidence():
    missing = _positive_training_metadata()
    missing.pop("training_raw_safety_passed")
    evidence = positive_claim_training_evidence(missing)
    assert evidence["policy_improvement_training_evidence_verified"] is False
    assert "training_raw_safety_passed_not_true" in evidence[
        "policy_improvement_training_evidence_failures"
    ]

    inconsistent = _positive_training_metadata(
        training_raw_safety_policy_steps_audited=1999
    )
    evidence = positive_claim_training_evidence(inconsistent)
    assert evidence["policy_improvement_training_evidence_verified"] is False
    assert "training_raw_safety_policy_steps_audited_projection_mismatch" in (
        evidence["policy_improvement_training_evidence_failures"]
    )


def test_loaded_actor_is_bound_to_resolved_training_archive(tmp_path):
    config = load_connector_residual_sac_config(CONFIG_PATH)
    resolved = resolved_config_document(config)
    archive = tmp_path / "resolved_training_config.yaml"
    archive.write_text(
        yaml.safe_dump(resolved, sort_keys=True), encoding="utf-8"
    )
    actor_hash = "a" * 64
    metadata = {
        "actor_final_state_sha256": actor_hash,
        "actor_reload_verified": True,
        "reloaded_actor_state_sha256": actor_hash,
        "source_resolved_training_config_path": str(archive),
        "source_resolved_training_config_sha256": file_sha256(archive),
        "training_config": resolved,
    }
    validate_loaded_actor_training_binding(
        metadata, config, actor_hash
    )

    with pytest.raises(ValueError, match="loaded actor"):
        validate_loaded_actor_training_binding(
            metadata, config, "b" * 64
        )
    mismatched = dict(metadata, training_config={})
    with pytest.raises(ValueError, match="configuration differs"):
        validate_loaded_actor_training_binding(
            mismatched, config, actor_hash
        )


def test_physical_report_enforces_stage_axial_progress_fraction():
    from kcg_connector.residual_rl import (
        ConnectorResidualState,
        load_connector_residual_config,
    )

    task_path = (
        PACKAGE_ROOT.parents[0]
        / "kcg_connector/config/connector_task.yaml"
    )
    base = load_connector_residual_config(task_path)
    config = replace(base, minimum_axial_progress_fraction=0.90)

    def report_at_fraction(fraction, *, raw_safe=True):
        state = ConnectorResidualState(
            phase_progress=1.0,
            q7_position_rad=-config.target_angle_rad,
            q7_tracking_error_rad=0.0,
            q7_velocity_rad_s=0.0,
            nut_angle_rad=config.target_angle_rad,
            nut_angular_velocity_rad_s=0.0,
            axial_travel_m=fraction * config.expected_axial_travel_m,
            axial_velocity_m_s=0.0,
            grasp_translation_error_m=(0.0, 0.0, 0.0),
            grasp_rotation_error_rad=(0.0, 0.0, 0.0),
            finger_torques_nm=(0.10, 0.10, 0.10),
            finger_torque_deltas_nm=(0.0, 0.0, 0.0),
            clamp_positions_rad=config.clamp_nominal_positions_rad,
            stable_hold_seconds=config.success_hold_duration_s,
        )
        backend = SimpleNamespace(
            previous_state=state,
            scene=SimpleNamespace(residual_config=config),
            start_q7=0.0,
            episode_safety=SimpleNamespace(
                max_abs_velocity=0.0,
                max_limit_violation=0.0,
                max_finger_torque_delta=0.10,
                finite_throughout=True,
            ),
        )
        raw_safety = {
            "failure_reasons": (
                [] if raw_safe else ["finger_overload_transient"]
            ),
            "finite_throughout": True,
            "limits": {
                "physics_substep_max_abs_finger_base_torque_nm": (
                    config.maximum_absolute_finger_torque_nm
                ),
                "physics_substep_max_abs_q7_velocity_rad_s": (
                    config.maximum_q7_speed_rad_s * 1.10
                ),
                "physics_substep_max_joint_limit_violation_rad": 0.02,
                "policy_boundary_max_abs_nut_angular_velocity_rad_s": (
                    config.maximum_q7_speed_rad_s * 1.25
                ),
                "policy_boundary_max_abs_q7_tracking_error_rad": (
                    config.maximum_q7_tracking_error_rad
                ),
                "policy_boundary_max_grasp_rotation_error_rad": (
                    config.maximum_grasp_rotation_error_rad
                ),
                "policy_boundary_max_grasp_translation_error_m": (
                    config.maximum_grasp_translation_error_m
                ),
            },
            "metrics": {
                "physics_substep_max_abs_finger_base_torque_nm": (
                    0.10
                    if raw_safe
                    else config.maximum_absolute_finger_torque_nm + 0.01
                ),
                "physics_substep_max_abs_joint_velocity_rad_s": 0.0,
                "physics_substep_max_abs_q7_velocity_rad_s": 0.0,
                "physics_substep_max_joint_limit_violation_rad": 0.0,
                "policy_boundary_max_abs_nut_angular_velocity_rad_s": 0.0,
                "policy_boundary_max_abs_q7_tracking_error_rad": 0.0,
                "policy_boundary_max_grasp_rotation_error_rad": 0.0,
                "policy_boundary_max_grasp_translation_error_m": 0.0,
            },
            "passed": raw_safe,
            "sampling": {
                "physics_substep": {
                    "includes_episode_initial_snapshot": True,
                    "rate_hz": 240.0,
                    "samples": 241,
                },
                "policy_boundary": {
                    "includes_episode_initial_snapshot": True,
                    "rate_hz": 10.0,
                    "samples": 11,
                },
            },
            "signal_source": "raw_physics",
        }
        backend.raw_safety_report = raw_safety
        final_info = {
            "raw_safety_failure_reasons": raw_safety[
                "failure_reasons"
            ],
            "raw_safety_passed": raw_safe,
            "raw_safety_peaks": raw_safety["metrics"],
            "safety_signal_source": "raw_physics",
        }
        return physical_episode_report(
            backend,
            np.zeros(OBSERVATION_SIZE, dtype=np.float32),
            episode=1,
            episode_return=1.0,
            episode_steps=10,
            terminated=True,
            truncated=False,
            termination_reason="success",
            final_info=final_info,
            reset_info={"safety_signal_source": "raw_physics"},
            seed=42,
        )

    insufficient = report_at_fraction(0.89)
    complete = report_at_fraction(0.90)
    assert insufficient["axial_progress_gate_passed"] is False
    assert insufficient["passed"] is False
    assert complete["minimum_axial_progress_fraction"] == pytest.approx(0.90)
    assert complete["axial_progress_gate_passed"] is True
    assert complete["passed"] is True
    unsafe = report_at_fraction(0.90, raw_safe=False)
    assert unsafe["termination_reason"] == "success"
    assert unsafe["raw_safety_passed"] is False
    assert unsafe["passed"] is False


def test_evaluation_summary_reports_only_failed_termination_reasons():
    reports = [
        _evaluation_report(
            passed=True, seed=1, policy_steps=28, episode_return=12.0
        ),
        _evaluation_report(
            passed=False,
            seed=2,
            reason="lost_grasp",
            policy_steps=9,
            episode_return=-5.0,
        ),
    ]
    summary = aggregate_evaluation_reports(reports, 0.5)
    assert summary["success_count"] == 1
    assert summary["success_rate"] == 0.5
    assert summary["failure_reason_counts"] == {"lost_grasp": 1}
    assert summary["safety_failure_count"] == 1
    assert summary["acceptance_passed"] is False


def test_evaluation_may_tolerate_timeout_but_never_a_safety_failure():
    reports = [
        _evaluation_report(passed=True, seed=index)
        for index in range(19)
    ]
    reports.append(
        _evaluation_report(
            passed=False,
            seed=19,
            reason="time_limit",
            policy_steps=40,
        )
    )
    summary = aggregate_evaluation_reports(reports, 0.95)
    assert summary["success_rate"] == pytest.approx(0.95)
    assert summary["safety_failure_count"] == 0
    assert summary["acceptance_passed"] is True

    reports[-1]["termination_reason"] = "cross_thread"
    summary = aggregate_evaluation_reports(reports, 0.95)
    assert summary["safety_failure_count"] == 1
    assert summary["acceptance_passed"] is False


def test_evaluation_counts_raw_violation_independently_of_reason():
    unsafe = _evaluation_report(
        passed=False,
        seed=1,
        reason="success",
        raw_safety_passed=False,
    )
    summary = aggregate_evaluation_reports([unsafe], 0.0)
    assert summary["raw_safety_failure_count"] == 1
    assert summary["termination_safety_failure_count"] == 0
    assert summary["safety_failure_count"] == 1
    assert summary["acceptance_passed"] is False


@pytest.mark.parametrize(
    "changes",
    [
        {"passed": "false"},
        {"raw_safety_passed": "false"},
        {"return": float("nan")},
        {"seed": True},
        {"safety_signal_source": "policy_observation"},
    ],
)
def test_evaluation_report_schema_rejects_truthy_or_nonfinite_values(
    changes,
):
    report = _evaluation_report(passed=False, seed=1)
    report.update(changes)
    with pytest.raises(ValueError, match="evaluation"):
        aggregate_evaluation_reports([report], 0.0)


def _evaluation_report(
    *,
    passed,
    seed,
    reason=None,
    raw_safety_passed=True,
    policy_steps=26,
    episode_return=None,
):
    if episode_return is None:
        episode_return = 10.0 if passed else 0.0
    return {
        "passed": passed,
        "policy_steps": policy_steps,
        "raw_safety_failure_reasons": (
            [] if raw_safety_passed else ["test_raw_safety_violation"]
        ),
        "raw_safety_passed": raw_safety_passed,
        "return": episode_return,
        "safety_signal_source": "raw_physics",
        "seed": seed,
        "termination_reason": (
            "success" if passed else (reason or "time_limit")
        ),
    }


def test_paired_randomization_comparison_ignores_only_episode_counter():
    first = {
        "episode": 1,
        "seed": 10000,
        "hand_kp_scale": 0.98,
        "clamp_nominal_offsets_rad": [0.001, -0.002, 0.003],
    }
    repeated = dict(first, episode=2)
    changed = dict(repeated, hand_kp_scale=1.02)
    assert comparable_episode_randomization(first) == (
        comparable_episode_randomization(repeated)
    )
    assert comparable_episode_randomization(first) != (
        comparable_episode_randomization(changed)
    )
    assert comparable_episode_randomization(None) is None


def test_reset_initial_signature_comparison_has_physical_tolerances():
    first = {
        "body_position": np.zeros(3),
        "nut_position": [0.0, 0.0, 0.0],
        "q7": 0.0,
    }
    assert normalized_reset_initial_signature(first) == {
        "body_position": [0.0, 0.0, 0.0],
        "nut_position": [0.0, 0.0, 0.0],
        "q7": 0.0,
    }
    within = {
        "body_position": [0.0001, 0.0, 0.0],
        "nut_position": [0.0, 0.0001, 0.0],
        "q7": math.radians(0.1),
    }
    assert compare_reset_initial_signatures(first, within)["passed"] is True
    outside = dict(within, q7=math.nextafter(math.radians(0.1), math.inf))
    assert compare_reset_initial_signatures(first, outside)["passed"] is False


def test_paired_execution_order_counterbalances_one_based_pairs():
    assert paired_execution_order(1) == (
        ("zero", False),
        ("trained_deterministic", True),
    )
    assert paired_execution_order(2) == (
        ("trained_deterministic", True),
        ("zero", False),
    )
    assert paired_execution_order(3) == paired_execution_order(1)
    assert paired_execution_order(np.int64(4)) == paired_execution_order(2)


@pytest.mark.parametrize("invalid", [True, 0, -1, 1.0, "1"])
def test_paired_execution_order_rejects_invalid_pair_indices(invalid):
    with pytest.raises(ValueError, match="positive integer"):
        paired_execution_order(invalid)


def test_paired_reports_require_exact_match_booleans_and_equal_seeds():
    zero = [_evaluation_report(passed=False, seed=10)]
    trained = [_evaluation_report(passed=True, seed=10)]
    with pytest.raises(ValueError, match="exact booleans"):
        aggregate_paired_evaluation_reports(
            zero, trained, ["false"], training_evidence_verified=True
        )

    trained[0] = _evaluation_report(passed=True, seed=11)
    summary = aggregate_paired_evaluation_reports(
        zero, trained, [True], training_evidence_verified=True
    )
    assert summary["paired_seed_mismatch_count"] == 1
    assert summary["paired_data_integrity_passed"] is False
    assert summary["policy_improvement_criteria_passed"] is False


def test_paired_perfect_zero_baseline_makes_no_improvement_claim():
    zero = [
        _evaluation_report(passed=True, seed=10000 + index)
        for index in range(20)
    ]
    trained = [dict(report) for report in zero]
    summary = aggregate_paired_evaluation_reports(
        zero, trained, [True] * 20
    )
    assert summary["paired_data_integrity_passed"] is True
    assert summary["zero_policy_summary"]["success_rate"] == 1.0
    assert summary["trained_policy_summary"]["success_rate"] == 1.0
    assert summary["success_rate_improvement"] == 0.0
    assert summary["paired_regression_count"] == 0
    assert summary["paired_exact_mcnemar_one_sided_p_value"] == 1.0
    assert summary[
        "trained_success_rate_clopper_pearson_lower_bound"
    ] == pytest.approx(0.8608916593)
    assert summary["policy_improvement_criteria_passed"] is False


def test_exact_paired_statistics_match_known_values():
    assert exact_mcnemar_one_sided_p_value(0, 0) == 1.0
    assert exact_mcnemar_one_sided_p_value(2, 0) == pytest.approx(0.25)
    assert exact_mcnemar_one_sided_p_value(10, 0) == pytest.approx(
        1.0 / 1024.0
    )
    assert clopper_pearson_lower_bound(20, 20) == pytest.approx(
        0.8608916593
    )
    assert clopper_pearson_lower_bound(99, 100) == pytest.approx(
        0.9534401885
    )


def test_small_paired_sample_cannot_make_positive_claim():
    zero = [
        _evaluation_report(passed=index < 17, seed=10000 + index)
        for index in range(20)
    ]
    trained = [
        _evaluation_report(passed=index < 19, seed=10000 + index)
        for index in range(20)
    ]
    summary = aggregate_paired_evaluation_reports(
        zero,
        trained,
        [True] * 20,
        training_evidence_verified=True,
    )
    assert summary["zero_policy_summary"]["success_rate"] == 0.85
    assert summary["trained_policy_summary"]["success_rate"] == 0.95
    assert summary["success_rate_improvement"] == pytest.approx(0.10)
    assert summary["paired_regression_count"] == 0
    assert summary["paired_improvement_count"] == 2
    assert summary["paired_exact_mcnemar_one_sided_p_value"] == pytest.approx(
        0.25
    )
    assert summary["paired_statistical_evidence_passed"] is False
    assert summary["policy_improvement_criteria_passed"] is False

    weakened = aggregate_paired_evaluation_reports(
        zero,
        trained,
        [True] * 20,
        confidence_level=0.50,
        maximum_paired_p_value=1.0,
        minimum_improvement_margin=0.0,
        minimum_paired_episodes=1,
        minimum_trained_success_rate=0.0,
        training_evidence_verified=True,
    )
    assert weakened["minimum_paired_episodes"] == 100
    assert weakened["confidence_level"] == pytest.approx(0.95)
    assert weakened["maximum_paired_p_value"] == pytest.approx(0.05)
    assert weakened["minimum_improvement_margin"] == pytest.approx(0.10)
    assert weakened["minimum_trained_success_rate"] == pytest.approx(0.95)
    assert weakened["policy_improvement_criteria_passed"] is False


def test_paired_claim_requires_training_statistics_and_no_regression():
    zero = [
        _evaluation_report(passed=index < 90, seed=10000 + index)
        for index in range(100)
    ]
    trained = [
        _evaluation_report(passed=True, seed=10000 + index)
        for index in range(100)
    ]
    summary = aggregate_paired_evaluation_reports(
        zero,
        trained,
        [True] * 100,
        training_evidence_verified=True,
    )
    assert summary["zero_policy_summary"]["success_rate"] == 0.90
    assert summary["trained_policy_summary"]["success_rate"] == 1.0
    assert summary["success_rate_improvement"] == pytest.approx(0.10)
    assert summary["paired_improvement_count"] == 10
    assert summary["paired_regression_count"] == 0
    assert summary["paired_statistical_evidence_passed"] is True
    assert summary["policy_improvement_criteria_passed"] is True

    no_training_evidence = aggregate_paired_evaluation_reports(
        zero,
        trained,
        [True] * 100,
        training_evidence_verified=False,
    )
    assert no_training_evidence[
        "policy_improvement_training_evidence_verified"
    ] is False
    assert no_training_evidence[
        "policy_improvement_criteria_passed"
    ] is False

    regressed = [dict(report) for report in trained]
    regressed[0] = _evaluation_report(
        passed=False, seed=10000, reason="time_limit"
    )
    regressed_summary = aggregate_paired_evaluation_reports(
        zero,
        regressed,
        [True] * 100,
        training_evidence_verified=True,
    )
    assert regressed_summary["paired_regression_count"] == 1
    assert (
        regressed_summary["policy_improvement_criteria_passed"] is False
    )


def test_paired_integrity_rejects_domain_mismatch_or_safety_failure():
    zero = [
        _evaluation_report(passed=False, seed=10000 + index)
        for index in range(20)
    ]
    trained = [
        _evaluation_report(passed=index < 19, seed=10000 + index)
        for index in range(20)
    ]
    mismatched = aggregate_paired_evaluation_reports(
        zero, trained, [True] * 19 + [False]
    )
    assert mismatched["paired_data_integrity_passed"] is False
    assert mismatched["policy_improvement_criteria_passed"] is False

    unsafe = [dict(report) for report in trained]
    unsafe[-1] = _evaluation_report(
        passed=False, seed=10019, reason="lost_grasp"
    )
    unsafe_summary = aggregate_paired_evaluation_reports(
        zero, unsafe, [True] * 20
    )
    assert unsafe_summary["trained_policy_summary"][
        "safety_failure_count"
    ] == 1
    assert unsafe_summary["paired_data_integrity_passed"] is False
    assert unsafe_summary["policy_improvement_criteria_passed"] is False


def test_orchestration_import_does_not_load_training_or_sim_modules():
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PACKAGE_ROOT)
    script = """
import importlib
import sys

module = importlib.import_module("kcg_rl.connector_residual_sac")
assert module.ACTION_SIZE == 4
for name in ("torch", "stable_baselines3", "gymnasium", "omni", "isaacsim"):
    assert name not in sys.modules, name
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
