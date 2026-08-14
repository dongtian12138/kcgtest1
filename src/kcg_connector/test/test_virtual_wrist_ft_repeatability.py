"""Pure tests for strict multi-run virtual wrist-FT aggregation."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from kcg_connector.virtual_wrist_ft_repeatability import (
    ARTIFACT_SCHEMA_VERSION,
    MONITOR_CONFIG_PATH,
    REQUEST_SCHEMA_VERSION,
    RESULT_BANNER,
    RUNNER_SOURCE_PATH,
    aggregate_repeatability,
    main,
)
from kcg_connector.virtual_wrist_ft_runtime import (
    HOME_TARE_PHASE,
    PAYLOAD_CAPTURE_PHASE,
    PROTECTED_PHASES,
    SCHEMA_VERSION,
    load_virtual_wrist_ft_monitor_config,
)


REPOSITORY = Path(__file__).resolve().parents[3]
SCALAR_NAMES = (
    "lateral_force_n",
    "axial_force_n",
    "bending_torque_nm",
    "tightening_torque_nm",
)


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sample(step, phase, scale, *, contact):
    sample = {
        "global_step": step,
        "timestamp_s": step / 240.0,
        "runtime_phase": f"fixture_{phase.lower()}",
        "policy_phase": phase,
        "source_frame": "handbase_link",
        "target_frame": "connector_task_frame",
        "raw_wrench": [scale + index for index in range(6)],
        "canonical_wrench_sensor": [
            -(scale + index) for index in range(6)
        ],
    }
    if contact:
        sample.update(
            {
                "compensated_wrench_sensor": [
                    scale + 0.1 * index for index in range(6)
                ],
                "compensated_wrench_task": [
                    scale + 0.2 * index for index in range(6)
                ],
                "task_scalars": {
                    name: scale + index
                    for index, name in enumerate(SCALAR_NAMES)
                },
            }
        )
    return sample


def _report(scale):
    config = load_virtual_wrist_ft_monitor_config()
    peaks = {}
    for phase_index, phase in enumerate(PROTECTED_PHASES):
        peaks[phase] = {}
        for scalar_index, name in enumerate(SCALAR_NAMES):
            value = scale + phase_index + scalar_index + 1.0
            peaks[phase][name] = {
                "absolute_peak": value,
                "signed_value_at_peak": -value,
                "sample": _sample(
                    1000 + 10 * phase_index + scalar_index,
                    phase,
                    value,
                    contact=True,
                ),
            }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "MONITOR_ONLY",
        "measurement_joint": config.measurement_joint,
        "reaction_row_index": 8,
        "metadata_joint_index_offset": config.metadata_joint_index_offset,
        "raw_frame": config.raw_frame,
        "task_frame": config.task_frame_id,
        "canonical_from_raw": [
            list(row) for row in config.canonical_from_raw
        ],
        "home_empty_baseline_canonical": [
            scale + index for index in range(6)
        ],
        "payload_baseline_canonical": [
            scale + index + 10.0 for index in range(6)
        ],
        "payload_increment_estimate_canonical": [10.0] * 6,
        "phase_sample_counts": {
            HOME_TARE_PHASE: 120,
            PAYLOAD_CAPTURE_PHASE: 120,
            **{phase: 10 for phase in PROTECTED_PHASES},
            "OTHER": 20,
        },
        "protected_phase_peaks": peaks,
        "last_sample": _sample(2000, "OTHER", scale, contact=False),
        "compensation_mode": "captured_payload_quasistatic_baseline",
        "dynamic_inertia_compensation_complete": False,
        "orientation_dependent_gravity_compensation_complete": False,
        "same_scene_threshold_calibration_status": config.threshold_status,
        "calibrated_safety_limits": None,
        "monitor_only": True,
        "modifies_e2e_pass_gate": False,
        "residual_v1_enabled": False,
        "safety_gate_claimed": False,
        "assembly_success_claimed_from_wrench": False,
    }


def _metrics(scale):
    return {
        "scene": "kcg_d38999_tabletop_pick_v1",
        "gui": False,
        "passed": True,
        "end_to_end_probe_requested": True,
        "wrist_ft_monitor_requested": True,
        "end_to_end": {"passed": True},
        "virtual_wrist_ft_monitor": _report(scale),
    }


def _write_metrics(tmp_path, index, metrics, kind):
    suffix = ".log" if kind.endswith("jsonl_log") else ".json"
    path = tmp_path / f"run_{index}{suffix}"
    encoded = json.dumps(metrics, allow_nan=False, sort_keys=True)
    if kind == "headless_e2e_jsonl_log":
        content = f"Isaac startup\n{encoded}\n{RESULT_BANNER}\n"
    else:
        content = encoded + "\n"
    path.write_text(content, encoding="utf-8")
    return path


def _request(tmp_path, metrics_documents, *, kinds=None):
    source_path = REPOSITORY / RUNNER_SOURCE_PATH
    config_path = REPOSITORY / MONITOR_CONFIG_PATH
    if kinds is None:
        kinds = ["headless_e2e_jsonl_log"] * len(metrics_documents)
    runs = []
    for index, (metrics, kind) in enumerate(zip(metrics_documents, kinds)):
        path = _write_metrics(tmp_path, index, metrics, kind)
        runs.append(
            {
                "run_id": f"headless-{index + 1}",
                "metrics_artifact": {
                    "kind": kind,
                    "path": str(path),
                    "sha256": _digest(path),
                },
                "runner_source": {
                    "path": RUNNER_SOURCE_PATH,
                    "sha256": _digest(source_path),
                },
                "monitor_config": {
                    "path": MONITOR_CONFIG_PATH,
                    "sha256": _digest(config_path),
                },
            }
        )
    document = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "minimum_runs": 3,
        "policy": {
            "monitor_only": True,
            "statistics_only": True,
            "generate_safety_thresholds": False,
            "claim_calibration": False,
            "claim_training_ready": False,
        },
        "runs": runs,
    }
    path = tmp_path / "repeatability_request.yaml"
    path.write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )
    return path, document


def _rewrite_request(path, document):
    path.write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )


def test_three_or_more_log_and_report_inputs_produce_statistics_only_evidence(
    tmp_path,
):
    request, _ = _request(
        tmp_path,
        [_metrics(1.0), _metrics(2.0), _metrics(3.0)],
        kinds=[
            "headless_e2e_jsonl_log",
            "headless_e2e_metrics_json",
            "headless_e2e_jsonl_log",
        ],
    )
    result = aggregate_repeatability(request, REPOSITORY)

    assert result["schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert result["status"] == "MONITOR_ONLY_REPEATABILITY_EVIDENCE"
    assert result["run_count"] == 3
    fx = result["observed_statistics"]["wrench_vectors"][
        "home_empty_baseline_canonical"
    ]["per_axis_observed"]["Fx"]
    assert fx["mean"] == pytest.approx(2.0)
    assert fx["sample_standard_deviation"] == pytest.approx(1.0)
    assert result["provenance"]["all_hashes_verified"] is True
    assert result["claims"] == {
        "monitor_only": True,
        "statistics_only": True,
        "safety_thresholds_generated": False,
        "calibration_claimed": False,
        "training_ready_claimed": False,
        "safety_gate_enabled": False,
        "e2e_gate_modified": False,
    }
    assert result["calibrated_safety_limits"] is None
    assert result["generated_safety_thresholds"] is None
    json.dumps(result, allow_nan=False, sort_keys=True)


@pytest.mark.parametrize(
    ("case", "match"),
    (
        ("status", "status must equal MONITOR_ONLY"),
        ("home_baseline", "six-element list"),
        ("payload_baseline", "six-element list"),
        ("last_sample", "must be a mapping"),
        ("calibrated_limits", "must remain null"),
        ("monitor_only", "must remain true"),
        ("safety_gate", "must remain false"),
        ("assembly_gate", "must remain false"),
        ("dynamic_claim", "must remain false"),
        ("gui", "gui must be false"),
        ("e2e_failed", "end_to_end.passed must be true"),
    ),
)
def test_report_and_headless_boundaries_fail_closed(tmp_path, case, match):
    documents = [_metrics(1.0), _metrics(2.0), _metrics(3.0)]
    report = documents[0]["virtual_wrist_ft_monitor"]
    if case == "status":
        report["status"] = "MONITOR_FAILED"
    elif case == "home_baseline":
        report["home_empty_baseline_canonical"] = None
    elif case == "payload_baseline":
        report["payload_baseline_canonical"] = None
    elif case == "last_sample":
        report["last_sample"] = None
    elif case == "calibrated_limits":
        report["calibrated_safety_limits"] = {"axial_force_n": 5.0}
    elif case == "monitor_only":
        report["monitor_only"] = False
    elif case == "safety_gate":
        report["safety_gate_claimed"] = True
    elif case == "assembly_gate":
        report["assembly_success_claimed_from_wrench"] = True
    elif case == "dynamic_claim":
        report["dynamic_inertia_compensation_complete"] = True
    elif case == "gui":
        documents[0]["gui"] = True
    elif case == "e2e_failed":
        documents[0]["end_to_end"]["passed"] = False
    request, _ = _request(tmp_path, documents)
    with pytest.raises(ValueError, match=match):
        aggregate_repeatability(request, REPOSITORY)


@pytest.mark.parametrize(
    ("binding", "match"),
    (
        ("metrics_artifact", "metrics_artifact SHA-256 mismatch"),
        ("runner_source", "runner_source SHA-256 mismatch"),
        ("monitor_config", "monitor_config SHA-256 mismatch"),
    ),
)
def test_every_artifact_source_and_config_hash_is_verified(
    tmp_path, binding, match
):
    request, document = _request(
        tmp_path, [_metrics(1.0), _metrics(2.0), _metrics(3.0)]
    )
    document["runs"][1][binding]["sha256"] = "0" * 64
    _rewrite_request(request, document)
    with pytest.raises(ValueError, match=match):
        aggregate_repeatability(request, REPOSITORY)


def test_less_than_three_or_reused_artifacts_are_not_repeatability(tmp_path):
    request, document = _request(
        tmp_path, [_metrics(1.0), _metrics(2.0), _metrics(3.0)]
    )
    two_runs = deepcopy(document)
    two_runs["runs"] = two_runs["runs"][:2]
    _rewrite_request(request, two_runs)
    with pytest.raises(ValueError, match="at least 3"):
        aggregate_repeatability(request, REPOSITORY)

    duplicated = deepcopy(document)
    duplicated["runs"][2]["metrics_artifact"] = deepcopy(
        duplicated["runs"][0]["metrics_artifact"]
    )
    _rewrite_request(request, duplicated)
    with pytest.raises(ValueError, match="must be distinct"):
        aggregate_repeatability(request, REPOSITORY)


@pytest.mark.parametrize(
    "policy_key",
    (
        "generate_safety_thresholds",
        "claim_calibration",
        "claim_training_ready",
    ),
)
def test_request_cannot_authorize_threshold_calibration_or_training_claims(
    tmp_path, policy_key
):
    request, document = _request(
        tmp_path, [_metrics(1.0), _metrics(2.0), _metrics(3.0)]
    )
    document["policy"][policy_key] = True
    _rewrite_request(request, document)
    match = f"policy.{policy_key} must remain False"
    with pytest.raises(ValueError, match=match):
        aggregate_repeatability(request, REPOSITORY)


def test_jsonl_log_requires_exact_pass_banner(tmp_path):
    request, document = _request(
        tmp_path, [_metrics(1.0), _metrics(2.0), _metrics(3.0)]
    )
    artifact = document["runs"][0]["metrics_artifact"]
    path = Path(artifact["path"])
    path.write_text(
        path.read_text(encoding="utf-8").replace(RESULT_BANNER, "FAILED"),
        encoding="utf-8",
    )
    artifact["sha256"] = _digest(path)
    _rewrite_request(request, document)
    with pytest.raises(ValueError, match="missing the exact PASS banner"):
        aggregate_repeatability(request, REPOSITORY)


def test_cli_writes_once_and_returns_nonzero_for_overwrite(tmp_path, capsys):
    request, _ = _request(
        tmp_path, [_metrics(1.0), _metrics(2.0), _metrics(3.0)]
    )
    output = tmp_path / "repeatability.json"
    arguments = [
        "--manifest",
        str(request),
        "--repository",
        str(REPOSITORY),
        "--output",
        str(output),
    ]
    assert main(arguments) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["run_count"] == 3
    assert main(arguments) == 2
    assert "File exists" in capsys.readouterr().err
