"""CPU-only fail-closed tests for four-position visual-XY evidence."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

import kcg_connector.d38999_visual_xy_pick_evidence as evidence


REPOSITORY = Path(__file__).resolve().parents[3]
REQUEST_PATH = REPOSITORY / evidence.DEFAULT_REQUEST_PATH
FINAL_ARTIFACT = (
    REPOSITORY
    / "artifacts/kcg_connector/d38999_visual_xy_pick_evidence_v1"
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPOSITORY / candidate


def _write_yaml(path: Path, document: dict) -> None:
    path.write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )


def _request_with_archivable_logs(tmp_path: Path) -> tuple[Path, dict]:
    """Replace ephemeral real logs with equivalent fixture logs.

    The reports, configs and CPU plans remain the byte-exact real GPU
    artifacts.  Only console-log location changes, keeping unit tests
    independent of whether `/tmp` survives a reboot.
    """

    request = yaml.safe_load(REQUEST_PATH.read_text(encoding="utf-8"))
    for index, run in enumerate(request["runs"]):
        report = json.loads(
            _resolve(run["report"]["path"]).read_text(encoding="utf-8")
        )
        log = tmp_path / f"source_{index}.log"
        log.write_text(
            "Isaac startup fixture\n"
            + json.dumps(report, allow_nan=False, sort_keys=True)
            + "\n"
            + evidence.PASS_BANNER
            + "\nIsaac shutdown fixture\n",
            encoding="utf-8",
        )
        run["console_log"] = {
            "path": str(log),
            "sha256": _digest(log),
            "size_bytes": log.stat().st_size,
            "schema_version": evidence.SOURCE_LOG_SCHEMA_VERSION,
        }
    path = tmp_path / "request.yaml"
    _write_yaml(path, request)
    return path, request


def test_four_real_gpu_reports_produce_limited_multi_position_evidence(
    tmp_path,
):
    request, _ = _request_with_archivable_logs(tmp_path)
    output = tmp_path / "bundle"
    result = evidence.aggregate_evidence(request, REPOSITORY, output)

    assert result["schema_version"] == evidence.EVIDENCE_SCHEMA_VERSION
    assert result["summary"] == {
        "pass_fraction": "4/4",
        "run_count": 4,
        "passed_run_count": 4,
        "failed_run_count": 0,
        "all_four_gpu_trials_passed": True,
        "all_hashes_sizes_and_schemas_verified": True,
    }
    visual = result["visual_xy_evidence"]
    assert visual["truth_xy_used_for_target"] is False
    assert visual["independent_probe_plan_consumed_visual_xy"] is True
    assert visual["loose_xy_error_m"]["maximum"] == pytest.approx(
        0.002868208614875893
    )
    assert visual["fixed_xy_error_m"]["maximum"] == pytest.approx(
        0.001823730164584062
    )
    assert visual["authored_loose_position_coverage_m"]["x_span"] == (
        pytest.approx(0.04)
    )
    assert visual["authored_loose_position_coverage_m"]["y_span"] == (
        pytest.approx(0.03)
    )

    physical = result["physical_pick_evidence"]
    assert physical["body_lift_m"]["minimum"] > 0.1114
    assert physical["body_tcp_slip_m"]["maximum"] < 0.0016
    assert physical["maximum_post_tare_absolute_delta_nm"]["maximum"] == (
        pytest.approx(0.3501985459588468)
    )
    assert physical["loaded_torque_channels"]["minimum"] == 3
    assert physical["final_contact_records"]["per_finger_body"]["minimum"] >= 1
    assert physical["final_contact_records"]["per_finger_nut"]["minimum"] >= 1
    assert physical["final_contact_records"]["all_runs_zero_table_contacts"]

    assert result["claims"]["full_6d_vision_claimed"] is False
    assert result["claims"]["production_control_authorized"] is False
    assert result["claims"]["same_condition_repeatability_claimed"] is False
    assert result["claims"]["rl_formal_readiness_gate_closed"] is False
    assert result["rl_readiness"]["classification"] == (
        "VALID_LIMITED_EVIDENCE_CANDIDATE"
    )
    assert result["rl_readiness"]["formal_gate_closed"] is False
    assert len(list((output / "logs").glob("*.log"))) == 4
    assert evidence.audit_artifact(output, REPOSITORY)["status"] == "PASS"


@pytest.mark.parametrize(
    ("binding_name", "field", "replacement", "match"),
    (
        ("report", "sha256", "0" * 64, "report SHA-256 mismatch"),
        ("report", "size_bytes", 1, "report byte-size mismatch"),
        (
            "report",
            "schema_version",
            "wrong",
            "report.schema_version mismatch",
        ),
        ("console_log", "sha256", "0" * 64, "console_log SHA-256 mismatch"),
        ("config", "sha256", "0" * 64, "config SHA-256 mismatch"),
        ("cpu_plan", "size_bytes", 1, "cpu_plan byte-size mismatch"),
    ),
)
def test_every_source_is_hash_size_and_schema_bound(
    tmp_path, binding_name, field, replacement, match
):
    request_path, request = _request_with_archivable_logs(tmp_path)
    request["runs"][1][binding_name][field] = replacement
    _write_yaml(request_path, request)
    with pytest.raises(evidence.EvidenceError, match=match):
        evidence.aggregate_evidence(
            request_path, REPOSITORY, tmp_path / "rejected"
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        (lambda report: report.update(passed=False), "report.passed must remain True"),
        (
            lambda report: report.update(truth_xy_used_for_target=True),
            "truth_xy_used_for_target must remain False",
        ),
        (lambda report: report.update(full_6d=True), "full_6d must remain False"),
        (
            lambda report: report.update(production_control_authorized=True),
            "production_control_authorized must remain False",
        ),
        (
            lambda report: report.update(contact_gate=False),
            "contact_gate must remain True",
        ),
        (
            lambda report: report["final_contacts"][
                "finger_body_group_records"
            ]["f1"].update(body=0),
            "f1.body must be an integer >= 1",
        ),
        (
            lambda report: report["pose_provider"]["diagnostics"][
                "endpoints"
            ]["loose_plug"].update(estimated_world_xy_m=[0.0, 0.0]),
            "visual loose XY was not preserved",
        ),
    ),
)
def test_report_claims_and_visual_target_chain_fail_closed(mutation, match):
    request = yaml.safe_load(REQUEST_PATH.read_text(encoding="utf-8"))
    run = request["runs"][0]
    report = json.loads(
        _resolve(run["report"]["path"]).read_text(encoding="utf-8")
    )
    mutation(report)
    with pytest.raises(evidence.EvidenceError, match=match):
        evidence._validate_report(
            report,
            expected_trial_id=run["expected_trial_id"],
            expected_loose_xy=tuple(run["expected_loose_xy_m"]),
            expected_fixed_xy=tuple(run["expected_fixed_xy_m"]),
            cpu_plan_path=_resolve(run["cpu_plan"]["path"]).resolve(),
        )


@pytest.mark.parametrize("failure", ("missing_banner", "wrong_json", "traceback"))
def test_console_log_must_bind_exact_report_and_pass_banner(tmp_path, failure):
    request_path, request = _request_with_archivable_logs(tmp_path)
    binding = request["runs"][0]["console_log"]
    log = Path(binding["path"])
    content = log.read_text(encoding="utf-8")
    if failure == "missing_banner":
        content = content.replace(evidence.PASS_BANNER, "no pass")
    elif failure == "wrong_json":
        report = json.loads(
            _resolve(request["runs"][0]["report"]["path"]).read_text(
                encoding="utf-8"
            )
        )
        report["body_lift_m"] += 0.001
        content = (
            "startup\n"
            + json.dumps(report, allow_nan=False, sort_keys=True)
            + "\n"
            + evidence.PASS_BANNER
            + "\n"
        )
    else:
        content += "Traceback (most recent call last):\n"
    log.write_text(content, encoding="utf-8")
    binding["sha256"] = _digest(log)
    binding["size_bytes"] = log.stat().st_size
    _write_yaml(request_path, request)
    with pytest.raises(evidence.EvidenceError, match="PASS|differs|failure"):
        evidence.aggregate_evidence(
            request_path, REPOSITORY, tmp_path / "rejected"
        )


@pytest.mark.parametrize(
    "policy_key",
    (
        "claim_same_condition_repeatability",
        "claim_full_6d",
        "claim_arbitrary_pose",
        "claim_production_control",
        "close_rl_readiness_gate",
    ),
)
def test_request_cannot_upgrade_limited_evidence_claims(tmp_path, policy_key):
    request_path, request = _request_with_archivable_logs(tmp_path)
    request["policy"][policy_key] = True
    _write_yaml(request_path, request)
    with pytest.raises(evidence.EvidenceError, match=f"policy.{policy_key}"):
        evidence.aggregate_evidence(
            request_path, REPOSITORY, tmp_path / "rejected"
        )


def test_duplicate_position_is_not_four_position_evidence(tmp_path):
    request_path, request = _request_with_archivable_logs(tmp_path)
    request["runs"][3]["expected_loose_xy_m"] = deepcopy(
        request["runs"][0]["expected_loose_xy_m"]
    )
    _write_yaml(request_path, request)
    with pytest.raises(evidence.EvidenceError, match="positions must be distinct"):
        evidence.aggregate_evidence(
            request_path, REPOSITORY, tmp_path / "rejected"
        )


def test_artifact_audit_uses_archived_log_and_rejects_tampering(tmp_path):
    request_path, _ = _request_with_archivable_logs(tmp_path)
    output = tmp_path / "bundle"
    evidence.aggregate_evidence(request_path, REPOSITORY, output)
    archived = next((output / "logs").glob("*.log"))
    archived.write_text(
        archived.read_text(encoding="utf-8") + "tampered\n",
        encoding="utf-8",
    )
    with pytest.raises(evidence.EvidenceError, match="byte-size mismatch"):
        evidence.audit_artifact(output, REPOSITORY)


def test_cli_writes_once_and_does_not_overwrite(tmp_path, capsys):
    request_path, _ = _request_with_archivable_logs(tmp_path)
    output = tmp_path / "bundle"
    arguments = [
        "--repository",
        str(REPOSITORY),
        "--request",
        str(request_path),
        "--output-dir",
        str(output),
    ]
    assert evidence.main(arguments) == 0
    assert evidence.main(arguments) == 2
    assert "output directory already exists" in capsys.readouterr().err


def test_committed_artifact_remains_self_auditing_without_tmp_logs():
    assert FINAL_ARTIFACT.is_dir()
    result = evidence.audit_artifact(FINAL_ARTIFACT, REPOSITORY)
    assert result["status"] == "PASS"
    assert result["pass_fraction"] == "4/4"
    assert result["original_tmp_logs_required"] is False
