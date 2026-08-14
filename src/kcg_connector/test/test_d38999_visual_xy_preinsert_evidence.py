"""CPU-only tests for strict four-run visual-XY preinsert evidence."""

import ast
from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest
import yaml

import kcg_connector.d38999_visual_xy_preinsert_evidence as evidence


REPOSITORY = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPOSITORY / evidence.DEFAULT_CONFIG_PATH
MANIFEST_PATH = REPOSITORY / evidence.DEFAULT_MANIFEST_PATH
MODULE_PATH = REPOSITORY / evidence.AGGREGATOR_SOURCE_PATH


def _real_inputs():
    config = evidence.load_evidence_config(CONFIG_PATH, REPOSITORY)
    manifest, paths = evidence.load_complete_source_manifest(
        config, REPOSITORY
    )
    return config, manifest, paths


def _copy_file(source: Path, repository: Path, shown_path: str) -> None:
    target = repository / shown_path
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _copy_source_repository(tmp_path: Path, *, include_runs: bool) -> Path:
    """Copy only the small immutable inputs needed by the pure aggregator."""

    root = tmp_path / "repository"
    config = evidence.load_evidence_config(CONFIG_PATH, REPOSITORY)
    _copy_file(CONFIG_PATH, root, evidence.DEFAULT_CONFIG_PATH)
    for run in config.runs:
        _copy_file(
            REPOSITORY / run.pick_config_path,
            root,
            run.pick_config_path,
        )
    _copy_file(
        REPOSITORY / config.preinsert_config_path,
        root,
        config.preinsert_config_path,
    )
    if include_runs:
        for run in config.runs:
            directory = Path(run.run_directory)
            for filename in (
                "report.json",
                "cpu_plan.json",
                "preinsert_cpu_plan.json",
            ):
                shown = str(directory / filename)
                _copy_file(REPOSITORY / shown, root, shown)
    return root


def test_module_is_independent_cpu_only_python():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    assert roots.isdisjoint(
        {
            "isaacsim",
            "omni",
            "pxr",
            "numpy",
            "torch",
            "rclpy",
            "cv2",
            "open3d",
        }
    )
    assert "set_world_pose" not in MODULE_PATH.read_text(encoding="utf-8")


def test_versioned_config_and_manifest_bind_four_complete_runs():
    config, manifest, paths = _real_inputs()
    assert config.path == CONFIG_PATH
    assert len(config.runs) == evidence.EXPECTED_RUN_COUNT
    assert manifest["schema_version"] == (
        evidence.SOURCE_MANIFEST_SCHEMA_VERSION
    )
    assert manifest["status"] == evidence.MANIFEST_COMPLETE
    assert manifest["complete_run_count"] == 4
    assert manifest["missing_paths"] == []
    assert manifest["claims_authorized"] is False
    assert len(paths) == 4
    assert all(set(run) == set(evidence._SOURCE_NAMES) for run in paths)


def test_missing_runs_remain_explicitly_fail_closed(tmp_path):
    repository = _copy_source_repository(tmp_path, include_runs=False)
    config_path = repository / evidence.DEFAULT_CONFIG_PATH
    manifest = evidence.write_source_manifest(config_path, repository)
    assert manifest["status"] == evidence.MANIFEST_INCOMPLETE
    assert manifest["complete_run_count"] == 0
    assert len(manifest["missing_paths"]) == 12
    assert all(
        source["state"] == "MISSING"
        for run in manifest["runs"]
        for name, source in run["sources"].items()
        if name in {"report", "cpu_plan", "preinsert_cpu_plan"}
    )
    config = evidence.load_evidence_config(config_path, repository)
    with pytest.raises(evidence.EvidenceError, match="manifest is incomplete"):
        evidence.load_complete_source_manifest(config, repository)


def test_four_real_reports_produce_only_limited_preinsert_evidence():
    result = evidence.aggregate_evidence(CONFIG_PATH, REPOSITORY)
    assert result["schema_version"] == evidence.EVIDENCE_SCHEMA_VERSION
    assert result["status"] == evidence.EVIDENCE_PASS
    assert result["summary"]["pass_fraction"] == "4/4"
    assert result["summary"]["all_original_visual_picks_passed"] is True
    assert result["summary"]["all_preinsert_probes_passed"] is True
    assert result["summary"]["all_torque_strictly_below_2nm"] is True
    assert result["xy_coverage"]["authored_loose_position_m"][
        "x_span"
    ] == pytest.approx(0.040)
    assert result["xy_coverage"]["authored_loose_position_m"][
        "y_span"
    ] == pytest.approx(0.030)
    assert result["xy_coverage"]["loose_xy_error_m"][
        "maximum"
    ] == pytest.approx(0.002868208614875893)
    assert result["observed_margins"]["outside_10mm_entry_margin_m"][
        "minimum"
    ] > 0.0017
    assert result["observed_margins"]["torque_margin_to_strict_2nm_nm"][
        "minimum"
    ] > 1.64
    assert len(result["per_run"]) == 4
    assert all(
        len(run["final_contacts"]["body_counts"]) == 3
        and min(run["final_contacts"]["body_counts"].values()) >= 1
        for run in result["per_run"]
    )


def test_first_report_blocks_engage_and_all_later_claims_remain_false():
    result = evidence.aggregate_evidence(CONFIG_PATH, REPOSITORY)
    block = result["engage_gate_assessment"]["blocking_run"]
    assert result["engage_gate_assessment"]["ready_for_engage"] is False
    assert block["run_id"] == "plus10_xy"
    assert block["observed_lateral_error_m"] == pytest.approx(
        0.0010686263230156254
    )
    assert block["observed_axis_error_rad"] == pytest.approx(
        0.04073269070182799
    )
    assert block["observed_combined_entry_error_m"] == pytest.approx(
        0.001475840603162595
    )
    assert block["all_three_engage_gates_failed"] is True
    claims = result["claims"]
    assert claims["engage_executed_or_authorized"] is False
    assert claims["insertion_executed_or_authorized"] is False
    assert claims["twist_executed_or_authorized"] is False
    assert claims["home_return_executed_or_authorized"] is False
    assert claims["full_6d_claimed"] is False
    assert claims["production_control_authorized"] is False
    assert claims["full_end_to_end_assembly_claimed"] is False


@pytest.mark.parametrize(
    ("source", "field", "replacement", "match"),
    (
        ("report", "sha256", "0" * 64, "SHA-256 mismatch"),
        ("cpu_plan", "size_bytes", 1, "byte-size mismatch"),
        ("preinsert_cpu_plan", "state", "MISSING", "must be BOUND"),
        ("pick_config", "schema_version", "wrong", "schema_version mismatch"),
        ("preinsert_config", "path", "wrong", "path mismatch"),
    ),
)
def test_every_source_path_size_hash_and_schema_is_bound(
    tmp_path, source, field, replacement, match
):
    repository = _copy_source_repository(tmp_path, include_runs=True)
    config_path = repository / evidence.DEFAULT_CONFIG_PATH
    evidence.write_source_manifest(config_path, repository)
    manifest_path = repository / evidence.DEFAULT_MANIFEST_PATH
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    document["runs"][1]["sources"][source][field] = replacement
    manifest_path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config = evidence.load_evidence_config(config_path, repository)
    with pytest.raises(evidence.EvidenceError, match=match):
        evidence.load_complete_source_manifest(config, repository)


def _validate_mutated_first_report(mutation):
    config, _, source_paths = _real_inputs()
    report = json.loads(
        source_paths[0]["report"].read_text(encoding="utf-8")
    )
    mutation(report)
    return evidence._validate_report(
        report,
        source_paths[0],
        config.runs[0],
        config.thresholds,
    )


def _zero_f2_body_contact(report):
    report["final_contacts"]["finger_body_group_records"]["f2"]["body"] = 0
    report["preinsert_probe"]["final_contacts"] = deepcopy(
        report["final_contacts"]
    )


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        (
            lambda report: report.update(passed=False),
            "passed must remain True",
        ),
        (
            lambda report: report["preinsert_probe"].update(passed=False),
            "preinsert_probe.passed must remain True",
        ),
        (
            lambda report: report["preinsert_probe"].update(
                same_world_capture_gate=False
            ),
            "same_world_capture_gate must remain True",
        ),
        (
            lambda report: report.update(object_pose_writes_after_physics=1),
            "object_pose_writes_after_physics",
        ),
        (
            lambda report: report["external_contact_records"].update(table=1),
            "external_contacts.table must be zero",
        ),
        (
            _zero_f2_body_contact,
            "f2.body must be an integer >= 1",
        ),
        (
            lambda report: report["preinsert_probe"].update(
                maximum_post_tare_absolute_delta_nm=2.0
            ),
            "torque",
        ),
        (
            lambda report: report["preinsert_probe"].update(
                maximum_joint_speed_rad_s=float("nan")
            ),
            "must be finite",
        ),
        (
            lambda report: report["preinsert_probe"][
                "post_hoc_actual_alignment"
            ].update(gap_m=0.009),
            "actual gap entered",
        ),
        (
            lambda report: report.update(engage_executed=True),
            "engage_executed must remain False",
        ),
        (
            lambda report: report.update(unexpected_schema_key=1),
            "keys differ",
        ),
    ),
)
def test_report_gates_and_exact_schema_fail_closed(mutation, match):
    with pytest.raises(evidence.EvidenceError, match=match):
        _validate_mutated_first_report(mutation)


@pytest.mark.parametrize(
    "claim",
    (
        "claim_engage",
        "claim_insertion",
        "claim_twist",
        "claim_home_return",
        "claim_full_6d",
        "claim_production_control",
        "claim_full_end_to_end",
    ),
)
def test_config_cannot_upgrade_limited_claims(tmp_path, claim):
    document = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    document["policy"][claim] = True
    path = tmp_path / "scope_upgrade.yaml"
    path.write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )
    with pytest.raises(evidence.EvidenceError, match=claim):
        evidence.load_evidence_config(path, REPOSITORY)


def test_cli_writes_report_once_without_overwrite(tmp_path, capsys):
    output = tmp_path / "report.json"
    arguments = [
        "--repository",
        str(REPOSITORY),
        "--config",
        str(CONFIG_PATH),
        "--output",
        str(output),
    ]
    assert evidence.main(arguments) == 0
    assert output.is_file()
    assert evidence.main(arguments) == 2
    assert "File exists" in capsys.readouterr().err
