"""Regression tests for the fail-closed self-collision inventory."""

import json
from pathlib import Path
import subprocess
import sys

from kcg_connector.self_collision_audit import (
    audit_self_collision,
    default_inputs,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parents[1]


def test_authoritative_inventory_is_exact_and_complete():
    report = audit_self_collision()

    assert report["urdf"]["link_count"] == 20
    assert report["urdf"]["collision_link_count"] == 17
    assert report["pair_inventory"] == {
        "candidate_collision_pair_count": 136,
        "srdf_disabled_entry_count": 96,
        "srdf_disabled_collision_pair_count": 96,
        "classification_counts": {
            "Never": 80,
            "Adjacent": 16,
            "Default": 40,
        },
        "classification_sum": 136,
    }
    assert report["srdf_integrity"] == {
        "unknown_links": [],
        "noncollision_link_references": [],
        "invalid_self_pairs": [],
        "duplicate_pairs": [],
        "unsupported_reasons": [],
    }


def test_never_risk_categories_cover_every_never_pair():
    report = audit_self_collision()

    assert report["never_pair_categories"] == {
        "arm_arm": 18,
        "arm_hand": 45,
        "finger_handbase": 5,
        "inter_finger": 10,
        "intra_finger": 2,
    }
    assert sum(report["never_pair_categories"].values()) == 80
    assert (
        "f1Link3 <-> iiwa_link_2"
        in report["never_pairs_by_category"]["arm_hand"]
    )
    assert (
        "f1Link1 <-> f2Link1"
        in report["never_pairs_by_category"]["inter_finger"]
    )


def test_current_moveit_and_isaac_configuration_fails_closed():
    report = audit_self_collision()

    assert report["status"] == "FAIL_CLOSED_UNVERIFIED"
    assert report["self_collision_verified"] is False
    assert report["full_path_self_collision_claim_allowed"] is False
    assert report["isaac"]["importer_allow_self_collision"] is False
    assert report["isaac"]["persisted_self_collision_attributes"] == {
        "newton:selfCollisionEnabled": False
    }
    assert len(report["blockers"]) == 3


def test_default_inputs_resolve_to_current_project_files():
    inputs = default_inputs()

    assert inputs.srdf == (
        PROJECT_ROOT / "src/kcg_moveit1/config/handarm.srdf"
    )
    for path in (
        inputs.urdf,
        inputs.srdf,
        inputs.isaac_importer,
        inputs.isaac_physics_usd,
    ):
        assert path.is_file()


def test_cli_json_reports_gate_and_uses_distinct_exit_codes():
    environment = {
        "PYTHONPATH": str(PACKAGE_ROOT),
    }
    command = [
        sys.executable,
        "-m",
        "kcg_connector.self_collision_audit",
        "--json",
    ]
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    report = json.loads(result.stdout)

    assert result.returncode == 2
    assert report["status"] == "FAIL_CLOSED_UNVERIFIED"
    report_only = subprocess.run(
        command + ["--report-only"],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert report_only.returncode == 0
