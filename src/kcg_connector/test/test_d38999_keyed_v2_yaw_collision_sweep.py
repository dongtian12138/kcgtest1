"""Pure-CPU contracts for the independent keyed-v2 yaw collision sweep."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from kcg_connector.d38999_keyed_public_spec_v2 import (
    RECOMMENDED_ASSET_NAME,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SWEEP_PATH = PACKAGE_ROOT / "isaac/d38999_keyed_v2_yaw_collision_sweep.py"


def _load_sweep():
    spec = importlib.util.spec_from_file_location(
        "d38999_keyed_v2_yaw_collision_sweep", SWEEP_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_sweep_import_is_lazy_and_needs_no_isaac_runtime():
    script = f"""
import importlib.util
import json
import sys
spec = importlib.util.spec_from_file_location('yaw_sweep', {str(SWEEP_PATH)!r})
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
for name in ('isaacsim', 'omni', 'pxr'):
    assert name not in sys.modules, name
print(json.dumps({{'lazy_import': True}}))
"""
    environment = dict(os.environ)
    python_path = str(PACKAGE_ROOT)
    if environment.get("PYTHONPATH"):
        python_path += ":" + environment["PYTHONPATH"]
    environment["PYTHONPATH"] = python_path
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert json.loads(result.stdout) == {"lazy_import": True}


def test_yaw_matrix_is_exact_and_gap_schedule_is_bounded():
    sweep = _load_sweep()
    assert sweep.YAW_SWEEP_DEG == (
        0.0,
        -0.02,
        0.02,
        -0.03,
        0.03,
        -0.04,
        0.04,
        -0.2,
        0.2,
        -0.333,
        0.333,
        -0.35,
        0.35,
        -0.5,
        0.5,
        180.0,
    )
    assert sweep.CORE_YAW_SWEEP_DEG == (0.0, -0.5, 0.5, 180.0)
    schedule = sweep.build_gap_schedule()
    assert len(schedule) == 166
    assert schedule[0] == pytest.approx(0.002)
    assert schedule[-1] == pytest.approx(-0.0145)
    assert all(first > second for first, second in zip(schedule, schedule[1:]))
    with pytest.raises(ValueError, match="bounded"):
        sweep.build_gap_schedule(0.002, -0.100, 0.0001)


def test_output_directory_refuses_overwrite_and_default_is_separate(tmp_path):
    sweep = _load_sweep()
    output = tmp_path / "new_yaw_evidence"
    assert sweep.safe_new_output_dir(output) == output.resolve()
    output.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        sweep.safe_new_output_dir(output)
    assert sweep.default_output_dir().name == (
        "d38999_keyed_v2_yaw_collision_sweep_v1"
    )
    assert sweep.default_asset_path().name == RECOMMENDED_ASSET_NAME


def test_contact_classification_is_pair_scoped_and_key_attributed():
    sweep = _load_sweep()
    body = "/World/Trial/LoosePlug/BodyAssembly"
    fixed = "/World/Trial/FixedReceptacle"
    assert sweep.classify_pair_contact(
        (
            body,
            fixed,
            body + "/CollisionKeys/Key_00_Main",
            fixed + "/CollisionKeyways/BlockingShell/Segment_001",
        ),
        body_path=body,
        fixed_path=fixed,
    ) == "polarization_key_or_keyway"
    assert sweep.classify_pair_contact(
        (body, fixed, body + "/MatingShell/Segment_000", fixed + "/RearBody"),
        body_path=body,
        fixed_path=fixed,
    ) == "pair_other_collision"
    assert sweep.classify_pair_contact(
        (body, "/World/Table"), body_path=body, fixed_path=fixed
    ) is None


def _trial(yaw_deg):
    return {"trial_id": f"yaw_{yaw_deg}", "yaw_deg": yaw_deg}


def _polarization_sample(gap_m):
    return {
        "commanded_gap_m": gap_m,
        "measured_body_tip_z_m": -gap_m,
        "contacts": [
            {
                "kind": "polarization_key_or_keyway",
                "paths": ["CollisionKeys", "CollisionKeyways"],
            }
        ],
    }


def test_offline_summary_records_onset_bracket_without_formal_pass():
    sweep = _load_sweep()
    samples = (
        {"commanded_gap_m": 0.0001, "contacts": []},
        _polarization_sample(0.0),
        _polarization_sample(-0.0001),
    )
    summary = sweep.summarize_trial(_trial(180.0), samples)
    onset = summary["first_polarization_contact"]
    assert onset["commanded_gap_m"] == 0.0
    assert onset["previous_commanded_gap_m"] == pytest.approx(0.0001)
    assert summary["polarization_blocked_before_contact_plane"] is True
    assert summary["polarization_onset_gap_bracket_m"] == pytest.approx(
        [0.0, 0.0001]
    )
    assert summary["polarization_onset_brackets_keyway_entry"] is True
    assert summary["first_pair_contact_is_polarization_contact"] is True
    assert summary["minimum_visual_electrical_plane_margin_m"] == pytest.approx(
        0.012
    )
    assert summary["evidence_verdict"] == (
        "POLARIZATION_CONTACT_AT_KEYWAY_ENTRY_BEFORE_"
        "VISUAL_ELECTRICAL_CONTACT_PLANE"
    )
    assert summary["formal_acceptance"] == "NOT_EVALUATED_EVIDENCE_ONLY"
    assert summary["control_promotion_allowed"] is False


def test_sweep_summary_requires_both_half_degree_signs_and_c2_branch():
    sweep = _load_sweep()
    summaries = [
        sweep.summarize_trial(
            _trial(0.0),
            (
                {"commanded_gap_m": 0.0, "contacts": []},
                {"commanded_gap_m": sweep.CONTACT_PLANE_GAP_M, "contacts": []},
            ),
        )
    ]
    for yaw_deg in sweep.WRONG_YAW_AUDIT_CASES_DEG:
        summaries.append(
            sweep.summarize_trial(
                _trial(yaw_deg),
                (
                    {"commanded_gap_m": 0.0001, "contacts": []},
                    {
                        "commanded_gap_m": 0.0,
                        "contacts": _polarization_sample(0.0)["contacts"],
                    },
                ),
            )
        )
    result = sweep.summarize_sweep(summaries)
    assert result["wrong_yaw_key_block_evidence_complete"] is True
    assert result["correct_n_yaw_case"] == {
        "yaw_deg": 0.0,
        "evidence_present": True,
        "commanded_through_contact_plane": True,
        "polarization_clear_to_contact_plane": True,
        "pair_clear_to_visual_electrical_contact_plane": True,
    }
    assert result["correct_and_wrong_yaw_evidence_complete"] is True
    assert result["result_kind"] == (
        "SIMULATION_ONLY_OFFLINE_COLLISION_EVIDENCE"
    )
    assert result["formal_acceptance"] == "NOT_EVALUATED_EVIDENCE_ONLY"
    assert result["simulation_insertion_control_authorized"] is False
    assert result["robot_control_authorized"] is False
    assert result["hardware_control_authorized"] is False
    assert result[
        "coupling_ring_initial_engagement_sequence_evaluated"
    ] is False
    assert result["electrical_contact_sequence_physics_evaluated"] is False


def test_wrong_yaw_requires_first_pair_contact_to_be_key_attributed():
    sweep = _load_sweep()
    summary = sweep.summarize_trial(
        _trial(0.5),
        (
            {"commanded_gap_m": 0.0001, "contacts": []},
            {
                "commanded_gap_m": 0.0,
                "contacts": [{"kind": "pair_other_collision", "paths": []}],
            },
            _polarization_sample(-0.0001),
        ),
    )
    assert summary["first_pair_contact_is_polarization_contact"] is False
    assert summary["polarization_onset_brackets_keyway_entry"] is False
    result = sweep.summarize_sweep(
        [summary]
    )
    assert result["wrong_yaw_key_block_evidence_complete"] is False


def test_source_freezes_truth_boundary_and_does_not_use_hashes_or_old_runner():
    source = SWEEP_PATH.read_text(encoding="utf-8")
    assert '"command_policy": "FIXED_OPEN_LOOP_YAW_AND_GAP_GRID"' in source
    assert '"control_reads_contact_report": False' in source
    assert '"control_reads_pose_truth": False' in source
    assert '"posthoc_audit_reads_contact_report": True' in source
    assert '"posthoc_audit_reads_pose_truth": True' in source
    assert '"contact_or_pose_changes_next_command": False' in source
    assert "command_schedule = tuple(gaps)" in source
    assert "capture_contacts_for_offline_audit" in source
    assert "get_full_contact_report()" in source
    assert "UsdPhysics.RigidBodyAPI.Apply" in source
    assert "nut_prim.SetActive(False)" in source
    assert "joint_prim.SetActive(False)" in source
    assert "DYNAMIC_BODY_POSE_RESET_ON_PRECOMMITTED_GRID" in source
    assert "d38999_tabletop_pick_smoke" not in source
    assert "hashlib" not in source
    assert "sha256" not in source.lower()


def test_runtime_result_is_evidence_only_and_fails_closed_on_exception():
    source = SWEEP_PATH.read_text(encoding="utf-8")
    assert '"thread_collision_mode": "unmodeled"' in source
    assert '"control_authorized": False' in source
    assert "run_completed = False" in source
    assert "except BaseException as exception" in source
    assert "traceback.print_exc()" in source
    assert "exit_code=0 if run_completed else 1" in source
