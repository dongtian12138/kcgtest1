"""Pure tests for the versioned D38999 nut damping scan contract."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from kcg_connector.d38999_nut_damping_scan import (
    BASELINE_CANDIDATE_ID,
    DEFAULT_D38999_NUT_DAMPING_SCAN_PATH,
    D38999_NUT_DAMPING_SCAN_SCHEMA_VERSION,
    load_d38999_nut_damping_scan,
    select_damping_candidate,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PACKAGE_ROOT / "isaac/d38999_nut_damping_scan.py"


def _document():
    return yaml.safe_load(
        DEFAULT_D38999_NUT_DAMPING_SCAN_PATH.read_text(encoding="utf-8")
    )


def _write(tmp_path, document):
    path = tmp_path / "scan.yaml"
    path.write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )
    return path


def _summary(
    candidate_id,
    nut_speed,
    relative_speed=None,
    scene_speed=None,
    *,
    safe=True,
):
    return {
        "candidate_id": candidate_id,
        "repeat_count": 2,
        "every_repeat_finite": safe,
        "every_repeat_scene_safety_pass": safe,
        "maximum_tail_scene_angular_speed_rad_s": (
            nut_speed if scene_speed is None else scene_speed
        ),
        "maximum_tail_nut_angular_speed_rad_s": nut_speed,
        "maximum_tail_relative_axis_speed_rad_s": (
            nut_speed if relative_speed is None else relative_speed
        ),
    }


def test_shipped_scan_keeps_baseline_and_increasing_rigid_body_candidates():
    config = load_d38999_nut_damping_scan()
    assert config.schema_version == D38999_NUT_DAMPING_SCAN_SCHEMA_VERSION
    assert config.baseline.candidate_id == BASELINE_CANDIDATE_ID
    assert config.baseline.mechanism == "none"
    assert config.baseline.angular_damping is None
    assert config.baseline.expected_resolved_angular_damping == (
        pytest.approx(0.05)
    )
    values = [item.angular_damping for item in config.candidates[1:]]
    assert values == pytest.approx([0.25, 0.50, 1.00, 2.00])
    assert all(
        item.mechanism == "physx_rigid_body_angular_damping"
        for item in config.candidates[1:]
    )
    assert all(not item.requires_articulation for item in config.candidates)
    assert config.experiment.settle_steps == 480
    assert config.experiment.tail_steps == 120
    assert config.acceptance.baseline_reproduction_metric == (
        "maximum_tail_any_dynamic_body_angular_speed_rad_s"
    )
    assert (
        config.acceptance.baseline_tail_scene_angular_speed_minimum_rad_s
        == pytest.approx(0.050)
    )
    assert (
        config.acceptance.baseline_tail_scene_angular_speed_maximum_rad_s
        == pytest.approx(0.200)
    )
    assert config.acceptance.candidate_speed_reduction_metric == (
        "maximum_tail_nut_angular_speed_rad_s"
    )
    assert config.acceptance.maximum_tail_nut_angular_speed_rad_s == (
        pytest.approx(0.020)
    )
    assert config.acceptance.minimum_speed_reduction_fraction == (
        pytest.approx(0.70)
    )
    json.dumps(config.as_dict(), allow_nan=False, sort_keys=True)


def test_joint_friction_is_explicitly_excluded_without_efficacy_claim():
    config = load_d38999_nut_damping_scan()
    excluded = {
        item.mechanism: item for item in config.excluded_mechanisms
    }
    joint_friction = excluded["physx_joint_friction"]
    assert joint_friction.target_component == (
        "non_articulation_revolute_joint"
    )
    assert joint_friction.runnable_in_v1 is False
    assert joint_friction.effectiveness_claimed is False
    assert config.physx_evidence.joint_friction_effect_claim == (
        "unverified_on_non_articulation_revolute"
    )


def test_selection_chooses_lowest_damping_that_meets_every_gate():
    config = load_d38999_nut_damping_scan()
    summaries = [
        _summary(BASELINE_CANDIDATE_ID, 0.090, scene_speed=0.090),
        _summary("nut_angular_damping_0p25", 0.030),
        _summary("nut_angular_damping_0p50", 0.018),
        _summary("nut_angular_damping_1p00", 0.010),
        _summary("nut_angular_damping_2p00", 0.005),
    ]
    selection = select_damping_candidate(config, summaries)
    assert selection.baseline_valid is True
    assert selection.selected_candidate_id == "nut_angular_damping_0p50"
    assert selection.selected_angular_damping == pytest.approx(0.50)
    assert selection.eligible_candidate_ids == (
        "nut_angular_damping_0p50",
        "nut_angular_damping_1p00",
        "nut_angular_damping_2p00",
    )
    assert selection.automatic_promotion_permitted is False
    assert selection.reason == "evidence_candidate_selected_not_promoted"


def test_selection_rejects_unreproduced_baseline_or_unsafe_candidate():
    config = load_d38999_nut_damping_scan()
    summaries = [
        _summary(BASELINE_CANDIDATE_ID, 0.030, scene_speed=0.030),
        _summary("nut_angular_damping_0p25", 0.010),
        _summary("nut_angular_damping_0p50", 0.008),
        _summary("nut_angular_damping_1p00", 0.005),
        _summary("nut_angular_damping_2p00", 0.003),
    ]
    selection = select_damping_candidate(config, summaries)
    assert selection.baseline_valid is False
    assert selection.selected_candidate_id is None
    assert selection.reason == "baseline_not_reproduced"

    summaries[0] = _summary(
        BASELINE_CANDIDATE_ID, 0.090, scene_speed=0.090
    )
    for index in range(1, len(summaries)):
        summaries[index]["every_repeat_scene_safety_pass"] = False
    selection = select_damping_candidate(config, summaries)
    assert selection.baseline_valid is True
    assert selection.selected_candidate_id is None
    assert selection.reason == "no_candidate_met_scan_acceptance"


def test_measured_scan_reproduces_old_scene_metric_but_selects_nothing():
    """The first GPU scan proved body motion, not 0.05 rad/s nut motion."""
    config = load_d38999_nut_damping_scan()
    summaries = [
        _summary(
            BASELINE_CANDIDATE_ID,
            0.029832101115878425,
            relative_speed=0.029773542253478003,
            scene_speed=0.08564867199179224,
        ),
        _summary(
            "nut_angular_damping_0p25",
            0.014557297588925187,
            relative_speed=0.04467175490077477,
            scene_speed=0.05629835,
        ),
        _summary(
            "nut_angular_damping_0p50",
            0.016319012323908186,
            relative_speed=0.015284763244193818,
            scene_speed=0.02728361,
        ),
        _summary(
            "nut_angular_damping_1p00",
            0.011616485981874007,
            relative_speed=0.02090102218256903,
            scene_speed=0.03338844,
        ),
        _summary(
            "nut_angular_damping_2p00",
            0.01562644843284575,
            relative_speed=0.021798240380021545,
            scene_speed=0.02482404,
        ),
    ]
    selection = select_damping_candidate(config, summaries)
    candidate_0p50_reduction = 1.0 - (
        summaries[2]["maximum_tail_nut_angular_speed_rad_s"]
        / summaries[0]["maximum_tail_nut_angular_speed_rad_s"]
    )
    assert selection.baseline_valid is True
    assert candidate_0p50_reduction == pytest.approx(0.4529714062)
    assert candidate_0p50_reduction < 0.70
    assert selection.selected_candidate_id is None
    assert selection.eligible_candidate_ids == ()
    assert selection.reason == "no_candidate_met_scan_acceptance"


@pytest.mark.parametrize(
    ("mutator", "message"),
    (
        (lambda doc: doc.update(extra=True), "keys are invalid"),
        (
            lambda doc: doc["scope"].update(
                source_asset_mutation_allowed=True
            ),
            "mutation",
        ),
        (
            lambda doc: doc["physx_evidence"].update(
                schema_default_angular_damping=0.0
            ),
            "default must be 0.05",
        ),
        (
            lambda doc: doc["candidates"][0].update(
                angular_damping=0.05
            ),
            "baseline must not author",
        ),
        (
            lambda doc: doc["candidates"][1].update(
                mechanism="physx_joint_friction"
            ),
            "rigid-body damping",
        ),
        (
            lambda doc: doc["excluded_mechanisms"][0].update(
                effectiveness_claimed=True
            ),
            "cannot be run or claimed",
        ),
        (
            lambda doc: doc["promotion_gates"].update(
                automatic_promotion=True
            ),
            "automatic promotion",
        ),
        (
            lambda doc: doc["experiment"].update(
                object_pose_writes_after_start=1
            ),
            "pose writes",
        ),
    ),
)
def test_loader_rejects_unsafe_or_unsubstantiated_scan_changes(
    tmp_path, mutator, message
):
    document = deepcopy(_document())
    mutator(document)
    with pytest.raises(ValueError, match=message):
        load_d38999_nut_damping_scan(_write(tmp_path, document))


def test_loader_and_headless_script_import_without_isaac_runtime():
    script = f'''
import importlib.util
import json
import sys
from kcg_connector.d38999_nut_damping_scan import load_d38999_nut_damping_scan
spec = importlib.util.spec_from_file_location(
    "damping_scan", {str(SCRIPT_PATH)!r}
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
for name in ("isaacsim", "omni", "pxr"):
    assert name not in sys.modules, name
print(json.dumps({{"lazy_import": True}}))
'''
    environment = dict(__import__("os").environ)
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


def test_headless_cli_exit_code_is_fail_closed_without_isaac_runtime():
    script = f'''
import importlib.util
spec = importlib.util.spec_from_file_location(
    "damping_scan", {str(SCRIPT_PATH)!r}
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert module._process_exit_code(True) == 0
assert module._process_exit_code(False) == 1
try:
    module._process_exit_code(1)
except TypeError:
    pass
else:
    raise AssertionError("non-boolean scan verdict was accepted")
'''
    environment = dict(__import__("os").environ)
    python_path = str(PACKAGE_ROOT)
    if environment.get("PYTHONPATH"):
        python_path += ":" + environment["PYTHONPATH"]
    environment["PYTHONPATH"] = python_path
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "return _process_exit_code(passed)" in source
    assert "raise SystemExit(main())" in source


def test_headless_script_is_scan_only_and_never_edits_source_asset():
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "CreateAngularDampingAttr" in source
    assert "PhysxJointAPI" not in source
    assert "jointFriction" not in source
    assert "SetAngularVelocity" not in source
    assert "set_angular_velocity" not in source
    assert "object_pose_writes_after_start" in source
    assert "source_asset_mutated" in source
    assert "exit_code=_process_exit_code(passed)" in source
    assert "raise SystemExit(main())" in source
