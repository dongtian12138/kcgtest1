"""Pure contracts for the opt-in 24-tooth jitter diagnostic."""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import runpy
import sys

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = PACKAGE_ROOT / "isaac/d38999_nut_tooth_jitter_probe.py"
REGRASP_PATH = PACKAGE_ROOT / "isaac/d38999_nut_regrasp_smoke.py"
E2E_PATH = PACKAGE_ROOT / "isaac/d38999_tabletop_pick_smoke.py"
PROJECT_ROOT = PACKAGE_ROOT.parents[1]


def _module():
    return runpy.run_path(str(PROBE_PATH), run_name="tooth_probe_test")


def _regrasp_module():
    """Import the runner without calling main or loading Isaac modules."""

    isaac_directory = str(REGRASP_PATH.parent)
    sys.path.insert(0, isaac_directory)
    try:
        spec = importlib.util.spec_from_file_location(
            "d38999_nut_regrasp_render_ab_test", REGRASP_PATH
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(isaac_directory)


class FakeSettings:
    def __init__(self, values):
        self.values = dict(values)

    def get(self, path):
        return self.values.get(path)


def test_probe_import_is_lazy_and_has_no_isaac_or_pxr_side_effects():
    before = set(sys.modules)
    _module()
    imported = set(sys.modules) - before
    assert not any(
        name == "isaacsim" or name.startswith(("isaacsim.", "omni.", "pxr"))
        for name in imported
    )
    roots = set()
    for node in ast.parse(PROBE_PATH.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    assert roots.isdisjoint({"isaacsim", "omni", "pxr", "numpy"})


def test_segment_colors_are_unique_stable_and_complete():
    colors = _module()["deterministic_segment_colors"]()
    assert list(colors) == [f"Segment_{index:02d}" for index in range(24)]
    assert len({tuple(value) for value in colors.values()}) == 24
    assert colors["Segment_00"] == [0.95, 0.266, 0.266]
    assert all(
        0.0 <= channel <= 1.0
        for rgb in colors.values()
        for channel in rgb
    )


def test_segment_path_extraction_is_exact_and_range_checked():
    extract = _module()["segment_index_from_path"]
    root = "/World/Loose/BodyAssembly/CouplingNut"
    assert extract(root + "/Segment_00") == 0
    assert extract(root + "/Segment_23/mesh") == 23
    assert extract(root + "/Segment_24") is None
    assert extract(root + "/Segment_2") is None
    assert extract(root + "/Other_00") is None


def test_segment00_schema_exposes_the_missing_zero_rotate_outlier():
    summarize = _module()["summarize_segment00_schema"]
    missing = summarize(["xformOp:translate", "xformOp:scale"])
    assert missing == {
        "explicit_rotate_z": False,
        "explicit_rotate_z_degrees": None,
        "op_names": ["xformOp:translate", "xformOp:scale"],
        "schema_outlier": True,
    }
    normalized = summarize(
        ["xformOp:translate", "xformOp:rotateZ", "xformOp:scale"],
        {"xformOp:rotateZ": 0.0},
    )
    assert normalized["schema_outlier"] is False
    assert normalized["explicit_rotate_z_degrees"] == 0.0


def test_both_smokes_have_opt_in_hooks_and_default_paths_are_unchanged():
    regrasp = REGRASP_PATH.read_text(encoding="utf-8")
    e2e = E2E_PATH.read_text(encoding="utf-8")
    for source in (regrasp, e2e):
        assert "--nut-tooth-jitter-output" in source
        assert "--nut-tooth-jitter-normalize-segment00-op" in source
        assert "--nut-tooth-jitter-colorize" in source
        assert "NutToothJitterProbe(" in source
        assert "tooth_probe.sample(" in source
        assert "tooth_probe.finalize()" in source
    assert 'str(phase).startswith("end_to_end_")' in e2e
    assert "get_full_contact_report()" in regrasp
    assert "get_full_contact_report()" in e2e


def test_session_ab_does_not_edit_the_checked_in_asset_or_safety_gates():
    source = PROBE_PATH.read_text(encoding="utf-8")
    assert "stage.GetSessionLayer()" in source
    assert "stage.SetEditTarget(previous)" in source
    assert ".write_text(" in source  # report.json only
    assert "checked-in USDA remains untouched" in source
    assert "set_world_pose(" not in source
    assert "set_local_pose(" not in source
    assert "hard_stop_nm" not in source
    assert "acceptance" not in source


def test_output_is_compact_unless_a_threshold_is_exceeded():
    source = PROBE_PATH.read_text(encoding="utf-8")
    assert '"summary.csv"' in source
    assert '"anomalies.jsonl"' in source
    assert '"report.json"' in source
    assert "if anomalous:" in source
    assert '"segments": details' in source
    assert "TRANSLATION_THRESHOLD_M = 1.0e-6" in source
    assert "ROTATION_THRESHOLD_RAD = 1.0e-5" in source


def test_render_ab_cli_defaults_to_zero_launch_overrides():
    runner = _regrasp_module()
    arguments = runner._parse_arguments(PROJECT_ROOT, [])
    assert arguments.nut_tooth_jitter_rtx_history is None
    assert arguments.nut_tooth_jitter_disable_fabric_scene_delegate is False
    assert runner._nut_tooth_jitter_render_extra_args(arguments) == []
    report = runner._nut_tooth_jitter_render_settings_report(
        arguments,
        FakeSettings(
            {
                runner.RTX_HISTORY_SETTING: 8,
                runner.FABRIC_SCENE_DELEGATE_SETTING: True,
            }
        ),
    )
    assert report["mode"] == "baseline"
    assert report["requested"] == {}
    assert report["actual"] == {
        runner.RTX_HISTORY_SETTING: 8,
        runner.FABRIC_SCENE_DELEGATE_SETTING: True,
    }
    assert report["exact_match"] is True


def test_render_ab_is_wired_through_launch_metrics_and_probe_report():
    source = REGRASP_PATH.read_text(encoding="utf-8")
    assert '"extra_args": render_extra_args' in source
    assert 'metrics["nut_tooth_jitter_render_ab"]' in source
    assert 'report["render_ab_launch"] = render_report' in source
    assert "carb.settings.get_settings()" in source
    assert "_require_exact_nut_tooth_jitter_render_settings(" in source


def test_render_ab_cli_modes_are_explicit_and_mutually_exclusive(tmp_path):
    runner = _regrasp_module()
    common = [
        "--gui",
        "--twist-probe",
        "--nut-tooth-jitter-output",
        str(tmp_path),
    ]
    rtx = runner._parse_arguments(
        PROJECT_ROOT,
        common + ["--nut-tooth-jitter-rtx-history", "512"],
    )
    assert runner._nut_tooth_jitter_render_extra_args(rtx) == [
        "--/rtx/scenedb/maxHistoryTransformCount=512"
    ]
    fabric = runner._parse_arguments(
        PROJECT_ROOT,
        common + ["--nut-tooth-jitter-disable-fabric-scene-delegate"],
    )
    assert runner._nut_tooth_jitter_render_extra_args(fabric) == [
        "--/app/useFabricSceneDelegate=false"
    ]
    with pytest.raises(SystemExit):
        runner._parse_arguments(
            PROJECT_ROOT,
            common
            + [
                "--nut-tooth-jitter-rtx-history",
                "512",
                "--nut-tooth-jitter-disable-fabric-scene-delegate",
            ],
        )
    with pytest.raises(SystemExit):
        runner._parse_arguments(
            PROJECT_ROOT,
            common + ["--nut-tooth-jitter-rtx-history", "256"],
        )


@pytest.mark.parametrize(
    "arguments",
    (
        ["--nut-tooth-jitter-rtx-history", "512"],
        ["--nut-tooth-jitter-disable-fabric-scene-delegate"],
        [
            "--twist-probe",
            "--nut-tooth-jitter-output",
            "/tmp/tooth-probe-test",
            "--nut-tooth-jitter-rtx-history",
            "512",
        ],
    ),
)
def test_render_ab_variants_require_gui_and_tooth_output(arguments):
    with pytest.raises(SystemExit):
        _regrasp_module()._parse_arguments(PROJECT_ROOT, arguments)


def test_post_launch_settings_report_is_exact_and_fail_closed(tmp_path):
    runner = _regrasp_module()
    arguments = runner._parse_arguments(
        PROJECT_ROOT,
        [
            "--gui",
            "--twist-probe",
            "--nut-tooth-jitter-output",
            str(tmp_path),
            "--nut-tooth-jitter-rtx-history",
            "512",
        ],
    )
    exact = runner._nut_tooth_jitter_render_settings_report(
        arguments,
        FakeSettings(
            {
                runner.RTX_HISTORY_SETTING: 512,
                runner.FABRIC_SCENE_DELEGATE_SETTING: True,
            }
        ),
    )
    assert exact == {
        "actual": {
            runner.RTX_HISTORY_SETTING: 512,
            runner.FABRIC_SCENE_DELEGATE_SETTING: True,
        },
        "extra_args": [
            "--/rtx/scenedb/maxHistoryTransformCount=512"
        ],
        "exact_match": True,
        "mismatches": [],
        "mode": "rtx_history_512",
        "requested": {runner.RTX_HISTORY_SETTING: 512},
        "validated_after_simulation_app_start": True,
    }
    runner._require_exact_nut_tooth_jitter_render_settings(exact)

    wrong_type = runner._nut_tooth_jitter_render_settings_report(
        arguments,
        FakeSettings(
            {
                runner.RTX_HISTORY_SETTING: 512.0,
                runner.FABRIC_SCENE_DELEGATE_SETTING: True,
            }
        ),
    )
    assert wrong_type["exact_match"] is False
    with pytest.raises(RuntimeError, match="do not exactly match"):
        runner._require_exact_nut_tooth_jitter_render_settings(wrong_type)


def test_fabric_disable_is_read_back_as_strict_bool(tmp_path):
    runner = _regrasp_module()
    arguments = runner._parse_arguments(
        PROJECT_ROOT,
        [
            "--gui",
            "--twist-probe",
            "--nut-tooth-jitter-output",
            str(tmp_path),
            "--nut-tooth-jitter-disable-fabric-scene-delegate",
        ],
    )
    exact = runner._nut_tooth_jitter_render_settings_report(
        arguments,
        FakeSettings(
            {
                runner.RTX_HISTORY_SETTING: 8,
                runner.FABRIC_SCENE_DELEGATE_SETTING: False,
            }
        ),
    )
    assert exact["requested"] == {
        runner.FABRIC_SCENE_DELEGATE_SETTING: False
    }
    assert exact["exact_match"] is True
    numeric_zero = runner._nut_tooth_jitter_render_settings_report(
        arguments,
        FakeSettings(
            {
                runner.RTX_HISTORY_SETTING: 8,
                runner.FABRIC_SCENE_DELEGATE_SETTING: 0,
            }
        ),
    )
    assert numeric_zero["exact_match"] is False


def test_verified_render_state_is_bound_into_probe_report(tmp_path):
    runner = _regrasp_module()
    initial = {"schema_version": "probe", "steps": 12}

    class FakeProbe:
        def finalize(self):
            return initial

    render = {"mode": "baseline", "exact_match": True}
    report = runner._finalize_tooth_probe(FakeProbe(), tmp_path, render)
    assert report["render_ab_launch"] == render
    saved = json.loads((tmp_path / "report.json").read_text())
    assert saved == report


def test_fast_shutdown_preserves_pass_and_failure_exit_codes():
    tree = ast.parse(REGRASP_PATH.read_text(encoding="utf-8"))
    close_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "simulation_app"
        and node.func.attr == "close"
    ]
    assert len(close_calls) == 1
    exit_keywords = [
        keyword
        for keyword in close_calls[0].keywords
        if keyword.arg == "exit_code"
    ]
    assert len(exit_keywords) == 1
    expression = ast.Expression(exit_keywords[0].value)
    ast.fix_missing_locations(expression)
    compiled = compile(expression, str(REGRASP_PATH), "eval")
    assert eval(compiled, {}, {"passed": True}) == 0
    assert eval(compiled, {}, {"passed": False}) == 1
