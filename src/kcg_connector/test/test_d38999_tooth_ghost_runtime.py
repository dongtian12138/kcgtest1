"""CPU-only tests for the prepared-tooth render ghost lifecycle."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATH = PACKAGE_ROOT / "isaac/d38999_tooth_ghost_runtime.py"
RUNNER_PATH = PACKAGE_ROOT / "isaac/d38999_nut_regrasp_smoke.py"


def _runtime_module():
    spec = importlib.util.spec_from_file_location(
        "d38999_tooth_ghost_runtime_test", RUNTIME_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _runner_module():
    isaac_directory = str(RUNNER_PATH.parent)
    sys.path.insert(0, isaac_directory)
    try:
        spec = importlib.util.spec_from_file_location(
            "d38999_nut_regrasp_ghost_test", RUNNER_PATH
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(isaac_directory)


class _FakePath:
    def __init__(self, value):
        self.value = str(value)

    def AppendProperty(self, name):
        return _FakePath(f"{self.value}.{name}")

    def __str__(self):
        return self.value


class _FakeSdf:
    Path = _FakePath


class _FakeLayer:
    anonymous = True
    identifier = "anon:test-session"

    def __init__(self):
        self.properties = set()

    def GetPropertyAtPath(self, path):
        return object() if str(path) in self.properties else None


class _FakeEditTarget:
    def __init__(self, layer):
        self._layer = layer

    def GetLayer(self):
        return self._layer


class _FakePrim:
    def __init__(self, stage, path, parent=None, *, imageable=True):
        self.stage = stage
        self.path = path
        self.parent = parent
        self.imageable = imageable
        self.session_visibility = None

    def GetPath(self):
        return self.path

    def GetName(self):
        return self.path.rsplit("/", 1)[-1]

    def GetParent(self):
        return self.parent

    def RemoveProperty(self, name):
        stage = self.stage
        assert stage.GetEditTarget().GetLayer() is stage.session
        stage.session.properties.discard(f"{self.path}.{name}")
        if name == "visibility":
            self.session_visibility = None


class _FakeStage:
    def __init__(self):
        self.session = _FakeLayer()
        self.root = _FakeLayer()
        self.edit_target = _FakeEditTarget(self.root)
        self.prims = []

    def Traverse(self):
        return list(self.prims)

    def GetSessionLayer(self):
        return self.session

    def GetEditTarget(self):
        return self.edit_target

    def SetEditTarget(self, value):
        self.edit_target = (
            value
            if isinstance(value, _FakeEditTarget)
            else _FakeEditTarget(value)
        )


class _FakeAttribute:
    def __init__(self, prim):
        self.prim = prim

    def __bool__(self):
        return True

    @property
    def _property_path(self):
        return f"{self.prim.path}.visibility"

    def Set(self, value):
        stage = self.prim.stage
        assert stage.GetEditTarget().GetLayer() is stage.session
        stage.session.properties.add(self._property_path)
        self.prim.session_visibility = str(value)

    def Clear(self):
        stage = self.prim.stage
        assert stage.GetEditTarget().GetLayer() is stage.session
        stage.session.properties.discard(self._property_path)
        self.prim.session_visibility = None


class _FakeImageable:
    def __init__(self, prim):
        self.prim = prim

    def __bool__(self):
        return self.prim.imageable

    def ComputeVisibility(self):
        current = self.prim
        while current is not None:
            if current.session_visibility == "invisible":
                return "invisible"
            current = current.parent
        return "inherited"

    def CreateVisibilityAttr(self):
        return _FakeAttribute(self.prim)

    def GetVisibilityAttr(self):
        return _FakeAttribute(self.prim)


class _FakeTokens:
    invisible = "invisible"


class _FakeUsdGeom:
    Imageable = _FakeImageable
    Tokens = _FakeTokens


def _stage_with_fingers():
    stage = _FakeStage()
    robot = _FakePrim(stage, "/World/HandArm")
    handbase = _FakePrim(stage, "/World/HandArm/chain/handbase_link", robot)
    stage.prims.extend((robot, handbase))
    for name in ("f1Link1", "f2Link1", "f3Link1"):
        root = _FakePrim(stage, f"{handbase.path}/{name}", handbase)
        visual = _FakePrim(stage, f"{root.path}/{name}", root)
        collision = _FakePrim(stage, f"{root.path}/{name}_convex", root)
        stage.prims.extend((root, visual, collision))
    return stage


def _physics_report():
    segment = {
        "contact_counterparts": ["/World/HandArm/finger"],
        "contact_records": 2,
        "maximum_contact_impulse_norm": 0.02,
        "maximum_local_rotation_error_rad": 0.0,
        "maximum_local_translation_error_m": 0.0,
        "maximum_parent_relative_rotation_error_rad": 0.0,
        "maximum_parent_relative_translation_error_m": 0.0,
        "minimum_contact_separation_m": -1.0e-5,
    }
    return {
        "schema_version": "kcg_d38999_nut_tooth_jitter_probe_v1",
        "anomaly_steps": 0,
        "phase_steps": {"q7_twist_probe_motion": 2},
        "segment_aggregate": {
            f"Segment_{index:02d}": dict(segment) for index in range(24)
        },
        "steps": 2,
        "thresholds": {"rotation_rad": 1.0e-5, "translation_m": 1.0e-6},
    }


def _write_bound_capture(module, root):
    physics = root / "physics"
    capture = root / "capture"
    physics.mkdir()
    capture.mkdir()
    report = physics / "report.json"
    summary = physics / "summary.csv"
    report.write_text(json.dumps(_physics_report()), encoding="utf-8")
    with summary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "global_step",
                "phase",
                "phase_step",
                "parent_angular_speed_rad_s",
                "segment_contact_records",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "global_step": 1,
                "phase": "q7_twist_probe_motion",
                "phase_step": 1,
                "parent_angular_speed_rad_s": 0.2,
                "segment_contact_records": 4,
            }
        )
    source_hash = "a" * 64
    manifest = capture / "video_capture_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": module.CAPTURE_SCHEMA_VERSION,
                "passed": True,
                "capture_source": {
                    "unchanged_during_capture": True,
                    "sha256_at_import": source_hash,
                    "sha256_at_start": source_hash,
                    "sha256_at_finalize": source_hash,
                },
                "cleanup": {"object_pose_writes": 0},
                "physics_evidence": {
                    "report_sha256": module.sha256_file(report),
                    "summary_sha256": module.sha256_file(summary),
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest, report, summary


def test_runtime_authors_only_three_inherited_visibility_opinions_and_restores(
    tmp_path,
):
    module = _runtime_module()
    stage = _stage_with_fingers()
    runtime = module.D38999ToothGhostRuntime(
        stage=stage,
        robot_root=module.EXPECTED_ROBOT_ROOT,
        output_directory=tmp_path / "ghost",
        runner_source_path=RUNNER_PATH,
        Sdf=_FakeSdf,
        UsdGeom=_FakeUsdGeom,
    )
    report = runtime.active_report()
    assert report["active"] is True
    assert report["authoring"]["properties"] == ["visibility"]
    assert len(stage.session.properties) == 3
    assert all(
        _FakeImageable(prim).ComputeVisibility() == "invisible"
        for prim in stage.prims
        if "/f" in prim.path
    )
    cleanup = runtime.restore()
    assert cleanup["session_visibility_opinions_removed"] is True
    assert cleanup[
        "effective_visibility_restored_to_pre_author_state"
    ] is True
    assert stage.session.properties == set()


def test_success_manifest_binds_runtime_sidecar_capture_and_physics(tmp_path):
    module = _runtime_module()
    stage = _stage_with_fingers()
    runtime = module.D38999ToothGhostRuntime(
        stage=stage,
        robot_root=module.EXPECTED_ROBOT_ROOT,
        output_directory=tmp_path / "ghost",
        runner_source_path=RUNNER_PATH,
        Sdf=_FakeSdf,
        UsdGeom=_FakeUsdGeom,
    )
    capture, report, summary = _write_bound_capture(module, tmp_path)
    result = runtime.finalize(
        capture_manifest_path=capture,
        physics_report_path=report,
        physics_summary_path=summary,
    )
    assert result["report"]["passed"] is True
    assert result["report"]["mutation_audit"] == {
        "collision_api_writes": 0,
        "material_writes": 0,
        "object_pose_writes": 0,
        "physics_api_writes": 0,
        "xform_writes": 0,
    }
    manifest = json.loads(
        (tmp_path / "ghost/manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["sources"]["runtime"]["sha256"] == module.sha256_file(
        RUNTIME_PATH
    )
    assert manifest["outputs"]["visibility_sidecar"]["sha256"] == (
        module.sha256_file(tmp_path / "ghost/visibility_sidecar.json")
    )
    assert runtime.restored is True


def test_preexisting_session_visibility_is_rejected_without_overwrite(
    tmp_path,
):
    module = _runtime_module()
    stage = _stage_with_fingers()
    first_root = module.discover_finger_roots(stage)[0]
    stage.session.properties.add(f"{first_root.path}.visibility")
    with pytest.raises(RuntimeError, match="pre-existing"):
        module.D38999ToothGhostRuntime(
            stage=stage,
            robot_root=module.EXPECTED_ROBOT_ROOT,
            output_directory=tmp_path / "ghost",
            runner_source_path=RUNNER_PATH,
            Sdf=_FakeSdf,
            UsdGeom=_FakeUsdGeom,
        )
    assert stage.session.properties == {f"{first_root.path}.visibility"}


def test_runner_ghost_cli_is_default_disabled_and_baseline_only(tmp_path):
    runner = _runner_module()
    repository = PACKAGE_ROOT.parents[1]
    default = runner._parse_arguments(repository, [])
    assert default.nut_tooth_ghost_fingers_output is None
    common = [
        "--gui",
        "--twist-probe",
        "--nut-tooth-jitter-output",
        str(tmp_path / "physics"),
        "--nut-tooth-sync-capture-output",
        str(tmp_path / "capture"),
        "--nut-tooth-ghost-fingers-output",
        str(tmp_path / "ghost"),
    ]
    parsed = runner._parse_arguments(repository, common)
    assert parsed.nut_tooth_ghost_fingers_output == str(tmp_path / "ghost")
    for variant in (
        ["--nut-tooth-jitter-rtx-history", "512"],
        ["--nut-tooth-jitter-normalize-segment00-op"],
        ["--rewind-probe"],
    ):
        with pytest.raises(SystemExit):
            runner._parse_arguments(repository, common + variant)


def test_runner_lifecycle_order_and_finally_restore():
    source = RUNNER_PATH.read_text(encoding="utf-8")
    construction = source.index(
        "tooth_ghost_runtime = D38999ToothGhostRuntime("
    )
    reset = source.index("\n        world.reset()")
    capture_finalize = source.index("tooth_sync_capture.finalize(")
    ghost_finalize = source.index(
        "ghost_result = tooth_ghost_runtime.finalize("
    )
    assert construction < reset
    assert capture_finalize < ghost_finalize
    final_block = source.rsplit("finally:", 1)[1]
    assert "tooth_ghost_runtime.restore()" in final_block
    # Static source audit complements the runtime mutation counters: the
    # runner never passes transform or physics bindings into the ghost helper.
    constructor_source = source[construction:reset]
    assert "UsdPhysics" not in constructor_source
    assert "PhysxSchema" not in constructor_source


def test_runtime_source_has_one_visibility_value_write_and_no_physics_api():
    source = RUNTIME_PATH.read_text(encoding="utf-8")
    # SetEditTarget is not a USD value write.  The only Attribute.Set call is
    # immediately chained from CreateVisibilityAttr on a finger root.
    assert source.count(".Set(\n") == 1
    assert "CreateVisibilityAttr().Set(" in source
    for forbidden in (
        "UsdPhysics",
        "PhysxSchema",
        "CollisionAPI",
        "XformOp",
        "set_world_pose",
        "set_local_pose",
        "set_joint_positions",
    ):
        assert forbidden not in source
