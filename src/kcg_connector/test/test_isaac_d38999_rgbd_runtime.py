"""Pure contracts for the reusable in-World D38999 RGB-D runtime."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import pytest

from kcg_connector.isaac_d38999_rgbd_runtime import (
    capture_d38999_rgbd_runtime,
    cleanup_rgbd_runtime_resources,
    endpoint_projection_records,
    ensure_d38999_rgbd_stage_prims,
    pause_timeline_for_rgbd_capture,
    restore_timeline_after_rgbd_capture,
    validate_real_endpoint_semantic_ids,
)


MODULE_PATH = (
    Path(__file__).parents[1]
    / "kcg_connector"
    / "isaac_d38999_rgbd_runtime.py"
)


def _source():
    return MODULE_PATH.read_text(encoding="utf-8")


def _top_level_import_roots(source):
    roots = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def _capture_with_bindings(bindings, output_dir=None):
    return capture_d38999_rgbd_runtime(
        bindings=bindings,
        simulation_app=None,
        world=None,
        stage=None,
        tabletop=None,
        rgbd=None,
        loose_prim=None,
        fixed_prim=None,
        body=None,
        output_dir=output_dir,
    )


class _TrackedResource:
    def __init__(self, name, calls, failing_methods=()):
        self.name = name
        self.calls = calls
        self.failing_methods = set(failing_methods)

    def _call(self, method):
        self.calls.append(f"{self.name}.{method}")
        if method in self.failing_methods:
            raise RuntimeError(f"{self.name} {method} failed")

    def detach(self):
        self._call("detach")

    def destroy(self):
        self._call("destroy")

    def clear(self):
        self._call("clear")

    def reset(self):
        self._call("reset")

    def remove(self):
        self._call("remove")


class _TimelineWorld:
    def __init__(self, playing):
        self.playing = playing
        self.play_calls = 0
        self.pause_calls = 0

    def is_playing(self):
        return self.playing

    def play(self):
        self.play_calls += 1
        self.playing = True

    def pause(self):
        self.pause_calls += 1
        self.playing = False


class _SimulationApp:
    def __init__(self):
        self.update_calls = 0

    def update(self):
        self.update_calls += 1


class _FakeAttribute:
    """Small USD-attribute double for persistent-stage contract tests."""

    def __init__(self, value=None, valid=True):
        self.value = value
        self.valid = valid

    def IsValid(self):
        return self.valid

    def Get(self):
        return self.value

    def Set(self, value):
        self.value = value
        return True


class _FakePrim:
    def __init__(
        self,
        path,
        type_name,
        *,
        valid=True,
        position=(0.0, 0.0, 0.0),
        forward=(0.0, 0.0, -1.0),
    ):
        self.path = path
        self.type_name = type_name
        self.valid = valid
        self.attributes = {}
        self.position = np.asarray(position, dtype=np.float64)
        self.forward = np.asarray(forward, dtype=np.float64)

    def IsValid(self):
        return self.valid

    def GetTypeName(self):
        return self.type_name

    def GetPath(self):
        return self.path

    def GetAttribute(self, name):
        return self.attributes.get(name, _FakeAttribute(valid=False))


class _FakeStage:
    def __init__(self):
        self.prims = {}

    def GetPrimAtPath(self, path):
        return self.prims.get(path, _FakePrim(path, "", valid=False))

    def define(self, path, type_name):
        prim = _FakePrim(path, type_name)
        self.prims[path] = prim
        return prim


class _FakeMatrix:
    def __init__(self, prim):
        self.prim = prim

    def ExtractTranslation(self):
        return self.prim.position

    def TransformDir(self, _direction):
        return self.prim.forward


class _FakeSchema:
    type_name = ""

    def __init__(self, prim):
        self.prim = prim.GetPrim() if hasattr(prim, "GetPrim") else prim

    @classmethod
    def Define(cls, stage, path):
        return cls(stage.define(path, cls.type_name))

    def GetPrim(self):
        return self.prim

    def _create(self, name, value):
        attribute = self.prim.attributes.setdefault(
            name, _FakeAttribute()
        )
        attribute.Set(value)
        return attribute


class _FakeXform(_FakeSchema):
    type_name = "Xform"


class _FakeDomeLight(_FakeSchema):
    type_name = "DomeLight"

    def CreateIntensityAttr(self, value):
        # Match the real UsdLux prim representation rather than the schema
        # method's abbreviated name.
        return self._create("inputs:intensity", value)

    def CreateColorAttr(self, value):
        return self._create("inputs:color", value)


class _FakeDistantLight(_FakeDomeLight):
    type_name = "DistantLight"


class _FakeCameraSchema(_FakeSchema):
    type_name = "Camera"

    def CreateVerticalApertureAttr(self):
        return self.prim.attributes.setdefault(
            "verticalAperture", _FakeAttribute()
        )


class _FakeXformable(_FakeSchema):
    def AddRotateXYZOp(self):
        self.prim.rotate_op_add_count = getattr(
            self.prim, "rotate_op_add_count", 0
        ) + 1
        return self.prim.attributes.setdefault(
            "xformOp:rotateXYZ", _FakeAttribute()
        )

    def ComputeLocalToWorldTransform(self, _time_code):
        return _FakeMatrix(self.prim)


class _FakeCameraCreator:
    def __init__(self, stage):
        self.stage = stage
        self.calls = 0

    def camera(self, **kwargs):
        self.calls += 1
        path = kwargs["parent"] + "/" + kwargs["name"]
        eye = np.asarray(kwargs["position"], dtype=np.float64)
        target = np.asarray(kwargs["look_at"], dtype=np.float64)
        forward = target - eye
        forward /= np.linalg.norm(forward)
        prim = self.stage.define(path, "Camera")
        prim.position = eye
        prim.forward = forward
        prim.attributes.update(
            {
                "clippingRange": _FakeAttribute(kwargs["clipping_range"]),
                "focalLength": _FakeAttribute(kwargs["focal_length"]),
                "horizontalAperture": _FakeAttribute(
                    kwargs["horizontal_aperture"]
                ),
            }
        )
        return prim


def _persistent_stage_fixture():
    stage = _FakeStage()
    creator = _FakeCameraCreator(stage)
    bindings = {
        "Gf": SimpleNamespace(
            Vec3f=lambda *values: values,
            Vec3d=lambda *values: values,
        ),
        "Usd": SimpleNamespace(
            TimeCode=SimpleNamespace(Default=lambda: object())
        ),
        "UsdGeom": SimpleNamespace(
            Camera=_FakeCameraSchema,
            Xform=_FakeXform,
            Xformable=_FakeXformable,
        ),
        "UsdLux": SimpleNamespace(
            DistantLight=_FakeDistantLight,
            DomeLight=_FakeDomeLight,
        ),
        "rep": SimpleNamespace(
            functional=SimpleNamespace(create=creator)
        ),
    }
    tabletop = SimpleNamespace(
        world=SimpleNamespace(root_prim_path="/World/Demo"),
        render=SimpleNamespace(
            dome_light_intensity=850.0,
            dome_light_color_rgb=(0.8, 0.9, 1.0),
            key_light_intensity=1300.0,
            key_light_color_rgb=(1.0, 0.95, 0.9),
            key_light_rotation_degrees_xyz=(-35.0, 25.0, 15.0),
        ),
    )
    rgbd = SimpleNamespace(
        camera=SimpleNamespace(
            prim_path="/World/Demo/GlobalRgbdCamera",
            eye_m=(0.72, -0.44, 0.82),
            target_m=(0.18, 0.02, 0.41),
            resolution=(640, 480),
        )
    )
    return stage, creator, bindings, tabletop, rgbd


def test_runtime_top_level_imports_no_isaac_or_image_bindings():
    roots = _top_level_import_roots(_source())
    assert roots.isdisjoint({"isaacsim", "omni", "pxr", "PIL"})


def test_two_same_stage_capture_setups_reuse_fixed_camera_and_lights():
    """The stage half of two injected captures must not mint ``_01`` prims."""
    stage, creator, bindings, tabletop, rgbd = _persistent_stage_fixture()

    first_camera, first = ensure_d38999_rgbd_stage_prims(
        bindings=bindings,
        stage=stage,
        tabletop=tabletop,
        rgbd=rgbd,
    )
    key_prim = stage.prims["/World/Demo/RgbdLighting/Key"]
    second_camera, second = ensure_d38999_rgbd_stage_prims(
        bindings=bindings,
        stage=stage,
        tabletop=tabletop,
        rgbd=rgbd,
    )

    assert first_camera is second_camera
    assert set(first["prim_lifecycle"].values()) == {"created"}
    assert set(second["prim_lifecycle"].values()) == {"reused"}
    assert creator.calls == 1
    assert key_prim.rotate_op_add_count == 1
    assert not any("_01" in path for path in stage.prims)
    assert sorted(stage.prims) == [
        "/World/Demo/GlobalRgbdCamera",
        "/World/Demo/RgbdLighting",
        "/World/Demo/RgbdLighting/Fill",
        "/World/Demo/RgbdLighting/Key",
    ]


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda stage: setattr(
                stage.prims["/World/Demo/RgbdLighting/Fill"],
                "type_name",
                "SphereLight",
            ),
            "expected 'DomeLight'",
        ),
        (
            lambda stage: stage.prims[
                "/World/Demo/GlobalRgbdCamera"
            ].attributes["focalLength"].Set(35.0),
            "focalLength differs from contract",
        ),
        (
            lambda stage: setattr(
                stage.prims["/World/Demo/GlobalRgbdCamera"],
                "forward",
                np.asarray((1.0, 0.0, 0.0)),
            ),
            "optical axis differs from contract",
        ),
    ),
)
def test_reused_stage_prims_fail_closed_on_type_or_parameter_drift(
    mutate, message
):
    stage, creator, bindings, tabletop, rgbd = _persistent_stage_fixture()
    ensure_d38999_rgbd_stage_prims(
        bindings=bindings,
        stage=stage,
        tabletop=tabletop,
        rgbd=rgbd,
    )
    mutate(stage)

    with pytest.raises(RuntimeError, match=message):
        ensure_d38999_rgbd_stage_prims(
            bindings=bindings,
            stage=stage,
            tabletop=tabletop,
            rgbd=rgbd,
        )
    assert creator.calls == 1


def test_endpoint_projection_records_enforce_strict_margin():
    records = endpoint_projection_records(
        {
            "loose_plug": (16.0, 16.0),
            "fixed_receptacle": (623.999, 463.999),
        },
        (640, 480),
    )
    assert records == {
        "loose_plug": {
            "in_frame": True,
            "margin_px": 16,
            "uv_px": [16.0, 16.0],
        },
        "fixed_receptacle": {
            "in_frame": True,
            "margin_px": 16,
            "uv_px": [623.999, 463.999],
        },
    }
    outside = endpoint_projection_records(
        {"right_edge": (624.0, 100.0), "top_edge": (100.0, 464.0)},
        (640, 480),
    )
    assert outside["right_edge"]["in_frame"] is False
    assert outside["top_edge"]["in_frame"] is False


@pytest.mark.parametrize(
    ("endpoint_uv", "resolution", "message"),
    (
        ({"loose": (1.0, 2.0)}, (640,), "width and height"),
        ({"loose": (1.0, 2.0)}, (32, 480), "too small"),
        ({"loose": (1.0,)}, (640, 480), "u and v"),
        ({"loose": (float("nan"), 2.0)}, (640, 480), "finite"),
    ),
)
def test_endpoint_projection_records_fail_closed(
    endpoint_uv, resolution, message
):
    with pytest.raises(ValueError, match=message):
        endpoint_projection_records(endpoint_uv, resolution)


def test_endpoint_semantic_ids_must_be_real_and_observed():
    result = validate_real_endpoint_semantic_ids(
        {"loose_plug": (3,), "fixed_receptacle": (2, 4)},
        (0, 1, 2, 3, 4),
    )
    assert result == {
        "loose_plug": (3,),
        "fixed_receptacle": (2, 4),
    }

    with pytest.raises(RuntimeError, match="BACKGROUND/UNLABELLED"):
        validate_real_endpoint_semantic_ids(
            {"loose_plug": (1,)}, (0, 1)
        )
    with pytest.raises(RuntimeError, match=r"no rendered pixels: \[4\]"):
        validate_real_endpoint_semantic_ids(
            {"fixed_receptacle": (2, 4)}, (0, 1, 2)
        )


def test_cleanup_releases_only_owned_render_resources_in_order():
    calls = []
    first = _TrackedResource("first", calls)
    second = _TrackedResource("second", calls)
    camera = _TrackedResource("camera", calls)
    render_product = _TrackedResource("render_product", calls)

    result = cleanup_rgbd_runtime_resources(
        [first, second], camera, render_product
    )

    assert calls == [
        "second.detach",
        "first.detach",
        "camera.destroy",
        "render_product.destroy",
    ]
    assert result == {
        "annotator_detach_count": 2,
        "camera_destroyed": True,
        "errors": [],
        "render_product_destroyed": True,
        "resources_released": True,
        "scene_cleared": False,
        "stage_prims_removed": 0,
        "world_reset": False,
    }
    assert not any(
        call.endswith((".clear", ".reset", ".remove")) for call in calls
    )


def test_cleanup_is_best_effort_and_reports_each_exception():
    calls = []
    good = _TrackedResource("good", calls)
    bad = _TrackedResource("bad", calls, {"detach"})
    camera = _TrackedResource("camera", calls, {"destroy"})
    render_product = _TrackedResource(
        "render_product", calls, {"destroy"}
    )

    result = cleanup_rgbd_runtime_resources(
        [good, bad], camera, render_product
    )

    assert calls == [
        "bad.detach",
        "good.detach",
        "camera.destroy",
        "render_product.destroy",
    ]
    assert result["annotator_detach_count"] == 1
    assert result["camera_destroyed"] is False
    assert result["render_product_destroyed"] is False
    assert result["resources_released"] is False
    assert result["errors"] == [
        "annotator.detach: RuntimeError: bad detach failed",
        "camera.destroy: RuntimeError: camera destroy failed",
        (
            "render_product.destroy: RuntimeError: "
            "render_product destroy failed"
        ),
    ]


def test_timeline_guard_restores_a_running_caller_after_api_drift():
    world = _TimelineWorld(False)
    app = _SimulationApp()
    result = restore_timeline_after_rgbd_capture(
        world, app, was_playing=True
    )
    assert result == {
        "playing_after_cleanup": False,
        "playing_after_restore": True,
        "playing_before_capture": True,
        "restore_attempted": True,
        "restored": True,
    }
    assert world.play_calls == 1
    assert app.update_calls == 1


def test_timeline_guard_preserves_an_intentionally_paused_caller():
    world = _TimelineWorld(False)
    app = _SimulationApp()
    result = restore_timeline_after_rgbd_capture(
        world, app, was_playing=False
    )
    assert result["restore_attempted"] is False
    assert result["restored"] is True
    assert world.play_calls == 0
    assert app.update_calls == 0


def test_timeline_guard_repauses_a_caller_started_by_api_drift():
    world = _TimelineWorld(True)
    app = _SimulationApp()
    result = restore_timeline_after_rgbd_capture(
        world, app, was_playing=False
    )
    assert result == {
        "playing_after_cleanup": True,
        "playing_after_restore": False,
        "playing_before_capture": False,
        "restore_attempted": True,
        "restored": True,
    }
    assert world.pause_calls == 1
    assert app.update_calls == 1


def test_capture_pause_freezes_a_running_world_without_physics_step():
    world = _TimelineWorld(True)
    app = _SimulationApp()
    result = pause_timeline_for_rgbd_capture(
        world, app, was_playing=True
    )
    assert result == {
        "pause_attempted": True,
        "paused_for_capture": True,
        "playing_during_capture": False,
    }
    assert world.pause_calls == 1
    assert app.update_calls == 1


def test_capture_pause_preserves_an_already_paused_world():
    world = _TimelineWorld(False)
    app = _SimulationApp()
    result = pause_timeline_for_rgbd_capture(
        world, app, was_playing=False
    )
    assert result["pause_attempted"] is False
    assert result["paused_for_capture"] is True
    assert world.pause_calls == 0
    assert app.update_calls == 0


def test_runtime_uses_camera_only_for_projection_without_frame_callback():
    source = _source()
    assert "camera.initialize(" not in source
    assert "pause_timeline=True" in source
    assert "pause_timeline=False" not in source


def test_capture_rejects_missing_bindings_before_touching_runtime_objects():
    with pytest.raises(ValueError, match="missing Isaac RGB-D runtime") as exc:
        _capture_with_bindings({})
    message = str(exc.value)
    for name in (
        "Camera",
        "Gf",
        "Usd",
        "UsdGeom",
        "UsdLux",
        "add_labels",
        "get_labels",
        "rep",
    ):
        assert name in message
    assert "Image" not in message


def test_capture_requires_image_binding_only_when_writing_artifacts():
    incomplete = {
        name: object()
        for name in (
            "Camera",
            "Gf",
            "Usd",
            "UsdGeom",
            "UsdLux",
            "add_labels",
            "get_labels",
            "rep",
        )
    }
    with pytest.raises(ValueError, match=r"missing.*Image"):
        _capture_with_bindings(incomplete, output_dir="unused")


def test_runtime_source_forbids_world_and_control_mutations():
    source = _source()
    for forbidden in (
        "World.clear_instance",
        "world.reset",
        "set_world_pose",
        "apply_action",
    ):
        assert forbidden not in source

    tree = ast.parse(source)
    cleanup = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "cleanup_rgbd_runtime_resources"
    )
    called_attributes = {
        node.func.attr
        for node in ast.walk(cleanup)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
    }
    assert called_attributes <= {"detach", "destroy", "append"}


def test_replicator_warmup_keeps_the_explicit_pause_and_zero_delta():
    source = _source()
    warmup = source.split("for _ in range(rgbd.camera.warmup_frames):", 1)[
        1
    ].split("projection_body_position", 1)[0]
    assert "pause_timeline=True" in warmup
    assert "delta_time=0.0" in warmup
    assert "world.step" not in warmup
