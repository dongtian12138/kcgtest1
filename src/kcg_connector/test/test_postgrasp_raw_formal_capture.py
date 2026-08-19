from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

import kcg_connector.isaac_d38999_rgbd_runtime as runtime


MODULE_PATH = (
    Path(__file__).parents[1]
    / "kcg_connector"
    / "isaac_d38999_rgbd_runtime.py"
)


class _Annotator:
    def __init__(self, data):
        self.data = data

    def attach(self, paths):
        self.paths = paths

    def detach(self):
        return None

    def get_data(self):
        return self.data


class _FakeRep:
    def __init__(self, rgb, depth, semantic_raises=True):
        self.rgb = rgb
        self.depth = depth
        self.semantic_called = False
        self.semantic_raises = semantic_raises

    def get(self, name, **kwargs):
        if name == "semantic_segmentation":
            self.semantic_called = True
            if self.semantic_raises:
                raise AssertionError("semantic annotator requested by formal capture")
            return _Annotator(np.zeros((12, 16), dtype=np.uint8))
        if name == "rgb":
            return _Annotator(self.rgb)
        if name == "distance_to_image_plane":
            return _Annotator(self.depth)
        raise AssertionError(name)


class _FakeCamera:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.world_pose_calls = 0

    def get_clipping_range(self):
        return (0.1, 10.0)

    def get_focal_length(self):
        return 24.0

    def get_horizontal_aperture(self):
        return 20.955

    def get_intrinsics_matrix(self):
        return np.eye(3)

    def get_world_pose(self):
        self.world_pose_calls += 1
        raise AssertionError("formal raw capture must not read camera world pose")

    def destroy(self):
        return None


def _make_rep(width=16, height=12, semantic_raises=True, inf_depth=False):
    rgb = np.zeros((height, width, 4), dtype=np.uint8)
    rgb[:, :, :3] = 120
    depth = np.full((height, width), 0.5, dtype=np.float32)
    if inf_depth:
        depth[0, :] = np.inf
    rep = _FakeRep(rgb, depth, semantic_raises=semantic_raises)
    rep.AnnotatorRegistry = SimpleNamespace(get_annotator=rep.get)
    rep.orchestrator = SimpleNamespace(step=lambda **kwargs: None)
    rep.create = SimpleNamespace(
        render_product=lambda prim, res, name: SimpleNamespace(path="/rp")
    )
    return rep, rgb, depth


def test_raw_formal_capture_never_requests_semantic_or_camera_truth(monkeypatch, tmp_path):
    rep, rgb, depth = _make_rep()
    fake_camera = _FakeCamera()
    monkeypatch.setattr(
        runtime,
        "ensure_d38999_rgbd_stage_prims",
        lambda **kwargs: ("/World/Camera", {"camera": "reused"}),
    )
    monkeypatch.setattr(
        runtime,
        "pause_timeline_for_rgbd_capture",
        lambda *args, **kwargs: {"paused_for_capture": True},
    )
    monkeypatch.setattr(
        runtime,
        "restore_timeline_after_rgbd_capture",
        lambda *args, **kwargs: {"restored": True},
    )
    monkeypatch.setattr(
        runtime,
        "cleanup_rgbd_runtime_resources",
        lambda *args, **kwargs: {"resources_released": True},
    )
    bindings = {
        "Camera": lambda **kwargs: fake_camera,
        "Gf": SimpleNamespace(),
        "Usd": SimpleNamespace(),
        "UsdGeom": SimpleNamespace(),
        "UsdLux": SimpleNamespace(),
        "rep": rep,
        "Image": Image,
    }
    world = SimpleNamespace(is_playing=lambda: True, pause=lambda: None)
    sim_app = SimpleNamespace(update=lambda: None)
    stage = SimpleNamespace()
    tabletop = SimpleNamespace()
    camera = SimpleNamespace(
        prim_path="/World/Camera",
        frame_id="raw",
        eye_m=(0.1, 0.0, 0.2),
        target_m=(0.0, 0.0, 0.0),
        resolution=(16, 12),
        frequency_hz=30,
        warmup_frames=1,
    )
    rgbd = SimpleNamespace(
        camera=camera,
        output=SimpleNamespace(
            rgb_filename="rgb.png", depth_numpy_filename="depth.npy"
        ),
    )
    capture = runtime.capture_d38999_rgbd_raw_formal(
        bindings=bindings,
        simulation_app=sim_app,
        world=world,
        stage=stage,
        tabletop=tabletop,
        rgbd=rgbd,
        output_dir=tmp_path / "raw",
    )
    assert capture.passed is True
    assert rep.semantic_called is False
    assert fake_camera.world_pose_calls == 0
    assert capture.metrics["semantic_annotator_used"] is False
    assert capture.metrics["endpoint_truth_read"] is False
    assert (tmp_path / "raw" / "rgb.png").is_file()
    assert (tmp_path / "raw" / "depth.npy").is_file()


def test_semantic_service_failure_does_not_change_raw_formal_archive(monkeypatch, tmp_path):
    captures = []
    for semantic_raises in (True, False):
        rep, rgb, depth = _make_rep(semantic_raises=semantic_raises)
        fake_camera = _FakeCamera()
        monkeypatch.setattr(
            runtime,
            "ensure_d38999_rgbd_stage_prims",
            lambda **kwargs: ("/World/Camera", {"camera": "reused"}),
        )
        monkeypatch.setattr(
            runtime,
            "pause_timeline_for_rgbd_capture",
            lambda *args, **kwargs: {"paused_for_capture": True},
        )
        monkeypatch.setattr(
            runtime,
            "restore_timeline_after_rgbd_capture",
            lambda *args, **kwargs: {"restored": True},
        )
        monkeypatch.setattr(
            runtime,
            "cleanup_rgbd_runtime_resources",
            lambda *args, **kwargs: {"resources_released": True},
        )
        bindings = {
            "Camera": lambda **kwargs: fake_camera,
            "Gf": SimpleNamespace(),
            "Usd": SimpleNamespace(),
            "UsdGeom": SimpleNamespace(),
            "UsdLux": SimpleNamespace(),
            "rep": rep,
            "Image": Image,
        }
        world = SimpleNamespace(is_playing=lambda: True, pause=lambda: None)
        sim_app = SimpleNamespace(update=lambda: None)
        camera = SimpleNamespace(
            prim_path="/World/Camera", frame_id="raw", eye_m=(0.1, 0.0, 0.2),
            target_m=(0.0, 0.0, 0.0), resolution=(16, 12), frequency_hz=30,
            warmup_frames=1,
        )
        rgbd = SimpleNamespace(
            camera=camera,
            output=SimpleNamespace(rgb_filename="rgb.png", depth_numpy_filename="depth.npy"),
        )
        capture = runtime.capture_d38999_rgbd_raw_formal(
            bindings=bindings, simulation_app=sim_app, world=world, stage=SimpleNamespace(),
            tabletop=SimpleNamespace(), rgbd=rgbd,
            output_dir=tmp_path / ("raw_" + str(semantic_raises)),
        )
        captures.append(capture)
    assert np.array_equal(captures[0].rgb, captures[1].rgb)
    assert np.array_equal(captures[0].depth, captures[1].depth)
    assert captures[0].metrics["semantic_annotator_used"] is False
    assert captures[1].metrics["semantic_annotator_used"] is False


def test_raw_formal_accepts_inf_background_and_preserves_depth_array(monkeypatch, tmp_path):
    rep, rgb, depth = _make_rep(inf_depth=True)
    fake_camera = _FakeCamera()
    monkeypatch.setattr(
        runtime,
        "ensure_d38999_rgbd_stage_prims",
        lambda **kwargs: ("/World/Camera", {"camera": "reused"}),
    )
    monkeypatch.setattr(
        runtime,
        "pause_timeline_for_rgbd_capture",
        lambda *args, **kwargs: {"paused_for_capture": True},
    )
    monkeypatch.setattr(
        runtime,
        "restore_timeline_after_rgbd_capture",
        lambda *args, **kwargs: {"restored": True},
    )
    monkeypatch.setattr(
        runtime,
        "cleanup_rgbd_runtime_resources",
        lambda *args, **kwargs: {"resources_released": True},
    )
    bindings = {
        "Camera": lambda **kwargs: fake_camera,
        "Gf": SimpleNamespace(),
        "Usd": SimpleNamespace(),
        "UsdGeom": SimpleNamespace(),
        "UsdLux": SimpleNamespace(),
        "rep": rep,
        "Image": Image,
    }
    world = SimpleNamespace(is_playing=lambda: True, pause=lambda: None)
    camera = SimpleNamespace(
        prim_path="/World/Camera", frame_id="raw", eye_m=(0.1, 0.0, 0.2),
        target_m=(0.0, 0.0, 0.0), resolution=(16, 12), frequency_hz=30,
        warmup_frames=1,
    )
    rgbd = SimpleNamespace(
        camera=camera,
        output=SimpleNamespace(rgb_filename="rgb.png", depth_numpy_filename="depth.npy"),
    )
    capture = runtime.capture_d38999_rgbd_raw_formal(
        bindings=bindings, simulation_app=SimpleNamespace(update=lambda: None),
        world=world, stage=SimpleNamespace(), tabletop=SimpleNamespace(),
        rgbd=rgbd, output_dir=tmp_path / "raw_inf",
    )
    assert capture.passed is True
    assert np.isinf(capture.depth[0, :]).all()
    assert capture.metrics["camera_frame_diagnostics"]["inf_depth_pixels_preserved"] > 0
    saved = np.load(tmp_path / "raw_inf" / "depth.npy")
    assert np.isinf(saved[0, :]).all()


def test_raw_formal_source_has_no_endpoint_truth_arguments():
    source = MODULE_PATH.read_text(encoding="utf-8")
    raw = source.split("def capture_d38999_rgbd_raw_formal(", 1)[1].split(
        "def ", 1
    )[0]
    assert "loose_prim" not in raw
    assert "fixed_prim" not in raw
    assert "body" not in raw
    assert '"semantic_segmentation"' not in raw
    assert "add_labels" not in raw
