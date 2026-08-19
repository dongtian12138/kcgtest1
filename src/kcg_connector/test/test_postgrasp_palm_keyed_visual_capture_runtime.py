"""CPU/static tests for the thin existing-Palm-prim capture adapter."""

from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from kcg_connector.d38999_keyed_public_spec_v2 import PAIR_MODEL_ID, PLUG_MODEL_ID


RUNTIME_PATH = (
    Path(__file__).parents[1]
    / "isaac"
    / "postgrasp_palm_keyed_visual_capture_runtime.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "postgrasp_palm_keyed_visual_capture_runtime", RUNTIME_PATH
)
capture_runtime = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(capture_runtime)


class _Attribute:
    def __init__(self, value, valid=True):
        self.value = value
        self.valid = valid

    def IsValid(self):
        return self.valid

    def Get(self):
        return self.value


class _PalmPrim:
    def __init__(self, *, valid=True, type_name="Camera", near_clip=0.02):
        self.valid = valid
        self.type_name = type_name
        self.near_clip = near_clip
        self.requested_attributes = []

    def IsValid(self):
        return self.valid

    def GetTypeName(self):
        return self.type_name

    def GetAttribute(self, name):
        self.requested_attributes.append(name)
        if name != "clippingRange":
            raise AssertionError(f"unexpected camera attribute read: {name}")
        return _Attribute((self.near_clip, 10.0))


class _Stage:
    def __init__(self, prim):
        self.prim = prim
        self.requested_paths = []

    def GetPrimAtPath(self, path):
        self.requested_paths.append(path)
        return self.prim


class _RenderProduct:
    def __init__(self, path="/Render/Product/PalmKeyed"):
        self.path = path
        self.destroy_calls = 0

    def destroy(self):
        self.destroy_calls += 1


class _Annotator:
    def __init__(self, data):
        self.data = data
        self.attached = []
        self.detached = []

    def attach(self, paths):
        self.attached.append(list(paths))

    def detach(self, paths):
        self.detached.append(list(paths))

    def get_data(self):
        return self.data


class _Rep:
    def __init__(self, rgb, depth):
        self.requested_annotators = []
        self.render_calls = []
        self.step_calls = []
        self.product = _RenderProduct()
        self.annotators = {
            "rgb": _Annotator(rgb),
            "distance_to_image_plane": _Annotator(depth),
        }
        self.AnnotatorRegistry = SimpleNamespace(
            get_annotator=self._get_annotator
        )
        self.create = SimpleNamespace(render_product=self._render_product)
        self.orchestrator = SimpleNamespace(step=self._step)

    def _get_annotator(self, name):
        self.requested_annotators.append(name)
        if name not in self.annotators:
            raise AssertionError(f"unexpected annotator requested: {name}")
        return self.annotators[name]

    def _render_product(self, prim, resolution, name):
        self.render_calls.append(
            {"prim": prim, "resolution": tuple(resolution), "name": name}
        )
        return self.product

    def _step(self, **kwargs):
        self.step_calls.append(dict(kwargs))


class _World:
    def __init__(self, playing=True):
        self.playing = playing
        self.pause_calls = 0
        self.play_calls = 0

    def is_playing(self):
        return self.playing

    def pause(self):
        self.pause_calls += 1
        self.playing = False

    def play(self):
        self.play_calls += 1
        self.playing = True


class _Robot:
    def __init__(self, before=None, after=None):
        self.before = np.zeros(9) if before is None else np.asarray(before)
        self.after = self.before.copy() if after is None else np.asarray(after)
        self.calls = 0

    def get_joint_positions(self):
        self.calls += 1
        return (self.before if self.calls == 1 else self.after).copy()


def _images():
    height, width = 720, 1280
    rgb = np.zeros((height, width, 4), dtype=np.uint8)
    rgb[:, :, :3] = (90, 100, 110)
    depth = np.full((height, width), np.inf, dtype=np.float32)
    depth[height // 3 : 2 * height // 3, width // 3 : 2 * width // 3] = 0.11
    return rgb, depth


def _t_wr():
    transform = np.eye(4)
    transform[:3, :3] = np.diag((1.0, -1.0, -1.0))
    transform[:3, 3] = (0.55, 0.185, 0.24)
    return transform


def _kwargs(tmp_path, *, playing=True, prim=None, robot=None):
    rgb, depth = _images()
    rep = _Rep(rgb, depth)
    palm_prim = _PalmPrim() if prim is None else prim
    world = _World(playing=playing)
    simulation_app = SimpleNamespace(update_calls=0)

    def update():
        simulation_app.update_calls += 1

    simulation_app.update = update
    values = {
        "output_subdir": tmp_path / "palm_capture",
        "stage": _Stage(palm_prim),
        "world": world,
        "simulation_app": simulation_app,
        "robot": _Robot() if robot is None else robot,
        "arm_indices": np.arange(7),
        "rep": rep,
        "palm_prim_path": (
            "/World/HandArm/Geometry/world/iiwa_link_0/iiwa_link_1/"
            "iiwa_link_2/iiwa_link_3/iiwa_link_4/iiwa_link_5/iiwa_link_6/"
            "iiwa_link_7/iiwa_link_ee/handbase_link/PalmCamera"
        ),
        "scene_schema_version": "kcg_d38999_keyed_v2_tabletop_scene_v1",
        "scene_profile_id": PAIR_MODEL_ID,
        "fixed_orientation_token": (
            "MATING_FACE_UP_RX_180_FOR_DOWNWARD_INSERTION"
        ),
        "keyed_model_id": PLUG_MODEL_ID,
        "T_HC_frozen_configured": np.eye(4),
        "T_WH_from_actual_q": np.eye(4),
        "T_WR_fixed_configured": _t_wr(),
        "T_RP_target_configured": np.eye(4),
    }
    return values, rep, palm_prim, world, simulation_app


def test_existing_palm_prim_rgbd_capture_reaches_precise_unknown_occlusion_stop(
    tmp_path,
):
    kwargs, rep, palm_prim, world, simulation_app = _kwargs(tmp_path)

    result = capture_runtime.run_postgrasp_palm_keyed_visual_capture(**kwargs)

    assert result["status"] == "CAPTURED_EVALUATOR_SAFE_STOP"
    assert result["rejection_code"] == "KEY_REGION_OCCLUSION_UNKNOWN"
    assert result["capture_passed"] is True
    assert result["observation_passed"] is False
    assert result["plan_authorized"] is False
    assert result["simulation_prealign_target_authorized"] is False
    assert result["control_authorized"] is False
    assert result["simulation_prealign_control_authorized"] is False
    assert result["simulation_insertion_control_authorized"] is False
    assert result["hardware_control_authorized"] is False
    assert result["visual_evaluator"]["rejection_code"] == (
        "KEY_REGION_OCCLUSION_UNKNOWN"
    )
    assert result["visual_evaluator"]["plan_authorized"] is False
    assert result["camera_pose_written"] is False
    assert result["object_pose_read"] is False
    assert result["semantic_annotator_requested"] is False
    assert palm_prim.requested_attributes == ["clippingRange"]
    assert world.is_playing() is True
    assert world.pause_calls == 1
    assert world.play_calls == 1
    assert simulation_app.update_calls == 2


def test_render_product_and_annotators_reuse_the_exact_same_palm_prim(tmp_path):
    kwargs, rep, palm_prim, _, _ = _kwargs(tmp_path)

    result = capture_runtime.run_postgrasp_palm_keyed_visual_capture(**kwargs)

    assert result["capture_passed"] is True
    assert len(rep.render_calls) == 1
    assert rep.render_calls[0]["prim"] is palm_prim
    assert rep.render_calls[0]["resolution"] == (1280, 720)
    assert rep.requested_annotators == ["rgb", "distance_to_image_plane"]
    assert len(rep.step_calls) == capture_runtime.WARMUP_FRAMES
    assert all(call["delta_time"] == 0.0 for call in rep.step_calls)
    assert all(call["pause_timeline"] is True for call in rep.step_calls)
    assert all(
        call["rt_subframes"] == capture_runtime.RT_SUBFRAMES
        for call in rep.step_calls
    )


def test_cleanup_detaches_each_annotator_from_the_same_render_product_path(
    tmp_path,
):
    kwargs, rep, _, _, _ = _kwargs(tmp_path)

    result = capture_runtime.run_postgrasp_palm_keyed_visual_capture(**kwargs)

    expected = [[rep.product.path]]
    assert rep.annotators["rgb"].attached == expected
    assert rep.annotators["distance_to_image_plane"].attached == expected
    assert rep.annotators["rgb"].detached == expected
    assert rep.annotators["distance_to_image_plane"].detached == expected
    assert rep.product.destroy_calls == 1
    assert result["resource_cleanup"]["resources_released"] is True
    assert result["resource_cleanup"]["camera_prim_preserved"] is True


def test_capture_reads_actual_arm_q_before_and_after_and_records_drift(tmp_path):
    before = np.zeros(9)
    after = before.copy()
    after[:7] = 1.0e-4
    robot = _Robot(before=before, after=after)
    kwargs, _, _, _, _ = _kwargs(tmp_path, robot=robot)

    result = capture_runtime.run_postgrasp_palm_keyed_visual_capture(**kwargs)

    assert robot.calls == 2
    assert result["joint_capture"]["actual_arm_q_before_capture_rad"] == [0.0] * 7
    assert result["joint_capture"]["actual_arm_q_after_capture_rad"] == [1.0e-4] * 7
    assert result["joint_capture"]["maximum_absolute_drift_rad"] == pytest.approx(
        1.0e-4
    )


def test_excess_joint_drift_returns_structured_abort_safe(tmp_path):
    before = np.zeros(9)
    after = before.copy()
    after[2] = capture_runtime.MAXIMUM_CAPTURE_Q_DRIFT_RAD + 1.0e-4
    kwargs, rep, _, world, _ = _kwargs(
        tmp_path, robot=_Robot(before=before, after=after)
    )

    result = capture_runtime.run_postgrasp_palm_keyed_visual_capture(**kwargs)

    assert result["status"] == "ABORT_SAFE"
    assert result["rejection_code"] == "CAPTURE_Q_DRIFT_ABOVE_LIMIT"
    assert result["plan_authorized"] is False
    assert result["control_authorized"] is False
    assert result["visual_evaluator"] is None
    assert rep.product.destroy_calls == 1
    assert world.is_playing() is True


def test_initially_paused_timeline_stays_paused(tmp_path):
    kwargs, rep, _, world, simulation_app = _kwargs(tmp_path, playing=False)

    result = capture_runtime.run_postgrasp_palm_keyed_visual_capture(**kwargs)

    assert result["status"] == "CAPTURED_EVALUATOR_SAFE_STOP"
    assert world.is_playing() is False
    assert world.pause_calls == 0
    assert world.play_calls == 0
    assert simulation_app.update_calls == 0
    assert all(call["delta_time"] == 0.0 for call in rep.step_calls)
    assert result["timeline"]["restored"] is True


def test_rgb_depth_and_concise_report_are_saved_without_digest_files(tmp_path):
    kwargs, _, _, _, _ = _kwargs(tmp_path)

    result = capture_runtime.run_postgrasp_palm_keyed_visual_capture(**kwargs)
    output = Path(kwargs["output_subdir"])
    loaded = json.loads((output / "report.json").read_text(encoding="utf-8"))

    assert (output / "rgb.png").is_file()
    assert (output / "depth_m.npy").is_file()
    assert np.load(output / "depth_m.npy").shape == (720, 1280)
    assert sorted(path.name for path in output.iterdir()) == [
        "depth_m.npy",
        "report.json",
        "rgb.png",
    ]
    assert loaded["status"] == result["status"]
    assert loaded["artifacts"]["rgb"] == "rgb.png"
    serialized = json.dumps(loaded).lower()
    assert "sha256" not in serialized
    assert "digest" not in serialized


def test_existing_output_subdir_is_never_modified_even_through_finally(tmp_path):
    kwargs, rep, _, _, _ = _kwargs(tmp_path)
    output = Path(kwargs["output_subdir"])
    output.mkdir(parents=True)
    old_report = output / "report.json"
    old_report.write_text("DO NOT OVERWRITE\n", encoding="utf-8")
    marker = output / "user_marker.txt"
    marker.write_text("preserve\n", encoding="utf-8")

    result = capture_runtime.run_postgrasp_palm_keyed_visual_capture(**kwargs)

    assert result["status"] == "ABORT_SAFE"
    assert result["rejection_code"] == "OUTPUT_SUBDIR_ALREADY_EXISTS"
    assert old_report.read_text(encoding="utf-8") == "DO NOT OVERWRITE\n"
    assert marker.read_text(encoding="utf-8") == "preserve\n"
    assert rep.render_calls == []


@pytest.mark.parametrize(
    ("prim", "reason_fragment"),
    (
        (_PalmPrim(valid=False), "unavailable"),
        (_PalmPrim(type_name="Xform"), "not a Camera"),
        (_PalmPrim(near_clip=0.10), "0.02 m"),
    ),
)
def test_invalid_existing_camera_contract_returns_abort_safe(
    tmp_path, prim, reason_fragment
):
    kwargs, rep, _, _, _ = _kwargs(tmp_path, prim=prim)

    result = capture_runtime.run_postgrasp_palm_keyed_visual_capture(**kwargs)

    assert result["status"] == "ABORT_SAFE"
    assert result["rejection_code"] == "CAPTURE_RUNTIME_EXCEPTION"
    assert reason_fragment in result["reason"]
    assert result["control_authorized"] is False
    assert rep.render_calls == []


def test_runtime_signature_has_no_camera_or_endpoint_pose_write_channels():
    parameters = set(
        inspect.signature(
            capture_runtime.run_postgrasp_palm_keyed_visual_capture
        ).parameters
    )
    assert {
        "palm_prim_path",
        "scene_schema_version",
        "scene_profile_id",
        "T_HC_frozen_configured",
        "T_WH_from_actual_q",
        "T_WR_fixed_configured",
        "T_RP_target_configured",
        "output_subdir",
    } <= parameters
    assert not parameters & {
        "camera_world_pose",
        "object_pose",
        "body_pose",
        "nut_pose",
        "contact",
        "collider",
        "semantic",
    }


def test_source_has_no_camera_authoring_pose_writes_or_digest_generation():
    source = RUNTIME_PATH.read_text(encoding="utf-8")
    for forbidden in (
        "UsdGeom.Camera.Define",
        "CreateClippingRangeAttr",
        "ClearXformOpOrder",
        "AddTransformOp",
        "set_world_pose(",
        "get_world_pose(",
        "hashlib",
        "sha256",
    ):
        assert forbidden not in source
    assert 'ANNOTATORS_EXACTLY = ("rgb", "distance_to_image_plane")' in source
    assert "delta_time=0.0" in source
