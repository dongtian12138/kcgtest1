import json
import inspect
from pathlib import Path

import numpy as np
from PIL import Image, PngImagePlugin
import pytest

import kcg_connector.d38999_key_yaw_isolated_evaluator as isolated


DECLARED_YAWS = tuple(-3.14 + index * 0.1 for index in range(64))
DECLARED_STRATA = {
    "yaw": [f"yaw-{index:02d}" for index in range(64)],
    "light": ["light-0", "light-1"],
    "pose": ["pose-0", "pose-1"],
}


def _write_observation(root: Path, sample_id: str) -> None:
    sample = root / sample_id
    sample.mkdir()
    width, height = 8, 6
    Image.new("RGB", (width, height), (20, 40, 60)).save(sample / "rgb.png")
    np.save(sample / "depth_m.npy", np.full((height, width), 0.2, dtype=np.float32))
    np.save(
        sample / "connector_face_mask.npy",
        np.ones((height, width), dtype=np.bool_),
    )
    np.save(
        sample / "occlusion_mask.npy",
        np.zeros((height, width), dtype=np.bool_),
    )
    intrinsics = {
        "width_px": width,
        "height_px": height,
        "fx_px": 10.0,
        "fy_px": 10.0,
        "cx_px": 3.5,
        "cy_px": 2.5,
    }
    (sample / "intrinsics.json").write_text(
        json.dumps(intrinsics), encoding="utf-8"
    )


def _write_truth(root: Path, sample_id: str) -> None:
    record = {
        "sample_id": sample_id,
        "expected_outcome": "VISIBLE_VALID",
        "axial_yaw_truth_rad": 0.0,
        "expected_hypothesis_id": "YAW_0",
        "strata": {"yaw": "yaw-00", "light": "light-0", "pose": "pose-0"},
    }
    (root / f"{sample_id}.json").write_text(json.dumps(record), encoding="utf-8")


def _fake_numeric_report(*args, **kwargs):
    predictions, truth = args[:2]
    assert {item["sample_id"] for item in predictions} == {
        item["sample_id"] for item in truth
    }
    assert all(len(item["sample_id"]) == 48 for item in truth)
    return {
        "passed": False,
        "metric_gates_passed": True,
        "formal_withheld_evidence_verified": False,
        "control_authorized": False,
    }


def _call(tmp_path, monkeypatch, predictor_source):
    observations = tmp_path / "observations"
    truth = tmp_path / "truth-secret"
    observations.mkdir()
    truth.mkdir()
    _write_observation(observations, "heldout_original_name_0001")
    _write_truth(truth, "heldout_original_name_0001")
    predictor = tmp_path / "predictor.py"
    predictor.write_text(predictor_source(observations, truth), encoding="utf-8")
    monkeypatch.setattr(isolated, "_run_existing_numeric_core", _fake_numeric_report)
    report = isolated.evaluate_isolated_key_yaw_benchmark(
        observations,
        str(truth.resolve()),
        predictor,
        keyed_model_id="dummy-model-for-isolation-test",
        dataset_tag="dummy-heldout",
        declared_yaw_values_rad=DECLARED_YAWS,
        declared_strata=DECLARED_STRATA,
    )
    return report


def test_bwrap_hides_truth_workspace_network_and_original_ids(tmp_path, monkeypatch):
    workspace = Path(__file__).resolve().parents[3]
    monkeypatch.setenv("KCG_PARENT_SECRET_SENTINEL", "must-not-cross-clearenv")

    def source(_observations, truth):
        return f'''
import os
from pathlib import Path
import socket

def predict(sample_directory):
    assert os.environ["PATH"] == "/usr/bin"
    assert os.environ["PYTHONDONTWRITEBYTECODE"] == "1"
    assert "KCG_PARENT_SECRET_SENTINEL" not in os.environ
    assert set(item.name for item in sample_directory.iterdir()) == {{
        "rgb.png", "depth_m.npy", "connector_face_mask.npy",
        "occlusion_mask.npy", "intrinsics.json"
    }}
    assert "heldout_original_name" not in sample_directory.name
    assert not Path({str(truth.resolve())!r}).exists()
    assert not Path({str(workspace)!r}).exists()
    for forbidden_write in (
        Path("/not-output"), Path("/input/not-output"),
        Path("/runner/not-output"), Path("/predictor/not-output")
    ):
        try:
            forbidden_write.write_text("must-fail")
        except OSError:
            pass
        else:
            raise AssertionError("a non-output sandbox path was writable")
    sock = socket.socket()
    try:
        sock.settimeout(0.2)
        try:
            sock.connect(("127.0.0.1", 9))
        except OSError:
            pass
        else:
            raise AssertionError("sandbox unexpectedly reached host networking")
    finally:
        sock.close()
    return {{
        "passed": True,
        "estimated_axial_yaw_rad": 0.0,
        "selected_hypothesis_id": "YAW_0",
        "shadow_only": True,
        "control_authorized": False,
    }}
'''

    report = _call(tmp_path, monkeypatch, source)
    receipt = report["os_isolation_receipt"]
    assert report["passed"] is False
    assert report["formal_withheld_evidence_verified"] is False
    assert report["same_account_local_execution"] is True
    assert receipt["truth_mounted_in_sandbox"] is False
    assert receipt["workspace_mounted"] is False
    assert receipt["anonymous_mapping_mounted"] is False
    assert receipt["hash_based_seal_used"] is False
    assert receipt["child_completed_monotonic_ns"] <= (
        receipt["prediction_validated_monotonic_ns"]
    ) <= receipt["truth_first_open_monotonic_ns"]
    assert receipt["truth_fd_dev_inode_ctime_checked"] is True
    assert receipt["truth_read_toctou_checked"] is True
    assert receipt["formal_withheld_evidence_verified"] is False
    assert "heldout_original_name_0001" not in json.dumps(report)
    for field in isolated.CONTROL_FIELDS:
        assert report[field] is False
        assert receipt[field] is False


def test_extra_prediction_field_causes_nonzero_child_failure(tmp_path, monkeypatch):
    observations = tmp_path / "observations"
    truth = tmp_path / "truth-secret"
    observations.mkdir()
    truth.mkdir()
    _write_observation(observations, "sample-1")
    _write_truth(truth, "sample-1")
    predictor = tmp_path / "bad_predictor.py"
    predictor.write_text(
        '''
def predict(_sample_directory):
    return {
        "passed": False,
        "estimated_axial_yaw_rad": None,
        "selected_hypothesis_id": None,
        "shadow_only": True,
        "control_authorized": False,
        "truth_yaw_rad": 0.0,
    }
''',
        encoding="utf-8",
    )
    opened_truth = False
    real_reader = isolated._read_truth_after_child

    def mark_truth(*args, **kwargs):
        nonlocal opened_truth
        opened_truth = True
        return real_reader(*args, **kwargs)

    monkeypatch.setattr(isolated, "_read_truth_after_child", mark_truth)
    with pytest.raises(isolated.IsolatedPredictionError, match="exited"):
        isolated.evaluate_isolated_key_yaw_benchmark(
            observations,
            str(truth.resolve()),
            predictor,
            keyed_model_id="dummy-model",
            dataset_tag="dummy-heldout",
            declared_yaw_values_rad=DECLARED_YAWS,
            declared_strata=DECLARED_STRATA,
        )
    assert opened_truth is False


def test_evaluator_has_no_caller_receipt_and_report_contains_no_hash_value(
    tmp_path, monkeypatch
):
    def source(_observations, _truth):
        return '''
def predict(_sample_directory):
    return {
        "passed": False,
        "estimated_axial_yaw_rad": None,
        "selected_hypothesis_id": None,
        "shadow_only": True,
        "control_authorized": False,
    }
'''

    report = _call(tmp_path, monkeypatch, source)
    parameters = inspect.signature(
        isolated.evaluate_isolated_key_yaw_benchmark
    ).parameters
    assert not any("receipt" in name for name in parameters)
    assert report["os_isolation_receipt"]["generated_by"] == (
        "PARENT_ISOLATED_EVALUATOR_NOT_CALLER"
    )
    assert report["os_isolation_receipt"]["formal_withheld_evidence_verified"] is False
    assert "sha256" not in json.dumps(report).lower()


def test_observation_extra_field_is_rejected_before_truth_open(tmp_path, monkeypatch):
    observations = tmp_path / "observations"
    truth = tmp_path / "truth-secret"
    observations.mkdir()
    truth.mkdir()
    _write_observation(observations, "sample-1")
    _write_truth(truth, "sample-1")
    (observations / "sample-1" / "authored_yaw_deg.json").write_text(
        "0.0", encoding="utf-8"
    )
    predictor = tmp_path / "predictor.py"
    predictor.write_text(
        "def predict(_sample):\n    raise AssertionError('must not run')\n",
        encoding="utf-8",
    )
    opened_truth = False

    def mark_truth(*args, **kwargs):
        nonlocal opened_truth
        opened_truth = True
        raise AssertionError("truth must not open")

    monkeypatch.setattr(isolated, "_read_truth_after_child", mark_truth)
    with pytest.raises(ValueError, match="exactly five"):
        isolated.evaluate_isolated_key_yaw_benchmark(
            observations,
            str(truth.resolve()),
            predictor,
            keyed_model_id="dummy-model",
            dataset_tag="dummy-heldout",
            declared_yaw_values_rad=DECLARED_YAWS,
            declared_strata=DECLARED_STRATA,
        )
    assert opened_truth is False


def test_open_fd_read_detects_ctime_and_size_change(tmp_path, monkeypatch):
    target = tmp_path / "truth.json"
    target.write_bytes(b"x" * (1024 * 1024 + 1))
    fd = isolated.os.open(target, isolated.os.O_RDONLY)
    real_read = isolated.os.read
    first = True

    def read_then_mutate(open_fd, count):
        nonlocal first
        chunk = real_read(open_fd, count)
        if first:
            first = False
            with target.open("ab") as stream:
                stream.write(b"changed")
        return chunk

    monkeypatch.setattr(isolated.os, "read", read_then_mutate)
    try:
        with pytest.raises(RuntimeError, match="changed while read"):
            isolated._read_open_fd(fd, "truth record")
    finally:
        isolated.os.close(fd)


def _assert_observation_attack_rejected_before_child(
    tmp_path, monkeypatch, mutate, match
):
    observations = tmp_path / "observations"
    truth = tmp_path / "truth-secret"
    observations.mkdir()
    truth.mkdir()
    _write_observation(observations, "sample-1")
    _write_truth(truth, "sample-1")
    mutate(observations / "sample-1", truth / "sample-1.json")
    predictor = tmp_path / "predictor.py"
    predictor.write_text(
        "def predict(_sample):\n    raise AssertionError('must not run')\n",
        encoding="utf-8",
    )
    child_started = False
    truth_opened = False

    def mark_child(*args, **kwargs):
        nonlocal child_started
        child_started = True
        raise AssertionError("child must not start")

    def mark_truth(*args, **kwargs):
        nonlocal truth_opened
        truth_opened = True
        raise AssertionError("truth reader must not run")

    monkeypatch.setattr(isolated.subprocess, "run", mark_child)
    monkeypatch.setattr(isolated, "_read_truth_after_child", mark_truth)
    with pytest.raises(ValueError, match=match):
        isolated.evaluate_isolated_key_yaw_benchmark(
            observations,
            str(truth.resolve()),
            predictor,
            keyed_model_id="dummy-model",
            dataset_tag="dummy-heldout",
            declared_yaw_values_rad=DECLARED_YAWS,
            declared_strata=DECLARED_STRATA,
        )
    assert child_started is False
    assert truth_opened is False


def test_hardlinked_truth_as_intrinsics_is_rejected_before_bwrap(
    tmp_path, monkeypatch
):
    def mutate(sample, truth_file):
        (sample / "intrinsics.json").unlink()
        isolated.os.link(truth_file, sample / "intrinsics.json")

    _assert_observation_attack_rejected_before_child(
        tmp_path, monkeypatch, mutate, "non-hardlinked"
    )


def test_png_text_metadata_is_rejected_before_bwrap(tmp_path, monkeypatch):
    def mutate(sample, _truth_file):
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("truth", "authored-yaw-secret")
        Image.new("RGB", (8, 6), (20, 40, 60)).save(
            sample / "rgb.png", pnginfo=metadata
        )

    _assert_observation_attack_rejected_before_child(
        tmp_path, monkeypatch, mutate, "metadata is forbidden"
    )


@pytest.mark.parametrize("dangerous_npy", ("object", "structured", "infinity"))
def test_dangerous_npy_is_rejected_before_bwrap(
    tmp_path, monkeypatch, dangerous_npy
):
    def mutate(sample, _truth_file):
        if dangerous_npy == "object":
            value = np.asarray([[{"truth": 1}]], dtype=object)
        elif dangerous_npy == "structured":
            value = np.zeros((6, 8), dtype=[("depth", "f4"), ("truth", "i4")])
        else:
            value = np.full((6, 8), np.inf, dtype=np.float32)
        np.save(sample / "depth_m.npy", value)

    _assert_observation_attack_rejected_before_child(
        tmp_path, monkeypatch, mutate, "safe NPY|structured|infinity"
    )


@pytest.mark.parametrize("drift", ("extra", "renamed"))
def test_extra_or_renamed_observation_file_is_rejected_before_bwrap(
    tmp_path, monkeypatch, drift
):
    def mutate(sample, _truth_file):
        if drift == "extra":
            (sample / "truth.json").write_text("{}", encoding="utf-8")
        else:
            (sample / "depth_m.npy").rename(sample / "depth-secret.npy")

    _assert_observation_attack_rejected_before_child(
        tmp_path, monkeypatch, mutate, "exactly five"
    )


def test_staged_observation_is_canonical_and_uses_fresh_single_link_inodes(tmp_path):
    observations = tmp_path / "observations"
    staging = tmp_path / "staging"
    observations.mkdir()
    staging.mkdir(mode=0o700)
    _write_observation(observations, "sample-1")
    mapping, _ = isolated._stage_observations(observations, staging)
    anonymous_id = next(iter(mapping))
    staged_sample = staging / anonymous_id
    source_ids = {
        (item.stat().st_dev, item.stat().st_ino)
        for item in (observations / "sample-1").iterdir()
    }
    for item in staged_sample.iterdir():
        item_stat = item.lstat()
        assert item_stat.st_nlink == 1
        assert (item_stat.st_dev, item_stat.st_ino) not in source_ids
    with Image.open(staged_sample / "rgb.png") as image:
        assert image.mode == "RGB"
        assert image.info == {}
    staged_depth = np.load(staged_sample / "depth_m.npy", allow_pickle=False)
    assert staged_depth.dtype == np.float32


@pytest.mark.parametrize("bad_intrinsics", ("extra", "boolean", "wrong_size"))
def test_intrinsics_schema_and_dimensions_are_rejected_before_bwrap(
    tmp_path, monkeypatch, bad_intrinsics
):
    def mutate(sample, _truth_file):
        path = sample / "intrinsics.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        if bad_intrinsics == "extra":
            value["truth"] = 1.0
        elif bad_intrinsics == "boolean":
            value["fx_px"] = True
        else:
            value["width_px"] = 9
        path.write_text(json.dumps(value), encoding="utf-8")

    _assert_observation_attack_rejected_before_child(
        tmp_path,
        monkeypatch,
        mutate,
        "exactly six|must be numeric|dimensions do not match",
    )


def test_observation_fifo_is_rejected_before_child_or_truth_open(
    tmp_path, monkeypatch
):
    def mutate(sample, _truth_file):
        (sample / "intrinsics.json").unlink()
        isolated.os.mkfifo(sample / "intrinsics.json")

    _assert_observation_attack_rejected_before_child(
        tmp_path, monkeypatch, mutate, "regular non-hardlinked"
    )


def test_prediction_fifo_parent_validation_is_nonblocking(tmp_path):
    fifo = tmp_path / "predictions.jsonl"
    isolated.os.mkfifo(fifo)
    started = isolated.time.monotonic()
    with pytest.raises(ValueError, match="regular single-link"):
        isolated._read_prediction_output(fifo)
    assert isolated.time.monotonic() - started < 1.0


def test_truth_fifo_parent_validation_is_nonblocking(tmp_path):
    truth = tmp_path / "truth"
    truth.mkdir()
    isolated.os.mkfifo(truth / "original.json")
    started = isolated.time.monotonic()
    with pytest.raises(ValueError, match="regular single-link"):
        isolated._read_truth_after_child(truth, {"a" * 48: "original"})
    assert isolated.time.monotonic() - started < 1.0


def test_prediction_parent_limits_and_deadline_fail_closed(tmp_path, monkeypatch):
    oversized = tmp_path / "oversized.jsonl"
    oversized.write_bytes(b"12345")
    fd = isolated.os.open(
        oversized, isolated.os.O_RDONLY | isolated.os.O_NONBLOCK
    )
    try:
        with pytest.raises(ValueError, match="byte limit"):
            isolated._read_open_fd(fd, "prediction output", maximum_bytes=4)
    finally:
        isolated.os.close(fd)
    with pytest.raises(TimeoutError, match="fixed parent deadline"):
        isolated._validated_predictions(
            b"{}\n", deadline_ns=isolated.time.monotonic_ns() - 1
        )
    monkeypatch.setattr(isolated, "MAXIMUM_PREDICTION_LINES", 1)
    with pytest.raises(ValueError, match="line-count limit"):
        isolated._validated_predictions(b"{}\n{}\n")
