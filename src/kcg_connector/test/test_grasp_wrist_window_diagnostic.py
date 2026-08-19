'''Targeted tests for the read-only wrist window diagnostic CLI (035).'''

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPOSITORY = Path(__file__).resolve().parents[3]
CLI = (
    REPOSITORY
    / "src/kcg_connector/isaac/d38999_grasp_wrist_window_diagnostic.py"
)
SOURCE_ROOT = str(REPOSITORY / "src" / "kcg_connector")

WRIST = [0.3, -0.1, 19.0, 0.1, 0.2, 0.01]
ROOT = {"f1": 0.25, "f2": 0.26, "f3": 0.25}


def _write_steps(
    directory: Path,
    *,
    phases=("physical_grip_consolidation",),
    wrist_fn=None,
    include_streams=("canonical",),
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for offset, phase in enumerate(phases):
        step = 9000 + offset
        record = {
            "global_step": step,
            "phase": phase,
            "contact_order": ["f2", "f3", "f1"],
            "finger_root_torque_proxy_nm": dict(ROOT),
            "finger_targets_rad": [0.7, 0.5, 0.7],
        }
        if phase == "physical_grip_consolidation":
            record["controller_evidence"] = {
                "consolidation_window_step": offset + 1
            }
        if wrist_fn is not None:
            values = wrist_fn(step)
        else:
            values = list(WRIST)
        if values is not None:
            if "raw" in include_streams:
                record["wrist_wrench_raw_sensor_frame"] = [
                    -v for v in values
                ]
            if "canonical" in include_streams:
                record["wrist_wrench_canonical"] = list(values)
            if "empty" in include_streams:
                record["wrist_wrench_empty_baseline_compensated"] = [
                    v * 0.5 for v in values
                ]
            if "payload" in include_streams:
                record["wrist_wrench_payload_reference"] = list(values)
                record["wrist_wrench_payload_reference_increment"] = [
                    v * 0.1 for v in values
                ]
        records.append(json.dumps(record))
    (directory / "controller_steps.jsonl").write_text(
        "\n".join(records) + "\n", encoding="utf-8"
    )


def _run_cli(episodes, output_dir, extra=()):
    environment = dict(os.environ)
    environment["PYTHONPATH"] = SOURCE_ROOT + os.pathsep + environment.get(
        "PYTHONPATH", ""
    )
    command = [sys.executable, str(CLI)]
    for episode in episodes:
        command += ["--episode", str(episode)]
    command += ["--output", str(output_dir), *extra]
    result = subprocess.run(
        command, capture_output=True, text=True, env=environment, check=False
    )
    return result.returncode, result.stdout, result.stderr


def _document(output_dir) -> dict:
    return json.loads(
        (output_dir / "wrist_window_diagnostic.json").read_text(
            encoding="utf-8"
        )
    )


def test_per_stream_separation_and_math(tmp_path):
    directory = tmp_path / "ep"
    _write_steps(
        directory,
        phases=("physical_grip_consolidation",) * 100,
        include_streams=("raw", "canonical", "empty", "payload"),
    )
    code, _stdout, stderr = _run_cli([directory], tmp_path / "out")
    assert code == 0, stderr
    doc = _document(tmp_path / "out")
    streams = doc["episodes"][0]["segments"]["consolidation_window"][
        "wrist_streams"
    ]
    assert set(streams) == {
        "wrist_wrench_raw_sensor_frame",
        "wrist_wrench_canonical",
        "wrist_wrench_empty_baseline_compensated",
        "wrist_wrench_payload_reference",
        "wrist_wrench_payload_reference_increment",
    }
    canonical = streams["wrist_wrench_canonical"]
    raw = streams["wrist_wrench_raw_sensor_frame"]
    assert canonical["available"] is True
    assert canonical["sample_count"] == 100
    assert raw["per_channel"]["fx_n"]["mean"] == pytest.approx(
        -canonical["per_channel"]["fx_n"]["mean"]
    )
    expected = np.asarray([WRIST[0]] * 100, dtype=np.float64)
    assert canonical["per_channel"]["fx_n"]["std"] == pytest.approx(
        float(np.std(expected))
    )
    assert canonical["per_channel"]["fx_n"]["rms"] == pytest.approx(
        float(np.sqrt(np.mean(expected ** 2)))
    )
    fx = canonical["correlation_matrix"]["fx_n"]
    assert fx["fx_n"] is None
    assert fx["fx_n_available"] is False


def test_missing_stream_is_unavailable(tmp_path):
    directory = tmp_path / "ep"
    _write_steps(
        directory,
        phases=("physical_grip_consolidation",) * 10,
        include_streams=("canonical",),
    )
    code, _stdout, stderr = _run_cli([directory], tmp_path / "out")
    assert code == 0, stderr
    doc = _document(tmp_path / "out")
    streams = doc["episodes"][0]["segments"]["consolidation_window"][
        "wrist_streams"
    ]
    assert streams["wrist_wrench_raw_sensor_frame"]["available"] is False
    assert streams["wrist_wrench_raw_sensor_frame"][
        "unavailable_and_not_inferred"
    ] is True


def test_dominant_frequency_and_rate_hz(tmp_path):
    directory = tmp_path / "ep"
    _write_steps(
        directory,
        phases=("physical_grip_consolidation",) * 100,
        wrist_fn=lambda step: (
            [1.0 if (step - 9000) % 2 == 0 else -1.0] + [0.0] * 5
        ),
    )
    code, _stdout, stderr = _run_cli(
        [directory], tmp_path / "out", extra=("--rate-hz", "240")
    )
    assert code == 0, stderr
    doc = _document(tmp_path / "out")
    fx = doc["episodes"][0]["segments"]["consolidation_window"][
        "wrist_streams"
    ]["wrist_wrench_canonical"]["per_channel"]["fx_n"]
    assert fx["dominant_cycles_per_step"] == pytest.approx(0.5)
    assert fx["dominant_hz"] == pytest.approx(120.0)
    assert fx["dominant_hz_available"] is True


def test_hz_unavailable_without_rate(tmp_path):
    directory = tmp_path / "ep"
    _write_steps(
        directory,
        phases=("physical_grip_consolidation",) * 20,
        wrist_fn=lambda step: [math.sin(step)] + [0.0] * 5,
    )
    code, _stdout, stderr = _run_cli([directory], tmp_path / "out")
    assert code == 0, stderr
    doc = _document(tmp_path / "out")
    fx = doc["episodes"][0]["segments"]["consolidation_window"][
        "wrist_streams"
    ]["wrist_wrench_canonical"]["per_channel"]["fx_n"]
    assert fx["dominant_hz"] is None
    assert fx["dominant_hz_available"] is False


@pytest.mark.parametrize("rate", ["0", "-1", "nan", "inf"])
def test_invalid_rate_hz_rejected(tmp_path, rate):
    directory = tmp_path / "ep"
    _write_steps(directory, phases=("physical_grip_consolidation",) * 5)
    code, _stdout, _stderr = _run_cli(
        [directory], tmp_path / "out", extra=("--rate-hz", rate)
    )
    assert code != 0


def test_overlap_per_stream_availability(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    c = tmp_path / "c"
    _write_steps(
        a,
        phases=("physical_grip_consolidation",) * 20,
        wrist_fn=lambda s: list(WRIST),
        include_streams=("canonical", "raw"),
    )
    _write_steps(
        b,
        phases=("physical_grip_consolidation",) * 20,
        wrist_fn=lambda s: list(WRIST),
        include_streams=("canonical", "raw"),
    )
    _write_steps(
        c,
        phases=("physical_grip_consolidation",) * 20,
        wrist_fn=lambda s: [WRIST[0] + 1.0] + WRIST[1:],
        include_streams=("canonical",),
    )
    code, _stdout, stderr = _run_cli(
        [a],
        tmp_path / "out",
        extra=("--compare-a", str(b), "--compare-b", str(a)),
    )
    assert code == 0, stderr
    doc = _document(tmp_path / "out")
    overlap = doc["prelift_overlap"]
    assert overlap["common_step_count"] == 20
    canonical = overlap["fields"]["wrist_wrench_canonical"]
    assert canonical["available"] is True
    assert canonical["identical"] is True
    assert canonical["sample_count"] == 20
    assert canonical["sha256_a"] == canonical["sha256_b"]
    code2, _stdout2, _stderr2 = _run_cli(
        [a],
        tmp_path / "out2",
        extra=("--compare-a", str(c), "--compare-b", str(a)),
    )
    assert code2 == 0
    doc2 = _document(tmp_path / "out2")
    field = doc2["prelift_overlap"]["fields"]["wrist_wrench_canonical"]
    assert field["identical"] is False
    assert field["max_abs_diff"] == pytest.approx(1.0)
    raw_field = doc2["prelift_overlap"]["fields"][
        "wrist_wrench_raw_sensor_frame"
    ]
    assert raw_field["available"] is False
    assert raw_field["unavailable_and_not_inferred"] is True


def test_overlap_does_not_claim_full_samples_per_field(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    _write_steps(
        a,
        phases=("physical_hand_closure",) * 5
        + ("physical_grip_consolidation",) * 15,
        include_streams=("canonical",),
        wrist_fn=lambda s: None,
    )
    _write_steps(
        b,
        phases=("physical_hand_closure",) * 5
        + ("physical_grip_consolidation",) * 15,
        include_streams=("canonical",),
        wrist_fn=lambda s: list(WRIST),
    )
    code, _stdout, stderr = _run_cli(
        [a],
        tmp_path / "out",
        extra=("--compare-a", str(b), "--compare-b", str(a)),
    )
    assert code == 0, stderr
    doc = _document(tmp_path / "out")
    field = doc["prelift_overlap"]["fields"]["wrist_wrench_canonical"]
    assert field["available"] is False
    assert field["unavailable_and_not_inferred"] is True
    targets = doc["prelift_overlap"]["fields"]["finger_targets_rad"]
    assert targets["sample_count"] == 20
    assert doc["prelift_overlap"]["common_step_count"] == 20


def test_bad_jsonl_line_and_non_increasing_steps_exit_one(tmp_path):
    directory = tmp_path / "ep"
    _write_steps(directory, phases=("physical_grip_consolidation",) * 5)
    with (directory / "controller_steps.jsonl").open("a") as handle:
        handle.write("{bad json\n")
    code, _stdout, _stderr = _run_cli([directory], tmp_path / "out")
    assert code == 1
    doc = _document(tmp_path / "out")
    assert doc["problems"]

    directory2 = tmp_path / "ep2"
    _write_steps(directory2, phases=("physical_grip_consolidation",) * 5)
    path = directory2 / "controller_steps.jsonl"
    lines = path.read_text().splitlines()
    lines[3] = json.dumps(json.loads(lines[2]))
    path.write_text("\n".join(lines) + "\n")
    code2, _stdout2, _stderr2 = _run_cli([directory2], tmp_path / "out2")
    assert code2 == 1


def test_nut_curve_unavailable_and_final_hold_note(tmp_path):
    directory = tmp_path / "ep"
    _write_steps(directory, phases=("physical_grip_lift_stage_3",) * 5)
    code, _stdout, stderr = _run_cli([directory], tmp_path / "out")
    assert code == 0, stderr
    doc = _document(tmp_path / "out")
    segment = doc["episodes"][0]["segments"]["physical_grip_lift_stage_3"]
    assert segment["nut_pose_velocity_per_step"] == (
        "unavailable_and_not_inferred"
    )
    assert "absent_in_this_schema" in doc["episodes"][0]["final_hold_note"]


def test_read_only_marker_and_finite_json(tmp_path):
    directory = tmp_path / "ep"
    _write_steps(
        directory,
        phases=("physical_grip_consolidation",) * 5,
        wrist_fn=lambda s: [0.0] * 6,
    )
    code, _stdout, stderr = _run_cli([directory], tmp_path / "out")
    assert code == 0, stderr
    doc = _document(tmp_path / "out")
    assert "read_only_note" in doc
    raw = (tmp_path / "out" / "wrist_window_diagnostic.json").read_text(
        encoding="utf-8"
    )
    assert "NaN" not in raw
    assert "Infinity" not in raw


def test_near_constant_channel_with_tiny_jitter(tmp_path):
    directory = tmp_path / "ep"
    _write_steps(
        directory,
        phases=("physical_grip_consolidation",) * 100,
        wrist_fn=lambda step: (
            [19.36711493333181 + (1e-17 if (step - 9000) % 2 else -1e-17)]
            + [0.0] * 5
        ),
    )
    code, _stdout, stderr = _run_cli(
        [directory], tmp_path / "out", extra=("--rate-hz", "240")
    )
    assert code == 0, stderr
    doc = _document(tmp_path / "out")
    segment = doc["episodes"][0]["segments"]["consolidation_window"]
    fx = segment["wrist_streams"]["wrist_wrench_canonical"][
        "per_channel"
    ]["fx_n"]
    assert fx["near_constant"] is True
    assert fx["mean"] == pytest.approx(19.36711493333181, rel=1e-12)
    assert "std" in fx and "rms" in fx
    assert fx["dominant_cycles_per_step"] is None
    assert fx["dominant_hz"] is None
    correlation = segment["wrist_streams"]["wrist_wrench_canonical"][
        "correlation_matrix"
    ]
    fx_row = correlation["fx_n"]
    for name in ("fx_n", "fy_n", "fz_n"):
        assert fx_row[name] is None
        assert fx_row[name + "_available"] is False
    raw = (tmp_path / "out" / "wrist_window_diagnostic.json").read_text(
        encoding="utf-8"
    )
    assert "NaN" not in raw
