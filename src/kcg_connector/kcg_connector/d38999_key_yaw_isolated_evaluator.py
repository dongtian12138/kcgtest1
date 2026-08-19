"""OS-isolated, same-account heldout evaluation skeleton for keyed D38999 yaw.

The prediction child receives anonymous copies of exactly five observation
files.  Truth is first opened by the parent only after the child has exited
and its exact prediction schema has been validated.  This produces an OS
isolation receipt, not independent formal-heldout evidence.

Truth files are named ``<original sample id>.json`` and must already use the
truth-record schema consumed by the existing local benchmark metrics core.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from io import BytesIO
import json
import math
from numbers import Real
import os
from pathlib import Path
import secrets
import stat
import subprocess
import tempfile
import time
from typing import Any

import numpy as np
from PIL import Image

from kcg_connector.d38999_key_yaw_benchmark import (
    CLAIMED_REVEAL_SCHEMA_VERSION,
    evaluate_local_key_yaw_benchmark_metrics,
    write_local_key_yaw_prediction_artifact,
)


BWRAP_PATH = "/usr/bin/bwrap"
PYTHON_PATH = "/usr/bin/python3"
SCHEMA_VERSION = "kcg_d38999_key_yaw_isolated_evaluation_v1"
RECEIPT_SCHEMA_VERSION = "kcg_d38999_key_yaw_os_isolation_receipt_v1"
PREDICTION_FIELDS = frozenset(
    {
        "sample_id",
        "passed",
        "estimated_axial_yaw_rad",
        "selected_hypothesis_id",
        "shadow_only",
        "control_authorized",
    }
)
OBSERVATION_FILES = frozenset(
    {
        "rgb.png",
        "depth_m.npy",
        "connector_face_mask.npy",
        "occlusion_mask.npy",
        "intrinsics.json",
    }
)
EXPECTED_C2_IDS = frozenset({"YAW_0", "YAW_PI"})
INTRINSIC_FIELDS = frozenset(
    {"width_px", "height_px", "fx_px", "fy_px", "cx_px", "cy_px"}
)
MAXIMUM_IMAGE_DIMENSION_PX = 4096
MAXIMUM_IMAGE_PIXELS = MAXIMUM_IMAGE_DIMENSION_PX**2
PARENT_OPERATION_DEADLINE_SECONDS = 1800.0
PARENT_ARTIFACT_DEADLINE_SECONDS = 10.0
MAXIMUM_SAMPLE_COUNT = 100_000
MAXIMUM_PREDICTION_OUTPUT_BYTES = 64 * 1024 * 1024
MAXIMUM_PREDICTION_LINES = MAXIMUM_SAMPLE_COUNT
MAXIMUM_PREDICTION_LINE_BYTES = 16 * 1024
MAXIMUM_TRUTH_RECORD_BYTES = 1024 * 1024
MAXIMUM_CODE_BYTES = 2 * 1024 * 1024
OBSERVATION_MAXIMUM_BYTES = {
    "rgb.png": 64 * 1024 * 1024,
    "depth_m.npy": 160 * 1024 * 1024,
    "connector_face_mask.npy": 20 * 1024 * 1024,
    "occlusion_mask.npy": 20 * 1024 * 1024,
    "intrinsics.json": 16 * 1024,
}
CONTROL_FIELDS = (
    "shadow_authorized",
    "control_authorized",
    "selected_for_control_allowed",
    "simulation_insertion_control_authorized",
    "robot_control_authorized",
    "hardware_control_authorized",
)


class IsolatedPredictionError(RuntimeError):
    """Raised when the prediction sandbox fails or violates its protocol."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _deadline_ns(seconds: float = PARENT_ARTIFACT_DEADLINE_SECONDS) -> int:
    return time.monotonic_ns() + int(seconds * 1_000_000_000)


def _bounded_deadline(deadline_ns: int | None, seconds: float) -> int:
    local = _deadline_ns(seconds)
    return local if deadline_ns is None else min(deadline_ns, local)


def _check_deadline(deadline_ns: int, label: str) -> None:
    if time.monotonic_ns() > deadline_ns:
        raise TimeoutError(f"{label} exceeded the fixed parent deadline")


def _nonblocking_read_flags() -> int:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


def _fd_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_ctime_ns),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_nlink),
    )


def _read_open_fd(
    fd: int,
    label: str,
    *,
    deadline_ns: int | None = None,
    maximum_bytes: int = MAXIMUM_PREDICTION_OUTPUT_BYTES,
) -> bytes:
    deadline = _bounded_deadline(deadline_ns, PARENT_ARTIFACT_DEADLINE_SECONDS)
    _check_deadline(deadline, label)
    before = os.fstat(fd)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ValueError(f"{label} must be a regular single-link file")
    if before.st_size < 0 or before.st_size > maximum_bytes:
        raise ValueError(f"{label} exceeds its fixed byte limit")
    chunks: list[bytes] = []
    total = 0
    while True:
        _check_deadline(deadline, label)
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > maximum_bytes:
            raise ValueError(f"{label} exceeds its fixed byte limit")
        chunks.append(chunk)
    after = os.fstat(fd)
    if _fd_identity(before) != _fd_identity(after):
        raise RuntimeError(f"{label} changed while read through its opened fd")
    return b"".join(chunks)


def _read_source_regular(
    source: Path,
    label: str,
    *,
    deadline_ns: int | None = None,
    maximum_bytes: int = MAXIMUM_CODE_BYTES,
) -> tuple[bytes, tuple[int, int]]:
    deadline = _bounded_deadline(deadline_ns, PARENT_ARTIFACT_DEADLINE_SECONDS)
    _check_deadline(deadline, label)
    source_lstat = source.lstat()
    if (
        not stat.S_ISREG(source_lstat.st_mode)
        or source_lstat.st_nlink != 1
        or source_lstat.st_size > maximum_bytes
    ):
        raise ValueError(f"{label} must be a regular non-hardlinked file")
    fd = os.open(source, _nonblocking_read_flags())
    try:
        opened = os.fstat(fd)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino)
            != (source_lstat.st_dev, source_lstat.st_ino)
        ):
            raise ValueError(f"{label} changed before its opened-fd check")
        payload = _read_open_fd(
            fd,
            label,
            deadline_ns=deadline,
            maximum_bytes=maximum_bytes,
        )
    finally:
        os.close(fd)
    return payload, (int(opened.st_dev), int(opened.st_ino))


def _write_new_staging_file(
    destination: Path,
    payload: bytes,
    source_identities: set[tuple[int, int]],
) -> None:
    with destination.open("xb") as stream:
        stream.write(payload)
    destination.chmod(0o400)
    created = destination.lstat()
    if (
        not stat.S_ISREG(created.st_mode)
        or created.st_nlink != 1
        or (int(created.st_dev), int(created.st_ino)) in source_identities
    ):
        raise RuntimeError("staged file is not a fresh single-link inode")


def _copy_regular_file(source: Path, destination: Path, label: str) -> None:
    payload, identity = _read_source_regular(
        source, label, maximum_bytes=MAXIMUM_CODE_BYTES
    )
    _write_new_staging_file(destination, payload, {identity})


def _canonical_rgb(
    payload: bytes, deadline_ns: int
) -> tuple[bytes, tuple[int, int]]:
    _check_deadline(deadline_ns, "rgb.png validation")
    try:
        with Image.open(BytesIO(payload)) as image:
            if image.format != "PNG" or getattr(image, "n_frames", 1) != 1:
                raise ValueError("rgb.png must be one PNG image")
            if image.info or len(image.getexif()) != 0:
                raise ValueError("rgb.png metadata is forbidden")
            width, height = image.size
            if (
                width <= 0
                or height <= 0
                or width > MAXIMUM_IMAGE_DIMENSION_PX
                or height > MAXIMUM_IMAGE_DIMENSION_PX
                or width * height > MAXIMUM_IMAGE_PIXELS
            ):
                raise ValueError("rgb.png dimensions are outside the safe limit")
            image.load()
            canonical = image.convert("RGB").copy()
    except (OSError, SyntaxError) as exc:
        raise ValueError("rgb.png is not a valid decodable PNG") from exc
    output = BytesIO()
    canonical.save(output, format="PNG", optimize=False, compress_level=9)
    _check_deadline(deadline_ns, "rgb.png validation")
    return output.getvalue(), (width, height)


def _load_plain_npy(payload: bytes, label: str, deadline_ns: int) -> np.ndarray:
    _check_deadline(deadline_ns, f"{label} validation")
    try:
        loaded = np.load(BytesIO(payload), allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"{label} is not a safe NPY array") from exc
    if not isinstance(loaded, np.ndarray):
        if hasattr(loaded, "close"):
            loaded.close()
        raise ValueError(f"{label} must be one plain NPY array")
    if loaded.dtype.hasobject or loaded.dtype.fields or loaded.dtype.subdtype:
        raise ValueError(f"{label} object/structured/metadata dtype is forbidden")
    if loaded.dtype.metadata is not None:
        raise ValueError(f"{label} dtype metadata is forbidden")
    _check_deadline(deadline_ns, f"{label} validation")
    return loaded


def _canonical_npy(array: np.ndarray) -> bytes:
    output = BytesIO()
    np.save(output, array, allow_pickle=False)
    return output.getvalue()


def _canonical_intrinsics(
    payload: bytes, shape: tuple[int, int], deadline_ns: int
) -> bytes:
    _check_deadline(deadline_ns, "intrinsics.json validation")
    def reject_constant(value: str) -> None:
        raise ValueError(f"intrinsics contains forbidden constant {value}")

    def exact_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("intrinsics contains duplicate fields")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=exact_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("intrinsics.json must be strict UTF-8 JSON") from exc
    if not isinstance(value, Mapping) or set(value) != INTRINSIC_FIELDS:
        raise ValueError("intrinsics.json must contain exactly six fields")
    normalized: dict[str, float | int] = {}
    for field in INTRINSIC_FIELDS:
        number = value[field]
        if isinstance(number, bool) or not isinstance(number, Real):
            raise ValueError(f"intrinsics.{field} must be numeric")
        converted = float(number)
        if not math.isfinite(converted):
            raise ValueError(f"intrinsics.{field} must be finite")
        if field in {"width_px", "height_px"}:
            if not converted.is_integer() or converted <= 0:
                raise ValueError(f"intrinsics.{field} must be a positive integer")
            normalized[field] = int(converted)
        else:
            normalized[field] = converted
    height, width = shape
    if (normalized["width_px"], normalized["height_px"]) != (width, height):
        raise ValueError("intrinsics dimensions do not match RGB-D")
    if normalized["fx_px"] <= 0.0 or normalized["fy_px"] <= 0.0:
        raise ValueError("intrinsics focal lengths must be positive")
    if not (0.0 <= normalized["cx_px"] < width):
        raise ValueError("intrinsics cx_px must lie inside the image")
    if not (0.0 <= normalized["cy_px"] < height):
        raise ValueError("intrinsics cy_px must lie inside the image")
    _check_deadline(deadline_ns, "intrinsics.json validation")
    return (
        json.dumps(normalized, allow_nan=False, sort_keys=True).encode("utf-8")
        + b"\n"
    )


def _canonicalize_observation(
    entries: Mapping[str, Path],
    destination: Path,
    operation_deadline_ns: int | None = None,
) -> None:
    deadline = _bounded_deadline(
        operation_deadline_ns, PARENT_ARTIFACT_DEADLINE_SECONDS
    )
    payloads: dict[str, bytes] = {}
    source_identities: set[tuple[int, int]] = set()
    for filename in sorted(OBSERVATION_FILES):
        payload, identity = _read_source_regular(
            entries[filename],
            f"observation field {filename}",
            deadline_ns=deadline,
            maximum_bytes=OBSERVATION_MAXIMUM_BYTES[filename],
        )
        payloads[filename] = payload
        source_identities.add(identity)
    rgb_payload, (width, height) = _canonical_rgb(payloads["rgb.png"], deadline)
    depth = _load_plain_npy(payloads["depth_m.npy"], "depth_m.npy", deadline)
    if depth.dtype not in (np.dtype("float32"), np.dtype("float64")) or depth.ndim != 2:
        raise ValueError("depth_m.npy must be a plain float32/float64 2-D array")
    if depth.shape != (height, width):
        raise ValueError("depth_m.npy dimensions do not match rgb.png")
    finite_depth = np.isfinite(depth)
    if np.any(np.isinf(depth)) or not np.any(finite_depth):
        raise ValueError("depth_m.npy must contain finite depth or NaN, not infinity")
    if np.any(depth[finite_depth] <= 0.0):
        raise ValueError("finite depth_m.npy values must be positive")
    depth = np.array(depth, dtype=depth.dtype, order="C", copy=True)
    depth[~finite_depth] = np.nan
    face = _load_plain_npy(
        payloads["connector_face_mask.npy"],
        "connector_face_mask.npy",
        deadline,
    )
    occlusion = _load_plain_npy(
        payloads["occlusion_mask.npy"], "occlusion_mask.npy", deadline
    )
    for label, mask in (
        ("connector_face_mask.npy", face),
        ("occlusion_mask.npy", occlusion),
    ):
        if mask.dtype != np.dtype("bool") or mask.ndim != 2:
            raise ValueError(f"{label} must be a plain bool 2-D array")
        if mask.shape != depth.shape:
            raise ValueError(f"{label} dimensions do not match RGB-D")
    face = np.array(face, dtype=np.bool_, order="C", copy=True)
    occlusion = np.array(occlusion, dtype=np.bool_, order="C", copy=True)
    intrinsics_payload = _canonical_intrinsics(
        payloads["intrinsics.json"], depth.shape, deadline
    )
    canonical_payloads = {
        "rgb.png": rgb_payload,
        "depth_m.npy": _canonical_npy(depth),
        "connector_face_mask.npy": _canonical_npy(face),
        "occlusion_mask.npy": _canonical_npy(occlusion),
        "intrinsics.json": intrinsics_payload,
    }
    for filename in sorted(OBSERVATION_FILES):
        _check_deadline(deadline, "observation canonical staging")
        _write_new_staging_file(
            destination / filename,
            canonical_payloads[filename],
            source_identities,
        )


def _stage_observations(
    observation_root: Path,
    anonymous_input: Path,
    operation_deadline_ns: int | None = None,
) -> tuple[dict[str, str], list[str]]:
    operation_deadline = (
        _deadline_ns(PARENT_OPERATION_DEADLINE_SECONDS)
        if operation_deadline_ns is None
        else operation_deadline_ns
    )
    _check_deadline(operation_deadline, "observation staging")
    root_stat = observation_root.lstat()
    if not stat.S_ISDIR(root_stat.st_mode) or observation_root.is_symlink():
        raise ValueError("observation_root must be a real directory")
    samples = sorted(item for item in observation_root.iterdir() if item.is_dir())
    if not samples:
        raise ValueError("observation_root contains no sample directories")
    if len(samples) > MAXIMUM_SAMPLE_COUNT:
        raise ValueError("observation_root exceeds the fixed sample-count limit")
    if {item.name for item in observation_root.iterdir()} != {
        item.name for item in samples
    }:
        raise ValueError("observation_root must contain only sample directories")
    randomized = list(samples)
    secrets.SystemRandom().shuffle(randomized)
    mapping: dict[str, str] = {}
    creation_order: list[str] = []
    for sample in randomized:
        _check_deadline(operation_deadline, "observation staging")
        if sample.is_symlink() or not sample.name or "/" in sample.name:
            raise ValueError("observation sample directory is invalid")
        entries = {item.name: item for item in sample.iterdir()}
        if set(entries) != OBSERVATION_FILES:
            raise ValueError("each observation must contain exactly five image files")
        anonymous_id = secrets.token_hex(24)
        while anonymous_id in mapping:
            anonymous_id = secrets.token_hex(24)
        destination = anonymous_input / anonymous_id
        destination.mkdir(mode=0o700)
        _canonicalize_observation(entries, destination, operation_deadline)
        destination.chmod(0o500)
        mapping[anonymous_id] = sample.name
        creation_order.append(anonymous_id)
    anonymous_input.chmod(0o500)
    return mapping, creation_order


def _copy_predictor(predictor_path: Path, destination: Path) -> None:
    if predictor_path.is_symlink() or not predictor_path.is_file():
        raise ValueError("predictor_path must be one regular Python file")
    if predictor_path.suffix != ".py":
        raise ValueError("predictor_path must name a Python source file")
    _copy_regular_file(predictor_path, destination, "predictor source")


def _bwrap_command(staging: Path) -> list[str]:
    worker = Path(__file__).with_name("d38999_key_yaw_isolated_worker.py")
    if worker.is_symlink() or not worker.is_file():
        raise RuntimeError("isolated worker is missing or is a symlink")
    runner = staging / "runner"
    runner.mkdir(mode=0o700)
    _copy_regular_file(worker, runner / "worker.py", "isolated worker")
    runner.chmod(0o500)
    return [
        BWRAP_PATH,
        "--unshare-all",
        "--new-session",
        "--die-with-parent",
        "--clearenv",
        "--cap-drop",
        "ALL",
        "--ro-bind",
        "/usr",
        "/usr",
        "--ro-bind",
        "/lib",
        "/lib",
        "--ro-bind",
        "/lib64",
        "/lib64",
        "--ro-bind",
        "/etc/alternatives",
        "/etc/alternatives",
        "--dir",
        "/dev",
        "--ro-bind",
        "/dev/null",
        "/dev/null",
        "--ro-bind",
        "/dev/urandom",
        "/dev/urandom",
        "--ro-bind",
        str(staging / "input"),
        "/input",
        "--ro-bind",
        str(runner),
        "/runner",
        "--ro-bind",
        str(staging / "predictor"),
        "/predictor",
        "--dir",
        "/output",
        "--remount-ro",
        "/",
        "--bind",
        str(staging / "output"),
        "/output",
        "--chdir",
        "/",
        "--setenv",
        "PATH",
        "/usr/bin",
        "--setenv",
        "PYTHONDONTWRITEBYTECODE",
        "1",
        PYTHON_PATH,
        "-I",
        "/runner/worker.py",
        "--input",
        "/input",
        "--output",
        "/output/predictions.jsonl",
        "--predictor",
        "/predictor/predictor.py",
    ]


def _validated_predictions(
    payload: bytes, deadline_ns: int | None = None
) -> list[dict[str, Any]]:
    deadline = _bounded_deadline(deadline_ns, PARENT_ARTIFACT_DEADLINE_SECONDS)
    _check_deadline(deadline, "prediction parsing and validation")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("prediction output must be UTF-8 JSONL") from exc
    if len(lines) > MAXIMUM_PREDICTION_LINES:
        raise ValueError("prediction output exceeds its fixed line-count limit")
    for line_number, line in enumerate(lines, 1):
        _check_deadline(deadline, "prediction parsing and validation")
        if not line.strip():
            continue
        if len(line.encode("utf-8")) > MAXIMUM_PREDICTION_LINE_BYTES:
            raise ValueError("prediction line exceeds its fixed byte limit")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"prediction line {line_number} is invalid JSON") from exc
        if not isinstance(value, Mapping) or set(value) != PREDICTION_FIELDS:
            raise ValueError(
                f"prediction line {line_number} does not have the exact schema"
            )
        sample_id = value["sample_id"]
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
            raise ValueError("prediction sample IDs must be unique non-empty text")
        if type(value["passed"]) is not bool:
            raise ValueError("prediction passed must be boolean")
        if value["shadow_only"] is not True or value["control_authorized"] is not False:
            raise ValueError("prediction authorization boundary was relaxed")
        estimate = value["estimated_axial_yaw_rad"]
        hypothesis = value["selected_hypothesis_id"]
        if value["passed"]:
            if isinstance(estimate, bool) or not isinstance(estimate, (int, float)):
                raise ValueError("passed prediction yaw must be numeric")
            if not math.isfinite(float(estimate)) or hypothesis not in EXPECTED_C2_IDS:
                raise ValueError("passed prediction yaw/branch is invalid")
        elif estimate is not None or hypothesis is not None:
            raise ValueError("rejected prediction retained a yaw or branch")
        seen.add(sample_id)
        records.append(dict(value))
    if not records:
        raise ValueError("prediction output is empty")
    return records


def _read_prediction_output(
    path: Path, operation_deadline_ns: int | None = None
) -> list[dict[str, Any]]:
    deadline = _bounded_deadline(
        operation_deadline_ns, PARENT_ARTIFACT_DEADLINE_SECONDS
    )
    fd = os.open(path, _nonblocking_read_flags())
    try:
        payload = _read_open_fd(
            fd,
            "prediction output",
            deadline_ns=deadline,
            maximum_bytes=MAXIMUM_PREDICTION_OUTPUT_BYTES,
        )
    finally:
        os.close(fd)
    return _validated_predictions(payload, deadline)


def _read_truth_after_child(
    truth_root: Path,
    anonymous_to_original: Mapping[str, str],
    operation_deadline_ns: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    operation_deadline = (
        _deadline_ns(PARENT_OPERATION_DEADLINE_SECONDS)
        if operation_deadline_ns is None
        else operation_deadline_ns
    )
    _check_deadline(operation_deadline, "truth reveal")
    if len(anonymous_to_original) > MAXIMUM_SAMPLE_COUNT:
        raise ValueError("truth reveal exceeds the fixed record-count limit")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    root_fd = os.open(truth_root, flags)
    truth_first_open_monotonic_ns = time.monotonic_ns()
    root_before = os.fstat(root_fd)
    if not stat.S_ISDIR(root_before.st_mode):
        os.close(root_fd)
        raise ValueError("truth_root must be a real directory")
    records: list[dict[str, Any]] = []
    try:
        expected_files = {
            f"{original_id}.json" for original_id in anonymous_to_original.values()
        }
        if set(os.listdir(root_fd)) != expected_files:
            raise ValueError("truth_root must contain exactly one record per sample")
        for anonymous_id, original_id in anonymous_to_original.items():
            _check_deadline(operation_deadline, "truth reveal")
            filename = f"{original_id}.json"
            record_deadline = _bounded_deadline(
                operation_deadline, PARENT_ARTIFACT_DEADLINE_SECONDS
            )
            fd = os.open(filename, _nonblocking_read_flags(), dir_fd=root_fd)
            try:
                payload = _read_open_fd(
                    fd,
                    "truth record",
                    deadline_ns=record_deadline,
                    maximum_bytes=MAXIMUM_TRUTH_RECORD_BYTES,
                )
            finally:
                os.close(fd)
            _check_deadline(record_deadline, "truth record parsing")
            try:
                value = json.loads(payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("truth record must be strict UTF-8 JSON") from exc
            _check_deadline(record_deadline, "truth record validation")
            if not isinstance(value, Mapping) or value.get("sample_id") != original_id:
                raise ValueError("truth record identity does not match its file")
            remapped = dict(value)
            remapped["sample_id"] = anonymous_id
            records.append(remapped)
        root_after = os.fstat(root_fd)
        if _fd_identity(root_before) != _fd_identity(root_after):
            raise RuntimeError(
                "truth directory changed while read through its opened fd"
            )
    finally:
        os.close(root_fd)
    return records, {
        "truth_first_open_monotonic_ns": truth_first_open_monotonic_ns,
        "truth_opened_only_after_prediction_validation": True,
        "truth_fd_identity_checked": True,
        "truth_fd_dev_inode_ctime_checked": True,
        "truth_read_toctou_checked": True,
    }


def _generated_reveal(manifest: Mapping[str, Any], truth_count: int) -> dict[str, Any]:
    completed = datetime.fromisoformat(
        manifest["prediction_completed_at_utc"].replace("Z", "+00:00")
    )
    revealed = datetime.now(timezone.utc)
    if revealed <= completed:
        revealed = completed + timedelta(microseconds=1)
    return {
        "schema_version": CLAIMED_REVEAL_SCHEMA_VERSION,
        "status": "CALLER_CLAIMS_TRUTH_AVAILABLE_FOR_EVALUATION",
        "run_id": manifest["run_id"],
        "dataset_tag": manifest["dataset_tag"],
        "truth_reveal_id": f"isolated-{secrets.token_hex(16)}",
        "truth_record_count": truth_count,
        "truth_revealed_at_utc": revealed.isoformat(timespec="microseconds").replace(
            "+00:00", "Z"
        ),
        "prediction_manifest_path": manifest["prediction_manifest_path"],
    }


def _run_existing_numeric_core(
    predictions: Sequence[Mapping[str, Any]],
    truth_records: Sequence[Mapping[str, Any]],
    evaluation_directory: Path,
    *,
    keyed_model_id: str,
    dataset_tag: str,
    declared_yaw_values_rad: Sequence[float],
    declared_strata: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    artifact = evaluation_directory / "predictions.jsonl"
    manifest_path = evaluation_directory / "predictions.manifest.json"
    run_id = f"isolated-{secrets.token_hex(16)}"
    manifest = write_local_key_yaw_prediction_artifact(
        predictions,
        prediction_artifact_path=artifact,
        prediction_manifest_path=manifest_path,
        dataset_tag=dataset_tag,
        run_id=run_id,
        declared_yaw_values_rad=declared_yaw_values_rad,
        declared_strata=declared_strata,
    )
    report = evaluate_local_key_yaw_benchmark_metrics(
        manifest_path,
        truth_records,
        keyed_model_id=keyed_model_id,
        dataset_tag=dataset_tag,
        claimed_reveal_metadata=_generated_reveal(manifest, len(truth_records)),
    )
    sanitized = dict(report)
    local_manifest = dict(sanitized.get("local_prediction_manifest", {}))
    if local_manifest:
        removed = "ISOLATED_TEMPORARY_ARTIFACT_REMOVED"
        local_manifest["prediction_artifact_path"] = removed
        local_manifest["prediction_manifest_path"] = removed
        sanitized["local_prediction_manifest"] = local_manifest
    claimed = dict(sanitized.get("caller_claimed_reveal_metadata", {}))
    if claimed:
        claimed["prediction_manifest_path"] = "ISOLATED_TEMPORARY_ARTIFACT_REMOVED"
        sanitized["caller_claimed_reveal_metadata"] = claimed
    sanitized["passed"] = False
    sanitized["formal_withheld_evidence_verified"] = False
    for field in CONTROL_FIELDS:
        sanitized[field] = False
    return sanitized


def evaluate_isolated_key_yaw_benchmark(
    observation_root: Path | str,
    truth_root: Path | str,
    predictor_path: Path | str,
    *,
    keyed_model_id: str,
    dataset_tag: str,
    declared_yaw_values_rad: Sequence[float],
    declared_strata: Mapping[str, Sequence[str]],
    timeout_seconds: float = 300.0,
) -> dict[str, Any]:
    """Run anonymous prediction in bwrap, then reveal truth to the parent.

    ``truth_root`` is deliberately not resolved, statted, listed, or opened
    until the child has exited and the prediction file has passed exact-schema
    validation.  Its JSON records must match the current local numerical-core
    truth schema.  No receipt can be supplied by the caller.
    """
    if not os.path.isabs(os.fspath(truth_root)):
        raise ValueError("truth_root must be absolute")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, Real)
        or not math.isfinite(float(timeout_seconds))
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be positive")
    observation_path = Path(observation_root).expanduser()
    predictor_source = Path(predictor_path).expanduser()
    if observation_path.is_symlink() or predictor_source.is_symlink():
        raise ValueError("observation_root and predictor_path must not be symlinks")
    observation_path = observation_path.absolute()
    predictor_source = predictor_source.absolute()
    truth_path = Path(os.fspath(truth_root))
    operation_deadline_ns = _deadline_ns(PARENT_OPERATION_DEADLINE_SECONDS)
    with tempfile.TemporaryDirectory(prefix="kcg-yaw-isolated-") as temporary:
        staging = Path(temporary)
        staging.chmod(0o700)
        anonymous_input = staging / "input"
        anonymous_input.mkdir(mode=0o700)
        output = staging / "output"
        output.mkdir(mode=0o700)
        predictor_directory = staging / "predictor"
        predictor_directory.mkdir(mode=0o700)
        _copy_predictor(predictor_source, predictor_directory / "predictor.py")
        predictor_directory.chmod(0o500)
        anonymous_to_original, creation_order = _stage_observations(
            observation_path, anonymous_input, operation_deadline_ns
        )
        command = _bwrap_command(staging)
        _check_deadline(operation_deadline_ns, "pre-child isolated evaluation")
        child_started_monotonic_ns = time.monotonic_ns()
        remaining_operation_seconds = (
            operation_deadline_ns - child_started_monotonic_ns
        ) / 1_000_000_000
        if remaining_operation_seconds <= 0.0:
            raise TimeoutError("isolated evaluation exceeded the parent deadline")
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=min(float(timeout_seconds), remaining_operation_seconds),
            check=False,
            close_fds=True,
        )
        child_completed_monotonic_ns = time.monotonic_ns()
        if completed.returncode != 0:
            detail = completed.stderr.strip()[-2000:]
            raise IsolatedPredictionError(
                f"isolated predictor exited {completed.returncode}: {detail}"
            )
        if {item.name for item in output.iterdir()} != {"predictions.jsonl"}:
            raise ValueError("sandbox output must contain exactly predictions.jsonl")
        predictions = _read_prediction_output(
            output / "predictions.jsonl", operation_deadline_ns
        )
        if {item["sample_id"] for item in predictions} != set(anonymous_to_original):
            raise ValueError(
                "prediction output does not cover exactly the staged samples"
            )
        prediction_validated_monotonic_ns = time.monotonic_ns()
        truth_records, truth_receipt = _read_truth_after_child(
            truth_path, anonymous_to_original, operation_deadline_ns
        )
        if not (
            child_completed_monotonic_ns
            <= prediction_validated_monotonic_ns
            <= truth_receipt["truth_first_open_monotonic_ns"]
        ):
            raise RuntimeError("truth reveal ordering invariant failed")
        evaluation_directory = staging / "parent-evaluation"
        evaluation_directory.mkdir(mode=0o700)
        numeric_report = _run_existing_numeric_core(
            predictions,
            truth_records,
            evaluation_directory,
            keyed_model_id=keyed_model_id,
            dataset_tag=dataset_tag,
            declared_yaw_values_rad=declared_yaw_values_rad,
            declared_strata=declared_strata,
        )
        receipt = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "receipt_id": f"os-isolation-{secrets.token_urlsafe(24)}",
            "generated_by": "PARENT_ISOLATED_EVALUATOR_NOT_CALLER",
            "generated_at_utc": _utc_now(),
            "bwrap_executable_fixed": BWRAP_PATH,
            "bwrap_unshare_all": True,
            "bwrap_new_session": True,
            "bwrap_die_with_parent": True,
            "environment_cleared": True,
            "child_inherited_secret_fds": False,
            "network_namespace_unshared": True,
            "host_network_configuration_mounted": False,
            "anonymous_observation_copy": True,
            "observation_raw_bytes_copied": False,
            "observation_canonical_decode_reencode": True,
            "observation_source_single_link_required": True,
            "staged_observation_fresh_inode_verified": True,
            "anonymous_ids_from_secrets": True,
            "sample_order_from_secrets": True,
            "anonymous_mapping_mounted": False,
            "anonymous_mapping_persisted": False,
            "staged_sample_count": len(creation_order),
            "observation_files_per_sample": len(OBSERVATION_FILES),
            "truth_mounted_in_sandbox": False,
            "original_dataset_root_mounted": False,
            "workspace_mounted": False,
            "runtime_mounts_read_only": True,
            "predictor_mount_read_only": True,
            "only_host_writable_bind_is_empty_output": True,
            "prediction_exact_schema_validated": True,
            "prediction_output_fd_identity_checked": True,
            "prediction_output_read_toctou_checked": True,
            "parent_nonblocking_file_opens": True,
            "parent_fixed_operation_deadline_seconds": (
                PARENT_OPERATION_DEADLINE_SECONDS
            ),
            "parent_fixed_artifact_deadline_seconds": (
                PARENT_ARTIFACT_DEADLINE_SECONDS
            ),
            "prediction_output_maximum_bytes": MAXIMUM_PREDICTION_OUTPUT_BYTES,
            "prediction_output_maximum_lines": MAXIMUM_PREDICTION_LINES,
            "child_started_monotonic_ns": child_started_monotonic_ns,
            "child_completed_monotonic_ns": child_completed_monotonic_ns,
            "prediction_validated_monotonic_ns": prediction_validated_monotonic_ns,
            **truth_receipt,
            "os_isolation_receipt_verified": True,
            "same_account_local_execution": True,
            "independent_heldout_operator_verified": False,
            "formal_withheld_evidence_verified": False,
            "formal_withheld_evidence_status": (
                "RECEIPT_ONLY_FUTURE_DEDICATED_FORMAL_ENTRY_REQUIRED"
            ),
            "hash_based_seal_used": False,
            "shadow_authorized": False,
            "control_authorized": False,
            "selected_for_control_allowed": False,
            "simulation_insertion_control_authorized": False,
            "robot_control_authorized": False,
            "hardware_control_authorized": False,
        }
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "OS_ISOLATED_NUMERIC_EVALUATION_FORMAL_WITHHELD_NOT_VERIFIED",
        "passed": False,
        "metric_gates_passed": numeric_report.get("metric_gates_passed") is True,
        "formal_withheld_evidence_verified": False,
        "formal_withheld_evidence_status": (
            "RECEIPT_ONLY_FUTURE_DEDICATED_FORMAL_ENTRY_REQUIRED"
        ),
        "same_account_local_execution": True,
        "os_isolation_receipt": receipt,
        "numeric_report": numeric_report,
    }
    for field in CONTROL_FIELDS:
        result[field] = False
    return result


__all__ = [
    "BWRAP_PATH",
    "IsolatedPredictionError",
    "OBSERVATION_FILES",
    "RECEIPT_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "evaluate_isolated_key_yaw_benchmark",
]
