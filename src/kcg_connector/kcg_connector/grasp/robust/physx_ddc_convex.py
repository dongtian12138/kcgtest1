"""Fail-closed offline extraction of the frozen nail-free PhysX hulls.

This module reads the two content-addressed convex-decomposition aggregates
named by an explicit JSON contract.  It does not import or start Kit, Isaac,
or PhysX.  Only the CVXM 14 / CLHL 9 layout used by the frozen cache is
accepted; this is intentionally not a general DerivedDataCache reader.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import io
import json
import math
import mmap
from pathlib import Path
import re
import struct
from typing import Any, Iterable, Mapping, Sequence
import zipfile

import numpy as np


SCHEMA_VERSION = "kcg_nailfree_physx_cooked_v2"
MANIFEST_SCHEMA_VERSION = "kcg_nailfree_physx_cooked_extract_v1"
INDEX_RECORD_SIZE = 72
SMALL_INDEX_MAGIC = bytes.fromhex("efbeaddeefbeafde")
LARGE_INDEX_MAGIC = bytes.fromhex("effaebfeeddaebfe")
DATA_RECORD_MAGIC = bytes.fromhex("efcdab9078563412")
DDC_KEY_SIZE = 32
DATA_HEADER_SIZE = 8 + DDC_KEY_SIZE
DATA_FOOTER_SIZE = 8
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_POINT_RE = re.compile(r"\(([^()]*)\)")
_EXPECTED_GROUP_IDS = {
    "f1_f3_shared_cooked_convex",
    "f2_cooked_convex",
}
_EXPECTED_GROUP_BINDINGS = {
    "f1_f3_shared_cooked_convex": {
        "asset_collision_mesh_ids": ("f1_collision", "f3_collision"),
        "terminal_links": ("f1Link3", "f3Link3"),
        "output_name": "f1_f3_cooked_convex.npz",
        "aggregate_key_hex": (
            "000200383e2a51b30e9b7fa0ca0d5534e359d3f1fe02ba9482e3d9674bed117b"
        ),
    },
    "f2_cooked_convex": {
        "asset_collision_mesh_ids": ("f2_collision",),
        "terminal_links": ("f2Link2",),
        "output_name": "f2_cooked_convex.npz",
        "aggregate_key_hex": (
            "0002003880c02f4ae45837c9a75421e1cec58593db7e45be572dec1ab4c2f845"
        ),
    },
}
_EXPECTED_EVIDENCE_BOUNDARY = {
    "binding_strength": (
        "TRACE_ASSET_HASH_PLUS_EXACT_STL_TO_USDA_TRANSFORM_PLUS_FROZEN_"
        "DDC_HASH_TIME_AND_GEOMETRY_MATCH_NOT_DIRECT_ASSET_HASH_INSIDE_DDC"
    ),
    "cache_is_mutable_outside_this_hash_frozen_snapshot": True,
    "classification": "OFFLINE_COOKED_CACHE_RECONSTRUCTION",
    "ddc_contains_direct_robot_asset_or_source_stl_sha256": False,
    "does_not_prove": [
        "which individual cooked hull generated a historical contact point",
        "TE connector grasp success",
        "50 mm lift or 2 s suspended hold",
        "dynamic robustness",
        "hardware behavior",
        "analytic global optimality",
    ],
    "engine_logs_retained": False,
    "hardware_authorized": False,
    "online_control_role_allowed": False,
    "runtime_physx_api_queried_by_extractor": False,
    "simulation_only": True,
    "stored_polygon_planes_are_preserved_but_not_claimed_as_exactly_"
    "coplanar_with_every_ring_vertex": True,
}


class ExtractionError(RuntimeError):
    """Raised when frozen evidence or cooked geometry fails validation."""


@dataclass(frozen=True)
class IndexRecord:
    storage: str
    index_offset: int
    data_offset: int
    payload_size: int
    key: bytes


@dataclass(frozen=True)
class CookedHull:
    key: bytes
    storage: str
    data_offset: int
    payload_size: int
    payload_sha256: str
    vertices: np.ndarray
    polygon_planes: np.ndarray
    polygon_vertex_indices: tuple[np.ndarray, ...]
    triangle_faces: np.ndarray
    stored_mass_m3: float
    signed_volume_m3: float
    bounds_m: np.ndarray
    diagnostics: Mapping[str, float]


def _fail(message: str) -> None:
    raise ExtractionError(message)


def _exact_keys(value: object, expected: Iterable[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{label} must be an object")
    expected_set = set(expected)
    actual_set = set(value)
    if actual_set != expected_set:
        _fail(
            f"{label} keys differ: missing={sorted(expected_set - actual_set)}, "
            f"extra={sorted(actual_set - expected_set)}"
        )
    return value


def _required_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{label} must be a non-empty string")
    return value


def _required_int(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(f"{label} must be an integer >= {minimum}")
    return value


def _required_float(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        _fail(f"{label} must be {'positive and ' if positive else ''}finite")
    return result


def _required_sha256(value: object, label: str) -> str:
    result = _required_string(value, label).lower()
    if _SHA256_RE.fullmatch(result) is None:
        _fail(f"{label} must be a lowercase SHA-256 digest")
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_file(specification: object, label: str) -> Path:
    spec = _exact_keys(
        specification, {"path", "size_bytes", "mtime_ns", "sha256"}, label
    )
    path = Path(_required_string(spec["path"], f"{label}.path"))
    if not path.is_absolute() or not path.is_file():
        _fail(f"{label}.path is not an existing absolute file: {path}")
    stat = path.stat()
    expected_size = _required_int(spec["size_bytes"], f"{label}.size_bytes")
    expected_mtime = _required_int(spec["mtime_ns"], f"{label}.mtime_ns")
    expected_hash = _required_sha256(spec["sha256"], f"{label}.sha256")
    if stat.st_size != expected_size:
        _fail(f"{label} size mismatch: {stat.st_size} != {expected_size}")
    if stat.st_mtime_ns != expected_mtime:
        _fail(f"{label} mtime mismatch: {stat.st_mtime_ns} != {expected_mtime}")
    actual_hash = _sha256_file(path)
    if actual_hash != expected_hash:
        _fail(f"{label} SHA-256 mismatch: {actual_hash} != {expected_hash}")
    return path


def _read_contract(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        _fail(f"contract is not a file: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExtractionError(f"cannot read contract {path}: {error}") from error
    contract = _exact_keys(
        raw,
        {
            "schema_version",
            "simulation_only",
            "format_lock",
            "validation",
            "source_files",
            "asset_collision_contract",
            "runtime_binding_evidence",
            "ddc",
            "aggregates",
            "evidence_boundary",
        },
        "contract",
    )
    if contract["schema_version"] != SCHEMA_VERSION:
        _fail(f"unsupported contract schema: {contract['schema_version']!r}")
    if contract["simulation_only"] is not True:
        _fail("contract.simulation_only must be true")
    return contract


def _verify_format_lock(value: object) -> Mapping[str, Any]:
    lock = _exact_keys(
        value,
        {
            "cvxm_version",
            "clhl_version",
            "cvxm_serial_flags",
            "little_endian_marker",
            "aggregate_child_count",
            "aggregate_child_key_offset",
            "aggregate_child_key_stride",
            "require_grb_edges",
            "require_no_gauss_map",
            "require_no_sdf",
        },
        "format_lock",
    )
    required = {
        "cvxm_version": 14,
        "clhl_version": 9,
        "cvxm_serial_flags": 0,
        "little_endian_marker": 1,
        "aggregate_child_count": 32,
        "aggregate_child_key_offset": 80,
        "aggregate_child_key_stride": 32,
        "require_grb_edges": True,
        "require_no_gauss_map": True,
        "require_no_sdf": True,
    }
    if dict(lock) != required:
        _fail(f"format_lock differs from the only supported layout: {lock!r}")
    return lock


def _verify_validation(value: object) -> Mapping[str, float]:
    document = _exact_keys(
        value,
        {
            "aabb_tolerance_m",
            "normal_norm_tolerance",
            "min_index_projection_tolerance_m",
            "volume_relative_tolerance",
            "source_transform_tolerance_m",
        },
        "validation",
    )
    return {
        key: _required_float(raw, f"validation.{key}", positive=True)
        for key, raw in document.items()
    }


def _verify_evidence_boundary(value: object) -> Mapping[str, Any]:
    boundary = _exact_keys(
        value, _EXPECTED_EVIDENCE_BOUNDARY.keys(), "evidence_boundary"
    )
    if boundary != _EXPECTED_EVIDENCE_BOUNDARY:
        _fail("evidence_boundary differs from the frozen simulation-only limits")
    return boundary


def _ordered_stl_vertices(path: Path) -> np.ndarray:
    vertices: list[tuple[float, float, float]] = []
    with path.open("r", encoding="latin-1") as stream:
        for line_number, line in enumerate(stream, 1):
            fields = line.split()
            if fields and fields[0] == "vertex":
                if len(fields) != 4:
                    _fail(f"malformed STL vertex at line {line_number}")
                try:
                    vertex = tuple(float(field) for field in fields[1:])
                except ValueError as error:
                    raise ExtractionError(
                        f"non-numeric STL vertex at line {line_number}"
                    ) from error
                if not all(math.isfinite(component) for component in vertex):
                    _fail(f"non-finite STL vertex at line {line_number}")
                vertices.append(vertex)  # type: ignore[arg-type]
    result = np.asarray(vertices, dtype=np.float64)
    if result.ndim != 2 or result.shape[1:] != (3,):
        _fail("source STL did not yield an (N, 3) vertex sequence")
    return result


def _asset_collision_point_arrays(path: Path) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    waiting_for_points = False
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if 'def Mesh "nailfree_collision"' in line:
                if waiting_for_points:
                    _fail("nested nailfree_collision definitions before points")
                waiting_for_points = True
                continue
            if waiting_for_points and "point3f[] points = [" in line:
                points: list[tuple[float, float, float]] = []
                for match in _POINT_RE.findall(line):
                    fields = [field.strip() for field in match.split(",")]
                    if len(fields) != 3:
                        _fail(f"malformed USDA point at line {line_number}")
                    try:
                        point = tuple(float(field) for field in fields)
                    except ValueError as error:
                        raise ExtractionError(
                            f"non-numeric USDA point at line {line_number}"
                        ) from error
                    if not all(math.isfinite(component) for component in point):
                        _fail(f"non-finite USDA point at line {line_number}")
                    points.append(point)  # type: ignore[arg-type]
                result.append(np.asarray(points, dtype=np.float64))
                waiting_for_points = False
    if waiting_for_points:
        _fail("last nailfree_collision has no points array")
    return result


def _exact_float_matrix(value: object, shape: tuple[int, ...], label: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ExtractionError(f"{label} is not numeric") from error
    if result.shape != shape or not np.all(np.isfinite(result)):
        _fail(f"{label} must be finite with shape {shape}, got {result.shape}")
    return result


def _verify_sources_and_binding(
    contract: Mapping[str, Any], validation: Mapping[str, float]
) -> tuple[Path, Path, Mapping[str, Mapping[str, Any]], Mapping[str, Any]]:
    sources = _exact_keys(
        contract["source_files"], {"robot_asset", "source_stl"}, "source_files"
    )
    asset_spec = _exact_keys(
        sources["robot_asset"],
        {"path", "size_bytes", "mtime_ns", "sha256"},
        "source_files.robot_asset",
    )
    stl_spec = _exact_keys(
        sources["source_stl"],
        {
            "path",
            "size_bytes",
            "mtime_ns",
            "sha256",
            "triangle_count",
            "ordered_vertex_count",
        },
        "source_files.source_stl",
    )
    asset_path = _verified_file(asset_spec, "source_files.robot_asset")
    stl_path = _verified_file(
        {key: stl_spec[key] for key in ("path", "size_bytes", "mtime_ns", "sha256")},
        "source_files.source_stl",
    )
    stl_vertices_mm = _ordered_stl_vertices(stl_path)
    expected_vertex_count = _required_int(
        stl_spec["ordered_vertex_count"], "source_files.source_stl.ordered_vertex_count", minimum=1
    )
    expected_triangle_count = _required_int(
        stl_spec["triangle_count"], "source_files.source_stl.triangle_count", minimum=1
    )
    if stl_vertices_mm.shape[0] != expected_vertex_count:
        _fail("source STL ordered vertex count mismatch")
    if expected_vertex_count != 3 * expected_triangle_count:
        _fail("source STL contract does not describe a triangle soup")

    asset_contract = _exact_keys(
        contract["asset_collision_contract"],
        {"required_text_occurrences", "meshes"},
        "asset_collision_contract",
    )
    text = asset_path.read_text(encoding="utf-8")
    occurrences = asset_contract["required_text_occurrences"]
    if not isinstance(occurrences, dict) or not occurrences:
        _fail("asset_collision_contract.required_text_occurrences must be non-empty")
    for needle, expected_raw in occurrences.items():
        needle_text = _required_string(needle, "asset text occurrence key")
        expected = _required_int(expected_raw, f"occurrence count for {needle_text!r}")
        actual = text.count(needle_text)
        if actual != expected:
            _fail(f"asset occurrence mismatch for {needle_text!r}: {actual} != {expected}")

    asset_points = _asset_collision_point_arrays(asset_path)
    raw_meshes = asset_contract["meshes"]
    if not isinstance(raw_meshes, list) or len(raw_meshes) != len(asset_points):
        _fail("asset collision mesh contract count differs from USDA")
    audited: dict[str, Mapping[str, Any]] = {}
    used_occurrences: set[int] = set()
    for index, raw in enumerate(raw_meshes):
        row = _exact_keys(
            raw,
            {
                "mesh_id",
                "occurrence_index_zero_based",
                "terminal_link",
                "point_count",
                "float32_sequence_sha256",
                "bounds_m",
                "source_transform",
            },
            f"asset_collision_contract.meshes[{index}]",
        )
        mesh_id = _required_string(row["mesh_id"], f"mesh[{index}].mesh_id")
        if mesh_id in audited:
            _fail(f"duplicate asset collision mesh id: {mesh_id}")
        occurrence = _required_int(
            row["occurrence_index_zero_based"], f"mesh[{index}].occurrence_index_zero_based"
        )
        if occurrence >= len(asset_points) or occurrence in used_occurrences:
            _fail(f"invalid or duplicate asset collision occurrence: {occurrence}")
        used_occurrences.add(occurrence)
        points = asset_points[occurrence]
        expected_count = _required_int(row["point_count"], f"mesh[{index}].point_count", minimum=1)
        if points.shape != (expected_count, 3):
            _fail(f"asset collision point count mismatch for {mesh_id}")
        point_digest = hashlib.sha256(
            np.asarray(points, dtype="<f4", order="C").tobytes(order="C")
        ).hexdigest()
        expected_digest = _required_sha256(
            row["float32_sequence_sha256"], f"mesh[{index}].float32_sequence_sha256"
        )
        if point_digest != expected_digest:
            _fail(f"asset collision point hash mismatch for {mesh_id}")
        bounds = np.stack((points.min(axis=0), points.max(axis=0)))
        expected_bounds = _exact_float_matrix(row["bounds_m"], (2, 3), f"mesh[{index}].bounds_m")
        if not np.array_equal(bounds, expected_bounds):
            _fail(f"asset collision bounds mismatch for {mesh_id}: {bounds.tolist()}")

        transform = _exact_keys(
            row["source_transform"], {"translation_m", "yaw_rad"}, f"mesh[{index}].source_transform"
        )
        translation = _exact_float_matrix(
            transform["translation_m"], (3,), f"mesh[{index}].source_transform.translation_m"
        )
        yaw = _required_float(transform["yaw_rad"], f"mesh[{index}].source_transform.yaw_rad")
        source_m = stl_vertices_mm / 1000.0
        cosine, sine = math.cos(yaw), math.sin(yaw)
        predicted = np.empty_like(source_m)
        predicted[:, 0] = cosine * source_m[:, 0] - sine * source_m[:, 1] + translation[0]
        predicted[:, 1] = sine * source_m[:, 0] + cosine * source_m[:, 1] + translation[1]
        predicted[:, 2] = source_m[:, 2] + translation[2]
        max_residual = float(np.linalg.norm(predicted - points, axis=1).max())
        if max_residual > validation["source_transform_tolerance_m"]:
            _fail(f"source-to-asset transform residual too large for {mesh_id}: {max_residual}")
        audited[mesh_id] = {
            "terminal_link": _required_string(row["terminal_link"], f"mesh[{index}].terminal_link"),
            "point_count": expected_count,
            "float32_sequence_sha256": point_digest,
            "bounds_m": bounds.tolist(),
            "max_source_transform_residual_m": max_residual,
        }
    terminal_bindings = [str(row["terminal_link"]) for row in audited.values()]
    if len(set(terminal_bindings)) != len(terminal_bindings):
        _fail("asset collision meshes do not bind distinct terminal links")

    runtime = _exact_keys(
        contract["runtime_binding_evidence"],
        {"file", "required_json_values"},
        "runtime_binding_evidence",
    )
    runtime_path = _verified_file(runtime["file"], "runtime_binding_evidence.file")
    try:
        runtime_json = json.loads(runtime_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExtractionError(f"cannot parse runtime binding evidence: {error}") from error
    required_values = runtime["required_json_values"]
    if not isinstance(required_values, list) or not required_values:
        _fail("runtime_binding_evidence.required_json_values must be non-empty")
    for item_index, item_raw in enumerate(required_values):
        item = _exact_keys(item_raw, {"path", "value"}, f"required_json_values[{item_index}]")
        key_path = item["path"]
        if not isinstance(key_path, list) or not key_path:
            _fail(f"required_json_values[{item_index}].path must be a non-empty list")
        current: Any = runtime_json
        for key in key_path:
            key_string = _required_string(key, f"required_json_values[{item_index}].path component")
            if not isinstance(current, dict) or key_string not in current:
                _fail(f"runtime binding JSON path is absent: {key_path!r}")
            current = current[key_string]
        if current != item["value"]:
            _fail(f"runtime binding JSON mismatch at {key_path!r}: {current!r}")
    return asset_path, stl_path, audited, runtime["file"]


def _build_index(
    index_path: Path, expected_small_count: int, expected_large_count: int
) -> Mapping[bytes, IndexRecord]:
    content = index_path.read_bytes()
    if len(content) % INDEX_RECORD_SIZE:
        _fail("DDC cacheindex length is not a multiple of 72")
    records: dict[bytes, IndexRecord] = {}
    counts = Counter()
    for offset in range(0, len(content), INDEX_RECORD_SIZE):
        magic = content[offset : offset + 8]
        if magic == SMALL_INDEX_MAGIC:
            storage = "small"
        elif magic == LARGE_INDEX_MAGIC:
            storage = "large"
        else:
            _fail(f"unknown DDC index magic at offset {offset}: {magic.hex()}")
        key = content[offset + 8 : offset + 40]
        data_offset, payload_size = struct.unpack_from("<QQ", content, offset + 40)
        if key in records:
            _fail(f"duplicate DDC key in cacheindex: {key.hex()}")
        records[key] = IndexRecord(storage, offset, data_offset, payload_size, key)
        counts[storage] += 1
    if counts != Counter({"small": expected_small_count, "large": expected_large_count}):
        _fail(f"DDC index storage counts differ: {dict(counts)}")
    return records


def _record_payload(record: IndexRecord, data: mmap.mmap) -> bytes:
    end = record.data_offset + DATA_HEADER_SIZE + record.payload_size + DATA_FOOTER_SIZE
    if end > len(data):
        _fail(f"DDC record extends past {record.storage} data file: {record.key.hex()}")
    if data[record.data_offset : record.data_offset + 8] != DATA_RECORD_MAGIC:
        _fail(f"DDC data magic mismatch for {record.key.hex()}")
    if data[record.data_offset + 8 : record.data_offset + 40] != record.key:
        _fail(f"DDC data key mismatch for {record.key.hex()}")
    begin = record.data_offset + DATA_HEADER_SIZE
    return bytes(data[begin : begin + record.payload_size])


def _unpack_from(fmt: str, payload: bytes, offset: int, label: str) -> tuple[tuple[Any, ...], int]:
    size = struct.calcsize(fmt)
    if offset + size > len(payload):
        _fail(f"truncated {label}")
    return struct.unpack_from(fmt, payload, offset), offset + size


def _parse_cvxm(
    payload: bytes,
    record: IndexRecord,
    payload_sha256: str,
    lock: Mapping[str, Any],
    validation: Mapping[str, float],
) -> CookedHull:
    if payload[:4] != b"NXS\x01" or payload[4:8] != b"CVXM":
        _fail(f"child {record.key.hex()} is not little-endian NXS/CVXM")
    (cvxm_version, serial_flags), cursor = _unpack_from("<II", payload, 8, "CVXM header")
    if cvxm_version != lock["cvxm_version"] or serial_flags != lock["cvxm_serial_flags"]:
        _fail(f"CVXM version/flags mismatch for {record.key.hex()}")
    if payload[cursor : cursor + 8] != b"ICE\x01CLHL":
        _fail(f"child {record.key.hex()} lacks little-endian ICE/CLHL")
    cursor += 8
    values, cursor = _unpack_from("<IIIII", payload, cursor, "CLHL counts")
    clhl_version, vertex_count, raw_edge_count, polygon_count, reference_count = values
    if clhl_version != lock["clhl_version"]:
        _fail(f"CLHL version mismatch for {record.key.hex()}")
    if raw_edge_count & 0xFFFF0000:
        _fail(f"unsupported CLHL edge flags for {record.key.hex()}")
    has_grb = bool(raw_edge_count & 0x8000)
    edge_count = raw_edge_count & 0x7FFF
    if lock["require_grb_edges"] is True and not has_grb:
        _fail(f"CLHL child lacks GRB edges: {record.key.hex()}")
    if not (4 <= vertex_count <= 255 and edge_count >= 6 and polygon_count >= 4):
        _fail(f"invalid CLHL topology counts for {record.key.hex()}")
    if vertex_count - edge_count + polygon_count != 2:
        _fail(f"Euler characteristic mismatch for {record.key.hex()}")

    vertex_bytes = 12 * vertex_count
    if cursor + vertex_bytes > len(payload):
        _fail(f"truncated CLHL vertices for {record.key.hex()}")
    vertices = np.frombuffer(payload, dtype="<f4", count=3 * vertex_count, offset=cursor).copy().reshape(-1, 3)
    cursor += vertex_bytes
    if not np.all(np.isfinite(vertices)):
        _fail(f"non-finite CLHL vertices for {record.key.hex()}")

    planes = np.empty((polygon_count, 4), dtype=np.float32)
    vertex_starts = np.empty(polygon_count, dtype=np.uint16)
    polygon_sizes = np.empty(polygon_count, dtype=np.uint8)
    minimum_indices = np.empty(polygon_count, dtype=np.uint8)
    for polygon_index in range(polygon_count):
        values, cursor = _unpack_from("<ffffHBB", payload, cursor, "CLHL polygon")
        planes[polygon_index] = values[:4]
        vertex_starts[polygon_index] = values[4]
        polygon_sizes[polygon_index] = values[5]
        minimum_indices[polygon_index] = values[6]
    if not np.all(np.isfinite(planes)):
        _fail(f"non-finite polygon planes for {record.key.hex()}")

    def take_array(dtype: str | np.dtype[Any], count: int, label: str) -> np.ndarray:
        nonlocal cursor
        item_size = np.dtype(dtype).itemsize
        if cursor + count * item_size > len(payload):
            _fail(f"truncated {label} for {record.key.hex()}")
        result = np.frombuffer(payload, dtype=dtype, count=count, offset=cursor).copy()
        cursor += count * item_size
        return result

    references = take_array(np.uint8, reference_count, "polygon references").astype(np.int64)
    faces_by_edges = take_array(np.uint8, 2 * edge_count, "faces by edges").astype(np.int64).reshape(-1, 2)
    faces_by_vertices = take_array(np.uint8, 3 * vertex_count, "faces by vertices").astype(np.int64).reshape(-1, 3)
    if has_grb:
        grb_edges = take_array("<u2", 2 * edge_count, "GRB edges").astype(np.int64).reshape(-1, 2)
    else:
        grb_edges = np.empty((0, 2), dtype=np.int64)

    bounds_and_mass = take_array("<f4", 8, "bounds and mass").astype(np.float64)
    if not np.all(np.isfinite(bounds_and_mass)) or bounds_and_mass[0] != 0.0:
        _fail(f"unsupported bounds/mass block for {record.key.hex()}")
    stored_bounds = np.stack((bounds_and_mass[1:4], bounds_and_mass[4:7]))
    stored_mass = float(bounds_and_mass[7])
    if stored_mass <= 0.0:
        _fail(f"non-positive stored mass for {record.key.hex()}")
    inertia_and_center = take_array("<f4", 12, "inertia and center of mass")
    if not np.all(np.isfinite(inertia_and_center)):
        _fail(f"non-finite mass properties for {record.key.hex()}")
    gauss_map_flag = float(take_array("<f4", 1, "gauss-map flag")[0])
    sdf_flag = float(take_array("<f4", 1, "SDF flag")[0])
    if lock["require_no_gauss_map"] is True and gauss_map_flag != -1.0:
        _fail(f"unexpected gauss map for {record.key.hex()}")
    if lock["require_no_sdf"] is True and sdf_flag != -1.0:
        _fail(f"unexpected SDF for {record.key.hex()}")
    internal = take_array("<f4", 4, "internal-object data")
    if not np.all(np.isfinite(internal)) or np.any(internal <= 0.0):
        _fail(f"invalid internal-object data for {record.key.hex()}")
    if cursor != len(payload):
        _fail(f"unparsed CVXM bytes for {record.key.hex()}: {len(payload) - cursor}")

    if int(polygon_sizes.astype(np.int64).sum()) != reference_count or reference_count != 2 * edge_count:
        _fail(f"polygon-reference count mismatch for {record.key.hex()}")
    expected_start = 0
    polygons: list[np.ndarray] = []
    topology_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    directed_edges: Counter[tuple[int, int]] = Counter()
    triangle_rows: list[tuple[int, int, int]] = []
    max_normal_error = 0.0
    min_projected_turn = math.inf
    min_newell_alignment = math.inf
    max_plane_vertex_residual = 0.0
    referenced_vertices: set[int] = set()
    vertices64 = vertices.astype(np.float64)
    for polygon_index in range(polygon_count):
        start = int(vertex_starts[polygon_index])
        size = int(polygon_sizes[polygon_index])
        if size < 3 or start != expected_start or start + size > reference_count:
            _fail(f"non-contiguous polygon references for {record.key.hex()}")
        expected_start += size
        indices = references[start : start + size]
        if np.any(indices < 0) or np.any(indices >= vertex_count) or len(set(indices.tolist())) != size:
            _fail(f"invalid polygon indices for {record.key.hex()}")
        polygons.append(indices.copy())
        referenced_vertices.update(indices.tolist())
        normal = planes[polygon_index, :3].astype(np.float64)
        normal_error = abs(float(np.linalg.norm(normal)) - 1.0)
        max_normal_error = max(max_normal_error, normal_error)
        if normal_error > validation["normal_norm_tolerance"]:
            _fail(f"polygon normal is not unit length for {record.key.hex()}")
        points = vertices64[indices]
        newell = np.zeros(3, dtype=np.float64)
        for point, following in zip(points, np.roll(points, -1, axis=0)):
            newell += np.cross(point, following)
        alignment = float(np.dot(newell, normal))
        min_newell_alignment = min(min_newell_alignment, alignment)
        if not math.isfinite(alignment) or alignment <= 0.0:
            _fail(f"polygon winding disagrees with its plane for {record.key.hex()}")
        for local_index in range(size):
            edge_before = points[local_index] - points[local_index - 1]
            edge_after = points[(local_index + 1) % size] - points[local_index]
            turn = float(np.dot(np.cross(edge_before, edge_after), normal))
            min_projected_turn = min(min_projected_turn, turn)
            if not math.isfinite(turn) or turn <= 0.0:
                _fail(f"polygon is not strictly convex in plane projection for {record.key.hex()}")
        residual = np.abs(points @ normal + float(planes[polygon_index, 3]))
        max_plane_vertex_residual = max(max_plane_vertex_residual, float(residual.max()))
        projections = vertices64 @ normal
        minimum_index = int(minimum_indices[polygon_index])
        if minimum_index >= vertex_count:
            _fail(f"polygon minimum vertex index is out of range for {record.key.hex()}")
        if float(projections[minimum_index] - projections.min()) > validation["min_index_projection_tolerance_m"]:
            _fail(f"polygon minimum vertex index is inconsistent for {record.key.hex()}")
        for first, second in zip(indices, np.roll(indices, -1)):
            a, b = int(first), int(second)
            if a == b:
                _fail(f"zero-length topological edge for {record.key.hex()}")
            directed_edges[(a, b)] += 1
            topology_faces[tuple(sorted((a, b)))].append(polygon_index)
        for fan_index in range(1, size - 1):
            triangle = (int(indices[0]), int(indices[fan_index]), int(indices[fan_index + 1]))
            cross = np.cross(
                vertices64[triangle[1]] - vertices64[triangle[0]],
                vertices64[triangle[2]] - vertices64[triangle[0]],
            )
            if float(np.dot(cross, normal)) <= 0.0:
                _fail(f"fan triangulation winding is invalid for {record.key.hex()}")
            triangle_rows.append(triangle)
    if expected_start != reference_count or referenced_vertices != set(range(vertex_count)):
        _fail(f"CLHL polygons do not cover all vertices for {record.key.hex()}")
    if len(topology_faces) != edge_count:
        _fail(f"CLHL edge count differs from polygon topology for {record.key.hex()}")
    for edge, adjacent in topology_faces.items():
        if len(adjacent) != 2:
            _fail(f"non-closed polygon edge {edge} for {record.key.hex()}")
        if directed_edges[(edge[0], edge[1])] != 1 or directed_edges[(edge[1], edge[0])] != 1:
            _fail(f"polygon edge winding is not paired for {record.key.hex()}")
    expected_face_pairs = sorted(tuple(sorted(value)) for value in topology_faces.values())
    actual_face_pairs = sorted(tuple(sorted(row.tolist())) for row in faces_by_edges)
    if expected_face_pairs != actual_face_pairs:
        _fail(f"faces-by-edges table mismatch for {record.key.hex()}")
    if np.any(faces_by_vertices >= polygon_count):
        _fail(f"faces-by-vertices index out of range for {record.key.hex()}")
    for vertex_index, face_row in enumerate(faces_by_vertices):
        actual_faces = {
            polygon_index
            for polygon_index, indices in enumerate(polygons)
            if vertex_index in indices
        }
        if len(set(face_row.tolist())) != 3 or not set(face_row.tolist()).issubset(actual_faces):
            _fail(f"faces-by-vertices table mismatch for {record.key.hex()}")
    grb_edge_set = {tuple(sorted(row.tolist())) for row in grb_edges}
    if has_grb and grb_edge_set != set(topology_faces):
        _fail(f"GRB edge table mismatch for {record.key.hex()}")

    triangle_faces = np.asarray(triangle_rows, dtype=np.int64)
    first = vertices64[triangle_faces[:, 0]]
    second = vertices64[triangle_faces[:, 1]]
    third = vertices64[triangle_faces[:, 2]]
    signed_volume = float(np.einsum("ij,ij->i", first, np.cross(second, third)).sum() / 6.0)
    if not math.isfinite(signed_volume) or signed_volume <= 0.0:
        _fail(f"non-positive closed-mesh volume for {record.key.hex()}")
    volume_relative_error = abs(signed_volume - stored_mass) / stored_mass
    if volume_relative_error > validation["volume_relative_tolerance"]:
        _fail(f"stored mass/mesh volume mismatch for {record.key.hex()}: {volume_relative_error}")
    computed_bounds = np.stack((vertices64.min(axis=0), vertices64.max(axis=0)))
    max_aabb_residual = float(np.abs(computed_bounds - stored_bounds).max())
    if max_aabb_residual > validation["aabb_tolerance_m"]:
        _fail(f"stored/computed AABB mismatch for {record.key.hex()}: {max_aabb_residual}")
    return CookedHull(
        key=record.key,
        storage=record.storage,
        data_offset=record.data_offset,
        payload_size=record.payload_size,
        payload_sha256=payload_sha256,
        vertices=vertices,
        polygon_planes=planes,
        polygon_vertex_indices=tuple(polygons),
        triangle_faces=triangle_faces,
        stored_mass_m3=stored_mass,
        signed_volume_m3=signed_volume,
        bounds_m=computed_bounds,
        diagnostics={
            "max_normal_norm_error": max_normal_error,
            "min_projected_convex_turn_m2": min_projected_turn,
            "min_newell_plane_alignment_m2": min_newell_alignment,
            "max_stored_plane_vertex_residual_m": max_plane_vertex_residual,
            "max_stored_aabb_residual_m": max_aabb_residual,
            "stored_mass_volume_relative_error": volume_relative_error,
        },
    )


def _group_arrays(hulls: Sequence[CookedHull]) -> Mapping[str, np.ndarray]:
    vertices: list[np.ndarray] = []
    triangles: list[np.ndarray] = []
    triangle_hulls: list[np.ndarray] = []
    polygon_indices: list[np.ndarray] = []
    polygon_offsets = [0]
    polygon_hulls: list[int] = []
    polygon_planes: list[np.ndarray] = []
    hull_vertex_offsets = [0]
    hull_polygon_offsets = [0]
    for hull_index, hull in enumerate(hulls):
        vertex_offset = hull_vertex_offsets[-1]
        vertices.append(hull.vertices)
        triangles.append(hull.triangle_faces + vertex_offset)
        triangle_hulls.append(np.full(hull.triangle_faces.shape[0], hull_index, dtype=np.int32))
        for indices, plane in zip(hull.polygon_vertex_indices, hull.polygon_planes):
            global_indices = indices.astype(np.int64) + vertex_offset
            polygon_indices.append(global_indices)
            polygon_offsets.append(polygon_offsets[-1] + global_indices.size)
            polygon_hulls.append(hull_index)
            polygon_planes.append(plane)
        hull_vertex_offsets.append(vertex_offset + hull.vertices.shape[0])
        hull_polygon_offsets.append(hull_polygon_offsets[-1] + len(hull.polygon_vertex_indices))
    return {
        "vertices": np.concatenate(vertices).astype("<f4", copy=False),
        "triangle_faces": np.concatenate(triangles).astype("<i8", copy=False),
        "triangle_hull_index": np.concatenate(triangle_hulls).astype("<i4", copy=False),
        "polygon_offsets": np.asarray(polygon_offsets, dtype="<i8"),
        "polygon_indices": np.concatenate(polygon_indices).astype("<i8", copy=False),
        "polygon_hull_index": np.asarray(polygon_hulls, dtype="<i4"),
        "polygon_planes": np.asarray(polygon_planes, dtype="<f4"),
        "hull_vertex_offsets": np.asarray(hull_vertex_offsets, dtype="<i8"),
        "hull_polygon_offsets": np.asarray(hull_polygon_offsets, dtype="<i8"),
        "hull_ddc_keys": np.stack(
            [np.frombuffer(hull.key, dtype=np.uint8) for hull in hulls]
        ).astype(np.uint8, copy=False),
        "hull_bounds_m": np.stack([hull.bounds_m for hull in hulls]).astype("<f8", copy=False),
        "hull_stored_mass_m3": np.asarray([hull.stored_mass_m3 for hull in hulls], dtype="<f8"),
        "hull_signed_volume_m3": np.asarray([hull.signed_volume_m3 for hull in hulls], dtype="<f8"),
    }


def _write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    if path.exists():
        _fail(f"refusing to overwrite output: {path}")
    with zipfile.ZipFile(path, mode="x", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for name in sorted(arrays):
            array = np.ascontiguousarray(arrays[name])
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, array, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, buffer.getvalue())


def _aggregate_expected(value: object, label: str) -> Mapping[str, Any]:
    expected = _exact_keys(
        value,
        {
            "hull_count",
            "small_child_count",
            "large_child_count",
            "vertex_count",
            "polygon_count",
            "triangle_count",
            "bounds_m",
            "ordered_child_payload_sha256",
        },
        label,
    )
    for key in (
        "hull_count",
        "small_child_count",
        "large_child_count",
        "vertex_count",
        "polygon_count",
        "triangle_count",
    ):
        _required_int(expected[key], f"{label}.{key}", minimum=1)
    _exact_float_matrix(expected["bounds_m"], (2, 3), f"{label}.bounds_m")
    _required_sha256(expected["ordered_child_payload_sha256"], f"{label}.ordered_child_payload_sha256")
    return expected


def extract(contract_path: Path, output_dir: Path) -> Mapping[str, Any]:
    """Verify the frozen cache and write two deterministic NPZ files plus a manifest."""

    if output_dir.exists():
        _fail(f"output directory already exists; refusing overwrite: {output_dir}")
    if not output_dir.parent.is_dir():
        _fail(f"output directory parent does not exist: {output_dir.parent}")
    contract = _read_contract(contract_path)
    lock = _verify_format_lock(contract["format_lock"])
    validation = _verify_validation(contract["validation"])
    evidence_boundary = _verify_evidence_boundary(contract["evidence_boundary"])
    asset_path, stl_path, asset_meshes, runtime_file_spec = _verify_sources_and_binding(
        contract, validation
    )

    ddc = _exact_keys(
        contract["ddc"],
        {
            "cacheindex",
            "smallcachedata",
            "largecachedata",
            "index_record_size",
            "small_index_magic_hex",
            "large_index_magic_hex",
            "data_record_magic_hex",
            "valid_small_index_records",
            "valid_large_index_records",
        },
        "ddc",
    )
    if (
        ddc["index_record_size"] != INDEX_RECORD_SIZE
        or ddc["small_index_magic_hex"] != SMALL_INDEX_MAGIC.hex()
        or ddc["large_index_magic_hex"] != LARGE_INDEX_MAGIC.hex()
        or ddc["data_record_magic_hex"] != DATA_RECORD_MAGIC.hex()
    ):
        _fail("DDC layout constants differ from the frozen parser")
    index_path = _verified_file(ddc["cacheindex"], "ddc.cacheindex")
    small_path = _verified_file(ddc["smallcachedata"], "ddc.smallcachedata")
    large_path = _verified_file(ddc["largecachedata"], "ddc.largecachedata")
    index = _build_index(
        index_path,
        _required_int(ddc["valid_small_index_records"], "ddc.valid_small_index_records"),
        _required_int(ddc["valid_large_index_records"], "ddc.valid_large_index_records"),
    )
    source_mtime = stl_path.stat().st_mtime_ns
    asset_mtime = asset_path.stat().st_mtime_ns
    ddc_mtimes = {path.stat().st_mtime_ns for path in (index_path, small_path, large_path)}
    runtime_path = Path(runtime_file_spec["path"])
    if len(ddc_mtimes) != 1:
        _fail("frozen DDC files do not share the same mtime")
    ddc_mtime = next(iter(ddc_mtimes))
    if not (source_mtime < asset_mtime < ddc_mtime < runtime_path.stat().st_mtime_ns):
        _fail("source/asset/DDC/runtime evidence mtimes are not strictly ordered")

    aggregate_rows = contract["aggregates"]
    if not isinstance(aggregate_rows, list) or len(aggregate_rows) != 2:
        _fail("contract.aggregates must contain exactly the two frozen groups")
    prepared_outputs: list[tuple[str, Mapping[str, np.ndarray], Mapping[str, Any]]] = []
    group_manifests: list[Mapping[str, Any]] = []
    output_names: set[str] = set()
    seen_group_ids: set[str] = set()
    seen_aggregate_keys: set[bytes] = set()
    seen_asset_mesh_ids: set[str] = set()
    seen_terminal_links: set[str] = set()
    seen_child_keys: set[bytes] = set()
    with small_path.open("rb") as small_stream, large_path.open("rb") as large_stream:
        with mmap.mmap(small_stream.fileno(), 0, access=mmap.ACCESS_READ) as small_data:
            with mmap.mmap(large_stream.fileno(), 0, access=mmap.ACCESS_READ) as large_data:
                data_by_storage = {"small": small_data, "large": large_data}
                for group_index, raw_group in enumerate(aggregate_rows):
                    group = _exact_keys(
                        raw_group,
                        {
                            "group_id",
                            "terminal_links",
                            "asset_collision_mesh_ids",
                            "output_name",
                            "aggregate",
                            "expected",
                        },
                        f"aggregates[{group_index}]",
                    )
                    group_id = _required_string(group["group_id"], f"aggregates[{group_index}].group_id")
                    if group_id not in _EXPECTED_GROUP_IDS or group_id in seen_group_ids:
                        _fail(f"unexpected or duplicate aggregate group id: {group_id}")
                    seen_group_ids.add(group_id)
                    terminal_links = group["terminal_links"]
                    mesh_ids = group["asset_collision_mesh_ids"]
                    if not isinstance(terminal_links, list) or not terminal_links:
                        _fail(f"{group_id}.terminal_links must be non-empty")
                    if not isinstance(mesh_ids, list) or not mesh_ids:
                        _fail(f"{group_id}.asset_collision_mesh_ids must be non-empty")
                    terminal_strings = [
                        _required_string(item, f"{group_id}.terminal_links") for item in terminal_links
                    ]
                    mesh_strings = [
                        _required_string(item, f"{group_id}.asset_collision_mesh_ids") for item in mesh_ids
                    ]
                    if len(set(terminal_strings)) != len(terminal_strings):
                        _fail(f"{group_id} contains duplicate terminal links")
                    if len(set(mesh_strings)) != len(mesh_strings):
                        _fail(f"{group_id} contains duplicate asset collision mesh ids")
                    if seen_terminal_links.intersection(terminal_strings):
                        _fail(f"terminal link occurs in more than one group: {group_id}")
                    if seen_asset_mesh_ids.intersection(mesh_strings):
                        _fail(f"asset collision mesh occurs in more than one group: {group_id}")
                    seen_terminal_links.update(terminal_strings)
                    seen_asset_mesh_ids.update(mesh_strings)
                    if any(item not in asset_meshes for item in mesh_strings):
                        _fail(f"{group_id} names an unknown asset collision mesh")
                    if sorted(asset_meshes[item]["terminal_link"] for item in mesh_strings) != sorted(terminal_strings):
                        _fail(f"{group_id} terminal links do not match asset mesh binding")
                    output_name = _required_string(group["output_name"], f"{group_id}.output_name")
                    expected_binding = _EXPECTED_GROUP_BINDINGS[group_id]
                    if (
                        tuple(sorted(mesh_strings))
                        != tuple(sorted(expected_binding["asset_collision_mesh_ids"]))
                        or tuple(sorted(terminal_strings))
                        != tuple(sorted(expected_binding["terminal_links"]))
                        or output_name != expected_binding["output_name"]
                    ):
                        _fail(f"frozen mesh/terminal/output binding differs for {group_id}")
                    if Path(output_name).name != output_name or not output_name.endswith(".npz"):
                        _fail(f"unsafe NPZ output name for {group_id}: {output_name}")
                    if output_name in output_names:
                        _fail(f"duplicate NPZ output name: {output_name}")
                    output_names.add(output_name)

                    aggregate_spec = _exact_keys(
                        group["aggregate"],
                        {"key_hex", "storage", "record_offset", "payload_size", "payload_sha256"},
                        f"{group_id}.aggregate",
                    )
                    aggregate_key_hex = _required_string(
                        aggregate_spec["key_hex"], f"{group_id}.aggregate.key_hex"
                    )
                    if aggregate_key_hex != expected_binding["aggregate_key_hex"]:
                        _fail(f"frozen aggregate key differs for {group_id}")
                    try:
                        aggregate_key = bytes.fromhex(aggregate_key_hex)
                    except ValueError as error:
                        raise ExtractionError(f"invalid aggregate key for {group_id}") from error
                    if len(aggregate_key) != DDC_KEY_SIZE or aggregate_key not in index:
                        _fail(f"aggregate key is absent from DDC index for {group_id}")
                    if aggregate_key in seen_aggregate_keys:
                        _fail(f"aggregate key is reused by multiple groups: {aggregate_key.hex()}")
                    seen_aggregate_keys.add(aggregate_key)
                    aggregate_record = index[aggregate_key]
                    if aggregate_record.storage != aggregate_spec["storage"]:
                        _fail(f"aggregate storage mismatch for {group_id}")
                    if aggregate_record.data_offset != aggregate_spec["record_offset"]:
                        _fail(f"aggregate offset mismatch for {group_id}")
                    if aggregate_record.payload_size != aggregate_spec["payload_size"]:
                        _fail(f"aggregate payload size mismatch for {group_id}")
                    aggregate_payload = _record_payload(
                        aggregate_record, data_by_storage[aggregate_record.storage]
                    )
                    aggregate_payload_hash = hashlib.sha256(aggregate_payload).hexdigest()
                    if aggregate_payload_hash != _required_sha256(
                        aggregate_spec["payload_sha256"], f"{group_id}.aggregate.payload_sha256"
                    ):
                        _fail(f"aggregate payload hash mismatch for {group_id}")

                    child_count = int(lock["aggregate_child_count"])
                    child_offset = int(lock["aggregate_child_key_offset"])
                    child_stride = int(lock["aggregate_child_key_stride"])
                    if child_offset + child_count * child_stride > len(aggregate_payload):
                        _fail(f"aggregate child-key table is truncated for {group_id}")
                    child_keys = [
                        aggregate_payload[
                            child_offset + child_index * child_stride :
                            child_offset + child_index * child_stride + DDC_KEY_SIZE
                        ]
                        for child_index in range(child_count)
                    ]
                    if len(set(child_keys)) != child_count or any(key not in index for key in child_keys):
                        _fail(f"aggregate child keys are duplicate or absent for {group_id}")
                    overlap = seen_child_keys.intersection(child_keys)
                    if overlap:
                        _fail(
                            f"cooked child key is reused across groups: "
                            f"{next(iter(overlap)).hex()}"
                        )
                    seen_child_keys.update(child_keys)
                    hulls: list[CookedHull] = []
                    ordered_payload_hasher = hashlib.sha256()
                    for child_key in child_keys:
                        record = index[child_key]
                        payload = _record_payload(record, data_by_storage[record.storage])
                        ordered_payload_hasher.update(payload)
                        payload_hash = hashlib.sha256(payload).hexdigest()
                        hulls.append(
                            _parse_cvxm(payload, record, payload_hash, lock, validation)
                        )
                    expected = _aggregate_expected(group["expected"], f"{group_id}.expected")
                    storage_counts = Counter(hull.storage for hull in hulls)
                    vertices_count = sum(hull.vertices.shape[0] for hull in hulls)
                    polygons_count = sum(len(hull.polygon_vertex_indices) for hull in hulls)
                    triangles_count = sum(hull.triangle_faces.shape[0] for hull in hulls)
                    bounds = np.stack(
                        (
                            np.min(np.stack([hull.bounds_m[0] for hull in hulls]), axis=0),
                            np.max(np.stack([hull.bounds_m[1] for hull in hulls]), axis=0),
                        )
                    )
                    actual_counts = {
                        "hull_count": len(hulls),
                        "small_child_count": storage_counts["small"],
                        "large_child_count": storage_counts["large"],
                        "vertex_count": vertices_count,
                        "polygon_count": polygons_count,
                        "triangle_count": triangles_count,
                    }
                    for key, actual in actual_counts.items():
                        if actual != expected[key]:
                            _fail(f"{group_id} {key} mismatch: {actual} != {expected[key]}")
                    expected_bounds = _exact_float_matrix(
                        expected["bounds_m"], (2, 3), f"{group_id}.expected.bounds_m"
                    )
                    if not np.array_equal(bounds, expected_bounds):
                        _fail(f"{group_id} union bounds mismatch: {bounds.tolist()}")
                    ordered_payload_hash = ordered_payload_hasher.hexdigest()
                    if ordered_payload_hash != expected["ordered_child_payload_sha256"]:
                        _fail(f"ordered child-payload hash mismatch for {group_id}")

                    arrays = _group_arrays(hulls)
                    diagnostic_names = sorted(hulls[0].diagnostics)
                    diagnostic_summary = {
                        f"maximum_{name}": max(float(hull.diagnostics[name]) for hull in hulls)
                        for name in diagnostic_names
                    }
                    diagnostic_summary.update(
                        {
                            f"minimum_{name}": min(float(hull.diagnostics[name]) for hull in hulls)
                            for name in diagnostic_names
                            if name.startswith("min_")
                        }
                    )
                    group_manifest = {
                        "group_id": group_id,
                        "terminal_links": terminal_strings,
                        "asset_collision_mesh_ids": mesh_strings,
                        "aggregate": {
                            "key_hex": aggregate_key.hex(),
                            "storage": aggregate_record.storage,
                            "record_offset": aggregate_record.data_offset,
                            "payload_size": aggregate_record.payload_size,
                            "payload_sha256": aggregate_payload_hash,
                        },
                        "child_records": [
                            {
                                "key_hex": hull.key.hex(),
                                "storage": hull.storage,
                                "record_offset": hull.data_offset,
                                "payload_size": hull.payload_size,
                                "payload_sha256": hull.payload_sha256,
                            }
                            for hull in hulls
                        ],
                        "ordered_child_payload_sha256": ordered_payload_hash,
                        "counts": actual_counts,
                        "bounds_m": bounds.tolist(),
                        "validation_diagnostics": diagnostic_summary,
                    }
                    prepared_outputs.append((output_name, arrays, group_manifest))
                    group_manifests.append(group_manifest)

    expected_mesh_ids = set(asset_meshes)
    expected_terminal_links = {
        str(row["terminal_link"]) for row in asset_meshes.values()
    }
    if seen_group_ids != _EXPECTED_GROUP_IDS:
        _fail(f"aggregate groups do not exactly cover {_EXPECTED_GROUP_IDS}")
    if seen_asset_mesh_ids != expected_mesh_ids:
        _fail("aggregate groups do not cover each asset collision mesh exactly once")
    if seen_terminal_links != expected_terminal_links:
        _fail("aggregate groups do not cover each terminal link exactly once")

    output_dir.mkdir(mode=0o755)
    output_rows: list[Mapping[str, Any]] = []
    for output_name, arrays, group_manifest in prepared_outputs:
        output_path = output_dir / output_name
        _write_deterministic_npz(output_path, arrays)
        output_rows.append(
            {
                "group_id": group_manifest["group_id"],
                "path": output_name,
                "size_bytes": output_path.stat().st_size,
                "sha256": _sha256_file(output_path),
                "arrays": {
                    name: {"dtype": str(array.dtype), "shape": list(array.shape)}
                    for name, array in sorted(arrays.items())
                },
            }
        )
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "simulation_only": True,
        "contract": {
            "path": str(contract_path.resolve()),
            "sha256": _sha256_file(contract_path),
        },
        "extractor": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256_file(Path(__file__)),
        },
        "verified_sources": {
            "robot_asset": str(asset_path),
            "source_stl": str(stl_path),
            "asset_collision_meshes": asset_meshes,
        },
        "groups": group_manifests,
        "outputs": output_rows,
        "evidence_boundary": evidence_boundary,
    }
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        _fail(f"refusing to overwrite manifest: {manifest_path}")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    arguments = _parse_arguments()
    try:
        manifest = extract(arguments.contract.resolve(), arguments.output_dir.resolve())
    except ExtractionError as error:
        raise SystemExit(f"FAIL_CLOSED: {error}") from error
    summary = {
        "output_dir": str(arguments.output_dir.resolve()),
        "outputs": manifest["outputs"],
        "groups": [
            {
                "group_id": group["group_id"],
                "counts": group["counts"],
                "bounds_m": group["bounds_m"],
            }
            for group in manifest["groups"]
        ],
        "evidence_class": "OFFLINE_COOKED_CACHE_RECONSTRUCTION",
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
