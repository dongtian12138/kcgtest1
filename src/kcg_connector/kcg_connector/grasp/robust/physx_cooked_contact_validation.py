"""Offline validation of saved TE contacts against frozen cooked fingertip hulls.

This is deliberately target-specific.  It never imports or starts pxr, Kit,
Isaac, or PhysX.  Historical positive-impulse contact points are transformed
with the same frozen robot-model FK used by ``evaluate_run.py`` and compared
with the exact source surfaces and the frozen, hash-bound cooked hull snapshot.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.metadata
import inspect
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any, Mapping, Sequence

import ijson
import manifold3d
import numpy as np
import scipy
from scipy.spatial import ConvexHull
import trimesh
from trimesh.proximity import ProximityQuery

from kcg_connector.grasp.carts_v2.models import load_v2_inputs
from kcg_connector.grasp.carts_v2.task_grip_surface import task_noncontact_triangles


SCHEMA_VERSION = "kcg_physx_cooked_contact_validation_v1"
RAW_SCHEMA_VERSION = 1
LINK_NAMES = ("f1Link3", "f2Link2", "f3Link3")
LINK_TO_ID = {name: index for index, name in enumerate(LINK_NAMES)}
MANIFEST_NAME = "manifest.json"
RAW_NAME = "contact_points.npz"
SUMMARY_NAME = "summary.json"
STRICT_INTERIOR_TOLERANCE_M = 1.0e-7
# Manifold is intentionally fed float32 millimetre coordinates.  At the
# approximately 60 mm fingertip scale one float32 ULP is about 7 nm, and the
# independently audited equivalent-union discrepancy is 26.9 nm.  This is a
# numerical lineage tolerance, not a contact-distance acceptance threshold.
EXPOSED_FACE_MAP_TOLERANCE_M = 1.0e-7
EXPECTED_COOKED_FILES = {
    "f1_f3_shared_cooked_convex": "f1_f3_cooked_convex.npz",
    "f2_cooked_convex": "f2_cooked_convex.npz",
}


class ValidationError(RuntimeError):
    """Raised when evidence binding or offline geometry fails closed."""


def _fail(message: str) -> None:
    raise ValidationError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        _fail(f"{label} is not a file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        _fail(f"{label} must contain a JSON object")
    return value


def _same(left: object, right: object, label: str) -> None:
    if left != right:
        _fail(f"{label} mismatch: {left!r} != {right!r}")


def _finite_vector(value: object, size: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        _fail(f"{label} must be a finite vector of length {size}")
    return result


def _trace_scalar(path: Path, prefix: str) -> object:
    with path.open("rb") as stream:
        iterator = ijson.items(stream, prefix, use_float=True)
        try:
            return next(iterator)
        except StopIteration:
            _fail(f"trace lacks {prefix}")


def _trace_tail_object(path: Path, key: str) -> Mapping[str, Any]:
    marker = f'\n  "{key}": '.encode("utf-8")
    size = path.stat().st_size
    window = min(size, 4 * 1024 * 1024)
    with path.open("rb") as stream:
        stream.seek(size - window)
        data = stream.read()
    offsets = []
    start = 0
    while True:
        index = data.find(marker, start)
        if index < 0:
            break
        offsets.append(index)
        start = index + len(marker)
    if len(offsets) != 1:
        _fail(f"trace tail must contain exactly one top-level {key!r} object")
    text = data[offsets[0] + len(marker):].decode("utf-8")
    try:
        value, _ = json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError as error:
        raise ValidationError(f"cannot decode trace tail object {key}: {error}") from error
    if not isinstance(value, dict):
        _fail(f"trace {key} is not an object")
    return value


def _below(path: str, root: str) -> bool:
    return path == root or path.startswith(root + "/")


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _module_evidence(module: object, distribution: str) -> Mapping[str, object]:
    module_path = Path(inspect.getfile(module)).resolve()
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        version = getattr(module, "__version__", "UNKNOWN")
    return {
        "version": str(version),
        "module_path": str(module_path),
        "module_file_sha256": _sha256_file(module_path),
    }


@dataclass
class CookedGeometry:
    group_id: str
    npz_path: Path
    vertices_m: np.ndarray
    triangle_faces: np.ndarray
    triangle_hull_index: np.ndarray
    triangle_polygon_index: np.ndarray
    polygon_planes: np.ndarray
    hull_vertex_offsets: np.ndarray
    all_mesh: trimesh.Trimesh
    all_query: ProximityQuery
    union_mesh: trimesh.Trimesh
    union_query: ProximityQuery
    union_source_triangle: np.ndarray
    hull_halfspaces: tuple[tuple[np.ndarray, np.ndarray], ...]
    union_metadata: Mapping[str, object]


def _surface_query(triangles: np.ndarray) -> tuple[trimesh.Trimesh, ProximityQuery]:
    triangles = np.asarray(triangles, dtype=np.float64)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or not len(triangles):
        _fail("surface triangles must have shape (N,3,3)")
    vertices = triangles.reshape(-1, 3)
    faces = np.arange(len(vertices), dtype=np.int64).reshape(-1, 3)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return mesh, ProximityQuery(mesh)


def _load_cooked_geometry(
    group_id: str,
    path: Path,
    output_spec: Mapping[str, Any],
) -> CookedGeometry:
    _same(path.name, output_spec.get("path"), f"{group_id} output filename")
    _same(path.stat().st_size, output_spec.get("size_bytes"), f"{group_id} size")
    _same(_sha256_file(path), output_spec.get("sha256"), f"{group_id} SHA-256")
    arrays_spec = output_spec.get("arrays")
    if not isinstance(arrays_spec, dict):
        _fail(f"{group_id} manifest arrays missing")
    with np.load(path, allow_pickle=False) as loaded:
        if set(loaded.files) != set(arrays_spec):
            _fail(f"{group_id} NPZ array names differ from manifest")
        arrays = {}
        for name, specification in arrays_spec.items():
            array = np.asarray(loaded[name])
            _same(list(array.shape), specification.get("shape"), f"{group_id}.{name}.shape")
            _same(str(array.dtype), specification.get("dtype"), f"{group_id}.{name}.dtype")
            arrays[name] = np.array(array, copy=True)
    vertices = np.asarray(arrays["vertices"], dtype=np.float64)
    faces = np.asarray(arrays["triangle_faces"], dtype=np.int64)
    triangle_hulls = np.asarray(arrays["triangle_hull_index"], dtype=np.int64)
    vertex_offsets = np.asarray(arrays["hull_vertex_offsets"], dtype=np.int64)
    polygon_offsets = np.asarray(arrays["polygon_offsets"], dtype=np.int64)
    planes = np.asarray(arrays["polygon_planes"], dtype=np.float64)
    if vertex_offsets.shape != (33,) or np.any(np.diff(vertex_offsets) <= 0):
        _fail(f"{group_id} hull vertex offsets invalid")
    if faces.shape[1:] != (3,) or triangle_hulls.shape != (len(faces),):
        _fail(f"{group_id} triangle arrays invalid")
    polygon_sizes = np.diff(polygon_offsets)
    if np.any(polygon_sizes < 3):
        _fail(f"{group_id} polygon sizes invalid")
    triangle_polygons = np.repeat(np.arange(len(planes), dtype=np.int64), polygon_sizes - 2)
    if triangle_polygons.shape != (len(faces),):
        _fail(f"{group_id} fan triangle/polygon lineage invalid")
    all_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    hull_manifolds = []
    halfspaces = []
    volume_sum_m3 = 0.0
    for hull in range(32):
        triangle_indices = np.flatnonzero(triangle_hulls == hull)
        start, end = int(vertex_offsets[hull]), int(vertex_offsets[hull + 1])
        local_vertices = vertices[start:end]
        local_faces = faces[triangle_indices] - start
        hull_mesh = trimesh.Trimesh(
            vertices=local_vertices, faces=local_faces, process=False
        )
        if not hull_mesh.is_watertight or hull_mesh.volume <= 0.0:
            _fail(f"{group_id} hull {hull} is not a positive watertight mesh")
        volume_sum_m3 += float(hull_mesh.volume)
        convex = ConvexHull(local_vertices)
        equations = np.asarray(convex.equations, dtype=np.float64)
        lengths = np.linalg.norm(equations[:, :3], axis=1)
        halfspaces.append(
            (equations[:, :3] / lengths[:, None], equations[:, 3] / lengths)
        )
        manifold_mesh = manifold3d.Mesh(
            vert_properties=np.ascontiguousarray(
                local_vertices * 1000.0, dtype=np.float32
            ),
            tri_verts=np.ascontiguousarray(local_faces, dtype=np.uint32),
            face_id=np.ascontiguousarray(triangle_indices, dtype=np.uint32),
            tolerance=0.0,
        )
        part = manifold3d.Manifold(manifold_mesh)
        if part.status() != manifold3d.Error.NoError:
            _fail(f"{group_id} hull {hull} manifold status: {part.status()}")
        hull_manifolds.append(part)
    union = manifold3d.Manifold.batch_boolean(hull_manifolds, manifold3d.OpType.Add)
    if union.status() != manifold3d.Error.NoError:
        _fail(f"{group_id} union status: {union.status()}")
    union_output = union.to_mesh()
    union_vertices = np.asarray(union_output.vert_properties, dtype=np.float64)[:, :3] / 1000.0
    union_faces = np.asarray(union_output.tri_verts, dtype=np.int64)
    union_source = np.asarray(union_output.face_id, dtype=np.int64)
    if union_source.shape != (len(union_faces),) or np.any(union_source < 0) or np.any(
        union_source >= len(faces)
    ):
        _fail(f"{group_id} union source face lineage invalid")
    union_mesh = trimesh.Trimesh(
        vertices=union_vertices, faces=union_faces, process=False
    )
    if not union_mesh.is_watertight or union_mesh.volume <= 0.0:
        _fail(f"{group_id} exposed union is not a positive watertight mesh")
    return CookedGeometry(
        group_id=group_id,
        npz_path=path,
        vertices_m=vertices,
        triangle_faces=faces,
        triangle_hull_index=triangle_hulls,
        triangle_polygon_index=triangle_polygons,
        polygon_planes=planes,
        hull_vertex_offsets=vertex_offsets,
        all_mesh=all_mesh,
        all_query=ProximityQuery(all_mesh),
        union_mesh=union_mesh,
        union_query=ProximityQuery(union_mesh),
        union_source_triangle=union_source,
        hull_halfspaces=tuple(halfspaces),
        union_metadata={
            "input_hull_count": 32,
            "input_triangle_count": int(len(faces)),
            "union_vertex_count": int(len(union_vertices)),
            "union_triangle_count": int(len(union_faces)),
            "source_triangle_with_union_contribution_count": int(len(np.unique(union_source))),
            "watertight": True,
            "component_count": int(len(union_mesh.split(only_watertight=False))),
            "sum_hull_volume_m3": volume_sum_m3,
            "union_volume_m3": float(union_mesh.volume),
            "union_surface_area_m2": float(union_mesh.area),
            "construction": "MANIFOLD3D_BATCH_UNION_MM_SCALE_WITH_SOURCE_FACE_ID",
        },
    )


def _classify_internal_projection(
    points: np.ndarray,
    owner_hulls: np.ndarray,
    halfspaces: Sequence[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = len(points)
    strict = np.zeros(count, dtype=np.bool_)
    boundary = np.zeros(count, dtype=np.bool_)
    buried_by = np.full(count, -1, dtype=np.int16)
    nearest_other_max = np.full(count, np.inf, dtype=np.float64)
    for hull, (normals, offsets) in enumerate(halfspaces):
        indices = np.flatnonzero(owner_hulls != hull)
        if not len(indices):
            continue
        maximum = np.max(points[indices] @ normals.T + offsets[None, :], axis=1)
        better = maximum < nearest_other_max[indices]
        nearest_other_max[indices[better]] = maximum[better]
        deeply_inside = maximum <= -STRICT_INTERIOR_TOLERANCE_M
        for local in np.flatnonzero(deeply_inside):
            index = int(indices[local])
            if not strict[index] or maximum[local] < nearest_other_max[index]:
                buried_by[index] = hull
            strict[index] = True
        boundary[indices] |= maximum <= STRICT_INTERIOR_TOLERANCE_M
    boundary &= ~strict
    return strict, boundary, buried_by, nearest_other_max


def _validate_bindings(arguments: argparse.Namespace) -> Mapping[str, Any]:
    trace = arguments.trace.resolve()
    evaluation_path = arguments.evaluation.resolve()
    cooked_dir = arguments.cooked_dir.resolve()
    if not trace.is_file() or not evaluation_path.is_file() or not cooked_dir.is_dir():
        _fail("trace/evaluation must be files and cooked-dir must be a directory")
    evaluation = _read_json(evaluation_path, "evaluation")
    manifest_path = cooked_dir / MANIFEST_NAME
    manifest = _read_json(manifest_path, "cooked manifest")
    _same(manifest.get("schema_version"), "kcg_nailfree_physx_cooked_extract_v1", "manifest schema")
    _same(manifest.get("simulation_only"), True, "manifest simulation-only")

    trace_head = {
        key: _trace_scalar(trace, key)
        for key in (
            "schema_version", "object_id", "candidate_id", "mode",
            "config_sha256", "physics_dt_s",
        )
    }
    trace_binding = _trace_tail_object(trace, "evidence_binding")
    audit_roots = _trace_tail_object(trace, "audit_roots")
    runtime = _trace_tail_object(trace, "runtime")
    for key in ("object_id", "candidate_id", "mode"):
        _same(trace_head[key], evaluation.get(key), f"trace/evaluation {key}")
    _same(trace_head["mode"], "grasp-lift", "trace mode")
    _same(trace_head["schema_version"], "carts_grasp_v2_dynamic_trace_v1", "trace schema")

    config_path = (
        arguments.config.resolve()
        if arguments.config is not None
        else Path(str(runtime.get("config_path", ""))).resolve()
    )
    _same(str(config_path), str(Path(str(runtime.get("config_path", ""))).resolve()), "config path")
    config_sha = _sha256_file(config_path)
    for label, value in (
        ("trace.config_sha256", trace_head["config_sha256"]),
        ("trace.runtime.config_sha256", runtime.get("config_sha256")),
        ("trace.evidence_binding.config_sha256", trace_binding.get("config_sha256")),
        ("evaluation.evidence_binding.config_sha256", evaluation.get("evidence_binding", {}).get("config_sha256")),
    ):
        _same(config_sha, value, label)

    manifest_contract = manifest.get("contract", {})
    contract_path = (
        arguments.contract.resolve()
        if arguments.contract is not None
        else Path(str(manifest_contract.get("path", ""))).resolve()
    )
    _same(str(contract_path), str(Path(str(manifest_contract.get("path", ""))).resolve()), "contract path")
    _same(_sha256_file(contract_path), manifest_contract.get("sha256"), "contract SHA-256")
    contract = _read_json(contract_path, "cooked contract")
    _same(contract.get("schema_version"), "kcg_nailfree_physx_cooked_v2", "contract schema")
    _same(contract.get("simulation_only"), True, "contract simulation-only")

    compared_binding_keys = (
        "config_sha256", "registered_grasp_sha256", "control_plan_sha256",
        "runtime_resources_sha256", "capacity_audit_sha256",
        "scene_evidence_sha256", "robot_asset_sha256", "object_asset_sha256",
    )
    evaluation_binding = evaluation.get("evidence_binding", {})
    for key in compared_binding_keys:
        _same(trace_binding.get(key), evaluation_binding.get(key), f"trace/evaluation binding {key}")
    evaluator_path = _repository_root() / "src/kcg_connector/isaac/carts_v2/evaluate_run.py"
    evaluator_sha = _sha256_file(evaluator_path)
    _same(evaluator_sha, runtime.get("source_sha256", {}).get("evaluate_run.py"), "trace evaluator source")
    _same(evaluator_sha, evaluation_binding.get("evaluator_source_sha256"), "evaluation evaluator source")

    pad_identity = evaluation.get("pad_surface_identity_evidence", {})
    _same(pad_identity.get("asset_binding_matches"), True, "evaluation robot asset binding")
    robot_path = Path(str(pad_identity.get("robot_asset_path", ""))).resolve()
    robot_sha = _sha256_file(robot_path)
    for label, value in (
        ("trace robot SHA", runtime.get("robot_asset_sha256")),
        ("evaluation robot SHA", evaluation_binding.get("robot_asset_sha256")),
        ("evaluation pad robot SHA", pad_identity.get("robot_asset_sha256")),
        ("evaluation trace-bound robot SHA", pad_identity.get("trace_bound_robot_asset_sha256")),
        ("contract robot SHA", contract.get("source_files", {}).get("robot_asset", {}).get("sha256")),
    ):
        _same(robot_sha, value, label)
    _same(str(robot_path), contract.get("source_files", {}).get("robot_asset", {}).get("path"), "contract robot path")
    verified_sources = manifest.get("verified_sources", {})
    _same(str(robot_path), verified_sources.get("robot_asset"), "manifest robot path")

    source_spec = contract.get("source_files", {}).get("source_stl", {})
    source_path = Path(str(source_spec.get("path", ""))).resolve()
    _same(_sha256_file(source_path), source_spec.get("sha256"), "source STL SHA-256")
    _same(str(source_path), verified_sources.get("source_stl"), "manifest source STL path")

    outputs = manifest.get("outputs")
    groups = manifest.get("groups")
    if not isinstance(outputs, list) or not isinstance(groups, list):
        _fail("manifest outputs/groups must be arrays")
    output_by_group = {str(row.get("group_id")): row for row in outputs}
    group_by_id = {str(row.get("group_id")): row for row in groups}
    _same(set(output_by_group), set(EXPECTED_COOKED_FILES), "manifest output groups")
    _same(set(group_by_id), set(EXPECTED_COOKED_FILES), "manifest geometry groups")
    expected_links = {
        "f1_f3_shared_cooked_convex": ["f1Link3", "f3Link3"],
        "f2_cooked_convex": ["f2Link2"],
    }
    for group_id, filename in EXPECTED_COOKED_FILES.items():
        _same(output_by_group[group_id].get("path"), filename, f"{group_id} output")
        _same(group_by_id[group_id].get("terminal_links"), expected_links[group_id], f"{group_id} terminal links")

    projection_rows = evaluation.get("pad_surface_identity_evidence", {}).get(
        "contact_point_projection", []
    )
    expected_positive_counts = {
        str(row.get("terminal_link")): int(row.get("positive_contact_point_count"))
        for row in projection_rows
    }
    _same(set(expected_positive_counts), set(LINK_NAMES), "evaluation positive-contact links")
    return {
        "trace_path": trace,
        "evaluation_path": evaluation_path,
        "cooked_dir": cooked_dir,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "contract_path": contract_path,
        "contract": contract,
        "config_path": config_path,
        "evaluation": evaluation,
        "trace_head": trace_head,
        "trace_binding": trace_binding,
        "runtime": runtime,
        "audit_roots": audit_roots,
        "robot_path": robot_path,
        "source_path": source_path,
        "output_by_group": output_by_group,
        "expected_positive_counts": expected_positive_counts,
        "evaluator_path": evaluator_path,
    }


def _collect_positive_contacts(
    trace_path: Path,
    object_root: str,
    robot_model: object,
) -> Mapping[str, np.ndarray]:
    rows: list[tuple[int, int, float, str, np.ndarray, np.ndarray, float, float]] = []
    with trace_path.open("rb") as stream:
        for sample in ijson.items(stream, "samples.item", use_float=True):
            pending: dict[str, list[Mapping[str, object]]] = {
                name: [] for name in LINK_NAMES
            }
            contacts = sample.get("contacts")
            if not isinstance(contacts, dict):
                _fail("trace sample contacts must be an object")
            for header in contacts.get("tensor_headers", ()):
                paths = tuple(map(str, header.get("paths", ())))
                if not any(_below(path, object_root) for path in paths):
                    continue
                links = [
                    name for name in LINK_NAMES
                    if any(f"/{name}" in path for path in paths)
                ]
                if len(links) != 1:
                    continue
                for contact in header.get("contacts", ()):
                    impulse = float(contact.get("normal_impulse_n_s", 0.0))
                    if impulse > 0.0:
                        pending[links[0]].append(contact)
            if not any(pending.values()):
                continue
            active_positions = _finite_vector(
                sample.get("active_positions_rad"), 11, "active_positions_rad"
            )
            transforms = robot_model.forward_kinematics(
                active_positions, enforce_limits=False
            )
            step = int(sample.get("step"))
            time_s = float(sample.get("simulation_time_s"))
            phase = str(sample.get("phase"))
            if step < 0 or not math.isfinite(time_s) or not phase:
                _fail("trace sample step/time/phase invalid")
            for link_name, link_contacts in pending.items():
                if not link_contacts:
                    continue
                transform = np.asarray(transforms[link_name], dtype=np.float64)
                if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
                    _fail(f"FK transform invalid for {link_name}")
                rotation = transform[:3, :3]
                translation = transform[:3, 3]
                for contact in link_contacts:
                    position_world = _finite_vector(
                        contact.get("position_m"), 3, "contact.position_m"
                    )
                    normal_world = _finite_vector(
                        contact.get("normal"), 3, "contact.normal"
                    )
                    normal_impulse = float(contact.get("normal_impulse_n_s"))
                    separation = float(contact.get("separation_m"))
                    if not math.isfinite(normal_impulse) or not math.isfinite(separation):
                        _fail("contact impulse/separation must be finite")
                    rows.append((
                        LINK_TO_ID[link_name], step, time_s, phase,
                        (position_world - translation) @ rotation,
                        normal_world @ rotation, normal_impulse, separation,
                    ))
    if not rows:
        _fail("trace has no positive terminal-link/object contacts")
    phases = sorted({row[3] for row in rows})
    phase_to_id = {phase: index for index, phase in enumerate(phases)}
    return {
        "link_id": np.asarray([row[0] for row in rows], dtype=np.int8),
        "step": np.asarray([row[1] for row in rows], dtype=np.int64),
        "simulation_time_s": np.asarray([row[2] for row in rows], dtype=np.float64),
        "phase_id": np.asarray([phase_to_id[row[3]] for row in rows], dtype=np.int16),
        "position_link_local_m": np.asarray([row[4] for row in rows], dtype=np.float64),
        "normal_link_local": np.asarray([row[5] for row in rows], dtype=np.float64),
        "normal_impulse_n_s": np.asarray([row[6] for row in rows], dtype=np.float64),
        "separation_m": np.asarray([row[7] for row in rows], dtype=np.float64),
        "phase_vocabulary": phases,
    }


def _analyze_contacts(
    base: Mapping[str, np.ndarray],
    inputs: object,
    cooked_by_link: Mapping[str, CookedGeometry],
) -> Mapping[str, np.ndarray]:
    count = len(base["link_id"])
    result: dict[str, np.ndarray] = {
        key: np.array(value, copy=True)
        for key, value in base.items()
        if key != "phase_vocabulary"
    }
    vector_fields = (
        "cooked_all_closest_position_link_local_m",
        "cooked_exposed_closest_position_link_local_m",
        "cooked_exposed_triangle_normal_link_local",
        "cooked_exposed_stored_plane_normal_link_local",
    )
    scalar_float_fields = (
        "source_full_distance_m", "pad_distance_m", "nonpad_distance_m",
        "cooked_all_distance_m", "cooked_exposed_distance_m",
        "cooked_exposed_minus_all_distance_m",
        "cooked_exposed_face_map_residual_m",
        "cooked_exposed_triangle_normal_abs_dot",
        "cooked_exposed_triangle_normal_angle_rad",
        "cooked_exposed_stored_plane_normal_abs_dot",
        "cooked_exposed_stored_plane_normal_angle_rad",
        "nearest_other_hull_max_halfspace_m",
    )
    integer_fields = (
        "cooked_all_nearest_hull", "cooked_all_nearest_triangle",
        "cooked_all_nearest_polygon", "cooked_exposed_nearest_hull",
        "cooked_exposed_nearest_triangle", "cooked_exposed_nearest_polygon",
        "cooked_exposed_union_triangle", "buried_by_hull",
    )
    for field in vector_fields:
        result[field] = np.full((count, 3), np.nan, dtype=np.float64)
    for field in scalar_float_fields:
        result[field] = np.full(count, np.nan, dtype=np.float64)
    for field in integer_fields:
        result[field] = np.full(count, -1, dtype=np.int32)
    result["all_nearest_strictly_buried"] = np.zeros(count, dtype=np.bool_)
    result["all_nearest_boundary_ambiguous"] = np.zeros(count, dtype=np.bool_)
    result["all_exposed_difference_gt_tolerance"] = np.zeros(count, dtype=np.bool_)

    noncontact = task_noncontact_triangles(
        inputs.hand_collision_triangles_by_link, inputs.task_grip_surfaces
    )
    surfaces = {
        surface.link_name: surface
        for surface in inputs.task_grip_surfaces.values()
    }
    for link_id, link_name in enumerate(LINK_NAMES):
        indices = np.flatnonzero(base["link_id"] == link_id)
        points = base["position_link_local_m"][indices]
        normals = base["normal_link_local"][indices]
        if not len(indices):
            _fail(f"no positive contact points for {link_name}")
        _, full_query = _surface_query(
            inputs.hand_collision_triangles_by_link[link_name]
        )
        _, pad_query = _surface_query(surfaces[link_name].triangles_local_m)
        _, nonpad_query = _surface_query(noncontact[link_name])
        result["source_full_distance_m"][indices] = full_query.on_surface(points)[1]
        result["pad_distance_m"][indices] = pad_query.on_surface(points)[1]
        result["nonpad_distance_m"][indices] = nonpad_query.on_surface(points)[1]

        geometry = cooked_by_link[link_name]
        all_closest, all_distance, all_triangle = geometry.all_query.on_surface(points)
        all_hull = geometry.triangle_hull_index[all_triangle]
        all_polygon = geometry.triangle_polygon_index[all_triangle]
        exposed_closest, exposed_distance, union_triangle = (
            geometry.union_query.on_surface(points)
        )
        mapped_closest, map_distance, mapped_triangle = (
            geometry.all_query.on_surface(exposed_closest)
        )
        if float(np.max(map_distance)) >= EXPOSED_FACE_MAP_TOLERANCE_M:
            _fail(
                f"{link_name} exposed-to-original face mapping residual is "
                f"{float(np.max(map_distance))} m"
            )
        mapped_hull = geometry.triangle_hull_index[mapped_triangle]
        mapped_polygon = geometry.triangle_polygon_index[mapped_triangle]
        exposed_normal = geometry.union_mesh.face_normals[union_triangle]
        stored_normal = geometry.polygon_planes[mapped_polygon, :3]
        stored_normal /= np.linalg.norm(stored_normal, axis=1)[:, None]
        exposed_dot = np.clip(
            np.abs(np.einsum("ij,ij->i", exposed_normal, normals)), 0.0, 1.0
        )
        stored_dot = np.clip(
            np.abs(np.einsum("ij,ij->i", stored_normal, normals)), 0.0, 1.0
        )
        strict, ambiguous, buried_by, nearest_other = _classify_internal_projection(
            all_closest, all_hull, geometry.hull_halfspaces
        )

        assignments = {
            "cooked_all_closest_position_link_local_m": all_closest,
            "cooked_exposed_closest_position_link_local_m": exposed_closest,
            "cooked_exposed_triangle_normal_link_local": exposed_normal,
            "cooked_exposed_stored_plane_normal_link_local": stored_normal,
            "cooked_all_distance_m": all_distance,
            "cooked_exposed_distance_m": exposed_distance,
            "cooked_exposed_minus_all_distance_m": exposed_distance - all_distance,
            "cooked_exposed_face_map_residual_m": map_distance,
            "cooked_exposed_triangle_normal_abs_dot": exposed_dot,
            "cooked_exposed_triangle_normal_angle_rad": np.arccos(exposed_dot),
            "cooked_exposed_stored_plane_normal_abs_dot": stored_dot,
            "cooked_exposed_stored_plane_normal_angle_rad": np.arccos(stored_dot),
            "nearest_other_hull_max_halfspace_m": nearest_other,
            "cooked_all_nearest_hull": all_hull,
            "cooked_all_nearest_triangle": all_triangle,
            "cooked_all_nearest_polygon": all_polygon,
            "cooked_exposed_nearest_hull": mapped_hull,
            "cooked_exposed_nearest_triangle": mapped_triangle,
            "cooked_exposed_nearest_polygon": mapped_polygon,
            "cooked_exposed_union_triangle": union_triangle,
            "buried_by_hull": buried_by,
            "all_nearest_strictly_buried": strict,
            "all_nearest_boundary_ambiguous": ambiguous,
            "all_exposed_difference_gt_tolerance": (
                exposed_distance - all_distance > STRICT_INTERIOR_TOLERANCE_M
            ),
        }
        for field, values in assignments.items():
            result[field][indices] = values
    for key, value in result.items():
        if value.dtype.kind == "f" and not np.all(np.isfinite(value)):
            _fail(f"nonfinite output array: {key}")
    return result


def _quantiles(values: np.ndarray) -> Mapping[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        name: float(value)
        for name, value in zip(
            ("median", "p95", "p99", "max"),
            np.percentile(values, (50.0, 95.0, 99.0, 100.0)),
        )
    }


def _subset_summary(raw: Mapping[str, np.ndarray], indices: np.ndarray) -> Mapping[str, Any]:
    indices = np.asarray(indices, dtype=np.int64)
    if not len(indices):
        _fail("cannot summarize an empty subset")
    distance_fields = (
        "source_full_distance_m", "pad_distance_m", "nonpad_distance_m",
        "cooked_all_distance_m", "cooked_exposed_distance_m",
        "cooked_exposed_minus_all_distance_m", "cooked_exposed_face_map_residual_m",
    )
    return {
        "point_count": int(len(indices)),
        "distance_quantiles_m": {
            "absolute_separation_m": _quantiles(np.abs(raw["separation_m"][indices])),
            **{field: _quantiles(raw[field][indices]) for field in distance_fields},
        },
        "normal_angle_quantiles_rad": {
            "exposed_triangle": _quantiles(
                raw["cooked_exposed_triangle_normal_angle_rad"][indices]
            ),
            "exposed_stored_plane": _quantiles(
                raw["cooked_exposed_stored_plane_normal_angle_rad"][indices]
            ),
        },
        "fractions": {
            "source_full_distance_le_50um": float(np.mean(
                raw["source_full_distance_m"][indices] <= 50.0e-6
            )),
            "cooked_all_distance_le_50um": float(np.mean(
                raw["cooked_all_distance_m"][indices] <= 50.0e-6
            )),
            "cooked_exposed_distance_le_50um": float(np.mean(
                raw["cooked_exposed_distance_m"][indices] <= 50.0e-6
            )),
            "cooked_exposed_le_abs_separation_plus_50um": float(np.mean(
                raw["cooked_exposed_distance_m"][indices]
                <= np.abs(raw["separation_m"][indices]) + 50.0e-6
            )),
            "cooked_exposed_strictly_closer_than_source_full": float(np.mean(
                raw["cooked_exposed_distance_m"][indices]
                < raw["source_full_distance_m"][indices]
            )),
            "all_nearest_strictly_buried": float(np.mean(
                raw["all_nearest_strictly_buried"][indices]
            )),
            "all_nearest_boundary_ambiguous": float(np.mean(
                raw["all_nearest_boundary_ambiguous"][indices]
            )),
            "all_exposed_difference_gt_0p1um": float(np.mean(
                raw["all_exposed_difference_gt_tolerance"][indices]
            )),
        },
        "counts": {
            "all_nearest_strictly_buried": int(np.count_nonzero(
                raw["all_nearest_strictly_buried"][indices]
            )),
            "all_nearest_boundary_ambiguous": int(np.count_nonzero(
                raw["all_nearest_boundary_ambiguous"][indices]
            )),
        },
    }


def _statistics(
    raw: Mapping[str, np.ndarray], phase_vocabulary: Sequence[str]
) -> Mapping[str, Any]:
    links = {}
    for link_id, link_name in enumerate(LINK_NAMES):
        link_indices = np.flatnonzero(raw["link_id"] == link_id)
        phases = {}
        for phase_id, phase in enumerate(phase_vocabulary):
            indices = np.flatnonzero(
                (raw["link_id"] == link_id) & (raw["phase_id"] == phase_id)
            )
            if len(indices):
                phases[phase] = _subset_summary(raw, indices)
        first_step = int(np.min(raw["step"][link_indices]))
        first_indices = np.flatnonzero(
            (raw["link_id"] == link_id) & (raw["step"] == first_step)
        )
        first_phase_id = int(raw["phase_id"][first_indices[0]])
        links[link_name] = {
            "total": _subset_summary(raw, link_indices),
            "phases": phases,
            "first_positive_contact_step": {
                "step": first_step,
                "simulation_time_s": float(raw["simulation_time_s"][first_indices[0]]),
                "phase": phase_vocabulary[first_phase_id],
                "summary": _subset_summary(raw, first_indices),
                "nearest_exposed_hull": raw["cooked_exposed_nearest_hull"][first_indices].tolist(),
                "nearest_exposed_triangle": raw["cooked_exposed_nearest_triangle"][first_indices].tolist(),
                "nearest_exposed_polygon": raw["cooked_exposed_nearest_polygon"][first_indices].tolist(),
            },
        }
    return {
        "all_points": _subset_summary(raw, np.arange(len(raw["link_id"]))),
        "links": links,
    }


def _output_evidence(
    bindings: Mapping[str, Any],
    cooked: Mapping[str, CookedGeometry],
    raw_path: Path,
) -> Mapping[str, Any]:
    input_paths = {
        "trace": bindings["trace_path"],
        "evaluation": bindings["evaluation_path"],
        "config": bindings["config_path"],
        "cooked_manifest": bindings["manifest_path"],
        "cooked_contract": bindings["contract_path"],
        "robot_asset": bindings["robot_path"],
        "source_stl": bindings["source_path"],
    }
    for group_id, geometry in cooked.items():
        input_paths[f"cooked_npz_{group_id}"] = geometry.npz_path
    code_paths = {
        "validator": Path(__file__).resolve(),
        "historical_evaluator": bindings["evaluator_path"],
        "models": Path(inspect.getfile(load_v2_inputs)).resolve(),
        "task_grip_surface": Path(inspect.getfile(task_noncontact_triangles)).resolve(),
    }
    return {
        "inputs": {
            label: {"path": str(path), "sha256": _sha256_file(path)}
            for label, path in input_paths.items()
        },
        "code": {
            label: {"path": str(path), "sha256": _sha256_file(path)}
            for label, path in code_paths.items()
        },
        "dependencies": {
            "python": {
                "version": platform.python_version(),
                "implementation": platform.python_implementation(),
                "executable": sys.executable,
            },
            "numpy": _module_evidence(np, "numpy"),
            "scipy": _module_evidence(scipy, "scipy"),
            "trimesh": _module_evidence(trimesh, "trimesh"),
            "ijson": _module_evidence(ijson, "ijson"),
            "manifold3d": _module_evidence(manifold3d, "manifold3d"),
        },
        "raw_output": {
            "path": raw_path.name,
            "sha256": _sha256_file(raw_path),
            "size_bytes": raw_path.stat().st_size,
        },
    }


def run(arguments: argparse.Namespace) -> Mapping[str, Any]:
    if arguments.output_dir.exists():
        _fail(f"refusing to overwrite existing output directory: {arguments.output_dir}")
    if arguments.config is None and arguments.contract is None:
        _fail("at least one of --config or --contract must be explicit")
    bindings = _validate_bindings(arguments)
    repository = _repository_root()
    inputs = load_v2_inputs(
        repository,
        config_path=bindings["config_path"],
        object_id=str(bindings["trace_head"]["object_id"]),
    )
    output_specs = bindings["output_by_group"]
    cooked_groups = {
        group_id: _load_cooked_geometry(
            group_id,
            bindings["cooked_dir"] / filename,
            output_specs[group_id],
        )
        for group_id, filename in EXPECTED_COOKED_FILES.items()
    }
    cooked_by_link = {
        "f1Link3": cooked_groups["f1_f3_shared_cooked_convex"],
        "f2Link2": cooked_groups["f2_cooked_convex"],
        "f3Link3": cooked_groups["f1_f3_shared_cooked_convex"],
    }
    object_root = str(bindings["audit_roots"].get("object", ""))
    if not object_root.startswith("/World/"):
        _fail("trace object audit root invalid")
    base = _collect_positive_contacts(
        bindings["trace_path"], object_root, inputs.robot_model
    )
    for link_id, link_name in enumerate(LINK_NAMES):
        actual = int(np.count_nonzero(base["link_id"] == link_id))
        _same(actual, bindings["expected_positive_counts"][link_name], f"{link_name} positive point count")
    phase_vocabulary = list(base["phase_vocabulary"])
    raw = _analyze_contacts(base, inputs, cooked_by_link)
    for name, value in raw.items():
        if value.dtype.kind in {"O", "S", "U", "V"}:
            _fail(f"raw NPZ array is not numeric/bool: {name} {value.dtype}")

    arguments.output_dir.mkdir(parents=True, exist_ok=False)
    raw_path = arguments.output_dir / RAW_NAME
    np.savez_compressed(raw_path, **raw)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "raw_schema_version": RAW_SCHEMA_VERSION,
        "simulation_only": True,
        "object_id": bindings["trace_head"]["object_id"],
        "candidate_id": bindings["trace_head"]["candidate_id"],
        "phase_vocabulary": phase_vocabulary,
        "link_vocabulary": list(LINK_NAMES),
        "array_schema": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in raw.items()
        },
        "constants": {
            "strict_other_hull_interior_tolerance_m": STRICT_INTERIOR_TOLERANCE_M,
            "exposed_face_map_tolerance_m": EXPOSED_FACE_MAP_TOLERANCE_M,
            "positive_contact_filter": "normal_impulse_n_s > 0.0",
            "fk_transform": "(world_position - link_translation) @ link_rotation",
        },
        "cooked_union": {
            group_id: geometry.union_metadata
            for group_id, geometry in cooked_groups.items()
        },
        "statistics": _statistics(raw, phase_vocabulary),
        "evidence_boundary": {
            "classification": "OFFLINE_HISTORICAL_TRACE_TO_FROZEN_COOKED_GEOMETRY_VALIDATION",
            "cache_binding": bindings["manifest"].get("evidence_boundary"),
            "coordinate_registration": (
                "Uses the same frozen robot-model FK and link-local transform as the "
                "hash-bound historical evaluator; it does not query historical runtime actor poses."
            ),
            "contact_position_semantics": (
                "Distance and normal agreement do not identify whether PhysX position is on "
                "shape0, shape1, or another representative contact location."
            ),
            "does_not_prove": [
                "TE connector grasp success", "50 mm lift", "2 s suspended hold",
                "dynamic robustness", "hardware behavior", "analytic global optimality",
                "which exact historical PhysX hull emitted each contact point",
            ],
            "hardware_authorized": False,
            "online_control_role_allowed": False,
        },
    }
    summary["evidence"] = _output_evidence(
        bindings, cooked_groups, raw_path
    )
    summary_path = arguments.output_dir / SUMMARY_NAME
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--cooked-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        summary = run(arguments)
    except (ValidationError, OSError, ValueError, KeyError) as error:
        parser.exit(2, f"validation failed: {error}\n")
    compact = {
        "schema_version": summary["schema_version"],
        "output_dir": str(arguments.output_dir.resolve()),
        "point_counts": {
            link: summary["statistics"]["links"][link]["total"]["point_count"]
            for link in LINK_NAMES
        },
    }
    print(json.dumps(compact, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ValidationError", "main", "run"]


def _collect_positive_contacts(
    trace_path: Path,
    *,
    object_root: str,
    robot_model: object,
) -> Mapping[str, np.ndarray | tuple[str, ...]]:
    rows: dict[str, list[Any]] = {
        "link_id": [],
        "step": [],
        "time_s": [],
        "phase": [],
        "world_position_m": [],
        "world_normal": [],
        "local_position_m": [],
        "local_normal": [],
        "normal_impulse_n_s": [],
        "separation_m": [],
    }
    with trace_path.open("rb") as stream:
        for sample in ijson.items(stream, "samples.item", use_float=True):
            if not isinstance(sample, dict):
                _fail("trace sample is not an object")
            step = int(sample.get("step"))
            time_s = float(sample.get("simulation_time_s"))
            phase = str(sample.get("phase"))
            active = np.asarray(sample.get("active_positions_rad"), dtype=np.float64)
            if active.ndim != 1 or not np.all(np.isfinite(active)):
                _fail(f"sample {step} has invalid active joint positions")
            contacts = sample.get("contacts")
            if not isinstance(contacts, dict):
                continue
            pending: dict[str, list[tuple[np.ndarray, np.ndarray, float, float]]] = {
                name: [] for name in LINK_NAMES
            }
            for header in contacts.get("tensor_headers", ()):
                if not isinstance(header, dict):
                    _fail(f"sample {step} has a non-object tensor header")
                paths = tuple(map(str, header.get("paths", ())))
                if not any(_below(path, object_root) for path in paths):
                    continue
                links = [
                    name
                    for name in LINK_NAMES
                    if any(f"/{name}" in path for path in paths)
                ]
                if len(links) != 1:
                    continue
                for contact in header.get("contacts", ()):
                    if not isinstance(contact, dict):
                        _fail(f"sample {step} has a non-object contact")
                    impulse = float(contact.get("normal_impulse_n_s", 0.0))
                    if impulse <= 0.0:
                        continue
                    point = _finite_vector(contact.get("position_m"), 3, "contact position")
                    normal = _finite_vector(contact.get("normal"), 3, "contact normal")
                    norm = float(np.linalg.norm(normal))
                    if not math.isfinite(norm) or norm <= 1.0e-14:
                        _fail(f"sample {step} has a zero contact normal")
                    normal = normal / norm
                    separation = float(contact.get("separation_m"))
                    if not math.isfinite(impulse) or not math.isfinite(separation):
                        _fail(f"sample {step} has a non-finite contact scalar")
                    pending[links[0]].append((point, normal, impulse, separation))
            if not any(pending.values()):
                continue
            transforms = robot_model.forward_kinematics(active, enforce_limits=False)
            for link_name, contacts_for_link in pending.items():
                if not contacts_for_link:
                    continue
                transform = np.asarray(transforms[link_name], dtype=np.float64)
                if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
                    _fail(f"sample {step} has an invalid {link_name} transform")
                rotation = transform[:3, :3]
                translation = transform[:3, 3]
                for world_point, world_normal, impulse, separation in contacts_for_link:
                    local_point = (world_point - translation) @ rotation
                    local_normal = world_normal @ rotation
                    local_normal /= np.linalg.norm(local_normal)
                    rows["link_id"].append(LINK_TO_ID[link_name])
                    rows["step"].append(step)
                    rows["time_s"].append(time_s)
                    rows["phase"].append(phase)
                    rows["world_position_m"].append(world_point)
                    rows["world_normal"].append(world_normal)
                    rows["local_position_m"].append(local_point)
                    rows["local_normal"].append(local_normal)
                    rows["normal_impulse_n_s"].append(impulse)
                    rows["separation_m"].append(separation)
    phases = tuple(sorted(set(map(str, rows["phase"]))))
    if not phases:
        _fail("trace contains no positive-impulse terminal contacts")
    phase_to_id = {value: index for index, value in enumerate(phases)}
    count = len(rows["link_id"])
    result: dict[str, np.ndarray | tuple[str, ...]] = {
        "phase_vocabulary": phases,
        "link_id": np.asarray(rows["link_id"], dtype=np.int8),
        "step": np.asarray(rows["step"], dtype=np.int64),
        "time_s": np.asarray(rows["time_s"], dtype=np.float64),
        "phase_id": np.asarray(
            [phase_to_id[str(value)] for value in rows["phase"]], dtype=np.int16
        ),
        "world_position_m": np.asarray(rows["world_position_m"], dtype=np.float64),
        "world_normal": np.asarray(rows["world_normal"], dtype=np.float64),
        "local_position_m": np.asarray(rows["local_position_m"], dtype=np.float64),
        "local_normal": np.asarray(rows["local_normal"], dtype=np.float64),
        "normal_impulse_n_s": np.asarray(rows["normal_impulse_n_s"], dtype=np.float64),
        "separation_m": np.asarray(rows["separation_m"], dtype=np.float64),
    }
    for name, value in result.items():
        if name == "phase_vocabulary":
            continue
        array = np.asarray(value)
        if len(array) != count or not np.all(np.isfinite(array)):
            _fail(f"collected contact array {name} is inconsistent")
    return result


def _acute_angle_rad(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3:
        _fail("normal angle inputs must have shape (N,3)")
    a /= np.linalg.norm(a, axis=1)[:, None]
    b /= np.linalg.norm(b, axis=1)[:, None]
    dot = np.abs(np.einsum("ij,ij->i", a, b))
    return np.arccos(np.clip(dot, 0.0, 1.0))


def _project_contacts(
    contacts: Mapping[str, np.ndarray | tuple[str, ...]],
    *,
    inputs: object,
    cooked_by_link: Mapping[str, CookedGeometry],
) -> Mapping[str, np.ndarray | tuple[str, ...]]:
    link_ids = np.asarray(contacts["link_id"], dtype=np.int8)
    local_points = np.asarray(contacts["local_position_m"], dtype=np.float64)
    local_normals = np.asarray(contacts["local_normal"], dtype=np.float64)
    count = len(link_ids)
    output: dict[str, np.ndarray | tuple[str, ...]] = {
        key: value for key, value in contacts.items()
    }
    scalar_float_names = (
        "source_full_distance_m",
        "source_pad_distance_m",
        "source_nonpad_distance_m",
        "source_nonpad_minus_pad_distance_m",
        "cooked_all_distance_m",
        "cooked_exposed_distance_m",
        "cooked_exposed_to_original_surface_m",
        "cooked_exposed_triangle_normal_angle_rad",
        "cooked_stored_plane_normal_angle_rad",
        "cooked_triangle_to_stored_plane_angle_rad",
        "cooked_all_nearest_other_hull_max_plane_m",
    )
    for name in scalar_float_names:
        output[name] = np.full(count, np.nan, dtype=np.float64)
    for name in (
        "source_full_face",
        "source_pad_face",
        "source_nonpad_face",
        "cooked_all_triangle",
        "cooked_all_hull",
        "cooked_exposed_union_triangle",
        "cooked_exposed_source_triangle",
        "cooked_exposed_hull",
        "cooked_exposed_polygon",
        "cooked_all_buried_by_hull",
    ):
        output[name] = np.full(count, -1, dtype=np.int32)
    output["cooked_all_projection_strictly_buried"] = np.zeros(count, dtype=np.bool_)
    output["cooked_all_projection_overlap_boundary"] = np.zeros(count, dtype=np.bool_)

    surfaces_by_link = {
        surface.link_name: surface for surface in inputs.task_grip_surfaces.values()
    }
    nonpad_by_link = task_noncontact_triangles(
        inputs.hand_collision_triangles_by_link, inputs.task_grip_surfaces
    )
    for link_id, link_name in enumerate(LINK_NAMES):
        indices = np.flatnonzero(link_ids == link_id)
        if not len(indices):
            _fail(f"trace has no positive contacts for {link_name}")
        points = local_points[indices]
        normals = local_normals[indices]
        full_mesh, full_query = _surface_query(
            np.asarray(inputs.hand_collision_triangles_by_link[link_name])
        )
        pad_mesh, pad_query = _surface_query(
            np.asarray(surfaces_by_link[link_name].triangles_local_m)
        )
        nonpad_mesh, nonpad_query = _surface_query(
            np.asarray(nonpad_by_link[link_name])
        )
        _full_point, full_distance, full_face = full_query.on_surface(points)
        _pad_point, pad_distance, pad_face = pad_query.on_surface(points)
        _nonpad_point, nonpad_distance, nonpad_face = nonpad_query.on_surface(points)
        output["source_full_distance_m"][indices] = full_distance
        output["source_pad_distance_m"][indices] = pad_distance
        output["source_nonpad_distance_m"][indices] = nonpad_distance
        output["source_nonpad_minus_pad_distance_m"][indices] = (
            nonpad_distance - pad_distance
        )
        output["source_full_face"][indices] = full_face
        output["source_pad_face"][indices] = pad_face
        output["source_nonpad_face"][indices] = nonpad_face

        cooked = cooked_by_link[link_name]
        all_point, all_distance, all_face = cooked.all_query.on_surface(points)
        exposed_point, exposed_distance, exposed_face = cooked.union_query.on_surface(points)
        source_triangle = cooked.union_source_triangle[exposed_face]
        exposed_hull = cooked.triangle_hull_index[source_triangle]
        exposed_polygon = cooked.triangle_polygon_index[source_triangle]
        all_hull = cooked.triangle_hull_index[all_face]
        output["cooked_all_distance_m"][indices] = all_distance
        output["cooked_exposed_distance_m"][indices] = exposed_distance
        output["cooked_all_triangle"][indices] = all_face
        output["cooked_all_hull"][indices] = all_hull
        output["cooked_exposed_union_triangle"][indices] = exposed_face
        output["cooked_exposed_source_triangle"][indices] = source_triangle
        output["cooked_exposed_hull"][indices] = exposed_hull
        output["cooked_exposed_polygon"][indices] = exposed_polygon

        _mapped, map_distance, _mapped_face = cooked.all_query.on_surface(exposed_point)
        output["cooked_exposed_to_original_surface_m"][indices] = map_distance
        maximum_map_distance = float(np.max(map_distance))
        if maximum_map_distance > EXPOSED_FACE_MAP_TOLERANCE_M:
            _fail(
                f"{link_name} exposed union map distance {maximum_map_distance:.17g} m "
                f"exceeds {EXPOSED_FACE_MAP_TOLERANCE_M:.17g} m"
            )

        triangle_normal = np.asarray(cooked.union_mesh.face_normals)[exposed_face]
        plane_normal = np.asarray(cooked.polygon_planes[exposed_polygon, :3], dtype=np.float64)
        plane_normal /= np.linalg.norm(plane_normal, axis=1)[:, None]
        output["cooked_exposed_triangle_normal_angle_rad"][indices] = _acute_angle_rad(
            normals, triangle_normal
        )
        output["cooked_stored_plane_normal_angle_rad"][indices] = _acute_angle_rad(
            normals, plane_normal
        )
        output["cooked_triangle_to_stored_plane_angle_rad"][indices] = _acute_angle_rad(
            triangle_normal, plane_normal
        )

        strict, boundary, buried_by, nearest_max = _classify_internal_projection(
            all_point, all_hull, cooked.hull_halfspaces
        )
        output["cooked_all_projection_strictly_buried"][indices] = strict
        output["cooked_all_projection_overlap_boundary"][indices] = boundary
        output["cooked_all_buried_by_hull"][indices] = buried_by
        output["cooked_all_nearest_other_hull_max_plane_m"][indices] = nearest_max

    for name, value in output.items():
        if name == "phase_vocabulary":
            continue
        array = np.asarray(value)
        if len(array) != count:
            _fail(f"projected array {name} length changed")
        if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
            _fail(f"projected array {name} contains non-finite values")
    return output


def _quantiles(values: np.ndarray) -> Mapping[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.all(np.isfinite(array)):
        _fail("cannot summarize an empty/non-finite array")
    probabilities = (0.5, 0.95, 0.99, 1.0)
    result = np.quantile(array, probabilities)
    return {
        "median": float(result[0]),
        "p95": float(result[1]),
        "p99": float(result[2]),
        "maximum": float(result[3]),
    }


def _link_statistics(
    raw: Mapping[str, np.ndarray | tuple[str, ...]],
    link_id: int,
) -> Mapping[str, Any]:
    mask = np.asarray(raw["link_id"]) == link_id
    indices = np.flatnonzero(mask)
    if not len(indices):
        _fail(f"no rows for {LINK_NAMES[link_id]}")
    separation = np.abs(np.asarray(raw["separation_m"])[indices])
    full = np.asarray(raw["source_full_distance_m"])[indices]
    pad = np.asarray(raw["source_pad_distance_m"])[indices]
    all_distance = np.asarray(raw["cooked_all_distance_m"])[indices]
    exposed = np.asarray(raw["cooked_exposed_distance_m"])[indices]
    triangle_angle = np.asarray(raw["cooked_exposed_triangle_normal_angle_rad"])[indices]
    plane_angle = np.asarray(raw["cooked_stored_plane_normal_angle_rad"])[indices]
    time = np.asarray(raw["time_s"])[indices]
    first_index = int(indices[int(np.argmin(time))])
    phase_ids = np.asarray(raw["phase_id"])[indices]
    phases = tuple(raw["phase_vocabulary"])
    phase_rows = {}
    for phase_id in np.unique(phase_ids):
        local = indices[phase_ids == phase_id]
        phase_rows[phases[int(phase_id)]] = {
            "count": int(len(local)),
            "cooked_exposed_distance_m": _quantiles(
                np.asarray(raw["cooked_exposed_distance_m"])[local]
            ),
        }
    cooked_threshold = exposed <= 50.0e-6
    source_threshold = full <= 50.0e-6
    return {
        "terminal_link": LINK_NAMES[link_id],
        "count": int(len(indices)),
        "absolute_separation_m": _quantiles(separation),
        "source_full_distance_m": _quantiles(full),
        "source_pad_distance_m": _quantiles(pad),
        "cooked_all_distance_m": _quantiles(all_distance),
        "cooked_exposed_distance_m": _quantiles(exposed),
        "cooked_exposed_normal_angle_rad": _quantiles(triangle_angle),
        "cooked_stored_plane_normal_angle_rad": _quantiles(plane_angle),
        "fraction_source_full_within_50um": float(np.mean(source_threshold)),
        "fraction_cooked_exposed_within_50um": float(np.mean(cooked_threshold)),
        "fraction_cooked_exposed_within_abs_separation_plus_50um": float(
            np.mean(exposed <= separation + 50.0e-6)
        ),
        "fraction_cooked_exposed_strictly_closer_than_source_full": float(
            np.mean(exposed < full)
        ),
        "cooked_median_reduction_fraction_from_source_full": float(
            1.0 - np.median(exposed) / np.median(full)
        ),
        "strictly_buried_all_face_projection_count": int(
            np.count_nonzero(
                np.asarray(raw["cooked_all_projection_strictly_buried"])[indices]
            )
        ),
        "overlap_boundary_all_face_projection_count": int(
            np.count_nonzero(
                np.asarray(raw["cooked_all_projection_overlap_boundary"])[indices]
            )
        ),
        "first_positive_contact": {
            "row_index": first_index,
            "step": int(np.asarray(raw["step"])[first_index]),
            "time_s": float(np.asarray(raw["time_s"])[first_index]),
            "phase": phases[int(np.asarray(raw["phase_id"])[first_index])],
            "absolute_separation_m": float(abs(np.asarray(raw["separation_m"])[first_index])),
            "source_full_distance_m": float(np.asarray(raw["source_full_distance_m"])[first_index]),
            "cooked_exposed_distance_m": float(np.asarray(raw["cooked_exposed_distance_m"])[first_index]),
            "cooked_exposed_normal_angle_rad": float(
                np.asarray(raw["cooked_exposed_triangle_normal_angle_rad"])[first_index]
            ),
            "cooked_exposed_hull": int(np.asarray(raw["cooked_exposed_hull"])[first_index]),
            "cooked_exposed_source_triangle": int(
                np.asarray(raw["cooked_exposed_source_triangle"])[first_index]
            ),
        },
        "by_phase": phase_rows,
    }


def _write_outputs(
    arguments: argparse.Namespace,
    binding: Mapping[str, Any],
    raw: Mapping[str, np.ndarray | tuple[str, ...]],
    cooked_by_group: Mapping[str, CookedGeometry],
) -> Mapping[str, Any]:
    output_dir = arguments.output_dir.resolve()
    if output_dir.exists():
        _fail(f"refusing to overwrite existing output directory: {output_dir}")
    if not output_dir.parent.is_dir():
        _fail(f"output parent is not a directory: {output_dir.parent}")
    output_dir.mkdir()
    arrays = {
        name: np.asarray(value)
        for name, value in raw.items()
        if name != "phase_vocabulary"
    }
    raw_path = output_dir / RAW_NAME
    np.savez_compressed(raw_path, **arrays)
    with np.load(raw_path, allow_pickle=False) as loaded:
        if set(loaded.files) != set(arrays):
            _fail("raw NPZ names changed after write")
        for name, expected in arrays.items():
            value = np.asarray(loaded[name])
            if value.dtype != expected.dtype or value.shape != expected.shape or not np.array_equal(
                value, expected, equal_nan=True
            ):
                _fail(f"raw NPZ round-trip mismatch: {name}")
    code_path = Path(__file__).resolve()
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "simulation_only": True,
        "hardware_authorized": False,
        "claim_scope": "OFFLINE_HISTORICAL_CONTACT_TO_COOKED_COLLIDER_VALIDATION",
        "inputs": {
            "trace": {"path": str(binding["trace_path"]), "sha256": _sha256_file(binding["trace_path"])},
            "evaluation": {"path": str(binding["evaluation_path"]), "sha256": _sha256_file(binding["evaluation_path"])},
            "cooked_manifest": {"path": str(binding["manifest_path"]), "sha256": _sha256_file(binding["manifest_path"])},
            "cooked_contract": {"path": str(binding["contract_path"]), "sha256": _sha256_file(binding["contract_path"])},
            "config": {"path": str(binding["config_path"]), "sha256": _sha256_file(binding["config_path"])},
            "robot_asset": {"path": str(binding["robot_path"]), "sha256": _sha256_file(binding["robot_path"])},
            "source_stl": {"path": str(binding["source_path"]), "sha256": _sha256_file(binding["source_path"])},
        },
        "implementation": {
            "path": str(code_path),
            "sha256": _sha256_file(code_path),
            "python": platform.python_version(),
            "numpy": _module_evidence(np, "numpy"),
            "scipy": _module_evidence(scipy, "scipy"),
            "trimesh": _module_evidence(trimesh, "trimesh"),
            "manifold3d": _module_evidence(manifold3d, "manifold3d"),
            "ijson": _module_evidence(ijson, "ijson"),
        },
        "phase_vocabulary": list(raw["phase_vocabulary"]),
        "raw": {
            "path": RAW_NAME,
            "sha256": _sha256_file(raw_path),
            "size_bytes": raw_path.stat().st_size,
            "schema_version": RAW_SCHEMA_VERSION,
            "arrays": {
                name: {"dtype": str(value.dtype), "shape": list(value.shape)}
                for name, value in sorted(arrays.items())
            },
        },
        "cooked_geometry": {
            group_id: dict(geometry.union_metadata)
            for group_id, geometry in cooked_by_group.items()
        },
        "links": [_link_statistics(raw, index) for index in range(len(LINK_NAMES))],
        "checks": {
            "expected_positive_contact_counts": binding["expected_positive_counts"],
            "actual_positive_contact_counts": {
                name: int(np.count_nonzero(np.asarray(raw["link_id"]) == index))
                for index, name in enumerate(LINK_NAMES)
            },
            "exposed_union_to_original_cooked_max_m": float(
                np.max(np.asarray(raw["cooked_exposed_to_original_surface_m"]))
            ),
            "strict_interior_tolerance_m": STRICT_INTERIOR_TOLERANCE_M,
            "exposed_face_map_tolerance_m": EXPOSED_FACE_MAP_TOLERANCE_M,
        },
        "evidence_boundary": {
            "cooked_cache_binding_is_direct_asset_hash_inside_ddc": False,
            "historical_engine_logs_retained": False,
            "contact_position_sidedness_known": False,
            "online_control_role_allowed": False,
            "proves_dynamic_grasp_success": False,
            "proves_analytic_global_optimality": False,
            "interpretation": (
                "Distances validate that saved positive-impulse contact positions are far "
                "closer to the hash-frozen cooked convex union than to the authored source "
                "surface. They do not identify which historical PhysX shape emitted each "
                "contact or define whether position_m belongs to shape 0, shape 1, or a "
                "contact construction point."
            ),
        },
    }
    _same(
        summary["checks"]["actual_positive_contact_counts"],
        summary["checks"]["expected_positive_contact_counts"],
        "raw/evaluation positive contact counts",
    )
    summary_path = output_dir / SUMMARY_NAME
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "output_dir": str(output_dir),
        "summary": {"path": SUMMARY_NAME, "sha256": _sha256_file(summary_path)},
        "raw": summary["raw"],
        "links": summary["links"],
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--cooked-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    binding = _validate_bindings(arguments)
    repository = _repository_root()
    inputs = load_v2_inputs(
        repository,
        config_path=binding["config_path"],
        object_id=str(binding["trace_head"]["object_id"]),
    )
    if inputs.task_grip_surfaces is None:
        _fail("bound V2 input lacks direct nail-free task surfaces")
    output_by_group = binding["output_by_group"]
    cooked_by_group = {
        group_id: _load_cooked_geometry(
            group_id,
            binding["cooked_dir"] / filename,
            output_by_group[group_id],
        )
        for group_id, filename in EXPECTED_COOKED_FILES.items()
    }
    cooked_by_link = {
        "f1Link3": cooked_by_group["f1_f3_shared_cooked_convex"],
        "f2Link2": cooked_by_group["f2_cooked_convex"],
        "f3Link3": cooked_by_group["f1_f3_shared_cooked_convex"],
    }
    object_root = str(binding["audit_roots"].get("object", ""))
    if not object_root.startswith("/"):
        _fail("trace object audit root is invalid")
    contacts = _collect_positive_contacts(
        binding["trace_path"],
        object_root=object_root,
        robot_model=inputs.robot_model,
    )
    raw = _project_contacts(contacts, inputs=inputs, cooked_by_link=cooked_by_link)
    result = _write_outputs(arguments, binding, raw, cooked_by_group)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
