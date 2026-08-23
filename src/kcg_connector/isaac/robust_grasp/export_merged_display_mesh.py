#!/usr/bin/env python3
"""Merge a frozen visual-subtree triangle export into a few display meshes.

The output is display packaging only.  It preserves every exported triangle,
groups faces only by their authored ``displayColor``, and deliberately authors
no rigid-body, collision, contact, or mass API.  The frozen source USD and NPZ
are opened read-only and are hash-checked again after the derived files are
written.

The source visual-subtree NPZ stores one path per source Gprim but not explicit
face ranges.  This tool reconstructs those ranges only when the number and
order of disconnected face components exactly match the path inventory.  Any
ambiguous mapping is rejected rather than guessed from connector-specific
names or coordinate thresholds.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "carts_grasp_merged_display_mesh_v1"
METHOD_ID = "CARTS_EXACT_FACE_INVENTORY_DISPLAY_COLOR_MERGE_V1"
CLAIM_SCOPE = "DISPLAY_PACKAGING_ONLY_NOT_COLLISION_OR_PHYSICS"
_GPRIM_TYPES = frozenset({"Mesh", "Cylinder", "Sphere"})
_DEFINITION = re.compile(r'^(\s*)def\s+(\w+)\s+"([^"]+)"')
_DISPLAY_COLOR = re.compile(
    r"primvars:displayColor\s*=\s*\[\(([^)]+)\)\]"
)


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-npz", type=Path, required=True)
    parser.add_argument("--source-usda", type=Path, required=True)
    parser.add_argument("--source-stage", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--composed-prefix", required=True)
    parser.add_argument("--source-prefix", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(tag: bytes, array: np.ndarray, dtype: str) -> str:
    canonical = np.asarray(array, dtype=dtype)
    digest = hashlib.sha256()
    digest.update(tag + b"\0")
    digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes(order="C"))
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _normalise_prefix(value: str, *, name: str) -> str:
    prefix = str(value).strip()
    if not prefix.startswith("/") or prefix == "/" or prefix.endswith("/"):
        raise ValueError(f"{name} must be an absolute non-root USD path without a trailing slash")
    return prefix


def parse_usda_gprim_display_colors(
    path: Path,
) -> dict[str, tuple[str, tuple[float, float, float]]]:
    """Read Gprim type and constant display color from deterministic USDA text."""

    if not path.is_file():
        raise FileNotFoundError(path)
    stack: list[tuple[int, str]] = []
    current_path: str | None = None
    records: dict[str, dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            definition = _DEFINITION.match(line)
            if definition:
                indentation = len(definition.group(1))
                type_name = definition.group(2)
                prim_name = definition.group(3)
                while stack and stack[-1][0] >= indentation:
                    stack.pop()
                stack.append((indentation, prim_name))
                current_path = "/" + "/".join(item[1] for item in stack)
                if type_name in _GPRIM_TYPES:
                    if current_path in records:
                        raise ValueError(f"duplicate USDA Gprim path: {current_path}")
                    records[current_path] = {
                        "type_name": type_name,
                        "display_color": None,
                        "definition_line": line_number,
                    }
                continue
            if current_path not in records:
                continue
            color_match = _DISPLAY_COLOR.search(line)
            if color_match is None:
                continue
            values = tuple(
                float(value.strip()) for value in color_match.group(1).split(",")
            )
            if len(values) != 3 or not all(math.isfinite(value) for value in values):
                raise ValueError(
                    f"invalid constant displayColor on {current_path}: {values}"
                )
            if records[current_path]["display_color"] is not None:
                raise ValueError(f"duplicate displayColor on {current_path}")
            records[current_path]["display_color"] = values

    missing = sorted(
        prim_path
        for prim_path, record in records.items()
        if record["display_color"] is None
    )
    if missing:
        raise ValueError(
            "every source Gprim must have one constant displayColor; missing: "
            + ", ".join(missing[:5])
        )
    return {
        prim_path: (
            str(record["type_name"]),
            tuple(float(value) for value in record["display_color"]),
        )
        for prim_path, record in records.items()
    }


def _find(parent: np.ndarray, item: int) -> int:
    root = int(item)
    while int(parent[root]) != root:
        parent[root] = parent[int(parent[root])]
        root = int(parent[root])
    return root


def _union(parent: np.ndarray, rank: np.ndarray, first: int, second: int) -> None:
    first_root = _find(parent, first)
    second_root = _find(parent, second)
    if first_root == second_root:
        return
    if int(rank[first_root]) < int(rank[second_root]):
        first_root, second_root = second_root, first_root
    parent[second_root] = first_root
    if int(rank[first_root]) == int(rank[second_root]):
        rank[first_root] += 1


def connected_face_components(faces: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return edge-connected face components ordered by first source face."""

    indexed = np.asarray(faces, dtype=np.int64)
    if indexed.ndim != 2 or indexed.shape[1:] != (3,) or len(indexed) == 0:
        raise ValueError("faces must be a non-empty Fx3 array")
    if np.any(indexed < 0):
        raise ValueError("faces contain a negative vertex index")
    parent = np.arange(len(indexed), dtype=np.int64)
    rank = np.zeros(len(indexed), dtype=np.int8)
    edges = np.sort(
        np.concatenate(
            (
                indexed[:, (0, 1)],
                indexed[:, (1, 2)],
                indexed[:, (2, 0)],
            ),
            axis=0,
        ),
        axis=1,
    )
    face_indices = np.tile(np.arange(len(indexed), dtype=np.int64), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    ordered_edges = edges[order]
    ordered_faces = face_indices[order]
    start = 0
    while start < len(ordered_edges):
        end = start + 1
        while end < len(ordered_edges) and np.array_equal(
            ordered_edges[end], ordered_edges[start]
        ):
            end += 1
        anchor = int(ordered_faces[start])
        for position in range(start + 1, end):
            _union(parent, rank, anchor, int(ordered_faces[position]))
        start = end

    grouped: dict[int, list[int]] = {}
    for face_index in range(len(indexed)):
        grouped.setdefault(_find(parent, face_index), []).append(face_index)
    components = tuple(
        np.asarray(face_group, dtype=np.int64)
        for face_group in sorted(grouped.values(), key=lambda values: values[0])
    )
    for component in components:
        component.setflags(write=False)
    return components


def _replace_prefix(path: str, source: str, target: str) -> str:
    if path == source:
        return target
    if not path.startswith(source + "/"):
        raise ValueError(f"source Gprim path {path!r} is outside {source!r}")
    return target + path[len(source) :]


def assign_face_colors(
    faces: np.ndarray,
    source_prim_paths: Sequence[str],
    appearance: Mapping[str, tuple[str, tuple[float, float, float]]],
    *,
    composed_prefix: str,
    source_prefix: str,
) -> tuple[np.ndarray, tuple[tuple[float, float, float], ...], Counter[str]]:
    """Bind every face to one authored source color without name heuristics."""

    composed = _normalise_prefix(composed_prefix, name="composed_prefix")
    source = _normalise_prefix(source_prefix, name="source_prefix")
    paths = tuple(str(value) for value in source_prim_paths)
    components = connected_face_components(faces)
    if len(components) != len(paths):
        raise ValueError(
            "disconnected face-component count does not match source Gprim inventory: "
            f"components={len(components)}, source_prims={len(paths)}"
        )

    face_color_indices = np.full(len(faces), -1, dtype=np.int64)
    component_colors: list[tuple[float, float, float]] = []
    source_types: Counter[str] = Counter()
    for index, (component, composed_path) in enumerate(zip(components, paths)):
        expected = np.arange(int(component[0]), int(component[-1]) + 1)
        if not np.array_equal(component, expected):
            raise ValueError(
                f"source component {index} is not a contiguous exporter face range"
            )
        source_path = _replace_prefix(composed_path, composed, source)
        try:
            type_name, color = appearance[source_path]
        except KeyError as error:
            raise ValueError(
                f"source Gprim appearance is missing for {source_path}"
            ) from error
        source_types[type_name] += 1
        component_colors.append(color)

    ordered_colors = tuple(sorted(set(component_colors)))
    color_to_index = {color: index for index, color in enumerate(ordered_colors)}
    for component, color in zip(components, component_colors):
        face_color_indices[component] = color_to_index[color]
    if np.any(face_color_indices < 0):
        raise ValueError("at least one source face has no display-color assignment")
    face_color_indices.setflags(write=False)
    return face_color_indices, ordered_colors, source_types


def _local_group(
    vertices_m: np.ndarray,
    faces: np.ndarray,
    source_face_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    source_faces = faces[source_face_indices]
    used_vertices = np.unique(source_faces.reshape(-1))
    local_faces = np.searchsorted(used_vertices, source_faces)
    local_vertices = vertices_m[used_vertices]
    return local_vertices, local_faces


def _format_float(value: float) -> str:
    number = float(np.float32(value))
    if number == 0.0:
        return "0"
    return repr(number)


def _append_integer_array(
    lines: list[str],
    declaration: str,
    values: Iterable[int],
    *,
    indentation: str,
    values_per_line: int = 24,
) -> None:
    materialised = [str(int(value)) for value in values]
    lines.append(f"{indentation}{declaration} = [")
    for start in range(0, len(materialised), values_per_line):
        chunk = ", ".join(materialised[start : start + values_per_line])
        suffix = "," if start + values_per_line < len(materialised) else ""
        lines.append(f"{indentation}    {chunk}{suffix}")
    lines.append(f"{indentation}]")


def _build_usda(
    groups: Sequence[dict[str, object]],
    *,
    source_mesh_sha256: str,
    source_stage_sha256: str,
    source_usda_sha256: str,
    source_prim_count: int,
    source_triangle_count: int,
    maximum_vertex_rounding_m: float,
) -> str:
    lines = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "World"',
        "    kilogramsPerUnit = 1",
        "    metersPerUnit = 1",
        '    upAxis = "Z"',
        ")",
        "",
        'def Xform "World"',
        "{",
        '    def Xform "D38999LoosePlugRenderMergedV1"',
        "    {",
        f'        custom string kcg:claimScope = "{CLAIM_SCOPE}"',
        '        custom bool kcg:collisionEligible = 0',
        '        custom bool kcg:formalGeometryCandidate = 0',
        f'        custom string kcg:methodId = "{METHOD_ID}"',
        '        custom bool kcg:physicsAuthored = 0',
        f'        custom string kcg:sourceMeshSha256 = "{source_mesh_sha256}"',
        f'        custom string kcg:sourceStageSha256 = "{source_stage_sha256}"',
        f'        custom string kcg:sourceUsdaSha256 = "{source_usda_sha256}"',
        f"        custom int kcg:sourcePrimCount = {source_prim_count}",
        f"        custom int kcg:sourceTriangleCount = {source_triangle_count}",
        f"        custom int kcg:mergedMeshCount = {len(groups)}",
        f"        custom double kcg:maximumVertexRoundingM = {maximum_vertex_rounding_m!r}",
        '        token purpose = "render"',
        "",
        '        def Xform "LoosePlug"',
        "        {",
    ]
    for group_index, group in enumerate(groups):
        vertices = np.asarray(group["vertices_m"], dtype=np.float32)
        faces = np.asarray(group["faces"], dtype=np.int64)
        color = tuple(float(value) for value in group["color"])
        lines.extend(
            (
                f'            def Mesh "Color_{group_index:03d}"',
                "            {",
                f"                custom int kcg:sourceFaceCount = {len(faces)}",
                '                uniform token orientation = "rightHanded"',
                '                uniform token subdivisionScheme = "none"',
                "                point3f[] points = [",
            )
        )
        for vertex_index, vertex in enumerate(vertices):
            suffix = "," if vertex_index + 1 < len(vertices) else ""
            coordinates = ", ".join(_format_float(value) for value in vertex)
            lines.append(f"                    ({coordinates}){suffix}")
        lines.append("                ]")
        _append_integer_array(
            lines,
            "int[] faceVertexCounts",
            (3 for _ in range(len(faces))),
            indentation="                ",
        )
        _append_integer_array(
            lines,
            "int[] faceVertexIndices",
            faces.reshape(-1),
            indentation="                ",
        )
        color_text = ", ".join(_format_float(value) for value in color)
        lines.extend(
            (
                f"                color3f[] primvars:displayColor = [({color_text})]",
                "            }",
                "",
            )
        )
    lines.extend(("        }", "    }", "}", ""))
    return "\n".join(lines)


def export_merged_display_mesh(
    *,
    mesh_npz: Path,
    source_usda: Path,
    source_stage: Path,
    source_manifest: Path,
    composed_prefix: str,
    source_prefix: str,
    output_path: Path,
    manifest_path: Path,
) -> dict[str, object]:
    inputs = (mesh_npz, source_usda, source_stage, source_manifest)
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_path.resolve() in {path.resolve() for path in inputs}:
        raise ValueError("derived output cannot overwrite a source file")
    if manifest_path.resolve() in {path.resolve() for path in inputs}:
        raise ValueError("derived manifest cannot overwrite a source file")

    hashes_before = {str(path.resolve()): _sha256(path) for path in inputs}
    source_document = json.loads(source_manifest.read_text(encoding="utf-8"))
    if source_document.get("output_sha256") != hashes_before[str(mesh_npz.resolve())]:
        raise ValueError("mesh NPZ does not match its frozen source manifest")
    if source_document.get("source_stage_sha256") != hashes_before[str(source_stage.resolve())]:
        raise ValueError("source stage does not match its frozen source manifest")

    with np.load(mesh_npz, allow_pickle=False) as archive:
        required = {"vertices_m", "faces", "source_prim_paths"}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"visual-subtree NPZ is missing arrays: {missing}")
        vertices_m = np.asarray(archive["vertices_m"], dtype=np.float64)
        faces = np.asarray(archive["faces"], dtype=np.int64)
        source_prim_paths = tuple(str(value) for value in archive["source_prim_paths"])
    if vertices_m.ndim != 2 or vertices_m.shape[1:] != (3,):
        raise ValueError("vertices_m must have shape (V, 3)")
    if faces.ndim != 2 or faces.shape[1:] != (3,):
        raise ValueError("faces must have shape (F, 3)")
    if not np.all(np.isfinite(vertices_m)):
        raise ValueError("vertices_m contains non-finite coordinates")
    if np.any(faces < 0) or np.any(faces >= len(vertices_m)):
        raise ValueError("face index is outside vertices_m")
    if int(source_document.get("source_mesh_prim_count", -1)) != len(source_prim_paths):
        raise ValueError("source Gprim count does not match the frozen manifest")

    appearance = parse_usda_gprim_display_colors(source_usda)
    face_color_indices, colors, source_types = assign_face_colors(
        faces,
        source_prim_paths,
        appearance,
        composed_prefix=composed_prefix,
        source_prefix=source_prefix,
    )
    groups: list[dict[str, object]] = []
    face_assignment = np.zeros(len(faces), dtype=np.int64)
    reconstructed = np.empty((len(faces), 3, 3), dtype=np.float64)
    quantized = np.empty((len(faces), 3, 3), dtype=np.float64)
    for color_index, color in enumerate(colors):
        source_face_indices = np.flatnonzero(face_color_indices == color_index)
        local_vertices, local_faces = _local_group(
            vertices_m, faces, source_face_indices
        )
        face_assignment[source_face_indices] += 1
        reconstructed[source_face_indices] = local_vertices[local_faces]
        quantized_vertices = local_vertices.astype(np.float32)
        quantized[source_face_indices] = quantized_vertices[local_faces]
        groups.append(
            {
                "color": color,
                "source_face_indices": source_face_indices,
                "vertices_m": local_vertices,
                "faces": local_faces,
            }
        )
    source_triangles = vertices_m[faces]
    if not np.all(face_assignment == 1):
        raise ValueError("source faces were not assigned exactly once")
    if not np.array_equal(reconstructed, source_triangles):
        raise ValueError("unquantized merged groups changed source triangle coordinates")
    vertex_rounding = np.linalg.norm(quantized - source_triangles, axis=2)
    maximum_vertex_rounding_m = float(np.max(vertex_rounding))

    text = _build_usda(
        groups,
        source_mesh_sha256=hashes_before[str(mesh_npz.resolve())],
        source_stage_sha256=hashes_before[str(source_stage.resolve())],
        source_usda_sha256=hashes_before[str(source_usda.resolve())],
        source_prim_count=len(source_prim_paths),
        source_triangle_count=len(faces),
        maximum_vertex_rounding_m=maximum_vertex_rounding_m,
    )
    forbidden_tokens = (
        "PhysicsCollisionAPI",
        "PhysicsRigidBodyAPI",
        "physics:collisionEnabled",
        "physics:rigidBodyEnabled",
        "physics:mass",
        "physxCollision:",
        "physxRigidBody:",
    )
    present_forbidden = [token for token in forbidden_tokens if token in text]
    if present_forbidden:
        raise ValueError(f"display-only USDA contains physics tokens: {present_forbidden}")
    if text.count('def Mesh "Color_') != len(groups):
        raise ValueError("generated USDA mesh inventory mismatch")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    output_sha256 = _sha256(output_path)
    hashes_after = {str(path.resolve()): _sha256(path) for path in inputs}
    if hashes_after != hashes_before:
        raise RuntimeError("a frozen source file changed while writing the derived asset")

    source_type_counts = dict(sorted(source_types.items()))
    expected_type_counts = dict(source_document.get("source_gprim_type_counts", {}))
    if source_type_counts != expected_type_counts:
        raise ValueError(
            "source Gprim type counts do not match the frozen source manifest: "
            f"parsed={source_type_counts}, expected={expected_type_counts}"
        )
    source_triangle_sha256 = _array_sha256(
        b"CARTS_SOURCE_TRIANGLE_SOUP_BINARY64_V1", source_triangles, "<f8"
    )
    quantized_triangle_sha256 = _array_sha256(
        b"CARTS_MERGED_TRIANGLE_SOUP_BINARY32_V1", quantized, "<f4"
    )
    document: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "claim_scope": CLAIM_SCOPE,
        "source_mesh_npz": str(mesh_npz.resolve()),
        "source_mesh_npz_sha256": hashes_before[str(mesh_npz.resolve())],
        "source_stage": str(source_stage.resolve()),
        "source_stage_sha256": hashes_before[str(source_stage.resolve())],
        "source_usda": str(source_usda.resolve()),
        "source_usda_sha256": hashes_before[str(source_usda.resolve())],
        "source_manifest": str(source_manifest.resolve()),
        "source_manifest_sha256": hashes_before[str(source_manifest.resolve())],
        "source_prim_count": len(source_prim_paths),
        "source_gprim_type_counts": source_type_counts,
        "source_vertex_count": len(vertices_m),
        "source_triangle_count": len(faces),
        "source_triangle_soup_binary64_sha256": source_triangle_sha256,
        "merged_mesh_count": len(groups),
        "merged_display_colors": [list(color) for color in colors],
        "merged_face_counts": [int(len(group["faces"])) for group in groups],
        "merged_vertex_counts": [int(len(group["vertices_m"])) for group in groups],
        "all_source_faces_assigned_exactly_once": True,
        "unquantized_triangle_coordinates_preserved_exactly": True,
        "usd_point_storage": "point3f",
        "maximum_vertex_rounding_m": maximum_vertex_rounding_m,
        "merged_triangle_soup_binary32_sha256": quantized_triangle_sha256,
        "analytic_primitive_tessellation": source_document.get(
            "analytic_primitive_tessellation"
        ),
        "output": str(output_path.resolve()),
        "output_sha256": output_sha256,
        "output_size_bytes": output_path.stat().st_size,
        "source_files_unchanged": True,
        "source_stage_modified": False,
        "physics_authored": False,
        "collision_eligible": False,
        "formal_geometry_candidate": False,
        "render_performance_measured": False,
        "isaac_load_validated": False,
        "dynamic_launch_allowed": False,
    }
    manifest_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return document


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _arguments(argv)
    report = export_merged_display_mesh(
        mesh_npz=arguments.mesh_npz,
        source_usda=arguments.source_usda,
        source_stage=arguments.source_stage,
        source_manifest=arguments.source_manifest,
        composed_prefix=arguments.composed_prefix,
        source_prefix=arguments.source_prefix,
        output_path=arguments.output,
        manifest_path=arguments.manifest,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
