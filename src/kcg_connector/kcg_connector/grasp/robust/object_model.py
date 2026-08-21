"""Provenance-bound object meshes for object-agnostic grasp synthesis.

This module deliberately stops at geometry and physical properties.  It does
not contain connector model names, grasp candidates, or simulator state.  STL
coordinates are accepted only with an explicit unit and are converted to SI at
the loader boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import struct
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


DEFAULT_EXTERNAL_SEMANTIC = "external_surface"
DETERMINISTIC_STL_LOADER = "CARTS_GRASP_DETERMINISTIC_STL_V1"
DETERMINISTIC_NPZ_LOADER = "CARTS_GRASP_DETERMINISTIC_VISUAL_SUBTREE_NPZ_V1"
CARTS_VISUAL_SUBTREE_NPZ = "CARTS_GRASP_VISUAL_SUBTREE_NPZ_V1"

_METERS_PER_UNIT = {
    "m": 1.0,
    "meter": 1.0,
    "metre": 1.0,
    "mm": 1.0e-3,
    "millimeter": 1.0e-3,
    "millimetre": 1.0e-3,
    "cm": 1.0e-2,
    "centimeter": 1.0e-2,
    "centimetre": 1.0e-2,
    "um": 1.0e-6,
    "micrometer": 1.0e-6,
    "micrometre": 1.0e-6,
    "in": 0.0254,
    "inch": 0.0254,
}


def file_sha256(path: Path | str) -> str:
    """Return the SHA-256 of the exact source bytes."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def meters_per_unit(unit: str) -> float:
    """Resolve an explicit length unit to its SI multiplier.

    STL has no normative unit field.  Guessing from a bounding box would make
    the same asset mean different things in different experiments, so unknown
    or missing units are rejected.
    """

    key = str(unit).strip().lower()
    try:
        return _METERS_PER_UNIT[key]
    except KeyError as error:
        supported = ", ".join(sorted(_METERS_PER_UNIT))
        raise ValueError(
            f"unsupported or implicit STL unit {unit!r}; supported units: {supported}"
        ) from error


def _readonly_array(
    value: Any,
    *,
    dtype: np.dtype[Any] | type,
    shape_tail: tuple[int, ...],
    name: str,
) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    tail_start = array.ndim - len(shape_tail)
    if array.ndim < len(shape_tail) or tuple(array.shape[tail_start:]) != shape_tail:
        raise ValueError(f"{name} must end in shape {shape_tail}, got {array.shape}")
    if np.issubdtype(array.dtype, np.floating) and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    array.setflags(write=False)
    return array


def _numeric_tolerance(array: np.ndarray, multiplier: float = 256.0) -> float:
    scale = max(1.0, float(np.linalg.norm(array, ord=np.inf)))
    return multiplier * np.finfo(np.float64).eps * scale


def _rigid_transform_parts(transform: Sequence[Sequence[float]]) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("rigid transform must be a finite 4x4 matrix")
    tolerance = _numeric_tolerance(matrix)
    if not np.allclose(
        matrix[3], np.asarray((0.0, 0.0, 0.0, 1.0)), rtol=0.0, atol=tolerance
    ):
        raise ValueError("rigid transform must have homogeneous last row [0, 0, 0, 1]")
    rotation = matrix[:3, :3]
    if not np.allclose(
        rotation.T @ rotation, np.eye(3), rtol=0.0, atol=tolerance
    ) or not np.isclose(
        np.linalg.det(rotation), 1.0, rtol=0.0, atol=tolerance
    ):
        raise ValueError("transform rotation must belong to SO(3)")
    return rotation, matrix[:3, 3].copy()


@dataclass(frozen=True)
class AssetProvenance:
    """Traceability record for an exact source mesh and its unit conversion."""

    source_path: str
    source_sha256: str
    source_class: str
    source_format: str
    source_unit: str
    meters_per_source_unit: float
    loader: str = DETERMINISTIC_STL_LOADER
    dropped_degenerate_face_count: int = 0

    def __post_init__(self) -> None:
        digest = str(self.source_sha256).lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("source_sha256 must be a 64-character hexadecimal digest")
        if not str(self.source_path):
            raise ValueError("source_path cannot be empty")
        if not str(self.source_class):
            raise ValueError("source_class cannot be empty")
        if self.source_format not in {
            "ASCII_STL",
            "BINARY_STL",
            CARTS_VISUAL_SUBTREE_NPZ,
        }:
            raise ValueError(
                "source_format must identify STL or the deterministic CARTS visual NPZ"
            )
        if not str(self.loader):
            raise ValueError("provenance loader cannot be empty")
        expected_scale = meters_per_unit(self.source_unit)
        if float(self.meters_per_source_unit) != expected_scale:
            raise ValueError("provenance unit multiplier does not match source_unit")
        if int(self.dropped_degenerate_face_count) < 0:
            raise ValueError("dropped_degenerate_face_count cannot be negative")
        object.__setattr__(self, "source_sha256", digest)
        object.__setattr__(self, "meters_per_source_unit", expected_scale)
        object.__setattr__(
            self, "dropped_degenerate_face_count", int(self.dropped_degenerate_face_count)
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "source_class": self.source_class,
            "source_format": self.source_format,
            "source_unit": self.source_unit,
            "meters_per_source_unit": self.meters_per_source_unit,
            "loader": self.loader,
            "dropped_degenerate_face_count": self.dropped_degenerate_face_count,
        }

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]


@dataclass(frozen=True)
class TriangleMesh:
    """Indexed triangular surface in meters with one semantic per face."""

    vertices_m: np.ndarray
    faces: np.ndarray
    face_semantics: tuple[str, ...]

    def __post_init__(self) -> None:
        vertices = _readonly_array(
            self.vertices_m, dtype=np.float64, shape_tail=(3,), name="vertices_m"
        )
        faces = _readonly_array(self.faces, dtype=np.int64, shape_tail=(3,), name="faces")
        if vertices.ndim != 2 or faces.ndim != 2:
            raise ValueError("vertices_m and faces must be rank-two arrays")
        if len(vertices) < 3 or len(faces) < 1:
            raise ValueError("triangle mesh must contain at least one face")
        if np.any(faces < 0) or np.any(faces >= len(vertices)):
            raise ValueError("face index is outside the vertex array")
        semantics = tuple(str(value) for value in self.face_semantics)
        if len(semantics) != len(faces) or any(not value for value in semantics):
            raise ValueError("face_semantics must contain one non-empty label per face")
        triangles = vertices[faces]
        doubled_areas = np.linalg.norm(
            np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
            axis=1,
        )
        if np.any(doubled_areas == 0.0):
            raise ValueError("triangle mesh contains an exactly degenerate face")
        object.__setattr__(self, "vertices_m", vertices)
        object.__setattr__(self, "faces", faces)
        object.__setattr__(self, "face_semantics", semantics)

    @property
    def face_vertices_m(self) -> np.ndarray:
        return self.vertices_m[self.faces]

    @property
    def face_area_vectors_m2(self) -> np.ndarray:
        triangles = self.face_vertices_m
        return 0.5 * np.cross(
            triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        )

    @property
    def face_areas_m2(self) -> np.ndarray:
        return np.linalg.norm(self.face_area_vectors_m2, axis=1)

    @property
    def face_normals(self) -> np.ndarray:
        vectors = self.face_area_vectors_m2
        return vectors / np.linalg.norm(vectors, axis=1)[:, None]

    @property
    def face_centroids_m(self) -> np.ndarray:
        return np.mean(self.face_vertices_m, axis=1)

    @property
    def bounds_m(self) -> tuple[np.ndarray, np.ndarray]:
        return np.min(self.vertices_m, axis=0), np.max(self.vertices_m, axis=0)

    def _edge_incidence(self) -> dict[tuple[int, int], list[tuple[int, int]]]:
        incidence: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for face_index, face in enumerate(self.faces):
            for start, end in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                start_int, end_int = int(start), int(end)
                key = (min(start_int, end_int), max(start_int, end_int))
                direction = 1 if (start_int, end_int) == key else -1
                incidence.setdefault(key, []).append((face_index, direction))
        return incidence

    @property
    def is_watertight(self) -> bool:
        return all(len(entries) == 2 for entries in self._edge_incidence().values())

    def connected_face_components(self) -> tuple[tuple[int, ...], ...]:
        neighbors: list[set[int]] = [set() for _ in range(len(self.faces))]
        for entries in self._edge_incidence().values():
            for face_index, _direction in entries:
                neighbors[face_index].update(
                    other for other, _other_direction in entries if other != face_index
                )
        unseen = set(range(len(self.faces)))
        components: list[tuple[int, ...]] = []
        while unseen:
            root = min(unseen)
            stack = [root]
            unseen.remove(root)
            component: list[int] = []
            while stack:
                face_index = stack.pop()
                component.append(face_index)
                for neighbor in sorted(neighbors[face_index], reverse=True):
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        stack.append(neighbor)
            components.append(tuple(sorted(component)))
        return tuple(components)

    def oriented_outward(self) -> "TriangleMesh":
        """Make every manifold component consistently outward oriented.

        Closed components use the sign of their oriented volume.  Open
        components can be made locally consistent but retain the source's
        global winding because an intrinsic inside/outside is unavailable.
        """

        incidence = self._edge_incidence()
        non_manifold = [edge for edge, entries in incidence.items() if len(entries) > 2]
        if non_manifold:
            raise ValueError(
                f"cannot orient non-manifold mesh; {len(non_manifold)} edges have >2 faces"
            )
        neighbors: list[list[tuple[int, int, int]]] = [
            [] for _ in range(len(self.faces))
        ]
        for entries in incidence.values():
            if len(entries) == 2:
                (first, first_direction), (second, second_direction) = entries
                neighbors[first].append((second, first_direction, second_direction))
                neighbors[second].append((first, second_direction, first_direction))

        orientation = np.zeros(len(self.faces), dtype=np.int8)
        components: list[list[int]] = []
        for root in range(len(self.faces)):
            if orientation[root] != 0:
                continue
            orientation[root] = 1
            stack = [root]
            component: list[int] = []
            while stack:
                face_index = stack.pop()
                component.append(face_index)
                for neighbor, own_direction, neighbor_direction in neighbors[face_index]:
                    required = -int(orientation[face_index]) * own_direction * neighbor_direction
                    if orientation[neighbor] == 0:
                        orientation[neighbor] = required
                        stack.append(neighbor)
                    elif int(orientation[neighbor]) != required:
                        raise ValueError("mesh surface is not consistently orientable")
            components.append(component)

        faces = np.array(self.faces, dtype=np.int64, copy=True)
        initially_flipped = np.flatnonzero(orientation < 0)
        faces[initially_flipped, 1], faces[initially_flipped, 2] = (
            faces[initially_flipped, 2].copy(),
            faces[initially_flipped, 1].copy(),
        )

        edge_counts = {edge: len(entries) for edge, entries in incidence.items()}
        for component in components:
            component_set = set(component)
            closed = True
            for edge, entries in incidence.items():
                if any(face_index in component_set for face_index, _ in entries):
                    if edge_counts[edge] != 2:
                        closed = False
                        break
            if not closed:
                continue
            component_faces = faces[np.asarray(component, dtype=np.int64)]
            triangles = self.vertices_m[component_faces]
            six_times_volume = float(
                np.sum(
                    np.einsum(
                        "ij,ij->i",
                        triangles[:, 0],
                        np.cross(triangles[:, 1], triangles[:, 2]),
                    )
                )
            )
            if six_times_volume < 0.0:
                indices = np.asarray(component, dtype=np.int64)
                faces[indices, 1], faces[indices, 2] = (
                    faces[indices, 2].copy(),
                    faces[indices, 1].copy(),
                )
        return TriangleMesh(
            faces=faces,
            vertices_m=self.vertices_m,
            face_semantics=self.face_semantics,
        )

    def transformed(self, transform: Sequence[Sequence[float]]) -> "TriangleMesh":
        rotation, translation = _rigid_transform_parts(transform)
        vertices = self.vertices_m @ rotation.T + translation
        return TriangleMesh(
            vertices_m=vertices,
            faces=self.faces,
            face_semantics=self.face_semantics,
        )

    def with_face_semantics(self, face_semantics: Iterable[str]) -> "TriangleMesh":
        return TriangleMesh(
            vertices_m=self.vertices_m,
            faces=self.faces,
            face_semantics=tuple(face_semantics),
        )

    @property
    def geometry_sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"CARTS_GRASP_TRIANGLE_MESH_SI_V1\0")
        for array in (
            np.asarray(self.vertices_m, dtype="<f8"),
            np.asarray(self.faces, dtype="<i8"),
        ):
            digest.update(np.asarray(array.shape, dtype="<i8").tobytes(order="C"))
            digest.update(array.tobytes(order="C"))
        for semantic in self.face_semantics:
            encoded = semantic.encode("utf-8")
            digest.update(struct.pack("<Q", len(encoded)))
            digest.update(encoded)
        return digest.hexdigest()


def _binary_triangle_array(data: bytes) -> np.ndarray | None:
    if len(data) < 84:
        return None
    triangle_count = struct.unpack_from("<I", data, 80)[0]
    expected_size = 84 + 50 * triangle_count
    if expected_size != len(data):
        return None
    record_dtype = np.dtype(
        [
            ("normal", "<f4", (3,)),
            ("vertices", "<f4", (3, 3)),
            ("attribute", "<u2"),
        ]
    )
    records = np.frombuffer(data, dtype=record_dtype, count=triangle_count, offset=84)
    return np.asarray(records["vertices"], dtype=np.float64)


def _ascii_triangle_array(data: bytes) -> np.ndarray:
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError("file is neither a strict binary STL nor ASCII STL") from error
    vertices: list[tuple[float, float, float]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        parts = line.strip().split()
        if not parts or parts[0].lower() != "vertex":
            continue
        if len(parts) != 4:
            raise ValueError(f"invalid ASCII STL vertex at line {line_number}")
        try:
            vertex = tuple(float(value) for value in parts[1:])
        except ValueError as error:
            raise ValueError(f"invalid ASCII STL number at line {line_number}") from error
        if not np.all(np.isfinite(vertex)):
            raise ValueError(f"non-finite ASCII STL vertex at line {line_number}")
        vertices.append(vertex)  # type: ignore[arg-type]
    if not vertices or len(vertices) % 3:
        raise ValueError("ASCII STL must contain exactly three vertices per facet")
    return np.asarray(vertices, dtype=np.float64).reshape((-1, 3, 3))


def _indexed_triangles(
    triangles_source_units: np.ndarray,
    *,
    scale_to_m: float,
    face_semantics: Sequence[str] | None,
    default_face_semantic: str,
) -> tuple[TriangleMesh, int]:
    if triangles_source_units.ndim != 3 or triangles_source_units.shape[1:] != (3, 3):
        raise ValueError("STL triangle array must have shape (face_count, 3, 3)")
    if len(triangles_source_units) == 0 or not np.all(np.isfinite(triangles_source_units)):
        raise ValueError("STL contains no finite triangles")
    if face_semantics is None:
        semantics = tuple(default_face_semantic for _ in range(len(triangles_source_units)))
    else:
        semantics = tuple(str(value) for value in face_semantics)
        if len(semantics) != len(triangles_source_units):
            raise ValueError("face_semantics length must equal STL facet count")

    vertex_index: dict[tuple[float, float, float], int] = {}
    vertices_source: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    kept_semantics: list[str] = []
    dropped = 0
    for triangle, semantic in zip(triangles_source_units, semantics):
        doubled_area = np.linalg.norm(
            np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        )
        if doubled_area == 0.0:
            dropped += 1
            continue
        face: list[int] = []
        for vertex in triangle:
            key = (float(vertex[0]), float(vertex[1]), float(vertex[2]))
            index = vertex_index.get(key)
            if index is None:
                index = len(vertices_source)
                vertex_index[key] = index
                vertices_source.append(key)
            face.append(index)
        faces.append((face[0], face[1], face[2]))
        kept_semantics.append(semantic)
    if not faces:
        raise ValueError("STL contains only degenerate facets")
    mesh = TriangleMesh(
        vertices_m=np.asarray(vertices_source, dtype=np.float64) * float(scale_to_m),
        faces=np.asarray(faces, dtype=np.int64),
        face_semantics=tuple(kept_semantics),
    )
    return mesh, dropped


def load_stl_mesh(
    path: Path | str,
    *,
    unit: str,
    source_class: str = "CAD_DERIVED_STL",
    face_semantics: Sequence[str] | None = None,
    default_face_semantic: str = DEFAULT_EXTERNAL_SEMANTIC,
    orient_outward: bool = True,
) -> tuple[TriangleMesh, AssetProvenance]:
    """Read a binary or ASCII STL deterministically and convert it to meters."""

    source_path = Path(path)
    data = source_path.read_bytes()
    triangles = _binary_triangle_array(data)
    if triangles is None:
        source_format = "ASCII_STL"
        triangles = _ascii_triangle_array(data)
    else:
        source_format = "BINARY_STL"
    scale = meters_per_unit(unit)
    mesh, dropped = _indexed_triangles(
        triangles,
        scale_to_m=scale,
        face_semantics=face_semantics,
        default_face_semantic=default_face_semantic,
    )
    if orient_outward:
        mesh = mesh.oriented_outward()
    provenance = AssetProvenance(
        source_path=str(source_path.resolve()),
        source_sha256=hashlib.sha256(data).hexdigest(),
        source_class=str(source_class),
        source_format=source_format,
        source_unit=str(unit).strip().lower(),
        meters_per_source_unit=scale,
        dropped_degenerate_face_count=dropped,
    )
    return mesh, provenance


def load_visual_subtree_npz_mesh(
    path: Path | str,
    *,
    source_class: str,
    face_semantics: Sequence[str] | None = None,
    default_face_semantic: str = DEFAULT_EXTERNAL_SEMANTIC,
    orient_outward: bool = False,
) -> tuple[TriangleMesh, AssetProvenance]:
    """Read the deterministic SI mesh emitted by the visual-subtree exporter.

    ``allow_pickle=False`` and exact array-shape validation keep this loader a
    geometry-only path.  ``source_prim_paths`` may be present for provenance,
    but it is never interpreted as collision or contact truth.
    """

    source_path = Path(path)
    data_hash = file_sha256(source_path)
    with np.load(source_path, allow_pickle=False) as archive:
        required = {"vertices_m", "faces"}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"visual-subtree NPZ is missing arrays: {missing}")
        vertices = np.asarray(archive["vertices_m"], dtype=np.float64)
        faces = np.asarray(archive["faces"], dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,):
        raise ValueError("visual-subtree vertices_m must have shape (V, 3)")
    if faces.ndim != 2 or faces.shape[1:] != (3,):
        raise ValueError("visual-subtree faces must have shape (F, 3)")
    if face_semantics is None:
        semantics = tuple(default_face_semantic for _ in range(len(faces)))
    else:
        semantics = tuple(str(value) for value in face_semantics)
        if len(semantics) != len(faces):
            raise ValueError("face_semantics length must equal NPZ face count")
    mesh = TriangleMesh(
        vertices_m=vertices,
        faces=faces,
        face_semantics=semantics,
    )
    if orient_outward:
        mesh = mesh.oriented_outward()
    provenance = AssetProvenance(
        source_path=str(source_path.resolve()),
        source_sha256=data_hash,
        source_class=str(source_class),
        source_format=CARTS_VISUAL_SUBTREE_NPZ,
        source_unit="m",
        meters_per_source_unit=1.0,
        loader=DETERMINISTIC_NPZ_LOADER,
        dropped_degenerate_face_count=0,
    )
    return mesh, provenance


@dataclass(frozen=True)
class ObjectGraspModel:
    """SI object geometry, physical properties, semantics, and provenance."""

    mesh: TriangleMesh
    provenance: AssetProvenance
    assembly_axis: np.ndarray
    mass_kg: float
    center_of_mass_m: np.ndarray
    inertia_kg_m2: np.ndarray
    allowed_contact_semantics: frozenset[str]
    forbidden_contact_semantics: frozenset[str] = field(default_factory=frozenset)
    assembly_axis_origin_m: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    source_to_object_transform: np.ndarray = field(
        default_factory=lambda: np.eye(4, dtype=np.float64)
    )

    def __post_init__(self) -> None:
        axis = _readonly_array(
            self.assembly_axis, dtype=np.float64, shape_tail=(3,), name="assembly_axis"
        )
        if axis.shape != (3,) or float(np.linalg.norm(axis)) == 0.0:
            raise ValueError("assembly_axis must be a non-zero three-vector")
        axis = np.array(axis / np.linalg.norm(axis), dtype=np.float64, copy=True)
        axis.setflags(write=False)
        axis_origin = _readonly_array(
            self.assembly_axis_origin_m,
            dtype=np.float64,
            shape_tail=(3,),
            name="assembly_axis_origin_m",
        )
        center_of_mass = _readonly_array(
            self.center_of_mass_m,
            dtype=np.float64,
            shape_tail=(3,),
            name="center_of_mass_m",
        )
        inertia = _readonly_array(
            self.inertia_kg_m2,
            dtype=np.float64,
            shape_tail=(3, 3),
            name="inertia_kg_m2",
        )
        if axis_origin.shape != (3,) or center_of_mass.shape != (3,) or inertia.shape != (3, 3):
            raise ValueError("axis origin, center of mass, and inertia must be 3D SI values")
        mass = float(self.mass_kg)
        if not np.isfinite(mass) or mass <= 0.0:
            raise ValueError("mass_kg must be finite and positive")
        tolerance = _numeric_tolerance(inertia)
        if not np.allclose(inertia, inertia.T, rtol=0.0, atol=tolerance):
            raise ValueError("inertia_kg_m2 must be symmetric")
        symmetric_inertia = np.array(0.5 * (inertia + inertia.T), copy=True)
        if float(np.min(np.linalg.eigvalsh(symmetric_inertia))) < -tolerance:
            raise ValueError("inertia_kg_m2 must be positive semidefinite")
        symmetric_inertia.setflags(write=False)
        allowed = frozenset(str(value) for value in self.allowed_contact_semantics)
        forbidden = frozenset(str(value) for value in self.forbidden_contact_semantics)
        if not allowed or any(not value for value in allowed | forbidden):
            raise ValueError("allowed contact semantics must be non-empty labels")
        if allowed & forbidden:
            raise ValueError("allowed and forbidden contact semantics must be disjoint")
        available = frozenset(self.mesh.face_semantics)
        if not allowed & available:
            raise ValueError("no mesh face has an allowed contact semantic")
        source_to_object = np.asarray(self.source_to_object_transform, dtype=np.float64)
        _rigid_transform_parts(source_to_object)
        source_to_object = np.array(source_to_object, copy=True)
        source_to_object.setflags(write=False)

        object.__setattr__(self, "assembly_axis", axis)
        object.__setattr__(self, "assembly_axis_origin_m", axis_origin)
        object.__setattr__(self, "center_of_mass_m", center_of_mass)
        object.__setattr__(self, "inertia_kg_m2", symmetric_inertia)
        object.__setattr__(self, "mass_kg", mass)
        object.__setattr__(self, "allowed_contact_semantics", allowed)
        object.__setattr__(self, "forbidden_contact_semantics", forbidden)
        object.__setattr__(self, "source_to_object_transform", source_to_object)

    @classmethod
    def from_stl(
        cls,
        path: Path | str,
        *,
        unit: str,
        source_class: str,
        assembly_axis: Sequence[float],
        mass_kg: float,
        center_of_mass_m: Sequence[float],
        inertia_kg_m2: Sequence[Sequence[float]],
        allowed_contact_semantics: Iterable[str] = (DEFAULT_EXTERNAL_SEMANTIC,),
        forbidden_contact_semantics: Iterable[str] = (),
        assembly_axis_origin_m: Sequence[float] = (0.0, 0.0, 0.0),
        face_semantics: Sequence[str] | None = None,
        default_face_semantic: str = DEFAULT_EXTERNAL_SEMANTIC,
        orient_outward: bool = True,
        require_watertight: bool = False,
    ) -> "ObjectGraspModel":
        mesh, provenance = load_stl_mesh(
            path,
            unit=unit,
            source_class=source_class,
            face_semantics=face_semantics,
            default_face_semantic=default_face_semantic,
            orient_outward=orient_outward,
        )
        if require_watertight and not mesh.is_watertight:
            raise ValueError("object grasp STL must be watertight")
        return cls(
            mesh=mesh,
            provenance=provenance,
            assembly_axis=np.asarray(assembly_axis, dtype=np.float64),
            assembly_axis_origin_m=np.asarray(assembly_axis_origin_m, dtype=np.float64),
            mass_kg=mass_kg,
            center_of_mass_m=np.asarray(center_of_mass_m, dtype=np.float64),
            inertia_kg_m2=np.asarray(inertia_kg_m2, dtype=np.float64),
            allowed_contact_semantics=frozenset(allowed_contact_semantics),
            forbidden_contact_semantics=frozenset(forbidden_contact_semantics),
        )

    @classmethod
    def from_visual_subtree_npz(
        cls,
        path: Path | str,
        *,
        source_class: str,
        assembly_axis: Sequence[float],
        mass_kg: float,
        center_of_mass_m: Sequence[float],
        inertia_kg_m2: Sequence[Sequence[float]],
        allowed_contact_semantics: Iterable[str] = (DEFAULT_EXTERNAL_SEMANTIC,),
        forbidden_contact_semantics: Iterable[str] = (),
        assembly_axis_origin_m: Sequence[float] = (0.0, 0.0, 0.0),
        face_semantics: Sequence[str] | None = None,
        default_face_semantic: str = DEFAULT_EXTERNAL_SEMANTIC,
        orient_outward: bool = False,
        require_watertight: bool = False,
    ) -> "ObjectGraspModel":
        mesh, provenance = load_visual_subtree_npz_mesh(
            path,
            source_class=source_class,
            face_semantics=face_semantics,
            default_face_semantic=default_face_semantic,
            orient_outward=orient_outward,
        )
        if require_watertight and not mesh.is_watertight:
            raise ValueError("object grasp NPZ must be watertight")
        return cls(
            mesh=mesh,
            provenance=provenance,
            assembly_axis=np.asarray(assembly_axis, dtype=np.float64),
            assembly_axis_origin_m=np.asarray(assembly_axis_origin_m, dtype=np.float64),
            mass_kg=mass_kg,
            center_of_mass_m=np.asarray(center_of_mass_m, dtype=np.float64),
            inertia_kg_m2=np.asarray(inertia_kg_m2, dtype=np.float64),
            allowed_contact_semantics=frozenset(allowed_contact_semantics),
            forbidden_contact_semantics=frozenset(forbidden_contact_semantics),
        )

    @property
    def geometry_sha256(self) -> str:
        return self.mesh.geometry_sha256

    @property
    def com_m(self) -> np.ndarray:
        return self.center_of_mass_m

    @property
    def contact_face_mask(self) -> np.ndarray:
        mask = np.asarray(
            [
                semantic in self.allowed_contact_semantics
                and semantic not in self.forbidden_contact_semantics
                for semantic in self.mesh.face_semantics
            ],
            dtype=bool,
        )
        mask.setflags(write=False)
        return mask

    def transformed(self, transform: Sequence[Sequence[float]]) -> "ObjectGraspModel":
        """Apply an SE(3) frame change to all geometric and inertial quantities."""

        rotation, translation = _rigid_transform_parts(transform)
        matrix = np.asarray(transform, dtype=np.float64)
        return ObjectGraspModel(
            mesh=self.mesh.transformed(matrix),
            provenance=self.provenance,
            assembly_axis=rotation @ self.assembly_axis,
            assembly_axis_origin_m=rotation @ self.assembly_axis_origin_m + translation,
            mass_kg=self.mass_kg,
            center_of_mass_m=rotation @ self.center_of_mass_m + translation,
            inertia_kg_m2=rotation @ self.inertia_kg_m2 @ rotation.T,
            allowed_contact_semantics=self.allowed_contact_semantics,
            forbidden_contact_semantics=self.forbidden_contact_semantics,
            source_to_object_transform=matrix @ self.source_to_object_transform,
        )

    def with_face_semantics(
        self,
        face_semantics: Iterable[str],
        *,
        allowed_contact_semantics: Iterable[str] | None = None,
        forbidden_contact_semantics: Iterable[str] | None = None,
    ) -> "ObjectGraspModel":
        """Return a contract-labelled view without changing source geometry."""

        return ObjectGraspModel(
            mesh=self.mesh.with_face_semantics(face_semantics),
            provenance=self.provenance,
            assembly_axis=self.assembly_axis,
            assembly_axis_origin_m=self.assembly_axis_origin_m,
            mass_kg=self.mass_kg,
            center_of_mass_m=self.center_of_mass_m,
            inertia_kg_m2=self.inertia_kg_m2,
            allowed_contact_semantics=(
                self.allowed_contact_semantics
                if allowed_contact_semantics is None
                else frozenset(allowed_contact_semantics)
            ),
            forbidden_contact_semantics=(
                self.forbidden_contact_semantics
                if forbidden_contact_semantics is None
                else frozenset(forbidden_contact_semantics)
            ),
            source_to_object_transform=self.source_to_object_transform,
        )


__all__ = [
    "AssetProvenance",
    "CARTS_VISUAL_SUBTREE_NPZ",
    "DEFAULT_EXTERNAL_SEMANTIC",
    "DETERMINISTIC_NPZ_LOADER",
    "DETERMINISTIC_STL_LOADER",
    "ObjectGraspModel",
    "TriangleMesh",
    "file_sha256",
    "load_stl_mesh",
    "load_visual_subtree_npz_mesh",
    "meters_per_unit",
]
