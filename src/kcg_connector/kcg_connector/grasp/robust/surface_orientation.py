"""Fail-closed orientation audit for source-indexed binary64 triangle meshes.

The audit in this module is deliberately narrower than a solid classifier.  It
proves closed two-manifold incidence, orientability, and a non-zero *exact*
signed six-volume for each source-indexed connected component.  It then picks
the unique locally-consistent winding whose component volume is positive.

Positive component volume is not, by itself, a proof of a material outward
normal.  A self-intersecting surface can have non-zero signed volume, and a
component soup needs nesting parity to distinguish exterior shells from void
boundaries.  V1 therefore records both checks as ``UNVERIFIED`` and always
keeps ``formal_outward_eligible`` false.  No geometric epsilon, coordinate
welding, collision truth, or object-specific rule is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import struct
from typing import Any, Sequence

import numpy as np


METHOD_ID = "CARTS_GRASP_SOURCE_INDEXED_SURFACE_ORIENTATION_V1"
UNVERIFIED = "UNVERIFIED"
VERIFIED = "VERIFIED"


class SurfaceBoundaryRole(str, Enum):
    """Explicit topology role assigned by an upstream object contract."""

    SINGLE_CLOSED_BOUNDARY = "SINGLE_CLOSED_BOUNDARY"
    SOURCE_INDEXED_CLOSED_COMPONENT_SOUP = (
        "SOURCE_INDEXED_CLOSED_COMPONENT_SOUP"
    )


class SurfaceOrientationAuditError(ValueError):
    """Fail-closed audit error with a stable machine-readable reason code."""

    def __init__(self, code: str, detail: str) -> None:
        if not code or not detail:
            raise ValueError(
                "surface orientation error code and detail cannot be empty"
            )
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(f"{self.code}: {self.detail}")


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


@dataclass(frozen=True)
class ExactDyadic:
    """Canonical integer-times-power-of-two representation.

    ``numerator * 2**binary_exponent`` is exact.  A non-zero numerator must be
    odd, so equal dyadic values have exactly one representation.
    """

    numerator: int
    binary_exponent: int

    def __post_init__(self) -> None:
        if type(self.numerator) is not int or type(self.binary_exponent) is not int:
            raise TypeError("ExactDyadic fields must be Python integers")
        if self.numerator == 0:
            if self.binary_exponent != 0:
                raise ValueError("zero ExactDyadic must use binary_exponent=0")
        elif self.numerator % 2 == 0:
            raise ValueError("non-zero ExactDyadic numerator must be odd")

    @property
    def sign(self) -> int:
        return (self.numerator > 0) - (self.numerator < 0)


@dataclass(frozen=True)
class SurfaceOrientationComponentRecord:
    """Immutable record for one canonically ordered source-face component."""

    component_index: int
    minimum_source_face_index: int
    source_face_count: int
    source_edge_count: int
    source_same_direction_edge_count: int
    source_winding_consistent: bool
    source_consistent_winding_sign_to_positive_volume: int
    positive_volume_winding_flip_count: int
    positive_signed_six_volume: ExactDyadic
    source_face_indices_sha256: str
    positive_volume_winding_sha256: str

    def __post_init__(self) -> None:
        integer_fields = (
            self.component_index,
            self.minimum_source_face_index,
            self.source_face_count,
            self.source_edge_count,
            self.source_same_direction_edge_count,
            self.source_consistent_winding_sign_to_positive_volume,
            self.positive_volume_winding_flip_count,
        )
        if any(type(value) is not int for value in integer_fields):
            raise TypeError("component integer fields must be Python integers")
        if self.component_index < 0 or self.minimum_source_face_index < 0:
            raise ValueError("component and source-face indices cannot be negative")
        if self.source_face_count < 1 or self.source_edge_count < 1:
            raise ValueError("component face and edge counts must be positive")
        if not 0 <= self.source_same_direction_edge_count <= self.source_edge_count:
            raise ValueError("invalid same-direction source edge count")
        if type(self.source_winding_consistent) is not bool:
            raise TypeError("source_winding_consistent must be bool")
        expected_consistency = self.source_same_direction_edge_count == 0
        if self.source_winding_consistent is not expected_consistency:
            raise ValueError("source winding status contradicts source edge directions")
        if self.source_consistent_winding_sign_to_positive_volume not in (-1, 0, 1):
            raise ValueError("source winding sign must belong to {-1, 0, 1}")
        if self.source_winding_consistent:
            if self.source_consistent_winding_sign_to_positive_volume == 0:
                raise ValueError(
                    "consistent source winding must expose its global sign"
                )
        elif self.source_consistent_winding_sign_to_positive_volume != 0:
            raise ValueError(
                "inconsistent source winding cannot expose one global sign"
            )
        if not 0 <= self.positive_volume_winding_flip_count <= self.source_face_count:
            raise ValueError("invalid positive-volume winding flip count")
        if self.positive_signed_six_volume.sign != 1:
            raise ValueError("component signed six-volume must be strictly positive")
        for digest in (
            self.source_face_indices_sha256,
            self.positive_volume_winding_sha256,
        ):
            if not _is_sha256(digest):
                raise ValueError("component digest must be lowercase SHA-256")


@dataclass(frozen=True)
class SurfaceOrientationCertificate:
    """Immutable V1 certificate and source-face winding maps.

    ``positive_volume_winding_sign_by_source_face[i]`` is ``+1`` when source
    face ``i`` is retained and ``-1`` when its last two indices must be
    exchanged.  The map establishes positive component volumes only; the
    explicit V1 claim boundary below prevents it from being promoted to a
    formal material-outward certificate.
    """

    method_id: str
    role: SurfaceBoundaryRole
    source_vertex_count: int
    source_face_count: int
    source_edge_count: int
    component_count: int
    boundary_edge_count: int
    non_manifold_edge_count: int
    winding_constraint_contradiction_count: int
    source_same_direction_edge_count: int
    components: tuple[SurfaceOrientationComponentRecord, ...]
    canonical_component_index_by_source_face: tuple[int, ...]
    positive_volume_winding_sign_by_source_face: tuple[int, ...]
    source_indexed_mesh_sha256: str
    canonical_sha256: str
    source_index_topology_status: str
    closed_two_manifold_status: str
    local_orientability_status: str
    exact_dyadic_component_volume_status: str
    component_positive_volume_orientation_status: str
    self_intersection_status: str
    nesting_parity_status: str
    formal_outward_eligible: bool
    formal_outward_ineligibility_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.method_id != METHOD_ID:
            raise ValueError("unexpected surface orientation method_id")
        if not isinstance(self.role, SurfaceBoundaryRole):
            raise TypeError("role must be a SurfaceBoundaryRole")
        count_fields = (
            self.source_vertex_count,
            self.source_face_count,
            self.source_edge_count,
            self.component_count,
            self.boundary_edge_count,
            self.non_manifold_edge_count,
            self.winding_constraint_contradiction_count,
            self.source_same_direction_edge_count,
        )
        if any(type(value) is not int or value < 0 for value in count_fields):
            raise ValueError("certificate counts must be nonnegative Python integers")
        if self.source_vertex_count < 3 or self.source_face_count < 1:
            raise ValueError("certificate needs a non-empty triangle mesh")
        if self.source_edge_count < 1 or self.component_count < 1:
            raise ValueError("certificate needs at least one edge and component")
        if self.boundary_edge_count != 0 or self.non_manifold_edge_count != 0:
            raise ValueError(
                "a certificate cannot contain an open or non-manifold edge"
            )
        if self.winding_constraint_contradiction_count != 0:
            raise ValueError("a certificate cannot contain a winding contradiction")
        if len(self.components) != self.component_count:
            raise ValueError("component tuple length contradicts component_count")
        if tuple(record.component_index for record in self.components) != tuple(
            range(self.component_count)
        ):
            raise ValueError("component records must use dense canonical indices")
        if (
            sum(record.source_face_count for record in self.components)
            != self.source_face_count
        ):
            raise ValueError("component face counts do not cover the source faces")
        if (
            sum(record.source_edge_count for record in self.components)
            != self.source_edge_count
        ):
            raise ValueError("component edge counts do not cover the source edges")
        if (
            sum(record.source_same_direction_edge_count for record in self.components)
            != self.source_same_direction_edge_count
        ):
            raise ValueError(
                "component source winding counts contradict the certificate"
            )
        if (
            len(self.canonical_component_index_by_source_face)
            != self.source_face_count
        ):
            raise ValueError(
                "component index map must contain one entry per source face"
            )
        if any(
            value < 0 or value >= self.component_count
            for value in self.canonical_component_index_by_source_face
        ):
            raise ValueError("component index map contains an invalid component")
        if (
            len(self.positive_volume_winding_sign_by_source_face)
            != self.source_face_count
        ):
            raise ValueError("winding map must contain one entry per source face")
        if any(
            value not in (-1, 1)
            for value in self.positive_volume_winding_sign_by_source_face
        ):
            raise ValueError("source-face winding signs must belong to {-1, +1}")
        for digest in (self.source_indexed_mesh_sha256, self.canonical_sha256):
            if not _is_sha256(digest):
                raise ValueError("certificate digest must be lowercase SHA-256")
        verified_fields = (
            self.source_index_topology_status,
            self.closed_two_manifold_status,
            self.local_orientability_status,
            self.exact_dyadic_component_volume_status,
            self.component_positive_volume_orientation_status,
        )
        if any(value != VERIFIED for value in verified_fields):
            raise ValueError("successful V1 certificate statuses must be VERIFIED")
        if self.self_intersection_status != UNVERIFIED:
            raise ValueError("V1 self-intersection status must remain UNVERIFIED")
        if self.nesting_parity_status != UNVERIFIED:
            raise ValueError("V1 nesting parity status must remain UNVERIFIED")
        if (
            type(self.formal_outward_eligible) is not bool
            or self.formal_outward_eligible
        ):
            raise ValueError("V1 cannot claim formal material-outward eligibility")
        expected_reasons = (
            "SELF_INTERSECTION_UNVERIFIED",
            "NESTING_PARITY_UNVERIFIED",
        )
        if self.formal_outward_ineligibility_reasons != expected_reasons:
            raise ValueError("V1 formal-outward ineligibility reasons are incomplete")


class _ParityDisjointSet:
    """Union-find with XOR winding constraints between source faces."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size
        self.parity_to_parent = [0] * size

    def find(self, item: int) -> tuple[int, int]:
        root = item
        parity = 0
        while self.parent[root] != root:
            parity ^= self.parity_to_parent[root]
            root = self.parent[root]

        cursor = item
        prefix = 0
        while self.parent[cursor] != cursor:
            parent = self.parent[cursor]
            edge_parity = self.parity_to_parent[cursor]
            self.parent[cursor] = root
            self.parity_to_parent[cursor] = parity ^ prefix
            prefix ^= edge_parity
            cursor = parent
        return root, parity

    def constrain(self, first: int, second: int, required_xor: int) -> bool:
        first_root, first_parity = self.find(first)
        second_root, second_parity = self.find(second)
        if first_root == second_root:
            return (first_parity ^ second_parity) == required_xor
        root_relation = first_parity ^ second_parity ^ required_xor
        if self.rank[first_root] < self.rank[second_root]:
            self.parent[first_root] = second_root
            self.parity_to_parent[first_root] = root_relation
        else:
            self.parent[second_root] = first_root
            self.parity_to_parent[second_root] = root_relation
            if self.rank[first_root] == self.rank[second_root]:
                self.rank[first_root] += 1
        return True


@dataclass
class _EdgeIncidence:
    first_face_index: int
    first_direction: int
    incidence_count: int = 1
    second_direction: int = 0


@dataclass
class _DyadicAccumulator:
    numerator: int = 0
    binary_exponent: int = 0

    def add(self, numerator: int, binary_exponent: int) -> None:
        if numerator == 0:
            return
        if self.numerator == 0:
            self.numerator = numerator
            self.binary_exponent = binary_exponent
            return
        if binary_exponent < self.binary_exponent:
            self.numerator = (
                self.numerator << (self.binary_exponent - binary_exponent)
            ) + numerator
            self.binary_exponent = binary_exponent
        else:
            self.numerator += numerator << (binary_exponent - self.binary_exponent)

    def canonical(self) -> ExactDyadic:
        numerator, exponent = _normalize_dyadic(
            self.numerator, self.binary_exponent
        )
        return ExactDyadic(numerator=numerator, binary_exponent=exponent)


def _normalize_dyadic(numerator: int, exponent: int) -> tuple[int, int]:
    if numerator == 0:
        return 0, 0
    magnitude = abs(numerator)
    trailing_zero_count = (magnitude & -magnitude).bit_length() - 1
    return numerator >> trailing_zero_count, exponent + trailing_zero_count


def _dyadic_add(
    first: tuple[int, int], second: tuple[int, int], *, negate_second: bool = False
) -> tuple[int, int]:
    first_numerator, first_exponent = first
    second_numerator, second_exponent = second
    if negate_second:
        second_numerator = -second_numerator
    if first_numerator == 0:
        return _normalize_dyadic(second_numerator, second_exponent)
    if second_numerator == 0:
        return _normalize_dyadic(first_numerator, first_exponent)
    exponent = min(first_exponent, second_exponent)
    numerator = (
        (first_numerator << (first_exponent - exponent))
        + (second_numerator << (second_exponent - exponent))
    )
    return _normalize_dyadic(numerator, exponent)


def _dyadic_multiply(
    first: tuple[int, int], second: tuple[int, int]
) -> tuple[int, int]:
    return _normalize_dyadic(
        first[0] * second[0], first[1] + second[1]
    )


def _binary64_dyadic_arrays(vertices_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return exact integer significands and binary exponents for float64 data."""

    contiguous = np.ascontiguousarray(vertices_m, dtype=np.float64)
    bit_rows = contiguous.view(np.uint64).reshape(contiguous.shape)
    significands = np.empty(contiguous.shape, dtype=np.int64)
    binary_exponents = np.empty(contiguous.shape, dtype=np.int16)
    for flat_index, raw_value in enumerate(bit_rows.reshape(-1)):
        bits = int(raw_value)
        sign_negative = bool(bits >> 63)
        exponent_bits = (bits >> 52) & 0x7FF
        fraction_bits = bits & ((1 << 52) - 1)
        if exponent_bits == 0:
            significand = fraction_bits
            exponent = -1074 if significand else 0
        else:
            significand = (1 << 52) | fraction_bits
            exponent = int(exponent_bits) - 1023 - 52
        if sign_negative:
            significand = -significand
        row, column = divmod(flat_index, 3)
        significands[row, column] = significand
        binary_exponents[row, column] = exponent
    return significands, binary_exponents


def _coordinate(
    significands: np.ndarray,
    exponents: np.ndarray,
    vertex_index: int,
    axis: int,
) -> tuple[int, int]:
    return int(significands[vertex_index, axis]), int(exponents[vertex_index, axis])


def _face_has_exact_nonzero_area(
    face: tuple[int, int, int],
    significands: np.ndarray,
    exponents: np.ndarray,
) -> bool:
    first, second, third = face
    first_point = tuple(
        _coordinate(significands, exponents, first, axis) for axis in range(3)
    )
    second_point = tuple(
        _coordinate(significands, exponents, second, axis) for axis in range(3)
    )
    third_point = tuple(
        _coordinate(significands, exponents, third, axis) for axis in range(3)
    )
    first_edge = tuple(
        _dyadic_add(second_point[axis], first_point[axis], negate_second=True)
        for axis in range(3)
    )
    second_edge = tuple(
        _dyadic_add(third_point[axis], first_point[axis], negate_second=True)
        for axis in range(3)
    )
    cross = (
        _dyadic_add(
            _dyadic_multiply(first_edge[1], second_edge[2]),
            _dyadic_multiply(first_edge[2], second_edge[1]),
            negate_second=True,
        ),
        _dyadic_add(
            _dyadic_multiply(first_edge[2], second_edge[0]),
            _dyadic_multiply(first_edge[0], second_edge[2]),
            negate_second=True,
        ),
        _dyadic_add(
            _dyadic_multiply(first_edge[0], second_edge[1]),
            _dyadic_multiply(first_edge[1], second_edge[0]),
            negate_second=True,
        ),
    )
    return any(component[0] != 0 for component in cross)


def _accumulate_face_signed_six_volume(
    accumulator: _DyadicAccumulator,
    face: tuple[int, int, int],
    orientation_sign: int,
    significands: np.ndarray,
    exponents: np.ndarray,
) -> None:
    first, second, third = face
    terms = (
        (first, 0, second, 1, third, 2, 1),
        (first, 1, second, 2, third, 0, 1),
        (first, 2, second, 0, third, 1, 1),
        (first, 2, second, 1, third, 0, -1),
        (first, 1, second, 0, third, 2, -1),
        (first, 0, second, 2, third, 1, -1),
    )
    for (
        first_vertex,
        first_axis,
        second_vertex,
        second_axis,
        third_vertex,
        third_axis,
        determinant_sign,
    ) in terms:
        first_numerator = int(significands[first_vertex, first_axis])
        second_numerator = int(significands[second_vertex, second_axis])
        third_numerator = int(significands[third_vertex, third_axis])
        numerator = first_numerator * second_numerator * third_numerator
        if numerator == 0:
            continue
        exponent = (
            int(exponents[first_vertex, first_axis])
            + int(exponents[second_vertex, second_axis])
            + int(exponents[third_vertex, third_axis])
        )
        accumulator.add(
            orientation_sign * determinant_sign * numerator,
            exponent,
        )


def _update_sized_bytes(digest: Any, payload: bytes) -> None:
    digest.update(struct.pack("<Q", len(payload)))
    digest.update(payload)


def _source_mesh_sha256(vertices_m: np.ndarray, faces: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(b"CARTS_GRASP_SOURCE_INDEXED_BINARY64_MESH_V1\0")
    for array, dtype in ((vertices_m, "<f8"), (faces, "<i8")):
        canonical = np.ascontiguousarray(array, dtype=dtype)
        digest.update(struct.pack("<Q", canonical.ndim))
        digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes(order="C"))
        _update_sized_bytes(digest, canonical.tobytes(order="C"))
    return digest.hexdigest()


def _canonical_certificate_sha256(
    *,
    role: SurfaceBoundaryRole,
    source_mesh_sha256: str,
    components: tuple[SurfaceOrientationComponentRecord, ...],
    component_index_by_face: tuple[int, ...],
    winding_sign_by_face: tuple[int, ...],
) -> str:
    digest = hashlib.sha256()
    digest.update(b"CARTS_GRASP_SURFACE_ORIENTATION_CERTIFICATE_CANONICAL_V1\0")
    _update_sized_bytes(digest, METHOD_ID.encode("ascii"))
    _update_sized_bytes(digest, role.value.encode("ascii"))
    digest.update(bytes.fromhex(source_mesh_sha256))
    digest.update(struct.pack("<Q", len(components)))
    for record in components:
        for value in (
            record.component_index,
            record.minimum_source_face_index,
            record.source_face_count,
            record.source_edge_count,
            record.source_same_direction_edge_count,
            int(record.source_winding_consistent),
            record.source_consistent_winding_sign_to_positive_volume,
            record.positive_volume_winding_flip_count,
            record.positive_signed_six_volume.binary_exponent,
        ):
            _update_sized_bytes(digest, str(value).encode("ascii"))
        _update_sized_bytes(
            digest,
            str(record.positive_signed_six_volume.numerator).encode("ascii"),
        )
        digest.update(bytes.fromhex(record.source_face_indices_sha256))
        digest.update(bytes.fromhex(record.positive_volume_winding_sha256))
    component_array = np.asarray(component_index_by_face, dtype="<i8")
    winding_array = np.asarray(winding_sign_by_face, dtype=np.int8)
    _update_sized_bytes(digest, component_array.tobytes(order="C"))
    _update_sized_bytes(digest, winding_array.tobytes(order="C"))
    for status in (
        VERIFIED,
        VERIFIED,
        VERIFIED,
        VERIFIED,
        VERIFIED,
        UNVERIFIED,
        UNVERIFIED,
    ):
        _update_sized_bytes(digest, status.encode("ascii"))
    digest.update(b"\x00")
    _update_sized_bytes(digest, b"SELF_INTERSECTION_UNVERIFIED")
    _update_sized_bytes(digest, b"NESTING_PARITY_UNVERIFIED")
    return digest.hexdigest()


def _validated_source_arrays(
    vertices_m: Sequence[Sequence[float]] | np.ndarray,
    faces: Sequence[Sequence[int]] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    raw_vertices = np.asarray(vertices_m)
    if raw_vertices.dtype != np.dtype(np.float64):
        raise SurfaceOrientationAuditError(
            "BINARY64_VERTICES_REQUIRED",
            f"vertices_m dtype must be float64, got {raw_vertices.dtype}",
        )
    vertices = np.array(raw_vertices, dtype=np.float64, order="C", copy=True)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) < 3:
        raise SurfaceOrientationAuditError(
            "INVALID_VERTEX_ARRAY", "vertices_m must have finite shape (V, 3), V >= 3"
        )
    if not np.all(np.isfinite(vertices)):
        raise SurfaceOrientationAuditError(
            "NONFINITE_VERTEX", "vertices_m contains NaN or infinity"
        )

    raw_faces = np.asarray(faces)
    if raw_faces.dtype.kind not in "iu":
        raise SurfaceOrientationAuditError(
            "INTEGER_FACES_REQUIRED",
            f"faces dtype must be integer, got {raw_faces.dtype}",
        )
    if raw_faces.ndim != 2 or raw_faces.shape[1:] != (3,) or len(raw_faces) < 1:
        raise SurfaceOrientationAuditError(
            "INVALID_FACE_ARRAY", "faces must have integer shape (F, 3), F >= 1"
        )
    if raw_faces.dtype.kind == "i" and int(np.min(raw_faces)) < 0:
        raise SurfaceOrientationAuditError(
            "FACE_INDEX_OUT_OF_RANGE", "faces contains a negative source vertex index"
        )
    maximum_index = int(np.max(raw_faces))
    if maximum_index >= len(vertices):
        raise SurfaceOrientationAuditError(
            "FACE_INDEX_OUT_OF_RANGE",
            f"maximum source vertex index {maximum_index} exceeds V={len(vertices)}",
        )
    if maximum_index > np.iinfo(np.int64).max:
        raise SurfaceOrientationAuditError(
            "FACE_INDEX_OUT_OF_RANGE", "source vertex index exceeds int64"
        )
    indexed_faces = np.array(raw_faces, dtype=np.int64, order="C", copy=True)
    repeated = np.flatnonzero(
        (indexed_faces[:, 0] == indexed_faces[:, 1])
        | (indexed_faces[:, 1] == indexed_faces[:, 2])
        | (indexed_faces[:, 2] == indexed_faces[:, 0])
    )
    if len(repeated):
        raise SurfaceOrientationAuditError(
            "REPEATED_FACE_VERTEX_INDEX",
            f"source face {int(repeated[0])} repeats a vertex index",
        )
    vertices.setflags(write=False)
    indexed_faces.setflags(write=False)
    return vertices, indexed_faces


def audit_surface_orientation(
    vertices_m: Sequence[Sequence[float]] | np.ndarray,
    faces: Sequence[Sequence[int]] | np.ndarray,
    *,
    role: SurfaceBoundaryRole,
) -> SurfaceOrientationCertificate:
    """Audit and orient a source-indexed binary64 triangular surface.

    Runtime and storage are expected ``O(F + E + V)`` under the standard
    expected-constant-time hash-table model.  In particular, component
    statistics are accumulated in single face and edge passes; no component
    rescans the complete edge collection.

    V1 rejects open, non-manifold, non-orientable, exactly degenerate, and
    exact-zero-volume inputs.  A successful return still does *not* certify a
    formal material-outward normal because self-intersection and nesting parity
    remain explicitly unverified.
    """

    if not isinstance(role, SurfaceBoundaryRole):
        raise SurfaceOrientationAuditError(
            "EXPLICIT_BOUNDARY_ROLE_REQUIRED",
            "role must be a SurfaceBoundaryRole enum value",
        )
    vertices, indexed_faces = _validated_source_arrays(vertices_m, faces)
    face_count = len(indexed_faces)
    disjoint_set = _ParityDisjointSet(face_count)
    incidence: dict[tuple[int, int], _EdgeIncidence] = {}
    winding_contradiction_edges: list[tuple[int, int]] = []

    for face_index, raw_face in enumerate(indexed_faces):
        first, second, third = (int(value) for value in raw_face)
        for start, end in ((first, second), (second, third), (third, first)):
            if start < end:
                edge = (start, end)
                direction = 1
            else:
                edge = (end, start)
                direction = -1
            row = incidence.get(edge)
            if row is None:
                incidence[edge] = _EdgeIncidence(face_index, direction)
                continue
            row.incidence_count += 1
            if row.incidence_count == 2:
                row.second_direction = direction
                required_xor = int(row.first_direction == direction)
                if not disjoint_set.constrain(
                    row.first_face_index, face_index, required_xor
                ):
                    winding_contradiction_edges.append(edge)

    boundary_edges = [
        edge for edge, row in incidence.items() if row.incidence_count == 1
    ]
    non_manifold_edges = [
        edge for edge, row in incidence.items() if row.incidence_count > 2
    ]
    if boundary_edges or non_manifold_edges:
        detail = (
            f"boundary_edge_count={len(boundary_edges)}, "
            f"non_manifold_edge_count={len(non_manifold_edges)}"
        )
        if non_manifold_edges:
            detail += f", first_non_manifold_edge={non_manifold_edges[0]}"
            code = "NON_MANIFOLD_SOURCE_INDEX_TOPOLOGY"
        else:
            detail += f", first_boundary_edge={boundary_edges[0]}"
            code = "OPEN_SOURCE_INDEX_TOPOLOGY"
        raise SurfaceOrientationAuditError(code, detail)
    if winding_contradiction_edges:
        contradiction_detail = (
            f"count={len(winding_contradiction_edges)}, "
            f"first_edge={winding_contradiction_edges[0]}"
        )
        raise SurfaceOrientationAuditError(
            "WINDING_CONSTRAINT_CONTRADICTION",
            contradiction_detail,
        )

    roots: list[int] = []
    parity_by_face: list[int] = []
    minimum_face_by_root: dict[int, int] = {}
    face_count_by_root: dict[int, int] = {}
    for face_index in range(face_count):
        root, parity = disjoint_set.find(face_index)
        roots.append(root)
        parity_by_face.append(parity)
        minimum_face_by_root[root] = min(
            face_index, minimum_face_by_root.get(root, face_index)
        )
        face_count_by_root[root] = face_count_by_root.get(root, 0) + 1
    ordered_roots = tuple(
        sorted(minimum_face_by_root, key=lambda root: minimum_face_by_root[root])
    )
    if (
        role is SurfaceBoundaryRole.SINGLE_CLOSED_BOUNDARY
        and len(ordered_roots) != 1
    ):
        raise SurfaceOrientationAuditError(
            "BOUNDARY_ROLE_COMPONENT_COUNT_MISMATCH",
            "SINGLE_CLOSED_BOUNDARY requires one component, "
            f"found {len(ordered_roots)}",
        )
    component_index_by_root = {
        root: component_index for component_index, root in enumerate(ordered_roots)
    }

    edge_count_by_root = {root: 0 for root in ordered_roots}
    same_direction_edge_count_by_root = {root: 0 for root in ordered_roots}
    for row in incidence.values():
        root = roots[row.first_face_index]
        edge_count_by_root[root] += 1
        if row.first_direction == row.second_direction:
            same_direction_edge_count_by_root[root] += 1

    significands, binary_exponents = _binary64_dyadic_arrays(vertices)
    volume_by_root = {root: _DyadicAccumulator() for root in ordered_roots}
    for face_index, raw_face in enumerate(indexed_faces):
        face = tuple(int(value) for value in raw_face)
        if not _face_has_exact_nonzero_area(
            face, significands, binary_exponents
        ):
            raise SurfaceOrientationAuditError(
                "EXACTLY_DEGENERATE_SOURCE_FACE",
                f"source face {face_index} has exact zero area",
            )
        local_orientation_sign = -1 if parity_by_face[face_index] else 1
        _accumulate_face_signed_six_volume(
            volume_by_root[roots[face_index]],
            face,
            local_orientation_sign,
            significands,
            binary_exponents,
        )

    exact_volume_by_root: dict[int, ExactDyadic] = {}
    global_sign_by_root: dict[int, int] = {}
    for root in ordered_roots:
        exact_volume = volume_by_root[root].canonical()
        if exact_volume.sign == 0:
            minimum_face = minimum_face_by_root[root]
            raise SurfaceOrientationAuditError(
                "EXACT_ZERO_COMPONENT_SIGNED_VOLUME",
                f"component rooted at source face {minimum_face} has exact zero "
                "signed six-volume",
            )
        global_sign = exact_volume.sign
        global_sign_by_root[root] = global_sign
        exact_volume_by_root[root] = ExactDyadic(
            numerator=abs(exact_volume.numerator),
            binary_exponent=exact_volume.binary_exponent,
        )

    component_index_by_face = tuple(
        component_index_by_root[root] for root in roots
    )
    winding_sign_by_face = tuple(
        (-1 if parity else 1) * global_sign_by_root[root]
        for root, parity in zip(roots, parity_by_face)
    )

    face_index_hashers = {root: hashlib.sha256() for root in ordered_roots}
    winding_hashers = {root: hashlib.sha256() for root in ordered_roots}
    flip_count_by_root = {root: 0 for root in ordered_roots}
    for face_index, (root, winding_sign) in enumerate(
        zip(roots, winding_sign_by_face)
    ):
        face_index_hashers[root].update(struct.pack("<Q", face_index))
        winding_hashers[root].update(struct.pack("<Qb", face_index, winding_sign))
        if winding_sign == -1:
            flip_count_by_root[root] += 1

    component_records: list[SurfaceOrientationComponentRecord] = []
    for component_index, root in enumerate(ordered_roots):
        same_direction_count = same_direction_edge_count_by_root[root]
        source_consistent = same_direction_count == 0
        flip_count = flip_count_by_root[root]
        if source_consistent:
            source_sign = -1 if flip_count == face_count_by_root[root] else 1
            if flip_count not in (0, face_count_by_root[root]):
                raise SurfaceOrientationAuditError(
                    "INTERNAL_WINDING_INVARIANT_FAILURE",
                    "consistent source component produced a non-global correction",
                )
        else:
            source_sign = 0
        component_records.append(
            SurfaceOrientationComponentRecord(
                component_index=component_index,
                minimum_source_face_index=minimum_face_by_root[root],
                source_face_count=face_count_by_root[root],
                source_edge_count=edge_count_by_root[root],
                source_same_direction_edge_count=same_direction_count,
                source_winding_consistent=source_consistent,
                source_consistent_winding_sign_to_positive_volume=source_sign,
                positive_volume_winding_flip_count=flip_count,
                positive_signed_six_volume=exact_volume_by_root[root],
                source_face_indices_sha256=face_index_hashers[root].hexdigest(),
                positive_volume_winding_sha256=winding_hashers[root].hexdigest(),
            )
        )
    components = tuple(component_records)
    source_mesh_digest = _source_mesh_sha256(vertices, indexed_faces)
    canonical_digest = _canonical_certificate_sha256(
        role=role,
        source_mesh_sha256=source_mesh_digest,
        components=components,
        component_index_by_face=component_index_by_face,
        winding_sign_by_face=winding_sign_by_face,
    )
    return SurfaceOrientationCertificate(
        method_id=METHOD_ID,
        role=role,
        source_vertex_count=len(vertices),
        source_face_count=face_count,
        source_edge_count=len(incidence),
        component_count=len(components),
        boundary_edge_count=0,
        non_manifold_edge_count=0,
        winding_constraint_contradiction_count=0,
        source_same_direction_edge_count=sum(
            same_direction_edge_count_by_root.values()
        ),
        components=components,
        canonical_component_index_by_source_face=component_index_by_face,
        positive_volume_winding_sign_by_source_face=winding_sign_by_face,
        source_indexed_mesh_sha256=source_mesh_digest,
        canonical_sha256=canonical_digest,
        source_index_topology_status=VERIFIED,
        closed_two_manifold_status=VERIFIED,
        local_orientability_status=VERIFIED,
        exact_dyadic_component_volume_status=VERIFIED,
        component_positive_volume_orientation_status=VERIFIED,
        self_intersection_status=UNVERIFIED,
        nesting_parity_status=UNVERIFIED,
        formal_outward_eligible=False,
        formal_outward_ineligibility_reasons=(
            "SELF_INTERSECTION_UNVERIFIED",
            "NESTING_PARITY_UNVERIFIED",
        ),
    )


__all__ = [
    "ExactDyadic",
    "METHOD_ID",
    "SurfaceBoundaryRole",
    "SurfaceOrientationAuditError",
    "SurfaceOrientationCertificate",
    "SurfaceOrientationComponentRecord",
    "UNVERIFIED",
    "VERIFIED",
    "audit_surface_orientation",
]
