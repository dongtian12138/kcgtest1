"""Fail-closed asset and pair-policy contract for CARTS collision proofs.

This module does not perform collision detection.  It binds immutable mesh
inputs and constructs the exhaustive pair inventory that a later continuous
collision certificate must cover.  Triangle roles are assigned only by exact
IEEE-754 geometry identity and multiset occurrence; no spatial tolerance,
nearest-neighbour match, or legacy contact-role score is accepted.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
import hashlib
import itertools
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence
import xml.etree.ElementTree as ET

import numpy as np

from kcg_connector.grasp.robust.object_model import (
    file_sha256,
    load_stl_mesh,
)


METHOD_ID = "CARTS_COLLISION_ASSET_AND_PAIR_CONTRACT_V1"
TRIANGLE_IDENTITY_POLICY = (
    "CANONICAL_UNORIENTED_LITTLE_ENDIAN_BINARY64_VERTICES_"
    "WITH_SIGNED_ZERO_PRESERVED_X_EXACT_OCCURRENCE_INDEX_X_"
    "CYCLIC_WINDING_IDENTITY"
)
TERMINAL_PARTITION_POLICY = (
    "EXACT_TRIANGLE_MULTISET_INTERSECTION_WITH_AMBIGUOUS_GEOMETRY_"
    "FORBIDDEN_DOMINATES_AND_ORPHAN_PAD_FACES_QUARANTINED"
)
SRDF_POLICY = (
    "DISABLED_COLLISION_ROWS_ARE_UNTRUSTED_METADATA_X_ALL_LINK_PAIRS_"
    "RESTART_FORBIDDEN_UNTIL_INDEPENDENT_PROOF"
)


class CollisionContractError(ValueError):
    """Raised when a collision proof input is absent or ambiguous."""


class CoverageMode(str, Enum):
    AUTHORED_VISUAL_SURFACE = "AUTHORED_VISUAL_SURFACE"
    DECLARED_CLOSED_BOUNDARY_UNVERIFIED = (
        "DECLARED_CLOSED_BOUNDARY_UNVERIFIED"
    )
    CONSERVATIVE_CONVEX_COVER_UNVERIFIED = (
        "CONSERVATIVE_CONVEX_COVER_UNVERIFIED"
    )
    SYNTHETIC_ARRAY_FIXTURE = "SYNTHETIC_ARRAY_FIXTURE"
    PROXY_AUDIT_ONLY = "PROXY_AUDIT_ONLY"

    @property
    def formal_solid_coverage(self) -> bool:
        """No V1 label is a validated solid-coverage certificate."""

        return False


class PairPolicy(str, Enum):
    FORBIDDEN = "FORBIDDEN"
    FIRST_CONTACT_ENDPOINT_ONLY = "FIRST_CONTACT_ENDPOINT_ONLY"
    PERSISTENT_GRASP_CONTACT = "PERSISTENT_GRASP_CONTACT"
    SUPPORT_CONTACT_AT_SEGMENT_START_ONLY = (
        "SUPPORT_CONTACT_AT_SEGMENT_START_ONLY"
    )
    STRUCTURAL_INTERFACE_PROVEN = "STRUCTURAL_INTERFACE_PROVEN"


@dataclass(frozen=True, order=True)
class TriangleInstanceKey:
    source_mesh_sha256: str
    canonical_geometry_sha256: str
    occurrence_index: int

    def __post_init__(self) -> None:
        for name, value in (
            ("source_mesh_sha256", self.source_mesh_sha256),
            ("canonical_geometry_sha256", self.canonical_geometry_sha256),
        ):
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise CollisionContractError(
                    f"{name} must be lowercase SHA-256"
                )
        if (
            not isinstance(self.occurrence_index, int)
            or isinstance(self.occurrence_index, bool)
            or self.occurrence_index < 0
        ):
            raise CollisionContractError(
                "triangle occurrence_index must be a nonnegative integer"
            )


@dataclass(frozen=True)
class VerifiedCollisionMesh:
    asset_id: str
    link_name: str
    path: Path
    sha256: str
    unit: str
    local_transform: np.ndarray
    coverage_mode: CoverageMode

    def __post_init__(self) -> None:
        if not isinstance(self.coverage_mode, CoverageMode):
            raise CollisionContractError(
                "collision mesh coverage_mode must be a registered enum"
            )
        if not self.asset_id or not self.link_name:
            raise CollisionContractError(
                "collision mesh asset_id and link_name must be non-empty"
            )
        path = Path(self.path).resolve()
        if not path.is_file():
            raise CollisionContractError(
                f"collision mesh is unavailable: {path}"
            )
        if file_sha256(path) != self.sha256:
            raise CollisionContractError(
                f"collision mesh SHA-256 mismatch: {path}"
            )
        if self.unit not in {"m", "mm"}:
            raise CollisionContractError(
                "collision mesh unit must be explicitly m or mm"
            )
        transform = np.array(self.local_transform, dtype=np.float64, copy=True)
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise CollisionContractError(
                "collision mesh local_transform must be finite 4x4"
            )
        if not np.array_equal(transform[3], (0.0, 0.0, 0.0, 1.0)):
            raise CollisionContractError(
                "collision mesh local_transform must be homogeneous"
            )
        rotation = transform[:3, :3]
        with np.errstate(over="ignore", invalid="ignore"):
            gram = rotation.T @ rotation
            determinant = float(np.linalg.det(rotation))
        if not np.all(np.isfinite(gram)) or not math.isfinite(determinant):
            raise CollisionContractError(
                "collision mesh local_transform rotation overflowed"
            )
        scale = max(1.0, float(np.max(np.abs(rotation))))
        numerical_bound = (
            256.0 * np.finfo(np.float64).eps * scale * scale
        )
        if (
            float(np.linalg.norm(gram - np.eye(3), ord=2))
            > numerical_bound
            or abs(determinant - 1.0) > numerical_bound
        ):
            raise CollisionContractError(
                "collision mesh local_transform rotation is not proper"
            )
        transform.setflags(write=False)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "local_transform", transform)


@dataclass(frozen=True, init=False)
class TerminalTrianglePartition:
    source_mesh: VerifiedCollisionMesh
    pad_source_path: Path
    pad_source_sha256: str
    input_provenance: str
    pad_allowed: tuple[TriangleInstanceKey, ...]
    nonpad_forbidden: tuple[TriangleInstanceKey, ...]
    orphan_pad_face_count: int
    ambiguous_source_face_count: int
    same_winding_match_count: int
    winding_mismatch_face_count: int
    orphan_pad_face_indices: tuple[int, ...]
    winding_mismatch_pad_face_indices: tuple[int, ...]
    source_face_count: int
    pad_face_count: int
    exact_cover_verified: bool
    complete_pad_identity_verified: bool
    formal_collision_eligible: bool
    partition_sha256: str

    def __init__(self) -> None:
        raise TypeError(
            "TerminalTrianglePartition is created only by verified factories"
        )

    def __post_init__(self) -> None:
        pad_path = Path(self.pad_source_path).resolve()
        if not pad_path.is_file() or file_sha256(pad_path) != (
            self.pad_source_sha256
        ):
            raise CollisionContractError(
                "terminal partition PAD source is not hash-bound"
            )
        if self.input_provenance not in {
            "HASH_BOUND_SOURCE_STL_AND_PAD_NPZ",
            "SYNTHETIC_ARRAY_FIXTURE_NOT_FILE_LINEAGE",
        }:
            raise CollisionContractError(
                "terminal partition input provenance is unregistered"
            )
        integer_fields = (
            self.orphan_pad_face_count,
            self.ambiguous_source_face_count,
            self.same_winding_match_count,
            self.winding_mismatch_face_count,
            self.source_face_count,
            self.pad_face_count,
        )
        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            for value in integer_fields
        ):
            raise CollisionContractError(
                "terminal partition counts must be nonnegative integers"
            )
        if len(self.pad_allowed) + len(self.nonpad_forbidden) != (
            self.source_face_count
        ):
            raise CollisionContractError(
                "terminal partition does not cover every source face"
            )
        allowed = set(self.pad_allowed)
        forbidden = set(self.nonpad_forbidden)
        if len(allowed) != len(self.pad_allowed):
            raise CollisionContractError(
                "terminal allowed triangle instances are duplicated"
            )
        if len(forbidden) != len(self.nonpad_forbidden):
            raise CollisionContractError(
                "terminal forbidden triangle instances are duplicated"
            )
        if allowed & forbidden:
            raise CollisionContractError(
                "terminal triangle instance has conflicting roles"
            )
        all_rows = self.pad_allowed + self.nonpad_forbidden
        if any(
            row.source_mesh_sha256 != self.source_mesh.sha256
            for row in all_rows
        ):
            raise CollisionContractError(
                "terminal triangle key has the wrong source mesh hash"
            )
        occurrence_rows: defaultdict[str, list[int]] = defaultdict(list)
        for row in all_rows:
            occurrence_rows[row.canonical_geometry_sha256].append(
                row.occurrence_index
            )
        if any(
            sorted(rows) != list(range(len(rows)))
            for rows in occurrence_rows.values()
        ):
            raise CollisionContractError(
                "terminal triangle occurrences are not contiguous"
            )
        expected_orphans = tuple(sorted(self.orphan_pad_face_indices))
        expected_mismatches = tuple(
            sorted(self.winding_mismatch_pad_face_indices)
        )
        if (
            self.orphan_pad_face_indices != expected_orphans
            or len(set(expected_orphans)) != len(expected_orphans)
            or len(expected_orphans) != self.orphan_pad_face_count
            or any(
                index < 0 or index >= self.pad_face_count
                for index in expected_orphans
            )
        ):
            raise CollisionContractError(
                "terminal orphan PAD face indices are inconsistent"
            )
        if (
            self.winding_mismatch_pad_face_indices != expected_mismatches
            or len(set(expected_mismatches)) != len(expected_mismatches)
            or len(expected_mismatches) != self.winding_mismatch_face_count
            or any(
                index < 0 or index >= self.pad_face_count
                for index in expected_mismatches
            )
            or set(expected_orphans) & set(expected_mismatches)
        ):
            raise CollisionContractError(
                "terminal winding-mismatch PAD indices are inconsistent"
            )
        expected_complete = (
            self.orphan_pad_face_count == 0
            and self.ambiguous_source_face_count == 0
            and self.winding_mismatch_face_count == 0
        )
        exact_match_count = (
            self.pad_face_count - self.orphan_pad_face_count
        )
        if (
            self.same_winding_match_count
            + self.winding_mismatch_face_count
            != exact_match_count
        ):
            raise CollisionContractError(
                "terminal winding counts disagree with exact matches"
            )
        if self.complete_pad_identity_verified != expected_complete:
            raise CollisionContractError(
                "complete PAD identity flag disagrees with exact partition"
            )
        expected_formal = (
            self.exact_cover_verified
            and self.complete_pad_identity_verified
            and self.source_mesh.coverage_mode.formal_solid_coverage
        )
        if self.formal_collision_eligible != expected_formal:
            raise CollisionContractError(
                "formal collision eligibility disagrees with coverage evidence"
            )
        expected_digest = _terminal_partition_digest(
            source_mesh=self.source_mesh,
            pad_source_sha256=self.pad_source_sha256,
            input_provenance=self.input_provenance,
            pad_allowed=self.pad_allowed,
            nonpad_forbidden=self.nonpad_forbidden,
            orphan_pad_face_indices=self.orphan_pad_face_indices,
            winding_mismatch_pad_face_indices=(
                self.winding_mismatch_pad_face_indices
            ),
            counts=integer_fields,
        )
        if self.partition_sha256 != expected_digest:
            raise CollisionContractError(
                "terminal partition digest is not canonical"
            )
        object.__setattr__(self, "pad_source_path", pad_path)

    @property
    def audit(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "method_id": METHOD_ID,
                "triangle_identity_policy": TRIANGLE_IDENTITY_POLICY,
                "terminal_partition_policy": TERMINAL_PARTITION_POLICY,
                "source_asset_id": self.source_mesh.asset_id,
                "source_mesh_sha256": self.source_mesh.sha256,
                "pad_source_sha256": self.pad_source_sha256,
                "input_provenance": self.input_provenance,
                "source_face_count": self.source_face_count,
                "pad_face_count": self.pad_face_count,
                "pad_allowed_face_count": len(self.pad_allowed),
                "nonpad_forbidden_face_count": len(self.nonpad_forbidden),
                "orphan_pad_face_count": self.orphan_pad_face_count,
                "ambiguous_source_face_count": (
                    self.ambiguous_source_face_count
                ),
                "same_winding_match_count": (
                    self.same_winding_match_count
                ),
                "winding_mismatch_face_count": (
                    self.winding_mismatch_face_count
                ),
                "orphan_pad_face_indices": list(
                    self.orphan_pad_face_indices
                ),
                "winding_mismatch_pad_face_indices": list(
                    self.winding_mismatch_pad_face_indices
                ),
                "exact_cover_verified": self.exact_cover_verified,
                "complete_pad_identity_verified": (
                    self.complete_pad_identity_verified
                ),
                "formal_collision_eligible": (
                    self.formal_collision_eligible
                ),
                "partition_sha256": self.partition_sha256,
            }
        )


@dataclass(frozen=True, order=True)
class DisabledCollisionAssertion:
    link_a: str
    link_b: str
    reason: str

    def __post_init__(self) -> None:
        if not self.link_a or not self.link_b or self.link_a == self.link_b:
            raise CollisionContractError(
                "disabled collision assertion needs two distinct links"
            )
        if self.reason not in {"Adjacent", "Never"}:
            raise CollisionContractError(
                "disabled collision assertion reason is not registered"
            )


@dataclass(frozen=True)
class SelfCollisionPairInventory:
    link_names: tuple[str, ...]
    all_pairs: tuple[tuple[str, str], ...]
    srdf_assertions: tuple[DisabledCollisionAssertion, ...]
    restarted_pair_policies: tuple[
        tuple[tuple[str, str], PairPolicy], ...
    ]
    inventory_sha256: str

    def __post_init__(self) -> None:
        if (
            len(self.link_names) < 2
            or any(
                not isinstance(name, str) or not name
                for name in self.link_names
            )
        ):
            raise CollisionContractError(
                "self-collision inventory needs at least two named links"
            )
        if len(set(self.link_names)) != len(self.link_names):
            raise CollisionContractError("self-collision link names repeat")
        if self.link_names != tuple(sorted(self.link_names)):
            raise CollisionContractError(
                "self-collision link names must use canonical order"
            )
        expected_pairs = tuple(itertools.combinations(self.link_names, 2))
        if self.all_pairs != expected_pairs:
            raise CollisionContractError(
                "self-collision inventory must contain every unordered pair"
            )
        pair_set = set(self.all_pairs)
        assertions = {
            _canonical_pair(row.link_a, row.link_b)
            for row in self.srdf_assertions
        }
        if len(assertions) != len(self.srdf_assertions):
            raise CollisionContractError(
                "SRDF contains duplicate disabled collision pairs"
            )
        if not assertions <= pair_set:
            raise CollisionContractError(
                "SRDF assertion references a link outside the inventory"
            )
        policy_rows = self.restarted_pair_policies
        policy_keys = tuple(pair for pair, _policy in policy_rows)
        if (
            len(policy_rows) != len(self.all_pairs)
            or len(set(policy_keys)) != len(policy_keys)
            or policy_keys != self.all_pairs
            or any(
                policy is not PairPolicy.FORBIDDEN
                for _pair, policy in policy_rows
            )
        ):
            raise CollisionContractError(
                "every self-collision pair must restart as FORBIDDEN"
            )
        expected_digest = _self_collision_inventory_digest(
            self.link_names,
            self.all_pairs,
            self.srdf_assertions,
            self.restarted_pair_policies,
        )
        if self.inventory_sha256 != expected_digest:
            raise CollisionContractError(
                "self-collision inventory digest is not canonical"
            )

    @property
    def audit(self) -> Mapping[str, object]:
        reason_counts = Counter(row.reason for row in self.srdf_assertions)
        return MappingProxyType(
            {
                "method_id": METHOD_ID,
                "srdf_policy": SRDF_POLICY,
                "link_count": len(self.link_names),
                "pair_count": len(self.all_pairs),
                "srdf_assertion_count": len(self.srdf_assertions),
                "srdf_reason_counts": dict(sorted(reason_counts.items())),
                "srdf_exemptions_applied": False,
                "restarted_forbidden_count": len(
                    self.restarted_pair_policies
                ),
                "inventory_sha256": self.inventory_sha256,
            }
        )


def _canonical_pair(first: str, second: str) -> tuple[str, str]:
    return (first, second) if first < second else (second, first)


def canonical_unoriented_triangle_bytes(triangle_m: np.ndarray) -> bytes:
    triangle = np.asarray(triangle_m, dtype=np.float64)
    if triangle.shape != (3, 3) or not np.all(np.isfinite(triangle)):
        raise CollisionContractError(
            "triangle identity requires finite shape (3, 3)"
        )
    canonical = np.array(triangle, dtype="<f8", copy=True)
    with np.errstate(over="ignore", invalid="ignore"):
        edges = canonical[1:] - canonical[0]
    if not np.all(np.isfinite(edges)):
        raise CollisionContractError(
            "triangle edge arithmetic overflowed"
        )
    edge_scale = float(np.max(np.abs(edges)))
    if edge_scale == 0.0:
        raise CollisionContractError(
            "degenerate triangle cannot receive a collision role"
        )
    scaled_cross = np.cross(
        edges[0] / edge_scale, edges[1] / edge_scale
    )
    if not np.all(np.isfinite(scaled_cross)) or float(
        np.linalg.norm(scaled_cross)
    ) == 0.0:
        raise CollisionContractError(
            "degenerate triangle cannot receive a collision role"
        )
    vertices = sorted(row.tobytes(order="C") for row in canonical)
    return b"".join(vertices)


def canonical_oriented_triangle_bytes(triangle_m: np.ndarray) -> bytes:
    """Return exact cyclic-winding identity while preserving IEEE bits."""

    triangle = np.asarray(triangle_m, dtype=np.float64)
    if triangle.shape != (3, 3) or not np.all(np.isfinite(triangle)):
        raise CollisionContractError(
            "triangle identity requires finite shape (3, 3)"
        )
    canonical = np.array(triangle, dtype="<f8", copy=True)
    with np.errstate(over="ignore", invalid="ignore"):
        edges = canonical[1:] - canonical[0]
    if not np.all(np.isfinite(edges)):
        raise CollisionContractError(
            "triangle edge arithmetic overflowed"
        )
    edge_scale = float(np.max(np.abs(edges)))
    if edge_scale == 0.0:
        raise CollisionContractError(
            "degenerate triangle cannot receive a collision role"
        )
    scaled_cross = np.cross(
        edges[0] / edge_scale, edges[1] / edge_scale
    )
    if not np.all(np.isfinite(scaled_cross)) or float(
        np.linalg.norm(scaled_cross)
    ) == 0.0:
        raise CollisionContractError(
            "degenerate triangle cannot receive a collision role"
        )
    rows = tuple(row.tobytes(order="C") for row in canonical)
    return min(
        rows[0] + rows[1] + rows[2],
        rows[1] + rows[2] + rows[0],
        rows[2] + rows[0] + rows[1],
    )


def canonical_oriented_triangle_sha256(triangle_m: np.ndarray) -> str:
    return hashlib.sha256(
        canonical_oriented_triangle_bytes(triangle_m)
    ).hexdigest()


def canonical_unoriented_triangle_sha256(triangle_m: np.ndarray) -> str:
    return hashlib.sha256(
        canonical_unoriented_triangle_bytes(triangle_m)
    ).hexdigest()


def triangle_instance_keys(
    triangles_m: np.ndarray,
    *,
    source_mesh_sha256: str,
) -> tuple[TriangleInstanceKey, ...]:
    triangles = np.asarray(triangles_m, dtype=np.float64)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise CollisionContractError(
            "triangle instance array must have shape (F, 3, 3)"
        )
    occurrences: defaultdict[str, int] = defaultdict(int)
    result: list[TriangleInstanceKey] = []
    for triangle in triangles:
        geometry = canonical_unoriented_triangle_sha256(triangle)
        occurrence = occurrences[geometry]
        occurrences[geometry] += 1
        result.append(
            TriangleInstanceKey(
                source_mesh_sha256=source_mesh_sha256,
                canonical_geometry_sha256=geometry,
                occurrence_index=occurrence,
            )
        )
    return tuple(result)


def _terminal_partition_digest(
    *,
    source_mesh: VerifiedCollisionMesh,
    pad_source_sha256: str,
    input_provenance: str,
    pad_allowed: tuple[TriangleInstanceKey, ...],
    nonpad_forbidden: tuple[TriangleInstanceKey, ...],
    orphan_pad_face_indices: tuple[int, ...],
    winding_mismatch_pad_face_indices: tuple[int, ...],
    counts: tuple[int, ...],
) -> str:
    digest = hashlib.sha256()
    digest.update(METHOD_ID.encode("ascii") + b"\0")
    for value in (
        source_mesh.asset_id,
        source_mesh.link_name,
        source_mesh.sha256,
        source_mesh.unit,
        source_mesh.coverage_mode.value,
        pad_source_sha256,
        input_provenance,
    ):
        digest.update(value.encode("utf-8") + b"\0")
    digest.update(
        np.asarray(source_mesh.local_transform, dtype="<f8").tobytes(
            order="C"
        )
    )
    for role, rows in ((b"A", pad_allowed), (b"F", nonpad_forbidden)):
        for row in rows:
            digest.update(role)
            digest.update(row.source_mesh_sha256.encode("ascii"))
            digest.update(row.canonical_geometry_sha256.encode("ascii"))
            digest.update(row.occurrence_index.to_bytes(8, "little"))
    for label, indices in (
        (b"O", orphan_pad_face_indices),
        (b"W", winding_mismatch_pad_face_indices),
    ):
        for index in indices:
            digest.update(label)
            digest.update(index.to_bytes(8, "little"))
    for count in counts:
        digest.update(count.to_bytes(8, "little"))
    return digest.hexdigest()


def _new_terminal_triangle_partition(
    **values: object,
) -> TerminalTrianglePartition:
    result = object.__new__(TerminalTrianglePartition)
    for name in TerminalTrianglePartition.__dataclass_fields__:
        object.__setattr__(result, name, values[name])
    result.__post_init__()
    return result


def _build_terminal_triangle_partition(
    *,
    source_mesh: VerifiedCollisionMesh,
    source_triangles_m: np.ndarray,
    pad_source_path: Path,
    pad_source_sha256: str,
    pad_triangles_m: np.ndarray,
    input_provenance: str,
) -> TerminalTrianglePartition:
    source_triangles = np.asarray(source_triangles_m, dtype=np.float64)
    pad_triangles = np.asarray(pad_triangles_m, dtype=np.float64)
    pad_path = Path(pad_source_path).resolve()
    if not pad_path.is_file() or file_sha256(pad_path) != pad_source_sha256:
        raise CollisionContractError(
            f"PAD source SHA-256 mismatch or unavailable: {pad_path}"
        )
    source_instances = triangle_instance_keys(
        source_triangles, source_mesh_sha256=source_mesh.sha256
    )
    source_counts = Counter(
        row.canonical_geometry_sha256 for row in source_instances
    )
    pad_geometry = tuple(
        canonical_unoriented_triangle_sha256(triangle)
        for triangle in pad_triangles
    )
    source_winding = tuple(
        canonical_oriented_triangle_sha256(triangle)
        for triangle in source_triangles
    )
    pad_winding = tuple(
        canonical_oriented_triangle_sha256(triangle)
        for triangle in pad_triangles
    )
    pad_counts = Counter(pad_geometry)
    ambiguous_geometry = {
        geometry
        for geometry, pad_count in pad_counts.items()
        if 0 < pad_count < source_counts.get(geometry, 0)
    }
    allowed_geometry = {
        geometry
        for geometry, source_count in source_counts.items()
        if source_count > 0
        and pad_counts.get(geometry, 0) >= source_count
        and geometry not in ambiguous_geometry
    }
    allowed = tuple(
        row
        for row in source_instances
        if row.canonical_geometry_sha256 in allowed_geometry
    )
    forbidden = tuple(
        row
        for row in source_instances
        if row.canonical_geometry_sha256 not in allowed_geometry
    )
    source_oriented_counts = Counter(zip(
        (
            row.canonical_geometry_sha256
            for row in source_instances
        ),
        source_winding,
    ))
    same_winding_indices: set[int] = set()
    remaining_source_oriented = Counter(source_oriented_counts)
    for pad_index, key in enumerate(zip(pad_geometry, pad_winding)):
        if remaining_source_oriented.get(key, 0) > 0:
            same_winding_indices.add(pad_index)
            remaining_source_oriented[key] -= 1
    remaining_source_geometry = Counter()
    for (geometry, _winding), count in remaining_source_oriented.items():
        remaining_source_geometry[geometry] += count
    mismatch_indices: list[int] = []
    orphan_indices: list[int] = []
    for pad_index, geometry in enumerate(pad_geometry):
        if pad_index in same_winding_indices:
            continue
        if remaining_source_geometry.get(geometry, 0) > 0:
            mismatch_indices.append(pad_index)
            remaining_source_geometry[geometry] -= 1
        else:
            orphan_indices.append(pad_index)
    same_winding_count = len(same_winding_indices)
    winding_mismatch_count = len(mismatch_indices)
    orphan_count = len(orphan_indices)
    ambiguous_source_count = sum(
        source_counts[geometry] for geometry in ambiguous_geometry
    )
    exact_cover = (
        len(allowed) + len(forbidden) == len(source_instances)
        and not (set(allowed) & set(forbidden))
    )
    complete_pad = (
        orphan_count == 0
        and ambiguous_source_count == 0
        and winding_mismatch_count == 0
    )
    formal = (
        exact_cover
        and complete_pad
        and source_mesh.coverage_mode.formal_solid_coverage
    )
    integer_counts = (
        orphan_count,
        ambiguous_source_count,
        same_winding_count,
        winding_mismatch_count,
        len(source_triangles),
        len(pad_triangles),
    )
    partition_sha256 = _terminal_partition_digest(
        source_mesh=source_mesh,
        pad_source_sha256=pad_source_sha256,
        input_provenance=input_provenance,
        pad_allowed=allowed,
        nonpad_forbidden=forbidden,
        orphan_pad_face_indices=tuple(orphan_indices),
        winding_mismatch_pad_face_indices=tuple(mismatch_indices),
        counts=integer_counts,
    )
    return _new_terminal_triangle_partition(
        source_mesh=source_mesh,
        pad_source_path=pad_path,
        pad_source_sha256=pad_source_sha256,
        input_provenance=input_provenance,
        pad_allowed=allowed,
        nonpad_forbidden=forbidden,
        orphan_pad_face_count=orphan_count,
        ambiguous_source_face_count=ambiguous_source_count,
        same_winding_match_count=same_winding_count,
        winding_mismatch_face_count=winding_mismatch_count,
        orphan_pad_face_indices=tuple(orphan_indices),
        winding_mismatch_pad_face_indices=tuple(mismatch_indices),
        source_face_count=len(source_triangles),
        pad_face_count=len(pad_triangles),
        exact_cover_verified=exact_cover,
        complete_pad_identity_verified=complete_pad,
        formal_collision_eligible=formal,
        partition_sha256=partition_sha256,
    )


def build_synthetic_terminal_triangle_partition(
    *,
    source_mesh: VerifiedCollisionMesh,
    source_triangles_m: np.ndarray,
    pad_source_path: Path,
    pad_source_sha256: str,
    pad_triangles_m: np.ndarray,
) -> TerminalTrianglePartition:
    """Build an array fixture that can never establish file lineage."""

    if source_mesh.coverage_mode is not CoverageMode.SYNTHETIC_ARRAY_FIXTURE:
        raise CollisionContractError(
            "public array builder is restricted to synthetic fixtures"
        )
    return _build_terminal_triangle_partition(
        source_mesh=source_mesh,
        source_triangles_m=source_triangles_m,
        pad_source_path=pad_source_path,
        pad_source_sha256=pad_source_sha256,
        pad_triangles_m=pad_triangles_m,
        input_provenance="SYNTHETIC_ARRAY_FIXTURE_NOT_FILE_LINEAGE",
    )


def load_exact_terminal_triangle_partition(
    *,
    asset_id: str,
    link_name: str,
    source_stl_path: Path | str,
    source_stl_sha256: str,
    source_unit: str,
    source_coverage_mode: CoverageMode,
    pad_npz_path: Path | str,
    pad_npz_sha256: str,
    local_transform: np.ndarray,
) -> TerminalTrianglePartition:
    source_path = Path(source_stl_path).resolve()
    pad_path = Path(pad_npz_path).resolve()
    source_mesh = VerifiedCollisionMesh(
        asset_id=asset_id,
        link_name=link_name,
        path=source_path,
        sha256=source_stl_sha256,
        unit=source_unit,
        local_transform=local_transform,
        coverage_mode=source_coverage_mode,
    )
    if not pad_path.is_file() or file_sha256(pad_path) != pad_npz_sha256:
        raise CollisionContractError(
            f"PAD source SHA-256 mismatch or unavailable: {pad_path}"
        )
    mesh, provenance = load_stl_mesh(
        source_path, unit=source_unit, orient_outward=False
    )
    if provenance.source_sha256 != source_stl_sha256:
        raise CollisionContractError(
            "STL loader provenance differs from verified source hash"
        )
    try:
        with np.load(pad_path, allow_pickle=False) as arrays:
            if "points_local_m" not in arrays or "faces" not in arrays:
                raise CollisionContractError(
                    "PAD NPZ must expose points_local_m and faces"
                )
            points = np.asarray(arrays["points_local_m"], dtype=np.float64)
            faces = np.asarray(arrays["faces"])
    except (OSError, ValueError, KeyError) as exc:
        raise CollisionContractError(
            f"PAD NPZ cannot be loaded: {pad_path}"
        ) from exc
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or not np.all(np.isfinite(points))
        or faces.ndim != 2
        or faces.shape[1] != 3
        or faces.dtype.kind not in "iu"
    ):
        raise CollisionContractError("PAD NPZ triangle arrays are malformed")
    faces = np.asarray(faces, dtype=np.int64)
    if np.any(faces < 0) or np.any(faces >= len(points)):
        raise CollisionContractError("PAD NPZ face index is outside points")
    return _build_terminal_triangle_partition(
        source_mesh=source_mesh,
        source_triangles_m=mesh.face_vertices_m,
        pad_source_path=pad_path,
        pad_source_sha256=pad_npz_sha256,
        pad_triangles_m=points[faces],
        input_provenance="HASH_BOUND_SOURCE_STL_AND_PAD_NPZ",
    )


def parse_srdf_disabled_collision_assertions(
    path: Path | str,
) -> tuple[DisabledCollisionAssertion, ...]:
    source = Path(path).resolve()
    if not source.is_file():
        raise CollisionContractError(f"SRDF is unavailable: {source}")
    try:
        root = ET.parse(source).getroot()
    except (ET.ParseError, OSError) as exc:
        raise CollisionContractError(
            f"SRDF cannot be parsed: {source}"
        ) from exc
    rows: list[DisabledCollisionAssertion] = []
    for element in root.findall(".//disable_collisions"):
        link_a = element.attrib.get("link1", "")
        link_b = element.attrib.get("link2", "")
        reason = element.attrib.get("reason", "")
        canonical = _canonical_pair(link_a, link_b)
        rows.append(
            DisabledCollisionAssertion(
                link_a=canonical[0],
                link_b=canonical[1],
                reason=reason,
            )
        )
    return tuple(rows)


def _self_collision_inventory_digest(
    link_names: tuple[str, ...],
    all_pairs: tuple[tuple[str, str], ...],
    assertions: tuple[DisabledCollisionAssertion, ...],
    policies: tuple[tuple[tuple[str, str], PairPolicy], ...],
) -> str:
    digest = hashlib.sha256()
    digest.update(METHOD_ID.encode("ascii") + b"\0")
    for name in link_names:
        digest.update(name.encode("utf-8") + b"\0")
    for pair in all_pairs:
        digest.update(b"P")
        digest.update(pair[0].encode("utf-8") + b"\0")
        digest.update(pair[1].encode("utf-8") + b"\0")
    for row in assertions:
        digest.update(b"S")
        digest.update(row.link_a.encode("utf-8") + b"\0")
        digest.update(row.link_b.encode("utf-8") + b"\0")
        digest.update(row.reason.encode("utf-8") + b"\0")
    for pair, policy in policies:
        digest.update(b"R")
        digest.update(pair[0].encode("utf-8") + b"\0")
        digest.update(pair[1].encode("utf-8") + b"\0")
        digest.update(policy.value.encode("ascii") + b"\0")
    return digest.hexdigest()


def build_self_collision_pair_inventory(
    *,
    link_names: Sequence[str],
    srdf_assertions: Sequence[DisabledCollisionAssertion],
) -> SelfCollisionPairInventory:
    links = tuple(sorted(str(name) for name in link_names))
    if len(links) < 2 or any(not name for name in links):
        raise CollisionContractError(
            "self-collision inventory needs at least two named links"
        )
    if len(set(links)) != len(links):
        raise CollisionContractError("self-collision link names repeat")
    all_pairs = tuple(itertools.combinations(links, 2))
    pair_set = set(all_pairs)
    canonical_assertions = tuple(sorted(
        DisabledCollisionAssertion(
            *_canonical_pair(row.link_a, row.link_b), row.reason
        )
        for row in srdf_assertions
    ))
    if any(
        (row.link_a, row.link_b) not in pair_set
        for row in canonical_assertions
    ):
        raise CollisionContractError(
            "SRDF assertion references a link outside the inventory order"
        )
    restarted = tuple(
        (pair, PairPolicy.FORBIDDEN) for pair in all_pairs
    )
    digest = _self_collision_inventory_digest(
        links, all_pairs, canonical_assertions, restarted
    )
    return SelfCollisionPairInventory(
        link_names=links,
        all_pairs=all_pairs,
        srdf_assertions=canonical_assertions,
        restarted_pair_policies=restarted,
        inventory_sha256=digest,
    )


__all__ = [
    "CollisionContractError",
    "CoverageMode",
    "DisabledCollisionAssertion",
    "METHOD_ID",
    "PairPolicy",
    "SRDF_POLICY",
    "SelfCollisionPairInventory",
    "TERMINAL_PARTITION_POLICY",
    "TRIANGLE_IDENTITY_POLICY",
    "TerminalTrianglePartition",
    "TriangleInstanceKey",
    "VerifiedCollisionMesh",
    "build_self_collision_pair_inventory",
    "build_synthetic_terminal_triangle_partition",
    "canonical_oriented_triangle_bytes",
    "canonical_oriented_triangle_sha256",
    "canonical_unoriented_triangle_bytes",
    "canonical_unoriented_triangle_sha256",
    "load_exact_terminal_triangle_partition",
    "parse_srdf_disabled_collision_assertions",
    "triangle_instance_keys",
]
