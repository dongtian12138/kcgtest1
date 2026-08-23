"""Exact material certificate for a union of source-authored positive solids.

One closed component soup is not automatically one material boundary.  A
component may describe an additive solid, a subtractive void, or an unrelated
shell.  This module accepts the multi-component representation only when an
upstream, hash-bound authority assigns every source component the additive
``POSITIVE_SOLID`` role.  It then proves every component is independently
closed and embedded with the existing exact binary64 predicate.

Components are allowed to touch, overlap, or nest.  They are interpreted as a
set union of positive solids, never as alternating outer/void shells.  The
result is suitable for conservative collision-material membership, but it is
not promoted to one minimal outer surface and does not prove clearance,
trajectory safety, contact semantics, or dynamic behavior.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from kcg_connector.grasp.robust.material_boundary import (
    MaterialBoundaryError,
    certify_single_embedded_material_boundary,
)
from kcg_connector.grasp.robust.surface_orientation import (
    SurfaceBoundaryRole,
    audit_surface_orientation,
)


METHOD_ID = "CARTS_EXACT_POSITIVE_SOLID_COMPONENT_UNION_V1"
MATERIAL_OPERATION = "UNION_OF_CERTIFIED_POSITIVE_SOLIDS"
COMPONENT_POLICY = (
    "SOURCE_COMPONENTS_ORDERED_BY_CONTIGUOUS_FIRST_FACE_X_"
    "EVERY_COMPONENT_EXACT_SINGLE_EMBEDDED_BOUNDARY"
)
INTER_COMPONENT_POLICY = (
    "TOUCH_OVERLAP_AND_NESTING_ALLOWED_X_NO_SUBTRACTIVE_COMPONENTS"
)
REGISTERED_ROLE_AUTHORITY_KINDS = frozenset(
    {
        "USD_GPRIM_KCG_POSITIVE_VOLUME_TRUE_V1",
        "SUPPLIER_STEP_SINGLE_SOLID_V1",
        "TEST_EXPLICIT_POSITIVE_SOLID_ROLE_V1",
    }
)
CLAIM_LIMITATIONS = (
    "STATIC_SOURCE_INDEXED_BINARY64_TRIANGLE_COMPONENTS_ONLY",
    "POSITIVE_SOLID_ROLE_IS_HASH_BOUND_UPSTREAM_EVIDENCE",
    "INTER_COMPONENT_INTERSECTIONS_NOT_REJECTED_FOR_SET_UNION",
    "NOT_ONE_MINIMAL_OUTER_BOUNDARY",
    "NO_VISUAL_TO_COLLISION_COVERAGE_OR_CONTACT_SEMANTIC_CLAIM",
    "NO_ROBOT_POSE_TRAJECTORY_ENVIRONMENT_CLEARANCE_OR_DYNAMIC_CLAIM",
)

_GPRIM_TYPES = frozenset(
    {"Mesh", "Cylinder", "Sphere", "Cube", "Capsule", "Cone"}
)
_USDA_DEFINITION = re.compile(r'^(\s*)def\s+(\w+)\s+"([^"]+)"')
_USDA_POSITIVE = re.compile(r"custom\s+bool\s+kcg:positiveVolume\s*=\s*([01])")
_USDA_CLOSED = re.compile(r"custom\s+bool\s+kcg:closedManifold\s*=\s*([01])")


class PositiveSolidUnionError(ValueError):
    """Fail-closed aggregate-boundary error with a stable reason code."""

    def __init__(self, code: str, detail: str) -> None:
        if not code or not detail:
            raise ValueError("positive-solid-union error fields cannot be empty")
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(f"{self.code}: {self.detail}")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _canonical_sha256(document: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            document,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _string_inventory_sha256(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    digest.update(b"CARTS_POSITIVE_SOLID_COMPONENT_IDENTITY_INVENTORY_V1\0")
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


@dataclass(frozen=True)
class UsdaGprimSolidRole:
    """Authored role fields found on one deterministic USDA Gprim."""

    prim_path: str
    type_name: str
    positive_volume: bool | None
    closed_manifold: bool | None

    def __post_init__(self) -> None:
        if (
            not self.prim_path.startswith("/")
            or self.type_name not in _GPRIM_TYPES
            or self.positive_volume not in (None, False, True)
            or self.closed_manifold not in (None, False, True)
        ):
            raise ValueError("USDA Gprim solid-role record is malformed")


def parse_usda_gprim_solid_roles(
    path: Path | str,
) -> Mapping[str, UsdaGprimSolidRole]:
    """Parse authored additive-solid role fields without loading PhysX."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    stack: list[tuple[int, str]] = []
    current_path: str | None = None
    records: dict[str, dict[str, object]] = {}
    with source.open("r", encoding="utf-8") as stream:
        for line in stream:
            definition = _USDA_DEFINITION.match(line)
            if definition is not None:
                indentation = len(definition.group(1))
                type_name = definition.group(2)
                prim_name = definition.group(3)
                while stack and stack[-1][0] >= indentation:
                    stack.pop()
                stack.append((indentation, prim_name))
                path_value = "/" + "/".join(row[1] for row in stack)
                if type_name in _GPRIM_TYPES:
                    if path_value in records:
                        raise PositiveSolidUnionError(
                            "DUPLICATE_USDA_GPRIM_PATH", path_value
                        )
                    records[path_value] = {
                        "type_name": type_name,
                        "positive_volume": None,
                        "closed_manifold": None,
                    }
                    current_path = path_value
                else:
                    current_path = None
                continue
            if current_path is None:
                continue
            positive = _USDA_POSITIVE.search(line)
            if positive is not None:
                if records[current_path]["positive_volume"] is not None:
                    raise PositiveSolidUnionError(
                        "DUPLICATE_USDA_POSITIVE_VOLUME_ROLE", current_path
                    )
                records[current_path]["positive_volume"] = positive.group(1) == "1"
            closed = _USDA_CLOSED.search(line)
            if closed is not None:
                if records[current_path]["closed_manifold"] is not None:
                    raise PositiveSolidUnionError(
                        "DUPLICATE_USDA_CLOSED_MANIFOLD_ROLE", current_path
                    )
                records[current_path]["closed_manifold"] = closed.group(1) == "1"
    if not records:
        raise PositiveSolidUnionError(
            "USDA_GPRIM_ROLE_INVENTORY_EMPTY", str(source)
        )
    return {
        prim_path: UsdaGprimSolidRole(
            prim_path=prim_path,
            type_name=str(record["type_name"]),
            positive_volume=record["positive_volume"],  # type: ignore[arg-type]
            closed_manifold=record["closed_manifold"],  # type: ignore[arg-type]
        )
        for prim_path, record in records.items()
    }


def bind_positive_usda_component_ids(
    component_ids: Sequence[str],
    role_inventory: Mapping[str, UsdaGprimSolidRole],
) -> tuple[str, ...]:
    """Return the exact component inventory only when every role is additive."""

    identities = tuple(str(value) for value in component_ids)
    if not identities or len(set(identities)) != len(identities):
        raise PositiveSolidUnionError(
            "INVALID_COMPONENT_IDENTITY_INVENTORY",
            "component identities must be non-empty and unique",
        )
    for identity in identities:
        role = role_inventory.get(identity)
        if role is None:
            raise PositiveSolidUnionError(
                "SOURCE_COMPONENT_ROLE_MISSING", identity
            )
        if role.positive_volume is not True:
            raise PositiveSolidUnionError(
                "SOURCE_COMPONENT_NOT_POSITIVE_SOLID", identity
            )
        if role.closed_manifold is not True:
            raise PositiveSolidUnionError(
                "SOURCE_COMPONENT_NOT_AUTHORED_CLOSED", identity
            )
    return identities


@dataclass(frozen=True)
class PositiveSolidComponentBoundaryRecord:
    component_index: int
    source_component_id: str
    minimum_source_face_index: int
    maximum_source_face_index: int
    source_vertex_count: int
    source_face_count: int
    source_face_pair_count: int
    pair_coverage_count: int
    source_face_indices_sha256: str
    source_indexed_component_mesh_sha256: str
    material_boundary_certificate_sha256: str

    def __post_init__(self) -> None:
        integer_fields = (
            self.component_index,
            self.minimum_source_face_index,
            self.maximum_source_face_index,
            self.source_vertex_count,
            self.source_face_count,
            self.source_face_pair_count,
            self.pair_coverage_count,
        )
        if any(type(value) is not int or value < 0 for value in integer_fields):
            raise ValueError("component-boundary counts must be nonnegative integers")
        if (
            not self.source_component_id
            or self.source_vertex_count < 4
            or self.source_face_count < 4
            or self.maximum_source_face_index - self.minimum_source_face_index + 1
            != self.source_face_count
            or self.source_face_pair_count
            != self.source_face_count * (self.source_face_count - 1) // 2
            or self.pair_coverage_count != self.source_face_pair_count
        ):
            raise ValueError("component-boundary inventory is inconsistent")
        for digest in (
            self.source_face_indices_sha256,
            self.source_indexed_component_mesh_sha256,
            self.material_boundary_certificate_sha256,
        ):
            if not _is_sha256(digest):
                raise ValueError("component-boundary digest is invalid")

    def as_dict(self) -> dict[str, object]:
        return {
            "component_index": self.component_index,
            "source_component_id": self.source_component_id,
            "minimum_source_face_index": self.minimum_source_face_index,
            "maximum_source_face_index": self.maximum_source_face_index,
            "source_vertex_count": self.source_vertex_count,
            "source_face_count": self.source_face_count,
            "source_face_pair_count": self.source_face_pair_count,
            "pair_coverage_count": self.pair_coverage_count,
            "source_face_indices_sha256": self.source_face_indices_sha256,
            "source_indexed_component_mesh_sha256": (
                self.source_indexed_component_mesh_sha256
            ),
            "material_boundary_certificate_sha256": (
                self.material_boundary_certificate_sha256
            ),
        }


@dataclass(frozen=True)
class PositiveSolidAggregateCertificate:
    method_id: str
    material_operation: str
    component_policy: str
    inter_component_policy: str
    source_asset_sha256: str
    source_indexed_mesh_sha256: str
    orientation_certificate_sha256: str
    role_authority_kind: str
    role_authority_sha256: str
    source_component_identity_sha256: str
    source_vertex_count: int
    source_face_count: int
    source_component_count: int
    embedded_positive_solid_count: int
    within_component_source_face_pair_count: int
    within_component_pair_coverage_count: int
    inter_component_face_pair_count: int
    component_face_count_histogram: tuple[tuple[int, int], ...]
    component_records: tuple[PositiveSolidComponentBoundaryRecord, ...]
    source_face_coverage_status: str
    component_embedding_status: str
    positive_solid_role_status: str
    inter_component_intersection_status: str
    formal_material_boundary_eligible: bool
    claim_limitations: tuple[str, ...]
    certificate_sha256: str

    def __post_init__(self) -> None:
        if (
            self.method_id != METHOD_ID
            or self.material_operation != MATERIAL_OPERATION
            or self.component_policy != COMPONENT_POLICY
            or self.inter_component_policy != INTER_COMPONENT_POLICY
            or self.role_authority_kind not in REGISTERED_ROLE_AUTHORITY_KINDS
        ):
            raise ValueError("positive-solid-union method contract changed")
        for digest in (
            self.source_asset_sha256,
            self.source_indexed_mesh_sha256,
            self.orientation_certificate_sha256,
            self.role_authority_sha256,
            self.source_component_identity_sha256,
            self.certificate_sha256,
        ):
            if not _is_sha256(digest):
                raise ValueError("positive-solid-union digest is invalid")
        count_fields = (
            self.source_vertex_count,
            self.source_face_count,
            self.source_component_count,
            self.embedded_positive_solid_count,
            self.within_component_source_face_pair_count,
            self.within_component_pair_coverage_count,
            self.inter_component_face_pair_count,
        )
        if any(type(value) is not int or value < 0 for value in count_fields):
            raise ValueError("positive-solid-union counts must be nonnegative integers")
        records = self.component_records
        if (
            self.source_vertex_count < 4
            or self.source_face_count < 4
            or self.source_component_count < 1
            or len(records) != self.source_component_count
            or self.embedded_positive_solid_count != self.source_component_count
            or tuple(row.component_index for row in records)
            != tuple(range(self.source_component_count))
            or len({row.source_component_id for row in records}) != len(records)
            or sum(row.source_face_count for row in records)
            != self.source_face_count
            or sum(row.source_face_pair_count for row in records)
            != self.within_component_source_face_pair_count
            or sum(row.pair_coverage_count for row in records)
            != self.within_component_pair_coverage_count
            or self.within_component_pair_coverage_count
            != self.within_component_source_face_pair_count
        ):
            raise ValueError("positive-solid component coverage is inconsistent")
        cursor = 0
        for record in records:
            if record.minimum_source_face_index != cursor:
                raise ValueError("component source-face ranges are not contiguous")
            cursor = record.maximum_source_face_index + 1
        total_pairs = self.source_face_count * (self.source_face_count - 1) // 2
        if (
            cursor != self.source_face_count
            or self.inter_component_face_pair_count
            != total_pairs - self.within_component_source_face_pair_count
            or self.component_face_count_histogram
            != tuple(sorted(Counter(row.source_face_count for row in records).items()))
            or self.source_component_identity_sha256
            != _string_inventory_sha256(
                tuple(row.source_component_id for row in records)
            )
        ):
            raise ValueError("aggregate source-face or identity accounting changed")
        if (
            self.source_face_coverage_status != "VERIFIED_EXACTLY_ONCE"
            or self.component_embedding_status != "VERIFIED_EVERY_COMPONENT"
            or self.positive_solid_role_status
            != "VERIFIED_EVERY_COMPONENT_BY_HASH_BOUND_AUTHORITY"
            or self.inter_component_intersection_status
            != "NOT_REJECTED_POSITIVE_SOLID_SET_UNION"
            or self.formal_material_boundary_eligible is not True
            or self.claim_limitations != CLAIM_LIMITATIONS
            or self.certificate_sha256 != _certificate_sha256(self)
        ):
            raise ValueError("positive-solid-union success boundary is incomplete")

    def as_dict(self) -> dict[str, object]:
        return _certificate_document(self, include_digest=True)


def _certificate_document(
    certificate: PositiveSolidAggregateCertificate,
    *,
    include_digest: bool,
) -> dict[str, object]:
    document: dict[str, object] = {
        "method_id": certificate.method_id,
        "material_operation": certificate.material_operation,
        "component_policy": certificate.component_policy,
        "inter_component_policy": certificate.inter_component_policy,
        "source_asset_sha256": certificate.source_asset_sha256,
        "source_indexed_mesh_sha256": certificate.source_indexed_mesh_sha256,
        "orientation_certificate_sha256": (
            certificate.orientation_certificate_sha256
        ),
        "role_authority_kind": certificate.role_authority_kind,
        "role_authority_sha256": certificate.role_authority_sha256,
        "source_component_identity_sha256": (
            certificate.source_component_identity_sha256
        ),
        "source_vertex_count": certificate.source_vertex_count,
        "source_face_count": certificate.source_face_count,
        "source_component_count": certificate.source_component_count,
        "embedded_positive_solid_count": certificate.embedded_positive_solid_count,
        "within_component_source_face_pair_count": (
            certificate.within_component_source_face_pair_count
        ),
        "within_component_pair_coverage_count": (
            certificate.within_component_pair_coverage_count
        ),
        "inter_component_face_pair_count": (
            certificate.inter_component_face_pair_count
        ),
        "component_face_count_histogram": [
            [face_count, component_count]
            for face_count, component_count in certificate.component_face_count_histogram
        ],
        "component_records": [row.as_dict() for row in certificate.component_records],
        "source_face_coverage_status": certificate.source_face_coverage_status,
        "component_embedding_status": certificate.component_embedding_status,
        "positive_solid_role_status": certificate.positive_solid_role_status,
        "inter_component_intersection_status": (
            certificate.inter_component_intersection_status
        ),
        "formal_material_boundary_eligible": (
            certificate.formal_material_boundary_eligible
        ),
        "claim_limitations": list(certificate.claim_limitations),
    }
    if include_digest:
        document["certificate_sha256"] = certificate.certificate_sha256
    return document


def _certificate_sha256(
    certificate: PositiveSolidAggregateCertificate,
) -> str:
    return _canonical_sha256(_certificate_document(certificate, include_digest=False))


def certify_positive_solid_component_union(
    vertices_m: Sequence[Sequence[float]] | np.ndarray,
    faces: Sequence[Sequence[int]] | np.ndarray,
    *,
    source_asset_sha256: str,
    source_component_ids: Sequence[str],
    positive_solid_component_ids: Sequence[str],
    role_authority_kind: str,
    role_authority_sha256: str,
) -> PositiveSolidAggregateCertificate:
    """Prove a contiguous component soup is a union of positive solids."""

    if not _is_sha256(source_asset_sha256):
        raise PositiveSolidUnionError(
            "INVALID_SOURCE_ASSET_SHA256", str(source_asset_sha256)
        )
    if role_authority_kind not in REGISTERED_ROLE_AUTHORITY_KINDS:
        raise PositiveSolidUnionError(
            "UNREGISTERED_ROLE_AUTHORITY_KIND", str(role_authority_kind)
        )
    if not _is_sha256(role_authority_sha256):
        raise PositiveSolidUnionError(
            "INVALID_ROLE_AUTHORITY_SHA256", str(role_authority_sha256)
        )
    component_ids = tuple(str(value) for value in source_component_ids)
    positive_ids = tuple(str(value) for value in positive_solid_component_ids)
    if (
        not component_ids
        or any(not value for value in component_ids)
        or len(set(component_ids)) != len(component_ids)
    ):
        raise PositiveSolidUnionError(
            "INVALID_COMPONENT_IDENTITY_INVENTORY",
            "source component identities must be non-empty and unique",
        )
    if (
        any(not value for value in positive_ids)
        or len(set(positive_ids)) != len(positive_ids)
        or set(positive_ids) != set(component_ids)
    ):
        raise PositiveSolidUnionError(
            "INCOMPLETE_POSITIVE_SOLID_ROLE_COVERAGE",
            f"source_components={len(component_ids)}, positive_roles={len(positive_ids)}",
        )

    orientation = audit_surface_orientation(
        vertices_m,
        faces,
        role=SurfaceBoundaryRole.SOURCE_INDEXED_CLOSED_COMPONENT_SOUP,
    )
    if orientation.component_count != len(component_ids):
        raise PositiveSolidUnionError(
            "SOURCE_COMPONENT_IDENTITY_COUNT_MISMATCH",
            f"topology_components={orientation.component_count}, "
            f"source_component_ids={len(component_ids)}",
        )
    vertices = np.asarray(vertices_m)
    indexed_faces = np.asarray(faces, dtype=np.int64)
    component_by_face = np.asarray(
        orientation.canonical_component_index_by_source_face,
        dtype=np.int64,
    )
    records: list[PositiveSolidComponentBoundaryRecord] = []
    for component in orientation.components:
        source_face_indices = np.flatnonzero(
            component_by_face == component.component_index
        )
        expected = np.arange(
            component.minimum_source_face_index,
            component.minimum_source_face_index + component.source_face_count,
            dtype=np.int64,
        )
        if not np.array_equal(source_face_indices, expected):
            raise PositiveSolidUnionError(
                "NONCONTIGUOUS_SOURCE_COMPONENT_FACE_RANGE",
                f"component_index={component.component_index}",
            )
        source_faces = indexed_faces[source_face_indices]
        used_vertices = np.unique(source_faces.reshape(-1))
        local_faces = np.searchsorted(used_vertices, source_faces)
        local_vertices = np.asarray(vertices[used_vertices], dtype=np.float64)
        try:
            material = certify_single_embedded_material_boundary(
                local_vertices,
                local_faces,
            )
        except (MaterialBoundaryError, ValueError) as error:
            raise PositiveSolidUnionError(
                "COMPONENT_MATERIAL_BOUNDARY_FAILURE",
                f"component_index={component.component_index}, "
                f"source_component_id={component_ids[component.component_index]!r}, "
                f"cause={error}",
            ) from error
        records.append(
            PositiveSolidComponentBoundaryRecord(
                component_index=component.component_index,
                source_component_id=component_ids[component.component_index],
                minimum_source_face_index=int(source_face_indices[0]),
                maximum_source_face_index=int(source_face_indices[-1]),
                source_vertex_count=material.source_vertex_count,
                source_face_count=material.source_face_count,
                source_face_pair_count=material.source_face_pair_count,
                pair_coverage_count=material.pair_coverage_count,
                source_face_indices_sha256=(
                    component.source_face_indices_sha256
                ),
                source_indexed_component_mesh_sha256=(
                    material.source_indexed_mesh_sha256
                ),
                material_boundary_certificate_sha256=(
                    material.certificate_sha256
                ),
            )
        )

    record_tuple = tuple(records)
    within_pairs = sum(row.source_face_pair_count for row in record_tuple)
    total_pairs = len(indexed_faces) * (len(indexed_faces) - 1) // 2
    values: dict[str, Any] = {
        "method_id": METHOD_ID,
        "material_operation": MATERIAL_OPERATION,
        "component_policy": COMPONENT_POLICY,
        "inter_component_policy": INTER_COMPONENT_POLICY,
        "source_asset_sha256": source_asset_sha256,
        "source_indexed_mesh_sha256": orientation.source_indexed_mesh_sha256,
        "orientation_certificate_sha256": orientation.canonical_sha256,
        "role_authority_kind": role_authority_kind,
        "role_authority_sha256": role_authority_sha256,
        "source_component_identity_sha256": _string_inventory_sha256(
            component_ids
        ),
        "source_vertex_count": len(vertices),
        "source_face_count": len(indexed_faces),
        "source_component_count": len(record_tuple),
        "embedded_positive_solid_count": len(record_tuple),
        "within_component_source_face_pair_count": within_pairs,
        "within_component_pair_coverage_count": within_pairs,
        "inter_component_face_pair_count": total_pairs - within_pairs,
        "component_face_count_histogram": tuple(
            sorted(Counter(row.source_face_count for row in record_tuple).items())
        ),
        "component_records": record_tuple,
        "source_face_coverage_status": "VERIFIED_EXACTLY_ONCE",
        "component_embedding_status": "VERIFIED_EVERY_COMPONENT",
        "positive_solid_role_status": (
            "VERIFIED_EVERY_COMPONENT_BY_HASH_BOUND_AUTHORITY"
        ),
        "inter_component_intersection_status": (
            "NOT_REJECTED_POSITIVE_SOLID_SET_UNION"
        ),
        "formal_material_boundary_eligible": True,
        "claim_limitations": CLAIM_LIMITATIONS,
    }
    provisional = object.__new__(PositiveSolidAggregateCertificate)
    for name in PositiveSolidAggregateCertificate.__dataclass_fields__:
        if name != "certificate_sha256":
            object.__setattr__(provisional, name, values[name])
    object.__setattr__(provisional, "certificate_sha256", "0" * 64)
    return PositiveSolidAggregateCertificate(
        **values,
        certificate_sha256=_certificate_sha256(provisional),
    )


def positive_solid_aggregate_certificate_from_dict(
    value: Mapping[str, object],
) -> PositiveSolidAggregateCertificate:
    """Reconstruct and validate one persisted aggregate certificate."""

    expected = set(PositiveSolidAggregateCertificate.__dataclass_fields__)
    missing = sorted(expected.difference(value))
    extra = sorted(set(value).difference(expected))
    if missing or extra:
        raise PositiveSolidUnionError(
            "AGGREGATE_CERTIFICATE_SCHEMA_MISMATCH",
            f"missing={missing}, extra={extra}",
        )
    raw_records = value["component_records"]
    if not isinstance(raw_records, list):
        raise PositiveSolidUnionError(
            "INVALID_COMPONENT_RECORDS", "component_records must be a list"
        )
    record_fields = set(PositiveSolidComponentBoundaryRecord.__dataclass_fields__)
    records: list[PositiveSolidComponentBoundaryRecord] = []
    for index, raw_record in enumerate(raw_records):
        if not isinstance(raw_record, Mapping):
            raise PositiveSolidUnionError(
                "INVALID_COMPONENT_RECORD", f"component_index={index}"
            )
        record_missing = sorted(record_fields.difference(raw_record))
        record_extra = sorted(set(raw_record).difference(record_fields))
        if record_missing or record_extra:
            raise PositiveSolidUnionError(
                "COMPONENT_RECORD_SCHEMA_MISMATCH",
                f"component_index={index}, missing={record_missing}, "
                f"extra={record_extra}",
            )
        records.append(PositiveSolidComponentBoundaryRecord(**raw_record))
    raw_histogram = value["component_face_count_histogram"]
    if not isinstance(raw_histogram, list) or any(
        not isinstance(row, list) or len(row) != 2 for row in raw_histogram
    ):
        raise PositiveSolidUnionError(
            "INVALID_COMPONENT_FACE_COUNT_HISTOGRAM",
            "histogram must be a list of two-integer rows",
        )
    raw_limitations = value["claim_limitations"]
    if not isinstance(raw_limitations, list):
        raise PositiveSolidUnionError(
            "INVALID_AGGREGATE_CLAIM_LIMITATIONS",
            "claim_limitations must be a list",
        )
    fields = dict(value)
    fields["component_records"] = tuple(records)
    fields["component_face_count_histogram"] = tuple(
        (int(row[0]), int(row[1])) for row in raw_histogram
    )
    fields["claim_limitations"] = tuple(str(row) for row in raw_limitations)
    return PositiveSolidAggregateCertificate(**fields)


__all__ = [
    "CLAIM_LIMITATIONS",
    "COMPONENT_POLICY",
    "INTER_COMPONENT_POLICY",
    "MATERIAL_OPERATION",
    "METHOD_ID",
    "PositiveSolidAggregateCertificate",
    "PositiveSolidComponentBoundaryRecord",
    "PositiveSolidUnionError",
    "REGISTERED_ROLE_AUTHORITY_KINDS",
    "UsdaGprimSolidRole",
    "bind_positive_usda_component_ids",
    "certify_positive_solid_component_union",
    "parse_usda_gprim_solid_roles",
    "positive_solid_aggregate_certificate_from_dict",
]
