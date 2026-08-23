"""Strict loader for persisted object material-boundary evidence."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

from kcg_connector.grasp.robust.material_boundary import (
    MaterialBoundaryCertificate,
)
from kcg_connector.grasp.robust.positive_solid_union import (
    PositiveSolidAggregateCertificate,
    positive_solid_aggregate_certificate_from_dict,
)
from kcg_connector.grasp.robust.surface_orientation import (
    SurfaceOrientationCertificate,
)


SCHEMA_VERSION = "carts_object_material_boundary_certificate_v1"
CLAIM_SCOPE = "STATIC_MESH_LOCAL_MATERIAL_ONLY_NO_ROUTE_SCENE_OR_DYNAMICS"
CURRENT_REPRESENTATION = "POSITIVE_SOLID_COMPONENT_UNION"
SINGLE_REPRESENTATION = "SINGLE_EMBEDDED_MATERIAL_BOUNDARY"
CURRENT_ROLE_KIND = "USD_GPRIM_KCG_POSITIVE_VOLUME_TRUE_V1"
TRANSFER_ROLE_KIND = "SUPPLIER_STEP_SINGLE_SOLID_V1"


class ObjectMaterialBoundaryError(ValueError):
    """Fail-closed persisted-evidence error with a stable reason code."""

    def __init__(self, code: str, detail: str) -> None:
        if not code or not detail:
            raise ValueError("object material-boundary error fields cannot be empty")
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(f"{self.code}: {self.detail}")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _exact_keys(
    document: Mapping[str, object],
    expected: set[str],
    label: str,
) -> None:
    missing = sorted(expected.difference(document))
    extra = sorted(set(document).difference(expected))
    if missing or extra:
        raise ObjectMaterialBoundaryError(
            "EVIDENCE_SCHEMA_MISMATCH",
            f"{label}: missing={missing}, extra={extra}",
        )


def _repository_file(repository: Path, value: object, label: str) -> Path:
    raw = Path(str(value))
    if raw.is_absolute():
        raise ObjectMaterialBoundaryError(
            "ABSOLUTE_EVIDENCE_SOURCE_PATH", label
        )
    path = (repository / raw).resolve()
    try:
        path.relative_to(repository)
    except ValueError as error:
        raise ObjectMaterialBoundaryError(
            "EVIDENCE_SOURCE_ESCAPES_REPOSITORY", label
        ) from error
    if not path.is_file():
        raise ObjectMaterialBoundaryError(
            "EVIDENCE_SOURCE_FILE_MISSING", f"{label}: {path}"
        )
    return path


def _single_certificate_from_dict(
    value: Mapping[str, object],
) -> MaterialBoundaryCertificate:
    expected = set(MaterialBoundaryCertificate.__dataclass_fields__)
    _exact_keys(value, expected, "single material certificate")
    fields = dict(value)
    raw_limitations = fields["claim_limitations"]
    if not isinstance(raw_limitations, list):
        raise ObjectMaterialBoundaryError(
            "INVALID_SINGLE_CERTIFICATE_LIMITATIONS",
            "claim_limitations must be a list",
        )
    fields["claim_limitations"] = tuple(str(row) for row in raw_limitations)
    return MaterialBoundaryCertificate(**fields)


@dataclass(frozen=True)
class ObjectMaterialBoundaryEvidence:
    schema_version: str
    claim_scope: str
    object_id: str
    representation: str
    source_asset_path: str
    source_asset_sha256: str
    role_authority_kind: str
    role_authority_path: str
    role_authority_sha256: str
    evidence_path: str
    evidence_sha256: str
    certificate: MaterialBoundaryCertificate | PositiveSolidAggregateCertificate
    physics_loaded: bool
    collision_or_contact_truth_read: bool
    source_modified: bool
    formal_material_boundary_eligible: bool

    def __post_init__(self) -> None:
        if (
            self.schema_version != SCHEMA_VERSION
            or self.claim_scope != CLAIM_SCOPE
            or not self.object_id
            or self.representation
            not in {CURRENT_REPRESENTATION, SINGLE_REPRESENTATION}
            or not self.source_asset_path
            or not self.role_authority_path
            or not self.evidence_path
        ):
            raise ValueError("object material-boundary evidence identity is invalid")
        for digest in (
            self.source_asset_sha256,
            self.role_authority_sha256,
            self.evidence_sha256,
        ):
            if not _is_sha256(digest):
                raise ValueError("object material-boundary evidence digest is invalid")
        if (
            self.physics_loaded is not False
            or self.collision_or_contact_truth_read is not False
            or self.source_modified is not False
            or self.formal_material_boundary_eligible is not True
            or self.certificate.formal_material_boundary_eligible is not True
        ):
            raise ValueError("object material-boundary evidence claim boundary changed")
        if self.representation == CURRENT_REPRESENTATION:
            if (
                not isinstance(self.certificate, PositiveSolidAggregateCertificate)
                or self.role_authority_kind != CURRENT_ROLE_KIND
                or self.certificate.source_asset_sha256
                != self.source_asset_sha256
                or self.certificate.role_authority_sha256
                != self.role_authority_sha256
            ):
                raise ValueError("positive-solid aggregate evidence is inconsistent")
        elif (
            not isinstance(self.certificate, MaterialBoundaryCertificate)
            or self.role_authority_kind != TRANSFER_ROLE_KIND
        ):
            raise ValueError("single-solid material evidence is inconsistent")

    @property
    def certificate_sha256(self) -> str:
        return self.certificate.certificate_sha256


def load_object_material_boundary_evidence(
    evidence_path: Path | str,
    *,
    repository_root: Path | str,
    expected_object_id: str,
    expected_source_asset_path: Path | str,
    expected_source_asset_sha256: str,
    orientation_certificate: SurfaceOrientationCertificate,
) -> ObjectMaterialBoundaryEvidence:
    """Load one hash-checked artifact and bind it to the in-memory mesh."""

    repository = Path(repository_root).resolve()
    path = Path(evidence_path).resolve()
    try:
        path.relative_to(repository)
    except ValueError as error:
        raise ObjectMaterialBoundaryError(
            "EVIDENCE_PATH_ESCAPES_REPOSITORY", str(path)
        ) from error
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ObjectMaterialBoundaryError(
            "INVALID_EVIDENCE_JSON", str(path)
        ) from error
    if not isinstance(document, Mapping):
        raise ObjectMaterialBoundaryError(
            "INVALID_EVIDENCE_DOCUMENT", "top level must be a mapping"
        )
    _exact_keys(
        document,
        {
            "schema_version",
            "claim_scope",
            "object_id",
            "representation",
            "source_asset",
            "role_authority",
            "physics_loaded",
            "collision_or_contact_truth_read",
            "source_modified",
            "formal_material_boundary_eligible",
            "certificate",
        },
        "object material evidence",
    )
    if document["object_id"] != expected_object_id:
        raise ObjectMaterialBoundaryError(
            "OBJECT_ID_MISMATCH", str(document["object_id"])
        )
    source_asset = document["source_asset"]
    role_authority = document["role_authority"]
    certificate_document = document["certificate"]
    if not isinstance(source_asset, Mapping) or not isinstance(
        role_authority, Mapping
    ) or not isinstance(certificate_document, Mapping):
        raise ObjectMaterialBoundaryError(
            "INVALID_EVIDENCE_NESTED_DOCUMENT",
            "source_asset, role_authority and certificate must be mappings",
        )
    _exact_keys(source_asset, {"path", "sha256"}, "source_asset")
    source_path = _repository_file(repository, source_asset["path"], "source_asset")
    expected_path = Path(expected_source_asset_path).resolve()
    if (
        source_path != expected_path
        or source_asset["sha256"] != expected_source_asset_sha256
        or _sha256(source_path) != expected_source_asset_sha256
    ):
        raise ObjectMaterialBoundaryError(
            "SOURCE_ASSET_BINDING_MISMATCH", expected_object_id
        )
    representation = str(document["representation"])
    if representation == CURRENT_REPRESENTATION:
        _exact_keys(
            role_authority,
            {"kind", "path", "sha256", "component_count", "required_fields"},
            "current role_authority",
        )
        if (
            role_authority["kind"] != CURRENT_ROLE_KIND
            or role_authority["required_fields"]
            != ["kcg:positiveVolume=true", "kcg:closedManifold=true"]
        ):
            raise ObjectMaterialBoundaryError(
                "CURRENT_ROLE_AUTHORITY_CHANGED", expected_object_id
            )
        certificate = positive_solid_aggregate_certificate_from_dict(
            certificate_document
        )
        if int(role_authority["component_count"]) != certificate.source_component_count:
            raise ObjectMaterialBoundaryError(
                "ROLE_COMPONENT_COUNT_MISMATCH", expected_object_id
            )
    elif representation == SINGLE_REPRESENTATION:
        _exact_keys(
            role_authority,
            {
                "kind",
                "path",
                "sha256",
                "original_step_path",
                "original_step_sha256",
                "solid_count",
            },
            "single role_authority",
        )
        if role_authority["kind"] != TRANSFER_ROLE_KIND or int(
            role_authority["solid_count"]
        ) != 1:
            raise ObjectMaterialBoundaryError(
                "TRANSFER_ROLE_AUTHORITY_CHANGED", expected_object_id
            )
        original_step = _repository_file(
            repository,
            role_authority["original_step_path"],
            "original_step",
        )
        if _sha256(original_step) != role_authority["original_step_sha256"]:
            raise ObjectMaterialBoundaryError(
                "ORIGINAL_STEP_SHA256_MISMATCH", expected_object_id
            )
        certificate = _single_certificate_from_dict(certificate_document)
    else:
        raise ObjectMaterialBoundaryError(
            "UNSUPPORTED_MATERIAL_REPRESENTATION", representation
        )
    role_path = _repository_file(
        repository,
        role_authority["path"],
        "role_authority",
    )
    role_sha256 = str(role_authority["sha256"])
    if not _is_sha256(role_sha256) or _sha256(role_path) != role_sha256:
        raise ObjectMaterialBoundaryError(
            "ROLE_AUTHORITY_SHA256_MISMATCH", expected_object_id
        )
    if (
        certificate.source_indexed_mesh_sha256
        != orientation_certificate.source_indexed_mesh_sha256
        or certificate.orientation_certificate_sha256
        != orientation_certificate.canonical_sha256
        or certificate.source_vertex_count
        != orientation_certificate.source_vertex_count
        or certificate.source_face_count != orientation_certificate.source_face_count
    ):
        raise ObjectMaterialBoundaryError(
            "CERTIFICATE_ORIENTATION_BINDING_MISMATCH", expected_object_id
        )
    evidence = ObjectMaterialBoundaryEvidence(
        schema_version=str(document["schema_version"]),
        claim_scope=str(document["claim_scope"]),
        object_id=expected_object_id,
        representation=representation,
        source_asset_path=str(source_asset["path"]),
        source_asset_sha256=expected_source_asset_sha256,
        role_authority_kind=str(role_authority["kind"]),
        role_authority_path=str(role_authority["path"]),
        role_authority_sha256=role_sha256,
        evidence_path=str(path.relative_to(repository)),
        evidence_sha256=_sha256(path),
        certificate=certificate,
        physics_loaded=document["physics_loaded"],  # type: ignore[arg-type]
        collision_or_contact_truth_read=document[
            "collision_or_contact_truth_read"
        ],  # type: ignore[arg-type]
        source_modified=document["source_modified"],  # type: ignore[arg-type]
        formal_material_boundary_eligible=document[
            "formal_material_boundary_eligible"
        ],  # type: ignore[arg-type]
    )
    return evidence


__all__ = [
    "CLAIM_SCOPE",
    "CURRENT_REPRESENTATION",
    "CURRENT_ROLE_KIND",
    "ObjectMaterialBoundaryError",
    "ObjectMaterialBoundaryEvidence",
    "SCHEMA_VERSION",
    "SINGLE_REPRESENTATION",
    "TRANSFER_ROLE_KIND",
    "load_object_material_boundary_evidence",
]
