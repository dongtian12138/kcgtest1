"""Hash-bound aggregate-xacro collision-link roster for CARTS-Grasp.

This module answers only which authored collision meshes must participate in
later proofs.  It does not claim that those triangle soups are embedded
solids, that the arm pose is known, or that terminal PAD faces have a common
source-index role in the collision mesh.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import itertools
import json
import math
import numbers
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET

import yaml


METHOD_ID = "CARTS_AUTHORITATIVE_AGGREGATE_XACRO_COLLISION_LINK_ROSTER_V1"
AUTHORITY_SCOPE = "AGGREGATE_XACRO_COLLISION_MESH_ROSTER_AND_BYTE_LINEAGE_ONLY"
INCLUDE_POLICY = "RELATIVE_DIRECT_INCLUDE_CLOSURE_WITHOUT_XACRO_PROGRAM_ELEMENTS"
MESH_UNIT_POLICY = (
    "ROS_URDF_SI_METRE_CONVENTION_WITH_EXPLICIT_PER_MESH_UNIT_M"
)
MOTION_MISSING_EVIDENCE = (
    "HASH_BOUND_ARM_IK_AND_APPROACH_CLOSURE_LIFT_TRAJECTORY"
)
TERMINAL_PAD_ROLE_REASON = (
    "TERMINAL_COLLISION_STL_AND_VERIFIED_PAD_VISUAL_SOURCE_DO_NOT_SHARE_"
    "SOURCE_FACE_INDICES"
)
SOLID_BOUNDARY_REASON = (
    "SELF_INTERSECTION_NESTING_AND_MATERIAL_OUTWARD_NOT_CERTIFIED"
)
_SCHEMA_VERSION = "carts_collision_roster_v1"
_XACRO_NAMESPACE = "{http://www.ros.org/wiki/xacro}"
_HEX = frozenset("0123456789abcdef")
EXPECTED_AGGREGATE_SOURCE = "src/iiwa_description/urdf/handarm.urdf.xacro"
EXPECTED_INCLUDE_SOURCES = (
    "src/iiwa_description/urdf/iiwa14.xacro",
    "src/iiwa_description/urdf/hand.xacro",
)
EXPECTED_INDEPENDENT_JOINTS = (
    "iiwa_joint_1",
    "iiwa_joint_2",
    "iiwa_joint_3",
    "iiwa_joint_4",
    "iiwa_joint_5",
    "iiwa_joint_6",
    "iiwa_joint_7",
    "f1j1",
    "f1j2",
    "f2j1",
    "f3j2",
)


class CollisionRosterError(ValueError):
    """Raised when roster bytes, XML structure, or declarations disagree."""


class _UniqueKeySafeLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise CollisionRosterError(f"duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise CollisionRosterError(f"{label} must be a string-keyed mapping")
    return value


def _exact_keys(value: Mapping[str, Any], expected: Sequence[str], label: str) -> None:
    if set(value) != set(expected):
        missing = sorted(set(expected) - set(value))
        extra = sorted(set(value) - set(expected))
        raise CollisionRosterError(
            f"{label} keys changed; missing={missing}, extra={extra}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_snapshot(path: Path, label: str) -> tuple[bytes, str]:
    try:
        content = path.read_bytes()
    except OSError as error:
        raise CollisionRosterError(f"{label} bytes cannot be read") from error
    return content, hashlib.sha256(content).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX for character in value)
    )


def _repository_path(root: Path, value: Any, label: str) -> tuple[str, Path]:
    if not isinstance(value, str) or not value or "\\" in value:
        raise CollisionRosterError(f"{label} must be a normalized repository path")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in ("", ".", "..") for part in pure.parts)
        or pure.as_posix() != value
    ):
        raise CollisionRosterError(f"{label} must be normalized and relative")
    try:
        path = (root / Path(*pure.parts)).resolve(strict=True)
        path.relative_to(root)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        raise CollisionRosterError(f"{label} is unavailable or escapes the repository") from error
    if not path.is_file() and not path.is_dir():
        raise CollisionRosterError(f"{label} is not a regular path")
    return value, path


def _verified_file(
    root: Path,
    document: Mapping[str, Any],
    label: str,
) -> "VerifiedRosterFile":
    _exact_keys(document, ("path", "sha256"), label)
    relative, path = _repository_path(root, document["path"], f"{label}.path")
    if not path.is_file() or not _valid_sha256(document["sha256"]):
        raise CollisionRosterError(f"{label} must bind a file and lowercase SHA-256")
    content, actual = _file_snapshot(path, label)
    if actual != document["sha256"]:
        raise CollisionRosterError(f"{label} SHA-256 mismatch")
    return VerifiedRosterFile(relative, path, actual, len(content), content)


def _finite_vector(value: Any, label: str, *, positive: bool = False) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise CollisionRosterError(f"{label} must contain exactly three numbers")
    if any(
        isinstance(item, bool) or not isinstance(item, numbers.Real)
        for item in value
    ):
        raise CollisionRosterError(f"{label} must contain numeric YAML scalars")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise CollisionRosterError(f"{label} must be finite")
    if positive and not all(item > 0.0 for item in result):
        raise CollisionRosterError(f"{label} must be strictly positive")
    return result  # type: ignore[return-value]


def _parse_xml_vector(value: str | None, default: str, label: str) -> tuple[float, float, float]:
    tokens = (default if value is None else value).split()
    if len(tokens) != 3:
        raise CollisionRosterError(f"{label} must contain exactly three numbers")
    try:
        result = tuple(float(token) for token in tokens)
    except ValueError as error:
        raise CollisionRosterError(f"{label} contains invalid XML numbers") from error
    if not all(math.isfinite(item) for item in result):
        raise CollisionRosterError(f"{label} must be finite")
    return result  # type: ignore[return-value]


@dataclass(frozen=True)
class VerifiedRosterFile:
    repository_path: str
    absolute_path: Path
    sha256: str
    byte_count: int
    content_bytes: bytes = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            not _valid_sha256(self.sha256)
            or not isinstance(self.content_bytes, bytes)
            or len(self.content_bytes) != self.byte_count
            or hashlib.sha256(self.content_bytes).hexdigest() != self.sha256
        ):
            raise CollisionRosterError(
                "verified roster file does not match its immutable byte snapshot"
            )


@dataclass(frozen=True)
class CollisionLinkBinding:
    ordinal: int
    link_name: str
    collision_ordinal: int
    mesh_uri: str
    repository_path: str
    absolute_path: Path
    sha256: str
    byte_count: int
    unit: str
    origin_xyz_m: tuple[float, float, float]
    origin_rpy_rad: tuple[float, float, float]
    scale: tuple[float, float, float]
    content_bytes: bytes = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            type(self.ordinal) is not int
            or self.ordinal < 0
            or type(self.collision_ordinal) is not int
            or self.collision_ordinal < 0
            or not self.link_name
            or self.unit != "m"
            or not _valid_sha256(self.sha256)
            or not isinstance(self.content_bytes, bytes)
            or len(self.content_bytes) != self.byte_count
            or hashlib.sha256(self.content_bytes).hexdigest() != self.sha256
        ):
            raise CollisionRosterError(
                "collision link binding is not an immutable SI-metre asset snapshot"
            )


@dataclass(frozen=True, init=False)
class AuthoritativeCollisionLinkRoster:
    aggregate_source: VerifiedRosterFile
    include_sources: tuple[VerifiedRosterFile, ...]
    links: tuple[CollisionLinkBinding, ...]
    excluded_noncollision_links: tuple[str, ...]
    all_self_pairs: tuple[tuple[str, str], ...]
    roster_sha256: str
    package_roots: tuple[tuple[str, str], ...]
    mesh_unit_policy: str
    motion_missing_evidence: str
    fixed_arm_during_local_finger_closure_allowed_only_after_arm_pose_binding: bool
    terminal_pad_role_reason: str
    nearest_surface_or_tolerance_mapping_allowed: bool
    solid_boundary_reason: str
    closed_orientable_positive_volume_is_not_sufficient: bool
    motion_binding_complete: bool
    terminal_pad_role_binding_complete: bool
    solid_boundary_binding_complete: bool
    formal_collision_roster_eligible: bool

    def __init__(self) -> None:
        raise TypeError(
            "AuthoritativeCollisionLinkRoster is created only by its verified loader"
        )

    def __post_init__(self) -> None:
        if (
            len(self.links) != 17
            or len(self.all_self_pairs) != 136
            or self.mesh_unit_policy != MESH_UNIT_POLICY
            or self.motion_missing_evidence != MOTION_MISSING_EVIDENCE
            or self.fixed_arm_during_local_finger_closure_allowed_only_after_arm_pose_binding
            is not True
            or self.terminal_pad_role_reason != TERMINAL_PAD_ROLE_REASON
            or self.nearest_surface_or_tolerance_mapping_allowed is not False
            or self.solid_boundary_reason != SOLID_BOUNDARY_REASON
            or self.closed_orientable_positive_volume_is_not_sufficient is not True
            or self.motion_binding_complete is not False
            or self.terminal_pad_role_binding_complete is not False
            or self.solid_boundary_binding_complete is not False
            or self.formal_collision_roster_eligible is not False
            or not _valid_sha256(self.roster_sha256)
        ):
            raise CollisionRosterError(
                "authoritative roster invariants or claim boundaries changed"
            )

    @property
    def link_names(self) -> tuple[str, ...]:
        return tuple(row.link_name for row in self.links)

    @property
    def audit(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "method_id": METHOD_ID,
                "authority_scope": AUTHORITY_SCOPE,
                "include_policy": INCLUDE_POLICY,
                "mesh_unit_policy": self.mesh_unit_policy,
                "xacro_execution_used": False,
                "collision_link_count": len(self.links),
                "self_pair_count": len(self.all_self_pairs),
                "excluded_noncollision_links": self.excluded_noncollision_links,
                "motion_missing_evidence": self.motion_missing_evidence,
                "terminal_pad_role_reason": self.terminal_pad_role_reason,
                "solid_boundary_reason": self.solid_boundary_reason,
                "motion_binding_complete": self.motion_binding_complete,
                "terminal_pad_role_binding_complete": (
                    self.terminal_pad_role_binding_complete
                ),
                "solid_boundary_binding_complete": (
                    self.solid_boundary_binding_complete
                ),
                "formal_collision_roster_eligible": (
                    self.formal_collision_roster_eligible
                ),
                "roster_sha256": self.roster_sha256,
            }
        )


def build_verified_aggregate_robot_xml(
    roster: AuthoritativeCollisionLinkRoster,
) -> bytes:
    """Assemble the hash-verified iiwa and hand XML without xacro execution."""

    if roster.aggregate_source.repository_path != EXPECTED_AGGREGATE_SOURCE:
        raise CollisionRosterError("aggregate robot source changed")
    include_paths = tuple(row.repository_path for row in roster.include_sources)
    if include_paths != EXPECTED_INCLUDE_SOURCES:
        raise CollisionRosterError("aggregate robot include order changed")
    combined = ET.Element("robot", {"name": "carts_verified_handarm"})
    seen: dict[str, set[str]] = {"link": set(), "joint": set()}
    for source in roster.include_sources:
        try:
            source_root = ET.fromstring(source.content_bytes)
        except ET.ParseError as error:
            raise CollisionRosterError(
                f"verified include XML is invalid: {source.repository_path}"
            ) from error
        for child in source_root:
            if child.tag not in seen:
                continue
            name = child.attrib.get("name", "")
            if not name or name in seen[child.tag]:
                raise CollisionRosterError(
                    f"duplicate or unnamed aggregate element: {child.tag}:{name}"
                )
            seen[child.tag].add(name)
            combined.append(ET.fromstring(ET.tostring(child, encoding="utf-8")))
    return ET.tostring(combined, encoding="utf-8")


def _resolved_package_roots(
    root: Path, document: Mapping[str, Any]
) -> dict[str, tuple[str, Path]]:
    result: dict[str, tuple[str, Path]] = {}
    for package, value in document.items():
        if not package or "/" in package or "\\" in package:
            raise CollisionRosterError("package name is malformed")
        relative, path = _repository_path(root, value, f"package_roots.{package}")
        if not path.is_dir():
            raise CollisionRosterError(f"package root is not a directory: {package}")
        result[package] = (relative, path)
    if not result:
        raise CollisionRosterError("at least one package root is required")
    return result


def _mesh_path_from_uri(
    uri: str,
    package_roots: Mapping[str, tuple[str, Path]],
) -> tuple[str, Path]:
    prefix = "package://"
    if not isinstance(uri, str) or not uri.startswith(prefix):
        raise CollisionRosterError("collision mesh URI must use package://")
    payload = uri[len(prefix):]
    package, separator, relative = payload.partition("/")
    if not separator or package not in package_roots:
        raise CollisionRosterError(f"collision mesh package is unregistered: {uri}")
    pure = PurePosixPath(relative)
    if (
        not pure.parts
        or pure.is_absolute()
        or any(part in ("", ".", "..") for part in pure.parts)
        or pure.as_posix() != relative
    ):
        raise CollisionRosterError(f"collision mesh URI is not normalized: {uri}")
    package_relative, package_root = package_roots[package]
    try:
        path = (package_root / Path(*pure.parts)).resolve(strict=True)
        path.relative_to(package_root)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        raise CollisionRosterError(f"collision mesh URI is unavailable: {uri}") from error
    repository_path = PurePosixPath(package_relative, pure).as_posix()
    return repository_path, path


def _source_rows(
    aggregate: VerifiedRosterFile,
    configured_includes: tuple[VerifiedRosterFile, ...],
    package_roots: Mapping[str, tuple[str, Path]],
) -> tuple[list[dict[str, object]], tuple[str, ...]]:
    try:
        aggregate_root = ET.fromstring(aggregate.content_bytes)
    except ET.ParseError as error:
        raise CollisionRosterError("aggregate xacro cannot be parsed") from error
    if aggregate_root.tag != "robot" or set(aggregate_root.attrib) != {"name"}:
        raise CollisionRosterError("aggregate source must be one named robot root")
    direct_children = list(aggregate_root)
    if not direct_children or any(
        child.tag != _XACRO_NAMESPACE + "include" for child in direct_children
    ):
        raise CollisionRosterError(
            "aggregate source must contain only direct relative xacro includes"
        )
    include_paths: list[Path] = []
    for index, child in enumerate(direct_children):
        if set(child.attrib) != {"filename"}:
            raise CollisionRosterError(
                "aggregate include must contain only a filename attribute"
            )
        filename = child.attrib.get("filename")
        if (
            filename is None
            or "\\" in filename
            or "$" in filename
            or PurePosixPath(filename).is_absolute()
            or any(part in ("", ".", "..") for part in PurePosixPath(filename).parts)
        ):
            raise CollisionRosterError("aggregate include is not a direct relative file")
        include_path = (aggregate.absolute_path.parent / filename).resolve()
        include_paths.append(include_path)
        if index >= len(configured_includes) or (
            configured_includes[index].absolute_path != include_path
        ):
            raise CollisionRosterError("configured include order differs from aggregate XML")
    if tuple(include_paths) != tuple(row.absolute_path for row in configured_includes):
        raise CollisionRosterError("configured include closure is incomplete")

    rows: list[dict[str, object]] = []
    all_links: list[str] = []
    for source in configured_includes:
        try:
            child_root = ET.fromstring(source.content_bytes)
        except ET.ParseError as error:
            raise CollisionRosterError("included xacro cannot be parsed") from error
        if child_root.tag != "robot" or set(child_root.attrib) != {"name"}:
            raise CollisionRosterError(
                "included source must be one named robot root"
            )
        if any(
            isinstance(element.tag, str)
            and element.tag.startswith(_XACRO_NAMESPACE)
            for element in child_root.iter()
        ):
            raise CollisionRosterError(
                "included sources contain executable xacro elements"
            )
        direct_links = child_root.findall("link")
        if len(direct_links) != len(child_root.findall(".//link")):
            raise CollisionRosterError(
                "included source contains a non-direct nested link"
            )
        for link in direct_links:
            if set(link.attrib) != {"name"} or any(
                child.tag not in {"inertial", "visual", "collision"}
                for child in list(link)
            ):
                raise CollisionRosterError(
                    "included link has unsupported attributes or children"
                )
            name = link.attrib.get("name", "")
            if not name or name in all_links:
                raise CollisionRosterError("included link names are empty or repeated")
            all_links.append(name)
            collisions = link.findall("collision")
            if len(collisions) != len(link.findall(".//collision")):
                raise CollisionRosterError(
                    "collision elements must be direct children of their link"
                )
            if len(collisions) > 1:
                raise CollisionRosterError(
                    "V1 roster requires at most one authored collision per link"
                )
            for collision_ordinal, collision in enumerate(collisions):
                if collision.attrib or any(
                    child.tag not in {"origin", "geometry", "material"}
                    for child in list(collision)
                ):
                    raise CollisionRosterError(
                        "collision has unsupported attributes or children"
                    )
                geometries = collision.findall("geometry")
                origins = collision.findall("origin")
                materials = collision.findall("material")
                if (
                    len(geometries) != 1
                    or len(origins) > 1
                    or len(materials) > 1
                    or len(geometries) != len(collision.findall(".//geometry"))
                    or len(origins) != len(collision.findall(".//origin"))
                ):
                    raise CollisionRosterError(
                        "collision must have exactly one direct geometry and at most one origin/material"
                    )
                geometry = geometries[0]
                if geometry.attrib or len(list(geometry)) != 1:
                    raise CollisionRosterError("collision geometry must contain one mesh")
                mesh = list(geometry)[0]
                if mesh.tag != "mesh" or set(mesh.attrib) not in (
                    {"filename"},
                    {"filename", "scale"},
                ):
                    raise CollisionRosterError("collision geometry must be a mesh URI")
                uri = mesh.attrib.get("filename")
                if uri is None:
                    raise CollisionRosterError("collision mesh filename is missing")
                repository_path, mesh_path = _mesh_path_from_uri(uri, package_roots)
                mesh_content, mesh_sha256 = _file_snapshot(
                    mesh_path, f"collision mesh {name}"
                )
                origin = origins[0] if origins else None
                if origin is not None and set(origin.attrib) - {"xyz", "rpy"}:
                    raise CollisionRosterError("collision origin has unsupported fields")
                rows.append(
                    {
                        "ordinal": len(rows),
                        "link_name": name,
                        "collision_ordinal": collision_ordinal,
                        "mesh_uri": uri,
                        "repository_path": repository_path,
                        "absolute_path": mesh_path,
                        "sha256": mesh_sha256,
                        "byte_count": len(mesh_content),
                        "content_bytes": mesh_content,
                        "origin_xyz_m": _parse_xml_vector(
                            None if origin is None else origin.attrib.get("xyz"),
                            "0 0 0",
                            f"{name}.collision.origin.xyz",
                        ),
                        "origin_rpy_rad": _parse_xml_vector(
                            None if origin is None else origin.attrib.get("rpy"),
                            "0 0 0",
                            f"{name}.collision.origin.rpy",
                        ),
                        "scale": _parse_xml_vector(
                            mesh.attrib.get("scale"),
                            "1 1 1",
                            f"{name}.collision.mesh.scale",
                        ),
                    }
                )
    noncollision = tuple(name for name in all_links if name not in {row["link_name"] for row in rows})
    return rows, noncollision


def _canonical_roster_document(
    aggregate: VerifiedRosterFile,
    includes: tuple[VerifiedRosterFile, ...],
    links: tuple[CollisionLinkBinding, ...],
    excluded: tuple[str, ...],
    all_pairs: tuple[tuple[str, str], ...],
    package_roots: tuple[tuple[str, str], ...],
    semantics: Mapping[str, object],
    states: Mapping[str, bool],
) -> dict[str, object]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "authority_scope": AUTHORITY_SCOPE,
        "include_policy": INCLUDE_POLICY,
        "mesh_unit_policy": MESH_UNIT_POLICY,
        "xacro_execution_used": False,
        "package_roots": {name: path for name, path in package_roots},
        "aggregate_source": {
            "path": aggregate.repository_path,
            "sha256": aggregate.sha256,
            "byte_count": aggregate.byte_count,
        },
        "include_sources": [
            {
                "path": row.repository_path,
                "sha256": row.sha256,
                "byte_count": row.byte_count,
            }
            for row in includes
        ],
        "links": [
            {
                "ordinal": row.ordinal,
                "link_name": row.link_name,
                "collision_ordinal": row.collision_ordinal,
                "mesh_uri": row.mesh_uri,
                "repository_path": row.repository_path,
                "sha256": row.sha256,
                "byte_count": row.byte_count,
                "unit": row.unit,
                "origin_xyz_m": list(row.origin_xyz_m),
                "origin_rpy_rad": list(row.origin_rpy_rad),
                "scale": list(row.scale),
            }
            for row in links
        ],
        "excluded_noncollision_links": list(excluded),
        "all_self_pairs": [list(pair) for pair in all_pairs],
        **semantics,
        **states,
    }


def _new_authoritative_roster(
    **values: object,
) -> AuthoritativeCollisionLinkRoster:
    result = object.__new__(AuthoritativeCollisionLinkRoster)
    for name in AuthoritativeCollisionLinkRoster.__dataclass_fields__:
        object.__setattr__(result, name, values[name])
    result.__post_init__()
    return result


def load_authoritative_collision_link_roster(
    contract_path: Path | str,
    *,
    repository_root: Path | str,
) -> AuthoritativeCollisionLinkRoster:
    """Rebuild and verify the exact aggregate-xacro collision mesh roster."""

    root = Path(repository_root).resolve()
    path = Path(contract_path)
    if not path.is_absolute():
        path = (root / path).resolve()
    try:
        document = _mapping(
            yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeySafeLoader),
            "collision roster contract",
        )
    except (OSError, yaml.YAMLError) as error:
        raise CollisionRosterError("collision roster YAML cannot be loaded") from error
    _exact_keys(
        document,
        (
            "schema_version",
            "method_id",
            "authority_scope",
            "xacro_execution_used",
            "include_policy",
            "mesh_unit_policy",
            "aggregate_source",
            "include_sources",
            "package_roots",
            "required_collision_link_count",
            "required_self_pair_count",
            "excluded_noncollision_links",
            "links",
            "motion_binding",
            "terminal_pad_role_binding",
            "solid_boundary_binding",
            "formal_collision_roster_eligible",
        ),
        "collision roster contract",
    )
    if (
        document["schema_version"] != _SCHEMA_VERSION
        or document["method_id"] != METHOD_ID
        or document["authority_scope"] != AUTHORITY_SCOPE
        or document["xacro_execution_used"] is not False
        or document["include_policy"] != INCLUDE_POLICY
        or document["mesh_unit_policy"] != MESH_UNIT_POLICY
    ):
        raise CollisionRosterError("collision roster method identity changed")
    aggregate = _verified_file(
        root, _mapping(document["aggregate_source"], "aggregate_source"), "aggregate_source"
    )
    raw_includes = document["include_sources"]
    if not isinstance(raw_includes, list) or not raw_includes:
        raise CollisionRosterError("include_sources must be a non-empty sequence")
    includes = tuple(
        _verified_file(root, _mapping(row, f"include_sources[{index}]"), f"include_sources[{index}]")
        for index, row in enumerate(raw_includes)
    )
    package_roots = _resolved_package_roots(
        root, _mapping(document["package_roots"], "package_roots")
    )
    derived_rows, derived_noncollision = _source_rows(
        aggregate, includes, package_roots
    )
    used_packages = {
        str(row["mesh_uri"])[len("package://"):].partition("/")[0]
        for row in derived_rows
    }
    if used_packages != set(package_roots):
        raise CollisionRosterError(
            "package_roots must exactly equal the packages used by collision meshes"
        )
    canonical_package_roots = tuple(
        sorted(
            (package, relative)
            for package, (relative, _path) in package_roots.items()
        )
    )
    raw_links = document["links"]
    if not isinstance(raw_links, list) or len(raw_links) != len(derived_rows):
        raise CollisionRosterError("declared links differ in length from aggregate XML")
    links: list[CollisionLinkBinding] = []
    link_keys = (
        "ordinal",
        "link_name",
        "collision_ordinal",
        "mesh_uri",
        "repository_path",
        "sha256",
        "unit",
        "origin_xyz_m",
        "origin_rpy_rad",
        "scale",
    )
    for index, (raw_value, derived) in enumerate(zip(raw_links, derived_rows)):
        raw = _mapping(raw_value, f"links[{index}]")
        _exact_keys(raw, link_keys, f"links[{index}]")
        if (
            type(raw["ordinal"]) is not int
            or type(raw["collision_ordinal"]) is not int
            or not isinstance(raw["link_name"], str)
            or not isinstance(raw["mesh_uri"], str)
            or not isinstance(raw["repository_path"], str)
            or not _valid_sha256(raw["sha256"])
        ):
            raise CollisionRosterError(
                f"links[{index}] scalar fields have invalid YAML types"
            )
        if raw["unit"] != "m":
            raise CollisionRosterError(
                f"links[{index}].unit must match the registered SI-metre roster"
            )
        declared_vector_values = {
            "origin_xyz_m": _finite_vector(raw["origin_xyz_m"], f"links[{index}].origin_xyz_m"),
            "origin_rpy_rad": _finite_vector(raw["origin_rpy_rad"], f"links[{index}].origin_rpy_rad"),
            "scale": _finite_vector(raw["scale"], f"links[{index}].scale", positive=True),
        }
        scalar_names = (
            "ordinal",
            "link_name",
            "collision_ordinal",
            "mesh_uri",
            "repository_path",
            "sha256",
        )
        if any(raw[name] != derived[name] for name in scalar_names) or any(
            declared_vector_values[name] != derived[name]
            for name in declared_vector_values
        ):
            raise CollisionRosterError(
                f"links[{index}] differs from aggregate XML or mesh bytes"
            )
        links.append(
            CollisionLinkBinding(
                ordinal=index,
                link_name=str(raw["link_name"]),
                collision_ordinal=int(raw["collision_ordinal"]),
                mesh_uri=str(raw["mesh_uri"]),
                repository_path=str(raw["repository_path"]),
                absolute_path=derived["absolute_path"],  # type: ignore[arg-type]
                sha256=str(raw["sha256"]),
                byte_count=int(derived["byte_count"]),
                unit=str(raw["unit"]),
                origin_xyz_m=declared_vector_values["origin_xyz_m"],
                origin_rpy_rad=declared_vector_values["origin_rpy_rad"],
                scale=declared_vector_values["scale"],
                content_bytes=derived["content_bytes"],  # type: ignore[arg-type]
            )
        )
    links_tuple = tuple(links)
    if len({row.link_name for row in links_tuple}) != len(links_tuple):
        raise CollisionRosterError("V1 collision links must be one mesh per unique link")
    expected_noncollision = document["excluded_noncollision_links"]
    if (
        not isinstance(expected_noncollision, list)
        or any(
            not isinstance(name, str) or not name
            for name in expected_noncollision
        )
        or tuple(expected_noncollision) != derived_noncollision
        or len(set(expected_noncollision)) != len(expected_noncollision)
    ):
        raise CollisionRosterError("excluded noncollision links differ from aggregate XML")
    link_names_sorted = tuple(sorted(row.link_name for row in links_tuple))
    all_pairs = tuple(itertools.combinations(link_names_sorted, 2))
    if (
        type(document["required_collision_link_count"]) is not int
        or type(document["required_self_pair_count"]) is not int
        or document["required_collision_link_count"] != len(links_tuple)
        or document["required_self_pair_count"] != len(all_pairs)
        or len(links_tuple) != 17
        or len(all_pairs) != 136
    ):
        raise CollisionRosterError("registered 17-link/136-pair cardinality changed")

    motion = _mapping(document["motion_binding"], "motion_binding")
    terminal = _mapping(
        document["terminal_pad_role_binding"], "terminal_pad_role_binding"
    )
    solid = _mapping(document["solid_boundary_binding"], "solid_boundary_binding")
    _exact_keys(
        motion,
        (
            "complete",
            "missing_evidence",
            "fixed_arm_during_local_finger_closure_allowed_only_after_arm_pose_binding",
        ),
        "motion_binding",
    )
    _exact_keys(
        terminal,
        ("complete", "reason", "nearest_surface_or_tolerance_mapping_allowed"),
        "terminal_pad_role_binding",
    )
    _exact_keys(
        solid,
        ("complete", "reason", "closed_orientable_positive_volume_is_not_sufficient"),
        "solid_boundary_binding",
    )
    if (
        motion["complete"] is not False
        or motion["missing_evidence"] != MOTION_MISSING_EVIDENCE
        or motion[
            "fixed_arm_during_local_finger_closure_allowed_only_after_arm_pose_binding"
        ]
        is not True
        or terminal["complete"] is not False
        or terminal["reason"] != TERMINAL_PAD_ROLE_REASON
        or solid["complete"] is not False
        or solid["reason"] != SOLID_BOUNDARY_REASON
        or terminal["nearest_surface_or_tolerance_mapping_allowed"] is not False
        or solid["closed_orientable_positive_volume_is_not_sufficient"] is not True
        or document["formal_collision_roster_eligible"] is not False
    ):
        raise CollisionRosterError("incomplete formal bindings cannot be upgraded")
    states = {
        "motion_binding_complete": False,
        "terminal_pad_role_binding_complete": False,
        "solid_boundary_binding_complete": False,
        "formal_collision_roster_eligible": False,
    }
    semantics: dict[str, object] = {
        "motion_missing_evidence": MOTION_MISSING_EVIDENCE,
        "fixed_arm_during_local_finger_closure_allowed_only_after_arm_pose_binding": True,
        "terminal_pad_role_reason": TERMINAL_PAD_ROLE_REASON,
        "nearest_surface_or_tolerance_mapping_allowed": False,
        "solid_boundary_reason": SOLID_BOUNDARY_REASON,
        "closed_orientable_positive_volume_is_not_sufficient": True,
    }
    canonical = _canonical_roster_document(
        aggregate,
        includes,
        links_tuple,
        tuple(expected_noncollision),
        all_pairs,
        canonical_package_roots,
        semantics,
        states,
    )
    canonical_bytes = json.dumps(
        canonical,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    roster_sha256 = hashlib.sha256(canonical_bytes).hexdigest()
    return _new_authoritative_roster(
        aggregate_source=aggregate,
        include_sources=includes,
        links=links_tuple,
        excluded_noncollision_links=tuple(expected_noncollision),
        all_self_pairs=all_pairs,
        roster_sha256=roster_sha256,
        package_roots=canonical_package_roots,
        mesh_unit_policy=MESH_UNIT_POLICY,
        motion_missing_evidence=MOTION_MISSING_EVIDENCE,
        fixed_arm_during_local_finger_closure_allowed_only_after_arm_pose_binding=True,
        terminal_pad_role_reason=TERMINAL_PAD_ROLE_REASON,
        nearest_surface_or_tolerance_mapping_allowed=False,
        solid_boundary_reason=SOLID_BOUNDARY_REASON,
        closed_orientable_positive_volume_is_not_sufficient=True,
        motion_binding_complete=False,
        terminal_pad_role_binding_complete=False,
        solid_boundary_binding_complete=False,
        formal_collision_roster_eligible=False,
    )


__all__ = [
    "AUTHORITY_SCOPE",
    "AuthoritativeCollisionLinkRoster",
    "CollisionLinkBinding",
    "CollisionRosterError",
    "EXPECTED_AGGREGATE_SOURCE",
    "EXPECTED_INCLUDE_SOURCES",
    "EXPECTED_INDEPENDENT_JOINTS",
    "INCLUDE_POLICY",
    "METHOD_ID",
    "VerifiedRosterFile",
    "build_verified_aggregate_robot_xml",
    "load_authoritative_collision_link_roster",
]
