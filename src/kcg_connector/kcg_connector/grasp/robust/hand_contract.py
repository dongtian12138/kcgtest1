"""Strict, provenance-preserving hand contract for CARTS-Grasp.

This module is intentionally narrower than :mod:`hand_model`.  ``hand_model``
implements kinematics and accepts convenient aliases for synthetic tests;
this module is the evidence boundary for the repository YAML.  It accepts no
missing fields or aliases, verifies every referenced byte stream, and refuses
to turn an uncalibrated user-provided force capability into physical truth.

The verified PAD mapping passed to ``ThreeFingerHandModel`` contains the whole
blue PAD meshes only.  Historical contact-face identifiers and the legacy
``0.90`` closure-alignment rule are neither loaded nor emitted.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from .hand_model import HandModelError, ThreeFingerHandModel


_SCHEMA_VERSION = "carts_hand_contact_v1"
_METHOD = "CARTS-Grasp"
_PAD_AUTHORITY = "USER_CONFIRMED_BLUE_PAD_BODY"
_MANIFEST_SEMANTIC_AUTHORITY = "USER_CONFIRMED_HAND_GEOMETRY_SEMANTICS"
_MANIFEST_SOURCE_AUTHORITY = "AUTHORED_HAND_STL_GEOMETRY"
_LEGACY_PAD_MANIFEST_SCHEMA = "D38999_BLUE_PAD_SDF_SOURCE_ASSET_V1"
_EXACT_PAD_MANIFEST_SCHEMA = "CARTS_EXACT_SOURCE_TERMINAL_PAD_V1"
_FORCE_SOURCE = "USER_PROVIDED_HARDWARE_CAPABILITY_PENDING_CALIBRATION"
_FORCE_ROLE = "OPTIMIZATION_UPPER_BOUND_NOT_BINARY_GRASP_THRESHOLD"
_CLOSURE_METHOD = "PRE_REGISTERED_SEQUENTIAL_FINGER_EXCLUSIVE_JOINT_PATHS"
_CLOSURE_SOURCE = "HAND_URDF_CHAIN_AND_PRE_REGISTERED_MECHANICAL_PROTOCOL"
_CLOSURE_NORMALIZATION = "MAXIMUM_ABSOLUTE_JOINT_WEIGHT_EQUALS_ONE"
_SHARED_JOINT_ROLE = "PREGRASP_CONFIGURATION_ONLY_NOT_CLOSURE"
OBJECT_CONTACT_NORMAL_POLICY = (
    "MOTION_ORIENTED_TWO_SIDED_OBJECT_NORMAL_LINE_OPPOSES_CERTIFIED_"
    "TRANSVERSE_PAD_CLOSING_VELOCITY_V1"
)
PAD_SURFACE_NORMAL_POLICY = (
    "POSITIVE_HASH_BOUND_PAD_SOURCE_WINDING_NORMAL_DOT_PAD_CLOSING_"
    "VELOCITY_POSITIVE_UP_TO_FLOAT_ERROR"
)
PAD_TRIANGLE_WINDING_SOURCE = (
    "HASH_BOUND_AUTHORED_HAND_STL_VIA_VERIFIED_SOURCE_MANIFEST"
)
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")

_EXPECTED_PADS = MappingProxyType(
    {
        "finger_1_pad": ("f1", "f1Link3"),
        "finger_2_pad": ("f2", "f2Link2"),
        "finger_3_pad": ("f3", "f3Link3"),
    }
)
_TRUTH_FIELDS = (
    "ground_truth_object_pose_allowed",
    "collision_name_allowed",
    "physx_contact_point_allowed",
    "physx_contact_normal_allowed",
    "semantic_contact_role_allowed",
)


class HandContractError(ValueError):
    """Raised when the CARTS hand evidence contract is incomplete or polluted."""


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader, node: yaml.nodes.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise HandContractError("YAML mapping keys must be scalar and hashable") from exc
        if duplicate:
            raise HandContractError(f"duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HandContractError(f"{label} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise HandContractError(f"{label} keys must be strings")
    return value


def _exact_keys(value: Mapping[str, Any], expected: Sequence[str], label: str) -> None:
    expected_set = set(expected)
    missing = sorted(expected_set - set(value))
    extra = sorted(set(value) - expected_set)
    if missing or extra:
        raise HandContractError(
            f"{label} schema mismatch; missing={missing}, extra={extra}"
        )


def _exact_string(value: Any, expected: str, label: str) -> str:
    if not isinstance(value, str) or value != expected:
        raise HandContractError(f"{label} must be exactly {expected!r}")
    return value


def _required_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise HandContractError(f"{label} must be a non-empty string")
    return value


def _exact_bool(value: Any, expected: bool, label: str) -> bool:
    if type(value) is not bool or value is not expected:
        raise HandContractError(f"{label} must be exactly {expected}")
    return value


def _positive_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HandContractError(f"{label} must be a finite positive number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise HandContractError(f"{label} must be a finite positive number")
    return parsed


def _finite_vector3(value: Any, label: str) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise HandContractError(f"{label} must contain exactly three finite numbers")
    if len(value) != 3:
        raise HandContractError(f"{label} must contain exactly three finite numbers")
    parsed: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise HandContractError(f"{label} must contain exactly three finite numbers")
        converted = float(item)
        if not math.isfinite(converted):
            raise HandContractError(f"{label} must contain exactly three finite numbers")
        parsed.append(converted)
    return parsed[0], parsed[1], parsed[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class VerifiedFileReference:
    """Repository-relative file whose bytes match an explicit SHA-256."""

    repository_relative_path: str
    absolute_path: Path
    sha256: str
    byte_count: int


def _verified_repository_file(
    repository_root: Path,
    reference: Any,
    expected_sha256: Any,
    label: str,
) -> VerifiedFileReference:
    relative_text = _required_string(reference, f"{label} path")
    if "\\" in relative_text:
        raise HandContractError(f"{label} path must use repository-relative POSIX syntax")
    pure = PurePosixPath(relative_text)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in ("", ".", "..") for part in pure.parts)
        or pure.as_posix() != relative_text
    ):
        raise HandContractError(f"{label} path must be normalized and repository-relative")

    declared_hash = _required_string(expected_sha256, f"{label} SHA-256")
    if _HEX_SHA256.fullmatch(declared_hash) is None:
        raise HandContractError(f"{label} SHA-256 must be 64 lowercase hexadecimal digits")

    try:
        absolute = (repository_root / Path(*pure.parts)).resolve(strict=True)
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        raise HandContractError(f"{label} referenced file is unavailable: {relative_text}") from exc
    try:
        absolute.relative_to(repository_root)
    except ValueError as exc:
        raise HandContractError(f"{label} path resolves outside repository root") from exc
    if not absolute.is_file():
        raise HandContractError(f"{label} reference is not a regular file")

    actual_hash = _sha256(absolute)
    if actual_hash != declared_hash:
        raise HandContractError(
            f"{label} SHA-256 mismatch: declared={declared_hash}, actual={actual_hash}"
        )
    return VerifiedFileReference(
        repository_relative_path=relative_text,
        absolute_path=absolute,
        sha256=actual_hash,
        byte_count=absolute.stat().st_size,
    )


def _load_unique_yaml(path: Path) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = yaml.load(stream, Loader=_UniqueKeySafeLoader)
    except (OSError, yaml.YAMLError, UnicodeError) as exc:
        raise HandContractError(f"cannot read hand YAML contract: {path}") from exc
    return _mapping(value, "hand contract root")


def _unique_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise HandContractError(f"duplicate JSON key in PAD source manifest: {key}")
        result[key] = value
    return result


def _load_source_manifest(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_unique_json_pairs
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HandContractError("PAD source manifest is not valid unique-key JSON") from exc
    return _mapping(value, "PAD source manifest")


def _immutable_array(value: np.ndarray, dtype: np.dtype[Any]) -> np.ndarray:
    """Return a C-order view backed by immutable ``bytes`` storage."""

    canonical = np.ascontiguousarray(value, dtype=dtype)
    immutable = np.frombuffer(canonical.tobytes(order="C"), dtype=canonical.dtype)
    immutable = immutable.reshape(canonical.shape)
    immutable.flags.writeable = False
    return immutable


def _inspect_pad_npz(path: Path, label: str) -> tuple[np.ndarray, np.ndarray]:
    """Inspect only whole-mesh arrays; historical semantic face IDs stay unused."""

    try:
        with np.load(path, allow_pickle=False) as archive:
            if "points_local_m" not in archive.files or "faces" not in archive.files:
                raise HandContractError(
                    f"{label} NPZ must contain points_local_m and faces"
                )
            points = np.asarray(archive["points_local_m"])
            faces = np.asarray(archive["faces"])
    except HandContractError:
        raise
    except (OSError, ValueError, KeyError) as exc:
        raise HandContractError(f"{label} NPZ cannot be loaded without pickle") from exc

    if (
        points.ndim != 2
        or points.shape[1:] != (3,)
        or points.shape[0] == 0
        or not np.issubdtype(points.dtype, np.floating)
        or not np.all(np.isfinite(points))
    ):
        raise HandContractError(f"{label} points_local_m must be a finite non-empty Nx3 float array")
    if (
        faces.ndim != 2
        or faces.shape[1:] != (3,)
        or faces.shape[0] == 0
        or not np.issubdtype(faces.dtype, np.integer)
    ):
        raise HandContractError(f"{label} faces must be a non-empty Mx3 integer array")
    if int(np.min(faces)) < 0 or int(np.max(faces)) >= points.shape[0]:
        raise HandContractError(f"{label} faces index vertices outside points_local_m")
    return (
        _immutable_array(points, np.dtype(np.float64)),
        _immutable_array(faces, np.dtype(np.int64)),
    )


@dataclass(frozen=True)
class VerifiedPad:
    name: str
    finger_name: str
    link_name: str
    origin_xyz_m: tuple[float, float, float]
    origin_rpy_rad: tuple[float, float, float]
    mesh: VerifiedFileReference
    coordinate_frame: str
    unit: str
    normal_force_capacity_n: float
    points_local_m: np.ndarray
    faces: np.ndarray

    @property
    def vertex_count(self) -> int:
        return int(self.points_local_m.shape[0])

    @property
    def triangle_count(self) -> int:
        return int(self.faces.shape[0])


@dataclass(frozen=True)
class CARTSHandContract:
    """Verified static hand evidence and explicit hand-model construction input."""

    contract_path: Path
    repository_root: Path
    urdf: VerifiedFileReference
    source_manifest: VerifiedFileReference
    base_link: str
    pads: tuple[VerifiedPad, VerifiedPad, VerifiedPad]
    closure_actuation_method: str
    closure_actuation_source: str
    closure_actuation_normalization: str
    shared_independent_joint_role: str
    object_contact_normal_policy: str
    pad_surface_normal_policy: str
    pad_triangle_winding_source: str
    pad_triangle_winding_consistency_required: bool
    closure_actuation_rows: tuple[
        tuple[str, tuple[tuple[str, float], ...]],
        tuple[str, tuple[tuple[str, float], ...]],
        tuple[str, tuple[tuple[str, float], ...]],
    ]
    force_capacity_value_source: str
    force_capacity_role: str
    hardware_authorized: bool
    physical_calibration_complete: bool
    simulator_readback_complete: bool
    dynamic_use_allowed: bool
    online_truth_firewall: tuple[tuple[str, bool], ...]

    @property
    def schema_version(self) -> str:
        return _SCHEMA_VERSION

    @property
    def method(self) -> str:
        return _METHOD

    @property
    def pad_by_name(self) -> Mapping[str, VerifiedPad]:
        return MappingProxyType({pad.name: pad for pad in self.pads})

    @property
    def truth_firewall_all_false(self) -> bool:
        return all(value is False for _name, value in self.online_truth_firewall)

    @property
    def dynamic_validation_complete(self) -> bool:
        """This V1 static contract intentionally cannot claim dynamic validation."""

        return False

    def to_hand_model_pad_contract(self) -> dict[str, Any]:
        """Return explicit inputs accepted by ``ThreeFingerHandModel``.

        ``mesh_scale`` is exactly one because the verified footprint unit is
        metre.  No face subset, alignment score, or inferred contact normal is
        included.
        """

        return {
            "schema_version": "carts_verified_whole_pad_geometry_v1",
            "pads": {
                pad.name: {
                    "name": pad.name,
                    "finger_name": pad.finger_name,
                    "link_name": pad.link_name,
                    "origin": {
                        "xyz_m": list(pad.origin_xyz_m),
                        "rpy_rad": list(pad.origin_rpy_rad),
                    },
                    "footprint": {
                        "kind": "mesh",
                        "mesh_uri": pad.mesh.repository_relative_path,
                        "mesh_scale": [1.0, 1.0, 1.0],
                    },
                    "contact_normal_pad": None,
                    "normal_force_capacity_n": pad.normal_force_capacity_n,
                }
                for pad in self.pads
            },
        }

    def build_hand_model(self) -> ThreeFingerHandModel:
        """Construct the existing kinematic model from already verified inputs."""

        try:
            return ThreeFingerHandModel.from_urdf(
                self.urdf.absolute_path,
                pad_geometry_contract=self.to_hand_model_pad_contract(),
                base_link=self.base_link,
            )
        except HandModelError as exc:
            raise HandContractError("verified contract is incompatible with hand URDF") from exc

    def closing_actuation_directions_unit(
        self, hand_model: ThreeFingerHandModel
    ) -> np.ndarray:
        """Build the declared finger-exclusive closure matrix in URDF order."""

        independent_names = tuple(hand_model.independent_joint_names)
        source_sets: dict[str, set[str]] = {}
        for finger_name, finger in hand_model.fingers.items():
            sources: set[str] = set()
            for joint_name in finger.joint_names:
                source_name = joint_name
                visited: set[str] = set()
                while hand_model.joints[source_name].mimic is not None:
                    if source_name in visited:
                        raise HandContractError("URDF mimic chain contains a cycle")
                    visited.add(source_name)
                    source_name = hand_model.joints[source_name].mimic.source_joint
                    if source_name not in hand_model.joints:
                        raise HandContractError("URDF mimic source is missing")
                if source_name in independent_names:
                    sources.add(source_name)
            source_sets[finger_name] = sources

        matrix = np.zeros((len(self.closure_actuation_rows), len(independent_names)))
        used_joints: set[str] = set()
        for row_index, (finger_name, weights) in enumerate(
            self.closure_actuation_rows
        ):
            other_sources = set().union(
                *(
                    sources
                    for other, sources in source_sets.items()
                    if other != finger_name
                )
            )
            exclusive_sources = source_sets[finger_name] - other_sources
            for joint_name, weight in weights:
                if joint_name not in exclusive_sources:
                    raise HandContractError(
                        f"closure_actuation.rows.{finger_name} uses shared or foreign "
                        f"joint {joint_name!r}"
                    )
                if joint_name in used_joints:
                    raise HandContractError(
                        "closure actuation joint supports must not overlap"
                    )
                used_joints.add(joint_name)
                matrix[row_index, independent_names.index(joint_name)] = weight
        matrix = np.frombuffer(
            np.ascontiguousarray(matrix).tobytes(order="C"), dtype=np.float64
        ).reshape(matrix.shape)
        matrix.setflags(write=False)
        return matrix


def _parse_closure_actuation(
    value: Any,
) -> tuple[
    str,
    str,
    str,
    str,
    tuple[
        tuple[str, tuple[tuple[str, float], ...]],
        tuple[str, tuple[tuple[str, float], ...]],
        tuple[str, tuple[tuple[str, float], ...]],
    ],
]:
    closure = _mapping(value, "closure_actuation")
    _exact_keys(
        closure,
        (
            "method",
            "source",
            "normalization",
            "shared_independent_joint_role",
            "rows",
        ),
        "closure_actuation",
    )
    method = _exact_string(
        closure["method"], _CLOSURE_METHOD, "closure_actuation.method"
    )
    source = _exact_string(
        closure["source"], _CLOSURE_SOURCE, "closure_actuation.source"
    )
    normalization = _exact_string(
        closure["normalization"],
        _CLOSURE_NORMALIZATION,
        "closure_actuation.normalization",
    )
    shared_role = _exact_string(
        closure["shared_independent_joint_role"],
        _SHARED_JOINT_ROLE,
        "closure_actuation.shared_independent_joint_role",
    )
    rows = _mapping(closure["rows"], "closure_actuation.rows")
    finger_order = ("f1", "f2", "f3")
    _exact_keys(rows, finger_order, "closure_actuation.rows")
    parsed_rows: list[tuple[str, tuple[tuple[str, float], ...]]] = []
    for finger_name in finger_order:
        row = _mapping(rows[finger_name], f"closure_actuation.rows.{finger_name}")
        _exact_keys(
            row,
            ("joint_weights",),
            f"closure_actuation.rows.{finger_name}",
        )
        weights = _mapping(
            row["joint_weights"],
            f"closure_actuation.rows.{finger_name}.joint_weights",
        )
        if not weights:
            raise HandContractError(
                f"closure_actuation.rows.{finger_name}.joint_weights cannot be empty"
            )
        parsed_weights: list[tuple[str, float]] = []
        for joint_name in sorted(weights):
            raw_weight = weights[joint_name]
            if isinstance(raw_weight, bool) or not isinstance(
                raw_weight, (int, float)
            ):
                raise HandContractError("closure actuation weights must be finite numbers")
            weight = float(raw_weight)
            if not math.isfinite(weight) or weight == 0.0:
                raise HandContractError(
                    "closure actuation weights must be finite and non-zero"
                )
            parsed_weights.append((joint_name, weight))
        if max(abs(weight) for _joint, weight in parsed_weights) != 1.0:
            raise HandContractError(
                f"closure_actuation.rows.{finger_name} violates max-absolute normalization"
            )
        parsed_rows.append((finger_name, tuple(parsed_weights)))
    return (
        method,
        source,
        normalization,
        shared_role,
        tuple(parsed_rows),  # type: ignore[return-value]
    )


def _parse_pad(
    name: str,
    value: Any,
    repository_root: Path,
) -> VerifiedPad:
    row = _mapping(value, f"pads.{name}")
    _exact_keys(
        row,
        (
            "finger_name",
            "link_name",
            "origin",
            "footprint",
            "contact_normal_pad",
            "normal_force_capacity_n",
        ),
        f"pads.{name}",
    )
    expected_finger, expected_link = _EXPECTED_PADS[name]
    _exact_string(row["finger_name"], expected_finger, f"pads.{name}.finger_name")
    _exact_string(row["link_name"], expected_link, f"pads.{name}.link_name")

    origin = _mapping(row["origin"], f"pads.{name}.origin")
    _exact_keys(origin, ("xyz_m", "rpy_rad"), f"pads.{name}.origin")
    xyz = _finite_vector3(origin["xyz_m"], f"pads.{name}.origin.xyz_m")
    rpy = _finite_vector3(origin["rpy_rad"], f"pads.{name}.origin.rpy_rad")

    footprint = _mapping(row["footprint"], f"pads.{name}.footprint")
    _exact_keys(
        footprint,
        ("kind", "mesh_uri", "mesh_sha256", "coordinate_frame", "unit"),
        f"pads.{name}.footprint",
    )
    _exact_string(footprint["kind"], "mesh", f"pads.{name}.footprint.kind")
    _exact_string(
        footprint["coordinate_frame"], expected_link, f"pads.{name}.footprint.coordinate_frame"
    )
    _exact_string(footprint["unit"], "m", f"pads.{name}.footprint.unit")
    mesh = _verified_repository_file(
        repository_root,
        footprint["mesh_uri"],
        footprint["mesh_sha256"],
        f"pads.{name}.footprint",
    )
    if row["contact_normal_pad"] is not None:
        raise HandContractError(
            f"pads.{name}.contact_normal_pad must remain null; normals come from closing kinematics"
        )
    force = _positive_float(
        row["normal_force_capacity_n"], f"pads.{name}.normal_force_capacity_n"
    )
    points_local_m, faces = _inspect_pad_npz(
        mesh.absolute_path, f"pads.{name}.footprint"
    )
    return VerifiedPad(
        name=name,
        finger_name=expected_finger,
        link_name=expected_link,
        origin_xyz_m=xyz,
        origin_rpy_rad=rpy,
        mesh=mesh,
        coordinate_frame=expected_link,
        unit="m",
        normal_force_capacity_n=force,
        points_local_m=points_local_m,
        faces=faces,
    )


def _validate_manifest_lineage(
    manifest_file: VerifiedFileReference, pads: Sequence[VerifiedPad]
) -> None:
    manifest = _load_source_manifest(manifest_file.absolute_path)
    manifest_schema = manifest.get("schema")
    if manifest_schema == _EXACT_PAD_MANIFEST_SCHEMA:
        pad_name_key = "pad_source_arrays"
        pad_sha_key = "pad_source_arrays_sha256"
        _exact_bool(
            manifest.get("coordinate_tolerance_used"),
            False,
            "source manifest coordinate_tolerance_used",
        )
        _exact_bool(
            manifest.get("source_vertex_changed"),
            False,
            "source manifest source_vertex_changed",
        )
    elif manifest_schema == _LEGACY_PAD_MANIFEST_SCHEMA:
        pad_name_key = "pad_sdf_source_arrays"
        pad_sha_key = "pad_sdf_source_arrays_sha256"
    else:
        raise HandContractError(
            f"PAD source manifest schema is not registered: {manifest_schema!r}"
        )
    _exact_string(
        manifest.get("source_authority"),
        _MANIFEST_SOURCE_AUTHORITY,
        "source manifest source_authority",
    )
    _exact_string(
        manifest.get("semantic_authority"),
        _MANIFEST_SEMANTIC_AUTHORITY,
        "source manifest semantic_authority",
    )
    _exact_string(manifest.get("local_points_unit"), "metre", "source manifest unit")
    _exact_bool(
        manifest.get("dynamic_use_allowed"), False, "source manifest dynamic_use_allowed"
    )
    _exact_bool(
        manifest.get("online_control_role_truth_allowed"),
        False,
        "source manifest online_control_role_truth_allowed",
    )
    links = manifest.get("links")
    if not isinstance(links, list) or len(links) != 3:
        raise HandContractError("PAD source manifest must contain exactly three link entries")
    by_finger: dict[int, Mapping[str, Any]] = {}
    for item in links:
        row = _mapping(item, "PAD source manifest link entry")
        number = row.get("finger_number")
        if isinstance(number, bool) or not isinstance(number, int) or number not in (1, 2, 3):
            raise HandContractError("PAD source manifest finger_number must be 1, 2, or 3")
        if number in by_finger:
            raise HandContractError("PAD source manifest finger entries must be unique")
        by_finger[number] = row

    for index, pad in enumerate(pads, start=1):
        row = by_finger[index]
        _exact_string(row.get("link_name"), pad.link_name, f"source manifest finger {index} link")
        _exact_string(
            row.get(pad_name_key),
            pad.mesh.absolute_path.name,
            f"source manifest finger {index} PAD NPZ",
        )
        _exact_string(
            row.get(pad_sha_key),
            pad.mesh.sha256,
            f"source manifest finger {index} PAD NPZ SHA-256",
        )
        diagnostics = _mapping(
            row.get("diagnostics"),
            f"source manifest finger {index} diagnostics",
        )
        _exact_bool(
            diagnostics.get("pad_component_is_winding_consistent"),
            True,
            f"source manifest finger {index} PAD winding consistency",
        )
        if manifest_schema == _EXACT_PAD_MANIFEST_SCHEMA:
            _exact_bool(
                diagnostics.get("exact_source_face_ordinal_lineage_complete"),
                True,
                f"source manifest finger {index} exact source lineage",
            )


def load_carts_hand_contract(
    contract_path: str | Path, *, repository_root: str | Path
) -> CARTSHandContract:
    """Load and audit the frozen CARTS hand YAML without semantic defaults."""

    try:
        root = Path(repository_root).resolve(strict=True)
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        raise HandContractError("repository_root must be an existing directory") from exc
    if not root.is_dir():
        raise HandContractError("repository_root must be an existing directory")
    supplied_path = Path(contract_path)
    path = (
        supplied_path.resolve()
        if supplied_path.is_absolute()
        else (root / supplied_path).resolve()
    )
    if not path.is_file():
        raise HandContractError(f"hand contract file is unavailable: {path}")
    root_value = _load_unique_yaml(path)
    _exact_keys(
        root_value,
        (
            "schema_version",
            "method",
            "hardware_authorized",
            "kinematics",
            "closure_actuation",
            "pad_semantics",
            "pads",
            "force_capacity",
            "online_truth_firewall",
        ),
        "hand contract root",
    )
    _exact_string(root_value["schema_version"], _SCHEMA_VERSION, "schema_version")
    _exact_string(root_value["method"], _METHOD, "method")
    hardware_authorized = _exact_bool(
        root_value["hardware_authorized"], False, "hardware_authorized"
    )

    kinematics = _mapping(root_value["kinematics"], "kinematics")
    _exact_keys(
        kinematics,
        (
            "urdf",
            "urdf_sha256",
            "base_link",
            "joint_limits_source",
            "jacobian_method",
            "closing_velocity_domain",
            "object_contact_normal_feasibility",
            "pad_surface_normal_feasibility",
            "object_specific_alignment_threshold_allowed",
        ),
        "kinematics",
    )
    urdf = _verified_repository_file(
        root, kinematics["urdf"], kinematics["urdf_sha256"], "kinematics.urdf"
    )
    base_link = _required_string(kinematics["base_link"], "kinematics.base_link")
    _exact_string(kinematics["joint_limits_source"], "URDF", "kinematics.joint_limits_source")
    _exact_string(
        kinematics["jacobian_method"],
        "ANALYTIC_GEOMETRIC_FROM_URDF_TREE",
        "kinematics.jacobian_method",
    )
    _exact_string(
        kinematics["closing_velocity_domain"],
        "PAD_LINEAR_JACOBIAN_TIMES_DECLARED_CLOSING_JOINT_VELOCITY",
        "kinematics.closing_velocity_domain",
    )
    _exact_string(
        kinematics["object_contact_normal_feasibility"],
        OBJECT_CONTACT_NORMAL_POLICY,
        "kinematics.object_contact_normal_feasibility",
    )
    _exact_string(
        kinematics["pad_surface_normal_feasibility"],
        PAD_SURFACE_NORMAL_POLICY,
        "kinematics.pad_surface_normal_feasibility",
    )
    _exact_bool(
        kinematics["object_specific_alignment_threshold_allowed"],
        False,
        "kinematics.object_specific_alignment_threshold_allowed",
    )
    (
        closure_method,
        closure_source,
        closure_normalization,
        shared_joint_role,
        closure_rows,
    ) = _parse_closure_actuation(root_value["closure_actuation"])

    semantics = _mapping(root_value["pad_semantics"], "pad_semantics")
    _exact_keys(
        semantics,
        (
            "authority",
            "source_manifest",
            "source_manifest_sha256",
            "whole_blue_pad_body_is_finite_footprint",
            "triangle_winding_source",
            "triangle_winding_consistency_required",
            "old_pad_contact_face_ids_used",
            "old_closure_alignment_0p90_used",
            "red_tip_component_allowed_as_pad",
        ),
        "pad_semantics",
    )
    _exact_string(semantics["authority"], _PAD_AUTHORITY, "pad_semantics.authority")
    source_manifest = _verified_repository_file(
        root,
        semantics["source_manifest"],
        semantics["source_manifest_sha256"],
        "pad_semantics.source_manifest",
    )
    _exact_bool(
        semantics["whole_blue_pad_body_is_finite_footprint"],
        True,
        "pad_semantics.whole_blue_pad_body_is_finite_footprint",
    )
    triangle_winding_source = _exact_string(
        semantics["triangle_winding_source"],
        PAD_TRIANGLE_WINDING_SOURCE,
        "pad_semantics.triangle_winding_source",
    )
    triangle_winding_consistency = _exact_bool(
        semantics["triangle_winding_consistency_required"],
        True,
        "pad_semantics.triangle_winding_consistency_required",
    )
    _exact_bool(
        semantics["old_pad_contact_face_ids_used"],
        False,
        "pad_semantics.old_pad_contact_face_ids_used",
    )
    _exact_bool(
        semantics["old_closure_alignment_0p90_used"],
        False,
        "pad_semantics.old_closure_alignment_0p90_used",
    )
    _exact_bool(
        semantics["red_tip_component_allowed_as_pad"],
        False,
        "pad_semantics.red_tip_component_allowed_as_pad",
    )

    pad_values = _mapping(root_value["pads"], "pads")
    _exact_keys(pad_values, tuple(_EXPECTED_PADS), "pads")
    pads = tuple(_parse_pad(name, pad_values[name], root) for name in _EXPECTED_PADS)
    _validate_manifest_lineage(source_manifest, pads)

    capacity = _mapping(root_value["force_capacity"], "force_capacity")
    _exact_keys(
        capacity,
        (
            "value_source",
            "role",
            "physical_tactile_or_load_cell_calibration_complete",
            "simulator_drive_and_jacobian_readback_complete",
            "dynamic_use_allowed",
        ),
        "force_capacity",
    )
    value_source = _exact_string(
        capacity["value_source"], _FORCE_SOURCE, "force_capacity.value_source"
    )
    role = _exact_string(capacity["role"], _FORCE_ROLE, "force_capacity.role")
    physical_calibration = _exact_bool(
        capacity["physical_tactile_or_load_cell_calibration_complete"],
        False,
        "force_capacity.physical_tactile_or_load_cell_calibration_complete",
    )
    simulator_readback = _exact_bool(
        capacity["simulator_drive_and_jacobian_readback_complete"],
        False,
        "force_capacity.simulator_drive_and_jacobian_readback_complete",
    )
    dynamic_allowed = _exact_bool(
        capacity["dynamic_use_allowed"], False, "force_capacity.dynamic_use_allowed"
    )

    firewall = _mapping(root_value["online_truth_firewall"], "online_truth_firewall")
    _exact_keys(firewall, _TRUTH_FIELDS, "online_truth_firewall")
    firewall_values = tuple(
        (name, _exact_bool(firewall[name], False, f"online_truth_firewall.{name}"))
        for name in _TRUTH_FIELDS
    )

    contract = CARTSHandContract(
        contract_path=path,
        repository_root=root,
        urdf=urdf,
        source_manifest=source_manifest,
        base_link=base_link,
        pads=pads,  # type: ignore[arg-type]
        closure_actuation_method=closure_method,
        closure_actuation_source=closure_source,
        closure_actuation_normalization=closure_normalization,
        shared_independent_joint_role=shared_joint_role,
        object_contact_normal_policy=OBJECT_CONTACT_NORMAL_POLICY,
        pad_surface_normal_policy=PAD_SURFACE_NORMAL_POLICY,
        pad_triangle_winding_source=triangle_winding_source,
        pad_triangle_winding_consistency_required=(
            triangle_winding_consistency
        ),
        closure_actuation_rows=closure_rows,
        force_capacity_value_source=value_source,
        force_capacity_role=role,
        hardware_authorized=hardware_authorized,
        physical_calibration_complete=physical_calibration,
        simulator_readback_complete=simulator_readback,
        dynamic_use_allowed=dynamic_allowed,
        online_truth_firewall=firewall_values,
    )

    model = contract.build_hand_model()
    contract.closing_actuation_directions_unit(model)
    for pad in pads:
        if model.fingers[pad.finger_name].terminal_link != pad.link_name:
            raise HandContractError(
                f"PAD {pad.name} link is not the URDF terminal link for {pad.finger_name}"
            )
        parsed_pad = model.pads[pad.name]
        if parsed_pad.finger_name != pad.finger_name or parsed_pad.link_name != pad.link_name:
            raise HandContractError(f"PAD {pad.name} lost its one-to-one URDF assignment")
    return contract


__all__ = [
    "CARTSHandContract",
    "HandContractError",
    "OBJECT_CONTACT_NORMAL_POLICY",
    "PAD_SURFACE_NORMAL_POLICY",
    "PAD_TRIANGLE_WINDING_SOURCE",
    "VerifiedFileReference",
    "VerifiedPad",
    "load_carts_hand_contract",
]
