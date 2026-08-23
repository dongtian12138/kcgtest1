"""Hash-bound table and fixture surfaces shared by both CARTS objects.

The grasp study needs the same immobile obstacles for the development and
transfer objects.  This module reads a small study contract, verifies its
selected values against both existing tabletop sources, and turns the table
and fixture boxes into deterministic world-frame triangle surfaces.

It deliberately does not bind a loose-object pose, a fixed receptacle, a
candidate robot route, simulator state, or hardware state.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


METHOD_ID = "CARTS_HASH_BOUND_SHARED_TABLE_FIXTURE_WORLD_V1"
EXPECTED_SCHEMA_VERSION = "carts_shared_table_fixture_world_v1"
EXPECTED_ENVIRONMENT_ID = "CARTS_SHARED_TABLE_FIXTURE_WORLD_V1"
EXPECTED_CLAIM_SCOPE = "STATIC_SHARED_TABLE_FIXTURE_COLLISION_INPUT_ONLY"
EXPECTED_OBJECT_IDS = (
    "current_d38999_26kj61sn_public_spec",
    "te_deutsch_d38999_26fj35pn_step",
)
EXPECTED_OBSTACLE_ORDER = ("table", "fixture")
CLAIM_LIMITATIONS = (
    "STATIC_TABLE_AND_FIXTURE_BOX_SURFACES_ONLY",
    "NO_FIXED_RECEPTACLE_OR_LOOSE_OBJECT_INITIAL_POSE",
    "NO_CANDIDATE_ROUTE_CONTINUOUS_MOTION_ISAAC_OR_HARDWARE_CLAIM",
)


class SharedEnvironmentError(ValueError):
    """Fail-closed shared-environment contract error."""

    def __init__(self, code: str, detail: str) -> None:
        if not code or not detail:
            raise ValueError("shared-environment error fields cannot be empty")
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(f"{self.code}: {self.detail}")


class _UniqueKeySafeLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise SharedEnvironmentError(
                "DUPLICATE_YAML_KEY", f"duplicate key {key!r}"
            )
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
        raise SharedEnvironmentError(
            "MAPPING_REQUIRED", f"{label} must be a string-keyed mapping"
        )
    return value


def _exact_keys(
    value: Mapping[str, Any], expected: Sequence[str], label: str
) -> None:
    expected_set = set(expected)
    missing = sorted(expected_set.difference(value))
    extra = sorted(set(value).difference(expected_set))
    if missing or extra:
        raise SharedEnvironmentError(
            "SCHEMA_MISMATCH",
            f"{label} missing={missing}, extra={extra}",
        )


def _repository_file(root: Path, raw_path: Any, label: str) -> Path:
    relative = Path(str(raw_path))
    if relative.is_absolute():
        raise SharedEnvironmentError(
            "ABSOLUTE_SOURCE_PATH_FORBIDDEN", f"{label} must be repository-relative"
        )
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise SharedEnvironmentError(
            "SOURCE_PATH_ESCAPES_REPOSITORY", f"{label}={relative}"
        ) from error
    if not candidate.is_file():
        raise SharedEnvironmentError(
            "SOURCE_FILE_MISSING", f"{label}={relative}"
        )
    return candidate


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified_source(
    root: Path, binding: Mapping[str, Any], label: str
) -> tuple[Path, str]:
    expected_keys = ("path", "sha256")
    if label == "tabletop_scene":
        expected_keys = ("path", "sha256", "schema_version")
    _exact_keys(binding, expected_keys, f"source_bindings.{label}")
    path = _repository_file(root, binding["path"], f"source_bindings.{label}.path")
    expected = str(binding["sha256"]).lower()
    actual = _file_sha256(path)
    if len(expected) != 64 or any(
        character not in "0123456789abcdef" for character in expected
    ):
        raise SharedEnvironmentError(
            "MALFORMED_SOURCE_SHA256", f"source_bindings.{label}.sha256"
        )
    if actual != expected:
        raise SharedEnvironmentError(
            "SOURCE_SHA256_MISMATCH", f"{label}: {actual} != {expected}"
        )
    return path, actual


def _load_yaml(path: Path, label: str) -> Mapping[str, Any]:
    try:
        return _mapping(
            yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeySafeLoader),
            label,
        )
    except yaml.YAMLError as error:
        raise SharedEnvironmentError(
            "INVALID_YAML", f"{label}: {error}"
        ) from error


def _fragment(
    document: Mapping[str, Any], dotted_path: str, label: str
) -> Mapping[str, Any]:
    current: Any = document
    for component in dotted_path.split("."):
        if not isinstance(current, Mapping) or component not in current:
            raise SharedEnvironmentError(
                "SOURCE_FRAGMENT_MISSING", f"{label}: {dotted_path}"
            )
        current = current[component]
    return _mapping(current, f"{label}.{dotted_path}")


def _vector3(value: Any, label: str, *, positive: bool = False) -> tuple[float, ...]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise SharedEnvironmentError(
            "FINITE_VECTOR3_REQUIRED", f"{label} must be a finite length-three vector"
        )
    if positive and np.any(array <= 0.0):
        raise SharedEnvironmentError(
            "POSITIVE_SIZE_REQUIRED", f"{label} must be strictly positive"
        )
    return tuple(float(row) for row in array)


def _box_triangles(
    center_m: tuple[float, ...], size_m: tuple[float, ...]
) -> np.ndarray:
    center = np.asarray(center_m, dtype=np.float64)
    half = 0.5 * np.asarray(size_m, dtype=np.float64)
    signs = np.asarray(
        (
            (-1.0, -1.0, -1.0),
            (1.0, -1.0, -1.0),
            (1.0, 1.0, -1.0),
            (-1.0, 1.0, -1.0),
            (-1.0, -1.0, 1.0),
            (1.0, -1.0, 1.0),
            (1.0, 1.0, 1.0),
            (-1.0, 1.0, 1.0),
        ),
        dtype=np.float64,
    )
    vertices = center + signs * half
    faces = np.asarray(
        (
            (0, 2, 1), (0, 3, 2),
            (4, 5, 6), (4, 6, 7),
            (0, 1, 5), (0, 5, 4),
            (1, 2, 6), (1, 6, 5),
            (2, 3, 7), (2, 7, 6),
            (3, 0, 4), (3, 4, 7),
        ),
        dtype=np.int64,
    )
    triangles = np.array(vertices[faces], dtype=np.float64, copy=True)
    triangles.setflags(write=False)
    return triangles


def _triangle_geometry_sha256(triangles: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(triangles, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(b"CARTS_WORLD_FRAME_TRIANGLE_SURFACE_V1\0")
    digest.update(np.asarray(contiguous.shape, dtype="<u8").tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class WorldCollisionBox:
    name: str
    role: str
    prim_path: str
    center_m: tuple[float, ...]
    size_m: tuple[float, ...]
    triangles_world_m: np.ndarray
    geometry_sha256: str

    def __post_init__(self) -> None:
        if self.name not in EXPECTED_OBSTACLE_ORDER or self.role not in {
            "TABLE", "FIXTURE"
        }:
            raise ValueError("world collision box identity changed")
        if not self.prim_path.startswith("/World/"):
            raise ValueError("world collision box prim path is invalid")
        if len(self.center_m) != 3 or len(self.size_m) != 3 or any(
            not np.isfinite(value) for value in (*self.center_m, *self.size_m)
        ) or any(value <= 0.0 for value in self.size_m):
            raise ValueError("world collision box dimensions are invalid")
        triangles = np.array(self.triangles_world_m, dtype=np.float64, copy=True)
        if triangles.shape != (12, 3, 3) or not np.all(np.isfinite(triangles)):
            raise ValueError("world collision box must contain twelve triangles")
        triangles.setflags(write=False)
        object.__setattr__(self, "triangles_world_m", triangles)
        if self.geometry_sha256 != _triangle_geometry_sha256(triangles):
            raise ValueError("world collision box geometry digest changed")


@dataclass(frozen=True)
class SharedTableFixtureWorldCertificate:
    method_id: str
    environment_id: str
    claim_scope: str
    registered_object_ids: tuple[str, ...]
    source_bindings: tuple[tuple[str, str, str], ...]
    root_frame: str
    simulator_root_prim_path: str
    robot_base_origin_m: tuple[float, ...]
    obstacles: tuple[WorldCollisionBox, ...]
    obstacle_count: int
    fixed_receptacle_geometry_included: bool
    loose_object_initial_pose_included: bool
    candidate_specific_robot_route_included: bool
    isaac_dynamic_state_included: bool
    hardware_state_included: bool
    table_fixture_world_binding_complete: bool
    claim_limitations: tuple[str, ...]
    audit: Mapping[str, object]
    certificate_sha256: str

    def __post_init__(self) -> None:
        if (
            self.method_id != METHOD_ID
            or self.environment_id != EXPECTED_ENVIRONMENT_ID
            or self.claim_scope != EXPECTED_CLAIM_SCOPE
            or self.registered_object_ids != EXPECTED_OBJECT_IDS
            or self.root_frame != "world"
            or self.simulator_root_prim_path != "/World/D38999TabletopV1"
            or self.robot_base_origin_m != (0.0, 0.0, 0.0)
            or tuple(row.name for row in self.obstacles) != EXPECTED_OBSTACLE_ORDER
            or self.obstacle_count != 2
            or any(
                value is not False
                for value in (
                    self.fixed_receptacle_geometry_included,
                    self.loose_object_initial_pose_included,
                    self.candidate_specific_robot_route_included,
                    self.isaac_dynamic_state_included,
                    self.hardware_state_included,
                )
            )
            or self.table_fixture_world_binding_complete is not True
            or self.claim_limitations != CLAIM_LIMITATIONS
            or len(self.source_bindings) != 2
            or any(len(row[2]) != 64 for row in self.source_bindings)
            or len(self.certificate_sha256) != 64
        ):
            raise ValueError("shared table/fixture/world certificate is incomplete")
        audit = MappingProxyType(dict(self.audit))
        object.__setattr__(self, "audit", audit)
        if self.certificate_sha256 != _certificate_sha256(self):
            raise ValueError("shared environment certificate digest changed")


def _field(
    source: SharedTableFixtureWorldCertificate | Mapping[str, object],
    name: str,
) -> object:
    if isinstance(source, Mapping):
        return source[name]
    return getattr(source, name)


def _certificate_document(
    certificate: SharedTableFixtureWorldCertificate | Mapping[str, object],
) -> dict[str, object]:
    obstacles = tuple(_field(certificate, "obstacles"))
    return {
        "method_id": _field(certificate, "method_id"),
        "environment_id": _field(certificate, "environment_id"),
        "claim_scope": _field(certificate, "claim_scope"),
        "registered_object_ids": list(_field(certificate, "registered_object_ids")),
        "source_bindings": [list(row) for row in _field(certificate, "source_bindings")],
        "root_frame": _field(certificate, "root_frame"),
        "simulator_root_prim_path": _field(certificate, "simulator_root_prim_path"),
        "robot_base_origin_m": [
            float(value).hex()
            for value in _field(certificate, "robot_base_origin_m")
        ],
        "obstacles": [
            {
                "name": row.name,
                "role": row.role,
                "prim_path": row.prim_path,
                "center_m": [float(value).hex() for value in row.center_m],
                "size_m": [float(value).hex() for value in row.size_m],
                "geometry_sha256": row.geometry_sha256,
            }
            for row in obstacles
        ],
        "obstacle_count": _field(certificate, "obstacle_count"),
        "fixed_receptacle_geometry_included": _field(
            certificate, "fixed_receptacle_geometry_included"
        ),
        "loose_object_initial_pose_included": _field(
            certificate, "loose_object_initial_pose_included"
        ),
        "candidate_specific_robot_route_included": _field(
            certificate, "candidate_specific_robot_route_included"
        ),
        "isaac_dynamic_state_included": _field(
            certificate, "isaac_dynamic_state_included"
        ),
        "hardware_state_included": _field(certificate, "hardware_state_included"),
        "table_fixture_world_binding_complete": _field(
            certificate, "table_fixture_world_binding_complete"
        ),
        "claim_limitations": list(_field(certificate, "claim_limitations")),
        "audit": dict(_field(certificate, "audit")),
    }


def _certificate_sha256(
    certificate: SharedTableFixtureWorldCertificate | Mapping[str, object],
) -> str:
    return hashlib.sha256(
        json.dumps(
            _certificate_document(certificate),
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _selected_source_values(
    source: Mapping[str, Any], obstacle_name: str
) -> tuple[str, tuple[float, ...], tuple[float, ...]]:
    if obstacle_name == "table":
        path = source.get("prim_path", source.get("path"))
        center = source.get("center_m")
        size = source.get("size_m")
    else:
        path = source.get("fixture_prim_path", source.get("path"))
        center = source.get("fixture_center_m", source.get("center_m"))
        size = source.get("fixture_size_m", source.get("size_m"))
    if not isinstance(path, str) or not path:
        raise SharedEnvironmentError(
            "SELECTED_SOURCE_PATH_MISSING", f"{obstacle_name} path"
        )
    return (
        path,
        _vector3(center, f"{obstacle_name}.source.center_m"),
        _vector3(size, f"{obstacle_name}.source.size_m", positive=True),
    )


def load_shared_table_fixture_world(
    config_path: Path | str,
    *,
    repository_root: Path | str,
) -> SharedTableFixtureWorldCertificate:
    """Load and cross-check the two static obstacles used by both objects."""

    root = Path(repository_root).resolve()
    raw_config = Path(config_path)
    path = raw_config if raw_config.is_absolute() else root / raw_config
    config = _load_yaml(path.resolve(), "shared environment config")
    _exact_keys(
        config,
        (
            "schema_version",
            "environment_id",
            "claim_scope",
            "registered_object_ids",
            "source_bindings",
            "world",
            "obstacles",
            "excluded_from_this_static_binding",
        ),
        "shared environment config",
    )
    if (
        config["schema_version"] != EXPECTED_SCHEMA_VERSION
        or config["environment_id"] != EXPECTED_ENVIRONMENT_ID
        or config["claim_scope"] != EXPECTED_CLAIM_SCOPE
        or tuple(config["registered_object_ids"]) != EXPECTED_OBJECT_IDS
    ):
        raise SharedEnvironmentError(
            "CONTRACT_IDENTITY_MISMATCH", "shared environment identity changed"
        )

    bindings = _mapping(config["source_bindings"], "source_bindings")
    _exact_keys(bindings, ("tabletop_scene", "physical_contract"), "source_bindings")
    scene_binding = _mapping(bindings["tabletop_scene"], "tabletop_scene")
    physical_binding = _mapping(bindings["physical_contract"], "physical_contract")
    scene_path, scene_sha = _verified_source(root, scene_binding, "tabletop_scene")
    physical_path, physical_sha = _verified_source(
        root, physical_binding, "physical_contract"
    )
    scene = _load_yaml(scene_path, "tabletop scene source")
    physical = _load_yaml(physical_path, "physical contract source")
    if scene.get("schema_version") != scene_binding["schema_version"]:
        raise SharedEnvironmentError(
            "SOURCE_SCHEMA_VERSION_MISMATCH", "tabletop scene schema changed"
        )

    world = _mapping(config["world"], "world")
    _exact_keys(
        world,
        ("root_frame", "simulator_root_prim_path", "robot_base_origin_m"),
        "world",
    )
    scene_world = _fragment(scene, "world", "tabletop scene source")
    physical_world = _fragment(
        physical, "fixture_and_world_model", "physical contract source"
    )
    if (
        world["root_frame"] != "world"
        or world["simulator_root_prim_path"] != scene_world.get("root_prim_path")
        or world["simulator_root_prim_path"] != physical_world.get("scene_root")
    ):
        raise SharedEnvironmentError(
            "WORLD_FRAME_SOURCE_MISMATCH", "world path disagrees across sources"
        )
    robot_base = _vector3(world["robot_base_origin_m"], "world.robot_base_origin_m")
    if robot_base != (0.0, 0.0, 0.0):
        raise SharedEnvironmentError(
            "ROBOT_BASE_ORIGIN_CHANGED", f"robot_base_origin_m={robot_base}"
        )

    obstacle_documents = _mapping(config["obstacles"], "obstacles")
    _exact_keys(obstacle_documents, EXPECTED_OBSTACLE_ORDER, "obstacles")
    obstacle_rows: list[WorldCollisionBox] = []
    for name in EXPECTED_OBSTACLE_ORDER:
        row = _mapping(obstacle_documents[name], f"obstacles.{name}")
        _exact_keys(
            row,
            (
                "role",
                "prim_path",
                "center_m",
                "size_m",
                "tabletop_scene_fragment",
                "physical_contract_fragment",
            ),
            f"obstacles.{name}",
        )
        expected = (
            str(row["prim_path"]),
            _vector3(row["center_m"], f"obstacles.{name}.center_m"),
            _vector3(row["size_m"], f"obstacles.{name}.size_m", positive=True),
        )
        selected_scene = _selected_source_values(
            _fragment(scene, str(row["tabletop_scene_fragment"]), "tabletop scene source"),
            name,
        )
        selected_physical = _selected_source_values(
            _fragment(physical, str(row["physical_contract_fragment"]), "physical contract source"),
            name,
        )
        if expected != selected_scene or expected != selected_physical:
            raise SharedEnvironmentError(
                "SELECTED_SOURCE_MISMATCH",
                f"{name}: config={expected}, scene={selected_scene}, physical={selected_physical}",
            )
        triangles = _box_triangles(expected[1], expected[2])
        obstacle_rows.append(
            WorldCollisionBox(
                name=name,
                role=str(row["role"]),
                prim_path=expected[0],
                center_m=expected[1],
                size_m=expected[2],
                triangles_world_m=triangles,
                geometry_sha256=_triangle_geometry_sha256(triangles),
            )
        )

    exclusions = _mapping(
        config["excluded_from_this_static_binding"],
        "excluded_from_this_static_binding",
    )
    exclusion_keys = (
        "fixed_receptacle_geometry",
        "loose_object_initial_pose",
        "candidate_specific_robot_route",
        "isaac_dynamic_state",
        "hardware_state",
    )
    _exact_keys(exclusions, exclusion_keys, "excluded_from_this_static_binding")
    if any(exclusions[key] is not True for key in exclusion_keys):
        raise SharedEnvironmentError(
            "REQUIRED_EXCLUSION_MISSING",
            "all non-static inputs must remain explicitly excluded",
        )

    values: dict[str, object] = {
        "method_id": METHOD_ID,
        "environment_id": EXPECTED_ENVIRONMENT_ID,
        "claim_scope": EXPECTED_CLAIM_SCOPE,
        "registered_object_ids": EXPECTED_OBJECT_IDS,
        "source_bindings": (
            ("tabletop_scene", str(scene_binding["path"]), scene_sha),
            ("physical_contract", str(physical_binding["path"]), physical_sha),
        ),
        "root_frame": "world",
        "simulator_root_prim_path": str(world["simulator_root_prim_path"]),
        "robot_base_origin_m": robot_base,
        "obstacles": tuple(obstacle_rows),
        "obstacle_count": len(obstacle_rows),
        "fixed_receptacle_geometry_included": False,
        "loose_object_initial_pose_included": False,
        "candidate_specific_robot_route_included": False,
        "isaac_dynamic_state_included": False,
        "hardware_state_included": False,
        "table_fixture_world_binding_complete": True,
        "claim_limitations": CLAIM_LIMITATIONS,
        "audit": MappingProxyType(
            {
                "source_file_count": 2,
                "registered_object_count": 2,
                "obstacle_count": 2,
                "world_triangle_count": 24,
                "selected_values_match_both_sources": True,
                "table_fixture_world_binding_complete": True,
                "dynamic_claimed": False,
            }
        ),
    }
    digest = _certificate_sha256(values)
    return SharedTableFixtureWorldCertificate(
        **values,
        certificate_sha256=digest,
    )


__all__ = [
    "CLAIM_LIMITATIONS",
    "EXPECTED_OBJECT_IDS",
    "METHOD_ID",
    "SharedEnvironmentError",
    "SharedTableFixtureWorldCertificate",
    "WorldCollisionBox",
    "load_shared_table_fixture_world",
]
