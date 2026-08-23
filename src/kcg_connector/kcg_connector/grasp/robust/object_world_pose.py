"""Hash-bound settled world poses for both CARTS-Grasp study objects.

The shared tabletop scene supplies only the XY station and the tabletop
orientation.  The final Z origin is derived independently for each verified
planning mesh by placing its lowest rotated material vertex on the certified
table top.  No legacy grasp candidate, simulator truth, or post-start pose
write is accepted as an input.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from kcg_connector.grasp.robust.aggregate_collision_inputs import (
    AggregateCollisionRuntimeInputCertificate,
)
from kcg_connector.grasp.robust.hand_model import rpy_rotation


METHOD_ID = "CARTS_HASH_BOUND_GEOMETRY_DERIVED_SETTLED_OBJECT_WORLD_POSE_V1"
EXPECTED_SCHEMA_VERSION = "carts_shared_object_placement_v1"
EXPECTED_PLACEMENT_ID = (
    "CARTS_SHARED_GEOMETRY_DERIVED_SETTLED_OBJECT_PLACEMENT_V1"
)
EXPECTED_CLAIM_SCOPE = "STATIC_SETTLED_OBJECT_WORLD_POSE_BINDING_ONLY"
EXPECTED_OBJECT_IDS = (
    "current_d38999_26kj61sn_public_spec",
    "te_deutsch_d38999_26fj35pn_step",
)
SETTLED_Z_METHOD = (
    "TABLE_TOP_MINUS_MINIMUM_ROTATED_VERIFIED_OBJECT_VERTEX_Z"
)
EXPECTED_STATION_XY_SOURCE_FRAGMENT = "loose_endpoint.initial_origin_m"
EXPECTED_ORIENTATION_SOURCE_FRAGMENT = (
    "asset_profile.loose_endpoint_rotation_degrees_xyz"
)
CLAIM_LIMITATIONS = (
    "STATIC_GEOMETRY_DERIVED_SETTLED_POSE_ONLY",
    "NO_DYNAMIC_SETTLING_STABILITY_OR_CONTACT_FORCE_CLAIM",
    "NO_CANDIDATE_IK_APPROACH_CLOSURE_LIFT_OR_COLLISION_CLAIM",
    "NO_ISAAC_HARDWARE_OR_POST_START_OBJECT_POSE_WRITE",
)


class ObjectWorldPoseError(ValueError):
    """Fail-closed object-placement contract error."""

    def __init__(self, code: str, detail: str) -> None:
        if not code or not detail:
            raise ValueError("object-world-pose error fields cannot be empty")
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
            raise ObjectWorldPoseError(
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
        raise ObjectWorldPoseError(
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
        raise ObjectWorldPoseError(
            "SCHEMA_MISMATCH", f"{label} missing={missing}, extra={extra}"
        )


def _load_yaml(path: Path, label: str) -> Mapping[str, Any]:
    try:
        return _mapping(
            yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeySafeLoader),
            label,
        )
    except yaml.YAMLError as error:
        raise ObjectWorldPoseError("INVALID_YAML", f"{label}: {error}") from error


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repository_file(root: Path, value: Any, label: str) -> Path:
    relative = Path(str(value))
    if relative.is_absolute():
        raise ObjectWorldPoseError(
            "ABSOLUTE_SOURCE_PATH_FORBIDDEN", f"{label} must be repository-relative"
        )
    result = (root / relative).resolve()
    try:
        result.relative_to(root)
    except ValueError as error:
        raise ObjectWorldPoseError(
            "SOURCE_PATH_ESCAPES_REPOSITORY", f"{label}={relative}"
        ) from error
    if not result.is_file():
        raise ObjectWorldPoseError("SOURCE_FILE_MISSING", f"{label}={relative}")
    return result


def _fragment(document: Mapping[str, Any], dotted_path: str, label: str) -> Any:
    current: Any = document
    for component in dotted_path.split("."):
        if not isinstance(current, Mapping) or component not in current:
            raise ObjectWorldPoseError(
                "SOURCE_FRAGMENT_MISSING", f"{label}: {dotted_path}"
            )
        current = current[component]
    return current


def _vector(value: Any, length: int, label: str) -> tuple[float, ...]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ObjectWorldPoseError(
            "FINITE_VECTOR_REQUIRED", f"{label} must contain {length} finite values"
        )
    return tuple(float(row) for row in array)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _float_hex(value: float) -> str:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ObjectWorldPoseError("NONFINITE_POSE_VALUE", repr(value))
    return parsed.hex()


def _proper_transform(value: object) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("world_from_object must be one finite 4x4 matrix")
    tolerance = 128.0 * np.finfo(np.float64).eps
    if (
        float(np.linalg.norm(matrix[3] - (0.0, 0.0, 0.0, 1.0))) > tolerance
        or float(np.linalg.norm(matrix[:3, :3].T @ matrix[:3, :3] - np.eye(3)))
        > tolerance
        or abs(float(np.linalg.det(matrix[:3, :3])) - 1.0) > tolerance
    ):
        raise ValueError("world_from_object must be a proper rigid transform")
    result = np.array(matrix, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _certificate_document(
    source: "SettledObjectWorldPoseCertificate | Mapping[str, object]",
) -> dict[str, object]:
    def field(name: str) -> object:
        return source[name] if isinstance(source, Mapping) else getattr(source, name)

    transform = np.asarray(field("world_from_object"), dtype=np.float64)
    return {
        "method_id": field("method_id"),
        "placement_id": field("placement_id"),
        "claim_scope": field("claim_scope"),
        "placement_config_sha256": field("placement_config_sha256"),
        "tabletop_scene_path": field("tabletop_scene_path"),
        "tabletop_scene_sha256": field("tabletop_scene_sha256"),
        "aggregate_collision_input_sha256": field(
            "aggregate_collision_input_sha256"
        ),
        "shared_environment_certificate_sha256": field(
            "shared_environment_certificate_sha256"
        ),
        "object_id": field("object_id"),
        "object_source_asset_sha256": field("object_source_asset_sha256"),
        "object_surface_geometry_sha256": field("object_surface_geometry_sha256"),
        "root_frame": field("root_frame"),
        "station_xy_m": [_float_hex(v) for v in field("station_xy_m")],
        "orientation_degrees_xyz": [
            _float_hex(v) for v in field("orientation_degrees_xyz")
        ],
        "world_from_object": [
            [_float_hex(value) for value in row] for row in transform
        ],
        "rotated_object_minimum_z_m": _float_hex(
            field("rotated_object_minimum_z_m")
        ),
        "table_top_z_m": _float_hex(field("table_top_z_m")),
        "table_contact_gap_m": _float_hex(field("table_contact_gap_m")),
        "transformed_bounds_world_m": [
            [_float_hex(value) for value in row]
            for row in field("transformed_bounds_world_m")
        ],
        "settled_origin_z_method": field("settled_origin_z_method"),
        "static_settled_pose_binding_complete": field(
            "static_settled_pose_binding_complete"
        ),
        "candidate_route_included": field("candidate_route_included"),
        "isaac_dynamic_state_included": field("isaac_dynamic_state_included"),
        "hardware_state_included": field("hardware_state_included"),
        "post_start_object_pose_write_allowed": field(
            "post_start_object_pose_write_allowed"
        ),
        "claim_limitations": list(field("claim_limitations")),
    }


def _certificate_sha256(
    source: "SettledObjectWorldPoseCertificate | Mapping[str, object]",
) -> str:
    return hashlib.sha256(
        json.dumps(
            _certificate_document(source),
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class SettledObjectWorldPoseCertificate:
    method_id: str
    placement_id: str
    claim_scope: str
    placement_config_sha256: str
    tabletop_scene_path: str
    tabletop_scene_sha256: str
    aggregate_collision_input_sha256: str
    shared_environment_certificate_sha256: str
    object_id: str
    object_source_asset_sha256: str
    object_surface_geometry_sha256: str
    root_frame: str
    station_xy_m: tuple[float, float]
    orientation_degrees_xyz: tuple[float, float, float]
    world_from_object: np.ndarray
    rotated_object_minimum_z_m: float
    table_top_z_m: float
    table_contact_gap_m: float
    transformed_bounds_world_m: tuple[
        tuple[float, float, float], tuple[float, float, float]
    ]
    settled_origin_z_method: str
    static_settled_pose_binding_complete: bool
    candidate_route_included: bool
    isaac_dynamic_state_included: bool
    hardware_state_included: bool
    post_start_object_pose_write_allowed: bool
    claim_limitations: tuple[str, ...]
    certificate_sha256: str

    def __post_init__(self) -> None:
        transform = _proper_transform(self.world_from_object)
        numeric = (
            *self.station_xy_m,
            *self.orientation_degrees_xyz,
            self.rotated_object_minimum_z_m,
            self.table_top_z_m,
            self.table_contact_gap_m,
            *(value for row in self.transformed_bounds_world_m for value in row),
        )
        lower, upper = self.transformed_bounds_world_m
        scale = max(1.0, abs(self.table_top_z_m))
        contact_tolerance = 256.0 * np.finfo(np.float64).eps * scale
        if (
            self.method_id != METHOD_ID
            or self.placement_id != EXPECTED_PLACEMENT_ID
            or self.claim_scope != EXPECTED_CLAIM_SCOPE
            or self.object_id not in EXPECTED_OBJECT_IDS
            or self.root_frame != "world"
            or not self.tabletop_scene_path.endswith(
                "d38999_multilayer_tabletop_scene_grasp_v1.yaml"
            )
            or any(
                not _is_sha256(value)
                for value in (
                    self.placement_config_sha256,
                    self.tabletop_scene_sha256,
                    self.aggregate_collision_input_sha256,
                    self.shared_environment_certificate_sha256,
                    self.object_source_asset_sha256,
                    self.object_surface_geometry_sha256,
                    self.certificate_sha256,
                )
            )
            or len(self.station_xy_m) != 2
            or len(self.orientation_degrees_xyz) != 3
            or not all(math.isfinite(float(value)) for value in numeric)
            or any(float(first) > float(second) for first, second in zip(lower, upper))
            or self.settled_origin_z_method != SETTLED_Z_METHOD
            or self.static_settled_pose_binding_complete is not True
            or any(
                value is not False
                for value in (
                    self.candidate_route_included,
                    self.isaac_dynamic_state_included,
                    self.hardware_state_included,
                    self.post_start_object_pose_write_allowed,
                )
            )
            or self.claim_limitations != CLAIM_LIMITATIONS
            or abs(float(lower[2]) - (self.table_top_z_m + self.table_contact_gap_m))
            > contact_tolerance
        ):
            raise ValueError("settled object world-pose certificate is incomplete")
        object.__setattr__(self, "world_from_object", transform)
        if self.certificate_sha256 != _certificate_sha256(self):
            raise ValueError("settled object world-pose certificate digest changed")

    @property
    def audit(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "object_id": self.object_id,
                "object_surface_geometry_sha256": self.object_surface_geometry_sha256,
                "shared_environment_certificate_sha256": (
                    self.shared_environment_certificate_sha256
                ),
                "world_origin_m": tuple(
                    float(value) for value in self.world_from_object[:3, 3]
                ),
                "table_top_z_m": self.table_top_z_m,
                "transformed_minimum_z_m": self.transformed_bounds_world_m[0][2],
                "static_settled_pose_binding_complete": True,
                "candidate_route_included": False,
                "dynamic_claimed": False,
                "certificate_sha256": self.certificate_sha256,
            }
        )


def certify_settled_object_world_pose(
    config_path: Path | str,
    *,
    aggregate_inputs: AggregateCollisionRuntimeInputCertificate,
    repository_root: Path | str,
) -> SettledObjectWorldPoseCertificate:
    """Derive one object's static tabletop pose from verified source bytes."""

    if not isinstance(
        aggregate_inputs, AggregateCollisionRuntimeInputCertificate
    ):
        raise ObjectWorldPoseError(
            "VERIFIED_AGGREGATE_INPUT_REQUIRED",
            "aggregate_inputs must be the exact certified type",
        )
    root = Path(repository_root).resolve()
    raw_path = Path(config_path)
    path = raw_path if raw_path.is_absolute() else root / raw_path
    path = path.resolve()
    config = _load_yaml(path, "object placement config")
    _exact_keys(
        config,
        (
            "schema_version",
            "placement_id",
            "claim_scope",
            "registered_object_ids",
            "source_bindings",
            "world_binding",
            "placement_rule",
        ),
        "object placement config",
    )
    if (
        config["schema_version"] != EXPECTED_SCHEMA_VERSION
        or config["placement_id"] != EXPECTED_PLACEMENT_ID
        or config["claim_scope"] != EXPECTED_CLAIM_SCOPE
        or tuple(config["registered_object_ids"]) != EXPECTED_OBJECT_IDS
        or aggregate_inputs.object_id not in EXPECTED_OBJECT_IDS
    ):
        raise ObjectWorldPoseError(
            "CONTRACT_IDENTITY_MISMATCH", "placement identity or object changed"
        )

    source_bindings = _mapping(config["source_bindings"], "source_bindings")
    _exact_keys(source_bindings, ("tabletop_scene",), "source_bindings")
    scene_binding = _mapping(source_bindings["tabletop_scene"], "tabletop_scene")
    _exact_keys(
        scene_binding, ("path", "sha256", "schema_version"), "tabletop_scene"
    )
    scene_path = _repository_file(root, scene_binding["path"], "tabletop_scene.path")
    scene_sha = _file_sha256(scene_path)
    expected_scene_sha = str(scene_binding["sha256"]).lower()
    if not _is_sha256(expected_scene_sha) or scene_sha != expected_scene_sha:
        raise ObjectWorldPoseError(
            "SOURCE_SHA256_MISMATCH", f"tabletop_scene: {scene_sha}"
        )
    environment_bindings = {
        name: (source_path, source_sha)
        for name, source_path, source_sha in (
            aggregate_inputs.shared_environment.source_bindings
        )
    }
    if environment_bindings.get("tabletop_scene") != (
        str(scene_binding["path"]),
        scene_sha,
    ):
        raise ObjectWorldPoseError(
            "SHARED_ENVIRONMENT_SOURCE_MISMATCH",
            "placement and environment use different tabletop bytes",
        )
    scene = _load_yaml(scene_path, "tabletop scene source")
    if scene.get("schema_version") != scene_binding["schema_version"]:
        raise ObjectWorldPoseError(
            "SOURCE_SCHEMA_VERSION_MISMATCH", "tabletop scene schema changed"
        )

    world = _mapping(config["world_binding"], "world_binding")
    _exact_keys(
        world,
        ("root_frame", "shared_environment_id", "table_obstacle_name"),
        "world_binding",
    )
    environment = aggregate_inputs.shared_environment
    if (
        world["root_frame"] != environment.root_frame
        or world["shared_environment_id"] != environment.environment_id
        or world["table_obstacle_name"] != "table"
    ):
        raise ObjectWorldPoseError(
            "WORLD_BINDING_MISMATCH", "placement and environment frames differ"
        )
    table = next(
        (row for row in environment.obstacles if row.name == "table"), None
    )
    if table is None:
        raise ObjectWorldPoseError("TABLE_OBSTACLE_MISSING", environment.environment_id)
    table_top_z = float(table.center_m[2] + 0.5 * table.size_m[2])

    rule = _mapping(config["placement_rule"], "placement_rule")
    _exact_keys(
        rule,
        (
            "station_xy_source_fragment",
            "station_xy_component_indices",
            "orientation_source_fragment",
            "orientation_unit",
            "settled_origin_z_method",
            "table_contact_gap_m",
            "airborne_initial_clearance_used",
            "historical_settled_pose_used_as_input",
            "legacy_grasp_candidate_used",
            "simulator_truth_used",
            "post_start_object_pose_write_allowed",
        ),
        "placement_rule",
    )
    if (
        rule["station_xy_source_fragment"]
        != EXPECTED_STATION_XY_SOURCE_FRAGMENT
        or rule["orientation_source_fragment"]
        != EXPECTED_ORIENTATION_SOURCE_FRAGMENT
        or tuple(rule["station_xy_component_indices"]) != (0, 1)
        or rule["orientation_unit"] != "deg"
        or rule["settled_origin_z_method"] != SETTLED_Z_METHOD
        or float(rule["table_contact_gap_m"]) != 0.0
        or any(
            rule[name] is not False
            for name in (
                "airborne_initial_clearance_used",
                "historical_settled_pose_used_as_input",
                "legacy_grasp_candidate_used",
                "simulator_truth_used",
                "post_start_object_pose_write_allowed",
            )
        )
    ):
        raise ObjectWorldPoseError(
            "PLACEMENT_RULE_CHANGED", "placement rule is no longer the frozen rule"
        )
    source_origin = _vector(
        _fragment(
            scene,
            str(rule["station_xy_source_fragment"]),
            "tabletop scene source",
        ),
        3,
        "source station origin",
    )
    orientation_deg = _vector(
        _fragment(
            scene,
            str(rule["orientation_source_fragment"]),
            "tabletop scene source",
        ),
        3,
        "source object orientation",
    )
    rotation = rpy_rotation(np.deg2rad(np.asarray(orientation_deg)))
    triangles = np.asarray(
        aggregate_inputs.object_surface.triangles_object_m,
        dtype=np.float64,
    )
    vertices = triangles.reshape((-1, 3))
    rotated = vertices @ rotation.T
    rotated_minimum_z = float(np.min(rotated[:, 2]))
    translation = np.asarray(
        (
            source_origin[0],
            source_origin[1],
            table_top_z - rotated_minimum_z,
        ),
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    transformed = rotated + translation
    lower = tuple(float(value) for value in np.min(transformed, axis=0))
    upper = tuple(float(value) for value in np.max(transformed, axis=0))

    values: dict[str, object] = {
        "method_id": METHOD_ID,
        "placement_id": EXPECTED_PLACEMENT_ID,
        "claim_scope": EXPECTED_CLAIM_SCOPE,
        "placement_config_sha256": _file_sha256(path),
        "tabletop_scene_path": str(scene_binding["path"]),
        "tabletop_scene_sha256": scene_sha,
        "aggregate_collision_input_sha256": aggregate_inputs.certificate_sha256,
        "shared_environment_certificate_sha256": environment.certificate_sha256,
        "object_id": aggregate_inputs.object_id,
        "object_source_asset_sha256": (
            aggregate_inputs.object_surface.source_asset_sha256
        ),
        "object_surface_geometry_sha256": (
            aggregate_inputs.object_surface.geometry_sha256
        ),
        "root_frame": environment.root_frame,
        "station_xy_m": (source_origin[0], source_origin[1]),
        "orientation_degrees_xyz": orientation_deg,
        "world_from_object": transform,
        "rotated_object_minimum_z_m": rotated_minimum_z,
        "table_top_z_m": table_top_z,
        "table_contact_gap_m": 0.0,
        "transformed_bounds_world_m": (lower, upper),
        "settled_origin_z_method": SETTLED_Z_METHOD,
        "static_settled_pose_binding_complete": True,
        "candidate_route_included": False,
        "isaac_dynamic_state_included": False,
        "hardware_state_included": False,
        "post_start_object_pose_write_allowed": False,
        "claim_limitations": CLAIM_LIMITATIONS,
    }
    return SettledObjectWorldPoseCertificate(
        **values,
        certificate_sha256=_certificate_sha256(values),
    )


__all__ = [
    "CLAIM_LIMITATIONS",
    "METHOD_ID",
    "ObjectWorldPoseError",
    "SETTLED_Z_METHOD",
    "SettledObjectWorldPoseCertificate",
    "certify_settled_object_world_pose",
]
