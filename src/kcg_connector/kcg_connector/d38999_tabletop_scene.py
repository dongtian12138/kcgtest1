"""Pure contract and deferred authoring for D38999 tabletop smoke v1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path
import re
from typing import Any, Mapping

import yaml


D38999_TABLETOP_SCHEMA_VERSION = "kcg_d38999_tabletop_scene_v1"
D38999_TABLETOP_SCHEMA_VERSION_KEYED_V2 = (
    "kcg_d38999_keyed_v2_tabletop_scene_v1"
)
D38999_TABLETOP_SCHEMA_VERSION_KEYED_V3_R12 = (
    "kcg_d38999_keyed_v3_tabletop_scene_r12_v1"
)
D38999_TABLETOP_SCHEMA_VERSION_MULTILAYER_GRASP = (
    "kcg_d38999_multilayer_tabletop_scene_grasp_v1"
)
DEFAULT_D38999_TABLETOP_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "config/d38999_tabletop_scene_v1.yaml"
)
EXPECTED_ASSET_BASENAME = "d38999_shell25j_61_pair_proxy_v1.usda"
EXPECTED_KEYED_V2_ASSET_BASENAME = "d38999_shell25j_25_61_n_keyed_physical_v3_r11.usda"
EXPECTED_KEYED_V3_R12_ASSET_BASENAME = "d38999_shell25j_25_61_n_keyed_physical_v3_r12.usda"
EXPECTED_MULTILAYER_GRASP_ASSET_BASENAME = "D38999_ASSEMBLY_CONTROL_V1.usda"
LEGACY_ASSET_BASENAME = "connector_pair.usda"
_PRIM_PATTERN = re.compile(r"^/World(?:/[A-Za-z_][A-Za-z0-9_]*)+$")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _exact_keys(
    value: Mapping[str, Any], expected: tuple[str, ...], label: str
) -> None:
    actual = set(value)
    wanted = set(expected)
    if actual != wanted:
        raise ValueError(
            f"{label} keys are invalid; "
            f"missing={sorted(wanted - actual)}, "
            f"unexpected={sorted(actual - wanted)}"
        )


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value.strip()


def _prim_path(value: Any, label: str) -> str:
    result = _text(value, label)
    if not _PRIM_PATTERN.fullmatch(result):
        raise ValueError(f"{label} must be an absolute /World prim path")
    return result


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _positive(value: Any, label: str) -> float:
    result = _finite(value, label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{label} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return result


def _vector3(
    value: Any, label: str, *, positive: bool = False
) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{label} must contain exactly three numbers")
    result = tuple(
        _finite(item, f"{label}[{index}]")
        for index, item in enumerate(value)
    )
    if positive and any(item <= 0.0 for item in result):
        raise ValueError(f"{label} entries must be positive")
    return result


def _color(value: Any, label: str) -> tuple[float, float, float]:
    result = _vector3(value, label)
    if any(item < 0.0 or item > 1.0 for item in result):
        raise ValueError(f"{label} entries must be in [0, 1]")
    return result


@dataclass(frozen=True)
class D38999TabletopAsset:
    local_path: str
    proxy_id: str
    reference_prim_path: str
    model_root_prim_path: str
    loose_plug_prim_path: str
    body_prim_path: str
    nut_prim_path: str
    joint_prim_path: str
    fixed_receptacle_prim_path: str


@dataclass(frozen=True)
class D38999TabletopAssetProfile:
    profile_id: str
    source_config: str
    loose_endpoint_orientation: str
    fixed_endpoint_orientation: str
    loose_endpoint_rotation_degrees_xyz: tuple[float, float, float]
    fixed_endpoint_rotation_degrees_xyz: tuple[float, float, float]
    body_mass_kg: float
    nut_mass_kg: float
    expected_body_collider_count: int
    expected_nut_collider_count: int


_LEGACY_PROFILE_ID = "d38999_shell25j_61_pair_proxy_v1"
_KEYED_V2_PROFILE_ID = (
    "d38999_shell25j_25_61_n_keyed_physical_pair_v3"
)
_KEYED_V3_R12_PROFILE_ID = (
    "d38999_shell25j_25_61_n_keyed_physical_pair_v3_r12"
)
_MULTILAYER_GRASP_PROFILE_ID = "D38999_ASSEMBLY_CONTROL_V1"
_KEYED_V2_MODEL_IDS = {
    "pairModelId": _KEYED_V2_PROFILE_ID,
    "loosePlugModelId": "d38999_26kj61sn_physical_proxy_v3",
    "fixedReceptacleModelId": "d38999_20kj61pn_physical_proxy_v3",
}
_KEYED_V3_R12_MODEL_IDS = {
    **_KEYED_V2_MODEL_IDS,
    "pairModelId": _KEYED_V3_R12_PROFILE_ID,
}
_PHYSICAL_PROFILE_IDS = frozenset(
    (
        _KEYED_V2_PROFILE_ID,
        _KEYED_V3_R12_PROFILE_ID,
        _MULTILAYER_GRASP_PROFILE_ID,
    )
)
_FIXTURE_JOINT_PROFILE_IDS = frozenset(
    (_KEYED_V2_PROFILE_ID, _KEYED_V3_R12_PROFILE_ID)
)
_PHYSICAL_MODEL_IDS = {
    _KEYED_V2_PROFILE_ID: _KEYED_V2_MODEL_IDS,
    _KEYED_V3_R12_PROFILE_ID: _KEYED_V3_R12_MODEL_IDS,
    _MULTILAYER_GRASP_PROFILE_ID: {
        "representationId": _MULTILAYER_GRASP_PROFILE_ID,
    },
}
_PROFILE_VALUES = {
    _LEGACY_PROFILE_ID: {
        "profile_id": _LEGACY_PROFILE_ID,
        "source_config": (
            "src/kcg_connector/config/d38999_shell25j_proxy_v1.yaml"
        ),
        "loose_endpoint_orientation": "LEGACY_V1_ASSET_IDENTITY",
        "fixed_endpoint_orientation": "LEGACY_V1_ASSET_IDENTITY",
        "loose_endpoint_rotation_degrees_xyz": (0.0, 0.0, 0.0),
        "fixed_endpoint_rotation_degrees_xyz": (0.0, 0.0, 0.0),
        "body_mass_kg": 0.08,
        "nut_mass_kg": 0.04,
        "expected_body_collider_count": 21,
        "expected_nut_collider_count": 24,
    },
    _KEYED_V2_PROFILE_ID: {
        "profile_id": _KEYED_V2_PROFILE_ID,
        "source_config": (
            "src/kcg_connector/config/d38999_keyed_v2_physical_model_contract_v1.yaml"
        ),
        "loose_endpoint_orientation": "MATING_FACE_UP_RX_180_FOR_REAR_DOWN",
        "fixed_endpoint_orientation": (
            "MATING_FACE_UP_RX_180_FOR_DOWNWARD_INSERTION"
        ),
        "loose_endpoint_rotation_degrees_xyz": (180.0, 0.0, 0.0),
        "fixed_endpoint_rotation_degrees_xyz": (180.0, 0.0, 0.0),
        "body_mass_kg": 0.23,
        "nut_mass_kg": 0.08,
        "expected_body_collider_count": 7577,
        "expected_nut_collider_count": 294,
    },
    _KEYED_V3_R12_PROFILE_ID: {
        "profile_id": _KEYED_V3_R12_PROFILE_ID,
        "source_config": (
            "src/kcg_connector/config/"
            "d38999_keyed_v3_physical_model_contract_r12_v1.yaml"
        ),
        "loose_endpoint_orientation": "MATING_FACE_UP_RX_180_FOR_REAR_DOWN",
        "fixed_endpoint_orientation": (
            "MATING_FACE_UP_RX_180_FOR_DOWNWARD_INSERTION"
        ),
        "loose_endpoint_rotation_degrees_xyz": (180.0, 0.0, 0.0),
        "fixed_endpoint_rotation_degrees_xyz": (180.0, 0.0, 0.0),
        "body_mass_kg": 0.23,
        "nut_mass_kg": 0.08,
        "expected_body_collider_count": 7438,
        "expected_nut_collider_count": 204,
    },
    _MULTILAYER_GRASP_PROFILE_ID: {
        "profile_id": _MULTILAYER_GRASP_PROFILE_ID,
        "source_config": (
            "src/kcg_connector/config/d38999_master_model_contract_v1.yaml"
        ),
        "loose_endpoint_orientation": "MATING_FACE_UP_RX_180_FOR_REAR_DOWN",
        "fixed_endpoint_orientation": (
            "MATING_FACE_UP_RX_180_FOR_DOWNWARD_INSERTION"
        ),
        "loose_endpoint_rotation_degrees_xyz": (180.0, 0.0, 0.0),
        "fixed_endpoint_rotation_degrees_xyz": (180.0, 0.0, 0.0),
        "body_mass_kg": 0.23,
        "nut_mass_kg": 0.08,
        "expected_body_collider_count": 72,
        "expected_nut_collider_count": 7,
    },
}
_SCHEMA_PROFILE_IDS = {
    D38999_TABLETOP_SCHEMA_VERSION: _LEGACY_PROFILE_ID,
    D38999_TABLETOP_SCHEMA_VERSION_KEYED_V2: _KEYED_V2_PROFILE_ID,
    D38999_TABLETOP_SCHEMA_VERSION_KEYED_V3_R12: _KEYED_V3_R12_PROFILE_ID,
    D38999_TABLETOP_SCHEMA_VERSION_MULTILAYER_GRASP: (
        _MULTILAYER_GRASP_PROFILE_ID
    ),
}
_PROFILE_ASSET_CONTRACTS = {
    _LEGACY_PROFILE_ID: {
        "local_path": (
            "artifacts/kcg_connector/isaac/"
            "d38999_shell25j_61_pair_proxy_v1.usda"
        ),
        "basename": EXPECTED_ASSET_BASENAME,
        "model_root_name": "D38999Shell25JProxy",
    },
    _KEYED_V2_PROFILE_ID: {
        "local_path": (
            "artifacts/kcg_connector/isaac/keyed_v3_physical_r11/"
            "d38999_shell25j_25_61_n_keyed_physical_v3_r11.usda"
        ),
        "basename": EXPECTED_KEYED_V2_ASSET_BASENAME,
        "model_root_name": "D38999Shell25JKeyedPhysicalV3",
    },
    _KEYED_V3_R12_PROFILE_ID: {
        "local_path": (
            "artifacts/kcg_connector/isaac/keyed_v3_physical_r12/"
            "d38999_shell25j_25_61_n_keyed_physical_v3_r12.usda"
        ),
        "basename": EXPECTED_KEYED_V3_R12_ASSET_BASENAME,
        "candidate_path_pattern": (
            r"artifacts/kcg_connector/isaac/keyed_v3_physical_r12/"
            r"candidates/(r12_candidate_0[1-4])/\1\.usda"
        ),
        "model_root_name": "D38999Shell25JKeyedPhysicalV3",
    },
    _MULTILAYER_GRASP_PROFILE_ID: {
        "local_path": (
            "artifacts/kcg_connector/isaac/d38999_multilayer_v1/"
            "D38999_ASSEMBLY_CONTROL_V1.usda"
        ),
        "basename": EXPECTED_MULTILAYER_GRASP_ASSET_BASENAME,
        "model_root_name": (
            "D38999MultilayerV1/D38999_ASSEMBLY_CONTROL_V1"
        ),
        "pair_path_suffix": "/D38999Pair",
        "build_result_path": (
            "artifacts/agent_control/tasks/"
            "D38999-AUTONOMOUS-DYNAMIC-CLOSEOUT-V2/"
            "DYN-A2-NOMINAL-INSERTION-V2/"
            "A2_RUN05_NUT_BODY_SHOULDER_TARGETED_FIX_RESULT.json"
        ),
        "build_result_sha256": (
            "a8d144799c3d5e38ff04a875f39180c9a92c074e0fc2d7b34ac59f82b5918718"
        ),
        "authorized_overrides_path": (
            "src/kcg_connector/config/"
            "d38999_assembly_control_authorized_overrides_v2.yaml"
        ),
        "generator_path": (
            "src/kcg_connector/isaac/build_d38999_multilayer_models.py"
        ),
        "physical_contract_path": (
            "src/kcg_connector/config/"
            "d38999_keyed_v3_physical_model_contract_r12_v1.yaml"
        ),
        "preserved_output_paths": {
            "D38999_LOCAL_CONTACT_REFERENCE_V1.usda": (
                "artifacts/kcg_connector/isaac/d38999_multilayer_v1/"
                "D38999_LOCAL_CONTACT_REFERENCE_V1.usda"
            ),
            "D38999_VISUAL_COMPLETE_V1.usda": (
                "artifacts/kcg_connector/isaac/d38999_multilayer_v1/"
                "D38999_VISUAL_COMPLETE_V1.usda"
            ),
            "MODEL_MAPPING.json": (
                "artifacts/kcg_connector/isaac/d38999_multilayer_v1/"
                "MODEL_MAPPING.json"
            ),
        },
    },
}


def _asset_path_is_allowed(
    profile_id: str,
    local_path: str,
    authorized_local_asset_path: str | None = None,
) -> bool:
    contract = _PROFILE_ASSET_CONTRACTS[profile_id]
    if local_path == contract["local_path"]:
        return True
    if (
        profile_id == _KEYED_V3_R12_PROFILE_ID
        and authorized_local_asset_path is not None
        and local_path == authorized_local_asset_path
    ):
        return True
    pattern = contract.get("candidate_path_pattern")
    return bool(pattern and re.fullmatch(pattern, local_path))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_profile(profile_id: str) -> D38999TabletopAssetProfile:
    try:
        return D38999TabletopAssetProfile(**_PROFILE_VALUES[profile_id])
    except KeyError as error:
        raise ValueError("unsupported D38999 tabletop asset profile") from error


def _load_profile(
    value: Any, schema_version: str
) -> D38999TabletopAssetProfile:
    profile_id = _SCHEMA_PROFILE_IDS[schema_version]
    canonical = _canonical_profile(profile_id)
    if schema_version == D38999_TABLETOP_SCHEMA_VERSION:
        if value is not None:
            raise ValueError("legacy tabletop schema cannot declare a profile")
        return canonical
    document = _mapping(value, "asset_profile")
    keys = tuple(D38999TabletopAssetProfile.__dataclass_fields__)
    _exact_keys(document, keys, "asset_profile")
    source_config = _text(
        document["source_config"], "asset_profile.source_config"
    )
    if Path(source_config).is_absolute() or ".." in Path(source_config).parts:
        raise ValueError("asset_profile.source_config must be repository-relative")
    result = D38999TabletopAssetProfile(
        profile_id=_text(document["profile_id"], "asset_profile.profile_id"),
        source_config=source_config,
        loose_endpoint_orientation=_text(
            document["loose_endpoint_orientation"],
            "asset_profile.loose_endpoint_orientation",
        ),
        fixed_endpoint_orientation=_text(
            document["fixed_endpoint_orientation"],
            "asset_profile.fixed_endpoint_orientation",
        ),
        loose_endpoint_rotation_degrees_xyz=_vector3(
            document["loose_endpoint_rotation_degrees_xyz"],
            "asset_profile.loose_endpoint_rotation_degrees_xyz",
        ),
        fixed_endpoint_rotation_degrees_xyz=_vector3(
            document["fixed_endpoint_rotation_degrees_xyz"],
            "asset_profile.fixed_endpoint_rotation_degrees_xyz",
        ),
        body_mass_kg=_positive(
            document["body_mass_kg"], "asset_profile.body_mass_kg"
        ),
        nut_mass_kg=_positive(
            document["nut_mass_kg"], "asset_profile.nut_mass_kg"
        ),
        expected_body_collider_count=_positive_integer(
            document["expected_body_collider_count"],
            "asset_profile.expected_body_collider_count",
        ),
        expected_nut_collider_count=_positive_integer(
            document["expected_nut_collider_count"],
            "asset_profile.expected_nut_collider_count",
        ),
    )
    if result != canonical:
        raise ValueError("asset_profile is not an allowlisted canonical profile")
    return result


@dataclass(frozen=True)
class D38999TabletopWorld:
    root_prim_path: str
    physics_material_prim_path: str


@dataclass(frozen=True)
class D38999TabletopSurface:
    prim_path: str
    center_m: tuple[float, float, float]
    size_m: tuple[float, float, float]
    color_rgb: tuple[float, float, float]
    static_friction: float
    dynamic_friction: float
    restitution: float

    @property
    def top_z_m(self) -> float:
        return self.center_m[2] + 0.5 * self.size_m[2]


@dataclass(frozen=True)
class D38999FixedEndpoint:
    fixture_prim_path: str
    fixture_center_m: tuple[float, float, float]
    fixture_size_m: tuple[float, float, float]
    fixture_color_rgb: tuple[float, float, float]
    receptacle_origin_m: tuple[float, float, float]
    receptacle_bottom_offset_m: float

    @property
    def fixture_top_z_m(self) -> float:
        return self.fixture_center_m[2] + 0.5 * self.fixture_size_m[2]


@dataclass(frozen=True)
class D38999LooseEndpoint:
    initial_origin_m: tuple[float, float, float]
    body_bottom_offset_m: float
    nut_bottom_offset_m: float
    footprint_radius_m: float
    initial_clearance_above_table_m: float
    minimum_endpoint_separation_m: float

    @property
    def initial_bottom_z_m(self) -> float:
        return min(
            self.initial_origin_m[2] + self.body_bottom_offset_m,
            self.initial_origin_m[2] + self.nut_bottom_offset_m,
        )


@dataclass(frozen=True)
class D38999TabletopPhysics:
    rate_hz: int
    gravity_m_s2: float
    settle_duration_s: float
    tail_duration_s: float
    minimum_vertical_drop_m: float
    maximum_vertical_drop_m: float
    maximum_transient_table_penetration_m: float
    maximum_final_surface_gap_m: float
    maximum_xy_drift_m: float
    maximum_tail_displacement_m: float
    maximum_tail_linear_speed_m_s: float
    maximum_tail_angular_speed_rad_s: float
    maximum_upright_axis_tilt_rad: float
    maximum_fixed_translation_drift_m: float
    maximum_fixed_rotation_drift_rad: float

    @property
    def settle_steps(self) -> int:
        return int(round(self.rate_hz * self.settle_duration_s))

    @property
    def tail_steps(self) -> int:
        return int(round(self.rate_hz * self.tail_duration_s))


@dataclass(frozen=True)
class D38999TabletopRender:
    camera_eye_m: tuple[float, float, float]
    camera_target_m: tuple[float, float, float]
    dome_light_intensity: float
    dome_light_color_rgb: tuple[float, float, float]
    key_light_intensity: float
    key_light_color_rgb: tuple[float, float, float]
    key_light_rotation_degrees_xyz: tuple[float, float, float]


@dataclass(frozen=True)
class D38999TabletopScene:
    schema_version: str
    asset: D38999TabletopAsset
    world: D38999TabletopWorld
    table: D38999TabletopSurface
    fixed_endpoint: D38999FixedEndpoint
    loose_endpoint: D38999LooseEndpoint
    physics: D38999TabletopPhysics
    render: D38999TabletopRender

    @property
    def asset_profile(self) -> D38999TabletopAssetProfile:
        return _canonical_profile(self.asset.proxy_id)

    @property
    def loose_settled_origin_m(self) -> tuple[float, float, float]:
        bottom_offset = min(
            self.loose_endpoint.body_bottom_offset_m,
            self.loose_endpoint.nut_bottom_offset_m,
        )
        return (
            self.loose_endpoint.initial_origin_m[0],
            self.loose_endpoint.initial_origin_m[1],
            self.table.top_z_m - bottom_offset,
        )

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if self.schema_version != D38999_TABLETOP_SCHEMA_VERSION:
            result["asset_profile"] = asdict(self.asset_profile)
        return json.loads(json.dumps(result, allow_nan=False, sort_keys=True))


def _load_asset(
    value: Any,
    profile: D38999TabletopAssetProfile,
    authorized_local_asset_path: str | None = None,
) -> D38999TabletopAsset:
    document = _mapping(value, "asset")
    all_keys = tuple(D38999TabletopAsset.__dataclass_fields__)
    _exact_keys(document, all_keys, "asset")
    local_path = _text(document["local_path"], "asset.local_path")
    if Path(local_path).is_absolute() or ".." in Path(local_path).parts:
        raise ValueError("asset.local_path must be repository-relative")
    asset_contract = _PROFILE_ASSET_CONTRACTS[profile.profile_id]
    if not _asset_path_is_allowed(
        profile.profile_id, local_path, authorized_local_asset_path
    ):
        if profile.profile_id == _LEGACY_PROFILE_ID:
            raise ValueError("asset must be the independent D38999 proxy USD")
        raise ValueError("asset path is not canonical for its profile")
    values = {
        name: _prim_path(document[name], f"asset.{name}")
        for name in all_keys
        if name.endswith("_prim_path")
    }
    result = D38999TabletopAsset(
        local_path=local_path,
        proxy_id=_text(document["proxy_id"], "asset.proxy_id"),
        **values,
    )
    if result.proxy_id != profile.profile_id:
        raise ValueError("asset.proxy_id differs from its allowlisted profile")
    prefix = (
        result.reference_prim_path
        + "/"
        + asset_contract["model_root_name"]
    )
    if result.model_root_prim_path != prefix:
        raise ValueError("asset model root does not match its reference")
    pair_prefix = prefix + asset_contract.get("pair_path_suffix", "")
    expected = {
        "loose_plug_prim_path": pair_prefix + "/LoosePlug",
        "body_prim_path": pair_prefix + "/LoosePlug/BodyAssembly",
        "nut_prim_path": pair_prefix + "/LoosePlug/CouplingNut",
        "joint_prim_path": pair_prefix + "/LoosePlug/CouplingNutJoint",
        "fixed_receptacle_prim_path": pair_prefix + "/FixedReceptacle",
    }
    for name, wanted in expected.items():
        if getattr(result, name) != wanted:
            suffix = (
                "D38999 USD"
                if profile.profile_id == _LEGACY_PROFILE_ID
                else "profile USD"
            )
            raise ValueError(f"asset.{name} does not match {suffix}")
    return result


def _load_world(value: Any) -> D38999TabletopWorld:
    document = _mapping(value, "world")
    keys = tuple(D38999TabletopWorld.__dataclass_fields__)
    _exact_keys(document, keys, "world")
    return D38999TabletopWorld(
        root_prim_path=_prim_path(
            document["root_prim_path"], "world.root_prim_path"
        ),
        physics_material_prim_path=_prim_path(
            document["physics_material_prim_path"],
            "world.physics_material_prim_path",
        ),
    )


def _load_surface(value: Any) -> D38999TabletopSurface:
    document = _mapping(value, "table")
    keys = tuple(D38999TabletopSurface.__dataclass_fields__)
    _exact_keys(document, keys, "table")
    result = D38999TabletopSurface(
        prim_path=_prim_path(document["prim_path"], "table.prim_path"),
        center_m=_vector3(document["center_m"], "table.center_m"),
        size_m=_vector3(document["size_m"], "table.size_m", positive=True),
        color_rgb=_color(document["color_rgb"], "table.color_rgb"),
        static_friction=_finite(
            document["static_friction"], "table.static_friction"
        ),
        dynamic_friction=_finite(
            document["dynamic_friction"], "table.dynamic_friction"
        ),
        restitution=_finite(document["restitution"], "table.restitution"),
    )
    if result.dynamic_friction < 0.0:
        raise ValueError("table friction must be nonnegative")
    if result.static_friction < result.dynamic_friction:
        raise ValueError("table static friction must not be below dynamic")
    if not 0.0 <= result.restitution <= 1.0:
        raise ValueError("table restitution must be in [0, 1]")
    return result


def _load_fixed(value: Any) -> D38999FixedEndpoint:
    document = _mapping(value, "fixed_endpoint")
    keys = tuple(D38999FixedEndpoint.__dataclass_fields__)
    _exact_keys(document, keys, "fixed_endpoint")
    return D38999FixedEndpoint(
        fixture_prim_path=_prim_path(
            document["fixture_prim_path"],
            "fixed_endpoint.fixture_prim_path",
        ),
        fixture_center_m=_vector3(
            document["fixture_center_m"],
            "fixed_endpoint.fixture_center_m",
        ),
        fixture_size_m=_vector3(
            document["fixture_size_m"],
            "fixed_endpoint.fixture_size_m",
            positive=True,
        ),
        fixture_color_rgb=_color(
            document["fixture_color_rgb"],
            "fixed_endpoint.fixture_color_rgb",
        ),
        receptacle_origin_m=_vector3(
            document["receptacle_origin_m"],
            "fixed_endpoint.receptacle_origin_m",
        ),
        receptacle_bottom_offset_m=_positive(
            document["receptacle_bottom_offset_m"],
            "fixed_endpoint.receptacle_bottom_offset_m",
        ),
    )


def _load_loose(value: Any) -> D38999LooseEndpoint:
    document = _mapping(value, "loose_endpoint")
    keys = tuple(D38999LooseEndpoint.__dataclass_fields__)
    _exact_keys(document, keys, "loose_endpoint")
    return D38999LooseEndpoint(
        initial_origin_m=_vector3(
            document["initial_origin_m"],
            "loose_endpoint.initial_origin_m",
        ),
        body_bottom_offset_m=_finite(
            document["body_bottom_offset_m"],
            "loose_endpoint.body_bottom_offset_m",
        ),
        nut_bottom_offset_m=_finite(
            document["nut_bottom_offset_m"],
            "loose_endpoint.nut_bottom_offset_m",
        ),
        footprint_radius_m=_positive(
            document["footprint_radius_m"],
            "loose_endpoint.footprint_radius_m",
        ),
        initial_clearance_above_table_m=_positive(
            document["initial_clearance_above_table_m"],
            "loose_endpoint.initial_clearance_above_table_m",
        ),
        minimum_endpoint_separation_m=_positive(
            document["minimum_endpoint_separation_m"],
            "loose_endpoint.minimum_endpoint_separation_m",
        ),
    )


def _load_physics(value: Any) -> D38999TabletopPhysics:
    document = _mapping(value, "physics")
    keys = tuple(D38999TabletopPhysics.__dataclass_fields__)
    _exact_keys(document, keys, "physics")
    values = {}
    for name in keys:
        if name == "rate_hz":
            values[name] = _positive_integer(document[name], f"physics.{name}")
        else:
            values[name] = _finite(document[name], f"physics.{name}")
    result = D38999TabletopPhysics(**values)
    if result.rate_hz != 240:
        raise ValueError("physics.rate_hz must be exactly 240 Hz")
    if result.gravity_m_s2 >= 0.0:
        raise ValueError("physics gravity must point down")
    if result.settle_duration_s < 2.0:
        raise ValueError("physics settle duration must be at least 2 seconds")
    if not 0.0 < result.tail_duration_s < result.settle_duration_s:
        raise ValueError("physics tail duration is invalid")
    for duration, label in (
        (result.settle_duration_s, "settle_duration_s"),
        (result.tail_duration_s, "tail_duration_s"),
    ):
        steps = duration * result.rate_hz
        if abs(steps - round(steps)) > 1.0e-9:
            raise ValueError(f"physics.{label} must contain whole steps")
    positive_names = (
        "minimum_vertical_drop_m",
        "maximum_vertical_drop_m",
        "maximum_transient_table_penetration_m",
        "maximum_final_surface_gap_m",
        "maximum_xy_drift_m",
        "maximum_tail_displacement_m",
        "maximum_tail_linear_speed_m_s",
        "maximum_tail_angular_speed_rad_s",
        "maximum_upright_axis_tilt_rad",
        "maximum_fixed_translation_drift_m",
        "maximum_fixed_rotation_drift_rad",
    )
    if any(getattr(result, name) <= 0.0 for name in positive_names):
        raise ValueError("physics gate thresholds must be positive")
    if result.minimum_vertical_drop_m >= result.maximum_vertical_drop_m:
        raise ValueError("physics vertical-drop interval is invalid")
    if result.maximum_transient_table_penetration_m > 0.005:
        raise ValueError("physics penetration gate exceeds safety bound")
    if result.maximum_upright_axis_tilt_rad > math.radians(10.0):
        raise ValueError("physics tilt gate exceeds 10 degrees")
    return result


def _load_render(value: Any) -> D38999TabletopRender:
    document = _mapping(value, "render")
    keys = tuple(D38999TabletopRender.__dataclass_fields__)
    _exact_keys(document, keys, "render")
    return D38999TabletopRender(
        camera_eye_m=_vector3(
            document["camera_eye_m"], "render.camera_eye_m"
        ),
        camera_target_m=_vector3(
            document["camera_target_m"], "render.camera_target_m"
        ),
        dome_light_intensity=_positive(
            document["dome_light_intensity"],
            "render.dome_light_intensity",
        ),
        dome_light_color_rgb=_color(
            document["dome_light_color_rgb"],
            "render.dome_light_color_rgb",
        ),
        key_light_intensity=_positive(
            document["key_light_intensity"],
            "render.key_light_intensity",
        ),
        key_light_color_rgb=_color(
            document["key_light_color_rgb"],
            "render.key_light_color_rgb",
        ),
        key_light_rotation_degrees_xyz=_vector3(
            document["key_light_rotation_degrees_xyz"],
            "render.key_light_rotation_degrees_xyz",
        ),
    )


def _validate_geometry(config: D38999TabletopScene) -> None:
    expected_root = "/World/D38999TabletopV1"
    if config.world.root_prim_path != expected_root:
        raise ValueError("world root is not canonical")
    if config.asset.reference_prim_path != expected_root + "/D38999Pair":
        raise ValueError("asset reference path is not canonical")
    for path in (
        config.world.physics_material_prim_path,
        config.table.prim_path,
        config.fixed_endpoint.fixture_prim_path,
    ):
        if not path.startswith(expected_root + "/"):
            raise ValueError("scene prim paths must remain inside root")
    if abs(config.table.top_z_m - 0.200) > 1.0e-9:
        raise ValueError("table top must retain the validated 0.200 m height")
    fixed = config.fixed_endpoint
    if abs(fixed.fixture_top_z_m - config.table.top_z_m - 0.040) > 1.0e-9:
        raise ValueError("fixture must sit exactly on the table")
    fixed_bottom = (
        fixed.receptacle_origin_m[2] - fixed.receptacle_bottom_offset_m
    )
    if abs(fixed_bottom - fixed.fixture_top_z_m) > 1.0e-9:
        raise ValueError("fixed receptacle must sit exactly on fixture")
    if (
        abs(fixed.receptacle_origin_m[0] - fixed.fixture_center_m[0])
        > 1.0e-9
        or abs(fixed.receptacle_origin_m[1] - fixed.fixture_center_m[1])
        > 1.0e-9
    ):
        raise ValueError("fixed receptacle must be centered on fixture")
    loose = config.loose_endpoint
    clearance = loose.initial_bottom_z_m - config.table.top_z_m
    if abs(clearance - loose.initial_clearance_above_table_m) > 1.0e-9:
        raise ValueError("loose plug must start at declared clearance")
    if not 0.010 <= clearance <= 0.020:
        raise ValueError("loose plug clearance must be approximately 15 mm")
    separation = math.hypot(
        loose.initial_origin_m[0] - fixed.receptacle_origin_m[0],
        loose.initial_origin_m[1] - fixed.receptacle_origin_m[1],
    )
    if separation < loose.minimum_endpoint_separation_m:
        raise ValueError("D38999 endpoints are not physically separated")
    profile = config.asset_profile
    if profile.profile_id == _LEGACY_PROFILE_ID:
        expected_endpoint_geometry = {
            "fixed_origin_z_m": 0.2615,
            "fixed_bottom_offset_m": 0.0215,
            "loose_initial_origin_z_m": 0.215,
            "body_bottom_offset_m": 0.0,
            "nut_bottom_offset_m": 0.007,
            "loose_settled_origin_z_m": 0.200,
        }
    else:
        expected_endpoint_geometry = {
            "fixed_origin_z_m": 0.272,
            "fixed_bottom_offset_m": 0.032,
            "loose_initial_origin_z_m": 0.2455,
            "body_bottom_offset_m": -0.0305,
            "nut_bottom_offset_m": -0.0305,
            "loose_settled_origin_z_m": 0.2305,
        }
    actual_endpoint_geometry = {
        "fixed_origin_z_m": fixed.receptacle_origin_m[2],
        "fixed_bottom_offset_m": fixed.receptacle_bottom_offset_m,
        "loose_initial_origin_z_m": loose.initial_origin_m[2],
        "body_bottom_offset_m": loose.body_bottom_offset_m,
        "nut_bottom_offset_m": loose.nut_bottom_offset_m,
        "loose_settled_origin_z_m": config.loose_settled_origin_m[2],
    }
    for name, expected in expected_endpoint_geometry.items():
        if not math.isclose(
            actual_endpoint_geometry[name], expected, abs_tol=1.0e-12
        ):
            raise ValueError(
                f"{name} is not canonical for tabletop asset profile"
            )
    table_half_x = 0.5 * config.table.size_m[0]
    table_half_y = 0.5 * config.table.size_m[1]
    for x, y, radius in (
        (*loose.initial_origin_m[:2], loose.footprint_radius_m),
        (*fixed.receptacle_origin_m[:2], 0.023),
    ):
        if (
            abs(x - config.table.center_m[0]) + radius > table_half_x
            or abs(y - config.table.center_m[1]) + radius > table_half_y
        ):
            raise ValueError("D38999 endpoint lies outside the table")


def load_d38999_tabletop_scene(
    config_path: Path | str = DEFAULT_D38999_TABLETOP_CONFIG_PATH,
    *,
    authorized_local_asset_path: str | None = None,
) -> D38999TabletopScene:
    """Load and validate the independent D38999 tabletop contract."""
    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    root = _mapping(document, "root")
    base_keys = (
        "schema_version",
        "asset",
        "world",
        "table",
        "fixed_endpoint",
        "loose_endpoint",
        "physics",
        "render",
    )
    schema_version = root.get("schema_version")
    if schema_version not in _SCHEMA_PROFILE_IDS:
        raise ValueError("unsupported D38999 tabletop schema")
    keys = (
        base_keys
        if schema_version == D38999_TABLETOP_SCHEMA_VERSION
        else base_keys + ("asset_profile",)
    )
    _exact_keys(root, keys, "root")
    profile = _load_profile(root.get("asset_profile"), schema_version)
    config = D38999TabletopScene(
        schema_version=schema_version,
        asset=_load_asset(root["asset"], profile, authorized_local_asset_path),
        world=_load_world(root["world"]),
        table=_load_surface(root["table"]),
        fixed_endpoint=_load_fixed(root["fixed_endpoint"]),
        loose_endpoint=_load_loose(root["loose_endpoint"]),
        physics=_load_physics(root["physics"]),
        render=_load_render(root["render"]),
    )
    _validate_geometry(config)
    config.as_dict()
    return config


def verify_d38999_tabletop_asset(
    config: D38999TabletopScene,
    repository_root: Path | str,
    *,
    authorized_local_asset_path: str | None = None,
    authorized_model: Any | None = None,
) -> Path:
    """Resolve one of the two allowlisted D38999 tabletop assets."""
    repository = Path(repository_root).expanduser().resolve()
    path = (repository / config.asset.local_path).resolve()
    try:
        path.relative_to(repository)
    except ValueError as error:
        raise ValueError("D38999 asset escapes repository") from error
    if path.name == LEGACY_ASSET_BASENAME:
        raise ValueError("legacy synthetic connector asset is forbidden")
    asset_contract = _PROFILE_ASSET_CONTRACTS[config.asset_profile.profile_id]
    if not _asset_path_is_allowed(
        config.asset_profile.profile_id,
        config.asset.local_path,
        authorized_local_asset_path,
    ) or not path.is_file():
        raise ValueError("allowlisted D38999 tabletop asset is missing")
    if config.asset_profile.profile_id in _PHYSICAL_PROFILE_IDS:
        source_path = (repository / config.asset_profile.source_config).resolve()
        try:
            source_path.relative_to(repository)
        except ValueError as error:
            raise ValueError("keyed-v2 source config escapes repository") from error
        if config.asset_profile.profile_id == _MULTILAYER_GRASP_PROFILE_ID:
            build_result_path = (
                repository / asset_contract["build_result_path"]
            ).resolve()
            if not build_result_path.is_file():
                raise ValueError("multilayer grasp build result is missing")
            if _sha256(build_result_path) != asset_contract["build_result_sha256"]:
                raise ValueError("multilayer grasp build result digest changed")
            build_result = json.loads(
                build_result_path.read_text(encoding="utf-8")
            )
            assembly_result = build_result.get("assembly_control", {})
            determinism = build_result.get("determinism", {})
            current_asset_sha256 = _sha256(path)
            if (
                build_result.get("task_id")
                != "DYN-A2-NOMINAL-INSERTION-V2"
                or build_result.get("outcome")
                != "STATIC_PASS_AWAITING_DYNAMIC_VALIDATION"
                or build_result.get("classification")
                != "FROZEN_NUT_BODY_PHYSICAL_SHOULDER_RESTORED"
                or assembly_result.get("path") != config.asset.local_path
                or assembly_result.get("sha256_after") != current_asset_sha256
                or build_result.get("master_contract", {}).get("path")
                != config.asset_profile.source_config
                or build_result.get("master_contract", {}).get("sha256")
                != _sha256(source_path)
                or determinism.get("identical") is not True
                or determinism.get("generation_pass_1_sha256")
                != current_asset_sha256
                or determinism.get("generation_pass_2_sha256")
                != current_asset_sha256
                or build_result.get("event_positions_modified") is not False
                or build_result.get("safety_limits_modified") is not False
                or build_result.get("visual_model_modified") is not False
                or build_result.get("local_contact_reference_modified")
                is not False
                or build_result.get("model_mapping_modified") is not False
                or build_result.get("new_connector_geometry_candidate_created")
                is not False
            ):
                raise ValueError(
                    "multilayer grasp asset differs from guarded build result"
                )

            guarded_sources = (
                (
                    build_result.get("authorized_overrides", {}),
                    asset_contract["authorized_overrides_path"],
                ),
                (
                    build_result.get("generator", {}),
                    asset_contract["generator_path"],
                ),
                (
                    build_result.get("physical_contract", {}),
                    asset_contract["physical_contract_path"],
                ),
            )
            for evidence, expected_relative_path in guarded_sources:
                evidence_path = (repository / expected_relative_path).resolve()
                try:
                    evidence_path.relative_to(repository)
                except ValueError as error:
                    raise ValueError(
                        "multilayer grasp evidence escapes repository"
                    ) from error
                if (
                    evidence.get("path") != expected_relative_path
                    or not evidence_path.is_file()
                    or evidence.get("sha256") != _sha256(evidence_path)
                ):
                    raise ValueError(
                        "multilayer grasp source evidence digest changed"
                    )

            preserved_outputs = build_result.get("preserved_outputs_after", {})
            if set(preserved_outputs) != set(
                asset_contract["preserved_output_paths"]
            ):
                raise ValueError("multilayer preserved output set changed")
            for name, relative_path in asset_contract[
                "preserved_output_paths"
            ].items():
                preserved_path = (repository / relative_path).resolve()
                if (
                    not preserved_path.is_file()
                    or preserved_outputs.get(name) != _sha256(preserved_path)
                ):
                    raise ValueError(
                        "multilayer preserved output digest changed"
                    )
        elif config.asset_profile.profile_id == _KEYED_V3_R12_PROFILE_ID:
            from .d38999_keyed_v3_physical_r12_contract import (
                candidate_model,
                load_r12_physical_model_contract,
            )

            if authorized_local_asset_path is not None:
                if (
                    config.asset.local_path != authorized_local_asset_path
                    or authorized_model is None
                ):
                    raise ValueError("local R12 asset lacks its exact authorized model")
                source = authorized_model
            else:
                source = load_r12_physical_model_contract(source_path)
                match = re.search(r"r12_candidate_(0[1-4])", config.asset.local_path)
                if match:
                    source = candidate_model(source, int(match.group(1)))
        else:
            from .d38999_keyed_v2_physical_model_contract import (
                load_physical_model_contract,
            )

            source = load_physical_model_contract(source_path)
        if config.asset_profile.profile_id != _MULTILAYER_GRASP_PROFILE_ID:
            if (
                source.document["identity"]["pair_model_id"]
                != config.asset_profile.profile_id
                or config.asset.proxy_id != config.asset_profile.profile_id
                or path.name
                != source.document["identity"]["recommended_asset_name"]
            ):
                raise ValueError("physical asset source identity differs")
    return path


def author_d38999_tabletop_scene(
    stage: Any,
    config: D38999TabletopScene,
    asset_path: Path | str,
    *,
    add_reference_to_stage: Any,
    Gf: Any,
    Sdf: Any,
    UsdGeom: Any,
    UsdPhysics: Any,
    UsdShade: Any,
    physics_utils: Any,
    authorized_local_asset_path: str | None = None,
) -> dict[str, Any]:
    """Author every object transform before the caller starts physics."""
    asset = Path(asset_path).expanduser().resolve()
    if (
        not _asset_path_is_allowed(
            config.asset_profile.profile_id,
            config.asset.local_path,
            authorized_local_asset_path,
        )
        or asset.name != Path(config.asset.local_path).name
        or not asset.is_file()
    ):
        raise ValueError("unexpected D38999 tabletop asset")
    if stage.GetPrimAtPath(config.world.root_prim_path).IsValid():
        raise RuntimeError("D38999 tabletop root already exists")
    UsdGeom.Xform.Define(stage, config.world.root_prim_path)

    def static_cube(path, center, size, color):
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(1.0)
        cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        transform = UsdGeom.Xformable(cube)
        transform.AddTranslateOp().Set(Gf.Vec3d(*center))
        transform.AddScaleOp().Set(Gf.Vec3f(*size))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        return cube.GetPrim()

    def unscaled_rigid_box_mesh(path, center, size, color):
        """Author an actual-size box whose rigid frame has no scale op.

        Joint anchors are expressed in a rigid body's local frame.  Placing a
        non-uniform scale on that same prim changes the interpretation of the
        anchor coordinates and can make PhysX repair a disjoint fixed joint on
        the first step.  The physical-r7 fixture therefore carries its metric
        dimensions in mesh points and has translation as its only xform op.
        """

        half_x, half_y, half_z = (0.5 * float(value) for value in size)
        points = [
            Gf.Vec3f(-half_x, -half_y, -half_z),
            Gf.Vec3f(half_x, -half_y, -half_z),
            Gf.Vec3f(half_x, half_y, -half_z),
            Gf.Vec3f(-half_x, half_y, -half_z),
            Gf.Vec3f(-half_x, -half_y, half_z),
            Gf.Vec3f(half_x, -half_y, half_z),
            Gf.Vec3f(half_x, half_y, half_z),
            Gf.Vec3f(-half_x, half_y, half_z),
        ]
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr([4, 4, 4, 4, 4, 4])
        mesh.CreateFaceVertexIndicesAttr(
            [
                0, 3, 2, 1,
                4, 5, 6, 7,
                0, 1, 5, 4,
                1, 2, 6, 5,
                2, 3, 7, 6,
                3, 0, 4, 7,
            ]
        )
        mesh.CreateExtentAttr(
            [
                Gf.Vec3f(-half_x, -half_y, -half_z),
                Gf.Vec3f(half_x, half_y, half_z),
            ]
        )
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        UsdGeom.Xformable(mesh).AddTranslateOp().Set(Gf.Vec3d(*center))
        prim = mesh.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
        mesh_collision.CreateApproximationAttr().Set("convexHull")
        return prim

    table_material = UsdShade.Material.Define(
        stage, config.world.physics_material_prim_path
    )
    table_material_api = UsdPhysics.MaterialAPI.Apply(
        table_material.GetPrim()
    )
    table_material_api.CreateStaticFrictionAttr(config.table.static_friction)
    table_material_api.CreateDynamicFrictionAttr(config.table.dynamic_friction)
    table_material_api.CreateRestitutionAttr(config.table.restitution)
    table_prim = static_cube(
        config.table.prim_path,
        config.table.center_m,
        config.table.size_m,
        config.table.color_rgb,
    )
    if config.asset_profile.profile_id in _FIXTURE_JOINT_PROFILE_IDS:
        fixture_prim = unscaled_rigid_box_mesh(
            config.fixed_endpoint.fixture_prim_path,
            config.fixed_endpoint.fixture_center_m,
            config.fixed_endpoint.fixture_size_m,
            config.fixed_endpoint.fixture_color_rgb,
        )
    else:
        fixture_prim = static_cube(
            config.fixed_endpoint.fixture_prim_path,
            config.fixed_endpoint.fixture_center_m,
            config.fixed_endpoint.fixture_size_m,
            config.fixed_endpoint.fixture_color_rgb,
        )
    physics_utils.add_physics_material_to_prim(
        stage,
        table_prim,
        Sdf.Path(config.world.physics_material_prim_path),
    )
    if table_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError("static tabletop collider became dynamic")

    fixture_material_path = (
        config.world.root_prim_path + "/FixtureAndReceptacleMaterial"
    )
    if config.asset_profile.profile_id in _PHYSICAL_PROFILE_IDS:
        fixture_material = UsdShade.Material.Define(
            stage, fixture_material_path
        )
        fixture_material_api = UsdPhysics.MaterialAPI.Apply(
            fixture_material.GetPrim()
        )
        fixture_material_api.CreateStaticFrictionAttr(0.35)
        fixture_material_api.CreateDynamicFrictionAttr(0.25)
        fixture_material_api.CreateRestitutionAttr(0.0)
        physics_utils.add_physics_material_to_prim(
            stage,
            fixture_prim,
            Sdf.Path(fixture_material_path),
        )
        if config.asset_profile.profile_id in _FIXTURE_JOINT_PROFILE_IDS:
            fixture_rigid = UsdPhysics.RigidBodyAPI.Apply(fixture_prim)
            fixture_rigid.CreateRigidBodyEnabledAttr(True)
            fixture_rigid.CreateKinematicEnabledAttr(False)
            fixture_mass = UsdPhysics.MassAPI.Apply(fixture_prim)
            fixture_mass.CreateMassAttr(5.0)
            fixture_mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
            fixture_mass.CreateDiagonalInertiaAttr(
                Gf.Vec3f(
                    0.0088333333333,
                    0.0088333333333,
                    0.0163333333333,
                )
            )
            fixture_mass.CreatePrincipalAxesAttr(Gf.Quatf(1.0))
    else:
        physics_utils.add_physics_material_to_prim(
            stage,
            fixture_prim,
            Sdf.Path(config.world.physics_material_prim_path),
        )
        if fixture_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError("legacy static fixture became dynamic")

    add_reference_to_stage(str(asset), config.asset.reference_prim_path)
    if config.asset_profile.profile_id in _PHYSICAL_PROFILE_IDS:
        model_root = stage.GetPrimAtPath(config.asset.model_root_prim_path)
        if not model_root.IsValid():
            raise RuntimeError("keyed-v2 referenced model root is missing")
        for name, expected in _PHYSICAL_MODEL_IDS[
            config.asset_profile.profile_id
        ].items():
            identity_attribute = model_root.GetAttribute(f"kcg:{name}")
            actual = identity_attribute.Get() if identity_attribute else None
            if actual != expected:
                raise RuntimeError(
                    f"keyed-v2 referenced model identity differs: {name}"
                )

    def set_single_translation(path, value):
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"D38999 asset prim is missing: {path}")
        xformable = UsdGeom.Xformable(prim)
        operations = xformable.GetOrderedXformOps()
        if not operations:
            operation = xformable.AddTranslateOp()
        elif (
            len(operations) == 1
            and operations[0].GetOpType() == UsdGeom.XformOp.TypeTranslate
        ):
            operation = operations[0]
        else:
            raise RuntimeError(f"incompatible transform stack: {path}")
        operation.Set(Gf.Vec3d(*value))

    def set_translation_and_rotation(path, value, rotation_degrees_xyz):
        if rotation_degrees_xyz == (0.0, 0.0, 0.0):
            set_single_translation(path, value)
            return
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"D38999 asset prim is missing: {path}")
        xformable = UsdGeom.Xformable(prim)
        if xformable.GetOrderedXformOps():
            raise RuntimeError(f"incompatible transform stack: {path}")
        xformable.AddTranslateOp().Set(Gf.Vec3d(*value))
        xformable.AddRotateXYZOp().Set(Gf.Vec3f(*rotation_degrees_xyz))

    set_translation_and_rotation(
        config.asset.fixed_receptacle_prim_path,
        config.fixed_endpoint.receptacle_origin_m,
        config.asset_profile.fixed_endpoint_rotation_degrees_xyz,
    )
    set_translation_and_rotation(
        config.asset.loose_plug_prim_path,
        config.loose_endpoint.initial_origin_m,
        config.asset_profile.loose_endpoint_rotation_degrees_xyz,
    )

    fixed = stage.GetPrimAtPath(config.asset.fixed_receptacle_prim_path)
    body = stage.GetPrimAtPath(config.asset.body_prim_path)
    nut = stage.GetPrimAtPath(config.asset.nut_prim_path)
    joint = stage.GetPrimAtPath(config.asset.joint_prim_path)
    if config.asset_profile.profile_id in _PHYSICAL_PROFILE_IDS:
        if not fixed.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError("physical fixed receptacle is not a rigid body")
        fixed_rigid = UsdPhysics.RigidBodyAPI(fixed)
        if fixed_rigid.GetRigidBodyEnabledAttr().Get() is not True:
            raise RuntimeError("physical fixed receptacle rigid body is disabled")
        expected_kinematic = bool(
            config.asset_profile.profile_id == _MULTILAYER_GRASP_PROFILE_ID
        )
        if fixed_rigid.GetKinematicEnabledAttr().Get() is not expected_kinematic:
            raise RuntimeError(
                "physical fixed receptacle kinematic contract differs"
            )
    elif fixed.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError("legacy fixed receptacle unexpectedly became dynamic")
    if not body.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError("D38999 plug body is not dynamic")
    if not nut.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError("D38999 coupling nut is not dynamic")
    if config.asset_profile.profile_id in _PHYSICAL_PROFILE_IDS:
        if not joint.IsA(UsdPhysics.Joint) or joint.IsA(
            UsdPhysics.RevoluteJoint
        ):
            raise RuntimeError("physical D6 coupling-nut joint is missing")

    if config.asset_profile.profile_id in _FIXTURE_JOINT_PROFILE_IDS:
        joints_root = config.world.root_prim_path + "/Joints"
        UsdGeom.Scope.Define(stage, joints_root)
        fixture_world_path = joints_root + "/FixtureToWorld"
        receptacle_fixture_path = joints_root + "/ReceptacleToFixture"

        fixture_world = UsdPhysics.FixedJoint.Define(
            stage, fixture_world_path
        )
        fixture_world.CreateBody0Rel().SetTargets([])
        fixture_world.CreateBody1Rel().SetTargets(
            [Sdf.Path(config.fixed_endpoint.fixture_prim_path)]
        )
        fixture_world.CreateLocalPos0Attr(
            Gf.Vec3f(*config.fixed_endpoint.fixture_center_m)
        )
        fixture_world.CreateLocalRot0Attr(Gf.Quatf(1.0))
        fixture_world.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        fixture_world.CreateLocalRot1Attr(Gf.Quatf(1.0))
        fixture_world.CreateJointEnabledAttr(True)
        fixture_world.CreateExcludeFromArticulationAttr(False)
        fixture_world.CreateCollisionEnabledAttr(False)

        receptacle_fixture = UsdPhysics.FixedJoint.Define(
            stage, receptacle_fixture_path
        )
        receptacle_fixture.CreateBody0Rel().SetTargets(
            [Sdf.Path(config.fixed_endpoint.fixture_prim_path)]
        )
        receptacle_fixture.CreateBody1Rel().SetTargets(
            [Sdf.Path(config.asset.fixed_receptacle_prim_path)]
        )
        receptacle_fixture.CreateLocalPos0Attr(
            Gf.Vec3f(0.0, 0.0, 0.052)
        )
        receptacle_fixture.CreateLocalRot0Attr(
            Gf.Quatf(0.0, Gf.Vec3f(1.0, 0.0, 0.0))
        )
        receptacle_fixture.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        receptacle_fixture.CreateLocalRot1Attr(Gf.Quatf(1.0))
        receptacle_fixture.CreateJointEnabledAttr(True)
        receptacle_fixture.CreateExcludeFromArticulationAttr(False)
        receptacle_fixture.CreateCollisionEnabledAttr(False)
    elif (
        config.asset_profile.profile_id not in _PHYSICAL_PROFILE_IDS
        and not joint.IsA(UsdPhysics.RevoluteJoint)
    ):
        raise RuntimeError("legacy D38999 coupling nut joint is missing")

    result = {
        "body_prim_path": config.asset.body_prim_path,
        "fixed_receptacle_prim_path": (
            config.asset.fixed_receptacle_prim_path
        ),
        "fixture_prim_path": config.fixed_endpoint.fixture_prim_path,
        "fixture_material_prim_path": (
            fixture_material_path
            if config.asset_profile.profile_id in _PHYSICAL_PROFILE_IDS
            else config.world.physics_material_prim_path
        ),
        "joint_prim_path": config.asset.joint_prim_path,
        "nut_prim_path": config.asset.nut_prim_path,
        "object_pose_writes_after_start": 0,
        "table_prim_path": config.table.prim_path,
    }
    if config.asset_profile.profile_id in _FIXTURE_JOINT_PROFILE_IDS:
        result["fixture_to_world_joint_path"] = fixture_world_path
        result["receptacle_to_fixture_joint_path"] = (
            receptacle_fixture_path
        )
        result["fixed_load_path"] = (
            "FixedReceptacle->FixedFixture->world"
        )
    elif config.asset_profile.profile_id == _MULTILAYER_GRASP_PROFILE_ID:
        result["fixed_load_path"] = "FixedReceptacle(kinematic)->world"
    result["asset_profile_id"] = config.asset_profile.profile_id
    result["asset_source_config"] = config.asset_profile.source_config
    json.dumps(result, allow_nan=False, sort_keys=True)
    return result


__all__ = [
    "DEFAULT_D38999_TABLETOP_CONFIG_PATH",
    "D38999_TABLETOP_SCHEMA_VERSION",
    "D38999_TABLETOP_SCHEMA_VERSION_KEYED_V2",
    "D38999_TABLETOP_SCHEMA_VERSION_KEYED_V3_R12",
    "D38999_TABLETOP_SCHEMA_VERSION_MULTILAYER_GRASP",
    "D38999TabletopAssetProfile",
    "D38999TabletopScene",
    "author_d38999_tabletop_scene",
    "load_d38999_tabletop_scene",
    "verify_d38999_tabletop_asset",
]
