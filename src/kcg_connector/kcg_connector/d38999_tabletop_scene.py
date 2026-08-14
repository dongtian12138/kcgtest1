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
DEFAULT_D38999_TABLETOP_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "config/d38999_tabletop_scene_v1.yaml"
)
EXPECTED_ASSET_BASENAME = "d38999_shell25j_61_pair_proxy_v1.usda"
LEGACY_ASSET_BASENAME = "connector_pair.usda"
_PRIM_PATTERN = re.compile(r"^/World(?:/[A-Za-z_][A-Za-z0-9_]*)+$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


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
    sha256: str
    proxy_id: str
    reference_prim_path: str
    model_root_prim_path: str
    loose_plug_prim_path: str
    body_prim_path: str
    nut_prim_path: str
    joint_prim_path: str
    fixed_receptacle_prim_path: str


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

    def as_dict(self) -> dict[str, Any]:
        return json.loads(
            json.dumps(asdict(self), allow_nan=False, sort_keys=True)
        )


def _load_asset(value: Any) -> D38999TabletopAsset:
    document = _mapping(value, "asset")
    keys = tuple(D38999TabletopAsset.__dataclass_fields__)
    _exact_keys(document, keys, "asset")
    local_path = _text(document["local_path"], "asset.local_path")
    if Path(local_path).is_absolute() or ".." in Path(local_path).parts:
        raise ValueError("asset.local_path must be repository-relative")
    if Path(local_path).name != EXPECTED_ASSET_BASENAME:
        raise ValueError("asset must be the independent D38999 proxy USD")
    sha256 = _text(document["sha256"], "asset.sha256")
    if not _SHA256_PATTERN.fullmatch(sha256):
        raise ValueError("asset.sha256 must be lowercase SHA-256")
    values = {
        name: _prim_path(document[name], f"asset.{name}")
        for name in keys
        if name.endswith("_prim_path")
    }
    result = D38999TabletopAsset(
        local_path=local_path,
        sha256=sha256,
        proxy_id=_text(document["proxy_id"], "asset.proxy_id"),
        **values,
    )
    if result.proxy_id != "d38999_shell25j_61_pair_proxy_v1":
        raise ValueError("asset.proxy_id is unsupported")
    prefix = result.reference_prim_path + "/D38999Shell25JProxy"
    if result.model_root_prim_path != prefix:
        raise ValueError("asset model root does not match its reference")
    expected = {
        "loose_plug_prim_path": prefix + "/LoosePlug",
        "body_prim_path": prefix + "/LoosePlug/BodyAssembly",
        "nut_prim_path": prefix + "/LoosePlug/CouplingNut",
        "joint_prim_path": prefix + "/LoosePlug/CouplingNutJoint",
        "fixed_receptacle_prim_path": prefix + "/FixedReceptacle",
    }
    for name, wanted in expected.items():
        if getattr(result, name) != wanted:
            raise ValueError(f"asset.{name} does not match D38999 USD")
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
) -> D38999TabletopScene:
    """Load and validate the independent D38999 tabletop contract."""
    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    root = _mapping(document, "root")
    keys = (
        "schema_version",
        "asset",
        "world",
        "table",
        "fixed_endpoint",
        "loose_endpoint",
        "physics",
        "render",
    )
    _exact_keys(root, keys, "root")
    if root["schema_version"] != D38999_TABLETOP_SCHEMA_VERSION:
        raise ValueError("unsupported D38999 tabletop schema")
    config = D38999TabletopScene(
        schema_version=D38999_TABLETOP_SCHEMA_VERSION,
        asset=_load_asset(root["asset"]),
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
    config: D38999TabletopScene, repository_root: Path | str
) -> Path:
    """Resolve and hash-pin the independent D38999 proxy asset."""
    repository = Path(repository_root).expanduser().resolve()
    path = (repository / config.asset.local_path).resolve()
    try:
        path.relative_to(repository)
    except ValueError as error:
        raise ValueError("D38999 asset escapes repository") from error
    if path.name == LEGACY_ASSET_BASENAME:
        raise ValueError("legacy synthetic connector asset is forbidden")
    if path.name != EXPECTED_ASSET_BASENAME or not path.is_file():
        raise ValueError("independent D38999 proxy asset is missing")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != config.asset.sha256:
        raise ValueError("D38999 tabletop asset SHA-256 mismatch")
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
) -> dict[str, Any]:
    """Author every object transform before the caller starts physics."""
    asset = Path(asset_path).expanduser().resolve()
    if asset.name != EXPECTED_ASSET_BASENAME or not asset.is_file():
        raise ValueError("unexpected D38999 proxy asset")
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

    material = UsdShade.Material.Define(
        stage, config.world.physics_material_prim_path
    )
    material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    material_api.CreateStaticFrictionAttr(config.table.static_friction)
    material_api.CreateDynamicFrictionAttr(config.table.dynamic_friction)
    material_api.CreateRestitutionAttr(config.table.restitution)
    table_prim = static_cube(
        config.table.prim_path,
        config.table.center_m,
        config.table.size_m,
        config.table.color_rgb,
    )
    fixture_prim = static_cube(
        config.fixed_endpoint.fixture_prim_path,
        config.fixed_endpoint.fixture_center_m,
        config.fixed_endpoint.fixture_size_m,
        config.fixed_endpoint.fixture_color_rgb,
    )
    for prim in (table_prim, fixture_prim):
        physics_utils.add_physics_material_to_prim(
            stage,
            prim,
            Sdf.Path(config.world.physics_material_prim_path),
        )
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError("static tabletop collider became dynamic")

    add_reference_to_stage(str(asset), config.asset.reference_prim_path)

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

    set_single_translation(
        config.asset.fixed_receptacle_prim_path,
        config.fixed_endpoint.receptacle_origin_m,
    )
    set_single_translation(
        config.asset.loose_plug_prim_path,
        config.loose_endpoint.initial_origin_m,
    )

    fixed = stage.GetPrimAtPath(config.asset.fixed_receptacle_prim_path)
    body = stage.GetPrimAtPath(config.asset.body_prim_path)
    nut = stage.GetPrimAtPath(config.asset.nut_prim_path)
    joint = stage.GetPrimAtPath(config.asset.joint_prim_path)
    if fixed.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError(
            "fixed D38999 receptacle unexpectedly became dynamic"
        )
    if not body.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError("D38999 plug body is not dynamic")
    if not nut.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError("D38999 coupling nut is not dynamic")
    if not joint.IsA(UsdPhysics.RevoluteJoint):
        raise RuntimeError("D38999 coupling nut joint is missing")

    result = {
        "asset_sha256": config.asset.sha256,
        "body_prim_path": config.asset.body_prim_path,
        "fixed_receptacle_prim_path": (
            config.asset.fixed_receptacle_prim_path
        ),
        "fixture_prim_path": config.fixed_endpoint.fixture_prim_path,
        "joint_prim_path": config.asset.joint_prim_path,
        "nut_prim_path": config.asset.nut_prim_path,
        "object_pose_writes_after_start": 0,
        "table_prim_path": config.table.prim_path,
    }
    json.dumps(result, allow_nan=False, sort_keys=True)
    return result


__all__ = [
    "DEFAULT_D38999_TABLETOP_CONFIG_PATH",
    "D38999_TABLETOP_SCHEMA_VERSION",
    "D38999TabletopScene",
    "author_d38999_tabletop_scene",
    "load_d38999_tabletop_scene",
    "verify_d38999_tabletop_asset",
]
