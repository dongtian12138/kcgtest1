"""Pure configuration and deferred USD authoring for tabletop scene v1.

The module deliberately does not import Isaac Sim, Omni, or Pixar USD at
module import time.  The standalone Isaac entry point passes those runtime
modules into :func:`author_isaac_tabletop_scene` after ``SimulationApp`` has
started.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from numbers import Integral, Real
from pathlib import Path
import re
from typing import Any, Mapping

import yaml


TABLETOP_SCHEMA_VERSION = "kcg_connector_tabletop_scene_v1"
DEFAULT_TABLETOP_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "config/connector_tabletop_scene_v1.yaml"
)
_PRIM_PATH_PATTERN = re.compile(
    r"^/World(?:/[A-Za-z_][A-Za-z0-9_]*)+$"
)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _exact_keys(
    value: Mapping[str, Any], expected: tuple[str, ...], name: str
) -> None:
    actual = set(value)
    required = set(expected)
    if actual != required:
        missing = sorted(required - actual)
        unexpected = sorted(actual - required)
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if unexpected:
            details.append("unexpected=" + ",".join(unexpected))
        raise ValueError(f"{name} keys are invalid: {'; '.join(details)}")


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_float(value: Any, name: str) -> float:
    result = _finite_float(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _vector3(
    value: Any, name: str, *, positive: bool = False
) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly three numbers")
    result = tuple(
        _finite_float(item, f"{name}[{index}]")
        for index, item in enumerate(value)
    )
    if positive and any(item <= 0.0 for item in result):
        raise ValueError(f"{name} entries must be positive")
    return result


def _color(value: Any, name: str) -> tuple[float, float, float]:
    result = _vector3(value, name)
    if any(item < 0.0 or item > 1.0 for item in result):
        raise ValueError(f"{name} entries must be in [0, 1]")
    return result


def _prim_path(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _PRIM_PATH_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must be an absolute /World USD prim path")
    return value


@dataclass(frozen=True)
class TabletopWorld:
    root_prim_path: str
    connector_reference_prim_path: str
    physics_material_prim_path: str


@dataclass(frozen=True)
class TabletopSurface:
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
class FixedEndpoint:
    fixture_prim_path: str
    fixture_center_m: tuple[float, float, float]
    fixture_size_m: tuple[float, float, float]
    fixture_color_rgb: tuple[float, float, float]
    receptacle_prim_path: str
    receptacle_origin_m: tuple[float, float, float]
    receptacle_bottom_offset_m: float

    @property
    def fixture_top_z_m(self) -> float:
        return self.fixture_center_m[2] + 0.5 * self.fixture_size_m[2]


@dataclass(frozen=True)
class LooseEndpoint:
    plug_prim_path: str
    body_prim_path: str
    nut_prim_path: str
    initial_center_m: tuple[float, float, float]
    body_bottom_offset_m: float
    nut_bottom_offset_m: float
    footprint_radius_m: float
    minimum_endpoint_separation_m: float

    @property
    def initial_bottom_z_m(self) -> float:
        return self.initial_center_m[2] - max(
            self.body_bottom_offset_m, self.nut_bottom_offset_m
        )


@dataclass(frozen=True)
class TabletopPhysics:
    rate_hz: int
    gravity_m_s2: float
    settle_duration_s: float
    tail_duration_s: float
    minimum_vertical_drop_m: float
    maximum_vertical_drop_m: float
    maximum_table_penetration_m: float
    maximum_surface_gap_m: float
    maximum_pickup_xy_drift_m: float
    maximum_tail_displacement_m: float
    maximum_final_linear_speed_m_s: float
    maximum_final_angular_speed_rad_s: float
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
class TabletopRender:
    camera_eye_m: tuple[float, float, float]
    camera_target_m: tuple[float, float, float]
    dome_light_intensity: float
    dome_light_color_rgb: tuple[float, float, float]
    key_light_intensity: float
    key_light_color_rgb: tuple[float, float, float]
    key_light_rotation_degrees_xyz: tuple[float, float, float]


@dataclass(frozen=True)
class ConnectorTabletopScene:
    schema_version: str
    world: TabletopWorld
    table: TabletopSurface
    fixed_endpoint: FixedEndpoint
    loose_endpoint: LooseEndpoint
    physics: TabletopPhysics
    render: TabletopRender

    def as_dict(self) -> dict[str, Any]:
        return json.loads(
            json.dumps(asdict(self), allow_nan=False, sort_keys=True)
        )


def _load_world(value: Any) -> TabletopWorld:
    document = _mapping(value, "world")
    _exact_keys(
        document,
        (
            "root_prim_path",
            "connector_reference_prim_path",
            "physics_material_prim_path",
        ),
        "world",
    )
    return TabletopWorld(
        root_prim_path=_prim_path(
            document["root_prim_path"], "world.root_prim_path"
        ),
        connector_reference_prim_path=_prim_path(
            document["connector_reference_prim_path"],
            "world.connector_reference_prim_path",
        ),
        physics_material_prim_path=_prim_path(
            document["physics_material_prim_path"],
            "world.physics_material_prim_path",
        ),
    )


def _load_table(value: Any) -> TabletopSurface:
    document = _mapping(value, "table")
    _exact_keys(
        document,
        (
            "prim_path",
            "center_m",
            "size_m",
            "color_rgb",
            "static_friction",
            "dynamic_friction",
            "restitution",
        ),
        "table",
    )
    static_friction = _finite_float(
        document["static_friction"], "table.static_friction"
    )
    dynamic_friction = _finite_float(
        document["dynamic_friction"], "table.dynamic_friction"
    )
    restitution = _finite_float(
        document["restitution"], "table.restitution"
    )
    if static_friction < 0.0 or dynamic_friction < 0.0:
        raise ValueError("table friction must be nonnegative")
    if static_friction < dynamic_friction:
        raise ValueError("table static friction must not be below dynamic")
    if not 0.0 <= restitution <= 1.0:
        raise ValueError("table restitution must be in [0, 1]")
    return TabletopSurface(
        prim_path=_prim_path(document["prim_path"], "table.prim_path"),
        center_m=_vector3(document["center_m"], "table.center_m"),
        size_m=_vector3(
            document["size_m"], "table.size_m", positive=True
        ),
        color_rgb=_color(document["color_rgb"], "table.color_rgb"),
        static_friction=static_friction,
        dynamic_friction=dynamic_friction,
        restitution=restitution,
    )


def _load_fixed(value: Any) -> FixedEndpoint:
    document = _mapping(value, "fixed_endpoint")
    _exact_keys(
        document,
        (
            "fixture_prim_path",
            "fixture_center_m",
            "fixture_size_m",
            "fixture_color_rgb",
            "receptacle_prim_path",
            "receptacle_origin_m",
            "receptacle_bottom_offset_m",
        ),
        "fixed_endpoint",
    )
    return FixedEndpoint(
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
        receptacle_prim_path=_prim_path(
            document["receptacle_prim_path"],
            "fixed_endpoint.receptacle_prim_path",
        ),
        receptacle_origin_m=_vector3(
            document["receptacle_origin_m"],
            "fixed_endpoint.receptacle_origin_m",
        ),
        receptacle_bottom_offset_m=_positive_float(
            document["receptacle_bottom_offset_m"],
            "fixed_endpoint.receptacle_bottom_offset_m",
        ),
    )


def _load_loose(value: Any) -> LooseEndpoint:
    document = _mapping(value, "loose_endpoint")
    _exact_keys(
        document,
        (
            "plug_prim_path",
            "body_prim_path",
            "nut_prim_path",
            "initial_center_m",
            "body_bottom_offset_m",
            "nut_bottom_offset_m",
            "footprint_radius_m",
            "minimum_endpoint_separation_m",
        ),
        "loose_endpoint",
    )
    return LooseEndpoint(
        plug_prim_path=_prim_path(
            document["plug_prim_path"], "loose_endpoint.plug_prim_path"
        ),
        body_prim_path=_prim_path(
            document["body_prim_path"], "loose_endpoint.body_prim_path"
        ),
        nut_prim_path=_prim_path(
            document["nut_prim_path"], "loose_endpoint.nut_prim_path"
        ),
        initial_center_m=_vector3(
            document["initial_center_m"],
            "loose_endpoint.initial_center_m",
        ),
        body_bottom_offset_m=_positive_float(
            document["body_bottom_offset_m"],
            "loose_endpoint.body_bottom_offset_m",
        ),
        nut_bottom_offset_m=_positive_float(
            document["nut_bottom_offset_m"],
            "loose_endpoint.nut_bottom_offset_m",
        ),
        footprint_radius_m=_positive_float(
            document["footprint_radius_m"],
            "loose_endpoint.footprint_radius_m",
        ),
        minimum_endpoint_separation_m=_positive_float(
            document["minimum_endpoint_separation_m"],
            "loose_endpoint.minimum_endpoint_separation_m",
        ),
    )


def _load_physics(value: Any) -> TabletopPhysics:
    document = _mapping(value, "physics")
    fields = (
        "rate_hz",
        "gravity_m_s2",
        "settle_duration_s",
        "tail_duration_s",
        "minimum_vertical_drop_m",
        "maximum_vertical_drop_m",
        "maximum_table_penetration_m",
        "maximum_surface_gap_m",
        "maximum_pickup_xy_drift_m",
        "maximum_tail_displacement_m",
        "maximum_final_linear_speed_m_s",
        "maximum_final_angular_speed_rad_s",
        "maximum_upright_axis_tilt_rad",
        "maximum_fixed_translation_drift_m",
        "maximum_fixed_rotation_drift_rad",
    )
    _exact_keys(document, fields, "physics")
    gravity = _finite_float(document["gravity_m_s2"], "physics.gravity_m_s2")
    if gravity >= 0.0:
        raise ValueError("physics gravity must point down")
    result = TabletopPhysics(
        rate_hz=_positive_integer(document["rate_hz"], "physics.rate_hz"),
        gravity_m_s2=gravity,
        **{
            name: _positive_float(document[name], f"physics.{name}")
            for name in fields
            if name not in {"rate_hz", "gravity_m_s2"}
        },
    )
    if result.settle_duration_s < 2.0:
        raise ValueError("physics settle duration must be at least 2 seconds")
    if result.tail_duration_s >= result.settle_duration_s:
        raise ValueError("physics tail duration must be shorter than settle")
    if result.tail_steps <= 0 or result.tail_steps >= result.settle_steps:
        raise ValueError("physics sampling windows are invalid")
    if result.minimum_vertical_drop_m >= result.maximum_vertical_drop_m:
        raise ValueError("physics vertical drop bounds are invalid")
    return result


def _load_render(value: Any) -> TabletopRender:
    document = _mapping(value, "render")
    _exact_keys(
        document,
        (
            "camera_eye_m",
            "camera_target_m",
            "dome_light_intensity",
            "dome_light_color_rgb",
            "key_light_intensity",
            "key_light_color_rgb",
            "key_light_rotation_degrees_xyz",
        ),
        "render",
    )
    result = TabletopRender(
        camera_eye_m=_vector3(
            document["camera_eye_m"], "render.camera_eye_m"
        ),
        camera_target_m=_vector3(
            document["camera_target_m"], "render.camera_target_m"
        ),
        dome_light_intensity=_positive_float(
            document["dome_light_intensity"],
            "render.dome_light_intensity",
        ),
        dome_light_color_rgb=_color(
            document["dome_light_color_rgb"],
            "render.dome_light_color_rgb",
        ),
        key_light_intensity=_positive_float(
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
    if result.camera_eye_m == result.camera_target_m:
        raise ValueError("render camera eye and target must differ")
    return result


def _inside_table_xy(
    point: tuple[float, float, float],
    radius_x: float,
    radius_y: float,
    table: TabletopSurface,
) -> bool:
    return bool(
        abs(point[0] - table.center_m[0]) + radius_x
        <= 0.5 * table.size_m[0]
        and abs(point[1] - table.center_m[1]) + radius_y
        <= 0.5 * table.size_m[1]
    )


def _validate_geometry(config: ConnectorTabletopScene) -> None:
    world = config.world
    paths = (
        world.root_prim_path,
        world.connector_reference_prim_path,
        world.physics_material_prim_path,
        config.table.prim_path,
        config.fixed_endpoint.fixture_prim_path,
        config.fixed_endpoint.receptacle_prim_path,
        config.loose_endpoint.plug_prim_path,
        config.loose_endpoint.body_prim_path,
        config.loose_endpoint.nut_prim_path,
    )
    if len(paths) != len(set(paths)):
        raise ValueError("tabletop USD prim paths must be unique")
    expected_world_paths = {
        "root": "/World/ConnectorTabletopV1",
        "connector": "/World/ConnectorTabletopV1/ConnectorPair",
        "material": "/World/ConnectorTabletopV1/TabletopMaterial",
        "table": "/World/ConnectorTabletopV1/Table",
        "fixture": "/World/ConnectorTabletopV1/FixedFixture",
    }
    actual_world_paths = {
        "root": world.root_prim_path,
        "connector": world.connector_reference_prim_path,
        "material": world.physics_material_prim_path,
        "table": config.table.prim_path,
        "fixture": config.fixed_endpoint.fixture_prim_path,
    }
    if actual_world_paths != expected_world_paths:
        raise ValueError("tabletop v1 world prim paths are not canonical")
    for path in paths[1:]:
        if not path.startswith(world.root_prim_path + "/"):
            raise ValueError("tabletop USD prim paths must share world root")
    connector = world.connector_reference_prim_path
    expected_paths = {
        "receptacle": connector + "/Receptacle",
        "plug": connector + "/Plug",
        "body": connector + "/Plug/BodyAssembly",
        "nut": connector + "/Plug/CouplingNut",
    }
    actual_paths = {
        "receptacle": config.fixed_endpoint.receptacle_prim_path,
        "plug": config.loose_endpoint.plug_prim_path,
        "body": config.loose_endpoint.body_prim_path,
        "nut": config.loose_endpoint.nut_prim_path,
    }
    if actual_paths != expected_paths:
        raise ValueError("connector child prim paths do not match the asset")

    table = config.table
    fixed = config.fixed_endpoint
    loose = config.loose_endpoint
    tolerance = 1.0e-9
    table_min_x = table.center_m[0] - 0.5 * table.size_m[0]
    if table_min_x < 0.14:
        raise ValueError("table must preserve KUKA base clearance")
    fixture_bottom = fixed.fixture_center_m[2] - 0.5 * fixed.fixture_size_m[2]
    if abs(fixture_bottom - table.top_z_m) > tolerance:
        raise ValueError("fixed fixture must sit exactly on the table")
    receptacle_bottom = (
        fixed.receptacle_origin_m[2] - fixed.receptacle_bottom_offset_m
    )
    if abs(receptacle_bottom - fixed.fixture_top_z_m) > tolerance:
        raise ValueError("fixed receptacle must sit exactly on its fixture")
    if fixed.receptacle_origin_m[:2] != fixed.fixture_center_m[:2]:
        raise ValueError("fixed receptacle must be centered on its fixture")
    initial_gap = loose.initial_bottom_z_m - table.top_z_m
    if initial_gap <= 0.0:
        raise ValueError("loose plug must start above the physical table")
    if not (
        config.physics.minimum_vertical_drop_m
        <= initial_gap
        <= config.physics.maximum_vertical_drop_m
    ):
        raise ValueError(
            "loose plug initial drop is outside acceptance bounds"
        )
    if not _inside_table_xy(
        loose.initial_center_m,
        loose.footprint_radius_m
        + config.physics.maximum_pickup_xy_drift_m,
        loose.footprint_radius_m
        + config.physics.maximum_pickup_xy_drift_m,
        table,
    ):
        raise ValueError(
            "loose plug drift envelope is outside the table"
        )
    if not _inside_table_xy(
        fixed.fixture_center_m,
        0.5 * fixed.fixture_size_m[0],
        0.5 * fixed.fixture_size_m[1],
        table,
    ):
        raise ValueError("fixed fixture is outside the table")
    endpoint_separation = math.hypot(
        loose.initial_center_m[0] - fixed.receptacle_origin_m[0],
        loose.initial_center_m[1] - fixed.receptacle_origin_m[1],
    )
    if endpoint_separation < loose.minimum_endpoint_separation_m:
        raise ValueError("loose and fixed connector ends are not separated")
    if loose.minimum_endpoint_separation_m < 0.300:
        raise ValueError("endpoint separation contract must be at least 0.3 m")

    physics = config.physics
    if physics.rate_hz != 240:
        raise ValueError("tabletop v1 physics rate must be exactly 240 Hz")
    for duration, name in (
        (physics.settle_duration_s, "settle_duration_s"),
        (physics.tail_duration_s, "tail_duration_s"),
    ):
        samples = duration * physics.rate_hz
        if not math.isclose(samples, round(samples), abs_tol=1.0e-9):
            raise ValueError(f"physics {name} must contain whole steps")
    conservative_upper_bounds = {
        "maximum_vertical_drop_m": 0.050,
        "maximum_table_penetration_m": 0.005,
        "maximum_surface_gap_m": 0.010,
        "maximum_pickup_xy_drift_m": 0.050,
        "maximum_tail_displacement_m": 0.005,
        "maximum_final_linear_speed_m_s": 0.050,
        "maximum_final_angular_speed_rad_s": 0.500,
        "maximum_upright_axis_tilt_rad": math.radians(10.0),
        "maximum_fixed_translation_drift_m": 0.00001,
        "maximum_fixed_rotation_drift_rad": 0.0001,
    }
    for name, upper_bound in conservative_upper_bounds.items():
        if getattr(physics, name) > upper_bound:
            raise ValueError(f"physics {name} exceeds the v1 safety bound")
    if (
        physics.maximum_tail_displacement_m
        > physics.maximum_pickup_xy_drift_m
    ):
        raise ValueError("tail displacement cannot exceed pickup drift")


def load_connector_tabletop_scene(
    config_path: str | Path = DEFAULT_TABLETOP_CONFIG_PATH,
) -> ConnectorTabletopScene:
    """Load and fail-closed validate the exact tabletop-v1 YAML."""

    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as stream:
        document = _mapping(yaml.safe_load(stream), "tabletop scene")
    _exact_keys(
        document,
        (
            "schema_version",
            "world",
            "table",
            "fixed_endpoint",
            "loose_endpoint",
            "physics",
            "render",
        ),
        "tabletop scene",
    )
    if document["schema_version"] != TABLETOP_SCHEMA_VERSION:
        raise ValueError(
            "unsupported tabletop scene schema: "
            f"{document['schema_version']!r}"
        )
    config = ConnectorTabletopScene(
        schema_version=TABLETOP_SCHEMA_VERSION,
        world=_load_world(document["world"]),
        table=_load_table(document["table"]),
        fixed_endpoint=_load_fixed(document["fixed_endpoint"]),
        loose_endpoint=_load_loose(document["loose_endpoint"]),
        physics=_load_physics(document["physics"]),
        render=_load_render(document["render"]),
    )
    _validate_geometry(config)
    config.as_dict()
    return config


def author_isaac_tabletop_scene(
    stage: Any,
    config: ConnectorTabletopScene,
    connector_asset_path: str | Path,
    *,
    add_reference_to_stage: Any,
    Gf: Any,
    Sdf: Any,
    UsdGeom: Any,
    UsdPhysics: Any,
    UsdShade: Any,
    physics_utils: Any,
) -> dict[str, Any]:
    """Author tabletop v1 after the caller has started Isaac Sim.

    All object transforms are authored before physics starts.  The returned
    paths are immutable handles for the standalone physical smoke test.
    """

    asset = Path(connector_asset_path).expanduser().resolve()
    if not asset.is_file():
        raise FileNotFoundError(asset)
    if config.schema_version != TABLETOP_SCHEMA_VERSION:
        raise ValueError("tabletop scene configuration is incompatible")

    if stage.GetPrimAtPath(config.world.root_prim_path).IsValid():
        raise RuntimeError("tabletop root already exists on the stage")
    root = UsdGeom.Xform.Define(stage, config.world.root_prim_path)
    if not root.GetPrim().IsValid():
        raise RuntimeError("could not author tabletop root")

    def static_cube(
        path: str,
        center: tuple[float, float, float],
        size: tuple[float, float, float],
        color: tuple[float, float, float],
    ) -> Any:
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
            raise RuntimeError("tabletop static collider became a rigid body")

    add_reference_to_stage(
        str(asset), config.world.connector_reference_prim_path
    )
    nested_scene_path = (
        config.world.connector_reference_prim_path + "/PhysicsScene"
    )
    nested_scene_removed = False
    if stage.GetPrimAtPath(nested_scene_path).IsValid():
        stage.RemovePrim(nested_scene_path)
        nested_scene_removed = True

    def set_single_translation(path: str, value: tuple[float, ...]) -> None:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"connector asset prim is missing: {path}")
        xformable = UsdGeom.Xformable(prim)
        operations = xformable.GetOrderedXformOps()
        if not operations:
            operation = xformable.AddTranslateOp()
        elif (
            len(operations) == 1
            and operations[0].GetOpType()
            == UsdGeom.XformOp.TypeTranslate
        ):
            operation = operations[0]
        else:
            raise RuntimeError(
                f"connector prim has an incompatible xform stack: {path}"
            )
        operation.Set(Gf.Vec3d(*value))

    set_single_translation(
        config.fixed_endpoint.receptacle_prim_path,
        config.fixed_endpoint.receptacle_origin_m,
    )
    set_single_translation(
        config.loose_endpoint.plug_prim_path,
        config.loose_endpoint.initial_center_m,
    )

    receptacle = stage.GetPrimAtPath(
        config.fixed_endpoint.receptacle_prim_path
    )
    body = stage.GetPrimAtPath(config.loose_endpoint.body_prim_path)
    nut = stage.GetPrimAtPath(config.loose_endpoint.nut_prim_path)
    if receptacle.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError("fixed receptacle unexpectedly became dynamic")
    if not body.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError("loose plug body is not a rigid body")
    if not nut.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError("loose coupling nut is not a rigid body")

    result = {
        "body_prim_path": config.loose_endpoint.body_prim_path,
        "connector_reference_prim_path": (
            config.world.connector_reference_prim_path
        ),
        "fixture_prim_path": config.fixed_endpoint.fixture_prim_path,
        "nested_physics_scene_removed": nested_scene_removed,
        "nut_prim_path": config.loose_endpoint.nut_prim_path,
        "plug_prim_path": config.loose_endpoint.plug_prim_path,
        "receptacle_prim_path": (
            config.fixed_endpoint.receptacle_prim_path
        ),
        "table_prim_path": config.table.prim_path,
    }
    json.dumps(result, allow_nan=False, sort_keys=True)
    return result


__all__ = [
    "DEFAULT_TABLETOP_CONFIG_PATH",
    "TABLETOP_SCHEMA_VERSION",
    "ConnectorTabletopScene",
    "FixedEndpoint",
    "LooseEndpoint",
    "TabletopPhysics",
    "TabletopRender",
    "TabletopSurface",
    "TabletopWorld",
    "author_isaac_tabletop_scene",
    "load_connector_tabletop_scene",
]
