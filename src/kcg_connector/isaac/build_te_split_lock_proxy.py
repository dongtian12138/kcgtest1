#!/usr/bin/env python3
"""Build and statically audit the TE split-lock contact-only development asset.

This is deliberately not an assembly runner.  It authors the three connector
parts plus one visible, finite-mass fixture that is supported by a finite
static table through frictional contact.  Nut/body bearing, thread and fixture
capture are collision-only.  No joint, drive, fixture constraint, pose
controller, or Isaac application is created here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


ASSET_NAME = "TE_J35_SPLIT_LOCK_PROXY_V1.usda"
AUDIT_NAME = "STATIC_AUDIT.json"
MODEL_PATH = "/World/TE_J35SplitLockProxy"
COMPONENT_PATHS = {
    "Receptacle": f"{MODEL_PATH}/Receptacle",
    "PlugBody": f"{MODEL_PATH}/PlugBody",
    "CouplingNut": f"{MODEL_PATH}/CouplingNut",
    "SupportFixture": f"{MODEL_PATH}/SupportFixture",
}
TABLE_PATH = f"{MODEL_PATH}/Table"
OFFICIAL_VISUAL_COMPONENTS = ("Receptacle", "PlugBody", "CouplingNut")
ANNULAR_FACE_COUNTS = [4, 4, 4, 4, 4, 4]
ANNULAR_FACE_INDICES = [
    3, 2, 1, 0,
    4, 5, 6, 7,
    0, 1, 5, 4,
    1, 2, 6, 5,
    2, 3, 7, 6,
    3, 0, 4, 7,
]


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=repository / "src/kcg_connector/config/te_split_lock_proxy_v1.yaml",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_config(path: Path) -> Mapping[str, Any]:
    import yaml

    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise TypeError("split-lock proxy config must be a mapping")
    identity = document.get("identity")
    if not isinstance(identity, Mapping):
        raise TypeError("identity must be a mapping")
    if identity.get("plug") != "D38999/26FJ35PN":
        raise ValueError("config does not identify the TE J35 plug")
    if identity.get("receptacle") != "D38999/20FJ35SN":
        raise ValueError("config does not identify the TE J35 receptacle")
    if identity.get("hardware_authorized") is not False:
        raise PermissionError("hardware_authorized must remain false")
    if identity.get("model_boundary") != (
        "DEVELOPMENT_CONTACT_BEARING_AND_SMALL_ANGLE_THREAD_WINDOW"
    ):
        raise ValueError("model boundary changed")
    return document


def _finite_positive(value: Any, field: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{field} must be finite and positive")
    return result


def _interval(value: Any, field: str) -> tuple[float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise TypeError(f"{field} must contain two numbers")
    low, high = (float(item) for item in value)
    if not all(math.isfinite(item) for item in (low, high)) or not low < high:
        raise ValueError(f"{field} must be a finite increasing interval")
    return low, high


def _relative_asset_path(repository: Path, configured: str, output: Path) -> str:
    source = (repository / configured).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    return os.path.relpath(source, output.parent.resolve())


def _annular_wedge_points(
    inner_radius: float,
    outer_radius: float,
    z_low: float,
    z_high: float,
    theta_low_deg: float,
    theta_high_deg: float,
) -> list[tuple[float, float, float]]:
    points: list[tuple[float, float, float]] = []
    for z_value in (z_low, z_high):
        for radius, theta_deg in (
            (inner_radius, theta_low_deg),
            (outer_radius, theta_low_deg),
            (outer_radius, theta_high_deg),
            (inner_radius, theta_high_deg),
        ):
            theta = math.radians(theta_deg)
            points.append(
                (radius * math.cos(theta), radius * math.sin(theta), z_value)
            )
    return points


def _extent(points: Sequence[Sequence[float]], Gf: Any) -> list[Any]:
    return [
        Gf.Vec3f(*(min(point[axis] for point in points) for axis in range(3))),
        Gf.Vec3f(*(max(point[axis] for point in points) for axis in range(3))),
    ]


def _apply_unknown_schema(prim: Any, schema_name: str, Sdf: Any) -> None:
    schemas = list(prim.GetAppliedSchemas())
    metadata = prim.GetMetadata("apiSchemas")
    if metadata is not None:
        schemas.extend(str(item) for item in metadata.GetAddedOrExplicitItems())
    if schema_name not in schemas:
        schemas.append(schema_name)
    prim.SetMetadata("apiSchemas", Sdf.TokenListOp.CreateExplicit(list(dict.fromkeys(schemas))))


class _Authorer:
    def __init__(
        self,
        *,
        stage: Any,
        document: Mapping[str, Any],
        repository: Path,
        output: Path,
        Gf: Any,
        Sdf: Any,
        UsdGeom: Any,
        UsdPhysics: Any,
        UsdShade: Any,
    ) -> None:
        self.stage = stage
        self.document = document
        self.repository = repository
        self.output = output
        self.Gf = Gf
        self.Sdf = Sdf
        self.UsdGeom = UsdGeom
        self.UsdPhysics = UsdPhysics
        self.UsdShade = UsdShade
        self.collisions_by_owner = {name: 0 for name in (*COMPONENT_PATHS, "Table")}
        self.collisions_by_role: dict[str, int] = {}
        self.materials: dict[str, Any] = {}

    def _custom(self, prim: Any, name: str, type_name: Any, value: Any) -> None:
        prim.CreateAttribute(f"kcg:{name}", type_name, custom=True).Set(value)

    def _material(self, role: str) -> Any:
        if role in self.materials:
            return self.materials[role]
        row = self.document["materials"][role]
        path = f"{MODEL_PATH}/Materials/{role}"
        material = self.UsdShade.Material.Define(self.stage, path)
        physics = self.UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics.CreateStaticFrictionAttr(float(row["static_friction"]))
        physics.CreateDynamicFrictionAttr(float(row["dynamic_friction"]))
        physics.CreateRestitutionAttr(float(row["restitution"]))
        self._custom(material.GetPrim(), "materialRole", self.Sdf.ValueTypeNames.String, role)
        self.materials[role] = material
        return material

    def _mark_collision(self, prim: Any, *, owner: str, role: str, material: str) -> None:
        collision = self.UsdPhysics.CollisionAPI.Apply(prim)
        collision.CreateCollisionEnabledAttr(True)
        offsets = self.document["contact_offsets"]
        _apply_unknown_schema(prim, "PhysxCollisionAPI", self.Sdf)
        prim.CreateAttribute(
            "physxCollision:contactOffset", self.Sdf.ValueTypeNames.Float, custom=False
        ).Set(float(offsets["contact_offset_m"]))
        prim.CreateAttribute(
            "physxCollision:restOffset", self.Sdf.ValueTypeNames.Float, custom=False
        ).Set(float(offsets["rest_offset_m"]))
        self.UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            self._material(material), materialPurpose="physics"
        )
        self._custom(prim, "owner", self.Sdf.ValueTypeNames.String, owner)
        self._custom(prim, "collisionRole", self.Sdf.ValueTypeNames.String, role)
        self.collisions_by_owner[owner] += 1
        self.collisions_by_role[role] = self.collisions_by_role.get(role, 0) + 1

    def _cube(
        self,
        path: str,
        *,
        owner: str,
        role: str,
        material: str,
        center: Sequence[float],
        dimensions: Sequence[float],
        rotate_z_deg: float = 0.0,
        display_color: Sequence[float] | None = None,
    ) -> None:
        cube = self.UsdGeom.Cube.Define(self.stage, path)
        cube.CreateSizeAttr(1.0)
        xform = self.UsdGeom.Xformable(cube.GetPrim())
        xform.AddTranslateOp().Set(self.Gf.Vec3d(*center))
        if rotate_z_deg:
            xform.AddRotateZOp().Set(float(rotate_z_deg))
        xform.AddScaleOp().Set(self.Gf.Vec3f(*dimensions))
        if display_color is not None:
            cube.CreateDisplayColorAttr([self.Gf.Vec3f(*display_color)])
            self._custom(
                cube.GetPrim(),
                "visiblePhysicalProxy",
                self.Sdf.ValueTypeNames.Bool,
                True,
            )
        self._mark_collision(cube.GetPrim(), owner=owner, role=role, material=material)

    def _cylinder(
        self,
        path: str,
        *,
        owner: str,
        role: str,
        material: str,
        center_z: float,
        radius: float,
        height: float,
    ) -> None:
        cylinder = self.UsdGeom.Cylinder.Define(self.stage, path)
        cylinder.CreateAxisAttr(self.UsdGeom.Tokens.z)
        cylinder.CreateRadiusAttr(radius)
        cylinder.CreateHeightAttr(height)
        self.UsdGeom.Xformable(cylinder.GetPrim()).AddTranslateOp().Set(
            self.Gf.Vec3d(0.0, 0.0, center_z)
        )
        self._mark_collision(
            cylinder.GetPrim(), owner=owner, role=role, material=material
        )

    def _sphere(
        self,
        path: str,
        *,
        owner: str,
        role: str,
        material: str,
        center: Sequence[float],
        radius: float,
    ) -> None:
        sphere = self.UsdGeom.Sphere.Define(self.stage, path)
        sphere.CreateRadiusAttr(radius)
        sphere.CreateExtentAttr(
            [
                self.Gf.Vec3f(-radius, -radius, -radius),
                self.Gf.Vec3f(radius, radius, radius),
            ]
        )
        self.UsdGeom.Xformable(sphere.GetPrim()).AddTranslateOp().Set(
            self.Gf.Vec3d(*center)
        )
        self._mark_collision(
            sphere.GetPrim(), owner=owner, role=role, material=material
        )

    def _capsule(
        self,
        path: str,
        *,
        owner: str,
        role: str,
        material: str,
        first: Sequence[float],
        second: Sequence[float],
        radius: float,
    ) -> None:
        direction = tuple(float(second[index]) - float(first[index]) for index in range(3))
        height = math.sqrt(sum(value * value for value in direction))
        if height <= 0.0:
            raise ValueError("thread capsule chord must be nonzero")
        center = tuple(
            0.5 * (float(first[index]) + float(second[index])) for index in range(3)
        )
        capsule = self.UsdGeom.Capsule.Define(self.stage, path)
        capsule.CreateAxisAttr(self.UsdGeom.Tokens.x)
        capsule.CreateRadiusAttr(radius)
        capsule.CreateHeightAttr(height)
        half_extent = 0.5 * height + radius
        capsule.CreateExtentAttr(
            [
                self.Gf.Vec3f(-half_extent, -radius, -radius),
                self.Gf.Vec3f(half_extent, radius, radius),
            ]
        )
        xform = self.UsdGeom.Xformable(capsule.GetPrim())
        xform.AddTranslateOp().Set(self.Gf.Vec3d(*center))
        vx, vy, vz = (value / height for value in direction)
        quaternion_real = math.sqrt(max(0.0, 0.5 * (1.0 + vx)))
        if quaternion_real <= 1.0e-12:
            quaternion = self.Gf.Quatf(0.0, self.Gf.Vec3f(0.0, 0.0, 1.0))
        else:
            quaternion = self.Gf.Quatf(
                quaternion_real,
                self.Gf.Vec3f(
                    0.0,
                    -vz / (2.0 * quaternion_real),
                    vy / (2.0 * quaternion_real),
                ),
            )
        xform.AddOrientOp().Set(quaternion)
        self._mark_collision(
            capsule.GetPrim(), owner=owner, role=role, material=material
        )

    def _annular_segments(
        self,
        parent: str,
        *,
        owner: str,
        role: str,
        material: str,
        radial_interval: Sequence[float],
        z_interval: Sequence[float],
        segment_count: int,
    ) -> None:
        inner, outer = _interval(radial_interval, f"{role}.radial_interval")
        z_low, z_high = _interval(z_interval, f"{role}.z_interval")
        if segment_count < 12:
            raise ValueError(f"{role}.segment_count must be at least 12")
        self.UsdGeom.Xform.Define(self.stage, parent)
        step = 360.0 / segment_count
        for index in range(segment_count):
            points = _annular_wedge_points(
                inner,
                outer,
                z_low,
                z_high,
                index * step,
                (index + 1) * step,
            )
            mesh = self.UsdGeom.Mesh.Define(self.stage, f"{parent}/Seg_{index:03d}")
            mesh.CreatePointsAttr([self.Gf.Vec3f(*point) for point in points])
            mesh.CreateFaceVertexCountsAttr(ANNULAR_FACE_COUNTS)
            mesh.CreateFaceVertexIndicesAttr(ANNULAR_FACE_INDICES)
            mesh.CreateSubdivisionSchemeAttr(self.UsdGeom.Tokens.none)
            mesh.CreateExtentAttr(_extent(points, self.Gf))
            mesh_collision = self.UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
            mesh_collision.CreateApproximationAttr(self.UsdPhysics.Tokens.convexHull)
            self._mark_collision(
                mesh.GetPrim(), owner=owner, role=role, material=material
            )

    def _rigid_component(self, name: str, translation: Sequence[float], mass: float) -> Any:
        xform = self.UsdGeom.Xform.Define(self.stage, COMPONENT_PATHS[name])
        xform.AddTranslateOp().Set(self.Gf.Vec3d(*translation))
        rigid = self.UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
        rigid.CreateRigidBodyEnabledAttr(True)
        rigid.CreateKinematicEnabledAttr(False)
        self.UsdPhysics.MassAPI.Apply(xform.GetPrim()).CreateMassAttr(mass)
        self._custom(xform.GetPrim(), "componentRole", self.Sdf.ValueTypeNames.String, name)
        return xform

    def _visual_reference(self, component: str, source_key: str, rotate_plug: bool) -> None:
        path = f"{COMPONENT_PATHS[component]}/Visual"
        visual = self.UsdGeom.Xform.Define(self.stage, path)
        configured = str(self.document["source_visual_assets"][source_key])
        visual.GetPrim().GetReferences().AddReference(
            _relative_asset_path(self.repository, configured, self.output)
        )
        if rotate_plug:
            rotation = self.document["component_frames"][
                "supplier_plug_visual_rotation_xyz_deg"
            ]
            visual.AddRotateXYZOp().Set(self.Gf.Vec3f(*rotation))

    def _author_receptacle(self) -> None:
        keying = self.document["keying_and_mating_collision"]
        component = COMPONENT_PATHS["Receptacle"]
        collision_root = f"{component}/Collision"
        self.UsdGeom.Scope.Define(self.stage, collision_root)
        segments = int(keying["receptacle_wall_segment_count"])
        bore = float(keying["receptacle_bore_radius_m"])
        outer = float(keying["receptacle_wall_outer_radius_m"])
        center_radius = 0.5 * (bore + outer)
        radial_thickness = outer - bore
        step = 360.0 / segments
        tangent = 0.92 * 2.0 * math.pi * center_radius / segments
        key_angles = [(-float(value)) % 360.0 for value in keying["key_angles_deg"]]
        widths = [float(keying["main_keyway_width_m"])] + [
            float(keying["minor_keyway_width_m"])
        ] * 4
        extra = float(keying["key_clearance_extra_m"])
        length = float(keying["engagement_length_m"])
        for index in range(segments):
            angle = index * step
            blocked = any(
                abs((angle - key_angle + 180.0) % 360.0 - 180.0)
                <= math.degrees(math.atan2(0.5 * width + extra, bore)) + 0.5 * step
                for key_angle, width in zip(key_angles, widths)
            )
            if blocked:
                continue
            theta = math.radians(angle)
            self._cube(
                f"{collision_root}/KeywayWall_{index:03d}",
                owner="Receptacle",
                role="receptacle_keyway_wall",
                material="shell_and_key",
                center=(
                    center_radius * math.cos(theta),
                    center_radius * math.sin(theta),
                    -0.5 * length,
                ),
                dimensions=(radial_thickness, tangent, length),
                rotate_z_deg=angle,
            )
        stop_thickness = float(keying["contact_face_stop_thickness_m"])
        self._cylinder(
            f"{collision_root}/ContactFaceStop",
            owner="Receptacle",
            role="receptacle_metal_stop",
            material="shell_and_key",
            center_z=-length - 0.5 * stop_thickness,
            radius=float(keying["contact_face_stop_radius_m"]),
            height=stop_thickness,
        )
        capture = self.document["support_fixture"]["receptacle_capture"]
        pad_size = float(capture["flange_pad_size_m"])
        pad_offset = float(capture["flange_pad_center_offset_m"])
        flange_thickness = float(capture["flange_thickness_m"])
        for index, (x_sign, y_sign) in enumerate(
            ((-1, -1), (-1, 1), (1, -1), (1, 1))
        ):
            self._cube(
                f"{collision_root}/FixtureInterface/FlangePad_{index}",
                owner="Receptacle",
                role="receptacle_fixture_flange_pad",
                material="support_fixture_contact",
                center=(
                    x_sign * pad_offset,
                    y_sign * pad_offset,
                    -0.5 * flange_thickness,
                ),
                dimensions=(pad_size, pad_size, flange_thickness),
                display_color=capture["receptacle_collision_color_rgb"],
            )

    def _author_body(self) -> None:
        keying = self.document["keying_and_mating_collision"]
        bearing = self.document["contact_bearing"]
        component = COMPONENT_PATHS["PlugBody"]
        collision_root = f"{component}/Collision"
        self.UsdGeom.Scope.Define(self.stage, collision_root)
        length = float(keying["engagement_length_m"])
        self._cylinder(
            f"{collision_root}/GuideShell",
            owner="PlugBody",
            role="plug_guide_shell",
            material="shell_and_key",
            center_z=0.5 * length,
            radius=float(keying["plug_shell_radius_m"]),
            height=length,
        )
        stop_thickness = float(keying["contact_face_stop_thickness_m"])
        self._cylinder(
            f"{collision_root}/ContactFaceStop",
            owner="PlugBody",
            role="plug_metal_stop",
            material="shell_and_key",
            center_z=0.5 * stop_thickness,
            radius=float(keying["contact_face_stop_radius_m"]) - 0.0001,
            height=stop_thickness,
        )
        shell = float(keying["plug_shell_radius_m"])
        key_outer = float(keying["plug_key_outer_radius_m"])
        center_radius = 0.5 * (shell + key_outer)
        widths = [float(keying["main_key_width_m"])] + [
            float(keying["minor_key_width_m"])
        ] * 4
        angles = [(-float(value)) % 360.0 for value in keying["key_angles_deg"]]
        for index, (angle, width) in enumerate(zip(angles, widths)):
            theta = math.radians(angle)
            self._cube(
                f"{collision_root}/Key_{index}",
                owner="PlugBody",
                role="plug_polarizing_key",
                material="shell_and_key",
                center=(
                    center_radius * math.cos(theta),
                    center_radius * math.sin(theta),
                    0.5 * length,
                ),
                dimensions=(key_outer - shell, width, length),
                rotate_z_deg=angle,
            )
        guide_segments = int(bearing["guide_segment_count"])
        shoulder_segments = int(bearing["shoulder_segment_count"])
        self._annular_segments(
            f"{collision_root}/ContactBearing/OuterJournal",
            owner="PlugBody",
            role="body_outer_journal",
            material="transparent_contact_bearing",
            radial_interval=bearing["body_outer_journal"]["radial_interval_m"],
            z_interval=bearing["body_outer_journal"]["local_z_interval_m"],
            segment_count=guide_segments,
        )
        for stop_name in ("front_axial_stop", "rear_axial_stop"):
            stop = bearing[stop_name]
            self._annular_segments(
                f"{collision_root}/ContactBearing/{stop_name}/BodyShoulder",
                owner="PlugBody",
                role=f"body_{stop_name}",
                material="transparent_contact_bearing",
                radial_interval=stop["radial_interval_m"],
                z_interval=stop["body_local_z_interval_m"],
                segment_count=shoulder_segments,
            )

    def _author_nut(self) -> None:
        outer = self.document["coupling_nut_outer_grip_collision"]
        bearing = self.document["contact_bearing"]
        component = COMPONENT_PATHS["CouplingNut"]
        collision_root = f"{component}/Collision"
        self.UsdGeom.Scope.Define(self.stage, collision_root)
        count = int(outer["segment_count"])
        radius = float(outer["segment_center_radius_m"])
        z_low, z_high = _interval(outer["local_z_interval_m"], "nut_outer.local_z")
        for index in range(count):
            angle = index * 360.0 / count
            theta = math.radians(angle)
            self._cube(
                f"{collision_root}/OuterGrip/Seg_{index:02d}",
                owner="CouplingNut",
                role="coupling_nut_outer_grip",
                material="coupling_nut_outer_grip",
                center=(radius * math.cos(theta), radius * math.sin(theta), 0.5 * (z_low + z_high)),
                dimensions=(
                    float(outer["radial_thickness_m"]),
                    float(outer["tangential_width_m"]),
                    z_high - z_low,
                ),
                rotate_z_deg=angle,
            )
        guide_segments = int(bearing["guide_segment_count"])
        shoulder_segments = int(bearing["shoulder_segment_count"])
        self._annular_segments(
            f"{collision_root}/ContactBearing/InnerGuide",
            owner="CouplingNut",
            role="nut_inner_guide",
            material="transparent_contact_bearing",
            radial_interval=bearing["nut_inner_guide"]["radial_interval_m"],
            z_interval=bearing["nut_inner_guide"]["local_z_interval_m"],
            segment_count=guide_segments,
        )
        for stop_name in ("front_axial_stop", "rear_axial_stop"):
            stop = bearing[stop_name]
            self._annular_segments(
                f"{collision_root}/ContactBearing/{stop_name}/NutShoulder",
                owner="CouplingNut",
                role=f"nut_{stop_name}",
                material="transparent_contact_bearing",
                radial_interval=stop["radial_interval_m"],
                z_interval=stop["nut_local_z_interval_m"],
                segment_count=shoulder_segments,
            )

    def _author_small_angle_thread(self) -> None:
        thread = self.document["small_angle_thread_contact"]
        if int(thread["start_count"]) != 3:
            raise ValueError("small-angle thread must remain three-start")
        phases = [float(value) for value in thread["start_phases_deg"]]
        if phases != [0.0, 120.0, 240.0]:
            raise ValueError("small-angle thread start phases changed")
        lead = _finite_positive(thread["lead_m_per_revolution"], "thread lead")
        pitch_radius = _finite_positive(thread["pitch_radius_m"], "thread pitch radius")
        radius = _finite_positive(thread["rail_capsule_radius_m"], "thread rail radius")
        follower_radius = _finite_positive(
            thread["follower_sphere_radius_m"], "thread follower radius"
        )
        entry_z = float(thread["fixed_rail_entry_z_m"])
        window_start = float(thread["rail_window_start_deg"])
        window_end = float(thread["rail_window_end_deg"])
        step = _finite_positive(thread["rail_segment_angle_deg"], "thread segment angle")
        segment_count_float = (window_end - window_start) / step
        segment_count = int(round(segment_count_float))
        if not math.isclose(segment_count_float, segment_count, abs_tol=1.0e-12):
            raise ValueError("thread window is not an integer number of segments")
        if segment_count <= 0:
            raise ValueError("thread window must contain segments")
        rail_root = f"{COMPONENT_PATHS['Receptacle']}/Collision/CouplingThread"
        for start_index, phase in enumerate(phases):
            for segment_index in range(segment_count):
                local0 = window_start + segment_index * step
                local1 = local0 + step
                theta0 = math.radians(phase + local0)
                theta1 = math.radians(phase + local1)
                z0 = entry_z + lead * (1.0 - local0 / 360.0)
                z1 = entry_z + lead * (1.0 - local1 / 360.0)
                first = (
                    pitch_radius * math.cos(theta0),
                    pitch_radius * math.sin(theta0),
                    z0,
                )
                second = (
                    pitch_radius * math.cos(theta1),
                    pitch_radius * math.sin(theta1),
                    z1,
                )
                self._capsule(
                    f"{rail_root}/Rail_{start_index}/Seg_{segment_index:03d}",
                    owner="Receptacle",
                    role="thread_rail_small_angle",
                    material="public_thread_contact_proxy",
                    first=first,
                    second=second,
                    radius=radius,
                )

        clock = float(thread["initial_follower_clock_deg"])
        surface_z = entry_z + lead * (1.0 - clock / 360.0)
        geometry_gap = _finite_positive(
            thread["initial_geometry_gap_m"], "thread initial geometry gap"
        )
        follower_world_z = surface_z + radius + follower_radius + geometry_gap
        nut_world_z = float(
            self.document["component_frames"]["coupling_nut_translation_m"][2]
        )
        follower_local_z = follower_world_z - nut_world_z
        follower_root = f"{COMPONENT_PATHS['CouplingNut']}/Collision/CouplingThread"
        for start_index, phase in enumerate(phases):
            theta = math.radians(phase + clock)
            self._sphere(
                f"{follower_root}/Follower_{start_index}",
                owner="CouplingNut",
                role="thread_follower_small_angle",
                material="public_thread_contact_proxy",
                center=(
                    pitch_radius * math.cos(theta),
                    pitch_radius * math.sin(theta),
                    follower_local_z,
                ),
                radius=follower_radius,
            )

    def _author_support_fixture(self) -> None:
        support = self.document["support_fixture"]
        table = support["table"]
        fixture = support["fixture"]
        capture = support["receptacle_capture"]
        self.UsdGeom.Xform.Define(self.stage, TABLE_PATH)
        self._cube(
            f"{TABLE_PATH}/Collision",
            owner="Table",
            role="finite_static_table",
            material="table_contact",
            center=table["center_m"],
            dimensions=table["size_m"],
            display_color=table["display_color_rgb"],
        )

        fixture_path = COMPONENT_PATHS["SupportFixture"]
        color = fixture["display_color_rgb"]
        self._cube(
            f"{fixture_path}/Collision/Base",
            owner="SupportFixture",
            role="support_fixture_base",
            material="support_fixture_contact",
            center=fixture["base_center_m"],
            dimensions=fixture["base_size_m"],
            display_color=color,
        )
        pad_size = float(capture["flange_pad_size_m"])
        pad_offset = float(capture["flange_pad_center_offset_m"])
        flange_half = 0.5 * float(capture["flange_side_m"])
        flange_thickness = float(capture["flange_thickness_m"])
        axial_gap = float(capture["axial_clearance_each_side_m"])
        lateral_gap = float(capture["lateral_clearance_each_side_m"])
        post_bottom = float(capture["support_post_bottom_z_m"])
        post_top = -flange_thickness - axial_gap
        for index, (x_sign, y_sign) in enumerate(
            ((-1, -1), (-1, 1), (1, -1), (1, 1))
        ):
            self._cube(
                f"{fixture_path}/Collision/LowerSupport/Post_{index}",
                owner="SupportFixture",
                role="support_fixture_lower_flange_support",
                material="support_fixture_contact",
                center=(
                    x_sign * pad_offset,
                    y_sign * pad_offset,
                    0.5 * (post_bottom + post_top),
                ),
                dimensions=(pad_size, pad_size, post_top - post_bottom),
                display_color=color,
            )

        wall_thickness = float(capture["side_wall_thickness_m"])
        wall_span = float(capture["side_wall_span_m"])
        wall_z_low, wall_z_high = _interval(
            capture["side_wall_z_interval_m"], "fixture side wall z"
        )
        wall_center = flange_half + lateral_gap + 0.5 * wall_thickness
        for axis, sign in (("X", -1), ("X", 1), ("Y", -1), ("Y", 1)):
            dimensions = (
                (wall_thickness, wall_span, wall_z_high - wall_z_low)
                if axis == "X"
                else (wall_span, wall_thickness, wall_z_high - wall_z_low)
            )
            center = (
                (sign * wall_center, 0.0, 0.5 * (wall_z_low + wall_z_high))
                if axis == "X"
                else (0.0, sign * wall_center, 0.5 * (wall_z_low + wall_z_high))
            )
            self._cube(
                f"{fixture_path}/Collision/LateralCapture/{axis}_{'Pos' if sign > 0 else 'Neg'}",
                owner="SupportFixture",
                role="support_fixture_lateral_flange_capture",
                material="support_fixture_contact",
                center=center,
                dimensions=dimensions,
                display_color=color,
            )

        retainer_size = float(capture["upper_retainer_xy_size_m"])
        retainer_offset = float(capture["upper_retainer_center_offset_m"])
        retainer_thickness = float(capture["upper_retainer_thickness_m"])
        retainer_bottom = axial_gap
        for index, (x_sign, y_sign) in enumerate(
            ((-1, -1), (-1, 1), (1, -1), (1, 1))
        ):
            self._cube(
                f"{fixture_path}/Collision/UpperRetention/Retainer_{index}",
                owner="SupportFixture",
                role="support_fixture_upper_flange_retention",
                material="support_fixture_contact",
                center=(
                    x_sign * retainer_offset,
                    y_sign * retainer_offset,
                    retainer_bottom + 0.5 * retainer_thickness,
                ),
                dimensions=(retainer_size, retainer_size, retainer_thickness),
                display_color=color,
            )

    def author(self) -> Mapping[str, Any]:
        from pxr import UsdGeom

        world = self.UsdGeom.Xform.Define(self.stage, "/World")
        self.stage.SetDefaultPrim(world.GetPrim())
        model = self.UsdGeom.Xform.Define(self.stage, MODEL_PATH)
        identity = self.document["identity"]
        self._custom(model.GetPrim(), "schemaVersion", self.Sdf.ValueTypeNames.String, str(self.document["schema_version"]))
        self._custom(model.GetPrim(), "modelBoundary", self.Sdf.ValueTypeNames.String, str(identity["model_boundary"]))
        self._custom(model.GetPrim(), "hardwareAuthorized", self.Sdf.ValueTypeNames.Bool, False)
        thread = self.document["small_angle_thread_contact"]
        self._custom(model.GetPrim(), "threadContactProxyPresent", self.Sdf.ValueTypeNames.Bool, True)
        self._custom(model.GetPrim(), "fullRevolutionThreadPresent", self.Sdf.ValueTypeNames.Bool, False)
        self._custom(model.GetPrim(), "threadLeadMPerRevolution", self.Sdf.ValueTypeNames.Double, float(thread["lead_m_per_revolution"]))
        self._custom(model.GetPrim(), "frozenProbeRotationDeg", self.Sdf.ValueTypeNames.Double, float(thread["frozen_probe_rotation_deg"]))

        frames = self.document["component_frames"]
        masses = self.document["mass_properties"]
        self._rigid_component("Receptacle", frames["receptacle_translation_m"], float(masses["receptacle_mass_kg"]))
        self._rigid_component("PlugBody", frames["plug_body_translation_m"], float(masses["plug_body_mass_kg"]))
        self._rigid_component("CouplingNut", frames["coupling_nut_translation_m"], float(masses["coupling_nut_mass_kg"]))
        self._rigid_component("SupportFixture", (0.0, 0.0, 0.0), float(masses["support_fixture_mass_kg"]))
        self._visual_reference("Receptacle", "receptacle", False)
        self._visual_reference("PlugBody", "plug_body", True)
        self._visual_reference("CouplingNut", "coupling_nut", True)
        self._author_receptacle()
        self._author_body()
        self._author_nut()
        self._author_small_angle_thread()
        self._author_support_fixture()
        return {
            "collisions_by_owner": dict(self.collisions_by_owner),
            "collisions_by_role": dict(sorted(self.collisions_by_role.items())),
        }


def _analytical_clearances(document: Mapping[str, Any]) -> Mapping[str, float]:
    bearing = document["contact_bearing"]
    offsets = document["contact_offsets"]
    keying = document["keying_and_mating_collision"]
    guide_segments = int(bearing["guide_segment_count"])
    body_outer = _interval(
        bearing["body_outer_journal"]["radial_interval_m"], "body journal radial"
    )[1]
    nut_inner = _interval(
        bearing["nut_inner_guide"]["radial_interval_m"], "nut guide radial"
    )[0]
    guide_clearance = nut_inner * math.cos(math.pi / guide_segments) - body_outer
    front = bearing["front_axial_stop"]
    rear = bearing["rear_axial_stop"]
    front_clearance = _interval(front["nut_local_z_interval_m"], "front nut z")[0] - _interval(front["body_local_z_interval_m"], "front body z")[1]
    rear_clearance = _interval(rear["body_local_z_interval_m"], "rear body z")[0] - _interval(rear["nut_local_z_interval_m"], "rear nut z")[1]
    contact_offset = float(offsets["contact_offset_m"])
    shell_clearance = float(keying["receptacle_bore_radius_m"]) - float(keying["plug_shell_radius_m"])
    key_outer_clearance = float(keying["receptacle_keyway_outer_radius_m"]) - float(keying["plug_key_outer_radius_m"])
    main_width_clearance = float(keying["main_keyway_width_m"]) - float(keying["main_key_width_m"])
    minor_width_clearance = float(keying["minor_keyway_width_m"]) - float(keying["minor_key_width_m"])
    frames = document["component_frames"]
    body_world_z = float(frames["plug_body_translation_m"][2])
    nut_world_z = float(frames["coupling_nut_translation_m"][2])
    receptacle_world_z = float(frames["receptacle_translation_m"][2])
    outer = document["coupling_nut_outer_grip_collision"]
    outer_inner_radius = (
        float(outer["segment_center_radius_m"])
        - 0.5 * float(outer["radial_thickness_m"])
    )
    body_bearing_outer = _interval(
        bearing["body_outer_journal"]["radial_interval_m"],
        "body journal radial",
    )[1]
    body_to_nut_outer_clearance = outer_inner_radius - max(
        body_bearing_outer, float(keying["plug_key_outer_radius_m"])
    )
    body_journal_z = _interval(
        bearing["body_outer_journal"]["local_z_interval_m"],
        "body journal z",
    )
    nut_guide_z = _interval(
        bearing["nut_inner_guide"]["local_z_interval_m"],
        "nut guide z",
    )
    body_front_z = _interval(
        front["body_local_z_interval_m"], "front body z"
    )
    nut_front_z = _interval(front["nut_local_z_interval_m"], "front nut z")
    body_rear_z = _interval(rear["body_local_z_interval_m"], "rear body z")
    nut_rear_z = _interval(rear["nut_local_z_interval_m"], "rear nut z")
    journal_to_shoulder_clearance = min(
        body_journal_z[0] - nut_front_z[1],
        nut_rear_z[0] - body_journal_z[1],
    )
    nut_guide_to_body_shoulder_clearance = min(
        nut_guide_z[0] - body_front_z[1],
        body_rear_z[0] - nut_guide_z[1],
    )
    nut_outer_z = _interval(outer["local_z_interval_m"], "nut outer z")
    receptacle_to_nut_axial_clearance = (
        nut_world_z + nut_outer_z[0] - receptacle_world_z
    )
    thread = document["small_angle_thread_contact"]
    lead = float(thread["lead_m_per_revolution"])
    pitch_radius = float(thread["pitch_radius_m"])
    rail_radius = float(thread["rail_capsule_radius_m"])
    follower_radius = float(thread["follower_sphere_radius_m"])
    thread_gap_command = float(thread["initial_geometry_gap_m"])
    thread_step_rad = math.radians(float(thread["rail_segment_angle_deg"]))
    chord_xy = 2.0 * pitch_radius * math.sin(0.5 * thread_step_rad)
    chord_z = lead * float(thread["rail_segment_angle_deg"]) / 360.0
    chord_length = math.hypot(chord_xy, chord_z)
    chord_vertical_component = chord_z / chord_length
    follower_center_vertical_separation = (
        rail_radius + follower_radius + thread_gap_command
    )
    follower_to_rail_axis_distance = follower_center_vertical_separation * math.sqrt(
        1.0 - chord_vertical_component * chord_vertical_component
    )
    thread_initial_clearance = (
        follower_to_rail_axis_distance - rail_radius - follower_radius
    )
    probe_rotation = float(thread["frozen_probe_rotation_deg"])
    follower_clock = float(thread["initial_follower_clock_deg"])
    expected_progress = lead * probe_rotation / 360.0
    start_margin = follower_clock - float(thread["rail_window_start_deg"])
    end_margin = float(thread["rail_window_end_deg"]) - (
        follower_clock + probe_rotation
    )
    surface_after_probe = float(thread["fixed_rail_entry_z_m"]) + lead * (
        1.0 - (follower_clock + probe_rotation) / 360.0
    )
    follower_bottom_after_probe = surface_after_probe + rail_radius + thread_gap_command
    rail_to_nut_guide_radial_clearance = (
        pitch_radius
        - rail_radius
        - _interval(
            bearing["nut_inner_guide"]["radial_interval_m"], "nut guide radial"
        )[1]
    )
    rail_to_nut_outer_radial_clearance = (
        outer_inner_radius - pitch_radius - rail_radius
    )
    thread_to_body_radial_clearance = (
        pitch_radius - max(rail_radius, follower_radius) - body_bearing_outer
    )
    thread_lead_angle_rad = math.atan2(lead, 2.0 * math.pi * pitch_radius)
    thread_material = document["materials"]["public_thread_contact_proxy"]
    thread_static_friction_angle_rad = math.atan(
        float(thread_material["static_friction"])
    )
    thread_dynamic_friction_angle_rad = math.atan(
        float(thread_material["dynamic_friction"])
    )
    receptacle_stop_top = (
        receptacle_world_z - float(keying["engagement_length_m"])
    )
    plug_stop_bottom = body_world_z
    metal_stop_axial_clearance = plug_stop_bottom - receptacle_stop_top
    support = document["support_fixture"]
    table = support["table"]
    fixture = support["fixture"]
    capture = support["receptacle_capture"]
    base_center_z = float(fixture["base_center_m"][2])
    base_height = float(fixture["base_size_m"][2])
    fixture_base_bottom = base_center_z - 0.5 * base_height
    fixture_base_top = base_center_z + 0.5 * base_height
    table_center_z = float(table["center_m"][2])
    table_height = float(table["size_m"][2])
    table_surface_from_box = table_center_z + 0.5 * table_height
    flange_side = float(capture["flange_side_m"])
    flange_thickness = float(capture["flange_thickness_m"])
    lateral_gap = float(capture["lateral_clearance_each_side_m"])
    axial_gap = float(capture["axial_clearance_each_side_m"])
    wall_thickness = float(capture["side_wall_thickness_m"])
    wall_inner_face = 0.5 * flange_side + lateral_gap
    wall_center = wall_inner_face + 0.5 * wall_thickness
    post_top = -flange_thickness - axial_gap
    retainer_bottom = axial_gap
    nut_world_bottom = nut_world_z + nut_outer_z[0]
    wall_z_high = _interval(
        capture["side_wall_z_interval_m"], "fixture side wall z"
    )[1]
    retainer_offset = float(capture["upper_retainer_center_offset_m"])
    retainer_half = 0.5 * float(capture["upper_retainer_xy_size_m"])
    retainer_inner_corner_radius = math.sqrt(2.0) * (
        retainer_offset - retainer_half
    )
    nut_outer_max_radius = (
        float(outer["segment_center_radius_m"])
        + 0.5 * float(outer["radial_thickness_m"])
    )
    result = {
        "bearing_radial_polygon_clearance_m": guide_clearance,
        "front_axial_shoulder_clearance_m": front_clearance,
        "rear_axial_shoulder_clearance_m": rear_clearance,
        "bearing_radial_clearance_after_two_contact_offsets_m": guide_clearance - 2.0 * contact_offset,
        "front_shoulder_clearance_after_two_contact_offsets_m": front_clearance - 2.0 * contact_offset,
        "rear_shoulder_clearance_after_two_contact_offsets_m": rear_clearance - 2.0 * contact_offset,
        "initial_key_engagement_depth_m": float(
            document["component_frames"]["initial_key_engagement_depth_m"]
        ),
        "plug_to_receptacle_metal_stop_axial_clearance_m": metal_stop_axial_clearance,
        "plug_shell_to_receptacle_bore_clearance_m": shell_clearance,
        "plug_key_to_keyway_outer_clearance_m": key_outer_clearance,
        "main_key_width_clearance_m": main_width_clearance,
        "minor_key_width_clearance_m": minor_width_clearance,
        "body_to_nut_outer_grip_radial_clearance_m": body_to_nut_outer_clearance,
        "body_journal_to_nut_shoulder_axial_clearance_m": journal_to_shoulder_clearance,
        "nut_guide_to_body_shoulder_axial_clearance_m": nut_guide_to_body_shoulder_clearance,
        "receptacle_to_nut_axial_clearance_m": receptacle_to_nut_axial_clearance,
        "body_and_nut_frame_z_coincidence_error_m": abs(body_world_z - nut_world_z),
        "thread_initial_follower_to_rail_clearance_m": thread_initial_clearance,
        "thread_initial_clearance_after_two_contact_offsets_m": thread_initial_clearance - 2.0 * contact_offset,
        "thread_rail_to_nut_inner_guide_radial_clearance_m": rail_to_nut_guide_radial_clearance,
        "thread_rail_to_nut_outer_grip_radial_clearance_m": rail_to_nut_outer_radial_clearance,
        "thread_rail_or_follower_to_body_radial_clearance_m": thread_to_body_radial_clearance,
        "thread_follower_bottom_z_after_frozen_probe_m": follower_bottom_after_probe,
        "thread_expected_progress_from_frozen_probe_m": expected_progress,
        "thread_expected_progress_config_error_m": abs(
            expected_progress - float(thread["expected_probe_axial_progress_m"])
        ),
        "thread_window_start_margin_deg": start_margin,
        "thread_window_end_margin_deg": end_margin,
        "thread_chord_length_m": chord_length,
        "thread_capsule_chord_vertical_component": chord_vertical_component,
        "thread_lead_angle_rad": thread_lead_angle_rad,
        "thread_static_friction_angle_rad": thread_static_friction_angle_rad,
        "thread_dynamic_friction_angle_rad": thread_dynamic_friction_angle_rad,
        "thread_static_friction_angle_margin_rad": (
            thread_static_friction_angle_rad - thread_lead_angle_rad
        ),
        "thread_dynamic_friction_angle_margin_rad": (
            thread_dynamic_friction_angle_rad - thread_lead_angle_rad
        ),
        "fixture_base_to_table_initial_gap_m": (
            fixture_base_bottom - float(table["surface_z_m"])
        ),
        "table_surface_config_to_box_error_m": abs(
            float(table["surface_z_m"]) - table_surface_from_box
        ),
        "fixture_lateral_capture_clearance_m": (
            wall_inner_face - 0.5 * flange_side
        ),
        "fixture_lower_flange_axial_clearance_m": (
            -flange_thickness - post_top
        ),
        "fixture_upper_flange_axial_clearance_m": retainer_bottom,
        "fixture_wall_to_nut_axial_clearance_m": (
            nut_world_bottom - wall_z_high
        ),
        "fixture_upper_retainer_to_nut_radial_clearance_m": (
            retainer_inner_corner_radius - nut_outer_max_radius
        ),
        "fixture_post_to_base_connection_error_m": abs(
            float(capture["support_post_bottom_z_m"]) - fixture_base_top
        ),
        "fixture_authored_wall_center_error_m": abs(
            wall_center
            - (
                0.5 * flange_side
                + lateral_gap
                + 0.5 * wall_thickness
            )
        ),
    }
    return result


def _audit_stage(
    asset: Path,
    document: Mapping[str, Any],
    authored: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    config: Path,
) -> Mapping[str, Any]:
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(asset))
    if stage is None:
        raise RuntimeError("could not reopen generated USD")
    rigid_paths: list[str] = []
    collision_paths: list[str] = []
    joint_like_paths: list[str] = []
    drive_schema_paths: list[str] = []
    kinematic_true_paths: list[str] = []
    constraint_relationship_paths: list[str] = []
    thread_collision_paths: list[str] = []
    fixture_collision_paths: list[str] = []
    table_collision_paths: list[str] = []
    receptacle_flange_collision_paths: list[str] = []
    visible_fixture_geometry_missing: list[str] = []
    material_binding_missing: list[str] = []
    physx_collision_schema_missing: list[str] = []
    masses: dict[str, float] = {}
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        type_name = prim.GetTypeName()
        schemas = list(prim.GetAppliedSchemas())
        schema_metadata = prim.GetMetadata("apiSchemas")
        if schema_metadata is not None:
            schemas.extend(
                str(item) for item in schema_metadata.GetAddedOrExplicitItems()
            )
        schemas = list(dict.fromkeys(schemas))
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_paths.append(path)
            if UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get() is True:
                kinematic_true_paths.append(path)
            mass = UsdPhysics.MassAPI(prim).GetMassAttr().Get()
            masses[path] = float(mass) if mass is not None else float("nan")
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collision_paths.append(path)
            if "PhysxCollisionAPI" not in schemas:
                physx_collision_schema_missing.append(path)
            role = prim.GetAttribute("kcg:collisionRole").Get()
            if role and "thread" in str(role).lower():
                thread_collision_paths.append(path)
            if path.startswith(COMPONENT_PATHS["SupportFixture"] + "/"):
                fixture_collision_paths.append(path)
                visible_flag = prim.GetAttribute("kcg:visiblePhysicalProxy").Get()
                display_color = prim.GetAttribute("primvars:displayColor").Get()
                if visible_flag is not True or not display_color:
                    visible_fixture_geometry_missing.append(path)
            if path.startswith(TABLE_PATH + "/"):
                table_collision_paths.append(path)
                visible_flag = prim.GetAttribute("kcg:visiblePhysicalProxy").Get()
                display_color = prim.GetAttribute("primvars:displayColor").Get()
                if visible_flag is not True or not display_color:
                    visible_fixture_geometry_missing.append(path)
            if role == "receptacle_fixture_flange_pad":
                receptacle_flange_collision_paths.append(path)
            binding = prim.GetRelationship("material:binding:physics")
            if not binding or not binding.GetTargets():
                material_binding_missing.append(path)
        if "Joint" in type_name or any(token in type_name for token in ("Revolute", "Prismatic", "Rack", "Pinion")):
            joint_like_paths.append(path)
        if any(schema.startswith("PhysicsDriveAPI") for schema in schemas):
            drive_schema_paths.append(path)
        for relationship in prim.GetRelationships():
            name = relationship.GetName()
            if name in ("physics:body0", "physics:body1") and relationship.GetTargets():
                constraint_relationship_paths.append(f"{path}.{name}")

    visual_reference_readback: dict[str, Mapping[str, Any]] = {}
    for component in OFFICIAL_VISUAL_COMPONENTS:
        visual = stage.GetPrimAtPath(f"{COMPONENT_PATHS[component]}/Visual")
        visual_reference_readback[component] = {
            "prim_present": bool(visual),
            "authored_reference_present": bool(
                visual and visual.HasAuthoredReferences()
            ),
            "resolved_child_count": len(visual.GetChildren()) if visual else 0,
        }

    clearances = _analytical_clearances(document)
    acceptance = document["static_acceptance"]
    thread = document["small_angle_thread_contact"]
    expected_rigid = sorted(COMPONENT_PATHS.values())
    expected_masses = {
        COMPONENT_PATHS["Receptacle"]: float(document["mass_properties"]["receptacle_mass_kg"]),
        COMPONENT_PATHS["PlugBody"]: float(document["mass_properties"]["plug_body_mass_kg"]),
        COMPONENT_PATHS["CouplingNut"]: float(document["mass_properties"]["coupling_nut_mass_kg"]),
        COMPONENT_PATHS["SupportFixture"]: float(document["mass_properties"]["support_fixture_mass_kg"]),
    }
    geometry_clearance_fields = [
        "bearing_radial_polygon_clearance_m",
        "front_axial_shoulder_clearance_m",
        "rear_axial_shoulder_clearance_m",
        "plug_to_receptacle_metal_stop_axial_clearance_m",
        "plug_shell_to_receptacle_bore_clearance_m",
        "plug_key_to_keyway_outer_clearance_m",
        "main_key_width_clearance_m",
        "minor_key_width_clearance_m",
        "body_to_nut_outer_grip_radial_clearance_m",
        "body_journal_to_nut_shoulder_axial_clearance_m",
        "nut_guide_to_body_shoulder_axial_clearance_m",
        "receptacle_to_nut_axial_clearance_m",
        "thread_initial_follower_to_rail_clearance_m",
        "thread_rail_to_nut_inner_guide_radial_clearance_m",
        "thread_rail_to_nut_outer_grip_radial_clearance_m",
        "thread_rail_or_follower_to_body_radial_clearance_m",
        "thread_follower_bottom_z_after_frozen_probe_m",
        "fixture_base_to_table_initial_gap_m",
        "fixture_lateral_capture_clearance_m",
        "fixture_lower_flange_axial_clearance_m",
        "fixture_upper_flange_axial_clearance_m",
        "fixture_wall_to_nut_axial_clearance_m",
        "fixture_upper_retainer_to_nut_radial_clearance_m",
    ]
    effective_clearance_fields = [
        "bearing_radial_clearance_after_two_contact_offsets_m",
        "front_shoulder_clearance_after_two_contact_offsets_m",
        "rear_shoulder_clearance_after_two_contact_offsets_m",
        "thread_initial_clearance_after_two_contact_offsets_m",
        "fixture_base_to_table_clearance_after_two_contact_offsets_m",
        "fixture_lateral_capture_clearance_after_two_contact_offsets_m",
        "fixture_lower_capture_clearance_after_two_contact_offsets_m",
        "fixture_upper_capture_clearance_after_two_contact_offsets_m",
    ]
    clearances = {
        **clearances,
        "fixture_base_to_table_clearance_after_two_contact_offsets_m": float(
            clearances["fixture_base_to_table_initial_gap_m"]
        )
        - 2.0 * float(document["contact_offsets"]["contact_offset_m"]),
        "fixture_lateral_capture_clearance_after_two_contact_offsets_m": float(
            clearances["fixture_lateral_capture_clearance_m"]
        )
        - 2.0 * float(document["contact_offsets"]["contact_offset_m"]),
        "fixture_lower_capture_clearance_after_two_contact_offsets_m": float(
            clearances["fixture_lower_flange_axial_clearance_m"]
        )
        - 2.0 * float(document["contact_offsets"]["contact_offset_m"]),
        "fixture_upper_capture_clearance_after_two_contact_offsets_m": float(
            clearances["fixture_upper_flange_axial_clearance_m"]
        )
        - 2.0 * float(document["contact_offsets"]["contact_offset_m"]),
    }
    minimum_geometry = min(float(clearances[field]) for field in geometry_clearance_fields)
    minimum_effective = min(float(clearances[field]) for field in effective_clearance_fields)
    mass_readback_abs_tolerance_kg = 1.0e-7
    mass_values_match = bool(
        set(masses) == set(expected_masses)
        and all(
            math.isclose(
                masses[path],
                expected_masses[path],
                rel_tol=0.0,
                abs_tol=mass_readback_abs_tolerance_kg,
            )
            for path in expected_masses
        )
    )
    gates = {
        "identity_hardware_authorized_false": document["identity"]["hardware_authorized"] is False,
        "exactly_expected_rigid_bodies": sorted(rigid_paths) == expected_rigid,
        "all_rigid_bodies_dynamic": not kinematic_true_paths,
        "mass_values_match_config": mass_values_match,
        "collision_count_matches_authoring": len(collision_paths) == sum(authored["collisions_by_owner"].values()),
        "each_component_has_collision": all(int(authored["collisions_by_owner"][name]) > 0 for name in COMPONENT_PATHS),
        "all_colliders_have_physics_material": not material_binding_missing,
        "all_colliders_have_physx_collision_schema": not physx_collision_schema_missing,
        "all_official_visual_references_resolved": all(
            row["prim_present"]
            and row["authored_reference_present"]
            and int(row["resolved_child_count"]) > 0
            for row in visual_reference_readback.values()
        ),
        "no_joint_or_constraint_prim": not joint_like_paths,
        "no_constraint_body_relationship": not constraint_relationship_paths,
        "no_drive_schema": not drive_schema_paths,
        "support_fixture_collision_count_matches_config": len(
            fixture_collision_paths
        )
        == int(acceptance["require_support_fixture_collision_count"]),
        "table_collision_count_matches_config": len(table_collision_paths)
        == int(acceptance["require_table_collision_count"]),
        "receptacle_flange_collision_count_matches_config": len(
            receptacle_flange_collision_paths
        )
        == int(acceptance["require_receptacle_flange_collision_count"]),
        "fixture_and_table_contact_geometry_is_visible": not visible_fixture_geometry_missing,
        "table_is_static_collision_without_rigid_body": bool(
            table_collision_paths
            and not stage.GetPrimAtPath(TABLE_PATH).HasAPI(UsdPhysics.RigidBodyAPI)
        ),
        "table_surface_config_matches_geometry": float(
            clearances["table_surface_config_to_box_error_m"]
        )
        <= 1.0e-12,
        "fixture_parts_form_one_rigid_body_without_connection_gap": max(
            float(clearances["fixture_post_to_base_connection_error_m"]),
            float(clearances["fixture_authored_wall_center_error_m"]),
        )
        <= 1.0e-12,
        "thread_collision_count_matches_frozen_window": len(thread_collision_paths)
        == int(acceptance["require_thread_collision_count"]),
        "thread_probe_relation_matches_config": float(
            clearances["thread_expected_progress_config_error_m"]
        )
        <= 1.0e-15,
        "thread_probe_remains_inside_authored_window": min(
            float(clearances["thread_window_start_margin_deg"]),
            float(clearances["thread_window_end_margin_deg"]),
        )
        >= float(thread["minimum_end_of_probe_window_margin_deg"]),
        "thread_static_self_locking_necessary_condition": float(
            clearances["thread_static_friction_angle_margin_rad"]
        )
        > 0.0,
        "thread_dynamic_friction_angle_exceeds_lead_angle": float(
            clearances["thread_dynamic_friction_angle_margin_rad"]
        )
        > 0.0,
        "minimum_geometry_clearance_met": minimum_geometry >= float(acceptance["minimum_geometry_clearance_m"]),
        "minimum_effective_clearance_met": minimum_effective >= float(acceptance["minimum_clearance_after_two_contact_offsets_m"]),
        "initial_analytical_penetration_absent": minimum_geometry > 0.0,
    }
    passed = all(gates.values())
    return {
        "schema_version": "kcg_te_split_lock_proxy_static_audit_v1",
        "status": "STATIC_PASS" if passed else "STATIC_FAIL",
        "passed": passed,
        "scope": "PURE_USD_STATIC_AUTHORING_AND_ANALYTICAL_CLEARANCE_NO_ISAAC_DYNAMICS",
        "physical_claims": {
            "three_independent_connector_parts_authored": all(
                path in rigid_paths
                for path in (
                    COMPONENT_PATHS["Receptacle"],
                    COMPONENT_PATHS["PlugBody"],
                    COMPONENT_PATHS["CouplingNut"],
                )
            ),
            "visible_finite_mass_contact_fixture_authored": bool(
                gates["support_fixture_collision_count_matches_config"]
                and gates["fixture_and_table_contact_geometry_is_visible"]
            ),
            "fixture_contact_support_dynamic_function_verified": False,
            "contact_bearing_dynamic_function_verified": False,
            "three_start_small_angle_thread_contact_geometry_authored": gates[
                "thread_collision_count_matches_frozen_window"
            ],
            "thread_or_locking_progress_verified": False,
            "robot_or_hand_drive_verified": False,
            "passive_release_hold_verified": False,
        },
        "inputs": {
            "config": str(config),
            "config_sha256": _sha256(config),
            "source_visual_sha256": dict(source_hashes),
        },
        "asset": {
            "path": str(asset),
            "sha256": _sha256(asset),
            "rigid_body_paths": rigid_paths,
            "mass_kg_by_path": masses,
            "mass_readback_abs_tolerance_kg": mass_readback_abs_tolerance_kg,
            "collision_count": len(collision_paths),
            "collisions_by_owner": authored["collisions_by_owner"],
            "collisions_by_role": authored["collisions_by_role"],
            "joint_like_paths": joint_like_paths,
            "drive_schema_paths": drive_schema_paths,
            "constraint_relationship_paths": constraint_relationship_paths,
            "kinematic_true_paths": kinematic_true_paths,
            "thread_collision_paths": thread_collision_paths,
            "fixture_collision_paths": fixture_collision_paths,
            "table_collision_paths": table_collision_paths,
            "receptacle_flange_collision_paths": receptacle_flange_collision_paths,
            "visible_fixture_geometry_missing_paths": visible_fixture_geometry_missing,
            "material_binding_missing_paths": material_binding_missing,
            "physx_collision_schema_missing_paths": physx_collision_schema_missing,
            "visual_reference_readback": visual_reference_readback,
        },
        "clearances": {
            **clearances,
            "minimum_geometry_clearance_m": minimum_geometry,
            "minimum_clearance_after_two_contact_offsets_m": minimum_effective,
            "method": "frozen_analytic_component_geometry_in_initial_authored_pose",
            "not_a_dynamic_contact_or_penetration_measurement": True,
        },
        "gates": gates,
        "forbidden_mechanisms": {
            "joint": False,
            "drive": False,
            "world_to_plug_constraint": False,
            "world_to_fixture_constraint": False,
            "receptacle_to_fixture_constraint": False,
            "rack_and_pinion": False,
            "direct_pose_write_runtime": False,
            "magnetic_attraction": False,
        },
        "hardware_authorized": False,
        "simulation_research_limits": dict(
            document["simulation_research_limits"]
        ),
        "simulation_app_started": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _arguments(argv)
    repository = Path(__file__).resolve().parents[3]
    config = arguments.config.expanduser().resolve()
    output_dir = arguments.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    document = _load_config(config)
    source_paths = {
        key: (repository / str(value)).resolve()
        for key, value in document["source_visual_assets"].items()
    }
    for source in source_paths.values():
        if not source.is_file():
            raise FileNotFoundError(source)
    source_hashes = {key: _sha256(path) for key, path in source_paths.items()}

    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    output_dir.mkdir(parents=True, exist_ok=False)
    asset = output_dir / ASSET_NAME
    stage = Usd.Stage.CreateNew(str(asset))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    authorer = _Authorer(
        stage=stage,
        document=document,
        repository=repository,
        output=asset,
        Gf=Gf,
        Sdf=Sdf,
        UsdGeom=UsdGeom,
        UsdPhysics=UsdPhysics,
        UsdShade=UsdShade,
    )
    authored = authorer.author()
    stage.GetRootLayer().Save()
    stage = None
    audit = _audit_stage(asset, document, authored, source_hashes, config)
    (output_dir / AUDIT_NAME).write_text(
        json.dumps(audit, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, allow_nan=False, sort_keys=True), flush=True)
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
