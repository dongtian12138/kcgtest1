#!/usr/bin/env python3
"""Build isolated J599/26FJ35PN and J599/20FJ35SN OpenUSD assets.

The public interface geometry and every contact coordinate come from the
machine-readable contract beside this source. Manufacturer-specific details
remain explicit visual assumptions. No existing repository asset is read or
modified by this generator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


MODEL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = MODEL_ROOT / "config" / "model_contract.json"
DEFAULT_CONTACTS = MODEL_ROOT / "data" / "contact_positions_25_35.csv"
DEFAULT_OUTPUT = MODEL_ROOT / "generated"
PAIR_ROOT = "/World/J599_25_35_N_Pair"
FIXED_PATH = PAIR_ROOT + "/FixedReceptacle_J599_20FJ35SN"
LOOSE_PATH = PAIR_ROOT + "/LoosePlug_J599_26FJ35PN"
BODY_PATH = LOOSE_PATH + "/Body"
NUT_PATH = LOOSE_PATH + "/CouplingNut"
NUT_JOINT_PATH = LOOSE_PATH + "/CouplingNutRevolute"
CONTACT_OFFSET_M = 1.0e-5
REST_OFFSET_M = 0.0


@dataclass(frozen=True)
class Contact:
    contact_id: int
    x_in: float
    y_in: float
    x_m: float
    y_m: float
    source_page: int


@dataclass(frozen=True)
class KeyFeature:
    index: int
    name: str
    angle_deg: float
    key_width_m: float
    keyway_width_m: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_contract(path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != "j599_25_35_standard_interface_contract_v1":
        raise ValueError("unexpected J599 contract schema")
    identity = document["identity"]
    expected = {
        "shell_size": 25,
        "shell_size_code": "J",
        "insert_arrangement": "25-35",
        "polarization": "N",
        "contact_count": 128,
        "contact_size": "22D",
    }
    for field, value in expected.items():
        if identity.get(field) != value:
            raise ValueError(f"identity.{field} must be {value!r}")
    if identity["plug"]["contact_style"] != "pin":
        raise ValueError("J599/26FJ35PN must be the pin plug")
    if identity["receptacle"]["contact_style"] != "socket":
        raise ValueError("J599/20FJ35SN must be the socket receptacle")
    authorization = document["authorization"]
    for forbidden_true in (
        "hardware_authorized",
        "hardware_exact_fidelity",
        "real_hardware_assembly_success_claimed",
    ):
        if authorization.get(forbidden_true) is not False:
            raise ValueError(f"authorization.{forbidden_true} must remain false")
    return document


def load_contacts(path: Path = DEFAULT_CONTACTS) -> tuple[Contact, ...]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    contacts: list[Contact] = []
    for row in rows:
        x_in = float(row["x_in"])
        y_in = float(row["y_in"])
        x_m = x_in * 0.0254
        y_m = y_in * 0.0254
        if abs(x_m * 1000.0 - float(row["x_mm"])) > 0.015:
            raise ValueError(f"contact {row['contact_id']} x conversion mismatch")
        if abs(y_m * 1000.0 - float(row["y_mm"])) > 0.015:
            raise ValueError(f"contact {row['contact_id']} y conversion mismatch")
        contacts.append(
            Contact(
                contact_id=int(row["contact_id"]),
                x_in=x_in,
                y_in=y_in,
                x_m=x_m,
                y_m=y_m,
                source_page=int(row["source_page"]),
            )
        )
    if [item.contact_id for item in contacts] != list(range(1, 129)):
        raise ValueError("contact IDs must be exactly 1 through 128")
    return tuple(contacts)


def key_features(contract: dict[str, Any]) -> tuple[KeyFeature, ...]:
    keying = contract["public_interface_geometry"]["keying_n"]
    angles = [keying["main_key_angle"], *keying["minor_key_angles"]]
    features = []
    for index, angle in enumerate(angles):
        main = index == 0
        key_width_mm = keying[
            "plug_main_key_width_basic_mm"
            if main
            else "plug_minor_key_width_basic_mm"
        ]
        keyway_width_mm = keying[
            "receptacle_main_keyway_width_basic_mm"
            if main
            else "receptacle_minor_keyway_width_basic_mm"
        ]
        if key_width_mm >= keyway_width_mm:
            raise ValueError("every nominal keyway must be wider than its key")
        features.append(
            KeyFeature(
                index=index,
                name="Main" if main else f"Minor{index}",
                angle_deg=float(angle),
                key_width_m=float(key_width_mm) * 1.0e-3,
                keyway_width_m=float(keyway_width_mm) * 1.0e-3,
            )
        )
    if len(features) != 5 or len({item.angle_deg for item in features}) != 5:
        raise ValueError("N polarization must contain five unique keys")
    return tuple(features)


def nominal_key_clearances_m(contract: dict[str, Any]) -> tuple[float, ...]:
    return tuple(
        feature.keyway_width_m - feature.key_width_m
        for feature in key_features(contract)
    )


def wrong_yaw_interferes(contract: dict[str, Any], yaw_deg: float) -> bool:
    geometry = contract["simulation_geometry"]
    radius = 0.5 * (
        float(geometry["plug_guide_radius_m"])
        + float(geometry["plug_key_outer_radius_m"])
    )
    tangential_shift = radius * abs(math.radians(float(yaw_deg)))
    return any(
        tangential_shift > 0.5 * clearance
        for clearance in nominal_key_clearances_m(contract)
    )


def _angles(count: int) -> tuple[float, ...]:
    return tuple(360.0 * index / count for index in range(count))


def _angular_distance(first: float, second: float) -> float:
    return abs((first - second + math.pi) % (2.0 * math.pi) - math.pi)


def _blocking_angles(
    features: Iterable[KeyFeature], bore_radius: float, segment_count: int
) -> tuple[float, ...]:
    step = 2.0 * math.pi / segment_count
    retained: list[float] = []
    for index in range(segment_count):
        angle = index * step
        omitted = False
        for feature in features:
            half_opening = math.atan2(0.5 * feature.keyway_width_m, bore_radius)
            if _angular_distance(angle, math.radians(feature.angle_deg)) <= (
                half_opening + 0.55 * step
            ):
                omitted = True
                break
        if not omitted:
            retained.append(math.degrees(angle))
    return tuple(retained)


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--contacts", type=Path, default=DEFAULT_CONTACTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def build_assets(
    contract_path: Path = DEFAULT_CONTRACT,
    contacts_path: Path = DEFAULT_CONTACTS,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    contract_path = contract_path.resolve()
    contacts_path = contacts_path.resolve()
    output_dir = output_dir.resolve()
    if MODEL_ROOT.resolve() not in output_dir.parents:
        raise ValueError("output directory must remain inside the isolated model root")
    contract = load_contract(contract_path)
    contacts = load_contacts(contacts_path)
    features = key_features(contract)
    wrong_yaw = contract["acceptance"]["static"]["wrong_yaw_deg_for_negative_case"]
    if not wrong_yaw_interferes(contract, wrong_yaw):
        raise ValueError("contract wrong-yaw case must analytically interfere")

    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    output_dir.mkdir(parents=True, exist_ok=True)
    visual_path = output_dir / "j599_25_35_pair_visual.usda"
    assembly_path = output_dir / "j599_25_35_pair_assembly.usda"
    binary_path = output_dir / "j599_25_35_pair_assembly.usdc"

    geometry = contract["simulation_geometry"]
    mass = contract["representative_mass_properties"]
    plug_guide_radius = float(geometry["plug_guide_radius_m"])
    plug_key_outer_radius = float(geometry["plug_key_outer_radius_m"])
    bore_radius = float(geometry["receptacle_bore_radius_m"])
    keyway_outer_radius = float(geometry["receptacle_keyway_outer_radius_m"])
    fixed_outer_radius = float(geometry["receptacle_shell_outer_radius_m"])
    key_length = float(geometry["plug_key_length_m"])
    initial_z = float(geometry["initial_plug_face_z_m"])
    collision_counts: dict[str, int] = {}

    colors = {
        "nickel": (0.49, 0.53, 0.58),
        "nickel_dark": (0.27, 0.30, 0.34),
        "insert": (0.12, 0.16, 0.17),
        "socket": (0.015, 0.018, 0.020),
        "gold": (0.88, 0.61, 0.12),
        "key": (0.78, 0.22, 0.08),
        "blue_band": (0.05, 0.20, 0.72),
        "red_band": (0.75, 0.04, 0.03),
        "hole": (0.035, 0.040, 0.045),
    }

    def make_stage(physics: bool) -> Any:
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())
        root = UsdGeom.Xform.Define(stage, PAIR_ROOT)
        metadata = {
            "schemaVersion": contract["schema_version"],
            "modelId": contract["model_id"],
            "partNumbers": "J599/26FJ35PN,J599/20FJ35SN",
            "manufacturer": "UNKNOWN",
            "contactArrangement": "25-35",
            "contactCount": 128,
            "contactSize": "22D",
            "polarization": "N",
            "keyAnglesDeg": "0,80,142,196,293",
            "threadStarts": 3,
            "threadLeadMmPerRevolution": 7.62,
            "representation": "assembly" if physics else "visual",
            "hardwareAuthorized": False,
            "hardwareExactFidelity": False,
            "realHardwareAssemblySuccessClaimed": False,
            "contactCollisionMode": "visual_only",
            "sourceContract": str(contract_path),
        }
        for key, value in metadata.items():
            root.GetPrim().SetCustomDataByKey(f"j599:{key}", value)
        return stage

    def author(stage: Any, physics: bool) -> dict[str, int]:
        counts = {
            "plug_pins": 0,
            "receptacle_sockets": 0,
            "keys": 0,
            "keyways": 0,
            "collision_prims": 0,
            "robot_or_hand_prims": 0,
        }

        def mark_collision(prim: Any, role: str) -> None:
            UsdPhysics.CollisionAPI.Apply(prim)
            # Author the PhysX schema properties directly so deterministic asset
            # generation does not depend on a running Kit extension registry.
            prim.CreateAttribute(
                "physxCollision:contactOffset",
                Sdf.ValueTypeNames.Float,
                custom=False,
            ).Set(CONTACT_OFFSET_M)
            prim.CreateAttribute(
                "physxCollision:restOffset",
                Sdf.ValueTypeNames.Float,
                custom=False,
            ).Set(REST_OFFSET_M)
            prim.SetCustomDataByKey("j599:collisionRole", role)
            collision_counts[role] = collision_counts.get(role, 0) + 1
            counts["collision_prims"] += 1

        def xform(path: str) -> Any:
            return UsdGeom.Xform.Define(stage, path)

        def cube(
            path: str,
            size: tuple[float, float, float],
            translation: tuple[float, float, float],
            color: tuple[float, float, float],
            *,
            rotation_z_deg: float = 0.0,
            collision_role: str | None = None,
            visible: bool = True,
        ) -> Any:
            shape = UsdGeom.Cube.Define(stage, path)
            shape.CreateSizeAttr(1.0)
            shape.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            transform = UsdGeom.Xformable(shape.GetPrim())
            transform.AddTranslateOp().Set(Gf.Vec3d(*translation))
            if abs(rotation_z_deg) > 1.0e-12:
                transform.AddRotateZOp().Set(rotation_z_deg)
            transform.AddScaleOp().Set(Gf.Vec3f(*size))
            if not visible:
                shape.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
            if physics and collision_role is not None:
                mark_collision(shape.GetPrim(), collision_role)
            return shape

        def cylinder(
            path: str,
            radius: float,
            height: float,
            translation: tuple[float, float, float],
            color: tuple[float, float, float],
            *,
            collision_role: str | None = None,
            visible: bool = True,
        ) -> Any:
            shape = UsdGeom.Cylinder.Define(stage, path)
            shape.CreateAxisAttr(UsdGeom.Tokens.z)
            shape.CreateRadiusAttr(radius)
            shape.CreateHeightAttr(height)
            shape.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            UsdGeom.Xformable(shape.GetPrim()).AddTranslateOp().Set(
                Gf.Vec3d(*translation)
            )
            if not visible:
                shape.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
            if physics and collision_role is not None:
                mark_collision(shape.GetPrim(), collision_role)
            return shape

        def ring_segments(
            parent: str,
            inner_radius: float,
            outer_radius: float,
            height: float,
            center_z: float,
            angles_deg: Iterable[float],
            reference_count: int,
            color: tuple[float, float, float],
            collision_role: str | None = None,
            visible: bool = True,
        ) -> None:
            xform(parent)
            center_radius = 0.5 * (inner_radius + outer_radius)
            radial_size = outer_radius - inner_radius
            tangential_size = 1.04 * 2.0 * math.pi * center_radius / reference_count
            for index, angle_deg in enumerate(angles_deg):
                angle = math.radians(angle_deg)
                cube(
                    f"{parent}/Segment_{index:03d}",
                    (radial_size, tangential_size, height),
                    (
                        center_radius * math.cos(angle),
                        center_radius * math.sin(angle),
                        center_z,
                    ),
                    color,
                    rotation_z_deg=angle_deg,
                    collision_role=collision_role,
                    visible=visible,
                )

        def band(
            parent: str,
            radius: float,
            z: float,
            color: tuple[float, float, float],
        ) -> None:
            ring_segments(
                parent,
                radius - 0.00025,
                radius,
                0.0012,
                z,
                _angles(72),
                72,
                color,
            )

        def helical_curves(parent: str, radius: float, z0: float, z1: float) -> None:
            xform(parent)
            thread = contract["public_interface_geometry"]["thread"]
            starts = int(thread["starts"])
            lead = float(thread["lead_mm_per_revolution"]) * 1.0e-3
            turns = abs(z1 - z0) / lead
            samples_per_start = 80
            for start in range(starts):
                curve = UsdGeom.BasisCurves.Define(stage, f"{parent}/Start_{start}")
                points = []
                phase = 2.0 * math.pi * start / starts
                for sample in range(samples_per_start):
                    fraction = sample / (samples_per_start - 1)
                    angle = phase + 2.0 * math.pi * turns * fraction
                    points.append(
                        Gf.Vec3f(
                            radius * math.cos(angle),
                            radius * math.sin(angle),
                            z0 + (z1 - z0) * fraction,
                        )
                    )
                curve.CreateTypeAttr(UsdGeom.Tokens.linear)
                curve.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
                curve.CreateCurveVertexCountsAttr([samples_per_start])
                curve.CreatePointsAttr(points)
                curve.CreateWidthsAttr([0.00032] * samples_per_start)
                curve.CreateDisplayColorAttr([Gf.Vec3f(*colors["nickel_dark"])])
                curve.GetPrim().SetCustomDataByKey(
                    "j599:threadStartPhaseDeg", 120.0 * start
                )
                curve.GetPrim().SetCustomDataByKey(
                    "j599:collisionMode", "visual_only"
                )

        fixed = xform(FIXED_PATH)
        fixed.GetPrim().SetCustomDataByKey("j599:partNumber", "J599/20FJ35SN")
        fixed.GetPrim().SetCustomDataByKey("j599:role", "fixed_socket_receptacle")
        fixed.GetPrim().SetCustomDataByKey(
            "j599:representativeMassKg", float(mass["receptacle_total_mass_kg"])
        )

        cube(
            FIXED_PATH + "/Visual/Flange",
            (0.046, 0.046, 0.00391),
            (0.0, 0.0, -0.0092),
            colors["nickel"],
        )
        cylinder(
            FIXED_PATH + "/Visual/RearShell",
            0.0190,
            0.0180,
            (0.0, 0.0, -0.0190),
            colors["nickel_dark"],
        )
        band(
            FIXED_PATH + "/Visual/RedMatedBand",
            0.01965,
            -0.0065,
            colors["red_band"],
        )
        mount_radius = 0.5 * 0.03810
        for index, angle_deg in enumerate((45.0, 135.0, 225.0, 315.0), 1):
            angle = math.radians(angle_deg)
            cylinder(
                f"{FIXED_PATH}/Visual/MountHole_{index}",
                0.00205,
                0.0041,
                (
                    mount_radius * math.cos(angle),
                    mount_radius * math.sin(angle),
                    -0.0092,
                ),
                colors["hole"],
            )

        insert_radius = float(geometry["receptacle_insert_radius_m"])
        cylinder(
            FIXED_PATH + "/Visual/SocketInsertFace",
            insert_radius,
            0.00055,
            (0.0, 0.0, -0.00032),
            colors["insert"],
        )
        sockets_parent = FIXED_PATH + "/Contacts/Sockets"
        xform(sockets_parent)
        socket_radius = (
            0.5
            * float(
                contract["public_interface_geometry"]["contacts"][
                    "socket_visual_entry_diameter_mm"
                ]
            )
            * 1.0e-3
        )
        for contact in contacts:
            socket = cylinder(
                f"{sockets_parent}/Socket_{contact.contact_id:03d}",
                socket_radius,
                0.00030,
                (contact.x_m, contact.y_m, 0.00002),
                colors["socket"],
            )
            socket.GetPrim().SetCustomDataByKey(
                "j599:contactId", contact.contact_id
            )
            socket.GetPrim().SetCustomDataByKey(
                "j599:coordinateSourcePage", contact.source_page
            )
            socket.GetPrim().SetCustomDataByKey(
                "j599:collisionMode", "visual_only"
            )
            counts["receptacle_sockets"] += 1

        guide_visual_angles = _blocking_angles(features, bore_radius, 180)
        ring_segments(
            FIXED_PATH + "/Visual/KeywayShell",
            bore_radius,
            fixed_outer_radius,
            key_length,
            -0.5 * key_length,
            guide_visual_angles,
            180,
            colors["nickel"],
        )
        helical_curves(
            FIXED_PATH + "/Visual/ExternalTripleStartThread",
            0.02035,
            -0.0065,
            -0.0003,
        )

        if physics:
            blocking_angles = _blocking_angles(features, bore_radius, 180)
            ring_segments(
                FIXED_PATH + "/Physics/KeywayBlockingShell",
                bore_radius,
                fixed_outer_radius,
                key_length,
                -0.5 * key_length,
                blocking_angles,
                180,
                colors["nickel"],
                collision_role="fixed_keyway_blocking_shell",
                visible=False,
            )
            wall_thickness = 0.00020
            radial_length = keyway_outer_radius - bore_radius
            radial_center = 0.5 * (keyway_outer_radius + bore_radius)
            for feature in features:
                angle = math.radians(feature.angle_deg)
                radial = (math.cos(angle), math.sin(angle))
                tangent = (-math.sin(angle), math.cos(angle))
                feature_parent = (
                    f"{FIXED_PATH}/Physics/Keyways/"
                    f"Keyway_{feature.index}_{feature.name}"
                )
                xform(feature_parent)
                for side_index, side_sign in enumerate((-1.0, 1.0)):
                    offset = side_sign * 0.5 * (
                        feature.keyway_width_m + wall_thickness
                    )
                    wall = cube(
                        f"{feature_parent}/SideWall_{side_index}",
                        (radial_length, wall_thickness, key_length),
                        (
                            radial_center * radial[0] + offset * tangent[0],
                            radial_center * radial[1] + offset * tangent[1],
                            -0.5 * key_length,
                        ),
                        colors["key"],
                        rotation_z_deg=feature.angle_deg,
                        collision_role="fixed_keyway_sidewall",
                        visible=False,
                    )
                    wall.GetPrim().SetCustomDataByKey(
                        "j599:keyIndex", feature.index
                    )
                counts["keyways"] += 1
            ring_segments(
                FIXED_PATH + "/Physics/MetalStop",
                0.0154,
                0.0163,
                0.00030,
                -0.00015,
                _angles(48),
                48,
                colors["nickel_dark"],
                collision_role="fixed_metal_stop",
                visible=False,
            )

        xform(LOOSE_PATH)
        body = xform(BODY_PATH)
        body.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, initial_z))
        body.GetPrim().SetCustomDataByKey("j599:partNumber", "J599/26FJ35PN")
        body.GetPrim().SetCustomDataByKey("j599:role", "loose_pin_plug_body")

        cylinder(
            BODY_PATH + "/Visual/PinInsertFace",
            insert_radius,
            0.00055,
            (0.0, 0.0, 0.00032),
            colors["insert"],
        )
        ring_segments(
            BODY_PATH + "/Visual/MatingShell",
            0.0167,
            plug_guide_radius,
            key_length,
            -0.5 * key_length,
            _angles(72),
            72,
            colors["nickel"],
        )
        cylinder(
            BODY_PATH + "/Visual/RearShell",
            float(geometry["plug_body_radius_m"]),
            0.0210,
            (0.0, 0.0, 0.0110),
            colors["nickel_dark"],
        )
        cylinder(
            BODY_PATH + "/Visual/WireSideRelief",
            0.0165,
            0.0100,
            (0.0, 0.0, 0.0255),
            colors["nickel"],
        )
        band(
            BODY_PATH + "/Visual/BlueBand",
            float(geometry["plug_body_radius_m"]) + 0.00008,
            0.0042,
            colors["blue_band"],
        )

        pins_parent = BODY_PATH + "/Contacts/Pins"
        xform(pins_parent)
        pin_radius = (
            0.5
            * float(
                contract["public_interface_geometry"]["contacts"][
                    "pin_working_diameter_mm"
                ]
            )
            * 1.0e-3
        )
        for contact in contacts:
            pin = cylinder(
                f"{pins_parent}/Pin_{contact.contact_id:03d}",
                pin_radius,
                0.0040,
                (contact.x_m, contact.y_m, -0.0020),
                colors["gold"],
            )
            pin.GetPrim().SetCustomDataByKey(
                "j599:contactId", contact.contact_id
            )
            pin.GetPrim().SetCustomDataByKey(
                "j599:coordinateSourcePage", contact.source_page
            )
            pin.GetPrim().SetCustomDataByKey(
                "j599:collisionMode", "visual_only"
            )
            counts["plug_pins"] += 1

        key_parent = BODY_PATH + "/Keys"
        xform(key_parent)
        key_radial_size = plug_key_outer_radius - plug_guide_radius
        key_center_radius = 0.5 * (plug_key_outer_radius + plug_guide_radius)
        for feature in features:
            angle = math.radians(feature.angle_deg)
            key = cube(
                f"{key_parent}/Key_{feature.index}_{feature.name}",
                (key_radial_size, feature.key_width_m, key_length),
                (
                    key_center_radius * math.cos(angle),
                    key_center_radius * math.sin(angle),
                    -0.5 * key_length,
                ),
                colors["key"],
                rotation_z_deg=feature.angle_deg,
                collision_role="plug_polarizing_key" if physics else None,
            )
            key.GetPrim().SetCustomDataByKey("j599:keyIndex", feature.index)
            key.GetPrim().SetCustomDataByKey(
                "j599:keyAngleDeg", feature.angle_deg
            )
            key.GetPrim().SetCustomDataByKey(
                "j599:keyWidthM", feature.key_width_m
            )
            counts["keys"] += 1

        if physics:
            UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
            mass_api = UsdPhysics.MassAPI.Apply(body.GetPrim())
            body_mass = float(mass["plug_body_mass_kg_simulation_split"])
            mass_api.CreateMassAttr(body_mass)
            mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.011))
            mass_api.CreateDiagonalInertiaAttr(
                Gf.Vec3f(8.2e-6, 8.2e-6, 8.1e-6)
            )
            mass_api.CreatePrincipalAxesAttr(Gf.Quatf(1.0))
            ring_segments(
                BODY_PATH + "/Physics/GuideShell",
                0.01725,
                plug_guide_radius,
                key_length,
                -0.5 * key_length,
                _angles(72),
                72,
                colors["nickel"],
                collision_role="plug_continuous_guide_shell",
                visible=False,
            )
            ring_segments(
                BODY_PATH + "/Physics/MetalStop",
                0.0154,
                0.0163,
                0.00030,
                0.00015,
                _angles(48),
                48,
                colors["nickel_dark"],
                collision_role="plug_metal_stop",
                visible=False,
            )

        nut = xform(NUT_PATH)
        nut.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, initial_z))
        nut.GetPrim().SetCustomDataByKey("j599:role", "separate_coupling_nut")
        nut.GetPrim().SetCustomDataByKey("j599:threadStarts", 3)
        nut.GetPrim().SetCustomDataByKey(
            "j599:threadLeadMPerRevolution", 0.00762
        )
        ring_segments(
            NUT_PATH + "/Visual/KnurledShell",
            0.0200,
            float(geometry["coupling_nut_outer_radius_m"]),
            0.0140,
            0.0005,
            _angles(32),
            32,
            colors["nickel"],
        )
        helical_curves(
            NUT_PATH + "/Visual/InternalThreadReference",
            0.0200,
            -0.0060,
            0.0060,
        )
        if physics:
            UsdPhysics.RigidBodyAPI.Apply(nut.GetPrim())
            nut_mass_api = UsdPhysics.MassAPI.Apply(nut.GetPrim())
            nut_mass_api.CreateMassAttr(
                float(mass["coupling_nut_mass_kg_simulation_split"])
            )
            nut_mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0005))
            nut_mass_api.CreateDiagonalInertiaAttr(
                Gf.Vec3f(6.5e-6, 6.5e-6, 1.21e-5)
            )
            nut_mass_api.CreatePrincipalAxesAttr(Gf.Quatf(1.0))

            joint = UsdPhysics.RevoluteJoint.Define(stage, NUT_JOINT_PATH)
            joint.CreateAxisAttr("Z")
            joint.CreateBody0Rel().SetTargets([body.GetPrim().GetPath()])
            joint.CreateBody1Rel().SetTargets([nut.GetPrim().GetPath()])
            joint.CreateLocalPos0Attr(Gf.Vec3f(0.0))
            joint.CreateLocalPos1Attr(Gf.Vec3f(0.0))
            joint.CreateLocalRot0Attr(Gf.Quatf(1.0))
            joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
            joint.CreateCollisionEnabledAttr(False)
            joint.GetPrim().SetCustomDataByKey(
                "j599:jointRole", "coupling_nut_revolute"
            )

        frame = xform(PAIR_ROOT + "/AssemblyFrame")
        frame.GetPrim().SetCustomDataByKey(
            "j599:origin", "receptacle_mating_face_center"
        )
        frame.GetPrim().SetCustomDataByKey(
            "j599:plusZ", "toward_approaching_plug"
        )
        frame.GetPrim().SetCustomDataByKey(
            "j599:plusX", "main_key_centerline_0deg"
        )
        frame.GetPrim().SetCustomDataByKey(
            "j599:contactCoordinateRule",
            "same_standard_table_in_world_assembly_frame",
        )
        return counts

    visual_stage = make_stage(False)
    visual_counts = author(visual_stage, False)
    visual_stage.GetRootLayer().Export(str(visual_path))

    collision_counts.clear()
    assembly_stage = make_stage(True)
    assembly_counts = author(assembly_stage, True)
    assembly_stage.GetRootLayer().Export(str(assembly_path))
    assembly_stage.GetRootLayer().Export(str(binary_path))

    outputs = {
        "visual_usda": visual_path,
        "assembly_usda": assembly_path,
        "assembly_usdc": binary_path,
    }
    for label, path in outputs.items():
        if not path.is_file() or path.stat().st_size < 1000:
            raise RuntimeError(f"{label} export is missing or unexpectedly small")

    report = {
        "schema_version": "j599_asset_build_report_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "GENERATED_NOT_DYNAMICALLY_VALIDATED",
        "model_id": contract["model_id"],
        "contract_sha256": _sha256(contract_path),
        "contact_table_sha256": _sha256(contacts_path),
        "input_checks": {
            "contact_count": len(contacts),
            "contact_ids_exact": True,
            "key_count": len(features),
            "nominal_key_clearances_m": nominal_key_clearances_m(contract),
            "wrong_yaw_interferes_analytic": wrong_yaw_interferes(
                contract, wrong_yaw
            ),
            "max_contact_radius_m": max(
                math.hypot(item.x_m, item.y_m) for item in contacts
            ),
        },
        "visual_counts": visual_counts,
        "assembly_counts": assembly_counts,
        "collision_role_counts": dict(sorted(collision_counts.items())),
        "outputs": {
            label: {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for label, path in outputs.items()
        },
        "claims": {
            "manufacturer_exact_fidelity": False,
            "hardware_authorized": False,
            "dynamic_assembly_passed": False,
            "generated_file_alone_is_acceptance": False,
        },
    }
    report_path = output_dir / "build_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report["build_report"] = str(report_path)
    return report


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    report = build_assets(args.contract, args.contacts, args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
