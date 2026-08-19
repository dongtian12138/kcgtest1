#!/usr/bin/env python3

"""Generate the independent public-spec D38999 keyed-v2 USD asset.

The pure planning helpers in this file deliberately import neither Isaac Sim
nor Pixar USD.  They make the source dimensions, prim identities, contact
layout, and key-before-contact ordering reviewable with ordinary CPU tests.
The simulator is imported only after :class:`isaacsim.SimulationApp` starts.

This is a simulation model derived from public interface specifications.  It
does not claim manufacturer CAD fidelity, hardware calibration, qualification,
or a model of the coupling thread.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import traceback
from typing import NamedTuple

from kcg_connector.d38999_keyed_public_spec_v2 import (
    DEFAULT_CONFIG_PATH,
    PAIR_MODEL_ID,
    PLUG_MODEL_ID,
    RECEPTACLE_MODEL_ID,
    RECOMMENDED_ASSET_NAME,
    ROOT_PRIM,
    MassPropertiesAssumption,
    PublicSpecKeyedV2,
    load_keyed_public_spec_v2,
    safe_new_asset_output,
)


MM_TO_M = 1.0e-3
INITIAL_PAIR_SEPARATION_M = -0.050
RECEPTACLE_GUIDE_LENGTH_M = 0.014
FIRST_ELECTRICAL_CONTACT_PLANE_Z_M = 0.012
POLARIZATION_COLLISION_PLANE_Z_M = 0.0
PLUG_MATING_SHELL_LENGTH_M = 0.012
KEYWAY_BLOCKING_SEGMENT_COUNT = 360
KEYWAY_WALL_THICKNESS_M = 0.00020
FIXED_SHELL_OUTER_RADIUS_M = 0.0230
PLUG_MATING_SHELL_INNER_RADIUS_M = 0.0145
RECEPTACLE_CONTACT_VISUAL_LENGTH_M = 0.0025
INSERT_FACE_VISUAL_THICKNESS_M = 0.0005
PLUG_SOCKET_VISUAL_DEPTH_M = 0.00035
COLLISION_CONTACT_OFFSET_M_SIM_ASSUMPTION = 1.0e-5
COLLISION_REST_OFFSET_M_SIM_ASSUMPTION = 0.0


class ContactVisualSpec(NamedTuple):
    """One MIL-STD-1560 25-61 position in the common assembly XY frame."""

    label: str
    x_m: float
    y_m: float


class KeyFeatureSpec(NamedTuple):
    """One plug key and its corresponding receptacle keyway."""

    index: int
    name: str
    angle_deg: float
    key_width_m: float
    keyway_width_m: float


class AssetPlan(NamedTuple):
    """Pure, immutable authoring plan consumed by the USD implementation."""

    root_path: str
    fixed_path: str
    loose_path: str
    body_path: str
    nut_path: str
    joint_path: str
    pair_model_id: str
    plug_model_id: str
    receptacle_model_id: str
    recommended_asset_name: str
    contact_collision_mode: str
    thread_collision_mode: str
    geometry_fidelity: str
    contacts: tuple[ContactVisualSpec, ...]
    keys: tuple[KeyFeatureSpec, ...]
    plug_shell_outer_radius_m: float
    plug_key_outer_radius_m: float
    plug_rear_body_radius_m: float
    receptacle_bore_radius_m: float
    receptacle_keyway_outer_radius_m: float
    plug_key_length_m: float
    receptacle_guide_length_m: float
    first_electrical_contact_plane_z_m: float
    polarization_collision_plane_z_m: float
    initial_pair_separation_m: float
    blocking_segment_count: int
    plug_socket_visual_front_offset_m: float
    body_mass_properties: MassPropertiesAssumption
    nut_mass_properties: MassPropertiesAssumption
    mass_property_source_kind: str
    mass_property_source_asset_revision: str


def recommended_output_path() -> Path:
    """Return the new keyed-v2 location; the output guard still forbids reuse."""

    return Path(__file__).resolve().parents[1] / "assets" / RECOMMENDED_ASSET_NAME


def _diameter_midpoint_radius_m(dimensions, minimum: str, maximum: str) -> float:
    return 0.25 * (
        float(dimensions[minimum]) + float(dimensions[maximum])
    ) * MM_TO_M


def _require_asset_contract(model: PublicSpecKeyedV2) -> None:
    geometry = model.document["asset_geometry"]
    expected = {
        "recommended_asset_name": RECOMMENDED_ASSET_NAME,
        "root_prim": ROOT_PRIM,
        "fixed_receptacle_suffix": "/FixedReceptacle",
        "loose_body_suffix": "/LoosePlug/BodyAssembly",
        "coupling_nut_suffix": "/LoosePlug/CouplingNut",
        "coupling_joint_suffix": "/LoosePlug/CouplingNutJoint",
        "contact_collision_mode": "visual_only",
        "key_collision_mode": "five_key_and_matching_keyway_walls",
        "thread_collision_mode": "unmodeled",
        "authoring_profile": "nominal_midpoint_geometry",
    }
    for field, required in expected.items():
        if geometry.get(field) != required:
            raise ValueError(f"asset_geometry.{field} must be {required}")
    if (
        geometry.get("plug_socket_visual_front_offset_m_sim_assumption")
        != 0.00005
    ):
        raise ValueError(
            "plug socket visual fronts must remain 50 um ahead of the face"
        )


def build_asset_plan(model: PublicSpecKeyedV2) -> AssetPlan:
    """Convert the validated public-spec document into an authoring plan."""

    _require_asset_contract(model)
    dimensions = model.document["interface_dimensions_mm"]
    plug_dimensions = dimensions["plug"]
    receptacle_dimensions = dimensions["receptacle"]
    keying = model.document["keying"]

    plug_shell_radius = _diameter_midpoint_radius_m(
        plug_dimensions,
        "w_between_keys_diameter_min",
        "w_between_keys_diameter_max",
    )
    plug_key_outer_radius = _diameter_midpoint_radius_m(
        plug_dimensions,
        "v_over_keys_diameter_min",
        "v_over_keys_diameter_max",
    )
    plug_rear_body_radius = (
        0.5
        * float(plug_dimensions["b_rear_shell_diameter_basic"])
        * MM_TO_M
    )
    receptacle_bore_radius = _diameter_midpoint_radius_m(
        receptacle_dimensions,
        "h_bore_diameter_min",
        "h_bore_diameter_max",
    )
    receptacle_keyway_outer_radius = _diameter_midpoint_radius_m(
        receptacle_dimensions,
        "j_keyway_outer_diameter_min",
        "j_keyway_outer_diameter_max",
    )
    if not (
        plug_shell_radius
        < receptacle_bore_radius
        < plug_key_outer_radius
        < receptacle_keyway_outer_radius
    ):
        raise ValueError("public interface radii do not form a mating keyed pair")

    angles = model.key_angles_deg
    features = tuple(
        KeyFeatureSpec(
            index=index,
            name="Main" if index == 0 else f"Minor{index}",
            angle_deg=angle,
            key_width_m=float(
                keying[
                    "plug_main_key_width_nominal"
                    if index == 0
                    else "plug_minor_key_width_nominal"
                ]
            )
            * MM_TO_M,
            keyway_width_m=float(
                keying[
                    "receptacle_main_keyway_width_nominal"
                    if index == 0
                    else "receptacle_minor_keyway_width_nominal"
                ]
            )
            * MM_TO_M,
        )
        for index, angle in enumerate(angles)
    )
    if len(features) != 5 or any(
        feature.key_width_m >= feature.keyway_width_m for feature in features
    ):
        raise ValueError("all five plug keys must have positive keyway clearance")

    contacts = tuple(
        ContactVisualSpec(
            label=contact.label,
            x_m=contact.pin_front_mm[0] * MM_TO_M,
            y_m=contact.pin_front_mm[1] * MM_TO_M,
        )
        for contact in model.contacts
    )
    if len(contacts) != 61:
        raise ValueError("the public 25-61 table must author 61 contact visuals")

    geometry = model.document["asset_geometry"]
    loose_path = ROOT_PRIM + "/LoosePlug"
    plan = AssetPlan(
        root_path=ROOT_PRIM,
        fixed_path=ROOT_PRIM + geometry["fixed_receptacle_suffix"],
        loose_path=loose_path,
        body_path=ROOT_PRIM + geometry["loose_body_suffix"],
        nut_path=ROOT_PRIM + geometry["coupling_nut_suffix"],
        joint_path=ROOT_PRIM + geometry["coupling_joint_suffix"],
        pair_model_id=PAIR_MODEL_ID,
        plug_model_id=PLUG_MODEL_ID,
        receptacle_model_id=RECEPTACLE_MODEL_ID,
        recommended_asset_name=RECOMMENDED_ASSET_NAME,
        contact_collision_mode=geometry["contact_collision_mode"],
        thread_collision_mode=geometry["thread_collision_mode"],
        geometry_fidelity=(
            "public_spec_nominal_midpoint_simulation_only_not_manufacturer_cad"
        ),
        contacts=contacts,
        keys=features,
        plug_shell_outer_radius_m=plug_shell_radius,
        plug_key_outer_radius_m=plug_key_outer_radius,
        plug_rear_body_radius_m=plug_rear_body_radius,
        receptacle_bore_radius_m=receptacle_bore_radius,
        receptacle_keyway_outer_radius_m=receptacle_keyway_outer_radius,
        plug_key_length_m=float(keying["plug_key_axial_length_min"]) * MM_TO_M,
        receptacle_guide_length_m=RECEPTACLE_GUIDE_LENGTH_M,
        first_electrical_contact_plane_z_m=FIRST_ELECTRICAL_CONTACT_PLANE_Z_M,
        polarization_collision_plane_z_m=POLARIZATION_COLLISION_PLANE_Z_M,
        initial_pair_separation_m=INITIAL_PAIR_SEPARATION_M,
        blocking_segment_count=KEYWAY_BLOCKING_SEGMENT_COUNT,
        plug_socket_visual_front_offset_m=float(
            geometry["plug_socket_visual_front_offset_m_sim_assumption"]
        ),
        body_mass_properties=model.body_mass_properties,
        nut_mass_properties=model.nut_mass_properties,
        mass_property_source_kind=model.mass_property_source_kind,
        mass_property_source_asset_revision=(
            model.mass_property_source_asset_revision
        ),
    )
    if not (
        plan.polarization_collision_plane_z_m
        < plan.first_electrical_contact_plane_z_m
        < plan.receptacle_guide_length_m
    ):
        raise ValueError("polarization collision must occur before contact")
    return plan


def _angular_distance_rad(first: float, second: float) -> float:
    return abs((first - second + math.pi) % (2.0 * math.pi) - math.pi)


def key_pattern_fits_at_yaw(plan: AssetPlan, yaw_deg: float) -> bool:
    """Conservatively test whether all five keys can enter unique keyways.

    This is not a physics replacement.  It is a pure review check that the
    authored nominal key widths and N-polarized pattern accept zero yaw but do
    not accidentally retain the old 180-degree/C2 ambiguity.
    """

    if not math.isfinite(float(yaw_deg)):
        raise ValueError("yaw_deg must be finite")
    if plan.plug_key_outer_radius_m >= plan.receptacle_keyway_outer_radius_m:
        return False

    candidates: list[tuple[int, ...]] = []
    yaw_rad = math.radians(float(yaw_deg))
    for key in plan.keys:
        key_angle = math.radians(key.angle_deg) + yaw_rad
        matching_slots = []
        for slot_index, slot in enumerate(plan.keys):
            delta = _angular_distance_rad(
                key_angle, math.radians(slot.angle_deg)
            )
            if delta > 0.5 * math.pi:
                continue
            tangential_envelope = (
                plan.plug_key_outer_radius_m * math.sin(delta)
                + 0.5 * key.key_width_m * math.cos(delta)
            )
            if tangential_envelope <= 0.5 * slot.keyway_width_m + 1.0e-12:
                matching_slots.append(slot_index)
        candidates.append(tuple(matching_slots))

    def assign(key_index: int, used_slots: frozenset[int]) -> bool:
        if key_index == len(candidates):
            return True
        return any(
            slot_index not in used_slots
            and assign(key_index + 1, used_slots | {slot_index})
            for slot_index in candidates[key_index]
        )

    return assign(0, frozenset())


def receptacle_blocking_segment_angles_deg(plan: AssetPlan) -> tuple[float, ...]:
    """Return annular-shell segments, leaving conservative keyway openings."""

    step = 2.0 * math.pi / plan.blocking_segment_count
    retained = []
    for index in range(plan.blocking_segment_count):
        angle = index * step
        intersects_opening = False
        for feature in plan.keys:
            opening_half_angle = math.atan2(
                0.5 * feature.keyway_width_m,
                plan.receptacle_bore_radius_m,
            )
            # Omit the whole segment when any part could intrude into a
            # keyway.  Exact-width collision is then supplied by sidewalls.
            if _angular_distance_rad(
                angle, math.radians(feature.angle_deg)
            ) <= opening_half_angle + 0.55 * step:
                intersects_opening = True
                break
        if not intersects_opening:
            retained.append(math.degrees(angle))
    return tuple(retained)


def _arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate the independent public-spec D38999 keyed-v2 USD"
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="validated public-spec keyed-v2 YAML",
    )
    parser.add_argument(
        "--output",
        default=str(recommended_output_path()),
        help=(
            "new .usd/.usda path; basename is fixed to "
            + RECOMMENDED_ASSET_NAME
        ),
    )
    return parser.parse_args(argv)


def main(argv=None):
    arguments = _arguments(argv)
    model = load_keyed_public_spec_v2(arguments.config)
    plan = build_asset_plan(model)
    output_path = safe_new_asset_output(arguments.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": True,
            "multi_gpu": False,
            "active_gpu": 0,
            "physics_gpu": 0,
        }
    )
    passed = False
    try:
        import omni.usd
        from omni.physx.scripts import physicsUtils as physics_utils
        from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade

        def author_explicit_mass_properties(prim, properties):
            mass_api = UsdPhysics.MassAPI.Apply(prim)
            mass_api.CreateMassAttr(properties.mass_kg)
            mass_api.CreateCenterOfMassAttr(
                Gf.Vec3f(*properties.center_of_mass_m)
            )
            mass_api.CreateDiagonalInertiaAttr(
                Gf.Vec3f(*properties.diagonal_inertia_kg_m2)
            )
            x, y, z, w = properties.principal_axes_xyzw
            mass_api.CreatePrincipalAxesAttr(Gf.Quatf(w, x, y, z))
            prim.SetCustomDataByKey(
                "kcg:massSource", "simulation_assumption_not_public_spec"
            )
            prim.SetCustomDataByKey(
                "kcg:massPropertiesSource", plan.mass_property_source_kind
            )
            prim.SetCustomDataByKey(
                "kcg:massPropertiesSourceAssetRevision",
                plan.mass_property_source_asset_revision,
            )

        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

        world = UsdGeom.Xform.Define(stage, "/World")
        root = UsdGeom.Xform.Define(stage, plan.root_path)
        root_prim = root.GetPrim()
        root_metadata = {
            "schemaVersion": model.document["schema_version"],
            "pairModelId": plan.pair_model_id,
            "loosePlugModelId": plan.plug_model_id,
            "fixedReceptacleModelId": plan.receptacle_model_id,
            "fidelity": plan.geometry_fidelity,
            "polarization": "N",
            "keyAnglesDeg": ",".join(
                f"{feature.angle_deg:g}" for feature in plan.keys
            ),
            "contactPattern": "MIL-STD-1560_25-61_visual-only",
            "contactCollisionMode": plan.contact_collision_mode,
            "keyCollisionMode": "five_key_and_matching_keyway_walls",
            "threadCollisionMode": plan.thread_collision_mode,
            "collisionContactOffset_m": COLLISION_CONTACT_OFFSET_M_SIM_ASSUMPTION,
            "collisionRestOffset_m": COLLISION_REST_OFFSET_M_SIM_ASSUMPTION,
            "collisionOffsetSource": "simulation_numeric_choice_not_public_spec",
            "assemblyPlusZ": "loose_plug_insertion_into_fixed_receptacle",
            "certificationClaim": "none",
            "realHardwareFidelityClaimed": False,
            "spaceQualificationClaimed": False,
        }
        for key, value in root_metadata.items():
            root_prim.SetCustomDataByKey(f"kcg:{key}", value)

        material_path = plan.root_path + "/Materials/MetalCollision"
        material = UsdShade.Material.Define(stage, material_path)
        material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        material_api.CreateStaticFrictionAttr(0.35)
        material_api.CreateDynamicFrictionAttr(0.25)
        material_api.CreateRestitutionAttr(0.0)
        collision_prims = []

        def mark_collision(prim, role):
            UsdPhysics.CollisionAPI.Apply(prim)
            physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            physx_collision.CreateContactOffsetAttr().Set(
                COLLISION_CONTACT_OFFSET_M_SIM_ASSUMPTION
            )
            physx_collision.CreateRestOffsetAttr().Set(
                COLLISION_REST_OFFSET_M_SIM_ASSUMPTION
            )
            prim.SetCustomDataByKey("kcg:collisionRole", role)
            collision_prims.append(prim)

        def cube(
            path,
            size,
            translation,
            color,
            *,
            rotation_z_deg=0.0,
            collision_role=None,
        ):
            shape = UsdGeom.Cube.Define(stage, path)
            shape.CreateSizeAttr(1.0)
            shape.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            transform = UsdGeom.Xformable(shape)
            transform.AddTranslateOp().Set(Gf.Vec3d(*translation))
            if abs(rotation_z_deg) > 1.0e-12:
                transform.AddRotateZOp().Set(rotation_z_deg)
            transform.AddScaleOp().Set(Gf.Vec3f(*size))
            if collision_role is not None:
                mark_collision(shape.GetPrim(), collision_role)
            return shape

        def cylinder(
            path,
            radius,
            height,
            translation,
            color,
            *,
            collision_role=None,
        ):
            shape = UsdGeom.Cylinder.Define(stage, path)
            shape.CreateAxisAttr(UsdGeom.Tokens.z)
            shape.CreateRadiusAttr(radius)
            shape.CreateHeightAttr(height)
            shape.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            UsdGeom.Xformable(shape).AddTranslateOp().Set(
                Gf.Vec3d(*translation)
            )
            if collision_role is not None:
                mark_collision(shape.GetPrim(), collision_role)
            return shape

        def ring_segments(
            parent_path,
            inner_radius,
            outer_radius,
            height,
            center_z,
            angles_deg,
            reference_segment_count,
            color,
            collision_role,
        ):
            UsdGeom.Xform.Define(stage, parent_path)
            center_radius = 0.5 * (inner_radius + outer_radius)
            radial_size = outer_radius - inner_radius
            tangential_size = (
                1.04
                * 2.0
                * math.pi
                * center_radius
                / reference_segment_count
            )
            for index, angle_deg in enumerate(angles_deg):
                angle_rad = math.radians(angle_deg)
                cube(
                    f"{parent_path}/Segment_{index:03d}",
                    (radial_size, tangential_size, height),
                    (
                        center_radius * math.cos(angle_rad),
                        center_radius * math.sin(angle_rad),
                        center_z,
                    ),
                    color,
                    rotation_z_deg=angle_deg,
                    collision_role=collision_role,
                )

        fixed = UsdGeom.Xform.Define(stage, plan.fixed_path)
        fixed.GetPrim().SetCustomDataByKey("kcg:role", "fixed_receptacle")
        fixed.GetPrim().SetCustomDataByKey(
            "kcg:modelId", plan.receptacle_model_id
        )
        fixed.GetPrim().SetCustomDataByKey(
            "kcg:polarizationCollisionPlaneZ_m",
            plan.polarization_collision_plane_z_m,
        )
        keyway_parent = plan.fixed_path + "/CollisionKeyways"
        UsdGeom.Xform.Define(stage, keyway_parent)
        blocking_angles = receptacle_blocking_segment_angles_deg(plan)
        ring_segments(
            keyway_parent + "/BlockingShell",
            plan.receptacle_bore_radius_m,
            FIXED_SHELL_OUTER_RADIUS_M,
            plan.receptacle_guide_length_m,
            0.5 * plan.receptacle_guide_length_m,
            blocking_angles,
            plan.blocking_segment_count,
            (0.18, 0.39, 0.64),
            "polarization_blocking_shell",
        )

        keyway_radial_length = (
            plan.receptacle_keyway_outer_radius_m
            - plan.receptacle_bore_radius_m
        )
        keyway_center_radius = 0.5 * (
            plan.receptacle_keyway_outer_radius_m
            + plan.receptacle_bore_radius_m
        )
        for feature in plan.keys:
            feature_path = (
                f"{keyway_parent}/Keyway_{feature.index:02d}_{feature.name}"
            )
            UsdGeom.Xform.Define(stage, feature_path)
            angle_rad = math.radians(feature.angle_deg)
            radial_x = math.cos(angle_rad)
            radial_y = math.sin(angle_rad)
            tangent_x = -radial_y
            tangent_y = radial_x
            for side_name, side_sign in (("Negative", -1.0), ("Positive", 1.0)):
                tangent_offset = side_sign * 0.5 * (
                    feature.keyway_width_m + KEYWAY_WALL_THICKNESS_M
                )
                wall = cube(
                    feature_path + f"/SideWall{side_name}",
                    (
                        keyway_radial_length,
                        KEYWAY_WALL_THICKNESS_M,
                        plan.receptacle_guide_length_m,
                    ),
                    (
                        keyway_center_radius * radial_x
                        + tangent_offset * tangent_x,
                        keyway_center_radius * radial_y
                        + tangent_offset * tangent_y,
                        0.5 * plan.receptacle_guide_length_m,
                    ),
                    (0.92, 0.26, 0.12),
                    rotation_z_deg=feature.angle_deg,
                    collision_role="polarization_keyway_sidewall",
                )
                wall.GetPrim().SetCustomDataByKey(
                    "kcg:keyAngleDeg", feature.angle_deg
                )
                wall.GetPrim().SetCustomDataByKey(
                    "kcg:keywayWidth_m", feature.keyway_width_m
                )
            outer_wall_radius = (
                plan.receptacle_keyway_outer_radius_m
                + 0.5 * KEYWAY_WALL_THICKNESS_M
            )
            outer_wall = cube(
                feature_path + "/OuterWall",
                (
                    KEYWAY_WALL_THICKNESS_M,
                    feature.keyway_width_m + 2.0 * KEYWAY_WALL_THICKNESS_M,
                    plan.receptacle_guide_length_m,
                ),
                (
                    outer_wall_radius * radial_x,
                    outer_wall_radius * radial_y,
                    0.5 * plan.receptacle_guide_length_m,
                ),
                (0.78, 0.20, 0.10),
                rotation_z_deg=feature.angle_deg,
                collision_role="polarization_keyway_outer_wall",
            )
            outer_wall.GetPrim().SetCustomDataByKey(
                "kcg:keyAngleDeg", feature.angle_deg
            )

        flange_center_z = plan.receptacle_guide_length_m + 0.0015
        cube(
            plan.fixed_path + "/FlangeVisual",
            (0.046, 0.046, 0.0030),
            (0.0, 0.0, flange_center_z),
            (0.26, 0.29, 0.33),
        ).GetPrim().SetCustomDataByKey("kcg:collisionMode", "visual_only")
        rear_body = cylinder(
            plan.fixed_path + "/RearBody",
            0.0190,
            0.0180,
            (0.0, 0.0, plan.receptacle_guide_length_m + 0.0090),
            (0.22, 0.25, 0.29),
            collision_role="fixed_rear_body",
        )
        rear_body.GetPrim().SetCustomDataByKey(
            "kcg:startsAfterKeyGuide", True
        )

        fixed_visuals = plan.fixed_path + "/ContactVisuals"
        fixed_pins = fixed_visuals + "/Pins"
        UsdGeom.Xform.Define(stage, fixed_visuals)
        UsdGeom.Xform.Define(stage, fixed_pins)
        cylinder(
            fixed_visuals + "/InsertFace",
            0.86 * plan.receptacle_bore_radius_m,
            INSERT_FACE_VISUAL_THICKNESS_M,
            (
                0.0,
                0.0,
                plan.first_electrical_contact_plane_z_m
                + RECEPTACLE_CONTACT_VISUAL_LENGTH_M
                + 0.5 * INSERT_FACE_VISUAL_THICKNESS_M,
            ),
            (0.81, 0.75, 0.60),
        ).GetPrim().SetCustomDataByKey("kcg:collisionMode", "visual_only")
        pin_radius_m = 0.5 * 0.040 * 25.4 * MM_TO_M
        for index, contact in enumerate(plan.contacts):
            pin = cylinder(
                f"{fixed_pins}/Pin_{index:02d}_{contact.label}",
                pin_radius_m,
                RECEPTACLE_CONTACT_VISUAL_LENGTH_M,
                (
                    contact.x_m,
                    contact.y_m,
                    plan.first_electrical_contact_plane_z_m
                    + 0.5 * RECEPTACLE_CONTACT_VISUAL_LENGTH_M,
                ),
                (0.91, 0.66, 0.16),
            )
            pin.GetPrim().SetCustomDataByKey("kcg:contactLabel", contact.label)
            pin.GetPrim().SetCustomDataByKey(
                "kcg:coordinateSource", "MIL-STD-1560_25-61_pin-front"
            )
            pin.GetPrim().SetCustomDataByKey(
                "kcg:collisionMode", "visual_only"
            )

        loose = UsdGeom.Xform.Define(stage, plan.loose_path)
        UsdGeom.Xformable(loose).AddTranslateOp().Set(
            Gf.Vec3d(0.0, 0.0, plan.initial_pair_separation_m)
        )
        loose.GetPrim().SetCustomDataByKey("kcg:role", "loose_plug")
        loose.GetPrim().SetCustomDataByKey("kcg:modelId", plan.plug_model_id)

        body = UsdGeom.Xform.Define(stage, plan.body_path)
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        author_explicit_mass_properties(
            body.GetPrim(), plan.body_mass_properties
        )
        full_plug_angles = tuple(
            360.0 * index / 96 for index in range(96)
        )
        ring_segments(
            plan.body_path + "/MatingShell",
            PLUG_MATING_SHELL_INNER_RADIUS_M,
            plan.plug_shell_outer_radius_m,
            PLUG_MATING_SHELL_LENGTH_M,
            -0.5 * PLUG_MATING_SHELL_LENGTH_M,
            full_plug_angles,
            96,
            (0.37, 0.40, 0.44),
            "plug_mating_shell",
        )
        rear_body = cylinder(
            plan.body_path + "/RearBody",
            plan.plug_rear_body_radius_m,
            0.0280,
            (0.0, 0.0, -0.0260),
            (0.29, 0.32, 0.36),
            collision_role="plug_rear_body",
        )
        rear_body.GetPrim().SetCustomDataByKey(
            "kcg:diameterSource", "MIL-DTL-38999/26G_A4_Figure1_B_shell25"
        )

        key_parent = plan.body_path + "/CollisionKeys"
        UsdGeom.Xform.Define(stage, key_parent)
        key_radial_size = (
            plan.plug_key_outer_radius_m - plan.plug_shell_outer_radius_m
        )
        key_center_radius = 0.5 * (
            plan.plug_key_outer_radius_m + plan.plug_shell_outer_radius_m
        )
        for feature in plan.keys:
            angle_rad = math.radians(feature.angle_deg)
            key = cube(
                f"{key_parent}/Key_{feature.index:02d}_{feature.name}",
                (key_radial_size, feature.key_width_m, plan.plug_key_length_m),
                (
                    key_center_radius * math.cos(angle_rad),
                    key_center_radius * math.sin(angle_rad),
                    -0.5 * plan.plug_key_length_m,
                ),
                (0.95, 0.31, 0.10),
                rotation_z_deg=feature.angle_deg,
                collision_role="polarization_key",
            )
            key.GetPrim().SetCustomDataByKey(
                "kcg:keyAngleDeg", feature.angle_deg
            )
            key.GetPrim().SetCustomDataByKey(
                "kcg:keyWidth_m", feature.key_width_m
            )
            key.GetPrim().SetCustomDataByKey(
                "kcg:engagesBeforeContact", True
            )

        plug_visuals = plan.body_path + "/ContactVisuals"
        plug_sockets = plug_visuals + "/Sockets"
        UsdGeom.Xform.Define(stage, plug_visuals)
        UsdGeom.Xform.Define(stage, plug_sockets)
        cylinder(
            plug_visuals + "/InsertFace",
            0.86 * plan.plug_shell_outer_radius_m,
            INSERT_FACE_VISUAL_THICKNESS_M,
            (0.0, 0.0, -0.5 * INSERT_FACE_VISUAL_THICKNESS_M),
            (0.76, 0.71, 0.57),
        ).GetPrim().SetCustomDataByKey("kcg:collisionMode", "visual_only")
        socket_radius_m = 0.5 * 0.049 * 25.4 * MM_TO_M
        for index, contact in enumerate(plan.contacts):
            socket = cylinder(
                f"{plug_sockets}/Socket_{index:02d}_{contact.label}",
                socket_radius_m,
                PLUG_SOCKET_VISUAL_DEPTH_M,
                (
                    contact.x_m,
                    contact.y_m,
                    plan.plug_socket_visual_front_offset_m
                    - 0.5 * PLUG_SOCKET_VISUAL_DEPTH_M,
                ),
                (0.055, 0.060, 0.067),
            )
            socket.GetPrim().SetCustomDataByKey(
                "kcg:contactLabel", contact.label
            )
            socket.GetPrim().SetCustomDataByKey(
                "kcg:coordinateSource",
                "MIL-STD-1560_pin-front-table_once_mating-transform-mirrors-view",
            )
            socket.GetPrim().SetCustomDataByKey(
                "kcg:collisionMode", "visual_only"
            )
            socket.GetPrim().SetCustomDataByKey(
                "kcg:frontPlaneOffset_m",
                plan.plug_socket_visual_front_offset_m,
            )

        nut = UsdGeom.Xform.Define(stage, plan.nut_path)
        UsdPhysics.RigidBodyAPI.Apply(nut.GetPrim())
        author_explicit_mass_properties(
            nut.GetPrim(), plan.nut_mass_properties
        )
        nut.GetPrim().SetCustomDataByKey(
            "kcg:threadCollisionMode", plan.thread_collision_mode
        )
        nut_center_z = -0.020
        nut_angles = tuple(360.0 * index / 32 for index in range(32))
        ring_segments(
            plan.nut_path + "/GripShell",
            0.0194,
            0.0240,
            0.0200,
            nut_center_z,
            nut_angles,
            32,
            (0.54, 0.57, 0.61),
            None,
        )
        # Keep the 32 facets as render geometry, but give the fingers one
        # continuous outer collision surface.  Overlapping facet colliders
        # can create a one-step impulse when a fingertip crosses a seam.
        nut_collision = cylinder(
            plan.nut_path + "/CollisionGripShell",
            0.0240,
            0.0200,
            (0.0, 0.0, nut_center_z),
            (0.54, 0.57, 0.61),
            collision_role="coupling_nut_continuous_grip",
        )
        nut_collision.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        nut_collision.GetPrim().SetCustomDataByKey(
            "kcg:collisionApproximation",
            "continuous_cylinder_for_grasp_stability",
        )

        joint = UsdPhysics.RevoluteJoint.Define(stage, plan.joint_path)
        joint.CreateAxisAttr("Z")
        joint.CreateBody0Rel().SetTargets([body.GetPrim().GetPath()])
        joint.CreateBody1Rel().SetTargets([nut.GetPrim().GetPath()])
        joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, nut_center_z))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, nut_center_z))
        joint.CreateLocalRot0Attr(Gf.Quatf(1.0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
        joint.CreateCollisionEnabledAttr(False)
        joint.GetPrim().SetCustomDataByKey(
            "kcg:threadCollisionMode", plan.thread_collision_mode
        )

        assembly_frame = UsdGeom.Xform.Define(
            stage, plan.root_path + "/connector_assembly_frame"
        )
        assembly_frame.GetPrim().SetCustomDataByKey("kcg:plusZ", "insertion")
        assembly_frame.GetPrim().SetCustomDataByKey(
            "kcg:plusX", "main_key_0deg"
        )
        assembly_frame.GetPrim().SetCustomDataByKey(
            "kcg:contactCoordinates",
            "pin_front_table_once_socket_view_opposite",
        )

        for prim in collision_prims:
            physics_utils.add_physics_material_to_prim(
                stage, prim, Sdf.Path(material_path)
            )

        stage.SetDefaultPrim(world.GetPrim())
        stage.GetRootLayer().Export(str(output_path))
        if not output_path.is_file() or output_path.stat().st_size < 1000:
            raise RuntimeError("public-spec keyed-v2 USD export failed")
        print("D38999 KEYED PUBLIC-SPEC V2 EXPORTED")
        print(f"  output: {output_path}")
        print(f"  pair model: {plan.pair_model_id}")
        print("  polarization: N (0/80/142/196/293 deg)")
        print("  contacts: 61 exact public-table positions, visual only")
        print("  plug socket fronts: 50 um ahead of opaque insert face")
        print("  mass/COM/inertia: explicit frozen r4 PhysX snapshot")
        print("  key/keyway collision: enabled before contact plane")
        print("  thread collision: unmodeled")
        print("  hardware/space qualification claim: none")
        passed = True
    except BaseException:
        traceback.print_exc()
        print("D38999 KEYED PUBLIC-SPEC V2 EXPORT FAILED", flush=True)
    finally:
        simulation_app.close(exit_code=0 if passed else 1)


if __name__ == "__main__":
    main()
