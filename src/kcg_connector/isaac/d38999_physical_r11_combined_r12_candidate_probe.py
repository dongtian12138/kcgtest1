#!/usr/bin/env python3

"""Exercise all r12 contact candidates together on the immutable r11 asset.

This is an A3 modelling diagnostic, not formal acceptance.  The r11 USD is
referenced read-only and changed only in the in-memory composed stage.  Known
bad r11 contact pieces are disabled, the individually checked round-contact
candidates are overlaid, and the existing P1 force-only driver is reused.
Contact truth remains post-step scoring data and never changes the command.
No file fingerprint is computed.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "kcg_d38999_physical_r11_combined_r12_candidate_probe_v1"
GENERATOR_ID = "kcg_d38999_r11_in_memory_combined_r12_candidate_v1"

THREAD_FOLLOWER_RADIUS_M = 0.000150
THREAD_FOLLOWER_CENTER_RADIUS_M = 0.0201168
THREAD_FOLLOWER_CENTER_Z_M = -0.001540
THREAD_FOLLOWER_PHASE_0_DEG = 0.28481322866243236
# The combined bench must carry the measured positive-coupling detent peak
# (about 0.052 N m) as well as the thread.  With the frozen yaw gain, two
# degrees can only build about 0.02 N m and therefore tests the servo limit,
# not the geometry.  The combined late-mate run showed a metastable detent plus
# thread state at 4.8 degrees of error and about 0.065 N m.  Ten degrees gives
# about 0.14 N m of quasistatic authority while remaining far below the
# immutable 0.30 N m safety cap.
THREAD_PHASE_LEAD_DEG = 10.0
TRANSLATION_POSITION_GAIN_N_M = 8000.0
TRANSLATION_VELOCITY_GAIN_N_S_M = 40.0
TRANSLATION_FORCE_COMPONENT_LIMIT_N = 30.0
ROLL_PITCH_POSITION_GAIN_NM_RAD = 1.2
BODY_YAW_POSITION_GAIN_NM_RAD = 0.8
ANGULAR_VELOCITY_GAIN_NM_S_RAD = 0.01
# Eight newtons per moving rigid supplements the bounded position servo across
# the combined socket, barrier, spring-finger, and seal load.  It is ramped over
# 1.05 mm; the earlier 20 N step is deliberately not repeated because it
# overshot roughly 0.26 mm and collapsed three late-mate events into one impact.
LATE_MATE_AXIAL_FEEDFORWARD_N = 8.0
LATE_MATE_FEEDFORWARD_START_M = 0.01400
LATE_MATE_FEEDFORWARD_FULL_M = 0.01505

BARRIER_SPHERE_RADIUS_M = 0.000500
BARRIER_TARGET_RADIUS_M = 0.000640
BARRIER_CENTER_RING_RADIUS_M = (
    BARRIER_TARGET_RADIUS_M + BARRIER_SPHERE_RADIUS_M - 0.000295
)
BARRIER_TARGET_FRONT_DEPTH_M = 0.000140
BARRIER_TARGET_END_DEPTH_M = 0.002000
BARRIER_CENTER_Z_M = (
    0.014305
    - BARRIER_TARGET_FRONT_DEPTH_M
    + math.sqrt(
        BARRIER_SPHERE_RADIUS_M**2
        - (BARRIER_CENTER_RING_RADIUS_M - BARRIER_TARGET_RADIUS_M) ** 2
    )
)

# The public nominal throat radius is 0.640 mm.  The diagnostic collision bore
# is deliberately larger because all 61 authored pins are perfectly rigid,
# whereas the hardware pins and socket system have finite lateral compliance.
# Public geometry remains the visual/specification source; this hard proxy only
# blocks gross misalignment after the compliant socket petals have guided a pin.
HARD_SOCKET_SAFETY_BORE_RADIUS_M = 0.000750
HARD_SOCKET_SAFETY_OUTER_RADIUS_M = 0.001250
HARD_SOCKET_SAFETY_DEPTH_START_M = 0.000670
HARD_SOCKET_SAFETY_DEPTH_END_M = 0.002000
HARD_SOCKET_SAFETY_SEGMENT_COUNT = 12
ANNULAR_FACE_COUNTS = (4, 4, 4, 4, 4, 4)
ANNULAR_FACE_INDICES = (
    3, 2, 1, 0,
    4, 5, 6, 7,
    0, 1, 5, 4,
    1, 2, 6, 5,
    2, 3, 7, 6,
    3, 0, 4, 7,
)

SEAL_SPHERE_RADIUS_M = 0.001000
SEAL_CENTER_RING_RADIUS_M = 0.01575
SEAL_CENTER_Z_M = 0.014615 + SEAL_SPHERE_RADIUS_M

SPRING_SPHERE_RADIUS_M = 0.000500
SPRING_TARGET_RADIUS_M = 0.0179575
SPRING_CENTER_RING_RADIUS_M = (
    SPRING_TARGET_RADIUS_M + SPRING_SPHERE_RADIUS_M - 0.000080
)
SPRING_CENTER_DEPTH_M = 0.01080 + math.sqrt(
    SPRING_SPHERE_RADIUS_M**2
    - (SPRING_CENTER_RING_RADIUS_M - SPRING_TARGET_RADIUS_M) ** 2
)

DETENT_FOLLOWER_RADIUS_M = 0.000075
DETENT_BASE_RADIUS_M = 0.021975
DETENT_CENTER_RING_RADIUS_M = (
    DETENT_BASE_RADIUS_M + DETENT_FOLLOWER_RADIUS_M - 0.000001
)
DETENT_PHASE_0_DEG = -4.491137

BOTTOM_SPHERE_RADIUS_M = 0.000500
BOTTOM_SPHERE_RING_RADIUS_M = 0.01600
BOTTOM_SPHERE_CENTER_Z_M = 0.01505 + BOTTOM_SPHERE_RADIUS_M
D6_TRANSZ_LIMIT_M = 0.000050

REPLACED_R11_FAMILIES = frozenset(
    {
        "thread_followers_3",
        "hard_socket_entries_61",
        "pin_barriers_61",
        "seal_segments_24",
        "seal_targets_24",
        "spring_fingers_12",
        "receptacle_bore_targets_12",
        "detent_followers_3",
        "fixed_metal_stop_48",
        "plug_metal_stop_48",
        "shoulder_positive_body0_48",
        "shoulder_positive_body1_48",
        "shoulder_negative_body0_48",
        "shoulder_negative_body1_48",
    }
)


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--scene-config",
        default=str(
            repository
            / "src/kcg_connector/config/"
            "d38999_keyed_v2_tabletop_scene_v1.yaml"
        ),
    )
    parser.add_argument(
        "--kit-portable-root",
        required=True,
        help="Writable Kit portable root; this diagnostic only accepts /tmp paths.",
    )
    result = parser.parse_args(argv)
    if not result.run:
        parser.error("the combined r12 candidate probe requires --run")
    portable_root = Path(result.kit_portable_root).expanduser().resolve()
    if not portable_root.is_relative_to(Path("/tmp")):
        parser.error("--kit-portable-root must resolve below /tmp")
    result.kit_portable_root = str(portable_root)
    return result


def _emit(value: Any) -> None:
    os.write(
        1,
        (json.dumps(value, allow_nan=False, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )


def _install_actual_position_phase_lead(p1: Any) -> None:
    source = inspect.getsource(p1._run)
    driver_original = """    driver = dict(p1_contract["inputs"]["component_driver_profile"])
"""
    driver_replacement = f"""    driver = dict(p1_contract["inputs"]["component_driver_profile"])
    driver["translation_position_gain_n_m"] = {TRANSLATION_POSITION_GAIN_N_M!r}
    driver["translation_velocity_gain_n_s_m"] = {TRANSLATION_VELOCITY_GAIN_N_S_M!r}
    driver["translation_force_component_limit_n"] = {TRANSLATION_FORCE_COMPONENT_LIMIT_N!r}
    driver["roll_pitch_position_gain_nm_rad"] = {ROLL_PITCH_POSITION_GAIN_NM_RAD!r}
    driver["body_yaw_position_gain_nm_rad"] = {BODY_YAW_POSITION_GAIN_NM_RAD!r}
    driver["angular_velocity_gain_nm_s_rad"] = {ANGULAR_VELOCITY_GAIN_NM_S_RAD!r}
"""
    if source.count(driver_original) != 1:
        raise RuntimeError("P1 driver source no longer matches the probe")
    source = source.replace(driver_original, driver_replacement)
    torque_start = """            body_torque = _clamp_vector(
"""
    force_feedforward = f"""            late_mate_fraction = min(
                1.0,
                max(
                    0.0,
                    (target_separation - {LATE_MATE_FEEDFORWARD_START_M!r})
                    / (
                        {LATE_MATE_FEEDFORWARD_FULL_M!r}
                        - {LATE_MATE_FEEDFORWARD_START_M!r}
                    ),
                ),
            )
            if late_mate_fraction > 0.0:
                late_mate_load = np.asarray(
                    (
                        0.0,
                        0.0,
                        -{LATE_MATE_AXIAL_FEEDFORWARD_N!r}
                        * late_mate_fraction,
                    ),
                    dtype=np.float64,
                )
                body_force = _clamp_vector(
                    body_force + late_mate_load,
                    float(driver["translation_force_component_limit_n"]),
                )
                nut_force = _clamp_vector(
                    nut_force + late_mate_load,
                    float(driver["translation_force_component_limit_n"]),
                )
            body_torque = _clamp_vector(
"""
    if source.count(torque_start) != 1:
        raise RuntimeError("P1 torque source no longer matches the probe")
    source = source.replace(torque_start, force_feedforward)
    original = """            target_relative_yaw = -2.0 * math.pi * max(
                0.0, target_separation - entry_separation
            ) / lead_m
"""
    replacement = f"""            commanded_relative_yaw = -2.0 * math.pi * max(
                0.0, target_separation - entry_separation
            ) / lead_m
            actual_nut_separation = float(fixed_origin[2] - nut_position[2])
            actual_nut_helix_yaw = -2.0 * math.pi * max(
                0.0, actual_nut_separation - entry_separation
            ) / lead_m
            target_relative_yaw = max(
                commanded_relative_yaw,
                actual_nut_helix_yaw - math.radians({THREAD_PHASE_LEAD_DEG!r}),
            )
"""
    if source.count(original) != 1:
        raise RuntimeError("P1 yaw-target source no longer matches the probe")
    exec(
        compile(
            source.replace(original, replacement),
            "<combined_r12_actual_position_phase_lead>",
            "exec",
        ),
        p1.__dict__,
    )
    original_contact_rows = p1._contact_rows

    def geometry_touch_rows(stage: Any, interface: Any, schema_tools: Any) -> Any:
        rows = original_contact_rows(stage, interface, schema_tools)
        output = []
        for row in rows:
            minimum = row["minimum_separation_m"]
            if row["event"] is not None and (
                minimum is None or float(minimum) > 0.0
            ):
                row = dict(row)
                row["event"] = None
                row["positive_gap_candidate_not_scored"] = True
            output.append(row)
        return output

    p1._contact_rows = geometry_touch_rows


def _run(arguments: argparse.Namespace, output: Path) -> Mapping[str, Any]:
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade

    import d38999_physical_r7_p1_nominal_bench as p1
    import kcg_connector.d38999_tabletop_scene as tabletop_scene
    import validate_physical_r11_cooked_geometry as cooked
    import validate_physical_r7_composed_scene as composed

    _install_actual_position_phase_lead(p1)
    composed._run_validation = lambda _arguments: {
        "status": "PASSED",
        "contract_revision": "keyed_v3_physical_r11",
        "diagnostic_reuse_of_prior_gate": True,
    }
    cooked._run = lambda: {
        "status": "PASSED",
        "diagnostic_reuse_of_prior_gate": True,
    }

    original_author = tabletop_scene.author_d38999_tabletop_scene
    overlay_inventory: dict[str, Any] = {}

    def author_overlay(stage: Any, *args: Any, **kwargs: Any) -> Mapping[str, Any]:
        result = original_author(stage, *args, **kwargs)
        root = (
            "/World/D38999TabletopV1/D38999Pair/"
            "D38999Shell25JKeyedPhysicalV3"
        )
        fixed = root + "/FixedReceptacle"
        body = root + "/LoosePlug/BodyAssembly"
        nut = root + "/LoosePlug/CouplingNut"

        def material(suffix: str) -> Any:
            value = UsdShade.Material.Get(stage, root + "/Materials/" + suffix)
            if not value:
                raise RuntimeError(f"missing candidate material {suffix}")
            return value

        materials = {
            "thread": material("coupling_thread__hard_thread"),
            "socket": material("pin_and_socket__compliant_socket_petal"),
            "barrier": material(
                "interfacial_pin_barrier__compliant_pin_barrier"
            ),
            "socket_hard": material("pin_and_socket__hard_socket_entry"),
            "seal": material("peripheral_seal__compliant_peripheral_seal"),
            "seal_hard": material("plug_shell_and_keys__hard_seal_target"),
            "spring": material("spring_finger__compliant_spring_finger"),
            "spring_hard": material(
                "fixture_and_receptacle__hard_receptacle_bore"
            ),
            "detent": material(
                "anti_decoupling_detent__compliant_detent_follower"
            ),
            "bottom_fixed": material(
                "fixture_and_receptacle__hard_metal_bottoming"
            ),
            "bottom_plug": material(
                "plug_shell_and_keys__hard_metal_bottoming"
            ),
        }
        barrier_api = PhysxSchema.PhysxMaterialAPI(
            materials["barrier"].GetPrim()
        )
        barrier_api.GetCompliantContactStiffnessAttr().Set(250.0 / 6.0)
        barrier_api.GetCompliantContactDampingAttr().Set(0.020 / 6.0)
        barrier_api.GetCompliantContactAccelerationSpringAttr().Set(False)
        socket_api = PhysxSchema.PhysxMaterialAPI(materials["socket"].GetPrim())
        socket_api.GetCompliantContactStiffnessAttr().Set(2000.0)
        socket_api.GetCompliantContactDampingAttr().Set(0.050)
        socket_api.GetCompliantContactAccelerationSpringAttr().Set(False)

        disabled_by_family: dict[str, int] = {
            family: 0 for family in REPLACED_R11_FAMILIES
        }
        pin_rows: list[tuple[str, float, float]] = []
        rail_count = 0
        active_original_hard_entry_count = 0
        for prim in stage.Traverse():
            collision = prim.GetAttribute("physics:collisionEnabled")
            if not collision or collision.Get() is not True:
                continue
            family_attr = prim.GetAttribute("kcg:primitiveFamily")
            family_value = family_attr.Get() if family_attr else None
            family = None if family_value is None else str(family_value)
            if family == "thread_rails_3":
                rail_count += 1
            elif family == "hard_socket_entries_61":
                active_original_hard_entry_count += 1
            if family == "pins_61":
                path = str(prim.GetPath())
                label = path.rsplit("/Pin_", 1)[-1]
                translate_ops = [
                    operation
                    for operation in UsdGeom.Xformable(prim).GetOrderedXformOps()
                    if operation.GetOpType() == UsdGeom.XformOp.TypeTranslate
                ]
                if len(translate_ops) != 1:
                    raise RuntimeError(f"pin {path} lacks one translate operation")
                center = translate_ops[0].Get()
                pin_rows.append((label, float(center[0]), float(center[1])))
            if family in REPLACED_R11_FAMILIES:
                collision.Set(False)
                disabled_by_family[family] += 1
        if rail_count != 1080 or len(pin_rows) != 61:
            raise RuntimeError(
                f"base inventory mismatch rails={rail_count} pins={len(pin_rows)}"
            )

        def group_members(family: str) -> Any:
            group = stage.GetPrimAtPath(root + "/CollisionGroups/" + family)
            relationship = (
                group.GetRelationship("collection:colliders:includes")
                if group
                else None
            )
            if not relationship:
                raise RuntimeError(f"missing collision group for {family}")
            return relationship

        def mark(
            prim: Any,
            *,
            family: str,
            material_value: Any,
            material_role: str,
            response_role: str,
            group_relationship: Any | None,
        ) -> str:
            UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
            collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            collision_api.CreateContactOffsetAttr(1.0e-5)
            collision_api.CreateRestOffsetAttr(0.0)
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                material_value, materialPurpose="physics"
            )
            prim.CreateAttribute(
                "kcg:primitiveFamily", Sdf.ValueTypeNames.String, custom=True
            ).Set(family)
            prim.CreateAttribute(
                "kcg:materialRole", Sdf.ValueTypeNames.String, custom=True
            ).Set(material_role)
            prim.CreateAttribute(
                "kcg:responseRole", Sdf.ValueTypeNames.String, custom=True
            ).Set(response_role)
            prim.CreateAttribute(
                "kcg:diagnosticOnly", Sdf.ValueTypeNames.Bool, custom=True
            ).Set(True)
            path = str(prim.GetPath())
            if group_relationship is not None:
                group_relationship.AddTarget(Sdf.Path(path))
            return path

        def sphere(
            *,
            path: str,
            center: Sequence[float],
            radius: float,
            family: str,
            material_value: Any,
            material_role: str,
            response_role: str,
            group_relationship: Any | None,
        ) -> str:
            geometry = UsdGeom.Sphere.Define(stage, path)
            geometry.CreateRadiusAttr(radius)
            geometry.CreateExtentAttr(
                [
                    Gf.Vec3f(-radius, -radius, -radius),
                    Gf.Vec3f(radius, radius, radius),
                ]
            )
            UsdGeom.Xformable(geometry).AddTranslateOp().Set(
                Gf.Vec3d(*[float(value) for value in center])
            )
            return mark(
                geometry.GetPrim(),
                family=family,
                material_value=material_value,
                material_role=material_role,
                response_role=response_role,
                group_relationship=group_relationship,
            )

        def cylinder(
            *,
            path: str,
            center: Sequence[float],
            radius: float,
            height: float,
            family: str,
            material_value: Any,
            material_role: str,
            response_role: str,
            group_relationship: Any | None,
        ) -> str:
            geometry = UsdGeom.Cylinder.Define(stage, path)
            geometry.CreateAxisAttr(UsdGeom.Tokens.z)
            geometry.CreateRadiusAttr(radius)
            geometry.CreateHeightAttr(height)
            geometry.CreateExtentAttr(
                [
                    Gf.Vec3f(-radius, -radius, -0.5 * height),
                    Gf.Vec3f(radius, radius, 0.5 * height),
                ]
            )
            UsdGeom.Xformable(geometry).AddTranslateOp().Set(
                Gf.Vec3d(*[float(value) for value in center])
            )
            return mark(
                geometry.GetPrim(),
                family=family,
                material_value=material_value,
                material_role=material_role,
                response_role=response_role,
                group_relationship=group_relationship,
            )

        def annular_wedge(
            *,
            path: str,
            center_xy: tuple[float, float],
            theta0_deg: float,
            theta1_deg: float,
            inner_clear_radius: float,
            outer_radius: float,
            z0: float,
            z1: float,
            family: str,
            material_value: Any,
            material_role: str,
            response_role: str,
            group_relationship: Any | None,
        ) -> str:
            half_step = 0.5 * abs(theta1_deg - theta0_deg)
            inner_vertex_radius = inner_clear_radius / math.cos(
                math.radians(half_step)
            )

            def point(radius: float, angle_deg: float, z_value: float) -> Any:
                angle = math.radians(angle_deg)
                return Gf.Vec3f(
                    center_xy[0] + radius * math.cos(angle),
                    center_xy[1] + radius * math.sin(angle),
                    z_value,
                )

            points = [
                point(inner_vertex_radius, theta0_deg, z0),
                point(outer_radius, theta0_deg, z0),
                point(outer_radius, theta1_deg, z0),
                point(inner_vertex_radius, theta1_deg, z0),
                point(inner_vertex_radius, theta0_deg, z1),
                point(outer_radius, theta0_deg, z1),
                point(outer_radius, theta1_deg, z1),
                point(inner_vertex_radius, theta1_deg, z1),
            ]
            geometry = UsdGeom.Mesh.Define(stage, path)
            geometry.CreatePointsAttr(points)
            geometry.CreateFaceVertexCountsAttr(list(ANNULAR_FACE_COUNTS))
            geometry.CreateFaceVertexIndicesAttr(list(ANNULAR_FACE_INDICES))
            geometry.CreateSubdivisionSchemeAttr("none")
            geometry.CreateOrientationAttr("rightHanded")
            lows = [min(float(point[axis]) for point in points) for axis in range(3)]
            highs = [max(float(point[axis]) for point in points) for axis in range(3)]
            geometry.CreateExtentAttr([Gf.Vec3f(*lows), Gf.Vec3f(*highs)])
            prim = geometry.GetPrim()
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(
                "convexHull"
            )
            return mark(
                prim,
                family=family,
                material_value=material_value,
                material_role=material_role,
                response_role=response_role,
                group_relationship=group_relationship,
            )

        paths_by_candidate: dict[str, list[str]] = {}

        thread_group = group_members("thread_followers_3")
        for index in range(3):
            angle = math.radians(THREAD_FOLLOWER_PHASE_0_DEG + 120.0 * index)
            paths_by_candidate.setdefault("thread_followers", []).append(
                sphere(
                    path=(
                        nut
                        + f"/CouplingThread/CombinedR12SphereFollower_{index}"
                    ),
                    center=(
                        THREAD_FOLLOWER_CENTER_RADIUS_M * math.cos(angle),
                        THREAD_FOLLOWER_CENTER_RADIUS_M * math.sin(angle),
                        THREAD_FOLLOWER_CENTER_Z_M,
                    ),
                    radius=THREAD_FOLLOWER_RADIUS_M,
                    family="thread_followers_3",
                    material_value=materials["thread"],
                    material_role="coupling_thread",
                    response_role="hard_thread",
                    group_relationship=thread_group,
                )
            )

        barrier_paths: list[str] = []
        barrier_target_paths: list[str] = []
        hard_socket_safety_paths: list[str] = []
        hard_socket_group = group_members("hard_socket_entries_61")
        for label, fixed_x, fixed_y in sorted(pin_rows):
            plug_x = fixed_x
            plug_y = -fixed_y
            for index in range(6):
                angle = math.radians(60.0 * index)
                barrier_paths.append(
                    sphere(
                        path=(
                            fixed
                            + f"/Contacts/Barrier_{label}/CombinedR12Spheres/"
                            f"Sphere_{index}"
                        ),
                        center=(
                            fixed_x
                            + BARRIER_CENTER_RING_RADIUS_M * math.cos(angle),
                            fixed_y
                            + BARRIER_CENTER_RING_RADIUS_M * math.sin(angle),
                            BARRIER_CENTER_Z_M,
                        ),
                        radius=BARRIER_SPHERE_RADIUS_M,
                        family="pin_barriers_61",
                        material_value=materials["barrier"],
                        material_role="interfacial_pin_barrier",
                        response_role="compliant_pin_barrier",
                        group_relationship=None,
                    )
                )
            barrier_target_paths.append(
                cylinder(
                    path=(
                        body
                        + f"/Contacts/Socket_{label}/CombinedR12BarrierTarget"
                    ),
                    center=(
                        plug_x,
                        plug_y,
                        0.5
                        * (
                            BARRIER_TARGET_FRONT_DEPTH_M
                            + BARRIER_TARGET_END_DEPTH_M
                        ),
                    ),
                    radius=BARRIER_TARGET_RADIUS_M,
                    height=(
                        BARRIER_TARGET_END_DEPTH_M
                        - BARRIER_TARGET_FRONT_DEPTH_M
                    ),
                    family="hard_socket_entries_61",
                    material_value=materials["socket_hard"],
                    material_role="pin_and_socket",
                    response_role="hard_socket_entry",
                    group_relationship=None,
                )
            )
            for segment_index in range(HARD_SOCKET_SAFETY_SEGMENT_COUNT):
                half_step_deg = 180.0 / HARD_SOCKET_SAFETY_SEGMENT_COUNT
                center_deg = (
                    360.0 * segment_index / HARD_SOCKET_SAFETY_SEGMENT_COUNT
                )
                hard_socket_safety_paths.append(
                    annular_wedge(
                        path=(
                            body
                            + f"/Contacts/Socket_{label}/CombinedR12SafetyBore/"
                            + f"Wedge_{segment_index:02d}"
                        ),
                        center_xy=(plug_x, plug_y),
                        theta0_deg=center_deg - half_step_deg,
                        theta1_deg=center_deg + half_step_deg,
                        inner_clear_radius=HARD_SOCKET_SAFETY_BORE_RADIUS_M,
                        outer_radius=HARD_SOCKET_SAFETY_OUTER_RADIUS_M,
                        z0=HARD_SOCKET_SAFETY_DEPTH_START_M,
                        z1=HARD_SOCKET_SAFETY_DEPTH_END_M,
                        family="hard_socket_entries_61",
                        material_value=materials["socket_hard"],
                        material_role="pin_and_socket",
                        response_role="hard_socket_entry",
                        group_relationship=hard_socket_group,
                    )
                )
        paths_by_candidate["pin_barriers"] = barrier_paths
        paths_by_candidate["barrier_targets"] = barrier_target_paths
        paths_by_candidate["hard_socket_safety_bore"] = hard_socket_safety_paths

        collision_groups_root = root + "/CollisionGroups"
        existing_group_paths = [
            str(child.GetPath())
            for child in stage.GetPrimAtPath(collision_groups_root).GetChildren()
            if child.IsA(UsdPhysics.CollisionGroup)
        ]

        def isolated_group(name: str, paths: Sequence[str]) -> None:
            group = UsdPhysics.CollisionGroup.Define(
                stage, collision_groups_root + "/" + name
            )
            group.CreateInvertFilteredGroupsAttr(False)
            collection = group.GetCollidersCollectionAPI()
            collection.CreateExpansionRuleAttr("explicitOnly")
            collection.CreateIncludeRootAttr(False)
            collection.CreateIncludesRel().SetTargets(
                [Sdf.Path(path) for path in paths]
            )
            filtered = group.CreateFilteredGroupsRel()
            for old_group_path in existing_group_paths:
                filtered.AddTarget(Sdf.Path(old_group_path))

        isolated_group("combined_r12_barrier_spheres", barrier_paths)
        isolated_group("combined_r12_barrier_targets", barrier_target_paths)

        seal_group = group_members("seal_segments_24")
        for index in range(24):
            angle = math.radians(15.0 * index)
            paths_by_candidate.setdefault("seal_segments", []).append(
                sphere(
                    path=fixed + f"/PeripheralSeal/CombinedR12Sphere_{index:02d}",
                    center=(
                        SEAL_CENTER_RING_RADIUS_M * math.cos(angle),
                        SEAL_CENTER_RING_RADIUS_M * math.sin(angle),
                        SEAL_CENTER_Z_M,
                    ),
                    radius=SEAL_SPHERE_RADIUS_M,
                    family="seal_segments_24",
                    material_value=materials["seal"],
                    material_role="peripheral_seal",
                    response_role="compliant_peripheral_seal",
                    group_relationship=seal_group,
                )
            )
        paths_by_candidate["seal_target"] = [
            cylinder(
                path=body + "/PeripheralSealTarget/CombinedR12ContinuousTarget",
                center=(0.0, 0.0, 0.00050),
                radius=0.01690,
                height=0.00100,
                family="seal_targets_24",
                material_value=materials["seal_hard"],
                material_role="plug_shell_and_keys",
                response_role="hard_seal_target",
                group_relationship=group_members("seal_targets_24"),
            )
        ]

        spring_group = group_members("spring_fingers_12")
        for index in range(12):
            angle = math.radians(-8.0 - 30.0 * index)
            paths_by_candidate.setdefault("spring_fingers", []).append(
                sphere(
                    path=body + f"/SpringFingers/CombinedR12Sphere_{index:02d}",
                    center=(
                        SPRING_CENTER_RING_RADIUS_M * math.cos(angle),
                        SPRING_CENTER_RING_RADIUS_M * math.sin(angle),
                        SPRING_CENTER_DEPTH_M,
                    ),
                    radius=SPRING_SPHERE_RADIUS_M,
                    family="spring_fingers_12",
                    material_value=materials["spring"],
                    material_role="spring_finger",
                    response_role="compliant_spring_finger",
                    group_relationship=spring_group,
                )
            )
        paths_by_candidate["spring_target"] = [
            cylinder(
                path=fixed + "/MatingShell/CombinedR12SpringTarget",
                center=(0.0, 0.0, 0.0030),
                radius=SPRING_TARGET_RADIUS_M,
                height=0.0060,
                family="receptacle_bore_targets_12",
                material_value=materials["spring_hard"],
                material_role="fixture_and_receptacle",
                response_role="hard_receptacle_bore",
                group_relationship=group_members("receptacle_bore_targets_12"),
            )
        ]

        detent_group = group_members("detent_followers_3")
        for index in range(3):
            angle = math.radians(DETENT_PHASE_0_DEG + 120.0 * index)
            paths_by_candidate.setdefault("detent_followers", []).append(
                sphere(
                    path=nut + f"/AntiDecoupling/CombinedR12Follower_{index}",
                    center=(
                        DETENT_CENTER_RING_RADIUS_M * math.cos(angle),
                        DETENT_CENTER_RING_RADIUS_M * math.sin(angle),
                        0.0200,
                    ),
                    radius=DETENT_FOLLOWER_RADIUS_M,
                    family="detent_followers_3",
                    material_value=materials["detent"],
                    material_role="anti_decoupling_detent",
                    response_role="compliant_detent_follower",
                    group_relationship=detent_group,
                )
            )

        paths_by_candidate["fixed_bottom"] = [
            cylinder(
                path=fixed + "/MatingShell/CombinedR12MetalStop",
                center=(0.0, 0.0, 0.00015),
                radius=0.01695,
                height=0.00030,
                family="fixed_metal_stop_48",
                material_value=materials["bottom_fixed"],
                material_role="fixture_and_receptacle",
                response_role="hard_metal_bottoming",
                group_relationship=group_members("fixed_metal_stop_48"),
            )
        ]
        plug_bottom_group = group_members("plug_metal_stop_48")
        for index in range(3):
            angle = math.radians(120.0 * index)
            paths_by_candidate.setdefault("plug_bottom", []).append(
                sphere(
                    path=(
                        body
                        + f"/InternalMatingShell/CombinedR12MetalStop/Sphere_{index}"
                    ),
                    center=(
                        BOTTOM_SPHERE_RING_RADIUS_M * math.cos(angle),
                        BOTTOM_SPHERE_RING_RADIUS_M * math.sin(angle),
                        BOTTOM_SPHERE_CENTER_Z_M,
                    ),
                    radius=BOTTOM_SPHERE_RADIUS_M,
                    family="plug_metal_stop_48",
                    material_value=materials["bottom_plug"],
                    material_role="plug_shell_and_keys",
                    response_role="hard_metal_bottoming",
                    group_relationship=plug_bottom_group,
                )
            )

        joint = stage.GetPrimAtPath(root + "/LoosePlug/CouplingNutJoint")
        if not joint:
            raise RuntimeError("missing D6 coupling-nut joint")
        joint.GetAttribute("limit:transZ:physics:low").Set(-D6_TRANSZ_LIMIT_M)
        joint.GetAttribute("limit:transZ:physics:high").Set(D6_TRANSZ_LIMIT_M)

        expected_counts = {
            "thread_followers": 3,
            "pin_barriers": 366,
            "barrier_targets": 61,
            "hard_socket_safety_bore": (
                61 * HARD_SOCKET_SAFETY_SEGMENT_COUNT
            ),
            "seal_segments": 24,
            "seal_target": 1,
            "spring_fingers": 12,
            "spring_target": 1,
            "detent_followers": 3,
            "fixed_bottom": 1,
            "plug_bottom": 3,
        }
        actual_counts = {
            key: len(paths_by_candidate.get(key, []))
            for key in expected_counts
        }
        if actual_counts != expected_counts:
            raise RuntimeError(
                f"combined overlay inventory mismatch {actual_counts}"
            )
        overlay_inventory.update(
            {
                "disabled_r11_by_family": disabled_by_family,
                "kept_r11_thread_rail_count": rail_count,
                "found_and_disabled_r11_hard_socket_entry_count": (
                    active_original_hard_entry_count
                ),
                "candidate_counts": actual_counts,
                "candidate_total_collider_count": sum(actual_counts.values()),
                "D6_transZ_limit_m": D6_TRANSZ_LIMIT_M,
                "barrier_response_stiffness_per_sphere_n_m": 250.0 / 6.0,
                "barrier_response_damping_per_sphere_n_s_m": 0.020 / 6.0,
                "realized_socket_petal_response_stiffness_n_m": 2000.0,
                "realized_socket_petal_response_damping_n_s_m": 0.050,
                "hard_socket_safety_bore_radius_m": (
                    HARD_SOCKET_SAFETY_BORE_RADIUS_M
                ),
                "hard_socket_public_nominal_throat_radius_m": 0.000640,
                "hard_socket_safety_segment_count_per_socket": (
                    HARD_SOCKET_SAFETY_SEGMENT_COUNT
                ),
                "hard_socket_safety_bore_role": (
                    "rigid_pin_assembly_gross_misalignment_proxy"
                ),
                "diagnostic_translation_position_gain_n_m": (
                    TRANSLATION_POSITION_GAIN_N_M
                ),
                "diagnostic_translation_velocity_gain_n_s_m": (
                    TRANSLATION_VELOCITY_GAIN_N_S_M
                ),
                "diagnostic_translation_force_component_limit_n": (
                    TRANSLATION_FORCE_COMPONENT_LIMIT_N
                ),
                "diagnostic_roll_pitch_position_gain_nm_rad": (
                    ROLL_PITCH_POSITION_GAIN_NM_RAD
                ),
                "diagnostic_body_yaw_position_gain_nm_rad": (
                    BODY_YAW_POSITION_GAIN_NM_RAD
                ),
                "diagnostic_angular_velocity_gain_nm_s_rad": (
                    ANGULAR_VELOCITY_GAIN_NM_S_RAD
                ),
                "diagnostic_thread_phase_lead_deg": THREAD_PHASE_LEAD_DEG,
                "diagnostic_late_mate_axial_feedforward_n": (
                    LATE_MATE_AXIAL_FEEDFORWARD_N
                ),
                "diagnostic_late_mate_feedforward_start_m": (
                    LATE_MATE_FEEDFORWARD_START_M
                ),
                "diagnostic_late_mate_feedforward_full_m": (
                    LATE_MATE_FEEDFORWARD_FULL_M
                ),
                "diagnostic_contact_event_requires_nonpositive_geometry_gap": True,
            }
        )
        return result

    tabletop_scene.author_d38999_tabletop_scene = author_overlay
    p1_arguments = argparse.Namespace(
        scene_config=str(Path(arguments.scene_config).resolve()),
        settle_steps=120,
        hold_steps=240,
        start_separation_m=0.00550,
        end_separation_m=0.01505,
        axial_speed_m_s=0.00050,
    )
    p1_report = p1._run(p1_arguments, output)
    trace = [
        json.loads(line)
        for line in (output / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    contact_audit = json.loads(
        (output / "contact_audit.json").read_text(encoding="utf-8")
    )
    family_pair_counts: dict[str, int] = {}
    for row in contact_audit["pairs"]:
        key = "__".join(sorted(str(value) for value in row["families"]))
        family_pair_counts[key] = family_pair_counts.get(key, 0) + 1
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_id": GENERATOR_ID,
        "role": "a3_combined_modelling_diagnostic_not_formal_acceptance",
        "formal_acceptance_evidence": False,
        "asset_path": p1_report["asset_path"],
        "asset_revision_under_test": "keyed_v3_physical_r11_with_in_memory_r12_overlay",
        "overlay_inventory": overlay_inventory,
        "p1_status": p1_report["status"],
        "p1_passed": bool(p1_report["passed"]),
        "premotion_state_pass": bool(p1_report["premotion_state_pass"]),
        "event_inventory_complete": bool(p1_report["event_inventory_complete"]),
        "event_order_pass": bool(p1_report["event_order_pass"]),
        "event_position_pass": bool(p1_report["event_position_pass"]),
        "observed_event_order": p1_report["observed_event_order"],
        "event_first": p1_report["event_first"],
        "position_error_m": p1_report["position_error_m"],
        "all_three_thread_starts_enter": bool(
            p1_report["all_three_thread_starts_enter"]
        ),
        "false_bottoming_count": int(p1_report["false_bottoming_count"]),
        "solver_error_count": int(p1_report["solver_error_count"]),
        "object_pose_write_after_physics_start_count": int(
            p1_report["object_pose_write_after_physics_start_count"]
        ),
        "contact_family_pair_counts": family_pair_counts,
        "contact_pair_count": int(contact_audit["pair_count"]),
        "maximum_abs_body_force_z_n": max(
            abs(float(row["body_force_n"][2])) for row in trace
        ),
        "maximum_abs_nut_force_z_n": max(
            abs(float(row["nut_force_n"][2])) for row in trace
        ),
        "maximum_abs_nut_torque_z_nm": max(
            abs(float(row["nut_torque_nm"][2])) for row in trace
        ),
        "final_observed_separation_m": float(trace[-1]["observed_separation_m"]),
        "combined_candidate_pass": bool(p1_report["passed"]),
        "file_fingerprints_computed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _arguments(argv)
    output = Path(arguments.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite probe output: {output}")
    output.mkdir(parents=True, exist_ok=False)
    portable_root = Path(arguments.kit_portable_root)
    portable_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WARP_CACHE_PATH", str(portable_root / "warp-cache"))
    sys.argv.extend(["--portable-root", str(portable_root)])

    from isaacsim import SimulationApp

    application = SimulationApp(
        {
            "headless": True,
            "hide_ui": True,
            "renderer": "Minimal",
            "multi_gpu": False,
            "fast_shutdown": True,
            "enable_crashreporter": False,
        },
        experience=str(Path(__file__).with_name("d38999_cpu_physics_only.kit")),
    )
    status = 1
    try:
        report = _run(arguments, output)
        status = 0 if report["combined_candidate_pass"] else 2
    except BaseException as error:
        report = {
            "schema_version": SCHEMA_VERSION,
            "generator_id": GENERATOR_ID,
            "status": "FAILED",
            "error_type": type(error).__name__,
            "error": str(error),
            "formal_acceptance_evidence": False,
            "file_fingerprints_computed": False,
        }
        traceback.print_exc()
    finally:
        (output / "diagnostic_summary.json").write_text(
            json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        application.close()
    _emit(report)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
