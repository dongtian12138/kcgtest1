#!/usr/bin/env python3

"""Statically validate the imported KUKA/three-finger-hand USD asset."""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any


KUKA_ACTIVE_JOINTS = tuple(f"iiwa_joint_{index}" for index in range(1, 8))
HAND_ACTIVE_JOINTS = ("f1j1", "f1j2", "f2j1", "f3j2")
HAND_MIMIC_JOINTS = {
    "f1j3": "f1j2",
    "f2j2": "f2j1",
    "f3j1": "f1j1",
    "f3j3": "f3j2",
}

EXPECTED_ENDPOINTS = {
    **{
        f"iiwa_joint_{index}": (f"iiwa_link_{index - 1}", f"iiwa_link_{index}")
        for index in range(1, 8)
    },
    "f1j1": ("handbase_link", "f1Link1"),
    "f1j2": ("f1Link1", "f1Link2"),
    "f1j3": ("f1Link2", "f1Link3"),
    "f2j1": ("handbase_link", "f2Link1"),
    "f2j2": ("f2Link1", "f2Link2"),
    "f3j1": ("handbase_link", "f3Link1"),
    "f3j2": ("f3Link1", "f3Link2"),
    "f3j3": ("f3Link2", "f3Link3"),
}

EXPECTED_RIGID_LINKS = (
    *(f"iiwa_link_{index}" for index in range(8)),
    "handbase_link",
    "f1Link1",
    "f1Link2",
    "f1Link3",
    "f2Link1",
    "f2Link2",
    "f3Link1",
    "f3Link2",
    "f3Link3",
)

PASS_BANNER = "ISAAC ROBOT ASSET TOPOLOGY PASSED"
FAIL_BANNER = "ISAAC ROBOT ASSET TOPOLOGY FAILED"
PHYSICAL_PASS_BANNER = "ISAAC ROBOT PHYSICAL R7 ASSET PASSED"
PHYSICAL_FAIL_BANNER = "ISAAC ROBOT PHYSICAL R7 ASSET FAILED"


class AssetValidationError(RuntimeError):
    """Carry all independently provable topology failures to the JSON report."""

    def __init__(self, errors: list[str], metrics: dict[str, Any]):
        super().__init__("; ".join(errors))
        self.errors = errors
        self.metrics = metrics


def _default_asset_path() -> Path:
    repository_root = Path(__file__).resolve().parents[3]
    return repository_root / "artifacts/kcg_connector/isaac/robot/handarm/handarm.usda"


def _relationship_targets(prim: Any, relationship_name: str) -> list[str]:
    relationship = prim.GetRelationship(relationship_name)
    if not relationship:
        return []
    return [str(path) for path in relationship.GetTargets()]


def _dependency_text(value: Any) -> str:
    path = getattr(value, "path", None)
    return str(path if path is not None else value)


def _require_single_named_prim(
    prims_by_name: dict[str, list[Any]], name: str, errors: list[str]
) -> Any | None:
    matches = prims_by_name.get(name, [])
    if len(matches) != 1:
        errors.append(
            f"expected exactly one prim named {name!r}, found "
            f"{[str(prim.GetPath()) for prim in matches]}"
        )
        return None
    return matches[0]


def _inspect_revolute_joint(
    stage: Any,
    usd_physics: Any,
    physics_path: Any,
    name: str,
    expected_mimic: str | None,
    errors: list[str],
) -> dict[str, Any]:
    joint_path = physics_path.AppendChild(name)
    prim = stage.GetPrimAtPath(joint_path)
    record: dict[str, Any] = {
        "path": str(joint_path),
        "body0": [],
        "body1": [],
        "axis": None,
        "angular_drive": False,
        "mimic_target": [],
    }

    if not prim or not prim.IsA(usd_physics.RevoluteJoint):
        errors.append(f"missing PhysicsRevoluteJoint: {joint_path}")
        return record

    applied_schemas = [str(schema) for schema in prim.GetAppliedSchemas()]
    body0 = _relationship_targets(prim, "physics:body0")
    body1 = _relationship_targets(prim, "physics:body1")
    mimic_targets = _relationship_targets(prim, "newton:mimicJoint")
    axis_value = usd_physics.RevoluteJoint(prim).GetAxisAttr().Get()
    has_drive = "PhysicsDriveAPI:angular" in applied_schemas
    has_mimic_api = "NewtonMimicAPI" in applied_schemas

    record.update(
        {
            "applied_schemas": applied_schemas,
            "axis": str(axis_value) if axis_value is not None else None,
            "body0": body0,
            "body1": body1,
            "angular_drive": has_drive,
            "mimic_target": mimic_targets,
        }
    )

    expected_body0, expected_body1 = EXPECTED_ENDPOINTS[name]
    for label, targets, expected_name in (
        ("body0", body0, expected_body0),
        ("body1", body1, expected_body1),
    ):
        if len(targets) != 1:
            errors.append(f"{name} {label} must have one target, found {targets}")
            continue
        target_prim = stage.GetPrimAtPath(targets[0])
        if not target_prim:
            errors.append(f"{name} {label} target does not resolve: {targets[0]}")
            continue
        if target_prim.GetName() != expected_name:
            errors.append(
                f"{name} {label} expected {expected_name!r}, got "
                f"{target_prim.GetName()!r} ({targets[0]})"
            )
        if not target_prim.HasAPI(usd_physics.RigidBodyAPI):
            errors.append(f"{name} {label} is not a rigid-body prim: {targets[0]}")

    if str(axis_value) != "Z":
        errors.append(f"{name} expected Z axis, got {axis_value!r}")
    if not has_drive:
        errors.append(f"{name} lost PhysicsDriveAPI:angular")

    if expected_mimic is None:
        if has_mimic_api or mimic_targets:
            errors.append(
                f"active joint {name} unexpectedly has mimic metadata: "
                f"api={has_mimic_api}, targets={mimic_targets}"
            )
    else:
        expected_target = str(physics_path.AppendChild(expected_mimic))
        if not has_mimic_api:
            errors.append(f"mimic joint {name} lost NewtonMimicAPI")
        if mimic_targets != [expected_target]:
            errors.append(
                f"mimic joint {name} expected target {expected_target}, "
                f"got {mimic_targets}"
            )

    return record


def _inspect_fixed_joint(
    stage: Any,
    usd_physics: Any,
    physics_path: Any,
    name: str,
    expected_body0: str,
    expected_body1: str,
    errors: list[str],
) -> dict[str, Any]:
    joint_path = physics_path.AppendChild(name)
    prim = stage.GetPrimAtPath(joint_path)
    record = {"path": str(joint_path), "body0": [], "body1": []}
    if not prim or not prim.IsA(usd_physics.FixedJoint):
        errors.append(f"missing PhysicsFixedJoint: {joint_path}")
        return record

    body0 = _relationship_targets(prim, "physics:body0")
    body1 = _relationship_targets(prim, "physics:body1")
    record.update({"body0": body0, "body1": body1})
    for label, targets, expected_name in (
        ("body0", body0, expected_body0),
        ("body1", body1, expected_body1),
    ):
        if len(targets) != 1:
            errors.append(f"{name} {label} must have one target, found {targets}")
            continue
        target = stage.GetPrimAtPath(targets[0])
        if not target:
            errors.append(f"{name} {label} target does not resolve: {targets[0]}")
        elif target.GetName() != expected_name:
            errors.append(
                f"{name} {label} expected {expected_name!r}, got "
                f"{target.GetName()!r} ({targets[0]})"
            )
    return record


def _value_tuple(value: Any) -> tuple[float, ...] | None:
    if value is None:
        return None
    try:
        return tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None


def _quat_tuple(value: Any) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    try:
        imaginary = value.GetImaginary()
        return (
            float(value.GetReal()),
            float(imaginary[0]),
            float(imaginary[1]),
            float(imaginary[2]),
        )
    except (AttributeError, TypeError, ValueError):
        values = _value_tuple(value)
        return values if values is not None and len(values) == 4 else None


def _normalized_quaternion(
    value: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    norm = math.sqrt(sum(component * component for component in value))
    if not math.isfinite(norm) or norm <= 0.0:
        return None
    return tuple(component / norm for component in value)


def _quaternion_from_rpy(
    rpy: list[float],
) -> tuple[float, float, float, float]:
    roll, pitch, yaw = (float(value) for value in rpy)
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return _normalized_quaternion(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        )
    )  # type: ignore[return-value]


def _quaternion_angle_error(
    actual: tuple[float, float, float, float] | None,
    expected: tuple[float, float, float, float],
) -> float:
    normalized = _normalized_quaternion(actual)
    if normalized is None:
        return math.inf
    dot = min(1.0, abs(sum(a * b for a, b in zip(normalized, expected))))
    return 2.0 * math.acos(dot)


def _quaternion_rotation_matrix(
    quaternion: tuple[float, float, float, float],
) -> list[list[float]]:
    w, x, y, z = quaternion
    return [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ]


def _reconstruct_inertia(
    diagonal: tuple[float, ...],
    quaternion: tuple[float, float, float, float],
) -> list[list[float]]:
    rotation = _quaternion_rotation_matrix(quaternion)
    return [
        [
            sum(rotation[row][axis] * diagonal[axis] * rotation[column][axis] for axis in range(3))
            for column in range(3)
        ]
        for row in range(3)
    ]


def _matrix4_is_identity(matrix: Any, tolerance: float = 1.0e-8) -> bool:
    if isinstance(matrix, tuple):
        matrix = matrix[0]
    try:
        return all(
            math.isclose(
                float(matrix[row][column]),
                1.0 if row == column else 0.0,
                abs_tol=tolerance,
            )
            for row in range(4)
            for column in range(4)
        )
    except (IndexError, TypeError, ValueError):
        return False


def _authored_attribute_value(prim: Any, name: str) -> Any:
    attribute = prim.GetAttribute(name)
    if not attribute:
        return None
    return attribute.Get()


def _resolved_material_binding_path(prim: Any) -> str | None:
    current = prim
    while current:
        for relationship_name in ("material:binding:physics", "material:binding"):
            targets = _relationship_targets(current, relationship_name)
            if len(targets) == 1:
                return targets[0]
            if len(targets) > 1:
                return None
        current = current.GetParent()
    return None


def _stl_triangles(path: Path) -> list[tuple[tuple[float, float, float], ...]]:
    data = path.read_bytes()
    triangles: list[tuple[tuple[float, float, float], ...]] = []
    if len(data) >= 84:
        count = struct.unpack_from("<I", data, 80)[0]
        if 84 + 50 * count == len(data):
            for index in range(count):
                values = struct.unpack_from("<12f", data, 84 + 50 * index)
                triangles.append(
                    tuple(
                        tuple(float(component) for component in values[offset:offset + 3])
                        for offset in (3, 6, 9)
                    )
                )
            return triangles
    vertices: list[tuple[float, float, float]] = []
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"unsupported STL encoding: {path}") from exc
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) == 4 and parts[0].lower() == "vertex":
            vertices.append(tuple(float(value) for value in parts[1:]))
    if len(vertices) % 3 != 0 or not vertices:
        raise ValueError(f"STL contains no complete triangles: {path}")
    return [tuple(vertices[index:index + 3]) for index in range(0, len(vertices), 3)]


def _usd_mesh_triangles(prim: Any, usd_geom: Any) -> list[tuple[tuple[float, float, float], ...]]:
    mesh = usd_geom.Mesh(prim)
    points = mesh.GetPointsAttr().Get()
    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    if points is None or counts is None or indices is None:
        return []
    if any(int(count) != 3 for count in counts):
        return []
    point_values = [tuple(float(component) for component in point) for point in points]
    triangles = []
    cursor = 0
    for count in counts:
        face = tuple(point_values[int(indices[cursor + offset])] for offset in range(int(count)))
        triangles.append(face)
        cursor += int(count)
    return triangles


def _canonical_triangle_coordinates(
    triangles: list[tuple[tuple[float, float, float], ...]],
    tolerance_m: float = 1.0e-7,
) -> tuple[tuple[tuple[int, int, int], ...], ...]:
    def quantize(point: tuple[float, float, float]) -> tuple[int, int, int]:
        return tuple(int(round(component / tolerance_m)) for component in point)

    return tuple(
        sorted(tuple(sorted(quantize(point) for point in triangle)) for triangle in triangles)
    )


def _inspect_physical_r7_contract(
    *,
    asset_path: Path,
    stage: Any,
    root_path: Any,
    all_prims: list[Any],
    rigid_bodies_by_name: dict[str, list[Any]],
    usd_geom: Any,
    usd_physics: Any,
    usd_shade: Any,
    errors: list[str],
) -> dict[str, Any]:
    from kcg_connector.d38999_keyed_v2_physical_model_contract import (
        REQUIRED_SELF_COLLISION_EXCLUSIONS,
        WORKSPACE_ROOT,
        load_physical_model_contract,
    )

    model = load_physical_model_contract()
    blueprint = model.document["realized_robot_hand_fixture_blueprint"]
    successor = blueprint["successor_robot_asset"]
    expected_asset = (WORKSPACE_ROOT / successor["output_path"]).resolve()
    if asset_path.resolve() != expected_asset:
        errors.append(
            f"physical-r7 validation requires {expected_asset}, got {asset_path.resolve()}"
        )
    if str(root_path) != successor["default_prim"]:
        errors.append(
            f"physical-r7 default prim expected {successor['default_prim']}, got {root_path}"
        )

    collision_contract = blueprint["collision_inventory"]
    expected_collision_rows = {
        row["link"]: row for row in collision_contract["per_link_source_inventory"]
    }
    collision_by_owner: dict[str, list[Any]] = {
        link: [] for link in expected_collision_rows
    }
    unexpected_collision_paths: list[str] = []
    for prim in all_prims:
        if not prim.HasAPI(usd_physics.CollisionAPI):
            continue
        owner = prim
        while owner and not owner.HasAPI(usd_physics.RigidBodyAPI):
            owner = owner.GetParent()
        owner_name = owner.GetName() if owner else None
        if owner_name in collision_by_owner:
            collision_by_owner[owner_name].append(prim)
        else:
            unexpected_collision_paths.append(str(prim.GetPath()))
    if unexpected_collision_paths:
        errors.append(
            f"physical-r7 robot has collision prims outside its 17 semantic owners: {unexpected_collision_paths}"
        )

    collision_records: dict[str, Any] = {}
    custom_names = collision_contract["required_custom_attribute_names"]
    material_values = model.document["material_roles"]["roles"]
    for link, expected in expected_collision_rows.items():
        matches = collision_by_owner[link]
        record = {"paths": [str(prim.GetPath()) for prim in matches]}
        collision_records[link] = record
        if len(matches) != 1:
            errors.append(f"{link} expected exactly one physical collider, found {record['paths']}")
            continue
        prim = matches[0]
        approximation = _authored_attribute_value(prim, "physics:approximation")
        source_uri = _authored_attribute_value(prim, custom_names["source_mesh_uri"])
        material_role = _authored_attribute_value(prim, custom_names["material_role"])
        response_role = _authored_attribute_value(prim, custom_names["response_role"])
        collision_enabled = usd_physics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        owner = prim
        while owner and not owner.HasAPI(usd_physics.RigidBodyAPI):
            owner = owner.GetParent()
        rigid_api = usd_physics.RigidBodyAPI(owner) if owner else None
        rigid_enabled = None if rigid_api is None else rigid_api.GetRigidBodyEnabledAttr().Get()
        kinematic_enabled = None if rigid_api is None else rigid_api.GetKinematicEnabledAttr().Get()
        ccd_enabled = None if owner is None else _authored_attribute_value(owner, "physxRigidBody:enableCCD")
        contact_offset = _authored_attribute_value(prim, "physxCollision:contactOffset")
        rest_offset = _authored_attribute_value(prim, "physxCollision:restOffset")
        material_path = _resolved_material_binding_path(prim)
        expected_material_path = f"{root_path}/PhysicsMaterials/{expected['material_role']}"
        material_prim = stage.GetPrimAtPath(material_path) if material_path else None
        source_relative = expected["mesh_uri"].removeprefix("package://iiwa_description/")
        source_path = (WORKSPACE_ROOT / "src/iiwa_description" / source_relative).resolve()
        try:
            source_triangles = _stl_triangles(source_path)
            realized_triangles = _usd_mesh_triangles(prim, usd_geom)
            source_mesh_equal = (
                _canonical_triangle_coordinates(source_triangles)
                == _canonical_triangle_coordinates(realized_triangles)
            )
        except (OSError, ValueError, struct.error) as exc:
            source_mesh_equal = False
            errors.append(f"{link} source/realized mesh readback failed: {exc}")
        local_identity = _matrix4_is_identity(
            usd_geom.Xformable(prim).GetLocalTransformation()
        )
        record.update(
            {
                "path": str(prim.GetPath()),
                "approximation": str(approximation) if approximation is not None else None,
                "source_mesh_uri": str(source_uri) if source_uri is not None else None,
                "material_role": str(material_role) if material_role is not None else None,
                "response_role": str(response_role) if response_role is not None else None,
                "local_transform_identity": local_identity,
                "collision_enabled": collision_enabled,
                "owner_rigid_body_enabled": rigid_enabled,
                "owner_kinematic_enabled": kinematic_enabled,
                "owner_ccd_enabled": ccd_enabled,
                "contact_offset_m": contact_offset,
                "rest_offset_m": rest_offset,
                "effective_material_binding": material_path,
                "source_triangle_count": len(source_triangles) if source_mesh_equal else None,
                "realized_triangle_count": len(realized_triangles) if source_mesh_equal else None,
                "source_mesh_numeric_equality": source_mesh_equal,
            }
        )
        if str(prim.GetTypeName()) != "Mesh":
            errors.append(f"{link} collider must be Mesh, got {prim.GetTypeName()}")
        if str(approximation) != collision_contract["every_usd_approximation"]:
            errors.append(f"{link} collider approximation changed: {approximation}")
        if str(source_uri) != expected["mesh_uri"]:
            errors.append(f"{link} collider source mesh identity changed: {source_uri}")
        if str(material_role) != expected["material_role"]:
            errors.append(f"{link} collider material role changed: {material_role}")
        if str(response_role) != expected["response_role"]:
            errors.append(f"{link} collider response role changed: {response_role}")
        if not local_identity:
            errors.append(f"{link} collider local transform or scale is not identity")
        if collision_enabled is not True:
            errors.append(f"{link} collisionEnabled must resolve true, got {collision_enabled}")
        if rigid_enabled is not True or kinematic_enabled is not False:
            errors.append(
                f"{link} rigid owner enable/kinematic state changed: "
                f"rigid={rigid_enabled}, kinematic={kinematic_enabled}"
            )
        if ccd_enabled is not True:
            errors.append(f"{link} owner CCD must resolve true, got {ccd_enabled}")
        if contact_offset is None or not math.isclose(
            float(contact_offset), float(collision_contract["every_collision_contactOffset_m"]),
            abs_tol=1.0e-10,
        ):
            errors.append(f"{link} contactOffset changed: {contact_offset}")
        if rest_offset is None or not math.isclose(
            float(rest_offset), float(collision_contract["every_collision_restOffset_m"]),
            abs_tol=1.0e-10,
        ):
            errors.append(f"{link} restOffset changed: {rest_offset}")
        if material_path != expected_material_path or not material_prim:
            errors.append(
                f"{link} effective material binding expected {expected_material_path}, "
                f"got {material_path}"
            )
        else:
            role_values = material_values[expected["material_role"]]
            resolved_role = _authored_attribute_value(material_prim, "kcg:materialRole")
            if str(resolved_role) != expected["material_role"]:
                errors.append(f"{link} bound material role changed: {resolved_role}")
            for attribute_name, target in (
                ("physics:staticFriction", role_values["static_friction"]),
                ("physics:dynamicFriction", role_values["dynamic_friction"]),
                ("physics:restitution", role_values["restitution"]),
            ):
                value = _authored_attribute_value(material_prim, attribute_name)
                if value is None or not math.isclose(
                    float(value), float(target), abs_tol=1.0e-7
                ):
                    errors.append(
                        f"{link} bound material {attribute_name} changed: {value}"
                    )
        if not source_mesh_equal:
            errors.append(f"{link} realized Mesh points/triangles differ from source STL")

    mass_contract = blueprint["mass_property_inventory"]
    mass_rows = {row["link"]: row for row in mass_contract["per_link_values"]}
    tolerances = mass_contract["readback_comparison"]
    mass_records: dict[str, Any] = {}
    total_mass = 0.0
    mass_api_prims = [prim for prim in all_prims if prim.HasAPI(usd_physics.MassAPI)]
    mass_api_paths = {str(prim.GetPath()) for prim in mass_api_prims}
    expected_mass_paths = {
        str(matches[0].GetPath())
        for link, matches in rigid_bodies_by_name.items()
        if link in mass_rows and len(matches) == 1
    }
    if mass_api_paths != expected_mass_paths or len(mass_api_prims) != 17:
        errors.append(
            "physical-r7 MassAPI inventory changed: "
            f"missing={sorted(expected_mass_paths - mass_api_paths)}, "
            f"unexpected={sorted(mass_api_paths - expected_mass_paths)}"
        )
    all_rigid_paths = {
        str(prim.GetPath()) for prim in all_prims if prim.HasAPI(usd_physics.RigidBodyAPI)
    }
    if all_rigid_paths != expected_mass_paths:
        errors.append(
            "physical-r7 rigid-body inventory changed: "
            f"missing={sorted(expected_mass_paths - all_rigid_paths)}, "
            f"unexpected={sorted(all_rigid_paths - expected_mass_paths)}"
        )
    for link, expected in mass_rows.items():
        matches = rigid_bodies_by_name.get(link, [])
        if len(matches) != 1:
            continue
        prim = matches[0]
        if not prim.HasAPI(usd_physics.MassAPI):
            errors.append(f"{link} lost PhysicsMassAPI")
            continue
        api = usd_physics.MassAPI(prim)
        rigid_api = usd_physics.RigidBodyAPI(prim)
        rigid_enabled = rigid_api.GetRigidBodyEnabledAttr().Get()
        kinematic_enabled = rigid_api.GetKinematicEnabledAttr().Get()
        mass = api.GetMassAttr().Get()
        com = _value_tuple(api.GetCenterOfMassAttr().Get())
        diagonal = _value_tuple(api.GetDiagonalInertiaAttr().Get())
        principal = _normalized_quaternion(_quat_tuple(api.GetPrincipalAxesAttr().Get()))
        mass_records[link] = {
            "mass_kg": None if mass is None else float(mass),
            "com_m": com,
            "diagonal_inertia_kg_m2": diagonal,
            "principal_axes_wxyz": principal,
            "rigid_body_enabled": rigid_enabled,
            "kinematic_enabled": kinematic_enabled,
        }
        if rigid_enabled is not True or kinematic_enabled is not False:
            errors.append(
                f"{link} mass owner enable/kinematic state changed: "
                f"rigid={rigid_enabled}, kinematic={kinematic_enabled}"
            )
        if mass is None or com is None or diagonal is None or principal is None:
            errors.append(f"{link} has incomplete mass-property readback")
            continue
        total_mass += float(mass)
        if not math.isclose(
            float(mass), float(expected["mass_kg"]),
            abs_tol=float(tolerances["mass_abs_tolerance_kg"]),
        ):
            errors.append(f"{link} mass changed: {mass}")
        if any(
            not math.isclose(
                actual, float(target),
                abs_tol=float(tolerances["com_component_abs_tolerance_m"]),
            )
            for actual, target in zip(com, expected["com_m"])
        ):
            errors.append(f"{link} center of mass changed: {com}")
        reconstructed = _reconstruct_inertia(diagonal, principal)
        if (
            any(value <= 0.0 or not math.isfinite(value) for value in diagonal)
            or diagonal[0] + diagonal[1] < diagonal[2]
            or diagonal[0] + diagonal[2] < diagonal[1]
            or diagonal[1] + diagonal[2] < diagonal[0]
        ):
            errors.append(f"{link} diagonal inertia is not physically admissible: {diagonal}")
        six = expected["inertia_six_kg_m2"]
        expected_tensor = [
            [six[0], six[3], six[4]],
            [six[3], six[1], six[5]],
            [six[4], six[5], six[2]],
        ]
        if any(
            not math.isclose(
                reconstructed[row][column], float(expected_tensor[row][column]),
                abs_tol=float(tolerances["full_inertia_component_abs_tolerance_kg_m2"]),
                rel_tol=float(tolerances["full_inertia_component_rel_tolerance"]),
            )
            for row in range(3)
            for column in range(3)
        ):
            errors.append(f"{link} reconstructed full inertia tensor changed")
    if not math.isclose(
        total_mass, float(mass_contract["expected_total_mass_kg"]),
        abs_tol=float(tolerances["mass_abs_tolerance_kg"]) * len(mass_rows),
    ):
        errors.append(f"17-link total mass changed: {total_mass}")

    joint_contract = blueprint["joint_property_inventory"]
    joint_tolerances = joint_contract["readback_tolerances"]
    physics_path = root_path.AppendChild("Physics")
    joint_records: dict[str, Any] = {}
    for expected in joint_contract["revolute_joints_exactly"]:
        joint_path = physics_path.AppendChild(expected["joint"])
        prim = stage.GetPrimAtPath(joint_path)
        if not prim or not prim.IsA(usd_physics.RevoluteJoint):
            errors.append(f"physical-r7 missing revolute joint {joint_path}")
            continue
        joint = usd_physics.RevoluteJoint(prim)
        base_joint = usd_physics.Joint(prim)
        applied_schemas = [str(schema) for schema in prim.GetAppliedSchemas()]
        required_schemas = set(joint_contract["drive_mapping"]["required_applied_schemas"])
        missing_schemas = required_schemas - set(applied_schemas)
        if missing_schemas:
            errors.append(
                f"{expected['joint']} missing required joint schemas: {sorted(missing_schemas)}"
            )
        joint_enabled = base_joint.GetJointEnabledAttr().Get()
        excluded_from_articulation = base_joint.GetExcludeFromArticulationAttr().Get()
        drive_type = _authored_attribute_value(prim, "drive:angular:physics:type")
        if joint_enabled is not True:
            errors.append(f"{expected['joint']} physics:jointEnabled changed: {joint_enabled}")
        if excluded_from_articulation is not False:
            errors.append(
                f"{expected['joint']} physics:excludeFromArticulation changed: "
                f"{excluded_from_articulation}"
            )
        if str(drive_type) != joint_contract["drive_mapping"]["drive:angular:physics:type"]:
            errors.append(f"{expected['joint']} drive type changed: {drive_type}")
        local_pos0 = _value_tuple(joint.GetLocalPos0Attr().Get())
        local_pos1 = _value_tuple(joint.GetLocalPos1Attr().Get())
        local_rot0 = _quat_tuple(joint.GetLocalRot0Attr().Get())
        local_rot1 = _quat_tuple(joint.GetLocalRot1Attr().Get())
        lower = joint.GetLowerLimitAttr().Get()
        upper = joint.GetUpperLimitAttr().Get()
        expected_quaternion = _quaternion_from_rpy(expected["rpy_rad"])
        record = {
            "path": str(joint_path), "localPos0": local_pos0,
            "localPos1": local_pos1, "localRot0_wxyz": local_rot0,
            "localRot1_wxyz": local_rot1, "lower_deg": lower,
            "upper_deg": upper,
            "applied_schemas": applied_schemas,
            "joint_enabled": joint_enabled,
            "exclude_from_articulation": excluded_from_articulation,
            "drive_type": None if drive_type is None else str(drive_type),
        }
        joint_records[expected["joint"]] = record
        if local_pos0 is None or any(
            not math.isclose(actual, float(target), abs_tol=float(joint_tolerances["local_position_abs_m"]))
            for actual, target in zip(local_pos0, expected["xyz_m"])
        ):
            errors.append(f"{expected['joint']} localPos0 changed: {local_pos0}")
        if local_pos1 is None or any(
            not math.isclose(value, 0.0, abs_tol=float(joint_tolerances["local_position_abs_m"]))
            for value in local_pos1
        ):
            errors.append(f"{expected['joint']} localPos1 changed: {local_pos1}")
        if _quaternion_angle_error(local_rot0, expected_quaternion) > float(joint_tolerances["reconstructed_rotation_angle_abs_rad"]):
            errors.append(f"{expected['joint']} localRot0 changed: {local_rot0}")
        if _quaternion_angle_error(local_rot1, (1.0, 0.0, 0.0, 0.0)) > float(joint_tolerances["reconstructed_rotation_angle_abs_rad"]):
            errors.append(f"{expected['joint']} localRot1 changed: {local_rot1}")
        expected_lower = math.degrees(float(expected["lower_rad"]))
        expected_upper = math.degrees(float(expected["upper_rad"]))
        if lower is None or not math.isclose(float(lower), expected_lower, abs_tol=float(joint_tolerances["limit_abs_deg"])):
            errors.append(f"{expected['joint']} lower limit changed: {lower}")
        if upper is None or not math.isclose(float(upper), expected_upper, abs_tol=float(joint_tolerances["limit_abs_deg"])):
            errors.append(f"{expected['joint']} upper limit changed: {upper}")
        numeric_attributes = {
            "drive:angular:physics:damping": (expected["damping"], "damping_abs"),
            "drive:angular:physics:maxForce": (expected["effort"], "effort_abs"),
            "physxJoint:maxJointVelocity": (math.degrees(float(expected["velocity_rad_s"])), "velocity_abs_deg_s"),
            "physxJoint:jointFriction": (expected["friction"], "friction_abs"),
        }
        for attribute_name, (target, tolerance_name) in numeric_attributes.items():
            value = _authored_attribute_value(prim, attribute_name)
            if value is None:
                value = 0.0 if attribute_name == "physxJoint:jointFriction" else None
            if value is None or not math.isclose(
                float(value), float(target), abs_tol=float(joint_tolerances[tolerance_name])
            ):
                errors.append(f"{expected['joint']} {attribute_name} changed: {value}")
        for forbidden in (
            "drive:angular:physics:stiffness",
            "drive:angular:physics:targetPosition",
            "drive:angular:physics:targetVelocity",
        ):
            attribute = prim.GetAttribute(forbidden)
            if attribute and attribute.HasAuthoredValueOpinion():
                errors.append(f"{expected['joint']} unexpectedly authors {forbidden}")
        mimic = expected["mimic"]
        targets = _relationship_targets(prim, "newton:mimicJoint")
        if mimic is None:
            if targets:
                errors.append(f"active joint {expected['joint']} has mimic target {targets}")
        else:
            target_path = str(physics_path.AppendChild(mimic["joint"]))
            if targets != [target_path]:
                errors.append(f"{expected['joint']} mimic target changed: {targets}")
            multiplier = _authored_attribute_value(prim, "newton:mimicMultiplier")
            offset = _authored_attribute_value(prim, "newton:mimicOffset")
            if multiplier is None or not math.isclose(
                float(multiplier), float(mimic["multiplier"]),
                abs_tol=float(joint_tolerances["mimic_multiplier_abs"]),
            ):
                errors.append(f"{expected['joint']} mimic multiplier changed: {multiplier}")
            if offset is None or not math.isclose(
                float(offset), math.degrees(float(mimic["offset_rad"])),
                abs_tol=math.degrees(float(joint_tolerances["mimic_offset_abs_rad"])),
            ):
                errors.append(f"{expected['joint']} mimic offset changed: {offset}")

    for expected in joint_contract["fixed_joints_exactly"]:
        path = physics_path.AppendChild(expected["joint"])
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsA(usd_physics.FixedJoint):
            errors.append(f"physical-r7 missing fixed joint {path}")
            continue
        joint = usd_physics.FixedJoint(prim)
        base_joint = usd_physics.Joint(prim)
        joint_enabled = base_joint.GetJointEnabledAttr().Get()
        excluded_from_articulation = base_joint.GetExcludeFromArticulationAttr().Get()
        collision_enabled = base_joint.GetCollisionEnabledAttr().Get()
        if joint_enabled is not True:
            errors.append(f"{expected['joint']} physics:jointEnabled changed: {joint_enabled}")
        if excluded_from_articulation is not False:
            errors.append(
                f"{expected['joint']} physics:excludeFromArticulation changed: "
                f"{excluded_from_articulation}"
            )
        if collision_enabled is not False:
            errors.append(
                f"{expected['joint']} physics:collisionEnabled changed: {collision_enabled}"
            )
        local_pos0 = _value_tuple(joint.GetLocalPos0Attr().Get())
        local_pos1 = _value_tuple(joint.GetLocalPos1Attr().Get())
        local_rot0 = _quat_tuple(joint.GetLocalRot0Attr().Get())
        local_rot1 = _quat_tuple(joint.GetLocalRot1Attr().Get())
        if local_pos0 is None or any(
            not math.isclose(actual, float(target), abs_tol=float(joint_tolerances["local_position_abs_m"]))
            for actual, target in zip(local_pos0, expected["xyz_m"])
        ) or local_pos1 is None or any(
            not math.isclose(value, 0.0, abs_tol=float(joint_tolerances["local_position_abs_m"]))
            for value in local_pos1
        ) or _quaternion_angle_error(local_rot0, _quaternion_from_rpy(expected["rpy_rad"])) > float(joint_tolerances["reconstructed_rotation_angle_abs_rad"]) or _quaternion_angle_error(local_rot1, (1.0, 0.0, 0.0, 0.0)) > float(joint_tolerances["reconstructed_rotation_angle_abs_rad"]):
            errors.append(f"{expected['joint']} fixed-joint local frame changed")

    expected_joint_paths = {
        str(physics_path.AppendChild(row["joint"]))
        for row in joint_contract["revolute_joints_exactly"]
    } | {
        str(physics_path.AppendChild(row["joint"]))
        for row in joint_contract["fixed_joints_exactly"]
    }
    realized_joint_paths = {
        str(prim.GetPath()) for prim in all_prims if prim.IsA(usd_physics.Joint)
    }
    if realized_joint_paths != expected_joint_paths:
        errors.append(
            "physical-r7 joint inventory changed: "
            f"missing={sorted(expected_joint_paths - realized_joint_paths)}, "
            f"unexpected={sorted(realized_joint_paths - expected_joint_paths)}"
        )
    expected_drive_api_count = len(joint_contract["revolute_joints_exactly"])
    realized_drive_api_prims = [
        prim for prim in all_prims
        if "PhysicsDriveAPI:angular" in [str(schema) for schema in prim.GetAppliedSchemas()]
    ]
    if len(realized_drive_api_prims) != expected_drive_api_count:
        errors.append(
            "physical-r7 angular DriveAPI inventory changed: "
            f"expected={expected_drive_api_count}, actual={len(realized_drive_api_prims)}"
        )

    articulation_paths = [
        prim for prim in all_prims if prim.HasAPI(usd_physics.ArticulationRootAPI)
    ]
    if len(articulation_paths) == 1:
        enabled = _authored_attribute_value(
            articulation_paths[0], "newton:selfCollisionEnabled"
        )
        if enabled is not True:
            errors.append(f"physical-r7 self-collision must resolve true, got {enabled}")
    self_pairs: list[tuple[str, str]] = []
    for link, matches in rigid_bodies_by_name.items():
        if link not in expected_collision_rows or len(matches) != 1:
            continue
        for target_path in _relationship_targets(matches[0], "physics:filteredPairs"):
            target = stage.GetPrimAtPath(target_path)
            if target:
                self_pairs.append(tuple(sorted((link, target.GetName()))))
    normalized_pairs = frozenset(self_pairs)
    if normalized_pairs != REQUIRED_SELF_COLLISION_EXCLUSIONS or len(self_pairs) != 16:
        errors.append(
            "physical-r7 self-collision filters changed: "
            f"count={len(self_pairs)}, pairs={sorted(normalized_pairs)}"
        )
    expected_filter_owner_paths = {
        str(matches[0].GetPath())
        for link, matches in rigid_bodies_by_name.items()
        if link in expected_collision_rows and len(matches) == 1
        and _relationship_targets(matches[0], "physics:filteredPairs")
    }
    unexpected_filter_sources: list[str] = []
    for prim in all_prims:
        targets = _relationship_targets(prim, "physics:filteredPairs")
        if targets and str(prim.GetPath()) not in expected_filter_owner_paths:
            unexpected_filter_sources.append(str(prim.GetPath()))
    if unexpected_filter_sources:
        errors.append(
            "physical-r7 self-collision has collider/ancestor FilteredPairs sources: "
            f"{sorted(unexpected_filter_sources)}"
        )
    collision_groups = [
        str(prim.GetPath())
        for prim in all_prims
        if prim.IsA(usd_physics.CollisionGroup)
        or _relationship_targets(prim, "physics:filteredGroups")
    ]
    if collision_groups:
        errors.append(
            "physical-r7 robot self-collision may not be altered by CollisionGroup: "
            f"{sorted(set(collision_groups))}"
        )

    frames = blueprint["semantic_frames_and_sensors"]
    tcp_matches = [prim for prim in all_prims if prim.GetName() == "grasp_tcp"]
    if len(tcp_matches) == 1:
        tcp = tcp_matches[0]
        expected_tcp = frames["grasp_tcp"]
        runtime_root = str(successor["runtime_reference_root"])
        expected_runtime_path = str(expected_tcp["path"])
        if not expected_runtime_path.startswith(runtime_root + "/"):
            errors.append(
                "grasp_tcp runtime path is outside the frozen robot reference root"
            )
            expected_asset_path = expected_runtime_path
        else:
            expected_asset_path = str(root_path) + expected_runtime_path[len(runtime_root):]
        if str(tcp.GetPath()) != expected_asset_path:
            errors.append(
                f"grasp_tcp expected exact asset path {expected_asset_path}, got {tcp.GetPath()}"
            )
        if str(tcp.GetTypeName()) != expected_tcp["typeName"]:
            errors.append(
                f"grasp_tcp expected type {expected_tcp['typeName']}, got {tcp.GetTypeName()}"
            )
        matrix = usd_geom.Xformable(tcp).GetLocalTransformation()
        if isinstance(matrix, tuple):
            matrix = matrix[0]
        translation = _value_tuple(matrix.ExtractTranslation())
        orientation = _quat_tuple(matrix.ExtractRotationQuat())
        if translation is None or any(
            not math.isclose(actual, float(target), abs_tol=1.0e-7)
            for actual, target in zip(translation, expected_tcp["translation_m"])
        ) or _quaternion_angle_error(
            orientation, tuple(expected_tcp["rotation_wxyz"])
        ) > 1.0e-6:
            errors.append("grasp_tcp realized local transform changed")
        for api, api_label in (
            (usd_physics.RigidBodyAPI, "PhysicsRigidBodyAPI"),
            (usd_physics.MassAPI, "PhysicsMassAPI"),
            (usd_physics.CollisionAPI, "PhysicsCollisionAPI"),
        ):
            if tcp.HasAPI(api):
                errors.append(f"grasp_tcp unexpectedly has {api_label}")
    if any(prim.GetName() in {"PalmCamera", "WristCamera", "PalmLiveViewCamera", "WristLiveViewCamera"} for prim in all_prims):
        errors.append("successor robot asset must not precontain runtime camera prims")

    return {
        "contract_revision": model.document["identity"]["successor_revision"],
        "expected_asset": str(expected_asset),
        "collision_inventory": collision_records,
        "mass_inventory": mass_records,
        "joint_inventory": joint_records,
        "self_collision_pair_count": len(self_pairs),
        "camera_prim_count": sum(
            prim.GetName() in {"PalmCamera", "WristCamera", "PalmLiveViewCamera", "WristLiveViewCamera"}
            for prim in all_prims
        ),
    }


def validate_asset(asset_path: Path, *, physical_r7_contract: bool = False) -> dict[str, Any]:
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils

    errors: list[str] = []
    metrics: dict[str, Any] = {"asset": str(asset_path), "status": "FAILED"}

    dependency_layers, dependency_assets, unresolved = UsdUtils.ComputeAllDependencies(
        Sdf.AssetPath(str(asset_path))
    )
    unresolved_dependencies = sorted(_dependency_text(value) for value in unresolved)
    metrics["dependencies"] = {
        "asset_count": len(dependency_assets),
        "layer_count": len(dependency_layers),
        "layers": sorted(str(layer.identifier) for layer in dependency_layers),
        "unresolved": unresolved_dependencies,
    }
    if unresolved_dependencies:
        errors.append(f"unresolved USD dependencies: {unresolved_dependencies}")

    stage = Usd.Stage.Open(str(asset_path), Usd.Stage.LoadAll)
    if stage is None:
        raise AssetValidationError(
            [f"could not open robot USD: {asset_path}", *errors], metrics
        )

    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        raise AssetValidationError(["USD has no default prim", *errors], metrics)

    root_path = default_prim.GetPath()
    physics_path = root_path.AppendChild("Physics")
    variant_selection = default_prim.GetVariantSets().GetVariantSelection("Physics")
    metrics["stage"] = {
        "default_prim": str(root_path),
        "physics_variant": variant_selection,
        "used_layers": sorted(
            str(layer.identifier) for layer in stage.GetUsedLayers()
        ),
    }
    if default_prim.GetName() != "handarm":
        errors.append(
            f"default prim expected 'handarm', got {default_prim.GetName()!r}"
        )
    if variant_selection != "physx":
        errors.append(
            f"Physics variant must be 'physx', got {variant_selection!r}"
        )
    if "IsaacRobotAPI" not in [str(value) for value in default_prim.GetAppliedSchemas()]:
        errors.append(f"default prim lost IsaacRobotAPI: {root_path}")

    all_prims = list(stage.Traverse())
    prims_by_name: dict[str, list[Any]] = {}
    for prim in all_prims:
        prims_by_name.setdefault(prim.GetName(), []).append(prim)

    articulation_roots = [
        prim
        for prim in all_prims
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    articulation_paths = [str(prim.GetPath()) for prim in articulation_roots]
    expected_articulation_path = str(root_path.AppendPath("Geometry/world"))
    if articulation_paths != [expected_articulation_path]:
        errors.append(
            "expected one articulation root at "
            f"{expected_articulation_path}, found {articulation_paths}"
        )

    rigid_body_prims = [
        prim for prim in all_prims if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    rigid_bodies_by_name: dict[str, list[Any]] = {}
    for prim in rigid_body_prims:
        rigid_bodies_by_name.setdefault(prim.GetName(), []).append(prim)

    rigid_link_paths: dict[str, str | None] = {}
    for name in EXPECTED_RIGID_LINKS:
        matches = rigid_bodies_by_name.get(name, [])
        rigid_link_paths[name] = str(matches[0].GetPath()) if len(matches) == 1 else None
        if len(matches) != 1:
            errors.append(
                f"expected exactly one rigid link named {name!r}, found "
                f"{[str(prim.GetPath()) for prim in matches]}"
            )

    grasp_tcp = _require_single_named_prim(prims_by_name, "grasp_tcp", errors)
    iiwa_link_ee = _require_single_named_prim(prims_by_name, "iiwa_link_ee", errors)
    grasp_tcp_path = str(grasp_tcp.GetPath()) if grasp_tcp else None
    iiwa_link_ee_path = str(iiwa_link_ee.GetPath()) if iiwa_link_ee else None
    if grasp_tcp:
        if grasp_tcp.GetParent().GetName() != "handbase_link":
            errors.append(
                f"grasp_tcp is not parented to handbase_link: {grasp_tcp.GetPath()}"
            )
        if "IsaacSiteAPI" not in [str(value) for value in grasp_tcp.GetAppliedSchemas()]:
            errors.append(f"grasp_tcp lost IsaacSiteAPI: {grasp_tcp.GetPath()}")

    metrics["articulation"] = {
        "root_paths": articulation_paths,
        "rigid_body_count": len(rigid_body_prims),
        "required_rigid_links": rigid_link_paths,
    }
    metrics["named_frames"] = {
        "grasp_tcp": grasp_tcp_path,
        "iiwa_link_ee": iiwa_link_ee_path,
    }

    joint_records: dict[str, dict[str, Any]] = {}
    for name in KUKA_ACTIVE_JOINTS:
        joint_records[name] = _inspect_revolute_joint(
            stage, UsdPhysics, physics_path, name, None, errors
        )
    for name in HAND_ACTIVE_JOINTS:
        joint_records[name] = _inspect_revolute_joint(
            stage, UsdPhysics, physics_path, name, None, errors
        )
    for name, source_name in HAND_MIMIC_JOINTS.items():
        joint_records[name] = _inspect_revolute_joint(
            stage, UsdPhysics, physics_path, name, source_name, errors
        )

    expected_revolute_names = set(EXPECTED_ENDPOINTS)
    actual_revolute_names = {
        prim.GetName()
        for prim in all_prims
        if prim.IsA(UsdPhysics.RevoluteJoint)
        and prim.GetPath().GetParentPath() == physics_path
    }
    if actual_revolute_names != expected_revolute_names:
        errors.append(
            "unexpected revolute-joint set: "
            f"expected={sorted(expected_revolute_names)}, "
            f"actual={sorted(actual_revolute_names)}"
        )

    fixed_joints = {
        "world_iiwa_joint": _inspect_fixed_joint(
            stage,
            UsdPhysics,
            physics_path,
            "world_iiwa_joint",
            "handarm",
            "iiwa_link_0",
            errors,
        ),
        "hand2arm": _inspect_fixed_joint(
            stage,
            UsdPhysics,
            physics_path,
            "hand2arm",
            "iiwa_link_ee",
            "handbase_link",
            errors,
        ),
    }

    metrics["joints"] = {
        "revolute_count": len(actual_revolute_names),
        "kuka_active": list(KUKA_ACTIVE_JOINTS),
        "hand_active": list(HAND_ACTIVE_JOINTS),
        "hand_mimic": dict(sorted(HAND_MIMIC_JOINTS.items())),
        "fixed": fixed_joints,
        "topology": dict(sorted(joint_records.items())),
    }

    if physical_r7_contract:
        metrics["physical_r7_contract"] = _inspect_physical_r7_contract(
            asset_path=asset_path,
            stage=stage,
            root_path=root_path,
            all_prims=all_prims,
            rigid_bodies_by_name=rigid_bodies_by_name,
            usd_geom=UsdGeom,
            usd_physics=UsdPhysics,
            usd_shade=UsdShade,
            errors=errors,
        )

    if errors:
        raise AssetValidationError(errors, metrics)

    metrics["status"] = "PASSED"
    return metrics


def _emit_failure(
    asset_path: Path,
    errors: list[str],
    metrics: dict[str, Any] | None = None,
    *,
    physical_r7_contract: bool = False,
) -> None:
    report = metrics.copy() if metrics else {"asset": str(asset_path)}
    report.update({"errors": errors, "status": "FAILED"})
    # Keep the JSON on stdout on both success and failure.  Kit wraps stderr
    # lines in log prefixes, which would make a failure report invalid JSON.
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    print(PHYSICAL_FAIL_BANNER if physical_r7_contract else FAIL_BANNER)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the composed KUKA/three-finger-hand Isaac USD asset."
    )
    parser.add_argument(
        "--asset",
        default=str(_default_asset_path()),
        help="Robot root USD (defaults to the repository handarm.usda).",
    )
    parser.add_argument(
        "--physical-r7-contract",
        action="store_true",
        help=(
            "Also enforce the fail-closed keyed-v3 physical-r7 collision, "
            "mass, joint, self-collision, TCP, and camera contract."
        ),
    )
    arguments = parser.parse_args()

    asset_path = Path(arguments.asset).expanduser().resolve()
    if not asset_path.is_file():
        _emit_failure(
            asset_path,
            [f"robot USD does not exist: {asset_path}"],
            physical_r7_contract=arguments.physical_r7_contract,
        )
        return 2

    simulation_app = None
    exit_code = 0
    try:
        from isaacsim import SimulationApp

        simulation_app = SimulationApp(
            {
                "headless": True,
                "hide_ui": True,
                "renderer": "Minimal",
                "multi_gpu": False,
                "fast_shutdown": True,
                "enable_crashreporter": False,
            }
        )
        try:
            report = validate_asset(
                asset_path,
                physical_r7_contract=arguments.physical_r7_contract,
            )
            print(json.dumps(report, ensure_ascii=False, sort_keys=True))
            print(
                PHYSICAL_PASS_BANNER
                if arguments.physical_r7_contract
                else PASS_BANNER
            )
        except AssetValidationError as error:
            exit_code = 1
            _emit_failure(
                asset_path,
                error.errors,
                error.metrics,
                physical_r7_contract=arguments.physical_r7_contract,
            )
        except Exception as error:  # Keep a machine-readable failure on API/runtime errors.
            exit_code = 1
            _emit_failure(
                asset_path,
                [f"{type(error).__name__}: {error}"],
                physical_r7_contract=arguments.physical_r7_contract,
            )
    except Exception as error:
        exit_code = 1
        _emit_failure(
            asset_path,
            [f"Isaac Sim startup failed: {type(error).__name__}: {error}"],
            physical_r7_contract=arguments.physical_r7_contract,
        )
    finally:
        if simulation_app is not None:
            # Isaac Sim 6 fast shutdown may terminate inside close(). Preserve failures.
            simulation_app.close(exit_code=exit_code)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
