#!/usr/bin/env python3

"""Finalize the imported hand-arm package as the frozen physical-r7 asset.

The Isaac URDF importer deliberately remains the source of kinematic and visual
topology.  This postprocessor makes its instanced collision meshes explicit,
authors only the physical values frozen in the A0 contract, and refuses to
overwrite the unique successor path.  It never computes a file fingerprint.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from kcg_connector.d38999_keyed_v2_physical_model_contract import (
    REQUIRED_SELF_COLLISION_EXCLUSIONS,
    WORKSPACE_ROOT,
    load_physical_model_contract,
)


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finalize the frozen hand-arm physical-r7 USD package"
    )
    parser.add_argument("--imported-asset", required=True)
    parser.add_argument("--output", default=None)
    return parser.parse_args(argv)


def _apply_unknown_schema(prim: Any, schema_name: str, sdf: Any) -> None:
    schemas = list(str(item) for item in prim.GetAppliedSchemas())
    metadata = prim.GetMetadata("apiSchemas")
    if metadata is not None:
        schemas.extend(str(item) for item in metadata.GetAddedOrExplicitItems())
    if schema_name not in schemas:
        schemas.append(schema_name)
    prim.SetMetadata(
        "apiSchemas", sdf.TokenListOp.CreateExplicit(list(dict.fromkeys(schemas)))
    )


def _custom_string(prim: Any, name: str, value: str, sdf: Any) -> None:
    prim.CreateAttribute(name, sdf.ValueTypeNames.String, custom=True).Set(str(value))


def _owner_by_name(stage: Any, usd_physics: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    duplicates: set[str] = set()
    for prim in stage.Traverse():
        if not prim.HasAPI(usd_physics.RigidBodyAPI):
            continue
        name = prim.GetName()
        if name in result:
            duplicates.add(name)
        result[name] = prim
    if duplicates:
        raise ValueError(f"duplicate rigid-body names: {sorted(duplicates)}")
    return result


def _colliders_by_owner(stage: Any, usd_physics: Any) -> dict[str, list[Any]]:
    result: dict[str, list[Any]] = {}
    for prim in stage.Traverse():
        if not prim.HasAPI(usd_physics.CollisionAPI):
            continue
        owner = prim.GetParent()
        while owner and not owner.HasAPI(usd_physics.RigidBodyAPI):
            owner = owner.GetParent()
        if not owner:
            raise ValueError(f"collider has no rigid owner: {prim.GetPath()}")
        result.setdefault(owner.GetName(), []).append(prim)
    return result


def _rotation_quaternion(matrix: np.ndarray, gf: Any) -> Any:
    if float(np.linalg.det(matrix)) < 0.0:
        matrix = matrix.copy()
        matrix[:, 0] *= -1.0
    m00, m01, m02 = (float(value) for value in matrix[0])
    m10, m11, m12 = (float(value) for value in matrix[1])
    m20, m21, m22 = (float(value) for value in matrix[2])
    trace = m00 + m11 + m22
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w, x, y, z = (
            0.25 * scale,
            (m21 - m12) / scale,
            (m02 - m20) / scale,
            (m10 - m01) / scale,
        )
    elif m00 > m11 and m00 > m22:
        scale = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w, x, y, z = (
            (m21 - m12) / scale,
            0.25 * scale,
            (m01 + m10) / scale,
            (m02 + m20) / scale,
        )
    elif m11 > m22:
        scale = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w, x, y, z = (
            (m02 - m20) / scale,
            (m01 + m10) / scale,
            0.25 * scale,
            (m12 + m21) / scale,
        )
    else:
        scale = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w, x, y, z = (
            (m10 - m01) / scale,
            (m02 + m20) / scale,
            (m12 + m21) / scale,
            0.25 * scale,
        )
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("invalid principal-axis rotation matrix")
    w, x, y, z = (value / norm for value in (w, x, y, z))
    if w < 0.0:
        w, x, y, z = (-w, -x, -y, -z)
    return gf.Quatf(w, gf.Vec3f(x, y, z))


def _principal_inertia(six: Sequence[float], gf: Any) -> tuple[Any, Any]:
    tensor = np.asarray(
        [
            [six[0], six[3], six[4]],
            [six[3], six[1], six[5]],
            [six[4], six[5], six[2]],
        ],
        dtype=np.float64,
    )
    if not np.allclose(tensor, np.diag(np.diag(tensor)), atol=0.0, rtol=0.0):
        values, vectors = np.linalg.eigh(tensor)
        if np.any(values <= 0.0):
            raise ValueError(f"non-positive frozen inertia eigenvalue: {values}")
        quaternion = _rotation_quaternion(vectors, gf)
        return gf.Vec3f(*(float(value) for value in values)), quaternion
    return (
        gf.Vec3f(float(six[0]), float(six[1]), float(six[2])),
        gf.Quatf(1.0, gf.Vec3f(0.0, 0.0, 0.0)),
    )


def _material(
    stage: Any,
    role: str,
    values: Mapping[str, Any],
    *,
    sdf: Any,
    usd_physics: Any,
    usd_shade: Any,
) -> Any:
    path = f"/handarm/PhysicsMaterials/{role}"
    material = usd_shade.Material.Define(stage, path)
    api = usd_physics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(float(values["static_friction"]))
    api.CreateDynamicFrictionAttr(float(values["dynamic_friction"]))
    api.CreateRestitutionAttr(float(values["restitution"]))
    prim = material.GetPrim()
    _apply_unknown_schema(prim, "PhysxMaterialAPI", sdf)
    prim.CreateAttribute(
        "physxMaterial:frictionCombineMode", sdf.ValueTypeNames.Token, custom=False
    ).Set("max")
    prim.CreateAttribute(
        "physxMaterial:restitutionCombineMode", sdf.ValueTypeNames.Token, custom=False
    ).Set("min")
    prim.CreateAttribute(
        "physxMaterial:dampingCombineMode", sdf.ValueTypeNames.Token, custom=False
    ).Set("max")
    prim.CreateAttribute(
        "physxMaterial:compliantContactStiffness",
        sdf.ValueTypeNames.Float,
        custom=False,
    ).Set(0.0)
    prim.CreateAttribute(
        "physxMaterial:compliantContactDamping",
        sdf.ValueTypeNames.Float,
        custom=False,
    ).Set(0.0)
    prim.CreateAttribute(
        "physxMaterial:compliantContactAccelerationSpring",
        sdf.ValueTypeNames.Bool,
        custom=False,
    ).Set(False)
    _custom_string(prim, "kcg:materialRole", role, sdf)
    _custom_string(prim, "kcg:responseRole", "hard_rigid", sdf)
    return material


def _deinstance_collision_sources(stage: Any) -> None:
    instance_paths = [prim.GetPath() for prim in stage.Traverse() if prim.IsInstance()]
    if len(instance_paths) != 34:
        raise ValueError(
            f"imported hand-arm expected 34 visual/collision instances, got {len(instance_paths)}"
        )
    for path in instance_paths:
        stage.OverridePrim(path).SetInstanceable(False)
    stage.GetRootLayer().Save()
    stage.Reload()


def _author_contract(stage: Any, document: Mapping[str, Any]) -> None:
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

    root = stage.GetDefaultPrim()
    if not root or str(root.GetPath()) != "/handarm":
        raise ValueError(f"imported default prim changed: {root.GetPath() if root else None}")
    variant = root.GetVariantSets().GetVariantSet("Physics")
    if variant.GetVariantSelection() != "physx":
        raise ValueError(f"imported Physics variant must be physx, got {variant.GetVariantSelection()}")

    blueprint = document["realized_robot_hand_fixture_blueprint"]
    collision_contract = blueprint["collision_inventory"]
    collision_rows = {
        row["link"]: row for row in collision_contract["per_link_source_inventory"]
    }
    expected_links = set(collision_rows)
    owners = _owner_by_name(stage, UsdPhysics)
    if set(owners) != expected_links:
        raise ValueError(
            "imported rigid-owner inventory changed: "
            f"missing={sorted(expected_links - set(owners))}, "
            f"unexpected={sorted(set(owners) - expected_links)}"
        )
    colliders = _colliders_by_owner(stage, UsdPhysics)
    if set(colliders) != expected_links or any(len(rows) != 1 for rows in colliders.values()):
        raise ValueError(
            "de-instanced collider inventory must contain exactly one Mesh per rigid owner"
        )

    material_values = document["material_roles"]["roles"]
    materials = {
        role: _material(
            stage,
            role,
            material_values[role],
            sdf=Sdf,
            usd_physics=UsdPhysics,
            usd_shade=UsdShade,
        )
        for role in ("robot_structure", "finger_structure", "fingertip_pad")
    }
    custom_names = collision_contract["required_custom_attribute_names"]
    for link, row in collision_rows.items():
        owner = owners[link]
        collider = colliders[link][0]
        if str(collider.GetTypeName()) != "Mesh":
            raise ValueError(f"{link} collision source is not a Mesh")
        collision = UsdPhysics.CollisionAPI.Apply(collider)
        collision.CreateCollisionEnabledAttr(True)
        mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(collider)
        mesh_collision.CreateApproximationAttr("convexHull")
        _apply_unknown_schema(collider, "PhysxCollisionAPI", Sdf)
        collider.CreateAttribute(
            "physxCollision:contactOffset", Sdf.ValueTypeNames.Float, custom=False
        ).Set(float(collision_contract["every_collision_contactOffset_m"]))
        collider.CreateAttribute(
            "physxCollision:restOffset", Sdf.ValueTypeNames.Float, custom=False
        ).Set(float(collision_contract["every_collision_restOffset_m"]))
        _custom_string(collider, custom_names["source_mesh_uri"], row["mesh_uri"], Sdf)
        _custom_string(collider, custom_names["material_role"], row["material_role"], Sdf)
        _custom_string(collider, custom_names["response_role"], row["response_role"], Sdf)
        UsdShade.MaterialBindingAPI.Apply(collider).Bind(
            materials[row["material_role"]], materialPurpose="physics"
        )
        rigid = UsdPhysics.RigidBodyAPI.Apply(owner)
        rigid.CreateRigidBodyEnabledAttr(True)
        rigid.CreateKinematicEnabledAttr(False)
        _apply_unknown_schema(owner, "PhysxRigidBodyAPI", Sdf)
        owner.CreateAttribute(
            "physxRigidBody:enableCCD", Sdf.ValueTypeNames.Bool, custom=False
        ).Set(True)

    mass_rows = {
        row["link"]: row
        for row in blueprint["mass_property_inventory"]["per_link_values"]
    }
    for link, row in mass_rows.items():
        api = UsdPhysics.MassAPI.Apply(owners[link])
        api.CreateMassAttr(float(row["mass_kg"]))
        api.CreateCenterOfMassAttr(Gf.Vec3f(*(float(value) for value in row["com_m"])))
        diagonal, principal = _principal_inertia(row["inertia_six_kg_m2"], Gf)
        api.CreateDiagonalInertiaAttr(diagonal)
        api.CreatePrincipalAxesAttr(principal)

    physics_path = "/handarm/Physics"
    for row in blueprint["joint_property_inventory"]["revolute_joints_exactly"]:
        mimic = row["mimic"]
        if mimic is None:
            continue
        joint = stage.GetPrimAtPath(f"{physics_path}/{row['joint']}")
        if not joint:
            raise ValueError(f"missing imported mimic joint {row['joint']}")
        _apply_unknown_schema(joint, "NewtonMimicAPI", Sdf)
        joint.CreateAttribute(
            "newton:mimicEnabled", Sdf.ValueTypeNames.Bool, custom=False
        ).Set(True)
        joint.CreateAttribute(
            "newton:mimicCoef1", Sdf.ValueTypeNames.Float, custom=False
        ).Set(float(mimic["multiplier"]))
        offset_degrees = math.degrees(float(mimic["offset_rad"]))
        joint.CreateAttribute(
            "newton:mimicCoef0", Sdf.ValueTypeNames.Float, custom=False
        ).Set(offset_degrees)
        joint.CreateAttribute(
            "newton:mimicMultiplier", Sdf.ValueTypeNames.Float, custom=True
        ).Set(float(mimic["multiplier"]))
        joint.CreateAttribute(
            "newton:mimicOffset", Sdf.ValueTypeNames.Float, custom=True
        ).Set(offset_degrees)
        joint.CreateRelationship("newton:mimicJoint", custom=False).SetTargets(
            [Sdf.Path(f"{physics_path}/{mimic['joint']}")]
        )

    articulation = stage.GetPrimAtPath("/handarm/Geometry/world")
    articulation.CreateAttribute(
        "newton:selfCollisionEnabled", Sdf.ValueTypeNames.Bool, custom=False
    ).Set(True)
    target_map: dict[str, list[Any]] = {}
    frozen_pairs = blueprint["self_collision"]["normalized_undirected_pairs_exactly"]
    if frozenset(tuple(sorted(pair)) for pair in frozen_pairs) != REQUIRED_SELF_COLLISION_EXCLUSIONS:
        raise ValueError("self-collision pair contract changed during authoring")
    for source, target in frozen_pairs:
        target_map.setdefault(source, []).append(owners[target].GetPath())
    for source, targets in target_map.items():
        api = UsdPhysics.FilteredPairsAPI.Apply(owners[source])
        api.CreateFilteredPairsRel().SetTargets(targets)

    root_layer_data = dict(stage.GetRootLayer().customLayerData)
    root_layer_data.update(
        {
            "kcg:successorRevision": document["identity"]["successor_revision"],
            "kcg:sourceKind": "FROZEN_SIMULATION_PROXY",
            "kcg:hashesGenerated": False,
        }
    )
    stage.GetRootLayer().customLayerData = root_layer_data
    stage.GetRootLayer().Save()


def _static_release_check(asset_path: Path, document: Mapping[str, Any]) -> None:
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(asset_path))
    if stage is None:
        raise RuntimeError(f"could not reopen authored USD: {asset_path}")
    root = stage.GetDefaultPrim()
    if not root or str(root.GetPath()) != "/handarm":
        raise RuntimeError("authored default prim changed")
    if root.GetVariantSets().GetVariantSet("Physics").GetVariantSelection() != "physx":
        raise RuntimeError("authored Physics variant changed")
    prims = list(stage.Traverse())
    counts = {
        "rigid": sum(prim.HasAPI(UsdPhysics.RigidBodyAPI) for prim in prims),
        "mass": sum(prim.HasAPI(UsdPhysics.MassAPI) for prim in prims),
        "collision": sum(prim.HasAPI(UsdPhysics.CollisionAPI) for prim in prims),
        "revolute": sum(prim.IsA(UsdPhysics.RevoluteJoint) for prim in prims),
        "fixed": sum(prim.IsA(UsdPhysics.FixedJoint) for prim in prims),
        "groups": sum(prim.IsA(UsdPhysics.CollisionGroup) for prim in prims),
    }
    if counts != {
        "rigid": 17,
        "mass": 17,
        "collision": 17,
        "revolute": 15,
        "fixed": 2,
        "groups": 0,
    }:
        raise RuntimeError(f"authored physical inventory changed: {counts}")
    materials = document["realized_robot_hand_fixture_blueprint"][
        "robot_material_partition"
    ]["role_to_links"]
    for role in materials:
        if not stage.GetPrimAtPath(f"/handarm/PhysicsMaterials/{role}"):
            raise RuntimeError(f"authored material missing: {role}")
    camera_names = {
        "PalmCamera", "WristCamera", "PalmLiveViewCamera", "WristLiveViewCamera"
    }
    if any(prim.GetName() in camera_names for prim in prims):
        raise RuntimeError("successor asset unexpectedly contains a camera")


def build_package(
    imported_asset: Path | str,
    destination_directory: Path | str,
) -> Path:
    """Build one postprocessed package at an absent destination (testable helper)."""
    imported_asset = Path(imported_asset).expanduser().resolve()
    destination_directory = Path(destination_directory).expanduser().resolve()
    if not imported_asset.is_file() or imported_asset.name != "handarm.usda":
        raise FileNotFoundError(f"imported handarm.usda is unavailable: {imported_asset}")
    if destination_directory.exists():
        raise FileExistsError(f"refusing to overwrite package: {destination_directory}")
    model = load_physical_model_contract()
    shutil.copytree(imported_asset.parent, destination_directory)
    output = destination_directory / "handarm.usda"
    from pxr import Usd

    stage = Usd.Stage.Open(str(output))
    if stage is None:
        raise RuntimeError(f"could not open imported USD: {output}")
    stage.SetEditTarget(stage.GetRootLayer())
    _deinstance_collision_sources(stage)
    _author_contract(stage, model.document)
    _static_release_check(output, model.document)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _arguments(argv)
    model = load_physical_model_contract()
    successor = model.document["realized_robot_hand_fixture_blueprint"][
        "successor_robot_asset"
    ]
    expected_output = (WORKSPACE_ROOT / successor["output_path"]).resolve()
    output = expected_output if arguments.output is None else Path(arguments.output).resolve()
    if output != expected_output:
        raise ValueError(f"successor robot output must be exactly {expected_output}")
    if output.exists() or output.parent.exists():
        raise FileExistsError(f"refusing to overwrite immutable robot package: {output.parent}")
    if model.a2_asset_authoring_allowed is not True:
        raise PermissionError("A2 robot asset authoring is not authorized")

    imported = Path(arguments.imported_asset).expanduser().resolve()
    output.parent.parent.mkdir(parents=True, exist_ok=True)
    staging_container = Path(
        tempfile.mkdtemp(prefix=".handarm_physical_r7_", dir=output.parent.parent)
    )
    staged_directory = staging_container / "package"
    staged_output = build_package(imported, staged_directory)
    if staged_output.name != output.name:
        raise RuntimeError("staged successor filename changed")
    os.replace(staged_directory, output.parent)
    staging_container.rmdir()
    print(f"ISAAC ROBOT PHYSICAL R7 EXPORTED: {output}")
    print("downstream_authorized=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
