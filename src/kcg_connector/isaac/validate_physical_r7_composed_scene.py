#!/usr/bin/env python3

"""Fail-closed, no-timeline A2 validation of the complete physical-r11 scene.

This gate composes the frozen connector, fixture, table, and successor robot,
but never resets, plays, or steps physics.  It writes no evidence file and
does not compute a file fingerprint.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Iterable, Sequence


PASS_BANNER = "ISAAC PHYSICAL R11 COMPOSED A2 PASSED"
FAIL_BANNER = "ISAAC PHYSICAL R11 COMPOSED A2 FAILED"


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the complete physical-r11 scene without starting physics"
    )
    parser.add_argument(
        "--scene-config",
        default="src/kcg_connector/config/d38999_keyed_v2_tabletop_scene_v1.yaml",
    )
    parser.add_argument(
        "--model-contract",
        default="src/kcg_connector/config/d38999_keyed_v2_physical_model_contract_v1.yaml",
    )
    parser.add_argument("--candidate-index", type=int, default=None)
    parser.add_argument("--authorized-local-candidate-result", default=None)
    parser.add_argument("--authorized-local-candidate-result-sha256", default=None)
    parser.add_argument("--kit-portable-root", required=True)
    result = parser.parse_args(argv)
    local_fields = (
        result.authorized_local_candidate_result,
        result.authorized_local_candidate_result_sha256,
    )
    if any(local_fields) and not all(local_fields):
        parser.error("local candidate result path and SHA-256 must be supplied together")
    if all(local_fields) and (
        result.candidate_index != 2
        or Path(result.model_contract).name
        != "d38999_keyed_v3_physical_model_contract_r12_v1.yaml"
    ):
        parser.error("the pinned local candidate is available only to R12 candidate2")
    return result


def _emit(value: Any) -> None:
    os.write(1, (str(value) + "\n").encode("utf-8"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _tuple(value: Any) -> tuple[float, ...]:
    return tuple(float(item) for item in value)


def _require_vector(
    actual: Any,
    expected: Iterable[float],
    label: str,
    *,
    tolerance: float = 1.0e-7,
) -> None:
    actual_values = _tuple(actual)
    expected_values = tuple(float(value) for value in expected)
    _require(len(actual_values) == len(expected_values), f"{label} length changed")
    _require(
        all(
            math.isclose(a, b, rel_tol=0.0, abs_tol=tolerance)
            for a, b in zip(actual_values, expected_values)
        ),
        f"{label} changed: {actual_values}",
    )


def _quat_tuple(value: Any) -> tuple[float, float, float, float]:
    imaginary = value.GetImaginary()
    return (
        float(value.GetReal()),
        float(imaginary[0]),
        float(imaginary[1]),
        float(imaginary[2]),
    )


def _require_quaternion(
    actual: Any,
    expected: Iterable[float],
    label: str,
    *,
    tolerance: float = 1.0e-7,
) -> None:
    values = _quat_tuple(actual)
    target = tuple(float(value) for value in expected)
    direct = max(abs(a - b) for a, b in zip(values, target))
    negated = max(abs(a + b) for a, b in zip(values, target))
    _require(min(direct, negated) <= tolerance, f"{label} changed: {values}")


def _targets(prim: Any, name: str) -> list[str]:
    relationship = prim.GetRelationship(name)
    return [] if not relationship else [str(path) for path in relationship.GetTargets()]


def _resolved_material_path(prim: Any) -> str | None:
    current = prim
    while current:
        for name in ("material:binding:physics", "material:binding"):
            targets = _targets(current, name)
            if len(targets) == 1:
                return targets[0]
            if len(targets) > 1:
                return None
        current = current.GetParent()
    return None


def _reference_asset_paths(prim: Any) -> tuple[str, ...]:
    metadata = prim.GetMetadata("references")
    if metadata is None:
        return ()
    return tuple(
        str(reference.assetPath)
        for reference in metadata.GetAddedOrExplicitItems()
        if str(reference.assetPath)
    )


def _validate_joint(
    stage: Any,
    path: str,
    expected: dict[str, Any],
    usd_physics: Any,
) -> None:
    prim = stage.GetPrimAtPath(path)
    _require(prim and prim.IsA(usd_physics.FixedJoint), f"missing fixed joint {path}")
    joint = usd_physics.FixedJoint(prim)
    _require(
        _targets(prim, "physics:body0") == expected["body0"],
        f"{path} body0 changed",
    )
    _require(
        _targets(prim, "physics:body1") == expected["body1"],
        f"{path} body1 changed",
    )
    _require_vector(joint.GetLocalPos0Attr().Get(), expected["local_pos0"], f"{path} localPos0")
    _require_vector(joint.GetLocalPos1Attr().Get(), expected["local_pos1"], f"{path} localPos1")
    _require_quaternion(joint.GetLocalRot0Attr().Get(), expected["local_rot0"], f"{path} localRot0")
    _require_quaternion(joint.GetLocalRot1Attr().Get(), expected["local_rot1"], f"{path} localRot1")
    _require(joint.GetJointEnabledAttr().Get() is True, f"{path} is disabled")
    _require(
        joint.GetExcludeFromArticulationAttr().Get() is False,
        f"{path} exclusion changed",
    )
    _require(joint.GetCollisionEnabledAttr().Get() is False, f"{path} enables collision")


def _validate_xform(
    stage: Any,
    path: str,
    translation: Iterable[float],
    rotation_xyz: Iterable[float],
    usd_geom: Any,
) -> None:
    prim = stage.GetPrimAtPath(path)
    _require(bool(prim), f"missing transformed prim {path}")
    operations = usd_geom.Xformable(prim).GetOrderedXformOps()
    _require(len(operations) == 2, f"{path} transform stack changed")
    _require(
        operations[0].GetOpType() == usd_geom.XformOp.TypeTranslate,
        f"{path} first transform is not translation",
    )
    _require(
        operations[1].GetOpType() == usd_geom.XformOp.TypeRotateXYZ,
        f"{path} second transform is not rotateXYZ",
    )
    _require_vector(operations[0].Get(), translation, f"{path} translation")
    _require_vector(operations[1].Get(), rotation_xyz, f"{path} rotationXYZ")


def _run_validation(
    arguments: argparse.Namespace,
    *,
    authorized_model: Any | None = None,
    authorized_local_asset_path: str | None = None,
) -> dict[str, Any]:
    from isaacsim.core.api import World
    from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
    from omni.physx.scripts import physicsUtils
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

    from kcg_connector.d38999_keyed_v2_a2_readback_result import (
        validate_a2_composed_asset_release,
    )
    from kcg_connector.d38999_keyed_v2_physical_model_contract import WORKSPACE_ROOT
    from kcg_connector.d38999_tabletop_scene import (
        author_d38999_tabletop_scene,
        load_d38999_tabletop_scene,
        verify_d38999_tabletop_asset,
    )

    # This import reuses the same in-process Kit schemas.  It does not start a
    # second SimulationApp and does not mutate either immutable asset.
    from validate_robot_asset import validate_asset as validate_robot_asset

    repository = WORKSPACE_ROOT
    model_contract_path = Path(arguments.model_contract).expanduser().resolve()
    if model_contract_path.name == "d38999_keyed_v3_physical_model_contract_r12_v1.yaml":
        from kcg_connector.d38999_keyed_v3_physical_r12_contract import (
            candidate_model,
            load_r12_physical_model_contract,
        )

        if authorized_model is not None:
            _require(
                arguments.candidate_index == 2
                and authorized_local_asset_path is not None,
                "local R12 model requires candidate2 and an exact asset path",
            )
            model = authorized_model
        else:
            model = load_r12_physical_model_contract(model_contract_path)
            if arguments.candidate_index is not None:
                model = candidate_model(model, arguments.candidate_index)
    else:
        from kcg_connector.d38999_keyed_v2_physical_model_contract import (
            load_physical_model_contract,
        )

        if arguments.candidate_index is not None:
            raise ValueError("candidate index is available only for r12")
        model = load_physical_model_contract(model_contract_path)
    _require(model.a2_asset_authoring_allowed is True, "A2 authoring gate is closed")
    connector_path = (
        repository
        / model.document["identity"]["recommended_asset_directory"]
        / model.document["identity"]["recommended_asset_name"]
    ).resolve()
    connector_release = validate_a2_composed_asset_release(
        connector_path, model=model
    )
    _require(connector_release.release_evidence is True, "connector A2 evidence is not releasable")

    successor = model.document["realized_robot_hand_fixture_blueprint"][
        "successor_robot_asset"
    ]
    robot_path = (repository / successor["output_path"]).resolve()
    robot_report = validate_robot_asset(robot_path, physical_r7_contract=True)
    robot_physical = robot_report["physical_r7_contract"]

    scene_path = Path(arguments.scene_config).expanduser().resolve()
    config = load_d38999_tabletop_scene(
        scene_path, authorized_local_asset_path=authorized_local_asset_path
    )
    asset_path = verify_d38999_tabletop_asset(
        config,
        repository,
        authorized_local_asset_path=authorized_local_asset_path,
        authorized_model=model if authorized_local_asset_path is not None else None,
    )
    _require(asset_path == connector_path, "scene does not reference the released connector")

    World.clear_instance()
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=1.0 / config.physics.rate_hz,
        rendering_dt=1.0 / 60.0,
        backend="numpy",
        device="cpu",
    )
    _require(world.is_playing() is False, "timeline started before A2 authoring")
    stage = get_current_stage()
    authored = author_d38999_tabletop_scene(
        stage,
        config,
        asset_path,
        add_reference_to_stage=add_reference_to_stage,
        Gf=Gf,
        Sdf=Sdf,
        UsdGeom=UsdGeom,
        UsdPhysics=UsdPhysics,
        UsdShade=UsdShade,
        physics_utils=physicsUtils,
        authorized_local_asset_path=authorized_local_asset_path,
    )
    _require(authored["object_pose_writes_after_start"] == 0, "scene reports post-start pose writes")
    _require(
        authored["fixed_load_path"] == "FixedReceptacle->FixedFixture->world",
        "fixture load path changed",
    )
    add_reference_to_stage(str(robot_path), successor["runtime_reference_root"])
    _require(world.is_playing() is False, "timeline started during A2 composition")

    fixture_contract = model.document["realized_robot_hand_fixture_blueprint"][
        "fixture_load_path"
    ]
    fixture = stage.GetPrimAtPath(fixture_contract["fixture_path"])
    table = stage.GetPrimAtPath(fixture_contract["table_path"])
    fixed = stage.GetPrimAtPath(fixture_contract["fixed_receptacle_path"])
    _require(bool(table) and table.HasAPI(UsdPhysics.CollisionAPI), "table collider is missing")
    _require(not table.HasAPI(UsdPhysics.RigidBodyAPI), "table became a rigid body")
    _require(not table.HasAPI(UsdPhysics.MassAPI), "table unexpectedly has mass")
    _require(bool(fixture) and fixture.HasAPI(UsdPhysics.CollisionAPI), "fixture collider is missing")
    fixture_geometry = fixture_contract["fixture_collision_geometry"]
    _require(fixture.IsA(UsdGeom.Mesh), "fixture is not the frozen metric Mesh")
    fixture_mesh = UsdGeom.Mesh(fixture)
    fixture_points = list(fixture_mesh.GetPointsAttr().Get())
    _require(
        len(fixture_points) == len(fixture_geometry["local_points_m"]),
        "fixture point count changed",
    )
    for index, (actual, expected) in enumerate(
        zip(fixture_points, fixture_geometry["local_points_m"])
    ):
        _require_vector(actual, expected, f"fixture point {index}")
    _require(
        list(fixture_mesh.GetFaceVertexCountsAttr().Get())
        == fixture_geometry["face_vertex_counts"],
        "fixture face counts changed",
    )
    _require(
        list(fixture_mesh.GetFaceVertexIndicesAttr().Get())
        == fixture_geometry["face_vertex_indices"],
        "fixture face indices changed",
    )
    fixture_extent = list(fixture_mesh.GetExtentAttr().Get())
    _require(len(fixture_extent) == 2, "fixture extent count changed")
    for index, (actual, expected) in enumerate(
        zip(fixture_extent, fixture_geometry["local_extent_m"])
    ):
        _require_vector(actual, expected, f"fixture extent {index}")
    _require(
        str(fixture_mesh.GetSubdivisionSchemeAttr().Get())
        == fixture_geometry["subdivision_scheme"],
        "fixture subdivision scheme changed",
    )
    _require(
        fixture.HasAPI(UsdPhysics.MeshCollisionAPI),
        "fixture MeshCollisionAPI is missing",
    )
    _require(
        str(UsdPhysics.MeshCollisionAPI(fixture).GetApproximationAttr().Get())
        == fixture_geometry["collision_approximation"],
        "fixture collision approximation changed",
    )
    fixture_ops = UsdGeom.Xformable(fixture).GetOrderedXformOps()
    _require(
        [str(operation.GetOpName()) for operation in fixture_ops]
        == fixture_geometry["transform_op_order"],
        "fixture transform stack changed",
    )
    _require_vector(
        fixture_ops[0].Get(),
        fixture_geometry["translation_m"],
        "fixture translation",
    )
    _require(
        UsdGeom.Xformable(fixture).GetResetXformStack()
        is fixture_geometry["reset_xform_stack"],
        "fixture reset-xform-stack changed",
    )
    fixture_rigid = UsdPhysics.RigidBodyAPI(fixture)
    _require(fixture_rigid.GetRigidBodyEnabledAttr().Get() is True, "fixture rigid body is disabled")
    _require(fixture_rigid.GetKinematicEnabledAttr().Get() is False, "fixture became kinematic")
    fixture_mass = UsdPhysics.MassAPI(fixture)
    fixture_values = fixture_contract["fixture_mass_properties"]
    _require(
        math.isclose(float(fixture_mass.GetMassAttr().Get()), float(fixture_values["mass_kg"]), abs_tol=1.0e-6),
        "fixture mass changed",
    )
    _require_vector(fixture_mass.GetCenterOfMassAttr().Get(), fixture_values["local_com_m"], "fixture COM")
    _require_vector(
        fixture_mass.GetDiagonalInertiaAttr().Get(),
        fixture_values["diagonal_inertia_kg_m2"],
        "fixture inertia",
        tolerance=1.0e-8,
    )
    _require_quaternion(
        fixture_mass.GetPrincipalAxesAttr().Get(),
        (1.0, 0.0, 0.0, 0.0),
        "fixture principal axes",
    )
    _require(bool(fixed) and fixed.HasAPI(UsdPhysics.RigidBodyAPI), "fixed receptacle rigid body is missing")
    fixed_rigid = UsdPhysics.RigidBodyAPI(fixed)
    _require(fixed_rigid.GetRigidBodyEnabledAttr().Get() is True, "fixed receptacle is disabled")
    _require(fixed_rigid.GetKinematicEnabledAttr().Get() is False, "fixed receptacle became kinematic")
    fixed_mass_contract = fixture_contract["connector_body_mass_derivation"]["bodies"]["FixedReceptacle"]
    fixed_mass = UsdPhysics.MassAPI(fixed)
    _require(
        math.isclose(float(fixed_mass.GetMassAttr().Get()), float(fixed_mass_contract["mass_kg"]), abs_tol=1.0e-6),
        "fixed receptacle mass changed",
    )
    _require_vector(fixed_mass.GetCenterOfMassAttr().Get(), fixed_mass_contract["local_com_m"], "fixed receptacle COM")
    _require_vector(
        fixed_mass.GetDiagonalInertiaAttr().Get(),
        fixed_mass_contract["diagonal_inertia_kg_m2"],
        "fixed receptacle inertia",
        tolerance=1.0e-9,
    )

    fixture_material_path = authored["fixture_material_prim_path"]
    _require(_resolved_material_path(fixture) == fixture_material_path, "fixture material binding changed")
    fixture_material = stage.GetPrimAtPath(fixture_material_path)
    for name, expected in (
        ("physics:staticFriction", 0.35),
        ("physics:dynamicFriction", 0.25),
        ("physics:restitution", 0.0),
    ):
        value = fixture_material.GetAttribute(name).Get()
        _require(value is not None and math.isclose(float(value), expected, abs_tol=1.0e-7), f"fixture {name} changed")

    _validate_joint(
        stage,
        fixture_contract["fixture_to_world_joint_path"],
        {
            "body0": [],
            "body1": [fixture_contract["fixture_path"]],
            "local_pos0": fixture_contract["fixture_to_world"]["localPos0_m"],
            "local_pos1": fixture_contract["fixture_to_world"]["localPos1_m"],
            "local_rot0": fixture_contract["fixture_to_world"]["localRot0_wxyz"],
            "local_rot1": fixture_contract["fixture_to_world"]["localRot1_wxyz"],
        },
        UsdPhysics,
    )
    _validate_joint(
        stage,
        fixture_contract["receptacle_to_fixture_joint_path"],
        {
            "body0": [fixture_contract["fixture_path"]],
            "body1": [fixture_contract["fixed_receptacle_path"]],
            "local_pos0": fixture_contract["receptacle_to_fixture"]["localPos0_m"],
            "local_pos1": fixture_contract["receptacle_to_fixture"]["localPos1_m"],
            "local_rot0": fixture_contract["receptacle_to_fixture"]["localRot0_wxyz"],
            "local_rot1": fixture_contract["receptacle_to_fixture"]["localRot1_wxyz"],
        },
        UsdPhysics,
    )
    fixed_joint_sources = [
        str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.IsA(UsdPhysics.FixedJoint)
        and _targets(prim, "physics:body1") == [fixture_contract["fixed_receptacle_path"]]
    ]
    _require(
        fixed_joint_sources == [fixture_contract["receptacle_to_fixture_joint_path"]],
        f"direct or duplicate fixed-receptacle load path found: {fixed_joint_sources}",
    )

    _validate_xform(
        stage,
        config.asset.fixed_receptacle_prim_path,
        config.fixed_endpoint.receptacle_origin_m,
        config.asset_profile.fixed_endpoint_rotation_degrees_xyz,
        UsdGeom,
    )
    _validate_xform(
        stage,
        config.asset.loose_plug_prim_path,
        config.loose_endpoint.initial_origin_m,
        config.asset_profile.loose_endpoint_rotation_degrees_xyz,
        UsdGeom,
    )

    pair_reference = stage.GetPrimAtPath(fixture_contract["pair_reference_path"])
    robot_reference = stage.GetPrimAtPath(successor["runtime_reference_root"])
    _require(bool(pair_reference), "connector reference prim is missing")
    _require(bool(robot_reference), "robot reference prim is missing")
    _require(
        _reference_asset_paths(pair_reference) == (str(connector_path),),
        f"connector reference source changed: {_reference_asset_paths(pair_reference)}",
    )
    _require(
        _reference_asset_paths(robot_reference) == (str(robot_path),),
        f"robot reference source changed: {_reference_asset_paths(robot_reference)}",
    )
    articulation = stage.GetPrimAtPath(successor["articulation_path"])
    _require(
        bool(articulation) and articulation.HasAPI(UsdPhysics.ArticulationRootAPI),
        "composed robot articulation is missing",
    )
    camera_names = {
        "PalmCamera", "WristCamera", "PalmLiveViewCamera", "WristLiveViewCamera"
    }
    _require(
        not any(prim.GetName() in camera_names for prim in stage.Traverse()),
        "A2 composed stage unexpectedly contains a runtime camera",
    )

    prims = list(stage.Traverse())
    counts = {
        "rigid_body": sum(prim.HasAPI(UsdPhysics.RigidBodyAPI) for prim in prims),
        "mass_api": sum(prim.HasAPI(UsdPhysics.MassAPI) for prim in prims),
        "collision": sum(prim.HasAPI(UsdPhysics.CollisionAPI) for prim in prims),
        "collision_group": sum(prim.IsA(UsdPhysics.CollisionGroup) for prim in prims),
        "fixed_joint": sum(prim.IsA(UsdPhysics.FixedJoint) for prim in prims),
    }
    _require(
        counts == {
            "rigid_body": 21,
            "mass_api": 21,
            "collision": connector_release.collider_row_count + 19,
            "collision_group": 28,
            "fixed_joint": 4,
        },
        f"composed physical inventory changed: {counts}",
    )
    _require(world.is_playing() is False, "timeline started during A2 readback")
    return {
        "status": "PASSED",
        "contract_revision": model.document["identity"]["successor_revision"],
        "connector_asset_path": str(connector_path),
        "timeline_started": False,
        "downstream_authorized": False,
        "connector": {
            "collider_rows": connector_release.collider_row_count,
            "property_rows": connector_release.property_row_count,
            "family_pair_rows": connector_release.family_pair_row_count,
            "filter_source_rows": connector_release.filter_source_row_count,
        },
        "robot": {
            "collider_owners": len(robot_physical["collision_inventory"]),
            "mass_owners": len(robot_physical["mass_inventory"]),
            "joint_rows": len(robot_physical["joint_inventory"]),
            "self_collision_pairs": robot_physical["self_collision_pair_count"],
            "camera_prims": robot_physical["camera_prim_count"],
        },
        "composed_counts": counts,
        "fixture_load_path": authored["fixed_load_path"],
        "object_pose_writes_after_start": authored["object_pose_writes_after_start"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _arguments(argv)
    portable = Path(arguments.kit_portable_root).expanduser().resolve()
    if not portable.is_relative_to(Path("/tmp")):
        raise ValueError("Kit portable root must be below /tmp")
    portable.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("WARP_CACHE_PATH", str(portable / "warp-cache"))
    sys.argv.extend(["--portable-root", str(portable)])
    from isaacsim import SimulationApp

    application = SimulationApp(
        {
            "headless": True,
            "hide_ui": True,
            "renderer": "Minimal",
            "multi_gpu": False,
            "fast_shutdown": True,
            "enable_crashreporter": False,
        }
    )
    status = 1
    try:
        local_authorization = None
        authorized_model = None
        authorized_local_asset_path = None
        if arguments.authorized_local_candidate_result is not None:
            from kcg_connector.d38999_keyed_v2_physical_model_contract import (
                WORKSPACE_ROOT,
            )
            from kcg_connector.d38999_keyed_v3_physical_r12_contract import (
                candidate_model,
                load_r12_physical_model_contract,
            )
            from kcg_connector.d38999_r12_local_candidate import (
                authorize_task_r12_006b_local_candidate,
            )

            frozen_model = candidate_model(
                load_r12_physical_model_contract(arguments.model_contract), 2
            )
            local_authorization = authorize_task_r12_006b_local_candidate(
                model=frozen_model,
                result_path=arguments.authorized_local_candidate_result,
                expected_result_sha256=(
                    arguments.authorized_local_candidate_result_sha256
                ),
                scene_config=arguments.scene_config,
                repository_root=WORKSPACE_ROOT,
            )
            authorized_model = local_authorization.model
            authorized_local_asset_path = (
                local_authorization.candidate_asset_relative_path
            )
        report = _run_validation(
            arguments,
            authorized_model=authorized_model,
            authorized_local_asset_path=authorized_local_asset_path,
        )
        if local_authorization is not None:
            report["local_candidate_authorization"] = (
                local_authorization.evidence()
            )
        _emit(json.dumps(report, ensure_ascii=False, sort_keys=True))
        _emit(PASS_BANNER)
        status = 0
    except BaseException as error:
        report = {
            "status": "FAILED",
            "error_type": type(error).__name__,
            "error": str(error),
            "downstream_authorized": False,
        }
        _emit(json.dumps(report, ensure_ascii=False, sort_keys=True))
        _emit(FAIL_BANNER)
        traceback.print_exc()
    finally:
        application.close()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
