#!/usr/bin/env python3

"""Audit authored, world, cooked, and runtime collision facts for A1 V2.

This is a diagnostic Isaac process.  Contact/collider truth is written only to
the post-hoc report and never participates in a controller.  The source USD is
opened read-only and is never saved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import traceback
from typing import Any, Iterable, Mapping, Sequence

import yaml


SCHEMA_VERSION = "kcg_d38999_multilayer_collision_audit_v2"
TASK_ID = "DYN-A1-EVENT-ONSET-CALIBRATION-V2"
HYPOTHESIS_ID = "A1-V2-H1-COOKED-SURFACE-AND-CONTACT-MARGIN-ONSET"
OUTPUT_RELATIVE = Path(
    "artifacts/agent_control/tasks/D38999-AUTONOMOUS-DYNAMIC-CLOSEOUT-V2/"
    "DYN-A1-EVENT-ONSET-CALIBRATION-V2/COLLISION_AUDIT_PRE_OFFSET"
)
CONTRACT_RELATIVE = Path(
    "src/kcg_connector/config/d38999_master_model_contract_v1.yaml"
)
PHYSICAL_CONTRACT_RELATIVE = Path(
    "src/kcg_connector/config/d38999_keyed_v3_physical_model_contract_r12_v1.yaml"
)
OVERRIDES_RELATIVE = Path(
    "src/kcg_connector/config/d38999_assembly_control_authorized_overrides_v2.yaml"
)
MODEL_RELATIVE = Path(
    "artifacts/kcg_connector/isaac/d38999_multilayer_v1/"
    "D38999_ASSEMBLY_CONTROL_V1.usda"
)
MAPPING_RELATIVE = Path(
    "artifacts/kcg_connector/isaac/d38999_multilayer_v1/MODEL_MAPPING.json"
)
EXPECTED_SHA256 = {
    "contract": "57010b3c6f8d2214712427193ef7b9b57ede3089a882aa88033f89774215c68b",
    "physical_contract": "6068066a2ac0339fa83caf2cc0c28050e76ed7e56e960da1b29e121a083b650e",
    "authorized_overrides": "392766e8eceb85a3c910b118c2ad998aef891a74e58c31cd94e383c9908535ce",
    "model": "d2e27acb3ccb8de6cf4ffad3d40940e3f8bcf2a4ba30dc223a8f2be11fdf9ea0",
    "mapping": "a31d4c0e7cb911f0371c257b0784506388f0f52d1d07401c1b1c2239fa47a783",
}
ROOT = "/World/D38999MultilayerV1/D38999_ASSEMBLY_CONTROL_V1"
PAIR_ROOT = ROOT + "/D38999Pair"
FIXED_PATH = PAIR_ROOT + "/FixedReceptacle"
BODY_PATH = PAIR_ROOT + "/LoosePlug/BodyAssembly"
NUT_PATH = PAIR_ROOT + "/LoosePlug/CouplingNut"
START_SEPARATION_M = 0.0055
TARGET_ROLES = {
    "continuous_shell_and_guidance",
    "continuous_keyway_wall",
    "continuous_polarizing_key",
    "continuous_real_metal_stop_fixed",
    "continuous_real_metal_stop_plug",
}
EXPECTED_TARGET_COUNTS = {"Mesh": 253, "Cube": 15, "Cylinder": 1}


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--physical-contract", required=True)
    parser.add_argument("--authorized-overrides", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--kit-portable-root", default=None)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def _repository() -> Path:
    return Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _frozen_path(raw: str, relative: Path, expected_sha: str, label: str) -> Path:
    actual = Path(raw).expanduser().resolve()
    expected = (_repository() / relative).resolve()
    if actual != expected:
        raise PermissionError(f"{label} path differs: {actual} != {expected}")
    if not actual.is_file():
        raise FileNotFoundError(actual)
    actual_sha = _sha256(actual)
    if actual_sha != expected_sha:
        raise PermissionError(
            f"{label} SHA-256 differs: {actual_sha} != {expected_sha}"
        )
    return actual


def _authorize(arguments: argparse.Namespace) -> dict[str, Any]:
    repository = _repository()
    output = Path(arguments.output_dir).expanduser().resolve()
    expected_output = (repository / OUTPUT_RELATIVE).resolve()
    if output != expected_output:
        raise PermissionError(f"output path differs: {output} != {expected_output}")
    state = json.loads(
        (repository / "artifacts/agent_control/MASTER_STATE.json").read_text(
            encoding="utf-8"
        )
    )
    queue = yaml.safe_load(
        (repository / "artifacts/agent_control/WORK_QUEUE.yaml").read_text(
            encoding="utf-8"
        )
    )
    v2 = state.get("autonomous_dynamic_closeout_v2", {})
    expected_state = {
        "task_id": "D38999-AUTONOMOUS-DYNAMIC-CLOSEOUT-V2",
        "phase": "DYN_A1_EVENT_ONSET_CALIBRATION_V2",
        "status": "VALIDATING",
    }
    state_mismatches = {
        key: {"actual": state.get(key), "expected": expected}
        for key, expected in expected_state.items()
        if state.get(key) != expected
    }
    if v2.get("current_node") != TASK_ID:
        state_mismatches["current_node"] = {
            "actual": v2.get("current_node"),
            "expected": TASK_ID,
        }
    if v2.get("only_stop_status") != "PARKED_EXTERNAL":
        state_mismatches["only_stop_status"] = {
            "actual": v2.get("only_stop_status"),
            "expected": "PARKED_EXTERNAL",
        }
    generation = v2.get("asset_generation", {})
    if (
        generation.get("status") != "STATIC_PASS"
        or generation.get("assembly_control_sha256_after")
        != EXPECTED_SHA256["model"]
    ):
        state_mismatches["asset_generation"] = {
            "actual": generation,
            "expected_status": "STATIC_PASS",
            "expected_model_sha256": EXPECTED_SHA256["model"],
        }
    if state_mismatches:
        raise PermissionError(f"MASTER_STATE V2 guard failed: {state_mismatches}")
    if queue.get("status") != "VALIDATING" or queue.get("current_task") != TASK_ID:
        raise PermissionError("WORK_QUEUE does not authorize this V2 A1 audit")

    paths = {
        "contract": _frozen_path(
            arguments.contract,
            CONTRACT_RELATIVE,
            EXPECTED_SHA256["contract"],
            "master contract",
        ),
        "physical_contract": _frozen_path(
            arguments.physical_contract,
            PHYSICAL_CONTRACT_RELATIVE,
            EXPECTED_SHA256["physical_contract"],
            "physical model contract",
        ),
        "authorized_overrides": _frozen_path(
            arguments.authorized_overrides,
            OVERRIDES_RELATIVE,
            EXPECTED_SHA256["authorized_overrides"],
            "authorized overrides",
        ),
        "model": _frozen_path(
            arguments.model, MODEL_RELATIVE, EXPECTED_SHA256["model"], "model"
        ),
        "mapping": _frozen_path(
            arguments.mapping,
            MAPPING_RELATIVE,
            EXPECTED_SHA256["mapping"],
            "mapping",
        ),
    }
    master = yaml.safe_load(paths["contract"].read_text(encoding="utf-8"))
    physical = yaml.safe_load(
        paths["physical_contract"].read_text(encoding="utf-8")
    )
    offsets = physical["a2_collision_authoring_blueprint"]["offset_classes"][
        "fine_connector"
    ]
    expected_offsets = {"contactOffset_m": 1.0e-5, "restOffset_m": 0.0}
    if offsets != expected_offsets:
        raise PermissionError(
            f"fine_connector offset contract differs: {offsets} != {expected_offsets}"
        )
    return {
        "output": output,
        "paths": paths,
        "master": master,
        "physical": physical,
        "expected_offsets": expected_offsets,
    }


def _json_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "NEGATIVE_INFINITY" if value < 0.0 else "POSITIVE_INFINITY"
        return value
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return str(value)
    return _json_scalar(converted)


def _bounds(points: Iterable[Sequence[float]]) -> dict[str, list[float]]:
    rows = [[float(value) for value in point] for point in points]
    if not rows:
        raise ValueError("cannot compute bounds of an empty point set")
    minimum = [min(point[axis] for point in rows) for axis in range(3)]
    maximum = [max(point[axis] for point in rows) for axis in range(3)]
    return {
        "minimum": minimum,
        "maximum": maximum,
        "extent": [maximum[axis] - minimum[axis] for axis in range(3)],
        "center": [0.5 * (minimum[axis] + maximum[axis]) for axis in range(3)],
    }


def _maximum_bound_error(left: Mapping[str, Sequence[float]], right: Mapping[str, Sequence[float]]) -> float:
    return max(
        abs(float(left[side][axis]) - float(right[side][axis]))
        for side in ("minimum", "maximum")
        for axis in range(3)
    )


def _matrix_rows(matrix: Any) -> list[list[float]]:
    return [[float(matrix[row][column]) for column in range(4)] for row in range(4)]


def _transform_points(matrix: Any, points: Iterable[Sequence[float]], gf: Any) -> list[list[float]]:
    transformed: list[list[float]] = []
    for point in points:
        value = matrix.Transform(gf.Vec3d(*(float(item) for item in point)))
        transformed.append([float(value[axis]) for axis in range(3)])
    return transformed


def _authored_local_vertices(prim: Any, usd_geom: Any) -> tuple[list[list[float]], str]:
    type_name = prim.GetTypeName()
    if type_name == "Mesh":
        values = usd_geom.Mesh(prim).GetPointsAttr().Get() or []
        return (
            [[float(point[axis]) for axis in range(3)] for point in values],
            "authored_mesh_points",
        )
    if type_name == "Cube":
        size = float(usd_geom.Cube(prim).GetSizeAttr().Get())
        half = 0.5 * size
        return (
            [
                [sx * half, sy * half, sz * half]
                for sx in (-1.0, 1.0)
                for sy in (-1.0, 1.0)
                for sz in (-1.0, 1.0)
            ],
            "analytic_cube_vertices_derived_from_authored_size",
        )
    if type_name == "Cylinder":
        cylinder = usd_geom.Cylinder(prim)
        radius = float(cylinder.GetRadiusAttr().Get())
        half = 0.5 * float(cylinder.GetHeightAttr().Get())
        points = []
        for index in range(64):
            angle = 2.0 * math.pi * index / 64.0
            for z_value in (-half, half):
                points.append(
                    [radius * math.cos(angle), radius * math.sin(angle), z_value]
                )
        return points, "analytic_cylinder_boundary_samples_from_authored_parameters"
    raise ValueError(f"unsupported target collider type: {type_name}")


def _attribute_readback(prim: Any, attribute_name: str, schema_attribute: Any) -> dict[str, Any]:
    direct = prim.GetAttribute(attribute_name)
    direct_valid = bool(direct)
    direct_authored = bool(direct_valid and direct.HasAuthoredValueOpinion())
    schema_valid = bool(schema_attribute)
    raw = schema_attribute.Get() if schema_valid else None
    if direct_authored:
        source = "USD_AUTHORED"
    elif raw is None:
        source = "PHYSX_INTERNAL_DEFAULT_NOT_EXPOSED_BY_SCHEMA_READBACK"
    elif isinstance(_json_scalar(raw), str):
        source = "PHYSX_SCHEMA_AUTO_SENTINEL"
    else:
        source = "PHYSX_SCHEMA_FALLBACK"
    return {
        "attribute_name": attribute_name,
        "direct_attribute_valid": direct_valid,
        "has_authored_value_opinion": direct_authored,
        "schema_attribute_valid": schema_valid,
        "runtime_schema_readback": _json_scalar(raw),
        "readback_source": source,
    }


def _offset_readback(prim: Any, physx_schema: Any) -> dict[str, Any]:
    api = physx_schema.PhysxCollisionAPI(prim)
    return {
        "contact_offset": _attribute_readback(
            prim,
            "physxCollision:contactOffset",
            api.GetContactOffsetAttr(),
        ),
        "rest_offset": _attribute_readback(
            prim,
            "physxCollision:restOffset",
            api.GetRestOffsetAttr(),
        ),
    }


def _owner_path(path: str) -> str:
    if path.startswith(FIXED_PATH + "/"):
        return FIXED_PATH
    if path.startswith(BODY_PATH + "/"):
        return BODY_PATH
    if path.startswith(NUT_PATH + "/"):
        return NUT_PATH
    raise ValueError(f"target collider has no recognized owner: {path}")


def _family(path: str, role: str) -> str:
    if role == "continuous_shell_and_guidance":
        return "fixed_continuous_guide" if path.startswith(FIXED_PATH) else "plug_continuous_guide"
    if role == "continuous_keyway_wall":
        return "ten_keyway_walls"
    if role == "continuous_polarizing_key":
        return "five_plug_keys"
    if role == "continuous_real_metal_stop_fixed":
        return "fixed_metal_stop"
    if role == "continuous_real_metal_stop_plug":
        return "plug_metal_stop"
    return role


def _set_initial_translation(stage: Any, path: str, usd_geom: Any, gf: Any) -> None:
    prim = stage.GetPrimAtPath(path)
    if not prim:
        raise RuntimeError(f"missing rigid body {path}")
    xformable = usd_geom.Xformable(prim)
    if xformable.GetOrderedXformOps():
        raise RuntimeError(f"unexpected pre-existing transform stack at {path}")
    xformable.AddTranslateOp().Set(gf.Vec3d(0.0, 0.0, -START_SEPARATION_M))


def _body_positions(stage: Any, usd_geom: Any) -> dict[str, list[float]]:
    cache = usd_geom.XformCache()
    result = {}
    for label, path in {
        "fixed_receptacle": FIXED_PATH,
        "body_assembly": BODY_PATH,
        "coupling_nut": NUT_PATH,
    }.items():
        value = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path)).ExtractTranslation()
        result[label] = [float(value[axis]) for axis in range(3)]
    return result


def _datum_separation(positions: Mapping[str, Sequence[float]]) -> float:
    return float(positions["fixed_receptacle"][2]) - float(
        positions["body_assembly"][2]
    )


def _inventory(stage: Any, usd_geom: Any, usd_physics: Any, physx_schema: Any, gf: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    xforms = usd_geom.XformCache()
    purposes = [
        usd_geom.Tokens.default_,
        usd_geom.Tokens.render,
        usd_geom.Tokens.proxy,
        usd_geom.Tokens.guide,
    ]
    bboxes = usd_geom.BBoxCache(0.0, purposes, useExtentsHint=False)
    owner_matrices = {
        path: xforms.GetLocalToWorldTransform(stage.GetPrimAtPath(path))
        for path in (FIXED_PATH, BODY_PATH, NUT_PATH)
    }
    rows: list[dict[str, Any]] = []
    type_counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    for prim in stage.Traverse():
        if not prim.HasAPI(usd_physics.CollisionAPI):
            continue
        role_attr = prim.GetAttribute("kcg:collisionRole")
        role = str(role_attr.Get()) if role_attr and role_attr.Get() is not None else "UNLABELED"
        if role not in TARGET_ROLES:
            continue
        path = str(prim.GetPath())
        owner = _owner_path(path)
        family = _family(path, role)
        local_vertices, local_source = _authored_local_vertices(prim, usd_geom)
        matrix = xforms.GetLocalToWorldTransform(prim)
        world_vertices = _transform_points(matrix, local_vertices, gf)
        owner_inverse = owner_matrices[owner].GetInverse()
        owner_local_vertices = _transform_points(owner_inverse, world_vertices, gf)
        aligned = bboxes.ComputeWorldBound(prim).ComputeAlignedRange()
        bbox_world = {
            "minimum": [float(aligned.GetMin()[axis]) for axis in range(3)],
            "maximum": [float(aligned.GetMax()[axis]) for axis in range(3)],
        }
        bbox_world["extent"] = [
            bbox_world["maximum"][axis] - bbox_world["minimum"][axis]
            for axis in range(3)
        ]
        bbox_world["center"] = [
            0.5 * (bbox_world["minimum"][axis] + bbox_world["maximum"][axis])
            for axis in range(3)
        ]
        approximation = prim.GetAttribute("physics:approximation")
        minimum_thickness = prim.GetAttribute(
            "physxConvexHullCollision:minThickness"
        )
        collision_enabled = prim.GetAttribute("physics:collisionEnabled")
        trace = prim.GetAttribute("kcg:traceLabel")
        type_name = prim.GetTypeName()
        type_counts[type_name] = type_counts.get(type_name, 0) + 1
        family_counts[family] = family_counts.get(family, 0) + 1
        rows.append(
            {
                "prim_path": path,
                "owner_path": owner,
                "family": family,
                "collision_role": role,
                "trace_label": str(trace.Get()) if trace and trace.Get() is not None else None,
                "type_name": type_name,
                "collision_enabled": bool(collision_enabled.Get()) if collision_enabled else True,
                "collision_approximation": str(approximation.Get()) if approximation and approximation.Get() is not None else "analytic_primitive",
                "authored_min_thickness_local_units": _json_scalar(
                    minimum_thickness.Get()
                    if minimum_thickness and minimum_thickness.Get() is not None
                    else None
                ),
                "authored_local_vertex_source": local_source,
                "authored_local_vertex_count": len(local_vertices),
                "authored_local_vertices": local_vertices,
                "authored_local_aabb_numeric": _bounds(local_vertices),
                "local_to_world_matrix": _matrix_rows(matrix),
                "world_center_m": _bounds(world_vertices)["center"],
                "world_aabb_from_vertices_m": _bounds(world_vertices),
                "world_aabb_from_bbox_cache_m": bbox_world,
                "owner_local_aabb_m": _bounds(owner_local_vertices),
                "offset_readback_before_world_reset": _offset_readback(
                    prim, physx_schema
                ),
            }
        )
    rows.sort(key=lambda row: row["prim_path"])
    return rows, {"type_counts": type_counts, "family_counts": family_counts}


def _union_bounds(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, list[float]]:
    return _bounds(
        point
        for row in rows
        for point in (
            row[key]["minimum"],
            row[key]["maximum"],
        )
    )


def _run(frozen: Mapping[str, Any], application: Any) -> dict[str, Any]:
    import carb.logging
    from isaacsim.core.api import World
    from isaacsim.core.utils.stage import get_current_stage
    from omni.physx import get_physx_cooking_interface
    from omni.physx.bindings._physx import PhysxCollisionRepresentationResult
    import omni.usd
    from pxr import Gf, PhysicsSchemaTools, PhysxSchema, UsdGeom, UsdPhysics, UsdUtils

    World.clear_instance()
    context = omni.usd.get_context()
    if context.get_stage() is not None:
        context.close_stage()
        application.update()
    if context.open_stage(str(frozen["paths"]["model"])) is not True:
        raise RuntimeError("failed to open V2 assembly-control asset")
    for _ in range(3):
        application.update()
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=1.0 / 240.0,
        rendering_dt=1.0 / 60.0,
        backend="numpy",
        device="cpu",
    )
    stage = get_current_stage()
    _set_initial_translation(stage, BODY_PATH, UsdGeom, Gf)
    _set_initial_translation(stage, NUT_PATH, UsdGeom, Gf)

    log_rows: list[dict[str, Any]] = []
    logging = carb.logging.acquire_logging()

    def on_log(source: str, level: int, filename: str, line_number: int, message: str) -> None:
        text = str(message)
        lowered = text.lower()
        if any(
            token in lowered
            for token in (
                "adjusted the thickness",
                "failed to cook",
                "physicsusd",
                "solver error",
            )
        ):
            log_rows.append(
                {
                    "source": str(source),
                    "level": int(level),
                    "filename": str(filename),
                    "line_number": int(line_number),
                    "message": text,
                }
            )

    logger_handle = logging.add_logger(on_log)
    try:
        rows, counts = _inventory(stage, UsdGeom, UsdPhysics, PhysxSchema, Gf)
        if counts["type_counts"] != EXPECTED_TARGET_COUNTS:
            raise RuntimeError(
                f"target collider type counts differ: {counts['type_counts']} != {EXPECTED_TARGET_COUNTS}"
            )
        before_positions = _body_positions(stage, UsdGeom)
        cooking = get_physx_cooking_interface()
        cooking.release_local_mesh_cache()
        stage_id = UsdUtils.StageCache.Get().Insert(stage).ToLongInt()
        mesh_rows = [row for row in rows if row["type_name"] == "Mesh"]
        cooking_failures: list[dict[str, Any]] = []
        maximum_cooked_bound_error_m = 0.0
        vertex_count_mismatch_count = 0
        xforms = UsdGeom.XformCache()
        owner_matrices = {
            path: xforms.GetLocalToWorldTransform(stage.GetPrimAtPath(path))
            for path in (FIXED_PATH, BODY_PATH, NUT_PATH)
        }
        rows_by_path = {row["prim_path"]: row for row in rows}
        for index, row in enumerate(mesh_rows, start=1):
            path = row["prim_path"]
            callback: dict[str, Any] = {}

            def on_result(result: Any, convexes: list[Any]) -> None:
                callback["result"] = result
                callback["convexes"] = convexes

            cooking.request_convex_collision_representation(
                stage_id=stage_id,
                collision_prim_id=PhysicsSchemaTools.sdfPathToInt(path),
                run_asynchronously=False,
                on_result=on_result,
            )
            valid = (
                callback.get("result")
                == PhysxCollisionRepresentationResult.RESULT_VALID
            )
            convexes = callback.get("convexes", [])
            if not valid or len(convexes) != 1:
                failure = {
                    "prim_path": path,
                    "result": str(callback.get("result")),
                    "convex_count": len(convexes),
                }
                cooking_failures.append(failure)
                row["cooked"] = {"status": "FAILED", **failure}
                continue
            cooked_local = [
                [float(vertex.x), float(vertex.y), float(vertex.z)]
                for vertex in convexes[0].vertices
            ]
            matrix = xforms.GetLocalToWorldTransform(stage.GetPrimAtPath(path))
            cooked_world = _transform_points(matrix, cooked_local, Gf)
            owner_local = _transform_points(
                owner_matrices[row["owner_path"]].GetInverse(), cooked_world, Gf
            )
            cooked_world_bounds = _bounds(cooked_world)
            error_m = _maximum_bound_error(
                row["world_aabb_from_vertices_m"], cooked_world_bounds
            )
            maximum_cooked_bound_error_m = max(
                maximum_cooked_bound_error_m, error_m
            )
            mismatch = len(cooked_local) != int(row["authored_local_vertex_count"])
            vertex_count_mismatch_count += int(mismatch)
            row["cooked"] = {
                "status": "VALID",
                "convex_count": 1,
                "vertex_count": len(cooked_local),
                "polygon_count": len(convexes[0].polygons),
                "local_aabb_numeric": _bounds(cooked_local),
                "world_aabb_m": cooked_world_bounds,
                "owner_local_aabb_m": _bounds(owner_local),
                "maximum_world_bound_error_m": error_m,
                "vertex_count_matches_authored": not mismatch,
            }
            if index % 25 == 0 or index == len(mesh_rows):
                os.write(
                    2,
                    f"A1_V2_COLLISION_AUDIT_HEARTBEAT cooked={index}/{len(mesh_rows)}\n".encode(),
                )

        for row in rows:
            if row["type_name"] != "Mesh":
                row["cooked"] = {
                    "status": "ANALYTIC_RUNTIME_SHAPE",
                    "convex_count": None,
                    "vertex_count": None,
                    "polygon_count": None,
                    "world_aabb_m": row["world_aabb_from_vertices_m"],
                    "owner_local_aabb_m": row["owner_local_aabb_m"],
                    "maximum_world_bound_error_m": 0.0,
                }

        world.get_physics_context().set_gravity(0.0)
        world.reset()
        after_reset_positions = _body_positions(stage, UsdGeom)
        for row in rows:
            prim = stage.GetPrimAtPath(row["prim_path"])
            row["offset_readback_after_world_reset"] = _offset_readback(
                prim, PhysxSchema
            )
        world.step(render=False)
        after_first_step_positions = _body_positions(stage, UsdGeom)
        world.stop()
    finally:
        logging.remove_logger(logger_handle)

    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_family.setdefault(row["family"], []).append(row)
    family_summary = {
        family: {
            "collider_count": len(family_rows),
            "type_counts": {
                type_name: sum(
                    row["type_name"] == type_name for row in family_rows
                )
                for type_name in sorted({row["type_name"] for row in family_rows})
            },
            "owner_local_union_aabb_m": _union_bounds(
                family_rows, "owner_local_aabb_m"
            ),
            "world_union_aabb_m": _union_bounds(
                family_rows, "world_aabb_from_vertices_m"
            ),
        }
        for family, family_rows in sorted(by_family.items())
    }
    key_zero = next(
        row for row in rows if row["prim_path"].endswith("/PolarizingKeys/Key_0")
    )
    key_zero_center = key_zero["owner_local_aabb_m"]["center"]
    key_zero_radius_m = math.hypot(key_zero_center[0], key_zero_center[1])
    physical = frozen["physical"]
    cooking_contract = physical["convex_cooking_representation"]
    bound_tolerance_m = float(cooking_contract["cooked_bounds_abs_tolerance_m"])
    all_offsets_authored = all(
        row["offset_readback_after_world_reset"][name][
            "has_authored_value_opinion"
        ]
        for row in rows
        for name in ("contact_offset", "rest_offset")
    )
    if cooking_failures or maximum_cooked_bound_error_m > bound_tolerance_m:
        classification = "COOKED_GEOMETRY_DRIFT_OR_FAILURE_CONFIRMED"
    elif not all_offsets_authored:
        classification = "COOKED_GEOMETRY_SUPPORTED_EXPLICIT_OFFSETS_MISSING"
    else:
        classification = "COOKED_GEOMETRY_AND_EXPLICIT_OFFSETS_SUPPORTED"
    physicsusd_errors = [
        row for row in log_rows if "physicsusd" in row["message"].lower()
    ]
    solver_errors = [
        row for row in log_rows if "solver error" in row["message"].lower()
    ]
    adjusted = [
        row for row in log_rows if "adjusted the thickness" in row["message"].lower()
    ]
    cpu_fallback = [
        row for row in log_rows if "failed to cook" in row["message"].lower()
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "COMPLETED",
        "classification": classification,
        "target_collider_count": len(rows),
        "target_counts": counts,
        "expected_target_type_counts": EXPECTED_TARGET_COUNTS,
        "family_summary": family_summary,
        "key_0_owner_local_center_radius_m": key_zero_radius_m,
        "key_0_owner_local_center_radius_mm": 1000.0 * key_zero_radius_m,
        "cooking": {
            "mesh_count": len(mesh_rows),
            "failure_count": len(cooking_failures),
            "failures": cooking_failures,
            "maximum_cooked_world_bound_error_m": maximum_cooked_bound_error_m,
            "bound_tolerance_m": bound_tolerance_m,
            "vertex_count_mismatch_count": vertex_count_mismatch_count,
            "adjusted_thickness_warning_count": len(adjusted),
            "cpu_fallback_message_count": len(cpu_fallback),
        },
        "offset_contract": frozen["expected_offsets"],
        "all_target_offsets_explicitly_authored": all_offsets_authored,
        "body_positions": {
            "before_world_reset": before_positions,
            "after_world_reset": after_reset_positions,
            "after_first_explicit_step": after_first_step_positions,
        },
        "datum_separation_m": {
            "before_world_reset": _datum_separation(before_positions),
            "after_world_reset": _datum_separation(after_reset_positions),
            "after_first_explicit_step": _datum_separation(
                after_first_step_positions
            ),
        },
        "colliders": rows,
        "runtime_log_evidence": {
            "retained_messages": log_rows,
            "physicsusd_error_count": len(physicsusd_errors),
            "solver_error_count": len(solver_errors),
        },
        "initial_pose_writes_before_physics_start_count": 2,
        "object_pose_write_after_physics_start_count": 0,
        "explicit_physics_step_count": 1,
        "isaac_process_count": 1,
        "simulation_started": True,
        "diagnostic_only": True,
        "control_consumed_object_truth": False,
        "control_consumed_contact_names": False,
        "control_consumed_contact_normals": False,
        "control_consumed_event_truth": False,
        "dynamic_pass_claimed": False,
        "formal_p1_pass_claimed": False,
        "formal_r12_generated": False,
        "hardware_authorized": False,
        "source_asset_written": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _arguments(argv)
    frozen = _authorize(arguments)
    if arguments.preflight_only:
        print(
            json.dumps(
                {
                    "task_id": TASK_ID,
                    "hypothesis_id": HYPOTHESIS_ID,
                    "status": "PREFLIGHT_PASS",
                    "input_sha256": EXPECTED_SHA256,
                    "output": str(frozen["output"]),
                    "expected_target_type_counts": EXPECTED_TARGET_COUNTS,
                    "expected_offset_contract": frozen["expected_offsets"],
                    "simulation_will_start": False,
                    "dynamic_pass_claimed": False,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    output = frozen["output"]
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evidence directory: {output}")
    output.mkdir(parents=True, exist_ok=False)
    if arguments.kit_portable_root is None:
        portable = Path(tempfile.mkdtemp(prefix="kcg-a1-v2-collision-audit-", dir="/tmp"))
    else:
        portable = Path(arguments.kit_portable_root).expanduser().resolve()
        if not portable.is_relative_to(Path("/tmp")):
            raise ValueError("Kit portable root must be below /tmp")
        portable.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("WARP_CACHE_PATH", str(portable / "warp-cache"))
    sys.argv = [sys.argv[0], "--portable-root", str(portable)]
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
    exit_code = 1
    try:
        report = _run(frozen, application)
        report["input_sha256"] = dict(EXPECTED_SHA256)
        report["kit_portable_root"] = str(portable)
        report["post_run_sha256"] = {
            label: _sha256(path) for label, path in frozen["paths"].items()
        }
        report["frozen_inputs_unchanged"] = (
            report["post_run_sha256"] == EXPECTED_SHA256
        )
        if not report["frozen_inputs_unchanged"]:
            raise RuntimeError("frozen inputs changed during collision audit")
        exit_code = 0
    except BaseException as error:
        traceback.print_exc()
        report = {
            "schema_version": SCHEMA_VERSION,
            "task_id": TASK_ID,
            "hypothesis_id": HYPOTHESIS_ID,
            "status": "ERROR",
            "classification": "DIAGNOSTIC_PROGRAM_ERROR",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "diagnostic_only": True,
            "dynamic_pass_claimed": False,
            "formal_p1_pass_claimed": False,
            "formal_r12_generated": False,
            "hardware_authorized": False,
            "object_pose_write_after_physics_start_count": 0,
            "source_asset_written": False,
        }
    finally:
        (output / "report.json").write_text(
            json.dumps(
                report,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        application.close()
    summary = {
        key: report.get(key)
        for key in (
            "status",
            "classification",
            "target_collider_count",
            "key_0_owner_local_center_radius_mm",
            "cooking",
            "all_target_offsets_explicitly_authored",
            "datum_separation_m",
            "runtime_log_evidence",
            "dynamic_pass_claimed",
        )
    }
    print(json.dumps(summary, allow_nan=False, ensure_ascii=False, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
