#!/usr/bin/env python3
"""Build the one authorized TASK-R12-006B local geometry candidate."""

from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[2]
SOURCE_ASSET_REL = Path(
    "artifacts/kcg_connector/isaac/keyed_v3_physical_r12/candidates/"
    "r12_candidate_02/r12_candidate_02.usda"
)
SOURCE_SCENE_REL = SOURCE_ASSET_REL.parent / "scene.yaml"
CANDIDATE_DIR_REL = Path("artifacts/agent_control/tasks/TASK-R12-006B/candidate")
CANDIDATE_ASSET_REL = CANDIDATE_DIR_REL / "task_r12_006b_local_candidate_01.usda"
CANDIDATE_SCENE_REL = CANDIDATE_DIR_REL / "scene.yaml"
RESULT_REL = CANDIDATE_DIR_REL / "CANDIDATE_BUILD_RESULT.json"
AUTHORIZED_F_DIAMETER_MM = 1.32
EXPECTED_STAGE_METERS_PER_UNIT = 1.0
EXPECTED_TARGET_POINT_SCALE = 0.001
TARGET_METERS_PER_LOCAL_UNIT = (
    EXPECTED_STAGE_METERS_PER_UNIT * EXPECTED_TARGET_POINT_SCALE
)
GEOMETRY_AUDIT_TOLERANCE_M = 3.0e-8
OLD_INNER_RADIUS_M = 0.000640
NEW_INNER_RADIUS_M = 0.000660
OUTER_RADIUS_M = 0.001250
OLD_INNER_RADIUS_LOCAL = OLD_INNER_RADIUS_M / TARGET_METERS_PER_LOCAL_UNIT
NEW_INNER_RADIUS_LOCAL = NEW_INNER_RADIUS_M / TARGET_METERS_PER_LOCAL_UNIT
OUTER_RADIUS_LOCAL = OUTER_RADIUS_M / TARGET_METERS_PER_LOCAL_UNIT
EXPECTED_BAND_PRIMS = 61 * 24
EXPECTED_MODIFIED_PRIMS = 2 * EXPECTED_BAND_PRIMS
EXPECTED_MODIFIED_POINTS = EXPECTED_BAND_PRIMS * (2 + 4)
TARGET_RE = re.compile(
    r"^/World/D38999Shell25JKeyedPhysicalV3/LoosePlug/BodyAssembly/"
    r"Contacts/Socket_[^/]+/HardEntry/Band_(01|02)/Wedge_[0-9]{2}$"
)
PAIR_INDEX = {0: 1, 3: 2, 4: 5, 7: 6}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument(
        "--socket-entry-f-diameter-mm",
        type=float,
        default=AUTHORIZED_F_DIAMETER_MM,
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distance_xy(left: Any, right: Any) -> float:
    return math.hypot(float(left[0]) - float(right[0]), float(left[1]) - float(right[1]))


def _distance_xyz(left: Any, right: Any) -> float:
    return math.sqrt(sum((float(left[index]) - float(right[index])) ** 2 for index in range(3)))


def _new_inner_point(inner: Any, outer: Any, Gf: Any) -> Any:
    delta_x = float(outer[0]) - float(inner[0])
    delta_y = float(outer[1]) - float(inner[1])
    distance = math.hypot(delta_x, delta_y)
    expected = OUTER_RADIUS_LOCAL - OLD_INNER_RADIUS_LOCAL
    local_tolerance = GEOMETRY_AUDIT_TOLERANCE_M / TARGET_METERS_PER_LOCAL_UNIT
    if not math.isclose(distance, expected, rel_tol=0.0, abs_tol=local_tolerance):
        raise ValueError(f"unexpected inner-to-outer radial gap: {distance}")
    unit_x = delta_x / distance
    unit_y = delta_y / distance
    center_x = float(inner[0]) - OLD_INNER_RADIUS_LOCAL * unit_x
    center_y = float(inner[1]) - OLD_INNER_RADIUS_LOCAL * unit_y
    return Gf.Vec3f(
        center_x + NEW_INNER_RADIUS_LOCAL * unit_x,
        center_y + NEW_INNER_RADIUS_LOCAL * unit_y,
        float(inner[2]),
    )


def _target_indices(band: str) -> tuple[int, ...]:
    if band == "01":
        return (4, 7)
    if band == "02":
        return (0, 3, 4, 7)
    raise ValueError(f"unauthorized band: {band}")


def _inventory(stage: Any) -> dict[str, Any]:
    type_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    prim_count = 0
    for prim in stage.TraverseAll():
        prim_count += 1
        type_counts[prim.GetTypeName() or "UNSPECIFIED"] += 1
        family = prim.GetAttribute("kcg:primitiveFamily")
        if family and family.HasAuthoredValueOpinion():
            family_counts[str(family.Get())] += 1
    return {
        "prim_count": prim_count,
        "type_counts": dict(sorted(type_counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
    }


def _property_value(property_object: Any) -> Any:
    if hasattr(property_object, "GetTargets"):
        return tuple(str(path) for path in property_object.GetTargets())
    return property_object.Get()


def _assert_non_target_properties_unchanged(source_stage: Any, candidate_stage: Any) -> int:
    source_paths = [str(prim.GetPath()) for prim in source_stage.TraverseAll()]
    candidate_paths = [str(prim.GetPath()) for prim in candidate_stage.TraverseAll()]
    if source_paths != candidate_paths:
        raise ValueError("candidate prim path inventory differs from candidate2")
    compared = 0
    for path in source_paths:
        source = source_stage.GetPrimAtPath(path)
        candidate = candidate_stage.GetPrimAtPath(path)
        if source.GetTypeName() != candidate.GetTypeName():
            raise ValueError(f"type changed at {path}")
        source_names = sorted(prop.GetName() for prop in source.GetProperties())
        candidate_names = sorted(prop.GetName() for prop in candidate.GetProperties())
        if source_names != candidate_names:
            raise ValueError(f"property inventory changed at {path}")
        target_points = TARGET_RE.fullmatch(path) is not None
        for name in source_names:
            if target_points and name == "points":
                continue
            source_property = source.GetProperty(name)
            candidate_property = candidate.GetProperty(name)
            if _property_value(source_property) != _property_value(candidate_property):
                raise ValueError(f"non-target property changed: {path}.{name}")
            compared += 1
    return compared


def _assert_target_points(
    source_stage: Any, candidate_stage: Any, UsdGeom: Any
) -> tuple[int, int, list[str], dict[str, float]]:
    modified_prim_count = 0
    modified_point_count = 0
    sample_paths: list[str] = []
    source_cache = UsdGeom.XformCache()
    candidate_cache = UsdGeom.XformCache()
    source_meters_per_unit = UsdGeom.GetStageMetersPerUnit(source_stage)
    candidate_meters_per_unit = UsdGeom.GetStageMetersPerUnit(candidate_stage)
    if not math.isclose(
        source_meters_per_unit,
        candidate_meters_per_unit,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("candidate metersPerUnit differs from candidate2")
    local_deltas: list[float] = []
    world_deltas_m: list[float] = []
    source_gaps_local: list[float] = []
    candidate_gaps_local: list[float] = []
    for source_prim in source_stage.TraverseAll():
        path = str(source_prim.GetPath())
        match = TARGET_RE.fullmatch(path)
        if match is None:
            continue
        candidate_prim = candidate_stage.GetPrimAtPath(path)
        source_points = list(UsdGeom.Mesh(source_prim).GetPointsAttr().Get())
        candidate_points = list(UsdGeom.Mesh(candidate_prim).GetPointsAttr().Get())
        if len(source_points) != 8 or len(candidate_points) != 8:
            raise ValueError(f"unexpected point count at {path}")
        scale = source_prim.GetAttribute("xformOp:scale").Get()
        if scale is None or any(
            not math.isclose(
                float(component),
                EXPECTED_TARGET_POINT_SCALE,
                rel_tol=0.0,
                abs_tol=1.0e-10,
            )
            for component in scale
        ):
            raise ValueError(f"unexpected target mesh scale at {path}: {scale}")
        source_to_world = source_cache.GetLocalToWorldTransform(source_prim)
        candidate_to_world = candidate_cache.GetLocalToWorldTransform(candidate_prim)
        authorized = set(_target_indices(match.group(1)))
        for index, (before, after) in enumerate(zip(source_points, candidate_points)):
            if index in authorized:
                delta = _distance_xy(before, after)
                expected_delta = (
                    NEW_INNER_RADIUS_M - OLD_INNER_RADIUS_M
                ) / TARGET_METERS_PER_LOCAL_UNIT
                if not math.isclose(
                    delta,
                    expected_delta,
                    rel_tol=0.0,
                    abs_tol=GEOMETRY_AUDIT_TOLERANCE_M / TARGET_METERS_PER_LOCAL_UNIT,
                ):
                    raise ValueError(f"wrong radial delta at {path}[{index}]: {delta}")
                source_gap = _distance_xy(before, source_points[PAIR_INDEX[index]])
                candidate_gap = _distance_xy(after, candidate_points[PAIR_INDEX[index]])
                if not math.isclose(
                    source_gap,
                    OUTER_RADIUS_LOCAL - OLD_INNER_RADIUS_LOCAL,
                    rel_tol=0.0,
                    abs_tol=GEOMETRY_AUDIT_TOLERANCE_M / TARGET_METERS_PER_LOCAL_UNIT,
                ):
                    raise ValueError(f"wrong source radial gap at {path}[{index}]")
                if not math.isclose(
                    candidate_gap,
                    OUTER_RADIUS_LOCAL - NEW_INNER_RADIUS_LOCAL,
                    rel_tol=0.0,
                    abs_tol=GEOMETRY_AUDIT_TOLERANCE_M / TARGET_METERS_PER_LOCAL_UNIT,
                ):
                    raise ValueError(f"wrong candidate radial gap at {path}[{index}]")
                before_world = source_to_world.Transform(before)
                after_world = candidate_to_world.Transform(after)
                world_delta_m = (
                    _distance_xyz(before_world, after_world) * source_meters_per_unit
                )
                if not math.isclose(
                    world_delta_m,
                    NEW_INNER_RADIUS_M - OLD_INNER_RADIUS_M,
                    rel_tol=0.0,
                    abs_tol=GEOMETRY_AUDIT_TOLERANCE_M,
                ):
                    raise ValueError(
                        f"wrong world radial delta at {path}[{index}]: {world_delta_m}"
                    )
                if not math.isclose(float(before[2]), float(after[2]), abs_tol=1.0e-10):
                    raise ValueError(f"z changed at {path}[{index}]")
                local_deltas.append(delta)
                world_deltas_m.append(world_delta_m)
                source_gaps_local.append(source_gap)
                candidate_gaps_local.append(candidate_gap)
                modified_point_count += 1
            elif before != after:
                raise ValueError(f"unauthorized point changed at {path}[{index}]")
        modified_prim_count += 1
        if len(sample_paths) < 6:
            sample_paths.append(path)
    if modified_prim_count != EXPECTED_MODIFIED_PRIMS:
        raise ValueError(
            f"modified prim count {modified_prim_count} != {EXPECTED_MODIFIED_PRIMS}"
        )
    if modified_point_count != EXPECTED_MODIFIED_POINTS:
        raise ValueError(
            f"modified point count {modified_point_count} != {EXPECTED_MODIFIED_POINTS}"
        )
    audit = {
        "stage_meters_per_unit": source_meters_per_unit,
        "target_point_scale": EXPECTED_TARGET_POINT_SCALE,
        "local_delta_min": min(local_deltas),
        "local_delta_max": max(local_deltas),
        "world_delta_m_min": min(world_deltas_m),
        "world_delta_m_max": max(world_deltas_m),
        "source_gap_local_min": min(source_gaps_local),
        "source_gap_local_max": max(source_gaps_local),
        "candidate_gap_local_min": min(candidate_gaps_local),
        "candidate_gap_local_max": max(candidate_gaps_local),
    }
    return modified_prim_count, modified_point_count, sample_paths, audit


def main() -> int:
    args = parse_args()
    if not args.run:
        print("未提供 --run；不会建立候选。")
        return 2
    if not math.isclose(
        args.socket_entry_f_diameter_mm,
        AUTHORIZED_F_DIAMETER_MM,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise SystemExit("只授权 socket_entry_F_diameter_mm=1.32")

    source_asset = REPOSITORY / SOURCE_ASSET_REL
    source_scene = REPOSITORY / SOURCE_SCENE_REL
    candidate_dir = REPOSITORY / CANDIDATE_DIR_REL
    candidate_asset = REPOSITORY / CANDIDATE_ASSET_REL
    candidate_scene = REPOSITORY / CANDIDATE_SCENE_REL
    result_path = REPOSITORY / RESULT_REL
    temporary_asset = candidate_asset.with_suffix(".building.usda")
    temporary_scene = candidate_scene.with_suffix(".building.yaml")
    protected_outputs = (
        candidate_dir,
        candidate_asset,
        candidate_scene,
        result_path,
        temporary_asset,
        temporary_scene,
    )
    if not source_asset.is_file() or not source_scene.is_file():
        raise SystemExit("候选2源资产或场景不存在")
    existing = [str(path) for path in protected_outputs if path.exists()]
    if existing:
        raise SystemExit(f"拒绝覆盖或重复建立唯一候选：{existing}")

    source_hash_before = sha256_file(source_asset)
    source_scene_hash = sha256_file(source_scene)
    candidate_dir.mkdir(parents=True, exist_ok=False)

    from pxr import Gf, Sdf, Usd, UsdGeom, Vt

    source_layer = Sdf.Layer.FindOrOpen(str(source_asset))
    if source_layer is None or not source_layer.Export(str(temporary_asset)):
        raise RuntimeError("无法从候选2导出局部候选临时层")
    candidate_stage = Usd.Stage.Open(str(temporary_asset), load=Usd.Stage.LoadAll)
    if candidate_stage is None:
        raise RuntimeError("无法打开局部候选临时层")
    meters_per_stage_unit = UsdGeom.GetStageMetersPerUnit(candidate_stage)
    if not math.isclose(
        meters_per_stage_unit,
        EXPECTED_STAGE_METERS_PER_UNIT,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError(
            f"unexpected metersPerUnit: {meters_per_stage_unit}"
        )

    authored_prim_count = 0
    authored_point_count = 0
    for prim in candidate_stage.TraverseAll():
        path = str(prim.GetPath())
        match = TARGET_RE.fullmatch(path)
        if match is None:
            continue
        mesh = UsdGeom.Mesh(prim)
        points_attr = mesh.GetPointsAttr()
        points = list(points_attr.Get())
        if len(points) != 8:
            raise ValueError(f"unexpected target point count at {path}")
        for inner_index in _target_indices(match.group(1)):
            points[inner_index] = _new_inner_point(
                points[inner_index], points[PAIR_INDEX[inner_index]], Gf
            )
            authored_point_count += 1
        points_attr.Set(Vt.Vec3fArray(points))
        authored_prim_count += 1
    if authored_prim_count != EXPECTED_MODIFIED_PRIMS:
        raise ValueError(
            f"authored prim count {authored_prim_count} != {EXPECTED_MODIFIED_PRIMS}"
        )
    if authored_point_count != EXPECTED_MODIFIED_POINTS:
        raise ValueError(
            f"authored point count {authored_point_count} != {EXPECTED_MODIFIED_POINTS}"
        )
    candidate_stage.GetRootLayer().Save()
    candidate_stage = None

    source_stage = Usd.Stage.Open(str(source_asset), load=Usd.Stage.LoadAll)
    candidate_stage = Usd.Stage.Open(str(temporary_asset), load=Usd.Stage.LoadAll)
    if source_stage is None or candidate_stage is None:
        raise RuntimeError("无法重开源或局部候选进行差异审计")
    source_inventory = _inventory(source_stage)
    candidate_inventory = _inventory(candidate_stage)
    if source_inventory != candidate_inventory:
        raise ValueError("局部候选的 prim/类型/家族清单发生变化")
    compared_non_target_properties = _assert_non_target_properties_unchanged(
        source_stage, candidate_stage
    )
    modified_prims, modified_points, sample_paths, point_audit = _assert_target_points(
        source_stage, candidate_stage, UsdGeom
    )
    source_stage = None
    candidate_stage = None

    scene_text = source_scene.read_text(encoding="utf-8")
    source_line = f"  local_path: {SOURCE_ASSET_REL}"
    candidate_line = f"  local_path: {CANDIDATE_ASSET_REL}"
    if scene_text.count(source_line) != 1:
        raise ValueError("候选2场景中的资产路径不是唯一预期值")
    temporary_scene.write_text(
        scene_text.replace(source_line, candidate_line, 1), encoding="utf-8"
    )

    temporary_asset.replace(candidate_asset)
    temporary_scene.replace(candidate_scene)
    source_hash_after = sha256_file(source_asset)
    if source_hash_after != source_hash_before:
        raise RuntimeError("候选2源资产在候选生成期间发生变化")

    result = {
        "schema_version": 1,
        "task_id": "TASK-R12-006B",
        "built_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "candidate_number": 1,
        "candidate_limit": 1,
        "structural_parameter_count": 1,
        "structural_parameter": {
            "name": "socket_entry_F_diameter_mm",
            "before_mm": 1.28,
            "after_mm": 1.32,
            "authorized_range_mm": [1.24, 1.32],
            "derived_inner_radius_before_m": OLD_INNER_RADIUS_M,
            "derived_inner_radius_after_m": NEW_INNER_RADIUS_M,
            "meters_per_stage_unit": meters_per_stage_unit,
            "target_point_scale": EXPECTED_TARGET_POINT_SCALE,
            "point_delta_local_units": (
                NEW_INNER_RADIUS_M - OLD_INNER_RADIUS_M
            ) / TARGET_METERS_PER_LOCAL_UNIT,
        },
        "source_asset": str(SOURCE_ASSET_REL),
        "source_asset_sha256_before": source_hash_before,
        "source_asset_sha256_after": source_hash_after,
        "candidate_asset": str(CANDIDATE_ASSET_REL),
        "candidate_asset_sha256": sha256_file(candidate_asset),
        "source_scene": str(SOURCE_SCENE_REL),
        "source_scene_sha256": source_scene_hash,
        "candidate_scene": str(CANDIDATE_SCENE_REL),
        "candidate_scene_sha256": sha256_file(candidate_scene),
        "modified_prim_count": modified_prims,
        "modified_point_count": modified_points,
        "expected_modified_prim_count": EXPECTED_MODIFIED_PRIMS,
        "expected_modified_point_count": EXPECTED_MODIFIED_POINTS,
        "sample_modified_prim_paths": sample_paths,
        "point_geometry_audit": point_audit,
        "compared_unchanged_property_count": compared_non_target_properties,
        "source_inventory": source_inventory,
        "candidate_inventory": candidate_inventory,
        "formal_contract_modified": False,
        "mass_material_friction_elasticity_modified": False,
        "source_candidate_modified": False,
        "passed": True,
        "argv": sys.argv,
    }
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
