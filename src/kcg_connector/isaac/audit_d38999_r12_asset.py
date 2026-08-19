#!/usr/bin/env python3

"""Read-only static USD inventory audit for one r12 candidate or release."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Sequence

from kcg_connector.d38999_keyed_v3_physical_r12_contract import (
    authorized_asset_path,
    candidate_model,
    load_r12_physical_model_contract,
)


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-index", type=int, default=None)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def _translation(prim: Any, UsdGeom: Any) -> list[float]:
    operations = UsdGeom.Xformable(prim).GetOrderedXformOps()
    for operation in operations:
        if operation.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            return [float(value) for value in operation.Get()]
    return [0.0, 0.0, 0.0]


def _material_path(prim: Any) -> str:
    relationship = prim.GetRelationship("material:binding:physics")
    targets = relationship.GetTargets() if relationship else []
    return str(targets[0]) if len(targets) == 1 else "UNAVAILABLE"


def run(candidate_index: int | None) -> dict[str, Any]:
    from pxr import Usd, UsdGeom, UsdPhysics

    model = load_r12_physical_model_contract()
    if candidate_index is not None:
        model = candidate_model(model, candidate_index)
    asset = authorized_asset_path(model)
    stage = Usd.Stage.Open(str(asset), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"could not open r12 asset: {asset}")
    rows = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        family_attr = prim.GetAttribute("kcg:primitiveFamily")
        family = family_attr.Get() if family_attr else None
        if not family:
            continue
        row = {
            "path": str(prim.GetPath()),
            "family": str(family),
            "type_name": str(prim.GetTypeName()),
            "recipe_id": str(prim.GetAttribute("kcg:primitiveRecipeId").Get()),
            "material_path": _material_path(prim),
            "translation_m": _translation(prim, UsdGeom),
        }
        if prim.IsA(UsdGeom.Sphere):
            row["radius_m"] = float(UsdGeom.Sphere(prim).GetRadiusAttr().Get())
        elif prim.IsA(UsdGeom.Cylinder):
            cylinder = UsdGeom.Cylinder(prim)
            row["radius_m"] = float(cylinder.GetRadiusAttr().Get())
            row["height_m"] = float(cylinder.GetHeightAttr().Get())
            row["axis"] = str(cylinder.GetAxisAttr().Get())
        elif prim.IsA(UsdGeom.Capsule):
            capsule = UsdGeom.Capsule(prim)
            row["radius_m"] = float(capsule.GetRadiusAttr().Get())
            row["height_m"] = float(capsule.GetHeightAttr().Get())
            row["axis"] = str(capsule.GetAxisAttr().Get())
        rows.append(row)
    family_counts = Counter(row["family"] for row in rows)
    type_counts = Counter(row["type_name"] for row in rows)
    repaired_families = {
        "detent_followers_3",
        "fixed_metal_stop_48",
        "plug_metal_stop_48",
        "shoulder_positive_body0_48",
        "shoulder_positive_body1_48",
        "shoulder_negative_body0_48",
        "shoulder_negative_body1_48",
    }
    repaired_rows = [row for row in rows if row["family"] in repaired_families]
    old_box = sum(
        row["family"] == "detent_followers_3"
        and row["type_name"] == "Mesh"
        for row in rows
    )
    old_shoulder = sum(
        row["family"].startswith("shoulder_") and "/Seg_" in row["path"]
        for row in rows
    )
    old_bottom = sum(
        row["family"] in {"fixed_metal_stop_48", "plug_metal_stop_48"}
        and "/Seg_" in row["path"]
        for row in rows
    )
    expected_repaired_counts = {
        "detent_followers_3": 3,
        "fixed_metal_stop_48": 1,
        "plug_metal_stop_48": 3,
        "shoulder_positive_body0_48": 1,
        "shoulder_positive_body1_48": 3,
        "shoulder_negative_body0_48": 1,
        "shoulder_negative_body1_48": 3,
    }
    expected_inventory = __import__(
        "kcg_connector.d38999_keyed_v2_a2_readback_result",
        fromlist=["_trusted_collider_inventory"],
    )._trusted_collider_inventory(model)
    expected_type_counts = Counter(
        row["typeName"] for row in expected_inventory.values()
    )
    expected_collider_count = len(expected_inventory)
    passed = bool(
        len(rows) == expected_collider_count
        and type_counts == expected_type_counts
        and all(family_counts[name] == count for name, count in expected_repaired_counts.items())
        and old_box == 0
        and old_shoulder == 0
        and old_bottom == 0
        and all(
            "coupling_bearing_and_shoulder__hard_nut_body_shoulder"
            in row["material_path"]
            for row in repaired_rows
            if row["family"].startswith("shoulder_")
        )
        and all(
            "hard_metal_bottoming" in row["material_path"]
            for row in repaired_rows
            if "metal_stop" in row["family"]
        )
    )
    return {
        "status": "PASSED" if passed else "FAILED",
        "asset_path": str(asset),
        "collider_count": len(rows),
        "expected_collider_count": expected_collider_count,
        "type_counts": dict(sorted(type_counts.items())),
        "expected_type_counts": dict(sorted(expected_type_counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
        "old_square_detent_follower_count": old_box,
        "old_segmented_shoulder_count": old_shoulder,
        "old_segmented_metal_bottoming_count": old_bottom,
        "new_round_detent_follower_count": family_counts["detent_followers_3"],
        "new_shoulder_collider_count": sum(
            family_counts[name] for name in family_counts if name.startswith("shoulder_")
        ),
        "new_metal_bottoming_collider_count": (
            family_counts["fixed_metal_stop_48"] + family_counts["plug_metal_stop_48"]
        ),
        "repaired_collider_rows": sorted(repaired_rows, key=lambda row: row["path"]),
        "passed": passed,
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _arguments(argv)
    output = Path(arguments.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite audit report: {output}")
    report = run(arguments.candidate_index)
    output.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, allow_nan=False, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
