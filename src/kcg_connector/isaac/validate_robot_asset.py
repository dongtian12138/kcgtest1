#!/usr/bin/env python3

"""Statically validate the imported KUKA/three-finger-hand USD asset."""

from __future__ import annotations

import argparse
import json
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


def validate_asset(asset_path: Path) -> dict[str, Any]:
    from pxr import Sdf, Usd, UsdPhysics, UsdUtils

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

    if errors:
        raise AssetValidationError(errors, metrics)

    metrics["status"] = "PASSED"
    return metrics


def _emit_failure(
    asset_path: Path,
    errors: list[str],
    metrics: dict[str, Any] | None = None,
) -> None:
    report = metrics.copy() if metrics else {"asset": str(asset_path)}
    report.update({"errors": errors, "status": "FAILED"})
    # Keep the JSON on stdout on both success and failure.  Kit wraps stderr
    # lines in log prefixes, which would make a failure report invalid JSON.
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    print(FAIL_BANNER)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the composed KUKA/three-finger-hand Isaac USD asset."
    )
    parser.add_argument(
        "--asset",
        default=str(_default_asset_path()),
        help="Robot root USD (defaults to the repository handarm.usda).",
    )
    arguments = parser.parse_args()

    asset_path = Path(arguments.asset).expanduser().resolve()
    if not asset_path.is_file():
        _emit_failure(asset_path, [f"robot USD does not exist: {asset_path}"])
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
            report = validate_asset(asset_path)
            print(json.dumps(report, ensure_ascii=False, sort_keys=True))
            print(PASS_BANNER)
        except AssetValidationError as error:
            exit_code = 1
            _emit_failure(asset_path, error.errors, error.metrics)
        except Exception as error:  # Keep a machine-readable failure on API/runtime errors.
            exit_code = 1
            _emit_failure(
                asset_path,
                [f"{type(error).__name__}: {error}"],
            )
    except Exception as error:
        exit_code = 1
        _emit_failure(
            asset_path,
            [f"Isaac Sim startup failed: {type(error).__name__}: {error}"],
        )
    finally:
        if simulation_app is not None:
            # Isaac Sim 6 fast shutdown may terminate inside close(). Preserve failures.
            simulation_app.close(exit_code=exit_code)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
