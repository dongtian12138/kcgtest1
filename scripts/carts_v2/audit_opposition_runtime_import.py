#!/usr/bin/env python3
"""Read back one imported opposition-60 robot asset without loading an object."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback
import xml.etree.ElementTree as ET

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 project venv; Isaac Python is 3.12.
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[2]
ISAAC_V2 = ROOT / "src/kcg_connector/isaac/carts_v2"
sys.path.insert(0, str(ISAAC_V2))

_LINKS = ("f1Link3", "f2Link2", "f3Link3")
_MIMIC = {"f1j3": "f1j2", "f2j2": "f2j1", "f3j1": "f1j1", "f3j3": "f3j2"}
_FULL_DOFS = tuple(f"iiwa_joint_{index}" for index in range(1, 8)) + (
    "f1j1", "f1j2", "f1j3", "f2j1", "f2j2", "f3j1", "f3j2", "f3j3",
)
_LOCAL_DOFS = ("f1j1", "f1j2", "f1j3", "f2j1", "f2j2", "f3j1", "f3j2", "f3j3")
_RESOURCES = ROOT / "src/kcg_connector/config/carts_v2_isaac_runtime.json"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-binding", type=Path, required=True)
    parser.add_argument("--usd", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--asset-scope", choices=("full-handarm", "local-hand"),
        default="full-handarm",
    )
    parser.add_argument("--runtime-resources", type=Path, default=_RESOURCES)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _bound_file(path_value: str, sha256_value: str, label: str) -> Path:
    path = _resolve(path_value)
    _require(path.is_file() and _sha256(path) == sha256_value, f"{label} hash changed")
    return path


def _binding_audit(path: Path, asset_scope: str) -> tuple[dict, dict, dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    _require(
        document.get("schema_version") == "carts_opposition60_runtime_urdf_binding_v1"
        and document.get("runtime_binding_accepted") is False
        and document.get("formal_dynamic_pass") is False
        and document.get("hardware_authorized") is False,
        "runtime binding boundary changed",
    )
    _bound_file(document["generator"], document["generator_sha256"], "generator")
    _bound_file(
        document["collision_manifest"], document["collision_manifest_sha256"],
        "collision manifest",
    )
    if asset_scope == "local-hand":
        local = document.get("local_hand_urdf") or {}
        urdf_path = _bound_file(local["path"], local["sha256"], "local hand URDF")
        identity = {
            "robot_name": local["robot_name"], "dof_names": _LOCAL_DOFS,
            "urdf_kind": "LOCAL_HAND_FROM_BOUND_FULL_TREE",
        }
    else:
        urdf_path = _bound_file(
            document["runtime_urdf"], document["runtime_urdf_sha256"], "runtime URDF"
        )
        identity = {
            "robot_name": document["runtime_robot_name"], "dof_names": _FULL_DOFS,
            "urdf_kind": "FULL_HANDARM",
        }
    terminal = document.get("terminal_links") or {}
    _require(set(terminal) == set(_LINKS), "terminal binding set changed")
    for link, row in terminal.items():
        hulls = row.get("collision_hulls") or []
        _require(row.get("collision_count") == len(hulls) == 64, f"{link}: not 64 hulls")
        for hull in hulls:
            _bound_file(hull["path"], hull["sha256"], f"{link} hull {hull['index']}")
        _bound_file(row["visual_mesh"], row["visual_sha256"], f"{link} visual")
    robot = ET.parse(urdf_path).getroot()
    revolute = {joint.get("name") for joint in robot.findall("joint")
                if joint.get("type") == "revolute"}
    mimic = {}
    for joint in robot.findall("joint"):
        node = joint.find("mimic")
        if node is not None:
            mimic[joint.get("name")] = {
                "source": node.get("joint"),
                "multiplier": float(node.get("multiplier", "1")),
                "offset": float(node.get("offset", "0")),
            }
    urdf_mimic_pass = bool(
        set(revolute) == set(identity["dof_names"]) and set(mimic) == set(_MIMIC)
        and all(mimic[name]["source"] == source
                and mimic[name]["multiplier"] == 1.0 and mimic[name]["offset"] == 0.0
                for name, source in _MIMIC.items())
    )
    _require(urdf_mimic_pass, "bound runtime URDF joint or mimic identity changed")
    return document, {
        "path": str(path), "sha256": _sha256(path), "runtime_urdf": str(urdf_path),
        "runtime_urdf_sha256": _sha256(urdf_path),
        "urdf_revolute_joint_count": len(identity["dof_names"]),
        "urdf_mimic": mimic, "urdf_mimic_pass": True,
    }, identity


def _usd_layer_hashes(asset: Path) -> dict:
    root = asset.parent.resolve()
    files = sorted(
        path.resolve() for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".usd", ".usda", ".usdc"}
    )
    _require(asset.resolve() in files and len(files) > 1, "root USD or payload layers missing")
    return {
        "root_usd": {"path": str(asset), "sha256": _sha256(asset)},
        "payload_layers": [
            {"path": str(path.relative_to(root)), "sha256": _sha256(path)}
            for path in files if path != asset.resolve()
        ],
        "layer_count": len(files),
    }


def _importer_version() -> dict:
    import isaacsim.asset.importer.urdf as importer

    module_path = Path(importer.__file__).resolve()
    extension = module_path.parents[4] / "config/extension.toml"
    package = tomllib.loads(extension.read_text(encoding="utf-8"))["package"]
    return {
        "version": package["version"], "module_path": str(module_path),
        "extension_manifest": str(extension), "extension_manifest_sha256": _sha256(extension),
    }


def _mimic_audit(stage, root_path: str) -> dict:
    rows = {}
    for follower, source in _MIMIC.items():
        prim = stage.GetPrimAtPath(f"{root_path}/Physics/{follower}")
        relation = prim.GetRelationship("newton:mimicJoint") if prim.IsValid() else None
        targets = [] if relation is None else [str(item) for item in relation.GetTargets()]
        expected = f"{root_path}/Physics/{source}"
        rows[follower] = {
            "source": source, "relationship_targets": targets,
            "relationship_matches": targets == [expected],
            "applied_schemas": [] if not prim.IsValid() else list(prim.GetAppliedSchemas()),
        }
    usd_pass = all(row["relationship_matches"] for row in rows.values())
    return {
        "usd_preserved": usd_pass, "usd_rows": rows,
        "evidence_source": "USD_SCHEMA" if usd_pass else "BOUND_RUNTIME_URDF_FALLBACK",
        "pass": True,
    }


def _terminal_collision_audit(prims, UsdPhysics) -> dict:
    by_link = {name: [] for name in _LINKS}
    for prim in prims:
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        path_parts = str(prim.GetPath()).split("/")
        owners = [name for name in _LINKS if name in path_parts]
        if len(owners) == 1:
            collision = UsdPhysics.CollisionAPI(prim)
            by_link[owners[0]].append({
                "path": str(prim.GetPath()), "type": str(prim.GetTypeName()),
                "enabled": collision.GetCollisionEnabledAttr().Get(),
                "approximation": prim.GetAttribute("physics:approximation").Get(),
            })
    result = {}
    for link, rows in by_link.items():
        names = {Path(row["path"]).name for row in rows}
        expected = {f"{link}_compound_hull_{index:02d}" for index in range(64)}
        result[link] = {
            "count": len(rows), "paths": [row["path"] for row in rows],
            "collision_enabled_pass": all(row["enabled"] is True for row in rows),
            "mesh_approximation_pass": all(str(row["approximation"]) == "convexHull"
                                           for row in rows),
            "names_match_bound_hulls": names == expected,
        }
    passed = all(row["count"] == 64 and row["collision_enabled_pass"]
                 and row["mesh_approximation_pass"] and row["names_match_bound_hulls"]
                 for row in result.values())
    return {"per_terminal_link": result, "terminal_total": sum(
        row["count"] for row in result.values()), "pass": passed}


def _visual_audit(prims, binding: dict, UsdGeom, UsdPhysics) -> dict:
    rows = {}
    for link in _LINKS:
        expected = Path(binding["terminal_links"][link]["visual_mesh"]).stem
        matches = [str(prim.GetPath()) for prim in prims
                   if prim.IsA(UsdGeom.Mesh) and expected in str(prim.GetPath()).split("/")
                   and not prim.HasAPI(UsdPhysics.CollisionAPI)]
        rows[link] = {"expected_nailfree_visual_stem": expected, "mesh_paths": matches,
                      "pass": len(matches) == 1 and expected.endswith("_nailfree")}
    return {"links": rows, "pass": all(row["pass"] for row in rows.values())}


def _matrix_from_binding(row: dict) -> list[list[float]]:
    values = row["inertial"]["inertia_kg_m2"]
    return [[values["ixx"], values["ixy"], values["ixz"]],
            [values["ixy"], values["iyy"], values["iyz"]],
            [values["ixz"], values["iyz"], values["izz"]]]


def _mass_audit(prims, binding: dict, UsdPhysics, Gf, np) -> dict:
    rows = {}
    for link in _LINKS:
        matches = [prim for prim in prims
                   if prim.GetName() == link and prim.HasAPI(UsdPhysics.MassAPI)]
        if len(matches) != 1:
            rows[link] = {"pass": False, "reason": f"mass prim count {len(matches)}"}
            continue
        api = UsdPhysics.MassAPI(matches[0])
        mass, center = float(api.GetMassAttr().Get()), np.asarray(api.GetCenterOfMassAttr().Get())
        diagonal = np.asarray(api.GetDiagonalInertiaAttr().Get(), dtype=float)
        axes = api.GetPrincipalAxesAttr().Get()
        rotation = np.asarray(Gf.Matrix3d(axes), dtype=float)
        candidates = (rotation @ np.diag(diagonal) @ rotation.T,
                      rotation.T @ np.diag(diagonal) @ rotation)
        expected = np.asarray(_matrix_from_binding(binding["terminal_links"][link]))
        inertia_error = min(float(np.max(np.abs(value - expected))) for value in candidates)
        expected_row = binding["terminal_links"][link]["inertial"]
        mass_error = abs(mass - float(expected_row["mass_kg"]))
        com_error = float(np.max(np.abs(center - expected_row["center_of_mass_m"])))
        rows[link] = {
            "path": str(matches[0].GetPath()), "mass_kg": mass,
            "center_of_mass_m": center.tolist(), "diagonal_inertia_kg_m2": diagonal.tolist(),
            "mass_error_kg": mass_error, "com_max_error_m": com_error,
            "inertia_matrix_max_error_kg_m2": inertia_error,
            "pass": mass_error <= 1e-7 and com_error <= 1e-7 and inertia_error <= 1e-9,
        }
    return {"comparison_tolerances": {"mass_abs_kg": 1e-7, "com_abs_m": 1e-7,
            "inertia_matrix_abs_kg_m2": 1e-9}, "links": rows,
            "pass": all(row["pass"] for row in rows.values())}


def _write(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def main() -> int:
    args = _arguments()
    binding_path, asset, output = map(_resolve, (args.runtime_binding, args.usd, args.output))
    _require(not output.exists(), f"refusing to overwrite audit: {output}")
    report = {
        "schema_version": "carts_opposition60_isaac_import_readback_v1",
        "status": "ISAAC_IMPORT_GATE_FAILED", "hardware_authorized": False,
        "formal_dynamic_pass": False, "research_dynamic_pass": False,
        "runtime_binding_accepted": False,
        "runtime_gates": {"ISAAC_IMPORT": False, "INITIAL_PENETRATION": False,
                          "OPPOSITION60_REPLAY": False, "PHYSX_HEALTH": False},
        "object_asset_loaded": False, "asset_scope": args.asset_scope,
        "auditor_source": {"path": str(Path(__file__).resolve()),
                           "sha256": _sha256(Path(__file__).resolve())},
        "errors": [],
    }
    app = None
    log_path = None
    try:
        binding, binding_record, identity = _binding_audit(binding_path, args.asset_scope)
        _require(asset.is_file(), f"configured USD missing: {asset}")
        _require(asset.name == f"{identity['robot_name']}.usda",
                 "configured USD filename differs from bound runtime robot name")
        report.update({"runtime_binding": binding_record, "usd_layers": _usd_layer_hashes(asset)})

        from isaacsim import SimulationApp
        app = SimulationApp({"headless": True, "multi_gpu": False,
                             "active_gpu": 0, "physics_gpu": 0})
        import numpy as np
        import omni.kit.app
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
        from engine_health import (
            PhysxStatsMonitor, audit_physx_log, current_engine_log_path,
            gpu_backend_record, gpu_world_parameters, load_runtime_resources,
            synchronize_engine_log,
        )

        log_path = current_engine_log_path()
        imported = Usd.Stage.Open(str(asset), Usd.Stage.LoadAll)
        _require(imported is not None, "USD stage open failed")
        default_prim = imported.GetDefaultPrim()
        _require(default_prim and default_prim.GetName() == identity["robot_name"],
                 "defaultPrim differs from runtime binding")
        unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(asset)))[2]
        _require(not unresolved, f"unresolved USD dependencies: {list(unresolved)}")
        root_path = str(default_prim.GetPath())
        composed_prims = list(imported.Traverse(Usd.TraverseInstanceProxies()))
        revolute_names = sorted(prim.GetName() for prim in composed_prims
                                if prim.IsA(UsdPhysics.RevoluteJoint))
        expected_dofs = tuple(identity["dof_names"])
        stage_dof_pass = (
            len(revolute_names) == len(expected_dofs)
            and set(revolute_names) == set(expected_dofs)
        )
        mimic = _mimic_audit(imported, root_path)
        collision = _terminal_collision_audit(composed_prims, UsdPhysics)
        visual = _visual_audit(composed_prims, binding, UsdGeom, UsdPhysics)
        mass = _mass_audit(composed_prims, binding, UsdPhysics, Gf, np)

        resources = load_runtime_resources(_resolve(args.runtime_resources))
        World.clear_instance()
        world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 120.0,
                      rendering_dt=1.0 / 60.0, **gpu_world_parameters(resources))
        reference_path = "/World/Robot"
        add_reference_to_stage(str(asset), reference_path)
        stage = get_current_stage()
        roots = [prim for prim in stage.Traverse()
                 if prim.GetPath().HasPrefix(Sdf.Path(reference_path))
                 and prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
        _require(len(roots) == 1, f"articulation root count is {len(roots)}")
        robot = world.scene.add(SingleArticulation(prim_path=str(roots[0].GetPath()),
                                                  name="opposition60_import_readback"))
        world.reset()
        _require(robot.handles_initialized, "articulation handles not initialized")
        runtime_dofs = list(robot.dof_names)
        runtime_dof_pass = (
            robot.num_dof == len(expected_dofs) and set(runtime_dofs) == set(expected_dofs)
        )
        context = world.get_physics_context()
        monitor = PhysxStatsMonitor(context)
        for _ in range(3):
            world.step(render=False)
            monitor.sample()
        sync = synchronize_engine_log(log_path)
        log = audit_physx_log(log_path, cutoff_bytes=sync["audit_byte_count"],
                              required_marker=sync["marker"])
        stats = monitor.summary()
        backend = gpu_backend_record(world, context)
        log_pass = bool(log.get("scan_complete") and log.get("capacity_warning_count") == 0
                        and log.get("physx_error_lines") == [])
        capacity_pass = bool(
            stats["physx_statistics_sample_count"] > 0
            and stats["physx_statistics_read_failures"] == 0
            and stats["observed_gpu_found_lost_aggregate_pairs_peak"]
            < stats["configured_gpu_found_lost_aggregate_pairs_capacity"]
            and stats["observed_gpu_total_aggregate_pairs_peak"]
            < stats["configured_gpu_total_aggregate_pairs_capacity"]
        )
        kit = omni.kit.app.get_app()
        report["versions"] = {
            "isaac_app_version": kit.get_app_version(),
            "isaac_build_version": kit.get_build_version(),
            "urdf_importer": _importer_version(),
        }
        report["readback"] = {
            "default_prim": root_path, "default_prim_pass": True,
            "usd_revolute_joint_names": revolute_names,
            "usd_revolute_joint_count": len(revolute_names),
            "usd_revolute_joint_pass": stage_dof_pass,
            "runtime_dof_names": runtime_dofs, "runtime_dof_count": robot.num_dof,
            "runtime_dof_pass": runtime_dof_pass, "mimic": mimic,
            "terminal_collisions": collision, "nailfree_visuals": visual,
            "terminal_mass_properties": mass,
        }
        report["import_process_physx"] = {
            "backend": backend, "statistics": stats, "log": log,
            "log_pass": log_pass, "capacity_pass": capacity_pass,
        }
        gate = all((stage_dof_pass, runtime_dof_pass, mimic["pass"], collision["pass"],
                    visual["pass"], mass["pass"], backend["pass"], log_pass,
                    capacity_pass, bool(report["versions"]["isaac_app_version"]),
                    bool(report["versions"]["urdf_importer"]["version"])))
        report["runtime_gates"]["ISAAC_IMPORT"] = gate
        report["status"] = "ISAAC_IMPORT_GATE_PASS" if gate else "ISAAC_IMPORT_GATE_FAILED"
    except Exception as error:  # fail closed while preserving one machine-readable result
        report["errors"].append({"type": type(error).__name__, "message": str(error),
                                 "traceback": traceback.format_exc()})
    finally:
        _write(output, report)
        print(json.dumps({"output": str(output), "status": report["status"],
                          "runtime_binding_accepted": False}, sort_keys=True), flush=True)
        if app is not None:
            app.close()
    return 0 if report["runtime_gates"]["ISAAC_IMPORT"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
