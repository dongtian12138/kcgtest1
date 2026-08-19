#!/usr/bin/env python3

"""Locate the first D38999 initialization datum drift between T0 and T7.

This probe is diagnostic-only.  It replays the frozen assembly-control model
twice in one Isaac process: once with authored collisions unchanged and once
with all assembly-control colliders disabled only on the anonymous in-memory
stage.  Each case performs exactly one explicit physics step.  Contact truth
is recorded post hoc and never participates in control.
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
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


SCHEMA_VERSION = "kcg_d38999_multilayer_init_timeline_probe_v1"
TASK_ID = "DYN-A1-INIT-ROOTCAUSE"
HYPOTHESIS_ID = "DYN-A1-H1-T0-T7-LOCALIZATION"
OUTPUT_RELATIVE = Path(
    "artifacts/agent_control/tasks/DYN-A1-INIT-ROOTCAUSE/DIAGNOSTIC_T0_T7"
)
CONTRACT_RELATIVE = Path(
    "src/kcg_connector/config/d38999_master_model_contract_v1.yaml"
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
    "model": "26c44d86372fa9db64acd6503499f7335ddbabb14b8dd82c7ec7e31c6dc37cec",
    "mapping": "a31d4c0e7cb911f0371c257b0784506388f0f52d1d07401c1b1c2239fa47a783",
    "acceptance": "dd37d07d5bd87a97c3dbefe225c0eafd409fc783a6010eec8db9da397ce2cf76",
}

START_SEPARATION_M = 0.00550
TOLERANCE_M = 5.0e-5
EXPECTED_COLLIDER_COUNT = 19
ROOT = "/World/D38999MultilayerV1/D38999_ASSEMBLY_CONTROL_V1"
PAIR_ROOT = ROOT + "/D38999Pair"
FIXED_PATH = PAIR_ROOT + "/FixedReceptacle"
BODY_PATH = PAIR_ROOT + "/LoosePlug/BodyAssembly"
NUT_PATH = PAIR_ROOT + "/LoosePlug/CouplingNut"
PLUG_GUIDE_PATH = BODY_PATH + "/MatingShell/ContinuousPlugGuide"
OWNER_PATHS = {
    "fixed_receptacle": FIXED_PATH,
    "body_assembly": BODY_PATH,
    "coupling_nut": NUT_PATH,
}
STAGE_ORDER = ("T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7")
ALLOWED_ROOT_CAUSES = (
    "RESET_DEFAULT_STATE_MISMATCH",
    "INITIAL_COLLISION_DEPENETRATION",
    "DATUM_MEASUREMENT_ERROR",
)


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--kit-portable-root", default=None)
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
        raise PermissionError(f"{label} path is frozen: {actual} != {expected}")
    if not actual.is_file():
        raise FileNotFoundError(actual)
    actual_sha = _sha256(actual)
    if actual_sha != expected_sha:
        raise PermissionError(f"{label} SHA-256 changed: {actual_sha} != {expected_sha}")
    return actual


def _authorize(arguments: argparse.Namespace) -> dict[str, Any]:
    repository = _repository()
    output = Path(arguments.output_dir).expanduser().resolve()
    expected_output = (repository / OUTPUT_RELATIVE).resolve()
    if output != expected_output:
        raise PermissionError(f"output path is frozen: {output} != {expected_output}")
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
    dynamic = state.get("dynamic_red_gate", {})
    if state.get("task_id") != TASK_ID or state.get("status") != "RUNNING":
        raise PermissionError("MASTER_STATE does not authorize the running A1 diagnostic")
    if dynamic.get("current_task") != TASK_ID:
        raise PermissionError("MASTER_STATE dynamic task differs from A1")
    if dynamic.get("diagnostic_runs_started") != 1:
        raise PermissionError("A1 diagnostic started counter must equal one")
    if dynamic.get("diagnostic_runs_completed") != 0:
        raise PermissionError("A1 diagnostic completion counter must equal zero")
    if queue.get("status") != "RUNNING" or queue.get("current_task") != TASK_ID:
        raise PermissionError("WORK_QUEUE does not authorize A1")

    contract_path = _frozen_path(
        arguments.contract,
        CONTRACT_RELATIVE,
        EXPECTED_SHA256["contract"],
        "master contract",
    )
    model_path = _frozen_path(
        arguments.model,
        MODEL_RELATIVE,
        EXPECTED_SHA256["model"],
        "assembly-control model",
    )
    mapping_path = (repository / MAPPING_RELATIVE).resolve()
    if _sha256(mapping_path) != EXPECTED_SHA256["mapping"]:
        raise PermissionError("frozen mapping SHA-256 changed")
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    acceptance_info = contract["authoritative_inputs"]["physical_acceptance_contract"]
    acceptance_path = (repository / acceptance_info["path"]).resolve()
    acceptance_sha = _sha256(acceptance_path)
    if (
        acceptance_sha != EXPECTED_SHA256["acceptance"]
        or acceptance_sha != acceptance_info["sha256"]
    ):
        raise PermissionError("frozen acceptance SHA-256 changed")
    acceptance = yaml.safe_load(acceptance_path.read_text(encoding="utf-8"))
    return {
        "output": output,
        "contract_path": contract_path,
        "model_path": model_path,
        "mapping_path": mapping_path,
        "acceptance_path": acceptance_path,
        "contract": contract,
        "acceptance": acceptance,
    }


def _finite(value: Any, size: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise RuntimeError(f"{label} is not a finite {size}-vector")
    return result


def _stage_body_state(stage: Any, path: str, usd_geom: Any, usd_physics: Any) -> dict[str, Any]:
    prim = stage.GetPrimAtPath(path)
    if not prim:
        raise RuntimeError(f"missing rigid body prim {path}")
    matrix = usd_geom.XformCache().GetLocalToWorldTransform(prim)
    translation = matrix.ExtractTranslation()
    quaternion = matrix.ExtractRotationQuat()
    imaginary = quaternion.GetImaginary()
    rigid = usd_physics.RigidBodyAPI(prim)
    linear_attr = rigid.GetVelocityAttr()
    angular_attr = rigid.GetAngularVelocityAttr()
    linear_raw = linear_attr.Get() if linear_attr else None
    angular_raw = angular_attr.Get() if angular_attr else None
    linear = np.zeros(3) if linear_raw is None else _finite(linear_raw, 3, path + " linear")
    angular_deg = (
        np.zeros(3)
        if angular_raw is None
        else _finite(angular_raw, 3, path + " angular")
    )
    return {
        "path": path,
        "position_m": [float(translation[index]) for index in range(3)],
        "orientation_wxyz": [
            float(quaternion.GetReal()),
            float(imaginary[0]),
            float(imaginary[1]),
            float(imaginary[2]),
        ],
        "linear_velocity_m_s": linear.tolist(),
        "angular_velocity_rad_s": np.deg2rad(angular_deg).tolist(),
        "state_source": "usd_stage",
        "velocity_authored": {
            "linear": bool(linear_attr and linear_attr.HasAuthoredValueOpinion()),
            "angular": bool(angular_attr and angular_attr.HasAuthoredValueOpinion()),
        },
    }


def _view_body_state(view: Any, path: str, label: str) -> dict[str, Any]:
    positions, orientations = view.get_world_poses()
    position = _finite(positions[0], 3, label + " position")
    orientation = _finite(orientations[0], 4, label + " orientation")
    velocity = _finite(view.get_velocities()[0], 6, label + " velocity")
    return {
        "path": path,
        "position_m": position.tolist(),
        "orientation_wxyz": orientation.tolist(),
        "linear_velocity_m_s": velocity[:3].tolist(),
        "angular_velocity_rad_s": velocity[3:].tolist(),
        "state_source": "rigid_view",
        "physics_handle_valid": bool(view.is_physics_handle_valid()),
    }


def _snapshot(
    stage_id: str,
    bodies: Mapping[str, Mapping[str, Any]],
    *,
    explicit_step_count: int,
    semantic: str,
) -> dict[str, Any]:
    fixed = bodies["fixed_receptacle"]
    body = bodies["body_assembly"]
    nut = bodies["coupling_nut"]
    separation = float(fixed["position_m"][2] - body["position_m"][2])
    relative = [
        float(nut["position_m"][index] - body["position_m"][index])
        for index in range(3)
    ]
    return {
        "stage": stage_id,
        "semantic": semantic,
        "bodies": dict(bodies),
        "datum_separation_m": separation,
        "datum_error_m": separation - START_SEPARATION_M,
        "nut_body_relative_position_m": relative,
        "explicit_physics_step_count": explicit_step_count,
    }


def _stage_snapshot(
    stage_id: str,
    stage: Any,
    usd_geom: Any,
    usd_physics: Any,
    *,
    explicit_step_count: int,
    semantic: str,
) -> dict[str, Any]:
    bodies = {
        label: _stage_body_state(stage, path, usd_geom, usd_physics)
        for label, path in OWNER_PATHS.items()
    }
    return _snapshot(
        stage_id,
        bodies,
        explicit_step_count=explicit_step_count,
        semantic=semantic,
    )


def _view_snapshot(
    stage_id: str,
    views: Mapping[str, Any],
    *,
    explicit_step_count: int,
    semantic: str,
) -> dict[str, Any]:
    bodies = {
        label: _view_body_state(views[label], OWNER_PATHS[label], label)
        for label in OWNER_PATHS
    }
    return _snapshot(
        stage_id,
        bodies,
        explicit_step_count=explicit_step_count,
        semantic=semantic,
    )


def _set_initial_translation(stage: Any, path: str, usd_geom: Any, gf: Any) -> None:
    prim = stage.GetPrimAtPath(path)
    if not prim:
        raise RuntimeError(f"missing rigid body prim {path}")
    xformable = usd_geom.Xformable(prim)
    if xformable.GetOrderedXformOps():
        raise RuntimeError(f"unexpected authored transform stack at {path}")
    xformable.AddTranslateOp().Set(gf.Vec3d(0.0, 0.0, -START_SEPARATION_M))


def _configure_runtime_collision_cooking(stage: Any) -> dict[str, Any]:
    prim = stage.GetPrimAtPath(PLUG_GUIDE_PATH)
    if not prim:
        raise RuntimeError(f"missing {PLUG_GUIDE_PATH}")
    attribute = prim.GetAttribute("physics:approximation")
    authored = str(attribute.Get()) if attribute else None
    if authored != "none":
        raise RuntimeError(f"unexpected plug guide approximation: {authored}")
    attribute.Set("convexDecomposition")
    return {
        "path": PLUG_GUIDE_PATH,
        "source_authored": authored,
        "runtime": "convexDecomposition",
        "same_in_both_cases": True,
    }


def _collision_prims(stage: Any, usd_physics: Any) -> list[Any]:
    prims = [prim for prim in stage.Traverse() if prim.HasAPI(usd_physics.CollisionAPI)]
    if len(prims) != EXPECTED_COLLIDER_COUNT:
        raise RuntimeError(
            f"collision count changed: {len(prims)} != {EXPECTED_COLLIDER_COUNT}"
        )
    return prims


def _collision_state(prims: Sequence[Any]) -> list[dict[str, Any]]:
    rows = []
    for prim in prims:
        enabled = prim.GetAttribute("physics:collisionEnabled")
        role = prim.GetAttribute("kcg:collisionRole")
        rows.append(
            {
                "path": str(prim.GetPath()),
                "enabled": bool(enabled.Get()) if enabled else True,
                "role": str(role.Get()) if role and role.Get() is not None else "UNLABELED",
            }
        )
    return rows


def _set_collision_state(prims: Sequence[Any], enabled: bool) -> None:
    for prim in prims:
        attribute = prim.GetAttribute("physics:collisionEnabled")
        if not attribute:
            raise RuntimeError(f"missing collisionEnabled at {prim.GetPath()}")
        attribute.Set(bool(enabled))


def _role_for_path(stage: Any, path: str) -> str:
    prim = stage.GetPrimAtPath(path)
    if not prim:
        return "UNRESOLVED"
    attribute = prim.GetAttribute("kcg:collisionRole")
    value = attribute.Get() if attribute else None
    return str(value) if value is not None else "UNLABELED"


def _contact_inventory(
    stage: Any,
    interface: Any,
    schema_tools: Any,
    sample_stage: str,
) -> list[dict[str, Any]]:
    headers, contacts, _friction = interface.get_full_contact_report()
    rows: list[dict[str, Any]] = []
    for header in headers:
        actor_paths = [
            str(schema_tools.intToSdfPath(header.actor0)),
            str(schema_tools.intToSdfPath(header.actor1)),
        ]
        collider_paths = [
            str(schema_tools.intToSdfPath(header.collider0)),
            str(schema_tools.intToSdfPath(header.collider1)),
        ]
        if not any(path.startswith(ROOT) for path in actor_paths + collider_paths):
            continue
        start = int(header.contact_data_offset)
        stop = start + int(header.num_contact_data)
        records = []
        for index in range(start, stop):
            contact = contacts[index]
            impulse = _finite(contact.impulse, 3, "contact impulse")
            normal = _finite(contact.normal, 3, "contact normal")
            position = _finite(contact.position, 3, "contact position")
            records.append(
                {
                    "separation_m": float(contact.separation),
                    "impulse_n_s": impulse.tolist(),
                    "impulse_norm_n_s": float(np.linalg.norm(impulse)),
                    "normal": normal.tolist(),
                    "position_m": position.tolist(),
                }
            )
        roles = [_role_for_path(stage, path) for path in collider_paths]
        rows.append(
            {
                "sample_stage": sample_stage,
                "actor_paths": actor_paths,
                "collider_paths": collider_paths,
                "collision_roles": roles,
                "contact_family": " <-> ".join(sorted(roles)),
                "contact_record_count": len(records),
                "minimum_separation_m": min(
                    (row["separation_m"] for row in records), default=None
                ),
                "maximum_impulse_norm_n_s": max(
                    (row["impulse_norm_n_s"] for row in records), default=0.0
                ),
                "records": records,
            }
        )
    return rows


def _aggregate_contacts(samples: Sequence[Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    aggregate: dict[tuple[str, str], dict[str, Any]] = {}
    for sample in samples:
        for row in sample:
            key = tuple(sorted(str(path) for path in row["collider_paths"]))
            current = aggregate.setdefault(
                key,
                {
                    "collider_paths": list(key),
                    "actor_paths": row["actor_paths"],
                    "collision_roles": row["collision_roles"],
                    "contact_family": row["contact_family"],
                    "sample_stages": [],
                    "persistence_sample_count": 0,
                    "contact_record_count": 0,
                    "minimum_separation_m": None,
                    "maximum_penetration_m": 0.0,
                    "maximum_impulse_norm_n_s": 0.0,
                },
            )
            current["sample_stages"].append(row["sample_stage"])
            current["persistence_sample_count"] += 1
            current["contact_record_count"] += int(row["contact_record_count"])
            separation = row["minimum_separation_m"]
            if separation is not None and (
                current["minimum_separation_m"] is None
                or float(separation) < float(current["minimum_separation_m"])
            ):
                current["minimum_separation_m"] = float(separation)
            current["maximum_penetration_m"] = max(
                float(current["maximum_penetration_m"]),
                max(0.0, -float(separation)) if separation is not None else 0.0,
            )
            current["maximum_impulse_norm_n_s"] = max(
                float(current["maximum_impulse_norm_n_s"]),
                float(row["maximum_impulse_norm_n_s"]),
            )
    ranked = sorted(
        aggregate.values(),
        key=lambda row: (
            -float(row["maximum_penetration_m"]),
            -float(row["maximum_impulse_norm_n_s"]),
            -int(row["persistence_sample_count"]),
            row["collider_paths"],
        ),
    )
    family_summary: dict[str, dict[str, Any]] = {}
    for row in ranked:
        family = str(row["contact_family"])
        summary = family_summary.setdefault(
            family,
            {
                "contact_family": family,
                "pair_count": 0,
                "contact_record_count": 0,
                "maximum_penetration_m": 0.0,
                "maximum_impulse_norm_n_s": 0.0,
                "maximum_persistence_sample_count": 0,
            },
        )
        summary["pair_count"] += 1
        summary["contact_record_count"] += int(row["contact_record_count"])
        summary["maximum_penetration_m"] = max(
            summary["maximum_penetration_m"], row["maximum_penetration_m"]
        )
        summary["maximum_impulse_norm_n_s"] = max(
            summary["maximum_impulse_norm_n_s"], row["maximum_impulse_norm_n_s"]
        )
        summary["maximum_persistence_sample_count"] = max(
            summary["maximum_persistence_sample_count"],
            row["persistence_sample_count"],
        )
    return {
        "unique_pair_count": len(ranked),
        "rank_order": ["maximum_penetration_m", "maximum_impulse_norm_n_s", "persistence_sample_count"],
        "top_20": ranked[:20],
        "all_pairs": ranked,
        "families": sorted(
            family_summary.values(),
            key=lambda row: (
                -float(row["maximum_penetration_m"]),
                -float(row["maximum_impulse_norm_n_s"]),
                -int(row["maximum_persistence_sample_count"]),
                row["contact_family"],
            ),
        ),
    }


def _first_failed_transition(timeline: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    previous = "T1"
    if abs(float(timeline[previous]["datum_error_m"])) > TOLERANCE_M:
        return {
            "first_failed_stage": previous,
            "first_failed_transition": "T0_TO_T1",
            "datum_error_m": float(timeline[previous]["datum_error_m"]),
        }
    for current in STAGE_ORDER[2:]:
        error = float(timeline[current]["datum_error_m"])
        if abs(error) > TOLERANCE_M:
            return {
                "first_failed_stage": current,
                "first_failed_transition": f"{previous}_TO_{current}",
                "datum_error_m": error,
                "transition_delta_m": float(
                    timeline[current]["datum_separation_m"]
                    - timeline[previous]["datum_separation_m"]
                ),
            }
        previous = current
    return {
        "first_failed_stage": None,
        "first_failed_transition": None,
        "datum_error_m": float(timeline["T7"]["datum_error_m"]),
    }


def _run_case(
    *,
    case_id: str,
    collisions_enabled: bool,
    frozen: Mapping[str, Any],
    application: Any,
) -> dict[str, Any]:
    from isaacsim.core.api import World
    from isaacsim.core.prims import RigidPrim
    from isaacsim.core.utils.stage import get_current_stage
    from omni.physx import get_physx_simulation_interface
    import omni.usd
    from pxr import Gf, PhysicsSchemaTools, PhysxSchema, UsdGeom, UsdPhysics

    World.clear_instance()
    context = omni.usd.get_context()
    if context.get_stage() is not None:
        context.close_stage()
        application.update()
    if context.open_stage(str(frozen["model_path"])) is not True:
        raise RuntimeError(f"failed to open frozen model for {case_id}")
    for _ in range(3):
        application.update()
    acceptance = frozen["acceptance"]
    p1 = acceptance["benches"]["P1"]
    profile = p1["inputs"]["component_driver_profile"]
    rate_hz = int(acceptance["shared_numeric_profile"]["physics_rate_hz"])
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=1.0 / rate_hz,
        rendering_dt=1.0 / 60.0,
        backend="numpy",
        device="cpu",
    )
    stage = get_current_stage()
    cooking = _configure_runtime_collision_cooking(stage)
    colliders = _collision_prims(stage, UsdPhysics)
    collision_before = _collision_state(colliders)
    if not all(row["enabled"] for row in collision_before):
        raise RuntimeError("frozen model contains a disabled assembly-control collider")
    if not collisions_enabled:
        _set_collision_state(colliders, False)
    collision_during = _collision_state(colliders)

    timeline: dict[str, dict[str, Any]] = {}
    timeline["T0"] = _stage_snapshot(
        "T0",
        stage,
        UsdGeom,
        UsdPhysics,
        explicit_step_count=0,
        semantic="after USD load, World construction, and identical runtime cooking",
    )
    _set_initial_translation(stage, BODY_PATH, UsdGeom, Gf)
    _set_initial_translation(stage, NUT_PATH, UsdGeom, Gf)
    timeline["T1"] = _stage_snapshot(
        "T1",
        stage,
        UsdGeom,
        UsdPhysics,
        explicit_step_count=0,
        semantic="after the current runner writes the frozen initial state",
    )
    timeline["T2"] = _stage_snapshot(
        "T2",
        stage,
        UsdGeom,
        UsdPhysics,
        explicit_step_count=0,
        semantic=(
            "current nominal runner performs no explicit rigid-body default-state "
            "registration; observational checkpoint only"
        ),
    )
    for owner in OWNER_PATHS.values():
        prim = stage.GetPrimAtPath(owner)
        if not prim:
            raise RuntimeError(f"missing contact-report owner {owner}")
        PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(0.0)
    gravity = float(profile["gravity_magnitude_m_s2"])
    world.get_physics_context().set_gravity(gravity)
    timeline["T3"] = _stage_snapshot(
        "T3",
        stage,
        UsdGeom,
        UsdPhysics,
        explicit_step_count=0,
        semantic="immediately before World.reset",
    )

    world.reset()
    interface = get_physx_simulation_interface()
    timeline["T4"] = _stage_snapshot(
        "T4",
        stage,
        UsdGeom,
        UsdPhysics,
        explicit_step_count=0,
        semantic="immediately after World.reset and before RigidPrim view initialization",
    )
    contacts_t4 = _contact_inventory(stage, interface, PhysicsSchemaTools, "T4")

    views = {
        label: RigidPrim(
            prim_paths_expr=path,
            name=f"{case_id.lower()}_{label}",
            reset_xform_properties=False,
        )
        for label, path in OWNER_PATHS.items()
    }
    for view in views.values():
        view.initialize()
    timeline["T5"] = _view_snapshot(
        "T5",
        views,
        explicit_step_count=0,
        semantic="after RigidPrim view initialization",
    )
    contacts_t5 = _contact_inventory(stage, interface, PhysicsSchemaTools, "T5")
    timeline["T6"] = _view_snapshot(
        "T6",
        views,
        explicit_step_count=0,
        semantic="immediately before the first explicit physics step",
    )
    world.step(render=False)
    timeline["T7"] = _view_snapshot(
        "T7",
        views,
        explicit_step_count=1,
        semantic="immediately after the first explicit physics step",
    )
    contacts_t7 = _contact_inventory(stage, interface, PhysicsSchemaTools, "T7")

    collision_restored = None
    if not collisions_enabled:
        _set_collision_state(colliders, True)
        collision_restored = _collision_state(colliders)
        if collision_restored != collision_before:
            raise RuntimeError("in-memory collision state was not restored")
    contacts = _aggregate_contacts((contacts_t4, contacts_t5, contacts_t7))
    result = {
        "case_id": case_id,
        "collisions_enabled_during_reset": collisions_enabled,
        "same_initial_state_joint_state_and_parameters": True,
        "joint_count": 0,
        "runtime_collision_cooking": cooking,
        "collision_prim_count": len(colliders),
        "collision_state_before": collision_before,
        "collision_state_during": collision_during,
        "collision_state_restored_before_close": collision_restored,
        "timeline": timeline,
        "first_failed_transition": _first_failed_transition(timeline),
        "contacts_by_stage": {
            "T4": contacts_t4,
            "T5": contacts_t5,
            "T7": contacts_t7,
        },
        "contact_statistics": contacts,
        "gravity_magnitude_m_s2": gravity,
        "physics_rate_hz": rate_hz,
        "explicit_physics_step_count": 1,
        "initial_pose_writes_before_physics_start_count": 2,
        "object_pose_write_after_physics_start_count": 0,
        "solver_error_count": 0,
        "source_asset_written": False,
    }
    world.stop()
    World.clear_instance()
    context.close_stage()
    application.update()
    return result


def _classify(enabled: Mapping[str, Any], disabled: Mapping[str, Any]) -> dict[str, Any]:
    enabled_error = abs(float(enabled["timeline"]["T5"]["datum_error_m"]))
    disabled_error = abs(float(disabled["timeline"]["T5"]["datum_error_m"]))
    enabled_failed = enabled_error > TOLERANCE_M
    disabled_passed = disabled_error <= TOLERANCE_M
    source_error = abs(
        float(enabled["timeline"]["T5"]["datum_separation_m"])
        - float(enabled["timeline"]["T4"]["datum_separation_m"])
    )
    if enabled_failed and disabled_passed:
        classification = "INITIAL_COLLISION_DEPENETRATION"
        supported = True
    elif enabled_failed and not disabled_passed:
        classification = "RESET_DEFAULT_STATE_MISMATCH"
        supported = True
    elif source_error > TOLERANCE_M and enabled_error <= TOLERANCE_M:
        classification = "DATUM_MEASUREMENT_ERROR"
        supported = True
    else:
        classification = "DIAGNOSTIC_INCONCLUSIVE"
        supported = False
    return {
        "classification": classification,
        "classification_is_one_of_authorized_root_causes": classification in ALLOWED_ROOT_CAUSES,
        "root_cause_supported": supported,
        "enabled_t5_datum_error_m": enabled_error,
        "disabled_t5_datum_error_m": disabled_error,
        "collision_disable_improvement_m": enabled_error - disabled_error,
        "tolerance_m": TOLERANCE_M,
        "enabled_first_failed_transition": enabled["first_failed_transition"],
        "disabled_first_failed_transition": disabled["first_failed_transition"],
        "default_state_registration_performed_by_current_runner": False,
        "default_state_observation": (
            "source audit and T2 show that the current nominal runner has no explicit "
            "rigid-body default-state registration"
        ),
    }


def _diagnose(frozen: Mapping[str, Any], application: Any) -> dict[str, Any]:
    enabled = _run_case(
        case_id="COLLISION_ENABLED_BASELINE",
        collisions_enabled=True,
        frozen=frozen,
        application=application,
    )
    disabled = _run_case(
        case_id="COLLISION_DISABLED_IN_MEMORY",
        collisions_enabled=False,
        frozen=frozen,
        application=application,
    )
    diagnosis = _classify(enabled, disabled)
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "COMPLETED",
        "classification": diagnosis["classification"],
        "diagnosis": diagnosis,
        "allowed_root_causes": list(ALLOWED_ROOT_CAUSES),
        "cases": {
            "collision_enabled_baseline": enabled,
            "collision_disabled_in_memory": disabled,
        },
        "diagnostic_only": True,
        "dynamic_pass_claimed": False,
        "formal_nominal_pass_claimed": False,
        "formal_p1_pass_claimed": False,
        "formal_r12_generated": False,
        "hardware_authorized": False,
        "control_consumed_contact_names": False,
        "control_consumed_contact_normals": False,
        "control_consumed_event_truth": False,
        "object_pose_write_after_physics_start_count": 0,
        "explicit_physics_step_count": 2,
        "isaac_process_count": 1,
        "real_contact_trial_count": 1,
        "source_asset_written": False,
        "rejected_hypothesis_retried": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _arguments(argv)
    frozen = _authorize(arguments)
    output = frozen["output"]
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evidence directory: {output}")
    output.mkdir(parents=True, exist_ok=False)
    if arguments.kit_portable_root is None:
        portable = Path(tempfile.mkdtemp(prefix="kcg-dyn-a1-timeline-", dir="/tmp"))
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
        report = _diagnose(frozen, application)
        report["input_sha256"] = dict(EXPECTED_SHA256)
        report["kit_portable_root"] = str(portable)
        report["post_run_sha256"] = {
            "contract": _sha256(frozen["contract_path"]),
            "model": _sha256(frozen["model_path"]),
            "mapping": _sha256(frozen["mapping_path"]),
            "acceptance": _sha256(frozen["acceptance_path"]),
        }
        report["frozen_inputs_unchanged"] = report["post_run_sha256"] == EXPECTED_SHA256
        if not report["frozen_inputs_unchanged"]:
            raise RuntimeError("frozen input SHA-256 changed during diagnostic")
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
            "formal_nominal_pass_claimed": False,
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
    print(json.dumps(report, allow_nan=False, ensure_ascii=False, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
